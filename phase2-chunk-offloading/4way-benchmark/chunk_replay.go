// chunk_replay: replay a captured vLLM ZMQ KV-event stream (capture.jsonl from a
// real multi-turn benchmark run) through the REAL llm-d ingestion path and
// measure how much of the GPU-cached prefix space is also matchable on the CPU
// tier (coverage) and the contiguous CPU match along the longest prefix.
//
// Usage: chunk_replay <capture.jsonl>
// All my logs are prefixed LLMDPROBE.
package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

const (
	blockSize = 16
	modelName = "Qwen/Qwen3-0.6B"
)

type capLine struct {
	Topic      string `json:"topic"`
	Seq        uint64 `json:"seq"`
	PayloadB64 string `json:"payload_b64"`
}

// vLLM batch: [ts, [event_bytes...], dp]
type vbatch struct {
	_      struct{} `msgpack:",array"`
	TS     float64
	Events []msgpack.RawMessage
	DP     *int `msgpack:",omitempty"`
}

func main() {
	log.SetLogger(zap.New(zap.UseDevMode(true)))
	ctx := context.Background()
	if len(os.Args) < 2 {
		fmt.Println("usage: chunk_replay <capture.jsonl>")
		os.Exit(1)
	}
	path := os.Args[1]

	// real llm-d ingestion stack
	index, _ := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 5_000_000, PodCacheSize: 100})
	tp, _ := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: blockSize})
	cfg := kvevents.DefaultConfig()
	cfg.Concurrency = 1
	cfg.DiscoverPods = false
	pool := kvevents.NewPool(cfg, index, tp, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	// Read capture; feed every frame to the real pool (in order), and in
	// parallel decode it ourselves to reconstruct the GPU prefix space.
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 64*1024*1024)

	engToReq := map[uint64]kvblock.BlockHash{} // engine hash -> requestKey (mimics llm-d chaining)
	allReq := map[kvblock.BlockHash]bool{}     // all GPU-cached canonical requestKeys
	var longest []kvblock.BlockHash            // reqKey seq of the largest single GPU store (a long prefix)
	var cpuEventHashes, cpuEventsResolvableNote int
	nframes, nGPU, nCPU, nRem := 0, 0, 0, 0

	for sc.Scan() {
		var cl capLine
		if json.Unmarshal(sc.Bytes(), &cl) != nil {
			continue
		}
		payload, _ := base64.StdEncoding.DecodeString(cl.PayloadB64)
		// feed the REAL pool
		pool.AddTask(&kvevents.RawMessage{Topic: cl.Topic, Sequence: cl.Seq, Payload: payload})
		nframes++

		// our own decode for the GPU prefix-space reconstruction
		var b vbatch
		if msgpack.Unmarshal(payload, &b) != nil {
			continue
		}
		for _, raw := range b.Events {
			var ev []msgpack.RawMessage
			if msgpack.Unmarshal(raw, &ev) != nil || len(ev) < 1 {
				continue
			}
			var tag string
			_ = msgpack.Unmarshal(ev[0], &tag)
			switch tag {
			case "BlockStored":
				var hashes []uint64
				var parent *uint64
				var tokens []uint32
				_ = msgpack.Unmarshal(ev[1], &hashes)
				_ = msgpack.Unmarshal(ev[2], &parent)
				_ = msgpack.Unmarshal(ev[3], &tokens)
				var medium string
				if len(ev) > 6 {
					_ = msgpack.Unmarshal(ev[6], &medium)
				}
				if medium == "CPU" {
					nCPU++
					cpuEventHashes += len(hashes)
					continue // CPU events don't define new GPU prefix space
				}
				nGPU++
				// reconstruct canonical requestKeys exactly like llm-d does
				parentReq := kvblock.EmptyBlockHash
				if parent != nil && *parent != 0 {
					if rk, ok := engToReq[*parent]; ok {
						parentReq = rk
					}
				}
				rks, e := tp.TokensToKVBlockKeys(parentReq, tokens, modelName, nil)
				if e != nil || len(rks) == 0 {
					continue
				}
				// GPU events are 1:1 (block_size=16): map each engine hash to its reqKey
				n := len(rks)
				if len(hashes) < n {
					n = len(hashes)
				}
				for i := 0; i < n; i++ {
					engToReq[hashes[i]] = rks[i]
					allReq[rks[i]] = true
				}
				if len(rks) > len(longest) {
					longest = append([]kvblock.BlockHash(nil), rks...)
				}
			case "BlockRemoved":
				nRem++
			}
		}
	}
	_ = cpuEventsResolvableNote
	time.Sleep(2 * time.Second) // drain the async pool

	// --- coverage: of all GPU-cached canonical blocks, how many are CPU-matchable? ---
	cpuCovered := 0
	reqList := make([]kvblock.BlockHash, 0, len(allReq))
	for rk := range allReq {
		reqList = append(reqList, rk)
	}
	for _, rk := range reqList {
		res, _ := index.Lookup(ctx, []kvblock.BlockHash{rk}, sets.New[string]())
		if hasTier(res[rk], "cpu") {
			cpuCovered++
		}
	}

	// --- contiguous CPU match along the longest prefix (the routing-relevant signal) ---
	contig := 0
	if len(longest) > 0 {
		res, _ := index.Lookup(ctx, longest, sets.New[string]())
		for _, rk := range longest {
			if hasTier(res[rk], "cpu") {
				contig++
			} else {
				break
			}
		}
	}

	cov := 0.0
	if len(allReq) > 0 {
		cov = float64(cpuCovered) / float64(len(allReq))
	}
	fmt.Printf("LLMDPROBE ===== REPLAY %s =====\n", path)
	fmt.Printf("LLMDPROBE frames=%d  GPU_stored_events=%d  CPU_stored_events=%d  removed_events=%d\n",
		nframes, nGPU, nCPU, nRem)
	fmt.Printf("LLMDPROBE total CPU engine-hashes emitted by vLLM = %d\n", cpuEventHashes)
	fmt.Printf("LLMDPROBE GPU-cached canonical blocks (distinct) = %d\n", len(allReq))
	fmt.Printf("LLMDPROBE CPU-matchable canonical blocks         = %d  (coverage = %.1f%%)\n",
		cpuCovered, 100*cov)
	fmt.Printf("LLMDPROBE longest single prefix = %d blocks; contiguous CPU match from start = %d\n",
		len(longest), contig)
	_ = sort.Ints
}

func hasTier(entries []kvblock.PodEntry, tier string) bool {
	for _, e := range entries {
		if e.DeviceTier == tier {
			return true
		}
	}
	return false
}
