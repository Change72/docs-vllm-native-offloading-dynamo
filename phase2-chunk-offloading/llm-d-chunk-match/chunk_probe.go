// chunk_probe: does llm-d's KV indexer match vLLM's native CPU-offload cache
// when vLLM chunking (block_size_factor > 1) is enabled?
//
// It drives the REAL llm-d ingestion path (VLLMAdapter -> kvevents.Pool ->
// kvblock.InMemoryIndex) with vLLM-format msgpack events whose shapes exactly
// match what vLLM's OffloadingConnector emits (verified empirically in phase-2
// step-1): GPU BlockStored carry tokens + per-block hashes; the CPU BlockStored
// is a placeholder (token_ids=[], block_size=0) carrying, per chunk, a SINGLE
// tail hash (= the chunk's last GPU block hash). Then GPU is evicted and we
// query the index for the prefix and measure the contiguous CPU-tier match.
//
// Run: CP_FACTOR=1 go run ./examples/chunk_probe   (and CP_FACTOR=3)
//
// All log lines I added are prefixed "LLMDPROBE". Everything else is llm-d's
// own (zap) output, emitted at TRACE so you can see its internal decisions
// (handleDeviceTierUpdate resolution, Lookup "cutting search", etc.).
package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/vmihailenco/msgpack/v5"
	uberzapcore "go.uber.org/zap/zapcore"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

const (
	blockSize = 16 // vLLM --block-size AND llm-d BlockSizeTokens
	modelName = "test-model"
	podID     = "vllm-pod1"
	topic     = "kv@" + podID + "@" + modelName
)

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// marshalBatch wraps one vLLM event (already an []any) into a vLLM msgpack
// event batch [ts, [event], dp_rank] exactly like ZmqEventPublisher does.
func marshalBatch(event []any) []byte {
	evPayload, _ := msgpack.Marshal(event)
	batch := []any{
		float64(time.Now().UnixNano()) / 1e9,
		[]msgpack.RawMessage{evPayload},
		nil,
	}
	b, _ := msgpack.Marshal(batch)
	return b
}

func main() {
	// TRACE-level zap so llm-d's own debug/trace lines are visible.
	baseLogger := zap.New(zap.UseDevMode(true), zap.Level(uberzapcore.Level(-1*kvTrace)))
	log.SetLogger(baseLogger)
	ctx := log.IntoContext(context.Background(), baseLogger)

	factor := envInt("CP_FACTOR", 1)
	nBlocks := envInt("CP_NBLOCKS", 6)
	if nBlocks%factor != 0 {
		nBlocks += factor - (nBlocks % factor) // round up to whole chunks
	}
	nTokens := nBlocks * blockSize

	fmt.Printf("\nLLMDPROBE ===== factor=%d nBlocks=%d nTokens=%d blockSize=%d =====\n",
		factor, nBlocks, nTokens, blockSize)

	// --- real llm-d ingestion stack ---
	index, err := kvblock.NewInMemoryIndex(nil)
	if err != nil {
		panic(err)
	}
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: blockSize})
	if err != nil {
		panic(err)
	}
	cfg := kvevents.DefaultConfig()
	cfg.Concurrency = 1
	cfg.DiscoverPods = false
	pool := kvevents.NewPool(cfg, index, tp, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)
	defer func() {
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		pool.Shutdown(sctx)
	}()

	// --- synthesize the exact vLLM event shapes ---
	// engine (vLLM) per-GPU-block hashes h[0..nBlocks-1]
	engineHashes := make([]uint64, nBlocks)
	for i := range engineHashes {
		engineHashes[i] = 0xB10C0000 + uint64(i)
	}
	// prefix tokens t[0..nTokens-1]
	tokens := make([]uint32, nTokens)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	// 1) GPU BlockStored — carries tokens + ALL per-block hashes (block_size=16).
	//    This populates engineKey->requestKey for every GPU block (1:1).
	gpuStore := []any{
		"BlockStored", engineHashes, nil, tokens, blockSize, nil, "GPU", nil,
	}
	pool.AddTask(&kvevents.RawMessage{Topic: topic, Payload: marshalBatch(gpuStore)})
	fmt.Printf("LLMDPROBE published GPU BlockStored: %d hashes, %d tokens, block_size=%d\n",
		len(engineHashes), len(tokens), blockSize)

	// 2) CPU BlockStored — placeholder (token_ids=[], block_size=0).
	//    factor=1 -> one hash per block; factor>1 -> ONE tail hash per chunk
	//    (= the chunk's last GPU block hash), which is what vLLM emits.
	var cpuHashes []uint64
	for c := 0; c*factor < nBlocks; c++ {
		tail := c*factor + factor - 1
		cpuHashes = append(cpuHashes, engineHashes[tail])
	}
	cpuStore := []any{
		"BlockStored", cpuHashes, nil, []uint32{}, 0, nil, "CPU", nil,
	}
	pool.AddTask(&kvevents.RawMessage{Topic: topic, Payload: marshalBatch(cpuStore)})
	fmt.Printf("LLMDPROBE published CPU BlockStored (placeholder): %d hash(es) %v, token_ids=[], block_size=0\n",
		len(cpuHashes), hexes(cpuHashes))

	// 3) GPU BlockRemoved — evict GPU tier (the regime where CPU must carry the prefix).
	gpuRemove := []any{"BlockRemoved", engineHashes, "GPU"}
	pool.AddTask(&kvevents.RawMessage{Topic: topic, Payload: marshalBatch(gpuRemove)})
	fmt.Printf("LLMDPROBE published GPU BlockRemoved: %d hashes (GPU tier evicted)\n", len(engineHashes))

	time.Sleep(1500 * time.Millisecond) // let the (async) pool drain

	// --- query: recompute the prefix's canonical request keys and look them up ---
	requestKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, modelName, nil)
	if err != nil {
		panic(err)
	}
	got, err := index.Lookup(ctx, requestKeys, sets.New[string]())
	if err != nil {
		fmt.Printf("LLMDPROBE Lookup error: %v\n", err)
	}

	fmt.Printf("LLMDPROBE --- per-block lookup (after GPU eviction) ---\n")
	cpuLit, contiguousCPU := 0, 0
	contiguousBroken := false
	for i, rk := range requestKeys {
		entries := got[rk]
		hasCPU, hasGPU := false, false
		for _, e := range entries {
			switch e.DeviceTier {
			case "cpu":
				hasCPU = true
			case "gpu":
				hasGPU = true
			}
		}
		if hasCPU {
			cpuLit++
		}
		if hasCPU && !contiguousBroken {
			contiguousCPU++
		} else {
			contiguousBroken = true
		}
		fmt.Printf("LLMDPROBE   block %d reqKey=%s pods=%v cpu=%v gpu=%v\n",
			i, rk.String(), entries, hasCPU, hasGPU)
	}

	fmt.Printf("LLMDPROBE ===== RESULT factor=%d: cpu_lit_blocks=%d/%d  contiguous_cpu_match_from_start=%d/%d  => %s =====\n\n",
		factor, cpuLit, nBlocks, contiguousCPU, nBlocks, verdict(contiguousCPU, nBlocks))
}

func verdict(match, n int) string {
	if match == n {
		return "FULL CPU prefix matched (llm-d can route to CPU cache)"
	}
	if match == 0 {
		return "CPU cache UNMATCHABLE (router credits 0 CPU blocks)"
	}
	return "PARTIAL"
}

func hexes(hs []uint64) []string {
	out := make([]string, len(hs))
	for i, h := range hs {
		out[i] = fmt.Sprintf("0x%x", h)
	}
	return out
}

// mirror of llm-d's logging.TRACE without importing the internal package path twice.
const kvTrace = 5
