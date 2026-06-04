# Step 2 — does llm-d match vLLM's CPU offload cache when chunking is on?

Question: vLLM's native `OffloadingConnector` CPU event only carries a block hash (no tokens). Does
llm-d's KV-router rebuild its index from tokens, and **when vLLM chunking (`block_size_factor > 1`)
is enabled, can llm-d still match the CPU cache — or does it fail?**

## TL;DR (measured, on the real llm-d code path)

| factor | CPU blocks lit up | **contiguous CPU match from start** | verdict |
|---|---|---|---|
| **1** (no chunking) | 6 / 6 | **6 / 6** | FULL — llm-d routes to the CPU cache |
| **3** (chunking)    | 2 / 6 | **0 / 6** | **CPU cache UNMATCHABLE — router credits 0 CPU blocks** |

**With chunking on, llm-d cannot match the offloaded CPU cache.** It is not corrupted — the index
just under-counts to uselessness: only the *tail* block of each chunk is ever marked CPU-resident,
and llm-d's contiguous prefix match breaks at the first non-tail block (block 0), so the router sees
0 CPU hits and won't route to the pod that actually holds the prefix on CPU.

## How llm-d ingests vLLM events (code, `llm-d-kv-cache-manager`)

- **Events WITH tokens (normal GPU events):** llm-d *recomputes its own* canonical block keys from
  `token_ids` (`token_processor.go::TokensToKVBlockKeys`, FNV-64a/CBOR @ its BlockSize) and stores an
  `engineKey(vLLM hash) -> requestKey` alias (`pool.go:382,397`). So yes — **llm-d rebuilds the index
  from tokens**; the vLLM block hash is only an alias.
- **CPU offload event (placeholder, `token_ids=[]`):** `TokensToKVBlockKeys([])` returns nil →
  `pool.go::handleDeviceTierUpdate` (L252) resolves each emitted hash via the *pre-existing*
  `engineKey->requestKey` table (built by the GPU event) and adds a `cpu` PodEntry. **It cannot
  rebuild from tokens (there are none); it depends entirely on the GPU event having mapped that exact
  hash.**
- **Matching (query):** `in_memory.go::Lookup` (L122-127) walks the request's blocks and **early-stops
  at the first block with no pods** ("cutting search") — i.e. contiguous prefix matching.

## Why factor=1 works but factor=3 breaks

- **factor=1:** the CPU event has *one hash per GPU block*. `handleDeviceTierUpdate` resolves all of
  them → **every block gets a CPU PodEntry** → contiguous match = full. (Note it works *despite*
  empty tokens, purely via the GPU-populated alias table + 1:1 hash↔block.)
- **factor=3:** the CPU event has *one tail hash per chunk* (= the chunk's last GPU block hash; this
  is exactly what vLLM emits, verified in step 1). It resolves to only the **tail block's** requestKey
  → only blocks 2 and 5 get a CPU PodEntry. After GPU eviction the non-tail blocks (0,1,3,4) have no
  pods at all, so the contiguous match dies at block 0 → **0 CPU hits**.

## Evidence (L4, `factor1.log` / `factor3.log`)

Harness drives the **real** llm-d path: `engineadapter.VLLMAdapter` (msgpack parse) →
`kvevents.Pool.processEventBatch` → `kvblock.InMemoryIndex`. The published events have the exact
shapes vLLM's `OffloadingConnector` emits (step-1 verified): GPU `BlockStored` (tokens + per-block
hashes); CPU `BlockStored` placeholder (`token_ids=[]`, `block_size=0`, per-chunk tail hash); GPU
`BlockRemoved` (evict). Then we recompute the prefix's canonical keys and `Lookup`.

**My logs are prefixed `LLMDPROBE`; everything else is llm-d's own (zap, `LEVEL(-5)`).**

factor=3, my per-block lookup after eviction (same 6 reqKeys as factor=1 — identical prefix; only the
CPU coverage differs):

```
LLMDPROBE published CPU BlockStored (placeholder): 2 hash(es) [0xb10c0002 0xb10c0005], token_ids=[], block_size=0
LLMDPROBE   block 0 reqKey=9540012594951042066  pods=[]                       cpu=false gpu=false
LLMDPROBE   block 1 reqKey=16921089685996993662 pods=[]                       cpu=false gpu=false
LLMDPROBE   block 2 reqKey=1342399872045281493  pods=[{vllm-pod1 cpu false}]  cpu=true  gpu=false
LLMDPROBE   block 3 reqKey=14412806440506676804 pods=[]                       cpu=false gpu=false
LLMDPROBE   block 4 reqKey=12157956504292006694 pods=[]                       cpu=false gpu=false
LLMDPROBE   block 5 reqKey=1040321297494662211  pods=[{vllm-pod1 cpu false}]  cpu=true  gpu=false
LLMDPROBE ===== RESULT factor=3: cpu_lit_blocks=2/6 contiguous_cpu_match_from_start=0/6 => CPU cache UNMATCHABLE =====
```

llm-d's **own** (built-in) lines confirm the non-tail blocks are dropped after GPU eviction:

```
LEVEL(-5) kvblock.InMemoryIndex.Evict  removed requestKey from index as no pods remain {"requestKey":"9540012594951042066"}   # block 0
LEVEL(-5) kvblock.InMemoryIndex.Evict  removed requestKey from index as no pods remain {"requestKey":"16921089685996993662"}  # block 1
LEVEL(-5) kvblock.InMemoryIndex.Evict  removed requestKey from index as no pods remain {"requestKey":"14412806440506676804"}  # block 3
LEVEL(-5) kvblock.InMemoryIndex.Evict  removed requestKey from index as no pods remain {"requestKey":"12157956504292006694"}  # block 4
```

factor=1 for contrast: all 6 blocks `cpu=true` → `contiguous_cpu_match_from_start=6/6`.

## The fix path already exists in llm-d

`in_memory.go::Add` already supports **1:many** (`1 eng, N req -> E0->[R0..R_{N-1}]`, L157-179). So if
the vLLM chunk CPU event carried `token_ids` (so `TokensToKVBlockKeys` yields the `factor` canonical
keys) + the single chunk hash, llm-d would mark **all** `factor` blocks CPU and matching would be
restored. → **Same root cause and same vLLM-side fix as Dynamo** (populate the chunk event payload);
llm-d's index layer is already built for it.

## Fidelity / scope

- The **llm-d code path is real** (cloned `llm-d-kv-cache-manager`, real adapter+pool+index).
- The **vLLM event shapes are real** — empirically verified in
  [step 1](../step1-verify-chunk-offload-fact.md) on actual vLLM (factor=3 CPU event = single tail
  hash, `token_ids=[]`, `block_size=0`). This harness replays those exact shapes through llm-d.
- Not yet done (optional further validation): a single live process e2e (real vLLM ZMQ →
  go-zeromq subscriber → query). The two real halves above already compose into the full answer.

## Reproduce

```bash
# Go 1.25 at /home/changg/workspace/.local/go ; harness in llm-d-kv-cache-manager/examples/chunk_probe
cd /home/changg/workspace/llm-d-kv-cache-manager
export PATH=/home/changg/workspace/.local/go/bin:$PATH GOTOOLCHAIN=local GOMODCACHE=/home/changg/workspace/.cache/gomod
go build ./examples/chunk_probe
CP_FACTOR=1 ./chunk_probe   # 6/6 contiguous CPU match
CP_FACTOR=3 ./chunk_probe   # 0/6 -> CPU cache unmatchable
```

`chunk_probe.go` (copy here), `factor1.log`, `factor3.log` are bundled in this folder.
Knobs: `CP_FACTOR` (chunk factor), `CP_NBLOCKS` (prefix length in GPU blocks).
