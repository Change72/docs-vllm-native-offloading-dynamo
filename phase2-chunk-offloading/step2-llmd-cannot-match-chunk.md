# Phase 2 · Step 2 — upstream chunk offloading silently breaks llm-d's CPU-tier matching

> **Step 1** ([step1-verify-chunk-offload-fact.md](step1-verify-chunk-offload-fact.md)) proved, on
> real vLLM, *what* a chunk is and that the CPU offload event is a self-non-describing placeholder
> when `block_size_factor > 1`.
> **Step 2 (this doc):** prove, end-to-end on real multi-turn traffic, that **upstream/default**
> vLLM chunk offloading makes the CPU cache **unmatchable** by llm-d's router — while by-block
> offloading works.

Scope: **upstream vLLM only** (no fork patches), comparing `block_size_factor = 1` (by-block) vs
`= 3` (chunk). The takeaway is the *default* behavior: turning on `offloaded_block_size > block_size`
silently breaks tier-aware routing.

## TL;DR

| upstream vLLM config | vLLM CPU event (real wire) | llm-d CPU coverage | **contiguous CPU match** (68-block prefix) | verdict |
|---|---|---:|---:|---|
| by-block (`factor=1`) | placeholder, **per-block** (`n_hashes=68, tok=0`) | **95.8%** | **68/68** | ✅ CPU cache matchable |
| chunk (`factor=3`)    | placeholder, **tail-only** (`n_hashes≈22, tok=0`) | **31.8%** | **0/68** | ❌ CPU cache unmatchable |

**Enabling chunked offloading on stock vLLM drops llm-d's CPU match from full to zero.** The chunk
event reports only one *tail* hash per group, so llm-d lights only ~1/`factor` of the blocks, and
because they are non-contiguous the router's prefix match dies at block 0.

## How it was measured (all real)

1. **vLLM serve** (`Qwen/Qwen3-0.6B`, upstream scheduler) with `OffloadingConnector` +
   `--enable-prefix-caching` + ZMQ KV events:
   `--kv-transfer-config {kv_connector:OffloadingConnector, kv_role:kv_both,
   kv_connector_extra_config:{spec_name:CPUOffloadingSpec, cpu_bytes_to_use:…[, block_size:48]}}`
   `--kv-events-config {publisher:zmq, endpoint:tcp://*:5557, topic:kv@…, enable_kv_cache_events:true}`.
   `block_size:48` (= 3×16) gives `factor=3`; omitting it gives `factor=1`.
2. **Workload**: vLLM's own `benchmark_serving_multi_turn.py` (16 conversations, 8–12 turns,
   256-token shared prefix, ~800-token per-conversation prefix).
3. **Capture** every ZMQ KV-event frame vLLM publishes.
4. **Replay** the captured frames through the **real llm-d index**
   (`llm-d-kv-cache-manager`: `VLLMAdapter → kvevents.Pool → kvblock.InMemoryIndex`), then measure
   over the GPU-cached prefix space how many canonical blocks are CPU-matchable (coverage) and the
   contiguous CPU run along the longest single prefix (the routing-relevant signal).

## Why chunk fails (code-grounded)

- vLLM's chunk CPU `BlockStored` carries **one tail hash per chunk** (= the chunk's last GPU block
  hash) with **`token_ids=[]`, `block_size=0`** (see Step 1).
- llm-d cannot recompute keys from tokens (there are none), so it takes the location-only path
  (`pool.go::handleDeviceTierUpdate`): it resolves each emitted hash via the GPU
  `engineKey→requestKey` alias table. A chunk reports only the **tail** hash → only the **tail
  block** of each group gets a CPU entry; the other `factor-1` blocks get none.
- llm-d's lookup (`in_memory.go::Lookup`) is a **contiguous** prefix walk that early-stops at the
  first block with no pods. Block 0 of the first chunk is a non-tail → it has no CPU entry → the
  contiguous CPU match is **0**, even though ~1/`factor` of blocks are lit.
- By-block (`factor=1`) is fine: one hash per block → every offloaded block resolves → full,
  contiguous match.

## Raw evidence

vLLM CPU `BlockStored` shapes (decoded from the real captures):

```
by-block (factor=1)  CPU store ex: n_hashes=68, tok_len=0, block_size=0   (placeholder, per-block)
chunk    (factor=3)  CPU store ex: n_hashes=22, tok_len=0, block_size=0   (placeholder, tail-only ≈68/3)
```

llm-d replay result:

```
by-block (factor=1)  GPU-cached blocks=1392  CPU-matchable=1334 (95.8%)  contiguous=68/68
chunk    (factor=3)  GPU-cached blocks=1392  CPU-matchable=443  (31.8%)  contiguous=0/68
```

## Conclusion

On stock vLLM, **`offloaded_block_size > block_size` (chunked CPU offloading) is incompatible with
llm-d's KV-cache-aware routing**: the offloaded prefix is real and reused *inside* vLLM, but the KV
event only self-describes one tail block per chunk, so llm-d (and any block-granular router) cannot
reconstruct the chunk → it credits ~0 CPU blocks and won't route to the pod holding the prefix on
CPU. The default-safe choice today is `factor=1` (by-block) offloading.

## Caveats

- Single L4 → one vLLM worker, so this measures llm-d's **CPU-tier index coverage/match** (the input
  to routing), not a multi-worker TTFT delta.
- `removed_events=0` in these runs (no GPU eviction at this scale), but the chunk failure is
  eviction-independent: block 0 of each chunk never gets a CPU entry, so contiguous match is 0
  regardless.

## Reproduce

Harness + raw captures: [`4way-benchmark/`](4way-benchmark/) (`run_config.sh`, `capture.py`,
`gen_small.json`, `chunk_replay.go`). Base-only:

```bash
cd /home/changg/workspace/.tmp/llmd_4way
bash run_config.sh base_byblock base 1     # factor=1
bash run_config.sh base_chunk   base 3     # factor=3
cd /home/changg/workspace/llm-d-kv-cache-manager && go build ./examples/chunk_replay
./chunk_replay /home/changg/workspace/.tmp/llmd_4way/runs/base_byblock/capture.jsonl
./chunk_replay /home/changg/workspace/.tmp/llmd_4way/runs/base_chunk/capture.jsonl
```
