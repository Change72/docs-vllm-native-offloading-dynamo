# Step 2 (e2e) — 4-way real multi-turn benchmark: does the token_id patch fix chunk for llm-d?

Real end-to-end run on L4: vLLM **serving** (`vllm serve` + `OffloadingConnector` + ZMQ KV events)
driven by vLLM's own **multi-turn benchmark** (`benchmark_serving_multi_turn.py`, Qwen3-0.6B), with
the real event stream captured and replayed through the **real llm-d index**
(`llm-d-kv-cache-manager`). Four configs:

|        | by-block (`block_size_factor=1`) | chunk (`block_size_factor=3`) |
|--------|---|---|
| **base** (upstream, placeholder payload) | config 1 | config 2 |
| **patched** (your branch, token_id side-table) | config 3 | config 4 |

## TL;DR

| config | vLLM CPU event (real wire) | llm-d CPU coverage | **contiguous CPU match** (68-block prefix) | verdict |
|---|---|---:|---:|---|
| base + by-block      | placeholder, **per-block** (`n_hashes=68, tok=0`) | 95.8% | **68/68** | ✅ works (via alias table) |
| base + chunk         | placeholder, **tail-only** (`n_hashes≈22, tok=0`) | 31.8% | **0/68** | ❌ unmatchable |
| **patched + by-block** | **self-describing** (`n_hashes=1, tok=16`) | 95.8% | **68/68** | ✅ works (self-describing) |
| **patched + chunk**  | **still placeholder** (`n_hashes=1, tok=0`) | 31.7% | **0/68** | ❌ unmatchable |

**Your `token_id` patch fixes by-block but does NOT fix chunk.** In chunk mode the CPU event is still
a single tail-hash placeholder, so llm-d lights only ~1/factor (tail) blocks, non-contiguously — the
contiguous prefix match the router needs is **0**, identical to base. Same root cause as Dynamo;
same fix still needed (populate the chunk event payload / fan out).

## Method

- **vLLM serve** per config: `Qwen/Qwen3-0.6B`, `--enable-prefix-caching`,
  `--gpu-memory-utilization 0.20`, `--max-model-len 4096`,
  `--kv-transfer-config {OffloadingConnector, kv_both, CPUOffloadingSpec[, block_size:48]}`,
  `--kv-events-config {publisher:zmq, endpoint:tcp://*:5557, topic:kv@worker1@…, enable:true}`.
  - **base** scheduler = `59d023619` (newer-main, placeholder); **patched** = `6f5d87288`
    (= same newer-main merged with your `e074f0a53` token_id side-table). Matched pair on one tree.
- **Workload**: `benchmark_serving_multi_turn.py` (`gen_small.json`: 16 conversations, 8–12 turns,
  256-token common prefix, ~800-token per-conv prefix). `--send-conversation-id` left OFF.
- **Capture**: a ZMQ SUB (`capture.py`) records every `[topic, seq, payload]` frame vLLM publishes.
- **Replay**: `chunk_replay.go` feeds the captured frames through the real
  `VLLMAdapter → kvevents.Pool → kvblock.InMemoryIndex`, then measures, over the GPU-cached prefix
  space, how many canonical blocks are CPU-matchable (coverage) and the contiguous CPU run along the
  longest single prefix (the routing-relevant signal). All my logs are `LLMDPROBE`-prefixed.

## What the numbers mean

- **Coverage** = (canonical blocks llm-d can match on the CPU tier) / (canonical blocks vLLM cached).
  by-block ≈ 96% (≈ every offloaded block); chunk ≈ 32% ≈ **1/factor** (only the chunk tail per group).
- **Contiguous CPU match** = how many blocks of a real prefix match consecutively from the start on
  the CPU tier — this is what the prefix-cache scorer actually uses. by-block = full (68/68); chunk =
  **0** because block 0 (a non-tail) has no CPU entry, so the contiguous walk dies immediately even
  though ~1/3 of blocks are lit.
- base vs patched for **by-block**: both full — base relies on the GPU `engineKey→requestKey` alias
  table (CPU event carries no tokens), patched is self-describing (`tok=16`). Same coverage outcome.
- base vs patched for **chunk**: identical failure — the patch's side-table only emits full payload
  when `block_size_factor==1`, so chunk stays a placeholder.

## Raw evidence

vLLM CPU `BlockStored` shapes (decoded from the real captures):

```
base_byblock     CPU store ex: n_hashes=68, tok_len=0,  block_size=0     (placeholder, per-block)
base_chunk       CPU store ex: n_hashes=22, tok_len=0,  block_size=0     (placeholder, tail-only ~68/3)
patched_byblock  CPU store ex: n_hashes=1,  tok_len=16, block_size=16    (self-describing)
patched_chunk    CPU store ex: n_hashes=1,  tok_len=0,  block_size=0     (STILL placeholder)
```

llm-d replay (`replay_results.txt`):

```
base_byblock     GPU-cached blocks=1392  CPU-matchable=1334 (95.8%)  contiguous=68/68
base_chunk       GPU-cached blocks=1392  CPU-matchable=443  (31.8%)  contiguous=0/68
patched_byblock  GPU-cached blocks=1392  CPU-matchable=1334 (95.8%)  contiguous=68/68
patched_chunk    GPU-cached blocks=1383  CPU-matchable=439  (31.7%)  contiguous=0/68
```

## Caveats / fidelity

- Single L4 → single vLLM worker, so this measures **llm-d's CPU-tier index coverage/match** (the
  input to routing), not a multi-worker TTFT delta (routing benefit needs ≥2 workers).
- `removed_events=0` in these runs (GPU didn't evict under 16 convs at util 0.20) — but the chunk
  failure is independent of eviction: block 0 of each chunk is a non-tail and never gets a CPU entry,
  so the contiguous CPU match is 0 regardless.
- Everything else is real: real vLLM serving + real multi-turn benchmark traffic + real llm-d
  ingestion code; the base/patched schedulers are a matched pair on one tree.

## Reproduce

```bash
# vLLM tree must be on the bugfix branch for the matched base/patched pair:
#   (cd /home/changg/workspace/vllm && git checkout bugfix/offloading-connector-blockstored-payload)
# base_scheduler.py = git show 59d023619:.../scheduler.py ; patched = that branch HEAD.
cd /home/changg/workspace/.tmp/llmd_4way
bash run_config.sh base_byblock    base    1
bash run_config.sh base_chunk      base    3
bash run_config.sh patched_byblock patched 1
bash run_config.sh patched_chunk   patched 3
# replay each runs/<name>/capture.jsonl through llm-d:
cd /home/changg/workspace/llm-d-kv-cache-manager && go build ./examples/chunk_replay
./chunk_replay /home/changg/workspace/.tmp/llmd_4way/runs/<name>/capture.jsonl
# restore vLLM branch afterwards:
#   (cd /home/changg/workspace/vllm && git checkout bench/multi_turn-conversation_id-opt-in)
```

Bundled here: `run_config.sh`, `capture.py`, `gen_small.json`, `chunk_replay.go`, `replay_results.txt`.
