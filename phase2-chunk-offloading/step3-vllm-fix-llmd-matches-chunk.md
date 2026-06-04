# Phase 2 · Step 3 — vLLM self-describes the chunk (full tokens + tail hash) → llm-d matches it (store + remove)

> **Step 1** ([step1-verify-chunk-offload-fact.md](step1-verify-chunk-offload-fact.md)) proved *what*
> a chunk is and that the CPU offload event is a self-non-describing placeholder when
> `block_size_factor > 1`.
> **Step 2** ([step2-llmd-cannot-match-chunk.md](step2-llmd-cannot-match-chunk.md)) proved, on real
> multi-turn traffic, that this drops llm-d's contiguous CPU match from **68/68 → 0/68**.
> **Step 3 (this doc):** fix it at the event boundary — vLLM publishes the chunk as **one
> representative hash (the chunk tail) + the whole chunk's `token_ids` + per-block `block_size`** —
> and show that llm-d's existing **1:many** path now lights every block of the chunk, restoring full
> contiguous match for **both store and remove**.

Scope: the fix is **vLLM-side only**; llm-d is unmodified. Dynamo is intentionally out of scope here
(handled separately) — the same payload feeds it once its indexer is taught the same 1:many mapping.

## TL;DR

| config (`factor=3`, real traffic) | vLLM CPU event (real wire) | llm-d CPU coverage | **contiguous CPU match** (longest prefix) | verdict |
|---|---|---:|---:|---|
| upstream chunk (Step 2) | placeholder, tail-only (`n_hashes≈22, tok=0, bs=0`) | 31.8% | **0/68** | ❌ unmatchable |
| **fixed chunk (this step)** | **`n_hashes=1, tok=48, bs=16`** (tail hash + whole chunk) | **94.3%** | **68/68** | ✅ matchable |
| by-block `factor=1` (reference) | `n_hashes=1, tok=16, bs=16` (per block) | 95.8% | 68/68 | ✅ matchable |

**One representative hash + the chunk's tokens is all llm-d needs.** It re-splits the 48 tokens into
the 3 constituent blocks itself and maps the single engine hash to all 3 (1:many), so the offloaded
chunk is indexed block-granularly and the prefix walk no longer dies at block 0.

## The fix (vLLM `scheduler.py`)

For `block_size_factor > 1`, `_build_event_metadata` now snapshots, per offloaded chunk:

- `block_hash` = **the chunk's tail block hash** = the chunk's `OffloadKey` (a single hash, `<…>`,
  not a per-block list).
- `token_ids` = **the whole chunk** (`factor × block_size` = 48 tokens).
- `block_size` = **the per-block token count** (= the GPU/hash block size, 16) — *not* the
  offloaded chunk size.
- `parent_block_hash` = the block just before the chunk (= the previous chunk's tail hash), so
  chunks chain.

`factor == 1` (by-block) is unchanged: one hash + one block of tokens. The on-wire `BlockStored` /
`BlockRemoved` field stays `list[…]`; we simply emit a length-1 list (same shape the Mooncake
connector and GPU `BlockRemoved` already use).

## Why it works (code-grounded)

**Store.** llm-d ingest (`pool.go::handleDeviceTierUpdate`) takes the token path because the event
now has tokens: `TokensToKVBlockKeys(parentReq, <48 tokens>, …)` splits by `block_size=16` into
**3 canonical requestKeys** (the real per-block keys, LoRA/hash-fn computed by the router). The
event's engineKeys = `<1 tail hash>`. `index.Add(<1 eng>, <3 req>)` hits the documented **1:many**
branch (`in_memory.go`), so `engineToRequestKeys[tail] = <r0, r1, r2>` and **all three blocks get a
CPU PodEntry** — not just the tail.

**Chaining.** Chunk *c*'s `parent` is chunk *c-1*'s tail hash. `GetRequestKey(prev_tail)` resolves
(via the same 1:many map built at store) to the previous chunk's **last** requestKey — exactly the
correct parent for chunk *c*'s first block — so the per-block keys are contiguous across chunks and
the prefix walk runs unbroken.

**Remove.** Eviction publishes a `BlockRemoved` carrying the **same single tail hash** the store
used. llm-d `Evict(tail)` follows `engineToRequestKeys[tail] = <r0, r1, r2>` and removes **all
three** blocks — the 1:many mapping fans the removal out on the router side, so vLLM never re-lists
per-block hashes and needs no removal-side bookkeeping.

## Why this shape (vs. fanning out hashes in vLLM)

Two payloads can restore block-granularity. We chose **single hash + tokens**:

- **Reuses llm-d's existing structure** (`engineToRequestKeys` 1:many) — **no new router state**.
- **vLLM emits 1 hash, not a `factor`-long list** — smaller events, and **removal needs no
  vLLM-side `OffloadKey → hashes` map** (one hash in, router fans out).
- **LoRA / different hash functions are the router's job** — it recomputes the per-block keys from
  the tokens, so vLLM never has to know the router's keying scheme.
- Symmetric and minimal: store and remove carry the identical single tail hash.

## Raw evidence (all real)

vLLM CPU `BlockStored` shape, decoded from the real capture (`runs/v2_chunk/capture.jsonl`):

```
CPU BlockStored: n_hashes=1  tok_len=48  block_size=16  parent=set     (tail hash + whole 3-block chunk)
```

(field layout confirmed: `[1] block_hashes=<1>`, `[3] token_ids=<48>`, `[4] block_size=16`,
`[6] medium=CPU`.)

llm-d replay (real `llm-d-kv-cache-manager` ingest: `VLLMAdapter → kvevents.Pool →
kvblock.InMemoryIndex`), fixed chunk run:

```
frames=342  GPU_stored_events=336  CPU_stored_events=443  removed_events=0
total CPU engine-hashes emitted by vLLM = 443      (== CPU_stored_events ⇒ exactly 1 hash per event)
GPU-cached canonical blocks (distinct)  = 1393
CPU-matchable canonical blocks          = 1314  (coverage = 94.3%)
longest single prefix = 68 blocks;  contiguous CPU match from start = 68
```

Side-by-side with Step 2's upstream chunk (`31.8%`, `0/68`) and by-block (`95.8%`, `68/68`): the fix
moves chunk offloading from unmatchable to on par with by-block, at **1/`factor` the CPU hashes on
the wire** (443 vs 1334).

Unit tests (vLLM), both by-block and chunk, store and remove, all green (57/57 in the file):

```
test_take_events_publishes_routable_block_stored          factor=1  store    (1 hash + 1 block)
test_take_events_parent_chain_continues_across_batches    factor=1  chain
test_take_events_remove_drains_side_table_and_preserves_medium  factor=1  remove
test_take_events_factor_gt_1_single_hash_whole_chunk      factor=3  store    (1 hash + 48 tok + bs=16 + chain)
test_take_events_factor_gt_1_removed_single_hash          factor=3  remove   (1 tail hash per chunk)
```

## Caveats

- Single L4 → one vLLM worker, so this measures llm-d's **CPU-tier index coverage/match** (the
  routing input), not a multi-worker TTFT delta — same scope as Step 2.
- Real traffic had `removed_events=0` (no GPU eviction at this scale). **Remove** is therefore
  verified by (a) the `factor=3` removal unit test and (b) the symmetric-hash mechanism above
  (store and remove emit the identical tail hash; llm-d's 1:many handles the fan-out), not by a
  live eviction count.
- The ~94% (not 100%) coverage matches the by-block reference (95.8%) — the gap is the usual
  in-flight / tail blocks not yet offloaded, not a chunk-specific loss; **contiguous match is the
  full 68/68**, which is what routing keys on.

## Reproduce

vLLM fix lives on branch **`feature/offloading-connector-chunk-payload`** (commit `f3a74482a`,
`vllm/.../offloading/scheduler.py`). Harness + replay (same as Step 2):

```bash
cd /home/changg/workspace/.tmp/llmd_4way
bash run_config.sh v2_chunk v2 3          # factor=3, fixed scheduler
cd /home/changg/workspace/llm-d-kv-cache-manager && go build -o chunk_replay ./examples/chunk_replay
./chunk_replay /home/changg/workspace/.tmp/llmd_4way/runs/v2_chunk/capture.jsonl
```
