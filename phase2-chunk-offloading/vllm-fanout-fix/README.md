# Phase 2 — the fix: vLLM-side chunk fan-out (no router change)

[Step 2](../step2-llmd-cannot-match-chunk.md) showed chunked CPU offloading
(`block_size_factor > 1`) makes the CPU cache **unmatchable** by llm-d (and Dynamo): the CPU event
is a single tail-hash placeholder. This is the fix, and it is **vLLM-only**.

## Idea

Make the chunk's CPU `BlockStored` event **self-describing at block granularity** — i.e. fan the
chunk out into its `hash_block_size_factor` constituent blocks: emit one engine hash per constituent
block + the whole chunk's `token_ids` + `block_size = hash_block_size` (not the offloaded size). The
event then looks exactly like a by-block event, which **both llm-d and Dynamo already consume** — no
router changes. llm-d re-splits `token_ids` at its block size → `factor` request keys, maps the
`factor` engine hashes 1:1, and lights **every** block of the chunk (contiguous match restored).

## The change (vLLM `offloading/scheduler.py`, ~28 lines)

Two edits, both in the scheduler's event-metadata path (full diff: `fanout.patch`):

1. **Relax the guard** in `_build_store_jobs` from "only `block_size_factor == 1 &&
   hash_block_size_factor == 1`" to "any full-attention group":

```python
-   if (block_size_factor == 1
-       and group_config.hash_block_size_factor == 1
-       and group_config.sliding_window_size_in_blocks is None):
+   if group_config.sliding_window_size_in_blocks is None:
        event_meta = self._build_event_metadata(req, group_config, offloaded_block_idx)
        if event_meta is not None:
            self._pending_event_metadata[offload_key] = event_meta
```

2. **Generalize `_build_event_metadata`** to fan a chunk out over `hbf = hash_block_size_factor`
   (for `hbf == 1` this is identical to the old single-block behavior):

```python
   hbf = group_config.hash_block_size_factor
   sub_block_size = group_config.offloaded_block_size // hbf      # = hash/GPU block size
   first = offload_block_idx * hbf                                # chunk c -> hash-blocks [c*hbf, (c+1)*hbf)
   last  = first + hbf
   block_hashes  = req.block_hashes[first:last]                   # factor hashes (was: [tail])
   parent        = req.block_hashes[first-1] if first>0 else None
   token_ids     = req.all_token_ids[c*offloaded : c*offloaded+offloaded]   # whole chunk
   block_size    = sub_block_size                                 # 16 (was: offloaded_block_size)
```

`take_events`/`_take_stored_event` already emit `block_hashes` as a list + `token_ids` + `block_size`,
so no change is needed there.

## Result (real multi-turn benchmark, factor=3, replayed through real llm-d index)

| factor=3 (chunk) | CPU event (real wire) | llm-d CPU coverage | **contiguous CPU match** |
|---|---|---:|---:|
| before (base / token_id patch) | `n_hashes=1, tok=0, block_size=0` | 31.7% | **0/64** |
| **after (fan-out fix)** | `n_hashes=3, tok=48, block_size=16` | **94.4%** | **64/64** |

Same workload/config as Step 2; only the vLLM scheduler changed. The chunk now matches the CPU cache
just like by-block (≈96%, full contiguous prefix). llm-d needed **zero** changes — its existing 1:1
`index.Add` path consumes the fanned-out event; the same self-describing event also unblocks Dynamo's
`create_stored_blocks` (which requires `block_size == kv_block_size` + per-block token slices).

## Scope / follow-ups

- **Full-attention only.** Sliding-window / SSM groups still fall back to the placeholder (excluded by
  the guard) — out of scope here.
- **Removal fan-out (TODO for correctness).** `_take_stored_event` pops the side-table entry at store
  time, so a later chunk `BlockRemoved` falls back to the single tail hash and would under-remove the
  `factor-1` non-tail blocks. Not exercised in these runs (`removed_events=0`), but a complete patch
  should persist a `chunk → constituent hashes` map until removal (clear it in `reset_cache`).
- **One vLLM-side change fixes both routers** (llm-d and Dynamo): same root cause, same fix.

## Reproduce

```bash
# on the bugfix tree, apply fanout_scheduler.py as the scheduler, run factor=3:
cd /home/changg/workspace/.tmp/llmd_4way
bash run_config.sh fanout_chunk fanout 3
cd /home/changg/workspace/llm-d-kv-cache-manager && go build ./examples/chunk_replay
./chunk_replay /home/changg/workspace/.tmp/llmd_4way/runs/fanout_chunk/capture.jsonl
```

Bundled: `fanout_scheduler.py` (full file), `fanout.patch` (diff vs the token_id patch).
