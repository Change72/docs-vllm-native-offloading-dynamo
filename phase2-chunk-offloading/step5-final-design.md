# Phase 2 · Step 5 — Final design: expose a chunk as N per-block events (Plan B)

> **Step 1** established what a chunk is (one `OffloadKey` = `factor` GPU blocks, keyed by the tail
> hash). **Step 2** showed upstream chunk events break llm-d's CPU matching (68/68 → 0/68).
> **Step 3** prototyped *Plan A* — a single chunk-tail hash + whole-chunk tokens + per-block
> `block_size`, with the router doing a 1:many expansion — and proved it works **on llm-d**.
> **Step 4** studied the *other* router, Dynamo, and found native CPU offload lands in the
> `LowerTierIndexer`, whose `(parent_seq_hash, local_hash) → child_seq_hash` chain model makes Plan A
> a poor fit; it ended with seven constraints (§9) any integration must satisfy.
> **Step 5 (this doc):** pick the minimum-mutation plan against those constraints. The answer is
> **Plan B — vLLM emits the chunk as its `factor` constituent per-block hashes + the whole-chunk
> tokens**, which both routers consume **with zero router-side changes**.

## TL;DR

Expose a chunk on the wire as **`factor` ordinary per-block blocks**:

- **Store**: one `BlockStored` carrying `block_hashes = [h₀ … h_{N-1}]` (the chunk's constituent GPU
  block hashes), `token_ids` = the whole chunk, `block_size` = the per-block token count.
- **Remove**: one `BlockRemoved` carrying the same `[h₀ … h_{N-1}]`.

To both routers this is indistinguishable from `factor` normal by-block events, so **llm-d and
Dynamo need no changes**. The entire cost is vLLM-side: emit the N hashes, and keep a side table
(`OffloadKey → [h₀ … h_{N-1}]`) alive from store until remove so the remove event can fan out.

This is the "Plan B" of [`chunk-event-exposure-plans.md`](chunk-event-exposure-plans.md); Step 4 is
what flips the decision from Plan A to Plan B.

## Why Plan B, not Plan A (mapped to Step 4 §9)

Plan A (single tail hash, router expands 1:many) is clean on llm-d — it reuses
`engineToRequestKeys` — but it violates the lower-tier constraints on Dynamo. Plan B satisfies all
seven:

| # (Step 4 §9) | Constraint | Plan A (single hash) | **Plan B (N hashes)** |
|---|---|---|---|
| 1 | Per-block resolution | 1 edge/chunk → underweights cache signal ×`factor`, or needs 1:many | ✅ N real per-block edges → full per-block credit |
| 2 | Engine-seq-hash opacity | would synthesize per-sub-block parents from `chunk_hash` | ✅ every hash (`h₀…h_{N-1}`, parent) is an engine value from the event |
| 3 | Cross-tier walk continuity | device tier hands off its **real** last engine hash `M`; the chunk's `chunk_hash ≠ M` → first probe `(M, local[D])` misses | ✅ chain uses the same real per-block hashes the device tier emits → `(M=h_{D-1}, local[D])` hits |
| 4 | Unique `TransitionKey` | `(chunk_hash, local_i)` reuses one parent for `factor` blocks → degenerate/colliding slots → chain truncates | ✅ `(h_{i-1}, local_i)` — real rolling hashes, unique per block |
| 5 | Reverse-table coverage | one `chunk_hash` → many edges; removal can't reach them all | ✅ each `h_i` → its `TransitionKey`; remove event carries all N |
| 6 | Async ordering (GPU-remove before CPU-store) | depends on GPU-event-built alias being present/retained | ✅ CPU event is self-contained (tokens → local hash); Dynamo holds **no** cross-tier translation state; the store→remove bridge is the vLLM side table |
| 7 | Lifecycle bound | — | ✅ vLLM side table bounded by CPU pool capacity, drained at eviction |

Constraints **3 and 4 are the decisive ones**: the lower-tier chain is built on the engine's real
per-block rolling hashes, and Plan A's single `chunk_hash` cannot stand in for them without breaking
both the device→host seam and the per-slot uniqueness the design relies on.

## How the lower tier consumes Plan B (native, no change)

A chunk arrives as `factor` blocks `[h₀ … h_{N-1}]` (each its own engine hash) + the whole chunk's
tokens. `convert_event` recomputes a per-block `local_hash` (`tokens_hash`) from the tokens — one per
block, `1:1` with the hashes — and the lower-tier write path is **exactly the normal per-block
chain**:

```
store:  parent = event.parent_hash            // = h₋₁, the block before the chunk
        for hᵢ in [h₀ … h_{N-1}]:
            edge[(parent, localᵢ)] = hᵢ        // distinct, real rolling hashes
            worker_map[hᵢ] = (parent, localᵢ)
            parent = hᵢ                         // roll forward

match:  worker leaves device at depth D, engine hash M = h_{D-1}
        probe (M, local[D]) → h_D → (h_D, local[D+1]) → h_{D+1} → …   // full per-block walk

remove: BlockRemoved[h₀ … h_{N-1}]  →  each worker_map[hᵢ] → edge → drop owner
```

The lower tier literally cannot tell the blocks came from a chunk; it sees an ordinary sequence and
credits every block. llm-d is the same: `index.Add(N engineKeys, N requestKeys)` → the `equal` (1:1)
branch, and removal evicts each hash. No 1:many anywhere.

## vLLM side (the only thing that changes)

This restores the Step-3 prototype's fan-out shape, with the side-table lifecycle you specified:

1. **Populate** (`_build_event_metadata`, which already holds the `Request`): snapshot
   `block_hashes = req.block_hashes[first_hash_idx : last_hash_idx]` (the `factor` constituent
   hashes) instead of the single tail. `token_ids` = whole chunk, `block_size` = per-block size,
   `parent_block_hash` = the block before the chunk. `factor == 1` is unchanged (one hash, one block).
2. **Store event** (`_take_stored_event`): emit `BlockStored([h₀…h_{N-1}], tokens, block_size)` and
   **do not pop** the side-table entry — it must survive for the remove.
3. **Remove event** (`_take_removed_event`): the manager hands back only the chunk's `OffloadKey`;
   look up the saved `[h₀…h_{N-1}]`, emit `BlockRemoved([h₀…h_{N-1}])`, then **pop** the entry.
4. **Bound / cleanup**: the side table is keyed by `OffloadKey`, sized `O(#cpu_chunks × factor)`,
   drained on eviction and cleared by `reset_cache` (Step 4 constraint 7).

Two points that are load-bearing:

- **`token_ids` must be the real chunk tokens, not empty.** Dynamo's lower tier derives each block's
  `local_hash` from the tokens; an empty / `block_size=0` payload is dropped. (llm-d could resolve N
  hashes via its GPU-event alias, but Dynamo's lower tier has no such alias and *needs* the tokens.)
- **The N hashes come from the populate path, not a separate GPU hook.** They are already in
  `req.block_hashes`; the side table only needs to keep them alive until the remove event (when the
  request is long gone) — that is the table's sole reason to exist.

## Cost / trade-off

| | Plan A (single hash, router 1:many) | **Plan B (N hashes, vLLM fan-out)** |
|---|---|---|
| vLLM | minimal | emit N hashes + side table (store→remove) |
| llm-d | reuse 1:many | no change (native 1:1) |
| Dynamo | rework lower-tier chain (breaks `parent_seq_hash`) | **no change** |
| wire / chunk | 1 hash | N hashes (≈16 extra bytes at `factor=3`, negligible vs tokens) |
| cache-hit granularity | per-chunk (×`factor` under-credit) unless 1:many | per-block (exact) |

Plan B moves a small, bounded cost onto vLLM (the side table) and leaves **both** routers and their
lower-tier indexes untouched. Given Dynamo's CPU tier is the lower-tier continuation index, this is
the minimum-mutation plan.

## Status

- **Dynamo**: reverted to zero diff — Plan B requires no router change. The earlier radix-tree edits
  (Step 3 era) were for the *device* tier, which never sees chunks, and have been removed.
- **vLLM**: implement Plan B (N constituent hashes + removal side table) on
  `feature/offloading-connector-chunk-payload`, replacing the Step-3 single-hash payload currently on
  the PR.
- **llm-d**: no change (Step 3 already showed it matches N-hash + token events natively).

## Relationship to Step 3

Step 3's result — that a self-describing chunk event makes llm-d match again — still holds; Plan A and
Plan B both carry tokens and both work on llm-d. Step 5 supersedes Step 3 only on **which payload
shape ships**: once Dynamo (the lower tier) is in scope, exposing the chunk as N per-block hashes
(Plan B) is strictly easier to integrate than the single-hash + 1:many of Plan A.
