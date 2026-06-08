# Step 4: Dynamo Lower-Tier Index

## Goal

Step 1 verified that vLLM emits chunk-level CPU offload events with only
`[chunk_hash, medium=CPU]` (parent hash, token IDs, and length all stripped).
Steps 2 and 3 showed that llm-d cannot match these chunk events without help,
and that vLLM's PR #43468 partially fixes the gap by snapshotting metadata in a
scheduler-side side table.

This step studies the **routing-side** consumer: Dynamo's KV indexer. We focus
on the lower-tier index (`LowerTierIndexer`) because that is the structure that
must absorb the chunk events. We also document how a query result from the
device-tier (`RadixTree` / `ConcurrentRadixTree`) is handed off into the
lower-tier walk, since this handoff determines what information a chunk event
must carry to be matchable.

The output of this step is a list of constraints (§9) that Step 5 will use to
pick the minimum-mutation integration plan.

---

## 1. Module layout

All paths relative to `lib/kv-router/src/indexer/`:

| File | Role |
|---|---|
| `radix_tree.rs` | Single-threaded device-tier index. `Rc<RefCell<RadixBlock>>`. |
| `concurrent_radix_tree.rs` | Thread-safe device-tier index. `Arc<RwLock<Block>>` plus per-worker reverse lookup. |
| `branch_sharded.rs` | Branch-Sharded Indexer (BSI). Default device-tier wrapper since v1.2.0; routing TRIE on top of N CRTC shards. |
| `positional.rs` | Alternative flat device-tier index. Compares engine `seq_hash` on the read path; not the default with vLLM. |
| `lower_tier.rs` | Lower-tier (CPU / Disk / External) continuation index. **Subject of this study.** |
| `lower_tier_indexers.rs` | Per-tier registry: one `LowerTierIndexer` per non-device `StorageTier`, plus the cross-tier query helper. |
| `thread_pool.rs` | Generic `SyncIndexer` wrapper that dispatches events to `N` OS threads via sticky-by-`worker_id` routing. |

The `Indexer` enum in `lib/llm/src/kv_router.rs` selects which device-tier
backend to instantiate; the lower-tier registry is held alongside it and is
populated lazily on the first non-device event for each `StorageTier`.

---

## 2. Lower-tier index data structures

```rust
// lib/kv-router/src/indexer/lower_tier.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    parent_hash: Option<ExternalSequenceBlockHash>,   // engine seq_hash
    local_hash: LocalBlockHash,                        // Dynamo-recomputed (xxh3, seed=1337)
}

enum EdgeOwnersEntry {
    Single { child_hash: ExternalSequenceBlockHash, owner: WorkerWithDpRank },
    Multi  { child_hash: ExternalSequenceBlockHash, owners: WorkerSet },
}

pub struct LowerTierIndexer {
    edges: DashMap<TransitionKey, EdgeOwnersEntry, FxBuildHasher>,
}

// Held thread-local in `SyncIndexer::worker`, NOT inside LowerTierIndexer.
type WorkerBlockIndex = FxHashMap<
    WorkerWithDpRank,
    FxHashMap<ExternalSequenceBlockHash, TransitionKey>,
>;
```

The same logical state is exposed in two views:

- **Forward (`edges`, shared `DashMap`)**:
  `(parent_seq_hash, local_hash) -> (child_seq_hash, worker set)`.
  This is what `find_matches` walks.
- **Reverse (`worker_blocks`, write-thread-local)**:
  `worker -> block_hash -> TransitionKey`.
  This is what remove events use to translate a `block_hash` back to the
  forward-table key that owns it.

`worker_blocks` is held in `SyncIndexer::worker` as plain thread-local state,
not as a `DashMap`. This is safe because `ThreadPoolIndexer<S>` routes every
event for a given `worker_id` to the same OS thread (sticky-by-worker), so the
reverse table is touched only by one thread.

`EdgeOwnersEntry::Multi` indicates *the same `child_hash` with multiple
owners*, not multiple distinct `child_hash`es. `EdgeOwnersEntry::insert`
returns `false` when called with a different `child_hash` than the one already
stored:

```rust
// lib/kv-router/src/indexer/lower_tier.rs (insert)
match self {
    Self::Single { child_hash: existing_hash, owner: existing_owner } => {
        if *existing_hash != child_hash {
            return false;                  // different child_hash → reject
        }
        ...
    }
    Self::Multi { child_hash: existing_hash, owners } => {
        if *existing_hash != child_hash {
            return false;                  // ditto
        }
        ...
    }
}
```

`store_blocks_impl` propagates the rejection by breaking out of the chain loop:

```rust
// lib/kv-router/src/indexer/lower_tier.rs (store_blocks_impl)
let inserted = match self.edges.entry(key) {
    Entry::Occupied(mut edge) => edge.get_mut().insert(block.block_hash, worker),
    Entry::Vacant(edge) => {
        edge.insert(EdgeOwnersEntry::new(block.block_hash, worker));
        true
    }
};
if !inserted {
    break;                                  // stop installing the rest of the chain
}
```

Implication: the lower-tier index is built around the assumption that each
`(parent_seq_hash, local_hash)` slot has exactly one continuation. Step 5 must
respect this when synthesizing edges from chunk events.

---

## 3. Why parent is in the key

The device-tier indexers and the lower-tier indexer both face the same
disambiguation problem: two blocks with the same token content (and therefore
the same `LocalBlockHash`) can appear in different sequences with different
prefixes. They must be tracked separately so that prefix matching does not
conflate them.

| Indexer | Disambiguation mechanism |
|---|---|
| `RadixTree` / `ConcurrentRadixTree` / `BranchShardedIndexer` | Tree topology. Children are keyed by `LocalBlockHash` only; the same `LocalBlockHash` under different parents lives at different nodes because the parents are different `Rc`/`Arc` pointers. |
| `LowerTierIndexer` | Explicit compound key `(parent_seq_hash, local_hash)`. There is no tree, so the parent must enter the key directly. |
| `PositionalIndexer` (alternative device-tier) | Compound key `(position, local_hash)` plus a stored `seq_hash` per slot for collision disambiguation. |

For the lower-tier index, the parent in the key is the **engine's
`ExternalSequenceBlockHash`** (the rolling hash from the inference engine),
not Dynamo's local hash. This is critical, because:

- Each adjacent block pair in any actually-stored sequence produces a unique
  `(parent, local)` even if many blocks share the same `local_hash`. The
  engine's rolling hash absorbs the prefix history.
- Synthetic prefixes (collisions, repeated tokens, agentic prompt
  boilerplate) do not collide in the index unless two blocks in the same
  sequence share *both* the previous block's seq_hash and the current block's
  local_hash, which essentially never happens in real engine output.

Step 5 will need to preserve this property when handling chunk events: any
proposal that uses a synthetic value (such as `chunk_hash`) as the parent for
multiple consecutive blocks loses uniqueness and re-introduces the very
collision problem the chain design was built to avoid.

---

## 4. Engine seq_hash semantics

Dynamo treats `ExternalSequenceBlockHash` as an **opaque pointer**. It is
never computed in the router process; it flows verbatim from the engine
through the publisher into the index, and back out as scoring identity:

1. Engine emits `BlockStored { block_hashes, parent_block_hash, token_ids, ... }`
   over ZMQ. `block_hashes` and `parent_block_hash` are engine-internal rolling
   hashes whose algorithm is not assumed to be known.
2. Dynamo's relay (`lib/llm/src/kv_router/publisher.rs`) recomputes only
   `LocalBlockHash` from `token_ids` using the canonical Dynamo hasher
   (`compute_block_hash_for_seq`, xxh3 with seed `1337`). `block_hash` and
   `parent_block_hash` are forwarded unchanged into the
   `KvCacheStoredBlockData` struct.
3. The device-tier indexer stores the engine's seq_hash on each node
   (`RadixBlock.block_hash: Option<ExternalSequenceBlockHash>`). The lower-tier
   indexer stores it twice: as `TransitionKey.parent_hash` (the parent
   pointer) and as `EdgeOwnersEntry.child_hash` (what the next walk step's
   parent should become).
4. When `find_matches` walks the lower-tier edges, the next iteration's parent
   is read from `EdgeOwnersEntry.child_hash` and used to look up the next
   `TransitionKey`. Dynamo never reconstructs this value; it consumes whatever
   the engine wrote.

Consequence: any router-side code path that requires a *specific* algorithm
for the engine's rolling hash is incompatible with this design. The only
device-tier index that violates this property is `PositionalIndexer`, which
recomputes `seq_hash` on the read path and therefore demands the publisher use
Dynamo's canonical recurrence; this is also why `PositionalIndexer` is not the
default with vLLM.

For the lower-tier index, the opacity is total. Any chunk event format can be
absorbed as long as the values it carries can be inserted into and looked up
from the existing forward and reverse tables.

---

## 5. Write path

```rust
// lib/kv-router/src/indexer/lower_tier.rs (store_blocks_impl, simplified)

fn store_blocks_impl(
    &self,
    worker_blocks: &mut WorkerBlockIndex,
    worker: WorkerWithDpRank,
    store_data: KvCacheStoreData,
) {
    let mut parent_hash = store_data.parent_hash;     // engine value, opaque
    let worker_map = worker_blocks.entry(worker).or_default();

    for block in store_data.blocks {
        let key = TransitionKey {
            parent_hash,
            local_hash: block.tokens_hash,             // Dynamo-recomputed
        };

        let inserted = match self.edges.entry(key) {
            Entry::Occupied(mut edge) => edge.get_mut().insert(block.block_hash, worker),
            Entry::Vacant(edge) => {
                edge.insert(EdgeOwnersEntry::new(block.block_hash, worker));
                true
            }
        };
        if !inserted {
            break;
        }

        worker_map.insert(block.block_hash, key);
        parent_hash = Some(block.block_hash);          // roll forward
    }
}
```

For each block in the batch:

1. Build `TransitionKey` from the rolling parent (engine's seq_hash) and the
   block's `tokens_hash` (Dynamo's recomputed `LocalBlockHash`).
2. Insert into `edges`. If the slot already exists with a *different*
   `child_hash`, the insert is rejected and the loop breaks.
3. On success, record the reverse mapping `worker_map[block.block_hash] = key`
   so a future remove can reach the forward entry.
4. Roll the parent forward: the just-inserted block's seq_hash becomes the
   parent for the next block in the batch.

The walk forward is what makes each `TransitionKey` unique even under
identical `tokens_hash`: the parent is the previous block's seq_hash, which
itself is unique to the actual sequence the engine produced.

---

## 6. Remove path

Remove events carry only `block_hashes: Vec<ExternalSequenceBlockHash>`, no
parent hashes and no local hashes. The reverse table is the bridge:

```rust
// lib/kv-router/src/indexer/lower_tier.rs (remove_blocks_impl, simplified)

for block_hash in block_hashes {
    let Some(key) = worker_map.remove(block_hash) else {
        return Err(KvCacheEventError::BlockNotFound);
    };
    self.remove_worker_from_edge(key, worker);
}

fn remove_worker_from_edge(&self, key: TransitionKey, worker: WorkerWithDpRank) {
    if let Entry::Occupied(mut edge) = self.edges.entry(key)
        && edge.get_mut().remove(worker)
    {
        edge.remove();
    }
}
```

Two-step lookup:

1. `worker_map.remove(block_hash)` returns the `TransitionKey` recorded at
   store time, or `None` if the block was never installed for this worker.
2. The `TransitionKey` selects the correct `edges` entry, and the worker is
   removed from its owner set. If the entry becomes empty it is purged.

The same two-table pattern (forward map plus per-worker reverse lookup keyed
on engine seq_hash) appears in all three indexers. It exists because the
engine's KV-event protocol identifies a removed block only by its seq_hash,
without re-stating the parent context. Each indexer must therefore record the
"where did I install this" pointer at store time.

---

## 7. RadixTree → LowerTier handoff

A query enters the indexer once, with the request's full
`Vec<LocalBlockHash>`. The device-tier indexer runs first; its result feeds
the lower-tier walk.

### 7.1 Device-tier produces continuation seeds

Device-tier `find_matches` returns `MatchDetails`:

```rust
pub struct MatchDetails {
    pub overlap_scores: OverlapScores,                         // worker -> matched depth
    pub last_matched_hashes: HashMap<WorkerWithDpRank,
                                     ExternalSequenceBlockHash>,
    ...
}
```

`last_matched_hashes[worker]` is the engine's seq_hash of the deepest block at
which `worker` was still on the matching path. It is read directly from the
`block_hash` field of the radix-tree node where the worker dropped out (or
where the query ran out of tokens). This is the same opaque engine value the
lower-tier walk needs as a starting parent.

### 7.2 Cross-tier walk

```rust
// lib/kv-router/src/indexer/lower_tier_indexers.rs (query_lower_tiers, simplified)

pub fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
) -> HashMap<StorageTier, LowerTierMatchDetails> {

    // Seed continuations from device-tier matches.
    let mut continuations = HashMap::new();
    for (worker, matched_blocks) in &device_matches.overlap_scores.scores {
        let last_hash = device_matches.last_matched_hashes.get(worker).copied();
        continuations.insert(
            *worker,
            LowerTierContinuation::new(*matched_blocks as usize, last_hash),
        );
    }

    let mut lower_tier_matches = HashMap::new();
    for storage_tier in [StorageTier::HostPinned, StorageTier::Disk, StorageTier::External] {
        let Some(indexer) = indexers.get(storage_tier) else { continue };

        // Workers that have a root entry on this tier but no device hit can
        // still enter via a from_root continuation.
        if let Some(&first_hash) = sequence.first() {
            for worker in indexer.backend().root_workers(first_hash) {
                continuations.entry(worker)
                    .or_insert_with(|| LowerTierContinuation::from_root(0));
            }
        }

        let tier_matches = indexer.backend()
            .query_match_details(sequence, &continuations);
        continuations = tier_matches.next_continuations.clone();
        lower_tier_matches.insert(storage_tier, tier_matches);
    }

    lower_tier_matches
}
```

The key abstraction is `LowerTierContinuation`:

```rust
pub struct LowerTierContinuation {
    pub start_pos: usize,
    pub last_matched_hash: Option<ExternalSequenceBlockHash>,
}
```

Each worker carries `(position_so_far, last_matched_engine_seq_hash)` from one
tier to the next. The `query_match_details` call on each tier walks edges
forward from `(last_matched_hash, sequence[start_pos])`, advancing
`last_matched_hash` to the next edge's `child_hash` on every successful step,
and emits a fresh continuation map that the next tier in the chain will
consume.

`lower_tier_query_order()` is fixed: HostPinned → Disk → External. Each tier
extends the previous one. A worker that matched 12 blocks on device, plus 4
more on host, enters disk with `start_pos=16` and the engine seq_hash of its
16th matched block.

### 7.3 What the handoff requires from each side

For the handoff to work correctly:

- **Device-tier output**: every worker in `overlap_scores.scores` must have a
  matching entry in `last_matched_hashes`. `RadixTree`/`ConcurrentRadixTree`
  populate this from the `block_hash` field of the node at which the worker
  drained.
- **Lower-tier index state**: must hold an edge keyed by `(last_matched_hash,
  sequence[start_pos])` whose owner set contains the worker, otherwise the
  walk does not advance even if the worker physically holds the block on the
  lower tier.
- **Tier ordering**: lower tiers walk in series, threading `next_continuations`
  through. A worker that drains on host at depth 18 cannot re-enter on disk at
  depth 16; it can only continue at depth 18 or be excluded from the disk walk
  entirely. The `from_root` path catches workers that have a tier-local root
  entry without a matching device prefix, but only at `start_pos=0`.

The hard requirement here is the second one. Any chunk-event handling scheme
proposed in Step 5 must arrange for the lower-tier index to contain an edge
that the cross-tier walk will actually probe. The walk's first probe key for
a worker that just left the device tier at depth `D` with engine seq_hash `M`
is `(M, sequence[D])`. If that edge is missing, the worker contributes zero
to the lower-tier score regardless of what it physically holds.

---

## 8. Multi-tier scoring

The scheduler combines per-tier overlap counts via the routing cost function
documented in `docs/components/router/routing-concepts.md`:

```
adjusted_prefill = max(
    raw_prefill_blocks
    - overlap_score_credit  * device_overlap_blocks
    - host_cache_hit_weight * host_overlap_blocks
    - disk_cache_hit_weight * disk_overlap_blocks
    - shared_cache_multiplier * shared_beyond_blocks,
    0,
)
cost = prefill_load_scale * adjusted_prefill + decode_blocks
```

`overlap_score_credit` defaults to `1.0`. Lower tiers carry smaller weights
that reflect the cost of loading their data back to GPU. The scheduler picks
the worker with the lowest cost, optionally with softmax sampling at non-zero
`router_temperature`.

Two implications:

- **Match depth granularity matters.** The cost function consumes block
  counts, not chunk counts. A scheme that credits one block per chunk on the
  CPU tier, where the chunk actually contains four blocks, weakens the
  cache-hit signal by 4× on that tier.
- **Continuity across tiers matters.** A worker that drops out at device
  depth `D` and cannot re-enter on host at depth `D` because the lower-tier
  edge is missing receives zero host credit even if it stores the next 30
  blocks on host. Step 5 must produce edges that the cross-tier walk in §7.2
  will actually traverse.

---

## 9. Constraints chunk-mode integration must satisfy

Distilled from §§2-8, any plan in Step 5 must obey the following:

1. **Per-block resolution on lower tiers.** The forward index uses
   `(parent_seq_hash, local_hash)` keys per block; the cost function multiplies
   by per-block counts. Crediting one synthetic edge per chunk underweights
   the cache-hit signal in proportion to the chunk size.

2. **Engine-seq-hash opacity.** The router does not know vLLM's rolling-hash
   algorithm and must not require it. All seq_hash values in the lower-tier
   index, including any synthesized for chunk events, must originate from
   engine events.

3. **Cross-tier walk continuity.** For a worker that leaves the device tier
   at depth `D` with engine seq_hash `M`, the lower-tier index must hold an
   edge whose `TransitionKey.parent_hash == M` and `TransitionKey.local_hash ==
   query.local_hashes[D]`, with the worker in the owner set. Without this
   edge, the worker contributes zero on the lower tier.

4. **Unique `TransitionKey` per stored block.** The forward table rejects
   different `child_hash` values under the same `(parent, local)` slot, and
   the chain-write loop breaks on rejection. Any synthetic key derivation
   that produces colliding slots silently truncates the chain.

5. **Reverse-table coverage.** Remove paths translate `block_hash` to
   `TransitionKey` via the per-worker reverse table. Any block that exists in
   the forward table without a matching reverse entry is unreachable for
   removal and will leak. Any block in the reverse table without a forward
   entry will produce a no-op edge remove and a dangling reference.

6. **Async ordering tolerance.** GPU `BlockRemoved(X)` may be observed before
   CPU `BlockStored(X)` because the CPU offload DMA is asynchronous. Any
   cross-tier translation state Dynamo holds for `X` must survive long enough
   for the late CPU event to find it.

7. **Lifecycle bound.** Whatever auxiliary state is introduced must be
   bounded by cluster-wide KV capacity (i.e., grow with the number of
   currently-cached blocks, not with the cumulative event count over time).

These seven constraints are the input to Step 5's plan evaluation.

---

## Code reference index

- `lib/kv-router/src/indexer/lower_tier.rs`:
  `TransitionKey`, `EdgeOwnersEntry`, `LowerTierIndexer`,
  `LowerTierContinuation`, `LowerTierMatchDetails`,
  `store_blocks_impl`, `remove_blocks_impl`, `remove_worker_from_edge`,
  `root_workers`, `query_match_details`.
- `lib/kv-router/src/indexer/lower_tier_indexers.rs`:
  `LowerTierIndexers`, `lower_tier_query_order`, `query_lower_tiers`,
  `TieredMatchDetails`.
- `lib/kv-router/src/indexer/concurrent_radix_tree.rs`:
  `Block.block_hash` (engine seq_hash on the node),
  `find_matches_impl` (populates `last_matched_hashes`).
- `lib/kv-router/src/indexer/radix_tree.rs`:
  `RadixBlock.block_hash`, `find_match_details`.
- `lib/kv-router/src/protocols.rs`:
  `LocalBlockHash`, `ExternalSequenceBlockHash`, `StorageTier`,
  `RouterEvent`, `KvCacheStoreData`, `compute_block_hash_for_seq`,
  `compute_next_seq_hash` (used only by `PositionalIndexer`, not by the
  lower-tier walk).
- `lib/llm/src/kv_router/publisher.rs`:
  `convert_event`, `create_stored_block_from_parts` (relay that recomputes
  `tokens_hash` from raw token IDs while forwarding `block_hash`
  unchanged).
- `lib/llm/src/kv_router.rs`:
  `find_best_match_details` (the entry point that calls `query_tiered_matches`
  and feeds the cost function).
