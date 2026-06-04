# Phase 2 (chunk offloading) · Step 1 — empirical verification of the chunk fact

> **Phase 2** tackles vLLM's group/chunk offloading (`offloaded_block_size > --block-size`, i.e.
> `block_size_factor > 1`) and how a router should consume it.
> **Step 1 (this doc):** verify, on clean upstream vLLM, *what* a chunk actually is and whether vLLM
> reuses it — establish the ground truth before any design.
> **Step 2 (next):** Dynamo / llm-d comparison of chunk-offload handling + performance.

Goal: before designing any Dynamo change, prove on real L4 hardware, on a **clean / unmodified
upstream vLLM**, what gets offloaded when `block_size_factor > 1`, how a chunk is identified, and
whether vLLM can reuse the chunks. The harness is built to be reusable for the Step 2 **llm-d** A/B.

## TL;DR — the problem is real and measured

With `kv_connector_extra_config["block_size"]` set to a multiple of the GPU block size (factor = 3),
vLLM's `OffloadingConnector`:

1. **Groups `factor` GPU/hash blocks into one CPU "chunk"** — chunk = 48 tokens = 3 × 16-token GPU
   blocks, one `OffloadKey` per chunk.
2. **Identifies the chunk by a single hash = the LAST sub-block's hash** — not a hash of the whole
   chunk, not a list. The first `factor-1` block hashes (and their tokens) are absent from the
   chunk's identity (`chunk_hash_eq_last_block=True` for all 10 chunks).
3. **Emits a placeholder KV event to any router**: `token_ids=[]`, `block_size=0`,
   `parent_block_hash=None`, `block_hashes=<one chunk-tail hash per chunk>`.
4. **Yet vLLM reuses the chunks perfectly** — after the prefix was evicted from GPU, vLLM reloaded
   all 10 chunks (480 tokens) CPU→GPU.

So the offloaded data is real, correct, and reusable *inside vLLM*. The gap is purely at the
**event boundary**: a router (Dynamo today, llm-d for comparison) gets one opaque tail-hash per
chunk with no tokens / no block_size / no parent, so it cannot reconstruct the `factor` blocks the
chunk covers.

## What was run

- **Clean upstream** vLLM merge-base `117afeea4` — the patched `scheduler.py` was swapped to the
  clean file for the run, then restored (`git diff HEAD -- scheduler.py` is empty). Only delta vs
  clean: observational `logger.info("CHUNKPROBE …")` lines (no behavior change), saved as
  `instrumented_clean_scheduler.py`.
- **Model** `facebook/opt-125m` (cached; full-attention → no sliding-window/SSM exclusion).
  **GPU** NVIDIA L4, venv `/home/changg/workspace/.venv-cu130` (torch 2.11.0+cu130).
- **Geometry** (`CHUNKPROBE CONFIG`): `block_size_factor=3`, `hash_block_size_factor=3`,
  `gpu_block_size=16`, `offloaded_block_size=48`, `sliding_window=None`, `num_workers=1`.
- **Driver** `run_offline.py` — pure offline `LLM` + `OffloadingConnector` (`CPUOffloadingSpec`,
  `cpu_bytes_to_use=1GiB`), no Dynamo/NATS/etcd. GPU KV forced tiny (`num_gpu_blocks_override=64`,
  1024 tok) to guarantee eviction. Phases: WARM (one 480-token prefix P → offload 10 chunks) →
  EVICT (8 × 1000 unique tokens → evict P from GPU) → REUSE (P again → reload from CPU).

## Log provenance — mine vs vLLM's

**Every line containing `CHUNKPROBE` is a `logger.info(...)` I added** (no behavior change); any
line without it is stock vLLM. The trap: my lines still carry vLLM's logger prefix, e.g.
`(EngineCore …) INFO … [scheduler.py:604] CHUNKPROBE …` — the `[scheduler.py:NNN]` part is vLLM's,
the payload from `CHUNKPROBE` on is mine.

| field in my logs | source |
|---|---|
| `block_size_factor`, `gpu_block_size`, `offloaded_block_size`, `hash_block_size_factor` | read verbatim from vLLM's `self.config` / `GroupOffloadConfig` |
| `key_hash`, `constituent_block_hashes`, `chunk_hash_eq_last_block` | derived from `offload_key` + `req.block_hashes` |
| `num_hashes`, `hashes`, `token_ids_len`, `block_size`, `parent` (in `EVENT`) | **derived prints** of the real `BlockStored` fields (see EVENT note) |
| `num_chunks_loaded`, `chunk_key_hashes` (in `REUSE/LOAD`) | derived from `keys_to_load` before `prepare_load` |

## Evidence (live L4 logs, `latest/run.log`)

**Geometry** — verbatim `CHUNKPROBE CONFIG`:

```
CHUNKPROBE CONFIG num_workers=1 block_size_factor=3 num_groups=1
CHUNKPROBE CONFIG group_idx=0 gpu_block_size=16 offloaded_block_size=48 hash_block_size_factor=3 sliding_window_size_in_blocks=None
```

`gpu_block_size=16` is `--block-size`; `offloaded_block_size=48` is
`kv_connector_extra_config["block_size"]`; `block_size_factor = 48//16 = 3` is computed by vLLM in
`base.py::OffloadingSpec.__init__`, not hand-set.

**Q1+Q2+Q3 (chunk size / hash / list)** — warm-phase `CHUNKPROBE STORE` (all 10 chunks, abbreviated):

```
STORE chunk_idx=0 offloaded_block_size=48 num_gpu_blocks_in_chunk=3 key_hash=70ca87…a6c5
  constituent_block_hashes=[62e880…d4e1, be6112…bd54, 70ca87…a6c5]  chunk_hash_eq_last_block=True
STORE chunk_idx=1 … key_hash=daf171…fc8c
  constituent_block_hashes=[ecf0c3…9c38, 83d563…af31, daf171…fc8c]  chunk_hash_eq_last_block=True
… chunk_idx=2..9 all chunk_hash_eq_last_block=True
```

⇒ chunk = 3 GPU blocks; identity = a single hash = the 3rd (last) block's hash; the other 2 block
hashes are not in the chunk identity.

**Router view** — `CHUNKPROBE EVENT` (my probe, inside `take_events`):

```
CHUNKPROBE EVENT type=stored medium=CPU num_hashes=10 hashes=<10 chunk-tail hashes> token_ids_len=0 block_size=0 parent=None
```

`num_hashes` / `hashes` / `token_ids_len` are **not** fields of `BlockStored` — they're my derived
prints. The real object clean vLLM builds is `BlockStored(block_hashes=<list, len 10>,
parent_block_hash=None, token_ids=[], block_size=0, medium="CPU", lora_id=None, lora_name=None)`.
So the only real payload is `block_hashes` + `medium`; a router can't recompute local hashes (no
tokens), can't tell the chunk spans 3 blocks (`block_size=0`), and can't chain it (no parent).

**Q4 (reuse works)** — Phase-3 `CHUNKPROBE REUSE/LOAD`:

```
REUSE/LOAD req=9-… num_chunks_loaded=10 (~480 tokens) chunk_key_hashes=[70ca87…, daf171…, …, 84aef5…]
```

⇒ vLLM reloaded the full 480-token prefix from CPU after GPU eviction — reuse works at chunk
(48-token) granularity, and the 10 loaded keys equal the 10 warm-stored chunk keys.

## How vLLM finds and reuses a chunk (code-grounded)

Does the engine look a chunk up by its `constituent_block_hashes` or by the single last-block
`key_hash`? **Only by the chunk-tail `OffloadKey` (last block's hash + group_idx) — one key per
chunk, end to end.** The non-last sub-block hashes are never keys.

1. **Key construction** — `update_offload_keys()` samples `req.block_hashes` with stride
   `hash_block_size_factor` starting at `hash_block_size_factor - 1`, keeping only each group's last
   hash: `offload_keys[n] = make_offload_key(block_hashes[factor*n + factor-1], gid)`. This is why
   `chunk_hash_eq_last_block=True`.

2. **Lookup (control plane — no bytes move).** `_lookup → _maximal_prefix_lookup` calls
   `manager.lookup(key)` once per chunk key, counting consecutive hits and stopping at the first
   miss. It only queries the manager's hash map (present? ready?) to decide *how many* leading
   chunks can be recovered:

```python
for key in keys:                       # keys = chunk-tail OffloadKeys
    result = self.manager.lookup(key, req_context)
    if not result:                     # True=ready / False=absent / None=write in-flight
        break                          # prefix must be contiguous
    hit_count += 1
```

3. **Load (data plane — the actual CPU→GPU copy), in two halves:**
   - *Scheduler side* (`update_state_after_alloc`): choose which chunks to bring back
     (`keys_to_load = offload_keys[start_block_idx:num_blocks]`), then `manager.prepare_load(keys)`
     resolves each `OffloadKey → CPU block id` (the manager dict is keyed purely on `OffloadKey`),
     bumps `ref_cnt` to pin them against eviction, and packages a `TransferJob(src=CPU blocks,
     dst=freshly-allocated GPU blocks)`. Still metadata — no bytes yet.
   - *Worker side* (`gpu_worker.py`): `swap_blocks_batch(...)` DMA-copies the bytes CPU→GPU. One
     CPU block is `gpu_page_size × factor` bytes, so `compute_sub_block_ptrs` fans it back out into
     `factor` GPU blocks. `complete_load` then drops `ref_cnt`.

So for the 480-token prefix: **10 chunks ⇒ 10 keys ⇒ 10 lookups ⇒ 10 loads** (matches
`num_chunks_loaded=10`). The 20 non-last sub-block hashes live only inside `req.block_hashes` and
are never indexed, looked up, or loaded.

**Granularity consequence:** the key is the cumulative tail hash, so a hit needs the request to
share **all `factor` sub-blocks** of a chunk; a prefix that diverges mid-chunk falls back to the
previous chunk boundary. Reuse is exact but only at `offloaded_block_size` (48-token) granularity.

## Implications

- **Dynamo (our consumer):** `convert.rs::create_stored_blocks` drops any block with
  `block_size != kv_block_size` and computes the local hash from `token_ids`. The placeholder
  (`block_size=0`, `token_ids=[]`) is dropped — the chunk never enters the lower-tier index. The
  loss is at the **event layer**, not the offload layer.
- **llm-d (future A/B):** per `LLMD_REPLY_DRAFT.md`, llm-d tolerates empty-token CPU events only by
  resolving the hash via its existing `engineKey → requestKey` table (built by the GPU
  `BlockStored`). For `factor > 1` it can update only the single tail-hash mapping — the **same**
  information loss. Shared root cause; the harness (`run_offline.py`, env-knobbed) can later drive a
  same-geometry, same-workload A/B comparing prefix-recovery + TTFT.

## Open question (deferred to design phase)

The chunk-tail hash is a real cumulative vLLM prefix hash, and reuse works at 48-token granularity.
The design choice: should a router be fed (a) the `factor` per-block hashes+tokens (vLLM-side
fan-out → stays block-granular, mid-chunk prefixes still match) or (b) a single coarse-block chunk?
This run does not pick a side; it only proves the chunk is a single-hash, multi-block,
self-non-describing event today while the underlying KV is real and reusable.

## Appendix: vLLM + Dynamo KV-cache / offloading internals (code-grounded digest)

Each claim is backed by the file/function named inline. Two paths: vLLM (one engine's data plane +
its own caches) and Dynamo (the multi-worker router index consuming vLLM's KV events).

### A1. The two `block_size` knobs differ

`--block-size` → GPU block size (`cache_config.block_size` = 16, logged `gpu_block_size=16`).
`kv_connector_extra_config["block_size"]` → offloaded chunk size (= 48, logged
`offloaded_block_size=48`). `base.py::OffloadingSpec.__init__` derives
`block_size_factor = 48//16 = 3`. When `cache_config.block_size == kv_cache_spec.block_size`
(standard), `hash_block_size == gpu_block_size`, so `hash_block_size_factor == factor`.

### A2. The block/index hash is a cumulative prefix-hash chain from the head

`kv_cache_utils.py::hash_block_tokens` chains each block onto its parent
(`hash(parent_block_hash, this_block_tokens, extra)`, with `NONE_HASH` as block 0's parent), and
`hash_request_tokens` walks `0,1,2,…` feeding each hash as the next block's parent. So
`block_hashes[i]` depends on all tokens in `[0 .. (i+1)*block_size)` — positional, head-anchored; a
hash matches only if the whole prefix matches. The chunk's OffloadKey is
`block_hashes[factor*c + factor-1]`, covering `[0 .. (c+1)*offloaded_block_size)`.

### A3. How `islice` samples chunk keys (`update_offload_keys`)

```python
for req_block_hash in islice(
    self.req.block_hashes,
    hbf * len(group_state.offload_keys) + hbf - 1,   # start = last hash of next chunk
    None,                                            # stop = until exhausted
    hbf,                                             # step = one chunk (hbf = hash_block_size_factor)
):
    group_state.offload_keys.append(make_offload_key(req_block_hash, group_idx))
```

`islice(it, start, stop, step)` is `it[start:stop:step]` but lazy/non-indexable. Here `step=hbf`
→ one landing per chunk; `+hbf-1` → lands on each chunk's **last** sub-block; `hbf*len(offload_keys)`
→ resume offset so incremental calls don't re-emit done chunks. Demo (`hbf=3`, `b0..b14`): 1st call
`start=2` → `b2,b5,b8,b11,b14`; after 4 chunks done, `start=14` → `b14`. Non-last hashes are never
sampled.

### A4. Offload advances from the head; the leftover is the tail

`num_blocks = num_offloadable_tokens // offloaded_block_size` (floor) counts whole chunks from token
0, `offload_keys[start:num_blocks]` slices from the head, `next_stored_block_idx` rises monotonically
from 0 — so CPU mirrors a **head-aligned, whole-chunk prefix**; the partial-chunk remainder is the
**tail**. Example (`factor=3`, 17 full blocks; verified by an `islice` + floor-division repro):

```
token sequence (head) ───────────────────────────────────────► (tail)
 block:  0  1  2 | 3  4  5 | 6  7  8 | 9 10 11 |12 13 14 |15 16
 chunk:  └chunk0 ┘ └chunk1 ┘ └chunk2 ┘ └chunk3 ┘ └chunk4 ┘ └leftvr┘
 to CPU:    ✔        ✔         ✔         ✔         ✔          ✗
            └──────── head: 5 whole chunks = 240 tokens ──────┘ └tail: 2 blocks┘
```

`islice(start=2, step=3)` over 17 elements → `b2,b5,b8,b11,b14` (next landing 17 is out of range, so
blocks 15/16 get no key); `272//48 = 5` chunks stored, 32 tokens (2 blocks) left. **Write-through
caveat:** during the request all blocks live on GPU; offloaded ones are merely *also* on CPU. The
tail remainder is "only-GPU / pending" — it offloads once decoding completes its chunk; only a
request that ends first leaves the tail CPU-less.

### B1. vLLM's GPU index and CPU index — both flat hash maps, two separate objects

| | GPU prefix cache | CPU offload index |
|---|---|---|
| Object | `BlockPool.cached_block_hash_to_block` (KV-cache manager) | `OffloadingManager` → `CachePolicy.blocks` (connector, scheduler-side) |
| Structure | flat hash map (`BlockHashToBlockMap`) | flat hash map (`OrderedDict`) + LRU/ARC ordering |
| Key | per-block `BlockHash` (+group) | `OffloadKey` = chunk-tail hash + 4-byte group_idx |
| Granularity | 1 GPU/hash block (16 tok) | 1 chunk (48 tok) |
| Queried | first, `get_computed_blocks` | only for the suffix the GPU missed, `get_num_new_matched_tokens` |

Neither is a radix tree — "longest prefix match" is a sequence of O(1) hash lookups (block 0,1,2…
stop at first miss); prefix semantics come from cumulative hashing (A2). They are two independent
objects queried in sequence, sharing only the underlying block-hash values (CPU samples them at
chunk granularity). Both live in **host (CPU) memory** in the scheduler process and index *metadata*
(hash → block_id); KV bytes live in GPU HBM / pinned host RAM. The GPU never hashes or does index
lookups — "GPU index" names the tier it tracks, not where it lives.

### B2. Why hash, not radix (inside one engine)

The deciding factor is the cumulative-hash design, not block count: each block hash already encodes
its full prefix, so prefix matching is O(1)-per-block lookups and prefix sharing is implicit (same
prefix → same hashes → same entries). A radix tree adds node split/merge, pointer chasing, and
concurrency complexity for no lookup-count gain. Moderate block counts only *support* the choice.

### C1. Dynamo router: device = radix tree, lower tiers = sharded-hash edge map

- **device tier** uses a concurrent radix tree (`indexer/concurrent_radix_tree_compressed/*`):
  the router must *discover*, across **all workers** with no anchor, each worker's longest cached
  prefix + an overlap **score**, plus frequency tracking and structure-aware cleanup — a branching,
  multi-owner ranking problem.
- **lower tiers** (HostPinned/Disk/External) use a sharded concurrent hash map of continuation edges
  (`indexer/lower_tier.rs`: `edges: DashMap<TransitionKey, EdgeOwnersEntry>`, keyed by
  `(parent_hash, local_hash) → child + owners`).

Concurrency note (correcting a common assumption): "big hash map ⇒ global lock" is false — `DashMap`
is sharded (key → shard → per-shard lock) and each lower-tier worker thread owns its `worker_blocks`
partition, so there's no global lock. Conversely a radix tree isn't automatically lock-free: every
query traverses the root, so upper nodes are contention hotspots unless you add lock-free/optimistic
machinery (`node_state.rs`, `sync_impl.rs`, `repair.rs`). Lookup count is the same (~`num_blocks`);
the real difference is *what query/lifecycle* each must support.

### C2. Why lower tiers only need "does this worker continue from here"

Routing walks Device → HostPinned → Disk → External as **one per-worker forward chain**
(`lower_tier_indexers.rs::query_lower_tiers`):

```rust
// seed each worker's continuation from its DEVICE match
continuations[worker] = LowerTierContinuation::new(device_matched_blocks, device_last_hash);
for tier in [HostPinned, Disk, External] {
    for w in tier.root_workers(first_hash) { continuations.entry(w).or_insert(from_root(0)); }
    let m = tier.query_match_details(sequence, &continuations);
    continuations = m.next_continuations;   // chain to the next tier
}
```

The device tier does the one-time global discovery + scoring; each lower tier only **extends** an
already-known per-worker prefix from a fixed anchor `(start_pos, last_hash)`. So it never re-ranks
across workers, never starts a fresh global search (beyond a single `root_workers(first_hash)` lookup
for CPU-only-from-root workers), and removals are per-edge with no aggregate state to recompute —
hence an edge hash map suffices, while the radix tree is reserved for the device tier.

### C3. Tier eviction is LRU/ARC (block/chunk granular) everywhere — not subtree

No tier uses subtree eviction as a *policy*. All eviction decisions are made by the **vLLM engine**;
the Dynamo router never evicts — it only mirrors the resulting `BlockStored`/`BlockRemoved`/`Cleared`
events.

- **Single CPU tier** (`CPUOffloadingSpec`, what we tested): when the pool is full,
  `cpu/manager.py::prepare_store` calls `self._policy.evict(n, protected)` → **LRU**
  (`policies/lru.py`, drops `OrderedDict` front) or **ARC** (`policies/arc.py`), at chunk
  granularity; emits `BlockRemoved(medium=CPU)`.
- **Multi-tier** (`TieringOffloadingSpec`, `tiering/manager.py`) is **inclusive cascade, not
  CPU→SSD demotion**: a stored block is written to CPU *and cascaded to every secondary tier* at
  store time ("Always offload to all tiers"; `complete_store` loops `tier.submit_store(...)`). So
  when CPU fills it just LRU/ARC-drops its own copy — the SSD copy already exists; nothing moves on
  eviction. A read that misses CPU but hits a secondary triggers a **promotion** (copy back to CPU,
  `lookup → _initiate_promotion`), since secondaries can't feed the GPU directly.
- The only subtree-aware work is the Dynamo device radix tree's **structural cleanup** when applying
  a removal (parent/child invariants, orphan pruning) — structure maintenance, not an eviction
  policy. The lower-tier edge map removes per-edge in O(1).

### Recap (for the design phase)

The chunk is real, reusable inside vLLM, indexed by a single cumulative tail-hash at
`offloaded_block_size` granularity (head-anchored, whole-chunk). The cross-process event is
self-non-describing (`token_ids=[]`, `block_size=0`, one tail hash). A router fix must reconstruct,
from that event, either (a) the `factor` per-block hashes+tokens or (b) a coarse single-chunk block.
This appendix is the mechanism evidence; the choice is deferred.

## Reproduce

`run_offline.py` and `instrumented_clean_scheduler.py` are bundled in this folder. Raw run logs
from the original run live under `/home/changg/workspace/.tmp/phase1_chunk_verify/` (scratch).

```bash
cd /home/changg/workspace/docs-vllm-native-offloading-dynamo/phase2-chunk-offloading
# re-apply the clean+instrumented scheduler onto the vLLM tree:
cp instrumented_clean_scheduler.py \
  /home/changg/workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
VLLM_LOGGING_LEVEL=INFO /home/changg/workspace/.venv-cu130/bin/python run_offline.py 2>&1 | tee run.log
grep CHUNKPROBE run.log
# restore the pristine scheduler afterwards:
cd /home/changg/workspace/vllm && git checkout HEAD -- \
  vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

Knobs (env): `CP_OFFLOADED_BLOCK_SIZE` (factor), `CP_BLOCK_SIZE`, `CP_GPU_BLOCKS`, `CP_PREFIX_LEN`,
`CP_N_FILLER`, `CP_FILLER_LEN`, `CP_MODEL`, `CP_MAX_MODEL_LEN`. Constraints:
`CP_GPU_BLOCKS * CP_BLOCK_SIZE >= CP_MAX_MODEL_LEN` and `CP_PREFIX_LEN % CP_OFFLOADED_BLOCK_SIZE == 0`.
