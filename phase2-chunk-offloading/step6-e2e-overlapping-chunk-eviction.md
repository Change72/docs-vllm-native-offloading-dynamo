# Phase 2 · Step 6 — E2E verification, and the overlapping-chunk eviction hazard

> **Step 5** ([step5-final-design.md](step5-final-design.md)) picked **Plan B**: vLLM publishes
> each offloaded chunk as ONE `BlockStored` carrying all `factor` constituent per-block hashes +
> whole-chunk tokens, and a `BlockRemoved` fanning out the same hashes at eviction
> (vLLM commit `c3e203a18` on `feature/offloading-connector-chunk-payload`).
> **Step 6 (this doc):** end-to-end **correctness** verification of those events on real traffic
> with **real CPU evictions**, against both routers' production index code — plus the one real
> bug it surfaced (fixed) and one residual hazard (open, measured, mitigation proposed).

## TL;DR

| layer | what ran | store | remove | verdict |
|---|---|---|---|---|
| vLLM wire | real serve, factor=3, 128 MB CPU pool → real LRU evictions; 313 ZMQ frames decoded | 365 events, all `n_hashes=3, tok=48, bs=16`, parent chain intact | 24 events, 1023 hashes; split exactly into **341 whole previously-stored chunks** | ✅ |
| llm-d | replay through real `VLLMAdapter → kvevents.Pool → InMemoryIndex` | **71/71** live blocks have a CPU pod | **516/516** evicted blocks have none | ✅ |
| Dynamo | replay through real `decode_event_batch → ZmqEventNormalizer → LowerTierIndexers` (CPU events only — proves self-containedness) | 11/11 chunk chains match expected HostPinned depth | before fix: 10/11 + **29 leaked edges**; after 1-line fix: **11/11** | ✅ after fix |

**The hazard — overlapping chunk events.** When a shared prefix is **not a multiple of
`offloaded_block_size`**, the boundary chunk of every conversation branch is a *different
OffloadKey that lists the same shared block hashes*. vLLM dedups per chunk key, so those hashes
are legitimately stored/removed **multiple times** on the wire. Two consequences:

1. **B1 (FIXED, Dynamo)** — the lower tier aborted a whole `BlockRemoved` batch on the first
   already-removed hash → leaked edges (unbounded; violates Step 4 §9 constraint 7).
2. **B2 (FIXED, vLLM)** — *first-evict-wins*: the shared block's index entry has **no refcount**
   in either router, so the first sibling's eviction deleted it while other siblings were still
   fully resident in vLLM's CPU pool → their router-side contiguous CPU match collapsed
   (measured: 3/3 → **0/3** on llm-d) while vLLM itself could still reuse them. Fixed by
   **reference-counting announced hashes in the vLLM side table** (`746d7dc26`): a hash enters
   `BlockRemoved` only when its last live chunk dies — both routers untouched. Post-fix probe:
   surviving sibling stays **3/3**; the re-captured wire stream has **zero** un-announced
   removals.

## 1. How it was verified (all real)

1. **vLLM** `feature/offloading-connector-chunk-payload` @ `c3e203a18`, `Qwen/Qwen3-0.6B`,
   factor=3, `cpu_bytes_to_use=128MiB` (small on purpose → the pool fills and LRU-evicts),
   ZMQ KV events on; multi-turn benchmark (16 conversations, 256-token shared prefix).
   Captured every raw frame (`run_evict.sh planb_chunk_evict planb 3 134217728`).
2. **Wire decode** (`decode_capture.py`): shape + accounting checks. All five PASS, including:
   every removed CPU hash was previously stored, and every `BlockRemoved` splits into whole
   previously-stored chunk groups — the removal fan-out is byte-exact under real eviction.
   (81 in-batch out-of-order arrivals observed — expected: `complete_store` iterates a job's
   `set`; events are self-contained so order is irrelevant.)
3. **llm-d** (`examples/chunk_replay_v2`): replays the same frames through the real ingest
   stack while independently tracking expected per-block liveness (store +1 / remove −1).
   Final state: every live block CPU-matchable, every dead block gone.
4. **Dynamo** (`lib/kv-router/tests/planb_capture_replay.rs`): replays **only the CPU events**
   through the production decode/normalize path into `LowerTierIndexers` — deliberately, since
   Plan B's claim is that CPU events are self-contained (no GPU alias needed). Reconstructs all
   root-to-leaf chunk chains and asserts the real `query_lower_tiers` from-root HostPinned depth
   equals the model's contiguous-live depth, per chain.

## 2. The hazard: one shared block, many chunks

Trigger: `shared_prefix_len % offloaded_block_size != 0`. With a 256-token shared prefix and
48-token chunks (16-token blocks, factor=3):

```
              shared 256-token prefix (16 blocks)            per-conversation
blocks:   0  1  2 │ 3  4  5 │ 6  7  8 │ 9 10 11 │ 12 13 14 │ 15 16A 17A …   conv A
                                                            │ 15 16B 17B …   conv B
chunks:   └─ c0 ─┘ └─ c1 ─┘ └─ c2 ─┘ └── c3 ──┘ └── c4 ──┘ └── c5 ───┘
          c0..c4: ALIGNED shared chunks → same OffloadKey for A and B
                  → vLLM stores ONE copy, emits ONE store / ONE remove. Clean.
          c5:     256 % 48 ≠ 0 → the boundary chunk STRADDLES the fork:
                  c5A = OffloadKey h17A, hashes [h15, h16A, h17A]
                  c5B = OffloadKey h17B, hashes [h15, h16B, h17B]
                  → different keys, TWO independent CPU copies, BOTH list h15.
```

Block 15's hash is identical across conversations (cumulative hash over the shared 256 tokens);
blocks 16+ diverge. So the wire stream is, per chunk, perfectly paired — but per *hash*:

```
store  c5A → BlockStored [h15, h16A, h17A]
store  c5B → BlockStored [h15, h16B, h17B]    ← h15 stored AGAIN
evict  c5A → BlockRemoved[h15, h16A, h17A]    ← h15 removed (1st time)
evict  c5B → BlockRemoved[h15, h16B, h17B]    ← h15 removed AGAIN
```

Measured in the capture: one boundary hash appeared in **11 distinct chunks**; 348 hashes were
stored more than once (fork variants + evict/re-store cycles).

**Why vLLM does not dedup this.** The store dedup unit is the OffloadKey
(`prepare_store` skips keys already present); c5A and c5B are *different keys*. The pool's
`ref_cnt` is a **transfer-time pin** (protects a block while a load/store is in flight), not a
"how many sequences share this block" count. Aligned shared chunks need no count (same key, one
copy); straddling chunks are different keys with independent copies — correct *inside* vLLM,
duplicated *on the wire*. **By-block mode (`factor=1`) cannot exhibit this**: the dedup unit is
the block itself, so a hash is stored/removed exactly once globally.

## 3. Consequence B1 (FIXED): batch-abort leak in Dynamo

Dynamo's lower-tier reverse table is a single-value map (`worker_map: hash → TransitionKey`) —
the second store of h15 overwrites (no count), the first remove deletes. The second remove
misses, and the original code **aborted the whole batch**:

```
evict c5B → BlockRemoved [ …54 hashes…, h15, h16B, h17B ]
                                         ▲
            worker_map.remove(h15) → None → return Err(BlockNotFound)
            → h16B, h17B and EVERY remaining hash in the batch never removed
```

Replaying the capture against exact single-value semantics: **3 aborts, 36 skipped hashes,
29 leaked edges** (Dynamo view: 100 live hashes; refcount truth: 71). Leaks are unbounded over
time (violates Step 4 §9 #7), and the same abort fires on the legitimate cross-tier race of
constraint #6 (GPU `BlockRemoved` observed before CPU `BlockStored`).

**Fix** (`lib/kv-router/src/indexer/lower_tier.rs::remove_blocks_impl`): skip absent hashes and
keep draining the batch; return `BlockNotFound` only if *every* hash was absent. This matches
llm-d's existing behavior (`Evict` on an unknown engine key is a no-op). After the fix: replay
11/11 chains PASS; all 545 kv-router unit tests green.

## 4. Consequence B2 (FIXED): first-evict-wins under-credit — both routers

The shared block has **one** index entry per router (Dynamo: edge `(h14,l15)→h15` +
`worker_map[h15]`, idempotent re-insert; llm-d: one `PodEntry` per requestKey, set semantics).
Neither is reference-counted, so the **first sibling's eviction deletes it for everyone**:

```
vLLM CPU pool after evict c5A:   c5A gone, c5B fully resident → lookup(B) = FULL HIT
router index  after evict c5A:   h15 entry deleted by c5A's BlockRemoved
                                  → B's chain breaks AT THE SHARED BLOCK
```

Synthetic proof on real llm-d code (`examples/overlap_probe`, two 48-token convs sharing
block 0):

```
after store A+B : contiguous CPU  A=3/3  B=3/3
after evict A   : contiguous CPU  A=0/3  B=0/3   ← B still resident in vLLM!
```

Dynamo has the identical failure by code inspection: `worker_map.remove(h15)` deletes the one
edge; B's later blocks (`(h15,l16B)→h16B`, still indexed) become unreachable by the chain walk.

**Severity.** The break lands at the *shared* block — i.e. at the head side of every sibling's
chain. Generalizing the 1k/3k question: conversations A (1k) and B (3k) share a non-aligned
prefix; when A's straddling chunk is LRU-evicted (A went cold), B's router-side CPU credit
collapses to the aligned shared chunks before the straddle point — for a head-of-chain shared
block, to **zero** — even though vLLM would serve B's full prefix from CPU. The direction is
strictly conservative (no mis-routing, no data corruption, no leak), but the CPU-tier routing
signal for *every still-hot sibling* is destroyed by one cold sibling's eviction. Shared-prefix
workloads (system prompts, few-shot templates) almost never align to `offloaded_block_size`, and
the duplication degree equals the fork-out (measured ×11 at 16 conversations).

Contrast with the **aligned** part of the prefix (c0..c4): one key, one copy, one store/remove —
if c4 is ever evicted, vLLM's own lookup truncates at the same point the router does. Router and
engine agree; that is ordinary cache behavior, not a defect. The defect window is exactly the
straddling chunk's shared blocks.

**The fix — reference-count announced hashes (vLLM `746d7dc26`).** The scheduler keeps, next to
`_pending_event_metadata`, a second map `_block_hash_ref_counts: hash → live chunk count`
(per-hash, cross-chunk — it cannot live inside the per-chunk metadata). It is incremented when a
`BlockStored` is **emitted** (not at populate time, so a failed store that never announces cannot
leak a count) and decremented on eviction; a hash enters `BlockRemoved` only when its count hits
zero — exactly when its last CPU copy disappears. Duplicate *stores* stay on the wire (idempotent
in both routers). Bounded by one entry per distinct offloaded hash; cleared by `reset_cache`.

Verified: re-captured run (`planb_rc_evict`) shows **0 un-announced removals** (per-hash announce
state machine PASS, 20 idempotent re-announces of shared hashes); llm-d replay 71/71 + 516/516;
Dynamo replay 11/11; the two-scenario `overlap_probe` shows the old wire shape collapsing B to
0/3 and the refcount shape keeping B at **3/3** (A correctly degrades to 1/3 — its shared block
is still served by B's chunk). vLLM unit tests: 76/76, including a new overlapping-chunks
refcount test.

## 5. Fix matrix

| # | where | what | status / cost |
|---|---|---|---|
| 1 | Dynamo `remove_blocks_impl` | skip absent hashes, drain batch (B1) | ✅ done, +20 lines, 545 tests green; also hardens the constraint-6 cross-tier race |
| 2 | **vLLM removal refcount (kills B2 for all routers)** | side table keeps `hash → live chunk count` (++ on BlockStored emission, −− on evict); `BlockRemoved` lists only hashes whose count hit 0. Duplicate *stores* stay (idempotent in both routers). | ✅ done (`746d7dc26`); both routers untouched; O(live hashes) extra state, cleared by `reset_cache` |
| 3 | router-side refcount | count duplicate stores in Dynamo's `worker_map` / llm-d's pod entries | superseded by 2 — two implementations of the same fix |
| 4 | deployment workaround | pad shared prefixes (system prompts / templates) to a multiple of `offloaded_block_size` | superseded by 2 |

Option 2 is the natural counterpart of Plan B's removal side table: the table already exists
per chunk; the refcount is the per-hash aggregation of it, and "announce removal only when the
last reference dies" is exactly the cache semantics routers expect. Defense in depth: fix 1 stays
— with fix 2 the duplicate-removal trigger disappears from this producer, but the lower tier
should still tolerate unknown hashes (constraint-6 ordering, other producers).

## 6. Reproduce

```bash
# 1) capture with real evictions (writes runs/planb_rc_evict/)
cd /home/changg/workspace/.tmp/llmd_4way && bash run_evict.sh planb_rc_evict planb 3 134217728
# 2) wire-shape + accounting checks (incl. the per-hash announce state machine)
python decode_capture.py runs/planb_rc_evict/capture.jsonl --factor 3 --block-size 16
# 3) llm-d store+remove replay
cd /home/changg/workspace/llm-d-kv-cache-manager && go build -o chunk_replay_v2_bin ./examples/chunk_replay_v2 \
  && ./chunk_replay_v2_bin /home/changg/workspace/.tmp/llmd_4way/runs/planb_rc_evict/capture.jsonl
# 4) Dynamo store+remove replay (CPU-events-only, from-root walk)
cd /home/changg/workspace/dynamo && DYNAMO_PLANB_CAPTURE=/home/changg/workspace/.tmp/llmd_4way/runs/planb_rc_evict/capture.jsonl \
  cargo test -p dynamo-kv-router --test planb_capture_replay -- --ignored --nocapture
# 5) first-evict-wins: old shape reproduces (B=0/3), refcount shape eliminates (B=3/3)
cd /home/changg/workspace/llm-d-kv-cache-manager && go run ./examples/overlap_probe
# (pre-refcount capture runs/planb_chunk_evict/ is kept for regression replays)
```
