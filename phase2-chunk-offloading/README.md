# Phase 2 — chunk (group) offloading

Phase 1 (the rest of this repo) made vLLM's CPU offload tier visible to Dynamo's KV-router and
tuned it on B200. Phase 2 tackles the remaining gap: **group/chunk offloading**, where
`kv_connector_extra_config["block_size"] > --block-size` (so `block_size_factor > 1`) makes vLLM
pack `factor` GPU blocks into one CPU "chunk". Today such chunks are emitted as self-non-describing
placeholder events and dropped by routers.

## Steps

- **Step 1 — verify the chunk fact** (done): on clean upstream vLLM, establish empirically what a
  chunk is (size, hash, single-vs-list), whether vLLM reuses it, and exactly what a router receives.
  → [`step1-verify-chunk-offload-fact.md`](step1-verify-chunk-offload-fact.md)
  - Harness (self-contained, reusable for Step 2): `run_offline.py`,
    `instrumented_clean_scheduler.py`.
- **Step 2 — llm-d cannot match the chunk** (done): on real multi-turn traffic, upstream chunk
  offloading drops llm-d's contiguous CPU match from 68/68 → 0/68.
  → [`step2-llmd-cannot-match-chunk.md`](step2-llmd-cannot-match-chunk.md)
- **Step 3 — vLLM fix, llm-d matches the chunk** (done): vLLM publishes one tail hash + the whole
  chunk's `token_ids` + per-block `block_size`; llm-d's existing 1:many path lights every block,
  restoring 94.3% coverage / 68/68 contiguous for both store and remove. vLLM-side only.
  → [`step3-vllm-fix-llmd-matches-chunk.md`](step3-vllm-fix-llmd-matches-chunk.md)
- **Step 4 — Dynamo lower-tier index study** (done): the *other* router. Native CPU offload lands in
  Dynamo's `LowerTierIndexer`, a `(parent_seq_hash, local_hash) → child_seq_hash` continuation chain;
  the study ends with 7 constraints any chunk-mode integration must satisfy.
  → [`step4-dynamo-lower-tier-index.md`](step4-dynamo-lower-tier-index.md)
- **Step 5 — final design: expose chunks as N per-block events (Plan B)** (done): against Step 4's
  constraints, the minimum-mutation plan is for vLLM to emit the chunk's `factor` constituent per-block
  hashes + whole-chunk tokens (+ a removal side table). Both llm-d and Dynamo consume it with **zero
  router changes**. Supersedes Step 3's Plan A once Dynamo is in scope.
  → [`step5-final-design.md`](step5-final-design.md)
- **Step 6 — E2E verification + the overlapping-chunk eviction hazard** (done): Plan-B store AND
  remove verified end-to-end on real traffic with real CPU evictions against both routers'
  production index code (llm-d 71/71 + 516/516; Dynamo 11/11 after a 1-line batch-abort fix).
  Surfaced, measured, and fixed the overlapping-chunk hazard: a shared prefix not aligned to
  `offloaded_block_size` makes sibling boundary chunks re-list the same block hashes, so the first
  sibling's eviction (a) aborted Dynamo's removal batch → leaked edges (fixed in Dynamo: skip
  absent hashes) and (b) truncated every other sibling's router-side CPU match while vLLM could
  serve them. A vLLM-side per-hash removal refcount was built and verified (3/3 restored on the
  surviving sibling), but per the team decision the shipped producer is **plain fan-out** and
  consumers deduplicate — Dynamo's standard deployment already runs an `EventDedupFilter`
  (ai-dynamo/dynamo#8012) with exactly the needed refcount semantics; the refcount/exactly-once
  producer variants are archived on `feature/offloading-events-exactly-once`. Known limitation:
  filter-less single-entry consumers (e.g. llm-d today) drop a shared block on the first
  sibling's eviction (under-credit only, never corruption).
  → [`step6-e2e-overlapping-chunk-eviction.md`](step6-e2e-overlapping-chunk-eviction.md)
  ([中文版](step6-e2e-overlapping-chunk-eviction.zh-CN.md))

- **Step 7 — real dynamo serve + vLLM e2e, metrics-verified.**
  Single L4: vLLM (chunked offload `factor=3`, opt-in self-describing events, 128 MB CPU pool
  with real LRU evictions) behind a real dynamo frontend (`--router-mode kv`) and worker
  publisher (EventDedupFilter in the path), zero external services (file discovery / TCP request
  plane / ZMQ event plane). Router `kv_cache_events_applied` reconciles **exactly** against a
  sidecar wire capture: stored 685 = 331 GPU + 354 CPU, removed 24, zero warnings. Found and
  fixed a metrics-wiring gap on the way (lower-tier indexers built without a metrics handle —
  dynamo `db0ec356`), plus a pitfall list (stale maturin binding, `--kv-events-config` must be
  explicit, which counter is whose).
  → [`step7-dynamo-vllm-real-e2e.md`](step7-dynamo-vllm-real-e2e.md)
  ([中文版](step7-dynamo-vllm-real-e2e.zh-CN.md))

## Step 1 one-liner

factor=3 ⇒ chunk = 48 tok = 3 GPU blocks, keyed by a single hash (= the last sub-block's hash);
the cross-process event carries only that tail hash (`token_ids=[]`, `block_size=0`, no parent),
yet vLLM itself reloads all chunks CPU→GPU correctly. The loss is at the event layer, not the
offload layer. See the doc for code-grounded evidence + a full internals appendix.
