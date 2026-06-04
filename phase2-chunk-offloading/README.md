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

## Step 1 one-liner

factor=3 ⇒ chunk = 48 tok = 3 GPU blocks, keyed by a single hash (= the last sub-block's hash);
the cross-process event carries only that tail hash (`token_ids=[]`, `block_size=0`, no parent),
yet vLLM itself reloads all chunks CPU→GPU correctly. The loss is at the event layer, not the
offload layer. See the doc for code-grounded evidence + a full internals appendix.
