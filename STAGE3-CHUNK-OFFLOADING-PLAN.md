# ===== STAGE 3 — Chunk offloading is now router-matchable (status + final plan) =====

> **Bottom line:** vLLM's chunked CPU offload (`block_size_factor > 1`) used to be invisible to
> KV-cache-aware routers. As of this stage it is **fully matchable on llm-d** — **94.3% CPU
> coverage, 68/68 contiguous prefix match, for both store and remove** — via a **vLLM-side-only**
> change that ships on the existing PR. Dynamo gets the same payload for free once its indexer
> learns the same 1:many mapping (owned separately).

This is the capstone of the chunk-offloading effort. The diagnosis and per-step evidence live under
[`phase2-chunk-offloading/`](phase2-chunk-offloading/); this file is the **plan of record**.

---

## The arc (three stages)

| stage | question | outcome |
|---|---|---|
| **Stage 1** — CPU tier → Dynamo (B200) | Can we make vLLM's by-block CPU offload visible to Dynamo's router and is it worth it? | ✅ Done & benchmarked: **+261% throughput, −5× TTFT p99** at saturation (this repo's main body + [`PRESENTATION.md`](PRESENTATION.md)). Landed via vLLM `e074f0a` + Dynamo `6c2a73a`/`5b7725f`. |
| **Stage 2** — does *chunked* offload survive routing? | When `offloaded_block_size > block_size`, can llm-d still match the CPU cache? | ❌ No. Upstream emits **one tail hash per chunk, no tokens** → llm-d contiguous match **collapses 68/68 → 0/68** ([step2](phase2-chunk-offloading/step2-llmd-cannot-match-chunk.md)). |
| **Stage 3** — fix it (this doc) | What minimal change makes chunked offload routable? | ✅ vLLM publishes **1 representative hash + whole-chunk tokens + per-block block_size**; llm-d's existing **1:many** lights every block → **94.3% / 68-68, store + remove** ([step3](phase2-chunk-offloading/step3-vllm-fix-llmd-matches-chunk.md)). |

---

## What shipped in Stage 3

- **The fix (vLLM only):** for a chunk, `_build_event_metadata` emits a single engine hash (the
  chunk tail = its `OffloadKey`) + the chunk's full `token_ids` + `block_size` = the per-block token
  count. A block-granular router re-splits the tokens into `factor` blocks and maps the one hash to
  all of them (1:many). `factor == 1` (by-block) is unchanged. Removal carries the same single hash;
  the router's 1:many evicts every block.
- **Why this shape:** reuses llm-d's existing `engineToRequestKeys` 1:many → **no new router state**,
  **no vLLM-side removal map**, smaller events (1 hash vs `factor`), and the router computes per-block
  keys from tokens (LoRA / hash-fn agnostic).
- **Validated:** 57/57 unit tests (by-block + chunk, store + remove); real multi-turn capture
  replayed through the actual `llm-d-kv-cache-manager` ingest stack.
- **Where it landed:** branch **`bugfix/offloading-connector-blockstored-payload`**, commit
  **`54044fad4`** — pushed onto the **existing PR** (extends its scope from by-block to by-block +
  chunk).

---

## Readiness matrix

| capability | state |
|---|---|
| By-block (`factor=1`) store + remove, routable | ✅ ready (on PR) |
| Chunk (`factor>1`) store + remove, routable on llm-d | ✅ ready (on PR) |
| Event fields `token_ids`, `block_size`, `parent_block_hash`, `lora_*`, `group_idx`, spec kind | ✅ populated |
| `extra_keys` (multimodal / `cache_salt` / prompt-embeds distinguisher) | ⛔ deferred → `None` |
| Sliding-window / SSM groups | ⛔ deferred → placeholder fallback |
| Dynamo router 1:many (mirror of llm-d) | ⛔ owned separately (not in this PR) |

Caveat to track: because `extra_keys=None`, **multimodal / `cache_salt`** requests are keyed from
tokens alone (true for the by-block path already on the PR, not new to chunk). The safe stop-gap is a
guard that routes those requests back to the placeholder payload until `extra_keys` is plumbed.

---

## Final plan — remaining work, in order

1. **Dynamo 1:many** *(owner: us, separate from this PR)* — teach the Dynamo indexer the same
   single-`engineKey` → `factor` `requestKey` expansion llm-d already does, so the identical vLLM
   payload routes on Dynamo. No further vLLM change required.
2. **`extra_keys` — next vLLM increment.**
   - *By-block:* easy — clean 1:1 with `block_hashes`; reuse `generate_block_hash_extra_keys`.
   - *Chunk (single-hash):* needs a small **contract decision** — the schema defines `extra_keys` as
     one entry per `block_hashes` entry, but a chunk has 1 hash and `factor` token-blocks; either
     align `extra_keys` to the re-split token-blocks (router cooperation) or fall back to multi-hash
     for MM/salt requests only. Plus MM e2e testing.
   - *Cheap safety guard (do first):* skip the snapshot for `mm_features` / `cache_salt` requests →
     placeholder fallback (~3 lines), making both paths MM-safe before the full plumbing.
3. **Sliding-window / SSM** *(later phase)* — not a field-fill: these are window/recurrent-state
   semantics, while prefix-model routers (llm-d) early-stop on the first gap. Making them routable
   needs **router-side** sliding-window/state-aware matching + a decision on emitted hash/parent
   semantics. Larger, cross-repo.

---

## Pointers

- Diagnosis & evidence: [`phase2-chunk-offloading/`](phase2-chunk-offloading/)
  — [step1 (what a chunk is)](phase2-chunk-offloading/step1-verify-chunk-offload-fact.md) ·
  [step2 (why it breaks llm-d)](phase2-chunk-offloading/step2-llmd-cannot-match-chunk.md) ·
  [step3 (the fix)](phase2-chunk-offloading/step3-vllm-fix-llmd-matches-chunk.md)
- Stage 1 (B200 Dynamo) writeup: [`PRESENTATION.md`](PRESENTATION.md)
- vLLM change: branch `bugfix/offloading-connector-blockstored-payload` @ `54044fad4`.
