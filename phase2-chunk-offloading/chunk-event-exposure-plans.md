# Stage 2 — Self-describing KV events under chunk offload (`factor > 1`)

> **Scope of this doc.** Proves the Stage-1 self-describing contract extends *correctly* to
> production-sized chunk offload (`block_size_factor = 16`), that the overlapping-chunk duplicate
> hazard is handled with no premature eviction, and that chunk mode preserves routing quality vs
> by-block (`factor = 1`) while reducing offload I/O. **Not** a perf-win story — that is Stage 1.
>
> Shared background (the bug, the router cost function, the 8×B200 perf sweep, deploy setup) lives in
> the Stage-1 whitepaper [`../../PRESENTATION.md`](../../PRESENTATION.md); this doc references it and
> does not re-derive it.

---

## 1. TL;DR

- Stage 1 fixed the by-block (`factor=1`) self-describing path; Stage 2 extends the **same contract**
  to chunk mode: one CPU `BlockStored` carries the chunk's **`F` constituent per-block hashes**, the
  **whole-chunk `token_ids`**, per-block `block_size`, parent hash, LoRA + group metadata.
- **What this doc establishes (all validated — §4/§5):**
  1. `F=16` event shape + reconciliation is correct (`n_hashes==16`, `len(token_ids)==256`, zero
     placeholders, router applies all stored/removed events with 0 errors) — §4.1, and at scale §5.1.
  2. Overlapping (non-aligned) chunks produce duplicate store/remove on the wire, and the consumer
     ref-counts them so **no block is evicted while a sibling chunk still references it** — §4.2.
  3. `F=16` serving perf ≈ `F=1` at **~24× fewer offload *store* events** (the removed-event row is
     ~2× fewer, so total applied KV events ≈2.4× fewer) — §5.2.
- **Deferred / out of scope:** sliding-window groups keep the legacy placeholder fallback;
  `extra_keys` (multimodal / cache-salt / prompt-embedding) paths are not validated here.

---

## 2. What chunk mode changes  *(distilled from Stage-1 §2.3/§3 and [step6](step6-e2e-overlapping-chunk-eviction.md))*

- `OffloadingSpec` with `block_size_factor = F` packs `F` consecutive GPU blocks into one CPU chunk
  (`cpu_page_size = gpu_page_size × F`) to cut I/O count + metadata.
- Stage 2 contract: the connector records chunk metadata while request/KV-group context is still in
  scope, then emits **one self-describing `BlockStored` per chunk** (all `F` per-block hashes + whole
  chunk tokens). At CPU eviction the matching `BlockRemoved` fans out the **same** `F` hashes.
- **Plain fan-out, not exactly-once.** If a shared prefix ends mid-chunk, two non-aligned sibling
  chunks list the same constituent hash → duplicate store/remove on the wire. Consumers indexing at
  per-block granularity **must ref-count/dedup** (Dynamo `EventDedupFilter`; llm-d equivalent). This
  is the one behavior by-block mode does not have.

---

## 3. Deploy method (how the F=16 runs are produced)

The cluster's Grove/DGD operator pins dynamo ≤1.2.x, incompatible with the vLLM build this feature
needs (see [`../../project-b200-chunk-dynamo-version-wall`] / Stage-1 notes) → the DGD path is blocked.
**Two operator-free deploys are used**, both on image
`change1472/dynamo-vllm-cpu-offload:stage2-reviewfix-20260619` (dynamo 1.3.0 + vLLM #43468; public),
command uses **`python3`** (image has no bare `python`).

**(a) Single-Pod bypass — used for the correctness runs S & B (1 worker).** `frontend` + `worker` as
two containers in one Pod sharing an `emptyDir` for `DYN_DISCOVERY_BACKEND=file`
(`DYN_FILE_KV=/shared/dynstore`, `DYN_REQUEST_PLANE=tcp`, `DYN_EVENT_PLANE=zmq`) — no Grove / NATS /
etcd. GPU: `nvidia.com/gpu:1` + toleration `nvidia.com/gpu Exists:NoSchedule`; `fsGroup:1000`. Model
loaded from a baked-in / small download (Qwen3-0.6B). Validated identically on L4 + B200 with a
1-request F=3 smoke (10 stored events, 0 errors) — S/B scale that to F=16.

**(b) NATS+etcd multi-worker — used for the effectiveness/parity runs A, A′ & the full at-scale run
(4 workers, Qwen3-32B).** File-discovery doesn't span pods, so multi-worker uses standard operator-free
dynamo discovery:
- **In-namespace NATS + etcd** (own instances, *not* another namespace's — that cross-pollutes
  discovery): NATS `nats:2.11.4` with `jetstream{}` + `max_payload 15 MiB`; etcd
  `bitnamilegacy/etcd:3.5.18` single-node, `ALLOW_NONE_AUTHENTICATION=yes`.
- **1 frontend Pod + N worker Pods**, each with `NATS_SERVER=nats://nats.<ns>.svc…:4222` +
  `ETCD_ENDPOINTS=http://etcd.<ns>…:2379`. `DYN_DISCOVERY_BACKEND` defaults to `etcd`; NATS is the
  request + event plane (no `file`/`tcp`/`zmq` overrides). vLLM still ZMQ-publishes its KV events on
  `tcp://*:20080`; the dynamo worker subscribes and re-publishes to the frontend over NATS.
- **Qwen3-32B from the RWX `model-cache` PVC**, mounted at **`/root/.cache/huggingface`** (where
  `dynamo_llm::hub` looks) + `HF_HUB_OFFLINE=1` — mounting at the wrong path (or omitting offline)
  makes `fetch_model` try HuggingFace and **429**. Per-worker GPU `nvidia.com/gpu:1`, mem limit
  **300 Gi** (the 150 GiB pinned CPU tier + ~61 GB model page-cache OOMs a smaller limit), and
  **podAntiAffinity 1-per-node** (don't co-locate the big pinned allocations).
- Workload driven by the vLLM multi-turn client (`benchmarks/multi_turn/`) run **inside the frontend
  Pod** vs `localhost:8600` (not port-forward — avoids a bandwidth bottleneck on perf numbers).

> CPU-tier sizing matters at F=16: one 256-token offload block ≈ 28 MB (Qwen3-0.6B) / ≈ 64 MB
> (Qwen3-32B) — far larger than F=3's ~5 MB. Too small a `cpu_bytes_to_use` → `prepare_store` returns
> `None` (`"cannot store blocks"`, zero CPU stores). S used 2 GiB, B 256 MiB (to force eviction),
> A/A′/full 150 GiB.

---

## 4. Correctness  *(verify-once; not part of any sweep)*

### 4.1 F=16 reconciliation — aligned prefix  *(run S — DONE 2026-06-23; A re-confirms at scale)*

The F=16 analogue of Stage-1 step7's F=3 result (`685 = 331 GPU + 354 CPU`, n_hashes=3).

**Run S** — Qwen3-0.6B, 1-Pod bypass on B200 `prctr-7wrxm`, image `…:stage2-reviewfix-20260619`,
offload `block_size=256` (F=16), 8 distinct ~2900-token prompts. ZMQ side-capture decoded with
`decode_capture.py --factor 16 --block-size 16`; frontend `/metrics` for applied reconciliation.

| check | expected | observed (run S) |
|---|---|---|
| CPU `BlockStored` `n_hashes` | `16` | **`16` — all 88 CPU stores (histogram `{16: 88}`)** |
| CPU `BlockStored` `len(token_ids)` | `256` | **`256` — shape check 2 PASS (`tok_len == 16×16`, `block_size==16`, all events)** |
| CPU placeholders (`block_size==0` / empty tokens) | `0` | **`0`** |
| wire stored total | `= GPU + CPU` | **`104 = 16 GPU + 88 CPU`** |
| `kv_cache_events_applied{stored,ok}` | `= wire stored`, 0 errors | **`104`, all error buckets `0`** |
| `kv_cache_events_applied{removed,ok}` | `> 0`, 0 errors | **`1196` (1194 GPU + 2 CPU), all error buckets `0`** |
| CPU `BlockRemoved` constituent hashes | match stored chunk metadata | **PASS — check 4 (no remove of an un-stored hash); removed in whole-chunk groups `{64, 176}` = 4 & 11 chunks ×16** |

Corroborating: `vllm:kv_offload_total_bytes_total{GPU_to_CPU}=2.58 GB`, `kv_offload_size_count{GPU_to_CPU}=8`
(one offload op/request) — the bytes actually moved, ~29.4 MB per 256-token block. Parent chain check 3
PASS (`parent=None` on exactly the 8 distinct request heads); the decoder's "out-of-order arrival"
note is the expected `complete_store` set-iteration artifact, not a failure.

> **F=16 offload-sizing finding (harness, not a contract property).** Offload is *proactive during
> prefill* — `_build_store_jobs` queues `num_offloadable_tokens // 256` blocks per scheduled request,
> not on GPU eviction. At F=16 one offload block ≈ **28 MB** for a 0.6B model (vs ~5 MB at F=3), so the
> Stage-1 smoke's `cpu_bytes_to_use=128 MB` (~4 blocks) is **too small** — a single ~2900-token request
> wants to store 11 blocks at once → `prepare_store` returns `None` (`"cannot store blocks"`, zero CPU
> stores). Raising the CPU tier to **2 GiB** (~73 blocks) makes stores succeed; 8 prompts (~88 blocks)
> overflow it so the tier also evicts → CPU `BlockRemoved`. This is a *workload/sizing* knob for a tiny
> synthetic smoke; the **event shape is unaffected** by it. (`--num-gpu-blocks-override 288` was also
> set to add GPU eviction realism but is *not* required for CPU offload.)

### 4.2 Overlap — no premature eviction  *(run B — DONE 2026-06-23, PASS)*

The chunk-specific hazard. **Run B** — same 1-Pod bypass, image/config as S but CPU tier shrunk to
**256 MiB** (~9 F=16 blocks) to force eviction. Workload: a **shared, chunk-non-aligned prefix**
(~600 tok, `% 256 = 88`, byte-identical via `PYTHONHASHSEED=0`) + per-request distinct ~600-tok suffix
across 8 requests → the boundary chunk (tokens 512–767) straddles shared→unique, so each sibling's
chunk-2 `BlockStored` re-announces the shared constituent block hashes. (Fully-aligned shared chunks 0/1
dedupe by key and are *not* re-announced — confirmed.)

- **Duplicate announcements observed on the wire:** `re_announced = 9` duplicate per-hash stores
  (decoder checks 5/6); CPU removes arrive in whole-chunk groups `{16:1, 32:4}` (144 hashes). Shape
  intact under overlap: all **18 CPU stores `n_hashes=16`**, zero placeholders (checks 1–2 PASS).
- **Invariant held — a block is retained while a sibling chunk still references it:** the decoder's
  `EventDedupFilter` simulation (ref-count per `(tier, hash)`, forward a remove only at refcount→0)
  **suppressed 12 non-final removes**, forwarded 132 → `applied_removes == wire_removes − 12`. At the
  producer, `removes-of-already-removed = 0` (check 5) — no erroneous double-remove.
- **`BlockNotFound` / missing-parent / failed-apply / warning logs: `0`** — frontend
  `kv_cache_events_applied` error buckets all `0` (block_not_found / parent_block_not_found /
  invalid_block, across stored/removed/cleared); no apply errors in the frontend log.
- **Apply stream reconciles after real `EventDedupFilter` semantics:** the live Dynamo worker-publisher
  path (ai-dynamo/dynamo#8012) applied `stored ok = 34` (18 CPU + 16 GPU events) and
  `removed ok = 114` (5 CPU + 109 GPU events) with **0 errors** — the duplicate announcements were
  absorbed without any net over- or under-eviction. Offload real: `vllm:kv_offload_total_bytes_total
  {GPU_to_CPU} = 504 MB` (8 ops) — exceeds the 256 MB tier, so CPU eviction genuinely occurred.

> Granularity note: the decoder sim counts per **hash** (12 suppressed); the frontend counts per
> **event** (114 removes applied). Both confirm the invariant — the sim proves the ref-count logic
> suppresses non-final shared-hash removes, the frontend's all-zero error buckets prove the real path
> applies cleanly with no premature-eviction fault.

### 4.3 Edges / scope

- **Trailing partial chunk** (prompt not a multiple of `F × block_size`) — **incidentally validated**:
  S's ~2900-tok and A's ~16K-tok prompts are *not* 256-multiples, yet every CPU `BlockStored` carried
  exactly `n_hashes=16` with **zero placeholders / zero partial (`n_hashes<16`) events**. The connector
  offloads only *full* 256-token chunks (`num_offloadable_tokens // 256`); the partial tail stays in GPU
  until it fills — no partial-chunk event, no error.
- **Sliding-window groups** — **deferred / out of scope.** SWA groups keep the legacy placeholder
  fallback; validating that a *mixed-attention* model doesn't emit self-describing for SWA groups needs
  such a model (the validated Qwen3 is full-attention). Not exercised here; no regression risk for
  full-attention models.
- **`extra_keys`** (multimodal / cache-salt / prompt-embedding) — **out of scope.** The schema has room
  but the path is untested here.

---

## 5. Effectiveness & chunk↔block parity  *(the "overall" comparison)*

Goal: chunk mode (`F=16`) keeps CPU-tier cache useful and routing quality ≈ by-block (`F=1`), at lower
offload I/O. Run on the **NATS+etcd operator-free deploy, 4 workers, Qwen3-32B, 150 GiB CPU tier/worker**,
driving the Stage-1 long-bench (`generate_multi_turn_longbench.json`, 128 conv · 15K prefix · 30–50 turns)
via the vLLM multi-turn client (`-p 8 -k 32 --no-early-stop --seed 0`). **A** = F=16, **A′** = F=1 on the
same image/workload/seed — a fresh same-image by-block baseline (Stage-1 §6 was vLLM 0.21; not reused).
A/A′ are the **bounded `-n 600`** parity pair; a separate **clean full F=16 run** (`-n 2000`, **2000
samples / 83 of 128 conversations complete / 0 failures**, fresh workers, baseline 0) provides the
at-scale §5.1 reuse + §6 reconciliation numbers.

### 5.1 CPU-tier is useful at F=16  *(DONE 2026-06-26, PASS — clean full run)*

Under the 15K-prefix × 30–50-turn workload the GPU KV (108K tokens at `gpu-mem 0.5`) cannot hold the
working set, so prefixes evict and the **CPU tier absorbs the reuse** (clean full F=16 run):

- **External (CPU-tier) prefix-cache hit rate ≈ 78%** per worker (77.6 / 78.1 / 78.9 / 79.0%), vs GPU
  prefix-cache only ~6% — i.e. **most prefix reuse is served from the CPU tier, not GPU**. Uniform
  across all 4 workers (no single worker starves). The bounded A run agrees (per-worker ~73–81%).
- **CPU→GPU reload real and large:** **~1.5–1.6 TB reloaded per worker** (`load_ops ~370–402`/worker,
  vs a cold-start baseline of 0) against ~0.41 TB offloaded out — evicted prefixes are served back from
  CPU on later turns, not recomputed.
- Routing: the KV router keeps each conversation on the worker holding its CPU-tier prefix (the uniform
  per-worker hit rates confirm even distribution).

### 5.2 Parity vs by-block  *(F=16 ↔ F=1, bounded A/A′, identical workload/seed)*

| metric | F=1 (A′, by-block) | F=16 (A, chunk) | note |
|---|---|---|---|
| samples / failures | 600 / 0 | 600 / 0 | both clean |
| TTFT (ms, avg) | 425 | 426 | **parity** |
| TPOT (ms, avg) | 21.1 | 21.0 | **parity** |
| per-request latency (ms, avg) | 1459 | 1451 | **parity** |
| input tokens (avg) | ~16.0K | ~16.0K | same workload |
| **KV store events applied (frontend)** | **263,492** | **10,879** | **F=16 = ~24× fewer** |
| KV removed events applied | 1,105,272 | 550,382 | F=16 ~2× fewer |
| apply errors (block_not_found / invalid / parent) | **0** | **0** | clean reconciliation both |
| CPU-tier prefix-cache hit rate | (workers torn down post-run; not captured) | ~77% | effectiveness shown on A (§5.1) |

> The point is not "chunk is faster" — it's "chunk **preserves serving performance** (TTFT/TPOT/latency
> all within noise) **while cutting the offload *store*-event volume ~24×**" (10.9K vs 263K store events).
> Note the scope: the **removed**-event row is only ~2× fewer, so **total** applied KV events drop ~2.4×,
> not 24× — the 24× is specifically the store-event row. Zero reconciliation errors at either factor. The
> ~24× exceeds the naive 16× because by-block also re-announces each block on overlap/reuse, while chunk
> mode batches `F` hashes per event.
> **Honest gaps:** A′'s per-worker CPU-tier hit-rate and routing quality were **not captured** (workers
> cleaned up before snapshot) — so for A′ only *serving-performance* parity (TTFT/TPOT/latency) is
> measured; whether F=1 reuses the CPU tier comparably is **unmeasured** (matched latency is consistent
> with it but does not prove it). Runs are bounded `-n 600` (≈600 turns / ~11–12 of 128 conversations),
> not the full sweep; 4 workers, not 8.

---

## 6. Conclusion

Stage 2 self-describing contract is **validated at production `F=16`**:
- **§4.1 shape + reconciliation** (run S): every CPU `BlockStored` carries `n_hashes=16` / `token_ids=256`,
  zero placeholders; the frontend applied 100% of stored/removed events with **0 errors** — and the clean
  full F=16 run reconciled **104K stored + 5.64M removed events at scale, still 0 errors**.
- **§4.2 overlap / no premature eviction** (run B): non-aligned sibling chunks re-announce shared block
  hashes (`re_announced > 0`); `EventDedupFilter` suppresses the non-final removes and the real Dynamo
  path applies with **0 errors** — the duplicate hazard is absorbed, no over/under-eviction.
- **§5 effectiveness + parity** (runs A / A′): chunk mode delivers **identical serving performance** to
  by-block (TTFT/TPOT/latency within noise) while the CPU tier serves **~78% of prefix reuse** and the
  **store-event volume drops ~24×** (10.9K vs 263K; the removed-event row is ~2× fewer, so *total* applied
  KV events ≈2.4× fewer) — with **0 reconciliation errors at both factors**.

Net: the F=16 chunk contract is correct, the overlap hazard is handled, and chunk mode preserves routing
quality + serving perf while cutting offload store-event volume ~24× (total KV events ~2.4×). **Scope
caveats:** SWA groups keep the placeholder fallback (out of scope — needs a mixed-attention model);
`extra_keys` (multimodal / cache-salt) untested; the A/A′ **parity** pair is bounded `-n 600`, the
at-scale §5.1/§6 run completed **83/128 conv**, all on **4 workers** (not 8); **A′'s per-worker CPU-hit% /
routing quality is unmeasured** (its workers were torn down before snapshot — only A′ serving-latency
parity is measured; the clean full F=16 run supplies the F=16 per-worker CPU-hit). See §5.2 honest-gaps.

---

## Appendix — Validation run checklist

Common: GPU block-size 16, ZMQ KV events, `self_describing_kv_events=true`. **Model:** A/A′/Full use
`Qwen/Qwen3-32B` (`gpu_memory_utilization=0.50`); the S/B correctness runs use `Qwen3-0.6B` (event shape
is model-independent). **Dropped** the unpatched/broken-router baseline (Stage-1 §6 owns it; #43468
merged) and the self-describing-OFF A/B (redundant). **Main run mirrors Stage-1** (same long-bench
workload + 150 GiB pool); only the overlap check uses a special shape.

| run | F | workload | workers | CPU pool | proves | doc § |
|---|---:|---|---|---|---|---|
| **S** ✅ | 16 | 8 distinct ~2900-tok prompts (Qwen3-0.6B) | 1 | **2 GiB** (see sizing note) | F=16 event **shape** (n_hashes=16, token_ids=256, zero placeholder) + reconciliation — **DONE 2026-06-23, PASS** (88 CPU stores all n_hashes=16; applied 104 stored / 1196 removed, 0 errors) | 4.1 |
| **B** ✅ | 16 | shared non-aligned prefix (~600 tok, `% 256 = 88`) + 8 distinct suffixes | 1 | **256 MiB** (forces eviction) | overlap / no premature eviction — **DONE 2026-06-23, PASS** (re_announced=9; sim suppressed=12 non-final removes; frontend 0 errors) | 4.2 |
| **A** ✅ | 16 | **Stage-1 long-bench** (`generate_multi_turn_longbench.json`: 128 conv · 15 K conv-prefix · 30–50 turns), bounded `-n 600 -k 32` | **4** | 150 GiB | reconciliation-at-scale + CPU-tier effectiveness + perf — **DONE 2026-06-24, PASS** (CPU-hit ~77%; 10.9K stored evts / 0 err; TTFT 426/TPOT 21/lat 1451) | 4.1 / 5 |
| **A′** ✅ | 1 | same as A (bounded `-n 600 -k 32`, same seed) | **4** | 150 GiB | same-image by-block parity baseline — **DONE 2026-06-24, PASS** (263K stored evts = 24× A / 0 err; TTFT 425/TPOT 21/lat 1459 = parity) | 5.2 |
| **Full** ✅ | 16 | Stage-1 long-bench, `-n 2000 -k 32 --no-early-stop` (**83/128 conv, 2000 samples, 0 fail**) | **4** | 150 GiB | at-scale CPU-tier effectiveness + reconciliation — **DONE 2026-06-26, PASS** (CPU-hit ~78%; **104K stored + 5.64M removed evts / 0 err**; ~1.6 TB reload/worker) | 5.1 / 6 |

**Execution order (as run):** **S + B first** on the cheap 1-Pod bypass (§3a), **then A + A′ + the full
at-scale F=16 run** on the NATS+etcd 4-worker deploy (§3b). **STATUS: all runs ✅ DONE + PASS** (S/B
2026-06-23, A/A′ 2026-06-24, full F=16 2026-06-26). The multi-worker topology decision landed on **(b) NATS+etcd**
(file-discovery doesn't span pods); 4 workers (not the originally-planned 8) — a de-risked first pass.
CPU-tier sized at 150 GiB/worker (Qwen3-32B per-block ≈ 64 MB).

**Why A reuses Stage-1's workload, not a small one:** Stage-1 deliberately sized the workload large
(15 K × ~40 turns → effective ISL 15–21 K) so the **GPU** KV (108 K tokens at `gpu-mem 0.5`) overflows
→ real GPU eviction → offload to CPU + reload. A small workload would understate the effect. The 150 GiB
CPU pool (~614 K tokens at ~64 MB/256-tok chunk) is *large enough to hold* what GPU evicts — so it's GPU
pressure, not CPU-pool overflow, that drives the offload/reload observed in the full run (~0.41 TB out /
~1.6 TB reloaded per worker; CPU-tier eviction is minimal). Reconciliation is read from the frontend
`/metrics` `kv_cache_events_applied` counters (per-block event *shape* comes from the S/B ZMQ captures).

**Why B is still separate:** the overlap hazard needs a **shared, chunk-non-aligned prefix across
requests**; Stage-1's long-bench has *no* shared cross-conversation prefix (reuse is intra-conversation
stickiness), so it cannot exercise sibling-chunk duplicates. B is a small single-worker correctness
run with a shared `% 256 ≠ 0` prefix + eviction pressure.

**Parity uses a fresh same-image F=1 run (A′), not Stage-1's published numbers.** Stage-1 §6 was
vLLM 0.21; the stage2 image is 0.22.1rc + #43468, so reusing those numbers is version-mismatched.
Ran **A′ = F=1 on the same stage2 image, same 4 workers, same workload + seed** as A — clean
apples-to-apples (only the factor differs). Stage-1 §6 F=1 is at most a rough cross-check.

**Deploy used (multi-worker, no Grove):** option **(b) NATS + etcd + 4 worker Pods** (standard
operator-free dynamo, closest to Stage-1) — see §3b for the full recipe + gotchas (HF-cache path/offline,
300 Gi mem limit, 1-per-node anti-affinity). The shared-RWX-PVC file-discovery alternative (a) was not
used. S/B stayed on the 1-Pod bypass (§3a).

**Evidence per run:** Pod/manifest YAML + image digest; worker + frontend logs; frontend `/metrics`
before/after; request + error counts. **S & B also:** ZMQ side-capture (`tcp://localhost:20080`)
decoded with the step7 decoder (the per-block event-shape check); A/A′/Full reconciliation comes from
the frontend `/metrics` counters + per-worker offload/prefix-cache metrics. **Pass criteria:** tables in
§4.1 / §4.2 / §5 + zero lower-tier `BlockNotFound` / failed-apply / warnings.
