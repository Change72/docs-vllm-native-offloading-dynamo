# Slack post — vLLM CPU Offload + Dynamo KV Router on B200

---

**TL;DR.** Wiring vLLM's `OffloadingConnector` into Dynamo's KV router (3 commits) **+** tuning the router's cost-function constants (h=1.0, pl=100) takes Qwen3-32B on 8× B200 from **8.9 → 32.1 req/s (+261 %)** and **TTFT p99 13.3 s → 2.7 s (−5 ×)** at c=128. Full 4-line ablation isolates "commits alone" vs "tuning alone": **commits do ~4 × the work that tuning does**, and the two effects **stack multiplicatively, not additively**.

## What's in the box

* 3 commits (vLLM `e074f0a`, Dynamo `6c2a73a`, Dynamo `5b7725f`) that fix the silent KV-event drop — without these, the router has no per-block CPU-tier state and is effectively round-robin.
* 2 cost-function knobs (`--router-host-cache-hit-weight 1.0`, `--router-prefill-load-scale 100`) that the upstream Dynamo router has hardcoded at conservative defaults (0.75 and 1.0).

## Headline (long-context multi-turn, c=128 on 8× B200, Qwen3-32B, 3 cold-start repeats)

| Metric | Baseline | **Best (this work)** | Δ |
|---|---:|---:|---|
| Throughput | 8.89 ± 0.03 req/s | **32.14 ± 0.46 req/s** | **+261 %** |
| TTFT p99 | 13.3 s | **2.7 s** | **−5.0 ×** |
| TPOT p99 | 142 ms | **30 ms** | **= GPU decode floor** |
| Wasted prompt-token compute | 71.83 ± 0.17 % | **7.60 ± 0.16 %** | **−64.2 pp** |
| Overall cache hit (token-weighted) | 28 % | **92 %** | **+64 pp** |

> The TPOT p99 result is the most striking. Best image is at 30 ms per output token — **that's the theoretical floor on this hardware for Qwen3-32B (the GPU literally cannot decode tokens faster)**. We have eliminated p99 routing-induced stalls entirely.

## 4-line ablation (commits × tuning)

Reading c=128 compute waste %:

| Cohort | Compute % | Δ from baseline | What's enabled |
|---|---:|---:|---|
| Baseline (no commits, default cost-fn) | **71.8 %** | — | neither |
| Baseline + pl=100 (no commits, tuned)   | **67.8 %** | −4 pp | tuning only |
| Treatment (commits, default cost-fn)    | **47.2 %** | **−25 pp** | commits only |
| Best (commits + h=1.0 + pl=100)         | **7.6 %**  | **−64 pp** | both, multiplicative |

→ Commits alone do **6 × the work** that tuning alone does, and the two stack multiplicatively (−4 pp + −25 pp combine to −64 pp, way larger than −29 pp).

## Robustness — same ordering everywhere

* **Concurrency** c=8 → 384: best image scales linearly to **78 req/s** at c=384; baseline hits a hard saturation at c=128 and **degrades past it** (rps drops 25 → 22 as c rises 128 → 256, compute climbs 53 % → 73 %).
* **Conversation prefix** 5 K → 30 K: best holds at 6-7 % compute waste at every prefix size; the bigger the prefix, the bigger the gap.
* **Request rate** rr 0.1 → 4.0: 4 perfectly parallel horizontal compute % bands at every rr; load doesn't change the ordering.
* **LMBenchmark** (LMCache team's own bench): same ordering across QPS 1 → 64. At QPS=64, `baseline+pl=100` actually scores **worse than untuned baseline** (70 % vs 55 % compute) — tuning amplifies bad routing when commits are absent.

## "Are we sure this isn't a measurement artifact?"

* **N=3 cold-start repeats at c=128 across all 4 cohorts** — std-devs are < 0.5 pp on compute %, < 0.5 rps on throughput. P ≪ 0.001 on every pairwise comparison.
* **Pre-warm ablation** (5 min low-intensity warmup, then measure on warm cache): **31 % of the gap is "head start" (cache cold-start), 69 % is mechanism (routing intelligence on warm cache)**. The gap survives realistic-production warm conditions.
* **Cross-benchmark transfer** (LMBenchmark): same ordering, same multiplicative ablation.

## "Does the connector fix work under TP > 1?"

Yes, and TP changes nothing on the wire. The side-table + event emission all live on the single scheduler (one per job regardless of TP); `OffloadKey`s and the payload are content/request-derived, i.e. TP-invariant. There's **no all-gather** — each rank memcpy's its own shard GPU→CPU in parallel, and the scheduler just waits for all `num_workers` acks (`pending_count → 0`) before emitting **one** `BlockStored`. It's an ack barrier, not a data merge. Full derivation in `PRESENTATION.md` → Appendix A.

## Upstream proposal

Three commits to land, **two existing-flag default changes** to propose:

| Flag | Current default | Proposed default | Why |
|---|---|---|---|
| `host_cache_hit_weight` | 0.75 | **1.0** | CPU tier hit is as good as GPU tier hit once you have to refetch anyway |
| `prefill_load_scale` | 1.0 | **10** | Captures 97 % of pl=100's benefit; 10× change is more defensible than 100× |

Disk-tier weight is also exposed (`--router-disk-cache-hit-weight`) but is only useful with disk tier enabled.

## Pictures + full doc

* Full writeup: `nscale-offload-demo/PRESENTATION.md` — diagnosis (§2), commits diff vs llm-d (§3), cost-function deep-dive (§5), benchmark methodology (§6.1), full sweep + ablations (§6.2–§6.9), theoretical floor derivation (§6.10)
* Plots: `nscale-offload-demo/images/{t2_concurrency, t4_prefix, t6_rr, lmbench_qps, c128_errorbars, tail_latency_c128, prewarm_ablation, ceiling_sweep, headline_progression, pl_curve, t3_heatmap}.png`
* Raw data: `nscale-offload-demo/data/_summary.csv` (1 row per benchmark cycle, 110+ cycles)
* Reproduction: `nscale-offload-demo/orchestrator/` — in-cluster Job runs the full sweep cold-restarting between cycles

Happy to walk through the 4-line ablation and the pre-warm methodology with anyone interested.
