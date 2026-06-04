# vLLM × Dynamo CPU Tier Integration — Diagnosis, Fix, Benchmarks

> ## ➡️ STAGE 3 (latest): chunk offloading is now router-matchable
> Chunked CPU offload (`block_size_factor > 1`) used to be invisible to KV-cache-aware routers.
> A **vLLM-side-only** fix (now on the existing PR, branch
> `bugfix/offloading-connector-blockstored-payload` @ `54044fad4`) makes it **fully matchable on
> llm-d — 94.3% coverage, 68/68 contiguous, store + remove**. Status, readiness matrix, and the
> remaining plan (Dynamo 1:many, `extra_keys`, sliding-window/SSM) are the **plan of record**:
> **→ [`STAGE3-CHUNK-OFFLOADING-PLAN.md`](STAGE3-CHUNK-OFFLOADING-PLAN.md)**

This repo contains the writeup, plots, raw data, deployment artifacts, and orchestration
scripts for the nscale B200 work on enabling vLLM's `OffloadingConnector` (CPU KV-cache
tier) to be actually visible to Dynamo's KV-router, plus the cost-function tuning that
extracts the remaining performance on top.

Headline result at production saturation (Qwen3-32B, 8× B200, long-context multi-turn,
c=128, 3 cold-start repeats):

| Metric | Baseline | **Best** | Δ |
|---|---:|---:|---|
| Throughput | 8.89 ± 0.03 req/s | **32.14 ± 0.46 req/s** | +261 % |
| TTFT p99 | 13.3 s | **2.7 s** | −5.0× |
| TPOT p99 | 142 ms | **30 ms** | = GPU decode floor |
| Prompt-token compute waste | 71.83 ± 0.17 % | **7.60 ± 0.16 %** | −64.2 pp |

Full write-up in [`PRESENTATION.md`](PRESENTATION.md); Slack-friendly TL;DR in
[`SLACK_SUMMARY.md`](SLACK_SUMMARY.md).

---

## Layout

```
.
├── PRESENTATION.md            Full writeup (diagnosis → fix → benchmarks → upstream PR proposal)
├── SLACK_SUMMARY.md           Short TL;DR
├── images/                    All plots referenced by PRESENTATION.md
├── data/
│   ├── _summary.csv           One row per benchmark cycle; columns documented in PRESENTATION §6.1
│   └── ph3-lmbench-rps-corrected.csv   LMBench RPS reparsed from raw bench.log (corrects a polling bug)
├── scripts/
│   ├── make_presentation_plots.py    Regenerates §6.x plots (t1..t6, headline, pl_curve, lmbench_qps)
│   ├── analyze_phase5.py             Regenerates §6.3.2 (error bars), §6.6.1 (4-line LMBench), §6.8, §6.9
│   ├── analyze_tail_latency.py       Regenerates §6.3.1 tail-latency CDFs from stats.json
│   └── reparse_ph3_lmbench.py        Counts `finished one request` lines in LMBench logs → valid RPS
├── deploy/                    Canonical DynamoGraphDeployment YAMLs
│   ├── baseline.yaml          BASELINE image (no commits, default cost-fn)
│   ├── best.yaml              BEST image (commits, h=1.0, pl=100, disk=0.25 CLI flags)
│   └── bench-client.yaml      Benchmark client pod (multi-turn + LMBench)
├── workloads/                 Workload generator configs for benchmark client
│   ├── generate_multi_turn_longbench.json    Primary (used by §6.2, §6.3, §6.6.1, §6.7)
│   ├── generate_multi_turn_saturated.json    Saturation regime (used by phase5b ceiling test)
│   ├── generate_multi_turn_ceiling.json      §6.9 ceiling test (prefix=6K, num_conv=400)
│   ├── cfg_prefix_{5000,25000,30000,40000}.json    §6.5 prefix sensitivity
│   └── … (other intermediate workload configs for reference)
└── orchestrator/              In-cluster Kubernetes Job orchestrator (runs the entire benchmark sweep)
    ├── 01-sa-rbac-pvc.yaml    ServiceAccount + RBAC + PVC for results
    ├── 03-pod.yaml            Master sweep pod (T1..T6 full sweep)
    ├── 06-phase3-pod.yaml     Phase 3 pod (denser pl curve + 30K/40K prefix)
    ├── 07-phase4-pod.yaml     Phase 4 pod (4-line ablation across T2/T4/T6/LMBench)
    ├── 08-phase5-pod.yaml     Phase 5 pod (statistical reps + pre-warm + ceiling + LMBench RPS fix)
    └── run-{sweep,phase3,phase4,phase5,retry,rr-extra}.sh    Phase scripts
```

## Three commits that this work depends on

* **vLLM `e074f0a`** — populate full `BlockStored` event payload (`token_ids`, `block_size`, `parent_block_hash`) from a side-table maintained by `OffloadingConnectorScheduler`
* **Dynamo `6c2a73a`** — accept `"CPU"` as an alias for the `HostPinned` tier so vLLM events match the indexer's medium-name expectations
* **Dynamo `5b7725f`** — surface `kv_cache_events_applied` counters on Prometheus so the new CPU-tier traffic is observable end-to-end

See `PRESENTATION.md` §2 for the diagnosis, §3 for the diff against `llm-d` and commit
details, §5 for the cost-function deep-dive, §6 for the benchmarks (incl. §6.4 for the
upstream-default recommendation and §6.10 for the theoretical-floor derivation).

## Reproducing the sweep (high level)

1. Apply the SA/RBAC/PVC: `kubectl apply -f orchestrator/01-sa-rbac-pvc.yaml`
2. Apply `deploy/bench-client.yaml` (sleeps forever; holds workload generator + bench client code)
3. Copy / mount the `workloads/*.json` into the bench-client pod at `/work/multi_turn/`
4. Apply `orchestrator/08-phase5-pod.yaml` — runs the full Phase 5 sweep, cold-restarts DGDs, captures metrics, writes results to PVC at `/results/`
5. Pull `/results/_summary.csv` and feed to `scripts/analyze_phase5.py` to regenerate the §6.x plots

The orchestrator uses an in-cluster `python:3.11-slim` pod that bootstraps `kubectl` via
`curl`. It deletes / recreates the `DynamoGraphDeployment` between every cycle to give
each cohort a fresh cold start, then runs the benchmark via `kubectl exec` into the
bench-client pod. See `orchestrator/run-phase5.sh` for the full state machine.

## Two-line manual smoke test

```bash
# Stand up baseline
kubectl apply -f deploy/baseline.yaml
kubectl apply -f deploy/bench-client.yaml
# … wait for 8 workers + frontend ready …

# Run a 1-shot benchmark from the client
kubectl exec bench-multi-turn -n changg-dynamo -- bash -c "cd /work/multi_turn && \
  python3 benchmark_serving_multi_turn.py \
    --model Qwen/Qwen3-32B --served-model-name Qwen/Qwen3-32B \
    --url http://qwen3-32b-offload-router-frontend.changg-dynamo.svc.cluster.local:8000 \
    --input-file generate_multi_turn_longbench.json \
    --num-clients 128 --max-active-conversations 128 \
    --request-rate 0.5 --warmup-step"

# Swap to best
kubectl delete dgd qwen3-32b-offload-router -n changg-dynamo
kubectl apply -f deploy/best.yaml
# … repeat the bench …
```

## FAQ

**"Are these numbers reproducible?"** Yes. §6.3.2 reports 3 cold-restart repeats at c=128
across all 4 cohorts; all std-devs are < 1.5 pp on compute % and < 0.5 rps on throughput.

**"Is the gap from a 'cache head-start' rather than mechanism?"** §6.8 directly tests
this with a pre-warm ablation. Result: 31 % of the gap is cache cold-start (warmup fills
it), 69 % is mechanism (routing decisions on warm cache).

**"Did you cherry-pick a favorable benchmark?"** §6.6 runs the LMCache team's
LMBenchmark with 4-line ablation across QPS 1 → 64. Same ordering everywhere
(baseline < baseline+pl=100 < treatment-default < best).

**"Why not just propose `pl=10` instead of `pl=100`?"** §6.4.1 covers this — `pl=10`
captures ~97 % of the benefit of `pl=100` with a 10× change to the default instead of
100×, easier to land upstream.

**"What's the next bottleneck after this?"** §6.9 ceiling test — best image scales to
78 rps at c=384 (3.6× baseline's c=128 peak) with compute % flat at 6-7 %. Routing is
no longer the limit; GPU compute / network / vLLM decode capacity is.
