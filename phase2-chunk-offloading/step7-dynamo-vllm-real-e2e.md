# Phase 2 · Step 7 — Real dynamo serve + vLLM e2e: chunked offload KV events, metrics-verified

> **Step 5** picked the wire shape (a chunk fans out as its constituent per-block hashes).
> **Step 6** found and characterized the overlapping-chunk hazards; per the team decision the
> producer ships **plain fan-out** (no vLLM-side refcount — that variant is archived on
> `feature/offloading-events-exactly-once`) and consumers deduplicate. Dynamo's standard
> deployment already does this: the worker-side publisher runs an `EventDedupFilter`
> (ai-dynamo/dynamo#8012; store increments a per-(dp_rank, tier) refcount, a remove passes only
> when the count hits 0).
> **Step 7 (this doc):** prove the whole thing on a **real single-GPU dynamo serve + vLLM
> stack** — not a replay: real engine, real publisher (filter in the path), real KV router,
> real LRU evictions — with the router-side `kv_cache_events_applied` counters reconciling
> **exactly** against a sidecar capture of the vLLM wire.

## TL;DR

| direction | vLLM wire (sidecar ZMQ capture, same run) | router applied (`kv_cache_events_applied`) |
|---|---|---|
| stored  | GPU 331 + CPU 354 = **685** (every CPU event `n_hashes=3`, zero placeholders) | **685** (`status="ok"`) |
| removed | CPU **24** (real 128 MB-pool LRU evictions; hash counts all multiples of 3) | **24** (`status="ok"`) |
| errors  | — | 0 warnings, 0 `block_not_found` |

Chunked offloading (`factor=3`) + opt-in self-describing events flow vLLM → ZMQ → dynamo worker
`KvEventPublisher` (listener → normalizer → **EventDedupFilter** → tier-routed indexers +
event-plane forward to the frontend KV router), and the counters match the wire **event for
event** in both directions.

## Stack under test

| component | version / branch |
|---|---|
| vLLM | PR [#43468](https://github.com/vllm-project/vllm/pull/43468) branch (`events.py` `OffloadingEventsTracker`, opt-in, plain fan-out), rebased on main `f712fd0d7` |
| dynamo | `feat/kv-router-cpu-medium-alias` @ `db0ec35619` — includes the `medium="CPU"` → `HostPinned` alias, lower-tier `kv_cache_events_applied` counting (`5b7725f4`), and the metrics wiring into lazily created lower-tier indexers (`db0ec356`, found by this very e2e — see Pitfalls) |
| python binding | `maturin develop --uv` from that dynamo checkout (`ai-dynamo-runtime 1.3.0`) |
| model / GPU | `Qwen/Qwen3-0.6B`, 1× NVIDIA L4 |
| external services | **none** — file-based discovery, TCP request plane, ZMQ event plane |

## Topology and event path

```
                         ┌────────────────────────────── worker process ─────────────────────────────┐
  vLLM EngineCore        │  dynamo.vllm                                                               │
  OffloadingConnector ──ZMQ(tcp://127.0.0.1:20080, msgpack)──► ZmqEventListener ─► ZmqEventNormalizer │
  (chunk=48tok, factor=3,│        (received/accepted counters)        ("CPU"→HostPinned, local hashes)│
   self-describing       │                                  ─► event processor: batching by tier      │
   BlockStored/Removed)  │                                     + EventDedupFilter (#8012)             │
                         │                                  ─► LocalKvIndexer (tier-routed:           │
                         │                                       GPU→radix, CPU→LowerTierIndexer)     │
                         │                                       └── kv_cache_events_applied ◄─ THE   │
                         │                                  ─► EventPlanePublisher ──ZMQ──┐    METRIC │
                         └────────────────────────────────────────────────────────────────┼──────────┘
  sidecar capture.py ◄───── second SUB on the same vLLM PUB (exact wire bytes)            │
                         ┌── frontend process ──────────────────────────────────────────┐ │
                         │  dynamo.frontend --router-mode kv                            │ │
                         │  RouterEvent subscriber (local_indexer mode) ◄───────────────┼─┘
                         │  Indexer::apply_event (tier-routed: GPU→primary radix,       │
                         │                        HostPinned→LowerTierIndexers)         │
                         │  HTTP :8600 (OpenAI API + /metrics, incl. forwarded          │
                         │              backend kv_metrics)                             │
                         └────────────────────────────────────────────────────────────┘
```

## Configuration that matters

| knob | value | why |
|---|---|---|
| `--block-size` (vLLM) | `16` | GPU/hash block size; also the per-block `block_size` on CPU events and the publisher's local-hash block size |
| `kv_connector_extra_config.block_size` | `48` | offloaded chunk size → `block_size_factor = 3` (the chunk mode under test) |
| `kv_connector_extra_config.cpu_bytes_to_use` | `134217728` (128 MB) | small CPU pool so the run produces **real LRU evictions** (the remove path is exercised, 24 events) |
| `kv_connector_extra_config.self_describing_kv_events` | `true` | the opt-in from PR #43468; without it CPU events keep the legacy placeholder payload |
| `--kv-events-config` | `{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}` | **must be passed explicitly** — `dynamo.vllm` does *not* enable vLLM KV events on its own (see Pitfalls) |
| `DYN_DISCOVERY_BACKEND=file`, `DYN_REQUEST_PLANE=tcp`, `DYN_EVENT_PLANE=zmq` | — | zero-dependency local mode: no etcd, no NATS, no docker |
| `PYTHONHASHSEED=0` | — | deterministic KV event IDs (matches the launch examples) |
| `--router-mode kv` (frontend) | — | KV-aware router; this deployment uses the "local_indexer mode" ZMQ event-plane subscription |
| `--gpu-memory-utilization 0.3`, `--max-model-len 4096`, `--enforce-eager` | — | quick startup; 3039 GPU blocks (large enough that the GPU tier does not evict — GPU removed=0 by design, isolating the CPU remove path) |

## Launch procedure

Everything is in [`dynamo-e2e/run_e2e_capture.sh`](dynamo-e2e/run_e2e_capture.sh). The essential
sequence:

```bash
export PYTHONHASHSEED=0 DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=$WD/dynstore \
       DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq

# 1) frontend: OpenAI API on :8600 + KV router
python -m dynamo.frontend --router-mode kv --router-reset-states --http-port 8600 &

# 2) worker: vLLM with chunked offloading + opt-in events + explicit kv-events-config
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B --block-size 16 --enforce-eager \
    --gpu-memory-utilization 0.3 --max-model-len 4096 \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
        "kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec",
        "cpu_bytes_to_use":134217728,"block_size":48,"self_describing_kv_events":true}}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events",
        "endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# 3) wait for health
curl -sf http://localhost:8600/v1/models | grep -q Qwen

# 4) sidecar capture: a second SUB on the same vLLM PUB records the exact wire
python capture.py "tcp://localhost:20080" sidecap.jsonl &

# 5) traffic: 16 multi-turn conversations, 256-token shared prefix
#    (vLLM's benchmark_serving_multi_turn.py with gen_small.json, 4 clients)
python benchmark_serving_multi_turn.py -i gen_small.json \
    --model Qwen/Qwen3-0.6B --url http://localhost:8600 \
    --num-clients 4 --max-active-conversations 8

# 6) metrics snapshot, then reconcile
curl -s localhost:8600/metrics > frontend_metrics.txt
python dynamo-e2e/decode_capture.py sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied frontend_metrics.txt | grep -v ' 0$'
```

## Result detail (one round of traffic)

Wire, decoded from the sidecar capture:

```
[CPU] stored_events=354 (placeholder=0)   n_hashes histogram: {3: 354}
      removed_events=24                   n_hashes all multiples of 3
[GPU] stored_events=331                   removed_events=0  (GPU pool never filled)
```

Router counters (frontend `/metrics`, the `dynamo_component="backend"` series is the worker's
LocalKvIndexer forwarded over the `kv_metrics` event-plane topic):

```
kv_cache_events_applied{event_type="stored", status="ok"}  685   # = 331 GPU + 354 CPU
kv_cache_events_applied{event_type="removed", status="ok"}  24   # = the 24 real CPU evictions
kv_cache_event_warnings                                      0
```

Listener-stage counters confirm nothing was dropped on the way in
(`kv_publisher_zmq_events_total`: `received == accepted` for both stored and removed; zero
filtered, zero conversion failures), and the worker log has zero
`Failed to apply event to local indexer` lines.

`applied` counts **events**; the dedup filter's per-hash interception inside a batch does not
change event counts. The hash-level filter semantics (a shared hash's non-final removal is
held back) are covered by the filter's own unit tests and by the simulation in
[`decode_capture.py`](dynamo-e2e/decode_capture.py) check 7.

## Pitfalls hit on the way (read before reproducing)

1. **Stale python binding.** The editable `dynamo._core` (`maturin develop`) is a build
   artifact; after moving the dynamo checkout forward the worker crashed with
   `DistributedRuntime.__new__() got an unexpected keyword argument 'event_plane'`.
   Re-run `maturin develop --uv` in `lib/bindings/python` whenever the Rust tree changes.
2. **`--kv-events-config` must be explicit.** Without it `dynamo.vllm` logs
   `Using kv_events_config ... None (use_kv_events=False)` and **no KV events exist at all**
   (router counters stay 0 while serving works fine).
3. **Know which counter you are reading.** `kv_cache_events_applied` with
   `dynamo_component="backend"` is the **worker LocalKvIndexer's** counter (forwarded to the
   frontend's `/metrics`). Before `db0ec356`, the lazily created lower-tier indexers were
   constructed without a metrics handle, so CPU traffic was applied correctly but **invisible**
   (counter showed 332 / 0 instead of 685 / 24) — which briefly looked like a broken event
   path. The wire capture + listener-stage counters are what disambiguate "not arriving" from
   "not counted".
4. **Frontend metrics live on the HTTP port** (`:8600/metrics`), not on `DYN_SYSTEM_PORT`.

## Reproduce

```bash
cd /home/changg/workspace/.tmp/dyn_e2e          # or copy dynamo-e2e/run_e2e_capture.sh anywhere
bash run_e2e_capture.sh                          # launches stack, traffic, capture, snapshots
python dynamo-e2e/decode_capture.py sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied frontend_metrics_cap.txt | grep -v ' 0$'
```

Prereqs: the vLLM PR branch installed editable in the venv, the dynamo branch built
(`maturin develop --uv`), `Qwen/Qwen3-0.6B` in the HF cache, one free GPU, and the multi-turn
benchmark client from `.tmp/llmd_4way/multi_turn_client` (vLLM's
`benchmark_serving_multi_turn.py` + `gen_small.json`).

Artifacts of the recorded run live in `/home/changg/workspace/.tmp/dyn_e2e/`: `worker.log`,
`frontend.log`, `sidecap.jsonl`, `frontend_metrics_*.txt`, `bench_*.log`.
