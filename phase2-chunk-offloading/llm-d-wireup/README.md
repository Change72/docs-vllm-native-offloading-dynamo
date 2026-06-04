# Step 2 prerequisite — wire vLLM ↔ llm-d KV offload (LMCache), prove data moves

Goal: connect vLLM to **llm-d's KV-offload data plane (LMCache)** and prove, with logs, that
vLLM **offloads KV data into LMCache (store)** and **loads it back out (retrieve)**. Local on L4,
in `.venv`, no EPP/Envoy/K8s. This is the data-path foundation before the Dynamo-vs-llm-d
chunk/routing comparison.

## What this uses

- vLLM `kv_transfer_config = {kv_connector: "LMCacheConnectorV1", kv_role: "kv_both"}` — this is
  exactly llm-d's `CONNECTOR=lmcache-connector` path from
  `guides/tiered-prefix-cache/cpu` (just run locally instead of via Helm/K8s).
- LMCache local CPU tier: `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE=2` (GB),
  `LMCACHE_CHUNK_SIZE=128`.
- Env: `.venv` (vllm 0.21 editable, lmcache 0.4.6 editable clone), CUDA 12.8, NVIDIA L4.

## The driver

`offload_lmcache.py` — three phases (same shape as the native-offload step-1 harness so the two
are comparable): WARM (store P) → EVICT (thrash GPU so P leaves the GPU prefix cache) → REUSE
(request P again → GPU misses → LMCache retrieves). opt-125m, 512-token shared prefix P.

```bash
cd /home/changg/workspace/.tmp/llmd_lmcache   # or this folder
source /home/changg/workspace/.envrc          # exports a short VLLM_RPC_BASE_PATH (see gotcha)
python offload_lmcache.py 2>&1 | tee run.log
grep -E "Storing KV cache for 512|hit tokens: 512|Retrieved 512 out of 512" run.log
```

## Log evidence (L4, clean `.envrc` env)

OFFLOAD (WARM — store P into LMCache CPU):

```
LMCache DEBUG: Storing KV cache for 512 out of 512 tokens (skip_leading_tokens=0) for request 0-...
```

LOAD (REUSE — GPU miss, retrieve P from LMCache CPU back to GPU):

```
LMCache INFO: Reqid: 9-..., Total tokens 516, Inference Engine computed tokens: 0, LMCache hit tokens: 512, need to load: 512
LMCache INFO: [req_id=9-...] Retrieved 512 out of 512 required tokens (from 512 total tokens). size: 0.0176 gb, cost 1.83 ms, throughput: 9.60 GB/s
```

`computed tokens: 0` (GPU evicted) + `hit tokens: 512` + `Retrieved 512/512` = the full prefix was
recovered from LMCache, not recomputed. vLLM↔LMCache offload/load works end to end.

## Gotcha fixed (env)

LMCache's offload/lookup servers bind ZMQ **Unix-domain sockets** under `VLLM_RPC_BASE_PATH`
(defaults to `$TMPDIR`). `.envrc` set `TMPDIR=$WORKSPACE/.tmp`; combined with a 36-char engine UUID
the socket path `"<base>/engine_<uuid>_service_offload_lmcache_rpc_port_<n>"` exceeds the 107-char
Unix-socket limit, so LMCache init **silently fails** (`Skipping post_init`) and **every lookup
returns 0 hits** (store "works" but retrieve never does — easy to misread as a logic bug).

Fix (already applied to `.envrc`): export a short IPC base, leaving `TMPDIR` (caches) untouched:

```bash
export VLLM_RPC_BASE_PATH="/tmp/vllm_rpc"
mkdir -p "$VLLM_RPC_BASE_PATH"
```

## Next

- Bring up the llm-d **EPP router** (no-k8s mode) so it consumes the CPU-tier KV events, for the
  apples-to-apples routing comparison vs Dynamo.
- Compare LMCache chunking (`LMCACHE_CHUNK_SIZE`, default 256) vs the vLLM-native
  `OffloadingConnector` chunking studied in step 1.
