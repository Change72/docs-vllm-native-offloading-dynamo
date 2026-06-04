#!/usr/bin/env python3
"""Wire vLLM <-> llm-d's KV offload data plane (LMCache) and PROVE data moves:
KV is offloaded INTO LMCache (store) and loaded back OUT of LMCache (retrieve).

This is the literal "vLLM offload/load data to/from llm-d" milestone, using
llm-d's LMCache connector path (guides/tiered-prefix-cache/cpu, CONNECTOR=lmcache).
Runs fully local on L4 in `.venv` — no EPP/Envoy/K8s.

Phases (same shape as the phase-1 native-offload harness, so results are comparable):
  1. WARM  - one long shared prefix P -> LMCache STORES P's chunks (offload to llm-d/LMCache).
  2. EVICT - unique filler -> evict P from the vLLM GPU prefix cache (LMCache keeps it).
  3. REUSE - request P again -> GPU misses, LMCache RETRIEVES P's chunks (load from llm-d/LMCache).

Watch the LMCache logs:
  "LMCache INFO: Storing KV cache for N out of N tokens for request ..."   <- offload
  "LMCache INFO: Retrieving/Reusing ... N tokens ..."                       <- load
"""
import os
import time

# LMCache's offload/lookup servers bind ZMQ Unix sockets under VLLM_RPC_BASE_PATH
# (defaults to $TMPDIR). The full "<base>/engine_<uuid>_service_..._rpc_port_<n>"
# path must stay < 107 chars or LMCache init silently fails (post_init skipped ->
# every lookup returns 0 hits). .envrc already exports a short VLLM_RPC_BASE_PATH;
# set a safe default here too so the driver works even if .envrc wasn't sourced.
os.environ.setdefault("VLLM_RPC_BASE_PATH", "/tmp/vllm_rpc")
os.makedirs(os.environ["VLLM_RPC_BASE_PATH"], exist_ok=True)

# ---- LMCache (llm-d KV offload backend) config: local CPU tier ----
os.environ.setdefault("LMCACHE_CHUNK_SIZE", "128")        # LMCache chunk = 128 tok (= 8 vLLM blocks)
os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")        # enable CPU offload backend
os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "2")  # 2 GB CPU pool (holds P across eviction)
os.environ.setdefault("LMCACHE_LOG_LEVEL", "DEBUG")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder

MODEL = os.environ.get("LM_MODEL", "facebook/opt-125m")
BLOCK_SIZE = int(os.environ.get("LM_BLOCK_SIZE", "16"))
GPU_BLOCKS = int(os.environ.get("LM_GPU_BLOCKS", "64"))      # 64*16 = 1024 tok GPU KV -> force eviction
PREFIX_LEN = int(os.environ.get("LM_PREFIX_LEN", "512"))     # 4 LMCache chunks @128
N_FILLER = int(os.environ.get("LM_N_FILLER", "8"))
FILLER_LEN = int(os.environ.get("LM_FILLER_LEN", "900"))
MAX_MODEL_LEN = int(os.environ.get("LM_MAX_MODEL_LEN", "1024"))
VOCAB_SAFE = 40000


def banner(m):
    print(f"\n{'='*78}\nLMCACHE-PHASE {m}\n{'='*78}", flush=True)


def main():
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(
        model=MODEL,
        kv_transfer_config=ktc,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.30,
        num_gpu_blocks_override=GPU_BLOCKS,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=2,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    prefix = [(7 + (i * 13) % VOCAB_SAFE) for i in range(PREFIX_LEN)]

    banner("1-WARM (expect LMCache STORE / offload of P)")
    llm.generate([TokensPrompt(prompt_token_ids=prefix + [101, 102, 103])], sp)

    banner("2-EVICT (fill GPU KV with unique content)")
    fillers = [
        TokensPrompt(prompt_token_ids=[((20000 + f * 97) + i * 31) % VOCAB_SAFE
                                       for i in range(FILLER_LEN)])
        for f in range(N_FILLER)
    ]
    llm.generate(fillers, sp)
    time.sleep(2.0)  # let async LMCache stores settle

    banner("3-REUSE (same prefix P, new suffix -> expect LMCache RETRIEVE / load)")
    llm.generate([TokensPrompt(prompt_token_ids=prefix + [201, 202, 203, 204])], sp)

    banner("DONE")
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("LMCACHE-DRIVER finished. Grep the log for 'LMCache'.", flush=True)


if __name__ == "__main__":
    main()
