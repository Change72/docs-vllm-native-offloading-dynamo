#!/usr/bin/env python3
"""Phase-1 verification: does vLLM's OffloadingConnector group GPU blocks into
larger CPU "chunks" when block_size_factor > 1, and can it reuse them?

This driver is intentionally backend-agnostic so the same workload shape can
later be pointed at llm-d (or any other native-offloading stack) for an A/B
performance comparison. It only depends on vLLM's offline `LLM` API + the
OffloadingConnector; no Dynamo / NATS / etcd required.

Three phases:
  1. WARM  - one long, block-aligned shared prefix P -> offloads P's chunks.
  2. EVICT - several unique long requests -> evict P from the GPU KV cache.
  3. REUSE - request P again (new suffix) -> should reload P's chunks CPU->GPU.

Watch the CHUNKPROBE log lines emitted by the (clean) scheduler:
  CHUNKPROBE CONFIG ...           offload geometry (factor, sizes)
  CHUNKPROBE STORE  ...           per-chunk hash vs constituent block hashes
  CHUNKPROBE EVENT  ...           exactly what a router would receive
  CHUNKPROBE REUSE/LOAD ...       proof of CPU->GPU chunk reuse
"""
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# ---- knobs (env-overridable so this doubles as a sweep / A-B harness) ----
MODEL = os.environ.get("CP_MODEL", "facebook/opt-125m")
BLOCK_SIZE = int(os.environ.get("CP_BLOCK_SIZE", "16"))
OFFLOADED_BLOCK_SIZE = int(os.environ.get("CP_OFFLOADED_BLOCK_SIZE", "48"))  # factor=3
GPU_BLOCKS = int(os.environ.get("CP_GPU_BLOCKS", "64"))      # 64*16=1024 tok GPU KV
CPU_BYTES = int(os.environ.get("CP_CPU_BYTES", str(1 << 30)))  # 1 GiB CPU pool
PREFIX_LEN = int(os.environ.get("CP_PREFIX_LEN", "480"))     # 10 chunks @48
N_FILLER = int(os.environ.get("CP_N_FILLER", "8"))
FILLER_LEN = int(os.environ.get("CP_FILLER_LEN", "1000"))    # < max_model_len, ~whole GPU KV
MAX_MODEL_LEN = int(os.environ.get("CP_MAX_MODEL_LEN", "1024"))  # must be <= GPU_BLOCKS*block
VOCAB_SAFE = int(os.environ.get("CP_VOCAB_SAFE", "40000"))   # below opt-125m vocab

assert PREFIX_LEN % OFFLOADED_BLOCK_SIZE == 0, "prefix must be chunk-aligned"
assert OFFLOADED_BLOCK_SIZE % BLOCK_SIZE == 0, "offloaded must be multiple of block"
FACTOR = OFFLOADED_BLOCK_SIZE // BLOCK_SIZE


def banner(msg):
    print(f"\n{'='*78}\nCHUNKPROBE-PHASE {msg}\n{'='*78}", flush=True)


def main():
    kv_transfer_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "CPUOffloadingSpec",
            "block_size": OFFLOADED_BLOCK_SIZE,
            "cpu_bytes_to_use": CPU_BYTES,
        },
    }

    print("CHUNKPROBE-DRIVER config: "
          + json.dumps({
              "model": MODEL, "block_size": BLOCK_SIZE,
              "offloaded_block_size": OFFLOADED_BLOCK_SIZE, "factor": FACTOR,
              "gpu_blocks_override": GPU_BLOCKS, "cpu_bytes": CPU_BYTES,
              "prefix_len": PREFIX_LEN, "n_filler": N_FILLER,
              "filler_len": FILLER_LEN,
          }), flush=True)

    llm = LLM(
        model=MODEL,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.30,
        num_gpu_blocks_override=GPU_BLOCKS,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=2,
        enforce_eager=True,
        kv_transfer_config=kv_transfer_config,
        # Enable KV cache events so the connector's take_events() runs and we can
        # log exactly what a downstream router (Dynamo / llm-d) would receive.
        # "null" publisher discards after emission -> no external infra needed.
        kv_events_config={"enable_kv_cache_events": True, "publisher": "null"},
    )

    sp = SamplingParams(temperature=0.0, max_tokens=8)

    # Deterministic shared prefix P (chunk-aligned). Distinct, > a few blocks.
    prefix = [(7 + (i * 13) % VOCAB_SAFE) for i in range(PREFIX_LEN)]

    banner("1-WARM (offload P's chunks)")
    llm.generate([TokensPrompt(prompt_token_ids=prefix + [101, 102, 103])], sp)

    banner("2-EVICT (fill GPU KV with unique content)")
    fillers = []
    for f in range(N_FILLER):
        base = 20000 + f * 97
        fillers.append(TokensPrompt(
            prompt_token_ids=[((base + i * 31) % VOCAB_SAFE) for i in range(FILLER_LEN)]
        ))
    llm.generate(fillers, sp)
    time.sleep(1.0)  # let async GPU->CPU stores + complete_store settle

    banner("3-REUSE (same prefix P, new suffix -> expect CPU->GPU reload)")
    llm.generate([TokensPrompt(prompt_token_ids=prefix + [201, 202, 203, 204])], sp)

    banner("DONE")
    print("CHUNKPROBE-DRIVER finished. Grep the log for 'CHUNKPROBE'.", flush=True)


if __name__ == "__main__":
    main()
