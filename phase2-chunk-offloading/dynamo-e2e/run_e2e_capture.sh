#!/bin/bash
set -u
WD=/home/changg/workspace/.tmp/dyn_e2e
source /home/changg/workspace/.envrc >/dev/null 2>&1
export PYTHONHASHSEED=0 DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=$WD/dynstore DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq DYN_LOG=info
rm -rf "$DYN_FILE_KV"; mkdir -p "$DYN_FILE_KV"
HTTP_PORT=8600

DYN_SYSTEM_PORT=8082 python -m dynamo.frontend --router-mode kv --router-reset-states --http-port $HTTP_PORT > "$WD/frontend.log" 2>&1 &
FRONTEND_PID=$!
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B --block-size 16 --enforce-eager \
    --gpu-memory-utilization 0.3 --max-model-len 4096 \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":134217728,"block_size":48,"self_describing_kv_events":true}}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
    > "$WD/worker.log" 2>&1 &
WORKER_PID=$!

for i in $(seq 1 150); do
  curl -sf "http://localhost:$HTTP_PORT/v1/models" 2>/dev/null | grep -q Qwen && { echo "[cap] healthy after ${i}x2s"; break; }
  kill -0 $WORKER_PID 2>/dev/null || { echo "[cap] WORKER DIED"; tail -20 "$WD/worker.log"; kill 0; exit 1; }
  sleep 2
done

python /home/changg/workspace/.tmp/llmd_4way/capture.py "tcp://localhost:20080" "$WD/sidecap.jsonl" > "$WD/sidecap.log" 2>&1 &
CAP_PID=$!
sleep 1

( cd /home/changg/workspace/.tmp/llmd_4way/multi_turn_client && \
  PYTHONPATH=. PYTHONSAFEPATH=0 python benchmark_serving_multi_turn.py \
    -i /home/changg/workspace/.tmp/llmd_4way/gen_small.json \
    --model Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
    --url "http://localhost:$HTTP_PORT" --num-clients 4 --max-active-conversations 8 ) > "$WD/bench_cap.log" 2>&1
echo "[cap] bench rc=$?"
sleep 3
curl -s "localhost:$HTTP_PORT/metrics" > "$WD/frontend_metrics_cap.txt"
kill -TERM $CAP_PID 2>/dev/null; sleep 2; kill -9 $CAP_PID 2>/dev/null
kill -TERM $WORKER_PID $FRONTEND_PID 2>/dev/null
for i in $(seq 1 15); do kill -0 $WORKER_PID 2>/dev/null || break; sleep 1; done
kill -9 $WORKER_PID $FRONTEND_PID 2>/dev/null
echo "[cap] DONE frames=$(wc -l < $WD/sidecap.jsonl)"
