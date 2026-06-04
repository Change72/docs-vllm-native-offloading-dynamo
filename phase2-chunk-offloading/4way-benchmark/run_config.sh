#!/bin/bash
# run_config.sh <name> <base|patched> <factor>
# Boots vLLM (chosen scheduler + chunk factor) with OffloadingConnector + ZMQ KV
# events, captures the real event stream, drives the multi-turn benchmark, tears down.
set -u
NAME="$1"; SCHED="$2"; FACTOR="$3"
WD=/home/changg/workspace/.tmp/llmd_4way
VLLM=/home/changg/workspace/vllm
SCHPATH="$VLLM/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
CLIENT="$WD/multi_turn_client"
PORT=8000
MODEL="Qwen/Qwen3-0.6B"
TOPIC="kv@worker1@$MODEL"
OUT="$WD/runs/$NAME"
mkdir -p "$OUT"

source /home/changg/workspace/.envrc >/dev/null 2>&1

echo "[$NAME] scheduler=$SCHED factor=$FACTOR"
cp "$WD/${SCHED}_scheduler.py" "$SCHPATH"
python -c "import py_compile,sys; py_compile.compile('$SCHPATH',doraise=True)" || { echo "scheduler compile FAILED"; exit 1; }

if [ "$FACTOR" = "1" ]; then
  EXTRA='{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":4294967296}'
else
  EXTRA='{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":4294967296,"block_size":48}'
fi

# 1) vLLM server
echo "[$NAME] starting vLLM server..."
VLLM_LOGGING_LEVEL=INFO vllm serve "$MODEL" \
  --port $PORT \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.20 \
  --max-model-len 4096 \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":$EXTRA}" \
  --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:5557\",\"topic\":\"$TOPIC\",\"enable_kv_cache_events\":true}" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "[$NAME] server pid=$SERVER_PID; waiting for /health..."

for i in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[$NAME] server healthy after ${i}x2s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[$NAME] SERVER DIED; tail:"; tail -20 "$OUT/server.log"; exit 1; fi
  sleep 2
done
if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "[$NAME] server never healthy"; tail -20 "$OUT/server.log"; kill $SERVER_PID 2>/dev/null; exit 1; fi

# 2) ZMQ capture
python "$WD/capture.py" "tcp://localhost:5557" "$OUT/capture.jsonl" > "$OUT/capture.log" 2>&1 &
CAP_PID=$!
sleep 1
echo "[$NAME] capture pid=$CAP_PID"

# 3) multi-turn benchmark
echo "[$NAME] running multi-turn benchmark..."
( cd "$CLIENT" && PYTHONPATH="$CLIENT" PYTHONSAFEPATH=0 python benchmark_serving_multi_turn.py \
    -i "$WD/gen_small.json" \
    --model "$MODEL" --served-model-name "$MODEL" \
    --url "http://localhost:$PORT" \
    --num-clients 4 --max-active-conversations 8 \
    --stats-json-output "$OUT/stats.json" ) > "$OUT/bench.log" 2>&1
echo "[$NAME] benchmark rc=$?"

# 4) teardown (SIGTERM capture first so it flushes)
sleep 2
kill -TERM $CAP_PID 2>/dev/null; sleep 2; kill -9 $CAP_PID 2>/dev/null
kill -TERM $SERVER_PID 2>/dev/null
for i in $(seq 1 15); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 1; done
kill -9 $SERVER_PID 2>/dev/null
sleep 1

FRAMES=$(wc -l < "$OUT/capture.jsonl" 2>/dev/null || echo 0)
echo "[$NAME] DONE: captured $FRAMES frames -> $OUT/capture.jsonl"
