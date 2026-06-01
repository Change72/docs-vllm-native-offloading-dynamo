#!/usr/bin/env bash
# Master sweep orchestrator (runs INSIDE orchestrator pod).
# Sequentially: cold-restart DGD → wait ready → run bench → dump artifacts.
# Failures don't abort — script marks cycle FAIL and continues.
set +e; set +u

NS=changg-dynamo
DGD=qwen3-32b-offload-router
CLIENT=bench-multi-turn
RESULTS=/results
mkdir -p $RESULTS
SUMMARY=$RESULTS/_summary.csv
[ -f $SUMMARY ] || echo "cycle_id,label,image,h,disk,pl,workload,c,rr,runtime_s,rps,ttft_mean,tpot_mean,latency_mean,gpu_block,ext_block,gpu_tok,ext_tok,compute_tok,overall_tok,g2c_GB,c2g_GB,bench_log_path,status" > $SUMMARY

BASELINE_IMG="docker.io/change1472/dynamo-vllm-cpu-offload@sha256:0ca50b96c070e728fbe806aa222f5e1ee5357aade03e31b8ef0facc47dd9ee34"
TUNABLE_IMG="docker.io/change1472/dynamo-vllm-cpu-offload@sha256:0340789292b9a70a218c3d5ef5cd3674e585a400a644812b20c58d2360fb4879"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $RESULTS/_master.log; }

# --- generate DGD yaml on-the-fly ---
gen_dgd_yaml() {
  local image=$1; local h=$2; local disk=$3; local pl=$4; local cpu_bytes=$5
  cat <<EOF
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: $DGD
  namespace: $NS
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: $image
          imagePullPolicy: IfNotPresent
          command: ["python3"]
          args:
          - -m
          - dynamo.frontend
          - --http-port
          - "8000"
          - --router-mode
          - kv
          - --router-reset-states
$( [ -n "$h" ] && printf '          - --router-host-cache-hit-weight\n          - "%s"\n' "$h" )
$( [ -n "$disk" ] && printf '          - --router-disk-cache-hit-weight\n          - "%s"\n' "$disk" )
$( [ -n "$pl" ] && printf '          - --router-prefill-load-scale\n          - "%s"\n' "$pl" )
          env:
          - name: DYN_ROUTER_MODE
            value: kv
          - name: NATS_SERVER
            value: nats://dynamo-platform-nats.dynamo-system.svc.cluster.local:4222
          - name: DYN_DISCOVERY_BACKEND
            value: kubernetes
  - name: VllmDecodeWorker
    type: worker
    replicas: 8
    podTemplate:
      spec:
        tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
        containers:
        - name: main
          image: $image
          imagePullPolicy: IfNotPresent
          command: ["python3"]
          args:
          - -m
          - dynamo.vllm
          - --model
          - Qwen/Qwen3-32B
          - --enforce-eager
          - --gpu-memory-utilization
          - "0.50"
          - --kv-transfer-config
          - '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":$cpu_bytes}}'
          - --kv-events-config
          - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
          envFrom:
          - secretRef:
              name: hf-token-secret
          env:
          - name: HF_HOME
            value: /home/dynamo/.cache/huggingface
          - name: DYN_HEALTH_CHECK_ENABLED
            value: "false"
          - name: NATS_SERVER
            value: nats://dynamo-platform-nats.dynamo-system.svc.cluster.local:4222
          - name: DYN_DISCOVERY_BACKEND
            value: kubernetes
          resources:
            limits:
              cpu: "16"
              memory: "300Gi"
              nvidia.com/gpu: "1"
            requests:
              cpu: "8"
              memory: "220Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
          - mountPath: /home/dynamo/.cache/huggingface
            name: model-cache
EOF
}

wait_ready() {
  for i in $(seq 1 180); do
    sleep 10
    local r=$(kubectl get pod -n $NS -l grove.io/podclique=$DGD-0-vllmdecodeworker --no-headers 2>/dev/null | grep "1/1" | wc -l | tr -d ' ')
    local f=$(kubectl get pod -n $NS -l grove.io/podclique=$DGD-0-frontend --no-headers 2>/dev/null | grep "1/1" | wc -l | tr -d ' ')
    if [ $((i % 6)) = 0 ]; then log "  wait $((i*10))s: workers=$r/8 frontend=$f/1"; fi
    [ "$r" = "8" ] && [ "$f" = "1" ] && return 0
  done
  return 1
}

cold_restart() {
  local image=$1; local h=$2; local disk=$3; local pl=$4
  # all our images use 150 GB CPU offload (matches all previous experiments)
  local cpu_bytes=161061273600
  log "cold restart: image=...${image: -12}  h=$h disk=$disk pl=$pl"
  kubectl delete dgd $DGD -n $NS --ignore-not-found 2>&1 | head -1 >> $RESULTS/_master.log
  sleep 15
  gen_dgd_yaml "$image" "$h" "$disk" "$pl" "$cpu_bytes" | kubectl apply -f - 2>&1 | head -1 >> $RESULTS/_master.log
  wait_ready
}

dump_artifacts() {
  local cycle_id=$1; local dir=$RESULTS/$cycle_id; mkdir -p $dir
  kubectl cp "$NS/$CLIENT:/work/_bench.log" "$dir/bench.log" 2>/dev/null
  kubectl cp "$NS/$CLIENT:/work/_stats.json" "$dir/stats.json" 2>/dev/null
  local fpod=$(kubectl get pod -n $NS -o name 2>/dev/null | grep frontend | head -1 | cut -d/ -f2)
  [ -n "$fpod" ] && kubectl exec "$fpod" -n $NS -- curl -sf http://localhost:8000/metrics > "$dir/metrics_frontend.txt" 2>/dev/null
  for w in $(kubectl get pod -n $NS -o name 2>/dev/null | grep vllmdecodeworker | cut -d/ -f2); do
    local short=$(echo $w | rev | cut -d- -f1 | rev)
    kubectl exec "$w" -n $NS -- curl -sf http://localhost:9090/metrics > "$dir/metrics_worker_${short}.txt" 2>/dev/null &
  done
  wait
}

# parse metrics + bench log → append summary CSV row
collect_metrics() {
  local cycle_id=$1; local label=$2; local image=$3; local h=$4; local disk=$5; local pl=$6
  local workload=$7; local c=$8; local rr=$9; local status="${10}"
  local dir=$RESULTS/$cycle_id
  python3 - "$dir" "$cycle_id" "$label" "$image" "$h" "$disk" "$pl" "$workload" "$c" "$rr" "$status" "$SUMMARY" <<'PY'
import sys, re, os, csv
from pathlib import Path
dir_, cycle_id, label, image, h, disk, pl, workload, c, rr, status, summary = sys.argv[1:13]
dir_p = Path(dir_)
def parse_bench():
    bl = (dir_p/'bench.log')
    if not bl.exists(): return {}
    t = bl.read_text()
    m = {}
    for k, p in [('runtime', r'runtime_sec\s*=\s*([\d.]+)'), ('rps', r'requests_per_sec\s*=\s*([\d.]+)')]:
        x = re.search(p, t)
        if x: m[k] = float(x.group(1))
    for col, regex in [('ttft_mean', r'^ttft_ms\s+[\d.]+\s+([\d.]+)'), 
                        ('tpot_mean', r'^tpot_ms\s+[\d.]+\s+([\d.]+)'),
                        ('latency_mean', r'^latency_ms\s+[\d.]+\s+([\d.]+)')]:
        x = re.search(regex, t, re.M)
        if x: m[col] = float(x.group(1))
    return m
def parse_metrics():
    g_h=g_q=e_h=e_q=g2c=c2g=tl=te=tc=0
    for wf in dir_p.glob('metrics_worker_*.txt'):
        txt = wf.read_text()
        def grab(pat, t=txt):
            return sum(float(m.group(1)) for m in re.finditer(pat, t, re.M))
        g_h += grab(r'^vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.eE+-]+)')
        g_q += grab(r'^vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.eE+-]+)')
        e_h += grab(r'^vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([\d.eE+-]+)')
        e_q += grab(r'^vllm:external_prefix_cache_queries_total\{[^}]*\}\s+([\d.eE+-]+)')
        g2c += grab(r'^vllm:kv_offload_total_bytes_total\{[^}]*GPU_to_CPU[^}]*\}\s+([\d.eE+-]+)')
        c2g += grab(r'^vllm:kv_offload_total_bytes_total\{[^}]*CPU_to_GPU[^}]*\}\s+([\d.eE+-]+)')
        tl += grab(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\}\s+([\d.eE+-]+)')
        te += grab(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="external_kv_transfer"[^}]*\}\s+([\d.eE+-]+)')
        tc += grab(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\}\s+([\d.eE+-]+)')
    tot = tl + te + tc
    return dict(gpu_block=g_h/g_q*100 if g_q else 0, ext_block=e_h/e_q*100 if e_q else 0,
                gpu_tok=tl/tot*100 if tot else 0, ext_tok=te/tot*100 if tot else 0,
                compute_tok=tc/tot*100 if tot else 0, overall_tok=(tl+te)/tot*100 if tot else 0,
                g2c_GB=g2c/1e9, c2g_GB=c2g/1e9)
b = parse_bench(); m = parse_metrics()
img_short = image.split('@')[-1][-12:] if '@' in image else image[-12:]
row = [cycle_id, label, img_short, h, disk, pl, workload, c, rr,
       f"{b.get('runtime', 0):.0f}", f"{b.get('rps', 0):.2f}",
       f"{b.get('ttft_mean', 0):.0f}", f"{b.get('tpot_mean', 0):.1f}", f"{b.get('latency_mean', 0):.0f}",
       f"{m['gpu_block']:.2f}", f"{m['ext_block']:.2f}",
       f"{m['gpu_tok']:.1f}", f"{m['ext_tok']:.1f}", f"{m['compute_tok']:.1f}", f"{m['overall_tok']:.1f}",
       f"{m['g2c_GB']:.0f}", f"{m['c2g_GB']:.0f}", str(dir_p/'bench.log'), status]
with open(summary, 'a') as f:
    csv.writer(f).writerow(row)
print(f"  summary: compute={m['compute_tok']:.1f}% overall={m['overall_tok']:.1f}% rps={b.get('rps', 0):.2f}")
PY
}

run_cycle() {
  local cycle_id=$1; local label=$2; local image=$3
  local h=$4; local disk=$5; local pl=$6
  local workload_json=$7; local c=$8; local rr=$9
  
  log ""
  log "============================================================"
  log "CYCLE $cycle_id  | $label | h=$h disk=$disk pl=$pl | $workload_json c=$c rr=$rr"
  log "============================================================"

  cold_restart "$image" "$h" "$disk" "$pl"
  if [ $? -ne 0 ]; then log "[FAIL] DGD not ready"; collect_metrics "$cycle_id" "$label" "$image" "$h" "$disk" "$pl" "$workload_json" "$c" "$rr" "FAIL_NOREADY"; return; fi

  log "  launching bench..."
  kubectl exec $CLIENT -n $NS -- bash -c "
    cd /work/multi_turn
    rm -f /work/_bench.log /work/_stats.json
    nohup python3 benchmark_serving_multi_turn.py \\
      --model Qwen/Qwen3-32B --served-model-name Qwen/Qwen3-32B \\
      --url http://qwen3-32b-offload-router-frontend.changg-dynamo.svc.cluster.local:8000 \\
      --input-file $workload_json \\
      --num-clients $c --max-active-conversations $c \\
      --request-rate $rr --warmup-step \\
      --stats-json-output /work/_stats.json > /work/_bench.log 2>&1 &
  " >> $RESULTS/_master.log

  # poll up to 60 min
  for i in $(seq 1 120); do
    sleep 30
    local done=$(kubectl exec $CLIENT -n $NS -- bash -c "grep -c 'Statistics summary' /work/_bench.log 2>/dev/null" 2>/dev/null | tr -d '\n' || echo 0)
    [ "$done" = "1" ] && { log "  [done at $((i/2))min]"; sleep 5; break; }
    local crash=$(kubectl exec $CLIENT -n $NS -- bash -c "grep -c 'AssertionError\|RuntimeError\|Traceback' /work/_bench.log 2>/dev/null" 2>/dev/null | tr -d '\n' || echo 0)
    [ "$crash" != "0" ] && [ "$crash" != "" ] && { log "  [CRASH at $((i/2))min]"; break; }
    [ $((i % 4)) = 0 ] && log "  poll $((i/2))min"
  done

  dump_artifacts "$cycle_id"
  collect_metrics "$cycle_id" "$label" "$image" "$h" "$disk" "$pl" "$workload_json" "$c" "$rr" "OK"
}

# ============================================================
# SWEEP PLAN
# ============================================================
T0=$(date +%s)
log "=========================="
log "MASTER SWEEP START $(date)"
log "results dir: $RESULTS"
log "=========================="

# ---------- TIER 1: Ablation matrix (long-bench, c=64) ----------
LONG=generate_multi_turn_longbench.json
log ""
log "##### TIER 1: ABLATION MATRIX (long-bench, c=64, rr=0.5) #####"
run_cycle t1-01-baseline-default       "baseline+default" $BASELINE_IMG "" "" ""    $LONG 64 0.5
run_cycle t1-02-baseline-pl100         "baseline+pl100"   $BASELINE_IMG "" "" "100" $LONG 64 0.5
run_cycle t1-03-tunable-h075-pl1       "tunable h=0.75 pl=1"   $TUNABLE_IMG "0.75" "0.25" "1"   $LONG 64 0.5
run_cycle t1-04-tunable-h075-pl100     "tunable h=0.75 pl=100" $TUNABLE_IMG "0.75" "0.25" "100" $LONG 64 0.5
run_cycle t1-05-tunable-h1-pl1         "tunable h=1.0 pl=1"    $TUNABLE_IMG "1.0"  "0.25" "1"   $LONG 64 0.5
run_cycle t1-06-tunable-h1-pl100       "tunable h=1.0 pl=100"  $TUNABLE_IMG "1.0"  "0.25" "100" $LONG 64 0.5

# ---------- TIER 2: Concurrency sweep ----------
log ""
log "##### TIER 2: CONCURRENCY SWEEP (long-bench, rr=0.5) #####"
for c in 8 16 32 64 96 128 192; do
  run_cycle "t2-baseline-c${c}"  "baseline c=$c"  $BASELINE_IMG ""    ""     ""    $LONG $c 0.5
  run_cycle "t2-best-c${c}"      "best c=$c"      $TUNABLE_IMG  "1.0" "0.25" "100" $LONG $c 0.5
done

# ---------- TIER 3: (h, pl) heatmap @ c=64 ----------
log ""
log "##### TIER 3: (h, pl) HEATMAP (long-bench, c=64) #####"
for h in 0.75 1.0 1.25; do
  for pl in 1 10 100; do
    # skip duplicates of T1
    if [ "$h" = "0.75" ] && [ "$pl" = "1" ]; then continue; fi
    if [ "$h" = "0.75" ] && [ "$pl" = "100" ]; then continue; fi
    if [ "$h" = "1.0" ] && [ "$pl" = "1" ]; then continue; fi
    if [ "$h" = "1.0" ] && [ "$pl" = "100" ]; then continue; fi
    run_cycle "t3-h${h}-pl${pl}" "tunable h=$h pl=$pl" $TUNABLE_IMG "$h" "0.25" "$pl" $LONG 64 0.5
  done
done

# ---------- TIER 4: conv_prefix workload sweep ----------
# generate configs on-the-fly
log ""
log "##### TIER 4: conv_prefix SWEEP (c=64) #####"
for prefix in 5000 25000; do
  cfg="/tmp/cfg_prefix_${prefix}.json"
  python3 -c "
import json
c = json.load(open('/configs/longbench-template.json'))
c['prompt_input']['prefix_num_tokens']['value'] = $prefix
c['num_conversations'] = max(64, min(2400000 // $prefix, 200))
json.dump(c, open('$cfg', 'w'))
"
  kubectl cp $cfg $NS/$CLIENT:/work/multi_turn/cfg_prefix_${prefix}.json
  for variant in baseline best; do
    if [ "$variant" = "baseline" ]; then img=$BASELINE_IMG; h=""; pl="";
    else img=$TUNABLE_IMG; h="1.0"; pl="100"; fi
    run_cycle "t4-${variant}-prefix${prefix}" "${variant} prefix=${prefix}" $img "$h" "0.25" "$pl" "cfg_prefix_${prefix}.json" 64 0.5
  done
done

# ---------- TIER 5: LMBench scaled ----------
log ""
log "##### TIER 5: LMBench scaled (NUM_USERS=64, QPS=32) #####"
run_lmbench() {
  local cycle_id=$1; local label=$2; local image=$3; local h=$4; local pl=$5
  log ""
  log "============================================================"
  log "CYCLE $cycle_id  | $label (LMBench)"
  log "============================================================"
  cold_restart "$image" "$h" "0.25" "$pl"
  [ $? -ne 0 ] && { log "[FAIL]"; return; }
  log "  launching LMBench..."
  kubectl exec $CLIENT -n $NS -- bash -c "
    cd /work/lmbench
    rm -f /work/_lmbench.log /work/_lmbench_output_32.csv
    export NUM_USERS_WARMUP=64 NUM_USERS=64 NUM_ROUNDS=40
    export SYSTEM_PROMPT=0 CHAT_HISTORY=15000 ANSWER_LEN=100 INIT_USER_ID=1 TEST_DURATION=200
    nohup ./long_input_short_output_run.sh 'Qwen/Qwen3-32B' \\
      'http://qwen3-32b-offload-router-frontend.changg-dynamo.svc.cluster.local:8000' \\
      '/work/_lmbench' 32 > /work/_lmbench.log 2>&1 &
  " >> $RESULTS/_master.log
  for i in $(seq 1 60); do
    sleep 30
    local done=$(kubectl exec $CLIENT -n $NS -- bash -c "grep -c 'Performance summary' /work/_lmbench.log 2>/dev/null" 2>/dev/null | tr -d '\n' || echo 0)
    [ "$done" -ge 2 ] && { log "  [done at $((i/2))min]"; sleep 5; break; }
    [ $((i % 4)) = 0 ] && log "  poll $((i/2))min"
  done
  # dump
  local dir=$RESULTS/$cycle_id; mkdir -p $dir
  kubectl cp "$NS/$CLIENT:/work/_lmbench.log" "$dir/bench.log" 2>/dev/null
  kubectl cp "$NS/$CLIENT:/work/_lmbench_output_32.csv" "$dir/lmbench.csv" 2>/dev/null
  for w in $(kubectl get pod -n $NS -o name 2>/dev/null | grep vllmdecodeworker | cut -d/ -f2); do
    short=$(echo $w | rev | cut -d- -f1 | rev)
    kubectl exec "$w" -n $NS -- curl -sf http://localhost:9090/metrics > "$dir/metrics_worker_${short}.txt" 2>/dev/null &
  done
  wait
  collect_metrics "$cycle_id" "$label" "$image" "$h" "0.25" "$pl" "lmbench" "64" "0.5" "OK"
}
run_lmbench t5-01-baseline      "lmbench baseline"   $BASELINE_IMG ""    ""
run_lmbench t5-02-tunable-h075-pl1   "lmbench h=0.75 pl=1"   $TUNABLE_IMG "0.75" "1"
run_lmbench t5-03-tunable-h1-pl1     "lmbench h=1.0 pl=1"    $TUNABLE_IMG "1.0"  "1"
run_lmbench t5-04-tunable-h1-pl100   "lmbench h=1.0 pl=100"  $TUNABLE_IMG "1.0"  "100"

# ---------- TIER 6: rr sweep @ c=64 ----------
log ""
log "##### TIER 6: rr SWEEP (long-bench, c=64) #####"
for rr in 0.1 1.0; do
  # 0.5 already in T1
  run_cycle "t6-baseline-rr${rr}" "baseline rr=$rr" $BASELINE_IMG ""    ""     ""    $LONG 64 $rr
  run_cycle "t6-best-rr${rr}"     "best rr=$rr"    $TUNABLE_IMG  "1.0" "0.25" "100" $LONG 64 $rr
done

T1=$(date +%s)
log ""
log "=========================="
log "SWEEP COMPLETE at $(date), elapsed $(( (T1-T0) / 60 )) min"
log "summary CSV: $SUMMARY"
log "=========================="
sleep infinity   # keep pod alive for kubectl cp
