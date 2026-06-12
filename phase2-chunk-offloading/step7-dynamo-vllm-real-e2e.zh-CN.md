# Phase 2 · Step 7 — 真实 dynamo serve + vLLM 端到端:chunk offload KV 事件,指标级验证

> **Step 5** 选定了 wire 形状(chunk 展开为其组成 per-block hash 列表)。
> **Step 6** 发现并刻画了重叠 chunk 的两类风险;按团队决策,producer 侧最终交付**纯 fan-out**
> (vLLM 侧不做 refcount——该变体存档于 `feature/offloading-events-exactly-once` 分支),由消费侧
> 去重。Dynamo 的标准部署本来就具备这一能力:worker 侧 publisher 自带 `EventDedupFilter`
> (ai-dynamo/dynamo#8012;store 对 per-(dp_rank, tier) 计数 ++,remove 仅在计数归零时放行)。
> **Step 7(本文档):** 在**真实的单 GPU dynamo serve + vLLM 栈**上验证全链路——不是重放:
> 真实引擎、真实 publisher(filter 在链路上)、真实 KV router、真实 LRU 驱逐——router 侧
> `kv_cache_events_applied` 计数与旁路抓取的 vLLM wire **逐条精确对账**。

## TL;DR

| 方向 | vLLM wire(同轮旁路 ZMQ 抓包) | router applied(`kv_cache_events_applied`) |
|---|---|---|
| stored  | GPU 331 + CPU 354 = **685**(每条 CPU 事件 `n_hashes=3`,零 placeholder) | **685**(`status="ok"`) |
| removed | CPU **24**(128 MB 小池的真实 LRU 驱逐;hash 数全为 3 的倍数) | **24**(`status="ok"`) |
| 错误    | — | 0 warnings,0 `block_not_found` |

Chunk offloading(`factor=3`)+ opt-in 自描述事件,经 vLLM → ZMQ → dynamo worker
`KvEventPublisher`(listener → normalizer → **EventDedupFilter** → tier-routed 索引 +
event plane 转发给 frontend KV router),双向计数与 wire **逐事件相等**。

## 被测栈

| 组件 | 版本 / 分支 |
|---|---|
| vLLM | PR [#43468](https://github.com/vllm-project/vllm/pull/43468) 分支(`events.py` 的 `OffloadingEventsTracker`,opt-in,纯 fan-out),rebase 到 main `f712fd0d7` |
| dynamo | `feat/kv-router-cpu-medium-alias` @ `db0ec35619` —— 含 `medium="CPU"` → `HostPinned` 别名、lower-tier 的 `kv_cache_events_applied` 计数(`5b7725f4`)、以及惰性创建的 lower-tier 索引器的 metrics 接线(`db0ec356`,正是本次 e2e 发现的——见"踩坑记录") |
| python binding | 该 dynamo checkout 的 `maturin develop --uv`(`ai-dynamo-runtime 1.3.0`) |
| 模型 / GPU | `Qwen/Qwen3-0.6B`,1× NVIDIA L4 |
| 外部服务 | **零依赖** —— 文件发现、TCP 请求面、ZMQ 事件面 |

## 拓扑与事件路径

```
                         ┌────────────────────────────── worker 进程 ────────────────────────────────┐
  vLLM EngineCore        │  dynamo.vllm                                                               │
  OffloadingConnector ──ZMQ(tcp://127.0.0.1:20080, msgpack)──► ZmqEventListener ─► ZmqEventNormalizer │
  (chunk=48tok, factor=3,│        (received/accepted 计数)            ("CPU"→HostPinned, local hash)  │
   自描述                │                                  ─► event processor: 按 tier 分批          │
   BlockStored/Removed)  │                                     + EventDedupFilter (#8012)             │
                         │                                  ─► LocalKvIndexer(tier 路由:            │
                         │                                       GPU→radix,CPU→LowerTierIndexer)    │
                         │                                       └── kv_cache_events_applied ◄─ 即   │
                         │                                  ─► EventPlanePublisher ──ZMQ──┐  本文指标 │
                         └────────────────────────────────────────────────────────────────┼──────────┘
  旁路 capture.py ◄────── 同一 vLLM PUB 上的第二个 SUB(精确 wire 字节)                  │
                         ┌── frontend 进程 ────────────────────────────────────────────┐ │
                         │  dynamo.frontend --router-mode kv                           │ │
                         │  RouterEvent 订阅(local_indexer 模式)◄─────────────────────┼─┘
                         │  Indexer::apply_event(tier 路由:GPU→primary radix,        │
                         │                        HostPinned→LowerTierIndexers)        │
                         │  HTTP :8600(OpenAI API + /metrics,含转发的                │
                         │              backend kv_metrics)                            │
                         └────────────────────────────────────────────────────────────┘
```

## 关键配置

| 配置项 | 取值 | 理由 |
|---|---|---|
| `--block-size`(vLLM) | `16` | GPU/hash block 大小;也是 CPU 事件的 per-block `block_size` 和 publisher 计算 local hash 的块大小 |
| `kv_connector_extra_config.block_size` | `48` | offloaded chunk 大小 → `block_size_factor = 3`(被测的 chunk 模式) |
| `kv_connector_extra_config.cpu_bytes_to_use` | `134217728`(128 MB) | 小 CPU 池,确保运行中出现**真实 LRU 驱逐**(remove 路径被实际执行,24 条事件) |
| `kv_connector_extra_config.self_describing_kv_events` | `true` | PR #43468 的 opt-in;不开则 CPU 事件保持旧 placeholder 形状 |
| `--kv-events-config` | `{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}` | **必须显式传** —— `dynamo.vllm` 不会自动开启 vLLM KV 事件(见"踩坑记录") |
| `DYN_DISCOVERY_BACKEND=file`、`DYN_REQUEST_PLANE=tcp`、`DYN_EVENT_PLANE=zmq` | — | 零依赖本地模式:无 etcd、无 NATS、无 docker |
| `PYTHONHASHSEED=0` | — | KV 事件 ID 确定性(与官方 launch 示例一致) |
| `--router-mode kv`(frontend) | — | KV-aware 路由;本部署形态走 "local_indexer 模式" 的 ZMQ event plane 订阅 |
| `--gpu-memory-utilization 0.3`、`--max-model-len 4096`、`--enforce-eager` | — | 快速启动;3039 个 GPU 块(足够大使 GPU 层不驱逐——GPU removed=0 是有意为之,隔离出 CPU remove 路径) |

## 启动流程

完整脚本见 [`dynamo-e2e/run_e2e_capture.sh`](dynamo-e2e/run_e2e_capture.sh)。核心序列:

```bash
export PYTHONHASHSEED=0 DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=$WD/dynstore \
       DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq

# 1) frontend:OpenAI API(:8600)+ KV router
python -m dynamo.frontend --router-mode kv --router-reset-states --http-port 8600 &

# 2) worker:vLLM,chunk offloading + opt-in 事件 + 显式 kv-events-config
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B --block-size 16 --enforce-eager \
    --gpu-memory-utilization 0.3 --max-model-len 4096 \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
        "kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec",
        "cpu_bytes_to_use":134217728,"block_size":48,"self_describing_kv_events":true}}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events",
        "endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# 3) 等待健康
curl -sf http://localhost:8600/v1/models | grep -q Qwen

# 4) 旁路抓包:同一 vLLM PUB 上的第二个 SUB,记录精确 wire 字节
python capture.py "tcp://localhost:20080" sidecap.jsonl &

# 5) 流量:16 个多轮会话,256-token 共享前缀
#    (vLLM 自带 benchmark_serving_multi_turn.py + gen_small.json,4 客户端)
python benchmark_serving_multi_turn.py -i gen_small.json \
    --model Qwen/Qwen3-0.6B --url http://localhost:8600 \
    --num-clients 4 --max-active-conversations 8

# 6) 指标快照 + 对账
curl -s localhost:8600/metrics > frontend_metrics.txt
python dynamo-e2e/decode_capture.py sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied frontend_metrics.txt | grep -v ' 0$'
```

## 结果明细(单轮流量)

旁路抓包解码出的 wire:

```
[CPU] stored_events=354 (placeholder=0)   n_hashes 直方图: {3: 354}
      removed_events=24                   hash 数全为 3 的倍数
[GPU] stored_events=331                   removed_events=0(GPU 池未满)
```

Router 计数(frontend `/metrics`;`dynamo_component="backend"` 系列是 worker 的
LocalKvIndexer 经 `kv_metrics` event plane topic 转发而来):

```
kv_cache_events_applied{event_type="stored", status="ok"}  685   # = 331 GPU + 354 CPU
kv_cache_events_applied{event_type="removed", status="ok"}  24   # = 24 次真实 CPU 驱逐
kv_cache_event_warnings                                      0
```

Listener 阶段计数确认入口零丢弃(`kv_publisher_zmq_events_total`:stored 与 removed 均
`received == accepted`;零过滤、零转换失败),worker 日志中
`Failed to apply event to local indexer` 出现 0 次。

`applied` 按**事件**计数;dedup filter 在批内做的 per-hash 拦截不改变事件数。hash 级的
filter 语义(共享 hash 的非末次 remove 被扣住)由 filter 自身单测和
[`decode_capture.py`](dynamo-e2e/decode_capture.py) 的检查 7(filter 模拟)覆盖。

## 踩坑记录(复现前必读)

1. **Binding 过期。** editable 安装的 `dynamo._core`(`maturin develop` 产物)是构建工件;
   dynamo 源码前进后 worker 启动即崩:
   `DistributedRuntime.__new__() got an unexpected keyword argument 'event_plane'`。
   Rust 树有变动后,在 `lib/bindings/python` 重跑 `maturin develop --uv`。
2. **`--kv-events-config` 必须显式传。** 否则 `dynamo.vllm` 打日志
   `Using kv_events_config ... None (use_kv_events=False)`,**全程没有任何 KV 事件**
   (服务一切正常,router 计数恒 0)。
3. **认清你读的是哪个计数器。** `dynamo_component="backend"` 的 `kv_cache_events_applied`
   是 **worker LocalKvIndexer** 的计数(转发到 frontend 的 `/metrics` 展示)。在
   `db0ec356` 之前,惰性创建的 lower-tier 索引器构造时没拿到 metrics 句柄,CPU 流量被正确
   apply 却**不可见**(计数显示 332 / 0 而非 685 / 24)——一度看起来像事件链路断了。
   用"旁路 wire 抓包 + listener 阶段计数"才能区分"没到达"与"没计数"。
4. **Frontend 的指标在 HTTP 端口**(`:8600/metrics`),不在 `DYN_SYSTEM_PORT`。

## 复现

```bash
cd /home/changg/workspace/.tmp/dyn_e2e          # 或把 dynamo-e2e/run_e2e_capture.sh 拷到任意目录
bash run_e2e_capture.sh                          # 起栈、打流量、抓包、快照
python dynamo-e2e/decode_capture.py sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied frontend_metrics_cap.txt | grep -v ' 0$'
```

前置条件:venv 内 editable 安装的 vLLM PR 分支、已构建的 dynamo 分支
(`maturin develop --uv`)、HF 缓存中的 `Qwen/Qwen3-0.6B`、一块空闲 GPU,以及
`.tmp/llmd_4way/multi_turn_client` 下的多轮 benchmark 客户端(vLLM 的
`benchmark_serving_multi_turn.py` + `gen_small.json`)。

本次记录运行的产物在 `/home/changg/workspace/.tmp/dyn_e2e/`:`worker.log`、
`frontend.log`、`sidecap.jsonl`、`frontend_metrics_*.txt`、`bench_*.log`。
