# Phase 2 · Step 6 — 端到端验证,以及「重叠 chunk 事件」驱逐隐患

> **Step 5**([step5-final-design.md](step5-final-design.md))定型 **Plan B**:vLLM 把每个
> offload chunk 以**一条** `BlockStored` 发布(携带全部 `factor` 个组成 block hash + 整 chunk
> token),驱逐时 `BlockRemoved` fan-out 同一组 hash(vLLM 提交 `c3e203a18`,
> 分支 `feature/offloading-connector-chunk-payload`)。
> **Step 6(本文):**在**含真实 CPU 驱逐**的真实流量上,对两个 router 的**生产索引代码**做
> 端到端**正确性**验证 —— 以及验证过程抓到的一个真实 bug(已修)和一个残留隐患
> (未修,已量化,附修复建议)。

## TL;DR

| 层 | 跑了什么 | store | remove | 结论 |
|---|---|---|---|---|
| vLLM wire | 真实 serve,factor=3,128 MB CPU pool → 真实 LRU 驱逐;解码 313 个 ZMQ 帧 | 365 条,全部 `n_hashes=3, tok=48, bs=16`,parent 链完整 | 24 条共 1023 个 hash;精确切分为 **341 个曾完整 store 过的整 chunk 组** | ✅ |
| llm-d | 重放进真实 `VLLMAdapter → kvevents.Pool → InMemoryIndex` | **71/71** 个应存活 block 有 CPU pod | **516/516** 个被驱逐 block 已无 pod | ✅ |
| Dynamo | 重放进真实 `decode_event_batch → ZmqEventNormalizer → LowerTierIndexers`(只喂 CPU 事件——顺带证明事件自包含) | 11/11 条 chunk 链与期望 HostPinned 深度一致 | 修复前:10/11 + **泄漏 29 条边**;一行级修复后:**11/11** | ✅(修复后) |

**隐患——重叠 chunk 事件。**当共享前缀长度**不是 `offloaded_block_size` 的整数倍**时,每个
会话分支的边界 chunk 是*不同的 OffloadKey,却列出相同的共享块 hash*。vLLM 按 chunk key 去重,
于是这些 hash 在 wire 上被合法地 store/remove **多次**。两个后果:

1. **B1(已修,Dynamo)**——lower tier 在批量 `BlockRemoved` 中遇到第一个已删 hash 就中断
   整批 → 边泄漏(无界,违反 Step 4 §9 约束 7)。
2. **B2(已修,vLLM)**——*先驱逐者连坐(first-evict-wins)*:共享块在两个 router 的索引
   条目都**没有引用计数**,第一个兄弟分支的驱逐会把它删掉,而其余兄弟在 vLLM CPU 池里完好
   可复用 → 它们在 router 侧的连续 CPU 匹配崩塌(实测 llm-d:3/3 → **0/3**),vLLM 自己却
   能全量复用。修复:**vLLM side table 对已宣告 hash 做引用计数**(`746d7dc26`),hash 仅在
   最后一个存活 chunk 死亡时进入 `BlockRemoved`——两个 router 零改动。修复后:幸存兄弟保持
   **3/3**;重新捕获的事件流中 un-announced removal 为 **0**。

## 1. 怎么验证的(全部真实)

1. **vLLM** `feature/offloading-connector-chunk-payload` @ `c3e203a18`,`Qwen/Qwen3-0.6B`,
   factor=3,`cpu_bytes_to_use=128MiB`(故意调小 → 池满触发 LRU 驱逐),开 ZMQ KV events;
   多轮 benchmark(16 个会话,256-token 共享前缀)。逐帧捕获原始字节
   (`run_evict.sh planb_chunk_evict planb 3 134217728`)。
2. **Wire 解码**(`decode_capture.py`):形状 + 对账,5 项全 PASS,包括:每个被移除的 CPU
   hash 都曾 store 过;每条 `BlockRemoved` 可精确切分为曾 store 过的整 chunk 组——真实驱逐下
   removal fan-out 逐字节正确。(观察到 81 次批内乱序到达——符合预期:`complete_store` 遍历
   job 的 `set`;事件自包含,顺序无关。)
3. **llm-d**(`examples/chunk_replay_v2`):同一帧流重放进真实 ingest 栈,同时独立维护
   期望存活状态(store +1 / remove −1),最终逐 block 对账。
4. **Dynamo**(`lib/kv-router/tests/planb_capture_replay.rs`):**只重放 CPU 事件**进生产
   解码/归一化路径 → `LowerTierIndexers`——刻意如此,因为 Plan B 的卖点就是 CPU 事件自包含
   (无需 GPU 别名)。重建全部 root-to-leaf chunk 链,逐链断言真实 `query_lower_tiers`
   from-root 的 HostPinned 深度 == 模型的连续存活深度。

## 2. 隐患:一个共享块,多个 chunk

触发条件:`shared_prefix_len % offloaded_block_size != 0`。以 256-token 共享前缀、48-token
chunk(16-token 块,factor=3)为例:

```
              共享 256-token 前缀(16 个块)                会话私有部分
块:      0  1  2 │ 3  4  5 │ 6  7  8 │ 9 10 11 │ 12 13 14 │ 15 16A 17A …   会话 A
                                                           │ 15 16B 17B …   会话 B
chunk:    └─ c0 ─┘ └─ c1 ─┘ └─ c2 ─┘ └── c3 ──┘ └── c4 ──┘ └── c5 ───┘
          c0..c4:对齐的共享 chunk → A、B 是同一个 OffloadKey
                  → vLLM 只存一份、只发一次 store/remove。干净。
          c5:    256 % 48 ≠ 0 → 边界 chunk 跨过分叉点:
                  c5A = OffloadKey h17A,hashes [h15, h16A, h17A]
                  c5B = OffloadKey h17B,hashes [h15, h16B, h17B]
                  → 不同 key、两份独立 CPU 数据,但都列出 h15。
```

块 15 的 hash 是前 256 个共享 token 的累积 hash,所有会话相同;块 16 起分叉。于是事件流
**按 chunk 看完美配对,按单个 hash 看却重复**:

```
store  c5A → BlockStored [h15, h16A, h17A]
store  c5B → BlockStored [h15, h16B, h17B]    ← h15 第二次 store
evict  c5A → BlockRemoved[h15, h16A, h17A]    ← h15 第一次 remove
evict  c5B → BlockRemoved[h15, h16B, h17B]    ← h15 第二次 remove!
```

实测:一个边界 hash 出现在 **11 个不同 chunk** 的事件里;348 个 hash 被多次 store
(分叉变体 + 驱逐重存循环)。

**为什么 vLLM 不去重。**store 去重单位是 OffloadKey(`prepare_store` 跳过已存 key);
c5A 和 c5B 是*不同的 key*。池里的 `ref_cnt` 是**传输期 pin**(load/store 进行中防驱逐),
不是「这个块被多少序列共享」的计数。对齐的共享 chunk 靠「同 key 一份」天然去重;跨界 chunk
是不同 key、各自独立的数据副本——vLLM **内部**完全正确,只是 **wire 上**重复。
**by-block(`factor=1`)模式不可能出现此问题**:去重单位就是块本身,每个 hash 全局只
store/remove 一次。

## 3. 后果 B1(已修):Dynamo 批量驱逐中断泄漏

Dynamo lower tier 的反查表是单值 map(`worker_map: hash → TransitionKey`)——h15 第二次
store 是覆盖(无计数),第一次 remove 删除条目,第二次 remove 查不到,原代码**中断整批**:

```
evict c5B → BlockRemoved [ …54 个 hash…, h15, h16B, h17B ]
                                          ▲
            worker_map.remove(h15) → None → return Err(BlockNotFound)
            → h16B、h17B 以及批内剩余全部 hash 都没被删除
```

按精确单值语义重放 capture:**3 次中断、跳过 36 个 hash、泄漏 29 条边**(Dynamo 视角存活
100 个 hash,引用计数真值 71)。泄漏随时间无界(违反 Step 4 §9 第 7 条);同一个中断对约束
6 的合法跨层乱序(GPU `BlockRemoved` 先于 CPU `BlockStored` 到达)同样会触发。

**修复**(`lib/kv-router/src/indexer/lower_tier.rs::remove_blocks_impl`):缺失 hash 记录并
跳过、继续排空整批;仅当*全部* hash 缺失才返回 `BlockNotFound`。这与 llm-d 的既有行为一致
(`Evict` 未知 engine key 本来就是 no-op)。修复后:重放 11/11 全过;kv-router 全部 545 个
单测通过。

## 4. 后果 B2(已修):先驱逐者连坐——两个 router 都有

共享块在每个 router 里都只有**一份**索引条目(Dynamo:边 `(h14,l15)→h15` +
`worker_map[h15]`,重复插入幂等;llm-d:requestKey 下一个 `PodEntry`,set 语义)。都没有
引用计数,于是**第一个兄弟的驱逐替所有人删掉了它**:

```
vLLM CPU 池(evict c5A 后):  c5A 没了,c5B 完整在 → lookup(B) = 全命中
router 索引(evict c5A 后):  h15 条目被 c5A 的 BlockRemoved 删除
                              → B 的链在共享块处断裂
```

真实 llm-d 代码上的合成实证(`examples/overlap_probe`,两个共享块 0 的 48-token 会话):

```
store A+B 之后:连续 CPU 匹配  A=3/3  B=3/3
evict A   之后:连续 CPU 匹配  A=0/3  B=0/3   ← B 在 vLLM 里还完整存着!
```

Dynamo 由代码审视可知行为相同:`worker_map.remove(h15)` 删掉唯一的边;B 后面的块
(`(h15,l16B)→h16B`,仍在索引里)从此链走不到。

**严重性。**断点落在*共享*块上,即每个兄弟链的头部一侧。推广到 1k/3k 的问题:会话 A(1k)
和 B(3k)共享不对齐前缀;A 变冷、它的跨界 chunk 被 LRU 驱逐后,B 在 router 侧的 CPU credit
塌缩到跨界点之前的对齐段——共享块在链头时直接归 **0**——而 vLLM 明明可以从 CPU 完整复用 B
的前缀。方向严格保守(不会错误路由、不损坏数据、不泄漏),但**一个冷兄弟的驱逐摧毁了所有
仍然热的兄弟的 CPU 层路由信号**。共享前缀负载(system prompt、few-shot 模板)几乎不可能恰好
对齐 `offloaded_block_size`,重复度等于分叉数(16 会话实测 ×11)。

对照**对齐**的共享段(c0..c4):同 key、一份数据、一次 store/remove——若 c4 真被驱逐,
vLLM 自己的 lookup 也在同一点截断,router 与引擎一致,这是普通的 cache 行为,不是缺陷。
缺陷窗口恰好就是跨界 chunk 的共享块。

**修复——对已宣告 hash 做引用计数(vLLM `746d7dc26`)。**scheduler 在
`_pending_event_metadata` 旁边新增一个 map `_block_hash_ref_counts: hash → 存活 chunk 计数`
(per-hash、跨 chunk 共享——放不进 per-chunk 的 metadata)。在 `BlockStored` **发出时** ++
(而非 populate 时,这样从未 announce 的失败 store 不会留下减不掉的计数),驱逐时 −−;
hash 仅在计数归零时进入 `BlockRemoved`——恰好是它最后一份 CPU 数据消失的时刻。重复 *store*
保留在 wire 上(两个 router 都幂等)。大小上界为每个去重 offload hash 一项;`reset_cache`
清空。

验证:重新捕获(`planb_rc_evict`)显示 **0 个 un-announced removal**(per-hash announce
状态机 PASS,20 次共享 hash 的幂等重复宣告);llm-d 重放 71/71 + 516/516;Dynamo 重放
11/11;双场景 `overlap_probe` 显示旧 wire 形状把 B 打到 0/3,refcount 形状让 B 保持
**3/3**(A 正确退化为 1/3——它的共享块仍由 B 的 chunk 提供)。vLLM 单测 76/76,含新增的
重叠 chunk 引用计数测试。

## 5. 修复方案矩阵

| # | 在哪 | 做什么 | 状态 / 成本 |
|---|---|---|---|
| 1 | Dynamo `remove_blocks_impl` | 缺失 hash 跳过、排空整批(消 B1) | ✅ 已做,+20 行,545 单测全过;同时加固约束 6 的跨层乱序 |
| 2 | **vLLM 移除引用计数(对所有 router 消 B2)** | side table 增加 `hash → 存活 chunk 计数`(BlockStored 发出时 ++,evict −−);`BlockRemoved` 只携带归零的 hash。重复 *store* 保留(两个 router 都幂等)。 | ✅ 已做(`746d7dc26`);router 零改动;额外状态 O(存活 hash 数),`reset_cache` 清空 |
| 3 | router 侧引用计数 | Dynamo `worker_map` / llm-d pod entry 加计数 | 被方案 2 取代——同一修复做两份 |
| 4 | 部署规避 | 把共享前缀(system prompt / 模板)pad 到 `offloaded_block_size` 整数倍 | 被方案 2 取代 |

方案 2 是 Plan B 移除 side table 的自然延伸:表本来就按 chunk 存在,引用计数只是它的
per-hash 聚合;「最后一个引用消失才宣告移除」恰好就是 router 期待的 cache 语义。纵深防御:
方案 1 保留——有了方案 2,本 producer 不再产生重复移除,但 lower tier 仍应容忍未知 hash
(约束 6 的乱序、其他 producer)。

## 6. 复现

```bash
# 1) 抓含真实驱逐的 capture(写入 runs/planb_rc_evict/)
cd /home/changg/workspace/.tmp/llmd_4way && bash run_evict.sh planb_rc_evict planb 3 134217728
# 2) wire 形状 + 对账(含 per-hash announce 状态机)
python decode_capture.py runs/planb_rc_evict/capture.jsonl --factor 3 --block-size 16
# 3) llm-d store+remove 重放
cd /home/changg/workspace/llm-d-kv-cache-manager && go build -o chunk_replay_v2_bin ./examples/chunk_replay_v2 \
  && ./chunk_replay_v2_bin /home/changg/workspace/.tmp/llmd_4way/runs/planb_rc_evict/capture.jsonl
# 4) Dynamo store+remove 重放(只喂 CPU 事件,from-root 链走)
cd /home/changg/workspace/dynamo && DYNAMO_PLANB_CAPTURE=/home/changg/workspace/.tmp/llmd_4way/runs/planb_rc_evict/capture.jsonl \
  cargo test -p dynamo-kv-router --test planb_capture_replay -- --ignored --nocapture
# 5) 先驱逐者连坐:旧形状复现(B=0/3),refcount 形状消除(B=3/3)
cd /home/changg/workspace/llm-d-kv-cache-manager && go run ./examples/overlap_probe
# (refcount 之前的 capture runs/planb_chunk_evict/ 保留作回归重放)
```
