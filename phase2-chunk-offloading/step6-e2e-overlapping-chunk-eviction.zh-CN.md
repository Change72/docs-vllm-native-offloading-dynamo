# Phase 2 · Step 6 — 重叠 chunk 驱逐隐患与当前合并决策

> **Step 5** 定下 wire shape：vLLM 对每个 offloaded chunk 发布一条 `BlockStored`，里面带
> 全部 `factor` 个组成 block hash 和整 chunk 的 token；CPU 驱逐时再发一条 `BlockRemoved`
> fan out 同一组 hash。
>
> **Step 6** 用真实 CPU 驱逐和非对齐 shared-prefix 流量压测了这个 wire shape。重点结论不是
> 需要一个新的 Dynamo lower-tier patch。当前 shipped 设计是：**vLLM plain fan-out +
> consumer 侧 refcount/dedup**。Dynamo 标准 worker publisher 路径里已经有
> `EventDedupFilter`，正好提供所需语义。

## TL;DR

| 层 | 验证内容 | 结果 |
|---|---|---|
| vLLM wire | 真实 serve、chunked offload、小 CPU pool、真实 LRU 驱逐 | CPU `BlockStored` / `BlockRemoved` 是自描述 chunk 事件；store 带组成 hash + 整 chunk tokens；remove fan out hashes |
| overlap hazard | shared prefix 长度不是 `offloaded_block_size` 的整数倍 | 兄弟分支的边界 chunk 可能合法地列出同一个共享 block hash，因此重复 store/remove announcement 是预期行为 |
| Dynamo shipped path | worker publisher 先 normalize，再跑 `EventDedupFilter`，然后进入 lower-tier indexer | 重复 per-hash announcement 在到达 `LowerTierIndexer` 前已经 ref-count/dedup |
| 无 filter consumer | 没有 refcount/dedup 的单值 indexer | 可能在第一个兄弟 chunk 被驱逐后保守地少算仍然驻留的共享块；这是 under-credit，不是数据错误 |

合并规则：**不要要求 vLLM 把 chunk removal 做成每个 block hash exactly-once。**
vLLM PR 是 self-describing event producer。任何把 chunk event 按 per-block 粒度建索引的
consumer，都需要处理重叠 chunk 产生的重复 announcement。

## 为什么重复是合理的

触发条件：`shared_prefix_len % offloaded_block_size != 0`。假设 GPU block 是 16 tokens，
offload chunk 是 48 tokens（`factor=3`），一个 256-token shared prefix 会落在 chunk 中间：

```text
              共享 256-token prefix                       会话私有部分
blocks:   0  1  2 │ 3  4  5 │ 6  7  8 │ 9 10 11 │ 12 13 14 │ 15 16A 17A ...   conv A
                                                            │ 15 16B 17B ...   conv B
chunks:   └─ c0 ─┘ └─ c1 ─┘ └─ c2 ─┘ └── c3 ──┘ └── c4 ──┘ └── c5 ───┘
          c0..c4: 对齐 shared chunks -> 同一个 OffloadKey，一份 CPU copy
          c5:     边界 chunk 跨过分叉点：
                  c5A = [h15, h16A, h17A]
                  c5B = [h15, h16B, h17B]
```

`h15` 是共享块，但两个 chunk 的后续块不同，所以它们的 `OffloadKey` 不同。vLLM 内部按
chunk key 去重是正确的：这里确实有两个独立 CPU chunk。落到 event wire 上就会变成：

```text
store  c5A -> BlockStored [h15, h16A, h17A]
store  c5B -> BlockStored [h15, h16B, h17B]    # h15 再次出现
evict  c5A -> BlockRemoved[h15, h16A, h17A]
evict  c5B -> BlockRemoved[h15, h16B, h17B]    # h15 再次出现
```

by-block 模式（`factor=1`）没有这个形状，因为存储去重单位和 router 的 per-block 索引单位一致。
chunk mode 刻意使用更大的存储单位，所以重复 per-block hash announcement 是 contract 的一部分。

## consumer 需要什么语义

按 per-block hash 建 lower-tier 索引的 consumer 需要以下语义之一：

1. **索引前 refcount/dedup。** 这是 Dynamo 标准路径。worker publisher 的
   `EventDedupFilter` 按 `(dp_rank, tier, block_hash)` 维护状态：重复 store 增计数，只有最后
   一个 live reference 消失时才放行 remove。
2. **indexer 内部 refcount。** 语义等价，只是实现位置更低。
3. **接受保守 under-credit。** 没有 filter 的单值 consumer 可能在第一个兄弟 chunk 被驱逐时删掉
   共享 hash，即使另一个兄弟 chunk 仍然包含它。这会丢 routing credit，但不会让 engine 返回错误数据。

当前 Dynamo 合并路径使用第 1 种。

## 当前 PR 不包含什么

- Dynamo PR **不依赖** `remove_blocks_impl` skip-absent-hashes patch。早期 direct lower-tier
  replay 用它做过诊断性 hardening 实验，但生产 event path 在 lower-tier indexer 前面有
  `EventDedupFilter`。
- vLLM PR **不发布** producer 侧 exactly-once per-hash removal refcount。那个变体用于证明隐患，
  但团队决策是 producer 保持 plain fan-out，由 consumer dedup 重叠 chunk announcement。
- 没有 filter 的 consumer（例如 direct llm-d replay）如果希望在非对齐 shared-prefix 负载下获得
  精确 CPU-tier credit，需要自己实现 refcount/dedup。

## 和 Step 7 的关系

Step 7 验证的是当前真正要合并的 Dynamo 路径：

- vLLM PR #43468，opt-in self-describing chunk events
- Dynamo PR #10368，`medium="CPU"` 路由到 HostPinned，并补上 lower-tier metrics wiring
- worker publisher 路径里有 `EventDedupFilter`
- 小 CPU pool 触发真实 CPU eviction
- wire capture 与 `kv_cache_events_applied` 精确对账

因此 Step 7 是合并相关证据；Step 6 是设计风险记录，用来解释为什么重叠 chunk 的重复
announcement 是预期行为，以及为什么 consumer 必须 dedup。
