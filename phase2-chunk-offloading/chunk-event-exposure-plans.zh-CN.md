# Chunk Offloading —— KV Event 暴露方式：问题、修正方案与路线图

状态：记录团队讨论的设计笔记（Chang / Ziqi / John-DLAlgo / Lin Hu / reviewers）。
English: [`chunk-event-exposure-plans.md`](chunk-event-exposure-plans.md)。

阅读顺序：**Part 1** —— 当前 llm-d 怎么匹配 offload 下来的 CPU tier，以及这条路径上一个（少见的）时序边界；
**Part 2** —— 为什么 chunk offloading 在没有 `token_ids` 时会失效；**Part 3** —— 修正方案一览；**Part 4** ——
每个方案细节；**Part 5** —— 路线图与待讨论点；**附录** —— 代码级参考。来源：`llm-d-kv-cache-manager`，
`pkg/kvevents/pool.go` + `pkg/kvcache/kvblock/in_memory.go`。

---

## Part 1 —— 当前 llm-d 如何匹配 CPU tier（engineKey → requestKey）

### 1.1 两级索引
`InMemoryIndex` 维护**两个** map：

- `data: requestKey → PodCache` —— 哪些 pod × device-tier 持有每个 *canonical* block；`Lookup` 匹配的就是它。
- `engineToRequestKeys: engineKey → []requestKey` —— 从某 engine 自己的 block hash，映射到它覆盖的若干
  canonical request key 的 LRU。

### 1.2 request key 来自 token；engine key 是事件里的原始 hash
收到 `BlockStored` 时，pool 真正入索引的 key 是从事件的 **token** 算出来的：

```go
// pool.go
requestKeys, _ := p.tokenProcessor.TokensToKVBlockKeys(
    parentRequestKey, ev.Tokens, effectiveModelName, extraFeatures)
```

`requestKeys` 是 canonical / 内容寻址（prefix-chained）——**任何 client 都能从 token 重算**。`engineKeys` 只是
事件里带来的原始 hash。`index.Add` 然后同时记下 `data[requestKey] += podEntry` 与
`engineToRequestKeys[engineKey] = requestKeys`。

### 1.3 location-only 路径（**无 token** 的事件）
若事件**不带 token**，pool 算不出 request key，只能拿每个 engine hash 去**已有**的 map 里 resolve：

```go
// handleDeviceTierUpdate  （len(requestKeys) == 0 时走这里）
for _, ek := range engineKeys {
    rk, err := p.index.GetRequestKey(ctx, ek)   // 需要一个**已存在**的映射
    ...
}
```

这条路只是对早先（带 token 的）事件建好的映射的一个**引用**，它**无法创建新的** canonical key。

### 1.4 生命周期 —— `engineToRequestKeys` 何时被驱逐？（有界）
两条移除路径：

1. **显式**（`Evict` 的 `EngineKey` 分支）：`BlockRemoved(engineKey)` 把它映射的 request key 的 pod entry 删掉后，
   若**全部**为空 → `engineToRequestKeys.Remove(key)`。
2. **LRU 容量**：它是个 LRU（`Size`，默认 `1e8`）→ 超限时自动丢最旧条目。

显式那条**只在收到 `BlockRemoved` 时触发**，所以那些后端从不发 remove 的 key（崩溃、不发 removal 的 tier）会一直
留着——**真正兜底的是 LRU，而不是显式移除**；没有 LRU 它会无限涨。（有界，但默认上限很大、且这种 LRU 有**两个**
含 `data`，要按内存调。）

### 1.5 一个时序边界（少见）：GPU `BlockRemoved` 早于 CPU `BlockStored`
值得一提，但实践中**少见**。vLLM 的 CPU offload 可能是 eviction-driven，所以同一个 block 的 GPU remove 和 CPU
offload 时间很近；若 GPU `BlockRemoved` 先被处理、把 request key 清空，`engineToRequestKeys[hash]` 就被 `Remove`。
此时**无 token** 的 CPU 事件（靠 hash resolve，§1.3）会 miss 而不被索引；**带 token** 的事件不受影响——它用自己的
token 重算 key 并重新 `Add`，与先后顺序无关。

真正重要的**不是**这个很窄的时序窗口（按序的实时流很少撞上），而是它暴露的**结构性依赖**：无 token 的事件只是对
GPU 事件所建映射的一个*引用*，所以它在 **router 重启 / 新副本 / 漏收 GPU event** 时同样会失败。带 token 的事件
**自洽**，没有这些问题。（带 token 的事件保留一个很小的依赖——parent resolve，`GetRequestKey(parentHash)`，miss
则跳过——但这实践中安全，因为 parent 属于同一段被 offload 的 prefix。）

---

## Part 2 —— 为什么 chunk 在没有 `token_ids` 时会让当前 llm-d 失效

### 2.1 chunk 是什么
`block_size_factor > 1` 时，vLLM 把 `factor` 个 GPU block 打包成一个 CPU chunk
（`offloaded_block_size = block_size × factor`），换取更大更少的传输。内部每个 chunk 只用**一个** OffloadKey =
chunk 的 **tail block hash**（block hash 是 prefix-chained 的，tail 已唯一确定 chunk 内容）。驱逐时 manager 也只
还回这个 **tail** key。

### 2.2 上游 chunk 事件几乎什么都没告诉 router
上游的 chunk CPU `BlockStored` 是 placeholder：**只有一个 tail hash，`token_ids=[]`，`block_size=0`**。没有 token，
它只能走 location-only 路径（§1.3）、用 alias resolve 那一个 tail hash——所以每个 chunk 至多让 **tail block** 拿到
CPU 条目。chunk 里其余 `factor-1` 个 block **从未被 announce**，**根本没有任何映射**。

### 2.3 于是新请求什么都匹配不上
新请求会从它的 token 重算出**逐块**的 request key `r0 … r_{N-1}`（router 按 canonical block 大小工作），然后**连续**
walk prefix，遇到第一个没有 CPU 条目的块就停。第一个 chunk 的 block 0（`r0`）从未被 announce → miss → 连续 CPU
匹配在 **0** 就死了，尽管 chunk 的 KV 大部分其实就在 CPU RAM 里。真实多轮流量实测：by-block `factor=1` = **68/68**
连续，chunk `factor=3` = **0/68**。

### 2.4 结论
当前 llm-d **无法匹配 chunk**，除非事件能让 router 重建出 chunk 的**逐块** canonical key。唯一自洽的办法就是带上
chunk 的 `token_ids`（router 把它重新切成逐块 key）。这是把 `token_ids` 放进事件的**首要**理由（自洽性，§1.5，是
附带的额外好处）。

---

## Part 3 —— 修正方案一览

> 这个 1→N 展开总得**放在某个地方**。方案 A 把它放在 **router**，即 `E3 → [R0,R1,R2,R3]`（engineKey → requestKey）。
> 方案 B/C 把*等价*的映射放在 **vLLM**，即 `H3 → [H0,H1,H2,H3]`（chunk tail hash → 各 constituent hash），用于
> removal fan-out。**同一个 1:many，只是放在不同侧。**

| 方案 | wire / chunk | indexer 改动 | vLLM 额外状态 | 自洽 / 重启安全 | extra_keys | 结论 |
|---|---|---|---|---|---|---|
| **A** 单 hash + token，router 1:many | 1 hash + N·bs token | 1:many `E3→[R0..R3]`（llm-d ✓） | 无 | ✅ 是 | 暂缓（重算） | ✅ **现在做** |
| **B** hash list + token（= 原生 batched） | N hash + token | 无（原生形状） | 1:many map `H3→[H0..H3]` | ✅ 是 | 暂缓 | 试过，放弃 |
| **C** hash list、无 token，GPU resolve + TTL | N hash | 无（靠 alias 1:1） | 1:many map `H3→[H0..H3]` | ❌ 依赖 GPU event | 从 GPU 继承 | 不建议 |
| **D** tail hash、indexer 用 block_size 倒推 | 1 hash | 特判 chunk | parent 回溯状态 | ❌ 依赖 GPU event | 继承 | 不建议 |
| **E** Dynamo 单独 chunk index | （n/a） | chunk-aware index | n/a | n/a | n/a | later / 按需 |

---

## Part 4 —— 各修正方案细节

### 方案 A —— 单 hash + 完整 token，router 做 1:many  ← 当前 PR
事件：1 个代表 hash（chunk tail）+ 整块 `token_ids` + `block_size` = **单块**大小。router 把 token 重新切成
`factor` 个 block，并把这一个 engine hash 映射到全部——这个 1:many `E3 → [R0,R1,R2,R3]` 放在 router（见附录）。

- **优点：** 自洽——不依赖其它事件、重启安全（也对 §1.5 时序边界免疫，因为它用自己 token 重 `Add`）。带 token 方案
  里 wire 最小（1 个 hash/parent，而非 N 个）。vLLM **零额外状态**——remove 也只发同一个 hash，router 的 1:many 顺带
  删掉全部 block。indexer 看到**统一** `block_size`（不感知 chunk、无 mixed size）。复用 llm-d 现成 1:many。
- **缺点：** router 需要 1:many（llm-d 已有；Dynamo 加一点）。要带 token（每 chunk ~N 个 token id）。`extra_keys`
  （多模态 / `cache_salt`）暂缓。
- **为什么：** 让 router 能重建逐块 key（Part 2），同时把 1→N 展开放到一个通用、轻量的 router 能力上，vLLM 状态
  最小。已在 llm-d 端到端验证：覆盖 94.3%、连续 68/68、store + remove。

### 方案 B —— hash list + 完整 token（vLLM fan-out = 原生 batched 形状）  ← 试过，已放弃
事件：1 个 event，`block_hashes` = 全部 N 个 constituent hash + token + `block_size` = 单块大小（与原生 GPU batched
event 完全一样）。

- **优点：** indexer **零改动**（标准 1:1，N hash 对 N token block）。自洽。
- **缺点：** vLLM 必须维护**自己这边的 1:many map**——`H3 → [H0,H1,H2,H3]`（chunk tail → 各 constituent hash）——
  驱逐时才能 fan-out 到全部 N（evict 时 manager 只有 tail）。这就是**和方案 A 同一个 1:many，只是换了一侧**：方案 A
  把 `E3 → [R0..R3]` 放在 router，方案 B 把 `H3 → [H0..H3]` 放在 vLLM。wire 最大。
- **为什么放弃：** 1:many 总要放在某处；放在 vLLM 是按 CPU pool 生命周期常驻的额外状态，而方案 A 直接复用 router
  现成的 1:many，免费。

### 方案 C —— hash list、无 token，靠 GPU event resolve + 加长 TTL  ← reviewer 提的
事件：store/remove 都发 constituent hash 的 list、**不带 token**；indexer 用早先 GPU event 建的 alias resolve；加长
TTL 让 alias 活得比 CPU 引用久。

- **优点：** wire 最小（只有 hash）。indexer 不感知 chunk。可躲开 CPU 侧 `extra_keys`（复用 vLLM 自己的 hash）。
- **缺点：** **不自洽**——CPU 条目只有在 GPU alias 被收到且保留时才能 resolve，所以**重启不安全**（新 router / 新副本
  没有 GPU 历史），也暴露在 §1.5 的时序边界下；"加长 TTL"把它变成调参旋钮。而且**照样需要同一个 vLLM 侧 1:many map**
  `H3 → [H0..H3]`（remove 发 list，但 evict 只给 tail）。
- **为什么不：** 用自洽性换一点 payload；KV event 不是带宽瓶颈（KV data 传输才是）。

### 方案 D —— 只发 tail hash，indexer 用 block_size 识别 chunk 再倒推
事件：tail hash + `block_size` = **chunk** 大小。indexer 判定是 chunk 并重建 constituent block（如沿 parent chain 回溯
`factor` 步）。

- **优点：** wire 最小（1 hash）。
- **缺点：** 逼 indexer **特判 chunk**（mixed block size + parent 回溯）——恰是 reviewer 想避免的复杂度。hash 不可逆，
  拿不到兄弟块。和方案 C 一样依赖 GPU event。
- **为什么不：** 所有方案里 indexer 复杂度最高；违背"indexer 不该 care chunking"。

### 方案 E —— Dynamo 侧单独的 chunk index（正交方案）
不（或不只）把 chunk 展开到 block 粒度，而在 Dynamo 维护 chunk-aware index，统一各后端的 chunk 概念（vLLM native、
LMCache、KVBM）。

- **优点：** 跨后端的统一 chunk 抽象；多个 chunk 来源汇聚时 future-proof。
- **缺点：** 重；把 chunk 感知塞回 indexer；更多状态 / 维护面；单一来源时 premature。方案 A 已让 chunk 对索引透明。
- **为什么（later）：** 只有当 ≥2 个真实 chunk 来源证明需要统一抽象时才值得。

---

## Part 5 —— 路线图与待讨论点

### 路线图（stage）
- **Stage 1（现在）—— 方案 A，纯文本 full-attention。** 在现有 PR 上发单 hash + token 的 chunk event。llm-d 当下
  就能用；Dynamo 加 1:many。再加个小 guard，让多模态 / `cache_salt` 请求在 `extra_keys` 落地前回退 placeholder。→ 打通 POC。
- **Stage 2 —— 补全 payload。** 填 `extra_keys`（by-block 容易；chunk 需定"一个 hash vs 每块 extra_keys"的 contract）；
  接上 sliding-window / SSM group（现在是 placeholder；需 router 侧 window/state-aware 匹配）。
- **Stage 3 —— Dynamo 单独 chunk index（方案 E），按需。** 当确需跨后端统一 chunk 抽象（LMCache / KVBM + vLLM）时再做。

*（Stage 2 与 3 可按"多模态支持"还是"多后端统一"哪个优先来调换。）*

### 待讨论 / 可能被 challenge 的点
- **"token payload 不小。"** 对，但 normal（by-block）offloading 本来也带 token；方案 A 是带 token 里最省的，而且 KV
  event 不是带宽瓶颈。
- **"indexer 不该 care chunking"（DLAlgo）。** 同意——方案 A 已满足：统一 `block_size`、无 mixed size；唯一需要的是通用
  的 1:many 映射。
- **"能不能完全不传 token？"（方案 C/D）。** 只会牺牲自洽性——一个 GPU event 依赖 + 重启脆弱性（§1.5）——而且照样需要
  vLLM 侧的 1:many map。
- **`extra_keys`（多模态 / `cache_salt`）。** 暂缓；和 PR 上 by-block 路径共有的正确性 caveat。Stage 1 建议加 guard。
- **1:many 放在哪。** 它总得放在某处：**A** → router `E3→[R0..R3]`（通用、llm-d 现成）；**B/C** → vLLM `H3→[H0..H3]`
  （removal fan-out）；**D** → indexer（chunk 特判）。只有 A 在 vLLM 侧不加新状态、在 indexer 侧不加 chunk 专属逻辑。

---

## 附录 —— 代码级参考：`Add` / 1:many / `Evict` / parent 链

### `Add` 用长度比例推断映射类型 —— 这就是 router 侧的 1:many
```go
// in_memory.go — Add()
//   equal  (4 eng, 4 req) -> 1:1    E0->R0, E1->R1, ...
//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, ...
//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
n := max(len(engineKeys), len(requestKeys))
for i := 0; i < n; i++ {
    ek := engineKeys[i*len(engineKeys)/n]
    rk := requestKeys[i*len(requestKeys)/n]
    newMappings[ek] = append(newMappings[ek], rk)
}
```
对方案 A 的 chunk event：`engineKeys = <1 个 tail hash>`、`requestKeys = <N 个 token 算出的 key>` → 走 `1:many`
分支 → `engineToRequestKeys[tail] = [r0…r_{N-1}]`（即 `E3 → [R0..R3]`），且这 N 个 request key 在 `data` 里都拿到 CPU
`PodEntry`。Lookup 只在 request key（token 算出的）上走——所以 chunk 对匹配透明，就像 N 个普通 block。

### `Evict` 是对称的 1:many —— 方案 A 为什么不需要 vLLM removal 状态
```go
// in_memory.go — Evict(), EngineKey 分支
rks, _ := m.engineToRequestKeys.Get(key)   // = [r0…r_{N-1}]
for _, rk := range rks {
    m.evictPodsFromRequestKey(rk, key, entries, ...)   // 从全部 N 个里删掉这个 pod entry
}
// 当每个 rk 都空了，再把 engineKey 映射也删掉
```
一个 engine hash → 连带删掉全部 N 个 canonical block。这正是 vLLM 能在 remove 时只发一个 tail hash、零 per-chunk 状态
的原因——router 的 `E3 → [R0..R3]` 负责 fan-out。（方案 B/C 则把这个 fan-out 挪到 vLLM 的 `H3 → [H0..H3]`。）

### 跨 chunk 的 parent 链
`GetRequestKey(engineKey)` 返回 `rks[len-1]`——该 engine hash 的**最后一个** request key。chunk *c* 的
`parent_block_hash` 是 chunk *c-1* 的 tail engine hash，于是 resolve 到 chunk *c-1* 的最后一个 block——正好是 chunk *c*
第一个 block 的正确 parent——canonical key 因此保持连续。

### 为什么这么设计
- **engineKey 与 requestKey 分离。** request key 是 canonical / 内容寻址（来自 token），异构 engine/后端能收敛到**同一个**
  可匹配 keyspace；router 匹配 request key，而它人人都能从 token 重算。
- **比例规则是通用的，不是 chunk 专属。** 同一段代码服务 1:1、many:1（去重）、1:many。indexer 根本没有"chunk"概念——
  chunk 只是一个 1:many 比例。这就是方案 A 让 chunk 对索引透明的根因。
- **`engineToRequestKeys` 本来就有用**：(a) location-only / device-tier 更新，(b) 驱逐，(c) parent resolve。1:many 是从
  这个已有结构里自然长出来的，而不是新加的特性。

### `engineKey → requestKey` 这个映射到底必要吗？（很多时候是 identity）
llm-d 的 `requestKey` 是**被设计来复现 vLLM 的 block hash 的**，不是另一套方案：

- llm-d（`token_processor.go`）：`requestKey = fnv64a(CBOR([parent, tokens, extra]))`，prefix-chained，`initHash`
  由 `HashSeed` 种子化。注释明说：*"mimics the chunkedTokenDatabase in the Python code"*、*"compatibility with
  vLLM's prefix caching algorithm"*、`HashSeed` *"aligned with vLLM's `PYTHONHASHSEED`"*。
- vLLM（`kv_cache_utils.py`）：block hash 用 CBOR-based 函数（`sha256_cbor` / `xxhash_cbor`）对 `[parent, tokens,
  extra_keys]` 计算，`NONE_HASH` 由 `PYTHONHASHSEED` 种子化。

同样的 CBOR 编码 `[parent, tokens, extra]`、同样的 prefix chain、同样的 seed → **只要部署对齐了 hash 函数 + seed，
`requestKey == engineKey`（逐块相等）。**（字节级相等需要函数也配对——vLLM `*_cbor` vs llm-d `fnv64a-cbor`——所以这是
个部署对齐要求，llm-d 配置里专门点了出来。）

对这个映射的影响：

| 用途 | 需要这个映射吗？ |
|---|---|
| Lookup（匹配新请求） | **从不**——从 token 重算出（相同的）key，直接命中 `data`。 |
| factor=1 removal（已对齐） | **冗余**——engine hash *就是* request key，可直接 evict。 |
| **chunk removal（方案 A 的 1:many）** | **必要**——tail engineKey 只等于最后一个 request key `R_{N-1}`；`R0…R_{N-2}` 是另外的 hash，从 tail 不可逆推，而 removal 只带 tail。 |
| 异构 / 未对齐的 engine | **必要**——翻译 `engineKey ≠ requestKey`。 |

所以对纯粹已对齐的 by-block，这个映射本质上就是 **identity（冗余）**；它真正承重只在 **方案 A 的 chunk removal** 和
**跨 engine 翻译**。推论：`E3 → [R0,R1,R2,R3]` 里，对齐后 `R3 == E3`，所以这个映射其实只是存了 tail hash 自己无法
再生成的 `R0,R1,R2`——这正是单 tail 的 `BlockRemoved` 需要它的原因。