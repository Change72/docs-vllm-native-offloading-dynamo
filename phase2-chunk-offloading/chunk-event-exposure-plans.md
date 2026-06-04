# Chunk Offloading — KV-Event Exposure: Problems, Fix Plans & Roadmap

## Part 1 — How llm-d matches the CPU tier today (engineKey → requestKey)

### 1.1 Two-level index
`InMemoryIndex` keeps **two** maps:

- `data: requestKey → PodCache` — which pods × device-tiers hold each *canonical* block; this is what
  `Lookup` matches on.
- `engineToRequestKeys: engineKey → []requestKey` — an LRU from an engine's own block hash to the
  canonical request key(s) it covers.

### 1.2 Request keys come from tokens; engine keys are the raw event hashes
On a `BlockStored`, the pool derives the keys it indexes from the event's **tokens**:

```go
// pool.go
requestKeys, _ := p.tokenProcessor.TokensToKVBlockKeys(
    parentRequestKey, ev.Tokens, effectiveModelName, extraFeatures)
```

llm-d's `requestKey` is **built to reproduce vLLM's block hash**. See Appendix.

`requestKeys` are canonical / content-addressed (prefix-chained) — **any client can recompute them
from tokens**. `engineKeys` are just the raw hashes carried in the event. `index.Add` then records
both `data[requestKey] += podEntry` and `engineToRequestKeys[engineKey] = requestKeys`.

### 1.3 The location-only path (events *without* tokens)
If an event carries **no tokens**, the pool can't compute request keys, so it falls back to resolving
each engine hash against the *existing* map:

```go
// handleDeviceTierUpdate  (used when len(requestKeys) == 0)
for _, ek := range engineKeys {
    rk, err := p.index.GetRequestKey(ctx, ek)   // needs a PRE-EXISTING mapping
    ...
}
```

This path is only a *reference* into the mapping built by earlier (token-carrying) events — it cannot
create new canonical keys.

### 1.4 Lifecycle — when is `engineToRequestKeys` evicted? (it is bounded)
Two removal paths:

1. **Explicit** (`Evict`, `EngineKey` case): a `BlockRemoved(engineKey)` drops the pod entries from its
   request keys; when **all** are empty → `engineToRequestKeys.Remove(key)`.
2. **LRU capacity**: it is an LRU (`Size`, default `1e8`) → the oldest entry is dropped when full.

The explicit path only fires on a `BlockRemoved`, so keys whose backend never emits a remove (crash,
a tier that doesn't publish removals) linger — **the LRU is the real backstop, not the explicit
removal**. Without it the map would grow unbounded. (Bounded, but the default cap is large and there
are *two* such LRUs incl. `data`; size them to your memory.)

---

## Part 2 — Why chunking breaks current llm-d (without `token_ids`)

### 2.1 What a chunk is
With `block_size_factor > 1`, vLLM groups `factor` GPU blocks into one CPU chunk
(`offloaded_block_size = block_size × factor`) for larger / fewer transfers. Internally each chunk is
keyed by a **single** OffloadKey = the chunk's **tail block hash** (block hashes are prefix-chained,
so the tail uniquely identifies the chunk's content). At eviction the manager hands back only that
**tail** key.

### 2.2 The upstream chunk event tells the router almost nothing
Upstream, a chunk's CPU `BlockStored` is a placeholder: **one tail hash, `token_ids=[]`,
`block_size=0`**. With no tokens, it can only take the location-only path (§1.3) and resolve that one
tail hash via the alias — so at most the **tail block** of each chunk gets a CPU entry. The other
`factor-1` blocks of the chunk are **never announced** and have **no mapping at all**.

### 2.3 A new request therefore matches nothing
A new request recomputes **per-block** request keys `r0 … r_{N-1}` from its tokens (the router works
at the canonical block size). It then walks the prefix **contiguously** and stops at the first block
with no CPU entry. Block 0 of the first chunk (`r0`) was never announced → miss → the contiguous CPU
match dies at **0**, even though most of the chunk's KV physically sits in CPU RAM. Measured on real
multi-turn traffic: by-block `factor=1` = **68/68** contiguous, chunk `factor=3` = **0/68**.

### 2.4 Conclusion
Current llm-d **cannot match a chunk** unless the event lets the router reconstruct the chunk's
**per-block** canonical keys. The only self-contained way to do that is to carry the chunk's
`token_ids` (the router re-splits them into per-block keys). This is the primary reason to put
`token_ids` in the event (self-containedness, §1.5, is a secondary bonus).

---

## Part 3 — Fix plans at a glance

> The 1→N expansion has to live **somewhere** under the chunk mode. Plan A keeps it in the **router** as
> `E3 → [R0,R1,R2,R3]` (engineKey → requestKeys). Plans B/C keep the *equivalent* map in **vLLM** as
> `H3 → [H0,H1,H2,H3]` (chunk-tail hash → constituent hashes) for removal fan-out. Same 1:many,
> different side.

| Plan | Wire / chunk | Router change | vLLM change | Self-contained / restart-safe | extra_keys | Verdict |
|---|---|---|---|---|---|---|
| **A** single hash + tokens, router 1:many | 1 hash + N·bs tokens | 1:many `E3→[R0..R3]` (llm-d ✓) | none | ✅ yes | deferred (recompute) | ✅ **now** |
| **B** list of hashes + tokens (= native batched) | N hashes + tokens | none (native shape) | 1:many map `H3→[H0..H3]` | ✅ yes | deferred | tried, dropped |
| **C** list of hashes, NO tokens, GPU-resolve + TTL | N hashes | none (1:1 via alias) | 1:many map `H3→[H0..H3]` | ❌ depends on GPU events (External hash -> local hash) | inherited from GPU | not recommended |
| **D** tail hash, indexer back-derives via block_size | 1 hash | chunk special-case | parent-walk state | ❌ depends on GPU events (External hash -> local hash) | inherited | not recommended |
| **E** Dynamo dedicated chunk index | (n/a) | chunk-aware index | n/a | n/a | n/a | later / if needed |

---

## Part 4 — Fix plans in detail

### Plan A — Single hash + full tokens, router does 1:many  ← current PR
Event: 1 representative hash (chunk tail) + whole-chunk `token_ids` + `block_size` = **per-block**
size. The router re-splits tokens into `factor` blocks and maps the one engine hash to all of them —
the 1:many `E3 → [R0,R1,R2,R3]` lives in the router (see Appendix).

- **Pros:** Self-contained — no dependency on other events; restart-safe (and immune to the §1.5
  timing edge, since it re-`Add`s from its own tokens). Smallest of the token-carrying options (1
  hash/parent, not N). vLLM keeps **zero** extra state — removal sends the same single hash and the
  router's 1:many evicts all blocks. Indexer sees a **uniform** `block_size` (no chunk awareness, no
  mixed sizes). Reuses llm-d's existing 1:many.
- **Cons:** Needs a 1:many mapping in the router (llm-d has it; Dynamo: small addition). Carries
  tokens (~N token-ids/chunk). `extra_keys` (multimodal / `cache_salt`) deferred.
- **Why:** Lets the router reconstruct the per-block keys (Part 2) while pushing the 1→N expansion to
  a generic, lightweight router capability and keeping vLLM state minimal. Verified e2e on llm-d:
  94.3% coverage, 68/68 contiguous, store + remove.

### Plan B — List of hashes + full tokens (vLLM fan-out = native batched shape)  ← tried, dropped
Event: 1 event, `block_hashes` = all N constituent hashes + tokens + `block_size` = per-block size
(identical to vLLM's native GPU batched event).

- **Pros:** Indexer needs **zero** new logic (normal 1:1, N hashes ↔ N token-blocks). Self-contained.
- **Cons:** vLLM must keep its **own 1:many map** — `H3 → [H0,H1,H2,H3]` (chunk tail → all constituent
  hashes) — so eviction can fan the removal out to all N (at evict time the manager only has the tail).
  This is the **same 1:many as Plan A, just relocated**: Plan A keeps `E3 → [R0..R3]` in the router;
  Plan B keeps `H3 → [H0..H3]` in vLLM. Largest wire.
- **Why dropped:** The 1:many has to live somewhere; putting it in vLLM is extra per-chunk state for
  the whole CPU-pool lifetime, whereas Plan A reuses the router's existing 1:many for free.

### Plan C — List of hashes, NO tokens, resolve via GPU events + longer TTL  ← proposed by reviewers
Event: store/remove send a list of constituent hashes, **no tokens**; the indexer resolves each hash
via the alias built from earlier GPU events; TTL extended so the alias outlives the CPU reference.

- **Pros:** Smallest wire (hashes only). Indexer needs no chunk awareness. Can sidestep CPU-side
  `extra_keys` (reuses vLLM's own hashes, which already encode extra_keys).
- **Cons:** **Not self-contained** — CPU entries resolve only if the GPU-event alias was seen and
  retained, so it is **restart-unsafe** (a fresh router / new replica has no GPU history) and also
  exposed to the §1.5 timing edge; "longer TTL" turns this into a tuning knob. And it **still needs the
  same vLLM-side 1:many map** `H3 → [H0..H3]` (remove sends a list, but evict yields only the tail).
- **Why not:** Trades self-containedness for a small payload saving; KV events are not the bandwidth
  bottleneck (KV data transfer is).

### Plan D — Tail hash only; indexer detects chunk via block_size and back-derives
Event: tail hash + `block_size` = **chunk** size. The indexer infers a chunk and reconstructs the
constituent blocks (e.g. walk the parent chain back `factor` steps).

- **Pros:** Minimal wire (1 hash).
- **Cons:** Forces the indexer to **special-case chunks** (mixed block sizes + parent-chain walk) —
  exactly the complexity reviewers want to avoid. A hash can't be inverted to its siblings without
  extra state. Same GPU-event dependency as Plan C.
- **Why not:** Most indexer complexity of all options; contradicts "the indexer shouldn't care about
  chunking."

### Plan E — Dynamo-side dedicated chunk index  (orthogonal)
Instead of (or on top of) expanding chunks to block granularity, Dynamo keeps a chunk-aware index, to
unify chunk concepts across backends (vLLM native, LMCache, KVBM).

- **Pros:** One chunk abstraction across backends; future-proof if multiple chunk sources converge.
- **Cons:** Heavy; re-introduces chunk awareness into the indexer; more state / maintenance;
  premature with a single chunk source. Plan A already makes chunking transparent to the index.
- **Why (later):** Worth it only once ≥2 real chunk sources demonstrate the need for a shared
  abstraction.

---

## Part 5 — Recommended roadmap & open concerns

### Roadmap (stages)
- **Stage 1 (now) — Plan A, text full-attention.** Ship the single-hash + tokens chunk event (on the
  existing PR). llm-d works today; Dynamo adds the 1:many mapping. Add a small guard so multimodal /
  `cache_salt` requests fall back to the placeholder until `extra_keys` lands. → unblocks the POC.
- **Stage 2 — complete the payload.** Fill `extra_keys` (by-block is easy; chunk needs the
  "one-hash vs per-block extra_keys" contract decision) and wire sliding-window / SSM groups
  (currently placeholder; needs router-side window/state-aware matching).
- **Stage 3 — Dynamo dedicated chunk index (Plan E), only if needed.** When a backend-agnostic chunk
  abstraction (LMCache / KVBM + vLLM) is actually required.

*(Stage 2 vs 3 can reorder depending on whether multimodal support or multi-backend unification is the
higher priority.)*

### Open concerns / likely challenges
- **"The token payload isn't small."** True, but normal (by-block) offloading carries tokens too;
  Plan A is the smallest token-carrying shape, and KV events aren't the bandwidth bottleneck.
- **"The indexer shouldn't care about chunking" (DLAlgo).** Agreed — Plan A satisfies it: uniform
  `block_size`, no mixed sizes; the only thing required is the generic 1:many mapping.
- **"Can we avoid tokens entirely?" (Plans C/D).** Only by giving up self-containedness — a GPU-event
  dependency + restart fragility (§1.5) — and it still needs the vLLM-side 1:many map anyway.
- **`extra_keys` (multimodal / `cache_salt`).** Deferred; a correctness caveat shared with the
  by-block path already on the PR. Guard recommended for Stage 1.
- **Where the 1:many lives.** It must live somewhere: **A** → router `E3→[R0..R3]` (generic, llm-d has
  it); **B/C** → vLLM `H3→[H0..H3]` (removal fan-out); **D** → indexer (chunk special-casing). A is the
  only one that adds nothing new on the vLLM side and nothing chunk-specific on the indexer side.

---

## Appendix — Code-level reference: `Add` / 1:many / `Evict` / parent chaining

### `Add` infers the mapping from the length ratio — this is the router-side 1:many
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
For a Plan-A chunk event: `engineKeys = <1 tail hash>`, `requestKeys = <N token-derived keys>` → the
`1:many` branch → `engineToRequestKeys[tail] = [r0…r_{N-1}]` (i.e. `E3 → [R0..R3]`), and all N request
keys get the CPU `PodEntry` in `data`. Lookup walks the request keys (token-derived) only — so the
chunk is invisible to matching; it looks like N ordinary blocks.

### `Evict` is the symmetric 1:many — why Plan A needs no vLLM removal state
```go
// in_memory.go — Evict(), EngineKey case
rks, _ := m.engineToRequestKeys.Get(key)   // = [r0…r_{N-1}]
for _, rk := range rks {
    m.evictPodsFromRequestKey(rk, key, entries, ...)   // drop the pod entry from ALL N
}
// when every rk is empty, drop the engineKey mapping too
```
One engine hash → all N canonical blocks evicted. That is why vLLM can send a single tail hash on
removal and keep zero per-chunk state — the router's `E3 → [R0..R3]` does the fan-out. (Plans B/C move
this fan-out to vLLM's `H3 → [H0..H3]` instead.)

### Parent chaining across chunks
`GetRequestKey(engineKey)` returns `rks[len-1]` — the **last** request key of that engine hash. Chunk
*c*'s `parent_block_hash` is chunk *c-1*'s tail engine hash, so it resolves to chunk *c-1*'s last
block — the correct parent for chunk *c*'s first block — keeping the canonical keys contiguous.

### Why it's designed this way
- **engineKey vs requestKey separation.** Request keys are canonical / content-addressed (from
  tokens), so heterogeneous engines/backends with different internal hashing converge on **one**
  matchable keyspace; the router matches on request keys, which any client can recompute.
- **The ratio rule is generic, not chunk-specific.** The same code serves 1:1, many:1 (dedup), and
  1:many. The indexer never has a "chunk" concept — a chunk is merely a 1:many ratio. This is the root
  reason Plan A keeps chunking transparent to the index.
- **`engineToRequestKeys` already earns its keep** for (a) location-only / device-tier updates, (b)
  eviction, and (c) parent resolution. 1:many falls out of this existing structure rather than being a
  new feature.

### Is the `engineKey → requestKey` mapping even necessary? (often it's identity)
llm-d's `requestKey` is **built to reproduce vLLM's block hash**, not a separate scheme:

- llm-d (`token_processor.go`): `requestKey = fnv64a(CBOR([parent, tokens, extra]))`, prefix-chained,
  `initHash` seeded from `HashSeed`. Comments: *"mimics the chunkedTokenDatabase in the Python code,"*
  *"compatibility with vLLM's prefix caching algorithm,"* `HashSeed` *"aligned with vLLM's
  `PYTHONHASHSEED`."*
- vLLM (`kv_cache_utils.py`): block hashes use CBOR-based functions (`sha256_cbor` / `xxhash_cbor`)
  over `[parent, tokens, extra_keys]`, with `NONE_HASH` seeded from `PYTHONHASHSEED`.

Same CBOR encoding of `[parent, tokens, extra]`, same prefix chain, same seed → **when the deployment
aligns the hash function + seed, `requestKey == engineKey` per block.** (Byte-equality needs a matching
function pair — vLLM `*_cbor` vs llm-d `fnv64a-cbor` — so it's a deployment-alignment requirement the
llm-d config flags.)

Consequences for the mapping:

| use | needs the mapping? |
|---|---|
| Lookup (match a new request) | **Never** — recompute the (identical) key from tokens, hit `data` directly. |
| factor=1 removal (aligned) | **Redundant** — the engine hash *is* the request key; evict directly. |
| **chunk removal (Plan A 1:many)** | **Necessary** — the tail engineKey equals only the last request key `R_{N-1}`; `R0…R_{N-2}` are distinct hashes you can't invert from the tail, and removal carries only the tail. |
| heterogeneous / unaligned engines | **Necessary** — translates `engineKey ≠ requestKey`. |

So for plain aligned by-block the mapping is essentially **identity (redundant)**; it is load-bearing
only for **Plan A's chunk removal** and **cross-engine translation**. Corollary: in `E3 → [R0,R1,R2,R3]`,
alignment makes `R3 == E3`, so the mapping really just stores the `R0,R1,R2` the tail hash alone can't
regenerate — which is exactly why a single-tail `BlockRemoved` needs it.
