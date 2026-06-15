# Phase 2 · Step 6 — overlapping chunk eviction hazard and shipped decision

> **Step 5** picked the wire shape: vLLM publishes each offloaded chunk as one
> `BlockStored` carrying all `factor` constituent per-block hashes plus whole-chunk tokens, and a
> `BlockRemoved` fanning out the same hashes at CPU eviction.
>
> **Step 6** stress-tested that wire shape with real CPU evictions and non-aligned shared-prefix
> traffic. The important outcome is not a new Dynamo lower-tier patch. The shipped design is
> **plain fan-out from vLLM + consumer-side refcount/dedup**. Dynamo's standard worker publisher
> path already provides the needed dedup semantics through `EventDedupFilter`.

## TL;DR

| layer | what was verified | result |
|---|---|---|
| vLLM wire | real serve, chunked offload, small CPU pool, real LRU evictions | CPU `BlockStored` and `BlockRemoved` are self-describing chunk events; stores carry constituent hashes + whole-chunk tokens; removes fan out hashes |
| overlap hazard | shared prefix length not divisible by `offloaded_block_size` | sibling boundary chunks can legitimately list the same shared block hash, so duplicate store/remove announcements are expected |
| Dynamo shipped path | worker publisher normalizes events and runs `EventDedupFilter` before lower-tier indexing | duplicate per-hash announcements are ref-counted/deduplicated before they reach `LowerTierIndexer` |
| filter-less consumers | direct single-entry indexers without refcount/dedup | can conservatively under-credit a still-resident shared block after the first sibling chunk is evicted; this is not data corruption |

The key merge rule: **do not require vLLM to make chunk removals exactly-once per block hash.**
The vLLM PR is a self-describing event producer. Consumers that interpret chunk events at
per-block granularity must handle duplicate overlapping announcements.

## Why duplicates are real

Trigger: `shared_prefix_len % offloaded_block_size != 0`. With 16-token GPU blocks and a
48-token offload chunk (`factor=3`), a 256-token shared prefix ends in the middle of a chunk:

```text
              shared 256-token prefix                      per-conversation
blocks:   0  1  2 │ 3  4  5 │ 6  7  8 │ 9 10 11 │ 12 13 14 │ 15 16A 17A ...   conv A
                                                            │ 15 16B 17B ...   conv B
chunks:   └─ c0 ─┘ └─ c1 ─┘ └─ c2 ─┘ └── c3 ──┘ └── c4 ──┘ └── c5 ───┘
          c0..c4: aligned shared chunks -> same OffloadKey, one CPU copy
          c5:     boundary chunk crosses the fork:
                  c5A = [h15, h16A, h17A]
                  c5B = [h15, h16B, h17B]
```

Block `h15` is shared, but the two chunks have different `OffloadKey`s because the later blocks
diverge. vLLM's chunk-level dedup is correct internally: there are two independent CPU chunks.
On the event wire that means:

```text
store  c5A -> BlockStored [h15, h16A, h17A]
store  c5B -> BlockStored [h15, h16B, h17B]    # h15 appears again
evict  c5A -> BlockRemoved[h15, h16A, h17A]
evict  c5B -> BlockRemoved[h15, h16B, h17B]    # h15 appears again
```

By-block mode (`factor=1`) does not have this shape because the dedup unit is the same as the
router's per-block index unit. Chunk mode intentionally uses a larger storage unit, so duplicate
per-block hash announcements are part of the contract.

## What this means for consumers

A consumer that indexes lower-tier cache by per-block hash needs one of these behaviors:

1. **Refcount/dedup before indexing.** This is Dynamo's standard path. The worker publisher's
   `EventDedupFilter` tracks per `(dp_rank, tier, block_hash)` state, increments on duplicate
   stores, and only forwards a remove when the last live reference disappears.
2. **Refcount in the indexer itself.** Equivalent semantics, but implemented lower in the stack.
3. **Accept conservative under-credit.** A filter-less single-entry consumer may remove a shared
   hash when the first sibling chunk is evicted even though another sibling chunk still contains
   it. That loses routing credit, but it does not make the engine serve incorrect data.

The current merge path uses option 1 for Dynamo.

## What is not in the current PRs

- The Dynamo PR does **not** rely on a `remove_blocks_impl` skip-absent-hashes patch. Earlier
  direct lower-tier replays used that as a diagnostic hardening experiment, but the production
  event path has `EventDedupFilter` in front of lower-tier indexing.
- The vLLM PR does **not** ship an exactly-once per-hash removal refcount in the producer. That
  variant was useful to prove the hazard, but the team decision is to keep the producer as plain
  fan-out and require consumers to dedup duplicated chunk announcements.
- Filter-less consumers such as a direct llm-d replay need their own refcount/dedup if they want
  exact CPU-tier credit under non-aligned shared-prefix workloads.

## Relation to Step 7

Step 7 verifies the actual shipped Dynamo path on a real local stack:

- vLLM PR #43468, opt-in self-describing chunk events
- Dynamo PR #10368, `medium="CPU"` routed to HostPinned plus lower-tier metrics wiring
- worker publisher `EventDedupFilter` in the path
- real CPU evictions from a small CPU pool
- wire capture reconciled exactly with `kv_cache_events_applied`

That Step 7 result is the merge-relevant evidence. Step 6 is the design-risk record explaining
why duplicate overlapping chunk announcements are expected and why the consumer must dedup them.
