# PR body drafts for Stage 2 chunk offloading

## vLLM PR #43468

Title:

```text
[feature][kv_offload] Self-describing KV events for OffloadingConnector
```

Body:

````markdown
## What

This PR makes native `OffloadingConnector` KV-cache events self-describing behind an explicit
opt-in:

```python
kv_connector_extra_config={
    "self_describing_kv_events": True,
}
```

The flag is inert unless vLLM KV cache events are also enabled. With the flag off, the connector
keeps the existing placeholder event behavior.

For CPU offload stores, the connector now records the request metadata while it still has access
to the request and KV-cache-group context, then emits `BlockStored` events with:

- `block_hashes`
- `parent_block_hash`
- `token_ids`
- per-block `block_size`
- LoRA metadata
- `group_idx` and cache-spec metadata

For chunked offloading (`kv_connector_extra_config["block_size"] > --block-size`), one offloaded
CPU chunk is emitted as one `BlockStored` carrying all constituent per-block hashes plus the whole
chunk token span. `block_size` remains the GPU/hash block size, not the whole chunk size. Removal
events fan out the same constituent hashes.

## Why

The legacy connector events were useful for observability but not sufficient for external routers
or lower-tier indexers. They often carried placeholder payloads (`token_ids=[]`, `block_size=0`,
`parent_block_hash=None`), so consumers such as Dynamo could not reconstruct their own block keys
or parent chain and had to drop the CPU-tier events.

Making the producer self-describing keeps the event contract local to the connector: a consumer
does not need to join CPU events with a separate GPU event stream or depend on event ordering.

## Chunk overlap semantics

The producer intentionally uses plain fan-out. In chunk mode, if a shared prefix is not aligned to
the offloaded chunk size, two sibling chunks can legitimately list the same constituent per-block
hash. In that case duplicate store/remove announcements are expected on the wire.

Consumers that index at per-block granularity must ref-count or deduplicate those duplicate
announcements. Dynamo's standard worker publisher path already does this with `EventDedupFilter`.
Filter-less consumers may conservatively under-credit CPU-tier cache after overlapping chunk
evictions, but this does not create data corruption.

## Scope

- Native `OffloadingConnector`.
- Full-attention groups.
- By-block mode and chunk mode.
- Sliding-window groups keep the legacy fallback.
- `extra_keys`-heavy workloads such as multimodal/cache-salt/prompt-embedding paths should be
  validated separately before relying on the new payload for routing.

## Tests

- Unit coverage in `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` covers:
  - opt-in vs opt-out behavior
  - self-describing store payloads
  - chunked store payloads with multiple constituent block hashes
  - removal fan-out
  - event ordering independence
  - reset behavior

Recommended local command:

```bash
python -m pytest tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -q
```

## End-to-end validation

Validated with Dynamo PR #10368 on a real single-GPU L4 stack:

- model: `Qwen/Qwen3-0.6B`
- vLLM block size: 16
- offloaded chunk size: 48 (`factor=3`)
- CPU pool: 128 MiB, intentionally small to force real CPU LRU evictions
- explicit ZMQ KV events enabled
- `self_describing_kv_events=true`
- Dynamo worker publisher `EventDedupFilter` in the path

Result:

- CPU `BlockStored`: 354 events, zero placeholders, every CPU chunk store has `n_hashes=3`
- CPU `BlockRemoved`: 24 events from real CPU evictions
- GPU `BlockStored`: 331 events
- router metric `kv_cache_events_applied`: stored ok = 685, removed ok = 24
- zero lower-tier warnings / `BlockNotFound`
````

## Dynamo PR #10368

Suggested title:

```text
[kv-router] Route vLLM CPU KV events to HostPinned and count lower-tier applies
```

Body:

````markdown
## What

This PR makes Dynamo's KV router consume and observe vLLM native CPU-offload KV events.

Changes:

- Treat vLLM's `medium="CPU"` as an alias for the existing `HostPinned` tier.
- Preserve existing `CPU_PINNED` and `CPU_TIER1` behavior.
- Add wire tests for placeholder CPU payloads and full self-describing CPU payloads.
- Count lower-tier `Stored` / `Removed` / `Cleared` applies in `kv_cache_events_applied`.
- Wire the metrics handle into lazily-created lower-tier indexers used by the `dynamo-llm` router
  assembly path.

## Why

vLLM's native `OffloadingConnector` publishes CPU-tier KV events with `medium="CPU"`. Dynamo
previously did not classify that string as `HostPinned`, so the event could not be routed to the
lower-tier indexer as intended.

After adding the alias, the lower-tier path also needed observability. The first metrics patch made
`LowerTierIndexer::worker` increment the same `kv_cache_events_applied` counter as the primary
device-tier indexer, but the `dynamo-llm` router assembly path still constructed lower-tier
indexers without a metrics handle. The final wiring patch passes the shared metrics handle through
`LocalKvIndexer` and `LowerTierIndexers::new_with_metrics`, including lazily-created
HostPinned/Disk/External indexers.

This is why there are two metrics commits:

1. Apply-path support: count lower-tier events when a lower-tier indexer has metrics.
2. Production wiring: make sure production lower-tier indexers actually receive metrics.

## Relationship to vLLM #43468

This PR is the Dynamo side of the self-describing CPU-event path. It expects CPU `BlockStored`
events to carry enough payload for Dynamo to reconstruct local block hashes. The vLLM-side PR
provides that payload for native `OffloadingConnector`, including chunk mode.

For chunked offload, vLLM intentionally emits plain fan-out. Overlapping chunks may repeat the same
per-block hash on the wire. Dynamo's standard worker publisher path already runs
`EventDedupFilter`, which ref-counts duplicate per-worker/tier hash announcements before they reach
lower-tier indexing.

This PR does not rely on a `remove_blocks_impl` skip-absent-hashes change.

## Tests

Recommended local commands:

```bash
cargo test -p dynamo-kv-router --lib
cargo test -p dynamo-llm --lib kv_router
```

Focused coverage:

- `cpu_event_with_placeholder_payload_is_dropped_safely`
- `cpu_event_with_full_payload_is_indexable`

## End-to-end validation

Validated with vLLM PR #43468 on a real single-GPU L4 stack:

- model: `Qwen/Qwen3-0.6B`
- vLLM block size: 16
- offloaded chunk size: 48 (`factor=3`)
- CPU pool: 128 MiB, intentionally small to force real CPU LRU evictions
- explicit ZMQ KV events enabled
- `self_describing_kv_events=true`
- worker publisher `EventDedupFilter` in the path

Wire capture and router metrics reconciled exactly:

- wire stored: 685 total = 331 GPU + 354 CPU
- wire removed: 24 CPU removes
- router `kv_cache_events_applied`: stored ok = 685, removed ok = 24
- CPU placeholders: 0
- lower-tier warnings / `BlockNotFound`: 0
````
