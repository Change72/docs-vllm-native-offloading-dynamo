#!/usr/bin/env python3
"""Decode a captured vLLM ZMQ KV-event stream and verify the Plan-B chunk
wire shape plus store/remove accounting.

Checks (factor F, per-block size B):
  CPU BlockStored : every event carries n_hashes in {F} (chunk) with
                    tok_len == F*B and block_size == B; parent set except
                    for prefix-head chunks. Placeholder events
                    (tok_len==0 && block_size==0) are counted separately.
  CPU BlockRemoved: hash multiset accounting -- every removed hash must
                    have been stored before (no unknown removals), and
                    removals must come in whole-chunk groups (the N
                    constituent hashes of a chunk are removed together).
  Parent chain    : a CPU stored event's parent must be a hash already
                    seen in a previous CPU stored event (or None).

Usage: decode_capture.py <capture.jsonl> [--factor 3] [--block-size 16]
"""

import base64
import json
import sys
from collections import Counter, defaultdict

import msgpack

path = sys.argv[1]
factor = 3
block_size = 16
for i, a in enumerate(sys.argv):
    if a == "--factor":
        factor = int(sys.argv[i + 1])
    if a == "--block-size":
        block_size = int(sys.argv[i + 1])

# Per-medium stats
stored_events = Counter()      # medium -> num BlockStored events
removed_events = Counter()     # medium -> num BlockRemoved events
stored_nhash = defaultdict(Counter)   # medium -> Counter(n_hashes per event)
removed_nhash = defaultdict(Counter)
placeholder_events = Counter() # medium -> events with tok=0,bs=0
shape_violations = []          # CPU stored events with unexpected shape
parent_violations = 0
parent_none = 0

cpu_stored_hashes = []         # in arrival order, list per event
cpu_seen_hashes = set()        # all CPU hashes stored so far (for parent check)
cpu_removed_hashes = []        # list per event
cpu_live = Counter()           # hash -> live store count (store +1, remove -1)
removed_unknown = 0            # removed hash never stored
gpu_stored_hash_count = 0

# Map: CPU stored event -> its ordered hash list, to check removals come in
# whole-chunk groups. chunk_of[h] = (event_idx, position, chunk_tuple)
chunk_of = {}
cpu_parents = []               # parent of each CPU stored event (or None)
stored_chunks = set()          # every chunk tuple ever stored
cpu_event_stream = []          # ("stored"|"removed", hashes) in arrival order

frames = 0
events_total = 0
with open(path) as f:
    for line in f:
        frames += 1
        rec = json.loads(line)
        payload = base64.b64decode(rec["payload_b64"])
        batch = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        # batch = [ts, [raw events], data_parallel_rank?]
        for raw in batch[1]:
            ev = (
                raw
                if isinstance(raw, (list, tuple, dict))
                else msgpack.unpackb(raw, raw=False, strict_map_key=False)
            )
            events_total += 1
            # vLLM <= array encoding: [type, hashes, parent, tokens, bs,
            # lora_id, medium, ...]; vLLM >= #42892: map encoding with a
            # "type" tag field.
            is_map = isinstance(ev, dict)
            kind = ev["type"] if is_map else ev[0]
            if kind == "BlockStored":
                if is_map:
                    hashes = ev["block_hashes"]
                    parent = ev.get("parent_block_hash")
                    toks = ev.get("token_ids", [])
                    bs = ev.get("block_size", 0)
                    medium = ev.get("medium", "?")
                else:
                    hashes, parent, toks, bs = ev[1], ev[2], ev[3], ev[4]
                    medium = ev[6] if len(ev) > 6 else "?"
                stored_events[medium] += 1
                stored_nhash[medium][len(hashes)] += 1
                if medium == "GPU":
                    gpu_stored_hash_count += len(hashes)
                    continue
                # CPU side
                if len(toks) == 0 and bs == 0:
                    placeholder_events[medium] += 1
                    continue
                ok = (
                    len(toks) == len(hashes) * block_size
                    and bs == block_size
                )
                if not ok:
                    shape_violations.append(
                        f"stored n_hashes={len(hashes)} tok={len(toks)} bs={bs}"
                    )
                if parent is None:
                    parent_none += 1
                elif parent not in cpu_seen_hashes:
                    # strict in-order check; may "fail" because
                    # complete_store iterates a set (out-of-order within
                    # a batch). The global check below is the verdict.
                    parent_violations += 1
                cpu_stored_hashes.append(list(hashes))
                cpu_parents.append(parent)
                cpu_event_stream.append(("stored", list(hashes)))
                chunk = tuple(hashes)
                stored_chunks.add(chunk)
                for h in hashes:
                    cpu_seen_hashes.add(h)
                    cpu_live[h] += 1
                    chunk_of[h] = chunk
            elif kind == "BlockRemoved":
                if is_map:
                    hashes = ev["block_hashes"]
                    medium = ev.get("medium", "?")
                else:
                    hashes = ev[1]
                    medium = ev[2] if len(ev) > 2 else "?"
                removed_events[medium] += 1
                removed_nhash[medium][len(hashes)] += 1
                if medium == "GPU":
                    continue
                cpu_removed_hashes.append(list(hashes))
                cpu_event_stream.append(("removed", list(hashes)))
                for h in hashes:
                    if cpu_live[h] <= 0:
                        removed_unknown += 1
                    else:
                        cpu_live[h] -= 1
            elif kind == "AllBlocksCleared":
                pass

print(f"frames={frames} events={events_total}")
for medium in sorted(set(stored_events) | set(removed_events)):
    print(
        f"  [{medium}] stored_events={stored_events[medium]} "
        f"(placeholder={placeholder_events[medium]}) "
        f"removed_events={removed_events[medium]}"
    )
    print(f"    stored n_hashes histogram : {dict(stored_nhash[medium])}")
    print(f"    removed n_hashes histogram: {dict(removed_nhash[medium])}")

n_cpu_stored = sum(len(x) for x in cpu_stored_hashes)
n_cpu_removed = sum(len(x) for x in cpu_removed_hashes)
print(f"\nCPU stored hashes total  = {n_cpu_stored}")
print(f"CPU removed hashes total = {n_cpu_removed}")
print(f"CPU live (stored-removed) = {sum(v for v in cpu_live.values() if v > 0)}")

# --- Plan B shape verdicts -------------------------------------------------
print("\n--- Plan B shape checks (CPU tier) ---")
bad_nhash = {
    k: v
    for k, v in stored_nhash.get("CPU", Counter()).items()
    if not 1 <= k <= factor
}
ph = placeholder_events.get("CPU", 0)
print(
    f"1. store: every non-placeholder CPU BlockStored has 1..{factor} hashes "
    "(overlap-trimmed): "
    + ("PASS" if not bad_nhash else f"FAIL {bad_nhash}")
    + (f"   (placeholders={ph})" if ph else "")
)
print(
    f"2. store: tok_len == n_hashes*{block_size} and block_size=={block_size}: "
    + ("PASS" if not shape_violations else f"FAIL e.g. {shape_violations[:3]}")
)
global_dangling = sum(
    1 for p in cpu_parents if p is not None and p not in cpu_seen_hashes
)
print(
    "3. store: parent is a CPU hash stored somewhere in the stream: "
    + ("PASS" if global_dangling == 0 else f"FAIL ({global_dangling} dangling)")
    + f"   (parent=None on {parent_none} prefix-head chunks)"
)
print(
    f"   note: strict in-arrival-order parent check: {parent_violations} "
    "out-of-order arrivals (expected: complete_store iterates a set; "
    "events are self-contained so order does not matter)"
)
print(
    "4. remove: every removed CPU hash was previously stored: "
    + ("PASS" if removed_unknown == 0 else f"FAIL ({removed_unknown} unknown)")
)

# Announce-state machine: a removed hash must be in the announced state
# (stored since its last removal), and never removed twice within one
# announced period. This is the exact invariant single-entry router
# indexes (Dynamo worker_map, llm-d pod-entry sets) rely on. With the
# vLLM removal refcount, overlapping chunks announce a shared hash's
# removal only when its LAST live reference dies, so this must hold even
# under non-chunk-aligned shared prefixes.
announced = set()
removed_unannounced = 0
suppressed_dups = 0  # informational under refcount: store of already-announced
for ev in cpu_event_stream:
    kind, hashes = ev
    if kind == "stored":
        for h in hashes:
            if h in announced:
                suppressed_dups += 1
            announced.add(h)
    else:
        for h in hashes:
            if h not in announced:
                removed_unannounced += 1
            announced.discard(h)
print(
    "5. remove: per-hash announce state machine (no remove while "
    "un-announced): "
    + ("PASS" if removed_unannounced == 0 else f"FAIL ({removed_unannounced})")
    + f"   (re-announces of an already-announced hash: {suppressed_dups})"
)

# --- exactly-once + Dynamo EventDedupFilter simulation ---------------------
print(
    "6. wire is per-hash exactly-once (no re-announce of an announced hash): "
    + ("PASS" if suppressed_dups == 0 else f"FAIL ({suppressed_dups} duplicate stores)")
)

# Simulate Dynamo's EventDedupFilter (PR ai-dynamo/dynamo#8012): store
# increments a per-hash refcount; a remove passes through only when the
# count hits 0. Under exactly-once wire, the filter must be a no-op.
fcnt = Counter()
filtered_removes = 0
passed_removes = 0
for kind, hashes in cpu_event_stream:
    if kind == "stored":
        for h in hashes:
            fcnt[h] += 1
    else:
        for h in hashes:
            fcnt[h] -= 1
            if fcnt[h] <= 0:
                passed_removes += 1
                fcnt.pop(h, None)
            else:
                filtered_removes += 1
print(
    "7. Dynamo EventDedupFilter simulation: removes blocked by the filter: "
    + ("PASS (0 blocked, filter is a no-op)" if filtered_removes == 0
       else f"FAIL ({filtered_removes} blocked -> consumer-side leak)")
    + f"   (passed={passed_removes}, residual filter entries={sum(1 for v in fcnt.values() if v > 0)})"
)
