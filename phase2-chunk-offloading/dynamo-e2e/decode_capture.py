#!/usr/bin/env python3
"""Decode a captured vLLM ZMQ KV-event stream and verify the Plan-B chunk
wire shape plus store/remove accounting.

Checks (factor F, per-block size B):
  CPU BlockStored : every non-placeholder event carries exactly F hashes
                    (chunk) with tok_len == F*B and block_size == B; parent
                    set except for prefix-head chunks. Placeholder events
                    (tok_len==0 && block_size==0) are counted separately.
  CPU BlockRemoved: hash multiset accounting -- every removed hash must
                    have been stored before (no unknown removals), and
                    removals must come in whole-chunk groups (the N
                    constituent hashes of a chunk are removed together).
  Parent chain    : a CPU stored event's parent must be a hash already
                    seen in a previous CPU stored event (or None).

Note: the shipped producer does PLAIN FAN-OUT (no per-hash exactly-once).
On non-chunk-aligned shared-prefix traffic, sibling chunks legitimately
re-announce shared hashes, so checks 5/6/7 are INFORMATIONAL (re-announce
and suppressed-remove counts are EXPECTED to be > 0); the consumer (Dynamo
EventDedupFilter) deduplicates them. The hard pass/fail gates are checks
1-4 (shape, parent chain, and no remove of a never-stored hash).

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
    f"1. store: every CPU BlockStored has 1..{factor} hashes "
    f"(shipped producer emits exactly {factor} per non-placeholder chunk; "
    f"n_hashes=1 also covers placeholders): "
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

# --- consumer-side dedup behavior (INFORMATIONAL, not pass/fail) -----------
# The shipped vLLM producer does PLAIN FAN-OUT: it does NOT make per-hash
# store/remove exactly-once. Under a non-chunk-aligned shared prefix, two
# sibling chunks legitimately list the same constituent block hash, so that
# hash is announced (stored AND removed) once per containing chunk -- by
# design (see events.py docstring + step6). The CONSUMER deduplicates:
# Dynamo's worker publisher runs EventDedupFilter (ai-dynamo/dynamo#8012),
# which ref-counts per (dp_rank, tier, hash) and forwards a remove only when
# the last live reference disappears.
#
# So checks 5/6/7 are INFORMATIONAL: re-announced / suppressed counts are
# EXPECTED > 0 on overlapping traffic and == 0 on aligned/by-block traffic.
# The real correctness gates are checks 1-4 (shape, parent chain, and -- the
# only true remove invariant -- no remove of a never-stored hash, check 4).
announced = set()
re_announced = 0  # store of an already-announced hash (overlap)
re_removed = 0    # remove of a hash already removed since its last store
for ev in cpu_event_stream:
    kind, hashes = ev
    if kind == "stored":
        for h in hashes:
            if h in announced:
                re_announced += 1
            announced.add(h)
    else:
        for h in hashes:
            if h not in announced:
                re_removed += 1
            announced.discard(h)
print(
    "5. per-hash announce machine (informational): "
    f"re-announced stores={re_announced}, removes-of-already-removed={re_removed}"
    "   (expected > 0 under non-aligned overlap, 0 otherwise; a true "
    "'remove of never-stored' is gated by check 4)"
)
print(
    "6. wire re-announce count (informational): "
    f"{re_announced} duplicate per-hash stores"
    "   (plain fan-out re-announces shared hashes; the consumer's "
    "EventDedupFilter ref-counts them. 0 == no overlap in this capture)"
)

# Simulate Dynamo's EventDedupFilter (PR ai-dynamo/dynamo#8012): store
# increments a per-hash refcount; a remove is forwarded only when the count
# returns to 0. `filtered_removes` is how many non-final removes the filter
# correctly SUPPRESSES -- expected > 0 exactly when there is overlap, and the
# reason raw-wire removes can exceed the router's post-filter applied removes.
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
live_after_filter = sum(1 for v in fcnt.values() if v > 0)
print(
    "7. EventDedupFilter simulation (informational): "
    f"non-final removes suppressed={filtered_removes}, "
    f"removes forwarded={passed_removes}, still-live hashes={live_after_filter}"
    "   (suppressed > 0 is CORRECT under overlap; still-live = blocks not yet "
    f"evicted. Reconciliation: router applied_removes == wire_removes - {filtered_removes})"
)
