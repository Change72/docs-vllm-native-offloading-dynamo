# Development-machine checklist before image build

Run these on the development machine before building a Docker image. The goal is to prove the code
is correct locally, not to run the cluster sweep yet.

## 0. Confirm the branches

```bash
cd /home/changg/workspace/vllm
git rev-parse --short HEAD
git log -1 --oneline

cd /home/changg/workspace/dynamo
git rev-parse --short HEAD
git log -1 --oneline
```

Expected heads for the current review:

- vLLM: `66273a27d` or the latest revision of PR #43468
- Dynamo: `db0ec356` or the latest revision of PR #10368

## 1. Run the vLLM connector unit tests

```bash
cd /home/changg/workspace/vllm
python -m pytest tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -q
```

Pass criteria:

- all tests pass
- especially the opt-in, chunk store payload, removal fan-out, and reset-cache cases

## 2. Run Dynamo kv-router tests

```bash
cd /home/changg/workspace/dynamo
cargo test -p dynamo-kv-router --lib
```

Pass criteria:

- all kv-router lib tests pass
- the CPU medium alias tests pass:
  - `cpu_event_with_placeholder_payload_is_dropped_safely`
  - `cpu_event_with_full_payload_is_indexable`

Optional focused rerun if you want quick signal while debugging:

```bash
cargo test -p dynamo-kv-router cpu_event_with_placeholder_payload_is_dropped_safely
cargo test -p dynamo-kv-router cpu_event_with_full_payload_is_indexable
```

## 3. Run Dynamo LLM-side kv-router tests

```bash
cd /home/changg/workspace/dynamo
cargo test -p dynamo-llm --lib kv_router
```

Pass criteria:

- all `kv_router` tests in `dynamo-llm` pass
- no failures from the production assembly path that wires `LowerTierIndexers::new_with_metrics`

## 4. Reconcile the existing Step 7 smoke artifacts

If the Step 7 L4 run has already completed, first re-check the saved artifacts rather than rerunning
the whole stack:

```bash
cd /home/changg/workspace/docs-vllm-native-offloading-dynamo/phase2-chunk-offloading
python dynamo-e2e/decode_capture.py /home/changg/workspace/.tmp/dyn_e2e/sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied /home/changg/workspace/.tmp/dyn_e2e/frontend_metrics_*.txt | grep -v ' 0$'
grep -iE 'BlockNotFound|Failed to apply lower-tier|Failed to apply event to local indexer' /home/changg/workspace/.tmp/dyn_e2e/worker.log
```

Pass criteria:

- CPU stored events have `placeholder=0`
- CPU stored `n_hashes` is `{3: 354}` for the recorded run
- CPU removed events are present
- `kv_cache_events_applied` shows stored ok = 685 and removed ok = 24 for the recorded run
- the final grep finds no relevant lower-tier error lines

## 5. Rerun the L4 smoke only if code changed after the recorded run

```bash
cd /home/changg/workspace/docs-vllm-native-offloading-dynamo/phase2-chunk-offloading
bash dynamo-e2e/run_e2e_capture.sh
python dynamo-e2e/decode_capture.py sidecap.jsonl --factor 3 --block-size 16
grep kv_cache_events_applied frontend_metrics*.txt | grep -v ' 0$'
```

Pass criteria are the same as Step 4. Exact counts can move slightly with traffic timing, but the
invariants should not:

- zero CPU placeholders
- CPU chunk stores have `n_hashes == 3`
- CPU removes exist
- router applied counters reconcile with wire capture
- zero lower-tier warnings / `BlockNotFound`

## Stop point

Stop here and review the results before building the Docker image. The cluster sweep should wait
until the unit tests and L4 smoke are clean.
