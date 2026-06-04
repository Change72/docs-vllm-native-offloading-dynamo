#!/usr/bin/env python3
"""Capture vLLM's raw ZMQ KV-event frames to a JSONL file.

vLLM's ZmqEventPublisher sends multipart [topic, seq(8B), payload(msgpack)].
We connect a SUB, subscribe-all, and append one JSON line per message:
  {"topic": "<topic>", "seq": <int>, "payload_b64": "<base64 msgpack>"}

These are the EXACT bytes llm-d's VLLMAdapter parses; we replay them through
the real llm-d index in the Go harness.
"""
import base64
import json
import signal
import sys
import time

import zmq

endpoint = sys.argv[1] if len(sys.argv) > 1 else "tcp://localhost:5557"
outfile = sys.argv[2] if len(sys.argv) > 2 else "capture.jsonl"

ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect(endpoint)
sub.setsockopt(zmq.SUBSCRIBE, b"")  # all topics
sub.setsockopt(zmq.RCVTIMEO, 500)

stop = {"v": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))

n = 0
f = open(outfile, "w")
print(f"CAPTURE connected {endpoint} -> {outfile}", flush=True)
# Give the publisher a moment; run until SIGTERM.
while not stop["v"]:
    try:
        parts = sub.recv_multipart()
    except zmq.Again:
        continue
    except Exception as e:  # noqa: BLE001
        print(f"CAPTURE recv error: {e}", flush=True)
        break
    # [topic, seq, payload] (seq optional defensive)
    topic = parts[0].decode("utf-8", "replace") if parts else ""
    if len(parts) >= 3:
        seq = int.from_bytes(parts[1], "big")
        payload = parts[2]
    elif len(parts) == 2:
        seq, payload = 0, parts[1]
    else:
        continue
    f.write(json.dumps({"topic": topic, "seq": seq,
                        "payload_b64": base64.b64encode(payload).decode()}) + "\n")
    n += 1
    if n % 50 == 0:
        f.flush()
        print(f"CAPTURE {n} frames", flush=True)

f.flush()
f.close()
print(f"CAPTURE done: {n} frames -> {outfile}", flush=True)
