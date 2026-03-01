#!/usr/bin/env python3
"""Read and display trace events from preflight.jsonl"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/rahul/.octane/traces/preflight.jsonl"
errors_only = "--errors" in sys.argv

with open(path) as f:
    lines = [l.strip() for l in f if l.strip()]

print(f"Total events: {len(lines)}")
for l in lines:
    e = json.loads(l)
    evt = e["event_type"]
    src = e.get("source", "")
    err = str(e.get("error", ""))[:120]
    pay = str(e.get("payload", {}))[:100]
    ts = str(e.get("ts", ""))[:19]

    if errors_only and not (err or evt in ("agent_error", "guard_block", "code_exhausted", "pipeline_error")):
        continue

    info = err or pay
    print(f"{ts} {evt:<28} {src:<22} {info}")
