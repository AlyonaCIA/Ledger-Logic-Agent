#!/usr/bin/env python3
"""Deep-dive into specific episodes: show full API call log."""
import json
import sys

eps = [json.loads(l) for l in open("logs/episodes.jsonl") if l.strip()]

# Filter by revision
rev_filter = sys.argv[1] if len(sys.argv) > 1 else "00063"
task_filter = sys.argv[2] if len(sys.argv) > 2 else None

for e in eps:
    rev = str(e.get("revision", ""))
    if rev_filter not in rev:
        continue
    tt = e.get("task_type", "?")
    if task_filter and task_filter not in tt:
        continue

    print("=" * 100)
    print(f"REVISION: {rev}")
    print(f"TASK TYPE: {tt}")
    print(f"OUTCOME: {e.get('outcome', '?')}")
    print(f"METRICS: {e.get('metrics', {})}")
    print(f"PROMPT: {e.get('prompt', '')[:500]}")
    print()

    # Show parsed entities
    entities = e.get("parsed_entities", {})
    if entities:
        for k, v in entities.items():
            if k == "_raw_prompt":
                continue
            print(f"  ENTITY.{k}: {json.dumps(v, ensure_ascii=False)[:200]}")
        print()

    # Show all API calls
    api_calls = e.get("api_calls", [])
    print(f"API CALLS ({len(api_calls)}):")
    for i, c in enumerate(api_calls, 1):
        method = c.get("method", "?")
        url = c.get("path") or c.get("url", "?")
        status = c.get("status", 0)
        error = c.get("error", "")
        marker = " *** 4xx ***" if 400 <= status < 500 else ""
        print(f"  [{i:2d}] {method} {url} -> {status}{marker}")
        if error:
            print(f"       ERROR: {error[:300]}")
    print()
