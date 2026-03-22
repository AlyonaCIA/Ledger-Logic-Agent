#!/usr/bin/env python3
"""Check specific episodes in detail - with full entities."""
import json, sys

eps = [json.loads(l) for l in open("logs/episodes.jsonl") if l.strip()]

needle = sys.argv[1] if len(sys.argv) > 1 else "commande"
for ep in eps:
    p = ep.get("prompt", "")
    if needle not in p:
        continue
    rev = (ep.get("revision", "") or ep.get("git_sha", ""))[-8:]
    task = ep.get("task_type", "?")
    calls = len(ep.get("api_calls", []))
    out = ep.get("outcome", "?")
    print(f"[{rev}] {task} calls={calls} out={out}")
    print(f"  PROMPT: {p[:300]}")
    entities = ep.get("parsed_entities", {})
    for k, v in entities.items():
        if k == "_raw_prompt":
            continue
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    for c in ep.get("api_calls", []):
        print(f"  API: {c.get('method','?')} {c.get('path','?')} -> {c.get('status',0)}")
    print()
