#!/usr/bin/env python3
"""Map all unique prompts to their latest task classification and outcome."""
import json

eps = []
with open("logs/episodes.jsonl") as f:
    for line in f:
        eps.append(json.loads(line.strip()))

unique = {}
for ep in eps:
    p = ep.get("prompt", "")[:80].strip()
    key = p
    if key not in unique:
        unique[key] = {
            "prompt_full": ep.get("prompt", "")[:300],
            "revisions": set(),
            "latest_task": "",
            "latest_outcome": "",
            "latest_4xx": 0,
            "latest_rev": "",
            "latest_calls": 0,
            "latest_error": "",
        }
    rev = ep.get("revision", "") or ep.get("git_sha", "")
    unique[key]["revisions"].add(rev)
    unique[key]["latest_task"] = ep.get("task_type", "?")
    unique[key]["latest_outcome"] = ep.get("outcome", "?")
    unique[key]["latest_4xx"] = len(
        [c for c in ep.get("api_calls", []) if c.get("status", 200) >= 400]
    )
    unique[key]["latest_rev"] = rev[-8:] if rev else "?"
    unique[key]["latest_calls"] = len(ep.get("api_calls", []))
    unique[key]["latest_error"] = (ep.get("error") or "")[:100]

items = sorted(unique.values(), key=lambda x: x["latest_task"])
print(f"{'TASK TYPE':30s} {'OUT':10s} 4xx  CALLS  REVS  PROMPT")
print("-" * 120)
for u in items:
    revs = len(u["revisions"])
    prompt_short = u["prompt_full"][:80]
    print(
        f"{u['latest_task']:30s} {u['latest_outcome']:10s} "
        f"{u['latest_4xx']:3d}  {u['latest_calls']:5d}  {revs:4d}  {prompt_short}"
    )
    if u["latest_error"]:
        print(f"  ERROR: {u['latest_error']}")
