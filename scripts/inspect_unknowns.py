#!/usr/bin/env python3
"""Inspect 'unknown' task_type episodes to understand what they are."""
import json

eps = [json.loads(l) for l in open("logs/episodes.jsonl")]
unknowns = [e for e in eps if e.get("task_type") == "unknown"]

print(f"Total 'unknown' episodes: {len(unknowns)}\n")

for i, ep in enumerate(unknowns):
    rev = str(ep.get("revision", "") or ep.get("git_sha", "") or "")
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome", "?")
    calls = ep.get("api_calls", [])
    n_calls = len(calls)
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = ep.get("prompt", "")
    files = ep.get("files") or []
    
    # Get first 300 chars of prompt
    prompt_short = prompt[:400].replace("\n", " ")
    
    print(f"--- Episode {i+1} ---")
    print(f"  rev={rev} outcome={outcome} calls={n_calls} 4xx={n4xx}")
    if files:
        print(f"  files={files}")
    print(f"  prompt: {prompt_short}")
    
    # Show what API calls were made
    if calls:
        endpoints = [(c.get("method"), c.get("path", "")[:60], c.get("status")) for c in calls[:5]]
        for m, p, s in endpoints:
            print(f"    {m} {p} -> {s}")
        if len(calls) > 5:
            print(f"    ... +{len(calls)-5} more calls")
    print()
