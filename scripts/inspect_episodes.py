#!/usr/bin/env python3
"""Inspect recent episodes in detail — prompts, parsed data, API call bodies."""
import json, sys

eps = [json.loads(l) for l in open("logs/episodes.jsonl")]

# Focus on recent episodes
for ep in eps[-30:]:
    rev = str(ep.get("revision", "") or ep.get("git_sha", "") or "?")
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]

    tt = ep.get("task_type", "?")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    outcome = ep.get("outcome", "?")
    prompt = ep.get("prompt", "")
    entities = ep.get("parsed_entities", {})
    error = ep.get("error", "")

    print(f"\n{'='*70}")
    print(f"rev={rev} task={tt} outcome={outcome} n4xx={n4xx}")
    print(f"PROMPT: {prompt[:400]}")
    if error:
        print(f"ERROR: {error[:200]}")
    
    # Show parsed entities (skip internal fields)
    for k in sorted(entities.keys()):
        if k.startswith("_"):
            continue
        v = entities[k]
        if v is not None and v != {} and v != [] and v != "":
            vs = json.dumps(v, ensure_ascii=False)
            if len(vs) > 200:
                vs = vs[:200] + "..."
            print(f"  PARSED.{k} = {vs}")
    
    # Show API calls with body
    for c in calls:
        body = c.get("body_sent")
        body_str = ""
        if body:
            body_str = " body=" + json.dumps(body, ensure_ascii=False)[:200]
        cid = c.get("created_id", "")
        cid_str = f" id={cid}" if cid else ""
        st = c.get("status", 0)
        marker = " ***4XX***" if 400 <= st < 500 else ""
        method = c.get("method", "")
        path = c.get("path", "")
        resp = c.get("response_snippet", "")
        resp_str = ""
        if resp and isinstance(resp, str):
            resp_str = f" resp={resp[:100]}"
        elif resp and isinstance(resp, dict):
            resp_str = f" resp={json.dumps(resp, ensure_ascii=False)[:100]}"
        print(f"  API: {method} {path} -> {st}{marker}{cid_str}{body_str}{resp_str}")
