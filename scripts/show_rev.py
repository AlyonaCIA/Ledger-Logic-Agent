#!/usr/bin/env python3
"""Show detailed episodes for a given revision."""
import json, sys

rev_filter = sys.argv[1] if len(sys.argv) > 1 else "00062"
eps = [json.loads(l) for l in open("logs/episodes.jsonl")]

count = 0
for ep in eps:
    rev = str(ep.get("revision", "") or ep.get("git_sha", "") or "")
    if rev_filter not in rev:
        continue
    count += 1
    tt = ep.get("task_type", "?")
    outcome = ep.get("outcome", "?")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    ncalls = len(calls)
    ts = ep.get("timestamp", "")[:19]
    err = (ep.get("error", "") or "")[:150]
    prompt = ep.get("prompt", "")[:200].replace("\n", " ")
    files = ep.get("files", [])
    entities = ep.get("parsed_entities", {})

    print("=" * 70)
    print(f"[{ts}] task={tt} outcome={outcome} calls={ncalls} 4xx={n4xx}")
    print(f"  FILES: {files}")
    print(f"  PROMPT: {prompt}")
    if err:
        print(f"  ERROR: {err}")

    # Show key parsed entities
    for k in sorted(entities.keys()):
        if k.startswith("_"):
            continue
        v = entities[k]
        if v is not None and v != {} and v != [] and v != "":
            vs = json.dumps(v, ensure_ascii=False)
            if len(vs) > 200:
                vs = vs[:200] + "..."
            print(f"  PARSED.{k} = {vs}")

    # Show API calls
    for c in calls:
        st = c.get("status", 0)
        marker = " ***4XX***" if 400 <= st < 500 else ""
        method = c.get("method", "")
        path = c.get("path", "")
        cid = c.get("created_id", "")
        cid_str = f" id={cid}" if cid else ""
        print(f"    {method} {path} -> {st}{marker}{cid_str}")
    print()

print(f"Total {count} episodes for rev {rev_filter}")
