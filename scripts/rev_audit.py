#!/usr/bin/env python3
"""Audit recent revision episodes: tasks, errors, check details."""
import json
import sys

eps = [json.loads(l) for l in open("logs/episodes.jsonl") if l.strip()]

# Show episodes for revisions >= 55
target_revs = [f"000{r}" for r in range(55, 70)]

print("=" * 80)
print("RECENT REVISION EPISODES AUDIT")
print("=" * 80)

for e in eps:
    rev = str(e.get("revision", ""))
    if not any(r in rev for r in target_revs):
        continue
    rev_short = rev.split("-")[-1] if "-" in rev else rev
    rev_num = ""
    for t in target_revs:
        if t in rev:
            rev_num = t
            break

    tt = e.get("task_type", "?")
    outcome = e.get("outcome", "?")
    errs = e.get("metrics", {}).get("http_4xx", 0)
    calls = e.get("metrics", {}).get("api_calls", 0)
    prompt = (e.get("prompt", ""))[:180]

    # Show 4xx details
    api_log = e.get("api_calls", [])
    bad = []
    for c in api_log:
        if c.get("status", 0) >= 400:
            method = c.get("method", "?")
            url = c.get("url", "?")[:60]
            status = c.get("status", 0)
            bad.append(f"{method} {url} -> {status}")

    print(f"\n[rev {rev_num}] {tt:28s} 4xx={errs} calls={calls:2d} outcome={outcome}")
    if bad:
        for b in bad[:5]:
            print(f"  ERROR: {b}")
    print(f"  {prompt}")

# Summary by task_type across ALL episodes
print("\n" + "=" * 80)
print("TASK TYPE SUMMARY (all episodes)")
print("=" * 80)

from collections import defaultdict
stats = defaultdict(lambda: {"total": 0, "errors": 0, "4xx": 0, "prompts": []})

for e in eps:
    tt = e.get("task_type", "unknown")
    stats[tt]["total"] += 1
    if e.get("outcome") == "error":
        stats[tt]["errors"] += 1
    errs = e.get("metrics", {}).get("http_4xx", 0)
    if errs > 0:
        stats[tt]["4xx"] += 1

for tt, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
    print(f"  {tt:30s}  total={s['total']:3d}  errors={s['errors']:2d}  with_4xx={s['4xx']:2d}")

# Show all UNIQUE task prompts that we get from the competition
# (helps identify the 30 tasks)
print("\n" + "=" * 80)
print("UNIQUE COMPETITION PROMPTS (recent revs 55+)")
print("=" * 80)

seen_prompts = set()
for e in eps:
    rev = str(e.get("revision", ""))
    if not any(r in rev for r in target_revs):
        continue
    prompt = e.get("prompt", "")[:100]
    tt = e.get("task_type", "?")
    if prompt not in seen_prompts:
        seen_prompts.add(prompt)
        print(f"  [{tt:28s}] {prompt}")
