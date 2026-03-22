#!/usr/bin/env python3
"""
Comprehensive gap analysis: us vs #1 team.
Analyze ALL episodes to understand what each task number likely represents,
and find patterns for zero-score and low-score tasks.
"""
import json
from collections import Counter, defaultdict

eps = [json.loads(l) for l in open("logs/episodes.jsonl")]

# Our scores vs #1 team
our_scores = {
    1: 1.25, 2: 0.63, 3: 2.00, 4: 2.00, 5: 0, 6: 1.40,
    7: 2.00, 8: 0.71, 9: 0, 10: 0, 11: 0, 12: 0,
    13: 1.13, 14: 3.00, 15: 0.50, 16: 1.00, 17: 3.50, 18: 4.00,
    19: 2.32, 20: 0.60, 21: 2.57, 22: 0, 23: 0.60, 24: 0,
    25: 4.80, 26: 1.05, 27: 0.60, 28: 0.60, 29: 0.55, 30: 1.80,
}
top_scores = {
    1: 2.00, 2: 2.00, 3: 2.00, 4: 2.00, 5: 2.00, 6: 1.67,
    7: 2.00, 8: 2.00, 9: 4.00, 10: 4.00, 11: 4.00, 12: 4.00,
    13: 2.40, 14: 4.00, 15: 3.33, 16: 3.00, 17: 3.50, 18: 4.00,
    19: 2.73, 20: 2.40, 21: 2.57, 22: 0, 23: 0.60, 24: 2.25,
    25: 6.00, 26: 6.00, 27: 6.00, 28: 1.50, 29: 2.73, 30: 1.80,
}

print("=" * 80)
print("SCOREBOARD GAP ANALYSIS: Us vs #1")
print("=" * 80)

gaps = []
for t in range(1, 31):
    ours = our_scores.get(t, 0)
    theirs = top_scores.get(t, 0)
    gap = theirs - ours
    gaps.append((t, ours, theirs, gap))

gaps.sort(key=lambda x: -x[3])
total_gap = sum(g[3] for g in gaps)
print(f"\nTotal gap: {total_gap:.2f} points")
print(f"\nTask  Ours   #1    Gap    Priority")
print("-" * 50)
for t, ours, theirs, gap in gaps:
    if gap <= 0:
        continue
    prio = "CRITICAL" if gap >= 3 else "HIGH" if gap >= 1.5 else "MEDIUM" if gap >= 0.5 else "LOW"
    print(f"  {t:2d}  {ours:5.2f}  {theirs:5.2f}  {gap:+5.2f}  {prio}")

# Analyze what task types we see, their scores, and patterns
print("\n" + "=" * 80)
print("TASK TYPE ANALYSIS")
print("=" * 80)

by_type = defaultdict(list)
for ep in eps:
    tt = ep.get("task_type", "unknown")
    by_type[tt].append(ep)

print(f"\n{'Task Type':30s} {'Count':>6s} {'Pass':>5s} {'Fail':>5s} {'4xx':>5s} {'Has PDF':>8s}")
print("-" * 75)
for tt in sorted(by_type.keys()):
    eps_list = by_type[tt]
    count = len(eps_list)
    passed = sum(1 for e in eps_list if e.get("outcome") == "completed" and
                 sum(1 for c in e.get("api_calls", []) if 400 <= c.get("status", 0) < 500) == 0)
    failed = count - passed
    total_4xx = sum(
        sum(1 for c in e.get("api_calls", []) if 400 <= c.get("status", 0) < 500)
        for e in eps_list
    )
    has_pdf = sum(1 for e in eps_list if any("pdf" in f.lower() for f in (e.get("files") or [])))
    print(f"  {tt:28s} {count:6d} {passed:5d} {failed:5d} {total_4xx:5d} {has_pdf:8d}")

# Focus on PDF tasks
print("\n" + "=" * 80)
print("PDF/FILE TASKS (tasks with attachments)")
print("=" * 80)
for ep in eps:
    files = ep.get("files") or []
    if not files:
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", "") or "")
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    tt = ep.get("task_type", "?")
    outcome = ep.get("outcome", "?")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = ep.get("prompt", "")[:150].replace("\n", " ")
    print(f"\n  rev={rev} task={tt} outcome={outcome} n4xx={n4xx}")
    print(f"  files={files}")
    print(f"  prompt={prompt}")

# Analyze which task types appear with what frequency in recent revisions
print("\n" + "=" * 80)
print("TASK TYPES THAT NEVER APPEAR IN OUR LOGS (potential zero-scores)")
print("=" * 80)

# List all unique task types we've seen
all_types = set(by_type.keys())
print(f"All seen task types ({len(all_types)}): {sorted(all_types)}")

# Check outcomes by task type for recent revisions
print("\n" + "=" * 80)
print("RECENT (rev 55+) PERFORMANCE BY TASK TYPE")
print("=" * 80)
recent = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "4xx": 0})
for ep in eps:
    rev = str(ep.get("revision", "") or ep.get("git_sha", "") or "")
    # Extract revision number
    import re
    m = re.search(r'(\d{5})', rev)
    if not m:
        continue
    rev_num = int(m.group(1))
    if rev_num < 55:
        continue
    tt = ep.get("task_type", "unknown")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    outcome = ep.get("outcome", "?")
    if outcome == "completed" and n4xx == 0:
        recent[tt]["pass"] += 1
    elif outcome == "error":
        recent[tt]["error"] += 1
    else:
        recent[tt]["fail"] += 1
    recent[tt]["4xx"] += n4xx

for tt in sorted(recent.keys()):
    r = recent[tt]
    total = r["pass"] + r["fail"] + r["error"]
    rate = r["pass"] / total * 100 if total else 0
    print(f"  {tt:28s} pass={r['pass']:2d} fail={r['fail']:2d} err={r['error']:2d} 4xx={r['4xx']:2d} rate={rate:.0f}%")
