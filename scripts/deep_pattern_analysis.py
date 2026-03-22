#!/usr/bin/env python3
"""Deep pattern analysis of ALL episodes, focusing on rev 0059 and recent submissions."""
import json
from collections import defaultdict

episodes = [json.loads(l) for l in open("logs/episodes.jsonl")]

# Focus on recent revisions
recent_revs = set()
for e in episodes:
    rev = e.get("revision") or e.get("git_sha") or ""
    if "0058" in rev or "0059" in rev:
        recent_revs.add(rev)

print(f"Total episodes: {len(episodes)}")
print(f"Recent revisions found: {recent_revs}")
print()

# ── Per-revision per-task success analysis ──
rev_task_results = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0, "errors": [], "prompts": []}))

for e in episodes:
    rev = e.get("revision") or e.get("git_sha") or ""
    tt = e.get("task_type") or "unknown"
    calls = e.get("api_calls") or []
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    n5xx = sum(1 for c in calls if 500 <= c.get("status", 0) < 600)
    
    # Determine pass/fail: competition checks for 0 4xx errors
    passed = n4xx == 0 and n5xx == 0
    
    bucket = rev_task_results[rev][tt]
    bucket["total"] += 1
    if passed:
        bucket["pass"] += 1
    else:
        bucket["fail"] += 1
        failed_calls = [f"{c['method']} {c['path']} -> {c['status']}" for c in calls if c.get("status", 0) >= 400]
        bucket["errors"].append(failed_calls)
        bucket["prompts"].append(e.get("prompt", "")[:150])

# Show latest rev results first
print("=" * 80)
print("RESULTS BY REVISION (latest first)")
print("=" * 80)
all_revs = sorted(rev_task_results.keys(), reverse=True)
for rev in all_revs[:5]:  # last 5 revisions
    tasks = rev_task_results[rev]
    total_pass = sum(t["pass"] for t in tasks.values())
    total_fail = sum(t["fail"] for t in tasks.values())
    total = sum(t["total"] for t in tasks.values())
    print(f"\n{'─'*60}")
    print(f"REV: {rev}  ({total_pass}/{total} passed, {total_fail} failed)")
    print(f"{'─'*60}")
    for tt in sorted(tasks.keys()):
        t = tasks[tt]
        status = "✅" if t["fail"] == 0 else "❌"
        print(f"  {status} {tt}: {t['pass']}/{t['total']}")
        if t["fail"] > 0:
            for i, (errs, prompt) in enumerate(zip(t["errors"], t["prompts"])):
                print(f"     FAIL #{i+1}: {prompt[:100]}")
                for err in errs[:3]:
                    print(f"       → {err}")

# ── Overall pattern: which tasks NEVER pass? ──
print("\n" + "=" * 80)
print("TASKS THAT HAVE NEVER PASSED (across ALL revisions)")
print("=" * 80)
all_task_stats = defaultdict(lambda: {"pass": 0, "fail": 0})
for rev, tasks in rev_task_results.items():
    for tt, t in tasks.items():
        all_task_stats[tt]["pass"] += t["pass"]
        all_task_stats[tt]["fail"] += t["fail"]

for tt in sorted(all_task_stats.keys()):
    s = all_task_stats[tt]
    total = s["pass"] + s["fail"]
    rate = s["pass"] / total * 100 if total else 0
    if rate < 60:
        print(f"  ⚠️  {tt}: {s['pass']}/{total} ({rate:.0f}%)")

# ── Most common 4xx error paths ──
print("\n" + "=" * 80)
print("TOP 4XX ERROR PATHS (all revisions)")
print("=" * 80)
error_counts = defaultdict(int)
for e in episodes:
    for c in (e.get("api_calls") or []):
        if 400 <= c.get("status", 0) < 500:
            key = f"{c['method']} {c.get('path','?')} -> {c['status']}"
            error_counts[key] += 1

for key, count in sorted(error_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {count:3d}x  {key}")

# ── Rev 0059 specific analysis ──
print("\n" + "=" * 80)
print("REV 0059 DETAILED ANALYSIS")
print("=" * 80)
for e in episodes:
    rev = e.get("revision") or e.get("git_sha") or ""
    if "0059" not in rev:
        continue
    tt = e.get("task_type") or "unknown"
    calls = e.get("api_calls") or []
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = e.get("prompt", "")[:200]
    print(f"\n  [{tt}] n4xx={n4xx}")
    print(f"  prompt: {prompt}")
    for c in calls:
        status = c.get("status", 0)
        marker = " ⚠️" if status >= 400 else ""
        created = f" created={c.get('created_id')}" if c.get('created_id') else ""
        print(f"    {c['method']} {c.get('path','?')} -> {status}{created}{marker}")

# ── Check classification: are prompts being classified correctly? ──
print("\n" + "=" * 80)
print("MISCLASSIFICATION CHECK (recent)")
print("=" * 80)
for e in episodes[-30:]:
    tt = e.get("task_type") or "unknown"
    prompt = (e.get("prompt") or "").lower()
    rev = e.get("revision") or ""
    
    # Check for obvious mismatches
    if tt == "unknown":
        print(f"  ❓ UNKNOWN: rev={rev[-8:]} prompt={prompt[:120]}")
    elif "payroll" in prompt and "lønn" in prompt and tt != "run_payroll":
        print(f"  ⚠️ Should be run_payroll but is {tt}: {prompt[:120]}")
    elif "credit" in prompt and "note" in prompt and tt != "create_credit_note":
        print(f"  ⚠️ Should be credit_note but is {tt}: {prompt[:120]}")
