#!/usr/bin/env python3
"""Deep analysis of all failed episodes with full context."""
import json
from collections import defaultdict

with open('logs/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f if l.strip()]

print(f"Total episodes: {len(episodes)}\n")

# Group by task type
by_task = defaultdict(list)
for ep in episodes:
    task = ep.get('task_type', ep.get('task', 'unknown'))
    by_task[task].append(ep)

# Success rate by task
print("=" * 70)
print("TASK SUCCESS RATES (all revisions)")
print("=" * 70)
for task, eps in sorted(by_task.items(), key=lambda x: len(x[1]), reverse=True):
    ok = sum(1 for e in eps if e.get('outcome') == 'completed' and not any(c.get('status', 200) >= 400 for c in e.get('api_calls', [])))
    total = len(eps)
    errs = total - ok
    pct = f"{100*ok//total}%" if total else "n/a"
    print(f"  {task:35s} {ok:3d}/{total:3d} ({pct:>4s})  err={errs}")

# Show ALL failed episodes with full detail
print("\n" + "=" * 70)
print("DETAILED FAILURES (all revisions)")
print("=" * 70)

for ep in episodes:
    outcome = ep.get('outcome', '?')
    calls = ep.get('api_calls', [])
    errs4 = [c for c in calls if c.get('status', 200) >= 400]
    
    if outcome != 'completed' or errs4:
        rev = str(ep.get('revision', '?'))[-8:]
        task = ep.get('task_type', '?')
        error = (ep.get('error', '') or '')[:200]
        prompt = ep.get('prompt', '')[:200].replace('\n', ' ')
        ts = ep.get('timestamp', '')[:16]
        
        print(f"\n--- {ts} rev={rev} task={task} outcome={outcome} ---")
        print(f"  prompt: {prompt}")
        if error:
            print(f"  error: {error}")
        for c in errs4:
            ep_detail = c.get('response_body', c.get('error', ''))
            if isinstance(ep_detail, str):
                ep_detail = ep_detail[:200]
            print(f"  4xx: {c.get('method','?')} {c.get('endpoint','?')} -> {c.get('status','?')} | {ep_detail}")
        
        # Show parsed entities
        entities = ep.get('parsed_entities', {})
        if entities:
            for k, v in entities.items():
                if v:
                    print(f"  entity.{k}: {json.dumps(v, ensure_ascii=False)[:200]}")

# Count error types
print("\n" + "=" * 70)
print("ERROR PATTERN SUMMARY")
print("=" * 70)
error_patterns = defaultdict(int)
for ep in episodes:
    outcome = ep.get('outcome', '?')
    if outcome != 'completed':
        error = (ep.get('error', '') or '')[:100]
        task = ep.get('task_type', '?')
        error_patterns[f"{task}: {error}"] += 1
    calls = ep.get('api_calls', [])
    for c in calls:
        if c.get('status', 200) >= 400:
            endpoint = c.get('endpoint', '?')
            status = c.get('status', '?')
            task = ep.get('task_type', '?')
            error_patterns[f"{task}: {c.get('method','?')} {endpoint} -> {status}"] += 1

for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
    print(f"  {count:3d}x  {pattern}")

# Recent revisions performance
print("\n" + "=" * 70)
print("RECENT REVISIONS PERFORMANCE")
print("=" * 70)
by_rev = defaultdict(lambda: {"ok": 0, "fail": 0, "tasks": defaultdict(lambda: {"ok": 0, "fail": 0})})
for ep in episodes:
    rev = str(ep.get('revision', '?'))
    task = ep.get('task_type', '?')
    outcome = ep.get('outcome', '?')
    calls = ep.get('api_calls', [])
    has4xx = any(c.get('status', 200) >= 400 for c in calls)
    
    if outcome == 'completed' and not has4xx:
        by_rev[rev]["ok"] += 1
        by_rev[rev]["tasks"][task]["ok"] += 1
    else:
        by_rev[rev]["fail"] += 1
        by_rev[rev]["tasks"][task]["fail"] += 1

for rev in sorted(by_rev.keys()):
    data = by_rev[rev]
    total = data["ok"] + data["fail"]
    pct = f"{100*data['ok']//total}%" if total else "n/a"
    short_rev = rev[-8:] if len(rev) > 8 else rev
    print(f"\n  {short_rev}: {data['ok']}/{total} ({pct})")
    for task, td in sorted(data["tasks"].items()):
        if td["fail"] > 0:
            print(f"    FAIL: {task} ({td['fail']}x)")
