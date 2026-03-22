#!/usr/bin/env python3
"""Analyze episodes from logs/episodes.jsonl"""
import json, sys
from collections import defaultdict

with open('logs/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f if l.strip()]

by_task = defaultdict(list)
for e in episodes:
    tt = e.get('task_type', 'unknown')
    by_task[tt].append(e)

print('=== ALL TASK TYPES (all-time) ===')
for tt in sorted(by_task.keys()):
    eps = by_task[tt]
    errors = sum(1 for e in eps if e.get('outcome') != 'completed' or e.get('metrics', {}).get('error_4xx', 0) > 0)
    clean = len(eps) - errors
    print(f'  {tt:35} total={len(eps):3} clean={clean:3} errors={errors:3}')

print()
print('=== LATEST 30 EPISODES ===')
for e in episodes[-30:]:
    tt = e.get('task_type', 'unknown')
    rev = e.get('git_sha', '?')
    outcome = e.get('outcome', '?')
    m = e.get('metrics', {})
    calls = m.get('calls', 0)
    e4 = m.get('error_4xx', 0)
    e5 = m.get('error_5xx', 0)
    total_ms = m.get('latency_total_ms', 0)
    prompt_snip = (e.get('prompt', ''))[:70]
    print(f'  [{e.get("request_id","?"):8}] rev={rev:14} {tt:30} out={outcome:10} calls={calls:2} 4xx={e4} 5xx={e5} {total_ms:6}ms | {prompt_snip}')

print()
print('=== ERRORS IN LAST 50 EPISODES ===')
for e in episodes[-50:]:
    if e.get('outcome') != 'completed' or e.get('metrics', {}).get('error_4xx', 0) > 0:
        tt = e.get('task_type', 'unknown')
        rev = e.get('git_sha', '?')
        err = (e.get('error') or '')[:100]
        m = e.get('metrics', {})
        e4 = m.get('error_4xx', 0)
        calls_detail = []
        for c in e.get('api_calls', []):
            if c.get('status', 200) >= 400:
                calls_detail.append(f"  {c.get('method')} {c.get('path')} -> {c.get('status')}")
        prompt_snip = (e.get('prompt', ''))[:80]
        print(f'  [{e.get("request_id","?"):8}] {tt:30} 4xx={e4} err={err}')
        print(f'    prompt: {prompt_snip}')
        for cd in calls_detail:
            print(f'    {cd}')
        print()
