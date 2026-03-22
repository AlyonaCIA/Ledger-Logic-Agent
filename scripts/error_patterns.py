#!/usr/bin/env python3
"""Show error patterns and recent revision details."""
import json
from collections import defaultdict

with open('logs/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f if l.strip()]

# Error patterns
error_patterns = defaultdict(int)
for ep in episodes:
    outcome = ep.get('outcome', '?')
    if outcome != 'completed':
        error = (ep.get('error', '') or '')[:80]
        task = ep.get('task_type', '?')
        error_patterns[f"{task}: {error}"] += 1
    calls = ep.get('api_calls', [])
    for c in calls:
        if c.get('status', 200) >= 400:
            task = ep.get('task_type', '?')
            error_patterns[f"{task}: {c.get('method','?')} {c.get('endpoint','?')} -> {c.get('status','?')}"] += 1

print('ERROR PATTERN SUMMARY (top 30):')
for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:30]:
    print(f'  {count:3d}x  {pattern}')

# Recent revisions only (39+)
print('\nRECENT REVISIONS DETAIL (039+):')
for ep in episodes:
    rev = str(ep.get('revision', ''))
    if not any(x in rev for x in ['0039', '0041', '0042', '0043', '0056', '0057', '0058']):
        continue
    task = ep.get('task_type', '?')
    outcome = ep.get('outcome', '?')
    calls = ep.get('api_calls', [])
    has4xx = any(c.get('status', 200) >= 400 for c in calls)
    prompt = ep.get('prompt', '')[:140].replace('\n', ' ')
    status = 'FAIL' if outcome != 'completed' or has4xx else 'OK'
    
    print(f'\n  {status} rev={rev[-8:]} task={task} outcome={outcome}')
    if has4xx:
        for c in calls:
            if c.get('status', 200) >= 400:
                print(f'    4xx: {c.get("method","?")} {c.get("endpoint","?")} -> {c.get("status","?")}')
    if status == 'FAIL':
        err = (ep.get('error','') or '')[:150]
        if err:
            print(f'    error: {err}')
    print(f'    prompt: {prompt}')

# Show full prompts for failed episodes on recent revisions
print('\n\nFULL PROMPTS FOR RECENT FAILURES:')
for ep in episodes:
    rev = str(ep.get('revision', ''))
    if not any(x in rev for x in ['0039', '0041', '0042', '0043', '0056', '0057', '0058']):
        continue
    outcome = ep.get('outcome', '?')
    calls = ep.get('api_calls', [])
    has4xx = any(c.get('status', 200) >= 400 for c in calls)
    if outcome != 'completed' or has4xx:
        task = ep.get('task_type', '?')
        prompt = ep.get('prompt', '').replace('\n', ' ')
        entities = ep.get('parsed_entities', {})
        print(f'\n--- rev={rev[-8:]} task={task} ---')
        print(f'  FULL PROMPT: {prompt}')
        print(f'  ENTITIES: {json.dumps(entities, ensure_ascii=False)[:400]}')
        err = (ep.get('error','') or '')
        if err:
            print(f'  ERROR: {err[:300]}')
