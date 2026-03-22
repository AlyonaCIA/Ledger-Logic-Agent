#!/usr/bin/env python3
"""Quick analysis of episode logs."""
import json
from collections import defaultdict

episodes = []
for path in ['logs/episodes.jsonl', 'episodes.jsonl']:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    except Exception:
        pass

task_stats = defaultdict(lambda: {'total': 0, 'completed': 0, 'error': 0, 'timeout': 0, 'errors_4xx': []})
for ep in episodes:
    tt = ep.get('task_type', 'unknown')
    outcome = ep.get('outcome', 'unknown')
    task_stats[tt]['total'] += 1
    task_stats[tt][outcome] = task_stats[tt].get(outcome, 0) + 1
    m = ep.get('metrics', {})
    task_stats[tt]['errors_4xx'].append(m.get('error_4xx', 0))

print(f'Total episodes: {len(episodes)}')
print(f'\n{"Task Type":<30} {"Total":>5} {"OK":>5} {"Err":>5} {"TO":>5} {"Avg4xx":>7}')
print('-' * 70)
for tt in sorted(task_stats, key=lambda t: -task_stats[t]['total']):
    s = task_stats[tt]
    avg4xx = sum(s['errors_4xx']) / len(s['errors_4xx']) if s['errors_4xx'] else 0
    print(f'{tt:<30} {s["total"]:>5} {s.get("completed",0):>5} {s.get("error",0):>5} {s.get("timeout",0):>5} {avg4xx:>7.1f}')

# Show recent errors
print('\n\n=== RECENT ERRORS ===')
recent = sorted(episodes, key=lambda e: e.get('timestamp', ''))[-50:]
for ep in recent:
    if ep.get('outcome') != 'completed' or ep.get('metrics', {}).get('error_4xx', 0) > 2:
        print(f"  {ep.get('timestamp','?')} task={ep.get('task_type','?')} outcome={ep.get('outcome','?')} "
              f"4xx={ep.get('metrics',{}).get('error_4xx',0)} calls={ep.get('metrics',{}).get('calls',0)} "
              f"err={str(ep.get('error',''))[:100]}")
