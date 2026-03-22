#!/usr/bin/env python3
"""Analyze recent episodes for failures."""
import json

with open('logs/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f if l.strip()]

# Focus on recent revisions
recent = [e for e in episodes if any(x in str(e.get('revision','')) for x in ['0056','0057','0058','0043','0042','0041','0039'])]
print(f"Recent episodes (rev 039+): {len(recent)}")

if not recent:
    # Fall back to last 50
    recent = episodes[-50:]
    print(f"Fallback: last {len(recent)} episodes")

for ep in recent:
    rev = str(ep.get('revision','?'))[-8:]
    task = ep.get('task_type', ep.get('task','?'))
    outcome = ep.get('outcome','?')
    conf = ep.get('confidence', '?')
    error = (ep.get('error','') or '')[:150]
    calls = ep.get('api_calls', [])
    errs4 = [c for c in calls if c.get('status',200) >= 400]
    prompt = ep.get('prompt','')[:120].replace('\n',' ')
    checks = ep.get('checks', [])
    failed_checks = [c for c in checks if not c.get('passed', True)]
    
    has_issue = outcome != 'completed' or errs4 or failed_checks
    status = 'FAIL' if has_issue else 'OK'
    
    print(f"\n{status} rev={rev} task={task} outcome={outcome} conf={conf}")
    if error:
        print(f"   err: {error}")
    for e4 in errs4:
        print(f"   4xx: {e4.get('method','?')} {e4.get('endpoint','?')} -> {e4.get('status','?')}")
    for fc in failed_checks:
        print(f"   check_fail: {fc}")
    print(f"   prompt: {prompt}")
