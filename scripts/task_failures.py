#!/usr/bin/env python3
"""Show detailed failures for specific task types."""
import json

with open('logs/episodes.jsonl') as f:
    episodes = [json.loads(l) for l in f if l.strip()]

# Focus on task types with highest failure rates
focus_tasks = ['run_payroll', 'create_credit_note', 'create_travel_expense', 
               'create_supplier_invoice', 'register_payment', 'ledger_task',
               'create_project_invoice', 'bank_reconciliation', 'overdue_reminder']

for task_focus in focus_tasks:
    task_eps = [e for e in episodes if e.get('task_type') == task_focus]
    if not task_eps:
        continue
    
    ok = sum(1 for e in task_eps if e.get('outcome') == 'completed' and not any(c.get('status', 200) >= 400 for c in e.get('api_calls', [])))
    total = len(task_eps)
    print(f"\n{'='*70}")
    print(f"  {task_focus}: {ok}/{total} OK")
    print(f"{'='*70}")
    
    for ep in task_eps:
        outcome = ep.get('outcome', '?')
        calls = ep.get('api_calls', [])
        has4xx = any(c.get('status', 200) >= 400 for c in calls)
        prompt = ep.get('prompt', '').replace('\n', ' ')
        rev = str(ep.get('revision', ''))[-8:]
        entities = ep.get('parsed_entities', {})
        error = (ep.get('error', '') or '')
        
        status = 'FAIL' if outcome != 'completed' or has4xx else 'OK'
        
        print(f"\n  [{status}] rev={rev}")
        print(f"    prompt: {prompt[:300]}")
        if entities:
            for k, v in entities.items():
                if v and k != '_raw_prompt':
                    print(f"    {k}: {json.dumps(v, ensure_ascii=False)[:250]}")
        if has4xx:
            for c in calls:
                if c.get('status', 200) >= 400:
                    rb = c.get('response_body', '')
                    if isinstance(rb, dict):
                        rb = json.dumps(rb, ensure_ascii=False)[:200]
                    elif isinstance(rb, str):
                        rb = rb[:200]
                    print(f"    4xx: {c.get('method','?')} {c.get('endpoint','?')} -> {c.get('status','?')} | {rb}")
        if error:
            print(f"    error: {error[:200]}")

# Also show "unknown" misclassifications
print(f"\n{'='*70}")
print(f"  MISCLASSIFIED AS UNKNOWN")
print(f"{'='*70}")
unknown_eps = [e for e in episodes if e.get('task_type') == 'unknown']
for ep in unknown_eps[-15:]:
    prompt = ep.get('prompt', '').replace('\n', ' ')
    conf = ep.get('confidence', '?')
    rev = str(ep.get('revision', ''))[-8:]
    error = (ep.get('error', '') or '')[:100]
    print(f"\n  rev={rev} conf={conf}")
    print(f"    prompt: {prompt[:200]}")
    if error:
        print(f"    error: {error}")
