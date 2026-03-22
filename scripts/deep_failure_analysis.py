#!/usr/bin/env python3
"""Deep analysis of failure patterns and unknown tasks."""
import json

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

# Show unknown task prompts
print("=== UNKNOWN TASKS (prompt samples) ===")
unknowns = [e for e in episodes if e.get('task_type') == 'unknown']
for ep in unknowns[-15:]:
    print(f"  outcome={ep.get('outcome')} prompt: {ep.get('prompt','')[:200]}")
    print()

# Show create_invoice errors
print("\n=== CREATE_INVOICE ERRORS ===")
inv_errors = [e for e in episodes if e.get('task_type') == 'create_invoice' and e.get('outcome') != 'completed']
for ep in inv_errors[-5:]:
    print(f"  outcome={ep.get('outcome')} err={str(ep.get('error',''))[:200]} prompt: {ep.get('prompt','')[:200]}")
    print()

# Show create_customer errors  
print("\n=== CREATE_CUSTOMER ERRORS ===")
cust_errors = [e for e in episodes if e.get('task_type') == 'create_customer' and e.get('outcome') != 'completed']
for ep in cust_errors[-5:]:
    print(f"  outcome={ep.get('outcome')} err={str(ep.get('error',''))[:200]} prompt: {ep.get('prompt','')[:200]}")
    print()

# Show supplier invoice errors
print("\n=== SUPPLIER INVOICE ERRORS ===")
si_errors = [e for e in episodes if e.get('task_type') == 'create_supplier_invoice' and e.get('outcome') != 'completed']
for ep in si_errors[-5:]:
    print(f"  outcome={ep.get('outcome')} err={str(ep.get('error',''))[:200]} prompt: {ep.get('prompt','')[:200]}")
    print()

# Show travel expense errors
print("\n=== TRAVEL EXPENSE ERRORS ===")
te_errors = [e for e in episodes if e.get('task_type') == 'create_travel_expense' and e.get('outcome') != 'completed']
for ep in te_errors[-5:]:
    print(f"  outcome={ep.get('outcome')} err={str(ep.get('error',''))[:200]} prompt: {ep.get('prompt','')[:200]}")
    print()

# Show high-4xx tasks
print("\n=== HIGH 4xx ERROR RUNS (>2 errors) ===")
high_4xx = [e for e in episodes if e.get('metrics', {}).get('error_4xx', 0) > 2]
for ep in high_4xx[-10:]:
    m = ep.get('metrics', {})
    calls = ep.get('api_calls', [])
    errors = [c for c in calls if 400 <= c.get('status', 0) < 500]
    print(f"  task={ep.get('task_type')} 4xx={m.get('error_4xx')} calls={m.get('calls')} outcome={ep.get('outcome')}")
    for err in errors[:5]:
        print(f"    {err.get('method')} {err.get('path')} -> {err.get('status')}")
    print()

# Credit note errors
print("\n=== CREDIT NOTE ERRORS ===")
cn_errors = [e for e in episodes if e.get('task_type') == 'create_credit_note' and e.get('outcome') != 'completed']
for ep in cn_errors[-5:]:
    print(f"  outcome={ep.get('outcome')} err={str(ep.get('error',''))[:200]} prompt: {ep.get('prompt','')[:200]}")
    print()
