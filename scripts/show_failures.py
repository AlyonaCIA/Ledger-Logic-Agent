#!/usr/bin/env python3
"""Show detailed failure patterns for specific task types."""
import json

episodes = [json.loads(l) for l in open("logs/episodes.jsonl")]

for e in episodes:
    tt = e.get("task_type", "")
    if tt not in ("create_travel_expense", "create_credit_note", "ledger_task", "create_supplier_invoice"):
        continue

    rev = e.get("revision", "?")
    n4xx = e.get("metrics", {}).get("error_4xx", 0)
    calls = e.get("api_calls", [])
    failed_calls = [c for c in calls if c.get("status", 0) >= 400]
    ledger = e.get("parsed_entities", {}).get("ledger", {})
    subtask = ledger.get("subtask", "")

    if tt == "ledger_task" and "custom_dimension" not in subtask:
        continue

    if not failed_calls:
        continue  # skip successes

    print(f"=== {tt} rev={rev[-8:] if len(rev) > 8 else rev} n4xx={n4xx} subtask={subtask} ===")
    prompt = e.get("prompt", "")[:200]
    print(f"  prompt: {prompt}")
    for c in failed_calls[:5]:
        print(f"  FAIL: {c['method']} {c['path']} -> {c['status']}")
    print()
