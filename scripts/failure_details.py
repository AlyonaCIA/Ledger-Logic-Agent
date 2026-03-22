#!/usr/bin/env python3
"""Analyze specific failing task types in detail."""
import json
import re

eps = [json.loads(l) for l in open("logs/episodes.jsonl")]

print("=== RUN_PAYROLL EPISODES (all) ===")
for ep in eps:
    if ep.get("task_type") != "run_payroll":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    calls = ep.get("api_calls", [])
    n_calls = len(calls)
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    error = ep.get("error", "")
    print(f"rev={rev} outcome={outcome} calls={n_calls} 4xx={n4xx}")
    if error:
        print(f"  ERROR: {error[:300]}")
    print(f"  prompt: {prompt}")
    for c in calls:
        print(f"  {c.get('method')} {c.get('path', '')[:60]} -> {c.get('status')}")
        if 400 <= c.get("status", 0) < 500:
            print(f"    response: {str(c.get('response_body', ''))[:300]}")
    print()

print("\n=== CREATE_INVOICE RECENT FAILURES (rev 55+) ===")
for ep in eps:
    if ep.get("task_type") != "create_invoice":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    m = re.search(r"(\d{5})", rev)
    if not m or int(m.group(1)) < 55:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    if n4xx == 0:
        continue
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    print(f"rev={rev} outcome={outcome} 4xx={n4xx}")
    print(f"  prompt: {prompt}")
    for c in calls:
        if 400 <= c.get("status", 0) < 500:
            print(f"  {c.get('method')} {c.get('path', '')[:80]} -> {c.get('status')}")
            print(f"    body: {str(c.get('response_body', ''))[:300]}")
    print()

print("\n=== LEDGER_TASK RECENT FAILURES (rev 55+) ===")
for ep in eps:
    if ep.get("task_type") != "ledger_task":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    m = re.search(r"(\d{5})", rev)
    if not m or int(m.group(1)) < 55:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    error = ep.get("error", "")
    print(f"rev={rev} outcome={outcome} 4xx={n4xx}")
    print(f"  prompt: {prompt}")
    if error:
        print(f"  ERROR: {error[:300]}")
    for c in calls:
        if 400 <= c.get("status", 0) < 500:
            print(f"  {c.get('method')} {c.get('path', '')[:80]} -> {c.get('status')}")
            print(f"    body: {str(c.get('response_body', ''))[:300]}")
    print()

print("\n=== CREATE_SUPPLIER_INVOICE RECENT FAILURES (rev 55+) ===")
for ep in eps:
    if ep.get("task_type") != "create_supplier_invoice":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    m = re.search(r"(\d{5})", rev)
    if not m or int(m.group(1)) < 55:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    if n4xx == 0:
        continue
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    error = ep.get("error", "")
    print(f"rev={rev} outcome={outcome} 4xx={n4xx}")
    print(f"  prompt: {prompt}")
    if error:
        print(f"  ERROR: {error[:300]}")
    for c in calls:
        if 400 <= c.get("status", 0) < 500:
            print(f"  {c.get('method')} {c.get('path', '')[:80]} -> {c.get('status')}")
            print(f"    body: {str(c.get('response_body', ''))[:300]}")
    print()

print("\n=== CREATE_TRAVEL_EXPENSE EPISODES (all) ===")
for ep in eps:
    if ep.get("task_type") != "create_travel_expense":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    error = ep.get("error", "")
    print(f"rev={rev} outcome={outcome} calls={len(calls)} 4xx={n4xx}")
    if error:
        print(f"  ERROR: {error[:300]}")
    print(f"  prompt: {prompt}")
    for c in calls:
        if 400 <= c.get("status", 0) < 500:
            print(f"  {c.get('method')} {c.get('path', '')[:80]} -> {c.get('status')}")
            print(f"    body: {str(c.get('response_body', ''))[:300]}")
    print()

print("\n=== CREATE_CREDIT_NOTE EPISODES (all) ===")
for ep in eps:
    if ep.get("task_type") != "create_credit_note":
        continue
    rev = str(ep.get("revision", "") or ep.get("git_sha", ""))
    if "agent-" in rev:
        rev = rev.split("agent-")[1][:9]
    outcome = ep.get("outcome")
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    prompt = ep.get("prompt", "")[:300].replace("\n", " ")
    error = ep.get("error", "")
    print(f"rev={rev} outcome={outcome} calls={len(calls)} 4xx={n4xx}")
    if error:
        print(f"  ERROR: {error[:200]}")
    print(f"  prompt: {prompt}")
    for c in calls:
        if 400 <= c.get("status", 0) < 500:
            print(f"  {c.get('method')} {c.get('path', '')[:80]} -> {c.get('status')}")
            print(f"    body: {str(c.get('response_body', ''))[:300]}")
    print()
