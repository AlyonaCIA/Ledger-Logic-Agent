#!/usr/bin/env python3
"""Deep-dive analysis of specific failing task types."""
import json

episodes = [json.loads(l) for l in open("logs/episodes.jsonl")]

# ═══════════════════════════════════════════════════════
# 1. ALL run_payroll EPISODES
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("  ALL run_payroll EPISODES")
print("=" * 70)

payroll_eps = [ep for ep in episodes if ep.get("task_type") == "run_payroll"]
if not payroll_eps:
    print("  No run_payroll episodes found!")
for ep in payroll_eps:
    rev = (ep.get("revision") or ep.get("git_sha") or "?")[-12:]
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    pe = ep.get("parsed_entities", {})
    print(f"\n  rev={rev} outcome={ep.get('outcome')} n4xx={n4xx}")
    print(f"  prompt: {ep.get('prompt','')[:200]}")
    print(f"  payroll: {json.dumps(pe.get('payroll',{}), indent=4)}")
    print(f"  employee: {json.dumps(pe.get('employee',{}), indent=4)}")
    for c in calls:
        st = c.get("status", 0)
        marker = " *** 4XX ***" if st >= 400 else ""
        print(f"    {c.get('method')} {c.get('path')} -> {st}{marker}")
    if ep.get("error"):
        print(f"  ERROR: {ep['error'][:200]}")

# ═══════════════════════════════════════════════════════
# 2. ALL create_supplier_invoice FAILURES (recent revs)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  create_supplier_invoice FAILURES (rev 0050+)")
print("=" * 70)

for ep in episodes:
    if ep.get("task_type") != "create_supplier_invoice":
        continue
    rev = (ep.get("revision") or ep.get("git_sha") or "?")
    rev_num = ""
    for part in rev.split("-"):
        if part.isdigit() and len(part) == 5:
            rev_num = part
    if not rev_num or int(rev_num) < 50:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    outcome = ep.get("outcome", "?")
    passed = n4xx == 0 and outcome == "completed"
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] rev={rev[-12:]} outcome={outcome} n4xx={n4xx}")
    print(f"  prompt: {ep.get('prompt','')[:200]}")
    pe = ep.get("parsed_entities", {})
    inv = pe.get("invoice", {})
    cust = pe.get("customer", {})
    notes = pe.get("notes", "")
    print(f"  customer: name={cust.get('name')} org={cust.get('org_number')} is_supplier={cust.get('is_supplier')}")
    print(f"  invoice: num={inv.get('invoice_number')} amount={inv.get('amount')} excl_vat={inv.get('amount_excl_vat')} acct={inv.get('account_code')} date={inv.get('date')}")
    print(f"  notes: {notes}")
    for c in calls:
        st = c.get("status", 0)
        marker = " *** 4XX ***" if st >= 400 else ""
        print(f"    {c.get('method')} {c.get('path')} -> {st}{marker}")
    if ep.get("error"):
        print(f"  ERROR: {ep['error'][:200]}")

# ═══════════════════════════════════════════════════════
# 3. ALL create_travel_expense episodes (recent)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  create_travel_expense EPISODES (rev 0050+)")
print("=" * 70)

for ep in episodes:
    if ep.get("task_type") != "create_travel_expense":
        continue
    rev = (ep.get("revision") or ep.get("git_sha") or "?")
    rev_num = ""
    for part in rev.split("-"):
        if part.isdigit() and len(part) == 5:
            rev_num = part
    if not rev_num or int(rev_num) < 50:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    outcome = ep.get("outcome", "?")
    passed = n4xx == 0 and outcome == "completed"
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] rev={rev[-12:]} outcome={outcome} n4xx={n4xx}")
    print(f"  prompt: {ep.get('prompt','')[:200]}")
    for c in calls:
        st = c.get("status", 0)
        marker = " *** 4XX ***" if st >= 400 else ""
        print(f"    {c.get('method')} {c.get('path')} -> {st}{marker}")
    if ep.get("error"):
        print(f"  ERROR: {ep['error'][:200]}")

# ═══════════════════════════════════════════════════════
# 4. Check: what prompts are classified as "unknown"?
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  UNKNOWN TASK TYPE EPISODES (rev 0050+)")
print("=" * 70)

for ep in episodes:
    if ep.get("task_type") != "unknown":
        continue
    rev = (ep.get("revision") or ep.get("git_sha") or "?")
    rev_num = ""
    for part in rev.split("-"):
        if part.isdigit() and len(part) == 5:
            rev_num = part
    if not rev_num or int(rev_num) < 50:
        continue
    calls = ep.get("api_calls", [])
    outcome = ep.get("outcome", "?")
    print(f"\n  rev={rev[-12:]} outcome={outcome}")
    print(f"  prompt: {ep.get('prompt','')[:200]}")
    if ep.get("error"):
        print(f"  ERROR: {ep['error'][:200]}")

# ═══════════════════════════════════════════════════════
# 5. Check: what about ledger_task custom_dimension fail?
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ledger_task FAILURES (rev 0050+)")
print("=" * 70)

for ep in episodes:
    if ep.get("task_type") != "ledger_task":
        continue
    rev = (ep.get("revision") or ep.get("git_sha") or "?")
    rev_num = ""
    for part in rev.split("-"):
        if part.isdigit() and len(part) == 5:
            rev_num = part
    if not rev_num or int(rev_num) < 50:
        continue
    calls = ep.get("api_calls", [])
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    if n4xx == 0 and ep.get("outcome") == "completed":
        rev_short = rev[-12:]
        print(f"  [PASS] rev={rev_short} (skipping details)")
        continue
    outcome = ep.get("outcome", "?")
    print(f"\n  [FAIL] rev={rev[-12:]} outcome={outcome} n4xx={n4xx}")
    print(f"  prompt: {ep.get('prompt','')[:200]}")
    pe = ep.get("parsed_entities", {})
    print(f"  ledger: {json.dumps(pe.get('ledger',{}), indent=4)[:500]}")
    for c in calls:
        st = c.get("status", 0)
        marker = " *** 4XX ***" if st >= 400 else ""
        print(f"    {c.get('method')} {c.get('path')} -> {st}{marker}")
    if ep.get("error"):
        print(f"  ERROR: {ep['error'][:200]}")

print("\nDone.")
