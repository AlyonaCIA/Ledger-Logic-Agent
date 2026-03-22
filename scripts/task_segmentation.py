#!/usr/bin/env python3
"""Comprehensive task segmentation analysis across all episodes."""
import json
from collections import defaultdict

episodes = [json.loads(l) for l in open("logs/episodes.jsonl")]

# ═══════════════════════════════════════════════════════════════
# Helper: classify episode as pass/fail
# ═══════════════════════════════════════════════════════════════
def classify(ep):
    calls = ep.get("api_calls", [])
    outcome = ep.get("outcome", "?")
    n4xx = sum(1 for c in calls if 400 <= c.get("status", 0) < 500)
    return outcome == "completed" and n4xx == 0, n4xx, calls

# ═══════════════════════════════════════════════════════════════
# 1. TASK SEGMENTATION
# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  TASK SEGMENTATION — ALL EPISODES")
print("=" * 80)

task_data = defaultdict(lambda: {
    "total": 0, "passes": 0, "fail_4xx": 0, "fail_error": 0,
    "revs_pass": set(), "revs_fail": set(),
    "fail_details": [],
})

for ep in episodes:
    tt = ep.get("task_type", "unknown")
    rev = (ep.get("revision") or ep.get("git_sha") or "?")[-12:]
    passed, n4xx, calls = classify(ep)
    outcome = ep.get("outcome", "?")
    
    td = task_data[tt]
    td["total"] += 1
    
    if passed:
        td["passes"] += 1
        td["revs_pass"].add(rev)
    elif outcome == "error":
        td["fail_error"] += 1
        td["revs_fail"].add(rev)
        td["fail_details"].append({
            "type": "error", "rev": rev,
            "error": ep.get("error", "")[:120],
            "prompt": ep.get("prompt", "")[:100],
        })
    else:
        td["fail_4xx"] += 1
        td["revs_fail"].add(rev)
        bad = [f"{c['method']} {c.get('path','?')} -> {c['status']}" 
               for c in calls if c.get("status", 0) >= 400]
        td["fail_details"].append({
            "type": "4xx", "rev": rev, "n4xx": n4xx,
            "bad_calls": bad[:5],
            "prompt": ep.get("prompt", "")[:100],
        })

# Sort worst→best
sorted_tasks = sorted(task_data.items(), 
                       key=lambda x: x[1]["passes"] / max(x[1]["total"], 1))

for tt, td in sorted_tasks:
    rate = td["passes"] / td["total"] * 100 if td["total"] else 0
    bar = "#" * int(rate / 5) + "." * (20 - int(rate / 5))
    if rate == 0:
        icon = "CRIT"
    elif rate < 50:
        icon = "BAD "
    elif rate < 80:
        icon = "WARN"
    else:
        icon = "OK  "
    print(f"\n[{icon}] {tt:30s} {td['passes']:>3d}/{td['total']:>3d} ({rate:5.1f}%) [{bar}]")
    print(f"       pass={td['passes']} fail_4xx={td['fail_4xx']} fail_error={td['fail_error']}")
    if td["revs_pass"]:
        recent = sorted(td["revs_pass"])[-3:]
        print(f"       PASSED on revs: {', '.join(recent)}")
    if td["revs_fail"]:
        recent = sorted(td["revs_fail"])[-3:]
        print(f"       FAILED on revs: {', '.join(recent)}")

# ═══════════════════════════════════════════════════════════════
# 2. RECENT REVISIONS ONLY (0058+) — What matters NOW
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  RECENT PERFORMANCE (rev 0055+)")
print("=" * 80)

recent_tasks = defaultdict(lambda: {"pass": 0, "fail": 0, "details": []})
for ep in episodes:
    rev = (ep.get("revision") or ep.get("git_sha") or "?")
    # Only revs 0055 and later
    rev_num = ""
    for part in rev.split("-"):
        if part.isdigit() and len(part) == 5:
            rev_num = part
    if not rev_num or int(rev_num) < 55:
        continue
    
    tt = ep.get("task_type", "unknown")
    passed, n4xx, calls = classify(ep)
    
    if passed:
        recent_tasks[tt]["pass"] += 1
    else:
        recent_tasks[tt]["fail"] += 1
        bad = [f"{c['method']} {c.get('path','?')} -> {c['status']}" 
               for c in calls if c.get("status", 0) >= 400]
        recent_tasks[tt]["details"].append({
            "rev": rev[-12:],
            "n4xx": n4xx,
            "bad_calls": bad[:3],
            "error": ep.get("error", "")[:80] if ep.get("outcome") == "error" else "",
            "prompt": ep.get("prompt", "")[:80],
        })

for tt in sorted(recent_tasks, key=lambda x: recent_tasks[x]["fail"], reverse=True):
    d = recent_tasks[tt]
    total = d["pass"] + d["fail"]
    rate = d["pass"] / total * 100
    icon = "v" if rate == 100 else ("X" if rate == 0 else "~")
    print(f"\n  [{icon}] {tt:30s} {d['pass']}/{total} ({rate:.0f}%)")
    for detail in d["details"][-3:]:  # last 3 failures
        print(f"      FAIL rev={detail['rev']} n4xx={detail['n4xx']}")
        if detail["error"]:
            print(f"        error: {detail['error']}")
        for bc in detail["bad_calls"]:
            print(f"        -> {bc}")
        print(f"        prompt: {detail['prompt']}")

# ═══════════════════════════════════════════════════════════════
# 3. FAILURE DEEP-DIVE: For each failing task type, show ALL
#    failure patterns to find the root cause
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FAILURE PATTERN ANALYSIS (task types that fail on recent revs)")
print("=" * 80)

# Collect all unique error paths per task type
for tt in sorted(recent_tasks, key=lambda x: recent_tasks[x]["fail"], reverse=True):
    d = recent_tasks[tt]
    if d["fail"] == 0:
        continue
    print(f"\n--- {tt} (fails: {d['fail']}/{d['pass']+d['fail']}) ---")
    
    # All failure patterns for this task
    error_paths = defaultdict(int)
    for detail in d["details"]:
        for bc in detail["bad_calls"]:
            error_paths[bc] += 1
        if detail["error"]:
            error_paths[f"ERROR: {detail['error'][:60]}"] += 1
    
    for path, count in sorted(error_paths.items(), key=lambda x: -x[1]):
        print(f"    {count}x  {path}")

# ═══════════════════════════════════════════════════════════════
# 4. TASK ROUTING ANALYSIS - check if parser detects correctly
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TASK ROUTING — DETECTION ACCURACY")
print("=" * 80)

# Check for probable misclassifications
misclass = []
for ep in episodes[-50:]:
    tt = ep.get("task_type", "unknown")
    prompt = (ep.get("prompt") or "").lower()
    conf = ep.get("confidence", 0)
    rev = (ep.get("revision") or ep.get("git_sha") or "?")[-12:]
    
    expected = None
    if any(w in prompt for w in ["payroll", "lønn", "nómina", "gehalt", "salário", "salary"]):
        if "run" in prompt or "kjør" in prompt or "ejecut" in prompt or "führ" in prompt or "realiz" in prompt:
            expected = "run_payroll"
    elif any(w in prompt for w in ["credit note", "kreditnota", "nota de crédito", "gutschrift"]):
        expected = "create_credit_note"
    elif any(w in prompt for w in ["travel expense", "reiseregning", "dieta", "reisekosten"]):
        expected = "create_travel_expense"
    elif any(w in prompt for w in ["supplier invoice", "leverandørfaktura", "eingangsrechnung", "incomingInvoice"]):
        if "registrer" in prompt or "register" in prompt or "erfass" in prompt or "received" in prompt or "mottatt" in prompt:
            expected = "create_supplier_invoice"
    elif any(w in prompt for w in ["delete", "slett", "eliminar", "löschen"]) and any(w in prompt for w in ["voucher", "bilag"]):
        expected = "delete_voucher"
    
    if expected and tt != expected:
        misclass.append(f"  rev={rev} detected={tt:25s} expected={expected:25s} conf={conf:.2f}")
        misclass.append(f"    prompt: {prompt[:100]}")

if misclass:
    for line in misclass:
        print(line)
else:
    print("  No obvious misclassifications detected in last 50 episodes.")

# ═══════════════════════════════════════════════════════════════
# 5. TASK TYPE FREQUENCY — what competition sends most
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TASK FREQUENCY (what the competition sends)")
print("=" * 80)
freq = defaultdict(int)
for ep in episodes:
    freq[ep.get("task_type", "unknown")] += 1
for tt, count in sorted(freq.items(), key=lambda x: -x[1]):
    print(f"  {count:>4}x  {tt}")

# ═══════════════════════════════════════════════════════════════
# 6. NEW: Timeline — show pass/fail trend for key task types
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  TIMELINE — EVOLUTION PER TASK TYPE (rev 0050+)")
print("=" * 80)

key_tasks = ["create_invoice", "create_supplier_invoice", "create_credit_note",
             "create_travel_expense", "run_payroll", "ledger_task", 
             "create_employee", "create_customer", "create_product",
             "overdue_reminder", "register_payment", "create_project_invoice",
             "delete_voucher"]

for tt in key_tasks:
    timeline = []
    for ep in episodes:
        if ep.get("task_type") != tt:
            continue
        rev = (ep.get("revision") or ep.get("git_sha") or "?")
        rev_num = ""
        for part in rev.split("-"):
            if part.isdigit() and len(part) == 5:
                rev_num = part
        if not rev_num or int(rev_num) < 50:
            continue
        passed, n4xx, _ = classify(ep)
        timeline.append((rev_num, "v" if passed else "X"))
    
    if timeline:
        marks = " ".join(f"{r}:{'v' if m == 'v' else 'X'}" for r, m in sorted(timeline))
        print(f"  {tt:30s}  {marks}")

print("\n\nDone. Total episodes analyzed:", len(episodes))
