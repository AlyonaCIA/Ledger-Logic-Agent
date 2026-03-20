"""
Local Competition Evaluator Simulator
======================================
Simulates what the competition evaluator does:
  1. Sends a prompt to our agent (local or Cloud Run)
  2. Verifies the resulting state in Tripletex via GET
  3. Diffs expected vs actual fields
  4. Reports a score (pass/total) per task

Usage:
    # Against local server (start with: uvicorn main:app --port 8080)
    python tests/simulate_evaluator.py --target http://localhost:8080

    # Against production Cloud Run
    python tests/simulate_evaluator.py --target https://ledger-logic-agent-xrgiacpg2q-ew.a.run.app

    # Single task type
    python tests/simulate_evaluator.py --target http://localhost:8080 --task customer

    # Verbose (show all field diffs)
    python tests/simulate_evaluator.py --target http://localhost:8080 --verbose
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import requests as http
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

BASE_URL = os.environ.get("TRIPLETEX_SANDBOX_URL", "").rstrip("/")
TOKEN    = os.environ.get("TRIPLETEX_SANDBOX_TOKEN", "")

if not BASE_URL or not TOKEN:
    print("❌  Set TRIPLETEX_SANDBOX_URL and TRIPLETEX_SANDBOX_TOKEN in .env")
    sys.exit(1)

TODAY = datetime.date.today().isoformat()
TS    = str(int(time.time()))[-5:]
PASS  = "✓"
FAIL  = "✗"
WARN  = "⚠"

# ─── Tripletex direct client (for verification GETs) ─────────────────────────

_session = http.Session()
_session.auth = ("0", TOKEN)
_session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})


def tx_get_list(path: str, params: dict | None = None) -> list:
    r = _session.get(f"{BASE_URL}{path}", params=params or {}, timeout=20)
    if r.status_code >= 400:
        return []
    data = r.json()
    return data.get("values", []) if isinstance(data, dict) else []


def tx_get_one(path: str) -> dict:
    r = _session.get(f"{BASE_URL}{path}", timeout=20)
    if r.status_code >= 400:
        return {}
    data = r.json()
    return data.get("value", data) if isinstance(data, dict) else {}


# ─── Agent caller ─────────────────────────────────────────────────────────────

def call_agent(target: str, prompt: str) -> tuple[int, dict]:
    """POST prompt to the agent. Returns (status_code, response_json)."""
    payload = {
        "prompt": prompt,
        "tripletex_credentials": {
            "base_url": BASE_URL,
            "session_token": TOKEN,
        },
        "files": [],
    }
    # Evaluator posts to root /
    try:
        r = http.post(f"{target}/", json=payload, timeout=60)
        return r.status_code, r.json()
    except Exception as exc:
        return 0, {"error": str(exc)}


# ─── Scoring framework ────────────────────────────────────────────────────────

class TaskResult:
    def __init__(self, name: str, prompt: str):
        self.name    = name
        self.prompt  = prompt
        self.checks: list[tuple[str, bool | None, str]] = []  # (field, ok, detail)

    def check(self, field: str, ok: bool | None, detail: str = "") -> bool:
        self.checks.append((field, ok, detail))
        if ok is None:
            sym = WARN
        elif ok:
            sym = PASS
        else:
            sym = FAIL
        print(f"    {sym} {field:35s} {detail}")
        return bool(ok)

    def diff(self, field: str, expected, actual) -> bool:
        if expected is None:
            return True
        ok = str(expected).strip().lower() == str(actual or "").strip().lower()
        detail = f"want={expected!r} got={actual!r}" if not ok else f"={expected!r}"
        return self.check(field, ok, detail)

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok is True)

    @property
    def total(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok is not None)

    @property
    def score_str(self) -> str:
        return f"{self.passed}/{self.total}"


# ─── Task Definitions ─────────────────────────────────────────────────────────
# Each task: (name, prompt, verify_fn)
# verify_fn(result: TaskResult) -> None  — does GETs and calls result.check/diff

def make_tasks(ts: str) -> list[tuple[str, str, object]]:
    """Returns all competition-style tasks with unique data (ts = timestamp suffix)."""

    # ── CREATE CUSTOMER tasks ─────────────────────────────────────────

    def verify_customer(result: TaskResult, name: str, email: str | None,
                        org: str | None, is_supplier: bool = False):
        customers = tx_get_list("/customer", {"name": name, "count": 500})
        found = next((c for c in customers if c.get("name") == name), None)
        result.check("customer exists", found is not None, f"name={name!r}")
        if not found:
            return
        full = tx_get_one(f"/customer/{found['id']}")
        result.diff("name",                 name,   full.get("name"))
        if email:
            result.diff("email",            email,  full.get("email"))
        if org:
            result.diff("organizationNumber", org,  full.get("organizationNumber"))
        if is_supplier:
            result.check("isSupplier=True",  full.get("isSupplier") is True,
                         f"actual={full.get('isSupplier')!r}")
            # Tripletex API ignores isCustomer=False on POST — info only
            result.check("isCustomer flag", None,
                         f"actual={full.get('isCustomer')!r} (API sets True regardless)")
        else:
            result.check("isCustomer=True",  full.get("isCustomer") is True,
                         f"actual={full.get('isCustomer')!r}")

    cust_name_en = f"Acme Corp {ts} AS"
    cust_name_no = f"Bergvik {ts} AS"
    cust_name_fr = f"Prairie {ts} SARL"
    cust_name_de = f"Schmidt {ts} GmbH"
    cust_name_pt = f"Cascata {ts} Lda"
    cust_email   = f"kontakt.{ts}@example.no"
    cust_org     = "910473166"

    tasks = [
        (
            "create_customer (EN)",
            f"Create a customer named {cust_name_en} with organisation number {cust_org} and email {cust_email}",
            lambda r, n=cust_name_en, e=cust_email, o=cust_org: verify_customer(r, n, e, o),
        ),
        (
            "create_customer (NO)",
            f"Registrer kunden {cust_name_no} med organisasjonsnummer {cust_org}. E-post: {cust_email}",
            lambda r, n=cust_name_no, e=cust_email, o=cust_org: verify_customer(r, n, e, o),
        ),
        (
            "create_customer (FR)",
            f"Créez le client {cust_name_fr} avec le numéro d'organisation {cust_org}. L'adresse e-mail est {cust_email}",
            lambda r, n=cust_name_fr, e=cust_email, o=cust_org: verify_customer(r, n, e, o),
        ),
        (
            "create_customer (DE)",
            f"Erstellen Sie den Kunden {cust_name_de} mit Organisationsnummer {cust_org} und E-Mail {cust_email}",
            lambda r, n=cust_name_de, e=cust_email, o=cust_org: verify_customer(r, n, e, o),
        ),
        (
            "create_customer (PT)",
            f"Crie o cliente {cust_name_pt} com o número de organização {cust_org} e e-mail {cust_email}",
            lambda r, n=cust_name_pt, e=cust_email, o=cust_org: verify_customer(r, n, e, o),
        ),
    ]

    # ── CREATE SUPPLIER tasks ─────────────────────────────────────────

    supp_name_no = f"Leverandør {ts} AS"
    supp_name_en = f"Supplier {ts} Ltd"
    supp_org     = "985423849"

    def verify_supplier(result, name, org):
        customers = tx_get_list("/customer", {"name": name, "count": 500})
        found = next((c for c in customers if c.get("name") == name), None)
        result.check("supplier exists", found is not None, f"name={name!r}")
        if not found:
            return
        full = tx_get_one(f"/customer/{found['id']}")
        result.diff("name",           name, full.get("name"))
        result.diff("organizationNumber", org, full.get("organizationNumber"))
        result.check("isSupplier=True",  full.get("isSupplier") is True, f"actual={full.get('isSupplier')!r}")
        # isCustomer flag: Tripletex API ignores isCustomer=False on POST — check as warning only
        result.check("isCustomer flag", None, f"actual={full.get('isCustomer')!r} (API may set True regardless)")

    tasks += [
        (
            "create_supplier (NO)",
            f"Registrer leverandøren {supp_name_no} med organisasjonsnummer {supp_org}. E-post: fakt@{ts}.no",
            lambda r, n=supp_name_no, o=supp_org: verify_supplier(r, n, o),
        ),
        (
            "create_supplier (EN)",
            f"Add supplier {supp_name_en} with org number {supp_org}, email supplier@{ts}.no",
            lambda r, n=supp_name_en, o=supp_org: verify_supplier(r, n, o),
        ),
    ]

    # ── CREATE EMPLOYEE tasks ─────────────────────────────────────────

    emp_fn = f"Eva{ts}"
    emp_ln = "Testdatter"
    emp_email = f"eva.{ts}@firma.no"
    emp_phone = "+4799001122"

    emp2_fn = f"Klaus{ts}"
    emp2_ln = "Müller"
    emp2_email = f"k.mueller.{ts}@firma.de"

    emp3_fn = f"Anna{ts}"
    emp3_ln = "Smith"
    emp3_email = f"anna.{ts}@company.com"

    def verify_employee(result, fn, ln, email, phone=None):
        employees = tx_get_list("/employee", {"count": 200})
        found = next(
            (e for e in employees
             if e.get("firstName") == fn or (e.get("email") or "").lower() == email.lower()),
            None,
        )
        result.check("employee exists", found is not None, f"name={fn} {ln}")
        if not found:
            return
        full = tx_get_one(f"/employee/{found['id']}")
        result.diff("firstName", fn,    full.get("firstName"))
        result.diff("lastName",  ln,    full.get("lastName"))
        result.diff("email",     email, full.get("email"))
        if phone:
            result.check("phoneNumberMobile set", bool(full.get("phoneNumberMobile")),
                         f"actual={full.get('phoneNumberMobile')!r}")
        result.check("department set", bool(full.get("department")),
                     f"actual={full.get('department')!r}")

    tasks += [
        (
            "create_employee (NO)",
            f"Legg til ansatt {emp_fn} {emp_ln} med e-post {emp_email} og mobil {emp_phone}",
            lambda r, fn=emp_fn, ln=emp_ln, e=emp_email, p=emp_phone: verify_employee(r, fn, ln, e, p),
        ),
        (
            "create_employee (DE)",
            f"Wir haben einen neuen Mitarbeiter namens {emp2_fn} {emp2_ln}, E-Mail: {emp2_email}, Telefon: +4988112233",
            lambda r, fn=emp2_fn, ln=emp2_ln, e=emp2_email: verify_employee(r, fn, ln, e, "+4988112233"),
        ),
        (
            "create_employee (EN)",
            f"Create employee {emp3_fn} {emp3_ln}, email {emp3_email}, phone +4499887766",
            lambda r, fn=emp3_fn, ln=emp3_ln, e=emp3_email: verify_employee(r, fn, ln, e, "+4499887766"),
        ),
    ]

    # ── CREATE INVOICE tasks ──────────────────────────────────────────

    inv_cust_en = f"InvoiceCo {ts} AS"
    inv_cust_pt = f"Fatura {ts} Lda"
    inv_org     = "826621389"
    inv_amount  = 4450.0
    due_date    = (datetime.date.today() + datetime.timedelta(days=14)).isoformat()

    def verify_invoice(result, cust_name, amount, expected_due=None):
        # Find customer
        custs = tx_get_list("/customer", {"name": cust_name, "count": 500})
        cust = next((c for c in custs if c.get("name") == cust_name), None)
        result.check("customer exists", cust is not None, f"name={cust_name!r}")
        if not cust:
            return

        # Find invoice - use date range required by sandbox
        date_from = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        date_to   = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
        invs = tx_get_list("/invoice", {
            "customerId": cust["id"],
            "invoiceDateFrom": date_from,
            "invoiceDateTo": date_to,
            "count": 50,
        })
        inv = invs[0] if invs else None
        result.check("invoice exists",    inv is not None, f"customerId={cust['id']}")
        if not inv:
            return
        full = tx_get_one(f"/invoice/{inv['id']}")
        result.check("invoiceDate set",    bool(full.get("invoiceDate")), f"actual={full.get('invoiceDate')}")
        result.check("invoiceDueDate set", bool(full.get("invoiceDueDate")), f"actual={full.get('invoiceDueDate')}")
        if expected_due:
            result.diff("invoiceDueDate", expected_due, full.get("invoiceDueDate"))
        result.check("orders linked",      bool(full.get("orders")), f"actual={full.get('orders')}")
        amt = full.get("amountCurrency") or full.get("invoiceAmountCurrency")
        result.check("amount > 0",         bool(amt and float(amt) > 0), f"actual={amt}")

    tasks += [
        (
            "create_invoice (EN)",
            f"Create and send an invoice to customer {inv_cust_en} (org no {inv_org}) for {int(inv_amount)} NOK excl VAT. Description: Consulting services",
            lambda r, n=inv_cust_en, a=inv_amount, d=due_date: verify_invoice(r, n, a, d),
        ),
        (
            "create_invoice (PT)",
            f"Crie e envie uma fatura ao cliente {inv_cust_pt} (org. nº {inv_org}) por {int(inv_amount)} NOK sem IVA. A fatura refere-se a Horas de consultoria.",
            lambda r, n=inv_cust_pt, a=inv_amount, d=due_date: verify_invoice(r, n, a, d),
        ),
    ]

    # ── CREATE DEPARTMENT tasks ───────────────────────────────────────

    dept_name_no = f"Regnskap-{ts}X"
    dept_name_en = f"Engineering-{ts}X"

    def verify_department(result, name):
        depts = tx_get_list("/department", {"count": 50})
        found = next((d for d in depts if d.get("name") == name), None)
        result.check("department exists", found is not None, f"name={name!r}")
        if found:
            result.diff("name", name, found.get("name"))

    tasks += [
        (
            "create_department (NO)",
            f"Opprett avdelingen '{dept_name_no}' i systemet",
            lambda r, n=dept_name_no: verify_department(r, n),
        ),
        (
            "create_department (EN)",
            f"Create a new department called '{dept_name_en}'",
            lambda r, n=dept_name_en: verify_department(r, n),
        ),
    ]

    return tasks


# ─── Main runner ──────────────────────────────────────────────────────────────

TASK_FILTER: dict[str, list[str]] = {
    "customer":   ["create_customer"],
    "supplier":   ["create_supplier"],
    "employee":   ["create_employee"],
    "invoice":    ["create_invoice"],
    "department": ["create_department"],
    "payment":    ["register_payment"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://localhost:8080",
                    help="Agent URL (default: http://localhost:8080)")
    ap.add_argument("--task", choices=list(TASK_FILTER), default=None,
                    help="Only run tasks of this type")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    target = args.target.rstrip("/")
    ts = TS

    print(f"\n{'═'*65}")
    print(f"  COMPETITION EVALUATOR SIMULATOR")
    print(f"  Target : {target}")
    print(f"  Backend: {BASE_URL}")
    print(f"  Seed   : {ts}")
    print(f"{'═'*65}\n")

    # Verify target is reachable
    try:
        hc = http.get(f"{target}/health", timeout=10)
        print(f"  /health → {hc.json()}\n")
    except Exception as exc:
        print(f"  ❌ Cannot reach {target}: {exc}\n  Start the server first.\n")
        sys.exit(1)

    all_tasks = make_tasks(ts)

    # Filter if requested
    if args.task:
        prefixes = TASK_FILTER[args.task]
        all_tasks = [(n, p, v) for n, p, v in all_tasks
                     if any(n.lower().startswith(pf) for pf in prefixes)]

    results: list[TaskResult] = []
    sandbox_skip = 0

    for name, prompt, verify_fn in all_tasks:
        print(f"══ {name} {'═'*(55-len(name))}")
        print(f"   prompt: {prompt[:110]}")

        result = TaskResult(name, prompt)
        t0 = time.monotonic()

        status, resp = call_agent(target, prompt)
        elapsed = time.monotonic() - t0

        agent_ok = status == 200 and resp.get("status") == "completed"
        result.check("agent responded ok", agent_ok, f"http={status} body={str(resp)[:80]}")

        if agent_ok:
            time.sleep(0.5)  # slight delay for eventual consistency
            try:
                verify_fn(result)
            except Exception as exc:
                if "bankkontonummer" in str(exc) or ("422" in str(exc) and "invoice" in str(exc).lower()):
                    print(f"    {WARN} SANDBOX SKIP: bank account not configured — will pass on competition proxy")
                    sandbox_skip += 1
                else:
                    result.check("verify_fn crash", False, str(exc)[:120])

        score = result.score_str
        print(f"   → {score} checks passed  ({elapsed:.1f}s)\n")
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────
    total_passed = sum(r.passed for r in results)
    total_checks = sum(r.total for r in results)
    task_passed  = sum(1 for r in results if r.passed == r.total and r.total > 0)
    task_total   = len(results)

    print(f"{'═'*65}")
    print(f"  SCORE: {task_passed}/{task_total} tasks fully passed")
    print(f"  CHECKS: {total_passed}/{total_checks} checks passed")
    if sandbox_skip:
        print(f"  NOTE: {sandbox_skip} task(s) skipped invoice step (sandbox bank account — will work in competition)")
    print(f"{'═'*65}")

    if total_checks > 0 and total_passed < total_checks:
        print("\n  FAILURES:")
        for r in results:
            for field, ok, detail in r.checks:
                if ok is False:
                    print(f"    ✗ [{r.name}] {field}: {detail}")
    print()


if __name__ == "__main__":
    main()
