#!/usr/bin/env python3
"""Quick live smoke test against the deployed Cloud Run endpoint."""

import json
import os
import subprocess
import sys

BASE = "https://ledger-logic-agent-xrgiacpg2q-ew.a.run.app/"
TOKEN = os.environ.get("TRIPLETEX_SANDBOX_TOKEN", "")
URL   = os.environ.get("TRIPLETEX_SANDBOX_URL", "")

if not TOKEN or not URL:
    sys.exit("Set TRIPLETEX_SANDBOX_TOKEN and TRIPLETEX_SANDBOX_URL first.")

TESTS = [
    ("create_customer (EN)",  "Create a new customer called LiveSmoke Corp with email live@smoke.no"),
    ("create_employee (NB)",  "Opprett ansatt Erik Smoketest, epost erik@smoke.no, telefon 90000001"),
    ("create_department (ES)","Crea un departamento llamado 'Smoke-99'"),
    ("create_invoice (NB)",   "Opprett faktura til LiveSmoke Corp på 7500 kr for rådgivning"),
    ("register_payment (EN)", "Register payment for invoice from LiveSmoke Corp, amount 7500"),
]

passed = 0
failed = 0
for label, prompt in TESTS:
    payload = json.dumps({
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {"base_url": URL, "session_token": TOKEN},
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", BASE,
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=90,
        )
        resp = json.loads(result.stdout)
        status = resp.get("status", "?")
    except Exception as exc:
        status = f"ERROR({exc})"

    ok = status == "completed"
    passed += ok
    failed += not ok
    icon = "✓" if ok else "✗"
    print(f"  {icon} [{label}] → {status}")

print(f"\n  {passed}/{passed+failed} live tests passed")
