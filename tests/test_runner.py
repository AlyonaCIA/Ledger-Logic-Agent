"""
Local evaluation harness for the Ledger Logic Agent.

Usage:
    python tests/test_runner.py --url http://localhost:8000 [--api-key KEY]
    python tests/test_runner.py --unit          # parser/validator tests only (no server)

Output: per-task-type summary with success rate, avg calls, avg 4xx, avg latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── ensure repo root is on path when run directly ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# load .env so GCP_PROJECT_ID / TRIPLETEX_* are available when running locally
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

# ======================================================================
# Synthetic multilingual test dataset
# 7 task types × 5+ prompts per type covering all 7 languages
# ======================================================================

CASES: list[dict] = [
    # ── create_employee ───────────────────────────────────────────────
    {
        "task_type": "create_employee",
        "prompt": "Opprett en ansatt med navn Kari Nordmann, epost kari@example.no. Hun skal være kontoadministrator.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Create an employee named John Smith, email john.smith@company.com. He should be an account administrator.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Crea un empleado llamado María García, correo maria@empresa.es. Debe ser administradora de cuenta.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Criar um funcionário com nome Pedro Souza, email pedro@empresa.com.br.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Erstelle einen Mitarbeiter mit dem Namen Hans Müller, E-Mail hans@firma.de. Er soll Kontoadministrator sein.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Créer un employé nommé Marie Dupont, email marie@societe.fr. Elle doit être administratrice de compte.",
    },
    {
        "task_type": "create_employee",
        "prompt": "Opprett ein tilsett med namn Lars Olsen, epost lars@bedrift.no.",
    },
    # ── create_customer ───────────────────────────────────────────────
    {
        "task_type": "create_customer",
        "prompt": "Opprett en kunde med navn Acme AS, epost post@acme.no.",
    },
    {
        "task_type": "create_customer",
        "prompt": "Create a customer named Beta Corp, email contact@betacorp.com, phone +4412345678.",
    },
    {
        "task_type": "create_customer",
        "prompt": "Registrar un cliente llamado Gamma SL, correo info@gamma.es, teléfono +34612345678.",
    },
    {
        "task_type": "create_customer",
        "prompt": "Erstelle einen Kunden mit dem Namen Delta GmbH, E-Mail info@delta.de.",
    },
    {
        "task_type": "create_customer",
        "prompt": "Créer un client nommé Epsilon SARL, email contact@epsilon.fr.",
    },
    # ── create_product ────────────────────────────────────────────────
    {
        "task_type": "create_product",
        "prompt": "Opprett et produkt med navn Konsulenttime, pris 1200 kr ekskl. mva.",
    },
    {
        "task_type": "create_product",
        "prompt": "Create a product called 'Software License', price 500 NOK excluding VAT.",
    },
    {
        "task_type": "create_product",
        "prompt": "Cria um produto chamado 'Hora de Consultoria', preço 800 NOK sem IVA.",
    },
    {
        "task_type": "create_product",
        "prompt": "Créer un produit nommé 'Heure de conseil', prix 950 NOK hors TVA.",
    },
    # ── create_department ────────────────────────────────────────────
    {
        "task_type": "create_department",
        "prompt": "Opprett en avdeling med navn Salgsavdelingen.",
    },
    {
        "task_type": "create_department",
        "prompt": "Create a department called Marketing with department number 100.",
    },
    {
        "task_type": "create_department",
        "prompt": "Crea un departamento llamado Recursos Humanos, número 200.",
    },
    {
        "task_type": "create_department",
        "prompt": "Créer un département nommé Finance, numéro de département 300.",
    },
    # ── create_project ───────────────────────────────────────────────
    {
        "task_type": "create_project",
        "prompt": "Opprett et prosjekt kalt Digitaliseringsprosjektet for kunden Acme AS. Startdato 2026-04-01.",
    },
    {
        "task_type": "create_project",
        "prompt": "Create a project named 'Cloud Migration' for customer Beta Corp, starting 2026-05-01.",
    },
    {
        "task_type": "create_project",
        "prompt": "Criar um projeto chamado 'Transformação Digital' para o cliente Gamma Brasil, início 2026-06-01.",
    },
    # ── create_invoice ────────────────────────────────────────────────
    {
        "task_type": "create_invoice",
        "prompt": "Opprett en faktura til kunden Acme AS på 5000 kr ekskl. mva. Forfallsdato om 30 dager.",
    },
    {
        "task_type": "create_invoice",
        "prompt": "Create an invoice for customer Beta Corp. Line item: Consulting 10 hours at 1200 NOK/hour.",
    },
    {
        "task_type": "create_invoice",
        "prompt": "Créer une facture pour le client Epsilon SARL, montant 3 500 EUR.",
    },
    # ── update_employee ───────────────────────────────────────────────
    {
        "task_type": "update_employee",
        "prompt": "Oppdater telefonnummer for ansatt Kari Nordmann til +4798765432.",
    },
    {
        "task_type": "update_employee",
        "prompt": "Update the email address of employee John Smith to j.smith@newcompany.com.",
    },
]


# ======================================================================
# Keyword-based mock parser (no OpenAI needed – tests executor routing)
# ======================================================================

def _mock_parse(prompt: str, expected_task_type: str) -> dict:
    """
    Produce a minimal intent dict based on the expected task type.
    Extracts simple fields from the prompt via regex so the executor
    actually does something meaningful (not just a no-op).
    """
    import re

    p = prompt.lower()

    # -- name extraction (first quoted or CamelCase word cluster after 'named|kalt|nommé|llamado|nomeado') --
    _name = None
    m = re.search(r"(?:named?|kalt|nommé|llamado|nomeado|korrekt|kalla)\s+['\"]?([\w\s\-\.]+?)['\"]?(?:\s*,|\s+med|\s+for|\s+til|\s+email|$)", prompt, re.I)
    if m:
        _name = m.group(1).strip()

    # -- email --
    _email = None
    em = re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", prompt, re.I)
    if em:
        _email = em.group(0)

    # -- phone --
    _phone = None
    ph = re.search(r"(\+?[\d\s\-]{7,15})", prompt)
    if ph:
        _phone = ph.group(1).strip()

    # -- price / amount --
    _amount = None
    am = re.search(r"([\d\s,.]+)\s*(?:kr|nok|eur|usd|brl)", p)
    if am:
        try:
            _amount = float(am.group(1).replace(" ", "").replace(",", "."))
        except ValueError:
            pass

    # -- department number --
    _dept_num = None
    dn = re.search(r"(?:number|nummer|numéro|número)\s+(\d+)", p)
    if dn:
        _dept_num = dn.group(1)

    # Build intent based on task type
    tt = expected_task_type
    base = {"task_type": tt, "confidence": 0.95, "language_detected": "en", "missing_fields": []}

    if tt in ("create_employee", "update_employee", "delete_employee"):
        first, *rest = (_name or "Test User").split()
        base["employee"] = {
            "first_name": first,
            "last_name": " ".join(rest) if rest else "Employee",
            "email": _email,
            "is_account_admin": "admin" in p or "administrator" in p,
            "identifier": _name,
        }
    elif tt in ("create_customer", "update_customer", "delete_customer"):
        base["customer"] = {
            "name": _name or "Test Customer",
            "email": _email,
            "phone": _phone,
            "identifier": _name,
        }
    elif tt == "create_product":
        base["product"] = {
            "name": _name or "Test Product",
            "price_excl_vat": _amount,
        }
    elif tt == "create_department":
        base["department"] = {
            "name": _name or "Test Department",
            "number": _dept_num,
        }
    elif tt == "create_project":
        base["project"] = {
            "name": _name or "Test Project",
        }
        base["customer"] = {"name": "Test Customer"}
    elif tt == "create_invoice":
        base["customer"] = {"name": _name or "Test Customer"}
        base["invoice"] = {
            "amount": _amount or 1000.0,
            "description": "Service",
        }
    elif tt in ("register_payment", "create_credit_note"):
        base["invoice"] = {"identifier": "1001"}
        base["customer"] = {"name": _name or "Test Customer"}
    elif tt == "create_travel_expense":
        base["travel_expense"] = {"description": "Travel"}
    elif tt == "create_department":
        base["department"] = {"name": _name or "Department"}
    return base


# ======================================================================
# Minimal mock Tripletex client (for unit mode, no server needed)
# ======================================================================

class _MockClient:
    """Returns plausible fake responses to count calls without hitting an API."""

    def __init__(self) -> None:
        self.call_count = 0
        self.error_4xx = 0
        self.error_5xx = 0
        self.latency_ms_total = 0.0
        self.calls_by_endpoint: dict[str, int] = {}

    @property
    def error_count(self) -> int:
        return self.error_4xx + self.error_5xx

    def metrics_summary(self) -> dict:
        return {
            "call_count": self.call_count,
            "error_4xx": self.error_4xx,
            "error_5xx": self.error_5xx,
            "latency_ms_total": self.latency_ms_total,
            "calls_by_endpoint": self.calls_by_endpoint,
        }

    def _fake(self, method: str, path: str) -> dict:
        self.call_count += 1
        key = f"{method}:{path}"
        self.calls_by_endpoint[key] = self.calls_by_endpoint.get(key, 0) + 1
        return {"value": {"id": 999, "invoiceNumber": 1001}}

    def get(self, path: str, params: dict | None = None) -> dict:
        return {"values": [self._fake("GET", path)["value"]]}

    def get_list(self, path: str, params: dict | None = None) -> list:
        return []  # simulates empty sandbox → triggers create path

    def get_value(self, path: str, params: dict | None = None) -> dict:
        return self._fake("GET", path)["value"]

    def post(self, path: str, json: Any = None) -> dict:
        return self._fake("POST", path)

    def post_value(self, path: str, json: Any = None) -> dict:
        return self._fake("POST", path)["value"]

    def put(self, path: str, json: Any = None) -> dict:
        return self._fake("PUT", path)

    def put_value(self, path: str, json: Any = None) -> dict:
        return self._fake("PUT", path)["value"]

    def delete(self, path: str) -> None:
        self._fake("DELETE", path)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class CaseResult:
    prompt: str
    expected_task_type: str
    task_type_detected: str = "unknown"
    confidence: float = 0.0
    language: str = "?"
    missing_fields: list = field(default_factory=list)
    api_calls: int = 0
    error_4xx: int = 0
    error_5xx: int = 0
    latency_ms: float = 0.0
    success: bool = False
    error_msg: str = ""


# ======================================================================
# Unit runner  (parser only, no server)
# ======================================================================

async def run_unit(cases: list[dict], mock_parser: bool = False) -> list[CaseResult]:
    from agent.executor import execute_task

    if not mock_parser:
        from agent.parser import parse_task

    results: list[CaseResult] = []
    for case in cases:
        r = CaseResult(prompt=case["prompt"], expected_task_type=case["task_type"])
        t0 = time.monotonic()
        try:
            if mock_parser:
                intent = _mock_parse(case["prompt"], case["task_type"])
            else:
                intent = await parse_task(case["prompt"], files=[])
            r.task_type_detected = intent.get("task_type", "unknown")
            r.confidence = intent.get("confidence", 0.0)
            r.language = intent.get("language_detected", "?")
            r.missing_fields = intent.get("missing_fields") or []

            mock = _MockClient()
            execute_task(intent, mock)  # type: ignore[arg-type]
            r.api_calls = mock.call_count
            r.error_4xx = mock.error_4xx
            r.success = r.task_type_detected == r.expected_task_type
        except Exception as exc:
            r.error_msg = str(exc)
        r.latency_ms = (time.monotonic() - t0) * 1000
        results.append(r)
        _print_case(r)

    return results


# ======================================================================
# E2E runner (against real /solve endpoint)
# ======================================================================

async def run_e2e(
    cases: list[dict],
    url: str,
    api_key: str | None,
    base_url: str,
    session_token: str,
) -> list[CaseResult]:
    import httpx

    headers: dict = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: list[CaseResult] = []
    async with httpx.AsyncClient(timeout=310) as client:
        for case in cases:
            r = CaseResult(prompt=case["prompt"], expected_task_type=case["task_type"])
            payload = {
                "prompt": case["prompt"],
                "files": [],
                "tripletex_credentials": {
                    "base_url": base_url,
                    "session_token": session_token,
                },
            }
            t0 = time.monotonic()
            try:
                resp = await client.post(f"{url}/solve", json=payload, headers=headers)
                r.latency_ms = (time.monotonic() - t0) * 1000
                if resp.status_code == 200 and resp.json().get("status") == "completed":
                    r.success = True
                else:
                    r.error_msg = f"HTTP {resp.status_code}"
            except Exception as exc:
                r.latency_ms = (time.monotonic() - t0) * 1000
                r.error_msg = str(exc)
            results.append(r)
            _print_case(r)

    return results


# ======================================================================
# Reporting
# ======================================================================

def _print_case(r: CaseResult) -> None:
    status = "✓" if r.success else "✗"
    print(
        f"  {status} [{r.expected_task_type}] "
        f"detected={r.task_type_detected} conf={r.confidence:.2f} "
        f"lang={r.language} calls={r.api_calls} "
        f"4xx={r.error_4xx} {r.latency_ms:.0f}ms"
        + (f" ERR={r.error_msg}" if r.error_msg else "")
    )


def print_summary(results: list[CaseResult]) -> None:
    from collections import defaultdict

    by_type: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_type[r.expected_task_type].append(r)

    print("\n" + "=" * 70)
    print(f"{'TASK TYPE':<30} {'N':>3} {'OK%':>5} {'calls':>6} {'4xx':>5} {'ms':>7}")
    print("-" * 70)
    for tt, rs in sorted(by_type.items()):
        n = len(rs)
        ok = sum(1 for r in rs if r.success)
        calls = [r.api_calls for r in rs if r.api_calls]
        errs = [r.error_4xx for r in rs]
        lats = [r.latency_ms for r in rs]
        print(
            f"  {tt:<28} {n:>3} {ok/n*100:>4.0f}% "
            f"{statistics.mean(calls) if calls else 0:>6.1f} "
            f"{statistics.mean(errs):>5.1f} "
            f"{statistics.mean(lats):>7.0f}"
        )
    print("-" * 70)
    total = len(results)
    ok_total = sum(1 for r in results if r.success)
    all_lats = [r.latency_ms for r in results]
    print(
        f"  {'TOTAL':<28} {total:>3} {ok_total/total*100:>4.0f}% "
        f"{'':>6} {'':>5} {statistics.mean(all_lats):>7.0f}"
    )
    print("=" * 70)


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ledger Logic Agent – local eval runner")
    p.add_argument(
        "--unit",
        action="store_true",
        help="Real GPT-4o parser + mock-executor (needs OPENAI_API_KEY)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Keyword mock parser + mock-executor (no API key needed — tests executor routing)",
    )
    p.add_argument("--url", default="http://localhost:8000", help="Agent base URL")
    p.add_argument("--api-key", default=None, help="Bearer API key for /solve")
    p.add_argument("--base-url", default="https://tx-proxy.ainm.no/v2", help="Tripletex proxy URL")
    p.add_argument("--session-token", default="test-token", help="Tripletex session token")
    p.add_argument(
        "--task-type",
        default=None,
        help="Filter to a single task type (e.g. create_employee)",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    cases = CASES
    if args.task_type:
        cases = [c for c in CASES if c["task_type"] == args.task_type]
        if not cases:
            print(f"No cases found for task_type={args.task_type!r}")
            sys.exit(1)

    mode = "(mock parser — executor routing only)" if args.mock else "(unit mode — real Gemini parser)" if args.unit else f"against {args.url}"
    print(f"\nRunning {len(cases)} test cases {mode}\n")

    if args.unit or args.mock:
        results = await run_unit(cases, mock_parser=args.mock)
    else:
        results = await run_e2e(
            cases,
            url=args.url,
            api_key=args.api_key,
            base_url=args.base_url,
            session_token=args.session_token,
        )

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
