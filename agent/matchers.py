"""
Central entity resolver with a stable, layered matching policy.

All entity lookups in the agent flow through this module instead of being
scattered across executor.py. This ensures consistent behaviour, a single
place to fix bugs, and a clear failure taxonomy.

Matching policy (applied in order):
  1. Exact normalized   – lowercased + stripped comparison on `name`
  2. Stable-key match   – email, org_number, employee ID (when provided)
  3. Contains fallback  – substring match, only if policy `allow_fuzzy=True`
  4. None               – caller decides whether to create or abort

Every resolver returns either the matched Tripletex object dict or None.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from agent.client import TripletexClient

logger = logging.getLogger(__name__)

# ── Failure taxonomy ──────────────────────────────────────────────────────────
# Kept here so critic.py and shadow_score.py can import it.
KNOWN_FAILURES: dict[str, dict[str, str]] = {
    "wrong_customer_match": {
        "symptom": "created/updated wrong customer; invoice linked to unrelated contact",
        "fix": "use exact normalized name before falling back to substring",
        "workflow_patch": "resolve_customer → prefer exact match, abort if ambiguous",
    },
    "department_number_split": {
        "symptom": "department name trailing digits interpreted as department_number",
        "fix": "treat trailing digits as part of name unless preceded by explicit keyword",
        "workflow_patch": "parser instruction: department_number only on explicit keyword",
    },
    "invoice_lookup_needs_dates": {
        "symptom": "GET /invoice returns 422 or empty",
        "fix": "always include invoiceDateFrom/To spanning ±5 years",
        "workflow_patch": "_find_invoice_for_customer already applies this",
    },
    "customer_not_found_after_create": {
        "symptom": "customer created (201) but subsequent lookup returns None",
        "fix": "use the ID from the POST response instead of a second GET",
        "workflow_patch": "store created_id from api_trace and skip re-lookup",
    },
    "missing_bank_account": {
        "symptom": "POST /invoice/send returns 422",
        "fix": "sandbox has no bank account; works on competition proxy",
        "workflow_patch": "no action needed for competition",
    },
    "is_customer_always_true": {
        "symptom": "isCustomer=False not honoured by Tripletex API",
        "fix": "API ignores the field; cannot be forced to False",
        "workflow_patch": "remove isCustomer=False from supplier POST payload",
    },
    "fuzzy_name_picks_wrong_employee": {
        "symptom": "update/delete targets wrong employee due to partial name match",
        "fix": "exact full-name match first; prefer email when available",
        "workflow_patch": "resolve_employee: exact full name → email → fuzzy",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str | None) -> str:
    return (s or "").strip().lower()


def _exact_name(results: list[dict], name: str) -> dict | None:
    n = _normalize(name)
    for r in results:
        if _normalize(r.get("name")) == n:
            return r
    return None


def _exact_email(results: list[dict], email: str) -> dict | None:
    e = _normalize(email)
    for r in results:
        if _normalize(r.get("email")) == e:
            return r
    return None


def _contains_name(results: list[dict], name: str) -> dict | None:
    n = _normalize(name)
    for r in results:
        if n in _normalize(r.get("name")):
            return r
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public resolvers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_customer(
    client: TripletexClient,
    name: str | None,
    email: str | None = None,
    org_number: str | None = None,
    allow_fuzzy: bool = False,
) -> dict | None:
    """
    Resolve a customer by name (primary) with email / org_number as stable-key
    fallbacks.  Never returns a fuzzy match unless allow_fuzzy is explicitly
    set to True.
    """
    if not name and not email and not org_number:
        return None

    results = client.get_list(
        "/customer",
        params={"name": name or "", "count": 500, "fields": "id,name,email,organizationNumber"},
    )

    # 1. Exact normalized name
    if name:
        hit = _exact_name(results, name)
        if hit:
            logger.debug(f"resolve_customer: exact name match id={hit['id']}")
            return hit

    # 2. Stable-key: email
    if email and results:
        hit = _exact_email(results, email)
        if hit:
            logger.debug(f"resolve_customer: email match id={hit['id']}")
            return hit

    # 3. Stable-key: org_number
    if org_number and results:
        on = _normalize(org_number)
        for r in results:
            if _normalize(r.get("organizationNumber")) == on:
                logger.debug(f"resolve_customer: org_number match id={r['id']}")
                return r

    # 4. Contains fallback (opt-in)
    if allow_fuzzy and name and results:
        hit = _contains_name(results, name)
        if hit:
            logger.debug(f"resolve_customer: fuzzy match id={hit['id']}")
            return hit

    return None


def resolve_employee(
    client: TripletexClient,
    name: str | None = None,
    email: str | None = None,
    allow_fuzzy: bool = True,
) -> dict | None:
    """
    Resolve an employee by full name or email.
    Policy: exact full-name → exact email → substring full-name (if allow_fuzzy).
    """
    if not name and not email:
        return None

    employees = client.get_list(
        "/employee",
        params={"fields": "id,firstName,lastName,email", "count": 500},
    )

    def _full_name(e: dict) -> str:
        return f"{e.get('firstName', '')} {e.get('lastName', '')}".strip()

    # 1. Exact full name
    if name:
        n = _normalize(name)
        for e in employees:
            if _normalize(_full_name(e)) == n:
                return e

    # 2. Exact email
    if email:
        em = _normalize(email)
        for e in employees:
            if _normalize(e.get("email")) == em:
                return e

    # 3. Contains / partial name (allow_fuzzy)
    if allow_fuzzy and name:
        n = _normalize(name)
        for e in employees:
            if n in _normalize(_full_name(e)) or _normalize(_full_name(e)) in n:
                return e

    return None


def resolve_department(
    client: TripletexClient,
    name: str | None,
) -> dict | None:
    """Resolve a department by exact name."""
    if not name:
        return None
    depts = client.get_list("/department", params={"count": 500, "fields": "id,name"})
    n = _normalize(name)
    for d in depts:
        if _normalize(d.get("name")) == n:
            return d
    return None


def resolve_invoice(
    client: TripletexClient,
    customer_id: int | None = None,
    invoice_number: str | None = None,
) -> dict | None:
    """
    Resolve an invoice by customer_id or invoice number.
    Always includes a wide date range to avoid sandbox 422s.
    """
    date_from = (date.today() - timedelta(days=365 * 5)).isoformat()
    date_to = (date.today() + timedelta(days=365)).isoformat()

    params: dict[str, Any] = {
        "invoiceDateFrom": date_from,
        "invoiceDateTo": date_to,
        "count": 100,
        "fields": "id,invoiceNumber,amount,amountOutstanding,amountCurrency,customer,isCredited,isCreditNote",
    }
    if customer_id:
        params["customerId"] = customer_id

    try:
        invoices = client.get_list("/invoice", params=params)
    except Exception:
        invoices = []

    if not invoices and customer_id:
        # Fallback: fetch all and filter client-side
        try:
            all_inv = client.get_list(
                "/invoice",
                params={"invoiceDateFrom": date_from, "invoiceDateTo": date_to, "count": 200},
            )
            invoices = [
                i for i in all_inv
                if (i.get("customer") or {}).get("id") == customer_id
            ]
        except Exception:
            invoices = []

    if not invoices:
        return None

    # Filter by invoice number if given
    if invoice_number:
        inv_no = str(invoice_number).strip()
        for i in invoices:
            if str(i.get("invoiceNumber", "")) == inv_no:
                return i

    # Prefer not-yet-credited invoices with outstanding balance
    not_credited = [i for i in invoices if not i.get("isCredited") and not i.get("isCreditNote")]
    unpaid = [i for i in not_credited if (i.get("amountOutstanding") or 0) > 0]
    if unpaid:
        return unpaid[0]
    if not_credited:
        return not_credited[0]
    # Fallback: prefer unpaid regardless
    unpaid_any = [i for i in invoices if (i.get("amountOutstanding") or 0) > 0]
    return unpaid_any[0] if unpaid_any else invoices[0]
