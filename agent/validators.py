"""
Pre-flight validation and normalisation layer.

Every workflow calls validate_intent() before touching the API.
Rules:
  - Raise ValidationError only for truly unrecoverable missing data.
  - Normalise in-place so executors can trust field values.
  - Log warnings for fixable issues; fix them silently.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

logger = logging.getLogger(__name__)

# ── Regex helpers ──────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_FMT = "%Y-%m-%d"
_AMOUNT_STRIP = re.compile(r"[^\d.,\-]")  # keep digits, dot, comma, minus

# Currency symbols that sometimes leak into amounts
_CURRENCY_WORDS = {"kr", "nok", "eur", "usd", "gbp", "dkk", "sek", "chf"}


class ValidationError(ValueError):
    """Raised when an intent is too incomplete to execute safely."""


# ══════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════

def validate_intent(intent: dict) -> dict:
    """
    Normalise and validate all fields in the parsed intent dict.
    Returns the (possibly mutated) intent if OK.
    Raises ValidationError for unrecoverable problems.
    """
    task_type = intent.get("task_type", "unknown")

    # Per-task required field checks
    _REQUIRED: dict[str, list[tuple[str, str]]] = {
        "create_employee": [("employee", "first_name")],
        "create_customer": [("customer", "name")],
        "create_product": [("product", "name")],
        "create_invoice": [("customer", "name")],
        "register_payment": [],  # soft – we search at runtime
        "create_project": [("project", "name")],
        "create_department": [("department", "name")],
    }

    required = _REQUIRED.get(task_type, [])
    for (section, field) in required:
        section_val = intent.get(section)
        # department may be a list (multiple departments) or a single dict
        if isinstance(section_val, list):
            if not (section_val and section_val[0].get(field)):
                raise ValidationError(
                    f"task_type={task_type!r} requires {section}.{field}"
                )
        elif not (section_val or {}).get(field):
            raise ValidationError(
                f"task_type={task_type!r} requires {section}.{field}"
            )

    # Normalise each section
    _norm_employee(intent.get("employee"))
    _norm_customer(intent.get("customer"))
    _norm_product(intent.get("product"))
    _norm_invoice(intent.get("invoice"))
    _norm_payment(intent.get("payment"))
    _norm_travel_expense(intent.get("travel_expense"))
    _norm_project(intent.get("project"))

    return intent


# ══════════════════════════════════════════════════════════════════════════
# Section normalisers
# ══════════════════════════════════════════════════════════════════════════

def _norm_employee(emp: dict | None) -> None:
    if not emp:
        return
    if emp.get("email"):
        emp["email"] = _clean_email(emp["email"])
    if emp.get("first_name"):
        emp["first_name"] = emp["first_name"].strip()
    if emp.get("last_name"):
        emp["last_name"] = emp["last_name"].strip()
    if emp.get("phone"):
        emp["phone"] = _clean_phone(emp["phone"])
    if emp.get("start_date"):
        emp["start_date"] = _clean_date(emp["start_date"]) or emp["start_date"]


def _norm_customer(cust: dict | None) -> None:
    if not cust:
        return
    if cust.get("name"):
        cust["name"] = cust["name"].strip()
    if cust.get("email"):
        cust["email"] = _clean_email(cust["email"])
    if cust.get("phone"):
        cust["phone"] = _clean_phone(cust["phone"])
    if cust.get("org_number"):
        cust["org_number"] = re.sub(r"\s", "", cust["org_number"])


def _norm_product(prod: dict | None) -> None:
    if not prod:
        return
    if prod.get("name"):
        prod["name"] = prod["name"].strip()
    if prod.get("price_excl_vat") is not None:
        prod["price_excl_vat"] = _clean_amount(prod["price_excl_vat"])
    if prod.get("vat_rate") is not None:
        prod["vat_rate"] = _clean_vat_rate(prod["vat_rate"])


def _norm_invoice(inv: dict | None) -> None:
    if not inv:
        return
    if inv.get("date"):
        inv["date"] = _clean_date(inv["date"]) or inv["date"]
    if inv.get("amount") is not None:
        inv["amount"] = _clean_amount(inv["amount"])
    if inv.get("due_days") is not None:
        try:
            inv["due_days"] = int(inv["due_days"])
        except (ValueError, TypeError):
            inv["due_days"] = 14
    lines = inv.get("order_lines") or []
    for line in lines:
        if line.get("unit_price_excl_vat") is not None:
            line["unit_price_excl_vat"] = _clean_amount(line["unit_price_excl_vat"])
        if line.get("count") is not None:
            try:
                line["count"] = float(line["count"])
            except (ValueError, TypeError):
                line["count"] = 1.0


def _norm_payment(pay: dict | None) -> None:
    if not pay:
        return
    if pay.get("date"):
        pay["date"] = _clean_date(pay["date"]) or pay["date"]
    if pay.get("amount") is not None:
        pay["amount"] = _clean_amount(pay["amount"])


def _norm_travel_expense(te: dict | None) -> None:
    if not te:
        return
    if te.get("from_date"):
        te["from_date"] = _clean_date(te["from_date"]) or te["from_date"]
    if te.get("to_date"):
        te["to_date"] = _clean_date(te["to_date"]) or te["to_date"]
    if te.get("amount") is not None:
        te["amount"] = _clean_amount(te["amount"])


def _norm_project(proj: dict | None) -> None:
    if not proj:
        return
    if proj.get("name"):
        proj["name"] = proj["name"].strip()
    if proj.get("start_date"):
        proj["start_date"] = _clean_date(proj["start_date"]) or proj["start_date"]
    if proj.get("end_date"):
        proj["end_date"] = _clean_date(proj["end_date"]) or proj["end_date"]


# ══════════════════════════════════════════════════════════════════════════
# Atomic cleaners
# ══════════════════════════════════════════════════════════════════════════

def _clean_email(raw: str) -> str:
    """Lowercase, strip. Warn if format looks wrong but don't block."""
    cleaned = raw.strip().lower()
    if not _EMAIL_RE.match(cleaned):
        logger.warning(f"Suspicious email: {raw!r}")
    return cleaned


def _clean_phone(raw: str) -> str:
    """Strip everything except digits and leading +."""
    digits = re.sub(r"[^\d+]", "", raw.strip())
    return digits


def _clean_date(raw: str | None) -> str | None:
    """
    Try to parse several date formats and return YYYY-MM-DD.
    Returns None if the string can't be parsed.
    """
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    logger.warning(f"Could not parse date: {raw!r}")
    return None


def _clean_amount(raw: float | str | None) -> float:
    """Convert amount to float, stripping currency symbols."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    # Remove currency words
    for word in _CURRENCY_WORDS:
        s = re.sub(rf"\b{word}\b", "", s, flags=re.IGNORECASE)
    s = _AMOUNT_STRIP.sub("", s)
    # European decimal: "1.234,56" → "1234.56"
    if "," in s and "." in s:
        if s.index(",") > s.index("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        logger.warning(f"Could not parse amount: {raw!r}")
        return 0.0


def _clean_vat_rate(raw: float | str | None) -> float:
    """Return 0, 15, or 25 – the only valid Tripletex VAT rates."""
    val = _clean_amount(raw)
    if val > 30:
        val = val / 100  # already in percent × 100 form
    valid = {0.0, 15.0, 25.0}
    closest = min(valid, key=lambda v: abs(v - val))
    if closest != val:
        logger.warning(f"VAT rate {val} rounded to {closest}")
    return closest
