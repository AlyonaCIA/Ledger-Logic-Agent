from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from agent.models import FileAttachment

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Client initialisation (lazy)
# Priority: GEMINI_API_KEY (AI Studio) → Vertex AI (ADC/Workload Identity)
# ------------------------------------------------------------------ #

_client: genai.Client | None = None
_MODEL = "gemini-2.5-flash"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            logger.info("Gemini client: AI Studio (GEMINI_API_KEY)")
            _client = genai.Client(api_key=api_key)
        else:
            logger.info("Gemini client: Vertex AI (ADC/Workload Identity)")
            _client = genai.Client(
                vertexai=True,
                project=os.environ["GCP_PROJECT_ID"],
                location=os.getenv("VERTEX_LOCATION", "europe-west1"),
            )
    return _client


# ------------------------------------------------------------------ #
# System prompt – covers all task types in 7 languages
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """You are an expert accounting task analyzer for Tripletex (Norwegian ERP).

Parse prompts in ANY of these 7 languages: Norwegian Bokmål (nb), English (en), Spanish (es), Portuguese (pt), Norwegian Nynorsk (nn), German (de), French (fr).

Extract the EXACT task type and ALL entity fields from the prompt. Be precise.

══════════════════════════════════════════════
TASK TYPES
══════════════════════════════════════════════

create_employee
  Create a new employee. Extract:
  - first_name, last_name, email, phone
  - is_account_admin: TRUE if role is any of:
    nb: kontoadministrator, kontoadmin
    en: account administrator, account admin
    es: administrador de cuenta
    pt: administrador de conta
    nn: kontoadministrator
    de: Kontoadministrator, Kontoverwaltung
    fr: administrateur de compte

update_employee
  Update existing employee fields. Use "identifier" for the name/email used to find them.

delete_employee
  Remove an employee. Use "identifier" for name/email to find them.

create_customer
  Create customer/client. Extract: name, email, phone, org_number.
  Set is_supplier=true if they are a supplier.

update_customer
  Update customer info. Use "identifier" to find them.

delete_customer
  Remove a customer.

create_product
  Create a product or service. Extract: name, number, price_excl_vat, vat_rate (0/15/25), unit, description.

create_invoice
  Create an invoice. Steps: find/create customer → create order → create invoice.
  Extract under customer: name (identifier for lookup).
  Extract under invoice: date (YYYY-MM-DD, default today), due_days (default 14).
  Extract order_lines: each line has description, count, unit_price_excl_vat.
  If a single amount is given with no line detail, make one line with that amount.

register_payment
  Register a payment against an existing invoice.
  Extract under invoice: identifier (invoice number if numeric, else customer name).
  Extract under payment: amount, date (default today).
  Extract under customer: name (to find invoice by customer).

create_credit_note
  Create a credit note (reversal) for an invoice.
  Extract under invoice: identifier or date.
  Extract under customer: name to find the invoice.

create_travel_expense
  Create a travel expense report for an employee.
  Extract under employee: identifier (name to find them).
  Extract under travel_expense: description, from_date, to_date, amount, is_foreign_travel.

delete_travel_expense
  Delete a travel expense report.
  Extract under travel_expense: identifier (description or employee name).
  Extract under employee: identifier.

create_project
  Create a project linked to a customer.
  Extract under project: name, number, start_date, end_date, description.
  Extract under customer: name (to find/create customer).

create_department
  Create a department.
  Extract under department: name (the full department name as given, including any words or numbers), department_number (only if an explicit numeric code/ID is given separately from the name, e.g. "avdelingsnummer 100" or "number 100"; omit if the number is part of the name).

enable_module
  Enable a Tripletex module (department accounting, project accounting, travel expenses, etc.).
  Set module_name to: "department", "project", or "travel_expense" based on context.

delete_voucher
  Delete / reverse an incorrect ledger entry or voucher.
  Put any identifying info in notes.

══════════════════════════════════════════════
DATES
══════════════════════════════════════════════
Always output dates as YYYY-MM-DD.
TODAY = 2026-03-20.
If no date specified: use today for invoices/payments; for projects use today as start_date.

══════════════════════════════════════════════
AMOUNTS
══════════════════════════════════════════════
Extract numeric values only (strip currency symbols: kr, NOK, €, $, £, etc.).

══════════════════════════════════════════════
METADATA FIELDS (always required)
══════════════════════════════════════════════
confidence: float 0.0–1.0 — how certain you are about task_type and all key fields.
  1.0 = completely unambiguous
  0.7 = likely correct but some uncertainty
  <0.5 = use task_type "unknown"

language_detected: ISO 639-1 code of the prompt language.
  nb=Norwegian Bokmål, nn=Nynorsk, en=English, es=Spanish, pt=Portuguese, de=German, fr=French

missing_fields: list any REQUIRED fields that were NOT found in the prompt.
  e.g. ["employee.email", "invoice.amount"]
  Empty list [] if nothing is missing.

══════════════════════════════════════════════
OUTPUT FORMAT — CRITICAL
══════════════════════════════════════════════
Return ONLY a single valid JSON object. No markdown, no explanation, no code fences.

Required top-level keys: task_type, confidence, language_detected, missing_fields.
Optional entity keys (include only if relevant): employee, customer, product, invoice, payment, project, department, travel_expense, module_name, notes.

Set entity field values to null when not present in the prompt. Never invent data.

Example output:
{
  "task_type": "create_customer",
  "confidence": 0.98,
  "language_detected": "nb",
  "missing_fields": [],
  "customer": {"name": "Acme AS", "email": "post@acme.no", "phone": null, "org_number": null, "is_supplier": null, "identifier": null}
}
"""

# ------------------------------------------------------------------ #
# JSON repair / extraction helpers
# ------------------------------------------------------------------ #

def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract and parse a JSON object from model output.
    Handles markdown code fences and stray leading/trailing text.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _safe_parse(raw: str) -> dict[str, Any]:
    """Parse Gemini output into a valid intent dict, with a safe fallback."""
    try:
        result = _extract_json(raw)
        # Ensure mandatory keys
        result.setdefault("task_type", "unknown")
        result.setdefault("confidence", 0.0)
        result.setdefault("language_detected", "unknown")
        result.setdefault("missing_fields", [])
        return result
    except Exception as exc:
        logger.warning(f"JSON parse failed: {exc} | raw={raw[:200]!r}")
        return {
            "task_type": "unknown",
            "confidence": 0.0,
            "language_detected": "unknown",
            "missing_fields": ["parse_error"],
        }


# ------------------------------------------------------------------ #
# PDF text extraction
# ------------------------------------------------------------------ #

def _extract_pdf_text(b64_content: str, filename: str) -> str:
    """Extract text from a base64-encoded PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        raw = base64.b64decode(b64_content)
        doc = fitz.open(stream=raw, filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        extracted = "\n".join(pages_text).strip()
        if extracted:
            return f"[PDF: {filename}]\n{extracted[:4000]}"
    except Exception as exc:
        logger.warning(f"PDF extraction failed for {filename}: {exc}")
    return f"[Attached PDF: {filename} – could not extract text]"


# ------------------------------------------------------------------ #
# Main entry  (async-compatible via run_in_executor in main.py)
# ------------------------------------------------------------------ #

def _parse_task_sync(prompt: str, files: list[FileAttachment]) -> dict[str, Any]:
    """
    Synchronous Gemini call. Wrap in asyncio.get_event_loop().run_in_executor
    for async contexts, or call directly in sync tests.
    """
    parts: list[types.Part] = [types.Part.from_text(text=f"Task prompt:\n{prompt}")]

    for f in files:
        if f.mime_type.startswith("image/"):
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(f.content_base64),
                    mime_type=f.mime_type,
                )
            )
        elif f.mime_type == "application/pdf":
            pdf_text = _extract_pdf_text(f.content_base64, f.filename)
            parts.append(types.Part.from_text(text=pdf_text))
        else:
            parts.append(types.Part.from_text(text=f"[Attached file: {f.filename} ({f.mime_type})]"))

    client = _get_client()
    response = client.models.generate_content(
        model=_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=1024,
        ),
    )

    raw = response.text
    result = _safe_parse(raw)
    logger.info(f"Parsed intent: task={result.get('task_type')} conf={result.get('confidence'):.2f} lang={result.get('language_detected')}")
    return result


async def parse_task(
    prompt: str, files: list[FileAttachment]
) -> dict[str, Any]:
    """Async wrapper – runs the sync Gemini call in a thread pool."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _parse_task_sync, prompt, files)

