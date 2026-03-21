from __future__ import annotations

import base64
import json
import logging
import os
from datetime import date
from pathlib import Path
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

_SYSTEM_PROMPT_TEMPLATE = """You are an expert accounting task analyzer for Tripletex (Norwegian ERP).

Parse prompts in ANY of these 7 languages: Norwegian Bokmål (nb), English (en), Spanish (es), Portuguese (pt), Norwegian Nynorsk (nn), German (de), French (fr).

Extract the EXACT task type and ALL entity fields from the prompt. Be precise.

══════════════════════════════════════════════
TASK TYPES
══════════════════════════════════════════════

create_employee
  Create a new employee. Extract:
  - first_name, last_name, email, phone
  - start_date (YYYY-MM-DD): employment start date if mentioned
  - annual_salary: yearly salary in NOK if mentioned (e.g. "660 000 kr/år" → 660000)
  - work_percent: percentage of full-time employment if mentioned (e.g. "80 %" → 80)
  - job_title: job title / occupation (e.g. "Regnskapssjef", "Software Engineer")
  - date_of_birth (YYYY-MM-DD): birth date if mentioned
  - national_id_number: national ID / personnummer if mentioned
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
  ALSO MATCHES (multilingual): "commande" / "bon de commande" (fr), "Rechnung" / "Bestellung" (de), "factura" / "pedido" (es), "fatura" (pt), "bestilling" (nb/nn).
  NOTE: This is an OUTGOING invoice to a customer. For an INCOMING invoice from a supplier, use create_supplier_invoice instead.
  COMBINED INVOICE+PAYMENT: If the prompt ALSO asks to register payment after creating the invoice
  (nb: "registrer full betaling" / "registrer betaling", en: "register full payment" / "register payment",
   es: "registrar pago total", pt: "registre o pagamento", de: "zahlung registrieren", fr: "enregistrer le paiement"),
  ALSO extract under payment: {{amount: <total from order_lines or stated total>, date: today}}.
  This triggers automatic payment registration after invoice creation.

create_supplier_invoice
  Register an INCOMING invoice received FROM a supplier/vendor (leverandørfaktura, Lieferantenrechnung, facture fournisseur, factura de proveedor).
  DISTINCT from create_invoice: the money flows FROM us TO the supplier, not TO us.
  Use this when the prompt says: "received invoice from", "we got a bill from", "register supplier invoice", or uses supplier-specific terms.
  Extract under customer: name (= the supplier name), org_number, email; set is_supplier=true.
  Extract under invoice: invoice_number (external ref e.g. "INV-2026-3063"), amount (total incl VAT),
    amount_excl_vat (if stated), date (YYYY-MM-DD), vat_rate (default 25), account_code (GL account, e.g. 7300).

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
  Extract under employee: identifier (name to find them), email if given.
  Extract under travel_expense:
    - description: trip title/purpose
    - destination: destination city or country (e.g. "Bergen", "Oslo", "Paris")
    - from_date (YYYY-MM-DD): departure date if mentioned
    - to_date (YYYY-MM-DD): return date if mentioned
    - is_foreign_travel: true if international/foreign travel
    - per_diem_days: number of days (e.g. "5 days" → 5)
    - per_diem_rate: daily allowance rate in NOK/day if mentioned
    - costs: array of individual expenses, each with:
        type: one of "flight", "taxi", "hotel", "bus", "ferry", "train", "food", "parking", "other"
        amount: amount in NOK
        description: brief label (original language ok)
  Example for FR: "billet d'avion 3850 NOK et taxi 650 NOK" →
    costs: [{{"type":"flight","amount":3850,"description":"Billet d'avion"}},{{"type":"taxi","amount":650,"description":"Taxi"}}]
  NOTE: do NOT use this for payroll/salary tasks — use run_payroll instead.

run_payroll
  Record payroll/salary for an employee via the salary API.
  Use this for ALL payroll/salary processing tasks in ANY language:
    en: "run payroll", "process payroll", "process salary"
    nb/nn: "kjør lønn", "lønn", "lønning", "utbetal lønn"
    de: "Gehaltsabrechnung", "Lohnabrechnung", "Gehalt auszahlen", "führen Sie die Gehaltsabrechnung"
    es: "nómina", "ejecute la nómina", "procesar nómina"
    fr: "salaire", "fiche de paie", "traiter la paie", "effectuer la paie"
    pt: "folha de pagamento", "processar salário"
  Extract under employee: first_name, last_name, email, identifier (full name).
  Extract under payroll: base_salary (number, required), bonus (number, 0 if none),
    year (current year if not stated), month (current month if not stated).

create_project_invoice
  Register hours worked on a project for an employee, then generate a project invoice.
  Use when the prompt asks to BOTH register hours AND generate a project invoice:
    es: "Registre X horas...en la actividad...del proyecto...Genere una factura de proyecto"
    fr: "Enregistrez X heures...sur l'activité...du projet...Générez une facture de projet"
    en: "Register X hours...on activity...of project...generate a project invoice"
    nb: "Registrer X timer...på aktiviteten...for prosjektet...generer prosjektfaktura"
    de: "X Stunden erfassen...auf Aktivität...des Projekts...Projektrechnung erstellen"
    pt: "Registre X horas...na atividade...do projeto...gere uma fatura de projeto"
  Extract under employee: identifier (full name), email (if present), first_name, last_name.
  Extract under project: name (project name), activity (activity name, e.g. "Design").
  Extract under customer: name, org_number (if present).
  Extract under invoice: hours (number of hours), hourly_rate (rate per hour in NOK), date (today if not given).

delete_travel_expense
  Delete a travel expense report.
  Extract under travel_expense: identifier (description or employee name).
  Extract under employee: identifier.

create_project
  Create a project linked to a customer.
  Extract under project: name, number, start_date, end_date, description.
  Extract under customer: name (to find/create customer).

create_department
  Create one or more departments.
  If the prompt asks to create MULTIPLE departments, output "department" as a JSON array.
  Each element: {{"name": "...", "department_number": null}}.
  If only one department, output "department" as a single object (not an array).
  department_number: only if an explicit numeric code/ID is given separately from the name.

enable_module
  Enable a Tripletex module (department accounting, project accounting, travel expenses, etc.).
  Set module_name to: "department", "project", or "travel_expense" based on context.

delete_voucher
  Delete / reverse an incorrect ledger entry or voucher.
  Put any identifying info in notes.

ledger_task
  Post complex ledger entries: corrections, depreciation, monthly/annual close.
  Use when the task involves reversing wrong vouchers, posting depreciation entries,
  or doing period-end accounting close.
  Extract under ledger:
    - subtask: one of "correction", "depreciation", "monthly_close", "annual_close", "voucher"
    - description: description of what is being posted
    - date (YYYY-MM-DD): accounting date for the entry (default today)
    - date_from, date_to: date range to search for vouchers to correct (for "correction")
    - asset_cost: original asset cost (for depreciation)
    - years: useful life in years (for depreciation; annual_amount = asset_cost / years)
    - annual_amount: depreciation amount per year if directly stated
    - depreciation_account: GL account number for depreciation expense (default "6010")
    - accumulated_account: GL account number for accumulated depreciation (default "1209")
    - postings: array of {{account_number, account_name (optional), amount}} for generic entries

NOT SUPPORTED — use task_type "unknown" for:
  - Free accounting dimensions / free dimensions ("fri regnskapsdimensjon", "dimensión contable libre")
  - Any task not listed above

══════════════════════════════════════════════
MULTILINGUAL TERM → TRIPLETEX ENDPOINT CHEATSHEET
══════════════════════════════════════════════
Use this table to map foreign-language accounting terms directly to the
correct task_type without any ambiguity:

  nb: Faktura / Ordre     → create_invoice   (/invoice)
  en: Invoice / Order     → create_invoice   (/invoice)
  es: Factura / Pedido    → create_invoice   (/invoice)
  pt: Fatura              → create_invoice   (/invoice)
  de: Rechnung / Bestellung → create_invoice (/invoice)
  fr: Facture / Commande / Bon de commande → create_invoice (/invoice)

  nb: Leverandørfaktura   → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)
  en: Supplier invoice / Vendor invoice → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)
  de: Lieferantenrechnung → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)
  fr: Facture fournisseur → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)
  es: Factura de proveedor → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)
  pt: Fatura de fornecedor → create_supplier_invoice (/ledger/voucher with vendorInvoiceNumber)

  nb: Ansatt / Medarbeider → create_employee  (/employee)
  en: Employee / Staff     → create_employee  (/employee)
  es: Empleado             → create_employee  (/employee)
  pt: Funcionário / Empregado → create_employee (/employee)
  de: Mitarbeiter / Angestellter → create_employee (/employee)
  fr: Employé / Collaborateur → create_employee (/employee)

  nb: Kunde / Klient       → create_customer  (/customer)
  en: Customer / Client    → create_customer  (/customer)
  es: Cliente              → create_customer  (/customer)
  pt: Cliente              → create_customer  (/customer)
  de: Kunde / Klient       → create_customer  (/customer)
  fr: Client               → create_customer  (/customer)

  nb: Reise / Utlegg       → create_travel_expense (/travelExpense)
  de: Dienstreise / Reisekosten → create_travel_expense (/travelExpense)
  fr: Note de frais / Voyage → create_travel_expense (/travelExpense)
  es: Gasto de viaje       → create_travel_expense (/travelExpense)

  en: run payroll / salary / payroll         → run_payroll
  nb: kjør lønn / lønning / lønn             → run_payroll
  de: Gehaltsabrechnung / Lohnabrechnung     → run_payroll
  es: nómina / ejecute la nómina             → run_payroll
  fr: salaire / fiche de paie / traiter la paie → run_payroll
  pt: folha de pagamento / processar salário → run_payroll

  en: project invoice / register hours       → create_project_invoice
  es: factura de proyecto / registre horas   → create_project_invoice
  fr: facture de projet / enregistrez heures → create_project_invoice
  nb: prosjektfaktura / registrer timer      → create_project_invoice
  de: Projektrechnung / Stunden erfassen     → create_project_invoice
  pt: fatura de projeto / registre horas     → create_project_invoice

══════════════════════════════════════════════
DATES
══════════════════════════════════════════════
Always output dates as YYYY-MM-DD.
TODAY = {today}.
If no date specified: use today for invoices/payments; for projects use today as start_date.

══════════════════════════════════════════════
ATTACHMENTS & OCR
══════════════════════════════════════════════
If an image or PDF is attached:
1. Identify what it contains: Invoice, Receipt/travel expense, or identity document.
2. Extract: TOTAL AMOUNT, DATE (YYYY-MM-DD), VENDOR/SUPPLIER NAME, and any INVOICE NUMBER.
3. For travel receipts: also extract employee name and whether travel was foreign.
4. When the text prompt is vague (e.g. "process this receipt", "faktura vedlagt"),
   fill entity fields primarily from the file content, not the prompt.
5. If the file clearly shows an invoice, set task_type = "create_invoice" or
   "register_payment" based on context.

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
{{
  "task_type": "create_customer",
  "confidence": 0.98,
  "language_detected": "nb",
  "missing_fields": [],
  "customer": {{"name": "Acme AS", "email": "post@acme.no", "phone": null, "org_number": null, "is_supplier": null, "identifier": null}}
}}
"""

# ------------------------------------------------------------------ #
# JSON repair / extraction helpers
# ------------------------------------------------------------------ #

def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract and parse a JSON object from model output.
    Handles markdown code fences, stray leading/trailing text, and arrays.
    When the model returns a JSON array, the first element is used.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()

    # Fast path: try the full text as-is (handles both {} and [{...}] responses)
    try:
        result = json.loads(text)
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find first { ... } block and parse it
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: scan for balanced braces to find the first complete object
    depth = 0
    obj_start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                return json.loads(text[obj_start : i + 1])

    # Nothing worked — let json.loads raise its own error
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
    # Quick keyword hint so few-shot retrieval can pick relevant examples
    # before the LLM runs (avoids a second LLM call).
    from agent.executor import _keyword_fallback, _is_not_supported  # lightweight import

    # Fast-path: if the prompt is clearly unsupported (payroll etc.), skip LLM
    if _is_not_supported(prompt):
        logger.info("Parser fast-path: prompt is not supported (payroll/dimensions), returning unknown")
        return {
            "task_type": "unknown",
            "confidence": 1.0,
            "language_detected": "unknown",
            "missing_fields": [],
        }

    task_hint = _keyword_fallback(prompt)

    few_shot_block = _get_few_shots(task_hint if task_hint != "unknown" else None)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())

    parts: list[types.Part] = []
    if few_shot_block:
        parts.append(types.Part.from_text(text=few_shot_block))
    parts.append(types.Part.from_text(text=f"Task prompt:\n{prompt}"))

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
            system_instruction=system_prompt,
            temperature=0,
            # Increase tokens — complex multi-product invoices need space for full JSON
            max_output_tokens=4096,
            # Force valid JSON output: prevents markdown fences and truncated strings
            response_mime_type="application/json",
            # Disable automatic function calling — it is ON by default in genai 1.68+
            # and adds latency / can interfere with plain text generation
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    raw = response.text
    result = _safe_parse(raw)
    logger.info(f"Parsed intent: task={result.get('task_type')} conf={result.get('confidence'):.2f} lang={result.get('language_detected')}")
    return result


# ------------------------------------------------------------------ #
# Gold few-shot retrieval
# ------------------------------------------------------------------ #

_GOLD_PROMPTS: list[dict] | None = None
_GOLD_PATH = Path("gold/prompts.jsonl")


def _load_gold() -> list[dict]:
    """Load gold examples once; return empty list if file absent."""
    global _GOLD_PROMPTS
    if _GOLD_PROMPTS is None:
        _GOLD_PROMPTS = []
        if _GOLD_PATH.exists():
            with _GOLD_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            _GOLD_PROMPTS.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            logger.debug(f"Loaded {len(_GOLD_PROMPTS)} gold examples from {_GOLD_PATH}")
    return _GOLD_PROMPTS


def _get_few_shots(task_type_hint: str | None = None, max_examples: int = 2) -> str:
    """
    Return a formatted few-shot block for the given task type (or all types).
    Called before the LLM parse to inject 1–2 canonical examples.
    The hint is derived from a quick keyword scan of the prompt.
    """
    gold = _load_gold()
    if not gold:
        return ""

    # Filter by task type if we have a hint
    candidates = gold
    if task_type_hint:
        typed = [g for g in gold if g.get("task_type") == task_type_hint]
        if typed:
            candidates = typed

    # Take up to max_examples (pick diverse languages)
    selected: list[dict] = []
    seen_langs: set[str] = set()
    for ex in candidates[:max_examples * 3]:  # scan a bit wider for diversity
        lang = ex.get("language", "?")
        if lang not in seen_langs:
            selected.append(ex)
            seen_langs.add(lang)
        if len(selected) >= max_examples:
            break

    if not selected:
        selected = candidates[:max_examples]

    if not selected:
        return ""

    lines = ["\n══════════════════════════════════════════════",
             "FEW-SHOT EXAMPLES (from verified gold dataset)",
             "══════════════════════════════════════════════"]
    for i, ex in enumerate(selected, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"  prompt: {ex.get('prompt', '')}")
        entities = {k: v for k, v in ex.get("entities", {}).items() if k != "_raw_prompt"}
        compact = json.dumps(entities, ensure_ascii=False)
        lines.append(f"  output: {{\"task_type\": \"{ex.get('task_type')}\", \"confidence\": {ex.get('confidence', 1.0)}, \"language_detected\": \"{ex.get('language', '?')}\", \"missing_fields\": [], {compact[1:]}")
    lines.append("")
    return "\n".join(lines)


async def parse_task(
    prompt: str, files: list[FileAttachment]
) -> dict[str, Any]:
    """Async wrapper – runs the sync Gemini call in a thread pool."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _parse_task_sync, prompt, files)

