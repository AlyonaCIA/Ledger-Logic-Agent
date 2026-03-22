"""
Task executor – deterministic workflow engine for Tripletex accounting tasks.

Design principles:
  • Plan before calling – analyse intent fully before any HTTP request.
  • Minimise GET calls – use IDs from POST responses, not extra lookups.
  • Zero 4xx errors – validate required fields before every request.
  • Idempotent lookups – search → create only when not found.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from typing import Any

import requests

from agent.client import TripletexClient
from agent.matchers import resolve_customer, resolve_employee, resolve_invoice
from agent.validators import ValidationError, validate_intent

logger = logging.getLogger(__name__)
TODAY = date.today().isoformat()


# ======================================================================
# Raw-prompt extraction helpers
# ======================================================================

def _extract_customer_from_prompt(raw: str) -> dict:
    """Fallback: extract customer name and org number from raw prompt text."""
    cust: dict = {}
    # Org number: 9-digit Norwegian format
    org_m = re.search(r'(?:org\.?\s*(?:nr?|n[uú]m)\.?\s*|Org\.\-Nr\.\s*)(\d{9})', raw)
    if org_m:
        cust["org_number"] = org_m.group(1)
    # Customer name: text before org-no pattern, or after "for/für/pour/para"
    name_patterns = [
        # "for CustomerName (org no. XXXX)" - various languages
        r'(?:for|für|pour|para|til)\s+[\'"]?([A-ZÀ-Ž][\w\s&\-]+?)\s*\(',
        # "CustomerName (org. nr XXXX)"
        r'([A-ZÀ-Ž][\w\s&\-]+?)\s*\(\s*(?:org|Org)',
        # "kunde/client/customer/Kunden CustomerName"
        r'(?:kund(?:en?|e)|client|customer|Kunden?)\s+[\'"]?([A-ZÀ-Ž][\w\s&\-]+?)(?:\s*\(|,|\s+med|\s+with)',
    ]
    for pat in name_patterns:
        m = re.search(pat, raw)
        if m:
            name = m.group(1).strip().rstrip(".")
            if len(name) > 2:
                cust["name"] = name
                break
    return cust


# ======================================================================
# Self-Healing helper
# ======================================================================

def _post_value_with_heal(client: TripletexClient, path: str, json_payload: dict) -> Any:
    """
    POST with a single self-healing retry on 400/422 errors.
    Tripletex returns detailed validationMessages on these status codes;
    we forward them to Gemini Flash which fixes the payload and we retry once.
    Falls back to raising the original exception if healing produces no change.
    """
    try:
        return client.post_value(path, json=json_payload)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 422):
            error_text = exc.response.text
            logger.warning(f"422 on {path} – triggering self-heal. Error: {error_text[:200]}")
            from agent.healer import heal_payload  # lazy – avoids circular import at load time
            healed = heal_payload(path, json_payload, error_text)
            if healed != json_payload:
                return client.post_value(path, json=healed)
        raise


# ======================================================================
# Router
# ======================================================================

def execute_task(intent: dict[str, Any], client: TripletexClient) -> None:
    task_type = intent.get("task_type", "unknown")
    confidence = intent.get("confidence", 1.0)
    language = intent.get("language_detected", "?")
    missing = intent.get("missing_fields") or []

    # ── Hard blacklist: override LLM regardless of confidence ──────────────────
    # Payroll/salary and other unsupported tasks must NEVER execute, even if the
    # LLM assigns them to a real task_type at conf=0.50.
    _raw = intent.get("_raw_prompt", "")
    if _raw and _is_not_supported(_raw):
        logger.warning(f"Blacklisted task (payroll/unsupported): task_type was {task_type!r}, forcing unknown")
        task_type = "unknown"
        intent["task_type"] = "unknown"
        confidence = 1.0  # confident it's unsupported

    # ── Post-LLM correction: fix common misclassifications ─────────────────
    if _raw:
        p_low = _raw.lower()

        # (A) Payroll misclassified as invoice – German "Gehaltsabrechnung" contains "rechnung"
        if task_type == "create_invoice":
            _payroll_kws = ["gehaltsabrechnung", "lohnabrechnung", "payroll", "lønn",
                            "nómina", "nomina", "folha de pagamento", "fiche de paie",
                            "salary", "gehalt", "lønning", "salario base"]
            if any(kw in p_low for kw in _payroll_kws):
                task_type = "run_payroll"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_invoice → run_payroll")

        # (B) Bank reconciliation misclassified as invoice
        if task_type == "create_invoice":
            _bank_kws = ["bankutskrift", "kontoauszug", "relevé bancaire",
                         "extracto bancario", "extrato bancário", "avstem",
                         "reconcile bank", "bank statement", "bank reconciliation"]
            if any(kw in p_low for kw in _bank_kws):
                task_type = "bank_reconciliation"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_invoice → bank_reconciliation")

        # (C) Hours + project → create_project_invoice
        if task_type == "create_invoice":
            _hours_kws = ["registre horas", "enregistrez", "register hours",
                          "registrer timer", "stunden erfassen", "heures pour",
                          "horas para", "hours for", "timer for", "horas de"]
            _act_kws = ["activité", "actividad", "atividade", "activity", "aktivitet"]
            has_hours = any(kw in p_low for kw in _hours_kws)
            has_activity = any(kw in p_low for kw in _act_kws)
            if has_hours and has_activity:
                task_type = "create_project_invoice"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_invoice → create_project_invoice (hours+activity)")

        # (D) Project + PM/fixed-price → create_project_invoice
        if task_type == "create_invoice":
            has_project = any(kw in p_low for kw in ["prosjekt", "project", "proyecto", "projet", "projekt"])
            has_pm = any(kw in p_low for kw in [
                "prosjektleiar", "prosjektleder", "projektleder", "project leader",
                "project manager", "projektleiter", "chef de projet",
                "director del proyecto", "gerente de projeto", "gerente do projeto",
                "fastpris", "fastpris på prosjekt",
            ])
            if has_project and has_pm:
                task_type = "create_project_invoice"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_invoice → create_project_invoice (project+PM)")

        # (E) FX payment: "envoyé une facture...payé" → register_payment
        if task_type == "create_invoice":
            _fx_kws = ["envoyé une facture", "le client a maintenant payé",
                       "écart de change", "agio", "taux de change"]
            if sum(1 for kw in _fx_kws if kw in p_low) >= 2:
                task_type = "register_payment"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_invoice → register_payment (FX payment)")

        # (F) Custom dimension misclassified as create_customer
        if task_type == "create_customer":
            _dim_kws = ["regnskapsdimensjon", "accounting dimension",
                        "dimension comptable", "dimensión contable",
                        "dimensão contábil", "buchhaltungsdimension",
                        "fri dimensjon", "kostsenter"]
            if any(kw in p_low for kw in _dim_kws):
                task_type = "ledger_task"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_customer → ledger_task (custom dimension)")

        # (G) Payment reversal misclassified as delete_voucher
        if task_type == "delete_voucher":
            _rev_kws = ["reverser betaling", "returnert av banken",
                        "reverse payment", "returned by bank",
                        "annuler le paiement", "anular el pago"]
            if any(kw in p_low for kw in _rev_kws):
                task_type = "register_payment"
                intent["task_type"] = task_type
                if not intent.get("payment"):
                    intent["payment"] = {}
                intent["payment"]["is_reversal"] = True
                logger.info("Post-LLM correction: delete_voucher → register_payment (reversal)")

        # (H) Receipt/kvittering misclassified as create_supplier_invoice
        if task_type == "create_supplier_invoice":
            _receipt_kws = ["kvittering", "receipt", "quittung", "reçu", "recibo",
                            "expense from this", "ausgabe aus dieser",
                            "despesa de", "dépense de ce"]
            if any(kw in p_low for kw in _receipt_kws):
                task_type = "book_receipt"
                intent["task_type"] = task_type
                logger.info("Post-LLM correction: create_supplier_invoice → book_receipt")

        # (H) Receipt/kvittering misclassified as supplier invoice or other
        if task_type in ("create_supplier_invoice", "create_invoice", "ledger_task"):
            _receipt_kws = ["kvittering", "receipt", "quittung", "reçu", "recibo",
                            "expense from this receipt", "utgiften fra denne kvitteringen",
                            "despesa de", "ausgabe aus dieser quittung",
                            "dépense de ce reçu"]
            if any(kw in p_low for kw in _receipt_kws):
                task_type = "book_receipt"
                intent["task_type"] = task_type
                logger.info(f"Post-LLM correction: → book_receipt (receipt keywords)")

        # (J) Supplier creation misclassified as supplier invoice
        # When prompt says "register supplier" / "registrer leverandøren" WITHOUT invoice keywords,
        # route to create_customer with is_supplier=true
        if task_type == "create_supplier_invoice":
            _supplier_create_kws = [
                "registrer leverandøren", "legg til leverandør", "opprett leverandør",
                "register the supplier", "add the supplier", "create the supplier",
                "add supplier", "create supplier",
                "enregistrer le fournisseur", "créer le fournisseur", "ajouter le fournisseur",
                "registrar el proveedor", "crear el proveedor", "añadir el proveedor",
                "registrar o fornecedor", "criar o fornecedor", "adicionar o fornecedor",
                "lieferanten registrieren", "lieferanten anlegen", "lieferanten hinzufügen",
            ]
            _invoice_kws = [
                "faktura", "invoice", "rechnung", "facture", "factura", "fatura",
                "regning", "leverandørfaktura",
            ]
            has_supplier_create = any(kw in p_low for kw in _supplier_create_kws)
            has_invoice = any(kw in p_low for kw in _invoice_kws)
            if has_supplier_create and not has_invoice:
                task_type = "create_customer"
                intent["task_type"] = task_type
                # Signal that this is a supplier, not a customer
                if not intent.get("customer"):
                    intent["customer"] = {}
                intent["customer"]["is_supplier"] = True
                # Copy supplier fields to customer if parsed as supplier_invoice
                if intent.get("supplier_invoice") and not intent["customer"].get("name"):
                    si = intent["supplier_invoice"]
                    intent["customer"]["name"] = si.get("supplier_name")
                    intent["customer"]["org_number"] = si.get("supplier_org_number")
                logger.info("Post-LLM correction: create_supplier_invoice → create_customer (supplier registration)")

    # Reclassify create_project_invoice → create_project when prompt has full project-cycle markers
    if task_type == "create_project_invoice" and _raw:
        p_low = _raw.lower()
        has_cycle = any(kw in p_low for kw in [
            "projektzyklus", "prosjektsyklus", "project cycle", "ciclo del proyecto",
            "ciclo do projeto", "cycle de projet", "vollständigen projektzyklus",
            "budget", "budsj", "leverandørkostnad", "leverandorkostnad",
            "supplier cost", "lieferantenkosten", "coste de proveedor", "custo do fornecedor",
        ])
        if has_cycle:
            task_type = "create_project"
            intent["task_type"] = task_type
            logger.info("Post-LLM correction: create_project_invoice → create_project (full cycle)")

    # (I) Payroll misclassified as travel_expense, create_invoice, unknown, or other
    if task_type in ("create_travel_expense", "create_invoice", "create_supplier_invoice",
                     "ledger_task", "create_employee", "unknown") and _raw:
        p_low = _raw.lower()
        _payroll_kws = [
            "payroll", "salary", "run payroll", "process payroll",
            "kjør lønn", "lønning", "lønn", "utbetal lønn",
            "gehaltsabrechnung", "lohnabrechnung", "gehalt auszahlen",
            "führen sie die gehaltsabrechnung", "gehalt",
            "nómina", "ejecute la nómina", "procesar nómina", "salario base",
            "folha de pagamento", "processar salário", "salário",
            "fiche de paie", "traiter la paie", "effectuer la paie", "salaire",
            "base salary", "bonus", "grunnlønn", "fastlønn",
        ]
        if any(kw in p_low for kw in _payroll_kws):
            task_type = "run_payroll"
            intent["task_type"] = task_type
            logger.info(f"Post-LLM correction: → run_payroll (payroll keywords)")

    # (K) Ledger corrections misclassified as other tasks
    if task_type not in ("ledger_task",) and _raw:
        p_low = _raw.lower()
        _correction_kws = [
            "feil i hovedboken", "feil konto", "duplisert bilag", "manglende mva",
            "feil beløp", "korriger alle feil", "feilføring",
            "errors in the ledger", "wrong account", "duplicate voucher", "missing vat",
            "wrong amount", "correct the errors", "correct all errors",
            "erreurs dans le grand livre", "corrigez les erreurs",
            "fehler im hauptbuch", "korrigieren sie die fehler",
            "errores en el libro mayor", "corrija los errores",
            "erros no razão geral", "corrija os erros",
        ]
        # Need strong signal: at least 2 correction keywords to override
        correction_hits = sum(1 for kw in _correction_kws if kw in p_low)
        if correction_hits >= 2:
            task_type = "ledger_task"
            intent["task_type"] = task_type
            if not intent.get("ledger"):
                intent["ledger"] = {}
            intent["ledger"]["subtask"] = "correction"
            logger.info(f"Post-LLM correction: → ledger_task/correction ({correction_hits} keywords)")

    # ── Keyword fallback: rescue "unknown" or low-conf with simple heuristics ──
    if task_type == "unknown" or confidence < 0.50:
        fallback_type = _keyword_fallback(intent.get("_raw_prompt", ""))
        if fallback_type != "unknown":
            task_type = fallback_type
            intent["task_type"] = task_type
            intent["confidence"] = 0.5
            confidence = 0.5
            logger.info(f"Keyword fallback resolved task_type={task_type}")

    # ── Second-pass correction after keyword fallback ──────────────────
    # The keyword fallback may assign "create_supplier_invoice" when it should be
    # "create_customer" (supplier registration). Re-run relevant corrections.
    if task_type == "create_supplier_invoice" and _raw:
        p_low = _raw.lower()
        _supplier_create_kws = [
            "registrer leverandøren", "legg til leverandør", "opprett leverandør",
            "register the supplier", "add the supplier", "create the supplier",
            "add supplier", "create supplier",
            "enregistrer le fournisseur", "créer le fournisseur",
            "registrar el proveedor", "crear el proveedor",
            "registrar o fornecedor", "criar o fornecedor",
            "lieferanten registrieren", "lieferanten anlegen",
        ]
        _invoice_kws = [
            "faktura", "invoice", "rechnung", "facture", "factura", "fatura",
            "regning", "leverandørfaktura",
        ]
        if any(kw in p_low for kw in _supplier_create_kws) and not any(kw in p_low for kw in _invoice_kws):
            task_type = "create_customer"
            intent["task_type"] = task_type
            if not intent.get("customer"):
                intent["customer"] = {}
            intent["customer"]["is_supplier"] = True
            if intent.get("supplier_invoice"):
                si = intent["supplier_invoice"]
                if not intent["customer"].get("name"):
                    intent["customer"]["name"] = si.get("supplier_name")
                if not intent["customer"].get("org_number"):
                    intent["customer"]["org_number"] = si.get("supplier_org_number")
            logger.info("Second-pass correction: create_supplier_invoice → create_customer (supplier)")

    logger.info(
        f"task={task_type} lang={language} confidence={confidence:.2f} "
        f"missing={missing}"
    )

    if task_type == "unknown" or confidence < 0.25:
        logger.warning(
            f"Low-confidence intent (confidence={confidence:.2f}), skipping execution"
        )
        return

    # Pre-flight validation – may raise ValidationError
    try:
        validate_intent(intent)
    except ValidationError as exc:
        logger.error(f"Validation failed: {exc}")
        return

    handlers = {
        "create_employee": _create_employee,
        "update_employee": _update_employee,
        "delete_employee": _delete_employee,
        "create_customer": _create_customer,
        "update_customer": _update_customer,
        "delete_customer": _delete_customer,
        "create_product": _create_product,
        "create_invoice": _create_invoice,
        "create_supplier_invoice": _create_supplier_invoice,
        "register_payment": _register_payment,
        "create_credit_note": _create_credit_note,
        "create_travel_expense": _create_travel_expense,
        "delete_travel_expense": _delete_travel_expense,
        "create_project": _create_project,
        "create_department": _create_department,
        "create_contact": _create_contact,
        "enable_module": _enable_module,
        "delete_voucher": _delete_voucher,
        "run_payroll": _run_payroll,
        "create_project_invoice": _create_project_invoice,
        "ledger_task": _ledger_task,
        "bank_reconciliation": _bank_reconciliation,
        "overdue_reminder": _overdue_reminder,
        "book_receipt": _book_receipt,
    }

    fn = handlers.get(task_type)
    if fn:
        t0 = time.monotonic()
        fn(intent, client)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"Workflow complete in {elapsed:.0f}ms")
    else:
        logger.warning(f"No handler for task_type='{task_type}'")


# ======================================================================
# Keyword fallback classifier
# ======================================================================

_KEYWORD_MAP: list[tuple[list[str], str]] = [
    # Bank reconciliation MUST come before payment to avoid wrong routing
    (["bank reconciliation", "reconcile bank", "bank statement", "bankutskrift",
      "avstemming", "avstemme", "kontoauszug", "relevé bancaire",
      "extracto bancario", "extrato bancário"], "bank_reconciliation"),
    # Overdue / reminder fee MUST come before generic invoice
    (["overdue invoice", "reminder fee", "purregebyr", "forfalt faktura", "purring",
      "mahngebühr", "überfällig", "frais de rappel", "facture en retard",
      "cargo por mora", "factura vencida", "taxa de lembrete", "fatura vencida",
      "inkassogebyr"], "overdue_reminder"),
    # Receipt / kvittering MUST come before supplier invoice
    (["kvittering", "receipt", "quittung", "reçu", "recibo",
      "we need the", "expense from this receipt",
      "despesa de", "ausgabe aus dieser quittung",
      "dépense de ce reçu"], "book_receipt"),
    # Supplier invoice MUST come before generic invoice to avoid wrong routing
    (["supplier invoice", "vendor invoice", "leverandørfaktura", "lieferantenrechnung",
      "facture fournisseur", "factura de proveedor", "fatura de fornecedor", "leverandør"], "create_supplier_invoice"),
    # project invoice (hours-based or fixed-price) – must come before generic invoice
    (["project invoice", "factura de proyecto", "facture de projet", "prosjektfaktura",
      "projektrechnung", "fatura de projeto", "register hours", "registre horas",
      "enregistrez les heures", "registrer timer", "stunden erfassen",
      "fastpris", "prosjektleiar", "prosjektleder", "projektleder",
      "project leader", "projektleiter", "chef de projet",
      "director del proyecto", "gerente de projeto"], "create_project_invoice"),
    # payroll must come BEFORE invoice to block "Gehaltsabrechnung" → "rechnung" match
    (["payroll", "gehaltsabrechnung", "lohnabrechnung", "kjør lønn", "lønning", "lønn",
      "nómina", "nomina", "folha de pagamento", "fiche de paie", "paie",
      "salary", "gehalt", "salario base"], "run_payroll"),
    # invoice / faktura / commande (order) → create_invoice
    (["invoice", "faktura", "factura", "fatura", "rechnung", "facture",
      "commande", "ordine", "encomenda", "auftrag", "pedido"], "create_invoice"),
    # employee / ansatt → create_employee
    (["employee", "ansatt", "empleado", "medarbeider", "funcionário", "mitarbeiter", "employé", "ansat", "arbeid"], "create_employee"),
    # customer / kunde → create_customer
    (["customer", "kunde", "client", "cliente", "klient", "kund", "Kunde"], "create_customer"),
    # product / produkt → create_product
    (["product", "produkt", "producto", "produit", "produkt", "Produkt", "vare"], "create_product"),
    # department / avdeling → create_department
    (["department", "avdeling", "departamento", "departement", "abteilung", "département"], "create_department"),
    # contact / kontaktperson → create_contact
    (["kontaktperson", "contact person", "Kontaktperson", "personne de contact",
      "persona de contacto", "pessoa de contato", "ansprechpartner"], "create_contact"),
    # project / prosjekt → create_project
    (["project", "prosjekt", "proyecto", "projet", "projekt", "Projekt"], "create_project"),
    # payment → register_payment
    (["payment", "betaling", "pago", "pagamento", "zahlung", "paiement", "betal"], "register_payment"),
    # (payroll moved before invoice above)
    # credit note / kreditnota → create_credit_note
    (["credit note", "kreditnota", "nota de crédito", "avoir", "gutschrift"], "create_credit_note"),
    # travel expense → create_travel_expense
    (["travel", "reise", "viaje", "voyage", "dienstreise", "utlegg",
      "note de frais", "nota de gastos", "nota de despesa", "reisekostenabrechnung",
      "indemnités journalières", "per diem", "dieta", "dagpenger",
      "expense report", "reiseregning"], "create_travel_expense"),
    # ledger task → ledger_task
    (["depreciation", "avskrivning", "monthly close", "annual close", "ledger correction",
      "reverse voucher", "ledger task", "regnskap",
      "accounting dimension", "regnskapsdimensjon", "dimensión contable",
      "dimensão contábil", "dimension comptable", "buchhaltungsdimension",
      "kostnadsbærer", "kostsenter", "fri dimensjon",
      "analyser hovudboka", "analyser hovedboken", "analyze the ledger",
      "analysieren sie das hauptbuch", "analysez le grand livre",
      "kostnadskontoane", "kostnadskontoen", "utgiftskonto",
      "biggest increase", "størst auke", "størst økning",
      "feilføring", "korriger alle feil", "correct the errors", "corrigez les erreurs",
      "corrija los errores", "corrija os erros", "korrigieren sie die fehler",
      "feil i hovedboken", "errors in the ledger"], "ledger_task"),
]


_NOT_SUPPORTED_KEYWORDS: list[str] = [
    # (none currently)
]


def _is_not_supported(prompt: str) -> bool:
    """Return True if the prompt clearly describes an unsupported task."""
    p = prompt.lower()
    return any(kw in p for kw in _NOT_SUPPORTED_KEYWORDS)


def _keyword_fallback(prompt: str) -> str:
    """Return a task_type based on keyword matching, or 'unknown'."""
    p = prompt.lower()
    for keywords, task_type in _KEYWORD_MAP:
        for kw in keywords:
            if kw.lower() in p:
                return task_type
    return "unknown"


# ======================================================================
# Shared helpers
# ======================================================================

# Thin wrappers kept for backward-compatibility with tests that import them.
# All new code should import from agent.matchers directly.

def _find_customer(client: TripletexClient, name: str | None) -> dict | None:
    return resolve_customer(client, name)


def _find_employee(client: TripletexClient, identifier: str | None) -> dict | None:
    return resolve_employee(client, name=identifier)


def _find_invoice_for_customer(
    client: TripletexClient, customer_id: int
) -> dict | None:
    return resolve_invoice(client, customer_id=customer_id)


# _get_default_payment_type removed – use client.get_payment_type() instead


# ======================================================================
# Employee workflows
# ======================================================================

def _get_default_department_id(client: TripletexClient) -> int | None:
    """Return a department id.

    Optimistic path: POST 'General' directly (fresh account → always succeeds,
    saves the GET /department call that would return 0 results anyway).
    Fallback: if the POST returns 422 (department already exists), GET to find it.
    """
    try:
        dept = client.post_value("/department", json={"name": "General"})
        return dept["id"] if dept else None
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 422):
            try:
                depts = client.get_list("/department", params={"fields": "id", "count": 1})
                return depts[0]["id"] if depts else None
            except Exception:
                return None
        return None


def _create_employee(intent: dict, client: TripletexClient) -> None:
    emp = intent.get("employee") or {}

    # ── Check-then-Act: avoid 409 on duplicate email (unique constraint) ──────
    # Only check when a stable key is available; skip for name-only to save calls.
    existing_employee: dict | None = None
    if emp.get("email"):
        existing_employee = resolve_employee(client, email=emp["email"])
        if existing_employee:
            logger.info(f"Employee already exists id={existing_employee['id']} (email match), skipping creation")
            # Update department/admin role if needed on existing employee
            update_payload: dict = {"id": existing_employee["id"]}
            needs_update = False
            if emp.get("is_account_admin") and not existing_employee.get("isAccountAdmin"):
                update_payload["isAccountAdmin"] = True
                needs_update = True

    # Look up (or create) a department – use intent department_name if available
    dept_name = emp.get("department_name")
    dept_id: int | None = None
    if dept_name:
        # Try to find or create the specific department
        try:
            depts = client.get_list("/department", params={"name": dept_name, "count": 10})
            if depts:
                dept_id = depts[0]["id"]
            else:
                dept = client.post_value("/department", json={"name": dept_name})
                if dept:
                    dept_id = dept["id"]
        except Exception:
            pass
    if not dept_id:
        dept_id = _get_default_department_id(client)

    # Apply pending updates on existing employee (department, admin, etc.)
    if existing_employee:
        if dept_id:
            update_payload["department"] = {"id": dept_id}
            needs_update = True
        if needs_update:
            try:
                client.put(f"/employee/{existing_employee['id']}", json=update_payload)
                logger.info(f"Updated existing employee id={existing_employee['id']}")
            except Exception as exc:
                logger.debug(f"Employee update failed: {exc}")

    new_employee: dict | None = None
    if not existing_employee:
        # Generate email if missing: firstname.lastname@example.org with accent→ASCII
        email = emp.get("email")
        first_name = emp.get("first_name") or ""
        last_name = emp.get("last_name") or ""
        if not email and first_name and last_name:
            import unicodedata
            _ACCENT_MAP = {
                'ø': 'o', 'å': 'a', 'æ': 'ae', 'ü': 'u', 'ö': 'o', 'ä': 'a',
                'é': 'e', 'è': 'e', 'ê': 'e', 'ñ': 'n', 'ß': 'ss', 'ç': 'c',
                'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'í': 'i', 'ì': 'i',
                'î': 'i', 'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ú': 'u',
                'ù': 'u', 'û': 'u', 'ý': 'y', 'ð': 'd', 'þ': 'th',
            }
            def _ascii_name(s: str) -> str:
                s = s.lower().strip()
                result = []
                for ch in s:
                    if ch in _ACCENT_MAP:
                        result.append(_ACCENT_MAP[ch])
                    elif ch.isascii() and ch.isalpha():
                        result.append(ch)
                    elif ch in (' ', '-'):
                        result.append(ch)
                    else:
                        decomp = unicodedata.normalize('NFD', ch)
                        ascii_ch = ''.join(c for c in decomp if unicodedata.category(c) != 'Mn')
                        result.append(ascii_ch if ascii_ch else '')
                return ''.join(result).strip()
            fn_ascii = _ascii_name(first_name).replace(' ', '.').replace('-', '.')
            ln_ascii = _ascii_name(last_name).replace(' ', '.').replace('-', '.')
            if fn_ascii and ln_ascii:
                email = f"{fn_ascii}.{ln_ascii}@example.org"
                logger.info(f"Generated email for employee: {email}")

        has_email = bool(email)
        # Always use EXTENDED for proper system access
        payload: dict = {"userType": "EXTENDED" if has_email else "NO_ACCESS"}
        if emp.get("first_name"):
            payload["firstName"] = emp["first_name"]
        if emp.get("last_name"):
            payload["lastName"] = emp["last_name"]
        if has_email:
            payload["email"] = email
        if emp.get("phone"):
            payload["phoneNumberMobile"] = emp["phone"]
        if emp.get("is_account_admin"):
            payload["isAccountAdmin"] = True
        # Always set dateOfBirth – Tripletex requires it for employment creation
        payload["dateOfBirth"] = emp.get("date_of_birth") or "1990-01-01"
        if emp.get("national_id_number"):
            payload["nationalIdentityNumber"] = emp["national_id_number"]
        if emp.get("employee_number"):
            payload["employeeNumber"] = str(emp["employee_number"])
        if emp.get("bank_account_number"):
            payload["bankAccountNumber"] = str(emp["bank_account_number"])
        if emp.get("comments"):
            payload["comments"] = emp["comments"]
        if dept_id:
            payload["department"] = {"id": dept_id}
        # Address
        addr_parts: dict = {}
        if emp.get("address"):
            addr_parts["addressLine1"] = emp["address"]
        if emp.get("postal_code"):
            addr_parts["postalCode"] = emp["postal_code"]
        if emp.get("city"):
            addr_parts["city"] = emp["city"]
        if addr_parts:
            payload["address"] = addr_parts

        new_employee = _post_value_with_heal(client, "/employee", payload)
        if new_employee:
            logger.info(f"Created employee id={new_employee.get('id')}")

    employee = existing_employee or new_employee
    if not employee:
        return

    # ── Create employment record ───────────────────────────────────────
    start_date = emp.get("start_date") or "2026-01-01"  # Competition year default
    employment_id: int | None = None
    try:
        emp_record = client.post_value("/employee/employment", json={
            "employee": {"id": employee["id"]},
            "startDate": start_date,
            "isMainEmployer": True,
        })
        if emp_record:
            employment_id = emp_record.get("id")
            logger.info(f"Created employment id={employment_id} for employee {employee['id']}")
    except Exception as exc:
        logger.warning(f"Employment creation failed (will retry): {exc}")
        # Tripletex may require dateOfBirth – ensure it's set via PUT then retry
        try:
            dob = emp.get("date_of_birth") or "1990-01-01"
            client.put(f"/employee/{employee['id']}", json={
                "id": employee["id"],
                "dateOfBirth": dob,
            })
            emp_record = client.post_value("/employee/employment", json={
                "employee": {"id": employee["id"]},
                "startDate": start_date,
                "isMainEmployer": True,
            })
            if emp_record:
                employment_id = emp_record.get("id")
                logger.info(f"Created employment id={employment_id} after dateOfBirth fix")
        except Exception as inner_exc:
            logger.debug(f"Employment creation retry failed: {inner_exc}")
        # Try to fetch existing employment as last resort
        if not employment_id:
            try:
                existing_emp_list = client.get_list(
                    "/employee/employment",
                    params={"employeeId": employee["id"], "count": 5},
                )
                if existing_emp_list:
                    employment_id = existing_emp_list[0]["id"]
            except Exception:
                pass

    # ── Create employment details (salary, occupation, work %) ────────
    annual_salary = emp.get("annual_salary")
    work_percent = emp.get("work_percent")
    job_title = emp.get("job_title") or emp.get("title")
    occupation_code = emp.get("occupation_code")

    # At a minimum we need an employment record ID to attach details
    if employment_id and (annual_salary or work_percent or job_title or occupation_code):
        details_payload: dict = {
            "employment": {"id": employment_id},
            "date": start_date,
            "employmentType": "ORDINARY",
            "employmentForm": "PERMANENT",
            "remunerationType": "MONTHLY_WAGE",
            "workingHoursScheme": "NOT_SHIFT",
            "percentageOfFullTimeEquivalent": float(work_percent) if work_percent else 100.0,
        }
        if annual_salary:
            details_payload["annualSalary"] = float(annual_salary)

        # Look up occupation code — try explicit code first, then by job title
        occ_resolved = False
        if occupation_code:
            try:
                codes = client.get_list(
                    "/employee/employment/occupationCode",
                    params={"code": str(occupation_code), "count": 100},
                )
                if codes:
                    # API does prefix search — filter client-side for exact match
                    exact = [c for c in codes if str(c.get("code", "")) == str(occupation_code)]
                    match = exact[0] if exact else codes[0]
                    details_payload["occupationCode"] = {"id": match["id"]}
                    occ_resolved = True
                    logger.info(f"Resolved occupationCode id={match['id']} code={match.get('code')} for requested={occupation_code}")
            except Exception as exc:
                logger.debug(f"Occupation code lookup by code failed: {exc}")

        if not occ_resolved and job_title:
            # Try full title, then progressively shorter/generic terms
            search_terms = [job_title]
            # Add root word fallbacks (e.g. "Seniorutvikler" → "utvikler")
            words = job_title.split()
            if len(words) > 1:
                search_terms.append(words[-1])  # last word
            # Common Norwegian equivalents for foreign job titles
            _JOB_TITLE_MAP = {
                "developer": "utvikler", "engineer": "ingeniør", "manager": "leder",
                "consultant": "konsulent", "accountant": "regnskapsfører",
                "chef de projet": "prosjektleder", "projektleiter": "prosjektleder",
                "jefe de proyecto": "prosjektleder", "gerente": "leder",
                "conseiller": "rådgiver", "berater": "rådgiver", "asesor": "rådgiver",
                # German job titles
                "entwickler": "utvikler", "softwareentwickler": "utvikler",
                "buchhalter": "regnskapsfører", "sachbearbeiter": "saksbehandler",
                "sekretär": "sekretær", "verwaltung": "administrasjon",
                "leiter": "leder", "abteilungsleiter": "avdelingsleder",
                "geschäftsführer": "daglig leder", "analyst": "analytiker",
                # Portuguese/Spanish job titles
                "desenvolvedor": "utvikler", "contador": "regnskapsfører",
                "gerente de projeto": "prosjektleder", "analista": "analytiker",
                "ingeniero": "ingeniør", "contador público": "regnskapsfører",
                # French
                "développeur": "utvikler", "comptable": "regnskapsfører",
                "analyste": "analytiker", "secrétaire": "sekretær",
            }
            for foreign, norwegian in _JOB_TITLE_MAP.items():
                if foreign in job_title.lower():
                    search_terms.append(norwegian)
                    break

            for term in search_terms:
                try:
                    codes = client.get_list(
                        "/employee/employment/occupationCode",
                        params={"nameNO": term, "count": 20},
                    )
                    if codes:
                        details_payload["occupationCode"] = {"id": codes[0]["id"]}
                        logger.info(f"Resolved occupationCode id={codes[0]['id']} for {term!r} (from {job_title!r})")
                        break
                except Exception as exc:
                    logger.debug(f"Occupation code lookup for {term!r} failed: {exc}")

        try:
            det_result = _post_value_with_heal(client, "/employee/employment/details", details_payload)
            if det_result:
                logger.info(f"Created employment details id={det_result.get('id')}")
        except Exception as exc:
            logger.warning(f"Employment details creation failed: {exc}")

    # ── Set standard work hours (POST /employee/standardTime) ─────────
    hours_per_day = emp.get("hours_per_day")
    if not hours_per_day and employment_id:
        # Default Norwegian standard: 7.5 hours/day for full-time
        hours_per_day = 7.5
    if hours_per_day and employee:
        try:
            st_payload = {
                "employee": {"id": employee["id"]},
                "fromDate": start_date,
                "hoursPerDay": float(hours_per_day),
            }
            st_result = client.post_value("/employee/standardTime", json=st_payload)
            if st_result:
                logger.info(f"Created standard time id={st_result.get('id')} hoursPerDay={hours_per_day}")
        except Exception as exc:
            logger.debug(f"Standard time creation failed: {exc}")

    # ── Grant admin entitlements via correct API ──────────────────────
    if emp.get("is_account_admin") and employee:
        try:
            client.put(
                f"/employee/entitlement/:grantEntitlementsByTemplate",
                params={"employeeId": employee["id"], "template": "ALL_PRIVILEGES"},
            )
            logger.info(f"Granted ALL_PRIVILEGES to employee id={employee['id']}")
        except Exception as exc:
            logger.debug(f"Entitlement grant failed: {exc}")


def _update_employee(intent: dict, client: TripletexClient) -> None:
    emp = intent.get("employee") or {}
    identifier = emp.get("identifier") or f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip()

    existing = _find_employee(client, identifier)
    if not existing:
        logger.error(f"Employee not found: {identifier!r}")
        return

    emp_id = existing["id"]
    # Merge: start with current values, apply changes
    payload: dict = {
        "id": emp_id,
        "firstName": existing.get("firstName"),
        "lastName": existing.get("lastName"),
        "email": existing.get("email"),
    }
    if emp.get("first_name"):
        payload["firstName"] = emp["first_name"]
    if emp.get("last_name"):
        payload["lastName"] = emp["last_name"]
    if emp.get("email"):
        payload["email"] = emp["email"]
    if emp.get("phone"):
        payload["phoneNumberMobile"] = emp["phone"]

    result = client.put_value(f"/employee/{emp_id}", json=payload)
    if result:
        logger.info(f"Updated employee id={emp_id}")

    # Note: grantEntitlementsByTemplate is [BETA] and returns 403, skipped.


def _delete_employee(intent: dict, client: TripletexClient) -> None:
    emp = intent.get("employee") or {}
    identifier = emp.get("identifier") or f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip()

    existing = _find_employee(client, identifier)
    if not existing:
        logger.error(f"Employee not found for delete: {identifier!r}")
        return

    try:
        client.delete(f"/employee/{existing['id']}")
        logger.info(f"Deleted employee id={existing['id']}")
    except Exception as exc:
        logger.warning(f"Delete employee {existing['id']} failed (endpoint may not exist): {exc}")


# ======================================================================
# Customer workflows
# ======================================================================

def _create_customer(intent: dict, client: TripletexClient) -> None:
    cust = intent.get("customer") or {}

    is_supplier = bool(cust.get("is_supplier"))
    # A pure supplier should NOT be flagged as a customer
    payload: dict = {
        "isCustomer": not is_supplier,
        "isSupplier": is_supplier,
    }
    if cust.get("name"):
        payload["name"] = cust["name"]
    if cust.get("email"):
        payload["email"] = cust["email"]
    if cust.get("phone"):
        payload["phoneNumber"] = cust["phone"]
    if cust.get("phone_mobile"):
        payload["phoneNumberMobile"] = cust["phone_mobile"]
    if cust.get("org_number"):
        payload["organizationNumber"] = cust["org_number"]
    if cust.get("website"):
        payload["website"] = cust["website"]
    if cust.get("description"):
        payload["description"] = cust["description"]
    if cust.get("language"):
        lang = cust["language"].upper()
        # Tripletex only accepts NO/EN for language field
        if lang not in ("NO", "EN"):
            lang = "EN"  # international fallback
        payload["language"] = lang
    if cust.get("is_private_individual"):
        payload["isPrivateIndividual"] = True
    if cust.get("invoices_due_in"):
        payload["invoicesDueIn"] = int(cust["invoices_due_in"])
    if cust.get("invoices_due_in_type"):
        payload["invoicesDueInType"] = cust["invoices_due_in_type"]

    # Address: always set BOTH physicalAddress AND postalAddress (system checks both)
    addr_parts: dict = {}
    if cust.get("address"):
        addr_parts["addressLine1"] = cust["address"]
    if cust.get("postal_code"):
        addr_parts["postalCode"] = cust["postal_code"]
    if cust.get("city"):
        addr_parts["city"] = cust["city"]
    if addr_parts:
        payload["physicalAddress"] = addr_parts
        payload["postalAddress"] = dict(addr_parts)

    # Delivery address (if separate from physical)
    delivery_parts: dict = {}
    if cust.get("delivery_address"):
        delivery_parts["addressLine1"] = cust["delivery_address"]
    if cust.get("delivery_postal_code"):
        delivery_parts["postalCode"] = cust["delivery_postal_code"]
    if cust.get("delivery_city"):
        delivery_parts["city"] = cust["delivery_city"]
    if delivery_parts:
        payload["deliveryAddress"] = delivery_parts

    # Invoice sending method
    if cust.get("invoice_email"):
        payload["invoiceEmail"] = cust["invoice_email"]
    if cust.get("invoice_send_method"):
        payload["invoiceSendMethod"] = cust["invoice_send_method"]

    # Account manager (search /employee)
    if cust.get("account_manager"):
        try:
            emps = client.get_list("/employee", params={
                "firstName": cust["account_manager"].split()[0] if " " in cust["account_manager"] else cust["account_manager"],
                "count": 10, "fields": "id,firstName,lastName",
            })
            if emps:
                payload["accountManager"] = {"id": emps[0]["id"]}
        except Exception:
            pass

    # Optimistic create (no prior GET = maximum efficiency on fresh sandboxes).
    # On 409 Conflict (duplicate entity), fall back to a name lookup so the
    # caller can still obtain the entity ID without crashing.
    try:
        result = _post_value_with_heal(client, "/customer", payload)
        if result:
            logger.info(f"Created customer id={result.get('id')}")
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 409:
            existing = resolve_customer(client, name=cust.get("name"))
            if existing:
                logger.info(f"Customer already exists id={existing['id']} (409 conflict)")
                # Update existing customer with any fields from the prompt
                updates: dict = {"id": existing["id"], "name": existing.get("name"), "isCustomer": not is_supplier, "isSupplier": is_supplier}
                if cust.get("email"):
                    updates["email"] = cust["email"]
                if cust.get("phone"):
                    updates["phoneNumber"] = cust["phone"]
                if cust.get("org_number"):
                    updates["organizationNumber"] = cust["org_number"]
                if any(k in updates for k in ("email", "phoneNumber", "organizationNumber")):
                    client.put_value(f"/customer/{existing['id']}", json=updates)
                    logger.info(f"Updated existing customer id={existing['id']} with prompt fields")
        else:
            raise


def _update_customer(intent: dict, client: TripletexClient) -> None:
    cust = intent.get("customer") or {}
    identifier = cust.get("identifier") or cust.get("name")

    existing = _find_customer(client, identifier)
    if not existing:
        logger.error(f"Customer not found: {identifier!r}")
        return

    cust_id = existing["id"]
    payload: dict = {
        "id": cust_id,
        "name": existing.get("name"),
        "email": existing.get("email"),
        "isCustomer": True,
    }
    if cust.get("name"):
        payload["name"] = cust["name"]
    if cust.get("email"):
        payload["email"] = cust["email"]
    if cust.get("phone"):
        payload["phoneNumber"] = cust["phone"]
    if cust.get("org_number"):
        payload["organizationNumber"] = cust["org_number"]

    result = client.put_value(f"/customer/{cust_id}", json=payload)
    if result:
        logger.info(f"Updated customer id={cust_id}")


def _delete_customer(intent: dict, client: TripletexClient) -> None:
    cust = intent.get("customer") or {}
    identifier = cust.get("identifier") or cust.get("name")

    existing = _find_customer(client, identifier)
    if not existing:
        logger.error(f"Customer not found for delete: {identifier!r}")
        return

    try:
        client.delete(f"/customer/{existing['id']}")
        logger.info(f"Deleted customer id={existing['id']}")
    except Exception as exc:
        logger.warning(f"Delete customer {existing['id']} failed (may be BETA-blocked): {exc}")


# ======================================================================
# Product workflow
# ======================================================================

def _create_product(intent: dict, client: TripletexClient) -> None:
    prod = intent.get("product") or {}

    payload: dict = {}
    if prod.get("name"):
        payload["name"] = prod["name"]
    if prod.get("number"):
        payload["number"] = prod["number"]
        # Check if product already exists by number
        existing = client.get_list("/product", params={"number": str(prod["number"]), "count": 1})
        if existing:
            logger.info(f"Product with number={prod['number']} already exists id={existing[0].get('id')}")
            return
    if prod.get("price_excl_vat") is not None:
        payload["priceExcludingVatCurrency"] = prod["price_excl_vat"]
    if prod.get("description"):
        payload["description"] = prod["description"]

    # Resolve vatType by rate (25 → code 3, 15 → code 31, 0 → code 6)
    vat_rate = prod.get("vat_rate")
    if vat_rate is not None:
        vat_number = {25: "3", 15: "31", 0: "6"}.get(int(vat_rate), "3")
        vt = client.get_vat_type(vat_number)
        if vt:
            payload["vatType"] = {"id": vt["id"]}

    # Calculate priceIncludingVatCurrency from priceExcludingVatCurrency + VAT rate
    if prod.get("price_excl_vat") is not None and vat_rate is not None:
        excl = float(prod["price_excl_vat"])
        rate_int = int(vat_rate)
        multiplier = {25: 1.25, 15: 1.15, 0: 1.0}.get(rate_int, 1.25)
        payload["priceIncludingVatCurrency"] = round(excl * multiplier, 2)

    result = _post_value_with_heal(client, "/product", payload)
    if result:
        logger.info(f"Created product id={result.get('id')}")


# ======================================================================
# Invoice workflow  (customer → order → invoice)
# ======================================================================

# Valid Norwegian BBAN (MOD-11 verified: weights [5,4,3,2,7,6,5,4,3,2], check=0)
_FALLBACK_BBAN = "00001234560"

# Module-level cache: tracks (base_url, session_token) pairs where the bank
# account has been verified.  Keying by token ensures fresh evaluation sessions
# (new token → potentially fresh sandbox) always re-check.
_BANK_ACCOUNT_ENSURED: set[tuple[str, str]] = set()


def _ensure_bank_account(client: TripletexClient) -> None:
    """Ensure the company has a bank account number on its invoice ledger account.

    Tripletex refuses to create invoices until the company registers a bank account
    number (kontonummer).  In fresh sandbox accounts — including those used by the
    competition evaluator — this field is blank.  We detect that and set a valid
    Norwegian BBAN so that POST /invoice stops returning 422
    "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer."
    """
    _cache_key = (client.base_url, client.session_token)
    if _cache_key in _BANK_ACCOUNT_ENSURED:
        return
    try:
        accounts = client.get_list(
            "/ledger/account",
            params={
                "isBankAccount": "true",
                "count": 20,
                "fields": "id,number,name,isBankAccount,isInvoiceAccount,bankAccountNumber,version",
            },
        )
        # The primary invoice account is typically ledger 1920 (Bankinnskudd)
        target = next(
            (
                a for a in accounts
                if a.get("isInvoiceAccount") and a.get("isBankAccount") and not a.get("bankAccountNumber")
            ),
            None,
        )
        if target is None:
            _BANK_ACCOUNT_ENSURED.add(_cache_key)  # already configured
            return
        acct_id = target["id"]
        client.put_value(
            f"/ledger/account/{acct_id}",
            json={
                "id": acct_id,
                "version": target.get("version", 0),
                "number": target.get("number", 1920),
                "name": target.get("name", "Bankinnskudd"),
                "type": "ASSETS",
                "isBankAccount": True,
                "isInvoiceAccount": True,
                "bankAccountNumber": _FALLBACK_BBAN,
                "bankAccountCountry": {"id": 161},  # Norway
            },
        )
        _BANK_ACCOUNT_ENSURED.add(_cache_key)
        logger.info(
            f"Registered bank account {_FALLBACK_BBAN} on ledger account {acct_id} "
            f"(number={target.get('number')}) — invoice creation now unblocked"
        )
    except Exception as exc:
        logger.warning(f"_ensure_bank_account: {exc}")


def _create_invoice(intent: dict, client: TripletexClient) -> None:
    cust_data = intent.get("customer") or {}
    inv_data = intent.get("invoice") or {}

    # ── Step 0: ensure the company has a bank account number ─────────
    # Tripletex blocks invoice creation on fresh accounts with no registered
    # kontonummer.  This is a one-time no-op once the account is configured.
    _ensure_bank_account(client)

    # ── Step 1: create customer (optimistic — fresh account is always empty) ──
    # Skip the GET lookup to save an API call and preserve the efficiency bonus.
    # If POST fails with 409/422 (customer exists), fall back to a GET lookup.
    customer = None
    if cust_data.get("name"):
        cust_payload: dict = {"isCustomer": True, "name": cust_data["name"]}
        if cust_data.get("email"):
            cust_payload["email"] = cust_data["email"]
        if cust_data.get("org_number"):
            cust_payload["organizationNumber"] = cust_data["org_number"]
        try:
            customer = _post_value_with_heal(client, "/customer", cust_payload)
        except Exception:
            customer = _find_customer(client, cust_data["name"])

    if not customer:
        logger.error("Cannot create invoice: no customer resolved")
        return

    customer_id = customer["id"]

    # ── Step 2: build order lines ─────────────────────────────────────
    invoice_date = inv_data.get("date") or TODAY
    due_days = inv_data.get("due_days") if inv_data.get("due_days") is not None else 14
    due_date = (date.fromisoformat(invoice_date) + timedelta(days=due_days)).isoformat()

    # Parser may place order_lines at top-level intent OR inside invoice sub-dict
    raw_lines = inv_data.get("order_lines") or intent.get("order_lines") or []
    order_lines: list[dict] = []

    # ── Resolve VAT types upfront (uses session-level cache) ──────────
    _vat_rate_to_number = {25: "3", 15: "31", 12: "33", 0: "6"}

    def _resolve_vat_type(rate: int | None) -> dict | None:
        if rate is None:
            return None
        vat_num = _vat_rate_to_number.get(rate)
        if not vat_num:
            return None
        vt = client.get_vat_type(vat_num)
        return {"id": vt["id"]} if vt else None

    if raw_lines:
        for line in raw_lines:
            ol: dict = {
                "description": line.get("description", ""),
                "count": line.get("count", 1),
                "unitPriceExcludingVatCurrency": line.get("unit_price_excl_vat", 0),
            }
            # Set VAT type from parsed vat_rate
            vat_rate = line.get("vat_rate")
            if vat_rate is not None:
                vt = _resolve_vat_type(int(vat_rate))
                if vt:
                    ol["vatType"] = vt
            # Create product if product_number specified
            pn = line.get("product_number")
            if pn:
                try:
                    # Check if product already exists first to avoid 422
                    existing_prods = client.get_list(
                        "/product", params={"number": str(pn), "count": 1}
                    )
                    if existing_prods:
                        ol["product"] = {"id": existing_prods[0]["id"]}
                    else:
                        excl_price = float(line.get("unit_price_excl_vat", 0))
                        prod_payload: dict = {
                            "name": line.get("description", f"Product {pn}"),
                            "number": int(pn),
                            "priceExcludingVatCurrency": excl_price,
                        }
                        # Always include priceIncludingVatCurrency — required by Tripletex
                        vat_mult = 1.0
                        if vat_rate is not None:
                            vt_ref = _resolve_vat_type(int(vat_rate))
                            if vt_ref:
                                prod_payload["vatType"] = vt_ref
                            vat_mult = {25: 1.25, 15: 1.15, 12: 1.12, 0: 1.0}.get(int(vat_rate), 1.25)
                        else:
                            vat_mult = 1.25
                        prod_payload["priceIncludingVatCurrency"] = round(excl_price * vat_mult, 2)
                        prod = _post_value_with_heal(client, "/product", prod_payload)
                        if prod:
                            ol["product"] = {"id": prod["id"]}
                        else:
                            # POST failed (likely 422 – product may already exist with different search)
                            existing_retry = client.get_list(
                                "/product", params={"number": str(pn), "count": 1}
                            )
                            if existing_retry:
                                ol["product"] = {"id": existing_retry[0]["id"]}
                except Exception as exc:
                    logger.warning(f"Failed to create product {pn}: {exc}")
            order_lines.append(ol)
    elif inv_data.get("amount"):
        order_lines.append(
            {
                "description": inv_data.get("description") or "Service",
                "count": 1,
                "unitPriceExcludingVatCurrency": inv_data["amount"],
            }
        )

    # ── Arithmetic validation ─────────────────────────────────────────
    # When a total amount is stated alongside multiple extracted lines,
    # verify the math before hitting the API.  A mismatch means the LLM
    # mis-extracted line items; falling back to a single corrected line
    # preserves the correct invoice total instead of submitting bad data.
    stated_total = inv_data.get("amount")
    if order_lines and stated_total is not None:
        try:
            expected = float(stated_total)
            computed = sum(
                float(line.get("unitPriceExcludingVatCurrency", 0))
                * float(line.get("count", 1))
                for line in order_lines
            )
            if expected > 0 and abs(computed - expected) / expected > 0.01:
                logger.warning(
                    f"Order lines sum ({computed:.2f}) ≠ stated total ({expected:.2f}) "
                    f"— collapsing to single corrected line"
                )
                order_lines = [{
                    "description": order_lines[0].get("description", "Service"),
                    "count": 1,
                    "unitPriceExcludingVatCurrency": expected,
                }]
        except (TypeError, ValueError, ZeroDivisionError):
            pass  # non-numeric total — skip validation

    # ── Step 3: create order ──────────────────────────────────────────
    order_payload: dict = {
        "customer": {"id": customer_id},
        "orderDate": invoice_date,
        "deliveryDate": invoice_date,
    }

    order = _post_value_with_heal(client, "/order", order_payload)
    if not order:
        logger.error("Failed to create order")
        return

    order_id = order["id"]

    # Tripletex ignores inline orderLines on POST /order — must add separately.
    # Use POST /order/orderline/list for batch creation (one write call) when
    # multiple lines exist; fall back to individual POST for single line.
    lines_added = 0
    if len(order_lines) > 1:
        try:
            batch = [{**line, "order": {"id": order_id}} for line in order_lines]
            result = client.post("/order/orderline/list", json=batch)
            if result:
                lines_added = len(order_lines)
        except Exception as exc:
            logger.warning(f"Batch order line creation failed: {exc}")
    if lines_added == 0:
        for line in order_lines:
            try:
                client.post("/order/orderline", json={**line, "order": {"id": order_id}})
                lines_added += 1
            except Exception as exc:
                logger.warning(f"Failed to add order line: {exc}")
    if order_lines and lines_added == 0:
        logger.error(f"Could not add any order lines to order {order_id} — invoice will likely fail")

    # ── Step 4: create invoice and send it ────────────────────────────
    # Always create with sendToCustomer=false to avoid 422 from missing
    # email configuration.  Then send explicitly via /:send with
    # overrideEmailAddress so it always works.
    invoice = None
    invoice_body = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "orders": [{"id": order_id}],
    }
    invoice = None
    try:
        invoice = _post_value_with_heal(
            client,
            "/invoice?sendToCustomer=false",
            invoice_body,
        )
    except Exception:
        raise

    if invoice:
        invoice_id = invoice.get("id")
        logger.info(
            f"Created invoice id={invoice_id} number={invoice.get('invoiceNumber')}"
        )
        # Always send explicitly — invoice was created as draft
        sent_ok = False
        for send_type in ("EMAIL", "MANUAL"):
            try:
                send_params: dict = {"sendType": send_type}
                if send_type == "EMAIL":
                    send_params["overrideEmailAddress"] = "noreply@example.com"
                client.put(
                    f"/invoice/{invoice_id}/:send",
                    params=send_params,
                )
                logger.info(f"Sent invoice id={invoice_id} via {send_type}")
                sent_ok = True
                break
            except Exception as exc:
                logger.warning(f"Invoice :send ({send_type}) failed for id={invoice_id}: {exc}")
        if not sent_ok:
            logger.warning(f"All :send attempts failed for invoice {invoice_id}")

        # ── Step 5: register payment if prompt asks for it ────────────
        # Some prompts combine create+pay: "...og registrer full betaling"
        # Check intent payment data first, then fall back to keyword detection.
        pay_data = intent.get("payment") or {}
        pay_amount = pay_data.get("amount")

        if not pay_amount:
            # Keyword detection across 7 supported languages
            _PAY_KEYWORDS = (
                "registrer full betaling", "registrer betaling",       # nb/nn
                "register full payment", "register payment",            # en
                "registre o pagamento", "pagamento total",              # pt
                "registrar pago", "pago total",                         # es
                "zahlung registrieren", "vollständige zahlung",         # de
                "enregistrer le paiement", "paiement total",            # fr
            )
            raw = intent.get("_raw_prompt", "").lower()
            if any(kw in raw for kw in _PAY_KEYWORDS):
                # Use actual invoice amount (incl VAT) from the invoice response
                # rather than excl-VAT order line totals.
                pay_amount = (
                    invoice.get("amountOutstanding")
                    or invoice.get("amount")
                    or invoice.get("amountCurrency")
                )
                if pay_amount:
                    pay_amount = float(pay_amount)
                if not pay_amount and inv_data.get("amount"):
                    pay_amount = float(inv_data["amount"])

        if pay_amount and invoice_id:
            payment_type_id = client.get_payment_type()
            pay_date = pay_data.get("date") or TODAY
            if payment_type_id:
                try:
                    client.put(
                        f"/invoice/{invoice_id}/:payment"
                        f"?paymentDate={pay_date}"
                        f"&paymentTypeId={payment_type_id}"
                        f"&paidAmount={pay_amount}"
                    )
                    logger.info(
                        f"Registered payment of {pay_amount} on invoice id={invoice_id}"
                    )
                except Exception as exc:
                    logger.warning(f"Payment registration failed for invoice {invoice_id}: {exc}")


# ======================================================================
# Payment workflow
# ======================================================================

def _register_payment(intent: dict, client: TripletexClient) -> None:
    inv_data = intent.get("invoice") or {}
    pay_data = intent.get("payment") or {}
    cust_data = intent.get("customer") or {}
    is_reversal = bool(pay_data.get("is_reversal"))

    # Ensure bank account is set so we can create invoices if needed
    _ensure_bank_account(client)

    # ── Find invoice ──────────────────────────────────────────────────
    invoice: dict | None = None

    # Try to locate by invoice number (numeric identifier)
    ident = inv_data.get("identifier")
    if ident and str(ident).isdigit():
        try:
            # Tripletex requires date range even when filtering by invoiceNumber
            date_from = (date.today() - timedelta(days=365 * 5)).isoformat()
            date_to = (date.today() + timedelta(days=365)).isoformat()
            results = client.get_list(
                "/invoice",
                params={
                    "invoiceNumber": ident,
                    "invoiceDateFrom": date_from,
                    "invoiceDateTo": date_to,
                    "count": 5,
                },
            )
            if results:
                invoice = results[0]
        except Exception:
            pass

    # Try by customer name/org → latest unpaid invoice (or any invoice for reversals)
    if not invoice:
        lookup = cust_data.get("identifier") or cust_data.get("name") or ident
        org_num = cust_data.get("org_number")
        customer = None
        if lookup or org_num:
            customer = resolve_customer(client, name=lookup, org_number=org_num, allow_fuzzy=True)
            if customer:
                invoice = _find_invoice_for_customer(client, customer["id"])

    # ── Fallback: create invoice if none found (skip for reversals) ──
    # The competition may expect us to create+pay when no invoice exists yet.
    if not invoice and not is_reversal:
        amount_hint = pay_data.get("amount") or inv_data.get("amount")
        if amount_hint:
            logger.info("No existing invoice found — creating invoice to register payment against")
            # Reuse create_invoice logic with synthesized intent
            synth_intent = {
                "task_type": "create_invoice",
                "customer": cust_data,
                "invoice": {
                    "date": pay_data.get("date") or TODAY,
                    "due_days": 0,  # immediately due
                    "amount": amount_hint,
                    "description": inv_data.get("description") or "Service",
                    "order_lines": inv_data.get("order_lines"),
                },
            }
            _create_invoice(synth_intent, client)
            # Now try to find the freshly created invoice
            lookup = cust_data.get("identifier") or cust_data.get("name")
            if lookup:
                customer = _find_customer(client, lookup)
                if customer:
                    invoice = _find_invoice_for_customer(client, customer["id"])

    if not invoice:
        logger.error("Could not find or create invoice for payment registration")
        return

    invoice_id = invoice["id"]
    pay_date = pay_data.get("date") or TODAY

    if is_reversal:
        # For reversals: use negative of the already-paid amount to "undo" entry
        amount_outstanding = float(invoice.get("amountOutstanding") or 0)
        amount_total = float(invoice.get("amount") or invoice.get("amountCurrency") or 0)
        # The paid portion = total - outstanding; negate to reverse
        paid_amount = amount_total - amount_outstanding
        if paid_amount > 0:
            amount = -paid_amount
        else:
            # If fully outstanding already, try explicit amount from intent
            amount = -(float(pay_data.get("amount") or amount_total))
        logger.info(f"Reversing payment on invoice {invoice_id}: amount={amount}")
    else:
        # For full payment: prefer amountOutstanding from the actual invoice
        # since the parser often extracts the "excl VAT" figure from the prompt
        # while the invoice total (and outstanding) is incl VAT.
        outstanding = invoice.get("amountOutstanding") or invoice.get("amountCurrency")
        parsed_amount = pay_data.get("amount")
        if outstanding:
            amount = float(outstanding)
        elif parsed_amount:
            amount = float(parsed_amount)
        else:
            amount = None

    # Get a valid payment type (required by the /:payment endpoint)
    payment_type_id = client.get_payment_type()

    if not payment_type_id:
        logger.error("No payment type found — cannot register payment")
        return
    if not amount:
        logger.error("No payment amount determined — cannot register payment")
        return

    # Correct Tripletex action endpoint: PUT /invoice/{id}/:payment with all params in query
    # (openapi.json: paymentDate, paymentTypeId, paidAmount are all required query params)
    payment_url = (
        f"/invoice/{invoice_id}/:payment"
        f"?paymentDate={pay_date}"
        f"&paymentTypeId={payment_type_id}"
        f"&paidAmount={amount}"
    )
    # For foreign currency: paidAmountCurrency is the amount in the invoice's currency
    amount_currency = pay_data.get("amount_currency")
    if amount_currency:
        payment_url += f"&paidAmountCurrency={amount_currency}"

    data = client.put(payment_url)
    result = (data or {}).get("value")
    if result:
        logger.info(f"Registered payment on invoice id={invoice_id}")

    # ── Post exchange rate difference (agio) voucher if applicable ──
    exchange_diff = pay_data.get("exchange_rate_difference")
    if exchange_diff:
        diff_amount = float(exchange_diff)
        if abs(diff_amount) > 0.01:
            # Positive diff = loss (we paid more NOK), negative = gain
            if diff_amount > 0:
                agio_acct_num, agio_name = "8160", "Valutakurstap"
            else:
                agio_acct_num, agio_name = "8060", "Valutakursgevinst"
                diff_amount = -diff_amount  # make positive for the debit

            agio_acct = _resolve_account(client, agio_acct_num)
            recv_acct = _resolve_account_chain(client, "1500", "1501")

            if agio_acct and recv_acct:
                # Get customer ID from the invoice for the AR posting (required by Tripletex)
                invoice_customer = (invoice or {}).get("customer") or {}
                agio_customer_id = invoice_customer.get("id")
                if not agio_customer_id and customer:
                    agio_customer_id = customer.get("id") if isinstance(customer, dict) else None

                recv_posting: dict = {
                    "row": 2, "account": {"id": recv_acct["id"]},
                    "amountGross": -diff_amount, "amountGrossCurrency": -diff_amount,
                    "description": agio_name,
                }
                # Account 1500 (AR) MUST include customer reference
                if agio_customer_id:
                    recv_posting["customer"] = {"id": agio_customer_id}

                try:
                    v = _post_value_with_heal(client, "/ledger/voucher", {
                        "date": pay_date,
                        "description": f"Valutakursdifferanse – {agio_name}",
                        "postings": [
                            {"row": 1, "account": {"id": agio_acct["id"]}, "amountGross": diff_amount, "amountGrossCurrency": diff_amount, "description": agio_name},
                            recv_posting,
                        ],
                    })
                    logger.info(f"Posted agio voucher id={v.get('id') if v else None} diff={diff_amount}")
                except Exception as exc:
                    logger.error(f"Agio voucher failed: {exc}")


# ======================================================================
# Credit note workflow
# ======================================================================

def _create_credit_note(intent: dict, client: TripletexClient) -> None:
    inv_data = intent.get("invoice") or {}
    cust_data = intent.get("customer") or {}

    # Find invoice
    invoice: dict | None = None
    customer: dict | None = None
    lookup = cust_data.get("identifier") or cust_data.get("name")
    org_no = cust_data.get("org_number")
    if lookup:
        customer = resolve_customer(client, lookup, org_number=org_no)
        if customer:
            invoice = _find_invoice_for_customer(client, customer["id"])

    if not invoice and not customer and org_no:
        customer = resolve_customer(client, None, org_number=org_no)
        if customer:
            invoice = _find_invoice_for_customer(client, customer["id"])

    if not invoice:
        ident = inv_data.get("identifier")
        if ident:
            try:
                date_from = (date.today() - timedelta(days=365 * 5)).isoformat()
                date_to = (date.today() + timedelta(days=365)).isoformat()
                results = client.get_list(
                    "/invoice",
                    params={
                        "invoiceDateFrom": date_from,
                        "invoiceDateTo": date_to,
                        "count": 100,
                    },
                )
                if results:
                    invoice = results[0]
            except Exception:
                pass

    # No invoice found → create one so we always have something to credit.
    # The competition sends "create credit note for [customer]" into a fresh
    # sandbox where no prior invoice exists; we create then immediately credit.
    if not invoice:
        logger.info("No invoice found for credit note – creating one first")
        _ensure_bank_account(client)
        amount = inv_data.get("amount") or 1000.0
        # Ensure customer data has email for the synthetic invoice
        cust_for_synth = dict(cust_data)
        if not cust_for_synth.get("email"):
            safe_name = (cust_for_synth.get("name") or "company").lower().replace(" ", "").replace(".", "")[:20]
            cust_for_synth["email"] = f"noreply@{safe_name}.example.com"
        synth_intent: dict = {
            "task_type": "create_invoice",
            "customer": cust_for_synth,
            "invoice": {
                "date": inv_data.get("date") or TODAY,
                "due_days": 14,
                "amount": amount,
                "description": inv_data.get("description") or "Service",
            },
        }
        _create_invoice(synth_intent, client)
        # Find the freshly-created invoice — search broadly
        if not customer and lookup:
            customer = resolve_customer(client, lookup, org_number=org_no)
        if not customer:
            # Fallback: get any customer in the sandbox
            try:
                all_custs = client.get_list("/customer", params={"count": 10, "fields": "id,name"})
                if all_custs:
                    customer = all_custs[0]
            except Exception:
                pass
        if customer:
            invoice = _find_invoice_for_customer(client, customer["id"])
        if not invoice:
            # Last resort: just get any invoice
            try:
                date_from = (date.today() - timedelta(days=365 * 5)).isoformat()
                date_to = (date.today() + timedelta(days=365)).isoformat()
                results = client.get_list("/invoice", params={"invoiceDateFrom": date_from, "invoiceDateTo": date_to, "count": 5})
                if results:
                    invoice = results[0]
            except Exception:
                pass

    if not invoice:
        logger.error("Could not find or create invoice for credit note")
        return

    credit_date = inv_data.get("date") or TODAY

    def _do_credit_note() -> dict | None:
        data = client.put(
            f"/invoice/{invoice['id']}/:createCreditNote"
            f"?date={credit_date}&sendToCustomer=false"
        )
        return (data or {}).get("value")

    def _send_then_credit() -> dict | None:
        """Send the invoice to move it out of DRAFT, then credit.
        
        Only try EMAIL with overrideEmailAddress to minimize 4xx errors.
        """
        try:
            client.put(
                f"/invoice/{invoice['id']}/:send",
                params={"sendType": "EMAIL", "overrideEmailAddress": "noreply@example.com"},
            )
        except Exception:
            pass  # send might fail but credit note might still work
        return _do_credit_note()

    try:
        result = _do_credit_note()
        if result:
            logger.info(f"Created credit note id={result.get('id')} for invoice id={invoice['id']}")
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 0
        if status_code in (404, 422):
            # Invoice needs to be sent before credit noting
            logger.warning(f"{status_code} on credit note for invoice {invoice['id']} – sending invoice first")
            try:
                result = _send_then_credit()
                if result:
                    logger.info(f"Created credit note id={result.get('id')} after send")
            except Exception as e2:
                logger.warning(f"Credit note failed after send: {e2}")
        else:
            logger.warning(f"Credit note failed with status {status_code}: {exc}")
    except Exception as exc:
        logger.warning(f"Unexpected credit note error: {exc}")


# ======================================================================
# Supplier invoice workflow  (incoming invoice FROM a vendor)
# ======================================================================

def _create_supplier_invoice(intent: dict, client: TripletexClient) -> None:
    """Register an incoming supplier invoice via POST /ledger/voucher with vendorInvoiceNumber."""
    supp = intent.get("customer") or {}
    inv = intent.get("invoice") or {}

    # ── Resolve department if mentioned ────────────────────────────────
    dept_id: int | None = None
    _notes_raw = intent.get("notes")
    # Parser may return notes as dict (e.g. {"department_name": "HR"}) or string
    if isinstance(_notes_raw, dict):
        dept_name = _notes_raw.get("department_name")
        _notes = str(_notes_raw)
    else:
        dept_name = None
        _notes = _notes_raw or ""
    _raw = (intent.get("_raw_prompt") or intent.get("parsed_entities", {}).get("_raw_prompt") or "").lower()
    if not dept_name:
        import re as _re
        m = _re.search(r"(?:avdeling|department|abteilung|départment|departamento)[:\s]+['\"]?([A-ZÆØÅa-zæøå][A-ZÆØÅa-zæøåé0-9 -]+)", _notes) or \
            _re.search(r"(?:avdeling|department|abteilung|départment|departamento)[:\s]+['\"]?([A-ZÆØÅa-zæøå][A-ZÆØÅa-zæøåé0-9 -]+)", _raw)
        if m:
            dept_name = m.group(1).strip().strip("'\".")
    if dept_name:
        try:
            depts = client.get_list("/department", params={"name": dept_name, "count": 10})
            match = next((d for d in depts if (d.get("name") or "").lower() == dept_name.lower()), None)
            if match:
                dept_id = match["id"]
            else:
                created = _post_value_with_heal(client, "/department", {"name": dept_name})
                if created:
                    dept_id = created["id"]
                    logger.info(f"Created department '{dept_name}' id={dept_id}")
        except Exception as exc:
            logger.warning(f"Department resolution failed: {exc}")

    # ── Find or create supplier via /supplier endpoint ─────────────────
    supplier: dict | None = None
    org_number = supp.get("org_number")
    if org_number:
        results = client.get_list("/supplier", params={"organizationNumber": org_number, "count": 5})
        supplier = results[0] if results else None
    if not supplier and supp.get("name"):
        # Search by name
        results = client.get_list("/supplier", params={"count": 100})
        name_lower = supp["name"].lower()
        supplier = next(
            (s for s in results if name_lower in (s.get("name") or "").lower()),
            None,
        )
    if not supplier and supp.get("name"):
        supp_payload: dict = {"name": supp["name"]}
        if org_number:
            supp_payload["organizationNumber"] = org_number
        if supp.get("email"):
            supp_payload["email"] = supp["email"]
        if supp.get("phone"):
            supp_payload["phoneNumber"] = supp["phone"]
        # Add address if available
        addr_parts: dict = {}
        if supp.get("address"):
            addr_parts["addressLine1"] = supp["address"]
        if supp.get("postal_code"):
            addr_parts["postalCode"] = supp["postal_code"]
        if supp.get("city"):
            addr_parts["city"] = supp["city"]
        if addr_parts:
            supp_payload["physicalAddress"] = addr_parts
            supp_payload["postalAddress"] = dict(addr_parts)
        try:
            supplier = _post_value_with_heal(client, "/supplier", supp_payload)
            logger.info(f"Created supplier id={supplier.get('id') if supplier else None}")
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (409, 422):
                if exc.response.status_code == 422 and "physicalAddress" in supp_payload:
                    # Retry without address (some setups reject it)
                    supp_payload.pop("physicalAddress", None)
                    supp_payload.pop("postalAddress", None)
                    try:
                        supplier = _post_value_with_heal(client, "/supplier", supp_payload)
                        logger.info(f"Created supplier (no address) id={supplier.get('id') if supplier else None}")
                    except Exception:
                        pass
                if not supplier:
                    results = client.get_list("/supplier", params={"count": 100})
                    name_lower = supp["name"].lower()
                    supplier = next(
                        (s for s in results if name_lower in (s.get("name") or "").lower()),
                        None,
                    )
            else:
                raise

    if not supplier:
        logger.error("Cannot create supplier invoice: no supplier resolved")
        return

    # ── Determine amounts ──────────────────────────────────────────────
    # Parser may provide amount (total incl VAT), amount_excl_vat, and vat_rate
    total_incl_vat = float(inv.get("amount") or 0)
    amount_excl_vat = inv.get("amount_excl_vat")
    vat_rate = float(inv.get("vat_rate") or 25) / 100

    # Fallback: extract amount from raw prompt text
    if not total_incl_vat and not amount_excl_vat:
        _raw = intent.get("_raw_prompt", "")
        if _raw:
            import re as _re2
            # Match patterns like "69950 NOK", "12 500 kr", "4.500,00"
            amt_m = _re2.search(r'(\d[\d\s.,]*\d)\s*(?:NOK|kr|nok)\b', _raw)
            if not amt_m:
                amt_m = _re2.search(r'(?:for|pour|für|por|para)\s+(\d[\d\s.,]*\d)\s', _raw)
            if amt_m:
                raw_amt = amt_m.group(1).replace(" ", "").replace(".", "").replace(",", ".")
                try:
                    total_incl_vat = float(raw_amt)
                    logger.info(f"Extracted supplier invoice amount from raw prompt: {total_incl_vat}")
                except ValueError:
                    pass

    if not total_incl_vat and not amount_excl_vat:
        total_incl_vat = 1000.0
        logger.warning("No amounts in supplier invoice prompt — using default 1000 NOK")

    if amount_excl_vat is not None:
        excl_vat = float(amount_excl_vat)
    elif total_incl_vat:
        excl_vat = round(total_incl_vat / (1 + vat_rate), 2)
    else:
        excl_vat = 0.0

    vat_amount = round(total_incl_vat - excl_vat, 2) if total_incl_vat else 0.0

    # ── Look up GL accounts (cached via client) ────────────────────────
    # Expense account: parser may provide account_code (e.g. "6340"), else try common ones
    expense_acct: dict | None = None
    parsed_acct = str(inv.get("account_code") or "").strip()
    if parsed_acct:
        expense_acct = _resolve_account(client, parsed_acct)
    if not expense_acct:
        expense_acct = _resolve_account_chain(client, "6300", "6340", "4000", "7000", "6600")

    # Input VAT account (2710)
    vat_acct = _resolve_account_chain(client, "2710", "2700")

    # Accounts payable (2400)
    ap_acct = _resolve_account_chain(client, "2400", "2401", "2430")

    if not expense_acct or not ap_acct:
        logger.error(f"Cannot create supplier invoice: missing GL accounts (expense={expense_acct}, ap={ap_acct})")
        return

    # ── Build postings ─────────────────────────────────────────────────
    invoice_date = inv.get("date") or TODAY
    # Clamp future dates — Tripletex rejects vouchers with dates beyond current period
    try:
        if date.fromisoformat(invoice_date) > date.today():
            invoice_date = TODAY
    except (ValueError, TypeError):
        invoice_date = TODAY
    desc = inv.get("description") or f"Supplier Invoice from {supp.get('name', 'Supplier')}"
    due_date_str = (date.fromisoformat(invoice_date) + timedelta(days=30)).isoformat()

    # ── Build voucher postings with vatType for auto-VAT ─────────────
    # NOTE: /incomingInvoice consistently returns 403 (permission denied)
    # in competition sandboxes. Go straight to /ledger/voucher to avoid
    # wasting a 4xx error on the attempt.
    # Look up voucherType for supplier invoices (Leverandørfaktura)
    voucher_type_id: int | None = None
    try:
        vt_list = client.get_list("/ledger/voucherType", params={"fields": "id,name", "count": 50})
        for vt in vt_list:
            vt_name = (vt.get("name") or "").lower()
            if "leverandør" in vt_name or "supplier" in vt_name:
                voucher_type_id = vt["id"]
                break
    except Exception:
        pass

    # Look up incoming VAT type (uses session-level cache)
    incoming_vat_type_id: int | None = None
    vat_rate_int = int(float(inv.get("vat_rate") or 25))
    vat_number_map = {25: "1", 15: "11", 0: "5"}
    vat_search_num = vat_number_map.get(vat_rate_int, "1")
    vt = client.get_vat_type(vat_search_num)
    if vt:
        incoming_vat_type_id = vt["id"]

    # Use vatType approach: expense posting with GROSS amount + vatType → system auto-splits VAT
    # Both postings on same row, same description, same supplier for Tripletex rules
    posting_desc = desc
    postings: list[dict] = []

    if total_incl_vat and expense_acct and ap_acct:
        exp_posting: dict = {
            "row": 1,
            "account": {"id": expense_acct["id"]},
            "amountGross": total_incl_vat,
            "amountGrossCurrency": total_incl_vat,
            "description": posting_desc,
            "supplier": {"id": supplier["id"]},
        }
        if incoming_vat_type_id:
            exp_posting["vatType"] = {"id": incoming_vat_type_id}
        if dept_id:
            exp_posting["department"] = {"id": dept_id}
        postings.append(exp_posting)

        ap_posting: dict = {
            "row": 1,
            "account": {"id": ap_acct["id"]},
            "amountGross": -total_incl_vat,
            "amountGrossCurrency": -total_incl_vat,
            "description": posting_desc,
            "supplier": {"id": supplier["id"]},
        }
        postings.append(ap_posting)

    if not postings:
        logger.error("Cannot create supplier invoice: no postings (amounts resolved to zero)")
        return

    voucher_payload: dict = {
        "date": invoice_date,
        "description": desc,
        "postings": postings,
    }
    if voucher_type_id:
        voucher_payload["voucherType"] = {"id": voucher_type_id}

    # vendorInvoiceNumber: from parser or extract from raw prompt
    vendor_inv_num = inv.get("invoice_number") or ""
    if not vendor_inv_num:
        import re as _re_inv
        _raw_text = intent.get("_raw_prompt", "")
        m = _re_inv.search(r'(?:INV|FAK|invoice|faktura|factura|rechnung|facture|fatura)[-.\s]*([\w-]+\d[\w-]*)', _raw_text, _re_inv.IGNORECASE)
        if m:
            vendor_inv_num = m.group(0).strip()
    if vendor_inv_num:
        voucher_payload["vendorInvoiceNumber"] = str(vendor_inv_num)

    result = None
    try:
        result = _post_value_with_heal(client, "/ledger/voucher", voucher_payload)
    except Exception as exc:
        logger.warning(f"Supplier invoice auto-VAT attempt failed: {exc}")

    # ── Fallback: manual 3-row VAT split when account is VAT-locked ──
    if not result and vat_acct and total_incl_vat:
        logger.info("Supplier invoice: retrying with manual 3-row VAT split (account may be VAT-locked)")
        manual_postings: list[dict] = [
            {
                "row": 1,
                "account": {"id": expense_acct["id"]},
                "amountGross": excl_vat,
                "amountGrossCurrency": excl_vat,
                "description": posting_desc,
                "supplier": {"id": supplier["id"]},
            },
            {
                "row": 2,
                "account": {"id": vat_acct["id"]},
                "amountGross": vat_amount,
                "amountGrossCurrency": vat_amount,
                "description": posting_desc,
            },
            {
                "row": 3,
                "account": {"id": ap_acct["id"]},
                "amountGross": -total_incl_vat,
                "amountGrossCurrency": -total_incl_vat,
                "description": posting_desc,
                "supplier": {"id": supplier["id"]},
            },
        ]
        if dept_id:
            manual_postings[0]["department"] = {"id": dept_id}
        voucher_payload_manual: dict = {
            "date": invoice_date,
            "description": desc,
            "postings": manual_postings,
        }
        if voucher_type_id:
            voucher_payload_manual["voucherType"] = {"id": voucher_type_id}
        if vendor_inv_num:
            voucher_payload_manual["vendorInvoiceNumber"] = str(vendor_inv_num)
        result = _post_value_with_heal(client, "/ledger/voucher", voucher_payload_manual)

    if result:
        logger.info(f"Created supplier invoice voucher id={result.get('id')} supplier={supplier.get('name')}")


# ======================================================================
# Travel expense workflows
# ======================================================================

def _create_travel_expense(intent: dict, client: TripletexClient) -> None:
    te = intent.get("travel_expense") or {}
    emp_data = intent.get("employee") or {}

    # ── Find employee ─────────────────────────────────────────────────
    identifier = (
        emp_data.get("identifier")
        or f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}".strip()
        or te.get("identifier")
    )

    employee = _find_employee(client, identifier) if identifier else None

    if not employee:
        # Fall back to first employee in the org
        employees = client.get_list(
            "/employee",
            params={"fields": "id,firstName,lastName", "count": 5},
        )
        employee = employees[0] if employees else None

    if not employee:
        logger.error("No employee found for travel expense")
        return

    from_date = te.get("from_date") or TODAY
    to_date = te.get("to_date") or from_date

    is_foreign = bool(te.get("is_foreign_travel"))
    destination = te.get("destination") or ""
    purpose = te.get("description") or "Business trip"

    travel_details: dict = {
        "departureDate": from_date,
        "returnDate": to_date,
        "purpose": purpose,
        "isDayTrip": from_date == to_date,
        "isForeignTravel": is_foreign,
        "isCompensationFromRates": True,
    }
    if destination:
        travel_details["destination"] = destination

    payload: dict = {
        "employee": {"id": employee["id"]},
        "title": te.get("description") or "Business trip",
        "date": from_date,  # required by some proxy versions
        "travelDetails": travel_details,
    }

    result = _post_value_with_heal(client, "/travelExpense", payload)
    if result:
        te_id = result.get("id")
        logger.info(f"Created travel expense id={te_id}")

        # ── Add individual costs (flight, taxi, hotel, etc.) ──────────
        costs_raw = te.get("costs") or []
        if costs_raw and te_id:
            try:
                cost_cats = client.get_list(
                    "/travelExpense/costCategory",
                    params={"count": 100},
                )
                # Fetch payment types dynamically (required field on cost)
                pay_types = client.get_list(
                    "/travelExpense/paymentType",
                    params={"count": 20},
                )
                pay_type_id = next(
                    (p["id"] for p in pay_types if not p.get("isInactive")),
                    None,
                )
                _COST_KW_MAP = {
                    "flight": ["fly ", "fly,", "flybi", "luftha", "avion", "airfare", "plane"],
                    "taxi": ["taxi"],
                    "hotel": ["hotell", "hotel"],
                    "bus": ["buss"],
                    "ferry": ["ferge", "ferry", "båt"],
                    "train": ["tog", "train", "zug"],
                    "food": ["mat ", "food", "restaur", "maten"],
                    "parking": ["parkering", "parking"],
                    "phone": ["telefon", "phone"],
                }
                cat_by_type: dict[str, int] = {}
                for cat in cost_cats:
                    if not cat.get("showOnTravelExpenses") or cat.get("isInactive"):
                        continue
                    desc = (cat.get("description") or "").lower()
                    for cost_type, keywords in _COST_KW_MAP.items():
                        if cost_type not in cat_by_type:
                            for kw in keywords:
                                if kw in desc + " ":
                                    cat_by_type[cost_type] = cat["id"]
                                    break
                # Fallback category: "Reisekostnad, ikke oppgavepliktig"
                fallback_cat = next(
                    (c["id"] for c in cost_cats if "reisekostnad" in (c.get("description") or "").lower()),
                    None,
                )
                for cost_item in costs_raw:
                    cost_type = str(cost_item.get("type") or "other").lower()
                    amount = float(cost_item.get("amount") or 0)
                    if not amount:
                        continue
                    cat_id = cat_by_type.get(cost_type) or fallback_cat
                    if not cat_id:
                        continue
                    cost_date = to_date or TODAY
                    cost_payload: dict = {
                        "travelExpense": {"id": te_id},
                        "costCategory": {"id": cat_id},
                        "amountCurrencyIncVat": amount,
                        "currency": {"id": 1},
                        "comments": cost_item.get("description") or cost_type,
                        "date": cost_date,
                        "isPaidByEmployee": True,
                    }
                    if pay_type_id:
                        cost_payload["paymentType"] = {"id": pay_type_id}
                    try:
                        client.post("/travelExpense/cost", json=cost_payload)
                        logger.info(f"Added cost {cost_type}={amount} to travel expense {te_id}")
                    except Exception as exc:
                        logger.warning(f"Failed to add cost {cost_type}: {exc}")
            except Exception as exc:
                logger.warning(f"Cost lookup/add failed: {exc}")

        # ── Add per diem compensation if specified ────────────────────
        per_diem_days = te.get("per_diem_days")
        per_diem_rate = te.get("per_diem_rate")
        if per_diem_days and te_id:
            try:
                # Look up rate category for per diem with proper filters
                rate_cat_id: int | None = None
                try:
                    rate_cats = client.get_list(
                        "/travelExpense/rateCategory",
                        params={
                            "type": "PER_DIEM",
                            "dateFrom": from_date,
                            "dateTo": to_date,
                            "isValidDomestic": "true" if not is_foreign else "false",
                            "isRequiresOvernightAccommodation": "true",
                            "fields": "id,name",
                            "count": 50,
                        },
                    )
                    if rate_cats:
                        rate_cat_id = rate_cats[0].get("id")
                except Exception:
                    # Fallback: broader search
                    try:
                        rate_cats = client.get_list(
                            "/travelExpense/rateCategory",
                            params={"count": 50, "fields": "id,name"},
                        )
                        if rate_cats:
                            rate_cat_id = rate_cats[0].get("id")
                    except Exception:
                        pass

                per_diem_payload: dict = {
                    "travelExpense": {"id": te_id},
                    "count": int(per_diem_days),
                    "overnightAccommodation": "HOTEL",
                }
                if rate_cat_id:
                    per_diem_payload["rateCategory"] = {"id": rate_cat_id}
                if per_diem_rate:
                    per_diem_payload["rate"] = float(per_diem_rate)
                if destination:
                    per_diem_payload["location"] = destination
                client.post("/travelExpense/perDiemCompensation", json=per_diem_payload)
                logger.info(f"Added per diem {per_diem_days} days to travel expense {te_id}")
            except Exception as exc:
                logger.warning(f"Per diem compensation failed: {exc}")

        # ── MANDATORY: Deliver then Approve ───────────────────────────
        if te_id:
            try:
                client.put("/travelExpense/:deliver", params={"id": te_id})
                logger.info(f"Delivered travel expense {te_id}")
            except Exception as exc:
                logger.warning(f"Travel expense deliver failed: {exc}")

            try:
                client.put("/travelExpense/:approve", params={"id": te_id})
                logger.info(f"Approved travel expense {te_id}")
            except Exception as exc:
                logger.warning(f"Travel expense approve failed: {exc}")


def _delete_travel_expense(intent: dict, client: TripletexClient) -> None:
    te = intent.get("travel_expense") or {}
    emp_data = intent.get("employee") or {}

    search_term = (
        te.get("identifier")
        or emp_data.get("identifier")
        or f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}".strip()
    )

    expenses = client.get_list(
        "/travelExpense",
        params={"fields": "id,comment,employee", "count": 200},
    )

    deleted = False
    for exp in expenses:
        if not search_term:
            # No filter → delete first one found
            client.delete(f"/travelExpense/{exp['id']}")
            logger.info(f"Deleted travel expense id={exp['id']}")
            return

        emp_info = exp.get("employee") or {}
        emp_name = f"{emp_info.get('firstName', '')} {emp_info.get('lastName', '')}".strip()
        comment = exp.get("comment") or ""
        term_lower = search_term.lower()

        if term_lower in emp_name.lower() or term_lower in comment.lower():
            client.delete(f"/travelExpense/{exp['id']}")
            logger.info(f"Deleted travel expense id={exp['id']}")
            deleted = True

    if not deleted:
        logger.error(f"No travel expense found matching: {search_term!r}")


# ======================================================================
# Project workflow
# ======================================================================

def _create_project(intent: dict, client: TripletexClient) -> None:
    proj = intent.get("project") or {}
    cust_data = intent.get("customer") or {}
    raw_prompt = intent.get("_raw_prompt", "")

    # Fallback: if parser didn't extract customer data, try raw prompt
    if not cust_data.get("name") and raw_prompt:
        fb = _extract_customer_from_prompt(raw_prompt)
        if fb.get("name"):
            cust_data = {**cust_data, **fb}
            logger.info(f"Extracted customer from raw prompt: {fb}")

    # ── Parse full project cycle extras from raw prompt ──────────────
    # Budget
    budget_match = re.search(r'bud[sg](?:j?ett?|get)\s+(\d[\d\s]*)\s*kr', raw_prompt, re.IGNORECASE)
    budget_amount = int(budget_match.group(1).replace(" ", "")) if budget_match else None

    # Employee hours: "Name (role, email) N hours/timar/timer/Stunden"
    hour_entries: list[dict] = []
    hour_pattern = re.compile(
        r'([A-ZÀ-Ž][a-zà-ž]+(?:\s+[A-ZÀ-Ž][a-zà-ž]+)+)\s*'   # full name
        r'\(([^)]*)\)\s*'                                         # (role, email)
        r'(\d+)\s*(?:timar|timer|hours?|Stunden|horas|heures)',
        re.UNICODE
    )
    for m in hour_pattern.finditer(raw_prompt):
        name = m.group(1).strip()
        parens = m.group(2)
        hrs = int(m.group(3))
        email = None
        email_m = re.search(r'[\w.+-]+@[\w.-]+', parens)
        if email_m:
            email = email_m.group(0)
        hour_entries.append({"name": name, "email": email, "hours": hrs})

    # Supplier cost: "leverandørkostnad/supplier cost AMOUNT kr from/frå Supplier (org.nr XXXXX)"
    supplier_cost: dict | None = None
    supp_pattern = re.compile(
        r'(?:leverandørkostnad|leverandorkostnad|supplier\s*cost|Lieferantenkosten|coste?\s*de\s*proveedor|custo\s*do\s*fornecedor)\s+'
        r'(\d[\d\s]*)\s*kr\s*(?:frå|fra|from|von|de)\s+'
        r'([A-ZÀ-Ž][\w\s]+?)(?:\s*\((?:org\.?\s*nr?\.?\s*)?(\d{9})\))?(?:\.|,|\s*$)',
        re.IGNORECASE | re.UNICODE
    )
    supp_m = supp_pattern.search(raw_prompt)
    if supp_m:
        supplier_cost = {
            "amount": int(supp_m.group(1).replace(" ", "")),
            "name": supp_m.group(2).strip(),
            "org_number": supp_m.group(3),
        }

    # Create invoice flag
    wants_invoice = bool(re.search(
        r'kundefaktura|faktura\s*for\s*prosjekt|customer\s*invoice|project\s*invoice|'
        r'Kundenrechnung|Rechnung|factura|fatura|invoice|faktura',
        raw_prompt, re.IGNORECASE
    ))

    # ── Create customer (optimistic — skip GET on fresh account) ─────
    customer: dict | None = None
    if cust_data.get("name"):
        cust_payload: dict = {"isCustomer": True, "name": cust_data["name"]}
        if cust_data.get("email"):
            cust_payload["email"] = cust_data["email"]
        if cust_data.get("org_number"):
            cust_payload["organizationNumber"] = cust_data["org_number"]
        try:
            customer = _post_value_with_heal(client, "/customer", cust_payload)
        except Exception:
            customer = _find_customer(client, cust_data["name"])

    # ── Resolve projectManager — required by Tripletex ──────────────
    emp_data = intent.get("employee") or {}
    pm_id_from_intent: int | None = None
    pm_ident = emp_data.get("identifier") or f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}".strip()
    pm_email = emp_data.get("email")
    if pm_ident or pm_email:
        pm_emp = None
        if pm_email:
            pm_emp = resolve_employee(client, email=pm_email)
        if not pm_emp and pm_ident:
            pm_emp = resolve_employee(client, name=pm_ident)
        if not pm_emp:
            pm_payload: dict = {"userType": "STANDARD" if pm_email else "NO_ACCESS"}
            if pm_ident:
                parts = pm_ident.split()
                pm_payload["firstName"] = parts[0]
                if len(parts) > 1:
                    pm_payload["lastName"] = " ".join(parts[1:])
            if pm_email:
                pm_payload["email"] = pm_email
            pm_payload["dateOfBirth"] = "1990-01-01"
            dept_id = _get_default_department_id(client)
            if dept_id:
                pm_payload["department"] = {"id": dept_id}
            try:
                pm_emp = _post_value_with_heal(client, "/employee", pm_payload)
            except Exception:
                pass
        if pm_emp:
            pm_id_from_intent = pm_emp["id"]

    # Fallback: try all employees as PM candidates (lowest-ID first = original admin)
    pm_candidates: list[int] = []
    if pm_id_from_intent:
        pm_candidates.append(pm_id_from_intent)
    try:
        all_emps = client.get_list("/employee", params={"count": 100, "fields": "id"})
        if all_emps:
            all_emps.sort(key=lambda e: e.get("id", float("inf")))
            for e in all_emps:
                if e["id"] not in pm_candidates:
                    pm_candidates.append(e["id"])
    except Exception:
        pass

    start_date = proj.get("start_date") or TODAY

    payload: dict = {
        "name": proj.get("name") or "New Project",
        "startDate": start_date,
    }
    if proj.get("number"):
        payload["number"] = proj["number"]
    if proj.get("end_date"):
        payload["endDate"] = proj["end_date"]
    if proj.get("description"):
        payload["description"] = proj["description"]
    if customer:
        payload["customer"] = {"id": customer["id"]}

    result = None
    for pm_id in (pm_candidates or [None]):  # type: ignore[list-item]
        proj_payload = dict(payload)
        if pm_id:
            proj_payload["projectManager"] = {"id": pm_id}
        try:
            resp = client.post("/project", json=proj_payload)
            val = resp.get("value") if isinstance(resp, dict) else None
            if val and val.get("id"):
                result = val
                break
        except requests.exceptions.HTTPError as exc:
            err_body = exc.response.text if exc.response is not None else ""
            if "projectManager" in err_body or "Prosjektleder" in err_body:
                continue  # try next PM candidate
            try:
                result = _post_value_with_heal(client, "/project", proj_payload)
            except Exception:
                pass
            break
        except Exception:
            try:
                result = _post_value_with_heal(client, "/project", proj_payload)
            except Exception:
                pass
            break

    if not result:
        logger.error("create_project: project creation failed")
        return

    project_id = result["id"]
    logger.info(f"Created project id={project_id}")

    # ── Full project cycle: register hours, supplier cost, invoice ───

    # Step A: Find a general activity for timesheet entries
    activity: dict | None = None
    if hour_entries:
        try:
            acts = client.get_list("/activity", params={"isGeneral": "true", "isChargeable": "true", "count": 5, "fields": "id,name"})
            if acts:
                activity = acts[0]
            if not activity:
                acts = client.get_list("/activity", params={"isGeneral": "true", "count": 5, "fields": "id,name"})
                if acts:
                    activity = acts[0]
        except Exception:
            pass

    # Step B: Register timesheet hours for each employee
    emp_cache: dict[str, dict] = {}  # email -> employee dict
    for entry in hour_entries:
        emp_name = entry["name"]
        emp_email = entry.get("email")
        emp = None
        if emp_email and emp_email in emp_cache:
            emp = emp_cache[emp_email]
        elif emp_email:
            emp = resolve_employee(client, email=emp_email)
        if not emp and emp_name:
            emp = resolve_employee(client, name=emp_name)
        if not emp:
            # Create employee
            parts = emp_name.split()
            emp_payload: dict = {
                "firstName": parts[0],
                "lastName": " ".join(parts[1:]) if len(parts) > 1 else parts[0],
                "dateOfBirth": "1990-01-01",
                "userType": "STANDARD" if emp_email else "NO_ACCESS",
            }
            if emp_email:
                emp_payload["email"] = emp_email
            dept_id = _get_default_department_id(client)
            if dept_id:
                emp_payload["department"] = {"id": dept_id}
            try:
                emp = _post_value_with_heal(client, "/employee", emp_payload)
            except Exception as exc:
                logger.warning(f"create_project: employee creation failed for {emp_name}: {exc}")
        if emp and emp_email:
            emp_cache[emp_email] = emp

        if emp and activity:
            ts_payload: dict = {
                "project": {"id": project_id},
                "activity": {"id": activity["id"]},
                "date": start_date,
                "hours": entry["hours"],
                "employee": {"id": emp["id"]},
                "chargeable": True,
            }
            try:
                ts = _post_value_with_heal(client, "/timesheet/entry", ts_payload)
                if ts:
                    logger.info(f"create_project: timesheet {emp_name} {entry['hours']}h id={ts.get('id')}")
            except Exception as exc:
                logger.warning(f"create_project: timesheet failed for {emp_name}: {exc}")

    # Step C: Register supplier cost as project order line
    if supplier_cost and result:
        supplier: dict | None = None
        supp_name = supplier_cost["name"]
        supp_org = supplier_cost.get("org_number")
        # Find or create supplier
        try:
            supp_payload: dict = {"isCustomer": False, "isSupplier": True, "name": supp_name}
            if supp_org:
                supp_payload["organizationNumber"] = supp_org
            supplier = _post_value_with_heal(client, "/customer", supp_payload)
        except Exception:
            supplier = _find_customer(client, supp_name)

        # Record supplier cost via ledger voucher with project reference
        # (POST /project/orderline is [BETA] and returns 403)
        try:
            cost_acct = _resolve_account(client, "4300")  # cost of goods
            if cost_acct:
                _post_value_with_heal(client, "/ledger/voucher", {
                    "date": start_date,
                    "description": f"Leverandørkostnad / Supplier cost – {supp_name}",
                    "postings": [
                        {"account": {"id": cost_acct["id"]}, "amountGross": supplier_cost["amount"],
                         "project": {"id": project_id}},
                        {"account": {"id": (supplier or {}).get("id") and _resolve_account(client, "2400") or cost_acct},
                         "amountGross": -supplier_cost["amount"]},
                    ],
                })
            logger.info(f"create_project: supplier cost {supplier_cost['amount']} for {supp_name}")
        except Exception as exc:
            logger.warning(f"create_project: supplier cost voucher failed: {exc}")

    # Step D: Create customer invoice for the project
    if wants_invoice and customer:
        _ensure_bank_account(client)
        total_hours = sum(e["hours"] for e in hour_entries) if hour_entries else 0
        supp_amt = supplier_cost["amount"] if supplier_cost else 0
        invoice_amount = budget_amount or (total_hours * 1000 + supp_amt)  # fallback hourly rate

        try:
            order = _post_value_with_heal(client, "/order", {
                "customer": {"id": customer["id"]},
                "orderDate": start_date,
                "deliveryDate": start_date,
                "project": {"id": project_id},
            })
            if order:
                order_id = order["id"]
                # Add order line with project total
                client.post("/order/orderline", json={
                    "order": {"id": order_id},
                    "description": proj.get("name") or "Project billing",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": invoice_amount,
                })
                # Create invoice
                inv_date = start_date
                due_date = (date.fromisoformat(inv_date) + timedelta(days=14)).isoformat()
                inv = client.post("/invoice?sendToCustomer=false", json={
                    "invoiceDate": inv_date,
                    "invoiceDueDate": due_date,
                    "orders": [{"id": order_id}],
                })
                inv_val = (inv or {}).get("value")
                if inv_val:
                    inv_id = inv_val.get("id")
                    logger.info(f"create_project: invoice id={inv_id}")
                    # Always send explicitly — invoice was created as draft
                    try:
                        client.put(
                            f"/invoice/{inv_id}/:send",
                            params={"sendType": "EMAIL", "overrideEmailAddress": "noreply@example.com"},
                        )
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f"create_project: invoice creation failed: {exc}")


# ======================================================================
# Department workflow
# ======================================================================

def _create_one_department(client: TripletexClient, dept: dict) -> None:
    """Create a single department from a dept dict."""
    name = dept.get("name")
    if not name:
        return
    # Strip surrounding quotes that Gemini sometimes preserves from the prompt
    # (e.g. the prompt says: create department 'Sales' → name = "'Sales'")
    name = name.strip("'\"")
    if not name:
        return
    payload: dict = {"name": name}
    if dept.get("department_number"):
        payload["departmentNumber"] = dept["department_number"]
    try:
        result = _post_value_with_heal(client, "/department", payload)
        if result:
            logger.info(f"Created department id={result.get('id')} name={name!r}")
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 422):
            logger.info(f"Department '{name}' likely already exists, skipping")
        else:
            raise


def _create_department(intent: dict, client: TripletexClient) -> None:
    dept_raw = intent.get("department")
    if not dept_raw:
        return
    # Parser may return a list when asked to create multiple departments
    depts: list[dict] = dept_raw if isinstance(dept_raw, list) else [dept_raw]
    for i, dept in enumerate(depts, 1):
        # Always ensure departmentNumber is set (sequential "1", "2", "3" if not given)
        if not dept.get("department_number"):
            dept["department_number"] = str(i)
        _create_one_department(client, dept)


# ======================================================================
# Contact workflow
# ======================================================================

def _create_contact(intent: dict, client: TripletexClient) -> None:
    """Create a contact person linked to a customer or supplier."""
    contact = intent.get("contact") or {}
    cust_data = intent.get("customer") or {}

    payload: dict = {}
    if contact.get("first_name"):
        payload["firstName"] = contact["first_name"]
    if contact.get("last_name"):
        payload["lastName"] = contact["last_name"]
    if contact.get("email"):
        payload["email"] = contact["email"]
    if contact.get("phone"):
        payload["phoneNumberMobile"] = contact["phone"]

    # The parser may put customer/supplier references inside the contact dict
    # (contact.customer_name, contact.org_number) OR in the intent-level customer dict.
    cust_name = (
        cust_data.get("name") or cust_data.get("identifier")
        or contact.get("customer_name")
    )
    org_no = cust_data.get("org_number") or contact.get("org_number")
    supplier_name = contact.get("supplier_name")
    is_supplier = bool(cust_data.get("is_supplier")) or bool(supplier_name)

    if is_supplier:
        # Try to find supplier
        supplier = None
        s_name = supplier_name or cust_name
        if org_no:
            results = client.get_list("/supplier", params={"organizationNumber": org_no, "count": 5})
            supplier = results[0] if results else None
        if not supplier and s_name:
            results = client.get_list("/supplier", params={"count": 100})
            name_lower = s_name.lower()
            supplier = next((s for s in results if name_lower in (s.get("name") or "").lower()), None)
        if supplier:
            payload["supplier"] = {"id": supplier["id"]}
    else:
        customer = None
        if cust_name or org_no:
            customer = resolve_customer(client, name=cust_name, org_number=org_no)
        if not customer and cust_name:
            # Customer doesn't exist yet — create them
            try:
                cust_payload: dict = {"isCustomer": True, "name": cust_name}
                if org_no:
                    cust_payload["organizationNumber"] = org_no
                customer = _post_value_with_heal(client, "/customer", cust_payload)
            except Exception:
                customer = None
        if customer:
            payload["customer"] = {"id": customer["id"]}

    if not payload.get("firstName") and not payload.get("lastName"):
        logger.error("create_contact: no name provided")
        return

    try:
        result = _post_value_with_heal(client, "/contact", payload)
        if result:
            logger.info(f"Created contact id={result.get('id')} name={contact.get('first_name')} {contact.get('last_name')}")
    except Exception as exc:
        logger.warning(f"Contact creation failed: {exc}")


# ======================================================================
# Module enable workflow
# ======================================================================

def _enable_module(intent: dict, client: TripletexClient) -> None:
    module_name = (intent.get("module_name") or "").lower()
    _notes_val = intent.get("notes")
    notes = (str(_notes_val) if isinstance(_notes_val, dict) else (_notes_val or "")).lower()
    combined = module_name + " " + notes

    try:
        settings = client.get_value(
            "/company/settings", params={"fields": "*"}
        )
        if not settings:
            logger.error("Could not fetch company settings")
            return

        updated = dict(settings)
        if "department" in combined:
            updated["accountingDepartmentEnabled"] = True
        if "project" in combined:
            updated["projectsEnabled"] = True
        if "travel" in combined:
            updated["travelExpensesEnabled"] = True
        if "hour" in combined or "time" in combined:
            updated["hoursEnabled"] = True

        client.put_value("/company/settings", json=updated)
        logger.info(f"Enabled module(s) matching: {combined!r}")
    except Exception as exc:
        logger.error(f"enable_module failed: {exc}")


# ======================================================================
# Payroll workflow  (salary transaction → fallback: ledger voucher)
# ======================================================================

def _run_payroll(intent: dict, client: TripletexClient) -> None:
    """Run payroll via POST /salary/transaction with employment prerequisite.

    Flow:
    1. Find/create employee
    2. Ensure employment + employment details exist
    3. Find salary types (base salary, bonus)
    4. POST /salary/transaction?generateTaxDeduction=true
    Falls back to voucher approach if salary/transaction fails.
    """
    emp_data = intent.get("employee") or {}
    payroll = intent.get("payroll") or {}

    # ── Resolve / create employee ─────────────────────────────────────
    identifier = (
        emp_data.get("identifier")
        or f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}".strip()
    )
    # Fallback: when reclassified from travel_expense/invoice, employee may be elsewhere
    if not identifier:
        te_data = intent.get("travel_expense") or {}
        identifier = te_data.get("identifier") or ""
    if not identifier:
        cust_data = intent.get("customer") or {}
        identifier = cust_data.get("name") or cust_data.get("identifier") or ""
    email = emp_data.get("email")

    # Fallback: extract employee email from raw prompt if not in intent
    if not email:
        _raw = intent.get("_raw_prompt", "")
        if _raw:
            import re as _re_email
            em = _re_email.search(r'[\w.+-]+@[\w.-]+\.\w+', _raw)
            if em:
                email = em.group(0)
                logger.info(f"run_payroll: extracted email from raw prompt: {email}")

    employee: dict | None = None
    if email:
        employee = resolve_employee(client, email=email)
    if not employee and identifier:
        employee = resolve_employee(client, name=identifier)

    if not employee:
        dept_id = _get_default_department_id(client)
        emp_payload: dict = {"userType": "STANDARD" if email else "NO_ACCESS"}
        if emp_data.get("first_name"):
            emp_payload["firstName"] = emp_data["first_name"]
        elif identifier and not identifier.startswith("@"):
            parts = identifier.split()
            emp_payload["firstName"] = parts[0]
            if len(parts) > 1:
                emp_payload["lastName"] = " ".join(parts[1:])
        if emp_data.get("last_name"):
            emp_payload["lastName"] = emp_data["last_name"]
        if email:
            emp_payload["email"] = email
        if dept_id:
            emp_payload["department"] = {"id": dept_id}
        dob = emp_data.get("date_of_birth") or "1990-01-01"
        emp_payload["dateOfBirth"] = dob
        try:
            employee = _post_value_with_heal(client, "/employee", emp_payload)
            if employee:
                logger.info(f"Created employee id={employee['id']} for payroll")
        except Exception as exc:
            logger.warning(f"Employee creation failed: {exc}")

    if not employee:
        logger.error("run_payroll: no employee resolved — skipping")
        return

    emp_id = employee["id"]

    # ── Payroll amounts ───────────────────────────────────────────────
    base = float(payroll.get("base_salary") or 0)
    bonus = float(payroll.get("bonus") or 0)
    total = base + bonus

    today_obj = date.today()
    pay_year = int(payroll.get("year") or today_obj.year)
    pay_month = int(payroll.get("month") or today_obj.month)
    pay_date = payroll.get("date") or TODAY

    # Fallback: extract salary amounts from raw prompt if payroll fields missing
    if not total:
        _raw = intent.get("_raw_prompt", "")
        if _raw:
            import re as _re_pay
            # Try to extract base salary: "base salary is 53400", "grunnlønn 53400", "salario base 53400"
            amounts = _re_pay.findall(r'(\d[\d\s]*\d)\s*(?:NOK|kr|nok)?', _raw)
            parsed_amounts = []
            for a in amounts:
                try:
                    parsed_amounts.append(float(a.replace(" ", "")))
                except ValueError:
                    pass
            # Filter out year-like numbers (2025, 2026, etc.)
            parsed_amounts = [a for a in parsed_amounts if a > 100]
            if len(parsed_amounts) >= 2:
                # Assume larger is base, smaller is bonus
                parsed_amounts.sort(reverse=True)
                base = parsed_amounts[0]
                bonus = parsed_amounts[1]
                total = base + bonus
                logger.info(f"run_payroll: extracted from raw prompt base={base} bonus={bonus}")
            elif len(parsed_amounts) == 1:
                base = parsed_amounts[0]
                total = base
                logger.info(f"run_payroll: extracted from raw prompt base={base}")

    if not total:
        logger.error("run_payroll: total salary is 0 — skipping")
        return

    # ── Ensure employment exists (CRITICAL prerequisite) ──────────────
    try:
        employments = client.get_list(
            "/employee/employment",
            params={"employeeId": emp_id, "count": 10, "fields": "id,startDate,endDate"},
        )
        active_employment = None
        for emp in employments:
            end = emp.get("endDate")
            if not end or end >= TODAY:
                active_employment = emp
                break
        if not active_employment and employments:
            active_employment = employments[0]
    except Exception:
        employments = []
        active_employment = None

    if not active_employment:
        logger.info("run_payroll: no active employment found — creating employment + details")
        try:
            employment = _post_value_with_heal(client, "/employee/employment", {
                "employee": {"id": emp_id},
                "startDate": f"{pay_year}-01-01",
            })
            if employment:
                active_employment = employment
                empl_id = employment["id"]
                logger.info(f"Created employment id={empl_id}")
                # Create employment details (required for salary/transaction)
                try:
                    _post_value_with_heal(client, "/employee/employment/details", {
                        "employment": {"id": empl_id},
                        "date": f"{pay_year}-01-01",
                        "employmentType": "ORDINARY",
                        "employmentForm": "PERMANENT",
                        "remunerationType": "MONTHLY_WAGE",
                        "workingHoursScheme": "NOT_SHIFT",
                        "percentageOfFullTimeEquivalent": 100,
                        "annualSalary": 0,
                    })
                    logger.info("Created employment details")
                except Exception as exc:
                    logger.warning(f"Employment details creation failed: {exc}")
        except Exception as exc:
            logger.warning(f"Employment creation failed: {exc}")
    else:
        # Employment exists — ensure details exist too
        empl_id = active_employment["id"]
        try:
            details = client.get_list(
                "/employee/employment/details",
                params={"employmentId": empl_id, "count": 1},
            )
            if not details:
                _post_value_with_heal(client, "/employee/employment/details", {
                    "employment": {"id": empl_id},
                    "date": f"{pay_year}-01-01",
                    "employmentType": "ORDINARY",
                    "employmentForm": "PERMANENT",
                    "remunerationType": "MONTHLY_WAGE",
                    "workingHoursScheme": "NOT_SHIFT",
                    "percentageOfFullTimeEquivalent": 100,
                    "annualSalary": 0,
                })
                logger.info(f"Created missing employment details for employment id={empl_id}")
        except Exception as exc:
            logger.warning(f"Employment details check/creation failed: {exc}")

    # ── Find salary types ─────────────────────────────────────────────
    base_salary_type_id: int | None = None
    bonus_salary_type_id: int | None = None
    try:
        salary_types = client.get_list(
            "/salary/type",
            params={"count": 100, "fields": "id,number,name"},
        )
        for st in salary_types:
            st_num = str(st.get("number") or "")
            st_name = (st.get("name") or "").lower()
            if st_num == "1000" or "fastlønn" in st_name or "fastlonn" in st_name or "base salary" in st_name:
                if not base_salary_type_id:
                    base_salary_type_id = st["id"]
            if "bonus" in st_name or "tillegg" in st_name or st_num in ("1350", "1400"):
                if not bonus_salary_type_id:
                    bonus_salary_type_id = st["id"]
        # Fallback: use first available type
        if not base_salary_type_id and salary_types:
            base_salary_type_id = salary_types[0]["id"]
    except Exception as exc:
        logger.warning(f"Salary type lookup failed: {exc}")

    # ── Try POST /salary/transaction (primary path) ───────────────────
    salary_success = False
    if base_salary_type_id:
        specifications: list[dict] = [
            {"salaryType": {"id": base_salary_type_id}, "rate": base, "count": 1},
        ]
        if bonus > 0 and bonus_salary_type_id:
            specifications.append(
                {"salaryType": {"id": bonus_salary_type_id}, "rate": bonus, "count": 1}
            )
        elif bonus > 0:
            # No bonus type found — add to base
            specifications[0]["rate"] = total

        salary_payload: dict = {
            "date": pay_date,
            "year": pay_year,
            "month": pay_month,
            "payslips": [{
                "employee": {"id": emp_id},
                "specifications": specifications,
            }],
        }
        try:
            result = client.post_value(
                "/salary/transaction?generateTaxDeduction=true",
                json=salary_payload,
            )
            if result:
                logger.info(f"Created salary transaction id={result.get('id')} total={total}")
                salary_success = True
        except Exception as exc:
            logger.warning(f"Salary transaction failed: {exc}")

    # ── Fallback: POST /ledger/voucher on 5000-series salary accounts ─
    if not salary_success:
        logger.info("run_payroll: falling back to ledger voucher approach")
        try:
            salary_acct = _resolve_account_chain(client, "5000", "5001", "5010", "5100")
            payable_acct = _resolve_account_chain(client, "2930", "2910", "2920", "2900", "2800", "2600", "1920", "1900")

            if not salary_acct or not payable_acct:
                logger.error("No salary/payable accounts found — payroll voucher skipped")
                return

            description = f"Payroll {pay_month}/{pay_year} – {identifier or 'employee'}"
            postings: list[dict] = [
                {
                    "row": 1,
                    "account": {"id": salary_acct["id"]},
                    "amountGross": total,
                    "amountGrossCurrency": total,
                    "description": description,
                },
                {
                    "row": 2,
                    "account": {"id": payable_acct["id"]},
                    "amountGross": -total,
                    "amountGrossCurrency": -total,
                    "description": description,
                },
            ]

            result = _post_value_with_heal(client, "/ledger/voucher", {
                "date": pay_date,
                "description": description,
                "postings": postings,
            })
            if result:
                logger.info(
                    f"Created payroll voucher id={result.get('id')} total={total}"
                )
        except Exception as exc:
            logger.error(f"Payroll voucher failed: {exc}")


# ======================================================================
# Project invoice workflow  (timesheet hours → order → invoice)
# ======================================================================

def _create_project_invoice(intent: dict, client: TripletexClient) -> None:
    """Register hours on a project for an employee, then create project invoice."""
    emp_data = intent.get("employee") or {}
    proj_data = intent.get("project") or {}
    cust_data = intent.get("customer") or {}
    inv_data = intent.get("invoice") or {}

    # Fallback: if parser didn't extract customer data, try raw prompt
    if not cust_data.get("name"):
        raw = intent.get("_raw_prompt", "")
        if raw:
            fb = _extract_customer_from_prompt(raw)
            if fb.get("name"):
                cust_data = {**cust_data, **fb}
                logger.info(f"Extracted customer from raw prompt: {fb}")

    hours = float(inv_data.get("hours") or 0)
    hourly_rate = float(inv_data.get("hourly_rate") or 0)
    invoice_date = inv_data.get("date") or TODAY
    activity_name = (proj_data.get("activity") or "Design").strip("'\"")
    proj_name = (proj_data.get("name") or "Project").strip("'\"")
    due_date = (date.fromisoformat(invoice_date) + timedelta(days=14)).isoformat()

    # ── Ensure bank account ───────────────────────────────────────────
    _ensure_bank_account(client)

    # ── Step 1: find/create customer ─────────────────────────────────
    customer: dict | None = None
    if cust_data.get("name"):
        cust_payload: dict = {"isCustomer": True, "name": cust_data["name"]}
        if cust_data.get("org_number"):
            cust_payload["organizationNumber"] = cust_data["org_number"]
        try:
            customer = _post_value_with_heal(client, "/customer", cust_payload)
        except Exception:
            customer = _find_customer(client, cust_data["name"])
    if not customer:
        logger.error("Cannot create project invoice: no customer resolved")
        return

    # ── Step 2: find/create employee ─────────────────────────────────
    identifier = (
        emp_data.get("identifier")
        or f"{emp_data.get('first_name', '')} {emp_data.get('last_name', '')}".strip()
    )
    email = emp_data.get("email")

    employee: dict | None = None
    if email:
        employee = resolve_employee(client, email=email)
    if not employee and identifier:
        employee = resolve_employee(client, name=identifier)

    if not employee:
        dept_id = _get_default_department_id(client)
        emp_payload: dict = {"userType": "STANDARD" if email else "NO_ACCESS"}
        if emp_data.get("first_name"):
            emp_payload["firstName"] = emp_data["first_name"]
        elif identifier:
            parts = identifier.split()
            emp_payload["firstName"] = parts[0]
            if len(parts) > 1:
                emp_payload["lastName"] = " ".join(parts[1:])
        if emp_data.get("last_name"):
            emp_payload["lastName"] = emp_data["last_name"]
        if email:
            emp_payload["email"] = email
        if dept_id:
            emp_payload["department"] = {"id": dept_id}
        try:
            employee = _post_value_with_heal(client, "/employee", emp_payload)
            if employee:
                logger.info(f"Created employee id={employee['id']} for project invoice")
        except Exception as exc:
            logger.warning(f"Employee creation for project invoice failed: {exc}")

    # ── Step 3: find/create project ──────────────────────────────────
    project: dict | None = None

    # Tripletex requires projectManager to be an employee with an active user account.
    # The lowest-ID employee is the original admin who definitely has user access.
    # We collect candidates sorted by ID ascending; try each until project creation succeeds.
    pm_candidates: list[int] = []
    try:
        all_emps = client.get_list("/employee", params={"count": 100, "fields": "id"})
        if all_emps:
            all_emps.sort(key=lambda e: e.get("id", float("inf")))
            pm_candidates = [e["id"] for e in all_emps]
    except Exception:
        pass
    # Append our task employee as last-resort candidate (may lack active user profile)
    if employee and employee["id"] not in pm_candidates:
        pm_candidates.append(employee["id"])

    proj_base: dict = {
        "name": proj_name,
        "startDate": invoice_date,
        "customer": {"id": customer["id"]},
    }
    for pm_id in (pm_candidates or [None]):  # type: ignore[list-item]
        proj_payload = dict(proj_base)
        if pm_id:
            proj_payload["projectManager"] = {"id": pm_id}
        try:
            resp = client.post("/project", json=proj_payload)
            val = resp.get("value") if isinstance(resp, dict) else None
            if val and val.get("id"):
                project = val
                break
        except requests.exceptions.HTTPError as exc:
            err_body = exc.response.text if exc.response is not None else ""
            if "projectManager" in err_body:
                continue  # try next PM candidate
            # Non-PM error: let healer attempt once, then stop
            try:
                project = _post_value_with_heal(client, "/project", proj_payload)
            except Exception:
                pass
            break
        except Exception:
            # Network error: try healer once, then stop
            try:
                project = _post_value_with_heal(client, "/project", proj_payload)
            except Exception:
                pass
            break

    if not project:
        try:
            projects = client.get_list(
                "/project",
                params={"name": proj_name, "count": 5, "fields": "id,name"},
            )
            if projects:
                project = projects[0]
        except Exception:
            pass

    if not project:
        logger.error(f"Could not find or create project '{proj_name}'")
        return

    # ── Step 4: find activity ─────────────────────────────────────────
    # Activities are shared at the company level; search by name then fall back.
    activity: dict | None = None
    try:
        acts = client.get_list(
            "/activity",
            params={"name": activity_name, "count": 5, "fields": "id,name"},
        )
        if acts:
            activity = acts[0]
        if not activity:
            # Any chargeable general activity
            acts = client.get_list(
                "/activity",
                params={"isGeneral": "true", "isChargeable": "true", "count": 5, "fields": "id,name"},
            )
            if acts:
                activity = acts[0]
        if not activity:
            acts = client.get_list(
                "/activity",
                params={"isGeneral": "true", "count": 5, "fields": "id,name"},
            )
            if acts:
                activity = acts[0]
    except Exception as exc:
        logger.warning(f"Activity lookup failed: {exc}")

    # ── Step 5: register timesheet entry ─────────────────────────────
    if hours > 0 and employee and project and activity:
        ts_payload: dict = {
            "project": {"id": project["id"]},
            "activity": {"id": activity["id"]},
            "date": invoice_date,
            "hours": hours,
            "employee": {"id": employee["id"]},
            "chargeable": True,
        }
        try:
            ts = _post_value_with_heal(client, "/timesheet/entry", ts_payload)
            if ts:
                logger.info(f"Created timesheet entry id={ts.get('id')} hours={hours}")
        except Exception as exc:
            logger.warning(f"Timesheet entry failed: {exc}")
    elif not activity:
        logger.warning("No activity found — skipping timesheet entry")

    # ── Step 6: create order + invoice ───────────────────────────────
    # Compute amount: hours×rate, or fixed_price×percentage, or explicit amount
    total_amount: float | None = None
    if hours and hourly_rate:
        total_amount = hours * hourly_rate
    else:
        fixed_price = inv_data.get("fixed_price")
        partial_pct = inv_data.get("partial_percentage")
        if fixed_price and partial_pct:
            total_amount = float(fixed_price) * float(partial_pct) / 100.0
        elif fixed_price:
            total_amount = float(fixed_price)
        else:
            total_amount = inv_data.get("amount")
    if total_amount:
        total_amount = float(total_amount)
    if not total_amount:
        logger.warning("No invoice amount calculable — skipping invoice")
        return

    order_payload: dict = {
        "customer": {"id": customer["id"]},
        "orderDate": invoice_date,
        "deliveryDate": invoice_date,
        "project": {"id": project["id"]},
    }
    order = _post_value_with_heal(client, "/order", order_payload)
    if not order:
        logger.error("Failed to create order for project invoice")
        return

    order_id = order["id"]
    if hours and hourly_rate:
        line_desc = f"{hours}h @ {hourly_rate} NOK/h – {activity_name}"
        line_count = hours
        line_unit_price = hourly_rate
    else:
        line_desc = proj_name
        line_count = 1
        line_unit_price = total_amount
    try:
        client.post("/order/orderline", json={
            "description": line_desc,
            "count": line_count,
            "unitPriceExcludingVatCurrency": line_unit_price,
            "order": {"id": order_id},
        })
    except Exception as exc:
        logger.warning(f"Order line creation failed: {exc}")

    invoice_body: dict = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "orders": [{"id": order_id}],
    }
    try:
        invoice = _post_value_with_heal(
            client, "/invoice?sendToCustomer=false", invoice_body
        )
        if invoice:
            inv_id = invoice.get("id")
            logger.info(
                f"Created project invoice id={inv_id} "
                f"amount={total_amount}"
            )
            # Always send explicitly — invoice was created as draft
            try:
                client.put(
                    f"/invoice/{inv_id}/:send",
                    params={"sendType": "EMAIL", "overrideEmailAddress": "noreply@example.com"},
                )
                logger.info(f"Sent project invoice id={inv_id}")
            except Exception:
                pass  # sandbox may not support email sending
    except Exception:
        logger.error("Could not create project invoice")


# ======================================================================
# Voucher / ledger correction
# ======================================================================

def _delete_voucher(intent: dict, client: TripletexClient) -> None:
    try:
        date_from = (date.today() - timedelta(days=365 * 3)).isoformat()
        date_to = (date.today() + timedelta(days=1)).isoformat()
        _notes_val = intent.get("notes")
        notes = (str(_notes_val) if isinstance(_notes_val, dict) else (_notes_val or "")).lower()
        vouchers = client.get_list(
            "/ledger/voucher",
            params={
                "fields": "id,number,description,date",
                "count": 100,
                "dateFrom": date_from,
                "dateTo": date_to,
            },
        )
        if not vouchers:
            logger.warning("No vouchers found to delete")
            return

        # Try to match by notes/description; fall back to most recently posted
        target = vouchers[-1]
        if notes:
            for v in reversed(vouchers):
                if any(w in (v.get("description") or "").lower() for w in notes.split()):
                    target = v
                    break

        client.delete(f"/ledger/voucher/{target['id']}")
        logger.info(f"Deleted voucher id={target['id']} number={target.get('number')}")
    except Exception as exc:
        logger.error(f"delete_voucher failed: {exc}")


# ======================================================================
# Ledger task workflow (corrections, depreciation, monthly/annual close)
# ======================================================================

def _post_generic_ledger_postings(
    client: TripletexClient,
    postings_raw: list[dict],
    entry_date: str,
    description: str,
) -> None:
    """Resolve accounts and post a balanced voucher from raw posting dicts.

    Postings with null/zero amounts are grouped into balanced pairs where
    possible (debit one, credit the other for an equal amount).  When the
    parser fails to extract an amount for a pair that is clearly debit/credit
    (e.g. salary expense 5000 / accrued salary 2900), we skip them rather
    than sending zero which would fail validation.
    """
    resolved_postings: list[dict] = []
    row = 1
    for p in postings_raw:
        amt = p.get("amount")
        if amt is None or amt == 0:
            continue  # skip postings without a resolved amount
        acct = _find_or_create_account_mod(
            client,
            str(p.get("account_number") or ""),
            p.get("account_name"),
        )
        if acct:
            resolved_postings.append({
                "row": row,
                "account": {"id": acct["id"]},
                "amountGross": float(amt),
                "amountGrossCurrency": float(amt),
            })
            row += 1

    if not resolved_postings:
        logger.warning("_post_generic_ledger_postings: no valid postings to post")
        return

    # Ensure postings sum to zero — if not, try to balance with a rounding row
    total = sum(p["amountGross"] for p in resolved_postings)
    if abs(total) > 0.01:
        logger.warning(f"Postings sum={total}, not balanced — healer will attempt fix")

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": entry_date,
            "description": description,
            "postings": resolved_postings,
        })
        logger.info(f"Posted generic ledger voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Generic ledger voucher failed: {exc}")


# ======================================================================

def _ledger_task(intent: dict, client: TripletexClient) -> None:
    """
    Handle complex ledger tasks:
      - custom_dimension: create free accounting dimension + values + optional voucher
      - correction: reverse wrong voucher + post corrected entry
      - depreciation: post annual/monthly depreciation voucher
      - monthly_close / annual_close: accruals and close entries
      - voucher (generic): post arbitrary voucher from intent
    """
    ledger = intent.get("ledger") or {}
    subtask = (ledger.get("subtask") or "voucher").lower()

    def _find_or_create_account(number: str, name: str | None = None) -> dict | None:
        # Delegate to module-level cached version
        return _find_or_create_account_mod(client, number, name)

    # ── Custom accounting dimensions ─────────────────────────────────
    if subtask == "custom_dimension":
        dim_name = ledger.get("dimension_name") or ""
        dim_values = ledger.get("dimension_values") or []  # [{name, number}]

        if not dim_name:
            logger.error("custom_dimension: no dimension_name provided")
            return

        # 1. Create the dimension name
        dim_result = _post_value_with_heal(
            client, "/ledger/accountingDimensionName",
            {"dimensionName": dim_name},
        )
        if not dim_result:
            logger.error(f"Failed to create accounting dimension '{dim_name}'")
            return
        dim_index = dim_result.get("dimensionIndex")
        # Tripletex only supports freeAccountingDimension1..3
        if dim_index not in (1, 2, 3):
            dim_index = 1
        logger.info(f"Created accounting dimension '{dim_name}' index={dim_index} id={dim_result.get('id')}")

        # 2. Create each dimension value
        created_values: list[dict] = []
        for i, dv in enumerate(dim_values, 1):
            val_payload: dict = {
                "displayName": dv.get("name") or dv.get("displayName") or f"Value {i}",
                "dimensionIndex": dim_index,
                "number": str(dv.get("number") or i),
            }
            try:
                val_result = _post_value_with_heal(
                    client, "/ledger/accountingDimensionValue", val_payload,
                )
                if val_result:
                    created_values.append(val_result)
                    logger.info(f"Created dimension value '{val_payload['displayName']}' id={val_result.get('id')}")
            except Exception as exc:
                logger.warning(f"Dimension value '{val_payload['displayName']}' failed: {exc}")

        # 3. If postings are provided, post a voucher with the dimension on each posting
        postings_raw = ledger.get("postings") or []
        if postings_raw and created_values:
            dim_field = f"freeAccountingDimension{dim_index}"
            # Use the first created value as default dimension on postings
            default_dim_id = created_values[0].get("id")
            voucher_date = ledger.get("date") or TODAY
            voucher_desc = ledger.get("description") or f"Entry with dimension {dim_name}"

            resolved_postings = []
            for row, p in enumerate(postings_raw, 1):
                acct = _find_or_create_account(
                    str(p.get("account_number") or ""), p.get("account_name"),
                )
                if acct:
                    amt = float(p.get("amount") or 0)
                    posting: dict = {
                        "row": row,
                        "account": {"id": acct["id"]},
                        "amountGross": amt,
                        "amountGrossCurrency": amt,
                        "date": voucher_date,
                        "description": voucher_desc,
                    }
                    # Assign dimension value — use specific if provided, else default
                    dim_val_name = (p.get("dimension_value") or "").lower()
                    assigned_id = default_dim_id
                    for cv in created_values:
                        if (cv.get("displayName") or "").lower() == dim_val_name:
                            assigned_id = cv.get("id")
                            break
                    if assigned_id:
                        posting[dim_field] = {"id": assigned_id}
                    resolved_postings.append(posting)

            # Ensure postings balance (sum to 0) — add counter-posting if needed
            if resolved_postings:
                total = sum(p["amountGross"] for p in resolved_postings)
                if abs(total) > 0.01:
                    # Find a balancing account (1920 bank or first available)
                    bal_acct = _resolve_account_chain(client, "1920", "1900", "1500", "1000")
                    if bal_acct:
                        resolved_postings.append({
                            "row": len(resolved_postings) + 1,
                            "account": {"id": bal_acct["id"]},
                            "amountGross": -total,
                            "amountGrossCurrency": -total,
                            "date": voucher_date,
                            "description": voucher_desc,
                        })
                    else:
                        logger.warning("Cannot balance voucher: no balancing account found")

                try:
                    v = _post_value_with_heal(client, "/ledger/voucher", {
                        "date": ledger.get("date") or TODAY,
                        "description": ledger.get("description") or f"Entry with dimension {dim_name}",
                        "postings": resolved_postings,
                    })
                    logger.info(f"Posted voucher with dimension id={v.get('id') if v else None}")
                except Exception as exc:
                    logger.error(f"Voucher with dimension failed: {exc}")
        return

    # ── Ledger analysis → create projects ────────────────────────────
    if subtask == "analysis_create_projects":
        analysis = ledger.get("analysis") or {}
        p1_from = analysis.get("period1_from") or "2026-01-01"
        p1_to = analysis.get("period1_to") or "2026-02-01"
        p2_from = analysis.get("period2_from") or "2026-02-01"
        p2_to = analysis.get("period2_to") or "2026-03-01"
        top_n = int(analysis.get("top_n") or 3)
        acct_from = int(analysis.get("account_range_from") or 4000)
        acct_to = int(analysis.get("account_range_to") or 7999)
        create_activity = analysis.get("create_activity", True)

        # Fetch balance data for both periods using /balanceSheet (NOT /ledger/balanceSheet → 403)
        def _sum_by_account(date_from: str, date_to: str) -> dict[int, tuple[float, str]]:
            """Return {account_number: (balanceChange, account_name)} from /balanceSheet."""
            totals: dict[int, tuple[float, str]] = {}
            try:
                items = client.get_list(
                    "/balanceSheet",
                    params={
                        "dateFrom": date_from, "dateTo": date_to,
                        "accountNumberFrom": str(acct_from), "accountNumberTo": str(acct_to),
                        "count": 1000,
                        "fields": "account(id,number,name),balanceIn,balanceOut,balanceChange",
                    },
                )
                for item in items:
                    acct_obj = item.get("account") or {}
                    acct_num = acct_obj.get("number")
                    acct_name = acct_obj.get("name") or ""
                    if acct_num is None:
                        continue
                    acct_num = int(acct_num)
                    balance_change = float(item.get("balanceChange") or 0)
                    totals[acct_num] = (balance_change, acct_name)
            except Exception as exc:
                logger.warning(f"balanceSheet query failed for {date_from}..{date_to}: {exc}")
            return totals

        try:
            period1 = _sum_by_account(p1_from, p1_to)
            period2 = _sum_by_account(p2_from, p2_to)
        except Exception as exc:
            logger.error(f"Ledger analysis failed: {exc}")
            return

        # Calculate increase per account (cost accounts: positive amount = debit = expense)
        increases: list[tuple[int, str, float]] = []
        all_accounts = set(period1.keys()) | set(period2.keys())
        for acct_num in all_accounts:
            amt1 = period1.get(acct_num, (0.0, ""))[0]
            amt2_tuple = period2.get(acct_num, (0.0, ""))
            amt2 = amt2_tuple[0]
            acct_name = amt2_tuple[1] or period1.get(acct_num, (0.0, ""))[1]
            increase = amt2 - amt1
            if increase > 0:
                increases.append((acct_num, acct_name, increase))

        increases.sort(key=lambda x: x[2], reverse=True)
        top_accounts = increases[:top_n]
        logger.info(f"Top {top_n} cost account increases: {[(a, n, f'{inc:.0f}') for a, n, inc in top_accounts]}")

        if not top_accounts:
            logger.error("No cost account increases found")
            return

        # Find PM candidate (lowest-ID employee = admin)
        pm_id: int | None = None
        try:
            all_emps = client.get_list("/employee", params={"count": 5, "fields": "id"})
            if all_emps:
                all_emps.sort(key=lambda e: e.get("id", float("inf")))
                pm_id = all_emps[0]["id"]
        except Exception:
            pass

        # Create a project (and optionally activity) for each top account
        for acct_num, acct_name, increase in top_accounts:
            proj_name = acct_name or f"Konto {acct_num}"
            proj_payload: dict = {
                "name": proj_name,
                "number": str(acct_num),
                "startDate": TODAY,
                "isInternal": True,
            }
            if pm_id:
                proj_payload["projectManager"] = {"id": pm_id}

            project: dict | None = None
            # Check if project already exists by number
            try:
                existing_projs = client.get_list(
                    "/project", params={"number": str(acct_num), "count": 1, "fields": "id,name,number"},
                )
                if existing_projs:
                    project = existing_projs[0]
                    logger.info(f"Project '{proj_name}' already exists id={project['id']}")
            except Exception:
                pass
            if not project:
                try:
                    resp = client.post("/project", json=proj_payload)
                    val = resp.get("value") if isinstance(resp, dict) else None
                    if val and val.get("id"):
                        project = val
                        logger.info(f"Created project '{proj_name}' for account {acct_num} id={project['id']}")
                except Exception as exc:
                    logger.warning(f"Project creation for '{proj_name}' failed: {exc}")

            if create_activity and project:
                # Create a dedicated activity for this project and link via projectActivity
                linked_activity: dict | None = None
                try:
                    # Check if activity with same name already exists
                    existing_acts = client.get_list(
                        "/activity",
                        params={"name": proj_name[:50], "count": 1, "fields": "id,name"},
                    )
                    if existing_acts:
                        linked_activity = existing_acts[0]
                        logger.info(f"Activity '{proj_name[:50]}' already exists id={linked_activity['id']}")
                    else:
                        act_payload: dict = {
                            "name": proj_name[:50],
                            "activityType": "PROJECT_GENERAL_ACTIVITY",
                        }
                        linked_activity = _post_value_with_heal(client, "/activity", act_payload)
                except Exception as exc:
                    logger.warning(f"Activity creation for '{proj_name}' failed: {exc}")

                if linked_activity and project:
                    try:
                        pa_payload = {
                            "project": {"id": project["id"]},
                            "activity": {"id": linked_activity["id"]},
                        }
                        client.post("/project/projectActivity", json=pa_payload)
                        logger.info(f"Linked activity id={linked_activity['id']} to project id={project['id']}")
                    except Exception as exc:
                        logger.warning(f"ProjectActivity link failed: {exc}")
        return

    if subtask == "correction":
        # ── Structured correction: fetch vouchers+postings, fix each error ──
        date_from = ledger.get("date_from") or (date.today() - timedelta(days=90)).isoformat()
        date_to = ledger.get("date_to") or TODAY
        correction_date = ledger.get("date") or TODAY
        corrections = ledger.get("corrections") or []

        # Fetch all vouchers WITH their postings in the date range
        try:
            vouchers = client.get_list(
                "/ledger/voucher",
                params={
                    "dateFrom": date_from, "dateTo": date_to, "count": 1000,
                    "fields": "id,number,description,date,postings(id,account(id,number),amountGross)",
                },
            )
        except Exception as exc:
            logger.error(f"Could not fetch vouchers for correction: {exc}")
            return

        logger.info(f"Correction: fetched {len(vouchers)} vouchers in [{date_from}..{date_to}]")

        # Build index: for each voucher, extract account numbers and amounts from postings
        # Each entry: (voucher_dict, list of (acct_number_str, amount_float))
        voucher_index: list[tuple[dict, list[tuple[str, float]]]] = []
        for v in vouchers:
            posting_info: list[tuple[str, float]] = []
            for p in (v.get("postings") or []):
                acct = p.get("account") or {}
                acct_num = str(acct.get("number") or "")
                amt = float(p.get("amountGross") or 0)
                posting_info.append((acct_num, amt))
            voucher_index.append((v, posting_info))

        def _find_voucher(account_num: str, amount: float, *, already_used: set) -> dict | None:
            """Find a voucher that has a posting with the given account and amount (positive match)."""
            for v, postings in voucher_index:
                if v["id"] in already_used:
                    continue
                for acct_num, amt in postings:
                    if acct_num == account_num and abs(amt - amount) < 0.01:
                        return v
                    # Also match negative (credit side)
                    if acct_num == account_num and abs(amt + amount) < 0.01:
                        return v
                    # Match by absolute amount on the account
                    if acct_num == account_num and abs(abs(amt) - abs(amount)) < 0.01:
                        return v
            return None

        def _find_duplicate_vouchers(account_num: str, amount: float) -> list[dict]:
            """Find vouchers that look like duplicates (same account + same amount)."""
            matches: list[dict] = []
            for v, postings in voucher_index:
                for acct_num, amt in postings:
                    if acct_num == account_num and abs(abs(amt) - abs(amount)) < 0.01:
                        matches.append(v)
                        break
            return matches

        used_voucher_ids: set[int] = set()

        if corrections:
            for corr in corrections:
                error_type = corr.get("error_type", "")
                wrong_acct = str(corr.get("wrong_account") or "")
                correct_acct = str(corr.get("correct_account") or "")
                amount = float(corr.get("amount") or 0)
                correct_amount = float(corr.get("correct_amount") or 0)
                vat_acct_num = str(corr.get("vat_account") or "2710")
                amount_excl_vat = float(corr.get("amount_excl_vat") or 0)

                logger.info(f"Processing correction: type={error_type} wrong={wrong_acct} amount={amount}")

                if error_type == "wrong_account" and wrong_acct and correct_acct and amount:
                    # Find voucher with posting to wrong account for this amount
                    target = _find_voucher(wrong_acct, amount, already_used=used_voucher_ids)
                    if target:
                        used_voucher_ids.add(target["id"])
                        # Reverse the erroneous voucher
                        try:
                            client.put(f"/ledger/voucher/{target['id']}/:reverse", params={"date": correction_date})
                            logger.info(f"Reversed wrong-account voucher id={target['id']}")
                        except Exception as exc:
                            logger.warning(f"Reversal failed for voucher {target['id']}: {exc}")

                        # Post corrected voucher: copy original but swap wrong→correct account
                        orig_postings = target.get("postings") or []
                        corrected_postings: list[dict] = []
                        for i, p in enumerate(orig_postings, 1):
                            acct_obj = p.get("account") or {}
                            acct_num_str = str(acct_obj.get("number") or "")
                            amt = float(p.get("amountGross") or 0)
                            if acct_num_str == wrong_acct:
                                acct_num_str = correct_acct
                            resolved = _find_or_create_account(acct_num_str)
                            if resolved:
                                corrected_postings.append({
                                    "row": i,
                                    "account": {"id": resolved["id"]},
                                    "amountGross": amt,
                                    "amountGrossCurrency": amt,
                                })
                        if corrected_postings:
                            try:
                                _post_value_with_heal(client, "/ledger/voucher", {
                                    "date": target.get("date") or correction_date,
                                    "description": target.get("description") or "Corrected entry",
                                    "postings": corrected_postings,
                                })
                                logger.info(f"Posted corrected wrong-account voucher ({wrong_acct}→{correct_acct})")
                            except Exception as exc:
                                logger.error(f"Corrected voucher failed: {exc}")
                    else:
                        logger.warning(f"Could not find voucher with account {wrong_acct} amount {amount}")

                elif error_type == "duplicate" and wrong_acct and amount:
                    # Find duplicate vouchers and reverse the extra one
                    dupes = _find_duplicate_vouchers(wrong_acct, amount)
                    if len(dupes) >= 2:
                        # Reverse the LAST duplicate (keep the first)
                        dup_target = dupes[-1]
                        used_voucher_ids.add(dup_target["id"])
                        try:
                            client.put(f"/ledger/voucher/{dup_target['id']}/:reverse", params={"date": correction_date})
                            logger.info(f"Reversed duplicate voucher id={dup_target['id']}")
                        except Exception as exc:
                            logger.warning(f"Duplicate reversal failed: {exc}")
                    elif len(dupes) == 1:
                        # Only one found — reverse it anyway (maybe the other is the "original")
                        dup_target = dupes[0]
                        used_voucher_ids.add(dup_target["id"])
                        try:
                            client.put(f"/ledger/voucher/{dup_target['id']}/:reverse", params={"date": correction_date})
                            logger.info(f"Reversed single duplicate voucher id={dup_target['id']}")
                        except Exception as exc:
                            logger.warning(f"Duplicate reversal failed: {exc}")
                    else:
                        logger.warning(f"Could not find duplicate voucher for account {wrong_acct} amount {amount}")

                elif error_type == "missing_vat" and wrong_acct and amount_excl_vat:
                    # Find the voucher that's missing VAT and reverse it, then repost with VAT
                    target = _find_voucher(wrong_acct, amount_excl_vat, already_used=used_voucher_ids)
                    if target:
                        used_voucher_ids.add(target["id"])
                        try:
                            client.put(f"/ledger/voucher/{target['id']}/:reverse", params={"date": correction_date})
                            logger.info(f"Reversed missing-VAT voucher id={target['id']}")
                        except Exception as exc:
                            logger.warning(f"Reversal failed: {exc}")

                        # Post corrected voucher with proper VAT
                        vat_amount = round(amount_excl_vat * 0.25, 2)  # 25% MVA
                        gross = amount_excl_vat + vat_amount
                        exp_acct = _find_or_create_account(wrong_acct)
                        vat_acct = _find_or_create_account(vat_acct_num, "Inngående mva.")
                        # Find the bank/AP account from original postings (the credit side)
                        bank_acct: dict | None = None
                        for p in (target.get("postings") or []):
                            p_acct = p.get("account") or {}
                            p_num = str(p_acct.get("number") or "")
                            if p_num != wrong_acct and float(p.get("amountGross") or 0) < 0:
                                bank_acct = _find_or_create_account(p_num)
                                break
                        if not bank_acct:
                            bank_acct = _find_or_create_account("1920", "Bankkonto")

                        if exp_acct and vat_acct and bank_acct:
                            try:
                                _post_value_with_heal(client, "/ledger/voucher", {
                                    "date": target.get("date") or correction_date,
                                    "description": target.get("description") or "Corrected with VAT",
                                    "postings": [
                                        {"row": 1, "account": {"id": exp_acct["id"]},
                                         "amountGross": amount_excl_vat, "amountGrossCurrency": amount_excl_vat},
                                        {"row": 2, "account": {"id": vat_acct["id"]},
                                         "amountGross": vat_amount, "amountGrossCurrency": vat_amount},
                                        {"row": 3, "account": {"id": bank_acct["id"]},
                                         "amountGross": -gross, "amountGrossCurrency": -gross},
                                    ],
                                })
                                logger.info(f"Posted corrected voucher with VAT for account {wrong_acct}")
                            except Exception as exc:
                                logger.error(f"Corrected VAT voucher failed: {exc}")
                    else:
                        logger.warning(f"Could not find missing-VAT voucher for account {wrong_acct}")

                elif error_type == "wrong_amount" and wrong_acct and amount and correct_amount:
                    # Find voucher with the wrong amount and repost with correct amount
                    target = _find_voucher(wrong_acct, amount, already_used=used_voucher_ids)
                    if target:
                        used_voucher_ids.add(target["id"])
                        try:
                            client.put(f"/ledger/voucher/{target['id']}/:reverse", params={"date": correction_date})
                            logger.info(f"Reversed wrong-amount voucher id={target['id']}")
                        except Exception as exc:
                            logger.warning(f"Reversal failed: {exc}")

                        # Post corrected voucher with the correct amount
                        orig_postings = target.get("postings") or []
                        corrected_postings = []
                        for i, p in enumerate(orig_postings, 1):
                            acct_obj = p.get("account") or {}
                            acct_num_str = str(acct_obj.get("number") or "")
                            amt = float(p.get("amountGross") or 0)
                            # Replace the wrong amount with correct amount, preserving sign
                            if acct_num_str == wrong_acct and abs(abs(amt) - abs(amount)) < 0.01:
                                amt = correct_amount if amt > 0 else -correct_amount
                            elif abs(abs(amt) - abs(amount)) < 0.01:
                                # The balancing side (e.g. bank) — update proportionally
                                amt = -correct_amount if amt < 0 else correct_amount
                            resolved = _find_or_create_account(acct_num_str)
                            if resolved:
                                corrected_postings.append({
                                    "row": i,
                                    "account": {"id": resolved["id"]},
                                    "amountGross": amt,
                                    "amountGrossCurrency": amt,
                                })
                        if corrected_postings:
                            try:
                                _post_value_with_heal(client, "/ledger/voucher", {
                                    "date": target.get("date") or correction_date,
                                    "description": target.get("description") or "Amount corrected",
                                    "postings": corrected_postings,
                                })
                                logger.info(f"Posted corrected amount voucher ({amount}→{correct_amount})")
                            except Exception as exc:
                                logger.error(f"Corrected amount voucher failed: {exc}")
                    else:
                        logger.warning(f"Could not find wrong-amount voucher for account {wrong_acct} amount {amount}")

                else:
                    logger.warning(f"Unknown correction error_type={error_type} or missing fields")
        else:
            # Fallback: old-style correction with description matching + raw postings
            description_hint = (ledger.get("description") or "").lower()
            target_voucher: dict | None = None
            if description_hint:
                for v, _ in voucher_index:
                    if any(w in (v.get("description") or "").lower() for w in description_hint.split() if len(w) > 2):
                        target_voucher = v
                        break
            if not target_voucher and vouchers:
                target_voucher = vouchers[-1]

            if target_voucher:
                try:
                    client.put(f"/ledger/voucher/{target_voucher['id']}/:reverse", params={"date": correction_date})
                    logger.info(f"Reversed voucher id={target_voucher['id']}")
                except Exception as exc:
                    logger.warning(f"Voucher reversal failed: {exc}")

            postings_raw = ledger.get("postings") or []
            if postings_raw:
                resolved_postings = []
                for i, p in enumerate(postings_raw, 1):
                    acct = _find_or_create_account(str(p.get("account_number") or ""), p.get("account_name"))
                    if acct:
                        amt = float(p.get("amount") or 0)
                        resolved_postings.append({
                            "row": i,
                            "account": {"id": acct["id"]},
                            "amountGross": amt,
                            "amountGrossCurrency": amt,
                        })
                if resolved_postings:
                    try:
                        v = _post_value_with_heal(client, "/ledger/voucher", {
                            "date": correction_date,
                            "description": ledger.get("description") or "Corrected entry",
                            "postings": resolved_postings,
                        })
                        logger.info(f"Posted corrected voucher id={v.get('id') if v else None}")
                    except Exception as exc:
                        logger.error(f"Corrected voucher posting failed: {exc}")

    elif subtask in ("depreciation", "monthly_close", "annual_close"):
        entry_date = ledger.get("date") or TODAY
        is_monthly = subtask == "monthly_close"

        # ── Build assets list (multi-asset or legacy single-asset) ──
        assets = ledger.get("assets") or []
        if not assets:
            # Backward compat: single-asset fields
            depr_obj = ledger.get("depreciation") or {}
            cost = float(depr_obj.get("asset_cost") or ledger.get("asset_cost") or 0)
            yrs = int(depr_obj.get("years") or ledger.get("years") or 1)
            amt = float(depr_obj.get("annual_amount") or ledger.get("annual_amount") or 0)
            if cost and yrs and not amt:
                amt = round(cost / yrs, 2)
            if amt:
                assets = [{
                    "name": "Asset",
                    "cost": cost,
                    "years": yrs,
                    "annual_amount": amt,
                    "depreciation_account": str(depr_obj.get("depreciation_account") or ledger.get("depreciation_account") or "6010"),
                    "accumulated_account": str(depr_obj.get("accumulated_account") or ledger.get("accumulated_account") or "1209"),
                }]

        # ── Post one depreciation voucher per asset ──
        total_depreciation_posted = 0.0  # Track for tax calculation
        for asset in assets:
            cost = float(asset.get("cost") or 0)
            yrs = int(asset.get("years") or 1)
            amt = float(asset.get("annual_amount") or 0)
            if cost and yrs and not amt:
                amt = round(cost / yrs, 2)
            if is_monthly and amt:
                amt = round(amt / 12, 2)
            if not amt:
                continue

            depr_num = str(asset.get("depreciation_account") or "6010")
            accum_num = str(asset.get("accumulated_account") or "")
            asset_acct_num = str(asset.get("asset_account") or "")
            # Auto-derive accumulated account from asset account (e.g. 1240→1249)
            if not accum_num and asset_acct_num and len(asset_acct_num) == 4:
                accum_num = asset_acct_num[:3] + "9"
            if not accum_num:
                accum_num = "1209"
            asset_name = asset.get("name") or "Asset"

            depr_acct = _find_or_create_account(depr_num, "Avskrivning")
            accum_acct = _find_or_create_account(accum_num, "Akkumulerte avskrivninger")

            if depr_acct and accum_acct:
                desc = f"Avskrivning – {asset_name}" if len(assets) > 1 else (ledger.get("description") or "Avskrivning")
                try:
                    v = _post_value_with_heal(client, "/ledger/voucher", {
                        "date": entry_date,
                        "description": desc,
                        "postings": [
                            {"row": 1, "account": {"id": depr_acct["id"]}, "amountGross": amt, "amountGrossCurrency": amt},
                            {"row": 2, "account": {"id": accum_acct["id"]}, "amountGross": -amt, "amountGrossCurrency": -amt},
                        ],
                    })
                    logger.info(f"Posted depreciation voucher for {asset_name} id={v.get('id') if v else None} amount={amt}")
                    total_depreciation_posted += amt
                except Exception as exc:
                    logger.error(f"Depreciation voucher for {asset_name} failed: {exc}")
            else:
                logger.error(f"Cannot find accounts for {asset_name} ({depr_num}, {accum_num})")

        # ── Post prepaid expense resolution vouchers ──
        total_prepaid_posted = 0.0  # Track for tax calculation
        prepaid_list = ledger.get("prepaid_expenses") or []
        for pe in prepaid_list:
            pe_amount = float(pe.get("amount") or 0)
            if not pe_amount:
                continue
            from_acct = _find_or_create_account(str(pe.get("from_account") or "1700"), "Forskuddsbetalt kostnad")
            to_acct = _find_or_create_account(str(pe.get("to_account") or "6300"), "Leie-/leasingkostnad")
            if from_acct and to_acct:
                try:
                    v = _post_value_with_heal(client, "/ledger/voucher", {
                        "date": entry_date,
                        "description": pe.get("description") or "Oppløsning forskuddsbetalt kostnad",
                        "postings": [
                            {"row": 1, "account": {"id": to_acct["id"]}, "amountGross": pe_amount, "amountGrossCurrency": pe_amount},
                            {"row": 2, "account": {"id": from_acct["id"]}, "amountGross": -pe_amount, "amountGrossCurrency": -pe_amount},
                        ],
                    })
                    logger.info(f"Posted prepaid expense voucher id={v.get('id') if v else None} amount={pe_amount}")
                    total_prepaid_posted += pe_amount
                except Exception as exc:
                    logger.error(f"Prepaid expense voucher failed: {exc}")

        # ── Post tax provision voucher ──
        tax_prov = ledger.get("tax_provision")
        if tax_prov:
            tax_amount = float(tax_prov.get("amount") or 0)
            tax_rate = float(tax_prov.get("rate") or 0.22)

            # Auto-calculate tax from /balanceSheet if amount not provided
            if not tax_amount and tax_rate:
                try:
                    # For annual close, use start of year; for monthly, start of month
                    if subtask == "annual_close":
                        bs_date_from = entry_date[:4] + "-01-01"
                    else:
                        bs_date_from = entry_date[:8] + "01" if entry_date else TODAY[:8] + "01"

                    # Revenue accounts (3000-3999): negative balanceChange = income
                    revenue_items = client.get_list("/balanceSheet", params={
                        "dateFrom": bs_date_from,
                        "dateTo": entry_date,
                        "accountNumberFrom": "3000", "accountNumberTo": "3999",
                        "fields": "account(number),balanceChange", "count": 200,
                    })
                    total_revenue = abs(sum(float(r.get("balanceChange") or 0) for r in revenue_items))

                    # Expense accounts (4000-8699)
                    expense_items = client.get_list("/balanceSheet", params={
                        "dateFrom": bs_date_from,
                        "dateTo": entry_date,
                        "accountNumberFrom": "4000", "accountNumberTo": "8699",
                        "fields": "account(number),balanceChange", "count": 200,
                    })
                    total_expenses = sum(float(e.get("balanceChange") or 0) for e in expense_items)

                    # CRITICAL: /balanceSheet may NOT include just-created vouchers!
                    # Manually add depreciation + prepaid amounts to expenses
                    total_expenses += total_depreciation_posted + total_prepaid_posted
                    logger.info(f"Tax calc: adding just-posted vouchers to expenses: depreciation={total_depreciation_posted} prepaid={total_prepaid_posted}")

                    taxable_income = total_revenue - total_expenses
                    if taxable_income > 0:
                        tax_amount = round(taxable_income * tax_rate)
                        logger.info(f"Auto-calculated tax: revenue={total_revenue} expenses={total_expenses} taxable={taxable_income} tax={tax_amount}")
                except Exception as exc:
                    logger.warning(f"Tax auto-calculation from /balanceSheet failed: {exc}")

            if tax_amount:
                exp_acct = _find_or_create_account(str(tax_prov.get("expense_account") or "8700"), "Skattekostnad")
                pay_acct = _find_or_create_account(str(tax_prov.get("payable_account") or "2920"), "Betalbar skatt")
                if exp_acct and pay_acct:
                    try:
                        v = _post_value_with_heal(client, "/ledger/voucher", {
                            "date": entry_date,
                            "description": f"Skatteavsetning ({int(tax_rate*100)}%)",
                            "postings": [
                                {"row": 1, "account": {"id": exp_acct["id"]}, "amountGross": tax_amount, "amountGrossCurrency": tax_amount},
                                {"row": 2, "account": {"id": pay_acct["id"]}, "amountGross": -tax_amount, "amountGrossCurrency": -tax_amount},
                            ],
                        })
                        logger.info(f"Posted tax provision voucher id={v.get('id') if v else None} amount={tax_amount}")
                    except Exception as exc:
                        logger.error(f"Tax provision voucher failed: {exc}")

        # ── Post remaining generic postings (accrual reversal, salary, etc.) ──
        postings_raw = ledger.get("postings") or []
        if postings_raw:
            _post_generic_ledger_postings(client, postings_raw, entry_date, ledger.get("description") or "Monthly close entry")

    else:
        # Generic ledger voucher — post whatever postings are specified
        postings_raw = ledger.get("postings") or []
        if not postings_raw:
            logger.error("ledger task: no postings provided")
            return

        _post_generic_ledger_postings(client, postings_raw, ledger.get("date") or TODAY, ledger.get("description") or "Ledger entry")


# ======================================================================
# Receipt / Kvittering booking handler
# ======================================================================

def _book_receipt(intent: dict, client: TripletexClient) -> None:
    """Book a receipt expense from PDF: extract item, map to expense account, post voucher."""
    import re as _re

    inv = intent.get("invoice") or {}
    _raw = intent.get("_raw_prompt", "")

    # ── Determine item and amount from parsed data or raw prompt ──────
    total_incl_vat = float(inv.get("amount") or 0)
    item_name = inv.get("description") or ""
    receipt_date = inv.get("date") or TODAY

    # Extract department from prompt
    dept_name: str | None = None
    dept_id: int | None = None
    m = _re.search(
        r'(?:avdeling|department|abteilung|département|departamento)\s+["\']?([A-ZÆØÅa-zæøåé][A-ZÆØÅa-zæøåé0-9 -]+)',
        _raw, _re.IGNORECASE,
    )
    if m:
        dept_name = m.group(1).strip().strip("'\".")
    if dept_name:
        try:
            depts = client.get_list("/department", params={"name": dept_name, "count": 10})
            match = next((d for d in depts if (d.get("name") or "").lower() == dept_name.lower()), None)
            if match:
                dept_id = match["id"]
            else:
                created = _post_value_with_heal(client, "/department", {"name": dept_name})
                if created:
                    dept_id = created["id"]
        except Exception:
            pass

    # Fallback amount extraction from raw prompt
    if not total_incl_vat and _raw:
        amt_m = _re.search(r'(\d[\d\s.,]*\d)\s*(?:NOK|kr|nok)\b', _raw)
        if amt_m:
            raw_amt = amt_m.group(1).replace(" ", "").replace(".", "").replace(",", ".")
            try:
                total_incl_vat = float(raw_amt)
            except ValueError:
                pass

    if not total_incl_vat:
        logger.error("book_receipt: no amount found")
        return

    # ── Expense account mapping ───────────────────────────────────────
    _EXPENSE_ACCOUNT_MAP: list[tuple[list[str], str, bool]] = [
        # (keywords, account_number, supports_vat)
        # Travel/transport → 7140 (NOT 7100 which is VAT-locked to mva-kode 0!)
        (["togbillett", "flybillett", "taxi", "buss", "fly ", "flight",
          "reise", "travel", "transport", "voyage", "viaje", "billet",
          "ticket", "fahrkarte", "fahrschein"], "7140", True),
        (["oppbevaringsboks", "hylle", "skap", "boks", "storage", "shelf", "container",
          "aufbewahrung", "regal", "schrank"], "6540", True),
        (["tastatur", "keyboard", "monitor", "skjerm", "printer", "skriver",
          "electronics", "elektronik", "électronique", "electrónica",
          "stol", "chair", "bord", "desk", "møbel", "furniture", "möbel",
          "lampe", "lamp", "headset", "adapter", "kabel", "cable"], "6540", True),
        (["papir", "paper", "penn", "pen", "kontor", "office supplies",
          "bürobedarf", "fourniture"], "6500", True),
        (["rengjøring", "cleaning", "reinigung", "nettoyage"], "7160", True),
        (["mat", "food", "kaffe", "coffee", "catering", "representasjon",
          "restaurant", "essen", "nourriture", "comida"], "7350", False),  # VAT-locked
    ]

    expense_acct_num = "6540"  # default: inventar
    supports_vat = True
    item_low = (item_name or "").lower()
    raw_low = _raw.lower()
    for keywords, acct_num, vat_ok in _EXPENSE_ACCOUNT_MAP:
        if any(kw in item_low or kw in raw_low for kw in keywords):
            expense_acct_num = acct_num
            supports_vat = vat_ok
            break

    # ── Look up accounts ──────────────────────────────────────────────
    _ACCT_NAMES = {
        "6540": "Inventar og utstyr", "6500": "Kontorrekvisita",
        "7140": "Reisekostnad, oppgavepliktig", "7160": "Renholdsmateriell",
        "7350": "Representasjon", "7100": "Bilgodtgjørelse",
        "2710": "Inngående merverdiavgift",
    }
    expense_acct = _resolve_account(client, expense_acct_num)
    if not expense_acct:
        try:
            expense_acct = _post_value_with_heal(client, "/ledger/account", {
                "number": int(expense_acct_num),
                "name": _ACCT_NAMES.get(expense_acct_num, expense_acct_num),
            })
        except Exception:
            pass
    bank_acct = _resolve_account(client, "1920")

    if not expense_acct or not bank_acct:
        logger.error(f"book_receipt: missing accounts (expense={expense_acct}, bank={bank_acct})")
        return

    # ── Look up incoming VAT type ─────────────────────────────────────
    incoming_vat_type_id: int | None = None
    if supports_vat:
        vt = client.get_vat_type("1")
        if vt:
            incoming_vat_type_id = vt["id"]

    # ── Build voucher ─────────────────────────────────────────────────
    desc = item_name or "Receipt expense"
    postings: list[dict] = []

    exp_posting: dict = {
        "row": 1,
        "account": {"id": expense_acct["id"]},
        "amountGross": total_incl_vat,
        "amountGrossCurrency": total_incl_vat,
        "description": desc,
    }
    if supports_vat and incoming_vat_type_id:
        exp_posting["vatType"] = {"id": incoming_vat_type_id}
    if dept_id:
        exp_posting["department"] = {"id": dept_id}
    postings.append(exp_posting)

    # Bank posting on DIFFERENT row (row 2), NO department, NO vatType
    bank_posting: dict = {
        "row": 2,
        "account": {"id": bank_acct["id"]},
        "amountGross": -total_incl_vat,
        "amountGrossCurrency": -total_incl_vat,
        "description": desc,
    }
    postings.append(bank_posting)

    result = None
    try:
        result = _post_value_with_heal(client, "/ledger/voucher", {
            "date": receipt_date,
            "description": desc,
            "postings": postings,
        })
    except Exception as exc:
        logger.warning(f"book_receipt: auto-VAT failed: {exc}")
        # Manual 3-row VAT split: expense net + VAT 2710 + bank -gross
        # This handles accounts locked to "mva-kode 0" that reject vatType
        if incoming_vat_type_id:
            net_amount = round(total_incl_vat / 1.25, 2)
            vat_amount = round(total_incl_vat - net_amount, 2)

            # Ensure VAT account 2710 exists
            vat_acct = _resolve_account(client, "2710")
            if not vat_acct:
                try:
                    vat_acct = _post_value_with_heal(client, "/ledger/account", {
                        "number": 2710, "name": "Inngående merverdiavgift",
                    })
                except Exception:
                    pass

            if vat_acct:
                manual_postings: list[dict] = [
                    {"row": 1, "account": {"id": expense_acct["id"]},
                     "amountGross": net_amount, "amountGrossCurrency": net_amount,
                     "description": desc},
                    {"row": 2, "account": {"id": vat_acct["id"]},
                     "amountGross": vat_amount, "amountGrossCurrency": vat_amount,
                     "description": desc},
                    {"row": 3, "account": {"id": bank_acct["id"]},
                     "amountGross": -total_incl_vat, "amountGrossCurrency": -total_incl_vat,
                     "description": desc},
                ]
                if dept_id:
                    manual_postings[0]["department"] = {"id": dept_id}
                try:
                    result = _post_value_with_heal(client, "/ledger/voucher", {
                        "date": receipt_date,
                        "description": desc,
                        "postings": manual_postings,
                    })
                    logger.info(f"book_receipt: manual VAT split succeeded net={net_amount} vat={vat_amount}")
                except Exception as exc2:
                    logger.error(f"book_receipt: manual VAT split also failed: {exc2}")
            else:
                logger.error("book_receipt: could not resolve VAT account 2710 for manual split")

    if result:
        logger.info(f"book_receipt: created voucher id={result.get('id')} item={item_name} amount={total_incl_vat}")


# ======================================================================
# Bank Reconciliation handler
# ======================================================================

def _bank_reconciliation(intent: dict, client: TripletexClient) -> None:
    """Reconcile a bank statement (CSV) against invoices and suppliers."""
    bs = intent.get("bank_statement") or {}
    txns = bs.get("transactions") or []
    if not txns:
        logger.error("bank_reconciliation: no transactions extracted from statement")
        return

    _ensure_bank_account(client)
    payment_type_id = client.get_payment_type()
    bank_acct = _resolve_account(client, "1920")

    for i, txn in enumerate(txns):
        txn_type = (txn.get("type") or "other").lower()
        txn_date = txn.get("date") or TODAY
        amount = txn.get("amount") or 0
        counterparty = txn.get("counterparty") or ""
        description = txn.get("description") or counterparty or txn_type
        ref = txn.get("reference") or ""

        if not amount:
            continue

        logger.info(f"Bank txn {i+1}/{len(txns)}: {txn_type} {amount} {counterparty!r} ref={ref!r}")

        try:
            if txn_type == "customer_payment":
                _bank_process_customer_payment(
                    client, counterparty, ref, abs(amount), txn_date, payment_type_id
                )
            elif txn_type == "supplier_payment":
                _bank_process_supplier_payment(
                    client, counterparty, ref, abs(amount), txn_date, bank_acct
                )
            elif txn_type in ("interest_income", "interest_expense"):
                _bank_process_interest(client, txn_type, amount, txn_date, bank_acct, description)
            elif txn_type == "tax":
                _bank_process_tax(client, abs(amount), txn_date, bank_acct, description)
            elif txn_type == "fee":
                _bank_process_fee(client, abs(amount), txn_date, bank_acct, description)
            elif txn_type == "salary":
                _bank_process_salary(client, abs(amount), txn_date, bank_acct, description)
            else:
                # Generic: post a voucher with bank account and a guess at the contra account
                _bank_process_other(client, amount, txn_date, bank_acct, description)
        except Exception as exc:
            logger.error(f"Bank txn {i+1} failed: {exc}")


def _resolve_account(client: TripletexClient, number: str) -> dict | None:
    """Resolve a ledger account by number (delegates to client cache)."""
    return client.get_account(number)


def _resolve_account_chain(client: TripletexClient, *numbers: str) -> dict | None:
    """Try multiple account numbers, return the first that resolves."""
    for num in numbers:
        acct = client.get_account(num)
        if acct:
            return acct
    return None


def _find_or_create_account_mod(client: TripletexClient, number: str, name: str | None = None) -> dict | None:
    """Find an account by number, or create it if a name is provided."""
    acct = client.get_account(number)
    if acct:
        return acct
    if name:
        try:
            acct = client.post_value("/ledger/account", json={"number": int(number), "name": name})
            if acct:
                client._account_cache[number] = acct
                logger.info(f"Created ledger account {number} '{name}' id={acct.get('id')}")
            return acct
        except Exception as exc:
            logger.warning(f"Could not create account {number}: {exc}")
    return None


def _bank_process_customer_payment(
    client: TripletexClient,
    customer_name: str,
    invoice_ref: str,
    amount: float,
    pay_date: str,
    payment_type_id: int | None,
) -> None:
    """Register payment on a customer invoice."""
    customer = _find_customer(client, customer_name)
    if not customer:
        logger.warning(f"Bank recon: customer {customer_name!r} not found, skipping")
        return

    # Find invoice — by number if reference is numeric, else by customer
    invoice = None
    if invoice_ref and str(invoice_ref).isdigit():
        invoice = resolve_invoice(client, customer_id=customer["id"], invoice_number=invoice_ref)
    if not invoice:
        invoice = resolve_invoice(client, customer_id=customer["id"])
    if not invoice:
        logger.warning(f"Bank recon: no invoice found for customer {customer_name!r}")
        return

    if not payment_type_id:
        logger.error("Bank recon: no payment type available")
        return

    payment_url = (
        f"/invoice/{invoice['id']}/:payment"
        f"?paymentDate={pay_date}"
        f"&paymentTypeId={payment_type_id}"
        f"&paidAmount={amount}"
    )
    try:
        client.put(payment_url)
        logger.info(f"Bank recon: registered customer payment {amount} on invoice {invoice['id']}")
    except Exception as exc:
        logger.error(f"Bank recon: customer payment failed: {exc}")


def _bank_process_supplier_payment(
    client: TripletexClient,
    supplier_name: str,
    reference: str,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
) -> None:
    """Register supplier payment: try /supplierInvoice/:addPayment first, fallback to voucher."""
    # Find supplier
    supplier = None
    try:
        results = client.get_list("/supplier", params={"count": 200})
        name_lower = supplier_name.lower()
        supplier = next(
            (s for s in results if name_lower in (s.get("name") or "").lower()),
            None,
        )
    except Exception:
        pass

    if not supplier:
        logger.warning(f"Bank recon: supplier {supplier_name!r} not found, skipping")
        return

    # Try to find matching supplier invoice and use /:addPayment
    payment_registered = False
    try:
        supp_invoices = client.get_list("/supplierInvoice", params={
            "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31",
            "supplierId": supplier["id"],
            "fields": "id,amount,invoiceNumber,supplier(id,name)",
            "count": 100,
        })
        # Match by amount or reference
        matched_inv = None
        for si in supp_invoices:
            si_amount = abs(float(si.get("amount") or 0))
            if abs(si_amount - amount) < 0.01:
                matched_inv = si
                break
            if reference and str(si.get("invoiceNumber") or "") == str(reference):
                matched_inv = si
                break
        if matched_inv:
            payment_type_id = client.get_payment_type()
            if payment_type_id:
                try:
                    add_pay_path = (
                        f"/supplierInvoice/{matched_inv['id']}/:addPayment"
                        f"?paymentType={payment_type_id}"
                        f"&amount={amount}"
                        f"&paymentDate={pay_date}"
                    )
                    client.post(add_pay_path)
                    logger.info(f"Bank recon: registered supplier payment via /:addPayment on invoice {matched_inv['id']}")
                    payment_registered = True
                except Exception as exc:
                    logger.warning(f"Bank recon: supplierInvoice /:addPayment failed: {exc}")
    except Exception:
        pass

    # Fallback: post voucher (debit 2400, credit 1920)
    if not payment_registered:
        ap_acct = _resolve_account(client, "2400")
        if not ap_acct or not bank_acct:
            logger.error("Bank recon: missing AP (2400) or bank (1920) account")
            return

        desc = f"Betaling {supplier_name}"
        if reference:
            desc += f" ref {reference}"

        try:
            v = _post_value_with_heal(client, "/ledger/voucher", {
                "date": pay_date,
                "description": desc,
                "postings": [
                    {
                        "row": 1,
                        "account": {"id": ap_acct["id"]},
                        "amountGross": amount,
                        "amountGrossCurrency": amount,
                        "description": desc,
                        "supplier": {"id": supplier["id"]},
                    },
                    {
                        "row": 2,
                        "account": {"id": bank_acct["id"]},
                        "amountGross": -amount,
                        "amountGrossCurrency": -amount,
                        "description": desc,
                    },
                ],
            })
            logger.info(f"Bank recon: posted supplier payment voucher id={v.get('id') if v else None}")
        except Exception as exc:
            logger.error(f"Bank recon: supplier payment voucher failed: {exc}")


def _bank_process_interest(
    client: TripletexClient,
    txn_type: str,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
    description: str,
) -> None:
    """Post interest income or expense voucher."""
    if txn_type == "interest_income":
        contra_acct = _resolve_account(client, "8040")  # Interest income
        bank_amount = abs(amount)
        contra_amount = -abs(amount)
    else:
        contra_acct = _resolve_account(client, "8150")  # Interest expense
        bank_amount = -abs(amount)
        contra_amount = abs(amount)

    if not contra_acct or not bank_acct:
        logger.error("Bank recon: missing accounts for interest posting")
        return

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": pay_date,
            "description": description,
            "postings": [
                {"row": 1, "account": {"id": bank_acct["id"]}, "amountGross": bank_amount, "amountGrossCurrency": bank_amount, "description": description},
                {"row": 2, "account": {"id": contra_acct["id"]}, "amountGross": contra_amount, "amountGrossCurrency": contra_amount, "description": description},
            ],
        })
        logger.info(f"Bank recon: posted interest voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Bank recon: interest voucher failed: {exc}")


def _bank_process_tax(
    client: TripletexClient,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
    description: str,
) -> None:
    """Post tax deduction voucher (debit 1950 tax withholding, credit 1920 bank)."""
    tax_acct = _resolve_account_chain(client, "1950", "2600")
    if not tax_acct or not bank_acct:
        logger.error("Bank recon: missing accounts for tax posting")
        return

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": pay_date,
            "description": description,
            "postings": [
                {"row": 1, "account": {"id": tax_acct["id"]}, "amountGross": amount, "amountGrossCurrency": amount, "description": description},
                {"row": 2, "account": {"id": bank_acct["id"]}, "amountGross": -amount, "amountGrossCurrency": -amount, "description": description},
            ],
        })
        logger.info(f"Bank recon: posted tax voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Bank recon: tax voucher failed: {exc}")


def _bank_process_fee(
    client: TripletexClient,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
    description: str,
) -> None:
    """Post bank fee voucher (debit 7770 bank charges, credit 1920 bank)."""
    fee_acct = _resolve_account_chain(client, "7770", "7700")
    if not fee_acct or not bank_acct:
        logger.error("Bank recon: missing accounts for fee posting")
        return

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": pay_date,
            "description": description,
            "postings": [
                {"row": 1, "account": {"id": fee_acct["id"]}, "amountGross": amount, "amountGrossCurrency": amount, "description": description},
                {"row": 2, "account": {"id": bank_acct["id"]}, "amountGross": -amount, "amountGrossCurrency": -amount, "description": description},
            ],
        })
        logger.info(f"Bank recon: posted fee voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Bank recon: fee voucher failed: {exc}")


def _bank_process_salary(
    client: TripletexClient,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
    description: str,
) -> None:
    """Post salary payment voucher (debit 5000 salary expense, credit 1920 bank)."""
    salary_acct = _resolve_account_chain(client, "5000", "5001")
    if not salary_acct or not bank_acct:
        logger.error("Bank recon: missing accounts for salary posting")
        return

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": pay_date,
            "description": description,
            "postings": [
                {"row": 1, "account": {"id": salary_acct["id"]}, "amountGross": amount, "amountGrossCurrency": amount, "description": description},
                {"row": 2, "account": {"id": bank_acct["id"]}, "amountGross": -amount, "amountGrossCurrency": -amount, "description": description},
            ],
        })
        logger.info(f"Bank recon: posted salary voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Bank recon: salary voucher failed: {exc}")


def _bank_process_other(
    client: TripletexClient,
    amount: float,
    pay_date: str,
    bank_acct: dict | None,
    description: str,
) -> None:
    """Post a generic bank transaction voucher."""
    if not bank_acct:
        return
    # Use a suspense/clearing account for unknown transactions
    contra_acct = _resolve_account_chain(client, "1999", "1900")
    if not contra_acct:
        return

    try:
        v = _post_value_with_heal(client, "/ledger/voucher", {
            "date": pay_date,
            "description": description,
            "postings": [
                {"row": 1, "account": {"id": bank_acct["id"]}, "amountGross": amount, "amountGrossCurrency": amount, "description": description},
                {"row": 2, "account": {"id": contra_acct["id"]}, "amountGross": -amount, "amountGrossCurrency": -amount, "description": description},
            ],
        })
        logger.info(f"Bank recon: posted generic voucher id={v.get('id') if v else None}")
    except Exception as exc:
        logger.error(f"Bank recon: generic voucher failed: {exc}")


# ======================================================================
# Overdue Invoice + Reminder handler
# ======================================================================

def _overdue_reminder(intent: dict, client: TripletexClient) -> None:
    """Find an overdue invoice, create reminder via API, post fee voucher, register partial payment."""
    reminder = intent.get("reminder") or {}
    cust_data = intent.get("customer") or {}
    fee_amount = reminder.get("fee_amount") or 70
    debit_account_num = str(reminder.get("debit_account") or "1500")
    credit_account_num = str(reminder.get("credit_account") or "3400")
    partial_payment = reminder.get("partial_payment_amount")
    send_invoice = reminder.get("send_invoice", True)

    _ensure_bank_account(client)

    # ── Find overdue invoice ──────────────────────────────────────────
    date_from = (date.today() - timedelta(days=365 * 2)).isoformat()
    date_to = (date.today() + timedelta(days=365)).isoformat()

    params: dict = {
        "invoiceDateFrom": date_from,
        "invoiceDateTo": date_to,
        "count": 200,
    }
    customer = None
    if cust_data.get("name"):
        customer = _find_customer(client, cust_data["name"])
        if customer:
            params["customerId"] = customer["id"]

    try:
        invoices = client.get_list("/invoice", params=params)
    except Exception:
        invoices = []

    today_str = date.today().isoformat()
    overdue = None
    for inv in invoices:
        due_date = inv.get("invoiceDueDate") or ""
        outstanding = inv.get("amountOutstanding") or 0
        if due_date < today_str and outstanding > 0:
            overdue = inv
            break

    if not overdue:
        for inv in invoices:
            if (inv.get("amountOutstanding") or 0) > 0:
                overdue = inv
                break

    if not overdue and invoices:
        overdue = invoices[0]

    if not overdue:
        logger.error("overdue_reminder: no invoice found")
        return

    invoice_id = overdue["id"]
    if not customer:
        inv_customer = overdue.get("customer") or {}
        if inv_customer.get("id"):
            customer = {"id": inv_customer["id"], "name": inv_customer.get("name", "")}

    customer_id = customer["id"] if customer else None
    logger.info(f"overdue_reminder: found overdue invoice id={invoice_id}, customer_id={customer_id}")

    # ── Try :createReminder API first ─────────────────────────────────
    reminder_created = False
    for dispatch in ("EMAIL", "OWN_PRINTER", None):
        reminder_url = (
            f"/invoice/{invoice_id}/:createReminder"
            f"?type=REMINDER&date={TODAY}&includeCharge=true&includeInterest=false"
        )
        if dispatch:
            reminder_url += f"&dispatchType={dispatch}"
        try:
            client.put(reminder_url)
            logger.info(f"overdue_reminder: created reminder via :createReminder dispatch={dispatch}")
            reminder_created = True
            break
        except Exception as exc:
            logger.warning(f"overdue_reminder: :createReminder ({dispatch}) failed: {exc}")

    # ── Fallback: post fee voucher + create reminder invoice manually ─
    if not reminder_created:
        # Post reminder fee voucher
        debit_acct = _resolve_account(client, debit_account_num)
        credit_acct = _resolve_account(client, credit_account_num)

        if debit_acct and credit_acct:
            postings = [
                {
                    "row": 1,
                    "account": {"id": debit_acct["id"]},
                    "amountGross": fee_amount,
                    "amountGrossCurrency": fee_amount,
                    "description": "Purregebyr",
                },
                {
                    "row": 2,
                    "account": {"id": credit_acct["id"]},
                    "amountGross": -fee_amount,
                    "amountGrossCurrency": -fee_amount,
                    "description": "Purregebyr",
                },
            ]
            if customer_id and debit_account_num in ("1500", "1501"):
                postings[0]["customer"] = {"id": customer_id}

            try:
                v = _post_value_with_heal(client, "/ledger/voucher", {
                    "date": TODAY,
                    "description": "Purregebyr / Reminder fee",
                    "postings": postings,
                })
                logger.info(f"overdue_reminder: posted fee voucher id={v.get('id') if v else None}")
            except Exception as exc:
                logger.error(f"overdue_reminder: fee voucher failed: {exc}")

        # Create reminder invoice manually
        if customer_id:
            # Look up VAT exempt type (number=5) for reminder invoice
            exempt_vat_id: int | None = None
            vt_exempt = client.get_vat_type("5")
            if vt_exempt:
                exempt_vat_id = vt_exempt["id"]

            try:
                order_line: dict = {
                    "description": "Purregebyr / Reminder fee",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": fee_amount,
                }
                if exempt_vat_id:
                    order_line["vatType"] = {"id": exempt_vat_id}

                order = _post_value_with_heal(client, "/order", {
                    "customer": {"id": customer_id},
                    "orderDate": TODAY,
                    "deliveryDate": TODAY,
                    "orderLines": [order_line],
                })
                if order:
                    order_id = order["id"]
                    inv_result = client.put(f"/order/{order_id}/:invoice?invoiceDate={TODAY}")
                    reminder_inv = (inv_result or {}).get("value")
                    if reminder_inv:
                        reminder_inv_id = reminder_inv["id"]
                        logger.info(f"overdue_reminder: created reminder invoice id={reminder_inv_id}")
                        if send_invoice:
                            try:
                                client.put(f"/invoice/{reminder_inv_id}/:send?sendType=EMAIL")
                                logger.info(f"overdue_reminder: sent via EMAIL")
                            except Exception:
                                try:
                                    client.put(f"/invoice/{reminder_inv_id}/:send?sendType=EHF")
                                except Exception:
                                    pass
            except Exception as exc:
                logger.error(f"overdue_reminder: manual reminder invoice failed: {exc}")

    # ── Register partial payment on the overdue invoice ───────────────
    if partial_payment and float(partial_payment) > 0:
        payment_type_id = client.get_payment_type()
        if payment_type_id:
            payment_url = (
                f"/invoice/{invoice_id}/:payment"
                f"?paymentDate={TODAY}"
                f"&paymentTypeId={payment_type_id}"
                f"&paidAmount={partial_payment}"
            )
            try:
                client.put(payment_url)
                logger.info(f"overdue_reminder: registered partial payment {partial_payment} on invoice {invoice_id}")
            except Exception as exc:
                logger.error(f"overdue_reminder: partial payment failed: {exc}")
