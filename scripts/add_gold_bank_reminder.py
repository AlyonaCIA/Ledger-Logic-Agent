#!/usr/bin/env python3
"""Add gold examples for bank reconciliation and overdue reminder."""
import json
from pathlib import Path

GOLD_PATH = Path("gold/prompts.jsonl")


def add_example(entry: dict) -> None:
    with GOLD_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Bank reconciliation examples ──────────────────────────────────

add_example({
    "id": "gold-bank-recon-en",
    "task_type": "bank_reconciliation",
    "language": "en",
    "confidence": 0.95,
    "prompt": "Reconcile the bank statement (attached CSV) against open invoices in Tripletex. Match incoming payments to customer invoices and outgoing payments to supplier invoices. Handle partial payments correctly.",
    "entities": {
        "bank_statement": {
            "transactions": [
                {"date": "2026-01-16", "amount": 19187.5, "type": "customer_payment", "counterparty": "Taylor Ltd", "reference": "1001", "description": "Payment INV-1001"},
                {"date": "2026-01-17", "amount": 25437.5, "type": "customer_payment", "counterparty": "Smith Corp", "reference": "1002", "description": "Payment INV-1002"},
                {"date": "2026-01-23", "amount": -17650.0, "type": "supplier_payment", "counterparty": "Fischer GmbH", "reference": "S-2001", "description": "Supplier payment"},
                {"date": "2026-01-24", "amount": -13400.0, "type": "supplier_payment", "counterparty": "Mueller GmbH", "reference": "S-2002", "description": "Supplier payment"},
                {"date": "2026-01-27", "amount": 483.95, "type": "interest_income", "counterparty": "Bank", "reference": "", "description": "Interest income"},
                {"date": "2026-01-29", "amount": -1115.7, "type": "tax", "counterparty": "Skatteetaten", "reference": "", "description": "Tax withholding"}
            ]
        }
    }
})

add_example({
    "id": "gold-bank-recon-nb",
    "task_type": "bank_reconciliation",
    "language": "nb",
    "confidence": 0.95,
    "prompt": "Avstem bankutskriften (se vedlagt CSV) mot apne fakturaer i Tripletex. Match innbetalinger til kundefakturaer og utbetalinger til leverandorfakturaer.",
    "entities": {
        "bank_statement": {
            "transactions": [
                {"date": "2026-02-10", "amount": 15000.0, "type": "customer_payment", "counterparty": "Hansen AS", "reference": "2001", "description": "Innbetaling faktura 2001"},
                {"date": "2026-02-12", "amount": -8500.0, "type": "supplier_payment", "counterparty": "Berg Handel", "reference": "LF-301", "description": "Leverandorbetaling"},
                {"date": "2026-02-15", "amount": 250.0, "type": "interest_income", "counterparty": "Bank", "reference": "", "description": "Renteinntekter"},
                {"date": "2026-02-15", "amount": -50.0, "type": "fee", "counterparty": "Bank", "reference": "", "description": "Bankgebyr"}
            ]
        }
    }
})

add_example({
    "id": "gold-bank-recon-de",
    "task_type": "bank_reconciliation",
    "language": "de",
    "confidence": 0.95,
    "prompt": "Gleichen Sie den Kontoauszug (angehangtes CSV) mit den offenen Rechnungen in Tripletex ab. Ordnen Sie eingehende Zahlungen den Kundenrechnungen und ausgehende Zahlungen den Lieferantenrechnungen zu.",
    "entities": {
        "bank_statement": {
            "transactions": [
                {"date": "2026-03-05", "amount": 12500.0, "type": "customer_payment", "counterparty": "Schmidt AG", "reference": "3001", "description": "Zahlung Rechnung 3001"},
                {"date": "2026-03-08", "amount": -9800.0, "type": "supplier_payment", "counterparty": "Wagner GmbH", "reference": "LR-401", "description": "Lieferantenzahlung"},
                {"date": "2026-03-10", "amount": -350.0, "type": "fee", "counterparty": "Bank", "reference": "", "description": "Kontogebuhr"}
            ]
        }
    }
})


# ── Overdue reminder examples ────────────────────────────────────

add_example({
    "id": "gold-overdue-reminder-en",
    "task_type": "overdue_reminder",
    "language": "en",
    "confidence": 0.95,
    "prompt": "One of your customers has an overdue invoice. Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice.",
    "entities": {
        "reminder": {
            "fee_amount": 70,
            "debit_account": "1500",
            "credit_account": "3400",
            "partial_payment_amount": 5000,
            "send_invoice": True
        }
    }
})

add_example({
    "id": "gold-overdue-reminder-nb",
    "task_type": "overdue_reminder",
    "language": "nb",
    "confidence": 0.95,
    "prompt": "En av kundene har en forfalt faktura. Finn den forfalte fakturaen og poster et purregebyr pa 70 NOK. Debet kundefordringer (1500), kredit purregebyr (3400). Opprett en faktura for purregebyret og send den. Registrer ogsa en delbetaling pa 5000 NOK.",
    "entities": {
        "reminder": {
            "fee_amount": 70,
            "debit_account": "1500",
            "credit_account": "3400",
            "partial_payment_amount": 5000,
            "send_invoice": True
        }
    }
})

add_example({
    "id": "gold-overdue-reminder-de",
    "task_type": "overdue_reminder",
    "language": "de",
    "confidence": 0.95,
    "prompt": "Einer Ihrer Kunden hat eine uberfällige Rechnung. Finden Sie die uberfällige Rechnung und buchen Sie eine Mahngebuhr von 70 NOK. Soll Forderungen (1500), Haben Mahngebuhren (3400). Erstellen Sie eine Rechnung fur die Mahngebuhr und senden Sie sie.",
    "entities": {
        "reminder": {
            "fee_amount": 70,
            "debit_account": "1500",
            "credit_account": "3400",
            "partial_payment_amount": None,
            "send_invoice": True
        }
    }
})


print(f"Added 6 new gold examples. Checking total...")

count = 0
with GOLD_PATH.open(encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            json.loads(line)  # validate
            count += 1

print(f"Total gold examples: {count}")
