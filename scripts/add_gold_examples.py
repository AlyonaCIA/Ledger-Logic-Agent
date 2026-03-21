#!/usr/bin/env python3
"""Add gold few-shot examples for annual close, monthly close, and employee with salary."""
import json
import pathlib

GOLD_PATH = pathlib.Path(__file__).resolve().parent.parent / "gold" / "prompts.jsonl"

examples = [
    # ── Multi-asset annual close (nb) ──
    {
        "task_type": "ledger_task", "language": "nb",
        "prompt": "Utfør forenklet årsoppgjør for 2025: Beregn og bokfør årlige avskrivninger for tre eiendeler: Inventar (468300 kr, 10 år lineært, konto 1240), Kontormaskiner (149750 kr, 10 år, konto 1200), Kjøretøy (385000 kr, 8 år, konto 1260). Oppløs forskuddsbetalt husleie 73050 kr fra konto 1700 til konto 6300. Avsetning for skatt 22% av resultat 250000 kr.",
        "entities": {"ledger": {"subtask": "annual_close", "description": "Forenklet årsoppgjør 2025", "date": "2025-12-31",
            "assets": [
                {"name": "Inventar", "cost": 468300, "years": 10, "annual_amount": 46830, "asset_account": "1240", "depreciation_account": "6010", "accumulated_account": "1209"},
                {"name": "Kontormaskiner", "cost": 149750, "years": 10, "annual_amount": 14975, "asset_account": "1200", "depreciation_account": "6010", "accumulated_account": "1209"},
                {"name": "Kjøretøy", "cost": 385000, "years": 8, "annual_amount": 48125, "asset_account": "1260", "depreciation_account": "6010", "accumulated_account": "1209"}
            ],
            "prepaid_expenses": [{"amount": 73050, "from_account": "1700", "to_account": "6300", "description": "Oppløsning forskuddsbetalt husleie"}],
            "tax_provision": {"rate": 0.22, "amount": 55000, "expense_account": "8700", "payable_account": "2920"},
            "postings": []}},
        "confidence": 1.0, "request_id": "gold-annual-close-multi-nb", "git_sha": "gold"
    },
    # ── Multi-asset annual close (de) ──
    {
        "task_type": "ledger_task", "language": "de",
        "prompt": "Vereinfachter Jahresabschluss 2025: Abschreibung für drei Vermögenswerte: Büroausstattung (289500 NOK, 9 Jahre, Konto 1200), Fahrzeug (425000 NOK, 5 Jahre, Konto 1260), IT-Ausrüstung (183600 NOK, 3 Jahre, Konto 1250). Auflösung vorausbezahlter Miete 48000 NOK von Konto 1700 auf Konto 6300.",
        "entities": {"ledger": {"subtask": "annual_close", "description": "Vereinfachter Jahresabschluss 2025", "date": "2025-12-31",
            "assets": [
                {"name": "Büroausstattung", "cost": 289500, "years": 9, "annual_amount": 32166.67, "asset_account": "1200", "depreciation_account": "6010", "accumulated_account": "1209"},
                {"name": "Fahrzeug", "cost": 425000, "years": 5, "annual_amount": 85000, "asset_account": "1260", "depreciation_account": "6010", "accumulated_account": "1209"},
                {"name": "IT-Ausrüstung", "cost": 183600, "years": 3, "annual_amount": 61200, "asset_account": "1250", "depreciation_account": "6010", "accumulated_account": "1209"}
            ],
            "prepaid_expenses": [{"amount": 48000, "from_account": "1700", "to_account": "6300", "description": "Auflösung vorausbezahlte Miete"}],
            "tax_provision": None, "postings": []}},
        "confidence": 1.0, "request_id": "gold-annual-close-multi-de", "git_sha": "gold"
    },
    # ── Monthly close (nb) ──
    {
        "task_type": "ledger_task", "language": "nb",
        "prompt": "Utfør månedsslutt for mars 2026: 1) Tilbakefør periodisert kostnad 9000 kr fra konto 1700 til 6300. 2) Avskriv kontormaskiner: 289500 kr over 9 år. 3) Avsetning for lønn 5000 kr (konto 5000/2900).",
        "entities": {"ledger": {"subtask": "monthly_close", "description": "Månedsslutt mars 2026", "date": "2026-03-31",
            "assets": [{"name": "Kontormaskiner", "cost": 289500, "years": 9, "annual_amount": 32166.67, "asset_account": "1200", "depreciation_account": "6010", "accumulated_account": "1209"}],
            "prepaid_expenses": [{"amount": 9000, "from_account": "1700", "to_account": "6300", "description": "Tilbakeføring periodisert kostnad"}],
            "tax_provision": None,
            "postings": [{"account_number": "5000", "account_name": "Lønn", "amount": 5000}, {"account_number": "2900", "account_name": "Påløpt lønn", "amount": -5000}]}},
        "confidence": 1.0, "request_id": "gold-monthly-close-nb", "git_sha": "gold"
    },
    # ── Monthly close (pt) ──
    {
        "task_type": "ledger_task", "language": "pt",
        "prompt": "Fechamento mensal março 2026: 1) Reversão custo acumulado 9000 NOK conta 1700 para 6300. 2) Depreciação máquinas: 289500 NOK em 9 anos. 3) Provisão salário 5000 NOK (conta 5000/2900).",
        "entities": {"ledger": {"subtask": "monthly_close", "description": "Fechamento mensal março 2026", "date": "2026-03-31",
            "assets": [{"name": "Máquinas", "cost": 289500, "years": 9, "annual_amount": 32166.67, "asset_account": "1200", "depreciation_account": "6010", "accumulated_account": "1209"}],
            "prepaid_expenses": [{"amount": 9000, "from_account": "1700", "to_account": "6300", "description": "Reversão custo acumulado"}],
            "tax_provision": None,
            "postings": [{"account_number": "5000", "amount": 5000}, {"account_number": "2900", "amount": -5000}]}},
        "confidence": 1.0, "request_id": "gold-monthly-close-pt", "git_sha": "gold"
    },
    # ── Employee with salary/work% (nb) ──
    {
        "task_type": "create_employee", "language": "nb",
        "prompt": "Registrer ny ansatt basert på vedlagt arbeidsavtale: Navn: Hilde Bakken, Fødselsdato: 12.03.1988, Stilling: Lagermedarbeider, Stillingskode: 5223, Årslønn: 520 000 kr, Stillingsprosent: 100%, Startdato: 01.04.2026, E-post: hilde.bakken@firma.no",
        "entities": {"employee": {"first_name": "Hilde", "last_name": "Bakken", "email": "hilde.bakken@firma.no", "phone": None, "date_of_birth": "1988-03-12", "start_date": "2026-04-01", "annual_salary": 520000, "work_percent": 100, "job_title": "Lagermedarbeider", "occupation_code": "5223", "is_account_admin": None, "national_id_number": None, "identifier": None}},
        "confidence": 1.0, "request_id": "gold-employee-salary-nb", "git_sha": "gold"
    },
    # ── Employee with salary/work% (en) ──
    {
        "task_type": "create_employee", "language": "en",
        "prompt": "Register new employee from employment contract: Name: Sarah Johnson, DOB: 1990-05-22, Position: Accounting Manager, Annual salary: NOK 920000, Employment percentage: 80%, Start date: 2026-03-15, Email: sarah.johnson@company.com",
        "entities": {"employee": {"first_name": "Sarah", "last_name": "Johnson", "email": "sarah.johnson@company.com", "phone": None, "date_of_birth": "1990-05-22", "start_date": "2026-03-15", "annual_salary": 920000, "work_percent": 80, "job_title": "Accounting Manager", "occupation_code": None, "is_account_admin": None, "national_id_number": None, "identifier": None}},
        "confidence": 1.0, "request_id": "gold-employee-salary-en", "git_sha": "gold"
    },
    # ── Employee with salary/work% (pt) ──
    {
        "task_type": "create_employee", "language": "pt",
        "prompt": "Registrar novo funcionário conforme contrato: Nome: Ana Silva, Nascimento: 15/08/1985, Cargo: Assistente Financeiro, Salário anual: 660000 NOK, Percentual: 100%, Início: 01/05/2026, E-mail: ana.silva@empresa.pt",
        "entities": {"employee": {"first_name": "Ana", "last_name": "Silva", "email": "ana.silva@empresa.pt", "phone": None, "date_of_birth": "1985-08-15", "start_date": "2026-05-01", "annual_salary": 660000, "work_percent": 100, "job_title": "Assistente Financeiro", "occupation_code": None, "is_account_admin": None, "national_id_number": None, "identifier": None}},
        "confidence": 1.0, "request_id": "gold-employee-salary-pt", "git_sha": "gold"
    },
    # ── Annual close (en) ──
    {
        "task_type": "ledger_task", "language": "en",
        "prompt": "Perform simplified annual closing for 2025: Depreciate two assets: Office Equipment (350000 NOK, 7 years, account 1200) and Vehicles (480000 NOK, 6 years, account 1260). Resolve prepaid insurance 36000 NOK from account 1710 to 6340.",
        "entities": {"ledger": {"subtask": "annual_close", "description": "Simplified annual closing 2025", "date": "2025-12-31",
            "assets": [
                {"name": "Office Equipment", "cost": 350000, "years": 7, "annual_amount": 50000, "asset_account": "1200", "depreciation_account": "6010", "accumulated_account": "1209"},
                {"name": "Vehicles", "cost": 480000, "years": 6, "annual_amount": 80000, "asset_account": "1260", "depreciation_account": "6010", "accumulated_account": "1209"}
            ],
            "prepaid_expenses": [{"amount": 36000, "from_account": "1710", "to_account": "6340", "description": "Resolve prepaid insurance"}],
            "tax_provision": None, "postings": []}},
        "confidence": 1.0, "request_id": "gold-annual-close-multi-en", "git_sha": "gold"
    },
]

if __name__ == "__main__":
    with open(GOLD_PATH, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Added {len(examples)} gold examples. Total: {sum(1 for _ in open(GOLD_PATH))}")
