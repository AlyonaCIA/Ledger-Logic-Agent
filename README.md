# Ledger Logic Agent

AI agent that interprets natural-language accounting prompts (in **7 languages**) and executes the correct Tripletex ERP API calls — with self-healing retries, few-shot learning, and a full offline gold loop.

Built for the [AINM Tripletex competition](https://app.ainm.no/submit/tripletex).  
**Deployed revision:** `ledger-logic-agent-00016-lfj` · **Local simulator:** 12/14 tasks ✓

---

## Architecture

```
                        ┌────────────────────────────────────────────────────┐
                        │                  FastAPI  (main.py)                │
                        │   POST /   ·   POST /solve   ·   GET /health       │
                        └───────────────────────┬────────────────────────────┘
                                                │
                        ┌───────────────────────▼────────────────────────────┐
                        │              LLM Parser  (parser.py)               │
                        │   gemini-2.5-flash · 7 languages · zero-temp       │
                        │   ┌──────────────────────────────────────────────┐ │
                        │   │  few-shot retrieval  ← gold/prompts.jsonl    │ │
                        │   │  + OCR / PDF extraction  (PyMuPDF)           │ │
                        │   └──────────────────────────────────────────────┘ │
                        │   → { task_type, entities, language, confidence }  │
                        └───────────────────────┬────────────────────────────┘
                                                │
                        ┌───────────────────────▼────────────────────────────┐
                        │           Task Executor  (executor.py)             │
                        │   16 workflow handlers · deterministic dispatch     │
                        │   ┌──────────────────────────────────────────────┐ │
                        │   │  Entity Resolver  (matchers.py)              │ │
                        │   │  resolve_customer / employee / invoice       │ │
                        │   └──────────────────────────────────────────────┘ │
                        │   ┌──────────────────────────────────────────────┐ │
                        │   │  Self-Healer  (healer.py)          ← NEW     │ │
                        │   │  422 → Gemini fixes payload → retry once     │ │
                        │   └──────────────────────────────────────────────┘ │
                        │   ┌──────────────────────────────────────────────┐ │
                        │   │  Stable-ID cache  (process-level dict)       │ │
                        │   │  department_id · payment_type (no repeat GETs│ │
                        │   └──────────────────────────────────────────────┘ │
                        └───────────────────────┬────────────────────────────┘
                                                │
                        ┌───────────────────────▼────────────────────────────┐
                        │   Tripletex REST Client  (client.py)               │
                        │   budget 25 calls · call log · 4xx/5xx tracking    │
                        └───────────────────────┬────────────────────────────┘
                                                │
                        ┌───────────────────────▼────────────────────────────┐
                        │   Episode Log  (logs/episodes.jsonl)               │
                        └───────────────────────┬────────────────────────────┘
                                                │
              ┌──────────────────────────────── ▼ ─── offline learning loop ──────────────────────────┐
              │   make gold       gold_builder.py  →  gold/prompts.jsonl  (injected as few-shots)      │
              │   make critique   critic.py        →  logs/critiques.jsonl (LLM-as-judge on failures)  │
              │   make analyze    analyze.py        →  actionable insights from Cloud Logging           │
              └───────────────────────────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for design decisions, API quirks, and the full change log.

---

## Project Structure

```
.
├── agent/
│   ├── client.py           # Tripletex REST wrapper · call budget · 4xx/5xx tracking
│   ├── executor.py         # 16 task workflows · self-healing helper
│   ├── healer.py           # Gemini-powered 422 payload repair (self-healing)
│   ├── matchers.py         # Entity resolvers — customer / employee / invoice / dept
│   ├── models.py           # Pydantic request/response models
│   ├── parser.py           # LLM intent parser · few-shot retrieval · OCR
│   └── validators.py       # Pre-execution field validation
├── docs/
│   └── architecture.md     # Design decisions and change log
├── gold/
│   ├── .gitkeep
│   └── prompts.jsonl       # Verified few-shot examples (gitignored, built by make gold)
├── logs/
│   └── episodes.jsonl      # One JSON line per request (gitignored)
├── scripts/
│   ├── analyze.py          # Cloud Logging → insights
│   ├── critic.py           # LLM-as-judge on failing episodes
│   └── gold_builder.py     # Filter passing episodes → gold/prompts.jsonl
├── tests/
│   ├── simulate_evaluator.py   # Full competition simulator (14 tasks, 7 languages)
│   ├── shadow_check.py         # Read-only sandbox verification
│   ├── test_runner.py          # Integration tests
│   └── test_sandbox.py         # Connectivity tests
├── main.py                 # FastAPI app — POST /, POST /solve, GET /health
├── Dockerfile              # linux/amd64 for Cloud Run
├── deploy.sh               # Build → push → deploy
├── Makefile                # Dev targets (make help)
└── pyproject.toml          # Dependencies (uv)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- A [Gemini API key](https://aistudio.google.com/apikey) (for local dev)
- Tripletex sandbox credentials

### Setup

```bash
git clone https://github.com/your-org/ledger-logic-agent
cd ledger-logic-agent

# Create virtualenv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and TRIPLETEX_* vars
```

### Run locally

```bash
make dev                    # hot-reload on http://localhost:8080
```

### Test locally

```bash
# Full competition simulator (requires local server running)
make simulate

# Integration tests against sandbox
make test

# Shadow check (read-only, no writes)
make shadow
```

### All available targets

```
make help
```

---

## Deployment

### Deploy to Cloud Run

```bash
./deploy.sh                 # build Docker image → push to GCR → deploy
./deploy.sh smoke           # verify the deployed service responds correctly
```

### Verify deployed revision

```bash
curl https://ledger-logic-agent-xrgiacpg2q-ew.a.run.app/health
# → {"status":"ok","version":"1.0.0","git_sha":"ledger-logic-agent-00016-lfj"}
```

**Rule:** the `git_sha` must change after every deploy before submitting to the competition.

### Before every competition submit

1. `git status` — confirm all changes are committed
2. `make simulate` — confirm local score ≥ expected
3. `./deploy.sh` — build and deploy
4. `./deploy.sh smoke` — verify live endpoint  
5. `curl .../health | jq .git_sha` — confirm new revision
6. Submit at [app.ainm.no/submit/tripletex](https://app.ainm.no/submit/tripletex)
7. `make analyze` — review episodes after 10–15 min

---

## Episode Logs (Gold Dataset)

Every request appends a structured JSON line to `logs/episodes.jsonl`:

```json
{
  "request_id": "a1b2c3d4",
  "timestamp": "2026-03-20T09:30:00Z",
  "git_sha": "ledger-logic-agent-00011-c7m",
  "prompt": "Registrer kunden Bergvik AS med organisasjonsnummer 910473166",
  "task_type": "create_customer",
  "language_detected": "nb",
  "confidence": 1.0,
  "missing_fields": [],
  "parsed_entities": { "customer": { "name": "Bergvik AS", ... } },
  "api_calls": [{ "method": "POST", "path": "/customer", "status": 201, "ms": 312 }],
  "metrics": { "calls": 1, "error_4xx": 0, "error_5xx": 0, "latency_api_ms": 312 },
  "outcome": "completed",
  "error": null
}
```

To pull episodes from Cloud Logging and analyze them:

```bash
make logs-fetch             # fetch all stored episodes
make analyze                # fetch last 30 min and print insights
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Local only | AI Studio key — not needed on Cloud Run (uses Vertex AI via ADC) |
| `API_KEY` | Optional | Bearer token to protect `/solve` |
| `GCP_PROJECT_ID` | Cloud Run | GCP project for Vertex AI |
| `VERTEX_LOCATION` | Cloud Run | Region for Vertex AI (e.g. `europe-west1`) |

---

## Supported Task Types

| Task | Description |
|---|---|
| `create_customer` | Create a customer (or supplier with `isSupplier=true`) |
| `update_customer` | Update customer fields |
| `delete_customer` | Delete a customer |
| `create_employee` | Create an employee; optionally assign as account admin |
| `update_employee` | Update employee fields |
| `delete_employee` | Remove an employee |
| `create_product` | Create a product/service with price and VAT rate |
| `create_invoice` | Customer → Order → Invoice → Send (with self-healing on 422) |
| `register_payment` | Register a payment against an existing invoice |
| `create_credit_note` | Reverse an invoice with a credit note |
| `create_travel_expense` | File a travel expense report for an employee |
| `delete_travel_expense` | Delete a travel expense |
| `create_project` | Create a billable project linked to a customer |
| `create_department` | Create a department (idempotent — searches before creating) |
| `enable_module` | Enable department / project / travel-expense accounting |
| `delete_voucher` | Delete / reverse a ledger voucher |

All tasks support multilingual prompts: **nb · nn · en · es · pt · de · fr**.

