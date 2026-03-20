# Ledger Logic Agent

AI agent that interprets natural-language accounting prompts (in any language) and executes the correct Tripletex API calls to create customers, suppliers, employees, departments, invoices, and payments.

Built for the [AINM Tripletex competition](https://app.ainm.no/submit/tripletex).

---

## Architecture

```
User Request → LLM Parser → Task Planner → API Executor → Validation → Episode Log
```

See [docs/architecture.md](docs/architecture.md) for the full design, all decisions made during development, and known API quirks.

---

## Project Structure

```
.
├── agent/                  # Core agent package
│   ├── client.py           # Tripletex REST API wrapper
│   ├── executor.py         # Task workflows (create_customer, create_invoice, …)
│   ├── models.py           # Pydantic models for request/response
│   ├── parser.py           # LLM-based intent parser (Gemini 2.5 Flash)
│   └── validators.py       # Post-execution result verification
├── docs/
│   ├── architecture.md     # Design decisions and change log
│   └── initial-architecture.png
├── logs/
│   └── episodes.jsonl      # Gold dataset — one JSON line per request (gitignored)
├── scripts/
│   └── analyze.py          # Fetch Cloud Logging episodes → actionable insights
├── tests/
│   ├── simulate_evaluator.py  # Full competition simulator (14 tasks, 6 languages)
│   ├── shadow_check.py        # Read-only verification of existing Tripletex data
│   ├── test_runner.py         # Integration tests against sandbox
│   └── test_sandbox.py        # Low-level sandbox connectivity tests
├── main.py                 # FastAPI app — POST /, POST /solve, GET /health
├── Dockerfile              # linux/amd64 image for Cloud Run
├── deploy.sh               # Build → push → deploy to Cloud Run
├── Makefile                # Common dev tasks (see `make help`)
├── openapi.json            # Tripletex OpenAPI spec (reference)
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
# → {"status":"ok","version":"1.0.0","git_sha":"ledger-logic-agent-00011-c7m"}
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
| `create_customer` | Create a customer record in Tripletex |
| `create_supplier` | Create a supplier (customer with `isSupplier=true`) |
| `create_employee` | Create an employee, optionally assigned to an existing department |
| `create_department` | Create a department |
| `create_invoice` | Create a customer invoice and send it |
| `register_payment` | Register a payment against an invoice |

All tasks support multilingual prompts (Norwegian, English, German, French, Portuguese, and more).

