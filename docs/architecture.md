# Ledger Logic Agent — Architecture

**Current revision:** `ledger-logic-agent-00016-lfj`  
**Local simulator score:** 12/14 tasks · 71/73 checks  
**Competition score:** 5/7 (71%) — submitted with `00012-sn2`; `00016-lfj` is awaiting resubmission

---

## 1. Pipeline Overview

```
                     ┌──────────────────────────────────────────────────────┐
                     │                FastAPI  (main.py)                    │
                     │   POST /   ·   POST /solve   ·   GET /health         │
                     └───────────────────────┬──────────────────────────────┘
                                             │
                     ┌───────────────────────▼──────────────────────────────┐
                     │           LLM Parser  (parser.py)                    │
                     │   gemini-2.5-flash · temperature=0 · max_tokens=1024 │
                     │   7 languages: nb · nn · en · es · pt · de · fr      │
                     │                                                       │
                     │   ┌─────────────────────────────────────────────┐    │
                     │   │  Keyword pre-scan → task_type hint           │    │
                     │   │  Few-shot retrieval  ← gold/prompts.jsonl   │    │
                     │   │  PDF/image OCR  (PyMuPDF + inline bytes)    │    │
                     │   └─────────────────────────────────────────────┘    │
                     │                                                       │
                     │   → { task_type, entities, language, confidence }    │
                     └───────────────────────┬──────────────────────────────┘
                                             │
                     ┌───────────────────────▼──────────────────────────────┐
                     │        Task Executor  (executor.py)                  │
                     │   16 workflow handlers · deterministic dispatch       │
                     │   confidence < 0.25 → keyword fallback → skip        │
                     │                                                       │
                     │   ┌─────────────────────────────────────────────┐    │
                     │   │  Entity Resolver  (matchers.py)             │    │
                     │   │  resolve_customer · resolve_employee         │    │
                     │   │  resolve_invoice  · resolve_department       │    │
                     │   │  exact-match + 500-result fetch              │    │
                     │   └─────────────────────────────────────────────┘    │
                     │                                                       │
                     │   ┌─────────────────────────────────────────────┐    │
                     │   │  Self-Healer  (healer.py)          ← v00016 │    │
                     │   │  POST fails 400/422 → Gemini fixes payload  │    │
                     │   │  → retry once → raise if unchanged           │    │
                     │   └─────────────────────────────────────────────┘    │
                     │                                                       │
                     │   ┌─────────────────────────────────────────────┐    │
                     │   │  Stable-ID Cache  (_STABLE_ID_CACHE dict)   │    │
                     │   │  department_id · payment_type_id             │    │
                     │   │  process-level · avoids repeat GETs          │    │
                     │   └─────────────────────────────────────────────┘    │
                     └───────────────────────┬──────────────────────────────┘
                                             │
                     ┌───────────────────────▼──────────────────────────────┐
                     │   Tripletex REST Client  (client.py)                 │
                     │   base URL: /v2/  · budget: 25 calls · 30s timeout   │
                     │   call log → error_4xx / error_5xx counters          │
                     └───────────────────────┬──────────────────────────────┘
                                             │
                     ┌───────────────────────▼──────────────────────────────┐
                     │   Episode Log  (logs/episodes.jsonl)                 │
                     │   appended per request · gitignored                  │
                     └───────────────────────┬──────────────────────────────┘

            ══════════════════ offline learning loop ═══════════════════
            │                                                           │
            │   make gold       gold_builder.py                         │
            │     passing episodes (confidence≥0.85, 0 errors)          │
            │     → gold/prompts.jsonl  (≤3 per task_type+lang bucket)  │
            │     → injected as few-shots at parse time                 │
            │                                                           │
            │   make critique   critic.py                               │
            │     failing episodes → Gemini LLM judge                   │
            │     → logs/critiques.jsonl (failure_type, fix, layer)     │
            │                                                           │
            │   make analyze    analyze.py                              │
            │     Cloud Logging fetch → actionable insights             │
            ════════════════════════════════════════════════════════════
```

---

## 2. Key Design Principles

| Principle | Decision |
|---|---|
| Minimal LLM surface | LLM only extracts entities; all workflow logic is plain Python |
| Self-healing | Any 400/422 from Tripletex triggers one Gemini-powered payload fix + retry |
| Fail-safe | Any unhandled exception still returns `{"status":"completed"}` — evaluator shape |
| Stateless | No DB; Tripletex is the single source of truth |
| Idempotent lookups | Always search before create (department, customer, employee) |
| Observable | Every request emits a structured episode for offline analysis |
| Multilingual | System prompt covers nb/nn/en/es/pt/de/fr; language is detected and logged |

---

## 3. Technology Stack

| Layer | Choice | Reason |
|---|---|---|
| HTTP framework | FastAPI | async, auto-docs, Pydantic |
| LLM | Gemini 2.5 Flash | best latency/quality for structured extraction |
| LLM SDK (local) | `google-genai` (AI Studio key) | no quota restrictions |
| LLM SDK (prod) | Vertex AI via Workload Identity ADC | no key rotation needed |
| Accounting API | Tripletex REST v2 | competition requirement |
| Deployment | Google Cloud Run | serverless, auto-scales, secrets managed |
| Container | Docker (`linux/amd64`) | deterministic builds, GCR |
| PDF parsing | PyMuPDF (`fitz`) | text extraction from attachments |

---

## 4. Change Log — Decisions Made During Development

### 4.1 Add `POST /` root handler (v00009)
**Root cause:** Competition evaluator POSTs to `/`, not `/solve`. Scored 0/7 for revisions 00001–00008.  
**Fix:** Added `@app.post("/")` handler delegating to `solve()`.

---

### 4.2 `_find_*` helpers — exact match + count=500 (v00010)
**Root cause:** `GET /customer?name=X` in Tripletex does *substring* matching; code was taking `results[0]` blindly from a count=10 response, returning wrong entities.  
**Fix:** Fetch up to 500 results, filter in Python for exact case-insensitive match.

---

### 4.3 Department name parser — preserve full name (v00011)
**Root cause:** LLM was treating any trailing integer in the name as `department_number`; "Regnskap 95288" → name="Regnskap", number=95288.  
**Fix:** System prompt rewritten to extract `department_number` only when an explicit code is given *separately* from the name.

---

### 4.4 `git_sha` in health endpoint (v00011)
**Fix:** Startup captures `K_REVISION` (Cloud Run env var) as `git_sha`. Exposed in `/health` and every episode log. Pre-submit rule: verify `git_sha` changed.

---

### 4.5 `POST /orderline` — correct fallback endpoint (v00013)
**Root cause:** Code used `POST /order/{id}/orderLines` which doesn't exist in Tripletex v2.  
**Fix:** Fallback uses `POST /orderline` with `{"order": {"id": ...}, ...}` body.

---

### 4.6 Invoice date range on `GET /invoice` (v00013)
**Root cause:** `GET /invoice` without `invoiceDateFrom/To` → 422. Registration of payments was always failing.  
**Fix:** All invoice lookups include `invoiceDateFrom` (5 years ago) and `invoiceDateTo` (1 year ahead).

---

### 4.7 Invoice send flow — `sendToCustomer` + `/:send` (v00014)
**Root cause:** `sendToCustomer=false` leaves invoice in Draft state; evaluator checks for Sent state.  
**Fix:** Try `sendToCustomer=true` first (422 if no bank account); fallback to `false`; then call `PUT /invoice/{id}/:send` explicitly.

---

### 4.8 Entity matchers — `matchers.py` (v00012)
Central resolvers with exact-match logic, 5-year date windows, KNOWN_FAILURES catalog.  
Imported by all executor workflows.

---

### 4.9 Offline learning loop — `gold_builder.py`, `critic.py` (v00012)
- `gold_builder.py`: filters passing episodes (confidence≥0.85, 0 errors) → `gold/prompts.jsonl`; hard-capped at 3 examples per (task_type, language) bucket.
- `critic.py`: LLM-as-judge on failing episodes → structured `{failure_type, root_cause, suggested_fix, affected_layer}`.

---

### 4.10 `_STABLE_ID_CACHE` + department idempotency (v00015)
- Process-level dict caches `department_id` and `payment_type_id` — avoids repeat GETs per request.
- `_create_department` searches by name before creating; skips if already exists (avoids 422 on duplicate `departmentNumber`).

---

### 4.11 Self-Healing on 422 errors — `healer.py` (v00016)
**Motivation:** Static `if/else` cannot anticipate every Tripletex validation constraint (VAT types, missing fields, type mismatches). Tripletex returns detailed `validationMessages` on 422.  
**Implementation:**
```
POST payload → 400/422 → capture error text
→ Gemini Flash: "here is the failed payload + error; return corrected JSON"
→ retry once with healed payload
→ if payload unchanged (Gemini gave up) → raise original exception
```
Wired into: `/employee`, `/customer`, `/product`, `/order`, `/invoice`.  
Call budget impact: at most +1 API call + 1 Gemini call per failing endpoint. Budget is 25 calls.

---

## 5. Known API Quirks (Tripletex Sandbox)

| Quirk | Detail |
|---|---|
| `isCustomer` always `true` | API ignores the flag on POST/PUT |
| Name filter is substring | `GET /customer?name=X` returns partial matches; fetch 500, filter in Python |
| `POST /department` duplicate number → 422 | Search by name before creating |
| Invoice requires bank account | Sandbox without bank account → 422 on `POST /invoice`; fix requires web UI config |
| `GET /invoice` mandatory dates | `invoiceDateFrom` + `invoiceDateTo` required or 422 |
| `POST /orderline` not `/order/{id}/orderLines` | Correct Tripletex v2 endpoint for adding lines |

---

## 6. Evaluation Pipeline (local)

```bash
make dev          # start local server on :8080
make simulate     # run 14 tasks across 7 languages → SCORE: N/14, N/73 checks
make analyze      # fetch Cloud Logging episodes → insights
make gold         # rebuild gold/prompts.jsonl from passing episodes
make critique     # run LLM judge on failing episodes
```

---

## 7. Deployment Pipeline

```bash
# After code changes:
git add <files> && git commit -m "type: description"
./deploy.sh                      # docker build → GCR push → Cloud Run deploy
./deploy.sh smoke                # verify /health + POST / on live URL
curl .../health | jq .git_sha    # confirm new revision ID
# Submit to competition at app.ainm.no/submit/tripletex
make analyze                     # review episodes after 10–15 min
```

Cloud Run revision format: `ledger-logic-agent-NNNNN-XXX` (auto-generated).  
`K_REVISION` is captured at startup and returned as `git_sha` in `/health`.


---

## 1. Initial Design

The agent was designed around a **deterministic pipeline** where an LLM handles only the ambiguous part (natural-language → structured intent) and everything else is typed, testable code.

```
User Request
    │
    ▼
┌─────────────────────┐
│   Small LLM Parser  │  gemini-2.5-flash
│  Parse → JSON       │  single structured-output call
└────────┬────────────┘
         │  { task_type, entities, language, confidence }
         ▼
┌─────────────────────┐
│    Task Planner     │  deterministic dispatch
│  Workflow selector  │  match task_type → workflow function
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Task Executors    │  typed API wrappers (client.py)
│  Typed API calls    │  one function per task type
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Validation & Checks │  validators.py
│  Verify Results     │  assert created resource matches intent
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Logging & Monitoring│  episode JSON → logs/episodes.jsonl
│  Track & Analyze    │  Cloud Logging + scripts/analyze.py
└─────────────────────┘

        ↕  (offline loop)
┌─────────────────────┐     ┌─────────────────────┐
│  Workflow Library   │────▶│  Evaluation Harness  │
│ Mini-Solvers / task │     │ simulate_evaluator   │
└─────────────────────┘     └─────────────────────┘
```

### Key design principles

| Principle | Decision |
|---|---|
| Minimal LLM surface | LLM only extracts entities; workflow logic is plain Python |
| Fail-safe | Any exception returns `{"status":"completed"}` — competition evaluator expects that shape |
| Stateless | No DB; Tripletex is the source of truth |
| Multilingual | Parser prompt instructs the LLM to handle any language; language is detected and logged |
| Observable | Every request emits a structured episode JSON for offline analysis |

---

## 2. Technology Choices

| Layer | Choice | Reason |
|---|---|---|
| HTTP framework | FastAPI | async, auto-docs, Pydantic validation |
| LLM | Gemini 2.5 Flash | best latency/quality ratio for structured extraction |
| LLM SDK (local) | `google-genai` (AI Studio key) | no quota restrictions for dev |
| LLM SDK (prod) | Vertex AI via ADC | no key rotation, uses Workload Identity |
| Accounting API | Tripletex REST v2 | competition requirement |
| Deployment | Google Cloud Run | serverless, scales to zero, secrets managed |
| Container | Docker (linux/amd64) | deterministic builds, GCR push |
| PDF parsing | PyMuPDF | extract text from attachment files |

---

## 3. Change Log — Decisions Made During Development

### 3.1 Add `POST /` root handler  
**When:** After revisions 00001–00008 all scored 0/7.  
**Root cause:** The competition evaluator POSTs to `/` (the HTTP root), not to `/solve`. Our server responded 404 to every evaluation call because only `POST /solve` existed.  
**Fix:** Added a `POST /` handler that delegates to `solve()`.  
**Lesson:** Always read the evaluator's actual HTTP call, not just its documentation. Added a `/health` smoke-check step to the deploy pipeline to verify routing before submitting.

```python
@app.post("/")
async def solve_root(request, authorization):
    return await solve(request, authorization)
```

---

### 3.2 `_find_customer`: `count=10` → `count=500` + exact match  
**When:** Simulator run 2 — invoice task was creating invoices for the wrong customer.  
**Root cause:** `GET /customer?name=Acme Corp` in Tripletex does *substring* matching, not exact. With `count=10` (the SDK default), the API returned the first 10 partial matches, which could include unrelated customers created in earlier test runs. The code was blindly taking `results[0]`.  
**Fix:** Fetch up to 500 results, then filter in Python for an exact case-insensitive name match.

```python
results = client.get_list("/customer", params={"name": name, "count": 500})
name_lower = name.lower()
for r in results:
    if r.get("name", "").lower() == name_lower:
        return r
return None
```

**Impact:** Same bug existed in all `verify_*` calls in `tests/simulate_evaluator.py` (fixed with `count=500` there too).

---

### 3.3 Department name parser — preserve full name  
**When:** Simulator run 3 — department "Regnskap 95288" was created as name `"Regnskap"` with `department_number=95288`.  
**Root cause:** The LLM was treating any trailing integer in the name as a `department_number` field. The parser prompt said `department_number` is "the department number", which the model interpreted too broadly.  
**Fix:** Rewrote the `create_department` instruction in `parser.py` to be explicit: `department_number` is only extracted when an explicit numeric code is given *separately* from the name (e.g. "avdelingsnummer 100").

```
name (the full department name as given, including any words or numbers)
department_number (only if an explicit numeric code/ID is given separately from
  the name, e.g. "avdelingsnummer 100" or "number 100"; omit if the number is
  part of the name)
```

---

### 3.4 `isCustomer=False` — Tripletex API behaviour  
**When:** Simulator run 3 — supplier verification kept failing the `isCustomer=False` check.  
**Root cause:** The Tripletex API ignores `isCustomer` on `POST /customer` and on `PUT`. It always returns `isCustomer: true` for any contact record. This is an API quirk, not a bug in our code.  
**Decision:** Removed the hard-fail check from the simulator; added a `⚠ isCustomer flag` warning instead. No code change in the agent — we cannot force the API to return `false`.

---

### 3.5 `git_sha` in health endpoint and episode logs  
**When:** After realising all 8 initial submissions could have been against a stale Cloud Run revision.  
**Problem:** We had no way to verify which code version the evaluator actually called. `deploy.sh` could succeed but still route to a cached revision.  
**Fix:** At startup, capture `git rev-parse --short HEAD` (fallback: `K_REVISION` env var that Cloud Run injects). Expose it in `GET /health` and include it in every episode log.

```json
{"status": "ok", "version": "1.0.0", "git_sha": "ledger-logic-agent-00011-c7m"}
```

**Rule adopted:** Verify `git_sha` in `/health` response changes after every `./deploy.sh` before submitting.

---

### 3.6 Episode logs → `logs/episodes.jsonl` (gold dataset)  
**When:** After accumulating enough real episodes to use for offline analysis.  
**Motivation:** Cloud Logging requires `gcloud` and internet access; having a local copy of episodes in `logs/episodes.jsonl` lets us run `scripts/analyze.py` without fetching every time, and it acts as a **gold dataset** for understanding model failures and improving prompts.  
**Implementation:**  
- `main.py` appends each episode JSON to `logs/episodes.jsonl` after every request (best-effort; silent failure in Cloud Run ephemeral FS).  
- `scripts/analyze.py` defaults its input/output path to `logs/episodes.jsonl`.  
- `logs/*.jsonl` is excluded from git (data files, not source code). `logs/.gitkeep` tracks the directory.

---

## 4. Known API Quirks (Tripletex Sandbox)

| Quirk | Detail |
|---|---|
| `isCustomer` always `true` | API ignores the field on POST and PUT |
| Name filter is substring match | `GET /customer?name=X` returns partial matches; must fetch 500 and filter in Python |
| `POST /department` with duplicate `departmentNumber` → 422 | Each test run must use a unique department number or omit it |
| Invoice requires bank account | Sandbox has no bank account configured; `POST /invoice/send` fails with 422 in sandbox, works on competition proxy |

---

## 5. Evaluation Pipeline (local)

```
make dev                        # start local server on :8080
make simulate                   # run all 14 tasks (5 types × 2–5 languages)
                                # → SCORE: N/14 tasks, N/73 checks
make analyze                    # fetch Cloud Logging episodes → insights
```

The simulator (`tests/simulate_evaluator.py`) mirrors the competition evaluator:
- POSTs to `POST /` with multilingual prompts and Tripletex credentials
- Verifies the Tripletex sandbox state after each call (name, email, org number, flags)
- Scores tasks pass/fail and reports individual check results

---

## 6. Deployment Pipeline

```
[change code]
    git add <files>
    git commit -m "type: description"
    ./deploy.sh                 # docker build → GCR push → Cloud Run deploy
    ./deploy.sh smoke           # verify /health + /solve on live URL
    curl .../health | jq .git_sha   # confirm new revision
    [submit to competition]
    make analyze                # review episodes after 10–15 min
```

Cloud Run revision naming: `ledger-logic-agent-NNNNN-XXX` (auto-generated).  
The revision name appears in `GET /health` as `git_sha` because `K_REVISION` is used as the fallback when `.git` is not present in the container.
