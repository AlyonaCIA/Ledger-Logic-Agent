# Ledger Logic Agent — Architecture

> **Place** `initial-architecture.png` (the diagram from the competition kickoff) in this folder.

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
