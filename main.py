"""
Ledger Logic Agent – FastAPI entry point.

POST /solve   →  parse accounting prompt → execute Tripletex API calls → 200 {"status":"completed"}
GET  /health  →  liveness probe
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

load_dotenv()

# Capture git SHA at startup so every episode log carries it.
try:
    _GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    _GIT_SHA = os.getenv("K_REVISION", "unknown")  # Cloud Run sets K_REVISION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Separate logger that emits one JSON line per episode (easy to grep/export).
episode_logger = logging.getLogger("episode")

# ── Local episode log file (gold dataset for offline analysis) ────────────
# In Cloud Run the filesystem is ephemeral; use `make logs-fetch` to pull from
# Cloud Logging instead.  Locally this file accumulates across restarts.
_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)
_EPISODE_FILE = _LOGS_DIR / "episodes.jsonl"

# Optional API key to protect the /solve endpoint (set via env var API_KEY).
_API_KEY: str | None = os.getenv("API_KEY")

# Internal hard deadline: 240 s → leaves 60 s buffer before the 300 s competition timeout.
_INTERNAL_TIMEOUT_S = 240


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("Ledger Logic Agent started")
    yield
    logger.info("Ledger Logic Agent stopped")


app = FastAPI(title="Ledger Logic Agent", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ #
# /solve  – main competition endpoint
# ------------------------------------------------------------------ #

@app.post("/solve")
async def solve(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    t_start = time.monotonic()

    # ── Optional Bearer token auth ────────────────────────────────────
    if _API_KEY:
        expected = f"Bearer {_API_KEY}"
        if not authorization or authorization != expected:
            logger.warning("Unauthorized /solve request")
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # ── Parse request body ────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception as exc:
        logger.error(f"Malformed JSON body: {exc}")
        return JSONResponse({"status": "completed"})

    from agent.client import TripletexClient
    from agent.executor import execute_task
    from agent.models import SolveRequest
    from agent.parser import parse_task

    request_id = str(uuid.uuid4())[:8]

    try:
        req = SolveRequest(**body)
    except Exception as exc:
        logger.error(f"[{request_id}] Request validation error: {exc}")
        return JSONResponse({"status": "completed"})

    logger.info(f"[{request_id}] PROMPT ({len(req.prompt)}c): {req.prompt[:300]}")
    logger.info(f"[{request_id}] FILES: {[f.filename for f in req.files]}")

    client = TripletexClient(
        base_url=req.tripletex_credentials.base_url,
        session_token=req.tripletex_credentials.session_token,
    )

    task_intent: dict = {}
    outcome = "completed"
    error_msg: str | None = None

    try:
        async def _run() -> None:
            nonlocal task_intent
            task_intent = await parse_task(req.prompt, req.files)
            task_intent["_raw_prompt"] = req.prompt  # for keyword fallback
            await asyncio.get_event_loop().run_in_executor(
                None, execute_task, task_intent, client
            )

        await asyncio.wait_for(_run(), timeout=_INTERNAL_TIMEOUT_S)

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t_start
        outcome = "timeout"
        error_msg = f"internal timeout after {elapsed:.1f}s"
        logger.error(f"[{request_id}] {error_msg}")
    except Exception as exc:
        outcome = "error"
        error_msg = str(exc)
        logger.error(f"[{request_id}] Task execution error: {exc}", exc_info=True)

    elapsed_total_ms = round((time.monotonic() - t_start) * 1000)
    m = client.metrics_summary()

    # ── Structured SUMMARY log (single line, grep-friendly) ──────────
    logger.info(
        f"[{request_id}] SUMMARY | task={task_intent.get('task_type','unknown')} "
        f"lang={task_intent.get('language_detected','?')} "
        f"conf={task_intent.get('confidence',0):.2f} "
        f"calls={m['call_count']} 4xx={m['error_4xx']} 5xx={m['error_5xx']} "
        f"latency_api={m['latency_ms_total']}ms total={elapsed_total_ms}ms"
    )

    # ── Full episode record (one JSON line → Cloud Logging / export) ──
    episode = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _GIT_SHA,
        "prompt": req.prompt[:500],
        "files": [f.filename for f in req.files],
        "task_type": task_intent.get("task_type", "unknown"),
        "language_detected": task_intent.get("language_detected", "?"),
        "confidence": task_intent.get("confidence", 0),
        "missing_fields": task_intent.get("missing_fields", []),
        "parsed_entities": {
            k: v for k, v in task_intent.items()
            if k not in {"task_type", "language_detected", "confidence", "missing_fields"}
        },
        "api_calls": client.call_log,
        "metrics": {
            "calls": m["call_count"],
            "error_4xx": m["error_4xx"],
            "error_5xx": m["error_5xx"],
            "latency_api_ms": m["latency_ms_total"],
            "latency_total_ms": elapsed_total_ms,
        },
        "outcome": outcome,
        "error": error_msg,
    }
    episode_logger.info("EPISODE %s", json.dumps(episode, ensure_ascii=False))

    # Append to local gold-dataset file (best-effort; silent in Cloud Run).
    try:
        with _EPISODE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode, ensure_ascii=False) + "\n")
    except OSError:
        pass

    return JSONResponse({"status": "completed"})


# Competition evaluator may POST to root / instead of /solve
@app.post("/")
async def solve_root(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse:
    return await solve(request, authorization)


# ------------------------------------------------------------------ #
# /health  – liveness probe
# ------------------------------------------------------------------ #

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "1.0.0", "git_sha": _GIT_SHA})
