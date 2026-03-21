from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Hard cap: abort if we ever exceed this many API calls in one request.
# Bank reconciliation and complex tasks may need up to ~40 calls.
MAX_CALLS_PER_REQUEST = 45


class BudgetExceededError(RuntimeError):
    """Raised when the per-request API call budget is exhausted."""


class TripletexClient:
    """
    Instrumented wrapper around the Tripletex v2 REST API.

    Auth:  HTTP Basic  username="0"  password=session_token
    Every response envelope:
        GET many  -> {"fullResultSize": N, "values": [...]}
        GET one   -> {"value": {...}}
        POST/PUT  -> {"value": {...}}

    Metrics exposed after execution:
        call_count          total API calls made
        error_4xx           client errors (400, 404, 422 …)
        error_5xx           server errors (500 …)
        latency_ms_total    cumulative HTTP latency in ms
        calls_by_endpoint   {method:path -> count}
    """

    def __init__(
        self,
        base_url: str,
        session_token: str,
        max_calls: int = MAX_CALLS_PER_REQUEST,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token  # exposed for cache-keying in executor
        self.max_calls = max_calls

        # Metrics
        self.call_count = 0
        self.error_4xx = 0
        self.error_5xx = 0
        self.latency_ms_total = 0.0
        self.calls_by_endpoint: dict[str, int] = defaultdict(int)
        # Full call trace for episode logging
        self.call_log: list[dict] = []

        self._session = requests.Session()
        self._session.auth = ("0", session_token)
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    # Convenience alias kept for backward compatibility
    @property
    def error_count(self) -> int:
        return self.error_4xx + self.error_5xx

    def metrics_summary(self) -> dict:
        return {
            "call_count": self.call_count,
            "error_4xx": self.error_4xx,
            "error_5xx": self.error_5xx,
            "latency_ms_total": round(self.latency_ms_total),
            "calls_by_endpoint": dict(self.calls_by_endpoint),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.call_count >= self.max_calls:
            raise BudgetExceededError(
                f"API call budget of {self.max_calls} exceeded at {method} {path}"
            )

        self.call_count += 1
        key = f"{method}:{path}"
        self.calls_by_endpoint[key] += 1
        url = self._url(path)

        t0 = time.monotonic()
        try:
            resp = self._session.request(method, url, timeout=30, **kwargs)
        except requests.RequestException as exc:
            self.error_5xx += 1
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.latency_ms_total += elapsed_ms
            self.call_log.append({"method": method, "path": path, "status": 0, "ms": round(elapsed_ms), "error": str(exc)})
            logger.error(f"{method} {path} – network error: {exc}")
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.latency_ms_total += elapsed_ms

        status = resp.status_code
        log_entry: dict = {"method": method, "path": path, "status": status, "ms": round(elapsed_ms)}
        if 400 <= status < 500:
            self.error_4xx += 1
            logger.error(f"{method} {path} -> {status}: {resp.text[:600]}")
            self.call_log.append(log_entry)
            resp.raise_for_status()
        elif status >= 500:
            self.error_5xx += 1
            logger.error(f"{method} {path} -> {status}: {resp.text[:300]}")
            self.call_log.append(log_entry)
            resp.raise_for_status()
        else:
            logger.info(f"{method} {path} -> {status} ({elapsed_ms:.0f}ms)")
            # Capture the created/updated resource id for shadow-checking
            if resp.content and method in ("POST", "PUT"):
                try:
                    body = resp.json()
                    created_id = (body.get("value") or {}).get("id")
                    if created_id:
                        log_entry["created_id"] = created_id
                except Exception:
                    pass
            self.call_log.append(log_entry)

        if resp.content:
            try:
                return resp.json()
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------ #
    # Public verbs
    # ------------------------------------------------------------------ #

    def get(self, path: str, params: dict | None = None) -> Any:
        return self._call("GET", path, params=params)

    def post(self, path: str, json: Any = None) -> Any:
        return self._call("POST", path, json=json)

    def put(self, path: str, json: Any = None, params: dict | None = None) -> Any:
        return self._call("PUT", path, json=json, params=params)

    def delete(self, path: str) -> Any:
        return self._call("DELETE", path)

    # ------------------------------------------------------------------ #
    # Convenience wrappers
    # ------------------------------------------------------------------ #

    def get_list(self, path: str, params: dict | None = None) -> list:
        """GET a collection; returns the unwrapped list."""
        data = self.get(path, params=params or {})
        if isinstance(data, dict) and "values" in data:
            return data["values"]
        return []

    def get_value(self, path: str, params: dict | None = None) -> Any:
        """GET a single resource; returns the unwrapped value."""
        data = self.get(path, params=params)
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def post_value(self, path: str, json: Any = None) -> Any:
        """POST and return the created entity (unwrapped)."""
        data = self.post(path, json=json)
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def put_value(self, path: str, json: Any = None) -> Any:
        """PUT and return the updated entity (unwrapped)."""
        data = self.put(path, json=json)
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data
