"""
LLM-as-critic: analyses a failed episode and produces a structured critique.

Philosophy:
  - The LLM is NOT the judge of whether the result was correct.
  - The LLM IS used to explain WHY a trajectory failed and WHAT to fix.
  - Critiques are cheap (one LLM call per failed episode, offline).
  - Output is structured JSON → feeds gold_builder.py and prompt patches.

Usage:
    python -m scripts.critic                          # critique all failures in logs/episodes_failed.jsonl
    python -m scripts.critic logs/episodes.jsonl      # critique failures from any file
    python -m scripts.critic --episode '{"request_id":...}'  # critique single episode JSON

Output:
    logs/critiques.jsonl  — one critique JSON per line
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Known failure catalog (imported from matchers to stay DRY) ────────────────
try:
    from agent.matchers import KNOWN_FAILURES
except ImportError:
    # Allow running as standalone script without the agent package on sys.path
    KNOWN_FAILURES = {}

# ────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """You are an expert AI-system debugger specialising in accounting API integrations.

You receive a failed or partially-failed episode from an AI agent that drives the Tripletex accounting API.
Your job: diagnose WHY it failed and suggest a CONCRETE fix.

Return ONLY a single JSON object with exactly these keys:
{
  "failure_type": "one of the known types or a new snake_case label",
  "root_cause": "one sentence: what went wrong technically",
  "suggested_fix": "one sentence: what code or policy change fixes it",
  "affected_layer": "one of: parser | matcher | executor | api | validator | unknown",
  "prompt_patch": "optional: a sentence to add/change in the parser system prompt (null if not applicable)",
  "workflow_patch": "optional: a sentence describing the executor change (null if not applicable)",
  "confidence": float 0.0–1.0
}

Known failure types (prefer these labels when they match):
""" + json.dumps(list(KNOWN_FAILURES.keys()), indent=2) + """

Rules:
- Be concise — one sentence per field.
- If the failure is clearly a sandbox limitation (bank account, missing config), say so.
- Never invent API errors; base diagnosis only on the episode data provided."""


def _build_critic_prompt(episode: dict) -> str:
    """Render an episode as a human-readable critic prompt."""
    parts = [
        f"request_id: {episode.get('request_id', '?')}",
        f"prompt: {episode.get('prompt', '')[:300]}",
        f"task_type (predicted): {episode.get('task_type', '?')}",
        f"language: {episode.get('language_detected', '?')}",
        f"confidence: {episode.get('confidence', '?')}",
        f"missing_fields: {episode.get('missing_fields', [])}",
        f"outcome: {episode.get('outcome', '?')}",
        f"error: {episode.get('error', None)}",
    ]

    parsed = {k: v for k, v in episode.get("parsed_entities", {}).items() if k != "_raw_prompt"}
    parts.append(f"parsed_entities: {json.dumps(parsed, ensure_ascii=False)}")

    calls = episode.get("api_calls", [])
    parts.append(f"api_trace ({len(calls)} calls):")
    for c in calls:
        parts.append(f"  {c.get('method')} {c.get('path')} → {c.get('status')} ({c.get('ms')}ms)")
        if c.get("error"):
            parts.append(f"    error: {c['error']}")

    m = episode.get("metrics", {})
    parts.append(f"4xx_count: {m.get('error_4xx', 0)}  5xx_count: {m.get('error_5xx', 0)}")

    shadow = episode.get("shadow_verdict")
    if shadow:
        parts.append(f"shadow_verdict: {json.dumps(shadow, ensure_ascii=False)}")

    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Core critique function
# ────────────────────────────────────────────────────────────────────────────

def critique_episode(episode: dict) -> dict[str, Any]:
    """
    Run LLM critic on a single episode dict.
    Returns a critique dict; never raises — returns error_critique on failure.
    """
    import json as _json

    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        logger.error("google-genai not installed; cannot run critic")
        return _error_critique(episode, "google-genai not available")

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        try:
            client = genai.Client(
                vertexai=True,
                project=os.environ["GCP_PROJECT_ID"],
                location=os.getenv("VERTEX_LOCATION", "europe-west1"),
            )
        except KeyError:
            return _error_critique(episode, "No LLM credentials found")

    prompt_text = _build_critic_prompt(episode)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[gtypes.Part.from_text(text=prompt_text)],
            config=gtypes.GenerateContentConfig(
                system_instruction=_CRITIC_SYSTEM,
                temperature=0,
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        critique = _json.loads(raw)
    except Exception as exc:
        return _error_critique(episode, str(exc))

    critique["request_id"] = episode.get("request_id")
    critique["task_type"] = episode.get("task_type")
    critique["timestamp"] = episode.get("timestamp")
    return critique


def _error_critique(episode: dict, reason: str) -> dict:
    return {
        "request_id": episode.get("request_id"),
        "task_type": episode.get("task_type"),
        "timestamp": episode.get("timestamp"),
        "failure_type": "critic_error",
        "root_cause": reason,
        "suggested_fix": "fix critic setup",
        "affected_layer": "unknown",
        "prompt_patch": None,
        "workflow_patch": None,
        "confidence": 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Is-failure predicate
# ────────────────────────────────────────────────────────────────────────────

def is_failure(episode: dict) -> bool:
    """Return True if an episode should be critiqued."""
    m = episode.get("metrics", {})
    return (
        episode.get("outcome") != "completed"
        or m.get("error_4xx", 0) > 0
        or m.get("error_5xx", 0) > 0
        or episode.get("error") is not None
        or (episode.get("shadow_verdict") or {}).get("passed") is False
    )


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _load_episodes(path: Path) -> list[dict]:
    episodes = []
    if not path.exists():
        return episodes
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return episodes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="LLM critic for failed episodes")
    parser.add_argument("file", nargs="?", default=None,
                        help="episodes.jsonl to read (default: logs/episodes.jsonl)")
    parser.add_argument("--episode", default=None,
                        help="Single episode JSON string to critique")
    parser.add_argument("--out", default="logs/critiques.jsonl",
                        help="Output file for critiques")
    parser.add_argument("--all", action="store_true",
                        help="Critique all episodes, not just failures")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.episode:
        episodes = [json.loads(args.episode)]
    else:
        src = Path(args.file or "logs/episodes.jsonl")
        episodes = _load_episodes(src)
        print(f"Loaded {len(episodes)} episodes from {src}")

    to_critique = episodes if args.all else [e for e in episodes if is_failure(e)]
    print(f"Critiquing {len(to_critique)} episode(s)…")

    critiques = []
    for ep in to_critique:
        print(f"  [{ep.get('request_id', '?')}] {ep.get('task_type', '?')} …", end=" ", flush=True)
        c = critique_episode(ep)
        critiques.append(c)
        print(c.get("failure_type", "?"))

    with out_path.open("w", encoding="utf-8") as fh:
        for c in critiques:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(critiques)} critique(s) → {out_path}")

    # Print summary
    from collections import Counter
    counts = Counter(c.get("failure_type") for c in critiques)
    print("\nFailure types:")
    for ft, n in counts.most_common():
        print(f"  {n:3d}  {ft}")


if __name__ == "__main__":
    main()
