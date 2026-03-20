"""
Gold dataset builder.

Reads episodes from logs/episodes.jsonl (or Cloud Logging) and writes
a curated gold dataset used as few-shot examples in the parser.

A "gold" episode must satisfy ALL:
  - outcome == "completed"
  - error_4xx == 0
  - error_5xx == 0
  - confidence >= 0.85
  - task_type != "unknown"

Outputs:
  gold/prompts.jsonl    — (task_type, language, prompt, entities) — parser few-shots
  gold/workflows.jsonl  — (task_type, api_trace, latency_ms)      — workflow analysis

Usage:
    python scripts/gold_builder.py                 # build from logs/episodes.jsonl
    python scripts/gold_builder.py --fetch         # fetch fresh from Cloud Logging first
    python scripts/gold_builder.py --stats         # print stats only, no write
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


LOGS_DIR = Path("logs")
GOLD_DIR = Path("gold")
DEFAULT_INPUT = LOGS_DIR / "episodes.jsonl"
GOLD_PROMPTS = GOLD_DIR / "prompts.jsonl"
GOLD_WORKFLOWS = GOLD_DIR / "workflows.jsonl"

# Max examples per (task_type, language) pair in the prompts file.
# Keeps the few-shot bank small and representative.
MAX_PER_BUCKET = 3


# ─────────────────────────────────────────────────────────────────────────────
# Predicates
# ─────────────────────────────────────────────────────────────────────────────

def _is_gold(ep: dict) -> bool:
    m = ep.get("metrics", {})
    return (
        ep.get("outcome") == "completed"
        and m.get("error_4xx", 0) == 0
        and m.get("error_5xx", 0) == 0
        and ep.get("confidence", 0.0) >= 0.85
        and ep.get("task_type", "unknown") not in ("unknown", "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    episodes = []
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


def _fetch_from_cloud_logging(
    project: str = "ainm26osl-714",
    service: str = "ledger-logic-agent",
    limit: int = 500,
) -> list[dict]:
    """Pull episodes from Cloud Logging via gcloud CLI."""
    print(f"Fetching up to {limit} episodes from Cloud Logging…")
    filter_str = (
        f'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{service}" '
        f'textPayload:"EPISODE"'
    )
    try:
        out = subprocess.check_output(
            [
                "gcloud", "logging", "read", filter_str,
                f"--project={project}",
                f"--limit={limit}",
                "--format=value(textPayload)",
                "--order=asc",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"gcloud error: {exc}", file=sys.stderr)
        return []

    episodes = []
    for line in out.splitlines():
        prefix = "EPISODE "
        if "EPISODE" in line:
            idx = line.find(prefix)
            if idx != -1:
                payload = line[idx + len(prefix):]
                try:
                    episodes.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    print(f"Fetched {len(episodes)} episodes.")
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_example(ep: dict) -> dict:
    """Distil an episode into a parser few-shot example."""
    entities = {
        k: v for k, v in ep.get("parsed_entities", {}).items()
        if k not in {"_raw_prompt"}
    }
    return {
        "task_type": ep["task_type"],
        "language": ep.get("language_detected", "?"),
        "prompt": ep.get("prompt", "")[:400],
        "entities": entities,
        "confidence": ep.get("confidence", 0.0),
        "request_id": ep.get("request_id"),
        "git_sha": ep.get("git_sha"),
    }


def _build_workflow_example(ep: dict) -> dict:
    """Distil an episode into a workflow trace for analysis."""
    return {
        "task_type": ep["task_type"],
        "language": ep.get("language_detected", "?"),
        "api_trace": ep.get("api_calls", []),
        "latency_api_ms": (ep.get("metrics") or {}).get("latency_api_ms", 0),
        "latency_total_ms": (ep.get("metrics") or {}).get("latency_total_ms", 0),
        "missing_fields": ep.get("missing_fields", []),
        "request_id": ep.get("request_id"),
        "git_sha": ep.get("git_sha"),
    }


def build_gold(episodes: list[dict], dry_run: bool = False) -> dict:
    """
    Filter episodes → gold. Cap per bucket. Write outputs.
    Returns stats dict.
    """
    gold = [ep for ep in episodes if _is_gold(ep)]

    # Deduplicate by request_id
    seen: set[str] = set()
    unique_gold = []
    for ep in gold:
        rid = ep.get("request_id", "")
        if rid and rid not in seen:
            seen.add(rid)
            unique_gold.append(ep)
    gold = unique_gold

    # Cap per (task_type, language) bucket
    bucket_count: dict[tuple, int] = defaultdict(int)
    capped: list[dict] = []
    for ep in gold:
        key = (ep["task_type"], ep.get("language_detected", "?"))
        if bucket_count[key] < MAX_PER_BUCKET:
            capped.append(ep)
            bucket_count[key] += 1

    prompt_examples = [_build_prompt_example(ep) for ep in capped]
    workflow_examples = [_build_workflow_example(ep) for ep in capped]

    stats = {
        "total_input": len(episodes),
        "gold_candidates": len(gold),
        "after_cap": len(capped),
        "by_task_type": defaultdict(int),
    }
    for ep in capped:
        stats["by_task_type"][ep["task_type"]] += 1

    if not dry_run:
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        with GOLD_PROMPTS.open("w", encoding="utf-8") as fh:
            for ex in prompt_examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with GOLD_WORKFLOWS.open("w", encoding="utf-8") as fh:
            for ex in workflow_examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold few-shot dataset from episodes")
    parser.add_argument("file", nargs="?", default=str(DEFAULT_INPUT),
                        help=f"Input episodes.jsonl (default: {DEFAULT_INPUT})")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch fresh episodes from Cloud Logging before building")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats only, do not write gold files")
    parser.add_argument("--project", default="ainm26osl-714")
    parser.add_argument("--service", default="ledger-logic-agent")
    args = parser.parse_args()

    episodes = _load_jsonl(Path(args.file))
    print(f"Loaded {len(episodes)} episodes from {args.file}")

    if args.fetch:
        fresh = _fetch_from_cloud_logging(args.project, args.service)
        # Merge: existing first, then fresh (by request_id dedup in build_gold)
        episodes = episodes + fresh

    stats = build_gold(episodes, dry_run=args.stats)

    print(f"\n{'DRY-RUN ' if args.stats else ''}Gold dataset summary:")
    print(f"  Total input    : {stats['total_input']}")
    print(f"  Gold candidates: {stats['gold_candidates']}")
    print(f"  After cap      : {stats['after_cap']}  (max {MAX_PER_BUCKET}/bucket)")
    print(f"\n  By task type:")
    for tt, n in sorted(stats["by_task_type"].items()):
        print(f"    {n:3d}  {tt}")

    if not args.stats:
        print(f"\nWrote:")
        print(f"  {GOLD_PROMPTS}   ({stats['after_cap']} examples)")
        print(f"  {GOLD_WORKFLOWS}  ({stats['after_cap']} examples)")


if __name__ == "__main__":
    main()
