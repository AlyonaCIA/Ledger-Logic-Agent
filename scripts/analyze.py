#!/usr/bin/env python3
"""
Analyze episodes.jsonl and print actionable improvement insights.

Usage:
    # Fetch + accumulate (default — never overwrites):
    python scripts/analyze.py --fetch

    # Fetch only episodes from last 30 minutes:
    python scripts/analyze.py --fetch --freshness 30m

    # Analyze only the 20 most recent stored episodes:
    python scripts/analyze.py --recent 20

    # Analyze a specific file:
    python scripts/analyze.py some_other.jsonl

    # Overwrite (fresh start):
    python scripts/analyze.py --fetch --overwrite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Freshness parser ───────────────────────────────────────────────────────

def _parse_freshness(s: str) -> timedelta:
    """Parse '30m', '2h', '1d' → timedelta."""
    s = s.strip().lower()
    if s.endswith("m"):
        return timedelta(minutes=int(s[:-1]))
    if s.endswith("h"):
        return timedelta(hours=int(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=int(s[:-1]))
    raise ValueError(f"Cannot parse freshness: {s!r}. Use e.g. '30m', '2h', '1d'.")


# ── Fetch from Cloud Logging ───────────────────────────────────────────────

def fetch_episodes(
    project: str,
    limit: int,
    service: str,
    freshness: timedelta | None = None,
) -> list[dict]:
    filter_parts = [
        f'resource.type="cloud_run_revision"',
        f'resource.labels.service_name="{service}"',
        f'textPayload:"EPISODE"',
    ]
    if freshness is not None:
        cutoff = (datetime.now(timezone.utc) - freshness).strftime("%Y-%m-%dT%H:%M:%SZ")
        filter_parts.append(f'timestamp>"{cutoff}"')

    filter_str = " AND ".join(filter_parts)
    freshness_label = f" (last {str(freshness)})" if freshness else ""
    print(f"Fetching up to {limit} episodes from Cloud Logging ({project}/{service}){freshness_label}…")

    result = subprocess.run(
        ["gcloud", "logging", "read", filter_str,
         "--project", project, "--limit", str(limit), "--format", "json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR fetching logs: {result.stderr}")
        sys.exit(1)

    episodes = []
    for log_entry in json.loads(result.stdout or "[]"):
        # Capture revision from resource labels
        revision = (
            log_entry.get("resource", {}).get("labels", {}).get("revision_name", "")
        )
        msg = log_entry.get("textPayload", "")
        idx = msg.find("{")
        if idx >= 0:
            try:
                ep = json.loads(msg[idx:])
                if revision and not ep.get("revision"):
                    ep["revision"] = revision
                episodes.append(ep)
            except json.JSONDecodeError:
                pass

    print(f"Fetched {len(episodes)} episodes.")
    return episodes


# ── JSONL helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    episodes = []
    if not path.exists():
        return episodes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return episodes


def save_jsonl(path: Path, episodes: list[dict]) -> None:
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def merge_episodes(existing: list[dict], fresh: list[dict]) -> tuple[list[dict], int]:
    """
    Merge fresh into existing, deduplicating by request_id.
    Returns (merged_sorted_by_timestamp, count_added).
    """
    seen: dict[str, dict] = {ep["request_id"]: ep for ep in existing if ep.get("request_id")}
    added = 0
    for ep in fresh:
        rid = ep.get("request_id")
        if not rid:
            continue
        if rid not in seen:
            seen[rid] = ep
            added += 1
        else:
            # Keep the one with the newer timestamp
            existing_ts = seen[rid].get("timestamp", "")
            new_ts = ep.get("timestamp", "")
            if new_ts > existing_ts:
                seen[rid] = ep
    merged = sorted(seen.values(), key=lambda e: e.get("timestamp", ""))
    return merged, added


# ── Analysis helpers ───────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100*n//total}%" if total else "n/a"


def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def analyze(episodes: list[dict]) -> None:
    if not episodes:
        print("No episodes to analyze.")
        return

    total = len(episodes)

    # ── 1. Overall ─────────────────────────────────────────────────────
    section(f"OVERVIEW  ({total} episodes)")
    outcomes = defaultdict(int)
    for ep in episodes:
        outcomes[ep.get("outcome", "?")] += 1
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome:<15} {count:>4}  ({_pct(count, total)})")

    # timestamp range
    timestamps = [ep["timestamp"] for ep in episodes if ep.get("timestamp")]
    if timestamps:
        print(f"  oldest: {min(timestamps)[:19]}")
        print(f"  newest: {max(timestamps)[:19]}")

    # revisions seen
    revisions = {ep.get("revision", "") for ep in episodes if ep.get("revision")}
    if revisions:
        print(f"  revisions: {', '.join(sorted(revisions))}")

    # ── 2. Success rate & call stats by task_type ──────────────────────
    section("SUCCESS RATE + CALLS BY TASK TYPE")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_type", "unknown")].append(ep)

    header = f"  {'TASK TYPE':<30} {'N':>4} {'OK%':>5} {'AVG calls':>10} {'AVG 4xx':>8} {'AVG ms':>8} {'AVG conf':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for task, eps in sorted(by_task.items()):
        n = len(eps)
        ok = sum(1 for e in eps if e.get("outcome") == "completed" and e.get("metrics", {}).get("error_4xx", 0) == 0)
        avg_calls = sum(e.get("metrics", {}).get("calls", 0) for e in eps) / n
        avg_4xx   = sum(e.get("metrics", {}).get("error_4xx", 0) for e in eps) / n
        avg_ms    = sum(e.get("metrics", {}).get("latency_total_ms", 0) for e in eps) / n
        avg_conf  = sum(e.get("confidence", 0) for e in eps) / n
        flag = "  ◀ LOW CONF" if avg_conf < 0.8 else ""
        print(f"  {task:<30} {n:>4} {_pct(ok,n):>5} {avg_calls:>10.1f} {avg_4xx:>8.2f} {avg_ms:>8.0f} {avg_conf:>9.2f}{flag}")

    # ── 3. Confidence distribution ─────────────────────────────────────
    section("CONFIDENCE DISTRIBUTION")
    buckets = {"1.00": 0, "0.90-0.99": 0, "0.70-0.89": 0, "<0.70": 0}
    for ep in episodes:
        c = ep.get("confidence", 0)
        if c == 1.0:
            buckets["1.00"] += 1
        elif c >= 0.90:
            buckets["0.90-0.99"] += 1
        elif c >= 0.70:
            buckets["0.70-0.89"] += 1
        else:
            buckets["<0.70"] += 1
    for bucket, count in buckets.items():
        bar = "█" * (count * 30 // max(total, 1))
        print(f"  {bucket:<12} {count:>4}  {bar}")

    # ── 4. Top 4xx errors by endpoint ─────────────────────────────────
    section("TOP 4xx ERRORS BY ENDPOINT")
    err_counts: dict[str, int] = defaultdict(int)
    for ep in episodes:
        for call in ep.get("api_calls", []):
            if 400 <= call.get("status", 0) < 500:
                key = f"{call['method']} {call['path']}"
                err_counts[key] += 1
    if err_counts:
        for endpoint, count in sorted(err_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:>4}x  {endpoint}")
    else:
        print("  ✓ No 4xx errors recorded.")

    # ── 5. Failed episodes detail ──────────────────────────────────────
    section("FAILED EPISODES (outcome ≠ completed OR 4xx > 0)")
    failed = [
        ep for ep in episodes
        if ep.get("outcome") != "completed" or ep.get("metrics", {}).get("error_4xx", 0) > 0
    ]
    if not failed:
        print("  ✓ No failures.")
    else:
        print(f"  {len(failed)} failed episode(s):\n")
        for ep in failed[:20]:
            rev_label = f" rev={ep['revision'][-8:]}" if ep.get("revision") else ""
            ts_label  = f" @{ep['timestamp'][:16]}" if ep.get("timestamp") else ""
            print(f"  [{ep.get('request_id','?')}]{rev_label}{ts_label} "
                  f"task={ep.get('task_type','?')} "
                  f"lang={ep.get('language_detected','?')} "
                  f"conf={ep.get('confidence',0):.2f} "
                  f"4xx={ep.get('metrics',{}).get('error_4xx',0)} "
                  f"outcome={ep.get('outcome','?')}")
            if ep.get("error"):
                print(f"    error: {ep['error'][:120]}")
            for call in ep.get("api_calls", []):
                if call.get("status", 0) >= 400:
                    print(f"    ✗ {call['method']} {call['path']} → {call['status']}")
            print()

    # ── 6. Expensive workflows (calls > median) ────────────────────────
    section("MOST EXPENSIVE WORKFLOWS (by API calls)")
    all_calls = sorted(ep.get("metrics", {}).get("calls", 0) for ep in episodes)
    median_calls = all_calls[len(all_calls) // 2] if all_calls else 0
    expensive = [ep for ep in episodes if ep.get("metrics", {}).get("calls", 0) > median_calls]
    if expensive:
        by_count: dict[str, dict] = defaultdict(lambda: {"total_calls": 0, "n": 0})
        for ep in expensive:
            t = ep.get("task_type", "unknown")
            by_count[t]["total_calls"] += ep.get("metrics", {}).get("calls", 0)
            by_count[t]["n"] += 1
        for task, stats in sorted(by_count.items(), key=lambda x: -x[1]["total_calls"]):
            avg = stats["total_calls"] / stats["n"]
            print(f"  {task:<30}  avg {avg:.1f} calls  (n={stats['n']})")
    else:
        print("  All workflows at or below median call count.")

    # ── 7. Unknown task detections ─────────────────────────────────────
    section("UNKNOWN / LOW-CONFIDENCE DETECTIONS")
    unknowns = [ep for ep in episodes if ep.get("task_type") == "unknown" or ep.get("confidence", 1) < 0.7]
    if not unknowns:
        print("  ✓ No unknown or low-confidence detections.")
    else:
        print(f"  {len(unknowns)} episode(s):\n")
        for ep in unknowns[:10]:
            print(f"  [{ep.get('request_id','?')}] conf={ep.get('confidence',0):.2f} "
                  f"detected={ep.get('task_type','?')}")
            print(f"    prompt: {ep.get('prompt','')[:100]}")
            print()

    print(f"\n{'═'*60}")
    print(f"  Done. Analyzed {total} episodes.")
    print(f"{'═'*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Ledger Logic Agent episodes")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to episodes.jsonl (default: logs/episodes.jsonl)")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch episodes from Cloud Logging and accumulate")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite logs/episodes.jsonl instead of accumulating")
    parser.add_argument("--freshness", default=None,
                        help="Only fetch logs from last N (e.g. 30m, 2h, 1d)")
    parser.add_argument("--recent", type=int, default=None,
                        help="Analyze only the N most recent stored episodes")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: logs/episodes.jsonl)")
    parser.add_argument("--project", default="ainm26osl-714")
    parser.add_argument("--service", default="ledger-logic-agent")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    out_path = Path(args.out or args.file or "logs/episodes.jsonl")

    freshness: timedelta | None = None
    if args.freshness:
        freshness = _parse_freshness(args.freshness)

    if args.fetch:
        fresh = fetch_episodes(args.project, args.limit, args.service, freshness)

        if args.overwrite:
            episodes = fresh
            save_jsonl(out_path, episodes)
            print(f"Overwrote {out_path} with {len(episodes)} episodes.\n")
        else:
            existing = load_jsonl(out_path)
            episodes, added = merge_episodes(existing, fresh)
            save_jsonl(out_path, episodes)
            print(f"  existing loaded : {len(existing)}")
            print(f"  fetched         : {len(fresh)}")
            print(f"  new unique added: {added}")
            print(f"  total stored    : {len(episodes)}")
            if episodes:
                ts_list = [e["timestamp"] for e in episodes if e.get("timestamp")]
                if ts_list:
                    print(f"  oldest          : {min(ts_list)[:19]}")
                    print(f"  newest          : {max(ts_list)[:19]}")
            print()

    elif out_path.exists():
        episodes = load_jsonl(out_path)
        print(f"Loaded {len(episodes)} episodes from {out_path}\n")
    else:
        if args.file:
            print(f"File not found: {out_path}")
        else:
            print("No episodes file found. Run with --fetch to pull from Cloud Logging.")
        sys.exit(1)

    if args.recent and args.recent < len(episodes):
        print(f"[--recent {args.recent}] analyzing most recent {args.recent} of {len(episodes)} episodes\n")
        episodes = episodes[-args.recent:]

    analyze(episodes)


if __name__ == "__main__":
    main()

