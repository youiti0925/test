"""Standalone CLI for multi-run decision trace stats.

Usage:
  python -m src.fx.trace_stats_multi_cli runs/*/decision_traces.jsonl --pretty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .decision_trace_stats_multi import aggregate_many


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-stats-multi",
        description="Aggregate multiple decision_traces.jsonl/jsonl.gz files.",
    )
    parser.add_argument("paths", nargs="+", help="Trace JSONL/JSONL.GZ paths")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    try:
        payload = aggregate_many([Path(path) for path in args.paths])
    except ValueError as exc:
        print(f"trace-stats-multi: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(
        payload,
        indent=2 if args.pretty else None,
        default=str,
        ensure_ascii=False,
    ))
    if payload["consistency_checks"].get("errors_total", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
