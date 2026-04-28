"""Multi-run aggregation for decision trace stats.

This module intentionally keeps ``decision_trace_stats.aggregate_stats``
unchanged. ``aggregate_many`` calls it once per input file, then pools the
already-public aggregate sections into a per-run + global report.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .decision_trace_stats import aggregate_stats

_METRIC_KEYS: tuple[str, ...] = (
    "n_trades",
    "win_rate",
    "profit_factor",
    "total_return_pct",
    "max_drawdown_pct",
)


def _counter_add(target: Counter[str], source: dict[str, Any] | None) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if isinstance(key, str) and isinstance(value, int):
            target[key] += value


def _nested_counter_add(
    target: dict[str, Counter[str]],
    source: dict[str, dict[str, Any]] | None,
) -> None:
    if not isinstance(source, dict):
        return
    for outer_key, inner in source.items():
        if not isinstance(outer_key, str) or not isinstance(inner, dict):
            continue
        _counter_add(target[outer_key], inner)


def _read_metrics_summary(trace_path: Path) -> dict[str, Any] | None:
    """Read optional sibling summary.json and return a compact metrics slice."""
    summary_path = trace_path.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = payload
    compact = {key: metrics.get(key) for key in _METRIC_KEYS if key in metrics}
    return compact or None


def _per_run_view(stats: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    metadata = stats.get("metadata") or {}
    return {
        "input_path": str(trace_path),
        "run_id": metadata.get("run_id"),
        "symbol": metadata.get("symbol"),
        "timeframe": metadata.get("timeframe"),
        "n_traces": metadata.get("n_traces", 0),
        "metrics_summary": _read_metrics_summary(trace_path),
        "action_distribution": stats.get("action_distribution", {}),
        "gate_effect_distribution": stats.get("gate_effect_distribution", {}),
        "outcome_distribution": stats.get("outcome_distribution", {}),
        "cross_stats": stats.get("cross_stats", {}),
        "consistency_checks": stats.get("consistency_checks", {}),
    }


def _merge_hold_reason_rows(
    target: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]] | None,
) -> None:
    if not isinstance(source, dict):
        return
    for reason, row in source.items():
        if not isinstance(reason, str) or not isinstance(row, dict):
            continue
        dst = target.setdefault(
            reason,
            {
                "n": 0,
                "gate_effect": Counter(),
                "outcome_if_technical_action_taken": Counter(),
                "technical_only_action": Counter(),
                "return_n": 0,
                "return_sum": 0.0,
                "return_min": None,
                "return_max": None,
            },
        )
        n = row.get("n")
        if isinstance(n, int):
            dst["n"] += n
        _counter_add(dst["gate_effect"], row.get("gate_effect"))
        _counter_add(
            dst["outcome_if_technical_action_taken"],
            row.get("outcome_if_technical_action_taken"),
        )
        _counter_add(dst["technical_only_action"], row.get("technical_only_action"))

        rs = row.get("return_stats")
        if not isinstance(rs, dict):
            continue
        n_with_return = rs.get("n_with_return")
        sum_pct = rs.get("sum_pct")
        min_pct = rs.get("min_pct")
        max_pct = rs.get("max_pct")
        if isinstance(n_with_return, int) and n_with_return > 0 and isinstance(sum_pct, (int, float)):
            dst["return_n"] += n_with_return
            dst["return_sum"] += float(sum_pct)
        if isinstance(min_pct, (int, float)):
            value = float(min_pct)
            if dst["return_min"] is None or value < dst["return_min"]:
                dst["return_min"] = value
        if isinstance(max_pct, (int, float)):
            value = float(max_pct)
            if dst["return_max"] is None or value > dst["return_max"]:
                dst["return_max"] = value


def _finalise_hold_reason_rows(
    rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalised: dict[str, dict[str, Any]] = {}
    for reason, row in rows.items():
        n_with_return = row["return_n"]
        if n_with_return > 0:
            sum_pct = row["return_sum"]
            avg_pct = sum_pct / n_with_return
            min_pct = row["return_min"]
            max_pct = row["return_max"]
        else:
            sum_pct = avg_pct = min_pct = max_pct = None
        finalised[reason] = {
            "n": row["n"],
            "gate_effect": dict(row["gate_effect"]),
            "outcome_if_technical_action_taken": dict(
                row["outcome_if_technical_action_taken"]
            ),
            "technical_only_action": dict(row["technical_only_action"]),
            "return_stats": {
                "n_with_return": n_with_return,
                "avg_pct": avg_pct,
                "sum_pct": sum_pct,
                "min_pct": min_pct,
                "max_pct": max_pct,
            },
        }
    return finalised


def _pool_successful_stats(successes: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    final_action: Counter[str] = Counter()
    technical_action: Counter[str] = Counter()
    blocked_by: Counter[str] = Counter()
    rule_results: dict[str, Counter[str]] = defaultdict(Counter)
    gate_effect: Counter[str] = Counter()
    outcome: Counter[str] = Counter()
    entry_skipped_reason: Counter[str] = Counter()
    exit_reason: Counter[str] = Counter()
    top_hold_reasons: Counter[str] = Counter()
    hold_reason_rows: dict[str, dict[str, Any]] = {}
    gate_effect_by_technical_action: dict[str, Counter[str]] = defaultdict(Counter)
    final_action_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)

    entry_executed_true = 0
    entry_executed_false = 0
    exit_event_true = 0
    exit_event_false = 0

    for _path, stats in successes:
        actions = stats.get("action_distribution") or {}
        _counter_add(final_action, actions.get("final_action"))
        _counter_add(technical_action, actions.get("technical_only_action"))
        _counter_add(blocked_by, stats.get("blocked_by_distribution"))
        _nested_counter_add(rule_results, stats.get("rule_result_distribution"))
        _counter_add(gate_effect, stats.get("gate_effect_distribution"))
        _counter_add(outcome, stats.get("outcome_distribution"))

        entry = stats.get("entry_execution_distribution") or {}
        entry_executed_true += int(entry.get("entry_executed_true") or 0)
        entry_executed_false += int(entry.get("entry_executed_false") or 0)
        exit_event_true += int(entry.get("exit_event_true") or 0)
        exit_event_false += int(entry.get("exit_event_false") or 0)
        _counter_add(entry_skipped_reason, entry.get("entry_skipped_reason"))
        _counter_add(exit_reason, entry.get("exit_reason"))

        for item in stats.get("top_hold_reasons") or []:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            count = item.get("count")
            if isinstance(reason, str) and isinstance(count, int):
                top_hold_reasons[reason] += count

        cross = stats.get("cross_stats") or {}
        _merge_hold_reason_rows(hold_reason_rows, cross.get("hold_reason_outcome"))
        _nested_counter_add(
            gate_effect_by_technical_action,
            cross.get("gate_effect_by_technical_action"),
        )
        _nested_counter_add(
            final_action_by_outcome,
            cross.get("final_action_by_outcome"),
        )

    return {
        "action_distribution": {
            "final_action": dict(final_action),
            "technical_only_action": dict(technical_action),
        },
        "blocked_by_distribution": dict(blocked_by),
        "rule_result_distribution": {
            key: dict(counts) for key, counts in rule_results.items()
        },
        "gate_effect_distribution": dict(gate_effect),
        "outcome_distribution": dict(outcome),
        "entry_execution_distribution": {
            "entry_executed_true": entry_executed_true,
            "entry_executed_false": entry_executed_false,
            "entry_skipped_reason": dict(entry_skipped_reason),
            "exit_event_true": exit_event_true,
            "exit_event_false": exit_event_false,
            "exit_reason": dict(exit_reason),
        },
        "top_hold_reasons": [
            {"reason": reason, "count": count}
            for reason, count in top_hold_reasons.most_common(10)
        ],
        "cross_stats": {
            "hold_reason_outcome": _finalise_hold_reason_rows(hold_reason_rows),
            "gate_effect_by_technical_action": {
                key: dict(counts)
                for key, counts in gate_effect_by_technical_action.items()
            },
            "final_action_by_outcome": {
                key: dict(counts) for key, counts in final_action_by_outcome.items()
            },
        },
    }


def aggregate_many(
    paths: Iterable[Path | str],
    *,
    top_n_hold_reasons: int = 10,
) -> dict[str, Any]:
    """Aggregate multiple decision trace JSONL/GZ files.

    Invalid runs are collected in ``consistency_checks.failed_runs`` while any
    valid run continues to be included. If no valid runs remain, ``ValueError``
    is raised.
    """
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("aggregate_many: paths is empty")

    successes: list[tuple[Path, dict[str, Any]]] = []
    failed_runs: list[dict[str, str]] = []

    # Use a very high per-run limit so global top_hold_reasons can pool all
    # reasons, not just each run's public top N.
    internal_top_n = max(top_n_hold_reasons, 1_000_000)
    for path in path_list:
        try:
            stats = aggregate_stats(path, top_n_hold_reasons=internal_top_n)
        except (OSError, ValueError) as exc:
            failed_runs.append({"input_path": str(path), "error": str(exc)})
            continue
        successes.append((path, stats))

    if not successes:
        raise ValueError("aggregate_many: all runs failed")

    per_run = [_per_run_view(stats, path) for path, stats in successes]
    global_stats = _pool_successful_stats(successes)
    global_stats["top_hold_reasons"] = global_stats["top_hold_reasons"][:top_n_hold_reasons]

    metadata_rows = [stats.get("metadata") or {} for _path, stats in successes]
    run_ids = [m.get("run_id") for m in metadata_rows if isinstance(m.get("run_id"), str)]
    symbols = sorted({m.get("symbol") for m in metadata_rows if isinstance(m.get("symbol"), str)})
    timeframes = sorted({m.get("timeframe") for m in metadata_rows if isinstance(m.get("timeframe"), str)})
    schema_versions = sorted({
        m.get("trace_schema_version")
        for m in metadata_rows
        if isinstance(m.get("trace_schema_version"), str)
    })
    first_timestamps = [m.get("first_timestamp") for m in metadata_rows if isinstance(m.get("first_timestamp"), str)]
    last_timestamps = [m.get("last_timestamp") for m in metadata_rows if isinstance(m.get("last_timestamp"), str)]
    n_traces_total = sum(int(m.get("n_traces") or 0) for m in metadata_rows)

    warnings: list[str] = []
    duplicate_run_ids = sorted(
        run_id for run_id, count in Counter(run_ids).items() if count > 1
    )
    if duplicate_run_ids:
        warnings.append(f"duplicate run_id values: {duplicate_run_ids}")

    per_run_errors_total = sum(
        int((stats.get("consistency_checks") or {}).get("errors_total") or 0)
        for _path, stats in successes
    )
    per_run_warnings_total = sum(
        int((stats.get("consistency_checks") or {}).get("warnings_total") or 0)
        for _path, stats in successes
    )
    errors_total = per_run_errors_total + len(failed_runs)
    warnings_total = per_run_warnings_total + len(warnings)

    return {
        "metadata": {
            "n_runs": len(path_list),
            "n_runs_succeeded": len(successes),
            "n_runs_failed": len(failed_runs),
            "n_traces_total": n_traces_total,
            "symbols": symbols,
            "timeframes": timeframes,
            "run_ids": sorted(run_ids),
            "first_timestamp": min(first_timestamps) if first_timestamps else None,
            "last_timestamp": max(last_timestamps) if last_timestamps else None,
        },
        "per_run": per_run,
        "global": global_stats,
        "consistency_checks": {
            "errors_total": errors_total,
            "warnings_total": warnings_total,
            "run_id_unique_count": len(set(run_ids)),
            "schema_version_consistent": len(schema_versions) <= 1,
            "failed_runs": failed_runs,
            "duplicate_run_ids": duplicate_run_ids,
            "warnings": warnings,
        },
    }


__all__ = ["aggregate_many"]
