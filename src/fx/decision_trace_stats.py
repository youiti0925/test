"""Streaming aggregator for `decision_traces.jsonl` files.

Reads one record per line and accumulates counters into a single JSON-
serialisable summary. Designed to handle large run files without
materialising every trace in memory: each line is parsed, used to bump a
small set of counters, then dropped.

Public entry point: `aggregate_stats(path)`.

Failure semantics
-----------------
- Empty file or no parseable lines -> `ValueError`.
- Individual malformed lines -> recorded in
  `consistency_checks.errors` (capped to MAX_ERROR_LIST_SIZE entries) and
  `errors_total` is incremented; processing continues.
- Missing `rule_checks` / `future_outcome` -> recorded as warnings
  (capped to MAX_WARNING_LIST_SIZE entries).
- Out-of-order timestamps -> warning (not an error).
"""
from __future__ import annotations

import gzip as _gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, IO

from .decision_trace import RULE_TAXONOMY


# Sentinel used as the inner-dict factory for hold-reason aggregates so a
# fresh row gets every sub-counter pre-allocated.
def _new_hold_reason_row() -> dict[str, Any]:
    return {
        "n": 0,
        "gate_effect": Counter(),
        "outcome_if_technical_action_taken": Counter(),
        "technical_only_action": Counter(),
        # Running stats for hypothetical_technical_trade_return_pct so we
        # don't keep every value in memory. Finalised in
        # _finalise_cross_stats().
        "_return_count": 0,
        "_return_sum": 0.0,
        "_return_min": None,
        "_return_max": None,
    }


def _finalise_hold_reason_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert running counters/floats to the public output shape."""
    n_with_return = row["_return_count"]
    if n_with_return > 0:
        avg = row["_return_sum"] / n_with_return
        sum_pct = row["_return_sum"]
        min_pct = row["_return_min"]
        max_pct = row["_return_max"]
    else:
        avg = sum_pct = min_pct = max_pct = None
    return {
        "n": row["n"],
        "gate_effect": dict(row["gate_effect"]),
        "outcome_if_technical_action_taken": dict(
            row["outcome_if_technical_action_taken"]
        ),
        "technical_only_action": dict(row["technical_only_action"]),
        "return_stats": {
            "n_with_return": n_with_return,
            "avg_pct": avg,
            "sum_pct": sum_pct,
            "min_pct": min_pct,
            "max_pct": max_pct,
        },
    }


# ---------------------------------------------------------------------------
# Hard caps so a giant run with pathological errors does not balloon RAM.
# `*_total` keeps the precise count; the list keeps a representative sample.
# ---------------------------------------------------------------------------
MAX_ERROR_LIST_SIZE = 50
MAX_WARNING_LIST_SIZE = 50

# Set of rule_check.result values surfaced by the engine. Pre-initialised so
# every cell is visible even when a result class never fires.
_RULE_RESULT_BUCKETS: tuple[str, ...] = (
    "PASS", "BLOCK", "WARN", "SKIPPED", "NOT_REACHED", "INFO",
)


def _open_jsonl(path: Path) -> IO[str]:
    """Transparent open for `.jsonl` and `.jsonl.gz` (UTF-8 text mode)."""
    if path.suffix == ".gz":
        return _gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _empty_rule_result_distribution() -> dict[str, dict[str, int]]:
    """Pre-seed all 19 canonical rule_ids with zero counters.

    Visibility-by-default: a rule that never appeared still shows up with
    `{PASS: 0, BLOCK: 0, ...}`, which is informative for trace integrity
    audits (a missing rule_id == 0-everywhere is worth flagging).
    """
    return {
        rid: {bucket: 0 for bucket in _RULE_RESULT_BUCKETS}
        for rid in RULE_TAXONOMY
    }


def _bump_capped(target_list: list[str], message: str, cap: int) -> None:
    """Append `message` to `target_list` only while under the cap."""
    if len(target_list) < cap:
        target_list.append(message)


def aggregate_stats(
    jsonl_path: Path | str,
    *,
    top_n_hold_reasons: int = 10,
) -> dict[str, Any]:
    """Stream-read `jsonl_path` and return aggregate stats.

    Parameters
    ----------
    jsonl_path:
        Path to `decision_traces.jsonl` or `decision_traces.jsonl.gz`.
    top_n_hold_reasons:
        How many `(reason, count)` pairs to surface in `top_hold_reasons`.

    Returns
    -------
    A nested dict matching the structure documented in the module docstring.

    Raises
    ------
    FileNotFoundError
        if `jsonl_path` does not exist.
    ValueError
        if the file is empty or no line yielded a parseable record.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"aggregate_stats: file not found: {path}")

    # Distributions
    final_action_counter: Counter[str] = Counter()
    technical_action_counter: Counter[str] = Counter()
    blocked_by_counter: Counter[str] = Counter()
    rule_result_distribution = _empty_rule_result_distribution()
    gate_effect_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()

    entry_executed_counter: Counter[bool] = Counter()
    entry_skipped_counter: Counter[str] = Counter()
    exit_event_counter: Counter[bool] = Counter()
    exit_reason_counter: Counter[str] = Counter()

    hold_reason_counter: Counter[str] = Counter()

    # ── Cross-stats accumulators (PR #7) ─────────────────────────────────
    # `hold_reason_outcome` — keyed by decision.reason for HOLD bars only.
    # Each row carries Counters + running return stats so we never keep
    # the per-bar values in memory.
    hold_reason_aggregates: dict[str, dict[str, Any]] = {}
    # `gate_effect_by_technical_action` and `final_action_by_outcome`
    # apply to all bars regardless of final_action.
    gate_effect_by_tech: dict[str, Counter[str]] = defaultdict(Counter)
    final_action_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)

    # Metadata accumulators
    schema_versions: set[str] = set()
    run_ids: set[str] = set()
    symbols: set[str] = set()
    timeframes: set[str] = set()
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    prev_timestamp: str | None = None

    # Consistency
    n_traces = 0
    errors: list[str] = []
    warnings: list[str] = []
    errors_total = 0
    warnings_total = 0
    traces_missing_rule_checks = 0
    traces_missing_future_outcome = 0

    with _open_jsonl(path) as fp:
        for line_no, raw in enumerate(fp, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors_total += 1
                _bump_capped(
                    errors,
                    f"line {line_no}: malformed JSON ({exc.msg})",
                    MAX_ERROR_LIST_SIZE,
                )
                continue
            if not isinstance(rec, dict):
                errors_total += 1
                _bump_capped(
                    errors,
                    f"line {line_no}: top-level value is not an object",
                    MAX_ERROR_LIST_SIZE,
                )
                continue

            n_traces += 1

            # ── Metadata ─────────────────────────────────────────────
            if isinstance(rec.get("trace_schema_version"), str):
                schema_versions.add(rec["trace_schema_version"])
            if isinstance(rec.get("run_id"), str):
                run_ids.add(rec["run_id"])
            if isinstance(rec.get("symbol"), str):
                symbols.add(rec["symbol"])
            if isinstance(rec.get("timeframe"), str):
                timeframes.add(rec["timeframe"])

            ts = rec.get("timestamp")
            if isinstance(ts, str):
                if first_timestamp is None:
                    first_timestamp = ts
                last_timestamp = ts
                if prev_timestamp is not None and ts < prev_timestamp:
                    warnings_total += 1
                    _bump_capped(
                        warnings,
                        f"line {line_no}: timestamp not monotonic "
                        f"({ts} after {prev_timestamp})",
                        MAX_WARNING_LIST_SIZE,
                    )
                prev_timestamp = ts

            # ── Decision distributions ───────────────────────────────
            decision = rec.get("decision") or {}
            final_action = decision.get("final_action")
            tech_action = decision.get("technical_only_action")
            if isinstance(final_action, str):
                final_action_counter[final_action] += 1
            if isinstance(tech_action, str):
                technical_action_counter[tech_action] += 1

            blocked_by = decision.get("blocked_by") or []
            if not blocked_by:
                blocked_by_counter["no_block"] += 1
            else:
                for code in blocked_by:
                    if isinstance(code, str):
                        blocked_by_counter[code] += 1

            if final_action == "HOLD":
                reason = decision.get("reason")
                if isinstance(reason, str):
                    hold_reason_counter[reason] += 1

            # ── Rule check distribution ──────────────────────────────
            rule_checks = rec.get("rule_checks")
            if not isinstance(rule_checks, list):
                traces_missing_rule_checks += 1
                warnings_total += 1
                _bump_capped(
                    warnings,
                    f"line {line_no}: missing rule_checks",
                    MAX_WARNING_LIST_SIZE,
                )
            else:
                for rc in rule_checks:
                    if not isinstance(rc, dict):
                        continue
                    rid = rc.get("canonical_rule_id")
                    result = rc.get("result")
                    if not isinstance(rid, str) or not isinstance(result, str):
                        continue
                    bucket = rule_result_distribution.setdefault(
                        rid, {b: 0 for b in _RULE_RESULT_BUCKETS}
                    )
                    bucket[result] = bucket.get(result, 0) + 1

            # ── Future outcome distributions ─────────────────────────
            fo = rec.get("future_outcome")
            if fo is None:
                traces_missing_future_outcome += 1
                warnings_total += 1
                _bump_capped(
                    warnings,
                    f"line {line_no}: missing future_outcome",
                    MAX_WARNING_LIST_SIZE,
                )
            elif isinstance(fo, dict):
                ge = fo.get("gate_effect")
                if isinstance(ge, str):
                    gate_effect_counter[ge] += 1
                outcome = fo.get("outcome_if_technical_action_taken")
                if isinstance(outcome, str):
                    outcome_counter[outcome] += 1

            # ── Cross-stats updates (PR #7) ──────────────────────────
            # Read once per bar; the same fields are reused below. fo may
            # be None or non-dict, in which case ge/outcome stay None so
            # the cross counters bucket those bars under "MISSING_*".
            fo_dict = fo if isinstance(fo, dict) else {}
            ge_value = fo_dict.get("gate_effect") if isinstance(
                fo_dict.get("gate_effect"), str
            ) else None
            outcome_value = fo_dict.get(
                "outcome_if_technical_action_taken"
            ) if isinstance(
                fo_dict.get("outcome_if_technical_action_taken"), str
            ) else None
            return_value = fo_dict.get(
                "hypothetical_technical_trade_return_pct"
            )

            # gate_effect × technical_only_action — all bars
            if isinstance(tech_action, str) and ge_value is not None:
                gate_effect_by_tech[tech_action][ge_value] += 1

            # final_action × outcome_if_technical_action_taken — all bars
            if isinstance(final_action, str) and outcome_value is not None:
                final_action_by_outcome[final_action][outcome_value] += 1

            # hold_reason_outcome — only HOLD bars
            if final_action == "HOLD":
                reason_str = decision.get("reason")
                if isinstance(reason_str, str):
                    row = hold_reason_aggregates.setdefault(
                        reason_str, _new_hold_reason_row()
                    )
                    row["n"] += 1
                    if ge_value is not None:
                        row["gate_effect"][ge_value] += 1
                    if outcome_value is not None:
                        row["outcome_if_technical_action_taken"][
                            outcome_value
                        ] += 1
                    if isinstance(tech_action, str):
                        row["technical_only_action"][tech_action] += 1
                    if isinstance(return_value, (int, float)):
                        rv = float(return_value)
                        row["_return_count"] += 1
                        row["_return_sum"] += rv
                        if row["_return_min"] is None or rv < row["_return_min"]:
                            row["_return_min"] = rv
                        if row["_return_max"] is None or rv > row["_return_max"]:
                            row["_return_max"] = rv

            # ── Execution trace distributions ────────────────────────
            et = rec.get("execution_trace") or {}
            if isinstance(et.get("entry_executed"), bool):
                entry_executed_counter[et["entry_executed"]] += 1
            esr = et.get("entry_skipped_reason")
            if isinstance(esr, str):
                entry_skipped_counter[esr] += 1
            if isinstance(et.get("exit_event"), bool):
                exit_event_counter[et["exit_event"]] += 1
            ex_reason = et.get("exit_reason")
            if isinstance(ex_reason, str):
                exit_reason_counter[ex_reason] += 1

    if n_traces == 0:
        if errors_total > 0:
            raise ValueError(
                f"aggregate_stats: no parseable traces found in {path} "
                f"(errors_total={errors_total})"
            )
        raise ValueError(f"aggregate_stats: file is empty: {path}")

    # ── Build output ────────────────────────────────────────────────
    schema_versions_list = sorted(schema_versions)
    run_ids_list = sorted(run_ids)

    metadata = {
        "input_path": str(path),
        "n_traces": n_traces,
        "trace_schema_version": (
            schema_versions_list[0] if len(schema_versions_list) == 1 else None
        ),
        "run_id": run_ids_list[0] if len(run_ids_list) == 1 else None,
        "symbol": (
            sorted(symbols)[0] if len(symbols) == 1 else None
        ),
        "timeframe": (
            sorted(timeframes)[0] if len(timeframes) == 1 else None
        ),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }

    consistency_checks = {
        "n_traces": n_traces,
        "n_traces_positive": n_traces > 0,
        "run_id_consistent": len(run_ids_list) <= 1,
        "run_id_unique_count": len(run_ids_list),
        "trace_schema_version_consistent": len(schema_versions_list) <= 1,
        "schema_version_unique_count": len(schema_versions_list),
        "traces_missing_rule_checks": traces_missing_rule_checks,
        "traces_missing_future_outcome": traces_missing_future_outcome,
        "errors_total": errors_total,
        "warnings_total": warnings_total,
        "errors": errors,
        "warnings": warnings,
        "errors_truncated": errors_total > len(errors),
        "warnings_truncated": warnings_total > len(warnings),
    }
    if not consistency_checks["run_id_consistent"]:
        msg = (
            f"run_id mismatch: {len(run_ids_list)} distinct run_ids "
            f"({run_ids_list[:5]}{'...' if len(run_ids_list) > 5 else ''})"
        )
        _bump_capped(consistency_checks["errors"], msg, MAX_ERROR_LIST_SIZE)
        consistency_checks["errors_total"] += 1
    if not consistency_checks["trace_schema_version_consistent"]:
        msg = (
            f"trace_schema_version mismatch: "
            f"{schema_versions_list}"
        )
        _bump_capped(consistency_checks["errors"], msg, MAX_ERROR_LIST_SIZE)
        consistency_checks["errors_total"] += 1

    return {
        "metadata": metadata,
        "action_distribution": {
            "final_action": dict(final_action_counter),
            "technical_only_action": dict(technical_action_counter),
        },
        "blocked_by_distribution": dict(blocked_by_counter),
        "rule_result_distribution": rule_result_distribution,
        "gate_effect_distribution": dict(gate_effect_counter),
        "outcome_distribution": dict(outcome_counter),
        "entry_execution_distribution": {
            "entry_executed_true": entry_executed_counter.get(True, 0),
            "entry_executed_false": entry_executed_counter.get(False, 0),
            "entry_skipped_reason": dict(entry_skipped_counter),
            "exit_event_true": exit_event_counter.get(True, 0),
            "exit_event_false": exit_event_counter.get(False, 0),
            "exit_reason": dict(exit_reason_counter),
        },
        "top_hold_reasons": [
            {"reason": r, "count": c}
            for r, c in hold_reason_counter.most_common(top_n_hold_reasons)
        ],
        "cross_stats": {
            "hold_reason_outcome": {
                reason: _finalise_hold_reason_row(row)
                for reason, row in hold_reason_aggregates.items()
            },
            "gate_effect_by_technical_action": {
                tech: dict(counts)
                for tech, counts in gate_effect_by_tech.items()
            },
            "final_action_by_outcome": {
                action: dict(counts)
                for action, counts in final_action_by_outcome.items()
            },
        },
        "consistency_checks": consistency_checks,
    }


# ---------------------------------------------------------------------------
# Multi-run aggregation (PR #8)
# ---------------------------------------------------------------------------


# Subset of summary.json's `metrics` dict that the multi report surfaces in
# each per-run entry. Other metrics fields are still on disk for callers that
# read summary.json directly.
_METRICS_SUMMARY_KEYS: tuple[str, ...] = (
    "n_trades",
    "win_rate",
    "profit_factor",
    "total_return_pct",
    "max_drawdown_pct",
)


def _read_metrics_summary(jsonl_path: Path) -> dict | None:
    """Look for a sibling summary.json and pull a few key metrics.

    Returns None when the sibling file is missing or unreadable — the
    multi report is not allowed to fail just because one run was created
    by hand without summary.json.
    """
    summary_path = jsonl_path.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        s = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    metrics = s.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {k: metrics.get(k) for k in _METRICS_SUMMARY_KEYS}


def _add_counters(target: dict[str, int], source: dict[str, int]) -> None:
    """In-place: `target[k] += source[k]` over every key in source."""
    for k, v in source.items():
        if isinstance(v, (int, float)):
            target[k] = target.get(k, 0) + int(v)


def _pool_hold_reason_row(
    pooled: dict[str, Any], per_run: dict[str, Any]
) -> None:
    """Fold one per-run hold_reason_outcome row into the pooled row.

    Counters are summed; return_stats are pooled by summing
    n_with_return + sum_pct, taking cross-run min/max, and recomputing
    avg = sum/n at the end (avg recompute happens in
    `_finalise_pooled_hold_reason_row`).
    """
    pooled["n"] += int(per_run.get("n", 0))
    _add_counters(pooled["gate_effect"], per_run.get("gate_effect", {}))
    _add_counters(
        pooled["outcome_if_technical_action_taken"],
        per_run.get("outcome_if_technical_action_taken", {}),
    )
    _add_counters(
        pooled["technical_only_action"],
        per_run.get("technical_only_action", {}),
    )
    rs = per_run.get("return_stats") or {}
    n_run = int(rs.get("n_with_return") or 0)
    if n_run > 0:
        pooled["_return_count"] += n_run
        sum_pct = rs.get("sum_pct")
        if isinstance(sum_pct, (int, float)):
            pooled["_return_sum"] += float(sum_pct)
        run_min = rs.get("min_pct")
        run_max = rs.get("max_pct")
        if isinstance(run_min, (int, float)):
            if pooled["_return_min"] is None or run_min < pooled["_return_min"]:
                pooled["_return_min"] = float(run_min)
        if isinstance(run_max, (int, float)):
            if pooled["_return_max"] is None or run_max > pooled["_return_max"]:
                pooled["_return_max"] = float(run_max)


def _new_pooled_hold_reason_row() -> dict[str, Any]:
    """Pooled accumulator for one reason across all runs."""
    return {
        "n": 0,
        "gate_effect": {},
        "outcome_if_technical_action_taken": {},
        "technical_only_action": {},
        "_return_count": 0,
        "_return_sum": 0.0,
        "_return_min": None,
        "_return_max": None,
    }


def _finalise_pooled_hold_reason_row(row: dict[str, Any]) -> dict[str, Any]:
    n_with_return = row["_return_count"]
    if n_with_return > 0:
        avg = row["_return_sum"] / n_with_return
        sum_pct = row["_return_sum"]
        min_pct = row["_return_min"]
        max_pct = row["_return_max"]
    else:
        avg = sum_pct = min_pct = max_pct = None
    return {
        "n": row["n"],
        "gate_effect": dict(row["gate_effect"]),
        "outcome_if_technical_action_taken": dict(
            row["outcome_if_technical_action_taken"]
        ),
        "technical_only_action": dict(row["technical_only_action"]),
        "return_stats": {
            "n_with_return": n_with_return,
            "avg_pct": avg,
            "sum_pct": sum_pct,
            "min_pct": min_pct,
            "max_pct": max_pct,
        },
    }


def _compact_per_run_view(stats: dict[str, Any], jsonl_path: Path) -> dict[str, Any]:
    """Build the compact per_run entry from a full per-run aggregate_stats output."""
    md = stats["metadata"]
    return {
        "input_path": str(jsonl_path),
        "run_id": md["run_id"],
        "symbol": md["symbol"],
        "timeframe": md["timeframe"],
        "n_traces": md["n_traces"],
        "first_timestamp": md["first_timestamp"],
        "last_timestamp": md["last_timestamp"],
        "metrics_summary": _read_metrics_summary(jsonl_path),
        "action_distribution": stats["action_distribution"],
        "gate_effect_distribution": stats["gate_effect_distribution"],
        "outcome_distribution": stats["outcome_distribution"],
        "cross_stats": stats["cross_stats"],
        "consistency_checks": stats["consistency_checks"],
    }


def aggregate_many(
    paths: list[Path | str] | tuple[Path | str, ...],
    *,
    top_n_hold_reasons: int = 10,
) -> dict[str, Any]:
    """Aggregate stats across multiple decision_traces.jsonl files.

    Each path is processed independently via `aggregate_stats()`. Failures
    on individual paths are recorded in `consistency_checks.failed_runs`
    and do not abort the report — provided at least one path succeeds.

    Returns a 4-section dict: `metadata`, `per_run`, `global`,
    `consistency_checks`. `per_run` is a compact view (cross_stats +
    light distributions); `global` carries the full pooled distributions.

    Raises
    ------
    ValueError
        if `paths` is empty, or every path failed (no successful runs).
    """
    if not paths:
        raise ValueError("aggregate_many: no input paths provided")

    per_run: list[dict[str, Any]] = []
    failed_runs: list[dict[str, str]] = []
    multi_errors: list[str] = []
    multi_warnings: list[str] = []

    # Global counters
    g_final_action: dict[str, int] = {}
    g_tech_action: dict[str, int] = {}
    g_blocked_by: dict[str, int] = {}
    g_rule_results: dict[str, dict[str, int]] = _empty_rule_result_distribution()
    g_gate_effect: dict[str, int] = {}
    g_outcome: dict[str, int] = {}
    g_entry_executed_true = 0
    g_entry_executed_false = 0
    g_entry_skipped: dict[str, int] = {}
    g_exit_event_true = 0
    g_exit_event_false = 0
    g_exit_reason: dict[str, int] = {}
    g_top_hold: dict[str, int] = {}      # reason -> count, then most_common(N)
    g_hold_reason_pool: dict[str, dict[str, Any]] = {}
    g_gate_by_tech: dict[str, dict[str, int]] = {}
    g_final_by_outcome: dict[str, dict[str, int]] = {}

    # Global metadata accumulators
    g_run_ids: list[str] = []                    # preserve order, dedup at end
    g_schema_versions: set[str] = set()
    g_symbols: set[str] = set()
    g_timeframes: set[str] = set()
    g_first_ts: str | None = None
    g_last_ts: str | None = None
    g_n_traces_total = 0
    g_errors_total_runs = 0
    g_warnings_total_runs = 0

    # Process each path
    for path in paths:
        p = Path(path)
        try:
            stats = aggregate_stats(p, top_n_hold_reasons=top_n_hold_reasons)
        except (FileNotFoundError, ValueError) as exc:
            failed_runs.append({"path": str(p), "error": str(exc)})
            _bump_capped(
                multi_errors,
                f"failed to aggregate {p}: {exc}",
                MAX_ERROR_LIST_SIZE,
            )
            continue

        per_run.append(_compact_per_run_view(stats, p))

        # Fold into global ─────────────────────────────────────────────
        md = stats["metadata"]
        if isinstance(md.get("run_id"), str):
            g_run_ids.append(md["run_id"])
        if isinstance(md.get("trace_schema_version"), str):
            g_schema_versions.add(md["trace_schema_version"])
        if isinstance(md.get("symbol"), str):
            g_symbols.add(md["symbol"])
        if isinstance(md.get("timeframe"), str):
            g_timeframes.add(md["timeframe"])
        first_ts = md.get("first_timestamp")
        last_ts = md.get("last_timestamp")
        if isinstance(first_ts, str):
            if g_first_ts is None or first_ts < g_first_ts:
                g_first_ts = first_ts
        if isinstance(last_ts, str):
            if g_last_ts is None or last_ts > g_last_ts:
                g_last_ts = last_ts
        g_n_traces_total += int(md.get("n_traces") or 0)

        cc = stats["consistency_checks"]
        g_errors_total_runs += int(cc.get("errors_total") or 0)
        g_warnings_total_runs += int(cc.get("warnings_total") or 0)

        # Distribution counters
        ad = stats["action_distribution"]
        _add_counters(g_final_action, ad.get("final_action", {}))
        _add_counters(g_tech_action, ad.get("technical_only_action", {}))
        _add_counters(g_blocked_by, stats["blocked_by_distribution"])
        _add_counters(g_gate_effect, stats["gate_effect_distribution"])
        _add_counters(g_outcome, stats["outcome_distribution"])

        # rule_result_distribution (per-cell add)
        for rid, buckets in stats["rule_result_distribution"].items():
            cell = g_rule_results.setdefault(
                rid, {b: 0 for b in _RULE_RESULT_BUCKETS}
            )
            for bucket, count in buckets.items():
                cell[bucket] = cell.get(bucket, 0) + int(count)

        # entry_execution_distribution
        eed = stats["entry_execution_distribution"]
        g_entry_executed_true += int(eed.get("entry_executed_true") or 0)
        g_entry_executed_false += int(eed.get("entry_executed_false") or 0)
        _add_counters(g_entry_skipped, eed.get("entry_skipped_reason", {}))
        g_exit_event_true += int(eed.get("exit_event_true") or 0)
        g_exit_event_false += int(eed.get("exit_event_false") or 0)
        _add_counters(g_exit_reason, eed.get("exit_reason", {}))

        # top_hold_reasons — turn the per-run list back into a counter
        # so we can re-take most_common(N) globally.
        for entry in stats["top_hold_reasons"]:
            r = entry.get("reason")
            c = entry.get("count")
            if isinstance(r, str) and isinstance(c, int):
                g_top_hold[r] = g_top_hold.get(r, 0) + c

        # cross_stats
        cs = stats["cross_stats"]
        for reason, row in cs["hold_reason_outcome"].items():
            pooled = g_hold_reason_pool.setdefault(
                reason, _new_pooled_hold_reason_row()
            )
            _pool_hold_reason_row(pooled, row)

        for tech, counts in cs["gate_effect_by_technical_action"].items():
            target = g_gate_by_tech.setdefault(tech, {})
            _add_counters(target, counts)

        for fa, counts in cs["final_action_by_outcome"].items():
            target = g_final_by_outcome.setdefault(fa, {})
            _add_counters(target, counts)

    # All runs failed -> ValueError
    if not per_run:
        raise ValueError(
            f"aggregate_many: no successful runs (failed={len(failed_runs)})"
        )

    # Detect duplicate run_ids — warning only, not an error.
    seen: dict[str, int] = {}
    for rid in g_run_ids:
        seen[rid] = seen.get(rid, 0) + 1
    duplicates = {rid: c for rid, c in seen.items() if c > 1}
    for rid, c in duplicates.items():
        _bump_capped(
            multi_warnings,
            f"duplicate run_id encountered {c} times: {rid!r}",
            MAX_WARNING_LIST_SIZE,
        )

    schema_versions_list = sorted(g_schema_versions)
    run_ids_unique = sorted(set(g_run_ids))

    metadata = {
        "n_runs": len(paths),
        "n_runs_succeeded": len(per_run),
        "n_runs_failed": len(failed_runs),
        "n_traces_total": g_n_traces_total,
        "symbols": sorted(g_symbols),
        "timeframes": sorted(g_timeframes),
        "run_ids": run_ids_unique,
        "first_timestamp": g_first_ts,
        "last_timestamp": g_last_ts,
    }

    # Top hold reasons globally
    top_pairs = sorted(
        g_top_hold.items(), key=lambda kv: (-kv[1], kv[0])
    )[:top_n_hold_reasons]

    global_section = {
        "action_distribution": {
            "final_action": dict(g_final_action),
            "technical_only_action": dict(g_tech_action),
        },
        "blocked_by_distribution": dict(g_blocked_by),
        "rule_result_distribution": g_rule_results,
        "gate_effect_distribution": dict(g_gate_effect),
        "outcome_distribution": dict(g_outcome),
        "entry_execution_distribution": {
            "entry_executed_true": g_entry_executed_true,
            "entry_executed_false": g_entry_executed_false,
            "entry_skipped_reason": dict(g_entry_skipped),
            "exit_event_true": g_exit_event_true,
            "exit_event_false": g_exit_event_false,
            "exit_reason": dict(g_exit_reason),
        },
        "top_hold_reasons": [
            {"reason": r, "count": c} for r, c in top_pairs
        ],
        "cross_stats": {
            "hold_reason_outcome": {
                reason: _finalise_pooled_hold_reason_row(row)
                for reason, row in g_hold_reason_pool.items()
            },
            "gate_effect_by_technical_action": {
                tech: dict(counts)
                for tech, counts in g_gate_by_tech.items()
            },
            "final_action_by_outcome": {
                action: dict(counts)
                for action, counts in g_final_by_outcome.items()
            },
        },
    }

    # Multi-level errors_total = per-run total + multi-level errors (e.g.
    # failed_runs entries each generate one multi error line).
    multi_errors_total = g_errors_total_runs + len(failed_runs)
    multi_warnings_total = g_warnings_total_runs + len(duplicates)

    consistency_checks = {
        "n_runs": len(paths),
        "n_runs_succeeded": len(per_run),
        "n_runs_failed": len(failed_runs),
        "errors_total": multi_errors_total,
        "warnings_total": multi_warnings_total,
        "run_id_unique_count": len(run_ids_unique),
        "schema_version_consistent": len(schema_versions_list) <= 1,
        "schema_versions": schema_versions_list,
        "errors": multi_errors,
        "warnings": multi_warnings,
        "errors_truncated": multi_errors_total > len(multi_errors),
        "warnings_truncated": multi_warnings_total > len(multi_warnings),
        "failed_runs": failed_runs,
    }

    return {
        "metadata": metadata,
        "per_run": per_run,
        "global": global_section,
        "consistency_checks": consistency_checks,
    }


__all__ = [
    "aggregate_stats",
    "aggregate_many",
    "MAX_ERROR_LIST_SIZE",
    "MAX_WARNING_LIST_SIZE",
]
