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
        "consistency_checks": consistency_checks,
    }


__all__ = [
    "aggregate_stats",
    "MAX_ERROR_LIST_SIZE",
    "MAX_WARNING_LIST_SIZE",
]
