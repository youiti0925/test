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


# ---------------------------------------------------------------------------
# Long-term / macro-regime bucketers (PR-A)
#
# Each bucket function maps a numeric value (or None for unavailable) to a
# stable string label so the cross-stat accumulators key by category, not by
# raw float. Thresholds are conservative defensive defaults — they exist to
# *partition* observations for descriptive aggregation, NOT to define
# decision rules. No code outside this module's stats consumes these
# labels.
# ---------------------------------------------------------------------------


def _bucket_close_vs_sma(pct: float | None) -> str:
    if pct is None:
        return "unknown"
    if pct < -5.0:
        return "<-5%"
    if pct < -1.0:
        return "-5..-1%"
    if pct <= 1.0:
        return "-1..1%"
    if pct <= 5.0:
        return "1..5%"
    return ">5%"


def _bucket_pct_5d(pct: float | None) -> str:
    """Bucket a 5-day percent change for DXY-style series."""
    if pct is None:
        return "unknown"
    if pct < -1.0:
        return "down>1%"
    if pct < -0.2:
        return "down"
    if pct <= 0.2:
        return "flat"
    if pct <= 1.0:
        return "up"
    return "up>1%"


def _bucket_yield_5d_bp(bp: float | None) -> str:
    """Bucket a 5-day yield change reported in basis points (1 bp = 0.01 pp)."""
    if bp is None:
        return "unknown"
    if bp < -10.0:
        return "down>10bp"
    if bp < -2.0:
        return "down"
    if bp <= 2.0:
        return "flat"
    if bp <= 10.0:
        return "up"
    return "up>10bp"


def _bucket_yield_spread_level(level: float | None) -> str:
    if level is None:
        return "unknown"
    if level < -0.50:
        return "deep_inverted"
    if level < 0.0:
        return "inverted"
    if level < 0.50:
        return "near_zero"
    if level <= 1.50:
        return "normal"
    return "steep"


def usd_exposure_for(symbol: str | None, final_action: str | None) -> str:
    """Map (symbol, BUY/SELL/HOLD) to a USD-direction-normalised label.

    A BUY of `USDJPY=X` is long USD (you're buying USD against JPY); a
    BUY of `EURUSD=X` is short USD (you're buying EUR by selling USD).
    Without this normalisation, a `dxy × side` cross conflates the two
    pair conventions and the resulting "DXY up + SELL = bad" reading is
    incorrect — for USD-quote pairs the exposures cancel.

    Closed taxonomy:
      LONG_USD             — bar net-long USD (USDJPY BUY, EURUSD SELL, ...)
      SHORT_USD            — bar net-short USD
      HOLD                 — no entry intended
      UNKNOWN_OR_NON_USD   — symbol not in the USD-pair table or input is None
    """
    if final_action == "HOLD":
        return "HOLD"
    if not isinstance(symbol, str) or not isinstance(final_action, str):
        return "UNKNOWN_OR_NON_USD"
    if final_action not in ("BUY", "SELL"):
        return "UNKNOWN_OR_NON_USD"
    # USD-base (USD/XXX): BUY = long USD, SELL = short USD.
    if symbol in ("USDJPY=X", "USDCHF=X", "USDCAD=X"):
        return "LONG_USD" if final_action == "BUY" else "SHORT_USD"
    # USD-quote (XXX/USD): BUY = short USD, SELL = long USD.
    if symbol in ("EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X"):
        return "SHORT_USD" if final_action == "BUY" else "LONG_USD"
    return "UNKNOWN_OR_NON_USD"


def _bucket_waveform_confidence(c: float | None) -> str:
    """Bucket WaveformBias.confidence into low / mid / high / unknown."""
    if c is None:
        return "unknown"
    if c < 0.3:
        return "low<0.3"
    if c < 0.6:
        return "mid_0.3..0.6"
    return "high>=0.6"


def _bucket_top_similarity(s: float | None) -> str:
    """Bucket the top match similarity score into weak / medium / strong."""
    if s is None:
        return "unknown"
    if s < 0.6:
        return "weak<0.6"
    if s < 0.75:
        return "medium_0.6..0.75"
    return "strong>=0.75"


def _bucket_matched_count(n: int | None) -> str:
    """Bucket the number of similar past windows used by the bias."""
    if n is None:
        return "unknown"
    if n == 0:
        return "0"
    if n < 10:
        return "1-9"
    if n < 30:
        return "10-29"
    return "30+"


def _bucket_vix_level(level: float | None) -> str:
    if level is None:
        return "unknown"
    if level < 14.0:
        return "low"
    if level < 20.0:
        return "mid"
    if level < 30.0:
        return "high"
    return "extreme"


def _new_two_way_row() -> dict[str, Any]:
    """Counter row keyed by final_action, tracking outcome breakdown.

    Shape: {final_action: {"n": int, "outcome": Counter}}
    Used by the long_term / macro-regime cross_stats added in PR-A.
    """
    return {}


def _bump_two_way(
    target: dict[str, dict[str, Any]],
    final_action: str | None,
    outcome: str | None,
) -> None:
    if not isinstance(final_action, str):
        return
    leaf = target.setdefault(
        final_action, {"n": 0, "outcome": Counter()}
    )
    leaf["n"] += 1
    if isinstance(outcome, str):
        leaf["outcome"][outcome] += 1


def _finalise_two_way(target: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        action: {"n": leaf["n"], "outcome": dict(leaf["outcome"])}
        for action, leaf in target.items()
    }


def _finalise_pooled_two_way(
    pool: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Multi-run version: bucket → final_action → {n, outcome}."""
    return {
        bucket: {
            action: {"n": leaf["n"], "outcome": dict(leaf["outcome"])}
            for action, leaf in by_action.items()
        }
        for bucket, by_action in pool.items()
    }


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
    # ── Cross-stats accumulator (PR #10) ─────────────────────────────────
    # `blocked_by_outcome` — keyed by each decision.blocked_by code (the
    # closed Risk-Gate taxonomy: event_high / spread_abnormal / etc.) plus
    # the synthetic key "no_block" for traces with empty blocked_by. The
    # row schema mirrors hold_reason_outcome so the same finalisers and
    # pooling helpers can be reused. A single trace with multiple
    # blocked_by codes contributes once to EACH key (additive semantics).
    blocked_by_outcome_aggregates: dict[str, dict[str, Any]] = {}

    # ── Cross-stats accumulators (PR-A) ──────────────────────────────────
    # Long-term-trend / macro-regime crosses, all keyed by a string bucket
    # → final_action → {n, outcome counter}. Bars without the source slice
    # (older traces, or runs without macro fetch) are bucketed under
    # "unknown" rather than dropped — visibility-by-default.
    daily_trend_outcome: dict[str, dict[str, Any]] = {}
    weekly_trend_outcome: dict[str, dict[str, Any]] = {}
    monthly_trend_outcome: dict[str, dict[str, Any]] = {}
    close_vs_sma_200d_outcome: dict[str, dict[str, Any]] = {}
    weekly_return_outcome: dict[str, dict[str, Any]] = {}
    monthly_return_outcome: dict[str, dict[str, Any]] = {}
    dxy_trend_outcome: dict[str, dict[str, Any]] = {}
    us10y_trend_outcome: dict[str, dict[str, Any]] = {}
    yield_spread_outcome: dict[str, dict[str, Any]] = {}
    vix_regime_outcome: dict[str, dict[str, Any]] = {}
    # Joint cross: monthly_trend × dxy_5d_trend
    long_term_macro_outcome: dict[str, dict[str, Any]] = {}
    # PR-A v2: DXY × USD-direction-normalised exposure. Avoids conflating
    # USDJPY BUY (long USD) with EURUSD BUY (short USD).
    dxy_trend_by_usd_exposure_outcome: dict[str, dict[str, Any]] = {}
    # Per-symbol DXY × raw final_action breakdown — for symbol-aware reads.
    symbol_dxy_trend_outcome: dict[str, dict[str, Any]] = {}

    # PR #15: waveform-match cross-stats. All keyed by a deterministic
    # bucket label derived from trace.waveform.waveform_bias; bars
    # without a bias dict (None or pre-PR-#15 traces) bucket as
    # "unknown" for visibility-by-default.
    waveform_bias_direction_outcome: dict[str, dict[str, Any]] = {}
    waveform_confidence_bucket_outcome: dict[str, dict[str, Any]] = {}
    top_similarity_bucket_outcome: dict[str, dict[str, Any]] = {}
    matched_count_bucket_outcome: dict[str, dict[str, Any]] = {}
    waveform_bias_by_technical_action_outcome: dict[str, dict[str, Any]] = {}
    symbol_waveform_bias_outcome: dict[str, dict[str, Any]] = {}

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

            # blocked_by_outcome — every bar contributes (PR #10).
            # blocked_by is a list of Risk-Gate codes. Empty list →
            # synthetic key "no_block". Multiple codes → contribute once
            # to EACH key (a bar that triggers both event_high and
            # spread_abnormal counts in both buckets).
            if not blocked_by:
                blocked_keys: list[str] = ["no_block"]
            else:
                blocked_keys = [c for c in blocked_by if isinstance(c, str)]
                if not blocked_keys:
                    # blocked_by present but contained only non-strings
                    blocked_keys = ["no_block"]
            for code in blocked_keys:
                row = blocked_by_outcome_aggregates.setdefault(
                    code, _new_hold_reason_row()
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

            # ── Long-term / macro cross-stats (PR-A) ──────────────────
            # Read the optional slices once. Older traces may lack the
            # keys entirely; that's fine — every bucketer maps None to
            # "unknown" and the counts surface that as a regular bucket.
            ltt = rec.get("long_term_trend") if isinstance(
                rec.get("long_term_trend"), dict
            ) else {}
            mc = rec.get("macro_context") if isinstance(
                rec.get("macro_context"), dict
            ) else {}

            daily_trend_label = ltt.get("daily_trend") or "unknown"
            weekly_trend_label = ltt.get("weekly_trend") or "unknown"
            monthly_trend_label = ltt.get("monthly_trend") or "unknown"
            sma200_bucket = _bucket_close_vs_sma(
                ltt.get("close_vs_sma_200d_pct")
            )
            weekly_ret_bucket = _bucket_pct_5d(ltt.get("weekly_return_pct"))
            monthly_ret_bucket = _bucket_pct_5d(ltt.get("monthly_return_pct"))

            dxy_bucket = _bucket_pct_5d(mc.get("dxy_change_5d_pct"))
            us10y_bucket = _bucket_yield_5d_bp(mc.get("us10y_change_5d_bp"))
            spread_bucket = _bucket_yield_spread_level(
                mc.get("yield_spread_long_short")
            )
            vix_bucket = _bucket_vix_level(mc.get("vix"))
            symbol_for_norm = rec.get("symbol")
            usd_exposure = usd_exposure_for(symbol_for_norm, final_action)

            _bump_two_way(
                daily_trend_outcome.setdefault(
                    daily_trend_label, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                weekly_trend_outcome.setdefault(
                    weekly_trend_label, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                monthly_trend_outcome.setdefault(
                    monthly_trend_label, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                close_vs_sma_200d_outcome.setdefault(
                    sma200_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                weekly_return_outcome.setdefault(
                    weekly_ret_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                monthly_return_outcome.setdefault(
                    monthly_ret_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                dxy_trend_outcome.setdefault(
                    dxy_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                us10y_trend_outcome.setdefault(
                    us10y_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                yield_spread_outcome.setdefault(
                    spread_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                vix_regime_outcome.setdefault(
                    vix_bucket, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            joint_key = f"{monthly_trend_label}|dxy_5d={dxy_bucket}"
            _bump_two_way(
                long_term_macro_outcome.setdefault(
                    joint_key, _new_two_way_row()
                ),
                final_action, outcome_value,
            )

            # PR-A v2: DXY × USD-direction-normalised exposure.
            # The leaf is keyed by the normalised label (LONG_USD /
            # SHORT_USD / HOLD / UNKNOWN_OR_NON_USD) so DXY-up rows can
            # be read coherently across USD-base and USD-quote pairs.
            _bump_two_way(
                dxy_trend_by_usd_exposure_outcome.setdefault(
                    dxy_bucket, _new_two_way_row()
                ),
                usd_exposure, outcome_value,
            )

            # PR-A v2: per-symbol DXY × raw final_action. Keyed by
            # f"{symbol}|dxy_5d={bucket}" so each symbol's DXY-regime
            # behaviour is independently inspectable.
            sym_label = symbol_for_norm if isinstance(symbol_for_norm, str) else "?"
            sym_dxy_key = f"{sym_label}|dxy_5d={dxy_bucket}"
            _bump_two_way(
                symbol_dxy_trend_outcome.setdefault(
                    sym_dxy_key, _new_two_way_row()
                ),
                final_action, outcome_value,
            )

            # ── PR #15: waveform-match cross-stats ───────────────────
            # waveform_bias is the dict produced by waveform_bias_dict()
            # in decision_trace_build; it can also be None (pre-PR-#15
            # traces or library not attached). Every bucket falls back
            # to "unknown" for visibility-by-default.
            wf_section = (rec.get("waveform") or {}).get("waveform_bias")
            wf_dict = wf_section if isinstance(wf_section, dict) else {}
            wf_action = wf_dict.get("action") or wf_dict.get("expected_direction")
            if not isinstance(wf_action, str):
                wf_action = "unknown"
            wf_conf = wf_dict.get("confidence")
            if not isinstance(wf_conf, (int, float)):
                wf_conf = None
            wf_top = wf_dict.get("top_similarity")
            if not isinstance(wf_top, (int, float)):
                wf_top = None
            wf_matched = wf_dict.get("matched_count")
            if wf_matched is None:
                wf_matched = wf_dict.get("sample_count")
            if not isinstance(wf_matched, int):
                wf_matched = None

            _bump_two_way(
                waveform_bias_direction_outcome.setdefault(
                    wf_action, _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                waveform_confidence_bucket_outcome.setdefault(
                    _bucket_waveform_confidence(wf_conf), _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                top_similarity_bucket_outcome.setdefault(
                    _bucket_top_similarity(wf_top), _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            _bump_two_way(
                matched_count_bucket_outcome.setdefault(
                    _bucket_matched_count(wf_matched), _new_two_way_row()
                ),
                final_action, outcome_value,
            )
            # bias direction × technical_only_action — separate cross to
            # see whether bias agreed with the technical signal.
            _bump_two_way(
                waveform_bias_by_technical_action_outcome.setdefault(
                    wf_action, _new_two_way_row()
                ),
                tech_action if isinstance(tech_action, str) else None,
                outcome_value,
            )
            sym_wf_key = f"{sym_label}|wf_bias={wf_action}"
            _bump_two_way(
                symbol_waveform_bias_outcome.setdefault(
                    sym_wf_key, _new_two_way_row()
                ),
                final_action, outcome_value,
            )

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
            "blocked_by_outcome": {
                code: _finalise_hold_reason_row(row)
                for code, row in blocked_by_outcome_aggregates.items()
            },
            # PR-A — long-term / macro regime crosses. Keys are stable
            # bucket labels; "unknown" means the slice was absent or
            # could not be computed at that bar.
            "daily_trend_outcome": {
                k: _finalise_two_way(v) for k, v in daily_trend_outcome.items()
            },
            "weekly_trend_outcome": {
                k: _finalise_two_way(v) for k, v in weekly_trend_outcome.items()
            },
            "monthly_trend_outcome": {
                k: _finalise_two_way(v) for k, v in monthly_trend_outcome.items()
            },
            "close_vs_sma_200d_outcome": {
                k: _finalise_two_way(v)
                for k, v in close_vs_sma_200d_outcome.items()
            },
            "weekly_return_outcome": {
                k: _finalise_two_way(v)
                for k, v in weekly_return_outcome.items()
            },
            "monthly_return_outcome": {
                k: _finalise_two_way(v)
                for k, v in monthly_return_outcome.items()
            },
            "dxy_trend_outcome": {
                k: _finalise_two_way(v) for k, v in dxy_trend_outcome.items()
            },
            "us10y_trend_outcome": {
                k: _finalise_two_way(v)
                for k, v in us10y_trend_outcome.items()
            },
            "yield_spread_outcome": {
                k: _finalise_two_way(v)
                for k, v in yield_spread_outcome.items()
            },
            "vix_regime_outcome": {
                k: _finalise_two_way(v)
                for k, v in vix_regime_outcome.items()
            },
            "long_term_macro_outcome": {
                k: _finalise_two_way(v)
                for k, v in long_term_macro_outcome.items()
            },
            # PR-A v2 — USD-direction-normalised + per-symbol cuts.
            "dxy_trend_by_usd_exposure_outcome": {
                k: _finalise_two_way(v)
                for k, v in dxy_trend_by_usd_exposure_outcome.items()
            },
            "symbol_dxy_trend_outcome": {
                k: _finalise_two_way(v)
                for k, v in symbol_dxy_trend_outcome.items()
            },
            # PR #15 — waveform-match crosses.
            "waveform_bias_direction_outcome": {
                k: _finalise_two_way(v)
                for k, v in waveform_bias_direction_outcome.items()
            },
            "waveform_confidence_bucket_outcome": {
                k: _finalise_two_way(v)
                for k, v in waveform_confidence_bucket_outcome.items()
            },
            "top_similarity_bucket_outcome": {
                k: _finalise_two_way(v)
                for k, v in top_similarity_bucket_outcome.items()
            },
            "matched_count_bucket_outcome": {
                k: _finalise_two_way(v)
                for k, v in matched_count_bucket_outcome.items()
            },
            "waveform_bias_by_technical_action_outcome": {
                k: _finalise_two_way(v)
                for k, v in waveform_bias_by_technical_action_outcome.items()
            },
            "symbol_waveform_bias_outcome": {
                k: _finalise_two_way(v)
                for k, v in symbol_waveform_bias_outcome.items()
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
    # global top_hold_reasons is derived from g_hold_reason_pool below —
    # NOT from per-run top_hold_reasons (which is truncated to top-N and
    # would silently drop reasons #N+1 in every run that are #2 globally).
    g_hold_reason_pool: dict[str, dict[str, Any]] = {}
    g_gate_by_tech: dict[str, dict[str, int]] = {}
    g_final_by_outcome: dict[str, dict[str, int]] = {}
    # PR #10: Risk-Gate code → pooled accumulator. Same row schema as
    # g_hold_reason_pool, so the same finaliser applies.
    g_blocked_by_outcome_pool: dict[str, dict[str, Any]] = {}

    # PR-A: long-term / macro-regime pools.
    # Each is bucket → final_action → {n, outcome counter}.
    g_daily_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_weekly_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_monthly_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_close_vs_sma_200d_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_weekly_return_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_monthly_return_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_dxy_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_us10y_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_yield_spread_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_vix_regime_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_long_term_macro_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_dxy_trend_by_usd_exposure_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_symbol_dxy_trend_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_waveform_bias_direction_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_waveform_confidence_bucket_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_top_similarity_bucket_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_matched_count_bucket_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_waveform_bias_by_technical_action_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    g_symbol_waveform_bias_outcome: dict[str, dict[str, dict[str, Any]]] = {}

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

        # NOTE: per-run `stats["top_hold_reasons"]` is already truncated
        # to top-N at the per-run aggregator. Folding those truncated
        # lists into a global counter would silently drop reasons that
        # were #N+1 in every run but #2 globally. Skip it here — the
        # global top is rebuilt from `g_hold_reason_pool` (which has the
        # full per-reason count) after the loop.

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

        # PR #10: blocked_by_outcome — same pooling shape as hold_reason_outcome.
        for code, row in cs.get("blocked_by_outcome", {}).items():
            pooled = g_blocked_by_outcome_pool.setdefault(
                code, _new_pooled_hold_reason_row()
            )
            _pool_hold_reason_row(pooled, row)

        # PR-A: pool the long-term / macro-regime two-way crosses.
        # Each per-run cell shape is bucket → final_action → {n, outcome}.
        # Pooling sums n and the per-outcome counts cell-by-cell.
        def _pool_two_way(
            target: dict[str, dict[str, dict[str, Any]]],
            run_section: dict[str, Any],
        ) -> None:
            for bucket, by_action in run_section.items():
                if not isinstance(by_action, dict):
                    continue
                tgt_bucket = target.setdefault(bucket, {})
                for action, leaf in by_action.items():
                    if not isinstance(leaf, dict):
                        continue
                    tgt_leaf = tgt_bucket.setdefault(
                        action, {"n": 0, "outcome": Counter()}
                    )
                    tgt_leaf["n"] += int(leaf.get("n", 0))
                    out = leaf.get("outcome") or {}
                    if isinstance(out, dict):
                        for k, v in out.items():
                            tgt_leaf["outcome"][k] += int(v)

        _pool_two_way(g_daily_trend_outcome,
                      cs.get("daily_trend_outcome", {}))
        _pool_two_way(g_weekly_trend_outcome,
                      cs.get("weekly_trend_outcome", {}))
        _pool_two_way(g_monthly_trend_outcome,
                      cs.get("monthly_trend_outcome", {}))
        _pool_two_way(g_close_vs_sma_200d_outcome,
                      cs.get("close_vs_sma_200d_outcome", {}))
        _pool_two_way(g_weekly_return_outcome,
                      cs.get("weekly_return_outcome", {}))
        _pool_two_way(g_monthly_return_outcome,
                      cs.get("monthly_return_outcome", {}))
        _pool_two_way(g_dxy_trend_outcome,
                      cs.get("dxy_trend_outcome", {}))
        _pool_two_way(g_us10y_trend_outcome,
                      cs.get("us10y_trend_outcome", {}))
        _pool_two_way(g_yield_spread_outcome,
                      cs.get("yield_spread_outcome", {}))
        _pool_two_way(g_vix_regime_outcome,
                      cs.get("vix_regime_outcome", {}))
        _pool_two_way(g_long_term_macro_outcome,
                      cs.get("long_term_macro_outcome", {}))
        _pool_two_way(g_dxy_trend_by_usd_exposure_outcome,
                      cs.get("dxy_trend_by_usd_exposure_outcome", {}))
        _pool_two_way(g_symbol_dxy_trend_outcome,
                      cs.get("symbol_dxy_trend_outcome", {}))
        _pool_two_way(g_waveform_bias_direction_outcome,
                      cs.get("waveform_bias_direction_outcome", {}))
        _pool_two_way(g_waveform_confidence_bucket_outcome,
                      cs.get("waveform_confidence_bucket_outcome", {}))
        _pool_two_way(g_top_similarity_bucket_outcome,
                      cs.get("top_similarity_bucket_outcome", {}))
        _pool_two_way(g_matched_count_bucket_outcome,
                      cs.get("matched_count_bucket_outcome", {}))
        _pool_two_way(g_waveform_bias_by_technical_action_outcome,
                      cs.get("waveform_bias_by_technical_action_outcome", {}))
        _pool_two_way(g_symbol_waveform_bias_outcome,
                      cs.get("symbol_waveform_bias_outcome", {}))

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

    # Top hold reasons globally — derived from the *full* hold_reason_pool
    # (which has the precise count of every distinct decision.reason
    # across all HOLD bars in all runs). Re-deriving here is the only
    # way to be correct: per-run top-N truncation would silently drop
    # reasons that ranked just below the cut-off in each run but
    # cumulatively rank near the top globally. Pinned by
    # test_top_hold_reasons_pooled_includes_globally_ranked_reason.
    top_pairs = sorted(
        ((reason, row["n"]) for reason, row in g_hold_reason_pool.items()),
        key=lambda kv: (-kv[1], kv[0]),
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
            "blocked_by_outcome": {
                code: _finalise_pooled_hold_reason_row(row)
                for code, row in g_blocked_by_outcome_pool.items()
            },
            # PR-A — pooled long-term / macro-regime crosses.
            "daily_trend_outcome": _finalise_pooled_two_way(
                g_daily_trend_outcome
            ),
            "weekly_trend_outcome": _finalise_pooled_two_way(
                g_weekly_trend_outcome
            ),
            "monthly_trend_outcome": _finalise_pooled_two_way(
                g_monthly_trend_outcome
            ),
            "close_vs_sma_200d_outcome": _finalise_pooled_two_way(
                g_close_vs_sma_200d_outcome
            ),
            "weekly_return_outcome": _finalise_pooled_two_way(
                g_weekly_return_outcome
            ),
            "monthly_return_outcome": _finalise_pooled_two_way(
                g_monthly_return_outcome
            ),
            "dxy_trend_outcome": _finalise_pooled_two_way(
                g_dxy_trend_outcome
            ),
            "us10y_trend_outcome": _finalise_pooled_two_way(
                g_us10y_trend_outcome
            ),
            "yield_spread_outcome": _finalise_pooled_two_way(
                g_yield_spread_outcome
            ),
            "vix_regime_outcome": _finalise_pooled_two_way(
                g_vix_regime_outcome
            ),
            "long_term_macro_outcome": _finalise_pooled_two_way(
                g_long_term_macro_outcome
            ),
            "dxy_trend_by_usd_exposure_outcome": _finalise_pooled_two_way(
                g_dxy_trend_by_usd_exposure_outcome
            ),
            "symbol_dxy_trend_outcome": _finalise_pooled_two_way(
                g_symbol_dxy_trend_outcome
            ),
            "waveform_bias_direction_outcome": _finalise_pooled_two_way(
                g_waveform_bias_direction_outcome
            ),
            "waveform_confidence_bucket_outcome": _finalise_pooled_two_way(
                g_waveform_confidence_bucket_outcome
            ),
            "top_similarity_bucket_outcome": _finalise_pooled_two_way(
                g_top_similarity_bucket_outcome
            ),
            "matched_count_bucket_outcome": _finalise_pooled_two_way(
                g_matched_count_bucket_outcome
            ),
            "waveform_bias_by_technical_action_outcome": _finalise_pooled_two_way(
                g_waveform_bias_by_technical_action_outcome
            ),
            "symbol_waveform_bias_outcome": _finalise_pooled_two_way(
                g_symbol_waveform_bias_outcome
            ),
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
