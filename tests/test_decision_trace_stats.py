"""Tests for decision_trace_stats.aggregate_stats and the trace-stats CLI."""
from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace import RULE_TAXONOMY, TRACE_SCHEMA_VERSION
from src.fx.decision_trace_io import export_run
from src.fx.decision_trace_stats import (
    MAX_ERROR_LIST_SIZE,
    aggregate_stats,
)


# Resolve repo root so subprocess CLI invocations work on any host
# (developer laptop, CI, container) — never hardcode an absolute path.
# tests/ -> repo_root.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def _produce_jsonl(tmp_path: Path, *, seed: int = 1, n: int = 200,
                   sub: str = "run01") -> Path:
    df = _ohlcv(n, seed=seed)
    res = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                              capture_traces=True)
    out_dir = tmp_path / sub
    paths = export_run(res, out_dir=out_dir)
    return paths["decision_traces"]


# ─── Read paths: jsonl + gzip ──────────────────────────────────────────────


def test_aggregate_stats_reads_jsonl(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="r01")
    stats = aggregate_stats(jsonl_path)
    assert stats["metadata"]["n_traces"] > 0
    assert stats["metadata"]["input_path"] == str(jsonl_path)
    assert stats["metadata"]["trace_schema_version"] == TRACE_SCHEMA_VERSION


def test_aggregate_stats_reads_gzip_jsonl(tmp_path):
    df = _ohlcv(200, seed=2)
    res = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60)
    out_dir = tmp_path / "rgz"
    paths = export_run(res, out_dir=out_dir, gzip=True)
    stats = aggregate_stats(paths["decision_traces"])
    assert stats["metadata"]["n_traces"] == len(res.decision_traces)


def test_aggregate_stats_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_stats(tmp_path / "no_such_file.jsonl")


# ─── Distributions ─────────────────────────────────────────────────────────


def test_action_distribution_counts_match(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="ad")
    stats = aggregate_stats(jsonl_path)
    final = stats["action_distribution"]["final_action"]
    tech = stats["action_distribution"]["technical_only_action"]
    # Sum of counts must equal n_traces (every trace has both fields).
    assert sum(final.values()) == stats["metadata"]["n_traces"]
    assert sum(tech.values()) == stats["metadata"]["n_traces"]
    # All keys in {BUY, SELL, HOLD}.
    assert set(final.keys()) <= {"BUY", "SELL", "HOLD"}


def test_blocked_by_distribution_includes_no_block(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="bb")
    stats = aggregate_stats(jsonl_path)
    bb = stats["blocked_by_distribution"]
    # Some bars on this fixture have empty blocked_by; key must exist.
    assert "no_block" in bb
    # no_block + sum(other codes) cannot exceed n_traces (a bar can have
    # multiple codes, so it can also be larger — but no_block alone <= n)
    assert bb["no_block"] <= stats["metadata"]["n_traces"]


def test_rule_result_distribution_includes_all_19_canonical_rule_ids(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="rr")
    stats = aggregate_stats(jsonl_path)
    rd = stats["rule_result_distribution"]
    assert set(RULE_TAXONOMY.keys()) <= set(rd.keys())
    # Every cell present (pre-seeded) — even when zero.
    for rid in RULE_TAXONOMY:
        for bucket in ("PASS", "BLOCK", "WARN", "SKIPPED",
                       "NOT_REACHED", "INFO"):
            assert bucket in rd[rid], (
                f"{rid} missing bucket {bucket}"
            )


def test_gate_effect_distribution_present(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="ge")
    stats = aggregate_stats(jsonl_path)
    ge = stats["gate_effect_distribution"]
    # On a synthetic random fixture, NO_CHANGE is essentially guaranteed
    # somewhere; but we only require the key be a dict (not None) and
    # values be ints.
    assert isinstance(ge, dict)
    for v in ge.values():
        assert isinstance(v, int)


def test_entry_skipped_reason_counts_present(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="es")
    stats = aggregate_stats(jsonl_path)
    eed = stats["entry_execution_distribution"]
    assert "entry_executed_true" in eed and "entry_executed_false" in eed
    assert "exit_event_true" in eed and "exit_event_false" in eed
    assert isinstance(eed["entry_skipped_reason"], dict)
    # Sum of (executed_true + executed_false) == n_traces with bool fields.
    n = stats["metadata"]["n_traces"]
    assert eed["entry_executed_true"] + eed["entry_executed_false"] == n


def test_top_hold_reasons_lists_pairs(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="th")
    stats = aggregate_stats(jsonl_path, top_n_hold_reasons=5)
    pairs = stats["top_hold_reasons"]
    assert isinstance(pairs, list) and len(pairs) <= 5
    for p in pairs:
        assert "reason" in p and "count" in p
        assert isinstance(p["reason"], str) and isinstance(p["count"], int)


# ─── Consistency checks ───────────────────────────────────────────────────


def test_consistency_checks_detects_run_id_mismatch(tmp_path):
    """Concatenate two runs with distinct run_ids; aggregator must flag."""
    p1 = _produce_jsonl(tmp_path, sub="cm1", seed=11, n=100)
    p2 = _produce_jsonl(tmp_path, sub="cm2", seed=22, n=100)
    merged = tmp_path / "merged.jsonl"
    merged.write_bytes(p1.read_bytes() + p2.read_bytes())

    stats = aggregate_stats(merged)
    cc = stats["consistency_checks"]
    assert cc["run_id_consistent"] is False
    assert cc["run_id_unique_count"] == 2
    assert cc["errors_total"] >= 1
    # error message references the mismatch
    assert any("run_id mismatch" in m for m in cc["errors"])


def test_malformed_json_line_recorded_as_error(tmp_path):
    """Append a broken line; aggregator continues + errors_total > 0."""
    jsonl_path = _produce_jsonl(tmp_path, sub="mj")
    # Append a malformed line.
    with jsonl_path.open("a", encoding="utf-8") as fp:
        fp.write("{this is not json}\n")
    stats = aggregate_stats(jsonl_path)
    cc = stats["consistency_checks"]
    assert cc["errors_total"] >= 1
    assert any("malformed JSON" in m for m in cc["errors"])
    # Valid traces still counted.
    assert stats["metadata"]["n_traces"] > 0


def test_empty_file_raises_valueerror(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        aggregate_stats(empty)
    assert "empty" in str(exc.value).lower()


def test_all_lines_malformed_raises_valueerror(tmp_path):
    bad = tmp_path / "all_bad.jsonl"
    bad.write_text("{not json}\n[oops]\n}}}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        aggregate_stats(bad)
    # n_traces==0 path: errors_total tracked in message.
    assert "no parseable traces" in str(exc.value)


def test_errors_list_capped(tmp_path):
    """Inject more malformed lines than the cap; list is bounded but
    errors_total counts them all and errors_truncated=True."""
    jsonl_path = _produce_jsonl(tmp_path, sub="cap")
    with jsonl_path.open("a", encoding="utf-8") as fp:
        for _ in range(MAX_ERROR_LIST_SIZE + 20):
            fp.write("{broken}\n")
    stats = aggregate_stats(jsonl_path)
    cc = stats["consistency_checks"]
    assert cc["errors_total"] >= MAX_ERROR_LIST_SIZE + 20
    assert len(cc["errors"]) <= MAX_ERROR_LIST_SIZE
    assert cc["errors_truncated"] is True


def test_timestamp_order_warning(tmp_path):
    """Reverse two adjacent records; warning recorded, error NOT raised."""
    jsonl_path = _produce_jsonl(tmp_path, sub="ts")
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        pytest.skip("need at least 3 traces to swap")
    # Swap line 1 and line 2 to break monotonicity.
    lines[1], lines[2] = lines[2], lines[1]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = aggregate_stats(jsonl_path)
    cc = stats["consistency_checks"]
    assert cc["warnings_total"] >= 1
    assert any("timestamp not monotonic" in m for m in cc["warnings"])


def test_run_id_consistent_on_normal_export(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="ok")
    stats = aggregate_stats(jsonl_path)
    cc = stats["consistency_checks"]
    assert cc["run_id_consistent"] is True
    assert cc["trace_schema_version_consistent"] is True
    assert cc["traces_missing_rule_checks"] == 0
    assert cc["traces_missing_future_outcome"] == 0
    assert cc["errors_total"] == 0


# ─── CLI invocation tests ─────────────────────────────────────────────────


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.fx.cli", *args],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
    )


def test_cli_trace_stats_compact_json(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="cli1")
    cp = _run_cli("trace-stats", str(jsonl_path))
    assert cp.returncode == 0, cp.stderr
    # Compact: no leading whitespace before "metadata", single line for
    # the top dict (json.dumps without indent puts everything on one line).
    out = cp.stdout.strip()
    assert "\n" not in out, "compact output should be a single line"
    parsed = json.loads(out)
    assert parsed["metadata"]["n_traces"] > 0


def test_cli_trace_stats_pretty_json(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="cli2")
    cp = _run_cli("trace-stats", str(jsonl_path), "--pretty")
    assert cp.returncode == 0, cp.stderr
    out = cp.stdout
    # Pretty: indent=2 produces multi-line output and "  " indent at start of
    # nested keys.
    assert out.count("\n") > 5
    assert '\n  "metadata"' in out
    parsed = json.loads(out)
    assert parsed["metadata"]["n_traces"] > 0


def test_cli_exit_2_on_missing_file(tmp_path):
    cp = _run_cli("trace-stats", str(tmp_path / "no_such.jsonl"))
    assert cp.returncode == 2
    assert "trace-stats" in cp.stderr


def test_cli_exit_2_on_partial_read_with_errors(tmp_path):
    jsonl_path = _produce_jsonl(tmp_path, sub="cli3")
    with jsonl_path.open("a", encoding="utf-8") as fp:
        fp.write("{not json}\n")
    cp = _run_cli("trace-stats", str(jsonl_path))
    # Stats are emitted to stdout, but exit code is 2 to flag the error.
    assert cp.returncode == 2
    parsed = json.loads(cp.stdout)
    assert parsed["consistency_checks"]["errors_total"] >= 1


# ─── Trade-result invariance ──────────────────────────────────────────────


def test_aggregate_stats_does_not_change_backtest_results(tmp_path):
    """Calling aggregate_stats() must not perturb a subsequent backtest
    (no global state, no side-effects on the engine)."""
    df = _ohlcv(300, seed=99)
    res_a = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                                capture_traces=True)
    # Export + aggregate.
    out_dir = tmp_path / "inv"
    paths = export_run(res_a, out_dir=out_dir)
    _ = aggregate_stats(paths["decision_traces"])

    res_b = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                                capture_traces=True)

    def _summary(r):
        return [
            (t.entry_ts, t.side, t.exit_ts, t.exit_reason,
             round(t.return_pct, 6))
            for t in r.trades
        ], dict(r.hold_reasons), r.bars_processed

    assert _summary(res_a) == _summary(res_b)
    assert res_a.metrics() == res_b.metrics()


# ─── PR #7: cross_stats section ─────────────────────────────────────────────

def _build_synthetic_record(
    *,
    run_id: str = "bt_test_X_aaaa",
    timestamp: str = "2025-01-01T00:00:00+00:00",
    bar_index: int = 0,
    final_action: str = "HOLD",
    technical_action: str = "HOLD",
    reason: str = "Technical signal is HOLD; nothing to confirm",
    blocked_by: tuple = (),
    gate_effect: str | None = "NO_CHANGE",
    outcome: str | None = "N/A",
    hypothetical_return_pct: float | None = None,
    entry_executed: bool = False,
    exit_event: bool = False,
    entry_skipped_reason: str = "decision_was_HOLD",
) -> dict:
    """Minimal record matching the keys aggregate_stats() reads."""
    fo: dict = {}
    if gate_effect is not None:
        fo["gate_effect"] = gate_effect
    if outcome is not None:
        fo["outcome_if_technical_action_taken"] = outcome
    fo["hypothetical_technical_trade_return_pct"] = hypothetical_return_pct
    return {
        "run_id": run_id,
        "trace_schema_version": "decision_trace_v1",
        "bar_id": f"X_1h_{timestamp}",
        "timestamp": timestamp,
        "symbol": "X",
        "timeframe": "1h",
        "bar_index": bar_index,
        "decision": {
            "final_action": final_action,
            "technical_only_action": technical_action,
            "reason": reason,
            "blocked_by": list(blocked_by),
            "rule_chain": [],
            "advisory": {},
            "confidence": 0.0,
            "action_changed_by_engine": final_action != technical_action,
        },
        # Provide the canonical rule_id so rule_result_distribution stays
        # populated; other rule_ids are absent (which is allowed — the
        # aggregator pre-seeds 19 cells regardless).
        "rule_checks": [
            {"canonical_rule_id": "final_decision",
             "result": "BLOCK" if final_action == "HOLD" else "PASS",
             "computed": True, "used_in_decision": True,
             "value": final_action, "threshold": None,
             "evidence_ids": [], "reason": reason,
             "source_chain_step": "terminal"},
        ],
        "future_outcome": fo,
        "execution_trace": {
            "entry_executed": entry_executed,
            "entry_price": None,
            "entry_skipped_reason": entry_skipped_reason,
            "exit_event": exit_event,
            "exit_reason": None,
            "exit_price": None,
        },
    }


def _write_jsonl(tmp_path: Path, records: list[dict],
                 name: str = "synth.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return p


def test_cross_stats_top_level_key_present(tmp_path):
    p = _write_jsonl(tmp_path, [_build_synthetic_record()])
    stats = aggregate_stats(p)
    assert "cross_stats" in stats
    cs = stats["cross_stats"]
    assert set(cs.keys()) == {
        "hold_reason_outcome",
        "gate_effect_by_technical_action",
        "final_action_by_outcome",
        "blocked_by_outcome",
        # PR-A: long-term trend + macro context crosses
        "daily_trend_outcome",
        "weekly_trend_outcome",
        "monthly_trend_outcome",
        "close_vs_sma_200d_outcome",
        "weekly_return_outcome",
        "monthly_return_outcome",
        "dxy_trend_outcome",
        "us10y_trend_outcome",
        "yield_spread_outcome",
        "vix_regime_outcome",
        "long_term_macro_outcome",
    }


def test_cross_stats_hold_reason_outcome_aggregates_per_reason(tmp_path):
    """Two distinct HOLD reasons must produce two separate rows, each
    accumulating its own gate_effect / outcome / tech_action counters."""
    recs = [
        _build_synthetic_record(
            reason="Counter-trend BUY",
            technical_action="BUY",
            gate_effect="PROTECTED",
            outcome="LOSS_AVOIDED",
            timestamp="2025-01-01T00:00:00+00:00", bar_index=0,
        ),
        _build_synthetic_record(
            reason="Counter-trend BUY",
            technical_action="BUY",
            gate_effect="COST_OPPORTUNITY",
            outcome="WIN_MISSED",
            timestamp="2025-01-01T01:00:00+00:00", bar_index=1,
        ),
        _build_synthetic_record(
            reason="Risk gate blocked: FOMC",
            technical_action="SELL",
            gate_effect="PROTECTED",
            outcome="LOSS_AVOIDED",
            timestamp="2025-01-01T02:00:00+00:00", bar_index=2,
        ),
    ]
    p = _write_jsonl(tmp_path, recs)
    cs = aggregate_stats(p)["cross_stats"]
    hro = cs["hold_reason_outcome"]
    assert set(hro.keys()) == {"Counter-trend BUY", "Risk gate blocked: FOMC"}
    assert hro["Counter-trend BUY"]["n"] == 2
    assert hro["Counter-trend BUY"]["gate_effect"] == {
        "PROTECTED": 1, "COST_OPPORTUNITY": 1
    }
    assert hro["Counter-trend BUY"]["outcome_if_technical_action_taken"] == {
        "LOSS_AVOIDED": 1, "WIN_MISSED": 1
    }
    assert hro["Counter-trend BUY"]["technical_only_action"] == {"BUY": 2}
    assert hro["Risk gate blocked: FOMC"]["n"] == 1
    assert hro["Risk gate blocked: FOMC"]["technical_only_action"] == {"SELL": 1}


def test_cross_stats_protected_vs_cost_opportunity_separated_per_reason(tmp_path):
    """Within one reason, PROTECTED and COST_OPPORTUNITY must be cleanly
    separated — they were collapsing on a buggy implementation."""
    recs = [
        _build_synthetic_record(reason="rA", gate_effect="PROTECTED",
                                outcome="LOSS_AVOIDED", bar_index=i,
                                timestamp=f"2025-01-01T{i:02d}:00:00+00:00")
        for i in range(3)
    ] + [
        _build_synthetic_record(reason="rA", gate_effect="COST_OPPORTUNITY",
                                outcome="WIN_MISSED", bar_index=10 + i,
                                timestamp=f"2025-01-02T{i:02d}:00:00+00:00")
        for i in range(2)
    ]
    p = _write_jsonl(tmp_path, recs)
    cs = aggregate_stats(p)["cross_stats"]
    row = cs["hold_reason_outcome"]["rA"]
    assert row["n"] == 5
    assert row["gate_effect"] == {"PROTECTED": 3, "COST_OPPORTUNITY": 2}
    assert row["outcome_if_technical_action_taken"] == {
        "LOSS_AVOIDED": 3, "WIN_MISSED": 2
    }


def test_cross_stats_return_stats_avg_sum_min_max(tmp_path):
    returns = [-0.5, 0.3, 0.1, -0.2, 0.4]   # sum 0.1, avg 0.02, min -0.5, max 0.4
    recs = [
        _build_synthetic_record(
            reason="rB", hypothetical_return_pct=r,
            timestamp=f"2025-01-01T{i:02d}:00:00+00:00", bar_index=i,
        )
        for i, r in enumerate(returns)
    ]
    p = _write_jsonl(tmp_path, recs)
    cs = aggregate_stats(p)["cross_stats"]
    rs = cs["hold_reason_outcome"]["rB"]["return_stats"]
    assert rs["n_with_return"] == 5
    assert rs["sum_pct"] == pytest.approx(0.1)
    assert rs["avg_pct"] == pytest.approx(0.02)
    assert rs["min_pct"] == pytest.approx(-0.5)
    assert rs["max_pct"] == pytest.approx(0.4)


def test_cross_stats_return_stats_null_when_no_returns(tmp_path):
    """When every hypothetical_return_pct is null/missing, return_stats
    must report n_with_return=0 with all aggregate fields null."""
    recs = [
        _build_synthetic_record(reason="rC", hypothetical_return_pct=None,
                                bar_index=i,
                                timestamp=f"2025-01-01T{i:02d}:00:00+00:00")
        for i in range(3)
    ]
    p = _write_jsonl(tmp_path, recs)
    rs = aggregate_stats(p)["cross_stats"]["hold_reason_outcome"]["rC"][
        "return_stats"
    ]
    assert rs["n_with_return"] == 0
    assert rs["avg_pct"] is None
    assert rs["sum_pct"] is None
    assert rs["min_pct"] is None
    assert rs["max_pct"] is None


def test_cross_stats_gate_effect_by_technical_action(tmp_path):
    """Bucket every bar's gate_effect under its technical_only_action."""
    recs = [
        _build_synthetic_record(final_action="HOLD", technical_action="BUY",
                                gate_effect="PROTECTED", bar_index=0,
                                timestamp="2025-01-01T00:00:00+00:00"),
        _build_synthetic_record(final_action="HOLD", technical_action="BUY",
                                gate_effect="COST_OPPORTUNITY", bar_index=1,
                                timestamp="2025-01-01T01:00:00+00:00"),
        _build_synthetic_record(final_action="BUY", technical_action="BUY",
                                gate_effect="NO_CHANGE", bar_index=2,
                                timestamp="2025-01-01T02:00:00+00:00",
                                entry_executed=True,
                                entry_skipped_reason="entry_executed"),
        _build_synthetic_record(final_action="HOLD", technical_action="SELL",
                                gate_effect="NO_CHANGE", bar_index=3,
                                timestamp="2025-01-01T03:00:00+00:00"),
        _build_synthetic_record(final_action="HOLD", technical_action="HOLD",
                                gate_effect="NO_CHANGE", bar_index=4,
                                timestamp="2025-01-01T04:00:00+00:00"),
    ]
    p = _write_jsonl(tmp_path, recs)
    geba = aggregate_stats(p)["cross_stats"][
        "gate_effect_by_technical_action"
    ]
    assert geba["BUY"] == {
        "PROTECTED": 1, "COST_OPPORTUNITY": 1, "NO_CHANGE": 1
    }
    assert geba["SELL"] == {"NO_CHANGE": 1}
    assert geba["HOLD"] == {"NO_CHANGE": 1}


def test_cross_stats_final_action_by_outcome(tmp_path):
    recs = [
        _build_synthetic_record(final_action="BUY", outcome="WIN",
                                bar_index=0,
                                timestamp="2025-01-01T00:00:00+00:00",
                                entry_executed=True,
                                entry_skipped_reason="entry_executed"),
        _build_synthetic_record(final_action="BUY", outcome="LOSS",
                                bar_index=1,
                                timestamp="2025-01-01T01:00:00+00:00",
                                entry_executed=True,
                                entry_skipped_reason="entry_executed"),
        _build_synthetic_record(final_action="HOLD", outcome="LOSS_AVOIDED",
                                bar_index=2,
                                timestamp="2025-01-01T02:00:00+00:00"),
        _build_synthetic_record(final_action="HOLD", outcome="N/A",
                                bar_index=3,
                                timestamp="2025-01-01T03:00:00+00:00"),
    ]
    p = _write_jsonl(tmp_path, recs)
    fabo = aggregate_stats(p)["cross_stats"]["final_action_by_outcome"]
    assert fabo["BUY"] == {"WIN": 1, "LOSS": 1}
    assert fabo["HOLD"] == {"LOSS_AVOIDED": 1, "N/A": 1}


def test_cross_stats_with_malformed_line_still_aggregates_valid_lines(tmp_path):
    """A malformed line must NOT block cross-stats accumulation for the
    valid lines around it."""
    p = tmp_path / "mixed.jsonl"
    valid = [
        _build_synthetic_record(reason="rA", technical_action="BUY",
                                gate_effect="PROTECTED",
                                outcome="LOSS_AVOIDED", bar_index=0,
                                timestamp="2025-01-01T00:00:00+00:00"),
        _build_synthetic_record(reason="rA", technical_action="BUY",
                                gate_effect="COST_OPPORTUNITY",
                                outcome="WIN_MISSED", bar_index=2,
                                timestamp="2025-01-01T02:00:00+00:00"),
    ]
    p.write_text(
        json.dumps(valid[0]) + "\n"
        "{not json}\n"
        + json.dumps(valid[1]) + "\n",
        encoding="utf-8",
    )
    stats = aggregate_stats(p)
    assert stats["consistency_checks"]["errors_total"] >= 1
    row = stats["cross_stats"]["hold_reason_outcome"]["rA"]
    assert row["n"] == 2
    assert row["gate_effect"] == {
        "PROTECTED": 1, "COST_OPPORTUNITY": 1
    }


def test_cross_stats_empty_when_no_hold_bars(tmp_path):
    """A fixture made entirely of BUY bars yields hold_reason_outcome={}
    while the other two cross dicts still get populated."""
    recs = [
        _build_synthetic_record(
            final_action="BUY", technical_action="BUY",
            outcome="WIN", gate_effect="NO_CHANGE",
            bar_index=i,
            timestamp=f"2025-01-01T{i:02d}:00:00+00:00",
            entry_executed=True, entry_skipped_reason="entry_executed",
        )
        for i in range(3)
    ]
    p = _write_jsonl(tmp_path, recs)
    cs = aggregate_stats(p)["cross_stats"]
    assert cs["hold_reason_outcome"] == {}
    assert cs["gate_effect_by_technical_action"]["BUY"] == {"NO_CHANGE": 3}
    assert cs["final_action_by_outcome"]["BUY"] == {"WIN": 3}


def test_existing_top_level_keys_unchanged(tmp_path):
    """PR #7 must not change the existing 9 top-level sections."""
    p = _write_jsonl(tmp_path, [_build_synthetic_record()])
    stats = aggregate_stats(p)
    expected_existing = {
        "metadata", "action_distribution", "blocked_by_distribution",
        "rule_result_distribution", "gate_effect_distribution",
        "outcome_distribution", "entry_execution_distribution",
        "top_hold_reasons", "consistency_checks",
    }
    assert expected_existing <= set(stats.keys())
    # Type contract preserved
    assert isinstance(stats["action_distribution"]["final_action"], dict)
    assert isinstance(stats["consistency_checks"]["errors_total"], int)
    assert isinstance(stats["top_hold_reasons"], list)


def test_real_export_produces_valid_cross_stats(tmp_path):
    """End-to-end: run_engine_backtest -> export_run -> aggregate_stats.
    Sanity-check the cross_stats shape on real engine output."""
    df = _ohlcv(200, seed=7)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    paths = export_run(res, out_dir=tmp_path / "real")
    cs = aggregate_stats(paths["decision_traces"])["cross_stats"]
    # Shape only — content varies with random fixture.
    assert isinstance(cs["hold_reason_outcome"], dict)
    assert isinstance(cs["gate_effect_by_technical_action"], dict)
    assert isinstance(cs["final_action_by_outcome"], dict)
    # Sum of final_action_by_outcome must equal the count of bars where a
    # gate_effect was recorded for that final_action — i.e. all bars that
    # carry both final_action and outcome strings. With the real engine,
    # every processed bar has both, so the cross sum must equal n_traces.
    total_fabo = sum(
        sum(counts.values()) for counts in cs["final_action_by_outcome"].values()
    )
    assert total_fabo == len(res.decision_traces)
    # hold_reason_outcome n's must sum to the count of HOLD bars.
    n_hold_bars = sum(
        1 for t in res.decision_traces if t.decision.final_action == "HOLD"
    )
    sum_hro_n = sum(row["n"] for row in cs["hold_reason_outcome"].values())
    assert sum_hro_n == n_hold_bars


def test_gitignore_excludes_runs():
    """Generated trace exports under runs/ must never be committed."""
    gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in gi.splitlines()]
    assert "runs/" in lines, (
        f".gitignore must contain a line 'runs/' to exclude trace export "
        f"artefacts; current contents:\n{gi}"
    )


# ─── PR #8: aggregate_many / trace-stats-multi ──────────────────────────────


def _write_run_dir(
    tmp_path: Path,
    name: str,
    *,
    records: list[dict],
    summary_metrics: dict | None = None,
) -> Path:
    """Materialise a runs/<name>/ directory with decision_traces.jsonl
    and (optionally) summary.json. Returns the jsonl path."""
    d = tmp_path / name
    d.mkdir()
    jsonl = d / "decision_traces.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    if summary_metrics is not None:
        (d / "summary.json").write_text(
            json.dumps({"metrics": summary_metrics}),
            encoding="utf-8",
        )
    return jsonl


def test_aggregate_many_reads_multiple_jsonl(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1",
                                timestamp="2025-01-01T00:00:00+00:00"),
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(run_id="r2",
                                timestamp="2025-01-02T00:00:00+00:00"),
    ])
    out = aggregate_many([p1, p2])
    assert out["metadata"]["n_runs"] == 2
    assert out["metadata"]["n_runs_succeeded"] == 2
    assert out["metadata"]["n_runs_failed"] == 0
    assert set(out.keys()) == {
        "metadata", "per_run", "global", "consistency_checks"
    }


def test_per_run_count_matches_input_paths(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    paths = [
        _write_run_dir(tmp_path, f"r{i}", records=[
            _build_synthetic_record(
                run_id=f"r{i}",
                timestamp=f"2025-01-{i+1:02d}T00:00:00+00:00",
            )
        ])
        for i in range(3)
    ]
    out = aggregate_many(paths)
    assert len(out["per_run"]) == 3
    for entry, p in zip(out["per_run"], paths):
        assert entry["input_path"] == str(p)
        assert "run_id" in entry
        assert "n_traces" in entry
        assert "cross_stats" in entry


def test_global_n_traces_total_equals_sum_of_per_run(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    sizes = [3, 5, 2]
    paths = []
    for i, n in enumerate(sizes):
        recs = [
            _build_synthetic_record(
                run_id=f"r{i}",
                timestamp=f"2025-01-{i+1:02d}T{j:02d}:00:00+00:00",
                bar_index=j,
            )
            for j in range(n)
        ]
        paths.append(_write_run_dir(tmp_path, f"r{i}", records=recs))
    out = aggregate_many(paths)
    assert out["metadata"]["n_traces_total"] == sum(sizes)
    assert sum(e["n_traces"] for e in out["per_run"]) == sum(sizes)


def test_global_cross_stats_hold_reason_outcome_pools_across_runs(tmp_path):
    """Same `decision.reason` in two runs must collapse into one global row
    with summed counters."""
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(
            run_id="r1", reason="rA", technical_action="BUY",
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            timestamp="2025-01-01T00:00:00+00:00", bar_index=0,
        ),
        _build_synthetic_record(
            run_id="r1", reason="rA", technical_action="BUY",
            gate_effect="COST_OPPORTUNITY", outcome="WIN_MISSED",
            timestamp="2025-01-01T01:00:00+00:00", bar_index=1,
        ),
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(
            run_id="r2", reason="rA", technical_action="BUY",
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            timestamp="2025-01-02T00:00:00+00:00", bar_index=0,
        ),
    ])
    out = aggregate_many([p1, p2])
    pooled = out["global"]["cross_stats"]["hold_reason_outcome"]["rA"]
    assert pooled["n"] == 3
    assert pooled["gate_effect"] == {"PROTECTED": 2, "COST_OPPORTUNITY": 1}
    assert pooled["outcome_if_technical_action_taken"] == {
        "LOSS_AVOIDED": 2, "WIN_MISSED": 1
    }
    assert pooled["technical_only_action"] == {"BUY": 3}


def test_return_stats_pooled_sum_avg_min_max(tmp_path):
    """Two runs with separate return values must pool into the right
    sum / avg / cross-run min / cross-run max."""
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(
            run_id="r1", reason="rA",
            hypothetical_return_pct=v,
            timestamp=f"2025-01-01T{i:02d}:00:00+00:00", bar_index=i,
        )
        for i, v in enumerate([-0.5, 0.3])    # sum=-0.2, n=2, min=-0.5, max=0.3
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(
            run_id="r2", reason="rA",
            hypothetical_return_pct=v,
            timestamp=f"2025-01-02T{i:02d}:00:00+00:00", bar_index=i,
        )
        for i, v in enumerate([0.1, -0.2, 0.4])  # sum=0.3, n=3, min=-0.2, max=0.4
    ])
    out = aggregate_many([p1, p2])
    rs = out["global"]["cross_stats"]["hold_reason_outcome"]["rA"][
        "return_stats"
    ]
    # sum = -0.5 + 0.3 + 0.1 - 0.2 + 0.4 = 0.1
    # n   = 2 + 3 = 5
    # avg = 0.1 / 5 = 0.02
    # min = min(-0.5, -0.2) = -0.5
    # max = max(0.3, 0.4)   = 0.4
    assert rs["n_with_return"] == 5
    assert rs["sum_pct"] == pytest.approx(0.1)
    assert rs["avg_pct"] == pytest.approx(0.02)
    assert rs["min_pct"] == pytest.approx(-0.5)
    assert rs["max_pct"] == pytest.approx(0.4)


def test_run_id_unique_count_detects_multiple_distinct_run_ids(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="run_a",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(run_id="run_b",
                                timestamp="2025-01-02T00:00:00+00:00")
    ])
    out = aggregate_many([p1, p2])
    assert out["consistency_checks"]["run_id_unique_count"] == 2
    assert out["consistency_checks"]["warnings_total"] == 0


def test_run_id_unique_count_when_duplicated_is_warning_not_error(tmp_path):
    """Two files carrying the same run_id is a warning (not an error)."""
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="dup",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(run_id="dup",
                                timestamp="2025-01-02T00:00:00+00:00")
    ])
    out = aggregate_many([p1, p2])
    cc = out["consistency_checks"]
    assert cc["run_id_unique_count"] == 1
    assert cc["warnings_total"] >= 1
    assert any("duplicate run_id" in w for w in cc["warnings"])
    # Not an error: errors_total stays 0 (no malformed lines, no failed runs)
    assert cc["errors_total"] == 0


def test_malformed_line_in_one_run_propagates_to_global_errors(tmp_path):
    """A run with a malformed line still succeeds (per_run still appears),
    but its per-run errors_total folds into global errors_total."""
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    p2 = tmp_path / "r2" / "decision_traces.jsonl"
    p2.parent.mkdir()
    p2.write_text(
        json.dumps(_build_synthetic_record(
            run_id="r2", timestamp="2025-01-02T00:00:00+00:00"
        )) + "\n"
        + "{not json}\n",
        encoding="utf-8",
    )
    out = aggregate_many([p1, p2])
    cc = out["consistency_checks"]
    assert cc["errors_total"] >= 1
    # Both runs still appear in per_run (the bad line was skipped, the
    # good line was counted).
    assert len(out["per_run"]) == 2


def test_empty_run_recorded_in_failed_runs(tmp_path):
    """A completely empty file is treated as a failed run; remaining
    successful runs continue feeding the global aggregate."""
    from src.fx.decision_trace_stats import aggregate_many
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    empty_dir = tmp_path / "r2"
    empty_dir.mkdir()
    p2 = empty_dir / "decision_traces.jsonl"
    p2.write_text("", encoding="utf-8")
    out = aggregate_many([p1, p2])
    assert len(out["per_run"]) == 1
    cc = out["consistency_checks"]
    assert cc["n_runs_succeeded"] == 1
    assert cc["n_runs_failed"] == 1
    assert len(cc["failed_runs"]) == 1
    assert cc["failed_runs"][0]["path"] == str(p2)
    assert cc["errors_total"] >= 1


def test_all_runs_invalid_raises_value_error(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    p1 = tmp_path / "empty.jsonl"
    p1.write_text("", encoding="utf-8")
    p2 = tmp_path / "missing.jsonl"   # never written
    with pytest.raises(ValueError) as exc:
        aggregate_many([p1, p2])
    assert "no successful runs" in str(exc.value)


def test_aggregate_many_empty_paths_raises_value_error():
    from src.fx.decision_trace_stats import aggregate_many
    with pytest.raises(ValueError) as exc:
        aggregate_many([])
    assert "no input paths" in str(exc.value)


def test_metrics_summary_extracted_from_sibling_summary_json(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    p = _write_run_dir(
        tmp_path, "r1",
        records=[_build_synthetic_record(
            run_id="r1", timestamp="2025-01-01T00:00:00+00:00"
        )],
        summary_metrics={
            "n_trades": 19, "win_rate": 0.263, "profit_factor": 0.58,
            "total_return_pct": -1.589, "max_drawdown_pct": -2.18,
            "extra_field_we_dont_need": "ignored",
        },
    )
    out = aggregate_many([p])
    ms = out["per_run"][0]["metrics_summary"]
    assert ms is not None
    assert ms["n_trades"] == 19
    assert ms["win_rate"] == pytest.approx(0.263)
    assert ms["profit_factor"] == pytest.approx(0.58)
    assert ms["total_return_pct"] == pytest.approx(-1.589)
    assert ms["max_drawdown_pct"] == pytest.approx(-2.18)


def test_metrics_summary_null_when_summary_json_missing(tmp_path):
    from src.fx.decision_trace_stats import aggregate_many
    p = _write_run_dir(
        tmp_path, "r1",
        records=[_build_synthetic_record(
            run_id="r1", timestamp="2025-01-01T00:00:00+00:00"
        )],
        summary_metrics=None,
    )
    out = aggregate_many([p])
    assert out["per_run"][0]["metrics_summary"] is None


def test_existing_aggregate_stats_single_path_unchanged(tmp_path):
    """Sanity: aggregate_stats() (single-path API) still returns the same
    9 top-level keys after PR #8."""
    p = _write_jsonl(tmp_path, [_build_synthetic_record()])
    stats = aggregate_stats(p)
    expected = {
        "metadata", "action_distribution", "blocked_by_distribution",
        "rule_result_distribution", "gate_effect_distribution",
        "outcome_distribution", "entry_execution_distribution",
        "top_hold_reasons", "cross_stats", "consistency_checks",
    }
    assert expected <= set(stats.keys())


def test_cli_trace_stats_multi_compact_and_pretty(tmp_path):
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(run_id="r2",
                                timestamp="2025-01-02T00:00:00+00:00")
    ])
    cp_compact = _run_cli("trace-stats-multi", str(p1), str(p2))
    assert cp_compact.returncode == 0, cp_compact.stderr
    out_compact = cp_compact.stdout.strip()
    assert "\n" not in out_compact
    parsed = json.loads(out_compact)
    assert parsed["metadata"]["n_runs"] == 2
    assert len(parsed["per_run"]) == 2

    cp_pretty = _run_cli("trace-stats-multi", str(p1), str(p2), "--pretty")
    assert cp_pretty.returncode == 0, cp_pretty.stderr
    assert cp_pretty.stdout.count("\n") > 5
    parsed_p = json.loads(cp_pretty.stdout)
    assert parsed_p["metadata"]["n_runs"] == 2


def test_cli_exit_2_on_partial_run_failure(tmp_path):
    """One good + one empty: stats JSON is still printed, exit code is 2."""
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1",
                                timestamp="2025-01-01T00:00:00+00:00")
    ])
    empty_dir = tmp_path / "r2"
    empty_dir.mkdir()
    p2 = empty_dir / "decision_traces.jsonl"
    p2.write_text("", encoding="utf-8")
    cp = _run_cli("trace-stats-multi", str(p1), str(p2))
    assert cp.returncode == 2
    parsed = json.loads(cp.stdout)
    assert parsed["consistency_checks"]["n_runs_failed"] == 1
    assert len(parsed["consistency_checks"]["failed_runs"]) == 1


def test_cli_exit_2_on_all_runs_invalid(tmp_path):
    p1 = tmp_path / "empty.jsonl"
    p1.write_text("", encoding="utf-8")
    cp = _run_cli("trace-stats-multi", str(p1))
    assert cp.returncode == 2
    assert "trace-stats-multi" in cp.stderr


def test_aggregate_many_does_not_change_aggregate_stats_behaviour(tmp_path):
    """Calling aggregate_many() must not perturb a parallel single-path
    aggregate_stats() — they read independently."""
    from src.fx.decision_trace_stats import aggregate_many
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(
            reason="rA", gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            hypothetical_return_pct=-0.5,
            timestamp="2025-01-01T00:00:00+00:00", bar_index=0,
        ),
        _build_synthetic_record(
            reason="rA", gate_effect="COST_OPPORTUNITY", outcome="WIN_MISSED",
            hypothetical_return_pct=0.3,
            timestamp="2025-01-01T01:00:00+00:00", bar_index=1,
        ),
    ])
    single = aggregate_stats(p)
    multi = aggregate_many([p])
    # The multi report's per_run[0].cross_stats should mirror single.cross_stats
    assert (
        multi["per_run"][0]["cross_stats"]["hold_reason_outcome"]
        == single["cross_stats"]["hold_reason_outcome"]
    )
    # Re-call single — must be byte-identical (no global state mutation).
    single_again = aggregate_stats(p)
    assert single == single_again


def test_top_hold_reasons_pooled_includes_globally_ranked_reason(tmp_path):
    """Regression for a pooling bug: per-run top_hold_reasons is truncated
    to top-N by aggregate_stats(). If aggregate_many() folded those
    truncated lists into the global top, a reason that was #N+1 in every
    run but #2 globally would silently disappear.

    Build 3 runs each with:
      rA=10  (per-run #1)
      rB=8   (per-run #2)
      rGLOBAL=3   (per-run #3 — drops out of any per-run top-2)

    Globally: rA=30 / rB=24 / rGLOBAL=9. Asking for global top-3 must
    return all three; rGLOBAL must NOT be silently dropped.
    """
    from src.fx.decision_trace_stats import aggregate_many

    def make_run(run_id: str, day: int) -> Path:
        recs: list[dict] = []
        i = 0
        for _ in range(10):
            recs.append(_build_synthetic_record(
                run_id=run_id, reason="rA",
                timestamp=f"2025-01-{day:02d}T{i:02d}:00:00+00:00",
                bar_index=i,
            ))
            i += 1
        for _ in range(8):
            recs.append(_build_synthetic_record(
                run_id=run_id, reason="rB",
                timestamp=f"2025-01-{day:02d}T{i:02d}:00:00+00:00",
                bar_index=i,
            ))
            i += 1
        for _ in range(3):
            recs.append(_build_synthetic_record(
                run_id=run_id, reason="rGLOBAL",
                timestamp=f"2025-01-{day:02d}T{i:02d}:00:00+00:00",
                bar_index=i,
            ))
            i += 1
        return _write_run_dir(tmp_path, run_id, records=recs)

    paths = [make_run("r0", 1), make_run("r1", 2), make_run("r2", 3)]

    # Sanity: per-run top-2 truncates rGLOBAL out of every run.
    for p in paths:
        per_run = aggregate_stats(p, top_n_hold_reasons=2)
        per_run_reasons = {x["reason"] for x in per_run["top_hold_reasons"]}
        assert per_run_reasons == {"rA", "rB"}, (
            f"per-run setup wrong: {per_run_reasons!r}"
        )

    out = aggregate_many(paths, top_n_hold_reasons=3)
    global_pairs = [
        (entry["reason"], entry["count"])
        for entry in out["global"]["top_hold_reasons"]
    ]
    assert ("rA", 30) in global_pairs
    assert ("rB", 24) in global_pairs
    assert ("rGLOBAL", 9) in global_pairs, (
        f"BUG: rGLOBAL missing from global top despite being #3 globally; "
        f"got {global_pairs!r}. This regression means aggregate_many() is "
        f"summing per-run top-N lists instead of recomputing from the "
        f"full hold_reason_pool."
    )

    # The same reason must also be present in cross_stats.hold_reason_outcome
    # (no truncation applies there at all).
    hro = out["global"]["cross_stats"]["hold_reason_outcome"]
    assert hro["rGLOBAL"]["n"] == 9


def test_top_hold_reasons_global_is_derived_from_hold_reason_pool(tmp_path):
    """Top hold reasons must equal `most_common(N)` over the full
    hold_reason_pool, NOT over per-run truncated lists."""
    from src.fx.decision_trace_stats import aggregate_many

    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(run_id="r1", reason="rX",
                                timestamp="2025-01-01T00:00:00+00:00",
                                bar_index=0),
        _build_synthetic_record(run_id="r1", reason="rX",
                                timestamp="2025-01-01T01:00:00+00:00",
                                bar_index=1),
        _build_synthetic_record(run_id="r1", reason="rY",
                                timestamp="2025-01-01T02:00:00+00:00",
                                bar_index=2),
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(run_id="r2", reason="rY",
                                timestamp="2025-01-02T00:00:00+00:00",
                                bar_index=0),
        _build_synthetic_record(run_id="r2", reason="rY",
                                timestamp="2025-01-02T01:00:00+00:00",
                                bar_index=1),
        _build_synthetic_record(run_id="r2", reason="rY",
                                timestamp="2025-01-02T02:00:00+00:00",
                                bar_index=2),
    ])
    out = aggregate_many([p1, p2])
    hro = out["global"]["cross_stats"]["hold_reason_outcome"]
    assert hro["rX"]["n"] == 2 and hro["rY"]["n"] == 4
    # global top must reflect hro counts exactly (rY=4 then rX=2).
    pairs = [(e["reason"], e["count"])
             for e in out["global"]["top_hold_reasons"]]
    assert pairs[0] == ("rY", 4)
    assert pairs[1] == ("rX", 2)


# ─── PR #10: cross_stats.blocked_by_outcome ──────────────────────────────


def test_blocked_by_outcome_top_level_key_present(tmp_path):
    """`cross_stats.blocked_by_outcome` must always be present (possibly
    with only a `no_block` row when no Risk Gate block fired)."""
    p = _write_jsonl(tmp_path, [_build_synthetic_record()])
    stats = aggregate_stats(p)
    cs = stats["cross_stats"]
    assert "blocked_by_outcome" in cs
    assert isinstance(cs["blocked_by_outcome"], dict)


def test_blocked_by_empty_buckets_into_no_block(tmp_path):
    """A trace with `decision.blocked_by=[]` must contribute to the
    synthetic key `no_block`, not to any code key."""
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(blocked_by=()),
        _build_synthetic_record(blocked_by=(),
                                timestamp="2025-01-01T01:00:00+00:00",
                                bar_index=1),
    ])
    bbo = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"]
    assert "no_block" in bbo
    assert bbo["no_block"]["n"] == 2
    # No real-code keys.
    assert all(k == "no_block" for k in bbo.keys())


def test_blocked_by_event_high_buckets_into_event_high(tmp_path):
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(
            blocked_by=("event_high",),
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
        ),
    ])
    bbo = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"]
    assert "event_high" in bbo
    assert bbo["event_high"]["n"] == 1
    assert bbo["event_high"]["gate_effect"] == {"PROTECTED": 1}
    assert bbo["event_high"]["outcome_if_technical_action_taken"] == {
        "LOSS_AVOIDED": 1
    }
    # `no_block` row should NOT exist for this single trace.
    assert "no_block" not in bbo


def test_blocked_by_multiple_codes_contributes_to_each_key(tmp_path):
    """One trace with two codes must contribute +1 to each code's row.
    Each row's `n` reflects per-code counts (additive semantics), so
    sum over rows can exceed n_traces — that's by design."""
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(
            blocked_by=("event_high", "spread_abnormal"),
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            hypothetical_return_pct=-0.3,
        ),
    ])
    bbo = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"]
    assert bbo["event_high"]["n"] == 1
    assert bbo["spread_abnormal"]["n"] == 1
    assert bbo["event_high"]["gate_effect"] == {"PROTECTED": 1}
    assert bbo["spread_abnormal"]["gate_effect"] == {"PROTECTED": 1}
    # return value flows into BOTH rows
    assert bbo["event_high"]["return_stats"]["sum_pct"] == pytest.approx(-0.3)
    assert bbo["spread_abnormal"]["return_stats"]["sum_pct"] == pytest.approx(
        -0.3
    )


def test_blocked_by_outcome_gate_effect_buckets(tmp_path):
    """Multiple gate_effect values for the same blocked_by code must
    aggregate cleanly into PROTECTED / COST_OPPORTUNITY / NO_CHANGE."""
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(blocked_by=("event_high",),
                                gate_effect="PROTECTED",
                                outcome="LOSS_AVOIDED",
                                bar_index=0),
        _build_synthetic_record(blocked_by=("event_high",),
                                gate_effect="COST_OPPORTUNITY",
                                outcome="WIN_MISSED",
                                timestamp="2025-01-01T01:00:00+00:00",
                                bar_index=1),
        _build_synthetic_record(blocked_by=("event_high",),
                                gate_effect="NO_CHANGE",
                                outcome="N/A",
                                timestamp="2025-01-01T02:00:00+00:00",
                                bar_index=2),
    ])
    row = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"]["event_high"]
    assert row["n"] == 3
    assert row["gate_effect"] == {
        "PROTECTED": 1, "COST_OPPORTUNITY": 1, "NO_CHANGE": 1,
    }
    assert row["outcome_if_technical_action_taken"] == {
        "LOSS_AVOIDED": 1, "WIN_MISSED": 1, "N/A": 1,
    }


def test_blocked_by_outcome_technical_only_action_buckets(tmp_path):
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(blocked_by=("event_high",),
                                technical_action="BUY", bar_index=0),
        _build_synthetic_record(blocked_by=("event_high",),
                                technical_action="SELL",
                                timestamp="2025-01-01T01:00:00+00:00",
                                bar_index=1),
        _build_synthetic_record(blocked_by=("event_high",),
                                technical_action="HOLD",
                                timestamp="2025-01-01T02:00:00+00:00",
                                bar_index=2),
    ])
    row = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"]["event_high"]
    assert row["technical_only_action"] == {
        "BUY": 1, "SELL": 1, "HOLD": 1,
    }


def test_blocked_by_outcome_return_stats(tmp_path):
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(blocked_by=("event_high",),
                                hypothetical_return_pct=v,
                                timestamp=f"2025-01-01T{i:02d}:00:00+00:00",
                                bar_index=i)
        for i, v in enumerate([-0.5, 0.3, 0.1, -0.2, 0.4])
    ])
    rs = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"][
        "event_high"
    ]["return_stats"]
    assert rs["n_with_return"] == 5
    assert rs["sum_pct"] == pytest.approx(0.1)
    assert rs["avg_pct"] == pytest.approx(0.02)
    assert rs["min_pct"] == pytest.approx(-0.5)
    assert rs["max_pct"] == pytest.approx(0.4)


def test_blocked_by_outcome_return_stats_null_when_no_returns(tmp_path):
    p = _write_jsonl(tmp_path, [
        _build_synthetic_record(blocked_by=("event_high",),
                                hypothetical_return_pct=None,
                                bar_index=i,
                                timestamp=f"2025-01-01T{i:02d}:00:00+00:00")
        for i in range(3)
    ])
    rs = aggregate_stats(p)["cross_stats"]["blocked_by_outcome"][
        "event_high"
    ]["return_stats"]
    assert rs["n_with_return"] == 0
    assert rs["avg_pct"] is None
    assert rs["sum_pct"] is None
    assert rs["min_pct"] is None
    assert rs["max_pct"] is None


def test_blocked_by_outcome_with_malformed_line_still_aggregates(tmp_path):
    """A malformed line must NOT block valid lines from updating
    blocked_by_outcome rows."""
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        json.dumps(_build_synthetic_record(
            blocked_by=("event_high",),
            gate_effect="PROTECTED",
            outcome="LOSS_AVOIDED",
            bar_index=0,
        )) + "\n"
        "{not json}\n"
        + json.dumps(_build_synthetic_record(
            blocked_by=("event_high",),
            gate_effect="COST_OPPORTUNITY",
            outcome="WIN_MISSED",
            bar_index=2,
            timestamp="2025-01-01T02:00:00+00:00",
        )) + "\n",
        encoding="utf-8",
    )
    stats = aggregate_stats(p)
    assert stats["consistency_checks"]["errors_total"] >= 1
    row = stats["cross_stats"]["blocked_by_outcome"]["event_high"]
    assert row["n"] == 2
    assert row["gate_effect"] == {"PROTECTED": 1, "COST_OPPORTUNITY": 1}


def test_aggregate_many_pools_blocked_by_outcome(tmp_path):
    """run1 + run2 with overlapping codes must collapse into one global
    row each, with summed counters and pooled return_stats."""
    p1 = _write_run_dir(tmp_path, "r1", records=[
        _build_synthetic_record(
            run_id="r1", blocked_by=("event_high",),
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            hypothetical_return_pct=-0.5,
            timestamp="2025-01-01T00:00:00+00:00", bar_index=0,
        ),
        _build_synthetic_record(
            run_id="r1", blocked_by=("event_high",),
            gate_effect="COST_OPPORTUNITY", outcome="WIN_MISSED",
            hypothetical_return_pct=0.3,
            timestamp="2025-01-01T01:00:00+00:00", bar_index=1,
        ),
    ])
    p2 = _write_run_dir(tmp_path, "r2", records=[
        _build_synthetic_record(
            run_id="r2", blocked_by=("event_high",),
            gate_effect="PROTECTED", outcome="LOSS_AVOIDED",
            hypothetical_return_pct=-0.2,
            timestamp="2025-01-02T00:00:00+00:00", bar_index=0,
        ),
        _build_synthetic_record(
            run_id="r2", blocked_by=("spread_abnormal",),
            gate_effect="NO_CHANGE", outcome="N/A",
            hypothetical_return_pct=None,
            timestamp="2025-01-02T01:00:00+00:00", bar_index=1,
        ),
    ])
    from src.fx.decision_trace_stats import aggregate_many
    out = aggregate_many([p1, p2])
    g_bbo = out["global"]["cross_stats"]["blocked_by_outcome"]
    # event_high in r1 (n=2) + r2 (n=1)  =>  global n=3
    assert g_bbo["event_high"]["n"] == 3
    assert g_bbo["event_high"]["gate_effect"] == {
        "PROTECTED": 2, "COST_OPPORTUNITY": 1
    }
    # spread_abnormal only in r2  =>  global n=1
    assert g_bbo["spread_abnormal"]["n"] == 1
    # return_stats pooling for event_high: -0.5 + 0.3 + (-0.2) = -0.4 / n=3
    rs = g_bbo["event_high"]["return_stats"]
    assert rs["n_with_return"] == 3
    assert rs["sum_pct"] == pytest.approx(-0.4)
    assert rs["avg_pct"] == pytest.approx(-0.4 / 3)
    assert rs["min_pct"] == pytest.approx(-0.5)
    assert rs["max_pct"] == pytest.approx(0.3)


def test_existing_cross_stats_keys_unchanged_by_blocked_by_outcome(tmp_path):
    """PR #10 + PR-A: cross_stats includes the original three sections,
    PR #10's blocked_by_outcome, and PR-A's long-term/macro crosses."""
    p = _write_jsonl(tmp_path, [_build_synthetic_record()])
    cs = aggregate_stats(p)["cross_stats"]
    assert {
        "hold_reason_outcome",
        "gate_effect_by_technical_action",
        "final_action_by_outcome",
        "blocked_by_outcome",        # PR #10
        # PR-A long-term trend + macro context crosses
        "daily_trend_outcome",
        "weekly_trend_outcome",
        "monthly_trend_outcome",
        "close_vs_sma_200d_outcome",
        "weekly_return_outcome",
        "monthly_return_outcome",
        "dxy_trend_outcome",
        "us10y_trend_outcome",
        "yield_spread_outcome",
        "vix_regime_outcome",
        "long_term_macro_outcome",
    } == set(cs.keys())


def test_blocked_by_outcome_real_export_pipeline(tmp_path):
    """End-to-end: run_engine_backtest → export_run → aggregate_stats →
    cross_stats.blocked_by_outcome carries (at least) the no_block row."""
    df = _ohlcv(200, seed=11)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    paths = export_run(res, out_dir=tmp_path / "real")
    bbo = aggregate_stats(paths["decision_traces"])["cross_stats"][
        "blocked_by_outcome"
    ]
    # Without seeded events, every bar's blocked_by is empty -> no_block.
    assert "no_block" in bbo
    assert bbo["no_block"]["n"] == len(res.decision_traces)
