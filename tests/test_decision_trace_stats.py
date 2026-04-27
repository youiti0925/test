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
