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
        cwd="/home/user/test",
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
