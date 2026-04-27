"""Tests for decision_trace_io.export_run — JSONL/JSON disk export."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace import TRACE_SCHEMA_VERSION
from src.fx.decision_trace_io import export_run


def _ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def _run(tmp_path, *, seed: int = 1, n: int = 200, capture_traces: bool = True):
    df = _ohlcv(n, seed=seed)
    return run_engine_backtest(
        df, "USDJPY=X", interval="1h", warmup=60,
        capture_traces=capture_traces,
    )


# ─── Required tests (8) ─────────────────────────────────────────────────────


def test_export_creates_decision_traces_jsonl(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run01"
    paths = export_run(res, out_dir=out_dir)
    f = paths["decision_traces"]
    assert f.exists()
    assert f.stat().st_size > 0
    assert f.name == "decision_traces.jsonl"


def test_jsonl_line_count_equals_decision_traces(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run02"
    paths = export_run(res, out_dir=out_dir)
    with paths["decision_traces"].open("r", encoding="utf-8") as fp:
        lines = [ln for ln in fp.read().splitlines() if ln]
    assert len(lines) == len(res.decision_traces)


def test_jsonl_first_line_parses_as_json_with_required_keys(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run03"
    paths = export_run(res, out_dir=out_dir)
    with paths["decision_traces"].open("r", encoding="utf-8") as fp:
        first = fp.readline()
    rec = json.loads(first)
    required = {
        "run_id", "trace_schema_version", "bar_id", "timestamp",
        "symbol", "timeframe", "bar_index",
        "market", "technical", "waveform", "higher_timeframe",
        "fundamental", "execution_assumption", "execution_trace",
        "rule_checks", "decision",
    }
    assert required <= set(rec.keys())
    assert rec["trace_schema_version"] == TRACE_SCHEMA_VERSION


def test_export_creates_run_metadata_json(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run04"
    paths = export_run(res, out_dir=out_dir)
    rm_path = paths["run_metadata"]
    assert rm_path.exists() and rm_path.name == "run_metadata.json"
    rm = json.loads(rm_path.read_text(encoding="utf-8"))
    for key in ("run_id", "symbol", "timeframe", "trace_schema_version",
                "strategy_config_hash", "data_snapshot_hash"):
        assert key in rm


def test_export_creates_summary_json_with_required_keys(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run05"
    paths = export_run(res, out_dir=out_dir)
    s_path = paths["summary"]
    assert s_path.exists() and s_path.name == "summary.json"
    s = json.loads(s_path.read_text(encoding="utf-8"))
    for key in ("run_id", "symbol", "interval", "bars_processed",
                "n_traces", "n_trades", "metrics", "trace_schema_version",
                "created_at", "output_files", "export_gzip"):
        assert key in s, f"missing summary key: {key}"
    assert s["n_traces"] == len(res.decision_traces)
    assert s["n_trades"] == len(res.trades)
    assert s["bars_processed"] == res.bars_processed
    assert s["export_gzip"] is False


def test_run_id_consistent_across_metadata_traces_summary(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run06"
    paths = export_run(res, out_dir=out_dir)
    rm = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    s = json.loads(paths["summary"].read_text(encoding="utf-8"))
    rm_run_id = rm["run_id"]
    s_run_id = s["run_id"]
    assert rm_run_id == s_run_id
    with paths["decision_traces"].open("r", encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            assert json.loads(ln)["run_id"] == rm_run_id


def test_overwrite_false_raises_on_any_existing_target(tmp_path):
    """Even one existing target file must abort the entire export."""
    res = _run(tmp_path)
    out_dir = tmp_path / "run07"
    out_dir.mkdir()
    # Plant a single stale file; the other two are absent. Strict mode must
    # still refuse the export — partial overwrite is forbidden.
    (out_dir / "summary.json").write_text("{\"stale\": true}", encoding="utf-8")

    with pytest.raises(FileExistsError) as exc_info:
        export_run(res, out_dir=out_dir)
    assert "summary.json" in str(exc_info.value)

    # Sanity: the other two were not created (all-or-nothing).
    assert not (out_dir / "decision_traces.jsonl").exists()
    assert not (out_dir / "run_metadata.json").exists()
    # Stale file untouched.
    assert json.loads((out_dir / "summary.json").read_text()) == {"stale": True}


def test_export_does_not_change_trades_or_metrics(tmp_path):
    """Exporting must be observation-only — trades / metrics identical with
    or without the export step."""
    df = _ohlcv(300, seed=11)
    res_a = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                                capture_traces=True)
    res_b = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                                capture_traces=True)
    out_dir = tmp_path / "run08"
    export_run(res_b, out_dir=out_dir)

    def _summary(r):
        return [
            (t.entry_ts, t.side, t.exit_ts, t.exit_reason,
             round(t.return_pct, 6))
            for t in r.trades
        ], dict(r.hold_reasons), r.bars_processed

    assert _summary(res_a) == _summary(res_b)
    # metrics() includes synthetic_execution metadata — both must match
    assert res_a.metrics() == res_b.metrics()


# ─── Added tests (4) ────────────────────────────────────────────────────────


def test_export_raises_when_capture_traces_false(tmp_path):
    res = _run(tmp_path, capture_traces=False)
    out_dir = tmp_path / "run09"
    with pytest.raises(ValueError) as exc_info:
        export_run(res, out_dir=out_dir)
    assert "run_metadata is None" in str(exc_info.value)
    # No files created when validation fails.
    assert not any(out_dir.iterdir()) if out_dir.exists() else True


def test_export_raises_on_empty_trace_list(tmp_path):
    """capture_traces=True but warmup >= len(df) -> 0 traces produced."""
    df = _ohlcv(60, seed=5)
    res = run_engine_backtest(df, "USDJPY=X", interval="1h", warmup=60,
                              capture_traces=True)
    assert len(res.decision_traces) == 0
    out_dir = tmp_path / "run10"
    with pytest.raises(ValueError) as exc_info:
        export_run(res, out_dir=out_dir)
    assert "decision_traces is empty" in str(exc_info.value)


def test_summary_output_files_match_actual_filenames(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run11"
    paths = export_run(res, out_dir=out_dir)
    s = json.loads(paths["summary"].read_text(encoding="utf-8"))
    of = s["output_files"]
    # Relative names exactly match what's on disk in out_dir.
    for key in ("run_metadata", "decision_traces", "summary"):
        assert (out_dir / of[key]).exists(), (
            f"summary.output_files[{key}]={of[key]!r} does not exist in {out_dir}"
        )
    assert of["run_metadata"] == "run_metadata.json"
    assert of["decision_traces"] == "decision_traces.jsonl"
    assert of["summary"] == "summary.json"


def test_every_jsonl_line_carries_run_id(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run12"
    paths = export_run(res, out_dir=out_dir)
    expected_run_id = res.run_metadata.run_id
    with paths["decision_traces"].open("r", encoding="utf-8") as fp:
        for i, ln in enumerate(fp):
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            assert "run_id" in rec, f"line {i}: missing run_id"
            assert rec["run_id"] == expected_run_id, (
                f"line {i}: run_id mismatch {rec['run_id']!r} "
                f"!= {expected_run_id!r}"
            )


# ─── Optional: --gzip flag (small, in-scope per plan) ───────────────────────


def test_gzip_writes_jsonl_gz_and_is_readable(tmp_path):
    res = _run(tmp_path)
    out_dir = tmp_path / "run13"
    paths = export_run(res, out_dir=out_dir, gzip=True)
    assert paths["decision_traces"].name == "decision_traces.jsonl.gz"
    assert paths["decision_traces"].exists()
    # Round-trip: gzip-readable, line count matches, run_id present.
    with gzip.open(paths["decision_traces"], "rt", encoding="utf-8") as fp:
        lines = [ln for ln in fp.read().splitlines() if ln]
    assert len(lines) == len(res.decision_traces)
    rec = json.loads(lines[0])
    assert rec["run_id"] == res.run_metadata.run_id
    # summary reflects the chosen filename and gzip flag.
    s = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert s["export_gzip"] is True
    assert s["output_files"]["decision_traces"] == "decision_traces.jsonl.gz"
