"""Tests for royal_road_v2_summary aggregator + fail-soft wiring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace_io import _build_summary, export_run
from src.fx.royal_road_v2_summary import (
    SCHEMA_VERSION,
    compute_royal_road_v2_summary,
)


def _ohlcv(n: int = 250, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_v2_summary_schema_version():
    s = compute_royal_road_v2_summary(traces=[])
    assert s["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "royal_road_v2_summary_v1"


def test_v2_summary_empty_input_returns_full_shape():
    s = compute_royal_road_v2_summary(traces=[])
    expected_keys = {
        "schema_version", "n_traces_with_v2",
        "action_distribution", "label_distribution", "mode_distribution",
        "block_reasons_top", "cautions_top",
        "evidence_axes_counts",
        "support_resistance_strength_distribution",
        "trendline_signal_distribution",
        "chart_pattern_distribution",
        "lower_tf_trigger_distribution",
        "macro_alignment_distribution",
        "macro_event_tone_distribution",
        "stop_mode_distribution",
        "selected_stop_distance_atr_distribution",
        "rr_selected_distribution",
    }
    assert set(s.keys()) == expected_keys
    assert s["n_traces_with_v2"] == 0


def test_v2_summary_appears_in_summary_json_when_v2_active(tmp_path: Path):
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    out = export_run(result=res, out_dir=tmp_path, overwrite=True)
    summary = json.loads(out["summary"].read_text())
    assert "royal_road_v2_summary" in summary
    block = summary["royal_road_v2_summary"]
    assert block["schema_version"] == SCHEMA_VERSION
    assert block["n_traces_with_v2"] > 0
    # action distribution must sum to n_traces_with_v2.
    assert (
        sum(block["action_distribution"].values())
        == block["n_traces_with_v2"]
    )


def test_v2_summary_present_with_zero_count_when_default_profile(tmp_path: Path):
    """default current_runtime: no v2 traces; the summary block should
    still exist with n_traces_with_v2 == 0 (shape stable)."""
    df = _ohlcv()
    res = run_engine_backtest(df, "EURUSD=X", interval="1h", warmup=60)
    out = export_run(result=res, out_dir=tmp_path, overwrite=True)
    summary = json.loads(out["summary"].read_text())
    assert "royal_road_v2_summary" in summary
    assert summary["royal_road_v2_summary"]["n_traces_with_v2"] == 0


def test_v2_summary_failure_recorded_as_error(monkeypatch):
    """Aggregator failure must NOT break the rest of summary.json
    (fail-soft). The error string must land in
    `royal_road_v2_summary_error` and the metrics block must still be
    present."""
    df = _ohlcv(150)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )

    def _explode(**kwargs):
        raise RuntimeError("synthetic v2 summary failure")

    monkeypatch.setattr(
        "src.fx.royal_road_v2_summary.compute_royal_road_v2_summary",
        _explode,
    )
    summary = _build_summary(res, output_files={}, gzip_enabled=False)
    assert "royal_road_v2_summary" not in summary
    assert "royal_road_v2_summary_error" in summary
    assert "RuntimeError" in summary["royal_road_v2_summary_error"]
    # Pre-existing keys remain.
    assert "metrics" in summary
    assert "n_traces" in summary


def test_v2_summary_block_reasons_sorted():
    """When traces have repeated block reasons, top counts are
    descending."""
    from src.fx.royal_road_v2_summary import _v2_dict
    fake_traces = []

    class _Sl:
        def __init__(self, payload): self._p = payload
        def to_dict(self): return self._p

    class _Tr:
        def __init__(self, payload):
            self.royal_road_decision_v2 = _Sl(payload)
            self.technical_confluence = None

    base = {
        "action": "HOLD", "mode": "balanced",
        "block_reasons": [], "cautions": [],
        "evidence_axes": {}, "evidence_axes_count": {},
        "support_resistance_v2": {},
        "trendline_context": {},
        "chart_pattern_v2": {},
        "lower_tf_trigger": {"available": False},
        "macro_alignment": {"currency_bias": "UNKNOWN", "event_tone": "UNKNOWN"},
        "structure_stop_plan": {"chosen_mode": "atr"},
    }
    for _ in range(10):
        p = dict(base, block_reasons=["near_strong_resistance_for_buy"])
        fake_traces.append(_Tr(p))
    for _ in range(3):
        p = dict(base, block_reasons=["htf_counter_trend_for_buy"])
        fake_traces.append(_Tr(p))
    for _ in range(1):
        p = dict(base, block_reasons=["sr_fake_breakout"])
        fake_traces.append(_Tr(p))

    s = compute_royal_road_v2_summary(traces=fake_traces)
    top = s["block_reasons_top"]
    counts = [r["count"] for r in top]
    assert counts == sorted(counts, reverse=True)
    assert top[0]["reason"] == "near_strong_resistance_for_buy"
    assert top[0]["count"] == 10
