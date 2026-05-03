"""Tests for technical_confluence_summary aggregator (PR-C2).

Two layers:
  A. Direct unit tests of `compute_technical_confluence_summary` with
     hand-built trace inputs so the aggregation rules are pinned.
  B. End-to-end via `run_engine_backtest` + `_build_summary` so the
     summary actually lands in summary.json shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.decision_trace_io import _build_summary, export_run
from src.fx.technical_confluence_summary import (
    SCHEMA_VERSION,
    _bin_score,
    compute_technical_confluence_summary,
)


# ───────────────────────── synthetic trace shims ─────────────────────────


class _FakeSlice:
    def __init__(self, payload: dict | None) -> None:
        self._payload = payload

    def to_dict(self) -> dict | None:
        return self._payload


class _FakeTrace:
    def __init__(self, payload: dict | None) -> None:
        self.technical_confluence = _FakeSlice(payload) if payload else None


def _payload(
    *,
    label: str = "WEAK_BUY_SETUP",
    score: float = 0.0,
    bullish: list[str] | None = None,
    bearish: list[str] | None = None,
    avoid: list[str] | None = None,
    near_support: bool = False,
    near_resistance: bool = False,
    candle: dict | None = None,
    indicator: dict | None = None,
    risk: dict | None = None,
) -> dict:
    return {
        "policy_version": "technical_confluence_v1",
        "market_regime": "RANGE",
        "dow_structure": {},
        "support_resistance": {
            "near_support": near_support,
            "near_resistance": near_resistance,
        },
        "candlestick_signal": candle or {},
        "chart_pattern": {},
        "indicator_context": indicator or {},
        "risk_plan_obs": risk or {},
        "vote_breakdown": {},
        "final_confluence": {
            "label": label,
            "score": score,
            "bullish_reasons": bullish or [],
            "bearish_reasons": bearish or [],
            "avoid_reasons": avoid or [],
        },
    }


# ───────────────────────── A. Direct unit tests ──────────────────────────


def test_summary_schema_version():
    s = compute_technical_confluence_summary(traces=[])
    assert s["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "technical_confluence_summary_v1"


def test_summary_empty_input_returns_full_shape():
    """Empty input must still emit every key — shape-stable summary."""
    s = compute_technical_confluence_summary(traces=[])
    expected = {
        "schema_version", "n_traces_with_confluence",
        "label_distribution", "score_distribution",
        "top_bullish_reasons", "top_bearish_reasons", "top_avoid_reasons",
        "near_support_count", "near_resistance_count",
        "candlestick_signal_counts", "indicator_context_counts",
        "risk_plan_obs_summary",
    }
    assert set(s.keys()) == expected
    assert s["n_traces_with_confluence"] == 0
    assert all(v == 0 for v in s["label_distribution"].values())
    assert s["top_bullish_reasons"] == []


def test_label_distribution_counts_each_label():
    traces = [
        _FakeTrace(_payload(label="STRONG_BUY_SETUP")),
        _FakeTrace(_payload(label="STRONG_BUY_SETUP")),
        _FakeTrace(_payload(label="WEAK_SELL_SETUP")),
        _FakeTrace(_payload(label="AVOID_TRADE")),
        _FakeTrace(_payload(label="UNKNOWN")),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    assert s["label_distribution"]["STRONG_BUY_SETUP"] == 2
    assert s["label_distribution"]["WEAK_SELL_SETUP"] == 1
    assert s["label_distribution"]["AVOID_TRADE"] == 1
    assert s["label_distribution"]["UNKNOWN"] == 1
    # Other labels remain at 0 (shape stable)
    assert s["label_distribution"]["NO_TRADE"] == 0
    assert s["n_traces_with_confluence"] == 5


def test_score_distribution_bins_correctly():
    """Boundaries: <-0.5, [-0.5, 0), [0, 0.5], >0.5."""
    assert _bin_score(-0.6) == "<-0.5"
    assert _bin_score(-0.5) == "[-0.5, 0)"
    assert _bin_score(-0.01) == "[-0.5, 0)"
    assert _bin_score(0.0) == "[0, 0.5]"
    assert _bin_score(0.4) == "[0, 0.5]"
    assert _bin_score(0.5) == "[0, 0.5]"
    assert _bin_score(0.51) == ">0.5"

    traces = [
        _FakeTrace(_payload(score=-1.0)),
        _FakeTrace(_payload(score=-0.4)),
        _FakeTrace(_payload(score=0.4)),
        _FakeTrace(_payload(score=0.9)),
        _FakeTrace(_payload(score=0.9)),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    assert s["score_distribution"]["<-0.5"] == 1
    assert s["score_distribution"]["[-0.5, 0)"] == 1
    assert s["score_distribution"]["[0, 0.5]"] == 1
    assert s["score_distribution"][">0.5"] == 2


def test_top_reasons_aggregated_by_frequency():
    traces = [
        _FakeTrace(_payload(bullish=["near_support", "ma_trend_buy"])),
        _FakeTrace(_payload(bullish=["near_support"])),
        _FakeTrace(_payload(bearish=["bearish_pinbar"])),
        _FakeTrace(_payload(avoid=["fake_breakout", "fake_breakout"])),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    reasons = {r["reason"]: r["count"] for r in s["top_bullish_reasons"]}
    assert reasons["near_support"] == 2
    assert reasons["ma_trend_buy"] == 1
    assert s["top_bearish_reasons"][0] == {"reason": "bearish_pinbar", "count": 1}
    assert s["top_avoid_reasons"][0] == {"reason": "fake_breakout", "count": 2}


def test_near_level_counts():
    traces = [
        _FakeTrace(_payload(near_support=True)),
        _FakeTrace(_payload(near_support=True)),
        _FakeTrace(_payload(near_resistance=True)),
        _FakeTrace(_payload()),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    assert s["near_support_count"] == 2
    assert s["near_resistance_count"] == 1


def test_candlestick_signal_counts():
    traces = [
        _FakeTrace(_payload(candle={"bullish_pinbar": True})),
        _FakeTrace(_payload(candle={"bullish_pinbar": True, "bullish_engulfing": True})),
        _FakeTrace(_payload(candle={"harami": True})),
        _FakeTrace(_payload()),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    cc = s["candlestick_signal_counts"]
    assert cc["bullish_pinbar"] == 2
    assert cc["bullish_engulfing"] == 1
    assert cc["harami"] == 1
    assert cc["bearish_pinbar"] == 0


def test_indicator_context_counts_partition():
    traces = [
        _FakeTrace(_payload(indicator={
            "rsi_trend_danger": True, "bb_squeeze": True,
            "ma_trend_support": "BUY",
        })),
        _FakeTrace(_payload(indicator={
            "rsi_trend_danger": False, "bb_expansion": True,
            "ma_trend_support": "SELL",
        })),
        _FakeTrace(_payload(indicator={
            "bb_band_walk": True, "ma_trend_support": "NEUTRAL",
        })),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    ic = s["indicator_context_counts"]
    assert ic["rsi_trend_danger_true"] == 1
    assert ic["bb_squeeze_true"] == 1
    assert ic["bb_expansion_true"] == 1
    assert ic["bb_band_walk_true"] == 1
    assert ic["ma_trend_support_buy"] == 1
    assert ic["ma_trend_support_sell"] == 1
    assert ic["ma_trend_support_neutral"] == 1


def test_risk_plan_obs_summary_averaging():
    traces = [
        _FakeTrace(_payload(risk={
            "invalidation_clear": True,
            "structure_stop_price": 99.0,
            "structure_stop_distance_atr": 1.0,
            "rr_structure_based": 3.0,
        })),
        _FakeTrace(_payload(risk={
            "invalidation_clear": True,
            "structure_stop_price": 98.0,
            "structure_stop_distance_atr": 2.0,
            "rr_structure_based": 1.5,
        })),
        _FakeTrace(_payload(risk={
            "invalidation_clear": False,
            "structure_stop_price": None,
            "structure_stop_distance_atr": None,
        })),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    rp = s["risk_plan_obs_summary"]
    assert rp["invalidation_clear_true_count"] == 2
    assert rp["structure_stop_present_count"] == 2
    assert rp["structure_stop_distance_atr_avg"] == pytest.approx(1.5)
    assert rp["rr_structure_based_avg"] == pytest.approx(2.25)


def test_traces_with_no_confluence_are_skipped():
    """A trace whose `technical_confluence` is None must not contribute
    to `n_traces_with_confluence` and must not crash the aggregator."""
    traces = [
        _FakeTrace(None),
        _FakeTrace(_payload(label="STRONG_BUY_SETUP")),
        _FakeTrace(None),
    ]
    s = compute_technical_confluence_summary(traces=traces)
    assert s["n_traces_with_confluence"] == 1
    assert s["label_distribution"]["STRONG_BUY_SETUP"] == 1


# ───────────────────────── B. End-to-end via _build_summary ──────────────


def _ohlcv(n: int = 250, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_summary_appears_in_summary_json_via_export(tmp_path: Path):
    """End-to-end: run_engine_backtest → export_run → summary.json
    must contain `technical_confluence_summary` with our schema."""
    res = run_engine_backtest(_ohlcv(), "X", interval="1h", warmup=60)
    out = export_run(result=res, out_dir=tmp_path, overwrite=True)
    summary = json.loads(out["summary"].read_text())
    assert "technical_confluence_summary" in summary
    block = summary["technical_confluence_summary"]
    assert block["schema_version"] == SCHEMA_VERSION
    # Should have populated the labels seen in the run
    total = sum(block["label_distribution"].values())
    assert total == block["n_traces_with_confluence"]
    assert total > 0


def test_summary_failure_is_recorded_as_technical_confluence_error(monkeypatch):
    """If the aggregator raises, the summary must capture the error
    string and still produce a usable summary (fail-soft)."""
    res = run_engine_backtest(_ohlcv(150), "X", interval="1h", warmup=60)

    def _explode(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        "src.fx.technical_confluence_summary."
        "compute_technical_confluence_summary",
        _explode,
    )
    summary = _build_summary(res, output_files={}, gzip_enabled=False)
    assert "technical_confluence_summary" not in summary
    assert "technical_confluence_error" in summary
    assert "RuntimeError" in summary["technical_confluence_error"]
    # The pre-existing summary block must still be intact
    assert "metrics" in summary
    assert "n_traces" in summary
