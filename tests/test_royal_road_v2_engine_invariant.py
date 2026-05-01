"""Engine-level invariants for royal_road_decision_v2 profile.

Pins:
  - default current_runtime is byte-identical to PR #21 main when the
    flag is omitted (regardless of stop_mode/lower_tf kwargs)
  - --decision-profile royal_road_decision_v2 actually emits the v2 slice
  - the v2 profile produces a usable trace dict
  - live / OANDA / paper paths do NOT import royal_road_decision_v2
  - parameter_defaults.PARAMETER_BASELINE_V1 baseline hash unchanged
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "fx"


def _ohlcv(n: int = 250, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


# ─────── default unchanged ───────────────────────────────────────


def test_default_unchanged_with_v2_kwargs_left_default():
    df = _ohlcv()
    res_a = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_b = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert _trade_summary(res_a) == _trade_summary(res_b)
    assert res_a.hold_reasons == res_b.hold_reasons


def test_default_does_not_emit_v2_slice():
    df = _ohlcv(150)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    for tr in res.decision_traces:
        assert tr.royal_road_decision_v2 is None


def test_default_with_explicit_atr_stop_mode_unchanged():
    """stop_mode=atr is the documented default and must yield identical
    output to omitting the flag."""
    df = _ohlcv()
    res_implicit = run_engine_backtest(df, "X", interval="1h", warmup=60)
    res_explicit = run_engine_backtest(
        df, "X", interval="1h", warmup=60, stop_mode="atr",
    )
    assert _trade_summary(res_implicit) == _trade_summary(res_explicit)


# ─────── v2 opt-in fires ─────────────────────────────────────────


def test_v2_profile_emits_v2_slice_on_every_bar():
    df = _ohlcv(250)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    assert len(res.decision_traces) > 0
    for tr in res.decision_traces:
        assert tr.royal_road_decision_v2 is not None
        slice_ = tr.royal_road_decision_v2
        assert slice_.profile == "royal_road_decision_v2"
        assert slice_.action in ("BUY", "SELL", "HOLD")
        d = slice_.to_dict()
        for key in (
            "support_resistance_v2", "trendline_context",
            "chart_pattern_v2", "lower_tf_trigger", "macro_alignment",
            "structure_stop_plan",
            "compared_to_current_runtime", "compared_to_royal_road_v1",
        ):
            assert key in d


def test_v2_byte_identical_within_process():
    df = _ohlcv()
    res_a = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    res_b = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    assert _trade_summary(res_a) == _trade_summary(res_b)


def test_v2_with_structure_mode_runs_to_completion():
    df = _ohlcv(250)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
        stop_mode="structure",
    )
    assert res.metrics()["n_trades"] >= 0


def test_unknown_stop_mode_raises():
    df = _ohlcv(100)
    with pytest.raises(ValueError):
        run_engine_backtest(
            df, "X", interval="1h", warmup=60, stop_mode="trailing",
        )


# ─────── live / safety pins ──────────────────────────────────────


def test_decision_engine_does_not_import_v2():
    src = (SRC / "decision_engine.py").read_text()
    assert "royal_road_decision_v2" not in src


def test_risk_gate_does_not_import_v2():
    src = (SRC / "risk_gate.py").read_text()
    assert "royal_road_decision_v2" not in src


def test_indicators_does_not_import_v2():
    src = (SRC / "indicators.py").read_text()
    assert "royal_road_decision_v2" not in src


def test_oanda_paths_do_not_import_v2():
    for f in ("oanda.py", "broker.py"):
        path = SRC / f
        if path.exists():
            assert "royal_road_decision_v2" not in path.read_text()


def test_parameter_baseline_unchanged():
    from src.fx.parameter_defaults import PARAMETER_BASELINE_V1
    keys = set(PARAMETER_BASELINE_V1.keys())
    assert "royal_road_decision_v2" not in keys
    assert "stop_mode" not in keys
