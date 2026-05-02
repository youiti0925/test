"""Tests for entry_plan_v1 — READY / WAIT_BREAKOUT / WAIT_RETEST /
HOLD state machine.

Covers user spec cases 5-11:

  5. WNL not broken → WAIT_BREAKOUT / HOLD
  6. broken but no retest → WAIT_RETEST / HOLD
  7. retest + bullish candle + RR≥2 → READY / BUY
  8. retest + bearish candle + RR≥2 → READY / SELL
  9. no stop → HOLD
 10. no target → HOLD
 11. RR<2 → HOLD
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.entry_plan import SCHEMA_VERSION, build_entry_plan


def _df_simple(n: int = 50, start_price: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = start_price + np.cumsum(rng.standard_normal(n) * 0.001)
    return pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _df_breakout_then_retest(side: str = "BUY") -> pd.DataFrame:
    """Build a deterministic OHLC: 30 bars below trigger, then a
    breakout, then a retest, then a confirmation bar.
    """
    idx = pd.date_range("2025-01-01", periods=50, freq="1h", tz="UTC")
    if side == "BUY":
        # 0-29: below NL=1.10
        # 30-39: breakout (close>1.10)
        # 40-44: retest at NL
        # 45-49: bullish run, last bar a clear bullish body
        closes_list = [1.090] * 30 + [1.105] * 10 + [1.100] * 5 + [1.110] * 5
        opens_list = [c - 0.0001 if i < 49 else c - 0.0030 for i, c in enumerate(closes_list)]
        lows_list = [c - 0.0010 for c in closes_list]
        highs_list = [c + 0.0010 for c in closes_list]
    else:
        closes_list = [1.110] * 30 + [1.095] * 10 + [1.100] * 5 + [1.090] * 5
        opens_list = [c + 0.0001 if i < 49 else c + 0.0030 for i, c in enumerate(closes_list)]
        lows_list = [c - 0.0010 for c in closes_list]
        highs_list = [c + 0.0010 for c in closes_list]
    return pd.DataFrame({
        "open": opens_list, "high": highs_list, "low": lows_list,
        "close": closes_list, "volume": [1000] * 50,
    }, index=idx)


def _pattern_levels_buy(*, broken: bool, target: float = 1.150) -> dict:
    return {
        "schema_version": "pattern_levels_v1",
        "available": True,
        "pattern_kind": "double_bottom",
        "side": "BUY",
        "status": "neckline_broken" if broken else "forming",
        "parts": {
            "B1": {"price": 1.090, "index": 5},
            "NL": {"price": 1.100, "index": 15},
            "B2": {"price": 1.092, "index": 25},
        },
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.100,
        "stop_price": 1.085,
        "target_price": target,
        "target_extended_price": 1.180,
        "rr_at_reference": 1.0,
        "rr_at_extended_target": 5.33,  # (1.18-1.10)/(1.10-1.085) ≈ 5.33
        "breakout_confirmed": broken,
        "retest_confirmed": False,
        "pattern_height": 0.010,
    }


def _pattern_levels_sell(*, broken: bool) -> dict:
    return {
        "schema_version": "pattern_levels_v1",
        "available": True,
        "pattern_kind": "double_top",
        "side": "SELL",
        "status": "neckline_broken" if broken else "forming",
        "parts": {
            "P1": {"price": 1.110, "index": 5},
            "NL": {"price": 1.100, "index": 15},
            "P2": {"price": 1.108, "index": 25},
        },
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.100,
        "stop_price": 1.115,
        "target_price": 1.090,
        "target_extended_price": 1.080,
        "rr_at_reference": 1.0,
        "rr_at_extended_target": 5.33,
        "breakout_confirmed": broken,
        "retest_confirmed": False,
        "pattern_height": 0.010,
    }


# ── Case 5: WNL not broken → WAIT_BREAKOUT / HOLD ────────────────
def test_wait_breakout_when_neckline_not_broken():
    pl = _pattern_levels_buy(broken=False)
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "doji", "direction": "NEUTRAL"},
        df_window=_df_simple(),
        last_close=1.099,
        atr_value=0.005,
    )
    assert plan["entry_status"] == "WAIT_BREAKOUT"
    assert plan["side"] == "BUY"
    assert plan["entry_type"] == "breakout"
    assert "wnl_not_broken" in plan["block_reasons"]


# ── Case 6: broken but no retest → WAIT_RETEST / HOLD ────────────
def test_wait_retest_when_no_pullback():
    pl = _pattern_levels_buy(broken=True)
    # df where price kept rising and never came back to NL
    idx = pd.date_range("2025-01-01", periods=40, freq="1h", tz="UTC")
    closes = pd.Series([1.090] * 20 + [1.105 + i * 0.001 for i in range(20)])
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.001,
        "low": closes - 0.001, "close": closes,
        "volume": [1000] * 40,
    }, index=idx)
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "large_bull", "direction": "BUY"},
        df_window=df,
        last_close=float(df["close"].iloc[-1]),
        atr_value=0.005,
    )
    assert plan["entry_status"] == "WAIT_RETEST"
    assert plan["breakout_confirmed"] is True
    assert plan["retest_confirmed"] is False
    assert "awaiting_retest_confirmation" in plan["block_reasons"]


# ── Case 7: retest + bullish + RR≥2 → READY / BUY ────────────────
def test_ready_buy_on_retest_with_bullish_confirmation():
    pl = _pattern_levels_buy(broken=True)
    df = _df_breakout_then_retest(side="BUY")
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "bullish_engulfing", "direction": "BUY"},
        df_window=df,
        last_close=float(df["close"].iloc[-1]),
        atr_value=0.005,
    )
    assert plan["entry_status"] == "READY"
    assert plan["side"] == "BUY"
    assert plan["breakout_confirmed"] is True
    assert plan["retest_confirmed"] is True
    assert plan["rr"] is not None
    assert plan["rr"] >= 2.0


# ── Case 8: retest + bearish + RR≥2 → READY / SELL ───────────────
def test_ready_sell_on_retest_with_bearish_confirmation():
    pl = _pattern_levels_sell(broken=True)
    df = _df_breakout_then_retest(side="SELL")
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "bearish_engulfing", "direction": "SELL"},
        df_window=df,
        last_close=float(df["close"].iloc[-1]),
        atr_value=0.005,
    )
    assert plan["entry_status"] == "READY"
    assert plan["side"] == "SELL"
    assert plan["retest_confirmed"] is True


# ── Case 9: no stop → HOLD ───────────────────────────────────────
def test_hold_when_stop_missing():
    pl = _pattern_levels_buy(broken=True)
    pl["stop_price"] = None
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "bullish_engulfing", "direction": "BUY"},
        df_window=_df_simple(),
        last_close=1.110,
        atr_value=0.005,
    )
    assert plan["entry_status"] == "HOLD"
    assert "missing_stop" in plan["block_reasons"]


# ── Case 10: no target → HOLD ────────────────────────────────────
def test_hold_when_target_missing():
    pl = _pattern_levels_buy(broken=True)
    pl["target_price"] = None
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "bullish_engulfing", "direction": "BUY"},
        df_window=_df_simple(),
        last_close=1.110,
        atr_value=0.005,
    )
    assert plan["entry_status"] == "HOLD"
    assert "missing_target" in plan["block_reasons"]


# ── Case 11: RR<2 → HOLD ─────────────────────────────────────────
def test_hold_when_rr_below_threshold():
    pl = _pattern_levels_buy(broken=True)
    pl["rr_at_extended_target"] = 1.5
    df = _df_breakout_then_retest(side="BUY")
    plan = build_entry_plan(
        pattern_levels=pl,
        candle_review={"bar_type": "bullish_engulfing", "direction": "BUY"},
        df_window=df,
        last_close=float(df["close"].iloc[-1]),
        atr_value=0.005,
        min_rr=2.0,
    )
    assert plan["entry_status"] == "HOLD"
    assert any("rr" in r for r in plan["block_reasons"])


# ── Schema invariants ────────────────────────────────────────────
def test_schema_version():
    pl = _pattern_levels_buy(broken=False)
    plan = build_entry_plan(
        pattern_levels=pl, candle_review=None,
        df_window=_df_simple(), last_close=1.099,
        atr_value=0.005,
    )
    assert plan["schema_version"] == SCHEMA_VERSION == "entry_plan_v1"


def test_unavailable_when_pattern_levels_missing():
    plan = build_entry_plan(
        pattern_levels=None, candle_review=None,
        df_window=_df_simple(), last_close=1.0, atr_value=0.005,
    )
    assert plan["entry_status"] == "HOLD"
    assert plan["side"] == "NEUTRAL"


def test_neutral_side_pattern_holds():
    pl = _pattern_levels_buy(broken=False)
    pl["side"] = "NEUTRAL"
    plan = build_entry_plan(
        pattern_levels=pl, candle_review=None,
        df_window=_df_simple(), last_close=1.0, atr_value=0.005,
    )
    assert plan["entry_status"] == "HOLD"
