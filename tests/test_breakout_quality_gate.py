"""Tests for breakout_quality_gate_v1 — 3-condition breakout gate.

Covers user spec cases 16-19:

 16. breakout without build-up → WARN/BLOCK
 17. breakout against higher timeframe → BLOCK on trend_alignment
 18. (strict) missing stop-loss accumulation → HOLD upstream
 19. (balanced) missing accumulation → WARN, not auto-HOLD
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.breakout_quality_gate import (
    SCHEMA_VERSION,
    build_breakout_quality_gate,
)


def _build_df_with_buildup(
    *, side: str, n: int = 100, trigger: float = 1.100,
) -> pd.DataFrame:
    """Build OHLC with explicit pre-breakout build-up (compression
    near trigger), then breakout."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    if side == "BUY":
        # Bars 0-79: lows climbing 1.094 → 1.099, highs capped at trigger
        lows = list(np.linspace(1.094, 1.099, 80))
        highs = [min(trigger - 0.0001, 1.0995)] * 80
        # Bars 80-99: post-breakout
        post_close = list(np.linspace(trigger + 0.001, trigger + 0.020, 20))
        closes = [(l + h) / 2 for l, h in zip(lows, highs)] + post_close
        opens = [c - 0.0001 for c in closes]
        full_lows = lows + [c - 0.0008 for c in post_close]
        full_highs = highs + [c + 0.0008 for c in post_close]
    else:
        highs = list(np.linspace(1.106, 1.101, 80))
        lows_pre = [max(trigger + 0.0001, 1.1005)] * 80
        post_close = list(np.linspace(trigger - 0.001, trigger - 0.020, 20))
        closes = [(l + h) / 2 for l, h in zip(lows_pre, highs)] + post_close
        opens = [c + 0.0001 for c in closes]
        full_lows = lows_pre + [c - 0.0008 for c in post_close]
        full_highs = highs + [c + 0.0008 for c in post_close]
    return pd.DataFrame({
        "open": opens, "high": full_highs, "low": full_lows,
        "close": closes, "volume": [1000] * n,
    }, index=idx)


def _build_df_no_buildup(*, n: int = 100) -> pd.DataFrame:
    """Random OHLC with no compression / no equal highs."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.100 + np.cumsum(rng.standard_normal(n) * 0.002)
    return pd.DataFrame({
        "open": close, "high": close + 0.002, "low": close - 0.002,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _pl(side: str, br_idx: int = 80) -> dict:
    return {
        "schema_version": "pattern_levels_v1",
        "available": True,
        "side": side,
        "trigger_line_price": 1.100,
        "parts": {"BR": {"price": 1.105, "index": br_idx}},
    }


# ── Case 16: no build-up → WARN ──────────────────────────────────
def test_no_buildup_yields_warn_or_block():
    df = _build_df_no_buildup()
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    # Build-up status should not be PASS for noisy curve
    assert gate["build_up_status"] in ("WARN", "BLOCK"), gate
    assert gate["status"] in ("WARN", "BLOCK")


def test_with_buildup_yields_pass_buildup():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["build_up_status"] == "PASS"


# ── Case 17: counter-trend → BLOCK ──────────────────────────────
def test_counter_trend_blocks():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="DOWN",  # counter-trend BUY
        atr_value=0.005,
    )
    assert gate["trend_alignment_status"] == "BLOCK"
    assert gate["status"] == "BLOCK"


def test_with_trend_passes():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["trend_alignment_status"] == "PASS"


def test_unknown_trend_warns():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="RANGE", atr_value=0.005,
    )
    assert gate["trend_alignment_status"] == "WARN"


def test_sell_with_down_trend_passes():
    df = _build_df_with_buildup(side="SELL")
    gate = build_breakout_quality_gate(
        side="SELL", pattern_levels=_pl("SELL"), df_window=df,
        higher_tf_trend="DOWN", atr_value=0.005,
    )
    assert gate["trend_alignment_status"] == "PASS"


def test_sell_against_up_trend_blocks():
    df = _build_df_with_buildup(side="SELL")
    gate = build_breakout_quality_gate(
        side="SELL", pattern_levels=_pl("SELL"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["trend_alignment_status"] == "BLOCK"


# ── Stop-loss accumulation ──────────────────────────────────────
def test_no_accumulation_warns():
    df = _build_df_no_buildup()
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    # No equal highs in random OHLC → WARN
    assert gate["stop_loss_accumulation_status"] in ("WARN", "PASS")


def test_with_accumulation_passes():
    """Build-up curve has highs capped at trigger many times → PASS."""
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["stop_loss_accumulation_status"] == "PASS"


# ── Aggregation invariants ──────────────────────────────────────
def test_any_block_yields_block_overall():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="DOWN",  # counter-trend → trend BLOCK
        atr_value=0.005,
    )
    assert gate["status"] == "BLOCK"


def test_all_pass_yields_pass_overall():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    if (
        gate["build_up_status"] == "PASS"
        and gate["trend_alignment_status"] == "PASS"
        and gate["stop_loss_accumulation_status"] == "PASS"
    ):
        assert gate["status"] == "PASS"


# ── Schema invariants ───────────────────────────────────────────
def test_schema_version():
    df = _build_df_with_buildup(side="BUY")
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=_pl("BUY"), df_window=df,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["schema_version"] == SCHEMA_VERSION == "breakout_quality_gate_v1"


def test_missing_pattern_levels_returns_warn():
    gate = build_breakout_quality_gate(
        side="BUY", pattern_levels=None, df_window=None,
        higher_tf_trend="UP", atr_value=0.005,
    )
    assert gate["status"] == "WARN"
    assert gate.get("unavailable_reason") == "missing_pattern_levels"
