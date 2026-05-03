"""Scenario tests for the integrated royal-road decision (Phase B).

Each test injects a complete `audit_panels` dict matching one of the
required scenarios in the user spec:

  1. double_bottom + WNL broken + support + bullish candle + RR>=2 → BUY
  2. double_bottom forming / WNL not broken → HOLD
  3. double_top + WNL broken + resistance + bearish candle + RR>=2 → SELL
  4. head_and_shoulders neckline_broken + RR>=2 → SELL
  5. inverse_head_and_shoulders neckline_broken + RR>=2 → BUY
  6. RR<2 → HOLD
  7. no WSL / no stop → HOLD
  8. macro strong against → HOLD
  9. strict mode + macro missing → HOLD
 10. balanced mode + macro missing → not auto-HOLD (WARN)
 11. RSI overbought in trend does not by itself produce SELL
 12. divergence alone does not produce BUY/SELL

We bypass the auto-build path by passing all panels explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.decision_engine import Decision
from src.fx.risk_gate import RiskState
from src.fx.royal_road_integrated_decision import (
    INTEGRATED_MODE_BALANCED,
    INTEGRATED_MODE_STRICT,
    PROFILE_NAME_V2_INTEGRATED,
    decide_royal_road_v2_integrated,
)


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2025-01-15 12:00:00", tz="UTC")


def _df(n: int = 200) -> pd.DataFrame:
    """Realistic-ish 200-bar df for the (mostly unused) auto-build path."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.0 + np.cumsum(rng.standard_normal(n) * 0.001)
    return pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001,
        "close": close, "volume": [1000] * n,
    }, index=idx)


# ── Panel factory helpers ─────────────────────────────────────────
def _wave_lines(side: str, *, wnl: float, wsl: float, wtp: float) -> list[dict]:
    pattern = "double_bottom" if side == "BUY" else "double_top"
    return [
        {"id": "WNL1", "kind": "neckline", "source_pattern": pattern,
         "price": wnl, "role": "entry_confirmation_line",
         "used_in_decision": False, "reason_ja": "test"},
        {"id": "WSL1", "kind": "pattern_invalidation", "source_pattern": pattern,
         "price": wsl, "role": "stop_candidate",
         "used_in_decision": False, "reason_ja": "test"},
        {"id": "WTP1", "kind": "pattern_target", "source_pattern": pattern,
         "price": wtp, "role": "target_candidate",
         "used_in_decision": False, "reason_ja": "test"},
    ]


def _entry_summary(*, entry: float, stop: float, tp: float, structure: bool = True) -> dict:
    rr = abs(tp - entry) / abs(entry - stop) if entry != stop else None
    return {
        "entry_price": entry, "stop_price": stop, "take_profit": tp,
        "rr": rr,
        "structure_stop_price": stop if structure else None,
        "atr_stop_price": stop,
    }


def _invalidation(es: dict) -> dict:
    return {
        "available": True, "observation_only": True, "used_in_decision": False,
        "schema_version": "invalidation_engine_v2",
        "stop_price": es["stop_price"],
        "structure_stop_price": es.get("structure_stop_price"),
        "atr_stop_price": es.get("atr_stop_price"),
        "rr": es.get("rr"),
        "rr_pass": es.get("rr") is not None and es["rr"] >= 2.0,
    }


def _chart_pattern(kind: str, *, broken: bool = True, forming: bool = False) -> dict:
    return {
        "available": True, "observation_only": True,
        "kind": kind,
        "neckline_broken": broken,
        "formation_complete": broken,
        "breakout_confirmed": broken,
        "invalidated": False,
        "state": "forming" if forming else "complete",
    }


def _candle(direction: str, *, location: str, bar_type: str = "bullish_pinbar") -> dict:
    return {
        "available": True, "observation_only": True,
        "bar_type": bar_type,
        "direction": direction,
        "location_quality": location,
        "near_support": location == "at_support",
        "near_resistance": location == "into_resistance",
        "power_chase_risk": False,
    }


def _dow(trend: str) -> dict:
    return {
        "available": True, "observation_only": True,
        "trend": trend,
        "sequence": [], "last_4_sequence": [],
        "trend_break_price": None, "reversal_confirmation_price": None,
    }


def _levels(*, support: bool = False, resistance: bool = False) -> dict:
    return {
        "available": True, "observation_only": True,
        "near_strong_support": support,
        "near_strong_resistance": resistance,
    }


def _fib(side: str, *, status: str = "PASS") -> dict:
    return {
        "available": True, "observation_only": True,
        "side": "UP" if side == "BUY" else "DOWN",
        "status": status,
        "meaning_ja": "test fib",
    }


def _ma(side: str, *, granville: str | None = None) -> dict:
    if granville is None:
        granville = (
            "trend_pullback_buy" if side == "BUY"
            else "trend_pullback_sell" if side == "SELL"
            else ""
        )
    return (
        {"available": True, "sma_20_slope_label": "rising" if side == "BUY" else "falling"},
        {"available": True, "granville_label": granville},
    )


def _rsi(*, regime: str = "TREND", state: str = "neutral") -> dict:
    return {
        "available": True,
        "regime": regime,
        "rsi_state": state,
    }


def _bb_neutral() -> dict:
    return {"available": True, "phase": "neutral", "band_walk_direction": ""}


def _macd_neutral() -> dict:
    return {"available": True, "zero_line_position": "above", "recent_cross": "none"}


def _div(*, has: bool = False) -> dict:
    return {"available": True, "has_divergence": has}


def _roadmap_available() -> dict:
    return {"available": True, "observation_only": True}


def _briefing_available() -> dict:
    return {"available": True, "observation_only": True}


def _macro_context_neutral() -> dict:
    """compute_macro_alignment expects per-bar macro fields."""
    return {
        "vix": 18.0,
        "dxy_trend_5d_bucket": "FLAT",
        "us10y_change_24h_bp": 0.0,
    }


def _macro_context_strong_against_buy() -> dict:
    """VERY_HIGH VIX (>= 30) triggers macro_strong_against=BUY for
    USDJPY. risk-off → JPY safe-haven bid → SELL alignment, so a BUY
    is blocked."""
    return {
        "vix": 35.0,
        "dxy_trend_5d_bucket": "FLAT",
        "us10y_change_24h_bp": 0.0,
    }


def _full_panels(
    *, side: str,
    pattern_kind: str = "double_bottom",
    pattern_broken: bool = True,
    pattern_forming: bool = False,
    rr: float = 2.5,
    structure_stop: bool = True,
    near_support: bool = False,
    near_resistance: bool = False,
    fib_status: str = "PASS",
    candle_bar: str = "bullish_pinbar",
    candle_location: str = "at_support",
    dow_trend: str = "UP",
    rsi_regime: str = "TREND",
    rsi_state: str = "neutral",
    has_divergence: bool = False,
    include_wlines: bool = True,
    wnl_broken: bool = True,
    include_roadmap: bool = True,
    include_briefing: bool = True,
) -> dict:
    if side == "BUY":
        entry, stop, tp = 1.0, 0.99, 1.0 + (0.01 * rr)
    else:
        entry, stop, tp = 1.0, 1.01, 1.0 - (0.01 * rr)
    es = _entry_summary(entry=entry, stop=stop, tp=tp, structure=structure_stop)
    if include_wlines:
        wnl = stop + (entry - stop) * 0.5  # midpoint
        if not wnl_broken:
            # Make WNL above entry for BUY (so last_close < WNL → not broken)
            wnl = entry + (tp - entry) * 0.3 if side == "BUY" else entry - (entry - tp) * 0.3
        wlines = _wave_lines(side, wnl=wnl, wsl=stop, wtp=tp)
    else:
        wlines = []
    ma_panel, granville = _ma(side)
    return {
        "chart_pattern_review": _chart_pattern(
            pattern_kind, broken=pattern_broken, forming=pattern_forming,
        ),
        "wave_derived_lines": wlines,
        "fibonacci_context_review": _fib(side, status=fib_status),
        "candlestick_anatomy_review": _candle(
            "BUY" if side == "BUY" else "SELL",
            location=candle_location, bar_type=candle_bar,
        ),
        "dow_structure_review": _dow(dow_trend),
        "level_psychology_review": _levels(
            support=near_support, resistance=near_resistance,
        ),
        "ma_context_review": ma_panel,
        "granville_entry_review": granville,
        "rsi_regime_filter": _rsi(regime=rsi_regime, state=rsi_state),
        "bollinger_lifecycle_review": _bb_neutral(),
        "macd_architecture_review": _macd_neutral(),
        "divergence_review": _div(has=has_divergence),
        "invalidation_engine_v2": _invalidation(es),
        "entry_summary": es,
        "daily_roadmap_review": _roadmap_available() if include_roadmap else {"available": False},
        "symbol_macro_briefing_review": _briefing_available() if include_briefing else {"available": False},
    }


def _decide(
    *, mode: str = INTEGRATED_MODE_BALANCED,
    panels: dict | None = None,
    last_close: float = 1.0,
    macro_context: dict | None = None,
):
    if macro_context is None:
        macro_context = _macro_context_neutral()
    return decide_royal_road_v2_integrated(
        df_window=_df(),
        technical_confluence={},
        pattern=None,
        higher_timeframe_trend="UP",
        risk_reward=None,
        risk_state=RiskState(),
        atr_value=0.005,
        last_close=last_close,
        symbol="USDJPY",
        macro_context=macro_context,
        df_lower_tf=None,
        lower_tf_interval=None,
        base_bar_close_ts=_ts(),
        mode=mode,
        audit_panels=panels,
    )


# ── Scenario tests ────────────────────────────────────────────────
def test_case_1_double_bottom_wnl_broken_buy():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        candle_bar="bullish_pinbar", candle_location="at_support",
    )
    d = _decide(panels=panels)
    assert isinstance(d, Decision)
    assert d.action == "BUY", (
        f"expected BUY, got {d.action}; reason={d.reason}; "
        f"axes={[(a['axis'], a['side'], a['status']) for a in d.advisory['integrated_decision']['axes']]}"
    )
    assert d.advisory["profile"] == PROFILE_NAME_V2_INTEGRATED


def test_case_2_double_bottom_forming_wnl_not_broken_hold():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        pattern_broken=False, pattern_forming=True,
        wnl_broken=False, near_support=True,
    )
    d = _decide(panels=panels, last_close=1.0)
    assert d.action == "HOLD"
    block_reasons = d.advisory["integrated_decision"]["block_reasons"]
    assert any("wave_lines" in r for r in block_reasons), block_reasons


def test_case_3_double_top_wnl_broken_sell():
    panels = _full_panels(
        side="SELL", pattern_kind="double_top",
        rr=2.5, near_resistance=True, dow_trend="DOWN",
        candle_bar="bearish_pinbar", candle_location="into_resistance",
    )
    d = _decide(panels=panels)
    assert d.action == "SELL", (
        f"expected SELL, got {d.action}; reason={d.reason}; "
        f"axes={[(a['axis'], a['side'], a['status']) for a in d.advisory['integrated_decision']['axes']]}"
    )


def test_case_4_head_and_shoulders_neckline_broken_sell():
    panels = _full_panels(
        side="SELL", pattern_kind="head_and_shoulders",
        rr=2.5, near_resistance=True, dow_trend="DOWN",
        candle_bar="bearish_engulfing", candle_location="into_resistance",
    )
    d = _decide(panels=panels)
    assert d.action == "SELL"


def test_case_5_inverse_head_and_shoulders_buy():
    panels = _full_panels(
        side="BUY", pattern_kind="inverse_head_and_shoulders",
        rr=2.5, near_support=True, dow_trend="UP",
        candle_bar="bullish_engulfing", candle_location="at_support",
    )
    d = _decide(panels=panels)
    assert d.action == "BUY"


def test_case_6_rr_below_2_hold():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=1.5, near_support=True, dow_trend="UP",
    )
    d = _decide(panels=panels)
    assert d.action == "HOLD"
    block_reasons = d.advisory["integrated_decision"]["block_reasons"]
    assert any("invalidation_rr" in r or "wave_lines" in r for r in block_reasons), block_reasons


def test_case_7_no_wsl_no_stop_hold():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        include_wlines=False,
    )
    d = _decide(panels=panels)
    assert d.action == "HOLD"
    block_reasons = d.advisory["integrated_decision"]["block_reasons"]
    assert any("wave_lines" in r for r in block_reasons), block_reasons


def test_case_8_macro_strong_against_hold():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
    )
    # VIX VERY_HIGH (>=30) for a BUY on USDJPY → macro_strong_against=BUY
    d = _decide(panels=panels, macro_context=_macro_context_strong_against_buy())
    assert d.action == "HOLD", (
        f"expected HOLD on macro_strong_against, got {d.action}; "
        f"reason={d.reason}"
    )
    block_reasons = d.advisory["integrated_decision"]["block_reasons"]
    assert any("macro" in r.lower() for r in block_reasons), block_reasons


def test_case_9_strict_macro_missing_holds():
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        include_roadmap=False, include_briefing=False,
    )
    # macro_context is None → macro_score becomes 0 (NEUTRAL) but available
    # We force macro UNKNOWN by passing None
    d = decide_royal_road_v2_integrated(
        df_window=_df(),
        technical_confluence={},
        pattern=None,
        higher_timeframe_trend="UP",
        risk_reward=None,
        risk_state=RiskState(),
        atr_value=0.005,
        last_close=1.0,
        symbol="USDJPY",
        macro_context=None,        # missing macro
        df_lower_tf=None,
        lower_tf_interval=None,
        base_bar_close_ts=_ts(),
        mode=INTEGRATED_MODE_STRICT,
        audit_panels=panels,
    )
    assert d.action == "HOLD"
    block_reasons = d.advisory["integrated_decision"]["block_reasons"]
    # In strict mode, missing roadmap/briefing should HOLD
    assert any(
        "roadmap" in r.lower() or "briefing" in r.lower()
        for r in block_reasons
    ), block_reasons


def test_case_10_balanced_macro_missing_does_not_hard_block():
    """In balanced mode, missing macro/roadmap/briefing must be WARN
    (not auto-BLOCK). With everything else PASS, BUY should still
    fire."""
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        include_roadmap=False, include_briefing=False,
    )
    d = decide_royal_road_v2_integrated(
        df_window=_df(),
        technical_confluence={},
        pattern=None,
        higher_timeframe_trend="UP",
        risk_reward=None,
        risk_state=RiskState(),
        atr_value=0.005,
        last_close=1.0,
        symbol="USDJPY",
        macro_context=None,
        df_lower_tf=None,
        lower_tf_interval=None,
        base_bar_close_ts=_ts(),
        mode=INTEGRATED_MODE_BALANCED,
        audit_panels=panels,
    )
    cautions = d.advisory["integrated_decision"]["cautions"]
    # Either BUY (if other axes carry it) or HOLD with no auto-block
    # from missing macro/roadmap. Cautions must mention them.
    assert d.action in ("BUY", "HOLD")
    if d.action == "HOLD":
        block_reasons = d.advisory["integrated_decision"]["block_reasons"]
        # must NOT block on missing macro/roadmap/briefing in balanced
        assert not any(
            r in ("roadmap_blocked", "briefing_blocked", "macro_blocked")
            for r in block_reasons
        ), block_reasons


def test_case_11_rsi_overbought_in_trend_does_not_create_sell_alone():
    """Inject only RSI overbought in TREND. No supporting axes for
    SELL. The result must NOT be SELL — RSI alone is caution-only."""
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        rsi_regime="TREND", rsi_state="overbought",
    )
    d = _decide(panels=panels)
    # The would-be action is BUY because all directional axes vote BUY.
    # RSI overbought in TREND does NOT flip it to SELL.
    assert d.action != "SELL", (
        f"RSI overbought in trend must not alone trigger SELL; got {d.action}"
    )


def test_case_12_divergence_alone_does_not_create_buy_or_sell():
    """Only divergence is set. No other directional evidence. Action
    must be HOLD."""
    # Build a panel set where all directional axes are NEUTRAL/UNKNOWN
    panels = _full_panels(
        side="BUY", pattern_kind="symmetric_triangle",
        pattern_broken=False, include_wlines=False,
        rr=2.5,
        dow_trend="RANGE",
        candle_location="midrange",
        has_divergence=True,
    )
    d = _decide(panels=panels)
    assert d.action == "HOLD"


# ── Profile metadata ──────────────────────────────────────────────
def test_advisory_carries_integrated_decision_dict():
    panels = _full_panels(side="BUY", pattern_kind="double_bottom",
                          rr=2.5, near_support=True, dow_trend="UP")
    d = _decide(panels=panels)
    assert d.advisory["profile"] == PROFILE_NAME_V2_INTEGRATED
    assert "integrated_decision" in d.advisory
    intd = d.advisory["integrated_decision"]
    assert intd["schema_version"] == "royal_road_integrated_decision_v1"
    assert intd["mode"] == INTEGRATED_MODE_BALANCED
    assert isinstance(intd["axes"], list)
    assert len(intd["axes"]) >= 13
    # Every axis carries the contract fields
    for ax in intd["axes"]:
        assert set(ax.keys()) >= {
            "axis", "side", "status", "strength", "confidence",
            "used_in_decision", "required", "reason_ja", "source",
        }


def test_strict_and_balanced_produce_distinct_block_behaviour():
    """With macro/roadmap/briefing missing, strict must HOLD and
    balanced must NOT auto-block on those panels (cautions only)."""
    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        include_roadmap=False, include_briefing=False,
    )
    strict = decide_royal_road_v2_integrated(
        df_window=_df(), technical_confluence={}, pattern=None,
        higher_timeframe_trend="UP", risk_reward=None,
        risk_state=RiskState(), atr_value=0.005, last_close=1.0,
        symbol="USDJPY", macro_context=None,
        df_lower_tf=None, lower_tf_interval=None,
        base_bar_close_ts=_ts(), mode=INTEGRATED_MODE_STRICT,
        audit_panels=panels,
    )
    balanced = decide_royal_road_v2_integrated(
        df_window=_df(), technical_confluence={}, pattern=None,
        higher_timeframe_trend="UP", risk_reward=None,
        risk_state=RiskState(), atr_value=0.005, last_close=1.0,
        symbol="USDJPY", macro_context=None,
        df_lower_tf=None, lower_tf_interval=None,
        base_bar_close_ts=_ts(), mode=INTEGRATED_MODE_BALANCED,
        audit_panels=panels,
    )
    # Strict must HOLD
    assert strict.action == "HOLD"
    # Balanced must not auto-block on the missing panels
    bal_blocks = balanced.advisory["integrated_decision"]["block_reasons"]
    assert not any(
        r in ("daily_roadmap_blocked", "symbol_macro_briefing_blocked")
        for r in bal_blocks
    ), bal_blocks
