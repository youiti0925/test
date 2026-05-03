"""Unit tests for `royal_road_decision.decide_royal_road` rule chain.

Updated for v1.1 modes (strict / balanced / exploratory). Each test
feeds a hand-built `technical_confluence_v1` dict, an empty
df-based RiskState (so the gate passes), an explicit mode, and
asserts the rule output.

The default mode (`balanced`) is verified separately. Existing v1.0
tests that asserted strict-only behaviour now run under mode='strict'.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision_modes import (
    DEFAULT_ROYAL_ROAD_MODE,
    supported_royal_road_modes,
    get_royal_road_mode_config as get_mode_config,
)
from src.fx.royal_road_decision import (
    PROFILE_NAME,
    SUPPORTED_DECISION_PROFILES,
    compare_decisions,
    decide_royal_road,
    validate_decision_profile,
    validate_royal_road_mode,
)


# ─────────── helpers ──────────────────────────────────────────────


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _state() -> RiskState:
    return RiskState(
        df=_ohlcv(200),
        events=(),
        spread_pct=None,
        sentiment_snapshot=None,
        now=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
    )


def _confluence(
    *,
    label: str = "STRONG_BUY_SETUP",
    score: float = 0.6,
    invalidation_clear: bool = True,
    structure_stop: float | None = 99.0,
    near_support: bool = True,
    near_resistance: bool = False,
    role_reversal: bool = False,
    candle_bull: bool = False,
    candle_bear: bool = False,
    market_regime: str = "TREND_UP",
    structure_code: str = "HL",
    bos_up: bool = False,
    bos_down: bool = False,
    chart_pattern: dict | None = None,
    avoid_reasons: list[str] | None = None,
    bullish_reasons: list[str] | None = None,
    bearish_reasons: list[str] | None = None,
) -> dict:
    return {
        "policy_version": "technical_confluence_v1",
        "market_regime": market_regime,
        "dow_structure": {
            "structure_code": structure_code,
            "last_swing_high": 101.0, "last_swing_low": 99.0,
            "bos_up": bos_up, "bos_down": bos_down,
        },
        "support_resistance": {
            "near_support": near_support,
            "near_resistance": near_resistance,
            "breakout": False, "pullback": False,
            "role_reversal": role_reversal, "fake_breakout": False,
        },
        "candlestick_signal": {
            "bullish_pinbar": candle_bull,
            "bearish_pinbar": candle_bear,
            "bullish_engulfing": False, "bearish_engulfing": False,
            "harami": False,
            "strong_bull_body": False, "strong_bear_body": False,
            "rejection_wick": False,
        },
        "chart_pattern": chart_pattern or {},
        "indicator_context": {},
        "risk_plan_obs": {
            "atr_stop_distance_atr": 2.0,
            "structure_stop_price": structure_stop,
            "structure_stop_distance_atr": 1.2 if structure_stop else None,
            "rr_atr_based": 1.5,
            "rr_structure_based": 2.5 if structure_stop else None,
            "invalidation_clear": invalidation_clear,
        },
        "vote_breakdown": {},
        "final_confluence": {
            "label": label, "score": score,
            "bullish_reasons": bullish_reasons or [],
            "bearish_reasons": bearish_reasons or [],
            "avoid_reasons":   avoid_reasons or [],
        },
    }


def _decide(tc: dict, *, mode: str = "balanced", **overrides):
    kwargs = dict(
        technical_signal="HOLD",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=_state(),
        technical_confluence=tc,
        mode=mode,
    )
    kwargs.update(overrides)
    return decide_royal_road(**kwargs)


# ─────────── happy paths (STRONG) ─────────────────────────────────


def test_strong_buy_with_full_evidence_returns_buy_in_strict():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP", near_support=True,
        market_regime="TREND_UP", structure_code="HL",
        candle_bull=True, score=0.7,
    ), mode="strict")
    assert d.action == "BUY"
    adv = d.advisory
    assert adv["profile"] == PROFILE_NAME
    assert adv["mode"] == "strict"
    assert adv["mode_status"] == "heuristic_not_validated"
    assert adv["mode_needs_validation"] is True
    assert adv["score"] == 0.7
    assert adv["block_reasons"] == []
    assert adv["cautions"] == []
    assert adv["evidence_axes_count"]["bullish"] >= 1
    assert 0.55 <= d.confidence <= 0.95


def test_strong_buy_with_full_evidence_returns_buy_in_balanced():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP", near_support=True,
        market_regime="TREND_UP", structure_code="HL",
        candle_bull=True, score=0.7,
    ), mode="balanced")
    assert d.action == "BUY"
    assert d.advisory["mode"] == "balanced"
    assert d.advisory["mode_status"] == "heuristic_not_validated_default"


def test_strong_sell_with_full_evidence_returns_sell_in_balanced():
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP",
            near_support=False, near_resistance=True,
            market_regime="TREND_DOWN", structure_code="LH",
            candle_bull=False, candle_bear=True, score=-0.6,
        ),
        higher_timeframe_trend="DOWNTREND",
        mode="balanced",
    )
    assert d.action == "SELL"


# ─────────── strict: WEAK → HOLD ──────────────────────────────────


def test_strict_weak_buy_setup_is_hold():
    d = _decide(_confluence(label="WEAK_BUY_SETUP"), mode="strict")
    assert d.action == "HOLD"
    assert any(
        r.startswith("weak_setup:WEAK_BUY_SETUP")
        for r in d.advisory["block_reasons"]
    )


def test_strict_weak_sell_setup_is_hold():
    d = _decide(_confluence(label="WEAK_SELL_SETUP"), mode="strict")
    assert d.action == "HOLD"
    assert any(
        r.startswith("weak_setup:WEAK_SELL_SETUP")
        for r in d.advisory["block_reasons"]
    )


def test_strict_invalidation_unclear_blocks_buy():
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", invalidation_clear=False),
        mode="strict",
    )
    assert d.action == "HOLD"
    assert "invalidation_unclear" in d.advisory["block_reasons"]
    assert d.advisory["cautions"] == []


def test_strict_structure_stop_missing_blocks_buy():
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        mode="strict",
    )
    assert d.action == "HOLD"
    assert "structure_stop_missing" in d.advisory["block_reasons"]


# ─────────── balanced: WEAK + 2 axes → entry ──────────────────────


def test_balanced_weak_buy_with_two_axes_returns_buy():
    """Two bullish axes: bullish_structure (TREND_UP) + near_support."""
    d = _decide(_confluence(
        label="WEAK_BUY_SETUP",
        market_regime="TREND_UP", structure_code="HL",
        near_support=True, near_resistance=False,
        candle_bull=False,
    ), mode="balanced")
    assert d.action == "BUY"
    assert d.advisory["mode"] == "balanced"
    assert d.advisory["evidence_axes_count"]["bullish"] >= 2
    assert d.advisory["min_evidence_axes_required"] == 2
    assert d.advisory["block_reasons"] == []


def test_balanced_weak_sell_with_two_axes_returns_sell():
    d = _decide(
        _confluence(
            label="WEAK_SELL_SETUP",
            market_regime="TREND_DOWN", structure_code="LH",
            near_support=False, near_resistance=True,
            candle_bull=False, candle_bear=False,
        ),
        higher_timeframe_trend="DOWNTREND",
        mode="balanced",
    )
    assert d.action == "SELL"
    assert d.advisory["evidence_axes_count"]["bearish"] >= 2


def test_balanced_weak_buy_with_one_axis_only_returns_hold():
    """Only near_support, no bullish structure / candle."""
    d = _decide(_confluence(
        label="WEAK_BUY_SETUP",
        market_regime="RANGE", structure_code="MIXED",
        near_support=True, near_resistance=False,
        candle_bull=False,
    ), mode="balanced")
    assert d.action == "HOLD"
    assert any(
        "insufficient_buy_evidence_axes" in r
        for r in d.advisory["block_reasons"]
    )


# ─────────── balanced: opposite-side level still blocks ───────────


def test_balanced_near_resistance_blocks_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        near_support=False, near_resistance=True,
        market_regime="TREND_UP", structure_code="HL",
        candle_bull=True,
    ), mode="balanced")
    assert d.action == "HOLD"
    assert "near_resistance_for_buy" in d.advisory["block_reasons"]


def test_balanced_near_support_blocks_sell():
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP",
            near_support=True, near_resistance=False,
            market_regime="TREND_DOWN", structure_code="LH",
            candle_bear=True, score=-0.6,
        ),
        higher_timeframe_trend="DOWNTREND",
        mode="balanced",
    )
    assert d.action == "HOLD"
    assert "near_support_for_sell" in d.advisory["block_reasons"]


# ─────────── balanced: cautions for soft-fail items ───────────────


def test_balanced_structure_stop_missing_emits_caution_not_block():
    """STRONG setup with all evidence but structure_stop missing →
    BUY allowed in balanced; structure_stop_missing surfaces in cautions."""
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP", structure_stop=None,
        near_support=True, market_regime="TREND_UP",
        structure_code="HL", candle_bull=True,
    ), mode="balanced")
    assert d.action == "BUY"
    assert "structure_stop_missing_soft" in d.advisory["cautions"]
    assert "structure_stop_missing" not in d.advisory["block_reasons"]


def test_balanced_invalidation_unclear_emits_caution_not_block():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP", invalidation_clear=False,
        near_support=True, market_regime="TREND_UP",
        structure_code="HL", candle_bull=True,
    ), mode="balanced")
    assert d.action == "BUY"
    assert "invalidation_unclear_soft" in d.advisory["cautions"]
    assert "invalidation_unclear" not in d.advisory["block_reasons"]


# ─────────── balanced: serious avoid still blocks ─────────────────


def test_balanced_serious_avoid_blocks_strong_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        near_support=True, market_regime="TREND_UP",
        structure_code="HL", candle_bull=True,
        avoid_reasons=["fake_breakout"],
    ), mode="balanced")
    assert d.action == "HOLD"
    assert any("fake_breakout" in r for r in d.advisory["block_reasons"])


# ─────────── balanced: htf counter-trend still blocks ─────────────


def test_balanced_htf_counter_trend_blocks_buy():
    d = _decide(
        _confluence(
            label="STRONG_BUY_SETUP", near_support=True,
            market_regime="TREND_UP", structure_code="HL",
            candle_bull=True,
        ),
        higher_timeframe_trend="DOWNTREND",
        mode="balanced",
    )
    assert d.action == "HOLD"
    assert "htf_counter_trend_for_buy" in d.advisory["block_reasons"]


# ─────────── exploratory: WEAK + 1 axis → entry ───────────────────


def test_exploratory_weak_buy_with_single_axis_returns_buy():
    d = _decide(_confluence(
        label="WEAK_BUY_SETUP",
        market_regime="RANGE", structure_code="MIXED",
        near_support=True, near_resistance=False,
        candle_bull=False,
    ), mode="exploratory")
    assert d.action == "BUY"
    assert d.advisory["mode"] == "exploratory"
    assert d.advisory["mode_status"] == "exploratory_not_validated"
    assert d.advisory["min_evidence_axes_required"] == 1


def test_exploratory_blocks_htf_counter_trend_per_remote_modes_config():
    """Remote modes.py defines exploratory with htf_counter_trend_blocks=True
    (same as strict / balanced). exploratory differs from balanced in
    `min_evidence_axes_weak` and `serious_avoid_reasons` only — not in
    htf alignment. This test pins the remote config so a future widening
    of exploratory would have to be deliberate."""
    cfg = get_mode_config("exploratory")
    assert cfg.htf_counter_trend_blocks is True
    d = _decide(
        _confluence(
            label="STRONG_BUY_SETUP", near_support=True,
            market_regime="TREND_UP", structure_code="HL",
            candle_bull=True,
        ),
        higher_timeframe_trend="DOWNTREND",
        mode="exploratory",
    )
    assert d.action == "HOLD"
    assert "htf_counter_trend_for_buy" in d.advisory["block_reasons"]


# ─────────── default mode is balanced ─────────────────────────────


def test_default_mode_is_balanced():
    assert DEFAULT_ROYAL_ROAD_MODE == "balanced"
    # Calling without explicit mode should land on balanced.
    d = decide_royal_road(
        technical_signal="HOLD",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=_state(),
        technical_confluence=_confluence(label="STRONG_BUY_SETUP"),
    )
    assert d.advisory["mode"] == "balanced"
    assert d.advisory["mode_status"] == "heuristic_not_validated_default"


# ─────────── advisory shape (mode metadata always present) ────────


def test_advisory_carries_mode_metadata_on_buy():
    d = _decide(_confluence(label="STRONG_BUY_SETUP"), mode="balanced")
    adv = d.advisory
    for key in (
        "profile", "mode", "mode_status", "mode_needs_validation",
        "score", "reasons", "block_reasons", "cautions",
        "evidence_axes", "evidence_axes_count",
        "min_evidence_axes_required", "source",
    ):
        assert key in adv, f"advisory missing key: {key}"


def test_advisory_carries_mode_metadata_on_hold():
    d = _decide(_confluence(label="WEAK_BUY_SETUP"), mode="strict")
    adv = d.advisory
    for key in (
        "profile", "mode", "mode_status", "mode_needs_validation",
        "score", "reasons", "block_reasons", "cautions",
        "evidence_axes", "evidence_axes_count",
        "min_evidence_axes_required", "source",
    ):
        assert key in adv, f"advisory missing key: {key}"


# ─────────── compare_decisions taxonomy (unchanged) ───────────────


class _Dummy:
    def __init__(self, action: str) -> None:
        self.action = action


def test_compare_same_action_same():
    out = compare_decisions(
        decision_current=_Dummy("BUY"), decision_royal=_Dummy("BUY"),
    )
    assert out["difference_type"] == "same"
    assert out["same_action"] is True


def test_compare_current_buy_royal_hold():
    out = compare_decisions(
        decision_current=_Dummy("BUY"), decision_royal=_Dummy("HOLD"),
    )
    assert out["difference_type"] == "current_buy_royal_hold"


def test_compare_current_hold_royal_sell():
    out = compare_decisions(
        decision_current=_Dummy("HOLD"), decision_royal=_Dummy("SELL"),
    )
    assert out["difference_type"] == "current_hold_royal_sell"


def test_compare_opposite_direction():
    out = compare_decisions(
        decision_current=_Dummy("BUY"), decision_royal=_Dummy("SELL"),
    )
    assert out["difference_type"] == "opposite_direction"


# ─────────── profile + mode validation ────────────────────────────


def test_supported_profiles_includes_default_and_royal():
    assert "current_runtime" in SUPPORTED_DECISION_PROFILES
    assert PROFILE_NAME in SUPPORTED_DECISION_PROFILES


def test_validate_decision_profile_rejects_unknown():
    with pytest.raises(ValueError):
        validate_decision_profile("nonexistent_profile")


def test_supported_modes_set():
    assert set(SUPPORTED_ROYAL_ROAD_MODES) == {
        "strict", "balanced", "exploratory",
    }


def test_validate_royal_road_mode_rejects_unknown():
    with pytest.raises(ValueError):
        validate_royal_road_mode("aggressive")


def test_get_mode_config_returns_expected_values():
    s = get_mode_config("strict")
    b = get_mode_config("balanced")
    e = get_mode_config("exploratory")

    assert s.allow_weak_entries is False
    assert s.require_invalidation_clear is True
    assert s.require_structure_stop is True

    assert b.allow_weak_entries is True
    assert b.require_invalidation_clear is False
    assert b.require_structure_stop is False
    assert b.min_evidence_axes_weak == 2

    assert e.allow_weak_entries is True
    # Remote modes.py keeps htf counter-trend blocking on in exploratory;
    # exploratory differs from balanced in min_evidence_axes_weak and
    # serious_avoid_reasons only.
    assert e.htf_counter_trend_blocks is True
    assert e.min_evidence_axes_weak == 1

    # All three modes are still flagged as needing validation.
    for cfg in (s, b, e):
        assert cfg.needs_validation is True

SUPPORTED_ROYAL_ROAD_MODES = supported_royal_road_modes()
