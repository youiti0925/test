"""Tests for StrategyConfig flowing through to decide().

Pinned guarantees:
  * `decide()`'s `min_risk_reward` and `min_confidence` honour caller
    overrides (StrategyConfig is just the source of those overrides).
  * Loading a YAML/JSON config and passing its values through changes
    HOLD vs. BUY at the boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.fx.analyst import TradeSignal
from src.fx.decision_engine import decide
from src.fx.patterns import PatternResult, TrendState
from src.fx.risk_gate import RiskState
from src.fx.strategy_config import RiskConfig, StrategyConfig, WaveformConfig


def _llm(action: str, conf: float) -> TradeSignal:
    return TradeSignal(
        action=action, confidence=conf, reason="t", key_risks=[],
        expected_direction="UP" if action == "BUY"
        else "DOWN" if action == "SELL" else "FLAT",
        expected_magnitude_pct=0.5, horizon_bars=4,
        invalidation_price=99.0,
        raw_response={},
    )


def test_min_confidence_override_blocks_borderline_llm():
    """conf=0.65 passes the default (0.6) but fails a stricter 0.7 floor."""
    base = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.65),
    )
    assert base.action == "BUY"

    strict = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.65),
        min_confidence=0.7,
    )
    assert strict.action == "HOLD"


def test_min_risk_reward_override_blocks_borderline():
    base = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=1.6,                   # passes default 1.5
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
    )
    assert base.action == "BUY"

    strict = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=1.6,                   # fails stricter 2.0 floor
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
        min_risk_reward=2.0,
    )
    assert strict.action == "HOLD"


def test_strategy_config_supplies_decide_thresholds(tmp_path: Path):
    """End-to-end: StrategyConfig values map onto decide() kwargs."""
    cfg = StrategyConfig(
        risk=RiskConfig(stop_atr=2.0, take_profit_atr=4.0,
                        min_risk_reward=2.5, risk_pct=0.005,
                        max_holding_bars=24),
        waveform=WaveformConfig(
            window_bars=60, min_similar_count=20, min_confidence=0.75,
        ),
    )
    p = tmp_path / "cfg.json"
    cfg.write(p)
    loaded = StrategyConfig.load(p)

    # The min_risk_reward floor from the config must reject 2.0 RR.
    d = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,                  # below the 2.5 floor in cfg
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
        min_risk_reward=loaded.risk.min_risk_reward,
        min_confidence=loaded.waveform.min_confidence,
    )
    assert d.action == "HOLD"

    # And a setup that clears both floors goes through.
    d2 = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,                  # above 2.5
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),  # above 0.75
        min_risk_reward=loaded.risk.min_risk_reward,
        min_confidence=loaded.waveform.min_confidence,
    )
    assert d2.action == "BUY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
