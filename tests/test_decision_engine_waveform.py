"""Decision Engine + waveform_bias integration tests (spec §7.4 / §19).

Pinned guarantees:
  * Waveform alone NEVER triggers a BUY/SELL (technical=HOLD → HOLD).
  * Waveform agreement with the technical signal can bump confidence
    a little but cannot lift it past the LLM/min_confidence checks.
  * Waveform DISAGREEMENT with the technical signal vetoes to HOLD.
  * Risk Gate still wins over everything (FOMC + waveform=BUY → HOLD).
  * No-LLM path still benefits from waveform agreement.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fx.analyst import TradeSignal
from src.fx.calendar import Event
from src.fx.decision_engine import decide
from src.fx.patterns import PatternResult, TrendState
from src.fx.risk_gate import RiskState
from src.fx.waveform_backtest import WaveformBias


def _bias(action: str, conf: float = 0.7, samples: int = 30) -> WaveformBias:
    return WaveformBias(
        action=action, confidence=conf, sample_count=samples,
        horizon_bars=24,
        bullish_count=samples if action == "BUY" else 0,
        bearish_count=samples if action == "SELL" else 0,
        neutral_count=0,
        avg_forward_return_pct=0.5 if action == "BUY"
        else (-0.5 if action == "SELL" else 0.0),
        avg_max_favorable_pct=0.7,
        avg_max_adverse_pct=-0.3,
        avg_match_score=0.85,
        method="dtw",
        reason="synthetic",
    )


def _llm(action: str, conf: float = 0.8) -> TradeSignal:
    return TradeSignal(
        action=action, confidence=conf, reason="t", key_risks=[],
        expected_direction="UP" if action == "BUY" else "DOWN" if action == "SELL" else "FLAT",
        expected_magnitude_pct=0.5, horizon_bars=4,
        invalidation_price=99.0 if action == "BUY" else 101.0,
        raw_response={},
    )


def test_waveform_alone_never_trades_when_technical_is_hold():
    d = decide(
        technical_signal="HOLD",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.99),
        waveform_bias=_bias("BUY", conf=0.95),
    )
    assert d.action == "HOLD"


def test_waveform_disagreement_vetoes_to_hold():
    d = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY"),
        waveform_bias=_bias("SELL", conf=0.8),
    )
    assert d.action == "HOLD"
    assert "waveform" in d.reason.lower()
    # Advisory still records the bias for later inspection
    assert d.advisory.get("waveform_bias", {}).get("action") == "SELL"


def test_waveform_agreement_bumps_confidence_modestly():
    base = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
        waveform_bias=None,
    )
    boosted = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
        waveform_bias=_bias("BUY", conf=0.9),
    )
    assert base.action == "BUY"
    assert boosted.action == "BUY"
    assert boosted.confidence > base.confidence
    # Must not exceed the 0.95 cap.
    assert boosted.confidence <= 0.95


def test_risk_gate_still_wins_over_waveform_buy():
    """FOMC inside the window forces HOLD even if waveform_bias=BUY."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    fomc = Event(when=now + timedelta(hours=1), currency="USD",
                 title="FOMC", impact="high")
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,
        risk_state=RiskState(events=(fomc,), now=now),
        llm_signal=_llm("BUY"),
        waveform_bias=_bias("BUY", conf=0.95),
    )
    assert d.action == "HOLD"
    assert "event_high" in d.blocked_by


def test_waveform_hold_is_neutral_does_not_block():
    """A HOLD bias from waveform shouldn't veto a clean BUY."""
    d = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
        waveform_bias=_bias("HOLD", conf=0.0, samples=0),
    )
    assert d.action == "BUY"


def test_no_llm_with_agreeing_waveform_still_buys():
    d = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=None,
        waveform_bias=_bias("BUY", conf=0.8),
    )
    assert d.action == "BUY"
    # Base 0.55 (no LLM) + waveform bump
    assert d.confidence > 0.55
    assert d.confidence <= 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
