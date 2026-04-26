"""Required tests from spec §15.

Each test pins a single safety guarantee. Read the names — they ARE the
specification. If any of these regress, the engine's safety story has
been silently weakened.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.analyst import TradeSignal
from src.fx.calendar import Event
from src.fx.decision_engine import decide
from src.fx.patterns import PatternResult, TrendState, analyse, detect_swings
from src.fx.risk_gate import RiskState, evaluate as gate_evaluate


def _ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": [1000] * n},
        index=idx,
    )


def _llm(action: str, conf: float = 0.8) -> TradeSignal:
    return TradeSignal(
        action=action,
        confidence=conf,
        reason="test",
        key_risks=[],
        expected_direction="UP" if action == "BUY" else "DOWN" if action == "SELL" else "FLAT",
        expected_magnitude_pct=0.5,
        horizon_bars=4,
        invalidation_price=99.0 if action == "BUY" else 101.0,
        raw_response={},
    )


# ---------------------------------------------------------------------------
# spec §15
# ---------------------------------------------------------------------------


def test_patterns_no_future_leak():
    """detect_swings must never reference a bar past lookback at the right edge."""
    df = _ohlcv(200, seed=1)
    lookback = 3
    swings_full, _ = detect_swings(df, lookback=lookback)

    # Trim the tail by exactly `lookback` bars and re-run. Every swing the
    # full run reported in the trimmed range should also appear in the
    # truncated run; nothing newer should silently appear because that would
    # mean the original used data after the truncation point.
    df_trim = df.iloc[: len(df) - lookback]
    swings_trim, _ = detect_swings(df_trim, lookback=lookback)
    full_in_trim = [s for s in swings_full if s.index < len(df_trim) - lookback]
    trim_indices = {s.index for s in swings_trim}
    for s in full_in_trim:
        assert s.index in trim_indices, (
            f"Swing at {s.index} present in full run but not in truncated — leak!"
        )


def test_triple_top_requires_neckline_break():
    """A triple-top CANDIDATE alone must not yield a SELL."""
    pattern = PatternResult(
        detected_pattern="TRIPLE_TOP_CANDIDATE",
        pattern_confidence=0.8,
        neckline=99.0,
        neckline_broken=False,            # explicit: not yet
        trend_state=TrendState.RANGE,
    )
    d = decide(
        technical_signal="SELL",
        pattern=pattern,
        higher_timeframe_trend="DOWNTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("SELL"),
    )
    assert d.action == "HOLD"
    assert "neckline" in d.reason.lower()


def test_triple_top_with_neckline_break_can_sell():
    """Same setup, but neckline broken → SELL allowed."""
    pattern = PatternResult(
        detected_pattern="TRIPLE_TOP_CANDIDATE",
        pattern_confidence=0.8,
        neckline=99.0,
        neckline_broken=True,
        trend_state=TrendState.RANGE,
    )
    d = decide(
        technical_signal="SELL",
        pattern=pattern,
        higher_timeframe_trend="DOWNTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("SELL"),
    )
    assert d.action == "SELL"


def test_fomc_forces_hold():
    """FOMC inside the 6h window forces HOLD even with perfect technicals."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    fomc = Event(
        when=now + timedelta(hours=2),
        currency="USD", title="FOMC Statement & Rate Decision", impact="high",
    )
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,
        risk_state=RiskState(events=(fomc,), now=now),
        llm_signal=_llm("BUY", conf=0.95),
    )
    assert d.action == "HOLD"
    assert "event_high" in d.blocked_by


def test_boj_forces_hold():
    """BOJ meeting inside the 6h window forces HOLD."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    boj = Event(
        when=now + timedelta(hours=3),
        currency="JPY", title="BOJ Policy Rate Decision", impact="high",
    )
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,
        risk_state=RiskState(events=(boj,), now=now),
        llm_signal=_llm("BUY"),
    )
    assert d.action == "HOLD"


def test_high_impact_event_forces_hold():
    """CPI / NFP within their 2h window also force HOLD."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    cpi = Event(when=now + timedelta(hours=1, minutes=30),
                currency="USD", title="CPI YoY", impact="high")
    d = decide(
        technical_signal="SELL",
        pattern=None,
        higher_timeframe_trend="DOWNTREND",
        risk_reward=2.0,
        risk_state=RiskState(events=(cpi,), now=now),
        llm_signal=_llm("SELL"),
    )
    assert d.action == "HOLD"


def test_sentiment_alone_never_trades():
    """If technical=HOLD, no amount of LLM enthusiasm flips us into BUY/SELL."""
    d = decide(
        technical_signal="HOLD",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=3.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.99),
    )
    assert d.action == "HOLD"


def test_sentiment_spike_raises_risk():
    """Mention spike + extreme velocity + crisis keyword should block via gate."""
    sentiment = {
        "mention_count_24h": 500,
        "sentiment_velocity": 0.9,
        "notable_posts": [{"text": "BREAKING fomc shock"}],
    }
    state = RiskState(sentiment_snapshot=sentiment)
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "sentiment_spike"


def test_data_missing_forces_hold():
    """Empty / too-short price frames force HOLD."""
    state = RiskState(df=pd.DataFrame())
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "data_quality"


def test_spread_abnormal_forces_hold():
    state = RiskState(spread_pct=0.5)  # 50 bp — clearly abnormal for FX majors
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "spread_abnormal"


def test_ai_cannot_override_risk_gate():
    """Even with a high-confidence LLM BUY, if the gate blocks, action stays HOLD."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    fomc = Event(when=now + timedelta(hours=1),
                 currency="USD", title="FOMC", impact="high")
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=5.0,
        risk_state=RiskState(events=(fomc,), now=now),
        llm_signal=_llm("BUY", conf=0.99),
    )
    assert d.action == "HOLD"
    assert d.advisory["llm_action"] == "BUY"
    assert d.advisory["llm_confidence"] == 0.99


def test_postmortem_classification_field_persists(tmp_path):
    """loss_category column must round-trip in the postmortems table."""
    from src.fx.storage import Storage

    storage = Storage(tmp_path / "fx.db")
    aid = storage.save_analysis(
        symbol="X", snapshot={"k": 1}, technical_signal="BUY", final_action="BUY",
    )
    pid = storage.save_prediction(
        analysis_id=aid, symbol="X", interval="1h", entry_price=1.0,
        action="BUY", confidence=0.7, reason="r",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=0.99,
    )
    storage.update_prediction_evaluation(
        pid, "WRONG", "DOWN", -0.4, 0.1, -0.6, True, "n",
    )
    storage.save_postmortem(
        prediction_id=pid,
        root_cause="EVENT_VOLATILITY",
        narrative="hit by surprise CPI",
        proposed_rule="extend CPI window to 3h",
        tags="cpi,event",
    )
    # Manually set the new spec §12 column
    with storage._conn() as conn:
        conn.execute(
            "UPDATE postmortems SET loss_category = ?, is_system_accident = 1 WHERE prediction_id = ?",
            ("B", pid),
        )
        row = conn.execute(
            "SELECT loss_category, is_system_accident FROM postmortems WHERE prediction_id = ?",
            (pid,),
        ).fetchone()
    assert row["loss_category"] == "B"
    assert row["is_system_accident"] == 1


# ---------------------------------------------------------------------------
# Extra defensive tests
# ---------------------------------------------------------------------------


def test_higher_tf_counter_trend_buy_blocked():
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="DOWNTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY"),
    )
    assert d.action == "HOLD"
    assert "counter-trend" in d.reason.lower()


def test_low_risk_reward_blocked():
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.2,        # below 1.5 floor
        risk_state=RiskState(),
        llm_signal=_llm("BUY"),
    )
    assert d.action == "HOLD"


def test_low_llm_confidence_blocked():
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.4),
    )
    assert d.action == "HOLD"


def test_clean_buy_passes_all_gates():
    """Sanity: with everything in alignment, the engine returns BUY."""
    d = decide(
        technical_signal="BUY",
        pattern=PatternResult(trend_state=TrendState.UPTREND),
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(),
        llm_signal=_llm("BUY", conf=0.8),
    )
    assert d.action == "BUY"
    assert d.confidence == 0.8


def test_no_llm_path_still_subject_to_gates():
    """No LLM at all should still be gated by the Risk Gate."""
    now = datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)
    fomc = Event(when=now, currency="USD", title="FOMC", impact="high")
    d = decide(
        technical_signal="BUY",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=2.0,
        risk_state=RiskState(events=(fomc,), now=now),
        llm_signal=None,
    )
    assert d.action == "HOLD"


def test_pattern_analyse_yields_structure_codes():
    """Sanity: analyse() returns swing_structure / market_structure on real data."""
    df = _ohlcv(200, seed=2)
    p = analyse(df, lookback=3)
    # On random data we expect at least a few swings to be found
    assert len(p.swing_highs) + len(p.swing_lows) >= 2
    for code in p.market_structure:
        assert code in ("HH", "HL", "LH", "LL")
