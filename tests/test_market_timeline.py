"""Tests for the point-in-time market timeline (spec §5)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.calendar import Event
from src.fx.macro import MacroSnapshot
from src.fx.market_timeline import build_timeline


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


def test_timeline_row_count_matches_warmup_offset():
    df = _ohlcv(200)
    tl = build_timeline(df, symbol="X", interval="1h", warmup=60)
    assert len(tl) == len(df) - 60


def test_timeline_no_future_leak_in_decision_column():
    """Trimming the tail must not change earlier decision values."""
    df = _ohlcv(300, seed=7)
    full = build_timeline(df, symbol="X", interval="1h", warmup=60)
    trim = build_timeline(df.iloc[:-50], symbol="X", interval="1h", warmup=60)

    full_by_ts = {r.timestamp: r for r in full.rows}
    for r in trim.rows:
        assert r.timestamp in full_by_ts
        full_r = full_by_ts[r.timestamp]
        assert r.final_decision_action == full_r.final_decision_action, (
            f"Decision at {r.timestamp} differs between full and trimmed runs "
            "— future-leak in build_timeline."
        )
        assert r.technical_signal == full_r.technical_signal


def test_timeline_macro_columns_use_asof():
    df = _ohlcv(200, seed=8)
    # Build a synthetic MacroSnapshot whose value flips after a known cutoff
    cutoff = df.index[100]
    s = pd.Series(
        [1.0] * 100 + [2.0] * (len(df) - 100), index=df.index, name="us10y",
    )
    macro = MacroSnapshot(
        base_index=df.index, series={"us10y": s}, fetch_errors={},
    )

    tl = build_timeline(
        df, symbol="X", interval="1h", warmup=60, macro=macro,
    )
    # Rows before cutoff must see 1.0; rows at/after cutoff must see 2.0.
    for r in tl.rows:
        if r.timestamp < cutoff:
            assert r.us10y == 1.0
        else:
            assert r.us10y == 2.0


def test_timeline_streaming_fields_only_on_last_row():
    df = _ohlcv(200, seed=9)
    sentiment = {
        "sentiment_score": 0.4, "mention_count_24h": 100,
        "notable_posts": [{"text": "fomc panic"}],
    }
    tl = build_timeline(
        df, symbol="X", interval="1h", warmup=60,
        sentiment_now=sentiment, spread_pct_now=0.02,
        bid_now=99.5, ask_now=99.6,
    )
    assert tl.rows[0].sentiment_score is None
    assert tl.rows[0].spread_pct is None
    assert tl.rows[-1].sentiment_score == 0.4
    assert tl.rows[-1].spread_pct == 0.02
    assert tl.rows[-1].bid == 99.5
    assert tl.rows[-1].ask == 99.6


def test_timeline_event_risk_label():
    df = _ohlcv(200, seed=10)
    # FOMC right in the middle, ±24h window.
    mid = df.index[100].to_pydatetime()
    fomc = Event(when=mid, currency="USD", title="FOMC", impact="high")
    tl = build_timeline(
        df, symbol="USDJPY=X", interval="1h", warmup=60, events=(fomc,),
    )
    near = [r for r in tl.rows if abs((r.timestamp - mid).total_seconds()) <= 23 * 3600]
    far = [r for r in tl.rows if abs((r.timestamp - mid).total_seconds()) > 25 * 3600]
    assert any(r.event_risk_level == "HIGH" for r in near)
    assert all(r.event_risk_level == "LOW" for r in far)


def test_timeline_to_frame_indexed_by_timestamp():
    df = _ohlcv(150, seed=11)
    tl = build_timeline(df, symbol="X", interval="1h", warmup=60)
    frame = tl.to_frame()
    assert frame.index.name == "timestamp"
    assert "close" in frame.columns
    assert "final_decision_action" in frame.columns
    assert len(frame) == len(tl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
