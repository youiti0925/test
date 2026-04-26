"""OHLCV data fetching via yfinance.

Covers FX pairs (USDJPY=X), indices, equities, and crypto (BTC-USD).
For production FX, swap the fetch_ohlcv implementation with an OANDA or
broker-native client while keeping the same return shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    period: str = "60d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV dataframe indexed by timestamp.

    yfinance intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo.
    Intraday intervals <1h are capped at ~60 days of history upstream.
    """
    kwargs: dict = {"interval": interval, "auto_adjust": False, "progress": False}
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = yf.download(symbol, **kwargs)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} interval={interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "ts"
    return df


def latest_bars(df: pd.DataFrame, n: int = 50) -> list[Bar]:
    tail = df.tail(n)
    return [
        Bar(
            ts=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for idx, row in tail.iterrows()
    ]
