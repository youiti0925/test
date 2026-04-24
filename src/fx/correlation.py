"""Multi-symbol correlation analysis.

Computes rolling return correlations for a primary symbol against a basket
of related instruments. The output is a small numeric dict suitable for
inclusion in the LLM analyst prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .data import fetch_ohlcv

# Curated default basket for each major symbol. Not exhaustive — enough for
# Claude to spot obvious confirmations or divergences. Extend as needed.
RELATED_SYMBOLS: dict[str, list[str]] = {
    "USDJPY=X": ["EURJPY=X", "GBPJPY=X", "AUDJPY=X", "DX-Y.NYB"],
    "EURUSD=X": ["GBPUSD=X", "USDCHF=X", "EURJPY=X", "DX-Y.NYB"],
    "GBPUSD=X": ["EURUSD=X", "EURGBP=X", "DX-Y.NYB"],
    "AUDUSD=X": ["NZDUSD=X", "USDCAD=X", "DX-Y.NYB"],
    "BTC-USD": ["ETH-USD", "SOL-USD", "^GSPC"],
    "ETH-USD": ["BTC-USD", "SOL-USD", "^GSPC"],
}


@dataclass(frozen=True)
class CorrelationSnapshot:
    primary: str
    period_days: int
    correlations: dict[str, float]
    last_change_pct_24h: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "period_days": self.period_days,
            "correlations": {
                k: round(v, 3) for k, v in self.correlations.items()
            },
            "last_change_pct_24h": {
                k: round(v, 3) for k, v in self.last_change_pct_24h.items()
            },
        }


def related_for(symbol: str) -> list[str]:
    return RELATED_SYMBOLS.get(symbol, [])


def _returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def build_correlation_snapshot(
    primary: str,
    others: list[str] | None = None,
    interval: str = "1d",
    period: str = "60d",
    fetcher: Callable[..., pd.DataFrame] = fetch_ohlcv,
) -> CorrelationSnapshot:
    """Fetch OHLCV for each symbol and compute return correlations.

    Missing or failed fetches are skipped; correlation is reported for the
    symbols that succeeded.
    """
    others = others if others is not None else related_for(primary)

    try:
        primary_df = fetcher(primary, interval=interval, period=period)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to fetch primary {primary}: {e}") from e
    primary_ret = _returns(primary_df)

    correlations: dict[str, float] = {}
    changes: dict[str, float] = {
        primary: _change_pct(primary_df, 1),
    }
    for sym in others:
        try:
            df = fetcher(sym, interval=interval, period=period)
        except Exception:  # noqa: BLE001
            continue
        other_ret = _returns(df)
        aligned = pd.concat([primary_ret, other_ret], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.notna(corr):
            correlations[sym] = float(corr)
            changes[sym] = _change_pct(df, 1)

    period_days = _estimate_period_days(primary_df)
    return CorrelationSnapshot(
        primary=primary,
        period_days=period_days,
        correlations=correlations,
        last_change_pct_24h=changes,
    )


def _change_pct(df: pd.DataFrame, lookback: int) -> float:
    close = df["close"]
    if len(close) <= lookback:
        return 0.0
    prev = close.iloc[-1 - lookback]
    last = close.iloc[-1]
    return float(100.0 * (last - prev) / prev) if prev else 0.0


def _estimate_period_days(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    span = df.index[-1] - df.index[0]
    return int(span.total_seconds() // 86400)
