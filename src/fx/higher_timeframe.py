"""Higher-timeframe context.

For an intraday signal on (say) the 1h chart, the engine needs to know
whether the daily/4h is up, down, or ranging. We re-fetch a longer
window at the higher interval and run the same swing analysis.

`infer_higher_interval` is intentionally simple — bumps roughly one
order of magnitude. Override per pair if needed.
"""
from __future__ import annotations

from typing import Callable

from .data import fetch_ohlcv
from .patterns import TrendState, analyse


HIGHER_INTERVAL_MAP = {
    "1m": "15m",
    "5m": "1h",
    "15m": "4h",
    "30m": "4h",
    "1h": "1d",
    "2h": "1d",
    "4h": "1wk",
    "1d": "1wk",
}


def infer_higher_interval(interval: str) -> str:
    return HIGHER_INTERVAL_MAP.get(interval, "1d")


def fetch_and_classify(
    symbol: str,
    interval: str,
    *,
    period: str = "180d",
    fetcher: Callable = fetch_ohlcv,
) -> str:
    """Return the trend label of the higher timeframe (or 'UNKNOWN' on failure)."""
    higher = infer_higher_interval(interval)
    try:
        df = fetcher(symbol, interval=higher, period=period)
        if df.empty or len(df) < 30:
            return "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    pattern = analyse(df)
    return pattern.trend_state.value
