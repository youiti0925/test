"""Macro context: yields, dollar index, equity indices, volatility.

Pulls a curated set of "everything moves with this" series from
yfinance and aligns them to an arbitrary base index. Aligned values are
the *most recent value at or before* each base timestamp — strictly
point-in-time so the timeline / backtest never sees future data.

Symbols (all yfinance):
  ^TNX   US 10-year yield (×10; we divide back to %)
  ^IRX   US 13-week T-bill yield (proxy for short end)
  ^FVX   US 5-year yield (proxy for "2-year-ish" — yfinance lacks ^TUX)
  DX-Y.NYB    DXY dollar index
  ^N225  Nikkei 225
  ^GSPC  S&P 500
  ^IXIC  Nasdaq Composite
  ^VIX   CBOE volatility index

Why "^FVX" not the 2y? Yahoo doesn't expose a clean 2-year yield ticker
for free; the 5y is the closest reliably-sourced proxy. The field is
named `us_short_yield_proxy` so callers don't read it as the literal 2y.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

from .data import fetch_ohlcv


# Map of macro slot → yfinance symbol. Order matters: it's the order the
# fetcher tries (and the order the timeline emits columns).
MACRO_SYMBOLS: dict[str, str] = {
    "us10y": "^TNX",
    "us_short_yield_proxy": "^FVX",
    "dxy": "DX-Y.NYB",
    "nikkei": "^N225",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
}


@dataclass(frozen=True)
class MacroSnapshot:
    """Aligned macro series for a single base symbol+interval window.

    Each value series shares the base DataFrame's index. Missing series
    (fetch failed, market closed) are absent from `series` so callers
    can surface "data unavailable" rather than silent zeros.
    """

    base_index: pd.DatetimeIndex
    series: dict[str, pd.Series] = field(default_factory=dict)
    fetch_errors: dict[str, str] = field(default_factory=dict)

    def value_at(self, slot: str, ts: pd.Timestamp) -> float | None:
        s = self.series.get(slot)
        if s is None or s.empty:
            return None
        # Use 'pad' / asof to take the most recent value at or before ts.
        # `s.asof(ts)` returns NaN if no prior observation — convert to None.
        try:
            v = s.asof(ts)
        except (KeyError, TypeError):
            return None
        if pd.isna(v):
            return None
        return float(v)

    def yield_spread_long_short(self, ts: pd.Timestamp) -> float | None:
        long = self.value_at("us10y", ts)
        short = self.value_at("us_short_yield_proxy", ts)
        if long is None or short is None:
            return None
        return long - short

    def to_dict(self) -> dict:
        return {
            "slots": list(self.series.keys()),
            "missing": list(self.fetch_errors.keys()),
            "errors": dict(self.fetch_errors),
            "first_ts": (
                str(self.base_index[0]) if len(self.base_index) else None
            ),
            "last_ts": (
                str(self.base_index[-1]) if len(self.base_index) else None
            ),
        }


def fetch_macro_snapshot(
    base_index: pd.DatetimeIndex,
    *,
    interval: str = "1d",
    period: str = "2y",
    slots: Iterable[str] | None = None,
    fetcher: Callable = fetch_ohlcv,
) -> MacroSnapshot:
    """Fetch each requested macro slot and align to `base_index`.

    Parameters
    ----------
    base_index:
        DatetimeIndex of the primary instrument. Macro series are
        re-indexed onto this index using last-known-value forward-fill.
    interval:
        Macro fetch interval — daily is fine for most use cases. The
        timeline builder downsamples / aligns regardless.
    period:
        Macro fetch period; should comfortably cover `base_index`.
    slots:
        Subset of MACRO_SYMBOLS to fetch; default = all.
    """
    wanted = list(slots) if slots is not None else list(MACRO_SYMBOLS.keys())
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    for slot in wanted:
        sym = MACRO_SYMBOLS.get(slot)
        if sym is None:
            errors[slot] = f"unknown slot {slot!r}"
            continue
        try:
            df = fetcher(sym, interval=interval, period=period)
        except Exception as e:  # noqa: BLE001
            errors[slot] = f"fetch failed: {e}"
            continue
        if df.empty:
            errors[slot] = "fetch returned empty"
            continue
        s = df["close"].copy()
        # ^TNX, ^FVX, ^IRX are quoted ×10 (e.g. 4.25% shows as 42.5).
        # Bring them back to actual percent for human-friendly logs.
        if slot in ("us10y", "us_short_yield_proxy"):
            s = s / 10.0
        s.name = slot
        series[slot] = s

    return MacroSnapshot(base_index=base_index, series=series, fetch_errors=errors)
