"""Event-window price-action overlay (spec §6).

Given a list of macro events (FOMC, BOJ, CPI, NFP, ...) and an OHLCV
DataFrame, compute how price moved before / after each one. Output is
a dataframe of one row per event plus aggregate statistics — exactly
what you want to feed into "FOMC前後の値動き" research.

What's NOT in here (deliberately)
---------------------------------
* No "actual minus forecast" surprise calculation. yfinance / events.json
  don't reliably carry forecasts at past timestamps; you'd need a paid
  econ feed to do this honestly. The forecast/previous fields on Event
  are passed through unchanged so a future caller can compute surprise
  if they have the data.
* No "the cause was the event" claim. Per spec §17, this module
  *correlates* price-action with event timestamps; it does not
  attribute. attribution.py (Phase D) is where attribution lives.

Future-leak rules
-----------------
* `before_*_return` only uses bars with timestamp < event.when.
* `after_*_return` only uses bars with timestamp ≥ event.when.
* Aggregates are computed over the per-event rows directly — no
  forward-looking aggregation. A forthcoming event for which the after-
  window hasn't elapsed yet contributes None and is excluded from the
  averages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from .calendar import Event
from .risk import atr as compute_atr


# Default windows we measure. Spec §6 lists these explicitly.
DEFAULT_BEFORE_WINDOWS = (("before_30m", 30 / 60), ("before_2h", 2.0))
DEFAULT_AFTER_WINDOWS = (
    ("after_30m", 30 / 60),
    ("after_1h", 1.0),
    ("after_4h", 4.0),
    ("after_24h", 24.0),
)


@dataclass(frozen=True)
class EventWindowRow:
    event_when: datetime
    event_currency: str
    event_title: str
    event_impact: str
    event_keyword: str | None         # CPI / FOMC / NFP / ...
    pre_price: float | None           # close immediately before event
    post_price: float | None          # close immediately after event
    returns: dict[str, float | None]  # window-name → percent return
    max_favorable_after_pct: float | None
    max_adverse_after_pct: float | None
    atr_multiple_after: float | None  # |return_24h| / atr_at_event
    volatility_spike: bool            # max_adverse - max_favorable > 2*ATR

    def to_dict(self) -> dict:
        return {
            "event_when": self.event_when.isoformat(),
            "event_currency": self.event_currency,
            "event_title": self.event_title,
            "event_impact": self.event_impact,
            "event_keyword": self.event_keyword,
            "pre_price": self.pre_price,
            "post_price": self.post_price,
            "returns": dict(self.returns),
            "max_favorable_after_pct": self.max_favorable_after_pct,
            "max_adverse_after_pct": self.max_adverse_after_pct,
            "atr_multiple_after": self.atr_multiple_after,
            "volatility_spike": self.volatility_spike,
        }


@dataclass
class EventOverlayResult:
    rows: list[EventWindowRow] = field(default_factory=list)
    keyword_filter: str | None = None
    impact_filter: str | None = None

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        records = []
        for r in self.rows:
            d = r.to_dict()
            for win, val in d.pop("returns").items():
                d[f"return_{win}"] = val
            records.append(d)
        return pd.DataFrame.from_records(records)

    def aggregate(self) -> dict:
        """Mean / median / hit-rate per measured window."""
        df = self.to_frame()
        if df.empty:
            return {"n_events": 0, "windows": {}}

        windows = {}
        for col in df.columns:
            if not col.startswith("return_"):
                continue
            vals = df[col].dropna()
            if vals.empty:
                continue
            up = (vals > 0).sum()
            down = (vals < 0).sum()
            windows[col] = {
                "n": int(vals.count()),
                "mean_pct": float(vals.mean()),
                "median_pct": float(vals.median()),
                "std_pct": float(vals.std()) if len(vals) > 1 else 0.0,
                "share_up": float(up / vals.count()),
                "share_down": float(down / vals.count()),
            }

        atr_mult = df["atr_multiple_after"].dropna()
        return {
            "n_events": len(df),
            "n_volatility_spikes": int(df["volatility_spike"].sum()),
            "atr_multiple_after_mean": (
                float(atr_mult.mean()) if not atr_mult.empty else None
            ),
            "windows": windows,
            "filters": {
                "keyword": self.keyword_filter,
                "impact": self.impact_filter,
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


KEYWORD_MAP = (
    ("FOMC", ("FOMC",)),
    ("FOMC_MINUTES", ("FOMC MINUTES",)),
    ("BOJ", ("BOJ", "日銀")),
    ("ECB", ("ECB",)),
    ("CPI", ("CPI",)),
    ("PCE", ("PCE",)),
    ("NFP", ("NFP", "NONFARM", "EMPLOYMENT")),
    ("GDP", ("GDP",)),
    ("RETAIL_SALES", ("RETAIL SALES",)),
    ("ISM", ("ISM",)),
    ("RATE_DECISION", ("RATE DECISION",)),
    ("INTERVENTION", ("INTERVENTION", "介入")),
)


def event_keyword(event: Event) -> str | None:
    """Map an event title onto our closed taxonomy of keywords."""
    upper = (event.title or "").upper()
    for key, needles in KEYWORD_MAP:
        for n in needles:
            if n in upper:
                return key
    return None


def _ensure_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    return t


def _close_at_or_before(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Last close at or before `target`, or None if none."""
    sub = df.loc[df.index <= target]
    if sub.empty:
        return None
    return sub.index[-1], float(sub["close"].iloc[-1])


def _close_at_or_after(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    sub = df.loc[df.index >= target]
    if sub.empty:
        return None
    return sub.index[0], float(sub["close"].iloc[0])


def _window_return(df: pd.DataFrame, anchor: pd.Timestamp,
                   delta: timedelta, *, side: str) -> float | None:
    """Percent return between `anchor` and `anchor +/- delta` close.

    side="before": return = (anchor_close - prior_close) / prior_close
    side="after":  return = (target_close - anchor_close) / anchor_close
    """
    a = _close_at_or_after(df, anchor) if side == "after" else _close_at_or_before(df, anchor)
    if a is None:
        return None
    if side == "after":
        target = anchor + delta
        b = _close_at_or_after(df, target)
        if b is None:
            return None
        anchor_price = a[1]
        target_price = b[1]
    else:
        target = anchor - delta
        b = _close_at_or_before(df, target)
        if b is None:
            return None
        anchor_price = b[1]
        target_price = a[1]
    if anchor_price == 0:
        return None
    return 100.0 * (target_price - anchor_price) / anchor_price


def _max_excursion_after(
    df: pd.DataFrame, anchor: pd.Timestamp, hours: float
) -> tuple[float | None, float | None]:
    """Max favorable (high) and adverse (low) percent move within the window."""
    end = anchor + timedelta(hours=hours)
    sub = df.loc[(df.index >= anchor) & (df.index <= end)]
    if sub.empty:
        return None, None
    a = _close_at_or_after(df, anchor)
    if a is None:
        return None, None
    base = a[1]
    if base == 0:
        return None, None
    favorable = 100.0 * (sub["high"].max() - base) / base
    adverse = 100.0 * (sub["low"].min() - base) / base
    return float(favorable), float(adverse)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overlay_events(
    df: pd.DataFrame,
    events: Iterable[Event],
    *,
    keyword: str | None = None,
    impact: str | None = "high",
    before_windows: tuple[tuple[str, float], ...] = DEFAULT_BEFORE_WINDOWS,
    after_windows: tuple[tuple[str, float], ...] = DEFAULT_AFTER_WINDOWS,
) -> EventOverlayResult:
    """Compute pre/post returns for each `event` in `df`.

    Parameters
    ----------
    df:
        OHLCV with tz-aware DatetimeIndex.
    events:
        Macro events to anchor on.
    keyword:
        Restrict to a single keyword from KEYWORD_MAP (e.g. "CPI"). None
        keeps all events.
    impact:
        Restrict to events of this impact. Default "high"; pass None to
        keep medium/low too.
    before_windows / after_windows:
        Iterable of (label, hours). Returns are reported per label.
    """
    if df.index.tzinfo is None:
        df = df.tz_localize("UTC")

    atr_series = compute_atr(df, period=14)

    rows: list[EventWindowRow] = []
    for ev in events:
        kw = event_keyword(ev)
        if keyword is not None and kw != keyword:
            continue
        if impact is not None and (ev.impact or "medium").lower() != impact.lower():
            continue

        anchor = _ensure_utc(ev.when)
        # Event must lie within the df's range to be measurable.
        if anchor < df.index[0] or anchor > df.index[-1]:
            continue

        pre = _close_at_or_before(df, anchor)
        post = _close_at_or_after(df, anchor)

        returns: dict[str, float | None] = {}
        for name, hours in before_windows:
            returns[name] = _window_return(
                df, anchor, timedelta(hours=hours), side="before"
            )
        for name, hours in after_windows:
            returns[name] = _window_return(
                df, anchor, timedelta(hours=hours), side="after"
            )

        # Use 24h as the canonical reference for max excursion + ATR multiple
        max_fav, max_adv = _max_excursion_after(df, anchor, 24.0)

        # ATR at event time — most recent value at or before anchor
        atr_at_event = None
        atr_sub = atr_series.loc[atr_series.index <= anchor].dropna()
        if not atr_sub.empty:
            atr_at_event = float(atr_sub.iloc[-1])

        atr_multiple = None
        if atr_at_event and post is not None and pre is not None:
            ret24 = returns.get("after_24h")
            if ret24 is not None and post[1] != 0:
                # |close_24h_after - close_at_event| / atr_at_event in price units
                close_at = post[1]
                close_24h = close_at * (1 + ret24 / 100.0)
                if atr_at_event > 0:
                    atr_multiple = abs(close_24h - close_at) / atr_at_event

        # Volatility spike heuristic: round-trip > 2*ATR within the
        # 24h after-window.
        spike = False
        if (atr_at_event and atr_at_event > 0
                and max_fav is not None and max_adv is not None
                and pre is not None):
            base = pre[1]
            roundtrip_price = (max_fav - max_adv) / 100.0 * base
            spike = roundtrip_price > 2.0 * atr_at_event

        rows.append(EventWindowRow(
            event_when=anchor.to_pydatetime(),
            event_currency=ev.currency,
            event_title=ev.title,
            event_impact=ev.impact or "medium",
            event_keyword=kw,
            pre_price=pre[1] if pre else None,
            post_price=post[1] if post else None,
            returns=returns,
            max_favorable_after_pct=max_fav,
            max_adverse_after_pct=max_adv,
            atr_multiple_after=atr_multiple,
            volatility_spike=spike,
        ))

    return EventOverlayResult(
        rows=rows, keyword_filter=keyword, impact_filter=impact,
    )


__all__ = [
    "EventWindowRow",
    "EventOverlayResult",
    "overlay_events",
    "event_keyword",
    "KEYWORD_MAP",
]
