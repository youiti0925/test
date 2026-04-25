"""Point-in-time market timeline.

Stitches every information source the engine consumes onto a single
time-aligned table — one row per OHLCV bar, every column the value the
engine *would have seen* at that bar. Used downstream by:

  * Backtests that need fundamental context per bar
  * Event-overlay analysis (compute returns around CPI / FOMC / BOJ)
  * Waveform-similarity research (Phase C)
  * UI: "what was the world like at the moment of trade #237?"

Critical correctness rule
-------------------------
Every column at row[i] is computed from data with timestamp ≤ index[i].
We never reach forward. The pattern / technical / decision columns use
the same `df.iloc[: i + 1]` pattern as `backtest_engine`. Macro series
use `Series.asof(ts)`, which is forward-fill from the past only.

Static vs streaming snapshot fields
-----------------------------------
Some fields (sentiment, news, spread) are by nature streaming; we have
no historical archive yet. Those columns are populated only when the
caller passes a snapshot for the *current* bar. Backfill for older bars
stays None — better honest gaps than fake numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Iterable

import pandas as pd

from .calendar import Event
from .decision_engine import decide as decide_action
from .indicators import build_snapshot, technical_signal
from .macro import MacroSnapshot
from .patterns import analyse as analyse_patterns
from .risk import atr as compute_atr
from .risk_gate import RiskState


@dataclass(frozen=True)
class MarketTimelineRow:
    """One row of the timeline. Mirrors spec §5.

    Many columns are Optional because not every data source covers every
    bar (e.g. macro markets close, sentiment archive doesn't go back).
    Consumers MUST treat None as "unknown", not as zero.
    """

    timestamp: pd.Timestamp
    symbol: str
    timeframe: str

    open: float
    high: float
    low: float
    close: float
    volume: float | None

    technical_snapshot: dict
    technical_signal: str

    pattern_summary: dict
    swing_structure: list[str]
    market_structure: list[str]
    detected_pattern: str | None
    trend_state: str

    higher_timeframe_trend: str

    economic_events_nearby: list[dict]
    event_risk_level: str

    us10y: float | None
    us_short_yield_proxy: float | None
    yield_spread: float | None
    dxy: float | None
    nikkei: float | None
    sp500: float | None
    nasdaq: float | None
    vix: float | None

    sentiment_score: float | None
    sentiment_volume: int | None
    sentiment_keywords: list[str]

    bid: float | None
    ask: float | None
    spread_pct: float | None

    # Engine output for this bar (decision the production rules would have made)
    final_decision_action: str
    blocked_by: list[str]
    rule_chain: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat()
            if hasattr(self.timestamp, "isoformat")
            else str(self.timestamp),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": self.volume,
            "technical_signal": self.technical_signal,
            "technical_snapshot": self.technical_snapshot,
            "pattern_summary": self.pattern_summary,
            "swing_structure": self.swing_structure,
            "market_structure": self.market_structure,
            "detected_pattern": self.detected_pattern,
            "trend_state": self.trend_state,
            "higher_timeframe_trend": self.higher_timeframe_trend,
            "economic_events_nearby": self.economic_events_nearby,
            "event_risk_level": self.event_risk_level,
            "us10y": self.us10y,
            "us_short_yield_proxy": self.us_short_yield_proxy,
            "yield_spread": self.yield_spread,
            "dxy": self.dxy,
            "nikkei": self.nikkei,
            "sp500": self.sp500,
            "nasdaq": self.nasdaq,
            "vix": self.vix,
            "sentiment_score": self.sentiment_score,
            "sentiment_volume": self.sentiment_volume,
            "sentiment_keywords": self.sentiment_keywords,
            "bid": self.bid, "ask": self.ask, "spread_pct": self.spread_pct,
            "final_decision_action": self.final_decision_action,
            "blocked_by": self.blocked_by,
            "rule_chain": self.rule_chain,
        }


@dataclass
class MarketTimeline:
    rows: list[MarketTimelineRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Flatten to a DataFrame indexed by timestamp.

        Nested fields (lists, dicts) are kept as object columns — pandas
        handles that fine for parquet/csv export when stringified.
        """
        if not self.rows:
            return pd.DataFrame()
        records = [r.to_dict() for r in self.rows]
        df = pd.DataFrame.from_records(records)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    def __len__(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# Internal helpers — static event filter & higher-TF resample (offline)
# ---------------------------------------------------------------------------


def _events_within(events: tuple[Event, ...], ts: pd.Timestamp,
                   hours: float = 24.0) -> list[Event]:
    """Events whose `when` is within ±`hours` of `ts`."""
    if not events:
        return []
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    target = ts.to_pydatetime()
    return [
        e for e in events
        if abs((e.when - target).total_seconds()) <= hours * 3600
    ]


def _event_risk_label(events: list[Event]) -> str:
    """Promote LOW/MEDIUM/HIGH from the highest-impact event present."""
    if not events:
        return "LOW"
    impacts = [(e.impact or "medium").lower() for e in events]
    if any(i == "high" for i in impacts):
        return "HIGH"
    if any(i == "medium" for i in impacts):
        return "MEDIUM"
    return "LOW"


def _resample_higher_tf(df_window: pd.DataFrame, base_interval: str) -> str:
    rule = {
        "1m": "15min", "5m": "1h", "15m": "4h", "1h": "1D", "1d": "1W",
    }.get(base_interval)
    if rule is None or len(df_window) < 30:
        return "UNKNOWN"
    try:
        higher = df_window.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    if len(higher) < 10:
        return "UNKNOWN"
    return analyse_patterns(higher).trend_state.value


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_timeline(
    df: pd.DataFrame,
    symbol: str,
    *,
    interval: str = "1h",
    warmup: int = 50,
    events: Iterable[Event] = (),
    macro: MacroSnapshot | None = None,
    use_higher_tf: bool = True,
    sentiment_now: dict | None = None,
    spread_pct_now: float | None = None,
    bid_now: float | None = None,
    ask_now: float | None = None,
) -> MarketTimeline:
    """Build a point-in-time timeline from `df`.

    Parameters
    ----------
    df:
        OHLCV with tz-aware DatetimeIndex.
    interval:
        Base timeframe. Used for the higher-TF resample mapping.
    warmup:
        Number of leading bars to skip before the first row. Patterns
        and ATR need history to converge — and a too-early row would
        carry blank fields anyway.
    events:
        Calendar events. Each row only sees events within ±24h of its
        timestamp; the column lists those events as dicts.
    macro:
        Pre-fetched macro series aligned via `Series.asof`. Pass None
        to skip the macro columns (they'll all be None).
    sentiment_now / spread_pct_now / bid_now / ask_now:
        Streaming-only fields — only the most recent bar gets them.
        Older bars retain None because we don't have an archive.
    """
    timeline = MarketTimeline()
    events_tuple = tuple(events)
    atr_series = compute_atr(df, period=14)

    last_index = len(df) - 1

    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        bar = window.iloc[-1]

        snap = build_snapshot(symbol, window)
        tech = technical_signal(snap)
        pattern = analyse_patterns(window)

        higher_tf = (
            _resample_higher_tf(window, interval) if use_higher_tf else "UNKNOWN"
        )

        nearby = _events_within(events_tuple, ts, hours=24.0)
        event_risk = _event_risk_label(nearby)

        # Decision Engine view at this bar — keeps the timeline aligned
        # 1:1 with backtest_engine results.
        risk_state = RiskState(
            df=window,
            events=tuple(_events_within(events_tuple, ts, hours=48.0)),
            spread_pct=None,
            sentiment_snapshot=None,
            now=(
                ts.to_pydatetime()
                if ts.tzinfo
                else ts.tz_localize(timezone.utc).to_pydatetime()
            ),
        )
        atr_val = atr_series.iloc[i]
        rr = (
            1.5
            if (not pd.isna(atr_val) and atr_val > 0)
            else None
        )
        decision = decide_action(
            technical_signal=tech,
            pattern=pattern,
            higher_timeframe_trend=higher_tf,
            risk_reward=rr,
            risk_state=risk_state,
            llm_signal=None,
        )

        # Macro values — strictly point-in-time via asof.
        if macro is not None:
            us10y = macro.value_at("us10y", ts)
            short_y = macro.value_at("us_short_yield_proxy", ts)
            spread = (
                None if (us10y is None or short_y is None)
                else us10y - short_y
            )
            dxy = macro.value_at("dxy", ts)
            nikkei = macro.value_at("nikkei", ts)
            sp500 = macro.value_at("sp500", ts)
            nasdaq = macro.value_at("nasdaq", ts)
            vix = macro.value_at("vix", ts)
        else:
            us10y = short_y = spread = None
            dxy = nikkei = sp500 = nasdaq = vix = None

        # Streaming fields: only the most recent bar gets them — older
        # bars don't have archived sentiment/spread data.
        is_now = i == last_index
        sentiment_score = None
        sentiment_volume = None
        sentiment_kw: list[str] = []
        bid = ask = spread_pct = None
        if is_now:
            if sentiment_now:
                sentiment_score = sentiment_now.get("sentiment_score")
                sentiment_volume = sentiment_now.get("mention_count_24h")
                # Keywords: best-effort, reuse the same heuristic context.py uses.
                from .context import keywords_from_sentiment
                sentiment_kw = keywords_from_sentiment(sentiment_now)
            spread_pct = spread_pct_now
            bid = bid_now
            ask = ask_now

        timeline.rows.append(MarketTimelineRow(
            timestamp=ts,
            symbol=symbol,
            timeframe=interval,
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar["volume"]) if "volume" in bar else None,
            technical_snapshot=snap.to_dict(),
            technical_signal=tech,
            pattern_summary=pattern.to_dict(),
            swing_structure=list(pattern.swing_structure),
            market_structure=list(pattern.market_structure),
            detected_pattern=pattern.detected_pattern,
            trend_state=pattern.trend_state.value,
            higher_timeframe_trend=higher_tf,
            economic_events_nearby=[e.to_dict() for e in nearby],
            event_risk_level=event_risk,
            us10y=us10y,
            us_short_yield_proxy=short_y,
            yield_spread=spread,
            dxy=dxy,
            nikkei=nikkei,
            sp500=sp500,
            nasdaq=nasdaq,
            vix=vix,
            sentiment_score=sentiment_score,
            sentiment_volume=sentiment_volume,
            sentiment_keywords=sentiment_kw,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            final_decision_action=decision.action,
            blocked_by=list(decision.blocked_by),
            rule_chain=list(decision.rule_chain),
        ))

    return timeline


__all__ = [
    "MarketTimeline",
    "MarketTimelineRow",
    "build_timeline",
]
