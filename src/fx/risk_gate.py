"""Risk Gate — strict pre-checks that can ONLY produce HOLD.

The Decision Engine consults the gate first. Anything blocked here
results in HOLD regardless of what the technical signal, the LLM, or
the sentiment data say. AI cannot override this.

Gate ordering (spec §6, extended in §11/§12):
  1. data_quality       missing data / price anomaly / API key
  2. calendar_stale     events.json missing / stale / unreadable (live only)
  3. event_high         high-impact macro event in window
  4. spread_abnormal    spread blew out / unobserved on a live broker
  5. daily_loss_cap     cumulative loss above limit
  6. consecutive_losses too many losses in a row
  7. rule_unverified    rule version was changed too recently to trust
  8. sentiment_spike    crowd panic + crisis keywords

Each gate is a small, named function returning Maybe[BlockReason]. The
top-level `evaluate` runs them in order and returns the FIRST block.
This makes the cause traceable in logs and dashboards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from .calendar import CalendarFreshness, Event


@dataclass(frozen=True)
class BlockReason:
    """Why the gate forces a HOLD."""

    code: str        # data_quality | event_high | spread_abnormal | ...
    message: str     # short, human-readable
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    allow_trade: bool
    block: BlockReason | None
    blocked_codes: tuple[str, ...] = ()  # all gates that would block, for diagnostics

    def to_dict(self) -> dict:
        return {
            "allow_trade": self.allow_trade,
            "block": (
                {"code": self.block.code, "message": self.block.message,
                 "detail": self.block.detail}
                if self.block else None
            ),
            "blocked_codes": list(self.blocked_codes),
        }


# ---------------------------------------------------------------------------
# Per-gate checks
# ---------------------------------------------------------------------------


def check_data_quality(df: pd.DataFrame, min_bars: int = 50) -> BlockReason | None:
    if df is None or df.empty:
        return BlockReason("data_quality", "No price data")
    if len(df) < min_bars:
        return BlockReason(
            "data_quality",
            f"Only {len(df)} bars (need ≥ {min_bars})",
            {"bars": len(df)},
        )
    last_close = df["close"].iloc[-1]
    if not math.isfinite(last_close) or last_close <= 0:
        return BlockReason(
            "data_quality",
            f"Last close is not a valid positive number: {last_close}",
        )
    # NaNs in the latest 5 bars suggest stale or broken feed.
    if df[["open", "high", "low", "close"]].tail(5).isna().any().any():
        return BlockReason("data_quality", "Recent NaN in OHLC")
    return None


# Default windows per spec §6 (docs/DESIGN.md §6).
HIGH_IMPACT_WINDOWS_HOURS = {
    "FOMC": 6,
    "BOJ": 6,
    "ECB": 4,
    "BOE": 6,         # PR #18: spec is RATE_DECISION = ±6h, and BoE is
                      # a rate-decision body. Explicit BOE = 6h here.
    "CPI": 2,
    "PCE": 2,
    "NFP": 2,         # Non-farm payrolls
    "GDP": 2,
    "RATE_DECISION": 6,
    "INTERVENTION": 12,
}

# PR #18 — version stamp recorded in run_metadata.calendar so a future
# audit can reconcile event_high counts across schema/window changes.
EVENT_WINDOW_POLICY_VERSION = "v1"

# PR #18 — title alias map. Each canonical kind lists the substrings
# the title matcher should accept AFTER upper-casing AND replacing
# `-` / `_` with spaces. This is what lets "Non-Farm Payrolls" match
# NFP and "BoE Rate Decision (MPC)" match BOE — both of which the
# legacy substring matcher missed and silently fell through to the
# impact-based 4h fallback.
#
# Order matters: more specific keys come first so a title containing
# both "BOE" and "RATE DECISION" lands on BOE (the central-bank
# specific window).
_TITLE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FOMC",          ("FOMC", "FEDERAL OPEN MARKET COMMITTEE")),
    ("BOJ",           ("BOJ", "BANK OF JAPAN")),
    ("ECB",           ("ECB", "EUROPEAN CENTRAL BANK")),
    ("BOE",           ("BOE", "BANK OF ENGLAND")),
    ("CPI",           ("CPI", "CONSUMER PRICE")),
    ("PCE",           ("PCE", "PERSONAL CONSUMPTION EXPENDITURE")),
    ("NFP",           ("NFP",
                       "NON FARM PAYROLLS",
                       "NONFARM PAYROLLS",
                       "EMPLOYMENT SITUATION")),
    ("GDP",           ("GDP", "GROSS DOMESTIC PRODUCT")),
    ("INTERVENTION",  ("INTERVENTION",)),
    # RATE_DECISION is the catch-all for unbranded "Rate Decision" /
    # "Interest Rate Decision" titles. Checked LAST so a BoE-tagged
    # title routes to BOE first.
    ("RATE_DECISION", ("RATE DECISION", "INTEREST RATE DECISION")),
)


def infer_event_kind(event: Event) -> str | None:
    """Canonical kind for an event. PR #18.

    Resolution order:
    1. ``event.kind`` if it already names a known bucket.
    2. Title alias map (covers NFP / BoE / Rate Decision spellings the
       legacy substring matcher missed).

    Returns None when nothing matches — caller falls back to the
    impact-based window. Exposed for trace-stats / audit code.
    """
    explicit = (getattr(event, "kind", None) or "").upper()
    if explicit and explicit in HIGH_IMPACT_WINDOWS_HOURS:
        return explicit
    title_norm = (event.title or "").upper().replace("-", " ").replace("_", " ")
    for kind_canon, aliases in _TITLE_ALIASES:
        for alias in aliases:
            if alias in title_norm:
                return kind_canon
    return None


def _window_hours_for(event: Event) -> float:
    """HOLD window for an event in hours. PR-#18-revised resolution order:

    1. Explicit ``event.window_hours`` (a per-event override for future
       feeds that ship a known cadence).
    2. ``event.kind`` mapped through HIGH_IMPACT_WINDOWS_HOURS.
    3. Title alias map — covers "Non-Farm Payrolls" / "BoE Rate Decision"
       / "Rate Decision" that the legacy substring matcher missed.
    4. Impact-based fallback (high=4h / medium=2h / low=0.5h). Same as
       legacy, but very few real events should reach this branch now.

    Substantive change vs pre-PR-#18: NFP recovers its spec ±2h (was
    ±4h fallback); BoE Rate Decision gains ±6h (was ±4h fallback).
    FOMC / BOJ / ECB / CPI windows are unchanged.
    """
    explicit_window = getattr(event, "window_hours", None)
    if isinstance(explicit_window, (int, float)) and explicit_window > 0:
        return float(explicit_window)
    kind = infer_event_kind(event)
    if kind is not None:
        return float(HIGH_IMPACT_WINDOWS_HOURS[kind])
    impact = (event.impact or "medium").lower()
    if impact == "high":
        return 4.0
    if impact == "medium":
        return 2.0
    return 0.5


def check_high_impact_event(
    events: Iterable[Event],
    *,
    now: datetime | None = None,
) -> BlockReason | None:
    """Force HOLD if a HIGH-impact event falls inside its before/after window.

    The window is symmetric: an FOMC at 18:00 with a 6h window blocks
    trading from 12:00 the same day until 00:00 the next day.
    """
    now = now or datetime.now(timezone.utc)
    for ev in events or []:
        if (ev.impact or "medium").lower() != "high":
            continue
        window_h = _window_hours_for(ev)
        diff = abs((ev.when - now).total_seconds()) / 3600.0
        if diff <= window_h:
            return BlockReason(
                "event_high",
                f"{ev.title} ({ev.currency}) within ±{window_h:.0f}h "
                f"(now Δ={diff:.1f}h)",
                {
                    "event": ev.title,
                    "currency": ev.currency,
                    "window_hours": window_h,
                    "delta_hours": round(diff, 2),
                },
            )
    return None


def check_spread(
    spread_pct: float | None,
    *,
    max_pct: float = 0.05,
    require_spread: bool = False,
) -> BlockReason | None:
    """Block when spread (as % of price) is abnormally wide.

    `max_pct = 0.05` ≈ 5 bp; FX majors are usually <1 bp during liquid
    sessions. yfinance does not give bid/ask, so callers may pass None
    if the data is unavailable — we fall through silently in that case
    rather than constantly forcing HOLD.

    When `require_spread=True` (live / demo broker per spec §12), a missing
    spread_pct is itself a block: we refuse to trade without observing
    Bid/Ask, since we cannot then verify execution conditions.
    """
    if spread_pct is None:
        if require_spread:
            return BlockReason(
                "spread_unavailable",
                "Spread (Bid/Ask) unavailable; live broker requires observed spread",
            )
        return None
    if not math.isfinite(spread_pct):
        return BlockReason("spread_abnormal", f"Spread is non-finite: {spread_pct}")
    if spread_pct > max_pct:
        return BlockReason(
            "spread_abnormal",
            f"Spread {spread_pct:.4f}% exceeds {max_pct:.4f}%",
            {"spread_pct": spread_pct},
        )
    return None


def check_daily_loss_cap(
    pnl_today: float | None,
    *,
    cap: float,
) -> BlockReason | None:
    """Cumulative realised PnL today (negative numbers = losses) below cap → HOLD."""
    if pnl_today is None or cap is None:
        return None
    if pnl_today <= -abs(cap):
        return BlockReason(
            "daily_loss_cap",
            f"Today's PnL {pnl_today:.2f} ≤ −{abs(cap):.2f} cap",
            {"pnl_today": pnl_today, "cap": cap},
        )
    return None


def check_consecutive_losses(
    streak: int | None,
    *,
    cap: int = 3,
) -> BlockReason | None:
    if streak is None:
        return None
    if streak >= cap:
        return BlockReason(
            "consecutive_losses",
            f"{streak} consecutive losses ≥ {cap} cap",
            {"streak": streak, "cap": cap},
        )
    return None


def check_rule_unverified(
    rule_version_age_h: float | None,
    *,
    min_age_hours: float = 24.0,
) -> BlockReason | None:
    """A rule change less than `min_age_hours` ago hasn't been validated yet."""
    if rule_version_age_h is None:
        return None
    if rule_version_age_h < min_age_hours:
        return BlockReason(
            "rule_unverified",
            f"Rule changed {rule_version_age_h:.1f}h ago (< {min_age_hours}h)",
            {"rule_version_age_hours": rule_version_age_h},
        )
    return None


# Sentiment-driven elevation: a flood of panic / FOMC / 介入 keywords in the
# last hour bumps the gate even if no explicit event is on the calendar.
SENTIMENT_KEYWORD_TRIGGERS = (
    "fomc", "boj", "日銀", "cpi", "pce", "nfp", "rate decision",
    "intervention", "介入", "panic", "crash", "breaking",
)


def check_calendar_freshness(
    freshness: CalendarFreshness | None,
    *,
    require_fresh: bool = False,
) -> BlockReason | None:
    """Block when the economic-calendar file is stale, missing, or unreadable.

    Two operating modes (spec §11):
      - `require_fresh=False` (analyst / research): caller may still want
        to know the calendar is unhealthy, so we never block here. The
        warning lives in the freshness object itself.
      - `require_fresh=True`  (live / demo entry): a non-"fresh" calendar
        forces HOLD. Missing FOMC because the feed is two weeks old is
        exactly what we refuse to gamble on.
    """
    if not require_fresh or freshness is None:
        return None
    if freshness.is_fresh:
        return None
    return BlockReason(
        "calendar_stale",
        f"Calendar {freshness.status}: {freshness.detail or 'no detail'}",
        {
            "status": freshness.status,
            "age_hours": freshness.age_hours,
            "event_count": freshness.event_count,
            "max_age_hours": freshness.max_age_hours,
        },
    )


def check_sentiment_spike(
    sentiment_snapshot: dict | None,
    *,
    mention_count_threshold: int = 200,
    velocity_threshold: float = 0.6,
) -> BlockReason | None:
    """Heuristic gate: very high mention volume + extreme velocity = volatility risk."""
    if not sentiment_snapshot:
        return None
    mentions = sentiment_snapshot.get("mention_count_24h") or 0
    velocity = abs(sentiment_snapshot.get("sentiment_velocity") or 0.0)
    keywords_hit = []
    for post in sentiment_snapshot.get("notable_posts") or []:
        text = (post.get("text") or "").lower()
        for kw in SENTIMENT_KEYWORD_TRIGGERS:
            if kw in text:
                keywords_hit.append(kw)
    keywords_hit = list(set(keywords_hit))

    if mentions >= mention_count_threshold and velocity >= velocity_threshold:
        return BlockReason(
            "sentiment_spike",
            f"Mention spike ({mentions}) + velocity {velocity:.2f}; keywords: {keywords_hit}",
            {
                "mentions": mentions,
                "velocity": velocity,
                "keywords": keywords_hit,
            },
        )
    return None


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskState:
    """Bundle of mutable state the gate consults.

    All fields are optional — the corresponding check is simply skipped
    when the caller can't compute the value.
    """

    df: pd.DataFrame | None = None
    events: tuple[Event, ...] = ()
    spread_pct: float | None = None
    pnl_today: float | None = None
    daily_loss_cap: float | None = None
    consecutive_losses: int | None = None
    consecutive_losses_cap: int = 3
    rule_version_age_hours: float | None = None
    rule_min_age_hours: float = 24.0
    sentiment_snapshot: dict | None = None
    now: datetime | None = None
    # Calendar health view + whether to enforce it. Live/demo entry callers
    # set require_calendar_fresh=True; analyst/research paths leave it False
    # so a stale feed produces a warning, not a hard HOLD.
    calendar_freshness: CalendarFreshness | None = None
    require_calendar_fresh: bool = False
    # Spread enforcement. When True, a missing spread_pct (callers couldn't
    # observe Bid/Ask) blocks trading — required for live brokers per spec §12.
    require_spread: bool = False


def evaluate(state: RiskState) -> GateResult:
    """Run every gate; return on the FIRST block.

    Spec §6 ordering: data_quality → event_high → spread_abnormal →
    daily_loss_cap → consecutive_losses → rule_unverified.
    """
    checks: list[BlockReason | None] = [
        check_data_quality(state.df) if state.df is not None else None,
        check_calendar_freshness(
            state.calendar_freshness, require_fresh=state.require_calendar_fresh
        ),
        check_high_impact_event(state.events, now=state.now),
        check_spread(state.spread_pct, require_spread=state.require_spread),
        check_daily_loss_cap(state.pnl_today, cap=state.daily_loss_cap)
        if state.daily_loss_cap is not None else None,
        check_consecutive_losses(state.consecutive_losses,
                                 cap=state.consecutive_losses_cap),
        check_rule_unverified(state.rule_version_age_hours,
                              min_age_hours=state.rule_min_age_hours),
        check_sentiment_spike(state.sentiment_snapshot),
    ]
    blocks = [c for c in checks if c is not None]
    if blocks:
        return GateResult(
            allow_trade=False,
            block=blocks[0],
            blocked_codes=tuple(b.code for b in blocks),
        )
    return GateResult(allow_trade=True, block=None, blocked_codes=())
