"""fundamental_sidebar_v1 — right-side fundamental / event-risk panel.

Aggregates the existing event calendar + macro-alignment + risk_gate
outputs into a single dict the visual_audit right sidebar can render.

The panel reuses everything Phase A–F already built:
  - Event window resolution (risk_gate._window_hours_for + _TITLE_ALIASES)
  - Symbol→currency mapping (calendar.currencies_for)
  - MacroAlignmentSnapshot (DXY / yields / VIX)
  - FundamentalSlice carriers (nearby / blocking / warning event lists)

What's new is the `event_risk_status` four-state:

  CLEAR    — no high-impact event within its HOLD window
  WARNING  — medium-impact event within its window, OR high-impact
             outside HOLD window but inside warning window (2× HOLD)
  BLOCK    — high-impact event inside its HOLD window
             (entry_plan READY → action becomes WAIT_EVENT_CLEAR)
  UNKNOWN  — no event feed available

`now_trade_allowed` is False whenever event_risk_status is BLOCK.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final, Iterable

from .calendar import Event, currencies_for
from .risk_gate import (
    HIGH_IMPACT_WINDOWS_HOURS,
    _window_hours_for,
    infer_event_kind,
)


SCHEMA_VERSION: Final[str] = "fundamental_sidebar_v1"


# Status tokens
CLEAR: Final[str] = "CLEAR"
WARNING: Final[str] = "WARNING"
BLOCK: Final[str] = "BLOCK"
UNKNOWN: Final[str] = "UNKNOWN"


def _event_to_dict(
    ev: Event, *, now: datetime, status: str, window_h: float,
) -> dict:
    """Render an Event as a sidebar-friendly dict (no Event class
    references in the rendering path)."""
    diff_h = (ev.when - now).total_seconds() / 3600.0
    kind = infer_event_kind(ev) or ""
    when_iso = (
        ev.when.isoformat()
        if isinstance(ev.when, datetime)
        else str(ev.when)
    )
    return {
        "title": ev.title,
        "currency": ev.currency,
        "kind": kind,
        "impact": ev.impact,
        "when": when_iso,
        "minutes_until": int(round(diff_h * 60)),
        "window_hours": float(window_h),
        "status": status,
        "in_block_window": status == BLOCK,
        "in_warning_window": status == WARNING,
    }


def _classify_one(
    ev: Event, *, now: datetime,
) -> tuple[str, float]:
    """Return (status, window_hours_used)."""
    window_h = _window_hours_for(ev)
    diff_h = abs((ev.when - now).total_seconds()) / 3600.0
    impact = (ev.impact or "medium").lower()
    if impact == "high":
        if diff_h <= window_h:
            return BLOCK, window_h
        if diff_h <= 2.0 * window_h:
            return WARNING, window_h
        return CLEAR, window_h
    if impact == "medium":
        if diff_h <= window_h:
            return WARNING, window_h
        return CLEAR, window_h
    # low impact — never blocks; warning only inside its window
    if diff_h <= window_h:
        return WARNING, window_h
    return CLEAR, window_h


def _macro_drivers_from_alignment(
    macro_alignment: dict | None,
) -> list[dict]:
    """Convert macro_alignment dict into sidebar driver rows."""
    if not macro_alignment:
        return []
    out: list[dict] = []
    for key, label_ja in (
        ("dxy_alignment", "DXY"),
        ("yield_alignment", "米10年金利"),
        ("vix_regime", "VIX"),
        ("currency_bias", "通貨ペア合成バイアス"),
    ):
        v = macro_alignment.get(key)
        if v is None:
            continue
        out.append({
            "name": key,
            "label_ja": label_ja,
            "value": str(v),
        })
    score = macro_alignment.get("macro_score")
    if score is not None:
        out.append({
            "name": "macro_score",
            "label_ja": "マクロスコア (-1.0〜+1.0)",
            "value": f"{float(score):+.2f}",
        })
    return out


def _missing_data_from_macro(
    macro_alignment: dict | None,
    macro_context: dict | None,
) -> list[dict]:
    """Detect missing-data conditions and label them explicitly."""
    out: list[dict] = []
    align = macro_alignment or {}
    ctx = macro_context or {}

    def _label(key: str, label_ja: str) -> None:
        out.append({
            "name": key,
            "label_ja": label_ja,
            "status": "未接続",
        })

    if not ctx:
        _label("macro_context", "マクロ実データ feed")
    if ctx.get("dxy_trend_5d_bucket") in (None, ""):
        _label("dxy", "DXY")
    if ctx.get("us10y_change_24h_bp") is None:
        _label("us10y_yield", "米10年金利")
    if ctx.get("us2y_yield") is None:
        _label("us2y_yield", "米2年金利")
    if ctx.get("vix") is None:
        _label("vix", "VIX")
    if not align.get("event_tone"):
        _label("news_calendar", "ニュース / 経済カレンダー (event_tone)")
    return out


def build_fundamental_sidebar(
    *,
    symbol: str,
    now: datetime | None,
    events: Iterable[Event] | None,
    macro_alignment: dict | None,
    macro_context: dict | None,
    final_action: str | None = None,
    entry_status: str | None = None,
    within_hours: int = 24,
) -> dict:
    """Build the fundamental_sidebar_v1 dict.

    `events` may be None when the calendar feed is unavailable — in
    that case event_risk_status becomes UNKNOWN.
    """
    now_dt = now or datetime.now(timezone.utc)

    if events is None:
        # No feed — surface as UNKNOWN, do NOT auto-permit trade
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": symbol,
            "timestamp": now_dt.isoformat(),
            "event_risk_status": UNKNOWN,
            "now_trade_allowed": False,
            "reason_ja": (
                "経済イベント feed が未接続のため、イベントリスクを判定"
                "できません。新規エントリーは見送りを推奨します。"
            ),
            "nearby_events": [],
            "blocking_events": [],
            "warning_events": [],
            "macro_drivers": _macro_drivers_from_alignment(macro_alignment),
            "missing_data": (
                _missing_data_from_macro(macro_alignment, macro_context)
                + [{"name": "event_calendar", "label_ja": "経済カレンダー",
                    "status": "未接続"}]
            ),
            "right_panel_summary_ja": (
                "イベントカレンダー未接続。テクニカルが整っていても、"
                "FOMC / BOJ / CPI / NFP の有無を別チャネルで確認して"
                "ください。"
            ),
            "final_action": final_action,
            "entry_status": entry_status,
        }

    # Filter events relevant to symbol
    sym_currencies = set(currencies_for(symbol))
    relevant: list[Event] = []
    for ev in events:
        if not sym_currencies or ev.currency in sym_currencies:
            relevant.append(ev)

    blocking: list[dict] = []
    warning: list[dict] = []
    nearby: list[dict] = []

    for ev in relevant:
        status, window_h = _classify_one(ev, now=now_dt)
        ev_dict = _event_to_dict(
            ev, now=now_dt, status=status, window_h=window_h,
        )
        if status == BLOCK:
            blocking.append(ev_dict)
        elif status == WARNING:
            warning.append(ev_dict)
        # nearby = anything within next `within_hours`
        if 0 <= (ev.when - now_dt).total_seconds() / 3600.0 <= within_hours:
            nearby.append(ev_dict)

    # Aggregate event_risk_status
    if blocking:
        event_risk_status = BLOCK
    elif warning:
        event_risk_status = WARNING
    else:
        event_risk_status = CLEAR

    now_trade_allowed = event_risk_status != BLOCK

    if event_risk_status == BLOCK and blocking:
        first = blocking[0]
        mins = first["minutes_until"]
        if mins >= 0:
            timing_ja = f"{mins // 60} 時間 {mins % 60} 分後"
        else:
            timing_ja = f"{abs(mins) // 60} 時間 {abs(mins) % 60} 分前"
        reason_ja = (
            f"{first['title']} ({first['kind'] or first['impact']}) が "
            f"{timing_ja}にあるため新規エントリー禁止です "
            f"(±{first['window_hours']:.0f}h ウィンドウ)。"
        )
        right_panel_summary_ja = (
            f"現在はイベントリスクが高い ({first['title']}) ため、"
            "テクニカルが良くても新規エントリーは避けます。"
        )
    elif event_risk_status == WARNING and warning:
        first = warning[0]
        reason_ja = (
            f"{first['title']} ({first['kind'] or first['impact']}) が"
            f"接近中。caution として扱います。"
        )
        right_panel_summary_ja = (
            "イベントリスク警戒。サイズを抑えるか様子見を検討してください。"
        )
    else:
        reason_ja = "直近のイベントリスクは検出されません。"
        right_panel_summary_ja = (
            "イベントリスクは現時点でクリア。テクニカル判断に従って構いません。"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "timestamp": now_dt.isoformat(),
        "event_risk_status": event_risk_status,
        "now_trade_allowed": now_trade_allowed,
        "reason_ja": reason_ja,
        "nearby_events": nearby,
        "blocking_events": blocking,
        "warning_events": warning,
        "macro_drivers": _macro_drivers_from_alignment(macro_alignment),
        "missing_data": _missing_data_from_macro(
            macro_alignment, macro_context,
        ),
        "right_panel_summary_ja": right_panel_summary_ja,
        "final_action": final_action,
        "entry_status": entry_status,
    }


__all__ = [
    "SCHEMA_VERSION",
    "CLEAR",
    "WARNING",
    "BLOCK",
    "UNKNOWN",
    "build_fundamental_sidebar",
]
