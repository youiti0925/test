"""Pure formatting functions — given event data, produce a human-readable
notification body. No I/O, no LINE-specific markup (LINE renders plain
text, no markdown). Each formatter targets ~5-10 lines so push messages
stay scannable on mobile.
"""
from __future__ import annotations

from typing import Iterable


def format_signal(
    *,
    symbol: str,
    action: str,
    confidence: float,
    expected_direction: str,
    expected_magnitude_pct: float,
    horizon_bars: int,
    invalidation_price: float | None,
    reason: str,
    plan: dict | None = None,
    base_url: str | None = None,
    analysis_id: int | None = None,
) -> str:
    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "•")
    lines = [
        f"{icon} {symbol}  {action}  (conf {confidence:.2f})",
        f"  expects {expected_direction} {expected_magnitude_pct:.2f}% within {horizon_bars} bars",
    ]
    if invalidation_price is not None:
        lines.append(f"  invalidation: {invalidation_price}")
    if plan:
        lines.append(
            f"  entry {plan['entry']} · stop {plan['stop']} · TP {plan['take_profit']}"
        )
        lines.append(
            f"  size {plan['size']} · risk ${plan['risk_dollars']} · R:R {plan['reward_to_risk']}"
        )
    if reason:
        lines.append(f"  reason: {_clip(reason, 240)}")
    if base_url and analysis_id is not None:
        lines.append(f"  → {base_url}/analysis/{analysis_id}")
    return "\n".join(lines)


def format_verdict(
    *,
    symbol: str,
    action: str,
    status: str,
    actual_direction: str | None,
    actual_magnitude_pct: float | None,
    note: str | None,
    base_url: str | None = None,
    analysis_id: int | None = None,
) -> str:
    icon = {"CORRECT": "✅", "PARTIAL": "🟡", "WRONG": "❌"}.get(status, "•")
    lines = [
        f"{icon} {symbol}  {action} → {status}",
        f"  actual: {actual_direction or '—'} ({_fmt_pct(actual_magnitude_pct)})",
    ]
    if note:
        lines.append(f"  {_clip(note, 200)}")
    if base_url and analysis_id is not None:
        lines.append(f"  → {base_url}/analysis/{analysis_id}")
    return "\n".join(lines)


def format_lesson(
    *,
    symbol: str,
    root_cause: str,
    proposed_rule: str,
    narrative: str,
    base_url: str | None = None,
    analysis_id: int | None = None,
) -> str:
    lines = [
        f"🧠 New lesson — {symbol}  [{root_cause}]",
        f"  rule: {_clip(proposed_rule, 220)}",
        f"  why: {_clip(narrative, 220)}",
    ]
    if base_url and analysis_id is not None:
        lines.append(f"  → {base_url}/analysis/{analysis_id}")
    return "\n".join(lines)


def format_summary(
    *,
    period: str,
    counts: dict,
    sentiment: dict | None = None,
) -> str:
    lines = [f"📊 Summary — {period}"]
    if counts:
        lines.append(
            "  predictions: "
            + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        )
    if sentiment:
        sent = ", ".join(
            f"{sym} {s['sentiment_score']:+.2f}" for sym, s in sentiment.items()
        )
        if sent:
            lines.append(f"  sentiment: {sent}")
    return "\n".join(lines)


def format_command_help() -> str:
    return (
        "🤖 commands:\n"
        "  analyze <SYMBOL>     run a one-shot analysis\n"
        "  sentiment <SYMBOL>   show cached crowd sentiment\n"
        "  status               quick stats\n"
        "  lessons              recent post-mortems\n"
        "  subscribe            receive notifications\n"
        "  unsubscribe          stop notifications\n"
        "  help                 this menu"
    )


def _clip(text: str, max_len: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "—"
