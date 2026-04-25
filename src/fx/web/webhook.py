"""LINE webhook command parser.

Pure dispatch logic, kept separate from the Flask route so it can be
unit-tested without spinning up a server.

Supported commands (case-insensitive, leading slash optional):
  help                         show menu
  subscribe                    register sender for notifications
  unsubscribe                  remove sender
  status                       quick stats (analyses, predictions, lessons)
  analyze <SYMBOL>             trigger an analysis (best-effort)
  sentiment <SYMBOL>           show cached crowd-sentiment summary
  lessons [<SYMBOL>]           recent post-mortems
  predictions [<status>]       recent predictions, optional status filter
"""
from __future__ import annotations

import json
from typing import Callable

from src.fx.sentiment import read_for as read_sentiment
from src.fx.storage import Storage
from src.notify.formatters import format_command_help


def handle_command(
    text: str,
    *,
    storage: Storage,
    user_id: str,
    backend: str = "line",
    display_name: str | None = None,
    run_analyze: Callable[[str], dict] | None = None,
) -> str:
    """Return the reply text for a single user command."""
    if not text:
        return format_command_help()

    parts = text.strip().lstrip("/").split()
    if not parts:
        return format_command_help()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("help", "h", "?", "menu"):
        return format_command_help()

    if cmd == "subscribe":
        storage.add_subscriber(backend=backend, user_id=user_id, display_name=display_name)
        return "✅ Subscribed. You'll get push notifications for new signals, verdicts, and lessons."

    if cmd == "unsubscribe":
        ok = storage.deactivate_subscriber(backend=backend, user_id=user_id)
        return "👋 Unsubscribed." if ok else "Not currently subscribed."

    if cmd == "status":
        stats = storage.dashboard_stats()
        by = stats.get("predictions_by_status") or {}
        return (
            "📊 Status\n"
            f"  analyses: {stats.get('total_analyses', 0)}\n"
            f"  predictions: pending={by.get('PENDING', 0)} "
            f"correct={by.get('CORRECT', 0)} "
            f"wrong={by.get('WRONG', 0)} "
            f"partial={by.get('PARTIAL', 0)}\n"
            f"  lessons: {stats.get('postmortems', 0)}\n"
            f"  win rate: {stats.get('weighted_win_rate', 0)*100:.1f}%"
        )

    if cmd == "lessons":
        symbol = args[0] if args else None
        if symbol:
            items = storage.relevant_postmortems(symbol, limit=3)
        else:
            items = storage.list_postmortems(limit=3)
        if not items:
            return "No lessons yet."
        out = ["🧠 Recent lessons:"]
        for it in items:
            sym = it.get("symbol", "?")
            cause = it.get("root_cause", "?")
            rule = (it.get("proposed_rule") or "")[:140]
            out.append(f"  [{cause}] {sym} — {rule}")
        return "\n".join(out)

    if cmd == "predictions":
        status = args[0].upper() if args else "PENDING"
        items = storage.list_predictions(status=status, limit=5)
        if not items:
            return f"No {status} predictions."
        out = [f"📋 {status} predictions:"]
        for it in items:
            out.append(
                f"  {it['symbol']} {it['action']} "
                f"({it['expected_direction']} {it['expected_magnitude_pct']}%) "
                f"@ {it['ts'][:16]}"
            )
        return "\n".join(out)

    if cmd == "sentiment":
        if not args:
            return "Usage: sentiment <SYMBOL>"
        symbol = args[0]
        snap = read_sentiment(symbol)
        if not snap:
            return f"No sentiment snapshot for {symbol}. Run sentiment-refresh first."
        score = snap.get("sentiment_score", 0)
        vel = snap.get("sentiment_velocity", 0)
        n = snap.get("mention_count_24h", 0)
        return (
            f"💬 {symbol}  score {score:+.2f}  velocity {vel:+.2f}  mentions {n}\n"
            f"  (snapshot: {snap.get('as_of', '?')})"
        )

    if cmd == "analyze":
        if not args:
            return "Usage: analyze <SYMBOL>"
        if run_analyze is None:
            return (
                "Analyze runs in the background. Result will be pushed to "
                "subscribers when ready. (Trigger from CLI or Web UI for now.)"
            )
        try:
            result = run_analyze(args[0])
        except Exception as e:  # noqa: BLE001
            return f"Analyze failed: {e}"
        return json.dumps(result, indent=2)[:1000]

    return f"Unknown command: {cmd}\n\n{format_command_help()}"
