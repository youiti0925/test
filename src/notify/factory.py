"""Pick the right notifier from environment.

Resolution order:
  1. NOTIFY_BACKEND env var ("line" | "log" | "null") if set.
  2. If LINE_CHANNEL_ACCESS_TOKEN is set, default to LineNotifier.
  3. Otherwise NullNotifier.

Centralized so callers don't repeat this logic.
"""
from __future__ import annotations

import os

from .base import Notifier
from .line import LineNotifier
from .log import LogNotifier
from .null import NullNotifier


def build_notifier(default_recipients: list[str] | None = None) -> Notifier:
    backend = os.environ.get("NOTIFY_BACKEND", "").lower().strip()
    if backend == "line":
        return LineNotifier(recipients=default_recipients or [])
    if backend == "log":
        return LogNotifier()
    if backend == "null":
        return NullNotifier()
    if os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"):
        return LineNotifier(recipients=default_recipients or [])
    return NullNotifier()
