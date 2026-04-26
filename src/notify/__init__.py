"""Push-notification subsystem.

Notifier is the only thing the FX pipeline knows about — it can be
swapped between LINE (production), Log (dev), or Null (silent for
tests) without touching call sites.
"""
from .base import Notification, Notifier
from .factory import build_notifier
from .formatters import (
    format_lesson,
    format_signal,
    format_summary,
    format_verdict,
)
from .line import LineNotifier
from .log import LogNotifier
from .null import NullNotifier

__all__ = [
    "Notification",
    "Notifier",
    "LineNotifier",
    "LogNotifier",
    "NullNotifier",
    "build_notifier",
    "format_signal",
    "format_verdict",
    "format_lesson",
    "format_summary",
]
