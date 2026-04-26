"""No-op notifier. Use when notifications are disabled."""
from __future__ import annotations

from .base import Notification, Notifier


class NullNotifier(Notifier):
    name = "null"

    def push(self, notification: Notification) -> bool:  # noqa: ARG002
        return False
