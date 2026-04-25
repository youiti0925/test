"""Console / log-only notifier — useful for dev and tests."""
from __future__ import annotations

import sys

from .base import Notification, Notifier


class LogNotifier(Notifier):
    name = "log"

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.sent: list[Notification] = []

    def push(self, notification: Notification) -> bool:
        self.sent.append(notification)
        recipients = ",".join(notification.recipients) or "broadcast"
        line = f"[notify:{notification.kind}] -> {recipients}\n{notification.text}\n"
        try:
            self.stream.write(line)
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass
        return True
