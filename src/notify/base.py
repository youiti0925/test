"""Notifier abstraction shared across all backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Notification:
    """A single message to push to one or more recipients.

    `kind` is a short tag the formatter and downstream filters use:
      "signal" | "verdict" | "lesson" | "summary" | "command_reply"

    `recipients` is provider-specific (LINE userId, Slack channel, ...).
    Empty list means broadcast to all active subscribers.
    """

    text: str
    kind: str = "info"
    recipients: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class Notifier(ABC):
    """Backend-independent push interface."""

    name: str

    @abstractmethod
    def push(self, notification: Notification) -> bool:
        """Send the notification. Return True iff at least one recipient
        successfully received it.

        Implementations must NOT raise on transient errors; a failure to
        notify must never crash the analysis pipeline.
        """

    def push_many(self, notifications: Iterable[Notification]) -> int:
        """Convenience wrapper. Returns the number of successful pushes."""
        ok = 0
        for n in notifications:
            try:
                if self.push(n):
                    ok += 1
            except Exception:  # noqa: BLE001
                continue
        return ok
