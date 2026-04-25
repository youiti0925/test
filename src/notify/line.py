"""LINE Messaging API integration.

Only the bits we actually use: push, reply, and webhook signature
verification. We deliberately do NOT use line-bot-sdk so the dependency
footprint stays small and tests don't need a heavy stub.

Setup:
  1. Create a Messaging API channel in the LINE Developers console.
  2. Note the Channel access token and Channel secret.
  3. Set:
       export LINE_CHANNEL_ACCESS_TOKEN="..."
       export LINE_CHANNEL_SECRET="..."
  4. Set the webhook URL in the console to your public deployment:
       https://your-host/webhook/line
  5. Send a message to your bot — it auto-subscribes the sender
     (see src/fx/web/routes.py).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Callable, Iterable

import requests

from .base import Notification, Notifier

PUSH_URL = "https://api.line.me/v2/bot/message/push"
REPLY_URL = "https://api.line.me/v2/bot/message/reply"
MAX_TEXT_LEN = 4900  # LINE's hard limit is 5000 per message


class LineNotifier(Notifier):
    name = "line"

    def __init__(
        self,
        access_token: str | None = None,
        recipients: Iterable[str] | None = None,
        timeout_s: float = 8.0,
        http_post: Callable | None = None,
    ) -> None:
        self.access_token = access_token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
        self.default_recipients = tuple(recipients or ())
        self.timeout_s = timeout_s
        self._post = http_post or requests.post

    @property
    def configured(self) -> bool:
        return bool(self.access_token)

    def push(self, notification: Notification) -> bool:
        if not self.configured:
            return False
        targets = notification.recipients or self.default_recipients
        if not targets:
            return False
        text = _truncate(notification.text)
        ok_count = 0
        for user_id in targets:
            try:
                resp = self._post(
                    PUSH_URL,
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "to": user_id,
                        "messages": [{"type": "text", "text": text}],
                    },
                    timeout=self.timeout_s,
                )
                if 200 <= resp.status_code < 300:
                    ok_count += 1
            except Exception:  # noqa: BLE001
                continue
        return ok_count > 0

    def reply(self, reply_token: str, text: str) -> bool:
        if not self.configured:
            return False
        try:
            resp = self._post(
                REPLY_URL,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "replyToken": reply_token,
                    "messages": [{"type": "text", "text": _truncate(text)}],
                },
                timeout=self.timeout_s,
            )
            return 200 <= resp.status_code < 300
        except Exception:  # noqa: BLE001
            return False


def verify_signature(channel_secret: str, body: bytes, signature_header: str) -> bool:
    """Verify the HMAC-SHA256 signature LINE attaches to webhook bodies.

    Always use hmac.compare_digest to defeat timing attacks.
    """
    if not channel_secret or not signature_header:
        return False
    digest = hmac.new(
        channel_secret.encode("utf-8"), body, hashlib.sha256
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature_header)


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_LEN:
        return text
    return text[: MAX_TEXT_LEN - 3] + "..."
