"""Tests for the notification subsystem.

No real LINE call is ever made; we inject a fake `requests.post`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.fx.storage import Storage
from src.fx.web.webhook import handle_command
from src.notify import (
    LineNotifier,
    LogNotifier,
    Notification,
    NullNotifier,
    build_notifier,
    format_lesson,
    format_signal,
    format_summary,
    format_verdict,
)
from src.notify.line import verify_signature


# --- Formatters ---


def test_format_signal_includes_essentials():
    s = format_signal(
        symbol="USDJPY=X",
        action="BUY",
        confidence=0.75,
        expected_direction="UP",
        expected_magnitude_pct=0.4,
        horizon_bars=6,
        invalidation_price=149.8,
        reason="strong RSI",
        analysis_id=42,
        base_url="http://localhost:5000",
    )
    assert "USDJPY=X" in s
    assert "BUY" in s
    assert "149.8" in s
    assert "/analysis/42" in s


def test_format_signal_omits_link_without_base_url():
    s = format_signal(
        symbol="X", action="BUY", confidence=0.7,
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=99.0, reason="r",
    )
    assert "→" not in s


def test_format_verdict_marks_status_with_icon():
    s = format_verdict(
        symbol="X", action="BUY", status="CORRECT",
        actual_direction="UP", actual_magnitude_pct=0.5, note="hit target",
    )
    assert "✅" in s
    assert "CORRECT" in s


def test_format_lesson_truncates_long_strings():
    s = format_lesson(
        symbol="X",
        root_cause="NEWS_SHOCK",
        proposed_rule="x" * 500,
        narrative="y" * 500,
    )
    assert len(s) < 700  # truncation kicked in
    assert "NEWS_SHOCK" in s


def test_format_summary_handles_empty_counts():
    s = format_summary(period="evaluate", counts={})
    assert s.startswith("📊")


# --- Backends ---


def test_null_notifier_is_silent():
    n = NullNotifier()
    assert n.push(Notification(text="hi", recipients=("u1",))) is False


def test_log_notifier_records_messages(capsys):
    n = LogNotifier()
    assert n.push(Notification(text="hello", kind="signal", recipients=("u1",)))
    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert len(n.sent) == 1


def test_line_notifier_skips_when_not_configured():
    n = LineNotifier(access_token=None)
    assert n.push(Notification(text="x", recipients=("u",))) is False


@dataclass
class _Resp:
    status_code: int = 200


def test_line_notifier_pushes_when_configured():
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append({"url": url, "headers": headers, "json": json})
        return _Resp(status_code=200)

    n = LineNotifier(
        access_token="t-test",
        recipients=["U-1", "U-2"],
        http_post=fake_post,
    )
    assert n.push(Notification(text="hi", kind="signal")) is True
    assert len(seen) == 2
    assert all(s["headers"]["Authorization"] == "Bearer t-test" for s in seen)


def test_line_notifier_truncates_oversize_text():
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json)
        return _Resp(status_code=200)

    n = LineNotifier(access_token="t", recipients=["U-1"], http_post=fake_post)
    n.push(Notification(text="x" * 6000, recipients=("U-1",)))
    assert len(seen[0]["messages"][0]["text"]) <= 4900


def test_line_signature_verification_round_trip():
    secret = "shh"
    body = b'{"events":[]}'
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    sig = base64.b64encode(digest).decode()
    assert verify_signature(secret, body, sig) is True
    assert verify_signature(secret, body, "wrong") is False
    assert verify_signature("", body, sig) is False


def test_factory_picks_null_when_nothing_set(monkeypatch):
    for v in ("NOTIFY_BACKEND", "LINE_CHANNEL_ACCESS_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert isinstance(build_notifier(), NullNotifier)


def test_factory_picks_line_when_token_set(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    n = build_notifier()
    assert isinstance(n, LineNotifier)


def test_factory_respects_explicit_backend(monkeypatch):
    monkeypatch.setenv("NOTIFY_BACKEND", "log")
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    n = build_notifier()
    assert isinstance(n, LogNotifier)


# --- Subscribers persistence ---


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "fx.db")


def test_add_subscriber_then_active_list(storage: Storage):
    sid = storage.add_subscriber("line", "U1", "Alice")
    assert sid >= 1
    active = storage.active_subscribers(backend="line")
    assert len(active) == 1
    assert active[0]["display_name"] == "Alice"


def test_add_subscriber_idempotent_reactivates(storage: Storage):
    storage.add_subscriber("line", "U1")
    storage.deactivate_subscriber("line", "U1")
    assert storage.active_subscribers() == []
    storage.add_subscriber("line", "U1")  # add again
    assert len(storage.active_subscribers()) == 1


def test_active_subscribers_filter_by_kind(storage: Storage):
    storage.add_subscriber("line", "U1")
    # Default: all kinds enabled
    assert len(storage.active_subscribers(kind="signal")) == 1
    assert len(storage.active_subscribers(kind="lesson")) == 1


# --- Webhook command parser ---


def test_handle_command_help(storage: Storage):
    reply = handle_command("help", storage=storage, user_id="U1")
    assert "commands" in reply.lower()


def test_handle_command_subscribe_persists(storage: Storage):
    reply = handle_command("subscribe", storage=storage, user_id="U1", display_name="Alice")
    assert "Subscribed" in reply
    assert len(storage.active_subscribers(backend="line")) == 1


def test_handle_command_unsubscribe(storage: Storage):
    storage.add_subscriber("line", "U1")
    reply = handle_command("unsubscribe", storage=storage, user_id="U1")
    assert "Unsubscribed" in reply
    assert storage.active_subscribers() == []


def test_handle_command_status_includes_zeros(storage: Storage):
    reply = handle_command("status", storage=storage, user_id="U1")
    assert "analyses" in reply
    assert "win rate" in reply


def test_handle_command_unknown_returns_help(storage: Storage):
    reply = handle_command("xyzzy", storage=storage, user_id="U1")
    assert "Unknown command" in reply
    assert "commands" in reply.lower()


def test_handle_command_lessons_empty(storage: Storage):
    reply = handle_command("lessons", storage=storage, user_id="U1")
    assert reply.startswith("No lessons")


def test_handle_command_predictions_default_pending(storage: Storage):
    reply = handle_command("predictions", storage=storage, user_id="U1")
    assert "PENDING" in reply or reply.startswith("No")


def test_handle_command_sentiment_requires_symbol(storage: Storage):
    reply = handle_command("sentiment", storage=storage, user_id="U1")
    assert "Usage" in reply


def test_handle_command_slash_prefix_accepted(storage: Storage):
    reply = handle_command("/help", storage=storage, user_id="U1")
    assert "commands" in reply.lower()
