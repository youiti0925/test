"""Tests for the Flask dashboard.

We construct the app with an isolated tmp Storage so the real DB is not
touched. Pipeline streaming is verified by calling the generator directly
with mocks for the network paths (yfinance, anthropic).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.fx.config import Config
from src.fx.storage import Storage
from src.fx.web.app import create_app


@pytest.fixture
def app(tmp_path: Path):
    cfg = Config(
        anthropic_api_key=None,  # disables the Claude path in routes
        model="claude-opus-4-7",
        effort="medium",
        db_path=tmp_path / "fx.db",
        default_symbol="USDJPY=X",
        default_interval="1h",
    )
    storage = Storage(cfg.db_path)
    return create_app(config=cfg, storage=storage, events_path=tmp_path / "events.json")


@pytest.fixture
def client(app):
    return app.test_client()


def test_dashboard_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Dashboard" in r.data
    assert b"Run a new analysis" in r.data


def test_analyze_page_renders(client):
    r = client.get("/analyze")
    assert r.status_code == 200
    assert b"Pipeline" in r.data
    assert b"USDJPY=X" in r.data


def test_predictions_page_renders_empty(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert b"No predictions match this filter" in r.data


def test_lessons_page_renders_empty(client):
    r = client.get("/lessons")
    assert r.status_code == 200
    assert b"No post-mortems yet" in r.data


def test_analysis_detail_404_for_unknown_id(client):
    r = client.get("/analysis/9999")
    assert r.status_code == 404


def test_analysis_detail_renders_when_present(client, app):
    storage = app.config["FX_STORAGE"]
    aid = storage.save_analysis(
        symbol="USDJPY=X",
        snapshot={"last_close": 150.0, "rsi_14": 50},
        technical_signal="HOLD",
        final_action="HOLD",
    )
    r = client.get(f"/analysis/{aid}")
    assert r.status_code == 200
    assert b"USDJPY=X" in r.data
    assert b"HOLD" in r.data


def test_status_color_filter():
    from src.fx.web.routes import status_color

    assert status_color("CORRECT") == "emerald"
    assert status_color("WRONG") == "rose"
    assert status_color("UNKNOWN") == "slate"


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": [1000] * n,
        },
        index=idx,
    )


def test_pipeline_stream_runs_offline(app):
    """Drive the generator with mocked network calls and assert step order."""
    from src.fx.web.pipeline_stream import run_pipeline

    storage = app.config["FX_STORAGE"]
    cfg = app.config["FX_CONFIG"]

    df = _synthetic_df()
    with patch("src.fx.web.pipeline_stream.fetch_ohlcv", return_value=df), \
         patch("src.fx.web.pipeline_stream.fetch_headlines", return_value=[]):
        events = list(
            run_pipeline(
                symbol="USDJPY=X",
                interval="1h",
                period="60d",
                cfg=cfg,
                storage=storage,
                events_path=Path("/nonexistent"),
                want_correlation=False,
                want_events=False,
                want_lessons=False,
                want_news=False,
                want_llm=False,
            )
        )

    steps = [e["step"] for e in events]
    assert "fetch_ohlcv" in steps
    assert "snapshot" in steps
    assert "decision" in steps
    assert steps[-1] == "done"
    final = events[-1]
    assert final["data"]["analysis_id"] is not None


def test_analyze_stream_endpoint_returns_sse(client):
    """Endpoint should be SSE; we parse the first chunk to confirm shape."""
    df = _synthetic_df(n=120)
    with patch("src.fx.web.pipeline_stream.fetch_ohlcv", return_value=df), \
         patch("src.fx.web.pipeline_stream.fetch_headlines", return_value=[]):
        r = client.get(
            "/analyze/stream"
            "?symbol=USDJPY=X&interval=1h&period=60d"
            "&llm=0&news=0&correlation=0&events=0&lessons=0"
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.content_type
        body = r.get_data(as_text=True)
        # Each SSE event begins with "data: " and ends with a blank line
        chunks = [c for c in body.split("\n\n") if c.strip()]
        assert len(chunks) >= 3
        first = json.loads(chunks[0].removeprefix("data: "))
        assert first["step"] == "fetch_ohlcv"
        assert first["status"] == "ok"


def test_dashboard_stats_are_zero_initially(app):
    storage = app.config["FX_STORAGE"]
    stats = storage.dashboard_stats()
    assert stats["total_analyses"] == 0
    assert stats["weighted_win_rate"] == 0.0
    assert stats["postmortems"] == 0


def test_line_webhook_subscribes_via_text_message(client, app):
    """Posting a 'subscribe' message should add the sender as a subscriber.

    No LINE_CHANNEL_SECRET → signature check is skipped (dev mode).
    """
    payload = {
        "events": [
            {
                "type": "message",
                "replyToken": "rt-1",
                "source": {"userId": "U-test", "type": "user"},
                "message": {"type": "text", "text": "subscribe"},
            }
        ]
    }
    r = client.post("/webhook/line", json=payload)
    assert r.status_code == 200
    storage = app.config["FX_STORAGE"]
    assert len(storage.active_subscribers(backend="line")) == 1


def test_line_webhook_rejects_bad_signature(client, monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "shh")
    r = client.post(
        "/webhook/line",
        data=b'{"events":[]}',
        headers={"X-Line-Signature": "wrong"},
    )
    assert r.status_code == 403


def test_dashboard_stats_aggregate_correctly(app):
    storage = app.config["FX_STORAGE"]
    aid = storage.save_analysis(
        symbol="X", snapshot={"k": 1}, technical_signal="BUY", final_action="BUY"
    )
    pid = storage.save_prediction(
        analysis_id=aid, symbol="X", interval="1h", entry_price=1.0,
        action="BUY", confidence=0.7, reason="r",
        expected_direction="UP", expected_magnitude_pct=0.5,
        horizon_bars=4, invalidation_price=0.99,
    )
    storage.update_prediction_evaluation(
        pid, "CORRECT", "UP", 0.6, 0.6, -0.1, False, "ok"
    )
    stats = storage.dashboard_stats()
    assert stats["predictions_by_status"]["CORRECT"] == 1
    assert stats["weighted_win_rate"] == 1.0
