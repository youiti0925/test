"""Tests for the event-band SVG overlay rendered by visual_audit.

Spec required cases:
  - chart SVG contains an event band when fundamental_sidebar
    surfaces a blocking event
  - red translucent rectangle for BLOCK, orange for WARNING
  - vertical centre line at the event timestamp
  - event-label text identifies the event kind / title
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.fx.visual_audit import _event_bands_svg_fragment


NOW = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)


def _idx(n: int = 24) -> list:
    return list(pd.date_range(NOW - timedelta(hours=n - 1), NOW,
                              freq="1h", tz="UTC"))


def _x_of(i: int) -> float:
    return 50 + (i + 0.5) * 20


def test_block_event_renders_red_rect_and_dashed_line():
    out = _event_bands_svg_fragment(
        events=[{
            "when": (NOW - timedelta(hours=4)).isoformat(),
            "window_hours": 6.0,
            "status": "BLOCK",
            "kind": "FOMC",
            "title": "FOMC Statement",
        }],
        visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    )
    assert "<rect" in out
    assert "event-block" in out
    assert "stroke-dasharray='4,3'" in out  # vertical centre dash
    assert "FOMC" in out


def test_warning_event_renders_orange():
    out = _event_bands_svg_fragment(
        events=[{
            "when": (NOW - timedelta(hours=4)).isoformat(),
            "window_hours": 2.0,
            "status": "WARNING",
            "kind": "CPI",
            "title": "CPI",
        }],
        visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    )
    assert "event-warning" in out
    assert "#f9a825" in out  # orange


def test_offscreen_event_skipped():
    """Event 200h in the future shouldn't draw anything."""
    out = _event_bands_svg_fragment(
        events=[{
            "when": (NOW + timedelta(hours=200)).isoformat(),
            "window_hours": 6.0, "status": "BLOCK",
            "kind": "FOMC", "title": "FOMC Statement",
        }],
        visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    )
    # opening / closing <g> only — no rect, no line, no text
    assert "<rect" not in out
    assert "<line" not in out


def test_no_events_returns_empty_string():
    assert _event_bands_svg_fragment(
        events=[], visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    ) == ""
    assert _event_bands_svg_fragment(
        events=None, visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    ) == ""


def test_label_includes_kind_and_title_truncated():
    long_title = "Federal Open Market Committee Statement (Quarterly)"
    out = _event_bands_svg_fragment(
        events=[{
            "when": (NOW - timedelta(hours=2)).isoformat(),
            "window_hours": 6.0, "status": "BLOCK",
            "kind": "FOMC", "title": long_title,
        }],
        visible_index=_idx(24),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=480.0,
        bar_w=20.0, margin_l=50.0,
    )
    assert "FOMC" in out
    # Title truncated to 24 chars in label
    assert long_title not in out


def test_multiple_events_each_get_their_own_band():
    """Both events sit within the visible 48h window so both render."""
    events = [
        {
            "when": (NOW - timedelta(hours=20)).isoformat(),
            "window_hours": 6.0, "status": "BLOCK",
            "kind": "FOMC", "title": "FOMC",
        },
        {
            "when": (NOW - timedelta(hours=4)).isoformat(),
            "window_hours": 2.0, "status": "WARNING",
            "kind": "CPI", "title": "CPI",
        },
    ]
    n_bars = 48
    bar_w = 20.0
    plot_w = bar_w * n_bars  # match bar width × count
    out = _event_bands_svg_fragment(
        events=events, visible_index=_idx(n_bars),
        x_of=_x_of, margin_t=20.0, plot_h=400.0, plot_w=plot_w,
        bar_w=bar_w, margin_l=50.0,
    )
    assert out.count("<rect") == 2
    assert "event-block" in out
    assert "event-warning" in out
