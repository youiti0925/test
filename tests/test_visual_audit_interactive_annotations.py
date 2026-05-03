"""Tests for the static-HTML interactive annotation layer + the
WAIT_EVENT_CLEAR end-to-end behaviour through the integrated decision.

Spec required cases:
  - mobile HTML contains the .g5-info-panel right sidebar
  - mobile HTML contains the manual annotation controls
  - mobile HTML contains the JS layer (localStorage / export / SVG click)
  - WAIT_EVENT_CLEAR fires when entry_plan READY meets event BLOCK
  - default current_runtime unchanged
  - live / OANDA / paper untouched
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.calendar import Event
from src.fx.risk_gate import RiskState
from src.fx.royal_road_integrated_decision import (
    decide_royal_road_v2_integrated,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_FX = REPO_ROOT / "src" / "fx"
NOW = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)


def _df_realistic(symbol: str = "EURUSD=X") -> pd.DataFrame:
    """120-bar synthetic OHLC ending at NOW so the chart renders."""
    n = 120
    idx = pd.date_range(NOW - timedelta(hours=n - 1), NOW,
                        freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    close = 1.10 + np.cumsum(rng.standard_normal(n) * 0.001)
    return pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001,
        "close": close, "volume": [1000] * n,
    }, index=idx)


# ── HTML smoke: mobile single-file contains G.5 elements ─────────
def test_mobile_html_contains_g5_right_panel(tmp_path: Path):
    """End-to-end: run the smoke generator's DB demo and verify the
    rendered HTML embeds the .g5-info-panel + .g5-anno-controls +
    JS layer."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "db_buy.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "g5-info-panel" in html, "right info panel missing"
    assert "g5-anno-controls" in html, "manual annotation controls missing"
    assert "g5-anno-add" in html, "追加 button missing"
    assert "g5-anno-export" in html, "export button missing"
    # JS layer must be present
    assert "gx_user_chart_annotations__" in html, (
        "localStorage key prefix missing — JS layer not embedded"
    )
    assert "annotations.json" in html or "annotations__" in html, (
        "export JSON link template missing"
    )


def test_mobile_html_includes_six_panel_rows(tmp_path: Path):
    """The right sidebar has 6 numbered sections per spec."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "db_buy_panels.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    for header in (
        "1. 現在の状態",
        "2. エントリープラン",
        "3. ファンダ・イベント",
        "4. 王道根拠",
        "5. 未接続 / 注意",
        "6. 手動線操作",
    ):
        assert header in html, f"sidebar row missing: {header!r}"


def test_mobile_html_explicitly_states_display_only_default(
    tmp_path: Path,
):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_visual_audit_smoke import (
        _build_double_bottom_integrated_buy_demo_mobile,
    )
    out = tmp_path / "db_buy_policy.html"
    _build_double_bottom_integrated_buy_demo_mobile(out_path=out)
    html = out.read_text(encoding="utf-8")
    assert "display_only" in html, "default policy not surfaced"


# ── WAIT_EVENT_CLEAR end-to-end ──────────────────────────────────
def test_integrated_decision_emits_wait_event_clear_on_blocking_event():
    """When entry_plan would be READY but a BLOCK-window event sits
    inside ±window, the integrated profile downgrades to
    WAIT_EVENT_CLEAR and emits HOLD with an event-specific reason."""
    # Use the test_royal_road_integrated_decision fixtures helper
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from tests.test_royal_road_integrated_decision import _full_panels

    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        candle_bar="bullish_pinbar", candle_location="at_support",
    )
    fomc = Event(
        when=NOW + timedelta(hours=4), currency="USD",
        title="FOMC Statement", impact="high", kind="FOMC",
    )
    panels["events_for_sidebar"] = [fomc]

    df = _df_realistic("USDJPY=X")
    d = decide_royal_road_v2_integrated(
        df_window=df, technical_confluence={}, pattern=None,
        higher_timeframe_trend="UP", risk_reward=None,
        risk_state=RiskState(),
        atr_value=0.005, last_close=1.0,
        symbol="USDJPY=X", macro_context={"vix": 18.0},
        df_lower_tf=None, lower_tf_interval=None,
        base_bar_close_ts=pd.Timestamp(NOW),
        mode="integrated_balanced",
        audit_panels=panels,
    )
    assert d.action == "HOLD"
    ep = d.advisory.get("entry_plan") or {}
    assert ep.get("entry_status") == "WAIT_EVENT_CLEAR"
    assert ep.get("downgraded_from_ready_by_event_risk") is True
    assert "FOMC" in (ep.get("reason_ja") or "")
    fs = d.advisory.get("fundamental_sidebar") or {}
    assert fs.get("event_risk_status") == "BLOCK"
    assert len(fs.get("blocking_events") or []) >= 1


def test_no_event_keeps_entry_plan_ready_path():
    """Symmetric control: with NO blocking event, the plan stays
    READY (or whatever it would have been without G)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from tests.test_royal_road_integrated_decision import _full_panels

    panels = _full_panels(
        side="BUY", pattern_kind="double_bottom",
        rr=2.5, near_support=True, dow_trend="UP",
        candle_bar="bullish_pinbar", candle_location="at_support",
    )
    panels["events_for_sidebar"] = []  # empty event feed → CLEAR
    df = _df_realistic("USDJPY=X")
    d = decide_royal_road_v2_integrated(
        df_window=df, technical_confluence={}, pattern=None,
        higher_timeframe_trend="UP", risk_reward=None,
        risk_state=RiskState(), atr_value=0.005, last_close=1.0,
        symbol="USDJPY=X", macro_context={"vix": 18.0},
        df_lower_tf=None, lower_tf_interval=None,
        base_bar_close_ts=pd.Timestamp(NOW),
        mode="integrated_balanced",
        audit_panels=panels,
    )
    ep = d.advisory.get("entry_plan") or {}
    # entry_status should NOT be WAIT_EVENT_CLEAR in this path
    assert ep.get("entry_status") != "WAIT_EVENT_CLEAR"
    fs = d.advisory.get("fundamental_sidebar") or {}
    assert fs.get("event_risk_status") in ("CLEAR", "WARNING", "UNKNOWN")


# ── Live/OANDA/paper isolation (Phase G modules) ─────────────────
def test_live_paths_do_not_reference_phase_g_modules():
    """broker.py / oanda.py must not import or reference any Phase G
    module (fundamental_sidebar / user_chart_annotations /
    manual_line_overrides)."""
    for fname in ("broker.py", "oanda.py"):
        text = (SRC_FX / fname).read_text(encoding="utf-8")
        for forbidden in (
            "fundamental_sidebar",
            "user_chart_annotations",
            "manual_line_overrides",
        ):
            assert forbidden not in text, (
                f"{fname} references Phase G module {forbidden!r}"
            )


def test_default_current_runtime_unchanged():
    """CLI default --decision-profile must remain current_runtime."""
    import re
    cli_text = (SRC_FX / "cli.py").read_text(encoding="utf-8")
    m = re.search(
        r'add_argument\(\s*"--decision-profile"[^)]*default="([^"]+)"',
        cli_text, re.DOTALL,
    )
    assert m is not None
    assert m.group(1) == "current_runtime"
