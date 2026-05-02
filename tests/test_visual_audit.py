"""Tests for visual_audit (sidecar JSON + optional matplotlib render).

Pins:
  1. JSON schema shape (closed key set on top level + overlays).
  2. Future-leak: bars whose ts > parent_bar_ts must NOT be inside the
     render window the audit reports.
  3. selected / rejected candidates from SR / trendlines / patterns
     must appear in the sidecar overlays.
  4. default current_runtime profile produces NO audit (status="v2_absent").
  5. matplotlib-missing case writes an `.image_unavailable` marker
     file but the JSON sidecar is always written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.visual_audit import (
    SCHEMA_VERSION,
    build_visual_audit_payload,
    render_visual_audit,
)


def _ohlcv(n: int = 250, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


# ─────── 1. JSON schema shape ─────────────────────────────────────


def test_payload_emits_required_top_level_keys():
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    expected = {
        "schema_version", "parent_bar_ts",
        "render_window_start_ts", "render_window_end_ts",
        "bars_used_in_render", "max_render_bars",
        "title",
        "profile", "mode", "action",
        "best_setup", "reconstruction_quality",
        "overlays", "royal_road_decision_v2",
    }
    assert set(p.keys()) == expected
    assert p["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "visual_audit_v1"


def test_overlays_section_emits_required_keys():
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    overlays = p["overlays"]
    expected_overlay_keys = {
        "level_zones_selected", "level_zones_rejected",
        "trendlines_selected", "trendlines_rejected",
        "patterns_selected", "patterns_rejected",
        "lower_tf_trigger", "structure_stop_plan",
    }
    assert set(overlays.keys()) == expected_overlay_keys
    # Lists are always present (even when empty)
    for k in (
        "level_zones_selected", "level_zones_rejected",
        "trendlines_selected", "trendlines_rejected",
        "patterns_selected", "patterns_rejected",
    ):
        assert isinstance(overlays[k], list)


# ─────── 2. Future leak ──────────────────────────────────────────


def test_future_bars_not_in_render():
    """Bars whose timestamp > parent_bar_ts must NOT appear in the
    rendered window. We pin this by checking that
    `render_window_end_ts <= parent_bar_ts` and that
    `bars_used_in_render` does not exceed the count of df bars at
    or before parent_bar_ts."""
    df = _ohlcv(300)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    # Pick an early bar so there are clearly future bars beyond it.
    tr = res.decision_traces[80]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    parent_ts = pd.Timestamp(p["parent_bar_ts"])
    end_ts = pd.Timestamp(p["render_window_end_ts"])
    # Strict invariant: end of render window cannot be after parent.
    assert end_ts <= parent_ts, (
        f"future leak: render_window_end_ts ({end_ts}) > parent_bar_ts ({parent_ts})"
    )
    # bars_used_in_render <= number of df bars at or before parent.
    n_at_or_before = int((df.index <= parent_ts).sum())
    assert p["bars_used_in_render"] <= n_at_or_before


def test_future_bars_in_extended_df_do_not_change_payload():
    """Pass a df that includes "future" bars beyond parent_bar_ts.
    The payload must be identical to one built from only the visible
    portion."""
    df_full = _ohlcv(400)
    res = run_engine_backtest(
        df_full.iloc[:200], "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    parent_ts = pd.Timestamp(tr.timestamp)
    df_visible_only = df_full[df_full.index <= parent_ts]
    p_visible = build_visual_audit_payload(
        trace=tr, df=df_visible_only,
    )
    p_with_future = build_visual_audit_payload(
        trace=tr, df=df_full,
    )
    # Render windows must match exactly.
    assert p_visible["render_window_end_ts"] == p_with_future["render_window_end_ts"]
    assert p_visible["bars_used_in_render"] == p_with_future["bars_used_in_render"]
    assert p_visible["parent_bar_ts"] == p_with_future["parent_bar_ts"]


# ─────── 3. selected/rejected appear ─────────────────────────────


def test_selected_and_rejected_candidates_appear_in_overlays():
    df = _ohlcv(400)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    # Find a trace where selected SR / trendlines exist.
    target_payload = None
    for tr in res.decision_traces[80:]:
        p = build_visual_audit_payload(trace=tr, df=df)
        if p is None:
            continue
        if (
            p["overlays"]["level_zones_selected"]
            or p["overlays"]["trendlines_selected"]
        ):
            target_payload = p
            break
    assert target_payload is not None, (
        "expected at least one trace with selected SR or trendlines "
        "in the v2 active run"
    )
    # selected / rejected entries carry the documented keys.
    if target_payload["overlays"]["level_zones_selected"]:
        lvl = target_payload["overlays"]["level_zones_selected"][0]
        for key in (
            "kind", "zone_low", "zone_high", "price",
            "first_touch_ts", "last_touch_ts",
            "touch_count", "confidence", "reasons",
        ):
            assert key in lvl
    for rj in target_payload["overlays"]["level_zones_rejected"]:
        assert "reject_reason" in rj
        assert rj["reject_reason"] is not None
    for rj in target_payload["overlays"]["trendlines_rejected"]:
        assert "reject_reason" in rj
        assert rj["reject_reason"] is not None
    for rj in target_payload["overlays"]["patterns_rejected"]:
        assert "reject_reason" in rj
        assert rj["reject_reason"] is not None


# ─────── 4. default profile = no audit ───────────────────────────


def test_default_profile_produces_no_visual_audit():
    df = _ohlcv()
    res = run_engine_backtest(df, "EURUSD=X", interval="1h", warmup=60)
    tr = res.decision_traces[100]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is None


def test_default_profile_render_returns_v2_absent_status(tmp_path: Path):
    df = _ohlcv()
    res = run_engine_backtest(df, "EURUSD=X", interval="1h", warmup=60)
    tr = res.decision_traces[100]
    out = render_visual_audit(trace=tr, df=df, out_dir=tmp_path)
    assert out["status"] == "v2_absent"
    # No json or png written.
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit.png").exists()


# ─────── 5. JSON sidecar always written when v2 active ───────────


def test_v2_active_writes_json_sidecar(tmp_path: Path):
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[150]
    out = render_visual_audit(trace=tr, df=df, out_dir=tmp_path, name_prefix="t")
    json_path = tmp_path / "t.json"
    assert json_path.exists()
    body = json.loads(json_path.read_text())
    assert body["schema_version"] == SCHEMA_VERSION
    # Either a real PNG or an .image_unavailable marker exists.
    png_path = tmp_path / "t.png"
    marker_path = tmp_path / "t.image_unavailable"
    assert png_path.exists() or marker_path.exists()


def test_matplotlib_missing_emits_marker_file(tmp_path: Path):
    """Even when matplotlib cannot render, the audit must (a) write
    the JSON, (b) emit a marker file describing the failure."""
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    out = render_visual_audit(trace=tr, df=df, out_dir=tmp_path, name_prefix="m")
    json_path = tmp_path / "m.json"
    assert json_path.exists()
    if out.get("image_status") == "matplotlib_missing":
        marker = tmp_path / "m.image_unavailable"
        assert marker.exists()
        assert "matplotlib" in marker.read_text().lower()
    else:
        # If matplotlib happens to be available, the PNG is written.
        assert (tmp_path / "m.png").exists()


def test_payload_title_carries_quality_and_action():
    """The audit title is human-readable and includes the v2 action +
    reconstruction_quality."""
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[150]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    assert "quality=" in p["title"]
    # action is one of BUY / SELL / HOLD (or absent on early-return paths)
    if p.get("action"):
        assert p["action"] in ("BUY", "SELL", "HOLD")
