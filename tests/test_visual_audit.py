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


# ─────── batch report (visual_audit_report_v1) ───────────────────


def _v2_run(n: int = 250):
    df = _ohlcv(n)
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    return df, res


def test_report_writes_top_level_files(tmp_path: Path):
    from src.fx.visual_audit import (
        BATCH_SCHEMA_VERSION, render_visual_audit_report,
    )
    df, res = _v2_run()
    out = render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    assert out["schema_version"] == BATCH_SCHEMA_VERSION
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "cases.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "assets" / "style.css").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == BATCH_SCHEMA_VERSION
    assert summary["max_cases"] == 10
    cases = json.loads((tmp_path / "cases.json").read_text())
    assert len(cases) == out["n_cases"]


def test_report_per_case_files_present(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=5,
    )
    sym_dir = tmp_path / "EURUSD=X"
    assert sym_dir.exists()
    case_dirs = sorted(p for p in sym_dir.iterdir() if p.is_dir())
    assert case_dirs, "expected at least one case directory"
    for cd in case_dirs:
        assert (cd / "audit.json").exists()
        assert (cd / "detail.html").exists()
        assert (cd / "feedback_template.json").exists()
        # Either rendered .png or .image_unavailable marker per chart slot.
        for stem in ("chart_main", "chart_micro", "chart_short",
                     "chart_medium", "chart_long", "chart_lower_tf"):
            png = cd / f"{stem}.png"
            marker = cd / f"{stem}.image_unavailable"
            assert png.exists() or marker.exists(), (
                f"{cd.name}/{stem}: neither png nor marker"
            )


def test_audit_json_has_chart_status_and_no_future_leak_check(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(
        (p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir()),
        None,
    )
    assert cd is not None
    audit = json.loads((cd / "audit.json").read_text())
    for key in (
        "case_id", "symbol", "interval",
        "parent_bar_ts", "render_window_end_ts",
        "charts", "chart_status",
        "no_future_leak_check", "current_runtime_vs_royal_v2",
        "royal_road_decision_v2",
    ):
        assert key in audit, f"audit.json missing {key}"
    assert audit["no_future_leak_check"]["passed"] is True
    # render_window_end_ts <= parent_bar_ts (already pinned elsewhere
    # but worth re-asserting in the report context).
    assert (
        audit["render_window_end_ts"] <= audit["parent_bar_ts"]
    )


def test_audit_json_includes_selected_and_rejected(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    found_with_overlays = False
    for cd in (tmp_path / "EURUSD=X").iterdir():
        if not cd.is_dir():
            continue
        audit = json.loads((cd / "audit.json").read_text())
        ov = audit.get("overlays") or {}
        for key in (
            "level_zones_selected", "level_zones_rejected",
            "trendlines_selected", "trendlines_rejected",
            "patterns_selected", "patterns_rejected",
        ):
            assert key in ov
        if (
            ov["level_zones_selected"] or ov["trendlines_selected"]
        ):
            found_with_overlays = True
    assert found_with_overlays, (
        "expected at least one case with non-empty selected overlays"
    )


def test_feedback_template_schema(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=2,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    fb = json.loads((cd / "feedback_template.json").read_text())
    assert "case_id" in fb
    hr = fb["human_review"]
    for k in ("sr_zones", "trendlines", "patterns", "final_decision"):
        assert k in hr
    assert "system_action_reasonable" in hr["final_decision"]
    assert "preferred_action" in hr["final_decision"]


def test_default_profile_produces_no_cases(tmp_path: Path):
    """Default current_runtime traces have no v2 slice → the report
    has 0 cases (no per-case dirs written, but top-level index/summary
    do still exist with n_cases=0)."""
    from src.fx.visual_audit import render_visual_audit_report
    df = _ohlcv()
    res = run_engine_backtest(df, "EURUSD=X", interval="1h", warmup=60)
    out = render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    assert out["n_cases"] == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_cases"] == 0
    # No symbol directory should be created.
    assert not (tmp_path / "EURUSD=X").exists()


def test_index_html_contains_priority_classes(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    html = (tmp_path / "index.html").read_text()
    # Index links to per-case detail.html relatively.
    assert "detail.html" in html
    # At least one priority CSS class is applied.
    assert any(
        f"priority-{p}" in html
        for p in (
            "v2_directional", "v2_vs_current_diff", "high_quality_hold",
            "low_quality_directional_attempt", "fake_breakout_block",
            "random_hold_filler",
        )
    )


def test_detail_html_contains_step_trace_and_compare_section(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    html = (cd / "detail.html").read_text()
    # Step trace section
    assert "Step 1: Risk Gate" in html
    assert "Step 9: Reconstruction Quality" in html
    # Comparison section
    assert "current_runtime vs royal_road_v2" in html
    assert "best_setup" in html
    assert "setup_candidates" in html


def test_candlestick_helper_has_body_and_wick_logic_in_source():
    """Structural pin: the candle render helper must contain BOTH
    body and wick drawing primitives. matplotlib is not installed in
    CI, so we verify by source inspection rather than by image."""
    import inspect
    from src.fx import visual_audit
    src = inspect.getsource(visual_audit._render_candle_image)
    assert "vlines" in src, "candle helper should use vlines for wick + body"
    assert "low" in src and "high" in src
    # Bullish / bearish branching present (color split):
    assert "bullish" in src or "bull_x" in src
    # Rejected trendlines must NOT be skipped (must appear in render path).
    assert "trendlines_rejected" in src


def test_rejected_trendlines_overlay_present_when_v2_active(tmp_path: Path):
    """Pin: rejected trendlines are not silently dropped; they appear
    in the overlays section of the audit.json so the renderer (and a
    human reading the JSON) can see them."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    seen_rejected = False
    for cd in (tmp_path / "EURUSD=X").iterdir():
        if not cd.is_dir():
            continue
        audit = json.loads((cd / "audit.json").read_text())
        rj = (audit.get("overlays") or {}).get("trendlines_rejected", [])
        if rj:
            seen_rejected = True
            for entry in rj:
                assert "reject_reason" in entry
                assert entry["reject_reason"] is not None
            break
    assert seen_rejected, "expected at least one case with rejected trendlines"


def test_multi_scale_unavailable_recorded_in_audit(tmp_path: Path):
    """Short df → medium / long scales must be reported as
    available=False with unavailable_reason=short_history (or similar)
    in audit.json. micro / short still available."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run(n=200)
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    ms = audit["chart_status"]["multi_scale"]
    # long bar count = 1000 → 200-bar fixture cannot satisfy → False
    assert ms["long"]["available"] is False
    assert ms["long"]["unavailable_reason"] is not None


def test_lower_tf_unavailable_recorded(tmp_path: Path):
    """No df_lower_by_symbol attached → lower_tf chart marked
    unavailable in audit.json."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    ltf = audit["chart_status"]["lower_tf"]
    assert ltf["available"] is False
    assert ltf["unavailable_reason"]


def test_report_is_deterministic_across_runs(tmp_path: Path):
    """Same inputs → identical cases.json + summary.json (modulo
    file-creation timestamps which we don't write). This pins
    reproducibility for review workflows."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=out_a, max_cases=8,
    )
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=out_b, max_cases=8,
    )
    assert (out_a / "cases.json").read_text() == (out_b / "cases.json").read_text()
    assert (out_a / "summary.json").read_text() == (out_b / "summary.json").read_text()


def test_current_runtime_vs_royal_v2_section_in_audit(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=5,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    cmp_ = audit["current_runtime_vs_royal_v2"]
    for key in (
        "current_runtime_action", "royal_road_v2_action",
        "difference_type", "current_runtime_reason", "royal_v2_reason",
    ):
        assert key in cmp_
    # difference_type must be from the closed taxonomy v2 emits.
    assert cmp_["difference_type"] in {
        "same",
        "current_buy_royal_hold", "current_sell_royal_hold",
        "current_hold_royal_buy", "current_hold_royal_sell",
        "opposite_direction", "other", None,
    }


def test_future_bars_not_in_batch_report(tmp_path: Path):
    """For every case in the report, render_window_end_ts <=
    parent_bar_ts (no future leak)."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run(n=300)
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    for cd in (tmp_path / "EURUSD=X").iterdir():
        if not cd.is_dir():
            continue
        audit = json.loads((cd / "audit.json").read_text())
        end_ts = pd.Timestamp(audit["render_window_end_ts"])
        parent_ts = pd.Timestamp(audit["parent_bar_ts"])
        assert end_ts <= parent_ts
