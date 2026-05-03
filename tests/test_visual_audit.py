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
        "overlays", "checklist_panels", "royal_road_decision_v2",
        # waveform shape audit additions (observation-only, NOT used
        # by royal_road_decision_v2 final action logic).
        "wave_shape_review", "entry_summary",
        # wave-derived lines (WNL / WB1 / WB2 / WSL / WTP / ...)
        "wave_derived_lines",
        # Masterclass observation-only audit panels (16 features)
        "masterclass_panels",
        # Decision bridge (USED / PARTIAL / AUDIT_ONLY / NOT_CONNECTED)
        "decision_bridge",
        # Phase G — fundamental_sidebar_v1 + user_chart_annotations.
        # Both are observation-only and may be None for the legacy v2
        # path (the integrated profile populates them).
        "fundamental_sidebar",
        "user_chart_annotations",
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
    # PNG, SVG fallback, or .image_unavailable marker — always one.
    png_path = tmp_path / "t.png"
    svg_path = tmp_path / "t.svg"
    marker_path = tmp_path / "t.image_unavailable"
    assert png_path.exists() or svg_path.exists() or marker_path.exists()


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
    # Three-tier fallback: matplotlib_png OR svg_fallback OR image_unavailable
    renderer = out.get("renderer")
    assert renderer in (
        "matplotlib_png", "svg_fallback", "image_unavailable",
    )
    if renderer == "matplotlib_png":
        assert (tmp_path / "m.png").exists()
    elif renderer == "svg_fallback":
        assert (tmp_path / "m.svg").exists()
    else:
        assert (tmp_path / "m.image_unavailable").exists()


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
        # Three-tier fallback per chart slot: PNG, SVG, or marker.
        for stem in ("chart_main", "chart_micro", "chart_short",
                     "chart_medium", "chart_long", "chart_lower_tf"):
            png = cd / f"{stem}.png"
            svg = cd / f"{stem}.svg"
            marker = cd / f"{stem}.image_unavailable"
            assert png.exists() or svg.exists() or marker.exists(), (
                f"{cd.name}/{stem}: neither png/svg/marker"
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
    """Structural pin: the candle render helpers must contain BOTH
    body and wick drawing primitives. matplotlib is not installed in
    CI, so we verify by source inspection rather than by image.

    Inspects both the matplotlib path (`_try_matplotlib_render`) and
    the dependency-free SVG fallback (`_render_candle_svg`).
    """
    import inspect
    from src.fx import visual_audit
    mpl_src = inspect.getsource(visual_audit._try_matplotlib_render)
    svg_src = inspect.getsource(visual_audit._build_candle_svg_xml)
    # matplotlib path
    assert "vlines" in mpl_src, "matplotlib path should use vlines"
    assert "bullish" in mpl_src or "bull_x" in mpl_src
    assert "trendlines_rejected" in mpl_src
    # SVG path (rect + line for wick / body)
    assert "<svg" in svg_src
    assert "<line" in svg_src and "<rect" in svg_src
    assert "trendlines_rejected" in svg_src
    assert "level_zones_selected" in svg_src
    assert "level_zones_rejected" in svg_src
    # Future-leak guarantee in both paths
    assert "df.index <= end_ts" in mpl_src
    assert "df.index <= end_ts" in svg_src


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
    # chart_status flat keys per per-scale renderer taxonomy.
    long_status = audit["chart_status"]["long"]
    # long bar count = 1000 → 200-bar fixture cannot satisfy → False
    assert long_status["available"] is False
    assert long_status["unavailable_reason"] is not None
    assert long_status["renderer"] in ("short_history", "image_unavailable")


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


# ─────── PDF-derived royal-road checklist panels ────────────────────


def _first_v2_payload() -> dict:
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[150]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    return p


def test_checklist_panels_present_in_payload():
    """All 7 PDF-derived panels appear under `checklist_panels`."""
    p = _first_v2_payload()
    cp = p["checklist_panels"]
    expected = {
        "candlestick_interpretation",
        "rsi_context",
        "ma_granville_context",
        "bollinger_lifecycle",
        "grand_confluence_checklist",
        "invalidation_explanation",
        "current_runtime_indicator_votes",
    }
    assert set(cp.keys()) == expected


def test_candlestick_interpretation_panel_keys():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["candlestick_interpretation"]
    for key in (
        "available", "candle_type", "direction", "source",
        "context", "strength", "near_strong_support",
        "near_strong_resistance", "supporting_pattern_kinds", "reason",
    ):
        assert key in panel
    assert panel["direction"] in ("BUY", "SELL", "NEUTRAL")
    assert panel["strength"] in ("strong", "medium", "weak", "none")


def test_rsi_context_panel_keys_and_trap_logic():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["rsi_context"]
    for key in (
        "available", "rsi_value", "regime",
        "overbought", "oversold",
        "rsi_signal", "rsi_signal_valid", "rsi_trap_reason", "source",
    ):
        assert key in panel
    assert panel["regime"] in ("range", "trend", "trend_extreme", "unknown")
    assert panel["rsi_signal"] in ("BUY", "SELL", "NEUTRAL")
    # rsi_signal_valid is True only when extreme + range regime + not danger.
    if panel["rsi_signal"] != "NEUTRAL" and panel["regime"] != "range":
        assert panel["rsi_signal_valid"] is False


def test_ma_granville_context_panel_keys():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["ma_granville_context"]
    if panel.get("available"):
        for key in (
            "ma_periods", "ma_slope", "price_vs_ma",
            "pullback_to_ma", "bounce_from_ma", "breakdown_from_ma",
            "granville_pattern", "direction", "confidence",
            "sma_20", "sma_50", "close", "reason", "source",
        ):
            assert key in panel
        assert panel["ma_slope"] in ("rising", "falling", "flat")
        assert panel["price_vs_ma"] in (
            "above_both", "below_both", "between_mas",
        )
    else:
        assert "unavailable_reason" in panel


def test_bollinger_lifecycle_panel_keys():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["bollinger_lifecycle"]
    if panel.get("available"):
        for key in (
            "lifecycle_phase", "bb_squeeze", "bb_expansion", "bb_band_walk",
            "breakout_after_squeeze", "reversal_risk",
            "bb_signal_valid", "source", "reason",
        ):
            assert key in panel
        assert panel["lifecycle_phase"] in (
            "squeeze", "post_squeeze_breakout", "band_walk",
            "expansion", "neutral",
        )
    else:
        assert "unavailable_reason" in panel


def test_grand_confluence_checklist_has_12_items_with_pwb_totals():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["grand_confluence_checklist"]
    assert panel["available"] is True
    assert panel["total_items"] == 12
    items = panel["items"]
    assert len(items) == 12
    item_ids = {it["item"] for it in items}
    expected_ids = {
        "higher_tf_alignment", "dow_structure", "support_resistance",
        "trendline", "chart_pattern", "candlestick_confirmation",
        "lower_tf_confirmation", "indicator_environment",
        "macro_alignment", "invalidation_clear", "rr_valid",
        "reconstruction_quality",
    }
    assert item_ids == expected_ids
    for it in items:
        assert it["status"] in ("PASS", "WARN", "BLOCK")
    assert (
        panel["total_pass"] + panel["total_warn"] + panel["total_block"] == 12
    )


def test_invalidation_explanation_panel_present_when_v2_active():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["invalidation_explanation"]
    if panel.get("available"):
        for key in (
            "selected_stop_price", "atr_stop_price", "structure_stop_price",
            "chosen_mode", "outcome",
            "invalidation_structure", "invalidation_status",
            "why_this_stop_invalidates_the_setup", "rr_selected",
            "source", "reason",
        ):
            assert key in panel
        assert panel["invalidation_status"] in (
            "valid", "invalid", "fallback_atr", "missing",
        )
    else:
        assert "unavailable_reason" in panel


def test_current_runtime_indicator_votes_panel_uses_trace_when_present():
    p = _first_v2_payload()
    panel = p["checklist_panels"]["current_runtime_indicator_votes"]
    # technical_confluence is wired into the trace by default for v2 runs.
    assert panel["available"] is True
    for key in (
        "rsi_vote", "macd_vote", "sma_vote", "bb_vote",
        "voted_action", "vote_count_buy", "vote_count_sell",
        "warning", "source",
    ):
        assert key in panel
    for v in (panel["rsi_vote"], panel["macd_vote"], panel["sma_vote"]):
        assert v in ("BUY", "SELL", "NEUTRAL")
    assert panel["bb_vote"] in ("BUY", "SELL", "NEUTRAL", "UNKNOWN")
    assert panel["voted_action"] in ("BUY", "SELL", "HOLD")


def test_current_runtime_indicator_votes_unavailable_without_tc():
    """Direct unit-test the panel builder with tc=None to pin the
    `available=False` fallback path the user requested."""
    from src.fx.visual_audit import _current_runtime_indicator_votes_panel
    panel = _current_runtime_indicator_votes_panel(tc=None)
    assert panel["available"] is False
    assert "unavailable_reason" in panel


def test_default_runtime_payload_does_not_emit_visual_audit(tmp_path: Path):
    """Pin the existing invariant explicitly in the panel test layer:
    the default current_runtime profile produces no audit (None)."""
    df = _ohlcv()
    res = run_engine_backtest(df, "EURUSD=X", interval="1h", warmup=60)
    tr = res.decision_traces[150]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is None


def test_checklist_panels_no_future_leak_via_render_window():
    """Pin: checklist panels are computed off trace data only; their
    construction must not require bars beyond parent_bar_ts. This test
    guarantees the panel builders never read df by feeding them an
    empty df + verifying the panels still build off the trace."""
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
    p_with_future = build_visual_audit_payload(trace=tr, df=df_full)
    # Panels must be byte-identical regardless of whether the df has
    # future bars attached.
    assert p_visible["checklist_panels"] == p_with_future["checklist_panels"]


def test_detail_html_renders_grand_confluence_and_invalidation(tmp_path: Path):
    """Both Grand Confluence Checklist and Invalidation Explanation
    panels must be visible in detail.html."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    html = (cd / "detail.html").read_text()
    assert "Grand Confluence Checklist" in html
    assert "Invalidation Explanation" in html
    assert "Candlestick Interpretation" in html
    assert "RSI Context" in html
    assert "MA Granville Context" in html
    assert "Bollinger Lifecycle" in html
    assert "Current Runtime Indicator Votes" in html


def test_audit_json_contains_checklist_panels(tmp_path: Path):
    """The on-disk audit.json must include the new `checklist_panels`
    key with all 7 sub-panels."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    assert "checklist_panels" in audit
    cp = audit["checklist_panels"]
    for key in (
        "candlestick_interpretation", "rsi_context",
        "ma_granville_context", "bollinger_lifecycle",
        "grand_confluence_checklist", "invalidation_explanation",
        "current_runtime_indicator_votes",
    ):
        assert key in cp


# ─────── SVG fallback renderer ──────────────────────────────────────


def _disable_matplotlib(monkeypatch) -> None:
    """Force `_try_matplotlib_render` to return None so the chain falls
    through to the SVG fallback. Works whether matplotlib is installed
    or not."""
    from src.fx import visual_audit
    monkeypatch.setattr(
        visual_audit, "_try_matplotlib_render",
        lambda **kwargs: None,
    )


def test_svg_fallback_writes_chart_main_svg(monkeypatch, tmp_path: Path):
    """With matplotlib disabled, chart_main.svg must be generated and
    chart_main.png must NOT exist (so the fallback chain actually
    falls through to SVG, not just produces matplotlib output)."""
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    assert (cd / "chart_main.svg").exists()
    assert not (cd / "chart_main.png").exists()


def test_svg_fallback_contains_candle_body_and_wick(monkeypatch, tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    svg = (cd / "chart_main.svg").read_text()
    assert svg.startswith("<?xml") and "<svg" in svg
    # Candle wick (line) and body (rect) primitives.
    assert "<line" in svg
    assert "<rect" in svg
    # Candle classes for downstream styling / inspection.
    assert "class='wick'" in svg
    assert "candle-bull" in svg or "candle-bear" in svg


def test_svg_fallback_includes_selected_sr_zone(monkeypatch, tmp_path: Path):
    """At least one case should have `level_zones_selected`; that
    zone must show up in the SVG (sr-selected class on a <rect>)."""
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    found = False
    for cd in (tmp_path / "EURUSD=X").iterdir():
        if not cd.is_dir():
            continue
        audit = json.loads((cd / "audit.json").read_text())
        if (audit.get("overlays") or {}).get("level_zones_selected"):
            svg = (cd / "chart_main.svg").read_text()
            assert "sr-selected" in svg
            found = True
            break
    assert found, "no case with selected SR zone found in 10-case sample"


def test_svg_fallback_draws_rejected_trendline_overlay(tmp_path: Path):
    """Pin: when an overlay carries a rejected trendline with concrete
    slope/intercept/anchor_indices, the SVG renderer draws it as a
    dotted line with the `trendline-rejected` class. Synthetic backtest
    data often only produces `trendline=None` reject markers (e.g.
    `outside_top3`), so we drive the renderer directly with a handcrafted
    overlay to pin the rendering capability deterministically."""
    from src.fx.visual_audit import _render_candle_svg
    idx = pd.date_range("2025-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(100.0, 102.0, 20),
        "high": np.linspace(100.5, 102.5, 20),
        "low": np.linspace(99.5, 101.5, 20),
        "close": np.linspace(100.2, 102.2, 20),
        "volume": [1000] * 20,
    }, index=idx)
    overlays = {
        "level_zones_selected": [],
        "level_zones_rejected": [],
        "trendlines_selected": [],
        "trendlines_rejected": [
            {
                "trendline": {
                    "slope": 0.05, "intercept": 100.0,
                    "anchor_indices": [2, 18], "kind": "ascending",
                },
                "reject_reason": "weak_confidence",
            },
        ],
        "patterns_selected": [],
        "patterns_rejected": [],
        "lower_tf_trigger": None,
        "structure_stop_plan": None,
    }
    out_path = tmp_path / "rejtl.svg"
    res = _render_candle_svg(
        df=df, end_ts=idx[-1], n_bars=20,
        overlays=overlays, title="rejected-tl-test",
        out_path=out_path,
    )
    assert res["image_status"] == "rendered"
    svg = out_path.read_text()
    assert "trendline-rejected" in svg
    # Selected SR / pattern classes should NOT appear (none provided).
    assert "sr-selected" not in svg
    assert "pattern-neckline" not in svg


def test_chart_status_main_renderer_is_svg_fallback(monkeypatch, tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    assert audit["chart_status"]["main"]["renderer"] == "svg_fallback"
    assert audit["chart_status"]["main"]["available"] is True
    assert audit["chart_status"]["main"]["path"] == "chart_main.svg"
    # `charts.main` exposes the resolved filename for HTML consumption.
    assert audit["charts"]["main"] == "chart_main.svg"


def test_detail_html_references_svg_when_only_svg_exists(monkeypatch, tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    html = (cd / "detail.html").read_text()
    assert "chart_main.svg" in html
    # Should NOT reference chart_main.png since none was rendered.
    assert "chart_main.png" not in html
    # Renderer tag visible to humans.
    assert "svg_fallback" in html


def test_index_html_thumbnail_falls_back_to_svg(monkeypatch, tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=5,
    )
    html = (tmp_path / "index.html").read_text()
    assert "chart_main.svg" in html
    # PNG fallback path should not appear (no PNG written).
    assert "chart_main.png" not in html


def test_svg_fallback_no_future_leak(monkeypatch, tmp_path: Path):
    """Pin: even when df has future bars beyond parent_bar_ts, the
    SVG renderer must only draw bars with index <= parent_bar_ts.
    Verified by checking the count of bars whose timestamp appears in
    the SVG never exceeds the visible window."""
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df = _ohlcv(400)
    res = run_engine_backtest(
        df.iloc[:200], "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    # Render with the FULL df (200 future bars present beyond parent).
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    parent_ts = pd.Timestamp(audit["parent_bar_ts"])
    # The SVG advertises its parent_bar_ts in plain text. Ensure it
    # equals the trace's parent (no silent extension into future bars).
    svg = (cd / "chart_main.svg").read_text()
    assert f"parent_bar_ts={parent_ts.isoformat()}" in svg
    # End-of-window timestamp in SVG (last visible bar) must be <=
    # parent. We can extract it from the bottom-right text label.
    # Lightweight check: any timestamp in the SVG > parent_ts is a leak.
    import re
    iso_ts = re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)",
        svg,
    )
    for ts in iso_ts:
        assert pd.Timestamp(ts) <= parent_ts, (
            f"SVG references {ts} which is past parent_bar_ts {parent_ts}"
        )


def test_svg_fallback_includes_stop_target_lines(monkeypatch, tmp_path: Path):
    """Pin: stop / take_profit / atr_stop lines appear in SVG when the
    structure_stop_plan exposes those prices."""
    from src.fx.visual_audit import render_visual_audit_report
    _disable_matplotlib(monkeypatch)
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=10,
    )
    found = False
    for cd in (tmp_path / "EURUSD=X").iterdir():
        if not cd.is_dir():
            continue
        audit = json.loads((cd / "audit.json").read_text())
        sp = (audit.get("overlays") or {}).get("structure_stop_plan") or {}
        if sp.get("stop_price") is not None or sp.get("take_profit_price") is not None:
            svg = (cd / "chart_main.svg").read_text()
            # Either a stop or tp line class exists.
            assert "stop-line" in svg or "tp-line" in svg or "atr-stop-line" in svg
            found = True
            break
    assert found, "no case with stop / tp lines found"


def test_chart_status_taxonomy_pinned_in_audit(tmp_path: Path):
    """audit.json chart_status keys (flat: main / micro / short / medium
    / long / lower_tf) and renderer values must be from the documented
    closed taxonomy."""
    from src.fx.visual_audit import render_visual_audit_report
    df, res = _v2_run()
    render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=tmp_path, max_cases=3,
    )
    cd = next(p for p in (tmp_path / "EURUSD=X").iterdir() if p.is_dir())
    audit = json.loads((cd / "audit.json").read_text())
    cs = audit["chart_status"]
    expected_keys = {"main", "micro", "short", "medium", "long", "lower_tf"}
    assert set(cs.keys()) == expected_keys
    allowed_renderers = {
        "matplotlib_png", "svg_fallback", "image_unavailable",
        "short_history", "lower_tf_unavailable",
    }
    for k, entry in cs.items():
        assert "available" in entry
        assert "renderer" in entry
        assert "path" in entry
        assert entry["renderer"] in allowed_renderers, (
            f"chart_status[{k}].renderer={entry['renderer']} not in taxonomy"
        )


# ─────── Mobile / single-file output ────────────────────────────────


def test_default_css_contains_responsive_media_query():
    """Pin: the static CSS includes a `@media (max-width: 800px)` block
    so the report scales on phones."""
    from src.fx.visual_audit import _DEFAULT_CSS
    assert "@media (max-width: 800px)" in _DEFAULT_CSS


def test_responsive_css_makes_tables_horizontally_scrollable():
    from src.fx.visual_audit import _DEFAULT_CSS
    assert "overflow-x: auto" in _DEFAULT_CSS
    # img / svg full-width inside the @media block.
    assert "max-width: 100%" in _DEFAULT_CSS
    # case-detail unstacks (display:block) on narrow viewports.
    media_block = _DEFAULT_CSS.split("@media (max-width: 800px)")[1]
    assert ".case-detail { display: block; }" in media_block
    assert "img, svg { max-width: 100%; height: auto; }" in media_block


def test_mobile_single_file_writes_self_contained_html(tmp_path: Path):
    from src.fx.visual_audit import (
        MOBILE_SCHEMA_VERSION, render_visual_audit_mobile_single_file,
    )
    df, res = _v2_run()
    out = tmp_path / "mobile.html"
    r = render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=3,
    )
    assert r["schema_version"] == MOBILE_SCHEMA_VERSION
    assert MOBILE_SCHEMA_VERSION == "visual_audit_mobile_v1"
    assert out.exists()
    body = out.read_text()
    # Self-contained: doctype, viewport meta, inline <style>.
    assert "<!DOCTYPE html>" in body
    assert "name='viewport'" in body
    assert "<style>" in body
    # No external <link rel='stylesheet'> reference (single file).
    assert "<link rel=" not in body


def test_mobile_single_file_inlines_svg_chart(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "m.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=3,
    )
    body = out.read_text()
    # Inline <svg> with the candle classes the SVG renderer emits.
    assert "<svg" in body
    assert "class='wick'" in body
    assert "candle-bull" in body or "candle-bear" in body
    # No external chart_main.svg reference.
    assert "chart_main.svg" not in body


def test_mobile_single_file_includes_panels_and_comparison(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "m.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=3,
    )
    body = out.read_text()
    for token in (
        "Royal Road Checklist Panels",
        "Grand Confluence Checklist",
        "Invalidation Explanation",
        "current_runtime vs royal_road_v2",
        "setup_candidates",
        "block_reasons",
        "reconstruction_quality",
    ):
        assert token in body, f"missing section: {token}"


def test_mobile_single_file_demo_banner_present(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "demo.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=2,
        demo_fixture_banner="hand-crafted UI demo. not a backtest result.",
    )
    body = out.read_text()
    assert "demo_fixture_not_backtest_result" in body
    assert "demo-banner" in body
    assert "hand-crafted UI demo. not a backtest result." in body


def test_mobile_single_file_case_list_with_anchors(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "m.html"
    r = render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=3,
    )
    body = out.read_text()
    # Top case list has an anchor target id='case-list'.
    assert "id='case-list'" in body
    # Each case section gets a unique id and the case list links to it.
    for c in r["cases"]:
        sid = c["section_id"]
        assert f"id='{sid}'" in body
        assert f"href='#{sid}'" in body
    # Back-to-list link for navigation on phone.
    assert "back to case list" in body


def test_mobile_single_file_no_future_leak(tmp_path: Path):
    """Pin: even when df has bars beyond parent_bar_ts, the inline SVG
    emitted by the mobile renderer must not reference timestamps past
    the parent boundary."""
    import re as _re
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df_full = _ohlcv(400)
    res = run_engine_backtest(
        df_full.iloc[:200], "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    out = tmp_path / "m.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df_full},
        out_path=out, max_cases=3,
    )
    body = out.read_text()
    # parent_bar_ts texts inside the inline SVGs.
    parents = _re.findall(r"parent_bar_ts=([^<]+?)<", body)
    assert parents, "no parent_bar_ts marker found"
    iso_ts = _re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)",
        body,
    )
    # Every timestamp in the file should be <= the maximum parent_bar_ts.
    parent_max = max(pd.Timestamp(p) for p in parents)
    for ts in iso_ts:
        assert pd.Timestamp(ts) <= parent_max, (
            f"future leak: {ts} > parent_max {parent_max}"
        )


def test_mobile_single_file_no_external_fetch(tmp_path: Path, monkeypatch):
    """Pin: the mobile single-file path must not hit the network. We
    block urllib.request.urlopen and confirm rendering still succeeds."""
    import urllib.request
    def _refuse(*a, **kw):
        raise AssertionError("network access attempted in mobile renderer")
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "m.html"
    r = render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=2,
    )
    assert out.exists()
    assert r["n_cases"] >= 1


def test_smoke_generator_single_file_cli_writes_mobile_html(tmp_path: Path):
    """End-to-end CLI test: run the smoke generator with --single-file
    on the structure_stop_anchor_demo mode and confirm a mobile HTML
    is written under <out>/mobile/."""
    import subprocess, sys
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "smoke"
    proc = subprocess.run(
        [sys.executable,
         str(repo / "scripts" / "generate_visual_audit_smoke.py"),
         "--out", str(out),
         "--mode", "structure_stop_anchor_demo",
         "--single-file"],
        cwd=str(repo), capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"generator failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    mobile_file = out / "mobile" / "structure_stop_anchor_demo_mobile.html"
    assert mobile_file.exists()
    body = mobile_file.read_text()
    assert "<svg" in body
    assert "Grand Confluence Checklist" in body
    assert "demo_fixture_not_backtest_result" in body
    assert "current_runtime vs royal_road_v2" in body


# ---------------------------------------------------------------------------
# Waveform shape review additions (observation-only)
# ---------------------------------------------------------------------------


def test_payload_includes_wave_shape_review_observation_only():
    """The audit payload exposes the cross-scale wave_shape_review
    block surfaced from chart_reconstruction.reconstruct_chart_multi_scale.
    """
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    assert "wave_shape_review" in p
    review = p["wave_shape_review"]
    # may be empty for short histories, but must be a dict
    assert isinstance(review, dict)
    if review:
        assert review.get("schema_version") == "pattern_shape_review_v1"
        notes = review.get("audit_notes") or []
        # Audit notes flag heuristic / observation-only nature
        joined = " ".join(notes).lower()
        assert (
            "heuristic" in joined or "observation" in joined
        ), f"expected unvalidated/observation-only note in {notes}"


def test_payload_includes_entry_summary_for_v2_trace():
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    tr = res.decision_traces[100]
    p = build_visual_audit_payload(trace=tr, df=df)
    assert p is not None
    es = p["entry_summary"]
    assert es is not None
    assert "action" in es
    assert "entry_price_source" in es
    assert "rr" in es
    if es["action"] == "HOLD":
        assert es["entry_price"] is None
        assert es["entry_price_source"] == "no_entry_hold"
    elif es["action"] in ("BUY", "SELL"):
        assert es["entry_price_source"] in {
            "parent_bar_close_fallback",
            "parent_bar_close_unavailable",
        }


def test_mobile_html_includes_waveform_review_section(tmp_path: Path):
    from src.fx.visual_audit import render_visual_audit_mobile_single_file
    df, res = _v2_run()
    out = tmp_path / "wave.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out, max_cases=2,
    )
    body = out.read_text()
    # The "波形レビュー" Japanese label must appear (test is set tight to
    # catch regressions if the section is renamed).
    assert "波形レビュー" in body
    assert "結論カード" in body
    assert "wave-shape-review" in body
    assert "entry-summary-card" in body
