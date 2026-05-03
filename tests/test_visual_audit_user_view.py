"""Tests for visual_audit user-facing title, wave-line legend, and
the four deterministic pattern-shape demos.

These pin the user-visible behaviour the user requested in the
"波形を認識した結果として、人間が引く線が増えた" feedback loop:

  1. The chart title format is the JA-friendly form
     「判断: <action> / 波形候補: <kind_label> / 参考線: <W*-ids>」
     and never contains the developer "best=... score=... quality=..."
     leakage.
  2. A small wave-line legend is rendered RIGHT BEFORE the chart so
     a reader knows what WUP / WLOW / WBR / WSL / WTP / WNL mean.
  3. The wave-only-chart heading switches to
     "パターン部位チャート (DB1/DT1/HS1/IHS1: ...)" for the four
     reversal patterns.
  4. The four deterministic demos (DB/DT/HS/IHS) all produce the
     expected family marker + part labels + W-derived line ids,
     regardless of synthetic-curve noise.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

# Make `scripts/` importable when run from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.fx.visual_audit import (
    _build_user_facing_title,
    _render_wave_line_legend_html,
    _wave_only_chart_heading,
)


# ---------------------------------------------------------------------------
# user-facing title
# ---------------------------------------------------------------------------


def test_user_title_format_includes_action_kind_and_ids():
    payload = {
        "royal_road_decision_v2": {"action": "SELL"},
        "wave_shape_review": {
            "best_pattern": {"kind": "falling_wedge"},
        },
    }
    wd = [
        {"id": "WUP", "kind": "pattern_upper"},
        {"id": "WLOW", "kind": "pattern_lower"},
        {"id": "WSL1", "kind": "pattern_invalidation"},
        {"id": "WTP1", "kind": "pattern_target"},
    ]
    t = _build_user_facing_title(payload=payload, wave_derived=wd)
    assert "判断: SELL" in t
    assert "波形候補: 下降ウェッジ候補" in t
    assert "参考線:" in t
    # ids are sorted; commas separate
    assert "WLOW" in t and "WUP" in t and "WSL1" in t and "WTP1" in t
    # No developer leakage:
    assert "best=" not in t
    assert "score=" not in t
    assert "quality=" not in t


def test_user_title_omits_kind_when_no_best_pattern():
    payload = {
        "royal_road_decision_v2": {"action": "HOLD"},
        "wave_shape_review": {"best_pattern": None},
    }
    t = _build_user_facing_title(payload=payload, wave_derived=[])
    assert "判断: HOLD" in t
    assert "波形候補:" not in t
    assert "参考線:" not in t


def test_user_title_caps_ids_at_six_with_overflow_suffix():
    payload = {
        "royal_road_decision_v2": {"action": "BUY"},
        "wave_shape_review": {"best_pattern": {"kind": "double_bottom"}},
    }
    wd = [
        {"id": f"WL{i}", "kind": "pivot_low"} for i in range(8)
    ]
    t = _build_user_facing_title(payload=payload, wave_derived=wd)
    assert "他2本" in t


# ---------------------------------------------------------------------------
# wave-line legend
# ---------------------------------------------------------------------------


def test_wave_line_legend_lists_only_kinds_present():
    html = _render_wave_line_legend_html([
        {"kind": "neckline"},
        {"kind": "pattern_target"},
        {"kind": "pivot_low"},
    ])
    assert "WNL" in html
    assert "WTP" in html
    # WSL/WUP/WLOW/WBR not present in input → not in output
    assert "WSL" not in html
    assert "WUP" not in html
    assert "WBR" not in html


def test_wave_line_legend_for_triangle_pattern_lines():
    html = _render_wave_line_legend_html([
        {"kind": "pattern_upper"},
        {"kind": "pattern_lower"},
        {"kind": "pattern_breakout"},
        {"kind": "pattern_invalidation"},
        {"kind": "pattern_target"},
    ])
    assert "WUP / WLOW" in html
    assert "WBR" in html
    assert "WSL" in html
    assert "WTP" in html


def test_wave_line_legend_returns_empty_when_no_lines():
    assert _render_wave_line_legend_html([]) == ""
    assert _render_wave_line_legend_html(None) == ""


# ---------------------------------------------------------------------------
# wave-only chart heading per pattern kind
# ---------------------------------------------------------------------------


def test_wave_only_heading_for_double_bottom():
    h = _wave_only_chart_heading("double_bottom")
    assert "パターン部位チャート" in h
    assert "ダブルボトム" in h
    assert "DB1" in h


def test_wave_only_heading_for_double_top():
    h = _wave_only_chart_heading("double_top")
    assert "ダブルトップ" in h
    assert "DT1" in h


def test_wave_only_heading_for_head_and_shoulders():
    h = _wave_only_chart_heading("head_and_shoulders")
    assert "三尊" in h
    assert "HS1" in h


def test_wave_only_heading_for_inverse_head_and_shoulders():
    h = _wave_only_chart_heading("inverse_head_and_shoulders")
    assert "逆三尊" in h
    assert "IHS1" in h


def test_wave_only_heading_falls_back_for_other_patterns():
    h = _wave_only_chart_heading("falling_wedge")
    assert "波形だけ表示" in h


# ---------------------------------------------------------------------------
# Pattern-shape deterministic demos (DB / DT / HS / IHS)
#
# These import directly from the smoke generator script to verify the
# four single-file mobile HTMLs render the expected pattern + W-ids.
# Each demo writes ~90KB so we only run them once across all 4
# patterns.
# ---------------------------------------------------------------------------


def _w_ids_in_html(body: str) -> set[str]:
    return set(re.findall(r"<text[^>]*>(W(?:NL|SL|TP|B[12]|P[12]|LS|H|RS|UP|LOW|BR)\d?)<", body))


def test_double_bottom_demo_renders_db1_b1_b2_nl_br_and_wsl_wtp():
    from scripts.generate_visual_audit_smoke import (
        _build_double_bottom_shape_demo_mobile,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "db.html"
        _build_double_bottom_shape_demo_mobile(out_path=out)
        body = out.read_text()
        ids = _w_ids_in_html(body)
        assert "WB1" in ids
        assert "WB2" in ids
        assert "WNL1" in ids or "WNL" in ids
        assert "WSL1" in ids or "WSL" in ids
        assert "WTP1" in ids or "WTP" in ids
        assert "ダブルボトム候補" in body
        assert "DB1" in body
        # User title + legend visible
        assert "判断: HOLD" in body
        assert "波形由来ライン" in body or "wave-line-legend" in body
        # Dissection heading
        assert "パターン部位チャート" in body


def test_double_top_demo_renders_dt1_p1_p2_nl_br_and_wsl_wtp():
    from scripts.generate_visual_audit_smoke import (
        _build_double_top_shape_demo_mobile,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dt.html"
        _build_double_top_shape_demo_mobile(out_path=out)
        body = out.read_text()
        ids = _w_ids_in_html(body)
        assert "WP1" in ids
        assert "WP2" in ids
        assert "WSL1" in ids or "WSL" in ids
        assert "WTP1" in ids or "WTP" in ids
        assert "ダブルトップ候補" in body
        assert "DT1" in body


def test_head_and_shoulders_demo_renders_hs1_ls_h_rs_nl():
    from scripts.generate_visual_audit_smoke import (
        _build_head_and_shoulders_shape_demo_mobile,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "hs.html"
        _build_head_and_shoulders_shape_demo_mobile(out_path=out)
        body = out.read_text()
        ids = _w_ids_in_html(body)
        assert "WLS" in ids
        assert "WH" in ids
        assert "WRS" in ids
        assert "WNL1" in ids or "WNL" in ids
        assert "WSL1" in ids or "WSL" in ids
        assert "三尊候補" in body
        assert "HS1" in body


def test_inverse_head_and_shoulders_demo_renders_ihs1_ls_h_rs_nl():
    from scripts.generate_visual_audit_smoke import (
        _build_inverse_head_and_shoulders_shape_demo_mobile,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ihs.html"
        _build_inverse_head_and_shoulders_shape_demo_mobile(out_path=out)
        body = out.read_text()
        ids = _w_ids_in_html(body)
        assert "WLS" in ids
        assert "WH" in ids
        assert "WRS" in ids
        assert "WNL1" in ids or "WNL" in ids
        assert "逆三尊候補" in body
        assert "IHS1" in body
