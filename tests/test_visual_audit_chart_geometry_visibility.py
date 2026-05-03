"""Phase F/G preview — geometry-aware regression tests.

Locks in the user's contract that the preview chart contains the
ACTUAL line / polyline / circle / rect elements — not just text
substring labels — for every required visual element. Asserts the
existence (count >= N) of <line>, <polyline>, <circle>, <rect>
elements scoped to the SVG matched by data-testid.

Designed as a guard against the "文字だけテスト通過" regression where
substring assertions like `"DB1" in svg` could pass even when no
chart geometry was rendered.
"""
from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_smoke():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import generate_visual_audit_smoke as g
    return g


def _build(name: str, tmp_path: Path) -> str:
    g = _import_smoke()
    builders = {
        "double_bottom_integrated_buy_demo":
            g._build_double_bottom_integrated_buy_demo_mobile,
        "double_top_integrated_sell_demo":
            g._build_double_top_integrated_sell_demo_mobile,
        "wait_breakout_demo":  g._build_wait_breakout_demo_mobile,
        "wait_retest_demo":    g._build_wait_retest_demo_mobile,
        "wait_event_clear_demo": g._build_wait_event_clear_demo_mobile,
        "normal_integrated_balanced_report":
            lambda out_path: g._build_normal_integrated_mobile(
                out_path=out_path, mode="integrated_balanced",
            ),
        "normal_integrated_strict_report":
            lambda out_path: g._build_normal_integrated_mobile(
                out_path=out_path, mode="integrated_strict",
            ),
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


def extract_svg_by_testid(html: str, testid: str) -> str:
    """Return the FIRST <svg ... data-testid="<testid>" ...>...</svg>
    block, or empty string if not found."""
    pat = (
        rf'<svg[^>]*data-testid=["\']{re.escape(testid)}["\']'
        rf'[\s\S]*?</svg>'
    )
    m = re.search(pat, html)
    return m.group(0) if m else ""


def count_svg_elements(svg: str, tag: str, *, class_contains: str) -> int:
    """Count SVG elements of `tag` whose class attr contains
    `class_contains` as a whitespace-separated token. Counts both
    <tag .../> and <tag ...> ... </tag> forms."""
    pat = (
        rf"<{tag}\b[^>]*class=['\"][^'\"]*\b"
        + re.escape(class_contains)
        + r"\b[^'\"]*['\"]"
    )
    return len(re.findall(pat, svg))


# ─────────────────────────────────────────────────────────────────────
# double_bottom READY  (the canonical happy-path fixture)
# ─────────────────────────────────────────────────────────────────────


def test_double_bottom_main_svg_has_skeleton_polyline_geometry(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg, "main-candle-chart SVG missing"
    assert count_svg_elements(svg, "polyline",
                              class_contains="wave-skeleton-line") >= 1, (
        "main SVG must have at least 1 wave-skeleton-line polyline"
    )
    # 4-pivot double-bottom must have at least 4 pivot dots when the
    # skeleton is genuine; multi-scale skeletons usually have more.
    assert count_svg_elements(svg, "circle",
                              class_contains="wave-pivot-dot") >= 4, (
        "main SVG must have ≥ 4 wave-pivot-dot circles"
    )


def test_double_bottom_main_svg_has_w_lines_geometry(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    for cls in ("wnl-line", "wsl-line", "wtp-line"):
        n = count_svg_elements(svg, "line", class_contains=cls)
        assert n >= 1, (
            f"main SVG must have at least 1 <line> with class '{cls}'; "
            f"got {n}"
        )


def test_double_bottom_main_svg_has_entry_stop_tp_geometry(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    n_entry = count_svg_elements(svg, "line",
                                 class_contains="entry-line")
    n_stop  = count_svg_elements(svg, "line",
                                 class_contains="stop-line")
    n_tp    = count_svg_elements(svg, "line",
                                 class_contains="tp-line")
    assert n_entry >= 1, f"entry-line geometry missing (count={n_entry})"
    assert n_stop  >= 1, f"stop-line geometry missing (count={n_stop})"
    assert n_tp    >= 1, f"tp-line geometry missing (count={n_tp})"


def test_double_bottom_main_svg_has_db1_and_part_labels(tmp_path):
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    for tok in ("DB1", "B1", "B2", "NL", "BR"):
        assert tok in svg, f"main SVG missing token {tok!r}"


def test_double_bottom_pivot_dots_carry_data_price_attrs(tmp_path):
    """Geometry coords come from real prices/idx, not from fixed coords.
    Spot-check via data-price="..." data-idx="..." attributes."""
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert "data-price=" in svg, "wave-pivot-dot missing data-price attr"
    assert "data-idx=" in svg,   "wave-pivot-dot missing data-idx attr"


# ─────────────────────────────────────────────────────────────────────
# double_top READY
# ─────────────────────────────────────────────────────────────────────


def test_double_top_main_svg_has_skeleton_and_w_lines(tmp_path):
    html = _build("double_top_integrated_sell_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert count_svg_elements(svg, "polyline",
                              class_contains="wave-skeleton-line") >= 1
    assert count_svg_elements(svg, "circle",
                              class_contains="wave-pivot-dot") >= 4
    for cls in ("wnl-line", "wsl-line", "wtp-line"):
        assert count_svg_elements(svg, "line", class_contains=cls) >= 1
    for cls in ("entry-line", "stop-line", "tp-line"):
        assert count_svg_elements(svg, "line", class_contains=cls) >= 1
    for tok in ("DT1", "P1", "P2", "NL", "BR"):
        assert tok in svg


# ─────────────────────────────────────────────────────────────────────
# Trendlines + SR — must be present in the normal_balanced fixture
# (this is the synthetic random-walk preset where detect_trendlines /
# detect_levels actually find structures). At least one preview must
# exercise these element types.
# ─────────────────────────────────────────────────────────────────────


def test_normal_balanced_main_svg_has_trendline_geometry(tmp_path):
    html = _build("normal_integrated_balanced_report", tmp_path)
    # Check the FIRST main-candle-chart SVG (case-1)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    n = count_svg_elements(svg, "line",
                           class_contains="trendline-selected")
    assert n >= 1, (
        f"normal_balanced case-1 must show ≥ 1 trendline-selected "
        f"<line>; got {n}. Without this no preview demonstrates "
        f"trendline rendering geometry."
    )
    # T1 label must be near the trendline
    assert "T1" in svg, "T1 label missing next to trendline"


def test_normal_balanced_at_least_one_case_has_sr_zone_geometry(tmp_path):
    """At least one case in normal_balanced must render an SR zone
    rect, otherwise the SR rendering pipeline is unverified."""
    html = _build("normal_integrated_balanced_report", tmp_path)
    main_svgs = re.findall(
        r'<svg[^>]*data-testid=["\']main-candle-chart["\'][\s\S]*?</svg>',
        html,
    )
    assert main_svgs, "no main-candle-chart SVGs rendered"
    found_sr = False
    for svg in main_svgs:
        n = count_svg_elements(svg, "rect",
                               class_contains="sr-zone-selected")
        if n >= 1:
            found_sr = True
            break
    assert found_sr, (
        "no case in normal_balanced rendered any sr-zone-selected "
        "rect — SR rendering geometry untested"
    )


def test_zero_warning_when_no_trendlines_or_sr(tmp_path):
    """When a fixture has 0 trendlines / 0 SR, the chart MUST emit a
    'X 検出: 0本' warning text (no silent empty)."""
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    n_tl = count_svg_elements(svg, "line",
                              class_contains="trendline-selected")
    n_sr = count_svg_elements(svg, "rect",
                              class_contains="sr-zone-selected")
    if n_tl == 0:
        assert "トレンドライン検出: 0本" in svg, (
            "0 trendlines but no warning text in chart"
        )
    if n_sr == 0:
        assert "サポレジ検出: 0本" in svg, (
            "0 SR zones but no warning text in chart"
        )


# ─────────────────────────────────────────────────────────────────────
# WAIT_BREAKOUT / WAIT_RETEST / WAIT_EVENT_CLEAR — wave-shape geometry
# must remain visible even in WAIT states.
# ─────────────────────────────────────────────────────────────────────


def test_wait_breakout_main_svg_keeps_skeleton_and_w_lines(tmp_path):
    html = _build("wait_breakout_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert count_svg_elements(svg, "polyline",
                              class_contains="wave-skeleton-line") >= 1
    assert count_svg_elements(svg, "line",
                              class_contains="wnl-line") >= 1
    assert "WAIT BREAKOUT" in svg


def test_wait_retest_main_svg_keeps_skeleton_and_w_lines(tmp_path):
    html = _build("wait_retest_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert count_svg_elements(svg, "polyline",
                              class_contains="wave-skeleton-line") >= 1
    assert count_svg_elements(svg, "line",
                              class_contains="wnl-line") >= 1
    assert "WAIT RETEST" in svg


def test_wait_event_clear_main_svg_keeps_skeleton_w_lines_and_event(
    tmp_path,
):
    """The user's hard contract: WAIT_EVENT_CLEAR must NOT wipe the
    technical setup. The chart must still show wave skeleton +
    WNL/WSL/WTP lines + the event band, AND the WAIT EVENT badge."""
    html = _build("wait_event_clear_demo", tmp_path)
    svg = extract_svg_by_testid(html, "main-candle-chart")
    assert svg
    assert count_svg_elements(svg, "polyline",
                              class_contains="wave-skeleton-line") >= 1
    assert count_svg_elements(svg, "line",
                              class_contains="wnl-line") >= 1
    assert count_svg_elements(svg, "line",
                              class_contains="wsl-line") >= 1
    assert count_svg_elements(svg, "line",
                              class_contains="wtp-line") >= 1
    assert count_svg_elements(svg, "rect",
                              class_contains="event-window") >= 1
    assert "WAIT EVENT" in svg
    assert "EVENT" in svg


# ─────────────────────────────────────────────────────────────────────
# Side-panel statistics: detection counts surface in the right panel
# ─────────────────────────────────────────────────────────────────────


def test_side_panel_detection_stats_section_present(tmp_path):
    """6b. 検出統計 section must show selected/rejected counts so the
    user can audit whether 0 is genuine or a routing failure."""
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    assert "6b. 検出統計" in html
    # Must show both rows
    assert "support_resistance" in html
    assert "trendlines" in html


# ─────────────────────────────────────────────────────────────────────
# Visualisation debug count table beneath each chart
# ─────────────────────────────────────────────────────────────────────


def test_chart_debug_count_table_present(tmp_path):
    """The 可視化デバッグ <details> block must appear under each
    rendered case, listing element counts so the user can verify
    geometry without opening DevTools."""
    html = _build("double_bottom_integrated_buy_demo", tmp_path)
    assert "可視化デバッグ" in html
    assert "wave-skeleton-line (polyline)" in html
    assert "wave-pivot-dot   (circle)" in html
    assert "trendline-selected (line)" in html
    assert "sr-zone-selected (rect)" in html
    assert "wave overlay source" in html
