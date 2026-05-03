"""Visual-audit tests for the structural-lines panel + chart overlay
(Phase I follow-up).

Each preview build is expensive (~60-90 s), so HTMLs are shared
across tests via a module-scoped fixture.
"""
from __future__ import annotations

import re
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
        "double_top_integrated_sell_demo":
            g._build_double_top_integrated_sell_demo_mobile,
        "double_bottom_integrated_buy_demo":
            g._build_double_bottom_integrated_buy_demo_mobile,
        "normal_integrated_balanced_report":
            lambda out_path: g._build_normal_integrated_mobile(
                out_path=out_path, mode="integrated_balanced",
            ),
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_htmls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("visual_structural_lines_cache")
    return {
        "double_top_integrated_sell_demo":
            _build("double_top_integrated_sell_demo", tmp),
        "double_bottom_integrated_buy_demo":
            _build("double_bottom_integrated_buy_demo", tmp),
        "normal_integrated_balanced_report":
            _build("normal_integrated_balanced_report", tmp),
    }


def _extract_main_svg(html: str) -> str:
    pat = (
        r"<svg[^>]*data-testid=['\"]main-candle-chart['\"]"
        r"[\s\S]*?</svg>"
    )
    m = re.search(pat, html)
    return m.group(0) if m else ""


def _extract_panel(html: str) -> str:
    m = re.search(
        r"<div class='g5-row g5-structural-lines-panel'"
        r"[\s\S]*?</div>\s*<div class='g5-row\b",
        html,
    )
    return m.group(0) if m else ""


# ─────────────────────────────────────────────────────────────────
# Section presence
# ─────────────────────────────────────────────────────────────────


def test_visual_audit_shows_structural_lines_panel(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "構造ライン" in html
    assert "structural_lines_v1" in html
    assert "structural_neckline" in html
    assert "structural_invalidation" in html
    assert "structural_target" in html


def test_panel_renumbering_phase_i_followup(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "7. 王道手順チェック" in html
    assert "8. 構造ライン" in html
    assert "9. エントリー候補" in html
    assert "10. 手動線操作" in html


def test_structural_panel_lists_snl_sil_stp_for_double_top(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    panel = _extract_panel(html)
    assert panel, "structural-lines panel not found"
    # IDs of the three canonical wave-derived structural lines.
    assert "SNL1" in panel
    assert "SIL1" in panel
    assert "STP1" in panel


def test_structural_panel_carries_alignment_column(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    panel = _extract_panel(html)
    assert panel
    # Either MATCH / NEAR / CONFLICT / NONE / UNKNOWN must surface.
    assert any(
        a in panel for a in ("MATCH", "NEAR", "CONFLICT", "NONE", "UNKNOWN")
    )


# ─────────────────────────────────────────────────────────────────
# Chart SVG overlay
# ─────────────────────────────────────────────────────────────────


def test_main_svg_contains_structural_line_classes(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_main_svg(html)
    assert svg, "main-candle-chart SVG missing"
    assert "structural-line" in svg
    assert "data-structural-line-id" in svg


def test_main_svg_surfaces_snl1_in_grouped_label(built_htmls):
    """In royal-road focus mode, SNL1 lives inside the grouped
    neckline badge "WNL / ENTRY / SNL1" rather than a separate
    `<text>SNL1</text>` element. The id must still be grep-able
    via the data-structural-line-id attribute on the placeholder
    line element so consumers (audits, scripts) can correlate the
    neckline to the structural-lines payload."""
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_main_svg(html)
    assert "WNL / ENTRY / SNL1" in svg, (
        "grouped neckline label 'WNL / ENTRY / SNL1' missing"
    )
    assert (
        "data-structural-line-id='SNL1'" in svg
        or "data-structural-line-id=\"SNL1\"" in svg
    ), "SNL1 id placeholder element missing from main SVG"


def test_main_svg_distinguishes_numeric_t_labels_from_structural(
    built_htmls,
):
    """T1/T2/T3 (numeric) and STL/SNL/SIL/STP (structural) must
    coexist on the chart. They are emitted under different class
    families so a downstream consumer can pick one set."""
    html = built_htmls["normal_integrated_balanced_report"]
    svg = _extract_main_svg(html)
    assert svg
    # Numeric trendline class is intact (existing pin from geometry tests).
    assert "trendline-selected" in svg
    # Structural overlay sits next to it.
    assert "structural-line" in svg
    # The two label families are in distinct CSS classes.
    assert "trendline-label" in svg
    assert "structural-line-label" in svg


def test_panel_data_attributes_pin_schema_and_count(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "data-structural-lines-schema='structural_lines_v1'" in html
    assert "data-structural-lines-count" in html


def test_double_bottom_panel_renders_snl_sil_stp(built_htmls):
    html = built_htmls["double_bottom_integrated_buy_demo"]
    panel = _extract_panel(html)
    assert panel
    assert "SNL1" in panel
    assert "SIL1" in panel
    assert "STP1" in panel
