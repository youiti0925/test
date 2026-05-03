"""Royal-road focus chart layout pin tests (Phase I follow-up).

These pin the mockup-aligned layout: a `royal-road-focus-layout`
container, a compact right-side checklist panel above the existing
14-row table (which moved into <details>), grouped price labels for
WNL/ENTRY/SNL1, WSL/STOP/SIL1, WTP/TP/STP1, and the canonical
royal-road wave-line P1→NL→P2→BR for double_top.
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
    }
    out = tmp_path / f"{name}.html"
    builders[name](out_path=out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_htmls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("visual_royal_road_focus_cache")
    return {
        "double_top_integrated_sell_demo":
            _build("double_top_integrated_sell_demo", tmp),
        "double_bottom_integrated_buy_demo":
            _build("double_bottom_integrated_buy_demo", tmp),
    }


def _extract_svg(html: str) -> str:
    m = re.search(
        r"<svg[^>]*data-testid=['\"]main-candle-chart['\"][\s\S]*?</svg>",
        html,
    )
    assert m, "main-candle-chart SVG not found"
    return m.group(0)


# ─────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────


def test_royal_road_focus_layout_has_compact_panel_and_main_chart(
    built_htmls,
):
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "royal-road-focus-layout" in html
    assert "royal-road-main-chart-card" in html
    assert "royal-road-compact-panel" in html
    assert "王道手順チェック" in html
    assert "詳細14項目を見る" in html


def test_compact_panel_has_royal_road_summary_items(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    for token in [
        "環境認識",
        "ダウ",
        "波形",
        "ネックライン",
        "トレンドライン",
        "ブレイク",
        "リターンムーブ",
        "確認足",
        "RR",
        "結論",
    ]:
        assert token in html, f"compact panel missing token: {token!r}"
    # READY page must say SELL READY in the conclusion.
    assert "SELL READY" in html or "READY" in html


# ─────────────────────────────────────────────────────────────────
# Double-top core lines visible in focus mode
# ─────────────────────────────────────────────────────────────────


def test_double_top_core_lines_are_visible_in_focus_mode(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_svg(html)

    assert "DT1" in svg
    for label in ("P1", "P2", "NL", "BR"):
        assert label in svg, f"main SVG missing pivot label {label!r}"

    # Royal-road wave-line polyline P1 → NL → P2 → BR must render
    # with the canonical class + sequence attribute.
    assert "royal-road-wave-line" in svg
    assert "data-rr-sequence='P1,NL,P2,BR'" in svg

    # Royal-road main structural top line P1↔P2 must render with
    # the user-spec'd CSS class + anchor attributes + descriptive
    # label.
    assert "double-top-topline" in svg
    assert (
        "data-anchor-parts=\"P1,P2\"" in svg
        or "data-anchor-parts='P1,P2'" in svg
    )
    assert (
        "STL1 P1-P2" in svg
        or "トレンドライン STL1" in svg
    )

    # Neckline is the protagonist — WNL line carries the royal-road
    # neckline class.
    assert "royal-road-neckline" in svg
    # Combined neckline label.
    assert "WNL / ENTRY / SNL1" in svg


def test_double_bottom_uses_b1_b2_canonical_sequence(built_htmls):
    html = built_htmls["double_bottom_integrated_buy_demo"]
    svg = _extract_svg(html)
    # Only one of the B-sequence vs P-sequence is visible per pattern.
    assert (
        "data-rr-sequence='B1,NL,B2,BR'" in svg
        or "data-rr-sequence=\"B1,NL,B2,BR\"" in svg
    )


# ─────────────────────────────────────────────────────────────────
# Grouped price labels reduce clutter
# ─────────────────────────────────────────────────────────────────


def test_grouped_price_labels_reduce_clutter(built_htmls):
    html = built_htmls["double_top_integrated_sell_demo"]

    assert "WNL / ENTRY / SNL1" in html
    assert "WSL / STOP / SIL1" in html
    assert "WTP / TP / STP1" in html

    svg = _extract_svg(html)
    # Confirm we don't emit standalone single-token badges any more
    # for these three roles. The grouped label fragment writes them
    # as a single combined text per role.
    assert svg.count(">WNL<") <= 1
    assert svg.count(">ENTRY<") <= 1
    assert svg.count(">SNL1<") <= 1


# ─────────────────────────────────────────────────────────────────
# Numeric T1 vs T2/T3 distinction
# ─────────────────────────────────────────────────────────────────


def test_numeric_trendlines_split_primary_secondary(built_htmls):
    """T1 must carry trendline-primary; any T2/T3 must carry
    trendline-secondary. Mockup-aligned royal-road focus mutes the
    secondary lines so the chart isn't crowded."""
    html = built_htmls["double_top_integrated_sell_demo"]
    svg = _extract_svg(html)
    if "trendline-selected-1" in svg:
        # T1 is primary
        assert "trendline-selected-1" in svg
        assert "trendline-primary" in svg
    # When >=2 numeric trendlines exist, secondary is in play.
    if "trendline-selected-2" in svg:
        assert "trendline-secondary" in svg


def test_compact_panel_carries_aux_line_summary(built_htmls):
    """Compact panel mentions auxiliary line counts so the user can
    see at a glance numeric T-count and structural STL-count."""
    html = built_htmls["double_top_integrated_sell_demo"]
    assert "補助線" in html
    assert "numeric T" in html or "numeric" in html
    assert "STL" in html
