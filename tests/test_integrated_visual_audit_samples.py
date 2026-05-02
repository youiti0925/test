"""Tests for the smoke generator's integrated mobile HTML samples.

Pins:

  - The 5 integrated builders are registered in
    `_INTEGRATED_DEMO_BUILDERS`.
  - The CLI accepts each integrated-mode flag.
  - Each generated HTML carries the "★ この判断は王道統合ロジックで
    出ています" banner and the 4-tier USED / PARTIAL / AUDIT_ONLY /
    NOT_CONNECTED layout.
  - User-facing HTML body has no developer-style leakage (best=,
    quality=, score=N.NNN, bare >None<, NaN).
  - strict-mode HTML mentions strict; balanced-mode HTML mentions
    balanced.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_visual_audit_smoke import (  # noqa: E402
    _INTEGRATED_DEMO_BUILDERS,
)


INTEGRATED_DEMO_NAMES = (
    "normal_integrated_balanced_report",
    "normal_integrated_strict_report",
    "double_bottom_integrated_buy_demo",
    "double_top_integrated_sell_demo",
    "forming_pattern_integrated_hold_demo",
)


def test_all_five_integrated_builders_registered():
    """Phase E.4 requires exactly these 5 mobile demo modes."""
    assert set(_INTEGRATED_DEMO_BUILDERS.keys()) == set(INTEGRATED_DEMO_NAMES)


def test_smoke_generator_cli_accepts_integrated_modes():
    cli_text = (SCRIPTS / "generate_visual_audit_smoke.py").read_text(
        encoding="utf-8",
    )
    for name in INTEGRATED_DEMO_NAMES:
        assert f'"{name}"' in cli_text, (
            f"CLI does not accept --mode {name}"
        )


@pytest.fixture(scope="module")
def integrated_html_dir(tmp_path_factory) -> Path:
    """Run all 5 integrated builders once and cache the HTML output."""
    out = tmp_path_factory.mktemp("integrated_samples")
    out.mkdir(exist_ok=True)
    for name, builder in _INTEGRATED_DEMO_BUILDERS.items():
        builder(out_path=out / f"{name}_mobile.html")
    return out


@pytest.mark.parametrize("name", INTEGRATED_DEMO_NAMES)
def test_integrated_html_carries_banner(
    name: str, integrated_html_dir: Path,
):
    path = integrated_html_dir / f"{name}_mobile.html"
    assert path.exists(), path
    html = path.read_text(encoding="utf-8")
    assert "王道統合ロジック" in html, (
        f"{name}: missing integrated banner"
    )


@pytest.mark.parametrize("name", INTEGRATED_DEMO_NAMES)
def test_integrated_html_has_four_tier_layout(
    name: str, integrated_html_dir: Path,
):
    path = integrated_html_dir / f"{name}_mobile.html"
    html = path.read_text(encoding="utf-8")
    # If 0 cases, the HTML is short and skips the bridge sections —
    # but 4 of the 5 integrated builders MUST produce at least 1 case.
    cases_match = re.search(r"cases=(\d+)", html)
    n_cases = int(cases_match.group(1)) if cases_match else 0
    if n_cases == 0:
        pytest.skip(f"{name}: 0 cases — skipping bridge layout check")
    # All 4 sections must appear at least once
    for header in (
        "1. 最終判断",
        "最終判断に使ったもの",
        "(AUDIT_ONLY)",
        "(NOT_CONNECTED)",
    ):
        assert header in html, (
            f"{name}: missing bridge section header {header!r}"
        )


def test_balanced_html_mentions_balanced(integrated_html_dir: Path):
    html = (integrated_html_dir / "normal_integrated_balanced_report_mobile.html").read_text("utf-8")
    assert "balanced" in html


def test_strict_html_mentions_strict(integrated_html_dir: Path):
    html = (integrated_html_dir / "normal_integrated_strict_report_mobile.html").read_text("utf-8")
    assert "strict" in html


@pytest.mark.parametrize("name", INTEGRATED_DEMO_NAMES)
def test_integrated_html_no_user_visible_dev_leak(
    name: str, integrated_html_dir: Path,
):
    """User-facing body (outside <pre> debug blocks) must be clean."""
    html = (integrated_html_dir / f"{name}_mobile.html").read_text("utf-8")
    body = re.sub(r"<pre[^>]*>.*?</pre>", "", html, flags=re.DOTALL)
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL)
    assert "best=" not in body, f"{name}: leaks 'best=' to user HTML"
    assert "quality=" not in body, f"{name}: leaks 'quality=' to user HTML"
    assert not re.search(r"score=-?\d+\.\d{3}", body), (
        f"{name}: leaks score=N.NNN to user HTML"
    )
    assert ">None<" not in body, f"{name}: leaks bare >None< to user HTML"
    assert "NaN" not in body, f"{name}: leaks NaN to user HTML"


@pytest.mark.parametrize("name", INTEGRATED_DEMO_NAMES)
def test_integrated_html_has_categorized_tier3(
    name: str, integrated_html_dir: Path,
):
    """Tier 3 must be wrapped in masterclass-category collapsible
    blocks (one per category)."""
    html = (integrated_html_dir / f"{name}_mobile.html").read_text("utf-8")
    cases_match = re.search(r"cases=(\d+)", html)
    n_cases = int(cases_match.group(1)) if cases_match else 0
    if n_cases == 0:
        pytest.skip("0 cases — Tier 3 categorization not surfaced")
    assert "masterclass-category" in html, (
        f"{name}: Tier 3 not wrapped in masterclass-category blocks"
    )
    # At least one of the category names must appear
    cats = (
        "波形・構造", "ローソク足", "MA・グランビル",
        "指標 (BB/RSI/MACD/Div)", "フィボ",
        "監査・サマリ", "損切り・運用",
    )
    assert any(cat in html for cat in cats), (
        f"{name}: no masterclass category labels found"
    )
