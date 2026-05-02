"""User-facing HTML sanitization tests (Phase E.3).

The Tier 1 / Tier 2 / Tier 3 rendering must NOT leak developer-style
strings (best=, quality=, score=, NaN, plain "None") into the
HTML the user sees on a phone. This file pins the sanitization
contract for both desktop case detail HTML and mobile case sections.

Raw JSON inside <pre> debug blocks is acceptable — those are wrapped
in <details> by design. The user only opens them when troubleshooting.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.visual_audit import (
    _render_decision_bridge_html,
    _render_masterclass_panels_html,
    _setup_candidate_summary_ja,
    build_visual_audit_payload,
    render_visual_audit_mobile_single_file,
)


# ── Setup candidate summary JA ──────────────────────────────────
def test_setup_candidate_summary_replaces_score_with_japanese():
    s = _setup_candidate_summary_ja(
        0, {"side": "BUY", "score": 0.75, "confidence": 0.6, "label": "STRONG"}
    )
    assert "score=" not in s
    assert "confidence=" not in s
    assert "BUY" in s
    assert "強い" in s


def test_setup_candidate_summary_negative_score_japanese():
    s = _setup_candidate_summary_ja(
        0, {"side": "SELL", "score": -0.300, "confidence": 0.55, "label": ""}
    )
    assert "score=" not in s
    assert "-0.300" not in s
    assert "やや逆" in s
    assert "SELL" in s


def test_setup_candidate_summary_no_score_field():
    s = _setup_candidate_summary_ja(2, {"side": "BUY"})
    assert "—" in s
    assert "候補 #3" in s


# ── Bridge HTML sanitization ────────────────────────────────────
def _v2_slice_minimal(action: str = "BUY") -> dict:
    return {
        "action": action, "profile": "royal_road_decision_v2",
        "block_reasons": [], "structure_stop_plan": {"stop_price": 0.99},
        "macro_alignment": {"macro_score": 0.0},
        "support_resistance_v2": {},
    }


def _payload_minimal() -> dict:
    return {
        "royal_road_decision_v2": _v2_slice_minimal(),
        "wave_shape_review": {"available": True},
    }


def test_bridge_html_no_dev_score_string():
    from src.fx.decision_bridge import build_decision_bridge
    bridge = build_decision_bridge(_payload_minimal())
    html = _render_decision_bridge_html(
        bridge, entry_summary={"entry_price": 1.0, "rr": 2.5},
    )
    # Bridge body should not leak dev format
    assert "best=" not in html
    assert "quality=" not in html
    # The integer-like score literal `score=-0.300` must not appear
    assert not re.search(r"score=-?\d+\.\d{3}", html), (
        "bridge HTML leaks score=N.NNN dev format"
    )


def test_bridge_html_no_raw_none_placeholder():
    """`<p class='placeholder'>None</p>` must not appear — replaced
    with Japanese fallback."""
    from src.fx.decision_bridge import build_decision_bridge
    bridge = build_decision_bridge(_payload_minimal())
    html = _render_decision_bridge_html(
        bridge, entry_summary=None,
    )
    assert ">None<" not in html
    assert "NaN" not in html
    assert "nan</td>" not in html


# ── Masterclass Tier 3 categorization ───────────────────────────
def test_tier3_categorized_blocks_present():
    """Tier 3 panel render must wrap panels in category-level
    <details> blocks (波形・構造 / ローソク足 / MA / 指標 / フィボ /
    監査 / 損切り)."""
    panels_dict = {
        "available": True,
        "panels": {
            "candlestick_anatomy_review": {"available": True,
                                           "bar_type": "neutral_doji",
                                           "direction": "NEUTRAL"},
            "dow_structure_review": {"available": True, "trend": "UP"},
            "ma_context_review": {"available": True},
            "rsi_regime_filter": {"available": True},
            "fibonacci_context_review": {"available": True},
            "grand_confluence_v2": {"available": True},
            "invalidation_engine_v2": {"available": True},
        },
    }
    html = _render_masterclass_panels_html(panels_dict)
    # Tier 3 must wrap panels in category-level details blocks
    assert "masterclass-category" in html
    # At least one of the category labels must appear
    assert any(
        cat in html for cat in (
            "波形・構造", "ローソク足", "MA・グランビル",
            "指標 (BB/RSI/MACD/Div)", "フィボ", "監査・サマリ", "損切り・運用",
        )
    )


def test_tier3_panel_titles_no_dev_leak():
    """Panel titles inside Tier 3 do not contain dev strings."""
    panels_dict = {
        "available": True,
        "panels": {
            "candlestick_anatomy_review": {"available": True,
                                           "bar_type": "neutral_doji",
                                           "direction": "NEUTRAL"},
        },
    }
    html = _render_masterclass_panels_html(panels_dict)
    assert "best=" not in html
    assert "quality=" not in html
    assert not re.search(r"score=-?\d+\.\d{3}", html)


# ── End-to-end mobile HTML smoke ────────────────────────────────
def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.0 + np.cumsum(rng.standard_normal(n) * 0.001)
    return pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_mobile_html_no_user_visible_dev_leak(tmp_path: Path):
    """Run a real backtest and render the mobile HTML; assert that the
    body does not surface developer-style score=, best=, quality=, or
    bare None placeholders to the user.

    Raw JSON inside <pre> blocks is allowed to contain "score":, "None",
    etc. — those are debug details. We only flag leaks OUTSIDE <pre>."""
    df = _ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    out_path = tmp_path / "mobile.html"
    render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out_path,
        max_cases=3,
    )
    html = out_path.read_text(encoding="utf-8")
    # Strip <pre>...</pre> debug blocks (raw JSON allowed there)
    body = re.sub(r"<pre[^>]*>.*?</pre>", "", html, flags=re.DOTALL)
    # Strip <script>/<style> just in case
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL)
    # Now check for leaks
    assert "best=" not in body, "user-visible body leaks 'best='"
    assert "quality=" not in body, "user-visible body leaks 'quality='"
    assert not re.search(r"score=-?\d+\.\d{3}", body), (
        "user-visible body leaks score=N.NNN"
    )
    assert ">None<" not in body, "user-visible body has bare >None< cell"
    # NaN can creep in via float formatting; same restriction
    assert "NaN" not in body
    assert ">nan<" not in body
