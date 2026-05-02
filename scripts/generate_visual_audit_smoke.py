"""Smoke generator for the royal_road_v2 visual audit report.

Three modes (selectable via --mode):

  1. normal_v2_report           — deterministic synthetic OHLC →
                                  run_engine_backtest + royal_road_v2
                                  → visual audit. The default mode.

  2. structure_stop_anchor_demo — handcrafted trace whose v2 payload
                                  has a structure_stop_price anchored
                                  to a selected SR zone, so the
                                  invalidation_explanation panel
                                  produces a concrete structural
                                  sentence (not just "ATR-distance").

  3. current_vs_royal_diff_demo — handcrafted trace where current_runtime
                                  produced BUY but royal_v2 produced
                                  HOLD, so difference_type =
                                  current_buy_royal_hold and the
                                  comparison panel is non-trivial.

Demos 2 and 3 are FIXTURES for UI verification only — they do NOT
represent real backtest output. The orchestrator marks the report
with `demo_fixture_not_backtest_result=True` and renders a
prominent banner at the top of index.html so reviewers cannot
mistake them for live data.

Strict prohibitions (this script never violates these):
  - No external data fetch (no OANDA / yfinance / live)
  - No 90d / 180d / 365d comparison run
  - No live / paper / OANDA path touched
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Make `src` importable when run directly from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.fx.backtest_engine import run_engine_backtest  # noqa: E402
from src.fx.visual_audit import (  # noqa: E402
    render_visual_audit_mobile_single_file,
    render_visual_audit_report,
)


def _synthetic_ohlcv(n: int = 350, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.10 + np.cumsum(rng.standard_normal(n) * 0.0008)
    return pd.DataFrame({
        "open": close - 0.0001,
        "high": close + 0.0008,
        "low": close - 0.0008,
        "close": close,
        "volume": [1000] * n,
    }, index=idx)


# ---------------------------------------------------------------------------
# Mode 1 — normal_v2_report
# ---------------------------------------------------------------------------


def run_normal_v2_report(out_dir: Path, max_cases: int = 25) -> dict:
    df = _synthetic_ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    return render_visual_audit_report(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_dir=out_dir, max_cases=max_cases,
    )


# ---------------------------------------------------------------------------
# Handcrafted v2 payload helpers (for demo modes 2 + 3)
# ---------------------------------------------------------------------------


def _empty_v2_payload(*, action: str, mode: str = "balanced") -> dict:
    """Minimal v2 payload skeleton with all keys the visual_audit
    orchestrator + checklist panels read."""
    return {
        "profile": "royal_road_decision_v2",
        "mode": mode,
        "mode_status": "heuristic_not_validated_default",
        "mode_needs_validation": True,
        "action": action,
        "score": 0.0,
        "reasons": [],
        "block_reasons": [],
        "cautions": [],
        "evidence_axes": {"bullish": {}, "bearish": {}},
        "evidence_axes_count": {"bullish": 0, "bearish": 0},
        "min_evidence_axes_required": None,
        "support_resistance_v2": {
            "near_strong_support": False, "near_strong_resistance": False,
            "breakout": False, "pullback": False, "role_reversal": False,
            "fake_breakout": False, "reason": "demo",
            "selected_level_zones_top5": [],
            "rejected_level_zones": [],
        },
        "trendline_context": {
            "bullish_signal": False, "bearish_signal": False,
            "selected_trendlines_top3": [],
            "rejected_trendlines": [],
        },
        "chart_pattern_v2": {
            "bullish_breakout_confirmed": False,
            "bearish_breakout_confirmed": False,
            "selected_patterns_top5": [],
            "rejected_patterns": [],
        },
        "lower_tf_trigger": {
            "interval": None, "available": False, "reason": "demo_fixture",
            "bullish_trigger": False, "bearish_trigger": False,
        },
        "macro_alignment": {
            "currency_bias": "NEUTRAL", "macro_score": 0.0,
            "macro_strong_against": "UNKNOWN", "vix_regime": "NORMAL",
            "macro_reasons": [], "macro_block_reasons": [],
        },
        "structure_stop_plan": None,
        "setup_candidates": [],
        "best_setup": None,
        "reconstruction_quality": {
            "level_quality_score": 0.0,
            "trendline_quality_score": 0.0,
            "pattern_quality_score": 0.0,
            "lower_tf_quality_score": 0.0,
            "macro_quality_score": 0.0,
            "stop_plan_quality_score": 0.0,
            "total_reconstruction_score": 0.0,
            "weights": {},
        },
        "multi_scale_chart": {"scales": {}},
        "compared_to_current_runtime": {
            "current_action": action,
            "royal_road_action": action,
            "same_action": True,
            "difference_type": "same",
        },
        "source": "demo_fixture",
    }


def _empty_tc(close: float = 1.10) -> dict:
    return {
        "policy_version": "technical_confluence_v1",
        "market_regime": "TREND_UP",
        "dow_structure": {"structure_code": "HL"},
        "support_resistance": {
            "near_support": True, "near_resistance": False,
            "fake_breakout": False, "role_reversal": False,
        },
        "candlestick_signal": {
            "bullish_pinbar": True, "bearish_pinbar": False,
            "bullish_engulfing": False, "bearish_engulfing": False,
            "harami": False, "strong_bull_body": False,
            "strong_bear_body": False, "rejection_wick": False,
        },
        "chart_pattern": {},
        "indicator_context": {
            "rsi_value": 45.0, "rsi_range_valid": True,
            "rsi_trend_danger": False, "macd_momentum_up": True,
            "macd_momentum_down": False, "bb_squeeze": False,
            "bb_expansion": False, "bb_band_walk": False,
            "ma_trend_support": "BUY",
        },
        "risk_plan_obs": {"invalidation_clear": True},
        "vote_breakdown": {
            "indicator_buy_votes": 2, "indicator_sell_votes": 0,
            "voted_action": "BUY", "macro_alignment": "NEUTRAL",
            "htf_alignment": "BUY", "structure_alignment": "BUY",
            "sr_alignment": "BUY", "candle_alignment": "BUY",
        },
        "final_confluence": {
            "label": "STRONG_BUY_SETUP", "score": 0.7,
            "bullish_reasons": ["regime_trend_up", "near_support"],
            "bearish_reasons": [], "avoid_reasons": [],
        },
    }


def _make_demo_trace(
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    v2_payload: dict,
    technical_confluence: dict | None = None,
    close: float = 1.10,
    sma_20: float = 1.099,
    sma_50: float = 1.097,
) -> SimpleNamespace:
    """Wrap a handcrafted v2 payload into a duck-typed trace object the
    visual_audit orchestrator can consume."""
    v2_obj = SimpleNamespace(to_dict=lambda d=v2_payload: d)
    tc_obj = (
        SimpleNamespace(to_dict=lambda d=technical_confluence: d)
        if technical_confluence is not None else None
    )
    technical = SimpleNamespace(
        to_dict=lambda: {
            "rsi_14": 45.0, "macd": 0.0, "macd_signal": 0.0,
            "macd_hist": 0.0, "sma_20": sma_20, "sma_50": sma_50,
            "ema_12": close, "bb_upper": close + 0.005,
            "bb_lower": close - 0.005, "bb_position": 0.5,
            "change_pct_1h": 0.0, "change_pct_24h": 0.0,
            "atr_14": 0.001, "technical_only_action": "BUY",
            "technical_reason_codes": [], "reason_derivation": "demo",
        }
    )
    market = SimpleNamespace(
        to_dict=lambda: {
            "open": close, "high": close + 0.001,
            "low": close - 0.001, "close": close, "volume": 1000,
            "data_source": "demo", "is_complete_bar": True,
            "data_quality": "ok", "bars_available": 250,
            "missing_ohlcv_count": 0, "has_nan": False,
            "index_monotonic": True, "duplicate_timestamp_count": 0,
            "timezone": "UTC", "gap_detected": False,
            "quality_reason": None,
        }
    )
    return SimpleNamespace(
        symbol=symbol,
        timestamp=timestamp,
        royal_road_decision_v2=v2_obj,
        technical_confluence=tc_obj,
        technical=technical,
        market=market,
    )


# ---------------------------------------------------------------------------
# Mode 2 — structure_stop_anchor_demo
# ---------------------------------------------------------------------------


def run_structure_stop_anchor_demo(out_dir: Path) -> dict:
    """Build one handcrafted BUY case where the structure_stop_price
    is anchored to a selected support zone, so the
    invalidation_explanation panel produces a structural sentence."""
    df = _synthetic_ohlcv(n=300, seed=11)
    sym = "EURUSD=X"
    parent_ts = df.index[200]
    close = float(df["close"].iloc[200])
    # Place a selected support zone whose zone_low ≤ stop ≤ zone_high.
    structure_stop = close - 0.005
    zone_low = structure_stop - 0.0008
    zone_high = structure_stop + 0.0008
    v2 = _empty_v2_payload(action="BUY")
    v2["block_reasons"] = []
    v2["evidence_axes"]["bullish"] = {
        "near_strong_support": True, "bullish_structure": True,
    }
    v2["evidence_axes_count"] = {"bullish": 2, "bearish": 0}
    v2["support_resistance_v2"]["near_strong_support"] = True
    v2["support_resistance_v2"]["selected_level_zones_top5"] = [{
        "kind": "support", "price": structure_stop,
        "zone_low": zone_low, "zone_high": zone_high,
        "zone_width_atr": 0.6, "touch_count": 4,
        "first_touch_ts": None, "last_touch_ts": None,
        "first_touch_index": 50, "last_touch_index": 150,
        "broken_count": 0, "role_reversal_count": 0,
        "false_breakout_count": 0, "strength_score": 4.0,
        "recency_score": 0.9, "distance_to_close_atr": 0.4,
        "wick_touch_count": 3, "close_touch_count": 4,
        "body_break_count": 0, "wick_fakeout_count": 0,
        "rejection_count": 3, "confidence": 0.85,
        "reasons": ["strong_multi_touch", "demo_fixture"],
    }]
    v2["structure_stop_plan"] = {
        "chosen_mode": "structure", "outcome": "structure",
        "stop_price": structure_stop, "structure_stop_price": structure_stop,
        "atr_stop_price": close - 0.002,
        "take_profit_price": close + 0.0075,
        "rr_realized": 1.5, "invalidation_reason": None,
    }
    v2["reconstruction_quality"]["total_reconstruction_score"] = 0.62
    v2["reconstruction_quality"]["level_quality_score"] = 0.85
    v2["reconstruction_quality"]["stop_plan_quality_score"] = 1.0
    v2["compared_to_current_runtime"] = {
        "current_action": "BUY", "royal_road_action": "BUY",
        "same_action": True, "difference_type": "same",
    }
    tr = _make_demo_trace(
        symbol=sym, timestamp=parent_ts, v2_payload=v2,
        technical_confluence=_empty_tc(close=close),
        close=close,
    )
    return render_visual_audit_report(
        traces=[tr],
        df_by_symbol={sym: df},
        out_dir=out_dir, max_cases=5,
        demo_fixture_banner=(
            "structure_stop_anchor_demo — handcrafted BUY case where the "
            "structure_stop is anchored to a selected SUPPORT zone. UI "
            "verification only; not a backtest result."
        ),
    )


# ---------------------------------------------------------------------------
# Mode 3 — current_vs_royal_diff_demo
# ---------------------------------------------------------------------------


def run_current_vs_royal_diff_demo(out_dir: Path) -> dict:
    """Build one handcrafted case where current_runtime would BUY but
    royal_v2 holds, so difference_type = current_buy_royal_hold."""
    df = _synthetic_ohlcv(n=250, seed=23)
    sym = "EURUSD=X"
    parent_ts = df.index[180]
    close = float(df["close"].iloc[180])
    v2 = _empty_v2_payload(action="HOLD")
    v2["block_reasons"] = [
        "insufficient_buy_evidence_axes_v2:1<2",
        "insufficient_royal_road_reconstruction_quality",
    ]
    v2["cautions"] = ["macro_score_weak"]
    v2["reconstruction_quality"]["total_reconstruction_score"] = 0.32
    v2["compared_to_current_runtime"] = {
        "current_action": "BUY", "royal_road_action": "HOLD",
        "same_action": False, "difference_type": "current_buy_royal_hold",
    }
    v2["structure_stop_plan"] = {
        "chosen_mode": "atr", "outcome": "atr",
        "stop_price": close - 0.002, "structure_stop_price": None,
        "atr_stop_price": close - 0.002,
        "take_profit_price": close + 0.003,
        "rr_realized": 1.5, "invalidation_reason": None,
    }
    tr = _make_demo_trace(
        symbol=sym, timestamp=parent_ts, v2_payload=v2,
        technical_confluence=_empty_tc(close=close),
        close=close,
    )
    return render_visual_audit_report(
        traces=[tr],
        df_by_symbol={sym: df},
        out_dir=out_dir, max_cases=5,
        demo_fixture_banner=(
            "current_vs_royal_diff_demo — handcrafted case where "
            "current_runtime=BUY but royal_v2=HOLD "
            "(difference_type=current_buy_royal_hold). UI verification "
            "only; not a backtest result."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output directory (e.g. runs/visual_audit/royal_road_v2_smoke_svg)",
    )
    p.add_argument(
        "--mode",
        choices=("normal_v2_report",
                 "structure_stop_anchor_demo",
                 "current_vs_royal_diff_demo",
                 "double_bottom_shape_demo",
                 "double_top_shape_demo",
                 "head_and_shoulders_shape_demo",
                 "inverse_head_and_shoulders_shape_demo",
                 "normal_integrated_balanced_report",
                 "normal_integrated_strict_report",
                 "double_bottom_integrated_buy_demo",
                 "double_top_integrated_sell_demo",
                 "forming_pattern_integrated_hold_demo",
                 "all"),
        default="all",
        help="Which sample mode to generate. 'all' runs every demo "
             "into <out>/<mode> subdirectories.",
    )
    p.add_argument("--max-cases", type=int, default=25)
    p.add_argument(
        "--single-file", action="store_true",
        help="In addition to the regular per-mode reports, write "
             "self-contained mobile-friendly HTML files into "
             "<out>/mobile/<mode>_mobile.html (CSS + chart_main.svg "
             "embedded inline, no external assets).",
    )
    p.add_argument(
        "--mobile-cases", type=int, default=5,
        help="Number of cases per single-file mobile HTML (only used "
             "with --single-file). Defaults to 5; demos always have 1.",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, dict]] = []
    if args.mode in ("normal_v2_report", "all"):
        sub = (args.out / "normal_v2_report"
               if args.mode == "all" else args.out)
        runs.append(("normal_v2_report",
                     run_normal_v2_report(sub, max_cases=args.max_cases)))
    if args.mode in ("structure_stop_anchor_demo", "all"):
        sub = (args.out / "structure_stop_anchor_demo"
               if args.mode == "all" else args.out)
        runs.append(("structure_stop_anchor_demo",
                     run_structure_stop_anchor_demo(sub)))
    if args.mode in ("current_vs_royal_diff_demo", "all"):
        sub = (args.out / "current_vs_royal_diff_demo"
               if args.mode == "all" else args.out)
        runs.append(("current_vs_royal_diff_demo",
                     run_current_vs_royal_diff_demo(sub)))

    print("=" * 60)
    for name, r in runs:
        print(f"[{name}] out={r['out_dir']} n_cases={r['n_cases']}")
        print(f"          summary={r['summary']}")

    if args.single_file:
        mobile_dir = args.out / "mobile"
        mobile_dir.mkdir(parents=True, exist_ok=True)
        mobile_runs: list[tuple[str, dict]] = []
        # normal_v2_report — bundle up to mobile-cases cases.
        if args.mode in ("normal_v2_report", "all"):
            mobile_runs.append((
                "normal_v2_report_mobile.html",
                _build_normal_mobile(
                    out_path=mobile_dir / "normal_v2_report_mobile.html",
                    max_cases=args.mobile_cases,
                ),
            ))
        if args.mode in ("structure_stop_anchor_demo", "all"):
            mobile_runs.append((
                "structure_stop_anchor_demo_mobile.html",
                _build_structure_demo_mobile(
                    out_path=mobile_dir / "structure_stop_anchor_demo_mobile.html",
                ),
            ))
        if args.mode in ("current_vs_royal_diff_demo", "all"):
            mobile_runs.append((
                "current_vs_royal_diff_demo_mobile.html",
                _build_current_diff_demo_mobile(
                    out_path=mobile_dir / "current_vs_royal_diff_demo_mobile.html",
                ),
            ))
        # Pattern-shape deterministic demos (DB / DT / HS / IHS).
        # These are mobile-only — no per-mode multi-file directory.
        for demo_mode, builder in _PATTERN_SHAPE_DEMO_BUILDERS.items():
            if args.mode in (demo_mode, "all"):
                fname = f"{demo_mode}_mobile.html"
                mobile_runs.append((
                    fname,
                    builder(out_path=mobile_dir / fname),
                ))
        # Integrated profile demos (Phase E).
        for demo_mode, builder in _INTEGRATED_DEMO_BUILDERS.items():
            if args.mode in (demo_mode, "all"):
                fname = f"{demo_mode}_mobile.html"
                mobile_runs.append((
                    fname,
                    builder(out_path=mobile_dir / fname),
                ))
        print("=" * 60)
        print(f"mobile single-file output dir: {mobile_dir}")
        for fname, r in mobile_runs:
            print(f"[{fname}] n_cases={r['n_cases']} size={r['size_bytes']:,} bytes")
    return 0


def _build_normal_mobile(*, out_path: Path, max_cases: int) -> dict:
    df = _synthetic_ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2",
    )
    return render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out_path,
        max_cases=max_cases,
        title="visual_audit_mobile_v1 — normal_v2_report",
    )


def _build_structure_demo_mobile(*, out_path: Path) -> dict:
    """Reuse the same handcrafted trace as run_structure_stop_anchor_demo
    so the single-file output mirrors the multi-file demo content."""
    df = _synthetic_ohlcv(n=300, seed=11)
    sym = "EURUSD=X"
    parent_ts = df.index[200]
    close = float(df["close"].iloc[200])
    structure_stop = close - 0.005
    zone_low = structure_stop - 0.0008
    zone_high = structure_stop + 0.0008
    v2 = _empty_v2_payload(action="BUY")
    v2["evidence_axes"]["bullish"] = {
        "near_strong_support": True, "bullish_structure": True,
    }
    v2["evidence_axes_count"] = {"bullish": 2, "bearish": 0}
    v2["support_resistance_v2"]["near_strong_support"] = True
    v2["support_resistance_v2"]["selected_level_zones_top5"] = [{
        "kind": "support", "price": structure_stop,
        "zone_low": zone_low, "zone_high": zone_high,
        "zone_width_atr": 0.6, "touch_count": 4,
        "first_touch_ts": None, "last_touch_ts": None,
        "first_touch_index": 50, "last_touch_index": 150,
        "broken_count": 0, "role_reversal_count": 0,
        "false_breakout_count": 0, "strength_score": 4.0,
        "recency_score": 0.9, "distance_to_close_atr": 0.4,
        "wick_touch_count": 3, "close_touch_count": 4,
        "body_break_count": 0, "wick_fakeout_count": 0,
        "rejection_count": 3, "confidence": 0.85,
        "reasons": ["strong_multi_touch", "demo_fixture"],
    }]
    v2["structure_stop_plan"] = {
        "chosen_mode": "structure", "outcome": "structure",
        "stop_price": structure_stop, "structure_stop_price": structure_stop,
        "atr_stop_price": close - 0.002,
        "take_profit_price": close + 0.0075,
        "rr_realized": 1.5, "invalidation_reason": None,
    }
    v2["reconstruction_quality"]["total_reconstruction_score"] = 0.62
    v2["reconstruction_quality"]["level_quality_score"] = 0.85
    v2["reconstruction_quality"]["stop_plan_quality_score"] = 1.0
    tr = _make_demo_trace(
        symbol=sym, timestamp=parent_ts, v2_payload=v2,
        technical_confluence=_empty_tc(close=close),
        close=close,
    )
    return render_visual_audit_mobile_single_file(
        traces=[tr],
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=1,
        title="visual_audit_mobile_v1 — structure_stop_anchor_demo",
        demo_fixture_banner=(
            "structure_stop_anchor_demo — handcrafted BUY case where "
            "the structure_stop is anchored to a selected SUPPORT "
            "zone. UI verification only; not a backtest result."
        ),
    )


def _build_current_diff_demo_mobile(*, out_path: Path) -> dict:
    df = _synthetic_ohlcv(n=250, seed=23)
    sym = "EURUSD=X"
    parent_ts = df.index[180]
    close = float(df["close"].iloc[180])
    v2 = _empty_v2_payload(action="HOLD")
    v2["block_reasons"] = [
        "insufficient_buy_evidence_axes_v2:1<2",
        "insufficient_royal_road_reconstruction_quality",
    ]
    v2["cautions"] = ["macro_score_weak"]
    v2["reconstruction_quality"]["total_reconstruction_score"] = 0.32
    v2["compared_to_current_runtime"] = {
        "current_action": "BUY", "royal_road_action": "HOLD",
        "same_action": False, "difference_type": "current_buy_royal_hold",
    }
    v2["structure_stop_plan"] = {
        "chosen_mode": "atr", "outcome": "atr",
        "stop_price": close - 0.002, "structure_stop_price": None,
        "atr_stop_price": close - 0.002,
        "take_profit_price": close + 0.003,
        "rr_realized": 1.5, "invalidation_reason": None,
    }
    tr = _make_demo_trace(
        symbol=sym, timestamp=parent_ts, v2_payload=v2,
        technical_confluence=_empty_tc(close=close),
        close=close,
    )
    return render_visual_audit_mobile_single_file(
        traces=[tr],
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=1,
        title="visual_audit_mobile_v1 — current_vs_royal_diff_demo",
        demo_fixture_banner=(
            "current_vs_royal_diff_demo — handcrafted case where "
            "current_runtime=BUY but royal_v2=HOLD "
            "(difference_type=current_buy_royal_hold). UI verification "
            "only; not a backtest result."
        ),
    )


# ---------------------------------------------------------------------------
# Pattern-shape deterministic demos (DB / DT / HS / IHS)
#
# These produce synthetic OHLC matching one of the four named reversal
# patterns and render a single mobile HTML each. They are FIXTURES for
# UI verification of the pattern-dissection chart (DB1/B1/B2/NL/BR/WSL/WTP
# etc.); they are NOT backtest results. The orchestrator marks each one
# with `demo_fixture_not_backtest_result=True`.
#
# The synthetic OHLC follows a piecewise-linear price curve close to the
# corresponding pattern_template, plus light Gaussian noise so the
# encoder's swing detector has something to bite on.
# ---------------------------------------------------------------------------


def _piecewise_curve(knots: list[tuple[float, float]], n: int) -> "np.ndarray":
    """Linear interpolation between (x, y) knots over n points in [0,1]."""
    xs = np.array([k[0] for k in knots], dtype=float)
    ys = np.array([k[1] for k in knots], dtype=float)
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, xs, ys)


def _ohlc_from_curve(
    base: "np.ndarray", *, seed: int = 11, noise: float = 0.005,
    base_price: float = 1.10, scale: float = 0.05,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(base)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = base_price + (base + rng.normal(0.0, noise, size=n)) * scale
    return pd.DataFrame({
        "open": close - 0.0001,
        "high": close + 0.0008,
        "low": close - 0.0008,
        "close": close,
        "volume": [1000] * n,
    }, index=idx)


def _double_bottom_ohlcv(n: int = 320, seed: int = 11) -> pd.DataFrame:
    """Synthetic OHLC matching the double_bottom template.

    Length is 320 bars so the medium scale (last 300) captures the
    full pattern. The pattern itself spans the ENTIRE window so the
    encoder's ZigZag filter sees clean pivots regardless of which
    scale it runs against.
    """
    knots = [
        (0.00, 0.70), (0.20, 0.05), (0.45, 0.55),
        (0.70, 0.10), (1.00, 0.90),
    ]
    return _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)


def _double_top_ohlcv(n: int = 320, seed: int = 13) -> pd.DataFrame:
    knots = [
        (0.00, 0.30), (0.20, 0.95), (0.45, 0.45),
        (0.70, 0.90), (1.00, 0.10),
    ]
    return _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)


def _head_and_shoulders_ohlcv(n: int = 320, seed: int = 17) -> pd.DataFrame:
    knots = [
        (0.00, 0.35), (0.15, 0.75), (0.30, 0.55),
        (0.50, 0.95), (0.70, 0.55), (0.85, 0.75),
        (1.00, 0.10),
    ]
    return _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)


def _inverse_hs_ohlcv(n: int = 320, seed: int = 19) -> pd.DataFrame:
    knots = [
        (0.00, 0.65), (0.15, 0.25), (0.30, 0.45),
        (0.50, 0.05), (0.70, 0.45), (0.85, 0.25),
        (1.00, 0.90),
    ]
    return _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)


def _build_pattern_shape_demo_payload(
    *,
    df: pd.DataFrame,
    parent_idx: int,
    action: str,
    force_kind: str | None = None,
) -> tuple[dict, float, pd.Timestamp]:
    """Compute multi_scale_chart on the visible window of `df` so the
    encoder picks up the pattern, then plug that into a v2 payload
    skeleton so the existing audit pipeline renders it correctly.

    The wave_shape_review and wave_derived_lines are derived inside
    `build_visual_audit_payload` from this multi_scale_chart, so we
    don't need to construct them by hand.

    `force_kind`: when set (e.g. "double_bottom"), forces the
    cross-scale wave_shape_review.best_pattern to that pattern kind.
    For deterministic demos this guarantees the dissection chart
    shows the expected pattern even when DB / IHS or DT / HS tie
    on shape_score. The forced match is computed against the
    largest-scale skeleton that has pivots, with the matcher restricted
    to templates of the requested kind.
    """
    from src.fx.chart_reconstruction import reconstruct_chart_multi_scale
    from src.fx.pattern_shape_matcher import match_skeleton
    from src.fx.pattern_templates import templates_by_kind
    from src.fx.risk import atr as compute_atr
    from src.fx.waveform_encoder import encode_wave_skeleton

    visible = df.iloc[: parent_idx + 1]
    parent_ts = visible.index[-1]
    last_close = float(visible["close"].iloc[-1])
    atr_v = float(compute_atr(visible, 14).iloc[-1])
    multi = reconstruct_chart_multi_scale(
        visible, atr_value=atr_v, last_close=last_close,
    )
    if force_kind:
        # Override wave_shape_review.best_pattern: pick the largest
        # scale that has a non-empty skeleton, restrict the matcher
        # to templates of `force_kind`, and use that match.
        from src.fx.chart_reconstruction import SCALES as _SCALES

        forced_match = None
        forced_scale = None
        for scale_name in ("long", "medium", "short", "micro"):
            bars_required = _SCALES[scale_name]
            if len(visible) < bars_required:
                continue
            sub = visible.iloc[-bars_required:]
            skel = encode_wave_skeleton(
                sub, scale=scale_name, atr_value=atr_v,
            )
            if not skel.pivots:
                continue
            templates = templates_by_kind(force_kind)
            if not templates:
                continue
            matches = match_skeleton(skel, templates=templates)
            if matches and matches[0].shape_score >= 0.5:
                forced_match = matches[0]
                forced_scale = scale_name
                break
        if forced_match is not None:
            review_dict = multi.get("wave_shape_review") or {}
            review_dict["best_pattern"] = forced_match.to_dict()
            # Ensure per_scale entry for forced_scale reflects the
            # forced match so downstream consumers stay consistent.
            per = review_dict.get("per_scale") or {}
            if forced_scale in per:
                per[forced_scale]["best_shape"] = forced_match.kind
                per[forced_scale]["shape_score"] = (
                    float(forced_match.shape_score)
                )
                per[forced_scale]["status"] = forced_match.status
                per[forced_scale]["human_label"] = forced_match.human_label
                per[forced_scale]["trade_context"] = (
                    forced_match.human_explanation
                )
            review_dict["per_scale"] = per
            multi["wave_shape_review"] = review_dict
    v2 = _empty_v2_payload(action=action)
    v2["multi_scale_chart"] = multi
    if action in ("BUY", "SELL"):
        sign = 1.0 if action == "BUY" else -1.0
        stop_off = max(atr_v * 1.5, 0.001)
        tp_off = max(atr_v * 3.0, 0.002)
        v2["structure_stop_plan"] = {
            "chosen_mode": "structure", "outcome": "structure",
            "stop_price": last_close - sign * stop_off,
            "structure_stop_price": last_close - sign * stop_off,
            "atr_stop_price": last_close - sign * stop_off,
            "take_profit_price": last_close + sign * tp_off,
            "rr_realized": 2.0, "invalidation_reason": None,
        }
    else:
        v2["structure_stop_plan"] = None
    return v2, last_close, parent_ts


def _build_pattern_shape_demo_mobile(
    *,
    out_path: Path,
    df: pd.DataFrame,
    parent_idx: int,
    action: str,
    pattern_label: str,
    banner: str,
    force_kind: str | None = None,
) -> dict:
    sym = "EURUSD=X"
    v2, close, parent_ts = _build_pattern_shape_demo_payload(
        df=df, parent_idx=parent_idx, action=action,
        force_kind=force_kind,
    )
    tr = _make_demo_trace(
        symbol=sym, timestamp=parent_ts, v2_payload=v2,
        technical_confluence=_empty_tc(close=close),
        close=close,
    )
    return render_visual_audit_mobile_single_file(
        traces=[tr],
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=1,
        title=f"visual_audit_mobile_v1 — {pattern_label}",
        demo_fixture_banner=banner,
    )


def _build_double_bottom_shape_demo_mobile(*, out_path: Path) -> dict:
    return _build_pattern_shape_demo_mobile(
        out_path=out_path,
        df=_double_bottom_ohlcv(),
        parent_idx=319,
        action="HOLD",
        pattern_label="double_bottom_shape_demo",
        force_kind="double_bottom",
        banner=(
            "double_bottom_shape_demo — synthetic OHLC matching the "
            "double_bottom template. Used to verify pattern-dissection "
            "rendering (DB1 / B1 / B2 / NL / BR / WSL / WTP). Not a "
            "backtest result."
        ),
    )


def _build_double_top_shape_demo_mobile(*, out_path: Path) -> dict:
    return _build_pattern_shape_demo_mobile(
        out_path=out_path,
        df=_double_top_ohlcv(),
        parent_idx=319,
        action="HOLD",
        pattern_label="double_top_shape_demo",
        force_kind="double_top",
        banner=(
            "double_top_shape_demo — synthetic OHLC matching the "
            "double_top template. Used to verify pattern-dissection "
            "rendering (DT1 / P1 / P2 / NL / BR / WSL / WTP). Not a "
            "backtest result."
        ),
    )


def _build_head_and_shoulders_shape_demo_mobile(*, out_path: Path) -> dict:
    return _build_pattern_shape_demo_mobile(
        out_path=out_path,
        df=_head_and_shoulders_ohlcv(),
        parent_idx=319,
        action="HOLD",
        pattern_label="head_and_shoulders_shape_demo",
        force_kind="head_and_shoulders",
        banner=(
            "head_and_shoulders_shape_demo — synthetic OHLC matching "
            "the head_and_shoulders template. Used to verify pattern-"
            "dissection rendering (HS1 / LS / H / RS / NL / BR / WSL / "
            "WTP). Not a backtest result."
        ),
    )


def _build_inverse_head_and_shoulders_shape_demo_mobile(
    *, out_path: Path,
) -> dict:
    return _build_pattern_shape_demo_mobile(
        out_path=out_path,
        df=_inverse_hs_ohlcv(),
        parent_idx=319,
        action="HOLD",
        pattern_label="inverse_head_and_shoulders_shape_demo",
        force_kind="inverse_head_and_shoulders",
        banner=(
            "inverse_head_and_shoulders_shape_demo — synthetic OHLC "
            "matching the inverse_head_and_shoulders template. Used to "
            "verify pattern-dissection rendering (IHS1 / LS / H / RS / "
            "NL / BR / WSL / WTP). Not a backtest result."
        ),
    )


_PATTERN_SHAPE_DEMO_BUILDERS = {
    "double_bottom_shape_demo":
        _build_double_bottom_shape_demo_mobile,
    "double_top_shape_demo":
        _build_double_top_shape_demo_mobile,
    "head_and_shoulders_shape_demo":
        _build_head_and_shoulders_shape_demo_mobile,
    "inverse_head_and_shoulders_shape_demo":
        _build_inverse_head_and_shoulders_shape_demo_mobile,
}


# ---------------------------------------------------------------------------
# Integrated profile (Phase E.4) demos
# ---------------------------------------------------------------------------


def _build_normal_integrated_mobile(
    *, out_path: Path, mode: str, max_cases: int = 5,
) -> dict:
    """Mobile single-file HTML driven by run_engine_backtest with the
    integrated profile. Covers the realistic case mix (mostly HOLD on
    a synthetic random walk) so the user can see how integrated
    decisions render through the full visual_audit pipeline."""
    df = _synthetic_ohlcv()
    res = run_engine_backtest(
        df, "EURUSD=X", interval="1h", warmup=60,
        decision_profile="royal_road_decision_v2_integrated",
        integrated_mode=mode,
    )
    return render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={"EURUSD=X": df},
        out_path=out_path,
        max_cases=max_cases,
        title=f"visual_audit_mobile_v1 — integrated ({mode})",
        demo_fixture_banner=(
            "★ この判断は王道統合ロジックで出ています。\n"
            "波形 / Wライン / フィボ / ダウ / ローソク足 / MA / 損切り / RR を"
            "実際に判断へ使用しています。\n"
            f"mode = {mode}"
        ),
    )


def _double_bottom_with_retest_ohlcv(n: int = 320, seed: int = 11) -> pd.DataFrame:
    """DB curve that includes a post-breakout retest at the neckline,
    ending with a clearly bullish confirmation bar. Designed to
    produce READY/BUY through the integrated profile.
    """
    knots = [
        (0.00, 0.70),  # prior high
        (0.15, 0.05),  # B1
        (0.40, 0.55),  # NL
        (0.65, 0.10),  # B2
        (0.78, 0.62),  # breakout above NL
        (0.86, 0.55),  # retest at NL
        (1.00, 0.95),  # final bullish move
    ]
    df = _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)
    # Force the LAST bar to be a clear bullish body so the candle
    # anatomy review classifies it as bullish_engulfing / bullish_pinbar
    # (the readme-builder default uses tiny bodies).
    last_close = float(df["close"].iloc[-1])
    last_open = last_close - 0.0030  # bullish body
    df.iloc[-1, df.columns.get_loc("open")] = last_open
    df.iloc[-1, df.columns.get_loc("low")] = last_open - 0.0002
    df.iloc[-1, df.columns.get_loc("high")] = last_close + 0.0002
    return df


def _double_top_with_retest_ohlcv(n: int = 320, seed: int = 11) -> pd.DataFrame:
    """DT curve with retest + bearish confirmation bar."""
    knots = [
        (0.00, 0.30),  # prior low
        (0.15, 0.95),  # P1 (top)
        (0.40, 0.45),  # NL (trough between tops)
        (0.65, 0.90),  # P2 (top, similar height)
        (0.78, 0.38),  # breakdown below NL
        (0.86, 0.45),  # retest at NL
        (1.00, 0.05),  # final bearish move
    ]
    df = _ohlc_from_curve(_piecewise_curve(knots, n), seed=seed)
    last_close = float(df["close"].iloc[-1])
    last_open = last_close + 0.0030  # bearish body
    df.iloc[-1, df.columns.get_loc("open")] = last_open
    df.iloc[-1, df.columns.get_loc("high")] = last_open + 0.0002
    df.iloc[-1, df.columns.get_loc("low")] = last_close - 0.0002
    return df


def _build_double_bottom_integrated_buy_demo_mobile(*, out_path: Path) -> dict:
    """Hand-crafted double-bottom + retest OHLCV driven through the
    integrated profile so READY/BUY surfaces in the audit HTML.
    """
    df = _double_bottom_with_retest_ohlcv()
    sym = "EURUSD=X"
    res = run_engine_backtest(
        df, sym, interval="1h", warmup=30,
        decision_profile="royal_road_decision_v2_integrated",
        integrated_mode="integrated_balanced",
    )
    return render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=3,
        title="visual_audit_mobile_v1 — double_bottom integrated BUY demo",
        demo_fixture_banner=(
            "double_bottom + retest デモ。波形認識 → WNL 突破 → リターンムーブ"
            "→ 確定足を経由して、READY / BUY が出るケースを含みます。"
        ),
    )


def _build_double_top_integrated_sell_demo_mobile(*, out_path: Path) -> dict:
    df = _double_top_with_retest_ohlcv()
    sym = "EURUSD=X"
    res = run_engine_backtest(
        df, sym, interval="1h", warmup=30,
        decision_profile="royal_road_decision_v2_integrated",
        integrated_mode="integrated_balanced",
    )
    return render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=3,
        title="visual_audit_mobile_v1 — double_top integrated SELL demo",
        demo_fixture_banner=(
            "double_top + retest デモ。波形認識 → WNL 割れ → 戻り → 確定足"
            "を経由して、READY / SELL が出るケースを含みます。"
        ),
    )


def _build_forming_pattern_integrated_hold_demo_mobile(*, out_path: Path) -> dict:
    """A pattern that hasn't completed yet → wave_lines axis BLOCK →
    HOLD. The HTML shows how integrated reports the HOLD reason."""
    n = 200
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    # Choppy / range-bound — no clear pattern formation
    close = 1.0 + np.cumsum(rng.standard_normal(n) * 0.0005)
    df = pd.DataFrame({
        "open": close, "high": close + 0.0007, "low": close - 0.0007,
        "close": close, "volume": [1000] * n,
    }, index=idx)
    sym = "EURUSD=X"
    res = run_engine_backtest(
        df, sym, interval="1h", warmup=30,
        decision_profile="royal_road_decision_v2_integrated",
        integrated_mode="integrated_strict",
    )
    return render_visual_audit_mobile_single_file(
        traces=res.decision_traces,
        df_by_symbol={sym: df},
        out_path=out_path,
        max_cases=3,
        title="visual_audit_mobile_v1 — forming pattern integrated HOLD demo",
        demo_fixture_banner=(
            "形成中・条件不足のため HOLD になるケースのデモ。\n"
            "未接続データが必須条件のため HOLD (strict)。"
        ),
    )


_INTEGRATED_DEMO_BUILDERS = {
    "normal_integrated_balanced_report":
        lambda out_path: _build_normal_integrated_mobile(
            out_path=out_path, mode="integrated_balanced",
        ),
    "normal_integrated_strict_report":
        lambda out_path: _build_normal_integrated_mobile(
            out_path=out_path, mode="integrated_strict",
        ),
    "double_bottom_integrated_buy_demo":
        _build_double_bottom_integrated_buy_demo_mobile,
    "double_top_integrated_sell_demo":
        _build_double_top_integrated_sell_demo_mobile,
    "forming_pattern_integrated_hold_demo":
        _build_forming_pattern_integrated_hold_demo_mobile,
}


if __name__ == "__main__":
    raise SystemExit(main())
