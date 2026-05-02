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
from src.fx.visual_audit import render_visual_audit_report  # noqa: E402


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
                 "all"),
        default="all",
        help="Which sample mode to generate. 'all' runs all three "
             "into <out>/normal_v2_report, <out>/structure_stop_anchor_demo, "
             "<out>/current_vs_royal_diff_demo subdirectories.",
    )
    p.add_argument("--max-cases", type=int, default=25)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
