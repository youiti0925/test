"""Parameter calibration / sweep (spec §10).

Runs `backtest_engine.run_engine_backtest` against a grid of parameter
overrides and reports a side-by-side comparison. The headline column
is NOT raw return — per spec §10:

> 注意：最も利益が高い設定を採用するのではない。
> 安定していて、パラメータを少し変えても崩れない設定を優先する。

So `stability_score` is the report's front-and-centre metric. It
penalises the highest drawdown, low trade counts, and high losing
streaks; raw return only contributes when the rest of the picture is
sound.

This module is also offline-friendly: it accepts an OHLCV DataFrame
directly and never calls yfinance itself, so it's trivial to run
deterministically in tests.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .backtest_engine import EngineBacktestResult, run_engine_backtest
from .strategy_config import StrategyConfig


@dataclass(frozen=True)
class CalibrationRun:
    """One backtest at one parameter set."""

    overrides: dict
    metrics: dict
    stability_score: float

    def to_dict(self) -> dict:
        return {
            "overrides": dict(self.overrides),
            "stability_score": round(self.stability_score, 4),
            "metrics": dict(self.metrics),
        }


@dataclass
class CalibrationReport:
    base_config: StrategyConfig
    runs: list[CalibrationRun] = field(default_factory=list)

    def best(self) -> CalibrationRun | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.stability_score)

    def to_dict(self) -> dict:
        return {
            "base_config": self.base_config.to_dict(),
            "n_runs": len(self.runs),
            "best": self.best().to_dict() if self.best() else None,
            "runs": [r.to_dict() for r in
                     sorted(self.runs, key=lambda r: r.stability_score, reverse=True)],
        }


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------


def stability_score(metrics: dict) -> float:
    """Combine total_return, max_drawdown, profit_factor, n_trades, max
    losing streak into a single comparable number ∈ roughly [-1, 2].

    Formula (chosen for readability over rigour):

      score = clamp(total_return_pct / 50, -1, 1)
            + 0.5 * clamp(profit_factor / 2, 0, 1)
            - 0.5 * clamp(|max_drawdown_pct| / 30, 0, 1)
            - 0.2 * clamp(max_consecutive_losses / 10, 0, 1)
            + 0.2 * clamp(n_trades / 50, 0, 1)

    A flat run with no trades scores ~0; a high-return run with a
    catastrophic drawdown loses most of its premium; a stable but
    low-return run beats a manic-depressive winner.
    """
    if not metrics or metrics.get("n_trades", 0) == 0:
        return 0.0

    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    ret = float(metrics.get("total_return_pct", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    n = int(metrics.get("n_trades", 0))
    streak = int(metrics.get("max_consecutive_losses", 0))

    score = _clamp(ret / 50.0, -1.0, 1.0)
    score += 0.5 * _clamp(pf / 2.0, 0.0, 1.0)
    score -= 0.5 * _clamp(abs(dd) / 30.0, 0.0, 1.0)
    score -= 0.2 * _clamp(streak / 10.0, 0.0, 1.0)
    score += 0.2 * _clamp(n / 50.0, 0.0, 1.0)
    if not math.isfinite(score):
        return 0.0
    return float(score)


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------


# Limit which parameters we allow the sweep to vary. Adding a new dial
# means listing it here AND in `_apply_override` below — keeping the
# search surface explicit.
SWEEP_KEYS = (
    "stop_atr",
    "tp_atr",
    "max_holding_bars",
    "warmup",
    "use_higher_tf",
)


def _ranges_to_grid(ranges: dict[str, Iterable]) -> list[dict]:
    """{"stop_atr": [1.5, 2.0], "tp_atr": [3.0]} →
       [{"stop_atr": 1.5, "tp_atr": 3.0}, {"stop_atr": 2.0, "tp_atr": 3.0}]
    """
    keys = list(ranges.keys())
    values = [list(ranges[k]) for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def calibrate(
    df: pd.DataFrame,
    *,
    symbol: str,
    interval: str = "1h",
    base: StrategyConfig | None = None,
    ranges: dict[str, Iterable] | None = None,
    events=(),
    use_higher_tf: bool = True,
) -> CalibrationReport:
    """Run a grid of backtests and rank them by `stability_score`.

    `ranges` is `{param_name: iterable_of_values}` where param_name is
    one of `SWEEP_KEYS`. An empty/None `ranges` runs a single baseline
    backtest with the base config — useful as a sanity reference.
    """
    base = base or StrategyConfig(symbol=symbol, interval=interval)
    grid = _ranges_to_grid(ranges) if ranges else [{}]
    report = CalibrationReport(base_config=base)

    for overrides in grid:
        # Validate keys
        bad = [k for k in overrides if k not in SWEEP_KEYS]
        if bad:
            raise ValueError(
                f"Sweep keys must be in {SWEEP_KEYS}; got unknown {bad}"
            )

        kwargs = dict(
            symbol=symbol,
            interval=interval,
            warmup=overrides.get("warmup", 50),
            stop_atr_mult=overrides.get("stop_atr", base.risk.stop_atr),
            tp_atr_mult=overrides.get("tp_atr", base.risk.take_profit_atr),
            max_holding_bars=overrides.get(
                "max_holding_bars", base.risk.max_holding_bars,
            ),
            events=events,
            use_higher_tf=overrides.get("use_higher_tf", use_higher_tf),
        )
        result: EngineBacktestResult = run_engine_backtest(df, **kwargs)
        m = result.metrics()
        report.runs.append(CalibrationRun(
            overrides=overrides,
            metrics=m,
            stability_score=stability_score(m),
        ))

    return report


__all__ = [
    "SWEEP_KEYS",
    "CalibrationRun",
    "CalibrationReport",
    "stability_score",
    "calibrate",
]
