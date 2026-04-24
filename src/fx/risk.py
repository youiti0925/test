"""Risk management: ATR-based stops, Kelly sizing, fixed-fractional sizing.

All functions are pure and deterministic — easy to test, no I/O, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr_stop_loss(
    entry: float, atr_value: float, side: str, multiplier: float = 2.0
) -> float:
    """Stop price offset by `multiplier * ATR` on the adverse side of entry."""
    if atr_value <= 0:
        raise ValueError("atr_value must be positive")
    offset = multiplier * atr_value
    if side == "BUY":
        return entry - offset
    if side == "SELL":
        return entry + offset
    raise ValueError(f"Unknown side: {side}")


def atr_take_profit(
    entry: float,
    atr_value: float,
    side: str,
    multiplier: float = 3.0,
) -> float:
    """Take-profit at `multiplier * ATR` in the favorable direction."""
    if atr_value <= 0:
        raise ValueError("atr_value must be positive")
    offset = multiplier * atr_value
    if side == "BUY":
        return entry + offset
    if side == "SELL":
        return entry - offset
    raise ValueError(f"Unknown side: {side}")


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly-optimal fraction of capital.

    win_rate in [0, 1]; avg_win and avg_loss are absolute positive magnitudes
    (return percents or currency units). Negative or zero expectancy → 0.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    edge = win_rate - (1 - win_rate) / b
    return max(0.0, edge)


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
    cap: float = 0.2,
) -> float:
    """Most practitioners use fractional Kelly (¼ or ½) to reduce variance."""
    k = kelly_fraction(win_rate, avg_win, avg_loss)
    return min(cap, fraction * k)


def position_size(
    capital: float,
    entry: float,
    stop: float,
    risk_pct: float = 0.01,
) -> float:
    """Units of the instrument to buy/sell so the loss at `stop` equals
    `risk_pct` of `capital`. Works for any non-zero entry-stop distance.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    if entry <= 0:
        raise ValueError("entry must be positive")
    distance = abs(entry - stop)
    if distance == 0:
        raise ValueError("entry and stop must differ")
    risk_dollars = capital * risk_pct
    return risk_dollars / distance


@dataclass(frozen=True)
class RiskPlan:
    side: str
    entry: float
    stop: float
    take_profit: float
    size: float
    risk_dollars: float
    reward_to_risk: float

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry": round(self.entry, 6),
            "stop": round(self.stop, 6),
            "take_profit": round(self.take_profit, 6),
            "size": round(self.size, 6),
            "risk_dollars": round(self.risk_dollars, 2),
            "reward_to_risk": round(self.reward_to_risk, 2),
        }


def plan_trade(
    side: str,
    entry: float,
    atr_value: float,
    capital: float,
    stop_mult: float = 2.0,
    tp_mult: float = 3.0,
    risk_pct: float = 0.01,
) -> RiskPlan:
    """Combine ATR-based stops with fixed-fractional sizing."""
    stop = atr_stop_loss(entry, atr_value, side, stop_mult)
    tp = atr_take_profit(entry, atr_value, side, tp_mult)
    size = position_size(capital, entry, stop, risk_pct)
    rr = tp_mult / stop_mult
    return RiskPlan(
        side=side,
        entry=entry,
        stop=stop,
        take_profit=tp,
        size=size,
        risk_dollars=capital * risk_pct,
        reward_to_risk=rr,
    )
