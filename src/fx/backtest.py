"""Event-driven backtest engine.

Walks the dataframe one bar at a time, feeding each bar to the strategy.
Never uses future information. Tracks entries, exits, PnL, and drawdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import Snapshot, build_snapshot, technical_signal


@dataclass
class Position:
    side: str  # BUY or SELL
    entry: float
    entry_ts: pd.Timestamp
    size: float = 1.0


@dataclass
class ClosedTrade:
    side: str
    entry: float
    exit: float
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def metrics(self) -> dict:
        if not self.trades:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
            }
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        equity = np.array([eq for _, eq in self.equity_curve])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        total_return = (equity[-1] / equity[0] - 1.0) * 100 if len(equity) else 0.0
        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades),
            "profit_factor": gross_win / gross_loss,
            "total_return_pct": float(total_return),
            "max_drawdown_pct": float(dd.min() * 100),
            "avg_win_pct": (
                float(np.mean([t.return_pct for t in wins])) if wins else 0.0
            ),
            "avg_loss_pct": (
                float(np.mean([t.return_pct for t in losses])) if losses else 0.0
            ),
        }


SignalFn = Callable[[Snapshot], str]


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    signal_fn: SignalFn = technical_signal,
    initial_cash: float = 10_000.0,
    warmup: int = 50,
) -> BacktestResult:
    """Run a simple long/short backtest.

    At each bar:
      1. Build a snapshot from the window [0 .. i] (past data only).
      2. Call signal_fn to get BUY/SELL/HOLD.
      3. Open a new position if flat and signal is directional.
      4. Close the existing position if the signal flips.

    Exits at the bar's close price. One position at a time.
    """
    result = BacktestResult()
    position: Position | None = None
    cash = initial_cash
    equity = initial_cash

    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        price = float(window["close"].iloc[-1])
        snap = build_snapshot(symbol, window)
        signal = signal_fn(snap)

        if position is not None:
            should_close = (
                (position.side == "BUY" and signal == "SELL")
                or (position.side == "SELL" and signal == "BUY")
            )
            if should_close:
                direction = 1 if position.side == "BUY" else -1
                pnl = (price - position.entry) * position.size * direction
                return_pct = 100 * (price - position.entry) / position.entry * direction
                result.trades.append(
                    ClosedTrade(
                        side=position.side,
                        entry=position.entry,
                        exit=price,
                        entry_ts=position.entry_ts,
                        exit_ts=ts,
                        pnl=pnl,
                        return_pct=return_pct,
                    )
                )
                cash += pnl
                position = None

        if position is None and signal in ("BUY", "SELL"):
            size = cash / price
            position = Position(side=signal, entry=price, entry_ts=ts, size=size)

        if position is not None:
            direction = 1 if position.side == "BUY" else -1
            unrealized = (price - position.entry) * position.size * direction
            equity = cash + unrealized
        else:
            equity = cash
        result.equity_curve.append((ts, equity))

    return result
