"""Decision Engine-based backtest.

Replays bars one at a time and feeds each one through the SAME rule chain
that production trade/analyze uses: Risk Gate → patterns → higher-TF
alignment → risk-reward floor → (advisory LLM, opt-in). A bar produces
a position only when `decide()` returns BUY/SELL — exactly mirroring
live behaviour.

Why a separate file from `backtest.py`?
---------------------------------------
The legacy `backtest.py` runs the technical-only signal and is useful as
a quick sanity baseline. This module stacks on the full safety chain so
you can directly compare:

  legacy "what would the indicators alone have done?"
    vs.
  engine "what would the production engine have done?"

Two backtests on the same bars; the gap between them tells you what the
gate is actually saving you from. Per spec §4 / §18, the engine version
is the one to trust before letting any rule change reach live.

Point-in-time correctness
-------------------------
Every step uses `df.iloc[: i + 1]` exclusively. We never peek past `i`.
The engine itself relies on `patterns.detect_swings`, which already pins
"no future leak" via test_patterns_no_future_leak. Higher-timeframe
trend uses the same window resampled — it will not consult bars to the
right of `i`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .calendar import Event
from .decision_engine import decide as decide_action
from .higher_timeframe import HIGHER_INTERVAL_MAP
from .indicators import build_snapshot, technical_signal
from .patterns import TrendState, analyse as analyse_patterns
from .risk import atr as compute_atr
from .risk_gate import RiskState


@dataclass
class EnginePosition:
    side: str       # BUY | SELL
    entry: float
    entry_ts: pd.Timestamp
    size: float
    stop: float
    take_profit: float
    bars_held: int = 0


@dataclass
class EngineTrade:
    side: str
    entry: float
    exit: float
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl: float
    return_pct: float
    bars_held: int
    # Possible values: "stop" | "take_profit" | "max_holding" | "end_of_data".
    # NOTE: "flip" (close-on-opposite-signal) is INTENTIONALLY NOT
    # implemented — the engine relies on stop / TP / max_holding to
    # frame each trade. The legacy `backtest.py` is the flip-on-signal
    # path; pick that if you want flip semantics for comparison.
    exit_reason: str
    rule_chain: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()


@dataclass
class EngineBacktestResult:
    trades: list[EngineTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    # Counts of why a bar produced no entry — invaluable for understanding
    # what the gate is filtering. Keys are decision.blocked_by codes plus
    # synthetic "hold_no_signal" / "hold_pattern" / "hold_other" buckets.
    hold_reasons: dict[str, int] = field(default_factory=dict)
    bars_processed: int = 0

    def metrics(self) -> dict:
        bars = self.bars_processed or 1
        if not self.trades:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "hold_rate": 1.0,
                "hold_reasons": dict(self.hold_reasons),
                "bars_processed": self.bars_processed,
            }
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        equity = np.array([eq for _, eq in self.equity_curve])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        total_return = (equity[-1] / equity[0] - 1.0) * 100 if len(equity) else 0.0

        # Largest losing streak — survival metric, complements win_rate
        max_streak = streak = 0
        for t in self.trades:
            if t.pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

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
            "max_consecutive_losses": max_streak,
            "avg_holding_bars": float(np.mean([t.bars_held for t in self.trades])),
            "hold_rate": 1.0 - (len(self.trades) / bars),
            "hold_reasons": dict(self.hold_reasons),
            "exit_reasons": _count_by_attr(self.trades, "exit_reason"),
            "bars_processed": self.bars_processed,
        }


def _count_by_attr(trades: list[EngineTrade], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        key = getattr(t, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


# A user-supplied function that, given the window up to bar i, returns
# either a TradeSignal-like advisory or None. Backtest doesn't call the
# real LLM — too expensive and non-deterministic — so this is normally
# left as `None` (advisor disabled). It exists for parity tests.
LLMSignalFn = Callable[[pd.DataFrame], object]


def _resample_higher_tf(df_window: pd.DataFrame, base_interval: str) -> str:
    """Compute higher-timeframe trend without re-fetching from yfinance.

    Resamples the in-memory window and re-runs `analyse` — keeps the
    backtest fully offline AND point-in-time. Returns "UNKNOWN" if the
    base interval has no entry in the map (e.g. unsupported TF).

    Map mirrors the live-side `HIGHER_INTERVAL_MAP` so 30m/2h/4h
    backtests don't silently degrade to UNKNOWN — pandas freq aliases
    are spelled here (e.g. "4H" not "4h" since pandas needs uppercase
    for the hour alias when not paired with "min"; "1W" for weekly).
    """
    rule = {
        "1m": "15min",
        "5m": "1h",
        "15m": "4h",
        "30m": "4h",
        "1h": "1D",
        "2h": "1D",
        "4h": "1W",
        "1d": "1W",
    }.get(base_interval)
    if rule is None or len(df_window) < 30:
        return "UNKNOWN"
    try:
        higher = df_window.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    if len(higher) < 10:
        return "UNKNOWN"
    pat = analyse_patterns(higher)
    return pat.trend_state.value


def _bar_event_window(events: tuple[Event, ...], ts: pd.Timestamp) -> tuple[Event, ...]:
    """Events whose `when` is within ±48h of `ts` — fed to RiskGate."""
    if not events:
        return ()
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    target = ts.to_pydatetime()
    return tuple(
        e for e in events
        if abs((e.when - target).total_seconds()) <= 48 * 3600
    )


def run_engine_backtest(
    df: pd.DataFrame,
    symbol: str,
    *,
    interval: str = "1h",
    initial_cash: float = 10_000.0,
    warmup: int = 50,
    stop_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    max_holding_bars: int = 48,
    events: Iterable[Event] = (),
    llm_signal_fn: LLMSignalFn | None = None,
    use_higher_tf: bool = True,
) -> EngineBacktestResult:
    """Run a Decision Engine-driven backtest over `df`.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a tz-aware DatetimeIndex.
    interval:
        Base timeframe ("1h", "15m", ...). Used to compute the
        higher-timeframe trend in-memory.
    warmup:
        Number of leading bars skipped — the engine needs swings to form,
        and the gate's data_quality check requires ≥50 bars.
    stop_atr_mult / tp_atr_mult:
        Stop and take-profit are placed at `mult * ATR` from entry. Risk-
        reward fed to the engine is `tp_atr_mult / stop_atr_mult`, mirroring
        `risk.plan_trade`.
    max_holding_bars:
        Force-close after N bars even if neither stop nor TP fired.
        Without this, the win rate gets inflated by trades that never close.
    events:
        Tuple of macro events. The engine sees only events within ±48h of
        the bar — same window the live gate uses. (Calendar freshness is
        N/A in offline backtests.)
    llm_signal_fn:
        Optional callable producing an advisory TradeSignal per bar.
        Default `None` keeps the test deterministic.
    use_higher_tf:
        Compute higher-TF trend per bar (slightly slower). Default True.
    """
    result = EngineBacktestResult()
    pos: EnginePosition | None = None
    cash = initial_cash
    equity = initial_cash
    pending_chain: tuple[str, ...] = ()
    pending_blocked: tuple[str, ...] = ()

    # ATR is monotonic-suffix — compute once over the full df. Each bar
    # only reads atr_series.iloc[i], no future leak.
    atr_series = compute_atr(df, period=14)
    events_tuple = tuple(events)
    risk_reward = tp_atr_mult / stop_atr_mult

    for i in range(warmup, len(df)):
        result.bars_processed += 1
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        price = float(window["close"].iloc[-1])

        # Mark-to-market the existing position first; stops/TPs win over
        # any new signal on the same bar.
        if pos is not None:
            pos.bars_held += 1
            high = float(window["high"].iloc[-1])
            low = float(window["low"].iloc[-1])
            exit_reason: str | None = None
            exit_price: float | None = None
            if pos.side == "BUY":
                if low <= pos.stop:
                    exit_reason, exit_price = "stop", pos.stop
                elif high >= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit
            else:  # SELL
                if high >= pos.stop:
                    exit_reason, exit_price = "stop", pos.stop
                elif low <= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit

            if exit_reason is None and pos.bars_held >= max_holding_bars:
                exit_reason, exit_price = "max_holding", price

            if exit_reason is not None and exit_price is not None:
                direction = 1 if pos.side == "BUY" else -1
                pnl = (exit_price - pos.entry) * pos.size * direction
                ret_pct = 100 * (exit_price - pos.entry) / pos.entry * direction
                result.trades.append(
                    EngineTrade(
                        side=pos.side,
                        entry=pos.entry,
                        exit=exit_price,
                        entry_ts=pos.entry_ts,
                        exit_ts=ts,
                        pnl=pnl,
                        return_pct=ret_pct,
                        bars_held=pos.bars_held,
                        exit_reason=exit_reason,
                        rule_chain=pending_chain,
                        blocked_by=pending_blocked,
                    )
                )
                cash += pnl
                pos = None

        # Build the engine inputs from the window only. Skip bars where
        # ATR is still NaN — engine will refuse the trade anyway.
        atr_value = atr_series.iloc[i]
        if pd.isna(atr_value) or atr_value <= 0:
            _bump(result.hold_reasons, "atr_unavailable")
            equity = cash + _unrealized(pos, price)
            result.equity_curve.append((ts, equity))
            continue

        snap = build_snapshot(symbol, window)
        tech = technical_signal(snap)
        pattern = analyse_patterns(window)
        higher_tf = (
            _resample_higher_tf(window, interval) if use_higher_tf else "UNKNOWN"
        )

        # Approximate sentiment is unavailable in backtest — leave None;
        # gate skips. Spread is None too (no Bid/Ask history). The point
        # of this backtest is the rule chain, not execution simulation.
        risk_state = RiskState(
            df=window,
            events=_bar_event_window(events_tuple, ts),
            spread_pct=None,
            sentiment_snapshot=None,
            now=ts.to_pydatetime() if ts.tzinfo else ts.tz_localize(
                timezone.utc
            ).to_pydatetime(),
        )

        llm_signal = llm_signal_fn(window) if llm_signal_fn is not None else None

        decision = decide_action(
            technical_signal=tech,
            pattern=pattern,
            higher_timeframe_trend=higher_tf,
            risk_reward=risk_reward,
            risk_state=risk_state,
            llm_signal=llm_signal,  # type: ignore[arg-type]
        )

        # Categorise the HOLD reason — the gate's blocked_by codes are
        # already a closed taxonomy; otherwise bucket by the last rule
        # chain step we got to before bailing.
        if decision.action == "HOLD":
            if decision.blocked_by:
                for code in decision.blocked_by:
                    _bump(result.hold_reasons, code)
            else:
                last_step = decision.rule_chain[-1] if decision.rule_chain else "unknown"
                _bump(result.hold_reasons, f"hold_{last_step}")

        # Don't open a second position if one is already running. A flip
        # from BUY to SELL on the same bar is rare and would over-trade
        # in a backtest — close on flip is handled at the NEXT bar via
        # mark-to-market once the new direction's stop/TP frames it.
        if decision.action in ("BUY", "SELL") and pos is None:
            offset = stop_atr_mult * float(atr_value)
            tp_offset = tp_atr_mult * float(atr_value)
            if decision.action == "BUY":
                stop = price - offset
                tp = price + tp_offset
            else:
                stop = price + offset
                tp = price - tp_offset
            # Position size: full cash / price — the legacy backtest's
            # convention. Real sizing is risk-fraction; we keep this
            # comparable to backtest.py for direct A/B reads.
            size = cash / price
            pos = EnginePosition(
                side=decision.action,
                entry=price,
                entry_ts=ts,
                size=size,
                stop=stop,
                take_profit=tp,
            )
            pending_chain = decision.rule_chain
            pending_blocked = decision.blocked_by

        equity = cash + _unrealized(pos, price)
        result.equity_curve.append((ts, equity))

    # Force-close anything still open at the end so equity / trade counts
    # don't omit the final position's PnL.
    if pos is not None:
        ts = df.index[-1]
        price = float(df["close"].iloc[-1])
        direction = 1 if pos.side == "BUY" else -1
        pnl = (price - pos.entry) * pos.size * direction
        ret_pct = 100 * (price - pos.entry) / pos.entry * direction
        result.trades.append(
            EngineTrade(
                side=pos.side,
                entry=pos.entry,
                exit=price,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                pnl=pnl,
                return_pct=ret_pct,
                bars_held=pos.bars_held,
                exit_reason="end_of_data",
                rule_chain=pending_chain,
                blocked_by=pending_blocked,
            )
        )

    return result


def _unrealized(pos: EnginePosition | None, price: float) -> float:
    if pos is None:
        return 0.0
    direction = 1 if pos.side == "BUY" else -1
    return (price - pos.entry) * pos.size * direction


def _bump(d: dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1


__all__ = [
    "EnginePosition",
    "EngineTrade",
    "EngineBacktestResult",
    "run_engine_backtest",
    "HIGHER_INTERVAL_MAP",
]
