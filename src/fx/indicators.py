"""Pure technical indicators — no external deps beyond pandas/numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std}
    )


@dataclass(frozen=True)
class Snapshot:
    """Compact numeric summary passed to the LLM."""

    symbol: str
    last_close: float
    change_pct_1h: float
    change_pct_24h: float
    sma_20: float
    sma_50: float
    ema_12: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    bb_position: float  # 0..1 where price sits between bands

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "last_close": round(self.last_close, 6),
            "change_pct_1h": round(self.change_pct_1h, 4),
            "change_pct_24h": round(self.change_pct_24h, 4),
            "sma_20": round(self.sma_20, 6),
            "sma_50": round(self.sma_50, 6),
            "ema_12": round(self.ema_12, 6),
            "rsi_14": round(self.rsi_14, 2),
            "macd": round(self.macd, 6),
            "macd_signal": round(self.macd_signal, 6),
            "macd_hist": round(self.macd_hist, 6),
            "bb_upper": round(self.bb_upper, 6),
            "bb_lower": round(self.bb_lower, 6),
            "bb_position": round(self.bb_position, 3),
        }


def build_snapshot(symbol: str, df: pd.DataFrame) -> Snapshot:
    """Compute all indicators and return a point-in-time snapshot."""
    close = df["close"]
    macd_df = macd(close)
    bb = bollinger(close)

    last_close = float(close.iloc[-1])
    bb_upper = float(bb["bb_upper"].iloc[-1])
    bb_lower = float(bb["bb_lower"].iloc[-1])
    band_width = bb_upper - bb_lower
    bb_position = (
        (last_close - bb_lower) / band_width if band_width > 0 else 0.5
    )

    def pct_change(lookback: int) -> float:
        if len(close) <= lookback:
            return 0.0
        prev = close.iloc[-1 - lookback]
        return 100.0 * (last_close - prev) / prev if prev else 0.0

    return Snapshot(
        symbol=symbol,
        last_close=last_close,
        change_pct_1h=pct_change(1),
        change_pct_24h=pct_change(24),
        sma_20=float(sma(close, 20).iloc[-1]),
        sma_50=float(sma(close, 50).iloc[-1]),
        ema_12=float(ema(close, 12).iloc[-1]),
        rsi_14=float(rsi(close).iloc[-1]),
        macd=float(macd_df["macd"].iloc[-1]),
        macd_signal=float(macd_df["signal"].iloc[-1]),
        macd_hist=float(macd_df["hist"].iloc[-1]),
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_position=bb_position,
    )


def technical_signal(snap: Snapshot) -> str:
    """Simple rule-based signal used for backtests and as a sanity check.

    Returns one of: BUY | SELL | HOLD.
    """
    buy_signals = 0
    sell_signals = 0

    if snap.rsi_14 < 30:
        buy_signals += 1
    elif snap.rsi_14 > 70:
        sell_signals += 1

    if snap.macd_hist > 0 and snap.macd > snap.macd_signal:
        buy_signals += 1
    elif snap.macd_hist < 0 and snap.macd < snap.macd_signal:
        sell_signals += 1

    if snap.sma_20 > snap.sma_50:
        buy_signals += 1
    elif snap.sma_20 < snap.sma_50:
        sell_signals += 1

    if snap.bb_position < 0.2:
        buy_signals += 1
    elif snap.bb_position > 0.8:
        sell_signals += 1

    if buy_signals >= 3 and buy_signals > sell_signals:
        return "BUY"
    if sell_signals >= 3 and sell_signals > buy_signals:
        return "SELL"
    return "HOLD"
