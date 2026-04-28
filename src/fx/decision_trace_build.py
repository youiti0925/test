"""Builders for `decision_trace.BarDecisionTrace` instances.

Kept separate from `backtest_engine.py` so the engine's bar-loop reads
straight while the slice/rule-check construction lives next to its own
helpers. Decision logic lives in `decision_engine.py` / `risk_gate.py`;
nothing here changes any decision — these functions only observe and
record what already happened.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .calendar import Event
from .decision_engine import Decision
from .decision_trace import (
    BarDecisionTrace,
    DecisionSlice,
    ExecutionAssumptionSlice,
    ExecutionTraceSlice,
    FundamentalSlice,
    FutureOutcomeSlice,
    HORIZON_BARS_TABLE,
    HigherTimeframeSlice,
    LongTermTrendSlice,
    MacroContextSlice,
    MarketSlice,
    RULE_TAXONOMY,
    RuleCheck,
    TRACE_SCHEMA_VERSION,
    TechnicalSlice,
    WaveformSlice,
    horizon_to_bars,
    make_not_reached_check,
    make_skipped_check,
)
from .indicators import Snapshot
from .macro import MacroSnapshot
from .patterns import PatternResult, analyse as analyse_patterns
from .waveform_backtest import waveform_lookup
from .waveform_library import WaveformSample
from .waveform_matcher import compute_signature
from .risk_gate import (
    RiskState,
    check_calendar_freshness,
    check_consecutive_losses,
    check_daily_loss_cap,
    check_data_quality,
    check_high_impact_event,
    check_rule_unverified,
    check_sentiment_spike,
    check_spread,
    _window_hours_for,
)


def build_run_id(symbol: str) -> str:
    """`bt_{utc_compact}_{symbol}_{8charrand}` — sortable + unique."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(4)
    return f"bt_{stamp}_{symbol}_{rand}"


def _ts_iso(ts) -> str:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


def _events_to_dicts(events: Iterable[Event], ts_now: pd.Timestamp) -> list[dict]:
    out = []
    if ts_now.tzinfo is None:
        now_dt = ts_now.tz_localize(timezone.utc).to_pydatetime()
    else:
        now_dt = ts_now.to_pydatetime()
    for e in events:
        delta_h = (e.when - now_dt).total_seconds() / 3600.0
        out.append({
            "title": e.title,
            "currency": e.currency,
            "impact": e.impact,
            "when": e.when.isoformat(),
            "hours_to_event": round(abs(delta_h), 2),
            "signed_hours_to_event": round(delta_h, 2),
            "window_hours": _window_hours_for(e),
        })
    return out


def market_slice(df: pd.DataFrame, i: int, data_source: str) -> MarketSlice:
    """Inspect the visible window up to bar i and report quality flags."""
    window = df.iloc[: i + 1]
    row = df.iloc[i]
    last_5 = window[["open", "high", "low", "close"]].tail(5)
    has_nan = bool(last_5.isna().any().any())
    missing = int(window[["open", "high", "low", "close", "volume"]].tail(100).isna().sum().sum())
    idx = window.index
    monotonic = bool(idx.is_monotonic_increasing)
    duplicate = int(idx.duplicated().sum())
    tz = str(idx.tzinfo) if idx.tzinfo is not None else "naive"

    # Gap detection: max consecutive timestamp delta vs median.
    gap_detected = False
    if len(window) >= 3:
        deltas = idx.to_series().diff().dropna()
        if len(deltas) > 0:
            med = deltas.median()
            if med.total_seconds() > 0 and (deltas.max() / med) > 2.5:
                gap_detected = True

    # Quality classification — used by tests/19 and rule_check data_quality.
    quality = "OK"
    quality_reason: str | None = None
    if has_nan:
        quality = "HAS_NAN"
        quality_reason = "NaN found in OHLC of last 5 bars"
    elif duplicate > 0:
        quality = "DUPLICATE_INDEX"
        quality_reason = f"{duplicate} duplicate timestamps in window"
    elif tz == "naive":
        quality = "TIMEZONE_NAIVE"
        quality_reason = "DataFrame index is timezone-naive"
    elif gap_detected:
        quality = "GAP_DETECTED"
        quality_reason = "Bar interval has unusually large gaps"

    return MarketSlice(
        open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
        close=float(row["close"]), volume=float(row.get("volume", 0.0) or 0.0),
        data_source=data_source,
        is_complete_bar=True,
        data_quality=quality,
        bars_available=i + 1,
        missing_ohlcv_count=missing,
        has_nan=has_nan,
        index_monotonic=monotonic,
        duplicate_timestamp_count=duplicate,
        timezone=tz,
        gap_detected=gap_detected,
        quality_reason=quality_reason,
    )


def technical_slice(
    snap: Snapshot | None,
    atr_value: float | None,
    technical_only_action: str,
    technical_reason_codes: list[str],
) -> TechnicalSlice:
    if snap is None:
        return TechnicalSlice(
            rsi_14=float("nan"), macd=0.0, macd_signal=0.0, macd_hist=0.0,
            sma_20=0.0, sma_50=0.0, ema_12=0.0,
            bb_upper=0.0, bb_lower=0.0, bb_position=0.0,
            change_pct_1h=0.0, change_pct_24h=0.0,
            atr_14=atr_value,
            technical_only_action=technical_only_action,
            technical_reason_codes=tuple(technical_reason_codes),
            reason_derivation="snapshot_unavailable",
        )
    return TechnicalSlice(
        rsi_14=snap.rsi_14, macd=snap.macd, macd_signal=snap.macd_signal,
        macd_hist=snap.macd_hist, sma_20=snap.sma_20, sma_50=snap.sma_50,
        ema_12=snap.ema_12, bb_upper=snap.bb_upper, bb_lower=snap.bb_lower,
        bb_position=snap.bb_position,
        change_pct_1h=snap.change_pct_1h, change_pct_24h=snap.change_pct_24h,
        atr_14=atr_value,
        technical_only_action=technical_only_action,
        technical_reason_codes=tuple(technical_reason_codes),
        reason_derivation="shared_scoring_helper",
    )


def waveform_slice(
    pattern: PatternResult | None,
    atr_value: float | None,
    close: float,
    *,
    waveform_bias: dict | None = None,
) -> WaveformSlice:
    """Build the waveform slice. `waveform_bias` (PR #15) is the dict
    produced by `waveform_bias_dict()` — None means waveform lookup
    was not requested at all (vs an `{"unavailable_reason": ...}` dict
    which means it was requested but could not run)."""
    if pattern is None:
        return WaveformSlice(
            trend_state="UNKNOWN",
            detected_pattern=None, pattern_confidence=0.0,
            neckline=None, neckline_broken=False,
            distance_to_neckline_atr=None,
            rsi_divergence=False, macd_momentum_weakening=False,
            swing_points_recent=(), waveform_bias=waveform_bias,
            waveform_reason_codes=(),
        )
    distance = None
    if pattern.neckline is not None and atr_value and atr_value > 0:
        distance = (close - pattern.neckline) / atr_value
    swings = list(pattern.swing_highs) + list(pattern.swing_lows)
    swings.sort(key=lambda s: s.index)
    recent = [
        {"ts": s.ts.isoformat() if hasattr(s.ts, "isoformat") else str(s.ts),
         "price": float(s.price), "kind": s.kind}
        for s in swings[-10:]
    ]
    return WaveformSlice(
        trend_state=pattern.trend_state.value,
        detected_pattern=pattern.detected_pattern,
        pattern_confidence=float(pattern.pattern_confidence),
        neckline=(float(pattern.neckline) if pattern.neckline is not None else None),
        neckline_broken=bool(pattern.neckline_broken),
        distance_to_neckline_atr=(float(distance) if distance is not None else None),
        rsi_divergence=bool(pattern.rsi_divergence),
        macd_momentum_weakening=bool(pattern.macd_momentum_weakening),
        swing_points_recent=tuple(recent),
        waveform_bias=waveform_bias,
        waveform_reason_codes=(),
    )


def higher_tf_slice(
    interval: str, trend: str, technical_only_action: str, use_higher_tf: bool
) -> HigherTimeframeSlice:
    rule = {
        "1m": "15min", "5m": "1h", "15m": "4h", "30m": "4h",
        "1h": "1D", "2h": "1D", "4h": "1W", "1d": "1W",
    }.get(interval, "UNKNOWN")
    align = "UNKNOWN"
    if not use_higher_tf:
        align = "DISABLED"
    elif trend == "UNKNOWN":
        align = "UNKNOWN"
    elif technical_only_action == "BUY":
        align = "AGAINST" if trend == "DOWNTREND" else (
            "WITH" if trend == "UPTREND" else "NEUTRAL"
        )
    elif technical_only_action == "SELL":
        align = "AGAINST" if trend == "UPTREND" else (
            "WITH" if trend == "DOWNTREND" else "NEUTRAL"
        )
    else:
        align = "NEUTRAL"
    return HigherTimeframeSlice(source_interval=rule, trend=trend, alignment=align)


def _resample_trend(df_window: pd.DataFrame, rule: str, *, min_bars: int = 10) -> str:
    """Resample df_window to `rule` and run patterns.analyse on the result.

    Returns "UNKNOWN" when there isn't enough resampled history yet —
    same convention as the existing higher_tf logic.
    """
    if len(df_window) < 30:
        return "UNKNOWN"
    try:
        higher = df_window.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    if len(higher) < min_bars:
        return "UNKNOWN"
    return analyse_patterns(higher).trend_state.value


def long_term_trend_slice(df: pd.DataFrame, i: int) -> LongTermTrendSlice:
    """Multi-timeframe long-term trend / SMA / return slice.

    Reads ONLY `df.iloc[: i + 1]`. Trend labels are
    RANGE / UPTREND / DOWNTREND / VOLATILE / UNKNOWN (UNKNOWN until
    enough resampled bars exist). N-day return windows are sliced by
    timestamp so FX weekend gaps fall back to the prior trading day.

    SMA convention: canonical "N-day SMA" — resample to one close per
    calendar day (last bar of the day, weekends drop out), then mean of
    the trailing N daily closes. This matches the chartist reading of
    "200-day SMA" regardless of base interval and is robust to weekend
    gaps without weighting weekday hours.
    """
    window = df.iloc[: i + 1]
    n_bars = len(window)
    closes = window["close"].astype(float)
    ts_now = window.index[-1]
    close_now = float(closes.iloc[-1])

    unavailable: dict[str, str] = {}

    # Resample to one close per calendar day (weekends naturally drop out
    # because there are no FX bars on weekends). `last()` picks the last
    # bar of each day, then `dropna()` removes empty days.
    try:
        daily_closes = closes.resample("1D").last().dropna()
    except Exception:  # noqa: BLE001
        daily_closes = pd.Series(dtype=float)

    def _sma_daily(days: int) -> float | None:
        if len(daily_closes) < days:
            unavailable[f"sma_{days}d"] = (
                f"only {len(daily_closes)} daily closes (need ≥{days})"
            )
            return None
        v = float(daily_closes.iloc[-days:].mean())
        if not np.isfinite(v):
            unavailable[f"sma_{days}d"] = "non-finite mean"
            return None
        return v

    sma30 = _sma_daily(30)
    sma90 = _sma_daily(90)
    sma200 = _sma_daily(200)

    def _close_vs_sma(sma: float | None) -> float | None:
        if sma is None or sma == 0 or not np.isfinite(sma):
            return None
        return 100.0 * (close_now - sma) / sma

    # N-day returns: take the most recent close at-or-before ts_now − N.
    def _ret_back(days: int) -> float | None:
        cutoff = ts_now - pd.Timedelta(days=days)
        sub = closes.loc[closes.index <= cutoff]
        if sub.empty:
            unavailable[f"return_{days}d"] = (
                f"no bars on or before {days}d ago"
            )
            return None
        ref = float(sub.iloc[-1])
        if ref == 0 or not np.isfinite(ref):
            unavailable[f"return_{days}d"] = "ref close zero/non-finite"
            return None
        v = 100.0 * (close_now - ref) / ref
        if not np.isfinite(v):
            return None
        return v

    weekly_ret = _ret_back(7)
    monthly_ret = _ret_back(30)
    quarterly_ret = _ret_back(90)

    daily_trend = _resample_trend(window, "1D")
    weekly_trend = _resample_trend(window, "1W")
    # Pandas accepts "1MS" (month start) for monthly resample.
    monthly_trend = _resample_trend(window, "1MS")

    return LongTermTrendSlice(
        daily_trend=daily_trend,
        weekly_trend=weekly_trend,
        monthly_trend=monthly_trend,
        sma_30d=sma30,
        sma_90d=sma90,
        sma_200d=sma200,
        close_vs_sma_30d_pct=_close_vs_sma(sma30),
        close_vs_sma_90d_pct=_close_vs_sma(sma90),
        close_vs_sma_200d_pct=_close_vs_sma(sma200),
        weekly_return_pct=weekly_ret,
        monthly_return_pct=monthly_ret,
        quarterly_return_pct=quarterly_ret,
        bars_available=n_bars,
        unavailable_reasons=dict(unavailable),
    )


# Macro slots we expose in the trace. Order is the canonical export order.
_MACRO_LEVEL_SLOTS: tuple[str, ...] = (
    "us10y", "us_short_yield_proxy", "dxy", "vix",
    "sp500", "nasdaq", "nikkei",
)


def macro_context_slice(
    macro: MacroSnapshot | None,
    ts: pd.Timestamp,
) -> MacroContextSlice | None:
    """Snapshot the macro / market context at this bar.

    `MacroSnapshot.value_at(slot, ts)` uses Series.asof — strictly
    past-only, so this is point-in-time safe. Deltas are computed by
    calling `value_at(slot, ts - delta)` for the earlier reference.

    Returns None when `macro` is None (caller chose not to fetch macro
    or fetch failed entirely). When `macro` is non-None but a particular
    slot has no data covering this ts, that slot is None and listed in
    `missing_slots`.
    """
    if macro is None:
        return None

    # yfinance's daily series come back tz-naive; the engine's bar ts is
    # tz-aware (UTC). Series.asof raises TypeError on mixed-tz compare,
    # which MacroSnapshot.value_at swallows as None — i.e., every slot
    # silently looks "missing" even when the snapshot is fully populated.
    # Normalize ts once, here, to match the snapshot's index timezone.
    ts_lookup = ts
    for series in macro.series.values():
        idx_tz = getattr(series.index, "tz", None)
        if idx_tz is None and ts.tzinfo is not None:
            ts_lookup = ts.tz_localize(None)
        elif idx_tz is not None and ts.tzinfo is None:
            ts_lookup = ts.tz_localize(idx_tz)
        break

    levels: dict[str, float | None] = {}
    available: list[str] = []
    missing: list[str] = []
    for slot in _MACRO_LEVEL_SLOTS:
        v = macro.value_at(slot, ts_lookup)
        levels[slot] = v
        if v is None:
            missing.append(slot)
        else:
            available.append(slot)

    yield_spread = macro.yield_spread_long_short(ts_lookup)

    def _pct_change(slot: str, delta: pd.Timedelta) -> float | None:
        now = macro.value_at(slot, ts_lookup)
        prev = macro.value_at(slot, ts_lookup - delta)
        if now is None or prev is None or prev == 0 or not np.isfinite(prev):
            return None
        v = 100.0 * (now - prev) / prev
        return float(v) if np.isfinite(v) else None

    def _bp_change(slot: str, delta: pd.Timedelta) -> float | None:
        """Yield-level delta reported in basis points (1 bp = 0.01 pp).

        `MacroSnapshot` stores yield slots in percent (e.g. 4.50). The
        difference between two such values is in percentage points; we
        multiply by 100 to convert to basis points so callers and stats
        all read in bp uniformly.
        """
        now = macro.value_at(slot, ts_lookup)
        prev = macro.value_at(slot, ts_lookup - delta)
        if now is None or prev is None:
            return None
        v = (float(now) - float(prev)) * 100.0
        return v if np.isfinite(v) else None

    def _spread_change_bp(delta: pd.Timedelta) -> float | None:
        """Yield spread is yields(pp) − yields(pp); delta in basis points."""
        now = macro.yield_spread_long_short(ts_lookup)
        prev = macro.yield_spread_long_short(ts_lookup - delta)
        if now is None or prev is None:
            return None
        v = (float(now) - float(prev)) * 100.0
        return v if np.isfinite(v) else None

    one_day = pd.Timedelta(days=1)
    five_days = pd.Timedelta(days=5)

    return MacroContextSlice(
        us10y=levels["us10y"],
        us_short_yield_proxy=levels["us_short_yield_proxy"],
        yield_spread_long_short=yield_spread,
        dxy=levels["dxy"],
        vix=levels["vix"],
        sp500=levels["sp500"],
        nasdaq=levels["nasdaq"],
        nikkei=levels["nikkei"],
        dxy_change_24h_pct=_pct_change("dxy", one_day),
        dxy_change_5d_pct=_pct_change("dxy", five_days),
        us10y_change_24h_bp=_bp_change("us10y", one_day),
        us10y_change_5d_bp=_bp_change("us10y", five_days),
        yield_spread_change_5d_bp=_spread_change_bp(five_days),
        vix_change_24h_pct=_pct_change("vix", one_day),
        sp500_change_24h_pct=_pct_change("sp500", one_day),
        nasdaq_change_24h_pct=_pct_change("nasdaq", one_day),
        available_slots=tuple(available),
        missing_slots=tuple(missing),
        fetch_errors=dict(macro.fetch_errors),
    )


# ---------------------------------------------------------------------------
# Waveform-match context (PR #15)
# ---------------------------------------------------------------------------
# Bars-per-interval used for the horizon-aware look-ahead filter. We import
# the canonical table from decision_trace where it's already maintained for
# future_outcome bookkeeping; do NOT duplicate the table here.

from .decision_trace import _INTERVAL_MINUTES as _INTERVAL_MINUTES_TABLE


def _horizon_duration(
    interval: str, horizon_bars: int
) -> "pd.Timedelta | None":
    """Convert (interval, horizon_bars) to a wall-clock duration.

    Returns None when the interval string is not recognised; the caller
    must then disable waveform lookup (recording an unavailable_reason)
    rather than guess. PR-#15 spec mandates safe-side behaviour: never
    apply a filter weaker than what the horizon requires.
    """
    minutes = _INTERVAL_MINUTES_TABLE.get(interval)
    if minutes is None or horizon_bars <= 0:
        return None
    return pd.Timedelta(minutes=minutes * horizon_bars)


def _filter_library_no_lookahead(
    library: list[WaveformSample],
    bar_ts: pd.Timestamp,
    horizon_dur: pd.Timedelta,
) -> list[WaveformSample]:
    """Keep only samples whose forward-return label is fully realised by bar_ts.

    A sample's `forward_returns_pct[horizon_bars]` is computed from
    `closes[end + horizon]`, so it is only knowable AT that future
    timestamp. To match against the sample's labels at bar_ts without
    leaking future information, we require:

        sample.end_ts + horizon_dur <= bar_ts

    Stricter than `sample.end_ts < bar_ts`. Samples whose horizon
    straddles bar_ts are excluded — those labels were not yet fully
    realised at the moment the engine would have queried.
    """
    cutoff = bar_ts - horizon_dur
    out: list[WaveformSample] = []
    for s in library:
        sample_end = s.end_ts
        # Normalise tz so comparison cannot raise on mixed-tz inputs.
        sample_end_ts = pd.Timestamp(sample_end)
        if sample_end_ts.tzinfo is None and cutoff.tzinfo is not None:
            sample_end_ts = sample_end_ts.tz_localize(cutoff.tzinfo)
        elif sample_end_ts.tzinfo is not None and cutoff.tzinfo is None:
            sample_end_ts = sample_end_ts.tz_convert(None).tz_localize(None)
        if sample_end_ts <= cutoff:
            out.append(s)
    return out


def compute_library_id(library: list[WaveformSample], path: str | None) -> str:
    """Compact identifier for a waveform library: filename + size + ts range + sha8.

    Stored in trace.waveform.waveform_bias.library_id and in run_metadata
    for reproducibility. Kept short on purpose so JSONL traces don't blow
    up — the full library should be referenced by file path elsewhere.
    """
    import hashlib
    import os
    n = len(library)
    if n == 0:
        return f"{os.path.basename(path or '?')}|0|empty"
    first_ts = library[0].end_ts.isoformat()
    last_ts = library[-1].end_ts.isoformat()
    h = hashlib.sha256()
    h.update(f"{n}|{first_ts}|{last_ts}".encode("utf-8"))
    sha8 = h.hexdigest()[:8]
    fname = os.path.basename(path or "memory")
    return f"{fname}|n={n}|{first_ts}|{last_ts}|{sha8}"


def waveform_bias_dict(
    library: list[WaveformSample] | None,
    df: pd.DataFrame,
    i: int,
    *,
    interval: str,
    library_id: str | None = None,
    window_bars: int = 60,
    horizon_bars: int = 24,
    method: str = "dtw",
    min_score: float = 0.55,
    min_samples: int = 20,
    min_directional_share: float = 0.6,
    structure_weight: float = 0.25,
    top_k: int = 30,
) -> dict:
    """Run a horizon-aware waveform_lookup at bar i and return the bias dict.

    The dict is what's stored under `trace.waveform.waveform_bias`. When
    lookup is impossible (no library, unknown interval, no eligible
    samples after horizon-aware filter, or insufficient bars) the dict
    contains only `{"unavailable_reason": "..."}` plus the static
    config keys — never None — so consumers always know why.
    """
    base: dict = {
        "library_id": library_id,
        "horizon_bars": horizon_bars,
        "match_method": method,
        "window_bars": window_bars,
    }

    if library is None:
        return {**base, "unavailable_reason": "no library attached"}

    horizon_dur = _horizon_duration(interval, horizon_bars)
    if horizon_dur is None:
        return {
            **base,
            "unavailable_reason": (
                f"unknown interval {interval!r}; cannot compute "
                "horizon-aware look-ahead filter — disabling lookup"
            ),
        }

    window = df.iloc[: i + 1]
    if len(window) < window_bars:
        return {
            **base,
            "unavailable_reason": (
                f"only {len(window)} bars available, need ≥{window_bars} "
                "for waveform window"
            ),
        }

    bar_ts = window.index[-1]
    if not isinstance(bar_ts, pd.Timestamp):
        bar_ts = pd.Timestamp(bar_ts)

    safe_lib = _filter_library_no_lookahead(library, bar_ts, horizon_dur)
    if not safe_lib:
        return {
            **base,
            "unavailable_reason": (
                f"no library samples whose horizon completes by {bar_ts}; "
                f"library has {len(library)} samples but none satisfy "
                f"end_ts + {horizon_dur} <= bar_ts"
            ),
            "library_size": len(library),
            "eligible_size": 0,
        }

    target_window = window.iloc[-window_bars:]
    try:
        target_sig = compute_signature(target_window, method="z_score")
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "unavailable_reason": f"compute_signature failed: {exc}",
        }

    bias, matches = waveform_lookup(
        target_sig, safe_lib,
        horizon_bars=horizon_bars,
        method=method,
        top_k=top_k,
        min_score=min_score,
        min_sample_count=min_samples,
        min_directional_share=min_directional_share,
        structure_weight=structure_weight,
    )

    raw = bias.to_dict()
    n = bias.sample_count
    bullish_ratio = bias.bullish_count / n if n else None
    bearish_ratio = bias.bearish_count / n if n else None
    neutral_ratio = bias.neutral_count / n if n else None

    median_return = None
    if matches:
        usable_returns = [
            m.sample.forward_returns_pct.get(horizon_bars)
            for m in matches
            if m.sample.forward_returns_pct.get(horizon_bars) is not None
        ]
        if usable_returns:
            usable_returns.sort()
            mid = len(usable_returns) // 2
            if len(usable_returns) % 2:
                median_return = float(usable_returns[mid])
            else:
                median_return = float(
                    (usable_returns[mid - 1] + usable_returns[mid]) / 2
                )

    top_similarity = float(matches[0].score) if matches else None

    out = {
        **base,
        # Raw bias fields (kept verbatim for compat).
        **raw,
        # PR-#15 enriched aliases / analytics.
        "matched_count": n,
        "expected_direction": bias.action,
        "avg_future_return_pct": bias.avg_forward_return_pct,
        "median_future_return_pct": median_return,
        "top_similarity": top_similarity,
        "bullish_match_ratio": bullish_ratio,
        "bearish_match_ratio": bearish_ratio,
        "neutral_match_ratio": neutral_ratio,
        "library_size": len(library),
        "eligible_size": len(safe_lib),
    }
    return out


def fundamental_slice(
    events: tuple[Event, ...],
    ts: pd.Timestamp,
    blocked_codes: tuple[str, ...],
) -> FundamentalSlice:
    """Split ±48h events into nearby / blocking / warning buckets."""
    nearby = _events_to_dicts(events, ts)
    # blocking_events: high-impact events that fell inside their per-event
    # block window AND the gate actually surfaced event_high.
    blocking: list[dict] = []
    warning: list[dict] = []
    for ev_dict in nearby:
        if (ev_dict["impact"] or "").lower() != "high":
            continue
        in_window = ev_dict["hours_to_event"] <= ev_dict["window_hours"]
        if in_window and "event_high" in blocked_codes:
            blocking.append({**ev_dict, "blocked_by_rule": "event_high"})
        elif not in_window:
            warning.append(ev_dict)
        else:
            # In-window but gate didn't block (paranoia case): still mark as
            # blocking-class for diagnostics.
            blocking.append({**ev_dict, "blocked_by_rule": "event_high"})
    return FundamentalSlice(
        nearby_events=tuple(nearby),
        blocking_events=tuple(blocking),
        warning_events=tuple(warning),
        event_evidence_ids=(),
        missing_event_evidence_reason=(
            "events provided as in-memory Event objects; "
            "source_documents.jsonl ingestion not implemented (out of scope for v1)"
        ),
        macro_observations=(),
        macro_evidence_ids=(),
        missing_macro_evidence_reason=(
            "FRED/macro feed not connected in offline backtest"
        ),
        news_evidence_ids=(),
        missing_news_evidence_reason=(
            "news ingestion not connected to backtest_engine input"
        ),
        data_provenance={
            "events_source": "in-memory tuple",
            "freshness_check": "skipped_offline_backtest",
        },
    )


def execution_assumption_slice() -> ExecutionAssumptionSlice:
    return ExecutionAssumptionSlice(
        synthetic_execution=True,
        fill_model="close_price",
        spread_mode="not_modelled",
        slippage_mode="not_modelled",
        bid_ask_mode="not_modelled",
        sentiment_archive="not_available",
    )


def execution_trace_slice(
    *,
    position_before: dict | None,
    position_after: dict | None,
    had_open_position: bool,
    decision_action: str,
    bar_entry_executed: bool,
    bar_entry_price: float | None,
    bar_entry_skipped_reason: str,
    bar_exit_event: bool,
    bar_exit_reason: str | None,
    bar_exit_price: float | None,
    bar_exit_trade_id: str | None,
    bars_held_before: int | None,
) -> ExecutionTraceSlice:
    bars_held_after = (
        position_after["bars_held"] if position_after is not None else None
    )
    # entry_trade_id only set when an entry actually opened a position on
    # this bar — read it from position_after which the engine just wrote.
    if bar_entry_executed and position_after is not None:
        entry_trade_id = position_after.get("trade_id") or None
    else:
        entry_trade_id = None
    # exit_trade_id only set when this bar closed an existing position.
    exit_trade_id = bar_exit_trade_id if bar_exit_event else None
    # Compat field — prefer the new entry id when present, else exit id,
    # else fall back to the open position's id (so still-open bars match
    # the pre-PR behaviour). Tests should migrate to the explicit pair.
    if entry_trade_id is not None:
        trade_id = entry_trade_id
    elif exit_trade_id is not None:
        trade_id = exit_trade_id
    elif position_after is not None:
        trade_id = position_after.get("trade_id") or None
    else:
        trade_id = None
    return ExecutionTraceSlice(
        position_before=position_before,
        position_after=position_after,
        had_open_position=had_open_position,
        entry_signal=decision_action,
        entry_executed=bar_entry_executed,
        entry_price=bar_entry_price,
        entry_skipped_reason=bar_entry_skipped_reason,
        exit_event=bar_exit_event,
        exit_reason=bar_exit_reason,
        exit_price=bar_exit_price,
        entry_trade_id=entry_trade_id,
        exit_trade_id=exit_trade_id,
        trade_id=trade_id,
        bars_held_before=bars_held_before,
        bars_held_after=bars_held_after,
    )


def decision_slice(decision: Decision, technical_only_action: str) -> DecisionSlice:
    return DecisionSlice(
        technical_only_action=technical_only_action,
        final_action=decision.action,
        action_changed_by_engine=(decision.action != technical_only_action),
        blocked_by=tuple(decision.blocked_by),
        rule_chain=tuple(decision.rule_chain),
        confidence=float(decision.confidence),
        reason=decision.reason,
        advisory=dict(decision.advisory),
    )


def _gate_subcheck(
    rule_id: str,
    *,
    block,           # BlockReason | None — output of risk_gate.check_*
    computed: bool,  # was the input present so the check actually ran?
    skipped_reason: str,  # used when computed=False
    threshold: Any,
    value_when_pass: Any = None,
) -> RuleCheck:
    if not computed:
        return make_skipped_check(rule_id, skipped_reason, threshold=threshold,
                                  source_chain_step="risk_gate")
    if block is None:
        return RuleCheck(
            canonical_rule_id=rule_id, rule_group=RULE_TAXONOMY[rule_id],
            result="PASS", computed=True, used_in_decision=True,
            value=value_when_pass, threshold=threshold,
            evidence_ids=(),
            reason=f"{rule_id} passed",
            source_chain_step="risk_gate",
        )
    return RuleCheck(
        canonical_rule_id=rule_id, rule_group=RULE_TAXONOMY[rule_id],
        result="BLOCK", computed=True, used_in_decision=True,
        value=dict(block.detail) if block.detail else block.message,
        threshold=threshold, evidence_ids=(),
        reason=block.message, source_chain_step="risk_gate",
    )


def _chain_check(
    rule_id: str,
    *,
    gate_passed: bool,
    chain_steps: tuple[str, ...],
    computed: bool,
    value: Any,
    threshold: Any,
    pass_reason: str,
) -> RuleCheck:
    """Decision-engine chain step: PASS if reached + value satisfies, else NOT_REACHED.

    `chain_steps` is decision.rule_chain. We treat the chain step as
    'reached' iff its name is present in chain_steps.
    """
    chain_step_name_map = {
        "technical_directionality": "technical_directionality",
        "pattern_check": "pattern_check",
        "higher_tf_alignment": "higher_tf_alignment",
        "risk_reward_floor": "risk_reward_floor",
        "llm_advisory": "llm_advisory",
        "waveform_advisory": "waveform_advisory",
    }
    step = chain_step_name_map.get(rule_id, rule_id)
    if not gate_passed:
        return make_not_reached_check(
            rule_id,
            f"engine stopped at risk_gate before reaching {rule_id}",
            value=value, threshold=threshold, computed=computed,
        )
    reached = step in chain_steps
    if not reached:
        return make_not_reached_check(
            rule_id,
            f"chain stopped before {rule_id}",
            value=value, threshold=threshold, computed=computed,
        )
    return RuleCheck(
        canonical_rule_id=rule_id, rule_group=RULE_TAXONOMY[rule_id],
        result="PASS", computed=computed, used_in_decision=True,
        value=value, threshold=threshold, evidence_ids=(),
        reason=pass_reason, source_chain_step="decision_chain",
    )


def _build_gate_subchecks(
    risk_state: RiskState, decision: Decision
) -> tuple[list[RuleCheck], bool]:
    """Run all 8 gate sub-checks. Returns (checks, gate_passed)."""
    checks: list[RuleCheck] = []
    blocked_codes = tuple(decision.blocked_by)
    gate_codes = {
        "data_quality", "calendar_stale", "event_high",
        "spread_abnormal", "spread_unavailable",
        "daily_loss_cap", "consecutive_losses",
        "rule_unverified", "sentiment_spike",
    }
    gate_passed = not any(c in gate_codes for c in blocked_codes)

    # 1-8: gate sub-checks
    checks.append(_gate_subcheck(
        "data_quality",
        block=check_data_quality(risk_state.df) if risk_state.df is not None else None,
        computed=risk_state.df is not None,
        skipped_reason="no DataFrame provided to risk_state",
        threshold=">=50 bars and finite last close",
        value_when_pass={"bars": (len(risk_state.df) if risk_state.df is not None else 0)},
    ))
    checks.append(_gate_subcheck(
        "calendar_freshness",
        block=check_calendar_freshness(
            risk_state.calendar_freshness, require_fresh=risk_state.require_calendar_fresh
        ),
        computed=risk_state.require_calendar_fresh,
        skipped_reason="require_calendar_fresh=False (offline backtest)",
        threshold="status='fresh'",
    ))
    checks.append(_gate_subcheck(
        "event_high",
        block=check_high_impact_event(risk_state.events, now=risk_state.now),
        computed=True,
        skipped_reason="",
        threshold="<= per-event window_h",
        value_when_pass={"events_in_window": len(risk_state.events)},
    ))
    checks.append(_gate_subcheck(
        "spread_abnormal",
        block=check_spread(risk_state.spread_pct, require_spread=risk_state.require_spread),
        computed=risk_state.spread_pct is not None or risk_state.require_spread,
        skipped_reason="spread_pct is None in synthetic backtest",
        threshold="<=0.05% (or non-None when require_spread)",
    ))
    checks.append(_gate_subcheck(
        "daily_loss_cap",
        block=(check_daily_loss_cap(risk_state.pnl_today, cap=risk_state.daily_loss_cap)
               if risk_state.daily_loss_cap is not None else None),
        computed=risk_state.daily_loss_cap is not None,
        skipped_reason="daily_loss_cap not provided",
        threshold=risk_state.daily_loss_cap,
    ))
    checks.append(_gate_subcheck(
        "consecutive_losses",
        block=check_consecutive_losses(
            risk_state.consecutive_losses, cap=risk_state.consecutive_losses_cap
        ),
        computed=risk_state.consecutive_losses is not None,
        skipped_reason="consecutive_losses input not provided",
        threshold=risk_state.consecutive_losses_cap,
    ))
    checks.append(_gate_subcheck(
        "rule_unverified",
        block=check_rule_unverified(
            risk_state.rule_version_age_hours, min_age_hours=risk_state.rule_min_age_hours
        ),
        computed=risk_state.rule_version_age_hours is not None,
        skipped_reason="rule_version_age_hours not provided",
        threshold=f">={risk_state.rule_min_age_hours}h",
    ))
    checks.append(_gate_subcheck(
        "sentiment_spike",
        block=check_sentiment_spike(risk_state.sentiment_snapshot),
        computed=risk_state.sentiment_snapshot is not None,
        skipped_reason="sentiment_snapshot not provided",
        threshold="mention_count>=200 AND |velocity|>=0.6",
    ))
    return checks, gate_passed


def build_rule_checks_full(
    *,
    risk_state: RiskState,
    decision: Decision,
    pattern: PatternResult | None,
    higher_tf: str,
    risk_reward: float,
    min_risk_reward: float,
    atr_value: float | None,
    technical_only_action: str,
    llm_signal_present: bool,
    waveform_bias_present: bool,
    bar_exit_event: bool,
    bar_exit_reason: str | None,
    had_open_position: bool,
    bar_entry_executed: bool,
    bar_entry_skipped_reason: str,
    pos_low: float | None,
    pos_high: float | None,
    pos_stop: float | None,
    pos_tp: float | None,
    bars_held_after: int | None,
    max_holding_bars: int,
) -> tuple[RuleCheck, ...]:
    """Compose all 19 RuleCheck entries for a normal (non-atr-skip) bar."""
    checks, gate_passed = _build_gate_subchecks(risk_state, decision)

    # 9: atr_available — pre-decision indicator availability
    if atr_value is not None and atr_value > 0:
        checks.append(RuleCheck(
            canonical_rule_id="atr_available", rule_group=RULE_TAXONOMY["atr_available"],
            result="PASS", computed=True, used_in_decision=True,
            value=float(atr_value), threshold=">0",
            evidence_ids=(), reason=f"ATR(14)={atr_value:.6f}",
            source_chain_step="pre_decision",
        ))
    else:
        checks.append(RuleCheck(
            canonical_rule_id="atr_available", rule_group=RULE_TAXONOMY["atr_available"],
            result="BLOCK", computed=True, used_in_decision=True,
            value=atr_value, threshold=">0",
            evidence_ids=(), reason="ATR(14) is NaN or non-positive",
            source_chain_step="pre_decision",
        ))

    # 10: technical_directionality
    # value MUST be the actual technical_only_action computed by
    # indicators.technical_signal — never derive it from advisory or
    # pattern. Pinned by test_technical_directionality_value_matches_action.
    checks.append(_chain_check(
        "technical_directionality",
        gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
        computed=True,
        value=technical_only_action,
        threshold="BUY|SELL",
        pass_reason=f"technical_only_action={technical_only_action}",
    ))

    # 11: pattern_check
    checks.append(_chain_check(
        "pattern_check",
        gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
        computed=pattern is not None,
        value=(pattern.detected_pattern if pattern is not None else None),
        threshold=("neckline_broken=True for top/bottom" if pattern is not None else None),
        pass_reason=("no pattern conflict" if pattern is None else
                     f"pattern={pattern.detected_pattern}, neckline_broken={pattern.neckline_broken}"),
    ))

    # 12: higher_tf_alignment
    checks.append(_chain_check(
        "higher_tf_alignment",
        gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
        computed=higher_tf is not None,
        value=higher_tf, threshold="not against trend",
        pass_reason=f"higher_tf={higher_tf}",
    ))

    # 13: risk_reward_floor
    checks.append(_chain_check(
        "risk_reward_floor",
        gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
        computed=risk_reward is not None,
        value=float(risk_reward), threshold=f">={min_risk_reward}",
        pass_reason=f"RR={risk_reward:.3f}",
    ))

    # 14: llm_advisory
    if not llm_signal_present:
        checks.append(make_skipped_check(
            "llm_advisory", "llm_signal_fn=None",
            threshold=">=0.6 confidence and matching action",
            source_chain_step="decision_chain",
        ))
    else:
        checks.append(_chain_check(
            "llm_advisory",
            gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
            computed=True,
            value=decision.advisory.get("llm_action"),
            threshold=">=0.6 confidence and matching action",
            pass_reason=f"LLM={decision.advisory.get('llm_action')}",
        ))

    # 15: waveform_advisory
    if not waveform_bias_present:
        checks.append(make_skipped_check(
            "waveform_advisory", "waveform_bias=None",
            threshold="non-disagreeing bias",
            source_chain_step="decision_chain",
        ))
    else:
        checks.append(_chain_check(
            "waveform_advisory",
            gate_passed=gate_passed, chain_steps=tuple(decision.rule_chain),
            computed=True,
            value=decision.advisory.get("waveform_bias"),
            threshold="non-disagreeing bias",
            pass_reason="waveform agrees or neutral",
        ))

    # 16: position_state — BLOCK only when the slot is still occupied at
    # the moment decide() returned BUY/SELL. If an exit fired earlier on
    # this bar the slot was cleared before the decision, so the new entry
    # is permitted; recording BLOCK here would contradict
    # entry_execution=PASS and entry_executed=True. Pinned by
    # test_same_bar_exit_and_entry_records_both_trade_ids.
    final_action = decision.action
    slot_blocked = had_open_position and not bar_exit_event
    if final_action in ("BUY", "SELL") and slot_blocked:
        checks.append(RuleCheck(
            canonical_rule_id="position_state", rule_group=RULE_TAXONOMY["position_state"],
            result="BLOCK", computed=True, used_in_decision=True,
            value="already_in_position", threshold="one_position_at_a_time",
            evidence_ids=(),
            reason=("Final action was directional but an existing position is open; "
                    "no new entry"),
            source_chain_step="post_decision",
        ))
    elif final_action in ("BUY", "SELL") and had_open_position and bar_exit_event:
        checks.append(RuleCheck(
            canonical_rule_id="position_state", rule_group=RULE_TAXONOMY["position_state"],
            result="PASS", computed=True, used_in_decision=True,
            value="exit_then_entry", threshold="one_position_at_a_time",
            evidence_ids=(),
            reason=("Existing position closed earlier in this bar; slot is "
                    "free for new entry"),
            source_chain_step="post_decision",
        ))
    elif final_action in ("BUY", "SELL"):
        checks.append(RuleCheck(
            canonical_rule_id="position_state", rule_group=RULE_TAXONOMY["position_state"],
            result="PASS", computed=True, used_in_decision=True,
            value="no_open_position", threshold="one_position_at_a_time",
            evidence_ids=(),
            reason="No existing position; new entry permitted",
            source_chain_step="post_decision",
        ))
    else:
        checks.append(RuleCheck(
            canonical_rule_id="position_state", rule_group=RULE_TAXONOMY["position_state"],
            result="INFO", computed=True, used_in_decision=True,
            value=("has_open_position" if had_open_position else "no_open_position"),
            threshold="one_position_at_a_time",
            evidence_ids=(),
            reason="final_action=HOLD; position_state is informational only",
            source_chain_step="post_decision",
        ))

    # 17: entry_execution
    if bar_entry_executed:
        checks.append(RuleCheck(
            canonical_rule_id="entry_execution", rule_group=RULE_TAXONOMY["entry_execution"],
            result="PASS", computed=True, used_in_decision=True,
            value={"entry_executed": True, "reason": bar_entry_skipped_reason},
            threshold="decision in {BUY,SELL} AND no_open_position AND atr_available",
            evidence_ids=(), reason="entry executed",
            source_chain_step="post_decision",
        ))
    elif bar_entry_skipped_reason == "position_already_open":
        checks.append(RuleCheck(
            canonical_rule_id="entry_execution", rule_group=RULE_TAXONOMY["entry_execution"],
            result="BLOCK", computed=True, used_in_decision=True,
            value={"entry_executed": False, "reason": "position_already_open"},
            threshold="decision in {BUY,SELL} AND no_open_position AND atr_available",
            evidence_ids=(),
            reason="Decision was BUY/SELL but existing position blocked entry",
            source_chain_step="post_decision",
        ))
    else:
        checks.append(RuleCheck(
            canonical_rule_id="entry_execution", rule_group=RULE_TAXONOMY["entry_execution"],
            result="SKIPPED", computed=True, used_in_decision=False,
            value={"entry_executed": False, "reason": bar_entry_skipped_reason},
            threshold="decision in {BUY,SELL} AND no_open_position AND atr_available",
            evidence_ids=(),
            reason=f"entry skipped: {bar_entry_skipped_reason}",
            source_chain_step="post_decision",
        ))

    # 18: exit_check
    if not had_open_position:
        checks.append(make_skipped_check(
            "exit_check", "no_position_to_check",
            threshold="low<=stop OR high>=tp OR bars_held>=max_holding",
            source_chain_step="pre_decision",
        ))
    elif bar_exit_event:
        checks.append(RuleCheck(
            canonical_rule_id="exit_check", rule_group=RULE_TAXONOMY["exit_check"],
            result="PASS", computed=True, used_in_decision=True,
            value={"exited": True, "exit_reason": bar_exit_reason,
                   "low": pos_low, "high": pos_high,
                   "stop": pos_stop, "tp": pos_tp,
                   "bars_held": bars_held_after, "max_holding": max_holding_bars},
            threshold="low<=stop OR high>=tp OR bars_held>=max_holding",
            evidence_ids=(), reason=f"position exited via {bar_exit_reason}",
            source_chain_step="pre_decision",
        ))
    else:
        checks.append(RuleCheck(
            canonical_rule_id="exit_check", rule_group=RULE_TAXONOMY["exit_check"],
            result="PASS", computed=True, used_in_decision=True,
            value={"exited": False, "low": pos_low, "high": pos_high,
                   "stop": pos_stop, "tp": pos_tp,
                   "bars_held": bars_held_after, "max_holding": max_holding_bars},
            threshold="low<=stop OR high>=tp OR bars_held>=max_holding",
            evidence_ids=(), reason="position remains open",
            source_chain_step="pre_decision",
        ))

    # 19: final_decision (terminal — always PASS or BLOCK)
    if decision.action == "HOLD":
        checks.append(RuleCheck(
            canonical_rule_id="final_decision", rule_group=RULE_TAXONOMY["final_decision"],
            result="BLOCK", computed=True, used_in_decision=True,
            value=decision.action, threshold=None, evidence_ids=(),
            reason=decision.reason, source_chain_step="terminal",
        ))
    else:
        checks.append(RuleCheck(
            canonical_rule_id="final_decision", rule_group=RULE_TAXONOMY["final_decision"],
            result="PASS", computed=True, used_in_decision=True,
            value=decision.action, threshold=None, evidence_ids=(),
            reason=decision.reason, source_chain_step="terminal",
        ))

    return tuple(checks)


def build_atr_unavailable_trace(
    *,
    run_id: str,
    df: pd.DataFrame,
    i: int,
    ts: pd.Timestamp,
    symbol: str,
    interval: str,
    data_source: str,
    events_tuple: tuple[Event, ...],
    pos_after,           # EnginePosition | None
    position_before: dict | None,
    bars_held_before: int | None,
    had_open_position: bool,
    bar_exit_event: bool,
    bar_exit_reason: str | None,
    bar_exit_price: float | None,
    bar_exit_trade_id: str | None,
    macro: MacroSnapshot | None = None,
    waveform_bias: dict | None = None,
) -> BarDecisionTrace:
    """Trace for a bar where ATR is NaN — engine bailed before decide()."""
    from .backtest_engine import _position_dict  # local import to avoid cycle

    market = market_slice(df, i, data_source)
    technical = technical_slice(None, None, "HOLD", [])
    waveform = waveform_slice(
        None, None, market.close, waveform_bias=waveform_bias,
    )
    higher_tf = higher_tf_slice(interval, "UNKNOWN", "HOLD", use_higher_tf=False)
    fundamental = fundamental_slice(events_tuple, ts, blocked_codes=())
    exec_assumption = execution_assumption_slice()
    exec_trace = execution_trace_slice(
        position_before=position_before,
        position_after=_position_dict(pos_after),
        had_open_position=had_open_position,
        decision_action="HOLD",
        bar_entry_executed=False,
        bar_entry_price=None,
        bar_entry_skipped_reason="atr_unavailable",
        bar_exit_event=bar_exit_event,
        bar_exit_reason=bar_exit_reason,
        bar_exit_price=bar_exit_price,
        bar_exit_trade_id=bar_exit_trade_id,
        bars_held_before=bars_held_before,
    )
    # Compose 19 rule_checks: gates SKIPPED, atr_available BLOCK, chain
    # NOT_REACHED, position_state INFO, entry_execution SKIPPED.
    checks: list[RuleCheck] = []
    for rid in (
        "data_quality", "calendar_freshness", "event_high", "spread_abnormal",
        "daily_loss_cap", "consecutive_losses", "rule_unverified", "sentiment_spike",
    ):
        checks.append(make_skipped_check(
            rid, "engine bailed before risk_gate (atr_unavailable)",
            source_chain_step="risk_gate",
        ))
    checks.append(RuleCheck(
        canonical_rule_id="atr_available", rule_group=RULE_TAXONOMY["atr_available"],
        result="BLOCK", computed=True, used_in_decision=True,
        value=None, threshold=">0", evidence_ids=(),
        reason="ATR(14) is NaN or non-positive; bar skipped",
        source_chain_step="pre_decision",
    ))
    for rid in (
        "technical_directionality", "pattern_check", "higher_tf_alignment",
        "risk_reward_floor", "llm_advisory", "waveform_advisory",
    ):
        checks.append(make_not_reached_check(
            rid, "engine bailed before decide() (atr_unavailable)",
            computed=False,
        ))
    checks.append(RuleCheck(
        canonical_rule_id="position_state", rule_group=RULE_TAXONOMY["position_state"],
        result="INFO", computed=True, used_in_decision=False,
        value=("has_open_position" if had_open_position else "no_open_position"),
        threshold="one_position_at_a_time", evidence_ids=(),
        reason="bar skipped (atr_unavailable); no entry attempted",
        source_chain_step="post_decision",
    ))
    checks.append(make_skipped_check(
        "entry_execution", "atr_unavailable",
        threshold="decision in {BUY,SELL} AND no_open_position AND atr_available",
        source_chain_step="post_decision",
    ))
    checks.append(RuleCheck(
        canonical_rule_id="exit_check", rule_group=RULE_TAXONOMY["exit_check"],
        result=("PASS" if had_open_position else "SKIPPED"),
        computed=True, used_in_decision=had_open_position,
        value={"exited": bar_exit_event, "exit_reason": bar_exit_reason},
        threshold="low<=stop OR high>=tp OR bars_held>=max_holding",
        evidence_ids=(), reason="exit logic ran with available OHLC",
        source_chain_step="pre_decision",
    ))
    checks.append(RuleCheck(
        canonical_rule_id="final_decision", rule_group=RULE_TAXONOMY["final_decision"],
        result="BLOCK", computed=True, used_in_decision=True,
        value="HOLD", threshold=None, evidence_ids=(),
        reason="bar skipped because ATR not yet available",
        source_chain_step="terminal",
    ))

    decision = DecisionSlice(
        technical_only_action="HOLD",
        final_action="HOLD",
        action_changed_by_engine=False,
        blocked_by=("atr_unavailable",),
        rule_chain=(),
        confidence=0.0,
        reason="ATR(14) unavailable; engine could not run decide()",
        advisory={},
    )

    return BarDecisionTrace(
        run_id=run_id,
        trace_schema_version=TRACE_SCHEMA_VERSION,
        bar_id=f"{symbol}_{interval}_{ts.isoformat()}",
        timestamp=ts.isoformat(),
        symbol=symbol, timeframe=interval, bar_index=i,
        market=market, technical=technical, waveform=waveform,
        higher_timeframe=higher_tf, fundamental=fundamental,
        execution_assumption=exec_assumption, execution_trace=exec_trace,
        rule_checks=tuple(checks),
        decision=decision,
        future_outcome=None,
        long_term_trend=long_term_trend_slice(df, i),
        macro_context=macro_context_slice(macro, ts),
    )


def build_full_trace(
    *,
    run_id: str,
    df: pd.DataFrame,
    i: int,
    ts: pd.Timestamp,
    symbol: str,
    interval: str,
    data_source: str,
    events_tuple: tuple[Event, ...],
    snap: Snapshot,
    atr_value: float,
    tech: str,
    tech_reason_codes: list[str],
    pattern: PatternResult,
    higher_tf: str,
    use_higher_tf: bool,
    risk_state: RiskState,
    llm_signal,
    decision: Decision,
    risk_reward: float,
    stop_atr_mult: float,
    tp_atr_mult: float,
    max_holding_bars: int,
    position_before: dict | None,
    position_after: dict | None,
    bars_held_before: int | None,
    had_open_position: bool,
    bar_entry_executed: bool,
    bar_entry_price: float | None,
    bar_entry_skipped_reason: str,
    bar_exit_event: bool,
    bar_exit_reason: str | None,
    bar_exit_price: float | None,
    bar_exit_trade_id: str | None,
    macro: MacroSnapshot | None = None,
    waveform_bias: dict | None = None,
) -> BarDecisionTrace:
    """Full trace for a normally processed bar."""
    market = market_slice(df, i, data_source)
    technical = technical_slice(snap, atr_value, tech, tech_reason_codes)
    waveform = waveform_slice(
        pattern, atr_value, market.close, waveform_bias=waveform_bias,
    )
    higher_tf_s = higher_tf_slice(interval, higher_tf, tech, use_higher_tf)
    long_term = long_term_trend_slice(df, i)
    macro_ctx = macro_context_slice(macro, ts)
    fundamental = fundamental_slice(
        risk_state.events, ts, blocked_codes=tuple(decision.blocked_by)
    )
    exec_assumption = execution_assumption_slice()
    exec_trace = execution_trace_slice(
        position_before=position_before,
        position_after=position_after,
        had_open_position=had_open_position,
        decision_action=decision.action,
        bar_entry_executed=bar_entry_executed,
        bar_entry_price=bar_entry_price,
        bar_entry_skipped_reason=bar_entry_skipped_reason,
        bar_exit_event=bar_exit_event,
        bar_exit_reason=bar_exit_reason,
        bar_exit_price=bar_exit_price,
        bar_exit_trade_id=bar_exit_trade_id,
        bars_held_before=bars_held_before,
    )
    pos_low = float(df["low"].iloc[i]) if had_open_position else None
    pos_high = float(df["high"].iloc[i]) if had_open_position else None
    pos_stop = (position_before or {}).get("stop") if had_open_position else None
    pos_tp = (position_before or {}).get("take_profit") if had_open_position else None
    bars_held_after = (
        position_after.get("bars_held") if position_after is not None
        else (bars_held_before + 1 if bars_held_before is not None else None)
    )

    rule_checks = build_rule_checks_full(
        risk_state=risk_state, decision=decision, pattern=pattern,
        higher_tf=higher_tf, risk_reward=risk_reward,
        min_risk_reward=1.5, atr_value=atr_value,
        technical_only_action=tech,
        llm_signal_present=llm_signal is not None,
        waveform_bias_present=False,
        bar_exit_event=bar_exit_event, bar_exit_reason=bar_exit_reason,
        had_open_position=had_open_position,
        bar_entry_executed=bar_entry_executed,
        bar_entry_skipped_reason=bar_entry_skipped_reason,
        pos_low=pos_low, pos_high=pos_high,
        pos_stop=pos_stop, pos_tp=pos_tp,
        bars_held_after=bars_held_after,
        max_holding_bars=max_holding_bars,
    )

    decision_s = decision_slice(decision, technical_only_action=tech)

    return BarDecisionTrace(
        run_id=run_id,
        trace_schema_version=TRACE_SCHEMA_VERSION,
        bar_id=f"{symbol}_{interval}_{ts.isoformat()}",
        timestamp=ts.isoformat(),
        symbol=symbol, timeframe=interval, bar_index=i,
        market=market, technical=technical, waveform=waveform,
        higher_timeframe=higher_tf_s, fundamental=fundamental,
        execution_assumption=exec_assumption, execution_trace=exec_trace,
        rule_checks=rule_checks,
        decision=decision_s,
        future_outcome=None,
        long_term_trend=long_term,
        macro_context=macro_ctx,
    )


def _simulate_hypothetical_trade(
    df: pd.DataFrame, i: int, action: str,
    atr_value: float, stop_atr_mult: float, tp_atr_mult: float,
    max_holding_bars: int,
) -> tuple[str, float, int, float] | tuple[None, None, None, None]:
    """Walk forward from bar i with technical_only_action; same stop/tp/max as engine.

    Returns (exit_reason, exit_price, bars_held, return_pct). All None for
    HOLD or invalid action. Verification-only — never read by decide().
    """
    if action not in ("BUY", "SELL"):
        return (None, None, None, None)
    if atr_value is None or atr_value <= 0:
        return (None, None, None, None)
    entry_price = float(df["close"].iloc[i])
    if entry_price <= 0:
        return (None, None, None, None)
    offset = stop_atr_mult * atr_value
    tp_offset = tp_atr_mult * atr_value
    if action == "BUY":
        stop = entry_price - offset
        tp = entry_price + tp_offset
    else:
        stop = entry_price + offset
        tp = entry_price - tp_offset
    end = min(i + 1 + max_holding_bars, len(df))
    direction = 1 if action == "BUY" else -1
    for j in range(i + 1, end):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
        if action == "BUY":
            if low <= stop:
                exit_price = stop
                ret = 100 * (exit_price - entry_price) / entry_price * direction
                return ("stop", exit_price, j - i, ret)
            if high >= tp:
                exit_price = tp
                ret = 100 * (exit_price - entry_price) / entry_price * direction
                return ("take_profit", exit_price, j - i, ret)
        else:
            if high >= stop:
                exit_price = stop
                ret = 100 * (exit_price - entry_price) / entry_price * direction
                return ("stop", exit_price, j - i, ret)
            if low <= tp:
                exit_price = tp
                ret = 100 * (exit_price - entry_price) / entry_price * direction
                return ("take_profit", exit_price, j - i, ret)
    # Force-close at the last walked bar.
    if end - 1 >= i + 1:
        last_close = float(df["close"].iloc[end - 1])
        ret = 100 * (last_close - entry_price) / entry_price * direction
        reason = "max_holding" if (end - 1 - i) >= max_holding_bars else "end_of_data"
        return (reason, last_close, end - 1 - i, ret)
    return (None, None, None, None)


def populate_future_outcomes(
    *,
    traces: list[BarDecisionTrace],
    df: pd.DataFrame,
    interval: str,
    stop_atr_mult: float,
    tp_atr_mult: float,
    max_holding_bars: int,
    atr_series: pd.Series,
) -> None:
    """Second-pass: fill FutureOutcomeSlice on every trace.

    Decision pipeline never reaches this function; it runs after the bar
    loop completes. Pinned by test_future_outcome_does_not_affect_decisions.
    """
    horizons = HORIZON_BARS_TABLE.get(interval, {"1h": None, "4h": None, "24h": None})
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)

    for trace in traces:
        i = trace.bar_index
        close_i = closes[i]
        # Per-horizon returns
        future_buy: dict[str, float | None] = {"1h": None, "4h": None, "24h": None}
        future_sell: dict[str, float | None] = {"1h": None, "4h": None, "24h": None}
        available: list[str] = []
        unavailable: dict[str, str] = {}
        for hname, hbars in horizons.items():
            if hbars is None:
                unavailable[hname] = "interval coarser than horizon"
                continue
            if i + hbars >= n:
                unavailable[hname] = "horizon exceeds remaining bars"
                continue
            target_close = closes[i + hbars]
            ret_buy = 100.0 * (target_close - close_i) / close_i if close_i else None
            ret_sell = -ret_buy if ret_buy is not None else None
            future_buy[hname] = ret_buy
            future_sell[hname] = ret_sell
            available.append(hname)

        # MFE/MAE over 24h horizon (or shortest available)
        mfe_buy = mae_buy = mfe_sell = mae_sell = None
        h24 = horizons.get("24h")
        if h24 is not None and i + h24 < n:
            window_high = highs[i + 1: i + h24 + 1]
            window_low = lows[i + 1: i + h24 + 1]
            if len(window_high):
                mfe_buy = 100.0 * (float(window_high.max()) - close_i) / close_i
                mae_buy = 100.0 * (float(window_low.min()) - close_i) / close_i
                mfe_sell = -mae_buy
                mae_sell = -mfe_buy

        # Hypothetical trade in technical_only_action direction
        atr_at_i = float(atr_series.iloc[i]) if i < len(atr_series) else float("nan")
        hyp_reason, hyp_price, hyp_bars, hyp_ret = _simulate_hypothetical_trade(
            df, i, trace.decision.technical_only_action,
            atr_at_i if not np.isnan(atr_at_i) else 0.0,
            stop_atr_mult, tp_atr_mult, max_holding_bars,
        )

        # outcome / gate_effect classification
        if hyp_ret is None:
            outcome = "N/A"
        elif hyp_ret > 0:
            outcome = "WIN"
        elif hyp_ret < 0:
            outcome = "LOSS"
        else:
            outcome = "FLAT"

        tech_action = trace.decision.technical_only_action
        final_action = trace.decision.final_action
        if tech_action == final_action:
            gate_effect = "NO_CHANGE"
        elif tech_action in ("BUY", "SELL") and final_action == "HOLD":
            if outcome == "LOSS":
                gate_effect = "PROTECTED"
                outcome = "LOSS_AVOIDED"
            elif outcome == "WIN":
                gate_effect = "COST_OPPORTUNITY"
                outcome = "WIN_MISSED"
            else:
                gate_effect = "NO_CHANGE"
        else:
            gate_effect = "NO_CHANGE"

        trace.future_outcome = FutureOutcomeSlice(
            horizons_bars={k: int(v) if v is not None else None for k, v in horizons.items()},
            future_return_1h_if_buy_pct=future_buy["1h"],
            future_return_4h_if_buy_pct=future_buy["4h"],
            future_return_24h_if_buy_pct=future_buy["24h"],
            future_return_1h_if_sell_pct=future_sell["1h"],
            future_return_4h_if_sell_pct=future_sell["4h"],
            future_return_24h_if_sell_pct=future_sell["24h"],
            mfe_24h_if_buy_pct=mfe_buy,
            mae_24h_if_buy_pct=mae_buy,
            mfe_24h_if_sell_pct=mfe_sell,
            mae_24h_if_sell_pct=mae_sell,
            available_horizons=tuple(available),
            unavailable_horizons=unavailable,
            outcome_if_technical_action_taken=outcome,
            gate_effect=gate_effect,
            hypothetical_technical_trade_exit_reason=hyp_reason,
            hypothetical_technical_trade_exit_price=hyp_price,
            hypothetical_technical_trade_bars_held=hyp_bars,
            hypothetical_technical_trade_return_pct=hyp_ret,
        )


# ---------------------------------------------------------------------------
# Live-trade trace builders (PR #12)
# ---------------------------------------------------------------------------
#
# `cmd_trade` produces one BarDecisionTrace per invocation (one bar = one
# trade attempt), regardless of whether the action was BUY / SELL / HOLD.
# These helpers re-use the existing slice builders so the schema stays
# identical to backtest traces — only the metadata + execution_trace
# slice carry the live-specific knobs.
#
# Constraints
# - decision_trace.py schema is NOT changed.
# - backtest_engine.py entry/exit logic is NOT pulled in.
# - future_outcome stays None — live can't compute future returns
#   on the same bar; that's an offline-only enrichment.


# Synthetic bar_index used for live traces. Live runs do not have a
# numeric bar_index from a sweep loop; this constant makes downstream
# tools (trace-stats etc.) see a stable integer.
_LIVE_BAR_INDEX = 0


def build_live_run_metadata(
    *,
    run_id: str,
    symbol: str,
    interval: str,
    broker_label: str,
    dry_run: bool,
    config_payload: dict[str, Any] | None = None,
) -> "object":
    """Construct a RunMetadata for one live cmd_trade invocation.

    `broker_label` is one of "paper" / "oanda" / etc. — passed through to
    `RunMetadata.input_data_source` so trace-stats users can tell live
    runs apart from backtest runs.

    `execution_mode` reflects dry-run-vs-live-trade so the analyst can
    filter out dry-run traces when computing fill-aware metrics. Allowed
    values: "live_dry_run", "live_paper", "live_oanda".
    """
    from .decision_trace import (
        RunMetadata,
        TRACE_SCHEMA_VERSION,
        compute_strategy_config_hash,
        get_commit_sha,
    )

    if dry_run:
        execution_mode = "live_dry_run"
    else:
        execution_mode = f"live_{broker_label}"

    sha, sha_status = get_commit_sha()
    payload = config_payload or {}
    return RunMetadata(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        timeframe=interval,
        bar_range={"start": None, "end": None, "n_bars": 1, "warmup": 0},
        trace_schema_version=TRACE_SCHEMA_VERSION,
        strategy_config_hash=compute_strategy_config_hash(payload),
        # Live runs have no static dataframe to hash; record the broker
        # label so the trace makes its provenance obvious downstream.
        data_snapshot_hash=f"live:{broker_label}:{symbol}",
        input_data_source=broker_label,
        input_data_retrieved_at=datetime.now(timezone.utc).isoformat(),
        # Live still uses synthetic execution semantics for the analyst
        # path (PaperBroker / dry-run). For real OANDA fills the engine
        # already populates fill audit fields on the trades table; the
        # decision_trace itself just records the same intent.
        synthetic_execution=True,
        commit_sha=sha,
        commit_sha_status=sha_status,
        execution_mode=execution_mode,
        engine_version="cli.cmd_trade",
        timezone_name="UTC",
    )


def build_live_decision_trace(
    *,
    run_id: str,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    symbol: str,
    interval: str,
    snap: Snapshot,
    atr_value: float | None,
    technical_action: str,
    pattern: PatternResult,
    higher_tf: str,
    use_higher_tf: bool,
    risk_state: RiskState,
    llm_signal,
    decision: Decision,
    risk_reward: float,
    broker_label: str,
    dry_run: bool,
    order_placed: bool,
    fill: "object | None" = None,
) -> BarDecisionTrace:
    """Build one BarDecisionTrace for a live cmd_trade invocation.

    Reuses the same slice builders as backtest. The differences are:
    - `bar_index` is fixed at 0 (live = single-bar invocation).
    - `execution_trace.entry_signal` reflects the engine decision; the
      `entry_executed` / `exit_event` / `entry_price` fields encode
      whether an order actually went out (False for dry-run / HOLD).
    - `future_outcome` stays `None` — live cannot peek at future bars.
    - `decision.advisory` carries the broker label and dry-run flag so
      downstream filters can split live from paper from dry-run without
      changing the schema.
    """
    # Slice 1: market — last bar of df
    i_last = len(df) - 1
    market = market_slice(df, i_last, data_source=broker_label)

    # Slice 2: technical — derive reason codes from the same shared helper
    from .indicators import technical_signal_reasons
    _, tech_reason_codes = technical_signal_reasons(snap)
    technical = technical_slice(snap, atr_value, technical_action, tech_reason_codes)

    # Slice 3: waveform
    wave = waveform_slice(pattern, atr_value, market.close)

    # Slice 4: higher_tf
    higher_tf_s = higher_tf_slice(interval, higher_tf, technical_action, use_higher_tf)

    # Slice 5: fundamental — use whichever events the gate already saw
    fundamental = fundamental_slice(
        risk_state.events, ts, blocked_codes=tuple(decision.blocked_by)
    )

    # Slice 6: execution_assumption — re-use the schema-pinned helper. The
    # broker / dry-run distinction is carried via run_metadata.execution_mode
    # and decision.advisory; we don't bend execution_assumption fields.
    exec_assumption = execution_assumption_slice()

    # Slice 7: execution_trace — live-specific bookkeeping. The five
    # values in `decision_trace.ENTRY_SKIPPED_REASONS` are a CLOSED
    # taxonomy from the backtest world, so live cases must map to one of
    # them. We pick the closest taxonomy value and keep the live-specific
    # disposition (dry_run / order_not_placed / entry_executed / hold)
    # in `decision.advisory.live_execution_status` so callers can split
    # those cases without adding to the schema. Test pin:
    # tests/test_decision_trace.py::test_entry_skipped_reason_in_fixed_candidates
    # would fail if a live trace ever wrote a non-taxonomy value here.
    entry_executed = bool(order_placed)
    if decision.action == "HOLD":
        # Engine decided HOLD — straightforward map to the existing value.
        skipped_reason = "decision_was_HOLD"
        live_execution_status = "hold"
        entry_price = None
        trade_id = None
    else:
        # Engine decided BUY/SELL — the engine path was clean; whether
        # the broker actually filled is a wrapper-layer concern. We use
        # the "entry_executed" sentinel ("no engine-level skip occurred")
        # for all of: real fill, dry-run, broker-no-fill. The discriminator
        # lives in advisory below.
        skipped_reason = "entry_executed"
        if dry_run:
            live_execution_status = "dry_run"
            entry_price = None
            trade_id = None
        elif not order_placed:
            live_execution_status = "order_not_placed"
            entry_price = None
            trade_id = None
        else:
            live_execution_status = "entry_executed"
            entry_price = float(getattr(fill, "actual_fill_price", None) or 0.0) or None
            trade_id = getattr(fill, "broker_order_id", None) or None

    exec_trace = ExecutionTraceSlice(
        position_before=None,
        position_after=None,
        had_open_position=False,
        entry_signal=decision.action,
        entry_executed=entry_executed,
        entry_price=entry_price,
        entry_skipped_reason=skipped_reason,
        exit_event=False,
        exit_reason=None,
        exit_price=None,
        entry_trade_id=trade_id if entry_executed else None,
        exit_trade_id=None,
        trade_id=trade_id,
        bars_held_before=None,
        bars_held_after=None,
    )

    # Decoration on decision.advisory: stamp broker label + live execution
    # status WITHOUT mutating the original decision (frozen dataclass).
    # `live_execution_status` carries the live-specific disposition that
    # cannot be expressed via the closed `entry_skipped_reason` taxonomy.
    enriched_advisory = dict(decision.advisory)
    enriched_advisory.setdefault("live_broker", broker_label)
    enriched_advisory.setdefault("live_dry_run", dry_run)
    enriched_advisory.setdefault("live_order_placed", entry_executed)
    enriched_advisory.setdefault("live_execution_status", live_execution_status)

    decision_s = DecisionSlice(
        technical_only_action=technical_action,
        final_action=decision.action,
        action_changed_by_engine=(decision.action != technical_action),
        blocked_by=tuple(decision.blocked_by),
        rule_chain=tuple(decision.rule_chain),
        confidence=float(decision.confidence),
        reason=decision.reason,
        advisory=enriched_advisory,
    )

    # Rule checks — re-use the full builder. Live and backtest share the
    # same rule chain so this should yield identical PASS / NOT_REACHED
    # patterns for the same inputs.
    pos_low = pos_high = pos_stop = pos_tp = None
    rule_checks = build_rule_checks_full(
        risk_state=risk_state,
        decision=decision,
        pattern=pattern,
        higher_tf=higher_tf,
        risk_reward=risk_reward,
        min_risk_reward=1.5,
        atr_value=atr_value,
        technical_only_action=technical_action,
        llm_signal_present=llm_signal is not None,
        waveform_bias_present=False,
        bar_exit_event=False,
        bar_exit_reason=None,
        had_open_position=False,
        bar_entry_executed=entry_executed,
        bar_entry_skipped_reason=skipped_reason,
        pos_low=pos_low, pos_high=pos_high,
        pos_stop=pos_stop, pos_tp=pos_tp,
        bars_held_after=None,
        max_holding_bars=0,
    )

    return BarDecisionTrace(
        run_id=run_id,
        trace_schema_version=TRACE_SCHEMA_VERSION,
        bar_id=f"{symbol}_{interval}_{ts.isoformat()}",
        timestamp=ts.isoformat(),
        symbol=symbol,
        timeframe=interval,
        bar_index=_LIVE_BAR_INDEX,
        market=market,
        technical=technical,
        waveform=wave,
        higher_timeframe=higher_tf_s,
        fundamental=fundamental,
        execution_assumption=exec_assumption,
        execution_trace=exec_trace,
        rule_checks=rule_checks,
        decision=decision_s,
        future_outcome=None,
    )
