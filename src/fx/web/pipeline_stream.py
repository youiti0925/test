"""Pipeline streamer: runs the analysis and yields one event per step.

Used by the web dashboard's `/analyze/stream` SSE endpoint so the user
can watch the bot think in real time. Each yielded dict has:
  {"step": str, "status": "ok"|"skip"|"error", "elapsed_s": float, "data": ...}

The final event has step="done" and the full context.

This is a generator — we don't import it from the CLI path, which stays
on the batch `_gather_inputs` helper. Keeping the two paths separate
means the CLI and web views can evolve independently.
"""
from __future__ import annotations

import time
from typing import Any, Generator

from ..analyst import Analyst
from ..calendar import calendar_freshness, load_events, upcoming_for_symbol
from ..config import Config
from ..context import (
    MarketContext,
    detect_volume_spike,
    event_risk_level_from,
    keywords_from_sentiment,
)
from ..correlation import build_correlation_snapshot, related_for
from ..data import fetch_ohlcv
from ..decision_engine import decide as decide_action
from ..higher_timeframe import fetch_and_classify as fetch_higher_tf
from ..indicators import build_snapshot, technical_signal
from ..news import fetch_headlines
from ..patterns import analyse as analyse_patterns
from ..risk import atr, plan_trade
from ..risk_gate import RiskState
from ..sentiment import read_for as read_sentiment
from ..storage import Storage


def _event(step: str, status: str, elapsed_s: float, data: Any = None) -> dict:
    return {"step": step, "status": status, "elapsed_s": round(elapsed_s, 3), "data": data}


def run_pipeline(
    symbol: str,
    interval: str,
    period: str,
    cfg: Config,
    storage: Storage,
    *,
    events_path=None,
    want_news: bool = True,
    want_correlation: bool = True,
    want_events: bool = True,
    want_lessons: bool = True,
    want_sentiment: bool = True,
    want_llm: bool = True,
    capital: float = 10_000.0,
    risk_pct: float = 0.01,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    sentiment_max_age_s: int = 12 * 3600,
) -> Generator[dict, None, None]:
    """Yield one event per pipeline step. Final event has step='done'."""
    # Step 1: fetch OHLCV
    t0 = time.perf_counter()
    try:
        df = fetch_ohlcv(symbol, interval=interval, period=period)
    except Exception as e:  # noqa: BLE001
        yield _event("fetch_ohlcv", "error", time.perf_counter() - t0, str(e))
        return
    yield _event(
        "fetch_ohlcv",
        "ok",
        time.perf_counter() - t0,
        {
            "bars": len(df),
            "first_ts": str(df.index[0]),
            "last_ts": str(df.index[-1]),
            "last_close": float(df["close"].iloc[-1]),
        },
    )

    # Step 2: snapshot + rule-based signal
    t0 = time.perf_counter()
    snap = build_snapshot(symbol, df)
    tech = technical_signal(snap)
    yield _event(
        "snapshot",
        "ok",
        time.perf_counter() - t0,
        {"snapshot": snap.to_dict(), "technical_signal": tech},
    )

    # Step 2.5: pattern / market structure recognition
    t0 = time.perf_counter()
    pattern = analyse_patterns(df)
    yield _event(
        "patterns", "ok", time.perf_counter() - t0, pattern.to_dict()
    )

    # Step 2.6: higher-timeframe trend
    t0 = time.perf_counter()
    try:
        higher_tf = fetch_higher_tf(symbol, interval)
    except Exception as e:  # noqa: BLE001
        higher_tf = "UNKNOWN"
        yield _event("higher_timeframe", "error", time.perf_counter() - t0, str(e))
    else:
        yield _event(
            "higher_timeframe",
            "ok",
            time.perf_counter() - t0,
            {"trend": higher_tf},
        )

    # Step 3: ATR
    t0 = time.perf_counter()
    atr_value = float(atr(df).iloc[-1])
    yield _event("atr", "ok", time.perf_counter() - t0, {"atr_14": atr_value})

    # Step 4: headlines
    headlines = []
    t0 = time.perf_counter()
    if want_news:
        headlines = fetch_headlines(symbol, limit=5)
        yield _event(
            "headlines",
            "ok",
            time.perf_counter() - t0,
            {"count": len(headlines), "items": [h.to_dict() for h in headlines]},
        )
    else:
        yield _event("headlines", "skip", 0.0)

    # Step 5: correlation
    correlation = None
    t0 = time.perf_counter()
    if want_correlation:
        others = related_for(symbol)
        if others:
            try:
                correlation = build_correlation_snapshot(
                    symbol, others=others, interval="1d", period="60d"
                )
                yield _event(
                    "correlation",
                    "ok",
                    time.perf_counter() - t0,
                    correlation.to_dict(),
                )
            except Exception as e:  # noqa: BLE001
                yield _event("correlation", "error", time.perf_counter() - t0, str(e))
        else:
            yield _event(
                "correlation", "skip", 0.0, {"note": "no related symbols configured"}
            )
    else:
        yield _event("correlation", "skip", 0.0)

    # Step 6: upcoming events + calendar freshness
    events = []
    cal_health = None
    t0 = time.perf_counter()
    if want_events and events_path is not None:
        cal_health = calendar_freshness(events_path)
        if cal_health.is_fresh:
            all_events = load_events(events_path)
            events = upcoming_for_symbol(all_events, symbol, within_hours=24)
            yield _event(
                "upcoming_events",
                "ok",
                time.perf_counter() - t0,
                {
                    "count": len(events),
                    "items": [e.to_dict() for e in events],
                    "calendar": cal_health.to_dict(),
                },
            )
        else:
            yield _event(
                "upcoming_events",
                "skip",
                time.perf_counter() - t0,
                {
                    "note": (
                        f"calendar {cal_health.status}: "
                        f"{cal_health.detail or 'unhealthy'} — "
                        "live trading would be blocked"
                    ),
                    "calendar": cal_health.to_dict(),
                },
            )
    else:
        yield _event(
            "upcoming_events",
            "skip",
            0.0,
            {"note": "no events.json; run `fx calendar-seed` to bootstrap"},
        )

    # Step 7: past lessons
    t0 = time.perf_counter()
    past_lessons = []
    if want_lessons:
        past_lessons = storage.relevant_postmortems(symbol, limit=5)
    yield _event(
        "past_lessons",
        "ok" if want_lessons else "skip",
        time.perf_counter() - t0,
        {"count": len(past_lessons), "items": past_lessons},
    )

    # Step 7.5: crowd sentiment snapshot (lazy read of data/sentiment.json)
    sentiment = None
    t0 = time.perf_counter()
    if want_sentiment:
        sentiment = read_sentiment(symbol, max_age_s=sentiment_max_age_s)
        if sentiment:
            yield _event("sentiment", "ok", time.perf_counter() - t0, sentiment)
        else:
            yield _event(
                "sentiment", "skip", 0.0,
                {"note": "no recent snapshot — run `fx sentiment refresh`"},
            )
    else:
        yield _event("sentiment", "skip", 0.0)

    # Build MarketContext for the LLM and the decision engine
    context = MarketContext(
        symbol=symbol,
        interval=interval,
        snapshot=snap,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        spread_state="UNKNOWN",
        spread_pct=None,
        event_risk_level=event_risk_level_from(
            [e.to_dict() for e in events]
        ),
        economic_events_nearby=[e.to_dict() for e in events],
        sentiment_score=(sentiment or {}).get("sentiment_score"),
        sentiment_velocity=(sentiment or {}).get("sentiment_velocity"),
        sentiment_volume_spike=detect_volume_spike(sentiment),
        top_sentiment_keywords=keywords_from_sentiment(sentiment),
        risk_reward=1.5 if atr_value > 0 else None,
        atr_stop_distance=2.0 * atr_value if atr_value > 0 else None,
        technical_signal=tech,
    )

    # Step 8: Claude (advisory only)
    llm_signal = None
    t0 = time.perf_counter()
    if want_llm and cfg.anthropic_api_key:
        try:
            analyst = Analyst(model=cfg.model, effort=cfg.effort)
            llm_signal = analyst.analyze(
                context.to_dict(),
                headlines=headlines,
                correlation=correlation,
                upcoming_events=events,
                past_postmortems=past_lessons,
                sentiment_snapshot=sentiment,
            )
            yield _event(
                "claude",
                "ok",
                time.perf_counter() - t0,
                {
                    "action": llm_signal.action,
                    "confidence": llm_signal.confidence,
                    "reason": llm_signal.reason,
                    "key_risks": llm_signal.key_risks,
                    "expected_direction": llm_signal.expected_direction,
                    "expected_magnitude_pct": llm_signal.expected_magnitude_pct,
                    "horizon_bars": llm_signal.horizon_bars,
                    "invalidation_price": llm_signal.invalidation_price,
                    "model": cfg.model,
                    "effort": cfg.effort,
                },
            )
        except Exception as e:  # noqa: BLE001
            yield _event("claude", "error", time.perf_counter() - t0, str(e))
    elif not cfg.anthropic_api_key:
        yield _event(
            "claude",
            "skip",
            0.0,
            {"note": "ANTHROPIC_API_KEY not set — technical signal only"},
        )
    else:
        yield _event("claude", "skip", 0.0, {"note": "LLM disabled by flag"})

    # Step 9: fixed-rule Decision Engine (LLM is advisory only).
    # Web pipeline is research-mode: we surface calendar health but don't
    # force HOLD on stale data; CLI's `trade --broker oanda` is what
    # enables `require_calendar_fresh=True`/`require_spread=True`.
    risk_state = RiskState(
        df=df,
        events=tuple(events),
        spread_pct=None,
        sentiment_snapshot=sentiment,
        calendar_freshness=cal_health,
    )
    decision = decide_action(
        technical_signal=tech,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        risk_reward=context.risk_reward,
        risk_state=risk_state,
        llm_signal=llm_signal,
    )
    yield _event(
        "decision",
        "ok",
        0.0,
        decision.to_dict(),
    )

    # Step 10: risk plan (only if directional)
    plan = None
    if decision.action in ("BUY", "SELL") and atr_value > 0:
        plan = plan_trade(
            side=decision.action,
            entry=snap.last_close,
            atr_value=atr_value,
            capital=capital,
            stop_mult=stop_atr,
            tp_mult=tp_atr,
            risk_pct=risk_pct,
        )
        yield _event("risk_plan", "ok", 0.0, plan.to_dict())
    else:
        yield _event(
            "risk_plan",
            "skip",
            0.0,
            {"note": "No directional action; nothing to size"},
        )

    # Step 11: persist
    t0 = time.perf_counter()
    analysis_id = storage.save_analysis(
        symbol=symbol,
        snapshot=snap.to_dict(),
        technical_signal=tech,
        final_action=decision.action,
        llm_action=llm_signal.action if llm_signal else None,
        llm_confidence=llm_signal.confidence if llm_signal else None,
        llm_reason=llm_signal.reason if llm_signal else None,
    )
    prediction_id = None
    if llm_signal is not None:
        import json as _json  # local import to avoid widening module imports
        events_json = _json.dumps(
            [e.to_dict() for e in events], default=str
        ) if events else None
        prediction_id = storage.save_prediction(
            analysis_id=analysis_id,
            symbol=symbol,
            interval=interval,
            entry_price=snap.last_close,
            action=llm_signal.action,
            confidence=llm_signal.confidence,
            reason=llm_signal.reason,
            expected_direction=llm_signal.expected_direction,
            expected_magnitude_pct=llm_signal.expected_magnitude_pct,
            horizon_bars=llm_signal.horizon_bars,
            invalidation_price=llm_signal.invalidation_price,
            final_decision_action=decision.action,
            executed_action=decision.action,
            blocked_by=",".join(decision.blocked_by) if decision.blocked_by else None,
            final_reason=decision.reason,
            rule_chain=",".join(decision.rule_chain) if decision.rule_chain else None,
            risk_reward=context.risk_reward,
            detected_pattern=pattern.detected_pattern,
            trend_state=pattern.trend_state.value,
            higher_timeframe_trend=higher_tf,
            event_risk_level=context.event_risk_level,
            economic_events_nearby=events_json,
            sentiment_score=context.sentiment_score,
            sentiment_volume_spike=1 if context.sentiment_volume_spike else 0,
            spread_at_entry=context.spread_pct,
        )
    yield _event(
        "persist",
        "ok",
        time.perf_counter() - t0,
        {"analysis_id": analysis_id, "prediction_id": prediction_id},
    )

    yield _event("done", "ok", 0.0, {"analysis_id": analysis_id})
