"""Command-line entry point.

Usage examples:
  python -m src.fx.cli analyze --symbol USDJPY=X --interval 1h
  python -m src.fx.cli backtest --symbol USDJPY=X --interval 1h --period 180d
  python -m src.fx.cli review --limit 50
  python -m src.fx.cli watch --symbol USDJPY=X --interval 15m --every 900
  python -m src.fx.cli trade --symbol USDJPY=X --dry-run
  python -m src.fx.cli calendar-seed        # create a placeholder events.json
  python -m src.fx.cli evaluate             # score pending predictions
  python -m src.fx.cli postmortem           # learn from wrong predictions
  python -m src.fx.cli lessons --symbol USDJPY=X
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .analyst import Analyst
from .backtest import run_backtest
from .backtest_engine import run_engine_backtest
from .event_overlay import KEYWORD_MAP, overlay_events
from .macro import MACRO_SYMBOLS, fetch_macro_snapshot
from .market_timeline import build_timeline
from .waveform_backtest import waveform_lookup
from .waveform_library import build_library, read_library, write_library
from .waveform_matcher import compute_signature
from .broker import Broker, PaperBroker
from .calendar import (
    calendar_freshness,
    load_events,
    seed_defaults,
    upcoming_for_symbol,
)
from .config import Config
from .context import (
    MarketContext,
    detect_volume_spike,
    event_risk_level_from,
    keywords_from_sentiment,
)
from .correlation import build_correlation_snapshot, related_for
from .data import fetch_ohlcv
from .decision_engine import decide as decide_action
from .higher_timeframe import fetch_and_classify as fetch_higher_tf
from .indicators import build_snapshot, technical_signal
from .news import fetch_headlines
from .patterns import analyse as analyse_patterns
from .postmortem import Postmortem
from .prediction import evaluate_prediction, slice_bars_after
from .risk import atr, plan_trade
from .risk_gate import RiskState
from .sentiment import DEFAULT_PATH as SENTIMENT_PATH
from .sentiment import read_for as read_sentiment
from .storage import Storage
from src.sentiment.refresh import refresh as refresh_sentiment_snapshot
from src.sentiment.snapshot import load_snapshot as load_sentiment_snapshot

from src.notify import (
    Notification,
    build_notifier,
    format_lesson,
    format_signal,
    format_summary,
    format_verdict,
)

EVENTS_PATH = Path("data/events.json")


def _notify(storage: Storage, kind: str, text: str, min_confidence: float = 0.0) -> None:
    """Send `text` to every active subscriber that opted into `kind`.

    Failures are silent — the analysis path must never crash because
    LINE is down.
    """
    try:
        notifier = build_notifier()
        if notifier.name == "null":
            return
        subs = storage.active_subscribers(kind=kind)
        recipients = [
            s["user_id"]
            for s in subs
            if (s.get("min_confidence") or 0.0) <= min_confidence
        ]
        if not recipients and notifier.name == "log":
            recipients = ["broadcast"]
        if not recipients:
            return
        notifier.push(
            Notification(text=text, kind=kind, recipients=tuple(recipients))
        )
    except Exception:  # noqa: BLE001
        pass


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _gather_inputs(args, cfg: Config, storage: Storage):
    """Common input fetch for analyze / trade / watch.

    Builds a MarketContext, runs the fixed-rule Decision Engine, and
    returns everything the callers and notifications need. The LLM is
    consulted but cannot override the engine.
    """
    df = fetch_ohlcv(args.symbol, interval=args.interval, period=args.period)
    snap = build_snapshot(args.symbol, df)
    tech = technical_signal(snap)
    pattern = analyse_patterns(df)
    atr_value = float(atr(df).iloc[-1])

    headlines = []
    if not args.no_news:
        headlines = fetch_headlines(args.symbol, limit=args.news_limit)

    correlation = None
    if not args.no_correlation:
        others = related_for(args.symbol)
        if others:
            try:
                correlation = build_correlation_snapshot(
                    args.symbol, others=others, interval="1d", period="60d"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[warn] correlation fetch failed: {e}", file=sys.stderr)

    events = []
    cal_health = None
    if not args.no_events:
        cal_health = calendar_freshness(EVENTS_PATH)
        if cal_health.is_fresh:
            all_events = load_events(EVENTS_PATH)
            events = upcoming_for_symbol(
                all_events, args.symbol, within_hours=args.event_window_hours
            )
        else:
            print(
                f"[warn] calendar {cal_health.status}: "
                f"{cal_health.detail or 'unhealthy'}",
                file=sys.stderr,
            )

    past_lessons = []
    if not args.no_lessons:
        past_lessons = storage.relevant_postmortems(
            args.symbol, limit=args.lesson_limit
        )

    sentiment = None
    if not args.no_sentiment:
        sentiment = read_sentiment(args.symbol, max_age_s=12 * 3600)

    higher_tf = "UNKNOWN"
    if not getattr(args, "no_higher_tf", False):
        try:
            higher_tf = fetch_higher_tf(args.symbol, args.interval)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] higher-timeframe fetch failed: {e}", file=sys.stderr)

    # Estimate risk-reward from ATR (stop=2*ATR, TP=3*ATR → 1.5 floor)
    risk_reward = 1.5 if atr_value > 0 else None
    atr_stop_distance = 2.0 * atr_value if atr_value > 0 else None

    context = MarketContext(
        symbol=args.symbol,
        interval=args.interval,
        snapshot=snap,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        spread_state="UNKNOWN",  # yfinance has no bid/ask; left UNKNOWN
        spread_pct=None,
        event_risk_level=event_risk_level_from([e.to_dict() for e in events]),
        economic_events_nearby=[e.to_dict() for e in events],
        sentiment_score=(sentiment or {}).get("sentiment_score"),
        sentiment_velocity=(sentiment or {}).get("sentiment_velocity"),
        sentiment_volume_spike=detect_volume_spike(sentiment),
        top_sentiment_keywords=keywords_from_sentiment(sentiment),
        risk_reward=risk_reward,
        atr_stop_distance=atr_stop_distance,
        technical_signal=tech,
    )

    llm_signal = None
    if cfg.anthropic_api_key and not args.no_llm:
        analyst = Analyst(model=cfg.model, effort=cfg.effort)
        llm_signal = analyst.analyze(
            context.to_dict(),
            headlines=headlines,
            correlation=correlation,
            upcoming_events=events,
            past_postmortems=past_lessons,
            sentiment_snapshot=sentiment,
        )

    # `require_calendar_fresh` / `require_spread` are flipped on by trade callers
    # at decision time (see cmd_trade); analyze() leaves them False so research
    # workflows still produce useful output when feeds are stale.
    require_calendar_fresh = bool(getattr(args, "require_fresh_calendar", False))
    require_spread = bool(getattr(args, "require_spread", False))

    risk_state = RiskState(
        df=df,
        events=tuple(events),
        spread_pct=getattr(args, "_spread_pct_override", None),
        sentiment_snapshot=sentiment,
        calendar_freshness=cal_health,
        require_calendar_fresh=require_calendar_fresh,
        require_spread=require_spread,
    )

    decision = decide_action(
        technical_signal=tech,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        risk_reward=risk_reward,
        risk_state=risk_state,
        llm_signal=llm_signal,
    )

    return {
        "df": df,
        "snapshot": snap,
        "context": context,
        "pattern": pattern,
        "technical": tech,
        "headlines": headlines,
        "correlation": correlation,
        "events": events,
        "calendar_freshness": cal_health,
        "past_lessons": past_lessons,
        "sentiment": sentiment,
        "higher_tf": higher_tf,
        "llm": llm_signal,
        "decision": decision,
        "atr": atr_value,
    }


def cmd_analyze(args, cfg: Config, storage: Storage) -> int:
    ctx = _gather_inputs(args, cfg, storage)
    snap, tech, llm, decision = (
        ctx["snapshot"],
        ctx["technical"],
        ctx["llm"],
        ctx["decision"],
    )
    analysis_id = storage.save_analysis(
        symbol=args.symbol,
        snapshot=snap.to_dict(),
        technical_signal=tech,
        final_action=decision.action,
        llm_action=llm.action if llm else None,
        llm_confidence=llm.confidence if llm else None,
        llm_reason=llm.reason if llm else None,
    )

    prediction_id = None
    if llm is not None:
        # Spec §13: distinguish what the LLM advised, what the engine
        # decided, and what was actually executed. analyze() doesn't place
        # orders, so executed_action mirrors the engine's final action.
        ctx_obj = ctx["context"]
        events_json = json.dumps(
            [e.to_dict() for e in ctx["events"]], default=str
        ) if ctx["events"] else None
        prediction_id = storage.save_prediction(
            analysis_id=analysis_id,
            symbol=args.symbol,
            interval=args.interval,
            entry_price=snap.last_close,
            action=llm.action,
            confidence=llm.confidence,
            reason=llm.reason,
            expected_direction=llm.expected_direction,
            expected_magnitude_pct=llm.expected_magnitude_pct,
            horizon_bars=llm.horizon_bars,
            invalidation_price=llm.invalidation_price,
            final_decision_action=decision.action,
            executed_action=decision.action,
            blocked_by=",".join(decision.blocked_by) if decision.blocked_by else None,
            final_reason=decision.reason,
            rule_chain=",".join(decision.rule_chain) if decision.rule_chain else None,
            risk_reward=ctx_obj.risk_reward,
            detected_pattern=(
                ctx["pattern"].detected_pattern if ctx["pattern"] else None
            ),
            trend_state=(
                ctx["pattern"].trend_state.value if ctx["pattern"] else None
            ),
            higher_timeframe_trend=ctx["higher_tf"],
            event_risk_level=ctx_obj.event_risk_level,
            economic_events_nearby=events_json,
            sentiment_score=ctx_obj.sentiment_score,
            sentiment_volume_spike=(
                1 if ctx_obj.sentiment_volume_spike else 0
            ),
            spread_at_entry=ctx_obj.spread_pct,
        )

    # Push notification when LLM produced a directional, confident signal.
    if llm and decision.action in ("BUY", "SELL"):
        _notify(
            storage,
            kind="signal",
            min_confidence=llm.confidence,
            text=format_signal(
                symbol=args.symbol,
                action=llm.action,
                confidence=llm.confidence,
                expected_direction=llm.expected_direction,
                expected_magnitude_pct=llm.expected_magnitude_pct,
                horizon_bars=llm.horizon_bars,
                invalidation_price=llm.invalidation_price,
                reason=llm.reason,
                analysis_id=analysis_id,
            ),
        )

    _print(
        {
            "analysis_id": analysis_id,
            "prediction_id": prediction_id,
            "snapshot": snap.to_dict(),
            "technical_signal": tech,
            "atr_14": round(ctx["atr"], 6),
            "correlation": (
                ctx["correlation"].to_dict() if ctx["correlation"] else None
            ),
            "upcoming_events": [e.to_dict() for e in ctx["events"]],
            "headlines": [h.to_dict() for h in ctx["headlines"]],
            "past_lessons_used": len(ctx["past_lessons"]),
            "llm": (
                {
                    "action": llm.action,
                    "confidence": llm.confidence,
                    "reason": llm.reason,
                    "key_risks": llm.key_risks,
                    "expected_direction": llm.expected_direction,
                    "expected_magnitude_pct": llm.expected_magnitude_pct,
                    "horizon_bars": llm.horizon_bars,
                    "invalidation_price": llm.invalidation_price,
                }
                if llm
                else None
            ),
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        }
    )
    return 0


def cmd_trade(args, cfg: Config, storage: Storage) -> int:
    """Run an analysis and, if the decision is directional, place a trade.

    Default broker is PaperBroker. Real-broker support is explicitly opt-in
    via --broker oanda (requires environment guards, see src/fx/oanda.py).

    Live brokers (oanda) trigger two extra guards on top of the analyst path:
      * `require_calendar_fresh`: events.json must be fresh (spec §11)
      * `require_spread`: broker must return a Bid/Ask quote (spec §12)
    Both flow through Risk Gate, so the Decision Engine returns HOLD when
    either is missing — no order is placed.
    """
    is_live_broker = args.broker == "oanda"

    # Build broker first so we can query its quote BEFORE running the
    # Decision Engine. Quote is what produces spread_pct.
    broker: Broker = _build_broker(args)
    quote = None
    spread_pct = None
    if is_live_broker:
        try:
            quote = broker.quote(args.symbol)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] broker.quote failed: {e}", file=sys.stderr)
        if quote is not None:
            spread_pct = quote.spread_pct

    # Make the gathering step aware of live-broker enforcement. We reuse
    # _gather_inputs by stamping the args with the override fields it
    # checks via getattr — keeps backward compat for analyze/watch.
    args._spread_pct_override = spread_pct
    args.require_fresh_calendar = is_live_broker
    args.require_spread = is_live_broker

    ctx = _gather_inputs(args, cfg, storage)
    snap, decision, atr_value = ctx["snapshot"], ctx["decision"], ctx["atr"]

    if decision.action == "HOLD":
        _print(
            {
                "action": "HOLD",
                "reason": decision.reason,
                "blocked_by": list(decision.blocked_by),
                "spread_pct": spread_pct,
                "calendar": (
                    ctx["calendar_freshness"].to_dict()
                    if ctx.get("calendar_freshness") else None
                ),
            }
        )
        return 0

    if atr_value <= 0:
        print("[warn] ATR is non-positive; cannot size trade.", file=sys.stderr)
        return 1

    plan = plan_trade(
        side=decision.action,
        entry=snap.last_close,
        atr_value=atr_value,
        capital=args.capital,
        stop_mult=args.stop_atr,
        tp_mult=args.tp_atr,
        risk_pct=args.risk_pct,
    )

    if args.dry_run:
        _print(
            {
                "mode": "dry-run",
                "symbol": args.symbol,
                "decision": {
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
                "plan": plan.to_dict(),
                "spread_pct": spread_pct,
                "would_place_order": True,
            }
        )
        return 0

    pos = broker.place_order(
        symbol=args.symbol,
        side=plan.side,
        size=plan.size,
        price=plan.entry,
        stop=plan.stop,
        take_profit=plan.take_profit,
    )
    _print(
        {
            "mode": "live",
            "broker": args.broker,
            "position": {
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": pos.entry,
                "size": pos.size,
                "stop": pos.stop,
                "take_profit": pos.take_profit,
            },
            "plan": plan.to_dict(),
        }
    )
    return 0


def _build_broker(args) -> Broker:
    if args.broker == "paper":
        return PaperBroker(initial_cash=args.capital)
    if args.broker == "oanda":
        from .oanda import OANDABroker, OANDAConfig  # lazy import

        cfg = OANDAConfig.from_env()
        return OANDABroker(cfg, confirm_demo=args.confirm_demo)
    raise ValueError(f"Unknown broker: {args.broker}")


def cmd_backtest(args, cfg: Config, storage: Storage) -> int:
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    result = run_backtest(df, symbol=args.symbol, warmup=args.warmup)
    metrics = result.metrics()
    start_date = str(df.index[0])
    end_date = str(df.index[-1])
    storage.save_backtest(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        strategy="technical_consensus",
        metrics=metrics,
    )
    _print(
        {
            "symbol": args.symbol,
            "interval": args.interval,
            "period": f"{start_date} -> {end_date}",
            "bars": len(df),
            "metrics": metrics,
            "last_5_trades": [
                {
                    "side": t.side,
                    "entry": round(t.entry, 6),
                    "exit": round(t.exit, 6),
                    "return_pct": round(t.return_pct, 3),
                    "entry_ts": str(t.entry_ts),
                    "exit_ts": str(t.exit_ts),
                }
                for t in result.trades[-5:]
            ],
        }
    )
    return 0


def cmd_backtest_engine(args, cfg: Config, storage: Storage) -> int:
    """Backtest using the full Decision Engine + Risk Gate chain (spec §4)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    events = ()
    if not args.no_events:
        cal_health = calendar_freshness(EVENTS_PATH)
        if cal_health.is_fresh:
            events = tuple(load_events(EVENTS_PATH))
        else:
            print(
                f"[warn] calendar {cal_health.status}: "
                f"{cal_health.detail or 'unhealthy'} — "
                "running backtest without macro events",
                file=sys.stderr,
            )

    result = run_engine_backtest(
        df,
        symbol=args.symbol,
        interval=args.interval,
        warmup=args.warmup,
        stop_atr_mult=args.stop_atr,
        tp_atr_mult=args.tp_atr,
        max_holding_bars=args.max_holding_bars,
        events=events,
        use_higher_tf=not args.no_higher_tf,
    )
    metrics = result.metrics()
    start_date = str(df.index[0])
    end_date = str(df.index[-1])
    storage.save_backtest(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        strategy="decision_engine",
        metrics=metrics,
    )
    _print(
        {
            "symbol": args.symbol,
            "interval": args.interval,
            "period": f"{start_date} -> {end_date}",
            "bars": len(df),
            "metrics": metrics,
            "last_5_trades": [
                {
                    "side": t.side,
                    "entry": round(t.entry, 6),
                    "exit": round(t.exit, 6),
                    "return_pct": round(t.return_pct, 3),
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                    "entry_ts": str(t.entry_ts),
                    "exit_ts": str(t.exit_ts),
                }
                for t in result.trades[-5:]
            ],
        }
    )
    return 0


def cmd_build_timeline(args, cfg: Config, storage: Storage) -> int:
    """Build a point-in-time market timeline (spec §5)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    events: tuple = ()
    if not args.no_events:
        cal_health = calendar_freshness(EVENTS_PATH)
        if cal_health.is_fresh:
            events = tuple(load_events(EVENTS_PATH))
        else:
            print(
                f"[warn] calendar {cal_health.status}: "
                f"{cal_health.detail or 'unhealthy'} — "
                "timeline will have no event columns",
                file=sys.stderr,
            )

    macro = None
    if not args.no_macro:
        try:
            macro = fetch_macro_snapshot(
                df.index,
                interval="1d",
                period=args.macro_period,
                slots=args.macro_slots if args.macro_slots else None,
            )
            if macro.fetch_errors:
                print(
                    f"[warn] macro fetch errors: {macro.fetch_errors}",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] macro fetch failed entirely: {e}", file=sys.stderr)

    timeline = build_timeline(
        df,
        symbol=args.symbol,
        interval=args.interval,
        warmup=args.warmup,
        events=events,
        macro=macro,
        use_higher_tf=not args.no_higher_tf,
    )

    if args.output:
        out_path = Path(args.output)
        frame = timeline.to_frame()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            frame.to_parquet(out_path)
        else:
            # Default to CSV; nested cols get JSON-stringified for safety
            for col in frame.columns:
                if frame[col].dtype == object:
                    frame[col] = frame[col].apply(
                        lambda x: json.dumps(x, default=str)
                        if isinstance(x, (list, dict)) else x
                    )
            frame.to_csv(out_path)

    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "rows": len(timeline),
        "first_ts": (
            str(timeline.rows[0].timestamp) if timeline.rows else None
        ),
        "last_ts": (
            str(timeline.rows[-1].timestamp) if timeline.rows else None
        ),
        "macro_slots": (
            list(macro.series.keys()) if macro else []
        ),
        "macro_errors": (
            macro.fetch_errors if macro else {}
        ),
        "events_used": len(events),
        "output": args.output,
        "preview": [
            timeline.rows[i].to_dict()
            for i in (0, len(timeline) // 2, len(timeline) - 1)
            if 0 <= i < len(timeline)
        ],
    }
    _print(summary)
    return 0


def cmd_event_overlay(args, cfg: Config, storage: Storage) -> int:
    """Aggregate pre/post-event price action (spec §6)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    cal_health = calendar_freshness(EVENTS_PATH)
    if not cal_health.is_fresh:
        print(
            f"[warn] calendar {cal_health.status}: "
            f"{cal_health.detail or 'unhealthy'}",
            file=sys.stderr,
        )
    events = load_events(EVENTS_PATH) if EVENTS_PATH.exists() else []
    if not events:
        _print({"error": "no events available", "calendar": cal_health.to_dict()})
        return 1

    result = overlay_events(
        df, events,
        keyword=args.event,
        impact=args.impact,
    )
    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "filter_keyword": args.event,
        "filter_impact": args.impact,
        "n_events_matched": len(result.rows),
        "aggregate": result.aggregate(),
        "events": [r.to_dict() for r in result.rows[: args.limit]],
    })
    return 0


def cmd_waveform_build_library(args, cfg: Config, storage: Storage) -> int:
    """Build a waveform sample library from a long history (spec §7.2)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    samples = build_library(
        df,
        symbol=args.symbol,
        timeframe=args.interval,
        window_bars=args.window_bars,
        step_bars=args.step_bars,
        forward_horizons=tuple(args.horizons),
        normalize=args.normalize,
    )
    out_path = Path(args.output)
    n = write_library(out_path, samples, append=args.append)
    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "window_bars": args.window_bars,
        "step_bars": args.step_bars,
        "horizons": args.horizons,
        "samples_written": n,
        "output": str(out_path),
        "appended": bool(args.append),
        "first_sample_end": (
            samples[0].end_ts.isoformat() if samples else None
        ),
        "last_sample_end": (
            samples[-1].end_ts.isoformat() if samples else None
        ),
    })
    return 0


def cmd_waveform_match(args, cfg: Config, storage: Storage) -> int:
    """Match the current window against a library and report a bias (spec §7.3)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
    )
    if len(df) < args.window:
        print(f"[err] need ≥ {args.window} bars; got {len(df)}", file=sys.stderr)
        return 1
    target_window = df.iloc[-args.window:]
    target_sig = compute_signature(target_window, method=args.normalize)

    library = read_library(Path(args.library))
    if not library:
        print(
            f"[err] empty/missing library at {args.library} — "
            "run `waveform-build-library` first",
            file=sys.stderr,
        )
        return 1

    bias, matches = waveform_lookup(
        target_sig,
        library,
        horizon_bars=args.horizon,
        method=args.method,
        top_k=args.top_k,
        min_score=args.min_score,
        min_sample_count=args.min_samples,
        min_directional_share=args.min_share,
    )

    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "window_bars": args.window,
        "library": args.library,
        "library_size": len(library),
        "method": args.method,
        "horizon_bars": args.horizon,
        "bias": bias.to_dict(),
        "top_matches": [m.to_dict() for m in matches[: args.show]],
    })
    return 0


def cmd_review(args, cfg: Config, storage: Storage) -> int:
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot run review.", file=sys.stderr)
        return 1
    trades = storage.recent_trades(limit=args.limit)
    analyses = storage.recent_analyses(limit=args.limit)
    analyst = Analyst(model=cfg.model, effort="high")
    print(analyst.review(trades or analyses))
    return 0


def cmd_watch(args, cfg: Config, storage: Storage) -> int:
    while True:
        try:
            cmd_analyze(args, cfg, storage)
        except Exception as e:  # noqa: BLE001
            print(f"[watch] error: {e}", file=sys.stderr)
        time.sleep(args.every)


def cmd_evaluate(args, cfg: Config, storage: Storage) -> int:
    """Score every PENDING prediction whose horizon has elapsed."""
    pending = storage.pending_predictions()
    if not pending:
        _print({"evaluated": 0, "note": "No pending predictions."})
        return 0

    summary = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0,
               "INCONCLUSIVE": 0, "INSUFFICIENT_DATA": 0}
    details = []

    by_symbol: dict[tuple[str, str], list[dict]] = {}
    for p in pending:
        by_symbol.setdefault((p["symbol"], p["interval"]), []).append(p)

    for (symbol, interval), preds in by_symbol.items():
        try:
            df = fetch_ohlcv(symbol, interval=interval, period=args.period)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not fetch {symbol} {interval}: {e}", file=sys.stderr)
            continue
        for p in preds:
            after = slice_bars_after(df, p["ts"])
            verdict = evaluate_prediction(p, after)
            if verdict.status == "INSUFFICIENT_DATA":
                summary["INSUFFICIENT_DATA"] += 1
                continue
            storage.update_prediction_evaluation(
                prediction_id=p["id"],
                status=verdict.status,
                actual_direction=verdict.actual_direction,
                actual_magnitude_pct=verdict.actual_magnitude_pct,
                max_favorable_pct=verdict.max_favorable_pct,
                max_adverse_pct=verdict.max_adverse_pct,
                invalidation_hit=verdict.invalidation_hit,
                note=verdict.note,
            )
            summary[verdict.status] = summary.get(verdict.status, 0) + 1
            details.append(
                {
                    "prediction_id": p["id"],
                    "symbol": p["symbol"],
                    "action": p["action"],
                    "verdict": verdict.to_dict(),
                }
            )
            # Push verdict for terminal outcomes only (not INSUFFICIENT_DATA).
            if verdict.status in ("CORRECT", "PARTIAL", "WRONG"):
                _notify(
                    storage,
                    kind="verdict",
                    text=format_verdict(
                        symbol=p["symbol"],
                        action=p["action"],
                        status=verdict.status,
                        actual_direction=verdict.actual_direction,
                        actual_magnitude_pct=verdict.actual_magnitude_pct,
                        note=verdict.note,
                        analysis_id=p.get("analysis_id"),
                    ),
                )

    if any(summary.values()):
        _notify(
            storage,
            kind="summary",
            text=format_summary(period="evaluate", counts=summary),
        )
    _print({"summary": summary, "details": details})
    return 0


def cmd_postmortem(args, cfg: Config, storage: Storage) -> int:
    """Run Claude post-mortems on WRONG predictions that don't have one yet."""
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot run post-mortems.", file=sys.stderr)
        return 1

    wrongs = storage.wrong_predictions_without_postmortem(limit=args.limit)
    if not wrongs:
        _print({"analyzed": 0, "note": "No wrong predictions awaiting post-mortem."})
        return 0

    pm = Postmortem(model=cfg.model, effort="high")
    written = []
    for p in wrongs:
        verdict_view = {
            "status": p["status"],
            "actual_direction": p["actual_direction"],
            "actual_magnitude_pct": p["actual_magnitude_pct"],
            "max_favorable_pct": p["max_favorable_pct"],
            "max_adverse_pct": p["max_adverse_pct"],
            "invalidation_hit": bool(p["invalidation_hit"]),
            "note": p["evaluation_note"],
        }
        try:
            result = pm.analyze(p, verdict_view)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] post-mortem failed for prediction {p['id']}: {e}",
                  file=sys.stderr)
            continue
        # Spec §13: capture the diagnostic snapshot of what the engine
        # saw at decision time, so future readers see the WHY, not just
        # the prediction.
        pm_context = {
            "final_decision_action": p.get("final_decision_action"),
            "executed_action": p.get("executed_action"),
            "blocked_by": p.get("blocked_by"),
            "rule_chain": p.get("rule_chain"),
            "detected_pattern": p.get("detected_pattern"),
            "trend_state": p.get("trend_state"),
            "higher_timeframe_trend": p.get("higher_timeframe_trend"),
            "event_risk_level": p.get("event_risk_level"),
            "economic_events_nearby": p.get("economic_events_nearby"),
            "sentiment_score": p.get("sentiment_score"),
            "sentiment_volume_spike": p.get("sentiment_volume_spike"),
            "spread_at_entry": p.get("spread_at_entry"),
            "risk_reward": p.get("risk_reward"),
            "rule_version": p.get("rule_version"),
            "final_reason": p.get("final_reason"),
        }
        pm_id = storage.save_postmortem(
            prediction_id=p["id"],
            root_cause=result.root_cause,
            narrative=result.narrative,
            proposed_rule=result.proposed_rule,
            tags=",".join(result.tags),
            context=pm_context,
        )
        written.append(
            {
                "postmortem_id": pm_id,
                "prediction_id": p["id"],
                "root_cause": result.root_cause,
                "proposed_rule": result.proposed_rule,
            }
        )
        _notify(
            storage,
            kind="lesson",
            text=format_lesson(
                symbol=p["symbol"],
                root_cause=result.root_cause,
                proposed_rule=result.proposed_rule,
                narrative=result.narrative,
                analysis_id=p.get("analysis_id"),
            ),
        )

    _print({"written": written, "count": len(written)})
    return 0


def cmd_lessons(args, cfg: Config, storage: Storage) -> int:
    """Show what we've learned so far."""
    summary = storage.lesson_summary()
    recent = storage.relevant_postmortems(args.symbol, limit=args.limit) if args.symbol else []
    _print(
        {
            "root_cause_counts": summary,
            "recent_for_symbol": recent,
            "symbol": args.symbol,
        }
    )
    return 0


def cmd_sentiment_refresh(args, cfg: Config, storage: Storage) -> int:
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot score sentiment.", file=sys.stderr)
        return 1
    enable = {
        "reddit": not args.no_reddit,
        "stocktwits": not args.no_stocktwits,
        "tradingview": not args.no_tradingview,
        "twitter": not args.no_twitter,
        "rss": not args.no_rss,
    }
    result = refresh_sentiment_snapshot(
        symbols=args.symbols,
        output_path=SENTIMENT_PATH,
        enable=enable,
        translate=args.translate,
    )
    _print(
        {
            "elapsed_s": result.elapsed_s,
            "twitter_backend": result.twitter_backend,
            "counts_by_source": result.counts_by_source,
            "errors_by_source": result.errors_by_source,
            "snapshots": {
                sym: {
                    "mention_count_24h": s.mention_count_24h,
                    "sentiment_score": s.sentiment_score,
                    "sentiment_velocity": s.sentiment_velocity,
                }
                for sym, s in result.snapshots.items()
            },
        }
    )
    return 0


def cmd_sentiment_show(args, cfg: Config, storage: Storage) -> int:
    snaps = load_sentiment_snapshot(SENTIMENT_PATH)
    if args.symbol:
        snap = snaps.get(args.symbol)
        if snap is None:
            _print({"error": f"no snapshot for {args.symbol}"})
            return 1
        _print(snap.to_dict())
    else:
        _print(
            {
                "symbols": {sym: s.to_dict() for sym, s in snaps.items()},
            }
        )
    return 0


def cmd_calendar_seed(args, cfg: Config, storage: Storage) -> int:
    events = seed_defaults(EVENTS_PATH)
    _print(
        {
            "path": str(EVENTS_PATH),
            "seeded": [e.to_dict() for e in events],
            "note": "These are placeholder entries. Replace with a real feed.",
        }
    )
    return 0


def _add_context_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by analyze / watch / trade."""
    parser.add_argument("--symbol", default="USDJPY=X")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--no-correlation", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-lessons", action="store_true",
                        help="Skip injecting past post-mortems")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip the cached crowd-sentiment snapshot")
    parser.add_argument("--no-higher-tf", action="store_true",
                        help="Skip higher-timeframe fetch (faster, less safe)")
    parser.add_argument("--lesson-limit", type=int, default=5)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--event-window-hours", type=int, default=24)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fx", description="FX analysis toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Run a single analysis")
    _add_context_flags(a)
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("trade", help="Analyze and optionally place an order")
    _add_context_flags(t)
    t.add_argument("--broker", choices=["paper", "oanda"], default="paper")
    t.add_argument("--capital", type=float, default=10_000.0)
    t.add_argument("--risk-pct", type=float, default=0.01,
                   help="Fraction of capital risked per trade (default 1%%)")
    t.add_argument("--stop-atr", type=float, default=2.0)
    t.add_argument("--tp-atr", type=float, default=3.0)
    t.add_argument("--dry-run", action="store_true",
                   help="Print the plan but do not place the order")
    t.add_argument("--confirm-demo", action="store_true",
                   help="Required for OANDA broker; acknowledges demo-only use")
    t.set_defaults(func=cmd_trade)

    b = sub.add_parser("backtest", help="Backtest technical strategy")
    b.add_argument("--symbol", default="USDJPY=X")
    b.add_argument("--interval", default="1h")
    b.add_argument("--period", default="180d")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--warmup", type=int, default=50)
    b.set_defaults(func=cmd_backtest)

    bt = sub.add_parser(
        "build-timeline",
        help="Build a point-in-time market timeline (spec §5)",
    )
    bt.add_argument("--symbol", default="USDJPY=X")
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--period", default="180d")
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--warmup", type=int, default=50)
    bt.add_argument("--no-events", action="store_true")
    bt.add_argument("--no-macro", action="store_true",
                    help="Skip yields/DXY/VIX/equities fetch")
    bt.add_argument(
        "--macro-slots", nargs="*", default=None,
        help=f"Subset of {sorted(MACRO_SYMBOLS.keys())}; default = all",
    )
    bt.add_argument("--macro-period", default="2y",
                    help="yfinance period for macro fetch")
    bt.add_argument("--no-higher-tf", action="store_true")
    bt.add_argument(
        "--output", default=None,
        help="Optional path; .csv or .parquet — written if set",
    )
    bt.set_defaults(func=cmd_build_timeline)

    wbl = sub.add_parser(
        "waveform-build-library",
        help="Slide a window over history and write a JSONL waveform library (spec §7.2)",
    )
    wbl.add_argument("--symbol", default="USDJPY=X")
    wbl.add_argument("--interval", default="1h")
    wbl.add_argument("--period", default="2y")
    wbl.add_argument("--start", default=None)
    wbl.add_argument("--end", default=None)
    wbl.add_argument("--window-bars", type=int, default=60)
    wbl.add_argument("--step-bars", type=int, default=5)
    wbl.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24])
    wbl.add_argument("--normalize",
                     choices=["z_score", "start_price", "min_max"],
                     default="z_score")
    wbl.add_argument("--output", default="data/waveforms/library.jsonl")
    wbl.add_argument("--append", action="store_true",
                     help="Append to an existing library instead of overwriting")
    wbl.set_defaults(func=cmd_waveform_build_library)

    wm = sub.add_parser(
        "waveform-match",
        help="Match the most recent window against a library; report bias (spec §7.3)",
    )
    wm.add_argument("--symbol", default="USDJPY=X")
    wm.add_argument("--interval", default="1h")
    wm.add_argument("--period", default="60d")
    wm.add_argument("--window", type=int, default=60,
                    help="Bars at the right edge of the chart used as the target")
    wm.add_argument("--library", default="data/waveforms/library.jsonl")
    wm.add_argument("--method", choices=["dtw", "cosine", "correlation"],
                    default="dtw")
    wm.add_argument("--horizon", type=int, default=24)
    wm.add_argument("--top-k", type=int, default=30)
    wm.add_argument("--min-score", type=float, default=0.55)
    wm.add_argument("--min-samples", type=int, default=20)
    wm.add_argument("--min-share", type=float, default=0.6)
    wm.add_argument("--show", type=int, default=5,
                    help="How many top matches to print")
    wm.add_argument("--normalize",
                    choices=["z_score", "start_price", "min_max"],
                    default="z_score")
    wm.set_defaults(func=cmd_waveform_match)

    eo = sub.add_parser(
        "event-overlay",
        help="Aggregate pre/post-event price action (spec §6)",
    )
    eo.add_argument("--symbol", default="USDJPY=X")
    eo.add_argument("--interval", default="1h")
    eo.add_argument("--period", default="2y")
    eo.add_argument("--start", default=None)
    eo.add_argument("--end", default=None)
    eo.add_argument(
        "--event", default=None,
        choices=[k for k, _ in KEYWORD_MAP],
        help="Restrict to events matching this keyword",
    )
    eo.add_argument(
        "--impact", default="high",
        choices=["high", "medium", "low"],
        help="Restrict to events of this impact (None to include all)",
    )
    eo.add_argument("--limit", type=int, default=20,
                    help="Max per-event rows to print (aggregate is full)")
    eo.set_defaults(func=cmd_event_overlay)

    be = sub.add_parser(
        "backtest-engine",
        help="Backtest with the full Decision Engine + Risk Gate chain (spec §4)",
    )
    be.add_argument("--symbol", default="USDJPY=X")
    be.add_argument("--interval", default="1h")
    be.add_argument("--period", default="180d")
    be.add_argument("--start", default=None)
    be.add_argument("--end", default=None)
    be.add_argument("--warmup", type=int, default=50)
    be.add_argument("--stop-atr", type=float, default=2.0)
    be.add_argument("--tp-atr", type=float, default=3.0)
    be.add_argument("--max-holding-bars", type=int, default=48)
    be.add_argument("--no-events", action="store_true",
                    help="Ignore data/events.json (skip macro-event gate)")
    be.add_argument("--no-higher-tf", action="store_true",
                    help="Skip higher-timeframe alignment check")
    be.set_defaults(func=cmd_backtest_engine)

    r = sub.add_parser("review", help="Weekly self-review via Claude")
    r.add_argument("--limit", type=int, default=50)
    r.set_defaults(func=cmd_review)

    w = sub.add_parser("watch", help="Continuously analyze at a fixed interval")
    _add_context_flags(w)
    w.add_argument("--every", type=int, default=900, help="Seconds between analyses")
    w.set_defaults(func=cmd_watch)

    c = sub.add_parser("calendar-seed", help="Write a placeholder events.json")
    c.set_defaults(func=cmd_calendar_seed)

    e = sub.add_parser(
        "evaluate", help="Score pending predictions against actual price action"
    )
    e.add_argument(
        "--period", default="60d",
        help="Lookback window for price fetch (must cover the prediction's horizon)",
    )
    e.set_defaults(func=cmd_evaluate)

    pm = sub.add_parser(
        "postmortem", help="Run Claude post-mortems on WRONG predictions"
    )
    pm.add_argument("--limit", type=int, default=10)
    pm.set_defaults(func=cmd_postmortem)

    ls = sub.add_parser("lessons", help="Show accumulated post-mortem lessons")
    ls.add_argument("--symbol", default=None,
                    help="If set, also list recent lessons for this symbol")
    ls.add_argument("--limit", type=int, default=10)
    ls.set_defaults(func=cmd_lessons)

    sr = sub.add_parser("sentiment-refresh",
                        help="Collect + score crowd sentiment for the listed symbols")
    sr.add_argument("--symbols", nargs="+", default=["USDJPY=X", "EURUSD=X", "BTC-USD"])
    sr.add_argument("--no-reddit", action="store_true")
    sr.add_argument("--no-stocktwits", action="store_true")
    sr.add_argument("--no-tradingview", action="store_true")
    sr.add_argument("--no-twitter", action="store_true")
    sr.add_argument("--no-rss", action="store_true")
    sr.add_argument("--translate", action="store_true",
                    help="Translate non-English posts to English before scoring")
    sr.set_defaults(func=cmd_sentiment_refresh)

    ss = sub.add_parser("sentiment-show", help="Print the cached sentiment snapshot")
    ss.add_argument("--symbol", default=None)
    ss.set_defaults(func=cmd_sentiment_show)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    storage = Storage(cfg.db_path)
    return args.func(args, cfg, storage)


if __name__ == "__main__":
    sys.exit(main())
