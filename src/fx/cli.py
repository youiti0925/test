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
from .broker import Broker, PaperBroker
from .calendar import load_events, seed_defaults, upcoming_for_symbol
from .config import Config
from .correlation import build_correlation_snapshot, related_for
from .data import fetch_ohlcv
from .indicators import build_snapshot, technical_signal
from .news import fetch_headlines
from .postmortem import Postmortem
from .prediction import evaluate_prediction, slice_bars_after
from .risk import atr, plan_trade
from .storage import Storage
from .strategy import combine

EVENTS_PATH = Path("data/events.json")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _gather_inputs(args, cfg: Config, storage: Storage):
    """Common input fetch for analyze / trade / watch."""
    df = fetch_ohlcv(args.symbol, interval=args.interval, period=args.period)
    snap = build_snapshot(args.symbol, df)
    tech = technical_signal(snap)

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
    if not args.no_events and EVENTS_PATH.exists():
        all_events = load_events(EVENTS_PATH)
        events = upcoming_for_symbol(
            all_events, args.symbol, within_hours=args.event_window_hours
        )

    past_lessons = []
    if not args.no_lessons:
        past_lessons = storage.relevant_postmortems(
            args.symbol, limit=args.lesson_limit
        )

    llm_signal = None
    if cfg.anthropic_api_key and not args.no_llm:
        analyst = Analyst(model=cfg.model, effort=cfg.effort)
        llm_signal = analyst.analyze(
            snap.to_dict(),
            headlines=headlines,
            correlation=correlation,
            upcoming_events=events,
            past_postmortems=past_lessons,
        )

    decision = combine(tech, llm_signal)
    atr_value = float(atr(df).iloc[-1])
    return {
        "df": df,
        "snapshot": snap,
        "technical": tech,
        "headlines": headlines,
        "correlation": correlation,
        "events": events,
        "past_lessons": past_lessons,
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
    """
    ctx = _gather_inputs(args, cfg, storage)
    snap, decision, atr_value = ctx["snapshot"], ctx["decision"], ctx["atr"]

    if decision.action == "HOLD":
        _print({"action": "HOLD", "reason": decision.reason})
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

    broker: Broker = _build_broker(args)

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
        pm_id = storage.save_postmortem(
            prediction_id=p["id"],
            root_cause=result.root_cause,
            narrative=result.narrative,
            proposed_rule=result.proposed_rule,
            tags=",".join(result.tags),
        )
        written.append(
            {
                "postmortem_id": pm_id,
                "prediction_id": p["id"],
                "root_cause": result.root_cause,
                "proposed_rule": result.proposed_rule,
            }
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    storage = Storage(cfg.db_path)
    return args.func(args, cfg, storage)


if __name__ == "__main__":
    sys.exit(main())
