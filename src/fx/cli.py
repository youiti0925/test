"""Command-line entry point.

Usage examples:
  python -m src.fx.cli analyze --symbol USDJPY=X --interval 1h
  python -m src.fx.cli backtest --symbol USDJPY=X --interval 1h --period 180d
  python -m src.fx.cli review --limit 50
  python -m src.fx.cli watch --symbol USDJPY=X --interval 15m --every 900
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .analyst import Analyst
from .backtest import run_backtest
from .config import Config
from .data import fetch_ohlcv
from .indicators import build_snapshot, technical_signal
from .news import fetch_headlines
from .storage import Storage
from .strategy import combine


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def cmd_analyze(args, cfg: Config, storage: Storage) -> int:
    df = fetch_ohlcv(args.symbol, interval=args.interval, period=args.period)
    snap = build_snapshot(args.symbol, df)
    tech = technical_signal(snap)

    headlines = []
    if not args.no_news:
        headlines = fetch_headlines(args.symbol, limit=args.news_limit)

    llm_signal = None
    if cfg.anthropic_api_key and not args.no_llm:
        analyst = Analyst(model=cfg.model, effort=cfg.effort)
        llm_signal = analyst.analyze(snap.to_dict(), headlines=headlines)

    decision = combine(tech, llm_signal)
    analysis_id = storage.save_analysis(
        symbol=args.symbol,
        snapshot=snap.to_dict(),
        technical_signal=tech,
        final_action=decision.action,
        llm_action=llm_signal.action if llm_signal else None,
        llm_confidence=llm_signal.confidence if llm_signal else None,
        llm_reason=llm_signal.reason if llm_signal else None,
    )

    _print(
        {
            "analysis_id": analysis_id,
            "snapshot": snap.to_dict(),
            "technical_signal": tech,
            "headlines": [h.to_dict() for h in headlines],
            "llm": (
                {
                    "action": llm_signal.action,
                    "confidence": llm_signal.confidence,
                    "reason": llm_signal.reason,
                    "key_risks": llm_signal.key_risks,
                }
                if llm_signal
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
    payload = {"trades": trades, "recent_analyses": analyses}
    analyst = Analyst(model=cfg.model, effort="high")
    print(analyst.review(payload["trades"] or payload["recent_analyses"]))
    return 0


def cmd_watch(args, cfg: Config, storage: Storage) -> int:
    while True:
        try:
            cmd_analyze(args, cfg, storage)
        except Exception as e:  # noqa: BLE001
            print(f"[watch] error: {e}", file=sys.stderr)
        time.sleep(args.every)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fx", description="FX analysis toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Run a single analysis")
    a.add_argument("--symbol", default="USDJPY=X")
    a.add_argument("--interval", default="1h")
    a.add_argument("--period", default="60d")
    a.add_argument("--no-llm", action="store_true", help="Skip Claude call")
    a.add_argument("--no-news", action="store_true", help="Skip headline fetch")
    a.add_argument("--news-limit", type=int, default=5)
    a.set_defaults(func=cmd_analyze)

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
    w.add_argument("--symbol", default="USDJPY=X")
    w.add_argument("--interval", default="15m")
    w.add_argument("--period", default="5d")
    w.add_argument("--every", type=int, default=900, help="Seconds between analyses")
    w.add_argument("--no-llm", action="store_true")
    w.add_argument("--no-news", action="store_true")
    w.add_argument("--news-limit", type=int, default=5)
    w.set_defaults(func=cmd_watch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    storage = Storage(cfg.db_path)
    return args.func(args, cfg, storage)


if __name__ == "__main__":
    sys.exit(main())
