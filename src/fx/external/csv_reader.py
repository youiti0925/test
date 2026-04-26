"""CSV → ExternalSignal reader (spec §9).

The least-bad way to ingest a vendor's signal history without committing
to any specific API: ask the user to export to CSV and feed it here.
Required columns: symbol, timestamp. Everything else is optional and
mapped onto ExternalSignal fields when present.

Supported aliases (case-insensitive)
-----------------------------------
  symbol    : symbol, ticker, instrument
  timestamp : timestamp, time, datetime, date, ts
  action    : action, signal, direction, side
  confidence: confidence, score, conviction
  pattern   : pattern, setup
  event_risk: event_risk, event_level, risk
  sentiment : sentiment, sentiment_score
  entry_price : entry, entry_price, price
  stop_loss   : stop, stop_loss, sl
  take_profit : tp, take_profit, target

Anything else lands in `raw_payload` so nothing is silently dropped.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .comparison import ExternalSignal


_ALIASES = {
    "symbol": ("symbol", "ticker", "instrument"),
    "timestamp": ("timestamp", "time", "datetime", "date", "ts"),
    "action": ("action", "signal", "direction", "side"),
    "confidence": ("confidence", "score", "conviction"),
    "pattern": ("pattern", "setup"),
    "event_risk": ("event_risk", "event_level", "risk"),
    "sentiment_score": ("sentiment_score", "sentiment"),
    "entry_price": ("entry_price", "entry", "price"),
    "stop_loss": ("stop_loss", "stop", "sl"),
    "take_profit": ("take_profit", "tp", "target"),
    "timeframe": ("timeframe", "interval", "tf"),
}


def _resolve(row: dict, field_key: str) -> str | None:
    aliases = _ALIASES[field_key]
    for a in aliases:
        for k in row:
            if k.lower().strip() == a:
                return row[k]
    return None


def _parse_dt(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def read_external_csv(path: Path, *, source: str) -> list[ExternalSignal]:
    """Read a CSV and return list[ExternalSignal] tagged with `source`.

    Rows missing both `symbol` and `timestamp` are silently skipped —
    they're typically vendor-CSV header artifacts. Rows with a bad
    timestamp are skipped with no error to keep batch ingest robust.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    out: list[ExternalSignal] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = _resolve(row, "symbol")
            ts_raw = _resolve(row, "timestamp")
            ts = _parse_dt(ts_raw)
            if not sym or ts is None:
                continue
            tf = _resolve(row, "timeframe") or ""
            action_raw = _resolve(row, "action")
            action = action_raw.upper() if isinstance(action_raw, str) else None
            out.append(ExternalSignal(
                source=source,
                symbol=sym,
                timeframe=tf,
                timestamp=ts,
                action=action,
                confidence=_coerce_float(_resolve(row, "confidence")),
                pattern=_resolve(row, "pattern"),
                event_risk=_resolve(row, "event_risk"),
                sentiment_score=_coerce_float(_resolve(row, "sentiment_score")),
                entry_price=_coerce_float(_resolve(row, "entry_price")),
                stop_loss=_coerce_float(_resolve(row, "stop_loss")),
                take_profit=_coerce_float(_resolve(row, "take_profit")),
                raw_payload=dict(row),
            ))
    return out


__all__ = ["read_external_csv"]
