"""Broker abstraction + in-memory paper trading.

Real brokers (OANDA demo, bitFlyer, etc.) should implement the same
interface — keeps the strategy code broker-agnostic and lets us test the
full pipeline without any network or live order risk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Quote:
    """Current Bid/Ask snapshot from a real broker.

    Used by the Risk Gate: a missing or stale quote means we can't verify
    the spread, and the gate refuses the live trade. PaperBroker may
    return a synthetic Quote with zero spread for offline tests.
    """

    symbol: str
    bid: float
    ask: float
    ts: datetime

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as a percent of mid price (0.01 = 1%)."""
        m = self.mid
        if m <= 0:
            return float("inf")
        return 100.0 * self.spread / m


@dataclass
class OpenPosition:
    id: int
    symbol: str
    side: str  # BUY | SELL
    entry: float
    size: float
    opened_at: datetime
    stop: float | None = None
    take_profit: float | None = None

    def unrealized_pnl(self, current_price: float) -> float:
        direction = 1 if self.side == "BUY" else -1
        return (current_price - self.entry) * self.size * direction


@dataclass
class ClosedPosition:
    id: int
    symbol: str
    side: str
    entry: float
    exit: float
    size: float
    opened_at: datetime
    closed_at: datetime
    pnl: float


class Broker(ABC):
    @abstractmethod
    def balance(self) -> float: ...

    @abstractmethod
    def open_positions(self) -> list[OpenPosition]: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop: float | None = None,
        take_profit: float | None = None,
    ) -> OpenPosition: ...

    @abstractmethod
    def close_position(self, position_id: int, price: float) -> ClosedPosition: ...

    @abstractmethod
    def mark_to_market(self, price_by_symbol: dict[str, float]) -> None:
        """Check stops/TPs against current prices and close any that trigger."""

    def quote(self, symbol: str) -> Quote | None:
        """Return current Bid/Ask for `symbol`, or None if unavailable.

        Default implementation: not supported (returns None). Real brokers
        override this. Callers that need spread enforcement (live entry)
        must treat None as a hard refusal.
        """
        return None


@dataclass
class PaperBroker(Broker):
    """In-memory broker for dry runs and end-to-end tests.

    Starts with `initial_cash`, tracks open and closed positions, auto-closes
    on stop/TP hits. No slippage, no spread, no commissions — a baseline
    for sanity checks, not a realistic fill simulator.
    """

    initial_cash: float = 10_000.0
    _cash: float = field(init=False)
    _positions: dict[int, OpenPosition] = field(default_factory=dict)
    _closed: list[ClosedPosition] = field(default_factory=list)
    _next_id: int = 1

    def __post_init__(self) -> None:
        self._cash = self.initial_cash

    def balance(self) -> float:
        return self._cash

    def open_positions(self) -> list[OpenPosition]:
        return list(self._positions.values())

    def closed_positions(self) -> list[ClosedPosition]:
        return list(self._closed)

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop: float | None = None,
        take_profit: float | None = None,
    ) -> OpenPosition:
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if size <= 0:
            raise ValueError("size must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        pos = OpenPosition(
            id=self._next_id,
            symbol=symbol,
            side=side,
            entry=price,
            size=size,
            opened_at=datetime.now(timezone.utc),
            stop=stop,
            take_profit=take_profit,
        )
        self._positions[pos.id] = pos
        self._next_id += 1
        return pos

    def close_position(self, position_id: int, price: float) -> ClosedPosition:
        pos = self._positions.pop(position_id, None)
        if pos is None:
            raise KeyError(f"No open position {position_id}")
        pnl = pos.unrealized_pnl(price)
        closed = ClosedPosition(
            id=pos.id,
            symbol=pos.symbol,
            side=pos.side,
            entry=pos.entry,
            exit=price,
            size=pos.size,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            pnl=pnl,
        )
        self._closed.append(closed)
        self._cash += pnl
        return closed

    def quote(self, symbol: str) -> Quote | None:
        """Synthetic zero-spread quote based on the last submitted price.

        Returned only so backtests / dry-runs can pass *something* through
        the Risk Gate uniformly. Production callers must NOT use a paper
        quote to satisfy `require_spread`.
        """
        for pos in self._positions.values():
            if pos.symbol == symbol:
                return Quote(symbol=symbol, bid=pos.entry, ask=pos.entry,
                             ts=datetime.now(timezone.utc))
        return None

    def mark_to_market(self, price_by_symbol: dict[str, float]) -> None:
        for pos_id, pos in list(self._positions.items()):
            price = price_by_symbol.get(pos.symbol)
            if price is None:
                continue
            should_close = False
            if pos.side == "BUY":
                if pos.stop is not None and price <= pos.stop:
                    should_close = True
                elif pos.take_profit is not None and price >= pos.take_profit:
                    should_close = True
            else:  # SELL
                if pos.stop is not None and price >= pos.stop:
                    should_close = True
                elif pos.take_profit is not None and price <= pos.take_profit:
                    should_close = True
            if should_close:
                self.close_position(pos_id, price)
