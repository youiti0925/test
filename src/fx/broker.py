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


@dataclass(frozen=True)
class ExecutionFill:
    """Everything we need to audit a real fill (spec §12 / §16).

    Captured at place_order time and stashed on the broker as
    `last_fill`. Persisted by `cmd_trade` into the trades table so the
    weekly review can compute slippage and execution-latency stats
    without depending on broker logs.

    `expected_entry_price` is the price the engine asked for. The diff
    against `actual_fill_price` is the slippage we paid.
    """

    symbol: str
    side: str                       # BUY | SELL
    expected_entry_price: float
    actual_fill_price: float
    size: float
    bid: float | None
    ask: float | None
    spread_pct: float | None
    broker_order_id: str | None
    order_request_time: datetime
    order_response_time: datetime
    fill_time: datetime

    @property
    def slippage(self) -> float:
        """Signed slippage in price units (positive = paid more than expected)."""
        sign = 1 if self.side == "BUY" else -1
        return sign * (self.actual_fill_price - self.expected_entry_price)

    @property
    def slippage_pct(self) -> float | None:
        if self.expected_entry_price <= 0:
            return None
        return 100.0 * self.slippage / self.expected_entry_price

    @property
    def execution_latency_ms(self) -> float:
        return max(
            0.0,
            (self.order_response_time - self.order_request_time).total_seconds() * 1000.0,
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "expected_entry_price": self.expected_entry_price,
            "actual_fill_price": self.actual_fill_price,
            "size": self.size,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": self.spread_pct,
            "broker_order_id": self.broker_order_id,
            "order_request_time": self.order_request_time.isoformat(),
            "order_response_time": self.order_response_time.isoformat(),
            "fill_time": self.fill_time.isoformat(),
            "slippage": self.slippage,
            "slippage_pct": self.slippage_pct,
            "execution_latency_ms": self.execution_latency_ms,
        }


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
    """Broker interface.

    Concrete implementations should also expose a `last_fill` attribute
    (or override `last_execution_fill()`) so callers can persist the
    full execution audit (spec §12 / §16). The base class returns None
    by default.
    """

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

    def last_execution_fill(self) -> ExecutionFill | None:
        """Return the ExecutionFill from the most recent place_order, if any."""
        return getattr(self, "last_fill", None)


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
    last_fill: ExecutionFill | None = None

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
        now = datetime.now(timezone.utc)
        pos = OpenPosition(
            id=self._next_id,
            symbol=symbol,
            side=side,
            entry=price,
            size=size,
            opened_at=now,
            stop=stop,
            take_profit=take_profit,
        )
        self._positions[pos.id] = pos
        self._next_id += 1
        # Synthetic fill: zero spread, zero slippage, zero latency.
        # Production callers MUST distinguish paper from real fills via
        # `broker_order_id` (None here vs. an OANDA-issued id) when
        # interpreting trade analytics.
        self.last_fill = ExecutionFill(
            symbol=symbol, side=side,
            expected_entry_price=price, actual_fill_price=price,
            size=size, bid=price, ask=price, spread_pct=0.0,
            broker_order_id=None,
            order_request_time=now, order_response_time=now, fill_time=now,
        )
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
