"""OANDA demo broker scaffold.

**SAFETY FIRST.** This module is designed so it CANNOT place live-money
orders without three explicit conditions being met simultaneously:
  1. env var OANDA_ENV == 'practice' (hardcoded check, no override)
  2. env var OANDA_ALLOW_LIVE_ORDERS == 'yes' (explicit opt-in per session)
  3. The `OANDABroker` constructor is called with `confirm_demo=True`

Any other combination raises immediately. Live (real-money) accounts are
completely blocked. If you later want real-money trading, fork this and
accept the responsibility yourself — the author of this scaffold declines
to make that easy.

The `oandapyV20` library is NOT imported at module load time; it's imported
lazily inside the constructor so `pytest` and offline development never
require it to be installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .broker import Broker, ClosedPosition, ExecutionFill, OpenPosition, Quote


class LiveTradingBlocked(RuntimeError):
    pass


@dataclass
class OANDAConfig:
    api_key: str
    account_id: str
    env: str  # must be "practice"

    @classmethod
    def from_env(cls) -> "OANDAConfig":
        key = os.environ.get("OANDA_API_KEY")
        acct = os.environ.get("OANDA_ACCOUNT_ID")
        env = os.environ.get("OANDA_ENV", "practice")
        if not key or not acct:
            raise ValueError(
                "Set OANDA_API_KEY and OANDA_ACCOUNT_ID before using OANDABroker"
            )
        if env != "practice":
            raise LiveTradingBlocked(
                f"OANDA_ENV must be 'practice' for this scaffold; got {env!r}. "
                "Live accounts are blocked by design."
            )
        return cls(api_key=key, account_id=acct, env=env)


class OANDABroker(Broker):
    """Thin wrapper around oandapyV20 — **practice account only**.

    This is a scaffold: the method bodies below call the v20 API, but you
    should review and test each one against your own practice account
    before trusting them. Treat as a starting point, not a finished
    integration.
    """

    def __init__(self, config: OANDAConfig, confirm_demo: bool = False) -> None:
        if not confirm_demo:
            raise LiveTradingBlocked(
                "OANDABroker requires confirm_demo=True to instantiate. "
                "This is a deliberate friction point — remove only after "
                "you've read the source and verified the practice guard."
            )
        if os.environ.get("OANDA_ALLOW_LIVE_ORDERS") != "yes":
            raise LiveTradingBlocked(
                "Set OANDA_ALLOW_LIVE_ORDERS=yes to arm the broker. "
                "This is an explicit per-session opt-in."
            )
        if config.env != "practice":
            raise LiveTradingBlocked("config.env must be 'practice'")

        self.config = config

        # Lazy import so tests don't require the dependency.
        try:
            from oandapyV20 import API  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "oandapyV20 is required for OANDABroker. "
                "Install with: pip install oandapyV20"
            ) from e

        self._api = API(access_token=config.api_key, environment="practice")
        self._positions: dict[int, OpenPosition] = {}
        self._closed: list[ClosedPosition] = []
        self._next_id = 1
        # Most-recent ExecutionFill from place_order — written so cmd_trade
        # can persist slippage / latency / Bid-Ask without re-querying.
        self.last_fill: ExecutionFill | None = None

    # --- Broker interface ---

    def balance(self) -> float:  # pragma: no cover - network call
        from oandapyV20.endpoints.accounts import AccountSummary  # type: ignore

        r = AccountSummary(accountID=self.config.account_id)
        response = self._api.request(r)
        return float(response["account"]["balance"])

    def open_positions(self) -> list[OpenPosition]:
        return list(self._positions.values())

    def quote(self, symbol: str) -> Quote | None:  # pragma: no cover - network call
        """Fetch the latest Bid/Ask for `symbol` from OANDA.

        Returns None if the API call fails or the response is missing a
        bid/ask. The Decision Engine's spread gate treats None as a hard
        refusal when `require_spread=True` (live broker).
        """
        from oandapyV20.endpoints.pricing import PricingInfo  # type: ignore

        instrument = _to_oanda_instrument(symbol)
        try:
            r = PricingInfo(
                accountID=self.config.account_id,
                params={"instruments": instrument},
            )
            response = self._api.request(r)
        except Exception:  # noqa: BLE001
            return None
        prices = response.get("prices") or []
        if not prices:
            return None
        p = prices[0]
        bids = p.get("bids") or []
        asks = p.get("asks") or []
        if not bids or not asks:
            return None
        try:
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
        except (KeyError, ValueError, TypeError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        return Quote(
            symbol=symbol, bid=bid, ask=ask, ts=datetime.now(timezone.utc)
        )

    def place_order(  # pragma: no cover - network call
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop: float | None = None,
        take_profit: float | None = None,
    ) -> OpenPosition:
        from oandapyV20.endpoints.orders import OrderCreate  # type: ignore

        # Spec §12: live brokers must observe Bid/Ask before placing an
        # order — if we cannot read the spread we refuse to fire blindly.
        q = self.quote(symbol)
        if q is None:
            raise LiveTradingBlocked(
                f"No Bid/Ask quote available for {symbol}; "
                "refusing to place a live order without observed spread."
            )

        units = int(size) if side == "BUY" else -int(size)
        order_body: dict = {
            "order": {
                "instrument": _to_oanda_instrument(symbol),
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        if stop is not None:
            order_body["order"]["stopLossOnFill"] = {"price": f"{stop:.5f}"}
        if take_profit is not None:
            order_body["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}

        request_time = datetime.now(timezone.utc)
        r = OrderCreate(accountID=self.config.account_id, data=order_body)
        response = self._api.request(r) or {}
        response_time = datetime.now(timezone.utc)

        # Pull the actual fill price + broker order id from the response.
        # OANDA returns an `orderFillTransaction` (or `orderCancelTransaction`
        # if the request was rejected). We trust the response price over
        # our pre-trade quote — it's what we actually paid.
        fill_tx = (
            response.get("orderFillTransaction")
            or response.get("orderCreateTransaction")
            or {}
        )
        actual_entry = q.ask if side == "BUY" else q.bid
        broker_order_id = (
            fill_tx.get("id")
            or (response.get("orderCreateTransaction") or {}).get("id")
        )
        try:
            if "price" in fill_tx:
                actual_entry = float(fill_tx["price"])
        except (TypeError, ValueError):
            pass

        try:
            fill_time = (
                datetime.fromisoformat(fill_tx["time"].replace("Z", "+00:00"))
                if fill_tx.get("time") else response_time
            )
        except (TypeError, ValueError):
            fill_time = response_time

        pos = OpenPosition(
            id=self._next_id,
            symbol=symbol,
            side=side,
            entry=actual_entry,
            size=abs(size),
            opened_at=fill_time,
            stop=stop,
            take_profit=take_profit,
        )
        self._positions[pos.id] = pos
        self._next_id += 1

        self.last_fill = ExecutionFill(
            symbol=symbol, side=side,
            expected_entry_price=price,
            actual_fill_price=actual_entry,
            size=abs(size),
            bid=q.bid, ask=q.ask, spread_pct=q.spread_pct,
            broker_order_id=str(broker_order_id) if broker_order_id else None,
            order_request_time=request_time,
            order_response_time=response_time,
            fill_time=fill_time,
        )
        return pos

    def close_position(  # pragma: no cover - network call
        self, position_id: int, price: float
    ) -> ClosedPosition:
        from oandapyV20.endpoints.positions import PositionClose  # type: ignore

        pos = self._positions.pop(position_id, None)
        if pos is None:
            raise KeyError(f"No open position {position_id}")
        instrument = _to_oanda_instrument(pos.symbol)
        body = (
            {"longUnits": "ALL"} if pos.side == "BUY" else {"shortUnits": "ALL"}
        )
        r = PositionClose(
            accountID=self.config.account_id, instrument=instrument, data=body
        )
        self._api.request(r)

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
        return closed

    def mark_to_market(self, price_by_symbol: dict[str, float]) -> None:
        """OANDA enforces stops server-side; nothing to do client-side."""
        return None


def _to_oanda_instrument(symbol: str) -> str:
    """Convert yfinance-style symbols to OANDA's CCY_CCY format."""
    s = symbol.upper().replace("=X", "")
    if "_" in s:
        return s
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    if "-" in s:
        return s.replace("-", "_")
    return s
