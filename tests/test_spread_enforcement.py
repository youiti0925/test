"""Tests for spread enforcement on live brokers (spec §12)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.fx.broker import Quote
from src.fx.risk_gate import RiskState, check_spread, evaluate as gate_evaluate


def _df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
         "close": [1.0] * n, "volume": [1] * n},
        index=pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_check_spread_none_not_required():
    """Backward-compat: missing spread is silent in research mode."""
    assert check_spread(None, require_spread=False) is None


def test_check_spread_none_required_blocks():
    """Live broker mode: missing spread is itself a block."""
    block = check_spread(None, require_spread=True)
    assert block is not None
    assert block.code == "spread_unavailable"


def test_check_spread_normal():
    assert check_spread(0.01, require_spread=True) is None


def test_check_spread_wide_blocks():
    block = check_spread(0.5, require_spread=True)
    assert block is not None
    assert block.code == "spread_abnormal"


def test_gate_require_spread_blocks_without_quote():
    state = RiskState(df=_df(), spread_pct=None, require_spread=True)
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "spread_unavailable"


def test_gate_require_spread_allows_with_quote():
    state = RiskState(df=_df(), spread_pct=0.01, require_spread=True)
    res = gate_evaluate(state)
    assert res.allow_trade is True


def test_quote_spread_pct_calculation():
    q = Quote(
        symbol="USDJPY=X", bid=150.000, ask=150.030,
        ts=datetime.now(timezone.utc),
    )
    # 0.030 / 150.015 * 100 ≈ 0.02%
    assert q.spread_pct == pytest.approx(0.02, abs=0.001)
    assert q.mid == pytest.approx(150.015, abs=0.001)


def test_quote_spread_pct_zero_mid_returns_inf():
    q = Quote(
        symbol="X", bid=0.0, ask=0.0, ts=datetime.now(timezone.utc),
    )
    assert q.spread_pct == float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
