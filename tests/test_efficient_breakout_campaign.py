from __future__ import annotations

import numpy as np
import pandas as pd

from research.efficient_breakout_campaign import stage0_breakout_screen
from research.portfolio_breakout import EfficientAtrRiskBreakoutParameters


def _frames(rows: int = 1_500) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2022-01-01", periods=rows, freq="4h", tz="UTC")

    def frame(drift: float, phase: float) -> pd.DataFrame:
        changes = drift + 0.006 * np.sin(np.arange(rows) / 23.0 + phase)
        close = 100.0 * np.exp(np.cumsum(changes))
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000.0,
            },
            index=index,
        )

    return {
        "BTC-EUR": frame(0.0005, 0.0),
        "ETH-EUR": frame(0.0007, 0.7),
        "SOL-EUR": frame(0.0004, 1.4),
        "LINK-EUR": frame(0.0006, 2.1),
    }


def test_stage0_is_deterministic_causal_costed_and_non_authoritative() -> None:
    parameters = EfficientAtrRiskBreakoutParameters(
        entry_lookback=120,
        exit_lookback=60,
        trend_ema_period=600,
        rebalance_days=3,
        rebalance_buffer=0.05,
    )
    free = stage0_breakout_screen(
        _frames(),
        parameters,
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    costly = stage0_breakout_screen(
        _frames(),
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    repeated = stage0_breakout_screen(
        _frames(),
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )

    assert costly == repeated
    assert costly["net_return"] < free["net_return"]
    assert costly["result_type"] == "CAUSAL_APPROXIMATION_NOT_PROMOTION_EVIDENCE"
    assert costly["no_lookahead"]
    assert costly["decision_at_close_execution_next_open"]
    assert costly["orders_generated"] == 0
