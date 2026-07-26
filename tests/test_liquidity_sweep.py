from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.liquidity_sweep import (
    LiquiditySweepParameters,
    backtest_liquidity_sweep,
    liquidity_sweep_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2020-01-01",
        periods=360,
        freq="1D",
        tz="UTC",
    )
    frames: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(("BTC-EUR", "ETH-EUR")):
        close = 100.0 + offset * 20.0 + np.arange(len(index)) * 0.12
        open_ = close - 0.15
        high = close + 1.0
        low = close - 1.0
        volume = np.full(len(index), 1_000.0 + offset * 100.0)
        # Confirmed five-bar pivot low at t=240, knowable at t=242.
        low[240] = close[240] - 8.0
        # A later high-volume sweep recovers above that confirmed level.
        pivot = low[240]
        open_[250] = pivot + 2.0
        low[250] = pivot - 1.0
        close[250] = pivot + 3.0
        high[250] = close[250] + 1.0
        volume[250] = 3_000.0
        frames[market] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
    return frames


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=("BTC-EUR", "ETH-EUR"),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )


def test_liquidity_sweep_parameter_set_is_exact_and_unique() -> None:
    rows = liquidity_sweep_parameter_set()

    assert len(rows) == 8
    assert len({row.dna_hash for row in rows}) == 8
    assert {
        (row.fractal_side, row.minimum_relative_volume, row.maximum_holding_days)
        for row in rows
    } == {
        (side, volume, holding)
        for side in (2, 3)
        for volume in (1.0, 1.5)
        for holding in (10, 20)
    }


def test_confirmed_sweep_executes_next_open_and_respects_caps() -> None:
    parameters = LiquiditySweepParameters(
        fractal_side=2,
        minimum_relative_volume=1.5,
        maximum_holding_days=10,
    )
    result = backtest_liquidity_sweep(
        _frames(),
        parameters,
        fee_rate=0.001,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    signal_day = pd.Timestamp("2020-09-07", tz="UTC")
    execution_day = signal_day + pd.DateOffset(days=1)
    entries = result.decisions[
        result.decisions["entered_assets"].map(bool)
    ]

    assert not entries.empty
    assert signal_day in set(entries["decision_at"])
    assert execution_day in set(entries["executed_at"])
    assert result.integrity["confirmed_fractals_only"]
    assert result.integrity["decision_at_close_execution_next_open"]
    assert result.executed_weights.to_numpy().min() >= 0.0
    assert result.executed_weights.sum(axis=1).max() <= 0.40 + 1e-12
    assert result.executed_weights.max(axis=1).max() <= 0.20 + 1e-12


def test_cost_stress_is_monotonic_and_input_universe_fails_closed() -> None:
    parameters = liquidity_sweep_parameter_set()[0]
    frames = _frames()
    normal = backtest_liquidity_sweep(
        frames,
        parameters,
        fee_rate=0.001,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_liquidity_sweep(
        frames,
        parameters,
        fee_rate=0.002,
        slippage_bps=10.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )

    assert stressed.equity_curve.iloc[-1] <= normal.equity_curve.iloc[-1]

    rejected = dict(frames)
    rejected["ADA-EUR"] = frames["ETH-EUR"].copy()
    with pytest.raises(ValueError, match="outside fail-closed"):
        backtest_liquidity_sweep(
            rejected,
            parameters,
            fee_rate=0.001,
            slippage_bps=5.0,
            spread_bps=5.0,
            portfolio_policy=_policy(),
        )


def test_future_ohlcv_cannot_change_prior_liquidity_sweep_weights() -> None:
    frames = _frames()
    parameters = liquidity_sweep_parameter_set()[0]
    cutoff = frames["BTC-EUR"].index[300]
    baseline = backtest_liquidity_sweep(
        frames,
        parameters,
        fee_rate=0.001,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed = {
        market: frame.copy() for market, frame in frames.items()
    }
    for frame in changed.values():
        future = frame.index > cutoff
        multiplier = np.linspace(1.0, 4.0, int(future.sum()))
        frame.loc[future, ["open", "high", "low", "close"]] *= (
            multiplier[:, None]
        )
        frame.loc[future, "volume"] *= multiplier
    revised = backtest_liquidity_sweep(
        changed,
        parameters,
        fee_rate=0.001,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        revised.executed_weights.loc[:cutoff],
    )
