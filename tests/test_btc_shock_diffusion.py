from __future__ import annotations

import numpy as np
import pandas as pd

from research.btc_shock_diffusion import (
    BTCShockDiffusionParameters,
    backtest_btc_shock_diffusion,
    btc_shock_diffusion_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 950) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    generator = np.random.default_rng(827)
    btc_returns = generator.normal(0.0008, 0.007, rows)
    shock_positions = np.arange(260, rows - 2, 47)
    btc_returns[shock_positions] += 0.055
    result: dict[str, pd.DataFrame] = {}
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    for offset, market in enumerate(markets):
        if market == "BTC-EUR":
            returns = btc_returns.copy()
        else:
            beta = 0.75 + offset * 0.10
            returns = (
                beta * btc_returns
                + generator.normal(0.0002, 0.0025, rows)
            )
            returns[shock_positions] = 0.003
            returns[shock_positions + 1] += 0.045 + offset * 0.002
        close = (100.0 + offset * 10.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]]
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.004,
                "low": np.minimum(open_, close) * 0.996,
                "close": close,
                "volume": np.full(rows, 10_000.0),
            },
            index=index,
        )
    return result


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=(
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=200,
    )


def test_shock_diffusion_family_is_exactly_four_unique_dna() -> None:
    rows = btc_shock_diffusion_parameter_set()
    assert len(rows) == 4
    assert len({row.dna_hash for row in rows}) == 4
    assert {
        (row.shock_lookback, row.maximum_holding_days)
        for row in rows
    } == {(1, 3), (1, 5), (3, 3), (3, 5)}


def test_shock_diffusion_is_causal_bounded_and_orderless() -> None:
    result = backtest_btc_shock_diffusion(
        _frames(),
        BTCShockDiffusionParameters(
            shock_lookback=1,
            maximum_holding_days=3,
        ),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    assert result.signal_diagnostics["entry_signal_count"] > 0
    assert result.integrity["strictly_prior_beta_estimation"]
    assert result.integrity["strictly_prior_shock_baseline"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity[
        "btc_used_as_regime_and_information_only"
    ]
    assert result.integrity["orders_generated"] == 0
    assert result.metrics["maximum_realized_exposure"] <= 0.40


def test_shock_diffusion_cost_stress_is_monotonic() -> None:
    frames = _frames()
    parameters = BTCShockDiffusionParameters(
        shock_lookback=3,
        maximum_holding_days=5,
    )
    normal = backtest_btc_shock_diffusion(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_btc_shock_diffusion(
        frames,
        parameters,
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )
    assert stressed.metrics["net_return"] <= normal.metrics[
        "net_return"
    ]
    pd.testing.assert_frame_equal(
        normal.executed_weights,
        stressed.executed_weights,
    )


def test_future_prices_do_not_change_prior_shock_allocations() -> None:
    frames = _frames()
    parameters = BTCShockDiffusionParameters(
        shock_lookback=1,
        maximum_holding_days=5,
    )
    cutoff = frames["BTC-EUR"].index[700]
    baseline = backtest_btc_shock_diffusion(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    revised = {
        market: frame.copy() for market, frame in frames.items()
    }
    for frame in revised.values():
        mask = frame.index > cutoff
        multiplier = np.linspace(1.0, 2.0, int(mask.sum()))
        for column in ("open", "high", "low", "close"):
            frame.loc[mask, column] *= multiplier
    changed = backtest_btc_shock_diffusion(
        revised,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        changed.executed_weights.loc[:cutoff],
    )
