from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio_selection import RotationPortfolioPolicy
from research.volatility_contraction import (
    VolatilityContractionParameters,
    backtest_volatility_contraction,
    volatility_contraction_parameter_set,
)


def _frames(rows: int = 900) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        generator = np.random.default_rng(100 + offset)
        returns = generator.normal(0.0007, 0.012, rows)
        for start in (330, 500, 680):
            returns[start : start + 45] = generator.normal(
                0.0001,
                0.0015,
                45,
            )
            returns[start + 45 : start + 60] = (
                generator.normal(0.006, 0.005, 15)
            )
        close = (100.0 + offset * 10.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0
            + generator.normal(0.0, 0.001, rows)
        )
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.003,
                "low": np.minimum(open_, close) * 0.997,
                "close": close,
                "volume": np.full(rows, 1_000.0),
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
        minimum_history_observations=90,
    )


def _parameters() -> VolatilityContractionParameters:
    return VolatilityContractionParameters(
        volatility_lookback=20,
        contraction_quantile=0.30,
        entry_lookback=20,
        exit_lookback=10,
        target_annualized_volatility=0.15,
    )


def test_parameter_family_is_fixed_unique_and_complete() -> None:
    rows = volatility_contraction_parameter_set()

    assert len(rows) == 16
    assert len({row.dna_hash for row in rows}) == 16
    assert {row.volatility_lookback for row in rows} == {20, 40}
    assert {row.contraction_quantile for row in rows} == {
        0.20,
        0.30,
    }
    assert {
        (row.entry_lookback, row.exit_lookback)
        for row in rows
    } == {(20, 10), (55, 20)}
    assert {
        row.target_annualized_volatility for row in rows
    } == {0.10, 0.15}


def test_contraction_backtest_is_causal_bounded_and_orderless() -> None:
    result = backtest_volatility_contraction(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.signal_diagnostics["entry_signal_count"] > 0
    assert result.integrity[
        "strictly_prior_contraction_distribution"
    ]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["orders_generated"] == 0
    assert (
        result.metrics["maximum_realized_exposure"]
        <= 0.40 + 1e-12
    )
    assert (
        result.metrics["maximum_position_exposure_observed"]
        <= 0.20 + 1e-12
    )


def test_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_volatility_contraction(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_volatility_contraction(
        frames,
        _parameters(),
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )

    assert stressed.metrics["net_return"] <= normal.metrics[
        "net_return"
    ]
    assert stressed.cost_breakdown[
        "net_ending_equity"
    ] <= normal.cost_breakdown["net_ending_equity"]
    assert stressed.executed_weights.equals(
        normal.executed_weights
    )


def test_future_prices_do_not_change_prior_decisions() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[700]
    baseline = backtest_volatility_contraction(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed = {
        market: frame.copy()
        for market, frame in frames.items()
    }
    for frame in changed.values():
        future = frame.index > cutoff
        frame.loc[future, ["open", "high", "low", "close"]] *= 4.0
    revised = backtest_volatility_contraction(
        changed,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    baseline_prior = baseline.executed_weights.loc[:cutoff]
    revised_prior = revised.executed_weights.loc[:cutoff]

    pd.testing.assert_frame_equal(
        baseline_prior,
        revised_prior,
    )
