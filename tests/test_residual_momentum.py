from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio_selection import RotationPortfolioPolicy
from research.residual_momentum import (
    DAILY_PERIODS_PER_YEAR,
    ResidualMomentumParameters,
    backtest_residual_momentum,
    residual_momentum_parameter_set,
    residual_momentum_period_metrics,
)


def _frames(rows: int = 3_000) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    generator = np.random.default_rng(941)
    btc_returns = (
        0.0008
        + 0.006 * np.sin(np.arange(rows) / 90.0)
        + generator.normal(0.0, 0.015, rows)
    )
    returns_by_market = {
        "BTC-EUR": btc_returns,
        "ETH-EUR": (
            1.05 * btc_returns
            + 0.0012
            + 0.004 * np.sin(np.arange(rows) / 47.0)
            + generator.normal(0.0, 0.008, rows)
        ),
        "SOL-EUR": (
            1.25 * btc_returns
            + 0.0010
            + 0.006 * np.sin(np.arange(rows) / 61.0 + 1.2)
            + generator.normal(0.0, 0.011, rows)
        ),
        "LINK-EUR": (
            0.90 * btc_returns
            + 0.0009
            + 0.005 * np.sin(np.arange(rows) / 53.0 + 2.1)
            + generator.normal(0.0, 0.010, rows)
        ),
    }
    starts = {
        "BTC-EUR": 0,
        "ETH-EUR": 0,
        "SOL-EUR": 850,
        "LINK-EUR": 300,
    }
    result: dict[str, pd.DataFrame] = {}
    for offset, (market, returns) in enumerate(
        returns_by_market.items()
    ):
        start = starts[market]
        selected_index = index[start:]
        close = (100.0 + offset * 25.0) * np.exp(
            np.cumsum(returns[start:])
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0
            + generator.normal(0.0, 0.001, len(close))
        )
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.01,
                "low": np.minimum(open_, close) * 0.99,
                "close": close,
                "volume": generator.lognormal(
                    7.0,
                    0.25,
                    len(close),
                ),
            },
            index=selected_index,
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


def _parameters() -> ResidualMomentumParameters:
    return ResidualMomentumParameters(
        residual_lookback=20,
        beta_lookback=90,
        asset_ema_period=100,
    )


def test_residual_momentum_family_is_fixed_unique_and_complete() -> None:
    rows = residual_momentum_parameter_set()

    assert len(rows) == 8
    assert len({row.dna_hash for row in rows}) == 8
    assert {row.residual_lookback for row in rows} == {20, 60}
    assert {row.beta_lookback for row in rows} == {90, 180}
    assert {row.asset_ema_period for row in rows} == {100, 200}


def test_residual_momentum_is_causal_bounded_active_and_orderless() -> None:
    result = backtest_residual_momentum(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.metrics["rebalance_count"] > 20
    assert result.metrics["average_exposure"] > 0.10
    assert result.signal_diagnostics["btc_core_active_days"] > 100
    assert result.signal_diagnostics["satellite_active_days"] > 100
    assert result.integrity["rolling_beta_backward_only"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["point_in_time_history_gate"]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["maximum_positions_respected"]
    assert result.integrity["orders_generated"] == 0


def test_residual_momentum_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_residual_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_residual_momentum(
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
    pd.testing.assert_frame_equal(
        stressed.executed_weights,
        normal.executed_weights,
    )


def test_future_prices_do_not_change_prior_residual_weights() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[2_200]
    baseline = backtest_residual_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed = {
        market: frame.copy() for market, frame in frames.items()
    }
    for frame in changed.values():
        future = frame.index > cutoff
        frame.loc[future, ["open", "high", "low", "close"]] *= (
            np.linspace(1.0, 5.0, int(future.sum()))[:, None]
        )
    revised = backtest_residual_momentum(
        changed,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        revised.executed_weights.loc[:cutoff],
    )


def test_late_asset_cannot_be_selected_before_own_history_gate() -> None:
    frames = _frames()
    result = backtest_residual_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    sol_start = frames["SOL-EUR"].index[0]
    eligible_at = sol_start + pd.DateOffset(days=89)

    assert bool(
        (
            result.executed_weights.loc[
                result.executed_weights.index < eligible_at,
                "SOL-EUR",
            ]
            == 0.0
        ).all()
    )


def test_residual_period_metrics_use_daily_frequency() -> None:
    result = backtest_residual_momentum(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = residual_momentum_period_metrics(
        result.equity_curve,
        start="2020-01-01",
        end="2025-12-31",
    )

    assert len(returns) > 1_000
    assert metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == "DAILY_PORTFOLIO_RETURN"
