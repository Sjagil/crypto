from __future__ import annotations

import numpy as np
import pandas as pd

from research.dual_asset_trend import (
    DAILY_PERIODS_PER_YEAR,
    DualAssetTrendParameters,
    backtest_dual_asset_trend,
    dual_asset_trend_parameter_set,
    dual_asset_trend_period_metrics,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 3_000) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    generator = np.random.default_rng(1_147)
    cycle = 0.004 * np.sin(np.arange(rows) / 95.0)
    btc_returns = (
        0.0008
        + cycle
        + generator.normal(0.0, 0.016, rows)
    )
    eth_returns = (
        0.0010
        + 1.05 * cycle
        + 0.75 * (btc_returns - 0.0008 - cycle)
        + generator.normal(0.0, 0.012, rows)
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, (market, returns) in enumerate(
        {
            "BTC-EUR": btc_returns,
            "ETH-EUR": eth_returns,
        }.items()
    ):
        close = (100.0 + offset * 40.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0
            + generator.normal(0.0, 0.001, rows)
        )
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.01,
                "low": np.minimum(open_, close) * 0.99,
                "close": close,
                "volume": generator.lognormal(7.0, 0.2, rows),
            },
            index=index,
        )
    return result


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=("BTC-EUR", "ETH-EUR"),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )


def test_dual_asset_trend_has_exactly_one_frozen_dna() -> None:
    rows = dual_asset_trend_parameter_set()

    assert len(rows) == 1
    assert rows[0] == DualAssetTrendParameters()
    assert len(rows[0].dna_hash) == 64


def test_dual_asset_trend_is_causal_bounded_active_and_orderless() -> None:
    result = backtest_dual_asset_trend(
        _frames(),
        DualAssetTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.metrics["rebalance_count"] > 20
    assert result.metrics["average_exposure"] > 0.05
    assert result.risk_diagnostics[
        "predicted_volatility_observations"
    ] > 1_000
    assert result.risk_diagnostics[
        "predicted_volatility_maximum"
    ] <= 0.15 + 1e-12
    assert result.integrity["full_covariance_backward_only"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["daily_exit_entry_weekly"]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["maximum_positions_respected"]
    assert result.integrity["orders_generated"] == 0


def test_dual_asset_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_dual_asset_trend(
        frames,
        DualAssetTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_dual_asset_trend(
        frames,
        DualAssetTrendParameters(),
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


def test_future_prices_do_not_change_prior_dual_asset_weights() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[2_200]
    baseline = backtest_dual_asset_trend(
        frames,
        DualAssetTrendParameters(),
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
        multiplier = np.exp(
            np.linspace(0.0, 2.0, int(future.sum()))
        )
        frame.loc[future, ["open", "high", "low", "close"]] *= (
            multiplier[:, None]
        )
    revised = backtest_dual_asset_trend(
        changed,
        DualAssetTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        revised.executed_weights.loc[:cutoff],
    )


def test_covariance_target_reduces_weights_when_volatility_rises() -> None:
    frames = _frames()
    high_volatility = {
        market: frame.copy() for market, frame in frames.items()
    }
    cutoff = high_volatility["BTC-EUR"].index[2_000]
    generator = np.random.default_rng(2_009)
    for frame in high_volatility.values():
        mask = frame.index > cutoff
        shocks = np.exp(
            np.cumsum(generator.normal(0.0, 0.06, int(mask.sum())))
        )
        base = float(frame.loc[cutoff, "close"])
        frame.loc[mask, "close"] = base * shocks
        frame.loc[mask, "open"] = frame.loc[mask, "close"].shift(
            1,
            fill_value=base,
        )
        frame.loc[mask, "high"] = frame.loc[
            mask, ["open", "close"]
        ].max(axis=1) * 1.01
        frame.loc[mask, "low"] = frame.loc[
            mask, ["open", "close"]
        ].min(axis=1) * 0.99
    result = backtest_dual_asset_trend(
        high_volatility,
        DualAssetTrendParameters(),
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        portfolio_policy=_policy(),
    )
    exposure = result.executed_weights.sum(axis=1)
    earlier = exposure.loc[
        cutoff - pd.DateOffset(days=300) : cutoff
    ]
    later = exposure.loc[cutoff + pd.DateOffset(days=90) :]

    assert float(later.mean()) < float(earlier.mean())


def test_dual_asset_period_metrics_use_daily_frequency() -> None:
    result = backtest_dual_asset_trend(
        _frames(),
        DualAssetTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = dual_asset_trend_period_metrics(
        result.equity_curve,
        start="2020-01-01",
        end="2025-12-31",
    )

    assert len(returns) > 1_000
    assert metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == "DAILY_PORTFOLIO_RETURN"
