from __future__ import annotations

import numpy as np
import pandas as pd

from research.multi_horizon_trend import (
    DAILY_PERIODS_PER_YEAR,
    MultiHorizonTrendParameters,
    backtest_multi_horizon_trend,
    multi_horizon_trend_parameter_set,
    multi_horizon_trend_period_metrics,
)
from research.portfolio_selection import RotationPortfolioPolicy

MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")


def _frames(rows: int = 3_000) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    generator = np.random.default_rng(2_026)
    common = (
        0.0007
        + 0.003 * np.sin(np.arange(rows) / 110.0)
        + generator.normal(0.0, 0.012, rows)
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(MARKETS):
        returns = (
            (0.8 + 0.08 * offset) * common
            + generator.normal(0.0, 0.008 + 0.002 * offset, rows)
        )
        close = (100.0 + 30.0 * offset) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0 + generator.normal(0.0, 0.001, rows)
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
        allowed_markets=MARKETS,
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=241,
    )


def test_multi_horizon_trend_has_one_frozen_dna() -> None:
    rows = multi_horizon_trend_parameter_set()

    assert len(rows) == 1
    assert rows[0] == MultiHorizonTrendParameters()
    assert len(rows[0].dna_hash) == 64


def test_multi_horizon_trend_is_causal_bounded_and_orderless() -> None:
    result = backtest_multi_horizon_trend(
        _frames(),
        MultiHorizonTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.metrics["rebalance_count"] > 20
    assert result.metrics["average_exposure"] > 0.05
    assert result.integrity["allowed_markets_only"]
    assert result.integrity["closed_candles_only"]
    assert result.integrity["decision_at_close_execution_next_open"]
    assert result.integrity["point_in_time_history_gate"]
    assert result.integrity["structural_horizon_gate"]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity["maximum_position_exposure_respected"]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["maximum_positions_respected"]
    assert result.integrity["zero_yield_cash"]
    assert result.integrity["orders_generated"] == 0


def test_multi_horizon_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_multi_horizon_trend(
        frames,
        MultiHorizonTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_multi_horizon_trend(
        frames,
        MultiHorizonTrendParameters(),
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )

    assert stressed.metrics["net_return"] <= normal.metrics["net_return"]
    pd.testing.assert_frame_equal(
        stressed.executed_weights,
        normal.executed_weights,
    )


def test_future_prices_do_not_change_prior_multi_horizon_weights() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[2_200]
    baseline = backtest_multi_horizon_trend(
        frames,
        MultiHorizonTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed = {market: frame.copy() for market, frame in frames.items()}
    for frame in changed.values():
        future = frame.index > cutoff
        multiplier = np.exp(
            np.linspace(0.0, 2.0, int(future.sum()))
        )
        frame.loc[future, ["open", "high", "low", "close"]] *= (
            multiplier[:, None]
        )
    revised = backtest_multi_horizon_trend(
        changed,
        MultiHorizonTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        revised.executed_weights.loc[:cutoff],
    )


def test_structural_trend_gate_holds_cash_in_long_decline() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[1_900]
    for frame in frames.values():
        mask = frame.index >= cutoff
        declining = np.linspace(
            float(frame.loc[cutoff, "close"]),
            float(frame.loc[cutoff, "close"]) * 0.20,
            int(mask.sum()),
        )
        frame.loc[mask, "close"] = declining
        frame.loc[mask, "open"] = np.r_[
            declining[0],
            declining[:-1],
        ]
        frame.loc[mask, "high"] = frame.loc[
            mask, ["open", "close"]
        ].max(axis=1) * 1.01
        frame.loc[mask, "low"] = frame.loc[
            mask, ["open", "close"]
        ].min(axis=1) * 0.99
    result = backtest_multi_horizon_trend(
        frames,
        MultiHorizonTrendParameters(),
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        portfolio_policy=_policy(),
    )

    assert (
        result.executed_weights.iloc[-60:].sum(axis=1) <= 1e-12
    ).all()


def test_multi_horizon_period_metrics_use_daily_frequency() -> None:
    result = backtest_multi_horizon_trend(
        _frames(),
        MultiHorizonTrendParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = multi_horizon_trend_period_metrics(
        result.equity_curve,
        start="2020-01-01",
        end="2025-12-31",
    )

    assert len(returns) > 1_000
    assert metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == "DAILY_PORTFOLIO_RETURN"
