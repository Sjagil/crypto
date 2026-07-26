from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_selection import RotationPortfolioPolicy
from research.residual_reversal import (
    DAILY_PERIODS_PER_YEAR,
    ResidualReversalParameters,
    backtest_residual_reversal,
    residual_reversal_parameter_set,
    residual_reversal_period_metrics,
)


def _frames(rows: int = 1_100) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    generator = np.random.default_rng(20260726)
    btc_returns = (
        0.0010
        + 0.002 * np.sin(np.arange(rows) / 80.0)
        + generator.normal(0.0, 0.008, rows)
    )
    returns_by_market: dict[str, np.ndarray] = {
        "BTC-EUR": btc_returns,
    }
    for offset, market in enumerate(
        ("ETH-EUR", "SOL-EUR", "LINK-EUR"),
        start=1,
    ):
        residual = generator.normal(0.0002, 0.006 + offset * 0.001, rows)
        for shock in range(260 + offset * 11, rows - 20, 95):
            residual[shock] -= 0.11 + offset * 0.01
            residual[shock + 1 : shock + 5] += (
                0.025 + offset * 0.002
            )
        returns_by_market[market] = (
            (0.85 + offset * 0.12) * btc_returns + residual
        )

    result: dict[str, pd.DataFrame] = {}
    for offset, (market, returns) in enumerate(
        returns_by_market.items()
    ):
        close = (100.0 + offset * 20.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0 + generator.normal(0.0, 0.0008, rows)
        )
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close,
                "volume": generator.lognormal(8.0, 0.20, rows),
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


def _parameters() -> ResidualReversalParameters:
    return ResidualReversalParameters(
        beta_lookback=30,
        residual_horizon=3,
        entry_zscore=-1.5,
    )


def test_residual_reversal_parameter_set_is_exact_and_unique() -> None:
    rows = residual_reversal_parameter_set()

    assert len(rows) == 8
    assert len({row.dna_hash for row in rows}) == 8
    assert {
        (row.beta_lookback, row.residual_horizon, row.entry_zscore)
        for row in rows
    } == {
        (beta, horizon, entry)
        for beta in (30, 60)
        for horizon in (3, 5)
        for entry in (-1.5, -2.0)
    }


def test_residual_reversal_is_active_next_open_bounded_and_orderless() -> None:
    result = backtest_residual_reversal(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.metrics["rebalance_count"] > 20
    assert result.signal_diagnostics["entry_signal_count"] > 10
    assert result.integrity["strictly_prior_beta_estimation"]
    assert result.integrity["strictly_prior_zscore_baseline"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["benchmark_never_traded"]
    assert result.integrity["orders_generated"] == 0
    assert result.executed_weights.to_numpy().min() >= 0.0
    assert result.executed_weights.sum(axis=1).max() <= 0.40 + 1e-12
    assert result.executed_weights.max(axis=1).max() <= 0.20 + 1e-12
    entries = result.decisions[
        result.decisions["entry_markets"].map(bool)
    ]
    assert not entries.empty
    assert bool(
        (
            (
                entries["executed_at"] - entries["decision_at"]
            ).dt.total_seconds()
            == 86_400.0
        ).all()
    )


def test_residual_reversal_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_residual_reversal(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_residual_reversal(
        frames,
        _parameters(),
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


def test_future_prices_cannot_change_prior_residual_reversal_weights() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[850]
    baseline = backtest_residual_reversal(
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
        multiplier = np.linspace(1.0, 5.0, int(future.sum()))
        frame.loc[future, ["open", "high", "low", "close"]] *= (
            multiplier[:, None]
        )
    revised = backtest_residual_reversal(
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


def test_residual_reversal_rejects_assets_outside_allowlist() -> None:
    frames = _frames()
    frames["ADA-EUR"] = frames["ETH-EUR"].copy()

    with pytest.raises(ValueError, match="outside fail-closed"):
        backtest_residual_reversal(
            frames,
            _parameters(),
            fee_rate=0.0025,
            slippage_bps=8.0,
            spread_bps=5.0,
            portfolio_policy=_policy(),
        )


def test_residual_reversal_period_metrics_use_daily_frequency() -> None:
    result = backtest_residual_reversal(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = residual_reversal_period_metrics(
        result.equity_curve,
        start="2021-01-01",
        end="2022-12-31",
    )

    assert len(returns) > 300
    assert metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == "DAILY_PORTFOLIO_RETURN"
