from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_selection import RotationPortfolioPolicy
from research.sentiment_recovery import (
    DAILY_PERIODS_PER_YEAR,
    SentimentRecoveryParameters,
    backtest_sentiment_recovery,
    sentiment_recovery_parameter_set,
    sentiment_recovery_period_metrics,
)


def _frames(rows: int = 3_000) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(("BTC-EUR", "ETH-EUR")):
        generator = np.random.default_rng(800 + offset)
        returns = generator.normal(0.0007, 0.018, rows)
        close = (100.0 + offset * 20.0) * np.exp(
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


def _sentiment(rows: int = 3_000) -> pd.DataFrame:
    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )
    values = np.full(rows, 50.0)
    pattern = np.asarray(
        [18, 15, 17, 21, 27, 35, 45, 55, 65, 76, 80, 60],
        dtype=float,
    )
    for start in range(240, rows - len(pattern), 120):
        values[start : start + len(pattern)] = pattern
    observed_at = pd.Timestamp("2030-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "provider": "alternative_me",
            "available_at": index,
            "observed_at": observed_at,
            "point_in_time_status": "SOURCE_DAILY_TIMESTAMP",
            "fear_greed": values,
        }
    )


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=("BTC-EUR", "ETH-EUR"),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )


def _parameters() -> SentimentRecoveryParameters:
    return SentimentRecoveryParameters(
        fear_threshold=20,
        recovery_delta=5,
        trend_ema_period=100,
    )


def test_sentiment_parameter_family_is_fixed_unique_and_complete() -> None:
    rows = sentiment_recovery_parameter_set()

    assert len(rows) == 8
    assert len({row.dna_hash for row in rows}) == 8
    assert {row.fear_threshold for row in rows} == {20, 25}
    assert {row.recovery_delta for row in rows} == {5, 10}
    assert {row.trend_ema_period for row in rows} == {100, 200}


def test_sentiment_backtest_is_causal_bounded_and_orderless() -> None:
    result = backtest_sentiment_recovery(
        _frames(),
        _sentiment(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.signal_diagnostics["recovery_event_count"] > 10
    assert result.signal_diagnostics["entry_signal_count"] > 0
    assert result.integrity["sentiment_source_timestamped"]
    assert result.integrity["sentiment_backward_only_alignment"]
    assert not result.integrity["sentiment_backfill_before_inception"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["maximum_positions_respected"]
    assert result.integrity["orders_generated"] == 0
    assert result.metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR


def test_sentiment_cost_stress_is_monotonic() -> None:
    frames = _frames()
    sentiment = _sentiment()
    normal = backtest_sentiment_recovery(
        frames,
        sentiment,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_sentiment_recovery(
        frames,
        sentiment,
        _parameters(),
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )

    assert stressed.metrics["net_return"] <= normal.metrics[
        "net_return"
    ]
    assert stressed.executed_weights.equals(
        normal.executed_weights
    )


def test_future_sentiment_and_prices_do_not_change_prior_weights() -> None:
    frames = _frames()
    sentiment = _sentiment()
    cutoff = frames["BTC-EUR"].index[2_200]
    baseline = backtest_sentiment_recovery(
        frames,
        sentiment,
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
        frame.loc[future, ["open", "high", "low", "close"]] *= 4.0
    revised_sentiment = sentiment.copy()
    revised_sentiment.loc[
        revised_sentiment["available_at"] > cutoff,
        "fear_greed",
    ] = 1.0
    revised = backtest_sentiment_recovery(
        changed,
        revised_sentiment,
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


def test_sentiment_period_metrics_use_daily_frequency() -> None:
    result = backtest_sentiment_recovery(
        _frames(),
        _sentiment(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = sentiment_recovery_period_metrics(
        result.equity_curve,
        start=result.equity_curve.index[200],
        end=result.equity_curve.index[-1],
    )

    assert len(returns) > 1_000
    assert metrics["periods_per_year"] == DAILY_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == "DAILY_PORTFOLIO_RETURN"


def test_exact_duplicate_sentiment_snapshots_are_deterministic() -> None:
    frames = _frames()
    sentiment = _sentiment()
    duplicate = sentiment.iloc[[-1]].copy()
    duplicate["observed_at"] = pd.Timestamp(
        "2031-01-01",
        tz="UTC",
    )
    duplicated = pd.concat([sentiment, duplicate], ignore_index=True)

    baseline = backtest_sentiment_recovery(
        frames,
        sentiment,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    result = backtest_sentiment_recovery(
        frames,
        duplicated,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        result.executed_weights,
        baseline.executed_weights,
    )
    assert (
        result.signal_diagnostics[
            "sentiment_metadata"
        ]["exact_duplicate_snapshots_removed"]
        == 1
    )


def test_conflicting_duplicate_sentiment_snapshot_fails_closed() -> None:
    sentiment = _sentiment()
    duplicate = sentiment.iloc[[-1]].copy()
    duplicate["fear_greed"] = 99.0
    conflicting = pd.concat(
        [sentiment, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="conflicting duplicate availability",
    ):
        backtest_sentiment_recovery(
            _frames(),
            conflicting,
            _parameters(),
            fee_rate=0.0025,
            slippage_bps=8.0,
            spread_bps=5.0,
            portfolio_policy=_policy(),
        )
