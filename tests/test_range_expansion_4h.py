from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio_selection import RotationPortfolioPolicy
from research.range_expansion_4h import (
    FOUR_HOUR_PERIODS_PER_YEAR,
    RangeExpansion4hParameters,
    backtest_range_expansion_4h,
    range_expansion_4h_parameter_set,
    range_expansion_4h_period_metrics,
    relabel_4h_forward_summary,
)


def _frames(rows: int = 3_000) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="4h",
        tz="UTC",
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        generator = np.random.default_rng(500 + offset)
        returns = generator.normal(0.0003, 0.006, rows)
        volume = generator.lognormal(7.0, 0.25, rows)
        for start in (800, 1_300, 1_900, 2_500):
            returns[start : start + 8] = generator.normal(
                0.012,
                0.003,
                8,
            )
            volume[start : start + 8] *= 3.0
        close = (100.0 + offset * 10.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0 + generator.normal(0.0, 0.0008, rows)
        )
        spread = np.maximum(
            np.abs(close / open_ - 1.0),
            np.abs(returns),
        ) + 0.003
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1.0 + spread),
                "low": np.minimum(open_, close) * (1.0 - spread),
                "close": close,
                "volume": volume,
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
        minimum_history_observations=180,
    )


def _parameters() -> RangeExpansion4hParameters:
    return RangeExpansion4hParameters(
        entry_lookback=30,
        exit_lookback=15,
        range_expansion_multiple=1.0,
        relative_volume_multiple=1.0,
        asset_ema_period=300,
    )


def test_parameter_family_is_fixed_unique_and_complete() -> None:
    rows = range_expansion_4h_parameter_set()

    assert len(rows) == 16
    assert len({row.dna_hash for row in rows}) == 16
    assert {
        (row.entry_lookback, row.exit_lookback) for row in rows
    } == {(30, 15), (60, 30)}
    assert {row.range_expansion_multiple for row in rows} == {
        1.0,
        1.5,
    }
    assert {row.relative_volume_multiple for row in rows} == {
        1.0,
        1.5,
    }
    assert {row.asset_ema_period for row in rows} == {300, 600}


def test_4h_backtest_is_causal_bounded_and_orderless() -> None:
    result = backtest_range_expansion_4h(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.signal_diagnostics["entry_signal_count"] > 0
    assert result.integrity["prior_channel_only"]
    assert result.integrity["strictly_prior_atr_baseline"]
    assert result.integrity["strictly_prior_volume_baseline"]
    assert result.integrity["annualization_frequency_correct"]
    assert result.integrity["common_calendar_intersection_only"]
    assert not result.integrity["missing_bars_imputed"]
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
    assert result.metrics["periods_per_year"] == (
        FOUR_HOUR_PERIODS_PER_YEAR
    )


def test_4h_cost_stress_is_monotonic() -> None:
    frames = _frames()
    normal = backtest_range_expansion_4h(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_range_expansion_4h(
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
    assert stressed.executed_weights.equals(
        normal.executed_weights
    )


def test_future_4h_prices_do_not_change_prior_weights() -> None:
    frames = _frames()
    cutoff = frames["BTC-EUR"].index[2_200]
    baseline = backtest_range_expansion_4h(
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
        frame.loc[future, ["open", "high", "low", "close"]] *= 3.0
        frame.loc[future, "volume"] *= 7.0
    revised = backtest_range_expansion_4h(
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


def test_period_metrics_use_six_bars_per_day() -> None:
    result = backtest_range_expansion_4h(
        _frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    metrics, returns = range_expansion_4h_period_metrics(
        result.equity_curve,
        start=result.equity_curve.index[0],
        end=result.equity_curve.index[-1],
    )

    assert len(returns) > 180
    assert metrics["periods_per_year"] == FOUR_HOUR_PERIODS_PER_YEAR
    assert metrics["profit_factor_unit"] == (
        "FOUR_HOUR_PORTFOLIO_RETURN"
    )


def test_forward_summary_uses_4h_units_and_daily_checkpoints() -> None:
    summary = relabel_4h_forward_summary(
        {
            "closed_daily_observations": 181,
            "required_closed_daily_observations": 2_190,
            "remaining_closed_daily_observations": 2_009,
            "checks": {
                "minimum_closed_daily_observations": False,
                "minimum_rebalances": False,
            },
            "diagnostic_progress": {},
        }
    )

    assert summary["observation_unit"] == "CLOSED_FOUR_HOUR_BAR"
    assert summary["closed_4h_observations"] == 181
    assert summary["required_closed_4h_observations"] == 2_190
    assert summary["remaining_closed_4h_observations"] == 2_009
    assert "closed_daily_observations" not in summary
    assert "minimum_closed_daily_observations" not in summary["checks"]
    milestones = summary["diagnostic_progress"]["milestones"]
    assert [row["closed_4h_observations"] for row in milestones] == [
        180,
        540,
        1_080,
        2_190,
    ]
    assert milestones[0]["calendar_days_equivalent"] == 30
    assert milestones[-1]["purpose"] == "FORMAL_SAMPLE_GATE"
