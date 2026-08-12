from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_breakout import (
    AtrRiskBreakoutParameters,
    BreakoutPortfolioParameters,
    EfficientAtrRiskBreakoutParameters,
    backtest_breakout_portfolio,
    breakout_observer_snapshot,
    breakout_portfolio_parameter_set,
    efficient_atr_breakout_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy


def test_atr_risk_breakout_is_separate_bounded_4h_dna() -> None:
    frames = {
        market: frame.set_axis(
            pd.date_range(
                "2022-01-01",
                periods=len(frame),
                freq="4h",
                tz="UTC",
            )
        )
        for market, frame in _frames(rows=900).items()
    }
    parameters = AtrRiskBreakoutParameters()
    assert parameters.dna_hash not in {
        row.dna_hash for row in breakout_portfolio_parameter_set()
    }
    result = backtest_breakout_portfolio(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=4.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    assert result.summary()["timeframe"] == "4h"
    assert result.metrics["position_sizing_policy"] == "ATR_EQUAL_EUR_RISK"
    assert result.metrics["maximum_exposure_observed"] <= 0.40 + 1e-12
    assert result.metrics["maximum_position_exposure_observed"] <= 0.20 + 1e-12


def test_efficient_atr_grid_is_small_frozen_and_separate_from_v1() -> None:
    rows = efficient_atr_breakout_parameter_set()
    assert len(rows) == 24
    assert len({row.dna_hash for row in rows}) == len(rows)
    assert AtrRiskBreakoutParameters().dna_hash not in {row.dna_hash for row in rows}
    assert {row.rebalance_days for row in rows} == {3, 7}
    assert {row.rebalance_buffer for row in rows} == {0.05, 0.10}
    with pytest.raises(ValueError, match="pre-registered"):
        EfficientAtrRiskBreakoutParameters(
            entry_lookback=121,
            exit_lookback=60,
            trend_ema_period=600,
            rebalance_days=3,
            rebalance_buffer=0.05,
        )


def _frames(rows: int = 520) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2022-01-01", periods=rows, freq="1D", tz="UTC")

    def frame(drift: float, phase: float) -> pd.DataFrame:
        steps = (
            drift
            + 0.004 * np.sin(np.arange(rows) / 11.0 + phase)
            + 0.002 * np.cos(np.arange(rows) / 5.0 + phase)
        )
        values = 100.0 * np.exp(np.cumsum(steps))
        return pd.DataFrame(
            {
                "open": values,
                "high": values * 1.01,
                "low": values * 0.99,
                "close": values,
                "volume": 1_000.0,
            },
            index=index,
        )

    return {
        "BTC-EUR": frame(0.0010, 0.0),
        "ETH-EUR": frame(0.0014, 0.7),
        "SOL-EUR": frame(0.0007, 1.4),
        "LINK-EUR": frame(0.0009, 2.1),
    }


def _policy(markets: tuple[str, ...]) -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )


def _parameters(**updates) -> BreakoutPortfolioParameters:
    values = {
        "entry_lookback": 20,
        "exit_lookback": 10,
        "trend_ema_period": 50,
        "weighting": "equal",
    }
    values.update(updates)
    return BreakoutPortfolioParameters(**values)


def test_breakout_parameter_set_is_small_pre_registered_and_unique() -> None:
    rows = breakout_portfolio_parameter_set()
    assert len(rows) == 8
    assert len({row.dna_hash for row in rows}) == len(rows)
    assert {(row.entry_lookback, row.exit_lookback) for row in rows} == {
        (20, 10),
        (55, 20),
    }
    assert {row.trend_ema_period for row in rows} == {50, 200}
    assert {row.weighting for row in rows} == {"equal", "inverse_volatility"}


def test_breakout_backtest_is_next_open_costed_and_exposure_bounded() -> None:
    frames = _frames()
    result = backtest_breakout_portfolio(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    assert result.metrics["rebalance_count"] > 0
    assert result.metrics["closed_position_episodes"] > 0
    assert result.metrics["maximum_exposure_observed"] <= 0.40 + 1e-12
    assert result.metrics["maximum_position_exposure_observed"] <= 0.20 + 1e-12
    assert result.integrity["prior_channel_only"]
    assert result.integrity["decision_at_close_execution_next_open"]
    assert result.integrity["asset_pnl_reconciled"]
    assert result.integrity["terminal_liquidation_recorded"]
    assert result.cost_breakdown["total_cost_amount"] > 0
    assert (
        pd.to_datetime(result.decisions.iloc[0]["executed_at"], utc=True)
        > pd.to_datetime(result.decisions.iloc[0]["decision_at"], utc=True)
    )


def test_future_close_cannot_change_prior_breakout_decisions_or_equity() -> None:
    frames = _frames()
    baseline = backtest_breakout_portfolio(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    shocked = {market: frame.copy() for market, frame in frames.items()}
    shocked["SOL-EUR"].iloc[-1, shocked["SOL-EUR"].columns.get_loc("close")] *= 10.0
    changed = backtest_breakout_portfolio(
        shocked,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    cutoff = baseline.equity_curve.index[-2]
    pd.testing.assert_series_equal(
        baseline.equity_curve.loc[:cutoff],
        changed.equity_curve.loc[:cutoff],
    )
    baseline_decisions = baseline.decisions[
        pd.to_datetime(baseline.decisions["decision_at"], utc=True) < cutoff
    ].reset_index(drop=True)
    changed_decisions = changed.decisions[
        pd.to_datetime(changed.decisions["decision_at"], utc=True) < cutoff
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_decisions, changed_decisions)


def test_isolated_asset_gap_defers_execution_and_uses_causal_valuation() -> None:
    frames = _frames()
    baseline = backtest_breakout_portfolio(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    episode = baseline.position_episodes.iloc[0]
    market = str(episode["market"])
    opened_at = pd.to_datetime(episode["opened_at"], utc=True)
    closed_at = pd.to_datetime(episode["closed_at"], utc=True)
    candidates = frames[market].index[
        (frames[market].index > opened_at)
        & (frames[market].index < closed_at)
    ]
    assert len(candidates) > 0
    gapped = {name: frame.copy() for name, frame in frames.items()}
    gapped[market] = gapped[market].drop(index=candidates[len(candidates) // 2])

    result = backtest_breakout_portfolio(
        gapped,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )

    assert result.metrics["valuation_carry_rows"] == 1
    assert result.integrity["signal_ohlc_not_forward_filled"]
    assert result.integrity["valuation_carry_is_backward_only"]
    assert result.integrity["missing_candle_execution_deferred"]
    assert result.integrity["asset_pnl_reconciled"]


def test_breakout_costs_are_monotonic_and_unknown_asset_fails_closed() -> None:
    frames = _frames()
    free = backtest_breakout_portfolio(
        frames,
        _parameters(weighting="inverse_volatility"),
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    costly = backtest_breakout_portfolio(
        frames,
        _parameters(weighting="inverse_volatility"),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    assert costly.metrics["net_return"] < free.metrics["net_return"]
    with pytest.raises(ValueError, match="outside fail-closed"):
        backtest_breakout_portfolio(
            frames | {"TAO-EUR": frames["ETH-EUR"]},
            _parameters(),
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            portfolio_policy=_policy(tuple(frames)),
        )


def test_breakout_observer_never_creates_or_submits_orders() -> None:
    frames = _frames()
    result = backtest_breakout_portfolio(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    snapshot = breakout_observer_snapshot(result)
    assert snapshot["status"] == "FROZEN_FORWARD_RESEARCH"
    assert snapshot["orders_generated"] == 0
    assert snapshot["orders_submitted"] == 0
    assert not snapshot["candidate_promotion_implied"]
