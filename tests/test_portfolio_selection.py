from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_selection import (
    RotationParameters,
    RotationPortfolioPolicy,
    backtest_rotation,
    ensemble_rotation_parameter_grid,
    rotation_benchmark_suite,
    rotation_decision_snapshot,
    rotation_parameter_grid,
    rotation_period_metrics,
    rotation_regime_coverage,
)


def _rotation_frames(rows: int = 260) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2023-01-01", periods=rows, freq="1D", tz="UTC")

    def frame(values: np.ndarray) -> pd.DataFrame:
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
        "BTC-EUR": frame(100.0 * np.power(1.001, np.arange(rows))),
        "ETH-EUR": frame(100.0 * np.power(1.003, np.arange(rows))),
        "SOL-EUR": frame(100.0 * np.power(0.999, np.arange(rows))),
    }


def _parameters(**updates) -> RotationParameters:
    values = {
        "momentum_lookback": 20,
        "top_n": 1,
        "rebalance_days": 7,
        "asset_ema_period": 20,
        "btc_ema_period": 20,
        "require_btc_uptrend": True,
        "gross_exposure": 0.40,
        "minimum_cash": 0.20,
        "maximum_positions": 2,
    }
    values.update(updates)
    return RotationParameters(**values)


def test_rotation_parameter_grid_is_joint_deterministic_and_risk_bounded() -> None:
    first = rotation_parameter_grid(
        momentum_lookbacks=(20, 60),
        top_ns=(1, 2),
        rebalance_days=(7,),
        asset_ema_periods=(50,),
        btc_filters=(True, False),
        weightings=("equal", "inverse_volatility"),
    )
    second = rotation_parameter_grid(
        momentum_lookbacks=(20, 60),
        top_ns=(1, 2),
        rebalance_days=(7,),
        asset_ema_periods=(50,),
        btc_filters=(True, False),
        weightings=("equal", "inverse_volatility"),
    )
    assert len(first) == 16
    assert [item.dna_hash for item in first] == [item.dna_hash for item in second]
    assert len({item.dna_hash for item in first}) == len(first)
    assert all(item.gross_exposure == pytest.approx(0.40) for item in first)
    with pytest.raises(ValueError, match="maximum_positions"):
        _parameters(top_n=3)
    ensemble = ensemble_rotation_parameter_grid()
    assert len(ensemble) == 160
    assert all(item.gross_exposure == pytest.approx(0.25) for item in ensemble)
    assert all(len(item.momentum_lookbacks) >= 2 for item in ensemble)


def test_rotation_selects_relative_winner_and_records_terminal_liquidation() -> None:
    result = backtest_rotation(
        _rotation_frames(),
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    ranked = result.decisions[result.decisions["reason"] == "RANKED_MOMENTUM"]
    assert not ranked.empty
    assert all(assets == ["ETH-EUR"] for assets in ranked["selected_assets"])
    assert result.metrics["net_return"] > 0
    assert result.metrics["gross_return"] > result.metrics["net_return"]
    assert result.metrics["maximum_positions_observed"] == 1
    assert result.metrics["average_exposure"] <= 0.40 + 1e-12
    assert result.integrity["decision_at_close_execution_next_open"]
    assert result.integrity["terminal_liquidation_recorded"]
    assert result.decisions.iloc[-1]["reason"] == "TERMINAL_LIQUIDATION"
    assert result.executed_weights.iloc[-1].sum() == pytest.approx(0.0)


def test_rotation_future_close_cannot_change_prior_decisions_or_equity() -> None:
    frames = _rotation_frames()
    baseline = backtest_rotation(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    shocked = {market: frame.copy() for market, frame in frames.items()}
    shocked["SOL-EUR"].iloc[-1, shocked["SOL-EUR"].columns.get_loc("close")] *= 100.0
    changed = backtest_rotation(
        shocked,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    cutoff = baseline.equity_curve.index[-2]
    pd.testing.assert_series_equal(
        baseline.equity_curve.loc[:cutoff],
        changed.equity_curve.loc[:cutoff],
    )
    pd.testing.assert_frame_equal(
        baseline.decisions.iloc[:-1].reset_index(drop=True),
        changed.decisions.iloc[:-1].reset_index(drop=True),
    )


def test_rotation_btc_regime_moves_to_cash_and_costs_are_monotonic() -> None:
    frames = _rotation_frames()
    btc = frames["BTC-EUR"]
    decline = 150.0 * np.power(0.997, np.arange(len(btc)))
    btc.loc[:, ["open", "close"]] = np.column_stack((decline, decline))
    btc.loc[:, "high"] = decline * 1.01
    btc.loc[:, "low"] = decline * 0.99
    filtered = backtest_rotation(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    assert filtered.metrics["average_exposure"] == pytest.approx(0.0)
    assert filtered.metrics["net_return"] == pytest.approx(0.0)

    unfiltered = backtest_rotation(
        frames,
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    costly = backtest_rotation(
        frames,
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    assert costly.metrics["net_return"] < unfiltered.metrics["net_return"]
    assert costly.cost_breakdown["total_cost_amount"] > 0


def test_rotation_period_metrics_are_finite_and_effective_sample_is_bounded() -> None:
    result = backtest_rotation(
        _rotation_frames(),
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    metrics, returns = rotation_period_metrics(
        result.equity_curve,
        start=result.equity_curve.index[0],
        end=result.equity_curve.index[-1],
    )
    assert metrics["observations"] == len(returns)
    assert 1 <= metrics["effective_sample_size"] <= len(returns)
    assert np.isfinite(float(metrics["sharpe"]))
    assert float(metrics["daily_profit_factor"]) > 0


def test_rotation_asset_joins_only_after_its_own_causal_warmup() -> None:
    frames = _rotation_frames(320)
    baseline = backtest_rotation(
        frames,
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    late = frames["ETH-EUR"].iloc[180:].copy()
    late.loc[:, ["open", "close"]] *= 3.0
    late.loc[:, "high"] = late["close"] * 1.01
    late.loc[:, "low"] = late["close"] * 0.99
    expanded = backtest_rotation(
        frames | {"ADA-EUR": late},
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    cutoff = late.index[0] + pd.Timedelta(days=19)
    pd.testing.assert_series_equal(
        baseline.equity_curve.loc[:cutoff],
        expanded.equity_curve.loc[:cutoff],
    )
    assert expanded.integrity["point_in_time_asset_inception"]
    assert not expanded.integrity["common_history_only"]


def test_multi_horizon_continuous_regime_is_causal_and_scales_exposure() -> None:
    parameters = _parameters(
        additional_momentum_lookbacks=(60, 120),
        require_btc_uptrend=False,
        continuous_regime=True,
    )
    result = backtest_rotation(
        _rotation_frames(320),
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    ranked = result.decisions[result.decisions["reason"] == "RANKED_MOMENTUM"]
    assert not ranked.empty
    assert ranked["exposure_scale"].between(0.10, 1.0).all()
    assert result.executed_weights.abs().sum(axis=1).max() <= 0.40 + 1e-12
    assert parameters.momentum_lookbacks == (20, 60, 120)


def test_rotation_supports_an_explicit_non_eur_btc_benchmark() -> None:
    usdt_frames = {
        market.replace("-EUR", "-USDT"): frame
        for market, frame in _rotation_frames().items()
    }
    result = backtest_rotation(
        usdt_frames,
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        benchmark_market="BTC-USDT",
    )
    assert result.integrity["benchmark_market"] == "BTC-USDT"
    assert result.metrics["net_return"] > 0


def test_strict_portfolio_policy_rejects_unknown_assets_and_caps_each_position() -> None:
    policy = RotationPortfolioPolicy(
        allowed_markets=("BTC-EUR", "ETH-EUR", "SOL-EUR"),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=20,
    )
    single_winner = backtest_rotation(
        _rotation_frames(),
        _parameters(gross_exposure=0.40),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )
    assert single_winner.metrics["maximum_position_exposure_observed"] <= 0.20 + 1e-12
    assert single_winner.executed_weights.sum(axis=1).max() <= 0.40 + 1e-12
    assert single_winner.integrity["maximum_position_exposure_respected"]
    assert single_winner.integrity["minimum_cash_respected"]

    with pytest.raises(ValueError, match="outside fail-closed"):
        backtest_rotation(
            _rotation_frames() | {"TAO-EUR": _rotation_frames()["ETH-EUR"]},
            _parameters(),
            fee_rate=0.0025,
            slippage_bps=8.0,
            spread_bps=5.0,
            portfolio_policy=policy,
        )


def test_rotation_reports_unambiguous_events_samples_and_profit_factors() -> None:
    result = backtest_rotation(
        _rotation_frames(320),
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
    )
    metrics = result.metrics
    assert metrics["scheduled_rebalance_opportunities"] > 0
    assert metrics["buy_fills"] > 0
    assert metrics["sell_fills"] > 0
    assert metrics["closed_position_episodes"] == len(result.position_episodes)
    assert 1 <= metrics["portfolio_period_effective_sample_size"] <= metrics[
        "raw_portfolio_period_observations"
    ]
    for name in (
        "portfolio_period_profit_factor",
        "closed_position_profit_factor",
        "asset_trade_profit_factor",
        "rebalance_episode_profit_factor",
    ):
        assert float(metrics[name]) >= 0.0
    ranked = result.decisions[result.decisions["reason"] == "RANKED_MOMENTUM"]
    assert ranked["weight_changes"].map(lambda value: isinstance(value, dict)).all()
    assert ranked["target_weights"].map(lambda value: isinstance(value, dict)).all()


def test_asset_requires_policy_history_before_it_can_enter_ranking() -> None:
    frames = _rotation_frames(360)
    late = frames["ETH-EUR"].iloc[180:].copy()
    late.loc[:, ["open", "close"]] *= np.power(1.01, np.arange(len(late)))[:, None]
    late.loc[:, "high"] = late["close"] * 1.01
    late.loc[:, "low"] = late["close"] * 0.99
    expanded = frames | {"HYPE-EUR": late}
    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(expanded),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )
    result = backtest_rotation(
        expanded,
        _parameters(require_btc_uptrend=False),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )
    first_eligible = late.index[89]
    before = result.decisions[
        pd.to_datetime(result.decisions["decision_at"], utc=True) < first_eligible
    ]
    assert all("HYPE-EUR" not in assets for assets in before["selected_assets"])
    assert result.integrity["minimum_history_observations"] == 90


def test_rotation_asset_attribution_reconciles_and_benchmarks_share_timeline() -> None:
    frames = _rotation_frames(320)
    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(frames),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=20,
    )
    result = backtest_rotation(
        frames,
        _parameters(
            additional_momentum_lookbacks=(60,),
            require_btc_uptrend=False,
            continuous_regime=True,
        ),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )
    assert result.integrity["asset_pnl_reconciled"]
    attribution = result.metrics["asset_pnl_attribution"]
    assert set(attribution) == set(frames)
    assert sum(row["net_pnl_amount"] for row in attribution.values()) == pytest.approx(
        result.metrics["net_return"]
    )

    suite = rotation_benchmark_suite(
        frames,
        result.parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )
    assert suite["timeline"]["start"] == result.equity_curve.index[0].isoformat()
    assert suite["candidate"]["strategy_dna_hash"] == result.parameters.dna_hash
    assert set(suite["benchmarks"]) >= {
        "cash",
        "btc_buy_and_hold_full_exposure",
        "point_in_time_equal_weight_weekly_operational_exposure",
        "btc_buy_and_hold_volatility_matched",
    }
    assert set(suite["ablations"]) == {
        "multi_horizon_without_continuous_regime",
        "single_horizon_without_regime",
    }


def test_frozen_decision_snapshot_and_regime_coverage_never_create_orders() -> None:
    frames = _rotation_frames(320)
    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(frames),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )
    parameters = _parameters(
        additional_momentum_lookbacks=(60,),
        require_btc_uptrend=False,
        continuous_regime=True,
    )
    snapshot = rotation_decision_snapshot(
        frames,
        parameters,
        portfolio_policy=policy,
    )
    assert snapshot["status"] == "FROZEN_FORWARD_RESEARCH"
    assert snapshot["execution_instruction"].endswith("HYPOTHETICAL_ONLY")
    assert snapshot["orders_generated"] == 0
    assert snapshot["orders_submitted"] == 0
    assert snapshot["cash_fraction"] >= 0.60 - 1e-12

    result = backtest_rotation(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )
    coverage = rotation_regime_coverage(result.decisions, minimum_per_state=1)
    assert coverage["decision_observations"] > 0
    assert set(coverage["counts"]) == {"btc_trend", "volatility", "breadth"}
