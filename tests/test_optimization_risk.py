from __future__ import annotations

import math

import pytest

from research.backtest import BacktestConfig
from research.optimization import (
    chronological_split,
    combinatorial_purged_cross_validation,
    deflated_sharpe_ratio,
    multiple_testing_bootstrap,
    probability_of_backtest_overfitting,
    random_search,
    robust_score,
    strategy_lookahead_test,
    strategy_repainting_test,
    walk_forward_optimize,
    walk_forward_validate,
)
from research.strategies import get_strategy
from risk.risk_manager import (
    OperationalDegradation,
    PortfolioSnapshot,
    RiskManager,
    reliability_multiplier,
)


def test_split_search_and_walk_forward(
    features,
    isolated_settings,
    tmp_path,
) -> None:
    strategy = get_strategy("ema_trend_pullback")
    data = {"BTC-EUR": features}
    train, validation, holdout = chronological_split(data, purge_bars=2, embargo_bars=2)
    assert train["BTC-EUR"].index[-1] < validation["BTC-EUR"].index[0]
    assert validation["BTC-EUR"].index[-1] < holdout["BTC-EUR"].index[0]
    config = BacktestConfig(monte_carlo_runs=100)
    result = random_search(
        train,
        strategy,
        config,
        trials=2,
        seed=42,
        settings=isolated_settings,
        minimum_trades=1,
        checkpoint_path=tmp_path / "checkpoint.jsonl",
    )
    assert len(result.trials) == 2
    walk = walk_forward_validate(
        data,
        strategy,
        result.best_parameters,
        config,
        folds=2,
        purge_bars=1,
        embargo_bars=1,
        settings=isolated_settings,
    )
    assert walk.valid
    assert strategy_lookahead_test(features, strategy, result.best_parameters)
    assert strategy_repainting_test(features, strategy, result.best_parameters)


def test_true_walk_forward_reoptimizes_three_folds_and_stores_parameters(
    features,
    isolated_settings,
    tmp_path,
) -> None:
    strategy = get_strategy("ema_trend_pullback")
    result = walk_forward_optimize(
        {"BTC-EUR": features},
        strategy,
        BacktestConfig(monte_carlo_runs=100),
        folds=3,
        search_trials=2,
        purge_bars=1,
        embargo_bars=1,
        settings=isolated_settings,
        minimum_trades=1,
        checkpoint_path=tmp_path / "wfo.jsonl",
    )
    assert result.valid
    assert result.mode == "anchored_optimized"
    assert len(result.folds) == 3
    assert all(fold.selected_parameters for fold in result.folds)
    assert all(fold.parameter_hash for fold in result.folds)
    assert all(fold.optimization_trial_count == 2 for fold in result.folds)
    assert all(fold.train_end < fold.test_start for fold in result.folds)


def test_cpcv_purges_embargoes_and_reliability_never_increases_risk(
    features,
    isolated_settings,
) -> None:
    result = combinatorial_purged_cross_validation(
        features["close"].pct_change(),
        group_count=6,
        test_group_count=2,
        holding_spans=(3, 5),
        label_horizon_bars=2,
        indicator_lookback_bars=20,
        feature_availability_bars=1,
        final_holdout_start=features.index[-100],
    )
    assert result.path_count == 15
    assert result.information_horizon_bars == 20
    assert result.purged_observations > 0
    assert result.embargoed_observations > 0
    assert result.final_holdout_excluded
    multiplier = reliability_multiplier(
        {
            "holdout_expectancy_score": 0.8,
            "stressed_profit_factor_score": 0.7,
            "positive_walk_forward_fold_ratio": 0.75,
            "cpcv_path_consistency": result.path_consistency,
            "monte_carlo_probability_of_loss": 0.2,
            "parameter_stability": 0.8,
            "effective_sample_score": 0.9,
            "symbol_concentration": 0.25,
            "regime_concentration": 0.3,
            "recent_operational_score": 0.8,
            "data_completeness": 1.0,
            "provider_health": 1.0,
        }
    )
    assert 0.0 <= multiplier <= 1.0
    assert reliability_multiplier({}) == 0.0
    decision = RiskManager.from_settings(isolated_settings).assess_entry(
        market="BTC-EUR",
        entry_price=100.0,
        stop_price=95.0,
        snapshot=PortfolioSnapshot(
            equity_eur=10_000,
            cash_eur=10_000,
            day_start_equity_eur=10_000,
            peak_equity_eur=10_000,
            trades_today=0,
        ),
        reliability_evidence={},
        volatility_multiplier=2.0,
        correlation_multiplier=2.0,
        liquidity_multiplier=2.0,
        drawdown_multiplier=2.0,
    )
    assert not decision.approved
    assert "ZERO_SIZE" in decision.reason_codes


def test_multiple_testing_diagnostics_separate_signal_from_noise() -> None:
    import numpy as np
    import pandas as pd

    randomizer = np.random.default_rng(20260724)
    rows = 480
    returns = pd.DataFrame(
        {
            "persistent_signal": randomizer.normal(0.0025, 0.01, rows),
            "noise_a": randomizer.normal(0.0, 0.01, rows),
            "noise_b": randomizer.normal(0.0, 0.01, rows),
            "noise_c": randomizer.normal(0.0, 0.01, rows),
        }
    )
    result = multiple_testing_bootstrap(
        returns,
        bootstrap_samples=200,
        block_size=5,
        seed=7,
    )
    assert result.strategy_count == 4
    assert result.known_trial_count == 4
    assert result.observation_count == rows
    assert result.white_reality_check_pvalue < 0.10
    assert result.hansen_spa_pvalue < 0.10
    assert result.probability_of_backtest_overfitting is not None
    assert (
        result.deflated_sharpe_probabilities["persistent_signal"]
        > result.deflated_sharpe_probabilities["noise_a"]
    )
    conservative = multiple_testing_bootstrap(
        returns,
        bootstrap_samples=200,
        block_size=5,
        seed=7,
        known_trial_count=1_000,
    )
    assert conservative.known_trial_count == 1_000
    assert (
        conservative.deflated_sharpe_probabilities["persistent_signal"]
        <= result.deflated_sharpe_probabilities["persistent_signal"]
    )
    pbo, logits = probability_of_backtest_overfitting(returns, group_count=8)
    assert pbo is not None
    assert len(logits) > 0
    dsr = deflated_sharpe_ratio(
        returns["persistent_signal"],
        [0.0, 0.05, 0.10, 0.15],
    )
    assert dsr > 0.95


def test_degradation_persistence_hard_kill_and_manual_reset(tmp_path) -> None:
    monitor = OperationalDegradation(
        state_path=tmp_path / "degradation.json",
        audit_path=tmp_path / "degradation.jsonl",
        persistence=2,
    )
    assert monitor.evaluate(warning=("SLIPPAGE_HIGH",))["state"] == "NORMAL"
    assert monitor.evaluate(warning=("SLIPPAGE_HIGH",))["state"] == "WARNING"
    assert (
        monitor.evaluate(block_new_entries=("REQUIRED_CONTEXT_MISSING",))["state"]
        == "NORMAL"
    )
    assert (
        monitor.evaluate(block_new_entries=("REQUIRED_CONTEXT_MISSING",))["state"]
        == "BLOCK_NEW_ENTRIES"
    )
    restored = OperationalDegradation(
        state_path=tmp_path / "degradation.json",
        audit_path=tmp_path / "degradation.jsonl",
    )
    assert restored.evaluate()["state"] == "BLOCK_NEW_ENTRIES"
    with pytest.raises(RuntimeError):
        restored.manual_reset(
            confirmed=True,
            reason="not yet healthy",
            resolved_health_checks=False,
        )
    assert (
        restored.manual_reset(
            confirmed=True,
            reason="context restored",
            resolved_health_checks=True,
        )["state"]
        == "NORMAL"
    )
    assert (
        restored.evaluate(kill_switch=("NEGATIVE_BALANCE",))["state"]
        == "KILL_SWITCH"
    )


def test_robust_score_preserves_zero_risk_metrics() -> None:
    score = robust_score(
        {
            "trade_count": 30,
            "net_expectancy_r": 1.0,
            "profit_factor": 2.0,
            "maximum_drawdown": 0.0,
            "turnover": 0.0,
            "symbol_profit_concentration": 0.0,
            "time_under_water": 0.0,
            "net_return": 0.0,
            "win_rate": 0.5,
        },
        minimum_trades=30,
    )
    assert score == pytest.approx(math.sqrt(30) + math.log1p(2.0))
    assert math.isfinite(score)


def test_risk_manager_rejects_unknown_and_loss_limit(restrictive_settings) -> None:
    manager = RiskManager.from_settings(restrictive_settings)
    healthy = PortfolioSnapshot(
        equity_eur=10_000,
        cash_eur=10_000,
        day_start_equity_eur=10_000,
        peak_equity_eur=10_000,
        trades_today=0,
    )
    approved = manager.assess_entry(
        market="BTC-EUR",
        entry_price=100,
        stop_price=95,
        snapshot=healthy,
    )
    assert approved.approved
    unknown = manager.assess_entry(
        market="UNKNOWN-EUR",
        entry_price=100,
        stop_price=95,
        snapshot=healthy,
    )
    assert not unknown.approved
    loss = PortfolioSnapshot(
        equity_eur=9_700,
        cash_eur=9_700,
        day_start_equity_eur=10_000,
        peak_equity_eur=10_000,
        trades_today=0,
    )
    assert not manager.assess_entry(
        market="BTC-EUR",
        entry_price=100,
        stop_price=95,
        snapshot=loss,
    ).approved
