from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_storm import (
    STORM_TRIAL_COUNT,
    PortfolioStormDNA,
    large_matrix_multiple_testing,
    preregistered_storm_dna,
    run_portfolio_storm,
    storm_plan,
)


def _frames(rows: int = 620) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2023-01-01", periods=rows, freq="1D", tz="UTC")

    def frame(drift: float, phase: float) -> pd.DataFrame:
        change = (
            drift
            + 0.006 * np.sin(np.arange(rows) / 19.0 + phase)
            + 0.002 * np.cos(np.arange(rows) / 7.0 + phase)
        )
        close = 100.0 * np.exp(np.cumsum(change))
        open_price = close * (
            1.0 + 0.001 * np.sin(np.arange(rows) / 5.0 + phase)
        )
        return pd.DataFrame(
            {
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.01,
                "low": np.minimum(open_price, close) * 0.99,
                "close": close,
                "volume": 10_000.0,
            },
            index=index,
        )

    return {
        "BTC-EUR": frame(0.0005, 0.0),
        "ETH-EUR": frame(0.0007, 0.4),
        "SOL-EUR": frame(0.0004, 0.8),
        "LINK-EUR": frame(0.0006, 1.2),
    }


def test_storm_plan_is_deterministic_unique_and_strictly_risk_bounded():
    first = preregistered_storm_dna(trial_count=100, seed=7)
    second = preregistered_storm_dna(trial_count=100, seed=7)
    assert [row.dna_hash for row in first] == [
        row.dna_hash for row in second
    ]
    assert len({row.dna_hash for row in first}) == 100
    assert all(row.maximum_total_exposure <= 0.40 for row in first)
    assert all(row.maximum_position_exposure <= 0.20 for row in first)
    assert all(row.minimum_cash >= 0.60 for row in first)
    assert storm_plan()["trial_count"] == STORM_TRIAL_COUNT


def test_storm_rejects_risk_limit_relaxation():
    with pytest.raises(ValueError, match="strict limit"):
        PortfolioStormDNA(
            momentum_fast=20,
            momentum_slow=90,
            asset_ema_period=50,
            btc_ema_period=200,
            top_n=2,
            rebalance_days=7,
            regime_mapping="linear",
            weighting="equal",
            maximum_total_exposure=0.80,
            rebalance_buffer=0.05,
            require_btc_uptrend=True,
            minimum_cash=0.20,
        )


def test_large_matrix_multiple_testing_is_deterministic_and_uses_every_path():
    generator = np.random.default_rng(19)
    returns = generator.normal(0.0, 0.01, size=(96, 40))
    returns[:, 0] += 0.002
    first = large_matrix_multiple_testing(
        returns,
        bootstrap_samples=200,
        block_size=4,
        seed=17,
        batch_size=16,
    )
    second = large_matrix_multiple_testing(
        returns,
        bootstrap_samples=200,
        block_size=4,
        seed=17,
        batch_size=16,
    )

    assert first == second
    assert first["strategy_count"] == 40
    assert first["observation_count"] == 96
    assert 0 < first["white_reality_check_pvalue"] <= 1
    assert 0 < first["hansen_spa_pvalue"] <= 1
    assert first["probability_of_backtest_overfitting"] is not None


def test_small_storm_uses_development_only_pareto_and_never_promotes():
    dna = preregistered_storm_dna(trial_count=24, seed=11)
    report, matrix, timestamps = run_portfolio_storm(
        _frames(),
        dna,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        prior_known_trials=1_312,
        known_trial_count=1_312,
        maximum_survivors=8,
    )

    assert report["trial_count"] == 24
    assert report["new_strategy_trial_count"] == 0
    assert report["total_known_trials"] == 1_312
    assert report["selection_basis"] == "DEVELOPMENT_ONLY"
    assert report["selection_integrity"] == {
        "development_returns_only": True,
        "development_turnover_only": True,
        "validation_used_for_selection": False,
        "confirmation_used_for_selection": False,
    }
    assert report["pareto_survivor_count"] <= 8
    assert matrix.shape == (len(timestamps), 24)
    assert np.isfinite(matrix).all()
    assert report["multiple_testing"]["dsr_total_trial_denominator"] == 1_312
    assert report["multiple_testing"]["strategy_count"] == 24
    assert report["multiple_testing"]["white_spa_status"] == (
        "FORMALLY_EVALUATED_ALL_STORM_TRIALS"
    )
    assert report["research_pass"] is False
    assert report["paper_candidates"] == 0
    assert report["orders_generated"] == 0
    assert report["live_ready"] is False
    for survivor in report["pareto_survivors"]:
        assert set(survivor) >= {
            "development",
            "validation",
            "confirmation",
            "deflated_sharpe_probability",
        }
        assert survivor["paper_candidate_permitted"] is False
        assert survivor["live_ready"] is False
