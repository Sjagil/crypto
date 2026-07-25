from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import ResearchSettings
from core.contracts import ResearchStatus
from research.backtest import BacktestResult
from research.optimization import (
    CPCVResult,
    StabilityResult,
    WalkForwardResult,
    acceptance_gate,
)
from research.stochastic_validation import (
    StochasticValidationPolicy,
    dirichlet_time_concentration_stress,
    stationary_bootstrap_monte_carlo,
    validate_strategy_return_paths,
)


def _test_policy(**overrides: object) -> StochasticValidationPolicy:
    values = {
        "simulations": 500,
        "expected_block_length": 5,
        "maximum_drawdown": 0.20,
        "maximum_drawdown_breach_probability": 0.01,
        "maximum_terminal_loss_probability": 0.05,
        "minimum_p05_total_return": 0.0,
        "dirichlet_blocks": 8,
        "minimum_observations": 30,
        "seed": 20260725,
        "batch_size": 64,
    }
    values.update(overrides)
    return StochasticValidationPolicy(**values)


def test_stochastic_gates_are_deterministic_and_pass_strong_path() -> None:
    returns = np.full(240, 0.001, dtype=float)
    policy = _test_policy()

    first = validate_strategy_return_paths(returns, returns, policy=policy)
    second = validate_strategy_return_paths(returns, returns, policy=policy)

    assert first == second
    assert first["passed"]
    assert all(first["checks"].values())
    assert first["normal"]["monte_carlo"]["p05_total_return"] > 0.0
    assert (
        first["normal"]["monte_carlo"][
            "maximum_drawdown_breach_probability_upper_confidence_bound"
        ]
        < 0.01
    )
    assert all(profile["passed"] for profile in first["normal"]["dirichlet"]["profiles"])


def test_stationary_bootstrap_fails_drawdown_concentrated_path() -> None:
    returns = np.tile(
        np.array([0.002] * 18 + [-0.12, -0.12], dtype=float),
        12,
    )
    result = stationary_bootstrap_monte_carlo(
        returns,
        policy=_test_policy(),
    )

    assert not result["passed"]
    assert not result["checks"]["drawdown_breach_probability"]
    assert result["maximum_drawdown_breach_probability"] > 0.01


def test_dirichlet_fails_when_edge_depends_on_one_time_block() -> None:
    returns = np.concatenate(
        (
            np.full(30, 0.04),
            np.full(210, -0.001),
        )
    )
    result = dirichlet_time_concentration_stress(
        returns,
        policy=_test_policy(),
    )

    assert not result["passed"]
    assert any(not profile["passed"] for profile in result["profiles"])


def test_stochastic_validation_fails_closed_on_bad_or_short_returns() -> None:
    policy = _test_policy()
    short = stationary_bootstrap_monte_carlo(np.ones(10) * 0.01, policy=policy)
    invalid = dirichlet_time_concentration_stress(
        np.array([0.01] * 39 + [-1.0]),
        policy=policy,
    )

    assert not short["passed"]
    assert short["reason_codes"] == ["INSUFFICIENT_OBSERVATIONS"]
    assert not invalid["passed"]
    assert invalid["reason_codes"][0].startswith("INVALID_RETURN_PATH:")


def test_policy_hash_changes_with_a_formal_threshold() -> None:
    baseline = _test_policy()
    stricter = _test_policy(maximum_terminal_loss_probability=0.01)

    assert baseline.policy_hash != stricter.policy_hash


def _acceptance_result(returns: np.ndarray) -> BacktestResult:
    index = pd.date_range("2025-01-01", periods=len(returns) + 1, freq="D", tz="UTC")
    equity = 10_000.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    return BacktestResult(
        strategy_id="stochastic-unit",
        initial_cash_eur=10_000.0,
        ending_equity_eur=float(equity[-1]),
        equity_curve=pd.DataFrame(
            {
                "equity": equity,
                "exposure_fraction": np.full(len(equity), 0.20),
            },
            index=index,
        ),
        trades=(),
        orders=(),
        metrics={
            "trade_count": 100,
            "effective_sample_size": 100,
            "net_expectancy_r": 0.10,
            "profit_factor": 1.50,
            "maximum_drawdown": 0.10,
            "probability_of_loss": 0.0,
            "probability_of_30pct_drawdown": 0.0,
            "symbol_profit_concentration": 0.10,
        },
        integrity={
            "valid_data": True,
            "valid_intelligence_timing": True,
        },
    )


def test_general_acceptance_gate_enforces_stochastic_paths() -> None:
    research = ResearchSettings(
        minimum_trades=100,
        minimum_effective_sample_size=60,
        minimum_positive_folds=1,
        walk_forward_folds=2,
        minimum_cpcv_path_consistency=0.0,
        minimum_deflated_sharpe_probability=0.0,
        parameter_stability_required=False,
        monte_carlo_runs=500,
    )
    walk_forward = WalkForwardResult(
        mode="anchored",
        folds=(),
        positive_folds=1,
        fold_profit_concentration=0.10,
        valid=True,
    )
    cpcv = CPCVResult(
        group_count=4,
        path_count=6,
        information_horizon_bars=1,
        purged_observations=1,
        embargoed_observations=1,
        path_returns=(0.1,),
        path_drawdowns=(0.1,),
        path_expectancy=(0.1,),
        path_profit_factor=(1.5,),
        path_consistency=1.0,
        probability_of_backtest_overfitting=0.0,
        final_holdout_excluded=True,
    )
    stability = StabilityResult(
        stable=True,
        tested_neighbors=1,
        positive_neighbors=1,
        acceptable_score_fraction=1.0,
        neighbor_scores=(1.0,),
    )
    strong = _acceptance_result(np.full(240, 0.001, dtype=float))
    passed = acceptance_gate(
        normal=strong,
        stressed=strong,
        holdout=strong,
        walk_forward=walk_forward,
        cpcv=cpcv,
        stability=stability,
        research=research,
        eligibility_valid=True,
        lookahead_safe=True,
        repainting_safe=True,
        deflated_sharpe_probability=1.0,
    )
    assert passed.status is ResearchStatus.RESEARCH_PASS
    assert passed.metrics["monte_carlo_gate"]
    assert passed.metrics["dirichlet_gate"]

    fragile = _acceptance_result(
        np.tile(np.array([0.002] * 18 + [-0.12, -0.12]), 12)
    )
    rejected = acceptance_gate(
        normal=fragile,
        stressed=fragile,
        holdout=fragile,
        walk_forward=walk_forward,
        cpcv=cpcv,
        stability=stability,
        research=research,
        eligibility_valid=True,
        lookahead_safe=True,
        repainting_safe=True,
        deflated_sharpe_probability=1.0,
    )
    assert rejected.status is ResearchStatus.REJECTED_RISK_OF_RUIN
    assert rejected.reasons == ("STOCHASTIC_ROBUSTNESS_GATES_FAILED",)

    too_short = _acceptance_result(np.full(10, 0.001, dtype=float))
    insufficient = acceptance_gate(
        normal=too_short,
        stressed=too_short,
        holdout=too_short,
        walk_forward=walk_forward,
        cpcv=cpcv,
        stability=stability,
        research=research,
        eligibility_valid=True,
        lookahead_safe=True,
        repainting_safe=True,
        deflated_sharpe_probability=1.0,
    )
    assert insufficient.status is ResearchStatus.REJECTED_RISK_OF_RUIN
    assert insufficient.metrics["normal_drawdown_breach_probability"] == 1.0
