from __future__ import annotations

from decimal import Decimal

import pytest

from rl.position_management import (
    BaselinePolicy,
    PositionAction,
    PositionManagementEnvironment,
    PositionState,
    RLEligibilityEvidence,
    baseline_action,
    evaluate_rl_eligibility,
)


def _state(**overrides: object) -> PositionState:
    values = {
        "step": 0,
        "position_fraction": Decimal("1"),
        "unrealized_return": Decimal("0"),
        "drawdown": Decimal("0.01"),
        "volatility": Decimal("0.02"),
        "spread_fraction": Decimal("0.001"),
    }
    values.update(overrides)
    return PositionState(**values)


def test_actions_can_only_hold_or_reduce_spot_exposure() -> None:
    env = PositionManagementEnvironment(
        _state(), one_way_cost_fraction=Decimal("0.002")
    )
    transition = env.step(
        PositionAction.REDUCE_25,
        next_market_return=Decimal("0.01"),
        next_drawdown=Decimal("0.01"),
        next_volatility=Decimal("0.02"),
        next_spread_fraction=Decimal("0.001"),
    )
    assert transition.next_state.position_fraction == Decimal("0.75")
    assert transition.transaction_cost == Decimal("0.00050")
    assert transition.execution_authority is False
    assert transition.next_state.position_fraction <= transition.state.position_fraction


def test_exit_is_terminal_and_environment_cannot_continue() -> None:
    env = PositionManagementEnvironment(_state(), one_way_cost_fraction=Decimal("0.001"))
    transition = env.step(
        PositionAction.EXIT,
        next_market_return=Decimal("-0.02"),
        next_drawdown=Decimal("0.04"),
        next_volatility=Decimal("0.05"),
        next_spread_fraction=Decimal("0.002"),
    )
    assert transition.next_state.terminal is True
    with pytest.raises(RuntimeError, match="terminal"):
        env.step(
            PositionAction.HOLD,
            next_market_return=Decimal("0"),
            next_drawdown=Decimal("0"),
            next_volatility=Decimal("0"),
            next_spread_fraction=Decimal("0"),
        )


def test_deterministic_baselines_and_rl_gate_fail_closed() -> None:
    assert baseline_action(
        BaselinePolicy.DRAWDOWN_EXIT, _state(drawdown=Decimal("0.06"))
    ) is PositionAction.EXIT
    assert baseline_action(
        BaselinePolicy.VOLATILITY_REDUCER, _state(volatility=Decimal("0.05"))
    ) is PositionAction.REDUCE_50
    decision = evaluate_rl_eligibility(
        RLEligibilityEvidence(
            prospective_episode_count=0,
            completed_episode_count=0,
            distinct_regime_count=0,
            baseline_count=3,
            multi_seed_count=0,
        )
    )
    assert decision["status"] == "RL_TRAINING_BLOCKED"
    assert decision["authority"] == "SHADOW_ONLY"
    assert decision["execution_authority"] is False
    assert decision["automatic_promotion_permitted"] is False
