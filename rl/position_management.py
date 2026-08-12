"""Native, dependency-free SHADOW environment for position management.

FinRL informed the explicit state/action/reward separation and deterministic
evaluation boundary.  The implementation is native and intentionally smaller:
spot-only actions may hold or reduce exposure, never add leverage, short, place
an order, or bypass the canonical portfolio/risk/execution chain.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from core.contracts import FrozenModel

ZERO = Decimal("0")
ONE = Decimal("1")


class PositionAction(StrEnum):
    HOLD = "HOLD"
    REDUCE_25 = "REDUCE_25"
    REDUCE_50 = "REDUCE_50"
    EXIT = "EXIT"


class PositionState(FrozenModel):
    step: int = Field(ge=0)
    position_fraction: Decimal = Field(ge=ZERO, le=ONE)
    unrealized_return: Decimal
    drawdown: Decimal = Field(ge=ZERO, le=ONE)
    volatility: Decimal = Field(ge=ZERO)
    spread_fraction: Decimal = Field(ge=ZERO)
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    state: PositionState
    action: PositionAction
    next_state: PositionState
    reward: Decimal
    transaction_cost: Decimal
    execution_authority: bool = False


class PositionManagementEnvironment:
    """Pure state transition model suitable for later Gymnasium adaptation."""

    def __init__(
        self,
        initial_state: PositionState,
        *,
        one_way_cost_fraction: Decimal,
        drawdown_penalty: Decimal = Decimal("0.10"),
    ) -> None:
        if initial_state.terminal:
            raise ValueError("initial state cannot be terminal")
        if one_way_cost_fraction < ZERO:
            raise ValueError("cost fraction cannot be negative")
        if drawdown_penalty < ZERO:
            raise ValueError("drawdown penalty cannot be negative")
        self.state = initial_state
        self.one_way_cost_fraction = Decimal(one_way_cost_fraction)
        self.drawdown_penalty = Decimal(drawdown_penalty)

    @staticmethod
    def target_fraction(current: Decimal, action: PositionAction) -> Decimal:
        fractions = {
            PositionAction.HOLD: ONE,
            PositionAction.REDUCE_25: Decimal("0.75"),
            PositionAction.REDUCE_50: Decimal("0.50"),
            PositionAction.EXIT: ZERO,
        }
        return current * fractions[action]

    def step(
        self,
        action: PositionAction,
        *,
        next_market_return: Decimal,
        next_drawdown: Decimal,
        next_volatility: Decimal,
        next_spread_fraction: Decimal,
        final_step: bool = False,
    ) -> Transition:
        if self.state.terminal:
            raise RuntimeError("cannot step a terminal episode")
        current = self.state
        target = self.target_fraction(current.position_fraction, action)
        reduced = current.position_fraction - target
        transaction_cost = reduced * self.one_way_cost_fraction
        reward = (
            target * Decimal(next_market_return)
            - transaction_cost
            - target * Decimal(next_drawdown) * self.drawdown_penalty
        )
        next_state = PositionState(
            step=current.step + 1,
            position_fraction=target,
            unrealized_return=(ONE + current.unrealized_return)
            * (ONE + Decimal(next_market_return))
            - ONE,
            drawdown=Decimal(next_drawdown),
            volatility=Decimal(next_volatility),
            spread_fraction=Decimal(next_spread_fraction),
            terminal=bool(final_step or target == ZERO),
        )
        self.state = next_state
        return Transition(
            state=current,
            action=action,
            next_state=next_state,
            reward=reward,
            transaction_cost=transaction_cost,
        )


class BaselinePolicy(StrEnum):
    ALWAYS_HOLD = "ALWAYS_HOLD"
    DRAWDOWN_EXIT = "DRAWDOWN_EXIT"
    VOLATILITY_REDUCER = "VOLATILITY_REDUCER"


def baseline_action(policy: BaselinePolicy, state: PositionState) -> PositionAction:
    if policy is BaselinePolicy.ALWAYS_HOLD:
        return PositionAction.HOLD
    if policy is BaselinePolicy.DRAWDOWN_EXIT:
        return PositionAction.EXIT if state.drawdown >= Decimal("0.05") else PositionAction.HOLD
    if state.volatility >= Decimal("0.08"):
        return PositionAction.EXIT
    if state.volatility >= Decimal("0.04"):
        return PositionAction.REDUCE_50
    if state.volatility >= Decimal("0.025"):
        return PositionAction.REDUCE_25
    return PositionAction.HOLD


class RLEligibilityEvidence(FrozenModel):
    prospective_episode_count: int = Field(ge=0)
    completed_episode_count: int = Field(ge=0)
    distinct_regime_count: int = Field(ge=0)
    baseline_count: int = Field(ge=0)
    multi_seed_count: int = Field(ge=0)
    stress_test_passed: bool = False
    canonical_cost_model_used: bool = False
    point_in_time_inputs_verified: bool = False

    @model_validator(mode="after")
    def completed_not_greater_than_collected(self) -> "RLEligibilityEvidence":
        if self.completed_episode_count > self.prospective_episode_count:
            raise ValueError("completed episodes cannot exceed collected episodes")
        return self


def evaluate_rl_eligibility(evidence: RLEligibilityEvidence) -> dict[str, object]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("gymnasium", "torch", "stable_baselines3")
    }
    checks = {
        "minimum_prospective_episodes": evidence.prospective_episode_count >= 1000,
        "minimum_completed_episodes": evidence.completed_episode_count >= 750,
        "minimum_regimes": evidence.distinct_regime_count >= 3,
        "three_deterministic_baselines": evidence.baseline_count >= 3,
        "minimum_five_seeds": evidence.multi_seed_count >= 5,
        "stress_test_passed": evidence.stress_test_passed,
        "canonical_cost_model_used": evidence.canonical_cost_model_used,
        "point_in_time_inputs_verified": evidence.point_in_time_inputs_verified,
        **{f"dependency_{name}": present for name, present in dependencies.items()},
    }
    eligible = all(checks.values())
    return {
        "status": "RL_TRAINING_ELIGIBLE" if eligible else "RL_TRAINING_BLOCKED",
        "eligible": eligible,
        "authority": "SHADOW_ONLY",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "execution_authority": False,
        "automatic_promotion_permitted": False,
    }


__all__ = [
    "BaselinePolicy",
    "PositionAction",
    "PositionManagementEnvironment",
    "PositionState",
    "RLEligibilityEvidence",
    "Transition",
    "baseline_action",
    "evaluate_rl_eligibility",
]
