"""Research-only reinforcement-learning contracts.

Nothing in this package owns exchange credentials, order submission, portfolio
promotion or live authority.
"""

from rl.position_management import (
    BaselinePolicy,
    PositionAction,
    PositionManagementEnvironment,
    PositionState,
    RLEligibilityEvidence,
    evaluate_rl_eligibility,
)

__all__ = [
    "BaselinePolicy",
    "PositionAction",
    "PositionManagementEnvironment",
    "PositionState",
    "RLEligibilityEvidence",
    "evaluate_rl_eligibility",
]
