"""Fail-closed governance for future AI and machine-learning development.

The current system is a classical quantitative research platform. AI model
development remains embargoed until one immutable strategy has passed every
research and operational stage and has accumulated profitable real-live
evidence. This module deliberately contains no training or inference code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

AI_GOVERNANCE_POLICY_VERSION = "ai_development_embargo_v1"


@dataclass(frozen=True, slots=True)
class AIGovernancePolicy:
    """Minimum evidence required before AI work may even be proposed."""

    minimum_live_calendar_days: int = 180
    minimum_live_closed_trades: int = 30
    minimum_live_regimes: int = 2
    maximum_live_drawdown: float = 0.20

    def __post_init__(self) -> None:
        if self.minimum_live_calendar_days < 90:
            raise ValueError("minimum_live_calendar_days must be at least 90")
        if self.minimum_live_closed_trades < 30:
            raise ValueError("minimum_live_closed_trades must be at least 30")
        if self.minimum_live_regimes < 2:
            raise ValueError("minimum_live_regimes must be at least two")
        if not 0 < self.maximum_live_drawdown < 1:
            raise ValueError("maximum_live_drawdown must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class AIGovernanceEvidence:
    """Explicit, auditable evidence; missing evidence always fails closed."""

    candidate_id: str | None = None
    economic_validation_passed: bool = False
    historical_statistical_gates_passed: bool = False
    forward_validation_passed: bool = False
    shadow_validation_passed: bool = False
    paper_validation_passed: bool = False
    live_trading_active: bool = False
    live_calendar_days: int = 0
    live_closed_trades: int = 0
    live_regime_count: int = 0
    live_net_return_after_costs: float = 0.0
    live_maximum_drawdown: float | None = None
    unresolved_live_incidents: int = 0
    manual_ai_authorization: bool = False
    evidence_manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "AIGovernanceEvidence":
        if payload is None:
            return cls()
        known = {
            field: payload[field]
            for field in cls.__dataclass_fields__
            if field in payload
        }
        return cls(**known)


def evaluate_ai_governance(
    evidence: AIGovernanceEvidence | Mapping[str, Any] | None = None,
    *,
    policy: AIGovernancePolicy | None = None,
) -> dict[str, Any]:
    """Return the complete fail-closed AI eligibility decision."""

    selected_policy = policy or AIGovernancePolicy()
    selected_evidence = (
        evidence
        if isinstance(evidence, AIGovernanceEvidence)
        else AIGovernanceEvidence.from_mapping(evidence)
    )
    drawdown = selected_evidence.live_maximum_drawdown
    checks = {
        "candidate_identity_present": bool(selected_evidence.candidate_id),
        "economic_validation_passed": (
            selected_evidence.economic_validation_passed
        ),
        "historical_statistical_gates_passed": (
            selected_evidence.historical_statistical_gates_passed
        ),
        "forward_validation_passed": (
            selected_evidence.forward_validation_passed
        ),
        "shadow_validation_passed": (
            selected_evidence.shadow_validation_passed
        ),
        "paper_validation_passed": (
            selected_evidence.paper_validation_passed
        ),
        "live_trading_active": selected_evidence.live_trading_active,
        "minimum_live_calendar_days": (
            selected_evidence.live_calendar_days
            >= selected_policy.minimum_live_calendar_days
        ),
        "minimum_live_closed_trades": (
            selected_evidence.live_closed_trades
            >= selected_policy.minimum_live_closed_trades
        ),
        "minimum_live_regime_coverage": (
            selected_evidence.live_regime_count
            >= selected_policy.minimum_live_regimes
        ),
        "live_net_return_after_costs_positive": (
            selected_evidence.live_net_return_after_costs > 0
        ),
        "live_drawdown_within_mandate": (
            drawdown is not None
            and 0 <= abs(float(drawdown))
            <= selected_policy.maximum_live_drawdown
        ),
        "no_unresolved_live_incidents": (
            selected_evidence.unresolved_live_incidents == 0
        ),
        "evidence_manifest_sha256_present": bool(
            selected_evidence.evidence_manifest_sha256
        ),
        "manual_ai_authorization": (
            selected_evidence.manual_ai_authorization
        ),
    }
    eligible = all(checks.values())
    return {
        "policy_version": AI_GOVERNANCE_POLICY_VERSION,
        "status": (
            "AI_DEVELOPMENT_ELIGIBLE"
            if eligible
            else "AI_DEVELOPMENT_EMBARGOED"
        ),
        "eligible": eligible,
        "policy": asdict(selected_policy),
        "evidence": asdict(selected_evidence),
        "checks": checks,
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "blocked_capabilities": (
            []
            if eligible
            else [
                "AI_MODEL_DESIGN",
                "AI_MODEL_TRAINING",
                "AI_HYPERPARAMETER_SEARCH",
                "NEURAL_NETWORK_DEVELOPMENT",
                "MODEL_BASED_SIGNAL_SELECTION",
                "MODEL_INFERENCE_IN_SHADOW_PAPER_OR_LIVE",
            ]
        ),
        "permitted_now": [
            "CLASSICAL_RULE_BASED_RESEARCH",
            "DETERMINISTIC_BACKTESTING",
            "FORWARD_OBSERVATION",
            "SHADOW_PAPER_LIVE_LIFECYCLE_HARDENING",
            "GENERAL_CAUSAL_FEATURE_AUDITING",
        ],
        "automatic_promotion_permitted": False,
    }


def require_ai_development_eligible(
    evidence: AIGovernanceEvidence | Mapping[str, Any] | None = None,
    *,
    policy: AIGovernancePolicy | None = None,
) -> dict[str, Any]:
    """Raise unless every prerequisite and manual authorization is present."""

    decision = evaluate_ai_governance(evidence, policy=policy)
    if not decision["eligible"]:
        failed = ",".join(decision["failed_checks"])
        raise PermissionError(f"AI_DEVELOPMENT_EMBARGOED:{failed}")
    return decision


__all__ = [
    "AI_GOVERNANCE_POLICY_VERSION",
    "AIGovernanceEvidence",
    "AIGovernancePolicy",
    "evaluate_ai_governance",
    "require_ai_development_eligible",
]
