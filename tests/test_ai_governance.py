from __future__ import annotations

import pytest

from core.ai_governance import (
    AIGovernanceEvidence,
    evaluate_ai_governance,
    require_ai_development_eligible,
)
from core.cli import build_parser


def _complete_evidence() -> AIGovernanceEvidence:
    return AIGovernanceEvidence(
        candidate_id="LIVE_STRATEGY_V1",
        economic_validation_passed=True,
        historical_statistical_gates_passed=True,
        forward_validation_passed=True,
        shadow_validation_passed=True,
        paper_validation_passed=True,
        live_trading_active=True,
        live_calendar_days=180,
        live_closed_trades=30,
        live_regime_count=2,
        live_net_return_after_costs=0.08,
        live_maximum_drawdown=-0.12,
        unresolved_live_incidents=0,
        manual_ai_authorization=True,
        evidence_manifest_sha256="a" * 64,
    )


def test_ai_development_is_embargoed_without_live_evidence():
    decision = evaluate_ai_governance()

    assert decision["status"] == "AI_DEVELOPMENT_EMBARGOED"
    assert decision["eligible"] is False
    assert "AI_MODEL_TRAINING" in decision["blocked_capabilities"]
    assert "historical_statistical_gates_passed" in decision["failed_checks"]
    assert "live_trading_active" in decision["failed_checks"]
    assert decision["automatic_promotion_permitted"] is False

    args = build_parser().parse_args(["lab", "ai", "status"])
    assert args.lab_section == "ai"
    assert args.lab_action == "status"


def test_historical_or_paper_success_alone_cannot_open_ai_gate():
    evidence = _complete_evidence()
    incomplete = AIGovernanceEvidence(
        **{
            **{
                name: getattr(evidence, name)
                for name in evidence.__dataclass_fields__
            },
            "live_trading_active": False,
            "live_calendar_days": 0,
            "live_closed_trades": 0,
        }
    )

    decision = evaluate_ai_governance(incomplete)

    assert decision["eligible"] is False
    with pytest.raises(PermissionError, match="AI_DEVELOPMENT_EMBARGOED"):
        require_ai_development_eligible(incomplete)


def test_ai_gate_requires_complete_live_evidence_and_manual_authorization():
    decision = require_ai_development_eligible(_complete_evidence())

    assert decision["status"] == "AI_DEVELOPMENT_ELIGIBLE"
    assert decision["eligible"] is True
    assert decision["failed_checks"] == []
    assert decision["blocked_capabilities"] == []


def test_autopilot_feature_store_is_explicit_opt_in():
    parser = build_parser()
    default = parser.parse_args(["lab", "campaign", "autopilot"])
    opted_in = parser.parse_args(
        ["lab", "campaign", "autopilot", "--build-feature-store"]
    )

    assert default.build_feature_store is False
    assert opted_in.build_feature_store is True
