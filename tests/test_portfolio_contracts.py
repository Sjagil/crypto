from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from portfolio.contracts import (
    ExecutionIntent,
    ExecutionStyle,
    InvestmentDirection,
    InvestmentIntent,
    PortfolioTarget,
    PortfolioTargetAction,
    RiskApproval,
)

NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)


def _intent() -> InvestmentIntent:
    return InvestmentIntent.create(
        market="SOL-EUR",
        direction=InvestmentDirection.LONG,
        confidence=Decimal("0.72"),
        expected_return=Decimal("0.025"),
        expected_risk=Decimal("0.011"),
        horizon_seconds=86_400,
        strategy_id="trend_pullback_v1",
        family="trend_pullback",
        generated_at=NOW,
        valid_until=NOW + timedelta(hours=1),
        evidence_id="evidence-1",
    )


def _target(intent: InvestmentIntent) -> PortfolioTarget:
    return PortfolioTarget.create(
        market=intent.market,
        current_quantity=Decimal("1"),
        current_notional_eur=Decimal("140"),
        target_weight=Decimal("0.20"),
        target_notional_eur=Decimal("280"),
        target_quantity=Decimal("2"),
        source_intent_ids=(intent.intent_id,),
        source_strategies=(intent.strategy_id,),
        confidence=intent.confidence,
        expected_net_edge=Decimal("0.018"),
        risk_budget_eur=Decimal("2.80"),
        cluster="L1_ALTCOINS",
        generated_at=NOW,
        expires_at=NOW + timedelta(minutes=30),
        portfolio_state_hash="portfolio-state-1",
        cost_model_version="canonical-cost-v1",
    )


def test_contract_chain_is_deterministic_and_preserves_provenance() -> None:
    intent = _intent()
    assert intent == _intent()
    target = _target(intent)
    assert target == _target(intent)
    assert target.action is PortfolioTargetAction.ADD
    assert target.delta_quantity == Decimal("1")

    approval = RiskApproval.create(
        target_id=target.target_id,
        approved=True,
        approved_delta_quantity=target.delta_quantity,
        risk_eur=Decimal("2.80"),
        reason_codes=("APPROVED",),
        policy_version="risk-v1",
        account_state_hash="account-state-1",
        approved_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )
    execution = ExecutionIntent.create(
        target=target,
        approval=approval,
        style=ExecutionStyle.PASSIVE_LIMIT,
        maximum_notional_eur=Decimal("141"),
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        reason_codes=("TARGET_DELTA_APPROVED",),
    )

    assert execution.target_id == target.target_id
    assert execution.risk_approval_id == approval.approval_id
    assert execution.quantity == Decimal("1")
    assert execution.side == "BUY"


def test_target_delta_and_action_are_invariants() -> None:
    intent = _intent()
    values = _target(intent).model_dump()
    values["delta_quantity"] = Decimal("99")
    with pytest.raises(ValueError, match="delta_quantity"):
        PortfolioTarget.model_validate(values)


def test_blocked_risk_cannot_create_execution_intent() -> None:
    target = _target(_intent())
    blocked = RiskApproval.create(
        target_id=target.target_id,
        approved=False,
        approved_delta_quantity=Decimal("0"),
        risk_eur=None,
        reason_codes=("RECONCILIATION_REQUIRED",),
        policy_version="risk-v1",
        account_state_hash="account-state-1",
        approved_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )
    with pytest.raises(ValueError, match="blocked risk"):
        ExecutionIntent.create(
            target=target,
            approval=blocked,
            style=ExecutionStyle.PASSIVE_LIMIT,
            maximum_notional_eur=Decimal("141"),
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=5),
            reason_codes=("SHOULD_NOT_EXECUTE",),
        )


def test_wait_advice_is_not_an_executable_intent() -> None:
    target = _target(_intent())
    approval = RiskApproval.create(
        target_id=target.target_id,
        approved=True,
        approved_delta_quantity=target.delta_quantity,
        risk_eur=Decimal("2.80"),
        reason_codes=("APPROVED",),
        policy_version="risk-v1",
        account_state_hash="account-state-1",
        approved_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )
    with pytest.raises(ValueError, match="WAIT"):
        ExecutionIntent.create(
            target=target,
            approval=approval,
            style=ExecutionStyle.WAIT,
            maximum_notional_eur=Decimal("141"),
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=5),
            reason_codes=("WAIT_FOR_SPREAD",),
        )
