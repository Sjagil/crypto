from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from core.contracts import OrderTimeInForce
from portfolio.contracts import ExecutionStyle, InvestmentDirection, InvestmentIntent
from portfolio.targets import (
    build_execution_chain,
    construct_portfolio_target,
    order_intent_from_chain,
    validate_order_against_chain,
)

NOW = datetime(2026, 8, 11, 14, tzinfo=UTC)


def _intent(
    direction: InvestmentDirection = InvestmentDirection.LONG,
    *,
    expected_return: Decimal | None = Decimal("0.03"),
    strategy: str = "strategy-a",
) -> InvestmentIntent:
    return InvestmentIntent.create(
        market="SOL-EUR",
        direction=direction,
        confidence=Decimal("0.8"),
        expected_return=expected_return,
        expected_risk=Decimal("0.01"),
        horizon_seconds=3600,
        strategy_id=strategy,
        family="trend",
        generated_at=NOW,
        valid_until=NOW + timedelta(hours=1),
        evidence_id=f"evidence-{strategy}",
        reason_codes=("NO_EDGE",) if direction is InvestmentDirection.NO_TRADE else (),
    )


def _decision(intents=None):
    return construct_portfolio_target(
        intents or [_intent()],
        current_quantity=Decimal("1"),
        current_notional_eur=Decimal("100"),
        equity_eur=Decimal("1000"),
        mark_price=Decimal("100"),
        proposed_target_weight=Decimal("0.20"),
        risk_budget_eur=Decimal("5"),
        cluster="L1",
        decision_time=NOW,
        expires_at=NOW + timedelta(minutes=30),
        portfolio_state_hash="portfolio-state",
        cost_model_version="cost-v1",
        quantity_step=Decimal("0.01"),
    )


def test_constructs_target_weight_and_delta_from_normalized_intent() -> None:
    decision = _decision()
    assert decision.status == "TARGET_CREATED"
    assert decision.target.target_notional_eur == Decimal("200.00")
    assert decision.target.target_quantity == Decimal("2.00")
    assert decision.target.delta_quantity == Decimal("1.00")
    assert decision.target.source_strategies == ("strategy-a",)


def test_conflict_or_unknown_edge_preserves_current_holding() -> None:
    conflict = _decision(
        [_intent(), _intent(InvestmentDirection.FLAT, strategy="strategy-b")]
    )
    assert conflict.status == "NO_TRADE"
    assert conflict.target.delta_quantity == 0
    assert "CONFLICTING_STRATEGY_DIRECTIONS" in conflict.reason_codes

    unknown = _decision([_intent(expected_return=None)])
    assert unknown.status == "NO_TRADE"
    assert "EXPECTED_NET_EDGE_UNKNOWN" in unknown.reason_codes


def test_target_risk_execution_order_chain_is_exact() -> None:
    target = _decision().target
    chain = build_execution_chain(
        target,
        approved=True,
        approved_quantity=Decimal("1"),
        risk_eur=Decimal("5"),
        risk_reason_codes=("APPROVED",),
        policy_version="risk-v1",
        account_state_hash="account-state",
        approval_time=NOW,
        approval_expires_at=NOW + timedelta(minutes=10),
        execution_style=ExecutionStyle.PASSIVE_LIMIT,
        maximum_notional_eur=Decimal("105"),
        execution_expires_at=NOW + timedelta(minutes=5),
    )
    order = order_intent_from_chain(
        chain,
        idempotency_key="target-order-1",
        strategy_id="strategy-a",
        strategy_dna_hash="dna-a",
        signal_id="signal-a",
        limit_price=Decimal("100"),
        time_in_force=OrderTimeInForce.GTC,
        post_only=True,
    )

    validate_order_against_chain(order, chain)
    assert order.portfolio_target_id == target.target_id
    assert "CANONICAL_PORTFOLIO_TARGET_CHAIN" in order.reason_codes

    tampered = order.model_copy(update={"quantity": Decimal("0.5")})
    with pytest.raises(ValueError, match="QUANTITY_MISMATCH"):
        validate_order_against_chain(tampered, chain)
