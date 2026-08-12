from __future__ import annotations

from decimal import Decimal

import pytest

from core.contracts import ExecutionBlocked, OrderIntent, OrderSide, OrderType
from core.economics import CanonicalCostModel
from portfolio.buy_chain import (
    canonicalize_approved_buy_order,
    planned_target_net_edge,
)
from portfolio.targets import validate_order_against_chain


def _order() -> OrderIntent:
    return OrderIntent(
        intent_id="legacy-entry",
        idempotency_key="entry:one",
        market="BTC-EUR",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.001"),
        strategy_id="strategy-one",
        strategy_dna_hash="a" * 64,
        signal_id="signal-one",
        maximum_notional_eur=Decimal("60"),
        reason_codes=("CENTRAL_RISK_APPROVED",),
    )


def test_canonical_buy_chain_preserves_order_and_provenance(isolated_settings) -> None:
    plan = canonicalize_approved_buy_order(
        isolated_settings,
        _order(),
        mark_price=Decimal("50000"),
        current_quantity=Decimal("0"),
        equity_eur=Decimal("1000"),
        approved_risk_eur=Decimal("1"),
        expected_net_edge=Decimal("0.01"),
        confidence=Decimal("0.7"),
        family="MOMENTUM",
        evidence_id="evidence-one",
        policy_version="risk-v1",
        account_state={"equity": "1000", "reconciled": True},
        portfolio_state={"positions": 0, "exposure": "0"},
        horizon_seconds=3600,
    )

    assert plan.order.quantity == Decimal("0.001")
    assert plan.order.portfolio_target_id == plan.chain.target.target_id
    assert plan.order.risk_approval_id == plan.chain.approval.approval_id
    assert plan.order.execution_intent_id == plan.chain.execution.execution_intent_id
    assert plan.portfolio_decision.status == "TARGET_CREATED"
    validate_order_against_chain(plan.order, plan.chain)


@pytest.mark.parametrize(
    ("edge", "risk", "match"),
    [
        (Decimal("0"), Decimal("1"), "positive net edge"),
        (Decimal("0.01"), Decimal("0"), "positive approved risk"),
    ],
)
def test_canonical_buy_chain_rejects_missing_economic_or_risk_evidence(
    isolated_settings,
    edge: Decimal,
    risk: Decimal,
    match: str,
) -> None:
    with pytest.raises(ExecutionBlocked, match=match):
        canonicalize_approved_buy_order(
            isolated_settings,
            _order(),
            mark_price=Decimal("50000"),
            current_quantity=Decimal("0"),
            equity_eur=Decimal("1000"),
            approved_risk_eur=risk,
            expected_net_edge=edge,
            confidence=Decimal("0.7"),
            family="MOMENTUM",
            evidence_id="evidence-one",
            policy_version="risk-v1",
            account_state={"equity": "1000"},
            portfolio_state={"positions": 0},
            horizon_seconds=3600,
        )


def test_planned_target_edge_includes_canonical_roundtrip_costs() -> None:
    costs = CanonicalCostModel.create(
        maker_fee_fraction=0.0015,
        taker_fee_fraction=0.0025,
        spread_bps=5,
        slippage_bps=8,
    )
    edge = planned_target_net_edge(
        entry_price=Decimal("100"),
        target_price=Decimal("102"),
        costs=costs,
    )
    assert edge == Decimal("0.0129")
