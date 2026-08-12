"""Deterministic construction and validation of canonical portfolio targets."""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Sequence

from pydantic import model_validator

from core.contracts import FrozenModel, OrderIntent, OrderTimeInForce, OrderType
from portfolio.contracts import (
    ExecutionIntent,
    ExecutionStyle,
    InvestmentDirection,
    InvestmentIntent,
    PortfolioTarget,
    RiskApproval,
)

ZERO = Decimal("0")


class PortfolioConstructionDecision(FrozenModel):
    target: PortfolioTarget
    status: str
    reason_codes: tuple[str, ...]


class CanonicalExecutionChain(FrozenModel):
    target: PortfolioTarget
    approval: RiskApproval
    execution: ExecutionIntent

    @model_validator(mode="after")
    def validate_chain(self) -> "CanonicalExecutionChain":
        if self.approval.target_id != self.target.target_id:
            raise ValueError("risk approval target mismatch")
        if self.execution.target_id != self.target.target_id:
            raise ValueError("execution target mismatch")
        if self.execution.risk_approval_id != self.approval.approval_id:
            raise ValueError("execution risk approval mismatch")
        if self.execution.market != self.target.market:
            raise ValueError("execution market mismatch")
        return self


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= ZERO:
        raise ValueError("quantity step must be positive")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def construct_portfolio_target(
    intents: Sequence[InvestmentIntent],
    *,
    current_quantity: Decimal,
    current_notional_eur: Decimal,
    equity_eur: Decimal,
    mark_price: Decimal,
    proposed_target_weight: Decimal,
    risk_budget_eur: Decimal,
    cluster: str | None,
    decision_time: datetime,
    expires_at: datetime,
    portfolio_state_hash: str,
    cost_model_version: str,
    quantity_step: Decimal = Decimal("0.00000001"),
    rotation: bool = False,
) -> PortfolioConstructionDecision:
    """Resolve same-market strategy intents into one desired holding.

    Conflict, expiry, non-positive net edge, or NO_TRADE evidence preserves the
    current holding.  This function never creates an order.
    """

    if not intents:
        raise ValueError("portfolio construction requires at least one intent")
    markets = {intent.market for intent in intents}
    if len(markets) != 1:
        raise ValueError("portfolio target cannot combine different markets")
    if equity_eur <= ZERO or mark_price <= ZERO:
        raise ValueError("equity and mark price must be positive")
    if current_quantity < ZERO or current_notional_eur < ZERO:
        raise ValueError("current spot holdings cannot be negative")
    if proposed_target_weight < ZERO or proposed_target_weight > Decimal("1"):
        raise ValueError("target weight must be between zero and one")

    reasons: list[str] = []
    valid = [
        intent
        for intent in intents
        if intent.generated_at <= decision_time < intent.valid_until
    ]
    if len(valid) != len(intents):
        reasons.append("STALE_OR_FUTURE_INTENT")
    directions = {intent.direction for intent in valid}
    if InvestmentDirection.NO_TRADE in directions:
        reasons.append("NO_TRADE_INTENT_PRESENT")
    active_directions = directions - {InvestmentDirection.NO_TRADE}
    if InvestmentDirection.LONG in active_directions and active_directions & {
        InvestmentDirection.FLAT,
        InvestmentDirection.REDUCE,
    }:
        reasons.append("CONFLICTING_STRATEGY_DIRECTIONS")

    weighted_edges = [
        intent.expected_return * intent.confidence
        for intent in valid
        if intent.expected_return is not None
    ]
    expected_net_edge = (
        sum(weighted_edges, ZERO) / Decimal(len(weighted_edges))
        if weighted_edges
        else None
    )
    if expected_net_edge is None:
        reasons.append("EXPECTED_NET_EDGE_UNKNOWN")
    elif expected_net_edge <= ZERO and InvestmentDirection.LONG in active_directions:
        reasons.append("EXPECTED_NET_EDGE_NOT_POSITIVE")

    blocked = bool(reasons)
    if not valid:
        blocked = True
    if blocked:
        target_weight = current_notional_eur / equity_eur
        target_notional = current_notional_eur
        target_quantity = current_quantity
        status = "NO_TRADE"
    elif active_directions <= {InvestmentDirection.FLAT}:
        target_weight = ZERO
        target_notional = ZERO
        target_quantity = ZERO
        status = "TARGET_CREATED"
    elif active_directions <= {InvestmentDirection.REDUCE}:
        target_weight = min(
            proposed_target_weight,
            current_notional_eur / equity_eur,
        )
        target_notional = equity_eur * target_weight
        target_quantity = min(
            current_quantity,
            _floor_step(target_notional / mark_price, quantity_step),
        )
        status = "TARGET_CREATED"
    else:
        target_weight = proposed_target_weight
        target_notional = equity_eur * target_weight
        target_quantity = _floor_step(target_notional / mark_price, quantity_step)
        status = "TARGET_CREATED"

    if not reasons:
        reasons.append("PORTFOLIO_TARGET_CONSTRUCTED")
    confidence = min((intent.confidence for intent in valid), default=ZERO)
    target = PortfolioTarget.create(
        market=next(iter(markets)),
        current_quantity=current_quantity,
        current_notional_eur=current_notional_eur,
        target_weight=target_weight,
        target_notional_eur=target_notional,
        target_quantity=target_quantity,
        source_intent_ids=tuple(intent.intent_id for intent in intents),
        source_strategies=tuple(intent.strategy_id for intent in intents),
        confidence=confidence,
        expected_net_edge=expected_net_edge,
        risk_budget_eur=risk_budget_eur,
        cluster=cluster,
        generated_at=decision_time,
        expires_at=expires_at,
        portfolio_state_hash=portfolio_state_hash,
        cost_model_version=cost_model_version,
        rotation=rotation and not blocked,
        reason_codes=tuple(sorted(set(reasons))),
    )
    return PortfolioConstructionDecision(
        target=target,
        status=status,
        reason_codes=target.reason_codes,
    )


def build_execution_chain(
    target: PortfolioTarget,
    *,
    approved: bool,
    approved_quantity: Decimal,
    risk_eur: Decimal | None,
    risk_reason_codes: tuple[str, ...],
    policy_version: str,
    account_state_hash: str,
    approval_time: datetime,
    approval_expires_at: datetime,
    execution_style: ExecutionStyle,
    maximum_notional_eur: Decimal,
    execution_expires_at: datetime,
) -> CanonicalExecutionChain:
    requested_delta = target.delta_quantity
    approved_absolute = min(abs(requested_delta), Decimal(approved_quantity))
    signed_approved = (
        approved_absolute
        if requested_delta > ZERO
        else -approved_absolute
        if requested_delta < ZERO
        else ZERO
    )
    approval = RiskApproval.create(
        target_id=target.target_id,
        approved=approved,
        approved_delta_quantity=signed_approved if approved else ZERO,
        risk_eur=risk_eur,
        reason_codes=risk_reason_codes,
        policy_version=policy_version,
        account_state_hash=account_state_hash,
        approved_at=approval_time,
        expires_at=approval_expires_at,
    )
    execution = ExecutionIntent.create(
        target=target,
        approval=approval,
        style=execution_style,
        maximum_notional_eur=maximum_notional_eur,
        created_at=approval_time,
        expires_at=execution_expires_at,
        reason_codes=("TARGET_DELTA_RISK_APPROVED",),
    )
    return CanonicalExecutionChain(
        target=target,
        approval=approval,
        execution=execution,
    )


def order_intent_from_chain(
    chain: CanonicalExecutionChain,
    *,
    idempotency_key: str,
    strategy_id: str,
    strategy_dna_hash: str | None,
    signal_id: str | None,
    limit_price: Decimal | None = None,
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
    post_only: bool = False,
    cancel_on_disconnect_group: str | None = None,
    reason_codes: tuple[str, ...] = (),
) -> OrderIntent:
    style = chain.execution.style
    order_type = (
        OrderType.MARKET
        if style is ExecutionStyle.MARKET_WITHIN_BOUNDS
        else OrderType.LIMIT
    )
    if order_type is OrderType.LIMIT and limit_price is None:
        raise ValueError("limit execution style requires limit price")
    return OrderIntent(
        intent_id=chain.execution.execution_intent_id,
        idempotency_key=idempotency_key,
        market=chain.execution.market,
        side=chain.execution.side,
        order_type=order_type,
        quantity=chain.execution.quantity,
        limit_price=limit_price,
        time_in_force=time_in_force,
        post_only=post_only,
        cancel_on_disconnect_group=cancel_on_disconnect_group,
        strategy_id=strategy_id,
        strategy_dna_hash=strategy_dna_hash,
        signal_id=signal_id,
        portfolio_decision_id=chain.target.target_id,
        portfolio_target_id=chain.target.target_id,
        risk_approval_id=chain.approval.approval_id,
        execution_intent_id=chain.execution.execution_intent_id,
        maximum_notional_eur=chain.execution.maximum_notional_eur,
        reason_codes=tuple(
            dict.fromkeys((*reason_codes, "CANONICAL_PORTFOLIO_TARGET_CHAIN"))
        ),
    )


def validate_order_against_chain(
    order: OrderIntent,
    chain: CanonicalExecutionChain,
) -> None:
    expected = chain.execution
    failures: list[str] = []
    if order.portfolio_target_id != chain.target.target_id:
        failures.append("PORTFOLIO_TARGET_ID_MISMATCH")
    if order.risk_approval_id != chain.approval.approval_id:
        failures.append("RISK_APPROVAL_ID_MISMATCH")
    if order.execution_intent_id != expected.execution_intent_id:
        failures.append("EXECUTION_INTENT_ID_MISMATCH")
    if order.market != expected.market:
        failures.append("MARKET_MISMATCH")
    if order.side is not expected.side:
        failures.append("SIDE_MISMATCH")
    if order.quantity != expected.quantity:
        failures.append("QUANTITY_MISMATCH")
    if not chain.approval.approved:
        failures.append("RISK_NOT_APPROVED")
    if failures:
        raise ValueError("invalid canonical execution chain: " + ",".join(failures))


__all__ = [
    "CanonicalExecutionChain",
    "PortfolioConstructionDecision",
    "build_execution_chain",
    "construct_portfolio_target",
    "order_intent_from_chain",
    "validate_order_against_chain",
]
