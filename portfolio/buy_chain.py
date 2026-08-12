"""Fail-closed construction of canonical BUY execution chains."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any, Mapping

from config.settings import Settings
from core.contracts import ExecutionBlocked, OrderIntent, OrderSide, OrderType
from core.economics import CanonicalCostModel
from portfolio.contracts import ExecutionStyle, InvestmentDirection, InvestmentIntent
from portfolio.targets import (
    CanonicalExecutionChain,
    PortfolioConstructionDecision,
    build_execution_chain,
    construct_portfolio_target,
    order_intent_from_chain,
    validate_order_against_chain,
)
from utils.common import stable_hash, utc_now

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class CanonicalBuyPlan:
    investment_intent: InvestmentIntent
    portfolio_decision: PortfolioConstructionDecision
    chain: CanonicalExecutionChain
    order: OrderIntent


def _quantity_step(*values: Decimal) -> Decimal:
    exponent = min(Decimal(value).as_tuple().exponent for value in values)
    return Decimal("1").scaleb(exponent)


def _execution_style(order: OrderIntent) -> ExecutionStyle:
    if order.order_type is OrderType.MARKET:
        return ExecutionStyle.MARKET_WITHIN_BOUNDS
    if order.order_type is not OrderType.LIMIT:
        raise ExecutionBlocked("canonical BUY chain supports market or limit entry only")
    return (
        ExecutionStyle.PASSIVE_LIMIT
        if order.post_only
        else ExecutionStyle.AGGRESSIVE_LIMIT
    )


def planned_target_net_edge(
    *,
    entry_price: Decimal,
    target_price: Decimal,
    costs: CanonicalCostModel,
) -> Decimal:
    """Return target distance after the canonical conservative round-trip cost."""

    entry = Decimal(entry_price)
    target = Decimal(target_price)
    if entry <= ZERO or target <= entry:
        return ZERO
    return (target - entry) / entry - costs.conservative_roundtrip_fraction


def canonicalize_approved_buy_order(
    settings: Settings,
    order: OrderIntent,
    *,
    mark_price: Decimal,
    current_quantity: Decimal,
    equity_eur: Decimal,
    approved_risk_eur: Decimal,
    expected_net_edge: Decimal,
    confidence: Decimal,
    family: str,
    evidence_id: str,
    policy_version: str,
    account_state: Mapping[str, Any],
    portfolio_state: Mapping[str, Any],
    horizon_seconds: int,
    validity_seconds: int = 300,
) -> CanonicalBuyPlan:
    """Convert a pre-approved native BUY plan into the mandatory chain.

    This function does not approve risk.  It requires concrete upstream risk,
    economics, account and portfolio evidence and blocks on any missing or
    internally inconsistent fact.
    """

    if order.side is not OrderSide.BUY:
        raise ExecutionBlocked("canonical BUY chain received non-BUY order")
    if order.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
        raise ExecutionBlocked("canonical BUY chain received unsupported order type")
    price = max(Decimal(mark_price), order.limit_price or ZERO)
    quantity = Decimal(order.quantity)
    current = Decimal(current_quantity)
    equity = Decimal(equity_eur)
    risk = Decimal(approved_risk_eur)
    edge = Decimal(expected_net_edge)
    selected_confidence = Decimal(confidence)
    if price <= ZERO or quantity <= ZERO or current < ZERO or equity <= ZERO:
        raise ExecutionBlocked("canonical BUY chain received invalid financial state")
    if risk <= ZERO:
        raise ExecutionBlocked("canonical BUY chain requires positive approved risk")
    if edge <= ZERO:
        raise ExecutionBlocked("canonical BUY chain requires positive net edge")
    if not ZERO < selected_confidence <= Decimal("1"):
        raise ExecutionBlocked("canonical BUY chain confidence must be in (0, 1]")
    if not evidence_id or not policy_version or not family:
        raise ExecutionBlocked("canonical BUY chain provenance is incomplete")
    if not account_state or not portfolio_state:
        raise ExecutionBlocked("canonical BUY chain state evidence is missing")
    maximum_notional = order.maximum_notional_eur
    if maximum_notional is None or maximum_notional <= ZERO:
        raise ExecutionBlocked("canonical BUY chain requires an explicit notional cap")
    requested_notional = quantity * price
    if requested_notional > maximum_notional:
        raise ExecutionBlocked("canonical BUY chain exceeds the explicit notional cap")
    target_quantity = current + quantity
    target_notional = target_quantity * price
    if target_notional > equity:
        raise ExecutionBlocked("canonical BUY target exceeds reconciled equity")
    now = utc_now()
    valid_until = now + timedelta(seconds=validity_seconds)
    costs = CanonicalCostModel.from_settings(settings)
    investment = InvestmentIntent.create(
        market=order.market,
        direction=InvestmentDirection.LONG,
        confidence=selected_confidence,
        expected_return=edge,
        expected_risk=risk / requested_notional,
        horizon_seconds=horizon_seconds,
        strategy_id=order.strategy_id,
        family=family,
        generated_at=now,
        valid_until=valid_until,
        evidence_id=evidence_id,
        reason_codes=tuple(
            dict.fromkeys((*order.reason_codes, "UPSTREAM_RISK_AND_EDGE_APPROVED"))
        ),
    )
    portfolio_hash = stable_hash(portfolio_state, length=64)
    account_hash = stable_hash(account_state, length=64)
    target = construct_portfolio_target(
        (investment,),
        current_quantity=current,
        current_notional_eur=current * price,
        equity_eur=equity,
        mark_price=price,
        proposed_target_weight=target_notional / equity,
        risk_budget_eur=risk,
        cluster=None,
        decision_time=now,
        expires_at=valid_until,
        portfolio_state_hash=portfolio_hash,
        cost_model_version=costs.cost_model_version,
        quantity_step=_quantity_step(current, quantity),
    )
    if target.status != "TARGET_CREATED" or target.target.delta_quantity != quantity:
        raise ExecutionBlocked("canonical portfolio construction rejected BUY plan")
    approval_reasons = tuple(
        dict.fromkeys(
            (
                "APPROVED",
                *(
                    getattr(reason, "value", str(reason))
                    for reason in order.reason_codes
                ),
            )
        )
    )
    chain = build_execution_chain(
        target.target,
        approved=True,
        approved_quantity=quantity,
        risk_eur=risk,
        risk_reason_codes=approval_reasons,
        policy_version=policy_version,
        account_state_hash=account_hash,
        approval_time=now,
        approval_expires_at=valid_until,
        execution_style=_execution_style(order),
        maximum_notional_eur=maximum_notional,
        execution_expires_at=valid_until,
    )
    canonical_order = order_intent_from_chain(
        chain,
        idempotency_key=order.idempotency_key,
        strategy_id=order.strategy_id,
        strategy_dna_hash=order.strategy_dna_hash,
        signal_id=order.signal_id,
        limit_price=order.limit_price,
        time_in_force=order.time_in_force,
        post_only=order.post_only,
        cancel_on_disconnect_group=order.cancel_on_disconnect_group,
        reason_codes=order.reason_codes,
    )
    validate_order_against_chain(canonical_order, chain)
    return CanonicalBuyPlan(
        investment_intent=investment,
        portfolio_decision=target,
        chain=chain,
        order=canonical_order,
    )


__all__ = [
    "CanonicalBuyPlan",
    "canonicalize_approved_buy_order",
    "planned_target_net_edge",
]
