"""Native intent -> target -> risk -> execution contracts.

The design is informed by LEAN's separation of alpha, portfolio construction,
risk and execution, but the implementation and schemas are native to this
repository.  These objects do not submit orders and have no exchange authority.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from core.contracts import FrozenModel, OrderSide, normalize_market, require_utc
from utils.common import stable_hash, utc_now

ZERO = Decimal("0")


class InvestmentDirection(StrEnum):
    LONG = "LONG"
    FLAT = "FLAT"
    REDUCE = "REDUCE"
    NO_TRADE = "NO_TRADE"


class PortfolioTargetAction(StrEnum):
    ENTRY = "ENTRY"
    ADD = "ADD"
    REDUCE = "REDUCE"
    ROTATE = "ROTATE"
    FULL_EXIT = "FULL_EXIT"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


class ExecutionStyle(StrEnum):
    WAIT = "WAIT"
    PASSIVE_LIMIT = "PASSIVE_LIMIT"
    AGGRESSIVE_LIMIT = "AGGRESSIVE_LIMIT"
    MARKET_WITHIN_BOUNDS = "MARKET_WITHIN_BOUNDS"


def _identifier(prefix: str, payload: Any) -> str:
    return f"{prefix}_{stable_hash(payload, length=40)}"


def classify_target_action(
    current_quantity: Decimal,
    target_quantity: Decimal,
    *,
    rotation: bool = False,
) -> PortfolioTargetAction:
    if current_quantity < ZERO or target_quantity < ZERO:
        raise ValueError("spot quantities cannot be negative")
    if current_quantity == target_quantity:
        return PortfolioTargetAction.HOLD if current_quantity > ZERO else PortfolioTargetAction.NO_TRADE
    if rotation:
        return PortfolioTargetAction.ROTATE
    if current_quantity == ZERO and target_quantity > ZERO:
        return PortfolioTargetAction.ENTRY
    if target_quantity == ZERO:
        return PortfolioTargetAction.FULL_EXIT
    if target_quantity > current_quantity:
        return PortfolioTargetAction.ADD
    return PortfolioTargetAction.REDUCE


class InvestmentIntent(FrozenModel):
    """Normalized strategy output without order quantity or execution authority."""

    intent_id: str
    market: str
    direction: InvestmentDirection
    confidence: Decimal = Field(ge=ZERO, le=Decimal("1"))
    expected_return: Decimal | None = None
    expected_risk: Decimal | None = Field(default=None, ge=ZERO)
    horizon_seconds: int = Field(gt=0)
    strategy_id: str
    family: str
    generated_at: datetime
    valid_until: datetime
    evidence_id: str
    reason_codes: tuple[str, ...] = ()

    _market = field_validator("market")(normalize_market)
    _generated_at = field_validator("generated_at")(require_utc)
    _valid_until = field_validator("valid_until")(require_utc)

    @field_validator("intent_id", "strategy_id", "family", "evidence_id")
    @classmethod
    def required_identifier(cls, value: str) -> str:
        selected = value.strip()
        if not selected:
            raise ValueError("intent identifiers cannot be empty")
        return selected

    @model_validator(mode="after")
    def validate_window(self) -> "InvestmentIntent":
        if self.valid_until <= self.generated_at:
            raise ValueError("investment intent must expire after generation")
        if self.direction is InvestmentDirection.NO_TRADE and not self.reason_codes:
            raise ValueError("NO_TRADE intent requires reason codes")
        return self

    @classmethod
    def create(
        cls,
        *,
        market: str,
        direction: InvestmentDirection,
        confidence: Decimal,
        expected_return: Decimal | None,
        expected_risk: Decimal | None,
        horizon_seconds: int,
        strategy_id: str,
        family: str,
        generated_at: datetime,
        valid_until: datetime,
        evidence_id: str,
        reason_codes: tuple[str, ...] = (),
    ) -> "InvestmentIntent":
        identity = {
            "market": normalize_market(market),
            "direction": direction,
            "confidence": str(confidence),
            "expected_return": str(expected_return) if expected_return is not None else None,
            "expected_risk": str(expected_risk) if expected_risk is not None else None,
            "horizon_seconds": horizon_seconds,
            "strategy_id": strategy_id,
            "family": family,
            "generated_at": require_utc(generated_at).isoformat(),
            "valid_until": require_utc(valid_until).isoformat(),
            "evidence_id": evidence_id,
            "reason_codes": reason_codes,
        }
        return cls(intent_id=_identifier("investment_intent", identity), **identity)


class PortfolioTarget(FrozenModel):
    """Desired canonical spot holding; never an order instruction by itself."""

    target_id: str
    market: str
    current_quantity: Decimal = Field(ge=ZERO)
    current_notional_eur: Decimal = Field(ge=ZERO)
    target_weight: Decimal = Field(ge=ZERO, le=Decimal("1"))
    target_notional_eur: Decimal = Field(ge=ZERO)
    target_quantity: Decimal = Field(ge=ZERO)
    delta_quantity: Decimal
    source_intent_ids: tuple[str, ...]
    source_strategies: tuple[str, ...]
    confidence: Decimal = Field(ge=ZERO, le=Decimal("1"))
    expected_net_edge: Decimal | None = None
    risk_budget_eur: Decimal = Field(ge=ZERO)
    cluster: str | None = None
    action: PortfolioTargetAction
    generated_at: datetime
    expires_at: datetime
    portfolio_state_hash: str
    cost_model_version: str
    reason_codes: tuple[str, ...] = ()

    _market = field_validator("market")(normalize_market)
    _generated_at = field_validator("generated_at")(require_utc)
    _expires_at = field_validator("expires_at")(require_utc)

    @field_validator(
        "target_id",
        "portfolio_state_hash",
        "cost_model_version",
    )
    @classmethod
    def required_identifier(cls, value: str) -> str:
        selected = value.strip()
        if not selected:
            raise ValueError("target identifiers cannot be empty")
        return selected

    @model_validator(mode="after")
    def validate_target(self) -> "PortfolioTarget":
        if self.expires_at <= self.generated_at:
            raise ValueError("portfolio target must expire after generation")
        if not self.source_intent_ids or not self.source_strategies:
            raise ValueError("portfolio target requires source intent and strategy provenance")
        if len(set(self.source_intent_ids)) != len(self.source_intent_ids):
            raise ValueError("source intent IDs must be unique")
        expected_delta = self.target_quantity - self.current_quantity
        if self.delta_quantity != expected_delta:
            raise ValueError("delta_quantity must equal target minus current quantity")
        expected_action = classify_target_action(
            self.current_quantity,
            self.target_quantity,
            rotation=self.action is PortfolioTargetAction.ROTATE,
        )
        if self.action is not expected_action:
            raise ValueError("portfolio target action is inconsistent with quantities")
        if self.target_quantity == ZERO and self.target_notional_eur != ZERO:
            raise ValueError("zero target quantity requires zero target notional")
        return self

    @classmethod
    def create(
        cls,
        *,
        market: str,
        current_quantity: Decimal,
        current_notional_eur: Decimal,
        target_weight: Decimal,
        target_notional_eur: Decimal,
        target_quantity: Decimal,
        source_intent_ids: tuple[str, ...],
        source_strategies: tuple[str, ...],
        confidence: Decimal,
        expected_net_edge: Decimal | None,
        risk_budget_eur: Decimal,
        cluster: str | None,
        generated_at: datetime,
        expires_at: datetime,
        portfolio_state_hash: str,
        cost_model_version: str,
        rotation: bool = False,
        reason_codes: tuple[str, ...] = (),
    ) -> "PortfolioTarget":
        current_quantity = Decimal(current_quantity)
        target_quantity = Decimal(target_quantity)
        action = classify_target_action(
            current_quantity,
            target_quantity,
            rotation=rotation,
        )
        values = {
            "market": normalize_market(market),
            "current_quantity": current_quantity,
            "current_notional_eur": Decimal(current_notional_eur),
            "target_weight": Decimal(target_weight),
            "target_notional_eur": Decimal(target_notional_eur),
            "target_quantity": target_quantity,
            "delta_quantity": target_quantity - current_quantity,
            "source_intent_ids": tuple(sorted(source_intent_ids)),
            "source_strategies": tuple(sorted(source_strategies)),
            "confidence": Decimal(confidence),
            "expected_net_edge": (
                Decimal(expected_net_edge) if expected_net_edge is not None else None
            ),
            "risk_budget_eur": Decimal(risk_budget_eur),
            "cluster": cluster,
            "action": action,
            "generated_at": require_utc(generated_at),
            "expires_at": require_utc(expires_at),
            "portfolio_state_hash": portfolio_state_hash,
            "cost_model_version": cost_model_version,
            "reason_codes": reason_codes,
        }
        identity = {
            key: (
                value.isoformat()
                if isinstance(value, datetime)
                else str(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in values.items()
        }
        return cls(target_id=_identifier("portfolio_target", identity), **values)


class RiskApproval(FrozenModel):
    approval_id: str
    target_id: str
    approved: bool
    approved_delta_quantity: Decimal
    risk_eur: Decimal | None = Field(default=None, ge=ZERO)
    reason_codes: tuple[str, ...]
    policy_version: str
    account_state_hash: str
    approved_at: datetime
    expires_at: datetime

    _approved_at = field_validator("approved_at")(require_utc)
    _expires_at = field_validator("expires_at")(require_utc)

    @model_validator(mode="after")
    def validate_approval(self) -> "RiskApproval":
        if self.expires_at <= self.approved_at:
            raise ValueError("risk approval must expire after approval time")
        if self.approved:
            if self.approved_delta_quantity == ZERO:
                raise ValueError("approved target requires a non-zero approved delta")
            if "APPROVED" not in self.reason_codes:
                raise ValueError("approved risk decision requires APPROVED reason")
        else:
            if self.approved_delta_quantity != ZERO:
                raise ValueError("blocked risk decision cannot approve quantity")
            if not self.reason_codes:
                raise ValueError("blocked risk decision requires reason codes")
        return self

    @classmethod
    def create(
        cls,
        *,
        target_id: str,
        approved: bool,
        approved_delta_quantity: Decimal,
        risk_eur: Decimal | None,
        reason_codes: tuple[str, ...],
        policy_version: str,
        account_state_hash: str,
        approved_at: datetime,
        expires_at: datetime,
    ) -> "RiskApproval":
        values = {
            "target_id": target_id,
            "approved": approved,
            "approved_delta_quantity": Decimal(approved_delta_quantity),
            "risk_eur": Decimal(risk_eur) if risk_eur is not None else None,
            "reason_codes": reason_codes,
            "policy_version": policy_version,
            "account_state_hash": account_state_hash,
            "approved_at": require_utc(approved_at),
            "expires_at": require_utc(expires_at),
        }
        identity = {
            key: (
                value.isoformat()
                if isinstance(value, datetime)
                else str(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in values.items()
        }
        return cls(approval_id=_identifier("risk_approval", identity), **values)


class ExecutionIntent(FrozenModel):
    execution_intent_id: str
    target_id: str
    risk_approval_id: str
    market: str
    side: OrderSide
    quantity: Decimal = Field(gt=ZERO)
    style: ExecutionStyle
    maximum_notional_eur: Decimal = Field(gt=ZERO)
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    reason_codes: tuple[str, ...]

    _market = field_validator("market")(normalize_market)
    _created_at = field_validator("created_at")(require_utc)
    _expires_at = field_validator("expires_at")(require_utc)

    @model_validator(mode="after")
    def validate_execution_intent(self) -> "ExecutionIntent":
        if self.expires_at <= self.created_at:
            raise ValueError("execution intent must expire after creation")
        if self.style is ExecutionStyle.WAIT:
            raise ValueError("WAIT is advice, not an executable intent")
        if not self.target_id or not self.risk_approval_id:
            raise ValueError("execution requires target and risk approval provenance")
        return self

    @classmethod
    def create(
        cls,
        *,
        target: PortfolioTarget,
        approval: RiskApproval,
        style: ExecutionStyle,
        maximum_notional_eur: Decimal,
        created_at: datetime,
        expires_at: datetime,
        reason_codes: tuple[str, ...],
    ) -> "ExecutionIntent":
        if not approval.approved:
            raise ValueError("blocked risk approval cannot create execution intent")
        if approval.target_id != target.target_id:
            raise ValueError("risk approval does not belong to portfolio target")
        if approval.expires_at < expires_at:
            raise ValueError("execution intent cannot outlive risk approval")
        delta = approval.approved_delta_quantity
        values = {
            "target_id": target.target_id,
            "risk_approval_id": approval.approval_id,
            "market": target.market,
            "side": OrderSide.BUY if delta > ZERO else OrderSide.SELL,
            "quantity": abs(delta),
            "style": style,
            "maximum_notional_eur": Decimal(maximum_notional_eur),
            "created_at": require_utc(created_at),
            "expires_at": require_utc(expires_at),
            "reason_codes": reason_codes,
        }
        identity = {
            key: (
                value.isoformat()
                if isinstance(value, datetime)
                else str(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in values.items()
        }
        return cls(
            execution_intent_id=_identifier("execution_intent", identity),
            **values,
        )


__all__ = [
    "ExecutionIntent",
    "ExecutionStyle",
    "InvestmentDirection",
    "InvestmentIntent",
    "PortfolioTarget",
    "PortfolioTargetAction",
    "RiskApproval",
    "classify_target_action",
]
