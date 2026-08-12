"""Typed, point-in-time market-intelligence synthesis with zero order authority.

TradingAgents informed the separation of bullish, bearish and risk viewpoints.
This native implementation consumes already-collected evidence only; it does
not call an LLM, scrape data, size a position, or submit an order.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from core.contracts import FrozenModel, normalize_market, require_utc
from portfolio.contracts import InvestmentDirection, InvestmentIntent
from utils.common import stable_hash

ZERO = Decimal("0")
ONE = Decimal("1")


class IntelligencePerspective(StrEnum):
    BULL = "BULL"
    BEAR = "BEAR"
    RISK = "RISK"


class IntelligenceClaim(FrozenModel):
    claim_id: str
    perspective: IntelligencePerspective
    thesis: str
    signed_score: Decimal = Field(ge=Decimal("-1"), le=ONE)
    confidence: Decimal = Field(ge=ZERO, le=ONE)
    event_time: datetime
    available_at: datetime
    source_ids: tuple[str, ...]
    point_in_time: bool = True

    _event = field_validator("event_time")(require_utc)
    _available = field_validator("available_at")(require_utc)

    @model_validator(mode="after")
    def causal_and_sourced(self) -> "IntelligenceClaim":
        if self.available_at < self.event_time:
            raise ValueError("claim cannot be available before its event")
        if not self.thesis.strip() or not self.source_ids:
            raise ValueError("claim requires a thesis and source provenance")
        if not self.point_in_time:
            raise ValueError("non-point-in-time intelligence is not admissible")
        return self

    @classmethod
    def create(
        cls,
        *,
        perspective: IntelligencePerspective,
        thesis: str,
        signed_score: Decimal,
        confidence: Decimal,
        event_time: datetime,
        available_at: datetime,
        source_ids: tuple[str, ...],
    ) -> "IntelligenceClaim":
        values = {
            "perspective": perspective,
            "thesis": thesis.strip(),
            "signed_score": Decimal(signed_score),
            "confidence": Decimal(confidence),
            "event_time": require_utc(event_time),
            "available_at": require_utc(available_at),
            "source_ids": tuple(sorted(source_ids)),
        }
        identity = {
            **values,
            "event_time": values["event_time"].isoformat(),
            "available_at": values["available_at"].isoformat(),
            "signed_score": str(values["signed_score"]),
            "confidence": str(values["confidence"]),
            "perspective": perspective.value,
        }
        return cls(claim_id=f"intelligence_claim_{stable_hash(identity, length=40)}", **values)


class StructuredMarketIntelligence(FrozenModel):
    snapshot_id: str
    market: str
    decision_time: datetime
    claims: tuple[IntelligenceClaim, ...]
    weighted_score: Decimal = Field(ge=Decimal("-1"), le=ONE)
    confidence: Decimal = Field(ge=ZERO, le=ONE)
    complete_perspectives: bool
    conflict_detected: bool
    evidence_ids: tuple[str, ...]
    authority: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    execution_authority: Literal[False] = False

    _market = field_validator("market")(normalize_market)
    _decision = field_validator("decision_time")(require_utc)

    @model_validator(mode="after")
    def causal_snapshot(self) -> "StructuredMarketIntelligence":
        if not self.claims:
            raise ValueError("intelligence snapshot requires claims")
        if any(claim.available_at > self.decision_time for claim in self.claims):
            raise ValueError("future intelligence is not admissible")
        if tuple(sorted(self.evidence_ids)) != tuple(sorted(c.claim_id for c in self.claims)):
            raise ValueError("evidence IDs must match claims")
        return self


class AIDecisionSnapshot(FrozenModel):
    decision_id: str
    market: str
    observed_direction: InvestmentDirection
    confidence: Decimal = Field(ge=ZERO, le=ONE)
    expected_return: Decimal | None
    expected_risk: Decimal | None = Field(default=None, ge=ZERO)
    horizon_seconds: int = Field(gt=0)
    generated_at: datetime
    evidence_snapshot_id: str
    model_or_policy_id: str
    reason_codes: tuple[str, ...]
    authority: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    live_decision_influence: Literal[False] = False
    execution_authority: Literal[False] = False

    _market = field_validator("market")(normalize_market)
    _generated = field_validator("generated_at")(require_utc)


def synthesize_market_intelligence(
    *,
    market: str,
    decision_time: datetime,
    claims: tuple[IntelligenceClaim, ...],
) -> StructuredMarketIntelligence:
    decision_time = require_utc(decision_time)
    admissible = tuple(sorted(claims, key=lambda item: item.claim_id))
    if any(item.available_at > decision_time for item in admissible):
        raise ValueError("claim available after decision time")
    perspectives = {item.perspective for item in admissible}
    complete = perspectives == set(IntelligencePerspective)
    total_weight = sum((item.confidence for item in admissible), ZERO)
    weighted = (
        sum((item.signed_score * item.confidence for item in admissible), ZERO) / total_weight
        if total_weight > ZERO
        else ZERO
    )
    bull_strength = max(
        (
            item.confidence * max(ZERO, item.signed_score)
            for item in admissible
            if item.perspective is IntelligencePerspective.BULL
        ),
        default=ZERO,
    )
    bear_strength = max(
        (
            item.confidence * abs(min(ZERO, item.signed_score))
            for item in admissible
            if item.perspective is IntelligencePerspective.BEAR
        ),
        default=ZERO,
    )
    conflict = bull_strength >= Decimal("0.60") and bear_strength >= Decimal("0.60")
    confidence = min(ONE, total_weight / Decimal("3")) if complete and not conflict else ZERO
    body = {
        "market": normalize_market(market),
        "decision_time": decision_time.isoformat(),
        "claims": [item.claim_id for item in admissible],
        "weighted_score": str(weighted),
        "confidence": str(confidence),
        "complete": complete,
        "conflict": conflict,
    }
    return StructuredMarketIntelligence(
        snapshot_id=f"market_intelligence_{stable_hash(body, length=40)}",
        market=market,
        decision_time=decision_time,
        claims=admissible,
        weighted_score=weighted,
        confidence=confidence,
        complete_perspectives=complete,
        conflict_detected=conflict,
        evidence_ids=tuple(item.claim_id for item in admissible),
    )


def create_shadow_ai_decision(
    snapshot: StructuredMarketIntelligence,
    *,
    horizon_seconds: int,
    model_or_policy_id: str = "deterministic_structured_synthesis_v1",
) -> AIDecisionSnapshot:
    if not snapshot.complete_perspectives:
        direction = InvestmentDirection.NO_TRADE
        reasons = ("INCOMPLETE_INTELLIGENCE_PERSPECTIVES",)
    elif snapshot.conflict_detected:
        direction = InvestmentDirection.NO_TRADE
        reasons = ("HIGH_CONFIDENCE_BULL_BEAR_CONFLICT",)
    elif snapshot.weighted_score >= Decimal("0.20"):
        direction = InvestmentDirection.LONG
        reasons = ("POSITIVE_STRUCTURED_EVIDENCE",)
    elif snapshot.weighted_score <= Decimal("-0.20"):
        direction = InvestmentDirection.REDUCE
        reasons = ("NEGATIVE_STRUCTURED_EVIDENCE",)
    else:
        direction = InvestmentDirection.NO_TRADE
        reasons = ("INSUFFICIENT_STRUCTURED_EDGE",)
    body = {
        "snapshot": snapshot.snapshot_id,
        "direction": direction.value,
        "horizon_seconds": horizon_seconds,
        "model_or_policy_id": model_or_policy_id,
    }
    return AIDecisionSnapshot(
        decision_id=f"ai_decision_{stable_hash(body, length=40)}",
        market=snapshot.market,
        observed_direction=direction,
        confidence=snapshot.confidence,
        expected_return=snapshot.weighted_score,
        expected_risk=None,
        horizon_seconds=horizon_seconds,
        generated_at=snapshot.decision_time,
        evidence_snapshot_id=snapshot.snapshot_id,
        model_or_policy_id=model_or_policy_id,
        reason_codes=reasons,
    )


def shadow_ai_to_investment_intent(snapshot: AIDecisionSnapshot) -> InvestmentIntent:
    """Enter the canonical chain without granting SHADOW output directional power."""

    return InvestmentIntent.create(
        market=snapshot.market,
        direction=InvestmentDirection.NO_TRADE,
        confidence=ZERO,
        expected_return=None,
        expected_risk=None,
        horizon_seconds=snapshot.horizon_seconds,
        strategy_id=snapshot.model_or_policy_id,
        family="AI_SHADOW_INTELLIGENCE",
        generated_at=snapshot.generated_at,
        valid_until=snapshot.generated_at + timedelta(seconds=snapshot.horizon_seconds),
        evidence_id=snapshot.decision_id,
        reason_codes=(
            "AI_SHADOW_NO_DECISION_AUTHORITY",
            f"OBSERVED_{snapshot.observed_direction.value}",
            *snapshot.reason_codes,
        ),
    )


__all__ = [
    "AIDecisionSnapshot",
    "IntelligenceClaim",
    "IntelligencePerspective",
    "StructuredMarketIntelligence",
    "create_shadow_ai_decision",
    "shadow_ai_to_investment_intent",
    "synthesize_market_intelligence",
]
