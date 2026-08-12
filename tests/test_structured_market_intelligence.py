from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from core.structured_market_intelligence import (
    IntelligenceClaim,
    IntelligencePerspective,
    create_shadow_ai_decision,
    shadow_ai_to_investment_intent,
    synthesize_market_intelligence,
)
from portfolio.contracts import InvestmentDirection

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _claim(
    perspective: IntelligencePerspective,
    score: str,
    confidence: str = "0.8",
) -> IntelligenceClaim:
    return IntelligenceClaim.create(
        perspective=perspective,
        thesis=f"{perspective.value} point-in-time thesis",
        signed_score=Decimal(score),
        confidence=Decimal(confidence),
        event_time=NOW,
        available_at=NOW,
        source_ids=(f"source_{perspective.value.lower()}",),
    )


def test_complete_structured_evidence_stays_shadow_at_intent_boundary() -> None:
    evidence = synthesize_market_intelligence(
        market="BTC-EUR",
        decision_time=NOW,
        claims=(
            _claim(IntelligencePerspective.BULL, "0.9"),
            _claim(IntelligencePerspective.BEAR, "-0.1"),
            _claim(IntelligencePerspective.RISK, "0.0"),
        ),
    )
    decision = create_shadow_ai_decision(evidence, horizon_seconds=3600)
    assert decision.observed_direction is InvestmentDirection.LONG
    assert decision.live_decision_influence is False
    intent = shadow_ai_to_investment_intent(decision)
    assert intent.direction is InvestmentDirection.NO_TRADE
    assert intent.confidence == Decimal("0")
    assert "AI_SHADOW_NO_DECISION_AUTHORITY" in intent.reason_codes


def test_high_confidence_bull_bear_conflict_fails_closed() -> None:
    evidence = synthesize_market_intelligence(
        market="ETH-EUR",
        decision_time=NOW,
        claims=(
            _claim(IntelligencePerspective.BULL, "0.9", "0.9"),
            _claim(IntelligencePerspective.BEAR, "-0.9", "0.9"),
            _claim(IntelligencePerspective.RISK, "-0.2", "0.7"),
        ),
    )
    assert evidence.conflict_detected is True
    assert evidence.confidence == Decimal("0")
    assert create_shadow_ai_decision(evidence, horizon_seconds=900).observed_direction is InvestmentDirection.NO_TRADE
