"""Shared domain contracts for research, risk, intelligence and execution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainError(RuntimeError):
    """Base class for expected, machine-actionable domain failures."""

    code = "DOMAIN_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)


class ConfigurationError(DomainError):
    code = "CONFIGURATION_ERROR"


class DataValidationError(DomainError):
    code = "DATA_VALIDATION_ERROR"


class IntelligenceTimingError(DomainError):
    code = "INTELLIGENCE_TIMING_ERROR"


class EligibilityError(DomainError):
    code = "ELIGIBILITY_ERROR"


class RiskRejected(DomainError):
    code = "RISK_REJECTED"


class ExecutionBlocked(DomainError):
    code = "EXECUTION_BLOCKED"


class ReconciliationRequired(ExecutionBlocked):
    code = "RECONCILIATION_REQUIRED"


class EligibilityStatus(StrEnum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SignalAction(StrEnum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    REDUCE = "REDUCE"
    AVOID = "AVOID"
    NO_ENTRY = "NO_ENTRY"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class TimestampQuality(StrEnum):
    PROVEN = "PROVEN"
    SOURCE_REPORTED = "SOURCE_REPORTED"
    OBSERVED_ONLY = "OBSERVED_ONLY"
    UNKNOWN = "UNKNOWN"


class HistoricalCoverage(StrEnum):
    HISTORICAL = "HISTORICAL"
    FORWARD_ONLY = "FORWARD_ONLY"


class ProviderStatus(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    SKIPPED_MISSING_CREDENTIALS = "SKIPPED_MISSING_CREDENTIALS"
    BLOCKED_PLAN_LIMIT = "BLOCKED_PLAN_LIMIT"
    BLOCKED_PERMISSION = "BLOCKED_PERMISSION"
    BLOCKED_RATE_LIMIT = "BLOCKED_RATE_LIMIT"
    BLOCKED_PROVIDER_UNAVAILABLE = "BLOCKED_PROVIDER_UNAVAILABLE"
    UNSUPPORTED_ENDPOINT = "UNSUPPORTED_ENDPOINT"
    FAILED_VALIDATION = "FAILED_VALIDATION"


class HistoryProfile(StrEnum):
    SMOKE = "smoke"
    STANDARD = "standard"
    DEEP = "deep"
    MAXIMUM = "maximum"


class ResearchStatus(StrEnum):
    REJECTED_DATA = "REJECTED_DATA"
    REJECTED_INTELLIGENCE_TIMING = "REJECTED_INTELLIGENCE_TIMING"
    REJECTED_LOOKAHEAD = "REJECTED_LOOKAHEAD"
    REJECTED_REPAINTING = "REJECTED_REPAINTING"
    REJECTED_ELIGIBILITY = "REJECTED_ELIGIBILITY"
    REJECTED_INSUFFICIENT_TRADES = "REJECTED_INSUFFICIENT_TRADES"
    REJECTED_EFFECTIVE_SAMPLE = "REJECTED_EFFECTIVE_SAMPLE"
    REJECTED_EXPECTANCY = "REJECTED_EXPECTANCY"
    REJECTED_PROFIT_FACTOR = "REJECTED_PROFIT_FACTOR"
    REJECTED_STRESSED_COSTS = "REJECTED_STRESSED_COSTS"
    REJECTED_WALK_FORWARD = "REJECTED_WALK_FORWARD"
    REJECTED_DRAWDOWN = "REJECTED_DRAWDOWN"
    REJECTED_RISK_OF_RUIN = "REJECTED_RISK_OF_RUIN"
    REJECTED_PARAMETER_INSTABILITY = "REJECTED_PARAMETER_INSTABILITY"
    REJECTED_CONCENTRATION = "REJECTED_CONCENTRATION"
    MISSING_REQUIRED_CONTEXT = "MISSING_REQUIRED_CONTEXT"
    RESEARCH_PASS = "RESEARCH_PASS"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    LIVE_BLOCKED = "LIVE_BLOCKED"


class CandidateLifecycle(StrEnum):
    DISCOVERED = "DISCOVERED"
    SCREENING_SURVIVOR = "SCREENING_SURVIVOR"
    EXACT_SURVIVOR = "EXACT_SURVIVOR"
    VALIDATION_SURVIVOR = "VALIDATION_SURVIVOR"
    RESEARCH_PASS = "RESEARCH_PASS"
    SHADOW_CANDIDATE = "SHADOW_CANDIDATE"
    SHADOW_ACTIVE = "SHADOW_ACTIVE"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


_CANDIDATE_TRANSITIONS = {
    CandidateLifecycle.DISCOVERED: {CandidateLifecycle.SCREENING_SURVIVOR},
    CandidateLifecycle.SCREENING_SURVIVOR: {CandidateLifecycle.EXACT_SURVIVOR},
    CandidateLifecycle.EXACT_SURVIVOR: {CandidateLifecycle.VALIDATION_SURVIVOR},
    CandidateLifecycle.VALIDATION_SURVIVOR: {CandidateLifecycle.RESEARCH_PASS},
    CandidateLifecycle.RESEARCH_PASS: {CandidateLifecycle.SHADOW_CANDIDATE},
    CandidateLifecycle.SHADOW_CANDIDATE: {CandidateLifecycle.SHADOW_ACTIVE},
    CandidateLifecycle.SHADOW_ACTIVE: {
        CandidateLifecycle.PAPER_CANDIDATE,
        CandidateLifecycle.DEGRADED,
        CandidateLifecycle.SUSPENDED,
    },
    CandidateLifecycle.PAPER_CANDIDATE: {
        CandidateLifecycle.PAPER_ACTIVE,
        CandidateLifecycle.SUSPENDED,
    },
    CandidateLifecycle.PAPER_ACTIVE: {
        CandidateLifecycle.DEGRADED,
        CandidateLifecycle.SUSPENDED,
    },
    CandidateLifecycle.DEGRADED: {
        CandidateLifecycle.SUSPENDED,
        CandidateLifecycle.SHADOW_ACTIVE,
        CandidateLifecycle.PAPER_ACTIVE,
    },
    CandidateLifecycle.SUSPENDED: {
        CandidateLifecycle.SHADOW_CANDIDATE,
        CandidateLifecycle.PAPER_CANDIDATE,
        CandidateLifecycle.RETIRED,
    },
    CandidateLifecycle.RETIRED: set(),
}


class StreamEventType(StrEnum):
    TICKER = "ticker"
    TRADE = "trade"
    CANDLE = "candle"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    CONNECTION_STATUS = "connection_status"


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_market(value: str) -> str:
    normalized = value.strip().upper().replace("/", "-").replace("_", "-")
    parts = normalized.split("-")
    if len(parts) != 2 or not all(part.isalnum() for part in parts):
        raise ValueError("market must use BASE-QUOTE format")
    return normalized


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        hide_input_in_errors=True,
    )


class CandidateArtifact(FrozenModel):
    """Immutable operational hand-off produced by the canonical research path."""

    candidate_id: str
    strategy_dna_hash: str
    software_version: str
    signal_blocks: tuple[dict[str, Any], ...]
    parameters: dict[str, Any]
    parameter_hash: str
    logic_mode: str
    logic_weights: dict[str, float] = Field(default_factory=dict)
    exit_profile: dict[str, Any]
    risk_profile: dict[str, Any]
    eligible_markets: tuple[str, ...]
    supported_timeframes: tuple[str, ...]
    required_providers: tuple[str, ...] = ()
    required_context_datasets: tuple[str, ...] = ()
    data_hashes: dict[str, str]
    train_period: dict[str, str]
    validation_periods: tuple[dict[str, str], ...]
    final_holdout_period: dict[str, str]
    normal_cost_metrics: dict[str, Any]
    stressed_cost_metrics: dict[str, Any]
    double_cost_metrics: dict[str, Any]
    walk_forward_metrics: dict[str, Any]
    cpcv_diagnostics: dict[str, Any] | None = None
    monte_carlo_metrics: dict[str, Any]
    parameter_stability_metrics: dict[str, Any]
    asset_generalization_metrics: dict[str, Any]
    lifecycle_state: CandidateLifecycle
    created_at: datetime
    expires_at: datetime
    manifest_hash: str

    _created_at = field_validator("created_at")(require_utc)
    _expires_at = field_validator("expires_at")(require_utc)

    @classmethod
    def create(cls, **payload: Any) -> "CandidateArtifact":
        """Create a correctly hashed immutable artifact from research output."""

        parameters = dict(payload.get("parameters") or {})
        payload["parameter_hash"] = hashlib.sha256(
            json.dumps(
                parameters,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        draft = cls.model_construct(**payload, manifest_hash="")
        payload["manifest_hash"] = draft.expected_manifest_hash()
        return cls.model_validate(payload)

    @field_validator("eligible_markets")
    @classmethod
    def candidate_markets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("candidate requires at least one eligible market")
        return tuple(normalize_market(value) for value in values)

    @model_validator(mode="after")
    def validate_candidate(self) -> "CandidateArtifact":
        if self.expires_at <= self.created_at:
            raise ValueError("candidate expiry must follow creation")
        expected_parameter_hash = hashlib.sha256(
            json.dumps(
                self.parameters,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if self.parameter_hash != expected_parameter_hash:
            raise ValueError("candidate parameter hash mismatch")
        if not self.verify_manifest():
            raise ValueError("candidate manifest hash mismatch")
        return self

    def manifest_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"manifest_hash"},
        )

    def expected_manifest_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.manifest_payload(),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def verify_manifest(self) -> bool:
        return self.manifest_hash == self.expected_manifest_hash()

    def permits_transition(self, target: CandidateLifecycle) -> bool:
        return target in _CANDIDATE_TRANSITIONS[self.lifecycle_state]


class NormalizedDataRecord(FrozenModel):
    provider: str
    source_symbol: str
    canonical_market: str
    timestamp: datetime
    observed_at: datetime
    available_at: datetime | None = None
    data_kind: str
    timeframe: str | None = None
    closed: bool | None = None
    retrieval_run_id: str
    raw_hash: str
    raw_payload: Any | None = None
    values: dict[str, Any] = Field(default_factory=dict)

    _market = field_validator("canonical_market")(normalize_market)
    _timestamp = field_validator("timestamp")(require_utc)
    _observed_at = field_validator("observed_at")(require_utc)

    @field_validator("available_at")
    @classmethod
    def validate_available_at(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


class NormalizedStreamEvent(FrozenModel):
    event_type: StreamEventType
    provider: str
    source_symbol: str
    canonical_market: str
    timestamp: datetime
    observed_at: datetime
    sequence: int | None = None
    message_id: str
    payload: dict[str, Any] = Field(default_factory=dict)

    _market = field_validator("canonical_market")(normalize_market)
    _timestamp = field_validator("timestamp")(require_utc)
    _observed_at = field_validator("observed_at")(require_utc)


class EligibilityRecord(FrozenModel):
    market: str
    status: EligibilityStatus
    reason: str

    _market = field_validator("market")(normalize_market)

    @field_validator("reason")
    @classmethod
    def reason_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("eligibility reason is required")
        return value


class StrategySignal(FrozenModel):
    market: str
    timestamp: datetime
    action: SignalAction
    strategy_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    reason_codes: tuple[str, ...] = ()
    stop_price: Decimal | None = Field(default=None, gt=0)
    target_price: Decimal | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _market = field_validator("market")(normalize_market)
    _timestamp = field_validator("timestamp")(require_utc)

    @field_validator("strategy_id")
    @classmethod
    def strategy_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("strategy_id is required")
        return value


class OrderIntent(FrozenModel):
    intent_id: str
    idempotency_key: str
    market: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)
    limit_price: Decimal | None = Field(default=None, gt=0)
    strategy_id: str
    maximum_notional_eur: Decimal | None = Field(default=None, gt=0)
    reason_codes: tuple[str, ...] = ()

    _market = field_validator("market")(normalize_market)
    _created_at = field_validator("created_at")(require_utc)

    @field_validator("intent_id", "idempotency_key", "strategy_id")
    @classmethod
    def identifiers_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_limit(self) -> "OrderIntent":
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("market orders cannot include limit_price")
        return self


class Fill(FrozenModel):
    fill_id: str
    order_id: str
    intent_id: str
    market: str
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    fee_eur: Decimal = Field(ge=0)
    filled_at: datetime
    venue: str

    _market = field_validator("market")(normalize_market)
    _filled_at = field_validator("filled_at")(require_utc)


class OrderRecord(FrozenModel):
    order_id: str
    intent: OrderIntent
    status: OrderStatus
    filled_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    average_fill_price: Decimal | None = Field(default=None, gt=0)
    rejection_code: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    _updated_at = field_validator("updated_at")(require_utc)

    @model_validator(mode="after")
    def validate_fill_quantity(self) -> "OrderRecord":
        if self.filled_quantity > self.intent.quantity:
            raise ValueError("filled quantity exceeds order quantity")
        if self.status is OrderStatus.REJECTED and not self.rejection_code:
            raise ValueError("rejected orders require a rejection code")
        return self


class Trade(FrozenModel):
    trade_id: str
    market: str
    strategy_id: str
    entry_at: datetime
    exit_at: datetime
    quantity: Decimal = Field(gt=0)
    entry_price: Decimal = Field(gt=0)
    exit_price: Decimal = Field(gt=0)
    entry_fee_eur: Decimal = Field(ge=0)
    exit_fee_eur: Decimal = Field(ge=0)
    net_pnl_eur: Decimal
    r_multiple: float
    exit_reason: str
    mae_r: float | None = None
    mfe_r: float | None = None

    _market = field_validator("market")(normalize_market)
    _entry_at = field_validator("entry_at")(require_utc)
    _exit_at = field_validator("exit_at")(require_utc)

    @model_validator(mode="after")
    def validate_chronology(self) -> "Trade":
        if self.exit_at < self.entry_at:
            raise ValueError("trade exit cannot precede entry")
        return self


class StrategyMetadata(FrozenModel):
    strategy_id: str
    family: str
    description: str
    parameter_space: dict[str, tuple[Any, ...]] = Field(default_factory=dict)
    long_only: bool = True
    uses_intelligence: bool = False

    @model_validator(mode="after")
    def enforce_long_only(self) -> "StrategyMetadata":
        if not self.long_only:
            raise ValueError("active strategies must be long-only")
        return self


class GateResult(FrozenModel):
    status: ResearchStatus
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "GateResult":
        passing = {
            ResearchStatus.RESEARCH_PASS,
            ResearchStatus.PAPER_CANDIDATE,
        }
        if self.passed != (self.status in passing):
            raise ValueError("gate passed flag is inconsistent with status")
        if not self.passed and not self.reasons:
            raise ValueError("rejected gate result requires at least one reason")
        return self


class IntelligenceRecord(FrozenModel):
    event_id: str
    source: str
    url: str
    title: str
    summary: str = ""
    published_at: datetime | None = None
    observed_at: datetime
    timestamp_quality: TimestampQuality
    language: str = "en"
    entities: tuple[str, ...] = ()
    markets: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    relevance_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    deduplication_status: str = "UNIQUE"
    historical_coverage: HistoricalCoverage
    raw_hash: str

    _observed_at = field_validator("observed_at")(require_utc)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None

    @field_validator("markets")
    @classmethod
    def validate_markets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_market(value) for value in values)

    @model_validator(mode="after")
    def enforce_knowability(self) -> "IntelligenceRecord":
        if self.published_at and self.published_at > self.observed_at:
            raise ValueError("published_at cannot be after observed_at")
        if self.timestamp_quality in {
            TimestampQuality.OBSERVED_ONLY,
            TimestampQuality.UNKNOWN,
        } and self.historical_coverage is not HistoricalCoverage.FORWARD_ONLY:
            raise ValueError("uncertain publication time must be forward-only")
        if self.historical_coverage is HistoricalCoverage.FORWARD_ONLY:
            if self.published_at and self.published_at < self.observed_at:
                raise ValueError("forward-only records cannot be backdated")
        return self

    @property
    def usable_at(self) -> datetime:
        if (
            self.historical_coverage is HistoricalCoverage.HISTORICAL
            and self.published_at is not None
            and self.timestamp_quality
            in {TimestampQuality.PROVEN, TimestampQuality.SOURCE_REPORTED}
        ):
            return self.published_at
        return self.observed_at


class RiskDecision(FrozenModel):
    approved: bool
    reason_codes: tuple[str, ...]
    approved_quantity: Decimal = Field(default=Decimal("0"), ge=0)
    risk_eur: Decimal = Field(default=Decimal("0"), ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> "RiskDecision":
        if self.approved and self.approved_quantity <= 0:
            raise ValueError("approved decision requires positive quantity")
        if not self.approved and not self.reason_codes:
            raise ValueError("rejected decision requires reason codes")
        return self


__all__ = [
    "ConfigurationError",
    "CandidateArtifact",
    "CandidateLifecycle",
    "DataValidationError",
    "DomainError",
    "EligibilityError",
    "EligibilityRecord",
    "EligibilityStatus",
    "ExecutionBlocked",
    "Fill",
    "GateResult",
    "HistoricalCoverage",
    "IntelligenceRecord",
    "IntelligenceTimingError",
    "NormalizedDataRecord",
    "NormalizedStreamEvent",
    "OrderIntent",
    "OrderRecord",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "ReconciliationRequired",
    "ResearchStatus",
    "RiskDecision",
    "RiskRejected",
    "SignalAction",
    "StrategyMetadata",
    "StrategySignal",
    "StreamEventType",
    "TimestampQuality",
    "Trade",
    "normalize_market",
    "require_utc",
    "utc_now",
]
