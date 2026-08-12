"""Canonical immutable schemas for ML datasets, labels and model artifacts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping

from pydantic import Field, field_validator, model_validator

from core.contracts import FrozenModel, normalize_market, require_utc
from utils.common import stable_hash

ZERO = Decimal("0")


class ModelStatus(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    SHADOW = "SHADOW"
    CHALLENGER = "CHALLENGER"
    ADVISORY = "ADVISORY"
    CANARY = "CANARY"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    REJECTED = "REJECTED"


class LabelOutcome(StrEnum):
    TARGET_FIRST = "TARGET_FIRST"
    STOP_FIRST = "STOP_FIRST"
    TIMEOUT = "TIMEOUT"
    AMBIGUOUS_SAME_BAR = "AMBIGUOUS_SAME_BAR"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class CanonicalDatasetManifest(FrozenModel):
    dataset_id: str
    schema_version: str
    feature_version: str
    label_version: str | None
    source_hashes: dict[str, str]
    created_from: tuple[str, ...]
    time_start: datetime
    time_end: datetime
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    feature_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    missingness_profile: dict[str, Decimal]
    point_in_time_policy: dict[str, Any]
    content_hash: str

    _time_start = field_validator("time_start")(require_utc)
    _time_end = field_validator("time_end")(require_utc)

    @field_validator("symbols")
    @classmethod
    def normalized_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(normalize_market(value) for value in values))

    @model_validator(mode="after")
    def validate_manifest(self) -> "CanonicalDatasetManifest":
        if self.time_end < self.time_start:
            raise ValueError("dataset time_end precedes time_start")
        if not self.source_hashes:
            raise ValueError("dataset requires source hashes")
        if not self.created_from:
            raise ValueError("dataset requires provenance")
        if not self.symbols or not self.timeframes:
            raise ValueError("dataset requires symbols and timeframes")
        if self.point_in_time_policy.get("available_at_lte_decision_time") is not True:
            raise ValueError("dataset must enforce available_at <= decision_time")
        if self.point_in_time_policy.get("features_labels_separated") is not True:
            raise ValueError("features and labels must be separated")
        if any(value < ZERO or value > Decimal("1") for value in self.missingness_profile.values()):
            raise ValueError("missingness fractions must be between zero and one")
        return self

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        feature_version: str,
        label_version: str | None,
        source_hashes: Mapping[str, str],
        created_from: tuple[str, ...],
        time_start: datetime,
        time_end: datetime,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        feature_count: int,
        row_count: int,
        missingness_profile: Mapping[str, Decimal],
        point_in_time_policy: Mapping[str, Any],
    ) -> "CanonicalDatasetManifest":
        values = {
            "schema_version": schema_version,
            "feature_version": feature_version,
            "label_version": label_version,
            "source_hashes": dict(sorted(source_hashes.items())),
            "created_from": tuple(sorted(created_from)),
            "time_start": require_utc(time_start),
            "time_end": require_utc(time_end),
            "symbols": tuple(sorted(normalize_market(value) for value in symbols)),
            "timeframes": tuple(sorted(timeframes)),
            "feature_count": feature_count,
            "row_count": row_count,
            "missingness_profile": {
                key: Decimal(value) for key, value in sorted(missingness_profile.items())
            },
            "point_in_time_policy": dict(point_in_time_policy),
        }
        identity = _json_identity(values)
        content_hash = stable_hash(identity, length=64)
        return cls(
            dataset_id=f"dataset_{content_hash}",
            content_hash=content_hash,
            **values,
        )


class LabelSchema(FrozenModel):
    label_version: str
    profit_barrier_fraction: Decimal = Field(gt=ZERO)
    stop_barrier_fraction: Decimal = Field(gt=ZERO)
    maximum_holding_seconds: int = Field(gt=0)
    conservative_same_bar_policy: bool = True
    cost_model_version: str


class CanonicalLabelRecord(FrozenModel):
    label_id: str
    label_version: str
    candidate_id: str
    market: str
    decision_time: datetime
    feature_cutoff: datetime
    label_start: datetime
    label_end: datetime
    outcome: LabelOutcome
    target_first: bool | None
    stop_first: bool | None
    timeout: bool
    gross_return: Decimal | None
    net_return: Decimal | None
    mae: Decimal | None
    mfe: Decimal | None
    holding_seconds: int | None = Field(default=None, ge=0)
    fees_fraction: Decimal = Field(ge=ZERO)
    spread_fraction: Decimal = Field(ge=ZERO)
    slippage_fraction: Decimal = Field(ge=ZERO)
    exit_reason: str

    _market = field_validator("market")(normalize_market)
    _decision = field_validator("decision_time")(require_utc)
    _cutoff = field_validator("feature_cutoff")(require_utc)
    _label_start = field_validator("label_start")(require_utc)
    _label_end = field_validator("label_end")(require_utc)

    @model_validator(mode="after")
    def validate_timing(self) -> "CanonicalLabelRecord":
        if self.feature_cutoff > self.decision_time:
            raise ValueError("feature cutoff cannot exceed decision time")
        if self.label_start < self.decision_time:
            raise ValueError("label window cannot start before decision time")
        if self.label_end < self.label_start:
            raise ValueError("label window end precedes start")
        if self.outcome is LabelOutcome.TARGET_FIRST and self.target_first is not True:
            raise ValueError("target outcome flag mismatch")
        if self.outcome is LabelOutcome.STOP_FIRST and self.stop_first is not True:
            raise ValueError("stop outcome flag mismatch")
        return self


class ModelArtifactManifest(FrozenModel):
    model_id: str
    dataset_id: str
    feature_schema: str
    label_schema: str
    algorithm: str
    hyperparameters: dict[str, Any]
    train_range: tuple[datetime, datetime]
    validation_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    code_commit: str
    metrics: dict[str, Decimal | int | None]
    economic_metrics: dict[str, Decimal | int | None]
    calibration: dict[str, Decimal | int | None]
    regime_metrics: dict[str, Any]
    status: ModelStatus
    trained_at: datetime
    expires_at: datetime
    artifact_hash: str
    live_decision_influence: bool = False

    _trained_at = field_validator("trained_at")(require_utc)
    _expires_at = field_validator("expires_at")(require_utc)

    @model_validator(mode="after")
    def validate_model(self) -> "ModelArtifactManifest":
        for name, (start, end) in (
            ("train", self.train_range),
            ("validation", self.validation_range),
            ("test", self.test_range),
        ):
            require_utc(start)
            require_utc(end)
            if end < start:
                raise ValueError(f"{name} range is reversed")
        if not (
            self.train_range[1]
            <= self.validation_range[0]
            <= self.validation_range[1]
            <= self.test_range[0]
        ):
            raise ValueError("model ranges must be chronological and non-overlapping")
        if self.expires_at <= self.trained_at:
            raise ValueError("model expiry must follow training")
        if self.status in {ModelStatus.RESEARCH_ONLY, ModelStatus.SHADOW, ModelStatus.CHALLENGER}:
            if self.live_decision_influence:
                raise ValueError("research/shadow/challenger model cannot influence live decisions")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "ModelArtifactManifest":
        identity = _json_identity(payload)
        artifact_hash = stable_hash(identity, length=64)
        model_id = str(payload.pop("model_id", "") or f"model_{artifact_hash[:40]}")
        return cls(model_id=model_id, artifact_hash=artifact_hash, **payload)


def _json_identity(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return require_utc(value).isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_json_identity(item) for item in value]
    return value


__all__ = [
    "CanonicalDatasetManifest",
    "CanonicalLabelRecord",
    "LabelOutcome",
    "LabelSchema",
    "ModelArtifactManifest",
    "ModelStatus",
]
