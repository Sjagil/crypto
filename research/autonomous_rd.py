"""Bounded native research loop inspired by RD-Agent's lifecycle concepts.

The trace is immutable: every append returns a new trace.  It can remember
hypotheses, preregistered experiments and feedback, but it cannot edit strategy
approvals, model status, risk policy, exchange state or live authority.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from core.contracts import FrozenModel, require_utc
from utils.common import stable_hash


class ResearchRecordKind(StrEnum):
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT = "EXPERIMENT"
    FEEDBACK = "FEEDBACK"


class ExperimentDecision(StrEnum):
    ACCEPT_FOR_MORE_RESEARCH = "ACCEPT_FOR_MORE_RESEARCH"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class ResearchHypothesis(FrozenModel):
    hypothesis_id: str
    statement: str
    rationale: str
    falsification_criteria: tuple[str, ...]
    evidence_inputs: tuple[str, ...]
    created_at: datetime
    authority: Literal["RESEARCH_ONLY"] = "RESEARCH_ONLY"

    _created = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def complete_hypothesis(self) -> "ResearchHypothesis":
        if not self.statement.strip() or not self.rationale.strip():
            raise ValueError("hypothesis statement and rationale are required")
        if not self.falsification_criteria or not self.evidence_inputs:
            raise ValueError("hypothesis requires falsification criteria and evidence inputs")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "ResearchHypothesis":
        identity = _identity(payload)
        return cls(
            hypothesis_id=f"hypothesis_{stable_hash(identity, length=40)}",
            **payload,
        )


class PreregisteredExperiment(FrozenModel):
    experiment_id: str
    hypothesis_id: str
    dataset_ids: tuple[str, ...]
    code_commit: str
    metrics: tuple[str, ...]
    acceptance_thresholds: dict[str, Decimal]
    leakage_controls: tuple[str, ...]
    cost_model_version: str
    preregistered_at: datetime
    authority: Literal["RESEARCH_ONLY"] = "RESEARCH_ONLY"

    _created = field_validator("preregistered_at")(require_utc)

    @model_validator(mode="after")
    def complete_experiment(self) -> "PreregisteredExperiment":
        if not self.dataset_ids or not self.metrics or not self.leakage_controls:
            raise ValueError("experiment requires datasets, metrics and leakage controls")
        if not self.acceptance_thresholds or not self.cost_model_version.strip():
            raise ValueError("experiment requires thresholds and canonical cost version")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "PreregisteredExperiment":
        identity = _identity(payload)
        return cls(
            experiment_id=f"experiment_{stable_hash(identity, length=40)}",
            **payload,
        )


class ExperimentFeedback(FrozenModel):
    feedback_id: str
    experiment_id: str
    decision: ExperimentDecision
    observed_metrics: dict[str, Decimal | int | None]
    observations: tuple[str, ...]
    failure_reasons: tuple[str, ...] = ()
    next_hypothesis: str | None = None
    recorded_at: datetime
    live_promotion_authority: Literal[False] = False

    _recorded = field_validator("recorded_at")(require_utc)

    @model_validator(mode="after")
    def explain_decision(self) -> "ExperimentFeedback":
        if not self.observations:
            raise ValueError("feedback requires observations")
        if self.decision is ExperimentDecision.REJECT and not self.failure_reasons:
            raise ValueError("rejected experiment requires failure reasons")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "ExperimentFeedback":
        identity = _identity(payload)
        return cls(feedback_id=f"feedback_{stable_hash(identity, length=40)}", **payload)


class ResearchTraceRecord(FrozenModel):
    sequence: int = Field(ge=0)
    kind: ResearchRecordKind
    record_id: str
    parent_id: str | None
    content_hash: str
    payload: dict[str, Any]


class ResearchTrace(FrozenModel):
    trace_id: str
    records: tuple[ResearchTraceRecord, ...] = ()
    execution_authority: Literal[False] = False
    automatic_promotion_permitted: Literal[False] = False

    @classmethod
    def empty(cls, namespace: str) -> "ResearchTrace":
        return cls(trace_id=f"rd_trace_{stable_hash({'namespace': namespace}, length=40)}")

    def append(
        self,
        record: ResearchHypothesis | PreregisteredExperiment | ExperimentFeedback,
    ) -> "ResearchTrace":
        if isinstance(record, ResearchHypothesis):
            kind = ResearchRecordKind.HYPOTHESIS
            record_id = record.hypothesis_id
            parent_id = None
        elif isinstance(record, PreregisteredExperiment):
            kind = ResearchRecordKind.EXPERIMENT
            record_id = record.experiment_id
            parent_id = record.hypothesis_id
            if parent_id not in {item.record_id for item in self.records}:
                raise ValueError("experiment parent hypothesis is not in trace")
        else:
            kind = ResearchRecordKind.FEEDBACK
            record_id = record.feedback_id
            parent_id = record.experiment_id
            if parent_id not in {item.record_id for item in self.records}:
                raise ValueError("feedback parent experiment is not in trace")
        if record_id in {item.record_id for item in self.records}:
            raise ValueError("trace records are append-only and unique")
        payload = record.model_dump(mode="json")
        item = ResearchTraceRecord(
            sequence=len(self.records),
            kind=kind,
            record_id=record_id,
            parent_id=parent_id,
            content_hash=stable_hash(payload, length=64),
            payload=payload,
        )
        return self.model_copy(update={"records": self.records + (item,)})


def _identity(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return require_utc(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _identity(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_identity(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


__all__ = [
    "ExperimentDecision",
    "ExperimentFeedback",
    "PreregisteredExperiment",
    "ResearchHypothesis",
    "ResearchRecordKind",
    "ResearchTrace",
    "ResearchTraceRecord",
]
