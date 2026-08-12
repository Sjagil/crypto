"""FreqAI-inspired native freshness, warmup and causal-boundary checks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from core.contracts import require_utc
from ml.contracts import ModelArtifactManifest, ModelStatus


def audit_point_in_time_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    decision_time: datetime,
    minimum_warmup_rows: int,
) -> dict[str, Any]:
    decision_time = require_utc(decision_time)
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        available_at = row.get("available_at")
        event_time = row.get("event_time")
        if not isinstance(available_at, datetime) or not isinstance(event_time, datetime):
            failures.append({"row": index, "code": "TIMESTAMP_MISSING_OR_INVALID"})
            continue
        available_at = require_utc(available_at)
        event_time = require_utc(event_time)
        if available_at > decision_time:
            failures.append({"row": index, "code": "AVAILABLE_AFTER_DECISION"})
        if event_time > available_at:
            failures.append({"row": index, "code": "EVENT_AFTER_AVAILABILITY"})
        if row.get("is_final") is not True:
            failures.append({"row": index, "code": "INCOMPLETE_CANDLE_OR_FEATURE"})
        if not row.get("feature_version"):
            failures.append({"row": index, "code": "FEATURE_VERSION_MISSING"})
        if not row.get("provenance"):
            failures.append({"row": index, "code": "PROVENANCE_MISSING"})
    if len(rows) < minimum_warmup_rows:
        failures.append(
            {
                "row": None,
                "code": "INSUFFICIENT_WARMUP",
                "received": len(rows),
                "required": minimum_warmup_rows,
            }
        )
    return {
        "status": "PASSED" if not failures else "HARD_REJECT",
        "row_count": len(rows),
        "minimum_warmup_rows": minimum_warmup_rows,
        "failures": failures,
        "available_at_lte_decision_time": not any(
            row["code"] == "AVAILABLE_AFTER_DECISION" for row in failures
        ),
        "closed_final_inputs_only": not any(
            row["code"] == "INCOMPLETE_CANDLE_OR_FEATURE" for row in failures
        ),
    }


def evaluate_model_freshness(
    model: ModelArtifactManifest,
    *,
    decision_time: datetime,
) -> dict[str, Any]:
    decision_time = require_utc(decision_time)
    expired = decision_time > model.expires_at
    suspended = model.status in {ModelStatus.SUSPENDED, ModelStatus.REJECTED}
    usable_for_shadow = not expired and not suspended and model.status in {
        ModelStatus.SHADOW,
        ModelStatus.CHALLENGER,
        ModelStatus.ADVISORY,
        ModelStatus.CANARY,
        ModelStatus.ACTIVE,
    }
    return {
        "status": "READY" if usable_for_shadow else "BLOCKED",
        "expired": expired,
        "suspended_or_rejected": suspended,
        "shadow_inference_permitted": usable_for_shadow,
        "live_influence_permitted": (
            usable_for_shadow
            and model.status in {ModelStatus.CANARY, ModelStatus.ACTIVE}
            and model.live_decision_influence
        ),
        "fallback": "DETERMINISTIC_STRATEGY_ENGINE",
    }


__all__ = ["audit_point_in_time_features", "evaluate_model_freshness"]
