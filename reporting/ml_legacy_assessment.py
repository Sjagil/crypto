"""Assess legacy shadow artifacts against the canonical ML lifecycle.

The assessment is deliberately conservative.  It never converts an older
artifact into a canonical dataset or model when mandatory point-in-time and
split provenance cannot be proven from the artifact itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ml.contracts import ModelStatus
from ml.registry import evaluate_model_promotion
from utils.common import read_json, sha256_file, stable_hash, utc_iso

SCHEMA_VERSION = "legacy_shadow_ml_assessment_v1"


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def assess_legacy_shadow_ml(project_root: Path) -> dict[str, Any]:
    """Return a no-side-effect promotion and migration assessment."""

    root = project_root.resolve()
    intelligence = root / "output" / "intelligence"
    dataset_path = intelligence / "opportunity_training_rows.json"
    model_path = intelligence / "model_bundle.joblib"
    status_path = intelligence / "model_status.json"
    status = _mapping(status_path)
    dataset = _mapping(dataset_path)
    rows = [row for row in dataset.get("rows") or [] if isinstance(row, Mapping)]

    explicit_point_in_time_rows = sum(
        bool(row.get("event_time")) and bool(row.get("available_at"))
        for row in rows
    )
    separated_label_rows = sum(
        row.get("label_uses_future_features") is False for row in rows
    )
    fold_rows = [
        dict(row)
        for row in status.get("validation_folds") or []
        if isinstance(row, Mapping)
    ]
    folds_with_exact_ranges = sum(
        all(
            row.get(key)
            for key in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        )
        for row in fold_rows
    )
    dataset_blockers: list[str] = []
    if not rows:
        dataset_blockers.append("TRAINING_ROWS_MISSING")
    if explicit_point_in_time_rows != len(rows):
        dataset_blockers.append("EXPLICIT_EVENT_TIME_AND_AVAILABLE_AT_MISSING")
    if separated_label_rows != len(rows):
        dataset_blockers.append("FEATURE_LABEL_SEPARATION_NOT_PROVEN_PER_ROW")
    model_blockers: list[str] = []
    if not model_path.is_file():
        model_blockers.append("MODEL_ARTIFACT_MISSING")
    if folds_with_exact_ranges != len(fold_rows) or not fold_rows:
        model_blockers.append("EXACT_TRAIN_VALIDATION_TEST_RANGES_MISSING")
    if status.get("purged_walk_forward") is not True:
        model_blockers.append("PURGED_WALK_FORWARD_NOT_PROVEN")
    if not status.get("calibration_report"):
        model_blockers.append("CALIBRATION_PROVENANCE_INCOMPLETE")
    drift = dict(status.get("drift_monitor") or {})
    if str(drift.get("status") or "").upper().startswith("CRITICAL"):
        model_blockers.append("CRITICAL_FEATURE_DRIFT")

    evidence = {
        "dataset_immutable": False,
        "point_in_time_passed": not dataset_blockers,
        "lookahead_passed": separated_label_rows == len(rows) and bool(rows),
        "purged_walk_forward_passed": False,
        "economic_oos_positive": False,
        "calibration_acceptable": False,
        "minimum_sample_passed": bool(status.get("promotion_evaluation_ready")),
        "manual_authorization": False,
        "live_validation_passed": False,
    }
    promotion = evaluate_model_promotion(
        current_status=ModelStatus.RESEARCH_ONLY,
        requested_status=ModelStatus.SHADOW,
        evidence=evidence,
    )
    promotion["permitted"] = False
    promotion["failed_checks"] = sorted(
        set(promotion["failed_checks"])
        | {"canonical_legacy_migration_not_permitted"}
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": "BLOCKED_LEGACY_PROVENANCE",
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "present": dataset_path.is_file(),
            "sha256": sha256_file(dataset_path) if dataset_path.is_file() else None,
            "row_count": len(rows),
            "explicit_point_in_time_rows": explicit_point_in_time_rows,
            "feature_label_separated_rows": separated_label_rows,
            "canonical_registry_status": "NOT_REGISTERED",
            "blockers": dataset_blockers,
        },
        "model": {
            "path": str(model_path.resolve()),
            "present": model_path.is_file(),
            "sha256": sha256_file(model_path) if model_path.is_file() else None,
            "legacy_status": status.get("status"),
            "trained_until_timestamp": status.get("trained_until_timestamp"),
            "validation_fold_count": len(fold_rows),
            "folds_with_exact_ranges": folds_with_exact_ranges,
            "canonical_registry_status": "NOT_REGISTERED",
            "blockers": model_blockers,
        },
        "baseline_models": {
            "meta_labeler": "LEGACY_SHADOW_BLOCKED_FROM_CANONICAL_PROMOTION",
            "cross_sectional_ranker": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES",
            "net_return_model": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES",
            "downside_model": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES",
            "anomaly_detector": "LEGACY_SHADOW_ONLY",
        },
        "promotion_evaluation": promotion,
        "fallback": "DETERMINISTIC_STRATEGY_AND_RISK_ENGINE",
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_mutations": 0,
        "authority_changes": 0,
    }
    payload["assessment_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "generated_at"},
        length=64,
    )
    return payload


__all__ = ["SCHEMA_VERSION", "assess_legacy_shadow_ml"]
