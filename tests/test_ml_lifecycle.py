from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ml.contracts import (
    CanonicalDatasetManifest,
    LabelOutcome,
    LabelSchema,
    ModelArtifactManifest,
    ModelStatus,
)
from ml.labels import build_triple_barrier_label, freeze_labels
from ml.lifecycle import audit_point_in_time_features, evaluate_model_freshness
from ml.registry import ImmutableDatasetRegistry, ModelRegistry, evaluate_model_promotion

NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)


def _dataset() -> CanonicalDatasetManifest:
    return CanonicalDatasetManifest.create(
        schema_version="canonical_ml_dataset_v1",
        feature_version="features-v1",
        label_version="triple-barrier-v1",
        source_hashes={"candles": "a" * 64},
        created_from=("data.multi_source_platform.PointInTimeFeatureStore",),
        time_start=NOW - timedelta(days=30),
        time_end=NOW,
        symbols=("SOL-EUR", "BTC-EUR"),
        timeframes=("1h",),
        feature_count=42,
        row_count=1000,
        missingness_profile={"all_features": Decimal("0.01")},
        point_in_time_policy={
            "available_at_lte_decision_time": True,
            "features_labels_separated": True,
            "closed_candles_only": True,
        },
    )


def _model(status: ModelStatus = ModelStatus.SHADOW) -> ModelArtifactManifest:
    return ModelArtifactManifest.create(
        model_id="meta-logistic-v1",
        dataset_id=_dataset().dataset_id,
        feature_schema="features-v1",
        label_schema="triple-barrier-v1",
        algorithm="LogisticRegression",
        hyperparameters={"C": 1.0, "random_state": 7},
        train_range=(NOW - timedelta(days=30), NOW - timedelta(days=15)),
        validation_range=(NOW - timedelta(days=14), NOW - timedelta(days=8)),
        test_range=(NOW - timedelta(days=7), NOW - timedelta(days=1)),
        code_commit="23fd989dbc3976b4d8c309020d8db06ac4a56eb1",
        metrics={"brier": Decimal("0.20"), "sample_size": 500},
        economic_metrics={"net_expectancy": Decimal("0.001")},
        calibration={"ece": Decimal("0.04")},
        regime_metrics={"status": "NOT_EVALUABLE_IN_FIXTURE"},
        status=status,
        trained_at=NOW,
        expires_at=NOW + timedelta(days=7),
        live_decision_influence=False,
    )


def test_dataset_identity_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    first = _dataset()
    second = _dataset()
    assert first == second
    registry = ImmutableDatasetRegistry(tmp_path / "datasets")
    path = registry.register(first)
    assert registry.register(second) == path
    assert registry.load(first.dataset_id) == first


def test_dataset_rejects_non_point_in_time_policy() -> None:
    values = _dataset().model_dump()
    values["point_in_time_policy"]["available_at_lte_decision_time"] = False
    with pytest.raises(ValueError, match="available_at"):
        CanonicalDatasetManifest.model_validate(values)


def test_triple_barrier_separates_feature_cutoff_and_future_label_window(tmp_path: Path) -> None:
    schema = LabelSchema(
        label_version="triple-barrier-v1",
        profit_barrier_fraction=Decimal("0.03"),
        stop_barrier_fraction=Decimal("0.015"),
        maximum_holding_seconds=7200,
        cost_model_version="cost-v1",
    )
    label = build_triple_barrier_label(
        candidate_id="candidate-1",
        market="SOL-EUR",
        decision_time=NOW,
        feature_cutoff=NOW,
        entry_price=Decimal("100"),
        future_bars=(
            {
                "timestamp": NOW + timedelta(hours=1),
                "high": "101",
                "low": "99",
                "close": "100.5",
            },
            {
                "timestamp": NOW + timedelta(hours=2),
                "high": "104",
                "low": "100",
                "close": "103",
            },
        ),
        schema=schema,
        fees_fraction=Decimal("0.005"),
        spread_fraction=Decimal("0.001"),
    )
    assert label.outcome is LabelOutcome.TARGET_FIRST
    assert label.gross_return == Decimal("0.03")
    assert label.net_return == Decimal("0.024")
    assert label.mfe == Decimal("0.04")
    assert label.mae == Decimal("-0.01")
    freeze = freeze_labels([label], tmp_path)
    assert freeze["immutable"] is True
    assert freeze_labels([label], tmp_path)["hash"] == freeze["hash"]


def test_same_bar_target_and_stop_is_conservative_ambiguous() -> None:
    schema = LabelSchema(
        label_version="triple-barrier-v1",
        profit_barrier_fraction=Decimal("0.03"),
        stop_barrier_fraction=Decimal("0.02"),
        maximum_holding_seconds=3600,
        cost_model_version="cost-v1",
    )
    label = build_triple_barrier_label(
        candidate_id="candidate-ambiguous",
        market="SOL-EUR",
        decision_time=NOW,
        feature_cutoff=NOW,
        entry_price=Decimal("100"),
        future_bars=(
            {
                "timestamp": NOW + timedelta(hours=1),
                "high": "104",
                "low": "97",
                "close": "101",
            },
        ),
        schema=schema,
    )
    assert label.outcome is LabelOutcome.AMBIGUOUS_SAME_BAR
    assert label.stop_first is True
    assert label.gross_return == Decimal("-0.02")


def test_feature_lifecycle_rejects_future_incomplete_and_insufficient_warmup() -> None:
    report = audit_point_in_time_features(
        [
            {
                "event_time": NOW,
                "available_at": NOW + timedelta(minutes=1),
                "is_final": False,
                "feature_version": "v1",
                "provenance": "test",
            }
        ],
        decision_time=NOW,
        minimum_warmup_rows=2,
    )
    codes = {row["code"] for row in report["failures"]}
    assert report["status"] == "HARD_REJECT"
    assert {"AVAILABLE_AFTER_DECISION", "INCOMPLETE_CANDLE_OR_FEATURE", "INSUFFICIENT_WARMUP"} <= codes


def test_model_registry_and_promotion_are_shadow_first(tmp_path: Path) -> None:
    model = _model()
    registry = ModelRegistry(tmp_path / "models")
    path = registry.register(model)
    assert registry.register(model) == path
    assert registry.load(model.model_id, model.artifact_hash) == model

    shadow = evaluate_model_promotion(
        current_status=ModelStatus.RESEARCH_ONLY,
        requested_status=ModelStatus.SHADOW,
        evidence={
            "dataset_immutable": True,
            "point_in_time_passed": True,
            "lookahead_passed": True,
            "purged_walk_forward_passed": True,
        },
    )
    assert shadow["permitted"] is True
    active = evaluate_model_promotion(
        current_status=ModelStatus.SHADOW,
        requested_status=ModelStatus.ACTIVE,
        evidence=shadow["checks"],
    )
    assert active["permitted"] is False
    assert "manual_authorization" in active["failed_checks"]
    assert active["live_authority_change"] is False


def test_expired_shadow_model_falls_back_to_deterministic_engine() -> None:
    result = evaluate_model_freshness(
        _model(),
        decision_time=NOW + timedelta(days=8),
    )
    assert result["status"] == "BLOCKED"
    assert result["shadow_inference_permitted"] is False
    assert result["fallback"] == "DETERMINISTIC_STRATEGY_ENGINE"
