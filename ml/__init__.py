"""Fail-closed, shadow-first ML data and model lifecycle contracts."""

from ml.contracts import (
    CanonicalDatasetManifest,
    CanonicalLabelRecord,
    LabelSchema,
    ModelArtifactManifest,
    ModelStatus,
)
from ml.labels import build_triple_barrier_label, freeze_labels
from ml.lifecycle import audit_point_in_time_features, evaluate_model_freshness
from ml.registry import ImmutableDatasetRegistry, ModelRegistry, evaluate_model_promotion

__all__ = [
    "CanonicalDatasetManifest",
    "CanonicalLabelRecord",
    "ImmutableDatasetRegistry",
    "LabelSchema",
    "ModelArtifactManifest",
    "ModelRegistry",
    "ModelStatus",
    "audit_point_in_time_features",
    "build_triple_barrier_label",
    "evaluate_model_freshness",
    "evaluate_model_promotion",
    "freeze_labels",
]
