"""Content-addressed dataset and fail-closed model registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ml.contracts import CanonicalDatasetManifest, ModelArtifactManifest, ModelStatus
from utils.common import atomic_write_json, read_json


class ImmutableDatasetRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def register(self, manifest: CanonicalDatasetManifest) -> Path:
        path = self.root / manifest.dataset_id / "manifest.json"
        payload = manifest.model_dump(mode="json")
        if path.is_file():
            if read_json(path) != payload:
                raise ValueError("immutable dataset identity collision")
        else:
            atomic_write_json(path, payload)
        return path

    def load(self, dataset_id: str) -> CanonicalDatasetManifest:
        return CanonicalDatasetManifest.model_validate(
            read_json(self.root / dataset_id / "manifest.json")
        )


def evaluate_model_promotion(
    *,
    current_status: ModelStatus,
    requested_status: ModelStatus,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "dataset_immutable": evidence.get("dataset_immutable") is True,
        "point_in_time_passed": evidence.get("point_in_time_passed") is True,
        "lookahead_passed": evidence.get("lookahead_passed") is True,
        "purged_walk_forward_passed": evidence.get("purged_walk_forward_passed") is True,
        "economic_oos_positive": evidence.get("economic_oos_positive") is True,
        "calibration_acceptable": evidence.get("calibration_acceptable") is True,
        "minimum_sample_passed": evidence.get("minimum_sample_passed") is True,
        "manual_authorization": evidence.get("manual_authorization") is True,
        "live_validation_passed": evidence.get("live_validation_passed") is True,
    }
    if requested_status is ModelStatus.SHADOW:
        required = (
            "dataset_immutable",
            "point_in_time_passed",
            "lookahead_passed",
            "purged_walk_forward_passed",
        )
    elif requested_status is ModelStatus.CHALLENGER:
        required = (
            "dataset_immutable",
            "point_in_time_passed",
            "lookahead_passed",
            "purged_walk_forward_passed",
            "economic_oos_positive",
            "calibration_acceptable",
            "minimum_sample_passed",
        )
    elif requested_status in {ModelStatus.ADVISORY, ModelStatus.CANARY, ModelStatus.ACTIVE}:
        required = tuple(checks)
    elif requested_status in {ModelStatus.SUSPENDED, ModelStatus.REJECTED}:
        required = ()
    else:
        required = ("dataset_immutable", "point_in_time_passed", "lookahead_passed")
    failed = [name for name in required if not checks[name]]
    permitted = not failed
    return {
        "current_status": current_status.value,
        "requested_status": requested_status.value,
        "permitted": permitted,
        "required_checks": list(required),
        "checks": checks,
        "failed_checks": failed,
        "automatic_promotion": False,
        "live_authority_change": False,
    }


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def register(self, manifest: ModelArtifactManifest) -> Path:
        path = self.root / manifest.model_id / manifest.artifact_hash / "manifest.json"
        payload = manifest.model_dump(mode="json")
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError("immutable model artifact collision")
        else:
            atomic_write_json(path, payload)
        atomic_write_json(
            self.root / manifest.model_id / "latest.pointer.json",
            {
                "model_id": manifest.model_id,
                "artifact_hash": manifest.artifact_hash,
                "status": manifest.status.value,
                "manifest": str(path.resolve()),
            },
        )
        return path

    def load(self, model_id: str, artifact_hash: str) -> ModelArtifactManifest:
        return ModelArtifactManifest.model_validate(
            read_json(self.root / model_id / artifact_hash / "manifest.json")
        )


__all__ = [
    "ImmutableDatasetRegistry",
    "ModelRegistry",
    "evaluate_model_promotion",
]
