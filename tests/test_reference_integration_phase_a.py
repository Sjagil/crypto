from __future__ import annotations

import json
from pathlib import Path

from reporting.reference_integration_phase_a import (
    REFERENCE_ASSIGNMENTS,
    build_phase_a_inventory,
    snapshot_reference,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def test_all_reference_pins_licenses_sources_and_cleanliness_are_verified() -> None:
    snapshots = [snapshot_reference(WORKSPACE, row) for row in REFERENCE_ASSIGNMENTS]

    assert len(snapshots) == 9
    assert all(row["reference_clean"] for row in snapshots)
    assert all(row["commit_verified"] for row in snapshots)
    assert all(row["tree_verified"] for row in snapshots)
    assert all(row["branch_verified"] for row in snapshots)
    assert all(row["remote_verified"] for row in snapshots)
    assert all(row["license_verified"] for row in snapshots)
    assert all(row["source_symbols_verified"] for row in snapshots)


def test_each_reference_has_one_unique_primary_responsibility() -> None:
    roles = [row.primary_responsibility for row in REFERENCE_ASSIGNMENTS]
    assert len(roles) == len(set(roles)) == 9
    assert all(row.integration_mode.startswith("C_CONCEPT_REFERENCE_ONLY") for row in REFERENCE_ASSIGNMENTS)
    assert not any(row.runtime_dependency for row in REFERENCE_ASSIGNMENTS)


def test_phase_a_artifact_is_content_addressed_and_side_effect_free(tmp_path: Path) -> None:
    result = build_phase_a_inventory(WORKSPACE, output_root=tmp_path)
    artifact = Path(result["artifact_path"])
    persisted = json.loads(artifact.read_text(encoding="utf-8"))

    assert result["phase_status"] == "PASSED"
    assert artifact.parent.name == result["artifact_hash"]
    assert persisted["artifact_hash"] == result["artifact_hash"]
    assert all(result["phase_b_entry_gates"].values())
    assert set(result["side_effects"].values()) == {0}
    assert result["architecture_invariants"]["ml_authority"] == "SHADOW_ONLY"
    assert result["production_baseline"]["portfolio_target_owner"] == "portfolio.targets"
    assert any(
        row["order_intent_mentions"] > 0
        for row in result["production_baseline"]["direct_order_paths"]
    )
