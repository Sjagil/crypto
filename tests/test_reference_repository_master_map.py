from __future__ import annotations

import json
from pathlib import Path

from reporting.reference_repository_master_map import (
    REFERENCE_USAGE_REGISTRY,
    build_reference_master_map,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def test_usage_registry_covers_every_reference_with_real_native_call_sites() -> None:
    assert len(REFERENCE_USAGE_REGISTRY) == 9
    assert len({item.repo for item in REFERENCE_USAGE_REGISTRY}) == 9
    assert all(item.usage_count > 0 for item in REFERENCE_USAGE_REGISTRY)
    assert all(
        (WORKSPACE / call_site).is_file()
        for item in REFERENCE_USAGE_REGISTRY
        for call_site in item.native_call_sites
    )


def test_master_map_is_machine_readable_and_all_references_remain_unchanged(tmp_path: Path) -> None:
    destination = tmp_path / "reference_master_map.json"
    result = build_reference_master_map(WORKSPACE, output_path=destination)
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert result["repository_count"] == 9
    assert result["all_nine_present"] is True
    assert result["all_usage_counts_positive"] is True
    assert result["all_references_unchanged"] is True
    assert result["all_integrations_complete"] is True
    assert persisted["artifact_hash"] == result["artifact_hash"]
    assert result["orders_submitted"] == 0
    assert result["automatic_live_launch_permitted"] is False
