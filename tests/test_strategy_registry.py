from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.strategy_registry import (
    ContentAddressedTrialRegistry,
    StrategyRegistryIntegrityError,
    gaussian_plateau_table,
    plateau_selection_pbo,
)
from utils.common import atomic_write_json, read_json


def _matrix() -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    generator = np.random.default_rng(42)
    rows = 160
    data: dict[str, np.ndarray] = {}
    coordinates: dict[str, int] = {}
    groups: dict[str, str] = {}
    common = generator.normal(0.0008, 0.006, rows)
    for coordinate in range(-3, 4):
        name = f"strategy_{coordinate:+d}"
        penalty = abs(coordinate) * 0.00005
        data[name] = common - penalty + generator.normal(
            0.0,
            0.0002,
            rows,
        )
        coordinates[name] = coordinate
        groups[name] = "same-nuisance-parameters"
    return pd.DataFrame(data), coordinates, groups


def test_gaussian_plateau_requires_complete_positive_neighborhood() -> None:
    matrix, coordinates, groups = _matrix()
    table = gaussian_plateau_table(
        matrix,
        coordinates=coordinates,
        groups=groups,
    )

    assert not table.loc["strategy_-3", "complete_neighborhood"]
    assert table.loc["strategy_+0", "complete_neighborhood"]
    assert table.loc["strategy_+0", "all_neighbors_net_positive"]
    assert table.loc["strategy_+0", "plateau_eligible"]
    assert table.loc["strategy_+0", "neighbor_strategy_ids"] == [
        "strategy_-2",
        "strategy_-1",
        "strategy_+0",
        "strategy_+1",
        "strategy_+2",
    ]


def test_plateau_selection_pbo_is_deterministic_and_finite() -> None:
    matrix, coordinates, groups = _matrix()

    first = plateau_selection_pbo(
        matrix,
        coordinates=coordinates,
        groups=groups,
    )
    second = plateau_selection_pbo(
        matrix,
        coordinates=coordinates,
        groups=groups,
    )

    assert first == second
    assert first[0] is not None
    assert 0.0 <= float(first[0]) <= 1.0
    assert len(first[1]) > 0


def test_registry_is_content_addressed_idempotent_and_auditable(
    tmp_path,
) -> None:
    registry = ContentAddressedTrialRegistry(
        tmp_path,
        campaign_id="PLATEAU_V1",
    )
    payload = {
        "data_fingerprint": "data",
        "strategy_family": "absolute_momentum",
        "strategy_dna_hash": "dna",
        "parameters": {"windows": (20, 60, 120)},
        "metrics_at_birth": {"development_sharpe": 1.0},
        "return_path_hash": "returns",
        "selection_metadata": {"plateau_eligible": True},
    }

    first = registry.register(**payload)
    second = registry.register(**payload)
    audit = registry.audit()

    assert first["status"] == "REGISTERED_NEW_TRIAL"
    assert second["status"] == "REUSED_EXISTING_TRIAL"
    assert first["trial_id"] == second["trial_id"]
    assert audit["status"] == "PASSED"
    assert audit["unique_trial_count"] == 1
    assert audit["unique_epoch_record_count"] == 1
    assert audit["unique_strategy_dna_count"] == 1
    assert audit["unique_data_fingerprint_count"] == 1


def test_registry_separates_strategy_dna_from_data_epochs(tmp_path) -> None:
    registry = ContentAddressedTrialRegistry(
        tmp_path,
        campaign_id="PLATEAU_V1",
    )
    shared = {
        "strategy_family": "absolute_momentum",
        "strategy_dna_hash": "dna",
        "parameters": {"windows": (20, 60, 120)},
        "metrics_at_birth": {"development_sharpe": 1.0},
        "return_path_hash": "returns",
        "selection_metadata": {"plateau_eligible": True},
    }

    registry.register(data_fingerprint="epoch-one", **shared)
    registry.register(data_fingerprint="epoch-two", **shared)
    audit = registry.audit()

    assert audit["unique_trial_count"] == 2
    assert audit["unique_epoch_record_count"] == 2
    assert audit["unique_strategy_dna_count"] == 1
    assert audit["unique_data_fingerprint_count"] == 2
    assert audit["strategy_dna_hashes"] == ["dna"]
    assert audit["data_fingerprints"] == ["epoch-one", "epoch-two"]


def test_registry_rejects_record_and_chain_revision(tmp_path) -> None:
    registry = ContentAddressedTrialRegistry(
        tmp_path,
        campaign_id="PLATEAU_V1",
    )
    registered = registry.register(
        data_fingerprint="data",
        strategy_family="absolute_momentum",
        strategy_dna_hash="dna",
        parameters={"window": 20},
        metrics_at_birth={"development_sharpe": 1.0},
        return_path_hash="returns",
        selection_metadata={"plateau_eligible": True},
    )
    record_path = registered["path"]
    record = read_json(record_path)
    record["metrics_at_birth"]["development_sharpe"] = 99.0
    atomic_write_json(record_path, record)
    with pytest.raises(
        StrategyRegistryIntegrityError,
        match="RECORD_CORRUPT",
    ):
        registry.audit()

    clean_root = tmp_path / "chain"
    clean = ContentAddressedTrialRegistry(
        clean_root,
        campaign_id="PLATEAU_V1",
    )
    clean.register(
        data_fingerprint="data",
        strategy_family="absolute_momentum",
        strategy_dna_hash="dna",
        parameters={"window": 20},
        metrics_at_birth={"development_sharpe": 1.0},
        return_path_hash="returns",
        selection_metadata={"plateau_eligible": True},
    )
    index = deepcopy(clean.index())
    index["root_hash"] = "f" * 64
    atomic_write_json(clean.index_path, index)
    with pytest.raises(
        StrategyRegistryIntegrityError,
        match="ROOT_MISMATCH",
    ):
        clean.audit()
