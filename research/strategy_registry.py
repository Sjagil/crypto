"""Plateau-aware selection and content-addressed strategy trial accounting."""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
)

GAUSSIAN_NEIGHBORHOOD_KERNEL = np.asarray(
    [0.05, 0.25, 0.40, 0.25, 0.05],
    dtype=float,
)
GAUSSIAN_NEIGHBORHOOD_OFFSETS = (-2, -1, 0, 1, 2)


class StrategyRegistryIntegrityError(RuntimeError):
    """Raised when registered trial history is missing or revised."""


def _sharpe(values: pd.DataFrame) -> pd.Series:
    standard = values.std(ddof=1).replace(0.0, np.nan)
    return (values.mean() / standard).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(-math.inf)


def gaussian_plateau_table(
    strategy_returns: pd.DataFrame,
    *,
    coordinates: Mapping[str, int],
    groups: Mapping[str, Hashable],
) -> pd.DataFrame:
    """Score only complete N±2 neighborhoods on the supplied observations."""

    matrix = (
        strategy_returns.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
    )
    names = [str(column) for column in matrix.columns]
    if set(names) != set(coordinates) or set(names) != set(groups):
        raise ValueError(
            "plateau coordinates and groups must cover every strategy"
        )
    sharpes = _sharpe(matrix)
    by_coordinate = {
        (groups[name], int(coordinates[name])): name
        for name in names
    }
    if len(by_coordinate) != len(names):
        raise ValueError("plateau group coordinates must be unique")
    rows: list[dict[str, Any]] = []
    for name in names:
        group = groups[name]
        coordinate = int(coordinates[name])
        neighbor_names = [
            by_coordinate.get((group, coordinate + offset))
            for offset in GAUSSIAN_NEIGHBORHOOD_OFFSETS
        ]
        complete = all(neighbor_names)
        if complete:
            selected_names = [str(value) for value in neighbor_names]
            neighbor_sharpes = sharpes.reindex(
                selected_names
            ).to_numpy(dtype=float)
            finite = bool(np.isfinite(neighbor_sharpes).all())
            score = (
                float(
                    np.dot(
                        neighbor_sharpes,
                        GAUSSIAN_NEIGHBORHOOD_KERNEL,
                    )
                )
                if finite
                else -math.inf
            )
            neighbor_net_returns = (
                (1.0 + matrix[selected_names]).prod() - 1.0
            ).to_numpy(dtype=float)
            all_positive = bool(
                np.all(neighbor_net_returns > 0.0)
            )
            minimum_sharpe = (
                float(neighbor_sharpes.min())
                if finite
                else -math.inf
            )
        else:
            selected_names = []
            score = -math.inf
            all_positive = False
            minimum_sharpe = -math.inf
        rows.append(
            {
                "strategy_id": name,
                "group": str(group),
                "coordinate": coordinate,
                "complete_neighborhood": complete,
                "neighbor_strategy_ids": selected_names,
                "gaussian_smoothed_sharpe": score,
                "minimum_neighbor_sharpe": minimum_sharpe,
                "all_neighbors_net_positive": all_positive,
                "plateau_eligible": bool(
                    complete
                    and all_positive
                    and math.isfinite(score)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy_id")


def plateau_selection_pbo(
    strategy_returns: pd.DataFrame,
    *,
    coordinates: Mapping[str, int],
    groups: Mapping[str, Hashable],
    group_count: int = 8,
) -> tuple[float | None, tuple[float, ...]]:
    """Estimate PBO for development selection by Gaussian plateau score."""

    matrix = (
        strategy_returns.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
    )
    if matrix.shape[1] < 2 or len(matrix) < 8:
        return None, ()
    selected_groups = min(group_count, len(matrix) // 2)
    if selected_groups % 2:
        selected_groups -= 1
    if selected_groups < 4:
        return None, ()
    observation_groups = [
        np.asarray(group, dtype=int)
        for group in np.array_split(
            np.arange(len(matrix)),
            selected_groups,
        )
    ]
    half = selected_groups // 2
    logits: list[float] = []
    for train_groups in itertools.combinations(
        range(selected_groups),
        half,
    ):
        test_groups = tuple(
            index
            for index in range(selected_groups)
            if index not in train_groups
        )
        train_positions = np.concatenate(
            [observation_groups[index] for index in train_groups]
        )
        test_positions = np.concatenate(
            [observation_groups[index] for index in test_groups]
        )
        train = matrix.iloc[np.sort(train_positions)]
        test = matrix.iloc[np.sort(test_positions)]
        plateau = gaussian_plateau_table(
            train,
            coordinates=coordinates,
            groups=groups,
        )
        eligible = plateau[
            plateau["plateau_eligible"].astype(bool)
        ]
        if eligible.empty:
            continue
        winner = str(
            eligible["gaussian_smoothed_sharpe"].idxmax()
        )
        test_scores = _sharpe(test)
        eligible_test_scores = test_scores.reindex(
            eligible.index
        )
        winner_score = float(eligible_test_scores[winner])
        below = float(
            (eligible_test_scores < winner_score).sum()
        )
        tied = float(
            (eligible_test_scores == winner_score).sum()
        )
        relative_rank = (
            below + 0.5 * tied
        ) / len(eligible_test_scores)
        relative_rank = min(
            1.0 - 1e-9,
            max(1e-9, relative_rank),
        )
        logits.append(
            math.log(relative_rank / (1.0 - relative_rank))
        )
    if not logits:
        return None, ()
    return (
        float(np.mean(np.asarray(logits) <= 0.0)),
        tuple(logits),
    )


class ContentAddressedTrialRegistry:
    """Append unique trials and reject any historical content revision."""

    schema_version = "strategy_trial_registry_v1"

    def __init__(
        self,
        root: Path | str,
        *,
        campaign_id: str,
    ) -> None:
        self.root = Path(root)
        self.campaign_id = str(campaign_id)
        self.records_dir = self.root / "records"
        self.index_path = self.root / "index.json"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def _empty_index(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "entries": [],
            "root_hash": "0" * 64,
            "unique_trial_count": 0,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    def index(self) -> dict[str, Any]:
        payload = (
            read_json(self.index_path)
            if self.index_path.is_file()
            else self._empty_index()
        )
        if (
            payload.get("schema_version") != self.schema_version
            or payload.get("campaign_id") != self.campaign_id
        ):
            raise StrategyRegistryIntegrityError(
                "STRATEGY_REGISTRY_IDENTITY_MISMATCH"
            )
        return dict(payload)

    def register(
        self,
        *,
        data_fingerprint: str,
        strategy_family: str,
        strategy_dna_hash: str,
        parameters: Mapping[str, Any],
        metrics_at_birth: Mapping[str, Any],
        return_path_hash: str,
        selection_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        trial_id = stable_hash(
            {
                "campaign_id": self.campaign_id,
                "data_fingerprint": data_fingerprint,
                "strategy_family": strategy_family,
                "strategy_dna_hash": strategy_dna_hash,
            },
            length=64,
        )
        record = {
            "schema_version": "strategy_trial_record_v1",
            "campaign_id": self.campaign_id,
            "trial_id": trial_id,
            "data_fingerprint": str(data_fingerprint),
            "strategy_family": str(strategy_family),
            "strategy_dna_hash": str(strategy_dna_hash),
            "parameters": dict(parameters),
            "metrics_at_birth": dict(metrics_at_birth),
            "return_path_hash": str(return_path_hash),
            "selection_metadata": dict(selection_metadata),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        record["content_hash"] = stable_hash(record, length=64)
        path = self.records_dir / f"{trial_id}.json"
        index = self.index()
        existing_entries = {
            str(row["trial_id"]): row
            for row in index["entries"]
        }
        existing = existing_entries.get(trial_id)
        if existing is not None:
            if not path.is_file():
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_RECORD_MISSING"
                )
            stored = read_json(path)
            stored_without_hash = dict(stored)
            stored_content_hash = str(
                stored_without_hash.pop("content_hash", "")
            )
            if (
                stored_content_hash != record["content_hash"]
                or stable_hash(
                    stored_without_hash,
                    length=64,
                )
                != stored_content_hash
            ):
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_HISTORY_REVISION"
                )
            return {
                "status": "REUSED_EXISTING_TRIAL",
                "trial_id": trial_id,
                "content_hash": record["content_hash"],
                "path": str(path),
                "unique_trial_count": int(
                    index["unique_trial_count"]
                ),
            }
        atomic_write_json(path, record)
        previous_hash = str(index.get("root_hash") or "0" * 64)
        entry = {
            "sequence_number": len(index["entries"]) + 1,
            "trial_id": trial_id,
            "content_hash": record["content_hash"],
            "previous_record_hash": previous_hash,
            "record_path": str(path),
        }
        entry["record_hash"] = stable_hash(entry, length=64)
        entries = [*index["entries"], entry]
        index.update(
            {
                "entries": entries,
                "root_hash": entry["record_hash"],
                "unique_trial_count": len(entries),
            }
        )
        atomic_write_json(self.index_path, index)
        return {
            "status": "REGISTERED_NEW_TRIAL",
            "trial_id": trial_id,
            "content_hash": record["content_hash"],
            "path": str(path),
            "unique_trial_count": len(entries),
        }

    def audit(self) -> dict[str, Any]:
        index = self.index()
        previous_hash = "0" * 64
        seen: set[str] = set()
        strategy_dna_hashes: set[str] = set()
        data_fingerprints: set[str] = set()
        for sequence, raw_entry in enumerate(
            index["entries"],
            start=1,
        ):
            entry = dict(raw_entry)
            record_hash = str(entry.pop("record_hash"))
            if entry.get("sequence_number") != sequence:
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_SEQUENCE_MISMATCH"
                )
            if entry.get("previous_record_hash") != previous_hash:
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_CHAIN_MISMATCH"
                )
            if stable_hash(entry, length=64) != record_hash:
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_ENTRY_CORRUPT"
                )
            trial_id = str(entry["trial_id"])
            if trial_id in seen:
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_DUPLICATE_TRIAL"
                )
            seen.add(trial_id)
            path = Path(str(entry["record_path"]))
            if not path.is_file():
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_RECORD_MISSING"
                )
            record = read_json(path)
            content_hash = str(record.pop("content_hash"))
            if (
                content_hash != entry["content_hash"]
                or stable_hash(record, length=64) != content_hash
            ):
                raise StrategyRegistryIntegrityError(
                    "STRATEGY_REGISTRY_RECORD_CORRUPT"
                )
            strategy_dna_hashes.add(str(record["strategy_dna_hash"]))
            data_fingerprints.add(str(record["data_fingerprint"]))
            previous_hash = record_hash
        if previous_hash != index.get("root_hash"):
            raise StrategyRegistryIntegrityError(
                "STRATEGY_REGISTRY_ROOT_MISMATCH"
            )
        if len(seen) != int(index.get("unique_trial_count") or 0):
            raise StrategyRegistryIntegrityError(
                "STRATEGY_REGISTRY_COUNT_MISMATCH"
            )
        return {
            "status": "PASSED",
            "campaign_id": self.campaign_id,
            # Backward-compatible name: a trial record is one immutable
            # strategy-DNA evaluation on one specific data epoch.
            "unique_trial_count": len(seen),
            "unique_epoch_record_count": len(seen),
            "unique_strategy_dna_count": len(strategy_dna_hashes),
            "unique_data_fingerprint_count": len(data_fingerprints),
            "strategy_dna_hashes": sorted(strategy_dna_hashes),
            "data_fingerprints": sorted(data_fingerprints),
            "root_hash": previous_hash,
            "index_path": str(self.index_path),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }


__all__ = [
    "GAUSSIAN_NEIGHBORHOOD_KERNEL",
    "GAUSSIAN_NEIGHBORHOOD_OFFSETS",
    "ContentAddressedTrialRegistry",
    "StrategyRegistryIntegrityError",
    "gaussian_plateau_table",
    "plateau_selection_pbo",
]
