"""Authoritative, fail-closed accounting for every strategy evaluation.

The statistical denominator is an evaluation count, not merely a count of
distinct parameter dictionaries.  Re-running identical DNA on identical data
is deduplicated.  Selecting again on a genuinely new closed-data fingerprint is
a new research opportunity and therefore counts again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
)

GLOBAL_TRIAL_ACCOUNTING_SCHEMA = "global_trial_accounting_v1"
GLOBAL_TRIAL_ACCOUNTING_INDEX_SCHEMA = "global_trial_accounting_index_v1"
LEGACY_BASELINE_TRIAL_COUNT = 1_304


class GlobalTrialAccountingError(RuntimeError):
    """Raised when historical trial evidence is missing or inconsistent."""


def evaluation_identity(
    *,
    campaign_id: str,
    strategy_dna_hash: str,
    data_fingerprint: str,
) -> tuple[str, str, str]:
    """Return the canonical identity of one strategy/data evaluation."""

    values = (
        str(campaign_id).strip(),
        str(strategy_dna_hash).strip(),
        str(data_fingerprint).strip(),
    )
    if not all(values):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_IDENTITY_FIELD_MISSING"
        )
    return values


def deduplicate_evaluation_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deduplicate exact retries and reject conflicting identity reuse."""

    identities: dict[tuple[str, str, str], str] = {}
    strategy_identities: set[tuple[str, str]] = set()
    duplicate_retry_count = 0
    for raw_record in records:
        record = dict(raw_record)
        identity = evaluation_identity(
            campaign_id=str(record.get("campaign_id") or ""),
            strategy_dna_hash=str(
                record.get("strategy_dna_hash") or ""
            ),
            data_fingerprint=str(
                record.get("data_fingerprint") or ""
            ),
        )
        content_hash = str(record.get("content_hash") or "").strip()
        if not content_hash:
            content_hash = stable_hash(record, length=64)
        existing = identities.get(identity)
        if existing is not None:
            if existing != content_hash:
                raise GlobalTrialAccountingError(
                    "GLOBAL_TRIAL_IDENTITY_CONTENT_CONFLICT"
                )
            duplicate_retry_count += 1
            continue
        identities[identity] = content_hash
        strategy_identities.add(identity[:2])
    return {
        "evaluation_trial_count": len(identities),
        "unique_strategy_dna_count": len(strategy_identities),
        "duplicate_retry_count": duplicate_retry_count,
        "identity_root_hash": stable_hash(
            [
                {
                    "campaign_id": key[0],
                    "strategy_dna_hash": key[1],
                    "data_fingerprint": key[2],
                    "content_hash": identities[key],
                }
                for key in sorted(identities)
            ],
            length=64,
        ),
    }


def _require_report(
    path: Path,
    *,
    campaign: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_EVIDENCE_MISSING:{path}"
        )
    report = dict(read_json(path))
    if str(report.get("campaign") or "") != campaign:
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_CAMPAIGN_MISMATCH:{path}"
        )
    return report


def _aggregate_component(
    *,
    component_id: str,
    path: Path,
    campaign: str,
    evaluation_trial_count: int,
    unique_strategy_dna_count: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "campaign": campaign,
        "accounting_type": "IMMUTABLE_AGGREGATE_AT_BIRTH",
        "evaluation_trial_count": int(evaluation_trial_count),
        "unique_strategy_dna_count": int(
            unique_strategy_dna_count
        ),
        "evidence_path": str(path),
        "evidence_sha256": sha256_file(path),
        "notes": notes,
    }


def _legacy_component(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "diversified_rotation_campaign_v1.json"
    report = _require_report(
        path,
        campaign="DIVERSIFIED_ROTATION_V1",
    )
    total = int(report.get("total_known_trials") or 0)
    if total != LEGACY_BASELINE_TRIAL_COUNT:
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_LEGACY_BASELINE_MISMATCH"
        )
    return _aggregate_component(
        component_id="legacy_formal_campaigns_through_diversified_rotation",
        path=path,
        campaign="LEGACY_FORMAL_RESEARCH_BASELINE",
        evaluation_trial_count=total,
        unique_strategy_dna_count=total,
        notes=(
            "Accepted immutable cumulative baseline. Earlier individual "
            "trial records predate the content-addressed registry."
        ),
    )


def _breakout_component(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "portfolio_breakout_campaign_v1.json"
    report = _require_report(
        path,
        campaign="PORTFOLIO_BREAKOUT_V1",
    )
    prior = int(report.get("prior_trials_accounted") or 0)
    generated = int(report.get("parameters_tested") or 0)
    total = int(report.get("total_known_trials") or 0)
    if (
        prior != LEGACY_BASELINE_TRIAL_COUNT
        or generated <= 0
        or total != prior + generated
    ):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_BREAKOUT_RECONCILIATION_FAILED"
        )
    return _aggregate_component(
        component_id="portfolio_breakout_v1",
        path=path,
        campaign="PORTFOLIO_BREAKOUT_V1",
        evaluation_trial_count=generated,
        unique_strategy_dna_count=generated,
        notes="Eight preregistered breakout parameter sets.",
    )


def _storm_component(
    *,
    index_path: Path,
    campaign: str,
    component_id: str,
) -> dict[str, Any]:
    if not index_path.is_file():
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_STORM_INDEX_MISSING:{index_path}"
        )
    index = dict(read_json(index_path))
    if str(index.get("campaign") or "") != campaign:
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_STORM_CAMPAIGN_MISMATCH:{index_path}"
        )
    epochs = list(index.get("epochs") or [])
    if not epochs:
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_STORM_EPOCHS_EMPTY:{index_path}"
        )
    seen_epoch_identities: set[tuple[str, str]] = set()
    search_space_sizes: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for raw_epoch in epochs:
        epoch = dict(raw_epoch)
        data_fingerprint = str(
            epoch.get("data_fingerprint") or ""
        ).strip()
        report_path = Path(str(epoch.get("report") or ""))
        report = _require_report(
            report_path,
            campaign=campaign,
        )
        search_space_hash = str(
            epoch.get("strategy_search_space_hash")
            or report.get("search_space_hash")
            or ""
        ).strip()
        report_search_space_hash = str(
            report.get("search_space_hash") or ""
        ).strip()
        if (
            not data_fingerprint
            or not search_space_hash
            or (
                report_search_space_hash
                and report_search_space_hash != search_space_hash
            )
        ):
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_STORM_IDENTITY_INVALID"
            )
        trial_count = int(
            report.get("trial_count")
            or epoch.get("evaluated_strategy_count")
            or 0
        )
        if trial_count <= 0:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_STORM_COUNT_INVALID"
            )
        previous_size = search_space_sizes.setdefault(
            search_space_hash,
            trial_count,
        )
        if previous_size != trial_count:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_STORM_SEARCH_SPACE_SIZE_DRIFT"
            )
        identity = (search_space_hash, data_fingerprint)
        if identity in seen_epoch_identities:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_STORM_DUPLICATE_DATA_EPOCH"
            )
        seen_epoch_identities.add(identity)
        rows.append(
            {
                "epoch_id": str(epoch.get("epoch_id") or ""),
                "data_fingerprint": data_fingerprint,
                "strategy_search_space_hash": search_space_hash,
                "evaluated_strategy_count": trial_count,
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "report_total_known_trials_at_birth": int(
                    epoch.get(
                        "report_total_known_trials_at_birth"
                    )
                    or report.get("total_known_trials")
                    or 0
                ),
            }
        )
    return {
        "component_id": component_id,
        "campaign": campaign,
        "accounting_type": "SEARCH_SPACE_DNA_X_CLOSED_DATA_EPOCH",
        "evaluation_trial_count": sum(
            row["evaluated_strategy_count"] for row in rows
        ),
        "unique_strategy_dna_count": sum(
            search_space_sizes.values()
        ),
        "unique_data_epoch_count": len(rows),
        "unique_search_space_count": len(search_space_sizes),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "epochs": rows,
        "notes": (
            "An exact retry on an existing fingerprint is reused. A "
            "research ranking or selection run on a new closed-data "
            "fingerprint counts again; a frozen forward append does not."
        ),
    }


def _absolute_momentum_component(
    reports_dir: Path,
) -> dict[str, Any]:
    path = reports_dir / "absolute_momentum_campaign_v1.json"
    report = _require_report(
        path,
        campaign="ABSOLUTE_MOMENTUM_V1",
    )
    ledger = dict(report.get("exploration_ledger") or {})
    prior_key = "prior_formal_and_storm_trials"
    if prior_key not in ledger:
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_ABSOLUTE_MOMENTUM_LEDGER_MISSING"
        )
    generated = sum(
        int(value)
        for key, value in ledger.items()
        if key != prior_key
    )
    if generated <= 0 or int(report.get("total_known_trials") or 0) != sum(
        int(value) for value in ledger.values()
    ):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_ABSOLUTE_MOMENTUM_RECONCILIATION_FAILED"
        )
    component = _aggregate_component(
        component_id="absolute_momentum_v1_exploration",
        path=path,
        campaign="ABSOLUTE_MOMENTUM_V1",
        evaluation_trial_count=generated,
        unique_strategy_dna_count=generated,
        notes=(
            "Development, ablation, mean-reversion and midpoint paths; "
            "the prior cumulative field is excluded to prevent overlap."
        ),
    )
    component["exploration_ledger_excluding_prior"] = {
        key: int(value)
        for key, value in sorted(ledger.items())
        if key != prior_key
    }
    return component


def _content_addressed_component(
    registry_root: Path,
) -> dict[str, Any]:
    if not registry_root.is_dir():
        raise GlobalTrialAccountingError(
            f"GLOBAL_TRIAL_REGISTRY_ROOT_MISSING:{registry_root}"
        )
    all_records: list[dict[str, Any]] = []
    campaigns: list[dict[str, Any]] = []
    seen_campaigns: set[str] = set()
    for directory in sorted(
        path for path in registry_root.iterdir() if path.is_dir()
    ):
        index_path = directory / "index.json"
        if not index_path.is_file():
            raise GlobalTrialAccountingError(
                f"GLOBAL_TRIAL_REGISTRY_INDEX_MISSING:{directory}"
            )
        index = dict(read_json(index_path))
        campaign_id = str(index.get("campaign_id") or "").strip()
        if not campaign_id or campaign_id in seen_campaigns:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_REGISTRY_CAMPAIGN_DUPLICATE"
            )
        seen_campaigns.add(campaign_id)
        registry = ContentAddressedTrialRegistry(
            directory,
            campaign_id=campaign_id,
        )
        audit = registry.audit()
        campaign_records: list[dict[str, Any]] = []
        for raw_entry in index.get("entries") or []:
            record_path = Path(str(raw_entry.get("record_path") or ""))
            record = dict(read_json(record_path))
            campaign_records.append(record)
            all_records.append(record)
        deduplicated = deduplicate_evaluation_records(
            campaign_records
        )
        if (
            deduplicated["evaluation_trial_count"]
            != int(audit["unique_epoch_record_count"])
            or deduplicated["unique_strategy_dna_count"]
            != int(audit["unique_strategy_dna_count"])
        ):
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_REGISTRY_AUDIT_COUNT_MISMATCH"
            )
        campaigns.append(
            {
                "campaign_id": campaign_id,
                "evaluation_trial_count": deduplicated[
                    "evaluation_trial_count"
                ],
                "unique_strategy_dna_count": deduplicated[
                    "unique_strategy_dna_count"
                ],
                "unique_data_fingerprint_count": int(
                    audit["unique_data_fingerprint_count"]
                ),
                "duplicate_retry_count": deduplicated[
                    "duplicate_retry_count"
                ],
                "identity_root_hash": deduplicated[
                    "identity_root_hash"
                ],
                "registry_root_hash": str(audit["root_hash"]),
                "index_path": str(index_path),
                "index_sha256": sha256_file(index_path),
            }
        )
    global_deduplicated = deduplicate_evaluation_records(all_records)
    expected_evaluations = sum(
        int(row["evaluation_trial_count"]) for row in campaigns
    )
    if (
        global_deduplicated["evaluation_trial_count"]
        != expected_evaluations
    ):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_CROSS_REGISTRY_IDENTITY_COLLISION"
        )
    return {
        "component_id": "content_addressed_strategy_registries",
        "campaign": "MULTI_CAMPAIGN_CONTENT_ADDRESSED_REGISTRY",
        "accounting_type": "STRATEGY_DNA_X_CLOSED_DATA_EPOCH",
        **global_deduplicated,
        "campaign_count": len(campaigns),
        "campaigns": campaigns,
        "notes": (
            "Registry trial IDs already bind campaign, strategy DNA and "
            "data fingerprint. Byte-identical retries are deduplicated."
        ),
    }


def _persist_snapshot(
    *,
    lab_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    accounting_root = lab_dir / "trial_accounting"
    snapshots_dir = accounting_root / "snapshots"
    index_path = accounting_root / "index.json"
    report_path = (
        lab_dir / "reports" / "global_trial_accounting_v1.json"
    )
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    root_hash = str(payload["accounting_root_hash"])
    snapshot_path = snapshots_dir / f"{root_hash}.json"
    if snapshot_path.is_file():
        if read_json(snapshot_path) != payload:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_SNAPSHOT_HISTORY_REVISION"
            )
    else:
        atomic_write_json(snapshot_path, payload)
    index = (
        dict(read_json(index_path))
        if index_path.is_file()
        else {
            "schema_version": GLOBAL_TRIAL_ACCOUNTING_INDEX_SCHEMA,
            "entries": [],
            "root_hash": "0" * 64,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    )
    if (
        index.get("schema_version")
        != GLOBAL_TRIAL_ACCOUNTING_INDEX_SCHEMA
    ):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_INDEX_SCHEMA_MISMATCH"
        )
    _audit_accounting_index(index)
    entries = list(index.get("entries") or [])
    existing = next(
        (
            row
            for row in entries
            if row.get("accounting_root_hash") == root_hash
        ),
        None,
    )
    if existing is None:
        previous_hash = str(index.get("root_hash") or "0" * 64)
        entry = {
            "sequence_number": len(entries) + 1,
            "accounting_root_hash": root_hash,
            "global_multiple_testing_denominator": int(
                payload["global_multiple_testing_denominator"]
            ),
            "previous_record_hash": previous_hash,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": sha256_file(snapshot_path),
        }
        entry["record_hash"] = stable_hash(entry, length=64)
        entries.append(entry)
        index["entries"] = entries
        index["root_hash"] = entry["record_hash"]
        atomic_write_json(index_path, index)
    elif (
        str(existing.get("snapshot_sha256") or "")
        != sha256_file(snapshot_path)
    ):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_INDEX_SNAPSHOT_HASH_MISMATCH"
        )
    atomic_write_json(report_path, payload)
    return {
        **payload,
        "report": str(report_path),
        "snapshot": str(snapshot_path),
        "index": str(index_path),
    }


def _audit_accounting_index(index: Mapping[str, Any]) -> None:
    previous_hash = "0" * 64
    seen_roots: set[str] = set()
    for sequence, raw_entry in enumerate(
        index.get("entries") or [],
        start=1,
    ):
        entry = dict(raw_entry)
        record_hash = str(entry.pop("record_hash", ""))
        if int(entry.get("sequence_number") or 0) != sequence:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_SEQUENCE_MISMATCH"
            )
        if str(entry.get("previous_record_hash") or "") != previous_hash:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_CHAIN_MISMATCH"
            )
        if stable_hash(entry, length=64) != record_hash:
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_RECORD_CORRUPT"
            )
        accounting_root_hash = str(
            entry.get("accounting_root_hash") or ""
        )
        if (
            not accounting_root_hash
            or accounting_root_hash in seen_roots
        ):
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_ROOT_DUPLICATE"
            )
        seen_roots.add(accounting_root_hash)
        snapshot_path = Path(str(entry.get("snapshot_path") or ""))
        if (
            not snapshot_path.is_file()
            or sha256_file(snapshot_path)
            != str(entry.get("snapshot_sha256") or "")
        ):
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_SNAPSHOT_CORRUPT"
            )
        snapshot = dict(read_json(snapshot_path))
        if (
            str(snapshot.get("accounting_root_hash") or "")
            != accounting_root_hash
        ):
            raise GlobalTrialAccountingError(
                "GLOBAL_TRIAL_INDEX_ROOT_MISMATCH"
            )
        previous_hash = record_hash
    if previous_hash != str(index.get("root_hash") or ""):
        raise GlobalTrialAccountingError(
            "GLOBAL_TRIAL_INDEX_TIP_MISMATCH"
        )


def audit_global_trial_accounting(
    lab_dir: Path | str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Build the single authoritative multiple-testing denominator."""

    root = Path(lab_dir)
    reports_dir = root / "reports"
    components = [
        _legacy_component(reports_dir),
        _breakout_component(reports_dir),
        _storm_component(
            index_path=root / "storm_epochs" / "index.json",
            campaign="PORTFOLIO_STORM_V1",
            component_id="portfolio_storm_epoch_evaluations",
        ),
        _storm_component(
            index_path=root / "signal_storm_epochs" / "index.json",
            campaign="SIGNAL_SYNTHESIS_STORM_V1",
            component_id="signal_synthesis_storm_epoch_evaluations",
        ),
        _absolute_momentum_component(reports_dir),
        _content_addressed_component(
            root / "strategy_registry"
        ),
    ]
    evaluation_count = sum(
        int(component["evaluation_trial_count"])
        for component in components
    )
    unique_dna_count = sum(
        int(component["unique_strategy_dna_count"])
        for component in components
    )
    payload: dict[str, Any] = {
        "schema_version": GLOBAL_TRIAL_ACCOUNTING_SCHEMA,
        "status": "PASSED",
        "accounting_policy": {
            "trial_identity": (
                "CAMPAIGN_OR_SEARCH_SPACE_X_STRATEGY_DNA_X_"
                "CLOSED_DATA_FINGERPRINT"
            ),
            "same_dna_same_data": "DEDUPLICATED_EXACT_RETRY",
            "same_dna_new_closed_data_epoch": (
                "COUNTED_ONLY_FOR_RESEARCH_RERUN_OR_RESELECTION"
            ),
            "frozen_forward_observation": (
                "EXCLUDED_FROM_MULTIPLE_TESTING_DENOMINATOR"
            ),
            "historical_reports_mutated": False,
            "gaussian_smoothing_guarantees_pbo_pass": False,
        },
        "global_multiple_testing_denominator": evaluation_count,
        "evaluation_trial_count": evaluation_count,
        "unique_strategy_dna_equivalent_count": unique_dna_count,
        "components": components,
        "ai_governance_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    payload["accounting_root_hash"] = stable_hash(
        payload,
        length=64,
    )
    if persist:
        return _persist_snapshot(lab_dir=root, payload=payload)
    return payload


def global_multiple_testing_denominator(
    lab_dir: Path | str,
) -> int:
    """Return the current verified denominator without writing evidence."""

    return int(
        audit_global_trial_accounting(
            lab_dir,
            persist=False,
        )["global_multiple_testing_denominator"]
    )


def resolve_known_trial_count(
    lab_dir: Path | str,
    *,
    local_known_trial_count: int,
) -> int:
    """Use the global denominator once its required baseline exists.

    A clean repository must still be able to bootstrap the historical
    campaigns in their fixed order. Once the legacy baseline, both storm
    indexes, absolute-momentum ledger and registry root exist, any audit
    inconsistency is fatal and the global count is mandatory.
    """

    root = Path(lab_dir)
    required = (
        root / "reports" / "diversified_rotation_campaign_v1.json",
        root / "reports" / "portfolio_breakout_campaign_v1.json",
        root / "reports" / "absolute_momentum_campaign_v1.json",
        root / "storm_epochs" / "index.json",
        root / "signal_storm_epochs" / "index.json",
        root / "strategy_registry",
    )
    local = int(local_known_trial_count)
    if local <= 0:
        raise ValueError("local_known_trial_count must be positive")
    if not all(path.exists() for path in required):
        return local
    return max(
        local,
        global_multiple_testing_denominator(root),
    )


__all__ = [
    "GLOBAL_TRIAL_ACCOUNTING_SCHEMA",
    "GlobalTrialAccountingError",
    "audit_global_trial_accounting",
    "deduplicate_evaluation_records",
    "evaluation_identity",
    "global_multiple_testing_denominator",
    "resolve_known_trial_count",
]
