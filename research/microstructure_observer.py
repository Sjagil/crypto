"""Append-only, orderless observations for the frozen crowding hypothesis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from data.orderflow_recorder import audit_microstructure_snapshots
from research.microstructure_preregistration import (
    FAMILY_ID,
    write_crowding_avoidance_plan,
)
from utils.common import atomic_write_json, read_json, stable_hash

OBSERVER_SCHEMA = "crowding_avoidance_observation_v1"
MANIFEST_SCHEMA = "crowding_avoidance_observer_manifest_v1"
ZERO_HASH = "0" * 64
REQUIRED_FEATURES = (
    "funding_zscore",
    "open_interest_change",
    "perpetual_spot_base_volume_ratio",
    "spot_cvd_robust_zscore",
)


class MicrostructureObserverIntegrityError(RuntimeError):
    """Raised when immutable observer evidence no longer reconciles."""


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("microstructure timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _observation_hash(record: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            key: value
            for key, value in record.items()
            if key != "observation_hash"
        },
        length=64,
    )


def _market_features(row: Mapping[str, Any]) -> dict[str, Any]:
    positioning = dict(row.get("derivatives_positioning") or {})
    return {
        "funding_zscore": positioning.get("funding_zscore"),
        "open_interest_change": positioning.get(
            "open_interest_change"
        ),
        "perpetual_spot_base_volume_ratio": row.get(
            "perpetual_spot_base_volume_ratio"
        ),
        "spot_cvd_robust_zscore": row.get(
            "spot_cvd_robust_zscore"
        ),
    }


def _evaluate_market(
    row: Mapping[str, Any],
    *,
    dna: list[dict[str, Any]],
) -> dict[str, Any]:
    features = _market_features(row)
    missing = [
        name
        for name in REQUIRED_FEATURES
        if features.get(name) is None
    ]
    if missing:
        return {
            "market": str(row.get("market")),
            "status": "FEATURE_WARMUP",
            "missing_features": missing,
            "features": features,
            "dna_results": [
                {
                    "dna_id": parameters["id"],
                    "status": "NOT_EVALUATED",
                    "block_new_long": None,
                }
                for parameters in dna
            ],
            "block_signal_count": 0,
        }

    funding_z = float(features["funding_zscore"])
    oi_change = float(features["open_interest_change"])
    volume_ratio = float(
        features["perpetual_spot_base_volume_ratio"]
    )
    spot_cvd_z = float(features["spot_cvd_robust_zscore"])
    results: list[dict[str, Any]] = []
    for parameters in dna:
        checks = {
            "funding_z_min": (
                funding_z >= float(parameters["funding_z_min"])
            ),
            "open_interest_change_min": (
                oi_change
                >= float(parameters["open_interest_change_min"])
            ),
            "perpetual_spot_volume_ratio_min": (
                volume_ratio
                >= float(
                    parameters[
                        "perpetual_spot_volume_ratio_min"
                    ]
                )
            ),
            "spot_cvd_z_max": (
                spot_cvd_z <= float(parameters["spot_cvd_z_max"])
            ),
        }
        results.append(
            {
                "dna_id": parameters["id"],
                "status": "EVALUATED",
                "threshold_checks": checks,
                "block_new_long": all(checks.values()),
            }
        )
    return {
        "market": str(row.get("market")),
        "status": "EVALUATED",
        "missing_features": [],
        "features": features,
        "dna_results": results,
        "block_signal_count": sum(
            bool(result["block_new_long"]) for result in results
        ),
    }


def _build_observation(
    *,
    snapshot_path: Path,
    snapshot: Mapping[str, Any],
    snapshot_audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    sequence_number: int,
    previous_observation_hash: str,
) -> dict[str, Any]:
    hour_start = _utc(snapshot["hour_start"])
    hour_end = _utc(snapshot["hour_end"])
    source_eligible = bool(snapshot_audit.get("eligible"))
    if source_eligible:
        markets = [
            _evaluate_market(
                row,
                dna=[dict(item) for item in plan["primary_dna"]],
            )
            for row in snapshot.get("markets") or []
        ]
        status = (
            "EVALUATED"
            if markets
            and all(row["status"] == "EVALUATED" for row in markets)
            else "FEATURE_WARMUP"
        )
        reason_codes = (
            []
            if status == "EVALUATED"
            else ["REQUIRED_CAUSAL_FEATURE_HISTORY_INSUFFICIENT"]
        )
    else:
        markets = []
        status = "DATA_GAP_NOT_EVALUATED"
        reason_codes = list(snapshot_audit.get("reason_codes") or [])
    block_signal_count = sum(
        int(row["block_signal_count"]) for row in markets
    )
    body = {
        "schema_version": OBSERVER_SCHEMA,
        "family_id": FAMILY_ID,
        "plan_hash": plan["plan_hash"],
        "sequence_number": sequence_number,
        "observation_id": (
            f"{FAMILY_ID}:{hour_start.strftime('%Y%m%dT%H0000Z')}"
        ),
        "previous_observation_hash": previous_observation_hash,
        "hour_start": hour_start.isoformat(),
        "hour_end": hour_end.isoformat(),
        "recorded_at": snapshot["finalized_at"],
        "available_at": snapshot["finalized_at"],
        "source_snapshot": str(snapshot_path.resolve()),
        "source_snapshot_hash": snapshot["snapshot_hash"],
        "source_snapshot_eligible": source_eligible,
        "status": status,
        "reason_codes": reason_codes,
        "observation_cadence": "EVERY_FINALIZED_UTC_HOUR",
        "frozen_decision_horizon": plan["decision_horizon"],
        "four_hour_decision_boundary": hour_end.hour % 4 == 0,
        "markets": markets,
        "evaluated_market_count": sum(
            row["status"] == "EVALUATED" for row in markets
        ),
        "feature_warmup_market_count": sum(
            row["status"] == "FEATURE_WARMUP" for row in markets
        ),
        "block_signal_count": block_signal_count,
        "any_block_signal_observed": block_signal_count > 0,
        "research_selection_performed": False,
        "parameters_changed": False,
        "synthetic_data_used": False,
        "portfolio_target_generated": False,
        "orders_generated": 0,
        "paper_permitted": False,
        "live_permitted": False,
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
    }
    return {
        **body,
        "observation_hash": _observation_hash(body),
    }


def audit_crowding_observer(
    observer_directory: Path,
    *,
    expected_plan_hash: str | None = None,
) -> dict[str, Any]:
    """Verify the complete observation chain and its source snapshots."""

    observation_directory = observer_directory / "observations"
    paths = sorted(observation_directory.glob("*.json"))
    failures: list[str] = []
    previous_hash = ZERO_HASH
    statuses: dict[str, int] = {}
    block_signal_count = 0
    for sequence_number, path in enumerate(paths, start=1):
        try:
            record = dict(read_json(path))
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                f"UNREADABLE_OBSERVATION:{path.name}:{type(exc).__name__}"
            )
            continue
        if record.get("schema_version") != OBSERVER_SCHEMA:
            failures.append(f"UNSUPPORTED_SCHEMA:{path.name}")
        if int(record.get("sequence_number") or 0) != sequence_number:
            failures.append(f"SEQUENCE_MISMATCH:{path.name}")
        if record.get("previous_observation_hash") != previous_hash:
            failures.append(f"CHAIN_PREDECESSOR_MISMATCH:{path.name}")
        actual_hash = str(record.get("observation_hash") or "")
        if actual_hash != _observation_hash(record):
            failures.append(f"OBSERVATION_HASH_MISMATCH:{path.name}")
        expected_name = (
            _utc(record["hour_start"]).strftime("%Y%m%dT%H0000Z")
            + ".json"
        )
        if path.name != expected_name:
            failures.append(f"OBSERVATION_FILENAME_MISMATCH:{path.name}")
        if (
            expected_plan_hash is not None
            and record.get("plan_hash") != expected_plan_hash
        ):
            failures.append(f"PLAN_HASH_MISMATCH:{path.name}")
        source_path = Path(str(record.get("source_snapshot") or ""))
        if not source_path.is_file():
            failures.append(f"SOURCE_SNAPSHOT_MISSING:{path.name}")
        else:
            try:
                source = dict(read_json(source_path))
            except (OSError, TypeError, ValueError):
                failures.append(
                    f"SOURCE_SNAPSHOT_UNREADABLE:{path.name}"
                )
            else:
                if (
                    source.get("snapshot_hash")
                    != record.get("source_snapshot_hash")
                ):
                    failures.append(
                        f"SOURCE_SNAPSHOT_HASH_MISMATCH:{path.name}"
                    )
                source_body = {
                    key: value
                    for key, value in source.items()
                    if key != "snapshot_hash"
                }
                if source.get("snapshot_hash") != stable_hash(
                    source_body,
                    length=64,
                ):
                    failures.append(
                        f"SOURCE_SNAPSHOT_CHECKSUM_INVALID:{path.name}"
                    )
        if int(record.get("orders_generated") or 0) != 0:
            failures.append(f"ORDER_SIDE_EFFECT_DETECTED:{path.name}")
        if record.get("live_permitted") is not False:
            failures.append(f"LIVE_PERMISSION_DETECTED:{path.name}")
        status = str(record.get("status") or "UNKNOWN")
        statuses[status] = statuses.get(status, 0) + 1
        block_signal_count += int(
            record.get("block_signal_count") or 0
        )
        previous_hash = actual_hash
    return {
        "schema_version": "crowding_avoidance_observer_audit_v1",
        "status": "PASSED" if not failures else "FAILED",
        "family_id": FAMILY_ID,
        "record_count": len(paths),
        "root_hash": previous_hash,
        "status_counts": statuses,
        "block_signal_count": block_signal_count,
        "failures": failures,
        "orders_generated": 0,
        "paper_permitted": False,
        "live_permitted": False,
    }


def observe_microstructure_snapshots(
    *,
    feature_directory: Path,
    observer_directory: Path,
    plan_path: Path,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Append every sealed source hour once without selecting a winner."""

    plan = write_crowding_avoidance_plan(plan_path)
    observer_directory.mkdir(parents=True, exist_ok=True)
    observation_directory = observer_directory / "observations"
    observation_directory.mkdir(parents=True, exist_ok=True)
    initial_audit = audit_crowding_observer(
        observer_directory,
        expected_plan_hash=str(plan["plan_hash"]),
    )
    if initial_audit["status"] != "PASSED":
        raise MicrostructureObserverIntegrityError(
            "CROWDING_OBSERVER_HISTORY_INVALID:"
            + ",".join(initial_audit["failures"])
        )

    source_audit = audit_microstructure_snapshots(
        feature_directory,
        ledger_root=ledger_root,
    )
    audit_by_path = {
        str(Path(row["path"]).resolve()): row
        for row in (
            list(source_audit.get("excluded_snapshots") or [])
            + [
                row
                for row in _all_snapshot_audits(
                    feature_directory,
                    ledger_root=ledger_root,
                )
                if row.get("eligible")
            ]
        )
    }
    previous_hash = str(initial_audit["root_hash"])
    existing_count = int(initial_audit["record_count"])
    source_paths = sorted(feature_directory.glob("*.json"))
    existing_paths = sorted(observation_directory.glob("*.json"))
    if existing_count > len(source_paths):
        raise MicrostructureObserverIntegrityError(
            "CROWDING_OBSERVER_SOURCE_HISTORY_SHRANK"
        )
    appended = 0
    for index, snapshot_path in enumerate(source_paths):
        target = observation_directory / snapshot_path.name
        if index < existing_count:
            if target != existing_paths[index]:
                raise MicrostructureObserverIntegrityError(
                    "CROWDING_OBSERVER_SOURCE_ORDER_CHANGED"
                )
            continue
        snapshot = dict(read_json(snapshot_path))
        path_key = str(snapshot_path.resolve())
        snapshot_audit = audit_by_path.get(path_key)
        if snapshot_audit is None:
            raise MicrostructureObserverIntegrityError(
                f"CROWDING_SOURCE_AUDIT_MISSING:{snapshot_path.name}"
            )
        observation = _build_observation(
            snapshot_path=snapshot_path,
            snapshot=snapshot,
            snapshot_audit=snapshot_audit,
            plan=plan,
            sequence_number=index + 1,
            previous_observation_hash=previous_hash,
        )
        if target.is_file():
            if read_json(target) != observation:
                raise MicrostructureObserverIntegrityError(
                    f"CROWDING_OBSERVER_HISTORY_REVISION:{target.name}"
                )
        else:
            atomic_write_json(target, observation)
            appended += 1
        previous_hash = str(observation["observation_hash"])

    final_audit = audit_crowding_observer(
        observer_directory,
        expected_plan_hash=str(plan["plan_hash"]),
    )
    if final_audit["status"] != "PASSED":
        raise MicrostructureObserverIntegrityError(
            "CROWDING_OBSERVER_APPEND_FAILED:"
            + ",".join(final_audit["failures"])
        )
    manifest_body = {
        "schema_version": MANIFEST_SCHEMA,
        "family_id": FAMILY_ID,
        "plan_path": str(plan_path.resolve()),
        "plan_hash": plan["plan_hash"],
        "source_directory": str(feature_directory.resolve()),
        "observer_directory": str(observer_directory.resolve()),
        "source_snapshot_count": len(source_paths),
        "new_observation_count": appended,
        "observation_count": final_audit["record_count"],
        "observation_status_counts": final_audit["status_counts"],
        "block_signal_count": final_audit["block_signal_count"],
        "chain_root_hash": final_audit["root_hash"],
        "audit_status": final_audit["status"],
        "selection_performed": False,
        "backtest_performed": False,
        "orders_generated": 0,
        "paper_permitted": False,
        "live_permitted": False,
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
    }
    manifest = {
        **manifest_body,
        "manifest_hash": stable_hash(manifest_body, length=64),
    }
    atomic_write_json(observer_directory / "manifest.json", manifest)
    atomic_write_json(
        observer_directory / "audit.json",
        final_audit,
    )
    return {
        **manifest,
        "observer_audit": final_audit,
    }


def _all_snapshot_audits(
    feature_directory: Path,
    *,
    ledger_root: Path | None,
) -> list[dict[str, Any]]:
    """Expose per-file audit rows through isolated one-file directories."""

    full = audit_microstructure_snapshots(
        feature_directory,
        ledger_root=ledger_root,
    )
    excluded_by_path = {
        str(Path(row["path"]).resolve()): row
        for row in full.get("excluded_snapshots") or []
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(feature_directory.glob("*.json")):
        key = str(path.resolve())
        if key in excluded_by_path:
            rows.append(excluded_by_path[key])
        else:
            rows.append(
                {
                    "path": str(path),
                    "hour_start": dict(read_json(path)).get(
                        "hour_start"
                    ),
                    "eligible": True,
                    "reason_codes": [],
                    "snapshot_hash": dict(read_json(path)).get(
                        "snapshot_hash"
                    ),
                }
            )
    return rows


__all__ = [
    "MicrostructureObserverIntegrityError",
    "audit_crowding_observer",
    "observe_microstructure_snapshots",
]
