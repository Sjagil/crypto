"""Consolidated evidence matrix for research, paper and live readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from data.orderflow_recorder import verify_orderflow_ledger
from data.service_supervisor import CollectorSupervisor
from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
    utc_now,
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {}


def _requirement(
    requirement_id: str,
    *,
    status: str,
    evidence: Mapping[str, Any],
    blocks_live: bool,
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "status": status,
        "blocks_live": blocks_live,
        "evidence": dict(evidence),
    }


def prospective_readiness_report(settings: Any) -> dict[str, Any]:
    """Build one fail-closed matrix from current authoritative artifacts."""

    reports = settings.paths.lab_dir / "reports"
    operations = settings.paths.output_dir / "operations"
    checkpoints = settings.paths.checkpoints_dir
    volume_path = (
        reports / "volume_strategy_catalog_campaign_v1.json"
    )
    volume = _read(volume_path)
    readiness_path = operations / "microstructure_readiness.json"
    readiness = _read(readiness_path)
    stream_health_path = (
        operations / "orderflow_stream_health.json"
    )
    stream_health = _read(stream_health_path)
    supervisor = CollectorSupervisor(
        checkpoints_directory=checkpoints,
        operations_directory=operations,
    ).status()
    canary_path = (
        settings.paths.lab_dir
        / "manifests"
        / "live_canary_policy_v1.json"
    )
    canary = _read(canary_path)
    ai_path = reports / "ai_governance_status_v1.json"
    ai = _read(ai_path)
    ledger_audit = verify_orderflow_ledger(
        settings.paths.context_data_dir / "orderflow_stream"
    )
    regime_path = (
        reports / "volume_strategy_catalog_regimes_v1.csv"
    )
    volume_complete = (
        volume.get("status") == "COMPLETED_NOT_PROMOTED"
        and int(volume.get("generated_trial_count") or 0) > 0
    )
    supervisor_live = (
        supervisor.get("status")
        in {"MONITORING", "RUNNING_CHILD"}
        and not supervisor.get("disabled")
    )
    canary_registered = (
        canary.get("activation_status")
        == "REGISTERED_DISABLED"
        and not canary.get("enabled")
    )
    orderflow_ready = bool(readiness.get("backtest_permitted"))
    financial_candidate = bool(
        volume.get("research_pass")
        and int(volume.get("paper_candidates") or 0) > 0
        and volume.get("live_ready")
    )
    multiple_testing = dict(
        volume.get("multiple_testing_allowed_only") or {}
    )
    requirements = [
        _requirement(
            "PUBLIC_TRADE_AND_L2_STREAM",
            status=(
                "PASSED"
                if stream_health.get("status") == "HEALTHY"
                else "FAILED"
            ),
            evidence={
                "path": str(stream_health_path),
                "state": stream_health.get("status"),
                "record_count": stream_health.get("record_count"),
                "sequence_gaps": (
                    stream_health.get("provider") or {}
                ).get("sequence_gaps"),
                "dropped_messages": (
                    stream_health.get("provider") or {}
                ).get("dropped_messages"),
            },
            blocks_live=stream_health.get("status") != "HEALTHY",
        ),
        _requirement(
            "APPEND_ONLY_LEDGER_INTEGRITY",
            status=str(ledger_audit["status"]),
            evidence={
                "record_count": ledger_audit["record_count"],
                "root_hash": ledger_audit["root_hash"],
                "integrity_failures": ledger_audit[
                    "integrity_failures"
                ],
            },
            blocks_live=ledger_audit["status"] != "PASSED",
        ),
        _requirement(
            "AUTOMATIC_CRASH_RECOVERY",
            status="PASSED" if supervisor_live else "FAILED",
            evidence={
                "path": str(
                    operations
                    / "collector_supervisor_health.json"
                ),
                "supervisor_status": supervisor.get("status"),
                "restart_count": supervisor.get("restart_count"),
                "stale_service_locks_recovered": supervisor.get(
                    "stale_service_locks_recovered"
                ),
                "supervisor_pid": supervisor.get("supervisor_pid"),
                "child_pid": supervisor.get("child_pid"),
            },
            blocks_live=not supervisor_live,
        ),
        _requirement(
            "CANDLE_VOLUME_COMBINATION_CAMPAIGN",
            status="PASSED" if volume_complete else "MISSING",
            evidence={
                "path": str(volume_path),
                "generated_trials": volume.get(
                    "generated_trial_count"
                ),
                "allowed_universe_trials": volume.get(
                    "allowed_universe_trials"
                ),
                "market_timeframe_pairs": volume.get(
                    "market_timeframe_pairs"
                ),
                "total_known_trials": volume.get(
                    "total_known_trials"
                ),
            },
            blocks_live=not volume_complete,
        ),
        _requirement(
            "REGIME_PERFORMANCE_ATTRIBUTION",
            status="PASSED" if regime_path.is_file() else "MISSING",
            evidence={
                "path": str(regime_path),
                "best_regime_records": len(
                    volume.get(
                        "best_regime_by_strategy_axis"
                    )
                    or []
                ),
            },
            blocks_live=not regime_path.is_file(),
        ),
        _requirement(
            "PROSPECTIVE_ORDERFLOW_HISTORY_90D",
            status=(
                "PASSED" if orderflow_ready else "COLLECTING"
            ),
            evidence={
                "path": str(readiness_path),
                "complete_hours": readiness.get(
                    "complete_hours",
                    0,
                ),
                "consecutive_complete_hours": readiness.get(
                    "consecutive_complete_hours",
                    0,
                ),
                "minimum_days": 90,
                "synthetic_data_used": readiness.get(
                    "synthetic_data_used"
                ),
            },
            blocks_live=not orderflow_ready,
        ),
        _requirement(
            "FINANCIAL_STRATEGY_PROMOTION_GATES",
            status=(
                "PASSED"
                if financial_candidate
                else "FAILED_CLOSED"
            ),
            evidence={
                "research_pass": volume.get("research_pass"),
                "paper_candidates": volume.get(
                    "paper_candidates"
                ),
                "live_ready": volume.get("live_ready"),
                "holdout_status": volume.get("holdout_status"),
                "ordinary_pbo": multiple_testing.get(
                    "probability_of_backtest_overfitting"
                ),
                "plateau_selection_pbo": multiple_testing.get(
                    "plateau_selection_pbo"
                ),
                "white_reality_check_pvalue": (
                    multiple_testing.get(
                        "white_reality_check_pvalue"
                    )
                ),
                "hansen_spa_pvalue": multiple_testing.get(
                    "hansen_spa_pvalue"
                ),
            },
            blocks_live=not financial_candidate,
        ),
        _requirement(
            "FIVE_EURO_SPOT_CANARY",
            status=(
                "REGISTERED_DISABLED"
                if canary_registered
                else "MISSING_OR_UNSAFE"
            ),
            evidence={
                "path": str(canary_path),
                "policy_hash": canary.get("policy_hash"),
                "maximum_total_eur": canary.get(
                    "maximum_total_eur"
                ),
                "spot_only": canary.get("spot_only"),
                "autoscale": canary.get("autoscale"),
            },
            blocks_live=True,
        ),
        _requirement(
            "AI_ML_EMBARGO",
            status=str(
                ai.get(
                    "status",
                    "AI_DEVELOPMENT_EMBARGOED",
                )
            ),
            evidence={
                "path": str(ai_path),
                "training_permitted": ai.get(
                    "training_permitted",
                    False,
                ),
            },
            blocks_live=False,
        ),
    ]
    blocked = [
        row["requirement_id"]
        for row in requirements
        if row["blocks_live"] and row["status"] != "PASSED"
    ]
    body = {
        "schema_version": "prospective_research_readiness_v1",
        "generated_at": utc_now().isoformat(),
        "overall_status": (
            "OPERATIONALLY_LIVE_RESEARCH_FINANCIALLY_BLOCKED"
        ),
        "positive_verified_outcome": (
            "RESILIENT_REAL_TIME_PUBLIC_DATA_COLLECTION"
        ),
        "profitable_strategy_proven": financial_candidate,
        "orderflow_backtest_permitted": orderflow_ready,
        "paper_permitted": False,
        "live_permitted": False,
        "live_blockers": blocked,
        "requirements": requirements,
        "orders_generated": 0,
        "private_exchange_requests": 0,
    }
    return {**body, "report_hash": stable_hash(body, length=64)}


def write_prospective_readiness_report(
    settings: Any,
) -> tuple[Path, dict[str, Any]]:
    report = prospective_readiness_report(settings)
    path = (
        settings.paths.lab_dir
        / "reports"
        / "prospective_research_readiness_v1.json"
    )
    atomic_write_json(path, report)
    return path, report


__all__ = [
    "prospective_readiness_report",
    "write_prospective_readiness_report",
]
