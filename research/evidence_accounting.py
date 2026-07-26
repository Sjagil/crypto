"""Separate historical selection burden from prospective observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.global_trial_accounting import (
    audit_global_trial_accounting,
)
from utils.common import atomic_write_json, read_json, stable_hash

FORWARD_EVIDENCE_ACCOUNTING_SCHEMA = "forward_evidence_accounting_v1"


def _list_value(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


def audit_forward_evidence_accounting(
    lab_dir: Path | str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Audit observers without adding observations to the trial denominator."""

    root = Path(lab_dir)
    observer_root = root / "observers"
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    if observer_root.is_dir():
        for path in sorted(observer_root.rglob("*.json")):
            payload = dict(read_json(path))
            observations = _list_value(
                payload,
                "forward_observations",
            )
            decisions = _list_value(payload, "forward_decisions")
            reselections = _list_value(
                payload,
                "forward_reselection_events",
            )
            parameters_frozen = bool(
                payload.get("parameters_frozen")
                or payload.get("portfolio_policy_hash")
                or payload.get("strategy_dna_hash")
            )
            if reselections:
                failures.append(
                    f"FORWARD_RESELECTION_RECORDED:{path}"
                )
            if not parameters_frozen:
                failures.append(
                    f"FORWARD_FREEZE_EVIDENCE_MISSING:{path}"
                )
            rows.append(
                {
                    "path": str(path),
                    "family": str(
                        payload.get("family")
                        or payload.get("campaign")
                        or path.parent.name
                    ),
                    "strategy_dna_hash": payload.get(
                        "strategy_dna_hash"
                    ),
                    "status": payload.get("status"),
                    "parameters_frozen": parameters_frozen,
                    "forward_observation_count": len(observations),
                    "forward_decision_count": len(decisions),
                    "forward_reselection_event_count": len(
                        reselections
                    ),
                    "orders_generated": int(
                        payload.get("orders_generated") or 0
                    ),
                }
            )
    trial_audit = audit_global_trial_accounting(
        root,
        persist=False,
    )
    payload: dict[str, Any] = {
        "schema_version": FORWARD_EVIDENCE_ACCOUNTING_SCHEMA,
        "status": "PASSED" if not failures else "FAILED",
        "definitions": {
            "historical_evaluation_trials": (
                "Every strategy DNA evaluation on a closed-data "
                "fingerprint that participated in research."
            ),
            "historical_selection_events": (
                "A ranking, threshold choice, or winner selection. "
                "Legacy pre-registry events are not falsely inferred."
            ),
            "forward_observations": (
                "Prospective evidence appended after freeze without "
                "ranking or changing DNA."
            ),
            "forward_reselection_events": (
                "A new ranking or winner choice using expanded data; "
                "only these increase historical selection burden."
            ),
        },
        "historical_evaluation_trials": int(
            trial_audit["evaluation_trial_count"]
        ),
        "historical_selection_events": {
            "known_count": None,
            "coverage": "PARTIAL_PRE_REGISTRY_HISTORY",
            "reason": (
                "Legacy aggregate reports do not preserve every "
                "selection event; no unsupported exact count is claimed."
            ),
        },
        "unique_strategy_dna": int(
            trial_audit["unique_strategy_dna_equivalent_count"]
        ),
        "forward_observer_count": len(rows),
        "forward_observation_count": sum(
            row["forward_observation_count"] for row in rows
        ),
        "forward_decision_count": sum(
            row["forward_decision_count"] for row in rows
        ),
        "forward_reselection_event_count": sum(
            row["forward_reselection_event_count"] for row in rows
        ),
        "forward_observations_counted_in_multiple_testing": False,
        "global_multiple_testing_denominator": int(
            trial_audit["global_multiple_testing_denominator"]
        ),
        "observer_rows": rows,
        "integrity_failures": failures,
        "orders_generated": sum(
            row["orders_generated"] for row in rows
        ),
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    payload["evidence_root_hash"] = stable_hash(payload, length=64)
    if persist:
        report = (
            root
            / "reports"
            / "forward_evidence_accounting_v1.json"
        )
        atomic_write_json(report, payload)
        return {**payload, "report": str(report)}
    return payload


__all__ = [
    "FORWARD_EVIDENCE_ACCOUNTING_SCHEMA",
    "audit_forward_evidence_accounting",
]
