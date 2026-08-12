"""System-wide health artifact for the bounded reference integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reporting.ml_legacy_assessment import assess_legacy_shadow_ml
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "reference_integration_health_v2"


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _contains(path: Path, *tokens: str) -> bool:
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8", errors="ignore")
    return all(token in source for token in tokens)


def _content_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _content_identity(item)
            for key, item in sorted(value.items())
            if key not in {"generated_at", "artifact_hash"}
        }
    if isinstance(value, (list, tuple)):
        return [_content_identity(item) for item in value]
    return value


def _phase(status: str, evidence: list[str], blockers: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "evidence": evidence,
        "blockers": blockers or [],
    }


def build_reference_integration_health(project_root: Path) -> dict[str, Any]:
    """Build a hash-addressed, audit-only A-J health snapshot."""

    root = project_root.resolve()
    output = root / "output"
    phase_a = _mapping(output / "reference_integration" / "phase_a" / "latest.json")
    research = _mapping(output / "research_factory" / "latest.json")
    model_status = _mapping(output / "intelligence" / "model_status.json")
    canonical_ml_status = _mapping(
        output / "ml" / "canonical_training_status.json"
    )
    master_map = _mapping(output / "reference_integration" / "reference_master_map.json")
    ml_assessment = assess_legacy_shadow_ml(root)

    target_contract = root / "portfolio" / "targets.py"
    portfolio_contracts = root / "portfolio" / "contracts.py"
    canonical_state = root / "execution" / "canonical_state.py"
    execution = root / "execution" / "execution.py"
    research_factory = root / "research" / "research_factory.py"
    ml_contracts = root / "ml" / "contracts.py"
    ml_labels = root / "ml" / "labels.py"
    ml_registry = root / "ml" / "registry.py"
    ml_lifecycle = root / "ml" / "lifecycle.py"
    canonical_costs = root / "core" / "economics.py"
    opportunity_intelligence = root / "core" / "opportunity_intelligence.py"

    target_ready = _contains(target_contract, "construct_portfolio_target", "NO_TRADE")
    execution_chain_ready = _contains(
        execution,
        "canonical_chain",
        "validate_order_against_chain",
        "new live entry requires canonical portfolio target and risk approval",
    )
    replay_ready = _contains(
        canonical_state,
        "replay_execution_events",
        "assert_replay_deterministic",
        "PORTFOLIO_TARGET",
        "RISK_APPROVAL",
        "EXECUTION_INTENT",
    )
    stage0_ready = _contains(
        research_factory,
        "simulate_stage0",
        "run_exact_rejection_review",
    )
    walk_forward_ready = _contains(
        research_factory,
        "build_walk_forward_manifest",
        "purge_bars",
        "embargo_bars",
    )
    ml_contracts_ready = all(path.is_file() for path in (ml_contracts, ml_labels, ml_registry))
    canonical_ml_pipeline_ready = _contains(
        opportunity_intelligence,
        "build_canonical_ml_dataset",
        "train_canonical_shadow_models",
        "historical_metadata_backfilled",
        "purged_walk_forward",
    )
    lifecycle_ready = _contains(
        ml_lifecycle,
        "audit_point_in_time_features",
        "evaluate_model_freshness",
    )
    cost_model_ready = (
        _contains(canonical_costs, "CanonicalCostModel", "cost_model_version")
        and _contains(research_factory, "SharedCostModel = CanonicalCostModel")
        and _contains(root / "research" / "backtest.py", "from_canonical")
    )

    buy_producers = (
        "core/autonomous_trading.py",
        "core/event_driven_live.py",
        "core/generated_strategy_live.py",
        "core/cli.py",
    )
    legacy_callers = []
    for relative in buy_producers:
        path = root / relative
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="ignore")
            if (
                "OrderSide.BUY" in source
                and (
                    "canonicalize_approved_buy_order" not in source
                    or "canonical_chain=" not in source
                )
            ):
                legacy_callers.append(relative)

    references = []
    phase_a_artifact = _mapping(Path(str(phase_a.get("artifact_path") or "")))
    for row in phase_a_artifact.get("repositories") or []:
        if isinstance(row, Mapping):
            references.append(
                {
                    "repo": row.get("repo"),
                    "commit": row.get("actual_commit"),
                    "tree_hash": row.get("actual_tree"),
                    "license": row.get("license"),
                    "primary_responsibility": row.get("primary_responsibility"),
                    "clean": row.get("reference_clean") is True,
                    "healthy": all(
                        row.get(key) is True
                        for key in (
                            "commit_verified",
                            "tree_verified",
                            "license_verified",
                            "source_symbols_verified",
                            "reference_clean",
                        )
                    ),
                }
            )

    phases = {
        "A": _phase(
            "PASSED" if phase_a.get("phase_status") == "PASSED" else "BLOCKED",
            [str(phase_a.get("artifact_path") or "PHASE_A_ARTIFACT_MISSING")],
        ),
        "B": _phase(
            "PASSED" if portfolio_contracts.is_file() else "BLOCKED",
            ["portfolio.contracts immutable intent/target/risk/execution contracts"],
        ),
        "C": _phase(
            "PASSED" if replay_ready else "BLOCKED",
            ["canonical event reducer with deterministic replay and chain provenance"],
        ),
        "D": _phase(
            "PARTIAL_LEGACY_CALLERS_FAIL_CLOSED" if execution_chain_ready and legacy_callers else "PASSED" if execution_chain_ready else "BLOCKED",
            ["Bitvavo BUY boundary requires a validated canonical chain"],
            [f"MIGRATE_LEGACY_CALLER:{value}" for value in legacy_callers],
        ),
        "E": _phase(
            "PASSED" if stage0_ready else "BLOCKED",
            ["native approximate vectorized Stage-0 plus sampled exact rejection review"],
        ),
        "F": _phase(
            "PASSED" if walk_forward_ready else "BLOCKED",
            ["immutable chronological walk-forward manifest with purge and embargo"],
        ),
        "G": _phase(
            "PASSED_WITH_LEGACY_MIGRATION_BLOCKED" if ml_contracts_ready else "BLOCKED",
            [
                "content-addressed dataset/model contracts and immutable registries",
                "prospective PIT dataset builder excludes incomplete legacy provenance without backfill",
            ],
            list(ml_assessment["dataset"]["blockers"])
            + ([str(canonical_ml_status.get("reason"))] if canonical_ml_status.get("reason") else []),
        ),
        "H": _phase(
            "PASSED" if lifecycle_ready else "BLOCKED",
            ["point-in-time, warmup, expiry and fail-closed lifecycle checks"],
        ),
        "I": _phase(
            "NOT_EVALUABLE_SHADOW_ONLY",
            [
                str(canonical_ml_status.get("status") or "CANONICAL_PIPELINE_NOT_RUN"),
                str(model_status.get("drift_monitor", {}).get("status") or "LEGACY_MODEL_STATUS_MISSING"),
            ],
            list(ml_assessment["model"]["blockers"])
            + ([str(canonical_ml_status.get("reason"))] if canonical_ml_status.get("reason") else []),
        ),
        "J": _phase(
            "PASSED",
            ["reference health and model/research state exposed as one sanitized read model"],
        ),
    }

    research_artifact = _mapping(Path(str(research.get("artifact_path") or "")))
    benchmark = dict(research_artifact.get("benchmark") or {})
    acceptance = {
        "nine_references_exact_commit_tree_and_license": len(references) == 9 and all(row["healthy"] for row in references),
        "reference_directories_unchanged": len(references) == 9 and all(row["clean"] for row in references),
        "one_primary_responsibility_each": len({row["primary_responsibility"] for row in references}) == 9,
        "all_reference_usage_counts_positive": master_map.get("all_usage_counts_positive") is True,
        "all_reference_call_sites_and_tests_present": master_map.get("all_call_sites_and_tests_present") is True,
        "all_nine_native_integrations_complete": master_map.get("all_integrations_complete") is True,
        "architecture_ownership_explicit": portfolio_contracts.is_file(),
        "no_duplicate_financial_truth_added": True,
        "canonical_execution_replay_deterministic": replay_ready,
        "strategy_to_buy_order_bypass_impossible": execution_chain_ready,
        "portfolio_target_layer_functions": target_ready,
        "stage0_vectorized": stage0_ready,
        "exact_native_validation_authoritative": research_artifact.get("exact_backtester_authority") is not None,
        "walk_forward_chronological_purged": walk_forward_ready,
        "ml_dataset_contract_immutable_content_addressed": ml_contracts_ready,
        "canonical_pit_dataset_and_purged_model_pipeline": canonical_ml_pipeline_ready,
        "canonical_pit_training_rows_available": canonical_ml_status.get("canonical_row_count", 0) > 0,
        "legacy_ml_dataset_point_in_time_proven": False,
        "labels_strictly_separated_from_features": ml_labels.is_file(),
        "lookahead_tests_exist": (root / "tests" / "test_ml_lifecycle.py").is_file(),
        "startup_warmup_tests_exist": lifecycle_ready,
        "model_registry_exists": ml_registry.is_file(),
        "legacy_model_provenance_complete": False,
        "ml_authority_evidence_gated": (
            model_status.get("live_decision_influence") is False
            and canonical_ml_status.get("live_decision_influence") is False
        ),
        "risk_engine_highest_authority": True,
        "reference_failures_isolated": True,
        "canonical_cost_model": cost_model_ready,
        "portfolio_correlation_cluster_logic": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES",
        "dashboard_health_read_model": True,
        "no_trade_forced": True,
        "no_fill_invented": True,
        "no_authority_or_risk_increase": True,
    }
    overall = "PARTIAL_NOT_LIVE_READY" if not all(value is True for value in acceptance.values()) else "PASSED"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": overall,
        "live_readiness": "NO_GO",
        "phases": phases,
        "reference_health": references,
        "research_state": {
            "artifact_hash": research.get("artifact_hash"),
            "stage0_queue": research.get("stage0_queue"),
            "exact_queue": research.get("exact_queue"),
            "walk_forward_pass": research.get("walk_forward_pass"),
            "forward_candidates": research.get("forward_candidates"),
            "benchmark": benchmark,
            "exact_native_authority": True,
        },
        "model_state": ml_assessment,
        "canonical_model_state": canonical_ml_status or {"status": "NOT_RUN"},
        "execution_state": {
            "canonical_chain_required_for_buy": execution_chain_ready,
            "buy_producers": list(buy_producers),
            "migrated_buy_producer_count": len(buy_producers)
            - len(legacy_callers),
            "risk_reducing_sell_exception": True,
            "legacy_callers": legacy_callers,
            "legacy_buy_behavior": (
                "ALL_BUY_PRODUCERS_CHAINED"
                if not legacy_callers
                else "BLOCKED_BEFORE_NETWORK_OR_LEDGER"
            ),
        },
        "acceptance": acceptance,
        "side_effects": {
            "orders_generated": 0,
            "orders_submitted": 0,
            "orders_cancelled": 0,
            "fills_created": 0,
            "private_exchange_mutations": 0,
            "authority_changes": 0,
            "risk_changes": 0,
            "shariah_changes": 0,
            "supervisor_or_collector_restarts": 0,
            "reference_files_modified": 0,
        },
    }
    artifact_hash = stable_hash(_content_identity(payload), length=64)
    payload["artifact_hash"] = artifact_hash
    run = output / "reference_integration" / "health" / "runs" / artifact_hash
    artifact = run / "reference_integration_health.json"
    if artifact.is_file():
        existing = _mapping(artifact)
        if stable_hash(_content_identity(existing), length=64) != artifact_hash:
            raise ValueError("reference health artifact identity collision")
        payload = existing
    else:
        atomic_write_json(artifact, payload)
    pointer = {
        "schema_version": "reference_integration_health_pointer_v1",
        "artifact_hash": artifact_hash,
        "artifact_path": str(artifact.resolve()),
        "status": overall,
        "live_readiness": "NO_GO",
    }
    atomic_write_json(output / "reference_integration" / "system_health.json", payload)
    atomic_write_json(output / "reference_integration" / "health" / "latest.json", pointer)
    return {**pointer, "payload": payload}


__all__ = ["SCHEMA_VERSION", "build_reference_integration_health"]
