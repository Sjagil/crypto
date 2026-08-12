from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import research.p1_3_governance as governance
from data.multi_source_maturation import FamilyFreezeManager
from research.p1_3_governance import (
    AppendOnlyResearchLedger,
    HoldoutAccessError,
    ImmutableArtifactError,
    P13ResearchRunner,
    PartitionAccessGuard,
    PreregistrationBindings,
    PreregistrationStore,
    ResearchGateError,
    build_empty_results_template,
    build_preregistration_plan,
    build_variant_catalog,
    resolve_dataset_freeze,
)
from utils.common import read_json, stable_hash


def _bindings() -> PreregistrationBindings:
    return PreregistrationBindings(
        git_commit="unit-commit",
        git_worktree_state="DIRTY_SOURCE_HASHES_BOUND",
        p1_2_3_artifact_hash="a" * 64,
        p1_2_3_artifact_file_sha256="b" * 64,
        p1_2_3_artifact_path="prior-p1-2-3.json",
        p1_1_evidence_hash="c" * 64,
        p1_1_evidence_file_sha256="d" * 64,
        p1_1_evidence_path="prior-p1-1.json",
        readiness_policy_version="research_readiness_policy_v1",
        shared_cost_model={
            "cost_model_version": "shared_cost_v1:unit",
            "maker_fee_fraction": 0.0015,
            "taker_fee_fraction": 0.0025,
            "spread_bps": 5.0,
            "slippage_bps": 8.0,
            "failed_execution_allowance_bps": 0.0,
            "partial_fill_impact_bps": 0.0,
        },
        feature_schema_versions={
            "source_observation": "source_neutral_observation_v1",
            "bitvavo_l2": "bitvavo_l2_features_v2",
        },
        source_code_hashes={"native_backtester": "e" * 64},
    )


def _freeze(workspace: Path) -> dict:
    assessment = {
        "schema_version": "family_readiness_assessment_v1",
        "family": "FLOW_CONFIRMED_SWING",
        "state": "RESEARCH_USABLE",
        "policy_version": "research_readiness_policy_v1",
        "metrics": {"gap_fraction": 0.0},
    }
    result = FamilyFreezeManager(
        workspace / "output" / "multi_source" / "freezes"
    ).maybe_freeze(
        assessment=assessment,
        transition={"transition_id": "unit-transition"},
        source_manifests=[{"source": "bitvavo", "root_hash": "f" * 64}],
        assets=("CRYPTO:BTC", "CRYPTO:ETH", "CRYPTO:SOL"),
        features=("CVD", "TRADE_INTENSITY"),
        collection_start=datetime(2026, 1, 1, tzinfo=UTC),
        data_end=datetime(2026, 5, 1, tzinfo=UTC),
        coverage={"gap_fraction": 0.0},
        clock_metrics={"status": "PASS"},
        build_commit="unit-commit",
    )
    assert result["status"] == "FREEZE_CREATED"
    return result["freeze"]


def _runner(tmp_path: Path, monkeypatch) -> tuple[P13ResearchRunner, dict, dict]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    governance_root = tmp_path / "governance"
    preregistration = PreregistrationStore(governance_root).create(
        build_preregistration_plan(_bindings()),
        created_at="2026-08-11T00:00:00Z",
    )
    freeze = _freeze(workspace)
    monkeypatch.setattr(governance, "_git", lambda *_args: "unit-commit")
    return P13ResearchRunner(workspace, governance_root), preregistration, freeze


def test_preregistration_generation_is_pure_and_does_not_access_targets(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("prospective target/data access attempted")

    monkeypatch.setattr("pandas.read_parquet", forbidden)
    monkeypatch.setattr("pandas.read_csv", forbidden)
    monkeypatch.setattr("pandas.read_json", forbidden)
    plan = build_preregistration_plan(_bindings())
    assert plan["research_state"] == "DESIGN_FROZEN_RESEARCH_NOT_STARTED"
    assert plan["reporting_template"]["performance_fields_populated"] is False
    assert plan["authority_and_policy"]["orders_generated"] == 0
    source = inspect.getsource(governance)
    assert "from data.data_loader" not in source
    assert "from execution" not in source
    assert "import execution" not in source


def test_baseline_and_hypothesis_ladder_are_frozen_from_prior_evidence() -> None:
    plan = build_preregistration_plan(_bindings())
    baseline = plan["baseline"]
    assert baseline["family"] == "MEDIUM_TERM_TREND_PULLBACK"
    assert baseline["prior_status"] == "EXACT_REJECTED_NOT_PROMOTED"
    assert baseline["signal_logic"]["trend_days"] == 20
    assert baseline["signal_logic"]["continuation_days"] == 3
    assert baseline["signal_logic"]["pullback_atr_minimum"] == 0.5
    assert baseline["signal_logic"]["exit_days"] == 5
    assert baseline["stop_atr"] == 3.0
    assert baseline["target_atr"] == 6.0
    assert baseline["maximum_holding_bars"] == 84
    assert plan["ablation_ladder"]["mandatory_comparisons"] == [
        "A_VS_B",
        "B_VS_C",
        "C_VS_D",
    ]
    assert len(plan["hypotheses"]) == 6


def test_all_bounded_variants_are_counted_exactly_once() -> None:
    variants = build_variant_catalog()
    ids = [row["variant_id"] for row in variants]
    assert len(variants) == 131
    assert len(ids) == len(set(ids))
    plan = build_preregistration_plan(_bindings())
    testing = plan["multiple_testing"]
    assert testing["HYPOTHESIS_COUNT"] == 6
    assert testing["VARIANT_COUNT"] == 131
    assert testing["ladder_variant_counts"] == {"A": 1, "B": 6, "C": 30, "D": 90, "E": 4}
    assert testing["variant_catalog_hash"] == stable_hash(variants)


def test_identical_inputs_have_same_content_hash_and_creation_is_idempotent(tmp_path) -> None:
    first_plan = build_preregistration_plan(_bindings())
    second_plan = build_preregistration_plan(_bindings())
    assert stable_hash(first_plan) == stable_hash(second_plan)
    store = PreregistrationStore(tmp_path)
    first = store.create(first_plan, created_at="2026-08-11T00:00:00Z")
    second = store.create(second_plan, created_at="2099-01-01T00:00:00Z")
    assert first == second
    assert first["creation_timestamp"] == "2026-08-11T00:00:00Z"
    assert first["content_hash"] == second["content_hash"]
    assert first["preregistration_id"] == second["preregistration_id"]


def test_frozen_preregistration_rejects_modification_and_detects_manual_tamper(tmp_path) -> None:
    store = PreregistrationStore(tmp_path)
    plan = build_preregistration_plan(_bindings())
    frozen = store.create(plan, created_at="2026-08-11T00:00:00Z")
    replacement = dict(plan)
    replacement["primary_question"] = "changed"
    with pytest.raises(ImmutableArtifactError, match="cannot be modified"):
        store.reject_modification(frozen["preregistration_id"], replacement)
    path = store.path_for(frozen["preregistration_id"])
    payload = read_json(path)
    payload["plan"]["baseline"]["stop_atr"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImmutableArtifactError, match="content hash mismatch"):
        store.verify(frozen["preregistration_id"])


def test_empty_result_template_has_all_sections_and_no_performance() -> None:
    template = build_empty_results_template()
    assert set(template["sections"]) == {
        "baseline",
        "flow",
        "cross_venue",
        "l2",
        "costs",
        "ablation",
        "walk_forward",
        "holdout",
        "robustness",
        "failures",
    }
    assert template["performance_fields_populated"] is False
    assert all(row["metrics"] is None for row in template["sections"].values())


def test_runner_requires_both_immutable_ids_and_retains_failed_runs(
    tmp_path, monkeypatch
) -> None:
    runner, preregistration, freeze = _runner(tmp_path, monkeypatch)
    with pytest.raises(ResearchGateError, match="PREREGISTRATION_ID"):
        runner.authorize(preregistration_id=None, dataset_freeze_id=freeze["dataset_id"])
    with pytest.raises(ResearchGateError, match="DATASET_FREEZE_ID"):
        runner.authorize(
            preregistration_id=preregistration["preregistration_id"],
            dataset_freeze_id=None,
        )
    records = runner.ledger.records()
    assert len(records) == 2
    assert all(row["payload"]["status"] == "FAILED" for row in records)
    assert runner.ledger.summary()["failed_runs"] == 2


@pytest.mark.parametrize("alias", ["latest", "current", "status", "a" * 63])
def test_mutable_or_non_exact_dataset_alias_is_rejected(tmp_path, alias) -> None:
    with pytest.raises(ResearchGateError, match="64-hex"):
        resolve_dataset_freeze(tmp_path, alias)


def test_dataset_manifest_must_be_immutable_research_usable_and_hash_valid(tmp_path) -> None:
    freeze = _freeze(tmp_path)
    root = tmp_path / "output" / "multi_source" / "freezes"
    resolved = resolve_dataset_freeze(root, freeze["dataset_id"])
    assert resolved["holdout_status"] == "RESERVED_UNTOUCHED"
    path = Path(resolved["manifest_path"])
    payload = read_json(path)
    payload["immutable"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchGateError, match="mutable dataset"):
        resolve_dataset_freeze(root, freeze["dataset_id"])


def test_development_cannot_request_holdout_and_final_requires_candidate_hash(tmp_path) -> None:
    freeze = _freeze(tmp_path)
    development = PartitionAccessGuard.authorize(
        freeze,
        start=freeze["development_start"],
        end=freeze["development_end"],
        phase="DEVELOPMENT",
    )
    assert development["partition"] == "DEVELOPMENT_DATA"
    assert development["rows_loaded"] == 0
    with pytest.raises(HoldoutAccessError, match="cannot request final holdout"):
        PartitionAccessGuard.authorize(
            freeze,
            start=freeze["development_start"],
            end=freeze["holdout_end"],
            phase="DEVELOPMENT",
        )
    with pytest.raises(HoldoutAccessError, match="candidate hash"):
        PartitionAccessGuard.authorize(
            freeze,
            start=freeze["holdout_start"],
            end=freeze["holdout_end"],
            phase="FINAL_HOLDOUT",
        )


def test_guarded_authorization_is_reproducible_and_has_no_authority(
    tmp_path, monkeypatch
) -> None:
    runner, preregistration, freeze = _runner(tmp_path, monkeypatch)
    identity = {
        "preregistration_id": preregistration["preregistration_id"],
        "dataset_freeze_id": freeze["dataset_id"],
        "seed": 1729,
    }
    first = runner.authorize(**identity)
    second = runner.authorize(**identity)
    assert first["result_hash"] == second["result_hash"]
    assert first["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert first["research_executed"] is False
    assert first["performance_calculated"] is False
    assert first["orders_generated"] == 0
    assert first["live_authority_changed"] is False
    assert first["kraken_execution"] is False
    assert first["mexc_execution"] is False


def test_one_shot_holdout_attempt_is_enforced(tmp_path, monkeypatch) -> None:
    runner, preregistration, freeze = _runner(tmp_path, monkeypatch)
    identity = {
        "preregistration_id": preregistration["preregistration_id"],
        "dataset_freeze_id": freeze["dataset_id"],
        "seed": 1729,
        "phase": "FINAL_HOLDOUT",
        "candidate_hash": "frozen-candidate-1234567890",
    }
    assert runner.authorize(**identity)["status"] == "AUTHORIZED_NOT_EXECUTED"
    with pytest.raises(ResearchGateError, match="one-shot holdout"):
        runner.authorize(**identity)
    assert runner.ledger.records()[-1]["payload"]["status"] == "FAILED"


def test_append_only_ledger_detects_revision_and_keeps_failures(tmp_path) -> None:
    ledger = AppendOnlyResearchLedger(tmp_path / "runs.jsonl")
    ledger.append({"status": "FAILED", "reason": "unit"})
    ledger.append({"status": "AUTHORIZED_NOT_EXECUTED"})
    assert ledger.summary()["record_count"] == 2
    assert ledger.summary()["failed_runs"] == 1
    rows = ledger.path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["payload"]["reason"] = "rewritten"
    rows[0] = json.dumps(first)
    ledger.path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ImmutableArtifactError, match="record hash mismatch"):
        ledger.records()


def test_authority_shariah_and_reference_venue_invariants_are_explicit() -> None:
    plan = build_preregistration_plan(_bindings())
    authority = plan["authority_and_policy"]
    assert authority["spot_long_only"] is True
    assert authority["kraken_execution"] is False
    assert authority["mexc_execution"] is False
    assert authority["ml_authority"] == "SHADOW_ONLY_NO_TRAINING_IN_PREREGISTRATION"
    assert authority["live_promotion_in_scope"] is False
    assert "leverage" in authority["shariah_prohibited"]
    assert "interest" in authority["shariah_prohibited"]


def test_plan_change_requires_new_generation(tmp_path) -> None:
    store = PreregistrationStore(tmp_path)
    plan = build_preregistration_plan(_bindings())
    first = store.create(plan, created_at="2026-08-11T00:00:00Z")
    changed_bindings = replace(_bindings(), git_commit="new-commit")
    second = store.create(
        build_preregistration_plan(changed_bindings),
        created_at="2026-08-12T00:00:00Z",
    )
    assert first["preregistration_id"] != second["preregistration_id"]
    assert store.path_for(first["preregistration_id"]).is_file()
    assert store.path_for(second["preregistration_id"]).is_file()


def test_holdout_boundaries_are_newest_twenty_percent_with_forward_partition(tmp_path) -> None:
    freeze = _freeze(tmp_path)
    start = datetime.fromisoformat(freeze["development_start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(freeze["holdout_end"].replace("Z", "+00:00"))
    holdout = datetime.fromisoformat(freeze["holdout_start"].replace("Z", "+00:00"))
    assert holdout - start == timedelta(days=96)
    assert end - holdout == timedelta(days=24)
    assert freeze["future_data_default_partition"] == "POST_FREEZE_FORWARD_DATA"

