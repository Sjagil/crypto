from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import core.cli as cli
from config.settings import get_settings
from core.autopilot import (
    AutopilotLockError,
    AutopilotOrchestrator,
    AutopilotPolicy,
    DegradationObservation,
    assert_orderless_research_payload,
    performance_degradation_z_score,
)


def _data_stage(fingerprint: str = "data-v1"):
    return {
        "status": "DATA_AUDITED",
        "data_fingerprint": fingerprint,
        "orders_generated": 0,
    }


def _observer_stage():
    return {
        "status": "FROZEN_FORWARD_RESEARCH",
        "observer_count": 8,
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def test_storm_epoch_accounting_counts_new_closed_data_epochs():
    epochs = [
        {
            "epoch_id": "CANONICAL_INITIAL_STORM",
            "source": "CANONICAL_PORTFOLIO_STORM_V1",
            "data_fingerprint": "data-a",
            "strategy_search_space_hash": "space-a",
            "evaluated_strategy_count": 5_000,
            "total_known_trials": 6_312,
        },
        {
            "epoch_id": "data-epoch-two",
            "source": "AUTOPILOT_NEW_DATA_EPOCH",
            "data_fingerprint": "data-b",
            "strategy_search_space_hash": "space-a",
            "evaluated_strategy_count": 5_000,
            "total_known_trials": 11_312,
        },
        {
            "epoch_id": "new-search-space",
            "source": "AUTOPILOT_NEW_DATA_EPOCH",
            "data_fingerprint": "data-c",
            "strategy_search_space_hash": "space-b",
            "evaluated_strategy_count": 5_000,
            "total_known_trials": 16_312,
        },
    ]

    reconciled, total = cli._reconcile_storm_epoch_accounting(
        epochs,
        default_prior_known_trials=1_312,
    )

    assert total == 16_312
    assert [row["new_strategy_dna_count"] for row in reconciled] == [
        5_000,
        0,
        5_000,
    ]
    assert [
        row["new_evaluation_trial_count"]
        for row in reconciled
    ] == [0, 5_000, 5_000]
    assert [row["total_known_trials"] for row in reconciled] == [
        6_312,
        11_312,
        16_312,
    ]
    assert reconciled[1]["report_total_known_trials_at_birth"] == 11_312
    assert all(
        row["evaluation_epoch_count"] == 1 for row in reconciled
    )


def test_performance_degradation_z_score_is_finite_and_fail_closed():
    assert performance_degradation_z_score(
        live_return=-0.03,
        cv_mean=0.01,
        cv_std=0.02,
    ) == pytest.approx(-2.0)
    assert (
        performance_degradation_z_score(
            live_return=0.01,
            cv_mean=0.01,
            cv_std=0.0,
        )
        is None
    )


def test_autopilot_cycle_is_orderless_and_research_is_scheduled(tmp_path):
    current = datetime(2026, 7, 25, tzinfo=UTC)
    calls = {"feature_store": 0, "research": 0}

    def clock():
        return current

    def research():
        calls["research"] += 1
        return {
            "status": "COMPLETED_NOT_PROMOTED",
            "orders_generated": 0,
            "paper_candidates": 0,
            "live_orders": 0,
        }

    def feature_store():
        calls["feature_store"] += 1
        return {
            "status": "REUSED",
            "dataset_id": "causal-tensor-v1",
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    orchestrator = AutopilotOrchestrator(tmp_path, clock=clock)
    first = orchestrator.run_once(
        data_stage=_data_stage,
        feature_store_stage=feature_store,
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert first["status"] == "COMPLETED_ORDERLESS"
    assert first["research_ran"] is True
    assert calls["research"] == 1
    assert calls["feature_store"] == 1
    assert first["orders_generated"] == 0
    assert not orchestrator.lock_path.exists()

    current += timedelta(days=1)
    second = orchestrator.run_once(
        data_stage=_data_stage,
        feature_store_stage=feature_store,
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert second["status"] == "COMPLETED_ORDERLESS"
    assert second["research_ran"] is False
    assert second["research_reason"] == "DATA_UNCHANGED"
    assert calls["research"] == 1
    assert calls["feature_store"] == 2
    assert orchestrator.state()["cycle_count"] == 2
    assert orchestrator.state()["last_feature_store_dataset_id"] == "causal-tensor-v1"


def test_new_data_waits_for_research_interval_and_then_runs(tmp_path):
    current = datetime(2026, 7, 25, tzinfo=UTC)
    calls = {"research": 0}

    def clock():
        return current

    def research():
        calls["research"] += 1
        return {"status": "DONE", "orders_generated": 0}

    policy = AutopilotPolicy(research_interval_seconds=7 * 86_400)
    orchestrator = AutopilotOrchestrator(tmp_path, policy=policy, clock=clock)
    orchestrator.run_once(
        data_stage=lambda: _data_stage("v1"),
        research_stage=research,
        observer_stage=_observer_stage,
    )
    current += timedelta(days=2)
    waiting = orchestrator.run_once(
        data_stage=lambda: _data_stage("v2"),
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert waiting["research_reason"] == "RESEARCH_INTERVAL_NOT_ELAPSED"
    current += timedelta(days=6)
    due = orchestrator.run_once(
        data_stage=lambda: _data_stage("v2"),
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert due["research_ran"] is True
    assert due["research_reason"] == "NEW_DATA_AND_INTERVAL_ELAPSED"
    assert calls["research"] == 2


def test_early_degradation_is_diagnostic_only(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)
    diagnostic = orchestrator.run_once(
        data_stage=_data_stage,
        observer_stage=_observer_stage,
        degradation_observation=DegradationObservation(
            live_return=-0.06,
            cv_mean=0.01,
            cv_std=0.02,
            observation_count=30,
        ),
    )
    assert diagnostic["status"] == "COMPLETED_ORDERLESS"
    assert diagnostic["degradation"]["status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["degradation"]["formal_action_permitted"] is False
    assert orchestrator.kill_switch()["system_degraded"] is False


def test_formal_degradation_breach_persists_and_blocks_later_cycles(
    tmp_path,
):
    orchestrator = AutopilotOrchestrator(tmp_path)
    degraded = orchestrator.run_once(
        data_stage=_data_stage,
        observer_stage=_observer_stage,
        degradation_observation=DegradationObservation(
            live_return=-0.06,
            cv_mean=0.01,
            cv_std=0.02,
            observation_count=365,
        ),
    )
    assert degraded["status"] == "SYSTEM_DEGRADED"
    assert orchestrator.kill_switch()["system_degraded"] is True

    blocked = orchestrator.run_once(
        data_stage=_data_stage,
        observer_stage=_observer_stage,
    )
    assert blocked["status"] == "SYSTEM_DEGRADED"
    assert blocked["reason"] == "PERSISTENT_KILL_SWITCH_ACTIVE"

    reset = orchestrator.reset_kill_switch(
        reason="health checks and evidence reviewed",
        confirmed=True,
    )
    assert reset["system_degraded"] is False
    assert reset["events"][-1]["event"] == "MANUAL_RESET"


def test_stage_failure_activates_fail_closed_state(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)

    def broken():
        raise ValueError("corrupt input")

    result = orchestrator.run_once(
        data_stage=broken,
        observer_stage=_observer_stage,
    )
    assert result["status"] == "SYSTEM_DEGRADED"
    assert result["error"]["type"] == "ValueError"
    assert orchestrator.status()["orders_generated"] == 0
    assert orchestrator.kill_switch()["manual_reset_required"] is True


def test_preflight_runs_before_data_and_failure_activates_kill_switch(
    tmp_path,
) -> None:
    events: list[str] = []
    orchestrator = AutopilotOrchestrator(tmp_path)
    successful = orchestrator.run_once(
        preflight_stage=lambda: (
            events.append("preflight")
            or {"status": "PASSED", "orders_generated": 0}
        ),
        data_stage=lambda: (
            events.append("data") or _data_stage("v1")
        ),
        observer_stage=_observer_stage,
    )
    assert successful["status"] == "COMPLETED_ORDERLESS"
    assert events == ["preflight", "data"]
    assert successful["stages"][0]["stage"] == "LEDGER_PREFLIGHT"

    def corrupt_preflight():
        raise RuntimeError("FORWARD_LEDGER_HASH_CHAIN_MISMATCH")

    failed = AutopilotOrchestrator(tmp_path / "failed").run_once(
        preflight_stage=corrupt_preflight,
        data_stage=lambda: _data_stage("v1"),
        observer_stage=_observer_stage,
    )
    assert failed["status"] == "SYSTEM_DEGRADED"
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["orders_generated"] == 0


def test_order_or_promotion_payload_is_rejected():
    with pytest.raises(RuntimeError, match="ORDER_INVARIANT"):
        assert_orderless_research_payload({"live_orders": 1})
    with pytest.raises(RuntimeError, match="PROMOTION_INVARIANT"):
        assert_orderless_research_payload({"live_ready": True})


def test_active_lock_rejects_duplicate_cycle(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)
    orchestrator._acquire_lock()
    try:
        duplicate = AutopilotOrchestrator(tmp_path)
        with pytest.raises(AutopilotLockError):
            duplicate.run_once(
                data_stage=_data_stage,
                observer_stage=_observer_stage,
            )
    finally:
        orchestrator._release_lock()


def test_insufficient_forward_data_never_degrades_or_promotes(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)
    result = orchestrator.run_once(
        data_stage=_data_stage,
        observer_stage=_observer_stage,
        degradation_observation=DegradationObservation(
            live_return=-0.50,
            cv_mean=0.01,
            cv_std=0.02,
            observation_count=29,
        ),
    )
    assert result["status"] == "COMPLETED_ORDERLESS"
    assert result["degradation"]["status"] == "INSUFFICIENT_FORWARD_DATA"
    assert orchestrator.kill_switch()["system_degraded"] is False
    assert result["paper_candidate_permitted"] is False
    assert result["live_ready"] is False


def test_research_stage_accepts_compact_campaign_result(monkeypatch):
    monkeypatch.setattr(
        "research.global_trial_accounting."
        "audit_global_trial_accounting",
        lambda lab_dir, **kwargs: {
            "status": "PASSED",
            "global_multiple_testing_denominator": 32_100,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_autopilot_data_stage",
        lambda settings, **kwargs: {
            "data_fingerprint": "strict-data-v1",
            "daily_data_fingerprint": "strict-daily-v1",
            "signal_data_fingerprint": "strict-signal-v1",
            "orders_generated": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_autopilot_storm_epoch",
        lambda settings, **kwargs: {
            "status": "REUSED_EXISTING_STORM_EPOCH",
            "total_known_trials": 6_312,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_autopilot_signal_storm_epoch",
        lambda settings, **kwargs: {
            "status": "REUSED_EXISTING_SIGNAL_STORM_EPOCH",
            "total_known_trials": 16_312,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_breakout_portfolio_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "PORTFOLIO_BREAKOUT_V1",
            "parameters_tested": 8,
            "total_known_trials": 1_312,
            "economic_research_lead_count": 0,
            "statistically_qualified_count": 0,
            "frozen_candidate_unchanged": True,
            "paper_candidates": 0,
            "live_orders": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_absolute_momentum_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "ABSOLUTE_MOMENTUM_V1",
            "primary_policy_name": "ABS_MOM_VOL_05",
            "total_known_trials": 16_715,
            "pbo": 0.8857142857142857,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_absolute_momentum_plateau_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
            "generated_trial_count": 117,
            "registered_unique_plateau_trials": 117,
            "registered_epoch_records": 234,
            "total_known_trials": 16_832,
            "plateau_eligible_count": 81,
            "standard_pbo": 0.4857142857142857,
            "plateau_selection_pbo": 0.5142857142857142,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_volatility_contraction_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "VOLATILITY_CONTRACTION_V1",
            "generated_trial_count": 16,
            "registered_unique_trials": 16,
            "registered_epoch_records": 32,
            "total_known_trials": 16_848,
            "primary_strategy_id": (
                "VCB_V20_Q20_E55_X20_T10"
            ),
            "pbo": 0.4714285714285714,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_multi_alpha_ensemble_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "MULTI_ALPHA_ENSEMBLE_V1",
            "generated_trial_count": 1,
            "registered_unique_trials": 1,
            "registered_epoch_records": 2,
            "total_known_trials": 16_849,
            "primary_strategy_id": "MULTI_ALPHA_FIXED_V1",
            "economic_pass": False,
            "statistical_pass": False,
            "inherited_selection_bias_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_trend_pullback_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "TREND_PULLBACK_V1",
            "generated_trial_count": 12,
            "registered_unique_trials": 12,
            "registered_epoch_records": 24,
            "total_known_trials": 16_861,
            "primary_strategy_id": "TP_Z20_E15_EMA100",
            "pbo": 0.5571428571428572,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_range_expansion_4h_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "RANGE_EXPANSION_4H_V1_1",
            "generated_trial_count": 16,
            "registered_unique_trials": 16,
            "registered_epoch_records": 32,
            "total_known_trials": 16_877,
            "primary_strategy_id": (
                "RE4H_E60_X30_R15_V15_EMA600"
            ),
            "pbo": 0.22857142857142856,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_sentiment_recovery_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "SENTIMENT_RECOVERY_V1",
            "generated_trial_count": 8,
            "registered_unique_trials": 8,
            "registered_epoch_records": 16,
            "total_known_trials": 21_320,
            "primary_strategy_id": "SR_F20_D5_EMA100",
            "pbo": 0.25,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_residual_momentum_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "RESIDUAL_MOMENTUM_V1",
            "generated_trial_count": 8,
            "registered_unique_trials": 8,
            "registered_epoch_records": 16,
            "total_known_trials": 21_328,
            "primary_strategy_id": "RM_R60_B180_EMA200",
            "pbo": 0.42857142857142855,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_dual_asset_trend_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "DUAL_ASSET_TREND_V1",
            "generated_trial_count": 1,
            "registered_unique_trials": 1,
            "registered_epoch_records": 2,
            "total_known_trials": 21_329,
            "primary_strategy_id": "DAT_EMA200_COV60_VOL15",
            "pbo": None,
            "pbo_applicable": False,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_liquidity_sweep_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "LIQUIDITY_SWEEP_RECOVERY_V1",
            "generated_trial_count": 8,
            "registered_unique_trials": 8,
            "registered_epoch_records": 8,
            "total_known_trials": 21_337,
            "primary_strategy_id": "LS_F3_V15_H10",
            "pbo": 0.5142857142857142,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_residual_reversal_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "RESIDUAL_REVERSAL_V1",
            "generated_trial_count": 8,
            "registered_unique_trials": 8,
            "registered_epoch_records": 8,
            "total_known_trials": 21_345,
            "primary_strategy_id": "RR_B60_H5_Z20",
            "pbo": 0.02857142857142857,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_multi_alpha_ensemble_v2_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "MULTI_ALPHA_ENSEMBLE_V2",
            "generated_trial_count": 1,
            "registered_unique_trials": 1,
            "registered_epoch_records": 1,
            "total_known_trials": 16_911,
            "primary_strategy_id": "MULTI_ALPHA_FIXED_V2",
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_peer_residual_reversal_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "PEER_RESIDUAL_REVERSAL_V1",
            "generated_trial_count": 4,
            "registered_unique_trials": 4,
            "registered_epoch_records": 4,
            "total_known_trials": 16_915,
            "primary_strategy_id": "PRR_B60_H5_Z20",
            "pbo": 0.7142857142857143,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_btc_shock_diffusion_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "BTC_SHOCK_DIFFUSION_V1",
            "generated_trial_count": 4,
            "registered_unique_trials": 4,
            "registered_epoch_records": 4,
            "total_known_trials": 16_919,
            "primary_strategy_id": "BSD_S3_H3_Z10",
            "pbo": 0.6714285714285714,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_macro_liquidity_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "MACRO_LIQUIDITY_ROTATION_V1",
            "generated_trial_count": 2,
            "registered_unique_trials": 2,
            "registered_epoch_records": 2,
            "total_known_trials": 16_921,
            "primary_strategy_id": "FML_V3_WEEKLY",
            "pbo": 0.7428571428571429,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_multi_horizon_trend_campaign",
        lambda settings: {
            "status": "COMPLETED_NOT_PROMOTED",
            "campaign": "MULTI_HORIZON_TREND_V1",
            "generated_trial_count": 1,
            "registered_unique_trials": 1,
            "registered_epoch_records": 1,
            "total_known_trials": 32_101,
            "primary_strategy_id": "MHT_20_60_120_240_STRUCT240",
            "pbo": None,
            "pbo_applicable": False,
            "economic_pass": False,
            "statistical_pass": False,
            "observer_manifests": {},
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        },
    )
    result = cli._autopilot_research_stage(get_settings())
    assert result["prior_trials_accounted"] == 32_092
    assert result["total_known_trials"] == 32_100
    assert result["trial_accounting"]["status"] == "PASSED"
    assert (
        result["breakout_report_total_known_trials_at_birth"]
        == 1_312
    )
    assert result["portfolio_storm_total_known_trials"] == 6_312
    assert result["signal_synthesis_total_known_trials"] == 16_312
    assert (
        result["parallel_absolute_momentum_plateau_campaign"][
            "total_known_trials"
        ]
        == 16_832
    )
    assert not result[
        "parallel_absolute_momentum_plateau_campaign"
    ]["live_ready"]
    assert (
        result["parallel_volatility_contraction_campaign"][
            "total_known_trials"
        ]
        == 16_848
    )
    assert not result[
        "parallel_volatility_contraction_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_macro_liquidity_campaign"][
            "total_known_trials"
        ]
        == 16_921
    )
    assert not result["parallel_macro_liquidity_campaign"][
        "live_ready"
    ]
    assert (
        result["parallel_multi_alpha_ensemble_campaign"][
            "total_known_trials"
        ]
        == 16_849
    )
    assert not result[
        "parallel_multi_alpha_ensemble_campaign"
    ]["inherited_selection_bias_pass"]
    assert (
        result["parallel_trend_pullback_campaign"][
            "total_known_trials"
        ]
        == 16_861
    )
    assert not result[
        "parallel_trend_pullback_campaign"
    ]["economic_pass"]
    assert (
        result["parallel_range_expansion_4h_campaign"][
            "total_known_trials"
        ]
        == 16_877
    )
    assert not result[
        "parallel_range_expansion_4h_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_sentiment_recovery_campaign"][
            "total_known_trials"
        ]
        == 21_320
    )
    assert not result[
        "parallel_sentiment_recovery_campaign"
    ]["live_ready"]
    assert (
        result["parallel_residual_momentum_campaign"][
            "total_known_trials"
        ]
        == 21_328
    )
    assert not result[
        "parallel_residual_momentum_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_dual_asset_trend_campaign"][
            "total_known_trials"
        ]
        == 21_329
    )
    assert not result[
        "parallel_dual_asset_trend_campaign"
    ]["pbo_applicable"]
    assert (
        result["parallel_liquidity_sweep_campaign"][
            "total_known_trials"
        ]
        == 21_337
    )
    assert not result[
        "parallel_liquidity_sweep_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_residual_reversal_campaign"][
            "total_known_trials"
        ]
        == 21_345
    )
    assert (
        result["parallel_residual_reversal_campaign"]["pbo"]
        < 0.10
    )
    assert not result[
        "parallel_residual_reversal_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_multi_alpha_ensemble_v2_campaign"][
            "total_known_trials"
        ]
        == 16_911
    )
    assert not result[
        "parallel_multi_alpha_ensemble_v2_campaign"
    ]["economic_pass"]
    assert (
        result["parallel_peer_residual_reversal_campaign"][
            "total_known_trials"
        ]
        == 16_915
    )
    assert (
        result["parallel_peer_residual_reversal_campaign"]["pbo"]
        > 0.10
    )
    assert not result[
        "parallel_peer_residual_reversal_campaign"
    ]["statistical_pass"]
    assert (
        result["parallel_btc_shock_diffusion_campaign"][
            "total_known_trials"
        ]
        == 16_919
    )
    assert (
        result["parallel_btc_shock_diffusion_campaign"]["pbo"]
        > 0.10
    )
    assert not result[
        "parallel_btc_shock_diffusion_campaign"
    ]["economic_pass"]
    assert result["paper_candidate_permitted"] is False
    assert result["live_ready"] is False


def test_windows_autopilot_task_is_daily_orderless_and_dry_runnable():
    settings = get_settings()
    xml = cli._autopilot_task_xml(settings)
    assert "<DaysInterval>1</DaysInterval>" in xml
    assert "T03:15:00" in xml
    assert "<RunLevel>LeastPrivilege</RunLevel>" in xml
    assert "lab campaign autopilot --run-research --refresh-data" in xml
    assert "paper" not in xml.casefold()
    assert "live" not in xml.casefold()

    return_code, payload = cli._autopilot_task_command(
        settings,
        mode="task-install",
        confirmed=False,
        dry_run=True,
    )
    assert return_code == 0
    assert payload["status"] == "DRY_RUN"
    assert payload["orders_generated"] == 0
    assert payload["live_ready"] is False
    assert payload["schedule"] == (
        "DAILY_03:15_LOCAL_START_WHEN_AVAILABLE"
    )


def test_incomplete_daily_watermark_waits_without_kill_switch(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)
    result = orchestrator.run_once(
        data_stage=lambda: {
            **_data_stage(),
            "complete_daily_snapshot": False,
        },
        observer_stage=lambda: pytest.fail(
            "observer must not run on partial daily data"
        ),
        research_stage=lambda: pytest.fail(
            "research must not run on partial daily data"
        ),
    )

    assert result["status"] == "WAITING_FOR_COMPLETE_DAILY_SNAPSHOT"
    assert result["research_reason"] == "DATA_NOT_READY"
    assert orchestrator.kill_switch()["system_degraded"] is False


def test_daily_watermark_requires_exact_previous_utc_day(tmp_path):
    expected = pd.Timestamp("2026-07-24", tz="UTC")
    paths = {}
    for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"):
        path = tmp_path / f"{market}.parquet"
        pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.DatetimeIndex(
                [expected - pd.offsets.Day(1), expected]
            ),
        ).to_parquet(path)
        paths[market] = path

    complete = cli._daily_snapshot_watermark(
        paths,
        now=datetime(2026, 7, 25, 1, 15, tzinfo=UTC),
    )
    assert complete["complete_daily_snapshot"] is True

    lagging = pd.read_parquet(paths["LINK-EUR"]).iloc[:-1]
    lagging.to_parquet(paths["LINK-EUR"])
    waiting = cli._daily_snapshot_watermark(
        paths,
        now=datetime(2026, 7, 25, 1, 15, tzinfo=UTC),
    )
    assert waiting["complete_daily_snapshot"] is False
    assert waiting["checks"]["LINK-EUR"] is False
