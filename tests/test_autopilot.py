from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import core.cli as cli
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
    calls = {"research": 0}

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

    orchestrator = AutopilotOrchestrator(tmp_path, clock=clock)
    first = orchestrator.run_once(
        data_stage=_data_stage,
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert first["status"] == "COMPLETED_ORDERLESS"
    assert first["research_ran"] is True
    assert calls["research"] == 1
    assert first["orders_generated"] == 0
    assert not orchestrator.lock_path.exists()

    current += timedelta(days=1)
    second = orchestrator.run_once(
        data_stage=_data_stage,
        research_stage=research,
        observer_stage=_observer_stage,
    )
    assert second["status"] == "COMPLETED_ORDERLESS"
    assert second["research_ran"] is False
    assert second["research_reason"] == "DATA_UNCHANGED"
    assert calls["research"] == 1
    assert orchestrator.state()["cycle_count"] == 2


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


def test_degradation_breach_persists_and_blocks_later_cycles(tmp_path):
    orchestrator = AutopilotOrchestrator(tmp_path)
    degraded = orchestrator.run_once(
        data_stage=_data_stage,
        observer_stage=_observer_stage,
        degradation_observation=DegradationObservation(
            live_return=-0.06,
            cv_mean=0.01,
            cv_std=0.02,
            observation_count=30,
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
    result = cli._autopilot_research_stage(object())
    assert result["prior_trials_accounted"] == 1_304
    assert result["total_known_trials"] == 1_312
    assert result["paper_candidate_permitted"] is False
    assert result["live_ready"] is False
