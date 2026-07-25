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
    result = cli._autopilot_research_stage(object())
    assert result["prior_trials_accounted"] == 1_304
    assert result["total_known_trials"] == 1_312
    assert result["portfolio_storm_total_known_trials"] == 6_312
    assert result["signal_synthesis_total_known_trials"] == 16_312
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
