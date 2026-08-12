from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.event_driven_paper import run_event_driven_paper_once
from core.event_driven_playbooks import build_event_driven_opportunities
from reporting.canonical_economics import ECONOMIC_SCHEMA_VERSION, canonical_family
from tests.test_event_driven_playbooks import context, realtime_row
from utils.common import atomic_write_json, stable_hash


def _opportunities(observed: datetime) -> tuple[dict, list[dict]]:
    snapshot = {
        "observed_at": observed.isoformat(),
        "markets": [realtime_row("BTC-EUR"), realtime_row()],
    }
    return snapshot, build_event_driven_opportunities(
        snapshot,
        tactical_opportunities=[context()],
        macro_regime="MACRO_RISK_OFF",
    )


def _write_economics_gate(output: Path, families: set[str]) -> dict:
    artifact = {
        "schema_version": ECONOMIC_SCHEMA_VERSION,
        "promotion_recommendations": [
            {
                "strategy_family": family,
                "promotion_status": "BLOCKED_NEGATIVE_EXPECTANCY",
                "recommendation": "PAUSE_PAPER_GENERATION",
            }
            for family in sorted(families)
        ],
    }
    artifact["artifact_hash"] = stable_hash(artifact, length=64)
    artifact_path = (
        output
        / "economics"
        / "runs"
        / "test-run"
        / "canonical_strategy_family_economics.json"
    )
    atomic_write_json(artifact_path, artifact)
    atomic_write_json(
        output / "economics" / "latest.json",
        {
            "artifact_path": str(artifact_path.resolve()),
            "artifact_hash": artifact["artifact_hash"],
        },
    )
    return artifact


def test_event_paper_entry_restart_tp1_hold_and_hard_stop(
    isolated_settings,
    tmp_path: Path,
) -> None:
    isolated_settings.paths.output_dir = tmp_path / "output"
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    snapshot, opportunities = _opportunities(observed)

    opened = run_event_driven_paper_once(
        isolated_settings,
        opportunities=opportunities,
        realtime_snapshot=snapshot,
        observed_at=observed,
    )

    assert opened["real_orders"] == 0
    assert opened["private_exchange_requests"] == 0
    assert len(opened["positions"]) == 1
    assert opened["events"][0]["event"] == "PAPER_ORDER_INTENT_CREATED"
    position = next(iter(opened["positions"].values()))

    tp1_snapshot = {
        **snapshot,
        "observed_at": (observed + timedelta(minutes=5)).isoformat(),
        "markets": [
            snapshot["markets"][0],
            {
                **snapshot["markets"][1],
                "price": position["take_profit_1"],
            },
        ],
    }
    managed = run_event_driven_paper_once(
        isolated_settings,
        opportunities=opportunities,
        realtime_snapshot=tp1_snapshot,
        observed_at=observed + timedelta(minutes=5),
    )
    current = next(iter(managed["positions"].values()))
    assert current["tp1_reached"] is True
    assert current["stop_loss"] == current["entry_price"]

    held = run_event_driven_paper_once(
        isolated_settings,
        opportunities=[],
        realtime_snapshot=tp1_snapshot,
        observed_at=observed + timedelta(hours=7),
    )
    assert position["opportunity_id"] in held["positions"]

    stopped_snapshot = {
        **tp1_snapshot,
        "observed_at": (observed + timedelta(hours=8)).isoformat(),
        "markets": [
            {
                **row,
                "price": (
                    float(position["stop_loss"]) * 0.99
                    if row["market"] == position["market"]
                    else row["price"]
                ),
            }
            for row in tp1_snapshot["markets"]
        ],
    }
    closed = run_event_driven_paper_once(
        isolated_settings,
        opportunities=[],
        realtime_snapshot=stopped_snapshot,
        observed_at=observed + timedelta(hours=8),
    )
    assert position["opportunity_id"] not in closed["positions"]
    assert any(event["state"] == "CLOSED" for event in closed["events"])
    assert closed["reconciliation"]["healthy"] is True


def test_negative_economics_blocks_entries_but_not_existing_exit_management(
    isolated_settings,
    tmp_path: Path,
) -> None:
    isolated_settings.paths.output_dir = tmp_path / "output"
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    snapshot, opportunities = _opportunities(observed)
    entry_families = {
        canonical_family(str(row["playbook_id"]))[0]
        for row in opportunities
        if row["state"] == "ENTRY_READY" and not row["hard_blockers"]
    }

    opened = run_event_driven_paper_once(
        isolated_settings,
        opportunities=opportunities,
        realtime_snapshot=snapshot,
        observed_at=observed,
    )
    assert len(opened["positions"]) == 1
    position = next(iter(opened["positions"].values()))

    artifact = _write_economics_gate(
        isolated_settings.paths.output_dir,
        entry_families,
    )
    stopped_snapshot = {
        **snapshot,
        "markets": [
            {
                **row,
                "price": (
                    float(position["stop_loss"]) * 0.99
                    if row["market"] == position["market"]
                    else row["price"]
                ),
            }
            for row in snapshot["markets"]
        ],
    }
    closed = run_event_driven_paper_once(
        isolated_settings,
        opportunities=opportunities,
        realtime_snapshot=stopped_snapshot,
        observed_at=observed + timedelta(minutes=1),
    )

    assert closed["positions"] == {}
    remaining_families = sorted(
        entry_families - {canonical_family(str(position["playbook_id"]))[0]}
    )
    assert closed["entry_economics_gate"] == {
        "status": "READY",
        "new_entries_allowed": True,
        "block_all_new_entries": False,
        "paused_families": sorted(entry_families),
        "live_entry_families": [],
        "live_entry_strategy_dna_hashes": [],
        "live_new_entries_allowed": False,
        "artifact_path": closed["entry_economics_gate"]["artifact_path"],
        "artifact_hash": artifact["artifact_hash"],
        "entry_candidate_count": len(remaining_families),
        "blocked_entry_candidate_count": len(remaining_families),
        "eligible_entry_candidate_count": 0,
        "blocked_candidate_families": remaining_families,
    }
    assert any(
        event.get("state") == "CLOSED" and event.get("reason") == "HARD_STOP"
        for event in closed["events"]
    )
    assert any(
        event.get("event") == "PAPER_ENTRY_COHORT_GATED"
        and event.get("position_management_affected") is False
        for event in closed["events"]
    )


def test_invalid_present_economics_fails_closed_for_new_entries(
    isolated_settings,
    tmp_path: Path,
) -> None:
    isolated_settings.paths.output_dir = tmp_path / "output"
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    snapshot, opportunities = _opportunities(observed)
    artifact = _write_economics_gate(
        isolated_settings.paths.output_dir,
        {"MOMENTUM"},
    )
    latest_path = isolated_settings.paths.output_dir / "economics" / "latest.json"
    latest = {
        "artifact_path": str(
            (
                isolated_settings.paths.output_dir
                / "economics"
                / "runs"
                / "test-run"
                / "canonical_strategy_family_economics.json"
            ).resolve()
        ),
        "artifact_hash": f"tampered-{artifact['artifact_hash']}",
    }
    atomic_write_json(latest_path, latest)

    state = run_event_driven_paper_once(
        isolated_settings,
        opportunities=opportunities,
        realtime_snapshot=snapshot,
        observed_at=observed,
    )

    assert state["positions"] == {}
    assert state["entry_economics_gate"]["status"] == (
        "INVALID_EVIDENCE_FAIL_CLOSED"
    )
    assert state["entry_economics_gate"]["block_all_new_entries"] is True
    assert state["entry_economics_gate"]["blocked_entry_candidate_count"] == 5
