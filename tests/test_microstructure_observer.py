from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from research.microstructure_observer import (
    MicrostructureObserverIntegrityError,
    audit_crowding_observer,
    observe_microstructure_snapshots,
)
from utils.common import read_json, stable_hash


def _write_snapshot(
    directory: Path,
    hour_start: datetime,
    *,
    status: str = "COMPLETE",
    warmup: bool = False,
) -> Path:
    complete = status == "COMPLETE"
    derivatives = {
        "funding_zscore": None if warmup else 2.6,
        "open_interest_change": None if warmup else 0.16,
    }
    market = {
        "market": "BTC-EUR",
        "hour_start": hour_start.isoformat(),
        "hour_end": (hour_start + timedelta(hours=1)).isoformat(),
        "status": status,
        "reason_codes": [] if complete else ["SEQUENCE_GAP"],
        "first_arrival_timestamp": (
            hour_start + timedelta(seconds=20)
        ).isoformat(),
        "last_arrival_timestamp": (
            hour_start + timedelta(minutes=59)
        ).isoformat(),
        "required_field_coverage": {
            "spot_cvd_input_available": complete,
            "orderbook_available": complete,
            "funding_available": complete,
            "open_interest_available": complete,
            "basis_available": complete,
        },
        "source_record_hashes": ["a" * 64],
        "derivatives_positioning": derivatives,
        "perpetual_spot_volume_ratio": (
            None if warmup else 2.1
        ),
        "spot_cvd_robust_zscore": None if warmup else -0.6,
    }
    body = {
        "schema_version": "microstructure_hourly_snapshot_v1",
        "hour_start": hour_start.isoformat(),
        "hour_end": (hour_start + timedelta(hours=1)).isoformat(),
        "finalized_at": (
            hour_start + timedelta(hours=1, minutes=5)
        ).isoformat(),
        "status": status,
        "markets": [market],
        "stream_health": {
            "state": "CONNECTED",
            "sequence_gaps": 0 if complete else 1,
            "dropped_messages": 0,
            "reconnects": 0,
        },
        "ledger_root_hash": "b" * 64,
        "synthetic_data_used": False,
        "orders_generated": 0,
    }
    payload = {
        **body,
        "snapshot_hash": stable_hash(body, length=64),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / hour_start.strftime("%Y%m%dT%H0000Z.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_observer_records_gap_warmup_and_fixed_dna_signals(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features"
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    _write_snapshot(features, start, status="DATA_GAP")
    _write_snapshot(features, start + timedelta(hours=1), warmup=True)
    _write_snapshot(features, start + timedelta(hours=2))
    observer = tmp_path / "observer"
    result = observe_microstructure_snapshots(
        feature_directory=features,
        observer_directory=observer,
        plan_path=tmp_path / "plan.json",
    )
    assert result["observation_count"] == 3
    assert result["new_observation_count"] == 3
    assert result["observation_status_counts"] == {
        "DATA_GAP_NOT_EVALUATED": 1,
        "FEATURE_WARMUP": 1,
        "EVALUATED": 1,
    }
    assert result["block_signal_count"] == 4
    evaluated = read_json(
        observer
        / "observations"
        / "20260726T100000Z.json"
    )
    assert len(evaluated["markets"][0]["dna_results"]) == 4
    assert all(
        row["block_new_long"]
        for row in evaluated["markets"][0]["dna_results"]
    )
    assert not evaluated["research_selection_performed"]
    assert not evaluated["portfolio_target_generated"]
    assert evaluated["orders_generated"] == 0
    assert not evaluated["paper_permitted"]
    assert not evaluated["live_permitted"]
    assert audit_crowding_observer(observer)["status"] == "PASSED"


def test_observer_is_idempotent_and_append_only(tmp_path: Path) -> None:
    features = tmp_path / "features"
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    _write_snapshot(features, start)
    kwargs = {
        "feature_directory": features,
        "observer_directory": tmp_path / "observer",
        "plan_path": tmp_path / "plan.json",
    }
    first = observe_microstructure_snapshots(**kwargs)
    second = observe_microstructure_snapshots(**kwargs)
    assert first["observation_count"] == second["observation_count"] == 1
    assert second["new_observation_count"] == 0
    assert first["chain_root_hash"] == second["chain_root_hash"]

    observation_path = (
        tmp_path
        / "observer"
        / "observations"
        / "20260726T080000Z.json"
    )
    tampered = dict(read_json(observation_path))
    tampered["orders_generated"] = 1
    observation_path.write_text(
        json.dumps(tampered),
        encoding="utf-8",
    )
    with pytest.raises(
        MicrostructureObserverIntegrityError,
        match="HISTORY_INVALID",
    ):
        observe_microstructure_snapshots(**kwargs)


def test_observer_rejects_source_snapshot_revision(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features"
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    source = _write_snapshot(features, start)
    observer = tmp_path / "observer"
    observe_microstructure_snapshots(
        feature_directory=features,
        observer_directory=observer,
        plan_path=tmp_path / "plan.json",
    )
    revised = dict(read_json(source))
    revised["snapshot_hash"] = "f" * 64
    source.write_text(json.dumps(revised), encoding="utf-8")
    audit = audit_crowding_observer(observer)
    assert audit["status"] == "FAILED"
    assert any(
        reason.startswith("SOURCE_SNAPSHOT_HASH_MISMATCH")
        for reason in audit["failures"]
    )
