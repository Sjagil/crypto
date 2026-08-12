from __future__ import annotations

import json
import os
from pathlib import Path

from config.settings import PathSettings
from data.service_supervisor import CollectorSupervisor
from reporting.prospective_readiness import (
    prospective_readiness_report,
)
from utils.common import atomic_write_json


def supervisor(tmp_path: Path) -> CollectorSupervisor:
    return CollectorSupervisor(
        checkpoints_directory=tmp_path / "checkpoints",
        operations_directory=tmp_path / "operations",
        restart_delay_seconds=0.01,
        heartbeat_seconds=0.01,
    )


def test_disabled_supervisor_exits_without_starting_child(
    tmp_path: Path,
) -> None:
    selected = supervisor(tmp_path)
    selected.disabled_path.parent.mkdir(parents=True)
    selected.disabled_path.write_text("{}", encoding="utf-8")

    result = selected.run(
        ["not-executed"],
        working_directory=tmp_path,
    )

    assert result["status"] == "DISABLED"
    assert result["orders_generated"] == 0
    assert not selected.lock_path.exists()


def test_live_supervisor_lock_prevents_duplicate(
    tmp_path: Path,
) -> None:
    selected = supervisor(tmp_path)
    selected.lock_path.parent.mkdir(parents=True)
    selected.lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "owner_token": "other",
                "service": "collector-supervisor",
            }
        ),
        encoding="utf-8",
    )

    result = selected.run(
        ["not-executed"],
        working_directory=tmp_path,
    )

    assert result["status"] == "ALREADY_RUNNING"
    assert result["reason_code"] == "LIVE_SUPERVISOR_LOCK_PRESENT"
    assert result["private_exchange_requests"] == 0


def test_supervisor_preserves_lifetime_recovery_counters(
    tmp_path: Path,
) -> None:
    operations = tmp_path / "operations"
    operations.mkdir()
    atomic_write_json(
        operations / "collector_supervisor_health.json",
        {
            "restart_count": 3,
            "stale_service_locks_recovered": 2,
        },
    )
    selected = supervisor(tmp_path)
    selected.disabled_path.parent.mkdir(parents=True)
    selected.disabled_path.write_text("{}", encoding="utf-8")

    result = selected.run(
        ["not-executed"],
        working_directory=tmp_path,
    )

    assert result["restart_count"] == 3
    assert result["stale_service_locks_recovered"] == 2


def test_readiness_matrix_never_promotes_incomplete_evidence(
    isolated_settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    operations = settings.paths.output_dir / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        operations / "orderflow_stream_health.json",
        {"status": "HEALTHY", "record_count": 10, "provider": {}},
    )
    atomic_write_json(
        operations / "microstructure_readiness.json",
        {
            "backtest_permitted": False,
            "complete_hours": 1,
            "consecutive_complete_hours": 1,
            "synthetic_data_used": False,
        },
    )

    report = prospective_readiness_report(settings)

    assert report["overall_status"] == (
        "OPERATIONALLY_LIVE_RESEARCH_FINANCIALLY_BLOCKED"
    )
    assert not report["profitable_strategy_proven"]
    assert not report["orderflow_backtest_permitted"]
    assert not report["paper_permitted"]
    assert not report["live_permitted"]
    assert report["orders_generated"] == 0
    assert "PROSPECTIVE_ORDERFLOW_HISTORY_90D" in (
        report["live_blockers"]
    )
