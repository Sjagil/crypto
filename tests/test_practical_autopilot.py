from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from config.settings import Settings
from core.practical_autopilot import (
    PracticalAutopilot,
    PracticalAutopilotLockError,
)


def _isolated_autopilot(tmp_path: Path) -> PracticalAutopilot:
    autopilot = PracticalAutopilot(
        Settings.load(
            env_file=tmp_path / "does-not-exist.env",
            create_directories=False,
        )
    )
    autopilot.lock_path = tmp_path / "cycle.lock"
    autopilot.supervisor_lock_path = tmp_path / "supervisor.lock"
    autopilot.status_path = tmp_path / "status.json"
    autopilot.heartbeat_path = tmp_path / "heartbeat.json"
    autopilot.supervisor_path = tmp_path / "supervisor.json"
    autopilot.research_status_path = tmp_path / "research.json"
    autopilot.integrated_live_lock_path = tmp_path / "autonomous_live.lock"
    autopilot.companion_status_path = tmp_path / "companion_services.json"
    return autopilot


def test_pid_probe_detects_current_process_without_shell_command() -> None:
    assert PracticalAutopilot._pid_alive(os.getpid()) is True


def test_permanent_supervisor_lock_prevents_duplicate_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _isolated_autopilot(tmp_path)
    second = _isolated_autopilot(tmp_path)
    monkeypatch.setattr(
        PracticalAutopilot,
        "_pid_alive",
        staticmethod(lambda pid: pid == os.getpid()),
    )

    first._acquire_supervisor()
    try:
        with pytest.raises(PracticalAutopilotLockError):
            second._acquire_supervisor()
    finally:
        first._release_supervisor()

    second._acquire_supervisor()
    second._release_supervisor()
    assert not second.supervisor_lock_path.exists()


def test_background_research_running_from_old_supervisor_is_not_reported_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autopilot = _isolated_autopilot(tmp_path)
    autopilot.supervisor_path.write_text(
        json.dumps({"pid": 222}),
        encoding="utf-8",
    )
    autopilot.research_status_path.write_text(
        json.dumps({"status": "RUNNING", "supervisor_pid": 111}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        PracticalAutopilot,
        "_pid_alive",
        staticmethod(lambda pid: pid == 222),
    )

    status = autopilot.status()

    assert status["research_subprocess_active"] is False
    assert (
        status["background_research"]["status"]
        == "INTERRUPTED_RESTART_RECOVERABLE"
    )


def test_integrated_live_supervisor_and_continuous_research_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autopilot = _isolated_autopilot(tmp_path)
    autopilot.integrated_live_lock_path.write_text(
        json.dumps({"pid": 333}),
        encoding="utf-8",
    )
    autopilot.companion_status_path.write_text(
        json.dumps(
            {
                "services": {
                    "simple_lab": {
                        "status": "RUNNING",
                        "pid": 444,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        PracticalAutopilot,
        "_pid_alive",
        staticmethod(lambda pid: pid in {333, 444}),
    )

    status = autopilot.status()

    assert status["supervisor_running"] is True
    assert status["supervisor_mode"] == "AUTONOMOUS_LIVE_INTEGRATED"
    assert status["continuous_research"]["running"] is True
    assert status["research_subprocess_active"] is True


def test_external_live_lab_worker_suppresses_duplicate_research_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autopilot = _isolated_autopilot(tmp_path)
    isolated_paths = autopilot.settings.paths.model_copy(
        update={"lab_dir": tmp_path / "lab"}
    )
    autopilot.settings = autopilot.settings.model_copy(
        update={"paths": isolated_paths}
    )
    heartbeat = autopilot.settings.paths.lab_dir / "state" / "heartbeat.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(
        json.dumps({"pid": 4242, "status": "SCREENING:1h"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        PracticalAutopilot,
        "_pid_alive",
        staticmethod(lambda pid: pid == 4242),
    )

    assert autopilot._research_due() is False


@pytest.mark.asyncio
async def test_prospective_context_refresh_precedes_universe_with_core_markets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autopilot = _isolated_autopilot(tmp_path)
    captured: dict[str, Any] = {}
    completed_epoch = datetime(2026, 7, 31, 1, 0, tzinfo=UTC)

    class FakeDataLoader:
        def __init__(self, settings: Settings) -> None:
            captured["settings"] = settings

    class FakeCollector:
        def __init__(self, *, checkpoint_path: Path, snapshot_directory: Path) -> None:
            captured["checkpoint_path"] = checkpoint_path
            captured["snapshot_directory"] = snapshot_directory

        async def collect(
            self,
            *,
            loader: Any,
            markets: tuple[str, ...],
            observed_at: datetime,
        ) -> dict[str, Any]:
            captured["loader"] = loader
            captured["markets"] = markets
            captured["observed_at"] = observed_at
            return {
                "status": "PASSED",
                "last_completed_epoch": completed_epoch,
                "snapshot_path": str(tmp_path / "snapshot.json"),
                "ranking_count": 50,
                "derivatives_count": 4,
                "orders_generated": 0,
            }

    monkeypatch.setattr(
        "core.practical_autopilot.DataLoader",
        FakeDataLoader,
    )
    monkeypatch.setattr(
        "core.practical_autopilot.ProspectiveContextCollector",
        FakeCollector,
    )

    result = await autopilot._refresh_prospective_context()

    assert captured["settings"] is autopilot.settings
    assert captured["markets"] == tuple(
        autopilot.settings.market_data.symbols
    )
    assert "TAO-EUR" not in captured["markets"]
    assert result["status"] == "PASSED"
    assert result["ranking_count"] == 50
    assert result["derivatives_count"] == 4
    assert result["last_completed_epoch"].startswith("2026-07-31T01:00:00")
    assert result["universe_expansion_allowed"] is True
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0


@pytest.mark.asyncio
async def test_permanent_supervisor_keeps_execution_cycles_running_during_research(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autopilot = _isolated_autopilot(tmp_path)
    research_started = threading.Event()
    release_research = threading.Event()
    cycle_count = 0
    sleep_count = 0
    original_sleep = asyncio.sleep

    async def fake_run_once(*, run_research: bool = True) -> dict[str, Any]:
        nonlocal cycle_count
        assert run_research is False
        cycle_count += 1
        return {
            "status": "PASSED",
            "stages": {},
            "last_research_cycle_at": None,
        }

    def slow_research() -> dict[str, Any]:
        research_started.set()
        assert release_research.wait(timeout=2.0)
        return {"status": "PASSED", "orders_generated": 0}

    class StopSupervisor(RuntimeError):
        pass

    async def short_sleep(_: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        await original_sleep(0.02)
        if sleep_count >= 2:
            release_research.set()
            await original_sleep(0.02)
            raise StopSupervisor

    monkeypatch.setattr(autopilot, "_acquire_supervisor", lambda: None)
    monkeypatch.setattr(autopilot, "_release_supervisor", lambda: None)
    monkeypatch.setattr(autopilot, "_research_due", lambda: True)
    monkeypatch.setattr(autopilot, "_run_existing_research", slow_research)
    monkeypatch.setattr(autopilot, "run_once", fake_run_once)
    monkeypatch.setattr("core.practical_autopilot.asyncio.sleep", short_sleep)

    with pytest.raises(StopSupervisor):
        await autopilot.run()

    assert research_started.is_set()
    assert cycle_count >= 2
    research_status = json.loads(
        autopilot.research_status_path.read_text(encoding="utf-8")
    )
    assert research_status["execution_cycles_continue"] is True
