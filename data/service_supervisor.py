"""Least-privilege crash supervisor for the continuous shadow collector."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from data.data_loader import ContinuousDataService
from utils.common import atomic_write_json, read_json, utc_now


class CollectorSupervisor:
    """Monitor one collector process and restart it only after a real exit."""

    def __init__(
        self,
        *,
        checkpoints_directory: Path,
        operations_directory: Path,
        restart_delay_seconds: float = 30.0,
        heartbeat_seconds: float = 5.0,
    ) -> None:
        self.checkpoints_directory = checkpoints_directory
        self.operations_directory = operations_directory
        self.restart_delay_seconds = restart_delay_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.lock_path = (
            checkpoints_directory / "collector_supervisor.lock"
        )
        self.disabled_path = (
            checkpoints_directory / "collector_supervisor.disabled"
        )
        self.service_lock_path = (
            checkpoints_directory / "data_service.lock"
        )
        self.service_heartbeat_path = (
            checkpoints_directory / "operate-shadow_heartbeat.json"
        )
        self.health_path = (
            operations_directory / "collector_supervisor_health.json"
        )
        self._owner_token = uuid.uuid4().hex
        previous_health = (
            dict(read_json(self.health_path))
            if self.health_path.is_file()
            else {}
        )
        self._restart_count = int(
            previous_health.get("restart_count") or 0
        )
        self._stale_locks_recovered = int(
            previous_health.get(
                "stale_service_locks_recovered"
            )
            or 0
        )

    def _acquire(self) -> bool:
        self.checkpoints_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        inspection = ContinuousDataService.inspect_lock_path(
            self.lock_path
        )
        if not inspection["available"]:
            return False
        if inspection["exists"] and inspection["stale"]:
            ContinuousDataService.recover_stale_lock_path(
                self.lock_path
            )
        descriptor = os.open(
            self.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "owner_token": self._owner_token,
                    "hostname": socket.gethostname(),
                    "started_at": utc_now().isoformat(),
                    "service": "collector-supervisor",
                },
                stream,
            )
        return True

    def _release(self) -> None:
        if not self.lock_path.is_file():
            return
        try:
            owner = dict(read_json(self.lock_path))
        except (OSError, TypeError, ValueError):
            return
        if owner.get("owner_token") == self._owner_token:
            self.lock_path.unlink(missing_ok=True)

    def _write_health(
        self,
        *,
        state: str,
        child_pid: int | None = None,
        child_exit_code: int | None = None,
        reason_code: str,
    ) -> dict[str, Any]:
        service = ContinuousDataService.inspect_lock_path(
            self.service_lock_path
        )
        service_heartbeat = (
            dict(read_json(self.service_heartbeat_path))
            if self.service_heartbeat_path.is_file()
            else {}
        )
        payload = {
            "schema_version": "collector_supervisor_health_v1",
            "status": state,
            "reason_code": reason_code,
            "observed_at": utc_now().isoformat(),
            "supervisor_pid": os.getpid(),
            "child_pid": child_pid,
            "child_exit_code": child_exit_code,
            "restart_count": self._restart_count,
            "stale_service_locks_recovered": (
                self._stale_locks_recovered
            ),
            "disabled": self.disabled_path.is_file(),
            "service_lock": service,
            "service_heartbeat_state": service_heartbeat.get(
                "state"
            ),
            "service_heartbeat_at": service_heartbeat.get(
                "heartbeat_at"
            ),
            "least_privilege": True,
            "mode": "shadow",
            "private_exchange_requests": 0,
            "orders_generated": 0,
        }
        atomic_write_json(self.health_path, payload)
        return payload

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (
            time.monotonic() < deadline
            and not self.disabled_path.is_file()
        ):
            time.sleep(min(1.0, deadline - time.monotonic()))

    def run(
        self,
        command: Sequence[str],
        *,
        working_directory: Path,
    ) -> dict[str, Any]:
        """Run until explicitly disabled; never restart an intentional stop."""

        if not self._acquire():
            return self._write_health(
                state="ALREADY_RUNNING",
                reason_code="LIVE_SUPERVISOR_LOCK_PRESENT",
            )
        child: subprocess.Popen[Any] | None = None
        try:
            while not self.disabled_path.is_file():
                service = ContinuousDataService.inspect_lock_path(
                    self.service_lock_path
                )
                if not service["available"]:
                    self._write_health(
                        state="MONITORING",
                        child_pid=(
                            (service.get("owner") or {}).get("pid")
                        ),
                        reason_code="EXISTING_COLLECTOR_ALIVE",
                    )
                    self._wait(self.heartbeat_seconds)
                    continue
                if service["exists"] and service["stale"]:
                    ContinuousDataService.recover_stale_lock_path(
                        self.service_lock_path
                    )
                    self._stale_locks_recovered += 1
                child = subprocess.Popen(
                    list(command),
                    cwd=working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0
                    ),
                )
                self._write_health(
                    state="RUNNING_CHILD",
                    child_pid=child.pid,
                    reason_code="COLLECTOR_STARTED",
                )
                while (
                    child.poll() is None
                    and not self.disabled_path.is_file()
                ):
                    self._write_health(
                        state="RUNNING_CHILD",
                        child_pid=child.pid,
                        reason_code="COLLECTOR_PROCESS_ALIVE",
                    )
                    self._wait(self.heartbeat_seconds)
                if self.disabled_path.is_file():
                    self._write_health(
                        state="DISABLED_WAITING_FOR_CHILD",
                        child_pid=child.pid,
                        reason_code="INTENTIONAL_STOP_REQUESTED",
                    )
                    while child.poll() is None:
                        time.sleep(0.2)
                    break
                exit_code = child.returncode
                self._restart_count += 1
                self._write_health(
                    state="RESTART_WAIT",
                    child_pid=child.pid,
                    child_exit_code=exit_code,
                    reason_code="COLLECTOR_EXITED_UNEXPECTEDLY",
                )
                self._wait(self.restart_delay_seconds)
            return self._write_health(
                state="DISABLED",
                child_pid=child.pid if child else None,
                child_exit_code=(
                    child.returncode if child else None
                ),
                reason_code="SUPERVISOR_DISABLED",
            )
        finally:
            self._release()

    def status(self) -> dict[str, Any]:
        health = (
            dict(read_json(self.health_path))
            if self.health_path.is_file()
            else {
                "schema_version": (
                    "collector_supervisor_health_v1"
                ),
                "status": "NOT_STARTED",
            }
        )
        return {
            **health,
            "supervisor_lock": (
                ContinuousDataService.inspect_lock_path(
                    self.lock_path
                )
            ),
            "disabled": self.disabled_path.is_file(),
        }


__all__ = ["CollectorSupervisor"]
