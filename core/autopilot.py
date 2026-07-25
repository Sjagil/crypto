"""Fail-closed orchestration for recurring research and observer cycles.

The autopilot deliberately owns scheduling and evidence persistence only. It
cannot submit orders or promote a candidate to paper/live. Existing research
engines remain responsible for calculations and their own immutable DNA.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from utils.common import atomic_write_json, read_json, stable_hash, utc_now

Stage = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AutopilotPolicy:
    """Operational policy for a bounded autonomous research loop."""

    interval_seconds: float = 86_400.0
    research_interval_seconds: float = 604_800.0
    degradation_z_threshold: float = -2.0
    minimum_degradation_observations: int = 30
    stale_lock_seconds: float = 14_400.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.research_interval_seconds <= 0:
            raise ValueError("research_interval_seconds must be positive")
        if self.degradation_z_threshold >= 0:
            raise ValueError("degradation_z_threshold must be negative")
        if self.minimum_degradation_observations < 2:
            raise ValueError("minimum_degradation_observations must be at least 2")
        if self.stale_lock_seconds <= 0:
            raise ValueError("stale_lock_seconds must be positive")


@dataclass(frozen=True, slots=True)
class DegradationObservation:
    """Forward evidence used to assess performance degradation."""

    live_return: float
    cv_mean: float
    cv_std: float
    observation_count: int
    window: str = "30d"
    source: str = "forward_observer"


def performance_degradation_z_score(
    *,
    live_return: float,
    cv_mean: float,
    cv_std: float,
) -> float | None:
    """Return a finite degradation z-score, or ``None`` when undefined."""

    values = (live_return, cv_mean, cv_std)
    if not all(math.isfinite(float(value)) for value in values):
        return None
    if float(cv_std) <= 0:
        return None
    score = (float(live_return) - float(cv_mean)) / float(cv_std)
    return float(score) if math.isfinite(score) else None


def _timestamp(value: datetime | None = None) -> str:
    selected = value or utc_now()
    if selected.tzinfo is None or selected.utcoffset() is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _numeric_order_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for key, value in payload.items():
        normalized = str(key).casefold()
        if isinstance(value, Mapping):
            count += _numeric_order_count(value)
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    count += _numeric_order_count(item)
            if "order" in normalized:
                count += len(value)
            continue
        if "order" in normalized and isinstance(value, (int, float)):
            count += max(0, int(value))
    return count


def assert_orderless_research_payload(payload: Mapping[str, Any]) -> None:
    """Reject any stage result that implies execution or lifecycle promotion."""

    if _numeric_order_count(payload):
        raise RuntimeError("AUTOPILOT_ORDER_INVARIANT_VIOLATED")
    forbidden_truthy = {
        "live_ready",
        "paper_candidate_permitted",
        "paper_ready",
        "production_ready",
    }
    for key in forbidden_truthy:
        if bool(payload.get(key, False)):
            raise RuntimeError(f"AUTOPILOT_PROMOTION_INVARIANT_VIOLATED:{key}")


class AutopilotLockError(RuntimeError):
    """Raised when another healthy autopilot cycle owns the lock."""


class AutopilotOrchestrator:
    """Run persistent, bounded and orderless autonomous research cycles."""

    schema_version = 1

    def __init__(
        self,
        root: Path | str,
        *,
        policy: AutopilotPolicy | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.root = Path(root)
        self.policy = policy or AutopilotPolicy()
        self.clock = clock
        self.state_path = self.root / "state.json"
        self.kill_switch_path = self.root / "degradation_state.json"
        self.lock_path = self.root / "autopilot.lock"
        self.cycles_dir = self.root / "cycles"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cycles_dir.mkdir(parents=True, exist_ok=True)
        self._lock_token: str | None = None

    def _now(self) -> datetime:
        current = self.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            current = current.replace(tzinfo=UTC)
        return current.astimezone(UTC)

    def state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return dict(read_json(self.state_path))
        return {
            "schema_version": self.schema_version,
            "status": "NOT_RUN",
            "cycle_count": 0,
            "last_data_fingerprint": None,
            "last_research_at": None,
            "last_research_data_fingerprint": None,
            "last_feature_store_dataset_id": None,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    def kill_switch(self) -> dict[str, Any]:
        if self.kill_switch_path.is_file():
            return dict(read_json(self.kill_switch_path))
        return {
            "schema_version": self.schema_version,
            "system_degraded": False,
            "status": "HEALTHY",
            "activated_at": None,
            "reason": None,
            "manual_reset_required": False,
            "events": [],
        }

    def status(self) -> dict[str, Any]:
        state = self.state()
        switch = self.kill_switch()
        return {
            "status": (
                "SYSTEM_DEGRADED"
                if switch["system_degraded"]
                else state.get("status", "NOT_RUN")
            ),
            "policy": asdict(self.policy),
            "state": state,
            "kill_switch": switch,
            "lock_active": self.lock_path.is_file(),
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    def _activate_kill_switch(
        self,
        *,
        reason: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = self._now()
        previous = self.kill_switch()
        events = list(previous.get("events") or [])
        events.append(
            {
                "timestamp": _timestamp(now),
                "event": "ACTIVATED",
                "reason": reason,
                "evidence": dict(evidence or {}),
            }
        )
        state = {
            "schema_version": self.schema_version,
            "system_degraded": True,
            "status": "SYSTEM_DEGRADED",
            "activated_at": previous.get("activated_at") or _timestamp(now),
            "reason": reason,
            "manual_reset_required": True,
            "events": events,
        }
        atomic_write_json(self.kill_switch_path, state)
        return state

    def reset_kill_switch(
        self,
        *,
        reason: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        if not confirmed:
            raise PermissionError("manual confirmation is required")
        if not reason.strip():
            raise ValueError("reset reason is required")
        previous = self.kill_switch()
        events = list(previous.get("events") or [])
        events.append(
            {
                "timestamp": _timestamp(self._now()),
                "event": "MANUAL_RESET",
                "reason": reason.strip(),
            }
        )
        state = {
            "schema_version": self.schema_version,
            "system_degraded": False,
            "status": "HEALTHY",
            "activated_at": None,
            "reason": None,
            "manual_reset_required": False,
            "events": events,
        }
        atomic_write_json(self.kill_switch_path, state)
        return state

    def evaluate_degradation(
        self,
        observation: DegradationObservation | None,
    ) -> dict[str, Any]:
        if observation is None:
            return {
                "status": "INSUFFICIENT_FORWARD_DATA",
                "reason": "NO_DEGRADATION_OBSERVATION",
                "z_score": None,
                "threshold": self.policy.degradation_z_threshold,
            }
        payload = asdict(observation)
        if observation.observation_count < self.policy.minimum_degradation_observations:
            return {
                "status": "INSUFFICIENT_FORWARD_DATA",
                "reason": "MINIMUM_OBSERVATIONS_NOT_REACHED",
                "z_score": None,
                "threshold": self.policy.degradation_z_threshold,
                "observation": payload,
            }
        score = performance_degradation_z_score(
            live_return=observation.live_return,
            cv_mean=observation.cv_mean,
            cv_std=observation.cv_std,
        )
        if score is None:
            switch = self._activate_kill_switch(
                reason="DEGRADATION_METRIC_UNDEFINED",
                evidence=payload,
            )
            return {
                "status": "SYSTEM_DEGRADED",
                "reason": switch["reason"],
                "z_score": None,
                "threshold": self.policy.degradation_z_threshold,
                "observation": payload,
            }
        if score < self.policy.degradation_z_threshold:
            switch = self._activate_kill_switch(
                reason="PERFORMANCE_DEGRADATION_THRESHOLD_BREACHED",
                evidence={**payload, "z_score": score},
            )
            return {
                "status": "SYSTEM_DEGRADED",
                "reason": switch["reason"],
                "z_score": score,
                "threshold": self.policy.degradation_z_threshold,
                "observation": payload,
            }
        return {
            "status": "HEALTHY",
            "reason": "DEGRADATION_GATE_PASSED",
            "z_score": score,
            "threshold": self.policy.degradation_z_threshold,
            "observation": payload,
        }

    def _lock_age_seconds(self) -> float | None:
        if not self.lock_path.is_file():
            return None
        try:
            payload = read_json(self.lock_path)
            acquired = _parse_timestamp(payload.get("acquired_at"))
        except (OSError, ValueError, TypeError):
            acquired = None
        if acquired is None:
            return max(0.0, time.time() - self.lock_path.stat().st_mtime)
        return max(0.0, (self._now() - acquired).total_seconds())

    def _acquire_lock(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_file():
            age = self._lock_age_seconds()
            if age is None or age <= self.policy.stale_lock_seconds:
                raise AutopilotLockError("AUTOPILOT_ALREADY_RUNNING")
            self.lock_path.unlink(missing_ok=True)
        token = uuid.uuid4().hex
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise AutopilotLockError("AUTOPILOT_ALREADY_RUNNING") from exc
        try:
            payload = {
                "token": token,
                "pid": os.getpid(),
                "acquired_at": _timestamp(self._now()),
            }
            os.write(descriptor, f"{payload}\n".encode())
        finally:
            os.close(descriptor)
        atomic_write_json(
            self.lock_path,
            {
                "token": token,
                "pid": os.getpid(),
                "acquired_at": _timestamp(self._now()),
            },
        )
        self._lock_token = token

    def _release_lock(self) -> None:
        if self._lock_token is None:
            return
        try:
            payload = read_json(self.lock_path) if self.lock_path.is_file() else {}
            if payload.get("token") == self._lock_token:
                self.lock_path.unlink(missing_ok=True)
        finally:
            self._lock_token = None

    def _research_due(
        self,
        *,
        state: Mapping[str, Any],
        data_fingerprint: str | None,
        force: bool,
    ) -> tuple[bool, str]:
        if force:
            return True, "FORCED"
        last_research = _parse_timestamp(state.get("last_research_at"))
        if last_research is None:
            return True, "NEVER_RUN"
        if data_fingerprint == state.get(
            "last_research_data_fingerprint"
        ):
            return False, "DATA_UNCHANGED"
        due_at = last_research + timedelta(
            seconds=self.policy.research_interval_seconds
        )
        if self._now() < due_at:
            return False, "RESEARCH_INTERVAL_NOT_ELAPSED"
        return True, "NEW_DATA_AND_INTERVAL_ELAPSED"

    def _run_stage(self, name: str, stage: Stage) -> dict[str, Any]:
        started = self._now()
        payload = dict(stage())
        assert_orderless_research_payload(payload)
        return {
            "stage": name,
            "status": "PASSED",
            "started_at": _timestamp(started),
            "completed_at": _timestamp(self._now()),
            "payload": payload,
        }

    def run_once(
        self,
        *,
        data_stage: Stage,
        observer_stage: Stage,
        feature_store_stage: Stage | None = None,
        research_stage: Stage | None = None,
        degradation_observation: DegradationObservation | None = None,
        force_research: bool = False,
    ) -> dict[str, Any]:
        """Run one complete cycle and persist all evidence atomically."""

        self._acquire_lock()
        cycle_id = f"AUTO_{self._now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        started = self._now()
        prior = self.state()
        cycle: dict[str, Any] = {
            "schema_version": self.schema_version,
            "cycle_id": cycle_id,
            "started_at": _timestamp(started),
            "status": "RUNNING",
            "policy": asdict(self.policy),
            "stages": [],
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        cycle_path = self.cycles_dir / f"{cycle_id}.json"
        atomic_write_json(cycle_path, cycle)
        try:
            switch = self.kill_switch()
            if switch["system_degraded"]:
                cycle.update(
                    {
                        "status": "SYSTEM_DEGRADED",
                        "reason": "PERSISTENT_KILL_SWITCH_ACTIVE",
                        "completed_at": _timestamp(self._now()),
                        "kill_switch": switch,
                    }
                )
                atomic_write_json(cycle_path, cycle)
                return cycle

            data_result = self._run_stage("DATA_AUDIT", data_stage)
            cycle["stages"].append(data_result)
            data_fingerprint = data_result["payload"].get("data_fingerprint")

            feature_store_dataset_id = prior.get(
                "last_feature_store_dataset_id"
            )
            if feature_store_stage is not None:
                feature_result = self._run_stage(
                    "FEATURE_STORE",
                    feature_store_stage,
                )
                cycle["stages"].append(feature_result)
                feature_store_dataset_id = feature_result["payload"].get(
                    "dataset_id"
                )
            else:
                cycle["stages"].append(
                    {
                        "stage": "FEATURE_STORE",
                        "status": "SKIPPED",
                        "reason": "FEATURE_STORE_DISABLED",
                    }
                )

            research_ran = False
            research_reason = "RESEARCH_DISABLED"
            if research_stage is not None:
                research_due, research_reason = self._research_due(
                    state=prior,
                    data_fingerprint=(
                        str(data_fingerprint) if data_fingerprint else None
                    ),
                    force=force_research,
                )
                if research_due:
                    cycle["stages"].append(
                        self._run_stage("RESEARCH", research_stage)
                    )
                    research_ran = True
                else:
                    cycle["stages"].append(
                        {
                            "stage": "RESEARCH",
                            "status": "SKIPPED",
                            "reason": research_reason,
                        }
                    )
            else:
                cycle["stages"].append(
                    {
                        "stage": "RESEARCH",
                        "status": "SKIPPED",
                        "reason": research_reason,
                    }
                )

            observer_result = self._run_stage("OBSERVER_AUDIT", observer_stage)
            cycle["stages"].append(observer_result)
            degradation = self.evaluate_degradation(degradation_observation)
            cycle["degradation"] = degradation
            if degradation["status"] == "SYSTEM_DEGRADED":
                cycle["status"] = "SYSTEM_DEGRADED"
            else:
                cycle["status"] = "COMPLETED_ORDERLESS"
            cycle["research_ran"] = research_ran
            cycle["research_reason"] = research_reason
            cycle["completed_at"] = _timestamp(self._now())
            atomic_write_json(cycle_path, cycle)

            state = {
                "schema_version": self.schema_version,
                "status": cycle["status"],
                "cycle_count": int(prior.get("cycle_count") or 0) + 1,
                "last_cycle_id": cycle_id,
                "last_cycle_path": str(cycle_path),
                "last_started_at": cycle["started_at"],
                "last_completed_at": cycle["completed_at"],
                "last_data_fingerprint": data_fingerprint,
                "last_research_at": (
                    cycle["completed_at"]
                    if research_ran
                    else prior.get("last_research_at")
                ),
                "last_research_data_fingerprint": (
                    data_fingerprint
                    if research_ran
                    else prior.get("last_research_data_fingerprint")
                ),
                "last_feature_store_dataset_id": (
                    feature_store_dataset_id
                ),
                "research_ran": research_ran,
                "research_reason": research_reason,
                "degradation": degradation,
                "orders_generated": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
            atomic_write_json(self.state_path, state)
            return cycle
        except Exception as exc:
            switch = self._activate_kill_switch(
                reason="AUTOPILOT_STAGE_FAILURE",
                evidence={
                    "cycle_id": cycle_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            cycle.update(
                {
                    "status": "SYSTEM_DEGRADED",
                    "reason": switch["reason"],
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "kill_switch": switch,
                    "completed_at": _timestamp(self._now()),
                }
            )
            atomic_write_json(cycle_path, cycle)
            atomic_write_json(
                self.state_path,
                {
                    "schema_version": self.schema_version,
                    "status": "SYSTEM_DEGRADED",
                    "cycle_count": int(prior.get("cycle_count") or 0) + 1,
                    "last_cycle_id": cycle_id,
                    "last_cycle_path": str(cycle_path),
                    "last_started_at": cycle["started_at"],
                    "last_completed_at": cycle["completed_at"],
                    "last_data_fingerprint": prior.get(
                        "last_data_fingerprint"
                    ),
                    "last_research_at": prior.get("last_research_at"),
                    "last_research_data_fingerprint": prior.get(
                        "last_research_data_fingerprint"
                    ),
                    "last_feature_store_dataset_id": prior.get(
                        "last_feature_store_dataset_id"
                    ),
                    "orders_generated": 0,
                    "paper_candidate_permitted": False,
                    "live_ready": False,
                },
            )
            return cycle
        finally:
            self._release_lock()

    def run_loop(
        self,
        cycle: Callable[[], Mapping[str, Any]],
        *,
        max_cycles: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run recurring bounded cycles; ``None`` means until interrupted."""

        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be at least 1 or None")
        results: list[dict[str, Any]] = []
        while max_cycles is None or len(results) < max_cycles:
            results.append(dict(cycle()))
            if max_cycles is not None and len(results) >= max_cycles:
                break
            remaining = self.policy.interval_seconds
            while remaining > 0:
                pause = min(60.0, remaining)
                time.sleep(pause)
                remaining -= pause
        return results

    @staticmethod
    def fingerprint_files(paths: list[Path]) -> str:
        """Hash file metadata deterministically without loading large datasets."""

        rows: list[dict[str, Any]] = []
        for path in sorted({item.resolve() for item in paths}):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
        return stable_hash(rows, length=64)
