"""P1.2.2 prospective dataset maturation and research handoff governance.

This module evaluates collection quality only.  It deliberately exposes no
strategy runner, backtester, model trainer, order surface, or live-authority
mutation.  A research-ready transition may freeze data and notify an operator;
it can never start research automatically.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from utils.common import (
    atomic_write_json,
    parse_utc,
    read_json,
    stable_hash,
    stable_json,
    utc_iso,
    utc_now,
)

READINESS_POLICY_VERSION = "research_readiness_policy_v1"
READINESS_HISTORY_SCHEMA = "immutable_readiness_transition_v1"
FAMILY_FREEZE_SCHEMA = "family_dataset_freeze_v1"
STORAGE_POLICY_VERSION = "multi_source_storage_policy_v1"
STORAGE_SAMPLE_SCHEMA = "multi_source_storage_sample_v1"
OVERLAP_SCHEMA = "cross_venue_multi_resolution_overlap_v1"
OWNERSHIP_SCHEMA = "multi_source_collector_ownership_v1"
DEPLOYMENT_SCHEMA = "multi_source_collector_deployment_v1"
PRIMARY_ASSETS = ("CRYPTO:BTC", "CRYPTO:ETH", "CRYPTO:SOL")
RESOLUTIONS_SECONDS = (1, 5, 15, 30, 60, 300)


class CollectorAlreadyActive(RuntimeError):
    pass


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows ``os.kill(pid, 0)`` is not the harmless existence probe it
        # is on POSIX.  Query a minimal process handle without mutating it.
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class CollectorLease:
    """Atomic single-owner lease with safe stale-lock recovery."""

    def __init__(self, path: Path | str, history_root: Path | str) -> None:
        self.path = Path(path)
        self.history_root = Path(history_root)
        self.instance_id = uuid.uuid4().hex
        self.acquired = False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": OWNERSHIP_SCHEMA,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "acquired_at": utc_iso(),
            "heartbeat_at": utc_iso(),
            "dataset_writer_scope": "CANONICAL_MULTI_SOURCE_PIT",
            "orders_generated": 0,
            "private_exchange_requests": 0,
        }

    def acquire(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            payload = self._payload()
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                recovery_guard = self.path.with_name(f"{self.path.name}.recovery")
                for recovery_attempt in range(2):
                    try:
                        guard_descriptor = os.open(
                            recovery_guard,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                    except FileExistsError:
                        try:
                            guard = dict(read_json(recovery_guard))
                        except (OSError, TypeError, ValueError, json.JSONDecodeError):
                            guard = {"pid": -1}
                        if process_exists(int(guard.get("pid") or -1)):
                            raise CollectorAlreadyActive("COLLECTOR_ALREADY_ACTIVE")
                        recovery_guard.unlink(missing_ok=True)
                        if recovery_attempt == 0:
                            continue
                        raise RuntimeError("COLLECTOR_LOCK_RECOVERY_GUARD_FAILED")
                    with os.fdopen(
                        guard_descriptor, "w", encoding="utf-8", newline="\n"
                    ) as stream:
                        stream.write(stable_json({"pid": os.getpid()}))
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    break
                try:
                    try:
                        existing = dict(read_json(self.path))
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        existing = {"pid": -1, "status": "UNREADABLE_STALE_LOCK"}
                    owner_pid = int(existing.get("pid") or -1)
                    if process_exists(owner_pid):
                        raise CollectorAlreadyActive("COLLECTOR_ALREADY_ACTIVE")
                    body = {
                        **existing,
                        "recovered_at": utc_iso(),
                        "recovery_reason": "OWNER_PID_NOT_RUNNING",
                        "replacement_instance_id": self.instance_id,
                    }
                    identity = stable_hash(body, length=32)
                    target = self.history_root / f"stale-{identity}.json"
                    atomic_write_json(target, body)
                    self.path.unlink(missing_ok=True)
                finally:
                    recovery_guard.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(stable_json(payload, indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return payload
        raise RuntimeError("COLLECTOR_LOCK_RECOVERY_FAILED")

    def heartbeat(self) -> dict[str, Any]:
        if not self.acquired:
            raise RuntimeError("COLLECTOR_LEASE_NOT_ACQUIRED")
        existing = dict(read_json(self.path))
        if existing.get("instance_id") != self.instance_id:
            raise RuntimeError("COLLECTOR_LEASE_OWNERSHIP_LOST")
        payload = {**existing, "heartbeat_at": utc_iso()}
        atomic_write_json(self.path, payload)
        return payload

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = dict(read_json(self.path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing = {}
        if existing.get("instance_id") == self.instance_id:
            self.path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> "CollectorLease":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


class ReadinessLevel(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    COLLECTING = "COLLECTING"
    PARTIAL = "PARTIAL"
    EXPLORATORY_USABLE = "EXPLORATORY_USABLE"
    RESEARCH_USABLE = "RESEARCH_USABLE"
    ROBUSTNESS_USABLE = "ROBUSTNESS_USABLE"
    QUALITY_FAILED = "QUALITY_FAILED"


READINESS_RANK = {
    ReadinessLevel.NOT_STARTED: 0,
    ReadinessLevel.COLLECTING: 1,
    ReadinessLevel.PARTIAL: 2,
    ReadinessLevel.EXPLORATORY_USABLE: 3,
    ReadinessLevel.RESEARCH_USABLE: 4,
    ReadinessLevel.ROBUSTNESS_USABLE: 5,
    ReadinessLevel.QUALITY_FAILED: -1,
}


@dataclass(frozen=True, slots=True)
class ReadinessThreshold:
    name: str
    dataset_family: str
    exploratory_minimum_history_days: float
    research_minimum_history_days: float
    robustness_minimum_history_days: float
    exploratory_minimum_observations: int
    research_minimum_observations: int
    robustness_minimum_observations: int
    exploratory_minimum_valid_fraction: float
    research_minimum_valid_fraction: float
    robustness_minimum_valid_fraction: float
    exploratory_maximum_gap_fraction: float
    research_maximum_gap_fraction: float
    robustness_maximum_gap_fraction: float
    required_assets: tuple[str, ...]
    required_quality: tuple[str, ...]
    reason: str
    policy_version: str = READINESS_POLICY_VERSION

    def __post_init__(self) -> None:
        histories = (
            self.exploratory_minimum_history_days,
            self.research_minimum_history_days,
            self.robustness_minimum_history_days,
        )
        observations = (
            self.exploratory_minimum_observations,
            self.research_minimum_observations,
            self.robustness_minimum_observations,
        )
        valid = (
            self.exploratory_minimum_valid_fraction,
            self.research_minimum_valid_fraction,
            self.robustness_minimum_valid_fraction,
        )
        gaps = (
            self.exploratory_maximum_gap_fraction,
            self.research_maximum_gap_fraction,
            self.robustness_maximum_gap_fraction,
        )
        if tuple(sorted(histories)) != histories or tuple(sorted(observations)) != observations:
            raise ValueError("readiness evidence thresholds must increase by maturity level")
        if tuple(sorted(valid)) != valid:
            raise ValueError("valid fraction thresholds must increase by maturity level")
        if tuple(sorted(gaps, reverse=True)) != gaps:
            raise ValueError("gap thresholds must tighten by maturity level")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def research_readiness_policy_v1() -> dict[str, ReadinessThreshold]:
    """One versioned policy; no readiness constants are scattered in runtime code."""

    rows = (
        ReadinessThreshold(
            "BITVAVO_FLOW_POLICY",
            "BITVAVO_FLOW",
            3,
            14,
            60,
            50_000,
            250_000,
            1_000_000,
            0.97,
            0.99,
            0.995,
            0.03,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("TIMESTAMP_PASS", "AGGRESSOR_SEMANTICS_DECLARED", "MULTIPLE_PERIODS"),
            "Flow evidence needs multiple market sessions, causal timestamps and known aggressor semantics.",
        ),
        ReadinessThreshold(
            "BITVAVO_L2_POLICY",
            "BITVAVO_L2",
            3,
            14,
            60,
            5_000,
            50_000,
            250_000,
            0.97,
            0.99,
            0.995,
            0.03,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("BOOK_REPLAY_VALID", "TIMESTAMP_PASS", "MULTIPLE_PERIODS"),
            "L2 research requires stable snapshot/delta reconstruction rather than raw deltas alone.",
        ),
        ReadinessThreshold(
            "CROSS_VENUE_LEAD_LAG_POLICY",
            "CROSS_VENUE_LEAD_LAG",
            7,
            30,
            90,
            100_000,
            500_000,
            2_000_000,
            0.95,
            0.99,
            0.995,
            0.05,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("BITVAVO_CLOCK_PASS", "KRAKEN_CLOCK_PASS", "CLOCK_VALID_OVERLAP"),
            "Lead/lag needs synchronized wall-clock overlap and cannot be inferred from event volume alone.",
        ),
        ReadinessThreshold(
            "MULTI_VENUE_DISLOCATION_POLICY",
            "MULTI_VENUE_DISLOCATION",
            7,
            30,
            90,
            50_000,
            250_000,
            1_000_000,
            0.95,
            0.99,
            0.995,
            0.05,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("REFERENCE_PRICE_VALID", "QUOTE_NORMALIZED", "TIMESTAMP_PASS"),
            "Dislocation research needs comparable quotes and observable source disagreement.",
        ),
        ReadinessThreshold(
            "FLOW_CONFIRMED_SWING_POLICY",
            "FLOW_CONFIRMED_SWING",
            7,
            30,
            90,
            50_000,
            250_000,
            1_000_000,
            0.95,
            0.99,
            0.995,
            0.05,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("HTF_CANDLES_PRESENT", "BITVAVO_FLOW_VALID", "MULTIPLE_PERIODS"),
            "Swing confirmation requires causal HTF candles plus prospective flow over multiple sessions.",
        ),
        ReadinessThreshold(
            "LIQUIDITY_SHOCK_POLICY",
            "LIQUIDITY_SHOCK",
            14,
            30,
            90,
            10_000,
            100_000,
            500_000,
            0.97,
            0.99,
            0.995,
            0.03,
            0.01,
            0.005,
            PRIMARY_ASSETS,
            ("BOOK_REPLAY_VALID", "SPREAD_DEPTH_PRESENT", "FLOW_VALID"),
            "Liquidity withdrawal needs valid books, spread, depth and flow rather than execution deterioration alone.",
        ),
        ReadinessThreshold(
            "CMC_BREADTH_POLICY",
            "CMC_BREADTH",
            7,
            30,
            180,
            168,
            720,
            4_320,
            0.95,
            0.99,
            0.995,
            0.05,
            0.02,
            0.01,
            (),
            ("PIT_UNIVERSE", "NO_FUTURE_MEMBERSHIP", "CONSISTENT_DEFINITION"),
            "Breadth may mature before L2 but needs many PIT snapshots across multiple days.",
        ),
        ReadinessThreshold(
            "BTC_MARKET_REGIME_POLICY",
            "BTC_MARKET_REGIME",
            14,
            60,
            180,
            336,
            1_440,
            4_320,
            0.95,
            0.99,
            0.995,
            0.05,
            0.02,
            0.01,
            ("CRYPTO:BTC",),
            ("PIT_MARKET_CONTEXT", "BTC_PRICE_PRESENT", "MULTIPLE_PERIODS"),
            "Regime research requires BTC and market context across changing conditions.",
        ),
        ReadinessThreshold(
            "MEXC_DERIVATIVES_CONTEXT_POLICY",
            "MEXC_DERIVATIVES_CONTEXT",
            14,
            30,
            180,
            336,
            720,
            4_320,
            0.95,
            0.99,
            0.995,
            0.05,
            0.02,
            0.01,
            PRIMARY_ASSETS,
            ("PIT_DERIVATIVES_CONTEXT", "ASSET_IDENTITY", "INFORMATION_ONLY"),
            "Funding, open interest and basis stay contextual and need overlapping PIT history.",
        ),
        ReadinessThreshold(
            "EVENT_INTELLIGENCE_POLICY",
            "EVENT_INTELLIGENCE",
            30,
            90,
            365,
            50,
            200,
            1_000,
            0.90,
            0.97,
            0.99,
            0.10,
            0.03,
            0.01,
            (),
            ("FIRST_KNOWN_TIME", "ASSET_MAPPING", "SOURCE_QUALITY", "CATEGORY_DIVERSITY"),
            "Event alpha needs unique high-quality timestamped events, not duplicated RSS volume.",
        ),
    )
    return {row.dataset_family: row for row in rows}


def _meets(
    threshold: ReadinessThreshold,
    metrics: Mapping[str, Any],
    level: str,
) -> tuple[bool, list[str]]:
    history = float(metrics.get("history_days") or 0)
    observations = int(metrics.get("observations") or 0)
    valid_fraction = metrics.get("valid_fraction")
    gap_fraction = metrics.get("gap_fraction")
    present_assets = set(metrics.get("assets") or [])
    quality = set(metrics.get("quality") or [])
    minimum_history = float(getattr(threshold, f"{level}_minimum_history_days"))
    minimum_observations = int(getattr(threshold, f"{level}_minimum_observations"))
    minimum_valid = float(getattr(threshold, f"{level}_minimum_valid_fraction"))
    maximum_gap = float(getattr(threshold, f"{level}_maximum_gap_fraction"))
    reasons: list[str] = []
    if history < minimum_history:
        reasons.append("INSUFFICIENT_WALL_CLOCK_HISTORY")
    if observations < minimum_observations:
        reasons.append("INSUFFICIENT_OBSERVATIONS")
    if valid_fraction is None or float(valid_fraction) < minimum_valid:
        reasons.append("VALID_FRACTION_INSUFFICIENT")
    if gap_fraction is None or float(gap_fraction) > maximum_gap:
        reasons.append("GAP_FRACTION_EXCEEDED_OR_UNKNOWN")
    missing_assets = set(threshold.required_assets) - present_assets
    if missing_assets:
        reasons.append("REQUIRED_ASSETS_MISSING")
    missing_quality = set(threshold.required_quality) - quality
    if missing_quality:
        reasons.append("REQUIRED_QUALITY_MISSING")
    return not reasons, reasons


def assess_readiness(
    threshold: ReadinessThreshold,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    observations = int(metrics.get("observations") or 0)
    if observations <= 0:
        level = ReadinessLevel.NOT_STARTED
        reasons = ["NO_OBSERVATIONS"]
    else:
        severe_failure = bool(metrics.get("quality_failure"))
        robustness, robustness_reasons = _meets(threshold, metrics, "robustness")
        research, research_reasons = _meets(threshold, metrics, "research")
        exploratory, exploratory_reasons = _meets(threshold, metrics, "exploratory")
        if severe_failure:
            level = ReadinessLevel.QUALITY_FAILED
            reasons = list(metrics.get("quality_failure_reasons") or ["QUALITY_FAILURE"])
        elif robustness:
            level = ReadinessLevel.ROBUSTNESS_USABLE
            reasons = []
        elif research:
            level = ReadinessLevel.RESEARCH_USABLE
            reasons = robustness_reasons
        elif exploratory:
            level = ReadinessLevel.EXPLORATORY_USABLE
            reasons = research_reasons
        else:
            exploratory_history = threshold.exploratory_minimum_history_days
            exploratory_observations = threshold.exploratory_minimum_observations
            progressed = (
                float(metrics.get("history_days") or 0) >= exploratory_history * 0.25
                or observations >= exploratory_observations * 0.25
            )
            level = ReadinessLevel.PARTIAL if progressed else ReadinessLevel.COLLECTING
            reasons = exploratory_reasons
    body = {
        "schema_version": "family_readiness_assessment_v1",
        "policy_version": threshold.policy_version,
        "family": threshold.dataset_family,
        "state": level.value,
        "metrics": dict(metrics),
        "reason_codes": reasons,
        "threshold": threshold.to_dict(),
        "dataset_ready_for_research": READINESS_RANK[level]
        >= READINESS_RANK[ReadinessLevel.RESEARCH_USABLE],
        "strategy_ready": False,
        "trade_ready": False,
        "automatic_alpha_started": False,
        "automatic_ml_training_started": False,
        "execution_authority": False,
    }
    return {**body, "assessment_hash": stable_hash(body, length=64)}


HYPOTHESIS_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "H1_FLOW_CONFIRMED_SWING_READY": ("FLOW_CONFIRMED_SWING", "BITVAVO_FLOW"),
    "H2_L2_FILTERED_SWING_READY": ("BITVAVO_L2", "FLOW_CONFIRMED_SWING"),
    "H3_CROSS_VENUE_LEAD_LAG_READY": ("CROSS_VENUE_LEAD_LAG",),
    "H4_CROSS_VENUE_FLOW_READY": ("BITVAVO_FLOW", "CROSS_VENUE_LEAD_LAG"),
    "H5_LIQUIDITY_SHOCK_READY": ("LIQUIDITY_SHOCK", "BITVAVO_L2"),
    "H6_BREADTH_MOMENTUM_READY": ("CMC_BREADTH", "FLOW_CONFIRMED_SWING"),
    "H7_DERIVATIVES_MODIFIER_READY": ("MEXC_DERIVATIVES_CONTEXT",),
    "H8_EVENT_ALPHA_READY": ("EVENT_INTELLIGENCE",),
}


def hypothesis_readiness(
    assessments: Mapping[str, Mapping[str, Any]],
    *,
    frozen_families: Sequence[str] = (),
    spot_candidate_available: bool = False,
) -> dict[str, Any]:
    frozen = set(frozen_families)
    output: dict[str, Any] = {}
    for hypothesis, required in HYPOTHESIS_REQUIREMENTS.items():
        unready = [
            family
            for family in required
            if READINESS_RANK.get(
                ReadinessLevel(str((assessments.get(family) or {}).get("state") or "NOT_STARTED")),
                -1,
            )
            < READINESS_RANK[ReadinessLevel.RESEARCH_USABLE]
        ]
        extras: list[str] = []
        if (
            hypothesis == "H3_CROSS_VENUE_LEAD_LAG_READY"
            and "CROSS_VENUE_LEAD_LAG" not in frozen
        ):
            extras.append("UNTOUCHED_HOLDOUT_NOT_FROZEN")
        if hypothesis == "H7_DERIVATIVES_MODIFIER_READY" and not spot_candidate_available:
            extras.append("SPOT_STRATEGY_CANDIDATE_REQUIRED")
        ready = not unready and not extras
        output[hypothesis] = {
            "ready": ready,
            "required_families": list(required),
            "unready_families": unready,
            "additional_blockers": extras,
            "notification_text": "DATASET READY FOR RESEARCH" if ready else None,
            "strategy_ready": False,
            "trade_ready": False,
            "automatic_campaign_started": False,
        }
    return output


class ReadinessHistoryStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _latest(self, family: str) -> dict[str, Any]:
        path = self.root / family.casefold() / "latest.json"
        return dict(read_json(path)) if path.is_file() else {}

    def record(self, assessment: Mapping[str, Any], *, at: datetime | None = None) -> dict[str, Any]:
        selected = at or utc_now()
        family = str(assessment["family"])
        current = str(assessment["state"])
        latest = self._latest(family)
        if latest.get("current_state") == current:
            return {"status": "UNCHANGED", "transition": latest}
        body = {
            "schema_version": READINESS_HISTORY_SCHEMA,
            "policy_version": assessment.get("policy_version"),
            "family": family,
            "previous_state": latest.get("current_state"),
            "current_state": current,
            "transitioned_at": utc_iso(selected),
            "assessment_hash": assessment.get("assessment_hash"),
            "assessment": dict(assessment),
            "orders_generated": 0,
            "research_started": False,
            "strategy_promoted": False,
            "live_authority_changed": False,
        }
        transition_id = stable_hash(body, length=64)
        payload = {**body, "transition_id": transition_id}
        target = (
            self.root
            / family.casefold()
            / "history"
            / f"{selected:%Y%m%dT%H%M%S%fZ}-{transition_id[:16]}.json"
        )
        atomic_write_json(target, payload)
        latest_payload = {
            "schema_version": "readiness_transition_latest_v1",
            "family": family,
            "current_state": current,
            "transition_id": transition_id,
            "transition_path": str(target.resolve()),
            "transitioned_at": payload["transitioned_at"],
        }
        atomic_write_json(self.root / family.casefold() / "latest.json", latest_payload)
        return {"status": "TRANSITION_RECORDED", "transition": payload, "path": str(target)}


class FamilyFreezeManager:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _index_path(self, family: str) -> Path:
        return self.root / family.casefold() / "latest.json"

    def latest(self, family: str) -> dict[str, Any]:
        path = self._index_path(family)
        return dict(read_json(path)) if path.is_file() else {}

    def frozen_families(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                path.parent.name.upper()
                for path in self.root.glob("*/latest.json")
                if path.is_file()
            )
        )

    def maybe_freeze(
        self,
        *,
        assessment: Mapping[str, Any],
        transition: Mapping[str, Any] | None,
        source_manifests: Sequence[Mapping[str, Any]],
        assets: Sequence[str],
        features: Sequence[str],
        collection_start: datetime,
        data_end: datetime,
        coverage: Mapping[str, Any],
        clock_metrics: Mapping[str, Any],
        build_commit: str | None,
        holdout_fraction: float = 0.20,
        minimum_holdout_days: int = 7,
    ) -> dict[str, Any]:
        family = str(assessment["family"])
        state = ReadinessLevel(str(assessment["state"]))
        if READINESS_RANK[state] < READINESS_RANK[ReadinessLevel.RESEARCH_USABLE]:
            return {"status": "NOT_ELIGIBLE", "family": family}
        existing = self.latest(family)
        if existing:
            return {"status": "ALREADY_FROZEN", "family": family, "freeze": existing}
        start = collection_start.astimezone(UTC)
        end = data_end.astimezone(UTC)
        duration = end - start
        holdout_duration = max(timedelta(days=minimum_holdout_days), duration * holdout_fraction)
        if holdout_duration >= duration:
            return {"status": "INSUFFICIENT_HISTORY_FOR_HOLDOUT", "family": family}
        holdout_start = end - holdout_duration
        identity = {
            "schema_version": FAMILY_FREEZE_SCHEMA,
            "family": family,
            "policy_version": assessment.get("policy_version"),
            "source_manifests": [dict(row) for row in source_manifests],
            "assets": list(assets),
            "features": list(features),
            "development_start": utc_iso(start),
            "development_end": utc_iso(holdout_start),
            "holdout_start": utc_iso(holdout_start),
            "holdout_end": utc_iso(end),
            "data_end": utc_iso(end),
            "coverage": dict(coverage),
            "quality": dict(assessment),
            "clock_metrics": dict(clock_metrics),
            "gap_metrics": {
                "gap_fraction": (assessment.get("metrics") or {}).get("gap_fraction")
            },
            "build_commit": build_commit,
            "readiness_transition_id": (transition or {}).get("transition_id"),
        }
        dataset_id = stable_hash(identity, length=64)
        frozen_at = utc_now()
        payload = {
            **identity,
            "dataset_id": dataset_id,
            "freeze_timestamp": utc_iso(frozen_at),
            "holdout_status": "RESERVED_UNTOUCHED",
            "holdout_target_metrics_calculated": False,
            "post_freeze_forward_data_start": utc_iso(frozen_at),
            "future_data_default_partition": "POST_FREEZE_FORWARD_DATA",
            "immutable": True,
            "automatic_stage0_started": False,
            "automatic_backtest_started": False,
            "automatic_ml_training_started": False,
            "automatic_strategy_promotion": False,
            "live_authority_changed": False,
            "orders_generated": 0,
        }
        target = self.root / family.casefold() / dataset_id / "manifest.json"
        atomic_write_json(target, payload)
        index = {
            "schema_version": "family_dataset_freeze_latest_v1",
            "family": family,
            "dataset_id": dataset_id,
            "manifest_path": str(target.resolve()),
            "freeze_timestamp": payload["freeze_timestamp"],
            "holdout_status": payload["holdout_status"],
            "future_data_default_partition": payload["future_data_default_partition"],
        }
        atomic_write_json(self._index_path(family), index)
        return {"status": "FREEZE_CREATED", "family": family, "freeze": payload, "path": str(target)}

    def classify_timestamp(self, family: str, timestamp: datetime) -> str:
        latest = self.latest(family)
        if not latest:
            return "PRE_FREEZE_COLLECTION"
        manifest = dict(read_json(latest["manifest_path"]))
        selected = timestamp.astimezone(UTC)
        if selected >= parse_utc(manifest["post_freeze_forward_data_start"]):
            return "POST_FREEZE_FORWARD_DATA"
        if selected >= parse_utc(manifest["holdout_start"]):
            return "RESERVED_UNTOUCHED_HOLDOUT"
        return "DEVELOPMENT_DATA"


class StorageGrowthMonitor:
    def __init__(
        self,
        root: Path | str,
        *,
        warning_free_fraction: float = 0.20,
        critical_free_fraction: float = 0.10,
        warning_free_bytes: int = 100_000_000_000,
        critical_free_bytes: int = 50_000_000_000,
        minimum_sample_interval_seconds: float = 60.0,
    ) -> None:
        self.root = Path(root)
        self.warning_free_fraction = warning_free_fraction
        self.critical_free_fraction = critical_free_fraction
        self.warning_free_bytes = warning_free_bytes
        self.critical_free_bytes = critical_free_bytes
        self.minimum_sample_interval_seconds = minimum_sample_interval_seconds
        self.latest_path = self.root / "latest.json"

    def _samples(self, now: datetime) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cutoff = now - timedelta(days=8)
        for path in sorted((self.root / "history").rglob("*.json")):
            try:
                row = dict(read_json(path))
                if parse_utc(row["observed_at"]) >= cutoff:
                    rows.append(row)
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return rows

    @staticmethod
    def _flatten(cumulative: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
        return {
            str(key): {
                "events": float(value.get("events") or 0),
                "raw_bytes": float(value.get("raw_bytes") or 0),
                "compressed_events": float(value.get("compressed_events") or 0),
                "compressed_bytes": float(value.get("compressed_bytes") or 0),
            }
            for key, value in cumulative.items()
        }

    def observe(
        self,
        cumulative: Mapping[str, Mapping[str, Any]],
        *,
        disk_path: Path | str,
        force: bool = False,
        at: datetime | None = None,
    ) -> dict[str, Any]:
        now = at or utc_now()
        latest = dict(read_json(self.latest_path)) if self.latest_path.is_file() else {}
        if latest and not force:
            elapsed = (now - parse_utc(latest["observed_at"])).total_seconds()
            if elapsed < self.minimum_sample_interval_seconds:
                return self.report(at=now)
        usage = shutil.disk_usage(disk_path)
        body = {
            "schema_version": STORAGE_SAMPLE_SCHEMA,
            "policy_version": STORAGE_POLICY_VERSION,
            "observed_at": utc_iso(now),
            "cumulative": self._flatten(cumulative),
            "disk": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            "orders_generated": 0,
        }
        sample_hash = stable_hash(body, length=64)
        payload = {**body, "sample_hash": sample_hash}
        target = (
            self.root
            / "history"
            / f"date={now:%Y-%m-%d}"
            / f"sample-{now:%H%M%S}-{sample_hash[:16]}.json"
        )
        atomic_write_json(target, payload)
        atomic_write_json(self.latest_path, payload)
        return self.report(at=now)

    def report(self, *, at: datetime | None = None) -> dict[str, Any]:
        now = at or utc_now()
        samples = self._samples(now)
        if not samples and self.latest_path.is_file():
            samples = [dict(read_json(self.latest_path))]
        samples.sort(key=lambda row: row["observed_at"])
        if not samples:
            return {"status": "NOT_EVALUABLE", "policy_version": STORAGE_POLICY_VERSION}
        current = samples[-1]
        current_time = parse_utc(current["observed_at"])
        windows: dict[str, Any] = {}
        for label, seconds in (("1h", 3600), ("24h", 86400), ("7d", 604800)):
            target = current_time - timedelta(seconds=seconds)
            candidates = [row for row in samples[:-1] if parse_utc(row["observed_at"]) <= target]
            baseline = candidates[-1] if candidates else samples[0]
            baseline_time = parse_utc(baseline["observed_at"])
            elapsed = max(0.0, (current_time - baseline_time).total_seconds())
            rates: dict[str, Any] = {}
            keys = set(current["cumulative"]) | set(baseline["cumulative"])
            for key in sorted(keys):
                newest = current["cumulative"].get(key) or {}
                oldest = baseline["cumulative"].get(key) or {}
                event_delta = max(0.0, float(newest.get("events") or 0) - float(oldest.get("events") or 0))
                byte_delta = max(0.0, float(newest.get("raw_bytes") or 0) - float(oldest.get("raw_bytes") or 0))
                compressed_event_delta = max(
                    0.0,
                    float(newest.get("compressed_events") or 0)
                    - float(oldest.get("compressed_events") or 0),
                )
                compressed_byte_delta = max(
                    0.0,
                    float(newest.get("compressed_bytes") or 0)
                    - float(oldest.get("compressed_bytes") or 0),
                )
                rates[key] = {
                    "observed_seconds": elapsed,
                    "event_delta": event_delta,
                    "raw_byte_delta": byte_delta,
                    "compressed_event_delta": compressed_event_delta,
                    "compressed_byte_delta": compressed_byte_delta,
                    "raw_events_per_day": event_delta / elapsed * 86400 if elapsed else None,
                    "compressed_events_per_day": (
                        compressed_event_delta / elapsed * 86400 if elapsed else None
                    ),
                    "raw_gb_per_day": byte_delta / elapsed * 86400 / 1e9 if elapsed else None,
                    "compressed_gb_per_day": (
                        compressed_byte_delta / elapsed * 86400 / 1e9 if elapsed else None
                    ),
                }
            windows[label] = {
                "requested_seconds": seconds,
                "actual_observed_seconds": elapsed,
                "full_window_available": elapsed >= seconds * 0.95,
                "rates": rates,
            }
        disk = current["disk"]
        total = float(disk["total_bytes"])
        free = float(disk["free_bytes"])
        warning_threshold = max(self.warning_free_bytes, total * self.warning_free_fraction)
        critical_threshold = max(self.critical_free_bytes, total * self.critical_free_fraction)
        aggregate = windows["1h"]["rates"]
        growth_per_day = sum(
            float(row.get("raw_gb_per_day") or 0) * 1e9 for row in aggregate.values()
        )
        if free <= critical_threshold:
            status = "STORAGE_CRITICAL"
        elif free <= warning_threshold:
            status = "STORAGE_WARNING"
        else:
            status = "STORAGE_OK"

        def days_until(threshold: float) -> float | None:
            if free <= threshold:
                return 0.0
            return (free - threshold) / growth_per_day if growth_per_day > 0 else None

        return {
            "schema_version": "multi_source_storage_growth_report_v1",
            "policy_version": STORAGE_POLICY_VERSION,
            "status": status,
            "observed_at": current["observed_at"],
            "sample_count": len(samples),
            "windows": windows,
            "disk": {
                **disk,
                "free_fraction": free / total if total else None,
                "warning_threshold_bytes": warning_threshold,
                "critical_threshold_bytes": critical_threshold,
                "days_until_warning": days_until(warning_threshold),
                "days_until_critical": days_until(critical_threshold),
            },
            "optional_research_sources_pause_permitted": status == "STORAGE_CRITICAL",
            "bitvavo_execution_data_may_pause": False,
        }


class CrossVenueAlignmentMonitor:
    """Persist cumulative per-resolution overlap without mining target returns."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        state = dict(read_json(self.path)) if self.path.is_file() else {}
        self.counters: dict[str, dict[str, int]] = {
            str(key): {str(k): int(v) for k, v in dict(value).items()}
            for key, value in dict(state.get("counters") or {}).items()
        }
        self.active: dict[str, dict[str, Any]] = {
            str(key): dict(value) for key, value in dict(state.get("active") or {}).items()
        }
        self.reconnect_interruptions: dict[str, int] = {
            str(key): int(value)
            for key, value in dict(state.get("reconnect_interruptions") or {}).items()
        }

    @staticmethod
    def _counter_key(day: str, asset: str, resolution: int, kind: str) -> str:
        return f"{day}|{asset}|{resolution}|{kind}"

    @staticmethod
    def _bucket_key(asset: str, resolution: int, bucket: int, kind: str) -> str:
        return f"{asset}|{resolution}|{bucket}|{kind}"

    def _finalize(self, *, before_epoch: float) -> None:
        for key, row in tuple(self.active.items()):
            if float(row["bucket_end_epoch"]) > before_epoch:
                continue
            counter = self.counters.setdefault(
                str(row["counter_key"]),
                {
                    "union_buckets": 0,
                    "matched_buckets": 0,
                    "clock_valid_buckets": 0,
                    "freshness_valid_buckets": 0,
                },
            )
            mask = int(row.get("mask") or 0)
            counter["union_buckets"] += int(mask != 0)
            counter["matched_buckets"] += int(mask == 3)
            counter["clock_valid_buckets"] += int(
                mask == 3 and int(row.get("clock_mask") or 0) == 3
            )
            counter["freshness_valid_buckets"] += int(
                mask == 3 and int(row.get("freshness_mask") or 0) == 3
            )
            del self.active[key]

    def observe(
        self,
        *,
        source: str,
        canonical_asset_id: str,
        event_at: datetime,
        receive_at: datetime,
        kind: str = "trade",
    ) -> None:
        if source not in {"bitvavo", "kraken"}:
            return
        source_bit = 1 if source == "bitvavo" else 2
        event = event_at.astimezone(UTC)
        receive = receive_at.astimezone(UTC)
        latency = (receive - event).total_seconds()
        clock_valid = -0.250 <= latency <= 5.0
        freshness_valid = 0 <= latency <= 2.0
        epoch = event.timestamp()
        day = event.date().isoformat()
        for resolution in RESOLUTIONS_SECONDS:
            bucket = math.floor(epoch / resolution)
            key = self._bucket_key(canonical_asset_id, resolution, bucket, kind)
            row = self.active.setdefault(
                key,
                {
                    "counter_key": self._counter_key(
                        day,
                        canonical_asset_id,
                        resolution,
                        kind,
                    ),
                    "bucket_end_epoch": (bucket + 1) * resolution,
                    "mask": 0,
                    "clock_mask": 0,
                    "freshness_mask": 0,
                },
            )
            row["mask"] = int(row["mask"]) | source_bit
            if clock_valid:
                row["clock_mask"] = int(row["clock_mask"]) | source_bit
            if freshness_valid:
                row["freshness_mask"] = int(row["freshness_mask"]) | source_bit
        self._finalize(before_epoch=utc_now().timestamp() - 600)

    def record_reconnect(self, source: str) -> None:
        self.reconnect_interruptions[source] = self.reconnect_interruptions.get(source, 0) + 1

    def persist(self) -> None:
        body = {
            "schema_version": OVERLAP_SCHEMA,
            "counters": self.counters,
            "active": self.active,
            "reconnect_interruptions": self.reconnect_interruptions,
            "updated_at": utc_iso(),
            "orders_generated": 0,
        }
        atomic_write_json(self.path, {**body, "state_hash": stable_hash(body, length=64)})

    def snapshot(self, *, include_active: bool = True) -> dict[str, Any]:
        counters = {key: dict(value) for key, value in self.counters.items()}
        if include_active:
            for row in self.active.values():
                counter = counters.setdefault(
                    str(row["counter_key"]),
                    {
                        "union_buckets": 0,
                        "matched_buckets": 0,
                        "clock_valid_buckets": 0,
                        "freshness_valid_buckets": 0,
                    },
                )
                mask = int(row.get("mask") or 0)
                counter["union_buckets"] += int(mask != 0)
                counter["matched_buckets"] += int(mask == 3)
                counter["clock_valid_buckets"] += int(
                    mask == 3 and int(row.get("clock_mask") or 0) == 3
                )
                counter["freshness_valid_buckets"] += int(
                    mask == 3 and int(row.get("freshness_mask") or 0) == 3
                )
        by_resolution: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "union_buckets": 0,
                "matched_buckets": 0,
                "clock_valid_buckets": 0,
                "freshness_valid_buckets": 0,
                "per_asset": defaultdict(Counter),
                "daily": defaultdict(Counter),
            }
        )
        for key, row in counters.items():
            day, asset, resolution_text, kind = key.split("|", 3)
            resolution = int(resolution_text)
            if kind != "trade":
                continue
            selected = by_resolution[resolution_text]
            for metric, value in row.items():
                selected[metric] += int(value)
                selected["per_asset"][asset][metric] += int(value)
                selected["daily"][day][metric] += int(value)
            selected["resolution_seconds"] = resolution
        output: dict[str, Any] = {}
        for resolution, row in by_resolution.items():
            seconds = int(row["resolution_seconds"])

            def normalize(values: Mapping[str, int]) -> dict[str, Any]:
                selected_union = int(values.get("union_buckets") or 0)
                selected_matched = int(values.get("matched_buckets") or 0)
                return {
                    **dict(values),
                    "trade_overlap_seconds": selected_matched * seconds,
                    "trade_overlap_minutes": selected_matched * seconds / 60,
                    "percentage_overlap": (
                        selected_matched / selected_union if selected_union else 0.0
                    ),
                    "gap_rate": (
                        1 - selected_matched / selected_union if selected_union else None
                    ),
                    "clock_valid_overlap_seconds": int(
                        values.get("clock_valid_buckets") or 0
                    )
                    * seconds,
                    "freshness_valid_overlap_seconds": int(
                        values.get("freshness_valid_buckets") or 0
                    )
                    * seconds,
                }

            output[f"{resolution}s"] = {
                **normalize(row),
                "per_asset": {
                    asset: normalize(values)
                    for asset, values in dict(row["per_asset"]).items()
                },
                "daily": {
                    day: normalize(values) for day, values in dict(row["daily"]).items()
                },
                "readiness_is_resolution_specific": True,
            }
        return {
            "schema_version": OVERLAP_SCHEMA,
            "resolutions": output,
            "reconnect_interruptions": dict(self.reconnect_interruptions),
            "book_valid_overlap_status": "REQUIRES_BITVAVO_VALID_BOOK_INTERVALS",
            "arbitrary_cross_resolution_promotion": False,
        }


def bitvavo_l2_maturation(snapshot_root: Path | str) -> dict[str, Any]:
    root = Path(snapshot_root)
    per_asset: dict[str, dict[str, Any]] = {
        asset: {
            "asset": asset,
            "closed_intervals": 0,
            "valid_intervals": 0,
            "book_samples": 0,
            "trade_count": 0,
            "states": Counter(),
            "reason_counts": Counter(),
            "first_hour": None,
            "last_hour": None,
            "latest_spread_bps": None,
            "latest_microprice": None,
        }
        for asset in ("BTC-EUR", "ETH-EUR", "SOL-EUR")
    }
    for path in sorted(root.glob("*.json")):
        try:
            snapshot = dict(read_json(path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        for raw in snapshot.get("markets") or []:
            row = dict(raw)
            market = str(row.get("market") or "")
            if market not in per_asset:
                continue
            target = per_asset[market]
            reasons = [str(value) for value in row.get("reason_codes") or []]
            state = str(row.get("status") or snapshot.get("status") or "UNKNOWN")
            book_reasons = {
                "NO_VALID_ORDERBOOK",
                "BOOK_SEQUENCE_INVALID",
                "SEQUENCE_GAP",
                "NO_SNAPSHOT",
                "RECONNECT_RESET",
                "BOOK_STALE",
                "BOOK_INVALID",
            }
            valid = state == "COMPLETE" and not (set(reasons) & book_reasons)
            target["closed_intervals"] += 1
            target["valid_intervals"] += int(valid)
            target["book_samples"] += int(row.get("orderbook_sample_count") or 0)
            target["trade_count"] += int(row.get("trade_count") or 0)
            target["states"]["BOOK_VALID" if valid else "BOOK_GAPPED"] += 1
            target["reason_counts"].update(reasons)
            hour = snapshot.get("hour_start")
            target["first_hour"] = target["first_hour"] or hour
            target["last_hour"] = hour or target["last_hour"]
            target["latest_spread_bps"] = row.get("spread_bps")
            target["latest_microprice"] = row.get("microprice")
    rows: dict[str, Any] = {}
    for market, row in per_asset.items():
        closed = int(row["closed_intervals"])
        valid = int(row["valid_intervals"])
        rows[market] = {
            **{key: value for key, value in row.items() if key not in {"states", "reason_counts"}},
            "book_valid_fraction": valid / closed if closed else 0.0,
            "state_counts": dict(row["states"]),
            "reason_counts": dict(row["reason_counts"]),
            "quality_requirements_lowered": False,
        }
    aggregate_closed = sum(row["closed_intervals"] for row in rows.values())
    aggregate_valid = sum(row["valid_intervals"] for row in rows.values())
    first_values = [parse_utc(row["first_hour"]) for row in rows.values() if row["first_hour"]]
    last_values = [parse_utc(row["last_hour"]) for row in rows.values() if row["last_hour"]]
    duration = (
        (max(last_values) - min(first_values)).total_seconds() / 86400
        if first_values and last_values
        else 0.0
    )
    return {
        "schema_version": "bitvavo_l2_maturation_v1",
        "assets": rows,
        "history_days": duration,
        "closed_asset_intervals": aggregate_closed,
        "valid_asset_intervals": aggregate_valid,
        "book_valid_fraction": (
            aggregate_valid / aggregate_closed if aggregate_closed else 0.0
        ),
        "book_samples": sum(row["book_samples"] for row in rows.values()),
        "quality_failure": aggregate_closed > 0 and aggregate_valid == 0,
        "quality_failure_reasons": (
            ["PROSPECTIVE_BOOK_VALID_FRACTION_ZERO_INVESTIGATE_SEQUENCE_RESEED"]
            if aggregate_closed > 0 and aggregate_valid == 0
            else []
        ),
    }


def mexc_derivatives_maturation(context_root: Path | str) -> dict[str, Any]:
    """Inventory the existing public MEXC context store without collecting twice."""

    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError:
        return {
            "history_days": 0.0,
            "observations": 0,
            "valid_fraction": None,
            "gap_fraction": None,
            "assets": [],
            "quality": ["INFORMATION_ONLY"],
            "status": "NOT_EVALUABLE_DEPENDENCY_MISSING",
            "execution_authority": False,
        }
    root = Path(context_root)
    per_asset: dict[str, Any] = {}
    total_rows = 0
    valid_rows = 0
    missing_intervals = 0
    observed_intervals = 0
    durations: list[float] = []
    all_information_only = True
    all_pit = True
    for asset in ("BTC", "ETH", "SOL"):
        path = root / f"derivatives_mexc_{asset}.parquet"
        if not path.is_file():
            per_asset[asset] = {"status": "MISSING", "rows": 0}
            continue
        available_columns = set(pq.read_schema(path).names)
        selected_columns = [
            name
            for name in (
                "available_at",
                "observed_at",
                "observation_time",
                "point_in_time_status",
                "funding_rate",
                "open_interest",
                "basis",
                "execution_permitted",
                "canonical_market",
            )
            if name in available_columns
        ]
        frame = pq.read_table(path, columns=selected_columns).to_pandas()
        timestamp_column = next(
            (
                name
                for name in ("available_at", "observed_at", "observation_time")
                if name in frame
            ),
            None,
        )
        timestamps = (
            pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").dropna()
            if timestamp_column
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        complete = frame[["funding_rate", "open_interest", "basis"]].notna().all(axis=1)
        information_only = (
            not bool(frame["execution_permitted"].fillna(False).astype(bool).any())
            if "execution_permitted" in frame
            else False
        )
        pit = (
            bool(frame["point_in_time_status"].notna().all())
            if "point_in_time_status" in frame
            else False
        )
        gaps = 0
        expected_seconds: float | None = None
        if len(timestamps) >= 2:
            differences = timestamps.sort_values().diff().dt.total_seconds().dropna()
            positive = differences[differences > 0]
            if not positive.empty:
                expected_seconds = float(positive.median())
                gaps = int(
                    sum(max(0, math.ceil(value / expected_seconds) - 1) for value in positive)
                )
                observed_intervals += len(positive)
        rows = len(frame)
        valid = int(complete.sum())
        duration = (
            max(0.0, (timestamps.max() - timestamps.min()).total_seconds() / 86400)
            if len(timestamps) >= 2
            else 0.0
        )
        durations.append(duration)
        total_rows += rows
        valid_rows += valid
        missing_intervals += gaps
        all_information_only = all_information_only and information_only
        all_pit = all_pit and pit
        per_asset[asset] = {
            "status": "PRESENT",
            "path": str(path.resolve()),
            "rows": rows,
            "valid_rows": valid,
            "valid_fraction": valid / rows if rows else None,
            "history_days": duration,
            "first_known_at": utc_iso(timestamps.min().to_pydatetime())
            if len(timestamps)
            else None,
            "last_known_at": utc_iso(timestamps.max().to_pydatetime())
            if len(timestamps)
            else None,
            "expected_cadence_seconds": expected_seconds,
            "missing_intervals": gaps,
            "point_in_time": pit,
            "execution_permitted": not information_only,
        }
    assets = [f"CRYPTO:{asset}" for asset, row in per_asset.items() if row.get("rows")]
    quality = []
    if len(assets) == 3 and all_pit:
        quality.extend(["PIT_DERIVATIVES_CONTEXT", "ASSET_IDENTITY"])
    if all_information_only:
        quality.append("INFORMATION_ONLY")
    denominator = total_rows + missing_intervals
    return {
        "schema_version": "mexc_derivatives_context_maturation_v1",
        "history_days": min(durations or [0.0]),
        "observations": valid_rows,
        "valid_fraction": valid_rows / total_rows if total_rows else None,
        "gap_fraction": missing_intervals / denominator if denominator else None,
        "assets": assets,
        "quality": quality,
        "per_asset": per_asset,
        "observed_intervals": observed_intervals,
        "missing_intervals": missing_intervals,
        "spot_reference_separate": True,
        "derivatives_role": "INFORMATION_ONLY",
        "execution_authority": False,
        "staking_execution": False,
        "lending_execution": False,
        "shorting": False,
    }


def classify_event(
    *,
    source: str,
    title: str,
    summary: str,
    existing_categories: Sequence[str] = (),
) -> dict[str, Any]:
    text = f"{title} {summary}".casefold()
    asset_terms = {
        "CRYPTO:BTC": ("bitcoin", " btc "),
        "CRYPTO:ETH": ("ethereum", "ether", " eth "),
        "CRYPTO:SOL": ("solana", " sol "),
        "CRYPTO:LINK": ("chainlink", " link "),
    }
    category_terms = {
        "EXCHANGE_LISTING": ("listing", "lists ", "listed "),
        "EXCHANGE_DELISTING": ("delisting", "delists", "delisted"),
        "NETWORK_UPGRADE": ("upgrade", "mainnet", "hard fork", "fork"),
        "TOKEN_UNLOCK": ("token unlock", "vesting unlock"),
        "SECURITY_INCIDENT": ("hack", "exploit", "security incident", "breach"),
        "PROTOCOL_INCIDENT": ("protocol incident", "validator incident", "halt"),
        "EXCHANGE_OUTAGE": ("outage", "maintenance", "degraded"),
        "REGULATORY_EVENT": ("regulator", "regulation", "sec ", "mica", "lawsuit"),
        "OFFICIAL_ANNOUNCEMENT": ("announce", "announcement", "release"),
    }
    assets = [asset for asset, terms in asset_terms.items() if any(term in f" {text} " for term in terms)]
    categories = set(existing_categories)
    categories.update(
        category
        for category, terms in category_terms.items()
        if any(term in text for term in terms)
    )
    source_label = source.casefold()
    official = any(
        name in source_label
        for name in (
            "kraken",
            "bitvavo",
            "european central bank",
            "federal reserve",
            "official project",
        )
    )
    return {
        "canonical_asset_ids": assets,
        "event_categories": sorted(categories or {"GENERAL_PUBLIC_INFORMATION"}),
        "source_quality": "PRIMARY_OFFICIAL" if official else "SECONDARY_REPUTABLE",
        "high_value_event": bool(categories & set(category_terms)),
        "social_noise_source": False,
    }


def api_usage_report(
    budget: Mapping[str, Any],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = observed_at or utc_now()
    day = now.date().isoformat()
    providers: dict[str, Any] = {}
    for provider, raw in dict(budget.get("providers") or {}).items():
        row = dict(raw)
        today = dict((row.get("by_day") or {}).get(day) or {})
        requests_today = int(today.get("requests") or 0)
        credits_today = int(today.get("credits") or 0)
        elapsed_day_fraction = max(
            1 / 1440,
            (now.hour * 3600 + now.minute * 60 + now.second) / 86400,
        )
        cache_hits = int(row.get("cache_hits") or 0)
        avoided = int(row.get("duplicate_requests_avoided") or 0)
        attempts = requests_today + cache_hits + avoided
        providers[provider] = {
            "configured": True,
            "credits_today": credits_today,
            "requests_today": requests_today,
            "estimated_daily_credit_burn": credits_today / elapsed_day_fraction,
            "estimated_monthly_credit_burn": credits_today / elapsed_day_fraction * 30,
            "daily_credit_limit": row.get("daily_credit_limit"),
            "monthly_credit_limit": row.get("monthly_credit_limit"),
            "daily_credit_utilization": (
                credits_today / int(row["daily_credit_limit"])
                if row.get("daily_credit_limit")
                else None
            ),
            "cache_hit_or_avoidance_ratio": (
                (cache_hits + avoided) / attempts if attempts else 0.0
            ),
            "failed_requests": int(row.get("failed_requests") or 0),
            "rate_limit_events": int(row.get("rate_limit_events") or 0),
            "quota_exhaustion_events": int(row.get("quota_exhaustion_events") or 0),
            "credits_not_burned_merely_because_available": True,
        }
    return {
        "schema_version": "paid_api_usage_governance_v1",
        "observed_at": utc_iso(now),
        "providers": providers,
        "credentials_serialized": False,
    }


class RuntimePerformanceMonitor:
    def __init__(self) -> None:
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.previous: dict[str, Any] | None = None

    @staticmethod
    def _windows_process_metrics() -> dict[str, Any]:
        if os.name != "nt":
            return {"working_set_bytes": None, "io_read_bytes": None, "io_write_bytes": None}
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.GetProcessIoCounters.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(IO_COUNTERS),
            )
            kernel32.GetProcessIoCounters.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            memory = PROCESS_MEMORY_COUNTERS()
            memory.cb = ctypes.sizeof(memory)
            io = IO_COUNTERS()
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                raise OSError("GetProcessMemoryInfo failed")
            if not kernel32.GetProcessIoCounters(handle, ctypes.byref(io)):
                raise OSError("GetProcessIoCounters failed")
            return {
                "working_set_bytes": int(memory.WorkingSetSize),
                "peak_working_set_bytes": int(memory.PeakWorkingSetSize),
                "io_read_bytes": int(io.ReadTransferCount),
                "io_write_bytes": int(io.WriteTransferCount),
            }
        except (AttributeError, ctypes.ArgumentError, OSError, ValueError):
            return {"working_set_bytes": None, "io_read_bytes": None, "io_write_bytes": None}

    def snapshot(
        self,
        *,
        total_events: int,
        queue_size: int,
        queue_capacity: int,
        compaction: Mapping[str, Any],
        api_scheduler: Mapping[str, Any],
    ) -> dict[str, Any]:
        wall = time.perf_counter()
        cpu = time.process_time()
        process = self._windows_process_metrics()
        current = {
            "wall": wall,
            "cpu": cpu,
            "events": total_events,
            "io_read": process.get("io_read_bytes"),
            "io_write": process.get("io_write_bytes"),
        }
        previous = self.previous or {
            "wall": self.started_wall,
            "cpu": self.started_cpu,
            "events": 0,
            "io_read": current["io_read"],
            "io_write": current["io_write"],
        }
        elapsed = max(1e-9, wall - float(previous["wall"]))
        result = {
            "schema_version": "multi_source_runtime_performance_v1",
            "observed_at": utc_iso(),
            "events_per_second": max(0, total_events - int(previous["events"])) / elapsed,
            "feature_builds_per_second": None,
            "queue_backlog": queue_size,
            "queue_capacity": queue_capacity,
            "queue_utilization": queue_size / queue_capacity if queue_capacity else None,
            "cpu_process_utilization_single_core": max(
                0.0, (cpu - float(previous["cpu"])) / elapsed
            ),
            "memory": process,
            "disk_io_read_bytes_per_second": (
                (int(current["io_read"]) - int(previous["io_read"])) / elapsed
                if current["io_read"] is not None and previous["io_read"] is not None
                else None
            ),
            "disk_io_write_bytes_per_second": (
                (int(current["io_write"]) - int(previous["io_write"])) / elapsed
                if current["io_write"] is not None and previous["io_write"] is not None
                else None
            ),
            "parquet_compaction": dict(compaction),
            "api_scheduler": dict(api_scheduler),
            "execution_supervisor_dependency": False,
        }
        self.previous = current
        return result


def record_deployment_event(
    root: Path | str,
    *,
    instance_id: str,
    previous_status: Mapping[str, Any],
    current_status: Mapping[str, Any],
    reason: str,
    continuity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def compact_heads(status: Mapping[str, Any]) -> dict[str, Any]:
        return {
            source: {
                key: checkpoint.get(key)
                for key in (
                    "source",
                    "record_count",
                    "root_hash",
                    "first_known_at",
                    "last_known_at",
                    "last_segment",
                    "last_segment_size_bytes",
                )
            }
            for source, checkpoint in dict(
                status.get("ledger_checkpoints") or {}
            ).items()
        }

    previous_at = previous_status.get("observed_at")
    current_at = current_status.get("observed_at")
    observation_span_seconds = (
        max(0.0, (parse_utc(current_at) - parse_utc(previous_at)).total_seconds())
        if previous_at and current_at
        else None
    )
    body = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "instance_id": instance_id,
        "reason": reason,
        "recorded_at": utc_iso(),
        "previous_status_observed_at": previous_at,
        "current_status_observed_at": current_at,
        "status_observation_span_seconds": observation_span_seconds,
        "measured_status_gap_seconds": (continuity or {}).get(
            "maximum_restart_gap_seconds"
        ),
        "previous_ledger_heads": compact_heads(previous_status),
        "current_ledger_heads": compact_heads(current_status),
        "hash_continuity_expected": True,
        "orders_generated": 0,
        "live_authority_changed": False,
    }
    deployment_id = stable_hash(body, length=64)
    payload = {**body, "deployment_id": deployment_id}
    target = Path(root) / f"{deployment_id}.json"
    if not target.is_file():
        atomic_write_json(target, payload)
    return {**payload, "path": str(target.resolve())}


def verify_restart_continuity(
    previous_status: Mapping[str, Any],
    current_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify only the post-checkpoint suffix while a collector keeps running."""

    previous_heads = dict(previous_status.get("ledger_checkpoints") or {})
    current_heads = dict(current_status.get("ledger_checkpoints") or {})
    instance_start_raw = (current_status.get("ownership") or {}).get("acquired_at")
    instance_start = parse_utc(instance_start_raw) if instance_start_raw else None
    sources: dict[str, Any] = {}
    for source, previous_raw in previous_heads.items():
        previous = dict(previous_raw)
        current = dict(current_heads.get(source) or {})
        previous_count = int(previous.get("record_count") or 0)
        current_count = int(current.get("record_count") or 0)
        previous_hash = str(previous.get("root_hash") or "")
        current_hash = str(current.get("root_hash") or "")
        if current_count == previous_count and current_hash == previous_hash:
            sources[source] = {
                "status": "PASSED_UNCHANGED",
                "added_records": 0,
                "previous_root_hash": previous_hash,
                "current_root_hash": current_hash,
            }
            continue
        previous_path = Path(str(previous.get("last_segment") or ""))
        current_path = Path(str(current.get("last_segment") or ""))
        failures: list[str] = []
        added = 0
        first_previous: str | None = None
        previous_last_raw = previous.get("last_known_at")
        previous_last = parse_utc(previous_last_raw) if previous_last_raw else None
        last_before_instance: datetime | None = (
            previous_last
            if previous_last and instance_start and previous_last < instance_start
            else None
        )
        first_after_instance: datetime | None = None
        rolling = previous_hash
        schema_parent = next(
            (parent for parent in previous_path.parents if parent.name.startswith("schema=")),
            None,
        )
        if schema_parent is None or not previous_path.is_file() or not current_path.is_file():
            failures.append("SEGMENT_PATH_MISSING")
        else:
            root = schema_parent.parent
            started = False
            for path in sorted(root.rglob("events.jsonl")):
                if path.resolve() == previous_path.resolve():
                    started = True
                if not started:
                    continue
                start_offset = (
                    int(previous.get("last_segment_size_bytes") or 0)
                    if path.resolve() == previous_path.resolve()
                    else 0
                )
                end_offset = (
                    int(current.get("last_segment_size_bytes") or 0)
                    if path.resolve() == current_path.resolve()
                    else path.stat().st_size
                )
                if end_offset < start_offset:
                    failures.append(f"SEGMENT_SIZE_REGRESSION:{path}")
                    continue
                with path.open("rb") as stream:
                    stream.seek(start_offset)
                    remaining = end_offset - start_offset
                    raw = stream.read(remaining)
                for line in raw.splitlines(keepends=True):
                    if not line.endswith(b"\n"):
                        failures.append(f"NON_DURABLE_PARTIAL_SUFFIX:{path}")
                        continue
                    try:
                        record = dict(json.loads(line))
                    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
                        failures.append(f"INVALID_SUFFIX_JSON:{path}")
                        continue
                    claimed = str(record.pop("record_hash", ""))
                    prior = str(record.get("previous_record_hash") or "")
                    first_previous = first_previous or prior
                    if prior != rolling:
                        failures.append(f"SUFFIX_CHAIN_BREAK:{path}")
                    expected = stable_hash(record, length=64)
                    if claimed != expected:
                        failures.append(f"SUFFIX_HASH_MISMATCH:{path}")
                    rolling = claimed or expected
                    added += 1
                    known_raw = record.get("known_at")
                    if known_raw and instance_start is not None:
                        known_at = parse_utc(known_raw)
                        if known_at < instance_start:
                            last_before_instance = known_at
                        elif first_after_instance is None:
                            first_after_instance = known_at
                if path.resolve() == current_path.resolve():
                    break
        if added != max(0, current_count - previous_count):
            failures.append("SUFFIX_RECORD_COUNT_MISMATCH")
        if rolling != current_hash:
            failures.append("CURRENT_ROOT_HASH_MISMATCH")
        sources[source] = {
            "status": "PASSED" if not failures else "FAILED",
            "added_records": added,
            "previous_record_count": previous_count,
            "current_record_count": current_count,
            "previous_root_hash": previous_hash,
            "first_suffix_previous_hash": first_previous,
            "current_root_hash": current_hash,
            "calculated_root_hash": rolling,
            "failures": failures,
            "last_known_before_instance": (
                utc_iso(last_before_instance) if last_before_instance else None
            ),
            "first_known_after_instance": (
                utc_iso(first_after_instance) if first_after_instance else None
            ),
            "measured_restart_gap_seconds": (
                max(
                    0.0,
                    (first_after_instance - last_before_instance).total_seconds(),
                )
                if last_before_instance and first_after_instance
                else None
            ),
        }
    measured_gaps = [
        float(row["measured_restart_gap_seconds"])
        for row in sources.values()
        if row.get("measured_restart_gap_seconds") is not None
    ]
    return {
        "schema_version": "multi_source_restart_continuity_audit_v1",
        "status": (
            "PASSED"
            if sources and all(row["status"].startswith("PASSED") for row in sources.values())
            else "FAILED"
        ),
        "sources": sources,
        "instance_acquired_at": instance_start_raw,
        "maximum_restart_gap_seconds": max(measured_gaps) if measured_gaps else None,
        "duplicates_persisted": 0,
        "orders_generated": 0,
    }


__all__ = [
    "CollectorAlreadyActive",
    "CollectorLease",
    "CrossVenueAlignmentMonitor",
    "FamilyFreezeManager",
    "HYPOTHESIS_REQUIREMENTS",
    "READINESS_POLICY_VERSION",
    "ReadinessHistoryStore",
    "ReadinessLevel",
    "ReadinessThreshold",
    "RuntimePerformanceMonitor",
    "StorageGrowthMonitor",
    "api_usage_report",
    "assess_readiness",
    "bitvavo_l2_maturation",
    "classify_event",
    "hypothesis_readiness",
    "mexc_derivatives_maturation",
    "process_exists",
    "record_deployment_event",
    "research_readiness_policy_v1",
    "verify_restart_continuity",
]
