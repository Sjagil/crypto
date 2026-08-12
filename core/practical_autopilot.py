"""Permanent practical research, paper and Level-1 canary orchestrator."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp

from config.settings import Settings
from core.autonomous_trading import execute_autonomous_canary_once
from core.daily_profit_target import update_daily_profit_target
from core.generated_strategy_live import (
    deactivate_positive_strategy_live_authority,
    execute_generated_strategy_live_once,
)
from core.generated_strategy_paper import run_generated_paper_once
from core.paper_lifecycle import run_paper_once
from core.practical_governance import (
    GovernancePaths,
    build_portfolio_artifacts,
    build_top50_universe,
    deactivate_live_canary_authority,
    live_canary_authority,
    reclassify_existing_strategies,
)
from data.data_loader import DataLoader
from data.prospective_context import ProspectiveContextCollector
from notifications.telegram import TelegramNotifier
from utils.common import append_jsonl, atomic_write_json, read_json, utc_iso, utc_now

CRITICAL_LIVE_STATUSES = {
    "RECONCILIATION_REQUIRED",
    "UNKNOWN_REMOTE_OPEN_ORDER",
    "DAILY_LOSS_LIMIT",
    "MAXIMUM_DRAWDOWN_LIMIT",
    "KILL_SWITCH_ACTIVE",
    "UNSAFE_API_SCOPE",
}


class PracticalAutopilotLockError(RuntimeError):
    pass


class PracticalAutopilot:
    """One canonical process coordinating bounded research and execution stages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = GovernancePaths(settings.paths.project_root)
        self.paths.ensure()
        # The bounded research engine owns ``autopilot.lock``.  The permanent
        # execution-first supervisor needs an independent process lock so it
        # can invoke that existing engine without deadlocking itself.
        self.lock_path = self.paths.autopilot / "practical_autopilot.lock"
        self.supervisor_lock_path = (
            self.paths.autopilot / "practical_autopilot_supervisor.lock"
        )
        self.status_path = self.paths.autopilot / "status.json"
        self.heartbeat_path = self.paths.autopilot / "heartbeat.json"
        self.supervisor_path = self.paths.autopilot / "supervisor.json"
        self.research_status_path = (
            self.paths.autopilot / "background_research_status.json"
        )
        self.integrated_live_lock_path = (
            self.paths.output / "live" / "autonomous_live.lock"
        )
        self.companion_status_path = (
            self.paths.output / "live" / "companion_services.json"
        )
        self._lock_token: str | None = None
        self._supervisor_lock_token: str | None = None

    def _acquire_supervisor(self) -> None:
        """Own the permanent loop across sleep intervals and process restarts."""

        if self.supervisor_lock_path.is_file():
            try:
                existing = read_json(self.supervisor_lock_path)
            except (OSError, ValueError, TypeError):
                existing = {}
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid and self._pid_alive(existing_pid):
                raise PracticalAutopilotLockError(
                    f"autopilot supervisor active for pid={existing_pid}"
                )
            self.supervisor_lock_path.unlink(missing_ok=True)
        token = uuid.uuid4().hex
        try:
            descriptor = os.open(
                self.supervisor_lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
        except FileExistsError as exc:
            raise PracticalAutopilotLockError("autopilot supervisor lock race") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "token": token,
                    "pid": os.getpid(),
                    "acquired_at": utc_iso(),
                },
                handle,
            )
        self._supervisor_lock_token = token
        atomic_write_json(
            self.supervisor_path,
            {
                "schema_version": "practical_autopilot_supervisor_v1",
                "pid": os.getpid(),
                "started_at": utc_iso(),
                "command": "main.py autopilot run",
                "log": str(self.settings.paths.logs_dir / "practical_autopilot.log"),
                "execution_cycle_seconds": (
                    self.settings.autopilot_execution.execution_cycle_seconds
                ),
                "research_interval_hours": (
                    self.settings.autopilot_execution.min_cycle_interval_hours
                ),
                "process_lock": str(self.supervisor_lock_path),
            },
        )

    def _release_supervisor(self) -> None:
        if not self._supervisor_lock_token or not self.supervisor_lock_path.is_file():
            return
        try:
            payload = read_json(self.supervisor_lock_path)
        except (OSError, ValueError, TypeError):
            return
        if payload.get("token") == self._supervisor_lock_token:
            self.supervisor_lock_path.unlink(missing_ok=True)
        self._supervisor_lock_token = None

    def _acquire(self) -> None:
        if self.lock_path.is_file():
            try:
                existing = read_json(self.lock_path)
                age = time.time() - self.lock_path.stat().st_mtime
            except (OSError, ValueError, TypeError):
                age = 0.0
                existing = {}
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid and not self._pid_alive(existing_pid):
                self.lock_path.unlink(missing_ok=True)
                existing = {}
                age = float("inf")
            stale_after = max(
                900.0,
                self.settings.autopilot_execution.max_runtime_minutes_per_cycle * 60.0
                + 300.0,
            )
            if self.lock_path.is_file() and age <= stale_after:
                raise PracticalAutopilotLockError(
                    f"autopilot lock active for pid={existing.get('pid')}"
                )
            self.lock_path.unlink(missing_ok=True)
        token = uuid.uuid4().hex
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
        except FileExistsError as exc:
            raise PracticalAutopilotLockError("autopilot lock race") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "token": token,
                    "pid": os.getpid(),
                    "acquired_at": utc_iso(),
                },
                handle,
            )
        self._lock_token = token

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle,
                    ctypes.byref(exit_code),
                ):
                    return False
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _release(self) -> None:
        if not self._lock_token or not self.lock_path.is_file():
            return
        try:
            payload = read_json(self.lock_path)
        except (OSError, ValueError, TypeError):
            return
        if payload.get("token") == self._lock_token:
            self.lock_path.unlink(missing_ok=True)
        self._lock_token = None

    def _heartbeat(self, state: str, **extra: Any) -> None:
        atomic_write_json(
            self.heartbeat_path,
            {
                "schema_version": "practical_autopilot_heartbeat_v1",
                "heartbeat_at": utc_iso(),
                "state": state,
                "pid": os.getpid(),
                **extra,
            },
        )

    async def _bitvavo_markets(self) -> set[str]:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://api.bitvavo.com/v2/markets") as response:
                if response.status >= 400:
                    raise RuntimeError(f"BITVAVO_MARKETS_HTTP_{response.status}")
                payload = await response.json(content_type=None)
        if not isinstance(payload, list):
            raise RuntimeError("BITVAVO_MARKETS_INVALID")
        return {
            str(row.get("market") or "").upper()
            for row in payload
            if isinstance(row, dict)
            and str(row.get("market") or "").upper().endswith("-EUR")
            and str(row.get("status") or "trading").casefold() in {"trading", "active"}
        }

    def _research_due(self) -> bool:
        lab_heartbeat_path = (
            self.settings.paths.lab_dir / "state" / "heartbeat.json"
        )
        if lab_heartbeat_path.is_file():
            lab_heartbeat = dict(read_json(lab_heartbeat_path))
            lab_pid = int(lab_heartbeat.get("pid") or 0)
            if (
                lab_pid
                and lab_pid != os.getpid()
                and self._pid_alive(lab_pid)
            ):
                return False
        background = (
            dict(read_json(self.research_status_path))
            if self.research_status_path.is_file()
            else {}
        )
        background_pid = int(background.get("supervisor_pid") or 0)
        if (
            background.get("status") == "RUNNING"
            and background_pid
            and background_pid != os.getpid()
            and self._pid_alive(background_pid)
        ):
            return False
        completed_bundle_paths = (
            self.settings.paths.lab_dir
            / "reports"
            / "lower_timeframe_mtf_v1_report.json",
            self.settings.paths.lab_dir
            / "reports"
            / "owned_asset_high_sample_v1_report.json",
            self.settings.paths.output_dir
            / "hmm"
            / "reports"
            / "hmm_regime_campaign_v1.json",
            self.settings.paths.lab_dir
            / "reports"
            / "adaptive_crypto_intraday_v1.json",
            self.settings.paths.lab_dir
            / "reports"
            / "classical_strategy_factory_v1_report.json",
        )
        completed_bundle_at = None
        if all(path.is_file() for path in completed_bundle_paths):
            completed_bundle_at = min(
                datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=UTC,
                )
                for path in completed_bundle_paths
            )
        payload = (
            dict(read_json(self.status_path))
            if self.status_path.is_file()
            else {}
        )
        stored = payload.get("last_research_cycle_at")
        parsed = (
            datetime.fromisoformat(str(stored).replace("Z", "+00:00"))
            if stored
            else None
        )
        last_attempt = background.get("finished_at")
        parsed_attempt = (
            datetime.fromisoformat(str(last_attempt).replace("Z", "+00:00"))
            if last_attempt
            else None
        )
        effective = max(
            (
                value
                for value in (parsed, completed_bundle_at, parsed_attempt)
                if value is not None
            ),
            default=None,
        )
        if effective is None:
            return True
        return (
            utc_now() - effective
        ).total_seconds() >= self.settings.autopilot_execution.min_cycle_interval_hours * 3600

    def _run_existing_research(self) -> dict[str, Any]:
        main = str(self.settings.paths.project_root / "main.py")
        commands: list[list[str]] = [
            [
                sys.executable,
                main,
                "lab",
                "campaign",
                "autopilot",
                "--run-research",
                "--refresh-data",
            ],
        ]

        def stage_due(path: os.PathLike[str]) -> bool:
            candidate = os.fspath(path)
            if not os.path.isfile(candidate):
                return True
            age_seconds = time.time() - os.path.getmtime(candidate)
            return (
                age_seconds
                >= self.settings.autopilot_execution.min_cycle_interval_hours
                * 3600
            )

        def add_named_campaign(
            name: str,
            report_path: os.PathLike[str],
            *,
            workers: int = 2,
        ) -> None:
            if not stage_due(report_path):
                return
            commands.extend(
                [
                    [
                        sys.executable,
                        main,
                        "lab",
                        "campaign",
                        "run",
                        "--name",
                        name,
                        "--workers",
                        str(max(1, workers)),
                        "--max-trials",
                        str(
                            min(
                                12,
                                self.settings.autopilot_execution.max_parameter_variants_per_strategy,
                            )
                        ),
                        "--yes",
                    ],
                    [
                        sys.executable,
                        main,
                        "lab",
                        "campaign",
                        "report",
                        "--name",
                        name,
                    ],
                ]
            )

        reports = self.settings.paths.lab_dir / "reports"
        overlay_report = reports / "15m_entry_overlay_validation_v1.json"
        if stage_due(overlay_report):
            commands.append(
                [
                    sys.executable,
                    main,
                    "multi-timeframe",
                    "validate-15m",
                ]
            )
        limit_overlay_report = reports / "mtf_15m_limit_overlay_v1.json"
        if stage_due(limit_overlay_report):
            commands.append(
                [
                    sys.executable,
                    main,
                    "multi-timeframe",
                    "validate-limit-overlay",
                ]
            )
        add_named_campaign(
            "owned-asset-high-sample-v1",
            reports / "owned_asset_high_sample_v1_report.json",
            # Full per-asset 1h history for seven markets creates large
            # feature frames. One worker keeps peak memory below the lab
            # budget; live execution cycles continue independently.
            workers=1,
        )
        add_named_campaign(
            "lower-timeframe-mtf-v1",
            reports / "lower_timeframe_mtf_v1_report.json",
            # Native 15m full-history frames are larger than the 1h/4h owned
            # campaign. Run them serially so research cannot starve the
            # independent five-minute live execution loop.
            workers=1,
        )
        add_named_campaign(
            "long-history-intraday-v1",
            reports / "long_history_intraday_v1_report.json",
            # BTC/ETH/LINK have more than seven years of common 1h/4h data.
            # This promotion-focused run keeps the explicit 100-trade gate.
            workers=1,
        )
        hmm_report = (
            self.settings.paths.output_dir
            / "hmm"
            / "reports"
            / "hmm_regime_campaign_v1.json"
        )
        if stage_due(hmm_report):
            commands.append(
                [
                    sys.executable,
                    main,
                    "hmm",
                    "compare",
                ]
            )
        adaptive_report = reports / "adaptive_crypto_intraday_v1.json"
        if stage_due(adaptive_report):
            commands.append(
                [
                    sys.executable,
                    main,
                    "lab",
                    "campaign",
                    "run",
                    "--name",
                    "adaptive-crypto-intraday-v1",
                    "--yes",
                ]
            )
        classical_report = reports / "classical_strategy_factory_v1_report.json"
        if stage_due(classical_report):
            commands.extend(
                [
                    [
                        sys.executable,
                        main,
                        "lab",
                        "campaign",
                        "plan",
                        "--name",
                        "classical-strategy-factory-v1",
                        "--factory-trials",
                        "2000",
                    ],
                    [
                        sys.executable,
                        main,
                        "lab",
                        "campaign",
                        "run",
                        "--name",
                        "classical-strategy-factory-v1",
                        "--factory-trials",
                        "2000",
                        "--workers",
                        "2",
                        "--max-trials",
                        str(
                            min(
                                12,
                                self.settings.autopilot_execution.max_parameter_variants_per_strategy,
                            )
                        ),
                        "--yes",
                    ],
                    [
                        sys.executable,
                        main,
                        "lab",
                        "campaign",
                        "report",
                        "--name",
                        "classical-strategy-factory-v1",
                    ],
                ]
            )
        completed_stages: list[dict[str, Any]] = []
        deadline = time.monotonic() + (
            self.settings.autopilot_execution.max_runtime_minutes_per_cycle * 60
        )
        for stage_index, command in enumerate(commands):
            remaining = max(1.0, deadline - time.monotonic())
            allowed_return_codes = {0, 3} if stage_index == 0 else {0}
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.settings.paths.project_root,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    exc.stdout.decode("utf-8", errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else str(exc.stdout or "")
                )
                stderr = (
                    exc.stderr.decode("utf-8", errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else str(exc.stderr or "")
                )
                completed_stages.append(
                    {
                        "command": " ".join(command[2:]),
                        "return_code": 124,
                        "allowed_return_codes": sorted(allowed_return_codes),
                        "degraded_but_continuable": True,
                        "timed_out": True,
                        "timeout_seconds": remaining,
                        "stdout_tail": stdout[-2_000:],
                        "stderr_tail": stderr[-2_000:],
                    }
                )
                break
            completed_stages.append(
                {
                    "command": " ".join(command[2:]),
                    "return_code": completed.returncode,
                    "allowed_return_codes": sorted(allowed_return_codes),
                    "degraded_but_continuable": (
                        completed.returncode == 3
                        and completed.returncode in allowed_return_codes
                    ),
                    "stdout_tail": completed.stdout[-2_000:],
                    "stderr_tail": completed.stderr[-2_000:],
                }
            )
            if completed.returncode not in allowed_return_codes:
                break
        passed = len(completed_stages) == len(commands) and all(
            stage["return_code"] in stage["allowed_return_codes"]
            for stage in completed_stages
        )
        return {
            "status": "PASSED" if passed else "FAILED",
            "return_code": completed_stages[-1]["return_code"],
            "reason_code": (
                None
                if passed
                else "RESEARCH_CYCLE_TIMEOUT"
                if completed_stages[-1].get("timed_out")
                else "RESEARCH_STAGE_FAILED"
            ),
            "stages": completed_stages,
            "stdout_tail": completed_stages[-1]["stdout_tail"],
            "stderr_tail": completed_stages[-1]["stderr_tail"],
            "lower_timeframe_mtf_campaign_enabled": True,
            "timeframes": ["15m", "1h", "4h"],
            "owned_asset_high_sample_campaign_enabled": True,
            "owned_asset_markets": [
                "BTC-EUR",
                "ETH-EUR",
                "SOL-EUR",
                "TAO-EUR",
                "ICP-EUR",
                "NPC-EUR",
                "S-EUR",
            ],
            "stationary_bootstrap_monte_carlo": True,
            "dirichlet_time_concentration_stress": True,
            "strategy_charts": True,
            "classical_strategy_factory_enabled": True,
            "classical_strategy_factory_preregistered_dna": 2_000,
            "classical_strategy_factory_families": 51,
            "classical_strategy_factory_timeframe_priority": [
                "15m",
                "1h",
                "4h",
                "1d",
                "1W",
            ],
            "causal_hmm_regime_comparison": True,
            "hmm_observer_only": True,
            "orders_generated": 0,
        }

    def _record_background_research(
        self,
        *,
        status: str,
        started_at: datetime,
        result: dict[str, Any] | None = None,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        finished = status in {"PASSED", "FAILED"}
        payload = {
            "schema_version": "background_research_status_v1",
            "status": status,
            "started_at": utc_iso(started_at),
            "finished_at": utc_iso() if finished else None,
            "supervisor_pid": os.getpid(),
            "execution_cycles_continue": True,
            "orders_generated": 0,
            "orders_submitted": 0,
            "reason_code": reason_code,
            "result": result,
        }
        atomic_write_json(self.research_status_path, payload)
        return payload

    def _merge_background_research_status(
        self,
        cycle_status: dict[str, Any],
        background: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(cycle_status)
        stages = dict(merged.get("stages") or {})
        stages["research"] = background
        merged["stages"] = stages
        merged["research_subprocess_active"] = background.get("status") == "RUNNING"
        if background.get("status") == "PASSED":
            merged["last_research_cycle_at"] = background.get("finished_at")
        atomic_write_json(self.status_path, merged)
        return merged

    def _flush_telegram(self) -> dict[str, Any]:
        """Deliver queued lifecycle/research messages without affecting trading."""
        try:
            notifier = TelegramNotifier(
                self.settings.telegram,
                output_directory=self.settings.paths.output_dir / "notifications",
                allowed_markets=self.settings.operational.markets,
            )
            return notifier.flush()
        except Exception as exc:
            return {
                "status": "FAILED_ISOLATED",
                "reason_code": f"TELEGRAM_{type(exc).__name__.upper()}",
                "orders_generated": 0,
                "orders_submitted": 0,
            }

    async def _refresh_prospective_context(self) -> dict[str, Any]:
        """Refresh immutable CMC/MEXC facts before rebuilding the top-50 universe.

        ``market_data.symbols`` intentionally defines the derivatives context
        set.  It contains the liquid core context markets, while
        ``operational.markets`` can also contain explicitly approved spot-only
        exceptions such as TAO.  A missing derivatives venue for such an
        exception must not prevent a fresh CoinMarketCap universe snapshot.
        """

        collector = ProspectiveContextCollector(
            checkpoint_path=(
                self.settings.paths.checkpoints_dir
                / "prospective_context_hourly.json"
            ),
            snapshot_directory=(
                self.settings.paths.context_data_dir / "prospective_hourly"
            ),
        )
        result = await collector.collect(
            loader=DataLoader(self.settings),
            markets=tuple(self.settings.market_data.symbols),
            observed_at=utc_now(),
        )
        completed_epoch = result.get("last_completed_epoch")
        return {
            "status": result.get("status"),
            "reason_code": result.get("reason_code"),
            "last_completed_epoch": (
                utc_iso(completed_epoch)
                if isinstance(completed_epoch, datetime)
                else str(completed_epoch)
                if completed_epoch is not None
                else None
            ),
            "snapshot_path": result.get("snapshot_path"),
            "ranking_count": result.get("ranking_count"),
            "derivatives_count": result.get("derivatives_count"),
            "received_rankings": result.get("received_rankings"),
            "received_derivatives": result.get("received_derivatives"),
            "failures": list(result.get("failures") or []),
            "context_markets": list(self.settings.market_data.symbols),
            "universe_expansion_allowed": result.get("status")
            in {"PASSED", "UP_TO_DATE"},
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    async def run_once(
        self,
        *,
        run_research: bool = True,
        allow_live_new_entries: bool = True,
    ) -> dict[str, Any]:
        self._acquire()
        started = utc_now()
        stages: dict[str, Any] = {}
        failures: list[str] = []
        try:
            self._heartbeat("RUNNING", stage="provider_health")
            try:
                markets = await self._bitvavo_markets()
                stages["provider_health"] = {
                    "status": "PASSED",
                    "bitvavo_eur_markets": len(markets),
                }
            except Exception as exc:
                markets = set(self.settings.market_data.symbols)
                failures.append(f"PROVIDER_MARKETS:{type(exc).__name__}")
                stages["provider_health"] = {
                    "status": "DEGRADED",
                    "reason_code": type(exc).__name__,
                }

            self._heartbeat("RUNNING", stage="prospective_context")
            try:
                context = await self._refresh_prospective_context()
                context_status = str(context.get("status"))
                stages["prospective_context"] = {
                    **context,
                    "status": (
                        "PASSED"
                        if context_status in {"PASSED", "UP_TO_DATE"}
                        else "DEGRADED"
                    ),
                    "collector_status": context_status,
                }
                if context_status not in {"PASSED", "UP_TO_DATE"}:
                    failures.append(
                        "PROSPECTIVE_CONTEXT:"
                        + str(
                            context.get("reason_code")
                            or context_status
                            or "UNKNOWN"
                        )
                    )
            except Exception as exc:
                failures.append(
                    f"PROSPECTIVE_CONTEXT:{type(exc).__name__}"
                )
                stages["prospective_context"] = {
                    "status": "DEGRADED",
                    "collector_status": "FAILED_ISOLATED",
                    "reason_code": type(exc).__name__,
                    "universe_expansion_allowed": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }

            self._heartbeat("RUNNING", stage="top50_universe")
            try:
                universe = build_top50_universe(
                    self.settings.paths.project_root,
                    self.settings,
                    venue_markets=markets,
                )
                stages["universe"] = {
                    "status": "PASSED",
                    "top50_count": universe["count"],
                    "available_markets": universe["available_markets"],
                    "execution_eligible": universe["execution_eligible"],
                }
            except Exception as exc:
                failures.append(f"UNIVERSE:{type(exc).__name__}")
                stages["universe"] = {
                    "status": "FAILED",
                    "reason_code": type(exc).__name__,
                }

            self._heartbeat("RUNNING", stage="governance")
            governance = reclassify_existing_strategies(
                self.settings.paths.project_root,
                self.settings,
            )
            stages["governance"] = {
                "status": "PASSED",
                "research_positive": governance["research_positive"],
                "paper_active": governance["paper_active"],
                "live_canary_eligible": governance["live_canary_eligible"],
                "live_canary_active": governance["live_canary_active"],
            }

            self._heartbeat("RUNNING", stage="paper")
            from core.autonomous_trading import (
                build_fresh_autonomous_control_plane,
            )

            fresh_control = await build_fresh_autonomous_control_plane(
                self.settings,
            )
            paper = run_paper_once(
                self.settings,
                control_plane=fresh_control,
            )
            try:
                generated_paper = await run_generated_paper_once(self.settings)
            except Exception as exc:
                generated_paper = {
                    "status": "FAILED_ISOLATED",
                    "last_reason": f"GENERATED_PAPER_{type(exc).__name__.upper()}",
                    "orders_generated_this_cycle": 0,
                    "real_exchange_requests": 0,
                }
                failures.append("GENERATED_PAPER_STAGE_FAILED")
            stages["paper"] = {
                "status": paper.get("cycle_status"),
                "reason_code": paper.get("reason_code"),
                "paper_orders": paper.get("paper_orders"),
                "paper_fills": paper.get("paper_fills"),
                "open_positions": paper.get("open_positions"),
                "generated": {
                    "status": generated_paper.get("status"),
                    "reason_code": generated_paper.get("last_reason"),
                    "paper_active_candidates": generated_paper.get(
                        "paper_active_candidates"
                    ),
                    "paper_orders": generated_paper.get("paper_orders_placed"),
                    "paper_fills": generated_paper.get("paper_fills"),
                    "open_positions": generated_paper.get("open_positions"),
                    "orders_generated_this_cycle": generated_paper.get(
                        "orders_generated_this_cycle",
                        0,
                    ),
                    "auto_live_promotion": False,
                    "real_exchange_requests": 0,
                },
                "real_exchange_requests": 0,
            }

            self._heartbeat("RUNNING", stage="live_canary")
            authorized, _, authority_failures = live_canary_authority(
                self.settings.paths.project_root
            )
            live = await execute_autonomous_canary_once(
                self.settings,
                submit=authorized,
                allow_new_entry=allow_live_new_entries,
            )
            stages["live_canary"] = {
                "status": live.get("status"),
                "cycle_status": live.get("cycle_status"),
                "reason_code": live.get("reason_code"),
                "natural_signal": live.get("natural_signal"),
                "signal_diagnostics": live.get("signal_diagnostics"),
                "entry_liquidity": live.get("entry_liquidity"),
                "canary_limits": live.get("canary_limits"),
                "current_position": live.get("current_position"),
                "authorized": authorized,
                "authority_failures": authority_failures,
                "orders_generated": live.get("orders_generated", 0),
                "orders_submitted": live.get("orders_submitted", 0),
                "private_exchange_requests": live.get("private_exchange_requests", 0),
            }
            critical = (
                str(live.get("status")) in CRITICAL_LIVE_STATUSES
                or str(live.get("reason_code")) in CRITICAL_LIVE_STATUSES
            )
            if critical:
                deactivate_live_canary_authority(
                    self.settings.paths.project_root,
                    reason=str(live.get("reason_code") or live.get("status")),
                )
                failures.append("LIVE_CANARY_CRITICAL_AUTO_DEACTIVATED")

            self._heartbeat(
                "RUNNING",
                stage="generated_strategy_live_portfolio",
            )
            try:
                generated_live = await execute_generated_strategy_live_once(
                    self.settings,
                    submit=True,
                    allow_new_entry=bool(
                        allow_live_new_entries
                        and not live.get("orders_submitted")
                    ),
                )
            except Exception as exc:
                generated_live = {
                    "status": "FAILED_ISOLATED",
                    "last_reason": (
                        f"GENERATED_LIVE_{type(exc).__name__.upper()}"
                    ),
                    "orders_generated_this_cycle": 0,
                    "orders_submitted_this_cycle": 0,
                }
                failures.append("GENERATED_LIVE_STAGE_FAILED")
            stages["generated_strategy_live_portfolio"] = {
                "status": generated_live.get("status"),
                "reason_code": generated_live.get("last_reason"),
                "positions": generated_live.get("positions"),
                "selected_entry": generated_live.get("selected_entry"),
                "ranked_natural_entries": generated_live.get(
                    "ranked_natural_entries"
                ),
                "material_wallet_position_count": generated_live.get(
                    "material_wallet_position_count"
                ),
                "material_wallet_markets": generated_live.get(
                    "material_wallet_markets"
                ),
                "entry_liquidity": generated_live.get("entry_liquidity"),
                "orders_generated": generated_live.get(
                    "orders_generated_this_cycle",
                    0,
                ),
                "orders_submitted": generated_live.get(
                    "orders_submitted_this_cycle",
                    0,
                ),
            }
            if str(generated_live.get("status")) in {
                "RECONCILIATION_BLOCKED",
                "AUTHORITY_BLOCKED",
            }:
                deactivate_positive_strategy_live_authority(
                    self.settings,
                    reason=str(
                        generated_live.get("last_reason")
                        or generated_live.get("status")
                    ),
                )
                failures.append(
                    "GENERATED_LIVE_CRITICAL_AUTO_DEACTIVATED"
                )

            self._heartbeat("RUNNING", stage="portfolio")
            stages["portfolio"] = build_portfolio_artifacts(
                self.settings.paths.project_root,
                self.settings,
                governance,
            )
            latest_health_path = (
                self.settings.paths.output_dir
                / "operations"
                / "live_account_health.json"
            )
            latest_health = (
                dict(read_json(latest_health_path))
                if latest_health_path.is_file()
                else {}
            )
            estimated_equity = (
                latest_health.get("account", {})
                .get("portfolio_valuation", {})
                .get("estimated_total_equity_eur")
            )
            stages["daily_profit_target"] = update_daily_profit_target(
                self.settings,
                estimated_equity_eur=estimated_equity,
                valuation_status=str(
                    latest_health.get("account", {})
                    .get("portfolio_valuation", {})
                    .get("status")
                    or "VALUATION_PENDING"
                ),
            )

            research_result: dict[str, Any] = {
                "status": "NOT_DUE_OR_DISABLED",
                "orders_generated": 0,
            }
            research_executed = bool(
                run_research
                and self.settings.autopilot_execution.enabled
                and self._research_due()
            )
            if research_executed:
                self._heartbeat("RUNNING", stage="research")
                research_task = asyncio.create_task(
                    asyncio.to_thread(self._run_existing_research)
                )
                while not research_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(research_task),
                            timeout=30.0,
                        )
                    except TimeoutError:
                        self._heartbeat(
                            "RUNNING",
                            stage="research",
                            research_subprocess_active=True,
                        )
                research_result = await research_task
                if research_result["status"] != "PASSED":
                    failures.append("RESEARCH_STAGE_FAILED")
            stages["research"] = research_result

            # Research and lifecycle stages may enqueue summaries or alerts.
            # Delivery is deliberately last and isolated: a Telegram outage
            # must not change signal generation, paper state, or live orders.
            self._heartbeat("RUNNING", stage="telegram")
            stages["telegram"] = self._flush_telegram()

            finished = utc_now()
            status = {
                "schema_version": "practical_autopilot_status_v1",
                "status": "PASSED" if not failures else "DEGRADED",
                "started_at": utc_iso(started),
                "finished_at": utc_iso(finished),
                "duration_seconds": (finished - started).total_seconds(),
                "failures": failures,
                "stages": stages,
                "last_research_cycle_at": (
                    utc_iso(finished)
                    if research_executed and research_result["status"] == "PASSED"
                    else (
                        read_json(self.status_path).get("last_research_cycle_at")
                        if self.status_path.is_file()
                        else None
                    )
                ),
                "autopilot_auto_paper_promotion": True,
                "autopilot_auto_live_promotion": False,
                "live_canary_operator_authorized": authorized,
                "orders_generated": int(
                    stages["live_canary"].get("orders_generated") or 0
                ),
                "orders_submitted": int(
                    stages["live_canary"].get("orders_submitted") or 0
                ),
                "paper_orders_generated": int(
                    paper.get("orders_generated_this_cycle") or 0
                )
                + int(
                    generated_paper.get("orders_generated_this_cycle") or 0
                ),
            }
            atomic_write_json(self.status_path, status)
            append_jsonl(self.paths.autopilot / "research_cycles.jsonl", status)
            (self.paths.autopilot / "demotions.jsonl").touch(exist_ok=True)
            self._heartbeat("IDLE", last_cycle_status=status["status"])
            return status
        except Exception as exc:
            failed = {
                "schema_version": "practical_autopilot_status_v1",
                "status": "FAILED",
                "started_at": utc_iso(started),
                "failed_at": utc_iso(),
                "reason_code": type(exc).__name__,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
            atomic_write_json(self.status_path, failed)
            append_jsonl(self.paths.autopilot / "research_cycles.jsonl", failed)
            self._heartbeat("FAILED", reason_code=type(exc).__name__)
            raise
        finally:
            self._release()

    async def run(self) -> None:
        interval = float(
            self.settings.autopilot_execution.execution_cycle_seconds
        )
        research_task: asyncio.Task[dict[str, Any]] | None = None
        research_started_at: datetime | None = None
        background_status = (
            dict(read_json(self.research_status_path))
            if self.research_status_path.is_file()
            else {
                "schema_version": "background_research_status_v1",
                "status": "NOT_STARTED",
                "execution_cycles_continue": True,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )
        if (
            background_status.get("status") == "RUNNING"
            and int(background_status.get("supervisor_pid") or 0) != os.getpid()
        ):
            background_status = {
                **background_status,
                "status": "INTERRUPTED_RESTART_RECOVERABLE",
                "reason_code": "SUPERVISOR_RESTARTED",
                "execution_cycles_continue": True,
            }
            atomic_write_json(self.research_status_path, background_status)
        self._acquire_supervisor()
        try:
            while True:
                if research_task is not None and research_task.done():
                    assert research_started_at is not None
                    try:
                        research_result = research_task.result()
                        background_status = self._record_background_research(
                            status=(
                                "PASSED"
                                if research_result.get("status") == "PASSED"
                                else "FAILED"
                            ),
                            started_at=research_started_at,
                            result=research_result,
                            reason_code=(
                                None
                                if research_result.get("status") == "PASSED"
                                else "RESEARCH_STAGE_FAILED"
                            ),
                        )
                    except Exception as exc:
                        background_status = self._record_background_research(
                            status="FAILED",
                            started_at=research_started_at,
                            reason_code=type(exc).__name__,
                        )
                    research_task = None
                    research_started_at = None

                cycle_status = await self.run_once(run_research=False)

                if (
                    research_task is None
                    and self.settings.autopilot_execution.enabled
                    and self._research_due()
                ):
                    research_started_at = utc_now()
                    background_status = self._record_background_research(
                        status="RUNNING",
                        started_at=research_started_at,
                    )
                    research_task = asyncio.create_task(
                        asyncio.to_thread(self._run_existing_research)
                    )
                elif research_task is not None:
                    background_status = {
                        **background_status,
                        "status": "RUNNING",
                        "execution_cycles_continue": True,
                    }

                cycle_status = self._merge_background_research_status(
                    cycle_status,
                    background_status,
                )
                self._heartbeat(
                    "SLEEPING",
                    last_cycle_status=cycle_status.get("status"),
                    next_cycle_at=utc_iso(
                        utc_now() + timedelta(seconds=interval)
                    ),
                    research_subprocess_active=(research_task is not None),
                    background_research_status=background_status.get("status"),
                    execution_cycle_seconds=interval,
                    research_interval_hours=(
                        self.settings.autopilot_execution.min_cycle_interval_hours
                    ),
                )
                await asyncio.sleep(interval)
        finally:
            if research_task is not None and not research_task.done():
                research_task.cancel()
            self._release_supervisor()

    def status(self) -> dict[str, Any]:
        status = (
            dict(read_json(self.status_path))
            if self.status_path.is_file()
            else {"status": "NOT_RUN"}
        )
        heartbeat = (
            dict(read_json(self.heartbeat_path))
            if self.heartbeat_path.is_file()
            else {"state": "NOT_STARTED"}
        )
        supervisor = (
            dict(read_json(self.supervisor_path))
            if self.supervisor_path.is_file()
            else {}
        )
        supervisor_pid = int(supervisor.get("pid") or 0)
        practical_supervisor_running = self._pid_alive(supervisor_pid)
        integrated_live = (
            dict(read_json(self.integrated_live_lock_path))
            if self.integrated_live_lock_path.is_file()
            else {}
        )
        integrated_live_pid = int(integrated_live.get("pid") or 0)
        integrated_live_running = self._pid_alive(integrated_live_pid)
        companion_status = (
            dict(read_json(self.companion_status_path))
            if self.companion_status_path.is_file()
            else {}
        )
        simple_lab = dict(
            (companion_status.get("services") or {}).get("simple_lab") or {}
        )
        simple_lab_pid = int(simple_lab.get("pid") or 0)
        continuous_research_active = (
            str(simple_lab.get("status") or "").upper()
            in {"RUNNING", "STARTING"}
            and self._pid_alive(simple_lab_pid)
        )
        background_research = (
            dict(read_json(self.research_status_path))
            if self.research_status_path.is_file()
            else {"status": "NOT_STARTED"}
        )
        background_pid = int(background_research.get("supervisor_pid") or 0)
        if (
            background_research.get("status") == "RUNNING"
            and (
                not background_pid
                or background_pid != supervisor_pid
                or not self._pid_alive(background_pid)
            )
        ):
            background_research = {
                **background_research,
                "status": "INTERRUPTED_RESTART_RECOVERABLE",
                "reason_code": "SUPERVISOR_PROCESS_NOT_RUNNING",
            }
        batch_research_active = (
            background_research.get("status") == "RUNNING"
            and background_pid == supervisor_pid
            and practical_supervisor_running
        )
        return {
            **status,
            "heartbeat": heartbeat,
            "lock_active": self.lock_path.is_file(),
            "supervisor_lock_active": self.supervisor_lock_path.is_file(),
            "supervisor": supervisor,
            "supervisor_running": (
                practical_supervisor_running or integrated_live_running
            ),
            "supervisor_mode": (
                "PRACTICAL_AUTOPILOT"
                if practical_supervisor_running
                else "AUTONOMOUS_LIVE_INTEGRATED"
                if integrated_live_running
                else "NOT_RUNNING"
            ),
            "integrated_live_supervisor": {
                "pid": integrated_live_pid or None,
                "running": integrated_live_running,
            },
            "enabled": self.settings.autopilot_execution.enabled,
            "execution_cycle_seconds": (
                self.settings.autopilot_execution.execution_cycle_seconds
            ),
            "research_interval_hours": (
                self.settings.autopilot_execution.min_cycle_interval_hours
            ),
            "auto_paper_promotion": self.settings.autopilot_execution.auto_paper_promotion,
            "auto_live_promotion": self.settings.autopilot_execution.auto_live_promotion,
            "background_research": background_research,
            "continuous_research": {
                "status": (
                    "RUNNING"
                    if continuous_research_active
                    else str(simple_lab.get("status") or "NOT_RUNNING")
                ),
                "pid": simple_lab_pid or None,
                "running": continuous_research_active,
            },
            "research_subprocess_active": (
                batch_research_active or continuous_research_active
            ),
        }


__all__ = ["PracticalAutopilot", "PracticalAutopilotLockError"]
