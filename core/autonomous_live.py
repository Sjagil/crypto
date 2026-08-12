"""Single production supervisor for the dynamic approved spot universe."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from config.settings import Settings, normalize_timeframe
from core.contracts import ExecutionBlocked, ReconciliationRequired
from core.event_driven_live import (
    MAXIMUM_ORDER_EUR,
    deactivate_playbook_live,
    execute_event_driven_live_once,
    execution_block_reason_code,
    execution_block_requires_authority_deactivation,
    is_playbook_opportunity_authorized,
)
from core.event_driven_paper import (
    load_canonical_entry_economics_gate,
    run_event_driven_paper_once,
)
from core.event_driven_playbooks import (
    OpportunityLifecycleLedger,
    OpportunityState,
    build_event_driven_opportunities,
)
from core.execution_authority import build_execution_authority_matrix
from core.execution_dispositions import build_entry_ready_dispositions
from core.execution_evidence import build_execution_evidence_layers
from core.live_asset_preflight import live_account_health, live_asset_preflight
from core.live_strategy_accounting import rebuild_live_strategy_accounting
from core.market_exceptions import load_execution_market_exceptions
from core.practical_autopilot import PracticalAutopilot
from core.practical_governance import (
    governance_status,
    reclassify_existing_strategies,
)
from core.strategy_evidence_watch import build_strategy_evidence_watch
from core.swing_trading import execution_timeframe_allowed
from data.bitvavo_private_stream import BitvavoPrivateAccountStream
from data.data_loader import ContinuousDataService, DataLoader
from data.orderflow_recorder import (
    HashChainedOrderflowLedger,
    ProspectiveOrderflowRecorder,
)
from data.websocket_manager import WebSocketManager
from execution.execution import DurableLedger
from execution.position_tracker import PositionTracker
from execution.state_migration import (
    build_execution_divergence_report,
    execution_state_migration_status,
)
from notifications.telegram import TelegramNotifier
from reporting.canonical_economics import canonical_family
from reporting.telegram_signal_evidence import build_telegram_signal_evidence
from utils.common import append_jsonl, atomic_write_json, read_json, stable_hash, utc_iso

LAUNCH_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
    "NPC-EUR",
    "ADA-EUR",
)
APPROVAL_PHRASE = "LIVE_SPOT_CONFIRMED"
EVENT_STREAMS = (
    "signals",
    "orders",
    "fills",
    "positions",
    "pnl",
    "risk",
    "strategy_lifecycle",
    "research",
    "health",
    "errors",
)


class AutonomousLiveLockError(RuntimeError):
    pass


def amsterdam_macro_slot(
    observed_at: datetime | None = None,
) -> str | None:
    """Return the active scheduled Telegram macro slot, if any."""

    now_local = (observed_at or datetime.now(UTC)).astimezone(
        ZoneInfo("Europe/Amsterdam")
    )
    if now_local.hour not in {8, 23}:
        return None
    return f"{now_local.date().isoformat()} {now_local.hour:02d}:00"


def resolved_live_markets(settings: Settings) -> tuple[str, ...]:
    """Freeze one sanitized market set for monitoring and approved execution.

    Explicit market exceptions add data and monitoring coverage only.  Their
    registry contract still requires a separately approved strategy DNA and a
    natural signal before any order can be created.
    """

    path = (
        settings.paths.output_dir
        / "governance"
        / "live_universe.json"
    )
    artifact = dict(read_json(path)) if path.is_file() else {}
    dynamic_source = (
        artifact.get("selected_markets")
        if artifact.get("status") == "READY"
        and int(artifact.get("live_eligible_count") or 0) >= 5
        else settings.autonomous_live.markets
    )
    source = [
        *(dynamic_source or []),
        *load_execution_market_exceptions(settings),
    ]
    markets = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-")
            for market in source or []
            if str(market).strip().upper().endswith("-EUR")
            and settings.shariah.eligibility(
                str(market).strip().upper().replace("/", "-")
            ).status.value
            == "ALLOWED"
        )
    )
    if len(markets) < 5:
        raise ValueError("dynamic live universe has fewer than five markets")
    return markets


def resolved_orderflow_markets(
    settings: Settings,
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    """Return up to 25 tracked EUR spot markets without granting authority."""

    path = settings.paths.output_dir / "universe" / "tiered_trading_universe.json"
    artifact = dict(read_json(path)) if path.is_file() else {}
    top50_path = settings.paths.output_dir / "universe" / "top50_eligibility.json"
    top50 = dict(read_json(top50_path)) if top50_path.is_file() else {}
    top20_live = [
        str(row.get("eur_spot_market") or "")
        for row in top50.get("rows") or []
        if int(row.get("rank") or 10_000) <= 20
        and row.get("execution_eligibility") == "LIVE_ELIGIBLE"
        and row.get("eur_spot_market")
    ]
    source = [
        *top20_live,
        *(artifact.get("shadow_markets") or list(fallback)),
        *settings.autonomous_live.monitor_only_markets,
        *load_execution_market_exceptions(settings),
    ]
    markets = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-")
            for market in source
            if str(market).strip().upper().endswith("-EUR")
        )
    )
    return markets[:25] or tuple(fallback)


def resolved_ticker_tracking_markets(
    settings: Settings,
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    """Cheap ticker coverage for every available top-50/context EUR market."""

    tiered_path = (
        settings.paths.output_dir / "universe" / "tiered_trading_universe.json"
    )
    top50_path = settings.paths.output_dir / "universe" / "top50_current.json"
    tiered = dict(read_json(tiered_path)) if tiered_path.is_file() else {}
    top50 = dict(read_json(top50_path)) if top50_path.is_file() else {}
    source = [
        *(
            str(row.get("eur_spot_market") or "")
            for row in top50.get("rows") or []
            if row.get("eur_spot_market")
        ),
        *(tiered.get("discovery_markets") or []),
        *fallback,
        *settings.autonomous_live.monitor_only_markets,
        *load_execution_market_exceptions(settings),
    ]
    return tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-")
            for market in source
            if str(market).strip().upper().endswith("-EUR")
        )
    )


def select_intensive_tracking_markets(
    *,
    core_markets: Iterable[str],
    position_markets: Iterable[str],
    mover_ranking: Iterable[Mapping[str, Any]],
    current_markets: Iterable[str],
    maximum_markets: int,
    mover_slots: int,
) -> tuple[str, ...]:
    """Select bounded heavy tracking without granting trading authority."""

    mandatory = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-")
            for market in (*tuple(core_markets), *tuple(position_markets))
            if str(market).strip().upper().endswith("-EUR")
        )
    )
    capacity = max(maximum_markets, len(mandatory))
    qualified_movers = [
        str(row.get("market") or "").strip().upper().replace("/", "-")
        for row in mover_ranking
        if row.get("qualified_for_intensive_tracking") is True
        and str(row.get("market") or "").strip().upper().endswith("-EUR")
    ][:mover_slots]
    source = [*mandatory, *qualified_movers, *tuple(current_markets)]
    return tuple(dict.fromkeys(source))[:capacity]


def _tail_jsonl(path: Path, *, limit: int = 25) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _execution_evidence_summary(path: Path) -> dict[str, Any]:
    """Keep health/status output compact while preserving the full artifact."""

    if not path.is_file():
        return {"status": "NOT_YET_BUILT"}
    payload = dict(read_json(path))
    theoretical = dict(payload.get("theoretical_signal_pnl") or {})
    simulated = dict(payload.get("simulated_execution_pnl") or {})
    actual = dict(payload.get("actual_live_pnl") or {})
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "evidence_hash": payload.get("evidence_hash"),
        "artifact": str(path),
        "theoretical_signal_pnl": {
            key: theoretical.get(key)
            for key in (
                "status",
                "resolved_episode_count",
                "false_breakout_rate",
            )
        },
        "simulated_execution_pnl": {
            key: simulated.get(key)
            for key in (
                "status",
                "closed_round_trips",
                "net_pnl_eur",
                "net_expectancy_eur",
                "fees_eur",
            )
        },
        "actual_live_pnl": {
            key: actual.get(key)
            for key in (
                "status",
                "integrity_status",
                "closed_round_trips",
                "open_positions",
                "realised_pnl_eur",
                "unrealised_pnl_eur",
                "net_pnl_eur",
                "fees_eur",
                "active_strategy_count",
            )
        },
        "comparison_policy": payload.get("comparison_policy"),
    }


def _strategy_evidence_watch_summary(path: Path) -> dict[str, Any]:
    """Expose recommendation counts without serializing every strategy row."""

    if not path.is_file():
        return {"status": "NOT_YET_BUILT"}
    payload = dict(read_json(path))
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "artifact": str(path),
        "live_accounting_integrity": payload.get(
            "live_accounting_integrity"
        ),
        "watched_strategy_evidence_row_count": payload.get(
            "watched_strategy_evidence_row_count"
        ),
        "recommendation_counts": payload.get("recommendation_counts"),
        "automatic_authority_changes": bool(
            dict(payload.get("policy") or {}).get(
                "automatic_authority_changes"
            )
        ),
    }


def _strategy_entry_authority_summary(settings: Settings) -> dict[str, Any]:
    """Expose path-specific entry authority without conflating it with health.

    Account health is necessary but does not itself authorize a strategy.  The
    supervisor has two independent live entry paths, so operators need to see
    both persisted authorities and the aggregate explicitly.
    """

    playbook_path = settings.paths.project_root / "config" / "live_playbook_authority.json"
    generated_path = (
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    playbook = dict(read_json(playbook_path)) if playbook_path.is_file() else {}
    generated = dict(read_json(generated_path)) if generated_path.is_file() else {}
    approved_playbooks = [
        row
        for row in playbook.get("approved_playbooks") or []
        if row.get("active") is True
    ]
    playbook_active = playbook.get("active") is True and bool(approved_playbooks)
    generated_active = generated.get("active") is True
    economics = load_canonical_entry_economics_gate(settings)
    live_families = set(economics["live_entry_families"])
    live_dna = set(economics["live_entry_strategy_dna_hashes"])
    approved_live_families = sorted(
        {
            canonical_family(
                str(row.get("playbook_id") or ""),
                str(row.get("family") or ""),
            )[0]
            for row in approved_playbooks
        }
        & live_families
    )
    approved_live_dna = sorted(
        {
            str(row.get("strategy_dna_hash") or "").lower()
            for row in generated.get("approved_candidates") or []
            if row.get("strategy_dna_hash")
        }
        & live_dna
    )
    playbook_effective = playbook_active and bool(approved_live_families)
    generated_effective = generated_active and bool(approved_live_dna)
    return {
        "event_playbook": {
            "active": playbook_active,
            "approved_active_playbook_count": len(approved_playbooks),
            "authority_present": bool(playbook),
            "canonical_economics_entry_families": approved_live_families,
            "effective_entry_authorized": playbook_effective,
        },
        "generated_positive_portfolio": {
            "active": generated_active,
            "approved_candidate_count": len(
                generated.get("approved_candidates") or []
            ),
            "authority_present": bool(generated),
            "canonical_economics_entry_strategy_dna_hashes": (
                approved_live_dna
            ),
            "effective_entry_authorized": generated_effective,
        },
        "at_least_one_entry_path_authorized": (
            playbook_effective or generated_effective
        ),
        "canonical_economics": economics,
        "protective_exit_authority_is_independent": True,
    }


def _projected_entry_blockers(
    account: Mapping[str, Any],
    *,
    economics_entry_allowed: bool,
    control_state: str,
) -> dict[str, list[str]]:
    """Return complete blockers grouped by their independent authority scope."""

    account_blockers = [
        str(value)
        for value in account.get("entry_blockers") or []
        if str(value)
    ]
    if account.get("entry_allowed") is not True:
        account_status = str(account.get("status") or "NOT_READY").upper()
        if account_status not in {"READY", "HEALTHY"}:
            account_blockers.append(f"ACCOUNT_HEALTH_{account_status}")
        account_blockers.extend(
            str(value)
            for value in account.get("failures") or []
            if str(value)
        )
    strategy_blockers = (
        []
        if economics_entry_allowed
        else ["CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"]
    )
    control_blockers = (
        [] if control_state == "ENABLED" else [f"CONTROL_STATE_{control_state}"]
    )
    by_scope = {
        "account": list(dict.fromkeys(account_blockers)),
        "strategy_and_economics": strategy_blockers,
        "control": control_blockers,
    }
    by_scope["all"] = list(
        dict.fromkeys(
            value
            for scope in (
                "account",
                "strategy_and_economics",
                "control",
            )
            for value in by_scope[scope]
        )
    )
    return by_scope


class AutonomousLiveSupervisor:
    """Coordinate existing data, execution, risk and research components."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.markets = resolved_live_markets(settings)
        self.root = settings.paths.project_root.resolve()
        self.output = settings.paths.output_dir / "live"
        self.events = self.output / "events"
        self.state_path = self.output / "autonomous_live_state.json"
        self.authority_path = self.output / "autonomous_live_authority.json"
        self.status_path = self.output / "autonomous_live_status.json"
        self.heartbeat_path = self.output / "heartbeat.json"
        self.lock_path = self.output / "autonomous_live.lock"
        self.execution_cursor_path = self.output / "execution_event_cursor.json"
        self.realtime_microstructure_path = (
            self.output / "realtime_microstructure.json"
        )
        self.opportunity_lifecycle = OpportunityLifecycleLedger(
            ledger_path=self.events / "opportunity_lifecycle.jsonl",
            state_path=self.output / "opportunity_lifecycle_state.json",
        )
        self.performance_notification_state_path = (
            self.output / "performance_notification_state.json"
        )
        self.macro_notification_state_path = (
            self.output / "macro_notification_state.json"
        )
        self.companion_status_path = self.output / "companion_services.json"
        self.execution_ledger_path = (
            settings.paths.checkpoints_dir / "live_execution.jsonl"
        )
        self.position_tracker = PositionTracker(
            self.output / "position_tracker.json"
        )
        self.websocket = WebSocketManager(
            queue_size=settings.autonomous_live.websocket_queue_size,
            backpressure_policy="drop_oldest",
            maximum_connection_attempts=5,
            # The candle/control stream is deliberately low-volume.  Give a
            # quiet venue two minutes before reconnecting so a short CPU or
            # network stall cannot create a reconnect storm.
            inactivity_timeout=120.0,
        )
        configured_orderflow_markets = resolved_orderflow_markets(
            settings,
            fallback=self.markets,
        )
        # Monitor-only markets are mandatory inputs for the market-data and
        # microstructure recorders, but are deliberately absent from
        # ``self.markets`` and therefore receive no execution authority.
        self.core_orderflow_markets = tuple(
            dict.fromkeys(
                (
                    *self.markets,
                    *settings.autonomous_live.monitor_only_markets,
                )
            )
        )
        # Bootstrap execution-critical markets first.  Persisted mover state
        # can contain 35 books; seeding all of them through the venue REST
        # limiter before starting the recorder delayed live readiness by more
        # than a minute.  The 10-second rotation loop promotes the remaining
        # configured/dynamic movers in bounded causal batches after the core
        # (including ADA/NPC) is fully synchronized.
        self.orderflow_markets = tuple(
            dict.fromkeys(self.core_orderflow_markets)
        )
        self.ticker_tracking_markets = resolved_ticker_tracking_markets(
            settings,
            fallback=configured_orderflow_markets,
        )
        self.orderflow_websocket = WebSocketManager(
            queue_size=max(
                # Initial REST seeding buffers venue deltas for every
                # intensive market.  Production bursts exceeded the former
                # 20k queue by ~3.7k events before the recorder could apply
                # the snapshot nonce.  A 50k bounded queue absorbs that
                # causal startup window without dropping live facts.
                50_000,
                settings.autonomous_live.websocket_queue_size,
            ),
            backpressure_policy="drop_oldest",
            maximum_connection_attempts=5,
            inactivity_timeout=120.0,
            # Venue-wide ticker24h coverage is used for cheap mover ranking,
            # not tick-by-tick execution.  Coalescing it at the producer
            # keeps lossless trades/books responsive under market bursts.
            ticker_minimum_interval_seconds=1.0,
        )
        self.orderflow_recorder: ProspectiveOrderflowRecorder | None = None
        self.orderflow_recorder_task: asyncio.Task[None] | None = None
        self._last_trade_channel_recovery_monotonic = 0.0
        self._last_ticker_channel_recovery_monotonic = 0.0
        trade_key = settings.providers.bitvavo_trade_api_key
        trade_secret = settings.providers.bitvavo_trade_api_secret
        self.private_account_stream = (
            BitvavoPrivateAccountStream(
                api_key=trade_key,
                api_secret=trade_secret,
                markets=self.markets,
                queue_size=min(
                    2_000,
                    settings.autonomous_live.websocket_queue_size,
                ),
            )
            if trade_key is not None and trade_secret is not None
            else None
        )
        self.autopilot = PracticalAutopilot(settings)
        self.notifier = TelegramNotifier(
            settings.telegram,
            output_directory=(
                settings.paths.output_dir / "notifications"
            ),
            allowed_markets=self.markets,
        )
        self._lock_token: str | None = None
        self._stop = asyncio.Event()
        self._status_thread_stop = threading.Event()
        self._last_private_stream_ready: bool | None = None
        self._last_entry_blockers: tuple[str, ...] | None = None
        self._last_signal_ids: set[str] = set()
        self._last_disposition_ids: set[str] = set()
        self._last_companion_check_monotonic = 0.0
        self._last_public_stream_restart_monotonic = 0.0
        self._public_stream_restart_attempts = 0
        self._consecutive_healthy_reconciliations = 0
        self._last_active_trading_scan_slot: datetime | None = None
        self._active_scan_not_before_monotonic = time.monotonic() + max(
            60.0,
            float(settings.autonomous_live.health_seconds) * 6.0,
        )
        self._last_realtime_projection_monotonic = 0.0
        self._realtime_market_fingerprints: dict[str, tuple[Any, ...]] = {}
        self._last_event_paper_monotonic = 0.0
        self._last_event_live_monotonic = 0.0
        self._last_opportunity_audit_slot: datetime | None = None
        self._opportunity_audit_process: subprocess.Popen[bytes] | None = None
        self._last_intelligence_training_slot: datetime | None = None
        self._intelligence_training_process: subprocess.Popen[bytes] | None = None
        self._last_intensive_rotation_monotonic = 0.0
        self._companion_spawned: dict[str, int] = {}
        self._ensure_paths()
        self._load_signal_ids()

    def _public_subscriptions(self) -> dict[str, dict[str, Any]]:
        """Return the low-volume control/candle stream subscription.

        Trades and order-book deltas already have a dedicated, much larger
        recorder queue.  Subscribing to them a second time doubled provider
        traffic and let research scans starve the control stream.  Execution
        consumes the lossless recorder projection, while this connection
        remains responsible for candle monitoring and transport health.
        """

        return {
            "bitvavo": {
                "ticker": self.markets,
                "candles": {
                    "markets": self.markets,
                    # Weekly context is causally resampled by the continuous
                    # data service because Bitvavo has no weekly WS interval.
                    # Native tactical candles remain monitoring/context facts;
                    # realtime trades and books still own exact entry timing.
                    "interval": (
                        "1m",
                        "5m",
                        "15m",
                        "1h",
                        "2h",
                        "4h",
                        "1d",
                    ),
                },
            }
        }

    async def _communicate_worker(
        self,
        name: str,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes]:
        """Wait for a child and terminate its process tree when cancelled."""

        try:
            return await process.communicate()
        except asyncio.CancelledError:
            termination = "ALREADY_EXITED"
            if process.returncode is None:
                termination = "TERMINATE_REQUESTED"
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    await killer.wait()
                else:
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    termination = "TERMINATED"
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    termination = "KILLED_AFTER_TIMEOUT"
            self._event(
                "health",
                {
                    "event": "SHUTDOWN_CHILD_WORKER_TERMINATED",
                    "worker": name,
                    "pid": process.pid,
                    "termination": termination,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
            raise

    async def _run_active_trading_scan_isolated(self) -> dict[str, Any]:
        """Run the broad strategy scan outside the live supervisor process.

        The scan evaluates hundreds of strategies and invokes NumPy/pandas
        kernels.  A Python thread still competes for the GIL and previously
        delayed both public WebSockets long enough to make the live engine
        fail closed.  A below-normal, single-threaded child process preserves
        research throughput without taking scheduling time from market data,
        reconciliation, exits or the heartbeat.
        """

        maximum_rows = int(
            self.settings.autonomous_live.active_trading_maximum_rows
        )
        command = (
            sys.executable,
            str(self.root / "main.py"),
            "active-trading",
            "scan-all",
            "--no-execute",
            "--maximum-rows",
            str(maximum_rows),
        )
        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        status_path = self.output / "active_scan_worker_status.json"
        atomic_write_json(
            status_path,
            {
                "schema_version": "active_scan_worker_status_v1",
                "status": "STARTING",
                "started_at": utc_iso(),
                "maximum_rows": maximum_rows,
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.root),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )
        atomic_write_json(
            status_path,
            {
                "schema_version": "active_scan_worker_status_v1",
                "status": "RUNNING",
                "started_at": utc_iso(),
                "pid": process.pid,
                "maximum_rows": maximum_rows,
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        stdout, stderr = await self._communicate_worker(
            "active_trading_scan",
            process,
        )
        stderr_hash = stable_hash(stderr.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            atomic_write_json(
                status_path,
                {
                    "schema_version": "active_scan_worker_status_v1",
                    "status": "FAILED",
                    "completed_at": utc_iso(),
                    "return_code": process.returncode,
                    "stderr_hash": stderr_hash,
                    "secrets_serialized": False,
                    "execution_enabled": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
            raise RuntimeError(
                f"ACTIVE_TRADING_SCAN_WORKER_EXIT_{process.returncode}"
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ACTIVE_TRADING_SCAN_INVALID_OUTPUT") from exc
        if not isinstance(result, dict):
            raise RuntimeError("ACTIVE_TRADING_SCAN_INVALID_RESULT")
        atomic_write_json(
            status_path,
            {
                "schema_version": "active_scan_worker_status_v1",
                "status": "COMPLETED",
                "completed_at": utc_iso(),
                "return_code": process.returncode,
                "stderr_hash": stderr_hash,
                "result_status": result.get("status"),
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return dict(result)

    async def _run_opportunity_audit_isolated(self) -> dict[str, Any]:
        """Build the expensive daily counterfactual audit out of process.

        Parsing the append-only opportunity ledger is CPU/GIL heavy.  A
        background thread still starved the asyncio transport loop during
        startup and every tactical scan.  The canonical CLI worker writes the
        same audit artifact while the supervisor remains responsive.
        """

        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        status_path = self.output / "opportunity_audit_worker_status.json"
        atomic_write_json(
            status_path,
            {
                "schema_version": "opportunity_audit_worker_status_v1",
                "status": "STARTING",
                "started_at": utc_iso(),
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.root / "main.py"),
            "live",
            "opportunity-audit",
            cwd=str(self.root),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )
        atomic_write_json(
            status_path,
            {
                "schema_version": "opportunity_audit_worker_status_v1",
                "status": "RUNNING",
                "started_at": utc_iso(),
                "pid": process.pid,
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        stdout, stderr = await self._communicate_worker(
            "opportunity_audit",
            process,
        )
        stderr_hash = stable_hash(stderr.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            atomic_write_json(
                status_path,
                {
                    "schema_version": "opportunity_audit_worker_status_v1",
                    "status": "FAILED",
                    "completed_at": utc_iso(),
                    "return_code": process.returncode,
                    "stderr_hash": stderr_hash,
                    "secrets_serialized": False,
                    "execution_enabled": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
            raise RuntimeError(
                f"OPPORTUNITY_AUDIT_WORKER_EXIT_{process.returncode}"
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OPPORTUNITY_AUDIT_INVALID_OUTPUT") from exc
        if not isinstance(result, dict):
            raise RuntimeError("OPPORTUNITY_AUDIT_INVALID_RESULT")
        atomic_write_json(
            status_path,
            {
                "schema_version": "opportunity_audit_worker_status_v1",
                "status": "COMPLETED",
                "completed_at": utc_iso(),
                "return_code": process.returncode,
                "stderr_hash": stderr_hash,
                "result_status": result.get("pnl_status"),
                "execution_enabled": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return dict(result)

    def _trigger_opportunity_audit_worker(self) -> dict[str, Any]:
        """Start the non-executing audit without awaiting it in the live loop."""

        existing = self._opportunity_audit_process
        if existing is not None and existing.poll() is None:
            return {
                "status": "ALREADY_RUNNING",
                "pid": existing.pid,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        log_path = self.settings.paths.logs_dir / "opportunity_audit_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("ab")
        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.root / "main.py"),
                    "live",
                    "opportunity-audit",
                ],
                cwd=self.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            stream.close()
        self._opportunity_audit_process = process
        payload = {
            "schema_version": "opportunity_audit_worker_status_v1",
            "status": "RUNNING_DETACHED_FROM_LIVE_LOOP",
            "started_at": utc_iso(),
            "pid": process.pid,
            "log": str(log_path),
            "execution_enabled": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(
            self.output / "opportunity_audit_worker_status.json",
            payload,
        )
        return payload

    def _terminate_opportunity_audit_worker(self) -> str:
        process = self._opportunity_audit_process
        if process is None or process.poll() is not None:
            return "NOT_RUNNING"
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return "TERMINATED" if completed.returncode == 0 else "TERMINATION_FAILED"
        process.terminate()
        return "TERMINATE_REQUESTED"

    def _trigger_intelligence_training_worker(self) -> dict[str, Any]:
        """Continuously refresh leakage-safe ML data and shadow models."""

        existing = self._intelligence_training_process
        if existing is not None and existing.poll() is None:
            return {
                "status": "ALREADY_RUNNING",
                "pid": existing.pid,
                "authority": "SHADOW_ONLY",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        log_path = self.settings.paths.logs_dir / "intelligence_training_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("ab")
        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.root / "main.py"),
                    "live",
                    "intelligence-train-shadow",
                ],
                cwd=self.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                creationflags=creationflags,
                close_fds=True,
            )
        finally:
            stream.close()
        self._intelligence_training_process = process
        payload = {
            "schema_version": "intelligence_training_worker_v1",
            "status": "RUNNING_SHADOW_ONLY",
            "started_at": utc_iso(),
            "pid": process.pid,
            "log": str(log_path),
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(
            self.output / "intelligence_training_worker_status.json",
            payload,
        )
        return payload

    def _terminate_intelligence_training_worker(self) -> str:
        process = self._intelligence_training_process
        if process is None or process.poll() is not None:
            return "NOT_RUNNING"
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return "TERMINATED" if completed.returncode == 0 else "TERMINATION_FAILED"
        process.terminate()
        return "TERMINATE_REQUESTED"

    def _intelligence_training_worker_health(self) -> dict[str, Any]:
        """Reconcile the short-lived shadow trainer with its status artifact."""

        status_path = self.output / "intelligence_training_worker_status.json"
        existing = (
            dict(read_json(status_path)) if status_path.is_file() else {}
        )
        process = self._intelligence_training_process
        if process is None:
            return existing or {
                "status": "NOT_STARTED",
                "authority": "SHADOW_ONLY",
                "live_decision_influence": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        return_code = process.poll()
        if return_code is None:
            return existing
        payload = {
            **existing,
            "status": "COMPLETED" if return_code == 0 else "FAILED",
            "completed_at": utc_iso(),
            "return_code": return_code,
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(status_path, payload)
        self._intelligence_training_process = None
        return payload

    def _intelligence_model_health(self) -> dict[str, Any]:
        """Expose only sanitized shadow-model and drift state in heartbeat."""

        path = (
            self.settings.paths.output_dir
            / "intelligence"
            / "model_status.json"
        )
        existing = dict(read_json(path)) if path.is_file() else {}
        drift = dict(existing.get("drift_monitor") or {})
        return {
            "status": existing.get("status") or "DATA_PENDING",
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "row_count": int(existing.get("row_count") or 0),
            "trained_until_timestamp": existing.get(
                "trained_until_timestamp"
            ),
            "chronological_validation": bool(
                existing.get("chronological_validation")
            ),
            "drift_monitor": {
                "status": drift.get("status") or "DATA_PENDING",
                "critical_feature_count": int(
                    drift.get("critical_feature_count") or 0
                ),
                "warning_feature_count": int(
                    drift.get("warning_feature_count") or 0
                ),
                "authority": "SHADOW_ONLY",
                "live_decision_influence": False,
            },
            "fallback_policy": existing.get("fallback_policy")
            or "DETERMINISTIC_RULE_ENGINE",
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def _orderflow_subscriptions(self) -> dict[str, dict[str, Any]]:
        return {
            "bitvavo": {
                "ticker24h": self.ticker_tracking_markets,
                "trades": self.orderflow_markets,
                "book": self.orderflow_markets,
            }
        }

    def _initial_orderflow_subscriptions(self) -> dict[str, dict[str, Any]]:
        """Subscribe only books until the startup snapshots are installed.

        Venue-wide tickers plus every trade previously accumulated a very
        large pre-snapshot queue during REST seeding.  Books must be buffered
        for nonce continuity; tickers and trades can start immediately after
        the local books have a causal base.
        """

        return {"bitvavo": {"book": self.orderflow_markets}}

    def _write_top20_tracking_status(self) -> dict[str, Any]:
        current_path = (
            self.settings.paths.output_dir / "universe" / "top50_current.json"
        )
        eligibility_path = (
            self.settings.paths.output_dir / "universe" / "top50_eligibility.json"
        )
        current = dict(read_json(current_path)) if current_path.is_file() else {}
        eligibility = (
            dict(read_json(eligibility_path))
            if eligibility_path.is_file()
            else {}
        )
        eligibility_by_symbol = {
            str(row.get("symbol") or ""): dict(row)
            for row in eligibility.get("rows") or []
        }
        rows: list[dict[str, Any]] = []
        for source in current.get("rows") or []:
            if int(source.get("rank") or 10_000) > 20:
                continue
            symbol = str(source.get("symbol") or "")
            market = source.get("eur_spot_market")
            eligible = eligibility_by_symbol.get(symbol, {})
            if market in self.orderflow_markets:
                mode = "REALTIME_TICKER_TRADES_ORDERBOOK"
            elif market in self.ticker_tracking_markets:
                mode = "REALTIME_TICKER_CONTEXT_ONLY"
            else:
                mode = "CMC_CONTEXT_NO_BITVAVO_EUR_MARKET"
            rows.append(
                {
                    "rank": source.get("rank"),
                    "symbol": symbol,
                    "name": source.get("name"),
                    "eur_spot_market": market,
                    "tracking_mode": mode,
                    "realtime_ticker": market in self.ticker_tracking_markets,
                    "realtime_trades": market in self.orderflow_markets,
                    "realtime_orderbook": market in self.orderflow_markets,
                    "research_eligibility": eligible.get(
                        "research_eligibility"
                    ),
                    "execution_eligibility": eligible.get(
                        "execution_eligibility"
                    ),
                    "execution_reason": eligible.get("execution_reason"),
                    "available_at": source.get("available_at"),
                }
            )
        payload = {
            "schema_version": "coinmarketcap_top20_live_tracking_v1",
            "generated_at": utc_iso(),
            "source_snapshot_hash": current.get("source_snapshot_hash"),
            "source_collected_at": current.get("source_collected_at"),
            "top20_count": len(rows),
            "realtime_ticker_count": sum(
                row["realtime_ticker"] for row in rows
            ),
            "intensive_orderflow_count": sum(
                row["realtime_orderbook"] for row in rows
            ),
            "rows": rows,
            "execution_authority_unchanged": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(
            self.settings.paths.output_dir
            / "universe"
            / "top20_live_tracking.json",
            payload,
        )
        return payload

    async def _refresh_bitvavo_ticker_universe(self) -> dict[str, Any]:
        """Refresh cheap coverage from current venue facts before subscribing."""

        try:
            records = await DataLoader(
                self.settings
            ).download_market_metadata(
                provider="bitvavo",
                market=None,
                persist=False,
            )
            refresh_status = "READY"
        except Exception as exc:
            records = []
            refresh_status = "DEGRADED_CACHED_UNIVERSE"
            self._event(
                "errors",
                {
                    "event": "BITVAVO_TICKER_UNIVERSE_REFRESH_FAILED",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )
        venue_markets = tuple(
            dict.fromkeys(
                str(record.values.get("market") or record.canonical_market)
                .strip()
                .upper()
                .replace("/", "-")
                for record in records
                if str(
                    record.values.get("market") or record.canonical_market
                )
                .strip()
                .upper()
                .endswith("-EUR")
                and str(record.values.get("status") or "trading").casefold()
                == "trading"
            )
        )
        self.ticker_tracking_markets = tuple(
            dict.fromkeys((*venue_markets, *self.ticker_tracking_markets))
        )
        rows = [
            {
                "market": market,
                "realtime_ticker": True,
                "intensive_orderflow": market in self.orderflow_markets,
                "shariah_status": self.settings.shariah.eligibility(
                    market
                ).status.value,
                "execution_authority_granted": market in self.markets,
            }
            for market in self.ticker_tracking_markets
        ]
        payload = {
            "schema_version": "bitvavo_eur_ticker_universe_v1",
            "status": refresh_status,
            "generated_at": utc_iso(),
            "venue_eur_market_count": len(venue_markets),
            "ticker_market_count": len(rows),
            "intensive_orderflow_count": sum(
                row["intensive_orderflow"] for row in rows
            ),
            "rows": rows,
            "execution_authority_unchanged": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(
            self.settings.paths.output_dir
            / "universe"
            / "bitvavo_eur_ticker_universe.json",
            payload,
        )
        return payload

    async def _seed_orderflow_books(
        self,
        recorder: ProspectiveOrderflowRecorder,
        *,
        markets: Iterable[str] | None = None,
        pause_before_apply: bool = False,
    ) -> None:
        loader = DataLoader(self.settings)
        selected_markets = tuple(markets or self.orderflow_markets)

        async def download(market: str) -> Any:
            return await loader.download_orderbook_snapshot(
                provider="bitvavo",
                market=market,
                depth=min(
                    100,
                    self.settings.market_data.orderbook_maximum_depth,
                ),
                persist=False,
                mode="live",
            )

        # The WebSocket is already connected and buffering deltas.  Fetch all
        # REST seeds concurrently so the first and last market are based on the
        # same narrow time window.  Sequential seeding across 21 markets made
        # the early snapshots stale before the recorder resumed.
        snapshots = await asyncio.gather(
            *(download(market) for market in selected_markets)
        )
        if pause_before_apply:
            await recorder.pause()
        try:
            for snapshot in snapshots:
                recorder.seed_orderbook(snapshot)
        finally:
            if pause_before_apply:
                recorder.resume()

    async def _wait_for_orderflow_stream(
        self,
        *,
        timeout_seconds: float = 20.0,
        after: datetime | None = None,
    ) -> dict[str, Any]:
        """Wait until the new public connection is actually receiving data."""

        deadline = time.monotonic() + timeout_seconds
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline and not self._stop.is_set():
            latest = dict(self.orderflow_websocket.health("bitvavo"))
            last_message_at = latest.get("last_message_at")
            if (
                str(latest.get("state") or "").upper() == "CONNECTED"
                and last_message_at is not None
                and (
                    after is None
                    or datetime.fromisoformat(
                        str(last_message_at).replace("Z", "+00:00")
                    )
                    > after
                )
            ):
                return latest
            await asyncio.sleep(0.05)
        raise TimeoutError("ORDERFLOW_STREAM_NOT_READY_FOR_RESEED")

    def _open_position_markets(self) -> tuple[str, ...]:
        return tuple(
            market
            for market, position in self.position_tracker.positions.items()
            if position.owned_quantity > 0
        )

    def _armed_opportunity_markets(self) -> tuple[str, ...]:
        """Keep every active setup on the intensive realtime feed.

        A venue-wide ticker promotion may discover a mover, but demoting it
        again while its opportunity is WATCHING/ARMED/ENTRY_READY would break
        the required lifecycle.  Terminal and pre-discovery rows do not hold
        an intensive slot.
        """

        active_states = {
            OpportunityState.WATCHING.value,
            OpportunityState.ARMED.value,
            OpportunityState.ENTRY_READY.value,
            OpportunityState.ORDER_INTENT_CREATED.value,
            OpportunityState.ORDER_SUBMITTED.value,
            OpportunityState.PARTIALLY_FILLED.value,
            OpportunityState.FILLED.value,
            OpportunityState.MANAGING.value,
            OpportunityState.EXITING.value,
        }
        return tuple(
            dict.fromkeys(
                str(row.get("market") or "").strip().upper()
                for row in self.opportunity_lifecycle.state.values()
                if str(row.get("state") or "").upper() in active_states
                and str(row.get("market") or "").strip().upper().endswith(
                    "-EUR"
                )
            )
        )

    def _write_dynamic_mover_projection(
        self,
        *,
        ranking: list[dict[str, Any]],
        selected: tuple[str, ...],
        previous: tuple[str, ...],
    ) -> dict[str, Any]:
        for row in ranking:
            market = str(row.get("market") or "")
            row["intensive_tracking_selected"] = market in selected
            row["execution_authority_granted"] = market in self.markets
        payload = {
            "schema_version": "dynamic_intensive_movers_v1",
            "generated_at": utc_iso(),
            "ticker_market_count": len(self.ticker_tracking_markets),
            "ranking_count": len(ranking),
            "qualified_mover_count": sum(
                row.get("qualified_for_intensive_tracking") is True
                for row in ranking
            ),
            "intensive_market_limit": (
                self.settings.autonomous_live.intensive_market_limit
            ),
            "selected_markets": list(selected),
            "promoted_markets": sorted(set(selected) - set(previous)),
            "demoted_markets": sorted(set(previous) - set(selected)),
            "top_movers": ranking[:20],
            "execution_authority_unchanged": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(
            self.settings.paths.output_dir
            / "universe"
            / "dynamic_intensive_movers.json",
            payload,
        )
        return payload

    async def _rotate_intensive_markets(
        self,
        recorder: ProspectiveOrderflowRecorder,
    ) -> dict[str, int] | None:
        ranking = recorder.realtime_ticker_movers(
            minimum_quote_volume_eur=(
                self.settings.autonomous_live
                .dynamic_minimum_24h_quote_volume_eur
            )
        )
        selected = select_intensive_tracking_markets(
            core_markets=self.core_orderflow_markets,
            position_markets=(
                *self._open_position_markets(),
                *self._armed_opportunity_markets(),
            ),
            mover_ranking=ranking,
            current_markets=self.orderflow_markets,
            maximum_markets=(
                self.settings.autonomous_live.intensive_market_limit
            ),
            mover_slots=self.settings.autonomous_live.intensive_mover_slots,
        )
        previous = tuple(self.orderflow_markets)
        if set(selected) == set(previous):
            selected = previous
        if selected == previous:
            self._write_dynamic_mover_projection(
                ranking=ranking,
                selected=selected,
                previous=previous,
            )
            return None

        promoted = tuple(sorted(set(selected) - set(previous)))
        demoted = tuple(sorted(set(previous) - set(selected)))
        subscription_updates: list[dict[str, Any]] = []
        working = list(previous)

        if demoted:
            await recorder.pause()
            try:
                subscription_updates.append(
                    await self.orderflow_websocket.update_bitvavo_subscriptions(
                        unsubscribe={
                            "trades": demoted,
                            "book": demoted,
                        },
                    )
                )
                working = [market for market in working if market not in demoted]
                self.orderflow_markets = tuple(working)
                recorder.markets = self.orderflow_markets
            finally:
                recorder.resume()

        # Subscribe-before-snapshot preserves the venue nonce contract.  Small
        # batches bound how long existing live markets are paused while REST
        # seeds are fetched through Bitvavo's rate limiter.
        for offset in range(0, len(promoted), 5):
            chunk = promoted[offset : offset + 5]
            await recorder.pause()
            try:
                subscription_updates.append(
                    await self.orderflow_websocket.update_bitvavo_subscriptions(
                        subscribe={
                            "trades": chunk,
                            "book": chunk,
                        },
                    )
                )
                await self._seed_orderflow_books(
                    recorder,
                    markets=chunk,
                )
                working.extend(
                    market for market in chunk if market not in working
                )
                self.orderflow_markets = tuple(working)
                recorder.markets = self.orderflow_markets
                recovered = self.orderflow_websocket.health("bitvavo")
                recorder.acknowledge_stream_recovery(recovered)
            finally:
                recorder.resume()

        self.orderflow_markets = selected
        recorder.markets = selected
        projection = self._write_dynamic_mover_projection(
            ranking=ranking,
            selected=selected,
            previous=previous,
        )
        health = self.orderflow_websocket.health("bitvavo")
        recorder.acknowledge_stream_recovery(health)
        self._event(
            "health",
            {
                "event": "DYNAMIC_INTENSIVE_MARKETS_ROTATED",
                "previous_markets": list(previous),
                "selected_markets": list(selected),
                "promoted_markets": projection["promoted_markets"],
                "demoted_markets": projection["demoted_markets"],
                "subscription_updates": subscription_updates,
                "connection_preserved": all(
                    update.get("connection_preserved") is True
                    for update in subscription_updates
                ),
                "promotion_batch_size": 5,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return {
            key: int(health.get(key) or 0)
            for key in ("sequence_gaps", "dropped_messages", "reconnects")
        }

    async def _orderflow_loop(self) -> None:
        """Record top-20 prospective flow on an isolated public connection."""

        ledger = await asyncio.to_thread(
            lambda: HashChainedOrderflowLedger(
                # Start a distinct top-20 ledger.  The earlier four-market
                # chain remains immutable evidence and does not need to be
                # rescanned before every production restart.
                root=(
                    self.settings.paths.context_data_dir
                    / "orderflow_stream_top20_v2"
                ),
                checkpoint_path=(
                    self.settings.paths.checkpoints_dir
                    / "orderflow_stream_top20_v2_chain.json"
                ),
                maximum_storage_bytes=int(
                    self.settings.market_data.maximum_storage_gb * 1024**3
                ),
                checkpoint_first_recovery=True,
            )
        )
        recorder = ProspectiveOrderflowRecorder(
            ledger=ledger,
            database=None,
            markets=self.orderflow_markets,
            feature_directory=(
                self.settings.paths.context_data_dir / "microstructure_hourly"
            ),
            readiness_path=(
                self.settings.paths.output_dir
                / "operations"
                / "microstructure_readiness.json"
            ),
            health_path=(
                self.settings.paths.output_dir
                / "operations"
                / "orderflow_stream_health.json"
            ),
            positioning_directory=(
                self.settings.paths.context_data_dir / "prospective_hourly"
            ),
            realtime_candle_path=(
                self.settings.paths.output_dir
                / "operations"
                / "realtime_candles.json"
            ),
            flush_seconds=0.5,
            batch_size=1_000,
        )
        startup_observed_at = datetime.now(UTC)
        # Invalidate the previous process' projections before the new socket
        # starts seeding books.  Without this, health/status could briefly
        # report an old CONNECTED/HEALTHY snapshot during a fresh startup.
        # Empty candle state is preferable to stale data and remains an
        # explicit fail-closed execution input until real trades arrive.
        recorder.realtime_candle_builder.persist(
            observed_at=startup_observed_at
        )
        if recorder.health_path is not None:
            atomic_write_json(
                recorder.health_path,
                {
                    "schema_version": "orderflow_stream_health_v1",
                    "status": "STARTING",
                    "observed_at": startup_observed_at.isoformat(),
                    "reason_codes": ["STARTUP_SEEDING_BOOKS"],
                    "provider": {
                        "provider": "bitvavo",
                        "state": "STARTING",
                        "event_counts": {},
                        "event_last_message_at": {},
                        "event_last_message_age_ms": {},
                    },
                    "supervisor_pid": os.getpid(),
                    "synthetic_data_used": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
        self.orderflow_recorder = recorder
        baseline = {
            "sequence_gaps": 0,
            "dropped_messages": 0,
            "reconnects": 0,
        }
        try:
            await self.orderflow_websocket.start(
                self._initial_orderflow_subscriptions()
            )
            await self._wait_for_orderflow_stream()
            await self._seed_orderflow_books(recorder)
            current = self.orderflow_websocket.health("bitvavo")
            recorder.acknowledge_stream_recovery(
                current,
                reset_period_baselines=True,
            )
            baseline = {
                key: int(current.get(key) or 0) for key in baseline
            }
            self.orderflow_recorder_task = asyncio.create_task(
                recorder.run(self.orderflow_websocket)
            )
            await self.orderflow_websocket.update_bitvavo_subscriptions(
                subscribe={
                    "ticker24h": self.ticker_tracking_markets,
                    "trades": self.orderflow_markets,
                }
            )
            self._event(
                "health",
                {
                    "event": "TOP20_ORDERFLOW_RECORDING_STARTED",
                    "markets": list(self.orderflow_markets),
                    "historical_backfill_used": False,
                    "staged_startup": True,
                },
            )
            while not self._stop.is_set():
                if (
                    self.orderflow_recorder_task is not None
                    and self.orderflow_recorder_task.done()
                ):
                    exception = self.orderflow_recorder_task.exception()
                    raise RuntimeError(
                        "ORDERFLOW_RECORDER_STOPPED:"
                        + (
                            type(exception).__name__
                            if exception is not None
                            else "UNEXPECTED_COMPLETION"
                        )
                    )
                now_monotonic = time.monotonic()
                if (
                    now_monotonic - self._last_intensive_rotation_monotonic
                    >= self.settings.autonomous_live.intensive_rotation_seconds
                ):
                    rotated_baseline = await self._rotate_intensive_markets(
                        recorder
                    )
                    self._last_intensive_rotation_monotonic = time.monotonic()
                    if rotated_baseline is not None:
                        baseline = rotated_baseline
                health = self.orderflow_websocket.health("bitvavo")
                event_ages = dict(
                    health.get("event_last_message_age_ms") or {}
                )
                uptime_seconds = float(health.get("uptime_seconds") or 0.0)
                channel_recovery: dict[str, list[str]] = {}
                for channel, maximum_age_ms, state_attribute in (
                    (
                        "trades",
                        60_000.0,
                        "_last_trade_channel_recovery_monotonic",
                    ),
                    (
                        "ticker24h",
                        120_000.0,
                        "_last_ticker_channel_recovery_monotonic",
                    ),
                ):
                    event_name = "trade" if channel == "trades" else "ticker"
                    age_ms = event_ages.get(event_name)
                    stale = bool(
                        uptime_seconds >= maximum_age_ms / 1_000.0
                        and (age_ms is None or float(age_ms) > maximum_age_ms)
                    )
                    previous_recovery = float(getattr(self, state_attribute))
                    if (
                        stale
                        and now_monotonic - previous_recovery >= 60.0
                    ):
                        markets = (
                            self.orderflow_markets
                            if channel == "trades"
                            else self.ticker_tracking_markets
                        )
                        await self.orderflow_websocket.update_bitvavo_subscriptions(
                            unsubscribe={channel: markets},
                            subscribe={channel: markets},
                        )
                        setattr(self, state_attribute, time.monotonic())
                        channel_recovery[channel] = list(markets)
                if channel_recovery:
                    self._event(
                        "health",
                        {
                            "event": "ORDERFLOW_CHANNELS_RESUBSCRIBED",
                            "channels": channel_recovery,
                            "book_connection_preserved": True,
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                    health = self.orderflow_websocket.health("bitvavo")
                transport_failed = str(
                    health.get("state") or ""
                ).upper() in {
                    "FAILED",
                    "STOPPED",
                    "STALE",
                }
                counter_advanced = any(
                    int(health.get(key) or 0) > baseline[key]
                    for key in baseline
                )
                invalid_book_markets = (
                    recorder.invalid_orderbook_markets()
                )
                if (
                    transport_failed
                    or counter_advanced
                    or invalid_book_markets
                ):
                    before = dict(baseline)
                    await recorder.pause()
                    try:
                        # A nonce gap, queue drop, or provider-managed
                        # reconnect does not require another forced reconnect.
                        # Keep updates buffering, fetch fresh REST snapshots,
                        # discard deltas at/before each snapshot nonce, then
                        # resume exact +1 application.  This is Bitvavo's
                        # documented local-book recovery procedure and avoids
                        # the old reconnect loop that poisoned every interval.
                        if transport_failed:
                            await self.orderflow_websocket.stop()
                            await self.orderflow_websocket.start(
                                self._orderflow_subscriptions()
                            )
                            await self._wait_for_orderflow_stream()
                        # A locally invalid market may not advance the shared
                        # manager counter (for example an unusable market
                        # delta).  Repair those books proactively.  Global
                        # transport/counter failures still reseed all books.
                        reseed_markets = (
                            None
                            if transport_failed or counter_advanced
                            else invalid_book_markets
                        )
                        await self._seed_orderflow_books(
                            recorder,
                            markets=reseed_markets,
                        )
                        recovered = self.orderflow_websocket.health("bitvavo")
                        baseline = {
                            key: int(recovered.get(key) or 0)
                            for key in baseline
                        }
                        recorder.acknowledge_stream_recovery(recovered)
                        self._event(
                            "health",
                            {
                                "event": "ORDERFLOW_BOOKS_RESEEDED",
                                "transport_restarted": transport_failed,
                                "counter_deltas": {
                                    key: max(
                                        0,
                                        int(recovered.get(key) or 0)
                                        - before[key],
                                    )
                                    for key in before
                                },
                                "markets": list(
                                    reseed_markets
                                    or self.orderflow_markets
                                ),
                                "local_invalid_books": list(
                                    invalid_book_markets
                                ),
                            },
                        )
                    finally:
                        recorder.resume()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=5.0)
                except TimeoutError:
                    pass
        finally:
            recorder.stop()
            if self.orderflow_recorder_task is not None:
                await asyncio.gather(
                    self.orderflow_recorder_task,
                    return_exceptions=True,
                )
            await self.orderflow_websocket.stop()
            self.orderflow_recorder_task = None
            self.orderflow_recorder = None

    def _research_health(self) -> dict[str, Any]:
        """Report the effective research worker instead of a stale batch PID."""

        status = self.autopilot.status()
        background = dict(status.get("background_research") or {})
        continuous = dict(status.get("continuous_research") or {})
        continuous_running = bool(continuous.get("running"))
        batch_running = str(background.get("status") or "").upper() == "RUNNING"
        active = bool(
            status.get("research_subprocess_active")
            or continuous_running
            or batch_running
        )
        return {
            "status": "RUNNING" if active else "NOT_RUNNING",
            "mode": (
                "CONTINUOUS_SIMPLE_LAB"
                if continuous_running
                else "BACKGROUND_CAMPAIGN"
                if batch_running
                else "NONE"
            ),
            "research_subprocess_active": active,
            "continuous_research": continuous,
            "background_research": background,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    async def _recover_public_stream_if_needed(
        self,
        health: dict[str, Any],
    ) -> dict[str, Any]:
        """Restart a terminally failed public stream without restarting trading."""

        if str(health.get("state") or "").upper() != "FAILED":
            return health
        now = time.monotonic()
        cooldown = max(
            10.0,
            float(self.settings.autonomous_live.health_seconds) * 2.0,
        )
        if now - self._last_public_stream_restart_monotonic < cooldown:
            return health
        self._last_public_stream_restart_monotonic = now
        self._public_stream_restart_attempts += 1
        await self.websocket.stop()
        await self.websocket.start(self._public_subscriptions())
        await asyncio.sleep(0)
        recovered = self.websocket.health("bitvavo")
        self._event(
            "health",
            {
                "event": "PUBLIC_MARKET_STREAM_WATCHDOG_RESTART",
                "restart_attempt": self._public_stream_restart_attempts,
                "previous_state": health.get("state"),
                "current_state": recovered.get("state"),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return recovered

    def _ensure_paths(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.events.mkdir(parents=True, exist_ok=True)
        for stream in EVENT_STREAMS:
            (self.events / f"{stream}.jsonl").touch(exist_ok=True)

    def _load_signal_ids(self) -> None:
        """Restore durable signal deduplication across process restarts."""

        self._last_signal_ids = {
            str(row["signal_id"])
            for row in _tail_jsonl(
                self.events / "signals.jsonl",
                limit=10_000,
            )
            if row.get("signal_id")
        }

    def _record_signal_once(self, payload: dict[str, Any]) -> bool:
        signal_id = str(payload.get("signal_id") or "")
        if not signal_id or signal_id in self._last_signal_ids:
            return False
        self._last_signal_ids.add(signal_id)
        if len(self._last_signal_ids) > 10_000:
            self._load_signal_ids()
        self._event("signals", payload)
        return True

    def _recover_position_tracker_from_ledger(self) -> dict[str, Any]:
        """Rebuild the legacy position read model from canonical ledger replay."""

        if not self.execution_ledger_path.is_file():
            return {"status": "READY", "fills_replayed": 0}
        legacy_snapshot = self.position_tracker.snapshot()
        known_fill_ids = set(self.position_tracker.fill_ids)
        ledger = DurableLedger(self.execution_ledger_path)
        try:
            canonical = ledger.canonical_state()
            replay_report = ledger.canonical_replay_report()
        except (KeyError, ReconciliationRequired, TypeError, ValueError) as exc:
            failed_line = 1
            for line_number, line in enumerate(
                self.execution_ledger_path.read_text(
                    encoding="utf-8"
                ).splitlines(),
                start=1,
            ):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    failed_line = line_number
                    break
            self._event(
                "errors",
                {
                    "event": "POSITION_RECOVERY_FAILED",
                    "line_number": failed_line,
                    "exception_type": type(exc).__name__,
                },
            )
            return {
                "status": "RECONCILIATION_REQUIRED",
                "fills_replayed": 0,
                "failed_line": failed_line,
            }

        strategy_fill_ids = {
            fill_id
            for fill_id, fill in canonical.fills.items()
            if "NOT_STRATEGY_TRADE" not in str(fill.strategy_id or "")
            and not str(fill.strategy_id or "").startswith(
                "OPERATOR_INVENTORY_REALLOCATION"
            )
        }
        replayed = len(strategy_fill_ids - known_fill_ids)
        atomic_write_json(
            self.output / "canonical_execution_state.json",
            canonical.to_dict(),
        )
        atomic_write_json(
            self.output / "canonical_execution_replay.json",
            replay_report,
        )
        atomic_write_json(
            self.output / "canonical_execution_divergence.json",
            build_execution_divergence_report(legacy_snapshot, canonical),
        )
        atomic_write_json(
            self.output / "execution_state_migration_status.json",
            execution_state_migration_status(),
        )
        self.position_tracker.apply_canonical_state(canonical)
        return {"status": "READY", "fills_replayed": replayed}

    def _event(self, stream: str, payload: dict[str, Any]) -> None:
        if stream not in EVENT_STREAMS:
            raise ValueError(f"unknown autonomous-live event stream: {stream}")
        append_jsonl(
            self.events / f"{stream}.jsonl",
            {
                "event_id": stable_hash(
                    [stream, payload, utc_iso()],
                    length=40,
                ),
                "recorded_at": utc_iso(),
                **payload,
            },
        )

    def _record_entry_ready_dispositions(
        self,
        report: Mapping[str, Any],
    ) -> None:
        path = self.output / "entry_ready_dispositions.jsonl"
        for raw in report.get("rows") or []:
            row = dict(raw)
            identity = str(row.get("disposition_id") or "")
            if not identity or identity in self._last_disposition_ids:
                continue
            append_jsonl(path, row)
            self._last_disposition_ids.add(identity)
        # Bound only the in-memory deduplication set; the append-only ledger
        # remains complete on disk for forensic replay.
        if len(self._last_disposition_ids) > 20_000:
            self._last_disposition_ids = set(
                list(self._last_disposition_ids)[-10_000:]
            )
        atomic_write_json(
            self.output / "entry_ready_dispositions_status.json",
            dict(report),
        )

    def _sync_canonical_execution_events(self) -> dict[str, Any]:
        """Mirror new canonical execution records once into operator streams."""

        cursor = (
            dict(read_json(self.execution_cursor_path))
            if self.execution_cursor_path.is_file()
            else {}
        )
        consumed = max(0, int(cursor.get("consumed_lines") or 0))
        lines = (
            self.execution_ledger_path.read_text(encoding="utf-8").splitlines()
            if self.execution_ledger_path.is_file()
            else []
        )
        if consumed > len(lines):
            consumed = 0
        mirrored = 0
        for line_number, line in enumerate(lines[consumed:], start=consumed + 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self._event(
                    "errors",
                    {
                        "event": "CANONICAL_EXECUTION_LEDGER_INVALID_JSON",
                        "line_number": line_number,
                    },
                )
                break
            event_type = str(event.get("event_type") or "UNKNOWN")
            stream = (
                "fills"
                if event_type == "FILL"
                else "orders"
                if event_type in {"ORDER_INTENT", "ORDER_ACKNOWLEDGED"}
                else "risk"
            )
            self._event(
                stream,
                {
                    "event": f"CANONICAL_{event_type}",
                    "canonical_line_number": line_number,
                    "canonical_recorded_at": event.get("recorded_at"),
                    "payload": event.get("payload") or {},
                },
            )
            mirrored += 1
        consumed += mirrored
        atomic_write_json(
            self.execution_cursor_path,
            {
                "schema_version": "autonomous_live_execution_cursor_v1",
                "source": str(self.execution_ledger_path),
                "consumed_lines": consumed,
                "source_lines": len(lines),
                "updated_at": utc_iso(),
            },
        )
        return {
            "mirrored": mirrored,
            "consumed_lines": consumed,
            "source_lines": len(lines),
        }

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        try:
            selected = Decimal(str(value))
        except (ArithmeticError, TypeError, ValueError):
            return Decimal("0")
        return selected if selected.is_finite() else Decimal("0")

    def _orders_today(self, selected_date: str) -> int:
        if not self.execution_ledger_path.is_file():
            return 0
        count = 0
        for line in self.execution_ledger_path.read_text(
            encoding="utf-8"
        ).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "ORDER_ACKNOWLEDGED":
                continue
            recorded_at = str(event.get("recorded_at") or "")
            if recorded_at[:10] == selected_date:
                count += 1
        return count

    async def _notify_performance_snapshots(
        self,
        strategy_accounts: dict[str, Any],
    ) -> None:
        """Send durable trade-close and once-daily performance summaries."""

        state = (
            dict(read_json(self.performance_notification_state_path))
            if self.performance_notification_state_path.is_file()
            else {}
        )
        closed_counts = {
            str(key): max(0, int(value))
            for key, value in dict(
                state.get("closed_trade_counts") or {}
            ).items()
        }
        strategies = [
            dict(row)
            for row in strategy_accounts.get("strategies") or []
            if isinstance(row, dict)
        ]
        accepted = {"PENDING", "SENT", "SKIPPED_DUPLICATE"}
        for strategy in strategies:
            dna = str(
                strategy.get("strategy_dna")
                or strategy.get("strategy_id")
                or ""
            )
            if not dna:
                continue
            current_count = max(
                0,
                int(strategy.get("closed_trade_count") or 0),
            )
            previous_count = closed_counts.get(dna, 0)
            if current_count <= previous_count:
                closed_counts[dna] = max(previous_count, current_count)
                continue
            allocated = self._decimal(
                strategy.get("allocated_capital_eur")
            )
            net_pnl = self._decimal(strategy.get("net_pnl_eur"))
            payload = {
                **strategy,
                "strategy_equity_eur": str(allocated + net_pnl),
            }
            try:
                result = await asyncio.to_thread(
                    self.notifier.notify_strategy_performance,
                    payload,
                )
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "TELEGRAM_PERFORMANCE_NOTIFICATION_FAILURE",
                        "notification_type": "STRATEGY_PERFORMANCE",
                        "exception_type": type(exc).__name__,
                        "execution_affected": False,
                    },
                )
            else:
                if result.get("delivery_status") in accepted:
                    closed_counts[dna] = current_count

        selected_date = datetime.now(UTC).date().isoformat()
        if state.get("last_daily_date") != selected_date:
            health_path = (
                self.settings.paths.output_dir
                / "operations"
                / "live_account_health.json"
            )
            target_path = (
                self.settings.paths.output_dir
                / "portfolio"
                / "daily_profit_target.json"
            )
            health = (
                dict(read_json(health_path))
                if health_path.is_file()
                else {}
            )
            target = (
                dict(read_json(target_path))
                if target_path.is_file()
                else {}
            )
            account = dict(health.get("account") or {})
            valuation = dict(account.get("portfolio_valuation") or {})
            realised = sum(
                (
                    self._decimal(row.get("realised_pnl_eur"))
                    for row in strategies
                ),
                Decimal("0"),
            )
            unrealised = sum(
                (
                    self._decimal(row.get("unrealised_pnl_eur"))
                    for row in strategies
                ),
                Decimal("0"),
            )
            fees = sum(
                (
                    self._decimal(row.get("fees_paid_eur"))
                    for row in strategies
                ),
                Decimal("0"),
            )
            active_capital = sum(
                (
                    self._decimal(row.get("used_capital_eur"))
                    for row in strategies
                ),
                Decimal("0"),
            )
            maximum_drawdown = max(
                (
                    self._decimal(row.get("maximum_drawdown_eur"))
                    for row in strategies
                ),
                default=Decimal("0"),
            )
            ranked = sorted(
                strategies,
                key=lambda row: self._decimal(row.get("net_pnl_eur")),
                reverse=True,
            )
            payload = {
                "date_utc": selected_date,
                "wallet_value_eur": valuation.get(
                    "estimated_total_equity_eur"
                ),
                "daily_pnl_eur": target.get("mark_to_market_pnl_eur"),
                "realised_pnl_eur": str(realised),
                "unrealised_pnl_eur": str(unrealised),
                "best_strategy": (
                    ranked[0].get("strategy_id") if ranked else None
                ),
                "worst_strategy": (
                    ranked[-1].get("strategy_id") if ranked else None
                ),
                "fees_eur": str(fees),
                "maximum_drawdown_eur": str(maximum_drawdown),
                "active_capital_eur": str(active_capital),
                "cash_reserve_eur": account.get("eur_available"),
                "authority_status": (
                    f"{self._control_state()}/"
                    f"{ranked[0].get('authority_level') if ranked else 'NONE'}"
                ),
                "open_positions": sum(
                    int(row.get("open_trade_count") or 0)
                    for row in strategies
                ),
                "live_orders_today": self._orders_today(selected_date),
                "account_identity_hash": stable_hash(
                    ["bitvavo-live-account", str(self.root)],
                    length=16,
                ),
            }
            try:
                result = await asyncio.to_thread(
                    self.notifier.notify_daily_performance,
                    payload,
                )
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "TELEGRAM_PERFORMANCE_NOTIFICATION_FAILURE",
                        "notification_type": "DAILY_PERFORMANCE",
                        "exception_type": type(exc).__name__,
                        "execution_affected": False,
                    },
                )
            else:
                if result.get("delivery_status") in accepted:
                    state["last_daily_date"] = selected_date
        state.update(
            {
                "schema_version": "performance_notification_state_v1",
                "updated_at": utc_iso(),
                "closed_trade_counts": closed_counts,
                "orders_generated": 0,
                "orders_submitted": 0,
                "execution_affected_by_notification_failure": False,
            }
        )
        atomic_write_json(self.performance_notification_state_path, state)

    async def _record_runtime_snapshots(self) -> None:
        """Expose canonical position, risk and PnL state without inventing state."""

        strategy_accounts: dict[str, Any] | None = None
        try:
            strategy_accounts = rebuild_live_strategy_accounting(self.root)
            self._event(
                "strategy_lifecycle",
                {
                    "event": "STRATEGY_ACCOUNTING_SNAPSHOT",
                    "source": str(
                        self.output / "strategy_accounts.json"
                    ),
                    "integrity_status": strategy_accounts.get(
                        "integrity_status"
                    ),
                    "live_strategy_account_count": strategy_accounts.get(
                        "live_strategy_account_count"
                    ),
                    "hard_blockers": strategy_accounts.get(
                        "hard_blockers"
                    ),
                },
            )
        except Exception as exc:
            self._event(
                "errors",
                {
                    "event": "STRATEGY_ACCOUNTING_REBUILD_FAILED",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )
        if strategy_accounts is not None:
            await self._notify_performance_snapshots(strategy_accounts)
        try:
            evidence = build_execution_evidence_layers(self.root)
            evidence_watch = build_strategy_evidence_watch(
                self.root,
                execution_evidence=evidence,
            )
            self._event(
                "pnl",
                {
                    "event": "EXECUTION_EVIDENCE_LAYERS_SNAPSHOT",
                    "artifact": evidence["artifact"],
                    "evidence_hash": evidence["evidence_hash"],
                    "actual_live_status": evidence["actual_live_pnl"][
                        "status"
                    ],
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
            self._event(
                "strategy_lifecycle",
                {
                    "event": "STRATEGY_EVIDENCE_WATCH_SNAPSHOT",
                    "artifact": evidence_watch["artifact"],
                    "recommendation_counts": evidence_watch[
                        "recommendation_counts"
                    ],
                    "automatic_authority_changes": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
        except Exception as exc:
            self._event(
                "errors",
                {
                    "event": "EXECUTION_EVIDENCE_REBUILD_FAILED",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )
        try:
            telegram_evidence = build_telegram_signal_evidence(self.settings)
            self._event(
                "strategy_lifecycle",
                {
                    "event": "TELEGRAM_SIGNAL_EVIDENCE_SNAPSHOT",
                    "artifact": telegram_evidence["artifact"],
                    "claim_status": telegram_evidence["claim_under_test"][
                        "status"
                    ],
                    "paper_shadow_gate": telegram_evidence[
                        "paper_shadow_gate"
                    ]["status"],
                    "automatic_authority_changes": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
        except Exception as exc:
            self._event(
                "errors",
                {
                    "event": "TELEGRAM_SIGNAL_EVIDENCE_REBUILD_FAILED",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )
        artifacts = (
            (
                "positions",
                "POSITION_STATE_SNAPSHOT",
                self.settings.paths.output_dir / "reports" / "current_position.json",
            ),
            (
                "risk",
                "RISK_STATE_SNAPSHOT",
                self.settings.paths.output_dir
                / "governance"
                / "live_canary_risk_state.json",
            ),
            (
                "pnl",
                "DAILY_PNL_TARGET_SNAPSHOT",
                self.settings.paths.output_dir
                / "portfolio"
                / "daily_profit_target.json",
            ),
        )
        for stream, event, path in artifacts:
            if path.is_file():
                self._event(
                    stream,
                    {
                        "event": event,
                        "source": str(path),
                        "state": read_json(path),
                    },
                )

    async def _notify_system_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Notify downstream without allowing Telegram to affect execution."""

        try:
            await asyncio.to_thread(
                self.notifier.notify_system_event,
                event_type,
                payload,
            )
        except Exception as exc:
            self._event(
                "errors",
                {
                    "event": "TELEGRAM_NOTIFICATION_FAILURE",
                    "notification_type": event_type,
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )

    async def _notify_scheduled_macro(
        self,
        account_health: Mapping[str, Any],
    ) -> None:
        """Deliver exactly one macro note in each 08:00/23:00 NL slot."""

        slot = amsterdam_macro_slot()
        if slot is None:
            return
        state = (
            dict(read_json(self.macro_notification_state_path))
            if self.macro_notification_state_path.is_file()
            else {}
        )
        if state.get("last_slot") == slot:
            return
        macro_path = (
            self.settings.paths.output_dir
            / "active_trading"
            / "macro_crypto.json"
        )
        regime_path = (
            self.settings.paths.output_dir
            / "reports"
            / "current_regime.json"
        )
        if not macro_path.is_file() or not regime_path.is_file():
            return
        macro = dict(read_json(macro_path))
        structural = dict(read_json(regime_path))
        features = dict(macro.get("features") or {})
        account = dict(account_health.get("account") or {})
        reconciliation = dict(account_health.get("reconciliation") or {})
        event_state_path = self.output / "event_driven_execution_state.json"
        event_state = (
            dict(read_json(event_state_path))
            if event_state_path.is_file()
            else {}
        )
        macro_regime = str(macro.get("regime") or "UNKNOWN")
        one_day_up = features.get("btc_1d_trend_up") is True
        four_hour_up = features.get("btc_4h_trend_up") is True
        if not one_day_up and four_hour_up:
            active_playbooks = (
                "liquidity-sweep, failed-breakdown, VWAP-reclaim, "
                "selectieve relative-strength"
            )
            blocked_playbooks = "ongefilterde trendbreakouts"
            risk_multiplier = 0.4
        elif one_day_up and four_hour_up:
            active_playbooks = "trend, breakout, pullback, relative-strength"
            blocked_playbooks = "geen generieke macroblokkade"
            risk_multiplier = 1.0
        else:
            active_playbooks = "alleen bewezen defensieve/recoverysetups"
            blocked_playbooks = "trend- en momentumlongs"
            risk_multiplier = 0.25
        payload = {
            "observed_at": macro.get("observed_at"),
            "macro_regime": macro_regime,
            "structural_regime": structural.get("primary_regime"),
            "btc_1d_trend": "BULLISH" if one_day_up else "BEARISH",
            "btc_4h_trend": "BULLISH" if four_hour_up else "BEARISH",
            "btc_return_24h_pct": 100.0
            * float(features.get("btc_return_24h") or 0.0),
            "altcoin_breadth_pct": 100.0
            * float(features.get("altcoin_breadth") or 0.0),
            "btc_dominance_pct": 100.0
            * float(features.get("btc_dominance") or 0.0),
            "fear_greed": features.get("fear_greed"),
            "risk_multiplier": risk_multiplier,
            "active_playbooks": active_playbooks,
            "blocked_playbooks": blocked_playbooks,
            "entry_candidates": len(
                event_state.get("entry_candidates") or []
            ),
            "open_positions": len(event_state.get("positions") or {}),
            "open_orders": int(
                reconciliation.get("remote_open_orders") or 0
            ),
            "eur_available": account.get("eur_available"),
            "live_status": (
                "RUNNING"
                if self._control_state() == "ENABLED"
                and account_health.get("status") == "READY"
                else "DEGRADED"
            ),
        }
        try:
            result = await asyncio.to_thread(
                self.notifier.notify_macro_summary,
                payload,
                slot=slot,
            )
        except Exception as exc:
            self._event(
                "errors",
                {
                    "event": "TELEGRAM_MACRO_NOTIFICATION_FAILURE",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                },
            )
            return
        if result.get("delivery_status") in {
            "PENDING",
            "SENT",
            "SKIPPED_DUPLICATE",
        }:
            atomic_write_json(
                self.macro_notification_state_path,
                {
                    "schema_version": "macro_notification_state_v1",
                    "last_slot": slot,
                    "updated_at": utc_iso(),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
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
                # Windows can keep a process object accessible after exit while
                # another process still owns a handle. OpenProcess succeeding is
                # therefore insufficient for stale-lock detection.
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _companion_commands(self) -> dict[str, list[str]]:
        """Return canonical, secret-free commands for persistent companions."""

        venv_python = self.root / ".venv" / "Scripts" / "python.exe"
        python = str(venv_python if venv_python.is_file() else Path(sys.executable))
        main = str(self.root / "main.py")
        execution_exceptions = sorted(
            load_execution_market_exceptions(self.settings)
        )
        monitor_only_markets = sorted(
            self.settings.autonomous_live.monitor_only_markets
        )
        data_markets = sorted(
            set(execution_exceptions) | set(monitor_only_markets)
        )
        research_markets = tuple(
            dict.fromkeys((*self.markets, *monitor_only_markets))
        )
        data_sync_command = [
            python,
            main,
            "data",
            "sync",
            "--providers",
            "all",
            "--history-profile",
            "maximum",
            "--universe-size",
            "50",
            # Live-critical chains stay fresh first.  Slower research
            # timeframes are causally resampled/backfilled elsewhere and
            # must not make a 15-minute entry feed stale.
            "--timeframes",
            "15m,1h,4h,1d",
            "--resume",
            "--yes",
            "--continuous",
            "--context",
            "none",
            "--interval-seconds",
            "300",
        ]
        if data_markets:
            data_sync_command.extend(
                ["--extra-markets", ",".join(data_markets)]
            )
        return {
            "data_sync": data_sync_command,
            "simple_lab": [
                python,
                main,
                "simple-lab",
                "run",
                "--continuous",
                "--generation-batch-size",
                "100",
                "--backtest-batch-size",
                "8",
                "--timeframes",
                "15m,1h,2h,4h,1d,1W",
                "--markets",
                ",".join(research_markets),
                "--rows",
                "1000",
                "--minimum-exact-history-days",
                "365",
                "--max-markets-per-exact-cycle",
                "1",
                "--workers",
                "1",
                "--max-trials",
                "4",
                "--history-mode",
                "bounded",
                "--interval-seconds",
                "300",
                "--resume",
            ],
        }

    def _simple_lab_lock_path(self) -> Path:
        return (
            self.settings.paths.output_dir
            / "research"
            / "simple_strategy_lab"
            / "service.lock"
        )

    def _inspect_simple_lab_lock(self) -> dict[str, Any]:
        lock_path = self._simple_lab_lock_path()
        if not lock_path.is_file():
            return {
                "available": True,
                "exists": False,
                "stale": False,
                "reason_code": "LOCK_AVAILABLE",
                "owner": None,
                "lock_path": str(lock_path),
            }
        try:
            owner = dict(read_json(lock_path))
            pid = int(owner["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {
                "available": True,
                "exists": True,
                "stale": True,
                "reason_code": "INVALID_LOCK_METADATA_RECOVERABLE",
                "owner": None,
                "lock_path": str(lock_path),
            }
        alive = self._pid_alive(pid)
        return {
            "available": not alive,
            "exists": True,
            "stale": not alive,
            "reason_code": (
                "LOCK_HELD_BY_LIVE_PROCESS"
                if alive
                else "STALE_PROCESS_LOCK_RECOVERABLE"
            ),
            "owner": owner,
            "lock_path": str(lock_path),
        }

    def _archive_simple_lab_stale_lock(
        self,
        inspection: dict[str, Any],
    ) -> dict[str, Any]:
        lock_path = self._simple_lab_lock_path()
        if not inspection.get("exists"):
            return inspection | {"recovered": False}
        if not inspection.get("available") or not inspection.get("stale"):
            raise RuntimeError("SIMPLE_LAB_LIVE_LOCK_CANNOT_BE_RECOVERED")
        resolved = lock_path.resolve()
        if not resolved.is_relative_to(self.root):
            raise RuntimeError("SIMPLE_LAB_LOCK_OUTSIDE_PROJECT_ROOT")
        archive = lock_path.with_name(
            f"{lock_path.name}.stale.{time.time_ns()}"
        )
        os.replace(lock_path, archive)
        return inspection | {
            "recovered": True,
            "archive_path": str(archive),
            "reason_code": "STALE_LOCK_ARCHIVED",
        }

    def _companion_process_status(self, name: str) -> dict[str, Any]:
        if name == "data_sync":
            lock_path = self.settings.paths.checkpoints_dir / "data_service.lock"
            inspection = ContinuousDataService.inspect_lock_path(lock_path) | {
                "lock_path": str(lock_path)
            }
        elif name == "simple_lab":
            inspection = self._inspect_simple_lab_lock()
        else:
            raise ValueError(f"unknown companion service: {name}")

        owner = dict(inspection.get("owner") or {})
        owner_pid = int(owner.get("pid") or 0)
        if owner_pid > 0 and self._pid_alive(owner_pid):
            self._companion_spawned.pop(name, None)
            return {
                "status": "RUNNING",
                "pid": owner_pid,
                "lock": inspection,
            }

        spawned_pid = int(self._companion_spawned.get(name) or 0)
        if spawned_pid > 0 and self._pid_alive(spawned_pid):
            return {
                "status": "STARTING",
                "pid": spawned_pid,
                "lock": inspection,
            }
        self._companion_spawned.pop(name, None)

        if inspection.get("exists") and inspection.get("stale"):
            if name == "data_sync":
                recovery = ContinuousDataService.recover_stale_lock_path(
                    Path(str(inspection["lock_path"]))
                )
            else:
                recovery = self._archive_simple_lab_stale_lock(inspection)
            inspection = inspection | {"recovery": recovery}
        return {
            "status": "STOPPED",
            "pid": None,
            "lock": inspection,
        }

    def _spawn_companion(self, name: str, command: list[str]) -> int:
        log_directory = self.settings.paths.output_dir / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        stdout_path = log_directory / f"companion_{name}.out.log"
        stderr_path = log_directory / f"companion_{name}.err.log"
        creationflags = 0
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            popen_options["start_new_session"] = True
        with (
            stdout_path.open("a", encoding="utf-8") as stdout,
            stderr_path.open("a", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                close_fds=True,
                **popen_options,
            )
        self._companion_spawned[name] = process.pid
        return int(process.pid)

    def _ensure_companion_services(
        self,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Keep data and orderless research alive without affecting execution."""

        now_monotonic = time.monotonic()
        if (
            not force
            and now_monotonic - self._last_companion_check_monotonic < 60.0
            and self.companion_status_path.is_file()
        ):
            return dict(read_json(self.companion_status_path))
        self._last_companion_check_monotonic = now_monotonic
        services: dict[str, Any] = {}
        for name, command in self._companion_commands().items():
            try:
                status = self._companion_process_status(name)
                if status["status"] == "STOPPED":
                    pid = self._spawn_companion(name, command)
                    status = status | {
                        "status": "STARTING",
                        "pid": pid,
                        "started_by_live_supervisor": True,
                    }
                    self._event(
                        "health",
                        {
                            "event": "COMPANION_SERVICE_STARTED",
                            "service": name,
                            "pid": pid,
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                else:
                    status["started_by_live_supervisor"] = False
                services[name] = status
            except Exception as exc:
                services[name] = {
                    "status": "FAILED_TO_SUPERVISE",
                    "exception_type": type(exc).__name__,
                    "execution_affected": False,
                }
                self._event(
                    "errors",
                    {
                        "event": "COMPANION_SERVICE_SUPERVISION_FAILED",
                        "service": name,
                        "exception_type": type(exc).__name__,
                        "execution_affected": False,
                    },
                )
        payload = {
            "schema_version": "autonomous_live_companions_v1",
            "checked_at": utc_iso(),
            "services": services,
            "orders_generated": 0,
            "orders_submitted": 0,
            "execution_affected_by_companion_failure": False,
        }
        atomic_write_json(self.companion_status_path, payload)
        return payload

    def _acquire(self) -> None:
        if self.lock_path.is_file():
            existing = dict(read_json(self.lock_path))
            pid = int(existing.get("pid") or 0)
            if self._pid_alive(pid):
                raise AutonomousLiveLockError(
                    f"autonomous-live already active for pid={pid}"
                )
            self.lock_path.unlink(missing_ok=True)
        token = stable_hash([os.getpid(), utc_iso()], length=40)
        descriptor = os.open(
            self.lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "token": token,
                    "acquired_at": utc_iso(),
                },
                handle,
            )
        self._lock_token = token

    def _release(self) -> None:
        if not self._lock_token or not self.lock_path.is_file():
            return
        payload = dict(read_json(self.lock_path))
        if payload.get("token") == self._lock_token:
            self.lock_path.unlink(missing_ok=True)
        self._lock_token = None

    def _control_state(self) -> str:
        if not self.state_path.is_file():
            return "DISABLED"
        return str(read_json(self.state_path).get("state") or "DISABLED")

    def _write_control_state(self, state: str, *, reason: str) -> dict[str, Any]:
        payload = {
            "schema_version": "autonomous_live_control_v1",
            "state": state,
            "reason": reason,
            "updated_at": utc_iso(),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(self.state_path, payload)
        return payload

    async def enable(
        self,
        *,
        markets: Iterable[str],
        approval: str,
    ) -> dict[str, Any]:
        normalized = tuple(
            dict.fromkeys(
                str(market).strip().upper().replace("/", "-")
                for market in markets
            )
        )
        if approval.strip() != APPROVAL_PHRASE:
            raise PermissionError("autonomous-live approval phrase does not match")
        if set(normalized) != set(self.markets):
            raise PermissionError(
                "autonomous-live launch markets do not match the "
                "validated dynamic universe"
            )
        account = await live_account_health(
            self.settings,
            markets=normalized,
        )
        market_check = await live_asset_preflight(
            self.settings,
            markets=normalized,
        )
        if account.get("status") != "READY":
            return {
                "status": "BLOCKED",
                "reason_code": "PRIVATE_LIVE_PREFLIGHT_FAILED",
                "account_failures": account.get("failures", []),
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        rows = list(market_check.get("markets") or market_check.get("rows") or [])
        availability = {
            str(row.get("market")): (
                "AVAILABLE"
                if row.get("venue_available")
                else "UNAVAILABLE"
            )
            for row in rows
        }
        authority = {
            "schema_version": "autonomous_live_authority_v1",
            "active": True,
            "enabled_at": utc_iso(),
            "markets": list(normalized),
            "availability": availability,
            "approval_reference": "operator_live_spot_confirmed",
            "approval_phrase_stored": False,
            "service_authority_is_strategy_order_authority": False,
            "unknown_strategy_dna_auto_live_promotion": False,
            "spot_only": True,
            "withdrawals": False,
            "margin": False,
            "leverage": False,
            "shorting": False,
        }
        atomic_write_json(self.authority_path, authority)
        self._write_control_state("ENABLED", reason="OPERATOR_CONFIRMED")
        startup = {
            "schema_version": "autonomous_live_startup_v1",
            "status": "ENABLED",
            "enabled_at": authority["enabled_at"],
            "markets": list(normalized),
            "availability": availability,
            "account_health": account["status"],
            "reconciliation": account.get("reconciliation"),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(self.output / "startup_report.json", startup)
        self._event("health", {"event": "AUTONOMOUS_LIVE_ENABLED", **startup})
        return startup

    def pause(self) -> dict[str, Any]:
        return self._write_control_state(
            "PAUSED",
            reason="OPERATOR_PAUSE_NEW_ENTRIES",
        )

    def _automatic_pause(
        self,
        reason: str,
        *,
        recoverable: bool,
    ) -> dict[str, Any]:
        """Pause entries without impersonating an operator action."""

        # Shutdown is an operator-owned terminal request for this process.
        # A concurrent task failure must never downgrade it back to PAUSED,
        # otherwise the heartbeat no longer observes the shutdown request and
        # the old worker can survive a deployment restart indefinitely.
        if self._control_state() == "SHUTDOWN_REQUESTED":
            return dict(read_json(self.state_path))
        self._consecutive_healthy_reconciliations = 0
        category = "RECOVERABLE" if recoverable else "NONRECOVERABLE"
        return self._write_control_state(
            "PAUSED",
            reason=f"AUTO_{category}_{reason}",
        )

    def _observe_reconciliation_health(self, *, ready: bool) -> bool:
        """Recover only transient reconciliation pauses after three passes."""

        if not ready:
            self._consecutive_healthy_reconciliations = 0
            return False
        self._consecutive_healthy_reconciliations += 1
        if self._consecutive_healthy_reconciliations < 3:
            return False
        state = dict(read_json(self.state_path)) if self.state_path.is_file() else {}
        if state.get("state") != "PAUSED" or not str(
            state.get("reason") or ""
        ).startswith("AUTO_RECOVERABLE_"):
            return False

        # Reconciliation health is necessary but never allowed to override a
        # persisted kill switch.  True operator and non-recoverable pauses are
        # deliberately excluded above.
        from risk.risk_manager import KillSwitch

        kill_switch = KillSwitch(
            self.settings.paths.checkpoints_dir / "kill_switch.json"
        )
        if kill_switch.active:
            return False
        self._write_control_state(
            "ENABLED",
            reason="AUTO_RECOVERED_AFTER_3_HEALTHY_RECONCILIATIONS",
        )
        self._event(
            "health",
            {
                "event": "AUTOMATIC_ENTRY_PAUSE_RECOVERED",
                "healthy_reconciliations": (
                    self._consecutive_healthy_reconciliations
                ),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return True

    def resume(self) -> dict[str, Any]:
        authority = (
            dict(read_json(self.authority_path))
            if self.authority_path.is_file()
            else {}
        )
        if authority.get("active") is not True:
            raise PermissionError("autonomous-live authority is not active")
        return self._write_control_state("ENABLED", reason="OPERATOR_RESUME")

    def shutdown(self) -> dict[str, Any]:
        return self._write_control_state(
            "SHUTDOWN_REQUESTED",
            reason="OPERATOR_SHUTDOWN",
        )

    async def shutdown_bounded(
        self,
        *,
        timeout_seconds: float = 40.0,
    ) -> dict[str, Any]:
        """Request shutdown and safely bound a wedged Windows runtime.

        The normal in-process path drains tasks and closes both WebSockets.
        If the event loop is wedged before observing the request, this
        operator-side fallback may terminate only the exact locked process
        tree and only after a fresh reconciliation proves that there are no
        exchange orders or strategy-managed positions.  Any uncertainty is
        fail-closed and leaves the runtime alive for protective management.
        """

        request = self.shutdown()
        lock = dict(read_json(self.lock_path)) if self.lock_path.is_file() else {}
        pid = max(0, int(lock.get("pid") or 0))
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        while pid and self._pid_alive(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if not pid or not self._pid_alive(pid):
            return {
                **request,
                "status": "STOPPED",
                "pid": pid or None,
                "bounded_timeout_seconds": timeout_seconds,
                "forced_termination": False,
            }

        account = await live_account_health(
            self.settings,
            markets=self.markets,
        )
        reconciliation = dict(account.get("reconciliation") or {})
        position_path = (
            self.settings.paths.output_dir / "reports" / "current_position.json"
        )
        position_state = (
            dict(read_json(position_path)) if position_path.is_file() else {}
        )
        managed_position = position_state.get("position")
        safe_to_terminate = (
            account.get("status") == "READY"
            and reconciliation.get("healthy") is True
            and int(reconciliation.get("remote_open_orders") or 0) == 0
            and int(reconciliation.get("local_open_orders") or 0) == 0
            and not managed_position
        )
        diagnostics_path = self.output / "shutdown_diagnostics.json"
        if not safe_to_terminate:
            payload = {
                **request,
                "status": "SHUTDOWN_BLOCKED_ACTIVE_OR_UNCERTAIN_RISK_STATE",
                "pid": pid,
                "bounded_timeout_seconds": timeout_seconds,
                "forced_termination": False,
                "account_status": account.get("status"),
                "reconciliation": reconciliation,
                "managed_position_present": bool(managed_position),
                "protective_runtime_preserved": True,
            }
            atomic_write_json(diagnostics_path, payload)
            return payload

        termination_status = "UNSUPPORTED_PLATFORM"
        if os.name == "nt":
            completed = await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            termination_status = (
                "TERMINATED" if completed.returncode == 0 else "TASKKILL_FAILED"
            )
        else:
            os.kill(pid, signal.SIGTERM)
            termination_status = "TERMINATE_REQUESTED"
        await asyncio.sleep(0.5)
        stopped = not self._pid_alive(pid)
        payload = {
            **request,
            "status": (
                "STOPPED_BY_BOUNDED_SAFE_FALLBACK"
                if stopped
                else "SHUTDOWN_FALLBACK_FAILED"
            ),
            "pid": pid,
            "bounded_timeout_seconds": timeout_seconds,
            "forced_termination": True,
            "termination_status": termination_status,
            "fresh_reconciliation": reconciliation,
            "managed_position_present": False,
            "protective_runtime_preserved": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(diagnostics_path, payload)
        return payload

    async def reconcile(self) -> dict[str, Any]:
        result = await live_account_health(
            self.settings,
            markets=self.markets,
        )
        self._event(
            "positions",
            {
                "event": "ACCOUNT_RECONCILIATION",
                "status": result.get("status"),
                "account": result.get("account"),
                "reconciliation": result.get("reconciliation"),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        if result.get("status") != "READY":
            self._event(
                "errors",
                {
                    "event": "RECONCILIATION_REQUIRED",
                    "failures": result.get("failures", []),
                },
            )
        return result

    def positions(self) -> dict[str, Any]:
        account_path = (
            self.settings.paths.output_dir
            / "operations"
            / "live_account_health.json"
        )
        account = dict(read_json(account_path)) if account_path.is_file() else {}
        return {
            "status": "READY" if account else "NOT_RECONCILED",
            "exchange_account": account.get("account"),
            "local_tracker": self.position_tracker.snapshot(),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def orders(self, *, limit: int = 50) -> dict[str, Any]:
        rows = _tail_jsonl(self.events / "orders.jsonl", limit=limit)
        fills = _tail_jsonl(self.events / "fills.jsonl", limit=limit)
        return {
            "status": "READY",
            "order_event_count": len(rows),
            "fill_event_count": len(fills),
            "orders": rows,
            "fills": fills,
            "exchange_identifiers_masked": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def signals(self, *, limit: int = 50) -> dict[str, Any]:
        rows = _tail_jsonl(self.events / "signals.jsonl", limit=limit)
        return {
            "status": "READY",
            "count": len(rows),
            "signals": rows,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def strategies(self) -> dict[str, Any]:
        report = reclassify_existing_strategies(self.root, self.settings)
        accounting = rebuild_live_strategy_accounting(self.root)
        return {
            "status": "READY",
            "governance": governance_status(self.root),
            "research_positive": report.get("research_positive"),
            "paper_active": report.get("paper_active"),
            "live_canary_eligible": report.get("live_canary_eligible"),
            "live_canary_active": report.get("live_canary_active"),
            "live_strategy_accounting": accounting,
            "auto_live_promotion": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def research_status(self) -> dict[str, Any]:
        return {
            "status": "READY",
            "autopilot": self.autopilot.status(),
            "lab_worker": (
                read_json(self.settings.paths.lab_dir / "state" / "worker_status.json")
                if (self.settings.paths.lab_dir / "state" / "worker_status.json").is_file()
                else {"status": "NOT_RUNNING"}
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def status(self) -> dict[str, Any]:
        stored = dict(read_json(self.status_path)) if self.status_path.is_file() else {}
        lock = dict(read_json(self.lock_path)) if self.lock_path.is_file() else {}
        pid = int(lock.get("pid") or 0)
        process_running = self._pid_alive(pid)
        private_runtime = (
            stored.get("private_account_websocket")
            if process_running
            else (
                self.private_account_stream.health()
                if self.private_account_stream is not None
                else stored.get("private_account_websocket")
            )
        )
        ticker_path = (
            self.settings.paths.output_dir
            / "universe"
            / "bitvavo_eur_ticker_universe.json"
        )
        mover_path = (
            self.settings.paths.output_dir
            / "universe"
            / "dynamic_intensive_movers.json"
        )
        ticker_artifact = (
            dict(read_json(ticker_path)) if ticker_path.is_file() else {}
        )
        mover_artifact = (
            dict(read_json(mover_path)) if mover_path.is_file() else {}
        )
        fast_data_path = (
            self.settings.paths.output_dir
            / "active_trading"
            / "data_health.json"
        )
        fast_data_health = (
            dict(read_json(fast_data_path))
            if fast_data_path.is_file()
            else {"status": "NOT_YET_MEASURED"}
        )
        account_health_path = (
            self.settings.paths.output_dir
            / "operations"
            / "live_account_health.json"
        )
        account_health = (
            dict(read_json(account_health_path))
            if account_health_path.is_file()
            else {
                "entry_allowed": False,
                "entry_blockers": ["ACCOUNT_HEALTH_NOT_AVAILABLE"],
                "failures": ["ACCOUNT_HEALTH_NOT_AVAILABLE"],
                "risk_reduction_allowed": False,
            }
        )
        execution_authority = build_execution_authority_matrix(
            self.settings,
            account_health=account_health,
        )
        reported_ticker_markets = (
            tuple(
                str(row.get("market") or "")
                for row in ticker_artifact.get("rows") or []
                if row.get("realtime_ticker") is True and row.get("market")
            )
            if process_running and ticker_artifact.get("status") in {"READY", "DEGRADED_CACHED_UNIVERSE"}
            else self.ticker_tracking_markets
        )
        reported_orderflow_markets = (
            tuple(mover_artifact.get("selected_markets") or [])
            if process_running and mover_artifact.get("selected_markets")
            else self.orderflow_markets
        )
        return {
            **stored,
            "control_state": self._control_state(),
            "process_running": process_running,
            "pid": pid or None,
            "authority_active": (
                bool(read_json(self.authority_path).get("active"))
                if self.authority_path.is_file()
                else False
            ),
            "markets": list(self.markets),
            "orderflow_tracking_markets": list(reported_orderflow_markets),
            "orderflow_tracking_market_count": len(reported_orderflow_markets),
            "ticker_tracking_markets": list(reported_ticker_markets),
            "ticker_tracking_market_count": len(reported_ticker_markets),
            "dynamic_mover_tracking": (
                mover_artifact
                if mover_artifact
                else {"status": "AWAITING_TICKER_HISTORY"}
            ),
            # Overall process/stream liveness must not conceal stale strategy
            # inputs.  This remains dependency-scoped: a stale BCH 15m bar,
            # for example, blocks BCH/15m entries without stopping ETH risk
            # management or otherwise healthy markets.
            "fast_market_data_health": fast_data_health,
            "execution_authority": execution_authority,
            "orderflow_stream": (
                dict(
                    read_json(
                        self.settings.paths.output_dir
                        / "operations"
                        / "orderflow_stream_health.json"
                    )
                )
                if (
                    self.settings.paths.output_dir
                    / "operations"
                    / "orderflow_stream_health.json"
                ).is_file()
                else {"status": "NOT_STARTED"}
            ),
            "private_account_websocket": (
                private_runtime
                if private_runtime is not None
                else {
                    "provider": "bitvavo",
                    "channel": "account",
                    "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                    "ready_for_new_entries": False,
                    "secrets_serialized": False,
                }
            ),
            "event_streams": {
                stream: str(self.events / f"{stream}.jsonl")
                for stream in EVENT_STREAMS
            },
            "execution_evidence": (
                _execution_evidence_summary(
                    self.settings.paths.output_dir
                    / "operations"
                    / "execution_evidence_layers.json"
                )
            ),
            "strategy_evidence_watch": _strategy_evidence_watch_summary(
                self.settings.paths.output_dir
                / "operations"
                / "strategy_evidence_watch.json"
            ),
        }

    def health(self) -> dict[str, Any]:
        status = self.status()
        heartbeat = (
            dict(read_json(self.heartbeat_path))
            if self.heartbeat_path.is_file()
            else {}
        )
        public_stream = dict(status.get("websocket") or {})
        private_stream = dict(status.get("private_account_websocket") or {})
        process_running = bool(status.get("process_running"))
        control_runnable = status.get("control_state") in {"ENABLED", "PAUSED"}
        public_ready = public_stream.get("state") == "CONNECTED"
        private_ready = bool(private_stream.get("ready_for_new_entries"))
        fast_data_health = dict(status.get("fast_market_data_health") or {})
        health_status = (
            "NOT_RUNNING"
            if not process_running or not control_runnable
            else "HEALTHY"
            if public_ready and private_ready
            else "DEGRADED"
        )
        return {
            "status": health_status,
            "control_state": status.get("control_state"),
            "process_running": status.get("process_running"),
            "heartbeat": heartbeat,
            "websocket": public_stream,
            "private_account_websocket": private_stream,
            "orderflow_stream": status.get("orderflow_stream"),
            "public_stream_ready": public_ready,
            "private_stream_ready": private_ready,
            "fast_market_data_health": fast_data_health,
            "fast_market_data_dependency_scoped": bool(
                fast_data_health.get("dependency_scoped_fail_closed")
            ),
            "execution_evidence": status.get("execution_evidence"),
            "strategy_evidence_watch": status.get(
                "strategy_evidence_watch"
            ),
            "execution_authority": status.get("execution_authority"),
            "latest_reconciliation": status.get("latest_reconciliation"),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    async def _websocket_consumer(self) -> None:
        async for event in self.websocket.events():
            if self._stop.is_set():
                return
            payload = event.model_dump(mode="python")
            payload["event_type"] = event.event_type.value
            payload["timestamp"] = event.timestamp.isoformat()
            payload["observed_at"] = event.observed_at.isoformat()
            interval = str(event.payload.get("interval") or "5m")
            try:
                timeframe = normalize_timeframe(interval)
            except ValueError:
                timeframe = interval
            bucket = int(event.timestamp.timestamp()) // 300
            signal_id = stable_hash(
                [
                    "MARKET_MONITOR",
                    event.canonical_market,
                    timeframe,
                    event.event_type.value,
                    bucket,
                ],
                length=40,
            )
            if (
                event.canonical_market in self.markets
                and event.event_type.value == "candle"
                and execution_timeframe_allowed(timeframe)
            ):
                self._record_signal_once(
                    {
                        "signal_id": signal_id,
                        "timestamp_utc": event.observed_at.isoformat(),
                        "market": event.canonical_market,
                        "timeframe": timeframe,
                        "strategy_id": "MARKET_MONITOR",
                        "strategy_version": "1",
                        "signal": "NO_ENTRY",
                        "confidence": 0.0,
                        "regime": "UNCLASSIFIED",
                        "entry_price_reference": None,
                        "proposed_stop": None,
                        "proposed_target": None,
                        "risk_reward": None,
                        "position_size_eur": "0",
                        "risk_amount_eur": "0",
                        "spread_bps": None,
                        "estimated_slippage_bps": None,
                        "data_freshness": (
                            "LIVE_WEBSOCKET_INTRABAR"
                        ),
                        "reason_codes": [
                            (
                                "MONITORING_CANDLE_UPDATE"
                            )
                        ],
                        "blocking_reasons": ["NO_APPROVED_STRATEGY_CONSENSUS"],
                        "execution_status": "NOT_ACTIONABLE",
                    },
                )
            if event.event_type.value == "CONNECTION_STATUS":
                self._event("health", {"event": "WEBSOCKET_STATUS", **payload})
                if str(event.payload.get("status") or "").upper() == "CONNECTED":
                    reconciliation = await self.reconcile()
                    if reconciliation.get("status") != "READY":
                        self._automatic_pause(
                            "PUBLIC_STREAM_RECONNECT_RECONCILIATION_FAILED",
                            recoverable=True,
                        )

    async def _private_account_consumer(self) -> None:
        """Persist sanitized order/fill events and reconcile immediately."""

        if self.private_account_stream is None:
            return
        async for event in self.private_account_stream.events():
            if self._stop.is_set():
                return
            stream = "fills" if event.get("event") == "FILL" else "orders"
            self._event(
                stream,
                {
                    "event": f"BITVAVO_ACCOUNT_{event.get('event')}",
                    "payload": event,
                    "exchange_identifiers_masked": True,
                },
            )
            status = str(event.get("status") or "").replace("_", "").upper()
            remaining = self._decimal(event.get("amount_remaining"))
            filled = self._decimal(event.get("filled_amount"))
            if event.get("event") == "FILL":
                notification_type = (
                    "LIVE_ORDER_FILLED"
                    if remaining <= 0
                    else "LIVE_ORDER_PARTIALLY_FILLED"
                )
            elif status == "FILLED":
                notification_type = "LIVE_ORDER_FILLED"
            elif status in {"PARTIALLYFILLED", "PARTIAL"} or filled > 0:
                notification_type = "LIVE_ORDER_PARTIALLY_FILLED"
            elif status in {"CANCELED", "CANCELLED"}:
                notification_type = "LIVE_ORDER_CANCELLED"
            elif status in {"REJECTED", "EXPIRED"}:
                notification_type = "LIVE_ORDER_REJECTED"
            else:
                notification_type = "LIVE_ORDER_SUBMITTED"
            try:
                execution_state_path = (
                    self.settings.paths.output_dir
                    / "live"
                    / "event_driven_execution_state.json"
                )
                execution_state = (
                    dict(read_json(execution_state_path))
                    if execution_state_path.is_file()
                    else {}
                )
                managed = next(
                    (
                        dict(row)
                        for row in (execution_state.get("positions") or {}).values()
                        if str(row.get("market") or "")
                        == str(event.get("market") or "")
                    ),
                    {},
                )
                quantity = self._decimal(
                    event.get("filled_amount") or managed.get("quantity")
                )
                entry = self._decimal(
                    event.get("fill_price") or managed.get("entry_price")
                )
                stop = self._decimal(managed.get("stop_loss"))
                remaining_risk = quantity * abs(entry - stop)
                flow = dict(managed.get("realtime_inputs") or {})
                await asyncio.to_thread(
                    self.notifier.notify_order_event,
                    notification_type,
                    {
                        **event,
                        "requested_quantity": event.get("amount"),
                        "filled_quantity": event.get("filled_amount"),
                        "remaining_quantity": event.get("amount_remaining"),
                        "average_fill_price": event.get("fill_price"),
                        "verification_source": (
                            "BITVAVO_PRIVATE_ACCOUNT_STREAM"
                        ),
                        "strategy_id": managed.get("playbook_id"),
                        "timeframe": managed.get("context_timeframe"),
                        "stop_loss": managed.get("stop_loss"),
                        "take_profit_1": managed.get("take_profit_1"),
                        "take_profit_2": managed.get("take_profit_2"),
                        "regime": managed.get("macro_regime"),
                        "orderflow_status": (
                            f"buy={flow.get('taker_buy_ratio_1m')}; "
                            f"cvd={flow.get('cvd_quote_eur_1m')}; "
                            f"ofi={flow.get('ofi_1m')}"
                            if flow
                            else None
                        ),
                        "estimated_slippage_bps": flow.get(
                            "estimated_buy_slippage_bps"
                        ),
                        "remaining_risk_eur": str(remaining_risk),
                    },
                )
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "TELEGRAM_PRIVATE_ORDER_NOTIFICATION_FAILURE",
                        "exception_type": type(exc).__name__,
                        "execution_affected": False,
                    },
                )
            reconciliation = await self.reconcile()
            if reconciliation.get("status") != "READY":
                self._automatic_pause(
                    "PRIVATE_EVENT_RECONCILIATION_FAILED",
                    recoverable=True,
                )

    async def _reconciliation_loop(self) -> None:
        interval = self.settings.autonomous_live.reconciliation_seconds
        while not self._stop.is_set():
            try:
                result = await self.reconcile()
                if result.get("status") != "READY":
                    self._automatic_pause(
                        "PERIODIC_RECONCILIATION_FAILED",
                        recoverable=True,
                    )
                else:
                    self._observe_reconciliation_health(ready=True)
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "RECONCILIATION_EXCEPTION",
                        "exception_type": type(exc).__name__,
                    },
                )
                self._automatic_pause(
                    "PERIODIC_RECONCILIATION_EXCEPTION",
                    recoverable=True,
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    def _record_generated_live_cycle(
        self,
        result: Mapping[str, Any],
        generated_live: Mapping[str, Any],
    ) -> None:
        if not generated_live:
            return
        generated_entry = dict(
            generated_live.get("selected_entry")
            or (generated_live.get("ranked_natural_entries") or [{}])[0]
            or {}
        )
        generated_signal_id = str(
            generated_entry.get("signal_id")
            or stable_hash(
                [
                    "GENERATED_LIVE_CYCLE",
                    generated_live.get("reason_code"),
                    generated_live.get("status"),
                ],
                length=40,
            )
        )
        self._record_signal_once(
            {
                "signal_id": generated_signal_id,
                "timestamp_utc": result.get("finished_at"),
                "market": generated_entry.get("market"),
                "timeframe": generated_entry.get("timeframe"),
                "strategy_id": generated_entry.get("strategy_id")
                or "EXACT_POSITIVE_PORTFOLIO",
                "strategy_version": "frozen",
                "signal": (
                    "ENTER"
                    if generated_live.get("orders_submitted")
                    else "NO_ENTRY"
                ),
                "confidence": generated_entry.get("score"),
                "regime": None,
                "entry_price_reference": None,
                "proposed_stop": None,
                "proposed_target": None,
                "risk_reward": None,
                "position_size_eur": str(
                    (generated_live.get("authority") or {}).get(
                        "maximum_order_eur",
                        self.settings.execution.maximum_live_order_eur,
                    )
                ),
                "risk_amount_eur": None,
                "spread_bps": (
                    generated_live.get("entry_liquidity") or {}
                ).get("spread_bps"),
                "estimated_slippage_bps": (
                    generated_live.get("entry_liquidity") or {}
                ).get("estimated_slippage_bps"),
                "data_freshness": "CANONICAL_CLOSED_CANDLE",
                "reason_codes": [generated_live.get("reason_code")],
                "blocking_reasons": (
                    []
                    if generated_live.get("orders_submitted")
                    else [generated_live.get("reason_code")]
                ),
                "execution_status": generated_live.get("status"),
            },
        )
        if generated_live.get("orders_submitted"):
            self._event(
                "orders",
                {
                    "event": "GENERATED_LIVE_ORDER",
                    **generated_live,
                },
            )

    async def _execution_loop(self) -> None:
        interval = self.settings.autonomous_live.execution_cycle_seconds
        while not self._stop.is_set():
            control_state = self._control_state()
            if control_state in {"ENABLED", "PAUSED"}:
                try:
                    private_stream_ready = bool(
                        self.private_account_stream is not None
                        and self.private_account_stream.ready
                    )
                    public_stream_ready = (
                        self.websocket.health("bitvavo").get("state")
                        == "CONNECTED"
                    )
                    result = await self.autopilot.run_once(
                        run_research=False,
                        allow_live_new_entries=(
                            control_state == "ENABLED"
                            and private_stream_ready
                            and public_stream_ready
                        ),
                    )
                    live = dict(result.get("stages", {}).get("live_canary") or {})
                    opportunity = dict(live.get("natural_signal") or {})
                    liquidity = dict(live.get("entry_liquidity") or {})
                    signal_id = str(
                        opportunity.get("opportunity_id")
                        or stable_hash(
                            [
                                "CANONICAL_LIVE_CYCLE",
                                live.get("reason_code"),
                                "ETH-EUR",
                                "1d",
                            ],
                            length=40,
                        )
                    )
                    blockers = list(opportunity.get("blockers") or [])
                    if not private_stream_ready:
                        blockers.append("PRIVATE_ACCOUNT_STREAM_NOT_READY")
                    if not public_stream_ready:
                        blockers.append("PUBLIC_MARKET_STREAM_NOT_READY")
                    if not live.get("orders_submitted") and live.get("reason_code"):
                        blockers.append(live["reason_code"])
                    self._record_signal_once(
                        {
                            "signal_id": signal_id,
                            "timestamp_utc": result.get("finished_at"),
                            "market": opportunity.get("market") or "ETH-EUR",
                            "timeframe": opportunity.get("timeframe") or "1d",
                            "strategy_id": (
                                opportunity.get("strategy_id")
                                or "RR_B60_H5_Z20"
                            ),
                            "strategy_version": "frozen",
                            "signal": (
                                "ENTER"
                                if live.get("orders_submitted")
                                else "AVOID"
                                if opportunity.get("action") == "BUY"
                                else "NO_ENTRY"
                            ),
                            "confidence": opportunity.get("confidence"),
                            "regime": opportunity.get("regime_fit"),
                            "entry_price_reference": opportunity.get("entry_price"),
                            "proposed_stop": opportunity.get("stop_loss"),
                            "proposed_target": opportunity.get("take_profit_1"),
                            "risk_reward": opportunity.get("reward_risk"),
                            "position_size_eur": (
                                live.get("canary_limits") or {}
                            ).get("maximum_order_eur"),
                            "risk_amount_eur": live.get("planned_risk_eur"),
                            "spread_bps": liquidity.get("spread_bps"),
                            "estimated_slippage_bps": liquidity.get(
                                "estimated_buy_slippage_bps"
                            ),
                            "data_freshness": "CANONICAL_CLOSED_CANDLE",
                            "reason_codes": [live.get("reason_code")],
                            "blocking_reasons": list(dict.fromkeys(blockers)),
                            "execution_status": live.get("cycle_status"),
                        },
                    )
                    if live.get("orders_submitted"):
                        self._event("orders", {"event": "LIVE_ORDER", **live})
                    generated_live = dict(
                        result.get("stages", {}).get(
                            "generated_strategy_live_portfolio"
                        )
                        or {}
                    )
                    self._record_generated_live_cycle(
                        result,
                        generated_live,
                    )
                    self._sync_canonical_execution_events()
                    await self._record_runtime_snapshots()
                except Exception as exc:
                    self._event(
                        "errors",
                        {
                            "event": "EXECUTION_CYCLE_EXCEPTION",
                            "exception_type": type(exc).__name__,
                        },
                    )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    async def _active_trading_scan_loop(self) -> None:
        """Refresh tactical views exactly once per closed scan slot.

        Tactical entries should not wait for the next 15-minute close, but
        evaluating on every execution tick would repeatedly inspect the same
        closed market state. Five-minute slotting preserves causality and
        refreshes the actionable universe without taking execution ownership
        from the canonical live engine.
        """

        interval_minutes = int(
            self.settings.autonomous_live.active_trading_scan_minutes
        )
        poll_interval = min(
            float(
                self.settings.autonomous_live.active_trading_poll_seconds
            ),
            float(self.settings.autonomous_live.execution_cycle_seconds),
            float(interval_minutes * 60),
        )
        while not self._stop.is_set():
            streams_ready = bool(
                self.websocket.health("bitvavo").get("state") == "CONNECTED"
                and self.orderflow_websocket.health("bitvavo").get("state")
                == "CONNECTED"
                and self.orderflow_recorder is not None
            )
            if (
                not streams_ready
                or time.monotonic() < self._active_scan_not_before_monotonic
            ):
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=poll_interval,
                    )
                except TimeoutError:
                    pass
                continue
            observed = datetime.now(UTC)
            scan_slot = observed.replace(
                minute=(observed.minute // interval_minutes)
                * interval_minutes,
                second=0,
                microsecond=0,
            )
            if self._last_active_trading_scan_slot != scan_slot:
                try:
                    active_scan = (
                        await self._run_active_trading_scan_isolated()
                    )
                    self._last_active_trading_scan_slot = scan_slot
                    self._event(
                        "research",
                        {
                            "event": "ACTIVE_TRADING_FULL_SCAN",
                            "scan_slot": scan_slot.isoformat(),
                            "scan_interval_minutes": interval_minutes,
                            "scan_poll_seconds": poll_interval,
                            "scan_maximum_rows": (
                                self.settings.autonomous_live
                                .active_trading_maximum_rows
                            ),
                            "status": active_scan.get("status"),
                            "reason": active_scan.get("reason"),
                            "regime": (
                                active_scan.get("macro") or {}
                            ).get("regime"),
                            "markets_scanned": active_scan.get(
                                "market_count"
                            ),
                            "evaluations": active_scan.get("evaluations"),
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                except Exception as scan_exc:
                    self._event(
                        "errors",
                        {
                            "event": "ACTIVE_TRADING_SCAN_EXCEPTION",
                            "exception_type": type(scan_exc).__name__,
                            "execution_affected": False,
                        },
                    )
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=poll_interval,
                )
            except TimeoutError:
                pass

    def _playbook_live_authorized(
        self,
        opportunity: Mapping[str, Any],
    ) -> bool:
        """Require explicit playbook-band authority separate from service auth."""

        path = self.root / "config" / "live_playbook_authority.json"
        if not path.is_file():
            return False
        payload = dict(read_json(path))
        if payload.get("active") is not True:
            return False
        return is_playbook_opportunity_authorized(payload, opportunity)

    def _event_driven_snapshot_scope(
        self,
        snapshot: Mapping[str, Any],
        tactical: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], set[str], set[str]]:
        """Select changed and warm markets while retaining BTC context.

        The recorder remains subscribed to every deep market.  This method
        only limits strategy recalculation; it never drops transport data or
        changes execution authority.
        """

        rows = [dict(row) for row in snapshot.get("markets") or []]
        warm_ranked = [
            dict(row)
            for row in tactical
            if str(row.get("status") or "")
            in {"ACTIONABLE", "NEAR_ENTRY"}
        ]
        warm_ranked.extend(
            dict(row)
            for row in self.opportunity_lifecycle.state.values()
            if row.get("warm_candidate") is True
            or (
                str(row.get("state") or "") in {"ARMED", "WATCHING"}
                and float(row.get("score") or 0.0) >= 65.0
            )
        )
        warm_ranked.sort(
            key=lambda row: float(row.get("score") or 0.0),
            reverse=True,
        )
        warm_markets = set(
            list(
                dict.fromkeys(
                    str(row.get("market") or "") for row in warm_ranked
                )
            )[:10]
        )
        warm_markets.discard("")
        first_evaluation = not self._realtime_market_fingerprints
        changed_markets: set[str] = set()
        current_fingerprints: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            market = str(row.get("market") or "")
            if not market:
                continue
            book = dict(row.get("book") or {})
            price = float(row.get("price") or 0.0)
            ofi = float(row.get("ofi_1m") or 0.0)
            depth = float(book.get("depth_imbalance_within_10_bps") or 0.0)
            spread_ok = bool(book.get("spread_within_dynamic_cap"))
            relative_volume = float(row.get("relative_volume_1m") or 0.0)
            fingerprint = (
                price,
                "HOSTILE" if ofi < -0.15 else "SUPPORTIVE" if ofi > 0.03 else "NEUTRAL",
                "HOSTILE"
                if depth < -0.25
                else "SUPPORTIVE"
                if depth > 0.03
                else "NEUTRAL",
                spread_ok,
                relative_volume >= 1.3,
            )
            previous = self._realtime_market_fingerprints.get(market)
            price_moved = bool(
                previous is not None
                and float(previous[0] or 0.0) > 0.0
                and abs(price / float(previous[0]) - 1.0) >= 0.0005
            )
            category_changed = bool(
                previous is not None and previous[1:] != fingerprint[1:]
            )
            important_event = first_evaluation or price_moved or category_changed
            if important_event:
                changed_markets.add(market)
                current_fingerprints[market] = fingerprint
            elif previous is not None:
                current_fingerprints[market] = previous
            else:
                current_fingerprints[market] = fingerprint
        self._realtime_market_fingerprints = current_fingerprints
        evaluated_markets = changed_markets | warm_markets
        context_markets = set(evaluated_markets)
        if evaluated_markets and any(
            row.get("market") == "BTC-EUR" for row in rows
        ):
            context_markets.add("BTC-EUR")
        scoped = {
            **dict(snapshot),
            "markets": [
                row for row in rows if row.get("market") in context_markets
            ],
            "event_driven_scope": {
                "deep_market_count": len(rows),
                "changed_market_count": len(changed_markets),
                "warm_market_count": len(warm_markets),
                "evaluated_market_count": len(evaluated_markets),
                "evaluated_markets": sorted(evaluated_markets),
                "btc_context_included": "BTC-EUR" in context_markets,
            },
        }
        return scoped, evaluated_markets, warm_markets

    def _write_near_entry_registry(self) -> None:
        lifecycle_rows = [
            {**dict(row), "source": "EVENT_DRIVEN_LIFECYCLE"}
            for row in self.opportunity_lifecycle.state.values()
            if str(row.get("state") or "") not in {
                "CLOSED",
                "INVALIDATED",
                "EXPIRED",
            }
            and (
                row.get("warm_candidate") is True
                or str(row.get("state") or "") in {"ARMED", "ENTRY_READY"}
            )
        ]
        active_path = (
            self.settings.paths.output_dir
            / "active_trading"
            / "opportunities.json"
        )
        active = (
            dict(read_json(active_path))
            if active_path.is_file()
            else {}
        )
        tactical_candidates = [
            dict(row)
            for row in active.get("all") or active.get("rows") or []
            if row.get("near_entry") is True
            or str(row.get("status") or "") in {"ACTIONABLE", "NEAR_ENTRY"}
        ]
        tactical_candidates.sort(
            key=lambda row: float(row.get("score") or 0.0),
            reverse=True,
        )
        best_tactical_by_market: dict[str, dict[str, Any]] = {}
        for row in tactical_candidates:
            market = str(row.get("market") or "")
            if market and market not in best_tactical_by_market:
                best_tactical_by_market[market] = {
                    **row,
                    "source": "TACTICAL_FULL_SCAN",
                    "execution_note": (
                        "Discovery candidate; realtime economics, "
                        "microstructure and authority still apply."
                    ),
                }
        lifecycle_markets = {
            str(row.get("market") or "") for row in lifecycle_rows
        }
        rows = lifecycle_rows + [
            row
            for market, row in best_tactical_by_market.items()
            if market not in lifecycle_markets
        ]
        rows.sort(
            key=lambda row: (
                str(row.get("state") or "") == "ENTRY_READY",
                float(row.get("score") or 0.0),
            ),
            reverse=True,
        )
        rows = rows[:24]
        atomic_write_json(
            self.output / "near_entry_registry.json",
            {
                "schema_version": "near_entry_registry_v1",
                "updated_at": utc_iso(),
                "count": len(rows),
                "markets": list(
                    dict.fromkeys(str(row.get("market")) for row in rows)
                ),
                "rows": rows,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )

    async def _event_driven_playbook_loop(self) -> None:
        """Evaluate prospective mover facts every second without candle delay."""

        while not self._stop.is_set():
            recorder = self.orderflow_recorder
            if recorder is None:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.25)
                except TimeoutError:
                    pass
                continue
            try:
                snapshot = await asyncio.to_thread(
                    recorder.realtime_snapshot,
                    markets=self.orderflow_markets,
                    # Cost, impact and depth checks must model the same
                    # notional that the event executor can actually submit.
                    # Using the former EUR 5 preview here understated the
                    # execution cost of the approved EUR 10 canary.
                    order_notional_eur=float(MAXIMUM_ORDER_EUR),
                )
                active_path = (
                    self.settings.paths.output_dir
                    / "active_trading"
                    / "opportunities.json"
                )
                active = (
                    dict(read_json(active_path))
                    if active_path.is_file()
                    else {}
                )
                tactical = list(active.get("all") or active.get("rows") or [])
                macro_path = (
                    self.settings.paths.output_dir
                    / "active_trading"
                    / "macro_crypto.json"
                )
                macro = (
                    dict(read_json(macro_path))
                    if macro_path.is_file()
                    else {}
                )
                regime = str(macro.get("regime") or "UNKNOWN")
                scoped_snapshot, evaluated_markets, warm_markets = (
                    self._event_driven_snapshot_scope(snapshot, tactical)
                )
                opportunities = (
                    await asyncio.to_thread(
                        build_event_driven_opportunities,
                        scoped_snapshot,
                        tactical_opportunities=tactical,
                        macro_regime=regime,
                        macro_context=macro,
                        maker_fee_bps=float(
                            self.settings.costs.maker_fee * 10_000
                        ),
                        taker_fee_bps=float(
                            self.settings.costs.taker_fee * 10_000
                        ),
                    )
                    if evaluated_markets
                    else []
                )
                opportunities = [
                    row
                    for row in opportunities
                    if str(row.get("market") or "") in evaluated_markets
                ]
                # One market move can match several related playbooks.  Only
                # the economically best representative enters lifecycle and
                # execution; alternatives remain recorded on the cluster.
                from core.opportunity_intelligence import (
                    deduplicate_opportunities,
                    score_shadow_opportunity,
                )

                opportunities = await asyncio.to_thread(
                    deduplicate_opportunities,
                    opportunities,
                )
                for index, opportunity in enumerate(opportunities):
                    opportunity["ml_shadow"] = score_shadow_opportunity(
                        self.settings,
                        opportunity,
                    )
                    if index and index % 25 == 0:
                        await asyncio.sleep(0)
                for opportunity in opportunities:
                    authorized = self._playbook_live_authorized(opportunity)
                    opportunity["live_authority_granted"] = authorized
                    gate_matrix = dict(opportunity.get("gate_matrix") or {})
                    gate_matrix["STRATEGY_AUTHORITY"] = (
                        "PASS" if authorized else "FAIL"
                    )
                    opportunity["gate_matrix"] = gate_matrix
                    if not opportunity.get("hard_blockers"):
                        opportunity["next_required_condition"] = (
                            "ENTRY_PERSISTENCE"
                            if authorized
                            else "STRATEGY_AUTHORITY"
                        )
                    changed = self.opportunity_lifecycle.upsert(opportunity)
                    persisted = self.opportunity_lifecycle.state.get(
                        str(opportunity["opportunity_id"]),
                        {},
                    )
                    opportunity.update(
                        {
                            "state": persisted.get(
                                "state", opportunity["state"]
                            ),
                            "persistence_pending": persisted.get(
                                "persistence_pending", False
                            ),
                            "entry_persistence_seconds": persisted.get(
                                "entry_persistence_seconds", 0.0
                            ),
                        }
                    )
                    if not opportunity.get("persistence_pending") and (
                        opportunity.get("state")
                        == OpportunityState.ENTRY_READY.value
                    ):
                        opportunity["next_required_condition"] = (
                            "READY"
                            if authorized
                            else "STRATEGY_AUTHORITY"
                        )
                        persisted["next_required_condition"] = opportunity[
                            "next_required_condition"
                        ]
                    if not changed:
                        continue
                    signal_id = stable_hash(
                        [
                            opportunity["opportunity_id"],
                            opportunity["state"],
                            opportunity["tier"],
                        ],
                        length=40,
                    )
                    self._record_signal_once(
                        {
                            "signal_id": signal_id,
                            "timestamp_utc": opportunity["last_updated_at"],
                            "market": opportunity["market"],
                            "timeframe": "realtime+15m+1h+4h",
                            "strategy_id": opportunity["playbook_id"],
                            "strategy_version": opportunity["playbook_dna"],
                            "signal": (
                                "ENTER"
                                if opportunity["state"]
                                == OpportunityState.ENTRY_READY.value
                                and authorized
                                else "ARMED"
                                if opportunity["state"]
                                in {
                                    OpportunityState.ARMED.value,
                                    OpportunityState.ENTRY_READY.value,
                                }
                                else "WATCH",
                            ),
                            "confidence": opportunity["score"],
                            "regime": opportunity["macro_regime"],
                            "entry_price_reference": opportunity["entry_price"],
                            "proposed_stop": opportunity["stop_loss"],
                            "proposed_target": opportunity["take_profit_1"],
                            "risk_reward": 1.5,
                            "position_size_eur": (
                                str(
                                    self.settings.execution.maximum_live_order_eur
                                )
                                if authorized
                                else "0"
                            ),
                            "risk_amount_eur": None,
                            "spread_bps": opportunity[
                                "realtime_inputs"
                            ].get("spread_bps"),
                            "estimated_slippage_bps": opportunity[
                                "realtime_inputs"
                            ].get("estimated_buy_slippage_bps"),
                            "data_freshness": "PROSPECTIVE_REALTIME",
                            "reason_codes": [
                                f"PLAYBOOK_TIER_{opportunity['tier']}",
                                f"CONFIRMATIONS_{opportunity['confirmation_count']}",
                            ],
                            "blocking_reasons": [
                                *opportunity["hard_blockers"],
                                *(
                                    []
                                    if authorized
                                    else ["PLAYBOOK_LIVE_AUTHORITY_REQUIRED"]
                                ),
                            ],
                            "execution_status": opportunity["state"],
                        }
                    )
                    self._event(
                        "strategy_lifecycle",
                        {
                            "event": "EVENT_DRIVEN_PLAYBOOK_TRANSITION",
                            "opportunity": opportunity,
                            "live_authorized": authorized,
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                    if (
                        opportunity["state"]
                        == OpportunityState.ENTRY_READY.value
                        and authorized
                        and not opportunity.get("hard_blockers")
                    ):
                        await asyncio.to_thread(
                            self.notifier.notify_playbook_event,
                            opportunity,
                            live_authorized=authorized,
                        )
                preserved_ids = [
                    str(identity)
                    for identity, row in self.opportunity_lifecycle.state.items()
                    if str(row.get("market") or "") not in evaluated_markets
                ]
                invalidated = self.opportunity_lifecycle.invalidate_absent(
                    (
                        *preserved_ids,
                        *(
                            str(opportunity["opportunity_id"])
                            for opportunity in opportunities
                        ),
                    )
                )
                for opportunity in invalidated:
                    authorized = self._playbook_live_authorized(opportunity)
                    self._event(
                        "strategy_lifecycle",
                        {
                            "event": "EVENT_DRIVEN_PLAYBOOK_INVALIDATED",
                            "opportunity": opportunity,
                            "live_authorized": authorized,
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                    # Pre-entry invalidations and expirations remain in the
                    # append-only lifecycle ledger but are intentionally not
                    # pushed to Telegram. Execution/fill/position lifecycle
                    # events remain immediately notifiable.
                self.opportunity_lifecycle.expire()
                self._write_near_entry_registry()
                now_monotonic = time.monotonic()
                if now_monotonic - self._last_event_paper_monotonic >= 5:
                    paper = await asyncio.to_thread(
                        run_event_driven_paper_once,
                        self.settings,
                        opportunities=opportunities,
                        realtime_snapshot=snapshot,
                    )
                    for event in paper.get("events") or []:
                        identity = str(event.get("opportunity_id") or "")
                        lifecycle = str(event.get("state") or "")
                        if identity in self.opportunity_lifecycle.state and lifecycle:
                            self.opportunity_lifecycle.transition(
                                identity,
                                OpportunityState(lifecycle),
                                reason_codes=(
                                    str(event.get("reason") or event.get("event")),
                                    "PAPER_ONLY",
                                ),
                                details={"paper_event": event},
                            )
                        self._event(
                            "orders",
                            {
                                "event": str(event.get("event") or "PAPER_EVENT"),
                                "payload": event,
                                "paper_only": True,
                                "orders_generated": 0,
                                "orders_submitted": 0,
                            },
                        )
                        await asyncio.to_thread(
                            self.notifier.notify_order_event,
                            "PAPER_FILL",
                            {
                                **event,
                                "execution_mode": "PAPER_AUTO",
                                "paper_only": True,
                                "filled_quantity": event.get("quantity"),
                                "average_fill_price": event.get("price"),
                                "strategy_id": next(
                                    (
                                        row.get("playbook_id")
                                        for row in opportunities
                                        if row.get("opportunity_id") == identity
                                    ),
                                    None,
                                ),
                            },
                        )
                    self._last_event_paper_monotonic = now_monotonic
                if now_monotonic - self._last_event_live_monotonic >= 5:
                    control = self._control_state()
                    account_health_path = (
                        self.settings.paths.output_dir
                        / "operations"
                        / "live_account_health.json"
                    )
                    account_health = (
                        dict(read_json(account_health_path))
                        if account_health_path.is_file()
                        else {}
                    )
                    strategy_authority = _strategy_entry_authority_summary(
                        self.settings
                    )
                    event_authority = strategy_authority["event_playbook"]
                    event_live = await execute_event_driven_live_once(
                        self.settings,
                        opportunities=opportunities,
                        realtime_snapshot=snapshot,
                        submit=True,
                        allow_new_entry=(
                            control == "ENABLED"
                            and self.private_account_stream is not None
                            and self.private_account_stream.ready
                            and self.websocket.health("bitvavo").get("state")
                            == "CONNECTED"
                            and account_health.get("entry_allowed") is True
                            and event_authority[
                                "effective_entry_authorized"
                            ]
                        ),
                        allowed_economics_entry_families=(
                            event_authority[
                                "canonical_economics_entry_families"
                            ]
                        ),
                    )
                    dispositions = build_entry_ready_dispositions(
                        opportunities,
                        event_live,
                        observed_at=datetime.now(UTC),
                    )
                    new_execution_incident = bool(
                        dispositions.get("execution_incident")
                        and any(
                            str(row.get("disposition_id") or "")
                            not in self._last_disposition_ids
                            for row in dispositions.get("rows") or []
                        )
                    )
                    self._record_entry_ready_dispositions(dispositions)
                    if new_execution_incident:
                        self._event(
                            "errors",
                            {
                                "event": "ENTRY_READY_WITHOUT_FINAL_EXECUTION_REASON",
                                "entry_ready_count": dispositions.get(
                                    "entry_ready_count", 0
                                ),
                                "orders_submitted": event_live.get(
                                    "orders_submitted_this_cycle", 0
                                ),
                                "execution_status": event_live.get("status"),
                                "reason_code": event_live.get("reason_code"),
                                "execution_affected": True,
                            },
                        )
                    for event in event_live.get("events") or []:
                        identity = str(event.get("opportunity_id") or "")
                        event_name = str(event.get("event") or "")
                        if identity in self.opportunity_lifecycle.state:
                            if event_name == "LIVE_ORDER_SUBMITTED":
                                selected_state = OpportunityState.ORDER_SUBMITTED
                            elif event_name == "LIVE_ORDER_INTENT_CREATED":
                                selected_state = OpportunityState.ORDER_INTENT_CREATED
                            elif event_name == "LIVE_EXIT_SUBMITTED":
                                selected_state = OpportunityState.EXITING
                            elif event_name == "LIVE_ORDER_PARTIALLY_FILLED":
                                selected_state = OpportunityState.PARTIALLY_FILLED
                            elif event_name == "LIVE_POSITION_FILLED":
                                selected_state = OpportunityState.FILLED
                            elif event_name == "LIVE_POSITION_CLOSED":
                                selected_state = OpportunityState.CLOSED
                            else:
                                selected_state = OpportunityState.MANAGING
                            self.opportunity_lifecycle.transition(
                                identity,
                                selected_state,
                                reason_codes=(event_name,),
                                details={"live_event": event},
                            )
                        self._event(
                            "orders",
                            {
                                "event": event_name,
                                "payload": event,
                                "paper_only": False,
                                "orders_generated": event_live.get(
                                    "orders_generated_this_cycle", 0
                                ),
                                "orders_submitted": event_live.get(
                                    "orders_submitted_this_cycle", 0
                                ),
                            },
                        )
                        if event_name == "LIVE_POSITION_CLOSED":
                            closed_payload = {
                                **event,
                                "signal_id": identity,
                                "strategy_id": next(
                                    (
                                        row.get("playbook_id")
                                        for row in opportunities
                                        if row.get("opportunity_id") == identity
                                    ),
                                    None,
                                ),
                            }
                            await asyncio.to_thread(
                                self.notifier.notify_position_closed,
                                closed_payload,
                            )
                    if event_live.get("orders_submitted_this_cycle"):
                        self._sync_canonical_execution_events()
                    self._last_event_live_monotonic = now_monotonic
                if now_monotonic - self._last_realtime_projection_monotonic >= 5:
                    snapshot["event_driven_scope"] = (
                        scoped_snapshot.get("event_driven_scope") or {}
                    )
                    snapshot["event_driven_scope"]["warm_markets"] = sorted(
                        warm_markets
                    )
                    atomic_write_json(
                        self.realtime_microstructure_path,
                        snapshot,
                    )
                    self._last_realtime_projection_monotonic = now_monotonic
            except ReconciliationRequired as exc:
                self._automatic_pause(
                    "EVENT_DRIVEN_LIVE_EXECUTION_INCIDENT",
                    recoverable=True,
                )
                self._event(
                    "errors",
                    {
                        "event": "EVENT_DRIVEN_LIVE_EXECUTION_INCIDENT",
                        "exception_type": type(exc).__name__,
                        "playbook_authority_deactivated": False,
                        "execution_affected": True,
                    },
                )
                await self._notify_system_event(
                    "ORDER_REJECTED",
                    {
                        "provider": "bitvavo",
                        "status": "PLAYBOOK_ENTRIES_PAUSED_FOR_RECONCILIATION",
                        "mode": "LIVE_CANARY",
                    },
                )
            except ExecutionBlocked as exc:
                deactivate_authority = (
                    execution_block_requires_authority_deactivation(exc)
                )
                if deactivate_authority:
                    deactivate_playbook_live(self.settings)
                self._automatic_pause(
                    "EVENT_DRIVEN_LIVE_EXECUTION_INCIDENT",
                    recoverable=not deactivate_authority,
                )
                self._event(
                    "errors",
                    {
                        "event": "EVENT_DRIVEN_LIVE_EXECUTION_INCIDENT",
                        "exception_type": type(exc).__name__,
                        "reason_code": execution_block_reason_code(exc),
                        "playbook_authority_deactivated": deactivate_authority,
                        "execution_affected": True,
                    },
                )
                await self._notify_system_event(
                    "ORDER_REJECTED",
                    {
                        "provider": "bitvavo",
                        "status": (
                            "PLAYBOOK_AUTHORITY_DEACTIVATED"
                            if deactivate_authority
                            else "PLAYBOOK_ENTRY_BLOCKED_AUTHORITY_PRESERVED"
                        ),
                        "mode": "LIVE_CANARY",
                    },
                )
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "EVENT_DRIVEN_PLAYBOOK_EXCEPTION",
                        "exception_type": type(exc).__name__,
                        "execution_affected": False,
                    },
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except TimeoutError:
                pass

    async def _run_research_isolated(self) -> dict[str, Any]:
        """Run non-executing research in a cancellable child process."""

        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.root / "main.py"),
            "autonomous-live",
            "research-worker",
            cwd=str(self.root),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )
        stdout, stderr = await self._communicate_worker(
            "research",
            process,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "RESEARCH_WORKER_EXIT_"
                f"{process.returncode}_"
                f"{stable_hash(stderr, length=12)}"
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RESEARCH_WORKER_INVALID_OUTPUT") from exc
        if not isinstance(result, dict):
            raise RuntimeError("RESEARCH_WORKER_INVALID_RESULT")
        return result

    async def _research_loop(self) -> None:
        """Run existing research workers outside the live event loop."""

        interval = max(
            300.0,
            self.settings.autopilot_execution.min_cycle_interval_hours
            * 3_600.0,
        )
        poll_interval = min(300.0, interval)
        while not self._stop.is_set():
            research_status = self.autopilot.status()
            continuous = dict(
                research_status.get("continuous_research") or {}
            )
            if bool(continuous.get("running")):
                background = dict(
                    research_status.get("background_research") or {}
                )
                if (
                    str(background.get("status") or "").upper()
                    != "DEFERRED"
                    or background.get("reason_code")
                    != "CONTINUOUS_SIMPLE_LAB_ACTIVE"
                ):
                    self.autopilot._record_background_research(
                        status="DEFERRED",
                        started_at=datetime.now(UTC),
                        reason_code="CONTINUOUS_SIMPLE_LAB_ACTIVE",
                    )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=poll_interval,
                    )
                except TimeoutError:
                    pass
                continue
            if self.autopilot._research_due():
                started_at = datetime.now(UTC)
                self.autopilot._record_background_research(
                    status="RUNNING",
                    started_at=started_at,
                )
                try:
                    result = await self._run_research_isolated()
                    result_status = str(result.get("status") or "FAILED").upper()
                    completed_status = (
                        "PASSED" if result_status == "PASSED" else "FAILED"
                    )
                    reason_code = (
                        None
                        if completed_status == "PASSED"
                        else str(
                            result.get("reason_code")
                            or f"RESEARCH_{result_status}"
                        )
                    )
                    self.autopilot._record_background_research(
                        status=completed_status,
                        started_at=started_at,
                        result=result,
                        reason_code=reason_code,
                    )
                    self._event(
                        "research",
                        {
                            "event": "RESEARCH_CYCLE_COMPLETED",
                            "status": result_status,
                            "timeframes": result.get("timeframes"),
                            "orders_generated": 0,
                            "orders_submitted": 0,
                        },
                    )
                except Exception as exc:
                    self.autopilot._record_background_research(
                        status="FAILED",
                        started_at=started_at,
                        reason_code=(
                            f"RESEARCH_WORKER_{type(exc).__name__.upper()}"
                        ),
                    )
                    self._event(
                        "errors",
                        {
                            "event": "RESEARCH_WORKER_EXCEPTION",
                            "exception_type": type(exc).__name__,
                        },
                    )
            try:
                # Poll readiness frequently while `_research_due` preserves
                # the configured campaign interval.  This avoids waiting an
                # extra full four-hour interval after a restart just before
                # a campaign becomes due.
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=poll_interval,
                )
            except TimeoutError:
                pass

    async def _health_loop(self) -> None:
        interval = self.settings.autonomous_live.health_seconds
        while not self._stop.is_set():
            companion_services = await asyncio.to_thread(
                self._ensure_companion_services
            )
            top20_tracking = await asyncio.to_thread(
                self._write_top20_tracking_status
            )
            now = datetime.now(UTC)
            audit_slot = now.replace(
                minute=(now.minute // 15) * 15,
                second=0,
                microsecond=0,
            )
            opportunity_audit: dict[str, Any] | None = None
            if audit_slot != self._last_opportunity_audit_slot:
                self._trigger_opportunity_audit_worker()
                self._last_opportunity_audit_slot = audit_slot
            if audit_slot != self._last_intelligence_training_slot:
                self._trigger_intelligence_training_worker()
                self._last_intelligence_training_slot = audit_slot
            intelligence_training = self._intelligence_training_worker_health()
            audit_path = (
                self.settings.paths.output_dir
                / "operations"
                / "daily_opportunity_audit.json"
            )
            if audit_path.is_file():
                opportunity_audit = dict(read_json(audit_path))
            public_stream = await self._recover_public_stream_if_needed(
                self.websocket.health("bitvavo")
            )
            payload = {
                "schema_version": "autonomous_live_heartbeat_v1",
                "pid": os.getpid(),
                "heartbeat_at": utc_iso(),
                "control_state": self._control_state(),
                "websocket": public_stream,
                "private_account_websocket": (
                    self.private_account_stream.health()
                    if self.private_account_stream is not None
                    else {
                        "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                        "ready_for_new_entries": False,
                        "secrets_serialized": False,
                    }
                ),
                "research": self._research_health(),
                "companion_services": companion_services,
                "intelligence_training": intelligence_training,
                "intelligence_model": self._intelligence_model_health(),
                "top20_tracking": top20_tracking,
                "opportunity_audit": opportunity_audit,
                "orderflow_stream": (
                    dict(
                        read_json(
                            self.settings.paths.output_dir
                            / "operations"
                            / "orderflow_stream_health.json"
                        )
                    )
                    if (
                        self.settings.paths.output_dir
                        / "operations"
                        / "orderflow_stream_health.json"
                    ).is_file()
                    else {
                        "status": "STARTING",
                        "markets": list(self.orderflow_markets),
                    }
                ),
            }
            private_ready = bool(
                payload["private_account_websocket"].get(
                    "ready_for_new_entries"
                )
            )
            if (
                self._last_private_stream_ready is not None
                and private_ready != self._last_private_stream_ready
            ):
                await self._notify_system_event(
                    (
                        "PROVIDER_RECOVERED"
                        if private_ready
                        else "PROVIDER_OFFLINE"
                    ),
                    {
                        "provider": "bitvavo_private_account",
                        "status": (
                            "AUTHENTICATED"
                            if private_ready
                            else "NEW_ENTRIES_BLOCKED"
                        ),
                        "mode": "LIVE_CANARY",
                    },
                )
            self._last_private_stream_ready = private_ready
            account_health_path = (
                self.settings.paths.output_dir
                / "operations"
                / "live_account_health.json"
            )
            account_health = (
                dict(read_json(account_health_path))
                if account_health_path.is_file()
                else {}
            )
            await self._notify_scheduled_macro(account_health)
            entry_blockers = tuple(
                sorted(str(value) for value in account_health.get("entry_blockers") or [])
            )
            if entry_blockers != self._last_entry_blockers:
                if entry_blockers:
                    await self._notify_system_event(
                        "OPERATIONAL_DEGRADATION",
                        {
                            "status": "NEW_ENTRIES_BLOCKED",
                            "reason_code": ",".join(entry_blockers),
                            "mode": "LIVE_CANARY",
                        },
                    )
                elif self._last_entry_blockers:
                    await self._notify_system_event(
                        "OPERATIONAL_RECOVERY",
                        {
                            "status": "NEW_ENTRIES_ALLOWED",
                            "reason_code": "ENTRY_BLOCKERS_CLEARED",
                            "mode": "LIVE_CANARY",
                        },
                    )
            self._last_entry_blockers = entry_blockers
            atomic_write_json(self.heartbeat_path, payload)
            if self._control_state() == "SHUTDOWN_REQUESTED":
                self._stop.set()
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    async def _resilient_task(
        self,
        name: str,
        factory: Any,
    ) -> None:
        """Restart a failed supervisor task with bounded exponential backoff."""

        failures = 0
        while not self._stop.is_set():
            stable_since = time.monotonic()
            try:
                await factory()
                if self._stop.is_set():
                    return
                raise RuntimeError("supervisor task returned unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Count only a burst of consecutive failures.  A transport
                # that ran healthily for at least five minutes has recovered;
                # carrying an old reconnect across hours/days previously
                # caused a false non-recoverable pause.
                stable_runtime_seconds = time.monotonic() - stable_since
                if stable_runtime_seconds >= 300.0:
                    failures = 0
                failures += 1
                self._event(
                    "errors",
                    {
                        "event": "SUPERVISOR_TASK_RESTART",
                        "task": name,
                        "failure_count": failures,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc)[:200],
                        "stable_runtime_seconds": round(
                            stable_runtime_seconds, 3
                        ),
                    },
                )
                if failures >= 5:
                    self._automatic_pause(
                        f"SUPERVISOR_TASK_FAILURE_BUDGET_{name.upper()}",
                        recoverable=False,
                    )
                delay = min(30.0, 0.5 * (2 ** min(failures - 1, 6)))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    pass

    async def _control_watch_loop(self) -> None:
        """Observe operator shutdown independently of all heavy services."""

        while not self._stop.is_set():
            if self._control_state() == "SHUTDOWN_REQUESTED":
                self._event(
                    "health",
                    {
                        "event": "SHUTDOWN_REQUEST_OBSERVED",
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    },
                )
                self._stop.set()
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.25)
            except TimeoutError:
                pass

    async def _status_projection_loop(self) -> None:
        """Keep the operator status current without heavy accounting work."""

        while not self._stop.is_set():
            account_path = (
                self.settings.paths.output_dir
                / "operations"
                / "live_account_health.json"
            )
            account = (
                dict(read_json(account_path)) if account_path.is_file() else {}
            )
            heartbeat = (
                dict(read_json(self.heartbeat_path))
                if self.heartbeat_path.is_file()
                else {}
            )
            if heartbeat.get("pid") != os.getpid():
                heartbeat = {}
            strategy_authority = _strategy_entry_authority_summary(
                self.settings
            )
            account_entry_allowed = account.get("entry_allowed") is True
            control_state = self._control_state()
            economics_entry_allowed = strategy_authority[
                "at_least_one_entry_path_authorized"
            ]
            entry_blockers_by_scope = _projected_entry_blockers(
                account,
                economics_entry_allowed=economics_entry_allowed,
                control_state=control_state,
            )
            atomic_write_json(
                self.status_path,
                {
                    "schema_version": "autonomous_live_status_v1",
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "updated_at": utc_iso(),
                    "control_state": control_state,
                    "websocket": self.websocket.health("bitvavo"),
                    "private_account_websocket": (
                        self.private_account_stream.health()
                        if self.private_account_stream is not None
                        else {
                            "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                            "ready_for_new_entries": False,
                        }
                    ),
                    "account_health_status": account.get("status"),
                    "entry_allowed": account_entry_allowed,
                    "entry_allowed_scope": (
                        "ACCOUNT_STRATEGY_AND_CANONICAL_ECONOMICS"
                    ),
                    "account_entry_allowed": account_entry_allowed,
                    "effective_entry_allowed": bool(
                        account_entry_allowed
                        and economics_entry_allowed
                        and control_state == "ENABLED"
                    ),
                    "strategy_entry_authority": strategy_authority,
                    "entry_blockers": entry_blockers_by_scope["all"],
                    "entry_blockers_by_scope": entry_blockers_by_scope,
                    "latest_reconciliation": account.get("checked_at"),
                    "companion_services": heartbeat.get("companion_services"),
                    "orderflow_stream": heartbeat.get("orderflow_stream"),
                    "top20_tracking": heartbeat.get("top20_tracking"),
                    "opportunity_audit": heartbeat.get("opportunity_audit"),
                },
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=2.0)
            except TimeoutError:
                pass

    def _status_projection_thread_loop(self) -> None:
        """Project liveness independently from the asyncio execution loop."""

        while not self._status_thread_stop.is_set():
            try:
                account_path = (
                    self.settings.paths.output_dir
                    / "operations"
                    / "live_account_health.json"
                )
                account = (
                    dict(read_json(account_path))
                    if account_path.is_file()
                    else {}
                )
                heartbeat = (
                    dict(read_json(self.heartbeat_path))
                    if self.heartbeat_path.is_file()
                    else {}
                )
                if heartbeat.get("pid") != os.getpid():
                    heartbeat = {}
                strategy_authority = _strategy_entry_authority_summary(
                    self.settings
                )
                account_entry_allowed = account.get("entry_allowed") is True
                control_state = self._control_state()
                economics_entry_allowed = strategy_authority[
                    "at_least_one_entry_path_authorized"
                ]
                entry_blockers_by_scope = _projected_entry_blockers(
                    account,
                    economics_entry_allowed=economics_entry_allowed,
                    control_state=control_state,
                )
                atomic_write_json(
                    self.status_path,
                    {
                        "schema_version": "autonomous_live_status_v1",
                        "status": "RUNNING",
                        "pid": os.getpid(),
                        "updated_at": utc_iso(),
                        "control_state": control_state,
                        "websocket": self.websocket.health("bitvavo"),
                        "private_account_websocket": (
                            self.private_account_stream.health()
                            if self.private_account_stream is not None
                            else {
                                "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                                "ready_for_new_entries": False,
                            }
                        ),
                        "account_health_status": account.get("status"),
                        "entry_allowed": account_entry_allowed,
                        "entry_allowed_scope": (
                            "ACCOUNT_STRATEGY_AND_CANONICAL_ECONOMICS"
                        ),
                        "account_entry_allowed": account_entry_allowed,
                        "effective_entry_allowed": bool(
                            account_entry_allowed
                            and economics_entry_allowed
                            and control_state == "ENABLED"
                        ),
                        "strategy_entry_authority": strategy_authority,
                        "entry_blockers": entry_blockers_by_scope["all"],
                        "entry_blockers_by_scope": entry_blockers_by_scope,
                        "latest_reconciliation": account.get("checked_at"),
                        "companion_services": heartbeat.get(
                            "companion_services"
                        ),
                        "orderflow_stream": heartbeat.get("orderflow_stream"),
                        "top20_tracking": heartbeat.get("top20_tracking"),
                        "opportunity_audit": heartbeat.get(
                            "opportunity_audit"
                        ),
                    },
                )
            except Exception as exc:
                self._event(
                    "errors",
                    {
                        "event": "STATUS_PROJECTION_THREAD_FAILURE",
                        "exception_type": type(exc).__name__,
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    },
                )
            self._status_thread_stop.wait(2.0)

    def _shutdown_watchdog(
        self,
        *,
        timeout_seconds: float = 40.0,
    ) -> None:
        """Guarantee process termination after the persisted shutdown budget."""

        time.sleep(timeout_seconds)
        os._exit(0)

    async def run(self) -> None:
        authority = (
            dict(read_json(self.authority_path))
            if self.authority_path.is_file()
            else {}
        )
        if authority.get("active") is not True:
            raise PermissionError("run autonomous-live enable first")
        if self._control_state() not in {"ENABLED", "PAUSED"}:
            raise PermissionError("autonomous-live control state is not runnable")
        self._acquire()
        tasks: list[asyncio.Task[Any]] = []
        shutdown_path = self.output / "shutdown_diagnostics.json"
        loop = asyncio.get_running_loop()
        for selected in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(selected, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            companion_services = self._ensure_companion_services(force=True)
            recovery = self._recover_position_tracker_from_ledger()
            reconciliation = await self.reconcile()
            recovered_orphan_intents = (
                self.opportunity_lifecycle.recover_orphan_order_intents(
                    reconciliation_ready=(
                        reconciliation.get("status") == "READY"
                    ),
                )
            )
            if recovered_orphan_intents:
                self._event(
                    "strategy_lifecycle",
                    {
                        "event": "ORPHAN_ORDER_INTENTS_RECONCILED",
                        "recovered_count": recovered_orphan_intents,
                        "fresh_revalidation_required": True,
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    },
                )
            if (
                recovery.get("status") != "READY"
                or reconciliation.get("status") != "READY"
            ):
                self._automatic_pause(
                    (
                        "STARTUP_POSITION_RECOVERY_FAILED"
                        if recovery.get("status") != "READY"
                        else "STARTUP_RECONCILIATION_FAILED"
                    ),
                    recoverable=(recovery.get("status") == "READY"),
                )
            await self._refresh_bitvavo_ticker_universe()
            await self.websocket.start(self._public_subscriptions())
            if self.private_account_stream is not None:
                await self.private_account_stream.start()
                await self.private_account_stream.wait_until_ready(timeout=10.0)
            self._event(
                "health",
                {
                    "event": "AUTONOMOUS_LIVE_STARTED",
                    "pid": os.getpid(),
                    "markets": list(self.markets),
                    "orderflow_tracking_markets": list(self.orderflow_markets),
                    "ticker_tracking_markets": list(
                        self.ticker_tracking_markets
                    ),
                    "position_recovery": recovery,
                    "startup_reconciliation": reconciliation.get("status"),
                    "recovered_orphan_order_intents": (
                        recovered_orphan_intents
                    ),
                    "companion_services": companion_services,
                    "private_account_websocket": (
                        self.private_account_stream.health()
                        if self.private_account_stream is not None
                        else {
                            "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                            "ready_for_new_entries": False,
                        }
                    ),
                },
            )
            atomic_write_json(
                self.status_path,
                {
                    "schema_version": "autonomous_live_status_v1",
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "started_at": utc_iso(),
                    "control_state": self._control_state(),
                    "websocket": self.websocket.health("bitvavo"),
                    "private_account_websocket": (
                        self.private_account_stream.health()
                        if self.private_account_stream is not None
                        else {
                            "state": "DISABLED_MISSING_TRADE_CREDENTIALS",
                            "ready_for_new_entries": False,
                        }
                    ),
                    "startup_reconciliation": reconciliation.get("status"),
                    "entry_authority_deferred_to_account_health": True,
                },
            )
            await self._notify_system_event(
                "SERVICE_START",
                {
                    "provider": "bitvavo",
                    "status": (
                        "LIVE_RUNNING"
                        if (
                            self.private_account_stream is not None
                            and self.private_account_stream.ready
                        )
                        else "LIVE_DEGRADED"
                    ),
                    "mode": "LIVE_CANARY",
                },
            )
            active_scan_task = asyncio.create_task(
                self._resilient_task(
                    "active_trading_scan",
                    self._active_trading_scan_loop,
                ),
                name="active_trading_scan",
            )
            research_task = asyncio.create_task(
                self._resilient_task("research", self._research_loop),
                name="research",
            )
            self._status_thread_stop.clear()
            status_thread = threading.Thread(
                target=self._status_projection_thread_loop,
                daemon=True,
                name="autonomous-live-status-projection",
            )
            status_thread.start()
            tasks = [
                asyncio.create_task(
                    self._control_watch_loop(),
                    name="control_watch",
                ),
                asyncio.create_task(
                    self._resilient_task("websocket_consumer", self._websocket_consumer),
                    name="websocket_consumer",
                ),
                asyncio.create_task(
                    self._resilient_task("reconciliation", self._reconciliation_loop),
                    name="reconciliation",
                ),
                asyncio.create_task(
                    self._resilient_task("execution", self._execution_loop),
                    name="execution",
                ),
                active_scan_task,
                research_task,
                asyncio.create_task(
                    self._resilient_task("health", self._health_loop),
                    name="health",
                ),
                asyncio.create_task(
                    self._resilient_task("top20_orderflow", self._orderflow_loop),
                    name="top20_orderflow",
                ),
                asyncio.create_task(
                    self._resilient_task(
                        "event_driven_playbooks",
                        self._event_driven_playbook_loop,
                    ),
                    name="event_driven_playbooks",
                ),
            ]
            if self.private_account_stream is not None:
                tasks.append(
                    asyncio.create_task(
                        self._resilient_task(
                            "private_account_consumer",
                            self._private_account_consumer,
                        ),
                        name="private_account_consumer",
                    )
                )
            await self._stop.wait()
            threading.Thread(
                target=self._shutdown_watchdog,
                daemon=True,
                name="autonomous-live-shutdown-watchdog",
            ).start()
            requested_at = utc_iso()
            atomic_write_json(
                shutdown_path,
                {
                    "schema_version": "autonomous_live_shutdown_v1",
                    "status": "CANCELLING_TASKS",
                    "requested_at": requested_at,
                    "pid": os.getpid(),
                    "tasks": [
                        {
                            "name": task.get_name(),
                            "done": task.done(),
                            "cancelled": task.cancelled(),
                        }
                        for task in tasks
                    ],
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
            for task in tasks:
                task.cancel()
            _, pending_tasks = await asyncio.wait(tasks, timeout=15.0)
            timed_out = bool(pending_tasks)
            blocking = [
                {
                    "name": task.get_name(),
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                }
                for task in tasks
                if not task.done()
            ]
            atomic_write_json(
                shutdown_path,
                {
                    "schema_version": "autonomous_live_shutdown_v1",
                    "status": (
                        "TASK_TIMEOUT" if timed_out or blocking else "TASKS_STOPPED"
                    ),
                    "requested_at": requested_at,
                    "tasks_stopped_at": utc_iso(),
                    "pid": os.getpid(),
                    "bounded_task_timeout_seconds": 15,
                    "blocking_tasks": blocking,
                    "pending_async_tasks": len(blocking),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )
        finally:
            self._status_thread_stop.set()
            if "status_thread" in locals():
                status_thread.join(timeout=3.0)
            if self.orderflow_recorder is not None:
                self.orderflow_recorder.stop()
            audit_worker_close_status = self._terminate_opportunity_audit_worker()
            intelligence_worker_close_status = (
                self._terminate_intelligence_training_worker()
            )
            # Execution/reconciliation tasks are already cancelled above.
            # Release the single-instance lock even if a websocket close or
            # notification cleanup subsequently fails.
            self._release()
            close_status: dict[str, str] = {}
            if self.private_account_stream is not None:
                try:
                    await asyncio.wait_for(
                        self.private_account_stream.stop(), timeout=5.0
                    )
                    close_status["private_websocket"] = "CLOSED"
                except TimeoutError:
                    close_status["private_websocket"] = "TIMEOUT"
                except Exception as exc:
                    close_status["private_websocket"] = (
                        f"FAILED_{type(exc).__name__.upper()}"
                    )
            try:
                await asyncio.wait_for(self.websocket.stop(), timeout=5.0)
                close_status["public_websocket"] = "CLOSED"
            except TimeoutError:
                close_status["public_websocket"] = "TIMEOUT"
            except Exception as exc:
                close_status["public_websocket"] = (
                    f"FAILED_{type(exc).__name__.upper()}"
                )
            try:
                await asyncio.wait_for(
                    self._notify_system_event(
                        "SERVICE_STOP",
                        {
                            "provider": "bitvavo",
                            "status": "STOPPED",
                            "mode": "LIVE_CANARY",
                        },
                    ),
                    timeout=5.0,
                )
                close_status["stop_notification"] = "SENT"
            except TimeoutError:
                close_status["stop_notification"] = "TIMEOUT"
            except Exception as exc:
                close_status["stop_notification"] = (
                    f"FAILED_{type(exc).__name__.upper()}"
                )
            self._event("health", {"event": "AUTONOMOUS_LIVE_STOPPED"})
            atomic_write_json(
                self.status_path,
                {
                    "schema_version": "autonomous_live_status_v1",
                    "status": "STOPPED",
                    "stopped_at": utc_iso(),
                    "control_state": self._control_state(),
                },
            )
            prior_shutdown = (
                dict(read_json(shutdown_path))
                if shutdown_path.is_file()
                else {}
            )
            atomic_write_json(
                shutdown_path,
                {
                    **prior_shutdown,
                    "status": "STOPPED",
                    "stopped_at": utc_iso(),
                    "websocket_close_status": close_status,
                    "opportunity_audit_worker_close_status": (
                        audit_worker_close_status
                    ),
                    "intelligence_training_worker_close_status": (
                        intelligence_worker_close_status
                    ),
                    "lock_released": True,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                },
            )


__all__ = [
    "APPROVAL_PHRASE",
    "EVENT_STREAMS",
    "LAUNCH_MARKETS",
    "AutonomousLiveLockError",
    "AutonomousLiveSupervisor",
]
