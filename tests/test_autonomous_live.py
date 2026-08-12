from __future__ import annotations

import asyncio
import ctypes
import json
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import PathSettings, Settings
from core.autonomous_live import (
    APPROVAL_PHRASE,
    EVENT_STREAMS,
    LAUNCH_MARKETS,
    AutonomousLiveSupervisor,
    _execution_evidence_summary,
    _projected_entry_blockers,
    _strategy_entry_authority_summary,
    _strategy_evidence_watch_summary,
    amsterdam_macro_slot,
    resolved_orderflow_markets,
    resolved_ticker_tracking_markets,
    select_intensive_tracking_markets,
)
from core.contracts import NormalizedStreamEvent, StreamEventType
from core.event_driven_live import MAXIMUM_ORDER_EUR
from reporting.canonical_economics import ECONOMIC_SCHEMA_VERSION
from utils.common import atomic_write_json, stable_hash


def test_projected_entry_blockers_preserve_independent_scopes() -> None:
    blockers = _projected_entry_blockers(
        {
            "status": "BLOCKED",
            "entry_allowed": False,
            "entry_blockers": [],
            "failures": [
                "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"
            ],
        },
        economics_entry_allowed=False,
        control_state="PAUSED",
    )

    assert blockers == {
        "account": [
            "ACCOUNT_HEALTH_BLOCKED",
            "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION",
        ],
        "strategy_and_economics": [
            "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
        ],
        "control": ["CONTROL_STATE_PAUSED"],
        "all": [
            "ACCOUNT_HEALTH_BLOCKED",
            "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION",
            "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING",
            "CONTROL_STATE_PAUSED",
        ],
    }


def test_projected_entry_blockers_are_empty_when_every_scope_is_ready() -> None:
    blockers = _projected_entry_blockers(
        {"status": "READY", "entry_allowed": True},
        economics_entry_allowed=True,
        control_state="ENABLED",
    )

    assert blockers == {
        "account": [],
        "strategy_and_economics": [],
        "control": [],
        "all": [],
    }


def test_execution_evidence_status_projection_is_compact(tmp_path: Path) -> None:
    path = tmp_path / "execution_evidence_layers.json"
    atomic_write_json(
        path,
        {
            "schema_version": "execution_evidence_layers_v1",
            "generated_at": "2026-08-09T00:00:00Z",
            "evidence_hash": "hash",
            "theoretical_signal_pnl": {
                "status": "READY",
                "resolved_episode_count": 5,
                "false_breakout_rate": 0.2,
                "mfe_distribution_pct": {"large": "omitted"},
            },
            "simulated_execution_pnl": {
                "status": "READY",
                "closed_round_trips": 10,
                "net_pnl_eur": "1",
                "net_expectancy_eur": "0.1",
                "fees_eur": "0.2",
                "by_playbook": {"large": "omitted"},
            },
            "actual_live_pnl": {
                "status": "READY",
                "integrity_status": "PASSED",
                "closed_round_trips": 1,
                "open_positions": 1,
                "realised_pnl_eur": "0",
                "unrealised_pnl_eur": "0.1",
                "net_pnl_eur": "0.1",
                "fees_eur": "0.02",
                "active_strategy_count": 2,
                "active_strategies": [{"large": "omitted"}],
            },
            "comparison_policy": {"layers_are_not_interchangeable": True},
        },
    )

    summary = _execution_evidence_summary(path)

    assert "by_playbook" not in summary["simulated_execution_pnl"]
    assert "active_strategies" not in summary["actual_live_pnl"]
    assert summary["actual_live_pnl"]["integrity_status"] == "PASSED"


def test_strategy_evidence_watch_status_projection_is_compact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strategy_evidence_watch.json"
    atomic_write_json(
        path,
        {
            "schema_version": "strategy_evidence_watch_v1",
            "generated_at": "2026-08-09T00:00:00Z",
            "live_accounting_integrity": "PASSED",
            "watched_strategy_evidence_row_count": 2,
            "recommendation_counts": {"COLLECT_PAPER_AND_SHADOW": 2},
            "policy": {"automatic_authority_changes": False},
            "strategies": [{"large": "omitted"}],
        },
    )

    summary = _strategy_evidence_watch_summary(path)

    assert summary["watched_strategy_evidence_row_count"] == 2
    assert summary["automatic_authority_changes"] is False
    assert "strategies" not in summary


def test_strategy_entry_authority_summary_keeps_paths_separate(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    config = tmp_path / "config"
    governance = settings.paths.output_dir / "governance"
    config.mkdir(parents=True)
    governance.mkdir(parents=True)
    atomic_write_json(
        config / "live_playbook_authority.json",
        {
            "active": True,
            "approved_playbooks": [
                {"playbook_id": "MOMENTUM_BREAKOUT_V1", "active": True},
                {"playbook_id": "OFF", "active": False},
            ],
        },
    )
    atomic_write_json(
        governance / "positive_strategy_live_authority.json",
        {
            "active": False,
            "approved_candidates": [{"candidate_id": "one"}],
        },
    )

    summary = _strategy_entry_authority_summary(settings)

    assert summary["event_playbook"] == {
        "active": True,
        "approved_active_playbook_count": 1,
        "authority_present": True,
        "canonical_economics_entry_families": [],
        "effective_entry_authorized": False,
    }
    assert summary["generated_positive_portfolio"] == {
        "active": False,
        "approved_candidate_count": 1,
        "authority_present": True,
        "canonical_economics_entry_strategy_dna_hashes": [],
        "effective_entry_authorized": False,
    }
    assert summary["at_least_one_entry_path_authorized"] is False
    assert summary["canonical_economics"]["status"] == "NOT_AVAILABLE"
    assert summary["protective_exit_authority_is_independent"] is True

    artifact = {
        "schema_version": ECONOMIC_SCHEMA_VERSION,
        "promotion_recommendations": [
            {
                "strategy_family": "MOMENTUM",
                "promotion_status": "LIVE_VALIDATED",
                "recommendation": "ALLOW_BOUNDED_LIVE_ENTRY",
                "live_validated": True,
            }
        ],
    }
    artifact["artifact_hash"] = stable_hash(artifact, length=64)
    artifact_path = (
        settings.paths.output_dir
        / "economics"
        / "runs"
        / "test-live"
        / "canonical_strategy_family_economics.json"
    )
    atomic_write_json(artifact_path, artifact)
    atomic_write_json(
        settings.paths.output_dir / "economics" / "latest.json",
        {
            "artifact_path": str(artifact_path.resolve()),
            "artifact_hash": artifact["artifact_hash"],
        },
    )

    validated = _strategy_entry_authority_summary(settings)

    assert validated["event_playbook"]["effective_entry_authorized"] is True
    assert validated["event_playbook"][
        "canonical_economics_entry_families"
    ] == ["MOMENTUM"]
    assert validated["at_least_one_entry_path_authorized"] is True


class _SeedRecorder:
    def __init__(self) -> None:
        self.markets: list[str] = []
        self.events: list[str] = []

    def seed_orderbook(self, snapshot: object) -> None:
        self.events.append("seed")
        self.markets.append(str(getattr(snapshot, "canonical_market")))

    async def pause(self) -> None:
        self.events.append("pause")

    def resume(self) -> None:
        self.events.append("resume")


def test_macro_slots_follow_amsterdam_clock() -> None:
    assert (
        amsterdam_macro_slot(datetime(2026, 8, 5, 6, 0, tzinfo=UTC))
        == "2026-08-05 08:00"
    )
    assert (
        amsterdam_macro_slot(datetime(2026, 8, 5, 21, 0, tzinfo=UTC))
        == "2026-08-05 23:00"
    )
    assert amsterdam_macro_slot(datetime(2026, 8, 5, 12, 0, tzinfo=UTC)) is None


def _temporary_live_settings(
    settings: Settings,
    tmp_path: Path,
) -> Settings:
    paths = PathSettings(project_root=tmp_path)
    return settings.model_copy(update={"paths": paths})


def test_autonomous_live_settings_are_approved_spot_launch(
    isolated_settings: Settings,
) -> None:
    selected = isolated_settings.autonomous_live
    assert tuple(selected.markets) == LAUNCH_MARKETS
    assert tuple(selected.monitor_only_markets) == ("PYR-EUR",)
    assert selected.maximum_risk_per_trade_pct == 0.25
    assert selected.maximum_total_open_risk_pct == 1.0
    assert selected.maximum_total_crypto_exposure_pct == 80.0
    assert selected.minimum_cash_reserve_pct == 20.0
    assert selected.reconciliation_seconds == 30
    assert selected.active_trading_scan_minutes == 5
    assert selected.active_trading_poll_seconds == 30
    assert selected.active_trading_maximum_rows == 1_500
    assert MAXIMUM_ORDER_EUR == Decimal("25")


def test_event_driven_scope_recalculates_changed_and_warm_markets(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    snapshot = {
        "observed_at": datetime.now(UTC).isoformat(),
        "markets": [
            {
                "market": "BTC-EUR",
                "price": 50_000.0,
                "latest_event_at": "2026-08-07T18:00:00+00:00",
                "latest_trade_at": "2026-08-07T18:00:00+00:00",
                "book_update_count_10s": 3,
                "book": {"best_bid": 49_999.0, "best_ask": 50_001.0},
            },
            {
                "market": "ADA-EUR",
                "price": 0.50,
                "latest_event_at": "2026-08-07T18:00:00+00:00",
                "latest_trade_at": "2026-08-07T18:00:00+00:00",
                "book_update_count_10s": 3,
                "book": {"best_bid": 0.499, "best_ask": 0.501},
            },
        ],
    }
    _first, first_markets, _warm = supervisor._event_driven_snapshot_scope(
        snapshot, []
    )
    assert first_markets == {"BTC-EUR", "ADA-EUR"}

    scoped, selected, warm = supervisor._event_driven_snapshot_scope(
        snapshot,
        [{"market": "ADA-EUR", "status": "NEAR_ENTRY"}],
    )

    assert selected == {"ADA-EUR"}
    assert warm == {"ADA-EUR"}
    assert {row["market"] for row in scoped["markets"]} == {
        "BTC-EUR",
        "ADA-EUR",
    }
    assert scoped["event_driven_scope"]["evaluated_market_count"] == 1


@pytest.mark.asyncio
async def test_event_loop_prices_liquidity_for_actual_level_2_canary(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )

    class CaptureRecorder:
        requested_notional_eur: float | None = None

        def realtime_snapshot(
            self,
            *,
            markets: tuple[str, ...],
            order_notional_eur: float,
        ) -> dict[str, object]:
            assert markets == ("BTC-EUR",)
            self.requested_notional_eur = order_notional_eur
            supervisor._stop.set()
            return {"markets": []}

    recorder = CaptureRecorder()
    supervisor.orderflow_recorder = recorder  # type: ignore[assignment]
    supervisor.orderflow_markets = ("BTC-EUR",)
    supervisor._last_event_paper_monotonic = float("inf")
    supervisor._last_event_live_monotonic = float("inf")
    supervisor._last_realtime_projection_monotonic = float("inf")

    await supervisor._event_driven_playbook_loop()

    assert recorder.requested_notional_eur == 25.0


def test_orderflow_startup_stages_books_before_tickers_and_trades(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor.orderflow_markets = ("BTC-EUR", "ADA-EUR")
    supervisor.ticker_tracking_markets = ("BTC-EUR", "ADA-EUR", "NPC-EUR")

    assert supervisor._initial_orderflow_subscriptions() == {
        "bitvavo": {"book": ("BTC-EUR", "ADA-EUR")}
    }
    assert supervisor._orderflow_subscriptions() == {
        "bitvavo": {
            "ticker24h": ("BTC-EUR", "ADA-EUR", "NPC-EUR"),
            "trades": ("BTC-EUR", "ADA-EUR"),
            "book": ("BTC-EUR", "ADA-EUR"),
        }
    }


@pytest.mark.asyncio
async def test_autonomous_live_rejects_wrong_approval_before_private_calls(
    isolated_settings: Settings,
) -> None:
    supervisor = AutonomousLiveSupervisor(isolated_settings)
    with pytest.raises(PermissionError, match="approval phrase"):
        await supervisor.enable(
            markets=LAUNCH_MARKETS,
            approval="wrong",
        )


@pytest.mark.asyncio
async def test_autonomous_live_rejects_partial_or_expanded_market_set(
    isolated_settings: Settings,
) -> None:
    supervisor = AutonomousLiveSupervisor(isolated_settings)
    with pytest.raises(PermissionError, match="do not match"):
        await supervisor.enable(
            markets=("BTC-EUR", "ETH-EUR"),
            approval=APPROVAL_PHRASE,
        )
    with pytest.raises(PermissionError, match="do not match"):
        await supervisor.enable(
            markets=(*LAUNCH_MARKETS, "DOGE-EUR"),
            approval=APPROVAL_PHRASE,
        )


def test_autonomous_live_creates_all_append_only_event_streams(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    for stream in EVENT_STREAMS:
        assert (supervisor.events / f"{stream}.jsonl").is_file()
    assert supervisor.status()["authority_active"] is False
    assert supervisor.status()["process_running"] is False


def test_automatic_pause_cannot_overwrite_operator_shutdown(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    shutdown = supervisor.shutdown()

    result = supervisor._automatic_pause(
        "CONCURRENT_TASK_FAILURE",
        recoverable=False,
    )

    assert shutdown["state"] == "SHUTDOWN_REQUESTED"
    assert result["state"] == "SHUTDOWN_REQUESTED"
    assert supervisor._control_state() == "SHUTDOWN_REQUESTED"


@pytest.mark.asyncio
async def test_control_watcher_observes_external_shutdown_quickly(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    watcher = asyncio.create_task(supervisor._control_watch_loop())
    supervisor.shutdown()
    await asyncio.wait_for(watcher, timeout=1.0)
    assert supervisor._stop.is_set()


@pytest.mark.asyncio
async def test_bounded_shutdown_returns_when_runtime_is_already_gone(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor.lock_path.write_text('{"pid": 12345}', encoding="utf-8")
    monkeypatch.setattr(supervisor, "_pid_alive", lambda _pid: False)

    result = await supervisor.shutdown_bounded(timeout_seconds=0.01)

    assert result["status"] == "STOPPED"
    assert result["forced_termination"] is False


@pytest.mark.asyncio
async def test_bounded_shutdown_preserves_runtime_when_risk_state_uncertain(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor.lock_path.write_text('{"pid": 12345}', encoding="utf-8")
    monkeypatch.setattr(supervisor, "_pid_alive", lambda _pid: True)

    async def uncertain_health(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "status": "READY",
            "reconciliation": {
                "healthy": False,
                "remote_open_orders": 0,
                "local_open_orders": 0,
            },
        }

    monkeypatch.setattr(
        "core.autonomous_live.live_account_health",
        uncertain_health,
    )

    result = await supervisor.shutdown_bounded(timeout_seconds=0.01)

    assert result["status"] == "SHUTDOWN_BLOCKED_ACTIVE_OR_UNCERTAIN_RISK_STATE"
    assert result["forced_termination"] is False
    assert result["protective_runtime_preserved"] is True


@pytest.mark.asyncio
async def test_cancelled_isolated_worker_is_terminated_and_recorded(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    task = asyncio.create_task(
        supervisor._communicate_worker("test_worker", process)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10.0)
    assert process.returncode is not None
    health = (
        supervisor.events / "health.jsonl"
    ).read_text(encoding="utf-8")
    assert "SHUTDOWN_CHILD_WORKER_TERMINATED" in health
    assert "test_worker" in health


def test_coinmarketcap_top20_gets_explicit_live_tracking_projection(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    universe = settings.paths.output_dir / "universe"
    universe.mkdir(parents=True, exist_ok=True)
    symbols = [
        (1, "BTC", "BTC-EUR"),
        (2, "ETH", "ETH-EUR"),
        (3, "USDT", None),
        (4, "BNB", "BNB-EUR"),
        (5, "USDC", "USDC-EUR"),
        (6, "XRP", "XRP-EUR"),
        (7, "SOL", "SOL-EUR"),
        (8, "TRX", "TRX-EUR"),
        (9, "HYPE", "HYPE-EUR"),
        (10, "DOGE", "DOGE-EUR"),
        (11, "LEO", None),
        (12, "ZEC", None),
        (13, "ADA", "ADA-EUR"),
        (14, "XMR", None),
        (15, "LINK", "LINK-EUR"),
        (16, "XLM", "XLM-EUR"),
        (17, "DAI", None),
        (18, "CC", "CC-EUR"),
        (19, "BCH", "BCH-EUR"),
        (20, "USD1", None),
    ]
    current_rows = [
        {
            "rank": rank,
            "symbol": symbol,
            "name": symbol,
            "eur_spot_market": market,
            "available_at": "2026-08-03T00:00:00Z",
        }
        for rank, symbol, market in symbols
    ]
    eligibility_rows = [
        {
            **row,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": (
                "LIVE_ELIGIBLE"
                if row["eur_spot_market"]
                not in {"USDC-EUR", "TRX-EUR", "CC-EUR"}
                else "CONTEXT_ONLY"
            ),
            "execution_reason": "TEST",
        }
        for row in current_rows
    ]
    (universe / "top50_current.json").write_text(
        json.dumps(
            {
                "source_snapshot_hash": "test-hash",
                "source_collected_at": "2026-08-03T00:00:00Z",
                "rows": current_rows,
            }
        ),
        encoding="utf-8",
    )
    (universe / "top50_eligibility.json").write_text(
        json.dumps({"rows": eligibility_rows}),
        encoding="utf-8",
    )
    (universe / "tiered_trading_universe.json").write_text(
        json.dumps(
            {
                "shadow_markets": [
                    "BTC-EUR",
                    "ETH-EUR",
                    "SOL-EUR",
                    "LINK-EUR",
                    "ADA-EUR",
                ],
                "discovery_markets": [
                    row["eur_spot_market"]
                    for row in current_rows
                    if row["eur_spot_market"]
                ],
            }
        ),
        encoding="utf-8",
    )

    intensive = resolved_orderflow_markets(
        settings,
        fallback=settings.autonomous_live.markets,
    )
    ticker = resolved_ticker_tracking_markets(
        settings,
        fallback=intensive,
    )
    assert {"BTC-EUR", "ETH-EUR", "ADA-EUR", "BCH-EUR"} <= set(
        intensive
    )
    assert {"USDC-EUR", "TRX-EUR", "CC-EUR"} <= set(ticker)

    supervisor = AutonomousLiveSupervisor(settings)
    projection = supervisor._write_top20_tracking_status()
    by_symbol = {row["symbol"]: row for row in projection["rows"]}
    assert projection["top20_count"] == 20
    assert by_symbol["ADA"]["tracking_mode"] == (
        "REALTIME_TICKER_TRADES_ORDERBOOK"
    )
    assert by_symbol["USDC"]["tracking_mode"] == (
        "REALTIME_TICKER_CONTEXT_ONLY"
    )
    assert by_symbol["USDT"]["tracking_mode"] == (
        "CMC_CONTEXT_NO_BITVAVO_EUR_MARKET"
    )


def test_fast_movers_replace_only_non_core_intensive_markets() -> None:
    selected = select_intensive_tracking_markets(
        core_markets=("BTC-EUR", "ETH-EUR", "ADA-EUR"),
        position_markets=("TAO-EUR",),
        mover_ranking=(
            {
                "market": "NPC-EUR",
                "qualified_for_intensive_tracking": True,
                "score": 90.0,
            },
            {
                "market": "HYPE-EUR",
                "qualified_for_intensive_tracking": True,
                "score": 80.0,
            },
        ),
        current_markets=("BTC-EUR", "ETH-EUR", "ADA-EUR", "OLD-EUR"),
        maximum_markets=6,
        mover_slots=2,
    )

    assert selected == (
        "BTC-EUR",
        "ETH-EUR",
        "ADA-EUR",
        "TAO-EUR",
        "NPC-EUR",
        "HYPE-EUR",
    )


def test_active_opportunities_remain_on_intensive_stream(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor.opportunity_lifecycle._state = {
        "watching": {"market": "NPC-EUR", "state": "WATCHING"},
        "armed": {"market": "ADA-EUR", "state": "ARMED"},
        "ready": {"market": "TAO-EUR", "state": "ENTRY_READY"},
        "closed": {"market": "OLD-EUR", "state": "CLOSED"},
        "discovered": {"market": "NEW-EUR", "state": "DISCOVERED"},
    }

    assert supervisor._armed_opportunity_markets() == (
        "NPC-EUR",
        "ADA-EUR",
        "TAO-EUR",
    )


def test_public_stream_covers_all_native_tactical_candle_intervals(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)

    assert supervisor._public_subscriptions()["bitvavo"]["candles"][
        "interval"
    ] == ("1m", "5m", "15m", "1h", "2h", "4h", "1d")
    assert "book" not in supervisor._public_subscriptions()["bitvavo"]
    assert "trades" not in supervisor._public_subscriptions()["bitvavo"]


def test_live_supervisor_bootstraps_execution_core_before_dynamic_movers(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)

    assert supervisor.orderflow_markets == supervisor.core_orderflow_markets
    assert "PYR-EUR" in supervisor.orderflow_markets
    assert "PYR-EUR" not in supervisor.markets
    assert "ADA-EUR" in supervisor.orderflow_markets
    assert "NPC-EUR" in supervisor.orderflow_markets
    assert supervisor.orderflow_websocket.queue.maxsize >= 50_000


@pytest.mark.asyncio
async def test_startup_refresh_tracks_all_current_bitvavo_eur_tickers(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)

    class Loader:
        def __init__(self, selected: Settings) -> None:
            assert selected is settings

        async def download_market_metadata(self, **_: object) -> list[object]:
            return [
                SimpleNamespace(
                    canonical_market="BTC-EUR",
                    values={"market": "BTC-EUR", "status": "trading"},
                ),
                SimpleNamespace(
                    canonical_market="PYR-EUR",
                    values={"market": "PYR-EUR", "status": "trading"},
                ),
                SimpleNamespace(
                    canonical_market="ETH-USDC",
                    values={"market": "ETH-USDC", "status": "trading"},
                ),
                SimpleNamespace(
                    canonical_market="OLD-EUR",
                    values={"market": "OLD-EUR", "status": "halted"},
                ),
            ]

    monkeypatch.setattr("core.autonomous_live.DataLoader", Loader)
    result = await supervisor._refresh_bitvavo_ticker_universe()

    assert "PYR-EUR" in supervisor.ticker_tracking_markets
    assert "ETH-USDC" not in supervisor.ticker_tracking_markets
    assert "OLD-EUR" not in supervisor.ticker_tracking_markets
    assert result["venue_eur_market_count"] == 2
    assert (
        settings.paths.output_dir
        / "universe"
        / "bitvavo_eur_ticker_universe.json"
    ).is_file()


def test_status_uses_persisted_running_ticker_and_mover_coverage(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    universe = settings.paths.output_dir / "universe"
    universe.mkdir(parents=True, exist_ok=True)
    supervisor.lock_path.write_text(
        json.dumps({"pid": os.getpid()}),
        encoding="utf-8",
    )
    (universe / "bitvavo_eur_ticker_universe.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "rows": [
                    {"market": "BTC-EUR", "realtime_ticker": True},
                    {"market": "NEW-EUR", "realtime_ticker": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (universe / "dynamic_intensive_movers.json").write_text(
        json.dumps({"selected_markets": ["BTC-EUR", "NEW-EUR"]}),
        encoding="utf-8",
    )

    status = supervisor.status()

    assert status["ticker_tracking_markets"] == ["BTC-EUR", "NEW-EUR"]
    assert status["ticker_tracking_market_count"] == 2
    assert status["orderflow_tracking_markets"] == ["BTC-EUR", "NEW-EUR"]
    assert status["orderflow_tracking_market_count"] == 2


def test_autonomous_live_detects_current_process_on_windows(
    isolated_settings: Settings,
) -> None:
    supervisor = AutonomousLiveSupervisor(isolated_settings)
    assert supervisor._pid_alive(os.getpid()) is True


def test_autonomous_live_windows_pid_check_rejects_exited_process_handle(
    isolated_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Kernel32:
        def __init__(self) -> None:
            self.closed: list[int] = []

        @staticmethod
        def OpenProcess(*_args: object) -> int:
            return 123

        @staticmethod
        def GetExitCodeProcess(_handle: int, pointer: object) -> int:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulong)).contents.value = 0
            return 1

        def CloseHandle(self, handle: int) -> None:
            self.closed.append(handle)

    kernel32 = _Kernel32()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(
        "core.autonomous_live.ctypes.windll",
        SimpleNamespace(kernel32=kernel32),
    )

    supervisor = AutonomousLiveSupervisor(isolated_settings)
    assert supervisor._pid_alive(43612) is False
    assert kernel32.closed == [123]


def test_companion_commands_use_only_canonical_main_entrypoint(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )

    commands = supervisor._companion_commands()

    assert set(commands) == {"data_sync", "simple_lab"}
    assert commands["data_sync"][1] == str(tmp_path / "main.py")
    assert commands["data_sync"][2:4] == ["data", "sync"]
    assert "--continuous" in commands["data_sync"]
    timeframe_index = commands["data_sync"].index("--timeframes")
    context_index = commands["data_sync"].index("--context")
    assert commands["data_sync"][timeframe_index + 1] == "15m,1h,4h,1d"
    assert commands["data_sync"][context_index + 1] == "none"
    extra_index = commands["data_sync"].index("--extra-markets")
    assert commands["data_sync"][extra_index + 1] == "PYR-EUR"
    assert commands["simple_lab"][1] == str(tmp_path / "main.py")
    assert commands["simple_lab"][2:4] == ["simple-lab", "run"]
    assert "--continuous" in commands["simple_lab"]
    timeframe_index = commands["simple_lab"].index("--timeframes")
    assert commands["simple_lab"][timeframe_index + 1] == (
        "15m,1h,2h,4h,1d,1W"
    )
    market_index = commands["simple_lab"].index("--markets")
    assert commands["simple_lab"][market_index + 1] == ",".join(
        (*LAUNCH_MARKETS, "PYR-EUR")
    )
    batch_index = commands["simple_lab"].index("--backtest-batch-size")
    assert commands["simple_lab"][batch_index + 1] == "8"
    generation_index = commands["simple_lab"].index(
        "--generation-batch-size"
    )
    assert commands["simple_lab"][generation_index + 1] == "100"
    workers_index = commands["simple_lab"].index("--workers")
    assert commands["simple_lab"][workers_index + 1] == "1"
    history_index = commands["simple_lab"].index(
        "--minimum-exact-history-days"
    )
    assert commands["simple_lab"][history_index + 1] == "365"
    market_cycle_index = commands["simple_lab"].index(
        "--max-markets-per-exact-cycle"
    )
    assert commands["simple_lab"][market_cycle_index + 1] == "1"
    trial_index = commands["simple_lab"].index("--max-trials")
    assert commands["simple_lab"][trial_index + 1] == "4"
    assert all("order" not in argument.casefold() for command in commands.values() for argument in command)


def test_companion_sync_includes_fail_closed_execution_exceptions(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    config_directory = tmp_path / "config"
    config_directory.mkdir(parents=True)
    (config_directory / "execution_market_exceptions.yaml").write_text(
        """version: 1
default_policy: FAIL_CLOSED
markets:
  NPC-EUR:
    approved: true
    spot_only: true
    maximum_order_eur: 5.0
    maximum_total_exposure_eur: 10.0
    requires_approved_strategy_dna: true
    requires_natural_signal: true
""",
        encoding="utf-8",
    )
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )

    command = supervisor._companion_commands()["data_sync"]

    extra_index = command.index("--extra-markets")
    assert command[extra_index + 1] == "NPC-EUR,PYR-EUR"


def test_execution_exception_is_monitored_without_granting_dna_authority(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    config_directory = tmp_path / "config"
    config_directory.mkdir(parents=True)
    (config_directory / "execution_market_exceptions.yaml").write_text(
        """version: 1
default_policy: FAIL_CLOSED
markets:
  NPC-EUR:
    approved: true
    spot_only: true
    maximum_order_eur: 5.0
    maximum_total_exposure_eur: 10.0
    requires_approved_strategy_dna: true
    requires_natural_signal: true
""",
        encoding="utf-8",
    )
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )

    assert "NPC-EUR" in supervisor.markets
    assert supervisor.status()["authority_active"] is False


def test_automatic_pause_does_not_impersonate_operator_and_recovers_safely(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )

    paused = supervisor._automatic_pause(
        "PERIODIC_RECONCILIATION_FAILED",
        recoverable=True,
    )

    assert paused["reason"].startswith("AUTO_RECOVERABLE_")
    assert supervisor._observe_reconciliation_health(ready=True) is False
    assert supervisor._observe_reconciliation_health(ready=True) is False
    assert supervisor._observe_reconciliation_health(ready=True) is True
    assert supervisor._control_state() == "ENABLED"


def test_operator_and_nonrecoverable_pauses_never_auto_resume(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor.pause()
    for _ in range(4):
        assert supervisor._observe_reconciliation_health(ready=True) is False
    assert supervisor._control_state() == "PAUSED"

    supervisor._automatic_pause("TASK_FAILURE", recoverable=False)
    for _ in range(4):
        assert supervisor._observe_reconciliation_health(ready=True) is False
    assert supervisor._control_state() == "PAUSED"


def test_companion_supervision_respects_live_single_instance_locks(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    data_lock = settings.paths.checkpoints_dir / "data_service.lock"
    simple_lock = supervisor._simple_lab_lock_path()
    data_lock.parent.mkdir(parents=True, exist_ok=True)
    simple_lock.parent.mkdir(parents=True, exist_ok=True)
    data_lock.write_text(
        json.dumps({"pid": os.getpid(), "service_id": "data"}),
        encoding="utf-8",
    )
    simple_lock.write_text(
        json.dumps({"pid": os.getpid(), "mode": "CONTINUOUS"}),
        encoding="utf-8",
    )

    def unexpected_spawn(name: str, command: list[str]) -> int:
        raise AssertionError(f"unexpected spawn for {name}: {command}")

    monkeypatch.setattr(supervisor, "_spawn_companion", unexpected_spawn)

    result = supervisor._ensure_companion_services(force=True)

    assert result["services"]["data_sync"]["status"] == "RUNNING"
    assert result["services"]["simple_lab"]["status"] == "RUNNING"
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0


def test_companion_supervision_starts_only_missing_services(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    spawned: list[tuple[str, list[str]]] = []

    def record_spawn(name: str, command: list[str]) -> int:
        spawned.append((name, command))
        return 10_000 + len(spawned)

    monkeypatch.setattr(supervisor, "_spawn_companion", record_spawn)

    result = supervisor._ensure_companion_services(force=True)

    assert [name for name, _ in spawned] == ["data_sync", "simple_lab"]
    assert result["services"]["data_sync"]["status"] == "STARTING"
    assert result["services"]["simple_lab"]["status"] == "STARTING"
    assert result["execution_affected_by_companion_failure"] is False
    assert supervisor.companion_status_path.is_file()


@pytest.mark.asyncio
async def test_websocket_consumer_accepts_pydantic_stream_event(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    now = datetime.now(UTC)
    await supervisor.websocket.queue.put(
        NormalizedStreamEvent(
            event_type=StreamEventType.TICKER,
            provider="bitvavo",
            source_symbol="ETH-EUR",
            canonical_market="ETH-EUR",
            timestamp=now,
            observed_at=now,
            message_id="ticker-1",
            payload={"last_price": "1700"},
        )
    )
    supervisor.websocket._stop.set()

    await supervisor._websocket_consumer()

    assert supervisor.signals()["count"] == 0


@pytest.mark.asyncio
async def test_orderflow_reseed_waits_for_stream_and_seeds_all_markets_concurrently(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor.orderflow_markets = ("BTC-EUR", "ETH-EUR", "TAO-EUR")
    health_calls = 0

    def health(_: str) -> dict[str, object]:
        nonlocal health_calls
        health_calls += 1
        return {
            "state": "CONNECTED",
            "last_message_at": (
                None if health_calls == 1 else "2026-08-01T20:00:00Z"
            ),
        }

    supervisor.orderflow_websocket.health = health  # type: ignore[method-assign]
    active_downloads = 0
    maximum_active_downloads = 0

    class FakeLoader:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def download_orderbook_snapshot(self, **kwargs: object) -> object:
            nonlocal active_downloads, maximum_active_downloads
            active_downloads += 1
            maximum_active_downloads = max(maximum_active_downloads, active_downloads)
            await asyncio.sleep(0)
            active_downloads -= 1
            return type(
                "Snapshot",
                (),
                {"canonical_market": str(kwargs["market"])},
            )()

    monkeypatch.setattr("core.autonomous_live.DataLoader", FakeLoader)
    recorder = _SeedRecorder()

    await supervisor._wait_for_orderflow_stream(timeout_seconds=1.0)
    await supervisor._seed_orderflow_books(  # type: ignore[arg-type]
        recorder,
        pause_before_apply=True,
    )

    assert health_calls >= 2
    assert maximum_active_downloads == 3
    assert recorder.markets == ["BTC-EUR", "ETH-EUR", "TAO-EUR"]
    assert recorder.events == ["pause", "seed", "seed", "seed", "resume"]


@pytest.mark.asyncio
async def test_research_loop_records_current_background_lifecycle(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    monkeypatch.setattr(supervisor.autopilot, "_research_due", lambda: True)

    async def complete_research() -> dict[str, object]:
        supervisor._stop.set()
        return {
            "status": "PASSED",
            "timeframes": ["15m", "1h", "4h"],
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    monkeypatch.setattr(supervisor, "_run_research_isolated", complete_research)

    await supervisor._research_loop()

    status = json.loads(
        supervisor.autopilot.research_status_path.read_text(encoding="utf-8")
    )
    assert status["status"] == "PASSED"
    assert status["result"]["timeframes"] == ["15m", "1h", "4h"]
    assert status["orders_submitted"] == 0


@pytest.mark.asyncio
async def test_research_loop_defers_batch_when_continuous_lab_is_running(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    monkeypatch.setattr(
        supervisor.autopilot,
        "status",
        lambda: {
            "continuous_research": {
                "status": "RUNNING",
                "running": True,
                "pid": 123,
            },
            "background_research": {
                "status": "FAILED",
                "reason_code": "OLD_FAILURE",
            },
        },
    )
    research_due_called = False

    def research_due() -> bool:
        nonlocal research_due_called
        research_due_called = True
        return True

    monkeypatch.setattr(supervisor.autopilot, "_research_due", research_due)
    supervisor._stop.set()
    supervisor._stop.clear()

    async def stop_after_record(
        awaitable: object,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        supervisor._stop.set()

    monkeypatch.setattr(asyncio, "wait_for", stop_after_record)

    await supervisor._research_loop()

    status = json.loads(
        supervisor.autopilot.research_status_path.read_text(encoding="utf-8")
    )
    assert status["status"] == "DEFERRED"
    assert status["reason_code"] == "CONTINUOUS_SIMPLE_LAB_ACTIVE"
    assert status["orders_submitted"] == 0
    assert research_due_called is False


def test_canonical_execution_events_are_mirrored_once_across_restart(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(
        update={"paths": temporary_paths}
    )
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor.execution_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-07-29T00:00:00Z",
            "payload": {"market": "ETH-EUR", "side": "BUY"},
        },
        {
            "event_type": "ORDER_ACKNOWLEDGED",
            "recorded_at": "2026-07-29T00:00:01Z",
            "payload": {"status": "filled"},
        },
        {
            "event_type": "FILL",
            "recorded_at": "2026-07-29T00:00:02Z",
            "payload": {"market": "ETH-EUR", "fee_eur": "0.01"},
        },
    ]
    supervisor.execution_ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )
    first = supervisor._sync_canonical_execution_events()
    second = AutonomousLiveSupervisor(
        settings
    )._sync_canonical_execution_events()
    assert first["mirrored"] == 3
    assert second["mirrored"] == 0
    assert len(
        (supervisor.events / "orders.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 2
    assert len(
        (supervisor.events / "fills.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1


def test_signal_identity_is_durable_across_restart(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={"output_dir": tmp_path / "output"}
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    first = AutonomousLiveSupervisor(settings)
    signal = {
        "signal_id": "stable-signal-id",
        "market": "ETH-EUR",
        "timeframe": "1h",
        "signal": "NO_ENTRY",
    }
    assert first._record_signal_once(signal) is True
    second = AutonomousLiveSupervisor(settings)
    assert second._record_signal_once(signal) is False
    assert len(
        (first.events / "signals.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1


def test_position_tracker_recovers_live_fill_idempotently(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor.execution_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.execution_ledger_path.write_text(
        json.dumps(
            {
                "event_type": "FILL",
                "recorded_at": "2026-07-29T00:00:00Z",
                "payload": {
                    "fill_id": "fill-1",
                    "order_id": "order-1",
                    "client_order_id": "client-1",
                    "market": "ETH-EUR",
                    "side": "BUY",
                    "quantity": "0.002",
                    "price": "2500",
                    "fee_eur": "0.01",
                    "filled_at": "2026-07-29T00:00:00Z",
                    "venue": "bitvavo",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first = supervisor._recover_position_tracker_from_ledger()
    second = AutonomousLiveSupervisor(
        settings
    )._recover_position_tracker_from_ledger()
    assert first == {"status": "READY", "fills_replayed": 1}
    assert second == {"status": "READY", "fills_replayed": 0}
    assert (
        supervisor.position_tracker.positions["ETH-EUR"].owned_quantity
        == Decimal("0.002")
    )


def test_position_recovery_skips_operator_inventory_reallocation_fill(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor.execution_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor.execution_ledger_path.write_text(
        json.dumps(
            {
                "event_type": "FILL",
                "recorded_at": "2026-07-30T18:47:43Z",
                "payload": {
                    "fill_id": "inventory-fill-1",
                    "order_id": "inventory-order-1",
                    "market": "NPC-EUR",
                    "side": "SELL",
                    "quantity": "100",
                    "price": "0.005",
                    "fee_eur": "0.01",
                    "filled_at": 1785437262965,
                    "venue": "bitvavo",
                    "strategy_id": (
                        "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert supervisor._recover_position_tracker_from_ledger() == {
        "status": "READY",
        "fills_replayed": 0,
    }
    assert supervisor.position_tracker.positions == {}


@pytest.mark.asyncio
async def test_private_account_stream_failure_blocks_only_new_entries(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor._write_control_state("ENABLED", reason="TEST")
    captured: list[bool] = []

    class UnavailablePrivateStream:
        ready = False

    supervisor.private_account_stream = UnavailablePrivateStream()

    async def run_once(
        *,
        run_research: bool,
        allow_live_new_entries: bool,
    ) -> dict[str, object]:
        assert run_research is False
        captured.append(allow_live_new_entries)
        supervisor._stop.set()
        return {
            "finished_at": datetime.now(UTC).isoformat(),
            "stages": {
                "live_canary": {
                    "cycle_status": "NO_TRADE",
                    "reason_code": "NO_SIGNAL",
                    "orders_submitted": 0,
                    "natural_signal": {},
                    "entry_liquidity": {},
                    "canary_limits": {},
                }
            },
        }

    monkeypatch.setattr(supervisor.autopilot, "run_once", run_once)
    await supervisor._execution_loop()

    assert captured == [False]
    signals = supervisor.signals()["signals"]
    assert "PRIVATE_ACCOUNT_STREAM_NOT_READY" in signals[-1][
        "blocking_reasons"
    ]


@pytest.mark.asyncio
async def test_public_market_stream_failure_blocks_new_entries(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor._write_control_state("ENABLED", reason="TEST")
    captured: list[bool] = []

    class ReadyPrivateStream:
        ready = True

    supervisor.private_account_stream = ReadyPrivateStream()
    monkeypatch.setattr(
        supervisor.websocket,
        "health",
        lambda provider=None: {"state": "FAILED"},
    )

    async def run_once(
        *,
        run_research: bool,
        allow_live_new_entries: bool,
    ) -> dict[str, object]:
        assert run_research is False
        captured.append(allow_live_new_entries)
        supervisor._stop.set()
        return {
            "finished_at": datetime.now(UTC).isoformat(),
            "stages": {
                "live_canary": {
                    "cycle_status": "NO_TRADE",
                    "reason_code": "NO_SIGNAL",
                    "orders_submitted": 0,
                    "natural_signal": {},
                    "entry_liquidity": {},
                    "canary_limits": {},
                }
            },
        }

    monkeypatch.setattr(supervisor.autopilot, "run_once", run_once)
    await supervisor._execution_loop()

    assert captured == [False]
    signals = supervisor.signals()["signals"]
    assert "PUBLIC_MARKET_STREAM_NOT_READY" in signals[-1][
        "blocking_reasons"
    ]


@pytest.mark.asyncio
async def test_public_stream_watchdog_restarts_terminal_failure(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    calls: list[object] = []

    class FailedWebSocket:
        async def stop(self) -> None:
            calls.append("stop")

        async def start(self, subscriptions) -> None:
            calls.append(subscriptions)

        def health(self, provider=None):
            return {"state": "CONNECTING", "provider": provider}

    supervisor.websocket = FailedWebSocket()
    recovered = await supervisor._recover_public_stream_if_needed(
        {"state": "FAILED"}
    )

    assert calls[0] == "stop"
    assert calls[1]["bitvavo"]["ticker"] == supervisor.markets
    assert recovered["state"] == "CONNECTING"
    assert supervisor._public_stream_restart_attempts == 1


def test_health_degrades_when_public_stream_is_not_connected(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    supervisor._write_control_state("ENABLED", reason="TEST")
    supervisor.lock_path.write_text(
        json.dumps({"pid": os.getpid()}),
        encoding="utf-8",
    )
    supervisor.status_path.write_text(
        json.dumps(
            {
                "websocket": {"state": "FAILED"},
                "private_account_websocket": {
                    "state": "AUTHENTICATED",
                    "ready_for_new_entries": True,
                },
            }
        ),
        encoding="utf-8",
    )

    health = supervisor.health()

    assert health["status"] == "DEGRADED"
    assert health["public_stream_ready"] is False
    assert health["private_stream_ready"] is True


def test_research_health_prefers_running_continuous_service(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = AutonomousLiveSupervisor(
        _temporary_live_settings(isolated_settings, tmp_path)
    )
    monkeypatch.setattr(
        supervisor.autopilot,
        "status",
        lambda: {
            "research_subprocess_active": True,
            "continuous_research": {
                "status": "RUNNING",
                "running": True,
                "pid": 123,
            },
            "background_research": {
                "status": "INTERRUPTED_RESTART_RECOVERABLE",
                "reason_code": "SUPERVISOR_PROCESS_NOT_RUNNING",
            },
        },
    )

    research = supervisor._research_health()

    assert research["status"] == "RUNNING"
    assert research["mode"] == "CONTINUOUS_SIMPLE_LAB"
    assert research["research_subprocess_active"] is True


@pytest.mark.asyncio
async def test_live_performance_notifications_are_durable_and_orderless(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    temporary_paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": tmp_path / "output",
            "checkpoints_dir": tmp_path / "output" / "checkpoints",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": temporary_paths})
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor._write_control_state("ENABLED", reason="TEST")
    account_health = temporary_paths.output_dir / "operations"
    account_health.mkdir(parents=True, exist_ok=True)
    (account_health / "live_account_health.json").write_text(
        json.dumps(
            {
                "account": {
                    "eur_available": "30",
                    "portfolio_valuation": {
                        "estimated_total_equity_eur": "1000"
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    portfolio = temporary_paths.output_dir / "portfolio"
    portfolio.mkdir(parents=True, exist_ok=True)
    (portfolio / "daily_profit_target.json").write_text(
        json.dumps({"mark_to_market_pnl_eur": "5"}),
        encoding="utf-8",
    )
    calls: list[tuple[str, dict]] = []

    class FakeNotifier:
        def notify_strategy_performance(self, payload):
            calls.append(("strategy", dict(payload)))
            return {"delivery_status": "PENDING"}

        def notify_daily_performance(self, payload):
            calls.append(("daily", dict(payload)))
            return {"delivery_status": "PENDING"}

    supervisor.notifier = FakeNotifier()
    accounts = {
        "strategies": [
            {
                "strategy_id": "RR_B60_H5_Z20",
                "strategy_dna": "frozen-dna",
                "authority_level": "LIVE_CANARY",
                "allocated_capital_eur": "10",
                "used_capital_eur": "0",
                "closed_trade_count": 1,
                "open_trade_count": 0,
                "realised_pnl_eur": "1",
                "unrealised_pnl_eur": "0",
                "net_pnl_eur": "1",
                "fees_paid_eur": "0.02",
                "maximum_drawdown_eur": "0",
                "last_closed_trade": {
                    "market": "ETH-EUR",
                    "closed_at": "2026-07-30T12:00:00Z",
                    "net_pnl_eur": "1",
                    "fees_eur": "0.02",
                    "holding_seconds": 3600,
                },
            }
        ]
    }

    await supervisor._notify_performance_snapshots(accounts)
    await supervisor._notify_performance_snapshots(accounts)

    assert [name for name, _ in calls] == ["strategy", "daily"]
    assert calls[0][1]["strategy_equity_eur"] == "11"
    assert calls[1][1]["live_orders_today"] == 0
    state = json.loads(
        supervisor.performance_notification_state_path.read_text(
            encoding="utf-8"
        )
    )
    assert state["closed_trade_counts"] == {"frozen-dna": 1}
    assert state["last_daily_date"] == datetime.now(UTC).date().isoformat()
    assert state["orders_generated"] == 0
    assert state["orders_submitted"] == 0


@pytest.mark.asyncio
async def test_five_minute_active_scan_observes_but_never_owns_execution(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    supervisor._active_scan_not_before_monotonic = 0.0
    supervisor.websocket.health_state["bitvavo"].state = "CONNECTED"
    supervisor.websocket.health_state["bitvavo"].last_message_at = datetime.now(UTC)
    supervisor.orderflow_websocket.health_state["bitvavo"].state = "CONNECTED"
    supervisor.orderflow_websocket.health_state[
        "bitvavo"
    ].last_message_at = datetime.now(UTC)
    supervisor.orderflow_recorder = object()  # type: ignore[assignment]
    calls: list[dict[str, object]] = []

    async def fake_scan_all() -> dict[str, object]:
        calls.append({"isolated": True, "execute": False})
        supervisor._stop.set()
        return {
            "status": "LIVE_ACTIVE_NO_CURRENT_ENTRY",
            "reason": "NO_VALID_ENTRY_AFTER_FULL_SCAN",
            "market_count": 5,
            "evaluations": {"1h": 60, "2h": 50},
            "macro": {"regime": "MACRO_RISK_OFF"},
        }

    monkeypatch.setattr(supervisor, "_run_active_trading_scan_isolated", fake_scan_all)

    await supervisor._active_trading_scan_loop()

    assert calls == [
        {
            "isolated": True,
            "execute": False,
        }
    ]
    events = (supervisor.events / "research.jsonl").read_text(
        encoding="utf-8"
    )
    assert "ACTIVE_TRADING_FULL_SCAN" in events
    assert "NO_VALID_ENTRY_AFTER_FULL_SCAN" in events
    assert '"scan_interval_minutes":5' in events
    assert '"scan_poll_seconds":30.0' in events
    assert '"scan_maximum_rows":1500' in events


@pytest.mark.asyncio
async def test_opportunity_audit_runs_in_isolated_non_execution_worker(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    calls: list[tuple[object, ...]] = []

    class Process:
        pid = 1234
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                json.dumps(
                    {
                        "pnl_status": "VERIFIED_PAPER_CLOSED_FILLS",
                        "orders_generated": 0,
                        "orders_submitted": 0,
                    }
                ).encode(),
                b"",
            )

    async def create(*args: object, **kwargs: object) -> Process:
        calls.append((*args, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await supervisor._run_opportunity_audit_isolated()

    assert result["pnl_status"] == "VERIFIED_PAPER_CLOSED_FILLS"
    assert calls[0][2:4] == ("live", "opportunity-audit")
    status = json.loads(
        (supervisor.output / "opportunity_audit_worker_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "COMPLETED"
    assert status["orders_submitted"] == 0


def test_intelligence_training_worker_completion_is_reconciled(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    status_path = supervisor.output / "intelligence_training_worker_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "RUNNING_SHADOW_ONLY",
                "authority": "SHADOW_ONLY",
                "pid": 1234,
            }
        ),
        encoding="utf-8",
    )

    class CompletedProcess:
        @staticmethod
        def poll() -> int:
            return 0

    supervisor._intelligence_training_process = CompletedProcess()  # type: ignore[assignment]

    result = supervisor._intelligence_training_worker_health()

    assert result["status"] == "COMPLETED"
    assert result["return_code"] == 0
    assert result["live_decision_influence"] is False
    assert result["orders_submitted"] == 0
    assert supervisor._intelligence_training_process is None
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "COMPLETED"


def test_intelligence_model_health_exposes_shadow_drift_without_authority(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _temporary_live_settings(isolated_settings, tmp_path)
    supervisor = AutonomousLiveSupervisor(settings)
    status_path = (
        settings.paths.output_dir / "intelligence" / "model_status.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "SHADOW_ONLY",
                "authority": "SHADOW_ONLY",
                "live_decision_influence": False,
                "row_count": 600,
                "trained_until_timestamp": "2026-08-09T00:00:00Z",
                "chronological_validation": True,
                "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
                "drift_monitor": {
                    "status": "WARNING_DRIFT_SHADOW",
                    "critical_feature_count": 0,
                    "warning_feature_count": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    result = supervisor._intelligence_model_health()

    assert result["status"] == "SHADOW_ONLY"
    assert result["row_count"] == 600
    assert result["drift_monitor"]["status"] == "WARNING_DRIFT_SHADOW"
    assert result["authority"] == "SHADOW_ONLY"
    assert result["live_decision_influence"] is False
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0
