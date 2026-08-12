from __future__ import annotations

import asyncio
import ctypes
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from aiohttp import WSMsgType

from config.settings import PathSettings, Settings
from core.cli import _validated_continuous_data_service_launch, build_parser
from core.contracts import (
    DataValidationError,
    NormalizedDataRecord,
    NormalizedStreamEvent,
    ProviderStatus,
    StreamEventType,
)
from data.data_loader import ContinuousDataService, DataLoader
from data.database import Database
from data.market_data import (
    drop_open_candles,
    timeframe_delta,
    validate_ohlcv,
)
from data.orderbook_l2 import Level2OrderBook, SequenceGap
from data.websocket_manager import (
    StreamHealth,
    WebSocketManager,
    decode_mexc_protobuf,
)
from utils.common import read_json


def isolated_cache(settings: Settings, tmp_path) -> Settings:
    return settings.model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def test_continuous_service_windows_pid_check_rejects_exited_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Kernel32:
        def __init__(self) -> None:
            self.closed: list[int] = []

        @staticmethod
        def OpenProcess(*_args: object) -> int:
            return 321

        @staticmethod
        def GetExitCodeProcess(_handle: int, pointer: object) -> int:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulong)).contents.value = 0
            return 1

        def CloseHandle(self, handle: int) -> None:
            self.closed.append(handle)

    kernel32 = _Kernel32()
    monkeypatch.setattr("data.data_loader.os.name", "nt")
    monkeypatch.setattr(
        "data.data_loader.ctypes.windll",
        SimpleNamespace(kernel32=kernel32),
    )

    assert ContinuousDataService._process_alive(25712) is False
    assert kernel32.closed == [321]


def test_data_service_control_commands_are_registered() -> None:
    parser = build_parser()

    status = parser.parse_args(["data", "service-status"])
    restart = parser.parse_args(["data", "service-restart", "--timeout", "30"])

    assert status.data_command == "service-status"
    assert restart.data_command == "service-restart"
    assert restart.timeout == 30.0


def test_data_service_restart_only_replays_validated_research_sync(
    isolated_settings: Settings,
) -> None:
    executable = (
        isolated_settings.paths.project_root / ".venv" / "Scripts" / "python.exe"
    )
    main = isolated_settings.paths.project_root / "main.py"
    owner = {
        "service_id": "continuous-data-service",
        "mode": "research",
        "executable": str(executable),
        "command": [str(main), "data", "sync", "--continuous"],
    }

    validated_executable, command = _validated_continuous_data_service_launch(
        isolated_settings, owner
    )

    assert validated_executable == executable
    assert command == [str(main), "data", "sync", "--continuous"]


def test_data_service_restart_rejects_unrelated_command(
    isolated_settings: Settings,
) -> None:
    executable = (
        isolated_settings.paths.project_root / ".venv" / "Scripts" / "python.exe"
    )
    owner = {
        "service_id": "continuous-data-service",
        "mode": "research",
        "executable": str(executable),
        "command": [str(isolated_settings.paths.project_root / "main.py"), "live", "run"],
    }

    with pytest.raises(ValueError, match="DATA_SERVICE_COMMAND_NOT_CONTINUOUS_SYNC"):
        _validated_continuous_data_service_launch(isolated_settings, owner)


def test_relational_candle_projection_is_batched(
    isolated_settings: Settings,
) -> None:
    class RecordingDatabase:
        def __init__(self) -> None:
            self.batches: list[list[dict]] = []

        def upsert_records(
            self,
            table: str,
            records: list[dict],
        ) -> None:
            assert table == "candles"
            self.batches.append(records)

    market_data = isolated_settings.market_data.model_copy(
        update={"maximum_database_batch_size": 2}
    )
    settings = isolated_settings.model_copy(
        update={"market_data": market_data}
    )
    database = RecordingDatabase()
    loader = DataLoader(settings, database=database)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records = [
        NormalizedDataRecord(
            provider="kraken",
            source_symbol="XBTEUR",
            canonical_market="BTC-EUR",
            timestamp=start + timedelta(hours=offset),
            observed_at=start + timedelta(hours=offset + 1),
            available_at=start + timedelta(hours=offset + 1),
            data_kind="ohlcv",
            timeframe="1h",
            closed=True,
            retrieval_run_id="batch-test",
            raw_hash=f"hash-{offset}",
            raw_payload={},
            values={
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
            },
        )
        for offset in range(5)
    ]

    loader._database_upsert("candles", records)

    assert [len(batch) for batch in database.batches] == [2, 2, 1]


def test_raw_batches_are_content_addressed_across_retrieval_runs(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    loader = DataLoader(settings)
    observed = datetime(2026, 8, 11, tzinfo=UTC)

    def raw_record(run_id: str) -> NormalizedDataRecord:
        return NormalizedDataRecord(
            provider="fred",
            source_symbol="DGS10",
            canonical_market="BTC-EUR",
            timestamp=observed,
            observed_at=observed,
            available_at=observed,
            data_kind="macro_observation",
            retrieval_run_id=run_id,
            raw_hash="same-source-payload",
            raw_payload={"value": "4.25"},
            values={"series_id": "DGS10", "value": "4.25"},
        )

    first = loader._persist_raw_batch([raw_record("run-one")])
    second = loader._persist_raw_batch([raw_record("run-two")])

    assert first == second
    assert "run-one" not in first[0].name
    assert len(list(settings.paths.raw_data_dir.rglob("*.parquet"))) == 1


def test_storage_estimate_accounts_for_all_compressed_projections(
    isolated_settings: Settings,
) -> None:
    loader = DataLoader(isolated_settings)
    estimate = loader.estimate_fetch(
        providers=("bitvavo", "kraken", "mexc"),
        universe_size=2,
        history_profile="smoke",
        timeframes=("1m", "3m", "3h", "2d"),
    )

    assert estimate["estimated_total_storage_bytes"] == sum(
        estimate[key]
        for key in (
            "estimated_compressed_storage_bytes",
            "estimated_raw_storage_bytes",
            "estimated_normalized_storage_bytes",
            "estimated_context_storage_bytes",
        )
    )
    assert (
        estimate["storage_estimate_basis"]
        == "PARQUET_COMPRESSED_CACHE_72_RAW_128_"
        "NORMALIZED_96_CONTEXT_24_BYTES_PER_ROW"
    )
    assert estimate["storage_allowed"]


def test_candle_age_is_not_reported_as_websocket_transport_latency() -> None:
    now = datetime.now(UTC)
    health = StreamHealth(provider="bitvavo")
    candle = NormalizedStreamEvent(
        event_type=StreamEventType.CANDLE,
        provider="bitvavo",
        source_symbol="ETH-EUR",
        canonical_market="ETH-EUR",
        timestamp=now - timedelta(hours=1),
        observed_at=now,
        message_id="candle-latency-test",
        payload={"closed": False},
    )
    ticker = candle.model_copy(
        update={
            "event_type": StreamEventType.TICKER,
            "timestamp": now - timedelta(milliseconds=25),
            "message_id": "ticker-latency-test",
        }
    )

    WebSocketManager._record_transport_latency(health, candle)
    assert health.latencies_ms == []
    WebSocketManager._record_transport_latency(health, ticker)
    assert health.latencies_ms == pytest.approx([25.0])


def test_week_and_month_timeframes_do_not_collapse_to_minutes() -> None:
    assert timeframe_delta("3m") == timedelta(minutes=3)
    assert timeframe_delta("3h") == timedelta(hours=3)
    assert timeframe_delta("2d") == timedelta(days=2)
    assert timeframe_delta("1W") == timedelta(days=7)
    assert timeframe_delta("1w") == timedelta(days=7)
    assert timeframe_delta("1mo") == timedelta(days=30)
    assert timeframe_delta("1m") == timedelta(minutes=1)


def test_calendar_month_is_not_closed_before_next_month() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                ]
            ),
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [10.0, 10.0],
        }
    )
    with pytest.raises(
        DataValidationError,
        match="open candle",
    ):
        validate_ohlcv(
            frame.iloc[:1],
            timeframe="1mo",
            now=datetime(
                2025,
                1,
                31,
                23,
                59,
                tzinfo=UTC,
            ),
        )
    validated = validate_ohlcv(
        frame.iloc[:1],
        timeframe="1mo",
        now=datetime(2025, 2, 1, tzinfo=UTC),
    )
    assert len(validated) == 1
    closed = drop_open_candles(
        frame,
        timeframe="1mo",
        now=datetime(2025, 2, 15, tzinfo=UTC),
    )
    assert list(closed.index) == [
        pd.Timestamp("2025-01-01T00:00:00Z")
    ]


@pytest.mark.asyncio
async def test_capability_matrix_has_native_intervals_and_exact_statuses(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    loader = DataLoader(isolated_cache(isolated_settings, tmp_path))
    rows = await loader.capability_matrix(probe=False, persist=True)
    indexed = {row["provider"]: row for row in rows}
    assert {"bitvavo", "kraken", "mexc", "fred", "deribit"} <= set(indexed)
    assert "8h" in indexed["bitvavo"]["native_candle_intervals"]
    assert "1W" in indexed["kraken"]["native_candle_intervals"]
    assert "1mo" in indexed["mexc"]["native_candle_intervals"]
    assert all(row["status"] in {status.value for status in ProviderStatus} for row in rows)
    assert (loader.settings.paths.reports_dir / "provider_capabilities.json").is_file()
    assert (loader.settings.paths.reports_dir / "provider_capabilities.csv").is_file()


@pytest.mark.asyncio
async def test_month_timeframe_uses_distinct_storage_and_provider_interval(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)
    calls: list[tuple[str, dict]] = []

    async def request(method, url, params, headers):
        del method, headers
        selected = dict(params or {})
        calls.append((url, selected))
        timestamp = int(
            selected.get(
                "startTime",
                first.timestamp() * 1_000,
            )
        )
        if "bitvavo" in url:
            return [
                [
                    timestamp,
                    "1",
                    "2",
                    "0.5",
                    "1.5",
                    "10",
                ]
            ]
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                int(
                    (
                        first + timedelta(days=30)
                    ).timestamp()
                    * 1_000
                )
                - 1,
                "15",
                5,
            ]
        ]

    loader = DataLoader(settings, requester=request)
    assert loader._cache_path(
        "bitvavo",
        "BTC-EUR",
        "1m",
        "ohlcv",
    ) != loader._cache_path(
        "bitvavo",
        "BTC-EUR",
        "1mo",
        "ohlcv",
    )
    await loader.download_ohlcv(
        provider="bitvavo",
        market="BTC-EUR",
        timeframe="1mo",
        start=first,
        end=first + timedelta(days=60),
        resume=False,
        persist=False,
    )
    await loader.download_ohlcv(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1mo",
        start=first,
        end=first + timedelta(days=60),
        resume=False,
        persist=False,
    )
    assert len(calls) >= 2
    assert {
        params["interval"]
        for _, params in calls
    } == {"1M"}


@pytest.mark.asyncio
async def test_unsupported_native_timeframe_rejected_and_canonical_resample_is_causal(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    loader = DataLoader(settings)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="not native"):
        await loader.download_ohlcv(
            provider="kraken",
            market="BTC-EUR",
            timeframe="2h",
            start=start,
            end=start + timedelta(hours=2),
        )
    source = [
        NormalizedDataRecord(
            provider="kraken",
            source_symbol="XBTEUR",
            canonical_market="BTC-EUR",
            timestamp=start + timedelta(hours=offset),
            observed_at=start + timedelta(hours=offset + 1),
            available_at=start + timedelta(hours=offset + 1),
            data_kind="ohlcv",
            timeframe="1h",
            closed=True,
            retrieval_run_id="unit",
            raw_hash=f"hash-{offset}",
            raw_payload={"offset": offset},
            values={
                "open": offset + 1,
                "high": offset + 2,
                "low": offset + 0.5,
                "close": offset + 1.5,
                "volume": 10,
            },
        )
        for offset in range(4)
    ]
    result = loader.resample_candles(source, target_timeframe="2h")
    assert len(result) == 2
    assert result[0].values["open"] == 1
    assert result[0].values["high"] == 3
    assert result[0].values["low"] == 0.5
    assert result[0].values["close"] == 2.5
    assert result[0].values["volume"] == 20
    assert result[0].available_at == start + timedelta(hours=2)
    assert result[0].values["source_classification"] == "RESAMPLED_FROM_NATIVE"
    assert not result[0].values["gap_flag"]

    three_hour = loader.resample_candles(
        source,
        target_timeframe="3h",
    )
    assert len(three_hour) == 1
    assert three_hour[0].values["open"] == 1
    assert three_hour[0].values["close"] == 3.5
    assert three_hour[0].available_at == start + timedelta(hours=3)


@pytest.mark.asyncio
async def test_continuous_service_requires_explicit_stale_lock_recovery(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    service = ContinuousDataService(settings, heartbeat_seconds=0.01)
    service.lock_path.parent.mkdir(parents=True, exist_ok=True)
    service.lock_path.write_text(
        '{"pid": 2147483647, "started_at": "2020-01-01T00:00:00Z"}',
        encoding="utf-8",
    )
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(
        RuntimeError,
        match="DATA_SERVICE_STALE_LOCK_REQUIRES_EXPLICIT_RECOVERY",
    ):
        await service.start(operation, interval_seconds=0.01, once=True)
    recovery = ContinuousDataService.recover_stale_lock_path(service.lock_path)
    assert recovery["recovered"]
    assert Path(recovery["archive_path"]).is_file()
    await service.start(operation, interval_seconds=0.01, once=True)
    assert calls == 1
    assert service.status()["status"] == "STOPPED"
    assert not service.lock_path.exists()


@pytest.mark.asyncio
async def test_continuous_service_heartbeats_during_long_operation(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    service = ContinuousDataService(
        settings,
        heartbeat_seconds=0.01,
        service_id="long-data-cycle",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(
        service.start(operation, interval_seconds=10.0, once=True)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0.03)
    status = service.status()
    assert status["reason_code"] == "OPERATIONAL_CYCLE_ACTIVE"
    assert status["active_tasks"] == 1
    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert service.status()["status"] == "STOPPED"


def test_continuous_service_lock_inspection_and_owner_safe_release(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    owner = ContinuousDataService(settings, service_id="owner")
    contender = ContinuousDataService(settings, service_id="contender")
    owner._acquire_lock()
    try:
        inspection = ContinuousDataService.inspect_lock_path(owner.lock_path)
        assert not inspection["available"]
        assert not inspection["stale"]
        assert inspection["owner"]["service_id"] == "owner"
        contender._release_lock()
        assert owner.lock_path.exists()
        with pytest.raises(
            RuntimeError,
            match="DATA_SERVICE_SINGLE_INSTANCE_LOCKED",
        ):
            contender._acquire_lock()
    finally:
        owner._release_lock()
    assert not owner.lock_path.exists()


@pytest.mark.asyncio
async def test_continuous_service_pause_resume_drain_and_state_payload(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    database = Database(sqlite_path=tmp_path / "service.db")
    database.migrate()
    service = ContinuousDataService(
        settings,
        database=database,
        heartbeat_seconds=0.01,
        service_id="operate-shadow",
        mode="shadow",
    )
    first_cycle = asyncio.Event()
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        first_cycle.set()

    task = asyncio.create_task(
        service.start(operation, interval_seconds=10.0, once=False)
    )
    await asyncio.wait_for(first_cycle.wait(), timeout=1.0)
    service.pause()
    assert service.status()["state"] == "PAUSED"
    service.resume()
    service.drain()
    await asyncio.wait_for(task, timeout=1.0)
    status = service.status()
    assert status["state"] == "DRAINED"
    assert status["mode"] == "shadow"
    assert status["current_cycle"] >= 1
    assert not service._active_tasks
    assert database.health()["table_counts"]["data_service_state"] == 1
    database.close()


@pytest.mark.asyncio
async def test_continuous_service_polls_external_stop_during_interval(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    service = ContinuousDataService(
        settings,
        heartbeat_seconds=0.01,
        service_id="operate-shadow",
        mode="shadow",
    )
    first_cycle = asyncio.Event()

    async def operation() -> None:
        first_cycle.set()

    task = asyncio.create_task(
        service.start(operation, interval_seconds=60.0, once=False)
    )
    await asyncio.wait_for(first_cycle.wait(), timeout=1.0)
    service.control_path.write_text(
        '{"action":"STOP"}',
        encoding="utf-8",
    )
    await asyncio.wait_for(task, timeout=2.0)
    assert service.status()["state"] == "STOPPED"
    assert service.status()["stop_requested"]


@pytest.mark.asyncio
async def test_kraken_payload_rate_limit_retries_without_stop_iteration(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    attempts = 0

    async def request(method, url, params, headers):
        nonlocal attempts
        del method, url, headers
        attempts += 1
        if attempts == 1:
            return {"error": ["EAPI:Rate limit exceeded"], "result": {}}
        timestamp = int(start.timestamp())
        return {
            "error": [],
            "result": {
                "XXBTZEUR": [
                    [timestamp, "1", "2", "0.5", "1.5", "1", "10", 2]
                ],
                "last": int(params["since"]) + 3_600,
            },
        }

    records = await DataLoader(settings, requester=request).download_ohlcv(
        provider="kraken",
        market="BTC-EUR",
        timeframe="1h",
        start=start,
        end=start + timedelta(hours=1),
        resume=False,
    )
    assert attempts == 2
    assert len(records) == 1
    assert records[0].provider == "kraken"


@pytest.mark.asyncio
async def test_provider_normalization_pagination_and_resume(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    calls: list[dict] = []
    first = datetime(2025, 1, 1, tzinfo=UTC)

    async def request(method, url, params, headers):
        del method, url, headers
        calls.append(dict(params or {}))
        start = int(params["startTime"])
        end = int(params["endTime"])
        rows = []
        for timestamp in range(start, min(end, start + 2 * 3_600_000), 3_600_000):
            rows.append(
                [
                    timestamp,
                    "1",
                    "2",
                    "0.5",
                    "1.5",
                    "10",
                    timestamp + 3_600_000 - 1,
                    "15",
                    5,
                ]
            )
        return rows

    loader = DataLoader(settings, requester=request)
    records = await loader.download_ohlcv(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=5),
        resume=False,
        persist=True,
    )
    assert len(records) == 5
    assert len(calls) == 3
    assert all(record.provider == "mexc" for record in records)
    assert all(record.canonical_market == "BTC-USDT" for record in records)
    assert all(record.raw_hash and record.retrieval_run_id for record in records)
    calls.clear()
    resumed = await loader.download_ohlcv(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=7),
        resume=True,
        persist=False,
    )
    assert len(resumed) == 7
    assert calls[0]["startTime"] == int((first + timedelta(hours=5)).timestamp() * 1_000)


@pytest.mark.asyncio
async def test_bitvavo_backward_pagination_and_resume_backfills_prefix(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)
    calls: list[dict] = []

    async def request(method, url, params, headers):
        del method, url, headers
        selected = dict(params or {})
        calls.append(selected)
        start = int(selected["start"])
        end = int(selected["end"])
        available = [
            int((first + timedelta(hours=hour)).timestamp() * 1_000)
            for hour in range(8)
        ]
        timestamps = [
            value
            for value in available
            if start <= value < end
        ][-2:]
        return [
            [timestamp, "1", "2", "0.5", "1.5", "10"]
            for timestamp in reversed(timestamps)
        ]

    loader = DataLoader(settings, requester=request)
    initial = await loader.download_ohlcv(
        provider="bitvavo",
        market="BTC-EUR",
        timeframe="1h",
        start=first + timedelta(hours=2),
        end=first + timedelta(hours=5),
        resume=False,
        persist=True,
    )
    assert len(initial) == 4
    assert len(calls) == 2

    calls.clear()
    resumed = await loader.download_ohlcv(
        provider="bitvavo",
        market="BTC-EUR",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=7),
        resume=True,
        persist=False,
    )
    assert len(resumed) == 8
    assert {row.timestamp for row in resumed} == {
        first + timedelta(hours=hour) for hour in range(8)
    }
    assert all(
        int(call["end"]) % 3_600_000 == 0
        for call in calls
    )
    assert any(
        call["start"] == int(first.timestamp() * 1_000)
        and call["end"]
        <= int(
            (first + timedelta(hours=2)).timestamp()
            * 1_000
        )
        for call in calls
    )
    assert any(call["start"] >= int((first + timedelta(hours=6)).timestamp() * 1_000) for call in calls)


@pytest.mark.asyncio
async def test_bitvavo_weekly_request_never_uses_future_interval_end(
    isolated_settings: Settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    observed = datetime(2026, 8, 3, 18, tzinfo=UTC)
    calls: list[dict] = []

    async def request(method, url, params, headers):
        del method, url, headers
        calls.append(dict(params or {}))
        return []

    monkeypatch.setattr("data.data_loader.utc_now", lambda: observed)
    await DataLoader(settings, requester=request).download_ohlcv(
        provider="bitvavo",
        market="BTC-EUR",
        timeframe="1W",
        start=observed - timedelta(days=21),
        end=observed,
        resume=False,
    )

    assert calls
    assert all(
        int(call["end"]) <= int(observed.timestamp() * 1_000)
        for call in calls
    )
    bitvavo_monday_boundary_ms = int(
        datetime(2026, 8, 2, 22, tzinfo=UTC).timestamp() * 1_000
    )
    assert all(int(call["end"]) == bitvavo_monday_boundary_ms for call in calls)
    assert all(
        datetime.fromtimestamp(int(call["start"]) / 1_000, tz=UTC).weekday() == 6
        for call in calls
    )


@pytest.mark.asyncio
async def test_compact_native_sync_resumes_without_full_record_materialization(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)
    calls = 0

    async def request(method, url, params, headers):
        nonlocal calls
        del method, url, headers
        calls += 1
        start = int(params["startTime"])
        end = int(params["endTime"])
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                timestamp + 3_600_000 - 1,
                "15",
                5,
            ]
            for timestamp in range(
                start,
                end + 1,
                3_600_000,
            )
        ]

    loader = DataLoader(settings, requester=request)
    first_result = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=5),
        resume=True,
    )
    assert first_result["rows"] == 4
    assert first_result["resource_batching_only"]
    assert (
        settings.paths.cache_dir
        / "mexc"
        / "BTC-USDT_1h_ohlcv.parquet"
    ).is_file()
    assert (
        settings.paths.processed_data_dir
        / "mexc"
        / "BTC-USDT"
        / "1h.parquet"
    ).is_file()

    calls_before_resume = calls
    resumed = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=5),
        resume=True,
    )
    assert resumed["rows"] == 4
    assert calls == calls_before_resume


@pytest.mark.asyncio
async def test_compact_native_sync_persists_confirmed_empty_windows(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    paths = PathSettings(project_root=tmp_path)
    market_data = isolated_settings.market_data.model_copy(
        update={"maximum_database_batch_size": 1}
    )
    settings = isolated_settings.model_copy(
        update={
            "paths": paths,
            "market_data": market_data,
        }
    )
    first = datetime(2023, 1, 1, tzinfo=UTC)
    listing = first + timedelta(hours=10_000)
    calls = 0

    async def request(method, url, params, headers):
        nonlocal calls
        del method, url, headers
        calls += 1
        start = int(params["startTime"])
        end = int(params["endTime"])
        if start < int(listing.timestamp() * 1_000):
            return []
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                timestamp + 3_600_000 - 1,
                "15",
                5,
            ]
            for timestamp in range(
                start,
                end + 1,
                3_600_000,
            )
        ]

    loader = DataLoader(settings, requester=request)
    end = listing + timedelta(hours=4)
    first_result = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="NEW-USDT",
        timeframe="1h",
        start=first,
        end=end,
        resume=True,
    )
    assert first_result["rows"] == 3
    assert calls == 2
    markers = list(
        (
            settings.paths.cache_dir
            / "mexc"
            / ".NEW-USDT_1h_ohlcv.compact_parts"
        ).glob("*.done.json")
    )
    assert len(markers) == 2
    assert any(
        read_json(marker)["status"] == "CONFIRMED_EMPTY"
        for marker in markers
    )

    calls_before_resume = calls
    resumed = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="NEW-USDT",
        timeframe="1h",
        start=first,
        end=end,
        resume=True,
    )
    assert resumed["rows"] == 3
    assert calls == calls_before_resume


@pytest.mark.asyncio
async def test_compact_resample_is_causal_and_drops_incomplete_bucket(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)

    async def request(method, url, params, headers):
        del method, url, headers
        start = int(params["startTime"])
        end = int(params["endTime"])
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                timestamp + 3_600_000 - 1,
                "15",
                5,
            ]
            for timestamp in range(
                start,
                end + 1,
                3_600_000,
            )
            if timestamp
            != int(
                (first + timedelta(hours=4)).timestamp()
                * 1_000
            )
        ]

    loader = DataLoader(settings, requester=request)
    result = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="2h",
        start=first,
        end=first + timedelta(hours=7),
        resume=True,
    )
    assert result["rows"] == 2
    assert result["incomplete_buckets_excluded"] == 1
    target = (
        settings.paths.processed_data_dir
        / "mexc"
        / "BTC-USDT"
        / "2h.parquet"
    )
    frame = loader.load_local_dataset(target)
    assert len(frame) == 2
    assert set(frame["timeframe"]) == {"2h"}
    assert frame["closed"].all()
    resumed = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="2h",
        start=first,
        end=first + timedelta(hours=7),
        resume=True,
    )
    assert resumed["rows"] == 2
    assert resumed["reason_code"] == (
        "COMPACT_RESAMPLE_UP_TO_DATE"
    )


@pytest.mark.asyncio
async def test_compact_calendar_month_resamples_from_complete_daily_bars(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)

    async def request(method, url, params, headers):
        del method, url, headers
        start = int(params["startTime"])
        end = int(params["endTime"])
        day_ms = 86_400_000
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                timestamp + day_ms - 1,
                "15",
                5,
            ]
            for timestamp in range(
                start,
                end + 1,
                day_ms,
            )
        ]

    loader = DataLoader(settings, requester=request)
    result = await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1mo",
        start=first,
        end=datetime(2025, 3, 5, tzinfo=UTC),
        resume=True,
    )
    assert result["source_timeframe"] == "1d"
    assert result["rows"] == 2
    target = (
        settings.paths.processed_data_dir
        / "mexc"
        / "BTC-USDT"
        / "1mo.parquet"
    )
    frame = loader.load_local_dataset(target)
    assert list(
        pd.to_datetime(frame["timestamp"], utc=True)
    ) == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-02-01T00:00:00Z"),
    ]


@pytest.mark.asyncio
async def test_compact_canonical_materialization_is_streamed_and_manifested(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_cache(isolated_settings, tmp_path)
    first = datetime(2025, 1, 1, tzinfo=UTC)

    async def request(method, url, params, headers):
        del method, url, headers
        start = int(params["startTime"])
        end = int(params["endTime"])
        return [
            [
                timestamp,
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                timestamp + 3_600_000 - 1,
                "15",
                5,
            ]
            for timestamp in range(
                start,
                end + 1,
                3_600_000,
            )
        ]

    loader = DataLoader(settings, requester=request)
    await loader.sync_canonical_ohlcv_compact(
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        start=first,
        end=first + timedelta(hours=5),
        resume=True,
    )
    source = (
        settings.paths.processed_data_dir
        / "mexc"
        / "BTC-USDT"
        / "1h.parquet"
    )
    target = (
        settings.paths.processed_data_dir
        / "BTC-USDT_1h.parquet"
    )
    updates: list[dict[str, object]] = []
    result = loader.materialize_provider_ohlcv_compact(
        source,
        target,
        provider="mexc",
        market="BTC-USDT",
        timeframe="1h",
        maximum_staleness=timedelta(days=10_000),
        progress_callback=lambda update: updates.append(
            dict(update)
        ),
    )
    assert result["rows"] == 4
    assert result["resource_batching_only"]
    assert result["reason_code"] == (
        "CANONICAL_FILE_MATERIALIZED_COMPACT"
    )
    canonical = pd.read_parquet(target)
    assert list(canonical.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert len(canonical) == 4
    manifest = read_json(
        target.with_suffix(
            f"{target.suffix}.manifest.json"
        )
    )
    assert manifest["rows"] == 4
    assert manifest["provider"] == "mexc"
    assert manifest["quality"]["duplicate_timestamps"] == 0
    assert updates[-1]["subphase"] == (
        "CANONICAL_MATERIALIZATION_COMPLETE"
    )


def record(provider: str, close: float, timestamp: datetime) -> NormalizedDataRecord:
    return NormalizedDataRecord(
        provider=provider,
        source_symbol="BTCEUR",
        canonical_market="BTC-EUR",
        timestamp=timestamp,
        observed_at=timestamp,
        data_kind="ohlcv",
        timeframe="1h",
        closed=True,
        retrieval_run_id="run",
        raw_hash=f"hash-{provider}",
        values={"close": close},
    )


def test_conflicting_providers_and_stale_records(isolated_settings: Settings) -> None:
    loader = DataLoader(isolated_settings, requester=None)
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    chosen, conflicts = loader.reconcile_provider_series(
        {
            "bitvavo": [record("bitvavo", 10, timestamp)],
            "kraken": [record("kraken", 11, timestamp)],
        }
    )
    assert chosen[0].provider == "bitvavo"
    assert conflicts[0]["status"] == "CONFLICT"
    assert loader.stale_records(chosen, timedelta(hours=1))


@pytest.mark.asyncio
async def test_level2_snapshot_delta_spread_microprice_and_slippage() -> None:
    book = Level2OrderBook(provider="unit", market="BTC-EUR", maximum_depth=5)
    await book.initialize(
        bids=[["99", "2"], ["98", "3"]],
        asks=[["101", "1"], ["102", "4"]],
        sequence=10,
    )
    assert book.spread == Decimal("2")
    assert book.mid_price == Decimal("100")
    assert book.microprice == Decimal("100.3333333333333333333333333")
    assert book.weighted_average_execution_price(
        side="buy", quantity="2"
    ) == Decimal("101.5")
    assert book.estimated_slippage(side="buy", quantity="2") > 0
    assert await book.apply_delta(
        bids=[["99", "3"]],
        asks=[["101", "0"], ["100.5", "2"]],
        sequence=11,
        message_id="delta-1",
    )
    assert book.best_ask == Decimal("100.5")
    assert not await book.apply_delta(
        bids=[["99", "3"]],
        sequence=11,
        message_id="delta-1",
    )


@pytest.mark.asyncio
async def test_level2_gap_invalidates_and_requests_snapshot() -> None:
    refreshed = 0

    async def refresh() -> None:
        nonlocal refreshed
        refreshed += 1

    book = Level2OrderBook(
        provider="unit",
        market="BTC-EUR",
        refresh_snapshot=refresh,
    )
    await book.initialize(bids=[["99", "1"]], asks=[["101", "1"]], sequence=5)
    with pytest.raises(SequenceGap):
        await book.apply_delta(bids=[["99", "2"]], sequence=7)
    assert not book.valid
    assert refreshed == 1
    assert book.statistics["sequence_gaps"] == 1


@pytest.mark.asyncio
async def test_websocket_bounded_queue_duplicates_reconnect_and_resubscription() -> None:
    manager = WebSocketManager(queue_size=1, maximum_connection_attempts=2)
    now = datetime.now(UTC)
    for number in range(2):
        await manager._publish(
            NormalizedStreamEvent(
                event_type=StreamEventType.TICKER,
                provider="bitvavo",
                source_symbol="BTC-EUR",
                canonical_market="BTC-EUR",
                timestamp=now,
                observed_at=now,
                sequence=number + 1,
                message_id=f"message-{number}",
                payload={"price": number},
            )
        )
    assert manager.queue.qsize() == 1
    assert manager.health_state["bitvavo"].dropped_messages == 1
    subscriptions = manager.subscription_messages(
        "kraken", {"ticker": ["BTC/EUR"], "book": ["BTC/EUR"]}
    )
    assert len(subscriptions) == 2
    assert all(item["method"] == "subscribe" for item in subscriptions)

    calls = 0

    async def connection(provider, channels):
        nonlocal calls
        del provider, channels
        calls += 1
        if calls == 1:
            return
        manager._stop.set()

    manager._connection = connection
    await manager.run_provider("bitvavo", {"ticker": ["BTC-EUR"]})
    assert calls == 2
    assert manager.health_state["bitvavo"].reconnects == 1


@pytest.mark.asyncio
async def test_websocket_coalesces_venue_tickers_before_queue_backpressure() -> None:
    manager = WebSocketManager(
        queue_size=2,
        ticker_minimum_interval_seconds=1.0,
    )
    now = datetime.now(UTC)
    for number in range(3):
        await manager._publish(
            NormalizedStreamEvent(
                event_type=StreamEventType.TICKER,
                provider="bitvavo",
                source_symbol="ADA-EUR",
                canonical_market="ADA-EUR",
                timestamp=now,
                observed_at=now,
                message_id=f"ada-ticker-{number}",
                payload={"price": number + 1},
            )
        )

    assert manager.queue.qsize() == 1
    assert manager.health_state["bitvavo"].coalesced_messages == 2
    assert manager.health_state["bitvavo"].dropped_messages == 0
    assert manager.health("bitvavo")["coalesced_messages"] == 2


@pytest.mark.asyncio
async def test_websocket_burst_yields_to_execution_tasks() -> None:
    class Socket:
        def __init__(self) -> None:
            self.received = 0

        async def send_json(self, _payload: dict) -> None:
            return None

        async def receive(self) -> SimpleNamespace:
            self.received += 1
            if self.received >= 55:
                manager._stop.set()
            return SimpleNamespace(
                type=WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "event": "ticker24h",
                        "data": [
                            {
                                "market": "ADA-EUR",
                                "timestamp": int(
                                    datetime.now(UTC).timestamp() * 1_000
                                ),
                                "last": str(self.received),
                                "volume": "1000",
                                "volumeQuote": "1000",
                            }
                        ],
                    }
                ),
            )

        async def ping(self) -> None:
            return None

    class Connection:
        async def __aenter__(self) -> Socket:
            return socket

        async def __aexit__(self, *_args: object) -> None:
            return None

    socket = Socket()
    yielded = asyncio.Event()

    async def competing_execution_task() -> None:
        await asyncio.sleep(0)
        yielded.set()

    manager = WebSocketManager(
        queue_size=100,
        session=SimpleNamespace(),
        connect=lambda *_args, **_kwargs: Connection(),
    )
    manager.health_state["bitvavo"].last_error = "stale disconnect"
    competitor = asyncio.create_task(competing_execution_task())

    await manager._connection("bitvavo", {"ticker": ["ADA-EUR"]})
    await competitor

    assert socket.received == 55
    assert yielded.is_set()
    assert manager.health_state["bitvavo"].messages == 55
    assert manager.health_state["bitvavo"].last_error is None
    health = manager.health("bitvavo")
    assert health["event_counts"]["ticker"] == 55
    assert health["event_last_message_age_ms"]["ticker"] >= 0


@pytest.mark.asyncio
async def test_bitvavo_dynamic_subscription_update_preserves_connection() -> None:
    class Socket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def send_json(self, payload: dict) -> None:
            self.messages.append(payload)

    manager = WebSocketManager()
    socket = Socket()
    manager._sockets["bitvavo"] = socket
    manager._subscriptions["bitvavo"] = {
        "trades": ["OLD-EUR"],
        "book": ["OLD-EUR"],
    }
    manager._sequence[("bitvavo", "NEW-EUR", "book")] = 42

    result = await manager.update_bitvavo_subscriptions(
        subscribe={"trades": ("NEW-EUR",), "book": ("NEW-EUR",)},
        unsubscribe={"trades": ("OLD-EUR",), "book": ("OLD-EUR",)},
    )

    assert [message["action"] for message in socket.messages] == [
        "unsubscribe",
        "subscribe",
    ]
    assert result["connection_preserved"] is True
    assert ("bitvavo", "NEW-EUR", "book") not in manager._sequence
    assert manager.health_state["bitvavo"].subscriptions == 2
    assert manager._subscriptions["bitvavo"] == {
        "trades": ["NEW-EUR"],
        "book": ["NEW-EUR"],
    }


def test_mexc_current_protobuf_ticker_normalization() -> None:
    def varint(value: int) -> bytes:
        encoded = bytearray()
        while value > 0x7F:
            encoded.append((value & 0x7F) | 0x80)
            value >>= 7
        encoded.append(value)
        return bytes(encoded)

    def field(number: int, value: str | bytes) -> bytes:
        payload = value.encode() if isinstance(value, str) else value
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    ticker = (
        field(1, "100")
        + field(2, "2")
        + field(3, "101")
        + field(4, "3")
        + field(5, "44")
    )
    wrapper = (
        field(1, "spot@public.aggre.bookTicker.v3.api.pb@100ms@BTCUSDT")
        + field(3, "BTCUSDT")
        + varint(6 << 3)
        + varint(1_735_689_600_000)
        + field(315, ticker)
    )
    decoded = decode_mexc_protobuf(wrapper)
    assert decoded["symbol"] == "BTCUSDT"
    assert decoded["publicbookticker"]["askPrice"] == "101"
    events = WebSocketManager().parse_message("mexc", decoded)
    assert events[0].event_type is StreamEventType.TICKER
    assert events[0].canonical_market == "BTC-USDT"


def test_bitvavo_candle_subscription_and_plural_event_are_supported() -> None:
    manager = WebSocketManager()
    messages = manager.subscription_messages(
        "bitvavo",
        {
            "ticker": ["BTC-EUR"],
            "candles": {
                "markets": ["BTC-EUR", "ETH-EUR"],
                "interval": ["15m", "1h", "4h", "1d"],
            },
        },
    )
    candle_subscription = messages[0]["channels"][1]
    assert candle_subscription == {
        "name": "candles",
        "markets": ["BTC-EUR", "ETH-EUR"],
        "interval": ["15m", "1h", "4h", "1d"],
    }
    events = manager.parse_message(
        "bitvavo",
        {
            "event": "candles",
            "market": "BTC-EUR",
            "interval": "1h",
            "candle": [
                [
                    "1785283200000",
                    "100",
                    "105",
                    "99",
                    "103",
                    "12.5",
                ]
            ],
        },
    )
    assert len(events) == 1
    assert events[0].event_type is StreamEventType.CANDLE
    assert events[0].payload["interval"] == "1h"
