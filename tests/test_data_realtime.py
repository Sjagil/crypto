from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from config.settings import PathSettings, Settings
from core.contracts import (
    NormalizedDataRecord,
    NormalizedStreamEvent,
    ProviderStatus,
    StreamEventType,
)
from data.data_loader import ContinuousDataService, DataLoader
from data.database import Database
from data.market_data import timeframe_delta
from data.orderbook_l2 import Level2OrderBook, SequenceGap
from data.websocket_manager import WebSocketManager, decode_mexc_protobuf


def isolated_cache(settings: Settings, tmp_path) -> Settings:
    return settings.model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def test_week_and_month_timeframes_do_not_collapse_to_minutes() -> None:
    assert timeframe_delta("1W") == timedelta(days=7)
    assert timeframe_delta("1w") == timedelta(days=7)
    assert timeframe_delta("1M") == timedelta(days=30)
    assert timeframe_delta("1m") == timedelta(minutes=1)


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
    assert "1M" in indexed["mexc"]["native_candle_intervals"]
    assert all(row["status"] in {status.value for status in ProviderStatus} for row in rows)
    assert (loader.settings.paths.reports_dir / "provider_capabilities.json").is_file()
    assert (loader.settings.paths.reports_dir / "provider_capabilities.csv").is_file()


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
        timestamps = [value for value in available if start <= value <= end][-2:]
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
    assert any(call["end"] <= int((first + timedelta(hours=1)).timestamp() * 1_000) for call in calls)
    assert any(call["start"] >= int((first + timedelta(hours=6)).timestamp() * 1_000) for call in calls)


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
