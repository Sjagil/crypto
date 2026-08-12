from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.contracts import NormalizedDataRecord, NormalizedStreamEvent, StreamEventType
from data.bitvavo_l2_reconstruction_v2 import BitvavoBookState
from data.market_structure import BookQuality
from data.multi_source_runtime import MultiSourceCollector
from utils.common import stable_hash


class _Manager:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)

    def health(self, provider: str | None = None) -> dict:
        if provider is not None:
            return {"state": "CONNECTED"}
        return {
            name: {"state": "CONNECTED", "messages": 0, "dropped_messages": 0}
            for name in ("bitvavo", "kraken", "mexc")
        }


class _BlockingSnapshotLoader:
    def __init__(self, record: NormalizedDataRecord) -> None:
        self.record = record
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def download_orderbook_snapshot(self, **kwargs) -> NormalizedDataRecord:
        assert kwargs["provider"] == "bitvavo"
        assert kwargs["persist"] is False
        self.started.set()
        await self.release.wait()
        return self.record


def _collector(isolated_settings, tmp_path: Path, loader) -> MultiSourceCollector:
    paths = isolated_settings.paths.model_copy(
        update={
            "raw_data_dir": tmp_path / "raw",
            "data_dir": tmp_path / "data",
            "output_dir": tmp_path / "output",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": paths})
    return MultiSourceCollector(
        settings,
        websocket_manager=_Manager(),
        data_loader=loader,
    )


def _snapshot(at: datetime, nonce: int = 100) -> NormalizedDataRecord:
    payload = {
        "market": "BTC-EUR",
        "nonce": nonce,
        "bids": [["100", "2"]],
        "asks": [["101", "3"]],
    }
    return NormalizedDataRecord(
        provider="bitvavo",
        source_symbol="BTC-EUR",
        canonical_market="BTC-EUR",
        timestamp=at,
        observed_at=at,
        available_at=at,
        data_kind="orderbook_snapshot",
        retrieval_run_id="unit-reseed",
        raw_hash=stable_hash(payload),
        raw_payload=payload,
        values={
            "sequence": nonce,
            "bids": payload["bids"],
            "asks": payload["asks"],
        },
    )


def _delta(at: datetime, nonce: int = 101) -> NormalizedStreamEvent:
    return NormalizedStreamEvent(
        event_type=StreamEventType.ORDERBOOK_DELTA,
        provider="bitvavo",
        source_symbol="BTC-EUR",
        canonical_market="BTC-EUR",
        timestamp=at,
        observed_at=at,
        sequence=nonce,
        message_id=f"BTC-EUR:{nonce}",
        payload={"bids": [["100", "2.5"]], "asks": []},
    )


@pytest.mark.asyncio
async def test_runtime_reseed_buffers_deltas_and_persists_trusted_snapshot(
    isolated_settings,
    tmp_path: Path,
) -> None:
    at = datetime.now(UTC)
    loader = _BlockingSnapshotLoader(_snapshot(at))
    collector = _collector(isolated_settings, tmp_path, loader)
    assert collector.bitvavo_books["BTC-EUR"].state is BitvavoBookState.WAITING_FOR_SNAPSHOT

    reseed = asyncio.create_task(collector._reseed_bitvavo_market("BTC-EUR"))
    await loader.started.wait()
    collector._observe_book(_delta(at + timedelta(milliseconds=1)), "CRYPTO:BTC")
    assert len(collector._bitvavo_buffers["BTC-EUR"]) == 1
    loader.release.set()

    assert await reseed is True
    book = collector.bitvavo_books["BTC-EUR"]
    assert book.state is BitvavoBookState.VALID
    assert book.sequence == 101
    assert book.features(at + timedelta(milliseconds=1))["best_bid"] == "100"
    assert collector.book_coverage[("bitvavo", "CRYPTO:BTC")].current_state is BookQuality.BOOK_VALID
    snapshots = [
        row for row in collector.pending["bitvavo"] if row.data_type == "ORDERBOOK_SNAPSHOT"
    ]
    assert len(snapshots) == 1
    assert snapshots[0].quality_state == "TRUSTED_RESEED_SNAPSHOT"
    assert snapshots[0].metadata["orders_generated"] == 0
    assert snapshots[0].metadata["private_exchange_requests"] == 0


@pytest.mark.asyncio
async def test_failed_reseed_is_fail_closed_and_other_sources_are_unchanged(
    isolated_settings,
    tmp_path: Path,
) -> None:
    class _FailingLoader:
        async def download_orderbook_snapshot(self, **kwargs):
            del kwargs
            raise ConnectionError("unit")

    collector = _collector(isolated_settings, tmp_path, _FailingLoader())
    kraken_before = dict(collector.source_status["kraken"])
    assert await collector._reseed_bitvavo_market("BTC-EUR") is False
    assert collector.bitvavo_books["BTC-EUR"].features(datetime.now(UTC)) is None
    assert collector.source_status["kraken"] == kraken_before
    assert collector.snapshot()["execution"]["new_exchange_mutations"] == 0
