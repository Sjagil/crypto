from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.contracts import NormalizedStreamEvent, StreamEventType
from data.orderflow_recorder import (
    HashChainedOrderflowLedger,
    ProspectiveOrderflowRecorder,
    normalize_stream_event,
    seal_completed_orderflow_segments,
    summarize_orderflow_hour,
    verify_orderflow_ledger,
)
from data.websocket_manager import WebSocketManager


def event(
    *,
    kind: StreamEventType,
    at: datetime,
    message_id: str,
    payload: dict,
    sequence: int | None = None,
) -> NormalizedStreamEvent:
    return NormalizedStreamEvent(
        event_type=kind,
        provider="bitvavo",
        source_symbol="BTC-EUR",
        canonical_market="BTC-EUR",
        timestamp=at - timedelta(milliseconds=25),
        observed_at=at,
        sequence=sequence,
        message_id=message_id,
        payload=payload,
    )


def test_trade_event_has_forensic_point_in_time_fields() -> None:
    at = datetime(2026, 7, 26, 10, 1, tzinfo=UTC)
    record = normalize_stream_event(
        event(
            kind=StreamEventType.TRADE,
            at=at,
            message_id="trade-1",
            payload={
                "trade_id": "trade-1",
                "price": "100",
                "base_quantity": "0.25",
                "aggressor_side": "buy",
                "raw_payload_hash": "a" * 64,
            },
        )
    )
    assert record["exchange_timestamp"] < record["arrival_timestamp"]
    assert record["available_at"] == record["arrival_timestamp"]
    assert record["quote_quantity"] == "25.00"
    assert record["aggressor_side"] == "buy"
    assert record["raw_payload_hash"] == "a" * 64
    assert record["orders_generated"] == 0


def test_hash_chained_hourly_ledger_and_tamper_detection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stream"
    checkpoint = tmp_path / "checkpoint.json"
    ledger = HashChainedOrderflowLedger(
        root=root,
        checkpoint_path=checkpoint,
    )
    at = datetime(2026, 7, 26, 10, 1, tzinfo=UTC)
    ledger.append(
        [
            event(
                kind=StreamEventType.TRADE,
                at=at,
                message_id="trade-1",
                payload={
                    "price": "100",
                    "quantity": "1",
                    "side": "buy",
                },
            ),
            event(
                kind=StreamEventType.TICKER,
                at=at + timedelta(seconds=1),
                message_id="ticker-1",
                payload={"volume_24h": "1000"},
            ),
        ]
    )
    audit = verify_orderflow_ledger(root)
    assert audit["status"] == "PASSED"
    assert audit["record_count"] == 2
    recovered = HashChainedOrderflowLedger(
        root=root,
        checkpoint_path=checkpoint,
    )
    assert recovered.previous_hash == audit["root_hash"]

    segment = next(root.rglob("*.jsonl"))
    lines = segment.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["market"] = "ETH-EUR"
    lines[0] = json.dumps(payload)
    segment.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_orderflow_ledger(root)["status"] == "FAILED"


def test_closed_segment_is_losslessly_sealed_and_auditable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stream"
    ledger = HashChainedOrderflowLedger(
        root=root,
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    at = datetime(2026, 7, 26, 10, 1, tzinfo=UTC)
    ledger.append(
        [
            event(
                kind=StreamEventType.TRADE,
                at=at,
                message_id="trade-1",
                payload={
                    "price": "100",
                    "quantity": "1",
                    "side": "buy",
                },
            )
        ]
    )

    manifests = seal_completed_orderflow_segments(
        root,
        current_hour=at + timedelta(hours=1),
    )

    assert len(manifests) == 1
    assert manifests[0]["status"] == "SEALED_VERIFIED"
    assert not list(root.rglob("*.jsonl"))
    assert len(list(root.rglob("*.jsonl.xz"))) == 1
    assert verify_orderflow_ledger(root)["status"] == "PASSED"


def test_orderflow_storage_limit_fails_before_append(
    tmp_path: Path,
) -> None:
    ledger = HashChainedOrderflowLedger(
        root=tmp_path / "stream",
        checkpoint_path=tmp_path / "checkpoint.json",
        maximum_storage_bytes=1,
    )
    with pytest.raises(
        RuntimeError,
        match="ORDERFLOW_STORAGE_LIMIT",
    ):
        ledger.append(
            [
                event(
                    kind=StreamEventType.TRADE,
                    at=datetime(2026, 7, 26, 10, tzinfo=UTC),
                    message_id="trade",
                    payload={
                        "price": "100",
                        "quantity": "1",
                        "side": "buy",
                    },
                )
            ]
        )
    assert ledger.record_count == 0
    assert verify_orderflow_ledger(ledger.root)["record_count"] == 0


def test_hour_summary_calculates_delta_obi_and_microprice() -> None:
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    raw = [
        normalize_stream_event(
            event(
                kind=StreamEventType.TRADE,
                at=start + timedelta(minutes=1),
                message_id="buy",
                payload={
                    "trade_id": "buy",
                    "price": "100",
                    "quantity": "2",
                    "side": "buy",
                },
            )
        ),
        normalize_stream_event(
            event(
                kind=StreamEventType.TRADE,
                at=start + timedelta(minutes=2),
                message_id="sell",
                payload={
                    "trade_id": "sell",
                    "price": "99",
                    "quantity": "1",
                    "side": "sell",
                },
            )
        ),
        normalize_stream_event(
            event(
                kind=StreamEventType.TICKER,
                at=start + timedelta(minutes=30),
                message_id="ticker",
                payload={"volume_24h": "1000"},
            )
        ),
        normalize_stream_event(
            event(
                kind=StreamEventType.ORDERBOOK_DELTA,
                at=start + timedelta(minutes=59),
                message_id="book",
                sequence=10,
                payload={
                    "book_bids": [["99", "10"]],
                    "book_asks": [["101", "5"]],
                    "book_state_status": "SEQUENCE_APPLIED",
                },
            )
        ),
    ]
    for index, record in enumerate(raw):
        record["record_hash"] = str(index)
    summary = summarize_orderflow_hour(
        raw,
        market="BTC-EUR",
        hour_start=start,
        recorder_started_at=start,
        health={
            "state": "CONNECTED",
            "sequence_gaps": 0,
            "dropped_messages": 0,
            "reconnects": 0,
        },
    )
    assert summary["status"] == "COMPLETE"
    assert summary["trade_delta_base"] == pytest.approx(1)
    assert summary["trade_delta_quote"] == pytest.approx(101)
    assert summary["trade_delta_percentage"] == pytest.approx(1 / 3)
    assert summary["orderbook_imbalance"] == pytest.approx(1 / 3)
    assert summary["microprice"] == pytest.approx(100.3333333333)


def test_mid_hour_start_and_sequence_gap_fail_closed() -> None:
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    record = normalize_stream_event(
        event(
            kind=StreamEventType.TRADE,
            at=start + timedelta(minutes=30),
            message_id="trade",
            payload={"price": "100", "quantity": "1", "side": "buy"},
        )
    )
    summary = summarize_orderflow_hour(
        [record],
        market="BTC-EUR",
        hour_start=start,
        recorder_started_at=start + timedelta(minutes=20),
        health={
            "state": "CONNECTED",
            "sequence_gaps": 1,
            "dropped_messages": 0,
            "reconnects": 0,
        },
    )
    assert summary["status"] == "DATA_GAP"
    assert "RECORDER_STARTED_MID_HOUR" in summary["reason_codes"]
    assert "SEQUENCE_GAP" in summary["reason_codes"]
    assert not summary["synthetic_data_used"]


def test_orderbook_requires_seed_and_applies_exact_sequence(
    tmp_path: Path,
) -> None:
    recorder = ProspectiveOrderflowRecorder(
        ledger=HashChainedOrderflowLedger(
            root=tmp_path / "stream",
            checkpoint_path=tmp_path / "chain.json",
        ),
        database=FakeDatabase(),
        markets=("BTC-EUR",),
        feature_directory=tmp_path / "features",
        readiness_path=tmp_path / "readiness.json",
    )
    recorder.seed_orderbook(
        SimpleNamespace(
            canonical_market="BTC-EUR",
            raw_hash="s" * 64,
            values={
                "sequence": 10,
                "bids": [["99", "2"]],
                "asks": [["101", "3"]],
            },
        )
    )
    at = datetime(2026, 7, 26, 10, tzinfo=UTC)
    applied, gap = recorder._enrich_orderbooks(
        [
            event(
                kind=StreamEventType.ORDERBOOK_DELTA,
                at=at,
                message_id="book-11",
                sequence=11,
                payload={"bids": [["100", "1"]], "asks": []},
            ),
            event(
                kind=StreamEventType.ORDERBOOK_DELTA,
                at=at + timedelta(seconds=1),
                message_id="book-13",
                sequence=13,
                payload={"bids": [], "asks": [["102", "1"]]},
            ),
        ]
    )
    assert applied.payload["book_state_status"] == "SEQUENCE_APPLIED"
    assert applied.payload["snapshot_reference"] == "s" * 64
    assert applied.payload["book_bids"][0] == ["100", "1"]
    assert gap.payload["book_state_status"] == "SEQUENCE_GAP"
    assert not gap.payload["book_bids"]


def test_bitvavo_parser_preserves_trade_hash_and_quote_volume() -> None:
    manager = WebSocketManager()
    parsed = manager.parse_message(
        "bitvavo",
        {
            "event": "trade",
            "market": "BTC-EUR",
            "timestamp": 1_785_084_800_000,
            "id": "trade-1",
            "price": "50000",
            "amount": "0.01",
            "side": "buy",
        },
    )
    payload = parsed[0].payload
    assert payload["base_quantity"] == "0.01"
    assert payload["quote_quantity"] == "500.00"
    assert payload["aggressor_side"] == "buy"
    assert len(payload["raw_payload_hash"]) == 64


@pytest.mark.parametrize(
    "timestamp",
    [
        1_785_084_800_000,
        1_785_084_800_000_000,
        1_785_084_800_000_000_000,
    ],
)
def test_bitvavo_parser_infers_exchange_timestamp_unit(
    timestamp: int,
) -> None:
    parsed = WebSocketManager().parse_message(
        "bitvavo",
        {
            "event": "trade",
            "market": "BTC-EUR",
            "timestamp": timestamp,
            "id": str(timestamp),
            "price": "50000",
            "amount": "0.01",
            "side": "buy",
        },
    )
    assert (
        parsed[0].timestamp
        == datetime(2026, 7, 26, 16, 53, 20, tzinfo=UTC)
    )


class FakeDatabase:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {}

    def upsert_records(
        self,
        table: str,
        records: list[dict],
    ) -> None:
        self.rows.setdefault(table, []).extend(records)


class FakeStreamManager:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.counters = {
            "state": "CONNECTED",
            "sequence_gaps": 1,
            "dropped_messages": 0,
            "reconnects": 1,
        }

    def health(self, provider=None):
        return dict(self.counters)

    async def next_event(self, timeout):
        return await asyncio.wait_for(
            self.queue.get(),
            timeout=timeout,
        )


@pytest.mark.asyncio
async def test_recorder_pauses_for_reseed_and_acknowledges_recovery(
    tmp_path: Path,
) -> None:
    recorder = ProspectiveOrderflowRecorder(
        ledger=HashChainedOrderflowLedger(
            root=tmp_path / "stream",
            checkpoint_path=tmp_path / "chain.json",
        ),
        database=FakeDatabase(),
        markets=("BTC-EUR",),
        feature_directory=tmp_path / "features",
        readiness_path=tmp_path / "readiness.json",
        flush_seconds=0.05,
    )
    manager = FakeStreamManager()
    task = asyncio.create_task(recorder.run(manager))
    await asyncio.wait_for(recorder.pause(), timeout=1)
    assert recorder._paused.is_set()
    recorder.acknowledge_stream_recovery(manager.health())
    assert recorder._write_health(manager)["status"] == "HEALTHY"
    manager.counters["sequence_gaps"] = 2
    assert recorder._write_health(manager)["status"] == "DEGRADED"
    recorder.resume()
    recorder.stop()
    await asyncio.wait_for(task, timeout=1)


def test_hour_finalization_is_immutable_and_not_research_ready(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    ledger = HashChainedOrderflowLedger(
        root=tmp_path / "stream",
        checkpoint_path=tmp_path / "chain.json",
    )
    ledger.append(
        [
            event(
                kind=StreamEventType.TRADE,
                at=start + timedelta(minutes=1),
                message_id="trade",
                payload={
                    "price": "100",
                    "quantity": "1",
                    "side": "buy",
                },
            ),
            event(
                kind=StreamEventType.TICKER,
                at=start + timedelta(minutes=30),
                message_id="ticker",
                payload={"volume_24h": "1000"},
            ),
            event(
                kind=StreamEventType.ORDERBOOK_DELTA,
                at=start + timedelta(minutes=59),
                message_id="book",
                sequence=10,
                payload={
                    "book_bids": [["99", "10"]],
                    "book_asks": [["101", "5"]],
                    "book_state_status": "SEQUENCE_APPLIED",
                },
            ),
        ]
    )
    positioning = tmp_path / "positioning"
    positioning.mkdir()
    (positioning / "20260726T100000Z.json").write_text(
        json.dumps(
            {
                "derivatives_context": [
                    {
                        "canonical_market": "BTC-USDT",
                        "available_at": (
                            start + timedelta(hours=1)
                        ).isoformat(),
                        "raw_hash": "d" * 64,
                        "values": {
                            "event_time": start.isoformat(),
                            "arrival_time": (
                                start + timedelta(hours=1)
                            ).isoformat(),
                            "funding_rate": 0.0001,
                            "open_interest": 1000,
                            "basis": 1.5,
                            "perpetual_premium": 0.0001,
                            "perpetual_base_volume_24h": 2000,
                            "perpetual_quote_volume_24h": 200000,
                            "liquidation_status": (
                                "UNAVAILABLE_PUBLIC_ENDPOINT"
                            ),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    recorder = ProspectiveOrderflowRecorder(
        ledger=ledger,
        database=FakeDatabase(),
        markets=("BTC-EUR",),
        feature_directory=tmp_path / "features",
        readiness_path=tmp_path / "readiness.json",
        positioning_directory=positioning,
    )
    recorder.started_at = start
    result = recorder.finalize_previous_hour(
        observed_at=start + timedelta(hours=1, minutes=1),
        health={
            "bitvavo": {
                "state": "CONNECTED",
                "sequence_gaps": 0,
                "dropped_messages": 0,
                "reconnects": 0,
            }
        },
    )
    assert result["latest_hour_status"] == "COMPLETE"
    assert result["consecutive_complete_hours"] == 1
    assert not result["backtest_permitted"]
    assert not result["paper_permitted"]
    assert not result["live_permitted"]
    assert result["snapshot"]["orders_generated"] == 0
    repeated = recorder.finalize_previous_hour(
        observed_at=start + timedelta(hours=1, minutes=2),
        health={
            "bitvavo": {
                "state": "CONNECTED",
                "sequence_gaps": 0,
                "dropped_messages": 0,
                "reconnects": 0,
            }
        },
    )
    assert (
        repeated["snapshot"]["snapshot_hash"]
        == result["snapshot"]["snapshot_hash"]
    )
