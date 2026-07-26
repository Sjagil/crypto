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
    audit_microstructure_snapshots,
    current_microstructure_readiness,
    normalize_stream_event,
    prospective_milestone_status,
    seal_completed_orderflow_segments,
    summarize_orderflow_hour,
    verify_orderflow_ledger,
)
from data.websocket_manager import WebSocketManager
from utils.common import stable_hash


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
        current_hour=at + timedelta(hours=2),
    )

    assert len(manifests) == 1
    assert manifests[0]["status"] == "SEALED_VERIFIED"
    assert not list(root.rglob("*.jsonl"))
    assert len(list(root.rglob("*.jsonl.xz"))) == 1
    assert verify_orderflow_ledger(root)["status"] == "PASSED"


def test_late_append_is_losslessly_recovered_after_safe_seal(
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
                message_id="first",
                payload={
                    "price": "100",
                    "quantity": "1",
                    "side": "buy",
                },
            )
        ]
    )
    seal_completed_orderflow_segments(
        root,
        current_hour=at + timedelta(hours=2),
    )
    ledger.append(
        [
            event(
                kind=StreamEventType.TRADE,
                at=at + timedelta(minutes=30),
                message_id="late",
                payload={
                    "price": "101",
                    "quantity": "1",
                    "side": "sell",
                },
            )
        ]
    )
    assert verify_orderflow_ledger(root)["status"] == "FAILED"
    recovered = seal_completed_orderflow_segments(
        root,
        current_hour=at + timedelta(hours=2),
    )
    assert recovered[0]["status"] == (
        "SEALED_VERIFIED_LATE_APPEND_RECOVERED"
    )
    audit = verify_orderflow_ledger(root)
    assert audit["status"] == "PASSED"
    assert audit["record_count"] == 2
    assert not list(root.rglob("*.jsonl"))


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
                at=start + timedelta(minutes=58),
                message_id="book",
                sequence=10,
                payload={
                    "book_bids": [["99", "10"]],
                    "book_asks": [["101", "5"]],
                    "book_state_status": "SEQUENCE_APPLIED",
                },
            )
        ),
        normalize_stream_event(
            event(
                kind=StreamEventType.ORDERBOOK_DELTA,
                at=start + timedelta(minutes=59),
                message_id="book-sparse",
                sequence=11,
                payload={
                    "book_bids": [],
                    "book_asks": [],
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
    assert summary["orderbook_sample_count"] == 1
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


def test_bitvavo_ticker24h_preserves_comparable_spot_volume() -> None:
    parsed = WebSocketManager().parse_message(
        "bitvavo",
        {
            "event": "ticker24h",
            "data": {
                "market": "BTC-EUR",
                "timestamp": 1_785_084_800_000,
                "last": "50000",
                "bid": "49999",
                "ask": "50001",
                "volume": "123.45",
                "volumeQuote": "6172500",
            },
        },
    )
    assert len(parsed) == 1
    assert parsed[0].event_type is StreamEventType.TICKER
    assert parsed[0].canonical_market == "BTC-EUR"
    assert parsed[0].payload["ticker_kind"] == "24H"
    assert parsed[0].payload["volume_24h"] == "123.45"
    normalized = normalize_stream_event(parsed[0])
    assert normalized["spot_volume_24h"] == "123.45"
    assert normalized["spot_quote_volume_24h"] == "6172500"


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


def test_hour_finalization_waits_for_positioning_dependency(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    positioning = tmp_path / "positioning"
    recorder = ProspectiveOrderflowRecorder(
        ledger=HashChainedOrderflowLedger(
            root=tmp_path / "stream",
            checkpoint_path=tmp_path / "chain.json",
        ),
        database=FakeDatabase(),
        markets=("BTC-EUR",),
        feature_directory=tmp_path / "features",
        readiness_path=tmp_path / "readiness.json",
        positioning_directory=positioning,
        finalization_grace_minutes=5,
        positioning_timeout_minutes=10,
    )
    recorder.started_at = start
    health = {
        "bitvavo": {
            "state": "CONNECTED",
            "sequence_gaps": 0,
            "dropped_messages": 0,
            "reconnects": 0,
        }
    }
    deferred = recorder.finalize_previous_hour(
        observed_at=start + timedelta(hours=1, minutes=5),
        health=health,
    )
    target = tmp_path / "features" / "20260726T100000Z.json"
    assert deferred["finalization_state"] == "DEFERRED_CONTEXT_PENDING"
    assert deferred["reason_code"] == "POSITIONING_CONTEXT_PENDING"
    assert not deferred["snapshot_written"]
    assert not target.exists()
    assert not (tmp_path / "readiness.json").exists()

    positioning.mkdir()
    (positioning / "20260726T100000Z.json").write_text(
        json.dumps(
            {
                "derivatives_context": [
                    {
                        "canonical_market": "BTC-USDT",
                        "available_at": (
                            start + timedelta(hours=1, minutes=6)
                        ).isoformat(),
                        "raw_hash": "d" * 64,
                        "values": {
                            "funding_rate": 0.0001,
                            "open_interest": 1000,
                            "basis": 1.5,
                            "perpetual_base_volume_24h": 2000,
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
    finalized = recorder.finalize_previous_hour(
        observed_at=start + timedelta(hours=1, minutes=6),
        health=health,
    )
    assert finalized["finalization_state"] == "FINALIZED"
    assert finalized["snapshot"]["positioning_context_status"] == "AVAILABLE"
    assert target.is_file()
    assert all(
        "POSITIONING_CONTEXT_TIMEOUT"
        not in row["reason_codes"]
        for row in finalized["snapshot"]["markets"]
    )


def test_hour_finalization_records_explicit_positioning_timeout(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    recorder = ProspectiveOrderflowRecorder(
        ledger=HashChainedOrderflowLedger(
            root=tmp_path / "stream",
            checkpoint_path=tmp_path / "chain.json",
        ),
        database=FakeDatabase(),
        markets=("BTC-EUR",),
        feature_directory=tmp_path / "features",
        readiness_path=tmp_path / "readiness.json",
        positioning_directory=tmp_path / "positioning",
        finalization_grace_minutes=5,
        positioning_timeout_minutes=10,
    )
    recorder.started_at = start
    result = recorder.finalize_previous_hour(
        observed_at=start + timedelta(hours=1, minutes=10),
        health={
            "bitvavo": {
                "state": "CONNECTED",
                "sequence_gaps": 0,
                "dropped_messages": 0,
                "reconnects": 0,
            }
        },
    )
    assert result["finalization_state"] == "FINALIZED"
    assert result["snapshot"]["positioning_context_status"] == "TIMED_OUT"
    assert result["snapshot"]["status"] == "DATA_GAP"
    assert "POSITIONING_CONTEXT_TIMEOUT" in result["snapshot"]["markets"][0][
        "reason_codes"
    ]


def _write_auditable_snapshot(
    directory: Path,
    hour_start: datetime,
    *,
    complete: bool = True,
    tamper_hash: bool = False,
) -> Path:
    market_status = "COMPLETE" if complete else "DATA_GAP"
    body = {
        "schema_version": "microstructure_hourly_snapshot_v1",
        "hour_start": hour_start.isoformat(),
        "hour_end": (hour_start + timedelta(hours=1)).isoformat(),
        "finalized_at": (
            hour_start + timedelta(hours=1, minutes=1)
        ).isoformat(),
        "status": market_status,
        "markets": [
            {
                "market": "BTC-EUR",
                "hour_start": hour_start.isoformat(),
                "hour_end": (
                    hour_start + timedelta(hours=1)
                ).isoformat(),
                "status": market_status,
                "reason_codes": (
                    [] if complete else ["SEQUENCE_GAP"]
                ),
                "first_arrival_timestamp": (
                    hour_start + timedelta(seconds=30)
                ).isoformat(),
                "last_arrival_timestamp": (
                    hour_start + timedelta(minutes=59)
                ).isoformat(),
                "required_field_coverage": {
                    "spot_cvd_input_available": True,
                    "orderbook_available": True,
                    "funding_available": True,
                    "open_interest_available": True,
                    "basis_available": True,
                },
                "source_record_hashes": ["a" * 64],
            }
        ],
        "stream_health": {
            "state": "CONNECTED",
            "sequence_gaps": 0 if complete else 1,
            "dropped_messages": 0,
            "reconnects": 0,
        },
        "ledger_root_hash": "b" * 64,
        "synthetic_data_used": False,
        "orders_generated": 0,
    }
    snapshot_hash = stable_hash(body, length=64)
    payload = {
        **body,
        "snapshot_hash": (
            "f" * 64 if tamper_hash else snapshot_hash
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / hour_start.strftime(
        "%Y%m%dT%H0000Z.json"
    )
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_readiness_resets_on_latest_gap_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features"
    start = datetime(2026, 7, 26, 10, tzinfo=UTC)
    _write_auditable_snapshot(features, start)
    _write_auditable_snapshot(
        features,
        start + timedelta(hours=1),
    )
    _write_auditable_snapshot(
        features,
        start + timedelta(hours=2),
        complete=False,
    )
    audit = audit_microstructure_snapshots(features)
    assert audit["eligible_complete_hours"] == 2
    assert audit["consecutive_complete_hours"] == 0
    assert audit["excluded_snapshot_count"] == 1
    _write_auditable_snapshot(
        features,
        start + timedelta(hours=3),
        tamper_hash=True,
    )
    readiness = current_microstructure_readiness(features)
    assert readiness["complete_hours"] == 2
    assert readiness["consecutive_complete_hours"] == 0
    assert not readiness["backtest_permitted"]
    assert readiness["milestone_stage"] == "COLLECT"
    assert readiness["snapshot_audit"][
        "excluded_snapshot_count"
    ] == 2
    assert len(readiness["readiness_hash"]) == 64


def test_prospective_milestones_are_strict_and_non_promotional() -> None:
    before = prospective_milestone_status(90 * 24 - 1)
    technical = prospective_milestone_status(90 * 24)
    preliminary = prospective_milestone_status(180 * 24)
    formal = prospective_milestone_status(365 * 24)
    assert before["stage"] == "COLLECT"
    assert (
        technical["stage"]
        == "TECHNICAL_FEATURE_VALIDATION_ELIGIBLE"
    )
    assert (
        preliminary["stage"]
        == "PRELIMINARY_RESEARCH_ELIGIBLE"
    )
    assert (
        formal["stage"]
        == "FORMAL_REGIME_ASSESSMENT_ELIGIBLE"
    )
