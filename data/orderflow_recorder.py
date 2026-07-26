"""Durable, hash-chained recording of prospective public order flow."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import lzma
import os
import shutil
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.contracts import NormalizedStreamEvent, StreamEventType
from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
    stable_json,
    utc_now,
)

ORDERFLOW_SCHEMA = "prospective_orderflow_event_v1"
ORDERFLOW_CHECKPOINT_SCHEMA = "prospective_orderflow_checkpoint_v1"
MICROSTRUCTURE_SCHEMA = "microstructure_hourly_snapshot_v1"
ZERO_HASH = "0" * 64


def _sha256_file(
    path: Path,
    *,
    decompress: bool = False,
) -> str:
    digest = hashlib.sha256()
    if decompress and ".xz" in path.suffixes:
        opener = lzma.open
    elif decompress and ".gz" in path.suffixes:
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger_segments(
    directory: Path,
) -> tuple[list[Path], list[str]]:
    """Select one physical representation per logical hour segment."""

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in directory.rglob("*.jsonl*"):
        if path.name.endswith(
            (".jsonl", ".jsonl.gz", ".jsonl.xz")
        ):
            logical = str(path)
            for suffix in (".gz", ".xz"):
                logical = logical.removesuffix(suffix)
            grouped[logical].append(path)
    selected: list[Path] = []
    failures: list[str] = []
    for logical_path, representations in sorted(grouped.items()):
        raw = next(
            (
                path
                for path in representations
                if path.name.endswith(".jsonl")
            ),
            None,
        )
        compressed = next(
            (
                path
                for suffix in (".jsonl.xz", ".jsonl.gz")
                for path in representations
                if path.name.endswith(suffix)
            ),
            None,
        )
        if raw is not None and compressed is not None:
            if _sha256_file(raw) != _sha256_file(
                compressed,
                decompress=True,
            ):
                failures.append(
                    f"SEGMENT_REPRESENTATION_MISMATCH:{logical_path}"
                )
            selected.append(compressed)
        elif compressed is not None:
            selected.append(compressed)
        elif raw is not None:
            selected.append(raw)
    return selected, failures


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    if parsed.tzinfo is None:
        raise ValueError("orderflow timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _closed_hour_start(observed_at: datetime) -> datetime:
    normalized = _utc(observed_at).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    return normalized - timedelta(hours=1)


def _decimal(value: Any) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def normalize_stream_event(
    event: NormalizedStreamEvent,
) -> dict[str, Any]:
    """Convert one event into the immutable forensic schema."""

    payload = dict(event.payload)
    event_time = event.timestamp.astimezone(UTC)
    arrival_time = event.observed_at.astimezone(UTC)
    body: dict[str, Any] = {
        "schema_version": ORDERFLOW_SCHEMA,
        "exchange": event.provider,
        "market": event.canonical_market,
        "event_type": event.event_type.value,
        "exchange_timestamp": event_time.isoformat(),
        "arrival_timestamp": arrival_time.isoformat(),
        "available_at": arrival_time.isoformat(),
        "timestamp_quality": (
            "OBSERVED_ONLY"
            if event_time == arrival_time
            else "SOURCE_REPORTED"
        ),
        "latency_ms": max(
            0.0,
            (arrival_time - event_time).total_seconds() * 1_000,
        ),
        "source_sequence": event.sequence,
        "message_id": event.message_id,
        "raw_payload_hash": (
            payload.get("raw_payload_hash")
            or stable_hash(payload, length=64)
        ),
        "orders_generated": 0,
    }
    if event.event_type is StreamEventType.TRADE:
        price = payload.get("price")
        base_quantity = payload.get(
            "base_quantity",
            payload.get("quantity"),
        )
        quote_quantity = payload.get("quote_quantity")
        if quote_quantity is None:
            price_decimal = _decimal(price)
            quantity_decimal = _decimal(base_quantity)
            if price_decimal is not None and quantity_decimal is not None:
                quote_quantity = str(
                    price_decimal * quantity_decimal
                )
        body.update(
            {
                "trade_id": (
                    payload.get("trade_id") or event.message_id
                ),
                "price": price,
                "base_quantity": base_quantity,
                "quote_quantity": quote_quantity,
                "aggressor_side": (
                    payload.get("aggressor_side")
                    or payload.get("side")
                ),
            }
        )
    elif event.event_type in {
        StreamEventType.ORDERBOOK_DELTA,
        StreamEventType.ORDERBOOK_SNAPSHOT,
    }:
        body.update(
            {
                "sequence_start": payload.get(
                    "from_sequence",
                    event.sequence,
                ),
                "sequence_end": payload.get(
                    "to_sequence",
                    event.sequence,
                ),
                "bid_updates": payload.get("bids") or [],
                "ask_updates": payload.get("asks") or [],
                "book_bids": payload.get("book_bids") or [],
                "book_asks": payload.get("book_asks") or [],
                "snapshot_reference": payload.get(
                    "snapshot_reference"
                ),
                "book_state_status": payload.get(
                    "book_state_status"
                ),
                "checksum": payload.get("checksum"),
            }
        )
    elif event.event_type is StreamEventType.TICKER:
        body.update(
            {
                "last_price": payload.get("last_price"),
                "best_bid": payload.get("best_bid"),
                "best_ask": payload.get("best_ask"),
                "spot_volume_24h": payload.get("volume_24h"),
                "spot_quote_volume_24h": payload.get(
                    "quote_volume_24h"
                ),
                "ticker_kind": payload.get("ticker_kind"),
            }
        )
    return body


def verify_orderflow_ledger(
    root: Path | str,
) -> dict[str, Any]:
    """Verify every segment and the global cross-segment hash chain."""

    directory = Path(root)
    segments, failures = _ledger_segments(directory)
    previous_hash = ZERO_HASH
    count = 0
    first_arrival: str | None = None
    last_arrival: str | None = None
    for path in segments:
        if path.name.endswith(".xz"):
            opener = lzma.open
        elif path.name.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = dict(json.loads(line))
                except (TypeError, ValueError):
                    failures.append(
                        f"INVALID_JSON:{path}:{line_number}"
                    )
                    continue
                record_hash = str(record.pop("record_hash", ""))
                if record.get("previous_record_hash") != previous_hash:
                    failures.append(
                        f"CHAIN_BREAK:{path}:{line_number}"
                    )
                expected = stable_hash(record, length=64)
                if record_hash != expected:
                    failures.append(
                        f"HASH_MISMATCH:{path}:{line_number}"
                    )
                previous_hash = record_hash or expected
                arrival = record.get("arrival_timestamp")
                first_arrival = first_arrival or arrival
                last_arrival = arrival or last_arrival
                count += 1
    return {
        "schema_version": "prospective_orderflow_audit_v1",
        "status": "PASSED" if not failures else "FAILED",
        "record_count": count,
        "segment_count": len(segments),
        "root_hash": previous_hash,
        "first_arrival_timestamp": first_arrival,
        "last_arrival_timestamp": last_arrival,
        "integrity_failures": failures,
        "orders_generated": 0,
    }


def _segment_chain_boundary(
    path: Path,
) -> tuple[str | None, str | None, int]:
    if path.name.endswith(".xz"):
        opener = lzma.open
    elif path.name.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    first_previous: str | None = None
    last_hash: str | None = None
    count = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = dict(json.loads(line))
            if first_previous is None:
                first_previous = record.get("previous_record_hash")
            last_hash = record.get("record_hash")
            count += 1
    return first_previous, last_hash, count


def _decompressed_size(path: Path) -> int:
    size = 0
    opener = (
        lzma.open
        if path.name.endswith(".xz")
        else gzip.open
        if path.name.endswith(".gz")
        else open
    )
    with opener(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
    return size


def _recover_late_appended_segment(
    raw: Path,
    compressed: Path,
    manifest_path: Path,
    *,
    segment_hour: datetime,
) -> dict[str, Any]:
    """Losslessly join a sealed prefix and a late-appended suffix."""

    _, compressed_last, compressed_records = (
        _segment_chain_boundary(compressed)
    )
    raw_first_previous, _, raw_records = _segment_chain_boundary(raw)
    if (
        compressed_last is None
        or raw_first_previous != compressed_last
    ):
        raise RuntimeError(
            "ORDERFLOW_LATE_APPEND_CHAIN_BOUNDARY_MISMATCH"
        )
    original_compressed_hash = _sha256_file(compressed)
    late_suffix_hash = _sha256_file(raw)
    expected = hashlib.sha256()
    temporary = compressed.with_suffix(".jsonl.xz.recovery.tmp")
    temporary.unlink(missing_ok=True)
    with lzma.open(compressed, "rb") as sealed_prefix, raw.open(
        "rb"
    ) as late_suffix, lzma.open(
        temporary,
        "wb",
        preset=3,
    ) as target:
        for source in (sealed_prefix, late_suffix):
            while chunk := source.read(1024 * 1024):
                expected.update(chunk)
                target.write(chunk)
    restored_hash = _sha256_file(temporary, decompress=True)
    if restored_hash != expected.hexdigest():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "ORDERFLOW_LATE_APPEND_RECOVERY_MISMATCH"
        )
    temporary.replace(compressed)
    manifest = {
        "schema_version": "orderflow_segment_manifest_v1",
        "status": "SEALED_VERIFIED_LATE_APPEND_RECOVERED",
        "segment_hour": segment_hour.isoformat(),
        "compressed_path": str(compressed),
        "uncompressed_sha256": restored_hash,
        "compressed_sha256": _sha256_file(compressed),
        "original_compressed_sha256": original_compressed_hash,
        "late_suffix_sha256": late_suffix_hash,
        "sealed_prefix_records": compressed_records,
        "late_suffix_records": raw_records,
        "recovered_record_count": (
            compressed_records + raw_records
        ),
        "uncompressed_bytes": _decompressed_size(compressed),
        "compressed_bytes": compressed.stat().st_size,
        "orders_generated": 0,
    }
    atomic_write_json(manifest_path, manifest)
    raw.unlink()
    return manifest


def seal_completed_orderflow_segments(
    root: Path | str,
    *,
    current_hour: datetime,
    seal_lag_hours: int = 2,
) -> list[dict[str, Any]]:
    """Losslessly XZ-seal closed hours and publish verification manifests."""

    directory = Path(root)
    active_hour = _utc(current_hour).replace(
        minute=0,
        second=0,
        microsecond=0,
    ) - timedelta(hours=max(1, seal_lag_hours) - 1)
    sealed: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.jsonl")):
        try:
            relative = path.relative_to(directory)
            year, month, day = (
                int(relative.parts[0]),
                int(relative.parts[1]),
                int(relative.parts[2]),
            )
            segment_hour = datetime(
                year,
                month,
                day,
                int(path.stem),
                tzinfo=UTC,
            )
        except (IndexError, TypeError, ValueError):
            continue
        if segment_hour >= active_hour:
            continue
        compressed = path.with_suffix(".jsonl.xz")
        manifest_path = path.with_suffix(".manifest.json")
        original_hash = _sha256_file(path)
        if compressed.is_file():
            restored_hash = _sha256_file(
                compressed,
                decompress=True,
            )
            if restored_hash != original_hash:
                sealed.append(
                    _recover_late_appended_segment(
                        path,
                        compressed,
                        manifest_path,
                        segment_hour=segment_hour,
                    )
                )
                continue
            compressed_hash = _sha256_file(compressed)
            manifest = {
                "schema_version": "orderflow_segment_manifest_v1",
                "status": "SEALED_VERIFIED_RECOVERED",
                "segment_hour": segment_hour.isoformat(),
                "compressed_path": str(compressed),
                "uncompressed_sha256": original_hash,
                "compressed_sha256": compressed_hash,
                "uncompressed_bytes": path.stat().st_size,
                "compressed_bytes": compressed.stat().st_size,
                "orders_generated": 0,
            }
            atomic_write_json(manifest_path, manifest)
            path.unlink()
            sealed.append(manifest)
            continue
        temporary = compressed.with_suffix(".jsonl.xz.tmp")
        temporary.unlink(missing_ok=True)
        with path.open("rb") as source, lzma.open(
            temporary,
            "wb",
            preset=3,
        ) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        restored_hash = _sha256_file(
            temporary,
            decompress=True,
        )
        if restored_hash != original_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "ORDERFLOW_SEGMENT_COMPRESSION_MISMATCH"
            )
        temporary.replace(compressed)
        compressed_hash = _sha256_file(compressed)
        manifest = {
            "schema_version": "orderflow_segment_manifest_v1",
            "status": "SEALED_VERIFIED",
            "segment_hour": segment_hour.isoformat(),
            "compressed_path": str(compressed),
            "uncompressed_sha256": original_hash,
            "compressed_sha256": compressed_hash,
            "uncompressed_bytes": path.stat().st_size,
            "compressed_bytes": compressed.stat().st_size,
            "orders_generated": 0,
        }
        atomic_write_json(manifest_path, manifest)
        path.unlink()
        sealed.append(manifest)
    return sealed


class HashChainedOrderflowLedger:
    """Single-writer, hourly segmented append-only event ledger."""

    def __init__(
        self,
        *,
        root: Path,
        checkpoint_path: Path,
        maximum_storage_bytes: int | None = None,
    ) -> None:
        self.root = root
        self.checkpoint_path = checkpoint_path
        self.maximum_storage_bytes = maximum_storage_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        audit = verify_orderflow_ledger(self.root)
        if audit["status"] != "PASSED":
            raise RuntimeError("ORDERFLOW_LEDGER_INTEGRITY_FAILED")
        self.previous_hash = str(audit["root_hash"])
        self.record_count = int(audit["record_count"])
        self.last_arrival_timestamp = audit[
            "last_arrival_timestamp"
        ]
        self.storage_bytes = 0
        self.refresh_storage_bytes()
        if checkpoint_path.is_file():
            checkpoint = dict(read_json(checkpoint_path))
            checkpoint_count = int(
                checkpoint.get("record_count") or 0
            )
            if checkpoint_count > self.record_count or (
                checkpoint_count == self.record_count
                and checkpoint.get("root_hash")
                != self.previous_hash
            ):
                raise RuntimeError(
                    "ORDERFLOW_LEDGER_CHECKPOINT_MISMATCH"
                )
            if checkpoint_count < self.record_count:
                atomic_write_json(
                    checkpoint_path,
                    {
                        "schema_version": (
                            ORDERFLOW_CHECKPOINT_SCHEMA
                        ),
                        "status": "RECOVERED_AFTER_APPEND",
                        "root_hash": self.previous_hash,
                        "record_count": self.record_count,
                        "last_arrival_timestamp": audit[
                            "last_arrival_timestamp"
                        ],
                        "orders_generated": 0,
                    },
                )

    def refresh_storage_bytes(self) -> int:
        self.storage_bytes = sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )
        return self.storage_bytes

    def _segment(self, arrival: datetime) -> Path:
        return (
            self.root
            / arrival.strftime("%Y")
            / arrival.strftime("%m")
            / arrival.strftime("%d")
            / f"{arrival:%H}.jsonl"
        )

    def append(
        self,
        events: Iterable[NormalizedStreamEvent],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        grouped: dict[Path, list[bytes]] = defaultdict(
            list
        )
        next_hash = self.previous_hash
        next_count = self.record_count
        for event in events:
            body = normalize_stream_event(event)
            body["previous_record_hash"] = next_hash
            record_hash = stable_hash(body, length=64)
            record = {**body, "record_hash": record_hash}
            arrival = _utc(record["arrival_timestamp"])
            grouped[self._segment(arrival)].append(
                f"{stable_json(record)}\n".encode("utf-8")
            )
            records.append(record)
            next_hash = record_hash
            next_count += 1
        append_bytes = sum(
            len(encoded)
            for selected in grouped.values()
            for encoded in selected
        )
        if (
            self.maximum_storage_bytes is not None
            and self.storage_bytes + append_bytes
            > self.maximum_storage_bytes
        ):
            raise RuntimeError("ORDERFLOW_STORAGE_LIMIT")
        for path, selected in grouped.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                for encoded in selected:
                    stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        self.previous_hash = next_hash
        self.record_count = next_count
        self.storage_bytes += append_bytes
        if records:
            self.last_arrival_timestamp = records[-1][
                "arrival_timestamp"
            ]
            atomic_write_json(
                self.checkpoint_path,
                {
                    "schema_version": ORDERFLOW_CHECKPOINT_SCHEMA,
                    "status": "RECORDING",
                    "root_hash": self.previous_hash,
                    "record_count": self.record_count,
                    "last_arrival_timestamp": records[-1][
                        "arrival_timestamp"
                    ],
                    "last_segment": str(
                        self._segment(
                            _utc(
                                records[-1][
                                    "arrival_timestamp"
                                ]
                            )
                        )
                    ),
                    "orders_generated": 0,
                },
            )
        return records


def _book_metrics(
    record: Mapping[str, Any] | None,
    *,
    depth: int = 10,
) -> dict[str, Any]:
    if not record:
        return {
            "orderbook_imbalance": None,
            "microprice": None,
            "spread_bps": None,
            "book_sequence": None,
        }
    bids = list(
        record.get("book_bids")
        or record.get("bid_updates")
        or []
    )[:depth]
    asks = list(
        record.get("book_asks")
        or record.get("ask_updates")
        or []
    )[:depth]
    parsed_bids = [
        (_decimal(row[0]), _decimal(row[1]))
        for row in bids
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    parsed_asks = [
        (_decimal(row[0]), _decimal(row[1]))
        for row in asks
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    parsed_bids = [
        (price, quantity)
        for price, quantity in parsed_bids
        if price is not None and quantity is not None
    ]
    parsed_asks = [
        (price, quantity)
        for price, quantity in parsed_asks
        if price is not None and quantity is not None
    ]
    if not parsed_bids or not parsed_asks:
        return {
            "orderbook_imbalance": None,
            "microprice": None,
            "spread_bps": None,
            "book_sequence": record.get("sequence_end"),
        }
    bid_total = sum((row[1] for row in parsed_bids), Decimal(0))
    ask_total = sum((row[1] for row in parsed_asks), Decimal(0))
    total = bid_total + ask_total
    best_bid, best_bid_quantity = parsed_bids[0]
    best_ask, best_ask_quantity = parsed_asks[0]
    top_total = best_bid_quantity + best_ask_quantity
    mid = (best_bid + best_ask) / Decimal(2)
    return {
        "orderbook_imbalance": (
            float((bid_total - ask_total) / total)
            if total > 0
            else None
        ),
        "microprice": (
            float(
                (
                    best_ask * best_bid_quantity
                    + best_bid * best_ask_quantity
                )
                / top_total
            )
            if top_total > 0
            else None
        ),
        "spread_bps": (
            float((best_ask - best_bid) / mid * Decimal(10_000))
            if mid > 0
            else None
        ),
        "book_sequence": record.get("sequence_end"),
    }


def summarize_orderflow_hour(
    records: Iterable[Mapping[str, Any]],
    *,
    market: str,
    hour_start: datetime,
    recorder_started_at: datetime,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Calculate causal facts and explicitly grade hour completeness."""

    start = _utc(hour_start)
    end = start + timedelta(hours=1)
    selected = [
        dict(record)
        for record in records
        if record.get("market") == market
        and start
        <= _utc(record["arrival_timestamp"])
        < end
    ]
    trades = [
        record
        for record in selected
        if record.get("event_type") == StreamEventType.TRADE.value
    ]
    all_books = [
        record
        for record in selected
        if record.get("event_type")
        in {
            StreamEventType.ORDERBOOK_DELTA.value,
            StreamEventType.ORDERBOOK_SNAPSHOT.value,
        }
    ]
    books = [
        record
        for record in all_books
        if record.get("book_state_status")
        == "SEQUENCE_APPLIED"
    ]
    book_samples = [
        record
        for record in books
        if record.get("book_bids") and record.get("book_asks")
    ]
    tickers = [
        record
        for record in selected
        if record.get("event_type") == StreamEventType.TICKER.value
    ]
    volume_tickers = [
        record
        for record in tickers
        if _decimal(record.get("spot_volume_24h")) is not None
    ]
    buy_base = Decimal(0)
    sell_base = Decimal(0)
    buy_quote = Decimal(0)
    sell_quote = Decimal(0)
    unique_trades: set[str] = set()
    for record in trades:
        identity = str(
            record.get("trade_id")
            or record.get("raw_payload_hash")
        )
        if identity in unique_trades:
            continue
        unique_trades.add(identity)
        base = _decimal(record.get("base_quantity")) or Decimal(0)
        quote = _decimal(record.get("quote_quantity")) or Decimal(0)
        if str(record.get("aggressor_side")).casefold() == "buy":
            buy_base += base
            buy_quote += quote
        elif str(record.get("aggressor_side")).casefold() == "sell":
            sell_base += base
            sell_quote += quote
    total_base = buy_base + sell_base
    total_quote = buy_quote + sell_quote
    arrivals = sorted(
        _utc(record["arrival_timestamp"]) for record in selected
    )
    started = _utc(recorder_started_at)
    reasons: list[str] = []
    if started > start:
        reasons.append("RECORDER_STARTED_MID_HOUR")
    if not selected:
        reasons.append("NO_STREAM_EVENTS")
    if not trades:
        reasons.append("NO_TRADES")
    if not book_samples:
        reasons.append("NO_VALID_ORDERBOOK")
    if not tickers:
        reasons.append("NO_TICKER")
    elif not volume_tickers:
        reasons.append("NO_24H_SPOT_VOLUME")
    if any(
        record.get("book_state_status")
        in {
            "SNAPSHOT_MISSING",
            "SEQUENCE_UNAVAILABLE",
            "SEQUENCE_GAP",
        }
        for record in all_books
    ):
        reasons.append("BOOK_SEQUENCE_INVALID")
    if int(health.get("sequence_gaps") or 0):
        reasons.append("SEQUENCE_GAP")
    if int(health.get("dropped_messages") or 0):
        reasons.append("DROPPED_MESSAGES")
    if int(health.get("reconnects") or 0):
        reasons.append("STREAM_RECONNECTED")
    if str(health.get("state")) not in {"CONNECTED", "STOPPED"}:
        reasons.append("STREAM_NOT_HEALTHY")
    return {
        "market": market,
        "hour_start": start.isoformat(),
        "hour_end": end.isoformat(),
        "status": "COMPLETE" if not reasons else "DATA_GAP",
        "reason_codes": reasons,
        "stream_event_count": len(selected),
        "trade_count": len(unique_trades),
        "first_arrival_timestamp": (
            arrivals[0].isoformat() if arrivals else None
        ),
        "last_arrival_timestamp": (
            arrivals[-1].isoformat() if arrivals else None
        ),
        "aggressive_buy_base_volume": float(buy_base),
        "aggressive_sell_base_volume": float(sell_base),
        "trade_delta_base": float(buy_base - sell_base),
        "trade_delta_quote": float(buy_quote - sell_quote),
        "trade_delta_percentage": (
            float((buy_base - sell_base) / total_base)
            if total_base > 0
            else None
        ),
        "spot_base_volume": float(total_base),
        "spot_quote_volume": float(total_quote),
        "spot_volume_24h": (
            volume_tickers[-1].get("spot_volume_24h")
            if volume_tickers
            else None
        ),
        "orderbook_sample_count": len(book_samples),
        **_book_metrics(
            book_samples[-1] if book_samples else None
        ),
        "source_record_hashes": [
            record.get("record_hash") for record in selected
        ],
        "synthetic_data_used": False,
        "orders_generated": 0,
    }


def _read_segment(
    root: Path,
    hour_start: datetime,
) -> list[dict[str, Any]]:
    path = (
        root
        / hour_start.strftime("%Y")
        / hour_start.strftime("%m")
        / hour_start.strftime("%d")
        / f"{hour_start:%H}.jsonl"
    )
    xz_path = path.with_suffix(".jsonl.xz")
    gzip_path = path.with_suffix(".jsonl.gz")
    if not any(
        selected.is_file()
        for selected in (path, xz_path, gzip_path)
    ):
        return []
    selected = next(
        selected
        for selected in (path, xz_path, gzip_path)
        if selected.is_file()
    )
    if selected.name.endswith(".xz"):
        opener = lzma.open
    elif selected.name.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open
    with opener(selected, "rt", encoding="utf-8") as stream:
        return [
            dict(json.loads(line))
            for line in stream
            if line.strip()
        ]


def _positioning_context(
    directory: Path | None,
    *,
    hour_start: datetime,
) -> dict[str, dict[str, Any]]:
    if directory is None or not directory.is_dir():
        return {}
    current_path = directory / (
        hour_start.strftime("%Y%m%dT%H0000Z") + ".json"
    )
    if not current_path.is_file():
        return {}
    current = dict(read_json(current_path))
    prior_payloads = [
        dict(read_json(path))
        for path in sorted(directory.glob("*.json"))
        if path < current_path
    ][-30 * 24 :]
    prior_by_market: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for payload in prior_payloads:
        for record in payload.get("derivatives_context") or []:
            prior_by_market[str(record["canonical_market"])].append(
                dict(record)
            )
    result: dict[str, dict[str, Any]] = {}
    for record in current.get("derivatives_context") or []:
        market = str(record["canonical_market"])
        values = dict(record.get("values") or {})
        prior = prior_by_market.get(market, [])
        prior_rates = [
            float(item["values"]["funding_rate"])
            for item in prior
            if (item.get("values") or {}).get("funding_rate")
            is not None
        ]
        funding_zscore = None
        if len(prior_rates) >= 30:
            median = statistics.median(prior_rates)
            quartiles = statistics.quantiles(
                prior_rates,
                n=4,
                method="inclusive",
            )
            iqr = quartiles[2] - quartiles[0]
            if iqr > 0:
                funding_zscore = (
                    float(values.get("funding_rate") or 0) - median
                ) / iqr
        previous_oi = (
            (prior[-1].get("values") or {}).get("open_interest")
            if prior
            else None
        )
        current_oi = values.get("open_interest")
        oi_change = None
        if (
            previous_oi is not None
            and current_oi is not None
            and float(previous_oi) != 0
        ):
            oi_change = (
                float(current_oi) / float(previous_oi) - 1.0
            )
        result[market.split("-", 1)[0]] = {
            "derivatives_market": market,
            "event_time": values.get("event_time"),
            "arrival_time": values.get("arrival_time"),
            "available_at": record.get("available_at"),
            "raw_payload_hash": record.get("raw_hash"),
            "funding_rate": values.get("funding_rate"),
            "funding_zscore": funding_zscore,
            "funding_zscore_prior_observations": len(prior_rates),
            "open_interest": current_oi,
            "open_interest_change": oi_change,
            "perpetual_basis": values.get("basis"),
            "perpetual_premium": values.get(
                "perpetual_premium"
            ),
            "perpetual_base_volume_24h": values.get(
                "perpetual_base_volume_24h"
            ),
            "perpetual_quote_volume_24h": values.get(
                "perpetual_quote_volume_24h"
            ),
            "long_liquidations": values.get("long_liquidations"),
            "short_liquidations": values.get(
                "short_liquidations"
            ),
            "liquidation_status": values.get(
                "liquidation_status"
            ),
        }
    return result


def _apply_cvd_history(
    feature_directory: Path,
    *,
    target_path: Path,
    markets: list[dict[str, Any]],
) -> None:
    prior_by_market: dict[str, list[float]] = defaultdict(list)
    for path in sorted(feature_directory.glob("*.json")):
        if path >= target_path:
            continue
        payload = dict(read_json(path))
        for row in payload.get("markets") or []:
            if (
                row.get("status") == "COMPLETE"
                and row.get("spot_cvd_cumulative_base")
                is not None
            ):
                prior_by_market[str(row["market"])].append(
                    float(row["spot_cvd_cumulative_base"])
                )
    for row in markets:
        prior = prior_by_market.get(str(row["market"]), [])
        if row.get("status") != "COMPLETE":
            row["spot_cvd_cumulative_base"] = None
            row["spot_cvd_robust_zscore"] = None
            row["spot_cvd_prior_observations"] = len(prior)
            continue
        current = (
            (prior[-1] if prior else 0.0)
            + float(row["trade_delta_base"])
        )
        row["spot_cvd_cumulative_base"] = current
        row["spot_cvd_prior_observations"] = len(prior)
        row["spot_cvd_robust_zscore"] = None
        window = prior[-96:]
        if len(window) >= 96:
            median = statistics.median(window)
            quartiles = statistics.quantiles(
                window,
                n=4,
                method="inclusive",
            )
            iqr = quartiles[2] - quartiles[0]
            if iqr > 0:
                row["spot_cvd_robust_zscore"] = (
                    current - median
                ) / iqr


def _snapshot_source_hashes(
    ledger_root: Path,
    hour_start: datetime,
) -> set[str]:
    return {
        str(record.get("record_hash"))
        for record in _read_segment(ledger_root, hour_start)
        if record.get("record_hash")
    }


def _audit_hourly_snapshot(
    path: Path,
    *,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Verify one immutable hour before it can advance readiness."""

    reasons: list[str] = []
    try:
        payload = dict(read_json(path))
    except (OSError, TypeError, ValueError) as exc:
        return {
            "path": str(path),
            "hour_start": None,
            "eligible": False,
            "reason_codes": [f"UNREADABLE_SNAPSHOT:{type(exc).__name__}"],
        }
    if payload.get("schema_version") != MICROSTRUCTURE_SCHEMA:
        reasons.append("UNSUPPORTED_SCHEMA")
    snapshot_hash = payload.get("snapshot_hash")
    hash_body = {
        key: value
        for key, value in payload.items()
        if key != "snapshot_hash"
    }
    if (
        not isinstance(snapshot_hash, str)
        or snapshot_hash != stable_hash(hash_body, length=64)
    ):
        reasons.append("SNAPSHOT_HASH_MISMATCH")
    try:
        hour_start = _utc(payload["hour_start"])
        hour_end = _utc(payload["hour_end"])
    except (KeyError, TypeError, ValueError):
        hour_start = None
        hour_end = None
        reasons.append("INVALID_HOUR_BOUNDARY")
    if hour_start is not None:
        expected_name = hour_start.strftime("%Y%m%dT%H0000Z.json")
        if path.name != expected_name:
            reasons.append("FILENAME_HOUR_MISMATCH")
        if hour_end != hour_start + timedelta(hours=1):
            reasons.append("INVALID_HOUR_DURATION")
    if payload.get("status") != "COMPLETE":
        reasons.append("SNAPSHOT_NOT_COMPLETE")
    if payload.get("synthetic_data_used") is not False:
        reasons.append("SYNTHETIC_DATA_NOT_EXCLUDED")
    if int(payload.get("orders_generated") or 0) != 0:
        reasons.append("ORDER_SIDE_EFFECT_DETECTED")
    health = dict(payload.get("stream_health") or {})
    if str(health.get("state")) not in {"CONNECTED", "STOPPED"}:
        reasons.append("STREAM_NOT_HEALTHY")
    for counter in ("sequence_gaps", "dropped_messages", "reconnects"):
        if int(health.get(counter) or 0) != 0:
            reasons.append(f"STREAM_{counter.upper()}")
    markets = payload.get("markets")
    if not isinstance(markets, list) or not markets:
        reasons.append("MARKETS_MISSING")
        markets = []
    source_hashes: set[str] = set()
    for row in markets:
        market = str(row.get("market") or "UNKNOWN")
        if row.get("status") != "COMPLETE":
            reasons.append(f"MARKET_NOT_COMPLETE:{market}")
        if row.get("reason_codes"):
            reasons.append(f"MARKET_HAS_GAP_REASON:{market}")
        if hour_start is not None:
            try:
                first = _utc(row["first_arrival_timestamp"])
                last = _utc(row["last_arrival_timestamp"])
            except (KeyError, TypeError, ValueError):
                reasons.append(f"ARRIVAL_COVERAGE_MISSING:{market}")
            else:
                if first > hour_start + timedelta(minutes=5):
                    reasons.append(f"ARRIVAL_START_LATE:{market}")
                if last < hour_start + timedelta(minutes=55):
                    reasons.append(f"ARRIVAL_END_EARLY:{market}")
        coverage = dict(row.get("required_field_coverage") or {})
        for field in (
            "spot_cvd_input_available",
            "orderbook_available",
            "funding_available",
            "open_interest_available",
            "basis_available",
        ):
            if coverage.get(field) is not True:
                reasons.append(f"REQUIRED_FIELD_MISSING:{market}:{field}")
        row_hashes = row.get("source_record_hashes")
        if not isinstance(row_hashes, list) or not row_hashes:
            reasons.append(f"SOURCE_RECORD_HASHES_MISSING:{market}")
            continue
        for record_hash in row_hashes:
            if (
                not isinstance(record_hash, str)
                or len(record_hash) != 64
            ):
                reasons.append(f"SOURCE_RECORD_HASH_INVALID:{market}")
            else:
                source_hashes.add(record_hash)
    if ledger_root is not None and hour_start is not None:
        ledger_hashes = _snapshot_source_hashes(
            ledger_root,
            hour_start,
        )
        if not ledger_hashes:
            reasons.append("LEDGER_SEGMENT_MISSING")
        elif not source_hashes.issubset(ledger_hashes):
            reasons.append("SOURCE_RECORD_NOT_IN_LEDGER")
    unique_reasons = sorted(set(reasons))
    return {
        "path": str(path),
        "hour_start": (
            hour_start.isoformat() if hour_start is not None else None
        ),
        "eligible": not unique_reasons,
        "reason_codes": unique_reasons,
        "snapshot_hash": snapshot_hash,
    }


def audit_microstructure_snapshots(
    feature_directory: Path,
    *,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Return the fail-closed readiness evidence from all sealed hours."""

    audits = [
        _audit_hourly_snapshot(path, ledger_root=ledger_root)
        for path in sorted(feature_directory.glob("*.json"))
    ]
    eligible_epochs = [
        _utc(row["hour_start"])
        for row in audits
        if row["eligible"] and row["hour_start"] is not None
    ]
    consecutive = 0
    if audits:
        expected: datetime | None = None
        for row in reversed(audits):
            if not row["eligible"] or row["hour_start"] is None:
                break
            epoch = _utc(row["hour_start"])
            if expected is not None and epoch != expected:
                break
            consecutive += 1
            expected = epoch - timedelta(hours=1)
    excluded = [row for row in audits if not row["eligible"]]
    return {
        "audited_snapshot_count": len(audits),
        "eligible_complete_hours": len(eligible_epochs),
        "consecutive_complete_hours": consecutive,
        "first_complete_hour": (
            min(eligible_epochs).isoformat()
            if eligible_epochs
            else None
        ),
        "last_complete_hour": (
            max(eligible_epochs).isoformat()
            if eligible_epochs
            else None
        ),
        "latest_snapshot_hour": (
            audits[-1]["hour_start"] if audits else None
        ),
        "latest_snapshot_eligible": (
            bool(audits[-1]["eligible"]) if audits else False
        ),
        "excluded_snapshot_count": len(excluded),
        "excluded_snapshots": excluded,
    }


def prospective_milestone_status(
    consecutive_hours: int,
) -> dict[str, Any]:
    definitions = (
        ("technical_feature_validation", 90),
        ("preliminary_research", 180),
        ("formal_regime_assessment", 365),
    )
    milestones: dict[str, Any] = {}
    for name, days in definitions:
        required_hours = days * 24
        remaining_hours = max(0, required_hours - consecutive_hours)
        milestones[name] = {
            "minimum_consecutive_days": days,
            "required_complete_hours": required_hours,
            "eligible": consecutive_hours >= required_hours,
            "remaining_complete_hours": remaining_hours,
            "remaining_complete_days_equivalent": (
                remaining_hours / 24
            ),
        }
    if milestones["formal_regime_assessment"]["eligible"]:
        stage = "FORMAL_REGIME_ASSESSMENT_ELIGIBLE"
    elif milestones["preliminary_research"]["eligible"]:
        stage = "PRELIMINARY_RESEARCH_ELIGIBLE"
    elif milestones["technical_feature_validation"]["eligible"]:
        stage = "TECHNICAL_FEATURE_VALIDATION_ELIGIBLE"
    else:
        stage = "COLLECT"
    return {
        "stage": stage,
        "milestones": milestones,
    }


def _readiness_payload(
    *,
    feature_directory: Path,
    target: Path,
    snapshot: Mapping[str, Any],
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    audit = audit_microstructure_snapshots(
        feature_directory,
        ledger_root=ledger_root,
    )
    complete_hours = int(audit["eligible_complete_hours"])
    consecutive_complete_hours = int(
        audit["consecutive_complete_hours"]
    )
    technical_ready = consecutive_complete_hours >= 90 * 24
    milestone = prospective_milestone_status(
        consecutive_complete_hours
    )
    body = {
        "schema_version": "microstructure_readiness_v2",
        "status": (
            "TECHNICAL_FEATURE_VALIDATION_READY"
            if technical_ready
            else "COLLECTING_PROSPECTIVE_DATA"
        ),
        "milestone_stage": milestone["stage"],
        "latest_hour_status": snapshot["status"],
        "latest_snapshot": str(target),
        "complete_hours": complete_hours,
        "consecutive_complete_hours": consecutive_complete_hours,
        "complete_days_equivalent": complete_hours / 24,
        "consecutive_complete_days_equivalent": (
            consecutive_complete_hours / 24
        ),
        "first_complete_hour": audit["first_complete_hour"],
        "last_complete_hour": audit["last_complete_hour"],
        "minimum_days": {
            "technical_feature_validation": 90,
            "preliminary_research": 180,
            "formal_regime_assessment": 365,
        },
        "milestones": milestone["milestones"],
        "snapshot_audit": audit,
        "backtest_permitted": technical_ready,
        "paper_permitted": False,
        "live_permitted": False,
        "synthetic_data_used": False,
        "orders_generated": 0,
    }
    return {
        **body,
        "readiness_hash": stable_hash(body, length=64),
    }


def current_microstructure_readiness(
    feature_directory: Path,
    *,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Rebuild readiness from immutable snapshots, never from stale state."""

    targets = sorted(feature_directory.glob("*.json"))
    if targets:
        target = targets[-1]
        snapshot = dict(read_json(target))
        return _readiness_payload(
            feature_directory=feature_directory,
            target=target,
            snapshot=snapshot,
            ledger_root=ledger_root,
        )
    milestone = prospective_milestone_status(0)
    audit = audit_microstructure_snapshots(
        feature_directory,
        ledger_root=ledger_root,
    )
    body = {
        "schema_version": "microstructure_readiness_v2",
        "status": "COLLECTING_PROSPECTIVE_DATA",
        "milestone_stage": milestone["stage"],
        "latest_hour_status": "NOT_FINALIZED",
        "latest_snapshot": None,
        "complete_hours": 0,
        "consecutive_complete_hours": 0,
        "complete_days_equivalent": 0.0,
        "consecutive_complete_days_equivalent": 0.0,
        "first_complete_hour": None,
        "last_complete_hour": None,
        "minimum_days": {
            "technical_feature_validation": 90,
            "preliminary_research": 180,
            "formal_regime_assessment": 365,
        },
        "milestones": milestone["milestones"],
        "snapshot_audit": audit,
        "backtest_permitted": False,
        "paper_permitted": False,
        "live_permitted": False,
        "synthetic_data_used": False,
        "orders_generated": 0,
    }
    return {
        **body,
        "readiness_hash": stable_hash(body, length=64),
    }


class ProspectiveOrderflowRecorder:
    """Consume WebSocket events, batch-fsync them and publish gap-aware facts."""

    def __init__(
        self,
        *,
        ledger: HashChainedOrderflowLedger,
        database: Any,
        markets: tuple[str, ...],
        feature_directory: Path,
        readiness_path: Path,
        health_path: Path | None = None,
        positioning_directory: Path | None = None,
        flush_seconds: float = 1.0,
        batch_size: int = 500,
        finalization_grace_minutes: int = 5,
        positioning_timeout_minutes: int = 10,
    ) -> None:
        self.ledger = ledger
        self.database = database
        self.markets = markets
        self.feature_directory = feature_directory
        self.readiness_path = readiness_path
        self.health_path = health_path
        self.positioning_directory = positioning_directory
        self.flush_seconds = flush_seconds
        self.batch_size = batch_size
        self.finalization_grace_minutes = max(
            1,
            int(finalization_grace_minutes),
        )
        self.positioning_timeout_minutes = max(
            self.finalization_grace_minutes + 1,
            int(positioning_timeout_minutes),
        )
        self.started_at = utc_now()
        self._stop = asyncio.Event()
        self._pause_requested = asyncio.Event()
        self._paused = asyncio.Event()
        self._books: dict[
            str,
            dict[str, Any],
        ] = {}
        self._health_counters = {
            "sequence_gaps": 0,
            "dropped_messages": 0,
            "reconnects": 0,
        }
        self._acknowledged_stream_counters = {
            "sequence_gaps": 0,
            "dropped_messages": 0,
            "reconnects": 0,
        }
        self._last_recovery_at: str | None = None
        self._last_ticker_minute: dict[str, datetime] = {}
        self._last_book_state_bucket: dict[str, int] = {}

    def stop(self) -> None:
        self._stop.set()
        self._pause_requested.clear()

    async def pause(self) -> None:
        """Pause queue consumption so provider deltas can buffer during reseed."""

        self._pause_requested.set()
        while not self._paused.is_set() and not self._stop.is_set():
            await asyncio.sleep(0.01)

    def resume(self) -> None:
        self._pause_requested.clear()

    def acknowledge_stream_recovery(
        self,
        health: Mapping[str, Any],
    ) -> None:
        for counter in self._acknowledged_stream_counters:
            self._acknowledged_stream_counters[counter] = int(
                health.get(counter) or 0
            )
        self._last_recovery_at = utc_now().isoformat()

    def _write_health(self, manager: Any) -> dict[str, Any]:
        provider = dict(manager.health("bitvavo"))
        queue_size = manager.queue.qsize()
        queue_capacity = manager.queue.maxsize
        queue_utilization = (
            queue_size / queue_capacity if queue_capacity else 0.0
        )
        storage_limit = self.ledger.maximum_storage_bytes
        storage_utilization = (
            self.ledger.storage_bytes / storage_limit
            if storage_limit
            else None
        )
        reason_codes: list[str] = []
        if provider.get("state") != "CONNECTED":
            reason_codes.append("PROVIDER_NOT_CONNECTED")
        if int(provider.get("sequence_gaps") or 0) > int(
            self._acknowledged_stream_counters["sequence_gaps"]
        ):
            reason_codes.append("SEQUENCE_GAP_DETECTED")
        if int(provider.get("dropped_messages") or 0) > int(
            self._acknowledged_stream_counters["dropped_messages"]
        ):
            reason_codes.append("DROPPED_MESSAGE_DETECTED")
        if int(provider.get("reconnects") or 0) > int(
            self._acknowledged_stream_counters["reconnects"]
        ):
            reason_codes.append("STREAM_RECONNECTED")
        if queue_utilization >= 0.8:
            reason_codes.append("QUEUE_PRESSURE")
        if storage_utilization is not None and storage_utilization >= 1:
            reason_codes.append("STORAGE_LIMIT_REACHED")
        free_disk = shutil.disk_usage(self.ledger.root).free
        report = {
            "schema_version": "orderflow_stream_health_v1",
            "status": "HEALTHY" if not reason_codes else "DEGRADED",
            "reason_codes": reason_codes,
            "observed_at": utc_now().isoformat(),
            "provider": provider,
            "queue_size": queue_size,
            "queue_capacity": queue_capacity,
            "queue_utilization": queue_utilization,
            "record_count": self.ledger.record_count,
            "ledger_root_hash": self.ledger.previous_hash,
            "last_arrival_timestamp": (
                self.ledger.last_arrival_timestamp
            ),
            "last_stream_recovery_at": self._last_recovery_at,
            "storage_bytes": self.ledger.storage_bytes,
            "maximum_storage_bytes": storage_limit,
            "storage_utilization": storage_utilization,
            "free_disk_bytes": free_disk,
            "synthetic_data_used": False,
            "orders_generated": 0,
        }
        if self.health_path is not None:
            atomic_write_json(self.health_path, report)
        return report

    def seed_orderbook(self, snapshot: Any) -> None:
        """Seed a full REST snapshot before buffered WebSocket deltas apply."""

        values = dict(snapshot.values)
        bids: dict[Decimal, Decimal] = {}
        asks: dict[Decimal, Decimal] = {}
        for side, rows in (
            (bids, values.get("bids") or []),
            (asks, values.get("asks") or []),
        ):
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                price = _decimal(row[0])
                quantity = _decimal(row[1])
                if (
                    price is not None
                    and quantity is not None
                    and quantity > 0
                ):
                    side[price] = quantity
        sequence = values.get("sequence")
        self._books[snapshot.canonical_market] = {
            "bids": bids,
            "asks": asks,
            "snapshot_reference": snapshot.raw_hash,
            "last_sequence": (
                int(sequence) if sequence is not None else None
            ),
            "valid": bool(bids and asks and sequence is not None),
        }

    def _persist_database(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )
        table = {
            StreamEventType.TRADE.value: "trades",
            StreamEventType.TICKER.value: "ticker_events",
            StreamEventType.ORDERBOOK_DELTA.value: (
                "orderbook_snapshots"
            ),
            StreamEventType.ORDERBOOK_SNAPSHOT.value: (
                "orderbook_snapshots"
            ),
        }
        for record in records:
            target = table.get(str(record.get("event_type")))
            if not target:
                continue
            grouped[target].append(
                {
                    "external_id": stable_hash(
                        {
                            "provider": record.get("exchange"),
                            "market": record.get("market"),
                            "message_id": record.get("message_id"),
                            "event_type": record.get("event_type"),
                        },
                        length=32,
                    ),
                    **dict(record),
                }
            )
        for target, payloads in grouped.items():
            self.database.upsert_records(target, payloads)

    def _enrich_orderbooks(
        self,
        events: Iterable[NormalizedStreamEvent],
    ) -> list[NormalizedStreamEvent]:
        enriched: list[NormalizedStreamEvent] = []
        for event in events:
            if event.event_type not in {
                StreamEventType.ORDERBOOK_DELTA,
                StreamEventType.ORDERBOOK_SNAPSHOT,
            }:
                enriched.append(event)
                continue
            state = self._books.get(event.canonical_market)
            if state is None:
                payload = {
                    **event.payload,
                    "book_bids": [],
                    "book_asks": [],
                    "snapshot_reference": None,
                    "book_state_status": "SNAPSHOT_MISSING",
                }
                enriched.append(
                    event.model_copy(update={"payload": payload})
                )
                continue
            bids = state["bids"]
            asks = state["asks"]
            assert isinstance(bids, dict)
            assert isinstance(asks, dict)
            payload = dict(event.payload)
            last_sequence = state.get("last_sequence")
            if event.sequence is None or last_sequence is None:
                state["valid"] = False
                sequence_status = "SEQUENCE_UNAVAILABLE"
            elif event.sequence <= int(last_sequence):
                sequence_status = "STALE_BEFORE_SNAPSHOT"
            elif event.sequence != int(last_sequence) + 1:
                state["valid"] = False
                sequence_status = "SEQUENCE_GAP"
            else:
                sequence_status = "SEQUENCE_APPLIED"
            if sequence_status in {
                "SEQUENCE_UNAVAILABLE",
                "SEQUENCE_GAP",
                "STALE_BEFORE_SNAPSHOT",
            }:
                payload.update(
                    {
                        "book_bids": [],
                        "book_asks": [],
                        "snapshot_reference": state.get(
                            "snapshot_reference"
                        ),
                        "book_state_status": sequence_status,
                    }
                )
                enriched.append(
                    event.model_copy(update={"payload": payload})
                )
                continue
            for side, updates in (
                (bids, payload.get("bids") or []),
                (asks, payload.get("asks") or []),
            ):
                for row in updates:
                    if not isinstance(row, (list, tuple)) or len(row) < 2:
                        continue
                    price = _decimal(row[0])
                    quantity = _decimal(row[1])
                    if price is None or quantity is None:
                        continue
                    if quantity == 0:
                        side.pop(price, None)
                    else:
                        side[price] = quantity
            book_bids = [
                [str(price), str(quantity)]
                for price, quantity in sorted(
                    bids.items(),
                    reverse=True,
                )[:100]
            ]
            book_asks = [
                [str(price), str(quantity)]
                for price, quantity in sorted(asks.items())[:100]
            ]
            state["last_sequence"] = event.sequence
            state_bucket = int(event.observed_at.timestamp()) // 5
            include_book_state = (
                self._last_book_state_bucket.get(
                    event.canonical_market
                )
                != state_bucket
            )
            if include_book_state:
                self._last_book_state_bucket[
                    event.canonical_market
                ] = state_bucket
            payload.update(
                {
                    "book_bids": (
                        book_bids if include_book_state else []
                    ),
                    "book_asks": (
                        book_asks if include_book_state else []
                    ),
                    "snapshot_reference": state[
                        "snapshot_reference"
                    ],
                    "book_state_status": "SEQUENCE_APPLIED",
                }
            )
            enriched.append(
                event.model_copy(update={"payload": payload})
            )
        return enriched

    def _select_events(
        self,
        events: Iterable[NormalizedStreamEvent],
    ) -> list[NormalizedStreamEvent]:
        selected: list[NormalizedStreamEvent] = []
        for event in events:
            if event.event_type is not StreamEventType.TICKER:
                selected.append(event)
                continue
            minute = event.observed_at.astimezone(UTC).replace(
                second=0,
                microsecond=0,
            )
            if (
                self._last_ticker_minute.get(
                    event.canonical_market
                )
                == minute
            ):
                continue
            self._last_ticker_minute[event.canonical_market] = minute
            selected.append(event)
        return selected

    def finalize_previous_hour(
        self,
        *,
        observed_at: datetime,
        health: Mapping[str, Any],
    ) -> dict[str, Any]:
        hour_start = _closed_hour_start(observed_at)
        self.feature_directory.mkdir(parents=True, exist_ok=True)
        target = self.feature_directory / (
            hour_start.strftime("%Y%m%dT%H0000Z") + ".json"
        )
        if target.is_file():
            snapshot = dict(read_json(target))
            readiness = _readiness_payload(
                feature_directory=self.feature_directory,
                target=target,
                snapshot=snapshot,
                ledger_root=self.ledger.root,
            )
            atomic_write_json(self.readiness_path, readiness)
            readiness["sealed_segments"] = (
                seal_completed_orderflow_segments(
                    self.ledger.root,
                    current_hour=_utc(observed_at),
                )
            )
            self.ledger.refresh_storage_bytes()
            return {**readiness, "snapshot": snapshot}
        positioning_target = (
            self.positioning_directory
            / (hour_start.strftime("%Y%m%dT%H0000Z") + ".json")
            if self.positioning_directory is not None
            else None
        )
        positioning_deadline = (
            hour_start
            + timedelta(
                hours=1,
                minutes=self.positioning_timeout_minutes,
            )
        )
        positioning_pending = bool(
            positioning_target is not None
            and not positioning_target.is_file()
            and _utc(observed_at) < positioning_deadline
        )
        if positioning_pending:
            return {
                "schema_version": "microstructure_finalization_state_v1",
                "finalization_state": "DEFERRED_CONTEXT_PENDING",
                "status": "COLLECTING_PROSPECTIVE_DATA",
                "reason_code": "POSITIONING_CONTEXT_PENDING",
                "target_hour": hour_start.isoformat(),
                "positioning_snapshot": str(positioning_target),
                "retry_until": positioning_deadline.isoformat(),
                "snapshot_written": False,
                "backtest_permitted": False,
                "paper_permitted": False,
                "live_permitted": False,
                "orders_generated": 0,
            }
        positioning_timed_out = bool(
            positioning_target is not None
            and not positioning_target.is_file()
        )
        records = _read_segment(self.ledger.root, hour_start)
        provider_health = dict(health.get("bitvavo") or health)
        period_health = dict(provider_health)
        for counter in self._health_counters:
            current = int(provider_health.get(counter) or 0)
            period_health[counter] = max(
                0,
                current - self._health_counters[counter],
            )
            self._health_counters[counter] = current
        markets = [
            summarize_orderflow_hour(
                records,
                market=market,
                hour_start=hour_start,
                recorder_started_at=self.started_at,
                health=period_health,
            )
            for market in self.markets
        ]
        _apply_cvd_history(
            self.feature_directory,
            target_path=target,
            markets=markets,
        )
        positioning = _positioning_context(
            self.positioning_directory,
            hour_start=hour_start,
        )
        for row in markets:
            base = str(row["market"]).split("-", 1)[0]
            derivative = positioning.get(base)
            row["derivatives_positioning"] = derivative
            perpetual_base = _decimal(
                (derivative or {}).get(
                    "perpetual_base_volume_24h"
                )
            )
            spot_base = _decimal(row.get("spot_volume_24h"))
            row["perpetual_spot_base_volume_ratio"] = (
                float(perpetual_base / spot_base)
                if perpetual_base is not None
                and spot_base is not None
                and spot_base > 0
                else None
            )
            row["required_field_coverage"] = {
                "spot_cvd_input_available": (
                    row["trade_count"] > 0
                ),
                "orderbook_available": (
                    row["orderbook_imbalance"] is not None
                ),
                "funding_available": (
                    (derivative or {}).get("funding_rate")
                    is not None
                ),
                "open_interest_available": (
                    (derivative or {}).get("open_interest")
                    is not None
                ),
                "basis_available": (
                    (derivative or {}).get("perpetual_basis")
                    is not None
                ),
                "liquidations_observed": (
                    (derivative or {}).get(
                        "liquidation_status"
                    )
                    != "UNAVAILABLE_PUBLIC_ENDPOINT"
                ),
            }
            required_context = (
                "funding_available",
                "open_interest_available",
                "basis_available",
            )
            if any(
                not row["required_field_coverage"][field]
                for field in required_context
            ):
                row["status"] = "DATA_GAP"
                row["reason_codes"].append(
                    (
                        "POSITIONING_CONTEXT_TIMEOUT"
                        if positioning_timed_out
                        else "DERIVATIVES_CONTEXT_INCOMPLETE"
                    )
                )
            if row["perpetual_spot_base_volume_ratio"] is None:
                row["status"] = "DATA_GAP"
                row["reason_codes"].append(
                    "PERPETUAL_SPOT_VOLUME_RATIO_UNAVAILABLE"
                )
        body = {
            "schema_version": MICROSTRUCTURE_SCHEMA,
            "hour_start": hour_start.isoformat(),
            "hour_end": (
                hour_start + timedelta(hours=1)
            ).isoformat(),
            "finalized_at": _utc(observed_at).isoformat(),
            "status": (
                "COMPLETE"
                if all(row["status"] == "COMPLETE" for row in markets)
                else "DATA_GAP"
            ),
            "markets": markets,
            "stream_health": period_health,
            "ledger_root_hash": self.ledger.previous_hash,
            "positioning_context_status": (
                "TIMED_OUT"
                if positioning_timed_out
                else "AVAILABLE"
                if positioning_target is not None
                else "NOT_CONFIGURED"
            ),
            "synthetic_data_used": False,
            "orders_generated": 0,
        }
        snapshot = {
            **body,
            "snapshot_hash": stable_hash(body, length=64),
        }
        atomic_write_json(target, snapshot)
        readiness = _readiness_payload(
            feature_directory=self.feature_directory,
            target=target,
            snapshot=snapshot,
            ledger_root=self.ledger.root,
        )
        atomic_write_json(self.readiness_path, readiness)
        readiness["sealed_segments"] = (
            seal_completed_orderflow_segments(
                self.ledger.root,
                current_hour=_utc(observed_at),
            )
        )
        self.ledger.refresh_storage_bytes()
        return {
            **readiness,
            "finalization_state": "FINALIZED",
            "snapshot": snapshot,
        }

    async def run(self, manager: Any) -> None:
        buffer: list[NormalizedStreamEvent] = []
        last_flush = asyncio.get_running_loop().time()
        last_finalized: datetime | None = None
        last_finalization_attempt: datetime | None = None
        try:
            while (
                not self._stop.is_set()
                or not manager.queue.empty()
            ):
                if self._pause_requested.is_set():
                    self._paused.set()
                    while (
                        self._pause_requested.is_set()
                        and not self._stop.is_set()
                    ):
                        await asyncio.sleep(0.01)
                    self._paused.clear()
                    continue
                timeout = max(
                    0.05,
                    self.flush_seconds
                    - (
                        asyncio.get_running_loop().time()
                        - last_flush
                    ),
                )
                try:
                    event = await manager.next_event(timeout=timeout)
                    if (
                        event.event_type
                        is not StreamEventType.CONNECTION_STATUS
                    ):
                        buffer.append(event)
                except TimeoutError:
                    pass
                now_monotonic = asyncio.get_running_loop().time()
                if (
                    buffer
                    and (
                        len(buffer) >= self.batch_size
                        or now_monotonic - last_flush
                        >= self.flush_seconds
                    )
                ):
                    records = self.ledger.append(
                        self._enrich_orderbooks(
                            self._select_events(buffer)
                        )
                    )
                    self._persist_database(records)
                    buffer.clear()
                    last_flush = now_monotonic
                    self._write_health(manager)
                now = utc_now()
                current_closed = _closed_hour_start(now)
                if (
                    (
                        last_finalized is None
                        or current_closed > last_finalized
                    )
                    and now.minute >= self.finalization_grace_minutes
                    and (
                        last_finalization_attempt is None
                        or (
                            now - last_finalization_attempt
                        ).total_seconds()
                        >= 5
                    )
                ):
                    result = self.finalize_previous_hour(
                        observed_at=now,
                        health=manager.health(),
                    )
                    last_finalization_attempt = now
                    if (
                        result.get("finalization_state")
                        != "DEFERRED_CONTEXT_PENDING"
                    ):
                        last_finalized = current_closed
            self._write_health(manager)
        finally:
            if buffer:
                records = self.ledger.append(
                    self._enrich_orderbooks(
                        self._select_events(buffer)
                    )
                )
                self._persist_database(records)
            self._write_health(manager)


__all__ = [
    "HashChainedOrderflowLedger",
    "ProspectiveOrderflowRecorder",
    "audit_microstructure_snapshots",
    "current_microstructure_readiness",
    "normalize_stream_event",
    "prospective_milestone_status",
    "seal_completed_orderflow_segments",
    "summarize_orderflow_hour",
    "verify_orderflow_ledger",
]
