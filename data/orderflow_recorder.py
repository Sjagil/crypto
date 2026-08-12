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
from bisect import bisect_left
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.contracts import NormalizedStreamEvent, StreamEventType
from data.realtime_candle_builder import RealtimeCandleBuilder
from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
    stable_json,
    utc_now,
)

ORDERFLOW_SCHEMA = "prospective_orderflow_event_v1"
ORDERFLOW_CHECKPOINT_SCHEMA = "prospective_orderflow_checkpoint_v1"
MICROSTRUCTURE_SCHEMA = "microstructure_hourly_snapshot_v4"
MICROSTRUCTURE_15M_SCHEMA = "microstructure_15m_snapshot_v1"
SUPPORTED_MICROSTRUCTURE_SCHEMAS = {
    "microstructure_hourly_snapshot_v1",
    "microstructure_hourly_snapshot_v2",
    "microstructure_hourly_snapshot_v3",
    MICROSTRUCTURE_SCHEMA,
}
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


def _closed_quarter_start(observed_at: datetime) -> datetime:
    """Return the start of the most recently closed UTC 15-minute bucket."""

    observed = _utc(observed_at)
    current_start = observed.replace(
        minute=(observed.minute // 15) * 15,
        second=0,
        microsecond=0,
    )
    return current_start - timedelta(minutes=15)


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
        try:
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
        except FileNotFoundError:
            # The live recorder may atomically rotate/compress a segment after
            # discovery. Verification remains fail-closed and can pass on the
            # next stable snapshot instead of crashing the entire supervisor.
            failures.append(f"SEGMENT_ROTATED_DURING_VERIFY:{path}")
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


def _read_last_nonempty_json_line(path: Path) -> dict[str, Any]:
    """Read the last durable raw JSONL record without scanning the segment."""

    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        buffer = bytearray()
        while position > 0:
            position -= 1
            stream.seek(position)
            value = stream.read(1)
            if value == b"\n" and buffer:
                break
            if value not in {b"\n", b"\r"}:
                buffer.extend(value)
        raw = bytes(reversed(buffer)).decode("utf-8")
    if not raw.strip():
        raise RuntimeError("ORDERFLOW_CHECKPOINT_EMPTY_LAST_SEGMENT")
    try:
        selected = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ORDERFLOW_CHECKPOINT_INVALID_LAST_RECORD") from exc
    if not isinstance(selected, dict):
        raise RuntimeError("ORDERFLOW_CHECKPOINT_INVALID_LAST_RECORD")
    return dict(selected)


def verify_orderflow_checkpoint(
    root: Path | str,
    checkpoint_path: Path | str,
) -> dict[str, Any]:
    """Verify the durable chain head for bounded restart recovery.

    A successful result proves that the checkpoint references the exact last
    durable record, and that this record is internally hashed correctly.  The
    historical chain was verified before that checkpoint was published; a
    full forensic rescan remains available through :func:`verify_orderflow_ledger`.
    """

    directory = Path(root).resolve()
    checkpoint_file = Path(checkpoint_path)
    failures: list[str] = []
    try:
        checkpoint = dict(read_json(checkpoint_file))
    except (OSError, TypeError, ValueError):
        checkpoint = {}
        failures.append("CHECKPOINT_UNREADABLE")
    if checkpoint.get("schema_version") != ORDERFLOW_CHECKPOINT_SCHEMA:
        failures.append("CHECKPOINT_SCHEMA_MISMATCH")
    root_hash = str(checkpoint.get("root_hash") or "")
    count = int(checkpoint.get("record_count") or 0)
    last_segment_raw = str(checkpoint.get("last_segment") or "")
    last_segment: Path | None = None
    if len(root_hash) != 64 or any(value not in "0123456789abcdef" for value in root_hash):
        failures.append("CHECKPOINT_ROOT_HASH_INVALID")
    if count <= 0:
        failures.append("CHECKPOINT_RECORD_COUNT_INVALID")
    if not last_segment_raw:
        failures.append("CHECKPOINT_LAST_SEGMENT_MISSING")
    else:
        try:
            last_segment = Path(last_segment_raw).resolve()
            last_segment.relative_to(directory)
        except (OSError, ValueError):
            failures.append("CHECKPOINT_LAST_SEGMENT_OUTSIDE_ROOT")
            last_segment = None
    if last_segment is not None:
        if not last_segment.is_file() or not last_segment.name.endswith(".jsonl"):
            failures.append("CHECKPOINT_LAST_SEGMENT_UNAVAILABLE")
        else:
            expected_size = checkpoint.get("last_segment_size_bytes")
            if expected_size is not None and int(expected_size) != last_segment.stat().st_size:
                failures.append("CHECKPOINT_LAST_SEGMENT_SIZE_MISMATCH")
            try:
                last_record = _read_last_nonempty_json_line(last_segment)
                record_hash = str(last_record.pop("record_hash", ""))
                if record_hash != stable_hash(last_record, length=64):
                    failures.append("CHECKPOINT_LAST_RECORD_HASH_MISMATCH")
                if record_hash != root_hash:
                    failures.append("CHECKPOINT_CHAIN_HEAD_MISMATCH")
                if (
                    checkpoint.get("last_arrival_timestamp")
                    != last_record.get("arrival_timestamp")
                ):
                    failures.append("CHECKPOINT_LAST_ARRIVAL_MISMATCH")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(str(exc))
    return {
        "schema_version": "prospective_orderflow_checkpoint_audit_v1",
        "status": "PASSED" if not failures else "FAILED",
        "record_count": count,
        "root_hash": root_hash,
        "last_arrival_timestamp": checkpoint.get("last_arrival_timestamp"),
        "last_segment": str(last_segment) if last_segment is not None else None,
        "integrity_failures": failures,
        "recovery_mode": "CHECKPOINT_CHAIN_HEAD_VERIFIED",
        "full_historical_rescan_deferred": True,
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
        checkpoint_first_recovery: bool = False,
    ) -> None:
        self.root = root
        self.checkpoint_path = checkpoint_path
        self.maximum_storage_bytes = maximum_storage_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        audit = (
            verify_orderflow_checkpoint(self.root, checkpoint_path)
            if checkpoint_first_recovery and checkpoint_path.is_file()
            else verify_orderflow_ledger(self.root)
        )
        if audit["status"] != "PASSED":
            raise RuntimeError("ORDERFLOW_LEDGER_INTEGRITY_FAILED")
        self.recovery_mode = str(
            audit.get("recovery_mode") or "FULL_LEDGER_VERIFIED"
        )
        self.integrity_status = str(audit["status"])
        self.full_historical_rescan_deferred = bool(
            audit.get("full_historical_rescan_deferred", False)
        )
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
                    "last_segment_size_bytes": self._segment(
                        _utc(records[-1]["arrival_timestamp"])
                    ).stat().st_size,
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
    )
    asks = list(
        record.get("book_asks")
        or record.get("ask_updates")
        or []
    )
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
    best_bid, best_bid_quantity = parsed_bids[0]
    best_ask, best_ask_quantity = parsed_asks[0]
    top_total = best_bid_quantity + best_ask_quantity
    mid = (best_bid + best_ask) / Decimal(2)
    selected_bid_total = sum(
        (row[1] for row in parsed_bids[:depth]), Decimal(0)
    )
    selected_ask_total = sum(
        (row[1] for row in parsed_asks[:depth]), Decimal(0)
    )
    selected_total = selected_bid_total + selected_ask_total
    result = {
        "orderbook_imbalance": (
            float((selected_bid_total - selected_ask_total) / selected_total)
            if selected_total > 0
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
    for selected_depth in (5, 10, 25):
        depth_bids = parsed_bids[:selected_depth]
        depth_asks = parsed_asks[:selected_depth]
        bid_size = sum((row[1] for row in depth_bids), Decimal(0))
        ask_size = sum((row[1] for row in depth_asks), Decimal(0))
        depth_total = bid_size + ask_size
        result[f"orderbook_imbalance_top_{selected_depth}"] = (
            float((bid_size - ask_size) / depth_total)
            if depth_total > 0
            else None
        )
        result[f"bid_liquidity_top_{selected_depth}_quote"] = float(
            sum((price * quantity for price, quantity in depth_bids), Decimal(0))
        )
        result[f"ask_liquidity_top_{selected_depth}_quote"] = float(
            sum((price * quantity for price, quantity in depth_asks), Decimal(0))
        )
    for band_bps in (5, 10, 25, 50):
        fraction = Decimal(band_bps) / Decimal(10_000)
        lower, upper = mid * (Decimal(1) - fraction), mid * (Decimal(1) + fraction)
        band_bids = [row for row in parsed_bids if row[0] >= lower]
        band_asks = [row for row in parsed_asks if row[0] <= upper]
        bid_quote = sum(
            (price * quantity for price, quantity in band_bids), Decimal(0)
        )
        ask_quote = sum(
            (price * quantity for price, quantity in band_asks), Decimal(0)
        )
        band_total = bid_quote + ask_quote
        result[f"bid_liquidity_within_{band_bps}bps_quote"] = float(bid_quote)
        result[f"ask_liquidity_within_{band_bps}bps_quote"] = float(ask_quote)
        result[f"orderbook_imbalance_within_{band_bps}bps"] = (
            float((bid_quote - ask_quote) / band_total)
            if band_total > 0
            else None
        )
    return result


def summarize_orderflow_hour(
    records: Iterable[Mapping[str, Any]],
    *,
    market: str,
    hour_start: datetime,
    recorder_started_at: datetime,
    health: Mapping[str, Any],
    interval: timedelta = timedelta(hours=1),
) -> dict[str, Any]:
    """Calculate causal facts and explicitly grade interval completeness.

    The public name is retained for backwards compatibility. Hourly callers
    use the default; the live recorder also applies the exact same forensic
    calculation to separately sealed 15-minute entry buckets.
    """

    start = _utc(hour_start)
    end = start + interval
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
        reasons.append(
            "RECORDER_STARTED_MID_HOUR"
            if interval == timedelta(hours=1)
            else "RECORDER_STARTED_MID_INTERVAL"
        )
    if not selected:
        reasons.append("NO_STREAM_EVENTS")
    coverage_grace = timedelta(
        minutes=5 if interval >= timedelta(hours=1) else 2
    )
    if arrivals and arrivals[0] > start + coverage_grace:
        reasons.append("ARRIVAL_START_LATE")
    if arrivals and arrivals[-1] < end - coverage_grace:
        reasons.append("ARRIVAL_END_EARLY")
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
    latest_book_metrics = _book_metrics(
        book_samples[-1] if book_samples else None,
        depth=10,
    )
    first_book_metrics = _book_metrics(
        book_samples[0] if book_samples else None,
        depth=10,
    )
    initial_bid_quote = _decimal(
        first_book_metrics.get("bid_liquidity_top_25_quote")
    ) or Decimal(0)
    initial_ask_quote = _decimal(
        first_book_metrics.get("ask_liquidity_top_25_quote")
    ) or Decimal(0)
    final_bid_quote = _decimal(
        latest_book_metrics.get("bid_liquidity_top_25_quote")
    ) or Decimal(0)
    final_ask_quote = _decimal(
        latest_book_metrics.get("ask_liquidity_top_25_quote")
    ) or Decimal(0)
    bid_liquidity_change = final_bid_quote - initial_bid_quote
    ask_liquidity_change = final_ask_quote - initial_ask_quote
    ofi_quote = bid_liquidity_change - ask_liquidity_change
    initial_depth_quote = initial_bid_quote + initial_ask_quote
    ofi_normalized = (
        float(ofi_quote / initial_depth_quote)
        if initial_depth_quote > 0
        else None
    )
    ordered_trades = sorted(
        trades,
        key=lambda row: _utc(row["arrival_timestamp"]),
    )
    first_trade_price = (
        _decimal(ordered_trades[0].get("price"))
        if ordered_trades
        else None
    )
    last_trade_price = (
        _decimal(ordered_trades[-1].get("price"))
        if ordered_trades
        else None
    )
    price_change_fraction = (
        float(last_trade_price / first_trade_price - Decimal(1))
        if first_trade_price is not None
        and first_trade_price > 0
        and last_trade_price is not None
        else None
    )
    price_stability = (
        max(0.0, 1.0 - abs(price_change_fraction) / 0.0025)
        if price_change_fraction is not None
        else 0.0
    )
    buy_share = float(buy_base / total_base) if total_base > 0 else 0.0
    sell_share = float(sell_base / total_base) if total_base > 0 else 0.0
    positive_refill = max(0.0, min(1.0, (ofi_normalized or 0.0) + 0.5))
    negative_refill = max(0.0, min(1.0, -(ofi_normalized or 0.0) + 0.5))
    return {
        "market": market,
        "hour_start": start.isoformat(),
        "hour_end": end.isoformat(),
        "interval_minutes": int(interval.total_seconds() // 60),
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
        "aggressive_buy_share": buy_share,
        "aggressive_sell_share": sell_share,
        "price_change_fraction": price_change_fraction,
        "bid_liquidity_change_top_25_quote": float(bid_liquidity_change),
        "ask_liquidity_change_top_25_quote": float(ask_liquidity_change),
        "order_flow_imbalance_quote": float(ofi_quote),
        "order_flow_imbalance_normalized": ofi_normalized,
        "bullish_absorption_score": sell_share * price_stability * positive_refill,
        "bearish_absorption_score": buy_share * price_stability * negative_refill,
        "absorption_method": (
            "AGGRESSOR_SHARE_X_LOW_PRICE_IMPACT_X_TOP25_BOOK_REFILL"
        ),
        "spot_base_volume": float(total_base),
        "spot_quote_volume": float(total_quote),
        "spot_volume_24h": (
            volume_tickers[-1].get("spot_volume_24h")
            if volume_tickers
            else None
        ),
        "spot_quote_volume_24h": (
            volume_tickers[-1].get("spot_quote_volume_24h")
            if volume_tickers
            else None
        ),
        "spot_last_price_eur": (
            volume_tickers[-1].get("last_price")
            if volume_tickers
            else None
        ),
        "orderbook_sample_count": len(book_samples),
        **latest_book_metrics,
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
    exact_prior_records: dict[
        int,
        dict[str, dict[str, Any]],
    ] = {}
    for horizon_hours in (1, 4):
        prior_path = directory / (
            (hour_start - timedelta(hours=horizon_hours)).strftime(
                "%Y%m%dT%H0000Z"
            )
            + ".json"
        )
        payload = (
            dict(read_json(prior_path))
            if prior_path.is_file()
            else {}
        )
        exact_prior_records[horizon_hours] = {
            str(item["canonical_market"]): dict(
                item.get("values") or {}
            )
            for item in payload.get("derivatives_context") or []
        }
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
        current_oi = values.get("open_interest")
        oi_changes: dict[int, float | None] = {}
        oi_references: dict[int, Any] = {}
        for horizon_hours in (1, 4):
            reference = exact_prior_records[horizon_hours].get(
                market,
                {},
            ).get("open_interest")
            oi_references[horizon_hours] = reference
            oi_changes[horizon_hours] = (
                float(current_oi) / float(reference) - 1.0
                if reference is not None
                and current_oi is not None
                and float(reference) != 0
                else None
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
            "open_interest_change": oi_changes[4],
            "open_interest_change_1h": oi_changes[1],
            "open_interest_change_4h": oi_changes[4],
            "open_interest_change_horizon_hours": 4,
            "open_interest_reference_1h": oi_references[1],
            "open_interest_reference_4h": oi_references[4],
            "open_interest_reference_status": (
                "EXACT_T_MINUS_4H_AVAILABLE"
                if oi_references[4] is not None
                else "EXACT_T_MINUS_4H_MISSING"
            ),
            "perpetual_basis": values.get("basis"),
            "perpetual_premium": values.get(
                "perpetual_premium"
            ),
            "perpetual_index_price_usdt": values.get(
                "index_price"
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


def _snapshot_audit_fingerprint(
    path: Path,
    *,
    ledger_root: Path | None,
) -> dict[str, Any]:
    snapshot_stat = path.stat()
    fingerprint: dict[str, Any] = {
        "snapshot_sha256": _sha256_file(path),
        "snapshot_size": snapshot_stat.st_size,
        "snapshot_mtime_ns": snapshot_stat.st_mtime_ns,
    }
    if ledger_root is None:
        return fingerprint
    try:
        hour_start = _utc(dict(read_json(path))["hour_start"])
    except (KeyError, OSError, TypeError, ValueError):
        fingerprint["ledger_status"] = "HOUR_UNAVAILABLE"
        return fingerprint
    raw_path = (
        ledger_root
        / hour_start.strftime("%Y")
        / hour_start.strftime("%m")
        / hour_start.strftime("%d")
        / f"{hour_start:%H}.jsonl"
    )
    selected = next(
        (
            candidate
            for candidate in (
                raw_path,
                raw_path.with_suffix(".jsonl.xz"),
                raw_path.with_suffix(".jsonl.gz"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if selected is None:
        fingerprint["ledger_status"] = "SEGMENT_MISSING"
        return fingerprint
    segment_stat = selected.stat()
    fingerprint.update(
        {
            "ledger_status": "SEGMENT_PRESENT",
            "segment_path": str(selected),
            "segment_size": segment_stat.st_size,
            "segment_mtime_ns": segment_stat.st_mtime_ns,
        }
    )
    manifest_path = raw_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        fingerprint["manifest_sha256"] = _sha256_file(
            manifest_path
        )
    return fingerprint


def _load_snapshot_audit_cache(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {}
    expected = payload.get("cache_hash")
    body = {
        key: value
        for key, value in payload.items()
        if key != "cache_hash"
    }
    if (
        payload.get("schema_version")
        != "microstructure_snapshot_audit_cache_v1"
        or expected != stable_hash(body, length=64)
    ):
        return {}
    entries = payload.get("entries")
    return dict(entries) if isinstance(entries, dict) else {}


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
    if (
        payload.get("schema_version")
        not in SUPPORTED_MICROSTRUCTURE_SCHEMAS
    ):
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
        required_fields = [
            "spot_cvd_input_available",
            "orderbook_available",
            "funding_available",
            "open_interest_available",
            "basis_available",
        ]
        if payload.get("schema_version") == MICROSTRUCTURE_SCHEMA:
            required_fields.append(
                "quote_currency_conversion_available"
            )
        for field in required_fields:
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
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return the fail-closed readiness evidence from all sealed hours."""

    targets = sorted(feature_directory.glob("*.json"))
    cache_path = (
        feature_directory
        / ".audit"
        / "snapshot_audit_cache_v1.json"
    )
    cached = (
        _load_snapshot_audit_cache(cache_path)
        if use_cache
        else {}
    )
    entries: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    for path in targets:
        key = str(path.resolve())
        fingerprint = _snapshot_audit_fingerprint(
            path,
            ledger_root=ledger_root,
        )
        cached_entry = dict(cached.get(key) or {})
        cached_audit = cached_entry.get("audit")
        if (
            cached_entry.get("fingerprint") == fingerprint
            and isinstance(cached_audit, dict)
        ):
            audit = dict(cached_audit)
        else:
            audit = _audit_hourly_snapshot(
                path,
                ledger_root=ledger_root,
            )
        entries[key] = {
            "fingerprint": fingerprint,
            "audit": audit,
        }
        audits.append(audit)
    if use_cache:
        cache_body = {
            "schema_version": (
                "microstructure_snapshot_audit_cache_v1"
            ),
            "entries": entries,
            "orders_generated": 0,
        }
        atomic_write_json(
            cache_path,
            {
                **cache_body,
                "cache_hash": stable_hash(
                    cache_body,
                    length=64,
                ),
            },
        )
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
        "snapshot_audits": audits,
        "audit_cache_entry_count": len(entries),
        "audit_cache_status": (
            "INCREMENTAL_IMMUTABLE_FINGERPRINTS"
            if use_cache
            else "FULL_REAUDIT"
        ),
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


def microstructure_storage_runway(
    ledger_root: Path,
    *,
    maximum_storage_bytes: int,
    minimum_free_disk_bytes: int = 0,
    free_disk_bytes: int | None = None,
) -> dict[str, Any]:
    """Conservatively project lossless prospective-data storage runway."""

    sealed = sorted(
        [
            *ledger_root.rglob("*.jsonl.xz"),
            *ledger_root.rglob("*.jsonl.gz"),
        ],
        key=lambda path: path.stat().st_mtime,
    )
    sample = sealed[-24:]
    sample_sizes = [path.stat().st_size for path in sample]
    conservative_hour_bytes = max(sample_sizes, default=0)
    current_storage_bytes = sum(
        path.stat().st_size
        for path in ledger_root.rglob("*")
        if path.is_file()
    )
    observed_free_disk_bytes = (
        int(free_disk_bytes)
        if free_disk_bytes is not None
        else int(shutil.disk_usage(ledger_root).free)
    )
    cap_remaining_bytes = max(
        0,
        int(maximum_storage_bytes) - current_storage_bytes,
    )
    disk_remaining_bytes = max(
        0,
        observed_free_disk_bytes - int(minimum_free_disk_bytes),
    )
    usable_remaining_bytes = min(
        cap_remaining_bytes,
        disk_remaining_bytes,
    )
    projected: dict[str, dict[str, Any]] = {}
    for days in (90, 180, 365):
        required = conservative_hour_bytes * 24 * days
        projected[str(days)] = {
            "days": days,
            "projected_additional_bytes": required,
            "fits_configured_cap_and_free_disk": (
                conservative_hour_bytes > 0
                and required <= usable_remaining_bytes
            ),
        }
    if conservative_hour_bytes <= 0:
        status = "INSUFFICIENT_SEALED_SEGMENT_SAMPLE"
        capacity_hours = 0
    else:
        capacity_hours = (
            usable_remaining_bytes // conservative_hour_bytes
        )
        if not projected["365"][
            "fits_configured_cap_and_free_disk"
        ]:
            status = "INSUFFICIENT_FOR_365_DAY_WINDOW"
        elif len(sample) < 24:
            status = "PROVISIONALLY_SUFFICIENT_SAMPLE_LIMITED"
        else:
            status = "SUFFICIENT_FOR_365_DAY_WINDOW"
    return {
        "schema_version": "microstructure_storage_runway_v1",
        "status": status,
        "sealed_segment_sample_count": len(sample),
        "required_sample_count": 24,
        "conservative_hour_bytes": conservative_hour_bytes,
        "current_storage_bytes": current_storage_bytes,
        "maximum_storage_bytes": int(maximum_storage_bytes),
        "free_disk_bytes": observed_free_disk_bytes,
        "minimum_free_disk_bytes": int(minimum_free_disk_bytes),
        "usable_remaining_bytes": usable_remaining_bytes,
        "estimated_remaining_hours": int(capacity_hours),
        "estimated_remaining_days": float(capacity_hours / 24),
        "milestones": projected,
        "orders_generated": 0,
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
        fifteen_minute_feature_directory: Path | None = None,
        health_path: Path | None = None,
        positioning_directory: Path | None = None,
        realtime_candle_path: Path | None = None,
        flush_seconds: float = 1.0,
        batch_size: int = 500,
        finalization_grace_minutes: int = 5,
        positioning_timeout_minutes: int = 10,
    ) -> None:
        self.ledger = ledger
        self.database = database
        self.markets = markets
        self.feature_directory = feature_directory
        self.fifteen_minute_feature_directory = (
            fifteen_minute_feature_directory
            or feature_directory.parent / "microstructure_15m"
        )
        self.readiness_path = readiness_path
        self.health_path = health_path
        self.positioning_directory = positioning_directory
        self.realtime_candle_builder = RealtimeCandleBuilder(
            output_path=(
                realtime_candle_path
                or feature_directory.parent / "realtime_candles.json"
            )
        )
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
        self._quarter_health_counters = {
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
        # The durable ledger remains the forensic source of truth.  These
        # bounded, in-memory windows are a low-latency view of the same
        # prospective events for event-driven decisions; they are never used
        # to rewrite or backfill historical facts.
        self._realtime_trades: dict[
            str,
            deque[tuple[datetime, float, float, int]],
        ] = defaultdict(lambda: deque(maxlen=50_000))
        self._realtime_tickers: dict[
            str,
            deque[tuple[datetime, float, float | None]],
        ] = defaultdict(lambda: deque(maxlen=7_500))
        self._realtime_books: dict[
            str,
            deque[dict[str, Any]],
        ] = defaultdict(lambda: deque(maxlen=1_500))

    @staticmethod
    def _trim_realtime_window(
        rows: deque[Any],
        *,
        cutoff: datetime,
    ) -> None:
        while rows and rows[0][0] < cutoff:
            rows.popleft()

    def _record_realtime_event(
        self,
        event: NormalizedStreamEvent,
    ) -> None:
        market = event.canonical_market
        observed = event.observed_at.astimezone(UTC)
        cutoff = observed - timedelta(minutes=70)
        if event.event_type is StreamEventType.TRADE:
            price = _decimal(event.payload.get("price"))
            base_quantity = _decimal(
                event.payload.get(
                    "base_quantity",
                    event.payload.get("quantity"),
                )
            )
            quote_quantity = _decimal(
                event.payload.get("quote_quantity")
            )
            if (
                quote_quantity is None
                and price is not None
                and base_quantity is not None
            ):
                quote_quantity = price * base_quantity
            if price is None or quote_quantity is None or price <= 0:
                return
            side = str(
                event.payload.get("aggressor_side")
                or event.payload.get("side")
                or ""
            ).casefold()
            direction = 1 if side in {"buy", "bid"} else -1
            self.realtime_candle_builder.ingest_trade(
                market=market,
                timestamp=event.timestamp,
                observed_at=observed,
                price=float(price),
                base_quantity=float(base_quantity or 0),
                quote_quantity=float(quote_quantity),
                aggressor_side=side,
            )
            rows = self._realtime_trades[market]
            rows.append(
                (
                    observed,
                    float(price),
                    float(quote_quantity),
                    direction,
                )
            )
            self._trim_realtime_window(rows, cutoff=cutoff)
        elif event.event_type is StreamEventType.TICKER:
            price = _decimal(
                event.payload.get("last_price")
                or event.payload.get("price")
            )
            if price is None or price <= 0:
                return
            quote_volume = _decimal(event.payload.get("quote_volume_24h"))
            rows = self._realtime_tickers[market]
            rows.append(
                (
                    observed,
                    float(price),
                    float(quote_volume) if quote_volume is not None else None,
                )
            )
            self._trim_realtime_window(rows, cutoff=cutoff)

    def _record_realtime_book(
        self,
        *,
        market: str,
        observed_at: datetime,
        bids: Mapping[Decimal, Decimal],
        asks: Mapping[Decimal, Decimal],
        valid: bool,
    ) -> None:
        if not valid or not bids or not asks:
            return
        sorted_bids = sorted(bids.items(), reverse=True)[:20]
        sorted_asks = sorted(asks.items())[:20]
        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]
        best_bid_quantity = sorted_bids[0][1]
        best_ask_quantity = sorted_asks[0][1]
        midpoint = (best_bid + best_ask) / Decimal("2")
        if midpoint <= 0:
            return
        bid_depths = [
            float(price * quantity)
            for price, quantity in sorted_bids
        ]
        ask_depths = [
            float(price * quantity)
            for price, quantity in sorted_asks
        ]
        bid_top_5 = sum(bid_depths[:5])
        ask_top_5 = sum(ask_depths[:5])
        bid_top_10 = sum(bid_depths[:10])
        ask_top_10 = sum(ask_depths[:10])
        denominator_5 = bid_top_5 + ask_top_5
        denominator_10 = bid_top_10 + ask_top_10
        top_quantity = best_bid_quantity + best_ask_quantity
        microprice = (
            (
                best_ask * best_bid_quantity
                + best_bid * best_ask_quantity
            )
            / top_quantity
            if top_quantity > 0
            else midpoint
        )
        midpoint_float = float(midpoint)

        def depth_within_bps(
            levels: list[tuple[Decimal, Decimal]],
            *,
            side: str,
            distance_bps: float,
        ) -> float:
            total = 0.0
            for level_price, level_quantity in levels:
                distance = (
                    (midpoint_float - float(level_price)) / midpoint_float
                    if side == "bid"
                    else (float(level_price) - midpoint_float) / midpoint_float
                ) * 10_000.0
                if distance <= distance_bps + 1e-12:
                    total += float(level_price * level_quantity)
            return total

        distance_depth: dict[int, tuple[float, float]] = {}
        for distance_bps in (5, 10, 25):
            distance_depth[distance_bps] = (
                depth_within_bps(
                    sorted_bids,
                    side="bid",
                    distance_bps=float(distance_bps),
                ),
                depth_within_bps(
                    sorted_asks,
                    side="ask",
                    distance_bps=float(distance_bps),
                ),
            )

        weighted_bid = sum(
            float(price * quantity)
            / (
                1.0
                + max(
                    0.0,
                    (midpoint_float - float(price))
                    / midpoint_float
                    * 10_000.0,
                )
            )
            for price, quantity in sorted_bids[:10]
        )
        weighted_ask = sum(
            float(price * quantity)
            / (
                1.0
                + max(
                    0.0,
                    (float(price) - midpoint_float)
                    / midpoint_float
                    * 10_000.0,
                )
            )
            for price, quantity in sorted_asks[:10]
        )
        weighted_denominator = weighted_bid + weighted_ask
        current_spread_bps = float(
            (best_ask - best_bid) / midpoint * Decimal("10000")
        )
        prior_spreads = [
            float(row["spread_bps"])
            for row in list(self._realtime_books[market])[-299:]
            if row.get("spread_bps") is not None
        ]
        spread_sample = [*prior_spreads, current_spread_bps]
        spread_median = statistics.median(spread_sample)
        spread_mad = statistics.median(
            abs(value - spread_median) for value in spread_sample
        )
        spread_p75 = (
            statistics.quantiles(
                spread_sample,
                n=4,
                method="inclusive",
            )[2]
            if len(spread_sample) >= 2
            else current_spread_bps
        )
        dynamic_spread_cap = min(35.0, spread_p75 * 1.25)
        record = {
            "observed_at": observed_at.astimezone(UTC),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "midpoint": float(midpoint),
            "microprice": float(microprice),
            "microprice_edge_bps": float(
                (microprice - midpoint) / midpoint * Decimal("10000")
            ),
            "spread_bps": current_spread_bps,
            "spread_rolling_median_bps": spread_median,
            "spread_rolling_mad_bps": spread_mad,
            "spread_rolling_p75_bps": spread_p75,
            "spread_robust_zscore": (
                (current_spread_bps - spread_median) / spread_mad
                if spread_mad > 1e-12
                else 0.0
            ),
            "dynamic_spread_cap_bps": dynamic_spread_cap,
            "spread_within_dynamic_cap": (
                current_spread_bps <= dynamic_spread_cap
            ),
            "bid_depth_eur_top_5": bid_top_5,
            "ask_depth_eur_top_5": ask_top_5,
            "bid_depth_eur_top_10": bid_top_10,
            "ask_depth_eur_top_10": ask_top_10,
            "mlobi_top_5": (
                (bid_top_5 - ask_top_5) / denominator_5
                if denominator_5 > 0
                else None
            ),
            "mlobi_top_10": (
                (bid_top_10 - ask_top_10) / denominator_10
                if denominator_10 > 0
                else None
            ),
            "distance_weighted_imbalance_top_10": (
                (weighted_bid - weighted_ask) / weighted_denominator
                if weighted_denominator > 0
                else None
            ),
            **{
                f"bid_depth_eur_within_{distance}_bps": bid_depth
                for distance, (bid_depth, _ask_depth) in distance_depth.items()
            },
            **{
                f"ask_depth_eur_within_{distance}_bps": ask_depth
                for distance, (_bid_depth, ask_depth) in distance_depth.items()
            },
            **{
                f"depth_imbalance_within_{distance}_bps": (
                    (bid_depth - ask_depth) / (bid_depth + ask_depth)
                    if bid_depth + ask_depth > 0
                    else None
                )
                for distance, (bid_depth, ask_depth) in distance_depth.items()
            },
            "bid_book_slope_eur_per_bp": (
                distance_depth[25][0] / 25.0
            ),
            "ask_book_slope_eur_per_bp": (
                distance_depth[25][1] / 25.0
            ),
            "asks": [
                (float(price), float(quantity))
                for price, quantity in sorted_asks
            ],
        }
        rows = self._realtime_books[market]
        rows.append(record)
        cutoff = record["observed_at"] - timedelta(minutes=70)
        while rows and rows[0]["observed_at"] < cutoff:
            rows.popleft()

    @staticmethod
    def _window_trade_metrics(
        rows: list[tuple[datetime, float, float, int]],
        *,
        now: datetime,
        seconds: int,
    ) -> dict[str, float | int | None]:
        current = [
            row
            for row in rows
            if row[0] >= now - timedelta(seconds=seconds)
        ]
        if not current:
            return {
                "return": None,
                "quote_volume_eur": 0.0,
                "trade_count": 0,
                "taker_buy_ratio": None,
                "cvd_quote_eur": 0.0,
            }
        quote_volume = sum(row[2] for row in current)
        buy_volume = sum(row[2] for row in current if row[3] > 0)
        return {
            "return": (
                current[-1][1] / current[0][1] - 1.0
                if len(current) >= 2 and current[0][1] > 0
                else 0.0
            ),
            "quote_volume_eur": quote_volume,
            "trade_count": len(current),
            "taker_buy_ratio": (
                buy_volume / quote_volume if quote_volume > 0 else None
            ),
            "cvd_quote_eur": sum(row[2] * row[3] for row in current),
        }

    @staticmethod
    def _relative_window_activity(
        rows: list[tuple[datetime, float, float, int]],
        *,
        now: datetime,
        seconds: int,
        metric_index: int | None,
    ) -> float | None:
        current_start = now - timedelta(seconds=seconds)
        baseline_start = current_start - timedelta(seconds=seconds * 10)
        current = [row for row in rows if row[0] >= current_start]
        baseline = [
            row for row in rows if baseline_start <= row[0] < current_start
        ]
        if not current or not baseline:
            return None
        if metric_index is None:
            current_value = float(len(current))
            baseline_value = float(len(baseline)) / 10.0
        else:
            current_value = sum(row[metric_index] for row in current)
            baseline_value = (
                sum(row[metric_index] for row in baseline) / 10.0
            )
        return (
            current_value / baseline_value
            if baseline_value > 0
            else None
        )

    @staticmethod
    def _buy_slippage_bps(
        asks: list[tuple[float, float]],
        *,
        notional_eur: float,
    ) -> float | None:
        if not asks or notional_eur <= 0:
            return None
        best_ask = asks[0][0]
        remaining = notional_eur
        acquired = 0.0
        spent = 0.0
        for price, quantity in asks:
            available = price * quantity
            selected = min(available, remaining)
            if selected <= 0:
                continue
            spent += selected
            acquired += selected / price
            remaining -= selected
            if remaining <= 1e-12:
                break
        if remaining > 1e-9 or acquired <= 0 or best_ask <= 0:
            return None
        average = spent / acquired
        return (average / best_ask - 1.0) * 10_000.0

    @staticmethod
    def _compiled_trade_snapshot(
        rows: list[tuple[datetime, float, float, int]],
        *,
        now: datetime,
    ) -> dict[str, Any]:
        """Compile all realtime trade windows in one linear pass.

        The previous implementation rescanned the complete 70-minute trade
        deque for every return, CVD and activity horizon.  On liquid markets
        that meant millions of Python comparisons per second.  Prefix sums
        retain the exact causal definitions while making every window lookup
        constant-time after one bounded pass.
        """

        labels = {
            "10s": 10,
            "30s": 30,
            "90s": 90,
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "1h": 3_600,
        }
        empty_window: dict[str, float | int | None] = {
            "return": None,
            "quote_volume_eur": 0.0,
            "trade_count": 0,
            "taker_buy_ratio": None,
            "cvd_quote_eur": 0.0,
        }
        if not rows:
            return {
                "windows": {
                    label: dict(empty_window) for label in labels
                },
                "prior_1m_return": None,
                "one_minute_trades": [],
                "relative_volume_1m": None,
                "relative_volume_5m": None,
                "trade_intensity_1m": None,
            }

        timestamps: list[datetime] = []
        quote_prefix = [0.0]
        buy_prefix = [0.0]
        cvd_prefix = [0.0]
        for timestamp, _price, quote, direction in rows:
            timestamps.append(timestamp)
            quote_prefix.append(quote_prefix[-1] + quote)
            buy_prefix.append(
                buy_prefix[-1] + (quote if direction > 0 else 0.0)
            )
            cvd_prefix.append(cvd_prefix[-1] + quote * direction)

        end = len(rows)

        def index(seconds: int) -> int:
            return bisect_left(
                timestamps,
                now - timedelta(seconds=seconds),
            )

        starts = {label: index(seconds) for label, seconds in labels.items()}

        def window(start: int, stop: int = end) -> dict[str, float | int | None]:
            count = max(0, stop - start)
            if count == 0:
                return dict(empty_window)
            quote = quote_prefix[stop] - quote_prefix[start]
            buy = buy_prefix[stop] - buy_prefix[start]
            return {
                "return": (
                    rows[stop - 1][1] / rows[start][1] - 1.0
                    if count >= 2 and rows[start][1] > 0
                    else 0.0
                ),
                "quote_volume_eur": quote,
                "trade_count": count,
                "taker_buy_ratio": buy / quote if quote > 0 else None,
                "cvd_quote_eur": cvd_prefix[stop] - cvd_prefix[start],
            }

        windows = {
            label: window(starts[label]) for label in labels
        }
        prior_start = index(120)
        prior_end = starts["1m"]
        prior_1m_return = (
            rows[prior_end - 1][1] / rows[prior_start][1] - 1.0
            if prior_end - prior_start >= 2
            and rows[prior_start][1] > 0
            else None
        )

        def activity(seconds: int, *, volume: bool) -> float | None:
            current_start = index(seconds)
            baseline_start = index(seconds * 11)
            current_count = end - current_start
            baseline_count = current_start - baseline_start
            if current_count <= 0 or baseline_count <= 0:
                return None
            if volume:
                current_value = quote_prefix[end] - quote_prefix[current_start]
                baseline_value = (
                    quote_prefix[current_start]
                    - quote_prefix[baseline_start]
                ) / 10.0
            else:
                current_value = float(current_count)
                baseline_value = float(baseline_count) / 10.0
            return (
                current_value / baseline_value
                if baseline_value > 0
                else None
            )

        return {
            "windows": windows,
            "prior_1m_return": prior_1m_return,
            "one_minute_trades": rows[starts["1m"] :],
            "relative_volume_1m": activity(60, volume=True),
            "relative_volume_5m": activity(300, volume=True),
            "trade_intensity_1m": activity(60, volume=False),
        }

    def realtime_snapshot(
        self,
        *,
        markets: Iterable[str] | None = None,
        observed_at: datetime | None = None,
        order_notional_eur: float = 5.0,
    ) -> dict[str, Any]:
        """Return causal sub-minute flow facts for event-driven decisions."""

        now = _utc(observed_at or utc_now())
        selected_markets = tuple(markets or self.markets)
        output: list[dict[str, Any]] = []
        for market in selected_markets:
            trades = list(self._realtime_trades.get(market, ()))
            tickers = list(self._realtime_tickers.get(market, ()))
            books = list(self._realtime_books.get(market, ()))
            latest_trade_at = trades[-1][0] if trades else None
            latest_ticker_at = tickers[-1][0] if tickers else None
            latest_book = books[-1] if books else None
            latest_book_at = (
                latest_book.get("observed_at") if latest_book else None
            )
            timestamps = [
                value
                for value in (
                    latest_trade_at,
                    latest_ticker_at,
                    latest_book_at,
                )
                if isinstance(value, datetime)
            ]
            latest_at = max(timestamps) if timestamps else None
            age_seconds = (
                max(0.0, (now - latest_at).total_seconds())
                if latest_at is not None
                else None
            )
            compiled = self._compiled_trade_snapshot(trades, now=now)
            windows = compiled["windows"]
            current_1m = windows["1m"].get("return")
            prior_1m = compiled["prior_1m_return"]
            recent_books = [
                row
                for row in books
                if row["observed_at"] >= now - timedelta(minutes=1)
            ]
            persistent_books = [
                row
                for row in books
                if row["observed_at"] >= now - timedelta(seconds=10)
            ]
            first_book = recent_books[0] if recent_books else None

            def normalized_ofi(seconds: int) -> float | None:
                selected = [
                    row
                    for row in books
                    if row["observed_at"] >= now - timedelta(seconds=seconds)
                ]
                if len(selected) < 2:
                    return None
                first = selected[0]
                last = selected[-1]
                starting_depth = (
                    float(first["bid_depth_eur_top_10"])
                    + float(first["ask_depth_eur_top_10"])
                )
                if starting_depth <= 0:
                    return None
                bid_delta = (
                    float(last["bid_depth_eur_top_10"])
                    - float(first["bid_depth_eur_top_10"])
                )
                ask_delta = (
                    float(last["ask_depth_eur_top_10"])
                    - float(first["ask_depth_eur_top_10"])
                )
                return (bid_delta - ask_delta) / starting_depth

            ofi_windows = {
                label: normalized_ofi(seconds)
                for label, seconds in (
                    ("10s", 10),
                    ("30s", 30),
                    ("90s", 90),
                    ("300s", 300),
                )
            }
            bid_change = (
                float(latest_book["bid_depth_eur_top_10"])
                - float(first_book["bid_depth_eur_top_10"])
                if latest_book and first_book
                else None
            )
            ask_change = (
                float(latest_book["ask_depth_eur_top_10"])
                - float(first_book["ask_depth_eur_top_10"])
                if latest_book and first_book
                else None
            )
            depth_total = (
                float(first_book["bid_depth_eur_top_10"])
                + float(first_book["ask_depth_eur_top_10"])
                if first_book
                else 0.0
            )
            ofi = (
                (bid_change - ask_change) / depth_total
                if bid_change is not None
                and ask_change is not None
                and depth_total > 0
                else None
            )
            positive_book_persistence = (
                sum(
                    float(row.get("mlobi_top_10") or 0.0) > 0.03
                    for row in persistent_books
                )
                / len(persistent_books)
                if persistent_books
                else None
            )
            one_minute_trades = compiled["one_minute_trades"]
            current_price = (
                trades[-1][1]
                if trades
                else tickers[-1][1]
                if tickers
                else None
            )
            one_minute_low = (
                min(row[1] for row in one_minute_trades)
                if one_minute_trades
                else None
            )
            one_minute_first = (
                one_minute_trades[0][1] if one_minute_trades else None
            )
            sell_quote = sum(
                row[2] for row in one_minute_trades if row[3] < 0
            )
            one_minute_quote = sum(row[2] for row in one_minute_trades)
            price_stability = (
                max(
                    0.0,
                    1.0
                    - abs(
                        float(windows["1m"].get("return") or 0.0)
                    )
                    / 0.005,
                )
                if one_minute_trades
                else 0.0
            )
            bid_refill_ratio = (
                max(0.0, min(1.0, float(bid_change) / depth_total))
                if bid_change is not None and depth_total > 0
                else 0.0
            )
            ask_depletion_ratio = (
                max(0.0, min(1.0, -float(ask_change) / depth_total))
                if ask_change is not None and depth_total > 0
                else 0.0
            )
            bullish_absorption = (
                (sell_quote / one_minute_quote)
                * price_stability
                * bid_refill_ratio
                if one_minute_quote > 0
                else 0.0
            )
            downside_sweep_reclaim = bool(
                current_price is not None
                and one_minute_low is not None
                and one_minute_first is not None
                and one_minute_first > 0
                and one_minute_low / one_minute_first - 1.0 <= -0.003
                and current_price / one_minute_low - 1.0 >= 0.002
                and float(windows["1m"].get("cvd_quote_eur") or 0.0) > 0
            )
            output.append(
                {
                    "market": market,
                    "observed_at": now.isoformat(),
                    "latest_event_at": (
                        latest_at.isoformat() if latest_at else None
                    ),
                    "latest_trade_at": (
                        latest_trade_at.isoformat()
                        if latest_trade_at
                        else None
                    ),
                    "latest_ticker_at": (
                        latest_ticker_at.isoformat()
                        if latest_ticker_at
                        else None
                    ),
                    "age_seconds": age_seconds,
                    "fresh": age_seconds is not None and age_seconds <= 10,
                    "trade_age_seconds": (
                        max(0.0, (now - latest_trade_at).total_seconds())
                        if latest_trade_at is not None
                        else None
                    ),
                    "ticker_age_seconds": (
                        max(0.0, (now - latest_ticker_at).total_seconds())
                        if latest_ticker_at is not None
                        else None
                    ),
                    "price": current_price,
                    "windows": windows,
                    "acceleration_1m": (
                        float(current_1m) - prior_1m
                        if current_1m is not None and prior_1m is not None
                        else None
                    ),
                    "relative_volume_1m": compiled["relative_volume_1m"],
                    "relative_volume_5m": compiled["relative_volume_5m"],
                    "trade_intensity_1m": compiled["trade_intensity_1m"],
                    "book": (
                        {
                            key: value
                            for key, value in latest_book.items()
                            if key not in {"observed_at", "asks"}
                        }
                        if latest_book
                        else None
                    ),
                    "book_age_seconds": (
                        max(0.0, (now - latest_book_at).total_seconds())
                        if isinstance(latest_book_at, datetime)
                        else None
                    ),
                    "ofi_1m": ofi,
                    "ofi_windows": ofi_windows,
                    "mlobi_positive_persistence_10s": (
                        positive_book_persistence
                    ),
                    "microprice_edge_bps": (
                        latest_book.get("microprice_edge_bps")
                        if latest_book
                        else None
                    ),
                    "bid_replenishment_eur_1m": bid_change,
                    "bid_replenishment_ratio_1m": bid_refill_ratio,
                    "ask_depletion_eur_1m": (
                        -ask_change if ask_change is not None else None
                    ),
                    "ask_depletion_ratio_1m": ask_depletion_ratio,
                    "book_update_count_10s": len(persistent_books),
                    "book_update_count_1m": len(recent_books),
                    "bullish_absorption_score_1m": bullish_absorption,
                    "downside_sweep_reclaim_1m": downside_sweep_reclaim,
                    "estimated_buy_slippage_bps": (
                        self._buy_slippage_bps(
                            latest_book.get("asks") or [],
                            notional_eur=order_notional_eur,
                        )
                        if latest_book
                        else None
                    ),
                    "sequence_valid": bool(
                        (self._books.get(market) or {}).get("valid")
                    ),
                    "synthetic_data_used": False,
                }
            )
        return {
            "schema_version": "realtime_microstructure_snapshot_v1",
            "observed_at": now.isoformat(),
            "markets": output,
            "market_data_levels": {
                "L1": {
                    "status": "AVAILABLE_NATIVE",
                    "fields": ["ticker", "best_bid", "best_ask", "spread"],
                },
                "L2": {
                    "status": "AVAILABLE_NATIVE_AGGREGATED_PRICE_LEVELS",
                    "fields": [
                        "depth_5_10_25_bps",
                        "multi_level_imbalance",
                        "microprice",
                        "replenishment",
                        "ofi",
                        "slippage_depth_walk",
                    ],
                },
                "L3": {
                    "status": "UNAVAILABLE_ON_CURRENT_BITVAVO_FEED",
                    "individual_order_ids_available": False,
                    "synthetic_values_used": False,
                },
            },
            "orders_generated": 0,
            "synthetic_data_used": False,
        }

    def realtime_ticker_movers(
        self,
        *,
        observed_at: datetime | None = None,
        minimum_sample_span_seconds: float = 8.0,
        minimum_quote_volume_eur: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Rank positive venue-wide moves from the cheap ticker stream.

        The isolated order-flow connection subscribes to ticker24h for every
        Bitvavo EUR market, while trades and books remain resource-bounded.
        This projection makes that inexpensive coverage actionable for
        *data-promotion* only: it can rotate a market into intensive tracking,
        but it cannot grant strategy authority or create an order.
        """

        now = _utc(observed_at or utc_now())

        def window_return(
            rows: list[tuple[datetime, float, float | None]],
            seconds: int,
        ) -> float | None:
            if len(rows) < 2:
                return None
            cutoff = now - timedelta(seconds=seconds)
            candidates = [row for row in rows if row[0] >= cutoff]
            if len(candidates) < 2:
                return None
            span = (candidates[-1][0] - candidates[0][0]).total_seconds()
            required = max(float(seconds) * 0.75, minimum_sample_span_seconds)
            first = float(candidates[0][1])
            last = float(candidates[-1][1])
            if span < required or first <= 0.0:
                return None
            return last / first - 1.0

        ranking: list[dict[str, Any]] = []
        for market, source in self._realtime_tickers.items():
            rows = list(source)
            if len(rows) < 2:
                continue
            returns = {
                label: window_return(rows, seconds)
                for label, seconds in (
                    ("10s", 10),
                    ("1m", 60),
                    ("3m", 180),
                    ("5m", 300),
                    ("15m", 900),
                    ("1h", 3_600),
                )
            }
            available = [value for value in returns.values() if value is not None]
            if not available:
                continue
            return_1m = float(returns["1m"] or 0.0)
            return_10s = float(returns["10s"] or 0.0)
            return_3m = float(returns["3m"] or 0.0)
            return_5m = float(returns["5m"] or 0.0)
            return_15m = float(returns["15m"] or 0.0)
            return_1h = float(returns["1h"] or 0.0)
            latest_quote_volume = next(
                (
                    float(row[2])
                    for row in reversed(rows)
                    if row[2] is not None
                ),
                None,
            )
            acceleration = return_1m - max(0.0, return_5m / 5.0)
            activity = min(1.0, len(rows) / 60.0)
            score = 100.0 * (
                0.15 * min(1.0, max(0.0, return_10s) / 0.004)
                + 0.20 * min(1.0, max(0.0, return_1m) / 0.008)
                + 0.15 * min(1.0, max(0.0, return_3m) / 0.015)
                + 0.15 * min(1.0, max(0.0, return_5m) / 0.025)
                + 0.12 * min(1.0, max(0.0, return_15m) / 0.050)
                + 0.08 * min(1.0, max(0.0, return_1h) / 0.100)
                + 0.10 * min(1.0, max(0.0, acceleration) / 0.006)
                + 0.05 * activity
            )
            liquid = bool(
                latest_quote_volume is not None
                and latest_quote_volume >= minimum_quote_volume_eur
            )
            qualified = bool(
                liquid
                and (
                    return_10s >= 0.0015
                    or return_1m >= 0.002
                    or return_3m >= 0.004
                    or return_5m >= 0.006
                    or return_15m >= 0.010
                    or return_1h >= 0.020
                )
            )
            ranking.append(
                {
                    "market": market,
                    "observed_at": now.isoformat(),
                    "latest_ticker_at": rows[-1][0].isoformat(),
                    "latest_price": float(rows[-1][1]),
                    "returns": returns,
                    "acceleration_1m_vs_5m_rate": acceleration,
                    "ticker_sample_count": len(rows),
                    "quote_volume_24h_eur": latest_quote_volume,
                    "minimum_quote_volume_eur": minimum_quote_volume_eur,
                    "liquidity_qualified": liquid,
                    "score": score,
                    "qualified_for_intensive_tracking": qualified,
                    "execution_authority_granted": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            )
        ranking.sort(
            key=lambda row: (
                bool(row["qualified_for_intensive_tracking"]),
                float(row["score"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        return ranking

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
        *,
        reset_period_baselines: bool = False,
    ) -> None:
        for counter in self._acknowledged_stream_counters:
            current = int(health.get(counter) or 0)
            self._acknowledged_stream_counters[counter] = current
            # Initial synchronization can legitimately observe buffered
            # pre-snapshot deltas.  Those are discarded against the REST
            # nonce and must not poison the first interval.  During an actual
            # recovery, however, the counter delta remains evidence that the
            # affected hour/quarter was incomplete.  Keep the forensic period
            # baselines intact unless this is the initial synchronization.
            if reset_period_baselines:
                self._health_counters[counter] = current
                self._quarter_health_counters[counter] = current
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
            "ledger_integrity_status": self.ledger.integrity_status,
            "ledger_recovery_mode": self.ledger.recovery_mode,
            "full_historical_rescan_deferred": (
                self.ledger.full_historical_rescan_deferred
            ),
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

    def invalid_orderbook_markets(self) -> tuple[str, ...]:
        """Return subscribed markets whose local book needs a fresh seed.

        The transport-level sequence counter is useful, but it is not the
        final authority for every local-book failure.  A malformed delta or
        a market-specific nonce discontinuity can invalidate one book while
        the shared WebSocket remains connected.  Exposing that state lets the
        supervisor repair only the affected markets instead of leaving a
        valid opportunity permanently blocked.
        """

        return tuple(
            market
            for market in self.markets
            if not bool((self._books.get(market) or {}).get("valid"))
        )

    def _persist_database(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        if self.database is None:
            return
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
            else:
                # Every delta has already been applied to the in-memory book.
                # Persisting an empty shell for every high-frequency update
                # adds no analytical value: hourly OBI/OFI uses periodic full
                # states, while trades remain lossless.  Five-second states
                # keep the signal path responsive and auditable.
                continue
            sorted_bids = sorted(bids.items(), reverse=True)[:500]
            sorted_asks = sorted(asks.items())[:500]
            state["bids"] = dict(sorted_bids)
            state["asks"] = dict(sorted_asks)
            self._record_realtime_book(
                market=event.canonical_market,
                observed_at=event.observed_at,
                bids=state["bids"],
                asks=state["asks"],
                valid=bool(state.get("valid")),
            )
            book_bids = [
                [str(price), str(quantity)]
                for price, quantity in sorted_bids[:100]
            ]
            book_asks = [
                [str(price), str(quantity)]
                for price, quantity in sorted_asks[:100]
            ]
            payload.update(
                {
                    "book_bids": book_bids,
                    "book_asks": book_asks,
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

    def _prepare_batch(
        self,
        events: Iterable[NormalizedStreamEvent],
    ) -> list[NormalizedStreamEvent]:
        """Prepare one durable batch outside the supervisor event loop."""

        return self._enrich_orderbooks(self._select_events(events))

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
            perpetual_quote = _decimal(
                (derivative or {}).get(
                    "perpetual_quote_volume_24h"
                )
            )
            spot_quote = _decimal(
                row.get("spot_quote_volume_24h")
            )
            spot_last_price = _decimal(
                row.get("spot_last_price_eur")
            )
            perpetual_index_price = _decimal(
                (derivative or {}).get(
                    "perpetual_index_price_usdt"
                )
            )
            implied_usdt_per_eur = (
                perpetual_index_price / spot_last_price
                if perpetual_index_price is not None
                and perpetual_index_price > 0
                and spot_last_price is not None
                and spot_last_price > 0
                else None
            )
            spot_quote_usdt = (
                spot_quote * implied_usdt_per_eur
                if spot_quote is not None
                and spot_quote > 0
                and implied_usdt_per_eur is not None
                else None
            )
            quote_volume_ratio = (
                float(perpetual_quote / spot_quote_usdt)
                if perpetual_quote is not None
                and spot_quote_usdt is not None
                and spot_quote_usdt > 0
                else None
            )
            row["implied_usdt_per_eur"] = (
                float(implied_usdt_per_eur)
                if implied_usdt_per_eur is not None
                else None
            )
            row["spot_quote_volume_24h_usdt"] = (
                float(spot_quote_usdt)
                if spot_quote_usdt is not None
                else None
            )
            row["perpetual_spot_volume_ratio"] = (
                quote_volume_ratio
            )
            row["perpetual_spot_quote_volume_ratio"] = (
                quote_volume_ratio
            )
            row["volume_ratio_method"] = (
                "MEXC_PERPETUAL_USDT_AMOUNT24_DIVIDED_BY_"
                "BITVAVO_SPOT_EUR_VOLUMEQUOTE24H_CONVERTED_"
                "WITH_ASSET_IMPLIED_USDT_PER_EUR"
            )
            row["quote_currency_conversion_status"] = (
                "AVAILABLE"
                if implied_usdt_per_eur is not None
                else "UNAVAILABLE"
            )
            row["perpetual_spot_base_volume_ratio"] = None
            row["base_volume_ratio_status"] = (
                "NOT_COMPARABLE_MEXC_VOLUME24_IS_CONTRACT_COUNT"
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
                "quote_currency_conversion_available": (
                    implied_usdt_per_eur is not None
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
            if row["perpetual_spot_volume_ratio"] is None:
                row["status"] = "DATA_GAP"
                row["reason_codes"].append(
                    "PERPETUAL_SPOT_QUOTE_VOLUME_RATIO_UNAVAILABLE"
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

    def finalize_previous_quarter(
        self,
        *,
        observed_at: datetime,
        health: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal one completed spot-only 15-minute orderflow bucket.

        Derivatives and GEX remain independently timestamped regime overlays.
        Requiring those slower fields inside every 15-minute spot bucket would
        add latency without improving the point-in-time integrity of the flow
        confirmation itself.
        """

        interval_start = _closed_quarter_start(observed_at)
        interval_end = interval_start + timedelta(minutes=15)
        directory = self.fifteen_minute_feature_directory
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / interval_start.strftime(
            "%Y%m%dT%H%M00Z.json"
        )
        if target.is_file():
            return {
                "finalization_state": "ALREADY_FINALIZED",
                "snapshot": dict(read_json(target)),
            }

        provider_health = dict(health.get("bitvavo") or health)
        period_health = dict(provider_health)
        for counter in self._quarter_health_counters:
            current = int(provider_health.get(counter) or 0)
            period_health[counter] = max(
                0,
                current - self._quarter_health_counters[counter],
            )
            self._quarter_health_counters[counter] = current

        records = _read_segment(self.ledger.root, interval_start)
        markets = [
            summarize_orderflow_hour(
                records,
                market=market,
                hour_start=interval_start,
                recorder_started_at=self.started_at,
                health=period_health,
                interval=timedelta(minutes=15),
            )
            for market in self.markets
        ]
        _apply_cvd_history(
            directory,
            target_path=target,
            markets=markets,
        )
        body = {
            "schema_version": MICROSTRUCTURE_15M_SCHEMA,
            # Keep hour_start/hour_end aliases so the existing causal loader
            # can consume interval snapshots without a second timestamp path.
            "hour_start": interval_start.isoformat(),
            "hour_end": interval_end.isoformat(),
            "interval_start": interval_start.isoformat(),
            "interval_end": interval_end.isoformat(),
            "interval_minutes": 15,
            "finalized_at": _utc(observed_at).isoformat(),
            "status": (
                "COMPLETE"
                if all(row["status"] == "COMPLETE" for row in markets)
                else "DATA_GAP"
            ),
            "markets": markets,
            "stream_health": period_health,
            "ledger_root_hash": self.ledger.previous_hash,
            "synthetic_data_used": False,
            "orders_generated": 0,
        }
        snapshot = {
            **body,
            "snapshot_hash": stable_hash(body, length=64),
        }
        atomic_write_json(target, snapshot)
        return {
            "finalization_state": "FINALIZED",
            "snapshot": snapshot,
        }

    async def run(self, manager: Any) -> None:
        buffer: list[NormalizedStreamEvent] = []
        last_flush = asyncio.get_running_loop().time()
        last_finalized: datetime | None = None
        last_finalization_attempt: datetime | None = None
        last_quarter_finalized: datetime | None = None
        last_quarter_attempt: datetime | None = None
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
                        self._record_realtime_event(event)
                        buffer.append(event)
                        # A busy top-20 book stream can keep queue reads
                        # immediately ready.  Yield explicitly so execution,
                        # reconciliation and heartbeat tasks are never starved.
                        if len(buffer) % 50 == 0:
                            await asyncio.sleep(0.001)
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
                    batch = buffer.copy()
                    buffer.clear()
                    enriched = await asyncio.to_thread(
                        self._prepare_batch,
                        batch,
                    )
                    records = await asyncio.to_thread(
                        self.ledger.append,
                        enriched,
                    )
                    await asyncio.to_thread(
                        self._persist_database,
                        records,
                    )
                    last_flush = now_monotonic
                    self._write_health(manager)
                now = utc_now()
                closed_quarter = _closed_quarter_start(now)
                quarter_end = closed_quarter + timedelta(minutes=15)
                if (
                    (
                        last_quarter_finalized is None
                        or closed_quarter > last_quarter_finalized
                    )
                    and now >= quarter_end + timedelta(minutes=1)
                    and (
                        last_quarter_attempt is None
                        or (now - last_quarter_attempt).total_seconds() >= 5
                    )
                ):
                    await asyncio.to_thread(
                        self.finalize_previous_quarter,
                        observed_at=now,
                        health=manager.health(),
                    )
                    last_quarter_attempt = now
                    last_quarter_finalized = closed_quarter
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
                    result = await asyncio.to_thread(
                        self.finalize_previous_hour,
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
                enriched = await asyncio.to_thread(
                    self._prepare_batch,
                    buffer.copy(),
                )
                records = await asyncio.to_thread(
                    self.ledger.append,
                    enriched,
                )
                await asyncio.to_thread(
                    self._persist_database,
                    records,
                )
            self._write_health(manager)


__all__ = [
    "HashChainedOrderflowLedger",
    "ProspectiveOrderflowRecorder",
    "audit_microstructure_snapshots",
    "current_microstructure_readiness",
    "microstructure_storage_runway",
    "normalize_stream_event",
    "prospective_milestone_status",
    "seal_completed_orderflow_segments",
    "summarize_orderflow_hour",
    "verify_orderflow_checkpoint",
    "verify_orderflow_ledger",
]
