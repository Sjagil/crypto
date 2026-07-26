"""Measurable health evidence for the prospective market-data collector."""

from __future__ import annotations

import shutil
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from utils.common import read_json, stable_hash, utc_now


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _latest_payload(
    database: Any,
    table: str,
) -> dict[str, Any]:
    rows = database.fetch_recent_records(table, limit=1)
    return dict(rows[0].get("payload") or {}) if rows else {}


def _bytes_in(paths: Iterable[Path]) -> int:
    total = 0
    for root in paths:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    return total


def collector_health_report(
    *,
    settings: Any,
    database: Any,
    service_id: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = observed_at or utc_now()
    heartbeat_path = (
        settings.paths.checkpoints_dir
        / f"{service_id}_heartbeat.json"
    )
    lock_path = settings.paths.checkpoints_dir / "data_service.lock"
    heartbeat = (
        dict(read_json(heartbeat_path))
        if heartbeat_path.is_file()
        else {}
    )
    lock = (
        dict(read_json(lock_path))
        if lock_path.is_file()
        else {}
    )
    latest_trade = _latest_payload(database, "trades")
    latest_book = _latest_payload(
        database,
        "orderbook_snapshots",
    )
    trade_at = _timestamp(
        latest_trade.get("observed_at")
        or latest_trade.get("timestamp")
    )
    book_at = _timestamp(
        latest_book.get("observed_at")
        or latest_book.get("timestamp")
    )
    heartbeat_at = _timestamp(heartbeat.get("heartbeat_at"))
    heartbeat_age = (
        (now - heartbeat_at).total_seconds()
        if heartbeat_at
        else None
    )
    trade_age = (
        (now - trade_at).total_seconds() if trade_at else None
    )
    book_age = (
        (now - book_at).total_seconds() if book_at else None
    )
    providers = [
        dict(row.get("payload") or {})
        for row in database.fetch_records("provider_health")
    ]
    reconnect_count = sum(
        int(row.get("reconnect_count") or 0) for row in providers
    )
    gap_count = sum(
        int(
            row.get("gap_count")
            or row.get("sequence_gaps")
            or 0
        )
        for row in providers
    )
    disk = shutil.disk_usage(settings.paths.data_dir)
    free_disk_gb = disk.free / 1024**3
    raw_root = settings.paths.data_dir / "raw"
    orderflow_root = (
        settings.paths.context_data_dir / "orderflow_stream"
    )
    bytes_written = _bytes_in((raw_root, orderflow_root))
    maximum_storage_gb = float(
        getattr(
            settings.market_data,
            "maximum_storage_gb",
            50.0,
        )
    )
    maximum_storage_bytes = maximum_storage_gb * 1024**3
    lag_limit = max(
        180.0,
        float(settings.operational.cycle_seconds) * 3.0,
    )
    reason_codes: list[str] = []
    if heartbeat_age is None or heartbeat_age > lag_limit:
        reason_codes.append("HEARTBEAT_LAGGING")
    if trade_age is None or trade_age > lag_limit:
        reason_codes.append("TRADE_STREAM_LAGGING")
    if book_age is None or book_age > lag_limit:
        reason_codes.append("ORDERBOOK_STREAM_LAGGING")
    if gap_count:
        reason_codes.append("GAP_DETECTED")
    if free_disk_gb < settings.market_data.minimum_free_disk_gb:
        reason_codes.append("STORAGE_BLOCKED")
    if bytes_written >= maximum_storage_bytes:
        reason_codes.append("STORAGE_BLOCKED")
    status = (
        "STORAGE_BLOCKED"
        if "STORAGE_BLOCKED" in reason_codes
        else "GAP_DETECTED"
        if "GAP_DETECTED" in reason_codes
        else "LAGGING"
        if reason_codes
        else "HEALTHY"
    )
    report = {
        "schema_version": "collector_health_v1",
        "status": status,
        "reason_codes": reason_codes,
        "collector_id": service_id,
        "process_id": lock.get("pid") or heartbeat.get("pid"),
        "host": lock.get("hostname") or socket.gethostname(),
        "source": "bitvavo_spot_plus_cmc_mexc_context",
        "last_trade_timestamp": (
            trade_at.isoformat() if trade_at else None
        ),
        "last_orderbook_timestamp": (
            book_at.isoformat() if book_at else None
        ),
        "last_sequence_number": (
            (latest_book.get("values") or {}).get("sequence")
            if isinstance(latest_book.get("values"), dict)
            else latest_book.get("sequence")
        ),
        "last_heartbeat": (
            heartbeat_at.isoformat() if heartbeat_at else None
        ),
        "heartbeat_age_seconds": heartbeat_age,
        "reconnect_count": reconnect_count,
        "gap_count": gap_count,
        "bytes_written": bytes_written,
        "maximum_storage_bytes": maximum_storage_bytes,
        "storage_utilization": (
            bytes_written / maximum_storage_bytes
            if maximum_storage_bytes
            else None
        ),
        "free_disk_space_gb": free_disk_gb,
        "minimum_free_disk_space_gb": (
            settings.market_data.minimum_free_disk_gb
        ),
        "checksum": stable_hash(
            {
                "heartbeat": heartbeat,
                "last_trade": latest_trade,
                "last_orderbook": latest_book,
            },
            length=64,
        ),
        "orders_generated": 0,
    }
    return report


__all__ = ["collector_health_report"]
