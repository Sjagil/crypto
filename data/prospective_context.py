"""Hourly point-in-time CMC and derivatives context collection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
    utc_now,
)


def most_recent_closed_utc_hour(
    observed_at: datetime,
) -> datetime:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    normalized = observed_at.astimezone(UTC).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    return normalized - timedelta(hours=1)


class ProspectiveContextCollector:
    """Collect source facts only; it never creates signals or orders."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        snapshot_directory: Path,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.snapshot_directory = snapshot_directory

    def _last_epoch(self) -> datetime | None:
        if not self.checkpoint_path.is_file():
            return None
        value = read_json(self.checkpoint_path).get(
            "last_completed_epoch"
        )
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            raise RuntimeError("CONTEXT_CHECKPOINT_TIMEZONE_INVALID")
        return parsed.astimezone(UTC)

    def due(
        self,
        *,
        observed_at: datetime | None = None,
    ) -> tuple[bool, datetime]:
        epoch = most_recent_closed_utc_hour(
            observed_at or utc_now()
        )
        previous = self._last_epoch()
        return previous is None or previous < epoch, epoch

    async def collect(
        self,
        *,
        loader: Any,
        markets: tuple[str, ...],
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        now = observed_at or utc_now()
        is_due, epoch = self.due(observed_at=now)
        if not is_due:
            return {
                "status": "UP_TO_DATE",
                "last_completed_epoch": self._last_epoch(),
                "orders_generated": 0,
            }
        rankings = await loader.download_cmc_rankings(
            limit=50,
            convert="EUR",
            persist=True,
        )
        if len(rankings) != 50:
            return {
                "status": "BLOCK_NEW_ENTRIES",
                "reason_code": "CMC_TOP50_INCOMPLETE",
                "received_rankings": len(rankings),
                "target_epoch": epoch,
                "orders_generated": 0,
            }
        derivatives: list[Any] = []
        failures: list[str] = []
        for market in markets:
            base = market.split("-", 1)[0]
            try:
                records = await loader.download_derivatives_context(
                    provider="mexc",
                    market=f"{base}-USDT",
                    persist=True,
                )
                if not records:
                    failures.append(
                        f"DERIVATIVES_EMPTY:{base}-USDT"
                    )
                derivatives.extend(records)
            except Exception as exc:  # fail closed, preserve source reason
                failures.append(
                    f"DERIVATIVES_FAILED:{base}-USDT:"
                    f"{type(exc).__name__}"
                )
        if failures:
            return {
                "status": "BLOCK_NEW_ENTRIES",
                "reason_code": "PROSPECTIVE_CONTEXT_INCOMPLETE",
                "failures": failures,
                "received_rankings": len(rankings),
                "received_derivatives": len(derivatives),
                "target_epoch": epoch,
                "orders_generated": 0,
            }
        snapshot_body = {
            "schema_version": "prospective_context_snapshot_v1",
            "target_closed_utc_hour": epoch.isoformat(),
            "collected_at": now.astimezone(UTC).isoformat(),
            "coinmarketcap_top50": [
                {
                    "canonical_market": record.canonical_market,
                    "timestamp": record.timestamp.isoformat(),
                    "observed_at": record.observed_at.isoformat(),
                    "available_at": record.available_at.isoformat(),
                    "raw_hash": record.raw_hash,
                    "values": record.values,
                }
                for record in rankings
            ],
            "derivatives_context": [
                {
                    "canonical_market": record.canonical_market,
                    "timestamp": record.timestamp.isoformat(),
                    "observed_at": record.observed_at.isoformat(),
                    "available_at": record.available_at.isoformat(),
                    "raw_hash": record.raw_hash,
                    "values": record.values,
                }
                for record in derivatives
            ],
            "synthetic_data_used": False,
            "orders_generated": 0,
        }
        snapshot = {
            **snapshot_body,
            "snapshot_hash": stable_hash(
                snapshot_body,
                length=64,
            ),
        }
        self.snapshot_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        snapshot_path = self.snapshot_directory / (
            epoch.strftime("%Y%m%dT%H0000Z") + ".json"
        )
        if snapshot_path.is_file():
            if read_json(snapshot_path) != snapshot:
                raise RuntimeError(
                    "PROSPECTIVE_CONTEXT_HISTORY_REVISION"
                )
        else:
            atomic_write_json(snapshot_path, snapshot)
        checkpoint = {
            "schema_version": "prospective_context_checkpoint_v1",
            "status": "PASSED",
            "last_completed_epoch": epoch.isoformat(),
            "snapshot_path": str(snapshot_path),
            "snapshot_hash": snapshot["snapshot_hash"],
            "ranking_count": len(rankings),
            "derivatives_count": len(derivatives),
            "orders_generated": 0,
        }
        atomic_write_json(self.checkpoint_path, checkpoint)
        return checkpoint


__all__ = [
    "ProspectiveContextCollector",
    "most_recent_closed_utc_hour",
]
