from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from config.settings import PathSettings, Settings
from core.contracts import NormalizedDataRecord
from data.data_loader import DataLoader
from data.database import Database


def test_non_bitvavo_parquet_gap_remains_retryable(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_settings.model_copy(
        update={
            "paths": PathSettings(project_root=tmp_path),
        }
    )

    database = Database(
        sqlite_path=tmp_path / "watermarks.db",
    )
    database.migrate()

    loader = DataLoader(
        settings,
        database=database,
    )

    start = datetime(2025, 1, 1, tzinfo=UTC)

    frame = pd.DataFrame(
        {
            "timestamp": [
                start,
                start + timedelta(minutes=15),
                start + timedelta(minutes=45),
            ],
            "observed_at": [
                start + timedelta(hours=1),
            ] * 3,
            "available_at": [
                start + timedelta(hours=1),
            ] * 3,
            "raw_hash": ["a", "b", "c"],
        }
    )

    path = tmp_path / "kraken.parquet"
    frame.to_parquet(path, index=False)

    loader._update_watermark_from_parquet(
        provider="kraken",
        market="BTC-EUR",
        timeframe="15m",
        data_kind="ohlcv",
        path=path,
        completed_ranges=((start, start + timedelta(minutes=45)),),
    )

    rows = database.fetch_records("data_watermarks")

    assert len(rows) == 1
    assert rows[0]["status"] == "PARTIAL"

    payload = rows[0]["payload"]

    assert len(payload["missing_ranges"]) == 1
    assert payload["sparse_ranges"] == []


@pytest.mark.asyncio
async def test_compact_bitvavo_sync_writes_ready_sparse_watermark(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    settings = isolated_settings.model_copy(
        update={
            "paths": PathSettings(project_root=tmp_path),
        }
    )

    database = Database(
        sqlite_path=tmp_path / "compact.db",
    )
    database.migrate()

    loader = DataLoader(
        settings,
        database=database,
    )

    start = datetime(2025, 1, 1, tzinfo=UTC)
    sparse_timestamp = start + timedelta(minutes=30)

    async def fake_ohlcv(
        market: str,
        timeframe: str,
        selected_start: datetime,
        selected_end: datetime,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        rows: list[NormalizedDataRecord] = []
        cursor = selected_start
        interval = timedelta(minutes=15)

        while cursor <= selected_end:
            if cursor != sparse_timestamp:
                rows.append(
                    NormalizedDataRecord(
                        provider="bitvavo",
                        source_symbol=market,
                        canonical_market=market,
                        timestamp=cursor,
                        observed_at=selected_end + interval,
                        available_at=selected_end + interval,
                        data_kind="ohlcv",
                        timeframe=timeframe,
                        closed=True,
                        retrieval_run_id=run_id,
                        raw_hash=f"hash-{cursor.isoformat()}",
                        raw_payload={
                            "timestamp": cursor.isoformat(),
                        },
                        values={
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.5,
                            "volume": 1.0,
                        },
                    )
                )

            cursor += interval

        return rows

    loader.adapters["bitvavo"].ohlcv = fake_ohlcv

    result = await loader.sync_canonical_ohlcv_compact(
        provider="bitvavo",
        market="BTC-EUR",
        timeframe="15m",
        start=start,
        end=start + timedelta(hours=2),
        resume=False,
    )

    assert result["rows"] > 0

    rows = database.fetch_records("data_watermarks")

    assert len(rows) == 1
    assert rows[0]["provider"] == "bitvavo"
    assert rows[0]["status"] == "READY"

    payload = rows[0]["payload"]

    assert payload["missing_ranges"] == []
    assert len(payload["sparse_ranges"]) == 1
    assert (
        payload["sparse_range_semantics"]
        == "BITVAVO_NO_TRADE_INTERVAL_OMITTED"
    )
    assert payload["watermark_source"] == "PARQUET_SCAN_V1"
