from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from data.prospective_context import (
    ProspectiveContextCollector,
    most_recent_closed_utc_hour,
)
from utils.common import read_json


def _record(
    market: str,
    *,
    observed: datetime,
    rank: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        canonical_market=market,
        timestamp=observed,
        observed_at=observed,
        available_at=observed,
        raw_hash=f"hash-{market}-{rank}",
        values={"cmc_rank": rank} if rank is not None else {},
    )


class FakeLoader:
    def __init__(self, observed: datetime) -> None:
        self.observed = observed
        self.ranking_calls = 0
        self.derivative_calls = 0

    async def download_cmc_rankings(self, **kwargs):
        assert kwargs == {
            "limit": 50,
            "convert": "EUR",
            "persist": True,
        }
        self.ranking_calls += 1
        return [
            _record(
                f"ASSET{rank}-EUR",
                observed=self.observed,
                rank=rank,
            )
            for rank in range(1, 51)
        ]

    async def download_derivatives_context(
        self,
        *,
        provider,
        market,
        persist,
    ):
        assert provider == "mexc"
        assert persist
        self.derivative_calls += 1
        return [_record(market, observed=self.observed)]


def test_closed_hour_is_strictly_previous_utc_hour() -> None:
    assert most_recent_closed_utc_hour(
        datetime(2026, 7, 26, 17, 33, tzinfo=UTC)
    ) == datetime(2026, 7, 26, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        most_recent_closed_utc_hour(datetime(2026, 7, 26, 17, 33))


@pytest.mark.asyncio
async def test_hourly_context_is_point_in_time_and_idempotent(
    tmp_path,
) -> None:
    observed = datetime(2026, 7, 26, 17, 33, tzinfo=UTC)
    loader = FakeLoader(observed)
    collector = ProspectiveContextCollector(
        checkpoint_path=tmp_path / "checkpoint.json",
        snapshot_directory=tmp_path / "snapshots",
    )

    first = await collector.collect(
        loader=loader,
        markets=("BTC-EUR", "ETH-EUR"),
        observed_at=observed,
    )
    second = await collector.collect(
        loader=loader,
        markets=("BTC-EUR", "ETH-EUR"),
        observed_at=observed,
    )

    assert first["status"] == "PASSED"
    assert second["status"] == "UP_TO_DATE"
    assert loader.ranking_calls == 1
    assert loader.derivative_calls == 2
    snapshot = read_json(first["snapshot_path"])
    assert len(snapshot["coinmarketcap_top50"]) == 50
    assert len(snapshot["derivatives_context"]) == 2
    assert not snapshot["synthetic_data_used"]
    assert snapshot["orders_generated"] == 0
    assert all(
        row["available_at"] == observed.isoformat()
        for row in snapshot["coinmarketcap_top50"]
    )


@pytest.mark.asyncio
async def test_incomplete_top50_fails_closed_without_checkpoint(
    tmp_path,
) -> None:
    observed = datetime(2026, 7, 26, 17, 33, tzinfo=UTC)

    class Incomplete(FakeLoader):
        async def download_cmc_rankings(self, **kwargs):
            return (await super().download_cmc_rankings(**kwargs))[:49]

    collector = ProspectiveContextCollector(
        checkpoint_path=tmp_path / "checkpoint.json",
        snapshot_directory=tmp_path / "snapshots",
    )
    result = await collector.collect(
        loader=Incomplete(observed),
        markets=("BTC-EUR",),
        observed_at=observed,
    )
    assert result["status"] == "BLOCK_NEW_ENTRIES"
    assert result["reason_code"] == "CMC_TOP50_INCOMPLETE"
    assert not (tmp_path / "checkpoint.json").exists()
