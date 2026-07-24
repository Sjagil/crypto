from __future__ import annotations

from datetime import timedelta

import pytest

from config.settings import Settings
from data.data_loader import DataLoader
from utils.common import utc_now

pytestmark = pytest.mark.network


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "market"),
    (
        ("bitvavo", "BTC-EUR"),
        ("kraken", "BTC-EUR"),
        ("mexc", "BTC-USDT"),
    ),
)
async def test_public_exchange_historical_and_orderbook(
    isolated_settings: Settings,
    provider: str,
    market: str,
) -> None:
    loader = DataLoader(isolated_settings)
    end = utc_now()
    records = await loader.download_ohlcv(
        provider=provider,
        market=market,
        timeframe="1h",
        start=end - timedelta(hours=3),
        end=end,
        resume=False,
    )
    assert records
    assert all(record.provider == provider for record in records)
    snapshot = await loader.download_orderbook_snapshot(
        provider=provider,
        market=market,
        depth=10,
    )
    assert snapshot.values["bids"]
    assert snapshot.values["asks"]


@pytest.mark.asyncio
async def test_sec_public_data(isolated_settings: Settings) -> None:
    providers = isolated_settings.providers.model_copy(
        update={"sec_user_agent": "crypto-spot-research test@example.invalid"}
    )
    settings = isolated_settings.model_copy(update={"providers": providers})
    records = await DataLoader(settings).download_macro_series(
        provider="sec",
        series="0000320193",
    )
    assert records
    assert all(record.values["source_url"].startswith("https://www.sec.gov/") for record in records)
