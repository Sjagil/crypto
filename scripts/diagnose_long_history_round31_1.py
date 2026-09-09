#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from data.downloader import BitvavoProvider, KrakenProvider, PublicHttpClient
from data.market_data import drop_open_candles


async def main() -> int:
    settings = Settings.load(env_file=ROOT / ".env", create_directories=True)
    end = datetime.now(UTC)
    start = datetime(2019, 1, 1, tzinfo=UTC)
    results = []
    async with aiohttp.ClientSession() as session:
        client = PublicHttpClient(
            session,
            timeout_seconds=settings.scrapers.request_timeout_seconds,
            maximum_retries=settings.scrapers.maximum_retries,
            backoff_base_seconds=settings.scrapers.backoff_base_seconds,
        )
        for provider in (BitvavoProvider(client), KrakenProvider(client)):
            row = {"provider": provider.name, "market": "BTC-EUR", "timeframe": "1d"}
            try:
                frame = await provider.fetch_candles("BTC-EUR", "1d", start=start, end=end)
                frame = drop_open_candles(frame, timeframe="1d", now=end)
                row.update(status="READY", rows=len(frame), first=frame.index[0].isoformat(), last=frame.index[-1].isoformat())
            except Exception as exc:
                row.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            results.append(row)
    bitvavo_ready = any(r["provider"] == "bitvavo" and r["status"] == "READY" for r in results)
    payload = {
        "schema_version": "round31_1_long_history_diagnostic_v1",
        "status": "READY" if bitvavo_ready else "BLOCKED",
        "public_only": True,
        "private_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bitvavo_ready else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
