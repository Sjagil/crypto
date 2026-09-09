#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from data.downloader import (
    BitvavoProvider,
    KrakenProvider,
    PublicHttpClient,
)
from data.market_data import drop_open_candles, quality_report


async def main() -> int:
    settings = Settings.load(
        env_file=ROOT / ".env",
        create_directories=True,
    )
    end = datetime.now(UTC)
    cases = (
        ("1h", end - timedelta(days=10)),
        ("1d", end - timedelta(days=45)),
    )
    rows = []

    async with aiohttp.ClientSession() as session:
        client = PublicHttpClient(
            session,
            timeout_seconds=settings.scrapers.request_timeout_seconds,
            maximum_retries=settings.scrapers.maximum_retries,
            backoff_base_seconds=settings.scrapers.backoff_base_seconds,
        )
        providers = (
            BitvavoProvider(client),
            KrakenProvider(client),
        )
        for provider in providers:
            for timeframe, start in cases:
                record = {
                    "provider": provider.name,
                    "market": "BTC-EUR",
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }
                try:
                    frame = await provider.fetch_candles(
                        "BTC-EUR",
                        timeframe,
                        start=start,
                        end=end,
                    )
                    frame = drop_open_candles(
                        frame,
                        timeframe=timeframe,
                        now=end,
                    )
                    report = quality_report(
                        frame,
                        market="BTC-EUR",
                        timeframe=timeframe,
                        maximum_staleness=settings.market_data.maximum_staleness,
                        now=end,
                    )
                    record.update(
                        {
                            "status": "READY",
                            "rows": len(frame),
                            "first": frame.index[0].isoformat(),
                            "last": frame.index[-1].isoformat(),
                            "quality_valid": report.valid,
                            "quality_reasons": list(report.reasons),
                            "missing_fraction": report.missing_fraction,
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "status": "FAILED",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                rows.append(record)

    payload = {
        "schema_version": "round31_mac_public_candle_diagnostic_v1",
        "platform": sys.platform,
        "status": (
            "READY"
            if any(row["status"] == "READY" for row in rows)
            else "BLOCKED"
        ),
        "public_only": True,
        "private_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
        "results": rows,
    }
    output = (
        ROOT
        / "output"
        / "reports"
        / "round31_mac_public_candle_diagnostic.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifact"] = str(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
