"""Capture public Bitvavo snapshots into a separate immutable V2 seed ledger."""

from __future__ import annotations

import asyncio
from pathlib import Path

from config.settings import get_settings
from data.data_loader import DataLoader
from data.multi_source_platform import (
    DataClassification,
    ImmutableSourceLedger,
    SourceNeutralObservation,
    TimestampResolution,
)
from utils.common import utc_now


async def capture() -> dict[str, object]:
    workspace = Path(__file__).resolve().parents[1]
    settings = get_settings()
    loader = DataLoader(settings)
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR")
    records = await asyncio.gather(
        *(
            loader.download_orderbook_snapshot(
                provider="bitvavo",
                market=market,
                depth=500,
                persist=False,
                mode="public_research_l2_v2_replay_seed",
            )
            for market in markets
        )
    )
    observations = []
    for record in records:
        asset = record.canonical_market.split("-", 1)[0]
        observations.append(
            SourceNeutralObservation(
                source="bitvavo_l2_v2_seed",
                source_type="PUBLIC_REST",
                venue="bitvavo",
                canonical_asset_id=f"CRYPTO:{asset}",
                venue_instrument_id=record.canonical_market,
                data_type="ORDERBOOK_SNAPSHOT",
                provider_timestamp=record.timestamp,
                local_receive_timestamp=record.observed_at,
                normalized_timestamp=record.timestamp,
                persisted_timestamp=utc_now(),
                raw_payload=record.raw_payload,
                timestamp_resolution=TimestampResolution.RETRIEVAL_ONLY,
                quality_state="TRUSTED_RESEED_SNAPSHOT",
                source_event_id=record.raw_hash,
                classification=DataClassification.PROSPECTIVE_COLLECTION,
                metadata={
                    "canonical_market": record.canonical_market,
                    "source_sequence": record.values.get("sequence"),
                    "raw_payload_hash": record.raw_hash,
                    "public_only": True,
                    "orders_generated": 0,
                    "private_exchange_requests": 0,
                },
            )
        )
    ledger = ImmutableSourceLedger(
        workspace / "data_store" / "raw" / "bitvavo" / "l2_v2_reseed_evidence",
        "bitvavo_l2_v2_seed",
        workspace / "output" / "multi_source" / "p1_2_3" / "seed_checkpoint.json",
    )
    return ledger.append_many(observations)


def main() -> int:
    print(asyncio.run(capture()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
