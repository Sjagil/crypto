from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.contracts import HistoricalCoverage, TimestampQuality
from scrapers.intelligence import (
    SourceSpec,
    audit_intelligence,
    deduplicate_intelligence,
    enrich_and_filter,
    parse_source_page,
)
from scrapers.rss import FeedSpec, collect_registered_feeds, normalize_feed

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Unit</title>
<item><guid>btc-1</guid><title>Bitcoin exchange upgrade</title>
<link>https://example.test/btc?utm_source=x</link>
<pubDate>Wed, 01 Jan 2025 10:00:00 GMT</pubDate></item></channel></rss>"""

HTML = b"""<html><main>
<article><h2><a href="/btc?utm_source=x">Bitcoin rises after ETF inflow</a></h2>
<p>Crypto exchange liquidity improved.</p><time datetime="2025-01-01T10:00:00Z"></time></article>
<article><h2><a href="/shoes">Summer shoe collection launched</a></h2></article>
</main></html>"""


def test_rss_timestamp_and_injected_collection() -> None:
    observed = datetime(2025, 1, 2, tzinfo=UTC)
    feed = FeedSpec("UNIT", "Unit", "https://example.test/rss", "en", ("crypto",), True)
    records = normalize_feed(RSS, feed, observed_at=observed)
    assert records[0].timestamp_quality is TimestampQuality.SOURCE_REPORTED
    assert records[0].historical_coverage is HistoricalCoverage.HISTORICAL


@pytest.mark.asyncio
async def test_rss_collection_has_per_source_status() -> None:
    feed = FeedSpec("UNIT", "Unit", "https://example.test/rss", "en", ("crypto",), True)

    async def fetcher(_: FeedSpec) -> bytes:
        return RSS

    run = await collect_registered_feeds(
        feeds=(feed,),
        fetcher=fetcher,
        observed_at=datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert run.status == "OK"
    assert run.feeds[0].status == "OK"
    assert len(run.records) == 1


def test_web_records_are_relevant_deduplicated_and_auditable() -> None:
    spec = SourceSpec(
        "UNIT",
        "Unit",
        "https://example.test/news",
        "en",
        ("crypto_news",),
    )
    records = parse_source_page(
        HTML,
        spec,
        observed_at=datetime(2025, 1, 2, tzinfo=UTC),
    )
    relevant = enrich_and_filter(records, minimum_relevance_score=0.5)
    assert len(relevant) == 1
    duplicate = deduplicate_intelligence([relevant[0], relevant[0]])
    assert [item.deduplication_status for item in duplicate] == ["UNIQUE", "DUPLICATE"]
    audit = audit_intelligence(duplicate)
    assert audit["invalid_timing_count"] == 0
