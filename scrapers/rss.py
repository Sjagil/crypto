"""Bounded RSS/Atom ingestion with explicit publication-time knowability."""

from __future__ import annotations

import asyncio
import calendar
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import dateparser
import feedparser
from pydantic import BaseModel, ConfigDict, Field

from core.contracts import (
    HistoricalCoverage,
    IntelligenceRecord,
    TimestampQuality,
)
from utils.common import atomic_write_bytes, clean_text, sha256_bytes, stable_hash, utc_now


@dataclass(frozen=True)
class FeedSpec:
    feed_id: str
    publisher: str
    url: str
    language: str
    categories: tuple[str, ...]
    crypto_native: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "feed_id": self.feed_id,
            "publisher": self.publisher,
            "language": self.language,
            "categories": list(self.categories),
            "crypto_native": self.crypto_native,
        }


DEFAULT_FEEDS = (
    FeedSpec(
        "KRAKEN_BLOG",
        "Kraken",
        "https://blog.kraken.com/feed",
        "en",
        ("exchange", "crypto_market_structure"),
        True,
    ),
    FeedSpec(
        "BITVAVO_STATUS",
        "Bitvavo Status",
        "https://status.bitvavo.com/history.rss",
        "en",
        ("exchange_risk", "outage"),
        True,
    ),
    FeedSpec(
        "COINDESK",
        "CoinDesk",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "en",
        ("crypto_news",),
        True,
    ),
    FeedSpec(
        "ECB_PRESS",
        "European Central Bank",
        "https://www.ecb.europa.eu/rss/press.html",
        "en",
        ("macro_liquidity", "regulation"),
    ),
    FeedSpec(
        "FED_MONETARY_POLICY",
        "Federal Reserve",
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "en",
        ("macro_liquidity", "interest_rates"),
    ),
)


class FeedStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feed_id: str
    publisher: str
    status: str
    record_count: int = Field(ge=0)
    error_code: str | None = None
    observed_at: datetime


class RssCollection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    records: tuple[IntelligenceRecord, ...]
    feeds: tuple[FeedStatus, ...]
    observed_at: datetime


FeedFetcher = Callable[[FeedSpec], Awaitable[bytes]]


def _published_at(entry: Any, observed_at: datetime) -> tuple[datetime | None, TimestampQuality]:
    parsed_tuple = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_tuple:
        parsed = datetime.fromtimestamp(calendar.timegm(parsed_tuple), tz=UTC)
    else:
        raw = entry.get("published") or entry.get("updated")
        parsed = (
            dateparser.parse(
                str(raw),
                settings={
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "TO_TIMEZONE": "UTC",
                },
            )
            if raw
            else None
        )
        if parsed and (parsed.tzinfo is None or parsed.utcoffset() is None):
            parsed = None
    if parsed is None or parsed > observed_at:
        return None, TimestampQuality.OBSERVED_ONLY
    return parsed.astimezone(UTC), TimestampQuality.SOURCE_REPORTED


def normalize_feed(
    payload: bytes,
    spec: FeedSpec,
    *,
    observed_at: datetime | None = None,
) -> list[IntelligenceRecord]:
    observed = observed_at or utc_now()
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    parsed = feedparser.parse(payload)
    if parsed.bozo and not parsed.entries:
        raise ValueError("RSS or Atom payload is not parseable")
    records: list[IntelligenceRecord] = []
    for entry in parsed.entries:
        title = clean_text(entry.get("title"), maximum_length=500)
        url = clean_text(entry.get("link") or entry.get("id"))
        if not title or not url:
            continue
        summary = clean_text(
            entry.get("summary") or entry.get("description"),
            maximum_length=4_000,
        )
        published_at, quality = _published_at(entry, observed)
        forward_only = quality is TimestampQuality.OBSERVED_ONLY
        entry_identity = clean_text(entry.get("id") or url or title)
        raw_hash = stable_hash(
            {
                "feed": spec.feed_id,
                "identity": entry_identity,
                "title": title,
                "summary": summary,
                "published": published_at,
            }
        )
        records.append(
            IntelligenceRecord(
                event_id=stable_hash(
                    {"feed": spec.feed_id, "identity": entry_identity},
                    length=32,
                ),
                source=spec.publisher,
                url=url,
                title=title,
                summary=summary,
                published_at=None if forward_only else published_at,
                observed_at=observed,
                timestamp_quality=quality,
                language=spec.language,
                categories=spec.categories,
                relevance_score=1.0 if spec.crypto_native else 0.5,
                sentiment_score=0.0,
                impact_score=0.0,
                deduplication_status="UNIQUE",
                historical_coverage=(
                    HistoricalCoverage.FORWARD_ONLY
                    if forward_only
                    else HistoricalCoverage.HISTORICAL
                ),
                raw_hash=raw_hash,
            )
        )
    return records


def deduplicate(records: Iterable[IntelligenceRecord]) -> list[IntelligenceRecord]:
    seen: dict[str, str] = {}
    output: list[IntelligenceRecord] = []
    for record in records:
        previous_hash = seen.get(record.event_id)
        if previous_hash is None:
            status = "UNIQUE"
        elif previous_hash == record.raw_hash:
            status = "DUPLICATE"
        else:
            status = "UPDATED"
        seen[record.event_id] = record.raw_hash
        output.append(record.model_copy(update={"deduplication_status": status}))
    return output


async def collect_registered_feeds(
    *,
    feeds: tuple[FeedSpec, ...] = DEFAULT_FEEDS,
    fetcher: FeedFetcher | None = None,
    maximum_concurrency: int = 5,
    timeout_seconds: float = 30.0,
    maximum_retries: int = 3,
    raw_dir: Path | None = None,
    observed_at: datetime | None = None,
) -> RssCollection:
    observed = observed_at or utc_now()
    semaphore = asyncio.Semaphore(maximum_concurrency)
    records: list[IntelligenceRecord] = []
    statuses: list[FeedStatus] = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        headers={
            "Accept": "application/rss+xml, application/atom+xml, text/xml",
            "User-Agent": "crypto-spot-research/1",
        },
    ) as session:
        async def default_fetch(spec: FeedSpec) -> bytes:
            last_error: BaseException | None = None
            for attempt in range(maximum_retries + 1):
                try:
                    async with session.get(spec.url) as response:
                        if response.status == 429 or response.status >= 500:
                            raise ConnectionError(f"HTTP_{response.status}")
                        if response.status >= 400:
                            raise PermissionError(f"HTTP_{response.status}")
                        return await response.content.read(2_000_001)
                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                    last_error = exc
                    if attempt >= maximum_retries:
                        break
                    await asyncio.sleep(min(15.0, 0.5 * 2**attempt))
            raise ConnectionError("RSS_FETCH_FAILED") from last_error

        selected_fetcher = fetcher or default_fetch

        async def collect(spec: FeedSpec) -> None:
            async with semaphore:
                try:
                    payload = await selected_fetcher(spec)
                    if len(payload) > 2_000_000:
                        raise ValueError("RSS_PAYLOAD_TOO_LARGE")
                    if raw_dir is not None:
                        atomic_write_bytes(
                            raw_dir / f"{spec.feed_id}_{sha256_bytes(payload)[:16]}.xml",
                            payload,
                        )
                    normalized = normalize_feed(payload, spec, observed_at=observed)
                    records.extend(normalized)
                    statuses.append(
                        FeedStatus(
                            feed_id=spec.feed_id,
                            publisher=spec.publisher,
                            status="OK" if normalized else "EMPTY",
                            record_count=len(normalized),
                            observed_at=observed,
                        )
                    )
                except (OSError, ValueError, PermissionError, ConnectionError):
                    statuses.append(
                        FeedStatus(
                            feed_id=spec.feed_id,
                            publisher=spec.publisher,
                            status="BLOCKED",
                            record_count=0,
                            error_code="RSS_FETCH_OR_PARSE_FAILED",
                            observed_at=observed,
                        )
                    )

        await asyncio.gather(*(collect(spec) for spec in feeds))
    normalized_records = tuple(deduplicate(records))
    successful = sum(status.status == "OK" for status in statuses)
    overall = "OK" if successful == len(feeds) else ("PARTIAL" if successful else "FAILED")
    return RssCollection(
        status=overall,
        records=normalized_records,
        feeds=tuple(sorted(statuses, key=lambda item: item.feed_id)),
        observed_at=observed,
    )


def audit_entries(records: Iterable[IntelligenceRecord]) -> dict[str, Any]:
    selected = list(records)
    statuses = Counter(record.deduplication_status for record in selected)
    forward_only = sum(
        record.historical_coverage is HistoricalCoverage.FORWARD_ONLY
        for record in selected
    )
    return {
        "status": "OK" if selected else "EMPTY",
        "entry_count": len(selected),
        "deduplication_status_counts": dict(statuses),
        "forward_only_count": forward_only,
        "publication_dates_reconstructed": False,
    }


__all__ = [
    "DEFAULT_FEEDS",
    "FeedSpec",
    "FeedStatus",
    "RssCollection",
    "audit_entries",
    "collect_registered_feeds",
    "deduplicate",
    "normalize_feed",
]
