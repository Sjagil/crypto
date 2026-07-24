"""Crypto-relevant intelligence ingestion from broad web sources and RSS."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import aiohttp
import dateparser
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings
from core.contracts import (
    HistoricalCoverage,
    IntelligenceRecord,
    TimestampQuality,
)
from scrapers.rss import RssCollection, collect_registered_feeds
from utils.common import (
    atomic_write_bytes,
    atomic_write_json,
    clean_text,
    sha256_bytes,
    stable_hash,
    utc_now,
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    publisher: str
    url: str
    language: str
    categories: tuple[str, ...]
    crypto_native: bool = False
    allow_playwright: bool = True


DEFAULT_SOURCES = (
    SourceSpec(
        "BLOOMBERG_CRYPTO",
        "Bloomberg",
        "https://www.bloomberg.com/crypto",
        "en",
        ("crypto_news", "macro_liquidity"),
    ),
    SourceSpec(
        "FD_MARKETS",
        "FD",
        "https://fd.nl/financiele-markten",
        "nl",
        ("crypto_news", "macro_liquidity", "regulation"),
    ),
    SourceSpec(
        "FOREXFACTORY_CALENDAR",
        "ForexFactory",
        "https://www.forexfactory.com/calendar",
        "en",
        ("macro_calendar", "interest_rates", "inflation"),
    ),
    SourceSpec(
        "LSEG_INSIGHTS",
        "LSEG",
        "https://www.lseg.com/en/insights",
        "en",
        ("macro_liquidity", "regulation"),
    ),
    SourceSpec(
        "BITVAVO_BLOG",
        "Bitvavo",
        "https://blog.bitvavo.com/",
        "en",
        ("exchange", "crypto_news"),
        True,
    ),
    SourceSpec(
        "KRAKEN_BLOG",
        "Kraken",
        "https://blog.kraken.com/",
        "en",
        ("exchange", "crypto_news"),
        True,
    ),
)

CRYPTO_TERMS = {
    "bitcoin": ("BTC-EUR", "bitcoin"),
    "btc": ("BTC-EUR", "bitcoin"),
    "ethereum": ("ETH-EUR", "ethereum"),
    "ether": ("ETH-EUR", "ethereum"),
    "eth": ("ETH-EUR", "ethereum"),
    "solana": ("SOL-EUR", "solana"),
    "sol": ("SOL-EUR", "solana"),
    "chainlink": ("LINK-EUR", "chainlink"),
}
GENERAL_CRYPTO_TERMS = {
    "crypto",
    "cryptocurrency",
    "blockchain",
    "digital asset",
    "stablecoin",
    "token",
    "on-chain",
    "defi",
    "exchange",
}
MACRO_TERMS = {
    "central bank",
    "ecb",
    "federal reserve",
    "fed",
    "interest rate",
    "rate decision",
    "inflation",
    "liquidity",
    "money supply",
    "dollar",
    "dxy",
    "risk-on",
    "risk-off",
}
RISK_TERMS = {
    "hack",
    "exploit",
    "bridge attack",
    "depeg",
    "outage",
    "bankruptcy",
    "insolvency",
    "delisting",
    "regulation",
    "regulator",
    "custody",
    "token unlock",
    "validator incident",
    "liquidation",
}
POSITIVE_TERMS = {
    "approval",
    "adoption",
    "upgrade",
    "recovery",
    "inflow",
    "growth",
    "surge",
}
NEGATIVE_TERMS = {
    "hack",
    "exploit",
    "outage",
    "depeg",
    "ban",
    "lawsuit",
    "bankruptcy",
    "liquidation",
    "decline",
}


class SourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    publisher: str
    status: str
    fetched_records: int = Field(ge=0)
    relevant_records: int = Field(ge=0)
    error_code: str | None = None
    used_playwright: bool = False
    observed_at: datetime


class IntelligenceRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    records: tuple[IntelligenceRecord, ...]
    sources: tuple[SourceStatus, ...]
    observed_at: datetime
    output_path: Path | None = None
    audit: dict[str, Any]


PageFetcher = Callable[[SourceSpec], Awaitable[bytes]]


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    filtered = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith(("utm_", "fbclid", "gclid"))
    ]
    path = re.sub(r"/+$", "", parts.path) or "/"
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            path,
            urlencode(sorted(filtered)),
            "",
        )
    )


def classify_text(
    title: str,
    summary: str,
    *,
    base_categories: tuple[str, ...] = (),
    crypto_native: bool = False,
) -> tuple[float, tuple[str, ...], tuple[str, ...], tuple[str, ...], float, float]:
    text = f"{title} {summary}".casefold()
    tokens = set(re.findall(r"[a-z0-9-]+", text))
    markets: set[str] = set()
    entities: set[str] = set()
    direct_hits = 0
    for term, (market, entity) in CRYPTO_TERMS.items():
        if (len(term) <= 4 and term in tokens) or (len(term) > 4 and term in text):
            direct_hits += 1
            markets.add(market)
            entities.add(entity)
    general_hits = sum(term in text for term in GENERAL_CRYPTO_TERMS)
    macro_hits = sum(term in text for term in MACRO_TERMS)
    risk_hits = sum(term in text for term in RISK_TERMS)
    score = 0.15 if crypto_native else 0.0
    score += min(0.65, direct_hits * 0.35)
    score += min(0.35, general_hits * 0.12)
    score += min(0.50, macro_hits * 0.16)
    score += min(0.35, risk_hits * 0.12)
    score = min(1.0, score)

    categories = set(base_categories)
    if direct_hits or general_hits:
        categories.add("crypto_market")
    if macro_hits:
        categories.add("macro_liquidity")
    if any(term in text for term in ("regulation", "regulator", "lawsuit", "ban")):
        categories.add("regulation")
    if any(term in text for term in ("hack", "exploit", "bridge attack")):
        categories.add("hack_exploit")
    if "stablecoin" in text or "depeg" in text:
        categories.add("stablecoin_risk")
    if "exchange" in text or "custody" in text:
        categories.add("exchange_risk")
    if "token unlock" in text:
        categories.add("token_unlock")

    positive = sum(term in text for term in POSITIVE_TERMS)
    negative = sum(term in text for term in NEGATIVE_TERMS)
    sentiment = max(-1.0, min(1.0, (positive - negative) / max(1, positive + negative)))
    impact = min(1.0, 0.25 * risk_hits + 0.15 * macro_hits + 0.10 * direct_hits)
    return (
        score,
        tuple(sorted(entities)),
        tuple(sorted(markets)),
        tuple(sorted(categories)),
        sentiment,
        impact,
    )


def _explicit_timestamp(value: str, observed_at: datetime) -> datetime | None:
    text = clean_text(value)
    if not text:
        return None
    has_explicit_zone = bool(
        re.search(r"(?:Z|[+-]\d{2}:?\d{2}|GMT|UTC)\b", text, flags=re.I)
    )
    if not has_explicit_zone:
        return None
    parsed = dateparser.parse(
        text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TO_TIMEZONE": "UTC",
        },
    )
    if (
        parsed is None
        or parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed > observed_at
    ):
        return None
    return parsed.astimezone(UTC)


def _page_published_at(soup: BeautifulSoup, observed_at: datetime) -> datetime | None:
    selectors = (
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "date"}, "content"),
        ("meta", {"name": "pubdate"}, "content"),
        ("time", {"datetime": True}, "datetime"),
    )
    for tag, attrs, attribute in selectors:
        node = soup.find(tag, attrs=attrs)
        if node and node.get(attribute):
            if parsed := _explicit_timestamp(str(node.get(attribute)), observed_at):
                return parsed
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.get_text(strip=True))
        except (TypeError, json.JSONDecodeError):
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("datePublished"):
                if parsed := _explicit_timestamp(
                    str(candidate["datePublished"]), observed_at
                ):
                    return parsed
    return None


def parse_source_page(
    payload: bytes,
    spec: SourceSpec,
    *,
    observed_at: datetime | None = None,
    maximum_records: int = 100,
) -> list[IntelligenceRecord]:
    observed = observed_at or utc_now()
    soup = BeautifulSoup(payload, "lxml")
    page_published = _page_published_at(soup, observed)
    candidates = soup.select("article")
    if not candidates:
        candidates = soup.select("main h1, main h2, main h3, main a")
    records: list[IntelligenceRecord] = []
    seen_urls: set[str] = set()
    for node in candidates[: maximum_records * 3]:
        heading = node.select_one("h1, h2, h3, a") if node.name == "article" else node
        title = clean_text(heading.get_text(" ", strip=True) if heading else "")
        if len(title) < 8:
            continue
        link_node = (
            node.select_one("a[href]")
            if node.name == "article"
            else (node if node.name == "a" else node.find_parent("a"))
        )
        href = link_node.get("href") if link_node else None
        url = canonical_url(urljoin(spec.url, str(href or spec.url)))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        paragraph = node.select_one("p") if node.name == "article" else None
        summary = clean_text(
            paragraph.get_text(" ", strip=True) if paragraph else "",
            maximum_length=4_000,
        )
        time_node = node.select_one("time[datetime]") if node.name == "article" else None
        published = (
            _explicit_timestamp(str(time_node.get("datetime")), observed)
            if time_node
            else page_published
        )
        historical = published is not None
        score, entities, markets, categories, sentiment, impact = classify_text(
            title,
            summary,
            base_categories=spec.categories,
            crypto_native=spec.crypto_native,
        )
        raw_hash = stable_hash(
            {
                "source": spec.source_id,
                "url": url,
                "title": title,
                "summary": summary,
                "published_at": published,
            }
        )
        records.append(
            IntelligenceRecord(
                event_id=stable_hash(
                    {"source": spec.source_id, "url": url, "title": title},
                    length=32,
                ),
                source=spec.publisher,
                url=url,
                title=title,
                summary=summary,
                published_at=published,
                observed_at=observed,
                timestamp_quality=(
                    TimestampQuality.SOURCE_REPORTED
                    if historical
                    else TimestampQuality.OBSERVED_ONLY
                ),
                language=spec.language,
                entities=entities,
                markets=markets,
                categories=categories,
                relevance_score=score,
                sentiment_score=sentiment,
                impact_score=impact,
                deduplication_status="UNIQUE",
                historical_coverage=(
                    HistoricalCoverage.HISTORICAL
                    if historical
                    else HistoricalCoverage.FORWARD_ONLY
                ),
                raw_hash=raw_hash,
            )
        )
        if len(records) >= maximum_records:
            break
    return records


def enrich_and_filter(
    records: Iterable[IntelligenceRecord],
    *,
    minimum_relevance_score: float,
) -> list[IntelligenceRecord]:
    output: list[IntelligenceRecord] = []
    for record in records:
        score, entities, markets, categories, sentiment, impact = classify_text(
            record.title,
            record.summary,
            base_categories=record.categories,
            crypto_native=record.relevance_score >= 0.99,
        )
        if score < minimum_relevance_score:
            continue
        output.append(
            record.model_copy(
                update={
                    "entities": entities,
                    "markets": markets,
                    "categories": categories,
                    "relevance_score": score,
                    "sentiment_score": sentiment,
                    "impact_score": impact,
                }
            )
        )
    return output


def deduplicate_intelligence(
    records: Iterable[IntelligenceRecord],
) -> list[IntelligenceRecord]:
    seen: dict[str, str] = {}
    output: list[IntelligenceRecord] = []
    for record in sorted(records, key=lambda item: (item.usable_at, item.source, item.url)):
        key = stable_hash(
            {"url": canonical_url(record.url), "title": record.title.casefold()},
            length=32,
        )
        previous = seen.get(key)
        if previous is None:
            status = "UNIQUE"
        elif previous == record.raw_hash:
            status = "DUPLICATE"
        else:
            status = "UPDATED"
        seen[key] = record.raw_hash
        output.append(
            record.model_copy(
                update={"event_id": key, "deduplication_status": status}
            )
        )
    return output


async def _robots_allowed(
    session: aiohttp.ClientSession,
    source: SourceSpec,
    user_agent: str,
) -> bool:
    parts = urlsplit(source.url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        async with session.get(robots_url) as response:
            if response.status == 404:
                return True
            if response.status >= 400:
                return False
            text = await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(text.splitlines())
    return parser.can_fetch(user_agent, source.url)


async def _playwright_fetch(source: SourceSpec, timeout_seconds: float) -> bytes:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("PLAYWRIGHT_UNAVAILABLE") from exc
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(
                source.url,
                wait_until="domcontentloaded",
                timeout=int(timeout_seconds * 1_000),
            )
            return (await page.content()).encode("utf-8")
        finally:
            await browser.close()


async def collect_web_sources(
    *,
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    fetcher: PageFetcher | None = None,
    maximum_concurrency: int = 5,
    timeout_seconds: float = 30.0,
    maximum_retries: int = 3,
    playwright_fallback_enabled: bool = True,
    raw_dir: Path | None = None,
    observed_at: datetime | None = None,
    minimum_relevance_score: float = 0.50,
) -> tuple[list[IntelligenceRecord], list[SourceStatus]]:
    observed = observed_at or utc_now()
    semaphore = asyncio.Semaphore(maximum_concurrency)
    records: list[IntelligenceRecord] = []
    statuses: list[SourceStatus] = []
    user_agent = "crypto-spot-research/1"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers={"Accept": "text/html,application/xhtml+xml", "User-Agent": user_agent},
    ) as session:
        async def default_fetch(source: SourceSpec) -> bytes:
            if not await _robots_allowed(session, source, user_agent):
                raise PermissionError("ROBOTS_DISALLOWED_OR_UNVERIFIED")
            last_error: BaseException | None = None
            for attempt in range(maximum_retries + 1):
                try:
                    async with session.get(source.url) as response:
                        if response.status == 429 or response.status >= 500:
                            raise ConnectionError(f"HTTP_{response.status}")
                        if response.status >= 400:
                            raise PermissionError(f"HTTP_{response.status}")
                        return await response.content.read(5_000_001)
                except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                    last_error = exc
                    if attempt >= maximum_retries:
                        break
                    await asyncio.sleep(min(15.0, 0.5 * 2**attempt))
            raise ConnectionError("SOURCE_FETCH_FAILED") from last_error

        selected_fetcher = fetcher or default_fetch

        async def collect(source: SourceSpec) -> None:
            async with semaphore:
                used_playwright = False
                try:
                    try:
                        payload = await selected_fetcher(source)
                    except (ConnectionError, PermissionError):
                        if (
                            fetcher is not None
                            or not playwright_fallback_enabled
                            or not source.allow_playwright
                        ):
                            raise
                        if not await _robots_allowed(session, source, user_agent):
                            raise PermissionError("ROBOTS_DISALLOWED_OR_UNVERIFIED")
                        payload = await _playwright_fetch(source, timeout_seconds)
                        used_playwright = True
                    if len(payload) > 5_000_000:
                        raise ValueError("SOURCE_PAYLOAD_TOO_LARGE")
                    if raw_dir is not None:
                        atomic_write_bytes(
                            raw_dir
                            / f"{source.source_id}_{sha256_bytes(payload)[:16]}.html",
                            payload,
                        )
                    parsed = parse_source_page(payload, source, observed_at=observed)
                    relevant = enrich_and_filter(
                        parsed,
                        minimum_relevance_score=minimum_relevance_score,
                    )
                    records.extend(relevant)
                    statuses.append(
                        SourceStatus(
                            source_id=source.source_id,
                            publisher=source.publisher,
                            status="OK" if relevant else "NO_RELEVANT_RECORDS",
                            fetched_records=len(parsed),
                            relevant_records=len(relevant),
                            used_playwright=used_playwright,
                            observed_at=observed,
                        )
                    )
                except (OSError, ValueError, PermissionError, ConnectionError, RuntimeError):
                    statuses.append(
                        SourceStatus(
                            source_id=source.source_id,
                            publisher=source.publisher,
                            status="BLOCKED",
                            fetched_records=0,
                            relevant_records=0,
                            error_code="SOURCE_FETCH_OR_PARSE_FAILED",
                            used_playwright=used_playwright,
                            observed_at=observed,
                        )
                    )

        await asyncio.gather(*(collect(source) for source in sources))
    return records, sorted(statuses, key=lambda status: status.source_id)


def audit_intelligence(records: Iterable[IntelligenceRecord]) -> dict[str, Any]:
    selected = list(records)
    invalid_timing = 0
    forward_only = 0
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for record in selected:
        source_counts[record.source] += 1
        category_counts.update(record.categories)
        forward_only += record.historical_coverage is HistoricalCoverage.FORWARD_ONLY
        if record.usable_at > record.observed_at:
            invalid_timing += 1
        if (
            record.historical_coverage is HistoricalCoverage.FORWARD_ONLY
            and record.usable_at != record.observed_at
        ):
            invalid_timing += 1
    return {
        "valid": bool(selected) and invalid_timing == 0,
        "record_count": len(selected),
        "invalid_timing_count": invalid_timing,
        "forward_only_count": forward_only,
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
    }


def store_intelligence(
    records: Iterable[IntelligenceRecord],
    path: Path | str,
) -> Path:
    target = Path(path)
    selected = list(records)
    if not selected:
        raise ValueError("refusing to store an empty dataset as valid intelligence")
    rows: list[dict[str, Any]] = []
    for record in selected:
        row = record.model_dump(mode="json")
        row["usable_at"] = record.usable_at.isoformat()
        for name in ("entities", "markets", "categories"):
            row[name] = json.dumps(row[name], ensure_ascii=False)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["usable_at", "source", "event_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=".parquet",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_intelligence(path: Path | str) -> list[IntelligenceRecord]:
    frame = pd.read_parquet(Path(path), engine="pyarrow")
    records: list[IntelligenceRecord] = []
    for row in frame.to_dict(orient="records"):
        row.pop("usable_at", None)
        for name in ("entities", "markets", "categories"):
            row[name] = tuple(json.loads(row[name]))
        records.append(IntelligenceRecord.model_validate(row))
    return records


async def run_intelligence_pipeline(
    settings: Settings,
    *,
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    page_fetcher: PageFetcher | None = None,
    rss_fetcher: Callable[[Any], Awaitable[bytes]] | None = None,
    observed_at: datetime | None = None,
    include_rss: bool | None = None,
) -> IntelligenceRun:
    observed = observed_at or utc_now()
    raw_dir = settings.paths.raw_data_dir / "intelligence"
    web_records, web_statuses = await collect_web_sources(
        sources=sources,
        fetcher=page_fetcher,
        maximum_concurrency=settings.scrapers.maximum_concurrency,
        timeout_seconds=settings.scrapers.request_timeout_seconds,
        maximum_retries=settings.scrapers.maximum_retries,
        playwright_fallback_enabled=settings.scrapers.playwright_fallback_enabled,
        raw_dir=raw_dir,
        observed_at=observed,
        minimum_relevance_score=settings.scrapers.minimum_crypto_relevance_score,
    )
    use_rss = settings.scrapers.rss_enabled if include_rss is None else include_rss
    rss_collection: RssCollection | None = None
    rss_relevant_counts: Counter[str] = Counter()
    if use_rss:
        rss_collection = await collect_registered_feeds(
            fetcher=rss_fetcher,
            maximum_concurrency=settings.scrapers.maximum_concurrency,
            timeout_seconds=settings.scrapers.request_timeout_seconds,
            maximum_retries=settings.scrapers.maximum_retries,
            raw_dir=raw_dir / "rss",
            observed_at=observed,
        )
        rss_relevant = enrich_and_filter(
            rss_collection.records,
            minimum_relevance_score=settings.scrapers.minimum_crypto_relevance_score,
        )
        rss_relevant_counts.update(record.source for record in rss_relevant)
        web_records.extend(rss_relevant)
    records = tuple(deduplicate_intelligence(web_records))
    audit = audit_intelligence(records)
    statuses = list(web_statuses)
    if rss_collection:
        statuses.extend(
            SourceStatus(
                source_id=f"RSS:{feed.feed_id}",
                publisher=feed.publisher,
                status=(
                    feed.status
                    if feed.status == "BLOCKED"
                    else (
                        "OK"
                        if rss_relevant_counts[feed.publisher]
                        else "NO_RELEVANT_RECORDS"
                    )
                ),
                fetched_records=feed.record_count,
                relevant_records=rss_relevant_counts[feed.publisher],
                error_code=feed.error_code,
                observed_at=feed.observed_at,
            )
            for feed in rss_collection.feeds
        )
    successful = sum(status.status in {"OK", "NO_RELEVANT_RECORDS"} for status in statuses)
    status = (
        "FAILED"
        if not records
        else ("OK" if successful == len(statuses) else "PARTIAL")
    )
    output_path: Path | None = None
    if records:
        output_path = store_intelligence(
            records,
            settings.paths.intelligence_dir / "crypto_intelligence.parquet",
        )
    atomic_write_json(
        settings.paths.intelligence_dir / "scraper_status.json",
        {
            "status": status,
            "observed_at": observed,
            "sources": [item.model_dump(mode="json") for item in statuses],
            "audit": audit,
            "output_path": str(output_path) if output_path else None,
        },
    )
    return IntelligenceRun(
        status=status,
        records=records,
        sources=tuple(statuses),
        observed_at=observed,
        output_path=output_path,
        audit=audit,
    )


__all__ = [
    "DEFAULT_SOURCES",
    "IntelligenceRun",
    "SourceSpec",
    "SourceStatus",
    "audit_intelligence",
    "canonical_url",
    "classify_text",
    "collect_web_sources",
    "deduplicate_intelligence",
    "enrich_and_filter",
    "load_intelligence",
    "parse_source_page",
    "run_intelligence_pipeline",
    "store_intelligence",
]
