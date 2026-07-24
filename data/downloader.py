"""Canonical public crypto spot downloader for Bitvavo, Kraken and CMC."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config.settings import Settings
from core.contracts import DataValidationError, EligibilityStatus, normalize_market
from data.market_data import (
    OHLCV_COLUMNS,
    drop_open_candles,
    load_ohlcv,
    normalize_ohlcv,
    save_ohlcv,
    timeframe_delta,
)
from utils.common import atomic_write_json, utc_iso

BITVAVO_BASE_URL = "https://api.bitvavo.com/v2"
KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
CMC_BASE_URL = "https://pro-api.coinmarketcap.com"
BITVAVO_INTERVALS = frozenset(
    {
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "1W",
        "1M",
    }
)
KRAKEN_INTERVALS = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1_440,
    "1W": 10_080,
}


class DownloadResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str
    timeframe: str
    provider: str
    rows: int
    start: datetime
    end: datetime
    output_path: Path
    resumed: bool


class CandleProvider(Protocol):
    name: str

    async def fetch_candles(
        self,
        market: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame: ...


class AsyncRateLimiter:
    def __init__(self, minimum_interval_seconds: float) -> None:
        self._minimum_interval = max(0.0, minimum_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._minimum_interval:
                await asyncio.sleep(self._minimum_interval - elapsed)
            self._last_call = time.monotonic()


class PublicHttpClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        timeout_seconds: float,
        maximum_retries: int,
        backoff_base_seconds: float,
        minimum_interval_seconds: float = 0.05,
    ) -> None:
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.maximum_retries = maximum_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.rate_limiter = AsyncRateLimiter(minimum_interval_seconds)

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        last_error: BaseException | None = None
        for attempt in range(self.maximum_retries + 1):
            await self.rate_limiter.wait()
            try:
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 429 or response.status >= 500:
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.replace(".", "", 1).isdigit()
                            else self.backoff_base_seconds * 2**attempt
                        )
                        if attempt >= self.maximum_retries:
                            raise ConnectionError(
                                f"public provider request failed with HTTP {response.status}"
                            )
                        await asyncio.sleep(min(30.0, delay))
                        continue
                    if response.status >= 400:
                        raise ConnectionError(
                            f"public provider request rejected with HTTP {response.status}"
                        )
                    return await response.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= self.maximum_retries:
                    break
                await asyncio.sleep(
                    min(30.0, self.backoff_base_seconds * 2**attempt)
                )
        raise ConnectionError("public provider request failed after bounded retries") from last_error


class BitvavoProvider:
    name = "bitvavo"

    def __init__(self, client: PublicHttpClient) -> None:
        self.client = client

    async def fetch_candles(
        self,
        market: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if timeframe not in BITVAVO_INTERVALS:
            raise DataValidationError(f"Bitvavo does not support {timeframe}")
        normalized = normalize_market(market)
        interval_ms = int(timeframe_delta(timeframe).total_seconds() * 1_000)
        start_ms = int(start.timestamp() * 1_000) // interval_ms * interval_ms
        cursor_end = int(end.timestamp() * 1_000) // interval_ms * interval_ms
        rows: list[list[Any]] = []
        seen: set[int] = set()
        while cursor_end >= start_ms:
            query_end = (
                cursor_end
                if cursor_end > start_ms
                else start_ms + interval_ms - 1
            )
            payload = await self.client.get_json(
                f"{BITVAVO_BASE_URL}/{quote(normalized)}/candles",
                params={
                    "interval": timeframe,
                    "limit": 1_440,
                    "start": start_ms,
                    "end": query_end,
                },
                headers={"Accept": "application/json", "User-Agent": "crypto-spot-research/1"},
            )
            if not isinstance(payload, list) or not payload:
                break
            valid = [row for row in payload if isinstance(row, list) and len(row) >= 6]
            if not valid:
                raise DataValidationError("Bitvavo returned an invalid candle payload")
            for row in valid:
                timestamp = int(row[0])
                if timestamp not in seen:
                    rows.append(row[:6])
                    seen.add(timestamp)
            oldest = min(int(row[0]) for row in valid)
            if oldest <= start_ms:
                break
            next_end = oldest - interval_ms
            if next_end >= cursor_end:
                raise DataValidationError("Bitvavo candle pagination did not advance")
            cursor_end = next_end
        if not rows:
            raise DataValidationError(
                f"Bitvavo returned no candles for {normalized} {timeframe}"
            )
        frame = pd.DataFrame(rows, columns=("timestamp", *OHLCV_COLUMNS))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        return normalize_ohlcv(frame, market=normalized)


class KrakenProvider:
    name = "kraken"

    def __init__(self, client: PublicHttpClient) -> None:
        self.client = client

    async def fetch_candles(
        self,
        market: str,
        timeframe: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        try:
            interval = KRAKEN_INTERVALS[timeframe]
        except KeyError as exc:
            raise DataValidationError(f"Kraken does not support {timeframe}") from exc
        normalized = normalize_market(market)
        base, quote_currency = normalized.split("-")
        pair_base = "XBT" if base == "BTC" else base
        payload = await self.client.get_json(
            f"{KRAKEN_BASE_URL}/OHLC",
            params={
                "pair": f"{pair_base}/{quote_currency}",
                "interval": interval,
                "since": int(start.timestamp()),
                "assetVersion": 1,
            },
            headers={"Accept": "application/json", "User-Agent": "crypto-spot-research/1"},
        )
        if not isinstance(payload, dict) or payload.get("error"):
            raise DataValidationError("Kraken returned an error for public OHLC")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise DataValidationError("Kraken returned an invalid OHLC payload")
        rows = next(
            (
                value
                for key, value in result.items()
                if key != "last" and isinstance(value, list)
            ),
            None,
        )
        if not rows:
            raise DataValidationError(f"Kraken returned no candles for {normalized}")
        frame = pd.DataFrame(
            [row[:8] for row in rows],
            columns=(
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "vwap",
                "volume",
                "trade_count",
            ),
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        frame = frame.loc[frame["timestamp"] <= pd.Timestamp(end)]
        return normalize_ohlcv(frame, market=normalized, keep_extra_columns=True)


class CoinMarketCapEnricher:
    """Optional metadata and latest-quote enrichment; never used for execution."""

    name = "coinmarketcap"

    def __init__(self, client: PublicHttpClient, api_key: str | None) -> None:
        self.client = client
        self.api_key = api_key

    async def latest_quotes(
        self,
        markets: list[str],
        *,
        convert: str = "EUR",
    ) -> dict[str, Any]:
        if not self.api_key:
            return {
                "provider": self.name,
                "status": "SKIPPED_NO_API_KEY",
                "data": {},
            }
        symbols = sorted({normalize_market(market).split("-")[0] for market in markets})
        payload = await self.client.get_json(
            f"{CMC_BASE_URL}/v3/cryptocurrency/quotes/latest",
            params={"symbol": ",".join(symbols), "convert": convert},
            headers={
                "Accept": "application/json",
                "X-CMC_PRO_API_KEY": self.api_key,
                "User-Agent": "crypto-spot-research/1",
            },
        )
        if not isinstance(payload, dict) or "data" not in payload:
            raise DataValidationError("CoinMarketCap returned an invalid quote payload")
        return {
            "provider": self.name,
            "status": "OK",
            "observed_at": utc_iso(),
            "data": payload["data"],
        }


def merge_candles(existing: pd.DataFrame | None, downloaded: pd.DataFrame) -> pd.DataFrame:
    fresh = normalize_ohlcv(
        downloaded,
        market=downloaded.attrs.get("market"),
        keep_extra_columns=True,
    )
    if existing is None or existing.empty:
        return fresh
    previous = normalize_ohlcv(
        existing,
        market=existing.attrs.get("market"),
        keep_extra_columns=True,
    )
    overlap = previous.index.intersection(fresh.index)
    if len(overlap):
        left = previous.loc[overlap, OHLCV_COLUMNS].to_numpy(dtype=float)
        right = fresh.loc[overlap, OHLCV_COLUMNS].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-9, atol=1e-12):
            raise DataValidationError("provider candles conflict with stored history")
    combined = pd.concat([previous, fresh.loc[~fresh.index.isin(previous.index)]])
    combined = combined.sort_index()
    combined.attrs.update(fresh.attrs)
    return combined


class CanonicalDownloader:
    def __init__(
        self,
        settings: Settings,
        *,
        providers: dict[str, CandleProvider] | None = None,
    ) -> None:
        self.settings = settings
        self._injected_providers = providers

    async def download_one(
        self,
        *,
        market: str,
        timeframe: str,
        provider: CandleProvider,
        start: datetime,
        end: datetime,
        resume: bool = True,
    ) -> DownloadResult:
        normalized = normalize_market(market)
        eligibility = self.settings.shariah.eligibility(normalized)
        if eligibility.status is not EligibilityStatus.ALLOWED:
            raise PermissionError(
                f"data universe fails closed for {normalized}: {eligibility.reason}"
            )
        target = (
            self.settings.paths.processed_data_dir
            / f"{normalized}_{timeframe}.parquet"
        )
        existing: pd.DataFrame | None = None
        effective_start = start
        resumed = False
        if resume and target.is_file():
            existing = load_ohlcv(target, market=normalized, validate=True)
            effective_start = max(
                start,
                existing.index[-1].to_pydatetime() - timeframe_delta(timeframe),
            )
            resumed = True
        downloaded = await provider.fetch_candles(
            normalized,
            timeframe,
            start=effective_start,
            end=end,
        )
        downloaded = drop_open_candles(
            downloaded,
            timeframe=timeframe,
            now=end,
            close_grace_seconds=(
                self.settings.market_data.candle_close_grace_for(timeframe)
            ),
        )
        combined = merge_candles(existing, downloaded)
        _, manifest = save_ohlcv(
            combined,
            target,
            market=normalized,
            timeframe=timeframe,
            maximum_staleness=self.settings.market_data.maximum_staleness,
            now=end,
        )
        return DownloadResult(
            market=normalized,
            timeframe=timeframe,
            provider=provider.name,
            rows=manifest.rows,
            start=manifest.start,
            end=manifest.end,
            output_path=target,
            resumed=resumed,
        )

    async def download_all(
        self,
        *,
        markets: list[str] | None = None,
        timeframes: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        resume: bool = True,
        provider_preference: list[str] | None = None,
        write_enrichment: bool = True,
    ) -> list[DownloadResult]:
        selected_markets = markets or self.settings.market_data.symbols
        selected_timeframes = timeframes or self.settings.market_data.timeframes
        selected_start = start or self.settings.market_data.start_date or datetime(
            2017, 1, 1, tzinfo=UTC
        )
        selected_end = end or self.settings.market_data.end_date or datetime.now(UTC)
        if selected_start.tzinfo is None or selected_end.tzinfo is None:
            raise ValueError("download bounds must be timezone-aware")
        preferences = provider_preference or self.settings.market_data.providers
        semaphore = asyncio.Semaphore(self.settings.scrapers.maximum_concurrency)

        async with aiohttp.ClientSession() as session:
            client = PublicHttpClient(
                session,
                timeout_seconds=self.settings.scrapers.request_timeout_seconds,
                maximum_retries=self.settings.scrapers.maximum_retries,
                backoff_base_seconds=self.settings.scrapers.backoff_base_seconds,
            )
            providers: dict[str, CandleProvider] = self._injected_providers or {
                "bitvavo": BitvavoProvider(client),
                "kraken": KrakenProvider(client),
            }

            async def one(market: str, timeframe: str) -> DownloadResult:
                errors: list[str] = []
                async with semaphore:
                    for name in preferences:
                        provider = providers.get(name)
                        if provider is None:
                            continue
                        try:
                            return await self.download_one(
                                market=market,
                                timeframe=timeframe,
                                provider=provider,
                                start=selected_start,
                                end=selected_end,
                                resume=resume,
                            )
                        except (ConnectionError, DataValidationError) as exc:
                            errors.append(f"{name}:{type(exc).__name__}")
                    raise ConnectionError(
                        f"all public candle providers failed for "
                        f"{normalize_market(market)} {timeframe}: {errors}"
                    )

            results = await asyncio.gather(
                *(one(market, timeframe) for market in selected_markets for timeframe in selected_timeframes)
            )
            if write_enrichment and "coinmarketcap" in preferences:
                key = self.settings.providers.coinmarketcap_api_key
                enricher = CoinMarketCapEnricher(
                    client,
                    key.get_secret_value() if key else None,
                )
                enrichment = await enricher.latest_quotes(selected_markets)
                atomic_write_json(
                    self.settings.paths.processed_data_dir / "coinmarketcap_latest.json",
                    enrichment,
                )
        return list(results)


def provider_capabilities() -> dict[str, dict[str, Any]]:
    return {
        "bitvavo": {
            "role": "primary_public_candles",
            "timeframes": sorted(BITVAVO_INTERVALS),
            "credentials_required": False,
        },
        "kraken": {
            "role": "secondary_public_candles",
            "timeframes": list(KRAKEN_INTERVALS),
            "credentials_required": False,
            "maximum_recent_rows": 720,
        },
        "coinmarketcap": {
            "role": "optional_enrichment",
            "credentials_required": True,
            "execution": False,
        },
    }


__all__ = [
    "BitvavoProvider",
    "CandleProvider",
    "CanonicalDownloader",
    "CoinMarketCapEnricher",
    "DownloadResult",
    "KrakenProvider",
    "PublicHttpClient",
    "merge_candles",
    "provider_capabilities",
]
