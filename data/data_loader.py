"""Unified, point-in-time historical data access for public providers."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import os
import shutil
import socket
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import (
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_SECONDS,
    Settings,
    normalize_timeframe,
)
from core.contracts import (
    DataValidationError,
    HistoryProfile,
    NormalizedDataRecord,
    ProviderStatus,
    normalize_market,
)
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_text,
    stable_hash,
    stable_json,
    utc_iso,
    utc_now,
)

JsonRequester = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str] | None],
    Awaitable[Any],
]

BITVAVO_REST = "https://api.bitvavo.com/v2"
KRAKEN_REST = "https://api.kraken.com/0/public"
MEXC_REST = "https://api.mexc.com/api/v3"
CMC_REST = "https://pro-api.coinmarketcap.com/v1"
EODHD_REST = "https://eodhd.com/api"
FRED_REST = "https://api.stlouisfed.org/fred"
SEC_REST = "https://data.sec.gov"
ALTERNATIVE_ME_REST = "https://api.alternative.me"
DEFILLAMA_STABLECOINS_REST = "https://stablecoins.llama.fi"
DEFILLAMA_REST = "https://api.llama.fi"
DERIBIT_REST = "https://www.deribit.com/api/v2/public"
LOGGER = logging.getLogger("crypto.data_loader")
BITVAVO_TIMEZONE = ZoneInfo("Europe/Amsterdam")


def _bitvavo_week_boundary_ms(value_ms: int) -> int:
    """Floor to Bitvavo's Monday 00:00 Europe/Amsterdam request boundary."""
    instant = datetime.fromtimestamp(max(0, value_ms) / 1_000, tz=UTC)
    local = instant.astimezone(BITVAVO_TIMEZONE)
    monday = (local - timedelta(days=local.weekday())).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return int(monday.astimezone(UTC).timestamp() * 1_000)

CAPABILITY_RETRIEVED_AT = "2026-07-24"
PROVIDER_NATIVE_TIMEFRAMES: dict[str, tuple[str, ...]] = {
    "bitvavo": (
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
        "1mo",
    ),
    "kraken": ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1W"),
    "mexc": ("1m", "5m", "15m", "30m", "1h", "4h", "8h", "1d", "1W", "1mo"),
}
RESAMPLE_SOURCE: dict[str, dict[str, str]] = {
    "bitvavo": {
        "3m": "1m",
        "3h": "1h",
        "2d": "1d",
        "3d": "1d",
        "1mo": "1d",
    },
    "kraken": {
        "3m": "1m",
        "2h": "1h",
        "3h": "1h",
        "6h": "1h",
        "8h": "1h",
        "12h": "1h",
        "2d": "1d",
        "3d": "1d",
        "1mo": "1d",
    },
    "mexc": {
        "3m": "1m",
        "2h": "1h",
        "3h": "1h",
        "6h": "1h",
        "12h": "1h",
        "2d": "1d",
        "3d": "1d",
        "1mo": "1d",
    },
}

HISTORY_TARGETS: dict[HistoryProfile, dict[str, timedelta | None]] = {
    HistoryProfile.SMOKE: {
        timeframe: timedelta(seconds=TIMEFRAME_SECONDS[timeframe] * 500)
        for timeframe in SUPPORTED_TIMEFRAMES
    },
    HistoryProfile.STANDARD: {
        "1m": timedelta(days=180),
        "3m": timedelta(days=180),
        "5m": timedelta(days=365),
        "15m": timedelta(days=365 * 2),
        "30m": timedelta(days=365 * 2),
        "1h": timedelta(days=365 * 3),
        "2h": timedelta(days=365 * 4),
        "3h": timedelta(days=365 * 4),
        "4h": timedelta(days=365 * 4),
        "6h": timedelta(days=365 * 4),
        "8h": timedelta(days=365 * 4),
        "12h": timedelta(days=365 * 4),
        "1d": None,
        "2d": None,
        "3d": None,
        "1W": None,
        "1mo": None,
    },
    HistoryProfile.DEEP: {timeframe: None for timeframe in SUPPORTED_TIMEFRAMES},
    HistoryProfile.MAXIMUM: {timeframe: None for timeframe in SUPPORTED_TIMEFRAMES},
}


def _epoch(value: Any, *, milliseconds: bool = True) -> datetime:
    numeric = float(value)
    if milliseconds:
        numeric /= 1_000.0
    return datetime.fromtimestamp(numeric, tz=UTC)


def _iso(value: Any) -> datetime:
    text = str(value).strip()
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _market(value: str, quote: str = "EUR") -> str:
    cleaned = value.upper().replace("/", "-").replace("_", "-")
    if "-" not in cleaned and cleaned.endswith(quote):
        cleaned = f"{cleaned[: -len(quote)]}-{quote}"
    aliases = {"XBT": "BTC", "XDG": "DOGE"}
    parts = cleaned.split("-")
    if len(parts) == 2:
        cleaned = f"{aliases.get(parts[0], parts[0])}-{aliases.get(parts[1], parts[1])}"
    return normalize_market(cleaned)


def _raw_hash(value: Any) -> str:
    return sha256_text(stable_json(value))


class PublicHttpClient:
    """Small retrying client. It never accepts private exchange credentials."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        maximum_retries: int = 3,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.maximum_retries = maximum_retries
        self.session = session

    async def request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        owned = self.session is None
        session = self.session or aiohttp.ClientSession()
        try:
            for attempt in range(1, self.maximum_retries + 1):
                try:
                    async with session.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                    ) as response:
                        if response.status == 429 or response.status >= 500:
                            if attempt < self.maximum_retries:
                                retry_after = float(response.headers.get("Retry-After", 0))
                                await asyncio.sleep(max(retry_after, 0.25 * 2 ** (attempt - 1)))
                                continue
                        response.raise_for_status()
                        return await response.json(content_type=None)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt == self.maximum_retries:
                        raise
                    await asyncio.sleep(0.25 * 2 ** (attempt - 1))
        finally:
            if owned:
                await session.close()
        raise RuntimeError("public request exhausted retries")


class ProviderAdapter:
    name = "provider"

    def __init__(self, requester: JsonRequester) -> None:
        self.request = requester

    def symbol(self, market: str) -> str:
        return normalize_market(market)

    def _record(
        self,
        *,
        source_symbol: str,
        market: str,
        timestamp: datetime,
        observed_at: datetime,
        kind: str,
        run_id: str,
        raw: Any,
        values: dict[str, Any],
        timeframe: str | None = None,
        available_at: datetime | None = None,
        closed: bool | None = None,
    ) -> NormalizedDataRecord:
        return NormalizedDataRecord(
            provider=self.name,
            source_symbol=source_symbol,
            canonical_market=market,
            timestamp=timestamp,
            observed_at=observed_at,
            available_at=available_at,
            data_kind=kind,
            timeframe=timeframe,
            closed=closed,
            retrieval_run_id=run_id,
            raw_hash=_raw_hash(raw),
            raw_payload=raw,
            values=values,
        )


class BitvavoAdapter(ProviderAdapter):
    name = "bitvavo"

    async def ohlcv(
        self,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        rows: list[Any] = []
        seen: set[int] = set()
        interval_ms = TIMEFRAME_SECONDS[timeframe] * 1_000
        provider_interval = (
            "1M"
            if timeframe == "1mo"
            else timeframe
        )
        start_ms = int(start.timestamp() * 1_000) // interval_ms * interval_ms
        end_ms = int(end.timestamp() * 1_000)
        # Fetch closed candles only.  Advancing an in-progress weekly cursor
        # to its theoretical interval end can put `end` several days in the
        # future, which Bitvavo rejects with HTTP 400.  The same rule also
        # prevents an incomplete candle from entering lower-timeframe data.
        observed_ms = int(observed.timestamp() * 1_000)
        if timeframe == "1W":
            # Bitvavo validates weekly request boundaries at Monday 00:00 in
            # Europe/Amsterdam (Sunday 22:00 UTC during summer time), while its
            # returned candle timestamp is normalized to Monday UTC.
            start_ms = _bitvavo_week_boundary_ms(start_ms)
            selected_end_ms = min(end_ms, observed_ms)
            cursor_boundary = _bitvavo_week_boundary_ms(selected_end_ms)
            cursor_end = cursor_boundary - interval_ms
        else:
            cursor_end = min(end_ms, observed_ms) // interval_ms * interval_ms
        if cursor_end + interval_ms > observed_ms:
            cursor_end -= interval_ms
        while cursor_end >= start_ms:
            # Bitvavo requires `start` to align with a candle open and `end`
            # to align exactly with an interval end.  The previous `- 1 ms`
            # boundary was rejected for single-candle resume windows.
            query_end = cursor_end + interval_ms
            page = await self.request(
                "GET",
                f"{BITVAVO_REST}/{symbol}/candles",
                {
                    "interval": provider_interval,
                    "start": start_ms,
                    "end": query_end,
                    "limit": 1_440,
                },
                None,
            )
            if not page:
                break
            valid = [item for item in page if isinstance(item, list) and len(item) >= 6]
            if not valid:
                break
            for item in valid:
                timestamp = int(item[0])
                if start_ms <= timestamp <= end_ms and timestamp not in seen:
                    rows.append(item)
                    seen.add(timestamp)
            oldest = min(int(item[0]) for item in valid)
            if oldest <= start_ms:
                break
            next_end = oldest - interval_ms
            if next_end >= cursor_end:
                raise ValueError("Bitvavo candle pagination did not advance")
            cursor_end = next_end
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row[0]),
                observed_at=observed,
                kind="ohlcv",
                run_id=run_id,
                raw=row,
                timeframe=timeframe,
                closed=_epoch(row[0]) + timedelta(seconds=TIMEFRAME_SECONDS[timeframe]) <= observed,
                values={
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                },
            )
            for row in rows
        ]

    async def trades(self, market: str, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        rows = await self.request("GET", f"{BITVAVO_REST}/{symbol}/trades", {}, None)
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row["timestamp"]),
                observed_at=observed,
                kind="trade",
                run_id=run_id,
                raw=row,
                values={
                    "trade_id": row.get("id"),
                    "price": row.get("price"),
                    "quantity": row.get("amount"),
                    "side": row.get("side"),
                },
            )
            for row in rows
        ]

    async def ticker(self, market: str, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request("GET", f"{BITVAVO_REST}/ticker/24h", {"market": symbol}, None)
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=_epoch(raw.get("timestamp", observed.timestamp() * 1_000)),
            observed_at=observed,
            kind="ticker",
            run_id=run_id,
            raw=raw,
            values={
                "last_price": raw.get("last"),
                "best_bid": raw.get("bid"),
                "best_ask": raw.get("ask"),
                "volume_24h": raw.get("volume"),
            },
        )

    async def orderbook(self, market: str, depth: int, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request("GET", f"{BITVAVO_REST}/{symbol}/book", {"depth": depth}, None)
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=observed,
            observed_at=observed,
            kind="orderbook_snapshot",
            run_id=run_id,
            raw=raw,
            values={
                "bids": raw.get("bids", []),
                "asks": raw.get("asks", []),
                "sequence": raw.get("nonce"),
            },
        )

    async def metadata(self, market: str | None, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        raw = await self.request("GET", f"{BITVAVO_REST}/markets", {}, None)
        rows = [row for row in raw if market is None or row.get("market") == market]
        return [
            self._record(
                source_symbol=row["market"],
                market=_market(row["market"]),
                timestamp=observed,
                observed_at=observed,
                kind="market_metadata",
                run_id=run_id,
                raw=row,
                values=row,
            )
            for row in rows
        ]


class KrakenAdapter(ProviderAdapter):
    name = "kraken"

    def symbol(self, market: str) -> str:
        base, quote = normalize_market(market).split("-")
        return f"{'XBT' if base == 'BTC' else base}{quote}"

    @staticmethod
    def _result(payload: Mapping[str, Any]) -> tuple[str, list[Any]]:
        errors = [str(item) for item in payload.get("error", []) if item]
        if errors:
            message = "; ".join(errors)
            if any(marker in message.casefold() for marker in ("rate limit", "too many requests")):
                raise RuntimeError(f"BLOCKED_RATE_LIMIT:{message}")
            raise ValueError(f"KRAKEN_PROVIDER_ERROR:{message}")
        result = payload.get("result", {})
        key = next((key for key in result if key != "last"), None)
        if key is None:
            raise ValueError("KRAKEN_EMPTY_RESULT")
        return key, result[key]

    async def ohlcv(
        self,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        minutes = TIMEFRAME_SECONDS[timeframe] // 60
        cursor = int(start.timestamp())
        rows: list[Any] = []
        while cursor < int(end.timestamp()):
            for retry in range(4):
                raw = await self.request(
                    "GET",
                    f"{KRAKEN_REST}/OHLC",
                    {
                        "pair": symbol,
                        "interval": minutes,
                        "since": cursor,
                    },
                    None,
                )
                errors = [str(item) for item in raw.get("error", []) if item]
                rate_limited = any(
                    any(marker in item.casefold() for marker in ("rate limit", "too many requests"))
                    for item in errors
                )
                if not rate_limited or retry == 3:
                    break
                await asyncio.sleep(min(8.0, 0.5 * 2**retry))
            _, page = self._result(raw)
            selected = [row for row in page if int(row[0]) <= int(end.timestamp())]
            rows.extend(selected)
            next_cursor = int(raw.get("result", {}).get("last", cursor))
            if not page or next_cursor <= cursor:
                break
            cursor = next_cursor
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row[0], milliseconds=False),
                observed_at=observed,
                kind="ohlcv",
                run_id=run_id,
                raw=row,
                timeframe=timeframe,
                closed=_epoch(row[0], milliseconds=False)
                + timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
                <= observed,
                values={
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "vwap": row[5],
                    "volume": row[6],
                    "trade_count": row[7],
                },
            )
            for row in rows
        ]

    async def trades(self, market: str, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request("GET", f"{KRAKEN_REST}/Trades", {"pair": symbol}, None)
        _, rows = self._result(raw)
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row[2], milliseconds=False),
                observed_at=observed,
                kind="trade",
                run_id=run_id,
                raw=row,
                values={"price": row[0], "quantity": row[1], "side": row[3]},
            )
            for row in rows
        ]

    async def ticker(self, market: str, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request("GET", f"{KRAKEN_REST}/Ticker", {"pair": symbol}, None)
        _, row = self._result(raw)
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=observed,
            observed_at=observed,
            kind="ticker",
            run_id=run_id,
            raw=raw,
            values={
                "last_price": row["c"][0],
                "best_bid": row["b"][0],
                "best_ask": row["a"][0],
                "volume_24h": row["v"][1],
            },
        )

    async def orderbook(self, market: str, depth: int, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request(
            "GET", f"{KRAKEN_REST}/Depth", {"pair": symbol, "count": depth}, None
        )
        _, book = self._result(raw)
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=observed,
            observed_at=observed,
            kind="orderbook_snapshot",
            run_id=run_id,
            raw=raw,
            values={"bids": book.get("bids", []), "asks": book.get("asks", [])},
        )

    async def metadata(self, market: str | None, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        raw = await self.request("GET", f"{KRAKEN_REST}/AssetPairs", {}, None)
        records: list[NormalizedDataRecord] = []
        for symbol, row in raw.get("result", {}).items():
            alt = row.get("wsname") or row.get("altname") or symbol
            try:
                canonical = _market(alt)
            except ValueError:
                continue
            if market and canonical != normalize_market(market):
                continue
            records.append(
                self._record(
                    source_symbol=symbol,
                    market=canonical,
                    timestamp=observed,
                    observed_at=observed,
                    kind="market_metadata",
                    run_id=run_id,
                    raw=row,
                    values=row,
                )
            )
        return records


class MexcAdapter(ProviderAdapter):
    name = "mexc"

    def symbol(self, market: str) -> str:
        return normalize_market(market).replace("-", "")

    async def ohlcv(
        self,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        cursor = int(start.timestamp() * 1_000)
        end_ms = int(end.timestamp() * 1_000)
        rows: list[Any] = []
        while cursor < end_ms:
            mexc_interval = {
                "1h": "60m",
                "1mo": "1M",
            }.get(timeframe, timeframe)
            page = await self.request(
                "GET",
                f"{MEXC_REST}/klines",
                {
                    "symbol": symbol,
                    "interval": mexc_interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1_000,
                },
                None,
            )
            if not page:
                break
            rows.extend(page)
            next_cursor = max(int(row[0]) for row in page) + TIMEFRAME_SECONDS[timeframe] * 1_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row[0]),
                observed_at=observed,
                kind="ohlcv",
                run_id=run_id,
                raw=row,
                timeframe=timeframe,
                closed=int(row[6]) <= int(observed.timestamp() * 1_000),
                values={
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                    "quote_volume": row[7],
                    "trade_count": row[8] if len(row) > 8 else None,
                },
            )
            for row in rows
        ]

    async def trades(self, market: str, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        symbol = self.symbol(market)
        rows = await self.request(
            "GET", f"{MEXC_REST}/trades", {"symbol": symbol, "limit": 1_000}, None
        )
        return [
            self._record(
                source_symbol=symbol,
                market=market,
                timestamp=_epoch(row["time"]),
                observed_at=observed,
                kind="trade",
                run_id=run_id,
                raw=row,
                values={
                    "trade_id": row.get("id"),
                    "price": row.get("price"),
                    "quantity": row.get("qty"),
                    "side": "sell" if row.get("isBuyerMaker") else "buy",
                },
            )
            for row in rows
        ]

    async def ticker(self, market: str, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request("GET", f"{MEXC_REST}/ticker/24hr", {"symbol": symbol}, None)
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=_epoch(raw.get("closeTime", observed.timestamp() * 1_000)),
            observed_at=observed,
            kind="ticker",
            run_id=run_id,
            raw=raw,
            values={
                "last_price": raw.get("lastPrice"),
                "best_bid": raw.get("bidPrice"),
                "best_ask": raw.get("askPrice"),
                "volume_24h": raw.get("volume"),
            },
        )

    async def orderbook(self, market: str, depth: int, run_id: str) -> NormalizedDataRecord:
        observed = utc_now()
        symbol = self.symbol(market)
        raw = await self.request(
            "GET", f"{MEXC_REST}/depth", {"symbol": symbol, "limit": depth}, None
        )
        return self._record(
            source_symbol=symbol,
            market=market,
            timestamp=observed,
            observed_at=observed,
            kind="orderbook_snapshot",
            run_id=run_id,
            raw=raw,
            values={
                "bids": raw.get("bids", []),
                "asks": raw.get("asks", []),
                "sequence": raw.get("lastUpdateId"),
            },
        )

    async def metadata(self, market: str | None, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        raw = await self.request("GET", f"{MEXC_REST}/exchangeInfo", {}, None)
        records: list[NormalizedDataRecord] = []
        for row in raw.get("symbols", []):
            canonical = f"{row.get('baseAsset')}-{row.get('quoteAsset')}"
            try:
                canonical = normalize_market(canonical)
            except ValueError:
                continue
            if market and canonical != normalize_market(market):
                continue
            records.append(
                self._record(
                    source_symbol=row["symbol"],
                    market=canonical,
                    timestamp=observed,
                    observed_at=observed,
                    kind="market_metadata",
                    run_id=run_id,
                    raw=row,
                    values=row,
                )
            )
        return records


class DataLoader:
    """Provider registry, reconciliation, cache and persistence facade."""

    def __init__(
        self,
        settings: Settings,
        *,
        requester: JsonRequester | None = None,
        database: Any | None = None,
    ) -> None:
        self.settings = settings
        client = PublicHttpClient(
            timeout_seconds=settings.market_data.request_timeout_seconds,
            maximum_retries=settings.market_data.maximum_retries,
        )
        request = requester or client.request
        self.adapters: dict[str, ProviderAdapter] = {
            "bitvavo": BitvavoAdapter(request),
            "kraken": KrakenAdapter(request),
            "mexc": MexcAdapter(request),
        }
        self.request = request
        self.database = database
        self._health: dict[str, dict[str, Any]] = {
            name: {
                "status": ProviderStatus.PARTIAL.value,
                "reason_code": "NOT_PROBED_IN_CURRENT_PROCESS",
                "requests": 0,
                "errors": 0,
            }
            for name in self.list_providers()
        }

    def list_providers(self) -> tuple[str, ...]:
        return (
            "bitvavo",
            "kraken",
            "mexc",
            "coinmarketcap",
            "eodhd",
            "sec",
            "fred",
            "alternative_me",
            "defillama",
            "deribit",
            "coinglass",
            "glassnode",
            "cryptoquant",
            "coingecko",
            "polygon",
        )

    def provider_status(self, provider: str | None = None) -> dict[str, Any]:
        if provider is not None:
            return dict(self._health[provider.lower()])
        return {name: dict(status) for name, status in self._health.items()}

    def _credential_configured(self, provider: str) -> bool:
        configured = {
            "bitvavo": True,
            "kraken": True,
            "mexc": True,
            "coinmarketcap": self.settings.providers.coinmarketcap_api_key,
            "eodhd": self.settings.providers.eodhd_api_key,
            "fred": self.settings.providers.fred_api_key,
            "sec": self.settings.providers.sec_user_agent,
            "alternative_me": True,
            "defillama": True,
            "deribit": True,
            "coinglass": self.settings.providers.coinglass_api_key,
            "glassnode": self.settings.providers.glassnode_api_key,
            "cryptoquant": self.settings.providers.cryptoquant_api_key,
            "coingecko": self.settings.providers.coingecko_api_key,
            "polygon": self.settings.providers.polygon_api_key,
        }
        return bool(configured[provider])

    @staticmethod
    def _capability_templates() -> dict[str, dict[str, Any]]:
        common_exchange = {
            "provider_category": "CRYPTO_SPOT_DATA",
            "trades_endpoint": True,
            "ticker_endpoint": True,
            "order_book_endpoint": True,
            "websocket_support": True,
            "funding_support": False,
            "open_interest_support": False,
            "liquidation_support": False,
            "options_chain_support": False,
            "macro_support": False,
            "revision_or_vintage_support": False,
        }
        return {
            "bitvavo": {
                **common_exchange,
                "authentication_requirement": "PUBLIC_ENDPOINTS_NO_AUTH",
                "supported_quote_currencies": ["EUR"],
                "native_candle_intervals": list(PROVIDER_NATIVE_TIMEFRAMES["bitvavo"]),
                "maximum_rows_per_request": 1_440,
                "historical_pagination_model": "BACKWARD_START_END",
                "rate_limit_information": "WEIGHT_BASED; endpoint getCandles weight 1",
                "plan_limitations": [],
                "documentation": "https://docs.bitvavo.com/docs/rest-api/get-candlestick-data/",
            },
            "kraken": {
                **common_exchange,
                "authentication_requirement": "PUBLIC_ENDPOINTS_NO_AUTH",
                "supported_quote_currencies": ["EUR", "USD", "USDT"],
                "native_candle_intervals": list(PROVIDER_NATIVE_TIMEFRAMES["kraken"]),
                "maximum_rows_per_request": 720,
                "historical_pagination_model": "RECENT_WINDOW_SINCE_CURSOR",
                "rate_limit_information": "PUBLIC REST CALL COUNTER",
                "plan_limitations": [
                    "REST OHLC returns only the most recent provider window; deeper history accrues incrementally"
                ],
                "documentation": "https://docs.kraken.com/api/docs/rest-api/get-ohlc-data",
            },
            "mexc": {
                **common_exchange,
                "authentication_requirement": "PUBLIC_ENDPOINTS_NO_AUTH",
                "supported_quote_currencies": ["USDT", "USDC", "BTC", "ETH", "EUR"],
                "native_candle_intervals": list(PROVIDER_NATIVE_TIMEFRAMES["mexc"]),
                "maximum_rows_per_request": 1_000,
                "historical_pagination_model": "FORWARD_START_END",
                "rate_limit_information": "500 requests per endpoint per 10 seconds by IP",
                "plan_limitations": [
                    "native quote retained; USDT series are not execution-ready EUR candles"
                ],
                "funding_support": True,
                "open_interest_support": True,
                "liquidation_support": "PARTIAL_NO_PUBLIC_HISTORY",
                "documentation": "https://mexcdevelop.github.io/apidocs/spot_v3_en/",
            },
            "coinmarketcap": {
                "provider_category": "UNIVERSE_AND_GLOBAL_CRYPTO_CONTEXT",
                "authentication_requirement": "API_KEY",
                "supported_quote_currencies": ["EUR", "USD"],
                "native_candle_intervals": [],
                "maximum_rows_per_request": 5_000,
                "historical_pagination_model": "PLAN_DEPENDENT_TIME_PAGINATION",
                "trades_endpoint": False,
                "ticker_endpoint": False,
                "order_book_endpoint": False,
                "websocket_support": False,
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": False,
                "macro_support": True,
                "revision_or_vintage_support": False,
                "rate_limit_information": "CREDIT_AND_PLAN_BASED",
                "plan_limitations": ["historical endpoints depend on subscription plan"],
                "documentation": "https://coinmarketcap.com/api/documentation/v1/",
            },
            "eodhd": {
                "provider_category": "EXTERNAL_MARKET_AND_MACRO_CONTEXT",
                "authentication_requirement": "API_KEY",
                "supported_quote_currencies": ["USD"],
                "native_candle_intervals": [],
                "maximum_rows_per_request": 1_000,
                "historical_pagination_model": "DATE_RANGE_AND_OFFSET",
                "trades_endpoint": False,
                "ticker_endpoint": True,
                "order_book_endpoint": False,
                "websocket_support": "PLAN_DEPENDENT",
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": "MARKETPLACE_PLAN_DEPENDENT",
                "macro_support": True,
                "revision_or_vintage_support": "PROVIDER_REVISION_FIELDS_WHERE_AVAILABLE",
                "rate_limit_information": "DAILY_API_CREDITS_AND_MINUTE_LIMIT",
                "plan_limitations": ["economic events require an eligible plan"],
                "documentation": "https://eodhd.com/financial-apis/economic-events-data-api",
            },
            "fred": {
                "provider_category": "POINT_IN_TIME_MACRO",
                "authentication_requirement": "API_KEY",
                "supported_quote_currencies": [],
                "native_candle_intervals": [],
                "maximum_rows_per_request": 100_000,
                "historical_pagination_model": "OFFSET_AND_REALTIME_PERIOD",
                "trades_endpoint": False,
                "ticker_endpoint": False,
                "order_book_endpoint": False,
                "websocket_support": False,
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": False,
                "macro_support": True,
                "revision_or_vintage_support": True,
                "rate_limit_information": "DOCUMENTED_API_KEY_SERVICE_LIMITS",
                "plan_limitations": [],
                "documentation": "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
            },
            "sec": {
                "provider_category": "REGULATORY_CONTEXT",
                "authentication_requirement": "CONFIGURED_USER_AGENT",
                "supported_quote_currencies": [],
                "native_candle_intervals": [],
                "maximum_rows_per_request": 1_000,
                "historical_pagination_model": "SUBMISSIONS_PLUSHISTORICAL_FILES",
                "trades_endpoint": False,
                "ticker_endpoint": False,
                "order_book_endpoint": False,
                "websocket_support": False,
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": False,
                "macro_support": True,
                "revision_or_vintage_support": False,
                "rate_limit_information": "POLITE_ACCESS_AND_DECLARED_USER_AGENT_REQUIRED",
                "plan_limitations": [],
                "documentation": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
            },
            "alternative_me": {
                "provider_category": "CRYPTO_SENTIMENT_CONTEXT",
                "authentication_requirement": "NONE",
                "supported_quote_currencies": [],
                "native_candle_intervals": [],
                "maximum_rows_per_request": None,
                "historical_pagination_model": "LIMIT_ZERO_RETURNS_AVAILABLE_HISTORY",
                "trades_endpoint": False,
                "ticker_endpoint": False,
                "order_book_endpoint": False,
                "websocket_support": False,
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": False,
                "macro_support": True,
                "revision_or_vintage_support": False,
                "rate_limit_information": "PUBLIC_FAIR_USE",
                "plan_limitations": ["source attribution required"],
                "documentation": "https://alternative.me/crypto/fear-and-greed-index/",
            },
            "defillama": {
                "provider_category": "DEFI_AND_STABLECOIN_CONTEXT",
                "authentication_requirement": "NONE_FOR_PUBLIC_ENDPOINTS",
                "supported_quote_currencies": ["USD"],
                "native_candle_intervals": [],
                "maximum_rows_per_request": None,
                "historical_pagination_model": "DATASET_SPECIFIC_FULL_RESPONSE",
                "trades_endpoint": False,
                "ticker_endpoint": False,
                "order_book_endpoint": False,
                "websocket_support": False,
                "funding_support": False,
                "open_interest_support": False,
                "liquidation_support": False,
                "options_chain_support": False,
                "macro_support": True,
                "revision_or_vintage_support": False,
                "rate_limit_information": "PUBLIC_API_FAIR_USE",
                "plan_limitations": ["some datasets require DefiLlama Pro"],
                "documentation": "https://defillama.com/docs/api",
            },
            "deribit": {
                "provider_category": "CRYPTO_OPTIONS_CONTEXT",
                "authentication_requirement": "NONE_FOR_PUBLIC_MARKET_DATA",
                "supported_quote_currencies": ["USD", "USDC"],
                "native_candle_intervals": [],
                "maximum_rows_per_request": None,
                "historical_pagination_model": "CURRENT_INSTRUMENT_AND_BOOK_SNAPSHOTS",
                "trades_endpoint": True,
                "ticker_endpoint": True,
                "order_book_endpoint": True,
                "websocket_support": True,
                "funding_support": True,
                "open_interest_support": True,
                "liquidation_support": "PUBLIC_TRADES_CONTEXT_ONLY",
                "options_chain_support": True,
                "macro_support": False,
                "revision_or_vintage_support": False,
                "rate_limit_information": "CREDIT_BASED; instruments sustained 1 request/second",
                "plan_limitations": [],
                "documentation": "https://docs.deribit.com/api-reference/market-data/public-get_instruments",
            },
            **{
                name: {
                    "provider_category": "OPTIONAL_CONFIGURED_CONTEXT",
                    "authentication_requirement": "API_KEY",
                    "supported_quote_currencies": [],
                    "native_candle_intervals": [],
                    "maximum_rows_per_request": None,
                    "historical_pagination_model": "PROVIDER_SPECIFIC",
                    "trades_endpoint": False,
                    "ticker_endpoint": False,
                    "order_book_endpoint": False,
                    "websocket_support": False,
                    "funding_support": name in {"coinglass", "glassnode", "cryptoquant"},
                    "open_interest_support": name in {"coinglass", "glassnode", "cryptoquant"},
                    "liquidation_support": name in {"coinglass", "cryptoquant"},
                    "options_chain_support": False,
                    "macro_support": True,
                    "revision_or_vintage_support": False,
                    "rate_limit_information": "PLAN_DEPENDENT",
                    "plan_limitations": ["adapter not enabled without configured credentials"],
                    "documentation": None,
                }
                for name in (
                    "coinglass",
                    "glassnode",
                    "cryptoquant",
                    "coingecko",
                    "polygon",
                )
            },
        }

    @staticmethod
    def _probe_failure_status(exc: Exception) -> ProviderStatus:
        status = getattr(exc, "status", None)
        if status == 429:
            return ProviderStatus.BLOCKED_RATE_LIMIT
        if status in {401, 403}:
            return ProviderStatus.BLOCKED_PERMISSION
        if status in {402}:
            return ProviderStatus.BLOCKED_PLAN_LIMIT
        if status is not None and int(status) >= 500:
            return ProviderStatus.BLOCKED_PROVIDER_UNAVAILABLE
        if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
            return ProviderStatus.BLOCKED_PROVIDER_UNAVAILABLE
        return ProviderStatus.FAILED_VALIDATION

    async def capability_matrix(
        self,
        *,
        probe: bool = True,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        templates = self._capability_templates()
        earliest_by_provider: dict[str, str] = {}
        if self.database is not None:
            for stored in self.database.fetch_records("data_watermarks"):
                payload = dict(stored.get("payload") or {})
                provider = str(payload.get("provider") or stored.get("provider") or "")
                earliest = payload.get("earliest_stored_timestamp")
                if (
                    provider
                    and earliest
                    and (
                        provider not in earliest_by_provider
                        or str(earliest) < earliest_by_provider[provider]
                    )
                ):
                    earliest_by_provider[provider] = str(earliest)
        rows: list[dict[str, Any]] = []
        for provider in self.list_providers():
            configured = self._credential_configured(provider)
            row = {
                "provider": provider,
                **templates[provider],
                "configured_credential_status": (
                    "CONFIGURED"
                    if configured
                    and templates[provider]["authentication_requirement"]
                    not in {
                        "NONE",
                        "PUBLIC_ENDPOINTS_NO_AUTH",
                        "NONE_FOR_PUBLIC_MARKET_DATA",
                        "NONE_FOR_PUBLIC_ENDPOINTS",
                    }
                    else "NOT_REQUIRED"
                    if templates[provider]["authentication_requirement"]
                    in {
                        "NONE",
                        "PUBLIC_ENDPOINTS_NO_AUTH",
                        "NONE_FOR_PUBLIC_MARKET_DATA",
                        "NONE_FOR_PUBLIC_ENDPOINTS",
                    }
                    else "MISSING"
                ),
                "supported_markets": [],
                "earliest_obtainable_timestamp": None,
                "earliest_timestamp_basis": "UNKNOWN_NOT_CLAIMED",
                "last_successful_probe": None,
                "errors": [],
                "source_documentation_version_or_retrieval_date": CAPABILITY_RETRIEVED_AT,
            }
            if not configured:
                row["status"] = ProviderStatus.SKIPPED_MISSING_CREDENTIALS
                row["reason_code"] = "SKIPPED_MISSING_CREDENTIALS"
            elif not probe:
                row["status"] = ProviderStatus.PARTIAL
                row["reason_code"] = "CAPABILITIES_DOCUMENTED_NOT_PROBED"
            else:
                try:
                    payload = await self._probe_provider(provider)
                    row["supported_markets"] = payload.get("supported_markets", [])
                    row["supported_quote_currencies"] = payload.get(
                        "supported_quote_currencies",
                        row["supported_quote_currencies"],
                    )
                    row["earliest_obtainable_timestamp"] = payload.get(
                        "earliest_obtainable_timestamp"
                    ) or earliest_by_provider.get(provider)
                    row["earliest_timestamp_basis"] = (
                        "SAFE_PROBE"
                        if payload.get("earliest_obtainable_timestamp")
                        else "EARLIEST_SUCCESSFULLY_STORED_OBSERVATION"
                        if earliest_by_provider.get(provider)
                        else "UNKNOWN_NOT_CLAIMED"
                    )
                    row["last_successful_probe"] = utc_now().isoformat()
                    row["status"] = payload.get("status", ProviderStatus.READY)
                    row["reason_code"] = payload.get(
                        "reason_code", "SAFE_CAPABILITY_PROBE_SUCCEEDED"
                    )
                except Exception as exc:
                    row["status"] = self._probe_failure_status(exc)
                    row["reason_code"] = type(exc).__name__
                    row["errors"] = [{"type": type(exc).__name__}]
            row["status"] = str(row["status"])
            rows.append(row)
        if persist:
            self._persist_capabilities(rows)
        return rows

    async def _probe_provider(self, provider: str) -> dict[str, Any]:
        if provider == "bitvavo":
            raw = await self.request("GET", f"{BITVAVO_REST}/markets", {}, None)
            markets = [str(row.get("market")) for row in raw if row.get("market")]
        elif provider == "kraken":
            raw = await self.request("GET", f"{KRAKEN_REST}/AssetPairs", {}, None)
            markets = [
                str(row.get("wsname") or row.get("altname") or symbol)
                for symbol, row in raw.get("result", {}).items()
            ]
        elif provider == "mexc":
            raw = await self.request("GET", f"{MEXC_REST}/exchangeInfo", {}, None)
            markets = [
                f"{row.get('baseAsset')}-{row.get('quoteAsset')}"
                for row in raw.get("symbols", [])
                if row.get("baseAsset") and row.get("quoteAsset")
            ]
        elif provider == "coinmarketcap":
            key = self.settings.providers.coinmarketcap_api_key
            raw = await self.request(
                "GET",
                f"{CMC_REST}/cryptocurrency/listings/latest",
                {"limit": 1, "convert": "EUR"},
                {"X-CMC_PRO_API_KEY": key.get_secret_value()} if key else None,
            )
            markets = [
                f"{row.get('symbol')}-EUR" for row in raw.get("data", []) if row.get("symbol")
            ]
        elif provider == "eodhd":
            key = self.settings.providers.eodhd_api_key
            raw = await self.request(
                "GET",
                f"{EODHD_REST}/economic-events",
                {"api_token": key.get_secret_value(), "fmt": "json", "limit": 1} if key else None,
                None,
            )
            markets = ["ECONOMIC_EVENTS"] if isinstance(raw, list) else []
        elif provider == "fred":
            key = self.settings.providers.fred_api_key
            raw = await self.request(
                "GET",
                f"{FRED_REST}/series/observations",
                {
                    "series_id": "DFF",
                    "api_key": key.get_secret_value(),
                    "file_type": "json",
                    "limit": 1,
                }
                if key
                else None,
                None,
            )
            markets = ["DFF"] if raw.get("observations") is not None else []
        elif provider == "sec":
            agent = self.settings.providers.sec_user_agent
            raw = await self.request(
                "GET",
                f"{SEC_REST}/submissions/CIK0000320193.json",
                None,
                {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"} if agent else None,
            )
            markets = [str(raw.get("name") or "SEC_SUBMISSIONS")]
        elif provider == "alternative_me":
            raw = await self.request("GET", f"{ALTERNATIVE_ME_REST}/fng/", {"limit": 1}, None)
            markets = ["CRYPTO_FEAR_GREED"] if raw.get("data") else []
        elif provider == "defillama":
            raw = await self.request("GET", f"{DEFILLAMA_REST}/protocols", None, None)
            markets = [
                str(row.get("slug") or row.get("name"))
                for row in raw[:250]
                if row.get("slug") or row.get("name")
            ]
        elif provider == "deribit":
            raw = await self.request(
                "GET",
                f"{DERIBIT_REST}/get_instruments",
                {"currency": "BTC", "kind": "option", "expired": "false"},
                None,
            )
            markets = [
                str(row.get("instrument_name"))
                for row in raw.get("result", [])
                if row.get("instrument_name")
            ]
        else:
            return {
                "status": ProviderStatus.UNSUPPORTED_ENDPOINT,
                "reason_code": "OPTIONAL_PROVIDER_ADAPTER_NOT_IMPLEMENTED",
                "supported_markets": [],
            }
        quotes = (
            sorted(
                {
                    market.replace("/", "-").split("-")[-1]
                    for market in markets
                    if "-" in market.replace("/", "-")
                }
            )
            if provider in {"bitvavo", "kraken", "mexc", "coinmarketcap"}
            else []
        )
        return {
            "status": ProviderStatus.READY if markets else ProviderStatus.PARTIAL,
            "reason_code": (
                "SAFE_CAPABILITY_PROBE_SUCCEEDED" if markets else "EMPTY_PROBE_RESPONSE"
            ),
            "supported_markets": markets,
            "supported_quote_currencies": quotes,
        }

    def _persist_capabilities(self, rows: list[dict[str, Any]]) -> None:
        report_dir = self.settings.paths.reports_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_dir / "provider_capabilities.json", rows)
        frame = pd.DataFrame(rows)
        for column in frame:
            if frame[column].map(lambda value: isinstance(value, (list, dict))).any():
                frame[column] = frame[column].map(stable_json)
        temporary = report_dir / ".provider_capabilities.csv.tmp"
        frame.to_csv(temporary, index=False)
        os.replace(temporary, report_dir / "provider_capabilities.csv")
        if self.database is not None:
            self.database.upsert_records("provider_capabilities", rows)

    async def _tracked(self, provider: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        health = self._health[provider]
        health["requests"] += 1
        started = utc_now()
        try:
            result = await operation()
            health.update(
                status=ProviderStatus.READY.value,
                reason_code="PUBLIC_DATA_RECEIVED",
                last_success_at=utc_now().isoformat(),
                latency_ms=(utc_now() - started).total_seconds() * 1_000,
                consecutive_failures=0,
                stale=False,
            )
            self._persist_provider_health(provider)
            LOGGER.debug(
                "provider request completed",
                extra={
                    "component": "data_loader",
                    "provider": provider,
                    "operation": "public_request",
                    "duration": (utc_now() - started).total_seconds(),
                    "status": ProviderStatus.READY.value,
                    "reason_code": "PUBLIC_DATA_RECEIVED",
                    "retry_number": 0,
                },
            )
            return result
        except Exception as exc:
            health["errors"] += 1
            health.update(
                status=self._probe_failure_status(exc).value,
                last_error_at=utc_now().isoformat(),
                reason_code=type(exc).__name__,
                consecutive_failures=int(health.get("consecutive_failures") or 0) + 1,
            )
            self._persist_provider_health(provider)
            LOGGER.exception(
                "provider request failed",
                extra={
                    "component": "data_loader",
                    "provider": provider,
                    "operation": "public_request",
                    "duration": (utc_now() - started).total_seconds(),
                    "status": self._probe_failure_status(exc).value,
                    "reason_code": type(exc).__name__,
                    "exception_type": type(exc).__name__,
                    "retry_number": 0,
                },
            )
            raise

    def _persist_provider_health(self, provider: str) -> None:
        if self.database is None:
            return
        health = self._health[provider]
        self.database.upsert_records(
            "provider_health",
            [
                {
                    "external_id": f"{provider}:public_market_data",
                    "provider": provider,
                    "component": "public_market_data",
                    "last_success": health.get("last_success_at"),
                    "last_failure": health.get("last_error_at"),
                    "latency_ms": health.get("latency_ms"),
                    "consecutive_failures": health.get("consecutive_failures", 0),
                    "rate_limit_state": health.get("rate_limit_state", "NORMAL"),
                    "stale": health.get("stale", False),
                    "reconnect_count": health.get("reconnect_count", 0),
                    "status": health.get("status"),
                    "reason_code": health.get("reason_code"),
                    "observed_at": utc_now().isoformat(),
                }
            ],
        )

    @staticmethod
    def _deduplicate(records: Iterable[NormalizedDataRecord]) -> list[NormalizedDataRecord]:
        selected: dict[tuple[Any, ...], NormalizedDataRecord] = {}
        for record in records:
            key = (
                record.provider,
                record.canonical_market,
                record.data_kind,
                record.timeframe,
                record.timestamp,
                record.values.get("trade_id"),
            )
            selected[key] = record
        return sorted(selected.values(), key=lambda item: item.timestamp)

    async def download_ohlcv(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str | None = None,
        resume: bool = True,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        name = provider.lower()
        timeframe = normalize_timeframe(timeframe)
        if name not in self.adapters:
            raise ValueError(f"{name} is not an exchange-quality OHLCV provider")
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError("unsupported timeframe")
        if timeframe not in PROVIDER_NATIVE_TIMEFRAMES[name]:
            raise ValueError(f"{timeframe} is not native for {name}; use download_canonical_ohlcv")
        run = run_id or str(uuid.uuid4())
        cache = self._cache_path(name, normalize_market(market), timeframe, "ohlcv")
        cached: list[NormalizedDataRecord] = []
        requested_start = start
        requested_end = end
        if resume and (
            cache.with_suffix(".parquet").is_file() or cache.with_suffix(".csv").is_file()
        ):
            frame = self.load_local_dataset(cache)
            cached = self._records_from_frame(frame)
        interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
        missing_ranges: list[tuple[datetime, datetime]] = []
        if not cached:
            missing_ranges.append((requested_start, requested_end))
        else:
            earliest = min(item.timestamp for item in cached)
            latest = max(item.timestamp for item in cached)
            if requested_start < earliest:
                missing_ranges.append((requested_start, min(requested_end, earliest - interval)))
            if requested_end > latest + interval:
                missing_ranges.append((max(requested_start, latest + interval), requested_end))

        async def fetch_missing() -> list[NormalizedDataRecord]:
            downloaded: list[NormalizedDataRecord] = []
            for range_start, range_end in missing_ranges:
                if range_start <= range_end:
                    downloaded.extend(
                        await self.adapters[name].ohlcv(
                            normalize_market(market),
                            timeframe,
                            range_start,
                            range_end,
                            run,
                        )
                    )
            return downloaded

        new = await self._tracked(name, fetch_missing) if missing_ranges else []
        close_cutoff = requested_end - timedelta(
            seconds=self.settings.market_data.candle_close_grace_for(timeframe)
        )
        new = [
            record.model_copy(update={"closed": record.timestamp + interval <= close_cutoff})
            for record in new
        ]
        result = [
            record.model_copy(update={"closed": record.timestamp + interval <= close_cutoff})
            for record in self._deduplicate([*cached, *new])
        ]
        if persist:
            cache_exists = (
                cache.with_suffix(".parquet").is_file() or cache.with_suffix(".csv").is_file()
            )
            if new or not cache_exists:
                self._persist_dataset(cache, result)
                self._persist_normalized_dataset(
                    provider=name,
                    market=normalize_market(market),
                    timeframe=timeframe,
                    records=result,
                    source_timeframe=timeframe,
                )
            self._persist_raw_batch(new)
            # The canonical Parquet files retain the complete history. Keep
            # the relational projection bounded on resume: projecting a
            # multi-year 1m cache as one list can consume tens of gigabytes
            # without adding new evidence. Newly fetched rows are always
            # projected in full; an unchanged cache contributes a recent tail
            # so restored databases regain operationally relevant candles.
            projection = (
                self._deduplicate(new)
                if new
                else result[
                    -self.settings.market_data.maximum_database_batch_size :
                ]
            )
            self._database_upsert("candles", projection)
            self._update_watermark(
                provider=name,
                market=normalize_market(market),
                timeframe=timeframe,
                data_kind="ohlcv",
                records=result,
                completed_ranges=missing_ranges,
            )
        LOGGER.info(
            "market data batch completed",
            extra={
                "component": "data_loader",
                "provider": name,
                "dataset": "ohlcv",
                "market": normalize_market(market),
                "timeframe": timeframe,
                "requested_rows": max(
                    0,
                    int((requested_end - requested_start) / interval) + 1,
                ),
                "received_rows": len(new),
                "inserted_rows": len(new),
                "updated_rows": max(0, len(result) - len(new)),
                "duplicate_rows": max(0, len(cached) + len(new) - len(result)),
                "earliest_timestamp": (result[0].timestamp.isoformat() if result else None),
                "latest_timestamp": (result[-1].timestamp.isoformat() if result else None),
                "status": "READY" if result else "PARTIAL",
                "reason_code": (
                    "CLOSED_CANDLE_BATCH_PERSISTED" if result else "EMPTY_PROVIDER_RESPONSE"
                ),
                "retry_number": 0,
            },
        )
        return result

    async def download_canonical_ohlcv(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str | None = None,
        resume: bool = True,
        persist: bool = False,
    ) -> tuple[list[NormalizedDataRecord], dict[str, Any]]:
        """Fetch a native series or deterministically resample one provider only."""

        name = provider.casefold()
        target = normalize_timeframe(timeframe)
        if name not in PROVIDER_NATIVE_TIMEFRAMES:
            raise ValueError(f"{name} is not an OHLCV provider")
        native = target in PROVIDER_NATIVE_TIMEFRAMES[name]
        source = target if native else RESAMPLE_SOURCE.get(name, {}).get(target)
        if source is None:
            return [], {
                "provider": name,
                "market": normalize_market(market),
                "timeframe": target,
                "source_classification": "UNAVAILABLE",
                "reason_code": "NO_VALID_NATIVE_OR_RESAMPLE_SOURCE",
            }
        source_records = await self.download_ohlcv(
            provider=name,
            market=market,
            timeframe=source,
            start=start,
            end=end,
            run_id=run_id,
            resume=resume,
            persist=persist,
        )
        closed_source_records = [record for record in source_records if record.closed is True]
        if native:
            return closed_source_records, {
                "provider": name,
                "market": normalize_market(market),
                "timeframe": target,
                "source_classification": "PROVIDER_NATIVE",
                "source_timeframe": target,
                "rows": len(closed_source_records),
                "excluded_open_rows": len(source_records) - len(closed_source_records),
            }
        derived = self.resample_candles(
            closed_source_records,
            target_timeframe=target,
        )
        if persist:
            self._persist_normalized_dataset(
                provider=name,
                market=normalize_market(market),
                timeframe=target,
                records=derived,
                source_timeframe=source,
            )
            self._update_watermark(
                provider=name,
                market=normalize_market(market),
                timeframe=target,
                data_kind="ohlcv_resampled",
                records=derived,
                completed_ranges=((start, end),),
            )
        return derived, {
            "provider": name,
            "market": normalize_market(market),
            "timeframe": target,
            "source_classification": "RESAMPLED_FROM_NATIVE",
            "source_timeframe": source,
            "rows": len(derived),
            "lineage_hash": stable_hash(
                [item.raw_hash for item in closed_source_records], length=64
            ),
            "excluded_open_rows": len(source_records) - len(closed_source_records),
        }

    @staticmethod
    def _parquet_time_summary(path: Path) -> dict[str, Any]:
        """Read row counts and temporal bounds without materializing all rows."""

        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows <= 0:
            return {
                "rows": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
            }
        first = parquet.read_row_group(
            0,
            columns=["timestamp"],
        ).column("timestamp")
        last = parquet.read_row_group(
            parquet.metadata.num_row_groups - 1,
            columns=["timestamp"],
        ).column("timestamp")
        first_values = pd.to_datetime(
            first.to_pylist(),
            utc=True,
            errors="raise",
        )
        last_values = pd.to_datetime(
            last.to_pylist(),
            utc=True,
            errors="raise",
        )
        return {
            "rows": int(parquet.metadata.num_rows),
            "earliest_timestamp": first_values.min().to_pydatetime(),
            "latest_timestamp": last_values.max().to_pydatetime(),
        }

    @staticmethod
    def _merge_parquet_segments(
        segments: Iterable[Path],
        target: Path,
    ) -> Path:
        """Atomically concatenate sorted, non-overlapping Parquet segments."""

        selected = [
            path
            for path in segments
            if path.is_file()
            and DataLoader._parquet_time_summary(path)[
                "rows"
            ]
            > 0
        ]
        if not selected:
            raise ValueError("at least one Parquet segment is required")
        selected.sort(
            key=lambda path: (
                DataLoader._parquet_time_summary(path)[
                    "earliest_timestamp"
                ]
                or datetime.max.replace(tzinfo=UTC)
            )
        )
        schemas = [
            pq.ParquetFile(path).schema_arrow
            for path in selected
        ]
        schema = pa.unify_schemas(schemas)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
        )
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="snappy",
        )
        try:
            for path in selected:
                source = pq.ParquetFile(path)
                for index in range(source.metadata.num_row_groups):
                    table = source.read_row_group(index)
                    if table.schema != schema:
                        table = table.cast(schema, safe=False)
                    writer.write_table(table)
        finally:
            writer.close()
        os.replace(temporary, target)
        return target

    @staticmethod
    def _copy_native_cache_to_normalized(
        source: Path,
        target: Path,
        *,
        provider: str,
        market: str,
        timeframe: str,
    ) -> Path:
        """Project a provider cache to normalized storage one row group at a time."""

        parquet = pq.ParquetFile(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
        )
        writer: pq.ParquetWriter | None = None
        try:
            for index in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(index)
                size = table.num_rows
                table = table.append_column(
                    "source_provider",
                    pa.array([provider] * size),
                )
                table = table.append_column(
                    "source_timeframe",
                    pa.array([timeframe] * size),
                )
                table = table.append_column(
                    "quote_currency",
                    pa.array([market.split("-")[-1]] * size),
                )
                table = table.append_column(
                    "source_classification",
                    pa.array(["PROVIDER_NATIVE"] * size),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="snappy",
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("native cache contains no row groups")
        os.replace(temporary, target)
        return target

    async def _sync_native_ohlcv_compact(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str | None,
        resume: bool,
        progress_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> dict[str, Any]:
        """Resume a complete native series in bounded provider and Parquet batches."""

        name = provider.casefold()
        canonical_market = normalize_market(market)
        target = normalize_timeframe(timeframe)
        if target not in PROVIDER_NATIVE_TIMEFRAMES[name]:
            raise ValueError(
                f"{target} is not native for {name}"
            )
        run = run_id or str(uuid.uuid4())
        cache_base = self._cache_path(
            name,
            canonical_market,
            target,
            "ohlcv",
        )
        cache_path = cache_base.with_suffix(".parquet")
        csv_path = cache_base.with_suffix(".csv")
        if resume and not cache_path.is_file() and csv_path.is_file():
            legacy = pd.read_csv(csv_path)
            self._atomic_parquet(legacy, cache_path)
        interval = timedelta(seconds=TIMEFRAME_SECONDS[target])
        close_cutoff = end - timedelta(
            seconds=self.settings.market_data.candle_close_grace_for(
                target
            )
        )
        last_closed_open_seconds = (
            int((close_cutoff - interval).timestamp())
            // TIMEFRAME_SECONDS[target]
            * TIMEFRAME_SECONDS[target]
        )
        fetch_end = datetime.fromtimestamp(
            last_closed_open_seconds,
            tz=UTC,
        )
        existing_summary = (
            self._parquet_time_summary(cache_path)
            if resume and cache_path.is_file()
            else {
                "rows": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
            }
        )
        missing_ranges: list[tuple[datetime, datetime]] = []
        earliest = existing_summary["earliest_timestamp"]
        latest = existing_summary["latest_timestamp"]
        if fetch_end < start:
            missing_ranges = []
        elif not existing_summary["rows"]:
            missing_ranges.append((start, fetch_end))
        else:
            if start < earliest:
                missing_ranges.append(
                    (
                        start,
                        min(fetch_end, earliest - interval),
                    )
                )
            if fetch_end >= latest + interval:
                missing_ranges.append(
                    (
                        max(start, latest + interval),
                        fetch_end,
                    )
                )

        staging = cache_path.parent / (
            f".{cache_path.stem}.compact_parts"
        )
        staging.mkdir(parents=True, exist_ok=True)
        staged_parts: list[Path] = []
        rows_per_window = max(
            10_000,
            self.settings.market_data.maximum_database_batch_size
            * 4,
        )
        downloaded_rows = 0
        completed_ranges: list[tuple[datetime, datetime]] = []
        total_windows = sum(
            max(
                0,
                int(
                    (range_end - range_start)
                    / interval
                )
                + 1,
            )
            for range_start, range_end in missing_ranges
        )
        total_windows = (
            (total_windows + rows_per_window - 1)
            // rows_per_window
        )
        completed_windows = 0

        def notify(subphase: str, **details: Any) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "provider": name,
                    "market": canonical_market,
                    "timeframe": target,
                    "source_timeframe": target,
                    "subphase": subphase,
                    "completed_windows": completed_windows,
                    "total_windows": total_windows,
                    "downloaded_rows": downloaded_rows,
                    **details,
                }
            )

        notify(
            "NATIVE_HISTORY_DISCOVERY",
            cached_rows=int(existing_summary["rows"]),
        )
        for range_start, range_end in missing_ranges:
            cursor = range_start
            while cursor <= range_end:
                window_end = min(
                    range_end,
                    cursor
                    + interval * (rows_per_window - 1),
                )
                part = staging / (
                    f"{int(cursor.timestamp())}_"
                    f"{int(window_end.timestamp())}.parquet"
                )
                marker = part.with_suffix(".done.json")
                marker_payload = (
                    read_json(marker)
                    if marker.is_file()
                    else {}
                )
                marker_rows = int(
                    marker_payload.get("rows") or 0
                )
                marker_reusable = marker.is_file() and (
                    marker_rows == 0
                    or cache_path.is_file()
                    or part.is_file()
                )
                if part.is_file() or marker_reusable:
                    if part.is_file():
                        staged_parts.append(part)
                    if not marker.is_file():
                        atomic_write_json(
                            marker,
                            {
                                "provider": name,
                                "market": canonical_market,
                                "timeframe": target,
                                "start": cursor,
                                "end": window_end,
                                "rows": (
                                    pq.ParquetFile(
                                        part
                                    ).metadata.num_rows
                                ),
                                "status": (
                                    "FETCHED_AND_STAGED"
                                ),
                            },
                        )
                    completed_ranges.append((cursor, window_end))
                    completed_windows += 1
                    notify("NATIVE_WINDOW_RESUMED")
                    cursor = window_end + interval
                    continue

                async def fetch_window(
                    selected_start: datetime = cursor,
                    selected_end: datetime = window_end,
                ) -> list[NormalizedDataRecord]:
                    return await self.adapters[name].ohlcv(
                        canonical_market,
                        target,
                        selected_start,
                        selected_end,
                        run,
                    )

                fetched = await self._tracked(
                    name,
                    fetch_window,
                )
                bounded = self._deduplicate(
                    record.model_copy(
                        update={
                            "closed": (
                                record.timestamp + interval
                                <= close_cutoff
                            )
                        }
                    )
                    for record in fetched
                    if cursor
                    <= record.timestamp
                    <= window_end
                )
                if bounded:
                    self._persist_dataset(
                        part.with_suffix(""),
                        bounded,
                    )
                    self._persist_raw_batch(bounded)
                    self._database_upsert("candles", bounded)
                    staged_parts.append(part)
                    downloaded_rows += len(bounded)
                atomic_write_json(
                    marker,
                    {
                        "provider": name,
                        "market": canonical_market,
                        "timeframe": target,
                        "start": cursor,
                        "end": window_end,
                        "rows": len(bounded),
                        "status": (
                            "FETCHED_AND_STAGED"
                            if bounded
                            else "CONFIRMED_EMPTY"
                        ),
                    },
                )
                completed_ranges.append((cursor, window_end))
                completed_windows += 1
                notify(
                    "NATIVE_WINDOW_FETCHED",
                    latest_window_start=cursor,
                    latest_window_end=window_end,
                    latest_window_rows=len(bounded),
                )
                cursor = window_end + interval

        if staged_parts:
            notify(
                "MERGING_NATIVE_SEGMENTS",
                segment_count=len(staged_parts),
            )
            segments = [
                *(
                    [cache_path]
                    if cache_path.is_file()
                    else []
                ),
                *staged_parts,
            ]
            self._merge_parquet_segments(
                segments,
                cache_path,
            )
        final_cache_summary = (
            self._parquet_time_summary(cache_path)
            if cache_path.is_file()
            else {
                "rows": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
            }
        )
        if not final_cache_summary["rows"]:
            return {
                "provider": name,
                "market": canonical_market,
                "timeframe": target,
                "source_classification": "PROVIDER_NATIVE",
                "source_timeframe": target,
                "rows": 0,
                "received_rows": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
                "status": ProviderStatus.PARTIAL.value,
                "reason_code": "EMPTY_PROVIDER_RESPONSE",
                "resource_batching_only": True,
            }
        normalized_path = (
            self.settings.paths.processed_data_dir
            / name
            / canonical_market
            / f"{target}.parquet"
        )
        if staged_parts or not normalized_path.is_file():
            notify("COPYING_NATIVE_NORMALIZED")
            self._copy_native_cache_to_normalized(
                cache_path,
                normalized_path,
                provider=name,
                market=canonical_market,
                timeframe=target,
            )
        summary = final_cache_summary
        notify(
            "NATIVE_HISTORY_COMPLETE",
            cached_rows=int(summary["rows"]),
        )
        for part in staged_parts:
            part.unlink(missing_ok=True)
        try:
            staging.rmdir()
        except OSError:
            pass
        return {
            "provider": name,
            "market": canonical_market,
            "timeframe": target,
            "source_classification": "PROVIDER_NATIVE",
            "source_timeframe": target,
            "rows": summary["rows"],
            "received_rows": summary["rows"],
            "downloaded_rows": downloaded_rows,
            "earliest_timestamp": summary["earliest_timestamp"],
            "latest_timestamp": summary["latest_timestamp"],
            "status": (
                ProviderStatus.READY.value
                if summary["rows"]
                else ProviderStatus.PARTIAL.value
            ),
            "reason_code": (
                "COMPACT_NATIVE_HISTORY_READY"
                if summary["rows"]
                else "EMPTY_PROVIDER_RESPONSE"
            ),
            "completed_page_ranges": [
                [left.isoformat(), right.isoformat()]
                for left, right in completed_ranges
            ],
            "resource_batching_only": True,
        }

    def _resample_cached_ohlcv_compact(
        self,
        *,
        provider: str,
        market: str,
        source_timeframe: str,
        target_timeframe: str,
        progress_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> dict[str, Any]:
        """Resample a large native cache in bounded Arrow batches."""

        name = provider.casefold()
        canonical_market = normalize_market(market)
        source = normalize_timeframe(source_timeframe)
        target = normalize_timeframe(target_timeframe)
        source_path = self._cache_path(
            name,
            canonical_market,
            source,
            "ohlcv",
        ).with_suffix(".parquet")
        if not source_path.is_file():
            return {
                "provider": name,
                "market": canonical_market,
                "timeframe": target,
                "source_classification": (
                    "RESAMPLED_FROM_NATIVE"
                ),
                "source_timeframe": source,
                "rows": 0,
                "received_rows": 0,
                "status": ProviderStatus.PARTIAL.value,
                "reason_code": "SOURCE_CACHE_NOT_AVAILABLE",
                "resource_batching_only": True,
            }
        output = (
            self.settings.paths.processed_data_dir
            / name
            / canonical_market
            / f"{target}.parquet"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if (
            output.is_file()
            and output.stat().st_mtime_ns
            >= source_path.stat().st_mtime_ns
        ):
            summary = self._parquet_time_summary(output)
            if progress_callback is not None:
                progress_callback(
                    {
                        "provider": name,
                        "market": canonical_market,
                        "timeframe": target,
                        "source_timeframe": source,
                        "subphase": "RESAMPLE_UP_TO_DATE",
                        "processed_source_rows": 0,
                        "total_source_rows": int(
                            pq.ParquetFile(
                                source_path
                            ).metadata.num_rows
                        ),
                        "batch_index": 0,
                        "batch_count": 0,
                        "emitted_rows": int(summary["rows"]),
                    }
                )
            return {
                "provider": name,
                "market": canonical_market,
                "timeframe": target,
                "source_classification": (
                    "RESAMPLED_FROM_NATIVE"
                ),
                "source_timeframe": source,
                "rows": int(summary["rows"]),
                "received_rows": int(summary["rows"]),
                "earliest_timestamp": summary[
                    "earliest_timestamp"
                ],
                "latest_timestamp": summary[
                    "latest_timestamp"
                ],
                "status": ProviderStatus.READY.value,
                "reason_code": "COMPACT_RESAMPLE_UP_TO_DATE",
                "incomplete_buckets_excluded": None,
                "resource_batching_only": True,
            }
        temporary = output.with_name(
            f".{output.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
        )
        parquet = pq.ParquetFile(source_path)
        total_source_rows = int(parquet.metadata.num_rows)
        batch_size = 100_000
        batch_count = max(
            1,
            (total_source_rows + batch_size - 1)
            // batch_size,
        )
        processed_source_rows = 0
        processed_batches = 0
        carry = pd.DataFrame()
        writer: pq.ParquetWriter | None = None
        emitted_rows = 0
        incomplete_buckets = 0
        earliest_output: datetime | None = None
        latest_output: datetime | None = None
        now = utc_now()

        def notify(subphase: str) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "provider": name,
                    "market": canonical_market,
                    "timeframe": target,
                    "source_timeframe": source,
                    "subphase": subphase,
                    "processed_source_rows": (
                        processed_source_rows
                    ),
                    "total_source_rows": total_source_rows,
                    "batch_index": processed_batches,
                    "batch_count": batch_count,
                    "emitted_rows": emitted_rows,
                    "incomplete_buckets_excluded": (
                        incomplete_buckets
                    ),
                }
            )

        def bucket_bounds(
            timestamps: pd.Series,
        ) -> pd.Series:
            if target == "1mo":
                return (
                    timestamps.dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                    .dt.to_period("M")
                    .dt.start_time
                    .dt.tz_localize("UTC")
                )
            seconds = TIMEFRAME_SECONDS[target]
            nanoseconds = seconds * 1_000_000_000
            numeric = timestamps.astype("int64")
            return pd.to_datetime(
                (numeric // nanoseconds) * nanoseconds,
                utc=True,
            )

        def emit_groups(
            frame: pd.DataFrame,
            *,
            final: bool,
        ) -> pd.DataFrame:
            nonlocal writer
            nonlocal emitted_rows
            nonlocal incomplete_buckets
            nonlocal earliest_output
            nonlocal latest_output
            if frame.empty:
                return frame
            frame = frame.sort_values("timestamp")
            frame["bucket"] = bucket_bounds(frame["timestamp"])
            buckets = list(frame["bucket"].drop_duplicates())
            retained = pd.DataFrame()
            if not final and buckets:
                retained = frame.loc[
                    frame["bucket"] == buckets[-1]
                ].drop(columns=["bucket"])
                buckets = buckets[:-1]
            if not buckets:
                return retained
            working = frame.loc[
                frame["bucket"].isin(buckets)
            ].copy()
            if working.empty:
                return retained
            source_seconds = TIMEFRAME_SECONDS[source]
            grouped = working.groupby(
                "bucket",
                sort=True,
                observed=True,
            )
            aggregates = grouped.agg(
                source_symbol=("source_symbol", "first"),
                first_timestamp=("timestamp", "first"),
                last_timestamp=("timestamp", "last"),
                unique_timestamp_count=(
                    "timestamp",
                    "nunique",
                ),
                observed_at=("observed_at", "max"),
                retrieval_run_id=(
                    "retrieval_run_id",
                    "last",
                ),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                source_row_count=("timestamp", "size"),
            ).reset_index()
            hash_lists = grouped["raw_hash"].agg(list)
            aggregates["lineage_hash"] = (
                aggregates["bucket"]
                .map(hash_lists)
                .map(
                    lambda hashes: stable_hash(
                        [str(value) for value in hashes],
                        length=64,
                    )
                )
            )
            if target == "1mo":
                target_end = (
                    aggregates["bucket"]
                    + pd.offsets.MonthBegin(1)
                )
                expected = (
                    (
                        target_end
                        - aggregates["bucket"]
                    )
                    .dt.total_seconds()
                    .div(source_seconds)
                    .round()
                    .clip(lower=1)
                    .astype("int64")
                )
            else:
                target_end = (
                    aggregates["bucket"]
                    + pd.to_timedelta(
                        TIMEFRAME_SECONDS[target],
                        unit="s",
                    )
                )
                expected = pd.Series(
                    max(
                        1,
                        round(
                            TIMEFRAME_SECONDS[target]
                            / source_seconds
                        ),
                    ),
                    index=aggregates.index,
                    dtype="int64",
                )
            aggregates["target_end"] = target_end
            aggregates["expected_source_rows"] = expected
            complete = (
                (aggregates["target_end"] <= now)
                & (
                    aggregates["unique_timestamp_count"]
                    == aggregates["expected_source_rows"]
                )
                & (
                    aggregates["first_timestamp"]
                    == aggregates["bucket"]
                )
                & (
                    aggregates["last_timestamp"]
                    + pd.to_timedelta(
                        source_seconds,
                        unit="s",
                    )
                    == aggregates["target_end"]
                )
            )
            incomplete_buckets += int((~complete).sum())
            aggregates = aggregates.loc[complete]
            rows: list[dict[str, Any]] = []
            for row in aggregates.itertuples(index=False):
                lineage_hash = str(row.lineage_hash)
                values = {
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "source_timeframe": source,
                    "resampling_rule": (
                        "MS"
                        if target == "1mo"
                        else f"{TIMEFRAME_SECONDS[target]}s"
                    ),
                    "lineage_hash": lineage_hash,
                    "source_row_count": int(
                        row.source_row_count
                    ),
                    "expected_source_rows": int(
                        row.expected_source_rows
                    ),
                    "missing_source_rows": 0,
                    "gap_flag": False,
                    "source_classification": (
                        "RESAMPLED_FROM_NATIVE"
                    ),
                }
                rows.append(
                    {
                        "provider": name,
                        "source_symbol": str(row.source_symbol),
                        "canonical_market": canonical_market,
                        "timestamp": row.bucket.to_pydatetime(),
                        "observed_at": row.observed_at,
                        "available_at": row.target_end.to_pydatetime(),
                        "data_kind": "ohlcv_resampled",
                        "timeframe": target,
                        "closed": True,
                        "retrieval_run_id": str(
                            row.retrieval_run_id
                        ),
                        "raw_hash": lineage_hash,
                        "values": values,
                        "source_provider": name,
                        "source_timeframe": source,
                        "quote_currency": (
                            canonical_market.split("-")[-1]
                        ),
                        "source_classification": (
                            "RESAMPLED_FROM_NATIVE"
                        ),
                    }
                )
            if rows:
                table = pa.Table.from_pandas(
                    pd.DataFrame(rows),
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="snappy",
                    )
                elif table.schema != writer.schema:
                    table = table.cast(
                        writer.schema,
                        safe=False,
                    )
                writer.write_table(table)
                emitted_rows += len(rows)
                left = rows[0]["timestamp"]
                right = rows[-1]["timestamp"]
                earliest_output = (
                    left
                    if earliest_output is None
                    else min(earliest_output, left)
                )
                latest_output = (
                    right
                    if latest_output is None
                    else max(latest_output, right)
                )
            return retained

        columns = [
            "source_symbol",
            "timestamp",
            "observed_at",
            "retrieval_run_id",
            "raw_hash",
            "values",
            "closed",
        ]
        try:
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                processed_batches += 1
                processed_source_rows += len(batch)
                stored = batch.to_pandas()
                stored = stored.loc[
                    stored["closed"].fillna(False).astype(bool)
                ]
                if stored.empty:
                    continue
                values = pd.json_normalize(
                    stored["values"].map(
                        lambda item: (
                            item
                            if isinstance(item, dict)
                            else {}
                        )
                    )
                )
                compact = pd.DataFrame(
                    {
                        "source_symbol": stored[
                            "source_symbol"
                        ].astype(str).to_numpy(),
                        "timestamp": pd.to_datetime(
                            stored["timestamp"],
                            utc=True,
                            errors="raise",
                        ).to_numpy(),
                        "observed_at": pd.to_datetime(
                            stored["observed_at"],
                            utc=True,
                            errors="raise",
                        ).to_numpy(),
                        "retrieval_run_id": stored[
                            "retrieval_run_id"
                        ].astype(str).to_numpy(),
                        "raw_hash": stored[
                            "raw_hash"
                        ].astype(str).to_numpy(),
                        "open": pd.to_numeric(
                            values["open"],
                            errors="coerce",
                        ).to_numpy(),
                        "high": pd.to_numeric(
                            values["high"],
                            errors="coerce",
                        ).to_numpy(),
                        "low": pd.to_numeric(
                            values["low"],
                            errors="coerce",
                        ).to_numpy(),
                        "close": pd.to_numeric(
                            values["close"],
                            errors="coerce",
                        ).to_numpy(),
                        "volume": pd.to_numeric(
                            values.get(
                                "volume",
                                pd.Series(
                                    0.0,
                                    index=values.index,
                                ),
                            ),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .to_numpy(),
                    }
                ).dropna(
                    subset=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                    ]
                )
                carry = emit_groups(
                    pd.concat(
                        [carry, compact],
                        ignore_index=True,
                    ),
                    final=False,
                )
                notify("RESAMPLING_ARROW_BATCHES")
            emit_groups(carry, final=True)
            notify("RESAMPLE_COMPLETE")
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, output)
        return {
            "provider": name,
            "market": canonical_market,
            "timeframe": target,
            "source_classification": (
                "RESAMPLED_FROM_NATIVE"
            ),
            "source_timeframe": source,
            "rows": emitted_rows,
            "received_rows": emitted_rows,
            "earliest_timestamp": earliest_output,
            "latest_timestamp": latest_output,
            "status": (
                ProviderStatus.READY.value
                if emitted_rows
                else ProviderStatus.PARTIAL.value
            ),
            "reason_code": (
                "COMPACT_RESAMPLE_READY"
                if emitted_rows
                else "NO_COMPLETE_TARGET_BUCKETS"
            ),
            "incomplete_buckets_excluded": incomplete_buckets,
            "resource_batching_only": True,
        }

    async def sync_canonical_ohlcv_compact(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        run_id: str | None = None,
        resume: bool = True,
        progress_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> dict[str, Any]:
        """Memory-bounded canonical sync used by maximum-history campaigns."""

        name = provider.casefold()
        target = normalize_timeframe(timeframe)
        # Providers expose a native `1M`, but their start/end boundary
        # semantics differ and cannot be represented by the fixed seconds
        # used for intraday pagination.  Build canonical `1mo` from complete
        # 1d candles instead; this is causal and calendar aligned.
        native = (
            target in PROVIDER_NATIVE_TIMEFRAMES[name]
            and target != "1mo"
        )
        source = (
            target
            if native
            else RESAMPLE_SOURCE.get(name, {}).get(target)
        )
        if source is None:
            return {
                "provider": name,
                "market": normalize_market(market),
                "timeframe": target,
                "source_classification": "UNAVAILABLE",
                "rows": 0,
                "received_rows": 0,
                "status": ProviderStatus.PARTIAL.value,
                "reason_code": (
                    "NO_VALID_NATIVE_OR_RESAMPLE_SOURCE"
                ),
                "resource_batching_only": True,
            }
        source_summary = await self._sync_native_ohlcv_compact(
            provider=name,
            market=market,
            timeframe=source,
            start=start,
            end=end,
            run_id=run_id,
            resume=resume,
            progress_callback=progress_callback,
        )
        if native or not source_summary.get("rows"):
            return source_summary
        return self._resample_cached_ohlcv_compact(
            provider=name,
            market=market,
            source_timeframe=source,
            target_timeframe=target,
            progress_callback=progress_callback,
        )

    def materialize_provider_ohlcv_compact(
        self,
        source: Path,
        target: Path,
        *,
        provider: str,
        market: str,
        timeframe: str,
        maximum_staleness: timedelta,
        progress_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> dict[str, Any]:
        """Flatten a provider Parquet file without loading it in full.

        The canonical research file keeps only timestamp and OHLCV columns.
        Validation, gap accounting, provider lineage and the manifest are
        accumulated across Arrow batches, so maximum-history materialization
        remains bounded by ``batch_size``.
        """

        from data.market_data import (
            DataManifest,
            DataQualityReport,
            candle_close_timestamp,
            timeframe_delta,
        )

        canonical_market = normalize_market(market)
        normalized_timeframe = normalize_timeframe(timeframe)
        interval = timeframe_delta(normalized_timeframe)
        interval_seconds = int(interval.total_seconds())
        parquet = pq.ParquetFile(source)
        total_source_rows = int(parquet.metadata.num_rows)
        batch_size = 100_000
        batch_count = max(
            1,
            (total_source_rows + batch_size - 1)
            // batch_size,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet"
        )
        writer: pq.ParquetWriter | None = None
        row_count = 0
        processed_rows = 0
        processed_batches = 0
        first_timestamp: pd.Timestamp | None = None
        last_timestamp: pd.Timestamp | None = None
        previous_timestamp: pd.Timestamp | None = None
        largest_gap_bars = 0
        duplicate_count = 0
        invalid_count = 0
        source_classification = "PROVIDER_NATIVE"
        provider_hasher = hashlib.sha256()

        def notify(subphase: str) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "provider": provider,
                    "market": canonical_market,
                    "timeframe": normalized_timeframe,
                    "subphase": subphase,
                    "processed_source_rows": processed_rows,
                    "total_source_rows": total_source_rows,
                    "batch_index": processed_batches,
                    "batch_count": batch_count,
                    "emitted_rows": row_count,
                }
            )

        columns = [
            "timestamp",
            "raw_hash",
            "values",
            "closed",
            "source_classification",
        ]
        available_columns = set(parquet.schema_arrow.names)
        selected_columns = [
            column
            for column in columns
            if column in available_columns
        ]
        required = {"timestamp", "values"}
        if not required.issubset(selected_columns):
            raise DataValidationError(
                "provider Parquet lacks timestamp or values"
            )
        try:
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=selected_columns,
            ):
                processed_batches += 1
                processed_rows += len(batch)
                stored = batch.to_pandas()
                if "closed" in stored:
                    stored = stored.loc[
                        stored["closed"]
                        .fillna(False)
                        .astype(bool)
                    ]
                if stored.empty:
                    notify("MATERIALIZING_ARROW_BATCHES")
                    continue
                values = pd.json_normalize(
                    stored["values"].map(
                        lambda value: (
                            value
                            if isinstance(value, dict)
                            else {}
                        )
                    )
                )
                missing = [
                    column
                    for column in (
                        "open",
                        "high",
                        "low",
                        "close",
                    )
                    if column not in values
                ]
                if missing:
                    raise DataValidationError(
                        f"provider values lack OHLC columns: {missing}"
                    )
                frame = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            stored["timestamp"],
                            utc=True,
                            errors="raise",
                        ).to_numpy(),
                        "open": pd.to_numeric(
                            values["open"],
                            errors="coerce",
                        ).to_numpy(),
                        "high": pd.to_numeric(
                            values["high"],
                            errors="coerce",
                        ).to_numpy(),
                        "low": pd.to_numeric(
                            values["low"],
                            errors="coerce",
                        ).to_numpy(),
                        "close": pd.to_numeric(
                            values["close"],
                            errors="coerce",
                        ).to_numpy(),
                        "volume": pd.to_numeric(
                            values.get(
                                "volume",
                                pd.Series(
                                    0.0,
                                    index=values.index,
                                ),
                            ),
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .to_numpy(),
                    }
                )
                numeric = frame[
                    ["open", "high", "low", "close", "volume"]
                ]
                finite = numeric.notna().all(axis=1)
                valid = (
                    finite
                    & (frame[["open", "high", "low", "close"]] > 0)
                    .all(axis=1)
                    & (frame["volume"] >= 0)
                    & (
                        frame["high"]
                        >= frame[
                            ["open", "close", "low"]
                        ].max(axis=1)
                    )
                    & (
                        frame["low"]
                        <= frame[
                            ["open", "close", "high"]
                        ].min(axis=1)
                    )
                )
                if not bool(valid.all()):
                    invalid_count += int((~valid).sum())
                    raise DataValidationError(
                        "provider OHLCV contains invalid rows"
                    )
                frame = frame.sort_values("timestamp")
                timestamps = pd.DatetimeIndex(
                    frame["timestamp"]
                )
                duplicate_count += int(
                    timestamps.duplicated(keep=False).sum()
                )
                if duplicate_count:
                    raise DataValidationError(
                        "provider OHLCV contains duplicate timestamps"
                    )
                batch_first = pd.Timestamp(timestamps[0])
                batch_last = pd.Timestamp(timestamps[-1])
                if (
                    previous_timestamp is not None
                    and batch_first <= previous_timestamp
                ):
                    raise DataValidationError(
                        "provider OHLCV is not globally increasing"
                    )
                if normalized_timeframe == "1mo":
                    month_ordinals = (
                        timestamps.year * 12
                        + timestamps.month
                    )
                    if previous_timestamp is not None:
                        previous_ordinal = (
                            previous_timestamp.year * 12
                            + previous_timestamp.month
                        )
                        largest_gap_bars = max(
                            largest_gap_bars,
                            max(
                                0,
                                int(month_ordinals[0])
                                - int(previous_ordinal)
                                - 1,
                            ),
                        )
                    if len(month_ordinals) > 1:
                        largest_gap_bars = max(
                            largest_gap_bars,
                            max(
                                0,
                                int(
                                    pd.Series(
                                        month_ordinals
                                    ).diff().max()
                                )
                                - 1,
                            ),
                        )
                else:
                    combined_deltas = (
                        timestamps.to_series().diff()
                    )
                    if previous_timestamp is not None:
                        boundary_delta = (
                            batch_first - previous_timestamp
                        )
                        largest_gap_bars = max(
                            largest_gap_bars,
                            max(
                                0,
                                int(
                                    boundary_delta.total_seconds()
                                    // interval_seconds
                                )
                                - 1,
                            ),
                        )
                    if len(combined_deltas) > 1:
                        largest_delta = (
                            combined_deltas.dropna().max()
                        )
                        largest_gap_bars = max(
                            largest_gap_bars,
                            max(
                                0,
                                int(
                                    largest_delta.total_seconds()
                                    // interval_seconds
                                )
                                - 1,
                            ),
                        )
                first_timestamp = (
                    batch_first
                    if first_timestamp is None
                    else first_timestamp
                )
                last_timestamp = batch_last
                previous_timestamp = batch_last
                if "raw_hash" in stored:
                    for raw_hash in stored["raw_hash"].astype(
                        str
                    ):
                        encoded = raw_hash.encode("utf-8")
                        provider_hasher.update(
                            len(encoded).to_bytes(
                                4,
                                "big",
                            )
                        )
                        provider_hasher.update(encoded)
                if (
                    "source_classification" in stored
                    and stored[
                        "source_classification"
                    ].notna().any()
                ):
                    source_classification = str(
                        stored[
                            "source_classification"
                        ].dropna().iloc[-1]
                    )
                table = pa.Table.from_pandas(
                    frame,
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="snappy",
                    )
                elif table.schema != writer.schema:
                    table = table.cast(
                        writer.schema,
                        safe=False,
                    )
                writer.write_table(table)
                row_count += len(frame)
                notify("MATERIALIZING_ARROW_BATCHES")
        finally:
            if writer is not None:
                writer.close()
        if (
            writer is None
            or first_timestamp is None
            or last_timestamp is None
        ):
            temporary.unlink(missing_ok=True)
            raise DataValidationError(
                "provider OHLCV contains no closed rows"
            )
        now = utc_now()
        if (
            candle_close_timestamp(
                last_timestamp,
                normalized_timeframe,
            ).to_pydatetime()
            > now
        ):
            temporary.unlink(missing_ok=True)
            raise DataValidationError(
                "provider OHLCV contains an open candle"
            )
        os.replace(temporary, target)
        expected_rows = (
            (
                last_timestamp.year
                - first_timestamp.year
            )
            * 12
            + last_timestamp.month
            - first_timestamp.month
            + 1
            if normalized_timeframe == "1mo"
            else int(
                (
                    last_timestamp
                    - first_timestamp
                ).total_seconds()
                // interval_seconds
            )
            + 1
        )
        missing_rows = max(0, expected_rows - row_count)
        missing_fraction = (
            missing_rows / expected_rows
            if expected_rows
            else 0.0
        )
        age_seconds = max(
            0.0,
            (
                now
                - candle_close_timestamp(
                    last_timestamp,
                    normalized_timeframe,
                ).to_pydatetime()
            ).total_seconds(),
        )
        stale = (
            age_seconds
            > maximum_staleness.total_seconds()
        )
        reasons: list[str] = []
        if stale:
            reasons.append("STALE_DATA")
        if missing_fraction > 0.05:
            reasons.append("EXCESSIVE_MISSING_BARS")
        quality = DataQualityReport(
            market=canonical_market,
            timeframe=normalized_timeframe,
            rows=row_count,
            start=first_timestamp.to_pydatetime(),
            end=last_timestamp.to_pydatetime(),
            expected_rows=expected_rows,
            missing_rows=missing_rows,
            missing_fraction=missing_fraction,
            largest_gap_bars=largest_gap_bars,
            duplicate_timestamps=duplicate_count,
            stale=stale,
            age_seconds=age_seconds,
            valid=not reasons,
            reasons=tuple(reasons),
        )
        first = first_timestamp.to_pydatetime()
        last = last_timestamp.to_pydatetime()
        calendar_days = max(
            0.0,
            (last - first).total_seconds() / 86_400,
        )
        exact_required_start = (
            pd.Timestamp(last) - pd.DateOffset(years=7)
        ).to_pydatetime()
        seven_year_eligible = bool(
            first <= exact_required_start
            and quality.valid
        )
        base_asset, quote_asset = canonical_market.split(
            "-",
            1,
        )
        data_hash = sha256_file(target)
        created_at = utc_now()
        manifest = DataManifest(
            market=canonical_market,
            base_asset=base_asset,
            quote_asset=quote_asset,
            exchange=provider.upper(),
            provider=provider,
            timeframe=normalized_timeframe,
            rows=row_count,
            start=first,
            end=last,
            requested_start=first,
            requested_end=now,
            actual_first_timestamp=first,
            actual_last_timestamp=last,
            raw_calendar_days=calendar_days,
            usable_calendar_days=calendar_days,
            raw_bar_count=row_count,
            usable_bar_count=row_count,
            expected_bar_count=expected_rows,
            missing_bar_count=missing_rows,
            missing_bar_ratio=missing_fraction,
            duplicate_count=duplicate_count,
            invalid_bar_count=invalid_count,
            stale_bar_count=0,
            largest_gap=(
                (
                    f"{largest_gap_bars + 1} months"
                    if normalized_timeframe == "1mo"
                    else str(
                        interval
                        * (largest_gap_bars + 1)
                    )
                )
                if largest_gap_bars
                else None
            ),
            listing_date_if_known=first,
            source_segments=(
                {
                    "provider": provider,
                    "exchange": provider.upper(),
                    "market_identity": canonical_market,
                    "start": utc_iso(first),
                    "end": utc_iso(last),
                    "classification": source_classification,
                },
            ),
            dataset_hash=data_hash,
            generated_at=created_at,
            seven_year_eligible=seven_year_eligible,
            history_coverage_ratio=(
                calendar_days / (7 * 365.2425)
            ),
            rejection_reason=(
                None
                if seven_year_eligible
                else (
                    "DATA_QUALITY_FAILED"
                    if not quality.valid
                    else "INSUFFICIENT_MARKET_HISTORY"
                )
            ),
            columns=(
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ),
            data_file=target.name,
            sha256=data_hash,
            created_at=created_at,
            quality=quality,
        )
        manifest_path = target.with_suffix(
            f"{target.suffix}.manifest.json"
        )
        atomic_write_json(
            manifest_path,
            manifest.model_dump(mode="json"),
        )
        notify("CANONICAL_MATERIALIZATION_COMPLETE")
        return {
            "provider": provider,
            "market": canonical_market,
            "timeframe": normalized_timeframe,
            "rows": row_count,
            "path": target,
            "manifest": manifest_path,
            "sha256": data_hash,
            "provider_hash": provider_hasher.hexdigest(),
            "source_classification": source_classification,
            "status": ProviderStatus.READY.value,
            "reason_code": (
                "CANONICAL_FILE_MATERIALIZED_COMPACT"
            ),
            "resource_batching_only": True,
        }

    @staticmethod
    def resample_candles(
        records: Iterable[NormalizedDataRecord],
        *,
        target_timeframe: str,
    ) -> list[NormalizedDataRecord]:
        selected = list(records)
        if not selected:
            return []
        target = normalize_timeframe(target_timeframe)
        if target not in TIMEFRAME_SECONDS:
            raise ValueError("unsupported resample target")
        providers = {item.provider for item in selected}
        markets = {item.canonical_market for item in selected}
        source_timeframes = {item.timeframe for item in selected}
        if len(providers) != 1 or len(markets) != 1 or len(source_timeframes) != 1:
            raise ValueError("resampling cannot mix providers, markets or source intervals")
        if any(item.closed is not True for item in selected):
            raise ValueError("resampling requires closed source candles")
        source = next(iter(source_timeframes))
        if source is None or source not in TIMEFRAME_SECONDS:
            raise ValueError("source timeframe is unavailable")
        if TIMEFRAME_SECONDS[source] >= TIMEFRAME_SECONDS[target]:
            raise ValueError("resampling target must be coarser than source")
        frame = (
            pd.DataFrame(
                [
                    {
                        "timestamp": item.timestamp,
                        "open": float(item.values["open"]),
                        "high": float(item.values["high"]),
                        "low": float(item.values["low"]),
                        "close": float(item.values["close"]),
                        "volume": float(item.values.get("volume") or 0),
                        "raw_hash": item.raw_hash,
                        "retrieval_run_id": item.retrieval_run_id,
                        "observed_at": item.observed_at,
                    }
                    for item in selected
                ]
            )
            .set_index("timestamp")
            .sort_index()
        )
        rule = {
            "1W": "W-MON",
            "1mo": "MS",
        }.get(target, f"{TIMEFRAME_SECONDS[target]}s")
        grouped = frame.resample(
            rule,
            label="left",
            closed="left",
            origin="epoch",
        )
        output: list[NormalizedDataRecord] = []
        provider = next(iter(providers))
        market = next(iter(markets))
        now = utc_now()
        for timestamp, group in grouped:
            if group.empty:
                continue
            target_end = (
                timestamp + pd.offsets.MonthBegin(1)
                if target == "1mo"
                else timestamp + pd.to_timedelta(TIMEFRAME_SECONDS[target], unit="s")
            )
            if target_end.to_pydatetime().astimezone(UTC) > now:
                continue
            expected = max(
                1,
                round(
                    (
                        target_end.to_pydatetime().astimezone(UTC)
                        - timestamp.to_pydatetime().astimezone(UTC)
                    ).total_seconds()
                    / TIMEFRAME_SECONDS[source]
                ),
            )
            gap_count = max(0, expected - len(group))
            if gap_count:
                # A time bucket can be historically closed while still being
                # incomplete because the fetched source slice starts/ends
                # inside that bucket or contains a provider gap. Never turn
                # such a fragment into a tradable OHLCV candle. The missing
                # target interval remains visible through watermark gap
                # reconciliation and can be retried from native source data.
                continue
            lineage_hash = stable_hash(group["raw_hash"].tolist(), length=64)
            values = {
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group["close"].iloc[-1]),
                "volume": float(group["volume"].sum()),
                "source_timeframe": source,
                "resampling_rule": rule,
                "lineage_hash": lineage_hash,
                "source_row_count": len(group),
                "expected_source_rows": expected,
                "missing_source_rows": gap_count,
                "gap_flag": gap_count > 0,
                "source_classification": "RESAMPLED_FROM_NATIVE",
            }
            raw = {
                "source_hashes": group["raw_hash"].tolist(),
                "lineage": values,
            }
            output.append(
                NormalizedDataRecord(
                    provider=provider,
                    source_symbol=selected[0].source_symbol,
                    canonical_market=market,
                    timestamp=timestamp.to_pydatetime().astimezone(UTC),
                    observed_at=max(group["observed_at"]),
                    available_at=target_end.to_pydatetime().astimezone(UTC),
                    data_kind="ohlcv_resampled",
                    timeframe=target,
                    closed=True,
                    retrieval_run_id=str(group["retrieval_run_id"].iloc[-1]),
                    raw_hash=_raw_hash(raw),
                    raw_payload=raw,
                    values=values,
                )
            )
        return output

    @staticmethod
    def history_start(
        *,
        profile: str | HistoryProfile,
        timeframe: str,
        provider: str,
        end: datetime,
    ) -> datetime:
        selected = HistoryProfile(str(profile).casefold())
        normalized = normalize_timeframe(timeframe)
        target = HISTORY_TARGETS[selected][normalized]
        provider_floor = {
            "bitvavo": datetime(2018, 1, 1, tzinfo=UTC),
            "kraken": end
            - timedelta(
                seconds=TIMEFRAME_SECONDS[
                    normalized
                    if normalized in PROVIDER_NATIVE_TIMEFRAMES["kraken"]
                    else RESAMPLE_SOURCE["kraken"].get(normalized, "1h")
                ]
                * 720
            ),
            "mexc": datetime(2023, 1, 1, tzinfo=UTC),
        }.get(provider, datetime(2013, 4, 28, tzinfo=UTC))
        requested = end - target if target is not None else provider_floor
        return max(provider_floor, requested)

    def estimate_fetch(
        self,
        *,
        providers: Iterable[str],
        universe_size: int,
        history_profile: str,
        timeframes: Iterable[str],
    ) -> dict[str, Any]:
        selected_providers = [name for name in providers if name in PROVIDER_NATIVE_TIMEFRAMES]
        selected_timeframes = [normalize_timeframe(item) for item in timeframes]
        now = utc_now()
        rows = 0
        calls = 0
        by_provider: dict[str, Any] = {}
        for provider in selected_providers:
            provider_rows = 0
            provider_calls = 0
            maximum = 1_440 if provider == "bitvavo" else 720 if provider == "kraken" else 1_000
            for timeframe in selected_timeframes:
                source = (
                    timeframe
                    if timeframe in PROVIDER_NATIVE_TIMEFRAMES[provider]
                    else RESAMPLE_SOURCE.get(provider, {}).get(timeframe)
                )
                if source is None:
                    continue
                start = self.history_start(
                    profile=history_profile,
                    timeframe=timeframe,
                    provider=provider,
                    end=now,
                )
                count = max(
                    0,
                    int((now - start).total_seconds() / TIMEFRAME_SECONDS[source]),
                )
                provider_rows += count * universe_size
                provider_calls += max(1, (count + maximum - 1) // maximum) * universe_size
            rows += provider_rows
            calls += provider_calls
            by_provider[provider] = {
                "estimated_rows": provider_rows,
                "estimated_calls": provider_calls,
                "credential_configured": self._credential_configured(provider),
            }
        # All three durable history projections are Parquet-compressed. The
        # previous 240-byte "raw" assumption described an uncompressed JSON
        # row and blocked maximum history even when measured immutable raw
        # Parquet used roughly 83 bytes/candle. Keep a conservative 128-byte
        # allowance plus separate cache/normalized projections and context.
        compressed = rows * 72
        raw = rows * 128
        normalized = rows * 96
        context = int(rows * 24)
        total_storage = compressed + raw + normalized + context
        free = shutil.disk_usage(self.settings.paths.data_dir).free
        return {
            "status": "ESTIMATE",
            "providers": by_provider,
            "universe_size": universe_size,
            "history_profile": history_profile,
            "timeframes": selected_timeframes,
            "estimated_provider_calls": calls,
            "estimated_rows": rows,
            "estimated_compressed_storage_bytes": compressed,
            "estimated_raw_storage_bytes": raw,
            "estimated_normalized_storage_bytes": normalized,
            "estimated_context_storage_bytes": context,
            "estimated_total_storage_bytes": total_storage,
            "storage_estimate_basis": (
                "PARQUET_COMPRESSED_CACHE_72_RAW_128_"
                "NORMALIZED_96_CONTEXT_24_BYTES_PER_ROW"
            ),
            "estimated_duration_seconds": round(calls * 0.12, 1),
            "estimated_api_credits": {
                "coinmarketcap": "PLAN_DEPENDENT",
                "eodhd": "PLAN_DEPENDENT",
                "public_exchanges": 0,
            },
            "free_disk_bytes": free,
            "storage_allowed": (
                total_storage
                <= self.settings.market_data.maximum_storage_gb * 1024**3
                and free - total_storage
                >= self.settings.market_data.minimum_free_disk_gb * 1024**3
            ),
            "requires_confirmation": (
                history_profile.casefold() == HistoryProfile.MAXIMUM or calls > 10_000
            ),
        }

    async def download_trades(
        self,
        *,
        provider: str,
        market: str,
        run_id: str | None = None,
        persist: bool = False,
        mode: str = "research",
    ) -> list[NormalizedDataRecord]:
        name = provider.lower()
        if name not in self.adapters:
            raise ValueError("trades are supported only by exchange adapters")
        records = await self._tracked(
            name,
            lambda: self.adapters[name].trades(
                normalize_market(market), run_id or str(uuid.uuid4())
            ),
        )
        records = [
            record.model_copy(update={"values": {**record.values, "mode": mode}})
            for record in self._deduplicate(records)
        ]
        if persist:
            self._persist_raw_batch(records)
            self._database_upsert("trades", records)
        return records

    async def download_ticker(
        self,
        *,
        provider: str,
        market: str,
        run_id: str | None = None,
        persist: bool = False,
        mode: str = "research",
    ) -> NormalizedDataRecord:
        name = provider.lower()
        if name not in self.adapters:
            raise ValueError("ticker is supported only by exchange adapters")
        record = await self._tracked(
            name,
            lambda: self.adapters[name].ticker(
                normalize_market(market), run_id or str(uuid.uuid4())
            ),
        )
        record = record.model_copy(update={"values": {**record.values, "mode": mode}})
        if persist:
            self._persist_raw_batch([record])
            self._database_upsert("ticker_events", [record])
        return record

    async def download_orderbook_snapshot(
        self,
        *,
        provider: str,
        market: str,
        depth: int = 100,
        run_id: str | None = None,
        persist: bool = False,
        mode: str = "research",
    ) -> NormalizedDataRecord:
        name = provider.lower()
        record = await self._tracked(
            name,
            lambda: self.adapters[name].orderbook(
                normalize_market(market), depth, run_id or str(uuid.uuid4())
            ),
        )
        record = record.model_copy(update={"values": {**record.values, "mode": mode}})
        if persist:
            self._persist_raw_batch([record])
            self._database_upsert("orderbook_snapshots", [record])
        return record

    async def download_market_metadata(
        self,
        *,
        provider: str,
        market: str | None = None,
        run_id: str | None = None,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        name = provider.lower()
        if name in self.adapters:
            records = await self._tracked(
                name,
                lambda: self.adapters[name].metadata(market, run_id or str(uuid.uuid4())),
            )
        elif name == "coinmarketcap":
            records = await self._cmc_metadata(market, run_id or str(uuid.uuid4()))
        else:
            raise ValueError("market metadata provider is unsupported")
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records(f"{name}_market_metadata", records)
        return records

    async def download_cmc_rankings(
        self,
        *,
        limit: int = 100,
        convert: str = "EUR",
        run_id: str | None = None,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        """Fetch a current immutable CoinMarketCap rank observation.

        The caller owns snapshot persistence and eligibility decisions. This
        provider method deliberately records only source facts.
        """

        if limit < 1 or limit > 5_000:
            raise ValueError("CMC ranking limit must be between 1 and 5000")
        records = await self._cmc_rankings(
            limit=limit,
            convert=convert.upper(),
            run_id=run_id or str(uuid.uuid4()),
        )
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records("coinmarketcap_rankings", records)
        return records

    async def download_macro_series(
        self,
        *,
        provider: str,
        series: str,
        start: datetime | None = None,
        end: datetime | None = None,
        run_id: str | None = None,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        name = provider.lower()
        run = run_id or str(uuid.uuid4())
        if name == "fred":
            records = await self._fred(series, start, end, run)
        elif name == "eodhd":
            records = await self._eodhd(series, start, end, run)
        elif name == "sec":
            records = await self._sec(series, run)
        elif name == "coinmarketcap":
            if series.upper() in {"GLOBAL", "GLOBAL_METRICS"}:
                records = await self._cmc_global(run)
            else:
                records = await self._cmc_quotes(series, start, end, run)
        elif name == "alternative_me":
            records = await self._fear_and_greed(run)
        elif name == "defillama":
            records = await self._defillama(series, run)
        else:
            raise ValueError("unsupported macro provider")
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records(f"{name}_{series.casefold()}", records)
            table = "scraper_intelligence" if name == "sec" else "macro_observations"
            self._database_upsert(table, records)
        return records

    async def download_fred_vintages(
        self,
        *,
        series: str,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.fred_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        raw = await self._tracked(
            "fred",
            lambda: self.request(
                "GET",
                f"{FRED_REST}/series/vintagedates",
                {
                    "series_id": series,
                    "api_key": key.get_secret_value(),
                    "file_type": "json",
                    "limit": 10_000,
                    "sort_order": "asc",
                },
                None,
            ),
        )
        retrieval_run_id = str(uuid.uuid4())
        records = [
            NormalizedDataRecord(
                provider="fred",
                source_symbol=series,
                canonical_market="BTC-EUR",
                timestamp=_iso(value),
                observed_at=observed,
                available_at=_iso(value),
                data_kind="macro_vintage",
                retrieval_run_id=retrieval_run_id,
                raw_hash=_raw_hash({"series": series, "vintage_date": value}),
                raw_payload={"series": series, "vintage_date": value},
                values={
                    "series_id": series,
                    "vintage_date": value,
                    "point_in_time_status": "ALFRED_VINTAGE_DATE",
                },
            )
            for value in raw.get("vintage_dates", [])
        ]
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records(f"fred_{series}_vintages", records)
            self._database_upsert("macro_observations", records)
        return records

    async def download_fred_revisions(
        self,
        *,
        series: str,
        start: datetime | None = None,
        end: datetime | None = None,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.fred_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        params: dict[str, Any] = {
            "series_id": series,
            "api_key": key.get_secret_value(),
            "file_type": "json",
            "output_type": 2,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": 100_000,
        }
        if start:
            params["observation_start"] = start.date().isoformat()
        if end:
            params["observation_end"] = end.date().isoformat()
        raw = await self._tracked(
            "fred",
            lambda: self.request(
                "GET",
                f"{FRED_REST}/series/observations",
                params,
                None,
            ),
        )
        records: list[NormalizedDataRecord] = []
        for row in raw.get("observations", []):
            if row.get("value") in {None, "."}:
                continue
            observation = _iso(row["date"])
            vintage = _iso(row.get("realtime_start", row["date"]))
            records.append(
                NormalizedDataRecord(
                    provider="fred",
                    source_symbol=series,
                    canonical_market="BTC-EUR",
                    timestamp=observation,
                    observed_at=observed,
                    available_at=vintage,
                    data_kind="macro_revision",
                    retrieval_run_id=str(uuid.uuid4()),
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values={
                        "series_id": series,
                        "value": row["value"],
                        "observation_date": observation.isoformat(),
                        "vintage_date": vintage.date().isoformat(),
                        "revision_valid_until": row.get("realtime_end"),
                        "point_in_time_status": "ALFRED_VINTAGE_VALUE",
                    },
                )
            )
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records(f"fred_{series}_revisions", records)
            self._database_upsert("macro_observations", records)
        return records

    async def download_gex_context(
        self,
        *,
        underlying: str = "BTC",
        persist: bool = False,
    ) -> dict[str, Any]:
        from data.derivatives_context import (
            CryptoGEXAnalyzer,
            DeribitOptionsCollector,
        )
        from research.macro_context import summarize_gex_contracts

        contracts = await DeribitOptionsCollector(requester=self.request).collect(underlying)
        summary = CryptoGEXAnalyzer().calculate(contracts)
        if persist and contracts:
            contract_frame = pd.DataFrame([item.model_dump(mode="json") for item in contracts])
            contract_target = (
                self.settings.paths.context_data_dir
                / f"options_deribit_{underlying.upper()}.parquet"
            )
            self._atomic_parquet(contract_frame, contract_target)
            summary_frame = summarize_gex_contracts(contracts).reset_index(names="available_at")
            summary_frame["provider"] = "deribit"
            summary_frame["point_in_time_status"] = "FORWARD_ONLY"
            summary_frame["observed_at"] = summary_frame["available_at"]
            summary_frame["raw_hash"] = stable_hash(
                contract_frame.to_dict(orient="records"), length=64
            )
            summary_target = (
                self.settings.paths.context_data_dir
                / f"gex_{underlying.upper()}.parquet"
            )
            history = pd.DataFrame()
            if summary_target.is_file():
                history = pd.read_parquet(summary_target)
                if "available_at" in history:
                    history["available_at"] = pd.to_datetime(
                        history["available_at"], utc=True, errors="coerce"
                    )
            current_at = pd.Timestamp(summary_frame["available_at"].iloc[0])
            for label, hours in (("1h", 1), ("4h", 4), ("24h", 24)):
                prior = history.loc[
                    history.get("available_at", pd.Series(dtype="datetime64[ns, UTC]")).le(
                        current_at - pd.Timedelta(hours=hours)
                    )
                ] if not history.empty and "available_at" in history else pd.DataFrame()
                for source, prefix in (
                    ("absolute_gex", "absolute_gex"),
                    ("convention_signed_gex", "signed_gex"),
                ):
                    current_value = float(summary_frame[source].iloc[0])
                    previous_value = (
                        float(prior.iloc[-1][source])
                        if not prior.empty
                        and source in prior
                        and pd.notna(prior.iloc[-1][source])
                        else None
                    )
                    summary_frame[f"{prefix}_change_{label}"] = (
                        current_value / previous_value - 1.0
                        if previous_value not in {None, 0.0}
                        else None
                    )
            if not history.empty:
                summary_frame = pd.concat(
                    [history, summary_frame], ignore_index=True, sort=False
                )
            summary_frame = (
                # Identical option books observed at different times are still
                # distinct point-in-time observations.  Deduplicating only on
                # the payload hash moved the sole surviving ``available_at``
                # forward on every refresh and made the previously available
                # GEX state disappear from causal scans.
                summary_frame.drop_duplicates(
                    ["available_at", "raw_hash"],
                    keep="last",
                )
                .sort_values("available_at")
                .tail(24 * 90)
                .reset_index(drop=True)
            )
            self._atomic_parquet(
                summary_frame,
                summary_target,
            )
            if self.database is not None:
                self.database.upsert_records(
                    "derivatives_context",
                    [
                        {
                            "external_id": stable_hash(
                                [
                                    underlying,
                                    summary_frame["raw_hash"].iloc[0],
                                ],
                                length=64,
                            ),
                            "provider": "deribit",
                            "market": f"{underlying.upper()}-USD",
                            "timestamp": contracts[0].observed_at,
                            "observed_at": contracts[0].observed_at,
                            "available_at": contracts[0].available_at,
                            "status": summary.get("status"),
                            "data_kind": "gex_context",
                            **summary,
                        }
                    ],
                )
        return {
            **summary,
            "provider": "deribit",
            "underlying": underlying.upper(),
            "contracts": len(contracts),
            "execution_permitted": False,
        }

    async def download_derivatives_context(
        self,
        *,
        provider: str = "mexc",
        market: str = "BTC-USDT",
        run_id: str | None = None,
        persist: bool = False,
    ) -> list[NormalizedDataRecord]:
        from data.derivatives_context import FundingRateCollector

        collector = FundingRateCollector(requester=self.request)
        records = await collector.collect(provider=provider, market=market, run_id=run_id)
        if persist:
            self._persist_raw_batch(records)
            self._persist_context_records(
                f"derivatives_{provider}_{market.split('-')[0]}",
                records,
            )
            self._database_upsert("derivatives_context", records)
        return records

    def reconcile_provider_series(
        self,
        series: Mapping[str, Iterable[NormalizedDataRecord]],
        *,
        source_priority: tuple[str, ...] = ("bitvavo", "kraken", "mexc"),
        numeric_tolerance: float = 1e-8,
    ) -> tuple[list[NormalizedDataRecord], list[dict[str, Any]]]:
        priority = {name: rank for rank, name in enumerate(source_priority)}
        buckets: dict[tuple[Any, ...], list[NormalizedDataRecord]] = {}
        for records in series.values():
            for record in records:
                key = (
                    record.canonical_market,
                    record.timestamp,
                    record.data_kind,
                    record.timeframe,
                )
                buckets.setdefault(key, []).append(record)
        chosen: list[NormalizedDataRecord] = []
        conflicts: list[dict[str, Any]] = []
        for key, records in sorted(buckets.items(), key=lambda item: str(item[0])):
            ordered = sorted(records, key=lambda item: priority.get(item.provider, 10_000))
            chosen.append(ordered[0])
            if len(records) > 1:
                closes = [
                    float(item.values["close"])
                    for item in records
                    if item.values.get("close") is not None
                ]
                if closes and max(closes) - min(closes) > numeric_tolerance:
                    conflicts.append(
                        {
                            "key": [str(value) for value in key],
                            "providers": [item.provider for item in records],
                            "values": closes,
                            "status": "CONFLICT",
                        }
                    )
        return chosen, conflicts

    def load_local_dataset(self, path: Path | str) -> pd.DataFrame:
        selected = Path(path)
        if selected.suffix not in {".csv", ".parquet"}:
            parquet = selected.with_suffix(".parquet")
            selected = parquet if parquet.is_file() else selected.with_suffix(".csv")
        if selected.suffix == ".parquet":
            return pd.read_parquet(selected)
        if selected.suffix == ".csv":
            return pd.read_csv(selected)
        raise ValueError("local datasets must be CSV or Parquet")

    def stale_records(
        self, records: Iterable[NormalizedDataRecord], maximum_age: timedelta
    ) -> list[NormalizedDataRecord]:
        cutoff = utc_now() - maximum_age
        return [
            record for record in records if (record.available_at or record.observed_at) < cutoff
        ]

    def write_manifest(self, path: Path | str, records: Iterable[NormalizedDataRecord]) -> Path:
        selected = list(records)
        return atomic_write_json(
            Path(path),
            {
                "record_count": len(selected),
                "providers": sorted({item.provider for item in selected}),
                "raw_hashes": [item.raw_hash for item in selected],
                "created_at": utc_now().isoformat(),
            },
        )

    def _cache_path(self, provider: str, market: str, timeframe: str, kind: str) -> Path:
        return self.settings.paths.cache_dir / provider / f"{market}_{timeframe}_{kind}"

    def _persist_dataset(self, path: Path, records: Iterable[NormalizedDataRecord]) -> None:
        frame = pd.DataFrame(
            [item.model_dump(mode="json", exclude={"raw_payload"}) for item in records]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._atomic_parquet(frame, path.with_suffix(".parquet"))
        except (ImportError, OSError):
            target = path.with_suffix(".csv")
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, target)

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp.parquet")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        return target

    def _persist_normalized_dataset(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        records: Iterable[NormalizedDataRecord],
        source_timeframe: str,
    ) -> Path | None:
        selected = list(records)
        if not selected:
            return None
        target = self.settings.paths.processed_data_dir / provider / market / f"{timeframe}.parquet"
        frame = pd.DataFrame(
            [
                {
                    **item.model_dump(mode="json", exclude={"raw_payload"}),
                    "source_provider": provider,
                    "source_timeframe": source_timeframe,
                    "quote_currency": market.split("-")[-1],
                    "source_classification": (
                        "PROVIDER_NATIVE" if item.data_kind == "ohlcv" else "RESAMPLED_FROM_NATIVE"
                    ),
                }
                for item in selected
            ]
        )
        return self._atomic_parquet(frame, target)

    def _persist_context_records(
        self, dataset: str, records: Iterable[NormalizedDataRecord]
    ) -> Path | None:
        selected = list(records)
        if not selected:
            return None
        safe_name = "".join(
            character if character.isalnum() or character in {"_", "-"} else "_"
            for character in dataset
        )
        target = self.settings.paths.context_data_dir / f"{safe_name}.parquet"
        rows = []
        for item in selected:
            reserved_columns = {
                "provider",
                "source_symbol",
                "canonical_market",
                "observation_time",
                "available_at",
                "observed_at",
                "retrieved_at",
                "revision_or_vintage",
                "point_in_time_status",
                "raw_hash",
                "data_kind",
            }
            values = {
                key: value
                for key, value in item.values.items()
                if key not in reserved_columns
                and not isinstance(value, (dict, list, tuple))
            }
            rows.append(
                {
                    "provider": item.provider,
                    "source_symbol": item.source_symbol,
                    "canonical_market": item.canonical_market,
                    "observation_time": item.timestamp,
                    "available_at": item.available_at or item.observed_at,
                    "observed_at": item.observed_at,
                    "retrieved_at": item.observed_at,
                    "revision_or_vintage": (
                        item.values.get("vintage_start")
                        or item.values.get("vintage_date")
                        or item.values.get("revision_status")
                    ),
                    "point_in_time_status": item.values.get(
                        "point_in_time_status",
                        "SOURCE_AVAILABLE_AT" if item.available_at is not None else "FORWARD_ONLY",
                    ),
                    "raw_hash": item.raw_hash,
                    "data_kind": item.data_kind,
                    **values,
                }
            )
        frame = pd.DataFrame(rows)
        if target.is_file():
            existing = pd.read_parquet(target)
            frame = pd.concat([existing, frame], ignore_index=True)
        for timestamp_column in (
            "observation_time",
            "available_at",
            "observed_at",
            "retrieved_at",
        ):
            if timestamp_column in frame:
                frame[timestamp_column] = pd.to_datetime(
                    frame[timestamp_column],
                    utc=True,
                    errors="raise",
                )
        key_columns = [
            name
            for name in (
                "provider",
                "source_symbol",
                "observation_time",
                "available_at",
                "raw_hash",
            )
            if name in frame
        ]
        frame = (
            frame.drop_duplicates(key_columns, keep="last")
            .sort_values(["available_at", "observation_time"])
            .reset_index(drop=True)
        )
        return self._atomic_parquet(frame, target)

    def _persist_raw_batch(self, records: Iterable[NormalizedDataRecord]) -> list[Path]:
        selected = list(records)
        if not selected:
            return []
        groups: dict[tuple[str, str, str, str, str], list[NormalizedDataRecord]] = {}
        for item in selected:
            key = (
                item.provider,
                item.data_kind,
                item.canonical_market,
                item.timeframe or "none",
                item.retrieval_run_id,
            )
            groups.setdefault(key, []).append(item)
        outputs: list[Path] = []
        for (provider, kind, market, timeframe, run_id), rows in groups.items():
            batch_hash = stable_hash([item.raw_hash for item in rows], length=64)
            # Immutable raw payloads are content-addressed. Including the
            # retrieval UUID in this filename created another physical file
            # whenever an unchanged macro response was polled again.
            target = (
                self.settings.paths.raw_data_dir
                / provider
                / kind
                / market
                / timeframe
                / f"{batch_hash[:24]}.parquet"
            )
            if not target.is_file():
                frame = pd.DataFrame(
                    [
                        {
                            "provider": item.provider,
                            "source_symbol": item.source_symbol,
                            "canonical_market": item.canonical_market,
                            "timeframe": item.timeframe,
                            "data_kind": item.data_kind,
                            "timestamp": item.timestamp,
                            "observed_at": item.observed_at,
                            "available_at": item.available_at,
                            "retrieval_run_id": item.retrieval_run_id,
                            "raw_hash": item.raw_hash,
                            "raw_payload": stable_json(item.raw_payload),
                        }
                        for item in rows
                    ]
                )
                self._atomic_parquet(frame, target)
            outputs.append(target)
            if self.database is not None:
                self.database.upsert_records(
                    "raw_manifests",
                    [
                        {
                            "external_id": batch_hash,
                            "provider": provider,
                            "market": market,
                            "timeframe": timeframe,
                            "timestamp": min(item.timestamp for item in rows),
                            "observed_at": max(item.observed_at for item in rows),
                            "available_at": max(
                                (item.available_at or item.observed_at for item in rows)
                            ),
                            "status": "IMMUTABLE_RAW_BATCH",
                            "raw_hash": batch_hash,
                            "path": str(target),
                            "record_count": len(rows),
                            "data_kind": kind,
                            "retrieval_run_id": run_id,
                        }
                    ],
                )
        return outputs

    def _update_watermark(
        self,
        *,
        provider: str,
        market: str,
        timeframe: str,
        data_kind: str,
        records: Iterable[NormalizedDataRecord],
        completed_ranges: Iterable[tuple[datetime, datetime]],
    ) -> None:
        selected = list(records)
        if not selected or self.database is None:
            return
        external_id = stable_hash([provider, market, timeframe, data_kind], length=64)
        ordered_timestamps = sorted({item.timestamp for item in selected})
        missing_ranges: list[list[str]] = []
        if timeframe != "1mo":
            interval = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            for previous, current in zip(
                ordered_timestamps,
                ordered_timestamps[1:],
                strict=False,
            ):
                if current - previous > interval:
                    missing_ranges.append(
                        [
                            (previous + interval).isoformat(),
                            (current - interval).isoformat(),
                        ]
                    )
        record = {
            "external_id": external_id,
            "provider": provider,
            "market": market,
            "timeframe": timeframe,
            "timestamp": max(item.timestamp for item in selected),
            "observed_at": max(item.observed_at for item in selected),
            "available_at": max(item.available_at or item.observed_at for item in selected),
            "status": (
                ProviderStatus.READY.value if not missing_ranges else ProviderStatus.PARTIAL.value
            ),
            "data_kind": data_kind,
            "earliest_stored_timestamp": min(item.timestamp for item in selected).isoformat(),
            "latest_stored_timestamp": max(item.timestamp for item in selected).isoformat(),
            "last_successful_cursor": max(item.timestamp for item in selected).isoformat(),
            "next_cursor": (
                max(item.timestamp for item in selected)
                + timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
            ).isoformat(),
            "completed_page_ranges": [
                [start.isoformat(), end.isoformat()] for start, end in completed_ranges
            ],
            "missing_ranges": missing_ranges,
            "retry_ranges": [],
            "source_hash": stable_hash([item.raw_hash for item in selected], length=64),
            "updated_at": utc_now().isoformat(),
        }
        self.database.upsert_records("data_watermarks", [record])

    def _database_upsert(self, table: str, records: Iterable[NormalizedDataRecord]) -> None:
        if self.database is None:
            return
        batch_size = self.settings.market_data.maximum_database_batch_size
        batch: list[dict[str, Any]] = []
        for item in records:
            batch.append(
                item.model_dump(
                    mode="json",
                    exclude={"raw_payload"},
                )
            )
            if len(batch) >= batch_size:
                self.database.upsert_records(table, batch)
                batch = []
        if batch:
            self.database.upsert_records(table, batch)

    @staticmethod
    def _records_from_frame(frame: pd.DataFrame) -> list[NormalizedDataRecord]:
        records: list[NormalizedDataRecord] = []
        for row in frame.to_dict(orient="records"):
            values = row.get("values", {})
            if isinstance(values, str):
                try:
                    values = json.loads(values.replace("'", '"'))
                except json.JSONDecodeError:
                    values = {}
            row["values"] = values
            records.append(NormalizedDataRecord.model_validate(row))
        return records

    async def _fear_and_greed(self, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        raw = await self._tracked(
            "alternative_me",
            lambda: self.request(
                "GET",
                f"{ALTERNATIVE_ME_REST}/fng/",
                {"limit": 0, "format": "json"},
                None,
            ),
        )
        records: list[NormalizedDataRecord] = []
        for row in raw.get("data", []):
            timestamp = _epoch(row["timestamp"], milliseconds=False)
            records.append(
                NormalizedDataRecord(
                    provider="alternative_me",
                    source_symbol="CRYPTO_FEAR_GREED",
                    canonical_market="BTC-EUR",
                    timestamp=timestamp,
                    observed_at=observed,
                    available_at=timestamp,
                    data_kind="fear_greed",
                    retrieval_run_id=run_id,
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values={
                        "fear_greed": int(row["value"]),
                        "classification": row.get("value_classification"),
                        "provider_timestamp": timestamp.isoformat(),
                        "point_in_time_status": "SOURCE_DAILY_TIMESTAMP",
                        "attribution": "Alternative.me Crypto Fear & Greed Index",
                    },
                )
            )
        return records

    async def _defillama(self, series: str, run_id: str) -> list[NormalizedDataRecord]:
        observed = utc_now()
        selected = series.casefold()
        if selected in {"stablecoins", "stablecoin", "stablecoin_history"}:
            raw = await self._tracked(
                "defillama",
                lambda: self.request(
                    "GET",
                    f"{DEFILLAMA_STABLECOINS_REST}/stablecoincharts/all",
                    None,
                    None,
                ),
            )
            rows = raw if isinstance(raw, list) else raw.get("data", [])
            records = []
            for row in rows:
                timestamp = _epoch(
                    row.get("date") or row.get("timestamp"),
                    milliseconds=False,
                )
                total = row.get("totalCirculatingUSD") or row.get("total_circulating_usd")
                if isinstance(total, dict):
                    total = total.get("peggedUSD") or sum(
                        float(value or 0) for value in total.values()
                    )
                records.append(
                    NormalizedDataRecord(
                        provider="defillama",
                        source_symbol="STABLECOINS_ALL",
                        canonical_market="BTC-EUR",
                        timestamp=timestamp,
                        observed_at=observed,
                        available_at=observed,
                        data_kind="onchain_context",
                        retrieval_run_id=run_id,
                        raw_hash=_raw_hash(row),
                        raw_payload=row,
                        values={
                            "stablecoin_market_cap": total,
                            "point_in_time_status": "FORWARD_ONLY",
                            "historical_reference_date": timestamp.isoformat(),
                        },
                    )
                )
            return records
        if selected in {"protocols", "tvl"}:
            raw = await self._tracked(
                "defillama",
                lambda: self.request("GET", f"{DEFILLAMA_REST}/protocols", None, None),
            )
            rows = raw if isinstance(raw, list) else []
            return [
                NormalizedDataRecord(
                    provider="defillama",
                    source_symbol=str(row.get("slug") or row.get("name")),
                    canonical_market="BTC-EUR",
                    timestamp=observed,
                    observed_at=observed,
                    available_at=observed,
                    data_kind="onchain_context",
                    retrieval_run_id=run_id,
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values={
                        "protocol": row.get("name"),
                        "category": row.get("category"),
                        "chain": row.get("chain"),
                        "tvl": row.get("tvl"),
                        "point_in_time_status": "FORWARD_ONLY",
                    },
                )
                for row in rows
            ]
        raise ValueError("unsupported DefiLlama dataset")

    async def _fred(
        self,
        series: str,
        start: datetime | None,
        end: datetime | None,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.fred_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        params: dict[str, Any] = {
            "series_id": series,
            "api_key": key.get_secret_value(),
            "file_type": "json",
            "output_type": 4,
            "realtime_start": (
                start.date().isoformat()
                if start
                else (observed - timedelta(days=365 * 5)).date().isoformat()
            ),
            "realtime_end": "9999-12-31",
        }
        if start:
            params["observation_start"] = start.date().isoformat()
        if end:
            params["observation_end"] = end.date().isoformat()
        raw = await self._tracked(
            "fred", lambda: self.request("GET", f"{FRED_REST}/series/observations", params, None)
        )
        records: list[NormalizedDataRecord] = []
        for row in raw.get("observations", []):
            if row.get("value") == ".":
                continue
            observation_date = _iso(row["date"])
            available = _iso(row.get("realtime_start", row["date"]))
            records.append(
                NormalizedDataRecord(
                    provider="fred",
                    source_symbol=series,
                    canonical_market="BTC-EUR",
                    timestamp=observation_date,
                    observed_at=observed,
                    available_at=available,
                    data_kind="macro_observation",
                    retrieval_run_id=run_id,
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values={
                        "series_id": series,
                        "value": row["value"],
                        "observation_date": observation_date.isoformat(),
                        "vintage_start": row.get("realtime_start"),
                        "vintage_end": row.get("realtime_end"),
                    },
                )
            )
        return records

    async def _eodhd(
        self,
        series: str,
        start: datetime | None,
        end: datetime | None,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.eodhd_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        params: dict[str, Any] = {"api_token": key.get_secret_value(), "fmt": "json"}
        if start:
            params["from"] = start.date().isoformat()
        if end:
            params["to"] = end.date().isoformat()
        selected = series.casefold()
        if selected in {"events", "economic_events", "economic-events"}:
            params.update({"limit": 1_000, "offset": 0})
            raw = await self._tracked(
                "eodhd",
                lambda: self.request("GET", f"{EODHD_REST}/economic-events", params, None),
            )
            records = []
            for row in raw if isinstance(raw, list) else []:
                event_time = _iso(str(row["date"]).replace(" ", "T"))
                records.append(
                    NormalizedDataRecord(
                        provider="eodhd",
                        source_symbol=str(row.get("type") or "ECONOMIC_EVENT"),
                        canonical_market="BTC-EUR",
                        timestamp=event_time,
                        observed_at=observed,
                        available_at=observed,
                        data_kind="economic_event",
                        retrieval_run_id=run_id,
                        raw_hash=_raw_hash(row),
                        raw_payload=row,
                        values={
                            **row,
                            "event_at": event_time.isoformat(),
                            "point_in_time_status": "FORWARD_ONLY"
                            if event_time < observed
                            else "KNOWN_SCHEDULE_AT_RETRIEVAL",
                        },
                    )
                )
            return records
        if selected.startswith("macro:"):
            indicator = series.split(":", 1)[1]
            raw = await self._tracked(
                "eodhd",
                lambda: self.request(
                    "GET",
                    f"{EODHD_REST}/macro-indicator/USA",
                    {
                        "api_token": key.get_secret_value(),
                        "fmt": "json",
                        "indicator": indicator,
                    },
                    None,
                ),
            )
        else:
            raw = await self._tracked(
                "eodhd",
                lambda: self.request("GET", f"{EODHD_REST}/eod/{series}", params, None),
            )
        return [
            NormalizedDataRecord(
                provider="eodhd",
                source_symbol=series,
                canonical_market="BTC-EUR",
                timestamp=_iso(row.get("date") or row.get("Date")),
                observed_at=observed,
                available_at=observed,
                data_kind="macro_observation",
                retrieval_run_id=run_id,
                raw_hash=_raw_hash(row),
                raw_payload=row,
                values={
                    **row,
                    "point_in_time_status": "FORWARD_ONLY",
                },
            )
            for row in raw
            if row.get("date") or row.get("Date")
        ]

    async def _sec(self, cik: str, run_id: str) -> list[NormalizedDataRecord]:
        user_agent = self.settings.providers.sec_user_agent
        if not user_agent:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        normalized_cik = str(cik).lstrip("0").zfill(10)
        raw = await self._tracked(
            "sec",
            lambda: self.request(
                "GET",
                f"{SEC_REST}/submissions/CIK{normalized_cik}.json",
                None,
                {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            ),
        )
        recent = raw.get("filings", {}).get("recent", {})
        records: list[NormalizedDataRecord] = []
        crypto_terms = ("bitcoin", "ethereum", "crypto", "digital asset", "custody")
        for index, accession in enumerate(recent.get("accessionNumber", [])):
            primary = recent.get("primaryDocument", [""])[index]
            description = recent.get("primaryDocDescription", [""])[index]
            entities = tuple(term for term in crypto_terms if term in description.casefold())
            accepted = recent.get("acceptanceDateTime", [None])[index]
            timestamp = _iso(accepted) if accepted else observed
            url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(normalized_cik)}/{accession.replace('-', '')}/{primary}"
            )
            row = {
                "accession": accession,
                "form": recent.get("form", [""])[index],
                "accepted_at": timestamp.isoformat(),
                "source_url": url,
                "crypto_entities": entities,
                "category": "crypto_regulatory_filing" if entities else "filing",
                "point_in_time_status": "SOURCE_ACCEPTED_TIME" if accepted else "OBSERVED_ONLY",
            }
            records.append(
                NormalizedDataRecord(
                    provider="sec",
                    source_symbol=normalized_cik,
                    canonical_market="BTC-EUR",
                    timestamp=timestamp,
                    observed_at=observed,
                    available_at=timestamp if accepted else observed,
                    data_kind="sec_filing",
                    retrieval_run_id=run_id,
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values=row,
                )
            )
        return records

    async def _cmc_quotes(
        self,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.coinmarketcap_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        params: dict[str, Any] = {"symbol": symbol, "convert": "EUR"}
        if start:
            params["time_start"] = start.isoformat()
        if end:
            params["time_end"] = end.isoformat()
        raw = await self._tracked(
            "coinmarketcap",
            lambda: self.request(
                "GET",
                f"{CMC_REST}/cryptocurrency/quotes/historical",
                params,
                {"X-CMC_PRO_API_KEY": key.get_secret_value()},
            ),
        )
        quotes = raw.get("data", {}).get("quotes", [])
        return [
            NormalizedDataRecord(
                provider="coinmarketcap",
                source_symbol=symbol,
                canonical_market=f"{symbol.upper()}-EUR",
                timestamp=_iso(row["timestamp"]),
                observed_at=observed,
                available_at=observed,
                data_kind="sampled_quote",
                retrieval_run_id=run_id,
                raw_hash=_raw_hash(row),
                raw_payload=row,
                values={
                    **row,
                    "point_in_time_status": "FORWARD_ONLY",
                    "historical_reference_date": row["timestamp"],
                },
            )
            for row in quotes
        ]

    async def _cmc_global(self, run_id: str) -> list[NormalizedDataRecord]:
        key = self.settings.providers.coinmarketcap_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        raw = await self._tracked(
            "coinmarketcap",
            lambda: self.request(
                "GET",
                f"{CMC_REST}/global-metrics/quotes/latest",
                {"convert": "EUR"},
                {"X-CMC_PRO_API_KEY": key.get_secret_value()},
            ),
        )
        data = raw.get("data", {})
        quote = data.get("quote", {}).get("EUR", {})
        response_credit_count = raw.get("status", {}).get("credit_count")
        values = {
            "total_market_cap": quote.get("total_market_cap"),
            "total_volume_24h": quote.get("total_volume_24h"),
            "btc_dominance": (
                float(data["btc_dominance"]) / 100
                if data.get("btc_dominance") is not None
                else None
            ),
            "eth_dominance": (
                float(data["eth_dominance"]) / 100
                if data.get("eth_dominance") is not None
                else None
            ),
            "stablecoin_dominance": data.get("stablecoin_dominance"),
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "source_quality": "SAMPLED_CONTEXT_NOT_EXECUTION_DATA",
            "response_credit_count": response_credit_count,
        }
        return [
            NormalizedDataRecord(
                provider="coinmarketcap",
                source_symbol="GLOBAL_METRICS",
                canonical_market="BTC-EUR",
                timestamp=_iso(data.get("last_updated", observed.isoformat())),
                observed_at=observed,
                available_at=observed,
                data_kind="macro_observation",
                retrieval_run_id=run_id,
                raw_hash=_raw_hash(raw),
                raw_payload=raw,
                values=values,
            )
        ]

    async def _cmc_metadata(self, market: str | None, run_id: str) -> list[NormalizedDataRecord]:
        key = self.settings.providers.coinmarketcap_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        symbol = market.split("-")[0] if market else None
        params = {"symbol": symbol} if symbol else {"limit": 5_000}
        raw = await self._tracked(
            "coinmarketcap",
            lambda: self.request(
                "GET",
                f"{CMC_REST}/cryptocurrency/map",
                params,
                {"X-CMC_PRO_API_KEY": key.get_secret_value()},
            ),
        )
        return [
            NormalizedDataRecord(
                provider="coinmarketcap",
                source_symbol=row["symbol"],
                canonical_market=f"{row['symbol']}-EUR",
                timestamp=observed,
                observed_at=observed,
                available_at=observed,
                data_kind="market_metadata",
                retrieval_run_id=run_id,
                raw_hash=_raw_hash(row),
                raw_payload=row,
                values=row,
            )
            for row in raw.get("data", [])
        ]

    async def _cmc_rankings(
        self,
        *,
        limit: int,
        convert: str,
        run_id: str,
    ) -> list[NormalizedDataRecord]:
        key = self.settings.providers.coinmarketcap_api_key
        if key is None:
            raise PermissionError("SKIPPED_MISSING_CREDENTIALS")
        observed = utc_now()
        raw = await self._tracked(
            "coinmarketcap",
            lambda: self.request(
                "GET",
                f"{CMC_REST}/cryptocurrency/listings/latest",
                {
                    "start": 1,
                    "limit": limit,
                    "convert": convert,
                    "sort": "market_cap",
                    "sort_dir": "desc",
                    "cryptocurrency_type": "all",
                },
                {"X-CMC_PRO_API_KEY": key.get_secret_value()},
            ),
        )
        records: list[NormalizedDataRecord] = []
        response_credit_count = raw.get("status", {}).get("credit_count")
        for row in raw.get("data", []):
            quote = row.get("quote", {}).get(convert, {})
            provider_timestamp = _iso(
                row.get("last_updated")
                or quote.get("last_updated")
                or raw.get("status", {}).get("timestamp")
                or observed.isoformat()
            )
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            values = {
                "cmc_rank": row.get("cmc_rank"),
                "cmc_id": row.get("id"),
                "symbol": symbol,
                "name": row.get("name"),
                "slug": row.get("slug"),
                "market_cap": quote.get("market_cap"),
                "price": quote.get("price"),
                "percent_change_1h": quote.get("percent_change_1h"),
                "percent_change_24h": quote.get("percent_change_24h"),
                "percent_change_7d": quote.get("percent_change_7d"),
                "market_cap_dominance": quote.get("market_cap_dominance"),
                "fully_diluted_market_cap": quote.get("fully_diluted_market_cap"),
                "circulating_supply": row.get("circulating_supply"),
                "total_supply": row.get("total_supply"),
                "maximum_supply": row.get("max_supply"),
                "volume_24h": quote.get("volume_24h"),
                "provider_timestamp": provider_timestamp.isoformat(),
                "tags": row.get("tags") or [],
                "platform": row.get("platform"),
                "is_active": row.get("is_active"),
                "response_credit_count": response_credit_count,
            }
            records.append(
                NormalizedDataRecord(
                    provider="coinmarketcap",
                    source_symbol=str(row.get("id") or symbol),
                    canonical_market=f"{symbol}-EUR",
                    timestamp=provider_timestamp,
                    observed_at=observed,
                    available_at=observed,
                    data_kind="universe_ranking",
                    retrieval_run_id=run_id,
                    raw_hash=_raw_hash(row),
                    raw_payload=row,
                    values=values,
                )
            )
        return sorted(
            records,
            key=lambda record: int(record.values.get("cmc_rank") or 10**9),
        )


class ContinuousDataService:
    """Cooperative, single-instance scheduler for incremental data collection."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Any | None = None,
        heartbeat_seconds: float = 10.0,
        service_id: str = "continuous-data-service",
        mode: str = "research",
    ) -> None:
        self.settings = settings
        self.database = database
        self.heartbeat_seconds = heartbeat_seconds
        self.service_id = service_id
        self.mode = mode
        self.state = "STOPPED"
        self.current_cycle = 0
        self.last_completed_operation: str | None = None
        self.next_scheduled_operation: str | None = None
        self.active_candidate: str | None = None
        self.open_position_count = 0
        self.kill_switch_state = "INACTIVE"
        self._wake = asyncio.Event()
        self._stop_requested = False
        self._drain_requested = False
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._lock_owner_token = uuid.uuid4().hex
        self.lock_path = settings.paths.checkpoints_dir / "data_service.lock"
        self.heartbeat_path = settings.paths.checkpoints_dir / f"{service_id}_heartbeat.json"
        self.control_path = settings.paths.checkpoints_dir / f"{service_id}_control.json"

    @staticmethod
    def _process_alive(process_id: int) -> bool:
        if process_id <= 0:
            return False
        if os.name == "nt":
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information,
                False,
                process_id,
            )
            if not handle:
                # Access denied means the process exists but cannot be queried.
                return ctypes.get_last_error() == 5
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                    handle,
                    ctypes.byref(exit_code),
                ):
                    return False
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        try:
            os.kill(process_id, 0)
        except OSError:
            return False
        return True

    @classmethod
    def inspect_lock_path(cls, lock_path: Path) -> dict[str, Any]:
        """Inspect a lock without treating recoverable stale metadata as busy."""

        if not lock_path.is_file():
            return {
                "available": True,
                "exists": False,
                "stale": False,
                "reason_code": "LOCK_AVAILABLE",
                "owner": None,
            }
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            process_id = int(owner["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {
                "available": True,
                "exists": True,
                "stale": True,
                "reason_code": "INVALID_LOCK_METADATA_RECOVERABLE",
                "owner": None,
            }
        alive = cls._process_alive(process_id)
        return {
            "available": not alive,
            "exists": True,
            "stale": not alive,
            "reason_code": (
                "LOCK_HELD_BY_LIVE_PROCESS" if alive else "STALE_PROCESS_LOCK_RECOVERABLE"
            ),
            "owner": owner,
        }

    @classmethod
    def recover_stale_lock_path(cls, lock_path: Path) -> dict[str, Any]:
        """Archive a provably stale lock; never remove a live owner's lock."""

        inspection = cls.inspect_lock_path(lock_path)
        if not inspection["exists"]:
            return inspection | {"recovered": False}
        if not inspection["available"] or not inspection["stale"]:
            raise RuntimeError("DATA_SERVICE_LIVE_LOCK_CANNOT_BE_RECOVERED")
        archive = lock_path.with_name(
            f"{lock_path.name}.stale.{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        )
        os.replace(lock_path, archive)
        return inspection | {
            "recovered": True,
            "archive_path": str(archive),
            "reason_code": "STALE_LOCK_ARCHIVED",
        }

    def _acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_file():
            inspection = self.inspect_lock_path(self.lock_path)
            if inspection["available"] and inspection["stale"]:
                raise RuntimeError("DATA_SERVICE_STALE_LOCK_REQUIRES_EXPLICIT_RECOVERY")
            raise RuntimeError("DATA_SERVICE_SINGLE_INSTANCE_LOCKED")
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError("DATA_SERVICE_SINGLE_INSTANCE_LOCKED") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "started_at": utc_now().isoformat(),
                    "owner_token": self._lock_owner_token,
                    "service_id": self.service_id,
                    "mode": self.mode,
                    "hostname": socket.gethostname(),
                    "executable": sys.executable,
                    "command": sys.argv,
                    "lock_version": 2,
                },
                handle,
            )

    def _release_lock(self) -> None:
        if not self.lock_path.is_file():
            return
        try:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if owner.get("owner_token") == self._lock_owner_token:
            self.lock_path.unlink(missing_ok=True)

    def _heartbeat(self, *, reason_code: str) -> None:
        payload = {
            "external_id": self.service_id,
            "service_id": self.service_id,
            "mode": self.mode,
            "status": self.state,
            "state": self.state,
            "reason_code": reason_code,
            "pid": os.getpid(),
            "started_at": (
                json.loads(self.lock_path.read_text(encoding="utf-8")).get("started_at")
                if self.lock_path.is_file()
                else None
            ),
            "heartbeat_at": utc_now().isoformat(),
            "heartbeat": utc_now().isoformat(),
            "current_cycle": self.current_cycle,
            "last_completed_operation": self.last_completed_operation,
            "next_scheduled_operation": self.next_scheduled_operation,
            "queue_size": len(self._active_tasks),
            "active_candidate": self.active_candidate,
            "open_position_count": self.open_position_count,
            "kill_switch_state": self.kill_switch_state,
            "active_tasks": len(self._active_tasks),
            "stop_requested": self._stop_requested,
            "drain_requested": self._drain_requested,
            "live_orders": 0,
        }
        atomic_write_json(self.heartbeat_path, payload)
        if self.database is not None:
            self.database.upsert_records("data_service_state", [payload])

    async def start(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        interval_seconds: float,
        once: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("continuous interval must be positive")
        self._acquire_lock()
        self.state = "RUNNING"
        self._stop_requested = False
        self._drain_requested = False
        try:
            while not self._stop_requested:
                self._apply_control_request()
                if self.state == "PAUSED":
                    self._heartbeat(reason_code="SERVICE_PAUSED")
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(),
                            timeout=self.heartbeat_seconds,
                        )
                    except TimeoutError:
                        pass
                    continue
                task = asyncio.create_task(
                    operation(),
                    name=f"data-service-operation-{uuid.uuid4().hex[:8]}",
                )
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_while_active(task),
                    name=(
                        "data-service-heartbeat-"
                        f"{uuid.uuid4().hex[:8]}"
                    ),
                )
                self._active_tasks.add(task)
                try:
                    await task
                finally:
                    self._active_tasks.discard(task)
                    heartbeat_task.cancel()
                    await asyncio.gather(
                        heartbeat_task,
                        return_exceptions=True,
                    )
                self.current_cycle += 1
                self.last_completed_operation = "OPERATIONAL_CYCLE"
                self._heartbeat(reason_code="INCREMENTAL_CYCLE_COMPLETE")
                if once or self._drain_requested or self._stop_requested:
                    break
                self._wake.clear()
                wait_deadline = (
                    asyncio.get_running_loop().time()
                    + interval_seconds
                )
                while (
                    not self._stop_requested
                    and not self._drain_requested
                    and self.state != "PAUSED"
                ):
                    self._apply_control_request()
                    remaining = (
                        wait_deadline
                        - asyncio.get_running_loop().time()
                    )
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(),
                            timeout=min(1.0, remaining),
                        )
                        break
                    except TimeoutError:
                        continue
            self.state = "DRAINED" if self._drain_requested else "STOPPED"
            self._heartbeat(
                reason_code=("SERVICE_DRAINED" if self._drain_requested else "SERVICE_STOPPED")
            )
        finally:
            for task in tuple(self._active_tasks):
                task.cancel()
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
            self._release_lock()

    async def _heartbeat_while_active(
        self,
        operation: asyncio.Task[Any],
    ) -> None:
        while not operation.done():
            self._heartbeat(
                reason_code="OPERATIONAL_CYCLE_ACTIVE"
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(operation),
                    timeout=self.heartbeat_seconds,
                )
            except TimeoutError:
                continue

    def _apply_control_request(self) -> None:
        if not self.control_path.is_file():
            return
        try:
            payload = json.loads(self.control_path.read_text(encoding="utf-8"))
            action = str(payload.get("action") or "").upper()
        except (OSError, ValueError, TypeError):
            action = ""
        self.control_path.unlink(missing_ok=True)
        if action == "PAUSE" and self.state == "RUNNING":
            self.pause()
        elif action == "RESUME" and self.state == "PAUSED":
            self.resume()
        elif action == "DRAIN":
            self.drain()
        elif action == "STOP":
            self.stop()

    def pause(self) -> None:
        if self.state != "RUNNING":
            raise RuntimeError("data service is not running")
        self.state = "PAUSED"
        self._wake.set()
        self._heartbeat(reason_code="PAUSE_REQUESTED")

    def resume(self) -> None:
        if self.state != "PAUSED":
            raise RuntimeError("data service is not paused")
        self.state = "RUNNING"
        self._wake.set()
        self._heartbeat(reason_code="RESUME_REQUESTED")

    def drain(self) -> None:
        self._drain_requested = True
        self.state = "DRAINING"
        self._wake.set()
        self._heartbeat(reason_code="DRAIN_REQUESTED")

    def stop(self) -> None:
        self._stop_requested = True
        self._wake.set()
        self._heartbeat(reason_code="STOP_REQUESTED")

    def status(self) -> dict[str, Any]:
        if self.heartbeat_path.is_file():
            return json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
        return {
            "status": self.state,
            "reason_code": "SERVICE_NOT_STARTED",
            "active_tasks": len(self._active_tasks),
            "live_orders": 0,
        }


__all__ = [
    "BitvavoAdapter",
    "ContinuousDataService",
    "DataLoader",
    "HISTORY_TARGETS",
    "KrakenAdapter",
    "MexcAdapter",
    "PROVIDER_NATIVE_TIMEFRAMES",
    "PublicHttpClient",
    "RESAMPLE_SOURCE",
]
