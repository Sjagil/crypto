#!/usr/bin/env python3
"""
multi_venue_crypto_data_downloader.py

Data-only crypto market-data downloader for:
- Bitvavo public OHLCV
- Kraken public OHLCVT (REST recent; optional official bulk ZIP import)
- CoinMarketCap Pro OHLCV + historical quote series

Safety:
- GET requests only.
- No order, balance, withdrawal, or trading endpoints.
- Ignores all trade API keys and LIVE_TRADING_ALLOWED.
- Uses only public market-data endpoints plus the CoinMarketCap data key.

Python: 3.10+
Dependencies:
    pip install requests pandas pyarrow

Examples:
    python multi_venue_crypto_data_downloader.py self-test

    python multi_venue_crypto_data_downloader.py download ^
      --env-file .env ^
      --providers all ^
      --symbols BTC,ETH,SOL,LINK,ADA ^
      --quote EUR ^
      --start 2019-01-01 ^
      --timeframes all ^
      --history-policy smart ^
      --output-dir output/market_data/multi_venue_ohlcv ^
      --resume

Kraken full-history:
    Download the official Kraken OHLCVT ZIP, then pass:
      --kraken-bulk-zip C:\\path\\to\\Kraken_OHLCVT.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install with: pip install pandas") from exc

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with: pip install requests") from exc


VERSION = "1.0.0"
USER_AGENT = f"worldmonitor-multi-venue-downloader/{VERSION}"

BITVAVO_BASE = "https://api.bitvavo.com/v2"
KRAKEN_BASE = "https://api.kraken.com/0/public"
CMC_BASE = "https://pro-api.coinmarketcap.com"

BITVAVO_INTERVALS = (
    "1m", "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "8h", "12h", "1d", "1W", "1M",
)
KRAKEN_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1W": 10080,
    "15d": 21600,
}
KRAKEN_BULK_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "12h": 720,
    "1d": 1440,
}

# CoinMarketCap's OHLCV endpoint provides hourly and daily OHLCV periods.
CMC_OHLCV_INTERVALS = ("1h", "1d")
# CMC Quotes Historical provides sampled market quotes at these intervals.
CMC_QUOTE_INTERVALS = (
    "5m", "10m", "15m", "30m", "45m",
    "1h", "2h", "3h", "4h", "6h", "12h",
    "1d", "2d", "3d", "7d", "14d", "15d",
    "30d", "60d", "90d", "365d",
    "weekly", "monthly", "yearly",
)

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "45m": 2700,
    "1h": 3600,
    "2h": 7200,
    "3h": 10800,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "2d": 172800,
    "3d": 259200,
    "7d": 604800,
    "14d": 1209600,
    "15d": 1296000,
    "30d": 2592000,
    "60d": 5184000,
    "90d": 7776000,
    "365d": 31536000,
    "1W": 604800,
    "weekly": 604800,
}

DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "LINK", "ADA")
FIAT_AND_STABLE = {"EUR", "USD", "USDC", "USDT", "DAI", "GBP"}

STANDARD_COLUMNS = (
    "timestamp", "open", "high", "low", "close", "volume",
    "vwap", "trades", "market_cap", "provider", "symbol",
    "market", "quote", "interval", "data_kind", "bar_source",
    "base_interval", "is_closed",
)


@dataclass(frozen=True)
class CoverageRecord:
    provider: str
    data_kind: str
    symbol: str
    market: str
    interval: str
    rows: int
    first_timestamp: str | None
    last_timestamp: str | None
    gap_count: int | None
    estimated_missing_bars: int | None
    max_gap_bars: int | None
    output_path: str
    sha256: str
    status: str
    note: str = ""


@dataclass(frozen=True)
class FailureRecord:
    provider: str
    data_kind: str
    symbol: str
    market: str
    interval: str
    error_type: str
    message: str
    timestamp_utc: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | pd.Timestamp | str | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def parse_utc(value: str | None, *, default: datetime | None = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("A date/time value is required")
        return default
    if value.strip().lower() in {"now", "today"}:
        return utc_now()
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()


def load_env_file(path: Path) -> None:
    """Minimal .env parser. Existing environment variables win."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def setup_logger(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_venue_downloader")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "live_download.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


class HttpClient:
    def __init__(
        self,
        *,
        timeout: float,
        retries: int,
        backoff: float,
        min_delay: float,
        logger: logging.Logger,
    ) -> None:
        self.timeout = timeout
        self.min_delay = min_delay
        self.logger = logger
        self._last_request_at: dict[str, float] = {}
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff,
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))

    def _throttle(self, host_key: str) -> None:
        previous = self._last_request_at.get(host_key)
        if previous is not None:
            remaining = self.min_delay - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at[host_key] = time.monotonic()

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        host_key: str,
    ) -> Any:
        self._throttle(host_key)
        response = self.session.get(
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            headers=dict(headers or {}),
            timeout=self.timeout,
        )
        if not response.ok:
            body = response.text[:2000]
            raise RuntimeError(
                f"HTTP {response.status_code} for {response.url}: {body}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from {response.url}: {response.text[:1000]}") from exc
        return payload

    def get_bytes(self, url: str, *, host_key: str) -> bytes:
        self._throttle(host_key)
        response = self.session.get(url, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} for {response.url}")
        return response.content


def normalize_symbols(raw: str | None) -> list[str]:
    if raw:
        values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    else:
        tracked = os.environ.get("WM_TRACKED_WALLET_ASSETS", "")
        values = [item.strip().upper() for item in tracked.split(",") if item.strip()]
        values = [item for item in values if item not in FIAT_AND_STABLE]
        if not values:
            values = list(DEFAULT_SYMBOLS)
    aliases = {"XBT": "BTC"}
    result: list[str] = []
    for value in values:
        base = value.split("-")[0].split("/")[0]
        base = aliases.get(base, base)
        if base not in result:
            result.append(base)
    return result


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def interval_to_seconds(interval: str) -> int | None:
    return INTERVAL_SECONDS.get(interval)


def effective_start_for_policy(
    requested: datetime,
    interval: str,
    policy: str,
    end: datetime,
) -> datetime:
    if policy == "full":
        return requested
    if policy == "recent":
        lookback = 90 if interval.endswith(("m", "h")) else 730
        return max(requested, end - timedelta(days=lookback))
    # Smart policy: controls the explosive size of very small bars.
    smart_days = {
        "1m": 180,
        "5m": 730,
        "10m": 730,
        "15m": 1460,
        "30m": 1825,
        "45m": 1825,
    }
    days = smart_days.get(interval)
    return max(requested, end - timedelta(days=days)) if days else requested


def estimated_bar_count(start: datetime, end: datetime, interval: str) -> int | None:
    seconds = interval_to_seconds(interval)
    if seconds is None or end <= start:
        return None
    return max(0, math.ceil((end - start).total_seconds() / seconds))


def floor_timestamp_ms(value: datetime, interval: str) -> int:
    seconds = interval_to_seconds(interval)
    ts = int(value.timestamp())
    if seconds:
        return (ts // seconds) * seconds * 1000
    if interval == "1M":
        return int(datetime(value.year, value.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    raise ValueError(f"Cannot align interval {interval}")


def drop_open_candles(frame: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    now_ts = pd.Timestamp(now or utc_now())
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    if interval == "1M":
        starts = frame["timestamp"].dt.tz_convert("UTC")
        ends = starts + pd.offsets.MonthBegin(1)
        return frame.loc[ends <= now_ts].copy()
    seconds = interval_to_seconds(interval)
    if seconds is None:
        return frame
    return frame.loc[frame["timestamp"] + pd.to_timedelta(seconds, unit="s") <= now_ts].copy()


def standardize_frame(
    frame: pd.DataFrame,
    *,
    provider: str,
    symbol: str,
    market: str,
    quote: str,
    interval: str,
    data_kind: str,
    bar_source: str = "native",
    base_interval: str | None = None,
    closed_only: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vwap", "trades", "market_cap"):
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["timestamp"])
    if data_kind == "ohlcv":
        result = result.dropna(subset=["open", "high", "low", "close"])
        valid = (
            (result["high"] >= result[["open", "close", "low"]].max(axis=1))
            & (result["low"] <= result[["open", "close", "high"]].min(axis=1))
            & (result["open"] > 0)
            & (result["high"] > 0)
            & (result["low"] > 0)
            & (result["close"] > 0)
        )
        result = result.loc[valid].copy()

    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    result["provider"] = provider
    result["symbol"] = symbol
    result["market"] = market
    result["quote"] = quote
    result["interval"] = interval
    result["data_kind"] = data_kind
    result["bar_source"] = bar_source
    result["base_interval"] = base_interval or interval
    result["is_closed"] = True
    if closed_only and data_kind == "ohlcv":
        result = drop_open_candles(result, interval)
    return result.loc[:, STANDARD_COLUMNS].reset_index(drop=True)


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


def merge_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    if merged.empty:
        return merged
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    return merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def write_frame_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    also_csv: bool,
    logger: logging.Logger,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        primary = path
    except Exception as exc:
        logger.warning("[WRITE] Parquet unavailable for %s (%s); falling back to CSV", path, exc)
        primary = path.with_suffix(".csv")
        tmp = primary.with_suffix(primary.suffix + ".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, primary)
    if also_csv and primary.suffix != ".csv":
        csv_path = primary.with_suffix(".csv")
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        frame.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, csv_path)
    return primary


def audit_gaps(frame: pd.DataFrame, interval: str) -> tuple[int | None, int | None, int | None]:
    seconds = interval_to_seconds(interval)
    if frame.empty or len(frame) < 2 or seconds is None:
        return None, None, None
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).sort_values()
    deltas = timestamps.diff().dt.total_seconds().dropna()
    gap_bars = (deltas / seconds).round().astype(int) - 1
    positive = gap_bars[gap_bars > 0]
    return int(len(positive)), int(positive.sum()) if len(positive) else 0, int(positive.max()) if len(positive) else 0


def coverage_record(
    frame: pd.DataFrame,
    path: Path,
    *,
    provider: str,
    data_kind: str,
    symbol: str,
    market: str,
    interval: str,
    status: str,
    note: str = "",
) -> CoverageRecord:
    gap_count, missing, max_gap = audit_gaps(frame, interval)
    return CoverageRecord(
        provider=provider,
        data_kind=data_kind,
        symbol=symbol,
        market=market,
        interval=interval,
        rows=int(len(frame)),
        first_timestamp=iso_utc(frame["timestamp"].min()) if not frame.empty else None,
        last_timestamp=iso_utc(frame["timestamp"].max()) if not frame.empty else None,
        gap_count=gap_count,
        estimated_missing_bars=missing,
        max_gap_bars=max_gap,
        output_path=str(path),
        sha256=sha256_file(path) if path.exists() else "",
        status=status,
        note=note,
    )


def output_path(
    output_dir: Path,
    *,
    provider: str,
    data_kind: str,
    symbol: str,
    quote: str,
    interval: str,
) -> Path:
    safe_market = f"{symbol}-{quote}".replace("/", "-")
    return output_dir / provider / data_kind / safe_market / f"{interval}.parquet"


class BitvavoDownloader:
    def __init__(self, client: HttpClient, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger
        self._markets: set[str] | None = None

    def markets(self) -> set[str]:
        if self._markets is None:
            payload = self.client.get_json(
                f"{BITVAVO_BASE}/markets",
                host_key="bitvavo",
            )
            self._markets = {
                str(item.get("market", "")).upper()
                for item in payload
                if isinstance(item, Mapping)
            }
        return self._markets

    def download(
        self,
        *,
        symbol: str,
        quote: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        market = f"{symbol}-{quote}".upper()
        if market not in self.markets():
            raise ValueError(f"Bitvavo market not found: {market}")
        if interval not in BITVAVO_INTERVALS:
            raise ValueError(f"Unsupported Bitvavo interval: {interval}")

        rows: list[list[Any]] = []
        cursor_end_ms = floor_timestamp_ms(end, interval)
        start_ms = floor_timestamp_ms(start, interval)
        previous_oldest: int | None = None
        page = 0

        while cursor_end_ms >= start_ms:
            page += 1
            payload = self.client.get_json(
                f"{BITVAVO_BASE}/{market}/candles",
                params={"interval": interval, "limit": 1440, "end": cursor_end_ms},
                host_key="bitvavo",
            )
            if not isinstance(payload, list) or not payload:
                break
            parsed = [item for item in payload if isinstance(item, list) and len(item) >= 6]
            if not parsed:
                break
            rows.extend(parsed)
            oldest = min(int(item[0]) for item in parsed)
            newest = max(int(item[0]) for item in parsed)
            self.logger.info(
                "[BITVAVO] %-12s %-3s page=%4d rows=%7d oldest=%s newest=%s",
                market,
                interval,
                page,
                len(rows),
                pd.to_datetime(oldest, unit="ms", utc=True),
                pd.to_datetime(newest, unit="ms", utc=True),
            )
            if oldest <= start_ms:
                break
            if previous_oldest is not None and oldest >= previous_oldest:
                self.logger.warning("[BITVAVO] no pagination progress for %s %s", market, interval)
                break
            previous_oldest = oldest
            if interval == "1M":
                old_dt = pd.to_datetime(oldest, unit="ms", utc=True)
                cursor_end_ms = int((old_dt - pd.offsets.MonthBegin(1)).timestamp() * 1000)
            else:
                cursor_end_ms = oldest - (interval_to_seconds(interval) or 1) * 1000

        frame = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.loc[
            (frame["timestamp"] >= pd.Timestamp(start))
            & (frame["timestamp"] <= pd.Timestamp(end))
        ]
        return standardize_frame(
            frame,
            provider="bitvavo",
            symbol=symbol,
            market=market,
            quote=quote,
            interval=interval,
            data_kind="ohlcv",
        )


class KrakenDownloader:
    def __init__(self, client: HttpClient, logger: logging.Logger) -> None:
        self.client = client
        self.logger = logger
        self._pair_map: dict[tuple[str, str], str] | None = None

    @staticmethod
    def _kraken_alias(value: str) -> str:
        return {"BTC": "XBT", "DOGE": "XDG"}.get(value.upper(), value.upper())

    def pair_map(self) -> dict[tuple[str, str], str]:
        if self._pair_map is not None:
            return self._pair_map
        payload = self.client.get_json(
            f"{KRAKEN_BASE}/AssetPairs",
            params={"assetVersion": 1},
            host_key="kraken",
        )
        errors = payload.get("error", []) if isinstance(payload, Mapping) else []
        if errors:
            raise RuntimeError(f"Kraken AssetPairs error: {errors}")
        result = payload.get("result", {})
        mapping: dict[tuple[str, str], str] = {}
        for key, item in result.items():
            if not isinstance(item, Mapping):
                continue
            wsname = str(item.get("wsname") or key).replace("XBT", "BTC").replace("XDG", "DOGE")
            if "/" in wsname:
                base, quote = wsname.split("/", 1)
                mapping[(base.upper(), quote.upper())] = str(item.get("altname") or key)
        self._pair_map = mapping
        return mapping

    def resolve_pair(self, symbol: str, quote: str) -> str:
        mapping = self.pair_map()
        key = (symbol.upper(), quote.upper())
        if key in mapping:
            return mapping[key]
        alias_key = (self._kraken_alias(symbol), quote.upper())
        if alias_key in mapping:
            return mapping[alias_key]
        fallback = f"{self._kraken_alias(symbol)}{quote.upper()}"
        self.logger.warning("[KRAKEN] pair not found in map; trying fallback %s", fallback)
        return fallback

    def download_recent(
        self,
        *,
        symbol: str,
        quote: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if interval not in KRAKEN_INTERVAL_MINUTES:
            raise ValueError(f"Unsupported Kraken REST interval: {interval}")
        pair = self.resolve_pair(symbol, quote)
        payload = self.client.get_json(
            f"{KRAKEN_BASE}/OHLC",
            params={
                "pair": pair,
                "interval": KRAKEN_INTERVAL_MINUTES[interval],
                "since": int(start.timestamp()),
                "assetVersion": 1,
            },
            host_key="kraken",
        )
        errors = payload.get("error", []) if isinstance(payload, Mapping) else []
        if errors:
            raise RuntimeError(f"Kraken OHLC error for {pair}: {errors}")
        result = payload.get("result", {})
        data_key = next((key for key in result if key != "last"), None)
        raw = result.get(data_key, []) if data_key else []
        rows: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) < 8:
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="s", utc=True),
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "vwap": item[5],
                    "volume": item[6],
                    "trades": item[7],
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.loc[
                (frame["timestamp"] >= pd.Timestamp(start))
                & (frame["timestamp"] <= pd.Timestamp(end))
            ]
        self.logger.info(
            "[KRAKEN] %-12s %-3s rows=%d (REST is capped at 720 recent entries)",
            f"{symbol}-{quote}",
            interval,
            len(frame),
        )
        return standardize_frame(
            frame,
            provider="kraken",
            symbol=symbol,
            market=f"{symbol}-{quote}",
            quote=quote,
            interval=interval,
            data_kind="ohlcv",
        )

    def import_bulk_zip(
        self,
        zip_path: Path,
        *,
        symbols: Sequence[str],
        quote: str,
        intervals: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """
        Import Kraken's official OHLCVT ZIP.

        Known Kraken CSV shapes are handled flexibly:
        7 columns: time, open, high, low, close, volume, trades
        8 columns: time, open, high, low, close, vwap, volume, trades
        """
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        symbol_aliases = {symbol: self._kraken_alias(symbol) for symbol in symbols}
        desired_minutes = {KRAKEN_BULK_INTERVAL_MINUTES[i]: i for i in intervals if i in KRAKEN_BULK_INTERVAL_MINUTES}
        outputs: dict[tuple[str, str], list[pd.DataFrame]] = {}

        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".csv"):
                    continue
                stem = Path(member.filename).stem.upper()
                match = re.search(r"([A-Z0-9]+?)(?:_|-)(\d+)$", stem)
                if not match:
                    continue
                pair_token, minutes_raw = match.groups()
                minutes = int(minutes_raw)
                interval = desired_minutes.get(minutes)
                if interval is None:
                    continue
                matched_symbol = None
                for symbol, alias in symbol_aliases.items():
                    if pair_token in {
                        f"{alias}{quote}",
                        f"{symbol}{quote}",
                        f"X{alias}Z{quote}",
                    }:
                        matched_symbol = symbol
                        break
                if matched_symbol is None:
                    continue

                with archive.open(member) as handle:
                    raw = pd.read_csv(handle, header=None)
                if raw.shape[1] < 7:
                    self.logger.warning("[KRAKEN-BULK] skipped unknown CSV shape %s columns=%d", member.filename, raw.shape[1])
                    continue
                if raw.shape[1] >= 8:
                    raw = raw.iloc[:, :8]
                    raw.columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"]
                else:
                    raw = raw.iloc[:, :7]
                    raw.columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True, errors="coerce")
                raw = raw.loc[
                    (raw["timestamp"] >= pd.Timestamp(start))
                    & (raw["timestamp"] <= pd.Timestamp(end))
                ]
                standardized = standardize_frame(
                    raw,
                    provider="kraken",
                    symbol=matched_symbol,
                    market=f"{matched_symbol}-{quote}",
                    quote=quote,
                    interval=interval,
                    data_kind="ohlcv",
                    bar_source="official_bulk_zip",
                )
                outputs.setdefault((matched_symbol, interval), []).append(standardized)
                self.logger.info(
                    "[KRAKEN-BULK] %-40s symbol=%s interval=%s rows=%d",
                    member.filename,
                    matched_symbol,
                    interval,
                    len(standardized),
                )

        return {
            key: pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
            for key, frames in outputs.items()
        }


class CoinMarketCapDownloader:
    def __init__(self, client: HttpClient, logger: logging.Logger, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "CoinMarketCap API key missing. Set COINMARKETCAP_API_KEY or CMC_API_KEY."
            )
        self.client = client
        self.logger = logger
        self.headers = {"X-CMC_PRO_API_KEY": api_key}
        self._id_map: dict[str, int] = {}

    def resolve_ids(self, symbols: Sequence[str]) -> dict[str, int]:
        unresolved = [s for s in symbols if s not in self._id_map]
        if unresolved:
            payload = self.client.get_json(
                f"{CMC_BASE}/v1/cryptocurrency/map",
                params={
                    "symbol": ",".join(unresolved),
                    "listing_status": "active",
                    "sort": "cmc_rank",
                    "limit": 500,
                },
                headers=self.headers,
                host_key="cmc",
            )
            status = payload.get("status", {})
            if status.get("error_code", 0):
                raise RuntimeError(f"CMC map error: {status}")
            candidates: dict[str, list[Mapping[str, Any]]] = {}
            for item in payload.get("data", []):
                if isinstance(item, Mapping):
                    candidates.setdefault(str(item.get("symbol", "")).upper(), []).append(item)
            for symbol in unresolved:
                rows = candidates.get(symbol, [])
                if not rows:
                    self.logger.warning("[CMC] no active ID found for %s", symbol)
                    continue
                rows.sort(key=lambda item: (item.get("rank") is None, item.get("rank") or 10**9, item.get("id") or 10**9))
                self._id_map[symbol] = int(rows[0]["id"])
        return {symbol: self._id_map[symbol] for symbol in symbols if symbol in self._id_map}

    @staticmethod
    def _extract_single_asset_data(payload: Mapping[str, Any], cmc_id: int) -> Mapping[str, Any]:
        data = payload.get("data", {})
        if isinstance(data, Mapping) and "quotes" in data:
            return data
        if isinstance(data, Mapping):
            candidate = data.get(str(cmc_id)) or data.get(cmc_id)
            if isinstance(candidate, Mapping):
                return candidate
            for value in data.values():
                if isinstance(value, Mapping) and int(value.get("id", -1)) == cmc_id:
                    return value
        return {}

    def download_ohlcv(
        self,
        *,
        symbol: str,
        cmc_id: int,
        quote: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if interval not in CMC_OHLCV_INTERVALS:
            raise ValueError(f"CMC true OHLCV only supports configured periods: {CMC_OHLCV_INTERVALS}")
        time_period = "hourly" if interval == "1h" else "daily"
        period_seconds = interval_to_seconds(interval) or 86400
        rows: list[dict[str, Any]] = []

        # <= 9000 periods per request to remain under CMC's 10000-period cap.
        chunk_seconds = period_seconds * 9000
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(seconds=chunk_seconds))
            query_start = cursor - timedelta(seconds=period_seconds)
            payload = self.client.get_json(
                f"{CMC_BASE}/v2/cryptocurrency/ohlcv/historical",
                params={
                    "id": cmc_id,
                    "convert": quote,
                    "time_period": time_period,
                    "interval": interval,
                    "time_start": query_start.isoformat(),
                    "time_end": chunk_end.isoformat(),
                    "count": 10000,
                    "skip_invalid": "true",
                },
                headers=self.headers,
                host_key="cmc",
            )
            status = payload.get("status", {})
            if status.get("error_code", 0):
                raise RuntimeError(f"CMC OHLCV error: {status}")
            asset = self._extract_single_asset_data(payload, cmc_id)
            quotes = asset.get("quotes", []) if isinstance(asset, Mapping) else []
            for item in quotes:
                if not isinstance(item, Mapping):
                    continue
                converted = item.get("quote", {}).get(quote, {})
                if not converted:
                    continue
                rows.append(
                    {
                        "timestamp": item.get("time_open") or converted.get("timestamp"),
                        "open": converted.get("open"),
                        "high": converted.get("high"),
                        "low": converted.get("low"),
                        "close": converted.get("close"),
                        "volume": converted.get("volume"),
                        "market_cap": converted.get("market_cap"),
                    }
                )
            self.logger.info(
                "[CMC-OHLCV] %-8s %-3s chunk=%s..%s cumulative=%d",
                symbol,
                interval,
                cursor.date(),
                chunk_end.date(),
                len(rows),
            )
            cursor = chunk_end + timedelta(seconds=period_seconds)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.loc[
                (frame["timestamp"] >= pd.Timestamp(start))
                & (frame["timestamp"] <= pd.Timestamp(end))
            ]
        return standardize_frame(
            frame,
            provider="coinmarketcap",
            symbol=symbol,
            market=f"{symbol}-{quote}",
            quote=quote,
            interval=interval,
            data_kind="ohlcv",
        )

    def download_quotes(
        self,
        *,
        symbol: str,
        cmc_id: int,
        quote: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if interval not in CMC_QUOTE_INTERVALS:
            raise ValueError(f"Unsupported CMC quote interval: {interval}")
        rows: list[dict[str, Any]] = []
        seconds = interval_to_seconds(interval)
        # Calendar intervals are queried in one request for practical histories.
        if seconds:
            chunk_seconds = seconds * 9000
        else:
            chunk_seconds = int(timedelta(days=3650).total_seconds())

        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(seconds=chunk_seconds))
            payload = self.client.get_json(
                f"{CMC_BASE}/v3/cryptocurrency/quotes/historical",
                params={
                    "id": cmc_id,
                    "convert": quote,
                    "interval": interval,
                    "time_start": cursor.isoformat(),
                    "time_end": chunk_end.isoformat(),
                    "count": 10000,
                    "skip_invalid": "true",
                    "aux": "price,volume,market_cap,quote_timestamp",
                },
                headers=self.headers,
                host_key="cmc",
            )
            status = payload.get("status", {})
            if status.get("error_code", 0):
                raise RuntimeError(f"CMC quotes error: {status}")
            asset = self._extract_single_asset_data(payload, cmc_id)
            quotes = asset.get("quotes", []) if isinstance(asset, Mapping) else []
            for item in quotes:
                if not isinstance(item, Mapping):
                    continue
                converted = item.get("quote", {}).get(quote, {})
                if not converted:
                    continue
                price = converted.get("price")
                rows.append(
                    {
                        "timestamp": item.get("timestamp") or converted.get("timestamp"),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": converted.get("volume_24h") or converted.get("volume_24hr"),
                        "market_cap": converted.get("market_cap"),
                    }
                )
            self.logger.info(
                "[CMC-QUOTE] %-8s %-7s chunk=%s..%s cumulative=%d",
                symbol,
                interval,
                cursor.date(),
                chunk_end.date(),
                len(rows),
            )
            cursor = chunk_end + timedelta(seconds=seconds or 1)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.loc[
                (frame["timestamp"] >= pd.Timestamp(start))
                & (frame["timestamp"] <= pd.Timestamp(end))
            ]
        # Quote data is not true OHLCV; OHLC fields all equal the sampled price by design.
        return standardize_frame(
            frame,
            provider="coinmarketcap",
            symbol=symbol,
            market=f"{symbol}-{quote}",
            quote=quote,
            interval=interval,
            data_kind="quotes",
            bar_source="sampled_quote",
        )


def choose_provider_intervals(provider: str, requested: str, cmc_mode: str) -> tuple[list[str], list[str]]:
    if requested.lower() == "all":
        if provider == "bitvavo":
            return list(BITVAVO_INTERVALS), []
        if provider == "kraken":
            return list(KRAKEN_INTERVAL_MINUTES), []
        if provider == "coinmarketcap":
            ohlcv = list(CMC_OHLCV_INTERVALS) if cmc_mode in {"ohlcv", "both"} else []
            quotes = list(CMC_QUOTE_INTERVALS) if cmc_mode in {"quotes", "both"} else []
            return ohlcv, quotes
    values = parse_csv_list(requested)
    if provider == "coinmarketcap":
        ohlcv = [x for x in values if x in CMC_OHLCV_INTERVALS] if cmc_mode in {"ohlcv", "both"} else []
        quotes = [x for x in values if x in CMC_QUOTE_INTERVALS] if cmc_mode in {"quotes", "both"} else []
        return ohlcv, quotes
    allowed = BITVAVO_INTERVALS if provider == "bitvavo" else tuple(KRAKEN_INTERVAL_MINUTES)
    unknown = [x for x in values if x not in allowed]
    if unknown:
        raise ValueError(f"Unsupported {provider} intervals: {unknown}")
    return values, []


def get_resume_start(path: Path, default_start: datetime, interval: str) -> datetime:
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            return default_start
        path = csv_path
    existing = read_existing(path)
    if existing.empty:
        return default_start
    last = pd.to_datetime(existing["timestamp"], utc=True).max()
    seconds = interval_to_seconds(interval)
    if seconds:
        return max(default_start, (last + pd.to_timedelta(seconds, unit="s")).to_pydatetime())
    if interval in {"1M", "monthly"}:
        return max(default_start, (last + pd.offsets.MonthBegin(1)).to_pydatetime())
    if interval == "yearly":
        return max(default_start, (last + pd.offsets.YearBegin(1)).to_pydatetime())
    return default_start


def process_and_write(
    frame: pd.DataFrame,
    *,
    path: Path,
    resume: bool,
    also_csv: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, Path]:
    existing_path = path if path.exists() else path.with_suffix(".csv")
    existing = read_existing(existing_path) if resume and existing_path.exists() else pd.DataFrame()
    merged = merge_frames(existing, frame)
    written = write_frame_atomic(merged, path, also_csv=also_csv, logger=logger)
    return merged, written


def write_summary_artifacts(
    output_dir: Path,
    coverages: Sequence[CoverageRecord],
    failures: Sequence[FailureRecord],
    manifest: Mapping[str, Any],
) -> None:
    coverage_path = output_dir / "coverage.csv"
    pd.DataFrame([asdict(item) for item in coverages]).to_csv(coverage_path, index=False)
    failure_path = output_dir / "failures.jsonl"
    if failures:
        atomic_write_text(
            failure_path,
            "".join(
                json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
                for item in failures
            ),
        )
    elif failure_path.exists():
        failure_path.unlink()
    atomic_write_text(
        output_dir / "download_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str),
    )


def run_download(args: argparse.Namespace) -> int:
    env_path = Path(args.env_file).expanduser().resolve()
    load_env_file(env_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    logger = setup_logger(output_dir, args.verbose)

    symbols = normalize_symbols(args.symbols)
    quote = args.quote.upper()
    start = parse_utc(args.start, default=datetime(2019, 1, 1, tzinfo=timezone.utc))
    end = parse_utc(args.end, default=utc_now())
    if end <= start:
        raise SystemExit("--end must be later than --start")

    providers = ["bitvavo", "kraken", "coinmarketcap"] if args.providers == "all" else parse_csv_list(args.providers)
    unknown_providers = [p for p in providers if p not in {"bitvavo", "kraken", "coinmarketcap"}]
    if unknown_providers:
        raise SystemExit(f"Unknown providers: {unknown_providers}")

    client = HttpClient(
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        min_delay=args.min_request_delay,
        logger=logger,
    )
    coverage: list[CoverageRecord] = []
    failures: list[FailureRecord] = []
    job_plan: list[dict[str, Any]] = []

    for provider in providers:
        native, quotes = choose_provider_intervals(provider, args.timeframes, args.cmc_mode)
        for symbol in symbols:
            for interval in native:
                eff = effective_start_for_policy(start, interval, args.history_policy, end)
                job_plan.append(
                    {
                        "provider": provider,
                        "data_kind": "ohlcv",
                        "symbol": symbol,
                        "interval": interval,
                        "start": eff,
                        "estimated_bars": estimated_bar_count(eff, end, interval),
                    }
                )
            for interval in quotes:
                eff = effective_start_for_policy(start, interval, args.history_policy, end)
                # CoinMarketCap Startup-style intraday access is commonly limited to ~1 month.
                if args.history_policy == "smart" and interval.endswith(("m", "h")):
                    eff = max(eff, end - timedelta(days=args.cmc_intraday_days))
                job_plan.append(
                    {
                        "provider": provider,
                        "data_kind": "quotes",
                        "symbol": symbol,
                        "interval": interval,
                        "start": eff,
                        "estimated_bars": estimated_bar_count(eff, end, interval),
                    }
                )

    estimated_total = sum(item["estimated_bars"] or 0 for item in job_plan)
    logger.info("[PLAN] providers=%s symbols=%s jobs=%d estimated_fixed_bars=%d",
                ",".join(providers), ",".join(symbols), len(job_plan), estimated_total)
    logger.info("[SAFETY] DATA_ONLY=true HTTP_METHODS=GET ORDER_ENDPOINTS=false")
    if args.dry_run:
        pd.DataFrame(job_plan).to_csv(output_dir / "dry_run_plan.csv", index=False)
        logger.info("[DRY-RUN] wrote %s", output_dir / "dry_run_plan.csv")
        return 0

    manifest: dict[str, Any] = {
        "program": Path(__file__).name,
        "version": VERSION,
        "started_at_utc": iso_utc(utc_now()),
        "env_file": str(env_path),
        "providers": providers,
        "symbols": symbols,
        "quote": quote,
        "requested_start": iso_utc(start),
        "end": iso_utc(end),
        "timeframes": args.timeframes,
        "history_policy": args.history_policy,
        "closed_only": True,
        "resume": args.resume,
        "cmc_mode": args.cmc_mode,
        "kraken_rest_limit_note": "Kraken REST OHLC returns at most 720 recent entries per timeframe.",
        "kraken_bulk_zip": str(Path(args.kraken_bulk_zip).resolve()) if args.kraken_bulk_zip else None,
        "kraken_bulk_url": args.kraken_bulk_url,
        "authority": {
            "data_only": True,
            "order_authority": "NONE",
            "paper_authority": "NONE",
            "live_weight": 0.0,
        },
        "job_plan": [
            {**item, "start": iso_utc(item["start"])}
            for item in job_plan
        ],
    }

    bitvavo = BitvavoDownloader(client, logger) if "bitvavo" in providers else None
    kraken = KrakenDownloader(client, logger) if "kraken" in providers else None
    cmc: CoinMarketCapDownloader | None = None
    if "coinmarketcap" in providers:
        cmc_key = os.environ.get("COINMARKETCAP_API_KEY") or os.environ.get("CMC_API_KEY") or ""
        try:
            cmc = CoinMarketCapDownloader(client, logger, cmc_key)
        except Exception as exc:
            logger.error("[CMC] disabled: %s", exc)
            failures.append(
                FailureRecord(
                    provider="coinmarketcap",
                    data_kind="startup",
                    symbol="*",
                    market="*",
                    interval="*",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    timestamp_utc=iso_utc(utc_now()) or "",
                )
            )

    cmc_ids = cmc.resolve_ids(symbols) if cmc else {}

    # Optional Kraken official bulk ZIP import first.
    kraken_bulk: dict[tuple[str, str], pd.DataFrame] = {}
    kraken_bulk_path: Path | None = Path(args.kraken_bulk_zip).expanduser().resolve() if args.kraken_bulk_zip else None
    if kraken and args.kraken_bulk_url:
        try:
            cache_dir = output_dir / "_downloads"
            cache_dir.mkdir(parents=True, exist_ok=True)
            url_name = Path(args.kraken_bulk_url.split("?", 1)[0]).name
            if not url_name.lower().endswith(".zip"):
                url_name = "kraken_official_ohlcvt.zip"
            downloaded = cache_dir / url_name
            if not downloaded.exists() or downloaded.stat().st_size == 0:
                logger.info("[KRAKEN-BULK] downloading %s", args.kraken_bulk_url)
                payload = client.get_bytes(args.kraken_bulk_url, host_key="kraken_bulk")
                tmp_download = downloaded.with_suffix(downloaded.suffix + ".tmp")
                tmp_download.write_bytes(payload)
                os.replace(tmp_download, downloaded)
            else:
                logger.info("[KRAKEN-BULK] using cached ZIP %s", downloaded)
            kraken_bulk_path = downloaded
            manifest["kraken_bulk_downloaded_path"] = str(downloaded)
            manifest["kraken_bulk_download_sha256"] = sha256_file(downloaded)
        except Exception as exc:
            logger.exception("[KRAKEN-BULK] download failed")
            failures.append(
                FailureRecord(
                    provider="kraken",
                    data_kind="bulk_download",
                    symbol="*",
                    market="*",
                    interval="*",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    timestamp_utc=iso_utc(utc_now()) or "",
                )
            )

    if kraken and kraken_bulk_path:
        try:
            requested_kraken, _ = choose_provider_intervals("kraken", args.timeframes, args.cmc_mode)
            kraken_bulk = kraken.import_bulk_zip(
                kraken_bulk_path,
                symbols=symbols,
                quote=quote,
                intervals=requested_kraken,
                start=start,
                end=end,
            )
        except Exception as exc:
            logger.exception("[KRAKEN-BULK] import failed")
            failures.append(
                FailureRecord(
                    provider="kraken",
                    data_kind="bulk_zip",
                    symbol="*",
                    market="*",
                    interval="*",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    timestamp_utc=iso_utc(utc_now()) or "",
                )
            )

    completed = 0
    for job in job_plan:
        provider = job["provider"]
        data_kind = job["data_kind"]
        symbol = job["symbol"]
        interval = job["interval"]
        market = f"{symbol}-{quote}"
        path = output_path(
            output_dir,
            provider=provider,
            data_kind=data_kind,
            symbol=symbol,
            quote=quote,
            interval=interval,
        )
        effective_start = job["start"]
        if args.resume:
            effective_start = get_resume_start(path, effective_start, interval)
        if effective_start >= end:
            logger.info("[SKIP] %s %s %s %s already current", provider, symbol, interval, data_kind)
            existing_path = path if path.exists() else path.with_suffix(".csv")
            existing = read_existing(existing_path) if existing_path.exists() else pd.DataFrame()
            if existing_path.exists():
                coverage.append(
                    coverage_record(
                        existing,
                        existing_path,
                        provider=provider,
                        data_kind=data_kind,
                        symbol=symbol,
                        market=market,
                        interval=interval,
                        status="UP_TO_DATE",
                    )
                )
            continue

        try:
            logger.info(
                "[JOB %d/%d] provider=%s kind=%s market=%s interval=%s start=%s",
                completed + 1,
                len(job_plan),
                provider,
                data_kind,
                market,
                interval,
                effective_start,
            )
            if provider == "bitvavo":
                assert bitvavo is not None
                frame = bitvavo.download(
                    symbol=symbol,
                    quote=quote,
                    interval=interval,
                    start=effective_start,
                    end=end,
                )
            elif provider == "kraken":
                assert kraken is not None
                bulk_frame = kraken_bulk.get((symbol, interval), pd.DataFrame())
                recent = kraken.download_recent(
                    symbol=symbol,
                    quote=quote,
                    interval=interval,
                    start=effective_start,
                    end=end,
                )
                frame = merge_frames(bulk_frame, recent)
            elif provider == "coinmarketcap":
                if cmc is None or symbol not in cmc_ids:
                    raise RuntimeError(f"CMC unavailable or symbol ID unresolved: {symbol}")
                if data_kind == "ohlcv":
                    frame = cmc.download_ohlcv(
                        symbol=symbol,
                        cmc_id=cmc_ids[symbol],
                        quote=quote,
                        interval=interval,
                        start=effective_start,
                        end=end,
                    )
                else:
                    frame = cmc.download_quotes(
                        symbol=symbol,
                        cmc_id=cmc_ids[symbol],
                        quote=quote,
                        interval=interval,
                        start=effective_start,
                        end=end,
                    )
            else:  # pragma: no cover
                raise AssertionError(provider)

            merged, written = process_and_write(
                frame,
                path=path,
                resume=args.resume,
                also_csv=args.csv,
                logger=logger,
            )
            note = ""
            if provider == "kraken" and not kraken_bulk_path:
                note = "REST recent only: maximum 720 entries."
            if provider == "coinmarketcap" and data_kind == "quotes":
                note = "Sampled quote series; not true OHLCV candles."
            record = coverage_record(
                merged,
                written,
                provider=provider,
                data_kind=data_kind,
                symbol=symbol,
                market=market,
                interval=interval,
                status="OK" if len(merged) else "EMPTY",
                note=note,
            )
            coverage.append(record)
            logger.info(
                "[DONE] %s %s %s %s rows=%d first=%s last=%s gaps=%s file=%s",
                provider,
                data_kind,
                market,
                interval,
                record.rows,
                record.first_timestamp,
                record.last_timestamp,
                record.gap_count,
                written,
            )
        except Exception as exc:
            logger.exception("[FAIL] provider=%s kind=%s market=%s interval=%s", provider, data_kind, market, interval)
            failure = FailureRecord(
                provider=provider,
                data_kind=data_kind,
                symbol=symbol,
                market=market,
                interval=interval,
                error_type=type(exc).__name__,
                message=str(exc),
                timestamp_utc=iso_utc(utc_now()) or "",
            )
            failures.append(failure)
            append_jsonl(output_dir / "failures_live.jsonl", asdict(failure))
            if args.fail_fast:
                manifest["finished_at_utc"] = iso_utc(utc_now())
                manifest["status"] = "FAILED_FAST"
                write_summary_artifacts(output_dir, coverage, failures, manifest)
                raise
        finally:
            completed += 1
            if completed % max(1, args.checkpoint_every) == 0:
                checkpoint_manifest = {
                    **manifest,
                    "completed_jobs": completed,
                    "total_jobs": len(job_plan),
                    "coverage_rows": len(coverage),
                    "failures": len(failures),
                    "checkpoint_at_utc": iso_utc(utc_now()),
                }
                write_summary_artifacts(output_dir, coverage, failures, checkpoint_manifest)

    manifest["finished_at_utc"] = iso_utc(utc_now())
    manifest["completed_jobs"] = completed
    manifest["coverage_records"] = len(coverage)
    manifest["failure_count"] = len(failures)
    manifest["status"] = "COMPLETE_WITH_ERRORS" if failures else "COMPLETE"
    manifest["cmc_id_map"] = cmc_ids
    write_summary_artifacts(output_dir, coverage, failures, manifest)
    logger.info(
        "[COMPLETE] status=%s coverage=%d failures=%d output=%s",
        manifest["status"],
        len(coverage),
        len(failures),
        output_dir,
    )
    return 1 if failures and args.nonzero_on_partial else 0


def run_status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "download_manifest.json"
    coverage_path = output_dir / "coverage.csv"
    if not manifest_path.exists():
        print(f"No manifest found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        if not coverage.empty:
            summary = (
                coverage.groupby(["provider", "data_kind"], dropna=False)
                .agg(series=("interval", "count"), rows=("rows", "sum"))
                .reset_index()
            )
            print("\nCoverage summary:")
            print(summary.to_string(index=False))
    return 0


def run_self_test() -> int:
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [1, 2, 3, 4, 5],
        }
    )
    standardized = standardize_frame(
        frame,
        provider="test",
        symbol="BTC",
        market="BTC-EUR",
        quote="EUR",
        interval="1h",
        data_kind="ohlcv",
        closed_only=True,
    )
    assert len(standardized) == 5
    assert standardized["timestamp"].is_monotonic_increasing
    assert set(STANDARD_COLUMNS) == set(standardized.columns)
    gaps = audit_gaps(standardized, "1h")
    assert gaps == (0, 0, 0)

    duplicated = pd.concat([standardized, standardized.iloc[-1:]], ignore_index=True)
    merged = merge_frames(standardized, duplicated)
    assert len(merged) == 5

    invalid = frame.copy()
    invalid.loc[0, "high"] = 1
    cleaned = standardize_frame(
        invalid,
        provider="test",
        symbol="BTC",
        market="BTC-EUR",
        quote="EUR",
        interval="1h",
        data_kind="ohlcv",
    )
    assert len(cleaned) == 4

    assert effective_start_for_policy(
        datetime(2010, 1, 1, tzinfo=timezone.utc),
        "1m",
        "smart",
        now,
    ) == now - timedelta(days=180)
    assert choose_provider_intervals("bitvavo", "all", "both")[0] == list(BITVAVO_INTERVALS)
    assert "15d" in choose_provider_intervals("kraken", "all", "both")[0]
    cmc_ohlcv, cmc_quotes = choose_provider_intervals("coinmarketcap", "all", "both")
    assert cmc_ohlcv == list(CMC_OHLCV_INTERVALS)
    assert "5m" in cmc_quotes
    assert "1h" in cmc_quotes

    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "test.parquet"
        logger = logging.getLogger("self_test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        written = write_frame_atomic(standardized, out, also_csv=False, logger=logger)
        reloaded = read_existing(written)
        assert len(reloaded) == 5
        assert sha256_file(written)

    print("SELF_TEST_PASS")
    print(
        json.dumps(
            {
                "version": VERSION,
                "bitvavo_intervals": list(BITVAVO_INTERVALS),
                "kraken_rest_intervals": list(KRAKEN_INTERVAL_MINUTES),
                "kraken_bulk_intervals": list(KRAKEN_BULK_INTERVAL_MINUTES),
                "cmc_true_ohlcv_intervals": list(CMC_OHLCV_INTERVALS),
                "cmc_quote_intervals": list(CMC_QUOTE_INTERVALS),
                "order_authority": "NONE",
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download multi-timeframe crypto data from Bitvavo, Kraken, and CoinMarketCap."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download/update market data")
    download.add_argument("--env-file", default=".env")
    download.add_argument(
        "--providers",
        default="all",
        help="all or comma-separated: bitvavo,kraken,coinmarketcap",
    )
    download.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated base symbols. Defaults to WM_TRACKED_WALLET_ASSETS or BTC,ETH,SOL,LINK,ADA.",
    )
    download.add_argument("--quote", default="EUR")
    download.add_argument("--start", default="2019-01-01")
    download.add_argument("--end", default="now")
    download.add_argument(
        "--timeframes",
        default="all",
        help="all or comma-separated provider-supported intervals",
    )
    download.add_argument(
        "--history-policy",
        choices=("smart", "full", "recent"),
        default="smart",
        help="smart limits very small bars; full uses --start for every interval",
    )
    download.add_argument(
        "--cmc-mode",
        choices=("ohlcv", "quotes", "both"),
        default="both",
        help="CMC true OHLCV, sampled historical quotes, or both",
    )
    download.add_argument(
        "--cmc-intraday-days",
        type=int,
        default=31,
        help="Smart-policy CMC intraday lookback; Startup plans commonly allow about one month.",
    )
    download.add_argument(
        "--kraken-bulk-zip",
        default=None,
        help="Optional local official Kraken OHLCVT ZIP for full history.",
    )
    download.add_argument(
        "--kraken-bulk-url",
        default=None,
        help="Optional direct official Kraken OHLCVT ZIP URL; downloaded once into the output cache.",
    )
    download.add_argument(
        "--output-dir",
        default="output/market_data/multi_venue_ohlcv",
    )
    download.add_argument("--resume", action="store_true", help="Upsert existing files and fetch newer data")
    download.add_argument("--clean-output", action="store_true")
    download.add_argument("--csv", action="store_true", help="Also write CSV beside Parquet")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--fail-fast", action="store_true")
    download.add_argument("--nonzero-on-partial", action="store_true")
    download.add_argument("--checkpoint-every", type=int, default=5)
    download.add_argument("--timeout", type=float, default=45.0)
    download.add_argument("--retries", type=int, default=6)
    download.add_argument("--backoff", type=float, default=0.8)
    download.add_argument("--min-request-delay", type=float, default=0.12)
    download.add_argument("--verbose", action="store_true")
    download.set_defaults(func=run_download)

    status = subparsers.add_parser("status", help="Show manifest and coverage summary")
    status.add_argument(
        "--output-dir",
        default="output/market_data/multi_venue_ohlcv",
    )
    status.set_defaults(func=run_status)

    self_test = subparsers.add_parser("self-test", help="Run offline integrity tests")
    self_test.set_defaults(func=lambda _args: run_self_test())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user. Existing completed files remain intact.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
