#!/usr/bin/env python3
"""
Kraken Public Crypto Data Collector

Purpose:
- Pull public market data from Kraken Spot REST endpoints.
- Write OHLC or recent trades to CSV.
- Use public endpoints only. No API key. No secret. No trading. No private account calls.

Official endpoint notes:
- OHLC: https://api.kraken.com/0/public/OHLC
- Trades: https://api.kraken.com/0/public/Trades

This script is intentionally read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_URL = "https://api.kraken.com/0/public"
VALID_OHLC_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}


@dataclass(frozen=True)
class CollectorConfig:
    data_type: str
    pair: str
    interval: int
    since: Optional[str]
    days: Optional[int]
    out_dir: Path
    max_pages: int
    sleep_s: float
    timeout_s: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_since_arg(since: Optional[str], days: Optional[int]) -> Optional[str]:
    """Return a Kraken-compatible since value.

    For OHLC, Kraken expects a Unix timestamp in seconds.
    For Trades, Kraken accepts a since cursor. A Unix seconds timestamp is fine for the first request,
    then the API-provided `last` cursor is used for pagination.
    """
    if since:
        raw = since.strip()
        if raw.isdigit():
            return raw
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return str(int(dt.timestamp()))
        except ValueError as exc:
            raise SystemExit(f"Invalid --since value: {since}. Use Unix seconds or ISO format like 2026-07-01T00:00:00Z") from exc

    if days is not None:
        dt = utc_now() - timedelta(days=days)
        return str(int(dt.timestamp()))

    return None


def request_json(endpoint: str, params: Dict[str, Any], timeout_s: float, max_retries: int = 4) -> Dict[str, Any]:
    """Call a Kraken public REST endpoint with conservative retry/backoff."""
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}/{endpoint}"
    if query:
        url = f"{url}?{query}"

    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "worldmonitor-kraken-public-data-collector/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = resp.read().decode("utf-8")
            data = json.loads(payload)

            errors = data.get("error") or []
            if errors:
                joined = "; ".join(str(e) for e in errors)
                if "rate limit" in joined.lower() or "throttled" in joined.lower():
                    wait = min(30.0, 2.0 ** attempt)
                    print(f"RATE_LIMIT_OR_THROTTLED: {joined}. Sleeping {wait:.1f}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Kraken API error: {joined}")

            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = str(exc)
            wait = min(30.0, 2.0 ** attempt)
            if attempt < max_retries - 1:
                print(f"REQUEST_RETRY: endpoint={endpoint} attempt={attempt + 1} error={last_error}. Sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                break

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")


def kraken_result_pair_key(result: Dict[str, Any]) -> str:
    keys = [k for k in result.keys() if k != "last"]
    if not keys:
        raise RuntimeError("Kraken response did not contain a pair result key")
    return keys[0]


def safe_filename_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def make_output_path(config: CollectorConfig) -> Path:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    pair = safe_filename_part(config.pair)
    suffix = f"{config.data_type}_{pair}"
    if config.data_type == "ohlc":
        suffix += f"_{config.interval}m"
    return config.out_dir / f"kraken_{suffix}_{stamp}.csv"


def collect_ohlc(config: CollectorConfig) -> Tuple[Path, int]:
    if config.interval not in VALID_OHLC_INTERVALS:
        raise SystemExit(f"Invalid OHLC interval {config.interval}. Valid: {sorted(VALID_OHLC_INTERVALS)}")

    output_path = make_output_path(config)
    current_since = config.since
    seen_timestamps = set()
    total_rows = 0
    previous_last: Optional[str] = None

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_utc", "unix_ts", "open", "high", "low", "close", "vwap", "volume", "trade_count"])

        for page in range(config.max_pages):
            params: Dict[str, Any] = {"pair": config.pair, "interval": config.interval}
            if current_since:
                params["since"] = current_since

            data = request_json("OHLC", params, timeout_s=config.timeout_s)
            result = data.get("result") or {}
            pair_key = kraken_result_pair_key(result)
            rows: List[List[Any]] = result[pair_key]
            next_last = str(result.get("last", ""))

            new_rows = 0
            for row in rows:
                unix_ts = int(row[0])
                if unix_ts in seen_timestamps:
                    continue
                seen_timestamps.add(unix_ts)
                timestamp = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
                writer.writerow([timestamp, unix_ts, row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
                new_rows += 1
                total_rows += 1

            print(f"OHLC_PAGE_DONE page={page + 1} rows={new_rows} total={total_rows} last={next_last}")

            if not next_last or next_last == previous_last or new_rows == 0:
                break
            previous_last = next_last
            current_since = next_last
            time.sleep(config.sleep_s)

    return output_path, total_rows


def collect_trades(config: CollectorConfig) -> Tuple[Path, int]:
    output_path = make_output_path(config)
    current_since = config.since
    seen_trade_ids = set()
    total_rows = 0
    previous_last: Optional[str] = None

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_utc", "unix_ts", "price", "volume", "side", "order_type", "misc", "trade_id"])

        for page in range(config.max_pages):
            params: Dict[str, Any] = {"pair": config.pair, "count": 1000}
            if current_since:
                params["since"] = current_since

            data = request_json("Trades", params, timeout_s=config.timeout_s)
            result = data.get("result") or {}
            pair_key = kraken_result_pair_key(result)
            trades: List[List[Any]] = result[pair_key]
            next_last = str(result.get("last", ""))

            new_rows = 0
            for trade in trades:
                price, volume, unix_ts, side, order_type, misc = trade[:6]
                trade_id = trade[6] if len(trade) > 6 else ""
                unique_key = trade_id if trade_id != "" else (unix_ts, price, volume, side, order_type)
                if unique_key in seen_trade_ids:
                    continue
                seen_trade_ids.add(unique_key)
                timestamp = datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).isoformat()
                writer.writerow([timestamp, unix_ts, price, volume, side, order_type, misc, trade_id])
                new_rows += 1
                total_rows += 1

            print(f"TRADES_PAGE_DONE page={page + 1} rows={new_rows} total={total_rows} last={next_last}")

            if not next_last or next_last == previous_last or new_rows == 0:
                break
            previous_last = next_last
            current_since = next_last
            time.sleep(config.sleep_s)

    return output_path, total_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public crypto market data from Kraken to CSV.")
    parser.add_argument("--type", choices=["ohlc", "trades"], default="ohlc", help="Data type to download.")
    parser.add_argument("--pair", default="XBTUSD", help="Kraken pair, for example XBTUSD, ETHUSD, XBTEUR, ETHEUR.")
    parser.add_argument("--interval", type=int, default=1, help="OHLC interval in minutes. Ignored for trades.")
    parser.add_argument("--since", default=None, help="Unix seconds or ISO timestamp, for example 2026-07-01T00:00:00Z.")
    parser.add_argument("--days", type=int, default=1, help="Lookback window when --since is not provided.")
    parser.add_argument("--out", default="output/kraken", help="Output directory for CSV files.")
    parser.add_argument("--max-pages", type=int, default=3, help="Maximum API pages to fetch. Keep low for smoke tests.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between pages.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    since = parse_since_arg(args.since, args.days)
    config = CollectorConfig(
        data_type=args.type,
        pair=args.pair,
        interval=args.interval,
        since=since,
        days=args.days,
        out_dir=Path(args.out),
        max_pages=max(1, args.max_pages),
        sleep_s=max(0.0, args.sleep),
        timeout_s=max(1.0, args.timeout),
    )

    print("KRAKEN_PUBLIC_COLLECTOR_START")
    print(f"data_type={config.data_type} pair={config.pair} interval={config.interval} since={config.since} max_pages={config.max_pages}")
    print("safety=PUBLIC_MARKET_DATA_ONLY no_api_key no_secret no_private_endpoints no_orders")

    if config.data_type == "ohlc":
        output_path, total_rows = collect_ohlc(config)
    else:
        output_path, total_rows = collect_trades(config)

    print("KRAKEN_PUBLIC_COLLECTOR_DONE")
    print(f"output={output_path}")
    print(f"rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
