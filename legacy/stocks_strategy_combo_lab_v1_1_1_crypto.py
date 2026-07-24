#!/usr/bin/env python3
"""
Strategy Combo Research Lab
===========================

One-file, long-running backtest and parameter-search program for daily stock data.
It tests individual strategy variants first, then combinations of 2, 3, and 4
strategy families. It writes live progress to the terminal and persists every
completed result so interrupted runs can resume.

Key design choices
------------------
* Long-only and next-open execution by default.
* No lookahead: signals are calculated after close t and executed at open t+1.
* Normal and stressed transaction-cost scenarios.
* Parameter and exit experiments for every implemented strategy.
* Deterministic sampling when the full Cartesian search is too large.
* Combination modes: all, majority, or any entry vote.
* Single strategies are searched first. Combinations use shortlisted native-exit
  variants to reduce combinatorial overfitting.
* Results: CSV, JSON, JSONL checkpoint, HTML report, top trades, equity curves,
  fold statistics, Monte Carlo diagnostics, and live log file.

Typical commands
----------------
Install the minimal dependencies:
    pip install pandas numpy scipy rich pyarrow

Quick smoke run using downloaded data:
    python strategy_combo_research_lab.py run --download \
        --symbols BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR,ADA-EUR \
        --interval 1d --start 2019-01-01 --profile quick

Long run:
    python strategy_combo_research_lab.py run --download \
        --symbols BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR,ADA-EUR,AVGO,COST,ADBE,AMD \
        --interval 1d --start 2019-01-01 --profile deep \
        --combo-sizes 2,3,4 --combo-modes all,majority \
        --max-variants-per-strategy 80 --shortlist 30 \
        --max-combos 20000 --mc-paths 5000 --resume

Local directory with one CSV/Parquet per symbol:
    python strategy_combo_research_lab.py run \
        --data-dir data/daily --profile standard --resume

Long CSV/Parquet with a Symbol column:
    python strategy_combo_research_lab.py run \
        --data-file data/all_stocks.parquet --profile deep --resume

List implemented and blocked strategies:
    python strategy_combo_research_lab.py list-strategies

Run built-in validation:
    python strategy_combo_research_lab.py self-test

This is a research tool, not a live trading engine. It never sends orders.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None

try:
    from rich.logging import RichHandler
except Exception:  # pragma: no cover
    RichHandler = None

VERSION = "1.1.1-crypto"
TRADING_DAYS = 365.25  # compatibility fallback; metrics infer periods from timestamps
BITVAVO_API_BASE = "https://api.bitvavo.com/v2"
BITVAVO_INTERVALS_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
EPS = 1e-12


# ---------------------------------------------------------------------------
# Logging and utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()[:length]


def safe_name(value: str, max_length: int = 140) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:max_length] or "result"


def parse_csv_list(value: str, cast: Callable[[str], Any] = str) -> list[Any]:
    if not value:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "live_run.log"
    logger = logging.getLogger("strategy_lab")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if RichHandler is not None:
        console = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(console)
    else:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitConfig:
    name: str
    mode: str = "native"
    rsi_period: int | None = None
    rsi_level: float | None = None
    max_hold_bars: int | None = None
    stop_atr: float | None = None
    target_atr: float | None = None
    trailing_atr: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StrategyVariant:
    strategy: str
    family: str
    params: Mapping[str, Any]
    exit_config: ExitConfig
    description: str

    @property
    def id(self) -> str:
        payload = {
            "strategy": self.strategy,
            "family": self.family,
            "params": dict(self.params),
            "exit": self.exit_config.as_dict(),
        }
        return f"{self.strategy}__{stable_hash(payload, 18)}"

    @property
    def label(self) -> str:
        return f"{self.strategy} | {dict(self.params)} | exit={self.exit_config.name}"


@dataclass(frozen=True)
class ComboVariant:
    components: tuple[StrategyVariant, ...]
    entry_mode: str
    exit_vote: str = "any"
    max_hold_bars: int | None = None
    stop_atr: float | None = None
    target_atr: float | None = None

    @property
    def id(self) -> str:
        payload = {
            "components": [v.id for v in self.components],
            "entry_mode": self.entry_mode,
            "exit_vote": self.exit_vote,
            "max_hold_bars": self.max_hold_bars,
            "stop_atr": self.stop_atr,
            "target_atr": self.target_atr,
        }
        return f"COMBO_{len(self.components)}__{stable_hash(payload, 20)}"

    @property
    def label(self) -> str:
        members = " + ".join(v.strategy for v in self.components)
        return f"{members} | entry={self.entry_mode} | exit={self.exit_vote}"


@dataclass
class SignalBundle:
    entry: pd.Series
    native_exit: pd.Series
    atr: pd.Series
    close: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    result_id: str
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    net_return: float
    entry_fee: float
    exit_fee: float
    bars_held: int
    exit_reason: str
    mae: float
    mfe: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class BacktestOutput:
    result_id: str
    result_type: str
    label: str
    strategy_names: list[str]
    params: dict[str, Any]
    equity: pd.Series
    trades: list[Trade]
    exposure: float
    cost_scenario: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CostModel:
    commission_bps: float
    slippage_bps: float
    fixed_fee: float = 0.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10000.0


@dataclass
class RunContext:
    args: argparse.Namespace
    output_dir: Path
    logger: logging.Logger
    data: dict[str, pd.DataFrame]
    run_id: str
    checkpoint_path: Path
    completed: dict[str, dict[str, Any]]
    global_trials: int = 0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    description: str
    default_params: Mapping[str, Any]
    grid: Mapping[str, Sequence[Any]]
    builder: Callable[[pd.DataFrame, Mapping[str, Any], "IndicatorCache"], SignalBundle]
    combo_eligible: bool = True


# ---------------------------------------------------------------------------
# Indicator cache
# ---------------------------------------------------------------------------


class IndicatorCache:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame
        self._cache: dict[tuple[Any, ...], Any] = {}

    def _get(self, key: tuple[Any, ...], factory: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def sma(self, column: str, period: int) -> pd.Series:
        return self._get(
            ("sma", column, period),
            lambda: self.df[column].rolling(period, min_periods=period).mean(),
        )

    def ema(self, column: str, period: int) -> pd.Series:
        return self._get(
            ("ema", column, period),
            lambda: self.df[column].ewm(span=period, adjust=False, min_periods=period).mean(),
        )

    def true_range(self) -> pd.Series:
        def calc() -> pd.Series:
            high = self.df["High"]
            low = self.df["Low"]
            prev_close = self.df["Close"].shift(1)
            return pd.concat(
                [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)

        return self._get(("tr",), calc)

    def atr(self, period: int) -> pd.Series:
        return self._get(
            ("atr", period),
            lambda: self.true_range().ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean(),
        )

    def rsi(self, period: int) -> pd.Series:
        def calc() -> pd.Series:
            delta = self.df["Close"].diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            out = 100.0 - 100.0 / (1.0 + rs)
            out = out.where(avg_loss > 0.0, 100.0)
            out = out.where(avg_gain > 0.0, 0.0)
            return out

        return self._get(("rsi", period), calc)

    def adx(self, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        def calc() -> tuple[pd.Series, pd.Series, pd.Series]:
            high = self.df["High"]
            low = self.df["Low"]
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                index=self.df.index,
            )
            minus_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                index=self.df.index,
            )
            atr = self.atr(period)
            plus_di = 100.0 * plus_dm.ewm(
                alpha=1.0 / period, adjust=False, min_periods=period
            ).mean() / atr.replace(0.0, np.nan)
            minus_di = 100.0 * minus_dm.ewm(
                alpha=1.0 / period, adjust=False, min_periods=period
            ).mean() / atr.replace(0.0, np.nan)
            dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
            adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            return adx, plus_di, minus_di

        return self._get(("adx", period), calc)

    def williams_r(self, period: int) -> pd.Series:
        def calc() -> pd.Series:
            hh = self.df["High"].rolling(period, min_periods=period).max()
            ll = self.df["Low"].rolling(period, min_periods=period).min()
            return -100.0 * (hh - self.df["Close"]) / (hh - ll).replace(0.0, np.nan)

        return self._get(("williams", period), calc)

    def macd_hist(self, fast: int, slow: int, signal: int) -> pd.Series:
        def calc() -> pd.Series:
            macd = self.ema("Close", fast) - self.ema("Close", slow)
            sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
            return macd - sig

        return self._get(("macd_hist", fast, slow, signal), calc)

    def bollinger(self, period: int, upper_std: float, lower_std: float) -> tuple[pd.Series, pd.Series]:
        def calc() -> tuple[pd.Series, pd.Series]:
            mean = self.sma("Close", period)
            std = self.df["Close"].rolling(period, min_periods=period).std(ddof=0)
            return mean + upper_std * std, mean - lower_std * std

        return self._get(("bollinger", period, upper_std, lower_std), calc)

    def weekly_macd_hist(self, fast: int, slow: int, signal: int) -> pd.Series:
        def calc() -> pd.Series:
            weekly = self.df["Close"].resample("W-SUN").last().dropna()
            fast_ema = weekly.ewm(span=fast, adjust=False, min_periods=fast).mean()
            slow_ema = weekly.ewm(span=slow, adjust=False, min_periods=slow).mean()
            macd = fast_ema - slow_ema
            sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = macd - sig
            return hist.reindex(self.df.index, method="ffill")

        return self._get(("weekly_macd", fast, slow, signal), calc)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
    "symbol": "Symbol",
    "ticker": "Symbol",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjusted_close": "Adj Close",
    "volume": "Volume",
}


def normalize_ohlcv(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    df = frame.copy()
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data requires a Date/Datetime/Timestamp column or DatetimeIndex")
    else:
        idx = pd.to_datetime(df.index, errors="coerce", utc=True)
        df.index = idx.tz_convert(None)

    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    valid = (
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1))
        & (df["Low"] <= df[["Open", "Close", "High"]].min(axis=1))
    )
    df = df.loc[valid]
    df.attrs["symbol"] = symbol or frame.attrs.get("symbol", "UNKNOWN")
    return df


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow or fastparquet") from exc
    if suffix in {".feather"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported data file: {path}")


def load_data_from_file(path: Path, symbols: Sequence[str] | None) -> dict[str, pd.DataFrame]:
    raw = read_table(path)
    symbol_col = next((c for c in raw.columns if str(c).lower() in {"symbol", "ticker"}), None)
    if symbol_col is None:
        symbol = symbols[0] if symbols and len(symbols) == 1 else path.stem.upper()
        return {symbol: normalize_ohlcv(raw, symbol)}

    result: dict[str, pd.DataFrame] = {}
    raw_symbols = raw[symbol_col].astype(str).str.upper()
    wanted = {s.upper() for s in symbols} if symbols else None
    for symbol in sorted(raw_symbols.unique()):
        if wanted is not None and symbol not in wanted:
            continue
        part = raw.loc[raw_symbols == symbol].drop(columns=[symbol_col])
        result[symbol] = normalize_ohlcv(part, symbol)
    return result


def load_data_from_dir(path: Path, symbols: Sequence[str] | None) -> dict[str, pd.DataFrame]:
    wanted = {s.upper() for s in symbols} if symbols else None
    files = sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".parquet", ".pq", ".feather"}
    )
    result: dict[str, pd.DataFrame] = {}
    for file_path in files:
        symbol = file_path.stem.upper().split("_")[0]
        if wanted is not None and symbol not in wanted:
            continue
        try:
            raw = read_table(file_path)
            result[symbol] = normalize_ohlcv(raw, symbol)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {file_path}: {exc}") from exc
    return result


def download_yfinance(symbols: Sequence[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    """Optional compatibility downloader. Crypto research defaults to Bitvavo."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("--download-source yfinance requires: pip install yfinance") from exc

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        result[symbol.upper()] = normalize_ohlcv(raw, symbol.upper())
    return result


def _timestamp_ms(value: str | None, *, end_of_day: bool = False) -> int | None:
    if not value:
        return None
    ts = pd.Timestamp(value, tz="UTC")
    if end_of_day and len(value.strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return int(ts.timestamp() * 1000)


def _bitvavo_get_json(path: str, params: Mapping[str, Any], timeout: float = 30.0) -> Any:
    url = f"{BITVAVO_API_BASE}/{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": f"stocks-strategy-combo-lab/{VERSION}"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bitvavo HTTP {exc.code} for {path}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bitvavo connection failed for {path}: {exc}") from exc
    parsed = json.loads(payload)
    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(f"Bitvavo error for {path}: {parsed}")
    return parsed


def download_bitvavo(
    symbols: Sequence[str],
    interval: str,
    start: str | None,
    end: str | None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Download public Bitvavo OHLCV candles with backwards pagination.

    Bitvavo returns at most 1,440 candles per request. Missing intervals are not
    fabricated: only candles returned by the venue are retained.
    """
    if interval not in BITVAVO_INTERVALS_MS:
        raise ValueError(f"Unsupported Bitvavo interval {interval!r}; choose from {sorted(BITVAVO_INTERVALS_MS)}")
    interval_ms = BITVAVO_INTERVALS_MS[interval]
    raw_start_ms = _timestamp_ms(start)
    raw_end_ms = _timestamp_ms(end, end_of_day=True) or int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = (raw_start_ms // interval_ms) * interval_ms if raw_start_ms is not None else None
    end_ms = (raw_end_ms // interval_ms) * interval_ms
    result: dict[str, pd.DataFrame] = {}

    for raw_symbol in symbols:
        symbol = raw_symbol.upper().replace("/", "-")
        cursor_end = end_ms
        rows_by_ts: dict[int, list[Any]] = {}
        pages = 0
        while True:
            params: dict[str, Any] = {"interval": interval, "limit": 1440, "end": cursor_end}
            if start_ms is not None:
                params["start"] = start_ms
            payload = _bitvavo_get_json(f"{symbol}/candles", params)
            if not isinstance(payload, list) or not payload:
                break
            pages += 1
            parsed_rows: list[list[Any]] = []
            for row in payload:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                try:
                    ts = int(row[0])
                    parsed = [ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                except (TypeError, ValueError):
                    continue
                if start_ms is not None and ts < start_ms:
                    continue
                if ts <= end_ms:
                    rows_by_ts[ts] = parsed
                    parsed_rows.append(parsed)
            if not parsed_rows:
                break
            oldest = min(int(r[0]) for r in parsed_rows)
            if logger is not None:
                logger.info("Bitvavo %-12s page %3d | candles=%5d | oldest=%s", symbol, pages, len(rows_by_ts), pd.to_datetime(oldest, unit="ms", utc=True))
            if (start_ms is not None and oldest <= start_ms) or len(payload) < 1440:
                break
            next_end = oldest - interval_ms
            if next_end >= cursor_end:
                raise RuntimeError(f"Bitvavo pagination did not advance for {symbol}")
            cursor_end = next_end
            time.sleep(0.08)

        if not rows_by_ts:
            if logger is not None:
                logger.warning("Bitvavo returned no candles for %s", symbol)
            continue
        rows = [rows_by_ts[ts] for ts in sorted(rows_by_ts)]
        raw = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        raw["Date"] = pd.to_datetime(raw["Date"], unit="ms", utc=True)
        frame = normalize_ohlcv(raw, symbol)
        frame.attrs["interval"] = interval
        frame.attrs["venue"] = "BITVAVO"
        result[symbol] = frame
    return result


def prepare_universe(args: argparse.Namespace, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    symbols = [s.upper() for s in parse_csv_list(args.symbols)]
    if args.download:
        if not symbols:
            raise ValueError("--download requires --symbols")
        if args.download_source == "bitvavo":
            data = download_bitvavo(symbols, args.interval, args.start, args.end, logger)
        else:
            data = download_yfinance(symbols, args.start, args.end)
    elif args.data_file:
        data = load_data_from_file(Path(args.data_file), symbols or None)
    elif args.data_dir:
        data = load_data_from_dir(Path(args.data_dir), symbols or None)
    else:
        raise ValueError("Provide --download, --data-file, or --data-dir")

    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    cleaned: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        part = df
        if start is not None:
            part = part.loc[part.index >= start]
        if end is not None:
            part = part.loc[part.index <= end]
        if len(part) < args.min_bars:
            logger.warning("Skipping %s: only %d bars, minimum is %d", symbol, len(part), args.min_bars)
            continue
        cleaned[symbol] = part
        logger.info("Loaded %-12s %7d bars | %s to %s", symbol, len(part), part.index.min().date(), part.index.max().date())

    if not cleaned:
        raise RuntimeError("No usable symbols remained after data validation")
    return cleaned

# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------


def _bundle(df: pd.DataFrame, entry: pd.Series, exit_: pd.Series, cache: IndicatorCache, atr_period: int = 20, **meta: Any) -> SignalBundle:
    index = df.index
    return SignalBundle(
        entry=entry.reindex(index).fillna(False).astype(bool),
        native_exit=exit_.reindex(index).fillna(False).astype(bool),
        atr=cache.atr(atr_period).reindex(index),
        close=df["Close"],
        metadata=meta,
    )


def build_seven_day_pullback(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    n = int(p["lookback"])
    exit_n = int(p["exit_lookback"])
    trend = int(p["trend_sma"])
    prior_low = df["Low"].rolling(n, min_periods=n).min().shift(1)
    prior_high = df["High"].rolling(exit_n, min_periods=exit_n).max().shift(1)
    entry = (df["Close"] < prior_low) & (df["Close"] > c.sma("Close", trend))
    exit_ = df["Close"] > prior_high
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_nbar_contrarian(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    n = int(p["lookback"])
    prior_low = df["Low"].rolling(n, min_periods=n).min().shift(1)
    prior_high = df["High"].rolling(n, min_periods=n).max().shift(1)
    entry = df["Low"] < prior_low
    exit_ = df["High"] > prior_high
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_rsi_oversold(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    rsi = c.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_level"])
    exit_ = df["Close"] > df["High"].shift(1)
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)), rsi_period=int(p["rsi_period"]))


def build_rsi_adx(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    rsi = c.rsi(int(p["rsi_period"]))
    adx, plus_di, minus_di = c.adx(int(p["adx_period"]))
    entry = (rsi < float(p["entry_level"])) & (adx > float(p["adx_min"]))
    direction = str(p.get("direction_filter", "none"))
    if direction == "sma200":
        entry &= df["Close"] > c.sma("Close", 200)
    elif direction == "plus_di":
        entry &= plus_di > minus_di
    elif direction == "sma200_plus_di":
        entry &= (df["Close"] > c.sma("Close", 200)) & (plus_di > minus_di)
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)), rsi_period=int(p["rsi_period"]))


def build_rsi_first_profit(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    rsi = c.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_level"])
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    return _bundle(df, entry, pd.Series(False, index=df.index), c, int(p.get("atr_period", 20)), rsi_period=int(p["rsi_period"]), force_first_profit=True)


def build_rsi_threshold_exit(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    rsi = c.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_level"])
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = rsi > float(p["exit_level"])
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)), rsi_period=int(p["rsi_period"]))


def build_cumulative_rsi(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    rsi = c.rsi(int(p["rsi_period"]))
    cumulative = rsi.rolling(int(p["sum_days"]), min_periods=int(p["sum_days"])).sum()
    entry = cumulative < float(p["sum_threshold"])
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)), rsi_period=int(p["rsi_period"]))


def _consecutive_decreases(series: pd.Series, count: int) -> pd.Series:
    falling = series.diff() < 0
    return falling.rolling(count, min_periods=count).sum() == count


def build_lower_lows(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    entry = _consecutive_decreases(df["Low"], int(p["count"]))
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_lower_closes(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    entry = _consecutive_decreases(df["Close"], int(p["count"]))
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_return_atr_filter(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    ret = df["Close"].pct_change()
    short_atr = c.atr(int(p["atr_short"]))
    long_atr = c.atr(int(p["atr_long"]))
    relation = str(p["atr_relation"])
    vol_condition = short_atr > long_atr if relation == "expanding" else short_atr < long_atr
    entry = (ret <= float(p["return_threshold"])) & vol_condition
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_ema_stretch(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    ema = c.ema("Close", int(p["ema_period"]))
    mode = str(p["stretch_mode"])
    if mode == "percent":
        entry = df["Close"] < ema * (1.0 - float(p["stretch_value"]))
    else:
        atr = c.atr(int(p["atr_period"]))
        entry = (ema - df["Close"]) > float(p["stretch_value"]) * atr
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_mode = str(p.get("native_exit", "ema"))
    exit_ = df["Close"] > ema if exit_mode == "ema" else df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_williams_r(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    wr = c.williams_r(int(p["period"]))
    entry = wr < float(p["entry_level"])
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = wr > float(p["exit_level"])
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_williams_macd(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    wr = c.williams_r(int(p["williams_period"]))
    hist = c.macd_hist(int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    hist_min = hist.rolling(int(p["hist_low_lookback"]), min_periods=int(p["hist_low_lookback"])).min()
    entry = (wr < float(p["entry_level"])) & (hist <= hist_min)
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = hist > hist.shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_inverse_nbar_breakout(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    n = int(p["lookback"])
    entry = df["High"] > df["High"].rolling(n, min_periods=n).max().shift(1)
    exit_ = df["Low"] < df["Low"].rolling(n, min_periods=n).min().shift(1)
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_ma_crossover(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    fast = c.sma("Close", int(p["fast"]))
    slow = c.sma("Close", int(p["slow"]))
    entry = fast > slow
    exit_ = fast < slow
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_asymmetric_ma(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    entry_fast = c.sma("Close", int(p["entry_fast"]))
    entry_slow = c.sma("Close", int(p["entry_slow"]))
    exit_fast = c.sma("Close", int(p["exit_fast"]))
    exit_slow = c.sma("Close", int(p["exit_slow"]))
    entry = entry_fast > entry_slow
    exit_ = exit_fast < exit_slow
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_ma_channel(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    period = int(p["channel_period"])
    confirm = int(p["confirm_bars"])
    high_ma = c.sma("High", period)
    low_ma = c.sma("Low", period)
    above = df["Low"] > high_ma
    below = df["High"] < low_ma
    entry = above.rolling(confirm, min_periods=confirm).sum() == confirm
    exit_ = below.rolling(confirm, min_periods=confirm).sum() == confirm
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_bollinger_breakout(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    upper, lower = c.bollinger(int(p["period"]), float(p["upper_std"]), float(p["lower_std"]))
    entry = df["Close"] > upper
    exit_ = df["Close"] < lower
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_percent_flipper(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    lookback = int(p["lookback"])
    rolling_low = df["Low"].rolling(lookback, min_periods=lookback).min()
    rolling_high = df["High"].rolling(lookback, min_periods=lookback).max()
    entry = df["Close"] >= rolling_low * (1.0 + float(p["up_pct"]))
    exit_ = df["Close"] <= rolling_high * (1.0 - float(p["down_pct"]))
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_triple_screen(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    weekly_hist = c.weekly_macd_hist(int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
    ema = c.ema("Close", int(p["bear_ema"]))
    bear_power = df["Low"] - ema
    entry = (weekly_hist > 0) & (bear_power < 0) & (bear_power > bear_power.shift(1)) & (df["High"] > df["High"].shift(1))
    exit_ = weekly_hist < 0
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_market_structure_atr_pullback(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    breakout_lookback = int(p["breakout_lookback"])
    valid_bars = int(p["valid_bars"])
    atr_period = int(p["atr_period"])
    retrace_atr = float(p["retrace_atr"])
    prior_high = df["High"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
    breakout = df["Close"] > prior_high
    atr = c.atr(atr_period)
    entry = pd.Series(False, index=df.index)
    active_level: float | None = None
    expires = -1
    for i in range(len(df)):
        if bool(breakout.iloc[i]) and np.isfinite(atr.iloc[i]):
            active_level = float(df["Low"].iloc[i] - retrace_atr * atr.iloc[i])
            expires = i + valid_bars
        if active_level is not None and i <= expires and df["Low"].iloc[i] <= active_level <= df["High"].iloc[i]:
            entry.iloc[i] = True
            active_level = None
            expires = -1
        elif active_level is not None and i > expires:
            active_level = None
            expires = -1
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, atr_period)


def _candle_bodies(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    bullish = df["Close"] > df["Open"]
    bearish = df["Close"] < df["Open"]
    body_low = df[["Open", "Close"]].min(axis=1)
    body_high = df[["Open", "Close"]].max(axis=1)
    return bullish, bearish, body_low, body_high


def build_bearish_engulfing_contrarian(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    bullish, bearish, body_low, body_high = _candle_bodies(df)
    pattern = bullish.shift(1) & bearish & (body_high >= body_high.shift(1)) & (body_low <= body_low.shift(1))
    if p.get("trend_sma"):
        pattern &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, pattern, exit_, c, int(p.get("atr_period", 20)))


def build_bullish_engulfing(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    bullish, bearish, body_low, body_high = _candle_bodies(df)
    pattern = bearish.shift(1) & bullish & (body_high >= body_high.shift(1)) & (body_low <= body_low.shift(1))
    if p.get("trend_sma"):
        pattern &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, pattern, exit_, c, int(p.get("atr_period", 20)))


def build_bullish_harami(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    bullish, bearish, body_low, body_high = _candle_bodies(df)
    pattern = bearish.shift(1) & bullish & (body_high < body_high.shift(1)) & (body_low > body_low.shift(1))
    if p.get("trend_sma"):
        pattern &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, pattern, exit_, c, int(p.get("atr_period", 20)))


def build_piercing_line(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    bullish, bearish, _, _ = _candle_bodies(df)
    prev_mid = (df["Open"].shift(1) + df["Close"].shift(1)) / 2.0
    pattern = (
        bearish.shift(1)
        & bullish
        & (df["Open"] < df["Low"].shift(1) * (1.0 + float(p["gap_tolerance"])))
        & (df["Close"] > prev_mid)
        & (df["Close"] < df["Open"].shift(1))
    )
    if p.get("trend_sma"):
        pattern &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, pattern, exit_, c, int(p.get("atr_period", 20)))


def build_month_end_seasonality(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    dates = df.index.to_series()
    next_month = dates.shift(-1).dt.month
    is_month_end_bar = dates.dt.month != next_month
    entry = is_month_end_bar
    if p.get("trend_sma"):
        entry &= df["Close"] > c.sma("Close", int(p["trend_sma"]))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))


def build_thursday_seasonality(df: pd.DataFrame, p: Mapping[str, Any], c: IndicatorCache) -> SignalBundle:
    entry = (df.index.dayofweek == int(p["weekday"])) & (df["Close"] < c.sma("Close", int(p["sma_period"])))
    exit_ = df["Close"] > df["High"].shift(1)
    return _bundle(df, entry, exit_, c, int(p.get("atr_period", 20)))

# ---------------------------------------------------------------------------
# Strategy registry and parameter generation
# ---------------------------------------------------------------------------


def strategy_registry() -> dict[str, StrategySpec]:
    specs = [
        StrategySpec(
            "SEVEN_DAY_PULLBACK", "MEAN_REVERSION", "Close below prior N-day low while above long-term SMA; exit above prior high.",
            {"lookback": 7, "exit_lookback": 7, "trend_sma": 200, "atr_period": 20},
            {"lookback": [5, 7, 10, 14], "exit_lookback": [3, 5, 7, 10], "trend_sma": [100, 150, 200, 250], "atr_period": [14, 20]},
            build_seven_day_pullback,
        ),
        StrategySpec(
            "NBAR_CONTRARIAN", "MEAN_REVERSION", "Buy downside N-bar break as overreaction; exit on upside N-bar break.",
            {"lookback": 5, "atr_period": 20},
            {"lookback": [3, 5, 7, 10, 14, 20], "atr_period": [14, 20]},
            build_nbar_contrarian,
        ),
        StrategySpec(
            "RSI_OVERSOLD", "MEAN_REVERSION", "RSI oversold baseline with next-open execution.",
            {"rsi_period": 2, "entry_level": 15, "trend_sma": None, "atr_period": 20},
            {"rsi_period": [2, 3, 4, 5, 7, 10, 14], "entry_level": [5, 10, 15, 20, 25, 30], "trend_sma": [None, 100, 150, 200], "atr_period": [14, 20]},
            build_rsi_oversold,
        ),
        StrategySpec(
            "RSI_ADX", "MULTI_TIMEFRAME_MEAN_REVERSION", "RSI oversold with ADX trend-strength and optional bullish direction filter.",
            {"rsi_period": 2, "entry_level": 15, "adx_period": 5, "adx_min": 35, "direction_filter": "none", "atr_period": 20},
            {
                "rsi_period": [2, 3, 4, 5],
                "entry_level": [5, 10, 15, 20, 25, 30],
                "adx_period": [5, 7, 10, 14],
                "adx_min": [20, 25, 30, 35, 40, 45, 50],
                "direction_filter": ["none", "sma200", "plus_di", "sma200_plus_di"],
                "atr_period": [14, 20],
            },
            build_rsi_adx,
        ),
        StrategySpec(
            "RSI_FIRST_PROFIT", "MEAN_REVERSION", "Enter oversold and exit at next open after first profitable close.",
            {"rsi_period": 2, "entry_level": 20, "trend_sma": None, "atr_period": 20},
            {"rsi_period": [2, 3, 4, 5], "entry_level": [5, 10, 15, 20, 25, 30], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_rsi_first_profit,
        ),
        StrategySpec(
            "RSI_THRESHOLD_EXIT", "MEAN_REVERSION", "Enter oversold and hold until RSI reaches a high exit threshold.",
            {"rsi_period": 2, "entry_level": 15, "exit_level": 75, "trend_sma": None, "atr_period": 20},
            {"rsi_period": [2, 3, 4, 5], "entry_level": [5, 10, 15, 20, 25, 30], "exit_level": [50, 60, 70, 75, 80, 90], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_rsi_threshold_exit,
        ),
        StrategySpec(
            "CUMULATIVE_RSI", "MEAN_REVERSION", "Sum of short RSI readings identifies persistent weakness.",
            {"rsi_period": 2, "sum_days": 2, "sum_threshold": 20, "trend_sma": 200, "atr_period": 20},
            {"rsi_period": [2, 3, 4], "sum_days": [2, 3, 4], "sum_threshold": [10, 15, 20, 25, 30, 40, 50], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_cumulative_rsi,
        ),
        StrategySpec(
            "LOWER_LOWS", "MEAN_REVERSION", "Buy after N consecutive lower lows.",
            {"count": 3, "trend_sma": 200, "atr_period": 20},
            {"count": [2, 3, 4, 5], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_lower_lows,
        ),
        StrategySpec(
            "LOWER_CLOSES", "MEAN_REVERSION", "Buy after N consecutive lower closes.",
            {"count": 3, "trend_sma": 200, "atr_period": 20},
            {"count": [2, 3, 4, 5], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_lower_closes,
        ),
        StrategySpec(
            "RETURN_ATR_FILTER", "MEAN_REVERSION", "Buy a sharp down day under expanding or contracting ATR conditions.",
            {"return_threshold": -0.01, "atr_short": 5, "atr_long": 10, "atr_relation": "contracting", "trend_sma": 200, "atr_period": 20},
            {"return_threshold": [-0.005, -0.01, -0.015, -0.02, -0.03], "atr_short": [3, 5, 7, 10], "atr_long": [10, 14, 20, 30], "atr_relation": ["expanding", "contracting"], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_return_atr_filter,
        ),
        StrategySpec(
            "EMA_STRETCH", "MEAN_REVERSION", "Buy price stretched below short EMA by percent or ATR distance.",
            {"ema_period": 5, "stretch_mode": "percent", "stretch_value": 0.01, "trend_sma": 200, "native_exit": "ema", "atr_period": 5},
            {"ema_period": [3, 5, 8, 10, 20], "stretch_mode": ["percent", "atr"], "stretch_value": [0.005, 0.01, 0.015, 0.02, 0.5, 0.75, 1.0, 1.5], "trend_sma": [None, 100, 200], "native_exit": ["ema", "prev_high"], "atr_period": [5, 10, 20]},
            build_ema_stretch,
        ),
        StrategySpec(
            "WILLIAMS_R", "MEAN_REVERSION", "Williams %R oversold entry inside long-term uptrend.",
            {"period": 5, "entry_level": -90, "exit_level": -20, "trend_sma": 200, "atr_period": 20},
            {"period": [3, 5, 7, 10, 14, 20], "entry_level": [-80, -85, -90, -95], "exit_level": [-50, -30, -20, -10], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_williams_r,
        ),
        StrategySpec(
            "WILLIAMS_MACD", "MEAN_REVERSION", "Williams %R oversold plus MACD histogram exhaustion.",
            {"williams_period": 5, "entry_level": -90, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "hist_low_lookback": 5, "trend_sma": 200, "atr_period": 20},
            {"williams_period": [3, 5, 7, 10], "entry_level": [-80, -90, -95], "macd_fast": [6, 8, 12], "macd_slow": [19, 26, 35], "macd_signal": [5, 9], "hist_low_lookback": [3, 5, 7, 10], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_williams_macd,
        ),
        StrategySpec(
            "INVERSE_NBAR_BREAKOUT", "TREND", "Buy upside N-bar breakout; exit on downside N-bar break.",
            {"lookback": 5, "trend_sma": None, "atr_period": 20},
            {"lookback": [3, 5, 10, 20, 50, 100], "trend_sma": [None, 100, 200], "atr_period": [14, 20]},
            build_inverse_nbar_breakout,
        ),
        StrategySpec(
            "MA_CROSSOVER", "TREND", "Symmetric moving-average trend following.",
            {"fast": 50, "slow": 200, "atr_period": 20},
            {"fast": [20, 30, 50, 70, 100, 110], "slow": [100, 150, 180, 200, 210, 230, 250], "atr_period": [14, 20]},
            build_ma_crossover,
        ),
        StrategySpec(
            "ASYMMETRIC_MA", "TREND", "Faster entry crossover and slower exit crossover.",
            {"entry_fast": 70, "entry_slow": 210, "exit_fast": 110, "exit_slow": 230, "atr_period": 20},
            {"entry_fast": [20, 50, 70, 100], "entry_slow": [150, 180, 200, 210, 250], "exit_fast": [70, 100, 110, 130], "exit_slow": [200, 210, 230, 250, 300], "atr_period": [14, 20]},
            build_asymmetric_ma,
        ),
        StrategySpec(
            "MA_CHANNEL", "TREND", "N-day high/low moving-average channel with multi-bar confirmation.",
            {"channel_period": 10, "confirm_bars": 5, "atr_period": 20},
            {"channel_period": [5, 10, 15, 20, 30], "confirm_bars": [2, 3, 5, 7, 10], "atr_period": [14, 20]},
            build_ma_channel,
        ),
        StrategySpec(
            "BOLLINGER_BREAKOUT", "TREND", "Long-term high-sigma breakout with asymmetric lower-band exit.",
            {"period": 100, "upper_std": 3.0, "lower_std": 1.0, "atr_period": 20},
            {"period": [20, 50, 75, 100, 150, 200], "upper_std": [1.5, 2.0, 2.5, 3.0], "lower_std": [0.5, 1.0, 1.5, 2.0], "atr_period": [14, 20]},
            build_bollinger_breakout,
        ),
        StrategySpec(
            "PERCENT_FLIPPER", "TREND", "Enter X percent above rolling low and exit X percent below rolling high.",
            {"lookback": 50, "up_pct": 0.20, "down_pct": 0.20, "atr_period": 20},
            {"lookback": [20, 30, 50, 75, 100, 150], "up_pct": [0.10, 0.15, 0.20, 0.25, 0.30], "down_pct": [0.10, 0.15, 0.20, 0.25, 0.30], "atr_period": [14, 20]},
            build_percent_flipper,
        ),
        StrategySpec(
            "TRIPLE_SCREEN", "MULTI_TIMEFRAME_TREND", "Weekly MACD trend plus daily Bear Power recovery and price break.",
            {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bear_ema": 13, "atr_period": 20},
            {"macd_fast": [8, 12], "macd_slow": [19, 26, 35], "macd_signal": [5, 9], "bear_ema": [8, 13, 21], "atr_period": [14, 20]},
            build_triple_screen,
        ),
        StrategySpec(
            "MARKET_STRUCTURE_ATR_PULLBACK", "BREAKOUT_PULLBACK", "Breakout first, then buy a fixed ATR retracement within a validity window.",
            {"breakout_lookback": 63, "valid_bars": 10, "retrace_atr": 2.0, "atr_period": 20},
            {"breakout_lookback": [20, 50, 63, 100, 126], "valid_bars": [5, 10, 15, 20], "retrace_atr": [0.5, 1.0, 1.5, 2.0, 2.5], "atr_period": [14, 20, 30]},
            build_market_structure_atr_pullback,
        ),
        StrategySpec(
            "BEARISH_ENGULFING_CONTRARIAN", "CANDLE_REVERSAL", "Take bearish engulfing as contrarian long overreaction.",
            {"trend_sma": None, "atr_period": 20},
            {"trend_sma": [None, 50, 100, 200], "atr_period": [14, 20]},
            build_bearish_engulfing_contrarian,
        ),
        StrategySpec(
            "BULLISH_ENGULFING", "CANDLE_REVERSAL", "Conventional bullish engulfing long.",
            {"trend_sma": None, "atr_period": 20},
            {"trend_sma": [None, 50, 100, 200], "atr_period": [14, 20]},
            build_bullish_engulfing,
        ),
        StrategySpec(
            "BULLISH_HARAMI", "CANDLE_REVERSAL", "Bullish Harami reversal entry.",
            {"trend_sma": None, "atr_period": 20},
            {"trend_sma": [None, 50, 100, 200], "atr_period": [14, 20]},
            build_bullish_harami,
        ),
        StrategySpec(
            "PIERCING_LINE", "CANDLE_REVERSAL", "Piercing Line reversal entry.",
            {"gap_tolerance": 0.0, "trend_sma": None, "atr_period": 20},
            {"gap_tolerance": [-0.01, -0.005, 0.0, 0.005], "trend_sma": [None, 50, 100, 200], "atr_period": [14, 20]},
            build_piercing_line,
        ),
        StrategySpec(
            "MONTH_END_SEASONALITY", "SEASONALITY", "Enter on final trading day of month, optionally above trend SMA.",
            {"trend_sma": 200, "atr_period": 20},
            {"trend_sma": [None, 50, 100, 150, 200, 250], "atr_period": [14, 20]},
            build_month_end_seasonality,
        ),
        StrategySpec(
            "THURSDAY_SEASONALITY", "SEASONALITY", "Enter on selected weekday below a short SMA.",
            {"weekday": 3, "sma_period": 5, "atr_period": 20},
            {"weekday": [0, 1, 2, 3, 4], "sma_period": [3, 5, 7, 10, 20], "atr_period": [14, 20]},
            build_thursday_seasonality,
        ),
    ]
    return {spec.name: spec for spec in specs}


BLOCKED_STRATEGIES: dict[str, str] = {
    "ROTATIONAL_MOMENTUM": "Portfolio-level cross-sectional ranking requires a separate rebalance engine and is not mixed into per-symbol signal voting.",
    "INVERSE_VOLATILITY_ALLOCATION": "Portfolio weighting overlay, not a standalone entry/exit alpha strategy.",
    "SUPPLY_DEMAND_ZONES": "Swing validation, zone construction, expiry, and impulse rules are insufficiently objective.",
    "INTRADAY_MARKET_STRUCTURE": "Daily-bar runner cannot honestly recreate intraday execution.",
    "SHORT_STRATEGIES": "Program is long-only by design.",
    "SEC_FUNDAMENTAL_INFLECTION": "Requires point-in-time filing fundamentals not present in generic OHLCV input.",
}


def exit_grid(profile: str) -> list[ExitConfig]:
    native = ExitConfig("native", mode="native")
    if profile == "quick":
        return [native, ExitConfig("prev_high", mode="prev_high"), ExitConfig("time_10", mode="native", max_hold_bars=10)]
    standard = [
        native,
        ExitConfig("prev_high", mode="prev_high"),
        ExitConfig("first_profit", mode="first_profit"),
        ExitConfig("rsi_60", mode="rsi", rsi_period=2, rsi_level=60),
        ExitConfig("rsi_75", mode="rsi", rsi_period=2, rsi_level=75),
        ExitConfig("time_5", mode="native", max_hold_bars=5),
        ExitConfig("time_10", mode="native", max_hold_bars=10),
        ExitConfig("atr_stop2_target3", mode="native", stop_atr=2.0, target_atr=3.0),
    ]
    if profile == "standard":
        return standard
    return standard + [
        ExitConfig("time_3", mode="native", max_hold_bars=3),
        ExitConfig("time_20", mode="native", max_hold_bars=20),
        ExitConfig("time_40", mode="native", max_hold_bars=40),
        ExitConfig("rsi_50", mode="rsi", rsi_period=2, rsi_level=50),
        ExitConfig("rsi_70", mode="rsi", rsi_period=2, rsi_level=70),
        ExitConfig("rsi_80", mode="rsi", rsi_period=2, rsi_level=80),
        ExitConfig("atr_stop1.5_target2", mode="native", stop_atr=1.5, target_atr=2.0),
        ExitConfig("atr_stop2_target4", mode="native", stop_atr=2.0, target_atr=4.0),
        ExitConfig("atr_stop2.5_target5", mode="native", stop_atr=2.5, target_atr=5.0),
        ExitConfig("atr_trailing2", mode="native", trailing_atr=2.0),
        ExitConfig("atr_trailing3", mode="native", trailing_atr=3.0),
    ]


def valid_param_combo(strategy: str, params: Mapping[str, Any]) -> bool:
    if strategy == "MA_CROSSOVER":
        return int(params["fast"]) < int(params["slow"])
    if strategy == "ASYMMETRIC_MA":
        return int(params["entry_fast"]) < int(params["entry_slow"]) and int(params["exit_fast"]) < int(params["exit_slow"])
    if strategy == "RETURN_ATR_FILTER":
        return int(params["atr_short"]) < int(params["atr_long"])
    if strategy == "WILLIAMS_MACD":
        return int(params["macd_fast"]) < int(params["macd_slow"])
    if strategy == "TRIPLE_SCREEN":
        return int(params["macd_fast"]) < int(params["macd_slow"])
    if strategy == "EMA_STRETCH":
        value = float(params["stretch_value"])
        return (params["stretch_mode"] == "percent" and value <= 0.05) or (params["stretch_mode"] == "atr" and value >= 0.25)
    return True


def all_param_combinations(spec: StrategySpec) -> list[dict[str, Any]]:
    keys = list(spec.grid.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(spec.grid[k] for k in keys)):
        params = dict(zip(keys, values))
        if valid_param_combo(spec.name, params):
            combos.append(params)
    default = dict(spec.default_params)
    if default not in combos and valid_param_combo(spec.name, default):
        combos.insert(0, default)
    return combos


def deterministic_sample(items: Sequence[Any], limit: int, seed: int, always: Sequence[Any] = ()) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    selected: list[Any] = []
    for item in always:
        if item in items and item not in selected:
            selected.append(item)
    remaining = [item for item in items if item not in selected]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected


def generate_variants(spec: StrategySpec, profile: str, max_variants: int, seed: int) -> list[StrategyVariant]:
    param_combos = all_param_combinations(spec)
    profile_default_cap = {"quick": 3, "standard": 12, "deep": len(param_combos)}.get(profile, len(param_combos))
    requested_cap = len(param_combos) if max_variants <= 0 else max_variants
    parameter_cap = max(1, min(requested_cap, profile_default_cap, len(param_combos)))
    params_selected = deterministic_sample(
        param_combos,
        parameter_cap,
        seed + int(stable_hash(spec.name, 8), 16),
        always=[dict(spec.default_params)],
    )
    exits = exit_grid(profile)
    variants = [
        StrategyVariant(spec.name, spec.family, params, exit_cfg, spec.description)
        for params in params_selected
        for exit_cfg in exits
    ]
    cap = max_variants if max_variants > 0 else len(variants)
    default_variant = StrategyVariant(spec.name, spec.family, dict(spec.default_params), ExitConfig("native", mode="native"), spec.description)
    return deterministic_sample(
        variants,
        cap,
        seed + int(stable_hash(spec.name + "_variants", 8), 16),
        always=[default_variant],
    )

# ---------------------------------------------------------------------------
# Signal combination and event-driven backtest
# ---------------------------------------------------------------------------


def apply_exit_overlay(df: pd.DataFrame, bundle: SignalBundle, config: ExitConfig, cache: IndicatorCache) -> pd.Series:
    if config.mode == "native":
        return bundle.native_exit.copy()
    if config.mode == "prev_high":
        return (df["Close"] > df["High"].shift(1)).fillna(False)
    if config.mode == "rsi":
        period = int(config.rsi_period or 2)
        level = float(config.rsi_level or 70)
        return (cache.rsi(period) > level).fillna(False)
    if config.mode == "first_profit":
        return pd.Series(False, index=df.index)
    raise ValueError(f"Unknown exit mode: {config.mode}")


def combine_signal_bundles(
    bundles: Sequence[SignalBundle],
    entry_mode: str,
    exit_vote: str,
) -> SignalBundle:
    if not bundles:
        raise ValueError("Cannot combine zero signal bundles")
    entries = pd.concat([b.entry.astype(int) for b in bundles], axis=1).fillna(0)
    exits = pd.concat([b.native_exit.astype(int) for b in bundles], axis=1).fillna(0)
    n = len(bundles)
    if entry_mode == "all":
        entry = entries.sum(axis=1) == n
    elif entry_mode == "majority":
        entry = entries.sum(axis=1) >= (n // 2 + 1)
    elif entry_mode == "any":
        entry = entries.sum(axis=1) >= 1
    else:
        raise ValueError(f"Unknown entry combination mode: {entry_mode}")

    if exit_vote == "all":
        exit_ = exits.sum(axis=1) == n
    elif exit_vote == "majority":
        exit_ = exits.sum(axis=1) >= (n // 2 + 1)
    elif exit_vote == "any":
        exit_ = exits.sum(axis=1) >= 1
    else:
        raise ValueError(f"Unknown exit vote: {exit_vote}")

    atr = pd.concat([b.atr for b in bundles], axis=1).median(axis=1)
    return SignalBundle(entry.astype(bool), exit_.astype(bool), atr, bundles[0].close, {"component_count": n})


def commission_cost(notional: float, model: CostModel) -> float:
    return max(float(model.fixed_fee), abs(float(notional)) * model.commission_bps / 10000.0)


def buy_fill_price(open_price: float, model: CostModel) -> float:
    return float(open_price) * (1.0 + model.slippage_bps / 10000.0)


def sell_fill_price(raw_price: float, model: CostModel) -> float:
    return float(raw_price) * (1.0 - model.slippage_bps / 10000.0)


def execute_symbol_backtest(
    result_id: str,
    symbol: str,
    df: pd.DataFrame,
    bundle: SignalBundle,
    exit_config: ExitConfig,
    initial_capital: float,
    cost_model: CostModel,
    whole_shares: bool,
    conservative_intrabar: bool,
) -> tuple[pd.Series, list[Trade], float]:
    if df.empty:
        return pd.Series(dtype=float), [], 0.0

    cache = IndicatorCache(df)
    signal_exit = apply_exit_overlay(df, bundle, exit_config, cache)
    entry_signal = bundle.entry.fillna(False).astype(bool)
    atr = bundle.atr.reindex(df.index)

    cash = float(initial_capital)
    qty = 0.0
    entry_price = 0.0
    entry_total_cost = 0.0
    entry_fee = 0.0
    entry_date: pd.Timestamp | None = None
    entry_atr = np.nan
    stop_price: float | None = None
    target_price: float | None = None
    trailing_peak: float | None = None
    bars_held = 0
    pending_exit = False
    pending_exit_reason = "signal"
    entered_previous_open = False
    position_days = 0
    mae = 0.0
    mfe = 0.0
    trades: list[Trade] = []
    equity_values: list[float] = []

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    entries = entry_signal.to_numpy(bool)
    exits = signal_exit.to_numpy(bool)
    atr_values = atr.to_numpy(float)
    index = df.index

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, entry_price, entry_total_cost, entry_fee, entry_date
        nonlocal stop_price, target_price, trailing_peak, bars_held, mae, mfe
        if qty <= 0.0 or entry_date is None:
            return
        fill = sell_fill_price(max(raw_price, EPS), cost_model)
        gross_proceeds = qty * fill
        exit_fee = commission_cost(gross_proceeds, cost_model)
        proceeds = gross_proceeds - exit_fee
        cash += proceeds
        gross_pnl = qty * (fill - entry_price)
        net_pnl = proceeds - entry_total_cost
        net_return = net_pnl / max(entry_total_cost, EPS)
        trades.append(
            Trade(
                result_id=result_id,
                symbol=symbol,
                entry_date=entry_date.isoformat(),
                exit_date=index[i].isoformat(),
                entry_price=float(entry_price),
                exit_price=float(fill),
                quantity=float(qty),
                gross_pnl=float(gross_pnl),
                net_pnl=float(net_pnl),
                net_return=float(net_return),
                entry_fee=float(entry_fee),
                exit_fee=float(exit_fee),
                bars_held=int(bars_held),
                exit_reason=reason,
                mae=float(mae),
                mfe=float(mfe),
            )
        )
        qty = 0.0
        entry_price = 0.0
        entry_total_cost = 0.0
        entry_fee = 0.0
        entry_date = None
        stop_price = None
        target_price = None
        trailing_peak = None
        bars_held = 0
        mae = 0.0
        mfe = 0.0

    for i in range(len(df)):
        open_price = opens[i]
        high = highs[i]
        low = lows[i]
        close = closes[i]
        entered_this_open = False

        if qty > 0.0 and pending_exit:
            close_position(i, open_price, pending_exit_reason)
            pending_exit = False
            pending_exit_reason = "signal"

        if qty <= 0.0 and i > 0 and entries[i - 1]:
            fill = buy_fill_price(open_price, cost_model)
            max_notional = cash / (1.0 + cost_model.commission_bps / 10000.0)
            raw_qty = max_notional / max(fill, EPS)
            new_qty = math.floor(raw_qty) if whole_shares else raw_qty
            if new_qty > 0.0:
                notional = new_qty * fill
                fee = commission_cost(notional, cost_model)
                total_cost = notional + fee
                if total_cost <= cash + 1e-8:
                    cash -= total_cost
                    qty = float(new_qty)
                    entry_price = float(fill)
                    entry_total_cost = float(total_cost)
                    entry_fee = float(fee)
                    entry_date = index[i]
                    entry_atr = float(atr_values[i - 1]) if i > 0 and np.isfinite(atr_values[i - 1]) else np.nan
                    stop_price = entry_price - float(exit_config.stop_atr) * entry_atr if exit_config.stop_atr and np.isfinite(entry_atr) else None
                    target_price = entry_price + float(exit_config.target_atr) * entry_atr if exit_config.target_atr and np.isfinite(entry_atr) else None
                    trailing_peak = entry_price
                    bars_held = 0
                    mae = min(0.0, low / entry_price - 1.0)
                    mfe = max(0.0, high / entry_price - 1.0)
                    entered_this_open = True

        if qty > 0.0:
            position_days += 1
            bars_held += 1
            trailing_peak = max(float(trailing_peak or entry_price), high)
            mae = min(mae, low / entry_price - 1.0)
            mfe = max(mfe, high / entry_price - 1.0)

            if exit_config.trailing_atr and np.isfinite(atr_values[i]):
                trail = trailing_peak - float(exit_config.trailing_atr) * float(atr_values[i])
                stop_price = max(stop_price or -np.inf, trail)

            hit_stop = stop_price is not None and low <= stop_price
            hit_target = target_price is not None and high >= target_price
            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    reason = "stop_and_target_same_bar_stop_first" if conservative_intrabar else "stop_and_target_same_bar_target_first"
                    raw_exit = stop_price if conservative_intrabar else target_price
                elif hit_stop:
                    reason = "atr_stop"
                    raw_exit = open_price if open_price <= float(stop_price) else float(stop_price)
                else:
                    reason = "atr_target"
                    raw_exit = open_price if open_price >= float(target_price) else float(target_price)
                close_position(i, float(raw_exit), reason)

        if qty > 0.0:
            if exit_config.mode == "first_profit" or bool(bundle.metadata.get("force_first_profit", False)):
                if close > entry_price:
                    pending_exit = True
                    pending_exit_reason = "first_profitable_close"
            elif exits[i]:
                pending_exit = True
                pending_exit_reason = f"{exit_config.mode}_signal"

            if exit_config.max_hold_bars and bars_held >= int(exit_config.max_hold_bars):
                pending_exit = True
                pending_exit_reason = "time_stop"

        mark_equity = cash + qty * close
        equity_values.append(float(mark_equity))
        entered_previous_open = entered_this_open

    if qty > 0.0:
        close_position(len(df) - 1, closes[-1], "end_of_sample_forced_liquidation")
        equity_values[-1] = float(cash)

    equity = pd.Series(equity_values, index=index, name=symbol, dtype=float)
    exposure = position_days / max(len(df), 1)
    return equity, trades, float(exposure)


def aggregate_symbol_curves(curves: Mapping[str, pd.Series], initial_per_symbol: float) -> pd.Series:
    if not curves:
        return pd.Series(dtype=float)
    union = sorted(set().union(*(series.index for series in curves.values())))
    union_index = pd.DatetimeIndex(union)
    aligned: list[pd.Series] = []
    for symbol, curve in curves.items():
        series = curve.reindex(union_index).ffill().fillna(initial_per_symbol)
        aligned.append(series.rename(symbol))
    return pd.concat(aligned, axis=1).sum(axis=1).rename("PortfolioEquity")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak.replace(0.0, np.nan) - 1.0
    return float(abs(dd.min())) if not dd.empty else 0.0


def profit_factor_from_values(values: Sequence[float]) -> float:
    wins = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    if losses <= EPS:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 365.25
    ns = pd.Series(index).sort_values().diff().dropna().dt.total_seconds()
    ns = ns[ns > 0]
    if ns.empty:
        return 365.25
    median_seconds = float(ns.median())
    periods = 365.25 * 24.0 * 3600.0 / median_seconds
    return float(min(max(periods, 1.0), 600_000.0))


def annualized_sharpe(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3 or clean.std(ddof=1) <= EPS:
        return 0.0
    return float(clean.mean() / clean.std(ddof=1) * math.sqrt(periods_per_year))


def annualized_sortino(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    downside = clean[clean < 0]
    if len(clean) < 3 or len(downside) < 2 or downside.std(ddof=1) <= EPS:
        return 0.0
    return float(clean.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))


def deflated_sharpe_probability(returns: pd.Series, trials: int) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 10 or clean.std(ddof=1) <= EPS or norm is None:
        return 0.0
    daily_sr = float(clean.mean() / clean.std(ddof=1))
    skew = float(clean.skew()) if np.isfinite(clean.skew()) else 0.0
    kurt = float(clean.kurtosis() + 3.0) if np.isfinite(clean.kurtosis()) else 3.0
    trials = max(int(trials), 1)
    if trials <= 1:
        benchmark = 0.0
    else:
        euler_gamma = 0.5772156649015329
        a = float(norm.ppf(1.0 - 1.0 / trials))
        b = float(norm.ppf(1.0 - 1.0 / (trials * math.e)))
        benchmark = ((1.0 - euler_gamma) * a + euler_gamma * b) / math.sqrt(max(n - 1, 1))
    denominator = math.sqrt(max(1.0 - skew * daily_sr + ((kurt - 1.0) / 4.0) * daily_sr * daily_sr, EPS))
    z = (daily_sr - benchmark) * math.sqrt(n - 1) / denominator
    return float(norm.cdf(z))


def chronological_fold_metrics(equity: pd.Series, trades: Sequence[Trade], folds: int = 6) -> tuple[list[dict[str, Any]], int]:
    if equity.empty:
        return [], 0
    boundaries = np.linspace(0, len(equity), folds + 1, dtype=int)
    results: list[dict[str, Any]] = []
    positive = 0
    for fold in range(folds):
        start_i, end_i = int(boundaries[fold]), int(boundaries[fold + 1])
        if end_i - start_i < 2:
            continue
        segment = equity.iloc[start_i:end_i]
        start_date = segment.index.min()
        end_date = segment.index.max()
        fold_trades = [t for t in trades if start_date <= pd.Timestamp(t.exit_date) <= end_date]
        pnl_values = [t.net_pnl for t in fold_trades]
        fold_return = float(segment.iloc[-1] / max(segment.iloc[0], EPS) - 1.0)
        pf = profit_factor_from_values(pnl_values)
        if fold_return > 0:
            positive += 1
        results.append(
            {
                "fold": fold + 1,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "return": fold_return,
                "trades": len(fold_trades),
                "profit_factor": pf,
            }
        )
    return results, positive


def calculate_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    exposure: float,
    initial_capital: float,
    global_trials: int,
) -> dict[str, Any]:
    if equity.empty:
        return {
            "ending_equity": initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "trades": 0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "expectancy_bps": 0.0,
            "positive_folds": 0,
            "dsr_probability": 0.0,
        }

    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    periods_per_year = infer_periods_per_year(equity.index)
    total_return = float(equity.iloc[-1] / max(initial_capital, EPS) - 1.0)
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = float((equity.iloc[-1] / max(initial_capital, EPS)) ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    dd = max_drawdown(equity)
    sharpe = annualized_sharpe(daily_returns, periods_per_year)
    sortino = annualized_sortino(daily_returns, periods_per_year)
    trade_returns = [t.net_return for t in trades]
    trade_pnl = [t.net_pnl for t in trades]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    fold_data, positive_folds = chronological_fold_metrics(equity, trades, 6)
    monthly = equity.resample("ME").last().pct_change().dropna()
    total_costs = sum(t.entry_fee + t.exit_fee for t in trades)
    gross_profit = sum(max(t.gross_pnl, 0.0) for t in trades)
    gross_loss = -sum(min(t.gross_pnl, 0.0) for t in trades)
    trade_pf = profit_factor_from_values(trade_pnl)
    gross_pf = gross_profit / gross_loss if gross_loss > EPS else (999.0 if gross_profit > 0 else 0.0)
    avg_hold = float(np.mean([t.bars_held for t in trades])) if trades else 0.0
    median_hold = float(np.median([t.bars_held for t in trades])) if trades else 0.0
    avg_mae = float(np.mean([t.mae for t in trades])) if trades else 0.0
    avg_mfe = float(np.mean([t.mfe for t in trades])) if trades else 0.0
    expectancy_bps = float(np.mean(trade_returns) * 10000.0) if trades else 0.0
    win_rate = len(wins) / len(trades) if trades else 0.0
    payoff_ratio = (float(np.mean(wins)) / abs(float(np.mean(losses)))) if wins and losses and abs(float(np.mean(losses))) > EPS else 0.0
    dsr = deflated_sharpe_probability(daily_returns, max(global_trials, 1))

    return {
        "ending_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": cagr / dd if dd > EPS else 0.0,
        "annual_volatility": float(daily_returns.std(ddof=1) * math.sqrt(periods_per_year)),
        "periods_per_year": periods_per_year,
        "trades": len(trades),
        "gross_profit_factor": float(gross_pf),
        "profit_factor": float(trade_pf),
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "expectancy_bps": expectancy_bps,
        "average_trade_return": float(np.mean(trade_returns)) if trades else 0.0,
        "median_trade_return": float(np.median(trade_returns)) if trades else 0.0,
        "best_trade": float(max(trade_returns)) if trades else 0.0,
        "worst_trade": float(min(trade_returns)) if trades else 0.0,
        "average_hold_bars": avg_hold,
        "median_hold_bars": median_hold,
        "average_mae": avg_mae,
        "average_mfe": avg_mfe,
        "exposure": float(exposure),
        "total_costs": float(total_costs),
        "costs_pct_initial": float(total_costs / max(initial_capital, EPS)),
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "positive_folds": int(positive_folds),
        "folds": fold_data,
        "dsr_probability": dsr,
    }


def robust_score(metrics: Mapping[str, Any], min_trades: int) -> float:
    trades = int(metrics.get("trades", 0))
    pf = min(float(metrics.get("profit_factor", 0.0)), 5.0)
    expectancy = float(metrics.get("expectancy_bps", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    dd = float(metrics.get("max_drawdown", 1.0))
    folds = int(metrics.get("positive_folds", 0))
    dsr = float(metrics.get("dsr_probability", 0.0))
    sample_penalty = min(trades / max(min_trades, 1), 1.0)
    if pf <= 0:
        log_pf = -3.0
    else:
        log_pf = math.log(max(pf, 0.05))
    base_score = (
        2.0 * log_pf
        + 0.40 * sharpe
        + 0.008 * expectancy
        - 2.5 * dd
        + 0.18 * folds
        + 0.50 * dsr
    )
    # A spectacular PF on one or two trades is not a serious research result.
    # Scale the score by sample support and apply an explicit insufficiency penalty.
    score = base_score * sample_penalty - 4.0 * (1.0 - sample_penalty)
    return float(score)


def monte_carlo_trade_bootstrap(
    trades: Sequence[Trade],
    paths: int,
    seed: int,
    block_length: int = 5,
) -> dict[str, Any]:
    returns = np.asarray([t.net_return for t in trades], dtype=float)
    if paths <= 0 or len(returns) < 2:
        return {}
    rng = np.random.default_rng(seed)
    n = len(returns)
    terminal: list[float] = []
    max_dds: list[float] = []
    loss_count = 0
    for _ in range(paths):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            for j in range(block_length):
                sampled.append(float(returns[(start + j) % n]))
                if len(sampled) >= n:
                    break
        curve = np.cumprod(1.0 + np.asarray(sampled))
        peak = np.maximum.accumulate(curve)
        dd = 1.0 - curve / np.maximum(peak, EPS)
        terminal_return = float(curve[-1] - 1.0)
        terminal.append(terminal_return)
        max_dds.append(float(dd.max()))
        if terminal_return < 0:
            loss_count += 1
    return {
        "paths": int(paths),
        "block_length": int(block_length),
        "terminal_loss_probability": float(loss_count / paths),
        "median_terminal_return": float(np.median(terminal)),
        "p05_terminal_return": float(np.quantile(terminal, 0.05)),
        "p95_terminal_return": float(np.quantile(terminal, 0.95)),
        "median_max_drawdown": float(np.median(max_dds)),
        "p95_max_drawdown": float(np.quantile(max_dds, 0.95)),
    }

# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def build_signals_for_variant(
    variant: StrategyVariant,
    data: Mapping[str, pd.DataFrame],
    registry: Mapping[str, StrategySpec],
) -> dict[str, SignalBundle]:
    spec = registry[variant.strategy]
    output: dict[str, SignalBundle] = {}
    for symbol, df in data.items():
        cache = IndicatorCache(df)
        output[symbol] = spec.builder(df, variant.params, cache)
    return output


def run_variant_backtest(
    variant: StrategyVariant,
    data: Mapping[str, pd.DataFrame],
    registry: Mapping[str, StrategySpec],
    initial_capital: float,
    cost_model: CostModel,
    cost_scenario: str,
    whole_shares: bool,
    conservative_intrabar: bool,
    global_trials: int,
) -> BacktestOutput:
    signals = build_signals_for_variant(variant, data, registry)
    per_symbol_capital = initial_capital / len(data)
    curves: dict[str, pd.Series] = {}
    all_trades: list[Trade] = []
    exposures: list[float] = []
    for symbol, df in data.items():
        curve, trades, exposure = execute_symbol_backtest(
            variant.id,
            symbol,
            df,
            signals[symbol],
            variant.exit_config,
            per_symbol_capital,
            cost_model,
            whole_shares,
            conservative_intrabar,
        )
        curves[symbol] = curve
        all_trades.extend(trades)
        exposures.append(exposure)
    equity = aggregate_symbol_curves(curves, per_symbol_capital)
    exposure = float(np.mean(exposures)) if exposures else 0.0
    metrics = calculate_metrics(equity, all_trades, exposure, initial_capital, global_trials)
    return BacktestOutput(
        result_id=variant.id,
        result_type="single",
        label=variant.label,
        strategy_names=[variant.strategy],
        params={"strategy_params": dict(variant.params), "exit": variant.exit_config.as_dict()},
        equity=equity,
        trades=all_trades,
        exposure=exposure,
        cost_scenario=cost_scenario,
        metrics=metrics,
    )


def run_combo_backtest(
    combo: ComboVariant,
    data: Mapping[str, pd.DataFrame],
    registry: Mapping[str, StrategySpec],
    initial_capital: float,
    cost_model: CostModel,
    cost_scenario: str,
    whole_shares: bool,
    conservative_intrabar: bool,
    global_trials: int,
) -> BacktestOutput:
    component_signals: list[dict[str, SignalBundle]] = []
    for component in combo.components:
        native_component = StrategyVariant(
            strategy=component.strategy,
            family=component.family,
            params=component.params,
            exit_config=ExitConfig("native", mode="native"),
            description=component.description,
        )
        component_signals.append(build_signals_for_variant(native_component, data, registry))

    per_symbol_capital = initial_capital / len(data)
    curves: dict[str, pd.Series] = {}
    all_trades: list[Trade] = []
    exposures: list[float] = []
    combo_exit = ExitConfig(
        name="combo_native",
        mode="native",
        max_hold_bars=combo.max_hold_bars,
        stop_atr=combo.stop_atr,
        target_atr=combo.target_atr,
    )
    for symbol, df in data.items():
        bundles = [signals[symbol] for signals in component_signals]
        merged = combine_signal_bundles(bundles, combo.entry_mode, combo.exit_vote)
        curve, trades, exposure = execute_symbol_backtest(
            combo.id,
            symbol,
            df,
            merged,
            combo_exit,
            per_symbol_capital,
            cost_model,
            whole_shares,
            conservative_intrabar,
        )
        curves[symbol] = curve
        all_trades.extend(trades)
        exposures.append(exposure)
    equity = aggregate_symbol_curves(curves, per_symbol_capital)
    exposure = float(np.mean(exposures)) if exposures else 0.0
    metrics = calculate_metrics(equity, all_trades, exposure, initial_capital, global_trials)
    return BacktestOutput(
        result_id=combo.id,
        result_type=f"combo_{len(combo.components)}",
        label=combo.label,
        strategy_names=[v.strategy for v in combo.components],
        params={
            "components": [
                {"strategy": v.strategy, "family": v.family, "params": dict(v.params)}
                for v in combo.components
            ],
            "entry_mode": combo.entry_mode,
            "exit_vote": combo.exit_vote,
            "max_hold_bars": combo.max_hold_bars,
            "stop_atr": combo.stop_atr,
            "target_atr": combo.target_atr,
        },
        equity=equity,
        trades=all_trades,
        exposure=exposure,
        cost_scenario=cost_scenario,
        metrics=metrics,
    )


def output_to_row(output: BacktestOutput, min_trades: int) -> dict[str, Any]:
    metrics_flat = {k: v for k, v in output.metrics.items() if k != "folds"}
    row = {
        "result_id": output.result_id,
        "result_type": output.result_type,
        "label": output.label,
        "strategies": ",".join(output.strategy_names),
        "cost_scenario": output.cost_scenario,
        "params_json": stable_json(output.params),
        **metrics_flat,
    }
    row["sample_valid"] = int(output.metrics.get("trades", 0)) >= int(min_trades)
    row["robust_score"] = robust_score(output.metrics, min_trades)
    row["completed_at_utc"] = utc_now()
    return row


def append_checkpoint(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = f"{row['result_id']}::{row['cost_scenario']}"
                completed[key] = row
            except Exception:
                continue
    return completed


def should_skip(ctx: RunContext, result_id: str, cost_scenario: str) -> bool:
    return f"{result_id}::{cost_scenario}" in ctx.completed


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    started: float,
    row: Mapping[str, Any] | None,
    phase: str,
) -> None:
    elapsed = max(time.time() - started, 0.001)
    speed = current / elapsed
    remaining = (total - current) / speed if speed > 0 else math.inf
    eta = f"{remaining / 60.0:.1f}m" if np.isfinite(remaining) else "?"
    if row:
        logger.info(
            "%s [%d/%d] %.2f jobs/s ETA %s | %s | PF %.3f | DD %.1f%% | trades %d | score %.3f",
            phase,
            current,
            total,
            speed,
            eta,
            row.get("label", row.get("result_id", ""))[:100],
            float(row.get("profit_factor", 0.0)),
            100.0 * float(row.get("max_drawdown", 0.0)),
            int(row.get("trades", 0)),
            float(row.get("robust_score", 0.0)),
        )
    else:
        logger.info("%s [%d/%d] %.2f jobs/s ETA %s", phase, current, total, speed, eta)


def run_single_phase(
    ctx: RunContext,
    variants: Sequence[StrategyVariant],
    registry: Mapping[str, StrategySpec],
    cost_models: Mapping[str, CostModel],
) -> tuple[list[dict[str, Any]], dict[str, StrategyVariant]]:
    jobs = [(variant, scenario, model) for variant in variants for scenario, model in cost_models.items()]
    total = len(jobs)
    started = time.time()
    rows: list[dict[str, Any]] = []
    variant_map = {v.id: v for v in variants}
    ctx.logger.info("Starting singles phase: %d variants x %d cost scenarios = %d jobs", len(variants), len(cost_models), total)

    for current, (variant, scenario, model) in enumerate(jobs, start=1):
        key = f"{variant.id}::{scenario}"
        if should_skip(ctx, variant.id, scenario):
            row = ctx.completed[key]
            rows.append(row)
            if current % ctx.args.log_every == 0 or current == total:
                log_progress(ctx.logger, current, total, started, row, "SINGLES resume")
            continue
        try:
            ctx.global_trials += 1
            output = run_variant_backtest(
                variant,
                ctx.data,
                registry,
                ctx.args.capital,
                model,
                scenario,
                ctx.args.whole_shares,
                not ctx.args.target_first_intrabar,
                max(ctx.global_trials, len(variants)),
            )
            row = output_to_row(output, ctx.args.min_trades)
            row["folds_json"] = stable_json(output.metrics.get("folds", []))
            append_checkpoint(ctx.checkpoint_path, row)
            ctx.completed[key] = row
            rows.append(row)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            ctx.logger.exception("Single failed: %s", variant.label)
            row = {
                "result_id": variant.id,
                "result_type": "single",
                "label": variant.label,
                "strategies": variant.strategy,
                "cost_scenario": scenario,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "robust_score": -999.0,
                "completed_at_utc": utc_now(),
            }
            append_checkpoint(ctx.checkpoint_path, row)
            ctx.completed[key] = row
            rows.append(row)
        if current % ctx.args.log_every == 0 or current == total:
            log_progress(ctx.logger, current, total, started, rows[-1], "SINGLES")
    return rows, variant_map


def shortlist_variants(
    rows: Sequence[Mapping[str, Any]],
    variant_map: Mapping[str, StrategyVariant],
    scenario: str,
    shortlist_size: int,
    top_per_strategy: int,
    min_trades: int,
) -> list[StrategyVariant]:
    valid = [
        row for row in rows
        if row.get("cost_scenario") == scenario
        and row.get("status") != "ERROR"
        and row.get("result_id") in variant_map
    ]
    valid.sort(key=lambda r: float(r.get("robust_score", -999.0)), reverse=True)
    per_strategy: defaultdict[str, int] = defaultdict(int)
    selected: list[StrategyVariant] = []
    for row in valid:
        variant = variant_map[str(row["result_id"])]
        if variant.exit_config.mode != "native" or variant.exit_config.name != "native":
            continue
        if per_strategy[variant.strategy] >= top_per_strategy:
            continue
        selected.append(variant)
        per_strategy[variant.strategy] += 1
        if len(selected) >= shortlist_size:
            break
    return selected


def estimate_combo_count(shortlist: Sequence[StrategyVariant], sizes: Sequence[int], modes: Sequence[str], distinct_families: bool) -> int:
    count = 0
    for size in sizes:
        for combo in itertools.combinations(shortlist, size):
            if distinct_families and len({v.family for v in combo}) != size:
                continue
            if len({v.strategy for v in combo}) != size:
                continue
            count += len(modes)
    return count


def generate_combos(
    shortlist: Sequence[StrategyVariant],
    sizes: Sequence[int],
    modes: Sequence[str],
    exit_vote: str,
    distinct_families: bool,
    max_combos: int,
    seed: int,
    combo_max_hold: int | None,
    combo_stop_atr: float | None,
    combo_target_atr: float | None,
) -> list[ComboVariant]:
    combos: list[ComboVariant] = []
    for size in sizes:
        for components in itertools.combinations(shortlist, size):
            if len({v.strategy for v in components}) != size:
                continue
            if distinct_families and len({v.family for v in components}) != size:
                continue
            for mode in modes:
                combos.append(
                    ComboVariant(
                        components=tuple(components),
                        entry_mode=mode,
                        exit_vote=exit_vote,
                        max_hold_bars=combo_max_hold,
                        stop_atr=combo_stop_atr,
                        target_atr=combo_target_atr,
                    )
                )
    return deterministic_sample(combos, max_combos, seed)


def run_combo_phase(
    ctx: RunContext,
    combos: Sequence[ComboVariant],
    registry: Mapping[str, StrategySpec],
    cost_models: Mapping[str, CostModel],
) -> tuple[list[dict[str, Any]], dict[str, ComboVariant]]:
    jobs = [(combo, scenario, model) for combo in combos for scenario, model in cost_models.items()]
    total = len(jobs)
    started = time.time()
    rows: list[dict[str, Any]] = []
    combo_map = {c.id: c for c in combos}
    ctx.logger.info("Starting combo phase: %d combos x %d cost scenarios = %d jobs", len(combos), len(cost_models), total)
    for current, (combo, scenario, model) in enumerate(jobs, start=1):
        key = f"{combo.id}::{scenario}"
        if should_skip(ctx, combo.id, scenario):
            row = ctx.completed[key]
            rows.append(row)
            if current % ctx.args.log_every == 0 or current == total:
                log_progress(ctx.logger, current, total, started, row, "COMBOS resume")
            continue
        try:
            ctx.global_trials += 1
            output = run_combo_backtest(
                combo,
                ctx.data,
                registry,
                ctx.args.capital,
                model,
                scenario,
                ctx.args.whole_shares,
                not ctx.args.target_first_intrabar,
                max(ctx.global_trials, len(combos)),
            )
            row = output_to_row(output, ctx.args.min_trades)
            row["folds_json"] = stable_json(output.metrics.get("folds", []))
            append_checkpoint(ctx.checkpoint_path, row)
            ctx.completed[key] = row
            rows.append(row)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            ctx.logger.exception("Combo failed: %s", combo.label)
            row = {
                "result_id": combo.id,
                "result_type": f"combo_{len(combo.components)}",
                "label": combo.label,
                "strategies": ",".join(v.strategy for v in combo.components),
                "cost_scenario": scenario,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "robust_score": -999.0,
                "completed_at_utc": utc_now(),
            }
            append_checkpoint(ctx.checkpoint_path, row)
            ctx.completed[key] = row
            rows.append(row)
        if current % ctx.args.log_every == 0 or current == total:
            log_progress(ctx.logger, current, total, started, rows[-1], "COMBOS")
    return rows, combo_map

# ---------------------------------------------------------------------------
# Reporting and top-result detail export
# ---------------------------------------------------------------------------


def result_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    preferred = [
        "result_id",
        "result_type",
        "label",
        "strategies",
        "cost_scenario",
        "sample_valid",
        "robust_score",
        "trades",
        "profit_factor",
        "gross_profit_factor",
        "expectancy_bps",
        "win_rate",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "positive_folds",
        "dsr_probability",
        "exposure",
        "average_hold_bars",
        "average_mae",
        "average_mfe",
        "total_costs",
        "params_json",
        "completed_at_utc",
    ]
    ordered = [c for c in preferred if c in frame.columns] + [c for c in frame.columns if c not in preferred]
    return frame[ordered]


def display_number(value: Any, pct: bool = False) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pct:
        return f"{100.0 * number:.2f}%"
    return f"{number:.4f}"


def write_html_report(
    path: Path,
    args: argparse.Namespace,
    all_results: pd.DataFrame,
    top_results: pd.DataFrame,
    run_metadata: Mapping[str, Any],
) -> None:
    def table_html(frame: pd.DataFrame, limit: int = 100) -> str:
        if frame.empty:
            return "<p>No results.</p>"
        columns = [
            c for c in [
                "result_type", "label", "cost_scenario", "robust_score", "trades",
                "profit_factor", "expectancy_bps", "total_return", "max_drawdown",
                "sharpe", "positive_folds", "dsr_probability"
            ] if c in frame.columns
        ]
        shown = frame.head(limit)[columns].copy()
        for c in ["total_return", "max_drawdown", "dsr_probability"]:
            if c in shown.columns:
                shown[c] = shown[c].map(lambda x: display_number(x, pct=True))
        for c in ["robust_score", "profit_factor", "expectancy_bps", "sharpe"]:
            if c in shown.columns:
                shown[c] = shown[c].map(display_number)
        return shown.to_html(index=False, escape=True, border=0, classes="results")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Strategy Combo Research Lab Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
h1, h2 {{ margin-top: 1.4em; }}
code, pre {{ background: #f5f5f5; padding: 2px 5px; }}
pre {{ overflow-x: auto; padding: 14px; }}
table.results {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
table.results th, table.results td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
table.results th {{ background: #f0f0f0; position: sticky; top: 0; }}
.note {{ border-left: 4px solid #555; padding: 10px 14px; background: #f8f8f8; }}
</style>
</head>
<body>
<h1>Strategy Combo Research Lab</h1>
<p>Version {VERSION}. Created {utc_now()}.</p>
<div class="note"><strong>Research only.</strong> High-ranked results are hypotheses, not trading approval. Multiple testing, survivorship bias, execution assumptions, and incomplete point-in-time universes can invalidate apparent performance.</div>
<h2>Run metadata</h2>
<pre>{json.dumps(dict(run_metadata), indent=2, default=str)}</pre>
<h2>Top results</h2>
{table_html(top_results, 100)}
<h2>All results</h2>
<p>Total rows: {len(all_results)}</p>
{table_html(all_results, 300)}
<h2>Command arguments</h2>
<pre>{json.dumps(vars(args), indent=2, default=str)}</pre>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def rerun_result_for_details(
    result_id: str,
    scenario: str,
    variant_map: Mapping[str, StrategyVariant],
    combo_map: Mapping[str, ComboVariant],
    ctx: RunContext,
    registry: Mapping[str, StrategySpec],
    cost_models: Mapping[str, CostModel],
) -> BacktestOutput | None:
    if scenario not in cost_models:
        return None
    model = cost_models[scenario]
    if result_id in variant_map:
        return run_variant_backtest(
            variant_map[result_id], ctx.data, registry, ctx.args.capital, model, scenario,
            ctx.args.whole_shares, not ctx.args.target_first_intrabar, max(ctx.global_trials, 1)
        )
    if result_id in combo_map:
        return run_combo_backtest(
            combo_map[result_id], ctx.data, registry, ctx.args.capital, model, scenario,
            ctx.args.whole_shares, not ctx.args.target_first_intrabar, max(ctx.global_trials, 1)
        )
    return None


def export_top_details(
    top: pd.DataFrame,
    ctx: RunContext,
    registry: Mapping[str, StrategySpec],
    variant_map: Mapping[str, StrategyVariant],
    combo_map: Mapping[str, ComboVariant],
    cost_models: Mapping[str, CostModel],
) -> None:
    detail_dir = ctx.output_dir / "top_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in top.head(ctx.args.save_top_details).reset_index(drop=True).iterrows():
        result_id = str(row["result_id"])
        scenario = str(row["cost_scenario"])
        ctx.logger.info("Re-running top detail %d/%d: %s", rank + 1, min(len(top), ctx.args.save_top_details), row.get("label", result_id))
        output = rerun_result_for_details(result_id, scenario, variant_map, combo_map, ctx, registry, cost_models)
        if output is None:
            continue
        stem = f"{rank + 1:03d}_{safe_name(result_id)}_{scenario}"
        pd.DataFrame([t.as_dict() for t in output.trades]).to_csv(detail_dir / f"{stem}_trades.csv", index=False)
        output.equity.rename("equity").to_csv(detail_dir / f"{stem}_equity.csv", index_label="Date")
        detail = {
            "result_id": output.result_id,
            "label": output.label,
            "type": output.result_type,
            "strategies": output.strategy_names,
            "params": output.params,
            "cost_scenario": output.cost_scenario,
            "metrics": output.metrics,
            "monte_carlo": monte_carlo_trade_bootstrap(
                output.trades,
                ctx.args.mc_paths,
                ctx.args.seed + rank,
                ctx.args.mc_block_length,
            ),
        }
        (detail_dir / f"{stem}_summary.json").write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")


def write_final_outputs(
    ctx: RunContext,
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, StrategySpec],
    variant_map: Mapping[str, StrategyVariant],
    combo_map: Mapping[str, ComboVariant],
    cost_models: Mapping[str, CostModel],
) -> None:
    frame = result_dataframe(rows)
    if frame.empty:
        ctx.logger.warning("No result rows to write")
        return
    if "sample_valid" not in frame.columns:
        frame["sample_valid"] = False
    frame = frame.sort_values(["sample_valid", "robust_score", "profit_factor"], ascending=[False, False, False], na_position="last")
    frame.to_csv(ctx.output_dir / "all_results.csv", index=False)
    singles = frame[frame["result_type"] == "single"]
    combos = frame[frame["result_type"].astype(str).str.startswith("combo_")]
    singles.to_csv(ctx.output_dir / "single_results.csv", index=False)
    combos.to_csv(ctx.output_dir / "combo_results.csv", index=False)

    normal = frame[frame["cost_scenario"] == "normal"].copy()
    top = normal.sort_values(["sample_valid", "robust_score", "profit_factor"], ascending=[False, False, False]).head(ctx.args.top_results)
    top.to_csv(ctx.output_dir / "top_results.csv", index=False)

    run_metadata = {
        "version": VERSION,
        "run_id": ctx.run_id,
        "created_at_utc": utc_now(),
        "symbols": sorted(ctx.data),
        "bars": {symbol: len(df) for symbol, df in ctx.data.items()},
        "results": len(frame),
        "single_rows": len(singles),
        "combo_rows": len(combos),
        "global_trials": int(ctx.global_trials),
        "checkpoint": str(ctx.checkpoint_path),
        "data_hashes": {
            symbol: stable_hash({
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "rows": len(df),
                "close_head": df["Close"].head(10).round(8).tolist(),
                "close_tail": df["Close"].tail(10).round(8).tolist(),
            }, 64)
            for symbol, df in ctx.data.items()
        },
    }
    (ctx.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")
    (ctx.output_dir / "run_config.json").write_text(json.dumps(vars(ctx.args), indent=2, default=str), encoding="utf-8")
    write_html_report(ctx.output_dir / "report.html", ctx.args, frame, top, run_metadata)
    export_top_details(top, ctx, registry, variant_map, combo_map, cost_models)
    ctx.logger.info("Wrote final outputs to %s", ctx.output_dir)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def selected_registry(args: argparse.Namespace) -> dict[str, StrategySpec]:
    full = strategy_registry()
    if not args.strategies or args.strategies.lower() == "all":
        return full
    selected = {name.strip().upper() for name in args.strategies.split(",") if name.strip()}
    unknown = selected - set(full)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    return {name: full[name] for name in full if name in selected}


def resolve_crypto_fee_settings(args: argparse.Namespace) -> None:
    presets = {"maker": 15.0, "taker": 25.0}
    if args.fee_mode == "custom":
        if args.commission_bps is None:
            raise ValueError("--fee-mode custom requires --commission-bps")
    elif args.commission_bps is None:
        args.commission_bps = presets[args.fee_mode]
    # Explicit --commission-bps always wins, but keep the selected mode in metadata.
    args.commission_bps = float(args.commission_bps)


def make_cost_models(args: argparse.Namespace) -> dict[str, CostModel]:
    normal = CostModel(args.commission_bps, args.slippage_bps, args.fixed_fee)
    models = {"normal": normal}
    if not args.no_stress_costs:
        models["double"] = CostModel(
            args.commission_bps * args.stress_multiplier,
            args.slippage_bps * args.stress_multiplier,
            args.fixed_fee * args.stress_multiplier,
        )
    return models


def estimate_jobs(args: argparse.Namespace, registry: Mapping[str, StrategySpec]) -> dict[str, Any]:
    variants: list[StrategyVariant] = []
    for spec in registry.values():
        variants.extend(generate_variants(spec, args.profile, args.max_variants_per_strategy, args.seed))
    scenarios = 1 if args.no_stress_costs else 2
    shortlist_estimate = min(args.shortlist, sum(1 for v in variants if v.exit_config.name == "native"))
    sizes = parse_csv_list(args.combo_sizes, int)
    modes = parse_csv_list(args.combo_modes)
    rough_combo = sum(math.comb(shortlist_estimate, size) for size in sizes if shortlist_estimate >= size) * len(modes)
    capped_combo = rough_combo if args.max_combos <= 0 else min(rough_combo, args.max_combos)
    return {
        "strategies": len(registry),
        "single_variants": len(variants),
        "single_jobs": len(variants) * scenarios,
        "shortlist": shortlist_estimate,
        "rough_combo_candidates_before_family_filter": rough_combo,
        "combo_jobs_after_cap": capped_combo * scenarios,
        "total_jobs_estimate": len(variants) * scenarios + capped_combo * scenarios,
    }


def run_command(args: argparse.Namespace) -> int:
    resolve_crypto_fee_settings(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not args.resume and args.clean_output:
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, args.verbose)
    logger.info("Strategy Combo Research Lab v%s", VERSION)
    logger.info("This process is research-only and has no order execution code")
    logger.info("Market mode: crypto spot | source=%s | interval=%s | fee_mode=%s | commission=%.2f bps/side | slippage=%.2f bps/side", args.download_source, args.interval, args.fee_mode, args.commission_bps, args.slippage_bps)

    registry = selected_registry(args)
    estimates = estimate_jobs(args, registry)
    logger.info("Planned search: %s", json.dumps(estimates, sort_keys=True))
    if args.dry_run:
        print(json.dumps(estimates, indent=2))
        return 0

    data = prepare_universe(args, logger)
    run_payload = {
        "version": VERSION,
        "symbols": sorted(data),
        "profile": args.profile,
        "strategies": sorted(registry),
        "seed": args.seed,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "fee_mode": args.fee_mode,
        "interval": args.interval,
        "download_source": args.download_source,
    }
    run_id = stable_hash(run_payload, 24)
    checkpoint = output_dir / "results_checkpoint.jsonl"
    completed = load_checkpoint(checkpoint) if args.resume else {}
    if completed:
        logger.info("Resume enabled: loaded %d completed job rows", len(completed))

    ctx = RunContext(args, output_dir, logger, data, run_id, checkpoint, completed)
    cost_models = make_cost_models(args)
    variants: list[StrategyVariant] = []
    for spec in registry.values():
        generated = generate_variants(spec, args.profile, args.max_variants_per_strategy, args.seed)
        variants.extend(generated)
        logger.info("Generated %-35s %5d variants", spec.name, len(generated))

    all_rows: list[dict[str, Any]] = []
    variant_map: dict[str, StrategyVariant] = {}
    combo_map: dict[str, ComboVariant] = {}
    try:
        single_rows, variant_map = run_single_phase(ctx, variants, registry, cost_models)
        all_rows.extend(single_rows)

        shortlist = shortlist_variants(
            single_rows,
            variant_map,
            "normal",
            args.shortlist,
            args.top_per_strategy,
            args.min_trades,
        )
        logger.info("Shortlisted %d native-exit variants for combination testing", len(shortlist))
        for i, variant in enumerate(shortlist[:20], start=1):
            logger.info("Shortlist %02d: %s", i, variant.label)

        sizes = [size for size in parse_csv_list(args.combo_sizes, int) if size in {2, 3, 4}]
        modes = [mode for mode in parse_csv_list(args.combo_modes) if mode in {"all", "majority", "any"}]
        if not sizes or not modes:
            logger.warning("No valid combo sizes or modes, skipping combinations")
            combos: list[ComboVariant] = []
        else:
            estimated = estimate_combo_count(shortlist, sizes, modes, not args.allow_same_family_combos)
            logger.info("Eligible combinations before cap: %d", estimated)
            combos = generate_combos(
                shortlist,
                sizes,
                modes,
                args.combo_exit_vote,
                not args.allow_same_family_combos,
                args.max_combos,
                args.seed,
                args.combo_max_hold,
                args.combo_stop_atr,
                args.combo_target_atr,
            )
            logger.info("Combination jobs selected: %d unique combinations", len(combos))
        if combos:
            combo_rows, combo_map = run_combo_phase(ctx, combos, registry, cost_models)
            all_rows.extend(combo_rows)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Completed rows remain in %s and can be resumed with --resume", checkpoint)
    finally:
        # Include all checkpoint rows so partial and resumed runs still export cleanly.
        merged_rows = list(load_checkpoint(checkpoint).values())
        if merged_rows:
            write_final_outputs(ctx, merged_rows, registry, variant_map, combo_map, cost_models)

    logger.info("Run complete. Open %s", output_dir / "report.html")
    return 0


def list_strategies_command() -> int:
    registry = strategy_registry()
    print("IMPLEMENTED STRATEGIES")
    print("=" * 80)
    for spec in registry.values():
        combos = len(all_param_combinations(spec))
        print(f"{spec.name:35s} | {spec.family:28s} | raw parameter combinations: {combos}")
        print(f"  {spec.description}")
    print("\nBLOCKED OR STANDALONE-ONLY IDEAS")
    print("=" * 80)
    for name, reason in BLOCKED_STRATEGIES.items():
        print(f"{name:35s} | {reason}")
    return 0


def synthetic_data(seed: int = 7, bars: int = 900) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=bars)
    result: dict[str, pd.DataFrame] = {}
    for j, symbol in enumerate(["TESTA", "TESTB", "TESTC"]):
        drift = 0.0003 + j * 0.00005
        shocks = rng.normal(drift, 0.015 + j * 0.002, bars)
        close = 100.0 * np.exp(np.cumsum(shocks))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, bars))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.015, bars))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.015, bars))
        volume = rng.integers(100_000, 2_000_000, bars)
        frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)
        result[symbol] = normalize_ohlcv(frame, symbol)
    return result


def self_test_command() -> int:
    registry = strategy_registry()
    data = synthetic_data()
    cost = CostModel(10.0, 5.0, 0.0)
    tests = [
        StrategyVariant("RSI_ADX", registry["RSI_ADX"].family, dict(registry["RSI_ADX"].default_params), ExitConfig("native"), registry["RSI_ADX"].description),
        StrategyVariant("MA_CROSSOVER", registry["MA_CROSSOVER"].family, dict(registry["MA_CROSSOVER"].default_params), ExitConfig("native"), registry["MA_CROSSOVER"].description),
        StrategyVariant("SEVEN_DAY_PULLBACK", registry["SEVEN_DAY_PULLBACK"].family, dict(registry["SEVEN_DAY_PULLBACK"].default_params), ExitConfig("time_10", max_hold_bars=10), registry["SEVEN_DAY_PULLBACK"].description),
    ]
    outputs = [run_variant_backtest(v, data, registry, 100_000.0, cost, "normal", False, True, len(tests)) for v in tests]
    assert all(not out.equity.empty for out in outputs)
    assert all(np.isfinite(out.equity.to_numpy()).all() for out in outputs)
    native = tuple(StrategyVariant(v.strategy, v.family, v.params, ExitConfig("native"), v.description) for v in tests[:2])
    combo = ComboVariant(native, "majority", "any", max_hold_bars=30)
    combo_output = run_combo_backtest(combo, data, registry, 100_000.0, cost, "normal", False, True, 4)
    assert not combo_output.equity.empty
    assert combo_output.metrics["ending_equity"] > 0
    print("SELF_TEST_PASS")
    print(json.dumps({
        "single_metrics": [o.metrics for o in outputs],
        "combo_metrics": combo_output.metrics,
    }, indent=2, default=str))
    return 0


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", help="Directory containing one CSV/Parquet per crypto market")
    data.add_argument("--data-file", help="Long CSV/Parquet with optional Symbol/Market column")
    data.add_argument("--download", action="store_true", help="Download public OHLCV data")
    data.add_argument("--download-source", choices=["bitvavo", "yfinance"], default="bitvavo", help="Public data source; crypto default is Bitvavo")
    data.add_argument("--interval", choices=sorted(BITVAVO_INTERVALS_MS), default="1d", help="Bitvavo candle interval")
    data.add_argument("--symbols", default="", help="Comma-separated symbols")
    data.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    data.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    data.add_argument("--min-bars", type=int, default=300, help="Minimum candles per symbol")

    search = parser.add_argument_group("search")
    search.add_argument("--strategies", default="all", help="Comma-separated strategy names or all")
    search.add_argument("--profile", choices=["quick", "standard", "deep"], default="standard")
    search.add_argument("--max-variants-per-strategy", type=int, default=40, help="Deterministic cap after parameter and exit expansion; 0 means unlimited")
    search.add_argument("--shortlist", type=int, default=24, help="Maximum native-exit variants entering combination stage")
    search.add_argument("--top-per-strategy", type=int, default=1, help="Maximum shortlisted variants from one strategy")
    search.add_argument("--combo-sizes", default="2,3,4")
    search.add_argument("--combo-modes", default="all,majority", help="all,majority,any")
    search.add_argument("--combo-exit-vote", choices=["any", "majority", "all"], default="any")
    search.add_argument("--allow-same-family-combos", action="store_true", help="Allow redundant same-family signal combinations")
    search.add_argument("--max-combos", type=int, default=10000, help="Deterministic combo cap; 0 means exhaustive")
    search.add_argument("--combo-max-hold", type=int, default=None)
    search.add_argument("--combo-stop-atr", type=float, default=None)
    search.add_argument("--combo-target-atr", type=float, default=None)
    search.add_argument("--seed", type=int, default=4638)
    search.add_argument("--min-trades", type=int, default=100, help="Used by robust ranking sample penalty")

    costs = parser.add_argument_group("capital and costs")
    costs.add_argument("--capital", type=float, default=100_000.0)
    costs.add_argument("--fee-mode", choices=["maker", "taker", "custom"], default="taker", help="Bitvavo commission preset; explicit --commission-bps overrides the preset")
    costs.add_argument("--commission-bps", type=float, default=None, help="Commission per side in basis points; required for custom mode")
    costs.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage per side in basis points")
    costs.add_argument("--fixed-fee", type=float, default=0.0, help="Minimum fixed fee per side")
    costs.add_argument("--stress-multiplier", type=float, default=2.0)
    costs.add_argument("--no-stress-costs", action="store_true")
    costs.add_argument("--whole-shares", action="store_true", help="Disallow fractional units; normally leave off for crypto")
    costs.add_argument("--target-first-intrabar", action="store_true", help="Optimistic when stop and target are hit in same bar; default is stop first")

    output = parser.add_argument_group("output and diagnostics")
    output.add_argument("--output-dir", default="output/strategy_combo_research_lab")
    output.add_argument("--resume", action="store_true")
    output.add_argument("--clean-output", action="store_true", help="Delete output directory before non-resume run")
    output.add_argument("--log-every", type=int, default=1, help="Terminal progress frequency in jobs")
    output.add_argument("--top-results", type=int, default=100)
    output.add_argument("--save-top-details", type=int, default=20)
    output.add_argument("--mc-paths", type=int, default=5000, help="Monte Carlo paths for top-detail results")
    output.add_argument("--mc-block-length", type=int, default=5)
    output.add_argument("--verbose", action="store_true")
    output.add_argument("--dry-run", action="store_true", help="Estimate jobs without loading data or running tests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-file crypto spot strategy parameter, exit, and combination backtest lab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_run_arguments(run)
    sub.add_parser("list-strategies")
    sub.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "list-strategies":
        return list_strategies_command()
    if args.command == "self-test":
        return self_test_command()
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command with --resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)
