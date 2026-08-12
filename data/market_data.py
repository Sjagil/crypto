"""Canonical OHLCV normalization, validation, quality checks and storage."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config.settings import TIMEFRAME_SECONDS, normalize_timeframe
from core.contracts import DataValidationError, normalize_market
from utils.common import atomic_write_json, sha256_file, utc_iso

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
TIMESTAMP_ALIASES = ("timestamp", "datetime", "date", "time", "open_time")
COLUMN_ALIASES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str
    timeframe: str
    rows: int = Field(ge=0)
    start: datetime | None
    end: datetime | None
    expected_rows: int = Field(ge=0)
    missing_rows: int = Field(ge=0)
    missing_fraction: float = Field(ge=0.0, le=1.0)
    largest_gap_bars: int = Field(ge=0)
    duplicate_timestamps: int = Field(ge=0)
    stale: bool
    age_seconds: float | None = Field(default=None, ge=0.0)
    valid: bool
    reasons: tuple[str, ...] = ()


class DataManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 2
    market: str
    base_asset: str
    quote_asset: str
    exchange: str
    provider: str
    timeframe: str
    rows: int
    start: datetime
    end: datetime
    requested_start: datetime
    requested_end: datetime
    actual_first_timestamp: datetime
    actual_last_timestamp: datetime
    raw_calendar_days: float = Field(ge=0.0)
    usable_calendar_days: float = Field(ge=0.0)
    raw_bar_count: int = Field(ge=0)
    usable_bar_count: int = Field(ge=0)
    expected_bar_count: int = Field(ge=0)
    missing_bar_count: int = Field(ge=0)
    missing_bar_ratio: float = Field(ge=0.0, le=1.0)
    duplicate_count: int = Field(ge=0)
    invalid_bar_count: int = Field(ge=0)
    stale_bar_count: int = Field(ge=0)
    largest_gap: str | None
    listing_date_if_known: datetime | None
    source_segments: tuple[dict[str, Any], ...]
    dataset_hash: str
    generated_at: datetime
    seven_year_eligible: bool
    history_coverage_ratio: float = Field(ge=0.0)
    rejection_reason: str | None
    columns: tuple[str, ...]
    data_file: str
    sha256: str
    created_at: datetime
    quality: DataQualityReport


def timeframe_delta(timeframe: str) -> timedelta:
    try:
        normalized = normalize_timeframe(timeframe)
    except ValueError as exc:
        raise DataValidationError(f"unsupported timeframe: {timeframe}") from exc
    try:
        return timedelta(seconds=TIMEFRAME_SECONDS[normalized])
    except KeyError as exc:
        raise DataValidationError(f"unsupported timeframe: {timeframe}") from exc


def candle_close_index(
    index: pd.DatetimeIndex,
    timeframe: str,
) -> pd.DatetimeIndex:
    """Return the first instant at which each candle is fully knowable."""

    normalized = normalize_timeframe(timeframe)
    selected = pd.DatetimeIndex(index)
    if normalized == "1mo":
        return pd.DatetimeIndex(
            selected + pd.offsets.MonthBegin(1)
        )
    return pd.DatetimeIndex(
        selected
        + pd.Timedelta(timeframe_delta(normalized))
    )


def candle_close_timestamp(
    timestamp: datetime | pd.Timestamp,
    timeframe: str,
) -> pd.Timestamp:
    selected = pd.DatetimeIndex(
        [pd.Timestamp(timestamp)]
    )
    return pd.Timestamp(
        candle_close_index(selected, timeframe)[0]
    )


def _timestamp_series(data: pd.DataFrame) -> pd.Series:
    for name in TIMESTAMP_ALIASES:
        if name in data.columns:
            return data[name]
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(data.index, index=data.index)
    raise DataValidationError("OHLCV data requires a timestamp column or DatetimeIndex")


def normalize_ohlcv(
    data: pd.DataFrame,
    *,
    market: str | None = None,
    keep_extra_columns: bool = False,
) -> pd.DataFrame:
    """Normalize provider data without hiding duplicates or invalid values."""

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise DataValidationError("OHLCV data must be a non-empty DataFrame")
    frame = data.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns=COLUMN_ALIASES)
    raw_timestamp = _timestamp_series(frame)
    try:
        timestamps = pd.to_datetime(raw_timestamp, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("timestamps are not parseable as UTC") from exc
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    frame = frame.drop(columns=[name for name in TIMESTAMP_ALIASES if name in frame], errors="ignore")

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"missing OHLCV columns: {missing}")
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.index.has_duplicates:
        duplicates = int(frame.index.duplicated(keep=False).sum())
        raise DataValidationError(f"duplicate timestamps detected: {duplicates}")
    frame = frame.sort_index()
    selected = list(OHLCV_COLUMNS)
    if keep_extra_columns:
        selected.extend(column for column in frame.columns if column not in selected)
    result = frame[selected]
    if market is not None:
        result.attrs["market"] = normalize_market(market)
    return result


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    timeframe: str | None = None,
    now: datetime | None = None,
    closed_candles_only: bool = True,
    allow_missing_bars: bool = True,
    close_grace_seconds: float = 0.0,
) -> pd.DataFrame:
    """Fail closed on malformed, non-causal or internally inconsistent candles."""

    frame = normalize_ohlcv(
        data,
        market=data.attrs.get("market"),
        keep_extra_columns=True,
    )
    values = frame.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("OHLCV contains NaN or infinite values")
    if (frame.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        raise DataValidationError("OHLC prices must be strictly positive")
    if (frame["volume"] < 0).any():
        raise DataValidationError("volume cannot be negative")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("high is below another candle price")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("low is above another candle price")
    if not frame.index.is_monotonic_increasing:
        raise DataValidationError("timestamps must be monotonically increasing")

    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current_timestamp = pd.Timestamp(current.astimezone(UTC))
    if frame.index[-1] > current_timestamp:
        raise DataValidationError("OHLCV contains future timestamps")
    if close_grace_seconds < 0:
        raise ValueError("close_grace_seconds cannot be negative")
    if timeframe is not None:
        interval = pd.Timedelta(timeframe_delta(timeframe))
        grace = pd.to_timedelta(float(close_grace_seconds), unit="s")
        if (
            closed_candles_only
            and candle_close_timestamp(
                frame.index[-1],
                timeframe,
            )
            + grace
            > current_timestamp
        ):
            raise DataValidationError("OHLCV contains an open candle")
        deltas = frame.index.to_series().diff().dropna()
        if not allow_missing_bars and not deltas.empty:
            if normalize_timeframe(timeframe) == "1mo":
                expected = pd.date_range(
                    frame.index[0],
                    frame.index[-1],
                    freq="MS",
                )
                if not expected.equals(frame.index):
                    raise DataValidationError(
                        "OHLCV has missing or irregular bars"
                    )
            elif (deltas != interval).any():
                raise DataValidationError(
                    "OHLCV has missing or irregular bars"
                )
    frame.attrs.update(data.attrs)
    return frame


def drop_open_candles(
    data: pd.DataFrame,
    *,
    timeframe: str,
    now: datetime | None = None,
    close_grace_seconds: float = 0.0,
) -> pd.DataFrame:
    frame = normalize_ohlcv(
        data,
        market=data.attrs.get("market"),
        keep_extra_columns=True,
    )
    if close_grace_seconds < 0:
        raise ValueError("close_grace_seconds cannot be negative")
    current = pd.Timestamp((now or datetime.now(UTC)).astimezone(UTC))
    grace = pd.to_timedelta(
        float(close_grace_seconds),
        unit="s",
    )
    closes = candle_close_index(
        frame.index,
        timeframe,
    )
    result = frame.loc[
        closes + grace <= current
    ].copy()
    if result.empty:
        raise DataValidationError("no closed candles remain after filtering")
    result.attrs.update(frame.attrs)
    return result


def quality_report(
    data: pd.DataFrame,
    *,
    market: str,
    timeframe: str,
    maximum_staleness: timedelta,
    now: datetime | None = None,
    maximum_missing_fraction: float = 0.05,
) -> DataQualityReport:
    normalized_market = normalize_market(market)
    try:
        frame = normalize_ohlcv(data, market=normalized_market)
        validate_ohlcv(
            frame,
            timeframe=timeframe,
            now=now,
            closed_candles_only=False,
        )
    except DataValidationError as exc:
        return DataQualityReport(
            market=normalized_market,
            timeframe=timeframe,
            rows=len(data) if isinstance(data, pd.DataFrame) else 0,
            start=None,
            end=None,
            expected_rows=0,
            missing_rows=0,
            missing_fraction=0.0,
            largest_gap_bars=0,
            duplicate_timestamps=(
                int(data.index.duplicated(keep=False).sum())
                if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex)
                else 0
            ),
            stale=True,
            valid=False,
            reasons=(exc.code, str(exc)),
        )

    normalized_timeframe = normalize_timeframe(timeframe)
    interval = pd.Timedelta(
        timeframe_delta(normalized_timeframe)
    )
    expected = pd.date_range(
        frame.index[0],
        frame.index[-1],
        freq=(
            "MS"
            if normalized_timeframe == "1mo"
            else interval
        ),
    )
    missing = expected.difference(frame.index)
    expected_rows = len(expected)
    missing_fraction = len(missing) / expected_rows if expected_rows else 0.0
    deltas = frame.index.to_series().diff().dropna()
    if (
        normalized_timeframe == "1mo"
        and len(frame.index) > 1
    ):
        month_ordinals = (
            frame.index.year * 12
            + frame.index.month
        )
        largest_gap = max(
            0,
            int(np.diff(month_ordinals).max()) - 1,
        )
    else:
        largest_gap = (
            max(
                0,
                int(math.floor(deltas.max() / interval)) - 1,
            )
            if not deltas.empty
            else 0
        )
    current = (now or datetime.now(UTC)).astimezone(UTC)
    last_available_at = candle_close_timestamp(
        frame.index[-1],
        normalized_timeframe,
    ).to_pydatetime()
    age = max(0.0, (current - last_available_at).total_seconds())
    stale = age > maximum_staleness.total_seconds()
    reasons: list[str] = []
    if stale:
        reasons.append("STALE_DATA")
    if missing_fraction > maximum_missing_fraction:
        reasons.append("EXCESSIVE_MISSING_BARS")
    return DataQualityReport(
        market=normalized_market,
        timeframe=timeframe,
        rows=len(frame),
        start=frame.index[0].to_pydatetime(),
        end=frame.index[-1].to_pydatetime(),
        expected_rows=expected_rows,
        missing_rows=len(missing),
        missing_fraction=missing_fraction,
        largest_gap_bars=largest_gap,
        duplicate_timestamps=0,
        stale=stale,
        age_seconds=age,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def resample_ohlcv(
    data: pd.DataFrame,
    *,
    source_timeframe: str,
    target_timeframe: str,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    normalized_target = normalize_timeframe(target_timeframe)
    source_seconds = int(timeframe_delta(source_timeframe).total_seconds())
    target_seconds = int(timeframe_delta(normalized_target).total_seconds())
    if target_seconds <= source_seconds or target_seconds % source_seconds:
        raise DataValidationError("target timeframe must be a larger integer multiple")
    frame = validate_ohlcv(
        data,
        timeframe=source_timeframe,
        closed_candles_only=False,
    )
    # Fixed timedeltas are epoch-anchored; a seven-day timedelta therefore
    # starts on Thursday (1970-01-01).  Weekly trading candles use explicit
    # Monday 00:00 UTC boundaries throughout the repository.
    rule: str | pd.Timedelta = (
        "W-MON"
        if normalized_target == "1W"
        else pd.Timedelta(target_seconds, unit="s")
    )
    counts = frame["close"].resample(rule, label="left", closed="left").count()
    result = frame.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    if drop_incomplete:
        result = result.loc[counts == target_seconds // source_seconds]
    result = result.dropna(subset=list(OHLCV_COLUMNS))
    if result.empty:
        raise DataValidationError("resampling produced no complete candles")
    result.attrs.update(frame.attrs)
    return validate_ohlcv(
        result,
        timeframe=normalized_target,
        closed_candles_only=False,
    )


def _write_frame_atomic(
    data: pd.DataFrame,
    path: Path,
    *,
    file_format: Literal["csv", "parquet"],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=f".{file_format}",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        if file_format == "csv":
            data.to_csv(temporary, index=True, index_label="timestamp")
        else:
            data.to_parquet(temporary, index=True, engine="pyarrow")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_ohlcv(
    data: pd.DataFrame,
    path: Path | str,
    *,
    market: str,
    timeframe: str,
    maximum_staleness: timedelta = timedelta(days=3650),
    now: datetime | None = None,
    provider: str | None = None,
    exchange: str | None = None,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
    listing_date_if_known: datetime | None = None,
    source_segments: Sequence[dict[str, Any]] | None = None,
) -> tuple[Path, DataManifest]:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("OHLCV storage supports .csv and .parquet")
    frame = validate_ohlcv(
        data,
        timeframe=timeframe,
        now=now,
        closed_candles_only=True,
    )
    report = quality_report(
        frame,
        market=market,
        timeframe=timeframe,
        maximum_staleness=maximum_staleness,
        now=now,
    )
    _write_frame_atomic(
        frame,
        target,
        file_format="parquet" if suffix == ".parquet" else "csv",
    )
    normalized_market = normalize_market(market)
    base_asset, quote_asset = normalized_market.split("-", 1)
    selected_provider = str(
        provider
        or frame.attrs.get("provider")
        or "UNKNOWN_EXISTING_LOCAL"
    )
    selected_exchange = str(
        exchange
        or frame.attrs.get("exchange")
        or selected_provider
    ).upper()
    first = frame.index[0].to_pydatetime()
    last = frame.index[-1].to_pydatetime()
    calendar_days = max(0.0, (last - first).total_seconds() / 86_400)
    exact_required_start = (
        pd.Timestamp(last) - pd.DateOffset(years=7)
    ).to_pydatetime()
    seven_year_eligible = bool(
        first <= exact_required_start
        and report.valid
    )
    selected_segments = tuple(source_segments or ())
    if not selected_segments:
        selected_segments = (
            {
                "provider": selected_provider,
                "exchange": selected_exchange,
                "market_identity": normalized_market,
                "start": utc_iso(first),
                "end": utc_iso(last),
                "classification": "PROVIDER_NATIVE_OR_CANONICAL_RESAMPLE",
            },
        )
    selected_requested_start = requested_start or first
    selected_requested_end = requested_end or (now or datetime.now(UTC))
    data_hash = sha256_file(target)
    manifest = DataManifest(
        market=normalized_market,
        base_asset=base_asset,
        quote_asset=quote_asset,
        exchange=selected_exchange,
        provider=selected_provider,
        timeframe=timeframe,
        rows=len(frame),
        start=first,
        end=last,
        requested_start=selected_requested_start,
        requested_end=selected_requested_end,
        actual_first_timestamp=first,
        actual_last_timestamp=last,
        raw_calendar_days=calendar_days,
        usable_calendar_days=calendar_days,
        raw_bar_count=len(frame),
        usable_bar_count=len(frame),
        expected_bar_count=report.expected_rows,
        missing_bar_count=report.missing_rows,
        missing_bar_ratio=report.missing_fraction,
        duplicate_count=report.duplicate_timestamps,
        invalid_bar_count=0,
        stale_bar_count=0,
        largest_gap=(
            str(timeframe_delta(timeframe) * (report.largest_gap_bars + 1))
            if report.largest_gap_bars
            else None
        ),
        listing_date_if_known=listing_date_if_known or first,
        source_segments=selected_segments,
        dataset_hash=data_hash,
        generated_at=datetime.now(UTC),
        seven_year_eligible=seven_year_eligible,
        history_coverage_ratio=calendar_days / (7 * 365.2425),
        rejection_reason=(
            None
            if seven_year_eligible
            else (
                "DATA_QUALITY_FAILED"
                if not report.valid
                else "INSUFFICIENT_MARKET_HISTORY"
            )
        ),
        columns=tuple(frame.columns),
        data_file=target.name,
        sha256=data_hash,
        created_at=datetime.now(UTC),
        quality=report,
    )
    atomic_write_json(
        target.with_suffix(f"{target.suffix}.manifest.json"),
        manifest.model_dump(mode="json"),
    )
    return target, manifest


def load_ohlcv(
    path: Path | str,
    *,
    market: str | None = None,
    timeframe: str | None = None,
    validate: bool = True,
    closed_candles_only: bool = False,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source, engine="pyarrow")
    else:
        raise ValueError("OHLCV loader supports .csv and .parquet")
    result = normalize_ohlcv(frame, market=market, keep_extra_columns=True)
    if validate:
        result = validate_ohlcv(
            result,
            timeframe=timeframe,
            closed_candles_only=closed_candles_only,
        )
    return result


def inspect_file(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    manifest_path = source.with_suffix(f"{source.suffix}.manifest.json")
    frame = load_ohlcv(source, validate=False)
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "rows": len(frame),
        "start": utc_iso(frame.index[0].to_pydatetime()),
        "end": utc_iso(frame.index[-1].to_pydatetime()),
        "columns": list(frame.columns),
        "manifest": (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        ),
    }


__all__ = [
    "COLUMN_ALIASES",
    "DataManifest",
    "DataQualityReport",
    "OHLCV_COLUMNS",
    "candle_close_index",
    "candle_close_timestamp",
    "drop_open_candles",
    "inspect_file",
    "load_ohlcv",
    "normalize_ohlcv",
    "quality_report",
    "resample_ohlcv",
    "save_ohlcv",
    "timeframe_delta",
    "validate_ohlcv",
]
