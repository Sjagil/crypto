"""Calendar-based seven-year research eligibility and repository audit.

This module does not backtest, download or execute orders.  It provides one
shared definition of historical coverage for the existing downloader,
backtester, combinatorial lab, promotion gates and reporting layer.
"""

from __future__ import annotations

import html
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config.settings import normalize_timeframe
from data.market_data import OHLCV_COLUMNS, timeframe_delta
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    stable_hash,
    utc_iso,
)

REQUIRED_CALENDAR_YEARS = 7
REQUIRED_SEVEN_YEAR_DAYS = 7 * 365.2425
DEFAULT_MAXIMUM_MISSING_RATIO = 0.05
MINIMUM_TRADES_BY_TIMEFRAME: dict[str, int] = {
    "1d": 35,
    "4h": 70,
    "1h": 120,
    "15m": 250,
    "5m": 400,
}
FINAL_STRATEGY_STATUSES = {
    "PENDING_DATA",
    "DOWNLOADING",
    "DATA_READY",
    "INSUFFICIENT_MARKET_HISTORY",
    "DATA_QUALITY_FAILED",
    "QUEUED",
    "RUNNING",
    "BASELINE_COMPLETE",
    "INSUFFICIENT_TRADES",
    "FAILED_CAUSALITY",
    "FAILED_STRESS",
    "FAILED_WALK_FORWARD",
    "FAILED_STABILITY",
    "RESEARCH_SURVIVOR",
    "RESEARCH_REJECTED",
    "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY",
    "SEVEN_YEAR_RESEARCH_CANDIDATE",
}


class SevenYearDatasetManifest(BaseModel):
    """Machine-readable coverage proof for one market/timeframe dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "seven_year_dataset_manifest_v1"
    market: str
    base_asset: str
    quote_asset: str
    exchange: str
    provider: str
    timeframe: str
    requested_start: datetime
    requested_end: datetime
    actual_first_timestamp: datetime | None
    actual_last_timestamp: datetime | None
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
    largest_gap_bars: int = Field(ge=0)
    listing_date_if_known: datetime | None
    source_segments: tuple[dict[str, Any], ...]
    dataset_hash: str
    generated_at: datetime
    seven_year_eligible: bool
    history_coverage_ratio: float = Field(ge=0.0)
    rejection_reason: str | None
    quality_reasons: tuple[str, ...]
    closed_candles_only: bool
    warmup_bars: int = Field(ge=0)
    warmup_calendar_days: float = Field(ge=0.0)
    evaluation_start: datetime | None
    evaluation_end: datetime | None
    exact_required_evaluation_start: datetime
    source_manifest: str | None


class CommonWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "seven_year_common_window_v1"
    markets: tuple[str, ...]
    timeframes: tuple[str, ...]
    start: datetime | None
    end: datetime | None
    calendar_days: float = Field(ge=0.0)
    seven_year_eligible: bool
    rejection_reasons: tuple[str, ...]


def exact_calendar_start(end: datetime | pd.Timestamp, years: int = 7) -> datetime:
    """Return the exact calendar start, preserving UTC and leap-day semantics."""

    timestamp = pd.Timestamp(end)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return (timestamp - pd.DateOffset(years=years)).to_pydatetime()


def has_exact_calendar_years(
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    years: int = 7,
) -> bool:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp.tzinfo is None:
        start_timestamp = start_timestamp.tz_localize("UTC")
    else:
        start_timestamp = start_timestamp.tz_convert("UTC")
    if end_timestamp.tzinfo is None:
        end_timestamp = end_timestamp.tz_localize("UTC")
    else:
        end_timestamp = end_timestamp.tz_convert("UTC")
    return start_timestamp <= end_timestamp - pd.DateOffset(years=years)


def minimum_trades_for_timeframe(timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    return MINIMUM_TRADES_BY_TIMEFRAME.get(normalized, 35)


def _dataset_identity(path: Path) -> tuple[str, str]:
    match = re.fullmatch(
        r"(?P<market>[A-Za-z0-9]+-[A-Za-z0-9]+)_(?P<timeframe>[^.]+)",
        path.stem,
    )
    if match is None:
        raise ValueError(f"cannot infer market/timeframe from dataset name: {path.name}")
    return match.group("market").upper(), normalize_timeframe(match.group("timeframe"))


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], Path | None]:
    candidates = (
        path.with_suffix(f"{path.suffix}.manifest.json"),
        path.with_suffix(".manifest.json"),
        path.with_suffix(f"{path.suffix}.provenance.json"),
        path.with_suffix(".provenance.json"),
    )
    merged: dict[str, Any] = {}
    selected_path: Path | None = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = read_json(candidate)
        except (OSError, ValueError):
            continue
        if isinstance(value, Mapping):
            merged.update(dict(value))
            selected_path = candidate
    return merged, selected_path


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce"))
    else:
        timestamp_column = next(
            (
                name
                for name in ("timestamp", "datetime", "date", "time", "open_time")
                if name in frame.columns
            ),
            None,
        )
        if timestamp_column is None:
            return pd.DatetimeIndex([], tz="UTC", name="timestamp")
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
            name="timestamp",
        )
    return index


def _source_value(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _source_segments(
    source: Mapping[str, Any],
    *,
    path: Path,
    provider: str,
    exchange: str,
    first: datetime | None,
    last: datetime | None,
) -> tuple[dict[str, Any], ...]:
    explicit = _source_value(source, "source_segments", "segments")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        return tuple(dict(item) for item in explicit if isinstance(item, Mapping))
    return (
        {
            "provider": provider,
            "exchange": exchange,
            "market_identity": path.stem.rsplit("_", 1)[0].upper(),
            "start": utc_iso(first) if first else None,
            "end": utc_iso(last) if last else None,
            "classification": "EXISTING_CANONICAL_LOCAL_DATASET",
        },
    )


def audit_dataset(
    path: Path | str,
    *,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
    warmup_bars: int = 0,
    maximum_missing_ratio: float = DEFAULT_MAXIMUM_MISSING_RATIO,
    now: datetime | None = None,
) -> SevenYearDatasetManifest:
    """Audit one existing OHLCV file without inventing or filling candles."""

    source_path = Path(path)
    market, timeframe = _dataset_identity(source_path)
    base_asset, quote_asset = market.split("-", 1)
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    interval = timeframe_delta(timeframe)
    source, source_manifest_path = _load_source_manifest(source_path)
    provider_raw = _source_value(
        source,
        "provider",
        "primary_provider",
        "providers_used",
    )
    provider = (
        ",".join(str(value) for value in provider_raw)
        if isinstance(provider_raw, Sequence)
        and not isinstance(provider_raw, (str, bytes))
        else str(provider_raw or "UNKNOWN_EXISTING_LOCAL")
    )
    exchange = str(_source_value(source, "exchange", "venue") or provider).upper()
    requested_end = generated_at
    requested_start = exact_calendar_start(requested_end, minimum_years)

    try:
        if source_path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(source_path)
        elif source_path.suffix.casefold() == ".csv":
            frame = pd.read_csv(source_path)
        else:
            raise ValueError("unsupported dataset extension")
    except (OSError, ValueError, ImportError) as exc:
        return SevenYearDatasetManifest(
            market=market,
            base_asset=base_asset,
            quote_asset=quote_asset,
            exchange=exchange,
            provider=provider,
            timeframe=timeframe,
            requested_start=requested_start,
            requested_end=requested_end,
            actual_first_timestamp=None,
            actual_last_timestamp=None,
            raw_calendar_days=0.0,
            usable_calendar_days=0.0,
            raw_bar_count=0,
            usable_bar_count=0,
            expected_bar_count=0,
            missing_bar_count=0,
            missing_bar_ratio=0.0,
            duplicate_count=0,
            invalid_bar_count=0,
            stale_bar_count=0,
            largest_gap=None,
            largest_gap_bars=0,
            listing_date_if_known=None,
            source_segments=(),
            dataset_hash=sha256_file(source_path) if source_path.is_file() else "",
            generated_at=generated_at,
            seven_year_eligible=False,
            history_coverage_ratio=0.0,
            rejection_reason="DATA_QUALITY_FAILED",
            quality_reasons=(f"READ_FAILED:{type(exc).__name__}",),
            closed_candles_only=False,
            warmup_bars=warmup_bars,
            warmup_calendar_days=warmup_bars * interval.total_seconds() / 86_400,
            evaluation_start=None,
            evaluation_end=None,
            exact_required_evaluation_start=requested_start,
            source_manifest=str(source_manifest_path) if source_manifest_path else None,
        )

    raw_bar_count = len(frame)
    index = _utc_index(frame)
    valid_timestamp_mask = ~index.isna()
    duplicate_mask = pd.Series(index.duplicated(keep=False), index=frame.index)
    duplicate_count = int(duplicate_mask.sum())
    numeric = frame.reindex(columns=list(OHLCV_COLUMNS)).apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    positive_prices = (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = numeric["volume"].ge(0)
    valid_high = numeric["high"].ge(
        numeric[["open", "close", "low"]].max(axis=1)
    )
    valid_low = numeric["low"].le(
        numeric[["open", "close", "high"]].min(axis=1)
    )
    valid_ohlc = (
        pd.Series(finite, index=frame.index)
        & positive_prices
        & valid_volume
        & valid_high
        & valid_low
    )
    invalid_bar_count = int((~valid_ohlc).sum() + (~valid_timestamp_mask).sum())
    close_cutoff = pd.Timestamp(generated_at) - pd.Timedelta(interval)
    closed_mask = pd.Series(index <= close_cutoff, index=frame.index)
    stale_bar_count = int((~closed_mask).sum())
    usable_mask = (
        valid_ohlc
        & pd.Series(valid_timestamp_mask, index=frame.index)
        & ~duplicate_mask
        & closed_mask
    )
    usable = numeric.loc[usable_mask].copy()
    usable.index = pd.DatetimeIndex(index[usable_mask.to_numpy()], name="timestamp")
    usable = usable.sort_index()

    raw_valid_index = index[valid_timestamp_mask]
    first = raw_valid_index.min().to_pydatetime() if len(raw_valid_index) else None
    last = raw_valid_index.max().to_pydatetime() if len(raw_valid_index) else None
    raw_calendar_days = (
        max(0.0, (last - first).total_seconds() / 86_400)
        if first and last
        else 0.0
    )
    warmup_calendar_days = warmup_bars * interval.total_seconds() / 86_400
    evaluation = usable.iloc[warmup_bars:] if len(usable) > warmup_bars else usable.iloc[0:0]
    evaluation_start = (
        evaluation.index[0].to_pydatetime() if not evaluation.empty else None
    )
    evaluation_end = (
        evaluation.index[-1].to_pydatetime() if not evaluation.empty else None
    )
    usable_calendar_days = (
        max(0.0, (evaluation_end - evaluation_start).total_seconds() / 86_400)
        if evaluation_start and evaluation_end
        else 0.0
    )
    exact_start = (
        exact_calendar_start(evaluation_end, minimum_years)
        if evaluation_end
        else requested_start
    )

    expected_bar_count = 0
    missing_bar_count = 0
    missing_bar_ratio = 0.0
    largest_gap: str | None = None
    largest_gap_bars = 0
    if len(usable.index):
        expected_bar_count = int(
            math.floor(
                (usable.index[-1] - usable.index[0]).total_seconds()
                / interval.total_seconds()
            )
            + 1
        )
        missing_bar_count = max(0, expected_bar_count - len(usable))
        missing_bar_ratio = (
            missing_bar_count / expected_bar_count if expected_bar_count else 0.0
        )
        deltas = usable.index.to_series().diff().dropna()
        if not deltas.empty:
            maximum_delta = deltas.max()
            largest_gap = str(maximum_delta)
            largest_gap_bars = max(
                0,
                int(math.floor(maximum_delta.total_seconds() / interval.total_seconds()))
                - 1,
            )

    last_closed_at = (
        usable.index[-1].to_pydatetime() + interval if not usable.empty else None
    )
    stale_dataset = (
        last_closed_at is None
        or generated_at - last_closed_at
        > max(timedelta(days=2), interval * 2)
    )
    source_segments = _source_segments(
        source,
        path=source_path,
        provider=provider,
        exchange=exchange,
        first=first,
        last=last,
    )
    quality_reasons: list[str] = []
    if duplicate_count:
        quality_reasons.append("DUPLICATE_CANDLES")
    if invalid_bar_count:
        quality_reasons.append("INVALID_OHLC")
    if missing_bar_ratio > maximum_missing_ratio:
        quality_reasons.append("EXCESSIVE_GAPS")
    if stale_dataset:
        quality_reasons.append("STALE_DATASET")
    if quote_asset != "EUR":
        quality_reasons.append("NON_EUR_MARKET_IDENTITY")
    if provider == "UNKNOWN_EXISTING_LOCAL":
        quality_reasons.append("SOURCE_PROVENANCE_INCOMPLETE")
    if any(
        str(segment.get("market_identity") or market).upper() != market
        for segment in source_segments
    ):
        quality_reasons.append("SOURCE_MARKET_IDENTITY_MISMATCH")
    if len(source_segments) > 1:
        transitions_verified = bool(
            source.get("source_transition_validated") is True
            or all(
                str(segment.get("overlap_check") or "").upper()
                in {"PASSED", "RECONCILED", "NO_OVERLAP_EXPECTED"}
                for segment in source_segments[1:]
            )
        )
        if not transitions_verified:
            quality_reasons.append("SOURCE_TRANSITION_UNVERIFIED")

    coverage_ok = bool(
        evaluation_start
        and evaluation_end
        and has_exact_calendar_years(evaluation_start, evaluation_end, minimum_years)
    )
    hard_quality_failure = any(
        reason
        in {
            "DUPLICATE_CANDLES",
            "INVALID_OHLC",
            "EXCESSIVE_GAPS",
            "NON_EUR_MARKET_IDENTITY",
            "SOURCE_MARKET_IDENTITY_MISMATCH",
            "SOURCE_TRANSITION_UNVERIFIED",
        }
        for reason in quality_reasons
    )
    seven_year_eligible = coverage_ok and not hard_quality_failure
    if hard_quality_failure:
        rejection_reason = "DATA_QUALITY_FAILED"
    elif not coverage_ok:
        rejection_reason = "INSUFFICIENT_MARKET_HISTORY"
    elif stale_dataset:
        rejection_reason = "STALE_DATASET"
    else:
        rejection_reason = None
    listing_raw = _source_value(
        source,
        "listing_date_if_known",
        "listing_date",
        "actual_first_timestamp",
        "start",
    )
    listing_date = None
    if listing_raw is not None:
        try:
            listing_date = pd.Timestamp(listing_raw).to_pydatetime()
            if listing_date.tzinfo is None:
                listing_date = listing_date.replace(tzinfo=UTC)
            listing_date = listing_date.astimezone(UTC)
        except (TypeError, ValueError):
            listing_date = None
    if listing_date is None:
        listing_date = first

    return SevenYearDatasetManifest(
        market=market,
        base_asset=base_asset,
        quote_asset=quote_asset,
        exchange=exchange,
        provider=provider,
        timeframe=timeframe,
        requested_start=requested_start,
        requested_end=requested_end,
        actual_first_timestamp=first,
        actual_last_timestamp=last,
        raw_calendar_days=raw_calendar_days,
        usable_calendar_days=usable_calendar_days,
        raw_bar_count=raw_bar_count,
        usable_bar_count=len(evaluation),
        expected_bar_count=expected_bar_count,
        missing_bar_count=missing_bar_count,
        missing_bar_ratio=missing_bar_ratio,
        duplicate_count=duplicate_count,
        invalid_bar_count=invalid_bar_count,
        stale_bar_count=stale_bar_count,
        largest_gap=largest_gap,
        largest_gap_bars=largest_gap_bars,
        listing_date_if_known=listing_date,
        source_segments=source_segments,
        dataset_hash=sha256_file(source_path),
        generated_at=generated_at,
        seven_year_eligible=seven_year_eligible,
        history_coverage_ratio=(
            usable_calendar_days / (minimum_years * 365.2425)
            if minimum_years > 0
            else 0.0
        ),
        rejection_reason=rejection_reason,
        quality_reasons=tuple(quality_reasons),
        closed_candles_only=stale_bar_count == 0,
        warmup_bars=warmup_bars,
        warmup_calendar_days=warmup_calendar_days,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        exact_required_evaluation_start=exact_start,
        source_manifest=str(source_manifest_path) if source_manifest_path else None,
    )


def common_window(
    manifests: Sequence[SevenYearDatasetManifest],
    *,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
) -> CommonWindow:
    markets = tuple(sorted({manifest.market for manifest in manifests}))
    timeframes = tuple(sorted({manifest.timeframe for manifest in manifests}))
    reasons: list[str] = []
    starts = [
        manifest.evaluation_start
        for manifest in manifests
        if manifest.evaluation_start is not None
    ]
    ends = [
        manifest.evaluation_end
        for manifest in manifests
        if manifest.evaluation_end is not None
    ]
    if len(starts) != len(manifests) or len(ends) != len(manifests):
        reasons.append("MISSING_DATASET_BOUNDARY")
    start = max(starts) if starts else None
    end = min(ends) if ends else None
    days = (
        max(0.0, (end - start).total_seconds() / 86_400)
        if start and end and end >= start
        else 0.0
    )
    if not start or not end or not has_exact_calendar_years(start, end, minimum_years):
        reasons.append("COMMON_WINDOW_SHORTER_THAN_REQUIRED")
    if any(not manifest.seven_year_eligible for manifest in manifests):
        reasons.append("CONSTITUENT_DATASET_NOT_SEVEN_YEAR_ELIGIBLE")
    return CommonWindow(
        markets=markets,
        timeframes=timeframes,
        start=start,
        end=end,
        calendar_days=days,
        seven_year_eligible=not reasons,
        rejection_reasons=tuple(sorted(set(reasons))),
    )


def strategy_history_status(
    *,
    manifests: Sequence[SevenYearDatasetManifest],
    timeframe: str,
    trade_count: int | None,
    rerun_complete: bool = False,
    normal_economics_passed: bool | None = None,
    causality_passed: bool | None = None,
    stress_passed: bool | None = None,
    walk_forward_passed: bool | None = None,
    stability_passed: bool | None = None,
) -> tuple[str, tuple[str, ...]]:
    if not manifests:
        return "PENDING_DATA", ("NO_MATCHING_DATASET_MANIFEST",)
    if any(manifest.rejection_reason == "DATA_QUALITY_FAILED" for manifest in manifests):
        return "DATA_QUALITY_FAILED", ("CONSTITUENT_DATA_QUALITY_FAILED",)
    if any(not manifest.seven_year_eligible for manifest in manifests):
        return (
            "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY",
            tuple(
                sorted(
                    {
                        manifest.rejection_reason or "INSUFFICIENT_MARKET_HISTORY"
                        for manifest in manifests
                        if not manifest.seven_year_eligible
                    }
                )
            ),
        )
    if not rerun_complete:
        return "QUEUED", ("SEVEN_YEAR_BASELINE_RERUN_REQUIRED",)
    minimum_trades = minimum_trades_for_timeframe(timeframe)
    if trade_count is None or trade_count < minimum_trades:
        return "INSUFFICIENT_TRADES", (f"MINIMUM_TRADES_{minimum_trades}",)
    if normal_economics_passed is not True:
        return "RESEARCH_REJECTED", ("NON_POSITIVE_NORMAL_COST_ECONOMICS",)
    if causality_passed is not True:
        return "FAILED_CAUSALITY", ("CAUSALITY_NOT_PROVEN",)
    if stress_passed is not True:
        return "FAILED_STRESS", ("STRESSED_COSTS_NOT_PASSED",)
    if walk_forward_passed is not True:
        return "FAILED_WALK_FORWARD", ("WALK_FORWARD_NOT_PASSED",)
    if stability_passed is not True:
        return "FAILED_STABILITY", ("PARAMETER_STABILITY_NOT_PASSED",)
    return "SEVEN_YEAR_RESEARCH_CANDIDATE", ()


def _rolling_twelve_month_metrics(
    equity: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return exact-timeframe rolling diagnostics over approximately one year."""

    if len(equity) < 3:
        return pd.DataFrame(index=equity.index), {}
    intervals = equity.index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = float(intervals.median())
    if not math.isfinite(median_seconds) or median_seconds <= 0:
        return pd.DataFrame(index=equity.index), {}
    periods_per_year = 365.2425 * 86_400.0 / median_seconds
    window = max(2, int(round(periods_per_year)))
    returns = equity.astype(float).pct_change()
    positive = returns.clip(lower=0.0)
    negative = -returns.clip(upper=0.0)
    rolling_negative = negative.rolling(window, min_periods=window).sum()
    frame = pd.DataFrame(
        {
            "rolling_12m_return": equity / equity.shift(window) - 1.0,
            "rolling_12m_sharpe": (
                returns.rolling(window, min_periods=window).mean()
                / returns.rolling(window, min_periods=window).std(ddof=1)
                * math.sqrt(periods_per_year)
            ),
            "rolling_12m_profit_factor": (
                positive.rolling(window, min_periods=window).sum()
                / rolling_negative.replace(0.0, np.nan)
            ),
        },
        index=equity.index,
    )

    def diagnostics(column: str) -> dict[str, float | None]:
        selected = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        if selected.empty:
            return {
                "latest": None,
                "minimum": None,
                "median": None,
                "maximum": None,
                "positive_fraction": None,
            }
        return {
            "latest": float(selected.iloc[-1]),
            "minimum": float(selected.min()),
            "median": float(selected.median()),
            "maximum": float(selected.max()),
            "positive_fraction": float((selected > 0.0).mean()),
        }

    return frame, {
        "window_bars": window,
        "periods_per_year": periods_per_year,
        "return": diagnostics("rolling_12m_return"),
        "sharpe": diagnostics("rolling_12m_sharpe"),
        "period_profit_factor": diagnostics(
            "rolling_12m_profit_factor"
        ),
        "profit_factor_definition": (
            "POSITIVE_PERIOD_RETURNS_DIVIDED_BY_ABSOLUTE_NEGATIVE_PERIOD_RETURNS"
        ),
    }


def _result_summary(result: Any) -> dict[str, Any]:
    from research.trading_math import bootstrap_expectancy

    metrics = dict(result.metrics)
    equity = result.equity_curve["equity"].astype(float)
    equity_returns = equity.pct_change().dropna()
    intervals = equity.index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = float(intervals.median()) if not intervals.empty else math.nan
    periods_per_year = (
        365.2425 * 86_400.0 / median_seconds
        if math.isfinite(median_seconds) and median_seconds > 0
        else math.nan
    )
    elapsed_years = max(
        (equity.index[-1] - equity.index[0]).total_seconds()
        / (365.2425 * 86_400.0),
        1.0 / 365.2425,
    )
    total_fees = float(sum(float(order.fee_eur) for order in result.orders))
    total_slippage = float(
        sum(
            abs(float(order.fill_price) - float(order.raw_price))
            * float(order.quantity)
            for order in result.orders
        )
    )
    costs = float(metrics.get("transaction_costs_eur") or 0.0)
    gross_ending_equity_estimate = float(result.ending_equity_eur) + costs
    annual = _annual_returns(equity)
    rolling_frame, rolling_summary = _rolling_twelve_month_metrics(equity)
    r_values = np.asarray(
        [float(trade.r_multiple) for trade in result.trades],
        dtype=float,
    )
    bootstrap = None
    if r_values.size:
        block_size = min(
            max(1, int(round(r_values.size ** (1 / 3)))),
            int(r_values.size),
        )
        bootstrap = bootstrap_expectancy(
            r_values,
            bootstrap_samples=500,
            block_size=block_size,
            seed=42,
        ).to_dict()
    metrics.update(
        {
            "gross_return_estimate": (
                gross_ending_equity_estimate / result.initial_cash_eur - 1.0
            ),
            "gross_ending_equity_estimate_eur": gross_ending_equity_estimate,
            "annualized_volatility": (
                float(equity_returns.std(ddof=1) * math.sqrt(periods_per_year))
                if len(equity_returns) > 1 and math.isfinite(periods_per_year)
                else 0.0
            ),
            "trades_per_year": len(result.trades) / elapsed_years,
            "time_in_market": float(
                (
                    result.equity_curve["exposure_fraction"].astype(float)
                    > 1e-12
                ).mean()
            ),
            "total_fees_eur": total_fees,
            "total_slippage_eur": total_slippage,
            "best_year": (
                max((float(row["net_return"]) for row in annual), default=None)
            ),
            "worst_year": (
                min((float(row["net_return"]) for row in annual), default=None)
            ),
            "positive_years": sum(
                float(row["net_return"]) > 0.0 for row in annual
            ),
            "negative_years": sum(
                float(row["net_return"]) <= 0.0 for row in annual
            ),
            "bootstrap_confidence_interval_lower_r": (
                bootstrap["lower_expectancy_r"] if bootstrap else None
            ),
            "bootstrap_confidence_interval_upper_r": (
                bootstrap["upper_expectancy_r"] if bootstrap else None
            ),
            "bootstrap_probability_expectancy_positive": (
                bootstrap["probability_expectancy_positive"]
                if bootstrap
                else None
            ),
            "rolling_12m_return_latest": dict(
                rolling_summary.get("return") or {}
            ).get("latest"),
            "rolling_12m_sharpe_latest": dict(
                rolling_summary.get("sharpe") or {}
            ).get("latest"),
            "rolling_12m_period_profit_factor_latest": dict(
                rolling_summary.get("period_profit_factor") or {}
            ).get("latest"),
        }
    )
    return {
        "strategy_id": result.strategy_id,
        "initial_cash_eur": result.initial_cash_eur,
        "ending_equity_eur": result.ending_equity_eur,
        "metrics": metrics,
        "bootstrap_expectancy": bootstrap,
        "rolling_12m": rolling_summary,
        "_rolling_frame": rolling_frame,
        "integrity": dict(result.integrity),
        "trade_count": len(result.trades),
        "order_count": len(result.orders),
    }


def _finite_json(value: Any) -> Any:
    """Replace non-finite research metrics before strict JSON serialization."""

    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _finite_json(selected)
            for key, selected in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_finite_json(selected) for selected in value)
    if isinstance(value, list):
        return [_finite_json(selected) for selected in value]
    return value


def _annual_returns(equity: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, values in equity.groupby(equity.index.year):
        starting = values.iloc[0]
        ending = values.iloc[-1]
        if not np.isfinite(starting) or float(starting) <= 0:
            continue
        rows.append(
            {
                "year": year,
                "net_return": float(ending / starting - 1.0),
                "classification": "POSITIVE" if ending > starting else "NEGATIVE",
            }
        )
    return rows


def _recent_windows(equity: pd.Series) -> list[dict[str, Any]]:
    end = equity.index[-1]
    windows = (
        ("5y", pd.DateOffset(years=5)),
        ("3y", pd.DateOffset(years=3)),
        ("2y", pd.DateOffset(years=2)),
        ("1y", pd.DateOffset(years=1)),
        ("180d", pd.DateOffset(days=180)),
    )
    rows: list[dict[str, Any]] = []
    for label, offset in windows:
        selected = equity.loc[equity.index >= end - offset]
        if len(selected) < 2 or float(selected.iloc[0]) <= 0:
            continue
        rows.append(
            {
                "window": label,
                "start": utc_iso(selected.index[0].to_pydatetime()),
                "end": utc_iso(selected.index[-1].to_pydatetime()),
                "net_return": float(selected.iloc[-1] / selected.iloc[0] - 1.0),
            }
        )
    return rows


def _analytic_regime_returns(
    features: pd.DataFrame,
    equity: pd.Series,
) -> list[dict[str, Any]]:
    aligned = features.reindex(equity.index)
    returns = equity.pct_change().fillna(0.0)
    volatility = aligned["close"].pct_change().rolling(30, min_periods=10).std()
    volatility_threshold = float(volatility.median(skipna=True))
    close_returns = aligned["close"].pct_change()
    rolling_five = aligned["close"].pct_change(5)
    rolling_twenty = aligned["close"].pct_change(20)
    range_fraction = (
        (aligned["high"] - aligned["low"])
        / aligned["close"].replace(0.0, np.nan)
    )
    range_threshold = float(
        range_fraction.rolling(90, min_periods=30).quantile(0.90).median(
            skipna=True
        )
    )
    bull = aligned["bull_regime"].fillna(False)
    bear = aligned["bear_regime"].fillna(False)
    masks = {
        "BULL_MARKET": bull,
        "BEAR_MARKET": bear,
        "SIDEWAYS_MARKET": ~(bull | bear),
        "HIGH_VOLATILITY": volatility > volatility_threshold,
        "LOW_VOLATILITY": volatility <= volatility_threshold,
        "LIQUIDITY_STRESS": (
            (range_fraction > range_threshold) & (close_returns < 0.0)
        ),
        "CRASH_PERIOD": (rolling_five <= -0.15) | (close_returns <= -0.08),
        "RECOVERY_PERIOD": (
            (rolling_twenty >= 0.15)
            & (aligned["close"] > aligned["ema_50"])
        ),
    }
    rows: list[dict[str, Any]] = []
    for label, mask in masks.items():
        selected = returns.loc[mask.fillna(False)]
        rows.append(
            {
                "regime": str(label),
                "bar_count": int(len(selected)),
                "compounded_return": float((1.0 + selected).prod() - 1.0),
                "mean_period_return": float(selected.mean()),
                "label_usage": "RETROSPECTIVE_ANALYTIC_ONLY_NOT_TRADABLE_INPUT",
                "overlap_allowed": True,
            }
        )
    return rows


def _walk_forward_oos_summary(
    result: Any,
    *,
    full_sample_expectancy_r: float,
) -> dict[str, Any]:
    folds = tuple(result.folds)
    trade_count = sum(int(fold.trade_count) for fold in folds)
    weighted_expectancy = (
        sum(
            float(fold.net_expectancy_r) * int(fold.trade_count)
            for fold in folds
        )
        / trade_count
        if trade_count
        else 0.0
    )
    finite_profit_factors = [
        float(fold.profit_factor)
        for fold in folds
        if math.isfinite(float(fold.profit_factor))
    ]
    efficiency = (
        weighted_expectancy / full_sample_expectancy_r
        if full_sample_expectancy_r > 0
        else None
    )
    return {
        "mode": result.mode,
        "fold_count": len(folds),
        "positive_folds": int(result.positive_folds),
        "positive_fold_fraction": (
            float(result.positive_folds / len(folds)) if folds else 0.0
        ),
        "oos_trade_count": trade_count,
        "oos_net_pnl_eur": float(
            sum(float(fold.net_pnl_eur) for fold in folds)
        ),
        "oos_weighted_expectancy_r": weighted_expectancy,
        "oos_mean_profit_factor": (
            float(np.mean(finite_profit_factors))
            if finite_profit_factors
            else None
        ),
        "walk_forward_efficiency": efficiency,
        "valid": bool(result.valid),
    }


def run_seven_year_strategy(
    settings: Any,
    *,
    market: str,
    timeframe: str,
    strategy_id: str,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
    warmup_bars: int = 250,
    folds: int = 6,
    purge_bars: int = 1,
    embargo_bars: int = 1,
    context_timeframes: Sequence[str] | None = None,
    context_warmup_bars: int = 80,
    evaluation_start: datetime | pd.Timestamp | None = None,
    evaluation_end: datetime | pd.Timestamp | None = None,
    window_kind: str = "FULL_HISTORY",
    output_directory: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one fixed strategy through the canonical causal seven-year stack."""

    from data.market_data import load_ohlcv
    from reporting.reports import write_backtest_report, write_run_manifest
    from research.backtest import BacktestConfig, BacktestEngine, CostModel
    from research.features import FeaturePipeline
    from research.optimization import (
        parameter_stability,
        strategy_lookahead_test,
        strategy_repainting_test,
        walk_forward_validate,
    )
    from research.strategies import get_strategy
    normalized_market = market.upper()
    normalized_timeframe = normalize_timeframe(timeframe)
    try:
        strategy = get_strategy(strategy_id)
    except KeyError:
        if not strategy_id.startswith("VOL_"):
            raise
        from research.volume_strategy_campaign import volume_strategy_adapter

        strategy = volume_strategy_adapter(strategy_id)
    declared_context = tuple(
        normalize_timeframe(str(value))
        for value in (
            context_timeframes
            if context_timeframes is not None
            else getattr(strategy, "required_higher_timeframes", ())
        )
    )
    declared_context = tuple(dict.fromkeys(declared_context))
    for selected in declared_context:
        if timeframe_delta(selected) <= timeframe_delta(normalized_timeframe):
            raise ValueError(
                "context timeframes must be strictly higher than the "
                f"execution timeframe: {selected}<={normalized_timeframe}"
            )
    selected_window_kind = str(window_kind).strip().upper()
    if selected_window_kind not in {"FULL_HISTORY", "COMMON_WINDOW"}:
        raise ValueError(f"unsupported seven-year window kind: {window_kind}")
    if (evaluation_start is None) != (evaluation_end is None):
        raise ValueError(
            "evaluation_start and evaluation_end must be supplied together"
        )
    dataset_path = (
        settings.paths.processed_data_dir
        / f"{normalized_market}_{normalized_timeframe}.parquet"
    )
    manifest = audit_dataset(
        dataset_path,
        minimum_years=minimum_years,
        warmup_bars=warmup_bars,
    )
    context_manifests: dict[str, SevenYearDatasetManifest] = {}
    for selected in declared_context:
        selected_path = (
            settings.paths.processed_data_dir
            / f"{normalized_market}_{selected}.parquet"
        )
        context_manifests[selected] = audit_dataset(
            selected_path,
            minimum_years=minimum_years,
            warmup_bars=context_warmup_bars,
        )
    constituent_manifests = [manifest, *context_manifests.values()]
    directory = output_directory or (
        settings.paths.output_dir
        / "research"
        / "seven_year"
        / "runs"
        / (
            f"{strategy_id}__{normalized_market}__{normalized_timeframe}"
            + (
                "__ctx-" + "-".join(declared_context)
                if declared_context
                else ""
            )
            + (
                "__common"
                if selected_window_kind == "COMMON_WINDOW"
                else ""
            )
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "seven_year_result.json"
    parameters = strategy.parameters()
    explicit_evaluation_start = (
        pd.Timestamp(evaluation_start)
        if evaluation_start is not None
        else None
    )
    explicit_evaluation_end = (
        pd.Timestamp(evaluation_end)
        if evaluation_end is not None
        else None
    )
    for name, value in (
        ("evaluation_start", explicit_evaluation_start),
        ("evaluation_end", explicit_evaluation_end),
    ):
        if value is not None:
            if value.tzinfo is None:
                value = value.tz_localize("UTC")
            else:
                value = value.tz_convert("UTC")
            if name == "evaluation_start":
                explicit_evaluation_start = value
            else:
                explicit_evaluation_end = value
    run_identity = stable_hash(
        {
            "schema": "seven_year_strategy_run_v3",
            "strategy_id": strategy_id,
            "parameters": parameters,
            "market": normalized_market,
            "timeframe": normalized_timeframe,
            "dataset_hashes": {
                normalized_timeframe: manifest.dataset_hash,
                **{
                    selected: item.dataset_hash
                    for selected, item in sorted(context_manifests.items())
                },
            },
            "context_timeframes": list(declared_context),
            "context_warmup_bars": context_warmup_bars,
            "evaluation_start": (
                explicit_evaluation_start.isoformat()
                if explicit_evaluation_start is not None
                else None
            ),
            "evaluation_end": (
                explicit_evaluation_end.isoformat()
                if explicit_evaluation_end is not None
                else None
            ),
            "window_kind": selected_window_kind,
            "minimum_years": minimum_years,
            "warmup_bars": warmup_bars,
            "folds": folds,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "costs": settings.costs.model_dump(mode="json"),
        },
        length=64,
    )
    if resume and report_path.is_file():
        previous = read_json(report_path)
        if (
            previous.get("schema_version") == "seven_year_strategy_run_v3"
            and previous.get("run_identity") == run_identity
        ):
            resumed_payload = dict(previous)
            normal_metrics = dict(
                dict(resumed_payload.get("normal_costs") or {}).get("metrics")
                or {}
            )
            if normal_metrics and (
                float(normal_metrics.get("net_return") or 0.0) <= 0
                or float(normal_metrics.get("profit_factor") or 0.0) <= 1
                or float(normal_metrics.get("net_expectancy_r") or 0.0) <= 0
            ):
                resumed_payload["status"] = "RESEARCH_REJECTED"
                resumed_payload["status_reasons"] = [
                    "NON_POSITIVE_NORMAL_COST_ECONOMICS"
                ]
                atomic_write_json(report_path, resumed_payload)
            return resumed_payload | {"resumed": True}

    if any(item.actual_last_timestamp is None for item in constituent_manifests):
        raise ValueError("one or more constituent datasets have no usable timestamps")
    available_start = max(
        pd.Timestamp(item.evaluation_start)
        for item in constituent_manifests
        if item.evaluation_start is not None
    )
    available_end = min(
        pd.Timestamp(item.evaluation_end)
        for item in constituent_manifests
        if item.evaluation_end is not None
    )
    selected_evaluation_start = (
        explicit_evaluation_start
        if explicit_evaluation_start is not None
        else available_start
    )
    selected_evaluation_end = (
        explicit_evaluation_end
        if explicit_evaluation_end is not None
        else available_end
    )
    if (
        selected_evaluation_start < available_start
        or selected_evaluation_end > available_end
    ):
        raise ValueError(
            "requested evaluation window exceeds the usable constituent overlap"
        )
    if not has_exact_calendar_years(
        selected_evaluation_start,
        selected_evaluation_end,
        minimum_years,
    ):
        status, reasons = strategy_history_status(
            manifests=constituent_manifests,
            timeframe=normalized_timeframe,
            trade_count=None,
            rerun_complete=False,
        )
        if status == "QUEUED":
            status = "INSUFFICIENT_MARKET_HISTORY"
            reasons = ("COMMON_EVALUATION_WINDOW_SHORTER_THAN_SEVEN_YEARS",)
        payload = {
            "schema_version": "seven_year_strategy_run_v3",
            "run_identity": run_identity,
            "generated_at": utc_iso(),
            "strategy_id": strategy_id,
            "strategy_dna_hash": stable_hash(
                {"strategy_id": strategy_id, "parameters": parameters},
                length=64,
            ),
            "legacy_strategy_dna_hash": getattr(
                strategy,
                "legacy_strategy_dna_hash",
                None,
            ),
            "canonical_adapter_dna_hash": getattr(
                strategy,
                "canonical_adapter_dna_hash",
                None,
            ),
            "material_difference_reason": getattr(
                strategy,
                "material_difference_reason",
                None,
            ),
            "parameters": parameters,
            "market": normalized_market,
            "timeframe": normalized_timeframe,
            "context_timeframes": list(declared_context),
            "window_kind": selected_window_kind,
            "status": status,
            "status_reasons": list(reasons),
            "evaluation": {
                "start": utc_iso(selected_evaluation_start),
                "end": utc_iso(selected_evaluation_end),
                "calendar_days": float(
                    (
                        selected_evaluation_end
                        - selected_evaluation_start
                    ).total_seconds()
                    / 86_400
                ),
                "warmup_bars": warmup_bars,
            },
            "dataset_manifests": {
                normalized_timeframe: manifest.model_dump(mode="json"),
                **{
                    selected: item.model_dump(mode="json")
                    for selected, item in sorted(context_manifests.items())
                },
            },
            "orders_generated": 0,
            "orders_submitted": 0,
            "live_orders_permitted": False,
        }
        payload = _finite_json(payload)
        atomic_write_json(report_path, payload)
        return payload
    required_raw_start = (
        selected_evaluation_start
        - warmup_bars * timeframe_delta(
        normalized_timeframe
        )
    )
    required_raw_starts = {
        normalized_timeframe: required_raw_start,
        **{
            selected: (
                selected_evaluation_start
                - context_warmup_bars * timeframe_delta(selected)
            )
            for selected in declared_context
        },
    }
    if (
        manifest.actual_first_timestamp is None
        or manifest.actual_first_timestamp > required_raw_start
        or any(
            context_manifests[selected].actual_first_timestamp is None
            or pd.Timestamp(
                context_manifests[selected].actual_first_timestamp
            )
            > required_raw_starts[selected]
            for selected in declared_context
        )
    ):
        payload = {
            "schema_version": "seven_year_strategy_run_v2",
            "run_identity": run_identity,
            "generated_at": utc_iso(),
            "strategy_id": strategy_id,
            "strategy_dna_hash": stable_hash(
                {"strategy_id": strategy_id, "parameters": parameters},
                length=64,
            ),
            "parameters": parameters,
            "market": normalized_market,
            "timeframe": normalized_timeframe,
            "context_timeframes": list(declared_context),
            "window_kind": selected_window_kind,
            "status": "INSUFFICIENT_MARKET_HISTORY",
            "status_reasons": ["INDICATOR_WARMUP_PREFIX_INSUFFICIENT"],
            "dataset_manifest": manifest.model_dump(mode="json"),
            "dataset_manifests": {
                normalized_timeframe: manifest.model_dump(mode="json"),
                **{
                    selected: item.model_dump(mode="json")
                    for selected, item in sorted(context_manifests.items())
                },
            },
            "required_raw_start": utc_iso(required_raw_start),
            "required_raw_starts": {
                selected: utc_iso(value)
                for selected, value in sorted(required_raw_starts.items())
            },
            "orders_generated": 0,
            "orders_submitted": 0,
            "live_orders_permitted": False,
        }
        atomic_write_json(report_path, payload)
        return payload

    raw = load_ohlcv(
        dataset_path,
        market=normalized_market,
        timeframe=normalized_timeframe,
        validate=True,
        closed_candles_only=True,
    )
    raw = raw.loc[
        (raw.index >= pd.Timestamp(required_raw_start))
        & (raw.index <= pd.Timestamp(selected_evaluation_end))
    ].copy()
    raw.attrs["market"] = normalized_market
    raw.attrs["timeframe"] = normalized_timeframe
    raw.attrs["data_provenance"] = {
        "source_type": "REAL_PROVIDER_DATA",
        "provider": manifest.provider,
        "exchange": manifest.exchange,
        "dataset_hash": manifest.dataset_hash,
        "source_segments": list(manifest.source_segments),
    }
    higher_timeframes: dict[str, pd.DataFrame] = {}
    for selected in declared_context:
        selected_manifest = context_manifests[selected]
        selected_path = (
            settings.paths.processed_data_dir
            / f"{normalized_market}_{selected}.parquet"
        )
        context = load_ohlcv(
            selected_path,
            market=normalized_market,
            timeframe=selected,
            validate=True,
            closed_candles_only=True,
        )
        context = context.loc[
            (context.index >= pd.Timestamp(required_raw_starts[selected]))
            & (context.index <= pd.Timestamp(selected_evaluation_end))
        ].copy()
        context.attrs.update(
            market=normalized_market,
            timeframe=selected,
            data_provenance={
                "source_type": "REAL_PROVIDER_DATA",
                "provider": selected_manifest.provider,
                "exchange": selected_manifest.exchange,
                "dataset_hash": selected_manifest.dataset_hash,
                "source_segments": list(selected_manifest.source_segments),
            },
        )
        higher_timeframes[selected] = context
    features = FeaturePipeline().build(
        raw,
        market=normalized_market,
        higher_timeframes=higher_timeframes,
    )
    evaluation = features.loc[
        (features.index >= pd.Timestamp(selected_evaluation_start))
        & (features.index <= pd.Timestamp(selected_evaluation_end))
    ].copy()
    evaluation.attrs.update(features.attrs)
    if not has_exact_calendar_years(
        evaluation.index[0],
        evaluation.index[-1],
        minimum_years,
    ):
        raise ValueError("evaluation window is shorter than exact calendar requirement")

    base_config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=2_000.0),
        bootstrap_samples=max(100, min(500, settings.research.bootstrap_samples)),
        monte_carlo_runs=max(100, min(500, settings.research.monte_carlo_runs)),
    )
    normal = BacktestEngine(base_config, settings=settings).run(
        {normalized_market: evaluation},
        strategy,
        parameters=parameters,
    )
    stressed_config = replace(
        base_config,
        costs=replace(
            base_config.costs,
            multiplier=max(
                1.0,
                float(settings.costs.stressed_cost_multiplier),
            ),
        ),
    )
    stressed = BacktestEngine(stressed_config, settings=settings).run(
        {normalized_market: evaluation},
        strategy,
        parameters=parameters,
    )
    double_config = replace(
        base_config,
        costs=CostModel(
            fee_fraction=base_config.costs.fee_fraction,
            slippage_bps=base_config.costs.slippage_bps,
            spread_bps=base_config.costs.spread_bps,
            multiplier=2.0,
        ),
    )
    double = BacktestEngine(double_config, settings=settings).run(
        {normalized_market: evaluation},
        strategy,
        parameters=parameters,
    )
    anchored = walk_forward_validate(
        {normalized_market: evaluation},
        strategy,
        parameters,
        base_config,
        folds=folds,
        mode="anchored",
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        settings=settings,
    )
    rolling = walk_forward_validate(
        {normalized_market: evaluation},
        strategy,
        parameters,
        base_config,
        folds=folds,
        mode="rolling",
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        settings=settings,
    )
    lookahead_safe = strategy_lookahead_test(evaluation, strategy, parameters)
    repainting_safe = strategy_repainting_test(evaluation, strategy, parameters)
    stability = parameter_stability(
        {normalized_market: evaluation},
        strategy,
        parameters,
        base_config,
        settings=settings,
        minimum_trades=minimum_trades_for_timeframe(normalized_timeframe),
    )
    minimum_trades = minimum_trades_for_timeframe(normalized_timeframe)
    stress_passed = bool(
        float(stressed.metrics["net_return"]) > 0
        and float(stressed.metrics["profit_factor"]) > 1
        and float(double.metrics["net_return"]) > 0
    )
    walk_forward_passed = bool(
        anchored.valid
        and rolling.valid
        and anchored.positive_folds >= math.ceil(folds / 2)
        and rolling.positive_folds >= math.ceil(folds / 2)
    )
    status, reasons = strategy_history_status(
        manifests=constituent_manifests,
        timeframe=normalized_timeframe,
        trade_count=int(normal.metrics["trade_count"]),
        rerun_complete=True,
        normal_economics_passed=bool(
            float(normal.metrics["net_return"]) > 0
            and float(normal.metrics["profit_factor"]) > 1
            and float(normal.metrics["net_expectancy_r"]) > 0
        ),
        causality_passed=lookahead_safe and repainting_safe,
        stress_passed=stress_passed,
        walk_forward_passed=walk_forward_passed,
        stability_passed=stability.stable,
    )
    capacity: list[dict[str, Any]] = []
    for capital in (2_000.0, 10_000.0, 100_000.0):
        impact_multiplier = math.sqrt(capital / 2_000.0)
        capacity_costs = replace(
            base_config.costs,
            slippage_bps=(
                base_config.costs.slippage_bps * impact_multiplier
            ),
            spread_bps=(
                base_config.costs.spread_bps * impact_multiplier
            ),
        )
        config = replace(
            base_config,
            initial_cash_eur=capital,
            costs=capacity_costs,
        )
        result = BacktestEngine(config, settings=settings).run(
            {normalized_market: evaluation},
            strategy,
            parameters=parameters,
        )
        capacity.append(
            {
                "capital_eur": capital,
                "net_return": result.metrics["net_return"],
                "maximum_drawdown": result.metrics["maximum_drawdown"],
                "turnover": result.metrics["turnover"],
                "transaction_costs_eur": result.metrics["transaction_costs_eur"],
                "trade_count": result.metrics["trade_count"],
                "estimated_execution_impact_bps": (
                    capacity_costs.slippage_bps
                    + capacity_costs.spread_bps / 2.0
                ),
                "impact_multiplier_vs_2000_eur": impact_multiplier,
                "market_impact_model": (
                    "SQUARE_ROOT_CAPITAL_SCALING_APPLIED_TO_SPREAD_AND_SLIPPAGE"
                ),
                "minimum_order_rejections": sum(
                    order.status == "REJECTED"
                    and "MINIMUM" in order.reason.upper()
                    for order in result.orders
                ),
            }
        )
    annual = _annual_returns(normal.equity_curve["equity"])
    regime = _analytic_regime_returns(evaluation, normal.equity_curve["equity"])
    recent = _recent_windows(normal.equity_curve["equity"])
    normal_summary = _result_summary(normal)
    stressed_summary = _result_summary(stressed)
    double_summary = _result_summary(double)
    rolling_frame = normal_summary.pop("_rolling_frame")
    stressed_summary.pop("_rolling_frame")
    double_summary.pop("_rolling_frame")
    full_expectancy = float(normal.metrics["net_expectancy_r"])
    oos_performance = {
        "anchored": _walk_forward_oos_summary(
            anchored,
            full_sample_expectancy_r=full_expectancy,
        ),
        "rolling": _walk_forward_oos_summary(
            rolling,
            full_sample_expectancy_r=full_expectancy,
        ),
    }
    report_artifacts = write_backtest_report(
        normal,
        directory,
        label="normal_costs",
    )
    rolling_path = directory / "rolling_12m_metrics.csv"
    atomic_write_text(
        rolling_path,
        rolling_frame.to_csv(index=True, lineterminator="\n"),
    )
    report_artifacts["rolling_12m_metrics"] = rolling_path
    payload = {
        "schema_version": "seven_year_strategy_run_v3",
        "run_identity": run_identity,
        "generated_at": utc_iso(),
        "strategy_id": strategy_id,
        "strategy_dna_hash": stable_hash(
            {"strategy_id": strategy_id, "parameters": parameters},
            length=64,
        ),
        "legacy_strategy_dna_hash": getattr(
            strategy,
            "legacy_strategy_dna_hash",
            None,
        ),
        "canonical_adapter_dna_hash": getattr(
            strategy,
            "canonical_adapter_dna_hash",
            None,
        ),
        "material_difference_reason": getattr(
            strategy,
            "material_difference_reason",
            None,
        ),
        "strategy_family": strategy.family,
        "parameters": parameters,
        "market": normalized_market,
        "timeframe": normalized_timeframe,
        "context_timeframes": list(declared_context),
        "window_kind": selected_window_kind,
        "status": status,
        "status_reasons": list(reasons),
        "minimum_trades_required": minimum_trades,
        "evaluation": {
            "start": utc_iso(evaluation.index[0].to_pydatetime()),
            "end": utc_iso(evaluation.index[-1].to_pydatetime()),
            "calendar_days": float(
                (evaluation.index[-1] - evaluation.index[0]).total_seconds()
                / 86_400
            ),
            "bar_count": len(evaluation),
            "warmup_bars": warmup_bars,
            "required_raw_start": utc_iso(required_raw_start),
            "dataset_hash": manifest.dataset_hash,
            "constituent_dataset_hashes": {
                normalized_timeframe: manifest.dataset_hash,
                **{
                    selected: item.dataset_hash
                    for selected, item in sorted(context_manifests.items())
                },
            },
            "common_overlap_start": utc_iso(available_start),
            "common_overlap_end": utc_iso(available_end),
        },
        "dataset_manifests": {
            normalized_timeframe: manifest.model_dump(mode="json"),
            **{
                selected: item.model_dump(mode="json")
                for selected, item in sorted(context_manifests.items())
            },
        },
        "multi_timeframe_integrity": {
            "enabled": bool(declared_context),
            "execution_timeframe": normalized_timeframe,
            "context_timeframes": list(declared_context),
            "alignment": "BACKWARD_ASOF_AFTER_FULL_SOURCE_CANDLE_CLOSE",
            "shortest_constituent_overlap_defines_window": True,
            "all_constituents_seven_year_eligible": all(
                item.seven_year_eligible for item in constituent_manifests
            ),
        },
        "normal_costs": normal_summary,
        "stressed_costs": stressed_summary,
        "double_costs": double_summary,
        "walk_forward": {
            "anchored": asdict(anchored),
            "rolling": asdict(rolling),
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "passed": walk_forward_passed,
        },
        "out_of_sample_performance": oos_performance,
        "causality": {
            "lookahead_safe": lookahead_safe,
            "repainting_safe": repainting_safe,
            "next_open_execution": normal.integrity["next_open_execution"],
            "closed_candles_only": normal.integrity["closed_candle_integrity"],
        },
        "parameter_stability": asdict(stability),
        "neighborhood_stability": asdict(stability),
        "annual_returns": annual,
        "regime_performance": regime,
        "recent_windows": recent,
        "capacity": capacity,
        "reporting_definitions": {
            "gross_return_estimate": (
                "NET_ENDING_EQUITY_PLUS_RECORDED_FEES_SPREAD_AND_SLIPPAGE"
            ),
            "profit_factor": "CLOSED_TRADE_R_MULTIPLES",
            "rolling_profit_factor": (
                "POSITIVE_PERIOD_RETURNS_DIVIDED_BY_ABSOLUTE_NEGATIVE_PERIOD_RETURNS"
            ),
            "regime_labels": (
                "RETROSPECTIVE_CAUSAL_AT_TIMESTAMP_ANALYTIC_LABELS_NOT_ENTRY_INPUTS"
            ),
            "ranking_uses_net_results_only": True,
        },
        "artifacts": {
            name: str(path.resolve()) for name, path in report_artifacts.items()
        },
        "resumed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "live_orders_permitted": False,
    }
    payload = _finite_json(payload)
    atomic_write_json(report_path, payload)
    manifest_path = write_run_manifest(
        directory / "seven_year_run_manifest.json",
        artifacts=tuple([report_path, *report_artifacts.values()]),
        settings=settings,
        run_kind="seven_year_research",
        run_id=run_identity,
    )
    payload["artifacts"]["run_manifest"] = str(manifest_path.resolve())
    payload = _finite_json(payload)
    atomic_write_json(report_path, payload)
    return payload


def record_seven_year_history_exclusion(
    settings: Any,
    *,
    strategy_id: str,
    strategy_dna_hash: str,
    markets: Sequence[str],
    timeframe: str,
    warmup_bars: int,
    material_difference_reason: str,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Persist a reproducible exclusion when post-warmup overlap is too short."""

    normalized_timeframe = normalize_timeframe(timeframe)
    normalized_markets = tuple(
        sorted(
            {
                str(market).upper().replace("/", "-").replace("_", "-")
                for market in markets
            }
        )
    )
    if not normalized_markets:
        raise ValueError("history exclusion requires at least one market")
    manifests = [
        audit_dataset(
            settings.paths.processed_data_dir
            / f"{market}_{normalized_timeframe}.parquet",
            minimum_years=minimum_years,
            warmup_bars=warmup_bars,
        )
        for market in normalized_markets
    ]
    window = common_window(manifests, minimum_years=minimum_years)
    if window.seven_year_eligible:
        raise ValueError(
            "cannot record short-history exclusion for an eligible overlap"
        )
    directory = output_directory or (
        settings.paths.output_dir
        / "research"
        / "seven_year"
        / "runs"
        / (
            f"{strategy_id}__{'-'.join(normalized_markets)}"
            f"__{normalized_timeframe}"
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    run_identity = stable_hash(
        {
            "schema": "seven_year_history_exclusion_v1",
            "strategy_id": strategy_id,
            "strategy_dna_hash": strategy_dna_hash,
            "markets": normalized_markets,
            "timeframe": normalized_timeframe,
            "warmup_bars": warmup_bars,
            "minimum_years": minimum_years,
            "dataset_hashes": {
                item.market: item.dataset_hash for item in manifests
            },
        },
        length=64,
    )
    payload = {
        "schema_version": "seven_year_history_exclusion_v1",
        "run_identity": run_identity,
        "generated_at": utc_iso(),
        "strategy_id": strategy_id,
        "strategy_dna_hash": strategy_dna_hash,
        "strategy_family": "PORTFOLIO",
        "parameters": {},
        "market": ",".join(normalized_markets),
        "assets_universe": list(normalized_markets),
        "timeframe": normalized_timeframe,
        "window_kind": "FULL_HISTORY",
        "status": "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY",
        "status_reasons": list(window.rejection_reasons),
        "evaluation": {
            "start": utc_iso(window.start) if window.start else None,
            "end": utc_iso(window.end) if window.end else None,
            "calendar_days": window.calendar_days,
            "warmup_bars": warmup_bars,
        },
        "dataset_manifests": {
            item.market: item.model_dump(mode="json") for item in manifests
        },
        "material_difference_reason": material_difference_reason,
        "rerun_performed": False,
        "exclusion_proven": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "live_orders_permitted": False,
    }
    payload = _finite_json(payload)
    atomic_write_json(directory / "seven_year_result.json", payload)
    return payload


def _robustness_score(run: Mapping[str, Any]) -> float:
    normal = dict(run.get("normal_costs") or {}).get("metrics") or {}
    stressed = dict(run.get("stressed_costs") or {}).get("metrics") or {}
    anchored = dict(run.get("walk_forward") or {}).get("anchored") or {}
    rolling = dict(run.get("walk_forward") or {}).get("rolling") or {}
    annual = list(run.get("annual_returns") or [])
    positive_year_fraction = (
        sum(float(row.get("net_return") or 0.0) > 0 for row in annual) / len(annual)
        if annual
        else 0.0
    )
    fold_count = max(
        1,
        len(anchored.get("folds") or []),
        len(rolling.get("folds") or []),
    )
    positive_fold_fraction = min(
        float(anchored.get("positive_folds") or 0),
        float(rolling.get("positive_folds") or 0),
    ) / fold_count

    def bounded(value: Any, lower: float, upper: float) -> float:
        number = float(value or 0.0)
        return min(1.0, max(0.0, (number - lower) / (upper - lower)))

    components = {
        "net_cagr": bounded(normal.get("cagr"), 0.0, 0.30),
        "profit_factor": bounded(normal.get("profit_factor"), 1.0, 2.0),
        "stressed_profit_factor": bounded(
            stressed.get("profit_factor"),
            1.0,
            1.75,
        ),
        "sharpe": bounded(normal.get("sharpe"), 0.0, 2.0),
        "sortino": bounded(normal.get("sortino"), 0.0, 3.0),
        "calmar": bounded(normal.get("calmar"), 0.0, 2.0),
        "drawdown_protection": 1.0
        - bounded(normal.get("maximum_drawdown"), 0.0, 0.50),
        "walk_forward": positive_fold_fraction,
        "positive_years": positive_year_fraction,
        "stability": (
            float(
                dict(run.get("parameter_stability") or {}).get(
                    "acceptable_score_fraction"
                )
                or 0.0
            )
        ),
    }
    weights = {
        "net_cagr": 0.15,
        "profit_factor": 0.15,
        "stressed_profit_factor": 0.15,
        "sharpe": 0.10,
        "sortino": 0.05,
        "calmar": 0.10,
        "drawdown_protection": 0.10,
        "walk_forward": 0.10,
        "positive_years": 0.05,
        "stability": 0.05,
    }
    return 100.0 * sum(
        components[name] * weights[name] for name in sorted(weights)
    )


def build_seven_year_rankings(
    root: Path,
    *,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Reconcile completed runs into explicit evidence and status rankings."""

    directory = output_directory or root / "output" / "research" / "seven_year"
    run_paths = sorted((directory / "runs").glob("**/seven_year_result.json"))
    completed: list[dict[str, Any]] = []
    for path in run_paths:
        try:
            run = dict(read_json(path))
        except (OSError, ValueError):
            continue
        run["result_path"] = str(path.resolve())
        normal_metrics = dict(
            dict(run.get("normal_costs") or {}).get("metrics") or {}
        )
        if normal_metrics and (
            float(normal_metrics.get("net_return") or 0.0) <= 0
            or float(normal_metrics.get("profit_factor") or 0.0) <= 1
            or float(normal_metrics.get("net_expectancy_r") or 0.0) <= 0
        ):
            run["status"] = "RESEARCH_REJECTED"
            run["status_reasons"] = ["NON_POSITIVE_NORMAL_COST_ECONOMICS"]
        run["robustness_score"] = _robustness_score(run)
        completed.append(run)
    completed.sort(
        key=lambda row: (
            -float(row["robustness_score"]),
            str(row.get("strategy_id")),
            str(row.get("market")),
            str(row.get("timeframe")),
        )
    )
    audit_path = directory / "history_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else {}
    audit_rankings = dict(audit.get("rankings") or {})
    rankings = {
        "SEVEN_YEAR_FULL_HISTORY": [
            row
            for row in completed
            if row.get("status") == "SEVEN_YEAR_RESEARCH_CANDIDATE"
            and row.get("window_kind", "FULL_HISTORY") == "FULL_HISTORY"
        ],
        "COMMON_WINDOW": [
            row
            for row in completed
            if row.get("window_kind") == "COMMON_WINDOW"
        ],
        "SHORT_HISTORY_RESEARCH_ONLY": list(
            audit_rankings.get("SHORT_HISTORY_RESEARCH_ONLY") or []
        ),
        "INSUFFICIENT_DATA_OR_TRADES": [
            row
            for row in completed
            if row.get("status")
            in {
                "PENDING_DATA",
                "INSUFFICIENT_MARKET_HISTORY",
                "DATA_QUALITY_FAILED",
                "INSUFFICIENT_TRADES",
            }
        ],
        "RESEARCH_REJECTED": [
            row
            for row in completed
            if row.get("status")
            in {
                "FAILED_CAUSALITY",
                "FAILED_STRESS",
                "FAILED_WALK_FORWARD",
                "FAILED_STABILITY",
                "RESEARCH_REJECTED",
            }
        ],
    }
    payload = {
        "schema_version": "seven_year_rankings_v2",
        "generated_at": utc_iso(),
        "ranking_method": {
            "score_range": [0, 100],
            "weights": {
                "net_cagr": 0.15,
                "profit_factor": 0.15,
                "stressed_profit_factor": 0.15,
                "sharpe": 0.10,
                "sortino": 0.05,
                "calmar": 0.10,
                "drawdown_protection": 0.10,
                "walk_forward": 0.10,
                "positive_years": 0.05,
                "parameter_stability": 0.05,
            },
            "clipping": "each component is clipped to [0,1]",
            "ranking_uses_net_results": True,
            "common_window_contains_all_completed_statuses": True,
            "common_window_is_not_a_promotion_shortlist": True,
        },
        "completed_run_count": len(completed),
        "status_counts": dict(
            sorted(Counter(str(row.get("status")) for row in completed).items())
        ),
        "rankings": rankings,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    rows: list[dict[str, Any]] = []
    for ranking_name, ranking in rankings.items():
        for rank, row in enumerate(ranking, 1):
            normal = dict(row.get("normal_costs") or {}).get("metrics") or {}
            stressed = dict(row.get("stressed_costs") or {}).get("metrics") or {}
            rows.append(
                {
                    "ranking": ranking_name,
                    "rank": rank,
                    "strategy_id": row.get("strategy_id")
                    or row.get("strategy_name"),
                    "strategy_dna_hash": row.get("strategy_dna_hash"),
                    "market": row.get("market")
                    or ",".join(row.get("assets_universe") or []),
                    "timeframe": row.get("timeframe"),
                    "status": row.get("status")
                    or row.get("new_seven_year_status"),
                    "robustness_score": row.get("robustness_score"),
                    "net_return": normal.get("net_return")
                    if normal
                    else row.get("legacy_net_return"),
                    "profit_factor": normal.get("profit_factor")
                    if normal
                    else row.get("legacy_profit_factor"),
                    "stressed_profit_factor": stressed.get("profit_factor"),
                    "maximum_drawdown": normal.get("maximum_drawdown")
                    if normal
                    else row.get("legacy_max_drawdown"),
                    "trade_count": normal.get("trade_count"),
                    "result_path": row.get("result_path"),
                }
            )
    rankings_json = directory / "rankings.json"
    rankings_csv = directory / "rankings.csv"
    rankings_html = directory / "rankings.html"
    pd.DataFrame(rows).to_csv(rankings_csv, index=False)
    html_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column) or ''))}</td>"
            for column in (
                "ranking",
                "rank",
                "strategy_id",
                "market",
                "timeframe",
                "status",
                "robustness_score",
                "net_return",
                "profit_factor",
                "stressed_profit_factor",
                "maximum_drawdown",
                "trade_count",
            )
        )
        + "</tr>"
        for row in rows
    )
    atomic_write_text(
        rankings_html,
        (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Seven-year rankings</title>"
            "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:"
            "collapse;width:100%}th,td{border:1px solid #ccd3da;padding:.35rem}"
            "th{background:#edf2f6}</style></head><body>"
            "<h1>Seven-year rankings</h1>"
            "<p>Net evidence only. Common-window rows include completed failures "
            "and are not a promotion shortlist.</p><table><thead><tr>"
            "<th>Ranking</th><th>Rank</th><th>Strategy</th><th>Market</th>"
            "<th>TF</th><th>Status</th><th>Score</th><th>Net return</th>"
            "<th>PF</th><th>Stress PF</th><th>Max DD</th><th>Trades</th>"
            f"</tr></thead><tbody>{html_rows}</tbody></table></body></html>"
        ),
    )
    payload["artifacts"] = {
        "json": str(rankings_json.resolve()),
        "csv": str(rankings_csv.resolve()),
        "html": str(rankings_html.resolve()),
    }
    atomic_write_json(rankings_json, payload)
    return payload


def build_legacy_comparison(
    directory: Path,
) -> dict[str, Any]:
    """Persist a transparent legacy-versus-seven-year comparison."""

    top30_path = directory / "legacy_top30_gap.json"
    if not top30_path.is_file():
        raise FileNotFoundError("legacy top-30 gap artifact is missing")
    legacy = dict(read_json(top30_path))
    rows: list[dict[str, Any]] = []
    for row in legacy.get("strategies") or []:
        rows.append(
            {
                "legacy_rank": row.get("legacy_rank"),
                "strategy_name": row.get("strategy_name"),
                "strategy_dna_hash": row.get("strategy_dna_hash"),
                "strategy_family": row.get("strategy_family"),
                "market": ",".join(row.get("assets_universe") or []),
                "timeframe": row.get("timeframe"),
                "legacy_net_return": row.get("legacy_net_return"),
                "legacy_profit_factor": row.get("legacy_profit_factor"),
                "legacy_max_drawdown": row.get("legacy_max_drawdown"),
                "seven_year_status": row.get("new_seven_year_status"),
                "seven_year_status_reasons": list(
                    row.get("new_status_reasons") or []
                ),
                "seven_year_net_return": row.get("new_net_return"),
                "seven_year_profit_factor": row.get("new_profit_factor"),
                "seven_year_max_drawdown": row.get("new_max_drawdown"),
                "seven_year_trade_count": row.get("new_trade_count"),
                "material_difference_reason": row.get(
                    "material_difference_reason"
                ),
                "result_path": row.get("seven_year_result_path"),
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )
    json_path = directory / "legacy_vs_seven_year.json"
    csv_path = directory / "legacy_vs_seven_year.csv"
    html_path = directory / "legacy_vs_seven_year.html"
    pd.DataFrame(rows).drop(
        columns=["seven_year_status_reasons"],
        errors="ignore",
    ).to_csv(csv_path, index=False)
    html_rows = "".join(
        "<tr>"
        f"<td>{row['legacy_rank']}</td>"
        f"<td>{html.escape(str(row['strategy_name']))}</td>"
        f"<td>{html.escape(str(row['timeframe']))}</td>"
        f"<td>{html.escape(str(row['legacy_net_return']))}</td>"
        f"<td>{html.escape(str(row['legacy_profit_factor']))}</td>"
        f"<td>{html.escape(str(row['seven_year_status']))}</td>"
        f"<td>{html.escape(str(row['seven_year_net_return']))}</td>"
        f"<td>{html.escape(str(row['seven_year_profit_factor']))}</td>"
        "</tr>"
        for row in rows
    )
    atomic_write_text(
        html_path,
        (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Legacy versus seven-year</title>"
            "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:"
            "collapse;width:100%}th,td{border:1px solid #ccd3da;padding:.35rem}"
            "th{background:#edf2f6}</style></head><body>"
            "<h1>Legacy versus seven-year evidence</h1>"
            "<p>Legacy values are retained as retrospective evidence and are "
            "not substituted for a causal seven-year rerun.</p>"
            "<table><thead><tr><th>Legacy rank</th><th>Strategy</th><th>TF</th>"
            "<th>Legacy return</th><th>Legacy PF</th><th>7y status</th>"
            "<th>7y return</th><th>7y PF</th></tr></thead>"
            f"<tbody>{html_rows}</tbody></table></body></html>"
        ),
    )
    payload = {
        "schema_version": "legacy_vs_seven_year_v1",
        "generated_at": utc_iso(),
        "row_count": len(rows),
        "status_counts": dict(
            sorted(
                Counter(str(row["seven_year_status"]) for row in rows).items()
            )
        ),
        "rows": rows,
        "artifacts": {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "html": str(html_path.resolve()),
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(json_path, payload)
    return payload


def _candidate_markets(candidate: Mapping[str, Any]) -> list[str]:
    raw = candidate.get("assets_universe")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [str(value).upper() for value in raw]
    else:
        values = re.findall(r"\b[A-Z0-9]+-EUR\b", str(raw or "").upper())
    if not values:
        values = re.findall(
            r"\b[A-Z0-9]+_EUR\b",
            str(candidate.get("strategy_name") or "").upper(),
        )
        values = [value.replace("_", "-") for value in values]
    return sorted(set(values))


def _legacy_row(
    candidate: Mapping[str, Any],
    manifests_by_key: Mapping[tuple[str, str], SevenYearDatasetManifest],
    *,
    legacy_rank: int,
) -> dict[str, Any]:
    timeframe = normalize_timeframe(str(candidate.get("timeframe") or "1d"))
    markets = _candidate_markets(candidate)
    matching = [
        manifests_by_key[(market, timeframe)]
        for market in markets
        if (market, timeframe) in manifests_by_key
    ]
    status, reasons = strategy_history_status(
        manifests=matching,
        timeframe=timeframe,
        trade_count=None,
        rerun_complete=False,
    )
    evidence = candidate.get("evidence") or []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    if isinstance(evidence, Sequence):
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            for key, target in (("test_start", starts), ("start", starts), ("test_end", ends), ("end", ends)):
                value = item.get(key)
                if value is None:
                    continue
                try:
                    target.append(pd.Timestamp(value))
                except (TypeError, ValueError):
                    continue
    legacy_start = min(starts).isoformat() if starts else None
    legacy_end = max(ends).isoformat() if ends else None
    legacy_days = (
        float((max(ends) - min(starts)).total_seconds() / 86_400)
        if starts and ends
        else None
    )
    return {
        "strategy_name": candidate.get("strategy_name"),
        "strategy_family": candidate.get("strategy_family"),
        "strategy_dna_hash": candidate.get("strategy_dna_hash"),
        "timeframe": timeframe,
        "assets_universe": markets,
        "legacy_rank": legacy_rank,
        "legacy_net_return": candidate.get("net_total_return"),
        "legacy_profit_factor": candidate.get("normal_profit_factor"),
        "legacy_max_drawdown": candidate.get("maximum_drawdown"),
        "legacy_test_start": legacy_start,
        "legacy_test_end": legacy_end,
        "legacy_history_days": legacy_days,
        "new_seven_year_status": status,
        "new_status_reasons": list(reasons),
        "new_net_return": None,
        "new_profit_factor": None,
        "new_max_drawdown": None,
        "new_trade_count": None,
        "new_walk_forward_result": None,
        "new_stress_result": None,
        "rank_change": None,
        "material_difference_reason": "SEVEN_YEAR_CAUSAL_RERUN_NOT_COMPLETED",
        "matching_dataset_manifests": [
            {
                "market": manifest.market,
                "timeframe": manifest.timeframe,
                "dataset_hash": manifest.dataset_hash,
                "seven_year_eligible": manifest.seven_year_eligible,
                "rejection_reason": manifest.rejection_reason,
            }
            for manifest in matching
        ],
    }


def _registered_dna_inventory(
    root: Path,
    manifests_by_key: Mapping[tuple[str, str], SevenYearDatasetManifest],
    *,
    minimum_years: int,
) -> list[dict[str, Any]]:
    database_path = root / "data_store" / "crypto.db"
    if not database_path.is_file():
        return []
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        combinations = [
            json.loads(raw)
            for (raw,) in connection.execute(
                "SELECT payload FROM strategy_combinations ORDER BY id"
            ).fetchall()
        ]
        trials = [
            json.loads(raw)
            for (raw,) in connection.execute(
                "SELECT payload FROM experiment_trials ORDER BY id"
            ).fetchall()
        ]
    finally:
        connection.close()
    latest_trial: dict[str, dict[str, Any]] = {}
    for trial in trials:
        dna = str(trial.get("strategy_dna_hash") or "")
        if not dna:
            continue
        latest_trial[dna] = trial
    rows: list[dict[str, Any]] = []
    for combination in combinations:
        dna = str(combination.get("strategy_dna_hash") or "")
        trial = latest_trial.get(dna)
        timeframes = [
            normalize_timeframe(str(value))
            for value in (
                (trial or {}).get("timeframes_tested")
                or combination.get("common_supported_timeframes")
                or combination.get("requested_timeframes")
                or []
            )
        ]
        markets = sorted(
            {
                str(value).upper()
                for value in ((trial or {}).get("assets_tested") or [])
            }
        )
        matching = [
            manifests_by_key[(market, timeframe)]
            for market in markets
            for timeframe in timeframes
            if (market, timeframe) in manifests_by_key
        ]
        expected_manifest_count = len(markets) * len(timeframes)
        reasons: list[str] = []
        static_status = str(
            combination.get("eligibility_status")
            or combination.get("status")
            or ""
        )
        if static_status == "INVALID_STATIC_RULES":
            final_status = "RESEARCH_REJECTED"
            reasons.append(
                str(combination.get("exclusion_reason") or "INVALID_STATIC_RULES")
            )
        elif trial is None:
            final_status = "QUEUED"
            reasons.append("BASELINE_TRIAL_NOT_RUN")
        elif expected_manifest_count == 0 or len(matching) < expected_manifest_count:
            final_status = "PENDING_DATA"
            reasons.append("MISSING_MARKET_TIMEFRAME_DATASET")
        elif any(
            manifest.rejection_reason == "DATA_QUALITY_FAILED"
            for manifest in matching
        ):
            final_status = "DATA_QUALITY_FAILED"
            reasons.append("CONSTITUENT_DATA_QUALITY_FAILED")
        elif any(not manifest.seven_year_eligible for manifest in matching):
            final_status = "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY"
            reasons.append("CONSTITUENT_HISTORY_SHORTER_THAN_SEVEN_YEARS")
        else:
            data_period = dict(trial.get("data_period") or {})
            try:
                period_start = pd.Timestamp(data_period["start"])
                period_end = pd.Timestamp(data_period["end"])
                period_seven_year = has_exact_calendar_years(
                    period_start,
                    period_end,
                    minimum_years,
                )
            except (KeyError, TypeError, ValueError):
                period_seven_year = False
            integrity = dict(trial.get("integrity") or {})
            if not period_seven_year:
                final_status = "QUEUED"
                reasons.append("SEVEN_YEAR_EVALUATION_RERUN_REQUIRED")
            elif integrity.get("exact_event_driven") is not True:
                final_status = "QUEUED"
                reasons.append("EXACT_CAUSAL_BACKTEST_REQUIRED")
            elif integrity.get("no_lookahead") is not True:
                final_status = "FAILED_CAUSALITY"
                reasons.append("NO_LOOKAHEAD_NOT_PROVEN")
            else:
                metrics = dict(trial.get("metrics") or {})
                primary_timeframe = timeframes[0] if timeframes else "1d"
                trade_count = int(metrics.get("trade_count") or 0)
                required_trades = minimum_trades_for_timeframe(primary_timeframe)
                if trade_count < required_trades:
                    final_status = "INSUFFICIENT_TRADES"
                    reasons.append(f"MINIMUM_TRADES_{required_trades}")
                else:
                    final_status = "BASELINE_COMPLETE"
                    reasons.append("DEEP_VALIDATION_PENDING")
        if final_status not in FINAL_STRATEGY_STATUSES:
            raise AssertionError(f"unknown seven-year strategy status: {final_status}")
        rows.append(
            {
                "strategy_dna_hash": dna,
                "combination_id": combination.get("combination_id"),
                "block_ids": list(combination.get("block_ids") or []),
                "families": list(combination.get("families") or []),
                "roles": list(combination.get("roles") or []),
                "combination_size": combination.get("combination_size"),
                "timeframes": timeframes,
                "markets": markets,
                "static_status": static_status,
                "latest_trial_id": (
                    trial.get("trial_id") if trial is not None else None
                ),
                "latest_trial_stage": (
                    trial.get("stage") if trial is not None else None
                ),
                "latest_trial_source": (
                    trial.get("source") if trial is not None else None
                ),
                "seven_year_status": final_status,
                "status_reasons": sorted(set(reasons)),
                "matching_dataset_count": len(matching),
                "expected_dataset_count": expected_manifest_count,
                "auto_live_promotion": False,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )
    return rows


def audit_repository(
    root: Path,
    *,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
    timeframes: Iterable[str] | None = None,
    markets: Iterable[str] | None = None,
    warmup_bars: int = 0,
    maximum_missing_ratio: float = DEFAULT_MAXIMUM_MISSING_RATIO,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Audit datasets and map every existing ranked strategy to a final status."""

    from reporting.top_existing_strategies import (
        collect_longlist,
        score_candidates,
        select_top_strategies,
    )

    normalized_timeframes = (
        {normalize_timeframe(value) for value in timeframes} if timeframes else None
    )
    normalized_markets = {value.upper() for value in markets} if markets else None
    normalized_dir = root / "data_store" / "normalized"
    dataset_paths = sorted(
        [
            *normalized_dir.glob("*.parquet"),
            *normalized_dir.glob("*.csv"),
        ]
    )
    manifests: list[SevenYearDatasetManifest] = []
    manifest_errors: list[dict[str, str]] = []
    non_ohlcv_manifests: list[dict[str, Any]] = []
    for path in dataset_paths:
        try:
            market, timeframe = _dataset_identity(path)
        except ValueError as exc:
            non_ohlcv_manifests.append(
                {
                    "schema_version": (
                        "seven_year_non_ohlcv_dataset_manifest_v1"
                    ),
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "dataset_hash": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "classification": (
                        "NON_OHLCV_CONTEXT_DATASET_NOT_BAR_RANKABLE"
                    ),
                    "seven_year_eligible": False,
                    "rejection_reason": "NOT_AN_OHLCV_MARKET_TIMEFRAME_DATASET",
                    "identity_inference_detail": str(exc),
                }
            )
            continue
        if normalized_timeframes and timeframe not in normalized_timeframes:
            continue
        if normalized_markets and market not in normalized_markets:
            continue
        manifests.append(
            audit_dataset(
                path,
                minimum_years=minimum_years,
                warmup_bars=warmup_bars,
                maximum_missing_ratio=maximum_missing_ratio,
                now=now,
            )
        )
    by_key = {(manifest.market, manifest.timeframe): manifest for manifest in manifests}
    registered_dna = _registered_dna_inventory(
        root,
        by_key,
        minimum_years=minimum_years,
    )
    candidates = score_candidates(collect_longlist(root))
    top30 = select_top_strategies(candidates, limit=30)
    candidate_rows = [
        _legacy_row(candidate, by_key, legacy_rank=index)
        for index, candidate in enumerate(
            sorted(
                candidates,
                key=lambda item: (
                    -float(item["scores"]["composite"]),
                    str(item["strategy_name"]),
                ),
            ),
            1,
        )
    ]
    completed_by_strategy: dict[str, dict[str, Any]] = {}
    completed_paths = sorted(
        (
            root
            / "output"
            / "research"
            / "seven_year"
            / "runs"
        ).glob("**/seven_year_result.json")
    )
    for result_path in completed_paths:
        try:
            result = dict(read_json(result_path))
        except (OSError, ValueError):
            continue
        if str(result.get("window_kind") or "FULL_HISTORY") != "FULL_HISTORY":
            continue
        strategy_name = str(result.get("strategy_id") or "")
        if not strategy_name:
            continue
        result["result_path"] = str(result_path.resolve())
        completed_by_strategy[strategy_name] = result
    for row in candidate_rows:
        result = completed_by_strategy.get(str(row["strategy_name"]))
        if result is None:
            continue
        normal = dict(result.get("normal_costs") or {}).get("metrics") or {}
        stressed = dict(result.get("stressed_costs") or {}).get("metrics") or {}
        walk_forward = dict(result.get("walk_forward") or {})
        row.update(
            {
                "new_seven_year_status": result.get("status"),
                "new_status_reasons": list(
                    result.get("status_reasons") or []
                ),
                "new_net_return": normal.get("net_return"),
                "new_profit_factor": normal.get("profit_factor"),
                "new_max_drawdown": normal.get("maximum_drawdown"),
                "new_trade_count": normal.get("trade_count"),
                "new_walk_forward_result": (
                    {
                        "anchored_positive_folds": dict(
                            walk_forward.get("anchored") or {}
                        ).get("positive_folds"),
                        "rolling_positive_folds": dict(
                            walk_forward.get("rolling") or {}
                        ).get("positive_folds"),
                        "passed": walk_forward.get("passed"),
                    }
                    if walk_forward
                    else None
                ),
                "new_stress_result": (
                    {
                        "net_return": stressed.get("net_return"),
                        "profit_factor": stressed.get("profit_factor"),
                    }
                    if stressed
                    else None
                ),
                "material_difference_reason": (
                    result.get("material_difference_reason")
                    or (
                        "POST_WARMUP_HISTORY_EXCLUSION"
                        if result.get("exclusion_proven")
                        else "SEVEN_YEAR_CAUSAL_RERUN_COMPLETED"
                    )
                ),
                "seven_year_result_path": result.get("result_path"),
                "new_evaluation": result.get("evaluation"),
            }
        )
    top30_names = {str(item["strategy_name"]) for item in top30}
    top30_rows = [
        row for row in candidate_rows if str(row["strategy_name"]) in top30_names
    ]
    top30_rows.sort(
        key=lambda row: next(
            index
            for index, item in enumerate(top30, 1)
            if item["strategy_name"] == row["strategy_name"]
        )
    )
    for index, row in enumerate(top30_rows, 1):
        row["legacy_rank"] = index

    positive_timeframe_rerun_queue = (
        build_positive_timeframe_rerun_queue(
            candidate_rows,
            minimum_years=minimum_years,
        )
    )
    eligible = [manifest for manifest in manifests if manifest.seven_year_eligible]
    short = [
        manifest
        for manifest in manifests
        if manifest.rejection_reason == "INSUFFICIENT_MARKET_HISTORY"
    ]
    quality_failed = [
        manifest
        for manifest in manifests
        if manifest.rejection_reason == "DATA_QUALITY_FAILED"
    ]
    status_counts = Counter(row["new_seven_year_status"] for row in candidate_rows)
    dna_status_counts = Counter(row["seven_year_status"] for row in registered_dna)
    common_windows: list[dict[str, Any]] = []
    for timeframe in sorted({manifest.timeframe for manifest in eligible}):
        selected = [
            manifest for manifest in eligible if manifest.timeframe == timeframe
        ]
        if len(selected) < 2:
            continue
        window = common_window(selected, minimum_years=minimum_years)
        common_windows.append(window.model_dump(mode="json"))
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    return {
        "schema_version": "seven_year_repository_audit_v1",
        "generated_at": utc_iso(generated_at),
        "repository_root": str(root.resolve()),
        "minimum_calendar_years": minimum_years,
        "minimum_calendar_days_reference": minimum_years * 365.2425,
        "exact_calendar_rule": "evaluation_start <= evaluation_end - DateOffset(years=min_years)",
        "live_orders_permitted": False,
        "summary": {
            "datasets_audited": len(manifests),
            "non_ohlcv_datasets_inventoried": len(
                non_ohlcv_manifests
            ),
            "seven_year_eligible_datasets": len(eligible),
            "short_history_datasets": len(short),
            "data_quality_failed_datasets": len(quality_failed),
            "strategy_candidates_inventoried": len(candidate_rows),
            "registered_strategy_dna_inventoried": len(registered_dna),
            "top30_inventoried": len(top30_rows),
            "positive_1h_candidates_replanned": sum(
                row["timeframe"] == "1h"
                for row in positive_timeframe_rerun_queue["jobs"]
            ),
            "positive_4h_candidates_replanned": sum(
                row["timeframe"] == "4h"
                for row in positive_timeframe_rerun_queue["jobs"]
            ),
            "positive_timeframe_jobs_queued": (
                positive_timeframe_rerun_queue["queued_count"]
            ),
            "strategy_status_counts": dict(sorted(status_counts.items())),
            "registered_dna_status_counts": dict(sorted(dna_status_counts.items())),
            "official_seven_year_results": 0,
            "official_common_window_results": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
        "minimum_trade_counts": MINIMUM_TRADES_BY_TIMEFRAME,
        "dataset_manifests": [
            manifest.model_dump(mode="json") for manifest in manifests
        ],
        "non_ohlcv_dataset_manifests": non_ohlcv_manifests,
        "dataset_manifest_errors": manifest_errors,
        "common_windows": common_windows,
        "strategy_inventory": candidate_rows,
        "registered_dna_inventory": registered_dna,
        "legacy_top30_gap": top30_rows,
        "positive_timeframe_rerun_queue": positive_timeframe_rerun_queue,
        "rankings": {
            "SEVEN_YEAR_FULL_HISTORY": [],
            "COMMON_WINDOW": [],
            "SHORT_HISTORY_RESEARCH_ONLY": [
                row
                for row in candidate_rows
                if row["new_seven_year_status"]
                == "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY"
            ],
            "INSUFFICIENT_DATA_OR_TRADES": [
                row
                for row in candidate_rows
                if row["new_seven_year_status"]
                in {
                    "PENDING_DATA",
                    "DATA_QUALITY_FAILED",
                    "INSUFFICIENT_TRADES",
                }
            ],
        },
        "limitations": [
            "Legacy performance is evidence only until rerun by the canonical causal backtester.",
            "Dataset coverage does not prove strategy warm-up coverage unless the run supplies its actual warmup_bars.",
            "Unknown local source provenance is retained as a warning and never rewritten as a known provider.",
            "No missing candles are synthesized or forward-filled.",
        ],
    }


def build_positive_timeframe_rerun_queue(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_years: int = REQUIRED_CALENDAR_YEARS,
) -> dict[str, Any]:
    """Build a deterministic, resumable plan for positive legacy 1h/4h DNA."""

    jobs: list[dict[str, Any]] = []
    for row in candidate_rows:
        timeframe = normalize_timeframe(str(row.get("timeframe") or ""))
        if timeframe not in {"1h", "4h"}:
            continue
        if float(row.get("legacy_net_return") or 0.0) <= 0.0:
            continue
        strategy_name = str(row.get("strategy_name") or "")
        strategy_dna_hash = str(row.get("strategy_dna_hash") or "")
        markets = tuple(
            sorted(str(value) for value in row.get("assets_universe") or [])
        )
        dataset_hashes = tuple(
            sorted(
                str(item.get("dataset_hash") or "")
                for item in row.get("matching_dataset_manifests") or []
            )
        )
        current_status = str(row.get("new_seven_year_status") or "QUEUED")
        adapter_kind = (
            "VOLUME_CATALOG_ADAPTER"
            if strategy_name.startswith("VOL_")
            else "CANONICAL_OR_FIXED_DNA_ADAPTER"
        )
        job_id = stable_hash(
            {
                "schema": "seven_year_positive_timeframe_job_v1",
                "strategy_name": strategy_name,
                "strategy_dna_hash": strategy_dna_hash,
                "markets": markets,
                "timeframe": timeframe,
                "minimum_years": minimum_years,
                "dataset_hashes": dataset_hashes,
            },
            length=64,
        )
        jobs.append(
            {
                "job_id": job_id,
                "resume_key": job_id,
                "strategy_name": strategy_name,
                "strategy_dna_hash": strategy_dna_hash,
                "strategy_family": row.get("strategy_family"),
                "markets": list(markets),
                "timeframe": timeframe,
                "minimum_years": minimum_years,
                "dataset_hashes": list(dataset_hashes),
                "adapter_kind": adapter_kind,
                "legacy_net_return": row.get("legacy_net_return"),
                "legacy_profit_factor": row.get("legacy_profit_factor"),
                "current_status": current_status,
                "disposition": (
                    "RUN_OR_RESUME"
                    if current_status == "QUEUED"
                    else "FINAL_STATUS_RECORDED"
                ),
                "status_reasons": list(row.get("new_status_reasons") or []),
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )
    jobs.sort(
        key=lambda row: (
            str(row["timeframe"]),
            str(row["strategy_name"]),
            str(row["job_id"]),
        )
    )
    identities = [str(row["job_id"]) for row in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError("positive timeframe rerun plan contains duplicate jobs")
    return {
        "schema_version": "seven_year_positive_timeframe_rerun_queue_v1",
        "minimum_years": minimum_years,
        "job_count": len(jobs),
        "queued_count": sum(
            row["disposition"] == "RUN_OR_RESUME" for row in jobs
        ),
        "final_status_count": sum(
            row["disposition"] == "FINAL_STATUS_RECORDED" for row in jobs
        ),
        "timeframe_counts": dict(
            sorted(Counter(str(row["timeframe"]) for row in jobs).items())
        ),
        "status_counts": dict(
            sorted(Counter(str(row["current_status"]) for row in jobs).items())
        ),
        "deduplicated": True,
        "resumable": True,
        "jobs": jobs,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def write_audit_artifacts(
    audit: Mapping[str, Any],
    directory: Path,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    manifests_dir = directory / "data_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "history_audit": directory / "history_audit.json",
        "data_coverage": directory / "data_coverage.json",
        "non_ohlcv_data_coverage": directory / "non_ohlcv_data_coverage.json",
        "strategy_inventory": directory / "strategy_inventory.json",
        "registered_dna_inventory": directory / "registered_dna_inventory.json",
        "legacy_top30_gap": directory / "legacy_top30_gap.json",
        "positive_timeframe_rerun_queue": (
            directory / "positive_timeframe_rerun_queue.json"
        ),
        "positive_timeframe_rerun_queue_csv": (
            directory / "positive_timeframe_rerun_queue.csv"
        ),
        "rankings": directory / "rankings.json",
        "strategy_inventory_csv": directory / "strategy_inventory.csv",
        "registered_dna_inventory_csv": directory / "registered_dna_inventory.csv",
        "data_coverage_csv": directory / "data_coverage.csv",
        "gap_report_markdown": directory / "gap_report.md",
        "gap_report_html": directory / "gap_report.html",
        "common_windows": directory / "common_windows.json",
        "common_windows_csv": directory / "common_windows.csv",
    }
    atomic_write_json(paths["history_audit"], dict(audit))
    atomic_write_json(
        paths["data_coverage"],
        {
            "schema_version": "seven_year_data_coverage_v1",
            "generated_at": audit["generated_at"],
            "minimum_calendar_years": audit["minimum_calendar_years"],
            "datasets": audit["dataset_manifests"],
        },
    )
    atomic_write_json(
        paths["non_ohlcv_data_coverage"],
        {
            "schema_version": "seven_year_non_ohlcv_data_coverage_v1",
            "generated_at": audit["generated_at"],
            "datasets": audit.get("non_ohlcv_dataset_manifests", []),
        },
    )
    atomic_write_json(
        paths["strategy_inventory"],
        {
            "schema_version": "seven_year_strategy_inventory_v1",
            "generated_at": audit["generated_at"],
            "strategies": audit["strategy_inventory"],
        },
    )
    atomic_write_json(
        paths["registered_dna_inventory"],
        {
            "schema_version": "seven_year_registered_dna_inventory_v1",
            "generated_at": audit["generated_at"],
            "strategies": audit["registered_dna_inventory"],
        },
    )
    atomic_write_json(
        paths["legacy_top30_gap"],
        {
            "schema_version": "seven_year_legacy_top30_gap_v1",
            "generated_at": audit["generated_at"],
            "strategies": audit["legacy_top30_gap"],
        },
    )
    atomic_write_json(
        paths["positive_timeframe_rerun_queue"],
        audit["positive_timeframe_rerun_queue"],
    )
    atomic_write_json(
        paths["rankings"],
        {
            "schema_version": "seven_year_rankings_v1",
            "generated_at": audit["generated_at"],
            "rankings": audit["rankings"],
        },
    )
    atomic_write_json(
        paths["common_windows"],
        {
            "schema_version": "seven_year_common_windows_v1",
            "generated_at": audit["generated_at"],
            "windows": audit.get("common_windows", []),
        },
    )
    pd.DataFrame(audit["strategy_inventory"]).drop(
        columns=["matching_dataset_manifests"],
        errors="ignore",
    ).to_csv(paths["strategy_inventory_csv"], index=False)
    pd.DataFrame(audit["registered_dna_inventory"]).to_csv(
        paths["registered_dna_inventory_csv"],
        index=False,
    )
    pd.DataFrame(audit["positive_timeframe_rerun_queue"]["jobs"]).drop(
        columns=["dataset_hashes", "markets", "status_reasons"],
        errors="ignore",
    ).to_csv(paths["positive_timeframe_rerun_queue_csv"], index=False)
    pd.DataFrame(audit["dataset_manifests"]).drop(
        columns=["source_segments", "quality_reasons"],
        errors="ignore",
    ).to_csv(paths["data_coverage_csv"], index=False)
    pd.DataFrame(audit.get("common_windows", [])).to_csv(
        paths["common_windows_csv"],
        index=False,
    )
    summary = dict(audit["summary"])
    status_rows = [
        (str(status), int(count))
        for status, count in dict(summary["strategy_status_counts"]).items()
    ]
    coverage_rows = sorted(
        audit["dataset_manifests"],
        key=lambda row: (str(row["timeframe"]), str(row["market"])),
    )
    markdown_lines = [
        "# Seven-year research gap report",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "No live orders are permitted by this research workflow.",
        "",
        "## Summary",
        "",
        f"- Datasets audited: {summary['datasets_audited']}",
        f"- Non-OHLCV datasets inventoried: {summary['non_ohlcv_datasets_inventoried']}",
        f"- Seven-year eligible datasets: {summary['seven_year_eligible_datasets']}",
        f"- Short-history datasets: {summary['short_history_datasets']}",
        f"- Data-quality failures: {summary['data_quality_failed_datasets']}",
        f"- Strategy candidates inventoried: {summary['strategy_candidates_inventoried']}",
        f"- Registered strategy DNA inventoried: {summary['registered_strategy_dna_inventoried']}",
        f"- Positive 1h candidates replanned: {summary['positive_1h_candidates_replanned']}",
        f"- Positive 4h candidates replanned: {summary['positive_4h_candidates_replanned']}",
        f"- Positive timeframe jobs queued: {summary['positive_timeframe_jobs_queued']}",
        f"- Official seven-year results: {summary['official_seven_year_results']}",
        "",
        "## Strategy status",
        "",
        "| Status | Count |",
        "|---|---:|",
        *[f"| {status} | {count} |" for status, count in status_rows],
        "",
        "## Dataset coverage",
        "",
        "| Market | Timeframe | Usable days | Coverage | Eligible | Reason |",
        "|---|---:|---:|---:|---:|---|",
        *[
            (
                f"| {row['market']} | {row['timeframe']} | "
                f"{float(row['usable_calendar_days']):.1f} | "
                f"{float(row['history_coverage_ratio']):.3f} | "
                f"{str(bool(row['seven_year_eligible'])).lower()} | "
                f"{row.get('rejection_reason') or ''} |"
            )
            for row in coverage_rows
        ],
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in audit["limitations"]],
        "",
    ]
    markdown = "\n".join(markdown_lines)
    atomic_write_text(paths["gap_report_markdown"], markdown)
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['market']))}</td>"
        f"<td>{html.escape(str(row['timeframe']))}</td>"
        f"<td>{float(row['usable_calendar_days']):.1f}</td>"
        f"<td>{float(row['history_coverage_ratio']):.3f}</td>"
        f"<td>{str(bool(row['seven_year_eligible'])).lower()}</td>"
        f"<td>{html.escape(str(row.get('rejection_reason') or ''))}</td>"
        "</tr>"
        for row in coverage_rows
    )
    html_document = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Seven-year research gap report</title>"
        "<style>body{font-family:system-ui;margin:2rem;color:#17212b}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd3da;padding:.4rem;text-align:left}"
        "th{background:#edf2f6}</style></head><body>"
        "<h1>Seven-year research gap report</h1>"
        f"<p>Generated: {html.escape(str(audit['generated_at']))}</p>"
        "<p><strong>No live orders are permitted by this research workflow.</strong></p>"
        f"<p>Datasets audited: {summary['datasets_audited']}; "
        f"eligible: {summary['seven_year_eligible_datasets']}; "
        f"short history: {summary['short_history_datasets']}; "
        f"quality failures: {summary['data_quality_failed_datasets']}.</p>"
        "<table><thead><tr><th>Market</th><th>Timeframe</th>"
        "<th>Usable days</th><th>Coverage</th><th>Eligible</th>"
        f"<th>Reason</th></tr></thead><tbody>{table_rows}</tbody></table>"
        "</body></html>"
    )
    atomic_write_text(paths["gap_report_html"], html_document)
    for manifest in audit["dataset_manifests"]:
        market = str(manifest["market"]).replace("-", "_")
        timeframe = str(manifest["timeframe"])
        atomic_write_json(
            manifests_dir / f"{market}_{timeframe}.json",
            manifest,
        )
    for manifest in audit.get("non_ohlcv_dataset_manifests", []):
        atomic_write_json(
            manifests_dir
            / f"non_ohlcv_{stable_hash(manifest['path'], length=16)}.json",
            manifest,
        )
    return {key: str(path.resolve()) for key, path in paths.items()}


__all__ = [
    "CommonWindow",
    "DEFAULT_MAXIMUM_MISSING_RATIO",
    "FINAL_STRATEGY_STATUSES",
    "MINIMUM_TRADES_BY_TIMEFRAME",
    "REQUIRED_CALENDAR_YEARS",
    "REQUIRED_SEVEN_YEAR_DAYS",
    "SevenYearDatasetManifest",
    "audit_dataset",
    "audit_repository",
    "build_legacy_comparison",
    "build_positive_timeframe_rerun_queue",
    "build_seven_year_rankings",
    "common_window",
    "exact_calendar_start",
    "has_exact_calendar_years",
    "minimum_trades_for_timeframe",
    "run_seven_year_strategy",
    "strategy_history_status",
    "write_audit_artifacts",
]
