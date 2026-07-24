from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core.contracts import DataValidationError
from data.market_data import (
    drop_open_candles,
    load_ohlcv,
    normalize_ohlcv,
    quality_report,
    resample_ohlcv,
    save_ohlcv,
    validate_ohlcv,
)


def test_duplicate_timestamps_fail(ohlcv: pd.DataFrame) -> None:
    duplicate = pd.concat([ohlcv.iloc[:2], ohlcv.iloc[[1]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        normalize_ohlcv(duplicate)


def test_closed_candles_resample_and_roundtrip(ohlcv: pd.DataFrame, tmp_path) -> None:
    now = ohlcv.index[-1].to_pydatetime() + timedelta(hours=2)
    with_open = pd.concat([ohlcv, ohlcv.iloc[[-1]].set_axis([pd.Timestamp(now)])])
    closed = drop_open_candles(with_open, timeframe="1h", now=now)
    assert closed.index[-1] == ohlcv.index[-1]
    four_hour = resample_ohlcv(closed, source_timeframe="1h", target_timeframe="4h")
    assert len(four_hour) == len(closed) // 4
    path, manifest = save_ohlcv(
        closed,
        tmp_path / "btc.parquet",
        market="BTC-EUR",
        timeframe="1h",
        now=now,
    )
    loaded = load_ohlcv(path, market="BTC-EUR", timeframe="1h")
    assert len(loaded) == len(closed)
    assert manifest.sha256


def test_quality_report_identifies_stale_data(ohlcv: pd.DataFrame) -> None:
    report = quality_report(
        ohlcv,
        market="BTC-EUR",
        timeframe="1h",
        maximum_staleness=timedelta(hours=6),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert report.stale
    assert not report.valid
    assert "STALE_DATA" in report.reasons


def test_candle_close_grace_rejects_just_closed_boundary(
    ohlcv: pd.DataFrame,
) -> None:
    last_start = ohlcv.index[-1].to_pydatetime()
    boundary = last_start + timedelta(hours=1)
    with pytest.raises(DataValidationError, match="open candle"):
        validate_ohlcv(
            ohlcv,
            timeframe="1h",
            now=boundary + timedelta(seconds=9),
            close_grace_seconds=10,
        )
    validated = validate_ohlcv(
        ohlcv,
        timeframe="1h",
        now=boundary + timedelta(seconds=10),
        close_grace_seconds=10,
    )
    assert validated.index[-1] == ohlcv.index[-1]


def test_drop_open_candles_applies_close_grace(ohlcv: pd.DataFrame) -> None:
    last_start = ohlcv.index[-1].to_pydatetime()
    boundary = last_start + timedelta(hours=1)
    filtered = drop_open_candles(
        ohlcv,
        timeframe="1h",
        now=boundary + timedelta(seconds=9),
        close_grace_seconds=10,
    )
    assert filtered.index[-1] == ohlcv.index[-2]
