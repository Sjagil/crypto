from __future__ import annotations

from datetime import timedelta

import pandas as pd

from core.contracts import (
    HistoricalCoverage,
    IntelligenceRecord,
    TimestampQuality,
)
from research.features import (
    FeaturePipeline,
    confirmed_fractals,
    multi_timeframe_fractal_alignment,
    volume_features,
)
from research.strategies import strategy_registry


def test_fractal_is_not_visible_until_confirmation(ohlcv: pd.DataFrame) -> None:
    small = ohlcv.iloc[:9].copy()
    small.loc[:, "high"] = [1, 2, 3, 7, 3, 2, 1, 2, 1]
    small.loc[:, "low"] = [0.8, 1, 2, 3, 2, 1, 0.5, 1, 0.7]
    result = confirmed_fractals(small, left=2, right=2)
    assert not bool(result["confirmed_fractal_high"].iloc[3])
    assert bool(result["confirmed_fractal_high"].iloc[5])
    assert result["confirmed_fractal_high_price"].iloc[5] == 7


def test_intelligence_is_causal(ohlcv: pd.DataFrame) -> None:
    observed = ohlcv.index[100].to_pydatetime() + timedelta(minutes=30)
    record = IntelligenceRecord(
        event_id="event",
        source="unit",
        url="https://example.test/bitcoin",
        title="Bitcoin exchange outage",
        observed_at=observed,
        timestamp_quality=TimestampQuality.OBSERVED_ONLY,
        markets=("BTC-EUR",),
        categories=("exchange_risk",),
        relevance_score=1.0,
        sentiment_score=-1.0,
        impact_score=1.0,
        historical_coverage=HistoricalCoverage.FORWARD_ONLY,
        raw_hash="hash",
    )
    features = FeaturePipeline().build(
        ohlcv,
        market="BTC-EUR",
        intelligence=[record],
    )
    assert features.loc[: ohlcv.index[100], "intelligence_event_count"].sum() == 0
    assert features.loc[ohlcv.index[101], "intelligence_event_count"] == 1


def test_higher_timeframe_state_waits_for_source_candle_close(
    ohlcv: pd.DataFrame,
) -> None:
    higher = ohlcv.iloc[:9].copy()
    higher.index = pd.date_range(
        "2023-01-01",
        periods=len(higher),
        freq="4h",
        tz="UTC",
    )
    higher.attrs["timeframe"] = "4h"
    base_index = pd.date_range(
        "2023-01-01",
        periods=20,
        freq="1h",
        tz="UTC",
    )
    aligned = multi_timeframe_fractal_alignment(
        base_index,
        {"4h": higher},
        base_timeframe="1h",
    )
    assert pd.isna(aligned.loc[base_index[2], "fractal_source_timestamp_4h"])
    assert (
        aligned.loc[base_index[3], "fractal_source_timestamp_4h"]
        == higher.index[0]
    )


def test_every_registered_strategy_is_long_only(features: pd.DataFrame) -> None:
    registry = strategy_registry()
    assert len(registry) == 14
    for strategy in registry.values():
        assert strategy.metadata.long_only
        output = strategy.generate(features)
        assert output.entry.dtype == bool
        assert output.exit.dtype == bool
        assert output.size_multiplier.between(0, 1).all()


def test_volume_catalog_is_causal_and_labels_candle_proxies(
    ohlcv: pd.DataFrame,
) -> None:
    selected = ohlcv.iloc[:180].copy()
    baseline = volume_features(selected)
    revised = selected.copy()
    revised.loc[revised.index[-1], "close"] *= 1.10
    revised.loc[revised.index[-1], "volume"] *= 10.0
    changed = volume_features(revised)

    required = {
        "estimated_quote_volume",
        "time_of_week_relative_volume_20",
        "volume_rate_of_change_10",
        "volume_macd_histogram",
        "price_volume_trend",
        "accumulation_distribution_line",
        "negative_volume_index",
        "positive_volume_index",
        "force_index_13",
        "ease_of_movement_14",
        "klinger_volume_oscillator",
        "volume_zone_oscillator_14",
        "amihud_illiquidity_20",
    }
    assert required <= set(baseline.columns)
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1],
        changed.iloc[:-1],
    )
    assert (
        baseline.attrs["directional_volume_semantics"]
        == "CANDLE_DIRECTION_PROXY_NOT_TRADE_DELTA_OR_CVD"
    )
    assert "cvd" in baseline.attrs[
        "unavailable_without_trade_or_l2_history"
    ]


def test_time_of_week_relative_volume_uses_only_prior_matching_slots() -> None:
    index = pd.date_range(
        "2025-01-06 12:00",
        periods=24 * 7 * 8,
        freq="1h",
        tz="UTC",
    )
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        },
        index=index,
    )
    target = pd.Timestamp("2025-02-24 12:00", tz="UTC")
    frame.loc[target, "volume"] = 20.0
    result = volume_features(frame)
    assert result.loc[target, "time_of_week_relative_volume_20"] == 2.0
