from __future__ import annotations

from datetime import timedelta

import pandas as pd

from core.contracts import (
    HistoricalCoverage,
    IntelligenceRecord,
    TimestampQuality,
)
from research.features import FeaturePipeline, confirmed_fractals
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


def test_every_registered_strategy_is_long_only(features: pd.DataFrame) -> None:
    registry = strategy_registry()
    assert len(registry) == 14
    for strategy in registry.values():
        assert strategy.metadata.long_only
        output = strategy.generate(features)
        assert output.entry.dtype == bool
        assert output.exit.dtype == bool
        assert output.size_multiplier.between(0, 1).all()
