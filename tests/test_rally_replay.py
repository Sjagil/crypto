from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from core.rally_replay import _find_replay_candidate


def test_replay_prefers_first_pullback_after_causal_impulse() -> None:
    start = datetime(2026, 8, 8, tzinfo=UTC)
    index = pd.date_range(start, periods=8, freq="15min", tz="UTC")
    feature = pd.DataFrame(
        {
            "open": [99.0, 101.8, 101.5, 101.7, 101.9, 102.0, 102.1, 102.2],
            "high": [102.0, 101.9, 101.8, 102.0, 102.1, 102.2, 102.3, 102.4],
            "low": [98.8, 101.0, 101.1, 101.4, 101.6, 101.8, 101.9, 102.0],
            "close": [101.8, 101.5, 101.7, 101.9, 102.0, 102.1, 102.2, 102.3],
            "volume": [200.0] * 8,
            "atr_14": [2.0] * 8,
            "ema_20": [100.0] * 8,
            "return_15m": [0.02, -0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001],
            "return_1h": [0.02, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021],
            "normalized_return_1h": [1.0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "relative_volume_20": [1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        },
        index=index,
    )
    five_index = pd.date_range(
        start - timedelta(hours=2),
        periods=40,
        freq="5min",
        tz="UTC",
    )
    closes = [100.0 + 0.03 * row for row in range(len(five_index))]
    five = pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.05 for value in closes],
            "low": [value - 0.05 for value in closes],
            "close": closes,
            "volume": [50.0] * len(five_index),
        },
        index=five_index,
    )
    breadth = pd.Series(0.75, index=index)
    btc_return = pd.Series(0.0, index=index)

    result = _find_replay_candidate(
        "SOL-EUR",
        feature,
        five,
        breadth,
        btc_return,
        start=start,
        end=start + timedelta(hours=2),
        roundtrip_cost_fraction=0.006,
    )

    assert result["family"] == "FIRST_PULLBACK_AFTER_IMPULSE_V1"
    assert result["counterfactual_is_not_live_fill"] is True
    assert result["microstructure_evidence"] == "HISTORICAL_ORDERFLOW_NOT_AVAILABLE"

