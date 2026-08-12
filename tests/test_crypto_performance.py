from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.crypto_performance import analyze_crypto_performance


def test_crypto_analyzer_is_json_safe_and_includes_crypto_risk_metrics() -> None:
    index = pd.date_range("2025-01-01", periods=24 * 30, freq="1h", tz="UTC")
    returns = 0.0001 + np.sin(np.arange(len(index)) / 10.0) * 0.002
    returns[240] = -0.12
    equity = pd.Series(100 * np.cumprod(1 + returns), index=index)
    benchmark = pd.Series(100 * np.cumprod(1 + returns * 0.6), index=index)

    report = analyze_crypto_performance(equity, benchmark_equity=benchmark)

    assert report["schema_version"] == "crypto_performance_risk_v1"
    assert report["analysis_only"] is True
    assert report["risk"]["max_drawdown"] < 0
    assert report["risk"]["expected_shortfall_95"] <= report["risk"]["value_at_risk_95"]
    assert report["crypto_specific"]["max_decline_24h"] < 0
    assert report["crypto_specific"]["crash_frequency_24h"]["at_least_5pct"] > 0
    assert report["performance"]["upside_capture"] is not None
    assert report["side_effects"]["orders_submitted"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([100.0, 101.0], index=pd.date_range("2025-01-01", periods=2, freq="1h")),
        pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"]),
        ),
        pd.Series(
            [100.0, 0.0],
            index=pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC"),
        ),
    ],
)
def test_crypto_analyzer_rejects_noncausal_or_invalid_equity(series: pd.Series) -> None:
    with pytest.raises(ValueError):
        analyze_crypto_performance(series)
