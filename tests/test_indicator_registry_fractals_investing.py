from __future__ import annotations

import numpy as np
import pandas as pd

from research.features import (
    FeaturePipeline,
    anchored_vwap,
    confirmed_fractals,
    fractal_research_labels,
    multi_timeframe_fractal_alignment,
)
from research.indicator_registry import CoverageStatus, indicator_registry
from research.investing import InvestmentScorer


def _fractal_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=15, freq="h", tz="UTC")
    high = np.array([2, 3, 4, 5, 9, 5, 4, 3, 2, 3, 4, 5, 4, 3, 2], dtype=float)
    low = np.array([1, 2, 3, 4, 5, 4, 3, 2, 0.5, 2, 3, 4, 3, 2, 1], dtype=float)
    close = (high + low) / 2.0
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.arange(1, len(index) + 1, dtype=float),
        },
        index=index,
    )


def test_registry_has_explicit_complete_deterministic_coverage() -> None:
    first = indicator_registry()
    second = indicator_registry()
    report = first.report()
    assert report["source_item_occurrences"] == 1149
    assert report["unique_canonical_indicators"] == len(first) == 1147
    assert sum(report["counts_by_status"].values()) == len(first)
    assert set(report["counts_by_status"]) == {status.value for status in CoverageStatus}
    assert report["registry_hash"] == second.report()["registry_hash"]
    names = [item.canonical_name for item in first.definitions()]
    assert len(names) == len(set(names))
    for alias, target in report["aliases"].items():
        assert first.resolve(alias).canonical_name == target
        assert first.get(alias).status is CoverageStatus.IMPLEMENTED_AS_ALIAS


def test_research_only_and_derivatives_context_cannot_create_entries() -> None:
    definitions = indicator_registry().definitions()
    research_only = [
        item
        for item in definitions
        if item.status is CoverageStatus.RESEARCH_ONLY
    ]
    assert research_only
    assert all(not item.tradable and not item.combinable for item in research_only)
    derivatives = [
        item for item in definitions if item.family in {"DERIVATIVES", "OPTIONS"}
    ]
    assert derivatives
    assert all(not item.tradable for item in derivatives)
    assert all(item.primary_role.value != "ENTRY" for item in derivatives)


def test_confirmed_fractal_windows_use_exact_confirmation_lag() -> None:
    frame = _fractal_frame()
    for window in (3, 5, 7):
        side = (window - 1) // 2
        result = confirmed_fractals(frame, left=side, right=side)
        pivot = 4
        confirmation = pivot + side
        assert bool(result["confirmed_fractal_high"].iloc[confirmation])
        assert result["confirmed_fractal_high_price"].iloc[confirmation] == 9
        assert (
            result["fractal_high_pivot_timestamp"].iloc[confirmation]
            == frame.index[pivot]
        )
        assert (
            result["fractal_high_confirmation_timestamp"].iloc[confirmation]
            == frame.index[confirmation]
        )
        assert not result["confirmed_fractal_high"].iloc[:confirmation].any()


def test_tradable_features_exclude_raw_fractals_and_forward_labels(
    ohlcv: pd.DataFrame,
) -> None:
    features = FeaturePipeline().build(ohlcv)
    assert not any(column.startswith("raw_fractal_") for column in features)
    assert not {
        "post_fractal_mfe",
        "post_fractal_mae",
        "fractal_efficiency",
    }.intersection(features)
    labels = fractal_research_labels(ohlcv)
    assert labels.attrs["research_labels_only"] is True
    assert labels.attrs["available_after_bars"] == 20


def test_anchored_vwap_and_mtf_alignment_are_prefix_causal(
    ohlcv: pd.DataFrame,
) -> None:
    split = len(ohlcv) - 20
    prefix = ohlcv.iloc[:split]
    pd.testing.assert_series_equal(
        anchored_vwap(prefix),
        anchored_vwap(ohlcv).iloc[:split],
    )
    base_index = prefix.index
    short = {"4h": ohlcv.iloc[: split - 10 : 4]}
    extended = {"4h": ohlcv.iloc[::4]}
    short_result = multi_timeframe_fractal_alignment(base_index, short)
    extended_result = multi_timeframe_fractal_alignment(base_index, extended)
    common_end = short["4h"].index[-1]
    pd.testing.assert_frame_equal(
        short_result.loc[:common_end],
        extended_result.loc[:common_end],
    )


def test_investing_score_is_transparent_and_missing_lowers_confidence() -> None:
    scorer = InvestmentScorer()
    partial = scorer.score(
        {
            "market_cap_fdv": 80,
            "market_cap_tvl": 20,
            "active_users": 70,
            "exploit_history": 10,
        }
    )
    complete = scorer.score(
        {
            component: 50
            for components in scorer.dimensions.values()
            for component in components
        }
    )
    assert partial.investing_only
    assert not partial.creates_trading_signal
    assert partial.confidence < complete.confidence == 1.0
    assert partial.missing_components
    assert set(partial.subscores) == set(scorer.dimensions)
    assert complete.configuration_hash == scorer.configuration_hash
