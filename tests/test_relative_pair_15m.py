from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from research.relative_pair_15m import (
    STRATEGY_SPECS,
    _simulate,
    align_closed_context,
    build_synthetic_cross,
    catalogue,
    pair_features,
    target_states,
)


def _frame(index: pd.DatetimeIndex, start: float, slope: float) -> pd.DataFrame:
    close = start + slope * np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )


def test_synthetic_cross_uses_inner_join_without_forward_fill() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    base = _frame(index.delete(2), 100.0, 1.0)
    benchmark = _frame(index, 50.0, 0.5)

    cross = build_synthetic_cross(base, benchmark, symbol="TEST/BTC")

    assert list(cross.index) == list(index.delete(2))
    assert cross.attrs["no_forward_fill"] is True
    assert cross.attrs["native_market"] is False
    timestamp = cross.index[0]
    assert cross.loc[timestamp, "high"] == (
        base.loc[timestamp, "high"] / benchmark.loc[timestamp, "low"]
    )
    assert cross.loc[timestamp, "low"] == (
        base.loc[timestamp, "low"] / benchmark.loc[timestamp, "high"]
    )


def test_context_is_available_only_after_higher_timeframe_close() -> None:
    execution_index = pd.date_range(
        "2025-01-01",
        periods=24,
        freq="15min",
        tz="UTC",
    )
    hourly_index = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
    context = build_synthetic_cross(
        _frame(hourly_index, 100.0, 1.0),
        _frame(hourly_index, 50.0, 0.1),
        symbol="TEST/BTC",
    )

    aligned = align_closed_context(execution_index, {"1h": context})
    source = pd.to_datetime(aligned["1h_source_timestamp"], utc=True)
    decisions = pd.Series(execution_index + timedelta(minutes=15), index=execution_index)

    assert (
        (source.dropna() + timedelta(hours=1))
        <= decisions.loc[source.dropna().index]
    ).all()
    assert pd.isna(aligned.iloc[0]["1h_source_timestamp"])


def test_pair_strategies_are_long_only_rotation_targets() -> None:
    execution_index = pd.date_range(
        "2025-01-01",
        periods=400,
        freq="15min",
        tz="UTC",
    )
    hourly_index = pd.date_range("2024-12-20", periods=400, freq="1h", tz="UTC")
    four_hour_index = pd.date_range("2024-11-01", periods=400, freq="4h", tz="UTC")
    base = _frame(execution_index, 100.0, 0.08)
    benchmark = _frame(execution_index, 50.0, 0.01)
    execution = build_synthetic_cross(base, benchmark, symbol="TEST/BTC")
    context = {
        "1h": build_synthetic_cross(
            _frame(hourly_index, 100.0, 0.3),
            _frame(hourly_index, 50.0, 0.02),
            symbol="TEST/BTC",
        ),
        "4h": build_synthetic_cross(
            _frame(four_hour_index, 100.0, 0.6),
            _frame(four_hour_index, 50.0, 0.03),
            symbol="TEST/BTC",
        ),
    }
    features = pair_features(execution, context)

    for spec in STRATEGY_SPECS:
        targets = target_states(features, spec.mechanism)
        assert set(targets.unique()) <= {"BASE", "BENCHMARK", "CASH"}
        assert "SHORT" not in set(targets.unique())


def test_two_leg_rotation_charges_both_switch_legs() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    base = _frame(index, 100.0, 0.0)
    benchmark = _frame(index, 50.0, 0.0)
    targets = pd.Series(
        ["BASE", "BASE", "BENCHMARK", "BENCHMARK", "CASH", "CASH", "CASH", "CASH"],
        index=index,
        dtype="string",
    )

    result = _simulate(
        base,
        benchmark,
        targets,
        STRATEGY_SPECS[0],
        fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=8.0,
        cost_multiplier=1.0,
    )

    assert result["rotations"] == 2
    assert result["closed_holding_episodes"] == 2
    assert result["modeled_costs_eur"] > 0
    assert result["ending_equity_eur"] < 10_000.0


def test_catalogue_never_grants_live_authority() -> None:
    payload = catalogue()

    assert payload["integrity"]["shorting"] is False
    assert all(row["live_authority"] is False for row in payload["strategies"])
    assert all(row["native_bitvavo_market"] is False for row in payload["pairs"])
