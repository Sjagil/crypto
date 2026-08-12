from __future__ import annotations

import numpy as np
import pandas as pd

from research.volume_strategy_campaign import (
    ORDERFLOW_DATA_BLOCKERS,
    VOLUME_STRATEGY_ARCHETYPES,
    VOLUME_STRATEGY_FORWARD_START,
    backtest_volume_strategy_batch,
    volume_strategy_adapter,
    volume_strategy_dna,
)


def _frame(rows: int = 900) -> pd.DataFrame:
    index = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="1h",
        tz="UTC",
    )
    trend = np.linspace(100.0, 180.0, rows)
    cycle = np.sin(np.arange(rows) / 13.0) * 2.0
    close = trend + cycle
    open_ = close * (1.0 + np.sin(np.arange(rows) / 7.0) * 0.001)
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = 100.0 + np.cos(np.arange(rows) / 11.0) * 20.0
    volume[::37] *= 2.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_volume_dna_covers_every_archetype_and_complete_plateau() -> None:
    pairs = (("BTC-EUR", "1h"), ("ETH-EUR", "4h"))
    rows = volume_strategy_dna(pairs)
    assert len(rows) == len(pairs) * len(VOLUME_STRATEGY_ARCHETYPES) * 5
    assert len({row.dna_hash for row in rows}) == len(rows)
    for pair in pairs:
        selected = [
            row
            for row in rows
            if (row.market, row.timeframe) == pair
        ]
        for archetype in VOLUME_STRATEGY_ARCHETYPES:
            assert {
                row.coordinate
                for row in selected
                if row.archetype == archetype
            } == {0, 1, 2, 3, 4}


def test_volume_batch_is_next_open_long_only_and_cost_stressed() -> None:
    frame = _frame()
    rows = tuple(
        row
        for row in volume_strategy_dna((("BTC-EUR", "1h"),))
        if row.archetype == "DONCHIAN_RVOL_BREAKOUT"
    )
    result = backtest_volume_strategy_batch(
        frame,
        frame,
        rows,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        stressed_cost_multiplier=2.0,
    )
    assert len(result.returns) == len(frame) - 1
    assert result.returns.index[0] == frame.index[1]
    assert result.positions.ge(0.0).all().all()
    assert result.positions.le(0.20 + 1e-12).all().all()
    assert (
        result.stressed_returns.sum()
        <= result.returns.sum() + 1e-12
    ).all()
    assert set(result.regimes) == {
        "btc_phase",
        "btc_volatility",
        "asset_trend",
        "participation",
        "session",
    }


def test_orderflow_families_fail_closed_without_real_history() -> None:
    assert ORDERFLOW_DATA_BLOCKERS["CVD_DIVERGENCE"].startswith(
        "MISSING_HISTORICAL"
    )
    assert "L2" in ORDERFLOW_DATA_BLOCKERS[
        "ORDER_BOOK_IMBALANCE_MICROPRICE"
    ]


def test_forward_observer_start_is_immutable() -> None:
    assert VOLUME_STRATEGY_FORWARD_START == "2026-07-27T00:00:00+00:00"


def test_canonical_volume_adapter_preserves_frozen_signal_parameters() -> None:
    strategy_id = "VOL_BTC_EUR_1h_DONCHIAN_RVOL_BREAKOUT_N2"
    adapter = volume_strategy_adapter(strategy_id)
    selected = next(
        row
        for row in volume_strategy_dna((("BTC-EUR", "1h"),))
        if row.strategy_id == strategy_id
    )
    assert adapter.legacy_strategy_dna_hash == selected.dna_hash
    assert {
        key: adapter.parameters()[key] for key in selected.parameters
    } == dict(selected.parameters)
    frame = _frame()
    baseline = adapter.generate(frame)
    revised = frame.copy()
    revised.loc[revised.index[-1], "close"] *= 1.05
    revised.loc[revised.index[-1], "high"] *= 1.05
    revised.loc[revised.index[-1], "volume"] *= 5.0
    changed = adapter.generate(revised)
    pd.testing.assert_series_equal(
        baseline.entry.iloc[:-1],
        changed.entry.iloc[:-1],
    )
    assert baseline.stop_distance.gt(0).all()
    assert baseline.target_distance.gt(0).all()
