from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS
from research.classical_strategy_factory import (
    CLASSICAL_FACTORY_DEFAULT_TRIALS,
    TIMEFRAME_ROUTES,
    classical_factory_plan,
    classical_family_catalog,
    generate_classical_strategy_dna,
)
from research.combinatorial_lab import (
    CLASSICAL_DISABLED_FAMILY_INTERFACES,
    CLASSICAL_ECONOMIC_FAMILY_TEMPLATES,
    BlockRole,
    LabRunner,
    signal_block_registry,
)
from research.features import (
    complexity_features,
    fractal_dimension_features,
    market_structure_features,
    mean_reversion_features,
    momentum_features,
    trend_features,
    volatility_features,
    volume_features,
)


def _ohlcv(rows: int) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    cycle = np.sin(np.arange(rows, dtype=float) / 9.0)
    close = 100.0 + np.arange(rows, dtype=float) * 0.05 + cycle
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.7,
            "low": close - 0.8,
            "close": close,
            "volume": 1_000.0 + 100.0 * np.cos(np.arange(rows) / 7.0),
        },
        index=index,
    )
    frame.attrs["timeframe"] = "1h"
    return frame


def test_classical_factory_preregisters_exact_unique_dna() -> None:
    first = classical_factory_plan()
    second = classical_factory_plan()

    assert first["trial_count"] == CLASSICAL_FACTORY_DEFAULT_TRIALS
    assert first["search_space_hash"] == second["search_space_hash"]
    assert first["strategy_dna_hashes"] == second["strategy_dna_hashes"]
    assert len(set(first["strategy_dna_hashes"])) == CLASSICAL_FACTORY_DEFAULT_TRIALS
    assert first["economic_family_count"] == len(
        CLASSICAL_ECONOMIC_FAMILY_TEMPLATES
    )
    assert first["orders_generated"] == 0
    assert first["orders_submitted"] == 0


def test_classical_factory_enforces_bounded_nonredundant_grammar() -> None:
    registry = signal_block_registry()
    for row in generate_classical_strategy_dna():
        roles = [registry[block_id].role for block_id in row.block_ids]
        assert roles.count(BlockRole.ENTRY_TRIGGER) == 1
        assert roles.count(BlockRole.REGIME_FILTER) <= 1
        confirmations = [
            registry[block_id]
            for block_id in row.block_ids
            if registry[block_id].role
            in {BlockRole.CONFIRMATION, BlockRole.TREND_FILTER}
        ]
        assert len(confirmations) <= 2
        assert len({block.redundancy_group for block in confirmations}) == len(
            confirmations
        )


def test_classical_routes_are_causal_and_higher_context_is_really_higher() -> None:
    for route in TIMEFRAME_ROUTES:
        signal = TIMEFRAME_SECONDS[route.signal_timeframe]
        assert TIMEFRAME_SECONDS[route.setup_timeframe] >= signal
        assert TIMEFRAME_SECONDS[route.regime_timeframe] >= signal
        if route.route_id.startswith("MTF_"):
            assert TIMEFRAME_SECONDS[route.setup_timeframe] > signal
            assert TIMEFRAME_SECONDS[route.regime_timeframe] > signal


def test_data_pending_families_have_no_synthetic_signal_blocks() -> None:
    registry = signal_block_registry()
    catalog = classical_family_catalog()

    assert catalog["synthetic_orderflow_used"] is False
    assert catalog["synthetic_derivatives_used"] is False
    assert catalog["disabled_family_count"] == len(
        CLASSICAL_DISABLED_FAMILY_INTERFACES
    )
    assert not set(CLASSICAL_DISABLED_FAMILY_INTERFACES).intersection(registry)


def test_lab_frames_add_causal_cross_sectional_breadth(
    isolated_settings,
    tmp_path,
) -> None:
    paths = isolated_settings.paths.model_copy(
        update={
            "lab_dir": (tmp_path / "lab").resolve(),
            "database_path": (tmp_path / "lab.db").resolve(),
            "checkpoints_dir": (tmp_path / "checkpoints").resolve(),
            "processed_data_dir": (tmp_path / "normalized").resolve(),
        }
    )
    runner = LabRunner(isolated_settings.model_copy(update={"paths": paths}))
    frames, _, provenance = runner._frames(
        markets=("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"),
        timeframe="1h",
        rows=300,
        data_mode="synthetic",
    )
    for period in (20, 50, 200):
        column = f"breadth_fraction_above_mean_{period}d"
        expected = pd.concat(
            [
                (frame["close"] > frame[f"ema_{period}"]).where(
                    frame[f"ema_{period}"].notna()
                )
                for frame in frames.values()
            ],
            axis=1,
        ).astype(float).mean(axis=1, skipna=True)
        for frame in frames.values():
            pd.testing.assert_series_equal(
                frame[column],
                expected.reindex(frame.index),
                check_names=False,
            )
            assert frame.attrs["data_provenance"][
                "cross_sectional_breadth"
            ]["closed_bars_only"]
    assert set(provenance) == set(frames)


def test_new_classical_features_are_prefix_causal() -> None:
    short = _ohlcv(260)
    extended = _ohlcv(300)
    functions = (
        trend_features,
        momentum_features,
        mean_reversion_features,
        volatility_features,
        volume_features,
        market_structure_features,
    )
    selected = (
        "linear_regression_slope_50",
        "ichimoku_bullish_reclaim",
        "multi_horizon_momentum_score",
        "mad_zscore_30",
        "volatility_expansion_breakout",
        "prior_volume_dryup",
        "previous_day_high",
        "previous_week_high",
    )
    short_features = pd.concat([function(short) for function in functions], axis=1)
    extended_features = pd.concat(
        [function(extended) for function in functions],
        axis=1,
    ).loc[short.index]
    pd.testing.assert_frame_equal(
        short_features.loc[:, selected],
        extended_features.loc[:, selected],
        check_dtype=False,
    )

    short_complexity = complexity_features(short["close"])
    extended_complexity = complexity_features(extended["close"]).loc[short.index]
    pd.testing.assert_frame_equal(
        short_complexity,
        extended_complexity,
        check_dtype=False,
    )

    advanced = fractal_dimension_features(
        short["close"],
        window=64,
        include_advanced_estimators=True,
    )
    assert {"dfa_exponent", "generalized_hurst_width"}.issubset(advanced.columns)
