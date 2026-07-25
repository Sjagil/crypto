from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multi_alpha_ensemble import (
    FROZEN_COMPONENT_DNA,
    MultiAlphaEnsembleParameters,
    backtest_multi_alpha_ensemble,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 800) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2023-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        generator = np.random.default_rng(300 + offset)
        returns = generator.normal(
            0.0008 + offset * 0.00005,
            0.012 + offset * 0.001,
            rows,
        )
        close = (100.0 + 10.0 * offset) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]] * (
            1.0 + generator.normal(0.0, 0.001, rows)
        )
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.003,
                "low": np.minimum(open_, close) * 0.997,
                "close": close,
                "volume": np.full(rows, 1_000.0),
            },
            index=index,
        )
    return result


def _component_weights(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    index = frames["BTC-EUR"].index
    columns = list(frames)
    result: dict[str, pd.DataFrame] = {}
    for component_index, (name, _) in enumerate(
        FROZEN_COMPONENT_DNA
    ):
        weights = pd.DataFrame(
            0.0,
            index=index,
            columns=columns,
        )
        first = columns[component_index % len(columns)]
        second = columns[(component_index + 1) % len(columns)]
        active = (
            np.arange(len(index)) // (35 + component_index * 5)
        ) % 2 == 0
        weights.loc[active, first] = 0.20
        weights.loc[active, second] = 0.10
        result[name] = weights
    return result


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=(
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )


def test_ensemble_dna_is_single_fixed_component_set() -> None:
    parameters = MultiAlphaEnsembleParameters()

    assert parameters.component_dna == FROZEN_COMPONENT_DNA
    assert parameters.target_annualized_volatility == 0.10
    assert parameters.volatility_lookback == 60
    assert len(parameters.dna_hash) == 64


def test_ensemble_is_causal_bounded_and_orderless() -> None:
    frames = _frames()
    result = backtest_multi_alpha_ensemble(
        frames,
        _component_weights(frames),
        MultiAlphaEnsembleParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    assert result.integrity[
        "component_weights_known_before_meta_decision"
    ]
    assert result.integrity[
        "meta_decision_execution_next_open"
    ]
    assert result.integrity["one_additional_causal_lag"]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["orders_generated"] == 0
    assert (
        result.metrics["maximum_realized_exposure"]
        <= 0.40 + 1e-12
    )


def test_ensemble_cost_stress_is_monotonic() -> None:
    frames = _frames()
    components = _component_weights(frames)
    normal = backtest_multi_alpha_ensemble(
        frames,
        components,
        MultiAlphaEnsembleParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_multi_alpha_ensemble(
        frames,
        components,
        MultiAlphaEnsembleParameters(),
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )

    assert stressed.metrics["net_return"] <= normal.metrics[
        "net_return"
    ]
    assert stressed.executed_weights.equals(
        normal.executed_weights
    )


def test_future_component_weights_do_not_change_prior_allocations() -> None:
    frames = _frames()
    components = _component_weights(frames)
    cutoff = frames["BTC-EUR"].index[600]
    baseline = backtest_multi_alpha_ensemble(
        frames,
        components,
        MultiAlphaEnsembleParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed = {
        name: weights.copy()
        for name, weights in components.items()
    }
    for weights in changed.values():
        weights.loc[weights.index > cutoff] = 0.0
        weights.loc[
            weights.index > cutoff,
            "LINK-EUR",
        ] = 0.20
    revised = backtest_multi_alpha_ensemble(
        frames,
        changed,
        MultiAlphaEnsembleParameters(),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )

    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        revised.executed_weights.loc[:cutoff],
    )


def test_conflicting_duplicate_component_timestamp_fails_closed() -> None:
    frames = _frames()
    components = _component_weights(frames)
    name = FROZEN_COMPONENT_DNA[0][0]
    source = components[name]
    duplicate = source.iloc[[-1]].copy()
    duplicate.iloc[0, 0] = 0.19
    components[name] = pd.concat([source, duplicate])

    with pytest.raises(
        ValueError,
        match="conflicting duplicate component weights",
    ):
        backtest_multi_alpha_ensemble(
            frames,
            components,
            MultiAlphaEnsembleParameters(),
            fee_rate=0.0025,
            slippage_bps=8.0,
            spread_bps=5.0,
            portfolio_policy=_policy(),
        )
