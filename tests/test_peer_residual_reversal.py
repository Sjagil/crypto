from __future__ import annotations

import numpy as np
import pandas as pd

from research.peer_residual_reversal import (
    PeerResidualReversalParameters,
    backtest_peer_residual_reversal,
    peer_residual_reversal_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 950) -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2022-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )
    generator = np.random.default_rng(718)
    common = generator.normal(0.0007, 0.010, rows)
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        idiosyncratic = generator.normal(
            0.0,
            0.003 + offset * 0.0005,
            rows,
        )
        if market != "BTC-EUR":
            idiosyncratic[
                260 + 17 * offset :: 73 + offset
            ] -= 0.06
        returns = common * (0.85 + offset * 0.08) + idiosyncratic
        close = (100.0 + offset * 10.0) * np.exp(
            np.cumsum(returns)
        )
        open_ = np.r_[close[0], close[:-1]]
        result[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.004,
                "low": np.minimum(open_, close) * 0.996,
                "close": close,
                "volume": np.full(rows, 10_000.0),
            },
            index=index,
        )
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
        minimum_history_observations=200,
    )


def test_peer_residual_family_is_exactly_four_unique_dna() -> None:
    rows = peer_residual_reversal_parameter_set()
    assert len(rows) == 4
    assert len({row.dna_hash for row in rows}) == 4
    assert {
        (row.beta_lookback, row.residual_horizon)
        for row in rows
    } == {(30, 3), (30, 5), (60, 3), (60, 5)}


def test_peer_residual_is_causal_bounded_and_orderless() -> None:
    result = backtest_peer_residual_reversal(
        _frames(),
        PeerResidualReversalParameters(
            beta_lookback=30,
            residual_horizon=3,
        ),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    assert result.integrity["strictly_prior_peer_beta_estimation"]
    assert result.integrity["strictly_prior_zscore_baseline"]
    assert result.integrity[
        "decision_at_close_execution_next_open"
    ]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity[
        "maximum_position_exposure_respected"
    ]
    assert result.integrity["btc_used_as_regime_only"]
    assert result.integrity["orders_generated"] == 0
    assert result.metrics["maximum_realized_exposure"] <= 0.40


def test_peer_residual_cost_stress_is_monotonic() -> None:
    frames = _frames()
    parameters = PeerResidualReversalParameters(
        beta_lookback=60,
        residual_horizon=5,
    )
    normal = backtest_peer_residual_reversal(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_peer_residual_reversal(
        frames,
        parameters,
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )
    assert stressed.metrics["net_return"] <= normal.metrics[
        "net_return"
    ]
    pd.testing.assert_frame_equal(
        normal.executed_weights,
        stressed.executed_weights,
    )


def test_future_prices_do_not_change_prior_peer_allocations() -> None:
    frames = _frames()
    parameters = PeerResidualReversalParameters(
        beta_lookback=30,
        residual_horizon=5,
    )
    cutoff = frames["BTC-EUR"].index[700]
    baseline = backtest_peer_residual_reversal(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    revised = {market: frame.copy() for market, frame in frames.items()}
    for market, frame in revised.items():
        mask = frame.index > cutoff
        multiplier = np.linspace(1.0, 2.0, int(mask.sum()))
        for column in ("open", "high", "low", "close"):
            frame.loc[mask, column] *= multiplier
    changed = backtest_peer_residual_reversal(
        revised,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    pd.testing.assert_frame_equal(
        baseline.executed_weights.loc[:cutoff],
        changed.executed_weights.loc[:cutoff],
    )
