from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.absolute_momentum import (
    AbsoluteMomentumParameters,
    absolute_momentum_parameter_set,
    backtest_absolute_momentum,
)
from research.forward_observer import (
    ForwardHistoryRevisionError,
    build_rotation_forward_evidence,
    merge_portfolio_forward_manifest,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 620) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2021-01-01", periods=rows, freq="1D", tz="UTC")

    def frame(drift: float, phase: float) -> pd.DataFrame:
        returns = (
            drift
            + 0.006 * np.sin(np.arange(rows) / 17.0 + phase)
            + 0.002 * np.cos(np.arange(rows) / 7.0 + phase)
        )
        values = 100.0 * np.exp(np.cumsum(returns))
        return pd.DataFrame(
            {
                "open": values,
                "high": values * 1.01,
                "low": values * 0.99,
                "close": values,
                "volume": 1_000.0,
            },
            index=index,
        )

    return {
        "BTC-EUR": frame(0.0012, 0.0),
        "ETH-EUR": frame(0.0014, 0.6),
        "SOL-EUR": frame(0.0010, 1.2),
        "LINK-EUR": frame(0.0009, 1.8),
    }


def _policy(markets: tuple[str, ...]) -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=0.20,
        maximum_position_exposure=0.20,
        minimum_cash=0.80,
        minimum_history_observations=90,
    )


def _parameters(**updates: object) -> AbsoluteMomentumParameters:
    values = {
        "momentum_lookbacks": (20, 60, 120),
        "minimum_positive_horizons": 2,
        "asset_ema_period": 200,
        "btc_ema_period": 200,
        "volatility_lookback": 60,
        "target_annualized_volatility": 0.05,
    }
    values.update(updates)
    return AbsoluteMomentumParameters(**values)


def test_absolute_momentum_parameter_set_is_fixed_and_unique() -> None:
    rows = absolute_momentum_parameter_set()

    assert len(rows) == 5
    assert len({row.dna_hash for row in rows}) == len(rows)
    assert {row.target_annualized_volatility for row in rows} == {
        0.04,
        0.05,
        0.06,
        0.08,
        0.10,
    }


def test_absolute_momentum_is_next_open_costed_and_exposure_bounded() -> None:
    frames = _frames()
    result = backtest_absolute_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )

    assert result.metrics["rebalance_count"] > 0
    assert result.metrics["maximum_realized_exposure"] <= 0.20 + 1e-12
    assert result.integrity["maximum_position_exposure_respected"]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["closed_candles_only"]
    assert result.integrity["next_open_execution"]
    assert result.integrity["orders_generated"] == 0
    assert result.cost_breakdown["total_cost_drag"] > 0.0


def test_future_change_cannot_alter_prior_weights_or_equity() -> None:
    frames = _frames()
    baseline = backtest_absolute_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    shocked = {market: frame.copy() for market, frame in frames.items()}
    shocked["SOL-EUR"].iloc[-1, shocked["SOL-EUR"].columns.get_loc("close")] *= 5.0
    changed = backtest_absolute_momentum(
        shocked,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )

    cutoff = baseline.equity_curve.index[-2]
    pd.testing.assert_series_equal(
        baseline.equity_curve.loc[:cutoff],
        changed.equity_curve.loc[:cutoff],
    )
    pd.testing.assert_frame_equal(
        baseline.executed_weights.iloc[:-1],
        changed.executed_weights.iloc[:-1],
    )


def test_cost_stress_is_monotonic_and_unknown_asset_fails_closed() -> None:
    frames = _frames()
    free = backtest_absolute_momentum(
        frames,
        _parameters(),
        fee_rate=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    stressed = backtest_absolute_momentum(
        frames,
        _parameters(),
        fee_rate=0.005,
        slippage_bps=10.0,
        spread_bps=10.0,
        portfolio_policy=_policy(tuple(frames)),
    )

    assert stressed.metrics["net_return"] < free.metrics["net_return"]
    with pytest.raises(ValueError, match="outside fail-closed"):
        backtest_absolute_momentum(
            frames | {"TAO-EUR": frames["ETH-EUR"]},
            _parameters(),
            fee_rate=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            portfolio_policy=_policy(tuple(frames)),
        )


def test_forward_observer_is_orderless_append_only_and_hash_chained() -> None:
    frames = _frames()
    result = backtest_absolute_momentum(
        frames,
        _parameters(),
        fee_rate=0.0025,
        slippage_bps=5.0,
        spread_bps=5.0,
        portfolio_policy=_policy(tuple(frames)),
    )
    forward_start = "2022-01-01T00:00:00+00:00"
    evidence = build_rotation_forward_evidence(
        result,
        frames,
        forward_start=forward_start,
        minimum_observations=365,
        minimum_rebalances=30,
    )
    manifest = merge_portfolio_forward_manifest(
        {
            "status": "FROZEN_FORWARD_RESEARCH",
            "source_candidate_identity": "absolute-momentum-test",
            "strategy_dna_hash": result.parameters.dna_hash,
            "execution_identity": result.summary()["execution_identity"],
            "forward_start": forward_start,
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        evidence,
        source_candidate_identity="absolute-momentum-test",
        strategy_dna_hash=result.parameters.dna_hash,
        execution_identity=result.summary()["execution_identity"],
        forward_start=forward_start,
    )

    assert manifest["forward_summary"]["closed_daily_observations"] > 0
    assert manifest["forward_hash_chain"]["record_count"] == len(
        manifest["forward_observations"]
    )
    assert manifest["orders_generated"] == 0
    assert manifest["orders_submitted"] == 0
    assert not manifest["paper_candidate_permitted"]
    assert not manifest["live_ready"]

    corrupted = deepcopy(manifest)
    corrupted["forward_observations"][0]["net_return"] += 0.01
    with pytest.raises(
        ForwardHistoryRevisionError,
        match="CHECKSUM|checksum|REVISION",
    ):
        merge_portfolio_forward_manifest(
            corrupted,
            evidence,
            source_candidate_identity="absolute-momentum-test",
            strategy_dna_hash=result.parameters.dna_hash,
            execution_identity=result.summary()["execution_identity"],
            forward_start=forward_start,
        )
