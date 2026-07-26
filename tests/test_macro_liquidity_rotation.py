from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.macro_liquidity_rotation import (
    MacroLiquidityParameters,
    backtest_macro_liquidity_rotation,
    build_macro_liquidity_votes,
    macro_liquidity_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy


def _frames(rows: int = 1_100) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2021-01-01", periods=rows, freq="D", tz="UTC")
    generator = np.random.default_rng(1407)
    result: dict[str, pd.DataFrame] = {}
    for offset, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    ):
        returns = generator.normal(0.0008 + offset * 0.00005, 0.008, rows)
        close = (100.0 + offset * 10.0) * np.exp(np.cumsum(returns))
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


def _macro(rows: int = 1_100) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp("2021-01-01", tz="UTC")
    definitions = {
        "WALCL": (pd.date_range(start, periods=158, freq="7D"), 100.0, 1.0),
        "M2SL": (pd.date_range(start, periods=37, freq="30D"), 200.0, 2.0),
        "NFCI": (pd.date_range(start, periods=158, freq="7D"), 1.0, -0.01),
    }
    output: dict[str, pd.DataFrame] = {}
    for series_id, (dates, initial, step) in definitions.items():
        available = dates + pd.offsets.Day(1)
        output[series_id] = pd.DataFrame(
            {
                "source_symbol": series_id,
                "observation_time": dates,
                "available_at": available,
                "point_in_time_status": "SOURCE_AVAILABLE_AT",
                "value": initial + np.arange(len(dates)) * step,
            }
        )
    return output


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=200,
    )


def test_macro_liquidity_family_is_exactly_two_unique_dna() -> None:
    rows = macro_liquidity_parameter_set()
    assert len(rows) == 2
    assert len({row.dna_hash for row in rows}) == 2
    assert {row.minimum_positive_votes for row in rows} == {2, 3}


def test_macro_votes_reject_forward_only_sources() -> None:
    macro = _macro()
    macro["WALCL"]["point_in_time_status"] = "FORWARD_ONLY"
    with pytest.raises(ValueError, match="SOURCE_AVAILABLE_AT"):
        build_macro_liquidity_votes(
            macro,
            target_index=_frames()["BTC-EUR"].index,
            parameters=MacroLiquidityParameters(minimum_positive_votes=2),
        )


def test_macro_liquidity_is_causal_bounded_and_orderless() -> None:
    result = backtest_macro_liquidity_rotation(
        _frames(),
        _macro(),
        MacroLiquidityParameters(minimum_positive_votes=2),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    assert result.signal_diagnostics["macro_risk_on_days"] > 0
    assert result.integrity["source_available_at_only"]
    assert result.integrity["macro_alignment_backward_only"]
    assert result.integrity["decision_at_close_execution_next_open"]
    assert result.integrity["maximum_exposure_respected"]
    assert result.integrity["minimum_cash_respected"]
    assert result.integrity["orders_generated"] == 0
    assert result.metrics["maximum_realized_exposure"] <= 0.40


def test_macro_liquidity_cost_stress_is_monotonic() -> None:
    frames = _frames()
    macro = _macro()
    parameters = MacroLiquidityParameters(minimum_positive_votes=3)
    normal = backtest_macro_liquidity_rotation(
        frames,
        macro,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    stressed = backtest_macro_liquidity_rotation(
        frames,
        macro,
        parameters,
        fee_rate=0.005,
        slippage_bps=16.0,
        spread_bps=10.0,
        portfolio_policy=_policy(),
    )
    assert stressed.metrics["net_return"] <= normal.metrics["net_return"]
    pd.testing.assert_frame_equal(
        normal.executed_weights,
        stressed.executed_weights,
    )


def test_future_macro_releases_do_not_change_prior_allocations() -> None:
    frames = _frames()
    macro = _macro()
    parameters = MacroLiquidityParameters(minimum_positive_votes=2)
    cutoff = frames["BTC-EUR"].index[800]
    baseline = backtest_macro_liquidity_rotation(
        frames,
        macro,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=_policy(),
    )
    changed_macro = {key: value.copy() for key, value in macro.items()}
    for frame in changed_macro.values():
        mask = pd.to_datetime(frame["available_at"], utc=True) > cutoff
        frame.loc[mask, "value"] = (
            pd.to_numeric(frame.loc[mask, "value"]) * -100.0
        )
    changed = backtest_macro_liquidity_rotation(
        frames,
        changed_macro,
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
