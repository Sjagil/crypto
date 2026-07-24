from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from core.contracts import ResearchStatus
from research.backtest import BacktestConfig, BacktestEngine
from research.optimization import (
    StabilityResult,
    WalkForwardResult,
    acceptance_gate,
)
from research.strategies import Strategy, StrategyOutput
from research.trading_math import (
    bootstrap_expectancy,
    calculate_position_size,
    empirical_risk_of_ruin,
    expectancy,
)


class FixedStrategy(Strategy):
    strategy_id = "fixed"
    family = "test"
    description = "Deterministic test signals."
    defaults = {"stop_atr": 1.0, "target_atr": 1.0, "trailing_atr": 0.0}
    parameter_space = {key: (value,) for key, value in defaults.items()}

    def generate(self, features, parameters=None):
        index = features.index
        entry = pd.Series(False, index=index)
        exit_ = pd.Series(False, index=index)
        entry.iloc[250] = True
        exit_.iloc[255] = True
        ones = pd.Series(100.0, index=index)
        return StrategyOutput(
            entry=entry,
            exit=exit_,
            avoid=pd.Series(False, index=index),
            reduce=pd.Series(False, index=index),
            stop_distance=ones,
            target_distance=ones * 20,
            trailing_distance=pd.Series(0.0, index=index),
            size_multiplier=pd.Series(1.0, index=index),
            maximum_holding_bars=None,
            entry_reason="UNIT_ENTRY",
            exit_reason="UNIT_EXIT",
        ).validate(index)


def test_expectancy_and_cost_aware_position_size() -> None:
    result = expectancy(0.5, 2.0, 1.0, cost_r=0.1)
    assert result.net_expectancy_r == 0.4
    size = calculate_position_size(
        10_000,
        0.01,
        100,
        95,
        fee_fraction_per_side=0.0025,
        slippage_fraction_per_side=0.001,
    )
    assert 0 < size.units
    assert size.actual_risk <= 100 + 1e-8
    bootstrap = bootstrap_expectancy(
        [1.0, -1.0, 2.0, -0.5, 1.2, -1.0],
        bootstrap_samples=200,
        block_size=2,
        seed=42,
    )
    simulation = empirical_risk_of_ruin(
        [1.0, -1.0, 2.0, -0.5, 1.2, -1.0],
        simulations=200,
        trades_per_simulation=20,
        block_size=2,
        seed=42,
    )
    assert bootstrap.bootstrap_samples == 200
    assert simulation.simulations == 200


def test_backtest_executes_at_next_open_and_charges_costs(features: pd.DataFrame) -> None:
    result = BacktestEngine(
        BacktestConfig(monte_carlo_runs=100),
    ).run({"BTC-EUR": features}, FixedStrategy())
    buy = next(order for order in result.orders if order.side == "BUY" and order.status == "FILLED")
    assert buy.executed_at == features.index[251].to_pydatetime()
    assert buy.signal_at == features.index[250].to_pydatetime()
    assert result.metrics["transaction_costs_eur"] > 0
    assert {
        "probability_of_10pct_drawdown",
        "probability_of_20pct_drawdown",
        "probability_of_30pct_drawdown",
        "probability_of_50pct_drawdown",
    } <= set(result.metrics)
    assert all(
        0.0 <= result.metrics[key] <= 1.0
        for key in (
            "probability_of_10pct_drawdown",
            "probability_of_20pct_drawdown",
            "probability_of_30pct_drawdown",
            "probability_of_50pct_drawdown",
        )
    )
    assert result.integrity["next_open_execution"]
    assert result.integrity["long_only_spot"]


def test_review_required_assets_need_explicit_research_only_scope(
    features: pd.DataFrame,
    isolated_settings,
) -> None:
    with pytest.raises(PermissionError, match="not ALLOWED"):
        BacktestEngine(
            BacktestConfig(bootstrap_samples=100, monte_carlo_runs=100),
            settings=isolated_settings,
        ).run({"BNB-EUR": features}, FixedStrategy())

    result = BacktestEngine(
        BacktestConfig(
            bootstrap_samples=100,
            monte_carlo_runs=100,
            allow_review_required_research_only=True,
        ),
        settings=isolated_settings,
    ).run({"BNB-EUR": features}, FixedStrategy())
    assert result.integrity["long_only_spot"]


def test_same_bar_stop_wins_and_acceptance_gate_rejects_low_sample(
    features: pd.DataFrame,
    isolated_settings,
) -> None:
    class SameBarStrategy(FixedStrategy):
        strategy_id = "same_bar"

        def generate(self, source, parameters=None):
            output = super().generate(source, parameters)
            return replace(
                output,
                target_distance=pd.Series(100.0, index=source.index),
            )

    source = features.copy()
    entry_open = float(source["open"].iloc[251])
    source.loc[source.index[251], "high"] = entry_open + 200
    source.loc[source.index[251], "low"] = entry_open - 200
    result = BacktestEngine(
        BacktestConfig(bootstrap_samples=100, monte_carlo_runs=100),
    ).run({"BTC-EUR": source}, SameBarStrategy())
    sell = next(order for order in result.orders if order.side == "SELL")
    assert sell.reason == "STOP_FIRST_SAME_BAR"
    gate = acceptance_gate(
        normal=result,
        stressed=result,
        holdout=result,
        walk_forward=WalkForwardResult(
            mode="anchored",
            folds=(),
            positive_folds=0,
            fold_profit_concentration=1.0,
            valid=False,
        ),
        stability=StabilityResult(
            stable=False,
            tested_neighbors=0,
            positive_neighbors=0,
            acceptable_score_fraction=0.0,
            neighbor_scores=(),
        ),
        research=isolated_settings.research,
        eligibility_valid=True,
        lookahead_safe=True,
        repainting_safe=True,
    )
    assert gate.status is ResearchStatus.REJECTED_INSUFFICIENT_TRADES


def test_backtest_detects_gaps_and_blocks_gap_entries(
    features: pd.DataFrame,
) -> None:
    source = features.drop(features.index[300]).copy()
    source.attrs.update(features.attrs)
    source.attrs["timeframe"] = "1h"
    result = BacktestEngine(
        BacktestConfig(bootstrap_samples=100, monte_carlo_runs=100),
    ).run({"BTC-EUR": source}, FixedStrategy())
    integrity = result.integrity["gap_integrity"]["BTC-EUR"]
    assert integrity["gap_events"] == 1
    assert integrity["missing_bars"] == 1
    assert result.integrity["gap_entry_blocking"]
