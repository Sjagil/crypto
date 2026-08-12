from __future__ import annotations

from dataclasses import asdict

import pytest

from core.economics import CanonicalCostModel
from research.backtest import CostModel
from research.research_factory import SharedCostModel


def test_research_cost_model_is_canonical_owner() -> None:
    assert SharedCostModel is CanonicalCostModel
    costs = CanonicalCostModel.create(
        maker_fee_fraction=0.0015,
        taker_fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=8.0,
        failed_execution_allowance_bps=0.0,
        partial_fill_impact_bps=0.0,
    )
    assert costs.cost_model_version.startswith("canonical_cost_v1:")
    assert CanonicalCostModel.create(
        **{
            key: value
            for key, value in asdict(costs).items()
            if key != "cost_model_version"
        }
    ).cost_model_version == costs.cost_model_version


def test_exact_backtester_adapts_canonical_values() -> None:
    costs = CanonicalCostModel.create(
        maker_fee_fraction=0.0015,
        taker_fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=8.0,
    )
    adapted = CostModel.from_canonical(costs, multiplier=1.25)
    assert adapted.fee_fraction == costs.taker_fee_fraction
    assert adapted.spread_bps == costs.spread_bps
    assert adapted.slippage_bps == costs.slippage_bps
    assert adapted.multiplier == 1.25


def test_canonical_cost_stress_is_monotonic_and_versioned() -> None:
    costs = CanonicalCostModel.create(
        maker_fee_fraction=0.0015,
        taker_fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=8.0,
    )
    stressed = costs.stressed(
        additional_roundtrip_bps=10,
        spread_multiplier=1.5,
        slippage_multiplier=2.0,
    )
    assert stressed.spread_bps > costs.spread_bps
    assert stressed.slippage_bps > costs.slippage_bps
    assert stressed.cost_model_version != costs.cost_model_version
    with pytest.raises(ValueError, match="below one"):
        costs.stressed(spread_multiplier=0.9)
