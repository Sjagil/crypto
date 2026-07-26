from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.statistical_evidence import (
    conservative_dsr_audit,
    deduplicated_equity_returns,
    exposure_matched_equal_weight_equity,
    hac_effective_sample_size,
    pnl_concentration_audit,
    unique_return_path_pbo,
)


def test_hac_ess_and_dsr_variants_are_conservative() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=240,
        freq="D",
        tz="UTC",
    )
    wave = np.sin(np.arange(len(index)) / 8.0) * 0.002
    candidate = pd.Series(0.0005 + wave, index=index)
    matrix = pd.DataFrame(
        {
            "candidate": candidate,
            "neighbor_a": candidate * 0.8
            + np.cos(np.arange(len(index))) * 0.0005,
            "neighbor_b": -candidate * 0.2
            + np.sin(np.arange(len(index)) / 3) * 0.001,
            "duplicate": candidate,
        },
        index=index,
    )
    sample = hac_effective_sample_size(candidate)
    assert sample["effective_sample_size"] < len(candidate)
    audit = conservative_dsr_audit(
        candidate,
        matrix,
        total_trials=100,
    )
    assert set(audit["probabilities"]) == {
        "daily_raw",
        "daily_hac",
        "weekly_raw",
        "weekly_ess",
    }
    assert audit["formal_probability"] == min(
        audit["probabilities"].values()
    )
    assert audit["total_historical_trials"] == 100


def test_unique_path_pbo_deduplicates_exact_returns() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=80,
        freq="D",
        tz="UTC",
    )
    base = pd.Series(
        np.sin(np.arange(80) / 4.0) * 0.01,
        index=index,
    )
    matrix = pd.DataFrame(
        {
            "a": base,
            "a_retry": base,
            "b": base.shift(1).fillna(0.0),
            "c": -base,
        }
    )
    audit = unique_return_path_pbo(matrix, group_count=8)
    assert audit["nominal_dna_count"] == 4
    assert audit["unique_return_path_count"] == 3
    assert audit["duplicate_return_path_count"] == 1
    assert audit["tie_handling"] == "MIDRANK_HALF_WEIGHT"
    assert audit["formal_worst_valid_pbo"] is not None


def test_terminal_duplicate_is_combined_before_returns() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-02T00:00:00Z",
        ]
    )
    equity = pd.Series([1.0, 1.10, 1.089], index=index)
    returns = deduplicated_equity_returns(equity)
    assert len(returns) == 1
    assert np.isclose(float(returns.iloc[0]), 0.089)


def test_exposure_matched_benchmark_uses_varying_exposure() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=8,
        freq="D",
        tz="UTC",
    )
    frames = {
        market: pd.DataFrame(
            {
                "open": 100.0
                * np.cumprod(np.r_[1.0, np.repeat(growth, 7)]),
                "close": 100.0
                * np.cumprod(np.r_[1.0, np.repeat(growth, 7)]),
            },
            index=index,
        )
        for market, growth in (
            ("BTC-EUR", 1.01),
            ("ETH-EUR", 1.03),
        )
    }
    equity_index = index[2:]
    result = SimpleNamespace(
        portfolio_policy=SimpleNamespace(
            allowed_markets=("BTC-EUR", "ETH-EUR"),
            minimum_history_observations=2,
        ),
        equity_curve=pd.Series(
            np.ones(len(equity_index)),
            index=equity_index,
        ),
        executed_weights=pd.DataFrame(
            {
                "BTC-EUR": [0.10, 0.10, 0.20, 0.20, 0.0, 0.0],
                "ETH-EUR": [0.10, 0.10, 0.20, 0.20, 0.0, 0.0],
            },
            index=equity_index,
        ),
    )
    benchmark = exposure_matched_equal_weight_equity(
        result,
        frames,
        one_way_cost=0.0,
    )
    returns = deduplicated_equity_returns(benchmark)
    assert len(returns) == len(equity_index) - 1
    assert float(returns.max()) > float(returns.min())
    assert benchmark.index.equals(equity_index)


def test_pnl_concentration_reports_all_required_dimensions() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=20,
        freq="D",
        tz="UTC",
    )
    equity = pd.Series(
        np.cumprod(1.0 + np.linspace(-0.002, 0.004, 20)),
        index=index,
    )
    decisions = pd.DataFrame(
        {
            "reason": ["RANKED_MOMENTUM", "RANKED_MOMENTUM"],
            "executed_at": [index[0], index[10]],
            "btc_uptrend": [True, False],
            "volatility_state": ["LOW", "HIGH"],
            "breadth_state": ["BROAD", "NARROW"],
        }
    )
    episodes = pd.DataFrame(
        {
            "weighted_pnl": [0.10, 0.05, -0.02, 0.03, 0.02, 0.01],
        }
    )
    result = SimpleNamespace(
        equity_curve=equity,
        decisions=decisions,
        position_episodes=episodes,
        metrics={
            "asset_pnl_attribution": {
                "BTC-EUR": {"net_pnl_amount": 0.10},
                "ETH-EUR": {"net_pnl_amount": 0.05},
                "SOL-EUR": {"net_pnl_amount": -0.01},
            }
        },
    )
    audit = pnl_concentration_audit(result)
    assert set(audit) >= {
        "asset",
        "year",
        "regime",
        "trades",
    }
    assert audit["asset"]["largest_positive_source"] == "BTC-EUR"
    assert audit["asset"]["effective_positive_sources"] > 1
    assert (
        audit["trades"]["top_five_positive_trade_pnl_share"]
        is not None
    )
