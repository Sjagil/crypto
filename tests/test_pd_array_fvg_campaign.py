from __future__ import annotations

import numpy as np
import pandas as pd

from core.cli import build_parser
from research.pd_array_fvg_campaign import (
    PdArrayFvgParameters,
    _formation_table,
    _normalized_ohlcv,
    backtest_pd_array_fvg,
    pd_array_fvg_parameter_set,
)


def _frame(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    phase = np.linspace(0.0, 14.0, rows)
    close = 100.0 + np.sin(phase) * 5.0 + np.linspace(0.0, 2.0, rows)
    return pd.DataFrame(
        {
            "open": close - 0.15,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.linspace(0.0, 1_000.0, rows),
        },
        index=index,
    )


def test_pd_array_fvg_dna_is_small_frozen_and_unique() -> None:
    candidates = pd_array_fvg_parameter_set()

    assert len(candidates) == 12
    assert len({candidate.dna_hash for candidate in candidates}) == 12
    assert {candidate.timeframe for candidate in candidates} == {"15m", "1h", "4h"}
    assert {candidate.signal_mode for candidate in candidates} == {"SWEEP", "SMT"}


def test_pd_array_fvg_formation_is_backward_only() -> None:
    full = _normalized_ohlcv(_frame())
    benchmark = _normalized_ohlcv(_frame().assign(low=lambda row: row["low"] * 0.99))
    parameters = PdArrayFvgParameters("1h", "SMT", 0.50)
    cutoff = 130

    short, short_diagnostics = _formation_table(
        full.iloc[:cutoff], benchmark.iloc[:cutoff], parameters
    )
    long, _ = _formation_table(full, benchmark, parameters)

    pd.testing.assert_frame_equal(short, long.iloc[:cutoff])
    assert short_diagnostics["valid_fvg_entry_setups"] == int(
        short["formation"].sum()
    )


def test_pd_array_fvg_waits_for_future_retrace_and_models_costs() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [101.0, 102.0, 101.0, 111.0, 101.0, 101.0],
            "low": [99.0, 101.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 101.5, 100.0, 110.0, 100.0, 100.0],
            "volume": [0.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=index,
    )
    formation = pd.DataFrame(
        {
            "formation": [False, True, False, False, False, False],
            "entry": [np.nan, 100.0, np.nan, np.nan, np.nan, np.nan],
            "stop": [np.nan, 95.0, np.nan, np.nan, np.nan, np.nan],
            "target": [np.nan, 110.0, np.nan, np.nan, np.nan, np.nan],
            "reward_risk": [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )
    prepared = (formation, {"valid_fvg_entry_setups": 1})
    parameters = PdArrayFvgParameters("1h", "SWEEP", 0.50)
    common = {
        "frame": data,
        "benchmark_frame": data,
        "parameters": parameters,
        "prepared_data": data,
        "prepared_benchmark": data,
        "prepared_formation": prepared,
        "slippage_bps": 0.0,
        "spread_bps": 0.0,
    }

    normal = backtest_pd_array_fvg(**common, fee_rate=0.0)
    stressed = backtest_pd_array_fvg(**common, fee_rate=0.01)

    assert normal["trades"][0]["entry_at"] == index[2].isoformat()
    assert normal["trades"][0]["exit_reason"] == "SWING_HIGH_TARGET"
    assert normal["metrics"]["net_total_return"] > stressed["metrics"]["net_total_return"]
    assert normal["integrity"]["orders_generated"] == 0


def test_pd_array_fvg_campaign_is_available_from_canonical_cli() -> None:
    args = build_parser().parse_args(
        ["lab", "campaign", "plan", "--name", "pd-array-fvg-v1"]
    )

    assert args.name == "pd-array-fvg-v1"
