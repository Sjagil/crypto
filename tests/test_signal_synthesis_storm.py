from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signal_synthesis_storm import (
    SIGNAL_STORM_CONTEXT_BLOCKERS,
    SIGNAL_STORM_MARKETS,
    SIGNAL_STORM_TRIAL_COUNT,
    SignalSynthesisDNA,
    preregistered_signal_dna,
    run_signal_synthesis_storm,
    signal_storm_plan,
)


def _feature_frames() -> dict[str, dict[str, pd.DataFrame]]:
    periods = {"1h": 14_400, "4h": 3_600, "1d": 600}
    frequencies = {"1h": "1h", "4h": "4h", "1d": "1D"}
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for timeframe, rows in periods.items():
        index = pd.date_range(
            "2023-01-01",
            periods=rows,
            freq=frequencies[timeframe],
            tz="UTC",
        )
        output[timeframe] = {}
        for offset, market in enumerate(SIGNAL_STORM_MARKETS):
            values = np.arange(rows, dtype=float)
            change = (
                0.00015
                + 0.0015 * np.sin(values / (31.0 + offset))
                + 0.0005 * np.cos(values / 11.0 + offset)
            )
            close = (100.0 + 5.0 * offset) * np.exp(np.cumsum(change))
            open_price = np.r_[close[0], close[:-1]]
            ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
            frame = pd.DataFrame(
                {
                    "open": open_price,
                    "high": np.maximum(open_price, close) * 1.006,
                    "low": np.minimum(open_price, close) * 0.994,
                    "close": close,
                    "volume": 10_000.0,
                    "atr_14": close * 0.012,
                    "ema_20": ema_20,
                    "roc_12": pd.Series(close).pct_change(12).fillna(0.0),
                    "btc_relative_momentum_20": (0.01 * np.sin(values / 23.0 + offset)),
                    "bearish_fvg": False,
                },
                index=index,
            )
            frame.attrs.update(market=market, timeframe=timeframe)
            output[timeframe][market] = frame
    return output


def _small_dna(count: int = 24) -> tuple[SignalSynthesisDNA, ...]:
    rows = []
    profiles = ("FIXED_R", "TRAILING_TREND", "TIME_REGIME")
    modes = ("LAYERED", "ALL", "MAJORITY", "WEIGHTED_VOTE")
    pairs = (
        ("BTC-EUR", "ETH-EUR"),
        ("BTC-EUR", "SOL-EUR"),
        ("BTC-EUR", "LINK-EUR"),
        ("ETH-EUR", "SOL-EUR"),
        ("ETH-EUR", "LINK-EUR"),
        ("SOL-EUR", "LINK-EUR"),
    )
    for index in range(count):
        confirmation = "btc_relative_momentum" if index % 2 == 0 else None
        avoidance = "bearish_fvg" if index % 3 == 0 else None
        block_parameters = {
            "positive_return_20": {},
            "price_above_ema20": {},
            "negative_return_exit": {},
        }
        if confirmation:
            block_parameters[confirmation] = {}
        if avoidance:
            block_parameters[avoidance] = {}
        rows.append(
            SignalSynthesisDNA(
                entry_block="positive_return_20",
                context_block="price_above_ema20",
                confirmation_block=confirmation,
                avoidance_block=avoidance,
                exit_block="negative_return_exit",
                overlay_block=None,
                timeframe=("1h", "4h", "1d")[index % 3],
                asset_pair=pairs[index % len(pairs)],
                logic_mode=modes[index % len(modes)],
                vote_threshold=(0.5, 0.6, 0.7)[index % 3],
                exit_profile=profiles[index % len(profiles)],
                stop_atr=(1.5, 2.0, 3.0)[index % 3],
                target_atr=(2.0, 3.0, 6.0)[index % 3],
                trailing_atr=(1.5, 2.5, 4.0)[index % 3],
                maximum_holding_bars=48 + index,
                block_parameters=block_parameters,
            )
        )
    return tuple(rows)


def test_signal_storm_plan_is_deterministic_unique_and_covers_registry():
    first = preregistered_signal_dna(trial_count=200, seed=17)
    second = preregistered_signal_dna(trial_count=200, seed=17)
    assert [row.dna_hash for row in first] == [row.dna_hash for row in second]
    assert len({row.dna_hash for row in first}) == 200
    assert all(row.maximum_total_exposure <= 0.40 for row in first)
    assert all(row.maximum_position_exposure <= 0.20 for row in first)
    assert all(row.minimum_cash >= 0.60 for row in first)

    plan = signal_storm_plan()
    assert plan["trial_count"] == SIGNAL_STORM_TRIAL_COUNT
    assert plan["registered_signal_blocks"] == 134
    assert plan["executable_signal_blocks"] == 133
    assert plan["covered_executable_blocks"] == 133
    assert plan["blocked_signal_blocks"] == dict(SIGNAL_STORM_CONTEXT_BLOCKERS)
    assert len(plan["families_covered"]) == 11


def test_signal_storm_rejects_risk_limit_relaxation():
    values = _small_dna(2)[0].to_dict()
    values["maximum_total_exposure"] = 0.80
    values["minimum_cash"] = 0.20
    with pytest.raises(ValueError, match="strict limit"):
        SignalSynthesisDNA.from_dict(values)


def test_small_signal_storm_uses_all_paths_and_never_promotes():
    dna = _small_dna()
    report, matrix, timestamps = run_signal_synthesis_storm(
        _feature_frames(),
        dna,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        prior_known_trials=6_312,
        bootstrap_samples=100,
        maximum_survivors=8,
        batch_size=8,
    )

    assert report["trial_count"] == len(dna)
    assert report["total_known_trials"] == 6_312 + len(dna)
    assert report["selection_basis"] == "DEVELOPMENT_ONLY"
    assert report["pareto_survivor_count"] <= 8
    assert matrix.shape == (len(timestamps), len(dna))
    assert np.isfinite(matrix).all()
    assert report["multiple_testing"]["strategy_count"] == len(dna)
    assert report["multiple_testing"]["dsr_total_trial_denominator"] == (6_312 + len(dna))
    assert report["multiple_testing"]["white_spa_status"] == (
        "FORMALLY_EVALUATED_ALL_SIGNAL_STORM_TRIALS"
    )
    assert report["development_screen"]["validation_or_confirmation_used"] is False
    for survivor in report["pareto_survivors"]:
        assert survivor["development"]["portfolio_period_profit_factor"] > 1.0
    assert report["screening_semantics"]["next_open_execution"] is True
    assert report["research_pass"] is False
    assert report["paper_candidates"] == 0
    assert report["orders_generated"] == 0
    assert report["live_ready"] is False
