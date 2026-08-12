from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from config.settings import HMMRegimeSettings
from core.cli import build_parser
from research.hmm_regime_campaign import (
    HMM_POLICY_MATRIX,
    RR_STRATEGY_DNA,
)
from research.hmm_regime_manager import (
    FEATURE_COLUMNS,
    InstitutionalHMMRegimeManager,
    causal_hmm_features,
    causal_market_context,
)


def _settings(**overrides: object) -> HMMRegimeSettings:
    return HMMRegimeSettings(
        maximum_iterations=50,
        maximum_training_observations=250,
        refit_interval_1w=13,
        **overrides,
    )


def _features(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(20260727)
    index = pd.date_range("2023-01-01", periods=rows, freq="7D", tz="UTC")
    regime = np.repeat((-1.0, 0.0, 1.0, 0.25), rows // 4)
    if len(regime) < rows:
        regime = np.pad(regime, (0, rows - len(regime)), mode="edge")
    return pd.DataFrame(
        {
            "log_return": 0.01 * regime + rng.normal(0.0, 0.006, rows),
            "realized_volatility": 0.02 + 0.01 * abs(regime)
            + rng.normal(0.0, 0.001, rows),
            "atr_fraction": 0.025 + 0.007 * abs(regime)
            + rng.normal(0.0, 0.001, rows),
            "trend_slope": 0.004 * regime + rng.normal(0.0, 0.001, rows),
            "range_efficiency": np.clip(
                0.35 + 0.2 * abs(regime) + rng.normal(0.0, 0.03, rows),
                0.01,
                0.99,
            ),
            "volume_zscore": regime + rng.normal(0.0, 0.2, rows),
            "breadth": np.clip(0.5 + 0.25 * regime, 0.01, 0.99),
            "average_cross_asset_correlation": np.clip(
                0.4 + 0.15 * abs(regime),
                -1.0,
                1.0,
            ),
        },
        index=index,
    )


def test_hmm_walk_forward_is_invariant_to_future_mutation() -> None:
    manager = InstitutionalHMMRegimeManager(_settings())
    original = _features()
    mutated = original.copy()
    cutoff = original.index[130]
    mutated.loc[mutated.index > cutoff, :] += 500.0

    first = manager.walk_forward(original, timeframe="1W")
    second = manager.walk_forward(mutated, timeframe="1W")

    assert_frame_equal(
        first.probabilities.loc[:cutoff],
        second.probabilities.loc[:cutoff],
        check_exact=True,
    )
    assert first.integrity["filtered_not_smoothed"] is True
    assert first.integrity["strictly_prior_model_fit"] is True
    assert all(
        pd.Timestamp(row["fitted_through"])
        < pd.Timestamp(row["inference_starts_at"])
        for row in first.fit_history
    )


def test_hmm_probabilities_forecasts_and_durations_are_valid() -> None:
    result = InstitutionalHMMRegimeManager(_settings()).walk_forward(
        _features(),
        timeframe="1W",
    )
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.posterior_entropy.between(0.0, 1.0 + 1e-12).all()
    assert result.risk_multiplier.between(0.0, 1.0).all()
    assert all(value >= 1.0 for value in result.expected_duration.values())
    for forecast in result.current_forecasts.values():
        assert sum(forecast.values()) == pytest.approx(1.0)


def test_hmm_backward_asof_never_uses_future_close() -> None:
    manager = InstitutionalHMMRegimeManager(_settings())
    inference = manager.walk_forward(_features(), timeframe="1W")
    target = pd.date_range(
        inference.probabilities.index[0],
        inference.probabilities.index[-1],
        freq="1D",
        tz="UTC",
    )
    aligned = manager.backward_asof_align(
        target,
        {"1W": inference},
        target_timeframe="1d",
    )
    available = aligned["1W_available_at"].dropna()
    assert (
        available
        <= aligned.loc[available.index, "decision_available_at"]
    ).all()


def test_hmm_execution_timestamp_does_not_add_target_interval() -> None:
    manager = InstitutionalHMMRegimeManager(_settings())
    inference = manager.walk_forward(_features(), timeframe="1W")
    target = inference.probabilities.index[-10:]
    aligned = manager.backward_asof_align(
        target,
        {"1W": inference},
        target_timeframe="1W",
        target_is_available_at=True,
    )
    expected = pd.Series(
        target,
        index=target,
        name="decision_available_at",
    )
    assert aligned["decision_available_at"].equals(expected)
    available = aligned["1W_available_at"].dropna()
    assert (available <= available.index).all()


def test_hmm_weight_scaling_preserves_40_20_60_limits() -> None:
    manager = InstitutionalHMMRegimeManager(_settings())
    index = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    targets = pd.DataFrame(
        {
            "BTC-EUR": [0.30, 0.20, 0.10],
            "ETH-EUR": [0.30, 0.20, 0.10],
            "SOL-EUR": [0.30, 0.20, 0.10],
        },
        index=index,
    )
    scaled = manager.apply_risk_multiplier(
        targets,
        pd.Series([1.0, 0.5, 0.0], index=index),
    )
    assert float(scaled.max(axis=1).max()) <= 0.20
    assert float(scaled.sum(axis=1).max()) <= 0.40
    assert float((1.0 - scaled.sum(axis=1)).min()) >= 0.60


def test_hmm_features_are_backward_only() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=90, freq="1D", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(8.0, 0.4, len(index)),
        },
        index=index,
    )
    cutoff = index[70]
    mutated = frame.copy()
    mutated.loc[mutated.index > cutoff, "close"] *= 20.0
    first = causal_hmm_features(frame)
    second = causal_hmm_features(mutated)
    assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff], check_exact=True)


def test_hmm_context_handles_different_native_weekly_boundaries() -> None:
    thursday = pd.date_range("2025-01-02", periods=60, freq="7D", tz="UTC")
    monday = pd.date_range("2025-01-06", periods=60, freq="7D", tz="UTC")

    def frame(index: pd.DatetimeIndex, slope: float) -> pd.DataFrame:
        close = 100.0 * np.exp(np.arange(len(index)) * slope)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000.0,
            },
            index=index,
        )

    breadth, correlation = causal_market_context(
        {
            "BTC-EUR": frame(thursday, 0.01),
            "ETH-EUR": frame(thursday, 0.008),
            "LINK-EUR": frame(monday, 0.006),
        }
    )
    assert breadth.reindex(thursday).dropna().index[-1] == thursday[-1]
    assert correlation.reindex(thursday).dropna().index[-1] == thursday[-1]


def test_hmm_settings_cannot_enable_execution_authority() -> None:
    with pytest.raises(ValidationError, match="observer-only"):
        HMMRegimeSettings(observer_only=False)


def test_hmm_matrix_is_closed_and_rr_identity_is_frozen() -> None:
    assert len(HMM_POLICY_MATRIX) == 3
    assert len({row["policy_id"] for row in HMM_POLICY_MATRIX}) == 3
    assert RR_STRATEGY_DNA == (
        "4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0"
    )
    assert all(column in FEATURE_COLUMNS for column in FEATURE_COLUMNS)


def test_hmm_cli_is_reachable_through_main_parser() -> None:
    args = build_parser().parse_args(["hmm", "status"])
    assert args.command == "hmm"
    assert args.hmm_command == "status"
