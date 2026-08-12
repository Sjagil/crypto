from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cli import build_parser
from research.hmm_duration_campaign import (
    PreparedFold,
    _apply_predictive_acceptance_gate,
    evaluate_duration_candidate,
)
from research.hmm_regime_manager import (
    ExplicitDurationHSMMFilter,
    HMMFitSnapshot,
    duration_hazard,
    shifted_poisson_duration_distribution,
)


def _snapshot() -> HMMFitSnapshot:
    return HMMFitSnapshot(
        timeframe="1d",
        fitted_through=pd.Timestamp("2025-01-31", tz="UTC"),
        training_started_at=pd.Timestamp("2025-01-01", tz="UTC"),
        feature_columns=("x",),
        center=np.array([0.0]),
        scale=np.array([1.0]),
        start_probability=np.array([0.8, 0.2]),
        transition_matrix=np.array([[0.90, 0.10], [0.15, 0.85]]),
        means=np.array([[0.0], [1.0]]),
        variances=np.array([[1.0], [1.0]]),
        state_labels=("CALM", "ACTIVE"),
        converged=True,
        iterations=5,
        model_hash="a" * 64,
    )


def _fold() -> PreparedFold:
    calm = np.array([0.0, -4.0])
    active = np.array([-4.0, 0.0])
    training = np.vstack([calm] * 12 + [active] * 8)
    validation = np.vstack(
        [calm] * 4
        + [active]
        + [calm] * 4
        + [active] * 5
        + [calm] * 4
    )
    return PreparedFold(
        timeframe="1d",
        fold=1,
        snapshot=_snapshot(),
        training_log_likelihoods=training,
        validation_log_likelihoods=validation,
        validation_index=pd.date_range(
            "2025-02-01",
            periods=len(validation),
            freq="1D",
            tz="UTC",
        ),
        base_negative_log_predictive_density=0.8,
        base_churn=0.25,
        base_switch_rate=0.2,
        base_mean_entropy=0.4,
        base_occupancy_entropy=0.8,
    )


def test_shifted_poisson_duration_is_proper_and_has_requested_mean() -> None:
    distribution = shifted_poisson_duration_distribution(14.0, 80)
    durations = np.arange(1, len(distribution) + 1, dtype=float)
    assert distribution.sum() == pytest.approx(1.0)
    assert float(distribution @ durations) == pytest.approx(14.0, abs=1e-8)
    assert bool((distribution >= 0.0).all())


def test_duration_hazard_is_bounded_and_exits_at_terminal_age() -> None:
    hazard = duration_hazard(
        shifted_poisson_duration_distribution(8.0, 40)
    )
    assert bool((hazard >= 0.0).all())
    assert bool((hazard <= 1.0).all())
    assert hazard[-1] == 1.0
    assert hazard[0] < hazard[10]


def test_explicit_duration_filter_tracks_state_and_age_probabilities() -> None:
    engine = ExplicitDurationHSMMFilter(
        start_probability=np.array([0.9, 0.1]),
        transition_matrix=np.array([[0.95, 0.05], [0.10, 0.90]]),
        expected_durations=np.array([12.0, 4.0]),
        maximum_duration=50,
    )
    for _ in range(6):
        step = engine.step(np.array([0.0, -6.0]))
    assert step.state_probability.sum() == pytest.approx(1.0)
    assert step.state_age_probability.sum() == pytest.approx(1.0)
    assert step.dominant_state == 0
    assert step.dominant_state_age >= 5


def test_explicit_duration_filter_does_not_force_a_switch_on_one_wick() -> None:
    engine = ExplicitDurationHSMMFilter(
        start_probability=np.array([0.99, 0.01]),
        transition_matrix=np.array([[0.90, 0.10], [0.10, 0.90]]),
        expected_durations=14.0,
        maximum_duration=60,
    )
    for _ in range(8):
        engine.step(np.array([0.0, -8.0]))
    wick = engine.step(np.array([-2.0, 0.0]))
    recovered = engine.step(np.array([0.0, -8.0]))
    assert wick.dominant_state == 0
    assert recovered.dominant_state == 0


def test_duration_candidate_is_deterministic_and_reports_collapse_guard() -> None:
    parameters = {
        "expected_dwell_1d": 10,
        "maximum_duration_factor": 4,
        "transition_shrinkage": 0.2,
    }
    first = evaluate_duration_candidate([_fold()], parameters)
    second = evaluate_duration_candidate([_fold()], parameters)
    assert first == second
    assert 0.0 <= first["churn"] <= 1.0
    assert 0.0 <= first["switch_rate"] <= 1.0
    assert 0.0 <= first["occupancy_entropy"] <= 1.0
    assert isinstance(first["noncollapsed"], bool)


def test_duration_hpo_cli_is_registered_and_bounded() -> None:
    args = build_parser().parse_args(
        [
            "hmm",
            "optimize-regimes",
            "--timeframes",
            "4h,1d",
            "--trials",
            "12",
            "--folds",
            "2",
        ]
    )
    assert args.command == "hmm"
    assert args.hmm_command == "optimize-regimes"
    assert args.timeframes == "4h,1d"
    assert args.trials == 12
    assert args.folds == 2


def test_duration_candidate_cannot_win_on_churn_when_prediction_degrades() -> None:
    payload = {
        "status": "COMPLETED_OBSERVER_ONLY",
        "selection": {
            "status": "SELECTED_NONCOLLAPSED_PREDICTIVE_PLATEAU",
            "selected_metrics": {
                "base_negative_log_predictive_density": 5.0,
                "predictive_nll_delta_vs_hmm": 0.5,
                "noncollapsed": True,
                "churn_delta_vs_hmm": -0.2,
            },
        },
    }
    result = _apply_predictive_acceptance_gate(payload)
    assert result["selection"]["accepted_candidate"] is False
    assert result["selection"]["predictive_noninferior"] is False
    assert result["status"].endswith("NO_ACCEPTED_CANDIDATE")
