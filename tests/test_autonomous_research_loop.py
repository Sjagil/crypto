from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from research.autonomous_rd import (
    ExperimentDecision,
    ExperimentFeedback,
    PreregisteredExperiment,
    ResearchHypothesis,
    ResearchTrace,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def test_hypothesis_experiment_feedback_trace_is_immutable_and_bounded() -> None:
    hypothesis = ResearchHypothesis.create(
        statement="A volatility reduction policy improves net downside control.",
        rationale="Prospective position episodes can falsify it.",
        falsification_criteria=("net downside is not lower after costs",),
        evidence_inputs=("dataset_prospective_positions_v1",),
        created_at=NOW,
    )
    experiment = PreregisteredExperiment.create(
        hypothesis_id=hypothesis.hypothesis_id,
        dataset_ids=("dataset_prospective_positions_v1",),
        code_commit="abc123",
        metrics=("net_return", "expected_shortfall"),
        acceptance_thresholds={"net_return": Decimal("0")},
        leakage_controls=("point_in_time", "purged_walk_forward"),
        cost_model_version="canonical_cost_v1",
        preregistered_at=NOW,
    )
    feedback = ExperimentFeedback.create(
        experiment_id=experiment.experiment_id,
        decision=ExperimentDecision.REJECT,
        observed_metrics={"net_return": Decimal("-0.01")},
        observations=("failed after canonical costs",),
        failure_reasons=("negative_net_return",),
        recorded_at=NOW,
    )
    empty = ResearchTrace.empty("position_management")
    with_hypothesis = empty.append(hypothesis)
    complete = with_hypothesis.append(experiment).append(feedback)
    assert empty.records == ()
    assert len(with_hypothesis.records) == 1
    assert [item.sequence for item in complete.records] == [0, 1, 2]
    assert complete.execution_authority is False
    assert complete.automatic_promotion_permitted is False
    assert feedback.live_promotion_authority is False


def test_trace_rejects_feedback_without_preregistered_parent() -> None:
    feedback = ExperimentFeedback.create(
        experiment_id="missing",
        decision=ExperimentDecision.INCONCLUSIVE,
        observed_metrics={},
        observations=("not enough data",),
        recorded_at=NOW,
    )
    with pytest.raises(ValueError, match="parent experiment"):
        ResearchTrace.empty("x").append(feedback)
