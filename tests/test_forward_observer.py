from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import core.cli as cli
from research.forward_observer import (
    FORWARD_OBSERVER_SCHEMA_VERSION,
    PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION,
    ForwardHistoryRevisionError,
    ForwardPerformanceGatePolicy,
    build_breakout_forward_evidence,
    build_forward_hash_chain,
    build_rotation_forward_evidence,
    merge_breakout_forward_manifest,
    merge_forward_observations,
    merge_portfolio_forward_manifest,
)
from research.portfolio_breakout import (
    BreakoutPortfolioParameters,
    backtest_breakout_portfolio,
)
from research.portfolio_selection import (
    CapitalUtilizationPolicy,
    RotationParameters,
    RotationPortfolioPolicy,
    backtest_rotation,
)


def _frames(rows: int = 540) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC")

    def frame(drift: float, phase: float) -> pd.DataFrame:
        step = (
            drift
            + 0.008 * np.sin(np.arange(rows) / 17.0 + phase)
            + 0.004 * np.cos(np.arange(rows) / 7.0 + phase)
        )
        close = 100.0 * np.exp(np.cumsum(step))
        open_price = close * (
            1.0 + 0.001 * np.sin(np.arange(rows) / 3.0 + phase)
        )
        return pd.DataFrame(
            {
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.01,
                "low": np.minimum(open_price, close) * 0.99,
                "close": close,
                "volume": 10_000.0,
            },
            index=index,
        )

    return {
        "BTC-EUR": frame(0.0008, 0.0),
        "ETH-EUR": frame(0.0010, 0.6),
        "SOL-EUR": frame(0.0005, 1.2),
        "LINK-EUR": frame(0.0007, 1.8),
    }


def _result(frames: dict[str, pd.DataFrame]):
    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(frames),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=90,
    )
    parameters = BreakoutPortfolioParameters(
        entry_lookback=20,
        exit_lookback=10,
        trend_ema_period=50,
        weighting="equal",
    )
    return backtest_breakout_portfolio(
        frames,
        parameters,
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
    )


def test_forward_evidence_is_realized_next_open_orderless_and_costed():
    frames = _frames()
    result = _result(frames)
    forward_start = frames["BTC-EUR"].index[-70]
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=forward_start,
    )

    assert len(evidence.observations) == 69
    assert evidence.summary["closed_daily_observations"] == 69
    assert evidence.summary["remaining_closed_daily_observations"] == 296
    progress = evidence.summary["diagnostic_progress"]
    assert progress["next_pending_milestone"] == 90
    assert progress["milestones"][0] == {
        "closed_daily_observations": 30,
        "reached": True,
        "remaining": 0,
        "purpose": "DIAGNOSTIC_ONLY",
    }
    assert progress["diagnostic_milestones_authorize_promotion"] is False
    assert evidence.summary["formal_performance_gates_evaluated"] is False
    assert evidence.summary["performance_metrics"] is None
    assert evidence.degradation_observation is not None
    assert evidence.degradation_observation["observation_count"] == 69
    assert any(item["decision_event"] for item in evidence.observations)
    for observation in evidence.observations:
        assert observation["schema_version"] == FORWARD_OBSERVER_SCHEMA_VERSION
        assert (
            pd.Timestamp(observation["realization_at"])
            > pd.Timestamp(observation["execution_at"])
        )
        assert sum(observation["target_weights"].values()) <= 0.40 + 1e-12
        assert observation["cash_fraction"] >= 0.60 - 1e-12
        assert observation["expected_cost_fraction"] >= 0
        assert observation["orders_generated"] == 0
        assert observation["orders_submitted"] == 0
        assert observation["paper_candidate_permitted"] is False
        assert observation["live_ready"] is False


def test_formal_performance_gates_activate_only_after_all_sample_requirements():
    frames = _frames()
    result = _result(frames)
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=frames["BTC-EUR"].index[-70],
        minimum_observations=30,
        minimum_rebalances=0,
        minimum_regime_decisions=0,
        performance_policy=ForwardPerformanceGatePolicy(
            minimum_profit_factor=1.0001,
            minimum_stressed_profit_factor=1.0,
            maximum_drawdown=0.99,
            minimum_effective_sample_size=2,
            bootstrap_samples=100,
            bootstrap_block_size=5,
        ),
    )

    assert evidence.summary["formal_performance_gates_evaluated"] is True
    assert evidence.summary["performance_metrics"] is not None
    assert evidence.summary["performance_checks"] is not None
    assert (
        evidence.summary["performance_metrics"]["raw_observations"]
        == 69
    )
    assert evidence.summary["status"] in {
        "FORWARD_PERFORMANCE_PASS",
        "FORWARD_PERFORMANCE_NOT_QUALIFIED",
    }
    assert evidence.summary["paper_candidate_permitted"] is False
    assert evidence.summary["live_ready"] is False


def test_no_post_freeze_candles_means_empty_collecting_evidence():
    frames = _frames()
    result = _result(frames)
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=frames["BTC-EUR"].index[-1]
        + pd.Timedelta(1, unit="D"),
    )

    assert evidence.observations == ()
    assert evidence.decisions == ()
    assert evidence.degradation_observation is None
    assert evidence.summary["status"] == "COLLECTING_FORWARD_DATA"
    assert evidence.summary["forward_net_return"] == 0.0


def test_empty_decision_stream_remains_valid_forward_evidence():
    frames = _frames()
    result = replace(_result(frames), decisions=pd.DataFrame())
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=frames["BTC-EUR"].index[-20],
    )

    assert len(evidence.observations) == 19
    assert evidence.decisions == ()
    assert all(not row["decision_event"] for row in evidence.observations)
    assert all(row["reason"] == "HOLD_UNCHANGED" for row in evidence.observations)


def test_forward_merge_is_idempotent_and_rejects_revision():
    frames = _frames()
    result = _result(frames)
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=frames["BTC-EUR"].index[-40],
    )
    first = merge_forward_observations([], evidence.observations)
    second = merge_forward_observations(first, evidence.observations)
    assert second == first
    chain = build_forward_hash_chain(first)
    assert chain["record_count"] == len(first)
    assert chain["entries"][0]["previous_record_hash"] == "0" * 64
    assert chain["entries"][-1]["record_hash"] == chain["root_hash"]

    corrected_frames = {market: frame.copy() for market, frame in frames.items()}
    for market in corrected_frames:
        row = corrected_frames[market].index[-1]
        corrected_frames[market].loc[
            row,
            ["open", "high", "low", "close"],
        ] *= 1.10
    corrected = build_breakout_forward_evidence(
        result,
        corrected_frames,
        forward_start=frames["BTC-EUR"].index[-40],
    )
    with pytest.raises(
        ForwardHistoryRevisionError,
        match="FORWARD_HISTORY_REVISION_DETECTED",
    ):
        merge_forward_observations(first, corrected.observations)


def test_forward_merge_rejects_source_truncation():
    frames = _frames()
    result = _result(frames)
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=frames["BTC-EUR"].index[-20],
    )
    existing = list(evidence.observations)
    with pytest.raises(
        ForwardHistoryRevisionError,
        match="FORWARD_SOURCE_TRUNCATION_DETECTED",
    ):
        merge_forward_observations(existing, evidence.observations[:-1])


def test_manifest_merge_preserves_identity_and_rejects_dna_mixing():
    frames = _frames()
    result = _result(frames)
    start = frames["BTC-EUR"].index[-30]
    evidence = build_breakout_forward_evidence(
        result,
        frames,
        forward_start=start,
    )
    identity = result.summary()["execution_identity"]
    merged = merge_breakout_forward_manifest(
        {
            "source_candidate_identity": "frozen",
            "strategy_dna_hash": result.parameters.dna_hash,
            "execution_identity": identity,
            "forward_start": start.isoformat(),
            "forward_observations": [],
        },
        evidence,
        source_candidate_identity="frozen",
        strategy_dna_hash=result.parameters.dna_hash,
        execution_identity=identity,
        forward_start=start,
    )
    assert len(merged["forward_observations"]) == 29
    assert merged["forward_hash_chain"]["record_count"] == 29
    assert (
        merged["forward_hash_chain"]["entries"][-1]["record_hash"]
        == merged["forward_hash_chain"]["root_hash"]
    )
    assert merged["forward_observer_schema_version"] == (
        FORWARD_OBSERVER_SCHEMA_VERSION
    )
    assert merged["orders_generated"] == 0
    assert merged["live_ready"] is False

    contaminated = copy.deepcopy(merged)
    contaminated["strategy_dna_hash"] = "different"
    with pytest.raises(
        ForwardHistoryRevisionError,
        match="identity mismatch",
    ):
        merge_breakout_forward_manifest(
            contaminated,
            evidence,
            source_candidate_identity="frozen",
            strategy_dna_hash=result.parameters.dna_hash,
            execution_identity=identity,
            forward_start=start,
        )


def test_campaign_recalculation_preserves_only_forward_fields():
    start = pd.Timestamp("2026-07-25", tz="UTC")
    existing = {
        "source_candidate_identity": "frozen",
        "strategy_dna_hash": "dna",
        "execution_identity": "execution",
        "forward_start": start.isoformat(),
        "latest_historical_decision": {"stale": True},
        "forward_observer_schema_version": "v1",
        "forward_observations": [{"observation_id": "one"}],
        "forward_decisions": [{"observation_id": "one"}],
        "forward_summary": {"closed_daily_observations": 1},
        "degradation_observation": None,
    }
    preserved = cli._preserved_breakout_forward_fields(
        existing,
        source_candidate_identity="frozen",
        strategy_dna_hash="dna",
        execution_identity="execution",
        forward_start=start,
    )

    assert preserved["forward_observations"] == [
        {"observation_id": "one"}
    ]
    assert "latest_historical_decision" not in preserved
    with pytest.raises(ForwardHistoryRevisionError, match="identity mismatch"):
        cli._preserved_breakout_forward_fields(
            existing,
            source_candidate_identity="different",
            strategy_dna_hash="dna",
            execution_identity="execution",
            forward_start=start,
        )


def test_rotation_allocation_policy_gets_distinct_append_only_forward_evidence():
    frames = _frames()
    allocation = CapitalUtilizationPolicy(
        name="TEST_SEMI_AGGRESSIVE",
        base_exposure_budget=0.80,
        maximum_total_exposure=0.80,
        maximum_position_exposure=0.40,
        minimum_cash=0.20,
    )
    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(frames),
        maximum_total_exposure=0.80,
        maximum_position_exposure=0.40,
        minimum_cash=0.20,
        minimum_history_observations=90,
    )
    result = backtest_rotation(
        frames,
        RotationParameters(
            momentum_lookback=20,
            additional_momentum_lookbacks=(90,),
            top_n=2,
            rebalance_days=7,
            asset_ema_period=50,
            btc_ema_period=200,
            require_btc_uptrend=True,
            continuous_regime=True,
            weighting="equal",
            gross_exposure=0.40,
            minimum_cash=0.20,
            maximum_positions=2,
        ),
        fee_rate=0.0025,
        slippage_bps=8.0,
        spread_bps=5.0,
        portfolio_policy=policy,
        capital_utilization_policy=allocation,
    )
    start = frames["BTC-EUR"].index[-45]
    evidence = build_rotation_forward_evidence(
        result,
        frames,
        forward_start=start,
    )
    identity = result.summary()["execution_identity"]
    merged = merge_portfolio_forward_manifest(
        {},
        evidence,
        source_candidate_identity="frozen",
        strategy_dna_hash=result.parameters.dna_hash,
        execution_identity=identity,
        forward_start=start,
    )

    assert evidence.schema_version == (
        PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION
    )
    assert len(merged["forward_observations"]) == 44
    assert merged["forward_observer_schema_version"] == (
        PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION
    )
    assert all(
        observation["cash_fraction"] >= 0.20 - 1e-12
        for observation in merged["forward_observations"]
    )
    assert all(
        sum(observation["target_weights"].values()) <= 0.80 + 1e-12
        for observation in merged["forward_observations"]
    )
    assert merged["orders_generated"] == 0
    assert merged["orders_submitted"] == 0
    assert merged["paper_candidate_permitted"] is False
    assert merged["live_ready"] is False
