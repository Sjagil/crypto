from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pandas as pd
import pytest

from config.settings import PathSettings, Settings
from research.research_factory import (
    FailedBreakdownReversalResearchAdapter,
    PromotionState,
    ProspectiveSnapshot,
    RejectionReason,
    ResearchCache,
    SharedCostModel,
    Stage0Hypothesis,
    Stage0Signals,
    build_research_factory_artifact,
    build_walk_forward_manifest,
    derive_dataset_identity,
    failed_breakdown_reversal_signals,
    generate_parameter_grid,
    load_immutable_ohlcv,
    multiple_testing_accounting,
    parameter_plateaus,
    prioritize_research_backlog,
    promotion_state_for,
    recursive_warmup_stability,
    simulate_stage0,
    stage0_causality_check,
    stage0_cost_stress,
    stage0_delay_liquidity_stress,
    static_lookahead_audit,
    strategy_result_correlation,
)
from utils.common import stable_hash


def _write_ohlcv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_parquet(path, index=False)


def _hypothesis() -> Stage0Hypothesis:
    return Stage0Hypothesis(
        hypothesis_id="test-failed-breakdown",
        strategy_family="FAILED_BREAKDOWN_REVERSAL",
        strategy_implementation="TEST_RESEARCH_ADAPTER",
        strategy_version="1.0.0",
        candidate_origin="STRUCTURAL_VARIANT",
        rationale="test",
        required_inputs=("open", "high", "low", "close", "volume"),
        optional_inputs=(),
        parameter_space={"lookback": (20, 40)},
        supported_timeframes=("1h",),
        side="LONG_ONLY",
        holding_semantics="SIGNAL_CLOSE_ENTRY_NEXT_OPEN",
        p0_5_classification="PROMISING_BUT_INSUFFICIENT_SAMPLE",
    )


def _identity(tmp_path: Path, frame: pd.DataFrame, market: str = "BTC-EUR"):
    path = tmp_path / f"{market}_1h.parquet"
    selected = frame.copy()
    selected.index.name = "timestamp"
    _write_ohlcv(path, selected)
    return load_immutable_ohlcv(
        path,
        provider="test",
        market=market,
        timeframe="1h",
    )


def test_dataset_identity_freezes_source_and_final_holdout(ohlcv, tmp_path) -> None:
    frame, identity = _identity(tmp_path, ohlcv)
    again, repeated = load_immutable_ohlcv(
        tmp_path / "BTC-EUR_1h.parquet",
        provider="test",
        market="BTC-EUR",
        timeframe="1h",
    )
    assert identity == repeated
    assert identity.quality.status == "READY"
    assert frame.equals(again)

    development = derive_dataset_identity(
        frame.iloc[:500], identity, purpose="TRAIN_VALIDATION_ONLY"
    )
    changed_final = frame.copy()
    changed_final.iloc[600:, changed_final.columns.get_loc("close")] *= 1.01
    repeated_development = derive_dataset_identity(
        changed_final.iloc[:500], identity, purpose="TRAIN_VALIDATION_ONLY"
    )
    assert development.dataset_id == repeated_development.dataset_id

    changed = ohlcv.copy()
    changed.iloc[100, changed.columns.get_loc("volume")] += 1.0
    _, changed_identity = _identity(tmp_path, changed)
    assert changed_identity.dataset_id != identity.dataset_id
    assert changed_identity.source_file_hash != identity.source_file_hash


def test_stage0_is_causal_next_open_cost_attributed_and_zero_authority(
    ohlcv, tmp_path
) -> None:
    frame, identity = _identity(tmp_path, ohlcv.iloc[:120])
    entry = pd.Series(False, index=frame.index)
    entry.iloc[10] = True
    exit_signal = pd.Series(False, index=frame.index)
    exit_signal.iloc[12] = True
    signals = Stage0Signals(
        entry=entry,
        exit=exit_signal,
        stop_distance=pd.Series(1_000.0, index=frame.index),
        target_distance=pd.Series(1_000.0, index=frame.index),
    )
    costs = SharedCostModel(
        cost_model_version="test-cost-v1",
        maker_fee_fraction=0.0,
        taker_fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=10.0,
    )
    result = simulate_stage0(
        frame,
        signals,
        hypothesis=_hypothesis(),
        dataset=identity,
        parameters={"maximum_holding_bars": 20},
        costs=costs,
        minimum_trades=1,
    )
    assert result.stage0_authority == "APPROXIMATE_RESEARCH_ONLY"
    assert result.trade_count == 1
    trade = result.trades[0]
    assert pd.Timestamp(trade.entry_timestamp) == frame.index[11]
    assert pd.Timestamp(trade.exit_timestamp) == frame.index[13]
    assert trade.raw_entry_price == pytest.approx(frame["open"].iloc[11])
    assert trade.fees_eur > 0
    assert trade.spread_cost_eur > 0
    assert trade.slippage_cost_eur > 0
    assert result.net_pnl_eur == pytest.approx(
        result.gross_pnl_eur
        - result.estimated_fees_eur
        - result.estimated_spread_eur
        - result.estimated_slippage_eur
    )


def test_bias_warmup_and_parameter_search_controls(ohlcv) -> None:
    parameters = FailedBreakdownReversalResearchAdapter().parameters()
    assert len(
        generate_parameter_grid(FailedBreakdownReversalResearchAdapter.parameter_space)
    ) == 64
    assert stage0_causality_check(
        ohlcv, failed_breakdown_reversal_signals, parameters
    )["status"] == "PASSED"
    assert static_lookahead_audit("x = close.shift(-1)")["status"] == "HARD_REJECT"
    assert static_lookahead_audit(failed_breakdown_reversal_signals)["status"] == "PASSED"
    warmup = recursive_warmup_stability(
        ohlcv,
        failed_breakdown_reversal_signals,
        parameters,
        startup_sizes=(200, 500, 700),
    )
    assert warmup["status"] == "PASSED"


def test_plateau_multiple_testing_and_versioned_cache(tmp_path) -> None:
    space = {"x": (1, 2, 3)}
    rows = [
        {
            "parameter_hash": stable_hash({"x": value}, length=32),
            "parameter_set": {"x": value},
            "net_expectancy_eur": 1.0,
        }
        for value in (1, 2, 3)
    ]
    plateaus = parameter_plateaus(rows, space)
    center = next(row for row in plateaus if row.parameter_hash == rows[1]["parameter_hash"])
    assert center.stable is True
    fragile_rows = [dict(row) for row in rows]
    fragile_rows[0]["net_expectancy_eur"] = -1.0
    fragile_rows[2]["net_expectancy_eur"] = -1.0
    isolated = parameter_plateaus(fragile_rows, space)[1]
    assert isolated.stable is False
    assert isolated.isolated_optimum is True

    trials = multiple_testing_accounting(
        hypotheses=1,
        parameter_combinations=64,
        assets=4,
        timeframes=2,
        regimes=1,
        cost_scenarios=4,
    )
    assert trials["total_tested_variants"] == 2_048
    assert trials["warning"] == "SELECTED_RESULTS_ARE_NOT_SINGLE_INDEPENDENT_TESTS"

    cache = ResearchCache(tmp_path / "cache")
    common = {
        "dataset_ids": ["dataset-v1"],
        "parameters": {"x": 1},
        "timeframe": "1h",
        "cost_model_version": "cost-v1",
        "feature_schema_version": "features-v1",
    }
    key = cache.key(strategy_version="strategy-v1", **common)
    assert key != cache.key(strategy_version="strategy-v2", **common)
    cache.store(key, {"result": "rejected"})
    assert cache.load(key)["result"] == "rejected"
    with pytest.raises(FileExistsError):
        cache.store(key, {"result": "promoted"})


def test_scheduler_promotion_rejection_and_prospective_snapshot_contract() -> None:
    scheduled = prioritize_research_backlog(
        [
            {
                "backlog_id": "negative",
                "p0_5_classification": "GROSS_NEGATIVE",
                "candidate_origin": "PARAMETER_VARIANT",
                "data_availability": 1.0,
                "sample_availability": 1.0,
                "diversification_potential": 1.0,
                "validation_cost": 0.0,
            },
            {
                "backlog_id": "structural",
                "p0_5_classification": "PROMISING",
                "candidate_origin": "STRUCTURAL_VARIANT",
                "data_availability": 1.0,
                "sample_availability": 0.5,
                "diversification_potential": 0.5,
                "validation_cost": 0.25,
            },
        ]
    )
    assert scheduled[0]["backlog_id"] == "structural"
    assert scheduled[1]["scheduler_state"] == "DEFER_DO_NOT_RESCUE"
    assert promotion_state_for(stage0_survivor_count=0, exact_status=None) == (
        PromotionState.STAGE0_REJECTED
    )
    assert promotion_state_for(
        stage0_survivor_count=1, exact_status="ROBUST_EXACT_PASS"
    ) == PromotionState.FORWARD_CANDIDATE
    assert RejectionReason.NEGATIVE_NET_EXPECTANCY.value == "NEGATIVE_NET_EXPECTANCY"

    values = {
        "candidate_id": "candidate-v1",
        "signal_timestamp": "2026-01-01T00:00:00Z",
        "features": {"atr": 1.2},
        "strategy_version": "strategy-v1",
        "parameters": {"lookback": 20},
        "market_context": {"market": "BTC-EUR"},
        "cost_prediction_version": "cost-v1",
        "entry_plan": {"side": "BUY", "type": "LIMIT"},
        "stop_targets": {"stop": 90.0, "target": 120.0},
    }
    snapshot = ProspectiveSnapshot.create(**values)
    assert snapshot == ProspectiveSnapshot.create(**values)
    assert snapshot.future_information is False
    assert snapshot.canonical_outcome_source == "P0_CANONICAL_FINANCIAL_STATE"
    with pytest.raises(FrozenInstanceError):
        snapshot.strategy_version = "strategy-v2"  # type: ignore[misc]


def test_manifest_has_explicit_purge_embargo_and_immutable_final(ohlcv, tmp_path) -> None:
    _, btc = _identity(tmp_path / "btc", ohlcv, "BTC-EUR")
    _, eth = _identity(tmp_path / "eth", ohlcv * 1.01, "ETH-EUR")
    costs = SharedCostModel(
        cost_model_version="cost-v1",
        maker_fee_fraction=0.0015,
        taker_fee_fraction=0.0025,
        spread_bps=5.0,
        slippage_bps=10.0,
    )
    manifest = build_walk_forward_manifest(
        [btc, eth],
        strategy_id="strategy-v1",
        parameter_search_scope={"x": [1, 2]},
        timeframe="1h",
        costs=costs,
        code_version="abc",
        purge_bars=40,
        embargo_bars=2,
        asset_holdout=("ETH-EUR",),
    )
    assert manifest.final_test_immutable is True
    assert manifest.asset_holdout == ("ETH-EUR",)
    for fold in manifest.folds:
        purge = pd.Timestamp(fold.validation_start) - pd.Timestamp(fold.train_end)
        embargo = pd.Timestamp(fold.test_start) - pd.Timestamp(fold.validation_end)
        assert purge.total_seconds() == 40 * 3_600
        assert embargo.total_seconds() == 2 * 3_600


def test_cost_stress_is_monotonic(ohlcv, tmp_path) -> None:
    frame, identity = _identity(tmp_path, ohlcv)
    parameters = FailedBreakdownReversalResearchAdapter().parameters()
    rows = stage0_cost_stress(
        frames={"btc": frame},
        datasets={"btc": identity},
        hypothesis=_hypothesis(),
        builder=failed_breakdown_reversal_signals,
        parameters=parameters,
        costs=SharedCostModel(
            cost_model_version="cost-v1",
            maker_fee_fraction=0.0015,
            taker_fee_fraction=0.0025,
            spread_bps=5.0,
            slippage_bps=10.0,
        ),
    )
    assert [row["scenario"] for row in rows] == [
        "BASE",
        "BASE_PLUS_10_BPS",
        "BASE_PLUS_25_BPS",
        "BASE_PLUS_50_BPS",
    ]
    assert [row["net_pnl_eur"] for row in rows] == sorted(
        (row["net_pnl_eur"] for row in rows), reverse=True
    )
    delay_rows = stage0_delay_liquidity_stress(
        frames={"btc": frame},
        datasets={"btc": identity},
        hypothesis=_hypothesis(),
        builder=failed_breakdown_reversal_signals,
        parameters=parameters,
        costs=SharedCostModel(
            cost_model_version="cost-v1",
            maker_fee_fraction=0.0015,
            taker_fee_fraction=0.0025,
            spread_bps=5.0,
            slippage_bps=10.0,
        ),
    )
    assert {row["scenario"] for row in delay_rows} == {
        "NEXT_OPEN_BASE",
        "ONE_EXTRA_BAR_DELAY",
        "TWO_EXTRA_BARS_DELAY",
        "LIQUIDITY_2X",
        "LIQUIDITY_3X",
    }


def test_strategy_correlation_flags_duplicate_paths(ohlcv, tmp_path) -> None:
    frame, identity = _identity(tmp_path, ohlcv)
    entry = pd.Series(False, index=frame.index)
    exit_signal = pd.Series(False, index=frame.index)
    entry.iloc[::48] = True
    exit_signal.iloc[1::48] = True
    result = simulate_stage0(
        frame,
        Stage0Signals(
            entry=entry,
            exit=exit_signal,
            stop_distance=pd.Series(1_000_000.0, index=frame.index),
            target_distance=pd.Series(1_000_000.0, index=frame.index),
        ),
        hypothesis=_hypothesis(),
        dataset=identity,
        parameters={"maximum_holding_bars": 10},
        costs=SharedCostModel(
            cost_model_version="zero",
            maker_fee_fraction=0.0,
            taker_fee_fraction=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        ),
        minimum_trades=1,
    )
    duplicate = replace(result, parameter_hash="different-parameter-same-path")
    correlations = strategy_result_correlation([result, duplicate])
    assert correlations[0]["daily_pnl_correlation"] == pytest.approx(1.0)
    assert correlations[0]["duplicate_alpha_warning"] is True


def test_full_factory_is_immutable_and_has_no_execution_authority(
    ohlcv, isolated_settings: Settings, tmp_path
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    datasets = []
    for market, scale in (("BTC-EUR", 1.0), ("ETH-EUR", 0.75)):
        path = settings.paths.processed_data_dir / f"{market}_1h.parquet"
        selected = ohlcv.iloc[:500].copy() * scale
        selected.index.name = "timestamp"
        _write_ohlcv(path, selected)
        datasets.append(
            {"path": path, "provider": "test", "market": market, "timeframe": "1h"}
        )
    p0_5_path = settings.paths.output_dir / "economics" / "runs" / "p05" / "evidence.json"
    p0_5_path.parent.mkdir(parents=True, exist_ok=True)
    p0_5 = {
        "artifact_hash": "p05-hash",
        "canonical_state_version": 1,
        "canonical_state_hash": "state-hash",
        "replay_deterministic": True,
        "closed_episode_count": 290,
        "aggregate": {"net_pnl_eur": -51.84, "profit_factor": 0.25},
        "validation_backlog_count": 852,
        "family_results": [
            {"dimension_value": f"NEGATIVE_{index}", "cost_classification": "GROSS_NEGATIVE"}
            for index in range(3)
        ]
        + [
            {
                "dimension_value": "FAILED_BREAKDOWN_REVERSAL",
                "cost_classification": "GROSS_AND_NET_POSITIVE",
                "net_pnl_eur": 1.0,
                "net_expectancy_eur": 0.1,
            }
        ],
        "strategy_results": [],
        "episodes": [],
        "promotion_recommendations": [],
    }
    p0_5_path.write_text(json.dumps(p0_5), encoding="utf-8")
    latest = settings.paths.output_dir / "economics" / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps({"artifact_path": str(p0_5_path)}), encoding="utf-8")

    first = build_research_factory_artifact(
        settings,
        dataset_specs=datasets,
        maximum_rows=500,
        execute_exact=False,
    )
    second = build_research_factory_artifact(
        settings,
        dataset_specs=datasets,
        maximum_rows=500,
        execute_exact=False,
    )
    assert first["run_id"] == second["run_id"]
    assert first["artifact_hash"] == second["artifact_hash"]
    artifact = json.loads(Path(first["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["p0_5_branch"]["decision"] == (
        "ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION"
    )
    assert artifact["hypothesis"]["strategy_family"] == (
        "FAILED_BREAKDOWN_REVERSAL"
    )
    assert artifact["stage0"]["tested_variant_count"] == 128
    assert artifact["stage0"]["authority"] == "APPROXIMATE_RESEARCH_ONLY"
    assert artifact["backlog"]["inventory_backlog_after"] == 852
    assert artifact["research_scheduler"]["fifo"] is False
    assert artifact["regime_holdout"]["status"].startswith("NOT_EVALUABLE")
    assert artifact["ml_status"]["authority"] == "SHADOW_ONLY"
    assert artifact["safety"] == {
        "real_orders_submitted": 0,
        "real_orders_cancelled": 0,
        "real_protective_orders_modified": 0,
        "private_bitvavo_mutations": 0,
        "live_authority_increase": False,
        "risk_limit_increase": False,
        "shariah_policy_weakening": False,
        "research_trades_placed": 0,
    }
    assert all(
        row["created_at"] == row["data_cutoff"]
        for row in artifact["experiment_contracts"]
    )
