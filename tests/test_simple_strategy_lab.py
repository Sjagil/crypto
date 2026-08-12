from __future__ import annotations

import json

import pandas as pd

from core.cli import (
    _simple_lab_canonical_candidate_pids,
    _simple_lab_generation_plan,
    _simple_lab_history_rows,
    _simple_lab_market_cycle,
    _simple_lab_requested_markets,
    _simple_lab_requested_timeframes,
    _simple_lab_validation_budget,
    build_parser,
)
from research.combinatorial_lab import (
    BlockDirection,
    BlockRole,
    CombinationGenerator,
    GenerationMode,
    LabRunner,
    LogicMode,
    _fast_screen_worker,
    signal_block_registry,
)
from research.simple_strategy_lab import (
    DEFAULT_TIMEFRAMES,
    SimpleStrategyResearchFactory,
    _unrank_combination,
    frequency_bucket,
    registry_driven_signal_blocks,
)


def test_combination_unranking_matches_itertools_order() -> None:
    assert _unrank_combination(5, 2, 0) == (0, 1)
    assert _unrank_combination(5, 2, 5) == (1, 3)
    assert _unrank_combination(5, 2, 9) == (3, 4)


def test_frequency_buckets_are_separate_from_economic_quality() -> None:
    assert frequency_bucket(0) == "ULTRA_LOW_FREQUENCY"
    assert frequency_bucket(3) == "LOW_FREQUENCY"
    assert frequency_bucket(12) == "MEDIUM_FREQUENCY"
    assert frequency_bucket(52) == "HIGH_FREQUENCY"
    assert frequency_bucket(250) == "VERY_HIGH_FREQUENCY"


def test_generation_pauses_while_validation_backlog_is_above_high_watermark() -> None:
    plan = _simple_lab_generation_plan(
        {"total_currently_queued": 127_000},
        requested_batch_size=2_000,
    )

    assert plan["status"] == "THROTTLED_VALIDATION_BACKLOG"
    assert plan["effective_batch_size"] == 0
    assert plan["requested_batch_size"] == 2_000


def test_generation_resumes_below_validation_backlog_high_watermark() -> None:
    plan = _simple_lab_generation_plan(
        {"total_currently_queued": 9_999},
        requested_batch_size=100,
    )

    assert plan["status"] == "GENERATION_ALLOWED"
    assert plan["effective_batch_size"] == 100


def test_validation_budget_increases_depth_without_increasing_workers() -> None:
    budget = _simple_lab_validation_budget(
        {"total_currently_queued": 127_000},
        requested_batch_size=2,
        requested_max_trials=2,
    )

    assert budget == {
        "backlog_priority_active": True,
        "effective_backtest_batch_size": 8,
        "effective_max_trials": 4,
    }


def test_registry_driven_blocks_expand_executable_indicator_coverage() -> None:
    original = signal_block_registry()
    expanded = registry_driven_signal_blocks()
    automatic = {
        block_id: block for block_id, block in expanded.items() if block_id.startswith("auto__")
    }
    assert len(expanded) > len(original)
    assert automatic
    assert any("momentum_rsi" in value for value in automatic)
    assert any("chaikin_money_flow" in value for value in automatic)
    assert "mtf__1d__regime_bullish" in expanded
    assert "15m" in expanded["mtf__1d__regime_bullish"].supported_timeframes
    assert "1d" not in expanded["mtf__1d__regime_bullish"].supported_timeframes
    assert all(
        not (block.direction is BlockDirection.BEARISH and block.role is BlockRole.ENTRY_TRIGGER)
        for block in expanded.values()
    )


def test_registry_driven_numeric_block_is_backward_only() -> None:
    expanded = registry_driven_signal_blocks()
    block = next(
        value
        for key, value in expanded.items()
        if key.startswith("auto__") and value.signal_kind == "RISING"
    )
    feature = pd.DataFrame(
        {block.feature: [1.0, 2.0, 1.5, 1.6]},
        index=pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
    )
    actual = block.calculate(feature).tolist()
    assert actual == [False, True, False, True]


def test_canonical_runner_accepts_registry_driven_blocks(
    isolated_settings,
) -> None:
    expanded = registry_driven_signal_blocks()
    runner = LabRunner(isolated_settings, registry=expanded)
    assert runner.registry.keys() == expanded.keys()
    assert any(value.startswith("auto__") for value in runner.registry)


def test_screen_worker_uses_injected_registry_driven_block(features) -> None:
    expanded = registry_driven_signal_blocks()
    block = next(
        value
        for key, value in expanded.items()
        if key.startswith("auto__")
        and value.role is BlockRole.ENTRY_TRIGGER
        and value.feature in features
    )
    selected = {block.block_id: block}
    combination = CombinationGenerator(selected).materialize_membership(
        (block.block_id,),
        logic_mode=LogicMode.LAYERED,
        timeframes=("1h",),
    )
    result = _fast_screen_worker(
        {"BTC-EUR": features},
        combination,
        {},
        0.005,
        selected,
    )
    assert result["source"] == "SCREENING_ONLY"
    assert result["signal_funnel"]["raw_bar_count"] == len(features)
    assert "profit_factor" in result
    assert "net_expectancy" in result
    assert "profit_factor" in result["signal_funnel"]["per_market"]["BTC-EUR"]
    assert "net_return" in result["signal_funnel"]["per_market"]["BTC-EUR"]


def test_materialize_membership_does_not_scan_siblings() -> None:
    registry = signal_block_registry()
    generator = CombinationGenerator(registry)
    result = generator.materialize_membership(
        ("rsi_oversold", "price_above_ema200"),
        logic_mode=LogicMode.LAYERED,
        timeframes=("1h", "4h"),
    )
    assert result.combination_size == 2
    assert result.block_ids == (
        "price_above_ema200",
        "rsi_oversold",
    )
    assert len(result.strategy_dna_hash) == 64
    assert result.requested_timeframes == ("1h", "4h")


def test_simple_factory_batches_without_content_cap(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    registry = {
        block_id: source[block_id]
        for block_id in (
            "rsi_oversold",
            "donchian20_breakout",
            "price_above_ema200",
            "relative_volume_expansion",
        )
    }
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "simple",
        registry=registry,
    )
    first = factory.materialize_batch(
        batch_size=3,
        complexities=(1, 2),
        timeframes=("15m", "1h"),
    )
    assert first["status"] == "BATCH_COMPLETE"
    assert first["queue"]["total_known_raw_memberships"] == 10
    assert first["queue"]["total_persisted"] == 3
    assert first["queue"]["content_limit"] is None
    assert first["orders_generated"] == 0
    assert first["queue"]["queue_path"].endswith(".sqlite3")
    assert "generation_queue_" in first["queue"]["queue_path"]
    assert "generation_cursor_" in first["queue"]["cursor_path"]

    second = factory.materialize_batch(
        batch_size=20,
        complexities=(1, 2),
        timeframes=("15m", "1h"),
    )
    assert second["status"] == "COMPLETE"
    assert second["queue"]["total_persisted"] == 10
    assert second["queue"]["total_remaining_to_materialize"] == 0
    assert second["queue"]["total_possible_registry_combinations"] == 10
    assert second["queue"]["total_unique_dna"] == 10
    assert second["queue"]["total_materialized_attempts"] == 10
    assert second["queue"]["total_deduplicated"] == 0
    assert (
        second["queue"]["total_materialized_causally_valid"] + second["queue"]["total_excluded"]
        == 10
    )
    assert isinstance(
        second["queue"]["exclusion_reason_counts"],
        dict,
    )
    assert second["queue"]["deduplicated"]
    assert second["queue"]["resumable"]

    summary = json.loads(
        (tmp_path / "simple" / "complete_search_space_summary.json").read_text(encoding="utf-8")
    )
    assert summary["generation_is_separate_from_validation"]
    assert summary["examples_are_not_whitelists"]
    assert summary["multi_timeframe_route_count"] == 1
    assert (tmp_path / "simple" / "signal_funnels.csv").is_file()
    assert (tmp_path / "simple" / "family_pair_coverage.csv").is_file()
    assert (tmp_path / "simple" / "multi_timeframe_route_coverage.csv").is_file()
    audit = json.loads(
        (tmp_path / "simple" / "objective_completion_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "IN_PROGRESS"
    assert audit["invariants"]["content_limit_is_none"]
    assert audit["invariants"]["resource_limits_only"]
    assert audit["invariants"]["resumable"]
    assert audit["invariants"]["deduplicated"]
    assert audit["invariants"]["research_orders_zero"]
    assert not audit["invariants"]["maximum_data_sync_complete"]
    assert audit["requirements"]["multi_timeframe_routes"]["route_count"] == 1


def test_registry_queue_generations_are_preserved(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    output_dir = tmp_path / "registry_generations"
    first = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=output_dir,
        registry={"rsi_oversold": source["rsi_oversold"]},
    )
    first.materialize_batch(
        batch_size=1,
        complexities=(1,),
        timeframes=("1h",),
    )
    second = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=output_dir,
        registry={
            "rsi_oversold": source["rsi_oversold"],
            "donchian20_breakout": source["donchian20_breakout"],
        },
    )

    state = json.loads((output_dir / "current_registry_state.json").read_text(encoding="utf-8"))

    assert second.queue_path != first.queue_path
    assert state["preserved_registry_queue_count"] == 1
    assert state["preserved_registry_queues"] == [str(first.queue_path.resolve())]
    assert first.queue_path.is_file()


def test_objective_completion_audit_requires_durable_complete_evidence(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    output_dir = tmp_path / "objective_audit"
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=output_dir,
        registry={"rsi_oversold": source["rsi_oversold"]},
    )
    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir.parent / "data_sync_progress.json").write_text(
        json.dumps(
            {
                "status": "PHASE_COMPLETE",
                "total_operations": 10,
                "completed_operations": 10,
                "failure_count": 0,
                "live_orders": 0,
            }
        ),
        encoding="utf-8",
    )
    trade_audit_dir = output_dir.parent / "trade_count_audit"
    trade_audit_dir.mkdir(parents=True, exist_ok=True)
    (trade_audit_dir / "summary.json").write_text(
        json.dumps({"status": "COMPLETE"}),
        encoding="utf-8",
    )
    (output_dir / "generation_summary.json").write_text(
        json.dumps({"timeframes": list(DEFAULT_TIMEFRAMES)}),
        encoding="utf-8",
    )
    (output_dir / "result_reconciliation_summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        ),
        encoding="utf-8",
    )
    result_row = {
        "strategy_dna_hash": "a" * 64,
        "strategy_variant_dna_hash": "b" * 64,
        "block_ids": "rsi_oversold",
        "market": "BTC-EUR",
        "timeframe": "1h",
        "status": "EXACT_REAL",
        "net_return": 0.1,
        "profit_factor": 1.2,
        "completed_trades": 10,
    }
    for filename in (
        "single_condition_results.csv",
        "two_block_results.csv",
        "three_block_results.csv",
        "four_block_results.csv",
    ):
        pd.DataFrame([result_row]).to_csv(
            output_dir / filename,
            index=False,
        )
    pd.DataFrame(
        [
            {
                **result_row,
                "timeframe": timeframe,
            }
            for timeframe in ("15m", "1h", "4h", "1d")
        ]
    ).to_csv(
        output_dir / "timeframe_results.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                **result_row,
                "block_ids": "fractal_high_breakout",
            },
            {
                **result_row,
                "block_ids": ("fractal_high_breakout|relative_volume_expansion"),
            },
        ]
    ).to_csv(
        output_dir / "fractal_results.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                **result_row,
                "block_ids": "bullish_engulfing",
            }
        ]
    ).to_csv(
        output_dir / "candlestick_results.csv",
        index=False,
    )
    for filename in (
        "frequency_buckets.csv",
        "signal_funnels.csv",
        "ablation_results.csv",
    ):
        pd.DataFrame([result_row]).to_csv(
            output_dir / filename,
            index=False,
        )
    pd.DataFrame([{"execution_timeframe": "1h"}]).to_csv(
        output_dir / "multi_timeframe_route_coverage.csv",
        index=False,
    )

    audit = factory.write_objective_completion_audit(
        queue={
            "content_limit": None,
            "resource_limit_only": True,
            "resumable": True,
            "deduplicated": True,
            "orders_generated": 0,
            "orders_submitted": 0,
            "complexity_status_counts": {"1": {"QUEUED": 0}},
            "registered_signal_blocks": 1,
            "total_known_raw_memberships": 1,
            "total_persisted": 1,
            "total_remaining_to_materialize": 0,
        }
    )

    assert audit["status"] == "COMPLETE"
    assert all(audit["invariants"].values())
    assert audit["requirements"]["timeframe_trade_evidence"]["1h"]["completed_round_trips"] == 10


def test_generation_batch_is_stratified_across_one_to_five_blocks(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    selected_ids = (
        "rsi_oversold",
        "rsi_recovery",
        "donchian20_breakout",
        "price_above_ema200",
        "relative_volume_expansion",
        "bollinger_lower_reversion",
    )
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "stratified",
        registry={block_id: source[block_id] for block_id in selected_ids},
    )
    result = factory.materialize_batch(
        batch_size=10,
        complexities=(1, 2, 3, 4, 5),
        timeframes=("15m", "1h", "4h"),
    )
    counts = result["queue"]["complexity_status_counts"]
    assert set(counts) == {"1", "2", "3", "4", "5"}
    assert all(sum(counts[str(size)].values()) == 2 for size in range(1, 6))
    assert result["cursor"]["scheduling"] == "DETERMINISTIC_COMPLEXITY_ROUND_ROBIN"
    assert set(result["cursor"]["positions"]) == {"1", "2", "3", "4", "5"}


def test_backtest_batch_is_stratified_across_one_to_five_blocks(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    selected_ids = (
        "rsi_oversold",
        "rsi_recovery",
        "donchian20_breakout",
        "price_above_ema200",
        "relative_volume_expansion",
        "bollinger_lower_reversion",
    )
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "stratified_backtest",
        registry={block_id: source[block_id] for block_id in selected_ids},
    )
    factory.materialize_batch(
        batch_size=15,
        complexities=(1, 2, 3, 4, 5),
        timeframes=("15m", "1h", "4h"),
    )

    selected = factory.queued_strategies(limit=10)
    complexities = [int(payload["combination_size"]) for payload in selected]

    assert complexities == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


def test_validation_schedule_finishes_standalone_before_family_rotation(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    selected_ids = (
        "fractal_high_breakout",
        "bullish_engulfing",
        "rsi_oversold",
        "relative_volume_expansion",
    )
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "validation_schedule",
        registry={block_id: source[block_id] for block_id in selected_ids},
    )
    factory.materialize_batch(
        batch_size=20,
        complexities=(1, 2),
        timeframes=("15m", "1h"),
    )

    first = factory.validation_schedule(cycle=1)
    assert first["phase"] == "STANDALONE_FIRST"
    assert first["complexity"] == 1
    assert first["family"] is None

    one_block = factory.queued_strategies(
        limit=100,
        complexity=1,
    )
    factory.update_strategy_status(
        (row["strategy_dna_hash"] for row in one_block),
        status="BASELINE_COMPLETED",
        reason="TEST",
    )
    second = factory.validation_schedule(cycle=1)
    assert second["phase"] == "FAMILY_ROUND_ROBIN"
    assert second["complexity"] is None
    assert second["family"] == "MARKET_STRUCTURE"
    assert second["family_rotation"][:3] == [
        "MARKET_STRUCTURE",
        "CANDLE",
        "MOMENTUM",
    ]


def test_complexity_specific_backtest_batch_remains_supported(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "complexity_specific",
        registry={
            block_id: source[block_id]
            for block_id in (
                "rsi_oversold",
                "rsi_recovery",
                "donchian20_breakout",
                "price_above_ema200",
            )
        },
    )
    factory.materialize_batch(
        batch_size=20,
        complexities=(1, 2, 3),
        timeframes=("15m", "1h"),
    )

    selected = factory.queued_strategies(
        limit=3,
        complexity=2,
    )

    assert len(selected) == 3
    assert all(int(payload["combination_size"]) == 2 for payload in selected)


def test_simple_factory_reconciles_canonical_results(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    registry = {"rsi_oversold": source["rsi_oversold"]}
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "simple",
        registry=registry,
    )
    factory.materialize_batch(
        batch_size=2,
        complexities=(1,),
        timeframes=("1h",),
    )
    queued = factory.queued_strategies(limit=1)[0]
    strategy_hash = queued["strategy_dna_hash"]
    experiment_hash = "a" * 64
    baseline = {
        "source": "FAST_SCREEN_REAL",
        "result_type": "BASELINE_SCREEN",
        "strategy_dna_hash": strategy_hash,
        "experiment_hash": experiment_hash,
        "parameter_hash": "b" * 64,
        "block_ids": ["rsi_oversold"],
        "families": ["MOMENTUM"],
        "assets_tested": ["BTC-EUR"],
        "timeframes_tested": ["1h"],
        "data_period": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2026-01-01T00:00:00Z",
        },
        "metrics": {
            "net_return": 0.10,
            "trade_count": 12,
        },
        "integrity": {
            "no_lookahead": True,
            "basic_costs_applied": True,
        },
        "screening": {
            "screening_return": 0.10,
            "profit_factor": 1.40,
            "trades": 12,
            "signal_funnel": {
                "raw_entry_signal_count": 20,
                "completed_round_trip_count": 12,
                "per_market": {
                    "BTC-EUR": {
                        "tradable_bar_count": 1000,
                        "raw_entry_signal_count": 20,
                        "edge_trigger_count": 15,
                        "blocked_existing_position": 3,
                        "blocked_risk": 0,
                        "completed_round_trip_count": 12,
                        "average_holding_bars": 8.5,
                        "net_return": 0.10,
                        "profit_factor": 1.40,
                    }
                },
            },
        },
    }

    class FakeDatabase:
        @staticmethod
        def fetch_recent_records(
            table_name: str,
            *,
            limit: int,
        ) -> list[dict]:
            del limit
            return [{"payload": baseline}] if table_name == "experiment_trials" else []

    result = factory.reconcile_canonical_results(
        FakeDatabase(),
        strategy_hashes=[strategy_hash],
    )
    assert result["strategy_count_with_evidence"] == 1
    assert result["positive_after_costs_count"] == 1
    queue = factory.queue_status(complexities=(1,))
    assert queue["status_counts"]["BASELINE_COMPLETED"] == 1
    indicator_rows = pd.read_csv(tmp_path / "simple" / "single_indicator_results.csv")
    assert indicator_rows.loc[0, "profit_factor"] == 1.4
    funnel_rows = pd.read_csv(tmp_path / "simple" / "signal_funnels.csv")
    assert funnel_rows.loc[0, "completed_round_trips"] == 12
    assert (tmp_path / "simple" / "canonical_result_evidence.json").is_file()
    evidence = json.loads(
        (tmp_path / "simple" / "canonical_result_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["rows"][0]["strategy_variant_dna_hash"]
    assert "strategy_variant_dna_hash" in indicator_rows.columns
    audit = json.loads(
        (tmp_path / "simple" / "objective_completion_audit.json").read_text(encoding="utf-8")
    )
    assert audit["invariants"]["canonical_results_reconciled"]
    assert audit["invariants"]["research_orders_zero"]

    second_baseline = {
        **baseline,
        "experiment_hash": "c" * 64,
        "parameter_hash": "d" * 64,
        "timeframes_tested": ["4h"],
    }

    class SecondFakeDatabase:
        @staticmethod
        def fetch_recent_records(
            table_name: str,
            *,
            limit: int,
        ) -> list[dict]:
            del limit
            return [{"payload": second_baseline}] if table_name == "experiment_trials" else []

    second_result = factory.reconcile_canonical_results(
        SecondFakeDatabase(),
        strategy_hashes=[strategy_hash],
    )
    assert second_result["experiment_count"] == 1
    assert second_result["cumulative_experiment_count"] == 2
    cumulative = json.loads(
        (tmp_path / "simple" / "canonical_result_evidence.json").read_text(encoding="utf-8")
    )
    assert len(cumulative["rows"]) == 2


def test_simple_factory_harvests_registered_canonical_results(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    registry = {
        "price_above_ema20": source["price_above_ema20"],
        "rsi_oversold": source["rsi_oversold"],
    }
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "simple",
        registry=registry,
    )
    factory.materialize_batch(
        batch_size=2,
        complexities=(1,),
        timeframes=("1h",),
    )
    queued = factory.queued_strategies(limit=1)[0]
    strategy_hash = queued["strategy_dna_hash"]
    discovered_pair = factory.generator.materialize_membership(
        ("price_above_ema20", "rsi_oversold"),
        logic_mode=LogicMode.LAYERED,
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )
    baseline = {
        "source": "FAST_SCREEN_REAL",
        "result_type": "BASELINE_SCREEN",
        "strategy_dna_hash": strategy_hash,
        "experiment_hash": "e" * 64,
        "parameter_hash": "f" * 64,
        "block_ids": ["rsi_oversold"],
        "families": ["MOMENTUM"],
        "assets_tested": ["BTC-EUR"],
        "timeframes_tested": ["1h"],
        "data_period": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2026-01-01T00:00:00Z",
        },
        "metrics": {
            "net_return": 0.12,
            "trade_count": 15,
        },
        "screening": {
            "screening_return": 0.12,
            "profit_factor": 1.35,
            "trades": 15,
            "signal_funnel": {
                "completed_round_trip_count": 15,
                "per_market": {
                    "BTC-EUR": {
                        "completed_round_trip_count": 15,
                        "profit_factor": 1.35,
                        "net_return": 0.12,
                    }
                },
            },
        },
    }
    pair_baseline = {
        **baseline,
        "strategy_dna_hash": (discovered_pair.strategy_dna_hash),
        "experiment_hash": "2" * 64,
        "block_ids": list(discovered_pair.block_ids),
    }
    unrelated = {
        **baseline,
        "strategy_dna_hash": "0" * 64,
        "experiment_hash": "1" * 64,
    }

    class FakeDatabase:
        @staticmethod
        def fetch_recent_records(
            table_name: str,
            *,
            limit: int,
        ) -> list[dict]:
            del limit
            return (
                [
                    {"payload": baseline},
                    {"payload": pair_baseline},
                    {"payload": unrelated},
                ]
                if table_name == "experiment_trials"
                else []
            )

    result = factory.reconcile_available_canonical_results(FakeDatabase())
    assert result["candidate_strategy_count"] == 3
    assert result["registered_candidate_count"] == 2
    assert result["registered_from_canonical_count"] == 2
    assert result["strategy_count_with_evidence"] == 2
    assert result["positive_after_costs_count"] == 2
    two_block_rows = pd.read_csv(tmp_path / "simple" / "two_block_results.csv")
    assert len(two_block_rows) == 1
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0


def test_simple_factory_harvests_old_canonical_results_by_cursor(
    isolated_settings,
    tmp_path,
) -> None:
    source = signal_block_registry()
    registry = {
        "fractal_high_breakout": source["fractal_high_breakout"],
        "relative_volume_expansion": source["relative_volume_expansion"],
    }
    factory = SimpleStrategyResearchFactory(
        isolated_settings,
        output_dir=tmp_path / "historical_harvest",
        registry=registry,
    )
    single = factory.generator.materialize_membership(
        ("fractal_high_breakout",),
        logic_mode=LogicMode.LAYERED,
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("4h",),
    )
    pair = factory.generator.materialize_membership(
        (
            "fractal_high_breakout",
            "relative_volume_expansion",
        ),
        logic_mode=LogicMode.LAYERED,
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("4h",),
    )

    def payload_for(combination, experiment_hash: str) -> dict:
        return {
            "source": "FAST_SCREEN_REAL",
            "strategy_dna_hash": combination.strategy_dna_hash,
            "experiment_hash": experiment_hash,
            "parameter_hash": "f" * 64,
            "block_ids": list(combination.block_ids),
            "families": ["FRACTAL"],
            "assets_tested": ["BTC-EUR"],
            "timeframes_tested": ["4h"],
            "data_period": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
            },
            "metrics": {
                "net_return": 0.04,
                "trade_count": 9,
            },
            "screening": {
                "screening_return": 0.04,
                "profit_factor": 1.2,
                "trades": 9,
                "signal_funnel": {
                    "completed_round_trip_count": 9,
                    "per_market": {
                        "BTC-EUR": {
                            "completed_round_trip_count": 9,
                            "profit_factor": 1.2,
                            "net_return": 0.04,
                        }
                    },
                },
            },
        }

    historical_records = [
        {"id": 7, "payload": payload_for(single, "7" * 64)},
        {"id": 8, "payload": payload_for(pair, "8" * 64)},
    ]

    class HistoricalDatabase:
        @staticmethod
        def fetch_recent_records(
            table_name: str,
            *,
            limit: int,
        ) -> list[dict]:
            del table_name, limit
            return []

        @staticmethod
        def fetch_records_after_id(
            table_name: str,
            *,
            after_id: int,
            limit: int,
        ) -> list[dict]:
            del limit
            return (
                [record for record in historical_records if int(record["id"]) > after_id]
                if table_name == "experiment_trials"
                else []
            )

        @staticmethod
        def fetch_records_by_payload_values(
            table_name: str,
            *,
            key: str,
            values,
        ) -> list[dict]:
            assert key == "strategy_dna_hash"
            selected = set(values)
            return (
                [
                    record
                    for record in historical_records
                    if record["payload"]["strategy_dna_hash"] in selected
                ]
                if table_name == "experiment_trials"
                else []
            )

    result = factory.reconcile_available_canonical_results(HistoricalDatabase())

    assert result["historical_records_scanned"] == 2
    assert result["strategy_count_with_evidence"] == 2
    assert result["registered_from_canonical_count"] == 2
    cursor = json.loads(
        (
            tmp_path
            / "historical_harvest"
            / (f"canonical_harvest_cursor_{factory.registry_hash[:16]}.json")
        ).read_text(encoding="utf-8")
    )
    assert cursor["positions"]["experiment_trials"] == 8
    fractal_rows = pd.read_csv(tmp_path / "historical_harvest" / "fractal_results.csv")
    assert set(fractal_rows["block_ids"].str.count(r"\|") + 1) == {1, 2}


def test_simple_lab_cli_exposes_resource_batches_and_backtests() -> None:
    parser = build_parser()
    generated = parser.parse_args(
        [
            "simple-lab",
            "generate",
            "--complexities",
            "1,2,3,4,5",
            "--timeframes",
            "15m,1h,4h,1d",
            "--batch-size",
            "500",
        ]
    )
    assert generated.simple_lab_command == "generate"
    assert generated.batch_size == 500
    backtest = parser.parse_args(
        [
            "simple-lab",
            "backtest-family",
            "--family",
            "VOLUME_FLOW",
            "--timeframes",
            "15m,1h,4h,1d",
        ]
    )
    assert backtest.simple_lab_command == "backtest-family"
    assert backtest.family == "VOLUME_FLOW"
    default_backtest = parser.parse_args(["simple-lab", "backtest"])
    assert default_backtest.timeframes == "all"
    assert default_backtest.minimum_exact_history_days == 365.0
    assert default_backtest.max_markets_per_exact_cycle == 0
    assert default_backtest.minimum_optimization_trades == 8
    run = parser.parse_args(
        [
            "simple-lab",
            "run",
            "--continuous",
            "--generation-batch-size",
            "20000",
            "--backtest-batch-size",
            "24",
        ]
    )
    assert run.simple_lab_command == "run"
    assert run.continuous
    assert run.generation_batch_size == 20_000
    assert run.complexities == "1,2,3,4,5"
    assert run.minimum_exact_history_days == 365.0
    assert run.max_markets_per_exact_cycle == 1
    assert run.minimum_optimization_trades == 8


def test_simple_lab_exact_history_rows_cover_one_calendar_year() -> None:
    assert (
        _simple_lab_history_rows(
            timeframe="15m",
            requested_rows=1_000,
            minimum_history_days=365.0,
        )
        == 35_041
    )
    assert (
        _simple_lab_history_rows(
            timeframe="1h",
            requested_rows=1_000,
            minimum_history_days=365.0,
        )
        == 8_761
    )
    assert (
        _simple_lab_history_rows(
            timeframe="4h",
            requested_rows=1_000,
            minimum_history_days=365.0,
        )
        == 2_191
    )


def test_simple_lab_requested_timeframes_preserves_focus() -> None:
    assert _simple_lab_requested_timeframes("1h,2h,1h") == (
        "1h",
        "2h",
    )
    all_timeframes = _simple_lab_requested_timeframes("all")
    assert "1h" in all_timeframes
    assert "2h" in all_timeframes


def test_simple_lab_stopped_heartbeat_does_not_defer_new_backtests() -> None:
    assert (
        _simple_lab_canonical_candidate_pids(
            {},
            {"pid": 1234, "status": "STOPPED"},
        )
        == ()
    )
    assert _simple_lab_canonical_candidate_pids(
        {"pid": 2345},
        {"pid": 1234, "status": "RUNNING"},
    ) == (2345, 1234)


def test_simple_lab_exact_market_cycle_rotates_without_duplication() -> None:
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR")
    assert _simple_lab_market_cycle(
        markets,
        maximum_markets=1,
        cycle_offset=0,
    ) == ["BTC-EUR"]
    assert _simple_lab_market_cycle(
        markets,
        maximum_markets=1,
        cycle_offset=1,
    ) == ["ETH-EUR"]
    assert _simple_lab_market_cycle(
        markets,
        maximum_markets=2,
        cycle_offset=2,
    ) == ["SOL-EUR", "BTC-EUR"]


def test_simple_lab_expands_dynamic_top50_research_markets(tmp_path) -> None:
    path = tmp_path / "top50_eligibility.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "rank": 1,
                        "eur_spot_market": "BTC-EUR",
                        "research_eligibility": "RESEARCH_ELIGIBLE",
                    },
                    {
                        "rank": 3,
                        "eur_spot_market": "USDT-EUR",
                        "research_eligibility": "CONTEXT_ONLY",
                    },
                    {
                        "rank": 4,
                        "eur_spot_market": "XRP-EUR",
                        "research_eligibility": "RESEARCH_ELIGIBLE",
                    },
                    {
                        "rank": 5,
                        "eur_spot_market": None,
                        "research_eligibility": "RESEARCH_ELIGIBLE",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _simple_lab_requested_markets(
        "TOP50_RESEARCH,NPC-EUR,BTC-EUR",
        top50_eligibility_path=path,
    ) == ("BTC-EUR", "XRP-EUR", "NPC-EUR")
    assert _simple_lab_requested_markets("ETH-EUR") == ("ETH-EUR",)
    assert _simple_lab_requested_markets(None) is None
