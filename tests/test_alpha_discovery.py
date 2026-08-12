from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.alpha_discovery as alpha_discovery
from config.settings import PathSettings, Settings
from research.alpha_discovery import (
    EconomicEdgePolicy,
    FailedFamilyArchive,
    FailedHypothesisRecord,
    FamilyClassification,
    HypothesisRegistry,
    build_alpha_discovery_artifact,
    build_point_in_time_panel,
    diagnose_p1_alpha_failure,
    family_desired_weights,
    forward_candidate_gate,
    initial_hypothesis_cards,
    liquidity_profiles,
    panel_causality_check,
    parameter_grid,
    simulate_panel_stage0,
    stage0_baselines,
)
from research.research_factory import SharedCostModel


def _frames(rows: int = 1_400) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2022-01-01", periods=rows, freq="4h", tz="UTC")
    output = {}
    for number, market in enumerate(
        ("BTC-EUR", "ETH-EUR", "SOL-EUR", "ADA-EUR", "XRP-EUR")
    ):
        rng = np.random.default_rng(100 + number)
        returns = (
            0.00025
            + 0.003 * np.sin(np.arange(rows) / 37.0 + number)
            + rng.normal(0.0, 0.007, rows)
        )
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        output[market] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close,
                "volume": (1_000.0 + number * 250.0)
                * (1.0 + 0.10 * np.sin(np.arange(rows) / 11.0 + number)),
            },
            index=index,
        )
    return output


def _costs(*, fee: float = 0.0025) -> SharedCostModel:
    return SharedCostModel(
        cost_model_version=f"test-cost-{fee}",
        maker_fee_fraction=0.0015,
        taker_fee_fraction=fee,
        spread_bps=5.0,
        slippage_bps=8.0,
    )


def test_hypothesis_cards_require_mechanism_and_reject_duplicates() -> None:
    cards = initial_hypothesis_cards()
    registry = HypothesisRegistry(cards)
    assert len(cards) == 8
    assert len(registry.registry_hash) == 48
    assert all(card.primary_entry_timeframe != "15m" for card in cards)
    assert sum(card.parameter_region_count for card in cards) == 122
    with pytest.raises(ValueError, match="duplicate hypothesis ID"):
        HypothesisRegistry((cards[0], cards[0]))
    with pytest.raises(ValueError, match="substantive economic mechanism"):
        replace(cards[0], hypothesis_id="bad", economic_mechanism="EMA plus RSI")
    with pytest.raises(ValueError, match="15m"):
        replace(
            cards[0],
            hypothesis_id="bad-15m",
            primary_entry_timeframe="15m",
            target_timeframes=("15m",),
        )


def test_derivatives_context_is_information_only() -> None:
    card = initial_hypothesis_cards()[-1]
    assert card.family == "DERIVATIVES_CONTEXT_MODIFIER"
    assert all(value.startswith("INFORMATION_ONLY:") for value in card.information_only_inputs)
    with pytest.raises(ValueError, match="INFORMATION_ONLY"):
        replace(
            card,
            hypothesis_id="unsafe-derivatives",
            information_only_inputs=("funding",),
        )


def test_minimum_economic_edge_is_cost_horizon_and_liquidity_aware() -> None:
    tier1 = EconomicEdgePolicy.from_costs(
        _costs(), holding_period_hours=72, liquidity_tier="TIER_1"
    )
    tier3 = EconomicEdgePolicy.from_costs(
        _costs(), holding_period_hours=12, liquidity_tier="TIER_3"
    )
    assert tier1.roundtrip_cost_bps == pytest.approx(71.0)
    assert tier3.minimum_move_cost_ratio > tier1.minimum_move_cost_ratio
    assert tier1.assess(300.0)["economically_large_enough"]
    assert not tier1.assess(100.0)["economically_large_enough"]


def test_point_in_time_panel_never_ranks_asset_before_history() -> None:
    frames = _frames()
    frames["SOL-EUR"] = frames["SOL-EUR"].iloc[300:]
    panel = build_point_in_time_panel(frames, timeframe="4h", minimum_history_bars=120)
    sol_first = frames["SOL-EUR"].index[0]
    assert not panel.eligible.loc[:sol_first, "SOL-EUR"].any()
    assert panel.eligible["SOL-EUR"].sum() == len(frames["SOL-EUR"]) - 119
    profiles = liquidity_profiles(frames)
    assert {row.tier for row in profiles} <= {"TIER_1", "TIER_2", "TIER_3"}
    assert all(row.point_in_time_status.startswith("PIT_") for row in profiles)


@pytest.mark.parametrize(
    "family",
    [
        "CROSS_SECTIONAL_MOMENTUM",
        "MEDIUM_TERM_TREND_PULLBACK",
        "VOLATILITY_CONTRACTION_EXPANSION",
        "QUALITY_CONSOLIDATION_BREAKOUT",
        "BTC_RELATIVE_ALT_ROTATION",
        "BREADTH_CONDITIONED_MOMENTUM",
        "SLOW_VOLUME_ACCUMULATION",
    ],
)
def test_each_supported_family_is_causal_and_isolated(family: str) -> None:
    panel = build_point_in_time_panel(_frames(), timeframe="4h")
    card = next(card for card in initial_hypothesis_cards() if card.family == family)
    parameters = parameter_grid(card)[0]
    weights = family_desired_weights(panel, family=family, parameters=parameters)
    assert weights.shape == panel.close.shape
    assert (weights.sum(axis=1) <= 0.20 + 1e-12).all()
    assert panel_causality_check(
        panel, family=family, parameters=parameters
    )["status"] == "PASSED"
    policy = EconomicEdgePolicy.from_costs(
        _costs(), holding_period_hours=72, liquidity_tier="TIER_1"
    )
    result = simulate_panel_stage0(
        panel,
        weights,
        card=card,
        parameters=parameters,
        costs=_costs(),
        edge_policy=policy,
    )
    assert result.family == family
    assert result.hypothesis_id == card.hypothesis_id
    assert result.authority == "APPROXIMATE_RESEARCH_ONLY"
    assert result.annualized_turnover >= 0.0
    assert result.trades_per_week >= 0.0


def test_gross_net_turnover_move_metrics_and_baselines() -> None:
    panel = build_point_in_time_panel(_frames(), timeframe="4h")
    card = initial_hypothesis_cards()[0]
    parameters = parameter_grid(card)[0]
    weights = family_desired_weights(
        panel, family=card.family, parameters=parameters
    )
    policy = EconomicEdgePolicy.from_costs(
        _costs(), holding_period_hours=72, liquidity_tier="TIER_1"
    )
    result = simulate_panel_stage0(
        panel,
        weights,
        card=card,
        parameters=parameters,
        costs=_costs(),
        edge_policy=policy,
    )
    assert result.net_pnl_eur < result.gross_pnl_eur
    assert result.turnover > 0
    assert result.median_mfe_bps is not None
    assert result.median_mae_bps is not None
    assert result.expected_move_cost_ratio is not None
    assert result.cost_as_fraction_of_gross_opportunity is not None
    baselines = stage0_baselines(panel, costs=_costs())
    assert set(baselines) == {
        "BTC_BUY_AND_HOLD",
        "PIT_EQUAL_WEIGHT_ELIGIBLE",
        "BTC_EXPOSURE_MATCHED_20PCT",
        "PIT_EQUAL_WEIGHT_EXPOSURE_MATCHED_20PCT",
        "CASH",
    }
    assert baselines["CASH"]["net_return"] == 0.0


def test_stage0_episode_defers_exit_across_missing_asset_bar() -> None:
    frames = _frames(rows=240)
    missing_time = frames["ETH-EUR"].index[120]
    frames["ETH-EUR"].loc[missing_time, ["open", "high", "low", "close"]] = np.nan
    panel = build_point_in_time_panel(
        frames, timeframe="4h", minimum_history_bars=20
    )
    weights = pd.DataFrame(0.0, index=panel.open.index, columns=panel.markets)
    weights.loc[weights.index[100:120], "ETH-EUR"] = 0.20

    episodes = alpha_discovery._position_episodes(
        panel,
        weights,
        roundtrip_cost_fraction=0.0071,
    )

    assert len(episodes) == 1
    assert episodes[0]["holding_bars"] == 21.0
    assert all(np.isfinite(value) for value in episodes[0].values())


def test_zero_activity_has_no_fabricated_asset_or_regime_robustness() -> None:
    panel = build_point_in_time_panel(_frames(rows=240), timeframe="4h")
    weights = pd.DataFrame(0.0, index=panel.open.index, columns=panel.markets)

    diagnostics = alpha_discovery.panel_asset_and_regime_diagnostics(
        panel,
        weights,
        costs=_costs(),
    )

    assert diagnostics["asset_classification"] == "NOT_EVALUABLE"
    assert diagnostics["regime_classification"] == "NOT_EVALUABLE"


def test_contraction_and_quality_breakout_require_structural_preconditions() -> None:
    frames = _frames(rows=700)
    index = frames["ETH-EUR"].index
    for market, frame in frames.items():
        base = 100.0 + 0.8 * np.sin(np.arange(len(index)) / 9.0 + len(market))
        frame.loc[:, "open"] = np.r_[base[0], base[:-1]]
        frame.loc[:, "close"] = base
        frame.loc[:, "high"] = np.maximum(frame["open"], frame["close"]) + 0.20
        frame.loc[:, "low"] = np.minimum(frame["open"], frame["close"]) - 0.20
        frame.loc[:, "volume"] = 1_000.0
    # Index 546 is a declared seven-day rebalance boundary on 4h bars.
    for market in frames:
        frames[market].iloc[500:546, frames[market].columns.get_loc("open")] = 100.0
        frames[market].iloc[500:546, frames[market].columns.get_loc("close")] = 100.0
        frames[market].iloc[500:546, frames[market].columns.get_loc("high")] = 100.1
        frames[market].iloc[500:546, frames[market].columns.get_loc("low")] = 99.9
    frames["ETH-EUR"].iloc[546, frames["ETH-EUR"].columns.get_loc("close")] = 103.0
    frames["ETH-EUR"].iloc[546, frames["ETH-EUR"].columns.get_loc("high")] = 103.2
    frames["ETH-EUR"].iloc[546, frames["ETH-EUR"].columns.get_loc("volume")] = 2_000.0
    panel = build_point_in_time_panel(frames, timeframe="4h", minimum_history_bars=20)

    contraction = next(
        card
        for card in initial_hypothesis_cards()
        if card.family == "VOLATILITY_CONTRACTION_EXPANSION"
    )
    contraction_parameters = {
        "compression_days": 5,
        "baseline_days": 60,
        "compression_quantile": 0.30,
        "breakout_days": 10,
    }
    contraction_weights = family_desired_weights(
        panel,
        family=contraction.family,
        parameters=contraction_parameters,
    )
    assert contraction_weights.loc[index[546], "ETH-EUR"] > 0.0

    quality = next(
        card
        for card in initial_hypothesis_cards()
        if card.family == "QUALITY_CONSOLIDATION_BREAKOUT"
    )
    quality_weights = family_desired_weights(
        panel,
        family=quality.family,
        parameters={
            "consolidation_days": 5,
            "maximum_range_atr": 3.0,
            "relative_rank_minimum": 0.60,
            "volume_multiple": 1.5,
        },
    )
    assert quality_weights.loc[index[546], "ETH-EUR"] > 0.0


def test_gross_positive_net_negative_has_separate_classification() -> None:
    panel = build_point_in_time_panel(_frames(), timeframe="4h")
    card = initial_hypothesis_cards()[0]
    parameters = parameter_grid(card)[0]
    weights = family_desired_weights(panel, family=card.family, parameters=parameters)
    stressed = _costs(fee=0.05)
    policy = EconomicEdgePolicy.from_costs(
        stressed, holding_period_hours=72, liquidity_tier="TIER_1"
    )
    result = simulate_panel_stage0(
        panel,
        weights,
        card=card,
        parameters=parameters,
        costs=stressed,
        edge_policy=policy,
    )
    if result.gross_pnl_eur > 0:
        assert result.classification == FamilyClassification.GROSS_POSITIVE_NET_NEGATIVE
    assert "NEGATIVE_NET_EXPECTANCY" in result.rejection_reasons


def test_failed_family_archive_is_immutable_and_prevents_rediscovery(tmp_path) -> None:
    card = initial_hypothesis_cards()[0]
    record = FailedHypothesisRecord(
        hypothesis_id=card.hypothesis_id,
        card_hash=card.card_hash,
        family=card.family,
        tested_parameter_hashes=("parameter-v1",),
        economic_results_hash="result-v1",
        rejection_reasons=("NEGATIVE_NET_EXPECTANCY",),
        data_version="data-v1",
        cost_model_version="cost-v1",
        stage0_engine_version="panel-v1",
        recorded_at_data_cutoff="2026-01-01T00:00:00Z",
    )
    archive = FailedFamilyArchive(tmp_path / "failed.json")
    archive.append(record)
    archive.append(record)
    assert len(archive.records()) == 1
    with pytest.raises(ValueError, match="already archived"):
        archive.append(replace(record, economic_results_hash="rewritten-result"))


def test_forward_gate_never_promotes_stage0_only_evidence() -> None:
    rejected = forward_candidate_gate(
        {
            "exact_status": "STAGE0_PROMISING",
            "net_expectancy": 1.0,
            "profit_factor": 2.0,
            "maximum_drawdown": 0.1,
            "positive_walk_forward_folds": 6,
            "parameter_plateau": True,
            "cost_plus_25_net_expectancy": 1.0,
            "liquidity_status": "TIER_1",
            "lookahead_safe": True,
            "asset_status": "MULTI_ASSET_ROBUST",
            "regime_status": "ROBUST",
        }
    )
    assert rejected["state"] == "NOT_PROMOTED"
    assert rejected["stage0_only_promotion"] is False
    assert rejected["automatic_authority"] is False


def test_p1_failure_diagnosis_computes_cost_to_edge_ratio() -> None:
    p1 = {
        "stage0": {
            "aggregate_parameter_results": [
                {
                    "parameter_hash": "best",
                    "trade_count": 100,
                    "gross_pnl_eur": 10.0,
                    "net_pnl_eur": -40.0,
                    "profit_factor": 0.5,
                    "turnover_eur": 20_000.0,
                    "positive_asset_count": 0,
                    "positive_timeframe_count": 0,
                }
            ]
        },
        "benchmark": {
            "false_negative_review_sample": {
                "exact_result": {"metrics": {"average_mfe_r": 0.5, "average_mae_r": -0.7}}
            }
        },
    }
    diagnosis = diagnose_p1_alpha_failure(p1)
    assert diagnosis["gross_edge_per_trade_eur"] == pytest.approx(0.1)
    assert diagnosis["estimated_roundtrip_cost_per_trade_eur"] == pytest.approx(0.5)
    assert diagnosis["cost_to_positive_gross_edge_ratio"] == pytest.approx(5.0)
    assert "EDGE_TOO_SMALL_FOR_COSTS" in diagnosis["classification"]


def test_full_alpha_discovery_artifact_is_immutable_and_zero_authority(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    for timeframe, rows, frequency in (("4h", 900, "4h"), ("1d", 500, "1D")):
        index = pd.date_range("2020-01-01", periods=rows, freq=frequency, tz="UTC")
        for number, market in enumerate(
            ("BTC-EUR", "ETH-EUR", "SOL-EUR", "ADA-EUR", "XRP-EUR", "LINK-EUR", "DOGE-EUR", "LTC-EUR")
        ):
            close = 100.0 * np.exp(
                np.cumsum(
                    0.0003
                    + 0.002 * np.sin(np.arange(rows) / 17.0 + number)
                )
            )
            frame = pd.DataFrame(
                {
                    "timestamp": index,
                    "open": np.r_[close[0], close[:-1]],
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "volume": 1_000.0 + number * 100.0,
                }
            )
            path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
    p0_path = settings.paths.output_dir / "economics" / "runs" / "p05.json"
    p0_path.parent.mkdir(parents=True, exist_ok=True)
    p0_path.write_text(json.dumps({"artifact_hash": "p05"}), encoding="utf-8")
    (settings.paths.output_dir / "economics" / "latest.json").write_text(
        json.dumps({"artifact_path": str(p0_path)}), encoding="utf-8"
    )
    p1_path = settings.paths.output_dir / "research_factory" / "runs" / "p1.json"
    p1_path.parent.mkdir(parents=True, exist_ok=True)
    p1_path.write_text(
        json.dumps(
            {
                "p0_5_branch": {
                    "decision": "ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION"
                },
                "stage0": {
                    "aggregate_parameter_results": [
                        {
                            "parameter_hash": "best",
                            "trade_count": 100,
                            "gross_pnl_eur": 10.0,
                            "net_pnl_eur": -40.0,
                            "profit_factor": 0.5,
                            "turnover_eur": 20_000.0,
                            "positive_asset_count": 0,
                            "positive_timeframe_count": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (settings.paths.output_dir / "research_factory" / "latest.json").write_text(
        json.dumps({"artifact_path": str(p1_path)}), encoding="utf-8"
    )
    first = build_alpha_discovery_artifact(
        settings, maximum_rows_4h=900, maximum_rows_1d=500
    )
    second = build_alpha_discovery_artifact(
        settings, maximum_rows_4h=900, maximum_rows_1d=500
    )
    assert first["run_id"] == second["run_id"]
    assert first["artifact_hash"] == second["artifact_hash"]
    artifact = json.loads(Path(first["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["search_space"]["parameter_region_count"] == 120
    assert artifact["stage0"]["authority"] == "APPROXIMATE_RESEARCH_ONLY"
    assert artifact["ml_status"]["authority"] == "SHADOW_ONLY"
    assert artifact["portfolio_allocator"]["built"] is False
    assert artifact["forward_candidates"] == []
    assert all(value == 0 for value in artifact["safety"].values())
    archive = json.loads(
        (settings.paths.output_dir / "alpha_discovery" / "failed_hypotheses.json").read_text(
            encoding="utf-8"
        )
    )
    assert archive["record_count"] == artifact["failed_hypothesis_archive"]["record_count"]
