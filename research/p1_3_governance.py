"""Immutable P1.3 preregistration and guarded research handoff.

This module deliberately does not import a prospective data loader, a target
builder, an execution client, or a strategy promotion surface.  It may freeze
an experiment plan and authorize a future immutable-dataset run; it cannot run
research or mutate live authority by itself.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from config.settings import get_settings
from research.research_factory import SharedCostModel
from utils.common import (
    append_jsonl,
    parse_utc,
    read_json,
    sha256_file,
    stable_hash,
    stable_json,
    utc_iso,
)

PREREGISTRATION_VERSION = "P1_3_PREREGISTRATION_V1"
PREREGISTRATION_SCHEMA = "p1_3_preregistration_envelope_v1"
RESULT_TEMPLATE_VERSION = "P1_3_RESULTS_TEMPLATE_V1"
LEDGER_VERSION = "p1_3_append_only_research_ledger_v1"
RUNNER_VERSION = "p1_3_guarded_runner_v1"
MULTIPLE_TESTING_VERSION = "p1_3_bh_fdr_and_complexity_penalty_v1"
COST_SCENARIO_VERSION = "p1_3_executable_cost_scenarios_v1"
BACKTESTER_VERSION = "native_crypto_next_open_event_engine_v1"

PRIMARY_ASSETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR")
ALLOWED_SEEDS = (1729, 20260811, 314159)
ALLOWED_LAGS_SECONDS = (1, 5, 15, 30, 60)
FLOW_WINDOWS_SECONDS = (60, 300)
EVALUATION_HORIZONS_HOURS = (24, 72, 120)
RESEARCH_USABLE_STATES = {"RESEARCH_USABLE", "ROBUSTNESS_USABLE"}
IMMUTABLE_ID = re.compile(r"^[0-9a-f]{64}$")

TARGET_FIELD_NAMES = (
    "future_return",
    "forward_return",
    "future_bid",
    "future_ask",
    "mfe_label",
    "mae_label",
    "pnl_label",
)

RESULT_SECTIONS = (
    "baseline",
    "flow",
    "cross_venue",
    "l2",
    "costs",
    "ablation",
    "walk_forward",
    "holdout",
    "robustness",
    "failures",
)


class P13GovernanceError(RuntimeError):
    """Base class for fail-closed P1.3 governance errors."""


class ImmutableArtifactError(P13GovernanceError):
    """Raised when an immutable artifact is missing, changed, or collides."""


class ResearchGateError(P13GovernanceError):
    """Raised when a future research run does not satisfy its frozen gate."""


class HoldoutAccessError(ResearchGateError):
    """Raised when development code attempts to access protected holdout data."""


@dataclass(frozen=True, slots=True)
class PreregistrationBindings:
    git_commit: str
    git_worktree_state: str
    p1_2_3_artifact_hash: str
    p1_2_3_artifact_file_sha256: str
    p1_2_3_artifact_path: str
    p1_1_evidence_hash: str
    p1_1_evidence_file_sha256: str
    p1_1_evidence_path: str
    readiness_policy_version: str
    shared_cost_model: Mapping[str, Any]
    feature_schema_versions: Mapping[str, str]
    source_code_hashes: Mapping[str, str]


def _named_flow_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "profile": "AGGRESSIVE_NET_FLOW_SIGN_60S",
            "window_seconds": 60,
            "inputs": ["aggressive_buy_notional", "aggressive_sell_notional"],
            "rule": "NET_AGGRESSIVE_FLOW_MUST_AGREE_WITH_LONG_BASELINE",
            "threshold": "STRICT_SIGN_ONLY",
        },
        {
            "profile": "AGGRESSIVE_NET_FLOW_SIGN_300S",
            "window_seconds": 300,
            "inputs": ["aggressive_buy_notional", "aggressive_sell_notional"],
            "rule": "NET_AGGRESSIVE_FLOW_MUST_AGREE_WITH_LONG_BASELINE",
            "threshold": "STRICT_SIGN_ONLY",
        },
        {
            "profile": "CVD_SLOPE_SIGN_300S",
            "window_seconds": 300,
            "inputs": ["causal_cvd_change"],
            "rule": "CVD_SLOPE_MUST_BE_POSITIVE",
            "threshold": "STRICT_SIGN_ONLY",
        },
        {
            "profile": "BUY_SELL_IMBALANCE_MODERATE_60S",
            "window_seconds": 60,
            "inputs": ["aggressive_buy_share"],
            "rule": "AGGRESSIVE_BUY_SHARE_AT_OR_ABOVE_REGION",
            "threshold": 0.55,
        },
        {
            "profile": "BUY_SELL_IMBALANCE_STRONG_300S",
            "window_seconds": 300,
            "inputs": ["aggressive_buy_share"],
            "rule": "AGGRESSIVE_BUY_SHARE_AT_OR_ABOVE_REGION",
            "threshold": 0.65,
        },
        {
            "profile": "OFI_SIGN_60S_IF_SEMANTICALLY_VALID",
            "window_seconds": 60,
            "inputs": ["order_flow_imbalance"],
            "rule": "OFI_MUST_BE_POSITIVE",
            "threshold": "STRICT_SIGN_ONLY",
            "availability_gate": "OFI_SEMANTICS_VALID_AT_DECISION_TIME",
        },
    )


def _named_l2_profiles() -> tuple[dict[str, Any], ...]:
    return (
        {
            "profile": "L2_PERMISSIVE",
            "maximum_spread_bps": 30.0,
            "minimum_directional_top_imbalance": -0.80,
            "maximum_adverse_microprice_displacement_bps": 3.0,
            "maximum_feature_age_seconds": 5,
        },
        {
            "profile": "L2_BALANCED",
            "maximum_spread_bps": 15.0,
            "minimum_directional_top_imbalance": -0.60,
            "maximum_adverse_microprice_displacement_bps": 2.0,
            "maximum_feature_age_seconds": 2,
        },
        {
            "profile": "L2_STRICT",
            "maximum_spread_bps": 5.0,
            "minimum_directional_top_imbalance": -0.40,
            "maximum_adverse_microprice_displacement_bps": 1.0,
            "maximum_feature_age_seconds": 1,
        },
    )


def build_variant_catalog() -> tuple[dict[str, Any], ...]:
    """Return the complete, bounded A-E trial catalog."""

    variants: list[dict[str, Any]] = [
        {
            "variant_id": "A_BASELINE",
            "ladder": "A",
            "hypothesis": "BASELINE_CONTROL",
            "parent": None,
            "parameters": {},
        }
    ]
    for flow in _named_flow_profiles():
        variants.append(
            {
                "variant_id": f"B_{flow['profile']}",
                "ladder": "B",
                "hypothesis": "H1_FLOW_CONFIRMED_SWING",
                "parent": "A_BASELINE",
                "parameters": {"bitvavo_flow": flow},
            }
        )
    b_variants = tuple(row for row in variants if row["ladder"] == "B")
    for flow_variant in b_variants:
        for lag in ALLOWED_LAGS_SECONDS:
            variants.append(
                {
                    "variant_id": f"C_{flow_variant['variant_id'][2:]}_KRAKEN_LAG_{lag}S",
                    "ladder": "C",
                    "hypothesis": "H2_KRAKEN_CONFIRMATION",
                    "parent": flow_variant["variant_id"],
                    "parameters": {
                        **flow_variant["parameters"],
                        "kraken_confirmation": {
                            "role": "CONFIRMATION_ONLY",
                            "lag_seconds": lag,
                            "rule": "SAME_FLOW_DIRECTION_AS_BITVAVO",
                        },
                    },
                }
            )
    c_variants = tuple(row for row in variants if row["ladder"] == "C")
    for flow_variant in c_variants:
        for l2 in _named_l2_profiles():
            variants.append(
                {
                    "variant_id": f"D_{flow_variant['variant_id'][2:]}_{l2['profile']}",
                    "ladder": "D",
                    "hypothesis": "H3_L2_EXECUTION_FILTER",
                    "parent": flow_variant["variant_id"],
                    "parameters": {**flow_variant["parameters"], "bitvavo_l2": l2},
                }
            )
    variants.extend(
        (
            {
                "variant_id": "E_CMC_BREADTH_SOFT_40",
                "ladder": "E",
                "hypothesis": "H5_BREADTH_MODIFIER",
                "parent": "FROZEN_DEVELOPMENT_CHAMPION_D",
                "parameters": {"breadth_floor": 0.40, "mode": "SOFT"},
            },
            {
                "variant_id": "E_CMC_BREADTH_SOFT_60",
                "ladder": "E",
                "hypothesis": "H5_BREADTH_MODIFIER",
                "parent": "FROZEN_DEVELOPMENT_CHAMPION_D",
                "parameters": {"breadth_floor": 0.60, "mode": "SOFT"},
            },
            {
                "variant_id": "E_DERIVATIVES_CROWDING_SOFT_24H",
                "ladder": "E",
                "hypothesis": "H6_DERIVATIVES_CONTEXT",
                "parent": "FROZEN_DEVELOPMENT_CHAMPION_D",
                "parameters": {"context_window_hours": 24, "mode": "SOFT"},
            },
            {
                "variant_id": "E_DERIVATIVES_CROWDING_SOFT_72H",
                "ladder": "E",
                "hypothesis": "H6_DERIVATIVES_CONTEXT",
                "parent": "FROZEN_DEVELOPMENT_CHAMPION_D",
                "parameters": {"context_window_hours": 72, "mode": "SOFT"},
            },
        )
    )
    ids = [str(row["variant_id"]) for row in variants]
    if len(ids) != len(set(ids)):
        raise AssertionError("P1.3 variant IDs must be unique")
    return tuple(variants)


def build_empty_results_template() -> dict[str, Any]:
    """Return a performance-empty schema instance for a future P1.3 run."""

    sections = {
        name: {
            "status": "NOT_EVALUATED",
            "metrics": None,
            "artifact_refs": [],
            "failure_reasons": [],
        }
        for name in RESULT_SECTIONS
    }
    return {
        "schema_version": RESULT_TEMPLATE_VERSION,
        "preregistration_id": None,
        "dataset_freeze_id": None,
        "candidate_freeze_hash": None,
        "seed": None,
        "result_hash": None,
        "performance_fields_populated": False,
        "sections": sections,
        "authority": {
            "research_only": True,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "orders_generated": 0,
        },
    }


def _cost_scenarios(shared_cost_model: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        "fee_semantics": "MARKETABLE_NEXT_OPEN_TAKER_BOTH_SIDES",
        "taker_fee_bps_per_side": 25.0,
        "maker_fee_bps_per_side": 15.0,
        "spread_bps_full": 5.0,
        "slippage_bps_per_side": 8.0,
        "size_penalty_bps_per_side": 0.0,
        "latency_penalty_bps_per_side": 0.0,
        "failed_or_nonfill_allowance_bps_per_side": 0.0,
        "partial_fill_penalty_bps_per_side": 0.0,
        "modeled_roundtrip_bps": 71.0,
    }
    realistic = {
        **base,
        "size_penalty_bps_per_side": 3.0,
        "latency_penalty_bps_per_side": 2.0,
        "failed_or_nonfill_allowance_bps_per_side": 2.0,
        "modeled_roundtrip_bps": 85.0,
    }
    stress = {
        **base,
        "spread_bps_full": 10.0,
        "slippage_bps_per_side": 16.0,
        "size_penalty_bps_per_side": 6.0,
        "latency_penalty_bps_per_side": 5.0,
        "failed_or_nonfill_allowance_bps_per_side": 4.0,
        "partial_fill_penalty_bps_per_side": 4.0,
        "modeled_roundtrip_bps": 130.0,
    }
    scenarios = {
        "BASE_COST": base,
        "REALISTIC_COST": realistic,
        "STRESS_COST": stress,
    }
    return {
        "version": f"{COST_SCENARIO_VERSION}:{stable_hash(scenarios, length=20)}",
        "shared_repository_model": dict(shared_cost_model),
        "scenarios": scenarios,
        "executable_path": (
            "entry at next executable ask or modeled fill; exit at executable bid or modeled "
            "fill; use the more conservative of observed spread and scenario spread"
        ),
        "mid_to_future_mid_is_primary": False,
    }


def build_preregistration_plan(bindings: PreregistrationBindings) -> dict[str, Any]:
    """Build the outcome-blind P1.3 plan from metadata-only inputs."""

    variants = build_variant_catalog()
    hypothesis_ids = (
        "H1_FLOW_CONFIRMED_SWING",
        "H2_KRAKEN_CONFIRMATION",
        "H3_L2_EXECUTION_FILTER",
        "H4_LIQUIDITY_WITHDRAWAL",
        "H5_BREADTH_MODIFIER",
        "H6_DERIVATIVES_CONTEXT",
    )
    ladder_counts = {
        ladder: sum(row["ladder"] == ladder for row in variants)
        for ladder in ("A", "B", "C", "D", "E")
    }
    return {
        "preregistration_version": PREREGISTRATION_VERSION,
        "research_state": "DESIGN_FROZEN_RESEARCH_NOT_STARTED",
        "primary_question": (
            "CAN PROSPECTIVE MICROSTRUCTURE AND CROSS-VENUE INFORMATION IMPROVE THE NET "
            "ECONOMICS OF AN OTHERWISE FIXED SWING SETUP?"
        ),
        "bindings": {
            **asdict(bindings),
            "backtester_version": BACKTESTER_VERSION,
            "runner_version": RUNNER_VERSION,
            "result_template_version": RESULT_TEMPLATE_VERSION,
        },
        "baseline": {
            "generation": "P1_1_MEDIUM_TERM_TREND_PULLBACK_BASELINE_V1",
            "selection_basis": "PRIOR_P1_1_EVIDENCE_ONLY_NO_PROSPECTIVE_TARGET_ACCESS",
            "prior_status": "EXACT_REJECTED_NOT_PROMOTED",
            "family": "MEDIUM_TERM_TREND_PULLBACK",
            "hypothesis_id": "p1.1_medium_term_trend_pullback_v1",
            "strategy_adapter": "P1_1_FROZEN_PANEL_SIGNAL",
            "signal_logic": {
                "timeframe": "4h",
                "context_timeframe": "1d",
                "trend_days": 20,
                "continuation_days": 3,
                "pullback_atr_minimum": 0.5,
                "pullback_atr_maximum": 2.5,
                "exit_days": 5,
                "top_n": 2,
                "rule": (
                    "close above 20d EMA; positive trend strength; close is 0.5-2.5 ATR "
                    "below the prior 3d high; close above its 2.5d lag; select top two positive "
                    "trend-strength eligible assets"
                ),
            },
            "entry_semantics": "closed 4h signal; native engine execution at next 4h open",
            "exit_semantics": "panel deselection, stop, target, trailing stop, or time exit",
            "stop_atr": 3.0,
            "target_atr": 6.0,
            "trailing_atr": 3.0,
            "maximum_holding_bars": 84,
            "expected_holding_period": "2_TO_10_DAYS_WITH_14_DAY_HARD_CAP",
            "position_sizing": {
                "maximum_research_exposure": 0.20,
                "equal_weight_across_at_most_two_selected_assets": True,
                "risk_per_trade_fraction": 0.005,
                "maximum_total_open_risk_fraction": 0.02,
                "maximum_position_fraction": 0.25,
                "maximum_portfolio_exposure_fraction": 0.75,
                "reserve_cash_fraction": 0.10,
            },
            "frozen_after_start": True,
            "replacement_requires_new_preregistration_generation": True,
        },
        "hypotheses": {
            "H1_FLOW_CONFIRMED_SWING": {
                "question": "Does causal Bitvavo spot flow confirmation improve the fixed setup?",
                "role": "CONFIRMATION_FILTER",
            },
            "H2_KRAKEN_CONFIRMATION": {
                "question": "Does independent Kraken spot flow reduce false Bitvavo entries?",
                "role": "CONFIRMATION_ONLY_NO_EXECUTION",
            },
            "H3_L2_EXECUTION_FILTER": {
                "question": "Can valid Bitvavo L2 avoid economically poor entries?",
                "role": "ENTRY_AND_EXECUTION_FILTER_NOT_STANDALONE_ALPHA",
            },
            "H4_LIQUIDITY_WITHDRAWAL": {
                "question": (
                    "Does pre-entry liquidity withdrawal predict continuation or only worse "
                    "execution?"
                ),
                "measurement": (
                    "separate matched-cost directional outcome from spread, slippage, fill and "
                    "entry-delay deterioration"
                ),
            },
            "H5_BREADTH_MODIFIER": {
                "question": "Does broad crypto participation improve the fixed spot setup?",
                "activation": "CMC_BREADTH_RESEARCH_USABLE_FREEZE_ONLY",
            },
            "H6_DERIVATIVES_CONTEXT": {
                "question": "Does derivatives positioning improve or harm spot entry selection?",
                "activation": "MEXC_DERIVATIVES_CONTEXT_RESEARCH_USABLE_FREEZE_ONLY",
                "role": "INFORMATION_ONLY_NO_DERIVATIVE_EXECUTION",
            },
        },
        "ablation_ladder": {
            "A": "FIXED_HTF_BASELINE",
            "B": "A_PLUS_BITVAVO_FLOW",
            "C": "B_PLUS_KRAKEN_FLOW_CONFIRMATION",
            "D": "C_PLUS_BITVAVO_L2_EXECUTION_FILTER",
            "E": "OPTIONAL_SEPARATELY_READY_CONTEXT_MODIFIER",
            "mandatory_comparisons": ["A_VS_B", "B_VS_C", "C_VS_D"],
            "fully_stacked_only_is_sufficient": False,
        },
        "market_and_time": {
            "assets": list(PRIMARY_ASSETS),
            "execution_venue": "bitvavo",
            "reference_venues": {"kraken": "RESEARCH_ONLY", "mexc": "CONTEXT_ONLY"},
            "baseline_timeframe": "4h",
            "context_timeframes": ["1d"],
            "microstructure_windows_seconds": list(FLOW_WINDOWS_SECONDS),
            "allowed_cross_venue_lags_seconds": list(ALLOWED_LAGS_SECONDS),
            "evaluation_horizons_hours": list(EVALUATION_HORIZONS_HOURS),
            "primary_horizon": "CANONICAL_BASELINE_EXIT_PATH",
            "hft_objective": False,
        },
        "causal_features": {
            "required_time_fields": [
                "observation_timestamp",
                "source_timestamp",
                "availability_timestamp",
                "feature_age_seconds",
            ],
            "decision_rule": "availability_timestamp <= decision_time",
            "bitvavo_flow": [
                "aggressive_buy_notional",
                "aggressive_sell_notional",
                "aggressive_buy_share",
                "causal_cvd_change",
                "trade_intensity",
                "order_flow_imbalance_if_semantically_valid",
            ],
            "kraken_flow": [
                "aggressive_buy_notional",
                "aggressive_sell_notional",
                "aggressive_buy_share",
                "causal_cvd_change",
            ],
            "bitvavo_l2_v2": [
                "spread_bps",
                "top_level_imbalance",
                "microprice",
                "mid",
                "book_state",
                "feature_age_seconds",
                "causal_spread_widening_60s",
                "causal_adverse_imbalance_change_60s",
            ],
            "l2_unavailable_rule": (
                "book_state other than VALID, schema mismatch, or excessive age means feature "
                "UNAVAILABLE; never repair with a later snapshot"
            ),
            "deferred_not_in_v1": [
                "multi_level_depth_notional",
                "standalone_microprice_directional_alpha",
                "cross_venue_premium_without_PIT_FX_and_quote_normalization",
            ],
        },
        "feature_age_limits": {
            "flow_60s_maximum_age_seconds": 5,
            "flow_300s_maximum_age_seconds": 15,
            "kraken_confirmation_maximum_age_seconds": 5,
            "l2_profile_specific_seconds": [1, 2, 5],
            "htf_closed_candle_only": True,
        },
        "allowed_parameter_grids": {
            "flow_profiles": list(_named_flow_profiles()),
            "kraken_lags_seconds": list(ALLOWED_LAGS_SECONDS),
            "l2_profiles": list(_named_l2_profiles()),
            "evaluation_horizons_hours": list(EVALUATION_HORIZONS_HOURS),
            "context_modifier_regions": {
                "cmc_breadth_floor": [0.40, 0.60],
                "derivatives_context_window_hours": [24, 72],
            },
            "grid_expansion_after_failure": False,
        },
        "forbidden_degrees_of_freedom": {
            "target_fields_during_preregistration": list(TARGET_FIELD_NAMES),
            "operations": [
                "future-return correlation scan",
                "best lag search outside frozen lag set",
                "best threshold search outside named profiles",
                "horizon mining",
                "stop or target optimization",
                "baseline replacement",
                "fold-boundary optimization by PnL",
                "holdout-driven tuning",
                "random-seed retries",
                "grid expansion after loss",
                "future snapshot repair",
            ],
            "timeframes_not_in_generation_v1": ["15m refinement", "1h alternate baseline"],
        },
        "cost_model": _cost_scenarios(bindings.shared_cost_model),
        "metrics": {
            "primary": "NET_EXPECTANCY_R_AFTER_REALISTIC_COSTS",
            "secondary": [
                "profit_factor",
                "Sharpe_or_equivalent",
                "maximum_drawdown",
                "median_R",
                "trade_count",
                "turnover",
                "cost_per_trade",
                "MFE",
                "MAE",
                "MFE_capture",
                "false_entry_rate",
                "holding_period",
                "exposure",
                "parameter_stability",
                "asset_stability",
                "regime_stability",
            ],
            "incremental": [
                "delta_net_expectancy_R",
                "delta_profit_factor",
                "delta_maximum_drawdown",
                "delta_cost_and_turnover",
                "delta_false_entry_rate",
                "delta_MFE_capture",
            ],
            "signal_retention": [
                "baseline_signals",
                "signals_retained",
                "signals_rejected",
                "retention_percentage",
                "retained_signal_economics",
                "rejected_signal_economics",
            ],
            "false_entry_definition": (
                "after executable entry, canonical -1R invalidation occurs before +1R favorable "
                "excursion within the frozen strategy holding path"
            ),
            "mfe_capture_definition": (
                "realized executable R divided by positive available MFE_R, reported with zero and "
                "negative-MFE cases separately"
            ),
            "entry_delay": [
                "signal_time",
                "confirmation_time",
                "execution_time",
                "signal_to_confirmation_seconds",
                "confirmation_to_execution_seconds",
                "price_drift_bps",
            ],
            "missed_trade_cost": ["avoided_losers", "missed_winners", "net_filter_value"],
            "complexity": "incremental realistic-cost benefit per added ladder layer",
        },
        "sample_requirements": {
            "minimum_total_closed_trades": 60,
            "minimum_closed_trades_per_asset": 15,
            "minimum_independent_calendar_weeks": 8,
            "minimum_walk_forward_folds": 4,
            "minimum_total_frozen_history_days_for_exact_gate": 90,
            "readiness_policy_remains_sole_data_quality_trigger": True,
            "small_sample_promotion_permitted": False,
        },
        "walk_forward": {
            "method": "EXPANDING_CHRONOLOGICAL_FOUR_FOLD",
            "development_partition": "manifest development_start through development_end only",
            "initial_training_fraction_of_development": 0.50,
            "validation_fraction_per_fold": 0.125,
            "fold_count": 4,
            "purge_bars_4h": 84,
            "embargo_bars_4h": 2,
            "parameter_selection": (
                "development train/validation only; stable named region; no final-holdout retuning"
            ),
            "boundaries_optimized_by_pnl": False,
            "exact_engine": "research.backtest.BacktestEngine",
            "stage0_authority": "REJECT_ONLY_APPROXIMATE",
        },
        "holdout": {
            "partition": "newest manifest-protected 20 percent, minimum 7 days",
            "required_status": "RESERVED_UNTOUCHED",
            "development_access": False,
            "opened_only_after_candidate_hash_frozen": True,
            "one_shot_per_candidate_generation": True,
            "failure_action": "FAIL_GENERATION_NO_RETUNE_ON_SAME_HOLDOUT",
            "post_freeze_data": "FORWARD_EVIDENCE",
        },
        "multiple_testing": {
            "version": MULTIPLE_TESTING_VERSION,
            "hypothesis_ids": list(hypothesis_ids),
            "HYPOTHESIS_COUNT": len(hypothesis_ids),
            "VARIANT_COUNT": len(variants),
            "ladder_variant_counts": ladder_counts,
            "variant_catalog_hash": stable_hash(variants),
            "variant_catalog": list(variants),
            "required_runtime_counters": [
                "HYPOTHESIS_COUNT",
                "VARIANT_COUNT",
                "FAILED_VARIANTS",
                "SURVIVING_VARIANTS",
            ],
            "false_discovery": "Benjamini-Hochberg q=0.05 within each declared campaign family",
            "deflated_performance": "report deflated Sharpe where mathematically applicable",
            "sole_promotion_gate": False,
            "unavailable_variant_rule": "count as NOT_EVALUABLE; never silently substitute",
        },
        "success_and_promotion": {
            "realistic_net_expectancy_R_strictly_greater_than": 0.0,
            "minimum_delta_net_expectancy_R_vs_parent": 0.05,
            "minimum_realistic_profit_factor": 1.15,
            "minimum_delta_profit_factor_vs_parent": 0.10,
            "maximum_drawdown_fraction": 0.15,
            "maximum_relative_drawdown_worsening": 0.10,
            "minimum_positive_validation_folds": 3,
            "holdout_realistic_net_expectancy_R_strictly_greater_than": 0.0,
            "holdout_delta_net_expectancy_R_vs_parent_at_least": 0.0,
            "stress_net_expectancy_R_at_least": 0.0,
            "signal_retention_fraction_range": [0.25, 0.80],
            "maximum_positive_pnl_asset_concentration": 0.70,
            "minimum_nonnegative_assets": 2,
            "maximum_positive_pnl_single_14d_window_concentration": 0.50,
            "maximum_positive_pnl_single_regime_concentration": 0.80,
            "parameter_neighbor_minimum_delta_retention": 0.70,
            "complexity_preference": (
                "prefer simpler parent unless added layer improves expectancy by at least 0.025R, "
                "reduces executable cost by 10 bps/trade, or reduces false entries by 10 percent"
            ),
            "final_state_after_pass": "FORWARD_PAPER_CANARY_CANDIDATE_ONLY",
            "automatic_live_authority": False,
        },
        "failure_and_stop": {
            "candidate_failure": [
                "non-positive realistic net expectancy",
                "insufficient incremental economic value",
                "sample requirement failure",
                "walk-forward gate failure",
                "parameter spike without acceptable neighbor",
                "asset, time, or regime concentration breach",
                "pathological turnover or cost fragility",
                "one-shot holdout failure",
                "causal data or exact-engine integrity failure",
            ],
            "family_abandonment": [
                "no exact survivor in declared catalog",
                "all exact survivors fail walk-forward",
                "all frozen walk-forward survivors fail one-shot holdout",
            ],
            "after_family_failure": "ABANDON_GENERATION_OR_CREATE_NEW_PREREGISTRATION",
            "mutate_failed_grid": False,
        },
        "reproducibility": {
            "seeds": list(ALLOWED_SEEDS),
            "retry_until_better": False,
            "result_identity_inputs": [
                "dataset_freeze_id",
                "preregistration_id",
                "git_commit",
                "backtester_version",
                "cost_model_version",
                "feature_schema_versions",
                "seed",
            ],
        },
        "readiness_and_freeze_gate": {
            "policy_version": bindings.readiness_policy_version,
            "accepted_readiness_states": sorted(RESEARCH_USABLE_STATES),
            "dataset_source": "EXISTING_FAMILY_FREEZE_MANAGER_MANIFEST_BY_EXACT_ID",
            "mutable_latest_or_status_dataset_accepted": False,
            "decision_tree": [
                "FLOW_CONFIRMED_SWING -> P1.3_FLOW_CONFIRMED_SWING_RESEARCH",
                "CROSS_VENUE_LEAD_LAG -> P1.3_CROSS_VENUE_MARKET_STRUCTURE_RESEARCH",
                "CMC_BREADTH -> P1.3_BREADTH_CONDITIONED_RESEARCH",
            ],
            "current_action_until_gate": "CONTINUE_PROSPECTIVE_COLLECTION",
        },
        "authority_and_policy": {
            "spot_long_only": True,
            "shariah_prohibited": [
                "leverage",
                "shorting",
                "futures execution",
                "perpetual execution",
                "options execution",
                "lending",
                "interest",
                "staking execution",
            ],
            "bitvavo_execution_venue": True,
            "kraken_execution": False,
            "mexc_execution": False,
            "ml_authority": "SHADOW_ONLY_NO_TRAINING_IN_PREREGISTRATION",
            "rl_in_initial_ladder": False,
            "events_in_initial_ladder": False,
            "live_promotion_in_scope": False,
            "risk_caps_mutable": False,
            "orders_generated": 0,
        },
        "reference_concepts": {
            "NautilusTrader": "state and order lifecycle validation concepts",
            "LEAN": "portfolio and target semantics concepts",
            "vectorbt": "cheap reject-only screening concepts",
            "Freqtrade": "crypto lifecycle and bias-check concepts",
            "PyBroker": "walk-forward consistency concepts",
            "Qlib": "dataset and experiment recording concepts",
            "license_policy": "concept use only; do not copy restricted code",
        },
        "reporting_template": build_empty_results_template(),
        "notification": {
            "message": "P1.3 EXPERIMENT PLAN FROZEN",
            "buy_signal": False,
            "exchange_action": False,
        },
    }


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _latest_alpha_evidence(workspace: Path) -> Path:
    candidates = tuple(
        (workspace / "output" / "alpha_discovery" / "runs").glob(
            "*/alpha_discovery_evidence.json"
        )
    )
    if not candidates:
        raise FileNotFoundError("prior P1.1 alpha evidence is missing")
    dated = []
    for path in candidates:
        payload = read_json(path)
        dated.append((parse_utc(payload["created_at"]), path, payload))
    _, selected, payload = max(dated, key=lambda row: row[0])
    medium = (payload.get("exact_results") or {}).get("MEDIUM_TERM_TREND_PULLBACK") or {}
    expected = {
        "continuation_days": 3,
        "exit_days": 5,
        "pullback_atr": 0.5,
        "trend_days": 20,
    }
    if medium.get("frozen_family_parameters") != expected:
        raise ImmutableArtifactError("prior P1.1 frozen baseline parameters changed")
    return selected


def _p1_2_3_artifact(workspace: Path) -> Path:
    pointer = workspace / "output" / "multi_source" / "p1_2_3" / "P1_2_3_FINAL_LATEST.json"
    payload = read_json(pointer)
    run_id = str(payload.get("run_id") or "")
    selected = pointer.parent / f"{run_id}.json"
    if not selected.is_file():
        raise FileNotFoundError(f"immutable P1.2.3 artifact is missing: {selected}")
    immutable = read_json(selected)
    if immutable.get("artifact_hash") != payload.get("artifact_hash"):
        raise ImmutableArtifactError("P1.2.3 latest pointer does not match immutable artifact")
    return selected


def bindings_from_workspace(workspace: Path | str) -> PreregistrationBindings:
    """Read version metadata only; no prospective rows or outcome labels."""

    root = Path(workspace).resolve()
    p123_path = _p1_2_3_artifact(root)
    p123 = read_json(p123_path)
    p11_path = _latest_alpha_evidence(root)
    p11 = read_json(p11_path)
    settings = get_settings()
    shared_cost = SharedCostModel.from_settings(settings)
    source_paths = {
        "native_backtester": root / "research" / "backtest.py",
        "p1_1_baseline_source": root / "research" / "alpha_discovery.py",
        "readiness_policy_source": root / "data" / "multi_source_maturation.py",
        "freeze_contract_source": root / "data" / "multi_source_maturation.py",
        "bitvavo_l2_v2_source": root / "data" / "bitvavo_l2_reconstruction_v2.py",
    }
    dirty = bool(_git(root, "status", "--porcelain=v1"))
    return PreregistrationBindings(
        git_commit=_git(root, "rev-parse", "HEAD"),
        git_worktree_state="DIRTY_SOURCE_HASHES_BOUND" if dirty else "CLEAN",
        p1_2_3_artifact_hash=str(p123["artifact_hash"]),
        p1_2_3_artifact_file_sha256=sha256_file(p123_path),
        p1_2_3_artifact_path=str(p123_path.resolve()),
        p1_1_evidence_hash=str(p11["artifact_hash"]),
        p1_1_evidence_file_sha256=sha256_file(p11_path),
        p1_1_evidence_path=str(p11_path.resolve()),
        readiness_policy_version="research_readiness_policy_v1",
        shared_cost_model=asdict(shared_cost),
        feature_schema_versions={
            "source_observation": "source_neutral_observation_v1",
            "point_in_time_feature_store": "multi_source_pit_feature_store_v1",
            "bitvavo_l2": "bitvavo_l2_features_v2",
            "bitvavo_l2_reconstruction": "L2_RECONSTRUCTION_V2",
            "family_freeze": "family_dataset_freeze_v1",
        },
        source_code_hashes={name: sha256_file(path) for name, path in source_paths.items()},
    )


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = f"{stable_json(payload, indent=2)}\n".encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


class PreregistrationStore:
    """Content-addressed store that never overwrites a frozen generation."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, preregistration_id: str) -> Path:
        if not preregistration_id.startswith(f"{PREREGISTRATION_VERSION}_"):
            raise ImmutableArtifactError("invalid preregistration ID")
        if any(token in preregistration_id.casefold() for token in ("latest", "current", "/", "\\")):
            raise ImmutableArtifactError("mutable or path-like preregistration ID rejected")
        return self.root / "preregistrations" / preregistration_id / f"{PREREGISTRATION_VERSION}.json"

    def create(self, plan: Mapping[str, Any], *, created_at: str | None = None) -> dict[str, Any]:
        selected_plan = dict(plan)
        content_hash = stable_hash(selected_plan)
        preregistration_id = f"{PREREGISTRATION_VERSION}_{content_hash[:16]}"
        path = self.path_for(preregistration_id)
        if path.is_file():
            existing = self.verify(preregistration_id)
            if existing["content_hash"] != content_hash:
                raise ImmutableArtifactError("immutable preregistration collision")
            return existing
        envelope = {
            "schema_version": PREREGISTRATION_SCHEMA,
            "preregistration_id": preregistration_id,
            "creation_timestamp": created_at or utc_iso(),
            "content_hash": content_hash,
            "hash": content_hash,
            "plan": selected_plan,
            "immutable": True,
            "research_started": False,
            "performance_calculated": False,
            "orders_generated": 0,
            "live_authority_changed": False,
        }
        envelope["artifact_hash"] = stable_hash(envelope)
        try:
            _exclusive_write_json(path, envelope)
        except FileExistsError:
            existing = self.verify(preregistration_id)
            if existing["content_hash"] != content_hash:
                raise ImmutableArtifactError("immutable preregistration collision")
            return existing
        template_path = self.root / "templates" / f"{RESULT_TEMPLATE_VERSION}.json"
        template = build_empty_results_template()
        try:
            _exclusive_write_json(template_path, template)
        except FileExistsError:
            if stable_hash(read_json(template_path)) != stable_hash(template):
                raise ImmutableArtifactError("immutable P1.3 result template changed")
        notice_path = self.root / "notifications" / f"{preregistration_id}.json"
        notice = {
            "schema_version": "p1_3_plan_frozen_notification_v1",
            "preregistration_id": preregistration_id,
            "recorded_at": envelope["creation_timestamp"],
            "message": "P1.3 EXPERIMENT PLAN FROZEN",
            "buy_signal": False,
            "sent_to_exchange": False,
        }
        try:
            _exclusive_write_json(notice_path, notice)
        except FileExistsError:
            if stable_hash(read_json(notice_path)) != stable_hash(notice):
                raise ImmutableArtifactError("immutable notification record changed")
        return envelope

    def verify(self, preregistration_id: str) -> dict[str, Any]:
        path = self.path_for(preregistration_id)
        if not path.is_file():
            raise ImmutableArtifactError("preregistration ID does not exist")
        payload = dict(read_json(path))
        if payload.get("preregistration_id") != preregistration_id:
            raise ImmutableArtifactError("preregistration identity mismatch")
        if payload.get("immutable") is not True:
            raise ImmutableArtifactError("preregistration is not immutable")
        expected_content = stable_hash(payload.get("plan"))
        if payload.get("content_hash") != expected_content or payload.get("hash") != expected_content:
            raise ImmutableArtifactError("preregistration content hash mismatch")
        artifact_body = {key: value for key, value in payload.items() if key != "artifact_hash"}
        if payload.get("artifact_hash") != stable_hash(artifact_body):
            raise ImmutableArtifactError("preregistration artifact hash mismatch")
        return payload

    def reject_modification(self, preregistration_id: str, replacement: Mapping[str, Any]) -> None:
        current = self.verify(preregistration_id)
        if stable_hash(replacement) != current["content_hash"]:
            raise ImmutableArtifactError(
                "frozen preregistration cannot be modified; create a new generation"
            )


class AppendOnlyResearchLedger:
    """Durable JSONL hash chain retaining successes and failures."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        previous = "GENESIS"
        for index, row in enumerate(rows, start=1):
            if row.get("sequence") != index or row.get("previous_hash") != previous:
                raise ImmutableArtifactError("P1.3 research ledger sequence or chain mismatch")
            body = {key: value for key, value in row.items() if key != "record_hash"}
            if row.get("record_hash") != stable_hash(body):
                raise ImmutableArtifactError("P1.3 research ledger record hash mismatch")
            previous = str(row["record_hash"])
        return rows

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        rows = self.records()
        body = {
            "schema_version": LEDGER_VERSION,
            "sequence": len(rows) + 1,
            "previous_hash": rows[-1]["record_hash"] if rows else "GENESIS",
            "payload": dict(payload),
        }
        record = {**body, "record_hash": stable_hash(body)}
        append_jsonl(self.path, record)
        return record

    def summary(self) -> dict[str, Any]:
        rows = self.records()
        statuses = [str((row.get("payload") or {}).get("status")) for row in rows]
        return {
            "schema_version": LEDGER_VERSION,
            "record_count": len(rows),
            "failed_runs": statuses.count("FAILED"),
            "authorized_runs": statuses.count("AUTHORIZED_NOT_EXECUTED"),
            "root_hash": rows[-1]["record_hash"] if rows else "GENESIS",
        }


def _freeze_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema = manifest.get("schema_version")
    if schema == "family_dataset_freeze_v1":
        keys = (
            "schema_version",
            "family",
            "policy_version",
            "source_manifests",
            "assets",
            "features",
            "development_start",
            "development_end",
            "holdout_start",
            "holdout_end",
            "data_end",
            "coverage",
            "quality",
            "clock_metrics",
            "gap_metrics",
            "build_commit",
            "readiness_transition_id",
        )
    elif schema == "multi_source_dataset_freeze_v1":
        keys = (
            "schema_version",
            "family",
            "collection_epoch",
            "trainable_cutoff",
            "holdout_start",
            "data_end",
            "source_manifests",
            "readiness",
        )
    else:
        raise ResearchGateError("unsupported dataset freeze schema")
    return {key: manifest.get(key) for key in keys}


def resolve_dataset_freeze(freeze_root: Path | str, dataset_freeze_id: str) -> dict[str, Any]:
    """Resolve and cryptographically validate one exact immutable freeze ID."""

    if not dataset_freeze_id or not IMMUTABLE_ID.fullmatch(dataset_freeze_id):
        raise ResearchGateError("exact 64-hex DATASET_FREEZE_ID is required")
    if any(token in dataset_freeze_id.casefold() for token in ("latest", "current", "status")):
        raise ResearchGateError("mutable dataset aliases are forbidden")
    root = Path(freeze_root).resolve()
    matches = [path for path in root.glob(f"*/{dataset_freeze_id}/manifest.json") if path.is_file()]
    if len(matches) != 1:
        raise ResearchGateError("dataset freeze ID is missing or ambiguous")
    path = matches[0].resolve()
    if any(part.casefold() in {"latest", "current", "status"} for part in path.parts):
        raise ResearchGateError("mutable dataset path rejected")
    manifest = dict(read_json(path))
    if manifest.get("dataset_id") != dataset_freeze_id:
        raise ResearchGateError("dataset freeze identity mismatch")
    if manifest.get("immutable") is not True:
        raise ResearchGateError("mutable dataset rejected")
    if manifest.get("holdout_status") != "RESERVED_UNTOUCHED":
        raise ResearchGateError("dataset does not have an untouched holdout")
    if stable_hash(_freeze_identity(manifest)) != dataset_freeze_id:
        raise ResearchGateError("dataset freeze content hash mismatch")
    readiness = manifest.get("quality") or manifest.get("readiness") or {}
    if readiness.get("state") not in RESEARCH_USABLE_STATES:
        raise ResearchGateError("dataset family is not RESEARCH_USABLE")
    return {**manifest, "manifest_path": str(path), "manifest_sha256": sha256_file(path)}


class PartitionAccessGuard:
    """Authorize timestamp ranges without exposing protected rows."""

    @staticmethod
    def authorize(
        manifest: Mapping[str, Any],
        *,
        start: str,
        end: str,
        phase: str,
        candidate_hash: str | None = None,
    ) -> dict[str, Any]:
        selected_start = parse_utc(start)
        selected_end = parse_utc(end)
        if selected_end < selected_start:
            raise HoldoutAccessError("requested range end precedes start")
        development_start = parse_utc(
            str(manifest.get("development_start") or manifest.get("collection_epoch"))
        )
        development_end = parse_utc(
            str(manifest.get("development_end") or manifest.get("trainable_cutoff"))
        )
        holdout_start = parse_utc(str(manifest["holdout_start"]))
        holdout_end = parse_utc(str(manifest["data_end"]))
        if phase == "DEVELOPMENT":
            if selected_start < development_start or selected_end > development_end:
                raise HoldoutAccessError("development code cannot request final holdout rows")
            partition = "DEVELOPMENT_DATA"
        elif phase == "FINAL_HOLDOUT":
            if not candidate_hash or len(candidate_hash) < 16:
                raise HoldoutAccessError("frozen candidate hash is required before holdout access")
            if selected_start < holdout_start or selected_end > holdout_end:
                raise HoldoutAccessError("final holdout request is outside protected boundaries")
            partition = "RESERVED_UNTOUCHED_HOLDOUT"
        else:
            raise HoldoutAccessError("unknown research phase")
        return {
            "phase": phase,
            "partition": partition,
            "start": utc_iso(selected_start),
            "end": utc_iso(selected_end),
            "rows_loaded": 0,
        }


class P13ResearchRunner:
    """Future run preflight; authorization is intentionally not execution."""

    def __init__(self, workspace: Path | str, governance_root: Path | str | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.governance_root = Path(governance_root or self.workspace / "output" / "research_governance" / "p1_3")
        self.store = PreregistrationStore(self.governance_root)
        self.ledger = AppendOnlyResearchLedger(self.governance_root / "ledger" / "research_runs.jsonl")

    def _fail(self, reason: str, **identity: Any) -> None:
        self.ledger.append(
            {
                "status": "FAILED",
                "recorded_at": utc_iso(),
                "reason": reason,
                **identity,
                "research_executed": False,
                "performance_calculated": False,
                "orders_generated": 0,
                "live_authority_changed": False,
            }
        )

    def authorize(
        self,
        *,
        preregistration_id: str | None,
        dataset_freeze_id: str | None,
        seed: int = ALLOWED_SEEDS[0],
        phase: str = "DEVELOPMENT",
        candidate_hash: str | None = None,
    ) -> dict[str, Any]:
        identity = {
            "preregistration_id": preregistration_id,
            "dataset_freeze_id": dataset_freeze_id,
            "seed": seed,
            "phase": phase,
        }
        try:
            if not preregistration_id:
                raise ResearchGateError("PREREGISTRATION_ID is required")
            if not dataset_freeze_id:
                raise ResearchGateError("DATASET_FREEZE_ID is required")
            preregistration = self.store.verify(preregistration_id)
            plan = preregistration["plan"]
            if seed not in plan["reproducibility"]["seeds"]:
                raise ResearchGateError("seed is outside the frozen seed set")
            if plan["bindings"]["git_commit"] != _git(self.workspace, "rev-parse", "HEAD"):
                raise ResearchGateError("git commit differs from preregistration binding")
            freeze = resolve_dataset_freeze(
                self.workspace / "output" / "multi_source" / "freezes",
                dataset_freeze_id,
            )
            if phase == "FINAL_HOLDOUT":
                prior_attempt = any(
                    (row.get("payload") or {}).get("phase") == "FINAL_HOLDOUT"
                    and (row.get("payload") or {}).get("candidate_hash") == candidate_hash
                    and (row.get("payload") or {}).get("dataset_freeze_id") == dataset_freeze_id
                    for row in self.ledger.records()
                )
                if prior_attempt:
                    raise ResearchGateError("one-shot holdout already opened for candidate")
            family = str(freeze.get("family") or "")
            run_identity = {
                "runner_version": RUNNER_VERSION,
                "preregistration_id": preregistration_id,
                "preregistration_hash": preregistration["content_hash"],
                "dataset_freeze_id": dataset_freeze_id,
                "dataset_manifest_sha256": freeze["manifest_sha256"],
                "family": family,
                "phase": phase,
                "candidate_hash": candidate_hash,
                "seed": seed,
                "git_commit": plan["bindings"]["git_commit"],
                "backtester_version": plan["bindings"]["backtester_version"],
                "cost_model_version": plan["cost_model"]["version"],
                "feature_schema_versions": plan["bindings"]["feature_schema_versions"],
            }
            authorization = {
                **run_identity,
                "run_identity_hash": stable_hash(run_identity),
                "status": "AUTHORIZED_NOT_EXECUTED",
                "research_executed": False,
                "performance_calculated": False,
                "orders_generated": 0,
                "live_authority_changed": False,
                "kraken_execution": False,
                "mexc_execution": False,
            }
            authorization["result_hash"] = stable_hash(authorization)
            self.ledger.append({**authorization, "recorded_at": utc_iso()})
            return authorization
        except (ImmutableArtifactError, ResearchGateError, subprocess.SubprocessError) as exc:
            self._fail(str(exc), **identity, candidate_hash=candidate_hash)
            if isinstance(exc, ResearchGateError):
                raise
            raise ResearchGateError(str(exc)) from exc


__all__ = [
    "ALLOWED_SEEDS",
    "AppendOnlyResearchLedger",
    "HoldoutAccessError",
    "ImmutableArtifactError",
    "P13ResearchRunner",
    "PREREGISTRATION_VERSION",
    "PartitionAccessGuard",
    "PreregistrationBindings",
    "PreregistrationStore",
    "ResearchGateError",
    "RESULT_TEMPLATE_VERSION",
    "TARGET_FIELD_NAMES",
    "bindings_from_workspace",
    "build_empty_results_template",
    "build_preregistration_plan",
    "build_variant_catalog",
    "resolve_dataset_freeze",
]
