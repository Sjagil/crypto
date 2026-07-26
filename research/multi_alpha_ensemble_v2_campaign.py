"""Campaign service for the frozen classical multi-alpha v2 ensemble."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config.settings import Settings
from research.absolute_momentum import (
    AbsoluteMomentumParameters,
    backtest_absolute_momentum,
)
from research.forward_observer import (
    ForwardPerformanceGatePolicy,
    build_rotation_forward_evidence,
    merge_portfolio_forward_manifest,
    validate_forward_manifest_identity,
)
from research.liquidity_sweep_campaign import (
    _json_ready,
    _stochastic_validation,
)
from research.multi_alpha_ensemble import (
    backtest_multi_alpha_ensemble,
)
from research.multi_alpha_ensemble_v2 import (
    FROZEN_COMPONENT_DNA_V2,
    MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION,
    MULTI_ALPHA_ENSEMBLE_V2_FAMILY,
    MultiAlphaEnsembleV2Parameters,
)
from research.optimization import multiple_testing_bootstrap
from research.portfolio_selection import RotationPortfolioPolicy, rotation_period_metrics
from research.residual_reversal import (
    ResidualReversalParameters,
    backtest_residual_reversal,
)
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
)

MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN = "MULTI_ALPHA_ENSEMBLE_V2"
MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS = 16_910
MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START = pd.Timestamp(
    "2026-07-26T00:00:00+00:00"
)
MULTI_ALPHA_ENSEMBLE_V2_PERIODS = {
    "development": ("2019-12-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "confirmation": ("2025-07-01", "2026-07-24"),
}


def multi_alpha_ensemble_v2_campaign_path(
    settings: Settings,
) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "multi_alpha_ensemble_campaign_v2.json"
    )


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=(
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ),
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=200,
    )


def _expected_plan() -> dict[str, Any]:
    parameters = MultiAlphaEnsembleV2Parameters()
    return {
        "schema_version": "multi_alpha_ensemble_plan_v2",
        "status": "CAMPAIGN_PLAN",
        "campaign": MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN,
        "strategy_family": MULTI_ALPHA_ENSEMBLE_V2_FAMILY,
        "engine_version": MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION,
        "timeframe": "1d",
        "economic_hypothesis": (
            "A fixed causal combination of a long-horizon absolute-"
            "momentum sleeve and a short-horizon BTC-beta-residual "
            "reversal sleeve can diversify trend and pullback regimes "
            "without relaxing the shared portfolio risk limits."
        ),
        "trial_count": 1,
        "strategy_dna_hash": parameters.dna_hash,
        "strategy_dna": asdict(parameters),
        "search_space_hash": stable_hash(
            [parameters.dna_hash],
            length=64,
        ),
        "component_dna": dict(FROZEN_COMPONENT_DNA_V2),
        "selection_basis": "NONE_SINGLE_FIXED_DNA",
        "component_allocation": "EQUAL_FIXED_SLEEVES",
        "meta_execution": (
            "COMPONENT_EXECUTED_WEIGHTS_KNOWN_THEN_META_DECISION_"
            "EXECUTED_NEXT_DAILY_OPEN"
        ),
        "periods": MULTI_ALPHA_ENSEMBLE_V2_PERIODS,
        "portfolio_policy": asdict(_policy()),
        "base_known_trials": (
            MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS
        ),
        "projected_total_known_trials": (
            MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS + 1
        ),
        "forward_requirement": {
            "minimum_closed_daily_observations": 365,
            "minimum_rebalances": 30,
            "minimum_regime_coverage_per_state": 5,
        },
        "known_limitations": [
            "COMPONENTS_WERE_SELECTED_USING_HISTORICAL_RESEARCH",
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
            "HISTORICAL_RESULTS_CANNOT_AUTHORIZE_PROMOTION",
            "FORWARD_EVIDENCE_REQUIRED",
            "NO_AUTOMATIC_PROMOTION",
        ],
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def plan_multi_alpha_ensemble_v2_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Persist or verify the immutable single-DNA campaign plan."""

    expected = _expected_plan()
    plan_path = (
        settings.paths.lab_dir
        / "reports"
        / "multi_alpha_ensemble_plan_v2.json"
    )
    immutable_fields = (
        "campaign",
        "strategy_family",
        "engine_version",
        "trial_count",
        "strategy_dna_hash",
        "strategy_dna",
        "search_space_hash",
        "component_dna",
        "selection_basis",
        "component_allocation",
        "meta_execution",
        "periods",
        "portfolio_policy",
        "base_known_trials",
        "projected_total_known_trials",
        "forward_requirement",
    )
    if plan_path.is_file():
        stored = read_json(plan_path)
        for field in immutable_fields:
            if _json_ready(stored.get(field)) != _json_ready(
                expected.get(field)
            ):
                raise RuntimeError(
                    f"MULTI_ALPHA_ENSEMBLE_V2_PLAN_DRIFT:{field}"
                )
    else:
        atomic_write_json(plan_path, expected)
    return {
        **expected,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
    }


def _preserved_forward_fields(
    existing: Mapping[str, Any],
    *,
    source_candidate_identity: str,
    strategy_dna_hash: str,
    execution_identity: str,
) -> dict[str, Any]:
    validate_forward_manifest_identity(
        existing,
        source_candidate_identity=source_candidate_identity,
        strategy_dna_hash=strategy_dna_hash,
        execution_identity=execution_identity,
        forward_start=MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START,
    )
    return {
        field: existing[field]
        for field in (
            "forward_observer_schema_version",
            "forward_observations",
            "forward_hash_chain",
            "forward_decisions",
            "forward_summary",
            "degradation_observation",
        )
        if field in existing
    }


def _forward_policy(settings: Settings) -> ForwardPerformanceGatePolicy:
    return ForwardPerformanceGatePolicy(
        minimum_profit_factor=settings.research.minimum_profit_factor,
        minimum_stressed_profit_factor=(
            settings.research.minimum_stressed_profit_factor
        ),
        maximum_drawdown=settings.research.maximum_drawdown,
        minimum_effective_sample_size=(
            settings.research.minimum_effective_sample_size
        ),
        stressed_cost_multiplier=(
            settings.costs.stressed_cost_multiplier
        ),
        bootstrap_samples=(
            settings.research.multiple_testing_bootstrap_samples
        ),
        bootstrap_block_size=max(
            10,
            settings.research.multiple_testing_block_size,
        ),
        bootstrap_seed=settings.app.random_seed,
    )


def run_multi_alpha_ensemble_v2_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Execute the frozen two-sleeve ensemble without promotion authority."""

    plan = plan_multi_alpha_ensemble_v2_campaign(settings)
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    paths = {
        market: (
            settings.paths.processed_data_dir
            / f"{market}_1d.parquet"
        )
        for market in markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing multi-alpha-v2 datasets: {missing}"
        )
    frames = {
        market: pd.read_parquet(path)
        for market, path in paths.items()
    }
    data_hashes = {
        market: sha256_file(path) for market, path in paths.items()
    }
    data_fingerprint = stable_hash(data_hashes, length=64)
    ensemble_policy = _policy()
    absolute_policy = RotationPortfolioPolicy(
        allowed_markets=markets,
        maximum_total_exposure=0.20,
        maximum_position_exposure=0.20,
        minimum_cash=0.80,
        minimum_history_observations=90,
    )
    parameters = MultiAlphaEnsembleV2Parameters()
    absolute_parameters = AbsoluteMomentumParameters(
        target_annualized_volatility=0.05
    )
    residual_parameters = ResidualReversalParameters(
        beta_lookback=60,
        residual_horizon=5,
        entry_zscore=-2.0,
    )
    actual_component_dna = {
        "ABSOLUTE_MOMENTUM_VOL_05": (
            absolute_parameters.dna_hash
        ),
        "RESIDUAL_REVERSAL_B60_H5_Z20": (
            residual_parameters.dna_hash
        ),
    }
    if actual_component_dna != dict(FROZEN_COMPONENT_DNA_V2):
        raise RuntimeError(
            "MULTI_ALPHA_ENSEMBLE_V2_COMPONENT_DNA_DRIFT"
        )
    absolute = backtest_absolute_momentum(
        frames,
        absolute_parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=absolute_policy,
    )
    residual = backtest_residual_reversal(
        frames,
        residual_parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=ensemble_policy,
    )
    component_weights = {
        "ABSOLUTE_MOMENTUM_VOL_05": absolute.executed_weights,
        "RESIDUAL_REVERSAL_B60_H5_Z20": (
            residual.executed_weights
        ),
    }
    normal = backtest_multi_alpha_ensemble(
        frames,
        component_weights,
        parameters,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=ensemble_policy,
    )
    stressed = backtest_multi_alpha_ensemble(
        frames,
        component_weights,
        parameters,
        fee_rate=(
            settings.costs.default_fee
            * settings.costs.stressed_cost_multiplier
        ),
        slippage_bps=(
            settings.costs.slippage_bps
            * settings.costs.stressed_cost_multiplier
        ),
        spread_bps=(
            settings.costs.spread_bps
            * settings.costs.stressed_cost_multiplier
        ),
        portfolio_policy=ensemble_policy,
    )
    periods: dict[str, Any] = {}
    stressed_periods: dict[str, Any] = {}
    development_returns: pd.Series | None = None
    for period, bounds in MULTI_ALPHA_ENSEMBLE_V2_PERIODS.items():
        metrics, returns = rotation_period_metrics(
            normal.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )
        periods[period] = metrics
        stressed_periods[period] = rotation_period_metrics(
            stressed.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )[0]
        if period == "development":
            development_returns = returns
    if development_returns is None or development_returns.empty:
        raise RuntimeError(
            "MULTI_ALPHA_ENSEMBLE_V2_DEVELOPMENT_EMPTY"
        )

    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "multi_alpha_ensemble_v2",
        campaign_id=MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN,
    )
    registration = registry.register(
        data_fingerprint=data_fingerprint,
        strategy_family=MULTI_ALPHA_ENSEMBLE_V2_FAMILY,
        strategy_dna_hash=parameters.dna_hash,
        parameters=asdict(parameters),
        metrics_at_birth={
            **periods["development"],
            "full_sample_metrics": normal.metrics,
        },
        return_path_hash=stable_hash(
            [
                round(float(value), 15)
                for value in development_returns.to_numpy(dtype=float)
            ],
            length=64,
        ),
        selection_metadata={
            "selection_basis": "NONE_SINGLE_FIXED_DNA",
            "inherited_component_selection_bias": True,
            "validation_used": False,
            "confirmation_used": False,
        },
    )
    registry_audit = registry.audit()
    total_known_trials = (
        MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS
        + int(registry_audit["unique_strategy_dna_count"])
    )
    matrix = pd.DataFrame(
        {"MULTI_ALPHA_FIXED_V2": development_returns}
    ).dropna()
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=(
            settings.research.multiple_testing_bootstrap_samples
        ),
        block_size=max(
            10,
            settings.research.multiple_testing_block_size,
        ),
        seed=settings.app.random_seed,
        known_trial_count=total_known_trials,
    )
    stochastic = _stochastic_validation(
        settings,
        normal_equity=normal.equity_curve,
        stressed_equity=stressed.equity_curve,
    )
    economic_checks = {
        "all_periods_positive": all(
            float(periods[period]["net_return"]) > 0.0
            for period in MULTI_ALPHA_ENSEMBLE_V2_PERIODS
        ),
        "all_stressed_periods_positive": all(
            float(stressed_periods[period]["net_return"]) > 0.0
            for period in MULTI_ALPHA_ENSEMBLE_V2_PERIODS
        ),
        "minimum_rebalances": (
            int(normal.metrics["rebalance_count"])
            >= settings.research.minimum_trades
        ),
        "minimum_effective_sample": (
            int(
                normal.metrics[
                    "portfolio_period_effective_sample_size"
                ]
            )
            >= settings.research.minimum_effective_sample_size
        ),
        "profit_factor": (
            float(normal.metrics["portfolio_period_profit_factor"])
            >= settings.research.minimum_profit_factor
        ),
        "validation_profit_factor": (
            float(
                periods["validation"][
                    "portfolio_period_profit_factor"
                ]
            )
            >= settings.research.minimum_profit_factor
        ),
        "stressed_validation_profit_factor": (
            float(
                stressed_periods["validation"][
                    "portfolio_period_profit_factor"
                ]
            )
            >= settings.research.minimum_stressed_profit_factor
        ),
        "maximum_drawdown": (
            abs(float(normal.metrics["maximum_drawdown"]))
            <= settings.research.maximum_drawdown
        ),
        "exposure_limits_respected": all(
            bool(normal.integrity[field])
            for field in (
                "maximum_exposure_respected",
                "maximum_position_exposure_respected",
                "minimum_cash_respected",
                "maximum_positions_respected",
            )
        ),
        "component_dna_frozen": bool(
            normal.integrity["component_dna_frozen"]
        ),
    }
    component_reports = {
        "absolute_momentum": read_json(
            settings.paths.lab_dir
            / "reports"
            / "absolute_momentum_campaign_v1.json"
        ),
        "residual_reversal": read_json(
            settings.paths.lab_dir
            / "reports"
            / "residual_reversal_campaign_v1.json"
        ),
    }
    inherited_pbo = {
        "absolute_momentum": component_reports[
            "absolute_momentum"
        ]["multiple_testing"][
            "probability_of_backtest_overfitting"
        ],
        "residual_reversal": component_reports[
            "residual_reversal"
        ]["multiple_testing"][
            "probability_of_backtest_overfitting"
        ],
    }
    inherited_bias_pass = all(
        value is not None
        and float(value)
        <= (
            settings.research
            .maximum_probability_of_backtest_overfitting
        )
        for value in inherited_pbo.values()
    )
    statistical_checks = {
        "deflated_sharpe": (
            float(
                multiple.deflated_sharpe_probabilities.get(
                    "MULTI_ALPHA_FIXED_V2",
                    0.0,
                )
            )
            >= settings.research.minimum_deflated_sharpe_probability
        ),
        "white_reality_check": (
            multiple.white_reality_check_pvalue
            <= settings.research.maximum_white_reality_check_pvalue
        ),
        "hansen_spa": (
            multiple.hansen_spa_pvalue
            <= settings.research.maximum_hansen_spa_pvalue
        ),
        "single_preregistered_dna_no_meta_selection": (
            multiple.probability_of_backtest_overfitting is None
        ),
        "inherited_component_selection_bias": inherited_bias_pass,
        "monte_carlo": bool(
            stochastic["normal"]["monte_carlo"]["passed"]
            and stochastic["stressed"]["monte_carlo"]["passed"]
        ),
        "dirichlet": bool(
            stochastic["normal"]["dirichlet"]["passed"]
            and stochastic["stressed"]["dirichlet"]["passed"]
        ),
        "untouched_holdout": False,
    }
    gates = {
        "economic_checks": economic_checks,
        "statistical_checks": statistical_checks,
        "deflated_sharpe_probability": float(
            multiple.deflated_sharpe_probabilities.get(
                "MULTI_ALPHA_FIXED_V2",
                0.0,
            )
        ),
        "inherited_component_pbo": inherited_pbo,
        "stochastic_validation": stochastic,
        "economic_pass": all(economic_checks.values()),
        "statistical_pass": all(statistical_checks.values()),
        "research_pass": False,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    primary_result = {
        "strategy_id": "MULTI_ALPHA_FIXED_V2",
        "strategy_dna_hash": parameters.dna_hash,
        "parameters": asdict(parameters),
        "registration": registration,
        "normal": normal.summary(),
        "stressed": stressed.summary(),
        "periods": periods,
        "stressed_periods": stressed_periods,
        "gates": gates,
    }

    execution_identity = normal.summary()["execution_identity"]
    source_identity = stable_hash(
        {
            "campaign": MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN,
            "strategy_dna_hash": parameters.dna_hash,
            "portfolio_policy_hash": ensemble_policy.policy_hash,
            "forward_start": (
                MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START.isoformat()
            ),
        },
        length=64,
    )
    observer_path = (
        settings.paths.lab_dir
        / "observers"
        / "multi_alpha_ensemble_v2"
        / "multi_alpha_fixed_v2.json"
    )
    observer = {
        "status": "FROZEN_FORWARD_RESEARCH",
        "family": MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN,
        "policy_name": "MULTI_ALPHA_FIXED_V2",
        "source_candidate_identity": source_identity,
        "strategy_dna_hash": parameters.dna_hash,
        "execution_identity": execution_identity,
        "parameters": asdict(parameters),
        "portfolio_policy": asdict(ensemble_policy),
        "portfolio_policy_hash": ensemble_policy.policy_hash,
        "forward_start": (
            MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START.isoformat()
        ),
        "minimum_forward_closed_daily_observations": 365,
        "minimum_forward_rebalances": 30,
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    if observer_path.is_file():
        observer.update(
            _preserved_forward_fields(
                read_json(observer_path),
                source_candidate_identity=source_identity,
                strategy_dna_hash=parameters.dna_hash,
                execution_identity=execution_identity,
            )
        )
    evidence = build_rotation_forward_evidence(
        normal,
        frames,
        forward_start=MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START,
        minimum_observations=365,
        minimum_rebalances=30,
        performance_policy=_forward_policy(settings),
    )
    observer = merge_portfolio_forward_manifest(
        observer,
        evidence,
        source_candidate_identity=source_identity,
        strategy_dna_hash=parameters.dna_hash,
        execution_identity=execution_identity,
        forward_start=MULTI_ALPHA_ENSEMBLE_V2_FORWARD_START,
    )
    observer["data_hashes"] = data_hashes
    atomic_write_json(observer_path, _json_ready(observer))

    report_path = multi_alpha_ensemble_v2_campaign_path(settings)
    payload = {
        "schema_version": "multi_alpha_ensemble_report_v2",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN,
        "strategy_family": MULTI_ALPHA_ENSEMBLE_V2_FAMILY,
        "engine_version": (
            MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION
        ),
        "timeframe": "1d",
        "plan": plan["plan"],
        "plan_sha256": plan["plan_sha256"],
        "search_space_hash": plan["search_space_hash"],
        "selection_basis": plan["selection_basis"],
        "selection_integrity": {
            "single_fixed_dna": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
            "inherited_component_selection_bias": True,
        },
        "generated_trial_count": 1,
        "registered_unique_trials": int(
            registry_audit["unique_strategy_dna_count"]
        ),
        "registered_epoch_records": int(
            registry_audit["unique_epoch_record_count"]
        ),
        "base_known_trials": (
            MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS
        ),
        "total_known_trials": total_known_trials,
        "primary_strategy_id": "MULTI_ALPHA_FIXED_V2",
        "primary_result": primary_result,
        "multiple_testing": asdict(multiple),
        "trial_registry": registry_audit,
        "data_fingerprint": data_fingerprint,
        "data_hashes": data_hashes,
        "periods": MULTI_ALPHA_ENSEMBLE_V2_PERIODS,
        "portfolio_policy": asdict(ensemble_policy),
        "component_dna": dict(FROZEN_COMPONENT_DNA_V2),
        "holdout_status": (
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS"
        ),
        "forward_requirement": plan["forward_requirement"],
        "observer_manifests": {
            "MULTI_ALPHA_FIXED_V2": str(observer_path)
        },
        "forward_summaries": {
            "MULTI_ALPHA_FIXED_V2": observer["forward_summary"]
        },
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }
    atomic_write_json(report_path, _json_ready(payload))
    csv_path = report_path.with_suffix(".csv")
    pd.DataFrame(
        [
            {
                "strategy_id": "MULTI_ALPHA_FIXED_V2",
                "strategy_dna_hash": parameters.dna_hash,
                "full_net_return": normal.metrics["net_return"],
                "cagr": normal.metrics["annualized_return"],
                "sharpe": normal.metrics["sharpe"],
                "maximum_drawdown": (
                    normal.metrics["maximum_drawdown"]
                ),
                "average_exposure": (
                    normal.metrics["average_exposure"]
                ),
                "profit_factor": normal.metrics[
                    "portfolio_period_profit_factor"
                ],
                "development_net_return": periods[
                    "development"
                ]["net_return"],
                "validation_net_return": periods[
                    "validation"
                ]["net_return"],
                "confirmation_net_return": periods[
                    "confirmation"
                ]["net_return"],
                "economic_pass": gates["economic_pass"],
                "statistical_pass": gates["statistical_pass"],
            }
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "generated_trial_count": 1,
        "registered_unique_trials": payload[
            "registered_unique_trials"
        ],
        "registered_epoch_records": payload[
            "registered_epoch_records"
        ],
        "total_known_trials": total_known_trials,
        "primary_strategy_id": "MULTI_ALPHA_FIXED_V2",
        "economic_pass": gates["economic_pass"],
        "statistical_pass": gates["statistical_pass"],
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": payload["observer_manifests"],
        "forward_summaries": payload["forward_summaries"],
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


__all__ = [
    "MULTI_ALPHA_ENSEMBLE_V2_BASE_KNOWN_TRIALS",
    "MULTI_ALPHA_ENSEMBLE_V2_CAMPAIGN",
    "multi_alpha_ensemble_v2_campaign_path",
    "plan_multi_alpha_ensemble_v2_campaign",
    "run_multi_alpha_ensemble_v2_campaign",
]
