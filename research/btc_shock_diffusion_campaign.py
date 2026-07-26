"""Preregistered campaign for causal BTC shock diffusion research."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config.settings import Settings
from research.btc_shock_diffusion import (
    BTC_SHOCK_DIFFUSION_ENGINE_VERSION,
    BTC_SHOCK_DIFFUSION_FAMILY,
    BTCShockDiffusionParameters,
    backtest_btc_shock_diffusion,
    btc_shock_diffusion_parameter_set,
    residual_reversal_period_metrics,
)
from research.forward_observer import (
    ForwardPerformanceGatePolicy,
    build_rotation_forward_evidence,
    merge_portfolio_forward_manifest,
    validate_forward_manifest_identity,
)
from research.global_trial_accounting import resolve_known_trial_count
from research.liquidity_sweep_campaign import (
    _benchmark_summary,
    _json_ready,
    _stochastic_validation,
)
from research.optimization import multiple_testing_bootstrap
from research.portfolio_selection import RotationPortfolioPolicy
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
)

BTC_SHOCK_DIFFUSION_CAMPAIGN = "BTC_SHOCK_DIFFUSION_V1"
BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS = 16_915
BTC_SHOCK_DIFFUSION_FORWARD_START = pd.Timestamp(
    "2026-07-26T00:00:00+00:00"
)
BTC_SHOCK_DIFFUSION_PERIODS = {
    "development": ("2019-12-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "confirmation": ("2025-07-01", "2026-07-24"),
}


def btc_shock_diffusion_campaign_path(
    settings: Settings,
) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "btc_shock_diffusion_campaign_v1.json"
    )


def _market_paths(settings: Settings) -> dict[str, Path]:
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
            f"missing shock-diffusion datasets: {missing}"
        )
    return paths


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


def _candidate_name(
    candidate: BTCShockDiffusionParameters,
) -> str:
    return (
        f"BSD_S{candidate.shock_lookback}_"
        f"H{candidate.maximum_holding_days}_Z10"
    )


def _expected_plan() -> dict[str, Any]:
    candidates = btc_shock_diffusion_parameter_set()
    return {
        "schema_version": "btc_shock_diffusion_plan_v1",
        "status": "CAMPAIGN_PLAN",
        "campaign": BTC_SHOCK_DIFFUSION_CAMPAIGN,
        "strategy_family": BTC_SHOCK_DIFFUSION_FAMILY,
        "engine_version": BTC_SHOCK_DIFFUSION_ENGINE_VERSION,
        "timeframe": "1d",
        "economic_hypothesis": (
            "Large positive BTC information shocks can diffuse into "
            "liquid altcoins with a short delay. A satellite whose "
            "completed-close move remains below the move implied by "
            "its strictly prior BTC beta may deliver short-horizon "
            "continuation while BTC and the satellite remain in "
            "positive long-term trends."
        ),
        "trial_count": len(candidates),
        "strategy_dna_hashes": [
            candidate.dna_hash for candidate in candidates
        ],
        "strategy_dna": [
            asdict(candidate) for candidate in candidates
        ],
        "search_space_hash": stable_hash(
            [candidate.dna_hash for candidate in candidates],
            length=64,
        ),
        "selection_basis": (
            "DEVELOPMENT_SHARPE_ONLY_WITH_ALL_TRIALS_ACCOUNTED"
        ),
        "periods": BTC_SHOCK_DIFFUSION_PERIODS,
        "portfolio_policy": asdict(_policy()),
        "signal_policy": {
            "information_source": "BTC_COMPLETED_DAILY_CLOSE",
            "shock_definition": (
                "ONE_OR_THREE_DAY_BTC_LOG_RETURN_AT_LEAST_ONE_"
                "STRICTLY_PRIOR_180_DAY_STANDARD_DEVIATION_ABOVE_MEAN"
            ),
            "beta_estimation": (
                "NINETY_DAILY_RETURNS_ENDING_BEFORE_SIGNAL_RETURN"
            ),
            "satellites": ["ETH-EUR", "SOL-EUR", "LINK-EUR"],
            "underreaction_score": (
                "STRICTLY_PRIOR_BETA_TIMES_BTC_SHOCK_MINUS_"
                "SATELLITE_REALIZED_MOVE"
            ),
            "entry": (
                "TOP_TWO_POSITIVE_UNDERREACTION_SCORES_ONLY"
            ),
            "trend_filter": (
                "BTC_AND_SATELLITE_CLOSE_ABOVE_CAUSAL_EMA200"
            ),
            "exit": (
                "FIXED_THREE_OR_FIVE_DAY_HOLD_OR_EARLY_TREND_LOSS"
            ),
            "execution": "SIGNAL_AT_DAILY_CLOSE_EXECUTE_NEXT_OPEN",
            "maximum_positions": 2,
            "position_weight": 0.20,
            "cash_yield": 0.0,
        },
        "base_known_trials": BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS,
        "projected_total_known_trials": (
            BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS
            + len(candidates)
        ),
        "bootstrap_block_days": 10,
        "forward_requirement": {
            "minimum_closed_daily_observations": 365,
            "minimum_rebalances": 30,
            "minimum_regime_coverage_per_state": 5,
        },
        "known_limitations": [
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
            "HISTORICAL_EVIDENCE_CONTAMINATED_BY_PRIOR_RESEARCH",
            "LONG_ONLY_IMPLEMENTATION_IS_NOT_MARKET_NEUTRAL",
            "BTC_ALTCOIN_LEAD_LAG_CAN_DECAY_OR_REVERSE",
            "FORWARD_EVIDENCE_REQUIRED",
            "NO_AUTOMATIC_PROMOTION",
        ],
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def plan_btc_shock_diffusion_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Persist or verify the immutable four-DNA campaign plan."""

    expected = _expected_plan()
    plan_path = (
        settings.paths.lab_dir
        / "reports"
        / "btc_shock_diffusion_plan_v1.json"
    )
    immutable_fields = (
        "campaign",
        "strategy_family",
        "engine_version",
        "trial_count",
        "strategy_dna_hashes",
        "strategy_dna",
        "search_space_hash",
        "selection_basis",
        "periods",
        "portfolio_policy",
        "signal_policy",
        "base_known_trials",
        "projected_total_known_trials",
        "bootstrap_block_days",
        "forward_requirement",
    )
    if plan_path.is_file():
        stored = read_json(plan_path)
        for field in immutable_fields:
            if _json_ready(stored.get(field)) != _json_ready(
                expected.get(field)
            ):
                raise RuntimeError(
                    f"BTC_SHOCK_DIFFUSION_PLAN_DRIFT:{field}"
                )
    else:
        atomic_write_json(plan_path, _json_ready(expected))
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
        forward_start=BTC_SHOCK_DIFFUSION_FORWARD_START,
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


def run_btc_shock_diffusion_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Execute the preregistered family and persist fail-closed evidence."""

    plan = plan_btc_shock_diffusion_campaign(settings)
    paths = _market_paths(settings)
    frames = {
        market: pd.read_parquet(path)
        for market, path in paths.items()
    }
    data_hashes = {
        market: sha256_file(path) for market, path in paths.items()
    }
    data_fingerprint = stable_hash(data_hashes, length=64)
    policy = _policy()
    candidates = btc_shock_diffusion_parameter_set()
    normal_results: dict[str, Any] = {}
    candidate_periods: dict[str, dict[str, Any]] = {}
    development_paths: dict[str, pd.Series] = {}
    for candidate in candidates:
        name = _candidate_name(candidate)
        result = backtest_btc_shock_diffusion(
            frames,
            candidate,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        normal_results[name] = result
        candidate_periods[name] = {}
        for period, bounds in BTC_SHOCK_DIFFUSION_PERIODS.items():
            metrics, returns = residual_reversal_period_metrics(
                result.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            candidate_periods[name][period] = metrics
            if period == "development":
                development_paths[name] = returns
    matrix = pd.concat(development_paths, axis=1).dropna()
    if matrix.empty or matrix.shape[1] != len(candidates):
        raise RuntimeError(
            "BTC_SHOCK_DIFFUSION_DEVELOPMENT_MATRIX_INVALID"
        )
    ranked_names = sorted(
        normal_results,
        key=lambda name: (
            -float(candidate_periods[name]["development"]["sharpe"]),
            name,
        ),
    )
    primary_name = ranked_names[0]
    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "btc_shock_diffusion_v1",
        campaign_id=BTC_SHOCK_DIFFUSION_CAMPAIGN,
    )
    registrations: dict[str, Any] = {}
    for rank, name in enumerate(ranked_names, start=1):
        result = normal_results[name]
        registrations[name] = registry.register(
            data_fingerprint=data_fingerprint,
            strategy_family=BTC_SHOCK_DIFFUSION_FAMILY,
            strategy_dna_hash=result.parameters.dna_hash,
            parameters=asdict(result.parameters),
            metrics_at_birth={
                **candidate_periods[name]["development"],
                "full_sample_metrics": result.metrics,
            },
            return_path_hash=stable_hash(
                [
                    round(float(value), 15)
                    for value in matrix[name].to_numpy(dtype=float)
                ],
                length=64,
            ),
            selection_metadata={
                "selection_basis": "DEVELOPMENT_SHARPE_ONLY",
                "development_rank": rank,
                "selected_primary": name == primary_name,
                "validation_used": False,
                "confirmation_used": False,
            },
        )
    registry_audit = registry.audit()
    total_known_trials = resolve_known_trial_count(
        settings.paths.lab_dir,
        local_known_trial_count=(
            BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS
            + int(registry_audit["unique_strategy_dna_count"])
        ),
    )
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
    stressed_results: dict[str, Any] = {}
    stressed_periods: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        name = _candidate_name(candidate)
        result = backtest_btc_shock_diffusion(
            frames,
            candidate,
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
            portfolio_policy=policy,
        )
        stressed_results[name] = result
        stressed_periods[name] = {
            period: residual_reversal_period_metrics(
                result.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )[0]
            for period, bounds in BTC_SHOCK_DIFFUSION_PERIODS.items()
        }

    primary = normal_results[primary_name]
    primary_stressed = stressed_results[primary_name]
    periods = candidate_periods[primary_name]
    primary_stressed_periods = stressed_periods[primary_name]
    stochastic = _stochastic_validation(
        settings,
        normal_equity=primary.equity_curve,
        stressed_equity=primary_stressed.equity_curve,
    )
    economic_checks = {
        "all_periods_positive": all(
            float(periods[period]["net_return"]) > 0.0
            for period in BTC_SHOCK_DIFFUSION_PERIODS
        ),
        "all_stressed_periods_positive": all(
            float(primary_stressed_periods[period]["net_return"])
            > 0.0
            for period in BTC_SHOCK_DIFFUSION_PERIODS
        ),
        "minimum_rebalances": (
            int(primary.metrics["rebalance_count"])
            >= settings.research.minimum_trades
        ),
        "minimum_effective_sample": (
            int(
                primary.metrics[
                    "portfolio_period_effective_sample_size"
                ]
            )
            >= settings.research.minimum_effective_sample_size
        ),
        "profit_factor": (
            float(primary.metrics["portfolio_period_profit_factor"])
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
                primary_stressed_periods["validation"][
                    "portfolio_period_profit_factor"
                ]
            )
            >= settings.research.minimum_stressed_profit_factor
        ),
        "maximum_drawdown": (
            abs(float(primary.metrics["maximum_drawdown"]))
            <= settings.research.maximum_drawdown
        ),
        "strictly_prior_factor_causality": bool(
            primary.integrity["strictly_prior_beta_estimation"]
            and primary.integrity["strictly_prior_shock_baseline"]
            and primary.integrity[
                "decision_at_close_execution_next_open"
            ]
        ),
        "exposure_limits_respected": all(
            bool(primary.integrity[field])
            for field in (
                "maximum_exposure_respected",
                "maximum_position_exposure_respected",
                "minimum_cash_respected",
                "maximum_positions_respected",
                "btc_used_as_regime_and_information_only",
            )
        ),
    }
    pbo = multiple.probability_of_backtest_overfitting
    statistical_checks = {
        "deflated_sharpe": (
            float(
                multiple.deflated_sharpe_probabilities.get(
                    primary_name,
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
        "pbo": bool(
            pbo is not None
            and pbo
            <= (
                settings.research
                .maximum_probability_of_backtest_overfitting
            )
        ),
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
    primary_result = {
        "strategy_id": primary_name,
        "strategy_dna_hash": primary.parameters.dna_hash,
        "parameters": asdict(primary.parameters),
        "registration": registrations[primary_name],
        "development_selection_rank": 1,
        "normal": primary.summary(),
        "periods": periods,
        "stressed": primary_stressed.summary(),
        "stressed_periods": primary_stressed_periods,
        "gates": {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "deflated_sharpe_probability": float(
                multiple.deflated_sharpe_probabilities.get(
                    primary_name,
                    0.0,
                )
            ),
            "pbo": pbo,
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    }

    observer_manifests: dict[str, str] = {}
    forward_summaries: dict[str, Any] = {}
    for name, result in normal_results.items():
        execution_identity = result.summary()["execution_identity"]
        source_identity = stable_hash(
            {
                "campaign": BTC_SHOCK_DIFFUSION_CAMPAIGN,
                "strategy_dna_hash": result.parameters.dna_hash,
                "portfolio_policy_hash": policy.policy_hash,
                "forward_start": (
                    BTC_SHOCK_DIFFUSION_FORWARD_START.isoformat()
                ),
            },
            length=64,
        )
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "btc_shock_diffusion_v1"
            / f"{name.lower()}.json"
        )
        observer = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "family": BTC_SHOCK_DIFFUSION_CAMPAIGN,
            "policy_name": name,
            "source_candidate_identity": source_identity,
            "strategy_dna_hash": result.parameters.dna_hash,
            "execution_identity": execution_identity,
            "parameters": asdict(result.parameters),
            "portfolio_policy": asdict(policy),
            "portfolio_policy_hash": policy.policy_hash,
            "forward_start": (
                BTC_SHOCK_DIFFUSION_FORWARD_START.isoformat()
            ),
            "forward_observation_timeframe": "1d",
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
                    strategy_dna_hash=result.parameters.dna_hash,
                    execution_identity=execution_identity,
                )
            )
        evidence = build_rotation_forward_evidence(
            result,
            frames,
            forward_start=BTC_SHOCK_DIFFUSION_FORWARD_START,
            minimum_observations=365,
            minimum_rebalances=30,
            performance_policy=_forward_policy(settings),
        )
        observer = merge_portfolio_forward_manifest(
            observer,
            evidence,
            source_candidate_identity=source_identity,
            strategy_dna_hash=result.parameters.dna_hash,
            execution_identity=execution_identity,
            forward_start=BTC_SHOCK_DIFFUSION_FORWARD_START,
        )
        observer["data_hashes"] = data_hashes
        atomic_write_json(observer_path, _json_ready(observer))
        observer_manifests[name] = str(observer_path)
        forward_summaries[name] = observer["forward_summary"]

    candidate_results = [
        {
            "strategy_id": name,
            "strategy_dna_hash": result.parameters.dna_hash,
            "parameters": asdict(result.parameters),
            "registration": registrations[name],
            "development_selection_rank": ranked_names.index(name) + 1,
            "normal": result.summary(),
            "periods": candidate_periods[name],
            "stressed": stressed_results[name].summary(),
            "stressed_periods": stressed_periods[name],
            "selected_primary": name == primary_name,
        }
        for name, result in normal_results.items()
    ]
    benchmarks = {
        mode: _benchmark_summary(
            frames,
            mode=mode,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
        )
        for mode in ("CASH", "BTC_TREND_20", "BTC_ETH_TREND_20_20")
    }
    report_path = btc_shock_diffusion_campaign_path(settings)
    payload = {
        "schema_version": "btc_shock_diffusion_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": BTC_SHOCK_DIFFUSION_CAMPAIGN,
        "strategy_family": BTC_SHOCK_DIFFUSION_FAMILY,
        "engine_version": BTC_SHOCK_DIFFUSION_ENGINE_VERSION,
        "timeframe": "1d",
        "plan": plan["plan"],
        "plan_sha256": plan["plan_sha256"],
        "search_space_hash": plan["search_space_hash"],
        "selection_basis": plan["selection_basis"],
        "selection_integrity": {
            "development_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
            "all_trials_registered": True,
        },
        "generated_trial_count": len(candidates),
        "registered_unique_trials": int(
            registry_audit["unique_strategy_dna_count"]
        ),
        "registered_epoch_records": int(
            registry_audit["unique_epoch_record_count"]
        ),
        "base_known_trials": BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS,
        "total_known_trials": total_known_trials,
        "primary_strategy_id": primary_name,
        "primary_result": primary_result,
        "candidate_results": candidate_results,
        "benchmarks": benchmarks,
        "multiple_testing": asdict(multiple),
        "pbo": pbo,
        "trial_registry": registry_audit,
        "data_fingerprint": data_fingerprint,
        "data_hashes": data_hashes,
        "periods": BTC_SHOCK_DIFFUSION_PERIODS,
        "portfolio_policy": asdict(policy),
        "signal_policy": plan["signal_policy"],
        "holdout_status": (
            "HISTORICAL_EVIDENCE_CONTAMINATED_BY_PRIOR_RESEARCH"
        ),
        "forward_requirement": plan["forward_requirement"],
        "observer_manifests": observer_manifests,
        "forward_summaries": forward_summaries,
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
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "shock_lookback": row["parameters"][
                    "shock_lookback"
                ],
                "maximum_holding_days": row["parameters"][
                    "maximum_holding_days"
                ],
                "development_rank": row[
                    "development_selection_rank"
                ],
                "development_net_return": row["periods"][
                    "development"
                ]["net_return"],
                "validation_net_return": row["periods"][
                    "validation"
                ]["net_return"],
                "confirmation_net_return": row["periods"][
                    "confirmation"
                ]["net_return"],
                "full_net_return": row["normal"]["metrics"][
                    "net_return"
                ],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "maximum_drawdown": row["normal"]["metrics"][
                    "maximum_drawdown"
                ],
                "average_exposure": row["normal"]["metrics"][
                    "average_exposure"
                ],
                "selected_primary": row["selected_primary"],
            }
            for row in candidate_results
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": payload["campaign"],
        "generated_trial_count": len(candidates),
        "registered_unique_trials": payload[
            "registered_unique_trials"
        ],
        "registered_epoch_records": payload[
            "registered_epoch_records"
        ],
        "total_known_trials": total_known_trials,
        "primary_strategy_id": primary_name,
        "pbo": pbo,
        "economic_pass": primary_result["gates"]["economic_pass"],
        "statistical_pass": primary_result["gates"][
            "statistical_pass"
        ],
        "report": str(report_path),
        "csv": str(csv_path),
        "observer_manifests": observer_manifests,
        "forward_summaries": forward_summaries,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }


__all__ = [
    "BTC_SHOCK_DIFFUSION_BASE_KNOWN_TRIALS",
    "BTC_SHOCK_DIFFUSION_CAMPAIGN",
    "btc_shock_diffusion_campaign_path",
    "plan_btc_shock_diffusion_campaign",
    "run_btc_shock_diffusion_campaign",
]
