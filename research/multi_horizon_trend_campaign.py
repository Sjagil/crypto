"""Preregistered campaign for one fixed multi-horizon trend ensemble."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel

from config.settings import Settings
from research.forward_observer import (
    ForwardPerformanceGatePolicy,
    build_rotation_forward_evidence,
    merge_portfolio_forward_manifest,
    validate_forward_manifest_identity,
)
from research.global_trial_accounting import resolve_known_trial_count
from research.multi_horizon_trend import (
    DAILY_PERIODS_PER_YEAR,
    MULTI_HORIZON_TREND_ENGINE_VERSION,
    MULTI_HORIZON_TREND_FAMILY,
    backtest_multi_horizon_trend,
    multi_horizon_trend_parameter_set,
    multi_horizon_trend_period_metrics,
)
from research.optimization import multiple_testing_bootstrap
from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _profit_factor,
    _validated_panel,
)
from research.stochastic_validation import (
    policy_from_research_settings,
    validate_strategy_return_paths,
)
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
)

MULTI_HORIZON_TREND_CAMPAIGN = "MULTI_HORIZON_TREND_V1"
MULTI_HORIZON_TREND_FORWARD_START = pd.Timestamp(
    "2026-07-26T00:00:00+00:00"
)
MULTI_HORIZON_TREND_PERIODS = {
    "development": ("2019-12-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "confirmation": ("2025-07-01", "2026-07-25"),
}
MULTI_HORIZON_TREND_POLICY_NAME = "MHT_20_60_120_240_STRUCT240"
MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, Path, Decimal)):
        return str(value)
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def multi_horizon_trend_campaign_path(settings: Settings) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "multi_horizon_trend_campaign_v1.json"
    )


def _market_paths(settings: Settings) -> dict[str, Path]:
    paths = {
        market: settings.paths.processed_data_dir / f"{market}_1d.parquet"
        for market in MARKETS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing multi-horizon trend datasets: {missing}"
        )
    return paths


def _policy() -> RotationPortfolioPolicy:
    return RotationPortfolioPolicy(
        allowed_markets=MARKETS,
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=241,
    )


def _expected_plan() -> dict[str, Any]:
    candidate = multi_horizon_trend_parameter_set()[0]
    policy = _policy()
    return {
        "schema_version": "multi_horizon_trend_plan_v1",
        "status": "CAMPAIGN_PLAN",
        "campaign": MULTI_HORIZON_TREND_CAMPAIGN,
        "strategy_family": MULTI_HORIZON_TREND_FAMILY,
        "engine_version": MULTI_HORIZON_TREND_ENGINE_VERSION,
        "timeframe": "1d",
        "economic_hypothesis": (
            "Persistent crypto trends occur across multiple daily horizons. "
            "Requiring positive 240-day momentum and scaling each allowed "
            "asset by the fraction of positive 20/60/120/240-day votes can "
            "capture durable upside while retaining at least 60% cash."
        ),
        "trial_count": 1,
        "strategy_dna_hashes": [candidate.dna_hash],
        "strategy_dna": [asdict(candidate)],
        "search_space_hash": stable_hash(
            [candidate.dna_hash],
            length=64,
        ),
        "selection_basis": "SINGLE_FIXED_DNA_NO_HISTORICAL_SELECTION",
        "discovery_governance": {
            "preregistered_before_result_inspection": True,
            "within_family_parameter_search": False,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
            "global_historical_holdout_available": False,
            "historical_results_cannot_authorize_promotion": True,
            "forward_evidence_is_decisive": True,
        },
        "periods": MULTI_HORIZON_TREND_PERIODS,
        "portfolio_policy": asdict(policy),
        "execution_policy": {
            "signal": "CLOSED_DAILY_CANDLE_ONLY",
            "execution": "NEXT_DAILY_OPEN",
            "rebalance": "DAILY_ONLY_WHEN_TARGET_CHANGES",
            "cash_yield": 0.0,
            "cost_stress_multiplier": 2.0,
        },
        "global_trial_accounting": (
            "AUTHORITATIVE_CONTENT_ADDRESSED_DENOMINATOR"
        ),
        "bootstrap_block_days": 10,
        "pbo_policy": (
            "NOT_APPLICABLE_SINGLE_FIXED_DNA_NO_WITHIN_FAMILY_SELECTION"
        ),
        "forward_requirement": {
            "minimum_closed_daily_observations": 365,
            "minimum_rebalances": 30,
            "minimum_regime_coverage_per_state": 5,
        },
        "known_limitations": [
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
            "PBO_NOT_APPLICABLE_DOES_NOT_MEAN_PBO_PASSED",
            "FORWARD_EVIDENCE_REQUIRED",
            "NO_AUTOMATIC_PROMOTION",
        ],
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def plan_multi_horizon_trend_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Persist or verify the immutable single-DNA campaign plan."""

    expected = _expected_plan()
    plan_path = (
        settings.paths.lab_dir
        / "reports"
        / "multi_horizon_trend_plan_v1.json"
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
        "discovery_governance",
        "periods",
        "portfolio_policy",
        "execution_policy",
        "global_trial_accounting",
        "bootstrap_block_days",
        "pbo_policy",
        "forward_requirement",
    )
    if plan_path.is_file():
        stored = read_json(plan_path)
        for field in immutable_fields:
            if _json_ready(stored.get(field)) != _json_ready(
                expected.get(field)
            ):
                raise RuntimeError(
                    f"MULTI_HORIZON_TREND_PLAN_DRIFT:{field}"
                )
    else:
        atomic_write_json(plan_path, _json_ready(expected))
    return {
        **expected,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
    }


def _mean_block_bootstrap_ci_lower(
    returns: pd.Series,
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < max(30, block_size * 2):
        return -math.inf
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(values) / block_size))
    starts = np.arange(0, len(values) - block_size + 1)
    simulated = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(starts, size=blocks, replace=True)
        path = np.concatenate(
            [
                values[start : start + block_size]
                for start in selected
            ]
        )[: len(values)]
        simulated[sample] = float(path.mean())
    return float(np.quantile(simulated, 0.025))


def _stochastic_validation(
    settings: Settings,
    *,
    normal_equity: pd.Series,
    stressed_equity: pd.Series,
) -> dict[str, Any]:
    policy = policy_from_research_settings(
        settings.research,
        seed=settings.app.random_seed,
        expected_block_length=10,
    )
    # The autopilot validates many campaigns sequentially on Windows. A small
    # deterministic batch bounds peak memory without changing the simulation
    # count, thresholds, strategy DNA, or trial accounting.
    policy = replace(policy, batch_size=min(policy.batch_size, 64))
    normal_returns = (
        normal_equity.pct_change(fill_method=None)
        .dropna()
        .to_numpy(dtype=float)
    )
    stressed_returns = (
        stressed_equity.pct_change(fill_method=None)
        .dropna()
        .to_numpy(dtype=float)
    )
    return validate_strategy_return_paths(
        normal_returns,
        stressed_returns,
        policy=policy,
        seed_offset=220_000,
    )


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
        forward_start=MULTI_HORIZON_TREND_FORWARD_START,
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


def _benchmark_summary(
    frames: Mapping[str, pd.DataFrame],
    *,
    mode: str,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
) -> dict[str, Any]:
    opens, closes = _validated_panel(
        frames,
        benchmark_market="BTC-EUR",
        portfolio_policy=_policy(),
    )
    target = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    if mode == "BTC_BUY_HOLD_20":
        target["BTC-EUR"] = 0.20
    elif mode == "EQUAL_ALLOWED_BUY_HOLD_40":
        target.loc[:, :] = 0.10
    elif mode != "CASH":
        raise ValueError(f"unsupported benchmark mode: {mode}")
    executed = target.iloc[242:].copy()
    open_returns = opens.shift(-1).div(opens).sub(1.0)
    open_returns.iloc[-1] = closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    open_returns = open_returns.reindex(executed.index).fillna(0.0)
    gross = (executed * open_returns).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].abs().sum())
    turnover.iloc[-1] += float(executed.iloc[-1].sum())
    one_way_cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    net = (1.0 - turnover * one_way_cost) * (1.0 + gross) - 1.0
    equity = (1.0 + net).cumprod()
    standard = float(net.std(ddof=0))
    return {
        "benchmark": mode,
        "net_return": float(equity.iloc[-1] - 1.0),
        "sharpe": (
            float(
                net.mean() / standard * math.sqrt(DAILY_PERIODS_PER_YEAR)
            )
            if standard > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(
            (equity / equity.cummax() - 1.0).min()
        ),
        "portfolio_period_profit_factor": _profit_factor(net),
        "average_exposure": float(executed.sum(axis=1).mean()),
        "orders_generated": 0,
    }


def run_multi_horizon_trend_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Execute the fixed DNA and persist fail-closed evidence."""

    plan = plan_multi_horizon_trend_campaign(settings)
    paths = _market_paths(settings)
    frames = {
        market: pd.read_parquet(path) for market, path in paths.items()
    }
    data_hashes = {
        market: sha256_file(path) for market, path in paths.items()
    }
    data_fingerprint = stable_hash(data_hashes, length=64)
    policy = _policy()
    candidate = multi_horizon_trend_parameter_set()[0]
    normal = backtest_multi_horizon_trend(
        frames,
        candidate,
        fee_rate=settings.costs.default_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    periods: dict[str, Any] = {}
    period_returns: dict[str, pd.Series] = {}
    for period, bounds in MULTI_HORIZON_TREND_PERIODS.items():
        metrics, returns = multi_horizon_trend_period_metrics(
            normal.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )
        periods[period] = metrics
        period_returns[period] = returns
    development_returns = period_returns["development"]
    if development_returns.empty:
        raise RuntimeError("MULTI_HORIZON_DEVELOPMENT_RETURNS_MISSING")

    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "multi_horizon_trend_v1",
        campaign_id=MULTI_HORIZON_TREND_CAMPAIGN,
    )
    registration = registry.register(
        data_fingerprint=data_fingerprint,
        strategy_family=MULTI_HORIZON_TREND_FAMILY,
        strategy_dna_hash=candidate.dna_hash,
        parameters=asdict(candidate),
        metrics_at_birth={
            **periods["development"],
            "full_sample_metrics": normal.summary()["metrics"],
        },
        return_path_hash=stable_hash(
            [
                round(float(value), 15)
                for value in development_returns.to_numpy(dtype=float)
            ],
            length=64,
        ),
        selection_metadata={
            "selection_basis": "SINGLE_FIXED_DNA",
            "development_rank": None,
            "validation_used": False,
            "confirmation_used": False,
        },
    )
    registry_audit = registry.audit()
    total_known_trials = resolve_known_trial_count(
        settings.paths.lab_dir,
        local_known_trial_count=int(
            registry_audit["unique_strategy_dna_count"]
        ),
    )
    multiple = multiple_testing_bootstrap(
        development_returns.to_frame(
            name=MULTI_HORIZON_TREND_POLICY_NAME
        ),
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
    stressed = backtest_multi_horizon_trend(
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
    stressed_periods = {
        period: multi_horizon_trend_period_metrics(
            stressed.equity_curve,
            start=bounds[0],
            end=bounds[1],
        )[0]
        for period, bounds in MULTI_HORIZON_TREND_PERIODS.items()
    }
    stochastic = _stochastic_validation(
        settings,
        normal_equity=normal.equity_curve,
        stressed_equity=stressed.equity_curve,
    )
    confirmation_ci_lower = _mean_block_bootstrap_ci_lower(
        period_returns["confirmation"],
        samples=max(
            1_000,
            settings.research.multiple_testing_bootstrap_samples,
        ),
        block_size=10,
        seed=settings.app.random_seed + 220_001,
    )
    economic_checks = {
        "all_periods_positive": all(
            float(periods[period]["net_return"]) > 0.0
            for period in MULTI_HORIZON_TREND_PERIODS
        ),
        "all_stressed_periods_positive": all(
            float(stressed_periods[period]["net_return"]) > 0.0
            for period in MULTI_HORIZON_TREND_PERIODS
        ),
        "confirmation_expectancy_ci_lower_positive": (
            confirmation_ci_lower > 0.0
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
        "causal_execution": bool(
            normal.integrity["closed_candles_only"]
            and normal.integrity[
                "decision_at_close_execution_next_open"
            ]
            and normal.integrity["point_in_time_history_gate"]
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
    }
    dsr = float(
        multiple.deflated_sharpe_probabilities.get(
            MULTI_HORIZON_TREND_POLICY_NAME,
            0.0,
        )
    )
    statistical_checks = {
        "deflated_sharpe": (
            dsr
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
        "monte_carlo": bool(
            stochastic["normal"]["monte_carlo"]["passed"]
            and stochastic["stressed"]["monte_carlo"]["passed"]
        ),
        "dirichlet": bool(
            stochastic["normal"]["dirichlet"]["passed"]
            and stochastic["stressed"]["dirichlet"]["passed"]
        ),
        "historical_selection_uncontaminated": False,
        "untouched_holdout": False,
    }
    primary_result = {
        "strategy_id": MULTI_HORIZON_TREND_POLICY_NAME,
        "strategy_dna_hash": candidate.dna_hash,
        "parameters": asdict(candidate),
        "registration": registration,
        "normal": normal.summary(),
        "periods": periods,
        "stressed": stressed.summary(),
        "stressed_periods": stressed_periods,
        "gates": {
            "economic_checks": economic_checks,
            "statistical_checks": statistical_checks,
            "confirmation_daily_expectancy_ci_lower_95": (
                confirmation_ci_lower
            ),
            "deflated_sharpe_probability": dsr,
            "pbo": {
                "applicable": False,
                "value": None,
                "reason": (
                    "SINGLE_FIXED_DNA_NO_WITHIN_FAMILY_SELECTION"
                ),
                "not_applicable_is_not_a_pass": True,
            },
            "stochastic_validation": stochastic,
            "economic_pass": all(economic_checks.values()),
            "statistical_pass": all(statistical_checks.values()),
            "research_pass": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
    }

    execution_identity = normal.summary()["execution_identity"]
    source_candidate_identity = stable_hash(
        {
            "campaign": MULTI_HORIZON_TREND_CAMPAIGN,
            "strategy_dna_hash": candidate.dna_hash,
            "portfolio_policy_hash": policy.policy_hash,
            "forward_start": (
                MULTI_HORIZON_TREND_FORWARD_START.isoformat()
            ),
        },
        length=64,
    )
    observer_path = (
        settings.paths.lab_dir
        / "observers"
        / "multi_horizon_trend_v1"
        / f"{MULTI_HORIZON_TREND_POLICY_NAME.lower()}.json"
    )
    observer = {
        "status": "FROZEN_FORWARD_RESEARCH",
        "family": MULTI_HORIZON_TREND_CAMPAIGN,
        "policy_name": MULTI_HORIZON_TREND_POLICY_NAME,
        "source_candidate_identity": source_candidate_identity,
        "strategy_dna_hash": candidate.dna_hash,
        "execution_identity": execution_identity,
        "parameters": asdict(candidate),
        "portfolio_policy": asdict(policy),
        "portfolio_policy_hash": policy.policy_hash,
        "forward_start": MULTI_HORIZON_TREND_FORWARD_START.isoformat(),
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
                source_candidate_identity=source_candidate_identity,
                strategy_dna_hash=candidate.dna_hash,
                execution_identity=execution_identity,
            )
        )
    evidence = build_rotation_forward_evidence(
        normal,
        frames,
        forward_start=MULTI_HORIZON_TREND_FORWARD_START,
        minimum_observations=365,
        minimum_rebalances=30,
        performance_policy=ForwardPerformanceGatePolicy(
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
        ),
    )
    observer = merge_portfolio_forward_manifest(
        observer,
        evidence,
        source_candidate_identity=source_candidate_identity,
        strategy_dna_hash=candidate.dna_hash,
        execution_identity=execution_identity,
        forward_start=MULTI_HORIZON_TREND_FORWARD_START,
    )
    observer["data_hashes"] = data_hashes
    atomic_write_json(observer_path, _json_ready(observer))

    benchmarks = {
        mode: _benchmark_summary(
            frames,
            mode=mode,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
        )
        for mode in (
            "CASH",
            "BTC_BUY_HOLD_20",
            "EQUAL_ALLOWED_BUY_HOLD_40",
        )
    }
    report_path = multi_horizon_trend_campaign_path(settings)
    payload = {
        "schema_version": "multi_horizon_trend_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": MULTI_HORIZON_TREND_CAMPAIGN,
        "strategy_family": MULTI_HORIZON_TREND_FAMILY,
        "engine_version": MULTI_HORIZON_TREND_ENGINE_VERSION,
        "timeframe": "1d",
        "plan": plan["plan"],
        "plan_sha256": plan["plan_sha256"],
        "search_space_hash": plan["search_space_hash"],
        "selection_basis": plan["selection_basis"],
        "selection_integrity": {
            "single_fixed_dna": True,
            "within_family_historical_selection": False,
            "preregistered_before_result_inspection": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "generated_trial_count": 1,
        "registered_unique_trials": int(
            registry_audit["unique_strategy_dna_count"]
        ),
        "registered_epoch_records": int(
            registry_audit["unique_epoch_record_count"]
        ),
        "total_known_trials": total_known_trials,
        "primary_strategy_id": MULTI_HORIZON_TREND_POLICY_NAME,
        "primary_result": primary_result,
        "candidate_results": [primary_result],
        "benchmarks": benchmarks,
        "multiple_testing": asdict(multiple),
        "pbo": None,
        "pbo_applicable": False,
        "pbo_policy": plan["pbo_policy"],
        "trial_registry": registry_audit,
        "data_fingerprint": data_fingerprint,
        "data_hashes": data_hashes,
        "periods": MULTI_HORIZON_TREND_PERIODS,
        "portfolio_policy": asdict(policy),
        "execution_policy": plan["execution_policy"],
        "discovery_governance": plan["discovery_governance"],
        "holdout_status": (
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS"
        ),
        "forward_requirement": plan["forward_requirement"],
        "observer_manifests": {
            MULTI_HORIZON_TREND_POLICY_NAME: str(observer_path)
        },
        "forward_summaries": {
            MULTI_HORIZON_TREND_POLICY_NAME: observer["forward_summary"]
        },
        "economic_pass": all(economic_checks.values()),
        "statistical_pass": all(statistical_checks.values()),
        "research_pass": False,
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
                "strategy_id": MULTI_HORIZON_TREND_POLICY_NAME,
                "strategy_dna_hash": candidate.dna_hash,
                "development_net_return": periods["development"][
                    "net_return"
                ],
                "validation_net_return": periods["validation"][
                    "net_return"
                ],
                "confirmation_net_return": periods["confirmation"][
                    "net_return"
                ],
                "stressed_confirmation_net_return": stressed_periods[
                    "confirmation"
                ]["net_return"],
                "full_net_return": normal.metrics["net_return"],
                "annualized_return": normal.metrics["annualized_return"],
                "sharpe": normal.metrics["sharpe"],
                "maximum_drawdown": normal.metrics["maximum_drawdown"],
                "average_exposure": normal.metrics["average_exposure"],
                "profit_factor": normal.metrics[
                    "portfolio_period_profit_factor"
                ],
                "effective_sample_size": normal.metrics[
                    "portfolio_period_effective_sample_size"
                ],
                "confirmation_ci_lower": confirmation_ci_lower,
                "deflated_sharpe_probability": dsr,
                "white_reality_check_pvalue": (
                    multiple.white_reality_check_pvalue
                ),
                "hansen_spa_pvalue": multiple.hansen_spa_pvalue,
                "economic_pass": all(economic_checks.values()),
                "statistical_pass": all(statistical_checks.values()),
                "orders_generated": 0,
                "live_ready": False,
            }
        ]
    ).to_csv(csv_path, index=False)
    return {
        "campaign": MULTI_HORIZON_TREND_CAMPAIGN,
        "status": payload["status"],
        "report": str(report_path),
        "csv": str(csv_path),
        "plan": plan["plan"],
        "generated_trial_count": 1,
        "registered_unique_trials": payload[
            "registered_unique_trials"
        ],
        "registered_epoch_records": payload[
            "registered_epoch_records"
        ],
        "total_known_trials": total_known_trials,
        "primary_strategy_id": MULTI_HORIZON_TREND_POLICY_NAME,
        "pbo": None,
        "pbo_applicable": False,
        "economic_pass": payload["economic_pass"],
        "statistical_pass": payload["statistical_pass"],
        "observer_manifests": payload["observer_manifests"],
        "orders_generated": 0,
        "live_ready": False,
    }


__all__ = [
    "MULTI_HORIZON_TREND_CAMPAIGN",
    "MULTI_HORIZON_TREND_FORWARD_START",
    "MULTI_HORIZON_TREND_PERIODS",
    "MULTI_HORIZON_TREND_POLICY_NAME",
    "multi_horizon_trend_campaign_path",
    "plan_multi_horizon_trend_campaign",
    "run_multi_horizon_trend_campaign",
]
