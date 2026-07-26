"""Preregistered campaign service for confirmed liquidity-sweep recovery."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from pydantic import BaseModel

from config.settings import Settings
from research.forward_observer import (
    ForwardPerformanceGatePolicy,
    build_rotation_forward_evidence,
    merge_portfolio_forward_manifest,
    validate_forward_manifest_identity,
)
from research.liquidity_sweep import (
    DAILY_PERIODS_PER_YEAR,
    LIQUIDITY_SWEEP_ENGINE_VERSION,
    LIQUIDITY_SWEEP_FAMILY,
    LiquiditySweepParameters,
    backtest_liquidity_sweep,
    liquidity_sweep_parameter_set,
    liquidity_sweep_period_metrics,
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

LIQUIDITY_SWEEP_CAMPAIGN = "LIQUIDITY_SWEEP_RECOVERY_V1"
LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS = 21_329
LIQUIDITY_SWEEP_FORWARD_START = pd.Timestamp(
    "2026-07-26T00:00:00+00:00"
)
LIQUIDITY_SWEEP_PERIODS = {
    "development": ("2019-12-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-06-30"),
    "confirmation": ("2025-07-01", "2026-07-24"),
}


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


def liquidity_sweep_campaign_path(settings: Settings) -> Path:
    return (
        settings.paths.lab_dir
        / "reports"
        / "liquidity_sweep_campaign_v1.json"
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
            f"missing liquidity-sweep datasets: {missing}"
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


def _candidate_name(candidate: LiquiditySweepParameters) -> str:
    volume = int(round(candidate.minimum_relative_volume * 10))
    return (
        f"LS_F{candidate.fractal_side}_"
        f"V{volume:02d}_H{candidate.maximum_holding_days}"
    )


def _expected_plan() -> dict[str, Any]:
    candidates = liquidity_sweep_parameter_set()
    policy = _policy()
    return {
        "schema_version": "liquidity_sweep_plan_v1",
        "status": "CAMPAIGN_PLAN",
        "campaign": LIQUIDITY_SWEEP_CAMPAIGN,
        "strategy_family": LIQUIDITY_SWEEP_FAMILY,
        "engine_version": LIQUIDITY_SWEEP_ENGINE_VERSION,
        "timeframe": "1d",
        "economic_hypothesis": (
            "A recovery above a previously confirmed fractal low after "
            "an intraday liquidity sweep, supported by contemporaneously "
            "known spot volume and an established BTC/asset EMA200 regime, "
            "can capture event-driven continuation without relying on "
            "cross-sectional momentum ranking."
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
        "periods": LIQUIDITY_SWEEP_PERIODS,
        "portfolio_policy": asdict(policy),
        "signal_policy": {
            "entry": (
                "CONFIRMED_FRACTAL_LOW_SWEEP_AND_CLOSE_RECOVERY"
            ),
            "trend_filter": "ASSET_AND_BTC_ABOVE_EMA200",
            "volume": "CURRENT_CLOSED_DAILY_VOLUME_VS_ROLLING20",
            "exit": (
                "BEARISH_SWEEP_OR_EMA200_LOSS_OR_CONFIRMED_HIGH_"
                "RECOVERY_OR_MAXIMUM_HOLD"
            ),
            "execution": "SIGNAL_AT_DAILY_CLOSE_EXECUTE_NEXT_OPEN",
            "ranking": "RECOVERY_FRACTION_TIMES_RELATIVE_VOLUME",
            "maximum_positions": 2,
            "position_weight": 0.20,
            "cash_yield": 0.0,
        },
        "base_known_trials": LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS,
        "projected_total_known_trials": (
            LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS + len(candidates)
        ),
        "bootstrap_block_days": 10,
        "forward_requirement": {
            "minimum_closed_daily_observations": 365,
            "minimum_rebalances": 30,
            "minimum_regime_coverage_per_state": 5,
        },
        "known_limitations": [
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
            "SPOT_VOLUME_IS_EXCHANGE_SPECIFIC",
            "FORWARD_EVIDENCE_REQUIRED",
            "NO_AUTOMATIC_PROMOTION",
        ],
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def plan_liquidity_sweep_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Persist or verify the immutable eight-DNA campaign plan."""

    expected = _expected_plan()
    plan_path = (
        settings.paths.lab_dir
        / "reports"
        / "liquidity_sweep_plan_v1.json"
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
                    f"LIQUIDITY_SWEEP_PLAN_DRIFT:{field}"
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
        forward_start=LIQUIDITY_SWEEP_FORWARD_START,
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
    return validate_strategy_return_paths(
        normal_equity.pct_change(fill_method=None)
        .dropna()
        .to_numpy(dtype=float),
        stressed_equity.pct_change(fill_method=None)
        .dropna()
        .to_numpy(dtype=float),
        policy=policy,
        seed_offset=180_000,
    )


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
    target = pd.DataFrame(
        0.0,
        index=opens.index,
        columns=opens.columns,
    )
    ema = closes.ewm(span=200, adjust=False, min_periods=200).mean()
    btc_regime = closes["BTC-EUR"] > ema["BTC-EUR"]
    if mode == "BTC_TREND_20":
        target["BTC-EUR"] = btc_regime.astype(float) * 0.20
    elif mode == "BTC_ETH_TREND_20_20":
        target["BTC-EUR"] = btc_regime.astype(float) * 0.20
        target["ETH-EUR"] = (
            btc_regime & (closes["ETH-EUR"] > ema["ETH-EUR"])
        ).astype(float) * 0.20
    elif mode != "CASH":
        raise ValueError(f"unsupported benchmark mode: {mode}")
    executed = target.shift(1).fillna(0.0).iloc[201:].copy()
    open_returns = opens.shift(-1).div(opens).sub(1.0)
    open_returns.iloc[-1] = (
        closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    )
    gross = (
        executed * open_returns.reindex(executed.index).fillna(0.0)
    ).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].sum())
    turnover.iloc[-1] += float(executed.iloc[-1].sum())
    cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    net = (1.0 - turnover * cost) * (1.0 + gross) - 1.0
    equity = (1.0 + net).cumprod()
    standard = float(net.std(ddof=0))
    return {
        "benchmark": mode,
        "net_return": float(equity.iloc[-1] - 1.0),
        "sharpe": (
            float(
                net.mean()
                / standard
                * math.sqrt(DAILY_PERIODS_PER_YEAR)
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


def run_liquidity_sweep_campaign(
    settings: Settings,
) -> dict[str, Any]:
    """Execute the preregistered family and persist fail-closed evidence."""

    plan = plan_liquidity_sweep_campaign(settings)
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
    candidates = liquidity_sweep_parameter_set()
    normal_results: dict[str, Any] = {}
    candidate_periods: dict[str, dict[str, Any]] = {}
    development_paths: dict[str, pd.Series] = {}
    for candidate in candidates:
        name = _candidate_name(candidate)
        result = backtest_liquidity_sweep(
            frames,
            candidate,
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            portfolio_policy=policy,
        )
        normal_results[name] = result
        candidate_periods[name] = {}
        for period, bounds in LIQUIDITY_SWEEP_PERIODS.items():
            metrics, returns = liquidity_sweep_period_metrics(
                result.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )
            candidate_periods[name][period] = metrics
            if period == "development":
                development_paths[name] = returns
    matrix = pd.concat(development_paths, axis=1).dropna()
    if matrix.empty or matrix.shape[1] != len(candidates):
        raise RuntimeError("LIQUIDITY_SWEEP_DEVELOPMENT_MATRIX_INVALID")
    development_sharpes = {
        name: float(candidate_periods[name]["development"]["sharpe"])
        for name in normal_results
    }
    primary_name = max(
        sorted(development_sharpes),
        key=lambda name: development_sharpes[name],
    )

    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / "liquidity_sweep_v1",
        campaign_id=LIQUIDITY_SWEEP_CAMPAIGN,
    )
    registrations: dict[str, Any] = {}
    for rank, name in enumerate(
        sorted(
            development_sharpes,
            key=lambda key: development_sharpes[key],
            reverse=True,
        ),
        start=1,
    ):
        result = normal_results[name]
        registrations[name] = registry.register(
            data_fingerprint=data_fingerprint,
            strategy_family=LIQUIDITY_SWEEP_FAMILY,
            strategy_dna_hash=result.parameters.dna_hash,
            parameters=asdict(result.parameters),
            metrics_at_birth={
                **candidate_periods[name]["development"],
                "full_sample_metrics": result.summary()["metrics"],
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
    total_known_trials = (
        LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS
        + int(registry_audit["unique_strategy_dna_count"])
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
        stressed = backtest_liquidity_sweep(
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
        stressed_results[name] = stressed
        stressed_periods[name] = {
            period: liquidity_sweep_period_metrics(
                stressed.equity_curve,
                start=bounds[0],
                end=bounds[1],
            )[0]
            for period, bounds in LIQUIDITY_SWEEP_PERIODS.items()
        }
    primary = normal_results[primary_name]
    primary_stressed = stressed_results[primary_name]
    stochastic = _stochastic_validation(
        settings,
        normal_equity=primary.equity_curve,
        stressed_equity=primary_stressed.equity_curve,
    )
    periods = candidate_periods[primary_name]
    primary_stressed_periods = stressed_periods[primary_name]
    economic_checks = {
        "all_periods_positive": all(
            float(periods[period]["net_return"]) > 0.0
            for period in LIQUIDITY_SWEEP_PERIODS
        ),
        "all_stressed_periods_positive": all(
            float(
                primary_stressed_periods[period]["net_return"]
            )
            > 0.0
            for period in LIQUIDITY_SWEEP_PERIODS
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
        "confirmed_fractal_causality": bool(
            primary.integrity["confirmed_fractals_only"]
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
                "campaign": LIQUIDITY_SWEEP_CAMPAIGN,
                "strategy_dna_hash": result.parameters.dna_hash,
                "portfolio_policy_hash": policy.policy_hash,
                "forward_start": (
                    LIQUIDITY_SWEEP_FORWARD_START.isoformat()
                ),
            },
            length=64,
        )
        observer_path = (
            settings.paths.lab_dir
            / "observers"
            / "liquidity_sweep_v1"
            / f"{name.lower()}.json"
        )
        observer = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "family": LIQUIDITY_SWEEP_CAMPAIGN,
            "policy_name": name,
            "source_candidate_identity": source_identity,
            "strategy_dna_hash": result.parameters.dna_hash,
            "execution_identity": execution_identity,
            "parameters": asdict(result.parameters),
            "portfolio_policy": asdict(policy),
            "portfolio_policy_hash": policy.policy_hash,
            "forward_start": (
                LIQUIDITY_SWEEP_FORWARD_START.isoformat()
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
            forward_start=LIQUIDITY_SWEEP_FORWARD_START,
            minimum_observations=365,
            minimum_rebalances=30,
            performance_policy=ForwardPerformanceGatePolicy(
                minimum_profit_factor=(
                    settings.research.minimum_profit_factor
                ),
                minimum_stressed_profit_factor=(
                    settings.research.minimum_stressed_profit_factor
                ),
                maximum_drawdown=(
                    settings.research.maximum_drawdown
                ),
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
            source_candidate_identity=source_identity,
            strategy_dna_hash=result.parameters.dna_hash,
            execution_identity=execution_identity,
            forward_start=LIQUIDITY_SWEEP_FORWARD_START,
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
            "development_selection_rank": sorted(
                development_sharpes,
                key=lambda key: development_sharpes[key],
                reverse=True,
            ).index(name)
            + 1,
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
    report_path = liquidity_sweep_campaign_path(settings)
    payload = {
        "schema_version": "liquidity_sweep_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": LIQUIDITY_SWEEP_CAMPAIGN,
        "strategy_family": LIQUIDITY_SWEEP_FAMILY,
        "engine_version": LIQUIDITY_SWEEP_ENGINE_VERSION,
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
        "base_known_trials": LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS,
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
        "periods": LIQUIDITY_SWEEP_PERIODS,
        "portfolio_policy": asdict(policy),
        "signal_policy": plan["signal_policy"],
        "holdout_status": (
            "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS"
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
    "LIQUIDITY_SWEEP_BASE_KNOWN_TRIALS",
    "LIQUIDITY_SWEEP_CAMPAIGN",
    "liquidity_sweep_campaign_path",
    "plan_liquidity_sweep_campaign",
    "run_liquidity_sweep_campaign",
]
