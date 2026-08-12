"""Frozen robustness replay for exact-positive classical strategy DNA."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from reporting.strategy_evidence_charts import generate_strategy_evidence_bundle
from research.backtest import BacktestConfig, BacktestEngine, BacktestResult
from research.combinatorial_lab import (
    CombinationGenerator,
    CombinatorialStrategy,
    LabRunner,
    LogicMode,
    signal_block_registry,
)
from research.stochastic_validation import (
    policy_from_research_settings,
    validate_strategy_return_paths,
)
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso


def _daily_returns(result: BacktestResult) -> pd.Series:
    if result.equity_curve.empty:
        return pd.Series(dtype=float)
    return (
        result.equity_curve["equity"]
        .astype(float)
        .resample("1D")
        .last()
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _combination(candidate: Mapping[str, Any]):
    registry = signal_block_registry()
    blocks = tuple(str(item) for item in candidate.get("block_ids") or [])
    generated = CombinationGenerator(registry).generate(
        sizes=(len(blocks),),
        logic_modes=(LogicMode(str(candidate.get("logic_mode") or "LAYERED")),),
        block_ids=blocks,
        timeframes=(str(candidate["timeframe"]),),
    )
    exact = [row for row in generated if row.block_ids == tuple(sorted(blocks))]
    if len(exact) != 1:
        raise ValueError("FROZEN_CLASSICAL_COMBINATION_NOT_RECONSTRUCTABLE")
    if exact[0].strategy_dna_hash != str(candidate["strategy_dna_hash"]):
        raise ValueError("FROZEN_CLASSICAL_STRATEGY_DNA_MISMATCH")
    return exact[0], registry


def _metric_subset(result: BacktestResult) -> dict[str, Any]:
    return {
        key: result.metrics.get(key)
        for key in (
            "net_return",
            "cagr",
            "profit_factor",
            "net_expectancy_r",
            "sharpe",
            "sortino",
            "calmar",
            "maximum_drawdown",
            "trade_count",
            "effective_sample_size",
            "average_exposure",
            "transaction_costs_eur",
            "turnover",
            "monte_carlo_p95_drawdown",
            "probability_of_loss",
        )
    }


def validate_classical_positive_candidates(
    settings: Settings,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay frozen candidates under normal/stressed costs, MC and Dirichlet."""

    report_path = (
        settings.paths.lab_dir
        / "reports"
        / "classical_positive_robustness_v1.json"
    )
    cached = (
        dict(read_json(report_path))
        if report_path.is_file()
        else {
            "schema_version": "classical_positive_robustness_v1",
            "candidates": [],
        }
    )
    cached_by_dna = {
        str(row.get("strategy_dna_hash")): dict(row)
        for row in cached.get("candidates") or []
    }
    runner = LabRunner(settings)
    frame_cache: dict[tuple[Any, ...], dict[str, pd.DataFrame]] = {}
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        dna = str(candidate["strategy_dna_hash"])
        frozen_hash = str(candidate["frozen_candidate_hash"])
        prior = cached_by_dna.get(dna)
        if prior and str(prior.get("frozen_candidate_hash")) == frozen_hash:
            results.append(prior)
            continue
        period = dict(candidate.get("data_period") or {})
        cache_key = (
            tuple(candidate.get("markets") or []),
            str(candidate["timeframe"]),
            period.get("start"),
            period.get("end"),
        )
        frames = frame_cache.get(cache_key)
        if frames is None:
            frames, _, _ = runner._frames(
                markets=list(candidate.get("markets") or []),
                timeframe=str(candidate["timeframe"]),
                rows=None,
                data_mode="real",
                start_at=pd.Timestamp(period["start"]) if period.get("start") else None,
                end_at=pd.Timestamp(period["end"]) if period.get("end") else None,
            )
            frame_cache[cache_key] = frames
        combination, registry = _combination(candidate)
        strategy = CombinatorialStrategy(
            combination,
            registry,
            block_parameters=dict(candidate.get("parameters") or {}),
        )
        normal = BacktestEngine(
            BacktestConfig.from_settings(settings, stressed=False),
            settings=settings,
        ).run(frames, strategy)
        stressed = BacktestEngine(
            BacktestConfig.from_settings(settings, stressed=True),
            settings=settings,
        ).run(frames, strategy)
        normal_returns = _daily_returns(normal)
        stressed_returns = _daily_returns(stressed)
        common = normal_returns.index.intersection(stressed_returns.index)
        stochastic = validate_strategy_return_paths(
            normal_returns.reindex(common).to_numpy(dtype=float),
            stressed_returns.reindex(common).to_numpy(dtype=float),
            policy=policy_from_research_settings(
                settings.research,
                seed=settings.lab.deterministic_seed + int(dna[:8], 16),
                expected_block_length=10,
            ),
        )
        benchmark = frames.get("BTC-EUR")
        if benchmark is None:
            benchmark = next(iter(frames.values()))
        evidence = generate_strategy_evidence_bundle(
            settings.paths.lab_dir / "charts" / "classical_positive",
            strategy_dna=dna,
            timeframe=str(candidate["timeframe"]),
            normal_result=normal,
            stressed_result=stressed,
            stochastic=stochastic,
            benchmark=benchmark,
        )
        results.append(
            {
                "strategy_dna_hash": dna,
                "frozen_candidate_hash": frozen_hash,
                "validated_at": utc_iso(),
                "normal": _metric_subset(normal),
                "stressed": _metric_subset(stressed),
                "stochastic_validation": stochastic,
                "evidence_bundle": evidence,
                "economic_checks": {
                    "normal_positive": (
                        float(normal.metrics.get("net_return") or 0.0) > 0.0
                        and float(normal.metrics.get("profit_factor") or 0.0) > 1.0
                        and float(normal.metrics.get("net_expectancy_r") or 0.0) > 0.0
                    ),
                    "stressed_positive": (
                        float(stressed.metrics.get("net_return") or 0.0) > 0.0
                        and float(stressed.metrics.get("profit_factor") or 0.0) > 1.0
                    ),
                },
                "integrity": {
                    "normal": normal.integrity,
                    "stressed": stressed.integrity,
                },
                "parameters_unchanged": True,
                "strategy_dna_unchanged": True,
                "orders_generated": 0,
                "orders_submitted": 0,
                "validation_hash": stable_hash(
                    {
                        "frozen_candidate_hash": frozen_hash,
                        "normal": _metric_subset(normal),
                        "stressed": _metric_subset(stressed),
                        "stochastic_policy": stochastic["policy_hash"],
                    },
                    length=64,
                ),
            }
        )
    payload = {
        "schema_version": "classical_positive_robustness_v1",
        "generated_at": utc_iso(),
        "candidate_count": len(results),
        "candidates": sorted(
            results,
            key=lambda row: str(row["strategy_dna_hash"]),
        ),
        "stationary_bootstrap_monte_carlo": True,
        "dirichlet_time_concentration_stress": True,
        "strategy_charts": True,
        "academic_failures_are_capital_warnings": True,
        "auto_live_promotion": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(report_path, payload)
    return payload


__all__ = ["validate_classical_positive_candidates"]
