"""Deterministic strategy search, walk-forward validation and acceptance gates."""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from config.settings import ResearchSettings, Settings
from core.contracts import GateResult, ResearchStatus
from research.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from research.strategies import Strategy
from utils.common import append_jsonl, stable_hash


@dataclass(frozen=True)
class SearchTrial:
    trial_id: str
    parameters: dict[str, Any]
    status: str
    score: float | None
    metrics: dict[str, float | int | bool | None]
    error_code: str | None = None


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    strategy_id: str
    best_parameters: dict[str, Any]
    best_score: float
    trials: tuple[SearchTrial, ...]
    resumed_trials: int = 0


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    trade_count: int
    net_expectancy_r: float
    profit_factor: float
    net_pnl_eur: float
    maximum_drawdown: float
    selected_parameters: dict[str, Any] | None = None
    parameter_hash: str | None = None
    optimization_trial_count: int = 0


@dataclass(frozen=True)
class WalkForwardResult:
    mode: str
    folds: tuple[WalkForwardFold, ...]
    positive_folds: int
    fold_profit_concentration: float
    valid: bool


@dataclass(frozen=True)
class CPCVResult:
    group_count: int
    path_count: int
    information_horizon_bars: int
    purged_observations: int
    embargoed_observations: int
    path_returns: tuple[float, ...]
    path_drawdowns: tuple[float, ...]
    path_expectancy: tuple[float, ...]
    path_profit_factor: tuple[float, ...]
    path_consistency: float
    probability_of_backtest_overfitting: float | None
    final_holdout_excluded: bool


@dataclass(frozen=True)
class MultipleTestingResult:
    strategy_count: int
    observation_count: int
    bootstrap_samples: int
    block_size: int
    white_reality_check_pvalue: float
    hansen_spa_pvalue: float
    probability_of_backtest_overfitting: float | None
    pbo_logits: tuple[float, ...]
    deflated_sharpe_probabilities: dict[str, float]
    known_trial_count: int | None = None


@dataclass(frozen=True)
class StabilityResult:
    stable: bool
    tested_neighbors: int
    positive_neighbors: int
    acceptable_score_fraction: float
    neighbor_scores: tuple[float, ...]


@dataclass(frozen=True)
class ResearchOutcome:
    strategy_id: str
    parameters: dict[str, Any]
    optimization: OptimizationResult
    normal_result: BacktestResult
    stressed_result: BacktestResult
    holdout_result: BacktestResult
    walk_forward: WalkForwardResult
    cpcv: CPCVResult
    stability: StabilityResult
    deflated_sharpe_probability: float
    gate: GateResult
    lookahead_safe: bool
    repainting_safe: bool


def robust_score(
    metrics: dict[str, float | int | bool | None],
    *,
    minimum_trades: int,
) -> float:
    def numeric(name: str, default: float) -> float:
        value = metrics.get(name)
        if value is None:
            return default
        selected = float(value)
        return selected if math.isfinite(selected) else default

    trades = int(metrics.get("trade_count") or 0)
    expectancy = numeric("net_expectancy_r", 0.0)
    profit_factor = numeric("profit_factor", 0.0)
    drawdown = numeric("maximum_drawdown", 1.0)
    turnover = numeric("turnover", 0.0)
    concentration = numeric("symbol_profit_concentration", 1.0)
    underwater = numeric("time_under_water", 1.0)
    net_return = numeric("net_return", 0.0)
    win_rate = numeric("win_rate", 0.0)
    finite_pf = min(10.0, profit_factor) if math.isfinite(profit_factor) else 10.0
    score = (
        expectancy * math.sqrt(max(1, trades))
        + math.log1p(max(0.0, finite_pf))
        + net_return
        - 4.0 * drawdown
        - 0.01 * turnover
        - concentration
        - underwater
    )
    if trades < minimum_trades:
        score -= 3.0 * (minimum_trades - trades) / max(1, minimum_trades)
    if expectancy <= 0:
        score -= 2.0
    if trades >= 20 and (win_rate > 0.95 or not math.isfinite(profit_factor)):
        score -= 5.0
    return float(score)


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value
            if not isinstance(value, float) or math.isfinite(value)
            else None
        )
        for key, value in metrics.items()
    }


def _load_checkpoint(path: Path | None) -> dict[str, SearchTrial]:
    if path is None or not path.is_file():
        return {}
    trials: dict[str, SearchTrial] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            trial = SearchTrial(**payload)
        except (json.JSONDecodeError, TypeError):
            continue
        trials[trial.trial_id] = trial
    return trials


def _candidate_id(strategy: Strategy, parameters: dict[str, Any]) -> str:
    return stable_hash(
        {"strategy_id": strategy.strategy_id, "parameters": parameters},
        length=24,
    )


def _evaluate_candidate(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    parameters: dict[str, Any],
    *,
    settings: Settings | None,
    minimum_trades: int,
) -> SearchTrial:
    trial_id = _candidate_id(strategy, parameters)
    try:
        selected = strategy.parameters(parameters)
        search_config = replace(config, monte_carlo_runs=min(100, config.monte_carlo_runs))
        result = BacktestEngine(search_config, settings=settings).run(
            data_by_market,
            strategy,
            parameters=selected,
        )
        score = robust_score(result.metrics, minimum_trades=minimum_trades)
        return SearchTrial(
            trial_id=trial_id,
            parameters=selected,
            status="COMPLETE",
            score=score,
            metrics=_safe_metrics(result.metrics),
        )
    except (ValueError, PermissionError, ArithmeticError) as exc:
        return SearchTrial(
            trial_id=trial_id,
            parameters=dict(parameters),
            status="FAILED",
            score=None,
            metrics={},
            error_code=type(exc).__name__,
        )


def _run_candidates(
    candidates: list[dict[str, Any]],
    *,
    method: str,
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    settings: Settings | None,
    minimum_trades: int,
    checkpoint_path: Path | None,
) -> OptimizationResult:
    resumed = _load_checkpoint(checkpoint_path)
    trials: dict[str, SearchTrial] = dict(resumed)
    for candidate in candidates:
        trial_id = _candidate_id(strategy, strategy.parameters(candidate))
        if trial_id in trials:
            continue
        trial = _evaluate_candidate(
            data_by_market,
            strategy,
            config,
            candidate,
            settings=settings,
            minimum_trades=minimum_trades,
        )
        trials[trial.trial_id] = trial
        if checkpoint_path is not None:
            append_jsonl(checkpoint_path, asdict(trial))
    successful = [
        trial
        for trial in trials.values()
        if trial.status == "COMPLETE" and trial.score is not None
    ]
    if not successful:
        raise RuntimeError("no optimization trial completed successfully")
    best = max(successful, key=lambda trial: (float(trial.score), trial.trial_id))
    return OptimizationResult(
        method=method,
        strategy_id=strategy.strategy_id,
        best_parameters=best.parameters,
        best_score=float(best.score),
        trials=tuple(sorted(trials.values(), key=lambda trial: trial.trial_id)),
        resumed_trials=len(resumed),
    )


def grid_search(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    settings: Settings | None = None,
    minimum_trades: int = 30,
    checkpoint_path: Path | None = None,
    maximum_candidates: int = 5_000,
) -> OptimizationResult:
    names = sorted(strategy.parameter_space)
    combinations = math.prod(len(strategy.parameter_space[name]) for name in names)
    if combinations > maximum_candidates:
        raise ValueError(
            f"grid has {combinations} candidates; maximum is {maximum_candidates}"
        )
    candidates = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(strategy.parameter_space[name] for name in names))
    ]
    if not candidates:
        candidates = [{}]
    return _run_candidates(
        candidates,
        method="grid",
        data_by_market=data_by_market,
        strategy=strategy,
        config=config,
        settings=settings,
        minimum_trades=minimum_trades,
        checkpoint_path=checkpoint_path,
    )


def random_search(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    trials: int = 100,
    seed: int = 42,
    settings: Settings | None = None,
    minimum_trades: int = 30,
    checkpoint_path: Path | None = None,
) -> OptimizationResult:
    if trials < 1:
        raise ValueError("trials must be positive")
    randomizer = random.Random(seed)
    names = sorted(strategy.parameter_space)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    maximum_unique = math.prod(
        len(strategy.parameter_space[name]) for name in names
    ) if names else 1
    while len(candidates) < min(trials, maximum_unique):
        candidate = {
            name: randomizer.choice(strategy.parameter_space[name])
            for name in names
        }
        key = stable_hash(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return _run_candidates(
        candidates,
        method="random",
        data_by_market=data_by_market,
        strategy=strategy,
        config=config,
        settings=settings,
        minimum_trades=minimum_trades,
        checkpoint_path=checkpoint_path,
    )


def coordinate_search(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    rounds: int = 3,
    settings: Settings | None = None,
    minimum_trades: int = 30,
    checkpoint_path: Path | None = None,
) -> OptimizationResult:
    current = strategy.parameters()
    all_candidates: list[dict[str, Any]] = [current]
    for _ in range(rounds):
        changed = False
        for name in sorted(strategy.parameter_space):
            candidates = []
            for value in strategy.parameter_space[name]:
                candidate = dict(current)
                candidate[name] = value
                candidates.append(candidate)
                all_candidates.append(candidate)
            partial = _run_candidates(
                candidates,
                method="coordinate_step",
                data_by_market=data_by_market,
                strategy=strategy,
                config=config,
                settings=settings,
                minimum_trades=minimum_trades,
                checkpoint_path=checkpoint_path,
            )
            if partial.best_parameters != current:
                current = partial.best_parameters
                changed = True
        if not changed:
            break
    unique = {
        _candidate_id(strategy, strategy.parameters(candidate)): candidate
        for candidate in all_candidates
    }
    return _run_candidates(
        list(unique.values()),
        method="coordinate",
        data_by_market=data_by_market,
        strategy=strategy,
        config=config,
        settings=settings,
        minimum_trades=minimum_trades,
        checkpoint_path=checkpoint_path,
    )


def optuna_search(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    trials: int = 100,
    seed: int = 42,
    settings: Settings | None = None,
    minimum_trades: int = 30,
    checkpoint_path: Path | None = None,
) -> OptimizationResult:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    candidates: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        parameters: dict[str, Any] = {}
        typed_specs = getattr(strategy, "optimizer_parameter_specs", {})
        for name, values in strategy.parameter_space.items():
            specification = typed_specs.get(name)
            kind = getattr(getattr(specification, "kind", None), "value", None)
            if kind == "INTEGER":
                parameters[name] = trial.suggest_int(
                    name,
                    int(specification.minimum),
                    int(specification.maximum),
                    step=int(specification.step or 1),
                )
            elif kind in {"HALF_STEP", "DECIMAL"}:
                exact_values = list(specification.values())
                selected = trial.suggest_categorical(
                    name,
                    [str(value) for value in exact_values],
                )
                parameters[name] = specification.validate(selected)
            elif kind in {"CHOICE", "TIMEFRAME", "BOOLEAN"}:
                parameters[name] = trial.suggest_categorical(name, list(values))
            elif specification is not None:
                parameters[name] = trial.suggest_float(
                    name,
                    float(specification.minimum),
                    float(specification.maximum),
                    step=float(specification.step) if specification.step else None,
                    log=specification.optimizer_distribution == "LOG",
                )
            else:
                parameters[name] = trial.suggest_categorical(name, list(values))
        candidates.append(parameters)
        result = _evaluate_candidate(
            data_by_market,
            strategy,
            config,
            parameters,
            settings=settings,
            minimum_trades=minimum_trades,
        )
        if result.status != "COMPLETE" or result.score is None:
            raise optuna.TrialPruned()
        return result.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, catch=(ValueError,))
    unique = {
        _candidate_id(strategy, strategy.parameters(candidate)): candidate
        for candidate in candidates
    }
    return _run_candidates(
        list(unique.values()),
        method="optuna",
        data_by_market=data_by_market,
        strategy=strategy,
        config=config,
        settings=settings,
        minimum_trades=minimum_trades,
        checkpoint_path=checkpoint_path,
    )


def _slice(
    frame: pd.DataFrame,
    start: int | None,
    stop: int | None,
) -> pd.DataFrame:
    result = frame.iloc[start:stop].copy()
    result.attrs.update(frame.attrs)
    return result


def chronological_split(
    data_by_market: dict[str, pd.DataFrame],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train plus validation must leave a final holdout")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge and embargo cannot be negative")
    train: dict[str, pd.DataFrame] = {}
    validation: dict[str, pd.DataFrame] = {}
    holdout: dict[str, pd.DataFrame] = {}
    for market, frame in data_by_market.items():
        count = len(frame)
        first = int(count * train_fraction)
        second = int(count * (train_fraction + validation_fraction))
        train_stop = max(0, first - purge_bars)
        validation_start = min(count, first + embargo_bars)
        validation_stop = max(validation_start, second - purge_bars)
        holdout_start = min(count, second + embargo_bars)
        train[market] = _slice(frame, 0, train_stop)
        validation[market] = _slice(frame, validation_start, validation_stop)
        holdout[market] = _slice(frame, holdout_start, None)
        if min(len(train[market]), len(validation[market]), len(holdout[market])) < 2:
            raise ValueError(f"split leaves insufficient data for {market}")
    return train, validation, holdout


def combinatorial_purged_cross_validation(
    returns: pd.Series,
    *,
    group_count: int = 6,
    test_group_count: int = 2,
    holding_spans: tuple[int, ...] = (),
    label_horizon_bars: int = 0,
    indicator_lookback_bars: int = 0,
    feature_availability_bars: int = 0,
    final_holdout_start: Any | None = None,
) -> CPCVResult:
    """Supplemental CPCV diagnostics; callers must keep final holdout excluded."""

    selected = returns.dropna().astype(float).sort_index()
    holdout_excluded = final_holdout_start is not None
    if final_holdout_start is not None:
        selected = selected.loc[selected.index < final_holdout_start]
    if group_count < 3 or test_group_count < 1 or test_group_count >= group_count:
        raise ValueError("invalid CPCV group configuration")
    if len(selected) < group_count * 2:
        raise ValueError("insufficient observations for CPCV")
    horizons = (
        [label_horizon_bars, indicator_lookback_bars, feature_availability_bars]
        + [int(value) for value in holding_spans]
    )
    if any(value < 0 for value in horizons):
        raise ValueError("CPCV information horizons cannot be negative")
    information_horizon = max(horizons, default=0)
    groups = [
        np.asarray(group, dtype=int)
        for group in np.array_split(np.arange(len(selected)), group_count)
    ]
    paths = list(itertools.combinations(range(group_count), test_group_count))
    path_returns: list[float] = []
    path_drawdowns: list[float] = []
    path_expectancy: list[float] = []
    path_profit_factor: list[float] = []
    purged_total = 0
    embargoed_total = 0
    for path in paths:
        test_positions = np.concatenate([groups[index] for index in path])
        train_mask = np.ones(len(selected), dtype=bool)
        train_mask[test_positions] = False
        purged_mask = np.zeros(len(selected), dtype=bool)
        embargo_mask = np.zeros(len(selected), dtype=bool)
        for group_index in path:
            block = groups[group_index]
            start, stop = int(block[0]), int(block[-1])
            purge_start = max(0, start - information_horizon)
            purge_stop = min(len(selected), stop + 1)
            purged_mask[purge_start:purge_stop] = True
            embargo_start = stop + 1
            embargo_stop = min(len(selected), embargo_start + information_horizon)
            embargo_mask[embargo_start:embargo_stop] = True
        purged_mask[test_positions] = False
        embargo_mask[test_positions] = False
        purged_total += int(np.count_nonzero(train_mask & purged_mask))
        embargoed_total += int(np.count_nonzero(train_mask & embargo_mask))
        train_mask &= ~(purged_mask | embargo_mask)
        if not bool(train_mask.any()):
            raise ValueError("CPCV horizon removes every training observation")
        path_values = selected.iloc[np.sort(test_positions)]
        equity = (1.0 + path_values).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        gains = float(path_values[path_values > 0].sum())
        losses = abs(float(path_values[path_values < 0].sum()))
        path_returns.append(float(equity.iloc[-1] - 1.0))
        path_drawdowns.append(abs(float(drawdown.min())))
        path_expectancy.append(float(path_values.mean()))
        path_profit_factor.append(gains / losses if losses > 0 else math.inf)
    consistency = float(np.mean(np.asarray(path_expectancy) > 0))
    finite_returns = np.asarray(path_returns, dtype=float)
    pbo = (
        float(np.mean(finite_returns < np.median(finite_returns)))
        if len(finite_returns) >= 4
        else None
    )
    return CPCVResult(
        group_count=group_count,
        path_count=len(paths),
        information_horizon_bars=information_horizon,
        purged_observations=purged_total,
        embargoed_observations=embargoed_total,
        path_returns=tuple(path_returns),
        path_drawdowns=tuple(path_drawdowns),
        path_expectancy=tuple(path_expectancy),
        path_profit_factor=tuple(path_profit_factor),
        path_consistency=consistency,
        probability_of_backtest_overfitting=pbo,
        final_holdout_excluded=holdout_excluded,
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: Iterable[float],
    *,
    observed_sharpe: float | None = None,
    effective_sample_size: int | None = None,
    total_trials: int | None = None,
) -> float:
    """Probability that Sharpe exceeds the expected maximum from multiple trials."""

    selected = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(selected) < 3:
        return 0.0
    standard = float(selected.std(ddof=1))
    if standard <= 0 or not math.isfinite(standard):
        return 0.0
    raw_sharpe = float(selected.mean() / standard)
    reported_sharpe = (
        float(observed_sharpe)
        if observed_sharpe is not None and math.isfinite(float(observed_sharpe))
        else raw_sharpe
    )
    scale = abs(reported_sharpe / raw_sharpe) if abs(raw_sharpe) > 1e-12 else 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    raw_trials = np.asarray(
        [
            float(value) / scale
            for value in trial_sharpes
            if math.isfinite(float(value))
        ],
        dtype=float,
    )
    trial_count = max(1, len(raw_trials), int(total_trials or 0))
    trial_std = (
        float(raw_trials.std(ddof=1))
        if len(raw_trials) > 1
        else 0.0
    )
    expected_maximum = 0.0
    if trial_count > 1 and trial_std > 0:
        normal = NormalDist()
        euler_gamma = 0.5772156649015329
        first = normal.inv_cdf(1.0 - 1.0 / trial_count)
        second = normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        expected_maximum = trial_std * (
            (1.0 - euler_gamma) * first + euler_gamma * second
        )
    skewness = float(selected.skew())
    kurtosis = float(selected.kurt()) + 3.0
    sample_size = min(
        len(selected),
        max(3, effective_sample_size or len(selected)),
    )
    variance_adjustment = (
        1.0
        - skewness * raw_sharpe
        + ((kurtosis - 1.0) / 4.0) * raw_sharpe**2
    )
    if not math.isfinite(variance_adjustment) or variance_adjustment <= 0:
        return 0.0
    statistic = (
        (raw_sharpe - expected_maximum)
        * math.sqrt(sample_size - 1)
        / math.sqrt(variance_adjustment)
    )
    return float(min(1.0, max(0.0, NormalDist().cdf(statistic))))


def probability_of_backtest_overfitting(
    strategy_returns: pd.DataFrame,
    *,
    group_count: int = 8,
) -> tuple[float | None, tuple[float, ...]]:
    """Estimate PBO by selecting in-sample winners across symmetric partitions."""

    matrix = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if matrix.shape[1] < 2 or len(matrix) < 8:
        return None, ()
    selected_groups = min(group_count, len(matrix) // 2)
    if selected_groups % 2:
        selected_groups -= 1
    if selected_groups < 4:
        return None, ()
    groups = [
        np.asarray(group, dtype=int)
        for group in np.array_split(np.arange(len(matrix)), selected_groups)
    ]
    logits: list[float] = []
    half = selected_groups // 2
    for train_groups in itertools.combinations(range(selected_groups), half):
        test_groups = tuple(
            index
            for index in range(selected_groups)
            if index not in train_groups
        )
        train_positions = np.concatenate([groups[index] for index in train_groups])
        test_positions = np.concatenate([groups[index] for index in test_groups])
        train = matrix.iloc[np.sort(train_positions)]
        test = matrix.iloc[np.sort(test_positions)]
        train_standard = train.std(ddof=1).replace(0.0, np.nan)
        train_scores = (train.mean() / train_standard).fillna(-math.inf)
        winner = str(train_scores.idxmax())
        test_standard = test.std(ddof=1).replace(0.0, np.nan)
        test_scores = (test.mean() / test_standard).fillna(-math.inf)
        winner_score = float(test_scores[winner])
        below = float((test_scores < winner_score).sum())
        tied = float((test_scores == winner_score).sum())
        relative_rank = (below + 0.5 * tied) / len(test_scores)
        relative_rank = min(1.0 - 1e-9, max(1e-9, relative_rank))
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
    if not logits:
        return None, ()
    return float(np.mean(np.asarray(logits) <= 0.0)), tuple(logits)


def multiple_testing_bootstrap(
    strategy_returns: pd.DataFrame,
    *,
    bootstrap_samples: int = 1_000,
    block_size: int = 5,
    seed: int = 42,
    known_trial_count: int | None = None,
) -> MultipleTestingResult:
    """Run White Reality Check, Hansen SPA, PBO and per-strategy DSR."""

    if bootstrap_samples < 100:
        raise ValueError("multiple-testing bootstrap requires at least 100 samples")
    if block_size < 1:
        raise ValueError("bootstrap block size must be positive")
    matrix = (
        strategy_returns.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
    )
    if matrix.empty or matrix.shape[1] < 1:
        raise ValueError("multiple-testing matrix is empty")
    values = matrix.to_numpy(dtype=float)
    observations, strategies = values.shape
    if known_trial_count is not None and known_trial_count < strategies:
        raise ValueError("known trial count cannot be smaller than return matrix")
    means = values.mean(axis=0)
    standard = values.std(axis=0, ddof=1)
    standard_error = np.where(
        standard > 0,
        standard / math.sqrt(observations),
        math.inf,
    )
    observed_white = math.sqrt(observations) * max(0.0, float(means.max()))
    observed_spa = max(
        0.0,
        float(np.max(np.divide(means, standard_error))),
    )
    centered = values - means
    randomizer = np.random.default_rng(seed)
    white_exceedances = 0
    spa_exceedances = 0
    for _ in range(bootstrap_samples):
        indices: list[int] = []
        while len(indices) < observations:
            start = int(randomizer.integers(0, observations))
            indices.extend(
                (start + offset) % observations
                for offset in range(block_size)
            )
        sample = centered[np.asarray(indices[:observations], dtype=int)]
        sample_means = sample.mean(axis=0)
        white_statistic = math.sqrt(observations) * max(
            0.0,
            float(sample_means.max()),
        )
        spa_statistic = max(
            0.0,
            float(np.max(np.divide(sample_means, standard_error))),
        )
        white_exceedances += white_statistic >= observed_white
        spa_exceedances += spa_statistic >= observed_spa
    pbo, logits = probability_of_backtest_overfitting(matrix)
    trial_sharpes = np.divide(
        means,
        standard,
        out=np.zeros_like(means),
        where=standard > 0,
    )
    dsr = {
        str(column): deflated_sharpe_ratio(
            matrix[column],
            trial_sharpes,
            observed_sharpe=float(trial_sharpes[index]),
            total_trials=known_trial_count,
        )
        for index, column in enumerate(matrix.columns)
    }
    denominator = bootstrap_samples + 1
    return MultipleTestingResult(
        strategy_count=strategies,
        observation_count=observations,
        bootstrap_samples=bootstrap_samples,
        block_size=block_size,
        white_reality_check_pvalue=(white_exceedances + 1) / denominator,
        hansen_spa_pvalue=(spa_exceedances + 1) / denominator,
        probability_of_backtest_overfitting=pbo,
        pbo_logits=logits,
        deflated_sharpe_probabilities=dsr,
        known_trial_count=known_trial_count or strategies,
    )


def walk_forward_validate(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    parameters: dict[str, Any],
    config: BacktestConfig,
    *,
    folds: int = 6,
    mode: Literal["anchored", "rolling"] = "anchored",
    purge_bars: int = 0,
    embargo_bars: int = 0,
    settings: Settings | None = None,
) -> WalkForwardResult:
    if folds < 2:
        raise ValueError("walk-forward requires at least two folds")
    minimum_count = min(len(frame) for frame in data_by_market.values())
    initial_train = max(2, minimum_count // 3)
    remaining = minimum_count - initial_train
    test_size = remaining // folds
    if test_size < 2:
        raise ValueError("insufficient rows for requested walk-forward folds")
    fold_results: list[WalkForwardFold] = []
    for fold in range(folds):
        train_end = initial_train + fold * test_size
        test_start = train_end + embargo_bars
        test_end = (
            minimum_count
            if fold == folds - 1
            else min(minimum_count, test_start + test_size)
        )
        train_start = 0 if mode == "anchored" else max(0, train_end - initial_train)
        effective_train_end = max(train_start + 1, train_end - purge_bars)
        train_slice = {
            market: _slice(frame, train_start, effective_train_end)
            for market, frame in data_by_market.items()
        }
        test_slice = {
            market: _slice(frame, test_start, test_end)
            for market, frame in data_by_market.items()
        }
        # The train slice is intentionally materialized for auditability even
        # when parameters are fixed for validation.
        if any(len(frame) < 2 for frame in train_slice.values()) or any(
            len(frame) < 2 for frame in test_slice.values()
        ):
            continue
        validation_config = replace(
            config,
            monte_carlo_runs=min(100, config.monte_carlo_runs),
        )
        result = BacktestEngine(validation_config, settings=settings).run(
            test_slice,
            strategy,
            parameters=parameters,
        )
        pnl = sum(float(trade.net_pnl_eur) for trade in result.trades)
        first_market = sorted(data_by_market)[0]
        fold_results.append(
            WalkForwardFold(
                fold=fold + 1,
                train_start=str(train_slice[first_market].index[0]),
                train_end=str(train_slice[first_market].index[-1]),
                test_start=str(test_slice[first_market].index[0]),
                test_end=str(test_slice[first_market].index[-1]),
                trade_count=int(result.metrics["trade_count"]),
                net_expectancy_r=float(result.metrics["net_expectancy_r"]),
                profit_factor=float(result.metrics["profit_factor"]),
                net_pnl_eur=pnl,
                maximum_drawdown=float(result.metrics["maximum_drawdown"]),
            )
        )
    positive = sum(fold.net_expectancy_r > 0 for fold in fold_results)
    positive_profit = sum(max(0.0, fold.net_pnl_eur) for fold in fold_results)
    concentration = (
        max((max(0.0, fold.net_pnl_eur) for fold in fold_results), default=0.0)
        / positive_profit
        if positive_profit > 0
        else 1.0
    )
    return WalkForwardResult(
        mode=f"{mode}_fixed_parameters",
        folds=tuple(fold_results),
        positive_folds=positive,
        fold_profit_concentration=concentration,
        valid=len(fold_results) == folds,
    )


def walk_forward_optimize(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    config: BacktestConfig,
    *,
    folds: int = 6,
    mode: Literal["anchored", "rolling"] = "anchored",
    search_method: Literal["grid", "random", "coordinate", "optuna"] = "random",
    search_trials: int = 20,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    settings: Settings | None = None,
    minimum_trades: int = 30,
    checkpoint_path: Path | None = None,
) -> WalkForwardResult:
    """Re-optimize on each train fold and freeze parameters on its next fold."""

    if folds < 2:
        raise ValueError("walk-forward optimization requires at least two folds")
    minimum_count = min(len(frame) for frame in data_by_market.values())
    initial_train = max(2, minimum_count // 3)
    test_size = (minimum_count - initial_train) // folds
    if test_size < 2:
        raise ValueError("insufficient rows for requested walk-forward folds")
    fold_results: list[WalkForwardFold] = []
    for fold in range(folds):
        train_end = initial_train + fold * test_size
        validation_start = train_end + embargo_bars
        validation_end = (
            minimum_count
            if fold == folds - 1
            else min(minimum_count, validation_start + test_size)
        )
        train_start = 0 if mode == "anchored" else max(0, train_end - initial_train)
        effective_train_end = max(train_start + 1, train_end - purge_bars)
        train_slice = {
            market: _slice(frame, train_start, effective_train_end)
            for market, frame in data_by_market.items()
        }
        validation_slice = {
            market: _slice(frame, validation_start, validation_end)
            for market, frame in data_by_market.items()
        }
        if any(len(frame) < 2 for frame in train_slice.values()) or any(
            len(frame) < 2 for frame in validation_slice.values()
        ):
            continue
        fold_checkpoint = (
            checkpoint_path.with_name(
                f"{checkpoint_path.stem}.wfo-fold-{fold + 1}{checkpoint_path.suffix}"
            )
            if checkpoint_path is not None
            else None
        )
        if search_method == "grid":
            optimized = grid_search(
                train_slice,
                strategy,
                config,
                settings=settings,
                minimum_trades=minimum_trades,
                checkpoint_path=fold_checkpoint,
            )
        elif search_method == "coordinate":
            optimized = coordinate_search(
                train_slice,
                strategy,
                config,
                settings=settings,
                minimum_trades=minimum_trades,
                checkpoint_path=fold_checkpoint,
            )
        elif search_method == "optuna":
            optimized = optuna_search(
                train_slice,
                strategy,
                config,
                trials=search_trials,
                seed=(settings.app.random_seed if settings else 42) + fold,
                settings=settings,
                minimum_trades=minimum_trades,
                checkpoint_path=fold_checkpoint,
            )
        else:
            optimized = random_search(
                train_slice,
                strategy,
                config,
                trials=search_trials,
                seed=(settings.app.random_seed if settings else 42) + fold,
                settings=settings,
                minimum_trades=minimum_trades,
                checkpoint_path=fold_checkpoint,
            )
        frozen = dict(optimized.best_parameters)
        result = BacktestEngine(
            replace(config, monte_carlo_runs=min(100, config.monte_carlo_runs)),
            settings=settings,
        ).run(validation_slice, strategy, parameters=frozen)
        pnl = sum(float(trade.net_pnl_eur) for trade in result.trades)
        first_market = sorted(data_by_market)[0]
        fold_results.append(
            WalkForwardFold(
                fold=fold + 1,
                train_start=str(train_slice[first_market].index[0]),
                train_end=str(train_slice[first_market].index[-1]),
                test_start=str(validation_slice[first_market].index[0]),
                test_end=str(validation_slice[first_market].index[-1]),
                trade_count=int(result.metrics["trade_count"]),
                net_expectancy_r=float(result.metrics["net_expectancy_r"]),
                profit_factor=float(result.metrics["profit_factor"]),
                net_pnl_eur=pnl,
                maximum_drawdown=float(result.metrics["maximum_drawdown"]),
                selected_parameters=frozen,
                parameter_hash=stable_hash(frozen),
                optimization_trial_count=len(optimized.trials),
            )
        )
    positive = sum(fold.net_expectancy_r > 0 for fold in fold_results)
    positive_profit = sum(max(0.0, fold.net_pnl_eur) for fold in fold_results)
    concentration = (
        max((max(0.0, fold.net_pnl_eur) for fold in fold_results), default=0.0)
        / positive_profit
        if positive_profit > 0
        else 1.0
    )
    return WalkForwardResult(
        mode=f"{mode}_optimized",
        folds=tuple(fold_results),
        positive_folds=positive,
        fold_profit_concentration=concentration,
        valid=len(fold_results) == folds,
    )


def parameter_stability(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    parameters: dict[str, Any],
    config: BacktestConfig,
    *,
    settings: Settings | None = None,
    minimum_trades: int = 30,
) -> StabilityResult:
    base_trial = _evaluate_candidate(
        data_by_market,
        strategy,
        config,
        parameters,
        settings=settings,
        minimum_trades=minimum_trades,
    )
    if base_trial.score is None:
        return StabilityResult(False, 0, 0, 0.0, ())
    neighbors: list[dict[str, Any]] = []
    for name, values in strategy.parameter_space.items():
        if name not in parameters:
            continue
        ordered = list(values)
        try:
            position = ordered.index(parameters[name])
        except ValueError:
            continue
        for neighbor_index in (position - 1, position + 1):
            if 0 <= neighbor_index < len(ordered):
                neighbor = dict(parameters)
                neighbor[name] = ordered[neighbor_index]
                neighbors.append(neighbor)
    scores: list[float] = []
    positive = 0
    acceptable = 0
    threshold = base_trial.score - max(0.5, abs(base_trial.score) * 0.25)
    for neighbor in neighbors:
        trial = _evaluate_candidate(
            data_by_market,
            strategy,
            config,
            neighbor,
            settings=settings,
            minimum_trades=minimum_trades,
        )
        if trial.score is None:
            continue
        scores.append(trial.score)
        positive += float(trial.metrics.get("net_expectancy_r") or 0.0) > 0
        acceptable += trial.score >= threshold
    fraction = acceptable / len(scores) if scores else 0.0
    return StabilityResult(
        stable=bool(scores) and fraction >= 0.60 and positive >= math.ceil(len(scores) / 2),
        tested_neighbors=len(scores),
        positive_neighbors=positive,
        acceptable_score_fraction=fraction,
        neighbor_scores=tuple(scores),
    )


def strategy_lookahead_test(
    features: pd.DataFrame,
    strategy: Strategy,
    parameters: dict[str, Any],
    *,
    cutoff_fraction: float = 0.80,
) -> bool:
    cutoff = max(2, int(len(features) * cutoff_fraction))
    baseline = strategy.generate(features, parameters)
    perturbed = features.copy()
    numeric = perturbed.select_dtypes(include=[np.number]).columns
    perturbed.loc[perturbed.index[cutoff:], numeric] = (
        perturbed.loc[perturbed.index[cutoff:], numeric] * 7.0 + 123.0
    )
    candidate = strategy.generate(perturbed, parameters)
    return bool(
        baseline.entry.iloc[:cutoff].equals(candidate.entry.iloc[:cutoff])
        and baseline.exit.iloc[:cutoff].equals(candidate.exit.iloc[:cutoff])
    )


def strategy_repainting_test(
    features: pd.DataFrame,
    strategy: Strategy,
    parameters: dict[str, Any],
    *,
    prefix_fraction: float = 0.80,
) -> bool:
    cutoff = max(2, int(len(features) * prefix_fraction))
    full = strategy.generate(features, parameters)
    prefix = features.iloc[:cutoff].copy()
    prefix.attrs.update(features.attrs)
    partial = strategy.generate(prefix, parameters)
    return bool(
        full.entry.iloc[:cutoff].equals(partial.entry)
        and full.exit.iloc[:cutoff].equals(partial.exit)
    )


def acceptance_gate(
    *,
    normal: BacktestResult,
    stressed: BacktestResult,
    holdout: BacktestResult,
    walk_forward: WalkForwardResult,
    cpcv: CPCVResult | None = None,
    stability: StabilityResult,
    research: ResearchSettings,
    eligibility_valid: bool,
    lookahead_safe: bool,
    repainting_safe: bool,
    deflated_sharpe_probability: float | None = None,
    promote_to_paper: bool = False,
) -> GateResult:
    checks: list[tuple[bool, ResearchStatus, str]] = [
        (
            bool(normal.integrity.get("valid_data")),
            ResearchStatus.REJECTED_DATA,
            "INVALID_DATA",
        ),
        (
            bool(normal.integrity.get("valid_intelligence_timing")),
            ResearchStatus.REJECTED_INTELLIGENCE_TIMING,
            "INVALID_INTELLIGENCE_TIMING",
        ),
        (lookahead_safe, ResearchStatus.REJECTED_LOOKAHEAD, "LOOKAHEAD_DETECTED"),
        (repainting_safe, ResearchStatus.REJECTED_REPAINTING, "REPAINTING_DETECTED"),
        (eligibility_valid, ResearchStatus.REJECTED_ELIGIBILITY, "MARKET_NOT_ALLOWED"),
        (
            int(normal.metrics["trade_count"]) >= research.minimum_trades,
            ResearchStatus.REJECTED_INSUFFICIENT_TRADES,
            "INSUFFICIENT_TRADES",
        ),
        (
            float(normal.metrics["effective_sample_size"])
            >= research.minimum_effective_sample_size,
            ResearchStatus.REJECTED_EFFECTIVE_SAMPLE,
            "INSUFFICIENT_EFFECTIVE_SAMPLE",
        ),
        (
            float(normal.metrics["net_expectancy_r"])
            > research.minimum_net_expectancy_r
            and float(holdout.metrics["net_expectancy_r"])
            > research.minimum_net_expectancy_r,
            ResearchStatus.REJECTED_EXPECTANCY,
            "NON_POSITIVE_EXPECTANCY",
        ),
        (
            float(normal.metrics["profit_factor"])
            >= research.minimum_profit_factor,
            ResearchStatus.REJECTED_PROFIT_FACTOR,
            "PROFIT_FACTOR_BELOW_GATE",
        ),
        (
            float(stressed.metrics["profit_factor"])
            >= research.minimum_stressed_profit_factor
            and float(stressed.metrics["net_expectancy_r"]) >= 0,
            ResearchStatus.REJECTED_STRESSED_COSTS,
            "STRESSED_COST_FAILURE",
        ),
        (
            walk_forward.valid
            and walk_forward.positive_folds >= research.minimum_positive_folds,
            ResearchStatus.REJECTED_WALK_FORWARD,
            "WALK_FORWARD_FAILURE",
        ),
        (
            cpcv is not None
            and cpcv.final_holdout_excluded
            and cpcv.path_consistency
            >= research.minimum_cpcv_path_consistency,
            ResearchStatus.REJECTED_WALK_FORWARD,
            "CPCV_PATH_CONSISTENCY_FAILURE",
        ),
        (
            deflated_sharpe_probability is not None
            and deflated_sharpe_probability
            >= research.minimum_deflated_sharpe_probability,
            ResearchStatus.REJECTED_EXPECTANCY,
            "DEFLATED_SHARPE_FAILURE",
        ),
        (
            float(normal.metrics["maximum_drawdown"]) <= research.maximum_drawdown,
            ResearchStatus.REJECTED_DRAWDOWN,
            "DRAWDOWN_ABOVE_GATE",
        ),
        (
            float(normal.metrics["probability_of_loss"])
            <= research.maximum_monte_carlo_probability_of_loss
            and float(normal.metrics["probability_of_30pct_drawdown"])
            <= research.maximum_probability_of_30pct_drawdown,
            ResearchStatus.REJECTED_RISK_OF_RUIN,
            "MONTE_CARLO_RISK_ABOVE_GATE",
        ),
        (
            stability.stable or not research.parameter_stability_required,
            ResearchStatus.REJECTED_PARAMETER_INSTABILITY,
            "PARAMETER_INSTABILITY",
        ),
        (
            float(normal.metrics["symbol_profit_concentration"])
            <= research.maximum_symbol_profit_concentration
            and walk_forward.fold_profit_concentration
            <= research.maximum_fold_profit_concentration,
            ResearchStatus.REJECTED_CONCENTRATION,
            "PROFIT_CONCENTRATION",
        ),
    ]
    for passed, status, reason in checks:
        if not passed:
            return GateResult(
                status=status,
                passed=False,
                reasons=(reason,),
                metrics={
                    "trade_count": int(normal.metrics["trade_count"]),
                    "net_expectancy_r": float(normal.metrics["net_expectancy_r"]),
                    "holdout_net_expectancy_r": float(
                        holdout.metrics["net_expectancy_r"]
                    ),
                    "profit_factor": float(normal.metrics["profit_factor"]),
                    "stressed_profit_factor": float(
                        stressed.metrics["profit_factor"]
                    ),
                    "maximum_drawdown": float(normal.metrics["maximum_drawdown"]),
                    "cpcv_path_consistency": (
                        cpcv.path_consistency if cpcv is not None else None
                    ),
                    "deflated_sharpe_probability": (
                        deflated_sharpe_probability
                    ),
                },
            )
    return GateResult(
        status=(
            ResearchStatus.PAPER_CANDIDATE
            if promote_to_paper
            else ResearchStatus.RESEARCH_PASS
        ),
        passed=True,
        reasons=("ALL_RESEARCH_GATES_PASSED",),
        metrics={
            "trade_count": int(normal.metrics["trade_count"]),
            "net_expectancy_r": float(normal.metrics["net_expectancy_r"]),
            "holdout_net_expectancy_r": float(holdout.metrics["net_expectancy_r"]),
            "profit_factor": float(normal.metrics["profit_factor"]),
            "stressed_profit_factor": float(stressed.metrics["profit_factor"]),
            "maximum_drawdown": float(normal.metrics["maximum_drawdown"]),
            "cpcv_path_consistency": (
                cpcv.path_consistency if cpcv is not None else None
            ),
            "deflated_sharpe_probability": deflated_sharpe_probability,
        },
    )


def run_research(
    data_by_market: dict[str, pd.DataFrame],
    strategy: Strategy,
    settings: Settings,
    *,
    capital_eur: float = 2_000.0,
    search_method: Literal["grid", "random", "coordinate", "optuna"] = "coordinate",
    search_trials: int = 50,
    purge_bars: int = 1,
    embargo_bars: int = 1,
    checkpoint_path: Path | None = None,
    promote_to_paper: bool = False,
    allow_review_required_research_only: bool = False,
) -> ResearchOutcome:
    train, validation, holdout = chronological_split(
        data_by_market,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    normal_config = BacktestConfig.from_settings(
        settings,
        initial_cash_eur=capital_eur,
        allow_review_required_research_only=allow_review_required_research_only,
    )
    search_functions = {
        "grid": lambda: grid_search(
            train,
            strategy,
            normal_config,
            settings=settings,
            minimum_trades=settings.research.minimum_trades,
            checkpoint_path=checkpoint_path,
        ),
        "random": lambda: random_search(
            train,
            strategy,
            normal_config,
            trials=search_trials,
            seed=settings.app.random_seed,
            settings=settings,
            minimum_trades=settings.research.minimum_trades,
            checkpoint_path=checkpoint_path,
        ),
        "coordinate": lambda: coordinate_search(
            train,
            strategy,
            normal_config,
            settings=settings,
            minimum_trades=settings.research.minimum_trades,
            checkpoint_path=checkpoint_path,
        ),
        "optuna": lambda: optuna_search(
            train,
            strategy,
            normal_config,
            trials=search_trials,
            seed=settings.app.random_seed,
            settings=settings,
            minimum_trades=settings.research.minimum_trades,
            checkpoint_path=checkpoint_path,
        ),
    }
    optimization = search_functions[search_method]()
    parameters = optimization.best_parameters
    pre_holdout = {
        market: _slice(frame, 0, int(len(frame) * 0.80) - purge_bars)
        for market, frame in data_by_market.items()
    }
    normal_result = BacktestEngine(normal_config, settings=settings).run(
        pre_holdout,
        strategy,
        parameters=parameters,
    )
    stressed_config = replace(
        normal_config,
        costs=replace(
            normal_config.costs,
            multiplier=settings.costs.stressed_cost_multiplier,
        ),
    )
    stressed_result = BacktestEngine(stressed_config, settings=settings).run(
        pre_holdout,
        strategy,
        parameters=parameters,
    )
    holdout_result = BacktestEngine(normal_config, settings=settings).run(
        holdout,
        strategy,
        parameters=parameters,
    )
    walk_forward = walk_forward_optimize(
        pre_holdout,
        strategy,
        normal_config,
        folds=settings.research.walk_forward_folds,
        mode="anchored",
        search_method=search_method,
        search_trials=search_trials,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        settings=settings,
        minimum_trades=settings.research.minimum_trades,
        checkpoint_path=checkpoint_path,
    )
    stability = parameter_stability(
        validation,
        strategy,
        parameters,
        normal_config,
        settings=settings,
        minimum_trades=settings.research.minimum_trades,
    )
    lookahead_safe = all(
        strategy_lookahead_test(frame, strategy, parameters)
        for frame in data_by_market.values()
    )
    repainting_safe = all(
        strategy_repainting_test(frame, strategy, parameters)
        for frame in data_by_market.values()
    )
    normal_returns = (
        normal_result.equity_curve["equity"]
        .astype(float)
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    holdout_start = min(frame.index[0] for frame in holdout.values())
    cpcv = combinatorial_purged_cross_validation(
        normal_returns,
        group_count=6,
        test_group_count=2,
        label_horizon_bars=purge_bars,
        indicator_lookback_bars=purge_bars,
        feature_availability_bars=embargo_bars,
        final_holdout_start=holdout_start,
    )
    trial_sharpes = [
        float(trial.metrics["sharpe"])
        for trial in optimization.trials
        if trial.metrics.get("sharpe") is not None
        and math.isfinite(float(trial.metrics["sharpe"]))
    ]
    deflated_sharpe_probability = deflated_sharpe_ratio(
        normal_returns,
        trial_sharpes,
        observed_sharpe=float(normal_result.metrics.get("sharpe") or 0.0),
        effective_sample_size=int(
            normal_result.metrics.get("effective_sample_size") or len(normal_returns)
        ),
    )
    eligibility_valid = all(
        settings.shariah.eligibility(market).status.value == "ALLOWED"
        for market in data_by_market
    )
    gate = acceptance_gate(
        normal=normal_result,
        stressed=stressed_result,
        holdout=holdout_result,
        walk_forward=walk_forward,
        cpcv=cpcv,
        stability=stability,
        research=settings.research,
        eligibility_valid=eligibility_valid,
        lookahead_safe=lookahead_safe,
        repainting_safe=repainting_safe,
        deflated_sharpe_probability=deflated_sharpe_probability,
        promote_to_paper=promote_to_paper,
    )
    return ResearchOutcome(
        strategy_id=strategy.strategy_id,
        parameters=parameters,
        optimization=optimization,
        normal_result=normal_result,
        stressed_result=stressed_result,
        holdout_result=holdout_result,
        walk_forward=walk_forward,
        cpcv=cpcv,
        stability=stability,
        deflated_sharpe_probability=deflated_sharpe_probability,
        gate=gate,
        lookahead_safe=lookahead_safe,
        repainting_safe=repainting_safe,
    )


__all__ = [
    "OptimizationResult",
    "ResearchOutcome",
    "CPCVResult",
    "MultipleTestingResult",
    "SearchTrial",
    "StabilityResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "acceptance_gate",
    "chronological_split",
    "combinatorial_purged_cross_validation",
    "coordinate_search",
    "deflated_sharpe_ratio",
    "grid_search",
    "optuna_search",
    "parameter_stability",
    "multiple_testing_bootstrap",
    "probability_of_backtest_overfitting",
    "random_search",
    "robust_score",
    "run_research",
    "strategy_lookahead_test",
    "strategy_repainting_test",
    "walk_forward_validate",
    "walk_forward_optimize",
]
