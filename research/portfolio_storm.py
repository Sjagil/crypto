"""Deterministic multi-objective portfolio-DNA research storm.

The storm is deliberately separate from the frozen rotation lead.  Every DNA
is declared before evaluation, selection uses development observations only,
and validation/confirmation metrics are exposed only for frozen Pareto
survivors.  The simulator is long-only, next-open and hard-capped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from research.optimization import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from utils.common import stable_hash

STORM_ENGINE_VERSION = "1.0.0"
STORM_TRIAL_COUNT = 5_000
STORM_SEED = 20260725

RegimeMapping = Literal["hard_gate", "linear", "piecewise"]
StormWeighting = Literal["equal", "inverse_volatility"]


@dataclass(frozen=True, slots=True)
class PortfolioStormDNA:
    momentum_fast: int
    momentum_slow: int
    asset_ema_period: int
    btc_ema_period: int
    top_n: int
    rebalance_days: int
    regime_mapping: RegimeMapping
    weighting: StormWeighting
    maximum_total_exposure: float
    rebalance_buffer: float
    require_btc_uptrend: bool
    maximum_position_exposure: float = 0.20
    minimum_cash: float = 0.60

    def __post_init__(self) -> None:
        if self.momentum_fast >= self.momentum_slow:
            raise ValueError("fast momentum must be shorter than slow momentum")
        if self.top_n not in {1, 2}:
            raise ValueError("strict storm supports top-1 or top-2")
        if self.regime_mapping not in {"hard_gate", "linear", "piecewise"}:
            raise ValueError("unsupported regime mapping")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError("unsupported weighting")
        if self.maximum_total_exposure > 0.40 + 1e-12:
            raise ValueError("storm total exposure exceeds strict limit")
        if self.maximum_position_exposure > 0.20 + 1e-12:
            raise ValueError("storm position exposure exceeds strict limit")
        if self.minimum_cash < 0.60 - 1e-12:
            raise ValueError("storm minimum cash violates strict reserve")
        if self.maximum_total_exposure > 1.0 - self.minimum_cash + 1e-12:
            raise ValueError("storm exposure violates minimum cash")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "STRICT_MULTI_OBJECTIVE_PORTFOLIO_STORM",
                "engine_version": STORM_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def preregistered_storm_dna(
    *,
    trial_count: int = STORM_TRIAL_COUNT,
    seed: int = STORM_SEED,
) -> tuple[PortfolioStormDNA, ...]:
    """Draw a deterministic unique subset from the declared finite grid."""

    if trial_count < 2:
        raise ValueError("storm requires at least two trials")
    rows = [
        PortfolioStormDNA(
            momentum_fast=fast,
            momentum_slow=slow,
            asset_ema_period=asset_ema,
            btc_ema_period=btc_ema,
            top_n=top_n,
            rebalance_days=rebalance,
            regime_mapping=regime,
            weighting=weighting,
            maximum_total_exposure=exposure,
            rebalance_buffer=buffer,
            require_btc_uptrend=btc_gate,
        )
        for (
            fast,
            slow,
            asset_ema,
            btc_ema,
            top_n,
            rebalance,
            regime,
            weighting,
            exposure,
            buffer,
            btc_gate,
        ) in product(
            (5, 10, 20, 30),
            (60, 90, 120, 180),
            (20, 50, 100, 200),
            (100, 200),
            (1, 2),
            (3, 7, 14),
            ("hard_gate", "linear", "piecewise"),
            ("equal", "inverse_volatility"),
            (0.20, 0.30, 0.40),
            (0.0, 0.025, 0.05),
            (False, True),
        )
        if fast < slow
    ]
    if trial_count > len(rows):
        raise ValueError("requested storm exceeds declared unique grid")
    generator = np.random.default_rng(seed)
    selected = generator.choice(
        len(rows),
        size=trial_count,
        replace=False,
    )
    result = tuple(rows[int(index)] for index in selected)
    if len({row.dna_hash for row in result}) != len(result):
        raise RuntimeError("storm preregistration contains duplicate DNA")
    return result


def storm_plan(
    *,
    trial_count: int = STORM_TRIAL_COUNT,
    seed: int = STORM_SEED,
) -> dict[str, Any]:
    rows = preregistered_storm_dna(trial_count=trial_count, seed=seed)
    hashes = [row.dna_hash for row in rows]
    return {
        "schema_version": "portfolio_storm_plan_v1",
        "status": "PREREGISTERED_NOT_RUN",
        "campaign": "PORTFOLIO_STORM_V1",
        "engine_version": STORM_ENGINE_VERSION,
        "seed": seed,
        "trial_count": len(rows),
        "strategy_dna_hashes": hashes,
        "strategy_dna": [asdict(row) for row in rows],
        "search_space_hash": stable_hash(hashes, length=64),
        "selection_basis": "DEVELOPMENT_ONLY",
        "selection_integrity": {
            "development_returns_only": True,
            "development_turnover_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "objectives": {
            "maximize": ["portfolio_period_profit_factor"],
            "minimize": ["ulcer_index", "turnover_efficiency"],
        },
        "maximum_total_exposure": 0.40,
        "maximum_position_exposure": 0.20,
        "minimum_cash": 0.60,
        "next_open_execution": True,
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _panel(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    if set(frames) != set(markets):
        raise ValueError("storm requires the exact strict four-asset universe")
    opens = pd.concat(
        {market: frames[market]["open"].astype(float) for market in markets},
        axis=1,
        join="inner",
    )
    closes = pd.concat(
        {market: frames[market]["close"].astype(float) for market in markets},
        axis=1,
        join="inner",
    )
    valid = (
        opens.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & closes.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    )
    opens = opens.loc[valid]
    closes = closes.loc[valid]
    if len(opens) < 500:
        raise ValueError("storm requires at least 500 common daily observations")
    if not opens.index.is_monotonic_increasing or opens.index.has_duplicates:
        raise ValueError("storm panel timestamps must be unique and sorted")
    return opens, closes


def _features(
    closes: pd.DataFrame,
    dna: tuple[PortfolioStormDNA, ...],
) -> dict[str, dict[int, np.ndarray] | np.ndarray]:
    horizons = sorted(
        {
            value
            for row in dna
            for value in (row.momentum_fast, row.momentum_slow)
        }
    )
    asset_emas = sorted({row.asset_ema_period for row in dna})
    btc_emas = sorted({row.btc_ema_period for row in dna})
    momentum = {
        value: (closes / closes.shift(value) - 1.0).to_numpy(dtype=float)
        for value in horizons
    }
    ema = {
        value: closes.ewm(
            span=value,
            adjust=False,
            min_periods=value,
        ).mean().to_numpy(dtype=float)
        for value in sorted(set(asset_emas) | set(btc_emas))
    }
    returns = closes.pct_change(fill_method=None)
    volatility = returns.rolling(20, min_periods=20).std(ddof=0).to_numpy(
        dtype=float
    )
    return {"momentum": momentum, "ema": ema, "volatility": volatility}


def _capped_weights(
    raw: np.ndarray,
    *,
    budget: float,
    position_cap: float,
) -> np.ndarray:
    weights = np.zeros_like(raw)
    active = raw > 0
    remaining = budget
    while active.any() and remaining > 1e-12:
        share = raw[active] / raw[active].sum() * remaining
        indices = np.flatnonzero(active)
        capacity = position_cap - weights[indices]
        allocation = np.minimum(share, capacity)
        weights[indices] += allocation
        remaining = budget - float(weights.sum())
        active[indices[capacity <= share + 1e-15]] = False
        if bool((capacity > share + 1e-15).all()):
            break
    return weights


def _simulate(
    opens: np.ndarray,
    closes: np.ndarray,
    feature_cache: dict[str, Any],
    dna: PortfolioStormDNA,
    *,
    one_way_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    count, assets = closes.shape
    returns = np.zeros(count - 1, dtype=np.float64)
    turnover_series = np.zeros(count - 1, dtype=np.float64)
    current = np.zeros(assets, dtype=float)
    fast = feature_cache["momentum"][dna.momentum_fast]
    slow = feature_cache["momentum"][dna.momentum_slow]
    asset_ema = feature_cache["ema"][dna.asset_ema_period]
    btc_ema = feature_cache["ema"][dna.btc_ema_period][:, 0]
    volatility = feature_cache["volatility"]
    warmup = max(
        dna.momentum_slow,
        dna.asset_ema_period,
        dna.btc_ema_period,
        20,
    )
    for execution in range(warmup + 1, count - 1):
        decision = execution - 1
        execution_turnover = 0.0
        if (decision - warmup) % dna.rebalance_days == 0:
            score = 0.5 * fast[decision] + 0.5 * slow[decision]
            eligible = (
                np.isfinite(score)
                & np.isfinite(asset_ema[decision])
                & np.isfinite(volatility[decision])
                & (score > 0)
                & (closes[decision] > asset_ema[decision])
            )
            btc_up = bool(
                math.isfinite(float(btc_ema[decision]))
                and closes[decision, 0] > btc_ema[decision]
            )
            hard_block = dna.require_btc_uptrend and not btc_up
            target = np.zeros(assets, dtype=float)
            if eligible.any() and not hard_block:
                breadth = float(eligible.mean())
                btc_vol = float(volatility[decision, 0])
                normalized = (
                    math.log(closes[decision, 0] / btc_ema[decision])
                    / max(1e-12, btc_vol * math.sqrt(dna.btc_ema_period))
                    if btc_ema[decision] > 0 and btc_vol > 0
                    else 0.0
                )
                trend_score = float(
                    np.clip(0.5 + 0.5 * normalized, 0.10, 1.0)
                )
                regime_score = float(
                    np.clip(0.5 * trend_score + 0.5 * breadth, 0.0, 1.0)
                )
                if dna.regime_mapping == "hard_gate":
                    budget = dna.maximum_total_exposure
                elif dna.regime_mapping == "piecewise":
                    multiplier = (0.0, 0.25, 0.50, 0.75, 1.0)[
                        int(
                            np.searchsorted(
                                (0.25, 0.40, 0.55, 0.70),
                                regime_score,
                                side="right",
                            )
                        )
                    ]
                    budget = dna.maximum_total_exposure * multiplier
                else:
                    budget = dna.maximum_total_exposure * regime_score
                ranked = np.argsort(np.where(eligible, score, -np.inf))[::-1]
                selected = ranked[np.isfinite(score[ranked])][: dna.top_n]
                raw = np.zeros(assets, dtype=float)
                if dna.weighting == "inverse_volatility":
                    raw[selected] = 1.0 / np.maximum(
                        volatility[decision, selected],
                        1e-12,
                    )
                else:
                    raw[selected] = 1.0
                target = _capped_weights(
                    raw,
                    budget=budget,
                    position_cap=dna.maximum_position_exposure,
                )
            if (
                set(np.flatnonzero(target > 1e-12))
                == set(np.flatnonzero(current > 1e-12))
                and np.max(np.abs(target - current)) < dna.rebalance_buffer
            ):
                target = current.copy()
            turnover = float(np.abs(target - current).sum())
            execution_turnover = turnover
            turnover_series[execution] = turnover
            current = target
        asset_return = opens[execution + 1] / opens[execution] - 1.0
        gross = float(current @ asset_return)
        cost = execution_turnover * one_way_cost
        returns[execution] = (1.0 - cost) * (1.0 + gross) - 1.0
    terminal = float(current.sum())
    if len(returns):
        turnover_series[-1] += terminal
        returns[-1] = (1.0 - terminal * one_way_cost) * (
            1.0 + returns[-1]
        ) - 1.0
    return returns, turnover_series


def _objectives(
    returns: np.ndarray,
    turnover: float,
) -> tuple[float, float, float, float]:
    positive = float(returns[returns > 0].sum())
    negative = float(abs(returns[returns < 0].sum()))
    profit_factor = positive / negative if negative > 0 else 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    ulcer = float(np.sqrt(np.mean(np.square(drawdown))))
    terminal = float(equity[-1])
    efficiency = turnover / max(terminal, 1e-12)
    standard = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / standard) if standard > 0 else 0.0
    return profit_factor, ulcer, efficiency, sharpe


def _pareto_indices(values: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for index, row in enumerate(values):
        dominates = (
            (values[:, 0] >= row[0])
            & (values[:, 1] <= row[1])
            & (values[:, 2] <= row[2])
            & (
                (values[:, 0] > row[0])
                | (values[:, 1] < row[1])
                | (values[:, 2] < row[2])
            )
        )
        if not bool(dominates.any()):
            selected.append(index)
    return np.asarray(selected, dtype=int)


def run_portfolio_storm(
    frames: Mapping[str, pd.DataFrame],
    dna: tuple[PortfolioStormDNA, ...],
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    prior_known_trials: int,
    maximum_survivors: int = 64,
) -> tuple[dict[str, Any], np.ndarray, pd.DatetimeIndex]:
    """Run predeclared DNA and return an orderless development-only selection."""

    if not dna:
        raise ValueError("storm DNA cannot be empty")
    opens, closes = _panel(frames)
    cache = _features(closes, dna)
    cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    matrix = np.zeros((len(opens) - 1, len(dna)), dtype=np.float32)
    development_turnovers = np.zeros(len(dna), dtype=float)
    open_values = opens.to_numpy(dtype=float)
    close_values = closes.to_numpy(dtype=float)
    common_warmup = (
        max(
            max(row.momentum_slow, row.asset_ema_period, row.btc_ema_period)
            for row in dna
        )
        + 1
    )
    retained_observations = len(opens) - 1 - common_warmup
    development_end = int(retained_observations * 0.60)
    for index, row in enumerate(dna):
        returns, turnover_series = _simulate(
            open_values,
            close_values,
            cache,
            row,
            one_way_cost=cost,
        )
        matrix[:, index] = returns.astype(np.float32)
        development_turnovers[index] = float(
            turnover_series[
                common_warmup : common_warmup + development_end
            ].sum()
        )
    matrix = matrix[common_warmup:]
    observations = matrix.shape[0]
    validation_end = int(observations * 0.80)
    development = matrix[:development_end].astype(float)
    objective_rows = np.asarray(
        [
            _objectives(
                development[:, index],
                development_turnovers[index],
            )
            for index in range(len(dna))
        ],
        dtype=float,
    )
    pareto = _pareto_indices(objective_rows[:, :3])
    if len(pareto) > maximum_survivors:
        pf_rank = pd.Series(objective_rows[pareto, 0]).rank(pct=True)
        ui_rank = pd.Series(-objective_rows[pareto, 1]).rank(pct=True)
        te_rank = pd.Series(-objective_rows[pareto, 2]).rank(pct=True)
        robust = (pf_rank + ui_rank + te_rank).to_numpy(dtype=float)
        pareto = pareto[np.argsort(robust)[::-1][:maximum_survivors]]

    development_frame = pd.DataFrame(
        development,
        columns=[row.dna_hash for row in dna],
    )
    pbo, pbo_logits = probability_of_backtest_overfitting(
        development_frame,
    )
    trial_sharpes = objective_rows[:, 3]
    survivors: list[dict[str, Any]] = []
    for index in pareto:
        row = dna[int(index)]
        validation = matrix[development_end:validation_end, index].astype(float)
        confirmation = matrix[validation_end:, index].astype(float)
        dsr = deflated_sharpe_ratio(
            pd.Series(development[:, index]),
            trial_sharpes,
            observed_sharpe=float(trial_sharpes[index]),
            total_trials=prior_known_trials + len(dna),
        )
        survivors.append(
            {
                "strategy_dna_hash": row.dna_hash,
                "parameters": asdict(row),
                "development": dict(
                    zip(
                        (
                            "portfolio_period_profit_factor",
                            "ulcer_index",
                            "turnover_efficiency",
                            "daily_sharpe",
                        ),
                        objective_rows[index],
                        strict=True,
                    )
                ),
                "validation": {
                    "net_return": float(np.prod(1.0 + validation) - 1.0),
                    "mean_return": float(validation.mean()),
                },
                "confirmation": {
                    "net_return": float(np.prod(1.0 + confirmation) - 1.0),
                    "mean_return": float(confirmation.mean()),
                },
                "deflated_sharpe_probability": dsr,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
        )
    report = {
        "schema_version": "portfolio_storm_report_v1",
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "PORTFOLIO_STORM_V1",
        "engine_version": STORM_ENGINE_VERSION,
        "trial_count": len(dna),
        "prior_known_trials": prior_known_trials,
        "total_known_trials": prior_known_trials + len(dna),
        "search_space_hash": stable_hash(
            [row.dna_hash for row in dna],
            length=64,
        ),
        "selection_basis": "DEVELOPMENT_ONLY",
        "selection_integrity": {
            "development_returns_only": True,
            "development_turnover_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "split": {
            "development_observations": development_end,
            "validation_observations": validation_end - development_end,
            "confirmation_observations": observations - validation_end,
        },
        "pareto_survivor_count": len(survivors),
        "positive_validation_survivors": sum(
            float(row["validation"]["net_return"]) > 0
            for row in survivors
        ),
        "positive_confirmation_survivors": sum(
            float(row["confirmation"]["net_return"]) > 0
            for row in survivors
        ),
        "pareto_survivors": survivors,
        "multiple_testing": {
            "pbo_development_all_trials": pbo,
            "pbo_logits": list(pbo_logits),
            "dsr_total_trial_denominator": prior_known_trials + len(dna),
            "white_spa_status": (
                "DEFERRED_NO_PASS_CLAIM_LARGE_MATRIX"
            ),
        },
        "maximum_total_exposure": 0.40,
        "maximum_position_exposure": 0.20,
        "minimum_cash": 0.60,
        "next_open_execution": True,
        "research_pass": False,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }
    return report, matrix, opens.index[1 + common_warmup :]


__all__ = [
    "STORM_ENGINE_VERSION",
    "STORM_SEED",
    "STORM_TRIAL_COUNT",
    "PortfolioStormDNA",
    "preregistered_storm_dna",
    "run_portfolio_storm",
    "storm_plan",
]
