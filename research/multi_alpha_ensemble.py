"""Causal portfolio-of-strategies allocator for frozen classical alpha DNA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _capped_allocations,
    _effective_sample_size,
    _profit_factor,
    _validated_panel,
)
from utils.common import stable_hash

MULTI_ALPHA_ENSEMBLE_FAMILY = "FROZEN_CLASSICAL_MULTI_ALPHA_ENSEMBLE"
MULTI_ALPHA_ENSEMBLE_ENGINE_VERSION = "1.0.0"

ABSOLUTE_MOMENTUM_COMPONENT_DNA = (
    "1d14e010495a45d9ea60090077c73c1fd357d897832611ca9f722f8f670c2012"
)
TURTLE_BREAKOUT_COMPONENT_DNA = (
    "9e7270baf87d72464aa078843d68fc4a284b45e332b20e4cfd13b5adcdc64d11"
)
VOLATILITY_CONTRACTION_COMPONENT_DNA = (
    "649975eeb78d897826e65ec06d73523e34299640183e751129c7a11a310a7b17"
)

FROZEN_COMPONENT_DNA = (
    ("ABSOLUTE_MOMENTUM_VOL_05", ABSOLUTE_MOMENTUM_COMPONENT_DNA),
    ("TURTLE_20_10_EMA200_EQUAL", TURTLE_BREAKOUT_COMPONENT_DNA),
    (
        "VOLATILITY_CONTRACTION_PRIMARY",
        VOLATILITY_CONTRACTION_COMPONENT_DNA,
    ),
)


@dataclass(frozen=True, slots=True)
class MultiAlphaEnsembleParameters:
    """The single preregistered v1 meta-allocation DNA."""

    target_annualized_volatility: float = 0.10
    volatility_lookback: int = 60
    rebalance_weekday: int = 6
    maximum_positions: int = 2
    component_allocation: str = "EQUAL_FIXED_SLEEVES"
    risk_model: str = "CAUSAL_DIAGONAL_VOLATILITY"
    component_dna: tuple[tuple[str, str], ...] = FROZEN_COMPONENT_DNA

    def __post_init__(self) -> None:
        if self.target_annualized_volatility != 0.10:
            raise ValueError("v1 volatility target is fixed at 10%")
        if self.volatility_lookback != 60:
            raise ValueError("v1 volatility lookback is fixed at 60")
        if self.rebalance_weekday != 6:
            raise ValueError("v1 rebalance weekday is fixed at Sunday")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")
        if self.component_allocation != "EQUAL_FIXED_SLEEVES":
            raise ValueError("v1 requires equal fixed sleeves")
        if self.risk_model != "CAUSAL_DIAGONAL_VOLATILITY":
            raise ValueError("v1 risk model is fixed")
        if self.component_dna != FROZEN_COMPONENT_DNA:
            raise ValueError("component DNA differs from preregistration")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": MULTI_ALPHA_ENSEMBLE_FAMILY,
                "engine_version": MULTI_ALPHA_ENSEMBLE_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


@dataclass(frozen=True)
class MultiAlphaEnsembleResult:
    parameters: MultiAlphaEnsembleParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    component_diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": MULTI_ALPHA_ENSEMBLE_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": MULTI_ALPHA_ENSEMBLE_ENGINE_VERSION,
            "parameters": asdict(self.parameters),
            "portfolio_policy": asdict(self.portfolio_policy),
            "portfolio_policy_hash": self.portfolio_policy.policy_hash,
            "execution_identity": stable_hash(
                {
                    "strategy_dna_hash": self.parameters.dna_hash,
                    "portfolio_policy_hash": (
                        self.portfolio_policy.policy_hash
                    ),
                },
                length=64,
            ),
            "metrics": dict(self.metrics),
            "integrity": dict(self.integrity),
            "cost_breakdown": dict(self.cost_breakdown),
            "component_diagnostics": dict(
                self.component_diagnostics
            ),
        }


def _validate_component_weights(
    component_weights: Mapping[str, pd.DataFrame],
    *,
    index: pd.DatetimeIndex,
    markets: pd.Index,
) -> dict[str, pd.DataFrame]:
    required = {name for name, _ in FROZEN_COMPONENT_DNA}
    if set(component_weights) != required:
        raise ValueError(
            "component weight labels differ from frozen ensemble DNA"
        )
    result: dict[str, pd.DataFrame] = {}
    for name, raw in component_weights.items():
        source = raw.copy()
        if not source.index.is_unique:
            conflicts = [
                timestamp
                for timestamp, group in source.groupby(
                    level=0,
                    sort=False,
                )
                if len(group.drop_duplicates()) != 1
            ]
            if conflicts:
                raise ValueError(
                    f"conflicting duplicate component weights: {name}"
                )
            source = source[
                ~source.index.duplicated(keep="last")
            ]
        frame = (
            source
            .reindex(index=index, columns=markets)
            .ffill()
            .fillna(0.0)
            .astype(float)
        )
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"non-finite component weights: {name}")
        if bool((frame < -1e-12).any().any()):
            raise ValueError(f"short component weights: {name}")
        result[name] = frame.clip(lower=0.0)
    return result


def _risk_targeted_desired(
    raw_weights: pd.DataFrame,
    annualized_volatility: pd.DataFrame,
    *,
    parameters: MultiAlphaEnsembleParameters,
    policy: RotationPortfolioPolicy,
) -> pd.DataFrame:
    desired = pd.DataFrame(
        0.0,
        index=raw_weights.index,
        columns=raw_weights.columns,
    )
    for timestamp in raw_weights.index:
        raw = raw_weights.loc[timestamp]
        active = raw[raw > 1e-12].sort_values(
            ascending=False
        ).head(parameters.maximum_positions)
        if active.empty:
            continue
        volatility = (
            annualized_volatility.loc[timestamp]
            .reindex(active.index)
            .replace(0.0, np.nan)
            .dropna()
        )
        active = active.reindex(volatility.index).dropna()
        if active.empty:
            continue
        proportions = active / active.sum()
        diagonal_volatility = float(
            np.sqrt(
                np.square(proportions * volatility).sum()
            )
        )
        if not math.isfinite(diagonal_volatility):
            continue
        total_exposure = min(
            policy.maximum_total_exposure,
            parameters.target_annualized_volatility
            / max(diagonal_volatility, 1e-12),
        )
        allocations = _capped_allocations(
            active,
            total_exposure=total_exposure,
            maximum_position_exposure=(
                policy.maximum_position_exposure
            ),
        )
        desired.loc[timestamp, allocations.index] = allocations
    return desired


def _metrics(
    equity: pd.Series,
    weights: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    returns = equity.pct_change(fill_method=None).dropna()
    elapsed_days = max(
        1.0,
        (equity.index[-1] - equity.index[0]).total_seconds()
        / 86_400.0,
    )
    years = elapsed_days / 365.25
    standard = float(returns.std(ddof=0))
    downside = returns[returns < 0.0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))))
        if len(downside)
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    effective_sample, lag_one = _effective_sample_size(
        returns
    )
    exposure = weights.sum(axis=1)
    return {
        "net_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(
            equity.iloc[-1] ** (1.0 / years) - 1.0
        ),
        "annualized_volatility": standard * math.sqrt(365.25),
        "sharpe": (
            float(
                returns.mean()
                / standard
                * math.sqrt(365.25)
            )
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(365.25)
            )
            if downside_deviation > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(
            returns
        ),
        "portfolio_period_effective_sample_size": (
            effective_sample
        ),
        "lag_one_autocorrelation": lag_one,
        "rebalance_count": int(
            (decisions["turnover"] > 1e-12).sum()
        ),
        "decision_count": int(len(decisions)),
        "average_exposure": float(exposure.mean()),
        "maximum_realized_exposure": float(exposure.max()),
        "maximum_position_exposure_observed": float(
            weights.max(axis=1).max()
        ),
        "maximum_positions_observed": int(
            (weights > 1e-12).sum(axis=1).max()
        ),
        "cash_fraction_average": float(1.0 - exposure.mean()),
        "observations": int(len(returns)),
    }


def backtest_multi_alpha_ensemble(
    frames: Mapping[str, pd.DataFrame],
    component_weights: Mapping[str, pd.DataFrame],
    parameters: MultiAlphaEnsembleParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> MultiAlphaEnsembleResult:
    """Combine frozen component weights and execute one day later."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark_market,
        portfolio_policy=portfolio_policy,
    )
    components = _validate_component_weights(
        component_weights,
        index=closes.index,
        markets=closes.columns,
    )
    sleeve_weight = 1.0 / len(components)
    combined = sum(
        (frame * sleeve_weight for frame in components.values()),
        start=pd.DataFrame(
            0.0,
            index=closes.index,
            columns=closes.columns,
        ),
    )
    annualized_volatility = (
        closes.pct_change(fill_method=None)
        .rolling(
            parameters.volatility_lookback,
            min_periods=parameters.volatility_lookback,
        )
        .std(ddof=0)
        * math.sqrt(365.25)
    )
    desired = _risk_targeted_desired(
        combined,
        annualized_volatility,
        parameters=parameters,
        policy=portfolio_policy,
    )
    selected = desired > 1e-12
    selection_changed = selected.ne(selected.shift(1)).any(axis=1)
    scheduled = pd.Series(
        closes.index.weekday == parameters.rebalance_weekday,
        index=closes.index,
    )
    signal_event = scheduled | selection_changed
    signal_targets = (
        desired.where(signal_event, np.nan).ffill().fillna(0.0)
    )
    executed = signal_targets.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)
    start = parameters.volatility_lookback + 2
    executed = executed.iloc[start:].copy()

    open_returns = opens.shift(-1).div(opens).sub(1.0)
    open_returns.iloc[-1] = (
        closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    )
    open_returns = open_returns.reindex(executed.index)
    held_missing = (
        open_returns.where(executed > 1e-12).isna()
        & (executed > 1e-12)
    )
    if bool(held_missing.any().any()):
        raise ValueError(
            "held ensemble asset lacks causal next valuation"
        )
    gross_returns = (
        executed * open_returns.fillna(0.0)
    ).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].sum())
    terminal_turnover = float(executed.iloc[-1].sum())
    turnover.iloc[-1] += terminal_turnover
    one_way_cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    cost_fraction = turnover * one_way_cost
    net_returns = (
        (1.0 - cost_fraction) * (1.0 + gross_returns) - 1.0
    )
    equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()
    equity.name = "net_equity"
    gross_equity.name = "gross_equity"

    decisions: list[dict[str, Any]] = []
    index = list(closes.index)
    for execution_position in range(start, len(index)):
        decision_position = execution_position - 1
        if not bool(signal_event.iloc[decision_position]):
            continue
        execution_at = index[execution_position]
        target = executed.loc[execution_at]
        execution_turnover = float(
            turnover.loc[execution_at]
            - (
                terminal_turnover
                if execution_position == len(index) - 1
                else 0.0
            )
        )
        decisions.append(
            {
                "decision_at": index[decision_position],
                "executed_at": execution_at,
                "reason": "MULTI_ALPHA_REBALANCE",
                "scheduled": bool(
                    scheduled.iloc[decision_position]
                ),
                "turnover": execution_turnover,
                "expected_cost_fraction": (
                    execution_turnover * one_way_cost
                ),
                "target_weights": {
                    market: float(value)
                    for market, value in target.items()
                    if float(value) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
            }
        )
    decisions.append(
        {
            "decision_at": index[-1],
            "executed_at": index[-1],
            "reason": "TERMINAL_LIQUIDATION",
            "scheduled": False,
            "turnover": terminal_turnover,
            "expected_cost_fraction": (
                terminal_turnover * one_way_cost
            ),
            "target_weights": {},
            "cash_fraction": 1.0,
        }
    )
    decisions_frame = pd.DataFrame(decisions)
    active_decisions = decisions_frame[
        decisions_frame["reason"] != "TERMINAL_LIQUIDATION"
    ]
    metrics = _metrics(equity, executed, active_decisions)
    component_active_days = {
        name: int((frame.sum(axis=1) > 1e-12).sum())
        for name, frame in components.items()
    }
    integrity = {
        "closed_candles_only": True,
        "component_weights_known_before_meta_decision": True,
        "meta_decision_execution_next_open": True,
        "one_additional_causal_lag": True,
        "long_only_spot": bool(
            (executed >= -1e-12).all().all()
        ),
        "maximum_exposure_respected": (
            metrics["maximum_realized_exposure"]
            <= portfolio_policy.maximum_total_exposure + 1e-12
        ),
        "maximum_position_exposure_respected": (
            metrics["maximum_position_exposure_observed"]
            <= portfolio_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": (
            metrics["maximum_realized_exposure"]
            <= 1.0 - portfolio_policy.minimum_cash + 1e-12
        ),
        "maximum_positions_respected": (
            metrics["maximum_positions_observed"]
            <= parameters.maximum_positions
        ),
        "component_dna_frozen": True,
        "component_selection_optimized": False,
        "orders_generated": 0,
    }
    return MultiAlphaEnsembleResult(
        parameters=parameters,
        portfolio_policy=portfolio_policy,
        metrics=metrics,
        integrity=integrity,
        cost_breakdown={
            "fee_rate": float(fee_rate),
            "slippage_bps": float(slippage_bps),
            "spread_bps": float(spread_bps),
            "one_way_cost_rate": float(one_way_cost),
            "turnover": float(turnover.sum()),
            "total_cost_fraction": float(cost_fraction.sum()),
            "gross_ending_equity": float(gross_equity.iloc[-1]),
            "net_ending_equity": float(equity.iloc[-1]),
        },
        equity_curve=equity,
        gross_equity_curve=gross_equity,
        executed_weights=executed,
        decisions=decisions_frame,
        component_diagnostics={
            "component_count": len(components),
            "component_dna": dict(FROZEN_COMPONENT_DNA),
            "fixed_sleeve_weight": sleeve_weight,
            "component_active_days": component_active_days,
            "selection_trials_in_meta_family": 1,
        },
    )


__all__ = [
    "ABSOLUTE_MOMENTUM_COMPONENT_DNA",
    "FROZEN_COMPONENT_DNA",
    "MULTI_ALPHA_ENSEMBLE_ENGINE_VERSION",
    "MULTI_ALPHA_ENSEMBLE_FAMILY",
    "TURTLE_BREAKOUT_COMPONENT_DNA",
    "VOLATILITY_CONTRACTION_COMPONENT_DNA",
    "MultiAlphaEnsembleParameters",
    "MultiAlphaEnsembleResult",
    "backtest_multi_alpha_ensemble",
]
