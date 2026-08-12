"""Causal volatility-contraction breakout portfolio research.

An asset becomes eligible only when a low-volatility state, measured against a
strictly older causal distribution, is followed by a prior-channel breakout.
Signals use completed daily closes and become weights at the next available
open. The module has no order or promotion authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
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

VOLATILITY_CONTRACTION_FAMILY = (
    "MULTI_ASSET_VOLATILITY_CONTRACTION_BREAKOUT"
)
VOLATILITY_CONTRACTION_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class VolatilityContractionParameters:
    """Immutable DNA for one contraction-then-breakout path."""

    volatility_lookback: int
    contraction_quantile: float
    entry_lookback: int
    exit_lookback: int
    target_annualized_volatility: float
    contraction_baseline: int = 252
    contraction_memory: int = 5
    asset_ema_period: int = 200
    btc_ema_period: int = 200
    maximum_positions: int = 2
    rebalance_weekday: int = 6

    def __post_init__(self) -> None:
        if self.volatility_lookback not in {20, 40}:
            raise ValueError("undeclared contraction volatility lookback")
        if self.contraction_quantile not in {0.20, 0.30}:
            raise ValueError("undeclared contraction quantile")
        if (self.entry_lookback, self.exit_lookback) not in {
            (20, 10),
            (55, 20),
        }:
            raise ValueError("undeclared entry/exit channel pair")
        if self.target_annualized_volatility not in {0.10, 0.15}:
            raise ValueError("undeclared contraction volatility target")
        if self.contraction_baseline < 100:
            raise ValueError("contraction baseline is too short")
        if self.contraction_memory < 1:
            raise ValueError("contraction memory must be positive")
        if min(self.asset_ema_period, self.btc_ema_period) < 100:
            raise ValueError("trend filters must be long horizon")
        if self.maximum_positions not in {1, 2}:
            raise ValueError("maximum positions must be one or two")
        if self.rebalance_weekday not in range(7):
            raise ValueError("rebalance weekday must be in [0, 6]")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": VOLATILITY_CONTRACTION_FAMILY,
                "engine_version": (
                    VOLATILITY_CONTRACTION_ENGINE_VERSION
                ),
                "parameters": asdict(self),
            },
            length=64,
        )


def volatility_contraction_parameter_set(
) -> tuple[VolatilityContractionParameters, ...]:
    """Return the exact 16-trial predeclared family."""

    rows = tuple(
        VolatilityContractionParameters(
            volatility_lookback=volatility_lookback,
            contraction_quantile=quantile,
            entry_lookback=entry,
            exit_lookback=exit_,
            target_annualized_volatility=target,
        )
        for (
            volatility_lookback,
            quantile,
            (entry, exit_),
            target,
        ) in product(
            (20, 40),
            (0.20, 0.30),
            ((20, 10), (55, 20)),
            (0.10, 0.15),
        )
    )
    if len(rows) != 16:
        raise RuntimeError("contraction family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("contraction family contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class VolatilityContractionResult:
    parameters: VolatilityContractionParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    signal_diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": VOLATILITY_CONTRACTION_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": (
                VOLATILITY_CONTRACTION_ENGINE_VERSION
            ),
            "parameters": asdict(self.parameters),
            "portfolio_policy": asdict(self.portfolio_policy),
            "portfolio_policy_hash": (
                self.portfolio_policy.policy_hash
            ),
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
            "signal_diagnostics": dict(self.signal_diagnostics),
        }


def _holding_state(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
) -> pd.DataFrame:
    """Build independent long-only asset states from causal events."""

    entry_values = entries.to_numpy(dtype=bool)
    exit_values = exits.to_numpy(dtype=bool)
    state = np.zeros_like(entry_values, dtype=bool)
    current = np.zeros(entry_values.shape[1], dtype=bool)
    for row in range(len(entry_values)):
        current = current & ~exit_values[row]
        current = current | (entry_values[row] & ~current)
        state[row] = current
    return pd.DataFrame(
        state,
        index=entries.index,
        columns=entries.columns,
    )


def _desired_weights(
    *,
    active: pd.DataFrame,
    strength: pd.DataFrame,
    annualized_volatility: pd.DataFrame,
    parameters: VolatilityContractionParameters,
    policy: RotationPortfolioPolicy,
) -> pd.DataFrame:
    desired = pd.DataFrame(
        0.0,
        index=active.index,
        columns=active.columns,
    )
    for timestamp in active.index:
        eligible = strength.loc[timestamp].where(
            active.loc[timestamp]
        ).dropna()
        if eligible.empty:
            continue
        selected = list(
            eligible.sort_values(ascending=False)
            .head(parameters.maximum_positions)
            .index
        )
        vol = (
            annualized_volatility.loc[timestamp]
            .reindex(selected)
            .replace(0.0, np.nan)
            .dropna()
        )
        if vol.empty:
            continue
        raw = 1.0 / vol
        base = raw / raw.sum()
        diagonal_volatility = float(
            np.sqrt(np.square(base * vol).sum())
        )
        if not math.isfinite(diagonal_volatility):
            continue
        target_exposure = min(
            policy.maximum_total_exposure,
            parameters.target_annualized_volatility
            / max(diagonal_volatility, 1e-12),
        )
        allocations = _capped_allocations(
            raw,
            total_exposure=target_exposure,
            maximum_position_exposure=(
                policy.maximum_position_exposure
            ),
        )
        desired.loc[timestamp, allocations.index] = allocations
    return desired


def _performance_metrics(
    equity: pd.Series,
    weights: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    periods_per_year: float = 365.25,
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
    turnover_events = (
        int((decisions["turnover"] > 1e-12).sum())
        if not decisions.empty
        else 0
    )
    return {
        "net_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(
            equity.iloc[-1] ** (1.0 / years) - 1.0
        ),
        "annualized_volatility": standard * math.sqrt(periods_per_year),
        "sharpe": (
            float(
                returns.mean()
                / standard
                * math.sqrt(periods_per_year)
            )
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(periods_per_year)
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
        "rebalance_count": turnover_events,
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
        "periods_per_year": float(periods_per_year),
    }


def backtest_volatility_contraction(
    frames: Mapping[str, pd.DataFrame],
    parameters: VolatilityContractionParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> VolatilityContractionResult:
    """Run one exact causal contraction-breakout portfolio path."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=portfolio_policy,
    )
    warmup = max(
        parameters.contraction_baseline
        + parameters.volatility_lookback,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        parameters.entry_lookback,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError(
            "insufficient history for contraction breakout"
        )
    daily_returns = closes.pct_change(fill_method=None)
    annualized_volatility = (
        daily_returns.rolling(
            parameters.volatility_lookback,
            min_periods=parameters.volatility_lookback,
        ).std(ddof=0)
        * math.sqrt(365.25)
    )
    contraction_threshold = (
        annualized_volatility.shift(1).rolling(
            parameters.contraction_baseline,
            min_periods=parameters.contraction_baseline,
        ).quantile(parameters.contraction_quantile)
    )
    contraction = (
        annualized_volatility <= contraction_threshold
    )
    recent_contraction = (
        contraction.astype(float)
        .rolling(
            parameters.contraction_memory,
            min_periods=1,
        )
        .max()
        .gt(0.0)
    )
    upper = closes.shift(1).rolling(
        parameters.entry_lookback,
        min_periods=parameters.entry_lookback,
    ).max()
    lower = closes.shift(1).rolling(
        parameters.exit_lookback,
        min_periods=parameters.exit_lookback,
    ).min()
    asset_ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    btc_ema = closes[benchmark].ewm(
        span=parameters.btc_ema_period,
        adjust=False,
        min_periods=parameters.btc_ema_period,
    ).mean()
    btc_regime = closes[benchmark] > btc_ema
    history_eligible = (
        closes.notna().cumsum()
        >= portfolio_policy.minimum_history_observations
    )
    valid = (
        history_eligible
        & closes.notna()
        & upper.notna()
        & lower.notna()
        & asset_ema.notna()
        & annualized_volatility.notna()
        & contraction_threshold.notna()
    )
    entries = (
        valid
        & recent_contraction
        & (closes > upper)
        & (closes > asset_ema)
    ).mul(btc_regime, axis=0)
    exits = (
        ~valid
        | (closes < lower)
        | (closes < asset_ema)
    ) | pd.DataFrame(
        np.repeat(
            (~btc_regime).to_numpy()[:, None],
            len(closes.columns),
            axis=1,
        ),
        index=closes.index,
        columns=closes.columns,
    )
    active = _holding_state(entries, exits)
    strength = (
        (closes / upper - 1.0)
        / annualized_volatility.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    desired = _desired_weights(
        active=active,
        strength=strength,
        annualized_volatility=annualized_volatility,
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
    signal_targets = desired.where(signal_event, np.nan).ffill()
    signal_targets = signal_targets.fillna(0.0)
    executed = signal_targets.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)

    start = warmup + 1
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
            "held contraction asset lacks causal next valuation"
        )
    gross_period_returns = (
        executed * open_returns.fillna(0.0)
    ).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].abs().sum())
    terminal_turnover = float(executed.iloc[-1].sum())
    turnover.iloc[-1] += terminal_turnover
    one_way_cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    cost_fraction = turnover * one_way_cost
    if bool((cost_fraction >= 1.0).any()):
        raise ValueError("contraction costs exhaust portfolio equity")
    net_period_returns = (
        (1.0 - cost_fraction)
        * (1.0 + gross_period_returns)
        - 1.0
    )
    equity = (1.0 + net_period_returns).cumprod()
    gross_equity = (1.0 + gross_period_returns).cumprod()
    equity.name = "net_equity"
    gross_equity.name = "gross_equity"

    decisions: list[dict[str, Any]] = []
    index = list(closes.index)
    prior_active = active.shift(1, fill_value=False)
    for execution_position in range(start, len(index)):
        decision_position = execution_position - 1
        execution_at = index[execution_position]
        if not bool(signal_event.iloc[decision_position]):
            continue
        target = executed.loc[execution_at]
        entry_assets = [
            market
            for market in closes.columns
            if bool(entries.iloc[decision_position][market])
        ]
        exit_assets = [
            market
            for market in closes.columns
            if bool(exits.iloc[decision_position][market])
            and bool(
                prior_active.iloc[decision_position][market]
            )
        ]
        reason = (
            "CONTRACTION_BREAKOUT_ENTRY"
            if entry_assets
            else "RISK_OR_CHANNEL_EXIT"
            if exit_assets
            else "SCHEDULED_RESIZE"
        )
        decisions.append(
            {
                "decision_at": index[decision_position],
                "executed_at": execution_at,
                "reason": reason,
                "scheduled": bool(
                    scheduled.iloc[decision_position]
                ),
                "turnover": float(
                    turnover.loc[execution_at]
                    - (
                        terminal_turnover
                        if execution_position == len(index) - 1
                        else 0.0
                    )
                ),
                "expected_cost_fraction": float(
                    max(
                        0.0,
                        turnover.loc[execution_at]
                        - (
                            terminal_turnover
                            if execution_position
                            == len(index) - 1
                            else 0.0
                        ),
                    )
                    * one_way_cost
                ),
                "target_weights": {
                    market: float(value)
                    for market, value in target.items()
                    if float(value) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "entry_signals": entry_assets,
                "exit_signals": exit_assets,
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
            "entry_signals": [],
            "exit_signals": list(
                executed.columns[
                    executed.iloc[-1] > 1e-12
                ]
            ),
        }
    )
    decisions_frame = pd.DataFrame(decisions)
    metrics = _performance_metrics(
        equity,
        executed,
        decisions_frame[
            decisions_frame["reason"] != "TERMINAL_LIQUIDATION"
        ],
    )
    integrity = {
        "closed_candles_only": True,
        "strictly_prior_contraction_distribution": True,
        "prior_channel_only": True,
        "decision_at_close_execution_next_open": True,
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
        "fail_closed_allowed_universe": bool(
            portfolio_policy.allowed_markets
        ),
        "point_in_time_asset_history_gate": True,
        "terminal_liquidation_recorded": (
            decisions_frame.iloc[-1]["reason"]
            == "TERMINAL_LIQUIDATION"
        ),
        "orders_generated": 0,
    }
    return VolatilityContractionResult(
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
        signal_diagnostics={
            "entry_signal_count": int(entries.sum().sum()),
            "contraction_observation_count": int(
                contraction.sum().sum()
            ),
            "active_asset_days": int(active.sum().sum()),
            "causal_contraction_threshold": True,
            "contraction_baseline": (
                parameters.contraction_baseline
            ),
        },
    )


__all__ = [
    "VOLATILITY_CONTRACTION_ENGINE_VERSION",
    "VOLATILITY_CONTRACTION_FAMILY",
    "VolatilityContractionParameters",
    "VolatilityContractionResult",
    "backtest_volatility_contraction",
    "volatility_contraction_parameter_set",
]
