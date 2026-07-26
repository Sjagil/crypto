"""Causal long-only multi-horizon trend ensemble for the allowed universe.

The family deliberately contains one fixed classical strategy DNA. Each asset
must have positive 240-day momentum before shorter 20/60/120-day votes can
increase its allocation. Decisions use only the just-closed daily candle and
execute at the next daily open. Cash earns zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _validated_panel,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

MULTI_HORIZON_TREND_ENGINE_VERSION = "1.0.0"
MULTI_HORIZON_TREND_FAMILY = (
    "FIXED_LONG_ONLY_MULTI_HORIZON_TREND_ENSEMBLE"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class MultiHorizonTrendParameters:
    """The one preregistered v1 strategy DNA."""

    momentum_horizons: tuple[int, ...] = (20, 60, 120, 240)
    structural_horizon: int = 240
    maximum_positions: int = 4
    target_weight_per_full_vote: float = 0.20
    rebalance_frequency: str = "DAILY_ON_VOTE_CHANGE"
    weighting: str = "POSITIVE_VOTE_FRACTION"
    structural_gate: str = "POSITIVE_240_DAY_MOMENTUM"

    def __post_init__(self) -> None:
        if self.momentum_horizons != (20, 60, 120, 240):
            raise ValueError("v1 momentum horizons are frozen")
        if self.structural_horizon != 240:
            raise ValueError("v1 structural horizon is frozen")
        if self.maximum_positions != 4:
            raise ValueError("v1 maximum positions is frozen at four")
        if self.target_weight_per_full_vote != 0.20:
            raise ValueError("v1 full-vote weight is frozen at 20%")
        if self.rebalance_frequency != "DAILY_ON_VOTE_CHANGE":
            raise ValueError("v1 rebalance frequency is frozen")
        if self.weighting != "POSITIVE_VOTE_FRACTION":
            raise ValueError("v1 weighting is frozen")
        if self.structural_gate != "POSITIVE_240_DAY_MOMENTUM":
            raise ValueError("v1 structural gate is frozen")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": MULTI_HORIZON_TREND_FAMILY,
                "engine_version": MULTI_HORIZON_TREND_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def multi_horizon_trend_parameter_set(
) -> tuple[MultiHorizonTrendParameters, ...]:
    """Return exactly one fixed, non-optimized v1 candidate."""

    rows = (MultiHorizonTrendParameters(),)
    if len({row.dna_hash for row in rows}) != 1:
        raise RuntimeError("multi-horizon trend DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class MultiHorizonTrendResult:
    parameters: MultiHorizonTrendParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    vote_fractions: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": MULTI_HORIZON_TREND_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": MULTI_HORIZON_TREND_ENGINE_VERSION,
            "timeframe": "1d",
            "periods_per_year": DAILY_PERIODS_PER_YEAR,
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
        }


def _vote_targets(
    closes: pd.DataFrame,
    *,
    parameters: MultiHorizonTrendParameters,
    policy: RotationPortfolioPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = {
        horizon: closes.pct_change(
            horizon,
            fill_method=None,
        )
        for horizon in parameters.momentum_horizons
    }
    votes = sum(
        (momentum[horizon] > 0.0).astype(float)
        for horizon in parameters.momentum_horizons
    ) / float(len(parameters.momentum_horizons))
    history = closes.notna().cumsum()
    structural = momentum[parameters.structural_horizon] > 0.0
    eligible = (
        structural
        & closes.notna()
        & (
            history
            >= max(
                policy.minimum_history_observations,
                parameters.structural_horizon + 1,
            )
        )
    )
    votes = votes.where(eligible, 0.0)
    desired = votes * parameters.target_weight_per_full_vote
    desired = desired.clip(
        lower=0.0,
        upper=policy.maximum_position_exposure,
    )
    row_total = desired.sum(axis=1)
    scale = (
        policy.maximum_total_exposure
        / row_total.where(row_total > 0.0, 1.0)
    ).clip(upper=1.0)
    desired = desired.mul(scale, axis=0)
    return desired, votes


def backtest_multi_horizon_trend(
    frames: Mapping[str, pd.DataFrame],
    parameters: MultiHorizonTrendParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> MultiHorizonTrendResult:
    """Run the exact daily-close/next-open multi-horizon trend policy."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    required = {"BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"}
    if set(portfolio_policy.allowed_markets) != required:
        raise ValueError(
            "multi-horizon trend policy requires the four allowed markets"
        )
    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=portfolio_policy,
    )
    if set(closes.columns) != required:
        raise ValueError(
            "multi-horizon trend requires exactly four allowed markets"
        )
    warmup = max(
        max(parameters.momentum_horizons) + 1,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient multi-horizon trend history")

    desired, votes = _vote_targets(
        closes,
        parameters=parameters,
        policy=portfolio_policy,
    )
    # Every target uses the completed close at t and becomes executable only
    # at open t+1. Repeated targets create no turnover.
    executed = desired.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)
    executed = executed.iloc[warmup + 1 :].copy()
    votes = votes.reindex(executed.index).copy()

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
            "held multi-horizon position lacks next valuation"
        )
    gross_period_returns = (
        executed * open_returns.fillna(0.0)
    ).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].abs().sum())
    turnover.iloc[-1] += float(executed.iloc[-1].sum())
    one_way_cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    cost_fraction = turnover * one_way_cost
    if bool((cost_fraction >= 1.0).any()):
        raise ValueError("multi-horizon costs exhaust equity")
    net_period_returns = (
        (1.0 - cost_fraction)
        * (1.0 + gross_period_returns)
        - 1.0
    )
    gross_equity = (1.0 + gross_period_returns).cumprod()
    equity = (1.0 + net_period_returns).cumprod()
    gross_equity.name = "gross_equity"
    equity.name = "equity"

    decision_rows: list[dict[str, Any]] = []
    for position, executed_at in enumerate(executed.index):
        executed_turnover = float(turnover.iloc[position])
        if executed_turnover <= 1e-12:
            continue
        signal_position = max(0, closes.index.get_loc(executed_at) - 1)
        target = executed.iloc[position]
        signal_votes = votes.iloc[position]
        decision_rows.append(
            {
                "decision_at": closes.index[signal_position],
                "executed_at": executed_at,
                "reason": "MULTI_HORIZON_VOTE_CHANGE",
                "turnover": executed_turnover,
                "expected_cost_fraction": (
                    executed_turnover * one_way_cost
                ),
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "vote_fractions": {
                    market: float(value)
                    for market, value in signal_votes.items()
                },
                "cash_fraction": float(1.0 - target.sum()),
            }
        )
    decisions = pd.DataFrame(
        decision_rows,
        columns=[
            "decision_at",
            "executed_at",
            "reason",
            "turnover",
            "expected_cost_fraction",
            "target_weights",
            "vote_fractions",
            "cash_fraction",
        ],
    )
    metrics = _performance_metrics(equity, executed, decisions)
    exposure = executed.sum(axis=1)
    integrity = {
        "allowed_markets_only": set(executed.columns) == required,
        "maximum_exposure_respected": bool(
            exposure.max()
            <= portfolio_policy.maximum_total_exposure + 1e-12
        ),
        "maximum_position_exposure_respected": bool(
            executed.max(axis=1).max()
            <= portfolio_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": bool(
            exposure.max()
            <= 1.0 - portfolio_policy.minimum_cash + 1e-12
        ),
        "maximum_positions_respected": bool(
            (executed > 1e-12).sum(axis=1).max()
            <= parameters.maximum_positions
        ),
        "closed_candles_only": True,
        "decision_at_close_execution_next_open": True,
        "point_in_time_history_gate": True,
        "structural_horizon_gate": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "zero_yield_cash": True,
        "orders_generated": 0,
    }
    if not all(
        value
        for key, value in integrity.items()
        if key != "orders_generated"
    ):
        raise RuntimeError(
            f"multi-horizon trend integrity failure: {integrity}"
        )
    return MultiHorizonTrendResult(
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
        decisions=decisions,
        vote_fractions=votes,
    )


def multi_horizon_trend_period_metrics(
    equity: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, Any], pd.Series]:
    """Return daily portfolio metrics for one fixed calendar period."""

    selected = equity.loc[
        pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")
    ]
    returns = selected.pct_change(fill_method=None).dropna()
    if len(selected) < 2:
        return (
            {
                "net_return": 0.0,
                "annualized_return": 0.0,
                "sharpe": 0.0,
                "maximum_drawdown": 0.0,
                "portfolio_period_profit_factor": 0.0,
                "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
                "positive_observations": 0,
                "negative_observations": 0,
                "periods_per_year": DAILY_PERIODS_PER_YEAR,
            },
            returns,
        )
    elapsed_days = max(
        1.0,
        (selected.index[-1] - selected.index[0]).total_seconds()
        / 86_400.0,
    )
    years = elapsed_days / DAILY_PERIODS_PER_YEAR
    standard = float(returns.std(ddof=0))
    drawdown = selected / selected.cummax() - 1.0
    net_return = float(selected.iloc[-1] / selected.iloc[0] - 1.0)
    positive = float(returns[returns > 0.0].sum())
    negative = abs(float(returns[returns < 0.0].sum()))
    profit_factor = (
        positive / negative
        if negative > 0.0
        else (math.inf if positive > 0.0 else 0.0)
    )
    return (
        {
            "net_return": net_return,
            "annualized_return": float(
                (1.0 + net_return) ** (1.0 / years) - 1.0
            ),
            "sharpe": (
                float(
                    returns.mean()
                    / standard
                    * math.sqrt(DAILY_PERIODS_PER_YEAR)
                )
                if standard > 0.0
                else 0.0
            ),
            "maximum_drawdown": float(drawdown.min()),
            "portfolio_period_profit_factor": float(profit_factor),
            "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
            "positive_observations": int((returns > 0.0).sum()),
            "negative_observations": int((returns < 0.0).sum()),
            "periods_per_year": DAILY_PERIODS_PER_YEAR,
        },
        returns,
    )


__all__ = [
    "DAILY_PERIODS_PER_YEAR",
    "MULTI_HORIZON_TREND_ENGINE_VERSION",
    "MULTI_HORIZON_TREND_FAMILY",
    "MultiHorizonTrendParameters",
    "MultiHorizonTrendResult",
    "backtest_multi_horizon_trend",
    "multi_horizon_trend_parameter_set",
    "multi_horizon_trend_period_metrics",
]
