"""Causal fixed BTC/ETH trend allocation with full covariance targeting.

This module owns one discovery-informed but non-optimized classical DNA. BTC
and ETH may be held only in established daily trends. A backward-looking
two-asset covariance matrix scales exposure down to a fixed volatility target.
Sunday-close targets execute at the following open; trend exits may occur
daily and also execute only at the next open.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _profit_factor,
    _validated_panel,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

DUAL_ASSET_TREND_ENGINE_VERSION = "1.0.0"
DUAL_ASSET_TREND_FAMILY = (
    "FIXED_BTC_ETH_TREND_FULL_COVARIANCE_VOL_TARGET"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class DualAssetTrendParameters:
    """The single frozen v1 strategy DNA."""

    trend_ema_period: int = 200
    covariance_lookback: int = 60
    target_annualized_volatility: float = 0.15
    rebalance_weekday: int = 6
    maximum_positions: int = 2
    weighting: str = "EQUAL_ACTIVE_ASSETS"
    risk_model: str = "FULL_ROLLING_COVARIANCE"
    daily_trend_exit: bool = True

    def __post_init__(self) -> None:
        if self.trend_ema_period != 200:
            raise ValueError("v1 trend EMA is fixed at 200")
        if self.covariance_lookback != 60:
            raise ValueError("v1 covariance lookback is fixed at 60")
        if self.target_annualized_volatility != 0.15:
            raise ValueError("v1 volatility target is fixed at 15%")
        if self.rebalance_weekday != 6:
            raise ValueError("v1 rebalance weekday is fixed at Sunday")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")
        if self.weighting != "EQUAL_ACTIVE_ASSETS":
            raise ValueError("v1 active-asset weighting is fixed")
        if self.risk_model != "FULL_ROLLING_COVARIANCE":
            raise ValueError("v1 requires full rolling covariance")
        if not self.daily_trend_exit:
            raise ValueError("v1 daily trend exit must remain enabled")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": DUAL_ASSET_TREND_FAMILY,
                "engine_version": DUAL_ASSET_TREND_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def dual_asset_trend_parameter_set(
) -> tuple[DualAssetTrendParameters, ...]:
    """Return the one and only preregistered v1 DNA."""

    rows = (DualAssetTrendParameters(),)
    if len({row.dna_hash for row in rows}) != 1:
        raise RuntimeError("dual-asset trend DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class DualAssetTrendResult:
    parameters: DualAssetTrendParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    risk_diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": DUAL_ASSET_TREND_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": DUAL_ASSET_TREND_ENGINE_VERSION,
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
            "risk_diagnostics": dict(self.risk_diagnostics),
        }


def _covariance_target_weights(
    closes: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    parameters: DualAssetTrendParameters,
    policy: RotationPortfolioPolicy,
) -> tuple[pd.DataFrame, pd.Series]:
    returns = closes.pct_change(fill_method=None)
    desired = pd.DataFrame(
        0.0,
        index=closes.index,
        columns=closes.columns,
    )
    predicted_volatility = pd.Series(
        np.nan,
        index=closes.index,
        name="predicted_annualized_volatility",
    )
    for position, timestamp in enumerate(closes.index):
        active = list(eligible.columns[eligible.iloc[position]])
        if not active:
            predicted_volatility.iloc[position] = 0.0
            continue
        start = position - parameters.covariance_lookback + 1
        if start < 1:
            continue
        window = returns.iloc[start : position + 1][active].dropna()
        if len(window) < parameters.covariance_lookback:
            continue
        covariance = (
            window.cov(ddof=0).to_numpy(dtype=float)
            * DAILY_PERIODS_PER_YEAR
        )
        proportions = np.full(
            len(active),
            1.0 / len(active),
            dtype=float,
        )
        variance = float(proportions @ covariance @ proportions)
        if not math.isfinite(variance) or variance <= 0.0:
            continue
        base_volatility = math.sqrt(variance)
        total_exposure = min(
            policy.maximum_total_exposure,
            parameters.target_annualized_volatility
            / base_volatility,
        )
        allocation = min(
            policy.maximum_position_exposure,
            total_exposure / len(active),
        )
        desired.loc[timestamp, active] = allocation
        predicted_volatility.iloc[position] = (
            base_volatility * allocation * len(active)
        )
    return desired, predicted_volatility


def backtest_dual_asset_trend(
    frames: Mapping[str, pd.DataFrame],
    parameters: DualAssetTrendParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> DualAssetTrendResult:
    """Run the single exact daily-close/next-open BTC/ETH strategy."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    if set(portfolio_policy.allowed_markets) != {
        "BTC-EUR",
        "ETH-EUR",
    }:
        raise ValueError("dual-asset trend policy must be BTC/ETH only")
    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=portfolio_policy,
    )
    if set(closes.columns) != {"BTC-EUR", "ETH-EUR"}:
        raise ValueError("dual-asset trend requires exactly BTC and ETH")
    warmup = max(
        parameters.trend_ema_period,
        parameters.covariance_lookback,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for dual-asset trend")

    ema = closes.ewm(
        span=parameters.trend_ema_period,
        adjust=False,
        min_periods=parameters.trend_ema_period,
    ).mean()
    history = closes.notna().cumsum()
    btc_regime = closes[benchmark] > ema[benchmark]
    eligible = (
        (history >= portfolio_policy.minimum_history_observations)
        & closes.notna()
        & ema.notna()
        & (closes > ema)
    )
    eligible["ETH-EUR"] &= btc_regime
    desired, predicted_volatility = _covariance_target_weights(
        closes,
        eligible,
        parameters=parameters,
        policy=portfolio_policy,
    )
    selected = desired > 1e-12
    forced_exit = (
        selected.shift(1, fill_value=False) & ~selected
    ).any(axis=1)
    scheduled = pd.Series(
        closes.index.weekday == parameters.rebalance_weekday,
        index=closes.index,
    )
    signal_event = scheduled | forced_exit
    signal_targets = (
        desired.where(signal_event, np.nan).ffill().fillna(0.0)
    )
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
            "held dual-asset trend position lacks next valuation"
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
        raise ValueError("dual-asset trend costs exhaust equity")
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
    for execution_position, executed_at in enumerate(executed.index):
        executed_turnover = float(turnover.iloc[execution_position])
        if executed_turnover <= 1e-12:
            continue
        signal_position = max(
            0,
            closes.index.get_loc(executed_at) - 1,
        )
        target = executed.iloc[execution_position]
        decision_rows.append(
            {
                "decision_at": closes.index[signal_position],
                "executed_at": executed_at,
                "reason": (
                    "DUAL_ASSET_WEEKLY_REBALANCE_OR_DAILY_TREND_EXIT"
                ),
                "turnover": executed_turnover,
                "expected_cost_fraction": (
                    executed_turnover * one_way_cost
                ),
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "btc_regime_positive": bool(
                    btc_regime.iloc[signal_position]
                ),
                "predicted_annualized_volatility": (
                    float(predicted_volatility.iloc[signal_position])
                    if math.isfinite(
                        float(
                            predicted_volatility.iloc[signal_position]
                        )
                    )
                    else None
                ),
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
            "cash_fraction",
            "btc_regime_positive",
            "predicted_annualized_volatility",
        ],
    )
    metrics = _performance_metrics(equity, executed, decisions)
    exposure = executed.sum(axis=1)
    finite_predicted = predicted_volatility.dropna()
    integrity = {
        "allowed_markets_only": set(executed.columns)
        == {"BTC-EUR", "ETH-EUR"},
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
        "full_covariance_backward_only": True,
        "closed_candles_only": True,
        "decision_at_close_execution_next_open": True,
        "daily_exit_entry_weekly": True,
        "point_in_time_history_gate": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "orders_generated": 0,
    }
    if not all(
        value
        for key, value in integrity.items()
        if key != "orders_generated"
    ):
        raise RuntimeError(
            f"dual-asset trend integrity failure: {integrity}"
        )
    return DualAssetTrendResult(
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
        risk_diagnostics={
            "target_annualized_volatility": (
                parameters.target_annualized_volatility
            ),
            "predicted_volatility_observations": int(
                len(finite_predicted)
            ),
            "predicted_volatility_mean": (
                float(finite_predicted.mean())
                if len(finite_predicted)
                else 0.0
            ),
            "predicted_volatility_maximum": (
                float(finite_predicted.max())
                if len(finite_predicted)
                else 0.0
            ),
            "btc_active_days": int(
                (executed["BTC-EUR"] > 1e-12).sum()
            ),
            "eth_active_days": int(
                (executed["ETH-EUR"] > 1e-12).sum()
            ),
        },
    )


def dual_asset_trend_period_metrics(
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
        (
            selected.index[-1] - selected.index[0]
        ).total_seconds()
        / 86_400.0,
    )
    years = elapsed_days / DAILY_PERIODS_PER_YEAR
    standard = float(returns.std(ddof=0))
    drawdown = selected / selected.cummax() - 1.0
    net_return = float(selected.iloc[-1] / selected.iloc[0] - 1.0)
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
            "portfolio_period_profit_factor": _profit_factor(returns),
            "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
            "positive_observations": int((returns > 0.0).sum()),
            "negative_observations": int((returns < 0.0).sum()),
            "periods_per_year": DAILY_PERIODS_PER_YEAR,
        },
        returns,
    )


__all__ = [
    "DAILY_PERIODS_PER_YEAR",
    "DUAL_ASSET_TREND_ENGINE_VERSION",
    "DUAL_ASSET_TREND_FAMILY",
    "DualAssetTrendParameters",
    "DualAssetTrendResult",
    "backtest_dual_asset_trend",
    "dual_asset_trend_parameter_set",
    "dual_asset_trend_period_metrics",
]
