"""Causal BTC-core and beta-residual satellite momentum research.

The portfolio holds a fixed 20% BTC core only in a long-term BTC uptrend and
may add one 20% altcoin satellite when its return in excess of a rolling,
backward-looking BTC beta is positive. Signals use completed daily closes,
rebalance weekly and become executable only at the following open.
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
    _profit_factor,
    _validated_panel,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

RESIDUAL_MOMENTUM_ENGINE_VERSION = "1.0.0"
RESIDUAL_MOMENTUM_FAMILY = (
    "BTC_TREND_CORE_BETA_RESIDUAL_MOMENTUM_SATELLITE"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class ResidualMomentumParameters:
    """Immutable DNA for one predeclared residual-momentum path."""

    residual_lookback: int
    beta_lookback: int
    asset_ema_period: int
    btc_ema_period: int = 200
    rebalance_weekday: int = 6
    core_weight: float = 0.20
    satellite_weight: float = 0.20
    maximum_positions: int = 2

    def __post_init__(self) -> None:
        if self.residual_lookback not in {20, 60}:
            raise ValueError("undeclared residual-momentum lookback")
        if self.beta_lookback not in {90, 180}:
            raise ValueError("undeclared rolling-beta lookback")
        if self.asset_ema_period not in {100, 200}:
            raise ValueError("undeclared satellite trend EMA")
        if self.btc_ema_period != 200:
            raise ValueError("v1 BTC trend EMA is fixed at 200")
        if self.rebalance_weekday != 6:
            raise ValueError("v1 rebalance weekday is fixed at Sunday")
        if self.core_weight != 0.20:
            raise ValueError("v1 BTC core weight is fixed at 20%")
        if self.satellite_weight != 0.20:
            raise ValueError("v1 satellite weight is fixed at 20%")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": RESIDUAL_MOMENTUM_FAMILY,
                "engine_version": RESIDUAL_MOMENTUM_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def residual_momentum_parameter_set(
) -> tuple[ResidualMomentumParameters, ...]:
    """Return the exact eight-trial preregistered family."""

    rows = tuple(
        ResidualMomentumParameters(
            residual_lookback=residual_lookback,
            beta_lookback=beta_lookback,
            asset_ema_period=asset_ema_period,
        )
        for residual_lookback, beta_lookback, asset_ema_period in product(
            (20, 60),
            (90, 180),
            (100, 200),
        )
    )
    if len(rows) != 8:
        raise RuntimeError("residual-momentum family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("residual-momentum family contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class ResidualMomentumResult:
    parameters: ResidualMomentumParameters
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
            "strategy_family": RESIDUAL_MOMENTUM_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": RESIDUAL_MOMENTUM_ENGINE_VERSION,
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
            "signal_diagnostics": dict(self.signal_diagnostics),
        }


def _rolling_betas(
    log_returns: pd.DataFrame,
    *,
    benchmark: str,
    lookback: int,
) -> pd.DataFrame:
    benchmark_returns = log_returns[benchmark]
    variance = benchmark_returns.rolling(
        lookback,
        min_periods=lookback,
    ).var(ddof=0)
    betas = pd.DataFrame(
        np.nan,
        index=log_returns.index,
        columns=log_returns.columns,
    )
    for market in log_returns.columns:
        if market == benchmark:
            betas[market] = 1.0
            continue
        covariance = log_returns[market].rolling(
            lookback,
            min_periods=lookback,
        ).cov(benchmark_returns, ddof=0)
        betas[market] = covariance.div(
            variance.replace(0.0, np.nan)
        )
    return betas.replace([np.inf, -np.inf], np.nan)


def _desired_weights(
    *,
    closes: pd.DataFrame,
    residual_score: pd.DataFrame,
    satellite_eligible: pd.DataFrame,
    btc_regime: pd.Series,
    parameters: ResidualMomentumParameters,
    benchmark: str,
) -> tuple[pd.DataFrame, pd.Series]:
    desired = pd.DataFrame(
        0.0,
        index=closes.index,
        columns=closes.columns,
    )
    selected_satellite = pd.Series(
        None,
        index=closes.index,
        dtype="object",
        name="selected_satellite",
    )
    for timestamp in closes.index:
        if not bool(btc_regime.loc[timestamp]):
            continue
        desired.loc[timestamp, benchmark] = parameters.core_weight
        scores = residual_score.loc[timestamp].where(
            satellite_eligible.loc[timestamp]
        ).dropna()
        scores = scores.drop(labels=[benchmark], errors="ignore")
        if scores.empty:
            continue
        selected = str(
            scores.sort_values(
                ascending=False,
                kind="mergesort",
            ).index[0]
        )
        desired.loc[timestamp, selected] = (
            parameters.satellite_weight
        )
        selected_satellite.loc[timestamp] = selected
    return desired, selected_satellite


def backtest_residual_momentum(
    frames: Mapping[str, pd.DataFrame],
    parameters: ResidualMomentumParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> ResidualMomentumResult:
    """Run one exact weekly next-open BTC-core/residual-satellite path."""

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
    if len(closes.columns) < 3:
        raise ValueError(
            "residual momentum requires BTC and at least two satellites"
        )
    warmup = max(
        parameters.beta_lookback,
        parameters.residual_lookback,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for residual momentum")

    log_close = np.log(closes.where(closes > 0.0))
    log_returns = log_close.diff()
    beta = _rolling_betas(
        log_returns,
        benchmark=benchmark,
        lookback=parameters.beta_lookback,
    )
    asset_momentum = log_close.diff(parameters.residual_lookback)
    benchmark_momentum = asset_momentum[benchmark]
    residual_score = asset_momentum.sub(
        beta.mul(benchmark_momentum, axis=0)
    )
    residual_score[benchmark] = 0.0

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
    history = closes.notna().cumsum()
    valid = (
        (history >= portfolio_policy.minimum_history_observations)
        & closes.notna()
        & asset_ema.notna()
        & beta.notna()
        & residual_score.notna()
    )
    satellite_eligible = (
        valid
        & (closes > asset_ema)
        & (residual_score > 0.0)
    ).mul(btc_regime, axis=0)
    satellite_eligible[benchmark] = False
    desired, selected_satellite = _desired_weights(
        closes=closes,
        residual_score=residual_score,
        satellite_eligible=satellite_eligible,
        btc_regime=btc_regime,
        parameters=parameters,
        benchmark=benchmark,
    )

    scheduled = pd.Series(
        closes.index.weekday == parameters.rebalance_weekday,
        index=closes.index,
    )
    signal_targets = desired.where(scheduled, np.nan).ffill()
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
            "held residual-momentum asset lacks causal next valuation"
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
        raise ValueError("residual-momentum costs exhaust equity")
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
        satellite = selected_satellite.iloc[signal_position]
        decision_rows.append(
            {
                "decision_at": closes.index[signal_position],
                "executed_at": executed_at,
                "reason": "BTC_CORE_RESIDUAL_SATELLITE_WEEKLY",
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
                "selected_satellite": (
                    str(satellite)
                    if satellite is not None
                    and not pd.isna(satellite)
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
            "selected_satellite",
        ],
    )
    metrics = _performance_metrics(equity, executed, decisions)
    exposure = executed.sum(axis=1)
    integrity = {
        "allowed_markets_only": set(executed.columns)
        <= set(portfolio_policy.allowed_markets),
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
        "rolling_beta_backward_only": True,
        "closed_candles_only": True,
        "decision_at_close_execution_next_open": True,
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
            f"residual-momentum integrity failure: {integrity}"
        )
    satellite_counts = (
        decisions["selected_satellite"]
        .dropna()
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
        if not decisions.empty
        else {}
    )
    return ResidualMomentumResult(
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
        signal_diagnostics={
            "btc_core_active_days": int(
                (executed[benchmark] > 1e-12).sum()
            ),
            "satellite_active_days": int(
                (
                    executed.drop(columns=[benchmark]).sum(axis=1)
                    > 1e-12
                ).sum()
            ),
            "satellite_selection_counts": satellite_counts,
            "positive_residual_asset_days": int(
                satellite_eligible.sum().sum()
            ),
            "beta_observations": int(beta.notna().sum().sum()),
        },
    )


def residual_momentum_period_metrics(
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
    "RESIDUAL_MOMENTUM_ENGINE_VERSION",
    "RESIDUAL_MOMENTUM_FAMILY",
    "ResidualMomentumParameters",
    "ResidualMomentumResult",
    "backtest_residual_momentum",
    "residual_momentum_parameter_set",
    "residual_momentum_period_metrics",
]
