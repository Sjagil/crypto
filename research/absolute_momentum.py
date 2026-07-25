"""Causal long-only absolute-momentum portfolio research.

The family differs from cross-sectional rotation: every asset independently
earns eligibility from its own time-series momentum and trend. No relative
top-N selection is performed. Signals use completed closes, execute at the
next open and remain bounded by an explicit volatility budget and spot-only
portfolio policy.
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
    _effective_sample_size,
    _profit_factor,
    _validated_panel,
)
from utils.common import stable_hash

ABSOLUTE_MOMENTUM_ENGINE_VERSION = "1.0.0"
ABSOLUTE_MOMENTUM_FAMILY = "MULTI_ASSET_ABSOLUTE_MOMENTUM_VOL_TARGET"
ABSOLUTE_MOMENTUM_PLATEAU_FAMILY = (
    "MULTI_ASSET_ABSOLUTE_MOMENTUM_GAUSSIAN_PLATEAU"
)
ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class AbsoluteMomentumParameters:
    """Immutable DNA for one absolute-momentum risk-budget path."""

    momentum_lookbacks: tuple[int, ...] = (20, 60, 120)
    minimum_positive_horizons: int = 2
    asset_ema_period: int = 200
    btc_ema_period: int = 200
    volatility_lookback: int = 60
    target_annualized_volatility: float = 0.05
    rebalance_weekday: int = 6
    weighting: str = "inverse_volatility"

    def __post_init__(self) -> None:
        if len(self.momentum_lookbacks) < 2:
            raise ValueError("absolute momentum requires at least two horizons")
        if tuple(sorted(set(self.momentum_lookbacks))) != self.momentum_lookbacks:
            raise ValueError("momentum lookbacks must be sorted and unique")
        if min(self.momentum_lookbacks) < 5:
            raise ValueError("momentum lookbacks must be at least five days")
        if not 1 <= self.minimum_positive_horizons <= len(self.momentum_lookbacks):
            raise ValueError("invalid positive-horizon requirement")
        if min(
            self.asset_ema_period,
            self.btc_ema_period,
            self.volatility_lookback,
        ) < 20:
            raise ValueError("trend and volatility windows must be at least 20")
        if not 0.0 < self.target_annualized_volatility <= 0.20:
            raise ValueError("volatility target must be in (0, 20%]")
        if self.rebalance_weekday not in range(7):
            raise ValueError("rebalance weekday must be in [0, 6]")
        if self.weighting != "inverse_volatility":
            raise ValueError("v1 supports only inverse-volatility weighting")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": ABSOLUTE_MOMENTUM_FAMILY,
                "engine_version": ABSOLUTE_MOMENTUM_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def absolute_momentum_parameter_set() -> tuple[AbsoluteMomentumParameters, ...]:
    """Return the five fully accounted risk-budget variants."""

    rows = tuple(
        AbsoluteMomentumParameters(target_annualized_volatility=target)
        for target in (0.04, 0.05, 0.06, 0.08, 0.10)
    )
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("absolute-momentum parameter set contains duplicate DNA")
    return rows


@dataclass(frozen=True, slots=True)
class AbsoluteMomentumPlateauDNA:
    """Independent candidate ID for a predeclared horizon plateau path."""

    horizon_shift: int
    volatility_lookback: int
    target_annualized_volatility: float
    base_momentum_lookbacks: tuple[int, ...] = (20, 60, 120)
    minimum_positive_horizons: int = 2
    asset_ema_period: int = 200
    btc_ema_period: int = 200
    rebalance_weekday: int = 6

    def __post_init__(self) -> None:
        shifted = self.momentum_lookbacks
        if min(shifted) < 5:
            raise ValueError("plateau shift creates an invalid horizon")
        if self.volatility_lookback not in {40, 60, 90}:
            raise ValueError("undeclared plateau volatility lookback")
        if self.target_annualized_volatility not in {
            0.04,
            0.05,
            0.06,
        }:
            raise ValueError("undeclared plateau volatility target")

    @property
    def momentum_lookbacks(self) -> tuple[int, ...]:
        return tuple(
            value + self.horizon_shift
            for value in self.base_momentum_lookbacks
        )

    @property
    def parameters(self) -> AbsoluteMomentumParameters:
        return AbsoluteMomentumParameters(
            momentum_lookbacks=self.momentum_lookbacks,
            minimum_positive_horizons=(
                self.minimum_positive_horizons
            ),
            asset_ema_period=self.asset_ema_period,
            btc_ema_period=self.btc_ema_period,
            volatility_lookback=self.volatility_lookback,
            target_annualized_volatility=(
                self.target_annualized_volatility
            ),
            rebalance_weekday=self.rebalance_weekday,
        )

    @property
    def nuisance_group(self) -> str:
        return stable_hash(
            {
                "volatility_lookback": self.volatility_lookback,
                "target_annualized_volatility": (
                    self.target_annualized_volatility
                ),
                "minimum_positive_horizons": (
                    self.minimum_positive_horizons
                ),
                "asset_ema_period": self.asset_ema_period,
                "btc_ema_period": self.btc_ema_period,
                "rebalance_weekday": self.rebalance_weekday,
            },
            length=32,
        )

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": ABSOLUTE_MOMENTUM_PLATEAU_FAMILY,
                "engine_version": (
                    ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION
                ),
                "parameters": asdict(self),
                "derived_momentum_lookbacks": (
                    self.momentum_lookbacks
                ),
            },
            length=64,
        )


def absolute_momentum_plateau_parameter_set(
) -> tuple[AbsoluteMomentumPlateauDNA, ...]:
    """Return all 117 predeclared plateau trials in deterministic order."""

    rows = tuple(
        AbsoluteMomentumPlateauDNA(
            horizon_shift=shift,
            volatility_lookback=volatility_lookback,
            target_annualized_volatility=target,
        )
        for shift, volatility_lookback, target in product(
            range(-6, 7),
            (40, 60, 90),
            (0.04, 0.05, 0.06),
        )
    )
    if len(rows) != 117:
        raise RuntimeError("plateau parameter set cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("plateau parameter set contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class AbsoluteMomentumResult:
    parameters: AbsoluteMomentumParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": ABSOLUTE_MOMENTUM_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": ABSOLUTE_MOMENTUM_ENGINE_VERSION,
            "parameters": asdict(self.parameters),
            "portfolio_policy": asdict(self.portfolio_policy),
            "portfolio_policy_hash": self.portfolio_policy.policy_hash,
            "execution_identity": stable_hash(
                {
                    "strategy_dna_hash": self.parameters.dna_hash,
                    "portfolio_policy_hash": self.portfolio_policy.policy_hash,
                },
                length=64,
            ),
            "metrics": dict(self.metrics),
            "integrity": dict(self.integrity),
            "cost_breakdown": dict(self.cost_breakdown),
        }


def _metrics(
    equity: pd.Series,
    weights: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    returns = equity.pct_change(fill_method=None).dropna()
    elapsed_days = max(1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0)
    years = elapsed_days / 365.25
    standard = float(returns.std(ddof=0))
    downside = float(returns[returns < 0].std(ddof=0))
    drawdown = equity / equity.cummax() - 1.0
    effective_sample, lag_one = _effective_sample_size(returns)
    exposure = weights.sum(axis=1)
    return {
        "net_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annualized_return": float(
            (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        ),
        "annualized_volatility": standard * math.sqrt(365.25),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(365.25))
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(returns.mean() / downside * math.sqrt(365.25))
            if downside > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "portfolio_period_effective_sample_size": effective_sample,
        "lag_one_autocorrelation": lag_one,
        "rebalance_count": int(len(decisions)),
        "average_exposure": float(exposure.mean()),
        "maximum_realized_exposure": float(exposure.max()),
        "cash_fraction_average": float(1.0 - exposure.mean()),
        "observations": int(len(returns)),
    }


def backtest_absolute_momentum(
    frames: Mapping[str, pd.DataFrame],
    parameters: AbsoluteMomentumParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> AbsoluteMomentumResult:
    """Run one exact next-open absolute-momentum portfolio path."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark_market,
        portfolio_policy=portfolio_policy,
    )
    history = closes.notna().cumsum()
    positive_votes = sum(
        closes.div(closes.shift(lookback)).sub(1.0).gt(0.0).astype(float)
        for lookback in parameters.momentum_lookbacks
    )
    asset_ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    btc_close = closes[benchmark]
    btc_ema = btc_close.ewm(
        span=parameters.btc_ema_period,
        adjust=False,
        min_periods=parameters.btc_ema_period,
    ).mean()
    btc_regime = btc_close > btc_ema
    eligible = (
        (history >= portfolio_policy.minimum_history_observations)
        & (closes > asset_ema)
        & (positive_votes >= parameters.minimum_positive_horizons)
    ).mul(btc_regime, axis=0)
    annualized_volatility = (
        closes.pct_change(fill_method=None)
        .rolling(
            parameters.volatility_lookback,
            min_periods=max(20, parameters.volatility_lookback * 2 // 3),
        )
        .std(ddof=0)
        * math.sqrt(365.25)
    )
    inverse_volatility = eligible.astype(float).div(
        annualized_volatility.replace(0.0, np.nan)
    )
    base_weights = inverse_volatility.div(
        inverse_volatility.sum(axis=1).replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)
    diagonal_portfolio_volatility = np.sqrt(
        ((base_weights * annualized_volatility) ** 2).sum(axis=1)
    ).replace(0.0, np.nan)
    gross_exposure = (
        parameters.target_annualized_volatility / diagonal_portfolio_volatility
    ).clip(upper=portfolio_policy.maximum_total_exposure).fillna(0.0)
    desired = base_weights.mul(gross_exposure, axis=0).clip(
        upper=portfolio_policy.maximum_position_exposure
    )
    desired = desired.mul(
        (
            portfolio_policy.maximum_total_exposure
            / desired.sum(axis=1).replace(0.0, np.nan)
        )
        .clip(upper=1.0)
        .fillna(0.0),
        axis=0,
    )

    signal_targets = pd.DataFrame(np.nan, index=opens.index, columns=opens.columns)
    rebalance_mask = opens.index.weekday == parameters.rebalance_weekday
    signal_targets.loc[rebalance_mask] = desired.loc[rebalance_mask]
    executed_weights = signal_targets.ffill().shift(1).fillna(0.0)
    executed_weights = executed_weights.where(opens.notna(), 0.0)
    turnover = executed_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed_weights.iloc[0].abs().sum())
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0

    forward_returns = opens.shift(-1).div(opens).sub(1.0)
    forward_returns.iloc[-1] = closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    held_and_missing = (executed_weights > 1e-12) & forward_returns.isna()
    if held_and_missing.any().any():
        raise ValueError("absolute momentum cannot value a held asset")
    gross_returns = (executed_weights * forward_returns.fillna(0.0)).sum(axis=1)
    net_factors = (1.0 - turnover * one_way_cost) * (1.0 + gross_returns)
    terminal_liquidation = float(executed_weights.iloc[-1].sum()) * one_way_cost
    net_factors.iloc[-1] *= 1.0 - terminal_liquidation
    gross_equity = pd.Series(
        np.r_[1.0, np.cumprod(1.0 + gross_returns.to_numpy(dtype=float))],
        index=pd.DatetimeIndex(
            [opens.index[0] - pd.DateOffset(days=1), *list(opens.index)]
        ),
        name="gross_equity",
    )
    equity = pd.Series(
        np.r_[1.0, np.cumprod(net_factors.to_numpy(dtype=float))],
        index=gross_equity.index,
        name="equity",
    )
    decision_rows: list[dict[str, Any]] = []
    for execution_position, executed_at in enumerate(opens.index):
        executed_turnover = float(turnover.iloc[execution_position])
        if executed_turnover <= 1e-12:
            continue
        signal_position = max(0, execution_position - 1)
        target = executed_weights.iloc[execution_position]
        decision_rows.append(
            {
                "decision_at": opens.index[signal_position],
                "executed_at": executed_at,
                "reason": "ABSOLUTE_MOMENTUM_WEEKLY_REBALANCE",
                "turnover": executed_turnover,
                "expected_cost_fraction": executed_turnover * one_way_cost,
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "btc_regime_positive": bool(
                    btc_regime.iloc[signal_position]
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
        ],
    )
    metrics = _metrics(
        equity,
        executed_weights.reindex(equity.index, method="bfill").fillna(0.0),
        decisions,
    )
    maximum_exposure = float(executed_weights.sum(axis=1).max())
    maximum_position = float(executed_weights.max(axis=1).max())
    minimum_cash = float((1.0 - executed_weights.sum(axis=1)).min())
    integrity = {
        "allowed_markets_only": set(executed_weights.columns)
        <= set(portfolio_policy.allowed_markets),
        "maximum_exposure_respected": (
            maximum_exposure <= portfolio_policy.maximum_total_exposure + 1e-12
        ),
        "maximum_position_exposure_respected": (
            maximum_position <= portfolio_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": minimum_cash >= portfolio_policy.minimum_cash - 1e-12,
        "closed_candles_only": True,
        "next_open_execution": True,
        "point_in_time_history_gate": True,
        "long_only_spot": bool((executed_weights >= -1e-12).all().all()),
        "orders_generated": 0,
    }
    if not all(
        value
        for key, value in integrity.items()
        if key != "orders_generated"
    ):
        raise RuntimeError(f"absolute-momentum integrity failure: {integrity}")
    return AbsoluteMomentumResult(
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
            "gross_ending_equity": float(gross_equity.iloc[-1]),
            "net_ending_equity": float(equity.iloc[-1]),
            "total_cost_drag": float(gross_equity.iloc[-1] - equity.iloc[-1]),
        },
        equity_curve=equity,
        gross_equity_curve=gross_equity,
        executed_weights=executed_weights,
        decisions=decisions,
    )


__all__ = [
    "ABSOLUTE_MOMENTUM_ENGINE_VERSION",
    "ABSOLUTE_MOMENTUM_FAMILY",
    "ABSOLUTE_MOMENTUM_PLATEAU_ENGINE_VERSION",
    "ABSOLUTE_MOMENTUM_PLATEAU_FAMILY",
    "AbsoluteMomentumParameters",
    "AbsoluteMomentumPlateauDNA",
    "AbsoluteMomentumResult",
    "absolute_momentum_plateau_parameter_set",
    "absolute_momentum_parameter_set",
    "backtest_absolute_momentum",
]
