"""Causal peer-basket beta-residual reversal for liquid EUR spot assets.

Each altcoin is compared with the equal-weight return of the other liquid
altcoins.  Rolling betas and residual standardization use only observations
strictly before the signal close.  The most negative peer-relative shocks are
held long only while BTC remains above its causal EMA200.  This module has no
order-submission or promotion authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _validated_panel,
)
from research.residual_reversal import (
    DAILY_PERIODS_PER_YEAR,
    _stateful_targets,
    residual_reversal_period_metrics,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

PEER_RESIDUAL_REVERSAL_ENGINE_VERSION = "1.0.0"
PEER_RESIDUAL_REVERSAL_FAMILY = (
    "ALT_PEER_BASKET_BETA_RESIDUAL_MEAN_REVERSION"
)


@dataclass(frozen=True, slots=True)
class PeerResidualReversalParameters:
    """One immutable member of the preregistered peer-residual family."""

    beta_lookback: int
    residual_horizon: int
    entry_zscore: float = -2.0
    zscore_lookback: int = 90
    exit_zscore: float = -0.25
    btc_ema_period: int = 200
    maximum_holding_days: int = 10
    maximum_positions: int = 2
    position_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.beta_lookback not in {30, 60}:
            raise ValueError("peer beta lookback must be 30 or 60")
        if self.residual_horizon not in {3, 5}:
            raise ValueError("peer residual horizon must be 3 or 5")
        if self.entry_zscore != -2.0:
            raise ValueError("peer entry z-score is fixed at -2.0")
        if self.zscore_lookback != 90:
            raise ValueError("peer z-score lookback is fixed at 90")
        if self.exit_zscore != -0.25:
            raise ValueError("peer exit z-score is fixed at -0.25")
        if self.btc_ema_period != 200:
            raise ValueError("peer BTC EMA is fixed at 200")
        if self.maximum_holding_days != 10:
            raise ValueError("peer maximum hold is fixed at ten days")
        if self.maximum_positions != 2:
            raise ValueError("peer maximum positions is fixed at two")
        if self.position_weight != 0.20:
            raise ValueError("peer position weight is fixed at 20%")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": PEER_RESIDUAL_REVERSAL_FAMILY,
                "engine_version": (
                    PEER_RESIDUAL_REVERSAL_ENGINE_VERSION
                ),
                "parameters": asdict(self),
            },
            length=64,
        )


def peer_residual_reversal_parameter_set(
) -> tuple[PeerResidualReversalParameters, ...]:
    """Return the exact four preregistered v1 strategy DNA rows."""

    rows = tuple(
        PeerResidualReversalParameters(
            beta_lookback=beta,
            residual_horizon=horizon,
        )
        for beta, horizon in product((30, 60), (3, 5))
    )
    if len(rows) != 4 or len({row.dna_hash for row in rows}) != 4:
        raise RuntimeError("peer-residual DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class PeerResidualReversalResult:
    parameters: PeerResidualReversalParameters
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
            "strategy_family": PEER_RESIDUAL_REVERSAL_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": PEER_RESIDUAL_REVERSAL_ENGINE_VERSION,
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


def _peer_residual_zscores(
    closes: pd.DataFrame,
    *,
    benchmark: str,
    parameters: PeerResidualReversalParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly prior peer-basket betas and residual z-scores."""

    returns = np.log(closes.where(closes > 0.0)).diff()
    satellites = [market for market in closes if market != benchmark]
    if len(satellites) < 3:
        raise ValueError(
            "peer residual reversal requires at least three satellites"
        )
    betas = pd.DataFrame(
        np.nan,
        index=closes.index,
        columns=closes.columns,
    )
    residuals = betas.copy()
    for market in satellites:
        peers = [peer for peer in satellites if peer != market]
        peer_return = returns[peers].mean(axis=1, skipna=False)
        prior_variance = (
            peer_return.rolling(
                parameters.beta_lookback,
                min_periods=parameters.beta_lookback,
            )
            .var(ddof=0)
            .shift(1)
        )
        prior_covariance = (
            returns[market]
            .rolling(
                parameters.beta_lookback,
                min_periods=parameters.beta_lookback,
            )
            .cov(peer_return, ddof=0)
            .shift(1)
        )
        beta = prior_covariance.div(
            prior_variance.replace(0.0, np.nan)
        )
        betas[market] = beta
        residuals[market] = returns[market] - beta * peer_return
    shock = residuals.rolling(
        parameters.residual_horizon,
        min_periods=parameters.residual_horizon,
    ).sum()
    prior_mean = (
        shock.rolling(
            parameters.zscore_lookback,
            min_periods=parameters.zscore_lookback,
        )
        .mean()
        .shift(1)
    )
    prior_std = (
        shock.rolling(
            parameters.zscore_lookback,
            min_periods=parameters.zscore_lookback,
        )
        .std(ddof=0)
        .shift(1)
    )
    zscore = shock.sub(prior_mean).div(
        prior_std.replace(0.0, np.nan)
    )
    betas[benchmark] = np.nan
    zscore[benchmark] = np.nan
    return (
        betas.replace([np.inf, -np.inf], np.nan),
        zscore.replace([np.inf, -np.inf], np.nan),
    )


def backtest_peer_residual_reversal(
    frames: Mapping[str, pd.DataFrame],
    parameters: PeerResidualReversalParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> PeerResidualReversalResult:
    """Run one exact daily next-open peer-residual path."""

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
        parameters.beta_lookback
        + parameters.residual_horizon
        + parameters.zscore_lookback,
        parameters.btc_ema_period,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for peer residual reversal")
    betas, zscore = _peer_residual_zscores(
        closes,
        benchmark=benchmark,
        parameters=parameters,
    )
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
        & betas.notna()
        & zscore.notna()
    )
    valid[benchmark] = False
    desired, signal_events = _stateful_targets(
        closes=closes,
        zscore=zscore,
        valid=valid,
        btc_regime=btc_regime,
        benchmark=benchmark,
        parameters=parameters,
    )
    executed = desired.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)
    executed = executed.iloc[warmup + 1 :].copy()
    if executed.empty:
        raise ValueError("peer-residual evaluation window is empty")
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
            "held peer-residual asset lacks causal next valuation"
        )
    gross_returns = (
        executed * open_returns.fillna(0.0)
    ).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].sum())
    turnover.iloc[-1] += float(executed.iloc[-1].sum())
    one_way_cost = (
        fee_rate
        + slippage_bps / 10_000.0
        + spread_bps / 20_000.0
    )
    cost_fraction = turnover * one_way_cost
    if bool((cost_fraction >= 1.0).any()):
        raise ValueError(
            "peer-residual transaction costs consume full capital"
        )
    net_returns = (
        (1.0 - cost_fraction) * (1.0 + gross_returns) - 1.0
    )
    equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()
    equity.name = "equity"
    gross_equity.name = "gross_equity"
    decision_rows: list[dict[str, Any]] = []
    for position, executed_at in enumerate(executed.index):
        execution_turnover = float(turnover.iloc[position])
        if execution_turnover <= 1e-12:
            continue
        signal_position = closes.index.get_loc(executed_at) - 1
        decision_at = closes.index[signal_position]
        target = executed.iloc[position]
        matching = signal_events.loc[
            signal_events["decision_at"] == decision_at
        ]
        event = matching.iloc[-1] if not matching.empty else None
        decision_rows.append(
            {
                "decision_at": decision_at,
                "executed_at": executed_at,
                "reason": "PEER_BETA_RESIDUAL_REVERSAL_DAILY",
                "turnover": execution_turnover,
                "expected_cost_fraction": (
                    execution_turnover * one_way_cost
                ),
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "entry_markets": (
                    list(event["entry_markets"])
                    if event is not None
                    else []
                ),
                "exit_reasons": (
                    dict(event["exit_reasons"])
                    if event is not None
                    else {}
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
            "entry_markets",
            "exit_reasons",
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
        "strictly_prior_peer_beta_estimation": True,
        "strictly_prior_zscore_baseline": True,
        "btc_used_as_regime_only": bool(
            (executed[benchmark].abs() <= 1e-12).all()
        ),
        "decision_at_close_execution_next_open": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "orders_generated": 0,
    }
    if (
        not all(
            bool(value)
            for key, value in integrity.items()
            if key != "orders_generated"
        )
        or int(integrity["orders_generated"]) != 0
    ):
        raise RuntimeError("peer-residual integrity failure")
    return PeerResidualReversalResult(
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
            "entry_signal_count": int(
                signal_events["entry_markets"].map(len).sum()
                if not signal_events.empty
                else 0
            ),
            "decision_count": int(len(decisions)),
            "satellite_count": len(closes.columns) - 1,
            "peer_basket_size": len(closes.columns) - 2,
        },
    )


__all__ = [
    "PEER_RESIDUAL_REVERSAL_ENGINE_VERSION",
    "PEER_RESIDUAL_REVERSAL_FAMILY",
    "PeerResidualReversalParameters",
    "PeerResidualReversalResult",
    "backtest_peer_residual_reversal",
    "peer_residual_reversal_parameter_set",
    "residual_reversal_period_metrics",
]
