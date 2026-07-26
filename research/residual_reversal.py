"""Causal beta-residual mean-reversion portfolio research.

This classical family tests whether unusually negative, BTC-beta-adjusted
returns of liquid EUR spot assets mean-revert while BTC remains in a long-term
uptrend. Every factor estimate is backward-looking, signals form only after a
daily close, and target weights become executable at the next daily open.
This module has no order-submission or promotion authority.
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

RESIDUAL_REVERSAL_ENGINE_VERSION = "1.0.0"
RESIDUAL_REVERSAL_FAMILY = (
    "BTC_REGIME_BETA_RESIDUAL_MEAN_REVERSION"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class ResidualReversalParameters:
    """One immutable member of the preregistered v1 family."""

    beta_lookback: int
    residual_horizon: int
    entry_zscore: float
    zscore_lookback: int = 90
    exit_zscore: float = -0.25
    btc_ema_period: int = 200
    maximum_holding_days: int = 10
    maximum_positions: int = 2
    position_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.beta_lookback not in {30, 60}:
            raise ValueError("v1 beta lookback must be 30 or 60")
        if self.residual_horizon not in {3, 5}:
            raise ValueError("v1 residual horizon must be 3 or 5")
        if self.entry_zscore not in {-1.5, -2.0}:
            raise ValueError("v1 entry z-score must be -1.5 or -2.0")
        if self.zscore_lookback != 90:
            raise ValueError("v1 z-score lookback is fixed at 90")
        if self.exit_zscore != -0.25:
            raise ValueError("v1 exit z-score is fixed at -0.25")
        if self.btc_ema_period != 200:
            raise ValueError("v1 BTC EMA is fixed at 200")
        if self.maximum_holding_days != 10:
            raise ValueError("v1 maximum hold is fixed at ten days")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")
        if self.position_weight != 0.20:
            raise ValueError("v1 position weight is fixed at 20%")
        if self.entry_zscore >= self.exit_zscore:
            raise ValueError("entry z-score must be below exit z-score")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": RESIDUAL_REVERSAL_FAMILY,
                "engine_version": RESIDUAL_REVERSAL_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def residual_reversal_parameter_set(
) -> tuple[ResidualReversalParameters, ...]:
    """Return the exact eight preregistered v1 strategy DNA rows."""

    rows = tuple(
        ResidualReversalParameters(
            beta_lookback=beta,
            residual_horizon=horizon,
            entry_zscore=entry,
        )
        for beta, horizon, entry in product(
            (30, 60),
            (3, 5),
            (-1.5, -2.0),
        )
    )
    if len(rows) != 8 or len({row.dna_hash for row in rows}) != 8:
        raise RuntimeError("residual-reversal DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class ResidualReversalResult:
    parameters: ResidualReversalParameters
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
            "strategy_family": RESIDUAL_REVERSAL_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": RESIDUAL_REVERSAL_ENGINE_VERSION,
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


def _strictly_prior_rolling_betas(
    log_returns: pd.DataFrame,
    *,
    benchmark: str,
    lookback: int,
) -> pd.DataFrame:
    """Estimate rolling betas using observations strictly before each signal."""

    benchmark_returns = log_returns[benchmark]
    variance = (
        benchmark_returns.rolling(
            lookback,
            min_periods=lookback,
        )
        .var(ddof=0)
        .shift(1)
    )
    betas = pd.DataFrame(
        np.nan,
        index=log_returns.index,
        columns=log_returns.columns,
    )
    for market in log_returns.columns:
        if market == benchmark:
            betas[market] = 1.0
            continue
        covariance = (
            log_returns[market]
            .rolling(
                lookback,
                min_periods=lookback,
            )
            .cov(benchmark_returns, ddof=0)
            .shift(1)
        )
        betas[market] = covariance.div(
            variance.replace(0.0, np.nan)
        )
    return betas.replace([np.inf, -np.inf], np.nan)


def _residual_zscores(
    closes: pd.DataFrame,
    *,
    benchmark: str,
    parameters: ResidualReversalParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly backward-estimated beta residuals and z-scores."""

    log_returns = np.log(closes.where(closes > 0.0)).diff()
    betas = _strictly_prior_rolling_betas(
        log_returns,
        benchmark=benchmark,
        lookback=parameters.beta_lookback,
    )
    one_bar_residual = log_returns.sub(
        betas.mul(log_returns[benchmark], axis=0)
    )
    one_bar_residual[benchmark] = np.nan
    residual_shock = one_bar_residual.rolling(
        parameters.residual_horizon,
        min_periods=parameters.residual_horizon,
    ).sum()
    prior_mean = (
        residual_shock.rolling(
            parameters.zscore_lookback,
            min_periods=parameters.zscore_lookback,
        )
        .mean()
        .shift(1)
    )
    prior_std = (
        residual_shock.rolling(
            parameters.zscore_lookback,
            min_periods=parameters.zscore_lookback,
        )
        .std(ddof=0)
        .shift(1)
    )
    zscore = residual_shock.sub(prior_mean).div(
        prior_std.replace(0.0, np.nan)
    )
    return betas, zscore.replace([np.inf, -np.inf], np.nan)


def _stateful_targets(
    *,
    closes: pd.DataFrame,
    zscore: pd.DataFrame,
    valid: pd.DataFrame,
    btc_regime: pd.Series,
    benchmark: str,
    parameters: ResidualReversalParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create close-time targets with deterministic holding-state exits."""

    desired = pd.DataFrame(
        0.0,
        index=closes.index,
        columns=closes.columns,
    )
    event_rows: list[dict[str, Any]] = []
    holdings: dict[str, int] = {}
    for timestamp in closes.index:
        regime_positive = bool(btc_regime.loc[timestamp])
        exits: dict[str, str] = {}
        if not regime_positive:
            exits = {
                market: "BTC_REGIME_EXIT"
                for market in holdings
            }
            holdings.clear()
        else:
            for market in tuple(holdings):
                score = zscore.at[timestamp, market]
                holdings[market] += 1
                if pd.isna(score):
                    exits[market] = "MISSING_FACTOR_EXIT"
                elif float(score) >= parameters.exit_zscore:
                    exits[market] = "RESIDUAL_NORMALIZATION_EXIT"
                elif (
                    holdings[market]
                    >= parameters.maximum_holding_days
                ):
                    exits[market] = "MAXIMUM_HOLD_EXIT"
                if market in exits:
                    holdings.pop(market)

        entries: list[str] = []
        if regime_positive and len(holdings) < parameters.maximum_positions:
            eligible = zscore.loc[timestamp].where(
                valid.loc[timestamp]
            ).dropna()
            eligible = eligible.drop(
                labels=[benchmark, *holdings],
                errors="ignore",
            )
            eligible = eligible[
                eligible <= parameters.entry_zscore
            ].sort_values(ascending=True, kind="mergesort")
            slots = parameters.maximum_positions - len(holdings)
            for market in eligible.index[:slots]:
                selected = str(market)
                holdings[selected] = 0
                entries.append(selected)

        for market in holdings:
            desired.at[timestamp, market] = parameters.position_weight
        if entries or exits:
            event_rows.append(
                {
                    "decision_at": timestamp,
                    "entry_markets": entries,
                    "exit_reasons": exits,
                    "btc_regime_positive": regime_positive,
                    "target_weights": {
                        market: float(weight)
                        for market, weight in desired.loc[
                            timestamp
                        ].items()
                        if float(weight) > 1e-12
                    },
                    "cash_fraction": float(
                        1.0 - desired.loc[timestamp].sum()
                    ),
                }
            )
    events = pd.DataFrame(
        event_rows,
        columns=[
            "decision_at",
            "entry_markets",
            "exit_reasons",
            "btc_regime_positive",
            "target_weights",
            "cash_fraction",
        ],
    )
    return desired, events


def backtest_residual_reversal(
    frames: Mapping[str, pd.DataFrame],
    parameters: ResidualReversalParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> ResidualReversalResult:
    """Run one exact daily next-open residual-reversal path."""

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
            "residual reversal requires BTC and at least two satellites"
        )
    warmup = max(
        parameters.beta_lookback
        + parameters.residual_horizon
        + parameters.zscore_lookback,
        parameters.btc_ema_period,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for residual reversal")

    betas, zscore = _residual_zscores(
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
        raise ValueError("residual-reversal evaluation window is empty")

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
            "held residual-reversal asset lacks causal next valuation"
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
        raise ValueError("residual-reversal costs exhaust equity")
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
        signal_position = closes.index.get_loc(executed_at) - 1
        target = executed.iloc[execution_position]
        matching = signal_events.loc[
            signal_events["decision_at"]
            == closes.index[signal_position]
        ]
        event = matching.iloc[-1] if not matching.empty else None
        decision_rows.append(
            {
                "decision_at": closes.index[signal_position],
                "executed_at": executed_at,
                "reason": "BETA_RESIDUAL_REVERSAL_DAILY",
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
            "entry_markets",
            "exit_reasons",
            "btc_regime_positive",
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
        "strictly_prior_beta_estimation": True,
        "strictly_prior_zscore_baseline": True,
        "benchmark_never_traded": bool(
            (executed[benchmark].abs() <= 1e-12).all()
        ),
        "closed_candles_only": True,
        "decision_at_close_execution_next_open": True,
        "point_in_time_history_gate": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "orders_generated": 0,
    }
    if not all(
        bool(value)
        for key, value in integrity.items()
        if key != "orders_generated"
    ) or int(integrity["orders_generated"]) != 0:
        failed = [
            key for key, passed in integrity.items() if not passed
        ]
        raise RuntimeError(
            f"residual-reversal integrity failure: {failed}"
        )
    cost_breakdown = {
        "fee_rate": float(fee_rate),
        "slippage_bps": float(slippage_bps),
        "spread_bps": float(spread_bps),
        "one_way_cost_rate": float(one_way_cost),
        "turnover": float(turnover.sum()),
        "total_cost_fraction": float(cost_fraction.sum()),
        "gross_ending_equity": float(gross_equity.iloc[-1]),
        "net_ending_equity": float(equity.iloc[-1]),
    }
    signal_diagnostics = {
        "entry_signal_count": int(
            signal_events["entry_markets"].map(len).sum()
            if not signal_events.empty
            else 0
        ),
        "decision_count": int(len(decisions)),
        "beta_lookback": parameters.beta_lookback,
        "residual_horizon": parameters.residual_horizon,
        "zscore_lookback": parameters.zscore_lookback,
        "entry_zscore": parameters.entry_zscore,
    }
    return ResidualReversalResult(
        parameters=parameters,
        portfolio_policy=portfolio_policy,
        metrics=metrics,
        integrity=integrity,
        cost_breakdown=cost_breakdown,
        equity_curve=equity,
        gross_equity_curve=gross_equity,
        executed_weights=executed,
        decisions=decisions,
        signal_diagnostics=signal_diagnostics,
    )


def residual_reversal_period_metrics(
    equity_curve: pd.Series,
    *,
    start: str,
    end: str,
) -> tuple[dict[str, Any], pd.Series]:
    """Return exact period metrics and the aligned daily return path."""

    selected = equity_curve.loc[start:end]
    if len(selected) < 2:
        return (
            {
                "net_return": 0.0,
                "annualized_return": 0.0,
                "sharpe": 0.0,
                "maximum_drawdown": 0.0,
                "portfolio_period_profit_factor": 0.0,
                "positive_observations": 0,
                "negative_observations": 0,
                "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
                "periods_per_year": DAILY_PERIODS_PER_YEAR,
            },
            pd.Series(dtype=float),
        )
    returns = selected.pct_change(fill_method=None).fillna(0.0)
    standard = float(returns.std(ddof=0))
    years = max(
        (selected.index[-1] - selected.index[0]).days / 365.25,
        1.0 / 365.25,
    )
    net_return = float(selected.iloc[-1] / selected.iloc[0] - 1.0)
    annualized = (
        float((1.0 + net_return) ** (1.0 / years) - 1.0)
        if net_return > -1.0
        else -1.0
    )
    metrics = {
        "net_return": net_return,
        "annualized_return": annualized,
        "sharpe": (
            float(
                returns.mean()
                / standard
                * math.sqrt(DAILY_PERIODS_PER_YEAR)
            )
            if standard > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(
            (selected / selected.cummax() - 1.0).min()
        ),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "positive_observations": int((returns > 0.0).sum()),
        "negative_observations": int((returns < 0.0).sum()),
        "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
        "periods_per_year": DAILY_PERIODS_PER_YEAR,
    }
    return metrics, returns


__all__ = [
    "DAILY_PERIODS_PER_YEAR",
    "RESIDUAL_REVERSAL_ENGINE_VERSION",
    "RESIDUAL_REVERSAL_FAMILY",
    "ResidualReversalParameters",
    "ResidualReversalResult",
    "backtest_residual_reversal",
    "residual_reversal_parameter_set",
    "residual_reversal_period_metrics",
]
