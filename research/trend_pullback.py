"""Causal long-only trend-filtered pullback portfolio research.

The family tests whether statistically unusual daily pullbacks mean-revert
while an asset and BTC remain in established uptrends. Signals are calculated
from completed daily closes, become executable only at the next open, and have
no order or promotion authority.
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
    _validated_panel,
)
from research.volatility_contraction import (
    _desired_weights,
    _holding_state,
    _performance_metrics,
)
from utils.common import stable_hash

TREND_PULLBACK_FAMILY = "MULTI_ASSET_TREND_FILTERED_PULLBACK"
TREND_PULLBACK_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class TrendPullbackParameters:
    """Immutable DNA for one predeclared pullback path."""

    zscore_lookback: int
    entry_zscore: float
    asset_ema_period: int
    exit_zscore: float = -0.25
    volatility_lookback: int = 20
    target_annualized_volatility: float = 0.10
    btc_ema_period: int = 200
    maximum_positions: int = 2
    rebalance_weekday: int = 6

    def __post_init__(self) -> None:
        if self.zscore_lookback not in {10, 20, 40}:
            raise ValueError("undeclared pullback z-score lookback")
        if self.entry_zscore not in {-1.5, -2.0}:
            raise ValueError("undeclared pullback entry z-score")
        if self.asset_ema_period not in {100, 200}:
            raise ValueError("undeclared pullback asset trend EMA")
        if self.exit_zscore != -0.25:
            raise ValueError("v1 pullback exit z-score is fixed")
        if self.volatility_lookback != 20:
            raise ValueError("v1 pullback volatility lookback is fixed")
        if self.target_annualized_volatility != 0.10:
            raise ValueError("v1 pullback volatility target is fixed")
        if self.btc_ema_period != 200:
            raise ValueError("v1 pullback BTC trend EMA is fixed")
        if self.maximum_positions != 2:
            raise ValueError("v1 pullback maximum positions is fixed")
        if self.rebalance_weekday != 6:
            raise ValueError("v1 pullback rebalance weekday is fixed")
        if self.entry_zscore >= self.exit_zscore:
            raise ValueError("entry z-score must be below exit z-score")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": TREND_PULLBACK_FAMILY,
                "engine_version": TREND_PULLBACK_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def trend_pullback_parameter_set(
) -> tuple[TrendPullbackParameters, ...]:
    """Return the exact 12-trial predeclared family."""

    rows = tuple(
        TrendPullbackParameters(
            zscore_lookback=lookback,
            entry_zscore=entry_zscore,
            asset_ema_period=asset_ema_period,
        )
        for lookback, entry_zscore, asset_ema_period in product(
            (10, 20, 40),
            (-1.5, -2.0),
            (100, 200),
        )
    )
    if len(rows) != 12:
        raise RuntimeError("pullback family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("pullback family contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class TrendPullbackResult:
    parameters: TrendPullbackParameters
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
            "strategy_family": TREND_PULLBACK_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": TREND_PULLBACK_ENGINE_VERSION,
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


def backtest_trend_pullback(
    frames: Mapping[str, pd.DataFrame],
    parameters: TrendPullbackParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> TrendPullbackResult:
    """Run one exact causal trend-filtered pullback path."""

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
        parameters.zscore_lookback,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        parameters.volatility_lookback,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for trend pullback")

    log_close = np.log(closes.where(closes > 0.0))
    rolling_mean = log_close.rolling(
        parameters.zscore_lookback,
        min_periods=parameters.zscore_lookback,
    ).mean()
    rolling_std = log_close.rolling(
        parameters.zscore_lookback,
        min_periods=parameters.zscore_lookback,
    ).std(ddof=0)
    zscore = (
        (log_close - rolling_mean)
        / rolling_std.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    daily_returns = closes.pct_change(fill_method=None)
    annualized_volatility = (
        daily_returns.rolling(
            parameters.volatility_lookback,
            min_periods=parameters.volatility_lookback,
        ).std(ddof=0)
        * math.sqrt(365.25)
    )
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
        & zscore.notna()
        & asset_ema.notna()
        & annualized_volatility.notna()
    )
    entries = (
        valid
        & (zscore <= parameters.entry_zscore)
        & (closes > asset_ema)
    ).mul(btc_regime, axis=0)
    regime_exit = pd.DataFrame(
        np.repeat(
            (~btc_regime).to_numpy()[:, None],
            len(closes.columns),
            axis=1,
        ),
        index=closes.index,
        columns=closes.columns,
    )
    exits = (
        ~valid
        | (zscore >= parameters.exit_zscore)
        | (closes <= asset_ema)
        | regime_exit
    )
    active = _holding_state(entries, exits)
    strength = (-zscore).replace([np.inf, -np.inf], np.nan)
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
    signal_targets = desired.where(signal_event, np.nan).ffill().fillna(0.0)
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
            "held pullback asset lacks causal next valuation"
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
        raise ValueError("pullback costs exhaust portfolio equity")
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
        if not bool(signal_event.iloc[decision_position]):
            continue
        execution_at = index[execution_position]
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
            and bool(prior_active.iloc[decision_position][market])
        ]
        reason = (
            "TREND_PULLBACK_ENTRY"
            if entry_assets
            else "MEAN_OR_REGIME_EXIT"
            if exit_assets
            else "SCHEDULED_RESIZE"
        )
        event_turnover = float(turnover.loc[execution_at])
        if execution_position == len(index) - 1:
            event_turnover -= terminal_turnover
        decisions.append(
            {
                "decision_at": index[decision_position],
                "executed_at": execution_at,
                "reason": reason,
                "scheduled": bool(scheduled.iloc[decision_position]),
                "turnover": max(0.0, event_turnover),
                "expected_cost_fraction": (
                    max(0.0, event_turnover) * one_way_cost
                ),
                "target_weights": {
                    market: float(value)
                    for market, value in target.items()
                    if float(value) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "entry_signals": entry_assets,
                "exit_signals": exit_assets,
                "entry_zscores": {
                    market: float(zscore.iloc[decision_position][market])
                    for market in entry_assets
                },
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
            "entry_zscores": {},
        }
    )
    decision_frame = pd.DataFrame(decisions)
    metrics = _performance_metrics(
        equity,
        executed,
        decision_frame,
    )
    signal_diagnostics = {
        "entry_signal_count": int(entries.sum().sum()),
        "exit_signal_count": int(
            (exits & prior_active).sum().sum()
        ),
        "active_asset_days": int(active.sum().sum()),
        "zscore_minimum": float(
            zscore.min(axis=1).min(skipna=True)
        ),
        "decision_events": int(signal_event.sum()),
    }
    integrity = {
        "closed_daily_signal_inputs": True,
        "zscore_uses_current_closed_candle_only": True,
        "decision_at_close_execution_next_open": bool(
            decision_frame.iloc[:-1].empty
            or (
                pd.to_datetime(
                    decision_frame.iloc[:-1]["executed_at"],
                    utc=True,
                )
                > pd.to_datetime(
                    decision_frame.iloc[:-1]["decision_at"],
                    utc=True,
                )
            ).all()
        ),
        "long_only": bool((executed >= -1e-12).all().all()),
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
        "orders_generated": 0,
    }
    return TrendPullbackResult(
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
        decisions=decision_frame,
        signal_diagnostics=signal_diagnostics,
    )


__all__ = [
    "TREND_PULLBACK_ENGINE_VERSION",
    "TREND_PULLBACK_FAMILY",
    "TrendPullbackParameters",
    "TrendPullbackResult",
    "backtest_trend_pullback",
    "trend_pullback_parameter_set",
]
