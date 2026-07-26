"""Causal daily crypto sentiment-recovery basket research.

The family uses the externally timestamped Alternative.me Crypto Fear & Greed
Index as a secondary context signal. An entry requires a fresh recovery from a
predeclared extreme-fear threshold while BTC and the selected spot asset remain
above long-horizon trends. Signals formed from completed daily observations
execute at the next daily open. This module cannot submit or promote orders.
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

SENTIMENT_RECOVERY_FAMILY = (
    "EXTERNAL_SENTIMENT_RECOVERY_FIXED_SPOT_BASKET"
)
SENTIMENT_RECOVERY_ENGINE_VERSION = "1.0.0"
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class SentimentRecoveryParameters:
    """Immutable DNA for one of eight declared sentiment paths."""

    fear_threshold: int
    recovery_delta: int
    trend_ema_period: int
    recovery_lookback_days: int = 7
    greed_exit_threshold: int = 75
    maximum_holding_days: int = 60
    target_weight_per_asset: float = 0.20
    maximum_positions: int = 2

    def __post_init__(self) -> None:
        if self.fear_threshold not in {20, 25}:
            raise ValueError("undeclared fear threshold")
        if self.recovery_delta not in {5, 10}:
            raise ValueError("undeclared sentiment recovery delta")
        if self.trend_ema_period not in {100, 200}:
            raise ValueError("undeclared sentiment trend EMA")
        if self.recovery_lookback_days != 7:
            raise ValueError("recovery lookback must remain seven days")
        if self.greed_exit_threshold != 75:
            raise ValueError("greed exit threshold must remain 75")
        if self.maximum_holding_days != 60:
            raise ValueError("maximum holding period must remain 60 days")
        if self.target_weight_per_asset != 0.20:
            raise ValueError("per-asset target weight must remain 20%")
        if self.maximum_positions != 2:
            raise ValueError("maximum positions must remain two")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": SENTIMENT_RECOVERY_FAMILY,
                "engine_version": SENTIMENT_RECOVERY_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def sentiment_recovery_parameter_set(
) -> tuple[SentimentRecoveryParameters, ...]:
    """Return the exact eight-trial preregistered family."""

    rows = tuple(
        SentimentRecoveryParameters(
            fear_threshold=fear,
            recovery_delta=recovery,
            trend_ema_period=ema,
        )
        for fear, recovery, ema in product(
            (20, 25),
            (5, 10),
            (100, 200),
        )
    )
    if len(rows) != 8:
        raise RuntimeError("sentiment family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("sentiment family contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class SentimentRecoveryResult:
    parameters: SentimentRecoveryParameters
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
            "strategy_family": SENTIMENT_RECOVERY_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": SENTIMENT_RECOVERY_ENGINE_VERSION,
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


def _validated_sentiment(
    sentiment: pd.DataFrame,
    *,
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, Any]]:
    required = {
        "available_at",
        "observed_at",
        "point_in_time_status",
        "fear_greed",
        "provider",
    }
    missing = required - set(sentiment.columns)
    if missing:
        raise ValueError(
            f"sentiment source lacks required columns: {sorted(missing)}"
        )
    selected = sentiment.loc[
        :,
        [
            "available_at",
            "observed_at",
            "point_in_time_status",
            "fear_greed",
            "provider",
        ],
    ].copy()
    selected["available_at"] = pd.to_datetime(
        selected["available_at"],
        utc=True,
        errors="raise",
    )
    selected["observed_at"] = pd.to_datetime(
        selected["observed_at"],
        utc=True,
        errors="raise",
    )
    selected = selected.sort_values(
        ["available_at", "observed_at"],
        kind="mergesort",
    )
    duplicate_mask = selected["available_at"].duplicated(
        keep=False
    )
    duplicate_rows_removed = 0
    if bool(duplicate_mask.any()):
        duplicate_rows = selected.loc[duplicate_mask]
        conflicts = (
            duplicate_rows.groupby("available_at", sort=False)[
                ["fear_greed", "provider", "point_in_time_status"]
            ]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if bool(conflicts.any()):
            raise ValueError(
                "sentiment source contains conflicting duplicate "
                "availability"
            )
        before = len(selected)
        selected = selected.drop_duplicates(
            subset=["available_at"],
            keep="first",
        )
        duplicate_rows_removed = before - len(selected)
    values = selected["fear_greed"].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("sentiment source contains non-finite values")
    if bool(((values < 0.0) | (values > 100.0)).any()):
        raise ValueError("sentiment values must remain in [0, 100]")
    if set(selected["provider"].astype(str)) != {"alternative_me"}:
        raise ValueError("unexpected sentiment provider")
    if set(selected["point_in_time_status"].astype(str)) != {
        "SOURCE_DAILY_TIMESTAMP"
    }:
        raise ValueError("sentiment source lacks daily source timestamps")

    source = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(selected["available_at"]),
        name="fear_greed",
    )
    aligned = source.reindex(index, method="ffill")
    source_times = pd.Series(source.index, index=source.index).reindex(
        index,
        method="ffill",
    )
    valid = aligned.notna()
    if valid.any():
        aligned_index = pd.Series(index, index=index)
        if bool((source_times.loc[valid] > aligned_index.loc[valid]).any()):
            raise RuntimeError("sentiment was aligned from the future")
    retrieval_is_retrospective = bool(
        (
            selected["observed_at"].astype("int64")
            - selected["available_at"].astype("int64")
            > 86_400_000_000_000
        ).any()
    )
    metadata = {
        "provider": "alternative_me",
        "point_in_time_status": "SOURCE_DAILY_TIMESTAMP",
        "backward_only_alignment": True,
        "backfill_before_first_observation": False,
        "retrospective_provider_history": retrieval_is_retrospective,
        "historical_revision_vintages_available": False,
        "source_start": selected["available_at"].iloc[0].isoformat(),
        "source_end": selected["available_at"].iloc[-1].isoformat(),
        "source_observations": int(len(selected)),
        "exact_duplicate_snapshots_removed": int(
            duplicate_rows_removed
        ),
    }
    return aligned, metadata


def _holding_state(
    entry_events: pd.DataFrame,
    exits: pd.DataFrame,
    *,
    maximum_holding_days: int,
) -> pd.DataFrame:
    active = pd.DataFrame(
        False,
        index=entry_events.index,
        columns=entry_events.columns,
    )
    for column in entry_events.columns:
        holding = False
        age = 0
        for position in range(len(entry_events)):
            if holding and (
                bool(exits.iloc[position][column])
                or age >= maximum_holding_days
            ):
                holding = False
                age = 0
            if not holding and bool(entry_events.iloc[position][column]):
                holding = True
                age = 0
            active.iloc[position, active.columns.get_loc(column)] = holding
            if holding:
                age += 1
    return active


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
    effective_sample, lag_one = _effective_sample_size(returns)
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
        "annualized_volatility": (
            standard * math.sqrt(DAILY_PERIODS_PER_YEAR)
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
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(DAILY_PERIODS_PER_YEAR)
            )
            if downside_deviation > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "portfolio_period_effective_sample_size": effective_sample,
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
        "period_unit": "DAILY_PORTFOLIO_RETURN",
        "periods_per_year": DAILY_PERIODS_PER_YEAR,
    }


def sentiment_recovery_period_metrics(
    equity_curve: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, float | int | str], pd.Series]:
    """Return daily metrics for one frozen calendar period."""

    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    start_at = (
        start_at.tz_localize("UTC")
        if start_at.tzinfo is None
        else start_at.tz_convert("UTC")
    )
    end_at = (
        end_at.tz_localize("UTC")
        if end_at.tzinfo is None
        else end_at.tz_convert("UTC")
    )
    selected = equity_curve.loc[
        (equity_curve.index >= start_at)
        & (equity_curve.index <= end_at)
    ].astype(float)
    if len(selected) < 90:
        raise ValueError("sentiment evaluation period needs 90 days")
    normalized = selected / float(selected.iloc[0])
    returns = normalized.pct_change(fill_method=None).dropna()
    standard = float(returns.std(ddof=0))
    elapsed_days = max(
        1.0,
        (normalized.index[-1] - normalized.index[0]).total_seconds()
        / 86_400.0,
    )
    metrics: dict[str, float | int | str] = {
        "net_return": float(normalized.iloc[-1] - 1.0),
        "annualized_return": float(
            normalized.iloc[-1] ** (365.25 / elapsed_days) - 1.0
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
        "maximum_drawdown": float(
            (normalized / normalized.cummax() - 1.0).min()
        ),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
        "positive_observations": int((returns > 0.0).sum()),
        "negative_observations": int((returns < 0.0).sum()),
        "periods_per_year": DAILY_PERIODS_PER_YEAR,
    }
    return metrics, returns


def backtest_sentiment_recovery(
    frames: Mapping[str, pd.DataFrame],
    sentiment: pd.DataFrame,
    parameters: SentimentRecoveryParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> SentimentRecoveryResult:
    """Run one exact next-open sentiment-recovery path."""

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
    if tuple(closes.columns) != ("BTC-EUR", "ETH-EUR"):
        raise ValueError("sentiment v1 requires the fixed BTC/ETH basket")
    fear_greed, sentiment_metadata = _validated_sentiment(
        sentiment,
        index=closes.index,
    )
    warmup = max(
        parameters.trend_ema_period,
        parameters.recovery_lookback_days,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 90:
        raise ValueError("insufficient history for sentiment recovery")

    trend = closes.ewm(
        span=parameters.trend_ema_period,
        adjust=False,
        min_periods=parameters.trend_ema_period,
    ).mean()
    history_eligible = (
        closes.notna().cumsum()
        >= portfolio_policy.minimum_history_observations
    )
    asset_trend = closes > trend
    btc_trend = asset_trend[benchmark]
    rolling_fear = fear_greed.rolling(
        parameters.recovery_lookback_days,
        min_periods=parameters.recovery_lookback_days,
    ).min()
    recovery_state = (
        (rolling_fear <= parameters.fear_threshold)
        & (fear_greed >= rolling_fear + parameters.recovery_delta)
    )
    recovery_event = recovery_state & ~recovery_state.shift(
        1,
        fill_value=False,
    )
    valid = (
        history_eligible
        & opens.notna()
        & closes.notna()
        & trend.notna()
    )
    entries = valid & asset_trend
    entries = entries.mul(
        recovery_event & btc_trend & fear_greed.notna(),
        axis=0,
    )
    exits = (
        ~valid
        | ~asset_trend
    )
    exits = exits | pd.DataFrame(
        np.repeat(
            (
                (~btc_trend)
                | (
                    fear_greed
                    >= parameters.greed_exit_threshold
                )
            ).to_numpy()[:, None],
            len(closes.columns),
            axis=1,
        ),
        index=closes.index,
        columns=closes.columns,
    )
    active = _holding_state(
        entries,
        exits,
        maximum_holding_days=parameters.maximum_holding_days,
    )
    desired = active.astype(float) * parameters.target_weight_per_asset
    desired = desired.clip(
        upper=portfolio_policy.maximum_position_exposure
    )
    total = desired.sum(axis=1)
    scale = (
        portfolio_policy.maximum_total_exposure
        / total.replace(0.0, np.nan)
    ).clip(upper=1.0).fillna(0.0)
    desired = desired.mul(scale, axis=0)
    signal_event = desired.ne(desired.shift(1)).any(axis=1)
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
        raise ValueError("held sentiment asset lacks next-open valuation")
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
        raise ValueError("sentiment costs exhaust portfolio equity")
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
    for execution_position in range(start, len(index)):
        decision_position = execution_position - 1
        if not bool(signal_event.iloc[decision_position]):
            continue
        execution_at = index[execution_position]
        target = executed.loc[execution_at]
        prior_target = (
            executed.iloc[execution_position - start - 1]
            if execution_position > start
            else pd.Series(0.0, index=executed.columns)
        )
        entered = list(
            target.index[
                (target > 1e-12) & (prior_target <= 1e-12)
            ]
        )
        exited = list(
            target.index[
                (target <= 1e-12) & (prior_target > 1e-12)
            ]
        )
        decisions.append(
            {
                "decision_at": index[decision_position],
                "executed_at": execution_at,
                "reason": (
                    "SENTIMENT_RECOVERY_ENTRY"
                    if entered
                    else "GREED_TREND_OR_TIME_EXIT"
                ),
                "scheduled": False,
                "turnover": float(
                    (target - prior_target).abs().sum()
                ),
                "expected_cost_fraction": float(
                    (target - prior_target).abs().sum()
                    * one_way_cost
                ),
                "target_weights": {
                    market: float(value)
                    for market, value in target.items()
                    if float(value) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "entry_assets": entered,
                "exit_assets": exited,
            }
        )
    decisions.append(
        {
            "decision_at": index[-1],
            "executed_at": index[-1],
            "reason": "TERMINAL_LIQUIDATION",
            "scheduled": False,
            "turnover": terminal_turnover,
            "expected_cost_fraction": terminal_turnover * one_way_cost,
            "target_weights": {},
            "cash_fraction": 1.0,
            "entry_assets": [],
            "exit_assets": list(
                executed.columns[executed.iloc[-1] > 1e-12]
            ),
        }
    )
    decision_frame = pd.DataFrame(decisions)
    metrics = _metrics(equity, executed, decision_frame)
    exposure = executed.sum(axis=1)
    integrity = {
        "timeframe": "1d",
        "periods_per_year": DAILY_PERIODS_PER_YEAR,
        "annualization_frequency_correct": True,
        "sentiment_source_timestamped": True,
        "sentiment_backward_only_alignment": bool(
            sentiment_metadata["backward_only_alignment"]
        ),
        "sentiment_backfill_before_inception": False,
        "retrospective_provider_history_disclosed": bool(
            sentiment_metadata["retrospective_provider_history"]
        ),
        "historical_revision_vintages_available": False,
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
        "orders_generated": 0,
    }
    return SentimentRecoveryResult(
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
        signal_diagnostics={
            "recovery_event_count": int(recovery_event.sum()),
            "entry_signal_count": int(entries.sum().sum()),
            "exit_signal_count": int(exits.sum().sum()),
            "active_asset_days": int(active.sum().sum()),
            "sentiment_metadata": sentiment_metadata,
        },
    )


__all__ = [
    "DAILY_PERIODS_PER_YEAR",
    "SENTIMENT_RECOVERY_ENGINE_VERSION",
    "SENTIMENT_RECOVERY_FAMILY",
    "SentimentRecoveryParameters",
    "SentimentRecoveryResult",
    "backtest_sentiment_recovery",
    "sentiment_recovery_parameter_set",
    "sentiment_recovery_period_metrics",
]
