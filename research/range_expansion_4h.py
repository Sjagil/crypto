"""Causal 4h range-expansion and volume-confirmation portfolio research.

Entries require a close above a strictly prior price channel, an expanded true
range versus strictly prior ATR, causal relative-volume confirmation and
long-horizon asset/BTC trend eligibility. Completed 4h bars generate signals
for the next available 4h open. This module cannot submit or promote orders.
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
from research.volatility_contraction import (
    _desired_weights,
    _holding_state,
)
from utils.common import stable_hash

RANGE_EXPANSION_4H_FAMILY = (
    "MULTI_ASSET_4H_RANGE_EXPANSION_VOLUME_CONFIRMATION"
)
RANGE_EXPANSION_4H_ENGINE_VERSION = "1.1.0"
FOUR_HOUR_PERIODS_PER_DAY = 6
FOUR_HOUR_PERIODS_PER_YEAR = 365.25 * FOUR_HOUR_PERIODS_PER_DAY


@dataclass(frozen=True, slots=True)
class RangeExpansion4hParameters:
    """Immutable DNA for one of the 16 predeclared 4h paths."""

    entry_lookback: int
    exit_lookback: int
    range_expansion_multiple: float
    relative_volume_multiple: float
    asset_ema_period: int
    atr_lookback: int = 42
    volume_lookback: int = 42
    volatility_lookback: int = 42
    target_annualized_volatility: float = 0.15
    btc_ema_period: int = 600
    maximum_positions: int = 2
    rebalance_interval_bars: int = 42

    def __post_init__(self) -> None:
        if (self.entry_lookback, self.exit_lookback) not in {
            (30, 15),
            (60, 30),
        }:
            raise ValueError("undeclared 4h channel pair")
        if self.range_expansion_multiple not in {1.0, 1.5}:
            raise ValueError("undeclared 4h range expansion multiple")
        if self.relative_volume_multiple not in {1.0, 1.5}:
            raise ValueError("undeclared 4h relative-volume multiple")
        if self.asset_ema_period not in {300, 600}:
            raise ValueError("undeclared 4h asset EMA")
        if (
            self.atr_lookback,
            self.volume_lookback,
            self.volatility_lookback,
        ) != (42, 42, 42):
            raise ValueError("v1 4h causal baselines are fixed at 42 bars")
        if self.target_annualized_volatility != 0.15:
            raise ValueError("v1 4h volatility target is fixed at 15%")
        if self.btc_ema_period != 600:
            raise ValueError("v1 4h BTC EMA is fixed at 600 bars")
        if self.maximum_positions != 2:
            raise ValueError("v1 4h maximum positions is fixed at two")
        if self.rebalance_interval_bars != 42:
            raise ValueError("v1 4h rebalance interval is fixed at 42 bars")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": RANGE_EXPANSION_4H_FAMILY,
                "engine_version": RANGE_EXPANSION_4H_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def range_expansion_4h_parameter_set(
) -> tuple[RangeExpansion4hParameters, ...]:
    """Return the exact 16-trial predeclared family."""

    rows = tuple(
        RangeExpansion4hParameters(
            entry_lookback=entry,
            exit_lookback=exit_,
            range_expansion_multiple=range_multiple,
            relative_volume_multiple=volume_multiple,
            asset_ema_period=ema,
        )
        for (
            (entry, exit_),
            range_multiple,
            volume_multiple,
            ema,
        ) in product(
            ((30, 15), (60, 30)),
            (1.0, 1.5),
            (1.0, 1.5),
            (300, 600),
        )
    )
    if len(rows) != 16:
        raise RuntimeError("4h range-expansion family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("4h range-expansion family contains duplicate DNA")
    return rows


def relabel_4h_forward_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Express generic observer counters in explicit four-hour units."""

    result = dict(summary)
    observation_count = int(
        result.pop(
            "closed_daily_observations",
            result.get("closed_4h_observations", 0),
        )
    )
    required = int(
        result.pop(
            "required_closed_daily_observations",
            result.get("required_closed_4h_observations", 0),
        )
    )
    result.pop("remaining_closed_daily_observations", None)
    result["observation_unit"] = "CLOSED_FOUR_HOUR_BAR"
    result["closed_4h_observations"] = observation_count
    result["required_closed_4h_observations"] = required
    result["remaining_closed_4h_observations"] = max(
        0,
        required - observation_count,
    )
    result["minimum_calendar_days_equivalent"] = int(
        math.ceil(required / FOUR_HOUR_PERIODS_PER_DAY)
    )

    checks = dict(result.get("checks") or {})
    sample_check = bool(
        checks.pop("minimum_closed_daily_observations", False)
    )
    checks["minimum_closed_4h_observations"] = sample_check
    result["checks"] = checks

    checkpoints = tuple(
        sorted(
            {
                30 * FOUR_HOUR_PERIODS_PER_DAY,
                90 * FOUR_HOUR_PERIODS_PER_DAY,
                180 * FOUR_HOUR_PERIODS_PER_DAY,
                required,
            }
        )
    )
    milestones = [
        {
            "closed_4h_observations": bars,
            "calendar_days_equivalent": (
                bars / FOUR_HOUR_PERIODS_PER_DAY
            ),
            "reached": observation_count >= bars,
            "remaining": max(0, bars - observation_count),
            "purpose": (
                "FORMAL_SAMPLE_GATE"
                if bars == required
                else "DIAGNOSTIC_ONLY"
            ),
        }
        for bars in checkpoints
    ]
    next_pending = next(
        (
            row["closed_4h_observations"]
            for row in milestones
            if not row["reached"]
        ),
        None,
    )
    result["diagnostic_progress"] = {
        "observation_unit": "CLOSED_FOUR_HOUR_BAR",
        "closed_4h_observations": observation_count,
        "formal_required_closed_4h_observations": required,
        "progress_fraction": min(
            1.0,
            observation_count / max(1, required),
        ),
        "milestones": milestones,
        "next_pending_milestone": next_pending,
        "diagnostic_milestones_authorize_promotion": False,
    }
    return result


@dataclass(frozen=True)
class RangeExpansion4hResult:
    parameters: RangeExpansion4hParameters
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
            "strategy_family": RANGE_EXPANSION_4H_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": RANGE_EXPANSION_4H_ENGINE_VERSION,
            "timeframe": "4h",
            "periods_per_year": FOUR_HOUR_PERIODS_PER_YEAR,
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


def _supplemental_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    benchmark_market: str,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for raw_market, raw in frames.items():
        market = raw_market.upper().replace("/", "-").replace("_", "-")
        missing = {"high", "low", "volume"} - set(raw.columns)
        if missing:
            raise ValueError(
                f"{market} lacks 4h range inputs: {sorted(missing)}"
            )
        selected = raw.loc[:, ["high", "low", "volume"]].copy()
        selected.index = pd.to_datetime(selected.index, utc=True)
        selected = selected[
            ~selected.index.duplicated(keep="last")
        ].sort_index()
        values = selected.to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise ValueError(
                f"{market} contains invalid 4h high/low/volume"
            )
        normalized[market] = selected
    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    if benchmark not in normalized:
        raise ValueError("BTC benchmark lacks supplemental 4h inputs")
    highs = pd.DataFrame(
        {
            market: frame["high"].reindex(index)
            for market, frame in normalized.items()
        },
        index=index,
        dtype=float,
    ).sort_index(axis=1)
    lows = pd.DataFrame(
        {
            market: frame["low"].reindex(index)
            for market, frame in normalized.items()
        },
        index=index,
        dtype=float,
    ).sort_index(axis=1)
    volumes = pd.DataFrame(
        {
            market: frame["volume"].reindex(index)
            for market, frame in normalized.items()
        },
        index=index,
        dtype=float,
    ).sort_index(axis=1)
    return highs, lows, volumes


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
            standard * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
        ),
        "sharpe": (
            float(
                returns.mean()
                / standard
                * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
            )
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
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
        "period_unit": "FOUR_HOUR_PORTFOLIO_RETURN",
        "periods_per_year": FOUR_HOUR_PERIODS_PER_YEAR,
    }


def range_expansion_4h_period_metrics(
    equity_curve: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, float | int | str], pd.Series]:
    """Return frequency-correct metrics for a frozen calendar period."""

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
    if len(selected) < 180:
        raise ValueError("4h evaluation period needs at least 180 bars")
    normalized = selected / float(selected.iloc[0])
    returns = normalized.pct_change(fill_method=None).dropna()
    standard = float(returns.std(ddof=0))
    elapsed_days = max(
        1.0,
        (normalized.index[-1] - normalized.index[0]).total_seconds()
        / 86_400.0,
    )
    years = elapsed_days / 365.25
    drawdown = normalized / normalized.cummax() - 1.0
    effective_sample, lag_one = _effective_sample_size(returns)
    metrics: dict[str, float | int | str] = {
        "observations": len(returns),
        "effective_sample_size": effective_sample,
        "lag_one_autocorrelation": lag_one,
        "net_return": float(normalized.iloc[-1] - 1.0),
        "annualized_return": float(
            normalized.iloc[-1] ** (1.0 / years) - 1.0
        ),
        "annualized_volatility": (
            standard * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
        ),
        "sharpe": (
            float(
                returns.mean()
                / standard
                * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
            )
            if standard > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "profit_factor_unit": "FOUR_HOUR_PORTFOLIO_RETURN",
        "positive_observations": int((returns > 0.0).sum()),
        "negative_observations": int((returns < 0.0).sum()),
        "periods_per_year": FOUR_HOUR_PERIODS_PER_YEAR,
    }
    return metrics, returns


def backtest_range_expansion_4h(
    frames: Mapping[str, pd.DataFrame],
    parameters: RangeExpansion4hParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> RangeExpansion4hResult:
    """Run one exact causal 4h range-expansion path."""

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
    highs, lows, volumes = _supplemental_panel(
        frames,
        benchmark_market=benchmark,
        index=closes.index,
    )
    benchmark_bar_count = len(closes)
    common_calendar = (
        opens.notna().all(axis=1)
        & closes.notna().all(axis=1)
        & highs.notna().all(axis=1)
        & lows.notna().all(axis=1)
        & volumes.notna().all(axis=1)
    )
    opens = opens.loc[common_calendar]
    closes = closes.loc[common_calendar]
    highs = highs.loc[common_calendar]
    lows = lows.loc[common_calendar]
    volumes = volumes.loc[common_calendar]
    if len(closes) < 1_000:
        raise ValueError(
            "common non-imputed 4h execution calendar is too short"
        )
    warmup = max(
        parameters.entry_lookback,
        parameters.exit_lookback,
        parameters.atr_lookback,
        parameters.volume_lookback,
        parameters.volatility_lookback,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient history for 4h range expansion")

    previous_close = closes.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (highs - lows).to_numpy(dtype=float),
                (highs - previous_close).abs().to_numpy(dtype=float),
                (lows - previous_close).abs().to_numpy(dtype=float),
            ]
        ),
        index=closes.index,
        columns=closes.columns,
    )
    prior_atr = true_range.shift(1).rolling(
        parameters.atr_lookback,
        min_periods=parameters.atr_lookback,
    ).mean()
    prior_volume_median = volumes.shift(1).rolling(
        parameters.volume_lookback,
        min_periods=parameters.volume_lookback,
    ).median()
    upper = highs.shift(1).rolling(
        parameters.entry_lookback,
        min_periods=parameters.entry_lookback,
    ).max()
    lower = lows.shift(1).rolling(
        parameters.exit_lookback,
        min_periods=parameters.exit_lookback,
    ).min()
    annualized_volatility = (
        closes.pct_change(fill_method=None).rolling(
            parameters.volatility_lookback,
            min_periods=parameters.volatility_lookback,
        ).std(ddof=0)
        * math.sqrt(FOUR_HOUR_PERIODS_PER_YEAR)
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
        & opens.notna()
        & closes.notna()
        & highs.notna()
        & lows.notna()
        & volumes.notna()
        & upper.notna()
        & lower.notna()
        & prior_atr.notna()
        & prior_volume_median.notna()
        & annualized_volatility.notna()
        & asset_ema.notna()
    )
    range_confirmation = (
        true_range
        >= prior_atr * parameters.range_expansion_multiple
    )
    volume_confirmation = (
        volumes
        >= (
            prior_volume_median
            * parameters.relative_volume_multiple
        )
    )
    entries = (
        valid
        & (closes > upper)
        & range_confirmation
        & volume_confirmation
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
        | (closes < lower)
        | (closes < asset_ema)
        | regime_exit
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
        np.arange(len(closes)) % parameters.rebalance_interval_bars
        == 0,
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
        raise ValueError("held 4h asset lacks next-open valuation")
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
        raise ValueError("4h costs exhaust portfolio equity")
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
            "RANGE_VOLUME_BREAKOUT_ENTRY"
            if entry_assets
            else "CHANNEL_OR_REGIME_EXIT"
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
            "entry_signals": [],
            "exit_signals": list(
                executed.columns[executed.iloc[-1] > 1e-12]
            ),
        }
    )
    decision_frame = pd.DataFrame(decisions)
    metrics = _metrics(equity, executed, decision_frame)
    signal_diagnostics = {
        "entry_signal_count": int(entries.sum().sum()),
        "exit_signal_count": int(
            (exits & prior_active).sum().sum()
        ),
        "range_confirmation_count": int(
            range_confirmation.sum().sum()
        ),
        "volume_confirmation_count": int(
            volume_confirmation.sum().sum()
        ),
        "active_asset_bars": int(active.sum().sum()),
        "decision_events": int(signal_event.sum()),
        "benchmark_bar_count": benchmark_bar_count,
        "common_execution_bar_count": int(len(closes)),
        "excluded_non_common_bars": int(
            benchmark_bar_count - len(closes)
        ),
    }
    integrity = {
        "timeframe": "4h",
        "periods_per_day": FOUR_HOUR_PERIODS_PER_DAY,
        "annualization_frequency_correct": True,
        "common_calendar_intersection_only": True,
        "missing_bars_imputed": False,
        "prior_channel_only": True,
        "strictly_prior_atr_baseline": True,
        "strictly_prior_volume_baseline": True,
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
    return RangeExpansion4hResult(
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
    "FOUR_HOUR_PERIODS_PER_DAY",
    "FOUR_HOUR_PERIODS_PER_YEAR",
    "RANGE_EXPANSION_4H_ENGINE_VERSION",
    "RANGE_EXPANSION_4H_FAMILY",
    "RangeExpansion4hParameters",
    "RangeExpansion4hResult",
    "backtest_range_expansion_4h",
    "range_expansion_4h_parameter_set",
    "range_expansion_4h_period_metrics",
    "relabel_4h_forward_summary",
]
