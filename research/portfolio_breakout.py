"""Causal multi-asset time-series breakout portfolio research.

Signals are formed from completed daily closes and executed no earlier than the
next available open.  The module owns only the breakout portfolio hypothesis;
data acquisition, optimization, execution and live permissions remain external.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

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
from utils.pandas_time import sunday_week_end_labels

BreakoutWeighting = Literal["equal", "inverse_volatility", "atr_risk"]

BREAKOUT_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class BreakoutPortfolioParameters:
    """Immutable DNA for one classic time-series breakout hypothesis."""

    entry_lookback: int
    exit_lookback: int
    trend_ema_period: int
    weighting: BreakoutWeighting
    volatility_lookback: int = 20
    maximum_positions: int = 2
    rebalance_days: int = 7
    rebalance_buffer: float = 0.05

    def __post_init__(self) -> None:
        if self.entry_lookback < 10:
            raise ValueError("entry lookback must be at least 10")
        if self.exit_lookback < 2 or self.exit_lookback >= self.entry_lookback:
            raise ValueError("exit lookback must be shorter than entry lookback")
        if self.trend_ema_period < 20:
            raise ValueError("trend EMA period must be at least 20")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError(f"unsupported breakout weighting: {self.weighting}")
        if self.volatility_lookback < 10:
            raise ValueError("volatility lookback must be at least 10")
        if self.maximum_positions < 1:
            raise ValueError("maximum positions must be positive")
        if self.rebalance_days < 1:
            raise ValueError("rebalance days must be positive")
        if not 0.0 <= self.rebalance_buffer < 1.0:
            raise ValueError("rebalance buffer must be in [0, 1)")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "MULTI_ASSET_TIME_SERIES_BREAKOUT",
                "engine_version": BREAKOUT_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


@dataclass(frozen=True)
class AtrRiskBreakoutParameters:
    """Separate 4h ATR-risk challenger; frozen daily Turtle DNA stays unchanged."""

    entry_lookback: int = 120
    exit_lookback: int = 60
    trend_ema_period: int = 600
    weighting: BreakoutWeighting = "atr_risk"
    volatility_lookback: int = 20
    maximum_positions: int = 3
    rebalance_days: int = 1
    rebalance_buffer: float = 0.02
    atr_lookback: int = 14
    atr_stop_multiple: float = 2.0
    risk_fraction_per_position: float = 0.005
    timeframe: str = "4h"

    def __post_init__(self) -> None:
        if self.timeframe != "4h":
            raise ValueError("ATR-risk breakout v1 is fixed to 4h")
        if (self.entry_lookback, self.exit_lookback) != (120, 60):
            raise ValueError("ATR-risk breakout v1 channel is fixed at 120/60")
        if self.trend_ema_period != 600:
            raise ValueError("ATR-risk breakout v1 EMA is fixed at 600")
        if self.weighting != "atr_risk":
            raise ValueError("ATR-risk breakout requires atr_risk weighting")
        if self.atr_lookback != 14 or self.atr_stop_multiple != 2.0:
            raise ValueError("ATR-risk breakout v1 uses ATR(14) and a 2x risk distance")
        if self.risk_fraction_per_position != 0.005:
            raise ValueError("ATR-risk breakout v1 risks 0.5% per position")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "MULTI_ASSET_4H_TURTLE_ATR_RISK",
                "engine_version": "1.0.0",
                "parameters": asdict(self),
            },
            length=64,
        )


@dataclass(frozen=True)
class EfficientAtrRiskBreakoutParameters:
    """Bounded low-turnover 4h ATR-risk research challenger.

    This is deliberately a separate family/version from the frozen v1 DNA.
    The parameter bounds prevent the campaign from turning into an unbounded
    optimizer while allowing channel, trend and rebalance efficiency studies.
    """

    entry_lookback: int
    exit_lookback: int
    trend_ema_period: int
    rebalance_days: int
    rebalance_buffer: float
    weighting: BreakoutWeighting = "atr_risk"
    volatility_lookback: int = 20
    maximum_positions: int = 3
    atr_lookback: int = 14
    atr_stop_multiple: float = 2.0
    risk_fraction_per_position: float = 0.005
    timeframe: str = "4h"

    def __post_init__(self) -> None:
        if self.timeframe != "4h":
            raise ValueError("efficient ATR-risk breakout is fixed to 4h")
        if self.entry_lookback not in {120, 180, 240}:
            raise ValueError("entry lookback is outside the pre-registered set")
        if self.exit_lookback not in {30, 60, 90}:
            raise ValueError("exit lookback is outside the pre-registered set")
        if self.exit_lookback >= self.entry_lookback:
            raise ValueError("exit lookback must be shorter than entry lookback")
        if self.trend_ema_period not in {600, 900, 1200}:
            raise ValueError("trend EMA is outside the pre-registered set")
        if self.rebalance_days not in {3, 7}:
            raise ValueError("rebalance cadence is outside the pre-registered set")
        if self.rebalance_buffer not in {0.05, 0.10}:
            raise ValueError("rebalance buffer is outside the pre-registered set")
        if self.weighting != "atr_risk":
            raise ValueError("efficient ATR-risk breakout requires atr_risk weighting")
        if self.volatility_lookback != 20:
            raise ValueError("efficient ATR-risk breakout uses volatility lookback 20")
        if self.maximum_positions != 3:
            raise ValueError("efficient ATR-risk breakout uses maximum three positions")
        if self.atr_lookback != 14 or self.atr_stop_multiple != 2.0:
            raise ValueError("efficient ATR-risk breakout uses ATR(14) and 2x risk distance")
        if self.risk_fraction_per_position != 0.005:
            raise ValueError("efficient ATR-risk breakout risks 0.5% per position")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "MULTI_ASSET_4H_EFFICIENT_TURTLE_ATR_RISK",
                "engine_version": "2.0.0",
                "parameters": asdict(self),
            },
            length=64,
        )


AtrBreakoutParameters = AtrRiskBreakoutParameters | EfficientAtrRiskBreakoutParameters
AnyBreakoutParameters = BreakoutPortfolioParameters | AtrBreakoutParameters


def _is_atr_risk(parameters: AnyBreakoutParameters) -> bool:
    return isinstance(
        parameters,
        (AtrRiskBreakoutParameters, EfficientAtrRiskBreakoutParameters),
    )


def efficient_atr_breakout_parameter_set() -> tuple[EfficientAtrRiskBreakoutParameters, ...]:
    """Return the 24 pre-registered efficiency challengers; never expand ad hoc."""

    rows = tuple(
        EfficientAtrRiskBreakoutParameters(
            entry_lookback=entry,
            exit_lookback=exit_,
            trend_ema_period=ema,
            rebalance_days=rebalance,
            rebalance_buffer=buffer,
        )
        for entry, exit_, ema in (
            (120, 30, 600),
            (120, 60, 600),
            (180, 60, 900),
            (180, 90, 900),
            (240, 60, 1200),
            (240, 90, 1200),
        )
        for rebalance in (3, 7)
        for buffer in (0.05, 0.10)
    )
    if len(rows) != 24 or len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("efficient ATR parameter set is not the frozen 24-DNA grid")
    return rows


def breakout_portfolio_parameter_set() -> tuple[BreakoutPortfolioParameters, ...]:
    """Return the eight pre-registered classic Turtle-style variants."""

    rows = (
        BreakoutPortfolioParameters(
            entry_lookback=entry,
            exit_lookback=exit_,
            trend_ema_period=ema,
            weighting=weighting,
        )
        for entry, exit_ in ((20, 10), (55, 20))
        for ema in (50, 200)
        for weighting in ("equal", "inverse_volatility")
    )
    result = tuple(rows)
    if len({row.dna_hash for row in result}) != len(result):
        raise RuntimeError("breakout parameter set contains duplicate DNA")
    return result


@dataclass(frozen=True)
class BreakoutPortfolioResult:
    parameters: AnyBreakoutParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    position_episodes: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        atr_risk = _is_atr_risk(self.parameters)
        efficient = isinstance(self.parameters, EfficientAtrRiskBreakoutParameters)
        return {
            "strategy_family": (
                "MULTI_ASSET_4H_EFFICIENT_TURTLE_ATR_RISK"
                if efficient
                else "MULTI_ASSET_4H_TURTLE_ATR_RISK"
                if atr_risk
                else "MULTI_ASSET_TIME_SERIES_BREAKOUT"
            ),
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "breakout_engine_version": (
                "2.0.0" if efficient else "1.0.0" if atr_risk else BREAKOUT_ENGINE_VERSION
            ),
            "timeframe": "4h" if atr_risk else "1d",
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


def _target_for_decision(
    *,
    decision_index: int,
    current: pd.Series,
    closes: pd.DataFrame,
    upper: pd.DataFrame,
    lower: pd.DataFrame,
    ema: pd.DataFrame,
    volatility: pd.DataFrame,
    atr_fraction: pd.DataFrame | None,
    parameters: AnyBreakoutParameters,
    policy: RotationPortfolioPolicy,
) -> tuple[pd.Series, dict[str, Any]]:
    close = closes.iloc[decision_index]
    history = closes.iloc[: decision_index + 1].notna().sum()
    history_eligible = history >= policy.minimum_history_observations
    valid = (
        history_eligible
        & close.notna()
        & upper.iloc[decision_index].notna()
        & lower.iloc[decision_index].notna()
        & ema.iloc[decision_index].notna()
        & volatility.iloc[decision_index].notna()
    )
    entry = (
        valid
        & (close > upper.iloc[decision_index])
        & (close > ema.iloc[decision_index])
    )
    exit_ = (
        valid
        & (
            (close < lower.iloc[decision_index])
            | (close < ema.iloc[decision_index])
        )
    )
    held = [market for market in current.index if float(current[market]) > 1e-12]
    retained = [market for market in held if not bool(exit_.get(market, True))]
    breakout_strength = (
        (close / upper.iloc[decision_index] - 1.0)
        / volatility.iloc[decision_index].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    candidates = list(
        breakout_strength.where(entry)
        .dropna()
        .sort_values(ascending=False)
        .index
    )
    selected = retained[: parameters.maximum_positions]
    for market in candidates:
        if market not in selected and len(selected) < parameters.maximum_positions:
            selected.append(market)
    zero = pd.Series(0.0, index=closes.columns, dtype=float)
    if not selected:
        return zero, {
            "reason": (
                "EXIT_TO_CASH" if held else "NO_BREAKOUT_SIGNAL"
            ),
            "selected_assets": [],
            "held_before": held,
            "entry_signals": [market for market in entry.index if bool(entry[market])],
            "exit_signals": [market for market in exit_.index if bool(exit_[market])],
            "breakout_scores": {
                market: float(value)
                for market, value in breakout_strength.dropna().items()
            },
            "history_eligible_assets": [
                market for market in history_eligible.index if bool(history_eligible[market])
            ],
            "cash_reason_codes": [
                "CASH_NO_BREAKOUT_SIGNAL",
                "CASH_OPERATIONAL_RESERVE",
            ],
            "cash_attribution": {
                "CASH_NO_BREAKOUT_SIGNAL": policy.maximum_total_exposure,
                "CASH_OPERATIONAL_RESERVE": policy.minimum_cash,
            },
            "target_total_exposure": 0.0,
            "target_cash_fraction": 1.0,
        }

    if parameters.weighting == "atr_risk":
        if atr_fraction is None:
            raw = pd.Series(dtype=float)
        else:
            risk_distance = (
                atr_fraction.iloc[decision_index]
                .reindex(selected)
                .mul(parameters.atr_stop_multiple)
                .replace(0.0, np.nan)
            )
            raw = (
                parameters.risk_fraction_per_position / risk_distance
            ).replace([np.inf, -np.inf], np.nan).dropna().clip(
                upper=policy.maximum_position_exposure,
            )
        selected = list(raw.index)
    elif parameters.weighting == "inverse_volatility":
        selected_volatility = (
            volatility.iloc[decision_index].reindex(selected).replace(0.0, np.nan)
        )
        raw = (1.0 / selected_volatility).dropna()
        selected = list(raw.index)
    else:
        raw = pd.Series(1.0, index=selected, dtype=float)
    if raw.empty:
        return zero, {
            "reason": "RISK_MODEL_UNAVAILABLE",
            "selected_assets": [],
            "held_before": held,
            "entry_signals": candidates,
            "exit_signals": [market for market in exit_.index if bool(exit_[market])],
            "breakout_scores": {},
            "history_eligible_assets": [
                market for market in history_eligible.index if bool(history_eligible[market])
            ],
            "cash_reason_codes": [
                "CASH_RISK_MODEL_FAILURE",
                "CASH_OPERATIONAL_RESERVE",
            ],
            "cash_attribution": {
                "CASH_RISK_MODEL_FAILURE": policy.maximum_total_exposure,
                "CASH_OPERATIONAL_RESERVE": policy.minimum_cash,
            },
            "target_total_exposure": 0.0,
            "target_cash_fraction": 1.0,
        }
    if parameters.weighting == "atr_risk":
        allocations = raw.copy()
        if float(allocations.sum()) > policy.maximum_total_exposure:
            allocations *= policy.maximum_total_exposure / float(allocations.sum())
    else:
        allocations = _capped_allocations(
            raw,
            total_exposure=policy.maximum_total_exposure,
            maximum_position_exposure=policy.maximum_position_exposure,
        )
    target = zero.copy()
    target.loc[allocations.index] = allocations
    allocated = float(target.sum())
    unused = max(0.0, policy.maximum_total_exposure - allocated)
    cash_attribution = {"CASH_OPERATIONAL_RESERVE": policy.minimum_cash}
    if unused > 1e-12:
        cash_attribution["CASH_POSITION_CAP"] = unused
    return target, {
        "reason": (
            "BREAKOUT_ENTRY"
            if any(market not in held for market in selected)
            else "HOLD_OR_RESIZE"
        ),
        "selected_assets": selected,
        "held_before": held,
        "entry_signals": candidates,
        "exit_signals": [market for market in exit_.index if bool(exit_[market])],
        "breakout_scores": {
            market: float(value)
            for market, value in breakout_strength.dropna().items()
        },
        "history_eligible_assets": [
            market for market in history_eligible.index if bool(history_eligible[market])
        ],
        "pre_cap_weights": {
            market: float(value)
            for market, value in (raw / raw.sum() * policy.maximum_total_exposure).items()
        },
        "weights_after_caps": {
            market: float(value) for market, value in allocations.items()
        },
        "cash_reason_codes": list(cash_attribution),
        "cash_attribution": cash_attribution,
        "target_total_exposure": allocated,
        "target_cash_fraction": float(1.0 - allocated),
    }


def backtest_breakout_portfolio(
    frames: Mapping[str, pd.DataFrame],
    parameters: AnyBreakoutParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> BreakoutPortfolioResult:
    """Run a next-open, long-only, cash-enabled breakout portfolio backtest."""

    if min(fee_rate, slippage_bps, spread_bps) < 0:
        raise ValueError("cost assumptions cannot be negative")
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=portfolio_policy,
    )
    # A venue may omit an isolated candle for one asset while the benchmark
    # calendar continues. Signals and executions always use the raw panels.
    # Mark-to-market alone carries the last observed price forward after the
    # asset's real inception, so a held position is not valued with future data
    # and a missing candle cannot fabricate a forced fill.
    valuation_opens = opens.ffill()
    valuation_closes = closes.ffill()
    valuation_carry_rows = int(
        ((valuation_opens.notna()) & opens.isna()).sum().sum()
    )
    atr_risk = _is_atr_risk(parameters)
    warmup = max(
        parameters.entry_lookback,
        parameters.exit_lookback,
        parameters.trend_ema_period,
        parameters.volatility_lookback,
        portfolio_policy.minimum_history_observations,
        parameters.atr_lookback if atr_risk else 0,
    )
    if len(closes) <= warmup + 2:
        raise ValueError("insufficient history for breakout portfolio")
    upper = closes.shift(1).rolling(
        parameters.entry_lookback,
        min_periods=parameters.entry_lookback,
    ).max()
    lower = closes.shift(1).rolling(
        parameters.exit_lookback,
        min_periods=parameters.exit_lookback,
    ).min()
    ema = closes.ewm(
        span=parameters.trend_ema_period,
        adjust=False,
        min_periods=parameters.trend_ema_period,
    ).mean()
    volatility = closes.pct_change(fill_method=None).rolling(
        parameters.volatility_lookback,
        min_periods=parameters.volatility_lookback,
    ).std(ddof=0)
    atr_fraction: pd.DataFrame | None = None
    if atr_risk:
        highs: dict[str, pd.Series] = {}
        lows: dict[str, pd.Series] = {}
        for raw_market, raw in frames.items():
            market = raw_market.upper().replace("/", "-").replace("_", "-")
            if {"high", "low"} - set(raw.columns):
                raise ValueError(f"{market} lacks high/low for ATR-risk sizing")
            normalized = raw.copy()
            normalized.index = pd.to_datetime(normalized.index, utc=True)
            normalized = normalized[
                ~normalized.index.duplicated(keep="last")
            ].sort_index()
            highs[market] = normalized["high"].reindex(closes.index)
            lows[market] = normalized["low"].reindex(closes.index)
        high_panel = pd.DataFrame(highs, index=closes.index)
        low_panel = pd.DataFrame(lows, index=closes.index)
        prior_close = closes.shift(1)
        true_range = pd.concat(
            [
                high_panel - low_panel,
                (high_panel - prior_close).abs(),
                (low_panel - prior_close).abs(),
            ],
            axis=1,
            keys=("range", "high_gap", "low_gap"),
        ).T.groupby(level=1).max().T
        atr_fraction = (
            true_range.rolling(
                parameters.atr_lookback,
                min_periods=parameters.atr_lookback,
            ).mean()
            / closes
        ).replace([np.inf, -np.inf], np.nan)
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    current = pd.Series(0.0, index=closes.columns, dtype=float)
    net_equity = 1.0
    gross_equity = 1.0
    total_turnover = 0.0
    total_cost = 0.0
    rebalance_count = 0
    buy_fills = 0
    sell_fills = 0
    open_episodes: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    weights: list[pd.Series] = []
    net_rows: list[tuple[pd.Timestamp, float]] = [(closes.index[warmup], 1.0)]
    gross_rows: list[tuple[pd.Timestamp, float]] = [(closes.index[warmup], 1.0)]
    asset_pnl = {market: 0.0 for market in closes.columns}
    asset_costs = {market: 0.0 for market in closes.columns}

    for execution_index in range(warmup + 1, len(closes)):
        decision_index = execution_index - 1
        scheduled = (decision_index - warmup) % parameters.rebalance_days == 0
        close = closes.iloc[decision_index]
        lower_now = lower.iloc[decision_index]
        ema_now = ema.iloc[decision_index]
        held = current[current > 1e-12].index
        urgent_exit = any(
            not math.isfinite(float(close[market]))
            or not math.isfinite(float(lower_now[market]))
            or not math.isfinite(float(ema_now[market]))
            or float(close[market]) < float(lower_now[market])
            or float(close[market]) < float(ema_now[market])
            for market in held
        )
        entry_now = bool(
            (
                (close > upper.iloc[decision_index])
                & (close > ema_now)
                & close.notna()
            ).any()
        )
        if scheduled or urgent_exit or entry_now:
            target, audit = _target_for_decision(
                decision_index=decision_index,
                current=current,
                closes=closes,
                upper=upper,
                lower=lower,
                ema=ema,
                volatility=volatility,
                atr_fraction=atr_fraction,
                parameters=parameters,
                policy=portfolio_policy,
            )
            executable = opens.iloc[execution_index].notna()
            # New entries and exits require an actual venue open. A currently
            # held asset with no candle remains held until the next executable
            # quote; no synthetic fill is created.
            target = target.where(executable, current)
            prior = current.copy()
            same_selection = set(target[target > 1e-12].index) == set(
                prior[prior > 1e-12].index
            )
            buffered = False
            if (
                same_selection
                and bool((target > 1e-12).any())
                and float((target - prior).abs().max()) < parameters.rebalance_buffer
            ):
                target = prior.copy()
                buffered = True
            changes = target - prior
            turnover = float(changes.abs().sum())
            cost_amount = net_equity * turnover * one_way_cost
            if turnover > 1e-15:
                net_equity -= cost_amount
                total_turnover += turnover
                total_cost += cost_amount
                rebalance_count += 1
                for market, change in changes.abs().items():
                    if float(change) > 1e-15:
                        asset_costs[market] += (
                            cost_amount * float(change) / turnover
                        )
            buy_fills += int((changes > 1e-12).sum())
            sell_fills += int((changes < -1e-12).sum())
            execution_prices = opens.iloc[execution_index]
            for market in closes.columns:
                before = float(prior[market])
                after = float(target[market])
                if before <= 1e-12 and after > 1e-12:
                    open_episodes[market] = {
                        "market": market,
                        "opened_at": opens.index[execution_index],
                        "entry_price": float(execution_prices[market]),
                        "entry_weight": after,
                    }
                elif before > 1e-12 and after <= 1e-12:
                    episode = open_episodes.pop(market, None)
                    if episode is not None:
                        gross_return = (
                            float(execution_prices[market])
                            / float(episode["entry_price"])
                            - 1.0
                        )
                        episodes.append(
                            episode
                            | {
                                "closed_at": opens.index[execution_index],
                                "exit_price": float(execution_prices[market]),
                                "gross_return": gross_return,
                                "net_return": gross_return - 2.0 * one_way_cost,
                                "close_reason": audit["reason"],
                            }
                        )
            current = target
            decisions.append(
                {
                    "decision_at": closes.index[decision_index],
                    "executed_at": opens.index[execution_index],
                    "scheduled": scheduled,
                    "urgent_exit": urgent_exit,
                    "turnover": turnover,
                    "expected_cost_fraction": turnover * one_way_cost,
                    "cost_amount": cost_amount,
                    "rebalance_buffer_applied": buffered,
                    "buy_fill_count": int((changes > 1e-12).sum()),
                    "sell_fill_count": int((changes < -1e-12).sum()),
                    "target_weights": {
                        market: float(value)
                        for market, value in target.items()
                        if float(value) > 1e-12
                    },
                    "cash_fraction": float(1.0 - target.sum()),
                    "equity_after_cost": net_equity,
                    **audit,
                }
            )

        terminal = execution_index == len(closes) - 1
        asset_returns = (
            valuation_closes.iloc[execution_index]
            / valuation_opens.iloc[execution_index]
            - 1.0
            if terminal
            else valuation_opens.iloc[execution_index + 1]
            / valuation_opens.iloc[execution_index]
            - 1.0
        )
        held = current[current > 1e-12].index
        missing_held = [
            market
            for market in held
            if not math.isfinite(float(asset_returns.get(market, np.nan)))
        ]
        if missing_held:
            raise ValueError(
                "held breakout asset lacks a causal next valuation:"
                f"{closes.index[execution_index]}:{missing_held}"
            )
        equity_before_return = net_equity
        portfolio_return = float((current * asset_returns).sum())
        for market, weight in current.items():
            if float(weight) > 1e-12:
                asset_pnl[market] += (
                    equity_before_return
                    * float(weight)
                    * float(asset_returns[market])
                )
        net_equity *= 1.0 + portfolio_return
        gross_equity *= 1.0 + portfolio_return
        if terminal:
            terminal_turnover = float(current.sum())
            terminal_cost = net_equity * terminal_turnover * one_way_cost
            net_equity -= terminal_cost
            total_turnover += terminal_turnover
            total_cost += terminal_cost
            if terminal_turnover > 1e-15:
                for market, weight in current.items():
                    if float(weight) > 1e-15:
                        asset_costs[market] += (
                            terminal_cost * float(weight) / terminal_turnover
                        )
            for market in list(open_episodes):
                episode = open_episodes.pop(market)
                exit_price = float(
                    valuation_closes[market].iloc[execution_index]
                )
                gross_return = exit_price / float(episode["entry_price"]) - 1.0
                episodes.append(
                    episode
                    | {
                        "closed_at": closes.index[execution_index],
                        "exit_price": exit_price,
                        "gross_return": gross_return,
                        "net_return": gross_return - 2.0 * one_way_cost,
                        "close_reason": "TERMINAL_LIQUIDATION",
                    }
                )
            sell_fills += int((current > 1e-12).sum())
            decisions.append(
                {
                    "decision_at": closes.index[execution_index],
                    "executed_at": closes.index[execution_index],
                    "scheduled": False,
                    "urgent_exit": False,
                    "reason": "TERMINAL_LIQUIDATION",
                    "selected_assets": [],
                    "turnover": terminal_turnover,
                    "expected_cost_fraction": terminal_turnover * one_way_cost,
                    "cost_amount": terminal_cost,
                    "rebalance_buffer_applied": False,
                    "buy_fill_count": 0,
                    "sell_fill_count": int((current > 1e-12).sum()),
                    "target_weights": {},
                    "cash_fraction": 1.0,
                    "equity_after_cost": net_equity,
                }
            )
            current *= 0.0
        timestamp = (
            closes.index[execution_index]
            if terminal
            else opens.index[execution_index + 1]
        )
        net_rows.append((timestamp, net_equity))
        gross_rows.append((timestamp, gross_equity))
        row = current.copy()
        row.name = timestamp
        weights.append(row)

    equity = pd.Series(
        [value for _, value in net_rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in net_rows]),
        name="net_equity",
        dtype=float,
    )
    gross = pd.Series(
        [value for _, value in gross_rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in gross_rows]),
        name="gross_equity",
        dtype=float,
    )
    executed = pd.DataFrame(weights).fillna(0.0)
    decisions_frame = pd.DataFrame(decisions)
    episodes_frame = pd.DataFrame(episodes)
    returns = equity.pct_change(fill_method=None).dropna()
    weekly_equity = equity.groupby(sunday_week_end_labels(equity.index)).last()
    weekly_returns = weekly_equity.pct_change(fill_method=None).dropna()
    ess, lag_one = _effective_sample_size(weekly_returns)
    elapsed_days = max(
        1.0,
        (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0,
    )
    years = elapsed_days / 365.25
    drawdown = equity / equity.cummax() - 1.0
    standard = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    )
    exposure = executed.sum(axis=1)
    closed_returns = (
        episodes_frame["net_return"].astype(float)
        if not episodes_frame.empty
        else pd.Series(dtype=float)
    )
    periods_per_year = 365.25 * 6.0 if atr_risk else 365.25
    metrics = {
        "net_return": float(equity.iloc[-1] - 1.0),
        "gross_return": float(gross.iloc[-1] - 1.0),
        "annualized_return": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_volatility": standard * math.sqrt(periods_per_year),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(periods_per_year))
            if standard > 0
            else 0.0
        ),
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(periods_per_year)
            )
            if downside_deviation > 0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "average_drawdown": (
            float(drawdown[drawdown < 0].mean())
            if bool((drawdown < 0).any())
            else 0.0
        ),
        "average_exposure": float(exposure.mean()),
        "maximum_exposure_observed": float(exposure.max()),
        "cash_fraction_average": float(1.0 - exposure.mean()),
        "maximum_positions_observed": int(
            (executed > 1e-12).sum(axis=1).max()
        ),
        "maximum_position_exposure_observed": float(executed.max(axis=1).max()),
        "turnover": total_turnover,
        "rebalance_count": rebalance_count,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "closed_position_episodes": int(len(episodes_frame)),
        "portfolio_period_profit_factor": _profit_factor(weekly_returns),
        "closed_position_profit_factor": _profit_factor(closed_returns),
        "portfolio_period_effective_sample_size": ess,
        "portfolio_period_lag_one_autocorrelation": lag_one,
        "raw_portfolio_period_observations": int(len(weekly_returns)),
        "decision_reason_counts": {
            str(reason): int(count)
            for reason, count in decisions_frame["reason"].value_counts().items()
        },
        "asset_pnl_attribution": {
            market: {
                "gross_pnl_amount": float(asset_pnl[market]),
                "cost_amount": float(asset_costs[market]),
                "net_pnl_amount": float(asset_pnl[market] - asset_costs[market]),
            }
            for market in closes.columns
        },
        "asset_pnl_reconciliation_error": float(
            sum(
                asset_pnl[market] - asset_costs[market]
                for market in closes.columns
            )
            - (net_equity - 1.0)
        ),
        "position_sizing_policy": (
            "ATR_EQUAL_EUR_RISK"
            if atr_risk
            else parameters.weighting.upper()
        ),
        "atr_risk_fraction_per_position": (
            parameters.risk_fraction_per_position if atr_risk else None
        ),
        "atr_stop_multiple": (
            parameters.atr_stop_multiple if atr_risk else None
        ),
        "periods_per_year": periods_per_year,
        "valuation_carry_rows": valuation_carry_rows,
    }
    integrity = {
        "no_lookahead": True,
        "prior_channel_only": True,
        "decision_at_close_execution_next_open": True,
        "closed_candles_only": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "maximum_positions_respected": (
            metrics["maximum_positions_observed"] <= parameters.maximum_positions
        ),
        "maximum_exposure_respected": (
            metrics["maximum_exposure_observed"]
            <= portfolio_policy.maximum_total_exposure + 1e-12
        ),
        "maximum_position_exposure_respected": (
            metrics["maximum_position_exposure_observed"]
            <= portfolio_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": (
            metrics["maximum_exposure_observed"]
            <= 1.0 - portfolio_policy.minimum_cash + 1e-12
        ),
        "fail_closed_allowed_universe": bool(portfolio_policy.allowed_markets),
        "point_in_time_asset_inception": True,
        "signal_ohlc_not_forward_filled": True,
        "valuation_carry_is_backward_only": True,
        "missing_candle_execution_deferred": True,
        "asset_pnl_reconciled": (
            abs(float(metrics["asset_pnl_reconciliation_error"])) <= 1e-10
        ),
        "terminal_liquidation_recorded": (
            not decisions_frame.empty
            and decisions_frame.iloc[-1]["reason"] == "TERMINAL_LIQUIDATION"
        ),
    }
    return BreakoutPortfolioResult(
        parameters=parameters,
        portfolio_policy=portfolio_policy,
        metrics=metrics,
        integrity=integrity,
        cost_breakdown={
            "fee_rate": float(fee_rate),
            "slippage_bps": float(slippage_bps),
            "spread_bps": float(spread_bps),
            "one_way_cost_rate": float(one_way_cost),
            "total_one_way_turnover": float(total_turnover),
            "total_cost_amount": float(total_cost),
            "gross_minus_net_return": float(gross.iloc[-1] - equity.iloc[-1]),
        },
        equity_curve=equity,
        gross_equity_curve=gross,
        executed_weights=executed,
        decisions=decisions_frame,
        position_episodes=episodes_frame,
    )


def breakout_observer_snapshot(result: BreakoutPortfolioResult) -> dict[str, Any]:
    """Return the latest historical research decision with zero order authority."""

    usable = result.decisions[
        result.decisions["reason"] != "TERMINAL_LIQUIDATION"
    ]
    latest = usable.iloc[-1].to_dict() if not usable.empty else {}
    return {
        "status": "FROZEN_FORWARD_RESEARCH",
        "strategy_dna_hash": result.parameters.dna_hash,
        "execution_identity": result.summary()["execution_identity"],
        "latest_historical_decision": latest,
        "execution_instruction": "NEXT_AVAILABLE_OPEN_HYPOTHETICAL_ONLY",
        "orders_generated": 0,
        "orders_submitted": 0,
        "candidate_promotion_implied": False,
    }


__all__ = [
    "BREAKOUT_ENGINE_VERSION",
    "AtrRiskBreakoutParameters",
    "BreakoutPortfolioParameters",
    "BreakoutPortfolioResult",
    "EfficientAtrRiskBreakoutParameters",
    "backtest_breakout_portfolio",
    "breakout_observer_snapshot",
    "breakout_portfolio_parameter_set",
    "efficient_atr_breakout_parameter_set",
]
