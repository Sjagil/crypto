"""Causal BTC information-shock diffusion into liquid altcoins.

The family tests whether unusually strong positive BTC returns diffuse into
ETH, SOL, and LINK with a short delay. At each completed daily close, the
engine estimates each altcoin's BTC beta strictly from prior observations,
compares its realized move with the beta-implied move, and may hold the most
underreacting satellites. Every target executes only at the following open.
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
from research.residual_reversal import residual_reversal_period_metrics
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

BTC_SHOCK_DIFFUSION_ENGINE_VERSION = "1.0.0"
BTC_SHOCK_DIFFUSION_FAMILY = (
    "BTC_POSITIVE_INFORMATION_SHOCK_ALTCOIN_UNDERREACTION"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class BTCShockDiffusionParameters:
    """Immutable DNA for one preregistered shock-diffusion path."""

    shock_lookback: int
    maximum_holding_days: int
    shock_zscore_threshold: float = 1.0
    shock_baseline_lookback: int = 180
    beta_lookback: int = 90
    btc_ema_period: int = 200
    asset_ema_period: int = 200
    maximum_positions: int = 2
    position_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.shock_lookback not in {1, 3}:
            raise ValueError("v1 shock lookback must be one or three")
        if self.maximum_holding_days not in {3, 5}:
            raise ValueError("v1 holding period must be three or five")
        if self.shock_zscore_threshold != 1.0:
            raise ValueError("v1 shock z-score threshold is fixed at one")
        if self.shock_baseline_lookback != 180:
            raise ValueError("v1 shock baseline is fixed at 180 days")
        if self.beta_lookback != 90:
            raise ValueError("v1 beta lookback is fixed at 90 days")
        if self.btc_ema_period != 200:
            raise ValueError("v1 BTC EMA is fixed at 200 days")
        if self.asset_ema_period != 200:
            raise ValueError("v1 asset EMA is fixed at 200 days")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")
        if self.position_weight != 0.20:
            raise ValueError("v1 position weight is fixed at 20%")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": BTC_SHOCK_DIFFUSION_FAMILY,
                "engine_version": BTC_SHOCK_DIFFUSION_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def btc_shock_diffusion_parameter_set(
) -> tuple[BTCShockDiffusionParameters, ...]:
    """Return the exact four-DNA preregistered family."""

    rows = tuple(
        BTCShockDiffusionParameters(
            shock_lookback=shock_lookback,
            maximum_holding_days=holding_days,
        )
        for shock_lookback, holding_days in product(
            (1, 3),
            (3, 5),
        )
    )
    if len(rows) != 4:
        raise RuntimeError("shock-diffusion family cardinality drift")
    if len({row.dna_hash for row in rows}) != len(rows):
        raise RuntimeError("shock-diffusion family contains duplicate DNA")
    return rows


@dataclass(frozen=True)
class BTCShockDiffusionResult:
    parameters: BTCShockDiffusionParameters
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
            "strategy_family": BTC_SHOCK_DIFFUSION_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": BTC_SHOCK_DIFFUSION_ENGINE_VERSION,
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


def _strictly_prior_betas(
    log_returns: pd.DataFrame,
    *,
    benchmark: str,
    lookback: int,
) -> pd.DataFrame:
    """Estimate satellite BTC betas without the current signal return."""

    prior = log_returns.shift(1)
    benchmark_returns = prior[benchmark]
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
        covariance = prior[market].rolling(
            lookback,
            min_periods=lookback,
        ).cov(benchmark_returns, ddof=0)
        betas[market] = covariance.div(
            variance.replace(0.0, np.nan)
        )
    return betas.replace([np.inf, -np.inf], np.nan)


def _shock_diffusion_scores(
    closes: pd.DataFrame,
    *,
    benchmark: str,
    parameters: BTCShockDiffusionParameters,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build causal beta-implied underreaction scores."""

    log_close = np.log(closes.where(closes > 0.0))
    log_returns = log_close.diff()
    betas = _strictly_prior_betas(
        log_returns,
        benchmark=benchmark,
        lookback=parameters.beta_lookback,
    )
    observed_moves = log_close.diff(parameters.shock_lookback)
    btc_shock = observed_moves[benchmark]
    prior_mean = btc_shock.rolling(
        parameters.shock_baseline_lookback,
        min_periods=parameters.shock_baseline_lookback,
    ).mean().shift(1)
    prior_std = btc_shock.rolling(
        parameters.shock_baseline_lookback,
        min_periods=parameters.shock_baseline_lookback,
    ).std(ddof=0).shift(1)
    shock_zscore = btc_shock.sub(prior_mean).div(
        prior_std.replace(0.0, np.nan)
    )
    implied_moves = betas.mul(btc_shock, axis=0)
    underreaction = implied_moves.sub(observed_moves)
    underreaction[benchmark] = np.nan
    return underreaction, shock_zscore, betas


def _stateful_targets(
    *,
    closes: pd.DataFrame,
    scores: pd.DataFrame,
    shock_zscore: pd.Series,
    eligible: pd.DataFrame,
    btc_regime: pd.Series,
    asset_regime: pd.DataFrame,
    parameters: BTCShockDiffusionParameters,
    benchmark: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    desired = pd.DataFrame(
        0.0,
        index=closes.index,
        columns=closes.columns,
    )
    remaining_days: dict[str, int] = {}
    event_rows: list[dict[str, Any]] = []
    satellites = [
        market for market in closes.columns if market != benchmark
    ]
    for timestamp in closes.index:
        exits: dict[str, str] = {}
        for market in list(remaining_days):
            if not bool(btc_regime.loc[timestamp]):
                exits[market] = "BTC_REGIME_LOSS"
                del remaining_days[market]
            elif not bool(asset_regime.loc[timestamp, market]):
                exits[market] = "ASSET_TREND_LOSS"
                del remaining_days[market]
            elif remaining_days[market] <= 0:
                exits[market] = "FIXED_HOLD_COMPLETE"
                del remaining_days[market]

        entries: list[str] = []
        shock_active = bool(
            btc_regime.loc[timestamp]
            and math.isfinite(float(shock_zscore.loc[timestamp]))
            and float(shock_zscore.loc[timestamp])
            >= parameters.shock_zscore_threshold
        )
        if shock_active:
            ranked = (
                scores.loc[timestamp, satellites]
                .where(eligible.loc[timestamp, satellites])
                .dropna()
            )
            ranked = ranked[ranked > 0.0].sort_values(
                ascending=False,
                kind="mergesort",
            )
            free_slots = (
                parameters.maximum_positions - len(remaining_days)
            )
            for market in ranked.index:
                market_name = str(market)
                if free_slots <= 0:
                    break
                if market_name in remaining_days:
                    continue
                remaining_days[market_name] = (
                    parameters.maximum_holding_days
                )
                entries.append(market_name)
                free_slots -= 1

        for market in remaining_days:
            desired.loc[timestamp, market] = (
                parameters.position_weight
            )
        if entries or exits:
            event_rows.append(
                {
                    "decision_at": timestamp,
                    "entry_markets": entries,
                    "exit_reasons": exits,
                    "btc_shock_zscore": (
                        float(shock_zscore.loc[timestamp])
                        if math.isfinite(
                            float(shock_zscore.loc[timestamp])
                        )
                        else None
                    ),
                }
            )
        for market in list(remaining_days):
            remaining_days[market] -= 1

    events = pd.DataFrame(
        event_rows,
        columns=[
            "decision_at",
            "entry_markets",
            "exit_reasons",
            "btc_shock_zscore",
        ],
    )
    return desired, events


def backtest_btc_shock_diffusion(
    frames: Mapping[str, pd.DataFrame],
    parameters: BTCShockDiffusionParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> BTCShockDiffusionResult:
    """Run one exact daily-close/next-open shock-diffusion path."""

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
    if set(closes.columns) != set(portfolio_policy.allowed_markets):
        raise ValueError("shock-diffusion universe must match allowlist")
    if len(closes.columns) < 4:
        raise ValueError(
            "shock diffusion requires BTC and three satellites"
        )
    warmup = max(
        parameters.shock_baseline_lookback
        + parameters.shock_lookback,
        parameters.beta_lookback + 1,
        parameters.btc_ema_period,
        parameters.asset_ema_period,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) <= warmup + 3:
        raise ValueError("insufficient shock-diffusion history")

    scores, shock_zscore, betas = _shock_diffusion_scores(
        closes,
        benchmark=benchmark,
        parameters=parameters,
    )
    btc_ema = closes[benchmark].ewm(
        span=parameters.btc_ema_period,
        adjust=False,
        min_periods=parameters.btc_ema_period,
    ).mean()
    asset_ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    btc_regime = closes[benchmark] > btc_ema
    asset_regime = closes > asset_ema
    history = closes.notna().cumsum()
    eligible = (
        (history >= portfolio_policy.minimum_history_observations)
        & closes.notna()
        & betas.notna()
        & scores.notna()
        & asset_regime
    )
    eligible[benchmark] = False
    desired, signal_events = _stateful_targets(
        closes=closes,
        scores=scores,
        shock_zscore=shock_zscore,
        eligible=eligible,
        btc_regime=btc_regime,
        asset_regime=asset_regime,
        parameters=parameters,
        benchmark=benchmark,
    )
    executed = desired.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)
    executed = executed.iloc[warmup + 1 :].copy()
    if executed.empty:
        raise ValueError("shock-diffusion evaluation window is empty")

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
            "held shock-diffusion asset lacks next valuation"
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
            "shock-diffusion costs consume full capital"
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
                "reason": "BTC_SHOCK_DIFFUSION_DAILY",
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
                "btc_shock_zscore": (
                    event["btc_shock_zscore"]
                    if event is not None
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
            "entry_markets",
            "exit_reasons",
            "btc_shock_zscore",
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
        "strictly_prior_shock_baseline": True,
        "btc_used_as_regime_and_information_only": bool(
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
        raise RuntimeError("shock-diffusion integrity failure")

    return BTCShockDiffusionResult(
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
            "shock_observation_count": int(
                (
                    shock_zscore
                    >= parameters.shock_zscore_threshold
                ).sum()
            ),
        },
    )


__all__ = [
    "BTC_SHOCK_DIFFUSION_ENGINE_VERSION",
    "BTC_SHOCK_DIFFUSION_FAMILY",
    "BTCShockDiffusionParameters",
    "BTCShockDiffusionResult",
    "backtest_btc_shock_diffusion",
    "btc_shock_diffusion_parameter_set",
    "residual_reversal_period_metrics",
]
