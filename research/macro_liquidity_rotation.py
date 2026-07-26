"""Causal FRED-only macro-liquidity rotation for allowed spot markets.

The family uses three slow, economically distinct liquidity votes:

* Federal Reserve balance-sheet expansion (WALCL, 13 releases);
* broad-money expansion (M2SL, 3 releases);
* easing financial conditions (NFCI, 4 releases).

Only source-reported ``available_at`` timestamps are accepted. Signals are
formed after a completed Sunday candle and execute at the following daily
open. The module has no order authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _validated_panel,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

MACRO_LIQUIDITY_ENGINE_VERSION = "1.0.0"
MACRO_LIQUIDITY_FAMILY = "FRED_LIQUIDITY_IMPULSE_RISK_ON_ROTATION"
DAILY_PERIODS_PER_YEAR = 365.25
REQUIRED_MACRO_STATUS = "SOURCE_AVAILABLE_AT"


@dataclass(frozen=True, slots=True)
class MacroLiquidityParameters:
    """Immutable DNA for one preregistered macro-consensus path."""

    minimum_positive_votes: int
    walcl_release_lookback: int = 13
    m2_release_lookback: int = 3
    nfci_release_lookback: int = 4
    asset_ema_period: int = 200
    position_weight: float = 0.10
    rebalance_weekday: int = 6

    def __post_init__(self) -> None:
        if self.minimum_positive_votes not in {2, 3}:
            raise ValueError("v1 macro consensus must require two or three votes")
        if self.walcl_release_lookback != 13:
            raise ValueError("v1 WALCL lookback is fixed at 13 releases")
        if self.m2_release_lookback != 3:
            raise ValueError("v1 M2 lookback is fixed at three releases")
        if self.nfci_release_lookback != 4:
            raise ValueError("v1 NFCI lookback is fixed at four releases")
        if self.asset_ema_period != 200:
            raise ValueError("v1 asset EMA is fixed at 200 days")
        if self.position_weight != 0.10:
            raise ValueError("v1 position weight is fixed at 10%")
        if self.rebalance_weekday != 6:
            raise ValueError("v1 rebalance decision is fixed to Sunday")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": MACRO_LIQUIDITY_FAMILY,
                "engine_version": MACRO_LIQUIDITY_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def macro_liquidity_parameter_set() -> tuple[MacroLiquidityParameters, ...]:
    """Return the exact two-DNA preregistered family."""

    rows = tuple(
        MacroLiquidityParameters(minimum_positive_votes=votes)
        for votes in (2, 3)
    )
    if len(rows) != 2 or len({row.dna_hash for row in rows}) != 2:
        raise RuntimeError("macro-liquidity DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class MacroLiquidityResult:
    parameters: MacroLiquidityParameters
    portfolio_policy: RotationPortfolioPolicy
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    macro_votes: pd.DataFrame
    signal_diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_family": MACRO_LIQUIDITY_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": MACRO_LIQUIDITY_ENGINE_VERSION,
            "timeframe": "1d",
            "periods_per_year": DAILY_PERIODS_PER_YEAR,
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
            "signal_diagnostics": dict(self.signal_diagnostics),
        }


def _release_impulse(
    frame: pd.DataFrame,
    *,
    series_id: str,
    release_lookback: int,
    direction: str,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    required = {
        "source_symbol",
        "available_at",
        "observation_time",
        "point_in_time_status",
        "value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{series_id} missing macro columns: {sorted(missing)}")
    rows = frame.copy()
    if set(rows["source_symbol"].astype(str).str.upper()) != {series_id}:
        raise ValueError(f"unexpected macro series in {series_id} source")
    statuses = set(rows["point_in_time_status"].astype(str))
    if statuses != {REQUIRED_MACRO_STATUS}:
        raise ValueError(
            f"{series_id} requires {REQUIRED_MACRO_STATUS}, got {sorted(statuses)}"
        )
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    rows["observation_time"] = pd.to_datetime(
        rows["observation_time"],
        utc=True,
    )
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    rows = (
        rows.sort_values(["available_at", "observation_time"])
        .drop_duplicates("available_at", keep="last")
    )
    values = pd.Series(
        rows["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(rows["available_at"]),
        name=series_id,
    ).sort_index()
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError(f"{series_id} release index is invalid")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{series_id} contains non-finite values")
    change = values.pct_change(release_lookback)
    if direction == "DOWN":
        change = values.diff(release_lookback).mul(-1.0)
    elif direction != "UP":
        raise ValueError(f"unsupported macro direction: {direction}")
    vote = change.gt(0.0).where(change.notna())
    aligned = vote.reindex(vote.index.union(target_index)).sort_index().ffill()
    return aligned.reindex(target_index).astype("boolean")


def build_macro_liquidity_votes(
    macro_frames: Mapping[str, pd.DataFrame],
    *,
    target_index: pd.DatetimeIndex,
    parameters: MacroLiquidityParameters,
) -> pd.DataFrame:
    """Build three causal votes aligned only from known release timestamps."""

    normalized = {str(key).upper(): value for key, value in macro_frames.items()}
    required = {"WALCL", "M2SL", "NFCI"}
    missing = required.difference(normalized)
    if missing:
        raise ValueError(f"missing macro series: {sorted(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("target index must be unique and sorted")
    votes = pd.DataFrame(
        {
            "walcl_expanding": _release_impulse(
                normalized["WALCL"],
                series_id="WALCL",
                release_lookback=parameters.walcl_release_lookback,
                direction="UP",
                target_index=index,
            ),
            "m2_expanding": _release_impulse(
                normalized["M2SL"],
                series_id="M2SL",
                release_lookback=parameters.m2_release_lookback,
                direction="UP",
                target_index=index,
            ),
            "nfci_easing": _release_impulse(
                normalized["NFCI"],
                series_id="NFCI",
                release_lookback=parameters.nfci_release_lookback,
                direction="DOWN",
                target_index=index,
            ),
        },
        index=index,
    )
    votes["ready"] = votes.notna().all(axis=1)
    votes["positive_vote_count"] = (
        votes.iloc[:, :3].fillna(False).astype(int).sum(axis=1)
    )
    votes["macro_risk_on"] = (
        votes["ready"]
        & votes["positive_vote_count"].ge(parameters.minimum_positive_votes)
    )
    return votes


def backtest_macro_liquidity_rotation(
    frames: Mapping[str, pd.DataFrame],
    macro_frames: Mapping[str, pd.DataFrame],
    parameters: MacroLiquidityParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> MacroLiquidityResult:
    """Run one exact Sunday-close/Monday-open macro-liquidity path."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=portfolio_policy,
    )
    if set(closes.columns) != set(portfolio_policy.allowed_markets):
        raise ValueError("macro-liquidity universe must match allowlist")
    if parameters.position_weight > portfolio_policy.maximum_position_exposure:
        raise ValueError("macro-liquidity position weight exceeds policy")

    votes = build_macro_liquidity_votes(
        macro_frames,
        target_index=closes.index,
        parameters=parameters,
    )
    ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    history = closes.notna().cumsum()
    eligible = (
        closes.gt(ema)
        & closes.notna()
        & history.ge(portfolio_policy.minimum_history_observations)
    )
    weekly = closes.index.dayofweek == parameters.rebalance_weekday
    desired = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for timestamp in closes.index[weekly]:
        if not bool(votes.at[timestamp, "macro_risk_on"]):
            continue
        active = eligible.loc[timestamp]
        desired.loc[timestamp, active] = parameters.position_weight
    desired = desired.where(
        pd.Series(weekly, index=closes.index),
        np.nan,
    ).ffill().fillna(0.0)
    desired = desired.clip(lower=0.0)
    desired_total = desired.sum(axis=1)
    if bool(
        (
            desired_total
            > portfolio_policy.maximum_total_exposure + 1e-12
        ).any()
    ):
        raise RuntimeError("macro-liquidity desired exposure exceeds policy")

    executed = desired.shift(1).fillna(0.0)
    executed = executed.where(opens.notna(), 0.0)
    ready_positions = np.flatnonzero(
        (votes["ready"] & pd.Series(weekly, index=closes.index)).to_numpy()
    )
    if not len(ready_positions):
        raise ValueError("macro-liquidity sources never become ready")
    start_position = int(ready_positions[0]) + 1
    executed = executed.iloc[start_position:].copy()
    if executed.empty:
        raise ValueError("macro-liquidity evaluation window is empty")

    open_returns = opens.shift(-1).div(opens).sub(1.0)
    open_returns.iloc[-1] = closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    open_returns = open_returns.reindex(executed.index)
    held_missing = open_returns.isna() & executed.gt(1e-12)
    if bool(held_missing.any().any()):
        raise ValueError("held macro-liquidity asset lacks next valuation")
    gross_returns = (executed * open_returns.fillna(0.0)).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(executed.iloc[0].sum())
    turnover.iloc[-1] += float(executed.iloc[-1].sum())
    one_way_cost = (
        fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    )
    cost_fraction = turnover * one_way_cost
    if bool((cost_fraction >= 1.0).any()):
        raise ValueError("macro-liquidity costs consume full capital")
    net_returns = (1.0 - cost_fraction) * (1.0 + gross_returns) - 1.0
    equity = (1.0 + net_returns).cumprod().rename("equity")
    gross_equity = (1.0 + gross_returns).cumprod().rename("gross_equity")

    decisions: list[dict[str, Any]] = []
    for position, executed_at in enumerate(executed.index):
        execution_turnover = float(turnover.iloc[position])
        if execution_turnover <= 1e-12:
            continue
        signal_position = closes.index.get_loc(executed_at) - 1
        decision_at = closes.index[signal_position]
        target = executed.iloc[position]
        decisions.append(
            {
                "decision_at": decision_at,
                "executed_at": executed_at,
                "reason": "FRED_MACRO_LIQUIDITY_WEEKLY",
                "turnover": execution_turnover,
                "expected_cost_fraction": execution_turnover * one_way_cost,
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "positive_vote_count": int(
                    votes.at[decision_at, "positive_vote_count"]
                ),
                "macro_risk_on": bool(
                    votes.at[decision_at, "macro_risk_on"]
                ),
            }
        )
    decision_frame = pd.DataFrame(decisions)
    metrics = _performance_metrics(equity, executed, decision_frame)
    exposure = executed.sum(axis=1)
    macro_statuses = {
        str(key).upper(): sorted(
            set(value["point_in_time_status"].astype(str))
        )
        for key, value in macro_frames.items()
    }
    integrity = {
        "allowed_markets_only": set(executed.columns)
        <= set(portfolio_policy.allowed_markets),
        "source_available_at_only": all(
            statuses == [REQUIRED_MACRO_STATUS]
            for statuses in macro_statuses.values()
        ),
        "forward_only_sources_rejected": True,
        "macro_alignment_backward_only": True,
        "weekly_sunday_close_decision": True,
        "decision_at_close_execution_next_open": True,
        "maximum_exposure_respected": bool(
            exposure.max()
            <= portfolio_policy.maximum_total_exposure + 1e-12
        ),
        "maximum_position_exposure_respected": bool(
            executed.max(axis=1).max()
            <= portfolio_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": bool(
            exposure.max() <= 1.0 - portfolio_policy.minimum_cash + 1e-12
        ),
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "cash_yield_zero": True,
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
        raise RuntimeError("macro-liquidity integrity failure")

    return MacroLiquidityResult(
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
        macro_votes=votes.reindex(executed.index),
        signal_diagnostics={
            "evaluation_start": executed.index.min().isoformat(),
            "evaluation_end": executed.index.max().isoformat(),
            "macro_ready_days": int(votes["ready"].sum()),
            "macro_risk_on_days": int(votes["macro_risk_on"].sum()),
            "weekly_signal_count": int(weekly.sum()),
            "decision_count": int(len(decision_frame)),
            "macro_statuses": macro_statuses,
        },
    )


__all__ = [
    "MACRO_LIQUIDITY_ENGINE_VERSION",
    "MACRO_LIQUIDITY_FAMILY",
    "MacroLiquidityParameters",
    "MacroLiquidityResult",
    "backtest_macro_liquidity_rotation",
    "build_macro_liquidity_votes",
    "macro_liquidity_parameter_set",
]
