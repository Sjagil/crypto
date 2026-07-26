"""Causal confirmed-fractal liquidity-sweep recovery portfolio research."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.features import confirmed_fractals
from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _profit_factor,
    _validated_panel,
)
from research.volatility_contraction import _performance_metrics
from utils.common import stable_hash

LIQUIDITY_SWEEP_ENGINE_VERSION = "1.0.0"
LIQUIDITY_SWEEP_FAMILY = (
    "CONFIRMED_FRACTAL_LIQUIDITY_SWEEP_RECOVERY"
)
DAILY_PERIODS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class LiquiditySweepParameters:
    """One immutable member of the preregistered v1 family."""

    fractal_side: int
    minimum_relative_volume: float
    maximum_holding_days: int
    volume_lookback: int = 20
    trend_ema_period: int = 200
    maximum_positions: int = 2
    position_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.fractal_side not in {2, 3}:
            raise ValueError("v1 fractal side must be two or three")
        if self.minimum_relative_volume not in {1.0, 1.5}:
            raise ValueError("v1 relative-volume floor must be 1.0 or 1.5")
        if self.maximum_holding_days not in {10, 20}:
            raise ValueError("v1 holding horizon must be 10 or 20 days")
        if self.volume_lookback != 20:
            raise ValueError("v1 volume lookback is fixed at 20")
        if self.trend_ema_period != 200:
            raise ValueError("v1 trend EMA is fixed at 200")
        if self.maximum_positions != 2:
            raise ValueError("v1 maximum positions is fixed at two")
        if self.position_weight != 0.20:
            raise ValueError("v1 position weight is fixed at 20%")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": LIQUIDITY_SWEEP_FAMILY,
                "engine_version": LIQUIDITY_SWEEP_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )


def liquidity_sweep_parameter_set(
) -> tuple[LiquiditySweepParameters, ...]:
    """Return the exact eight preregistered v1 strategy DNA rows."""

    rows = tuple(
        LiquiditySweepParameters(
            fractal_side=side,
            minimum_relative_volume=volume,
            maximum_holding_days=holding,
        )
        for side in (2, 3)
        for volume in (1.0, 1.5)
        for holding in (10, 20)
    )
    if len(rows) != 8 or len({row.dna_hash for row in rows}) != 8:
        raise RuntimeError("liquidity-sweep DNA cardinality drift")
    return rows


@dataclass(frozen=True)
class LiquiditySweepResult:
    parameters: LiquiditySweepParameters
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
            "strategy_family": LIQUIDITY_SWEEP_FAMILY,
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "engine_version": LIQUIDITY_SWEEP_ENGINE_VERSION,
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


def _ohlcv_panels(
    frames: Mapping[str, pd.DataFrame],
    *,
    benchmark_market: str,
    portfolio_policy: RotationPortfolioPolicy,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark_market,
        portfolio_policy=portfolio_policy,
    )
    benchmark_index = closes.index
    normalized = {
        raw_market.upper().replace("/", "-").replace("_", "-"): frame
        for raw_market, frame in frames.items()
    }
    panels: dict[str, pd.DataFrame] = {}
    for field in ("high", "low", "volume"):
        values: dict[str, pd.Series] = {}
        for market in closes.columns:
            frame = normalized[market]
            if field not in frame:
                raise ValueError(
                    f"{market} is missing required field: {field}"
                )
            series = pd.to_numeric(frame[field], errors="coerce").copy()
            series.index = pd.to_datetime(series.index, utc=True)
            series = (
                series[~series.index.duplicated(keep="last")]
                .sort_index()
                .reindex(benchmark_index)
            )
            values[market] = series
        panel = pd.DataFrame(values, index=benchmark_index, dtype=float)
        finite = panel.stack().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"non-finite {field} values")
        if (finite <= 0.0).any():
            raise ValueError(f"non-positive {field} values")
        panels[field] = panel
    if bool((panels["high"] < panels["low"]).any().any()):
        raise ValueError("daily high is below daily low")
    return (
        opens,
        closes,
        panels["high"],
        panels["low"],
        panels["volume"],
    )


def _signal_matrices(
    frames: Mapping[str, pd.DataFrame],
    *,
    parameters: LiquiditySweepParameters,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str,
) -> dict[str, pd.DataFrame | pd.Series]:
    opens, closes, highs, lows, volumes = _ohlcv_panels(
        frames,
        benchmark_market=benchmark_market,
        portfolio_policy=portfolio_policy,
    )
    previous_high = pd.DataFrame(
        np.nan,
        index=closes.index,
        columns=closes.columns,
    )
    previous_low = previous_high.copy()
    for market in closes.columns:
        frame = pd.DataFrame(
            {
                "open": opens[market],
                "high": highs[market],
                "low": lows[market],
                "close": closes[market],
                "volume": volumes[market],
            },
            index=closes.index,
        )
        fractals = confirmed_fractals(
            frame,
            left=parameters.fractal_side,
            right=parameters.fractal_side,
        )
        previous_high[market] = (
            fractals["confirmed_fractal_high_price"]
            .ffill()
            .shift(1)
        )
        previous_low[market] = (
            fractals["confirmed_fractal_low_price"]
            .ffill()
            .shift(1)
        )
    ema = closes.ewm(
        span=parameters.trend_ema_period,
        adjust=False,
        min_periods=parameters.trend_ema_period,
    ).mean()
    relative_volume = volumes.div(
        volumes.rolling(
            parameters.volume_lookback,
            min_periods=parameters.volume_lookback,
        ).mean()
    )
    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    btc_regime = closes[benchmark] > ema[benchmark]
    history = closes.notna().cumsum()
    valid = (
        (history >= portfolio_policy.minimum_history_observations)
        & previous_low.notna()
        & ema.notna()
        & relative_volume.notna()
    )
    bullish_sweep = (
        (lows < previous_low)
        & (closes > previous_low)
        & (opens >= previous_low)
    )
    bearish_sweep = (
        (highs > previous_high)
        & (closes < previous_high)
        & (opens <= previous_high)
    )
    entry = (
        valid
        & bullish_sweep
        & (relative_volume >= parameters.minimum_relative_volume)
        & (closes > ema)
    )
    entry = entry.mul(btc_regime, axis=0).astype(bool)
    structural_exit = (
        bearish_sweep
        | (closes < ema)
        | (closes >= previous_high)
    )
    recovery = (
        (closes - previous_low)
        .div((highs - lows).replace(0.0, np.nan))
        .clip(lower=0.0)
    )
    strength = recovery * relative_volume
    return {
        "opens": opens,
        "closes": closes,
        "entry": entry,
        "structural_exit": structural_exit,
        "strength": strength,
        "relative_volume": relative_volume,
        "btc_regime": btc_regime,
    }


def _desired_weights(
    *,
    entry: pd.DataFrame,
    structural_exit: pd.DataFrame,
    strength: pd.DataFrame,
    parameters: LiquiditySweepParameters,
    policy: RotationPortfolioPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    desired = pd.DataFrame(
        0.0,
        index=entry.index,
        columns=entry.columns,
    )
    held: dict[str, int] = {}
    event_rows: list[dict[str, Any]] = []
    for position, timestamp in enumerate(entry.index):
        exits: list[str] = []
        exit_reasons: dict[str, str] = {}
        for market, entered_at in tuple(held.items()):
            age = position - entered_at
            if bool(structural_exit.loc[timestamp, market]):
                exits.append(market)
                exit_reasons[market] = "STRUCTURAL_OR_TREND_EXIT"
            elif age >= parameters.maximum_holding_days:
                exits.append(market)
                exit_reasons[market] = "MAXIMUM_HOLDING_DAYS"
        for market in exits:
            held.pop(market, None)

        candidates = (
            strength.loc[timestamp]
            .where(entry.loc[timestamp])
            .dropna()
            .sort_values(ascending=False)
        )
        entered: list[str] = []
        for market in candidates.index:
            if market in held:
                continue
            if len(held) >= parameters.maximum_positions:
                break
            held[str(market)] = position
            entered.append(str(market))
        for market in held:
            desired.loc[timestamp, market] = parameters.position_weight
        if entered or exits:
            event_rows.append(
                {
                    "decision_at": timestamp,
                    "entered_assets": entered,
                    "exited_assets": exits,
                    "exit_reasons": exit_reasons,
                    "selected_assets": sorted(held),
                    "signal_strength": {
                        str(market): float(value)
                        for market, value in candidates.items()
                    },
                }
            )
    return desired, pd.DataFrame(event_rows)


def backtest_liquidity_sweep(
    frames: Mapping[str, pd.DataFrame],
    parameters: LiquiditySweepParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    portfolio_policy: RotationPortfolioPolicy,
    benchmark_market: str = "BTC-EUR",
) -> LiquiditySweepResult:
    """Run one exact close-signal/next-open liquidity-sweep path."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    signals = _signal_matrices(
        frames,
        parameters=parameters,
        portfolio_policy=portfolio_policy,
        benchmark_market=benchmark_market,
    )
    opens = signals["opens"]
    closes = signals["closes"]
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    desired, events = _desired_weights(
        entry=signals["entry"],
        structural_exit=signals["structural_exit"],
        strength=signals["strength"],
        parameters=parameters,
        policy=portfolio_policy,
    )
    warmup = max(
        parameters.trend_ema_period,
        parameters.volume_lookback,
        portfolio_policy.minimum_history_observations,
    )
    executed = desired.shift(1).fillna(0.0).iloc[warmup + 1 :].copy()
    executed = executed.where(opens.reindex(executed.index).notna(), 0.0)
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
        raise ValueError("held liquidity-sweep position lacks valuation")
    gross_returns = (
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
        raise ValueError("liquidity-sweep costs exhaust equity")
    net_returns = (
        (1.0 - cost_fraction) * (1.0 + gross_returns) - 1.0
    )
    gross_equity = (1.0 + gross_returns).cumprod()
    equity = (1.0 + net_returns).cumprod()
    gross_equity.name = "gross_equity"
    equity.name = "equity"

    decision_rows: list[dict[str, Any]] = []
    for position, executed_at in enumerate(executed.index):
        changed = float(turnover.iloc[position])
        if changed <= 1e-12:
            continue
        signal_position = max(0, closes.index.get_loc(executed_at) - 1)
        decision_at = closes.index[signal_position]
        matching = (
            events.loc[events["decision_at"] == decision_at]
            if not events.empty
            else pd.DataFrame()
        )
        metadata = (
            matching.iloc[-1].to_dict()
            if not matching.empty
            else {}
        )
        target = executed.iloc[position]
        decision_rows.append(
            {
                "decision_at": decision_at,
                "executed_at": executed_at,
                "reason": "LIQUIDITY_SWEEP_ENTRY_OR_EXIT",
                "turnover": changed,
                "expected_cost_fraction": changed * one_way_cost,
                "target_weights": {
                    market: float(weight)
                    for market, weight in target.items()
                    if float(weight) > 1e-12
                },
                "cash_fraction": float(1.0 - target.sum()),
                "entered_assets": metadata.get("entered_assets", []),
                "exited_assets": metadata.get("exited_assets", []),
                "exit_reasons": metadata.get("exit_reasons", {}),
                "btc_regime_positive": bool(
                    signals["btc_regime"].iloc[signal_position]
                ),
            }
        )
    decisions = pd.DataFrame(decision_rows)
    metrics = _performance_metrics(equity, executed, decisions)
    exposure = executed.sum(axis=1)
    entry = signals["entry"]
    assert isinstance(entry, pd.DataFrame)
    integrity = {
        "allowed_markets_only": set(executed.columns)
        == set(portfolio_policy.allowed_markets),
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
        "confirmed_fractals_only": True,
        "fractal_confirmation_lag_bars": parameters.fractal_side,
        "closed_candles_only": True,
        "decision_at_close_execution_next_open": True,
        "point_in_time_history_gate": True,
        "volume_known_at_decision_close": True,
        "long_only_spot": bool((executed >= -1e-12).all().all()),
        "orders_generated": 0,
    }
    if not all(
        value
        for key, value in integrity.items()
        if key not in {"orders_generated", "fractal_confirmation_lag_bars"}
    ):
        raise RuntimeError(
            f"liquidity-sweep integrity failure: {integrity}"
        )
    return LiquiditySweepResult(
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
            "entry_signal_count": int(entry.sum().sum()),
            "decision_count": int(len(decisions)),
            "relative_volume_floor": (
                parameters.minimum_relative_volume
            ),
            "fractal_window": parameters.fractal_side * 2 + 1,
            "maximum_holding_days": parameters.maximum_holding_days,
        },
    )


def liquidity_sweep_period_metrics(
    equity: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, Any], pd.Series]:
    """Return daily metrics for one fixed calendar period."""

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
        (selected.index[-1] - selected.index[0]).total_seconds()
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
    "LIQUIDITY_SWEEP_ENGINE_VERSION",
    "LIQUIDITY_SWEEP_FAMILY",
    "LiquiditySweepParameters",
    "LiquiditySweepResult",
    "backtest_liquidity_sweep",
    "liquidity_sweep_parameter_set",
    "liquidity_sweep_period_metrics",
]
