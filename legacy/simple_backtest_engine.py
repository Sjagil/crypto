
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


ParameterKind = Literal["integer", "half", "float", "choice"]


@dataclass(frozen=True)
class ParameterSpec:
    """
    Defines one optimizable strategy parameter.

    kind="integer":
        Uses whole-number values only.

    kind="half":
        Uses exact 0.5 increments. Decimal arithmetic prevents values such as
        14.4999999998.

    kind="float":
        Uses the supplied step.

    kind="choice":
        Uses the values supplied through choices.
    """
    name: str
    kind: ParameterKind
    start: float | int | None = None
    stop: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()

    def values(self) -> list[Any]:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"{self.name}: choices may not be empty")
            return list(self.choices)

        if self.start is None or self.stop is None:
            raise ValueError(f"{self.name}: start and stop are required")

        if self.kind == "integer":
            step = int(self.step or 1)
            if step <= 0:
                raise ValueError(f"{self.name}: step must be positive")
            return list(range(int(self.start), int(self.stop) + 1, step))

        effective_step = Decimal("0.5") if self.kind == "half" else Decimal(str(self.step))
        if effective_step <= 0:
            raise ValueError(f"{self.name}: step must be positive")

        current = Decimal(str(self.start))
        stop = Decimal(str(self.stop))
        result: list[float] = []
        while current <= stop:
            result.append(float(current))
            current += effective_step
        return result


@dataclass(frozen=True)
class CostModel:
    fee_pct_per_side: float = 0.0010
    slippage_pct_per_side: float = 0.0005

    @property
    def one_way_cost(self) -> float:
        return self.fee_pct_per_side + self.slippage_pct_per_side


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    position_fraction: float = 1.0
    risk_fraction_per_trade: float | None = 0.01
    costs: CostModel = field(default_factory=CostModel)
    allow_fractional_units: bool = True
    intrabar_priority: Literal["stop_first", "target_first"] = "stop_first"
    annualization_days: float = 365.25


@dataclass
class StrategySignals:
    """
    Signals are calculated on CLOSED candles.

    entry:
        True on candle t means enter on candle t+1 open.

    exit:
        True on candle t means exit on candle t+1 open.

    stop_pct / target_pct:
        Fractions from entry price, for example 0.025 means 2.5%.
    """
    entry: pd.Series
    exit: pd.Series
    stop_pct: float | pd.Series
    target_pct: float | pd.Series


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    units: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    reason: str
    bars_held: int


@dataclass
class BacktestResult:
    parameters: dict[str, Any]
    metrics: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.Series


StrategyFunction = Callable[[pd.DataFrame, Mapping[str, Any]], StrategySignals]
ConstraintFunction = Callable[[Mapping[str, Any]], bool]


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns.str.lower())
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = data.copy()
    frame.columns = [str(column).lower() for column in frame.columns]

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a pandas DatetimeIndex")

    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Datetime index contains duplicate timestamps")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric_columns)

    invalid = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame[numeric_columns[:4]] <= 0).any(axis=1)
        | (frame["volume"] < 0)
    )
    if invalid.any():
        bad_rows = int(invalid.sum())
        raise ValueError(f"OHLCV validation failed for {bad_rows} rows")

    return frame


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: float) -> pd.Series:
    """
    Generalized EMA. Fractional periods are mathematically defined through
    alpha = 2 / (period + 1).
    """
    if period <= 0:
        raise ValueError("EMA period must be positive")
    alpha = 2.0 / (float(period) + 1.0)
    return series.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=max(1, math.ceil(period)),
    ).mean()


def wilder_ema(series: pd.Series, period: float) -> pd.Series:
    """
    Wilder smoothing with alpha = 1 / period.
    Fractional periods are therefore explicitly defined.
    """
    if period <= 0:
        raise ValueError("Wilder period must be positive")
    alpha = 1.0 / float(period)
    return series.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=max(1, math.ceil(period)),
    ).mean()


def rsi(series: pd.Series, period: float = 14.0) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    average_gain = wilder_ema(gain, period)
    average_loss = wilder_ema(loss, period)

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    output = 100.0 - (100.0 / (1.0 + relative_strength))

    output = output.mask((average_loss == 0) & (average_gain > 0), 100.0)
    output = output.mask((average_gain == 0) & (average_loss > 0), 0.0)
    output = output.mask((average_gain == 0) & (average_loss == 0), 50.0)
    return output


def true_range(data: pd.DataFrame) -> pd.Series:
    previous_close = data["close"].shift(1)
    components = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


def atr(data: pd.DataFrame, period: float = 14.0) -> pd.Series:
    return wilder_ema(true_range(data), period)


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    SMA uses a fixed number of observations and therefore requires an integer.
    Do not silently pretend that an SMA(14.5) has a canonical meaning.
    """
    if not isinstance(period, (int, np.integer)):
        raise TypeError("SMA period must be an integer")
    if period <= 0:
        raise ValueError("SMA period must be positive")
    return series.rolling(period, min_periods=period).mean()


def fractional_sma_interpolated(series: pd.Series, period: float) -> pd.Series:
    """
    Optional experimental definition for fractional SMA periods.

    SMA(14.5) = 50% * SMA(14) + 50% * SMA(15).

    This is explicit and reproducible, but it is NOT the standard SMA.
    Use only when your research hypothesis intentionally requires it.
    """
    if period <= 0:
        raise ValueError("SMA period must be positive")
    lower = math.floor(period)
    upper = math.ceil(period)
    if lower == upper:
        return sma(series, lower)
    weight_upper = period - lower
    return (1.0 - weight_upper) * sma(series, lower) + weight_upper * sma(series, upper)


def crossed_above(left: pd.Series, right: pd.Series | float) -> pd.Series:
    if isinstance(right, pd.Series):
        return (left > right) & (left.shift(1) <= right.shift(1))
    return (left > right) & (left.shift(1) <= right)


def crossed_below(left: pd.Series, right: pd.Series | float) -> pd.Series:
    if isinstance(right, pd.Series):
        return (left < right) & (left.shift(1) >= right.shift(1))
    return (left < right) & (left.shift(1) >= right)


# ---------------------------------------------------------------------------
# Example strategy
# ---------------------------------------------------------------------------

def ema_rsi_pullback_strategy(
    data: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> StrategySignals:
    """
    Long-only example:
      - Trend: fast EMA above slow EMA and close above slow EMA.
      - Entry: RSI crosses above an entry threshold after weakness.
      - Exit: RSI reaches an exit threshold or fast EMA falls below slow EMA.
      - Stop and target are percentage based.

    Fractional EMA and RSI periods are deliberately supported.
    """
    close = data["close"]

    fast = ema(close, float(parameters["fast_ema"]))
    slow = ema(close, float(parameters["slow_ema"]))
    momentum = rsi(close, float(parameters["rsi_period"]))

    trend = (fast > slow) & (close > slow)
    entry = trend & crossed_above(momentum, float(parameters["rsi_entry"]))
    exit_signal = (
        crossed_above(momentum, float(parameters["rsi_exit"]))
        | crossed_below(fast, slow)
    )

    return StrategySignals(
        entry=entry.fillna(False),
        exit=exit_signal.fillna(False),
        stop_pct=float(parameters["stop_pct"]) / 100.0,
        target_pct=float(parameters["target_pct"]) / 100.0,
    )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        strategy: StrategyFunction,
        parameters: Mapping[str, Any],
    ) -> BacktestResult:
        frame = validate_ohlcv(data)
        signals = strategy(frame, parameters)
        self._validate_signals(frame, signals)

        cash = float(self.config.initial_cash)
        units = 0.0
        entry_price = math.nan
        entry_time: pd.Timestamp | None = None
        entry_cost_basis = 0.0
        entry_fee = 0.0
        stop_price = math.nan
        target_price = math.nan
        bars_held = 0

        trades: list[Trade] = []
        equity_values: list[float] = [cash]
        equity_index: list[pd.Timestamp] = [frame.index[0]]
        bars_in_market = 0

        pending_entry = False
        pending_exit = False

        for i in range(1, len(frame)):
            timestamp = frame.index[i]
            row = frame.iloc[i]
            previous_i = i - 1

            # Signals from candle t-1 execute at candle t open.
            previous_entry_signal = bool(signals.entry.iloc[previous_i])
            previous_exit_signal = bool(signals.exit.iloc[previous_i])

            if units > 0:
                bars_in_market += 1
                bars_held += 1

                if previous_exit_signal:
                    cash, trade = self._close_position(
                        cash=cash,
                        units=units,
                        raw_exit_price=float(row["open"]),
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_time=timestamp,
                        entry_cost_basis=entry_cost_basis,
                        entry_fee=entry_fee,
                        bars_held=bars_held,
                        reason="signal_exit",
                    )
                    trades.append(trade)
                    units = 0.0

            if units == 0 and previous_entry_signal and not previous_exit_signal:
                raw_entry = float(row["open"])
                fill_entry = raw_entry * (1.0 + self.config.costs.slippage_pct_per_side)

                stop_fraction = self._value_at(signals.stop_pct, previous_i)
                target_fraction = self._value_at(signals.target_pct, previous_i)

                if not (0 < stop_fraction < 1):
                    raise ValueError(f"Invalid stop_pct at {frame.index[previous_i]}: {stop_fraction}")
                if target_fraction <= 0:
                    raise ValueError(
                        f"Invalid target_pct at {frame.index[previous_i]}: {target_fraction}"
                    )

                allocation_cap = cash * self.config.position_fraction
                if self.config.risk_fraction_per_trade is None:
                    allocation = allocation_cap
                else:
                    risk_budget = cash * self.config.risk_fraction_per_trade
                    risk_based_allocation = risk_budget / stop_fraction
                    allocation = min(allocation_cap, risk_based_allocation)

                entry_fee = allocation * self.config.costs.fee_pct_per_side
                spendable = allocation - entry_fee

                units = spendable / fill_entry
                if not self.config.allow_fractional_units:
                    units = math.floor(units)

                if units > 0:
                    entry_cost_basis = units * fill_entry
                    cash -= entry_cost_basis + entry_fee
                    entry_price = fill_entry
                    entry_time = timestamp
                    stop_price = entry_price * (1.0 - stop_fraction)
                    target_price = entry_price * (1.0 + target_fraction)
                    bars_held = 0

            # Intrabar stop/target handling for positions entered earlier or at open.
            if units > 0:
                low_hit = float(row["low"]) <= stop_price
                high_hit = float(row["high"]) >= target_price

                exit_reason: str | None = None
                raw_exit_price: float | None = None

                # Gap handling at the current open.
                if float(row["open"]) <= stop_price:
                    exit_reason = "gap_stop"
                    raw_exit_price = float(row["open"])
                elif float(row["open"]) >= target_price:
                    exit_reason = "gap_target"
                    # Conservative limit-target assumption: fill at target.
                    raw_exit_price = target_price
                elif low_hit and high_hit:
                    if self.config.intrabar_priority == "stop_first":
                        exit_reason = "stop_and_target_same_bar_stop_first"
                        raw_exit_price = stop_price
                    else:
                        exit_reason = "stop_and_target_same_bar_target_first"
                        raw_exit_price = target_price
                elif low_hit:
                    exit_reason = "stop"
                    raw_exit_price = stop_price
                elif high_hit:
                    exit_reason = "target"
                    raw_exit_price = target_price

                if raw_exit_price is not None and exit_reason is not None:
                    cash, trade = self._close_position(
                        cash=cash,
                        units=units,
                        raw_exit_price=raw_exit_price,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        exit_time=timestamp,
                        entry_cost_basis=entry_cost_basis,
                        entry_fee=entry_fee,
                        bars_held=bars_held,
                        reason=exit_reason,
                    )
                    trades.append(trade)
                    units = 0.0

            mark_to_market = cash + units * float(row["close"])
            equity_values.append(mark_to_market)
            equity_index.append(timestamp)

        # Liquidate at final close.
        if units > 0:
            timestamp = frame.index[-1]
            cash, trade = self._close_position(
                cash=cash,
                units=units,
                raw_exit_price=float(frame["close"].iloc[-1]),
                entry_price=entry_price,
                entry_time=entry_time,
                exit_time=timestamp,
                entry_cost_basis=entry_cost_basis,
                entry_fee=entry_fee,
                bars_held=bars_held,
                reason="end_of_data",
            )
            trades.append(trade)
            equity_values[-1] = cash

        equity_curve = pd.Series(
            equity_values,
            index=pd.DatetimeIndex(equity_index),
            name="equity",
            dtype=float,
        )
        trades_frame = pd.DataFrame([trade.__dict__ for trade in trades])
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=trades_frame,
            bars_in_market=bars_in_market,
            total_bars=max(1, len(frame) - 1),
            annualization_days=self.config.annualization_days,
        )

        return BacktestResult(
            parameters=dict(parameters),
            metrics=metrics,
            trades=trades_frame,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _value_at(value: float | pd.Series, index: int) -> float:
        if isinstance(value, pd.Series):
            return float(value.iloc[index])
        return float(value)

    @staticmethod
    def _validate_signals(data: pd.DataFrame, signals: StrategySignals) -> None:
        for name, series in (("entry", signals.entry), ("exit", signals.exit)):
            if not isinstance(series, pd.Series):
                raise TypeError(f"{name} signal must be a pandas Series")
            if not series.index.equals(data.index):
                raise ValueError(f"{name} signal index does not match OHLCV index")

    def _close_position(
        self,
        *,
        cash: float,
        units: float,
        raw_exit_price: float,
        entry_price: float,
        entry_time: pd.Timestamp | None,
        exit_time: pd.Timestamp,
        entry_cost_basis: float,
        entry_fee: float,
        bars_held: int,
        reason: str,
    ) -> tuple[float, Trade]:
        if entry_time is None:
            raise RuntimeError("Missing entry_time for an open position")

        fill_exit = raw_exit_price * (1.0 - self.config.costs.slippage_pct_per_side)
        gross_proceeds = units * fill_exit
        exit_fee = gross_proceeds * self.config.costs.fee_pct_per_side
        net_proceeds = gross_proceeds - exit_fee
        cash += net_proceeds

        gross_pnl = units * (fill_exit - entry_price)
        net_pnl = net_proceeds - entry_cost_basis - entry_fee
        denominator = entry_cost_basis + entry_fee
        return_pct = net_pnl / denominator if denominator > 0 else 0.0

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=fill_exit,
            units=units,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            reason=reason,
            bars_held=bars_held,
        )
        return cash, trade


def calculate_metrics(
    *,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    bars_in_market: int,
    total_bars: int,
    annualization_days: float,
) -> dict[str, float]:
    start_equity = float(equity_curve.iloc[0])
    end_equity = float(equity_curve.iloc[-1])
    total_return = end_equity / start_equity - 1.0

    elapsed_days = max(
        (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86_400.0,
        1.0,
    )
    years = elapsed_days / annualization_days
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if end_equity > 0 else -1.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min())

    if trades.empty:
        return {
            "ending_equity": end_equity,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "trade_count": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade": 0.0,
            "exposure": bars_in_market / total_bars,
            "score": -math.inf,
        }

    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    win_rate = float((trades["net_pnl"] > 0).mean())
    average_trade = float(trades["return_pct"].mean())

    # Deliberately penalizes drawdown and tiny samples.
    finite_pf = min(profit_factor, 5.0) if math.isfinite(profit_factor) else 5.0
    sample_penalty = min(1.0, len(trades) / 100.0)
    score = (
        0.40 * total_return
        + 0.25 * finite_pf
        + 0.20 * average_trade * 100.0
        - 0.35 * abs(max_drawdown)
    ) * sample_penalty

    return {
        "ending_equity": end_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "trade_count": float(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "exposure": bars_in_market / total_bars,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Parameter search and validation split
# ---------------------------------------------------------------------------

def generate_parameter_grid(
    specs: Sequence[ParameterSpec],
) -> Iterable[dict[str, Any]]:
    names = [spec.name for spec in specs]
    value_lists = [spec.values() for spec in specs]
    for values in itertools.product(*value_lists):
        yield dict(zip(names, values, strict=True))


def count_combinations(specs: Sequence[ParameterSpec]) -> int:
    count = 1
    for spec in specs:
        count *= len(spec.values())
    return count


def optimize(
    *,
    data: pd.DataFrame,
    strategy: StrategyFunction,
    specs: Sequence[ParameterSpec],
    engine: BacktestEngine,
    constraint: ConstraintFunction | None = None,
    minimum_trades: int = 20,
    maximum_combinations: int = 100_000,
) -> pd.DataFrame:
    combinations = count_combinations(specs)
    if combinations > maximum_combinations:
        raise ValueError(
            f"Parameter grid contains {combinations:,} combinations. "
            f"Limit is {maximum_combinations:,}. Narrow the search space."
        )

    rows: list[dict[str, Any]] = []

    for parameters in generate_parameter_grid(specs):
        if constraint is not None and not constraint(parameters):
            continue

        result = engine.run(data, strategy, parameters)
        row = {
            **parameters,
            **result.metrics,
        }
        row["eligible"] = int(result.metrics["trade_count"]) >= minimum_trades
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["eligible", "score", "profit_factor", "total_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return results



def optimize_coordinate(
    *,
    data: pd.DataFrame,
    strategy: StrategyFunction,
    specs: Sequence[ParameterSpec],
    initial_parameters: Mapping[str, Any],
    engine: BacktestEngine,
    constraint: ConstraintFunction | None = None,
    rounds: int = 3,
    minimum_trades: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Coordinate search changes one parameter at a time while holding the others
    fixed. Multiple rounds allow earlier parameters to be revisited after later
    parameters changed.

    This avoids the combinatorial explosion of a full Cartesian grid. It does
    not guarantee the global optimum, so final candidates still need untouched
    validation and preferably walk-forward testing.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    current = dict(initial_parameters)
    spec_names = {spec.name for spec in specs}
    missing = spec_names.difference(current)
    if missing:
        raise ValueError(f"Initial parameters missing: {sorted(missing)}")

    all_trials: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    def evaluate(parameters: Mapping[str, Any], round_number: int, swept_name: str) -> dict[str, Any] | None:
        if constraint is not None and not constraint(parameters):
            return None

        key = tuple(sorted(parameters.items()))
        if key in seen:
            return None
        seen.add(key)

        result = engine.run(data, strategy, parameters)
        row = {
            **parameters,
            **result.metrics,
            "eligible": int(result.metrics["trade_count"]) >= minimum_trades,
            "round": round_number,
            "swept_parameter": swept_name,
        }
        all_trials.append(row)
        return row

    evaluate(current, 0, "initial")

    for round_number in range(1, rounds + 1):
        changed = False

        for spec in specs:
            candidates: list[dict[str, Any]] = []

            for value in spec.values():
                trial_parameters = dict(current)
                trial_parameters[spec.name] = value
                row = evaluate(trial_parameters, round_number, spec.name)

                if row is None:
                    # It may already have been evaluated. Recover it from trials.
                    for previous in reversed(all_trials):
                        if all(previous.get(name) == trial_parameters[name] for name in spec_names):
                            row = previous
                            break

                if row is not None:
                    candidates.append(row)

            if not candidates:
                continue

            candidates_frame = pd.DataFrame(candidates).sort_values(
                ["eligible", "score", "profit_factor", "total_return"],
                ascending=[False, False, False, False],
            )
            best_candidate = candidates_frame.iloc[0]
            old_value = current[spec.name]
            current[spec.name] = best_candidate[spec.name]
            changed = changed or current[spec.name] != old_value

        if not changed:
            break

    trials = pd.DataFrame(all_trials)
    if not trials.empty:
        trials = trials.sort_values(
            ["eligible", "score", "profit_factor", "total_return"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return current, trials


def validate_candidates(
    *,
    validation_data: pd.DataFrame,
    strategy: StrategyFunction,
    parameter_names: Sequence[str],
    candidates: pd.DataFrame,
    engine: BacktestEngine,
    top_n: int = 20,
    minimum_validation_trades: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        return pd.DataFrame()

    for _, train_row in candidates.head(top_n).iterrows():
        parameters = {name: train_row[name] for name in parameter_names}
        result = engine.run(validation_data, strategy, parameters)
        rows.append(
            {
                **parameters,
                **{
                    f"train_{key}": train_row[key]
                    for key in train_row.index
                    if key not in parameter_names
                },
                **{
                    f"validation_{key}": value
                    for key, value in result.metrics.items()
                },
                "validation_eligible": (
                    result.metrics["trade_count"] >= minimum_validation_trades
                ),
            }
        )

    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            [
                "validation_eligible",
                "validation_score",
                "validation_profit_factor",
                "validation_total_return",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return output


def optimize_coordinate_train_validate(
    *,
    data: pd.DataFrame,
    strategy: StrategyFunction,
    specs: Sequence[ParameterSpec],
    initial_parameters: Mapping[str, Any],
    engine: BacktestEngine,
    train_fraction: float = 0.70,
    top_n: int = 20,
    rounds: int = 3,
    constraint: ConstraintFunction | None = None,
    minimum_train_trades: int = 20,
    minimum_validation_trades: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = validate_ohlcv(data)
    split_index = int(len(frame) * train_fraction)

    if split_index < 100 or len(frame) - split_index < 50:
        raise ValueError("Not enough data for a meaningful train/validation split")

    train = frame.iloc[:split_index].copy()
    validation = frame.iloc[split_index:].copy()

    best_parameters, train_trials = optimize_coordinate(
        data=train,
        strategy=strategy,
        specs=specs,
        initial_parameters=initial_parameters,
        engine=engine,
        constraint=constraint,
        rounds=rounds,
        minimum_trades=minimum_train_trades,
    )

    validation_results = validate_candidates(
        validation_data=validation,
        strategy=strategy,
        parameter_names=[spec.name for spec in specs],
        candidates=train_trials,
        engine=engine,
        top_n=top_n,
        minimum_validation_trades=minimum_validation_trades,
    )
    return best_parameters, train_trials, validation_results

def optimize_train_validate(
    *,
    data: pd.DataFrame,
    strategy: StrategyFunction,
    specs: Sequence[ParameterSpec],
    engine: BacktestEngine,
    train_fraction: float = 0.70,
    top_n: int = 20,
    constraint: ConstraintFunction | None = None,
    minimum_train_trades: int = 20,
    minimum_validation_trades: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimize only on the train segment, then evaluate the top train candidates
    on untouched validation data.
    """
    frame = validate_ohlcv(data)
    split_index = int(len(frame) * train_fraction)

    if split_index < 100 or len(frame) - split_index < 50:
        raise ValueError("Not enough data for a meaningful train/validation split")

    train = frame.iloc[:split_index].copy()
    validation = frame.iloc[split_index:].copy()

    train_results = optimize(
        data=train,
        strategy=strategy,
        specs=specs,
        engine=engine,
        constraint=constraint,
        minimum_trades=minimum_train_trades,
    )

    validation_rows: list[dict[str, Any]] = []
    parameter_names = [spec.name for spec in specs]

    for _, train_row in train_results.head(top_n).iterrows():
        parameters = {name: train_row[name] for name in parameter_names}
        validation_result = engine.run(validation, strategy, parameters)
        validation_rows.append(
            {
                **parameters,
                **{f"train_{key}": train_row[key] for key in train_row.index if key not in parameter_names},
                **{f"validation_{key}": value for key, value in validation_result.metrics.items()},
                "validation_eligible": validation_result.metrics["trade_count"]
                >= minimum_validation_trades,
            }
        )

    validation_results = pd.DataFrame(validation_rows)
    if not validation_results.empty:
        validation_results = validation_results.sort_values(
            [
                "validation_eligible",
                "validation_score",
                "validation_profit_factor",
                "validation_total_return",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return train_results, validation_results


def default_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec("fast_ema", "half", 8.0, 20.0),
        ParameterSpec("slow_ema", "half", 30.0, 60.0),
        ParameterSpec("rsi_period", "half", 10.0, 20.0),
        ParameterSpec("rsi_entry", "half", 35.0, 50.0),
        ParameterSpec("rsi_exit", "half", 60.0, 80.0),
        ParameterSpec("stop_pct", "half", 1.0, 5.0),
        ParameterSpec("target_pct", "half", 2.0, 10.0),
    ]


def default_initial_parameters() -> dict[str, float]:
    return {
        "fast_ema": 12.0,
        "slow_ema": 40.0,
        "rsi_period": 14.0,
        "rsi_entry": 42.0,
        "rsi_exit": 68.0,
        "stop_pct": 2.5,
        "target_pct": 5.0,
    }


def default_constraint(parameters: Mapping[str, Any]) -> bool:
    return (
        float(parameters["fast_ema"]) < float(parameters["slow_ema"])
        and float(parameters["rsi_entry"]) < float(parameters["rsi_exit"])
        and float(parameters["target_pct"]) > float(parameters["stop_pct"])
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    timestamp_candidates = [
        name for name in frame.columns
        if str(name).lower() in {"timestamp", "datetime", "date", "time"}
    ]
    if not timestamp_candidates:
        raise ValueError(
            "CSV needs a timestamp column named timestamp, datetime, date, or time"
        )

    timestamp_column = timestamp_candidates[0]
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    frame = frame.set_index(timestamp_column)
    return validate_ohlcv(frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple long-only backtest engine with exact half-step parameter search."
    )
    parser.add_argument("csv", help="OHLCV CSV file")
    parser.add_argument("--output-dir", default="backtest_output")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--fee-pct", type=float, default=0.10, help="Fee per side in percent")
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.05,
        help="Slippage per side in percent",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--search-mode",
        choices=["coordinate", "grid"],
        default="coordinate",
        help="Coordinate search is the safe default; full grid is only for small spaces.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=1.0,
        help="Equity risk budget per trade in percent. Set below 0 to disable.",
    )
    args = parser.parse_args()

    data = load_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        risk_fraction_per_trade=(
            None if args.risk_per_trade_pct < 0
            else args.risk_per_trade_pct / 100.0
        ),
        costs=CostModel(
            fee_pct_per_side=args.fee_pct / 100.0,
            slippage_pct_per_side=args.slippage_pct / 100.0,
        ),
    )
    engine = BacktestEngine(config)

    specs = default_parameter_specs()
    raw_combinations = count_combinations(specs)
    print(f"Raw Cartesian combinations: {raw_combinations:,}")
    print(f"Search mode: {args.search_mode}")

    if args.search_mode == "coordinate":
        best_parameters, train_results, validation_results = (
            optimize_coordinate_train_validate(
                data=data,
                strategy=ema_rsi_pullback_strategy,
                specs=specs,
                initial_parameters=default_initial_parameters(),
                engine=engine,
                train_fraction=args.train_fraction,
                top_n=args.top_n,
                rounds=args.rounds,
                constraint=default_constraint,
            )
        )
        print(f"Best train parameters after coordinate search: {best_parameters}")
    else:
        train_results, validation_results = optimize_train_validate(
            data=data,
            strategy=ema_rsi_pullback_strategy,
            specs=specs,
            engine=engine,
            train_fraction=args.train_fraction,
            top_n=args.top_n,
            constraint=default_constraint,
        )

    train_path = output_dir / "train_results.csv"
    validation_path = output_dir / "validation_results.csv"
    train_results.to_csv(train_path, index=False)
    validation_results.to_csv(validation_path, index=False)

    print("\nBest validation candidates:")
    columns = [
        "fast_ema",
        "slow_ema",
        "rsi_period",
        "rsi_entry",
        "rsi_exit",
        "stop_pct",
        "target_pct",
        "validation_trade_count",
        "validation_profit_factor",
        "validation_total_return",
        "validation_max_drawdown",
        "validation_score",
    ]
    available = [column for column in columns if column in validation_results.columns]
    print(validation_results[available].head(10).to_string(index=False))
    print(f"\nSaved: {train_path}")
    print(f"Saved: {validation_path}")


if __name__ == "__main__":
    main()
