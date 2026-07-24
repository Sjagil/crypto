
from __future__ import annotations

"""
trading_math_engine.py

Standalone mathematics toolkit for trading systems.

Main subjects
-------------
- expectancy and break-even mathematics
- reward/risk and profit factor
- costs in R or cash
- position sizing from stop distance
- drawdown recovery and losing streaks
- confidence intervals and bootstrap uncertainty
- fixed-fraction equity compounding
- Monte Carlo risk of ruin
- Kelly and fractional Kelly

Important corrections
---------------------
A trade is not automatically a 50/50 event, and trades are not automatically
independent. Use your measured net trade results. When trades cluster by regime,
asset, or volatility, use block_size > 1 in bootstrap and empirical Monte Carlo.

Conventions
-----------
Percentages are decimal fractions:
    0.55 = 55%
    0.005 = 0.5%

R multiples:
    +2.0 = profit of two initial-risk units
    -1.0 = loss of one initial-risk unit

Costs in `cost_r` are total round-trip costs per trade, expressed in R.
"""

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite, log, sqrt
from statistics import NormalDist
from typing import Any, Iterable, Literal, Sequence

import numpy as np

Number = float | int
Side = Literal["long", "short"]


@dataclass(frozen=True)
class ExpectancyResult:
    win_rate: float
    loss_rate: float
    average_win_r: float
    average_loss_r: float
    cost_r: float
    gross_expectancy_r: float
    net_expectancy_r: float
    breakeven_win_rate: float
    payoff_ratio: float
    profit_factor: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PositionSizeResult:
    account_equity: float
    risk_fraction: float
    risk_budget: float
    entry_price: float
    stop_price: float
    stop_distance: float
    stop_distance_fraction: float
    estimated_cost_per_unit: float
    risk_per_unit: float
    units: float
    position_notional: float
    actual_risk: float
    actual_risk_fraction: float
    capped_by_max_position: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TradeStatistics:
    trade_count: int
    win_count: int
    loss_count: int
    flat_count: int
    win_rate: float
    average_win_r: float
    average_loss_r: float
    expectancy_r: float
    median_r: float
    standard_deviation_r: float
    standard_error_r: float
    profit_factor: float
    payoff_ratio: float
    total_r: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapResult:
    observed_expectancy_r: float
    confidence_level: float
    lower_expectancy_r: float
    upper_expectancy_r: float
    probability_expectancy_positive: float
    bootstrap_samples: int
    block_size: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    trades_per_simulation: int
    initial_equity: float
    risk_fraction: float
    ruin_drawdown: float
    risk_of_ruin: float
    probability_of_loss: float
    median_ending_equity: float
    mean_ending_equity: float
    p05_ending_equity: float
    p95_ending_equity: float
    median_max_drawdown: float
    p95_max_drawdown: float
    median_max_losing_streak: float
    p95_max_losing_streak: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _probability(value: Number, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _positive(value: Number, name: str, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not isfinite(value) or not valid:
        word = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {word}")
    return value


def _array(values: Iterable[Number], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


# ---------------------------------------------------------------------------
# Expectancy
# ---------------------------------------------------------------------------

def expectancy(
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> ExpectancyResult:
    """
    Net expectancy:
        p * average_win_r - (1-p) * average_loss_r - cost_r

    `average_loss_r` is passed as a positive magnitude.
    """
    p = _probability(win_rate, "win_rate")
    win = _positive(average_win_r, "average_win_r", allow_zero=True)
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", allow_zero=True)

    gross = p * win - (1.0 - p) * loss
    net = gross - cost
    return ExpectancyResult(
        win_rate=p,
        loss_rate=1.0 - p,
        average_win_r=win,
        average_loss_r=loss,
        cost_r=cost,
        gross_expectancy_r=gross,
        net_expectancy_r=net,
        breakeven_win_rate=breakeven_win_rate(win, loss, cost),
        payoff_ratio=win / loss,
        profit_factor=profit_factor(p, win, loss),
    )


def expectancy_cash(
    win_rate: Number,
    average_win_cash: Number,
    average_loss_cash: Number,
    average_cost_cash: Number = 0.0,
) -> float:
    p = _probability(win_rate, "win_rate")
    win = _positive(average_win_cash, "average_win_cash", allow_zero=True)
    loss = _positive(average_loss_cash, "average_loss_cash")
    cost = _positive(average_cost_cash, "average_cost_cash", allow_zero=True)
    return p * win - (1.0 - p) * loss - cost


def breakeven_win_rate(
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    win = _positive(average_win_r, "average_win_r", allow_zero=True)
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", allow_zero=True)
    return float(np.clip((loss + cost) / (win + loss), 0.0, 1.0))


def required_average_win_r(
    win_rate: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    p = _probability(win_rate, "win_rate")
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", allow_zero=True)
    return float("inf") if p == 0.0 else ((1.0 - p) * loss + cost) / p


def required_reward_to_risk(win_rate: Number, cost_r: Number = 0.0) -> float:
    return required_average_win_r(win_rate, 1.0, cost_r)


def profit_factor(
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
) -> float:
    p = _probability(win_rate, "win_rate")
    win = _positive(average_win_r, "average_win_r", allow_zero=True)
    loss = _positive(average_loss_r, "average_loss_r")
    denominator = (1.0 - p) * loss
    return float("inf") if denominator == 0.0 else p * win / denominator


def expected_total_r(trade_count: int, expectancy_r: Number) -> float:
    if not isinstance(trade_count, int) or trade_count < 0:
        raise ValueError("trade_count must be a non-negative integer")
    value = float(expectancy_r)
    if not isfinite(value):
        raise ValueError("expectancy_r must be finite")
    return trade_count * value


def binary_outcome_variance_r(
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    p = _probability(win_rate, "win_rate")
    win = _positive(average_win_r, "average_win_r", allow_zero=True)
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", allow_zero=True)
    outcomes = np.array([win - cost, -loss - cost])
    probabilities = np.array([p, 1.0 - p])
    mean = float(np.sum(probabilities * outcomes))
    return float(np.sum(probabilities * (outcomes - mean) ** 2))


def binary_expectancy_standard_error(
    trade_count: int,
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    if not isinstance(trade_count, int) or trade_count <= 0:
        raise ValueError("trade_count must be a positive integer")
    variance = binary_outcome_variance_r(
        win_rate, average_win_r, average_loss_r, cost_r
    )
    return sqrt(variance / trade_count)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    account_equity: Number,
    risk_fraction: Number,
    entry_price: Number,
    stop_price: Number,
    *,
    side: Side = "long",
    fee_fraction_per_side: Number = 0.0,
    slippage_fraction_per_side: Number = 0.0,
    contract_multiplier: Number = 1.0,
    max_position_fraction: Number = 1.0,
    allow_fractional_units: bool = True,
) -> PositionSizeResult:
    """
    Risk-based sizing including estimated entry and stop-exit friction.

    The result can be capped by `max_position_fraction`, preventing a very
    narrow stop from creating an absurdly large notional position.
    """
    equity = _positive(account_equity, "account_equity")
    risk_fraction = _probability(risk_fraction, "risk_fraction")
    entry = _positive(entry_price, "entry_price")
    stop = _positive(stop_price, "stop_price")
    fee = _positive(fee_fraction_per_side, "fee_fraction_per_side", True)
    slippage = _positive(slippage_fraction_per_side, "slippage_fraction_per_side", True)
    multiplier = _positive(contract_multiplier, "contract_multiplier")
    max_fraction = _probability(max_position_fraction, "max_position_fraction")

    if side == "long":
        if stop >= entry:
            raise ValueError("long stop_price must be below entry_price")
        distance = entry - stop
    elif side == "short":
        if stop <= entry:
            raise ValueError("short stop_price must be above entry_price")
        distance = stop - entry
    else:
        raise ValueError("side must be 'long' or 'short'")

    risk_budget = equity * risk_fraction
    friction = fee + slippage
    estimated_cost_per_unit = (entry + stop) * friction * multiplier
    risk_per_unit = distance * multiplier + estimated_cost_per_unit
    unconstrained_units = risk_budget / risk_per_unit

    max_notional = equity * max_fraction
    max_units = max_notional / (entry * multiplier)
    capped = unconstrained_units > max_units
    units = min(unconstrained_units, max_units)
    if not allow_fractional_units:
        units = float(floor(units))

    notional = units * entry * multiplier
    actual_risk = units * risk_per_unit
    return PositionSizeResult(
        account_equity=equity,
        risk_fraction=risk_fraction,
        risk_budget=risk_budget,
        entry_price=entry,
        stop_price=stop,
        stop_distance=distance,
        stop_distance_fraction=distance / entry,
        estimated_cost_per_unit=estimated_cost_per_unit,
        risk_per_unit=risk_per_unit,
        units=units,
        position_notional=notional,
        actual_risk=actual_risk,
        actual_risk_fraction=actual_risk / equity,
        capped_by_max_position=capped,
    )


def calculate_position_size_from_stop_fraction(
    account_equity: Number,
    risk_fraction: Number,
    entry_price: Number,
    stop_distance_fraction: Number,
    **kwargs: Any,
) -> PositionSizeResult:
    entry = _positive(entry_price, "entry_price")
    distance = _probability(stop_distance_fraction, "stop_distance_fraction")
    side = kwargs.get("side", "long")
    stop = entry * (1.0 - distance) if side == "long" else entry * (1.0 + distance)
    return calculate_position_size(
        account_equity, risk_fraction, entry, stop, **kwargs
    )


# ---------------------------------------------------------------------------
# Drawdown, streaks, and survival
# ---------------------------------------------------------------------------

def drawdown_recovery(drawdown_fraction: Number) -> float:
    drawdown = _probability(drawdown_fraction, "drawdown_fraction")
    return float("inf") if drawdown == 1.0 else drawdown / (1.0 - drawdown)


def equity_after_loss_streak(
    initial_equity: Number,
    risk_fraction: Number,
    consecutive_losses: int,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    equity = _positive(initial_equity, "initial_equity")
    risk = _probability(risk_fraction, "risk_fraction")
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", True)
    if not isinstance(consecutive_losses, int) or consecutive_losses < 0:
        raise ValueError("consecutive_losses must be a non-negative integer")
    fractional_loss = risk * (loss + cost)
    if fractional_loss >= 1.0:
        return 0.0
    return equity * (1.0 - fractional_loss) ** consecutive_losses


def drawdown_after_loss_streak(
    risk_fraction: Number,
    consecutive_losses: int,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    return 1.0 - equity_after_loss_streak(
        1.0, risk_fraction, consecutive_losses, average_loss_r, cost_r
    )


def losses_to_reach_drawdown(
    risk_fraction: Number,
    target_drawdown: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> int:
    risk = _probability(risk_fraction, "risk_fraction")
    target = _probability(target_drawdown, "target_drawdown")
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", True)
    if target == 0.0:
        return 0
    fractional_loss = risk * (loss + cost)
    if fractional_loss <= 0.0:
        return 10**18
    if fractional_loss >= 1.0:
        return 1
    return ceil(log(1.0 - target) / log(1.0 - fractional_loss))


def probability_at_least_one_losing_streak(
    loss_rate: Number,
    trade_count: int,
    streak_length: int,
) -> float:
    """
    Exact under independent Bernoulli trades. For regime-dependent trade logs,
    use empirical Monte Carlo with block_size > 1.
    """
    q = _probability(loss_rate, "loss_rate")
    if not isinstance(trade_count, int) or trade_count < 0:
        raise ValueError("trade_count must be a non-negative integer")
    if not isinstance(streak_length, int) or streak_length <= 0:
        raise ValueError("streak_length must be a positive integer")
    if streak_length > trade_count or q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0

    states = np.zeros(streak_length)
    states[0] = 1.0
    for _ in range(trade_count):
        nxt = np.zeros_like(states)
        nxt[0] = np.sum(states) * (1.0 - q)
        nxt[1:] = states[:-1] * q
        states = nxt
    return float(np.clip(1.0 - np.sum(states), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    *,
    cap_at_one: bool = True,
) -> float:
    p = _probability(win_rate, "win_rate")
    b = _positive(average_win_r, "average_win_r")
    a = _positive(average_loss_r, "average_loss_r")
    value = (p * b - (1.0 - p) * a) / (a * b)
    return min(value, 1.0) if cap_at_one else value


def fractional_kelly(
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    fraction: Number = 0.5,
) -> float:
    scalar = _probability(fraction, "fraction")
    return max(
        0.0,
        scalar * kelly_fraction(win_rate, average_win_r, average_loss_r),
    )


# ---------------------------------------------------------------------------
# Trade log and confidence
# ---------------------------------------------------------------------------

def _max_streak(mask: np.ndarray) -> int:
    current = maximum = 0
    for value in mask:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def trade_statistics(r_multiples: Iterable[Number]) -> TradeStatistics:
    values = _array(r_multiples, "r_multiples")
    wins, losses = values[values > 0], values[values < 0]
    count = len(values)
    average_win = float(np.mean(wins)) if len(wins) else 0.0
    average_loss = abs(float(np.mean(losses))) if len(losses) else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    std = float(np.std(values, ddof=1)) if count > 1 else 0.0

    return TradeStatistics(
        trade_count=count,
        win_count=len(wins),
        loss_count=len(losses),
        flat_count=int(np.sum(values == 0.0)),
        win_rate=len(wins) / count,
        average_win_r=average_win,
        average_loss_r=average_loss,
        expectancy_r=float(np.mean(values)),
        median_r=float(np.median(values)),
        standard_deviation_r=std,
        standard_error_r=std / sqrt(count),
        profit_factor=float("inf") if gross_loss == 0 else gross_profit / gross_loss,
        payoff_ratio=float("inf") if average_loss == 0 else average_win / average_loss,
        total_r=float(np.sum(values)),
        max_consecutive_wins=_max_streak(values > 0),
        max_consecutive_losses=_max_streak(values < 0),
    )


def wilson_win_rate_interval(
    wins: int,
    trade_count: int,
    confidence_level: Number = 0.95,
) -> tuple[float, float]:
    if not isinstance(wins, int) or wins < 0:
        raise ValueError("wins must be a non-negative integer")
    if not isinstance(trade_count, int) or trade_count <= 0 or wins > trade_count:
        raise ValueError("invalid trade_count or wins")
    confidence = _probability(confidence_level, "confidence_level")
    if confidence in {0.0, 1.0}:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p, n = wins / trade_count, float(trade_count)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    margin = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def minimum_trades_for_win_rate_margin(
    estimated_win_rate: Number,
    margin_of_error: Number,
    confidence_level: Number = 0.95,
) -> int:
    p = _probability(estimated_win_rate, "estimated_win_rate")
    margin = _positive(margin_of_error, "margin_of_error")
    confidence = _probability(confidence_level, "confidence_level")
    if confidence in {0.0, 1.0}:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return ceil(z * z * p * (1.0 - p) / (margin * margin))


def _bootstrap_sample(
    values: np.ndarray,
    size: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size == 1:
        return rng.choice(values, size=size, replace=True)
    starts = np.arange(0, len(values) - block_size + 1)
    blocks = ceil(size / block_size)
    chosen = rng.choice(starts, size=blocks, replace=True)
    return np.concatenate(
        [values[start:start + block_size] for start in chosen]
    )[:size]


def bootstrap_expectancy(
    r_multiples: Iterable[Number],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: Number = 0.95,
    block_size: int = 1,
    seed: int | None = None,
) -> BootstrapResult:
    values = _array(r_multiples, "r_multiples")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not isinstance(block_size, int) or not 1 <= block_size <= len(values):
        raise ValueError("block_size must be between 1 and sample length")
    confidence = _probability(confidence_level, "confidence_level")
    if confidence in {0.0, 1.0}:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        means[i] = np.mean(_bootstrap_sample(values, len(values), block_size, rng))

    alpha = 1.0 - confidence
    return BootstrapResult(
        observed_expectancy_r=float(np.mean(values)),
        confidence_level=confidence,
        lower_expectancy_r=float(np.quantile(means, alpha / 2.0)),
        upper_expectancy_r=float(np.quantile(means, 1.0 - alpha / 2.0)),
        probability_expectancy_positive=float(np.mean(means > 0.0)),
        bootstrap_samples=bootstrap_samples,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Equity and Monte Carlo
# ---------------------------------------------------------------------------

def compound_equity_from_r(
    r_multiples: Iterable[Number],
    initial_equity: Number,
    risk_fraction: Number,
) -> np.ndarray:
    values = _array(r_multiples, "r_multiples")
    equity = _positive(initial_equity, "initial_equity")
    risk = _probability(risk_fraction, "risk_fraction")
    curve = np.empty(len(values) + 1)
    curve[0] = equity
    for i, outcome in enumerate(values, start=1):
        curve[i] = max(0.0, curve[i - 1] * (1.0 + risk * outcome))
    return curve


def maximum_drawdown(equity_curve: Sequence[Number]) -> float:
    values = _array(equity_curve, "equity_curve")
    if np.any(values < 0):
        raise ValueError("equity_curve may not contain negative values")
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(
        peaks - values, peaks, out=np.zeros_like(values), where=peaks > 0
    )
    return float(np.max(drawdowns))


def expected_equity_binary(
    initial_equity: Number,
    trade_count: int,
    risk_fraction: Number,
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
) -> float:
    equity = _positive(initial_equity, "initial_equity")
    if not isinstance(trade_count, int) or trade_count < 0:
        raise ValueError("trade_count must be non-negative")
    risk = _probability(risk_fraction, "risk_fraction")
    edge = expectancy(win_rate, average_win_r, average_loss_r, cost_r).net_expectancy_r
    return max(0.0, equity * (1.0 + risk * edge) ** trade_count)


def _mc_result(
    ending: np.ndarray,
    max_dd: np.ndarray,
    max_streak: np.ndarray,
    ruined: np.ndarray,
    *,
    simulations: int,
    trades: int,
    initial_equity: float,
    risk_fraction: float,
    ruin_drawdown: float,
) -> MonteCarloResult:
    return MonteCarloResult(
        simulations=simulations,
        trades_per_simulation=trades,
        initial_equity=initial_equity,
        risk_fraction=risk_fraction,
        ruin_drawdown=ruin_drawdown,
        risk_of_ruin=float(np.mean(ruined)),
        probability_of_loss=float(np.mean(ending < initial_equity)),
        median_ending_equity=float(np.median(ending)),
        mean_ending_equity=float(np.mean(ending)),
        p05_ending_equity=float(np.quantile(ending, 0.05)),
        p95_ending_equity=float(np.quantile(ending, 0.95)),
        median_max_drawdown=float(np.median(max_dd)),
        p95_max_drawdown=float(np.quantile(max_dd, 0.95)),
        median_max_losing_streak=float(np.median(max_streak)),
        p95_max_losing_streak=float(np.quantile(max_streak, 0.95)),
    )


def monte_carlo_binary_system(
    *,
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
    risk_fraction: Number = 0.01,
    initial_equity: Number = 10_000.0,
    trades_per_simulation: int = 500,
    simulations: int = 10_000,
    ruin_drawdown: Number = 0.50,
    win_r_std: Number = 0.0,
    loss_r_std: Number = 0.0,
    seed: int | None = None,
) -> MonteCarloResult:
    p = _probability(win_rate, "win_rate")
    win = _positive(average_win_r, "average_win_r", True)
    loss = _positive(average_loss_r, "average_loss_r")
    cost = _positive(cost_r, "cost_r", True)
    risk = _probability(risk_fraction, "risk_fraction")
    equity0 = _positive(initial_equity, "initial_equity")
    ruin = _probability(ruin_drawdown, "ruin_drawdown")
    win_std = _positive(win_r_std, "win_r_std", True)
    loss_std = _positive(loss_r_std, "loss_r_std", True)
    if not isinstance(trades_per_simulation, int) or trades_per_simulation <= 0:
        raise ValueError("trades_per_simulation must be positive")
    if not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    ending, max_dd = np.empty(simulations), np.empty(simulations)
    max_streak = np.empty(simulations)
    ruined = np.zeros(simulations, dtype=bool)
    ruin_level = equity0 * (1.0 - ruin)

    for i in range(simulations):
        wins = rng.random(trades_per_simulation) < p
        win_values = (
            np.maximum(0.0, rng.normal(win, win_std, trades_per_simulation))
            if win_std else np.full(trades_per_simulation, win)
        )
        loss_values = (
            np.maximum(0.0, rng.normal(loss, loss_std, trades_per_simulation))
            if loss_std else np.full(trades_per_simulation, loss)
        )
        outcomes = np.where(wins, win_values - cost, -loss_values - cost)
        curve = compound_equity_from_r(outcomes, equity0, risk)
        ending[i] = curve[-1]
        max_dd[i] = maximum_drawdown(curve)
        max_streak[i] = _max_streak(outcomes < 0)
        ruined[i] = np.any(curve <= ruin_level)

    return _mc_result(
        ending, max_dd, max_streak, ruined,
        simulations=simulations,
        trades=trades_per_simulation,
        initial_equity=equity0,
        risk_fraction=risk,
        ruin_drawdown=ruin,
    )


def empirical_risk_of_ruin(
    r_multiples: Iterable[Number],
    *,
    risk_fraction: Number = 0.01,
    initial_equity: Number = 10_000.0,
    trades_per_simulation: int | None = None,
    simulations: int = 10_000,
    ruin_drawdown: Number = 0.50,
    block_size: int = 1,
    seed: int | None = None,
) -> MonteCarloResult:
    values = _array(r_multiples, "r_multiples")
    risk = _probability(risk_fraction, "risk_fraction")
    equity0 = _positive(initial_equity, "initial_equity")
    ruin = _probability(ruin_drawdown, "ruin_drawdown")
    trades = len(values) if trades_per_simulation is None else trades_per_simulation
    if not isinstance(trades, int) or trades <= 0:
        raise ValueError("trades_per_simulation must be positive")
    if not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("simulations must be positive")
    if not isinstance(block_size, int) or not 1 <= block_size <= len(values):
        raise ValueError("block_size must be between 1 and sample length")

    rng = np.random.default_rng(seed)
    ending, max_dd = np.empty(simulations), np.empty(simulations)
    max_streak = np.empty(simulations)
    ruined = np.zeros(simulations, dtype=bool)
    ruin_level = equity0 * (1.0 - ruin)

    for i in range(simulations):
        outcomes = _bootstrap_sample(values, trades, block_size, rng)
        curve = compound_equity_from_r(outcomes, equity0, risk)
        ending[i] = curve[-1]
        max_dd[i] = maximum_drawdown(curve)
        max_streak[i] = _max_streak(outcomes < 0)
        ruined[i] = np.any(curve <= ruin_level)

    return _mc_result(
        ending, max_dd, max_streak, ruined,
        simulations=simulations,
        trades=trades,
        initial_equity=equity0,
        risk_fraction=risk,
        ruin_drawdown=ruin,
    )


def system_math_summary(
    *,
    win_rate: Number,
    average_win_r: Number,
    average_loss_r: Number = 1.0,
    cost_r: Number = 0.0,
    risk_fraction: Number = 0.01,
    initial_equity: Number = 10_000.0,
    trade_count: int = 100,
    losing_streak_length: int = 7,
) -> dict[str, Any]:
    """One-call summary for strategy evaluation."""
    exp = expectancy(win_rate, average_win_r, average_loss_r, cost_r)
    ending_after_streak = equity_after_loss_streak(
        initial_equity, risk_fraction, losing_streak_length,
        average_loss_r, cost_r
    )
    return {
        "expectancy": exp.to_dict(),
        "kelly_fraction": kelly_fraction(
            win_rate, average_win_r, average_loss_r
        ),
        "half_kelly_fraction": fractional_kelly(
            win_rate, average_win_r, average_loss_r, 0.5
        ),
        "expected_equity_after_trades": expected_equity_binary(
            initial_equity, trade_count, risk_fraction, win_rate,
            average_win_r, average_loss_r, cost_r
        ),
        "losing_streak_length": losing_streak_length,
        "probability_of_losing_streak": probability_at_least_one_losing_streak(
            1.0 - float(win_rate), trade_count, losing_streak_length
        ),
        "equity_after_losing_streak": ending_after_streak,
        "drawdown_after_losing_streak": 1.0 - ending_after_streak / float(initial_equity),
    }


def main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Trading mathematics toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("expectancy")
    e.add_argument("--win-rate", type=float, required=True)
    e.add_argument("--avg-win-r", type=float, required=True)
    e.add_argument("--avg-loss-r", type=float, default=1.0)
    e.add_argument("--cost-r", type=float, default=0.0)

    p = sub.add_parser("position-size")
    p.add_argument("--equity", type=float, required=True)
    p.add_argument("--risk", type=float, required=True)
    p.add_argument("--entry", type=float, required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--fee", type=float, default=0.0)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--max-position", type=float, default=1.0)

    m = sub.add_parser("simulate")
    m.add_argument("--win-rate", type=float, required=True)
    m.add_argument("--avg-win-r", type=float, required=True)
    m.add_argument("--avg-loss-r", type=float, default=1.0)
    m.add_argument("--cost-r", type=float, default=0.0)
    m.add_argument("--risk", type=float, default=0.01)
    m.add_argument("--equity", type=float, default=10_000.0)
    m.add_argument("--trades", type=int, default=500)
    m.add_argument("--simulations", type=int, default=10_000)
    m.add_argument("--seed", type=int)

    args = parser.parse_args()
    if args.command == "expectancy":
        result = expectancy(
            args.win_rate, args.avg_win_r, args.avg_loss_r, args.cost_r
        )
    elif args.command == "position-size":
        result = calculate_position_size(
            args.equity, args.risk, args.entry, args.stop,
            fee_fraction_per_side=args.fee,
            slippage_fraction_per_side=args.slippage,
            max_position_fraction=args.max_position,
        )
    else:
        result = monte_carlo_binary_system(
            win_rate=args.win_rate,
            average_win_r=args.avg_win_r,
            average_loss_r=args.avg_loss_r,
            cost_r=args.cost_r,
            risk_fraction=args.risk,
            initial_equity=args.equity,
            trades_per_simulation=args.trades,
            simulations=args.simulations,
            seed=args.seed,
        )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "BootstrapResult",
    "ExpectancyResult",
    "MonteCarloResult",
    "PositionSizeResult",
    "TradeStatistics",
    "binary_expectancy_standard_error",
    "binary_outcome_variance_r",
    "bootstrap_expectancy",
    "breakeven_win_rate",
    "calculate_position_size",
    "calculate_position_size_from_stop_fraction",
    "compound_equity_from_r",
    "drawdown_after_loss_streak",
    "drawdown_recovery",
    "empirical_risk_of_ruin",
    "equity_after_loss_streak",
    "expectancy",
    "expectancy_cash",
    "expected_equity_binary",
    "expected_total_r",
    "fractional_kelly",
    "kelly_fraction",
    "losses_to_reach_drawdown",
    "maximum_drawdown",
    "minimum_trades_for_win_rate_margin",
    "monte_carlo_binary_system",
    "probability_at_least_one_losing_streak",
    "profit_factor",
    "required_average_win_r",
    "required_reward_to_risk",
    "system_math_summary",
    "trade_statistics",
    "wilson_win_rate_interval",
]
