from __future__ import annotations

"""Candle, fractal, liquidity and deterministic market-structure features.

All events are aligned to the candle on which they became knowable. A fractal
with two right-hand candles is therefore confirmed two candles after its pivot.
Use ``build_market_structure_features`` as the main entry point.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TiePolicy = Literal["strict", "first", "last"]
BreakBasis = Literal["close", "wick"]
RangeMode = Literal["full", "body"]


@dataclass(frozen=True)
class MarketStructureConfig:
    fractal_left: int = 2
    fractal_right: int = 2
    fractal_tie_policy: TiePolicy = "strict"
    atr_period: int = 14
    volume_z_window: int = 50

    displacement_atr_multiple: float = 1.25
    displacement_body_fraction: float = 0.60
    displacement_volume_z_min: float = 0.50

    break_basis: BreakBasis = "close"
    break_buffer_atr: float = 0.05
    structure_memory_bars: int = 250

    sweep_buffer_atr: float = 0.05
    sweep_reclaim_buffer_atr: float = 0.00
    sweep_max_age_bars: int = 100

    equal_level_tolerance_atr: float = 0.10
    equal_level_min_separation: int = 3
    equal_level_max_separation: int = 100

    fvg_min_atr: float = 0.10
    fvg_require_displacement: bool = True
    fvg_max_age_bars: int = 250

    order_block_lookback: int = 12
    order_block_range: RangeMode = "full"
    order_block_require_displacement: bool = True
    order_block_max_age_bars: int = 250

    doji_body_fraction: float = 0.10
    pin_wick_to_body: float = 2.0
    pin_opposite_wick_max_body: float = 0.75
    marubozu_body_fraction: float = 0.85

    def __post_init__(self) -> None:
        positive_ints = (
            "fractal_left", "fractal_right", "atr_period", "volume_z_window",
            "structure_memory_bars", "sweep_max_age_bars",
            "equal_level_min_separation", "equal_level_max_separation",
            "fvg_max_age_bars", "order_block_lookback",
            "order_block_max_age_bars",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.equal_level_min_separation > self.equal_level_max_separation:
            raise ValueError("minimum equal-level separation exceeds maximum")


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    mapping = {str(c).lower(): c for c in data.columns}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = data.rename(columns={v: k for k, v in mapping.items()})
    frame = frame[["open", "high", "low", "close", "volume"]].copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a pandas DatetimeIndex")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    invalid = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame["volume"] < 0)
    )
    if invalid.any():
        raise ValueError(f"Invalid OHLCV rows: {int(invalid.sum())}")
    return frame


def true_range(data: pd.DataFrame) -> pd.Series:
    previous_close = data["close"].shift(1)
    return pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    if not isinstance(period, int) or period <= 0:
        raise ValueError("ATR period must be a positive integer")
    return true_range(data).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def candle_geometry(data: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    out["candle_range"] = data["high"] - data["low"]
    out["real_body"] = (data["close"] - data["open"]).abs()
    out["body_fraction"] = out["real_body"] / out["candle_range"].replace(0, np.nan)
    out["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    out["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
    out["close_location"] = (
        (data["close"] - data["low"]) / out["candle_range"].replace(0, np.nan)
    )
    out["bullish_candle"] = data["close"] > data["open"]
    out["bearish_candle"] = data["close"] < data["open"]
    return out


def candlestick_patterns(data: pd.DataFrame, config: MarketStructureConfig) -> pd.DataFrame:
    g = candle_geometry(data)
    out = g.copy()
    safe_body = g["real_body"].replace(0, np.nan)
    prev_open, prev_close = data["open"].shift(1), data["close"].shift(1)
    prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
    body_high = data[["open", "close"]].max(axis=1)
    body_low = data[["open", "close"]].min(axis=1)

    out["doji"] = (g["body_fraction"] <= config.doji_body_fraction).fillna(False)
    out["marubozu"] = (g["body_fraction"] >= config.marubozu_body_fraction).fillna(False)
    out["bullish_pin_bar"] = (
        (g["lower_wick"] >= config.pin_wick_to_body * safe_body)
        & (g["upper_wick"] <= config.pin_opposite_wick_max_body * safe_body)
        & (g["close_location"] >= 0.60)
    ).fillna(False)
    out["bearish_pin_bar"] = (
        (g["upper_wick"] >= config.pin_wick_to_body * safe_body)
        & (g["lower_wick"] <= config.pin_opposite_wick_max_body * safe_body)
        & (g["close_location"] <= 0.40)
    ).fillna(False)
    out["bullish_engulfing"] = (
        (data["close"] > data["open"]) & (prev_close < prev_open)
        & (body_low <= prev_body_low) & (body_high >= prev_body_high)
    ).fillna(False)
    out["bearish_engulfing"] = (
        (data["close"] < data["open"]) & (prev_close > prev_open)
        & (body_low <= prev_body_low) & (body_high >= prev_body_high)
    ).fillna(False)
    out["inside_bar"] = (
        (data["high"] < data["high"].shift(1))
        & (data["low"] > data["low"].shift(1))
    ).fillna(False)
    out["outside_bar"] = (
        (data["high"] > data["high"].shift(1))
        & (data["low"] < data["low"].shift(1))
    ).fillna(False)
    out["bullish_harami"] = (
        (prev_close < prev_open) & (data["close"] > data["open"])
        & (body_high <= prev_body_high) & (body_low >= prev_body_low)
    ).fillna(False)
    out["bearish_harami"] = (
        (prev_close > prev_open) & (data["close"] < data["open"])
        & (body_high <= prev_body_high) & (body_low >= prev_body_low)
    ).fillna(False)

    first_bear = data["close"].shift(2) < data["open"].shift(2)
    first_bull = data["close"].shift(2) > data["open"].shift(2)
    first_body = (data["close"].shift(2) - data["open"].shift(2)).abs()
    middle_body = (data["close"].shift(1) - data["open"].shift(1)).abs()
    first_mid = (data["open"].shift(2) + data["close"].shift(2)) / 2
    out["morning_star"] = (
        first_bear & (middle_body <= 0.5 * first_body)
        & (data["close"] > data["open"]) & (data["close"] >= first_mid)
    ).fillna(False)
    out["evening_star"] = (
        first_bull & (middle_body <= 0.5 * first_body)
        & (data["close"] < data["open"]) & (data["close"] <= first_mid)
    ).fillna(False)
    return out


def _pivot_mask(
    series: pd.Series,
    left: int,
    right: int,
    mode: Literal["high", "low"],
    tie_policy: TiePolicy,
) -> pd.Series:
    values = series.to_numpy(float)
    result = np.zeros(len(values), dtype=bool)
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        extreme = np.nanmax(window) if mode == "high" else np.nanmin(window)
        positions = np.flatnonzero(window == extreme)
        if values[i] != extreme:
            continue
        center = left
        valid = (
            len(positions) == 1 if tie_policy == "strict"
            else positions[0] == center if tie_policy == "first"
            else positions[-1] == center
        )
        result[i] = valid
    return pd.Series(result, index=series.index)


def confirmed_fractals(data: pd.DataFrame, config: MarketStructureConfig) -> pd.DataFrame:
    raw_high = _pivot_mask(
        data["high"], config.fractal_left, config.fractal_right,
        "high", config.fractal_tie_policy,
    )
    raw_low = _pivot_mask(
        data["low"], config.fractal_left, config.fractal_right,
        "low", config.fractal_tie_policy,
    )
    out = pd.DataFrame(index=data.index)
    out["raw_fractal_high"] = raw_high
    out["raw_fractal_low"] = raw_low
    out["confirmed_fractal_high"] = raw_high.shift(
        config.fractal_right, fill_value=False
    )
    out["confirmed_fractal_low"] = raw_low.shift(
        config.fractal_right, fill_value=False
    )
    out["confirmed_fractal_high_price"] = data["high"].where(raw_high).shift(
        config.fractal_right
    )
    out["confirmed_fractal_low_price"] = data["low"].where(raw_low).shift(
        config.fractal_right
    )
    time_series = pd.Series(data.index, index=data.index)
    out["confirmed_fractal_high_source_time"] = time_series.where(raw_high).shift(
        config.fractal_right
    )
    out["confirmed_fractal_low_source_time"] = time_series.where(raw_low).shift(
        config.fractal_right
    )
    return out


def displacement_features(
    data: pd.DataFrame, config: MarketStructureConfig, atr_series: pd.Series
) -> pd.DataFrame:
    g = candle_geometry(data)
    out = pd.DataFrame(index=data.index)
    out["volume_z"] = rolling_zscore(data["volume"], config.volume_z_window)
    out["range_in_atr"] = g["candle_range"] / atr_series.replace(0, np.nan)
    base = (
        (out["range_in_atr"] >= config.displacement_atr_multiple)
        & (g["body_fraction"] >= config.displacement_body_fraction)
        & (out["volume_z"] >= config.displacement_volume_z_min)
    )
    out["bullish_displacement"] = (
        base & g["bullish_candle"] & (g["close_location"] >= 0.70)
    ).fillna(False)
    out["bearish_displacement"] = (
        base & g["bearish_candle"] & (g["close_location"] <= 0.30)
    ).fillna(False)
    return out


def market_structure_breaks(
    data: pd.DataFrame,
    fractals: pd.DataFrame,
    config: MarketStructureConfig,
    atr_series: pd.Series,
) -> pd.DataFrame:
    n = len(data)
    columns = {name: np.zeros(n, dtype=bool) for name in (
        "bullish_structure_break", "bearish_structure_break",
        "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch",
    )}
    last_highs, last_lows = np.full(n, np.nan), np.full(n, np.nan)
    states = np.zeros(n, dtype=np.int8)
    current_high = current_low = np.nan
    high_age = low_age = 10**9
    state = 0

    for i in range(n):
        if bool(fractals["confirmed_fractal_high"].iloc[i]):
            current_high = float(fractals["confirmed_fractal_high_price"].iloc[i])
            high_age = 0
        else:
            high_age += 1
        if bool(fractals["confirmed_fractal_low"].iloc[i]):
            current_low = float(fractals["confirmed_fractal_low_price"].iloc[i])
            low_age = 0
        else:
            low_age += 1
        last_highs[i], last_lows[i] = current_high, current_low
        if pd.isna(atr_series.iloc[i]):
            states[i] = state
            continue
        up_test = float(data["close"].iloc[i] if config.break_basis == "close" else data["high"].iloc[i])
        down_test = float(data["close"].iloc[i] if config.break_basis == "close" else data["low"].iloc[i])
        buffer = float(atr_series.iloc[i]) * config.break_buffer_atr
        up = np.isfinite(current_high) and high_age <= config.structure_memory_bars and up_test > current_high + buffer
        down = np.isfinite(current_low) and low_age <= config.structure_memory_bars and down_test < current_low - buffer
        columns["bullish_structure_break"][i] = up
        columns["bearish_structure_break"][i] = down
        if up and not down:
            columns["bullish_choch" if state < 0 else "bullish_bos"][i] = True
            state, current_high, high_age = 1, np.nan, 10**9
        elif down and not up:
            columns["bearish_choch" if state > 0 else "bearish_bos"][i] = True
            state, current_low, low_age = -1, np.nan, 10**9
        states[i] = state

    out = pd.DataFrame(columns, index=data.index)
    out["last_confirmed_swing_high"] = last_highs
    out["last_confirmed_swing_low"] = last_lows
    out["structure_state"] = states
    return out


def liquidity_sweeps(
    data: pd.DataFrame,
    fractals: pd.DataFrame,
    config: MarketStructureConfig,
    atr_series: pd.Series,
) -> pd.DataFrame:
    n = len(data)
    bullish, bearish = np.zeros(n, bool), np.zeros(n, bool)
    swept_low, swept_high = np.full(n, np.nan), np.full(n, np.nan)
    current_high = current_low = np.nan
    high_age = low_age = 10**9
    for i in range(n):
        if bool(fractals["confirmed_fractal_high"].iloc[i]):
            current_high = float(fractals["confirmed_fractal_high_price"].iloc[i]); high_age = 0
        else:
            high_age += 1
        if bool(fractals["confirmed_fractal_low"].iloc[i]):
            current_low = float(fractals["confirmed_fractal_low_price"].iloc[i]); low_age = 0
        else:
            low_age += 1
        if pd.isna(atr_series.iloc[i]):
            continue
        a = float(atr_series.iloc[i])
        sweep_buffer = a * config.sweep_buffer_atr
        reclaim_buffer = a * config.sweep_reclaim_buffer_atr
        if (
            np.isfinite(current_high) and high_age <= config.sweep_max_age_bars
            and float(data["high"].iloc[i]) > current_high + sweep_buffer
            and float(data["close"].iloc[i]) < current_high - reclaim_buffer
        ):
            bearish[i], swept_high[i] = True, current_high
            current_high, high_age = np.nan, 10**9
        if (
            np.isfinite(current_low) and low_age <= config.sweep_max_age_bars
            and float(data["low"].iloc[i]) < current_low - sweep_buffer
            and float(data["close"].iloc[i]) > current_low + reclaim_buffer
        ):
            bullish[i], swept_low[i] = True, current_low
            current_low, low_age = np.nan, 10**9
    return pd.DataFrame(
        {
            "bullish_liquidity_sweep": bullish,
            "bearish_liquidity_sweep": bearish,
            "swept_low_level": swept_low,
            "swept_high_level": swept_high,
        }, index=data.index,
    )


def equal_highs_lows(
    fractals: pd.DataFrame,
    config: MarketStructureConfig,
    atr_series: pd.Series,
) -> pd.DataFrame:
    n = len(fractals)
    eq_high, eq_low = np.zeros(n, bool), np.zeros(n, bool)
    high_level, low_level = np.full(n, np.nan), np.full(n, np.nan)
    prev_high = prev_low = np.nan
    prev_high_i = prev_low_i = -10**9
    for i in range(n):
        if pd.isna(atr_series.iloc[i]):
            continue
        tolerance = float(atr_series.iloc[i]) * config.equal_level_tolerance_atr
        if bool(fractals["confirmed_fractal_high"].iloc[i]):
            price = float(fractals["confirmed_fractal_high_price"].iloc[i])
            separation = i - prev_high_i
            if (
                np.isfinite(prev_high)
                and config.equal_level_min_separation <= separation <= config.equal_level_max_separation
                and abs(price - prev_high) <= tolerance
            ):
                eq_high[i], high_level[i] = True, (price + prev_high) / 2
            prev_high, prev_high_i = price, i
        if bool(fractals["confirmed_fractal_low"].iloc[i]):
            price = float(fractals["confirmed_fractal_low_price"].iloc[i])
            separation = i - prev_low_i
            if (
                np.isfinite(prev_low)
                and config.equal_level_min_separation <= separation <= config.equal_level_max_separation
                and abs(price - prev_low) <= tolerance
            ):
                eq_low[i], low_level[i] = True, (price + prev_low) / 2
            prev_low, prev_low_i = price, i
    return pd.DataFrame(
        {
            "equal_highs": eq_high, "equal_lows": eq_low,
            "equal_highs_level": high_level, "equal_lows_level": low_level,
        }, index=fractals.index,
    )


def _track_zone(
    data: pd.DataFrame,
    out: pd.DataFrame,
    event: str,
    lower_col: str,
    upper_col: str,
    prefix: str,
    direction: Literal["bullish", "bearish"],
    max_age: int,
    invalidate_on_close: bool,
) -> None:
    n = len(data)
    active, mitigated, invalidated = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    active_lower, active_upper, ages = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    lower = upper = np.nan
    age = 10**9
    zone_active = touched = False
    for i in range(n):
        if bool(out[event].iloc[i]):
            lower, upper = float(out[lower_col].iloc[i]), float(out[upper_col].iloc[i])
            age, zone_active, touched = 0, True, False
        if zone_active:
            overlaps = float(data["low"].iloc[i]) <= upper and float(data["high"].iloc[i]) >= lower
            if age > 0 and overlaps and not touched:
                mitigated[i], touched = True, True
            broken = (
                float(data["close"].iloc[i]) < lower if direction == "bullish"
                else float(data["close"].iloc[i]) > upper
            )
            if invalidate_on_close and age > 0 and broken:
                invalidated[i], zone_active = True, False
            if age > max_age:
                zone_active = False
        if zone_active:
            active[i], active_lower[i], active_upper[i], ages[i] = True, lower, upper, age
            age += 1
    out[f"{prefix}_active"] = active
    out[f"{prefix}_active_lower"] = active_lower
    out[f"{prefix}_active_upper"] = active_upper
    out[f"{prefix}_mitigated"] = mitigated
    out[f"{prefix}_invalidated"] = invalidated
    out[f"{prefix}_age"] = ages


def fair_value_gaps(
    data: pd.DataFrame,
    displacement: pd.DataFrame,
    config: MarketStructureConfig,
    atr_series: pd.Series,
) -> pd.DataFrame:
    bullish_gap = data["low"] - data["high"].shift(2)
    bearish_gap = data["low"].shift(2) - data["high"]
    bullish = bullish_gap > atr_series * config.fvg_min_atr
    bearish = bearish_gap > atr_series * config.fvg_min_atr
    if config.fvg_require_displacement:
        bullish &= displacement["bullish_displacement"].shift(1, fill_value=False)
        bearish &= displacement["bearish_displacement"].shift(1, fill_value=False)
    out = pd.DataFrame(index=data.index)
    out["bullish_fvg"] = bullish.fillna(False)
    out["bearish_fvg"] = bearish.fillna(False)
    out["bullish_fvg_lower"] = data["high"].shift(2).where(out["bullish_fvg"])
    out["bullish_fvg_upper"] = data["low"].where(out["bullish_fvg"])
    out["bearish_fvg_lower"] = data["high"].where(out["bearish_fvg"])
    out["bearish_fvg_upper"] = data["low"].shift(2).where(out["bearish_fvg"])
    _track_zone(data, out, "bullish_fvg", "bullish_fvg_lower", "bullish_fvg_upper", "bullish_fvg", "bullish", config.fvg_max_age_bars, False)
    _track_zone(data, out, "bearish_fvg", "bearish_fvg_lower", "bearish_fvg_upper", "bearish_fvg", "bearish", config.fvg_max_age_bars, False)
    return out


def _candle_zone(data: pd.DataFrame, i: int, mode: RangeMode) -> tuple[float, float]:
    if mode == "full":
        return float(data["low"].iloc[i]), float(data["high"].iloc[i])
    return (
        float(min(data["open"].iloc[i], data["close"].iloc[i])),
        float(max(data["open"].iloc[i], data["close"].iloc[i])),
    )


def order_blocks(
    data: pd.DataFrame,
    structure: pd.DataFrame,
    displacement: pd.DataFrame,
    config: MarketStructureConfig,
) -> pd.DataFrame:
    """OHLCV proxy: last opposite candle before displacement-backed BOS/CHoCH."""
    n = len(data)
    bull, bear = np.zeros(n, bool), np.zeros(n, bool)
    bl, bu, sl, su = (np.full(n, np.nan) for _ in range(4))
    bull_source, bear_source = np.full(n, -1, int), np.full(n, -1, int)
    bullish_break = (structure["bullish_bos"] | structure["bullish_choch"]).to_numpy(bool)
    bearish_break = (structure["bearish_bos"] | structure["bearish_choch"]).to_numpy(bool)
    for i in range(n):
        start = max(0, i - config.order_block_lookback)
        if bullish_break[i] and (
            not config.order_block_require_displacement or bool(displacement["bullish_displacement"].iloc[i])
        ):
            candidates = [j for j in range(i - 1, start - 1, -1) if data["close"].iloc[j] < data["open"].iloc[j]]
            if candidates:
                j = candidates[0]
                bl[i], bu[i] = _candle_zone(data, j, config.order_block_range)
                bull[i], bull_source[i] = True, j
        if bearish_break[i] and (
            not config.order_block_require_displacement or bool(displacement["bearish_displacement"].iloc[i])
        ):
            candidates = [j for j in range(i - 1, start - 1, -1) if data["close"].iloc[j] > data["open"].iloc[j]]
            if candidates:
                j = candidates[0]
                sl[i], su[i] = _candle_zone(data, j, config.order_block_range)
                bear[i], bear_source[i] = True, j
    out = pd.DataFrame(
        {
            "bullish_order_block": bull, "bearish_order_block": bear,
            "bullish_order_block_lower": bl, "bullish_order_block_upper": bu,
            "bearish_order_block_lower": sl, "bearish_order_block_upper": su,
        }, index=data.index,
    )
    bull_times = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]" if data.index.tz else "datetime64[ns]")
    bear_times = bull_times.copy()
    for i, j in enumerate(bull_source):
        if j >= 0: bull_times.iloc[i] = data.index[j]
    for i, j in enumerate(bear_source):
        if j >= 0: bear_times.iloc[i] = data.index[j]
    out["bullish_order_block_source_time"] = bull_times
    out["bearish_order_block_source_time"] = bear_times
    _track_zone(data, out, "bullish_order_block", "bullish_order_block_lower", "bullish_order_block_upper", "bullish_order_block", "bullish", config.order_block_max_age_bars, True)
    _track_zone(data, out, "bearish_order_block", "bearish_order_block_lower", "bearish_order_block_upper", "bearish_order_block", "bearish", config.order_block_max_age_bars, True)
    return out


def premium_discount_zones(data: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    high = structure["last_confirmed_swing_high"]
    low = structure["last_confirmed_swing_low"]
    valid = high.notna() & low.notna() & (high > low)
    midpoint = ((high + low) / 2).where(valid)
    return pd.DataFrame(
        {
            "dealing_range_high": high.where(valid),
            "dealing_range_low": low.where(valid),
            "dealing_range_midpoint": midpoint,
            "in_discount": (data["close"] < midpoint).fillna(False),
            "in_premium": (data["close"] > midpoint).fillna(False),
            "at_equilibrium": ((data["close"] - midpoint).abs() <= 0.001 * midpoint).fillna(False),
        }, index=data.index,
    )


def build_market_structure_features(
    data: pd.DataFrame,
    config: MarketStructureConfig | None = None,
) -> pd.DataFrame:
    config = config or MarketStructureConfig()
    frame = validate_ohlcv(data)
    atr_series = atr(frame, config.atr_period).rename("atr")
    candles = candlestick_patterns(frame, config)
    fractals = confirmed_fractals(frame, config)
    displacement = displacement_features(frame, config, atr_series)
    structure = market_structure_breaks(frame, fractals, config, atr_series)
    sweeps = liquidity_sweeps(frame, fractals, config, atr_series)
    equal_levels = equal_highs_lows(fractals, config, atr_series)
    fvgs = fair_value_gaps(frame, displacement, config, atr_series)
    obs = order_blocks(frame, structure, displacement, config)
    premium_discount = premium_discount_zones(frame, structure)
    out = pd.concat(
        [frame, atr_series, candles, fractals, displacement, structure,
         sweeps, equal_levels, fvgs, obs, premium_discount], axis=1
    )
    return out.loc[:, ~out.columns.duplicated(keep="first")]


def example_long_signal(features: pd.DataFrame) -> pd.Series:
    """Example only. Execute the signal no earlier than the next candle open."""
    return (
        features["bullish_liquidity_sweep"]
        & (features["structure_state"] >= 0)
        & features["in_discount"]
        & (features["bullish_pin_bar"] | features["bullish_engulfing"])
    ).fillna(False)



def liquidity_sweep_reversal_strategy(
    data: pd.DataFrame,
    parameters: dict[str, float | int],
):
    """Adapter for ``simple_backtest_engine.BacktestEngine``.

    Required parameter names:
        fractal_left, fractal_right, atr_period, sweep_buffer_atr,
        displacement_atr_multiple, stop_pct, target_pct

    Entry:
        bullish liquidity sweep, non-bearish structure, discount location, and
        bullish candle confirmation.

    Exit:
        bearish sweep, bearish CHoCH, or bearish BOS.
    """
    try:
        from simple_backtest_engine import StrategySignals
    except ImportError as exc:
        raise ImportError(
            "Place market_structure_patterns.py beside simple_backtest_engine.py"
        ) from exc

    config = MarketStructureConfig(
        fractal_left=int(parameters["fractal_left"]),
        fractal_right=int(parameters["fractal_right"]),
        atr_period=int(parameters["atr_period"]),
        sweep_buffer_atr=float(parameters["sweep_buffer_atr"]),
        displacement_atr_multiple=float(parameters["displacement_atr_multiple"]),
    )
    features = build_market_structure_features(data, config)
    entry = example_long_signal(features)
    exit_signal = (
        features["bearish_liquidity_sweep"]
        | features["bearish_choch"]
        | features["bearish_bos"]
    ).fillna(False)
    return StrategySignals(
        entry=entry,
        exit=exit_signal,
        stop_pct=float(parameters["stop_pct"]) / 100.0,
        target_pct=float(parameters["target_pct"]) / 100.0,
    )


def default_market_structure_parameter_specs():
    """Parameter grid compatible with the engine's ``ParameterSpec`` class."""
    try:
        from simple_backtest_engine import ParameterSpec
    except ImportError as exc:
        raise ImportError(
            "Place market_structure_patterns.py beside simple_backtest_engine.py"
        ) from exc
    return [
        ParameterSpec("fractal_left", "integer", 2, 5, 1),
        ParameterSpec("fractal_right", "integer", 2, 5, 1),
        ParameterSpec("atr_period", "integer", 10, 20, 1),
        ParameterSpec("sweep_buffer_atr", "half", 0.0, 1.0),
        ParameterSpec("displacement_atr_multiple", "half", 1.0, 2.5),
        ParameterSpec("stop_pct", "half", 1.0, 5.0),
        ParameterSpec("target_pct", "half", 2.0, 10.0),
    ]


def default_market_structure_parameters() -> dict[str, float | int]:
    return {
        "fractal_left": 2,
        "fractal_right": 2,
        "atr_period": 14,
        "sweep_buffer_atr": 0.0,
        "displacement_atr_multiple": 1.5,
        "stop_pct": 2.5,
        "target_pct": 5.0,
    }


def market_structure_parameter_constraint(parameters: dict[str, float | int]) -> bool:
    return (
        int(parameters["fractal_left"]) >= 1
        and int(parameters["fractal_right"]) >= 1
        and int(parameters["atr_period"]) >= 2
        and float(parameters["target_pct"]) > float(parameters["stop_pct"])
    )


def summarize_events(features: pd.DataFrame) -> pd.Series:
    names = [
        "confirmed_fractal_high", "confirmed_fractal_low",
        "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch",
        "bullish_liquidity_sweep", "bearish_liquidity_sweep",
        "equal_highs", "equal_lows", "bullish_fvg", "bearish_fvg",
        "bullish_order_block", "bearish_order_block",
        "bullish_engulfing", "bearish_engulfing",
        "bullish_pin_bar", "bearish_pin_bar",
    ]
    available = [name for name in names if name in features]
    return features[available].sum().sort_values(ascending=False)


__all__ = [
    "MarketStructureConfig", "atr", "build_market_structure_features",
    "candlestick_patterns", "confirmed_fractals", "displacement_features",
    "equal_highs_lows", "example_long_signal", "fair_value_gaps",
    "liquidity_sweeps", "liquidity_sweep_reversal_strategy",
    "default_market_structure_parameter_specs",
    "default_market_structure_parameters",
    "market_structure_parameter_constraint",
    "market_structure_breaks", "order_blocks",
    "premium_discount_zones", "summarize_events", "validate_ohlcv",
]
