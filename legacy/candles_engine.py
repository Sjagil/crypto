
from __future__ import annotations

"""
Professional deterministic candlestick classification engine.

The engine analyzes every closed OHLCV candle and produces:

1. Candle geometry
   - body, range, upper/lower wick
   - body/range and wick/range ratios
   - ATR-normalized range and body
   - close location and gap information

2. Market context, using prior candles only
   - uptrend, downtrend, range
   - volatility regime
   - volume regime

3. Single-candle classifications
   - doji family
   - spinning top / high-wave
   - hammer / hanging man
   - inverted hammer / shooting star
   - marubozu variants
   - pin bars
   - belt holds
   - strong and weak directional candles

4. Multi-candle patterns
   - engulfing and harami
   - piercing line and dark-cloud cover
   - tweezer top / bottom
   - inside and outside bars
   - morning and evening stars
   - three white soldiers / three black crows
   - three inside up / down
   - three outside up / down
   - rising and falling three methods
   - bullish and bearish kicker

5. Human-readable output
   - dominant_candle_type
   - candle_family
   - bias
   - strength_score
   - confidence_score
   - pattern_labels
   - explanation

Anti-look-ahead
---------------
All trend and regime context columns are shifted by one candle. Patterns on
candle t use candle t and historical candles only. Execute a strategy signal
no earlier than candle t+1 open.

Expected columns
----------------
open, high, low, close, volume

Basic use
---------
    engine = CandleEngine()
    features = engine.analyze(data)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


TrendState = Literal["uptrend", "downtrend", "range", "unknown"]


@dataclass(frozen=True)
class CandleEngineConfig:
    # Core normalization
    atr_period: int = 14
    volume_window: int = 50
    range_percentile_window: int = 100
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    trend_slope_bars: int = 5
    trend_return_lookback: int = 10

    # Candle shape thresholds
    doji_body_fraction: float = 0.10
    four_price_doji_range_atr: float = 0.03
    spinning_top_body_fraction: float = 0.30
    high_wave_wick_fraction: float = 0.35
    long_body_fraction: float = 0.65
    marubozu_body_fraction: float = 0.90
    marubozu_max_wick_fraction: float = 0.05
    strong_close_location: float = 0.80
    weak_close_location: float = 0.35

    # Pin / hammer geometry
    long_wick_to_body: float = 2.0
    extreme_wick_to_body: float = 3.0
    opposite_wick_max_body: float = 0.75
    hammer_body_position_min: float = 0.60
    shooting_body_position_max: float = 0.40

    # Relative candle size
    long_range_atr: float = 1.25
    very_long_range_atr: float = 2.00
    narrow_range_atr: float = 0.65

    # Pattern tolerance
    body_equality_atr: float = 0.10
    tweezer_tolerance_atr: float = 0.08
    gap_tolerance_atr: float = 0.02
    star_small_body_ratio: float = 0.45
    soldiers_min_body_fraction: float = 0.55
    soldiers_max_upper_wick_fraction: float = 0.20
    crows_max_lower_wick_fraction: float = 0.20

    # Context
    trend_min_return: float = 0.005
    trend_min_ema_separation_atr: float = 0.10
    volume_z_high: float = 1.0
    volume_z_extreme: float = 2.0
    volume_z_low: float = -1.0

    # Output
    explanation_decimal_places: int = 2

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "atr_period": self.atr_period,
            "volume_window": self.volume_window,
            "range_percentile_window": self.range_percentile_window,
            "trend_ema_fast": self.trend_ema_fast,
            "trend_ema_slow": self.trend_ema_slow,
            "trend_slope_bars": self.trend_slope_bars,
            "trend_return_lookback": self.trend_return_lookback,
        }
        for name, value in positive_integer_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.trend_ema_fast >= self.trend_ema_slow:
            raise ValueError("trend_ema_fast must be below trend_ema_slow")

        fractions = {
            "doji_body_fraction": self.doji_body_fraction,
            "spinning_top_body_fraction": self.spinning_top_body_fraction,
            "high_wave_wick_fraction": self.high_wave_wick_fraction,
            "long_body_fraction": self.long_body_fraction,
            "marubozu_body_fraction": self.marubozu_body_fraction,
            "marubozu_max_wick_fraction": self.marubozu_max_wick_fraction,
            "strong_close_location": self.strong_close_location,
            "weak_close_location": self.weak_close_location,
            "hammer_body_position_min": self.hammer_body_position_min,
            "shooting_body_position_max": self.shooting_body_position_max,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    columns = {str(column).lower(): column for column in data.columns}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = data.rename(columns={original: lower for lower, original in columns.items()})
    frame = frame[["open", "high", "low", "close", "volume"]].copy()

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a pandas DatetimeIndex")

    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("OHLCV index contains duplicate timestamps")

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


def atr(data: pd.DataFrame, period: int) -> pd.Series:
    return true_range(data).ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid <= values[-1]))

    return series.rolling(window, min_periods=window).apply(
        rank_last,
        raw=True,
    )


def geometry(data: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)

    body_high = data[["open", "close"]].max(axis=1)
    body_low = data[["open", "close"]].min(axis=1)

    result["candle_range"] = data["high"] - data["low"]
    result["real_body"] = (data["close"] - data["open"]).abs()
    result["body_high"] = body_high
    result["body_low"] = body_low
    result["upper_wick"] = data["high"] - body_high
    result["lower_wick"] = body_low - data["low"]

    safe_range = result["candle_range"].replace(0.0, np.nan)
    safe_body = result["real_body"].replace(0.0, np.nan)

    result["body_fraction"] = result["real_body"] / safe_range
    result["upper_wick_fraction"] = result["upper_wick"] / safe_range
    result["lower_wick_fraction"] = result["lower_wick"] / safe_range
    result["upper_wick_to_body"] = result["upper_wick"] / safe_body
    result["lower_wick_to_body"] = result["lower_wick"] / safe_body

    result["close_location"] = (data["close"] - data["low"]) / safe_range
    result["open_location"] = (data["open"] - data["low"]) / safe_range
    result["body_midpoint"] = (body_high + body_low) / 2.0

    result["bullish"] = data["close"] > data["open"]
    result["bearish"] = data["close"] < data["open"]
    result["flat"] = data["close"] == data["open"]

    result["range_atr"] = result["candle_range"] / atr_series.replace(0.0, np.nan)
    result["body_atr"] = result["real_body"] / atr_series.replace(0.0, np.nan)

    previous_close = data["close"].shift(1)
    result["opening_gap"] = data["open"] - previous_close
    result["opening_gap_pct"] = data["open"] / previous_close - 1.0
    result["opening_gap_atr"] = result["opening_gap"] / atr_series.replace(0.0, np.nan)
    result["gap_up"] = data["low"] > data["high"].shift(1)
    result["gap_down"] = data["high"] < data["low"].shift(1)
    result["body_gap_up"] = body_low > result["body_high"].shift(1)
    result["body_gap_down"] = body_high < result["body_low"].shift(1)

    return result


def market_context(
    data: pd.DataFrame,
    geometry_frame: pd.DataFrame,
    atr_series: pd.Series,
    config: CandleEngineConfig,
) -> pd.DataFrame:
    """
    Context is shifted one candle, preventing current candle information from
    defining its own prior trend.
    """
    result = pd.DataFrame(index=data.index)

    fast = data["close"].ewm(
        span=config.trend_ema_fast,
        adjust=False,
        min_periods=config.trend_ema_fast,
    ).mean()
    slow = data["close"].ewm(
        span=config.trend_ema_slow,
        adjust=False,
        min_periods=config.trend_ema_slow,
    ).mean()

    fast_prior = fast.shift(1)
    slow_prior = slow.shift(1)
    close_prior = data["close"].shift(1)
    atr_prior = atr_series.shift(1)
    slope_prior = fast.diff(config.trend_slope_bars).shift(1)
    return_prior = data["close"].pct_change(config.trend_return_lookback).shift(1)
    ema_separation = (fast_prior - slow_prior) / atr_prior.replace(0.0, np.nan)

    uptrend = (
        (fast_prior > slow_prior)
        & (slope_prior > 0)
        & (return_prior >= config.trend_min_return)
        & (ema_separation >= config.trend_min_ema_separation_atr)
        & (close_prior > slow_prior)
    )
    downtrend = (
        (fast_prior < slow_prior)
        & (slope_prior < 0)
        & (return_prior <= -config.trend_min_return)
        & (ema_separation <= -config.trend_min_ema_separation_atr)
        & (close_prior < slow_prior)
    )

    trend_state = pd.Series("range", index=data.index, dtype="object")
    trend_state.loc[uptrend] = "uptrend"
    trend_state.loc[downtrend] = "downtrend"
    trend_state.loc[slow_prior.isna()] = "unknown"

    volume_z = rolling_zscore(data["volume"], config.volume_window)
    volume_regime = pd.Series("normal", index=data.index, dtype="object")
    volume_regime.loc[volume_z >= config.volume_z_extreme] = "extreme"
    volume_regime.loc[
        (volume_z >= config.volume_z_high)
        & (volume_z < config.volume_z_extreme)
    ] = "high"
    volume_regime.loc[volume_z <= config.volume_z_low] = "low"
    volume_regime.loc[volume_z.isna()] = "unknown"

    range_percentile = rolling_percentile_rank(
        geometry_frame["candle_range"],
        config.range_percentile_window,
    )
    volatility_regime = pd.Series("normal", index=data.index, dtype="object")
    volatility_regime.loc[range_percentile >= 0.90] = "extreme"
    volatility_regime.loc[
        (range_percentile >= 0.70) & (range_percentile < 0.90)
    ] = "high"
    volatility_regime.loc[range_percentile <= 0.30] = "low"
    volatility_regime.loc[range_percentile.isna()] = "unknown"

    result["trend_ema_fast"] = fast
    result["trend_ema_slow"] = slow
    result["prior_trend_state"] = trend_state
    result["prior_return"] = return_prior
    result["prior_ema_separation_atr"] = ema_separation
    result["volume_z"] = volume_z
    result["volume_regime"] = volume_regime
    result["range_percentile"] = range_percentile
    result["volatility_regime"] = volatility_regime
    return result


def single_candle_patterns(
    data: pd.DataFrame,
    g: pd.DataFrame,
    context: pd.DataFrame,
    config: CandleEngineConfig,
) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    safe_body = g["real_body"].replace(0.0, np.nan)

    doji = g["body_fraction"] <= config.doji_body_fraction
    long_upper = g["upper_wick_to_body"] >= config.long_wick_to_body
    long_lower = g["lower_wick_to_body"] >= config.long_wick_to_body
    tiny_upper = g["upper_wick"] <= config.marubozu_max_wick_fraction * g["candle_range"]
    tiny_lower = g["lower_wick"] <= config.marubozu_max_wick_fraction * g["candle_range"]

    result["four_price_doji"] = (
        (g["range_atr"] <= config.four_price_doji_range_atr)
        & doji
    ).fillna(False)

    result["dragonfly_doji"] = (
        doji
        & long_lower
        & (g["upper_wick"] <= config.opposite_wick_max_body * safe_body)
        & (g["close_location"] >= 0.75)
    ).fillna(False)

    result["gravestone_doji"] = (
        doji
        & long_upper
        & (g["lower_wick"] <= config.opposite_wick_max_body * safe_body)
        & (g["close_location"] <= 0.25)
    ).fillna(False)

    result["long_legged_doji"] = (
        doji
        & (g["upper_wick_fraction"] >= 0.30)
        & (g["lower_wick_fraction"] >= 0.30)
    ).fillna(False)

    result["standard_doji"] = (
        doji
        & ~result["four_price_doji"]
        & ~result["dragonfly_doji"]
        & ~result["gravestone_doji"]
        & ~result["long_legged_doji"]
    ).fillna(False)

    result["spinning_top"] = (
        (g["body_fraction"] > config.doji_body_fraction)
        & (g["body_fraction"] <= config.spinning_top_body_fraction)
        & (g["upper_wick_fraction"] >= 0.20)
        & (g["lower_wick_fraction"] >= 0.20)
    ).fillna(False)

    result["high_wave"] = (
        (g["body_fraction"] <= config.spinning_top_body_fraction)
        & (g["upper_wick_fraction"] >= config.high_wave_wick_fraction)
        & (g["lower_wick_fraction"] >= config.high_wave_wick_fraction)
    ).fillna(False)

    result["bullish_marubozu"] = (
        g["bullish"]
        & (g["body_fraction"] >= config.marubozu_body_fraction)
        & tiny_upper
        & tiny_lower
    ).fillna(False)

    result["bearish_marubozu"] = (
        g["bearish"]
        & (g["body_fraction"] >= config.marubozu_body_fraction)
        & tiny_upper
        & tiny_lower
    ).fillna(False)

    result["bullish_opening_marubozu"] = (
        g["bullish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_lower
        & ~tiny_upper
    ).fillna(False)

    result["bullish_closing_marubozu"] = (
        g["bullish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_upper
        & ~tiny_lower
    ).fillna(False)

    result["bearish_opening_marubozu"] = (
        g["bearish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_upper
        & ~tiny_lower
    ).fillna(False)

    result["bearish_closing_marubozu"] = (
        g["bearish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_lower
        & ~tiny_upper
    ).fillna(False)

    lower_rejection_shape = (
        (g["lower_wick_to_body"] >= config.long_wick_to_body)
        & (g["upper_wick_to_body"] <= config.opposite_wick_max_body)
        & (g["body_midpoint"] >= data["low"] + config.hammer_body_position_min * g["candle_range"])
    )
    upper_rejection_shape = (
        (g["upper_wick_to_body"] >= config.long_wick_to_body)
        & (g["lower_wick_to_body"] <= config.opposite_wick_max_body)
        & (g["body_midpoint"] <= data["low"] + config.shooting_body_position_max * g["candle_range"])
    )

    result["bullish_pin_bar"] = (
        lower_rejection_shape & (g["close_location"] >= 0.60)
    ).fillna(False)
    result["bearish_pin_bar"] = (
        upper_rejection_shape & (g["close_location"] <= 0.40)
    ).fillna(False)

    result["hammer"] = (
        lower_rejection_shape
        & (context["prior_trend_state"] == "downtrend")
    ).fillna(False)
    result["hanging_man"] = (
        lower_rejection_shape
        & (context["prior_trend_state"] == "uptrend")
    ).fillna(False)
    result["inverted_hammer"] = (
        upper_rejection_shape
        & (context["prior_trend_state"] == "downtrend")
    ).fillna(False)
    result["shooting_star"] = (
        upper_rejection_shape
        & (context["prior_trend_state"] == "uptrend")
    ).fillna(False)

    result["bullish_belt_hold"] = (
        g["bullish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_lower
        & (g["close_location"] >= config.strong_close_location)
    ).fillna(False)
    result["bearish_belt_hold"] = (
        g["bearish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & tiny_upper
        & (g["close_location"] <= 1.0 - config.strong_close_location)
    ).fillna(False)

    result["long_bullish_candle"] = (
        g["bullish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & (g["range_atr"] >= config.long_range_atr)
    ).fillna(False)
    result["long_bearish_candle"] = (
        g["bearish"]
        & (g["body_fraction"] >= config.long_body_fraction)
        & (g["range_atr"] >= config.long_range_atr)
    ).fillna(False)

    result["bullish_displacement_candle"] = (
        result["long_bullish_candle"]
        & (g["close_location"] >= config.strong_close_location)
        & (context["volume_z"] >= config.volume_z_high)
    ).fillna(False)
    result["bearish_displacement_candle"] = (
        result["long_bearish_candle"]
        & (g["close_location"] <= 1.0 - config.strong_close_location)
        & (context["volume_z"] >= config.volume_z_high)
    ).fillna(False)

    result["narrow_range_candle"] = (
        g["range_atr"] <= config.narrow_range_atr
    ).fillna(False)
    result["very_long_range_candle"] = (
        g["range_atr"] >= config.very_long_range_atr
    ).fillna(False)

    result["shaved_head"] = tiny_upper.fillna(False)
    result["shaved_bottom"] = tiny_lower.fillna(False)

    return result


def multi_candle_patterns(
    data: pd.DataFrame,
    g: pd.DataFrame,
    context: pd.DataFrame,
    config: CandleEngineConfig,
) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    atr_series = g["candle_range"] / g["range_atr"].replace(0.0, np.nan)

    o0, h0, l0, c0 = data["open"], data["high"], data["low"], data["close"]
    o1, h1, l1, c1 = o0.shift(1), h0.shift(1), l0.shift(1), c0.shift(1)
    o2, h2, l2, c2 = o0.shift(2), h0.shift(2), l0.shift(2), c0.shift(2)
    o3, h3, l3, c3 = o0.shift(3), h0.shift(3), l0.shift(3), c0.shift(3)
    o4, h4, l4, c4 = o0.shift(4), h0.shift(4), l0.shift(4), c0.shift(4)

    bh0, bl0 = g["body_high"], g["body_low"]
    bh1, bl1 = bh0.shift(1), bl0.shift(1)
    body0, body1, body2 = g["real_body"], g["real_body"].shift(1), g["real_body"].shift(2)

    result["inside_bar"] = ((h0 < h1) & (l0 > l1)).fillna(False)
    result["outside_bar"] = ((h0 > h1) & (l0 < l1)).fillna(False)

    result["bullish_engulfing"] = (
        (c1 < o1)
        & (c0 > o0)
        & (bl0 <= bl1)
        & (bh0 >= bh1)
        & (body0 > body1)
    ).fillna(False)

    result["bearish_engulfing"] = (
        (c1 > o1)
        & (c0 < o0)
        & (bl0 <= bl1)
        & (bh0 >= bh1)
        & (body0 > body1)
    ).fillna(False)

    result["bullish_harami"] = (
        (c1 < o1)
        & (c0 > o0)
        & (bh0 < bh1)
        & (bl0 > bl1)
    ).fillna(False)

    result["bearish_harami"] = (
        (c1 > o1)
        & (c0 < o0)
        & (bh0 < bh1)
        & (bl0 > bl1)
    ).fillna(False)

    midpoint1 = (o1 + c1) / 2.0
    gap_tolerance = atr_series * config.gap_tolerance_atr

    result["piercing_line"] = (
        (c1 < o1)
        & (c0 > o0)
        & (o0 <= c1 + gap_tolerance)
        & (c0 > midpoint1)
        & (c0 < o1)
    ).fillna(False)

    result["dark_cloud_cover"] = (
        (c1 > o1)
        & (c0 < o0)
        & (o0 >= c1 - gap_tolerance)
        & (c0 < midpoint1)
        & (c0 > o1)
    ).fillna(False)

    result["tweezer_bottom"] = (
        (context["prior_trend_state"] == "downtrend")
        & ((l0 - l1).abs() <= atr_series * config.tweezer_tolerance_atr)
        & (c1 < o1)
        & (c0 > o0)
    ).fillna(False)

    result["tweezer_top"] = (
        (context["prior_trend_state"] == "uptrend")
        & ((h0 - h1).abs() <= atr_series * config.tweezer_tolerance_atr)
        & (c1 > o1)
        & (c0 < o0)
    ).fillna(False)

    first_midpoint = (o2 + c2) / 2.0
    small_middle = body1 <= config.star_small_body_ratio * body2

    result["morning_star"] = (
        (c2 < o2)
        & small_middle
        & (c0 > o0)
        & (c0 >= first_midpoint)
    ).fillna(False)

    result["evening_star"] = (
        (c2 > o2)
        & small_middle
        & (c0 < o0)
        & (c0 <= first_midpoint)
    ).fillna(False)

    # Kicker patterns. Body gaps are rare in crypto but valid in session markets.
    result["bullish_kicker"] = (
        (c1 < o1)
        & (c0 > o0)
        & (bl0 > bh1 - gap_tolerance)
        & (g["body_fraction"] >= config.long_body_fraction)
    ).fillna(False)

    result["bearish_kicker"] = (
        (c1 > o1)
        & (c0 < o0)
        & (bh0 < bl1 + gap_tolerance)
        & (g["body_fraction"] >= config.long_body_fraction)
    ).fillna(False)

    bullish_three = (c2 > o2) & (c1 > o1) & (c0 > o0)
    bearish_three = (c2 < o2) & (c1 < o1) & (c0 < o0)

    bodies_large = (
        (g["body_fraction"].shift(2) >= config.soldiers_min_body_fraction)
        & (g["body_fraction"].shift(1) >= config.soldiers_min_body_fraction)
        & (g["body_fraction"] >= config.soldiers_min_body_fraction)
    )

    result["three_white_soldiers"] = (
        bullish_three
        & bodies_large
        & (c1 > c2)
        & (c0 > c1)
        & (o1 >= bl2(data))
        & (o1 <= bh2(data))
        & (o0 >= bl1)
        & (o0 <= bh1)
        & (g["upper_wick_fraction"].shift(2) <= config.soldiers_max_upper_wick_fraction)
        & (g["upper_wick_fraction"].shift(1) <= config.soldiers_max_upper_wick_fraction)
        & (g["upper_wick_fraction"] <= config.soldiers_max_upper_wick_fraction)
    ).fillna(False)

    result["three_black_crows"] = (
        bearish_three
        & bodies_large
        & (c1 < c2)
        & (c0 < c1)
        & (o1 >= bl2(data))
        & (o1 <= bh2(data))
        & (o0 >= bl1)
        & (o0 <= bh1)
        & (g["lower_wick_fraction"].shift(2) <= config.crows_max_lower_wick_fraction)
        & (g["lower_wick_fraction"].shift(1) <= config.crows_max_lower_wick_fraction)
        & (g["lower_wick_fraction"] <= config.crows_max_lower_wick_fraction)
    ).fillna(False)

    result["three_inside_up"] = (
        (c2 < o2)
        & (c1 > o1)
        & (pd.concat([o1, c1], axis=1).max(axis=1) < pd.concat([o2, c2], axis=1).max(axis=1))
        & (pd.concat([o1, c1], axis=1).min(axis=1) > pd.concat([o2, c2], axis=1).min(axis=1))
        & (c0 > o0)
        & (c0 > o2)
    ).fillna(False)

    result["three_inside_down"] = (
        (c2 > o2)
        & (c1 < o1)
        & (pd.concat([o1, c1], axis=1).max(axis=1) < pd.concat([o2, c2], axis=1).max(axis=1))
        & (pd.concat([o1, c1], axis=1).min(axis=1) > pd.concat([o2, c2], axis=1).min(axis=1))
        & (c0 < o0)
        & (c0 < o2)
    ).fillna(False)

    result["three_outside_up"] = (
        result["bullish_engulfing"].shift(1, fill_value=False)
        & (c0 > o0)
        & (c0 > c1)
    ).fillna(False)

    result["three_outside_down"] = (
        result["bearish_engulfing"].shift(1, fill_value=False)
        & (c0 < o0)
        & (c0 < c1)
    ).fillna(False)

    # Five-candle continuation patterns.
    first_long_bull = (
        (c4 > o4)
        & (body0.shift(4) / (h4 - l4).replace(0.0, np.nan) >= config.long_body_fraction)
    )
    first_long_bear = (
        (c4 < o4)
        & (body0.shift(4) / (h4 - l4).replace(0.0, np.nan) >= config.long_body_fraction)
    )
    middle_small = (
        (body0.shift(3) < body0.shift(4) * 0.60)
        & (body0.shift(2) < body0.shift(4) * 0.60)
        & (body0.shift(1) < body0.shift(4) * 0.60)
    )
    middle_within_first = (
        (h3 <= h4) & (l3 >= l4)
        & (h2 <= h4) & (l2 >= l4)
        & (h1 <= h4) & (l1 >= l4)
    )

    result["rising_three_methods"] = (
        first_long_bull
        & middle_small
        & middle_within_first
        & (c0 > o0)
        & (c0 > h4)
    ).fillna(False)

    result["falling_three_methods"] = (
        first_long_bear
        & middle_small
        & middle_within_first
        & (c0 < o0)
        & (c0 < l4)
    ).fillna(False)

    return result


def bl2(data: pd.DataFrame) -> pd.Series:
    return data[["open", "close"]].min(axis=1).shift(2)


def bh2(data: pd.DataFrame) -> pd.Series:
    return data[["open", "close"]].max(axis=1).shift(2)


SINGLE_PATTERN_COLUMNS = [
    "four_price_doji",
    "dragonfly_doji",
    "gravestone_doji",
    "long_legged_doji",
    "standard_doji",
    "spinning_top",
    "high_wave",
    "bullish_marubozu",
    "bearish_marubozu",
    "bullish_opening_marubozu",
    "bullish_closing_marubozu",
    "bearish_opening_marubozu",
    "bearish_closing_marubozu",
    "bullish_pin_bar",
    "bearish_pin_bar",
    "hammer",
    "hanging_man",
    "inverted_hammer",
    "shooting_star",
    "bullish_belt_hold",
    "bearish_belt_hold",
    "long_bullish_candle",
    "long_bearish_candle",
    "bullish_displacement_candle",
    "bearish_displacement_candle",
    "narrow_range_candle",
    "very_long_range_candle",
    "shaved_head",
    "shaved_bottom",
]

MULTI_PATTERN_COLUMNS = [
    "inside_bar",
    "outside_bar",
    "bullish_engulfing",
    "bearish_engulfing",
    "bullish_harami",
    "bearish_harami",
    "piercing_line",
    "dark_cloud_cover",
    "tweezer_bottom",
    "tweezer_top",
    "morning_star",
    "evening_star",
    "bullish_kicker",
    "bearish_kicker",
    "three_white_soldiers",
    "three_black_crows",
    "three_inside_up",
    "three_inside_down",
    "three_outside_up",
    "three_outside_down",
    "rising_three_methods",
    "falling_three_methods",
]


BULLISH_PATTERNS = {
    "dragonfly_doji",
    "bullish_marubozu",
    "bullish_opening_marubozu",
    "bullish_closing_marubozu",
    "bullish_pin_bar",
    "hammer",
    "inverted_hammer",
    "bullish_belt_hold",
    "long_bullish_candle",
    "bullish_displacement_candle",
    "bullish_engulfing",
    "bullish_harami",
    "piercing_line",
    "tweezer_bottom",
    "morning_star",
    "bullish_kicker",
    "three_white_soldiers",
    "three_inside_up",
    "three_outside_up",
    "rising_three_methods",
}

BEARISH_PATTERNS = {
    "gravestone_doji",
    "bearish_marubozu",
    "bearish_opening_marubozu",
    "bearish_closing_marubozu",
    "bearish_pin_bar",
    "hanging_man",
    "shooting_star",
    "bearish_belt_hold",
    "long_bearish_candle",
    "bearish_displacement_candle",
    "bearish_engulfing",
    "bearish_harami",
    "dark_cloud_cover",
    "tweezer_top",
    "evening_star",
    "bearish_kicker",
    "three_black_crows",
    "three_inside_down",
    "three_outside_down",
    "falling_three_methods",
}

INDECISION_PATTERNS = {
    "four_price_doji",
    "standard_doji",
    "long_legged_doji",
    "spinning_top",
    "high_wave",
    "inside_bar",
}


# Higher number means the label takes precedence as dominant classification.
DOMINANT_PRIORITY = {
    "bullish_displacement_candle": 100,
    "bearish_displacement_candle": 100,
    "bullish_marubozu": 98,
    "bearish_marubozu": 98,
    "three_white_soldiers": 96,
    "three_black_crows": 96,
    "morning_star": 94,
    "evening_star": 94,
    "bullish_kicker": 93,
    "bearish_kicker": 93,
    "bullish_engulfing": 90,
    "bearish_engulfing": 90,
    "piercing_line": 88,
    "dark_cloud_cover": 88,
    "three_inside_up": 87,
    "three_inside_down": 87,
    "three_outside_up": 87,
    "three_outside_down": 87,
    "rising_three_methods": 86,
    "falling_three_methods": 86,
    "hammer": 84,
    "hanging_man": 84,
    "inverted_hammer": 84,
    "shooting_star": 84,
    "dragonfly_doji": 82,
    "gravestone_doji": 82,
    "bullish_pin_bar": 80,
    "bearish_pin_bar": 80,
    "tweezer_bottom": 78,
    "tweezer_top": 78,
    "bullish_belt_hold": 76,
    "bearish_belt_hold": 76,
    "bullish_harami": 72,
    "bearish_harami": 72,
    "long_legged_doji": 68,
    "high_wave": 66,
    "standard_doji": 64,
    "spinning_top": 62,
    "inside_bar": 60,
    "outside_bar": 59,
    "long_bullish_candle": 55,
    "long_bearish_candle": 55,
    "narrow_range_candle": 45,
}


def _humanize(label: str) -> str:
    return label.replace("_", " ")


def classify_output(
    data: pd.DataFrame,
    g: pd.DataFrame,
    context: pd.DataFrame,
    single: pd.DataFrame,
    multi: pd.DataFrame,
    config: CandleEngineConfig,
) -> pd.DataFrame:
    patterns = pd.concat([single, multi], axis=1)
    output = pd.DataFrame(index=data.index)

    labels_all: list[str] = []
    dominant_all: list[str] = []
    family_all: list[str] = []
    bias_all: list[str] = []
    strength_all: list[float] = []
    confidence_all: list[float] = []
    rejection_all: list[str] = []
    explanation_all: list[str] = []

    all_pattern_columns = [
        column
        for column in SINGLE_PATTERN_COLUMNS + MULTI_PATTERN_COLUMNS
        if column in patterns.columns
    ]

    decimals = config.explanation_decimal_places

    for i, timestamp in enumerate(data.index):
        active = [
            column for column in all_pattern_columns
            if bool(patterns[column].iloc[i])
        ]
        active_sorted = sorted(
            active,
            key=lambda name: DOMINANT_PRIORITY.get(name, 1),
            reverse=True,
        )

        if active_sorted:
            dominant = active_sorted[0]
        elif bool(g["bullish"].iloc[i]):
            dominant = "ordinary_bullish_candle"
        elif bool(g["bearish"].iloc[i]):
            dominant = "ordinary_bearish_candle"
        else:
            dominant = "flat_candle"

        bullish_score = sum(DOMINANT_PRIORITY.get(name, 30) for name in active if name in BULLISH_PATTERNS)
        bearish_score = sum(DOMINANT_PRIORITY.get(name, 30) for name in active if name in BEARISH_PATTERNS)
        indecision_score = sum(DOMINANT_PRIORITY.get(name, 30) for name in active if name in INDECISION_PATTERNS)

        # Geometry contributes, but less than named multi-candle patterns.
        body_fraction = float(g["body_fraction"].iloc[i]) if pd.notna(g["body_fraction"].iloc[i]) else 0.0
        range_atr = float(g["range_atr"].iloc[i]) if pd.notna(g["range_atr"].iloc[i]) else 0.0
        close_location = float(g["close_location"].iloc[i]) if pd.notna(g["close_location"].iloc[i]) else 0.5
        volume_z = float(context["volume_z"].iloc[i]) if pd.notna(context["volume_z"].iloc[i]) else 0.0

        geometry_strength = min(
            35.0,
            15.0 * body_fraction
            + 10.0 * min(range_atr / max(config.long_range_atr, 1e-9), 2.0)
            + 5.0 * min(max(volume_z, 0.0), 2.0),
        )

        if bool(g["bullish"].iloc[i]):
            bullish_score += geometry_strength * max(close_location, 0.20)
        elif bool(g["bearish"].iloc[i]):
            bearish_score += geometry_strength * max(1.0 - close_location, 0.20)

        if indecision_score > max(bullish_score, bearish_score) * 0.80:
            bias = "indecision"
        elif bullish_score > bearish_score * 1.10:
            bias = "bullish"
        elif bearish_score > bullish_score * 1.10:
            bias = "bearish"
        else:
            bias = "neutral"

        raw_strength = max(bullish_score, bearish_score, indecision_score)
        strength_score = float(np.clip(raw_strength / 1.50, 0.0, 100.0))

        dominant_priority = DOMINANT_PRIORITY.get(dominant, 35)
        context_bonus = 0.0
        trend = str(context["prior_trend_state"].iloc[i])
        if dominant in {"hammer", "inverted_hammer", "tweezer_bottom", "morning_star"} and trend == "downtrend":
            context_bonus += 10.0
        if dominant in {"hanging_man", "shooting_star", "tweezer_top", "evening_star"} and trend == "uptrend":
            context_bonus += 10.0
        if volume_z >= config.volume_z_high:
            context_bonus += 5.0
        if range_atr >= config.long_range_atr:
            context_bonus += 5.0

        confidence = float(np.clip(
            35.0
            + dominant_priority * 0.45
            + min(len(active), 4) * 3.0
            + context_bonus,
            0.0,
            100.0,
        ))

        upper_ratio = float(g["upper_wick_fraction"].iloc[i]) if pd.notna(g["upper_wick_fraction"].iloc[i]) else 0.0
        lower_ratio = float(g["lower_wick_fraction"].iloc[i]) if pd.notna(g["lower_wick_fraction"].iloc[i]) else 0.0
        if lower_ratio >= upper_ratio * 1.5 and lower_ratio >= 0.30:
            rejection = "lower_price_rejection"
        elif upper_ratio >= lower_ratio * 1.5 and upper_ratio >= 0.30:
            rejection = "higher_price_rejection"
        elif upper_ratio >= 0.25 and lower_ratio >= 0.25:
            rejection = "two_sided_rejection"
        else:
            rejection = "limited_rejection"

        if dominant in BULLISH_PATTERNS:
            family = "bullish_pattern"
        elif dominant in BEARISH_PATTERNS:
            family = "bearish_pattern"
        elif dominant in INDECISION_PATTERNS:
            family = "indecision_pattern"
        elif "bullish" in dominant:
            family = "bullish_basic"
        elif "bearish" in dominant:
            family = "bearish_basic"
        else:
            family = "neutral_basic"

        label_text = ";".join(active_sorted)
        explanation_parts = [
            f"type={_humanize(dominant)}",
            f"bias={bias}",
            f"prior_trend={trend}",
            f"body={body_fraction:.{decimals}f}",
            f"range_atr={range_atr:.{decimals}f}",
            f"close_location={close_location:.{decimals}f}",
            f"volume_z={volume_z:.{decimals}f}",
            f"rejection={rejection}",
        ]
        if active_sorted:
            explanation_parts.append(
                "patterns=" + ", ".join(_humanize(name) for name in active_sorted[:6])
            )

        labels_all.append(label_text)
        dominant_all.append(dominant)
        family_all.append(family)
        bias_all.append(bias)
        strength_all.append(round(strength_score, 4))
        confidence_all.append(round(confidence, 4))
        rejection_all.append(rejection)
        explanation_all.append(" | ".join(explanation_parts))

    output["dominant_candle_type"] = dominant_all
    output["candle_family"] = family_all
    output["candle_bias"] = bias_all
    output["candle_strength_score"] = strength_all
    output["classification_confidence"] = confidence_all
    output["rejection_type"] = rejection_all
    output["pattern_labels"] = labels_all
    output["pattern_count"] = [0 if not value else len(value.split(";")) for value in labels_all]
    output["explanation"] = explanation_all
    return output


class CandleEngine:
    def __init__(self, config: CandleEngineConfig | None = None):
        self.config = config or CandleEngineConfig()

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = validate_ohlcv(data)
        atr_series = atr(frame, self.config.atr_period).rename("atr")
        g = geometry(frame, atr_series)
        context = market_context(frame, g, atr_series, self.config)
        single = single_candle_patterns(frame, g, context, self.config)
        multi = multi_candle_patterns(frame, g, context, self.config)
        classification = classify_output(
            frame,
            g,
            context,
            single,
            multi,
            self.config,
        )

        output = pd.concat(
            [
                frame,
                atr_series,
                g,
                context,
                single,
                multi,
                classification,
            ],
            axis=1,
        )
        return output.loc[:, ~output.columns.duplicated(keep="first")]

    def analyze_csv(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        data = load_ohlcv_csv(input_path)
        features = self.analyze(data)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(output_path)
        return features

    @staticmethod
    def explain(
        features: pd.DataFrame,
        timestamp: pd.Timestamp | str | None = None,
    ) -> dict[str, Any]:
        if features.empty:
            raise ValueError("features is empty")

        if timestamp is None:
            row = features.iloc[-1]
            resolved_timestamp = features.index[-1]
        else:
            resolved_timestamp = pd.Timestamp(timestamp)
            if resolved_timestamp not in features.index:
                raise KeyError(f"Timestamp not present: {resolved_timestamp}")
            row = features.loc[resolved_timestamp]

        keys = [
            "dominant_candle_type",
            "candle_family",
            "candle_bias",
            "candle_strength_score",
            "classification_confidence",
            "rejection_type",
            "prior_trend_state",
            "volume_regime",
            "volatility_regime",
            "pattern_labels",
            "explanation",
        ]
        return {
            "timestamp": str(resolved_timestamp),
            **{key: row.get(key) for key in keys},
        }


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    timestamp_candidates = [
        column
        for column in data.columns
        if str(column).lower() in {"timestamp", "datetime", "date", "time"}
    ]
    if not timestamp_candidates:
        raise ValueError(
            "CSV requires timestamp, datetime, date, or time column"
        )

    timestamp_column = timestamp_candidates[0]
    data[timestamp_column] = pd.to_datetime(
        data[timestamp_column],
        utc=True,
        errors="raise",
    )
    return validate_ohlcv(data.set_index(timestamp_column))


def summarize_candles(features: pd.DataFrame) -> dict[str, pd.Series]:
    required = {
        "dominant_candle_type",
        "candle_bias",
        "prior_trend_state",
        "volume_regime",
        "volatility_regime",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing candle-engine output columns: {sorted(missing)}")

    return {
        "dominant_types": features["dominant_candle_type"].value_counts(),
        "biases": features["candle_bias"].value_counts(),
        "prior_trends": features["prior_trend_state"].value_counts(),
        "volume_regimes": features["volume_regime"].value_counts(),
        "volatility_regimes": features["volatility_regime"].value_counts(),
    }


def bullish_reversal_signal(features: pd.DataFrame) -> pd.Series:
    """
    Example only. Signal on closed candle; execute on next candle open.
    """
    bullish_reversal_patterns = (
        features["hammer"]
        | features["inverted_hammer"]
        | features["bullish_engulfing"]
        | features["morning_star"]
        | features["piercing_line"]
        | features["tweezer_bottom"]
    )
    return (
        bullish_reversal_patterns
        & (features["prior_trend_state"] == "downtrend")
        & (features["classification_confidence"] >= 70)
        & (features["volume_z"] >= 0)
    ).fillna(False)


def bearish_warning_signal(features: pd.DataFrame) -> pd.Series:
    """
    Warning for existing long positions. This does not imply short selling.
    """
    bearish_patterns = (
        features["shooting_star"]
        | features["hanging_man"]
        | features["bearish_engulfing"]
        | features["evening_star"]
        | features["dark_cloud_cover"]
        | features["tweezer_top"]
    )
    return (
        bearish_patterns
        & (features["prior_trend_state"] == "uptrend")
        & (features["classification_confidence"] >= 70)
    ).fillna(False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify OHLCV candles and candlestick patterns."
    )
    parser.add_argument("csv", help="Input OHLCV CSV")
    parser.add_argument(
        "--output",
        default="candles_features.csv",
        help="Output enriched CSV",
    )
    parser.add_argument(
        "--summary-json",
        default="candles_summary.json",
        help="Output summary JSON",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Print the last N classified candles",
    )
    args = parser.parse_args()

    engine = CandleEngine()
    features = engine.analyze_csv(args.csv, args.output)
    summaries = summarize_candles(features)

    summary_json = {
        key: {str(name): int(value) for name, value in series.items()}
        for key, series in summaries.items()
    }
    Path(args.summary_json).write_text(
        __import__("json").dumps(summary_json, indent=2),
        encoding="utf-8",
    )

    columns = [
        "open",
        "high",
        "low",
        "close",
        "dominant_candle_type",
        "candle_bias",
        "candle_strength_score",
        "classification_confidence",
        "prior_trend_state",
        "pattern_labels",
    ]
    print(features[columns].tail(args.last).to_string())
    print(f"\nSaved features: {args.output}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()


__all__ = [
    "CandleEngine",
    "CandleEngineConfig",
    "SINGLE_PATTERN_COLUMNS",
    "MULTI_PATTERN_COLUMNS",
    "atr",
    "bearish_warning_signal",
    "bullish_reversal_signal",
    "geometry",
    "load_ohlcv_csv",
    "market_context",
    "multi_candle_patterns",
    "single_candle_patterns",
    "summarize_candles",
    "validate_ohlcv",
]
