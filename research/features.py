"""Canonical causal feature pipeline for crypto spot research."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.contracts import IntelligenceRecord
from data.market_data import (
    OHLCV_COLUMNS,
    candle_close_index,
    timeframe_delta,
    validate_ohlcv,
)
from utils.common import atomic_write_json, stable_hash

EPSILON = 1e-12
_PARAMETERIZED_FEATURE_CACHE: dict[str, pd.Series] = {}
_PARAMETERIZED_FEATURE_CACHE_LIMIT = 512


def _smoothing_period(period: int | float | Decimal) -> tuple[float, int]:
    value = float(period)
    if not math.isfinite(value) or value < 1:
        raise ValueError("period must be a finite value of at least one")
    return value, int(math.ceil(value))


def sma(series: pd.Series, period: int) -> pd.Series:
    if not isinstance(period, int):
        raise TypeError("SMA windows are integer-only")
    return series.rolling(period, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    if not isinstance(period, int):
        raise TypeError("WMA windows are integer-only")
    if period < 1:
        raise ValueError("WMA period must be positive")
    weights = np.arange(1, period + 1, dtype=float)
    denominator = float(weights.sum())
    return series.rolling(period, min_periods=period).apply(
        lambda values: float(np.dot(values, weights) / denominator),
        raw=True,
    )


def vwma(price: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    if not isinstance(period, int):
        raise TypeError("VWMA windows are integer-only")
    denominator = volume.rolling(period, min_periods=period).sum()
    return (price * volume).rolling(period, min_periods=period).sum() / denominator.replace(
        0, np.nan
    )


def ema(series: pd.Series, period: int | float | Decimal) -> pd.Series:
    value, warmup = _smoothing_period(period)
    return series.ewm(span=value, adjust=False, min_periods=warmup).mean()


def dema(series: pd.Series, period: int | float | Decimal) -> pd.Series:
    first = ema(series, period)
    return 2.0 * first - ema(first, period)


def tema(series: pd.Series, period: int | float | Decimal) -> pd.Series:
    first = ema(series, period)
    second = ema(first, period)
    third = ema(second, period)
    return 3.0 * first - 3.0 * second + third


def hma(series: pd.Series, period: int) -> pd.Series:
    if not isinstance(period, int):
        raise TypeError("HMA windows are integer-only")
    if period < 2:
        raise ValueError("HMA period must be at least two")
    half = max(1, period // 2)
    root = max(1, int(math.sqrt(period)))
    return wma(2.0 * wma(series, half) - wma(series, period), root)


def kama(
    series: pd.Series,
    period: int = 10,
    *,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    if not all(isinstance(value, int) for value in (period, fast, slow)):
        raise TypeError("KAMA periods are integer-only")
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period, min_periods=period).sum()
    efficiency = change / volatility.replace(0, np.nan)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    smoothing = (efficiency * (fast_alpha - slow_alpha) + slow_alpha).pow(2)
    values = series.to_numpy(dtype=float)
    coefficients = smoothing.to_numpy(dtype=float)
    output = np.full(len(series), np.nan)
    if len(series) > period:
        output[period] = float(np.mean(values[: period + 1]))
        for index in range(period + 1, len(series)):
            if not np.isfinite(coefficients[index]):
                continue
            previous = output[index - 1]
            if not np.isfinite(previous):
                previous = values[index - 1]
            output[index] = previous + coefficients[index] * (values[index] - previous)
    return pd.Series(output, index=series.index)


def wilder(series: pd.Series, period: int | float | Decimal) -> pd.Series:
    value, warmup = _smoothing_period(period)
    return series.ewm(alpha=1.0 / value, adjust=False, min_periods=warmup).mean()


def true_range(data: pd.DataFrame) -> pd.Series:
    previous_close = data["close"].shift(1)
    return pd.concat(
        (
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)


def atr(data: pd.DataFrame, period: int | float | Decimal = 14) -> pd.Series:
    return wilder(true_range(data), period)


def rsi(series: pd.Series, period: int | float | Decimal = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = wilder(gains, period)
    average_loss = wilder(losses, period)
    ratio = average_gain / average_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result = result.mask((average_gain == 0) & (average_loss == 0), 50.0)
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return result


def rolling_zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def parameterized_feature_series(
    data: pd.DataFrame,
    feature_id: str,
    parameters: Mapping[str, Any],
    *,
    market: str | None = None,
    timeframe: str | None = None,
    provider_context_hash: str = "",
    cache_dir: Path | None = None,
) -> pd.Series:
    """Canonical parameterized feature calculation with deterministic caching."""

    frame = validate_ohlcv(
        data,
        timeframe=timeframe or data.attrs.get("timeframe"),
        closed_candles_only=bool(timeframe or data.attrs.get("timeframe")),
    )
    data_hash = sha256(
        pd.util.hash_pandas_object(
            frame.loc[:, list(OHLCV_COLUMNS)],
            index=True,
        ).values.tobytes()
    ).hexdigest()
    canonical = {
        key: (format(value, "f") if isinstance(value, Decimal) else str(value))
        for key, value in sorted(parameters.items())
    }
    key_material = {
        "data_hash": data_hash,
        "market": market or frame.attrs.get("market"),
        "timeframe": timeframe or frame.attrs.get("timeframe"),
        "feature_id": feature_id,
        "feature_version": "1.0.0",
        "parameters": canonical,
        "provider_context_hash": provider_context_hash,
    }
    cache_key = stable_hash(key_material, length=64)
    cached = _PARAMETERIZED_FEATURE_CACHE.get(cache_key)
    if cached is not None and cached.index.equals(frame.index):
        result = cached.copy()
        result.attrs["feature_cache"] = "MEMORY_HIT"
        result.attrs["feature_cache_key"] = cache_key
        return result
    persisted = cache_dir / f"{cache_key}.parquet" if cache_dir else None
    if persisted is not None and persisted.is_file():
        loaded = pd.read_parquet(persisted)
        result = loaded["value"]
        result.index = pd.DatetimeIndex(loaded.index)
        if result.index.equals(frame.index):
            _PARAMETERIZED_FEATURE_CACHE[cache_key] = result.copy()
            result.attrs["feature_cache"] = "PERSISTED_HIT"
            result.attrs["feature_cache_key"] = cache_key
            return result
    if feature_id == "rsi":
        result = rsi(frame["close"], parameters["period"])
    elif feature_id == "ema":
        result = ema(frame["close"], parameters["period"])
    elif feature_id == "atr":
        result = atr(frame, parameters["period"])
    elif feature_id == "bollinger_lower":
        period = int(parameters["period"])
        multiplier = float(parameters["multiplier"])
        center = sma(frame["close"], period)
        deviation = (
            frame["close"]
            .rolling(
                period,
                min_periods=period,
            )
            .std(ddof=0)
        )
        result = center - multiplier * deviation
    else:
        raise KeyError(f"unsupported parameterized feature: {feature_id}")
    result = result.rename(feature_id)
    result.attrs["feature_cache"] = "MISS"
    result.attrs["feature_cache_key"] = cache_key
    if len(_PARAMETERIZED_FEATURE_CACHE) >= _PARAMETERIZED_FEATURE_CACHE_LIMIT:
        _PARAMETERIZED_FEATURE_CACHE.pop(next(iter(_PARAMETERIZED_FEATURE_CACHE)))
    _PARAMETERIZED_FEATURE_CACHE[cache_key] = result.copy()
    if persisted is not None:
        persisted.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=persisted.parent,
            prefix=f".{cache_key}.",
            suffix=".parquet",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            result.to_frame("value").to_parquet(temporary)
            try:
                os.replace(temporary, persisted)
            except PermissionError:
                # Parallel process workers can finish the same deterministic
                # feature simultaneously.  On Windows replacing the file can
                # fail while the winning worker's completed parquet is open.
                if not persisted.is_file():
                    raise
        finally:
            temporary.unlink(missing_ok=True)
        manifest_path = persisted.with_suffix(".manifest.json")
        try:
            atomic_write_json(
                manifest_path,
                key_material | {"cache_key": cache_key, "rows": len(result)},
            )
        except PermissionError:
            if not manifest_path.is_file():
                raise
    return result


def return_features(
    data: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    *,
    maximum_benchmark_age: timedelta = timedelta(days=3),
    target_timeframe: str | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    close = data["close"]
    output["typical_price"] = (data["high"] + data["low"] + close) / 3.0
    output["median_price"] = (data["high"] + data["low"]) / 2.0
    output["weighted_close"] = (data["high"] + data["low"] + 2.0 * close) / 4.0
    output["ohlc_average"] = data[["open", "high", "low", "close"]].mean(axis=1)
    output["absolute_return"] = close.diff()
    output["percentage_return"] = close.pct_change(fill_method=None)
    output["log_return"] = np.log(close).diff()
    output["cumulative_return"] = (1.0 + output["percentage_return"]).cumprod() - 1.0
    for period in (7, 20, 30):
        output[f"rolling_return_{period}"] = close.pct_change(
            period,
            fill_method=None,
        )
    realized = output["log_return"].rolling(20, min_periods=20).std(ddof=0)
    output["volatility_adjusted_return_20"] = output["percentage_return"] / realized.replace(
        0, np.nan
    )
    if benchmark is not None:
        benchmark_close, _ = _causal_benchmark_close(
            data.index,
            benchmark,
            maximum_age=maximum_benchmark_age,
            target_timeframe=target_timeframe,
        )
        benchmark_return = benchmark_close.pct_change(fill_method=None)
        output["benchmark_relative_return"] = output["percentage_return"] - benchmark_return
        covariance = output["percentage_return"].rolling(60).cov(benchmark_return)
        variance = benchmark_return.rolling(60).var(ddof=1)
        output["rolling_beta_60"] = covariance / variance.replace(0, np.nan)
        output["rolling_alpha_60"] = (
            output["percentage_return"].rolling(60).mean()
            - output["rolling_beta_60"] * benchmark_return.rolling(60).mean()
        )
    else:
        output["benchmark_relative_return"] = np.nan
        output["rolling_beta_60"] = np.nan
        output["rolling_alpha_60"] = np.nan
    return output


def trend_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    for period in (20, 50, 200):
        output[f"sma_{period}"] = sma(data["close"], period)
        output[f"ema_{period}"] = ema(data["close"], period)
    output["wma_20"] = wma(data["close"], 20)
    output["vwma_20"] = vwma(data["close"], data["volume"], 20)
    output["dema_20"] = dema(data["close"], 20)
    output["tema_20"] = tema(data["close"], 20)
    output["hma_20"] = hma(data["close"], 20)
    output["kama_10"] = kama(data["close"], 10)
    output["ema_50_slope"] = (
        output["ema_50"].pct_change(5, fill_method=None) / 5.0
    )
    output["distance_ema_20"] = data["close"] / output["ema_20"] - 1.0
    output["distance_ema_50"] = data["close"] / output["ema_50"] - 1.0
    close_change = data["close"].diff().abs()
    output["trend_efficiency_20"] = (
        data["close"].diff(20).abs()
        / close_change.rolling(20, min_periods=20).sum().replace(0, np.nan)
    )

    regression_x = np.arange(50, dtype=float)
    regression_x_centered = regression_x - regression_x.mean()
    regression_denominator = float(np.dot(regression_x_centered, regression_x_centered))

    def regression_slope(values: np.ndarray) -> float:
        centered = values - float(np.mean(values))
        scale = max(abs(float(np.mean(values))), EPSILON)
        return float(np.dot(regression_x_centered, centered) / regression_denominator / scale)

    def regression_r_squared(values: np.ndarray) -> float:
        centered = values - float(np.mean(values))
        total = float(np.dot(centered, centered))
        if total <= EPSILON:
            return 0.0
        slope = float(np.dot(regression_x_centered, centered) / regression_denominator)
        fitted = slope * regression_x_centered
        return float(np.clip(np.dot(fitted, fitted) / total, 0.0, 1.0))

    output["linear_regression_slope_50"] = data["close"].rolling(
        50,
        min_periods=50,
    ).apply(regression_slope, raw=True)
    output["linear_regression_r2_50"] = data["close"].rolling(
        50,
        min_periods=50,
    ).apply(regression_r_squared, raw=True)
    for period in (20, 55):
        output[f"donchian_high_{period}"] = data["high"].rolling(period).max().shift(1)
        output[f"donchian_low_{period}"] = data["low"].rolling(period).min().shift(1)

    period = 14
    upward = data["high"].diff()
    downward = -data["low"].diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    atr14 = atr(data, period)
    plus_di = 100.0 * wilder(plus_dm, period) / atr14.replace(0, np.nan)
    minus_di = 100.0 * wilder(minus_dm, period) / atr14.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    output["plus_di_14"] = plus_di
    output["minus_di_14"] = minus_di
    output["adx_14"] = wilder(dx, period)

    def aroon_position(values: np.ndarray, high: bool) -> float:
        position = int(np.argmax(values) if high else np.argmin(values))
        return 100.0 * position / max(1, len(values) - 1)

    output["aroon_up_25"] = (
        data["high"]
        .rolling(25)
        .apply(
            lambda values: aroon_position(values, True),
            raw=True,
        )
    )
    output["aroon_down_25"] = (
        data["low"]
        .rolling(25)
        .apply(
            lambda values: aroon_position(values, False),
            raw=True,
        )
    )
    total_range = true_range(data).rolling(period).sum()
    price_range = data["high"].rolling(period).max() - data["low"].rolling(period).min()
    output["choppiness_14"] = (
        100.0 * np.log10(total_range / price_range.replace(0, np.nan)) / np.log10(period)
    )
    line, direction = supertrend(data, period=10, multiplier=3.0)
    output["supertrend"] = line
    output["supertrend_direction"] = direction
    output["supertrend_bullish_flip"] = (direction > 0) & (direction.shift(1) < 0)

    tenkan_high = data["high"].rolling(9, min_periods=9).max()
    tenkan_low = data["low"].rolling(9, min_periods=9).min()
    kijun_high = data["high"].rolling(26, min_periods=26).max()
    kijun_low = data["low"].rolling(26, min_periods=26).min()
    cloud_high = data["high"].rolling(52, min_periods=52).max()
    cloud_low = data["low"].rolling(52, min_periods=52).min()
    output["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2.0
    output["ichimoku_kijun"] = (kijun_high + kijun_low) / 2.0
    # These are current, backward-only cloud values. They are deliberately not
    # shifted into the future as charting packages commonly do.
    output["ichimoku_cloud_a_causal"] = (
        output["ichimoku_tenkan"] + output["ichimoku_kijun"]
    ) / 2.0
    output["ichimoku_cloud_b_causal"] = (cloud_high + cloud_low) / 2.0
    cloud_top = output[["ichimoku_cloud_a_causal", "ichimoku_cloud_b_causal"]].max(axis=1)
    output["ichimoku_bullish_reclaim"] = (
        (data["close"] > cloud_top)
        & (data["close"].shift(1) <= cloud_top.shift(1))
        & (output["ichimoku_tenkan"] > output["ichimoku_kijun"])
    )

    true_range_sum = true_range(data).rolling(period, min_periods=period).sum()
    positive_vm = (data["high"] - data["low"].shift(1)).abs()
    negative_vm = (data["low"] - data["high"].shift(1)).abs()
    output["vortex_plus_14"] = (
        positive_vm.rolling(period, min_periods=period).sum()
        / true_range_sum.replace(0, np.nan)
    )
    output["vortex_minus_14"] = (
        negative_vm.rolling(period, min_periods=period).sum()
        / true_range_sum.replace(0, np.nan)
    )
    output["vortex_bullish_cross"] = (
        (output["vortex_plus_14"] > output["vortex_minus_14"])
        & (output["vortex_plus_14"].shift(1) <= output["vortex_minus_14"].shift(1))
    )
    return output


def supertrend(
    data: pd.DataFrame,
    *,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    average_range = atr(data, period)
    midpoint = (data["high"] + data["low"]) / 2.0
    upper = midpoint + multiplier * average_range
    lower = midpoint - multiplier * average_range
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(1, index=data.index, dtype=np.int8)
    line = pd.Series(np.nan, index=data.index, dtype=float)
    for index in range(1, len(data)):
        previous = index - 1
        if np.isnan(average_range.iloc[index]):
            continue
        if (
            np.isnan(final_upper.iloc[previous])
            or upper.iloc[index] < final_upper.iloc[previous]
            or data["close"].iloc[previous] > final_upper.iloc[previous]
        ):
            final_upper.iloc[index] = upper.iloc[index]
        else:
            final_upper.iloc[index] = final_upper.iloc[previous]
        if (
            np.isnan(final_lower.iloc[previous])
            or lower.iloc[index] > final_lower.iloc[previous]
            or data["close"].iloc[previous] < final_lower.iloc[previous]
        ):
            final_lower.iloc[index] = lower.iloc[index]
        else:
            final_lower.iloc[index] = final_lower.iloc[previous]
        if direction.iloc[previous] < 0 and data["close"].iloc[index] > final_upper.iloc[previous]:
            direction.iloc[index] = 1
        elif (
            direction.iloc[previous] > 0 and data["close"].iloc[index] < final_lower.iloc[previous]
        ):
            direction.iloc[index] = -1
        else:
            direction.iloc[index] = direction.iloc[previous]
        line.iloc[index] = (
            final_lower.iloc[index] if direction.iloc[index] > 0 else final_upper.iloc[index]
        )
    return line, direction


def _streak(close: pd.Series) -> pd.Series:
    values = close.to_numpy(dtype=float)
    result = np.zeros(len(values), dtype=float)
    for index in range(1, len(values)):
        if values[index] > values[index - 1]:
            result[index] = max(1.0, result[index - 1] + 1.0)
        elif values[index] < values[index - 1]:
            result[index] = min(-1.0, result[index - 1] - 1.0)
    return pd.Series(result, index=close.index)


def momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    close = data["close"]
    output["rsi_14"] = rsi(close, 14)
    rsi_min = output["rsi_14"].rolling(14).min()
    rsi_max = output["rsi_14"].rolling(14).max()
    output["stoch_rsi_14"] = (
        100.0 * (output["rsi_14"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    )
    fast, slow = ema(close, 12), ema(close, 26)
    output["macd"] = fast - slow
    output["macd_signal"] = ema(output["macd"], 9)
    output["macd_histogram"] = output["macd"] - output["macd_signal"]
    output["ppo"] = 100.0 * (fast - slow) / slow.replace(0, np.nan)
    output["roc_12"] = close.pct_change(12, fill_method=None) * 100.0
    typical = (data["high"] + data["low"] + close) / 3.0
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    output["cci_20"] = (typical - typical_mean) / (0.015 * mean_deviation).replace(0, np.nan)
    highest = data["high"].rolling(14).max()
    lowest = data["low"].rolling(14).min()
    output["williams_r_14"] = -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)
    raw_flow = typical * data["volume"]
    direction = np.sign(typical.diff()).fillna(0)
    positive_flow = raw_flow.where(direction > 0, 0.0).rolling(14).sum()
    negative_flow = raw_flow.where(direction < 0, 0.0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    output["mfi_14"] = 100.0 - 100.0 / (1.0 + money_ratio)
    streak_rsi = rsi(_streak(close), 2)
    change = close.pct_change(fill_method=None)
    percent_rank = change.rolling(100).apply(
        lambda values: 100.0 * np.mean(values[:-1] < values[-1]),
        raw=True,
    )
    output["connors_rsi"] = (rsi(close, 3) + streak_rsi + percent_rank) / 3.0
    output["macd_bullish_cross"] = (output["macd"] > output["macd_signal"]) & (
        output["macd"].shift(1) <= output["macd_signal"].shift(1)
    )
    output["mfi_bullish_reclaim"] = (output["mfi_14"] > 40.0) & (
        output["mfi_14"].shift(1) <= 40.0
    )
    momentum_20 = close.pct_change(20, fill_method=None)
    momentum_60 = close.pct_change(60, fill_method=None)
    output["momentum_acceleration"] = momentum_20 - momentum_20.shift(10)
    output["multi_horizon_momentum_score"] = pd.concat(
        [
            close.pct_change(20, fill_method=None) > 0.0,
            momentum_60 > 0.0,
            close.pct_change(120, fill_method=None) > 0.0,
            close.pct_change(240, fill_method=None) > 0.0,
        ],
        axis=1,
    ).mean(axis=1)
    rolling_volatility = np.log(close).diff().rolling(20, min_periods=20).std(ddof=0)
    output["volatility_adjusted_momentum_20"] = (
        momentum_20 / rolling_volatility.replace(0, np.nan)
    )
    return output


def mean_reversion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Robust, backward-only deviation and reclaim features."""

    output = pd.DataFrame(index=data.index)
    close = data["close"].astype(float)
    median = close.rolling(30, min_periods=30).median()
    mad = close.rolling(30, min_periods=30).apply(
        lambda values: float(np.median(np.abs(values - np.median(values)))),
        raw=True,
    )
    output["mad_zscore_30"] = (
        (close - median) / (1.4826 * mad).replace(0, np.nan)
    )
    output["mad_zscore_reclaim"] = (output["mad_zscore_30"] > -2.0) & (
        output["mad_zscore_30"].shift(1) <= -2.0
    )
    output["keltner_lower_reclaim"] = (
        (close > ema(close, 20) - 2.0 * atr(data, 10))
        & (close.shift(1) <= (ema(close, 20) - 2.0 * atr(data, 10)).shift(1))
    )
    return output


def volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    tr = true_range(data)
    atr14 = atr(data, 14)
    output["true_range"] = tr
    output["atr_14"] = atr14
    output["normalized_atr_14"] = atr14 / data["close"]
    returns = np.log(data["close"]).diff()
    output["rolling_volatility_20"] = returns.rolling(20).std(ddof=0)
    middle = sma(data["close"], 20)
    standard = data["close"].rolling(20).std(ddof=0)
    output["bollinger_middle"] = middle
    output["bollinger_upper"] = middle + 2.0 * standard
    output["bollinger_lower"] = middle - 2.0 * standard
    output["bollinger_width"] = (output["bollinger_upper"] - output["bollinger_lower"]) / middle
    keltner_middle = ema(data["close"], 20)
    output["keltner_middle"] = keltner_middle
    output["keltner_upper"] = keltner_middle + 2.0 * atr(data, 10)
    output["keltner_lower"] = keltner_middle - 2.0 * atr(data, 10)
    squeeze = (output["bollinger_upper"] < output["keltner_upper"]) & (
        output["bollinger_lower"] > output["keltner_lower"]
    )
    output["bollinger_keltner_squeeze_state"] = squeeze
    # The breakout candle itself is not allowed to manufacture its own
    # contraction regime: only preceding, already closed candles count.
    output["prior_squeeze_within_12"] = (
        squeeze.shift(1, fill_value=False).rolling(12, min_periods=1).max().astype(bool)
    )

    log_hl = np.log(data["high"] / data["low"])
    log_co = np.log(data["close"] / data["open"])
    log_ho = np.log(data["high"] / data["open"])
    log_lo = np.log(data["low"] / data["open"])
    output["parkinson_volatility_20"] = np.sqrt(
        log_hl.pow(2).rolling(20).mean() / (4.0 * np.log(2.0))
    )
    garman_klass = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    output["garman_klass_volatility_20"] = np.sqrt(garman_klass.clip(lower=0).rolling(20).mean())
    rogers_satchell = log_ho * np.log(data["high"] / data["close"]) + log_lo * np.log(
        data["low"] / data["close"]
    )
    output["rogers_satchell_volatility_20"] = np.sqrt(
        rogers_satchell.clip(lower=0).rolling(20).mean()
    )
    overnight = np.log(data["open"] / data["close"].shift(1))
    close_open = np.log(data["close"] / data["open"])
    k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
    yz_variance = (
        overnight.rolling(20).var(ddof=1)
        + k * close_open.rolling(20).var(ddof=1)
        + (1.0 - k) * rogers_satchell.rolling(20).mean()
    )
    output["yang_zhang_volatility_20"] = np.sqrt(yz_variance.clip(lower=0))
    output["ewma_volatility"] = returns.ewm(alpha=0.06, adjust=False).std(bias=False)
    output["downside_volatility_20"] = (
        returns.clip(upper=0.0).pow(2).rolling(20, min_periods=20).mean().pow(0.5)
    )
    output["upside_volatility_20"] = (
        returns.clip(lower=0.0).pow(2).rolling(20, min_periods=20).mean().pow(0.5)
    )
    output["volatility_of_volatility_20"] = output[
        "rolling_volatility_20"
    ].rolling(20, min_periods=20).std(ddof=0)
    output["atr_percentile_100"] = output["normalized_atr_14"].rolling(
        100,
        min_periods=50,
    ).rank(pct=True)
    prior_range_median = tr.shift(1).rolling(50, min_periods=20).median()
    output["range_expansion_score"] = tr / prior_range_median.replace(0, np.nan)
    prior_volatility = output["rolling_volatility_20"].shift(1)
    output["jump_score"] = returns.abs() / prior_volatility.replace(0, np.nan)
    output["volatility_expansion_breakout"] = (
        (output["range_expansion_score"] > 1.5)
        & (data["close"] > data["high"].shift(1))
    )
    return output


def optional_garch_forecast(close: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=close.index, name="garch_forecast")
    clean_returns = (100.0 * np.log(close).diff()).dropna()
    if len(clean_returns) < 100:
        return output
    try:
        from arch import arch_model

        model = arch_model(
            clean_returns,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            rescale=False,
        )
        fit = model.fit(disp="off", show_warning=False)
        forecast = fit.forecast(horizon=1, reindex=True).variance.iloc[:, 0]
        output.loc[forecast.index] = np.sqrt(forecast) / 100.0
    except (ImportError, ValueError, ArithmeticError):
        pass
    return output


def volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build causal candle-volume features without inventing trade direction.

    ``volume`` is base-asset candle volume.  When the venue does not provide a
    native quote-volume column, ``estimated_quote_volume`` is the close-price
    conversion of that base volume.  Candle-directional volume is deliberately
    labelled as a proxy; it is not CVD, taker delta, footprint, or order flow.
    """

    output = pd.DataFrame(index=data.index)
    volume = pd.to_numeric(data["volume"], errors="raise").astype(float)
    close = pd.to_numeric(data["close"], errors="raise").astype(float)
    high = pd.to_numeric(data["high"], errors="raise").astype(float)
    low = pd.to_numeric(data["low"], errors="raise").astype(float)
    typical = (high + low + close) / 3.0
    native_quote = data.get("quote_volume")
    if native_quote is None:
        quote_volume = close * volume
        quote_volume_source = "ESTIMATED_CLOSE_TIMES_BASE_VOLUME"
    else:
        quote_volume = pd.to_numeric(native_quote, errors="raise").astype(float)
        quote_volume_source = "NATIVE_VENUE_QUOTE_VOLUME"

    output["base_volume"] = volume
    output["estimated_quote_volume"] = quote_volume
    output["log_base_volume"] = np.log1p(volume.clip(lower=0.0))
    output["log_quote_volume"] = np.log1p(quote_volume.clip(lower=0.0))
    for period in (5, 20, 50):
        output[f"volume_average_{period}"] = volume.rolling(
            period,
            min_periods=period,
        ).mean()
    volume_mean = output["volume_average_20"]
    output["volume_average_20"] = volume_mean
    output["volume_zscore_20"] = rolling_zscore(
        output["log_base_volume"],
        20,
    )
    output["relative_volume_20"] = volume / volume_mean.replace(0, np.nan)
    output["quote_relative_volume_20"] = quote_volume / quote_volume.rolling(
        20,
        min_periods=20,
    ).mean().replace(0, np.nan)
    output["volume_rate_of_change_10"] = volume.pct_change(
        10,
        fill_method=None,
    )
    output["volume_oscillator_5_20"] = (
        output["volume_average_5"] / volume_mean.replace(0, np.nan) - 1.0
    )
    fast_volume = volume.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()
    slow_volume = volume.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()
    output["volume_macd"] = fast_volume - slow_volume
    output["volume_macd_signal"] = output["volume_macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()
    output["volume_macd_histogram"] = (
        output["volume_macd"] - output["volume_macd_signal"]
    )

    direction = np.sign(close.diff()).fillna(0.0)
    directional_volume = direction * volume
    output["up_volume"] = volume.where(direction > 0.0, 0.0)
    output["down_volume"] = volume.where(direction < 0.0, 0.0)
    output["candle_directional_volume_proxy"] = directional_volume
    output["cumulative_directional_volume_proxy"] = (
        directional_volume.cumsum()
    )
    output["obv"] = output["cumulative_directional_volume_proxy"]
    output["obv_change_20"] = output["obv"].diff(20)
    output["obv_breakout_20"] = (
        output["obv"] > output["obv"].shift(1).rolling(20, min_periods=20).max()
    )

    price_return = close.pct_change(fill_method=None).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    output["price_volume_trend"] = (
        volume * price_return.fillna(0.0)
    ).cumsum()
    candle_range = (high - low).replace(0.0, np.nan)
    money_flow_multiplier = (
        (close - low) - (high - close)
    ) / candle_range
    money_flow_volume = money_flow_multiplier * volume
    output["accumulation_distribution_line"] = (
        money_flow_volume.fillna(0.0).cumsum()
    )
    output["chaikin_money_flow_20"] = money_flow_volume.rolling(
        20
    ).sum() / volume.rolling(20).sum().replace(0, np.nan)
    output["chaikin_money_flow_reclaim"] = (
        (output["chaikin_money_flow_20"] > 0.0)
        & (output["chaikin_money_flow_20"].shift(1) <= 0.0)
    )

    negative_volume_return = price_return.where(
        volume < volume.shift(1),
        0.0,
    ).fillna(0.0)
    positive_volume_return = price_return.where(
        volume > volume.shift(1),
        0.0,
    ).fillna(0.0)
    output["negative_volume_index"] = (
        1.0 + negative_volume_return
    ).cumprod() * 1_000.0
    output["positive_volume_index"] = (
        1.0 + positive_volume_return
    ).cumprod() * 1_000.0
    raw_force = close.diff() * volume
    output["force_index_1"] = raw_force
    output["force_index_13"] = raw_force.ewm(
        span=13,
        adjust=False,
        min_periods=13,
    ).mean()
    midpoint_move = ((high + low) / 2.0).diff()
    output["ease_of_movement_1"] = (
        midpoint_move * candle_range / volume.replace(0.0, np.nan)
    )
    output["ease_of_movement_14"] = output[
        "ease_of_movement_1"
    ].rolling(14, min_periods=14).mean()
    output["volume_zone_oscillator_14"] = (
        100.0
        * directional_volume.rolling(14, min_periods=14).sum()
        / volume.rolling(14, min_periods=14).sum().replace(0.0, np.nan)
    )

    trend = np.sign(typical.diff()).replace(0.0, np.nan).ffill().fillna(0.0)
    daily_measurement = high - low
    cumulative_measurement = np.zeros(len(data), dtype=float)
    dm_values = daily_measurement.fillna(0.0).to_numpy(dtype=float)
    trend_values = trend.to_numpy(dtype=float)
    for index in range(1, len(data)):
        cumulative_measurement[index] = (
            cumulative_measurement[index - 1] + dm_values[index]
            if trend_values[index] == trend_values[index - 1]
            else dm_values[index - 1] + dm_values[index]
        )
    measurement = pd.Series(
        cumulative_measurement,
        index=data.index,
        dtype=float,
    ).replace(0.0, np.nan)
    volume_force = (
        volume
        * trend
        * (2.0 * (daily_measurement / measurement) - 1.0).abs()
        * 100.0
    )
    output["klinger_volume_oscillator"] = volume_force.ewm(
        span=34,
        adjust=False,
        min_periods=34,
    ).mean() - volume_force.ewm(
        span=55,
        adjust=False,
        min_periods=55,
    ).mean()
    output["klinger_signal_13"] = output[
        "klinger_volume_oscillator"
    ].ewm(
        span=13,
        adjust=False,
        min_periods=13,
    ).mean()

    output["vwap_20"] = (
        (typical * volume).rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    output["anchored_vwap"] = anchored_vwap(data)
    output["vwap_reclaim"] = (close > output["vwap_20"]) & (
        close.shift(1) <= output["vwap_20"].shift(1)
    )
    output["anchored_vwap_reclaim"] = (close > output["anchored_vwap"]) & (
        close.shift(1) <= output["anchored_vwap"].shift(1)
    )
    output["prior_volume_dryup"] = (
        output["relative_volume_20"].shift(1).rolling(5, min_periods=3).mean() < 0.75
    )
    output["amihud_illiquidity_20"] = (
        price_return.abs()
        / quote_volume.replace(0.0, np.nan)
    ).rolling(20, min_periods=20).mean()
    if isinstance(data.index, pd.DatetimeIndex):
        slot = pd.MultiIndex.from_arrays(
            [
                data.index.dayofweek,
                data.index.hour,
                data.index.minute,
            ]
        )
        slot_baseline = volume.groupby(slot).transform(
            lambda values: values.shift(1).rolling(
                20,
                min_periods=5,
            ).median()
        )
        output["time_of_week_relative_volume_20"] = (
            volume / slot_baseline.replace(0.0, np.nan)
        )
    else:
        output["time_of_week_relative_volume_20"] = np.nan

    output.attrs["quote_volume_source"] = quote_volume_source
    output.attrs["directional_volume_semantics"] = (
        "CANDLE_DIRECTION_PROXY_NOT_TRADE_DELTA_OR_CVD"
    )
    output.attrs["unavailable_without_trade_or_l2_history"] = (
        "true_buy_sell_delta",
        "cvd",
        "footprint",
        "stacked_imbalance",
        "absorption",
        "vpin",
        "volume_profile_poc_vah_val_hvn_lvn",
        "historical_order_book_imbalance",
        "historical_microprice",
    )
    return output


def anchored_vwap(
    data: pd.DataFrame,
    anchor: pd.Timestamp | str | None = None,
) -> pd.Series:
    """Causal VWAP from an explicit or first-observation anchor."""

    if data.empty:
        return pd.Series(dtype=float, index=data.index, name="anchored_vwap")
    start = pd.Timestamp(anchor) if anchor is not None else data.index[0]
    if start.tzinfo is None and data.index.tz is not None:
        start = start.tz_localize(data.index.tz)
    selected = data.index >= start
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    numerator = (typical.where(selected) * data["volume"].where(selected)).cumsum()
    denominator = data["volume"].where(selected).cumsum()
    return (numerator / denominator.replace(0, np.nan)).rename("anchored_vwap")


def candle_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    body = (data["close"] - data["open"]).abs()
    upper = data["high"] - data[["open", "close"]].max(axis=1)
    lower = data[["open", "close"]].min(axis=1) - data["low"]
    bullish = data["close"] > data["open"]
    bearish = data["close"] < data["open"]
    output["candle_body_fraction"] = body / candle_range
    output["upper_wick_fraction"] = upper / candle_range
    output["lower_wick_fraction"] = lower / candle_range
    output["close_location_value"] = (data["close"] - data["low"]) / candle_range
    output["doji"] = output["candle_body_fraction"] <= 0.10
    output["spinning_top"] = (
        output["candle_body_fraction"].between(0.10, 0.30) & (upper >= body) & (lower >= body)
    )
    output["high_wave"] = (output["candle_body_fraction"] <= 0.30) & (
        (upper + lower) >= 0.70 * candle_range
    )
    output["hammer"] = (lower >= 2.0 * body) & (upper <= body) & (body > 0)
    output["inverted_hammer"] = bullish & (upper >= 2.0 * body) & (lower <= body) & (body > 0)
    output["shooting_star"] = (upper >= 2.0 * body) & (lower <= body) & (body > 0)
    output["hanging_man"] = bearish & (lower >= 2.0 * body) & (upper <= body) & (body > 0)
    output["bullish_engulfing"] = (
        bullish
        & bearish.shift(1, fill_value=False)
        & (data["open"] <= data["close"].shift(1))
        & (data["close"] >= data["open"].shift(1))
    )
    output["bearish_engulfing"] = (
        bearish
        & bullish.shift(1, fill_value=False)
        & (data["open"] >= data["close"].shift(1))
        & (data["close"] <= data["open"].shift(1))
    )
    output["bullish_harami"] = (
        bullish
        & bearish.shift(1, fill_value=False)
        & (data["open"] >= data["close"].shift(1))
        & (data["close"] <= data["open"].shift(1))
    )
    output["bearish_harami"] = (
        bearish
        & bullish.shift(1, fill_value=False)
        & (data["open"] <= data["close"].shift(1))
        & (data["close"] >= data["open"].shift(1))
    )
    output["inside_bar"] = (data["high"] < data["high"].shift(1)) & (
        data["low"] > data["low"].shift(1)
    )
    output["outside_bar"] = (data["high"] > data["high"].shift(1)) & (
        data["low"] < data["low"].shift(1)
    )
    output["morning_star_proxy"] = (
        bearish.shift(2, fill_value=False)
        & (body.shift(1) < body.shift(2) * 0.5)
        & bullish
        & (data["close"] > (data["open"].shift(2) + data["close"].shift(2)) / 2.0)
    )
    output["evening_star_proxy"] = (
        bullish.shift(2, fill_value=False)
        & (body.shift(1) < body.shift(2) * 0.5)
        & bearish
        & (data["close"] < (data["open"].shift(2) + data["close"].shift(2)) / 2.0)
    )
    output["three_white_soldiers"] = (
        bullish
        & bullish.shift(1, fill_value=False)
        & bullish.shift(2, fill_value=False)
        & (data["close"] > data["close"].shift(1))
        & (data["close"].shift(1) > data["close"].shift(2))
    )
    output["three_black_crows"] = (
        bearish
        & bearish.shift(1, fill_value=False)
        & bearish.shift(2, fill_value=False)
        & (data["close"] < data["close"].shift(1))
        & (data["close"].shift(1) < data["close"].shift(2))
    )
    output["bullish_pin_bar"] = bullish & (lower >= 2.5 * body) & (upper <= body)
    output["bearish_pin_bar"] = bearish & (upper >= 2.5 * body) & (lower <= body)
    output["bullish_marubozu"] = bullish & (output["candle_body_fraction"] >= 0.90)
    output["bearish_marubozu"] = bearish & (output["candle_body_fraction"] >= 0.90)
    output["rising_three_methods_proxy"] = (
        bullish
        & bullish.shift(4, fill_value=False)
        & (data["close"] > data["close"].shift(4))
        & bearish.shift(1, fill_value=False)
        & bearish.shift(2, fill_value=False)
        & bearish.shift(3, fill_value=False)
    )
    output["falling_three_methods_proxy"] = (
        bearish
        & bearish.shift(4, fill_value=False)
        & (data["close"] < data["close"].shift(4))
        & bullish.shift(1, fill_value=False)
        & bullish.shift(2, fill_value=False)
        & bullish.shift(3, fill_value=False)
    )
    return output


def confirmed_fractals(
    data: pd.DataFrame,
    *,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    if left < 1 or right < 1:
        raise ValueError("fractal left and right lags must be positive")
    output = pd.DataFrame(index=data.index)
    window = left + right + 1
    raw_high = (
        data["high"]
        .rolling(window, center=True)
        .apply(
            lambda values: float(np.argmax(values) == left),
            raw=True,
        )
        .fillna(0)
        .astype(bool)
    )
    raw_low = (
        data["low"]
        .rolling(window, center=True)
        .apply(
            lambda values: float(np.argmin(values) == left),
            raw=True,
        )
        .fillna(0)
        .astype(bool)
    )
    # A pivot at t becomes knowable only at t + right.
    confirmed_high = raw_high.shift(right, fill_value=False)
    confirmed_low = raw_low.shift(right, fill_value=False)
    output["confirmed_fractal_high"] = confirmed_high
    output["confirmed_fractal_low"] = confirmed_low
    output["confirmed_fractal_high_price"] = data["high"].where(raw_high).shift(right)
    output["confirmed_fractal_low_price"] = data["low"].where(raw_low).shift(right)
    timestamps = pd.Series(data.index, index=data.index)
    pivot_timestamp = timestamps.shift(right)
    output["fractal_high_pivot_timestamp"] = pivot_timestamp.where(confirmed_high)
    output["fractal_low_pivot_timestamp"] = pivot_timestamp.where(confirmed_low)
    output["fractal_high_confirmation_timestamp"] = timestamps.where(confirmed_high)
    output["fractal_low_confirmation_timestamp"] = timestamps.where(confirmed_low)
    return output


def fractal_family_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    for window in (3, 5, 7):
        side = (window - 1) // 2
        frame = confirmed_fractals(data, left=side, right=side)
        for column in frame:
            output[f"fractal_{window}_{column}"] = frame[column]
    return output


def market_structure_features(
    data: pd.DataFrame,
    *,
    fractal_left: int = 2,
    fractal_right: int = 2,
) -> pd.DataFrame:
    output = confirmed_fractals(
        data,
        left=fractal_left,
        right=fractal_right,
    )
    last_high = output["confirmed_fractal_high_price"].ffill()
    last_low = output["confirmed_fractal_low_price"].ffill()
    previous_high = last_high.shift(1)
    previous_low = last_low.shift(1)
    confirmed_high = output["confirmed_fractal_high"]
    confirmed_low = output["confirmed_fractal_low"]
    event_high = output["confirmed_fractal_high_price"]
    event_low = output["confirmed_fractal_low_price"]
    output["higher_high"] = confirmed_high & (event_high > previous_high)
    output["lower_high"] = confirmed_high & (event_high < previous_high)
    output["higher_low"] = confirmed_low & (event_low > previous_low)
    output["lower_low"] = confirmed_low & (event_low < previous_low)
    close = data["close"]
    output["bullish_bos"] = (close > previous_high) & (close.shift(1) <= previous_high)
    output["bearish_bos"] = (close < previous_low) & (close.shift(1) >= previous_low)
    state = np.zeros(len(data), dtype=np.int8)
    bullish_choch = np.zeros(len(data), dtype=bool)
    bearish_choch = np.zeros(len(data), dtype=bool)
    for index in range(1, len(data)):
        state[index] = state[index - 1]
        if bool(output["bullish_bos"].iloc[index]):
            bullish_choch[index] = state[index - 1] < 0
            state[index] = 1
        elif bool(output["bearish_bos"].iloc[index]):
            bearish_choch[index] = state[index - 1] > 0
            state[index] = -1
    output["bullish_choch"] = bullish_choch
    output["bearish_choch"] = bearish_choch
    output["swing_trend"] = state
    output["bullish_liquidity_sweep"] = (
        (data["low"] < previous_low)
        & (data["close"] > previous_low)
        & (data["open"] >= previous_low)
    )
    output["bearish_liquidity_sweep"] = (
        (data["high"] > previous_high)
        & (data["close"] < previous_high)
        & (data["open"] <= previous_high)
    )
    average_range = atr(data, 14)
    output["distance_to_last_fractal_high_atr"] = (last_high - close) / average_range.replace(
        0, np.nan
    )
    output["distance_to_last_fractal_low_atr"] = (close - last_low) / average_range.replace(
        0, np.nan
    )
    output["distance_to_last_fractal_high_price"] = last_high - close
    output["distance_to_last_fractal_low_price"] = close - last_low
    output["distance_to_last_fractal_high_percent"] = last_high / close.replace(0, np.nan) - 1.0
    output["distance_to_last_fractal_low_percent"] = close / last_low.replace(0, np.nan) - 1.0

    def bars_since(events: pd.Series) -> pd.Series:
        positions = np.arange(len(events), dtype=float)
        latest = pd.Series(
            np.where(events.to_numpy(dtype=bool), positions, np.nan),
            index=events.index,
        ).ffill()
        return pd.Series(positions, index=events.index) - latest

    output["bars_since_fractal_high"] = bars_since(confirmed_high)
    output["bars_since_fractal_low"] = bars_since(confirmed_low)
    output["fractal_density_50"] = (confirmed_high.astype(int) + confirmed_low.astype(int)).rolling(
        50, min_periods=1
    ).sum() / 50.0
    opposite = last_low.where(confirmed_high, last_high.where(confirmed_low))
    event_level = event_high.where(confirmed_high, event_low.where(confirmed_low))
    output["fractal_amplitude_atr"] = (
        (event_level - opposite).abs() / average_range.replace(0, np.nan)
    ).ffill()
    high_events = output["confirmed_fractal_high_price"].dropna()
    low_events = output["confirmed_fractal_low_price"].dropna()
    output["equal_highs"] = False
    output["equal_lows"] = False
    for events, name in ((high_events, "equal_highs"), (low_events, "equal_lows")):
        previous: float | None = None
        for timestamp, value in events.items():
            if previous is not None:
                tolerance = float(average_range.loc[timestamp]) * 0.10
                output.loc[timestamp, name] = abs(float(value) - previous) <= tolerance
            previous = float(value)
    body = (data["close"] - data["open"]).abs()
    output["bullish_displacement"] = (data["close"] > data["open"]) & (body > 1.25 * average_range)
    output["bearish_displacement"] = (data["close"] < data["open"]) & (body > 1.25 * average_range)
    output["bullish_fvg"] = data["low"] > data["high"].shift(2)
    output["bearish_fvg"] = data["high"] < data["low"].shift(2)
    output["bullish_fvg_lower"] = data["high"].shift(2).where(output["bullish_fvg"])
    output["bullish_fvg_upper"] = data["low"].where(output["bullish_fvg"])
    output["bearish_fvg_lower"] = data["high"].where(output["bearish_fvg"])
    output["bearish_fvg_upper"] = data["low"].shift(2).where(output["bearish_fvg"])
    output["bullish_order_block_proxy"] = (
        output["bullish_bos"]
        & output["bullish_displacement"]
        & (data["close"].shift(1) < data["open"].shift(1))
    )
    output["bearish_order_block_proxy"] = (
        output["bearish_bos"]
        & output["bearish_displacement"]
        & (data["close"].shift(1) > data["open"].shift(1))
    )
    range_high = data["high"].rolling(50).max()
    range_low = data["low"].rolling(50).min()
    output["range_position_50"] = (close - range_low) / (range_high - range_low).replace(0, np.nan)
    output["discount_zone"] = output["range_position_50"] < 0.5
    output["premium_zone"] = output["range_position_50"] > 0.5
    output["fractal_range_position"] = (close - last_low) / (last_high - last_low).replace(
        0, np.nan
    )
    output["fractal_trend_score"] = (
        (
            output["higher_high"].astype(int)
            + output["higher_low"].astype(int)
            - output["lower_high"].astype(int)
            - output["lower_low"].astype(int)
        )
        .rolling(20, min_periods=1)
        .sum()
    )
    output["fractal_high_breakout"] = output["bullish_bos"]
    output["fractal_low_breakdown"] = output["bearish_bos"]
    output["bullish_fractal_sweep"] = output["bullish_liquidity_sweep"]
    output["bearish_fractal_sweep"] = output["bearish_liquidity_sweep"]
    relative_volume = data["volume"] / data["volume"].rolling(20).mean().replace(0, np.nan)
    output["fractal_breakout_volume_confirmation"] = output["fractal_high_breakout"] & (
        relative_volume > 1.0
    )
    if "open_interest" in data:
        oi_change = pd.to_numeric(
            data["open_interest"],
            errors="coerce",
        ).pct_change(fill_method=None)
        output["fractal_breakout_open_interest_confirmation"] = output["fractal_high_breakout"] & (
            oi_change > 0
        )
    else:
        output["fractal_breakout_open_interest_confirmation"] = np.nan
    output["fractal_significance_score"] = (
        output["fractal_amplitude_atr"].clip(lower=0, upper=5) / 5.0
        + relative_volume.clip(lower=0, upper=3).fillna(0) / 3.0
    ) / 2.0
    recent_higher_low = output["higher_low"].shift(1).rolling(
        20,
        min_periods=1,
    ).max().astype(bool)
    output["higher_low_continuation"] = recent_higher_low & output["bullish_bos"]
    output["failed_breakout_reclaim"] = output["bullish_liquidity_sweep"]
    prior_inside_bar = (
        (data["high"] < data["high"].shift(1))
        & (data["low"] > data["low"].shift(1))
    )
    output["inside_bar_breakout"] = (
        data["close"] > data["high"].shift(1)
    ) & prior_inside_bar.shift(1, fill_value=False)

    if isinstance(data.index, pd.DatetimeIndex):
        day_key = data.index.normalize()
        day_high = data["high"].groupby(day_key).max().shift(1)
        output["previous_day_high"] = pd.Series(day_key, index=data.index).map(day_high)
        week_key = day_key - pd.to_timedelta(data.index.dayofweek, unit="D")
        week_high = data["high"].groupby(week_key).max().shift(1)
        output["previous_week_high"] = pd.Series(week_key, index=data.index).map(week_high)
        output["previous_day_high_breakout"] = (
            (data["close"] > output["previous_day_high"])
            & (data["close"].shift(1) <= output["previous_day_high"].shift(1))
        )
        output["previous_week_high_breakout"] = (
            (data["close"] > output["previous_week_high"])
            & (data["close"].shift(1) <= output["previous_week_high"].shift(1))
        )
    else:
        output["previous_day_high"] = np.nan
        output["previous_week_high"] = np.nan
        output["previous_day_high_breakout"] = False
        output["previous_week_high_breakout"] = False
    return output


def multi_timeframe_fractal_alignment(
    base_index: pd.DatetimeIndex,
    higher_timeframes: Mapping[str, pd.DataFrame] | None,
    *,
    base_timeframe: str | None = None,
) -> pd.DataFrame:
    """Align only higher-timeframe candles available by the base-candle close."""

    output = pd.DataFrame(index=base_index)
    states: list[pd.Series] = []
    if base_timeframe:
        base_delta = pd.Timedelta(timeframe_delta(base_timeframe))
    elif len(base_index) > 1:
        base_delta = pd.Timedelta(base_index.to_series().diff().dropna().median())
    else:
        base_delta = pd.Timedelta(0)
    decision_index = base_index + base_delta
    for timeframe, source in sorted((higher_timeframes or {}).items()):
        source.attrs.setdefault("timeframe", timeframe)
        frame = validate_ohlcv(
            source,
            timeframe=timeframe,
            closed_candles_only=True,
        )
        fractals = confirmed_fractals(frame, left=2, right=2)
        state = pd.Series(np.nan, index=frame.index, dtype=float)
        state.loc[fractals["confirmed_fractal_low"]] = 1.0
        state.loc[fractals["confirmed_fractal_high"]] = -1.0
        available_index = candle_close_index(
            frame.index,
            timeframe,
        )
        available_state = pd.Series(
            state.to_numpy(),
            index=available_index,
            dtype=float,
        ).ffill()
        aligned = available_state.reindex(decision_index, method="ffill")
        aligned.index = base_index
        output[f"fractal_alignment_{timeframe}"] = aligned
        source_timestamp = pd.Series(
            frame.index,
            index=available_index,
            dtype="datetime64[ns, UTC]",
        ).reindex(decision_index, method="ffill")
        source_timestamp.index = base_index
        output[f"fractal_source_timestamp_{timeframe}"] = source_timestamp
        states.append(aligned)
    output["multi_timeframe_fractal_alignment"] = (
        pd.concat(states, axis=1).mean(axis=1) if states else np.nan
    )
    return output


def higher_timeframe_regime_features(
    base_index: pd.DatetimeIndex,
    higher_timeframes: Mapping[str, pd.DataFrame] | None,
    *,
    base_timeframe: str | None = None,
) -> pd.DataFrame:
    """Causally expose completed 4h/1d trend and regime state."""

    output = pd.DataFrame(index=base_index)
    if base_timeframe:
        base_delta = pd.Timedelta(timeframe_delta(base_timeframe))
    elif len(base_index) > 1:
        base_delta = pd.Timedelta(base_index.to_series().diff().dropna().median())
    else:
        base_delta = pd.Timedelta(0)
    decision_index = base_index + base_delta
    for timeframe, source in sorted((higher_timeframes or {}).items()):
        frame = validate_ohlcv(
            source,
            timeframe=timeframe,
            closed_candles_only=True,
        )
        trend = trend_features(frame)
        regime = (
            (frame["close"] > trend["ema_200"])
            & (trend["ema_50"] > trend["ema_200"])
            & (trend["ema_50_slope"] > 0)
            & (trend["adx_14"] >= 20)
        )
        short_trend = (
            (frame["close"] > trend["ema_50"])
            & (trend["ema_50_slope"] > 0)
        )
        available_index = candle_close_index(
            frame.index,
            timeframe,
        )
        for suffix, state in (
            ("regime_bullish", regime),
            ("trend_bullish", short_trend),
        ):
            aligned = pd.Series(
                state.to_numpy(dtype=bool),
                index=available_index,
            ).reindex(decision_index, method="ffill")
            aligned.index = base_index
            output[f"htf_{timeframe}_{suffix}"] = (
                aligned.astype("boolean").fillna(False).astype(bool)
            )
        source_timestamp = pd.Series(
            frame.index,
            index=available_index,
            dtype="datetime64[ns, UTC]",
        ).reindex(decision_index, method="ffill")
        source_timestamp.index = base_index
        output[f"htf_{timeframe}_source_timestamp"] = source_timestamp
    return output


def _rolling_estimator(
    series: pd.Series,
    window: int,
    estimator: Any,
    *,
    minimum: int | None = None,
) -> pd.Series:
    required = minimum or window
    return series.rolling(window, min_periods=required).apply(
        lambda values: (
            float(estimator(values))
            if np.isfinite(values).all() and np.isfinite(estimator(values))
            else np.nan
        ),
        raw=True,
    )


def katz_fractal_dimension(values: np.ndarray) -> float:
    distances = np.abs(np.diff(values))
    length = float(distances.sum())
    diameter = float(np.max(np.abs(values - values[0])))
    if length <= 0 or diameter <= 0:
        return np.nan
    n = len(values) - 1
    return math.log10(n) / (math.log10(n) + math.log10(diameter / length))


def petrosian_fractal_dimension(values: np.ndarray) -> float:
    differences = np.diff(values)
    changes = int(np.sum(differences[1:] * differences[:-1] < 0))
    n = len(values)
    denominator = math.log10(n) + math.log10(n / (n + 0.4 * changes))
    return math.log10(n) / denominator if denominator else np.nan


def higuchi_fractal_dimension(values: np.ndarray, maximum_k: int = 8) -> float:
    n = len(values)
    if n < maximum_k * 2:
        return np.nan
    lengths: list[float] = []
    scales: list[float] = []
    for k in range(1, maximum_k + 1):
        segments: list[float] = []
        for start in range(k):
            count = (n - start - 1) // k
            if count < 2:
                continue
            distance = sum(
                abs(values[start + index * k] - values[start + (index - 1) * k])
                for index in range(1, count + 1)
            )
            segments.append(distance * (n - 1) / (count * k * k))
        if segments and np.mean(segments) > 0:
            lengths.append(math.log(float(np.mean(segments))))
            scales.append(math.log(1.0 / k))
    if len(lengths) < 2:
        return np.nan
    return float(np.polyfit(scales, lengths, 1)[0])


def hurst_exponent(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    cumulative = np.cumsum(centered)
    rescaled_range = float(np.ptp(cumulative))
    standard = float(np.std(values, ddof=1))
    if standard <= 0 or rescaled_range <= 0:
        return np.nan
    return math.log(rescaled_range / standard) / math.log(len(values))


def detrended_fluctuation_exponent(values: np.ndarray) -> float:
    """Small-window DFA estimate using only observations in ``values``."""

    centered = values - float(np.mean(values))
    integrated = np.cumsum(centered)
    scales = [size for size in (8, 16, 32, 64) if size <= len(values) // 2]
    fluctuations: list[float] = []
    usable_scales: list[float] = []
    for scale in scales:
        segment_count = len(values) // scale
        if segment_count < 2:
            continue
        residuals: list[float] = []
        x = np.arange(scale, dtype=float)
        for segment in range(segment_count):
            selected = integrated[segment * scale : (segment + 1) * scale]
            fitted = np.polyval(np.polyfit(x, selected, 1), x)
            residuals.append(float(np.mean((selected - fitted) ** 2)))
        fluctuation = math.sqrt(max(float(np.mean(residuals)), 0.0))
        if fluctuation > 0.0:
            usable_scales.append(math.log(float(scale)))
            fluctuations.append(math.log(fluctuation))
    if len(fluctuations) < 2:
        return np.nan
    return float(np.polyfit(usable_scales, fluctuations, 1)[0])


def generalized_hurst_width(values: np.ndarray) -> float:
    """Difference between q=1 and q=2 scaling exponents."""

    lags = np.asarray([1, 2, 4, 8, 16], dtype=int)
    lags = lags[lags < len(values) // 2]
    if len(lags) < 3:
        return np.nan
    exponents: list[float] = []
    for order in (1.0, 2.0):
        moments = np.asarray(
            [
                np.mean(np.abs(values[lag:] - values[:-lag]) ** order)
                for lag in lags
            ],
            dtype=float,
        )
        if np.any(moments <= 0.0):
            return np.nan
        slope = float(np.polyfit(np.log(lags), np.log(moments), 1)[0])
        exponents.append(slope / order)
    return abs(exponents[0] - exponents[1])


def complexity_features(close: pd.Series, *, window: int = 64) -> pd.DataFrame:
    """Fast causal entropy and multi-scale energy features."""

    if window < 16:
        raise ValueError("complexity window must be at least 16")
    output = pd.DataFrame(index=close.index)
    returns = np.log(close).diff()
    positive = (returns > 0.0).astype(float)
    probability = positive.rolling(window, min_periods=window).mean()
    complement = 1.0 - probability
    output["sign_entropy"] = -(
        probability * np.log2(probability.clip(lower=EPSILON))
        + complement * np.log2(complement.clip(lower=EPSILON))
    )

    first = returns.shift(2)
    second = returns.shift(1)
    third = returns
    permutation_code = (
        (first > second).astype(int)
        + 2 * (first > third).astype(int)
        + 4 * (second > third).astype(int)
    )
    permutation_entropy = pd.Series(0.0, index=close.index)
    for code in range(8):
        frequency = (permutation_code == code).astype(float).rolling(
            window,
            min_periods=window,
        ).mean()
        permutation_entropy -= frequency * np.log2(frequency.clip(lower=EPSILON))
    output["permutation_entropy_3"] = permutation_entropy / math.log2(6.0)

    fast_component = returns - returns.rolling(4, min_periods=4).mean()
    medium_component = returns.rolling(4, min_periods=4).mean() - returns.rolling(
        16,
        min_periods=16,
    ).mean()
    slow_component = returns.rolling(16, min_periods=16).mean()
    energies = pd.concat(
        [
            fast_component.pow(2).rolling(window, min_periods=window).mean(),
            medium_component.pow(2).rolling(window, min_periods=window).mean(),
            slow_component.pow(2).rolling(window, min_periods=window).mean(),
        ],
        axis=1,
    )
    energies.columns = [
        "wavelet_energy_fast",
        "wavelet_energy_medium",
        "wavelet_energy_slow",
    ]
    energy_share = energies.div(energies.sum(axis=1).replace(0, np.nan), axis=0)
    output = pd.concat([output, energies], axis=1)
    output["wavelet_entropy"] = -(
        energy_share * np.log2(energy_share.clip(lower=EPSILON))
    ).sum(axis=1) / math.log2(3.0)
    output["wavelet_trend_agreement"] = (
        (slow_component > 0.0)
        & (returns.rolling(4, min_periods=4).mean() > 0.0)
        & (returns.rolling(16, min_periods=16).mean() > 0.0)
    )
    return output


def fractal_dimension_features(
    close: pd.Series,
    *,
    window: int = 100,
    include_advanced_estimators: bool = False,
) -> pd.DataFrame:
    if not isinstance(window, int):
        raise TypeError("fractal-dimension window is integer-only")
    if window < 32:
        raise ValueError("fractal-dimension window must be at least 32")
    output = pd.DataFrame(index=close.index)
    output["katz_fractal_dimension"] = _rolling_estimator(close, window, katz_fractal_dimension)
    output["petrosian_fractal_dimension"] = _rolling_estimator(
        close, window, petrosian_fractal_dimension
    )
    output["higuchi_fractal_dimension"] = _rolling_estimator(
        close, window, higuchi_fractal_dimension
    )
    output["fractal_dimension_index"] = output[
        ["katz_fractal_dimension", "petrosian_fractal_dimension"]
    ].mean(axis=1)
    output["hurst_exponent"] = _rolling_estimator(close, window, hurst_exponent)
    if include_advanced_estimators:
        output["dfa_exponent"] = _rolling_estimator(
            close,
            window,
            detrended_fluctuation_exponent,
        )
        output["generalized_hurst_width"] = _rolling_estimator(
            close,
            window,
            generalized_hurst_width,
        )
    dimension = output["fractal_dimension_index"].clip(lower=1.0, upper=2.0)
    alpha = np.exp(-4.6 * (dimension - 1.0)).clip(lower=0.01, upper=1.0)
    values = close.to_numpy(dtype=float)
    weights = alpha.to_numpy(dtype=float)
    frama_values = np.full(len(close), np.nan)
    for index in range(1, len(close)):
        if not np.isfinite(weights[index]):
            continue
        previous = frama_values[index - 1]
        if not np.isfinite(previous):
            previous = values[index - 1]
        frama_values[index] = weights[index] * values[index] + (1 - weights[index]) * previous
    output["frama"] = frama_values
    output["fractal_estimator_window"] = float(window)
    return output


def fractal_research_labels(
    data: pd.DataFrame,
    *,
    horizon: int = 20,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    """Forward outcomes kept outside the tradable FeaturePipeline."""

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("label horizon must be a positive integer")
    fractals = confirmed_fractals(data, left=left, right=right)
    future_high = data["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    future_low = data["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    labels = pd.DataFrame(index=data.index)
    labels["post_fractal_mfe"] = (future_high / data["close"] - 1.0).where(
        fractals["confirmed_fractal_low"]
    )
    labels["post_fractal_mae"] = (future_low / data["close"] - 1.0).where(
        fractals["confirmed_fractal_low"]
    )
    labels["fractal_efficiency"] = labels["post_fractal_mfe"] / labels[
        "post_fractal_mae"
    ].abs().replace(0, np.nan)
    labels.attrs["research_labels_only"] = True
    labels.attrs["available_after_bars"] = horizon
    return labels


def _causal_benchmark_close(
    index: pd.DatetimeIndex,
    benchmark: pd.DataFrame,
    *,
    maximum_age: timedelta,
    target_timeframe: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    benchmark_timeframe = benchmark.attrs.get("timeframe")
    benchmark_frame = validate_ohlcv(
        benchmark,
        timeframe=str(benchmark_timeframe) if benchmark_timeframe else None,
        closed_candles_only=bool(benchmark_timeframe),
    )
    source_delta = (
        pd.Timedelta(timeframe_delta(str(benchmark_timeframe)))
        if benchmark_timeframe
        else pd.Timedelta(0)
    )
    target_delta = (
        pd.Timedelta(timeframe_delta(target_timeframe)) if target_timeframe else pd.Timedelta(0)
    )
    decision_index = index + target_delta
    available_index = benchmark_frame.index + source_delta
    close = pd.Series(
        benchmark_frame["close"].to_numpy(dtype=float),
        index=available_index,
    ).reindex(decision_index, method="ffill")
    close.index = index
    source_timestamp = pd.Series(
        benchmark_frame.index,
        index=available_index,
        dtype="datetime64[ns, UTC]",
    ).reindex(decision_index, method="ffill")
    source_timestamp.index = index
    age = index.to_series(index=index) - source_timestamp
    stale = age > pd.Timedelta(maximum_age)
    return close.mask(stale), source_timestamp.mask(stale)


def relative_strength_features(
    data: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    maximum_benchmark_age: timedelta = timedelta(days=3),
    target_timeframe: str | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    if benchmark is None:
        output["btc_relative_strength"] = np.nan
        output["btc_relative_momentum_20"] = np.nan
        output["btc_relative_return_zscore_60"] = np.nan
        output["btc_relative_reversal_reclaim"] = False
        output["btc_relative_momentum_persistence_5"] = np.nan
        output["btc_benchmark_source_timestamp"] = pd.NaT
        return output
    benchmark_close, source_timestamp = _causal_benchmark_close(
        data.index,
        benchmark,
        maximum_age=maximum_benchmark_age,
        target_timeframe=target_timeframe,
    )
    ratio = data["close"] / benchmark_close
    output["btc_relative_strength"] = ratio
    output["btc_relative_momentum_20"] = ratio.pct_change(
        20,
        fill_method=None,
    )
    relative_return = np.log(ratio).diff()
    relative_mean = relative_return.rolling(60, min_periods=60).mean()
    relative_standard = relative_return.rolling(60, min_periods=60).std(ddof=0)
    output["btc_relative_return_zscore_60"] = (
        (relative_return - relative_mean)
        / relative_standard.replace(0, np.nan)
    )
    output["btc_relative_reversal_reclaim"] = (
        (output["btc_relative_return_zscore_60"] > -2.0)
        & (output["btc_relative_return_zscore_60"].shift(1) <= -2.0)
    )
    output["btc_relative_momentum_persistence_5"] = (
        output["btc_relative_momentum_20"].rolling(5, min_periods=5).min()
    )
    output["btc_benchmark_source_timestamp"] = source_timestamp
    return output


def intelligence_features(
    index: pd.DatetimeIndex,
    records: Iterable[IntelligenceRecord] | None,
    *,
    market: str | None = None,
    window: str = "24h",
) -> pd.DataFrame:
    columns = (
        "intelligence_event_count",
        "intelligence_relevance_weighted_sentiment",
        "negative_risk_event_score",
        "regulation_event_score",
        "exchange_risk_score",
        "hack_exploit_score",
        "macro_liquidity_score",
        "stablecoin_risk_score",
        "intelligence_source_diversity",
        "intelligence_freshness_hours",
        "event_surprise",
    )
    output = pd.DataFrame(0.0, index=index, columns=columns)
    selected_market = market.upper() if market else None
    selected = [
        record
        for record in (records or [])
        if not record.markets or selected_market is None or selected_market in record.markets
    ]
    if not selected:
        output["intelligence_freshness_hours"] = np.nan
        return output
    event_rows: list[dict[str, Any]] = []
    for record in selected:
        categories = set(record.categories)
        relevance = record.relevance_score
        event_rows.append(
            {
                "timestamp": record.usable_at,
                "event_count": 1.0,
                "relevance": relevance,
                "weighted_sentiment": relevance * record.sentiment_score,
                "negative": relevance * record.impact_score * float(record.sentiment_score < 0),
                "regulation": relevance * float("regulation" in categories),
                "exchange": relevance * float("exchange_risk" in categories),
                "hack": relevance * float("hack_exploit" in categories),
                "macro": relevance * float("macro_liquidity" in categories),
                "stablecoin": relevance * float("stablecoin_risk" in categories),
                "source": record.source,
            }
        )
    events = pd.DataFrame(event_rows).sort_values("timestamp").set_index("timestamp")
    combined_index = index.union(events.index).sort_values()
    numeric = (
        events.drop(columns=["source"])
        .groupby(level=0)
        .sum()
        .reindex(combined_index, fill_value=0.0)
    )
    rolled = numeric.rolling(window, closed="both").sum().reindex(index)
    output["intelligence_event_count"] = rolled["event_count"]
    output["intelligence_relevance_weighted_sentiment"] = (
        rolled["weighted_sentiment"] / rolled["relevance"].replace(0, np.nan)
    ).fillna(0.0)
    output["negative_risk_event_score"] = rolled["negative"]
    output["regulation_event_score"] = rolled["regulation"]
    output["exchange_risk_score"] = rolled["exchange"]
    output["hack_exploit_score"] = rolled["hack"]
    output["macro_liquidity_score"] = rolled["macro"]
    output["stablecoin_risk_score"] = rolled["stablecoin"]
    source_flags = pd.get_dummies(events["source"], dtype=float).groupby(level=0).max()
    source_flags = source_flags.reindex(combined_index, fill_value=0.0)
    output["intelligence_source_diversity"] = (
        source_flags.rolling(window, closed="both").max().sum(axis=1).reindex(index)
    )
    marker = pd.Series(pd.NaT, index=combined_index, dtype="datetime64[ns, UTC]")
    marker.loc[events.index.unique()] = events.index.unique()
    last_seen = marker.ffill().reindex(index)
    output["intelligence_freshness_hours"] = (
        index.to_series(index=index) - last_seen
    ).dt.total_seconds() / 3_600.0
    output["event_surprise"] = 0.0
    return output


@dataclass(frozen=True)
class FeaturePipeline:
    include_optional_garch: bool = False
    include_advanced_fractal_estimators: bool = False
    fractal_left: int = 2
    fractal_right: int = 2
    fractal_dimension_window: int = 100

    def build(
        self,
        data: pd.DataFrame,
        *,
        market: str | None = None,
        benchmark: pd.DataFrame | None = None,
        intelligence: Iterable[IntelligenceRecord] | None = None,
        macro_context: pd.DataFrame | Mapping[str, Any] | None = None,
        higher_timeframes: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        timeframe = data.attrs.get("timeframe")
        frame = validate_ohlcv(
            data,
            timeframe=str(timeframe) if timeframe else None,
            closed_candles_only=True,
        )
        features = [
            frame.loc[:, OHLCV_COLUMNS],
            return_features(
                frame,
                benchmark,
                target_timeframe=str(timeframe) if timeframe else None,
            ),
            trend_features(frame),
            momentum_features(frame),
            mean_reversion_features(frame),
            volatility_features(frame),
            volume_features(frame),
            candle_features(frame),
            fractal_family_features(frame),
            market_structure_features(
                frame,
                fractal_left=self.fractal_left,
                fractal_right=self.fractal_right,
            ),
            fractal_dimension_features(
                frame["close"],
                window=self.fractal_dimension_window,
                include_advanced_estimators=self.include_advanced_fractal_estimators,
            ),
            complexity_features(frame["close"]),
            multi_timeframe_fractal_alignment(
                frame.index,
                higher_timeframes,
                base_timeframe=str(timeframe) if timeframe else None,
            ),
            higher_timeframe_regime_features(
                frame.index,
                higher_timeframes,
                base_timeframe=str(timeframe) if timeframe else None,
            ),
            relative_strength_features(
                frame,
                benchmark,
                target_timeframe=str(timeframe) if timeframe else None,
            ),
            intelligence_features(frame.index, intelligence, market=market),
        ]
        result = pd.concat(features, axis=1)
        if macro_context is not None:
            if isinstance(macro_context, pd.DataFrame):
                if not (
                    macro_context.attrs.get("canonical_macro_context")
                    and macro_context.attrs.get("point_in_time_aligned")
                ):
                    raise ValueError(
                        "prebuilt macro context requires canonical point-in-time provenance"
                    )
                macro_features = macro_context.reindex(result.index)
            else:
                from research.macro_context import MacroContextEngine

                macro_features = MacroContextEngine().build(result.index, **dict(macro_context))
            duplicate = sorted(set(result.columns).intersection(macro_features.columns))
            if duplicate:
                raise ValueError(f"macro feature names collide with market features: {duplicate}")
            result = pd.concat([result, macro_features], axis=1)
        if self.include_optional_garch:
            result["garch_forecast"] = optional_garch_forecast(frame["close"])
        result["bull_regime"] = (
            (result["ema_50"] > result["ema_200"])
            & (result["ema_50_slope"] > 0)
            & (result["adx_14"] >= 20)
        )
        result["bear_regime"] = (
            (result["ema_50"] < result["ema_200"])
            & (result["ema_50_slope"] < 0)
            & (result["adx_14"] >= 20)
        )
        result.attrs.update(frame.attrs)
        result.attrs["intelligence_timing_integrity"] = (
            "PASSED" if intelligence is not None else "NOT_USED"
        )
        result.attrs["benchmark_staleness_integrity"] = (
            "PASSED" if benchmark is not None else "NOT_USED"
        )
        result.attrs["higher_timeframe_integrity"] = "PASSED" if higher_timeframes else "NOT_USED"
        result.attrs["feature_knowability"] = {
            column: {
                "available_at": "candle_close",
                "lookahead_safe": True,
                "repaint": False,
            }
            for column in result.columns
        }
        if macro_context is not None:
            for column in macro_features.columns:
                result.attrs["feature_knowability"][column] = {
                    "available_at": "source_available_at",
                    "lookahead_safe": True,
                    "repaint": False,
                }
        result.attrs["feature_knowability"]["confirmed_fractal_high"]["confirmation_lag_bars"] = (
            self.fractal_right
        )
        result.attrs["feature_knowability"]["confirmed_fractal_low"]["confirmation_lag_bars"] = (
            self.fractal_right
        )
        result.attrs["research_labels_excluded"] = (
            "post_fractal_mfe",
            "post_fractal_mae",
            "fractal_efficiency",
        )
        assert_causal_features(result)
        return result


def assert_causal_features(features: pd.DataFrame) -> None:
    forbidden = [column for column in features if column.startswith("raw_fractal_")]
    if forbidden:
        raise ValueError(f"tradable feature output contains raw fractals: {forbidden}")
    metadata = features.attrs.get("feature_knowability")
    if not isinstance(metadata, dict):
        raise ValueError("feature knowability metadata is missing")
    unsafe = [
        column
        for column, details in metadata.items()
        if not details.get("lookahead_safe") or details.get("repaint")
    ]
    if unsafe:
        raise ValueError(f"unsafe or repainting features detected: {unsafe}")


def feature_registry() -> dict[str, dict[str, Any]]:
    families = {
        "trend": {"source": "OHLCV", "causal": True},
        "momentum": {"source": "OHLCV", "causal": True},
        "volatility": {"source": "OHLCV", "causal": True},
        "volume": {"source": "OHLCV", "causal": True},
        "candles": {"source": "OHLCV", "causal": True},
        "market_structure": {
            "source": "OHLCV",
            "causal": True,
            "fractal_confirmation_required": True,
        },
        "relative_strength": {"source": "BTC benchmark", "causal": True},
        "intelligence": {
            "source": "canonical intelligence",
            "causal": True,
            "uses_usable_at": True,
            "direct_trade_signal": False,
        },
    }
    from research.indicator_registry import indicator_coverage_report

    families["formal_indicator_registry"] = indicator_coverage_report()
    return families


__all__ = [
    "FeaturePipeline",
    "anchored_vwap",
    "assert_causal_features",
    "atr",
    "candle_features",
    "confirmed_fractals",
    "dema",
    "ema",
    "fractal_dimension_features",
    "fractal_family_features",
    "fractal_research_labels",
    "feature_registry",
    "intelligence_features",
    "hma",
    "higuchi_fractal_dimension",
    "hurst_exponent",
    "kama",
    "katz_fractal_dimension",
    "market_structure_features",
    "multi_timeframe_fractal_alignment",
    "momentum_features",
    "optional_garch_forecast",
    "parameterized_feature_series",
    "relative_strength_features",
    "return_features",
    "rolling_zscore",
    "rsi",
    "sma",
    "supertrend",
    "tema",
    "trend_features",
    "true_range",
    "volatility_features",
    "volume_features",
    "vwma",
    "wma",
    "wilder",
]
