"""Crypto-native performance and risk analysis without execution side effects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

CRASH_THRESHOLDS = (0.05, 0.10, 0.20)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _validate_equity(equity: pd.Series, *, name: str = "equity") -> pd.Series:
    if not isinstance(equity, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    if equity.index.tz is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    if len(equity) < 2:
        raise ValueError(f"{name} must contain at least two observations")
    values = pd.to_numeric(equity, errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    if (values <= 0).any():
        raise ValueError(f"{name} must contain positive values")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be sorted and unique")
    return values


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    seconds = index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = float(seconds.median())
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        raise ValueError("timestamps must have a positive observation interval")
    return 365.25 * 24.0 * 3600.0 / median_seconds


def _annualized_volatility(returns: pd.Series, periods: float) -> float | None:
    if len(returns) < 2:
        return None
    return _number(returns.std(ddof=1) * np.sqrt(periods))


def _downside_deviation(returns: pd.Series, periods: float) -> float | None:
    downside = returns.clip(upper=0.0)
    if len(downside) < 2:
        return None
    return _number(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods))


def _longest_recovery_days(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    underwater = equity < running_peak
    longest = pd.Timedelta(0)
    start: pd.Timestamp | None = None
    for timestamp, is_underwater in underwater.items():
        if is_underwater and start is None:
            start = timestamp
        elif not is_underwater and start is not None:
            longest = max(longest, timestamp - start)
            start = None
    if start is not None:
        longest = max(longest, equity.index[-1] - start)
    return float(longest.total_seconds() / 86400.0)


def _capture_ratios(
    returns: pd.Series,
    benchmark_equity: pd.Series | None,
) -> Mapping[str, float | None]:
    if benchmark_equity is None:
        return {"upside_capture": None, "downside_capture": None}
    benchmark = _validate_equity(benchmark_equity, name="benchmark_equity")
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark.pct_change().rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return {"upside_capture": None, "downside_capture": None}

    def capture(mask: pd.Series) -> float | None:
        subset = aligned.loc[mask]
        if subset.empty:
            return None
        denominator = float(subset["benchmark"].mean())
        if denominator == 0:
            return None
        return _number(float(subset["portfolio"].mean()) / denominator)

    return {
        "upside_capture": capture(aligned["benchmark"] > 0),
        "downside_capture": capture(aligned["benchmark"] < 0),
    }


def _horizon_returns(equity: pd.Series, frequency: str) -> pd.Series:
    sampled = equity.resample(frequency).last().dropna()
    return sampled.pct_change().dropna()


def analyze_crypto_performance(
    equity: pd.Series,
    *,
    benchmark_equity: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Return a JSON-safe, crypto-specific performance and risk report.

    The function only analyzes supplied series. It never reads credentials, writes
    artifacts, changes trading authority, or submits orders.
    """

    curve = _validate_equity(equity)
    returns = curve.pct_change().dropna()
    periods = _periods_per_year(curve.index)
    elapsed_years = (curve.index[-1] - curve.index[0]).total_seconds() / (
        365.25 * 86400.0
    )
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    cagr = (float(curve.iloc[-1] / curve.iloc[0]) ** (1.0 / elapsed_years) - 1.0) if elapsed_years > 0 else np.nan
    annual_return = float(returns.mean() * periods)
    annual_volatility = _annualized_volatility(returns, periods)
    downside_deviation = _downside_deviation(returns, periods)
    sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility not in (None, 0.0)
        else None
    )
    sortino = (
        (annual_return - risk_free_rate) / downside_deviation
        if downside_deviation not in (None, 0.0)
        else None
    )

    drawdown = curve / curve.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown))))
    var_95 = float(returns.quantile(0.05))
    tail = returns.loc[returns <= var_95]
    expected_shortfall_95 = float(tail.mean()) if not tail.empty else var_95

    returns_1h = _horizon_returns(curve, "1h")
    returns_4h = _horizon_returns(curve, "4h")
    returns_24h = _horizon_returns(curve, "24h")
    weekend = returns.loc[returns.index.dayofweek >= 5]
    weekday = returns.loc[returns.index.dayofweek < 5]
    weekend_volatility = _annualized_volatility(weekend, periods)
    weekday_volatility = _annualized_volatility(weekday, periods)
    return_std = float(returns.std(ddof=1)) if len(returns) >= 2 else np.nan
    tail_frequency = (
        float((returns.abs() > 3.0 * return_std).mean())
        if np.isfinite(return_std) and return_std > 0
        else 0.0
    )
    capture = _capture_ratios(returns, benchmark_equity)

    return {
        "schema_version": "crypto_performance_risk_v1",
        "analysis_only": True,
        "observation": {
            "start": curve.index[0].isoformat(),
            "end": curve.index[-1].isoformat(),
            "observations": int(len(curve)),
            "periods_per_year": _number(periods),
        },
        "performance": {
            "total_return": _number(total_return),
            "cagr": _number(cagr),
            "annualized_arithmetic_return": _number(annual_return),
            "annualized_volatility": annual_volatility,
            "sharpe_ratio": _number(sharpe),
            "sortino_ratio": _number(sortino),
            "calmar_ratio": _number(cagr / abs(max_drawdown)) if max_drawdown < 0 else None,
            "upside_capture": capture["upside_capture"],
            "downside_capture": capture["downside_capture"],
        },
        "risk": {
            "max_drawdown": _number(max_drawdown),
            "ulcer_index": _number(ulcer_index),
            "longest_recovery_days": _number(_longest_recovery_days(curve)),
            "value_at_risk_95": _number(var_95),
            "expected_shortfall_95": _number(expected_shortfall_95),
            "skewness": _number(returns.skew()),
            "excess_kurtosis": _number(returns.kurt()),
            "tail_event_frequency_3sigma": _number(tail_frequency),
        },
        "crypto_specific": {
            "weekend_volatility": weekend_volatility,
            "weekday_volatility": weekday_volatility,
            "weekend_to_weekday_volatility": (
                _number(weekend_volatility / weekday_volatility)
                if weekend_volatility is not None
                and weekday_volatility not in (None, 0.0)
                else None
            ),
            "max_decline_1h": _number(returns_1h.min()) if not returns_1h.empty else None,
            "max_decline_4h": _number(returns_4h.min()) if not returns_4h.empty else None,
            "max_decline_24h": _number(returns_24h.min()) if not returns_24h.empty else None,
            "crash_frequency_24h": {
                f"at_least_{int(threshold * 100)}pct": _number(
                    (returns_24h <= -threshold).mean()
                )
                if not returns_24h.empty
                else None
                for threshold in CRASH_THRESHOLDS
            },
        },
        "side_effects": {
            "orders_submitted": 0,
            "exchange_mutations": 0,
            "trading_authority_changed": False,
        },
    }
