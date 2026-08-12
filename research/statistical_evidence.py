"""Conservative statistical evidence for frozen portfolio research leads."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.optimization import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from research.portfolio_selection import (
    RotationBacktestResult,
    paired_block_bootstrap_difference,
)
from utils.common import stable_hash
from utils.pandas_time import sunday_week_end_labels


def _clean_returns(values: pd.Series) -> pd.Series:
    selected = values.replace([np.inf, -np.inf], np.nan).dropna()
    selected = selected.astype(float)
    if not selected.index.is_monotonic_increasing:
        selected = selected.sort_index()
    return selected


def deduplicated_equity_returns(equity: pd.Series) -> pd.Series:
    """Combine a terminal duplicate timestamp before calculating returns."""

    selected = equity[
        ~equity.index.duplicated(keep="last")
    ].astype(float)
    return _clean_returns(
        selected.pct_change(fill_method=None)
    )


def hac_effective_sample_size(
    returns: pd.Series,
    *,
    maximum_lag: int | None = None,
) -> dict[str, Any]:
    """Estimate a conservative ESS using only positive serial dependence."""

    selected = _clean_returns(returns)
    count = len(selected)
    if count < 3:
        return {
            "raw_observations": count,
            "maximum_lag": 0,
            "positive_autocorrelation_sum": 0.0,
            "effective_sample_size": count,
            "autocorrelations": {},
            "method": "POSITIVE_AUTOCORRELATION_TRUNCATED_HAC",
        }
    lag = (
        int(maximum_lag)
        if maximum_lag is not None
        else max(
            1,
            int(math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0))),
        )
    )
    lag = min(lag, count - 2)
    autocorrelations: dict[str, float] = {}
    positive_sum = 0.0
    for offset in range(1, lag + 1):
        correlation = float(selected.autocorr(lag=offset))
        if not math.isfinite(correlation):
            correlation = 0.0
        autocorrelations[str(offset)] = correlation
        positive_sum += max(0.0, correlation)
    denominator = 1.0 + 2.0 * positive_sum
    estimate = int(math.floor(count / denominator))
    return {
        "raw_observations": count,
        "maximum_lag": lag,
        "positive_autocorrelation_sum": positive_sum,
        "effective_sample_size": max(3, min(count, estimate)),
        "autocorrelations": autocorrelations,
        "method": "POSITIVE_AUTOCORRELATION_TRUNCATED_HAC",
    }


def _weekly_returns(daily: pd.Series) -> pd.Series:
    selected = 1.0 + _clean_returns(daily)
    return selected.groupby(sunday_week_end_labels(selected.index)).prod().sub(
        1.0
    ).dropna()


def _trial_sharpes(matrix: pd.DataFrame) -> list[float]:
    standard = matrix.std(ddof=1).replace(0.0, np.nan)
    values = (matrix.mean() / standard).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return [
        float(value)
        for value in values.dropna().to_numpy(dtype=float)
    ]


def conservative_dsr_audit(
    candidate_daily_returns: pd.Series,
    trial_daily_returns: pd.DataFrame,
    *,
    total_trials: int,
) -> dict[str, Any]:
    """Report raw, HAC, weekly and weekly-ESS DSR for one frozen DNA."""

    matrix = (
        trial_daily_returns.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
    )
    candidate = _clean_returns(candidate_daily_returns).reindex(
        matrix.index
    ).dropna()
    matrix = matrix.reindex(candidate.index).dropna(how="any")
    candidate = candidate.reindex(matrix.index)
    if len(candidate) < 8 or matrix.shape[1] < 2:
        raise ValueError("DSR audit requires aligned candidate and trial paths")
    daily_hac = hac_effective_sample_size(candidate)
    weekly_candidate = _weekly_returns(candidate)
    weekly_matrix = pd.DataFrame(
        {
            str(column): _weekly_returns(matrix[column])
            for column in matrix.columns
        }
    ).dropna(how="any")
    weekly_candidate = weekly_candidate.reindex(
        weekly_matrix.index
    ).dropna()
    weekly_matrix = weekly_matrix.reindex(
        weekly_candidate.index
    ).dropna(how="any")
    weekly_candidate = weekly_candidate.reindex(weekly_matrix.index)
    weekly_hac = hac_effective_sample_size(weekly_candidate)
    daily_trial_sharpes = _trial_sharpes(matrix)
    weekly_trial_sharpes = _trial_sharpes(weekly_matrix)
    variants = {
        "daily_raw": deflated_sharpe_ratio(
            candidate,
            daily_trial_sharpes,
            effective_sample_size=len(candidate),
            total_trials=total_trials,
        ),
        "daily_hac": deflated_sharpe_ratio(
            candidate,
            daily_trial_sharpes,
            effective_sample_size=int(
                daily_hac["effective_sample_size"]
            ),
            total_trials=total_trials,
        ),
        "weekly_raw": deflated_sharpe_ratio(
            weekly_candidate,
            weekly_trial_sharpes,
            effective_sample_size=len(weekly_candidate),
            total_trials=total_trials,
        ),
        "weekly_ess": deflated_sharpe_ratio(
            weekly_candidate,
            weekly_trial_sharpes,
            effective_sample_size=int(
                weekly_hac["effective_sample_size"]
            ),
            total_trials=total_trials,
        ),
    }
    return {
        "scope": "DEVELOPMENT_SELECTION_PERIOD",
        "total_historical_trials": int(total_trials),
        "family_return_paths": int(matrix.shape[1]),
        "daily": daily_hac,
        "weekly": weekly_hac,
        "probabilities": {
            key: float(value) for key, value in variants.items()
        },
        "formal_probability": float(min(variants.values())),
        "formal_rule": "MINIMUM_OF_ALL_CONSERVATIVELY_VALID_VARIANTS",
        "minimum_required": 0.95,
        "passed": min(variants.values()) >= 0.95,
    }


def unique_return_path_pbo(
    trial_returns: pd.DataFrame,
    *,
    group_count: int = 8,
) -> dict[str, Any]:
    """Report nominal and exact-return-path-deduplicated PBO."""

    matrix = (
        trial_returns.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
    )
    path_groups: dict[str, list[str]] = defaultdict(list)
    for column in matrix.columns:
        path_hash = stable_hash(
            [
                format(float(value), ".15g")
                for value in matrix[column].to_numpy(dtype=float)
            ],
            length=64,
        )
        path_groups[path_hash].append(str(column))
    representatives = [
        sorted(names)[0]
        for _, names in sorted(path_groups.items())
    ]
    unique = matrix.loc[:, representatives]
    nominal_pbo, nominal_logits = probability_of_backtest_overfitting(
        matrix,
        group_count=group_count,
    )
    unique_pbo, unique_logits = probability_of_backtest_overfitting(
        unique,
        group_count=group_count,
    )
    valid = [
        float(value)
        for value in (nominal_pbo, unique_pbo)
        if value is not None
    ]
    worst = max(valid) if valid else None
    return {
        "nominal_dna_count": int(matrix.shape[1]),
        "unique_return_path_count": int(unique.shape[1]),
        "duplicate_return_path_count": int(
            matrix.shape[1] - unique.shape[1]
        ),
        "nominal_pbo": nominal_pbo,
        "unique_return_path_pbo": unique_pbo,
        "formal_worst_valid_pbo": worst,
        "maximum_permitted": 0.10,
        "passed": worst is not None and worst <= 0.10,
        "tie_handling": "MIDRANK_HALF_WEIGHT",
        "nominal_split_count": len(nominal_logits),
        "unique_path_split_count": len(unique_logits),
        "return_path_groups": [
            {
                "return_path_hash": path_hash,
                "strategy_ids": sorted(names),
            }
            for path_hash, names in sorted(path_groups.items())
        ],
    }


def _concentration(
    contributions: Mapping[str, float],
) -> dict[str, Any]:
    positives = {
        str(key): max(0.0, float(value))
        for key, value in contributions.items()
        if math.isfinite(float(value))
    }
    total = sum(positives.values())
    shares = {
        key: value / total
        for key, value in positives.items()
        if total > 0 and value > 0
    }
    hhi = sum(share * share for share in shares.values())
    ordered = sorted(
        shares.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "positive_contributions": positives,
        "positive_contribution_shares": dict(ordered),
        "largest_positive_source": (
            ordered[0][0] if ordered else None
        ),
        "largest_positive_share": (
            float(ordered[0][1]) if ordered else None
        ),
        "hhi": float(hhi),
        "effective_positive_sources": (
            float(1.0 / hhi) if hhi > 0 else 0.0
        ),
    }


def pnl_concentration_audit(
    result: RotationBacktestResult,
) -> dict[str, Any]:
    """Attribute positive PnL concentration by asset, year, regime and trade."""

    daily = deduplicated_equity_returns(result.equity_curve)
    year_contributions = {
        str(year): float(values.sum())
        for year, values in daily.groupby(daily.index.year)
    }
    decisions = result.decisions.copy()
    decisions = decisions[
        decisions["reason"] != "TERMINAL_LIQUIDATION"
    ].copy()
    decisions["executed_at"] = pd.to_datetime(
        decisions["executed_at"],
        utc=True,
    )
    decisions = decisions.sort_values("executed_at")
    labels = pd.Series(
        [
            (
                f"BTC_{'UP' if bool(row.btc_uptrend) else 'DOWN'}"
                f"|VOL_{str(row.volatility_state)}"
                f"|BREADTH_{str(row.breadth_state)}"
            )
            for row in decisions.itertuples()
        ],
        index=pd.DatetimeIndex(decisions["executed_at"]),
        dtype=str,
    )
    labels = labels[~labels.index.duplicated(keep="last")]
    regime_for_return = labels.reindex(
        daily.index,
        method="ffill",
    ).fillna("UNCLASSIFIED")
    regime_contributions = {
        str(regime): float(
            daily[regime_for_return == regime].sum()
        )
        for regime in sorted(regime_for_return.unique())
    }
    asset_contributions = {
        market: float(values["net_pnl_amount"])
        for market, values in result.metrics[
            "asset_pnl_attribution"
        ].items()
    }
    episodes = result.position_episodes
    positive_trades = (
        episodes.loc[
            episodes["weighted_pnl"].astype(float) > 0,
            "weighted_pnl",
        ]
        .astype(float)
        .sort_values(ascending=False)
    )
    total_positive_trade_pnl = float(positive_trades.sum())
    top_five_share = (
        float(positive_trades.head(5).sum())
        / total_positive_trade_pnl
        if total_positive_trade_pnl > 0
        else None
    )
    return {
        "asset": _concentration(asset_contributions),
        "year": _concentration(year_contributions),
        "regime": _concentration(regime_contributions),
        "trades": {
            "positive_trade_count": int(len(positive_trades)),
            "top_five_positive_trade_pnl_share": top_five_share,
            "largest_positive_trade_pnl_share": (
                float(positive_trades.iloc[0])
                / total_positive_trade_pnl
                if len(positive_trades)
                and total_positive_trade_pnl > 0
                else None
            ),
        },
        "interpretation": (
            "Diagnostic only: no concentration threshold is selected "
            "retroactively from these results."
        ),
    }


def exposure_matched_equal_weight_equity(
    result: RotationBacktestResult,
    frames: Mapping[str, pd.DataFrame],
    *,
    one_way_cost: float,
) -> pd.Series:
    """Build a causal benchmark using the strategy's time-varying exposure."""

    benchmark = result.portfolio_policy.allowed_markets[0]
    normalized: dict[str, pd.DataFrame] = {}
    for raw_market, frame in frames.items():
        market = raw_market.upper().replace("/", "-").replace("_", "-")
        selected = frame.loc[:, ["open", "close"]].copy()
        selected.index = pd.to_datetime(selected.index, utc=True)
        normalized[market] = selected.sort_index()
    calendar = normalized[benchmark].index
    opens = pd.DataFrame(
        {
            market: frame["open"].reindex(calendar)
            for market, frame in normalized.items()
        },
        index=calendar,
        dtype=float,
    ).sort_index(axis=1)
    closes = pd.DataFrame(
        {
            market: frame["close"].reindex(calendar)
            for market, frame in normalized.items()
        },
        index=calendar,
        dtype=float,
    ).sort_index(axis=1)
    equity_index = result.equity_curve[
        ~result.equity_curve.index.duplicated(keep="last")
    ].index
    weights = result.executed_weights[
        ~result.executed_weights.index.duplicated(keep="first")
    ].reindex(equity_index).fillna(0.0)
    current = pd.Series(0.0, index=opens.columns, dtype=float)
    equity = 1.0
    rows: list[tuple[pd.Timestamp, float]] = [
        (equity_index[0], equity)
    ]
    history = closes.notna().cumsum()
    for location in range(1, len(equity_index)):
        timestamp = equity_index[location]
        previous = equity_index[location - 1]
        exposure = float(
            weights.loc[timestamp].abs().sum()
        )
        previous_location = int(calendar.searchsorted(previous))
        timestamp_location = int(calendar.searchsorted(timestamp))
        eligible = list(
            history.iloc[previous_location][
                (
                    history.iloc[previous_location]
                    >= result.portfolio_policy.minimum_history_observations
                )
                & opens.iloc[previous_location].notna()
                & opens.iloc[timestamp_location].notna()
            ].index
        )
        target = pd.Series(0.0, index=opens.columns, dtype=float)
        if eligible and exposure > 0:
            target.loc[eligible] = exposure / len(eligible)
        turnover = float((target - current).abs().sum())
        equity -= equity * turnover * one_way_cost
        if location == len(equity_index) - 1:
            asset_returns = (
                closes.loc[timestamp]
                / opens.loc[previous]
                - 1.0
            )
        else:
            asset_returns = (
                opens.loc[timestamp]
                / opens.loc[previous]
                - 1.0
            )
        held = target[target > 1e-12].index
        if not asset_returns.reindex(held).notna().all():
            raise ValueError(
                "exposure-matched benchmark cannot value held assets"
            )
        equity *= 1.0 + float((target * asset_returns).sum())
        current = target
        if location == len(equity_index) - 1:
            equity -= equity * float(current.sum()) * one_way_cost
            current *= 0.0
        rows.append((timestamp, equity))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in rows]),
        name="exposure_matched_equal_weight_equity",
        dtype=float,
    )


def exposure_matched_alpha_audit(
    normal: RotationBacktestResult,
    stressed: RotationBacktestResult,
    frames: Mapping[str, pd.DataFrame],
    *,
    normal_one_way_cost: float,
    stressed_one_way_cost: float,
    periods: Mapping[str, tuple[str, str] | list[str]],
    bootstrap_samples: int = 2_000,
    block_size: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap timing/selection alpha against matched exposure and calendar."""

    normal_benchmark = exposure_matched_equal_weight_equity(
        normal,
        frames,
        one_way_cost=normal_one_way_cost,
    )
    stressed_benchmark = exposure_matched_equal_weight_equity(
        stressed,
        frames,
        one_way_cost=stressed_one_way_cost,
    )
    normal_returns = deduplicated_equity_returns(normal.equity_curve)
    stressed_returns = deduplicated_equity_returns(
        stressed.equity_curve
    )
    normal_control = deduplicated_equity_returns(normal_benchmark)
    stressed_control = deduplicated_equity_returns(
        stressed_benchmark
    )

    def evaluate(
        candidate: pd.Series,
        control: pd.Series,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        if start is not None and end is not None:
            start_at = pd.Timestamp(start)
            end_at = pd.Timestamp(end)
            start_at = (
                start_at.tz_localize("UTC")
                if start_at.tzinfo is None
                else start_at.tz_convert("UTC")
            )
            end_at = (
                end_at.tz_localize("UTC")
                if end_at.tzinfo is None
                else end_at.tz_convert("UTC")
            )
            candidate = candidate.loc[
                (candidate.index >= start_at)
                & (candidate.index <= end_at)
            ]
            control = control.loc[
                (control.index >= start_at)
                & (control.index <= end_at)
            ]
        result = paired_block_bootstrap_difference(
            candidate,
            control,
            samples=bootstrap_samples,
            block_size=block_size,
            seed=seed,
        )
        return {
            **result,
            "ci_lower_positive": float(result["ci_lower_95"]) > 0,
        }

    normal_periods = {
        name: evaluate(
            normal_returns,
            normal_control,
            str(bounds[0]),
            str(bounds[1]),
        )
        for name, bounds in periods.items()
    }
    stressed_confirmation = evaluate(
        stressed_returns,
        stressed_control,
        str(periods["confirmation"][0]),
        str(periods["confirmation"][1]),
    )
    return {
        "benchmark": (
            "POINT_IN_TIME_EQUAL_WEIGHT_WITH_EXACT_TIME_VARYING_EXPOSURE"
        ),
        "normal_full_sample": evaluate(
            normal_returns,
            normal_control,
        ),
        "normal_periods": normal_periods,
        "double_cost_confirmation": stressed_confirmation,
        "formal_required_checks": {
            "validation_ci_lower_positive": normal_periods[
                "validation"
            ]["ci_lower_positive"],
            "confirmation_ci_lower_positive": normal_periods[
                "confirmation"
            ]["ci_lower_positive"],
            "double_cost_confirmation_ci_lower_positive": (
                stressed_confirmation["ci_lower_positive"]
            ),
        },
        "passed": all(
            (
                normal_periods["validation"]["ci_lower_positive"],
                normal_periods["confirmation"]["ci_lower_positive"],
                stressed_confirmation["ci_lower_positive"],
            )
        ),
    }


__all__ = [
    "conservative_dsr_audit",
    "deduplicated_equity_returns",
    "exposure_matched_alpha_audit",
    "exposure_matched_equal_weight_equity",
    "hac_effective_sample_size",
    "pnl_concentration_audit",
    "unique_return_path_pbo",
]
