"""Causal cross-sectional crypto rotation research.

The module deliberately keeps portfolio selection separate from the single-market
SignalBlock lab.  A decision is formed from information available at a daily close
and is executed at the following open.  Costs are charged on every one-way change
in portfolio weight, including the terminal liquidation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.common import stable_hash

WeightingMode = Literal["equal", "inverse_volatility"]
ExposureMappingMode = Literal["frozen", "continuous", "piecewise"]

ROTATION_ENGINE_VERSION = "1.0.0"
ROTATION_POLICY_VERSION = "1.0.0"
PORTFOLIO_METRICS_VERSION = "2.0.0"
CAPITAL_UTILIZATION_METRICS_VERSION = "1.0.0"


@dataclass(frozen=True)
class CapitalUtilizationPolicy:
    """Allocation policy layered on immutable signal DNA."""

    name: str
    base_exposure_budget: float | None
    maximum_total_exposure: float
    maximum_position_exposure: float
    minimum_cash: float
    regime_mapping: ExposureMappingMode = "continuous"
    rebalance_buffer: float = 0.0
    piecewise_score_thresholds: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70)
    piecewise_exposure_levels: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60, 0.80)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("capital utilization policy name cannot be empty")
        if self.base_exposure_budget is not None and not (
            0.0 < self.base_exposure_budget <= 1.0
        ):
            raise ValueError("base exposure budget must be in (0, 1]")
        if not 0.0 < self.maximum_total_exposure <= 1.0:
            raise ValueError("maximum total exposure must be in (0, 1]")
        if not 0.0 < self.maximum_position_exposure <= self.maximum_total_exposure:
            raise ValueError("maximum position exposure violates total exposure")
        if not 0.0 <= self.minimum_cash < 1.0:
            raise ValueError("minimum cash must be in [0, 1)")
        if self.maximum_total_exposure > 1.0 - self.minimum_cash + 1e-12:
            raise ValueError("maximum total exposure violates minimum cash")
        if self.regime_mapping not in {"frozen", "continuous", "piecewise"}:
            raise ValueError(f"unsupported regime mapping: {self.regime_mapping}")
        if not 0.0 <= self.rebalance_buffer < 1.0:
            raise ValueError("rebalance buffer must be in [0, 1)")
        if tuple(sorted(self.piecewise_score_thresholds)) != (
            self.piecewise_score_thresholds
        ):
            raise ValueError("piecewise score thresholds must be sorted")
        if len(self.piecewise_exposure_levels) != (
            len(self.piecewise_score_thresholds) + 1
        ):
            raise ValueError("piecewise levels must have one more value than thresholds")
        if self.regime_mapping == "piecewise" and any(
            level < 0.0 or level > self.maximum_total_exposure + 1e-12
            for level in self.piecewise_exposure_levels
        ):
            raise ValueError("piecewise exposure level violates policy maximum")

    @property
    def policy_hash(self) -> str:
        return stable_hash(
            {
                "policy": "CAPITAL_UTILIZATION_POLICY",
                "version": "1.0.0",
                "values": asdict(self),
            },
            length=64,
        )


def capital_utilization_policy_set() -> tuple[CapitalUtilizationPolicy, ...]:
    """Return the small pre-registered capital-utilization policy family."""

    return (
        CapitalUtilizationPolicy(
            name="FROZEN_CONTROL",
            base_exposure_budget=None,
            maximum_total_exposure=0.40,
            maximum_position_exposure=0.20,
            minimum_cash=0.60,
            regime_mapping="frozen",
        ),
        CapitalUtilizationPolicy(
            name="DEFENSIVE_40",
            base_exposure_budget=0.40,
            maximum_total_exposure=0.40,
            maximum_position_exposure=0.20,
            minimum_cash=0.60,
            regime_mapping="continuous",
        ),
        CapitalUtilizationPolicy(
            name="BALANCED_60",
            base_exposure_budget=0.60,
            maximum_total_exposure=0.60,
            maximum_position_exposure=0.30,
            minimum_cash=0.40,
            regime_mapping="continuous",
        ),
        CapitalUtilizationPolicy(
            name="SEMI_AGGRESSIVE_80",
            base_exposure_budget=0.80,
            maximum_total_exposure=0.80,
            maximum_position_exposure=0.40,
            minimum_cash=0.20,
            regime_mapping="continuous",
        ),
        CapitalUtilizationPolicy(
            name="PIECEWISE_SEMI_AGGRESSIVE_80",
            base_exposure_budget=0.80,
            maximum_total_exposure=0.80,
            maximum_position_exposure=0.40,
            minimum_cash=0.20,
            regime_mapping="piecewise",
            rebalance_buffer=0.05,
        ),
    )


@dataclass(frozen=True)
class RotationPortfolioPolicy:
    """Execution-universe and exposure policy, separate from signal DNA."""

    allowed_markets: tuple[str, ...] = ()
    maximum_total_exposure: float = 1.0
    maximum_position_exposure: float = 1.0
    minimum_cash: float = 0.0
    minimum_history_observations: int = 1

    def __post_init__(self) -> None:
        normalized = tuple(
            market.upper().replace("/", "-").replace("_", "-")
            for market in self.allowed_markets
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed markets must be unique")
        object.__setattr__(self, "allowed_markets", normalized)
        if not 0.0 < self.maximum_total_exposure <= 1.0:
            raise ValueError("maximum total exposure must be in (0, 1]")
        if not 0.0 < self.maximum_position_exposure <= 1.0:
            raise ValueError("maximum position exposure must be in (0, 1]")
        if self.maximum_position_exposure > self.maximum_total_exposure:
            raise ValueError("maximum position exposure exceeds total exposure")
        if not 0.0 <= self.minimum_cash < 1.0:
            raise ValueError("minimum cash must be in [0, 1)")
        if self.maximum_total_exposure > 1.0 - self.minimum_cash + 1e-12:
            raise ValueError("maximum total exposure violates minimum cash")
        if self.minimum_history_observations < 1:
            raise ValueError("minimum history observations must be positive")

    @property
    def policy_hash(self) -> str:
        return stable_hash(
            {
                "policy": "ROTATION_PORTFOLIO_POLICY",
                "version": PORTFOLIO_METRICS_VERSION,
                "values": asdict(self),
            },
            length=64,
        )


@dataclass(frozen=True)
class RotationParameters:
    """Immutable DNA for one cross-sectional momentum hypothesis."""

    momentum_lookback: int = 60
    additional_momentum_lookbacks: tuple[int, ...] = ()
    top_n: int = 2
    rebalance_days: int = 7
    asset_ema_period: int = 100
    btc_ema_period: int = 200
    require_btc_uptrend: bool = True
    continuous_regime: bool = False
    weighting: WeightingMode = "equal"
    volatility_lookback: int = 20
    gross_exposure: float = 0.40
    minimum_cash: float = 0.20
    maximum_positions: int = 2

    def __post_init__(self) -> None:
        if self.momentum_lookback < 2:
            raise ValueError("momentum_lookback must be at least 2")
        if any(lookback < 2 for lookback in self.additional_momentum_lookbacks):
            raise ValueError("additional momentum lookbacks must be at least 2")
        if len(set(self.momentum_lookbacks)) != len(self.momentum_lookbacks):
            raise ValueError("momentum lookbacks must be unique")
        if self.top_n < 1 or self.top_n > self.maximum_positions:
            raise ValueError("top_n must be between 1 and maximum_positions")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be positive")
        if self.asset_ema_period < 2 or self.btc_ema_period < 2:
            raise ValueError("EMA periods must be at least 2")
        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be at least 2")
        if self.weighting not in {"equal", "inverse_volatility"}:
            raise ValueError(f"unsupported weighting: {self.weighting}")
        if not 0.0 < self.gross_exposure <= 1.0:
            raise ValueError("gross_exposure must be in (0, 1]")
        if not 0.0 <= self.minimum_cash < 1.0:
            raise ValueError("minimum_cash must be in [0, 1)")
        if self.gross_exposure > 1.0 - self.minimum_cash + 1e-12:
            raise ValueError("gross_exposure violates minimum_cash")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "CROSS_SECTIONAL_MOMENTUM_ROTATION",
                "engine_version": ROTATION_ENGINE_VERSION,
                "parameters": asdict(self),
            },
            length=64,
        )

    @property
    def momentum_lookbacks(self) -> tuple[int, ...]:
        return (self.momentum_lookback, *self.additional_momentum_lookbacks)


@dataclass(frozen=True)
class RotationBacktestResult:
    parameters: RotationParameters
    portfolio_policy: RotationPortfolioPolicy
    capital_utilization_policy: CapitalUtilizationPolicy | None
    metrics: dict[str, Any]
    integrity: dict[str, Any]
    cost_breakdown: dict[str, float]
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    executed_weights: pd.DataFrame
    decisions: pd.DataFrame
    position_episodes: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        identity_payload = {
            "strategy_dna_hash": self.parameters.dna_hash,
            "portfolio_policy_hash": self.portfolio_policy.policy_hash,
        }
        if self.capital_utilization_policy is not None:
            identity_payload["capital_utilization_policy_hash"] = (
                self.capital_utilization_policy.policy_hash
            )
        summary = {
            "strategy_family": "CROSS_SECTIONAL_MOMENTUM_ROTATION",
            "strategy_dna_hash": self.parameters.dna_hash,
            "result_type": "EXACT_BACKTEST",
            "rotation_engine_version": ROTATION_ENGINE_VERSION,
            "rotation_policy_version": ROTATION_POLICY_VERSION,
            "portfolio_metrics_version": PORTFOLIO_METRICS_VERSION,
            "parameters": asdict(self.parameters),
            "portfolio_policy": asdict(self.portfolio_policy),
            "portfolio_policy_hash": self.portfolio_policy.policy_hash,
            "execution_identity": stable_hash(identity_payload, length=64),
            "metrics": dict(self.metrics),
            "integrity": dict(self.integrity),
            "cost_breakdown": dict(self.cost_breakdown),
        }
        if self.capital_utilization_policy is not None:
            summary["capital_utilization_policy"] = asdict(
                self.capital_utilization_policy
            )
            summary["capital_utilization_policy_hash"] = (
                self.capital_utilization_policy.policy_hash
            )
        return summary


def rotation_parameter_grid(
    *,
    momentum_lookbacks: Sequence[int] = (20, 40, 60, 90, 120, 180),
    top_ns: Sequence[int] = (1, 2),
    rebalance_days: Sequence[int] = (7, 14, 28),
    asset_ema_periods: Sequence[int] = (50, 100, 200),
    btc_filters: Sequence[bool] = (True, False),
    weightings: Sequence[WeightingMode] = ("equal", "inverse_volatility"),
    gross_exposure: float = 0.40,
    minimum_cash: float = 0.20,
    maximum_positions: int = 2,
) -> tuple[RotationParameters, ...]:
    """Create a deterministic joint-parameter screen, not sensitivity rows."""

    rows = (
        RotationParameters(
            momentum_lookback=lookback,
            top_n=top_n,
            rebalance_days=rebalance,
            asset_ema_period=ema_period,
            require_btc_uptrend=btc_filter,
            weighting=weighting,
            gross_exposure=gross_exposure,
            minimum_cash=minimum_cash,
            maximum_positions=maximum_positions,
        )
        for lookback, top_n, rebalance, ema_period, btc_filter, weighting in itertools.product(
            momentum_lookbacks,
            top_ns,
            rebalance_days,
            asset_ema_periods,
            btc_filters,
            weightings,
        )
    )
    unique = {row.dna_hash: row for row in rows}
    return tuple(unique[key] for key in sorted(unique))


def ensemble_rotation_parameter_grid(
    *,
    horizon_sets: Sequence[tuple[int, ...]] = (
        (20, 60),
        (20, 90),
        (20, 60, 120),
        (20, 90, 180),
        (20, 40, 90, 180),
    ),
    top_ns: Sequence[int] = (1, 2),
    rebalance_days: Sequence[int] = (7, 14),
    asset_ema_periods: Sequence[int] = (50, 200),
    continuous_regimes: Sequence[bool] = (False, True),
    weightings: Sequence[WeightingMode] = ("equal", "inverse_volatility"),
    gross_exposure: float = 0.25,
    minimum_cash: float = 0.20,
    maximum_positions: int = 2,
) -> tuple[RotationParameters, ...]:
    """Small economically declared multi-horizon continuation family."""

    rows = (
        RotationParameters(
            momentum_lookback=horizons[0],
            additional_momentum_lookbacks=tuple(horizons[1:]),
            top_n=top_n,
            rebalance_days=rebalance,
            asset_ema_period=ema_period,
            require_btc_uptrend=False,
            continuous_regime=continuous_regime,
            weighting=weighting,
            gross_exposure=gross_exposure,
            minimum_cash=minimum_cash,
            maximum_positions=maximum_positions,
        )
        for horizons, top_n, rebalance, ema_period, continuous_regime, weighting in (
            itertools.product(
                horizon_sets,
                top_ns,
                rebalance_days,
                asset_ema_periods,
                continuous_regimes,
                weightings,
            )
        )
    )
    unique = {row.dna_hash: row for row in rows}
    return tuple(unique[key] for key in sorted(unique))


def _validated_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    benchmark_market: str,
    portfolio_policy: RotationPortfolioPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    normalized_input_markets = {
        market.upper().replace("/", "-").replace("_", "-") for market in frames
    }
    if benchmark not in normalized_input_markets:
        raise ValueError(
            f"{benchmark} is required as the causal portfolio regime benchmark"
        )
    if portfolio_policy.allowed_markets:
        rejected = sorted(
            normalized_input_markets - set(portfolio_policy.allowed_markets)
        )
        if rejected:
            raise ValueError(f"markets outside fail-closed portfolio policy: {rejected}")
    if len(frames) < 2:
        raise ValueError("cross-sectional rotation requires at least two assets")

    normalized: dict[str, pd.DataFrame] = {}
    for raw_market, raw in frames.items():
        market = raw_market.upper().replace("/", "-").replace("_", "-")
        missing = {"open", "close"} - set(raw.columns)
        if missing:
            raise ValueError(f"{market} is missing required columns: {sorted(missing)}")
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise TypeError(f"{market} requires a DatetimeIndex")
        selected = raw.loc[:, ["open", "close"]].copy()
        selected.index = pd.to_datetime(selected.index, utc=True)
        selected = selected[~selected.index.duplicated(keep="last")].sort_index()
        if not selected.index.is_monotonic_increasing:
            raise ValueError(f"{market} timestamps must be monotonic")
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError(f"{market} contains non-finite open/close values")
        if (selected <= 0).any().any():
            raise ValueError(f"{market} contains non-positive open/close values")
        normalized[market] = selected

    # BTC is the canonical calendar. Other assets join the panel only when their
    # own provider history exists; no backfill is permitted before inception.
    common_index = normalized[benchmark].index.sort_values()
    if len(common_index) < 3:
        raise ValueError("fewer than three BTC benchmark observations")
    opens = pd.DataFrame(
        {market: frame["open"].reindex(common_index) for market, frame in normalized.items()},
        index=common_index,
        dtype=float,
    )
    closes = pd.DataFrame(
        {market: frame["close"].reindex(common_index) for market, frame in normalized.items()},
        index=common_index,
        dtype=float,
    )
    return opens.sort_index(axis=1), closes.sort_index(axis=1)


def _effective_sample_size(returns: pd.Series) -> tuple[int, float]:
    selected = returns.dropna().astype(float)
    if selected.empty:
        return 0, 0.0
    lag_one = float(selected.autocorr(lag=1)) if len(selected) > 2 else 0.0
    if not math.isfinite(lag_one) or abs(lag_one) >= 1.0:
        lag_one = 0.0
    estimate = round(len(selected) * (1.0 - lag_one) / max(1e-12, 1.0 + lag_one))
    return int(max(1, min(len(selected), estimate))), lag_one


def _profit_factor(values: Sequence[float] | pd.Series | np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected)]
    gains = float(selected[selected > 0].sum())
    losses = abs(float(selected[selected < 0].sum()))
    if losses > 0:
        return gains / losses
    return math.inf if gains > 0 else 0.0


def portfolio_sample_metrics(
    *,
    portfolio_period_returns: pd.Series,
    position_episodes: pd.DataFrame,
    rebalance_episode_returns: pd.Series,
) -> dict[str, float | int]:
    """Return explicitly named sample and profit-factor units."""

    effective_sample_size, lag_one = _effective_sample_size(
        portfolio_period_returns
    )
    closed_returns = (
        position_episodes["net_return"].astype(float)
        if not position_episodes.empty
        else pd.Series(dtype=float)
    )
    weighted_pnl = (
        position_episodes["weighted_pnl"].astype(float)
        if not position_episodes.empty
        else pd.Series(dtype=float)
    )
    return {
        "raw_portfolio_period_observations": int(len(portfolio_period_returns)),
        "portfolio_period_effective_sample_size": effective_sample_size,
        "portfolio_period_lag_one_autocorrelation": lag_one,
        "portfolio_period_profit_factor": _profit_factor(
            portfolio_period_returns
        ),
        "closed_position_profit_factor": _profit_factor(closed_returns),
        "asset_trade_profit_factor": _profit_factor(weighted_pnl),
        "rebalance_episode_profit_factor": _profit_factor(
            rebalance_episode_returns
        ),
        "closed_position_episodes": int(len(position_episodes)),
        "rebalance_episode_observations": int(len(rebalance_episode_returns)),
    }


def _capped_allocations(
    raw_weights: pd.Series,
    *,
    total_exposure: float,
    maximum_position_exposure: float,
) -> pd.Series:
    """Allocate exposure without exceeding either portfolio or asset caps."""

    raw = raw_weights.astype(float).clip(lower=0.0)
    if raw.empty or raw.sum() <= 0 or total_exposure <= 0:
        return pd.Series(0.0, index=raw.index, dtype=float)
    normalized = raw / raw.sum()
    allocated = pd.Series(0.0, index=raw.index, dtype=float)
    remaining = float(total_exposure)
    active = list(raw.index)
    while active and remaining > 1e-15:
        active_raw = normalized.reindex(active)
        proposal = active_raw / active_raw.sum() * remaining
        available = maximum_position_exposure - allocated.reindex(active)
        increments = proposal.combine(available, min).clip(lower=0.0)
        if float(increments.sum()) <= 1e-15:
            break
        allocated.loc[active] += increments
        remaining = max(0.0, total_exposure - float(allocated.sum()))
        active = [
            market
            for market in active
            if allocated[market] < maximum_position_exposure - 1e-15
        ]
    return allocated


def _target_weights(
    *,
    decision_index: int,
    closes: pd.DataFrame,
    momentum: pd.DataFrame,
    asset_ema: pd.DataFrame,
    btc_ema: pd.Series,
    volatility: pd.DataFrame,
    parameters: RotationParameters,
    portfolio_policy: RotationPortfolioPolicy,
    capital_utilization_policy: CapitalUtilizationPolicy | None,
    benchmark_market: str,
) -> tuple[pd.Series, dict[str, Any]]:
    zero = pd.Series(0.0, index=closes.columns, dtype=float)
    btc_close = float(closes[benchmark_market].iloc[decision_index])
    btc_trend = float(btc_ema.iloc[decision_index])
    risk_on = (
        not parameters.require_btc_uptrend
        or math.isfinite(btc_trend)
        and btc_close > btc_trend
    )
    btc_uptrend = bool(math.isfinite(btc_trend) and btc_close > btc_trend)
    btc_volatility = float(volatility[benchmark_market].iloc[decision_index])
    history_counts = closes.iloc[: decision_index + 1].notna().sum()
    available_history = history_counts.ge(
        portfolio_policy.minimum_history_observations
    )
    trend_eligible = (
        (closes.iloc[decision_index] > asset_ema.iloc[decision_index])
        & available_history
    )
    breadth_population = trend_eligible.where(
        closes.iloc[decision_index].notna()
        & asset_ema.iloc[decision_index].notna()
        & available_history
    ).dropna()
    breadth = float(breadth_population.mean()) if len(breadth_population) else 0.0
    expanding_volatility = (
        volatility[benchmark_market]
        .iloc[: decision_index + 1]
        .dropna()
    )
    volatility_reference = (
        float(expanding_volatility.median())
        if not expanding_volatility.empty
        else math.nan
    )
    normalized_trend = (
        math.log(btc_close / btc_trend)
        / max(1e-12, btc_volatility * math.sqrt(parameters.btc_ema_period))
        if math.isfinite(btc_trend)
        and btc_trend > 0
        and math.isfinite(btc_volatility)
        and btc_volatility > 0
        else 0.0
    )
    btc_trend_score = float(np.clip(0.5 + 0.5 * normalized_trend, 0.10, 1.0))
    volatility_score = (
        1.0
        if not math.isfinite(volatility_reference)
        or btc_volatility <= volatility_reference
        else 0.5
    )
    combined_regime_score = float(
        np.clip(0.5 * btc_trend_score + 0.5 * breadth, 0.0, 1.0)
    )
    regime = {
        "btc_uptrend": btc_uptrend,
        "btc_trend_score": btc_trend_score,
        "btc_volatility": btc_volatility,
        "volatility_score_diagnostic_only": volatility_score,
        "volatility_state": (
            "HIGH"
            if math.isfinite(volatility_reference)
            and btc_volatility > volatility_reference
            else "LOW"
        ),
        "breadth": breadth,
        "breadth_score": breadth,
        "breadth_state": "BROAD" if breadth >= 0.5 else "NARROW",
        "macro_score": None,
        "combined_regime_score": combined_regime_score,
        "regime_score_components_used": ["btc_trend_score", "breadth_score"],
    }
    scores = momentum.iloc[decision_index].replace([np.inf, -np.inf], np.nan)
    exclusion_reasons: dict[str, list[str]] = {}
    for market in closes.columns:
        reasons: list[str] = []
        if int(history_counts[market]) < portfolio_policy.minimum_history_observations:
            reasons.append("INSUFFICIENT_POINT_IN_TIME_HISTORY")
        if not math.isfinite(float(closes[market].iloc[decision_index])):
            reasons.append("MISSING_DECISION_CLOSE")
        if not math.isfinite(float(asset_ema[market].iloc[decision_index])):
            reasons.append("MISSING_ASSET_TREND")
        elif not bool(trend_eligible[market]):
            reasons.append("BELOW_ASSET_EMA")
        if not math.isfinite(float(scores[market])):
            reasons.append("MISSING_MOMENTUM")
        elif float(scores[market]) <= 0.0:
            reasons.append("NON_POSITIVE_MOMENTUM")
        if reasons:
            exclusion_reasons[market] = reasons
    eligible_scores = scores.where(trend_eligible).dropna().sort_values(ascending=False)
    eligible_scores = eligible_scores[eligible_scores > 0.0]
    eligible_assets = list(eligible_scores.index)
    rank_scores = {
        market: float(value)
        for market, value in scores.dropna().sort_values(ascending=False).items()
    }
    ranks = {
        market: rank
        for rank, market in enumerate(rank_scores, start=1)
    }

    base_budget = (
        parameters.gross_exposure
        if capital_utilization_policy is None
        or capital_utilization_policy.base_exposure_budget is None
        else capital_utilization_policy.base_exposure_budget
    )
    mapping = (
        "frozen"
        if capital_utilization_policy is None
        else capital_utilization_policy.regime_mapping
    )
    if mapping == "piecewise":
        policy = capital_utilization_policy
        assert policy is not None
        level_index = int(
            np.searchsorted(
                np.asarray(policy.piecewise_score_thresholds, dtype=float),
                combined_regime_score,
                side="right",
            )
        )
        raw_adjusted_budget = float(policy.piecewise_exposure_levels[level_index])
        exposure_scale = (
            raw_adjusted_budget / base_budget if base_budget > 0 else 0.0
        )
    else:
        exposure_scale = (
            float(np.clip(combined_regime_score, 0.10, 1.0))
            if parameters.continuous_regime
            else 1.0
        )
        raw_adjusted_budget = float(base_budget * exposure_scale)
    adjusted_budget = min(
        raw_adjusted_budget,
        portfolio_policy.maximum_total_exposure,
        1.0 - portfolio_policy.minimum_cash,
    )
    audit_base = {
        "eligible_assets": eligible_assets,
        "excluded_assets": sorted(exclusion_reasons),
        "asset_exclusion_reasons": exclusion_reasons,
        "momentum_scores": rank_scores,
        "momentum_ranks": ranks,
        "raw_policy_budget": float(base_budget),
        "regime_adjusted_budget_before_caps": raw_adjusted_budget,
        "regime_adjusted_budget_after_caps": adjusted_budget,
        "exposure_scale": exposure_scale,
        "regime_mapping": mapping,
        **regime,
    }

    def cash_audit(
        allocated: float,
        *,
        primary_reason: str | None = None,
        selected_count: int = 0,
    ) -> dict[str, Any]:
        reserve = float(portfolio_policy.minimum_cash)
        extra_capacity = max(
            0.0, portfolio_policy.maximum_total_exposure - allocated
        )
        attribution = {"CASH_OPERATIONAL_RESERVE": reserve}
        if primary_reason is not None:
            attribution[primary_reason] = extra_capacity
        else:
            low_regime = max(
                0.0,
                portfolio_policy.maximum_total_exposure - adjusted_budget,
            )
            if low_regime > 1e-12:
                attribution["CASH_LOW_REGIME_SCORE"] = low_regime
            allocation_gap = max(0.0, adjusted_budget - allocated)
            if allocation_gap > 1e-12:
                attribution[
                    "CASH_SINGLE_ASSET_CAP"
                    if selected_count < parameters.top_n
                    else "CASH_TOTAL_EXPOSURE_CAP"
                ] = allocation_gap
        attributed = float(sum(attribution.values()))
        residual = float(1.0 - allocated - attributed)
        if residual > 1e-12:
            attribution["CASH_TOTAL_EXPOSURE_CAP"] = (
                attribution.get("CASH_TOTAL_EXPOSURE_CAP", 0.0) + residual
            )
        return {
            "cash_reason_codes": [
                code for code, value in attribution.items() if value > 1e-12
            ],
            "cash_attribution": attribution,
            "final_cash_fraction": float(1.0 - allocated),
        }

    if not risk_on:
        return zero, {
            "risk_on": False,
            "selected_assets": [],
            "reason": "BTC_REGIME_FILTER",
            "pre_cap_weights": {},
            "weights_after_caps": {},
            **cash_audit(0.0, primary_reason="CASH_HARD_RISK_OFF"),
            **audit_base,
        }

    selected = list(eligible_scores.head(parameters.top_n).index)
    if not selected:
        reason_codes = ["CASH_NO_ELIGIBLE_ASSET"]
        if any(
            "BELOW_ASSET_EMA" in reasons
            for reasons in exclusion_reasons.values()
        ):
            reason_codes.append("CASH_TREND_FILTER")
        cash_details = cash_audit(
            0.0,
            primary_reason="CASH_NO_ELIGIBLE_ASSET",
        )
        cash_details["cash_reason_codes"] = [
            *cash_details["cash_reason_codes"],
            *[code for code in reason_codes if code not in cash_details["cash_reason_codes"]],
        ]
        return zero, {
            "risk_on": True,
            "selected_assets": [],
            "reason": "NO_POSITIVE_TREND_ELIGIBLE_ASSET",
            "pre_cap_weights": {},
            "weights_after_caps": {},
            **cash_details,
            **audit_base,
        }

    target_exposure = adjusted_budget

    if parameters.weighting == "inverse_volatility":
        selected_volatility = volatility.iloc[decision_index].reindex(selected)
        selected_volatility = selected_volatility.where(selected_volatility > 0).dropna()
        selected = list(selected_volatility.index)
        if not selected:
            return zero, {
                "risk_on": True,
                "selected_assets": [],
                "reason": "NO_FINITE_VOLATILITY",
                "pre_cap_weights": {},
                "weights_after_caps": {},
                **cash_audit(0.0, primary_reason="CASH_NO_ELIGIBLE_ASSET"),
                **audit_base,
            }
        raw = 1.0 / selected_volatility
        pre_cap = raw / raw.sum() * target_exposure
        allocations = _capped_allocations(
            raw,
            total_exposure=target_exposure,
            maximum_position_exposure=portfolio_policy.maximum_position_exposure,
        )
    else:
        pre_cap = pd.Series(
            target_exposure / len(selected),
            index=selected,
            dtype=float,
        )
        allocations = _capped_allocations(
            pd.Series(
                1.0,
                index=selected,
                dtype=float,
            ),
            total_exposure=target_exposure,
            maximum_position_exposure=portfolio_policy.maximum_position_exposure,
        )
    target = zero.copy()
    target.loc[selected] = allocations
    allocated = float(target.sum())
    return target, {
        "risk_on": True,
        "selected_assets": selected,
        "reason": "RANKED_MOMENTUM",
        "pre_cap_weights": {
            market: float(value) for market, value in pre_cap.items()
        },
        "weights_after_caps": {
            market: float(value) for market, value in allocations.items()
        },
        "target_total_exposure": allocated,
        "target_cash_fraction": float(1.0 - allocated),
        **cash_audit(allocated, selected_count=len(selected)),
        **audit_base,
    }


def backtest_rotation(
    frames: Mapping[str, pd.DataFrame],
    parameters: RotationParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    benchmark_market: str = "BTC-EUR",
    portfolio_policy: RotationPortfolioPolicy | None = None,
    capital_utilization_policy: CapitalUtilizationPolicy | None = None,
) -> RotationBacktestResult:
    """Run a causal next-open, long-only, cash-enabled portfolio backtest."""

    if min(fee_rate, slippage_bps, spread_bps) < 0:
        raise ValueError("cost assumptions cannot be negative")
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    selected_policy = portfolio_policy or RotationPortfolioPolicy(
        maximum_total_exposure=parameters.gross_exposure,
        maximum_position_exposure=parameters.gross_exposure,
        minimum_cash=parameters.minimum_cash,
    )
    if capital_utilization_policy is not None:
        expected_limits = (
            capital_utilization_policy.maximum_total_exposure,
            capital_utilization_policy.maximum_position_exposure,
            capital_utilization_policy.minimum_cash,
        )
        actual_limits = (
            selected_policy.maximum_total_exposure,
            selected_policy.maximum_position_exposure,
            selected_policy.minimum_cash,
        )
        if any(
            abs(expected - actual) > 1e-12
            for expected, actual in zip(expected_limits, actual_limits, strict=True)
        ):
            raise ValueError(
                "capital utilization policy limits must match portfolio policy"
            )
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=selected_policy,
    )
    warmup = max(
        *parameters.momentum_lookbacks,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        parameters.volatility_lookback,
    )
    if len(closes) <= warmup + 2:
        raise ValueError(
            f"insufficient common history: need more than {warmup + 2}, got {len(closes)}"
        )

    momentum = sum(
        np.log(closes / closes.shift(lookback)) / lookback
        for lookback in parameters.momentum_lookbacks
    ) / len(parameters.momentum_lookbacks)
    asset_ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    btc_ema = closes[benchmark].ewm(
        span=parameters.btc_ema_period,
        adjust=False,
        min_periods=parameters.btc_ema_period,
    ).mean()
    volatility = closes.pct_change(fill_method=None).rolling(
        parameters.volatility_lookback,
        min_periods=parameters.volatility_lookback,
    ).std(ddof=0)

    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    current = pd.Series(0.0, index=closes.columns, dtype=float)
    net_equity = 1.0
    gross_equity = 1.0
    total_turnover = 0.0
    total_cost_amount = 0.0
    rebalance_count = 0
    selection_changes = 0
    scheduled_rebalance_opportunities = 0
    unchanged_holding_decisions = 0
    buy_fills = 0
    sell_fills = 0
    previous_selection: tuple[str, ...] = ()
    equity_rows: list[tuple[pd.Timestamp, float]] = [(closes.index[warmup], net_equity)]
    gross_rows: list[tuple[pd.Timestamp, float]] = [(closes.index[warmup], gross_equity)]
    weight_rows: list[pd.Series] = []
    decision_rows: list[dict[str, Any]] = []
    position_episode_rows: list[dict[str, Any]] = []
    open_episodes: dict[str, dict[str, Any]] = {}
    asset_pnl_amount = {market: 0.0 for market in closes.columns}
    asset_cost_amount = {market: 0.0 for market in closes.columns}

    for execution_index in range(warmup + 1, len(closes)):
        decision_index = execution_index - 1
        scheduled = (decision_index - warmup) % parameters.rebalance_days == 0
        if scheduled:
            scheduled_rebalance_opportunities += 1
        btc_risk_off = (
            parameters.require_btc_uptrend
            and float(closes[benchmark].iloc[decision_index])
            <= float(btc_ema.iloc[decision_index])
        )
        if scheduled or btc_risk_off:
            target, decision = _target_weights(
                decision_index=decision_index,
                closes=closes,
                momentum=momentum,
                asset_ema=asset_ema,
                btc_ema=btc_ema,
                volatility=volatility,
                parameters=parameters,
                portfolio_policy=selected_policy,
                capital_utilization_policy=capital_utilization_policy,
                benchmark_market=benchmark,
            )
            executable = opens.iloc[execution_index].notna()
            target = target.where(executable, 0.0)
            unavailable_selected = [
                market
                for market in decision["selected_assets"]
                if not bool(executable[market])
            ]
            if unavailable_selected:
                decision["cash_reason_codes"] = list(
                    dict.fromkeys(
                        [
                            *decision.get("cash_reason_codes", []),
                            "CASH_LIQUIDITY_FILTER",
                        ]
                    )
                )
            if target.sum() > 0:
                decision["selected_assets"] = list(target[target > 0].index)
            else:
                decision["selected_assets"] = []
                if decision["reason"] == "RANKED_MOMENTUM":
                    decision["reason"] = "SELECTED_ASSETS_NOT_EXECUTABLE"
            prior = current.copy()
            rebalance_buffer = (
                capital_utilization_policy.rebalance_buffer
                if capital_utilization_policy is not None
                else 0.0
            )
            buffered = False
            if (
                rebalance_buffer > 0.0
                and bool(decision.get("risk_on"))
                and float((target - prior).abs().max()) < rebalance_buffer
            ):
                target = prior.copy()
                buffered = True
                decision["cash_reason_codes"] = list(
                    dict.fromkeys(
                        [
                            *decision.get("cash_reason_codes", []),
                            "CASH_REBALANCE_THRESHOLD",
                        ]
                    )
                )
                decision["weights_after_caps"] = {
                    market: float(value)
                    for market, value in target.items()
                    if float(value) > 1e-12
                }
                decision["final_cash_fraction"] = float(1.0 - target.sum())
            changes = target - prior
            turnover = float(changes.abs().sum())
            expected_cost_fraction = turnover * one_way_cost
            if turnover > 1e-15:
                cost_amount = net_equity * turnover * one_way_cost
                net_equity -= cost_amount
                total_cost_amount += cost_amount
                total_turnover += turnover
                rebalance_count += 1
                for market, absolute_change in changes.abs().items():
                    if absolute_change > 1e-15:
                        asset_cost_amount[market] += (
                            cost_amount * float(absolute_change) / turnover
                        )
            execution_prices = opens.iloc[execution_index]
            buy_fills += int((changes > 1e-12).sum())
            sell_fills += int((changes < -1e-12).sum())
            for market in closes.columns:
                prior_weight = float(prior[market])
                target_weight = float(target[market])
                if prior_weight <= 1e-12 and target_weight > 1e-12:
                    open_episodes[market] = {
                        "market": market,
                        "opened_at": opens.index[execution_index],
                        "entry_price": float(execution_prices[market]),
                        "entry_weight": target_weight,
                    }
                elif prior_weight > 1e-12 and target_weight <= 1e-12:
                    episode = open_episodes.pop(market, None)
                    if episode is not None:
                        gross_episode_return = (
                            float(execution_prices[market]) / episode["entry_price"] - 1.0
                        )
                        net_episode_return = gross_episode_return - 2.0 * one_way_cost
                        position_episode_rows.append(
                            episode
                            | {
                                "closed_at": opens.index[execution_index],
                                "exit_price": float(execution_prices[market]),
                                "gross_return": gross_episode_return,
                                "net_return": net_episode_return,
                                "weighted_pnl": (
                                    episode["entry_weight"] * net_episode_return
                                ),
                                "close_reason": decision["reason"],
                            }
                        )
            selection = tuple(decision["selected_assets"])
            if selection != previous_selection:
                selection_changes += 1
            elif scheduled:
                unchanged_holding_decisions += 1
            previous_selection = selection
            current = target
            decision_rows.append(
                {
                    "decision_at": closes.index[decision_index],
                    "executed_at": opens.index[execution_index],
                    "scheduled": scheduled,
                    "turnover": turnover,
                    "expected_cost_fraction": expected_cost_fraction,
                    "rebalance_buffer_applied": buffered,
                    "buy_fill_count": int((changes > 1e-12).sum()),
                    "sell_fill_count": int((changes < -1e-12).sum()),
                    "weight_changes": {
                        market: float(value)
                        for market, value in changes.items()
                        if abs(float(value)) > 1e-12
                    },
                    "target_weights": {
                        market: float(value)
                        for market, value in target.items()
                        if float(value) > 1e-12
                    },
                    "cash_fraction": float(1.0 - target.sum()),
                    "equity_after_cost": net_equity,
                    **decision,
                }
            )

        is_terminal = execution_index == len(closes) - 1
        if is_terminal:
            asset_returns = (
                closes.iloc[execution_index] / opens.iloc[execution_index] - 1.0
            )
        else:
            asset_returns = (
                opens.iloc[execution_index + 1] / opens.iloc[execution_index] - 1.0
            )
        held = current[current.abs() > 1e-12].index
        if not asset_returns.reindex(held).notna().all():
            missing_valuation = list(asset_returns.reindex(held).index[
                asset_returns.reindex(held).isna()
            ])
            raise ValueError(
                "held assets lack a causal next valuation: "
                f"{missing_valuation} at {closes.index[execution_index]}"
            )
        portfolio_return = float((current * asset_returns).sum())
        equity_before_return = net_equity
        for market, weight in current.items():
            if abs(float(weight)) > 1e-12:
                asset_pnl_amount[market] += (
                    equity_before_return
                    * float(weight)
                    * float(asset_returns[market])
                )
        net_equity *= 1.0 + portfolio_return
        gross_equity *= 1.0 + portfolio_return

        if is_terminal:
            terminal_turnover = float(current.abs().sum())
            terminal_cost = net_equity * terminal_turnover * one_way_cost
            net_equity -= terminal_cost
            total_cost_amount += terminal_cost
            total_turnover += terminal_turnover
            if terminal_turnover > 1e-15:
                for market, weight in current.abs().items():
                    if weight > 1e-15:
                        asset_cost_amount[market] += (
                            terminal_cost * float(weight) / terminal_turnover
                        )
            sell_fills += int((current > 1e-12).sum())
            for market in list(open_episodes):
                episode = open_episodes.pop(market)
                exit_price = float(closes[market].iloc[execution_index])
                gross_episode_return = exit_price / episode["entry_price"] - 1.0
                net_episode_return = gross_episode_return - 2.0 * one_way_cost
                position_episode_rows.append(
                    episode
                    | {
                        "closed_at": closes.index[execution_index],
                        "exit_price": exit_price,
                        "gross_return": gross_episode_return,
                        "net_return": net_episode_return,
                        "weighted_pnl": episode["entry_weight"] * net_episode_return,
                        "close_reason": "TERMINAL_LIQUIDATION",
                    }
                )
            terminal_changes = {
                market: -float(value)
                for market, value in current.items()
                if float(value) > 1e-12
            }
            current = current * 0.0
            decision_rows.append(
                {
                    "decision_at": closes.index[execution_index],
                    "executed_at": closes.index[execution_index],
                    "scheduled": False,
                    "risk_on": False,
                    "selected_assets": [],
                    "reason": "TERMINAL_LIQUIDATION",
                    "turnover": terminal_turnover,
                    "buy_fill_count": 0,
                    "sell_fill_count": len(terminal_changes),
                    "weight_changes": terminal_changes,
                    "target_weights": {},
                    "cash_fraction": 1.0,
                    "equity_after_cost": net_equity,
                }
            )

        timestamp = closes.index[execution_index] if is_terminal else opens.index[execution_index + 1]
        equity_rows.append((timestamp, net_equity))
        gross_rows.append((timestamp, gross_equity))
        weight_row = current.copy()
        weight_row.name = timestamp
        weight_rows.append(weight_row)

    equity = pd.Series(
        [value for _, value in equity_rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in equity_rows]),
        name="net_equity",
        dtype=float,
    )
    gross = pd.Series(
        [value for _, value in gross_rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in gross_rows]),
        name="gross_equity",
        dtype=float,
    )
    executed_weights = (
        pd.DataFrame(weight_rows).fillna(0.0)
        if weight_rows
        else pd.DataFrame(columns=closes.columns, dtype=float)
    )
    decisions = pd.DataFrame(decision_rows)
    position_episodes = pd.DataFrame(position_episode_rows)
    net_returns = equity.pct_change(fill_method=None).dropna()
    elapsed_days = max(1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0)
    years = elapsed_days / 365.25
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    annualized_volatility = float(net_returns.std(ddof=0) * math.sqrt(365.25))
    sharpe = (
        float(net_returns.mean() / net_returns.std(ddof=0) * math.sqrt(365.25))
        if len(net_returns) > 1 and net_returns.std(ddof=0) > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    yearly = equity.resample("YE").last().pct_change(fill_method=None).dropna()
    average_exposure = (
        float(executed_weights.abs().sum(axis=1).mean())
        if not executed_weights.empty
        else 0.0
    )
    exposure = (
        executed_weights.abs().sum(axis=1)
        if not executed_weights.empty
        else pd.Series(dtype=float)
    )
    positions_held = (
        (executed_weights.abs() > 1e-12).sum(axis=1)
        if not executed_weights.empty
        else pd.Series(dtype=float)
    )
    downside = net_returns[net_returns < 0]
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside))) if len(downside) else 0.0
    )
    sortino = (
        float(net_returns.mean() / downside_deviation * math.sqrt(365.25))
        if downside_deviation > 0
        else math.inf if net_returns.mean() > 0 else 0.0
    )
    losses = abs(float(net_returns[net_returns < 0].sum()))
    omega = (
        float(net_returns[net_returns > 0].sum()) / losses
        if losses > 0
        else math.inf if bool((net_returns > 0).any()) else 0.0
    )
    underwater = drawdown < -1e-12
    underwater_groups = (underwater != underwater.shift(fill_value=False)).cumsum()
    underwater_durations = (
        underwater.groupby(underwater_groups).sum()[lambda row: row > 0]
    )
    weekly_for_risk = equity.resample("W-SUN").last().pct_change(fill_method=None).dropna()
    monthly_for_risk = equity.resample("ME").last().pct_change(fill_method=None).dropna()
    tail_cutoff = float(net_returns.quantile(0.05)) if len(net_returns) else 0.0
    tail = net_returns[net_returns <= tail_cutoff]
    cash_attribution_average: dict[str, float] = {}
    if not decisions.empty and "cash_attribution" in decisions:
        cash_rows = decisions[
            decisions["reason"] != "TERMINAL_LIQUIDATION"
        ]["cash_attribution"].dropna()
        for row in cash_rows:
            for code, value in dict(row).items():
                cash_attribution_average[code] = (
                    cash_attribution_average.get(code, 0.0) + float(value)
                )
        if len(cash_rows):
            cash_attribution_average = {
                code: value / len(cash_rows)
                for code, value in sorted(cash_attribution_average.items())
            }
    exposure_buckets = {
        "cash_100pct": float((exposure <= 1e-12).mean()) if len(exposure) else 0.0,
        "exposure_0_to_20pct": float(
            ((exposure > 1e-12) & (exposure <= 0.20 + 1e-12)).mean()
        ) if len(exposure) else 0.0,
        "exposure_20_to_40pct": float(
            ((exposure > 0.20 + 1e-12) & (exposure <= 0.40 + 1e-12)).mean()
        ) if len(exposure) else 0.0,
        "exposure_40_to_60pct": float(
            ((exposure > 0.40 + 1e-12) & (exposure <= 0.60 + 1e-12)).mean()
        ) if len(exposure) else 0.0,
        "exposure_60_to_80pct": float(
            ((exposure > 0.60 + 1e-12) & (exposure <= 0.80 + 1e-12)).mean()
        ) if len(exposure) else 0.0,
        "exposure_above_80pct": float((exposure > 0.80 + 1e-12).mean())
        if len(exposure)
        else 0.0,
    }
    changed_decisions = decisions[
        (decisions["turnover"].astype(float) > 1e-12)
        & (decisions["reason"] != "TERMINAL_LIQUIDATION")
    ]
    rebalance_equity = pd.Series(
        [1.0, *changed_decisions["equity_after_cost"].astype(float).tolist()],
        dtype=float,
    )
    rebalance_episode_returns = rebalance_equity.pct_change(
        fill_method=None
    ).dropna()
    weekly_returns = (
        equity.resample("W-SUN").last().pct_change(fill_method=None).dropna()
    )
    sample_metrics = portfolio_sample_metrics(
        portfolio_period_returns=weekly_returns,
        position_episodes=position_episodes,
        rebalance_episode_returns=rebalance_episode_returns,
    )
    metrics = {
        "net_return": float(equity.iloc[-1] - 1.0),
        "gross_return": float(gross.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "omega": omega,
        "maximum_drawdown": maximum_drawdown,
        "average_drawdown": float(drawdown[drawdown < 0].mean())
        if bool((drawdown < 0).any())
        else 0.0,
        "maximum_drawdown_duration_observations": int(
            underwater_durations.max()
        )
        if len(underwater_durations)
        else 0,
        "time_underwater_fraction": float(underwater.mean()),
        "worst_week": float(weekly_for_risk.min()) if len(weekly_for_risk) else 0.0,
        "worst_month": float(monthly_for_risk.min()) if len(monthly_for_risk) else 0.0,
        "daily_cvar_95": float(tail.mean()) if len(tail) else 0.0,
        "calmar": (
            annualized_return / abs(maximum_drawdown)
            if maximum_drawdown < 0
            else math.inf
        ),
        "turnover": total_turnover,
        "rebalance_count": rebalance_count,
        "scheduled_rebalance_opportunities": scheduled_rebalance_opportunities,
        "unchanged_holding_decisions": unchanged_holding_decisions,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "selection_changes": selection_changes,
        "positive_years": int((yearly > 0).sum()),
        "negative_years": int((yearly < 0).sum()),
        "average_exposure": average_exposure,
        "median_exposure": float(exposure.median()) if len(exposure) else 0.0,
        "maximum_exposure_observed": float(exposure.max()) if len(exposure) else 0.0,
        "exposure_bucket_distribution": exposure_buckets,
        "average_positions_held": float(positions_held.mean())
        if len(positions_held)
        else 0.0,
        "cash_reason_attribution_average": cash_attribution_average,
        "return_per_average_exposure": (
            float(equity.iloc[-1] - 1.0) / average_exposure
            if average_exposure > 0
            else 0.0
        ),
        "maximum_positions_observed": (
            int((executed_weights.abs() > 1e-12).sum(axis=1).max())
            if not executed_weights.empty
            else 0
        ),
        "cash_fraction_average": 1.0 - average_exposure,
        "maximum_position_exposure_observed": (
            float(executed_weights.max(axis=1).max())
            if not executed_weights.empty
            else 0.0
        ),
        "daily_portfolio_observations": int(len(net_returns)),
        "portfolio_period_unit": "WEEKLY_PORTFOLIO_RETURN",
        "asset_pnl_attribution": {
            market: {
                "gross_pnl_amount": float(asset_pnl_amount[market]),
                "cost_amount": float(asset_cost_amount[market]),
                "net_pnl_amount": float(
                    asset_pnl_amount[market] - asset_cost_amount[market]
                ),
            }
            for market in closes.columns
        },
        "asset_pnl_reconciliation_error": float(
            sum(
                asset_pnl_amount[market] - asset_cost_amount[market]
                for market in closes.columns
            )
            - (net_equity - 1.0)
        ),
        **sample_metrics,
    }
    integrity = {
        "no_lookahead": True,
        "decision_at_close_execution_next_open": True,
        "closed_candles_only": True,
        "long_only_spot": bool((executed_weights >= -1e-12).all().all()),
        "cash_state_supported": True,
        "costs_applied_on_one_way_turnover": True,
        "terminal_liquidation_recorded": bool(
            not decisions.empty and decisions.iloc[-1]["reason"] == "TERMINAL_LIQUIDATION"
        ),
        "maximum_positions_respected": (
            metrics["maximum_positions_observed"] <= parameters.maximum_positions
        ),
        "maximum_exposure_respected": (
            executed_weights.abs().sum(axis=1).max()
            <= selected_policy.maximum_total_exposure + 1e-12
            if not executed_weights.empty
            else True
        ),
        "maximum_position_exposure_respected": (
            metrics["maximum_position_exposure_observed"]
            <= selected_policy.maximum_position_exposure + 1e-12
        ),
        "minimum_cash_respected": (
            executed_weights.abs().sum(axis=1).max()
            <= 1.0 - selected_policy.minimum_cash + 1e-12
            if not executed_weights.empty
            else True
        ),
        "fail_closed_allowed_universe": bool(selected_policy.allowed_markets),
        "allowed_markets": list(selected_policy.allowed_markets),
        "minimum_history_observations": selected_policy.minimum_history_observations,
        "asset_pnl_reconciled": (
            abs(float(metrics["asset_pnl_reconciliation_error"])) <= 1e-10
        ),
        "common_history_only": False,
        "point_in_time_asset_inception": True,
        "current_universe_retrospective": True,
        "benchmark_market": benchmark,
    }
    cost_breakdown = {
        "fee_rate": float(fee_rate),
        "slippage_bps": float(slippage_bps),
        "spread_bps": float(spread_bps),
        "one_way_cost_rate": float(one_way_cost),
        "total_one_way_turnover": float(total_turnover),
        "total_cost_amount": float(total_cost_amount),
        "gross_minus_net_return": float(gross.iloc[-1] - equity.iloc[-1]),
    }
    return RotationBacktestResult(
        parameters=parameters,
        portfolio_policy=selected_policy,
        capital_utilization_policy=capital_utilization_policy,
        metrics=metrics,
        integrity=integrity,
        cost_breakdown=cost_breakdown,
        equity_curve=equity,
        gross_equity_curve=gross,
        executed_weights=executed_weights,
        decisions=decisions,
        position_episodes=position_episodes,
    )


def _equity_curve_metrics(equity: pd.Series) -> dict[str, float | int]:
    selected = equity.astype(float)
    returns = selected.pct_change(fill_method=None).dropna()
    elapsed_days = max(
        1.0,
        (selected.index[-1] - selected.index[0]).total_seconds() / 86_400.0,
    )
    years = elapsed_days / 365.25
    standard = float(returns.std(ddof=0))
    drawdown = selected / selected.cummax() - 1.0
    return {
        "net_return": float(selected.iloc[-1] / selected.iloc[0] - 1.0),
        "annualized_return": float(
            (selected.iloc[-1] / selected.iloc[0]) ** (1.0 / years) - 1.0
        ),
        "annualized_volatility": standard * math.sqrt(365.25),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(365.25))
            if standard > 0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "observations": int(len(returns)),
    }


def _buy_and_hold_equity(
    prices: pd.DataFrame,
    weights: pd.Series,
    *,
    one_way_cost: float,
) -> pd.Series:
    selected_weights = weights[weights > 0].astype(float)
    selected_prices = prices.reindex(columns=selected_weights.index).dropna(how="any")
    if len(selected_prices) < 2:
        raise ValueError("benchmark requires at least two common price observations")
    ratios = selected_prices / selected_prices.iloc[0]
    cash = 1.0 - float(selected_weights.sum())
    entry_cost = float(selected_weights.sum()) * one_way_cost
    equity = cash + ratios.mul(selected_weights, axis=1).sum(axis=1) - entry_cost
    terminal_notional = float(
        (ratios.iloc[-1] * selected_weights).sum()
    )
    equity.iloc[-1] -= terminal_notional * one_way_cost
    equity.name = "benchmark_equity"
    return equity


def _point_in_time_equal_weight_equity(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    start: pd.Timestamp,
    exposure: float,
    minimum_history_observations: int,
    rebalance_days: int,
    one_way_cost: float,
) -> pd.Series:
    start_location = int(opens.index.searchsorted(start))
    if start_location >= len(opens) - 2:
        raise ValueError("equal-weight benchmark has insufficient timeline")
    current = pd.Series(0.0, index=opens.columns, dtype=float)
    equity = 1.0
    rows: list[tuple[pd.Timestamp, float]] = [(opens.index[start_location], equity)]
    for execution_index in range(start_location + 1, len(opens)):
        decision_index = execution_index - 1
        if (decision_index - start_location) % rebalance_days == 0:
            history = closes.iloc[: decision_index + 1].notna().sum()
            eligible = list(
                history[
                    (history >= minimum_history_observations)
                    & closes.iloc[decision_index].notna()
                    & opens.iloc[execution_index].notna()
                ].index
            )
            target = pd.Series(0.0, index=opens.columns, dtype=float)
            if eligible:
                target.loc[eligible] = exposure / len(eligible)
            turnover = float((target - current).abs().sum())
            equity -= equity * turnover * one_way_cost
            current = target
        terminal = execution_index == len(opens) - 1
        asset_returns = (
            closes.iloc[execution_index] / opens.iloc[execution_index] - 1.0
            if terminal
            else opens.iloc[execution_index + 1] / opens.iloc[execution_index] - 1.0
        )
        held = current[current > 1e-12].index
        if not asset_returns.reindex(held).notna().all():
            raise ValueError("equal-weight benchmark cannot value a held asset")
        equity *= 1.0 + float((current * asset_returns).sum())
        if terminal:
            equity -= equity * float(current.sum()) * one_way_cost
        timestamp = (
            closes.index[execution_index]
            if terminal
            else opens.index[execution_index + 1]
        )
        rows.append((timestamp, equity))
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in rows]),
        name="point_in_time_equal_weight_equity",
        dtype=float,
    )


def rotation_benchmark_suite(
    frames: Mapping[str, pd.DataFrame],
    parameters: RotationParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    benchmark_market: str = "BTC-EUR",
    portfolio_policy: RotationPortfolioPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate causal benchmarks and predeclared ablations on one timeline."""

    candidate = backtest_rotation(
        frames,
        parameters,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        benchmark_market=benchmark_market,
        portfolio_policy=portfolio_policy,
    )
    selected_policy = candidate.portfolio_policy
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=selected_policy,
    )
    start = candidate.equity_curve.index[0]
    price_panel = opens.loc[opens.index >= start].copy()
    price_panel.iloc[-1] = closes.reindex(price_panel.index).iloc[-1]
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    full_btc = _buy_and_hold_equity(
        price_panel,
        pd.Series({benchmark: 1.0}),
        one_way_cost=one_way_cost,
    )
    operational_btc = _buy_and_hold_equity(
        price_panel,
        pd.Series({benchmark: selected_policy.maximum_total_exposure}),
        one_way_cost=one_way_cost,
    )
    full_equal = _point_in_time_equal_weight_equity(
        opens,
        closes,
        start=start,
        exposure=1.0,
        minimum_history_observations=selected_policy.minimum_history_observations,
        rebalance_days=7,
        one_way_cost=one_way_cost,
    )
    operational_equal = _point_in_time_equal_weight_equity(
        opens,
        closes,
        start=start,
        exposure=selected_policy.maximum_total_exposure,
        minimum_history_observations=selected_policy.minimum_history_observations,
        rebalance_days=7,
        one_way_cost=one_way_cost,
    )
    btc_metrics = _equity_curve_metrics(full_btc)
    candidate_volatility = float(candidate.metrics["annualized_volatility"])
    btc_volatility = float(btc_metrics["annualized_volatility"])
    volatility_matched_exposure = float(
        np.clip(
            candidate_volatility / btc_volatility if btc_volatility > 0 else 0.0,
            0.0,
            1.0,
        )
    )
    volatility_matched_btc = _buy_and_hold_equity(
        price_panel,
        pd.Series({benchmark: volatility_matched_exposure}),
        one_way_cost=one_way_cost,
    )
    no_continuous = backtest_rotation(
        frames,
        replace(parameters, continuous_regime=False),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        benchmark_market=benchmark,
        portfolio_policy=selected_policy,
    )
    single_horizon = backtest_rotation(
        frames,
        replace(
            parameters,
            additional_momentum_lookbacks=(),
            require_btc_uptrend=False,
            continuous_regime=False,
        ),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        benchmark_market=benchmark,
        portfolio_policy=selected_policy,
    )
    return {
        "timeline": {
            "start": candidate.equity_curve.index[0].isoformat(),
            "end": candidate.equity_curve.index[-1].isoformat(),
            "calendar_days": (
                candidate.equity_curve.index[-1]
                - candidate.equity_curve.index[0]
            ).days,
        },
        "cost_model": {
            "fee_rate": fee_rate,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
            "one_way_cost_rate": one_way_cost,
        },
        "benchmarks": {
            "cash": {
                "net_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe": 0.0,
                "maximum_drawdown": 0.0,
            },
            "btc_buy_and_hold_full_exposure": btc_metrics,
            "btc_buy_and_hold_operational_exposure": _equity_curve_metrics(
                operational_btc
            ),
            "point_in_time_equal_weight_weekly_full_exposure": _equity_curve_metrics(
                full_equal
            ),
            "point_in_time_equal_weight_weekly_operational_exposure": _equity_curve_metrics(
                operational_equal
            ),
            "btc_buy_and_hold_volatility_matched": {
                **_equity_curve_metrics(volatility_matched_btc),
                "exposure": volatility_matched_exposure,
            },
        },
        "candidate": candidate.summary(),
        "ablations": {
            "multi_horizon_without_continuous_regime": no_continuous.summary(),
            "single_horizon_without_regime": single_horizon.summary(),
        },
        "interpretation": (
            "Benchmarks and ablations are descriptive diagnostics evaluated after "
            "candidate selection; they cannot retroactively qualify the frozen lead."
        ),
    }


def rotation_decision_snapshot(
    frames: Mapping[str, pd.DataFrame],
    parameters: RotationParameters,
    *,
    benchmark_market: str = "BTC-EUR",
    portfolio_policy: RotationPortfolioPolicy | None = None,
    capital_utilization_policy: CapitalUtilizationPolicy | None = None,
) -> dict[str, Any]:
    """Calculate today's frozen research decision without creating an order."""

    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    selected_policy = portfolio_policy or RotationPortfolioPolicy(
        maximum_total_exposure=parameters.gross_exposure,
        maximum_position_exposure=parameters.gross_exposure,
        minimum_cash=parameters.minimum_cash,
    )
    if capital_utilization_policy is not None:
        expected = (
            capital_utilization_policy.maximum_total_exposure,
            capital_utilization_policy.maximum_position_exposure,
            capital_utilization_policy.minimum_cash,
        )
        actual = (
            selected_policy.maximum_total_exposure,
            selected_policy.maximum_position_exposure,
            selected_policy.minimum_cash,
        )
        if any(
            abs(left - right) > 1e-12
            for left, right in zip(expected, actual, strict=True)
        ):
            raise ValueError(
                "capital utilization policy limits must match portfolio policy"
            )
    _, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=selected_policy,
    )
    warmup = max(
        *parameters.momentum_lookbacks,
        parameters.asset_ema_period,
        parameters.btc_ema_period,
        parameters.volatility_lookback,
        selected_policy.minimum_history_observations,
    )
    if len(closes) <= warmup:
        raise ValueError("insufficient history for frozen rotation decision")
    momentum = sum(
        np.log(closes / closes.shift(lookback)) / lookback
        for lookback in parameters.momentum_lookbacks
    ) / len(parameters.momentum_lookbacks)
    asset_ema = closes.ewm(
        span=parameters.asset_ema_period,
        adjust=False,
        min_periods=parameters.asset_ema_period,
    ).mean()
    btc_ema = closes[benchmark].ewm(
        span=parameters.btc_ema_period,
        adjust=False,
        min_periods=parameters.btc_ema_period,
    ).mean()
    volatility = closes.pct_change(fill_method=None).rolling(
        parameters.volatility_lookback,
        min_periods=parameters.volatility_lookback,
    ).std(ddof=0)
    target, decision = _target_weights(
        decision_index=len(closes) - 1,
        closes=closes,
        momentum=momentum,
        asset_ema=asset_ema,
        btc_ema=btc_ema,
        volatility=volatility,
        parameters=parameters,
        portfolio_policy=selected_policy,
        capital_utilization_policy=capital_utilization_policy,
        benchmark_market=benchmark,
    )
    scores = momentum.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "status": "FROZEN_FORWARD_RESEARCH",
        "decision_at": closes.index[-1].isoformat(),
        "execution_instruction": "NEXT_AVAILABLE_OPEN_HYPOTHETICAL_ONLY",
        "strategy_dna_hash": parameters.dna_hash,
        "portfolio_policy_hash": selected_policy.policy_hash,
        "capital_utilization_policy_hash": (
            capital_utilization_policy.policy_hash
            if capital_utilization_policy is not None
            else None
        ),
        "rank_scores": {
            market: float(score)
            for market, score in scores.sort_values(ascending=False).items()
        },
        "target_weights": {
            market: float(weight)
            for market, weight in target.items()
            if float(weight) > 1e-12
        },
        "cash_fraction": float(1.0 - target.sum()),
        "decision": decision,
        "orders_generated": 0,
        "orders_submitted": 0,
        "candidate_promotion_implied": False,
    }


def paired_block_bootstrap_difference(
    candidate_returns: pd.Series,
    control_returns: pd.Series,
    *,
    samples: int = 2_000,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired block bootstrap for the incremental mean return of two policies."""

    candidate = candidate_returns[
        ~candidate_returns.index.duplicated(keep="last")
    ].rename("candidate")
    control = control_returns[
        ~control_returns.index.duplicated(keep="last")
    ].rename("control")
    aligned = pd.concat(
        [candidate, control],
        axis=1,
        join="inner",
    ).dropna()
    differences = (
        aligned["candidate"].to_numpy(dtype=float)
        - aligned["control"].to_numpy(dtype=float)
    )
    if len(differences) < block_size or samples < 100:
        raise ValueError("paired block bootstrap requires enough data and samples")
    randomizer = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        indices: list[int] = []
        while len(indices) < len(differences):
            start = int(randomizer.integers(0, len(differences)))
            indices.extend(
                (start + offset) % len(differences)
                for offset in range(block_size)
            )
        means[sample] = float(
            differences[np.asarray(indices[: len(differences)], dtype=int)].mean()
        )
    observed = float(differences.mean())
    return {
        "observations": int(len(differences)),
        "bootstrap_samples": samples,
        "block_size": block_size,
        "mean_daily_return_difference": observed,
        "ci_lower_95": float(np.quantile(means, 0.025)),
        "ci_upper_95": float(np.quantile(means, 0.975)),
        "one_sided_pvalue_candidate_not_better": float(
            (1 + int((means <= 0.0).sum())) / (samples + 1)
        ),
    }


def capital_utilization_benchmark_suite(
    frames: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    minimum_history_observations: int,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    allowed_markets: Sequence[str],
    exposure_matches: Mapping[str, float] | None = None,
    benchmark_market: str = "BTC-EUR",
) -> dict[str, Any]:
    """Fixed and exposure-matched benchmarks for allocation-policy research."""

    policy = RotationPortfolioPolicy(
        allowed_markets=tuple(allowed_markets),
        maximum_total_exposure=1.0,
        maximum_position_exposure=1.0,
        minimum_cash=0.0,
        minimum_history_observations=minimum_history_observations,
    )
    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=policy,
    )
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    price_panel = opens.loc[opens.index >= start].copy()
    price_panel.iloc[-1] = closes.reindex(price_panel.index).iloc[-1]
    result: dict[str, Any] = {
        "cash": {
            "net_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "maximum_drawdown": 0.0,
        },
        "btc_buy_and_hold_100": _equity_curve_metrics(
            _buy_and_hold_equity(
                price_panel,
                pd.Series({benchmark: 1.0}),
                one_way_cost=one_way_cost,
            )
        ),
    }
    for exposure in (0.40, 0.60, 0.80, 1.0):
        equity = _point_in_time_equal_weight_equity(
            opens,
            closes,
            start=start,
            exposure=exposure,
            minimum_history_observations=minimum_history_observations,
            rebalance_days=7,
            one_way_cost=one_way_cost,
        )
        result[f"point_in_time_equal_weight_weekly_{int(exposure * 100)}"] = (
            _equity_curve_metrics(equity)
        )
    for name, raw_exposure in sorted((exposure_matches or {}).items()):
        exposure = float(np.clip(raw_exposure, 0.0, 1.0))
        equity = _point_in_time_equal_weight_equity(
            opens,
            closes,
            start=start,
            exposure=exposure,
            minimum_history_observations=minimum_history_observations,
            rebalance_days=7,
            one_way_cost=one_way_cost,
        )
        result[f"exposure_matched_equal_weight_{name}"] = {
            **_equity_curve_metrics(equity),
            "matched_exposure": exposure,
        }
    return result


def rotation_regime_coverage(
    decisions: pd.DataFrame,
    *,
    minimum_per_state: int = 5,
) -> dict[str, Any]:
    """Count independent decision observations across continuous regime axes."""

    usable = decisions[
        decisions["reason"].isin(
            {
                "RANKED_MOMENTUM",
                "NO_POSITIVE_TREND_ELIGIBLE_ASSET",
                "BTC_REGIME_FILTER",
                "SELECTED_ASSETS_NOT_EXECUTABLE",
            }
        )
    ].copy()
    axes = {
        "btc_trend": ("UP", "DOWN"),
        "volatility": ("HIGH", "LOW"),
        "breadth": ("BROAD", "NARROW"),
    }
    btc_uptrend = usable["btc_uptrend"].eq(True)
    counts = {
        "btc_trend": {
            "UP": int(btc_uptrend.sum()),
            "DOWN": int((~btc_uptrend).sum()),
        },
        "volatility": {
            state: int((usable["volatility_state"] == state).sum())
            for state in axes["volatility"]
        },
        "breadth": {
            state: int((usable["breadth_state"] == state).sum())
            for state in axes["breadth"]
        },
    }
    checks = {
        f"{axis}_{state}".lower(): counts[axis][state] >= minimum_per_state
        for axis, states in axes.items()
        for state in states
    }
    return {
        "decision_observations": int(len(usable)),
        "minimum_per_state": minimum_per_state,
        "counts": counts,
        "checks": checks,
        "passed": all(checks.values()),
    }


def rotation_period_metrics(
    equity_curve: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[dict[str, float | int], pd.Series]:
    """Return comparable metrics and daily returns for one frozen period."""

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
    selected = equity_curve.loc[
        (equity_curve.index >= start_at) & (equity_curve.index <= end_at)
    ].astype(float)
    if len(selected) < 30:
        raise ValueError("rotation evaluation period requires at least 30 observations")
    normalized = selected / float(selected.iloc[0])
    returns = normalized.pct_change(fill_method=None).dropna()
    standard = float(returns.std(ddof=0))
    elapsed_days = max(
        1.0,
        (normalized.index[-1] - normalized.index[0]).total_seconds() / 86_400.0,
    )
    years = elapsed_days / 365.25
    drawdown = normalized / normalized.cummax() - 1.0
    effective_sample_size, lag_one = _effective_sample_size(returns)
    metrics: dict[str, float | int] = {
        "observations": len(returns),
        "effective_sample_size": effective_sample_size,
        "lag_one_autocorrelation": lag_one,
        "net_return": float(normalized.iloc[-1] - 1.0),
        "annualized_return": float(normalized.iloc[-1] ** (1.0 / years) - 1.0),
        "annualized_volatility": standard * math.sqrt(365.25),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(365.25))
            if standard > 0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "daily_profit_factor": _profit_factor(returns),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "profit_factor_unit": "DAILY_PORTFOLIO_RETURN",
        "positive_observations": int((returns > 0).sum()),
        "negative_observations": int((returns < 0).sum()),
    }
    return metrics, returns


__all__ = [
    "CAPITAL_UTILIZATION_METRICS_VERSION",
    "CapitalUtilizationPolicy",
    "PORTFOLIO_METRICS_VERSION",
    "ROTATION_ENGINE_VERSION",
    "ROTATION_POLICY_VERSION",
    "RotationBacktestResult",
    "RotationParameters",
    "RotationPortfolioPolicy",
    "backtest_rotation",
    "capital_utilization_benchmark_suite",
    "capital_utilization_policy_set",
    "ensemble_rotation_parameter_grid",
    "portfolio_sample_metrics",
    "paired_block_bootstrap_difference",
    "rotation_benchmark_suite",
    "rotation_decision_snapshot",
    "rotation_period_metrics",
    "rotation_regime_coverage",
    "rotation_parameter_grid",
]
