"""Regime-aware crypto correlation and portfolio concentration analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from utils.common import utc_now


@dataclass(frozen=True)
class CorrelationDecision:
    approved: bool
    size_multiplier: float
    reason_codes: tuple[str, ...]
    correlated_exposure: float
    stale: bool


class CorrelationAnalyzer:
    def __init__(
        self,
        *,
        window: int = 90,
        minimum_samples: int = 30,
        maximum_age: timedelta = timedelta(hours=6),
        cluster_threshold: float = 0.75,
    ) -> None:
        if minimum_samples < 3 or window < minimum_samples:
            raise ValueError("correlation window must cover the minimum samples")
        self.window = window
        self.minimum_samples = minimum_samples
        self.maximum_age = maximum_age
        self.cluster_threshold = cluster_threshold

    def _validate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("returns require a DatetimeIndex")
        selected = returns.sort_index().tail(self.window).dropna(how="all")
        if len(selected) < self.minimum_samples:
            raise ValueError("INSUFFICIENT_CORRELATION_SAMPLES")
        return selected

    def is_stale(self, returns: pd.DataFrame, now: datetime | None = None) -> bool:
        if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
            return True
        latest = returns.index.max()
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        return (pd.Timestamp(now or utc_now()) - latest) > self.maximum_age

    def pearson(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self._validate(returns).corr(method="pearson", min_periods=self.minimum_samples)

    def spearman(self, returns: pd.DataFrame) -> pd.DataFrame:
        return self._validate(returns).corr(method="spearman", min_periods=self.minimum_samples)

    def exponentially_weighted(self, returns: pd.DataFrame, span: int = 30) -> pd.DataFrame:
        selected = self._validate(returns)
        covariance = selected.ewm(span=span, adjust=False).cov().dropna()
        latest = covariance.loc[selected.index[-1]]
        variances = np.diag(latest)
        denominator = np.sqrt(np.outer(variances, variances))
        return pd.DataFrame(
            latest.to_numpy() / np.where(denominator == 0, np.nan, denominator),
            index=latest.index,
            columns=latest.columns,
        )

    def downside(self, returns: pd.DataFrame) -> pd.DataFrame:
        selected = self._validate(returns)
        downside = selected.where(selected.lt(0))
        return downside.corr(min_periods=max(5, self.minimum_samples // 3))

    def btc_beta(self, returns: pd.DataFrame, btc_column: str = "BTC-EUR") -> pd.Series:
        selected = self._validate(returns)
        if btc_column not in selected:
            raise ValueError("BTC benchmark is missing")
        variance = selected[btc_column].var()
        if not np.isfinite(variance) or variance <= 0:
            return pd.Series(np.nan, index=selected.columns, name="btc_beta")
        return selected.apply(lambda column: column.cov(selected[btc_column]) / variance).rename(
            "btc_beta"
        )

    def common_factor_exposure(self, returns: pd.DataFrame) -> pd.Series:
        selected = self._validate(returns).dropna()
        standardized = (selected - selected.mean()) / selected.std(ddof=0).replace(0, np.nan)
        values = standardized.dropna(axis=1).to_numpy()
        _, _, vectors = np.linalg.svd(values, full_matrices=False)
        loading = vectors[0]
        if loading.sum() < 0:
            loading *= -1
        return pd.Series(
            loading,
            index=standardized.dropna(axis=1).columns,
            name="common_factor_loading",
        )

    def clusters(self, returns: pd.DataFrame) -> dict[str, int]:
        correlation = self.pearson(returns).abs()
        unseen = set(correlation.columns)
        result: dict[str, int] = {}
        cluster_id = 0
        while unseen:
            seed = unseen.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    item
                    for item in unseen
                    if correlation.loc[current, item] >= self.cluster_threshold
                }
                unseen -= neighbors
                component |= neighbors
                frontier.extend(neighbors)
            for item in component:
                result[item] = cluster_id
            cluster_id += 1
        return result

    def risk_statistics(
        self, returns: pd.DataFrame, weights: pd.Series | dict[str, float]
    ) -> dict[str, Any]:
        selected = self._validate(returns).dropna()
        weights_series = pd.Series(weights, dtype=float).reindex(selected.columns).fillna(0)
        total = weights_series.abs().sum()
        if total <= 0:
            raise ValueError("portfolio weights must be non-zero")
        weights_series /= total
        covariance = selected.cov() * 365.25
        vector = weights_series.to_numpy()
        portfolio_variance = float(vector @ covariance.to_numpy() @ vector)
        if portfolio_variance <= 0:
            marginal = pd.Series(0.0, index=weights_series.index)
        else:
            marginal = pd.Series(
                covariance.to_numpy() @ vector / np.sqrt(portfolio_variance),
                index=weights_series.index,
            )
        component = weights_series * marginal
        normalized_component = component / component.abs().sum() if component.abs().sum() else component
        hhi = float(np.square(weights_series.abs()).sum())
        effective_positions = 1.0 / hhi if hhi else 0.0
        correlation = selected.corr().fillna(0).to_numpy()
        concentration = float(vector @ correlation @ vector)
        return {
            "portfolio_volatility": np.sqrt(max(0.0, portfolio_variance)),
            "marginal_contribution_to_risk": marginal.to_dict(),
            "component_contribution_to_risk": normalized_component.to_dict(),
            "effective_position_count": effective_positions,
            "concentration_score": concentration,
            "weight_hhi": hhi,
            "clusters": self.clusters(selected),
        }

    def assess_proposal(
        self,
        *,
        market: str,
        proposed_weight: float,
        existing_weights: dict[str, float],
        returns: pd.DataFrame,
        correlated_risk_cap: float,
        large_position_threshold: float = 0.10,
    ) -> CorrelationDecision:
        stale = self.is_stale(returns)
        if stale and proposed_weight >= large_position_threshold:
            return CorrelationDecision(
                False,
                0.0,
                ("STALE_CORRELATION_FAIL_CLOSED",),
                0.0,
                True,
            )
        try:
            correlation = self.pearson(returns)
        except ValueError:
            if proposed_weight >= large_position_threshold:
                return CorrelationDecision(
                    False,
                    0.0,
                    ("INSUFFICIENT_CORRELATION_DATA",),
                    0.0,
                    stale,
                )
            return CorrelationDecision(True, 0.5, ("CORRELATION_UNCERTAIN_REDUCED",), 0.0, stale)
        if market not in correlation:
            return CorrelationDecision(
                proposed_weight < large_position_threshold,
                0.5 if proposed_weight < large_position_threshold else 0.0,
                ("MISSING_PROPOSED_MARKET_CORRELATION",),
                0.0,
                stale,
            )
        correlated = sum(
            weight
            for other, weight in existing_weights.items()
            if other in correlation and abs(correlation.loc[market, other]) >= self.cluster_threshold
        )
        total = correlated + proposed_weight
        if total > correlated_risk_cap:
            return CorrelationDecision(
                False,
                0.0,
                ("CORRELATED_RISK_CAP_EXCEEDED",),
                correlated,
                stale,
            )
        headroom = max(0.0, correlated_risk_cap - correlated)
        multiplier = min(1.0, headroom / proposed_weight) if proposed_weight else 0.0
        reason = "APPROVED" if multiplier == 1 else "CORRELATED_EXPOSURE_REDUCED"
        return CorrelationDecision(True, multiplier, (reason,), correlated, stale)


__all__ = ["CorrelationAnalyzer", "CorrelationDecision"]
