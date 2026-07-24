"""Transparent, non-trading long-horizon crypto investment scoring."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from utils.common import stable_hash


@dataclass(frozen=True)
class InvestmentSubscore:
    name: str
    score: float | None
    available_components: int
    expected_components: int
    components: Mapping[str, float | None]


@dataclass(frozen=True)
class InvestmentScore:
    overall_score: float | None
    confidence: float
    subscores: Mapping[str, InvestmentSubscore]
    missing_components: tuple[str, ...]
    methodology_version: str
    configuration_hash: str
    investing_only: bool = True
    creates_trading_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Inputs are normalized research assessments on [0, 100]. A negative weight
# means that a high raw assessment represents risk and is inverted.
DIMENSIONS: dict[str, dict[str, float]] = {
    "valuation": {
        "market_cap_fdv": 1.0,
        "market_cap_tvl": -1.0,
        "market_cap_revenue": -1.0,
        "revenue_yield": 1.0,
    },
    "adoption": {
        "active_users": 1.0,
        "new_users": 1.0,
        "developer_activity": 1.0,
        "ecosystem_growth": 1.0,
    },
    "usage": {
        "transaction_count": 1.0,
        "transfer_volume": 1.0,
        "dex_volume": 1.0,
        "stablecoin_activity": 1.0,
    },
    "tokenomics": {
        "token_utility": 1.0,
        "fee_capture": 1.0,
        "burns": 1.0,
        "staking_demand": 1.0,
    },
    "dilution_risk": {
        "emissions": -1.0,
        "inflation": -1.0,
        "unlock_to_volume": -1.0,
        "insider_allocation": -1.0,
    },
    "liquidity": {
        "spot_liquidity": 1.0,
        "exchange_availability": 1.0,
        "tvl": 1.0,
        "stablecoin_liquidity": 1.0,
    },
    "decentralization": {
        "validator_count": 1.0,
        "nakamoto_coefficient": 1.0,
        "client_diversity": 1.0,
        "governance_concentration": -1.0,
    },
    "security": {
        "audit_quality": 1.0,
        "bug_bounty": 1.0,
        "exploit_history": -1.0,
        "downtime": -1.0,
        "admin_key_risk": -1.0,
    },
    "holder_concentration": {
        "top_10_holder_concentration": -1.0,
        "whale_concentration": -1.0,
        "team_wallet_concentration": -1.0,
        "holder_growth": 1.0,
    },
    "economic_sustainability": {
        "protocol_revenue": 1.0,
        "token_holder_revenue": 1.0,
        "real_yield": 1.0,
        "incentive_dependency": -1.0,
    },
    "macro_sensitivity": {
        "rate_sensitivity": -1.0,
        "liquidity_sensitivity": -1.0,
        "btc_beta": -1.0,
        "risk_off_resilience": 1.0,
    },
    "data_quality": {
        "source_coverage": 1.0,
        "point_in_time_quality": 1.0,
        "freshness": 1.0,
        "independent_sources": 1.0,
    },
}


def _normalized(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if not 0.0 <= number <= 100.0:
        raise ValueError("investment component scores must be between 0 and 100")
    return number


class InvestmentScorer:
    """Calculate visible subscores; missing inputs reduce confidence only."""

    methodology_version = "1.0.0"

    def __init__(
        self,
        dimensions: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.dimensions = {
            name: dict(components)
            for name, components in (dimensions or DIMENSIONS).items()
        }
        self.configuration_hash = stable_hash(
            {
                "methodology_version": self.methodology_version,
                "dimensions": self.dimensions,
            },
            length=64,
        )

    def score(self, values: Mapping[str, Any]) -> InvestmentScore:
        subscores: dict[str, InvestmentSubscore] = {}
        missing: list[str] = []
        total_available = 0
        total_expected = 0
        for dimension, components in self.dimensions.items():
            contributions: dict[str, float | None] = {}
            weighted: list[tuple[float, float]] = []
            for component, weight in components.items():
                value = _normalized(values.get(component))
                total_expected += 1
                if value is None:
                    missing.append(component)
                    contributions[component] = None
                    continue
                total_available += 1
                transformed = value if weight >= 0 else 100.0 - value
                contributions[component] = transformed
                weighted.append((transformed, abs(float(weight))))
            denominator = sum(weight for _, weight in weighted)
            subscore = (
                sum(value * weight for value, weight in weighted) / denominator
                if denominator
                else None
            )
            subscores[dimension] = InvestmentSubscore(
                name=dimension,
                score=subscore,
                available_components=len(weighted),
                expected_components=len(components),
                components=contributions,
            )
        available_subscores = [
            item.score for item in subscores.values() if item.score is not None
        ]
        overall = (
            sum(available_subscores) / len(available_subscores)
            if available_subscores
            else None
        )
        confidence = total_available / total_expected if total_expected else 0.0
        return InvestmentScore(
            overall_score=overall,
            confidence=confidence,
            subscores=subscores,
            missing_components=tuple(sorted(missing)),
            methodology_version=self.methodology_version,
            configuration_hash=self.configuration_hash,
        )


__all__ = [
    "DIMENSIONS",
    "InvestmentScore",
    "InvestmentScorer",
    "InvestmentSubscore",
]
