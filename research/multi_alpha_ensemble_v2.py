"""Frozen classical trend plus residual-reversal ensemble DNA.

This module adds no new signal parameters.  It combines two already frozen,
economically distinct classical sleeves under one causal meta-allocation and
the existing 40% total / 20% per-asset portfolio limits.  Historical evidence
is explicitly discovery-contaminated and cannot authorize promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from utils.common import stable_hash

MULTI_ALPHA_ENSEMBLE_V2_FAMILY = (
    "FROZEN_ABSOLUTE_MOMENTUM_RESIDUAL_REVERSAL_ENSEMBLE"
)
MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION = "2.0.0"

ABSOLUTE_MOMENTUM_V1_COMPONENT_DNA = (
    "1d14e010495a45d9ea60090077c73c1fd357d897832611ca9f722f8f670c2012"
)
RESIDUAL_REVERSAL_V1_COMPONENT_DNA = (
    "4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0"
)
FROZEN_COMPONENT_DNA_V2 = (
    (
        "ABSOLUTE_MOMENTUM_VOL_05",
        ABSOLUTE_MOMENTUM_V1_COMPONENT_DNA,
    ),
    (
        "RESIDUAL_REVERSAL_B60_H5_Z20",
        RESIDUAL_REVERSAL_V1_COMPONENT_DNA,
    ),
)


@dataclass(frozen=True, slots=True)
class MultiAlphaEnsembleV2Parameters:
    """The single preregistered v2 meta-strategy DNA."""

    target_annualized_volatility: float = 0.10
    volatility_lookback: int = 60
    rebalance_weekday: int = 6
    maximum_positions: int = 2
    component_allocation: str = "EQUAL_FIXED_SLEEVES"
    risk_model: str = "CAUSAL_DIAGONAL_VOLATILITY"
    component_dna: tuple[tuple[str, str], ...] = (
        FROZEN_COMPONENT_DNA_V2
    )

    def __post_init__(self) -> None:
        if self.target_annualized_volatility != 0.10:
            raise ValueError("v2 volatility target is fixed at 10%")
        if self.volatility_lookback != 60:
            raise ValueError("v2 volatility lookback is fixed at 60")
        if self.rebalance_weekday != 6:
            raise ValueError("v2 rebalance weekday is fixed at Sunday")
        if self.maximum_positions != 2:
            raise ValueError("v2 maximum positions is fixed at two")
        if self.component_allocation != "EQUAL_FIXED_SLEEVES":
            raise ValueError("v2 requires equal fixed sleeves")
        if self.risk_model != "CAUSAL_DIAGONAL_VOLATILITY":
            raise ValueError("v2 risk model is fixed")
        if self.component_dna != FROZEN_COMPONENT_DNA_V2:
            raise ValueError("v2 component DNA differs from preregistration")

    @property
    def strategy_family(self) -> str:
        return MULTI_ALPHA_ENSEMBLE_V2_FAMILY

    @property
    def engine_version(self) -> str:
        return MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": self.strategy_family,
                "engine_version": self.engine_version,
                "parameters": asdict(self),
            },
            length=64,
        )


__all__ = [
    "ABSOLUTE_MOMENTUM_V1_COMPONENT_DNA",
    "FROZEN_COMPONENT_DNA_V2",
    "MULTI_ALPHA_ENSEMBLE_V2_ENGINE_VERSION",
    "MULTI_ALPHA_ENSEMBLE_V2_FAMILY",
    "RESIDUAL_REVERSAL_V1_COMPONENT_DNA",
    "MultiAlphaEnsembleV2Parameters",
]
