"""Canonical transaction-cost assumptions shared by research and validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from config.settings import Settings
from utils.common import stable_hash


@dataclass(frozen=True)
class CanonicalCostModel:
    """Versioned source of fee, spread and slippage assumptions.

    Exact engines may adapt these values to their native calculation types,
    but must retain ``cost_model_version`` in their evidence artifacts.
    """

    cost_model_version: str
    maker_fee_fraction: float
    taker_fee_fraction: float
    spread_bps: float
    slippage_bps: float
    failed_execution_allowance_bps: float = 0.0
    partial_fill_impact_bps: float = 0.0
    calibration_status: str = "SETTINGS_BASELINE_WITH_PAPER_TCA_ONLY_WHERE_OBSERVED"

    def __post_init__(self) -> None:
        for name in ("maker_fee_fraction", "taker_fee_fraction"):
            value = float(getattr(self, name))
            if not 0 <= value <= 0.05:
                raise ValueError(f"{name} must be between zero and 5%")
        for name in (
            "spread_bps",
            "slippage_bps",
            "failed_execution_allowance_bps",
            "partial_fill_impact_bps",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not self.cost_model_version:
            raise ValueError("cost model version is required")

    @classmethod
    def create(cls, **values: Any) -> "CanonicalCostModel":
        identity = {
            key: value
            for key, value in values.items()
            if key not in {"cost_model_version", "calibration_status"}
        }
        return cls(
            cost_model_version=f"canonical_cost_v1:{stable_hash(identity, length=20)}",
            **values,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "CanonicalCostModel":
        return cls.create(
            maker_fee_fraction=settings.costs.maker_fee,
            taker_fee_fraction=settings.costs.taker_fee,
            spread_bps=settings.costs.spread_bps,
            slippage_bps=settings.costs.slippage_bps,
            failed_execution_allowance_bps=0.0,
            partial_fill_impact_bps=0.0,
        )

    def stressed(
        self,
        *,
        additional_roundtrip_bps: float = 0.0,
        spread_multiplier: float = 1.0,
        slippage_multiplier: float = 1.0,
    ) -> "CanonicalCostModel":
        if additional_roundtrip_bps < 0:
            raise ValueError("additional cost stress cannot be negative")
        if spread_multiplier < 1 or slippage_multiplier < 1:
            raise ValueError("liquidity stress multipliers cannot be below one")
        side_addition = additional_roundtrip_bps / 2.0
        values = {
            "maker_fee_fraction": self.maker_fee_fraction,
            "taker_fee_fraction": self.taker_fee_fraction,
            "spread_bps": self.spread_bps * spread_multiplier,
            "slippage_bps": self.slippage_bps * slippage_multiplier
            + side_addition,
            "failed_execution_allowance_bps": self.failed_execution_allowance_bps,
            "partial_fill_impact_bps": self.partial_fill_impact_bps,
            "calibration_status": self.calibration_status,
        }
        return type(self).create(**values)

    def with_calibration(self, calibration_status: str) -> "CanonicalCostModel":
        if not calibration_status:
            raise ValueError("calibration status is required")
        return replace(self, calibration_status=calibration_status)

    def exact_backtest_inputs(self, *, multiplier: float = 1.0) -> dict[str, float]:
        if multiplier < 1:
            raise ValueError("cost multiplier cannot be below one")
        return {
            "fee_fraction": self.taker_fee_fraction,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "multiplier": multiplier,
        }

    @property
    def conservative_roundtrip_fraction(self) -> Decimal:
        """Conservative taker round-trip drag used at entry boundaries."""

        fee = Decimal(str(self.taker_fee_fraction)) * Decimal("2")
        spread = Decimal(str(self.spread_bps)) / Decimal("10000")
        slippage = (
            Decimal(str(self.slippage_bps)) * Decimal("2")
            / Decimal("10000")
        )
        execution_allowance = (
            Decimal(str(self.failed_execution_allowance_bps))
            + Decimal(str(self.partial_fill_impact_bps))
        ) / Decimal("10000")
        return fee + spread + slippage + execution_allowance


__all__ = ["CanonicalCostModel"]
