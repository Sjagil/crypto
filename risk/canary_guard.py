"""Immutable, fail-closed limits for the eventual €5 live canary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from config.settings import Settings
from utils.common import atomic_write_json, stable_hash

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class CanaryPolicy:
    """The live sleeve is deliberately disabled until every gate passes."""

    policy_id: str
    enabled: bool
    maximum_order_eur: Decimal
    maximum_total_eur: Decimal
    maximum_open_positions: int
    leverage: Decimal
    shorts_allowed: bool
    autoscale: bool
    spot_only: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "CanaryPolicy":
        execution = settings.execution
        return cls(
            policy_id="BITVAVO_SPOT_CANARY_V1",
            enabled=execution.live_canary_enabled,
            maximum_order_eur=Decimal(
                str(execution.maximum_live_order_eur)
            ),
            maximum_total_eur=Decimal(
                str(execution.maximum_live_total_eur)
            ),
            maximum_open_positions=(
                execution.maximum_live_open_positions
            ),
            leverage=Decimal("1"),
            shorts_allowed=False,
            autoscale=execution.live_canary_autoscale,
            spot_only=execution.spot_only,
        )

    @classmethod
    def from_cap_limits(
        cls,
        settings: Settings,
        *,
        maximum_order_eur: Decimal,
        maximum_total_eur: Decimal,
        maximum_open_positions: int,
        capital_level: int,
        enabled: bool,
    ) -> "CanaryPolicy":
        """Build an evidence- and operator-authorized runtime policy.

        The canonical settings remain the immutable Level-1 baseline. Higher
        values can reach this constructor only after practical-governance
        evidence, a separate capital-level approval and canonical preflight.
        """

        return cls(
            policy_id=f"BITVAVO_SPOT_CANARY_CAPITAL_LEVEL_{capital_level}",
            enabled=enabled,
            maximum_order_eur=maximum_order_eur,
            maximum_total_eur=maximum_total_eur,
            maximum_open_positions=maximum_open_positions,
            leverage=Decimal("1"),
            shorts_allowed=False,
            autoscale=False,
            spot_only=settings.execution.spot_only,
        )

    def validate(self) -> None:
        if self.maximum_order_eur <= ZERO:
            raise ValueError("canary maximum order must be positive")
        if self.maximum_total_eur <= ZERO:
            raise ValueError("canary maximum total must be positive")
        if self.maximum_order_eur > self.maximum_total_eur:
            raise ValueError("canary order cap exceeds total cap")
        if not 1 <= self.maximum_open_positions <= 3:
            raise ValueError("canary must allow between one and three positions")
        if self.leverage != Decimal("1"):
            raise ValueError("canary leverage must equal one")
        if self.shorts_allowed or self.autoscale or not self.spot_only:
            raise ValueError("canary policy violates spot-only bounds")

    def manifest(self) -> dict[str, Any]:
        self.validate()
        body = {
            "schema_version": "live_canary_policy_v1",
            **{
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in asdict(self).items()
            },
            "activation_status": (
                "ENABLED_AFTER_ALL_GATES"
                if self.enabled
                else "REGISTERED_DISABLED"
            ),
            "registration_is_order_permission": False,
            "human_change_requires_new_manifest": True,
        }
        return {
            **body,
            "policy_hash": stable_hash(body, length=64),
        }


@dataclass(frozen=True, slots=True)
class CanaryDecision:
    approved: bool
    approved_notional_eur: Decimal
    reason_code: str


class InstitutionalCanaryGuard:
    """Bound a prospective buy without ever increasing it to a venue minimum."""

    def __init__(self, policy: CanaryPolicy) -> None:
        policy.validate()
        self.policy = policy

    def assess_buy(
        self,
        *,
        requested_notional_eur: Decimal,
        current_total_exposure_eur: Decimal | None,
        current_open_positions: int,
        exchange_minimum_order_eur: Decimal,
    ) -> CanaryDecision:
        if not self.policy.enabled:
            return CanaryDecision(False, ZERO, "CANARY_DISABLED")
        if requested_notional_eur <= ZERO:
            return CanaryDecision(
                False,
                ZERO,
                "CANARY_NON_POSITIVE_REQUEST",
            )
        if current_total_exposure_eur is None:
            return CanaryDecision(
                False,
                ZERO,
                "CANARY_EXPOSURE_NOT_RECONCILED",
            )
        if (
            current_total_exposure_eur < ZERO
            or current_open_positions < 0
        ):
            return CanaryDecision(
                False,
                ZERO,
                "CANARY_RECONCILIATION_INVALID",
            )
        if current_open_positions >= self.policy.maximum_open_positions:
            return CanaryDecision(
                False,
                ZERO,
                "CANARY_POSITION_LIMIT",
            )
        remaining = max(
            ZERO,
            self.policy.maximum_total_eur
            - current_total_exposure_eur,
        )
        approved = min(
            requested_notional_eur,
            self.policy.maximum_order_eur,
            remaining,
        )
        if approved < exchange_minimum_order_eur:
            return CanaryDecision(
                False,
                ZERO,
                "CANARY_BELOW_EXCHANGE_MINIMUM",
            )
        return CanaryDecision(True, approved, "CANARY_APPROVED")


def write_canary_policy_manifest(
    settings: Settings,
    path: Path,
) -> dict[str, Any]:
    """Write one content-addressed policy record without activating it."""

    manifest = CanaryPolicy.from_settings(settings).manifest()
    if path.is_file():
        from utils.common import read_json

        if read_json(path) != manifest:
            raise RuntimeError("CANARY_POLICY_HISTORY_REVISION")
    else:
        atomic_write_json(path, manifest)
    return {
        **manifest,
        "manifest_path": str(path),
    }


__all__ = [
    "CanaryDecision",
    "CanaryPolicy",
    "InstitutionalCanaryGuard",
    "write_canary_policy_manifest",
]
