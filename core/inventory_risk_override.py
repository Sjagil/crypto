"""Explicit, quantity-bounded treatment of operator-owned legacy inventory."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Mapping

from config.settings import Settings
from utils.common import read_json


def evaluate_inventory_risk_override(
    settings: Settings,
    balances: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    path = settings.paths.project_root / "config" / "inventory_risk_override.json"
    payload = dict(read_json(path)) if path.is_file() else {}
    asset = str(payload.get("asset") or "").upper()
    maximum_quantity = Decimal(str(payload.get("maximum_quantity") or "0"))
    current_quantity = Decimal("0")
    for row in balances:
        if str(row.get("symbol") or "").upper() != asset:
            continue
        current_quantity += Decimal(str(row.get("available") or "0"))
        current_quantity += Decimal(str(row.get("inOrder") or "0"))
    active = (
        payload.get("active") is True
        and bool(asset)
        and maximum_quantity > 0
        and current_quantity <= maximum_quantity
    )
    return {
        "active": active,
        "asset": asset or None,
        "current_quantity": str(current_quantity),
        "maximum_quantity": str(maximum_quantity),
        "allow_inventory_increase": False,
        "maximum_additional_managed_exposure_eur": str(
            Decimal(
                str(
                    payload.get("maximum_additional_managed_exposure_eur")
                    or "0"
                )
            )
        ),
        "approval_reference": payload.get("approval_reference"),
        "reason": payload.get("reason"),
        "artifact": str(path),
    }
