"""Fail-closed baseline for assets held before a live strategy starts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.settings import Settings
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "preexisting_account_inventory_v1"


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() and parsed > 0 else Decimal("0")


def inventory_baseline_path(settings: Settings) -> Path:
    return (
        settings.paths.output_dir
        / "governance"
        / "preexisting_account_inventory.json"
    )


def write_inventory_baseline(
    settings: Settings,
    *,
    holdings: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    quantities: dict[str, str] = {}
    for row in holdings:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol == "EUR" or not symbol.replace("-", "").isalnum():
            continue
        quantity = _decimal(row.get("total"))
        if quantity > 0:
            quantities[symbol] = str(quantity)
    identity = {
        "strategy_id": str(authority.get("strategy_id") or ""),
        "strategy_dna": str(authority.get("strategy_dna") or ""),
        "market": str(authority.get("market") or "").upper(),
        "approval_reference": str(
            authority.get("operator_approval_reference") or ""
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_iso(),
        "source": "BITVAVO_PRIVATE_BALANCE_READ",
        "authority": identity,
        "quantities": dict(sorted(quantities.items())),
        "orders_generated": 0,
        "orders_submitted": 0,
        "withdrawals_attempted": 0,
    }
    payload["inventory_hash"] = stable_hash(
        {
            "schema_version": payload["schema_version"],
            "source": payload["source"],
            "authority": identity,
            "quantities": payload["quantities"],
        },
        length=64,
    )
    path = inventory_baseline_path(settings)
    atomic_write_json(path, payload)
    return {**payload, "artifact": str(path)}


def load_inventory_baseline(
    settings: Settings,
    *,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Decimal], tuple[str, ...]]:
    path = inventory_baseline_path(settings)
    if not path.is_file():
        return {}, ("PREEXISTING_INVENTORY_BASELINE_MISSING",)
    payload = dict(read_json(path))
    failures: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append("PREEXISTING_INVENTORY_SCHEMA_MISMATCH")
    expected_identity = {
        "strategy_id": str(authority.get("strategy_id") or ""),
        "strategy_dna": str(authority.get("strategy_dna") or ""),
        "market": str(authority.get("market") or "").upper(),
        "approval_reference": str(
            authority.get("operator_approval_reference") or ""
        ),
    }
    if dict(payload.get("authority") or {}) != expected_identity:
        failures.append("PREEXISTING_INVENTORY_AUTHORITY_MISMATCH")
    expected_hash = stable_hash(
        {
            "schema_version": payload.get("schema_version"),
            "source": payload.get("source"),
            "authority": payload.get("authority"),
            "quantities": payload.get("quantities"),
        },
        length=64,
    )
    if payload.get("inventory_hash") != expected_hash:
        failures.append("PREEXISTING_INVENTORY_HASH_MISMATCH")
    quantities = {
        str(symbol).upper(): _decimal(quantity)
        for symbol, quantity in dict(payload.get("quantities") or {}).items()
    }
    return quantities, tuple(dict.fromkeys(failures))


def reconcile_inventory(
    balances: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Decimal],
    *,
    prices_eur: Mapping[str, Decimal] | None = None,
    minimum_material_excess_eur: Decimal = Decimal("0"),
) -> dict[str, Any]:
    current: dict[str, Decimal] = {}
    for row in balances:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol == "EUR":
            continue
        quantity = _decimal(row.get("available")) + _decimal(
            row.get("inOrder")
        )
        if quantity > 0:
            current[symbol] = quantity
    raw_excess = {
        symbol: max(
            Decimal("0"),
            quantity - _decimal(baseline.get(symbol)),
        )
        for symbol, quantity in current.items()
    }
    raw_excess = {
        symbol: quantity
        for symbol, quantity in raw_excess.items()
        if quantity > 0
    }
    price_map = {
        str(symbol).upper(): _decimal(price)
        for symbol, price in dict(prices_eur or {}).items()
    }
    ignored_dust_excess: dict[str, Decimal] = {}
    excess: dict[str, Decimal] = {}
    for symbol, quantity in raw_excess.items():
        price = price_map.get(symbol, Decimal("0"))
        if (
            minimum_material_excess_eur > 0
            and price > 0
            and quantity * price < minimum_material_excess_eur
        ):
            ignored_dust_excess[symbol] = quantity
        else:
            excess[symbol] = quantity
    missing_or_reduced = {
        symbol: str(max(Decimal("0"), quantity - current.get(symbol, Decimal("0"))))
        for symbol, quantity in baseline.items()
        if current.get(symbol, Decimal("0")) < quantity
    }
    return {
        "current": {symbol: str(value) for symbol, value in sorted(current.items())},
        "baseline": {
            symbol: str(value) for symbol, value in sorted(baseline.items())
        },
        "excess": {symbol: str(value) for symbol, value in sorted(excess.items())},
        "ignored_dust_excess": {
            symbol: str(value)
            for symbol, value in sorted(ignored_dust_excess.items())
        },
        "minimum_material_excess_eur": str(minimum_material_excess_eur),
        "missing_or_reduced": dict(sorted(missing_or_reduced.items())),
        "reconciled": not excess,
    }


def classify_account_inventory(
    balances: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Decimal],
    managed_quantities: Mapping[str, Decimal] | None = None,
) -> list[dict[str, str]]:
    """Classify every real holding without claiming lifecycle ownership.

    A baseline holding is economically visible as grandfathered inventory.
    Only quantities present in canonical execution state are called managed.
    Any other positive holding is external/manual and remains fail-closed for
    autonomous exit logic, while still counting toward portfolio heat.
    """

    managed = {
        str(symbol).upper(): _decimal(quantity)
        for symbol, quantity in dict(managed_quantities or {}).items()
    }
    rows: list[dict[str, str]] = []
    for balance in balances:
        symbol = str(balance.get("symbol") or "").strip().upper()
        if not symbol or symbol == "EUR":
            continue
        available = _decimal(balance.get("available"))
        in_order = _decimal(balance.get("inOrder"))
        quantity = available + in_order
        if quantity <= 0:
            continue
        managed_quantity = min(quantity, managed.get(symbol, Decimal("0")))
        baseline_quantity = min(
            max(Decimal("0"), quantity - managed_quantity),
            _decimal(baseline.get(symbol)),
        )
        external_quantity = max(
            Decimal("0"),
            quantity - managed_quantity - baseline_quantity,
        )
        if managed_quantity > 0:
            classification = "MANAGED_POSITION"
        elif baseline_quantity > 0:
            classification = "GRANDFATHERED_INVENTORY"
        else:
            classification = "MANUAL_EXTERNAL_POSITION"
        rows.append(
            {
                "symbol": symbol,
                "market": f"{symbol}-EUR",
                "classification": classification,
                "total_quantity": str(quantity),
                "managed_quantity": str(managed_quantity),
                "grandfathered_quantity": str(baseline_quantity),
                "external_quantity": str(external_quantity),
                "mixed_ownership": str(
                    managed_quantity > 0 and external_quantity > 0
                ).lower(),
                "autonomous_exit_authority": str(
                    classification == "MANAGED_POSITION"
                ).lower(),
                "autonomous_exit_authority_quantity": str(managed_quantity),
                "external_quantity_remains_unmanaged": "true",
                "counts_toward_portfolio_heat": "true",
            }
        )
    return sorted(rows, key=lambda row: row["symbol"])


def expected_inventory_after_canonical_fills(
    settings: Settings,
    baseline: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    """Apply canonical post-baseline BUY/SELL fills to expected inventory."""

    expected = {
        str(symbol).upper(): _decimal(quantity)
        for symbol, quantity in baseline.items()
    }
    baseline_path = inventory_baseline_path(settings)
    if not baseline_path.is_file():
        return expected
    baseline_payload = dict(read_json(baseline_path))
    try:
        created_at = datetime.fromisoformat(
            str(baseline_payload.get("created_at") or "").replace(
                "Z", "+00:00"
            )
        )
    except ValueError:
        return expected
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    if not ledger.is_file():
        return expected
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != "FILL":
            continue
        try:
            recorded_at = datetime.fromisoformat(
                str(event.get("recorded_at") or "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=UTC)
        if recorded_at.astimezone(UTC) <= created_at.astimezone(UTC):
            continue
        payload = dict(event.get("payload") or {})
        market = str(payload.get("market") or "").upper()
        symbol = market.split("-", 1)[0]
        side = str(payload.get("side") or "").upper()
        quantity = _decimal(payload.get("quantity"))
        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0:
            continue
        current = expected.get(symbol, Decimal("0"))
        expected[symbol] = (
            current + quantity
            if side == "BUY"
            else max(Decimal("0"), current - quantity)
        )
    return expected


def grandfathered_inventory_from_expected(
    expected_inventory: Mapping[str, Decimal],
    managed_quantities: Mapping[str, Decimal] | None = None,
) -> dict[str, Decimal]:
    """Remove currently managed quantities from expected account inventory.

    ``expected_inventory_after_canonical_fills`` is the expected *total*
    wallet inventory and therefore already includes canonical open positions.
    Ownership classification must not count those quantities a second time as
    grandfathered inventory.
    """

    managed = {
        str(symbol).upper(): _decimal(quantity)
        for symbol, quantity in dict(managed_quantities or {}).items()
    }
    return {
        str(symbol).upper(): max(
            Decimal("0"),
            _decimal(quantity) - managed.get(str(symbol).upper(), Decimal("0")),
        )
        for symbol, quantity in expected_inventory.items()
        if _decimal(quantity) > managed.get(str(symbol).upper(), Decimal("0"))
    }


__all__ = [
    "classify_account_inventory",
    "expected_inventory_after_canonical_fills",
    "grandfathered_inventory_from_expected",
    "inventory_baseline_path",
    "load_inventory_baseline",
    "reconcile_inventory",
    "write_inventory_baseline",
]
