"""Orderless remediation evidence for externally created spot inventory."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from config.settings import Settings
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso, utc_now

SCHEMA_VERSION = "external_inventory_remediation_v1"
MIGRATION_SCHEMA_VERSION = "external_inventory_migration_contract_v1"


def _decimal(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _external_fill_evidence(
    path: Path,
    markets: set[str],
) -> dict[str, list[dict[str, str]]]:
    evidence = {market: [] for market in markets}
    if not path.is_file():
        return evidence
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        payload = dict(row.get("payload") or {})
        market = str(payload.get("market") or "").upper()
        if (
            row.get("event") != "BITVAVO_ACCOUNT_FILL"
            or market not in markets
            or payload.get("client_order_public_id")
        ):
            continue
        evidence[market].append(
            {
                "event_id": str(row.get("event_id") or ""),
                "market": market,
                "side": str(payload.get("side") or "").upper(),
                "quantity": str(_decimal(payload.get("amount"))),
                "price_eur": str(_decimal(payload.get("fill_price"))),
                "fee": str(_decimal(payload.get("fee"))),
                "fee_currency": str(payload.get("fee_currency") or ""),
                "venue_timestamp": str(payload.get("venue_timestamp") or ""),
                "canonical_client_order_identity_present": "false",
            }
        )
    return evidence


def _fill_totals(rows: list[Mapping[str, str]]) -> dict[str, str]:
    net_quantity = Decimal("0")
    eur_cash_delta = Decimal("0")
    for row in rows:
        quantity = _decimal(row.get("quantity"))
        price = _decimal(row.get("price_eur"))
        fee = _decimal(row.get("fee")) if row.get("fee_currency") == "EUR" else Decimal("0")
        if row.get("side") == "BUY":
            net_quantity += quantity
            eur_cash_delta -= quantity * price + fee
        elif row.get("side") == "SELL":
            net_quantity -= quantity
            eur_cash_delta += quantity * price - fee
    return {
        "net_quantity": str(net_quantity),
        "eur_cash_delta": str(eur_cash_delta),
    }


def _managed_protection(settings: Settings, market: str) -> dict[str, Any]:
    for filename in (
        "generated_strategy_live_state.json",
        "event_driven_execution_state.json",
    ):
        path = settings.paths.output_dir / "live" / filename
        if not path.is_file():
            continue
        payload = dict(read_json(path))
        positions = payload.get("positions") or {}
        candidates = positions.values() if isinstance(positions, dict) else positions
        for position in candidates:
            if not isinstance(position, Mapping):
                continue
            if (
                str(position.get("market") or "").upper() == market
                and str(position.get("status") or "").upper() == "OPEN"
            ):
                return {
                    "state_artifact": str(path),
                    "managed_quantity": str(_decimal(position.get("quantity"))),
                    "native_protective_stop_active": bool(
                        position.get("native_protective_stop_active")
                    ),
                    "native_protective_stop_status": position.get(
                        "protective_stop_status"
                    ),
                    "native_protective_stop_trigger": position.get(
                        "protective_stop_trigger"
                    ),
                }
    return {
        "managed_quantity": "0",
        "native_protective_stop_active": False,
        "native_protective_stop_status": None,
        "native_protective_stop_trigger": None,
    }


def build_external_inventory_remediation(
    settings: Settings,
    account_health: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed decision artifact; never submit or generate orders."""

    classifications = (
        (account_health.get("account") or {})
        .get("portfolio_heat", {})
        .get("inventory_classification", [])
    )
    valuations = {
        str(row.get("market") or "").upper(): row
        for row in (
            (account_health.get("account") or {})
            .get("portfolio_valuation", {})
            .get("holdings", [])
        )
    }
    external_rows = [
        row for row in classifications if _decimal(row.get("external_quantity")) > 0
    ]
    markets = {str(row.get("market") or "").upper() for row in external_rows}
    fill_evidence = _external_fill_evidence(
        settings.paths.output_dir / "live" / "events" / "fills.jsonl",
        markets,
    )
    positions: list[dict[str, Any]] = []
    for row in external_rows:
        market = str(row.get("market") or "").upper()
        quantity = _decimal(row.get("external_quantity"))
        valuation = valuations.get(market, {})
        price = _decimal(valuation.get("price_eur"))
        value = quantity * price
        if value < Decimal("1"):
            continue
        evidence = fill_evidence.get(market, [])
        totals = _fill_totals(evidence)
        managed = _managed_protection(settings, market)
        positions.append(
            {
                "market": market,
                "classification": row.get("classification"),
                "total_quantity": row.get("total_quantity"),
                "managed_quantity": row.get("managed_quantity"),
                "external_quantity": str(quantity),
                "external_estimated_value_eur": str(value),
                "mark_price_eur": str(price),
                "managed_protection": managed,
                "protection_scope": "CANONICAL_MANAGED_QUANTITY_ONLY",
                "external_quantity_has_bot_exit_authority": False,
                "external_fill_evidence": evidence,
                "external_fill_totals": totals,
                "fill_evidence_matches_external_quantity": (
                    abs(_decimal(totals["net_quantity"]) - quantity)
                    <= Decimal("0.00000001")
                ),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": "OPERATOR_DECISION_REQUIRED" if positions else "NO_MATERIAL_EXTERNAL_INVENTORY",
        "entry_authority_changed": False,
        "external_exit_authority_granted": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "positions": positions,
        "allowed_decisions": [
            {
                "decision": "KEEP_MANUAL_AND_UNMANAGED",
                "effect": "Bot keeps counting wallet heat but never exits or protects the external quantity.",
                "automatic_action": False,
            },
            {
                "decision": "RETURN_TO_CANONICAL_BASELINE_OUTSIDE_BOT",
                "effect": "Operator sells or transfers only the external quantity outside this bot; rerun account health afterward.",
                "automatic_action": False,
            },
            {
                "decision": "AUTHORIZE_MANAGED_MIGRATION",
                "effect": "Requires a separate explicit approval contract with quantity, strategy, stop, and maximum loss before implementation.",
                "automatic_action": False,
                "implementation_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
            },
        ],
        "safety": {
            "new_entries_remain_blocked": bool(positions),
            "managed_native_protection_may_continue": bool(
                account_health.get("managed_position_protection_eligible")
            ),
            "external_inventory_actions_allowed": False,
            "withdrawals_attempted": 0,
        },
    }
    path = settings.paths.output_dir / "operations" / "external_inventory_remediation.json"
    atomic_write_json(path, payload)
    return {**payload, "artifact": str(path)}


def _migration_contract_hash(contract: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            key: value
            for key, value in contract.items()
            if key not in {"contract_hash", "verification", "artifact", "status"}
        }
    )


def verify_external_inventory_migration_contract(
    contract: Mapping[str, Any],
    remediation: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify migration terms without granting authority or creating orders."""

    failures: list[str] = []
    if contract.get("schema_version") != MIGRATION_SCHEMA_VERSION:
        failures.append("MIGRATION_SCHEMA_INVALID")
    if contract.get("decision") != "AUTHORIZE_MANAGED_MIGRATION":
        failures.append("EXPLICIT_MIGRATION_DECISION_MISSING")
    market = str(contract.get("market") or "").upper()
    current = next(
        (
            row
            for row in remediation.get("positions", [])
            if str(row.get("market") or "").upper() == market
        ),
        None,
    )
    if current is None:
        failures.append("CURRENT_EXTERNAL_INVENTORY_NOT_FOUND")
    else:
        if _decimal(contract.get("external_quantity")) != _decimal(
            current.get("external_quantity")
        ):
            failures.append("EXTERNAL_QUANTITY_CHANGED")
        if _decimal(contract.get("managed_quantity_excluded")) != _decimal(
            current.get("managed_quantity")
        ):
            failures.append("MANAGED_QUANTITY_SCOPE_CHANGED")
        if not current.get("fill_evidence_matches_external_quantity"):
            failures.append("EXTERNAL_FILL_EVIDENCE_MISMATCH")
        if contract.get("remediation_snapshot_hash") != stable_hash(
            {
                "market": market,
                "external_quantity": current.get("external_quantity"),
                "managed_quantity": current.get("managed_quantity"),
                "external_fill_totals": current.get("external_fill_totals"),
            }
        ):
            failures.append("REMEDIATION_SNAPSHOT_CHANGED")

    for field in ("strategy_id", "strategy_dna_hash", "approval_reference"):
        if not str(contract.get(field) or "").strip():
            failures.append(f"{field.upper()}_MISSING")
    mark = _decimal(contract.get("mark_price_eur"))
    stop = _decimal(contract.get("protective_stop_trigger_eur"))
    maximum_loss = _decimal(contract.get("maximum_loss_eur"))
    quantity = _decimal(contract.get("external_quantity"))
    if stop <= 0 or mark <= 0 or stop >= mark:
        failures.append("PROTECTIVE_STOP_INVALID_FOR_SPOT_LONG")
    estimated_stop_loss = max(Decimal("0"), quantity * (mark - stop))
    if maximum_loss <= 0:
        failures.append("MAXIMUM_LOSS_INVALID")
    elif estimated_stop_loss > maximum_loss:
        failures.append("STOP_RISK_EXCEEDS_MAXIMUM_LOSS")
    try:
        expires_at = datetime.fromisoformat(
            str(contract.get("expires_at") or "").replace("Z", "+00:00")
        )
        if expires_at <= utc_now():
            failures.append("MIGRATION_CONTRACT_EXPIRED")
    except (TypeError, ValueError):
        failures.append("MIGRATION_EXPIRY_INVALID")
    acknowledgements = dict(contract.get("operator_acknowledgements") or {})
    for key in (
        "external_quantity_becomes_bot_managed",
        "managed_quantity_is_excluded",
        "protective_stop_may_realize_loss",
        "migration_does_not_enable_new_entries",
    ):
        if acknowledgements.get(key) is not True:
            failures.append(f"ACKNOWLEDGEMENT_MISSING:{key}")
    if contract.get("contract_hash") != _migration_contract_hash(contract):
        failures.append("CONTRACT_HASH_INVALID")
    return {
        "status": "VERIFIED_ORDERLESS" if not failures else "DRAFT_OPERATOR_INPUT_REQUIRED",
        "failures": failures,
        "estimated_stop_loss_eur": str(estimated_stop_loss),
        "authority_granted": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_external_inventory_migration_contract(
    settings: Settings,
    remediation: Mapping[str, Any],
    *,
    market: str,
    terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sealed migration draft tied to the current external quantity."""

    selected_market = market.upper()
    current = next(
        (
            row
            for row in remediation.get("positions", [])
            if str(row.get("market") or "").upper() == selected_market
        ),
        None,
    )
    if current is None:
        raise ValueError("CURRENT_EXTERNAL_INVENTORY_NOT_FOUND")
    supplied = dict(terms or {})
    snapshot_hash = stable_hash(
        {
            "market": selected_market,
            "external_quantity": current.get("external_quantity"),
            "managed_quantity": current.get("managed_quantity"),
            "external_fill_totals": current.get("external_fill_totals"),
        }
    )
    contract: dict[str, Any] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "created_at": utc_iso(),
        "market": selected_market,
        "decision": supplied.get("decision"),
        "external_quantity": current.get("external_quantity"),
        "managed_quantity_excluded": current.get("managed_quantity"),
        "mark_price_eur": current.get("mark_price_eur"),
        "remediation_snapshot_hash": snapshot_hash,
        "strategy_id": supplied.get("strategy_id"),
        "strategy_dna_hash": supplied.get("strategy_dna_hash"),
        "protective_stop_trigger_eur": supplied.get("protective_stop_trigger_eur"),
        "maximum_loss_eur": supplied.get("maximum_loss_eur"),
        "expires_at": supplied.get("expires_at"),
        "approval_reference": supplied.get("approval_reference"),
        "operator_acknowledgements": dict(
            supplied.get("operator_acknowledgements") or {}
        ),
        "scope": "EXACT_EXTERNAL_QUANTITY_ONLY",
        "entry_authority_changed": False,
        "external_exit_authority_granted": False,
        "activation_implemented": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    contract["contract_hash"] = _migration_contract_hash(contract)
    contract["verification"] = verify_external_inventory_migration_contract(
        contract, remediation
    )
    contract["status"] = contract["verification"]["status"]
    path = (
        settings.paths.output_dir
        / "operations"
        / f"external_inventory_migration_contract_{selected_market.replace('-', '_')}.json"
    )
    atomic_write_json(path, contract)
    return {**contract, "artifact": str(path)}


__all__ = [
    "build_external_inventory_migration_contract",
    "build_external_inventory_remediation",
    "verify_external_inventory_migration_contract",
]
