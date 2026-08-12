"""Capability-gated €10 live canaries for approved event-driven playbooks."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import aiohttp

from config.settings import Settings
from core.contracts import (
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderTimeInForce,
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
)
from core.economics import CanonicalCostModel
from core.event_driven_playbooks import PLAYBOOKS
from core.inventory_risk_override import evaluate_inventory_risk_override
from core.live_capital import (
    APPROVAL_PHRASE as CAPITAL_LEVEL_2_APPROVAL_PHRASE,
)
from core.live_capital import (
    CAPITAL_LEVEL,
    MAXIMUM_MANAGED_POSITIONS,
    MAXIMUM_RISK_PER_TRADE_EUR,
    MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR,
    capital_level_2_capacity,
    managed_live_portfolio,
    submit_level_2_buy_atomically,
)
from core.live_capital import (
    MAXIMUM_NEW_ORDERS_PER_DAY as LEVEL_2_MAXIMUM_NEW_ORDERS_PER_DAY,
)
from core.live_capital import (
    MAXIMUM_ORDER_EUR as LEVEL_2_MAXIMUM_ORDER_EUR,
)
from execution.execution import (
    LivePreflight,
    build_live_client,
    minimum_protectable_entry_notional,
    quantity_is_protectable_at_stop,
)
from portfolio.buy_chain import (
    canonicalize_approved_buy_order,
    planned_target_net_edge,
)
from reporting.canonical_economics import canonical_family
from risk.risk_manager import KillSwitch
from utils.common import atomic_write_json, read_json, stable_hash, utc_now

MAXIMUM_ORDER_EUR = LEVEL_2_MAXIMUM_ORDER_EUR
MAXIMUM_TOTAL_EXPOSURE_EUR = MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR
MAXIMUM_OPEN_POSITIONS = MAXIMUM_MANAGED_POSITIONS
MAXIMUM_NEW_ORDERS_PER_DAY = LEVEL_2_MAXIMUM_NEW_ORDERS_PER_DAY
MAKER_WAIT_SECONDS = 2.0
DEFAULT_NEW_FAMILY_EVIDENCE_MULTIPLIER = Decimal("0.40")


def execution_block_reason_code(exc: ExecutionBlocked) -> str:
    """Return a stable, secret-free reason for a blocked live operation."""

    message = str(exc).casefold()
    mappings = (
        ("unsafe_credential_scope", "UNSAFE_CREDENTIAL_SCOPE"),
        ("credentials or operator id are missing", "LIVE_CREDENTIALS_MISSING"),
        ("duplicate live order intent", "DUPLICATE_ORDER_INTENT"),
        ("daily new-order limit reached", "DAILY_NEW_ORDER_LIMIT_REACHED"),
        ("position limit reached", "OPEN_POSITION_LIMIT_REACHED"),
        ("total live canary cap", "TOTAL_EXPOSURE_LIMIT_REACHED"),
        ("explicit live notional cap", "ORDER_NOTIONAL_LIMIT_REACHED"),
        ("below exchange minimum", "EXCHANGE_MINIMUM_NOT_MET"),
        ("sell order exceeds reconciled", "OWNED_QUANTITY_RECONCILIATION_BLOCK"),
        ("total exposure is not reconciled", "EXPOSURE_RECONCILIATION_BLOCK"),
        ("exposure reconciliation is invalid", "EXPOSURE_RECONCILIATION_BLOCK"),
        ("capability is invalid", "LIVE_CAPABILITY_INVALID"),
        ("capability does not include", "LIVE_CAPABILITY_MARKET_MISMATCH"),
        ("rejected order", "VENUE_ORDER_REJECTED"),
        ("rejected cancellation", "VENUE_CANCELLATION_REJECTED"),
        ("market is not trading", "MARKET_NOT_TRADING"),
        ("market execution rules", "MARKET_RULES_BLOCKED"),
        ("live_entry_reservation_busy", "LIVE_ENTRY_RESERVATION_BUSY"),
        ("managed_position_limit_reached", "MANAGED_POSITION_LIMIT_REACHED"),
        ("managed_exposure_limit_reached", "MANAGED_EXPOSURE_LIMIT_REACHED"),
        (
            "maximum risk per trade",
            "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
        ),
        (
            "pending_order_exposure_unreconciled",
            "PENDING_ORDER_EXPOSURE_UNRECONCILED",
        ),
    )
    return next(
        (reason for marker, reason in mappings if marker in message),
        "EXECUTION_POLICY_BLOCKED",
    )


def execution_block_requires_authority_deactivation(
    exc: ExecutionBlocked,
) -> bool:
    """Only credential-scope failures invalidate persisted operator authority."""

    return execution_block_reason_code(exc) in {
        "UNSAFE_CREDENTIAL_SCOPE",
        "LIVE_CREDENTIALS_MISSING",
    }


def _blocked_cycle(
    settings: Settings,
    state: dict[str, Any],
    exc: ExecutionBlocked,
    *,
    phase: str,
    positions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail one cycle closed without erasing unrelated operator approval."""

    if execution_block_requires_authority_deactivation(exc):
        raise exc
    state.update(
        {
            "status": f"{phase}_BLOCKED",
            "reason_code": execution_block_reason_code(exc),
            "authority_deactivation_required": False,
            "positions": dict(positions),
        }
    )
    _save(settings, state)
    return state


def _paths(settings: Settings) -> dict[str, Path]:
    live = settings.paths.output_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    return {
        "authority": settings.paths.project_root
        / "config"
        / "live_playbook_authority.json",
        "state": live / "event_driven_execution_state.json",
        "ledger": settings.paths.checkpoints_dir / "live_execution.jsonl",
        "lifecycle": live / "opportunity_lifecycle_state.json",
    }


def _pending_acknowledged_buys(path: Path) -> list[dict[str, Any]]:
    """Return acknowledged live buys that lack a canonical terminal fill.

    A process can die after Bitvavo acknowledges a resting maker order and
    before its later fill is observed by the synchronous order path.  The
    append-only intent/acknowledgement pair is sufficient to recover that
    order by deterministic client identity on restart.
    """

    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    intents: dict[str, dict[str, Any]] = {}
    filled_order_ids: set[str] = set()
    cancelled_order_ids: set[str] = set()
    for event in events:
        payload = dict(event.get("payload") or {})
        event_type = str(event.get("event_type") or "")
        if event_type == "ORDER_INTENT":
            intent_id = str(payload.get("intent_id") or "")
            if intent_id:
                intents[intent_id] = payload
        elif event_type == "FILL":
            order_id = str(payload.get("order_id") or "")
            if order_id:
                filled_order_ids.add(order_id)
        elif event_type == "ORDER_CANCELLED":
            order_id = str(payload.get("order_id") or "")
            if order_id:
                cancelled_order_ids.add(order_id)
    pending: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "ORDER_ACKNOWLEDGED":
            continue
        acknowledgement = dict(event.get("payload") or {})
        order_id = str(acknowledgement.get("order_id") or "")
        if (
            str(acknowledgement.get("side") or "").upper() != "BUY"
            or not order_id
            or order_id in filled_order_ids
            or order_id in cancelled_order_ids
        ):
            continue
        intent = intents.get(str(acknowledgement.get("intent_id") or ""), {})
        pending.append(
            {
                **intent,
                **acknowledgement,
                "acknowledged_at": event.get("recorded_at"),
            }
        )
    return pending


def _canonical_playbook_counts(path: Path) -> tuple[int, int]:
    orders = 0
    fills = 0
    if not path.is_file():
        return orders, fills
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = dict(event.get("payload") or {})
        if not payload.get("signal_id"):
            continue
        if event.get("event_type") == "ORDER_ACKNOWLEDGED":
            orders += 1
        elif event.get("event_type") == "FILL":
            fills += 1
    return orders, fills


def _signal_fill_summary(path: Path, signal_id: str) -> dict[str, str]:
    buys: list[tuple[Decimal, Decimal, Decimal]] = []
    sells: list[tuple[Decimal, Decimal, Decimal]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = dict(event.get("payload") or {})
            if (
                event.get("event_type") != "FILL"
                or str(payload.get("signal_id") or "") != signal_id
            ):
                continue
            quantity = Decimal(str(payload.get("quantity") or "0"))
            price = Decimal(str(payload.get("price") or "0"))
            fee = Decimal(str(payload.get("fee_eur") or "0"))
            if quantity <= 0 or price <= 0:
                continue
            selected = (quantity, price, fee)
            if str(payload.get("side") or "").upper() == "BUY":
                buys.append(selected)
            elif str(payload.get("side") or "").upper() == "SELL":
                sells.append(selected)
    buy_quantity = sum((row[0] for row in buys), Decimal("0"))
    sell_quantity = sum((row[0] for row in sells), Decimal("0"))
    buy_notional = sum((row[0] * row[1] for row in buys), Decimal("0"))
    sell_notional = sum((row[0] * row[1] for row in sells), Decimal("0"))
    fees = sum((row[2] for row in buys + sells), Decimal("0"))
    return {
        "entry_price": str(
            buy_notional / buy_quantity if buy_quantity > 0 else Decimal("0")
        ),
        "exit_price": str(
            sell_notional / sell_quantity if sell_quantity > 0 else Decimal("0")
        ),
        "buy_quantity": str(buy_quantity),
        "sell_quantity": str(sell_quantity),
        "fees_eur": str(fees),
        "net_pnl_eur": str(sell_notional - buy_notional - fees),
    }


def _lifecycle_opportunity(
    settings: Settings,
    opportunity_id: str,
) -> dict[str, Any]:
    path = _paths(settings)["lifecycle"]
    if not path.is_file():
        return {}
    payload = dict(read_json(path))
    opportunities = payload.get("opportunities")
    if isinstance(opportunities, Mapping):
        return dict(opportunities.get(opportunity_id) or {})
    return dict(payload.get(opportunity_id) or {})


def _order_timestamp(payload: Mapping[str, Any], fallback: datetime) -> datetime:
    raw = payload.get("updated") or payload.get("created")
    try:
        if raw is not None:
            return datetime.fromtimestamp(float(raw) / 1000.0, tz=UTC)
    except (ArithmeticError, TypeError, ValueError, OSError):
        pass
    return fallback


def playbook_catalog() -> list[dict[str, Any]]:
    return [
        {
            "playbook_id": item.playbook_id,
            "family": item.family,
            "playbook_dna": item.dna,
            "execution_timeframes": list(item.execution_timeframes),
            "context_timeframes": list(item.context_timeframes),
            "expected_regimes": list(item.expected_regimes),
            "entry_logic": item.entry_logic,
            "orderflow_confirmation": item.orderflow_confirmation,
            "liquidity_rule": item.liquidity_rule,
            "invalidation": item.invalidation,
            "stop_method": item.stop_method,
            "take_profit_method": item.take_profit_method,
            "trailing_method": item.trailing_method,
            "time_stop": item.time_stop,
            "risk_policy": item.risk_policy,
            "parameter_band": {
                key: list(value) for key, value in item.parameter_band.items()
            },
            "parameter_band_hash": stable_hash(
                item.parameter_band, length=64
            ),
            "allowed_markets": "OPERATOR_PLAYBOOK_AUTHORITY",
        }
        for item in PLAYBOOKS
        if item.family != "ORDERFLOW_EXHAUSTION_EXIT"
    ]


def playbook_authority_status(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["authority"]
    authority = dict(read_json(path)) if path.is_file() else {}
    state_path = _paths(settings)["state"]
    state = dict(read_json(state_path)) if state_path.is_file() else {}
    return {
        "schema_version": "event_driven_playbook_authority_status_v1",
        "active": authority.get("active") is True,
        "approved_playbook_count": len(
            authority.get("approved_playbooks") or []
        ),
        "approved_playbooks": authority.get("approved_playbooks") or [],
        "catalog": playbook_catalog(),
        "maximum_order_eur": authority.get(
            "maximum_order_eur", str(MAXIMUM_ORDER_EUR)
        ),
        "maximum_total_exposure_eur": authority.get(
            "maximum_total_exposure_eur", str(MAXIMUM_TOTAL_EXPOSURE_EUR)
        ),
        "maximum_open_positions": authority.get("maximum_open_positions", 3),
        "maximum_new_orders_per_day": authority.get(
            "maximum_new_orders_per_day", 3
        ),
        "autoscale": False,
        "approval_phrase_stored": False,
        "execution_status": state.get("status") or "NOT_STARTED",
        "managed_position_count": len(state.get("positions") or {}),
        "orders_generated": int(state.get("orders_generated") or 0),
        "orders_submitted": int(state.get("orders_submitted") or 0),
        "fills_verified": int(state.get("fills_verified") or 0),
    }


def approval_phrase(playbook_id: str) -> str:
    selected = next(
        item for item in PLAYBOOKS if item.playbook_id == playbook_id
    )
    return f"LIVE EVENT PLAYBOOK {playbook_id} {selected.dna[:12]} CONFIRMED"


def approve_playbook_live(
    settings: Settings,
    *,
    playbook_id: str,
    markets: Iterable[str],
    approval: str,
    evidence_multiplier: Decimal | float | str = (
        DEFAULT_NEW_FAMILY_EVIDENCE_MULTIPLIER
    ),
) -> dict[str, Any]:
    selected = next(
        (item for item in PLAYBOOKS if item.playbook_id == playbook_id),
        None,
    )
    if selected is None or selected.family == "ORDERFLOW_EXHAUSTION_EXIT":
        raise ValueError("unknown or exit-only playbook")
    if approval.strip() != approval_phrase(playbook_id):
        raise PermissionError("event playbook approval phrase does not match")
    evidence = Decimal(str(evidence_multiplier))
    if evidence < Decimal("0.25") or evidence > Decimal("1"):
        raise ValueError("evidence multiplier must be between 0.25 and 1.00")
    service_path = (
        settings.paths.output_dir / "live" / "autonomous_live_authority.json"
    )
    service = dict(read_json(service_path)) if service_path.is_file() else {}
    service_markets = set(service.get("markets") or [])
    normalized = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-")
            for market in markets
        )
    )
    failures = [
        f"MARKET_NOT_IN_LIVE_SERVICE_AUTHORITY:{market}"
        for market in normalized
        if market not in service_markets
    ]
    failures.extend(
        f"MARKET_NOT_SHARIAH_ALLOWED:{market}"
        for market in normalized
        if settings.shariah.eligibility(market).status.value != "ALLOWED"
    )
    if not normalized:
        failures.append("NO_MARKETS_SELECTED")
    if failures:
        return {
            "status": "BLOCKED",
            "failures": failures,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    path = _paths(settings)["authority"]
    authority = dict(read_json(path)) if path.is_file() else {}
    rows = [
        dict(row)
        for row in authority.get("approved_playbooks") or []
        if row.get("playbook_id") != playbook_id
    ]
    rows.append(
        {
            "playbook_id": playbook_id,
            "family": selected.family,
            "playbook_dna": selected.dna,
            "parameter_band": {
                key: list(value)
                for key, value in selected.parameter_band.items()
            },
            "parameter_band_hash": stable_hash(
                selected.parameter_band, length=64
            ),
            "markets": list(normalized),
            "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
            "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
            "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
            "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
            "autoscale": False,
            "active": True,
            "authority_basis": "OPERATOR_MICRO_LIVE_PLAYBOOK_BAND_APPROVAL",
            "authority_level": "VALIDATED_PLAYBOOK",
            "authority_levels": ["EXACT_STRATEGY", "VALIDATED_PLAYBOOK"],
            "strategy_role": (
                "EXPERIMENTAL_CANARY"
                if evidence < Decimal("0.70")
                else "ACTIVE_ALPHA"
            ),
            "evidence_multiplier": str(evidence),
            "maximum_family_positions": 1,
            "execution_timeframes": list(selected.execution_timeframes),
            "expected_regimes": list(selected.expected_regimes),
            "approved_at": utc_now().isoformat(),
            "approval_phrase_stored": False,
        }
    )
    updated = {
        "schema_version": "event_driven_playbook_authority_v1",
        "active": True,
        "approved_playbooks": rows,
        "capital_level": CAPITAL_LEVEL,
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
        "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
        "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        "autoscale": False,
        "spot_only": True,
        "margin": False,
        "leverage": False,
        "shorting": False,
        "withdrawals": False,
        "approval_phrase_stored": False,
    }
    atomic_write_json(path, updated)
    return {
        "status": "APPROVED",
        "playbook_id": playbook_id,
        "playbook_dna": selected.dna,
        "markets": list(normalized),
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "evidence_multiplier": str(evidence),
        "maximum_effective_order_eur": str(MAXIMUM_ORDER_EUR * evidence),
        "maximum_family_positions": 1,
        "autoscale": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def deactivate_playbook_live(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["authority"]
    authority = dict(read_json(path)) if path.is_file() else {}
    authority["active"] = False
    authority["deactivated_at"] = utc_now().isoformat()
    authority["approval_phrase_stored"] = False
    atomic_write_json(path, authority)
    return playbook_authority_status(settings)


def migrate_playbook_live_capital_level_2(
    settings: Settings,
    *,
    approval_phrase: str,
) -> dict[str, Any]:
    """Raise caps for already approved playbooks without adding a family."""

    if approval_phrase.strip() != CAPITAL_LEVEL_2_APPROVAL_PHRASE:
        raise PermissionError("capital Level-2 approval phrase mismatch")
    path = _paths(settings)["authority"]
    if not path.is_file():
        raise FileNotFoundError("event playbook authority is missing")
    authority = dict(read_json(path))
    if (
        authority.get("active") is not True
        or authority.get("autoscale") is not False
        or authority.get("spot_only") is not True
    ):
        raise PermissionError("existing event playbook authority is invalid")
    portfolio = managed_live_portfolio(settings)
    if (
        int(portfolio["managed_position_count"]) > MAXIMUM_OPEN_POSITIONS
        or Decimal(portfolio["managed_exposure_eur"])
        > MAXIMUM_TOTAL_EXPOSURE_EUR
    ):
        raise PermissionError("current managed portfolio exceeds Level-2 caps")
    catalog = {row["playbook_id"]: row for row in playbook_catalog()}
    rows = []
    identity_migrations: list[dict[str, str]] = []
    for raw in authority.get("approved_playbooks") or []:
        row = dict(raw)
        current = catalog.get(str(row.get("playbook_id") or ""))
        if current is not None:
            same_family = row.get("family") == current.get("family")
            same_band = row.get("parameter_band_hash") == current.get(
                "parameter_band_hash"
            )
            old_dna = str(row.get("playbook_dna") or "")
            new_dna = str(current.get("playbook_dna") or "")
            if same_family and same_band and old_dna != new_dna:
                row.update(
                    {
                        "playbook_dna": new_dna,
                        "execution_timeframes": list(
                            current.get("execution_timeframes") or []
                        ),
                        "expected_regimes": list(
                            current.get("expected_regimes") or []
                        ),
                        "previous_playbook_dna": old_dna,
                        "identity_migration_reason": (
                            "CATALOG_METADATA_CHANGED_PARAMETER_BAND_UNCHANGED"
                        ),
                        "identity_migrated_at": utc_now().isoformat(),
                    }
                )
                identity_migrations.append(
                    {
                        "playbook_id": str(row["playbook_id"]),
                        "previous_playbook_dna": old_dna,
                        "playbook_dna": new_dna,
                    }
                )
        row.update(
            {
                "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
                "maximum_total_exposure_eur": str(
                    MAXIMUM_TOTAL_EXPOSURE_EUR
                ),
                "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
                "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
                "autoscale": False,
            }
        )
        rows.append(row)
    authority.update(
        {
            "approved_playbooks": rows,
            "capital_level": CAPITAL_LEVEL,
            "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
            "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
            "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
            "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
            "maximum_risk_per_trade_eur": "2",
            "autoscale": False,
            "last_capital_level_migration_at": utc_now().isoformat(),
            "approval_phrase_stored": False,
        }
    )
    atomic_write_json(path, authority)
    return {
        "status": "CAPITAL_LEVEL_2_ACTIVE",
        "capital_level": CAPITAL_LEVEL,
        "approved_playbook_count": len(rows),
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
        "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
        "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        "maximum_risk_per_trade_eur": "2",
        "autoscale": False,
        "approved_playbooks_unchanged": True,
        "identity_migration_count": len(identity_migrations),
        "identity_migrations": identity_migrations,
        "managed_portfolio": portfolio,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _state(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["state"]
    if path.is_file():
        return dict(read_json(path))
    return {
        "schema_version": "event_driven_execution_state_v1",
        "status": "READY",
        "positions": {},
        "orders_generated": 0,
        "orders_submitted": 0,
        "fills_verified": 0,
        "autoscale": False,
    }


def _save(settings: Settings, state: Mapping[str, Any]) -> None:
    atomic_write_json(_paths(settings)["state"], dict(state))


def _observed_facts_within_band(
    band: Mapping[str, Any],
    opportunity: Mapping[str, Any],
) -> bool:
    observed = dict(opportunity.get("validated_parameters") or {})
    score = observed.get("score", opportunity.get("score"))
    if "score" in band:
        try:
            lower, upper = (float(value) for value in band["score"])
            if score is None or not lower <= float(score) <= upper:
                return False
        except (TypeError, ValueError):
            return False
    for key in ("confirmations", "flow_confirmations", "reclaim_confirmations"):
        if key not in band:
            continue
        value = observed.get(key, opportunity.get("confirmation_count"))
        try:
            lower = float(band[key][0])
            if value is None or float(value) < lower:
                return False
        except (TypeError, ValueError):
            return False
        break
    return True


def _variant_within_approved_band(
    row: Mapping[str, Any],
    opportunity: Mapping[str, Any],
) -> bool:
    """Verify a variant from immutable parameters, never a claimed status."""

    if row.get("authority_level") != "VALIDATED_PLAYBOOK":
        return False
    parameters = dict(opportunity.get("playbook_parameters") or {})
    band = dict(row.get("parameter_band") or {})
    if not parameters or set(parameters) != set(band):
        return False
    try:
        if any(
            not float(bounds[0]) <= float(parameters[key]) <= float(bounds[1])
            for key, bounds in band.items()
        ):
            return False
    except (IndexError, TypeError, ValueError):
        return False
    base_dna = str(opportunity.get("base_playbook_dna") or "")
    if base_dna != str(row.get("playbook_dna") or ""):
        return False
    expected_dna = stable_hash(
        {
            "base_playbook_dna": base_dna,
            "family": opportunity.get("family"),
            "playbook_id": opportunity.get("playbook_id"),
            "playbook_parameters": parameters,
        },
        length=64,
    )
    return expected_dna == str(opportunity.get("playbook_dna") or "")


def is_playbook_opportunity_authorized(
    authority: Mapping[str, Any],
    opportunity: Mapping[str, Any],
) -> bool:
    return _authorized_playbook_row(authority, opportunity) is not None


def _authorized_playbook_row(
    authority: Mapping[str, Any],
    opportunity: Mapping[str, Any],
) -> dict[str, Any] | None:
    for raw in authority.get("approved_playbooks") or []:
        row = dict(raw)
        band = dict(row.get("parameter_band") or {})
        if (
            row.get("active") is not True
            or opportunity.get("market") not in set(row.get("markets") or [])
            or row.get("family") != opportunity.get("family")
            or Decimal(str(row.get("maximum_order_eur") or 0))
            > MAXIMUM_ORDER_EUR
            or row.get("autoscale") is not False
            or stable_hash(band, length=64)
            != str(row.get("parameter_band_hash") or "")
            or not _observed_facts_within_band(band, opportunity)
        ):
            continue
        exact = (
            row.get("playbook_dna") == opportunity.get("playbook_dna")
            and row.get("parameter_band_hash")
            == opportunity.get("parameter_band_hash")
        )
        if exact or _variant_within_approved_band(row, opportunity):
            return row
    return None


def _authorized(
    authority: Mapping[str, Any],
    opportunity: Mapping[str, Any],
) -> bool:
    return is_playbook_opportunity_authorized(authority, opportunity)


def _bounded_fraction(value: object, *, default: float = 0.0) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        selected = default
    return max(0.0, min(1.0, selected))


def _authority_evidence_multiplier(row: Mapping[str, Any]) -> Decimal:
    """Return explicit evidence authority; legacy approvals retain Level 2."""

    try:
        selected = Decimal(str(row.get("evidence_multiplier", "1")))
    except Exception:
        selected = Decimal("0")
    return max(Decimal("0.25"), min(Decimal("1"), selected))


def _candidate_selection_score(
    opportunity: Mapping[str, Any],
    authority_row: Mapping[str, Any],
) -> float:
    """Rank competing valid entries without creating an extra execution gate."""

    economics = dict(opportunity.get("execution_economics") or {})
    scorecard = dict(opportunity.get("execution_scorecard") or {})
    setup = _bounded_fraction(float(opportunity.get("score") or 0.0) / 100.0)
    ev = _bounded_fraction(
        float(economics.get("expected_net_value_bps") or 0.0) / 100.0
    )
    cost_ratio = float(economics.get("cost_to_target_2_ratio") or 1.0)
    ecr = (1.0 / cost_ratio) if cost_ratio > 0 else 0.0
    ecr_quality = _bounded_fraction(ecr / 4.0)
    evidence = float(_authority_evidence_multiplier(authority_row))
    mtf_raw = opportunity.get("weighted_timeframe_score")
    mtf = _bounded_fraction(mtf_raw, default=setup)
    micro = {
        "SUPPORTIVE": 1.0,
        "NEUTRAL": 0.5,
        "HOSTILE": 0.0,
    }.get(str(opportunity.get("microstructure_state") or "").upper(), 0.5)
    liquidity = _bounded_fraction(
        float(scorecard.get("friction_liquidity") or 0.0) / 5.0
    )
    return round(
        0.25 * setup
        + 0.25 * ev
        + 0.15 * ecr_quality
        + 0.15 * evidence
        + 0.10 * mtf
        + 0.05 * micro
        + 0.05 * liquidity,
        8,
    )


def _entry_rejection_details(
    authority: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    *,
    positions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Explain why currently entry-ready facts did not become an order.

    This is deliberately based only on the opportunities supplied to the
    current realtime cycle.  Persisted historical lifecycle rows are not
    revived and cannot masquerade as a fresh executable signal.
    """

    managed = positions or {}
    details: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if row.get("state") != "ENTRY_READY":
            continue
        reasons: list[str] = []
        blockers = [str(value) for value in row.get("hard_blockers") or []]
        if blockers:
            reasons.append("HARD_BLOCKERS_PRESENT")
        if not _authorized(authority, row):
            reasons.append("PLAYBOOK_OR_MARKET_NOT_AUTHORIZED")
        identity = str(row.get("opportunity_id") or "")
        if identity and identity in managed:
            reasons.append("OPPORTUNITY_ALREADY_MANAGED")
        if not reasons:
            reasons.append("QUALIFIED_IN_CURRENT_CYCLE")
        details.append(
            {
                "opportunity_id": identity,
                "market": row.get("market"),
                "playbook_id": row.get("playbook_id"),
                "tier": row.get("tier"),
                "score": row.get("score"),
                "reasons": reasons,
                "hard_blockers": blockers,
            }
        )
    return sorted(
        details,
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )[:50]


def _balance(
    balances: Iterable[Mapping[str, Any]],
    symbol: str,
    *,
    include_in_order: bool = False,
) -> Decimal:
    row = next(
        (
            item
            for item in balances
            if str(item.get("symbol") or "").upper() == symbol.upper()
        ),
        {},
    )
    available = Decimal(str(row.get("available") or "0"))
    if include_in_order:
        available += Decimal(
            str(row.get("inOrder") or row.get("in_order") or "0")
        )
    return available


def _risk_limited_entry_notional(
    *,
    desired_notional_eur: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
    maximum_risk_eur: Decimal = MAXIMUM_RISK_PER_TRADE_EUR,
) -> Decimal:
    """Cap notional so loss at the structural stop stays inside authority.

    Venue-minimum adjustment happens afterwards.  When the smallest safely
    protectable order is already too risky, the caller must block the entry
    rather than silently violating either invariant.
    """

    if (
        desired_notional_eur <= 0
        or entry_price <= 0
        or stop_price <= 0
        or stop_price >= entry_price
        or maximum_risk_eur <= 0
    ):
        return Decimal("0")
    stop_fraction = (entry_price - stop_price) / entry_price
    return min(desired_notional_eur, maximum_risk_eur / stop_fraction)


def _managed_exposure(positions: Mapping[str, Mapping[str, Any]]) -> Decimal:
    return sum(
        (
            Decimal(str(row.get("quantity") or "0"))
            * Decimal(str(row.get("entry_price") or "0"))
            for row in positions.values()
            if str(row.get("status") or "").upper()
            in {"ENTRY_PENDING", "OPEN", "MANAGING"}
        ),
        Decimal("0"),
    )


def _wallet_exposure(
    settings: Settings,
    balances: Iterable[Mapping[str, Any]],
    realtime_prices: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Value all wallet inventory separately from strategy-owned positions.

    A holding does not stop being portfolio risk merely because the current
    strategy did not open it.  Live prices are preferred and the sanitized
    account-health valuation is a fallback for dust or temporarily untracked
    markets.  Unknown positive holdings fail new entries closed.
    """

    fallback_prices: dict[str, Decimal] = {}
    health_path = (
        settings.paths.output_dir / "operations" / "live_account_health.json"
    )
    if health_path.is_file():
        health = dict(read_json(health_path))
        valuation = dict((health.get("account") or {}).get("portfolio_valuation") or {})
        for row in valuation.get("holdings") or []:
            symbol = str(row.get("symbol") or "").upper()
            try:
                price = Decimal(str(row.get("price_eur") or "0"))
            except (ArithmeticError, ValueError):
                continue
            if symbol and price > 0:
                fallback_prices[symbol] = price

    values: dict[str, Decimal] = {}
    unknown: list[str] = []
    eur_cash = Decimal("0")
    for row in balances:
        symbol = str(row.get("symbol") or "").upper()
        quantity = Decimal(str(row.get("available") or "0")) + Decimal(
            str(row.get("inOrder") or "0")
        )
        if quantity <= 0:
            continue
        if symbol == "EUR":
            eur_cash = quantity
            continue
        market = f"{symbol}-EUR"
        live = realtime_prices.get(market) or {}
        try:
            price = Decimal(str(live.get("price") or "0"))
        except (ArithmeticError, ValueError):
            price = Decimal("0")
        if price <= 0:
            price = fallback_prices.get(symbol, Decimal("0"))
        if price <= 0:
            unknown.append(symbol)
            continue
        values[symbol] = quantity * price

    asset_exposure = sum(values.values(), Decimal("0"))
    equity = eur_cash + asset_exposure
    largest_symbol = max(values, key=values.get) if values else None
    largest_value = values.get(largest_symbol, Decimal("0"))
    concentration = largest_value / equity if equity > 0 else Decimal("0")
    return {
        "status": "READY" if not unknown else "VALUATION_INCOMPLETE",
        "total_wallet_asset_exposure_eur": str(asset_exposure),
        "estimated_wallet_equity_eur": str(equity),
        "wallet_cash_eur": str(eur_cash),
        "largest_asset": largest_symbol,
        "largest_asset_exposure_eur": str(largest_value),
        "wallet_concentration_fraction": str(concentration),
        "unknown_positive_holdings": sorted(unknown),
        "holdings_eur": {
            symbol: str(value) for symbol, value in sorted(values.items())
        },
    }


def _microstructure_exit_reason(
    position: Mapping[str, Any],
    realtime: Mapping[str, Any],
    *,
    btc_realtime: Mapping[str, Any] | None = None,
    matching_opportunity: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the first material causal deterioration signal for a long."""

    # Microstructure may confirm a soft exit, but stale or unsequenced data
    # must never manufacture one.  Hard stops and targets are evaluated on a
    # separate deterministic path and remain active during a stream outage.
    if realtime.get("fresh") is not True:
        return None
    if realtime.get("sequence_valid") is not True:
        return None

    windows = dict(realtime.get("windows") or {})
    one = dict(windows.get("1m") or {})
    five = dict(windows.get("5m") or {})
    book = dict(realtime.get("book") or {})
    matched = dict((matching_opportunity or {}).get("realtime_inputs") or {})

    def number(*values: Any) -> float | None:
        for value in values:
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    buy_ratio = number(one.get("taker_buy_ratio"), matched.get("taker_buy_ratio_1m"))
    cvd = number(one.get("cvd_quote_eur"), matched.get("cvd_quote_eur_1m"))
    ofi = number(realtime.get("ofi_1m"), matched.get("ofi_1m"))
    mlobi = number(book.get("mlobi_top_10"), matched.get("mlobi_top_10"))
    bid_depth = number(
        book.get("bid_depth_eur_top_10"), matched.get("bid_depth_eur_top_10")
    )
    spread = number(book.get("spread_bps"), matched.get("spread_bps"))
    return_1m = number(one.get("return"), matched.get("return_1m"))
    return_5m = number(five.get("return"), matched.get("return_5m"))
    btc_return_5m = number(
        (((btc_realtime or {}).get("windows") or {}).get("5m") or {}).get("return")
    )
    relative_5m = (
        return_5m - btc_return_5m
        if return_5m is not None and btc_return_5m is not None
        else None
    )
    baseline_depth = number(position.get("entry_bid_depth_eur_top_10"))
    baseline_spread = number(position.get("entry_spread_bps"))

    if buy_ratio is not None and ofi is not None and buy_ratio < 0.45 and ofi < 0:
        return "ORDERFLOW_EXHAUSTION"
    if cvd is not None and buy_ratio is not None and cvd < 0 and buy_ratio < 0.48:
        return "NEGATIVE_CVD_REVERSAL"
    if (
        mlobi is not None
        and bid_depth is not None
        and baseline_depth is not None
        and baseline_depth > 0
        and mlobi < -0.15
        and bid_depth < baseline_depth * 0.60
    ):
        return "BID_SUPPORT_WITHDRAWAL"
    if (
        spread is not None
        and baseline_spread is not None
        and spread > max(20.0, baseline_spread * 2.5)
    ):
        return "SPREAD_EXPANSION"
    if (
        return_1m is not None
        and return_5m is not None
        and return_1m < -0.003
        and return_5m < -0.006
    ):
        return "MOMENTUM_DECAY"
    if relative_5m is not None and relative_5m < -0.008:
        return "RELATIVE_STRENGTH_DETERIORATION"
    return None


def _soft_exit_confirmed(
    position: dict[str, Any],
    *,
    reason: str | None,
    realtime: Mapping[str, Any],
    price: Decimal,
    now: datetime,
    minimum_observations: int | None = None,
    minimum_seconds: float | None = None,
) -> bool:
    """Require persistence and adverse price confirmation for soft exits.

    This never applies to a hard stop, target or time stop.  State is kept in
    the restart-safe position projection, so a transient order-flow flip
    cannot close a canary while a sustained deterioration still can.
    """

    windows = dict(realtime.get("windows") or {})
    one = dict(windows.get("1m") or {})
    book = dict(realtime.get("book") or {})

    def number(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    event_rate = number(
        one.get("trade_intensity")
        or realtime.get("trade_intensity_1m")
    )
    spread = number(book.get("spread_bps") or realtime.get("spread_bps"))
    bid_depth = number(
        book.get("bid_depth_eur_top_10")
        or realtime.get("bid_depth_eur_top_10")
    )
    context_timeframe = str(
        position.get("context_timeframe") or "15m"
    ).casefold()
    adaptive_seconds = 10.0
    if context_timeframe in {"4h", "1d", "1w"}:
        adaptive_seconds += 5.0
    if (spread is not None and spread > 15.0) or (
        bid_depth is not None and bid_depth < 250.0
    ):
        adaptive_seconds += 5.0
    if event_rate is not None and event_rate >= 2.0:
        adaptive_seconds -= 5.0
    required_seconds = max(
        10.0,
        min(20.0, adaptive_seconds if minimum_seconds is None else minimum_seconds),
    )
    required_observations = max(
        3,
        int(
            (4 if event_rate is not None and event_rate >= 3.0 else 3)
            if minimum_observations is None
            else minimum_observations
        ),
    )
    position["soft_exit_required_observations"] = required_observations
    position["soft_exit_required_seconds"] = required_seconds

    if not reason:
        position.pop("soft_exit_candidate", None)
        position.pop("soft_exit_candidate_since", None)
        position.pop("soft_exit_observations", None)
        return False
    if position.get("soft_exit_candidate") != reason:
        position["soft_exit_candidate"] = reason
        position["soft_exit_candidate_since"] = now.astimezone(UTC).isoformat()
        position["soft_exit_observations"] = 1
        return False
    position["soft_exit_observations"] = int(
        position.get("soft_exit_observations") or 0
    ) + 1
    try:
        since = datetime.fromisoformat(
            str(position["soft_exit_candidate_since"]).replace("Z", "+00:00")
        ).astimezone(UTC)
    except (KeyError, TypeError, ValueError):
        since = now
        position["soft_exit_candidate_since"] = now.astimezone(UTC).isoformat()
    try:
        return_1m = float(one.get("return"))
    except (TypeError, ValueError):
        return_1m = 0.0
    entry_price = Decimal(str(position.get("entry_price") or price))
    price_confirmed = price < entry_price or return_1m < 0
    return (
        int(position["soft_exit_observations"]) >= required_observations
        and (now - since).total_seconds() >= required_seconds
        and price_confirmed
    )


def _live_capability(
    settings: Settings,
    *,
    markets: tuple[str, ...],
    reconciliation_healthy: bool,
) -> Any:
    return LivePreflight.evaluate(
        settings,
        markets=markets,
        strategy_status=ResearchStatus.PAPER_CANDIDATE,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=reconciliation_healthy,
        kill_switch_active=KillSwitch(
            settings.paths.checkpoints_dir / "kill_switch.json"
        ).active,
        canary_exception_approved=True,
        operator_canary_authorized=True,
        portfolio_canary=True,
        cap_limits={
            "capital_level": CAPITAL_LEVEL,
            "max_order_eur": str(MAXIMUM_ORDER_EUR),
            "max_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
            "max_positions": MAXIMUM_OPEN_POSITIONS,
            "max_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        },
    )


def _entry_intent(
    opportunity: Mapping[str, Any],
    *,
    quantity: Decimal,
    limit_price: Decimal | None,
    attempt: int,
    post_only: bool,
    order_type: OrderType = OrderType.LIMIT,
    time_in_force: OrderTimeInForce | None = None,
) -> OrderIntent:
    identity = stable_hash(
        [
            "EVENT_DRIVEN_LIVE_ENTRY",
            opportunity["opportunity_id"],
            opportunity["playbook_dna"],
            attempt,
        ],
        length=40,
    )
    cancel_group = stable_hash(
        ["EVENT_ENTRY_COD", opportunity["opportunity_id"]], length=32
    )
    return OrderIntent(
        intent_id=identity[:32],
        idempotency_key=f"event-live-entry:{identity}",
        market=str(opportunity["market"]),
        side=OrderSide.BUY,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        # A passive maker attempt may rest briefly, but the urgent fallback
        # must be terminal.  IOC prevents an untracked marketable-limit order
        # from remaining open after this cycle/restart.
        time_in_force=time_in_force
        or (
            OrderTimeInForce.GTC
            if order_type is OrderType.MARKET or post_only
            else OrderTimeInForce.IOC
        ),
        post_only=post_only if order_type is OrderType.LIMIT else False,
        strategy_id=str(opportunity["playbook_id"]),
        strategy_dna_hash=str(opportunity["playbook_dna"]),
        signal_id=str(opportunity["opportunity_id"]),
        portfolio_decision_id=identity,
        cancel_on_disconnect_group=cancel_group,
        maximum_notional_eur=MAXIMUM_ORDER_EUR,
        reason_codes=(
            "EVENT_DRIVEN_PLAYBOOK_BAND_AUTHORITY",
            "QUOTE_MARKET_LIQUIDITY_PREFLIGHT"
            if order_type is OrderType.MARKET
            else "MAKER_FIRST"
            if post_only
            else "BOUNDED_MARKETABLE_LIMIT",
            f"ENTRY_ATTEMPT_{attempt}",
        ),
    )


def _exit_intent(
    position: Mapping[str, Any],
    *,
    quantity: Decimal,
    limit_price: Decimal,
    reason: str,
) -> OrderIntent:
    identity = stable_hash(
        [
            "EVENT_DRIVEN_LIVE_EXIT",
            position["opportunity_id"],
            reason,
            position.get("exit_attempt", 0),
        ],
        length=40,
    )
    return OrderIntent(
        intent_id=identity[:32],
        idempotency_key=f"event-live-exit:{identity}",
        market=str(position["market"]),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        time_in_force=OrderTimeInForce.IOC,
        post_only=False,
        strategy_id=str(position["playbook_id"]),
        strategy_dna_hash=str(position["playbook_dna"]),
        signal_id=str(position["opportunity_id"]),
        portfolio_decision_id=identity,
        reason_codes=(reason, "BOUNDED_MARKETABLE_EXIT"),
    )


def _protective_stop_intent(
    position: Mapping[str, Any],
    *,
    quantity: Decimal,
    trigger_price: Decimal,
) -> OrderIntent:
    """Create one durable venue-native stop for a verified spot fill."""

    identity = stable_hash(
        [
            "EVENT_DRIVEN_NATIVE_PROTECTIVE_STOP",
            position["opportunity_id"],
            position["playbook_dna"],
            str(quantity),
            str(trigger_price),
        ],
        length=40,
    )
    return OrderIntent(
        intent_id=identity[:32],
        idempotency_key=f"event-live-protective-stop:{identity}",
        market=str(position["market"]),
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=quantity,
        trigger_price=trigger_price,
        trigger_reference="bestBid",
        strategy_id=str(position["playbook_id"]),
        strategy_dna_hash=str(position["playbook_dna"]),
        signal_id=str(position["opportunity_id"]),
        portfolio_decision_id=identity,
        reason_codes=(
            "NATIVE_PROTECTIVE_STOP",
            "LOCAL_HARD_STOP_REMAINS_ACTIVE",
        ),
    )


def _supports_order_type(
    supported_order_types: Iterable[str],
    expected: str,
) -> bool:
    """Compare venue order-type metadata without casing assumptions."""

    normalized = {
        str(value).replace("_", "").replace("-", "").casefold()
        for value in supported_order_types
    }
    return expected.replace("_", "").replace("-", "").casefold() in normalized


async def _replace_native_protective_stop(
    client: Any,
    *,
    capability: Any,
    position: Mapping[str, Any],
    quantity: Decimal,
    trigger_price: Decimal,
    estimated_price: Decimal,
) -> dict[str, str]:
    """Atomically reconcile, cancel and replace a tightened native stop."""

    market = str(position["market"])
    current_client_id = str(
        position.get("protective_stop_client_order_id") or ""
    )
    if current_client_id:
        current = await client.get_order(
            market=market,
            client_order_id=current_client_id,
        )
        status = str(current.get("status") or "").replace(
            "_", ""
        ).replace("-", "").casefold()
        if status == "filled":
            raise ReconciliationRequired(
                "protective stop filled while replacement was requested"
            )
        if status in {"new", "awaitingtrigger", "partiallyfilled"}:
            await client.cancel_order(
                market=market,
                order_id=str(current["orderId"]),
                capability=capability,
            )
    balances = await client.balances()
    owned = _balance(
        balances,
        market.split("-")[0],
        include_in_order=True,
    )
    replacement_quantity = min(quantity, owned)
    if replacement_quantity <= 0:
        raise ReconciliationRequired(
            "protective stop replacement lacks owned quantity"
        )
    rules = await client.execution_market_rules(market)
    trigger = rules.price(trigger_price)
    intent = _protective_stop_intent(
        position,
        quantity=replacement_quantity,
        trigger_price=trigger,
    )
    order = await client.submit_order(
        intent,
        capability=capability,
        estimated_price=estimated_price,
        reconciled_owned_quantity=replacement_quantity,
        exchange_minimum_order_eur=rules.minimum_order_value_eur,
    )
    status = str(order.get("status") or "").replace("_", "").replace(
        "-", ""
    ).casefold()
    if status not in {"new", "awaitingtrigger"}:
        raise ReconciliationRequired(
            "replacement protective stop was not accepted"
        )
    return {
        "protective_stop_order_id": str(order["orderId"]),
        "protective_stop_client_order_id": client.client_order_id_for(
            intent.idempotency_key
        ),
        "protective_stop_status": str(order.get("status")),
        "protective_stop_trigger": str(trigger),
        "local_hard_stop_active": True,
    }


async def execute_event_driven_live_once(
    settings: Settings,
    *,
    opportunities: Iterable[Mapping[str, Any]],
    realtime_snapshot: Mapping[str, Any],
    submit: bool,
    allow_new_entry: bool,
    allowed_economics_entry_families: Iterable[str] | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Manage exits, then submit at most one approved natural micro entry."""

    now = (observed_at or utc_now()).astimezone(UTC)
    paths = _paths(settings)
    authority = (
        dict(read_json(paths["authority"]))
        if paths["authority"].is_file()
        else {}
    )
    state = _state(settings)
    canonical_orders, canonical_fills = _canonical_playbook_counts(
        paths["ledger"]
    )
    state.update(
        {
            "last_cycle_at": now.isoformat(),
            "orders_generated_this_cycle": 0,
            "orders_submitted_this_cycle": 0,
            "fills_verified_this_cycle": 0,
            "events": [],
            "orders_generated": max(
                int(state.get("orders_generated") or 0),
                canonical_orders,
            ),
            "orders_submitted": max(
                int(state.get("orders_submitted") or 0),
                canonical_orders,
            ),
            "fills_verified": max(
                int(state.get("fills_verified") or 0),
                canonical_fills,
            ),
        }
    )
    if authority.get("active") is not True:
        state.update(
            {
                "status": "AUTHORITY_DISABLED",
                "reason_code": "EVENT_PLAYBOOK_LIVE_AUTHORITY_DISABLED",
            }
        )
        _save(settings, state)
        return state
    rows = [dict(row) for row in opportunities]
    economics_families = {
        str(value).upper()
        for value in (allowed_economics_entry_families or [])
        if value
    }

    def economics_allows(row: Mapping[str, Any]) -> bool:
        return (
            canonical_family(str(row.get("playbook_id") or ""))[0]
            in economics_families
        )

    state["canonical_economics_entry_gate"] = {
        "allowed_families": sorted(economics_families),
        "new_entries_allowed": bool(economics_families),
        "position_management_affected": False,
    }
    prices = {
        str(row.get("market")): dict(row)
        for row in realtime_snapshot.get("markets") or []
        if row.get("market")
    }
    approved_markets = tuple(
        sorted(
            {
                str(market)
                for row in authority.get("approved_playbooks") or []
                if row.get("active") is True
                for market in row.get("markets") or []
            }
        )
    )
    if not approved_markets:
        state.update(
            {
                "status": "AUTHORITY_BLOCKED",
                "reason_code": "NO_APPROVED_EVENT_PLAYBOOK_MARKETS",
            }
        )
        _save(settings, state)
        return state
    if not submit:
        candidates = [
            row
            for row in rows
            if row.get("state") == "ENTRY_READY"
            and not row.get("hard_blockers")
            and _authorized(authority, row)
            and economics_allows(row)
        ]
        state.update(
            {
                "status": "READY_NOT_SUBMITTED",
                "reason_code": (
                    "APPROVED_EVENT_ENTRY_AVAILABLE"
                    if candidates
                    else (
                        "NO_APPROVED_EVENT_ENTRY_READY"
                        if economics_families
                        else "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
                    )
                ),
                "entry_candidates": candidates,
                "entry_ready_rejections": _entry_rejection_details(
                    authority,
                    rows,
                ),
            }
        )
        _save(settings, state)
        return state

    positions = {
        str(key): dict(value)
        for key, value in (state.get("positions") or {}).items()
    }
    preview_candidates = [
        row
        for row in rows
        if row.get("state") == "ENTRY_READY"
        and not row.get("hard_blockers")
        and _authorized(authority, row)
        and economics_allows(row)
        and row.get("opportunity_id") not in positions
    ]
    pending_acknowledged_buys = _pending_acknowledged_buys(paths["ledger"])
    if not positions and not preview_candidates and not pending_acknowledged_buys:
        state.update(
            {
                "status": "READY",
                "reason_code": (
                    "NO_APPROVED_EVENT_ENTRY_READY"
                    if economics_families
                    else "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
                ),
                "entry_ready_rejections": _entry_rejection_details(
                    authority,
                    rows,
                ),
                "positions": positions,
                "private_exchange_requests": 0,
            }
        )
        _save(settings, state)
        return state
    async with aiohttp.ClientSession() as session:
        try:
            client = build_live_client(
                settings,
                session=session,
                ledger_path=paths["ledger"],
            )
        except ExecutionBlocked as exc:
            return _blocked_cycle(
                settings,
                state,
                exc,
                phase="CLIENT_CONFIGURATION",
                positions=positions,
            )
        balances = await client.balances()
        reconciliation = await client.reconcile(markets=approved_markets)
        wallet_exposure = _wallet_exposure(
            settings,
            balances,
            prices,
        )
        state["managed_strategy_exposure_eur"] = str(
            _managed_exposure(positions)
        )
        state["wallet_exposure"] = wallet_exposure
        state["total_wallet_asset_exposure_eur"] = wallet_exposure[
            "total_wallet_asset_exposure_eur"
        ]
        state["wallet_concentration_fraction"] = wallet_exposure[
            "wallet_concentration_fraction"
        ]
        preflight = _live_capability(
            settings,
            markets=approved_markets,
            reconciliation_healthy=reconciliation.healthy,
        )
        if not preflight.passed or preflight.capability is None:
            state.update(
                {
                    "status": "PREFLIGHT_BLOCKED",
                    "reason_code": "EVENT_PLAYBOOK_LIVE_PREFLIGHT_BLOCKED",
                    "failures": list(preflight.failures),
                    "reconciliation": {
                        "healthy": reconciliation.healthy,
                        "reason_codes": list(reconciliation.reason_codes),
                    },
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        capability = preflight.capability

        # Restart recovery is deliberately performed before any new entry.
        # It turns an acknowledged maker order that filled after a process
        # interruption into a managed position using the original signal,
        # stop, targets and time-stop.  No new order is created here.
        for pending in pending_acknowledged_buys:
            opportunity_id = str(pending.get("signal_id") or "")
            market = str(pending.get("market") or "")
            client_order_id = str(pending.get("client_order_id") or "")
            if not opportunity_id or not market or not client_order_id:
                raise ReconciliationRequired(
                    "acknowledged live buy lacks recovery identity"
                )
            recovered_order = await client.get_order(
                market=market,
                client_order_id=client_order_id,
            )
            recovered_status = str(
                recovered_order.get("status") or ""
            ).replace("_", "").replace("-", "").casefold()
            recovered_quantity = Decimal(
                str(recovered_order.get("filledAmount") or "0")
            )
            if recovered_status == "partiallyfilled" and recovered_quantity > 0:
                order_id = str(recovered_order.get("orderId") or "")
                if not order_id:
                    raise ReconciliationRequired(
                        "partially filled live buy lacks venue order identity"
                    )
                await client.cancel_order(
                    market=market,
                    order_id=order_id,
                    capability=capability,
                )
                recovered_order = await client.get_order(
                    market=market,
                    client_order_id=client_order_id,
                )
                recovered_status = str(
                    recovered_order.get("status") or ""
                ).replace("_", "").replace("-", "").casefold()
                recovered_quantity = Decimal(
                    str(recovered_order.get("filledAmount") or "0")
                )
            recovered_terminal_partial = (
                recovered_status in {"canceled", "cancelled"}
                and recovered_quantity > 0
            )
            if (
                recovered_status != "filled"
                and not recovered_terminal_partial
            ) or recovered_quantity <= 0:
                state.update(
                    {
                        "status": "RECONCILIATION_BLOCKED",
                        "reason_code": "ACKNOWLEDGED_ENTRY_NOT_TERMINAL",
                        "positions": positions,
                    }
                )
                _save(settings, state)
                return state
            opportunity = next(
                (
                    dict(row)
                    for row in rows
                    if str(row.get("opportunity_id") or "")
                    == opportunity_id
                ),
                _lifecycle_opportunity(settings, opportunity_id),
            )
            required = {
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
                "time_stop_minutes",
            }
            if not opportunity or any(
                opportunity.get(field) is None for field in required
            ):
                raise ReconciliationRequired(
                    "acknowledged live fill lacks original exit plan"
                )
            quote = Decimal(
                str(recovered_order.get("filledAmountQuote") or "0")
            )
            recovered_price = (
                quote / recovered_quantity
                if quote > 0
                else Decimal(str(recovered_order.get("price") or "0"))
            )
            if recovered_price <= 0:
                raise ReconciliationRequired(
                    "acknowledged live fill lacks valid fill price"
                )
            client.record_final_fill(
                recovered_order,
                fallback_market=market,
                fallback_side=OrderSide.BUY,
                fallback_quantity=recovered_quantity,
                fallback_price=recovered_price,
                allow_terminal_partial=recovered_terminal_partial,
            )
            opened_at = _order_timestamp(recovered_order, now)
            realtime = prices.get(market) or {}
            book = dict(realtime.get("book") or {})
            recovered_position = {
                **opportunity,
                "opportunity_id": opportunity_id,
                "market": market,
                "playbook_id": str(pending.get("strategy_id") or opportunity.get("playbook_id") or ""),
                "playbook_dna": str(
                    pending.get("strategy_dna_hash")
                    or opportunity.get("playbook_dna")
                    or ""
                ),
                "status": "MANAGING",
                "quantity": str(recovered_quantity),
                "entry_price": str(recovered_price),
                "opened_at": opened_at.isoformat(),
                "entry_spread_bps": str(book.get("spread_bps") or "0"),
                "entry_bid_depth_eur_top_10": str(
                    book.get("bid_depth_eur_top_10") or "0"
                ),
                "initial_risk": str(
                    abs(
                        recovered_price
                        - Decimal(str(opportunity["stop_loss"]))
                    )
                ),
                "tp1_reached": False,
                "autoscale": False,
                "recovered_after_restart": True,
            }
            rules = await client.execution_market_rules(market)
            if not _supports_order_type(
                rules.supported_order_types,
                "stopLoss",
            ):
                raise ReconciliationRequired(
                    "recovered fill market lacks native protective stop"
                )
            recovered_trigger = rules.price(
                Decimal(str(opportunity["stop_loss"]))
            )
            protective_intent = _protective_stop_intent(
                recovered_position,
                quantity=recovered_quantity,
                trigger_price=recovered_trigger,
            )
            protective_order = await client.submit_order(
                protective_intent,
                capability=capability,
                estimated_price=recovered_price,
                reconciled_owned_quantity=recovered_quantity,
                exchange_minimum_order_eur=rules.minimum_order_value_eur,
            )
            protective_status = str(
                protective_order.get("status") or ""
            ).replace("_", "").replace("-", "").casefold()
            if protective_status not in {"new", "awaitingtrigger"}:
                raise ReconciliationRequired(
                    "recovered fill protective stop was not accepted"
                )
            recovered_position.update(
                {
                    "protective_stop_order_id": str(
                        protective_order["orderId"]
                    ),
                    "protective_stop_client_order_id": (
                        client.client_order_id_for(
                            protective_intent.idempotency_key
                        )
                    ),
                    "protective_stop_status": str(
                        protective_order.get("status")
                    ),
                    "protective_stop_trigger": str(recovered_trigger),
                    "local_hard_stop_active": True,
                }
            )
            positions[opportunity_id] = recovered_position
            state["orders_generated_this_cycle"] += 1
            state["orders_submitted_this_cycle"] += 1
            state["fills_verified_this_cycle"] += 1
            state["fills_verified"] = int(
                state.get("fills_verified") or 0
            ) + 1
            state["events"].append(
                {
                    "event": "LIVE_POSITION_RECOVERED_AFTER_RESTART",
                    "opportunity_id": opportunity_id,
                    "market": market,
                    "quantity": str(recovered_quantity),
                    "price": str(recovered_price),
                    "native_protective_stop": "ACCEPTED",
                }
            )

        for identity, position in list(positions.items()):
            market = str(position["market"])
            quantity = Decimal(str(position["quantity"]))
            protective_client_id = str(
                position.get("protective_stop_client_order_id") or ""
            )
            if protective_client_id:
                protective = await client.get_order(
                    market=market,
                    client_order_id=protective_client_id,
                )
                protective_status = str(
                    protective.get("status") or ""
                ).replace("_", "").replace("-", "").casefold()
                if protective_status == "filled":
                    client.record_final_fill(
                        protective,
                        fallback_market=market,
                        fallback_side=OrderSide.SELL,
                        fallback_quantity=quantity,
                        fallback_price=Decimal(
                            str(position["stop_loss"])
                        ),
                    )
                    positions.pop(identity, None)
                    state["fills_verified_this_cycle"] += 1
                    state["events"].append(
                        {
                            "event": "LIVE_NATIVE_STOP_FILLED",
                            "opportunity_id": identity,
                            "market": market,
                            "status": protective.get("status"),
                        }
                    )
                    continue
                if protective_status == "partiallyfilled":
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "reason_code": (
                                "PROTECTIVE_STOP_PARTIAL_FILL_REQUIRES_RECONCILIATION"
                            ),
                            "positions": positions,
                        }
                    )
                    _save(settings, state)
                    return state
            realtime = prices.get(market) or {}
            price = Decimal(str(realtime.get("price") or "0"))
            book = dict(realtime.get("book") or {})
            best_bid = Decimal(str(book.get("best_bid") or "0"))
            if price <= 0 or best_bid <= 0 or not realtime.get("fresh"):
                continue
            stop = Decimal(str(position["stop_loss"]))
            tp1 = Decimal(str(position["take_profit_1"]))
            tp2 = Decimal(str(position["take_profit_2"]))
            opened = datetime.fromisoformat(
                str(position["opened_at"]).replace("Z", "+00:00")
            ).astimezone(UTC)
            matching = max(
                (
                    row
                    for row in rows
                    if row.get("market") == market
                    and row.get("playbook_dna") == position.get("playbook_dna")
                ),
                key=lambda row: float(row.get("score") or 0),
                default={},
            )
            microstructure_exit = _microstructure_exit_reason(
                position,
                realtime,
                btc_realtime=prices.get("BTC-EUR"),
                matching_opportunity=matching,
            )
            time_review_due = (now - opened).total_seconds() >= (
                int(position.get("time_stop_minutes") or 30) * 60
            )
            entry_price = Decimal(str(position["entry_price"]))
            structure_no_longer_supported = (
                matching is None
                or bool(matching.get("hard_blockers"))
                or str(matching.get("state") or "").upper()
                in {"INVALIDATED", "EXPIRED"}
            )
            time_exit_confirmed = (
                time_review_due
                and not position.get("tp1_reached")
                and price <= entry_price
                and (structure_no_longer_supported or bool(microstructure_exit))
            )
            reason: str | None = None
            if price <= stop:
                reason = "HARD_STOP"
            elif price >= tp2:
                reason = "TAKE_PROFIT_2"
            elif time_exit_confirmed:
                reason = "TIME_AND_STRUCTURE_EXIT"
            elif _soft_exit_confirmed(
                position,
                reason=microstructure_exit,
                realtime=realtime,
                price=price,
                now=now,
            ):
                reason = microstructure_exit
            elif microstructure_exit:
                positions[identity] = position
                state["events"].append(
                    {
                        "event": "SOFT_EXIT_CONFIRMATION_PENDING",
                        "opportunity_id": identity,
                        "market": market,
                        "reason": microstructure_exit,
                        "observations": position.get(
                            "soft_exit_observations"
                        ),
                    }
                )
                continue
            elif price >= tp1 and not position.get("tp1_reached"):
                # A bounded canary exit must not assume that half the original
                # notional remains above Bitvavo's €5 venue minimum after
                # fees, price movement and quantity rounding.  Preserve the
                # runner and make risk-free instead.
                tightened_stop = Decimal(str(position["entry_price"]))
                position.update(
                    await _replace_native_protective_stop(
                        client,
                        capability=capability,
                        position=position,
                        quantity=quantity,
                        trigger_price=tightened_stop,
                        estimated_price=price,
                    )
                )
                position["tp1_reached"] = True
                position["stop_loss"] = str(tightened_stop)
                positions[identity] = position
                state["events"].append(
                    {
                        "event": "TP1_BREAKEVEN_ONLY_VENUE_MINIMUM",
                        "opportunity_id": identity,
                        "market": market,
                    }
                )
                continue
            elif position.get("tp1_reached"):
                initial_risk = Decimal(
                    str(
                        position.get("initial_risk")
                        or abs(
                            Decimal(str(position["entry_price"]))
                            - Decimal(str(position["stop_loss"]))
                        )
                    )
                )
                trailed = max(
                    Decimal(str(position["stop_loss"])),
                    Decimal(str(position["entry_price"])),
                    price - initial_risk,
                )
                current_stop = Decimal(str(position["stop_loss"]))
                if trailed > current_stop * Decimal("1.0025"):
                    position.update(
                        await _replace_native_protective_stop(
                            client,
                            capability=capability,
                            position=position,
                            quantity=quantity,
                            trigger_price=trailed,
                            estimated_price=price,
                        )
                    )
                    position["stop_loss"] = str(trailed)
                    positions[identity] = position
                    state["events"].append(
                        {
                            "event": "LIVE_RUNNER_TRAILING_STOP_UPDATED",
                            "opportunity_id": identity,
                            "market": market,
                            "stop_loss": str(trailed),
                        }
                    )
            if reason is None:
                continue
            try:
                rules = await client.execution_market_rules(market)
            except ExecutionBlocked:
                state.update(
                    {
                        "status": "EXECUTION_RULES_BLOCKED",
                        "reason_code": "EXIT_MARKET_RULES_TEMPORARILY_UNAVAILABLE",
                        "positions": positions,
                    }
                )
                _save(settings, state)
                return state
            exit_price = rules.price(
                best_bid * Decimal("0.999")
            )
            if protective_client_id:
                protective = await client.get_order(
                    market=market,
                    client_order_id=protective_client_id,
                )
                protective_status = str(
                    protective.get("status") or ""
                ).replace("_", "").replace("-", "").casefold()
                if protective_status == "filled":
                    client.record_final_fill(
                        protective,
                        fallback_market=market,
                        fallback_side=OrderSide.SELL,
                        fallback_quantity=quantity,
                        fallback_price=stop,
                    )
                    positions.pop(identity, None)
                    state["fills_verified_this_cycle"] += 1
                    state["fills_verified"] = int(
                        state.get("fills_verified") or 0
                    ) + 1
                    state["events"].append(
                        {
                            "event": "LIVE_NATIVE_STOP_FILLED",
                            "opportunity_id": identity,
                            "market": market,
                            "status": protective.get("status"),
                        }
                    )
                    continue
                if protective_status in {
                    "new",
                    "awaitingtrigger",
                    "partiallyfilled",
                }:
                    await client.cancel_order(
                        market=market,
                        order_id=str(protective["orderId"]),
                        capability=capability,
                    )
                    balances = await client.balances()
                    position["protective_stop_status"] = "CANCELLED_FOR_EXIT"
            owned = _balance(
                balances,
                market.split("-")[0],
                include_in_order=True,
            )
            sell_quantity = min(quantity, owned)
            if sell_quantity <= 0:
                state.update(
                    {
                        "status": "RECONCILIATION_BLOCKED",
                        "reason_code": "EVENT_POSITION_OWNED_QUANTITY_MISSING",
                        "positions": positions,
                    }
                )
                _save(settings, state)
                return state
            try:
                order = await client.submit_order(
                    _exit_intent(
                        position,
                        quantity=sell_quantity,
                        limit_price=exit_price,
                        reason=reason,
                    ),
                    capability=capability,
                    estimated_price=price,
                    reconciled_owned_quantity=owned,
                    exchange_minimum_order_eur=rules.minimum_order_value_eur,
                )
            except ExecutionBlocked as exc:
                return _blocked_cycle(
                    settings,
                    state,
                    exc,
                    phase="EXIT",
                    positions=positions,
                )
            state["orders_generated_this_cycle"] += 1
            state["orders_submitted_this_cycle"] += 1
            state["orders_generated"] = int(
                state.get("orders_generated") or 0
            ) + 1
            state["orders_submitted"] = int(
                state.get("orders_submitted") or 0
            ) + 1
            state["events"].append(
                {
                    "event": "LIVE_EXIT_SUBMITTED",
                    "opportunity_id": identity,
                    "market": market,
                    "reason": reason,
                    "status": order.get("status"),
                }
            )
            status = str(order.get("status") or "").casefold()
            filled = Decimal(str(order.get("filledAmount") or "0"))
            if status == "filled" or filled > 0:
                executed = filled if filled > 0 else sell_quantity
                remaining = max(Decimal("0"), quantity - executed)
                state["fills_verified_this_cycle"] += 1
                state["fills_verified"] = int(
                    state.get("fills_verified") or 0
                ) + 1
                if remaining > 0:
                    position["quantity"] = str(remaining)
                    position["exit_attempt"] = int(
                        position.get("exit_attempt") or 0
                    ) + 1
                    positions[identity] = position
                    event_name = "LIVE_POSITION_REDUCED"
                    cycle_status = "POSITION_REDUCED"
                else:
                    positions.pop(identity, None)
                    event_name = "LIVE_POSITION_CLOSED"
                    cycle_status = "POSITION_CLOSED"
                state["events"].append(
                    {
                        "event": event_name,
                        "opportunity_id": identity,
                        "market": market,
                        "reason": reason,
                        "quantity": str(executed),
                        "remaining_quantity": str(remaining),
                        "price": str(
                            Decimal(
                                str(order.get("filledAmountQuote") or "0")
                            )
                            / executed
                            if executed > 0
                            and Decimal(
                                str(order.get("filledAmountQuote") or "0")
                            )
                            > 0
                            else price
                        ),
                        **(
                            _signal_fill_summary(paths["ledger"], identity)
                            if remaining <= 0
                            else {}
                        ),
                    }
                )
                state["positions"] = positions
                state["status"] = cycle_status
                state["reason_code"] = reason
                _save(settings, state)
                return state
            state.update(
                {
                    "status": "EXIT_NOT_FILLED",
                    "reason_code": "BOUNDED_EXIT_IOC_TERMINAL_NO_FILL",
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state

        if not allow_new_entry:
            state.update(
                {
                    "status": "ENTRIES_DISABLED",
                    "reason_code": "SUPERVISOR_ENTRY_AUTHORITY_NOT_READY",
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        inventory_risk_override = evaluate_inventory_risk_override(
            settings,
            balances,
        )
        if wallet_exposure["status"] != "READY":
            state.update(
                {
                    "status": "PORTFOLIO_HEAT_BLOCKED",
                    "reason_code": "TOTAL_WALLET_EXPOSURE_VALUATION_INCOMPLETE",
                    "positions": positions,
                    "protective_exits_allowed": True,
                }
            )
            _save(settings, state)
            return state
        shared_managed = managed_live_portfolio(settings)
        occupied_markets = {
            str(row["market"]) for row in positions.values()
        } | {
            str(row.get("market") or "")
            for row in shared_managed.get("positions") or []
        }
        occupied_families = Counter(
            str(row.get("family") or "") for row in positions.values()
        )
        qualified: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for raw in rows:
            row = dict(raw)
            authority_row = _authorized_playbook_row(authority, row)
            family = str(row.get("family") or "")
            maximum_family_positions = int(
                (authority_row or {}).get("maximum_family_positions") or 1
            )
            if (
                row.get("state") != "ENTRY_READY"
                or row.get("hard_blockers")
                or authority_row is None
                or not economics_allows(row)
                or row.get("market") in occupied_markets
                or row.get("opportunity_id") in positions
                or occupied_families[family] >= maximum_family_positions
                or (
                    inventory_risk_override["active"]
                    and row.get("market")
                    == f"{inventory_risk_override['asset']}-EUR"
                )
            ):
                continue
            quality = _candidate_selection_score(row, authority_row)
            qualified.append((quality, row, authority_row))
        selected_tuple = max(qualified, key=lambda item: item[0], default=None)
        selected = selected_tuple[1] if selected_tuple is not None else None
        selected_authority = (
            selected_tuple[2] if selected_tuple is not None else {}
        )
        if selected is None:
            state.update(
                {
                    "status": "READY",
                    "reason_code": (
                        "NO_APPROVED_EVENT_ENTRY_READY"
                        if economics_families
                        else "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
                    ),
                    "entry_ready_rejections": _entry_rejection_details(
                        authority,
                        rows,
                        positions=positions,
                    ),
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        market = str(selected["market"])
        regime_risk_multiplier = max(
            Decimal("0"),
            min(
                Decimal("1"),
                Decimal(
                    str(
                        selected.get("playbook_risk_multiplier") or "1"
                    )
                ),
            ),
        )
        evidence_multiplier = _authority_evidence_multiplier(
            selected_authority
        )
        playbook_risk_multiplier = min(
            Decimal("1"),
            regime_risk_multiplier * evidence_multiplier,
        )
        selected["selection_quality_score"] = selected_tuple[0]
        selected["evidence_multiplier"] = str(evidence_multiplier)
        selected["regime_risk_multiplier"] = str(regime_risk_multiplier)
        selected["strategy_role"] = selected_authority.get(
            "strategy_role", "LEGACY_LEVEL_2"
        )
        entry_notional_eur = (
            MAXIMUM_ORDER_EUR * playbook_risk_multiplier
        )
        realtime = prices.get(market) or {}
        book = dict(realtime.get("book") or {})
        public_price = Decimal(str(realtime.get("price") or "0"))
        best_bid = Decimal(str(book.get("best_bid") or "0"))
        best_ask = Decimal(str(book.get("best_ask") or "0"))
        if (
            not realtime.get("fresh")
            or not realtime.get("sequence_valid")
            or public_price <= 0
            or best_bid <= 0
            or best_ask <= 0
        ):
            state.update(
                {
                    "status": "DATA_BLOCKED",
                    "reason_code": "REALTIME_ENTRY_FACTS_NOT_READY",
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        try:
            rules = await client.execution_market_rules(market)
        except ExecutionBlocked:
            state.update(
                {
                    "status": "EXECUTION_RULES_BLOCKED",
                    "reason_code": "ENTRY_MARKET_RULES_TEMPORARILY_UNAVAILABLE",
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        safe_minimum_order_eur = (
            rules.minimum_order_value_eur * Decimal("1.15")
        ).quantize(Decimal("0.01"))
        protective_trigger = rules.price(
            Decimal(str(selected["stop_loss"]))
        )
        risk_limited_notional_eur = _risk_limited_entry_notional(
            desired_notional_eur=entry_notional_eur,
            entry_price=best_ask,
            stop_price=protective_trigger,
        )
        if risk_limited_notional_eur <= 0:
            state.update(
                {
                    "status": "ENTRY_BLOCKED",
                    "reason_code": "INVALID_STRUCTURAL_STOP",
                    "positions": positions,
                    "entry_price": str(best_ask),
                    "protective_stop_trigger": str(protective_trigger),
                }
            )
            _save(settings, state)
            return state
        entry_notional_eur = min(
            entry_notional_eur,
            risk_limited_notional_eur,
        )
        try:
            protectable_minimum_order_eur = (
                minimum_protectable_entry_notional(
                    entry_price=best_ask,
                    stop_price=protective_trigger,
                    rules=rules,
                )
            )
        except ExecutionBlocked as exc:
            return _blocked_cycle(
                settings,
                state,
                exc,
                phase="ENTRY",
                positions=positions,
            )
        entry_notional_eur = max(
            entry_notional_eur,
            safe_minimum_order_eur,
            protectable_minimum_order_eur,
        )
        if entry_notional_eur > MAXIMUM_ORDER_EUR:
            state.update(
                {
                    "status": "ENTRY_BLOCKED",
                    "reason_code": "SAFE_VENUE_MINIMUM_EXCEEDS_LIVE_CAP",
                    "positions": positions,
                    "safe_minimum_order_eur": str(
                        safe_minimum_order_eur
                    ),
                    "protectable_minimum_order_eur": str(
                        protectable_minimum_order_eur
                    ),
                }
            )
            _save(settings, state)
            return state
        conservative_planned_risk_eur = (
            entry_notional_eur
            * (best_ask - protective_trigger)
            / best_ask
        )
        if conservative_planned_risk_eur > MAXIMUM_RISK_PER_TRADE_EUR:
            state.update(
                {
                    "status": "ENTRY_BLOCKED",
                    "reason_code": "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
                    "positions": positions,
                    "planned_risk_eur": str(
                        conservative_planned_risk_eur
                    ),
                    "maximum_risk_per_trade_eur": str(
                        MAXIMUM_RISK_PER_TRADE_EUR
                    ),
                    "risk_limited_notional_eur": str(
                        risk_limited_notional_eur
                    ),
                    "protectable_minimum_order_eur": str(
                        protectable_minimum_order_eur
                    ),
                }
            )
            _save(settings, state)
            return state
        capacity_ok, capacity_reason, shared_managed = (
            capital_level_2_capacity(
                settings,
                requested_notional_eur=entry_notional_eur,
            )
        )
        if not capacity_ok:
            state.update(
                {
                    "status": "PORTFOLIO_HEAT_BLOCKED",
                    "reason_code": capacity_reason,
                    "positions": positions,
                    "managed_portfolio": shared_managed,
                    "protective_exits_allowed": True,
                }
            )
            _save(settings, state)
            return state
        if _balance(balances, "EUR") < entry_notional_eur:
            state.update(
                {
                    "status": "BALANCE_BLOCKED",
                    "reason_code": "AVAILABLE_EUR_BELOW_SAFE_MINIMUM_ORDER",
                    "positions": positions,
                    "required_eur": str(entry_notional_eur),
                }
            )
            _save(settings, state)
            return state
        if not _supports_order_type(
            rules.supported_order_types,
            "stopLoss",
        ):
            state.update(
                {
                    "status": "ENTRY_BLOCKED",
                    "reason_code": "VENUE_NATIVE_STOP_LOSS_UNSUPPORTED",
                    "positions": positions,
                }
            )
            _save(settings, state)
            return state
        maker_price = rules.price(best_bid)
        maker_quantity = rules.amount(entry_notional_eur / maker_price)
        maker_feasible = (
            maker_quantity >= rules.minimum_order_amount
            and maker_quantity * maker_price
            >= rules.minimum_order_value_eur
            and quantity_is_protectable_at_stop(
                quantity=maker_quantity,
                stop_price=protective_trigger,
                rules=rules,
            )
        )
        # A sub-two-minimum canary cannot safely accept an arbitrary partial
        # maker fill: the remainder might be below the venue's stop minimum.
        # Use one bounded FOK limit so the entry is atomic at this size.
        atomic_canary_entry = (
            entry_notional_eur
            < safe_minimum_order_eur * Decimal("2")
        )
        order: dict[str, Any] | None = None
        state["events"].append(
            {
                "event": "LIVE_ORDER_INTENT_CREATED",
                "opportunity_id": str(selected["opportunity_id"]),
                "market": market,
                "playbook_id": str(selected["playbook_id"]),
                "playbook_dna": str(selected["playbook_dna"]),
                "maximum_notional_eur": str(entry_notional_eur),
                "planned_risk_eur": str(conservative_planned_risk_eur),
                "maximum_risk_per_trade_eur": str(
                    MAXIMUM_RISK_PER_TRADE_EUR
                ),
                "playbook_risk_multiplier": str(
                    playbook_risk_multiplier
                ),
                "regime_risk_multiplier": str(regime_risk_multiplier),
                "evidence_multiplier": str(evidence_multiplier),
                "selection_quality_score": selected[
                    "selection_quality_score"
                ],
                "strategy_role": selected["strategy_role"],
                "trigger_ts": selected.get("trigger_ts"),
                "order_intent_ts": now.isoformat(),
                "edge_consumed_at_submit": selected.get("edge_consumed"),
                "state": "ORDER_INTENT_CREATED",
            }
        )

        async def submit_reserved_event_entry(
            intent: OrderIntent,
        ) -> dict[str, Any]:
            notional_price = max(
                public_price,
                intent.limit_price or public_price,
            )
            planned_risk_eur = intent.quantity * (
                notional_price - protective_trigger
            )
            if (
                planned_risk_eur <= 0
                or planned_risk_eur > MAXIMUM_RISK_PER_TRADE_EUR
            ):
                raise ExecutionBlocked(
                    "event entry exceeds maximum risk per trade"
                )
            execution_economics = dict(
                selected.get("execution_economics") or {}
            )
            expected_net_value_bps = Decimal(
                str(execution_economics.get("expected_net_value_bps") or "0")
            )
            expected_net_edge = expected_net_value_bps / Decimal("10000")
            if expected_net_edge <= 0:
                expected_net_edge = planned_target_net_edge(
                    entry_price=notional_price,
                    target_price=Decimal(str(selected["take_profit_1"])),
                    costs=CanonicalCostModel.from_settings(settings),
                )
            confidence = min(
                Decimal("1"),
                max(
                    Decimal("0.01"),
                    Decimal(str(selected["selection_quality_score"])),
                ),
            )
            current_quantity = _balance(
                balances,
                market.split("-")[0],
            )
            canonical_plan = canonicalize_approved_buy_order(
                settings,
                intent,
                mark_price=notional_price,
                current_quantity=current_quantity,
                equity_eur=Decimal(
                    str(wallet_exposure["estimated_wallet_equity_eur"])
                ),
                approved_risk_eur=planned_risk_eur,
                expected_net_edge=expected_net_edge,
                confidence=confidence,
                family=str(selected.get("family") or "EVENT_PLAYBOOK"),
                evidence_id=str(selected["opportunity_id"]),
                policy_version=str(
                    authority.get("authority_hash")
                    or "event_playbook_live_authority_v1"
                ),
                account_state={
                    "wallet_exposure": wallet_exposure,
                    "reconciliation_healthy": reconciliation.healthy,
                    "reconciliation_reasons": list(
                        reconciliation.reason_codes
                    ),
                },
                portfolio_state={
                    "managed_portfolio": shared_managed,
                    "event_positions": positions,
                },
                horizon_seconds=max(
                    60,
                    int(selected.get("time_stop_minutes") or 30) * 60,
                ),
            )
            canonical_intent = canonical_plan.order

            async def submit_with_fresh_portfolio(
                fresh_portfolio: Mapping[str, Any],
            ) -> dict[str, Any]:
                return await client.submit_order(
                    canonical_intent,
                    capability=capability,
                    estimated_price=public_price,
                    reconciled_owned_quantity=Decimal("0"),
                    reconciled_total_exposure_eur=Decimal(
                        str(
                            fresh_portfolio[
                                "capacity_managed_exposure_eur"
                            ]
                        )
                    ),
                    reconciled_open_positions=int(
                        fresh_portfolio[
                            "capacity_managed_position_count"
                        ]
                    ),
                    exchange_minimum_order_eur=(
                        rules.minimum_order_value_eur
                    ),
                    canonical_chain=canonical_plan.chain,
                )

            approved, reason, _portfolio, result = (
                await submit_level_2_buy_atomically(
                    settings,
                    requested_notional_eur=(
                        canonical_intent.quantity * notional_price
                    ),
                    submit_order=submit_with_fresh_portfolio,
                )
            )
            if not approved or result is None:
                raise ExecutionBlocked(reason)
            return result

        if maker_feasible and not atomic_canary_entry:
            maker_intent = _entry_intent(
                selected,
                quantity=maker_quantity,
                limit_price=maker_price,
                attempt=1,
                post_only=True,
            )
            await client.arm_cancel_on_disconnect(
                group_id=str(maker_intent.cancel_on_disconnect_group),
                expiry_after_seconds=30,
            )
            try:
                order = await submit_reserved_event_entry(maker_intent)
            except ExecutionBlocked as exc:
                return _blocked_cycle(
                    settings,
                    state,
                    exc,
                    phase="ENTRY",
                    positions=positions,
                )
            state["orders_generated_this_cycle"] += 1
            state["orders_submitted_this_cycle"] += 1
            state["events"].append(
                {
                    "event": "LIVE_ORDER_SUBMITTED",
                    "opportunity_id": str(selected["opportunity_id"]),
                    "market": market,
                    "order_id": order.get("orderId"),
                    "submit_ts": now.isoformat(),
                    "exchange_ack_ts": datetime.now(UTC).isoformat(),
                    "attempt": "MAKER_FIRST",
                    "status": order.get("status"),
                }
            )
            if str(order.get("status") or "").casefold() != "filled":
                await asyncio.sleep(MAKER_WAIT_SECONDS)
                maker_order = await client.get_order(
                    market=market,
                    client_order_id=client.client_order_id_for(
                        maker_intent.idempotency_key
                    ),
                )
                status = str(maker_order.get("status") or "").casefold()
                if status != "filled":
                    try:
                        await client.cancel_order(
                            market=market,
                            order_id=str(maker_order["orderId"]),
                            capability=capability,
                        )
                    except ExecutionBlocked as exc:
                        raise ReconciliationRequired(
                            "maker cancellation requires reconciliation"
                        ) from exc
                    # Cancellation and a fill can race.  Resolve the terminal
                    # venue facts before deciding whether a fallback is safe.
                    maker_order = await client.get_order(
                        market=market,
                        client_order_id=client.client_order_id_for(
                            maker_intent.idempotency_key
                        ),
                    )
                filled_maker = Decimal(
                    str(maker_order.get("filledAmount") or "0")
                )
                if filled_maker > 0:
                    client.record_final_fill(
                        maker_order,
                        fallback_market=market,
                        fallback_side=OrderSide.BUY,
                        fallback_quantity=maker_quantity,
                        fallback_price=maker_price,
                        allow_terminal_partial=True,
                    )
                    if str(maker_order.get("status") or "").casefold() != "filled":
                        state["events"].append(
                            {
                                "event": "LIVE_ORDER_PARTIALLY_FILLED",
                                "opportunity_id": str(
                                    selected["opportunity_id"]
                                ),
                                "market": market,
                                "quantity": str(filled_maker),
                                "attempt": "MAKER_FIRST",
                            }
                        )
                    order = maker_order
                elif str(maker_order.get("status") or "").casefold() == "filled":
                    order = maker_order
                else:
                    order = None
        if order is None:
            marketable_price = rules.price_up(
                best_ask
                * (
                    Decimal("1")
                    + Decimal(
                        str(
                            min(
                                25.0,
                                max(
                                    1.0,
                                    float(
                                        realtime.get(
                                            "estimated_buy_slippage_bps"
                                        )
                                        or 1.0
                                    )
                                    + 1.0,
                                ),
                            )
                        )
                    )
                    / Decimal("10000")
                )
            )
            quantity = rules.amount(entry_notional_eur / marketable_price)
            if (
                quantity * marketable_price
                < rules.minimum_order_value_eur
                or not quantity_is_protectable_at_stop(
                    quantity=quantity,
                    stop_price=protective_trigger,
                    rules=rules,
                )
            ):
                # Never replace an infeasible bounded limit with an unbounded
                # market order.  Some coarse base-quantity precisions require
                # more than the explicit EUR 10 canary cap to reach the venue
                # minimum; that market is not executable at this authority
                # level and must fail closed.
                state.update(
                    {
                        "status": "ENTRY_BLOCKED",
                        "reason_code": "BOUNDED_LIMIT_INFEASIBLE_WITHIN_CANARY_CAP",
                        "positions": positions,
                    }
                )
                _save(settings, state)
                return state
            fallback_intent = _entry_intent(
                selected,
                quantity=quantity,
                limit_price=marketable_price,
                attempt=2,
                post_only=False,
                time_in_force=(
                    OrderTimeInForce.FOK
                    if atomic_canary_entry
                    else OrderTimeInForce.IOC
                ),
            )
            await client.arm_cancel_on_disconnect(
                group_id=str(
                    fallback_intent.cancel_on_disconnect_group
                ),
                expiry_after_seconds=30,
            )
            fallback_attempt = (
                "BOUNDED_MARKETABLE_FOK_ATOMIC_CANARY"
                if atomic_canary_entry
                else "BOUNDED_MARKETABLE_IOC"
            )
            try:
                order = await submit_reserved_event_entry(fallback_intent)
            except ExecutionBlocked as exc:
                return _blocked_cycle(
                    settings,
                    state,
                    exc,
                    phase="ENTRY",
                    positions=positions,
                )
            state["orders_generated_this_cycle"] += 1
            state["orders_submitted_this_cycle"] += 1
            state["events"].append(
                {
                    "event": "LIVE_ORDER_SUBMITTED",
                    "opportunity_id": str(selected["opportunity_id"]),
                    "market": market,
                    "order_id": order.get("orderId"),
                    "submit_ts": now.isoformat(),
                    "exchange_ack_ts": datetime.now(UTC).isoformat(),
                    "attempt": fallback_attempt,
                    "status": order.get("status"),
                }
            )
        status = str(order.get("status") or "").casefold()
        filled = Decimal(str(order.get("filledAmount") or "0"))
        if status == "filled" or filled > 0:
            quantity = filled if filled > 0 else Decimal(
                str(order.get("amount") or "0")
            )
            quote = Decimal(str(order.get("filledAmountQuote") or "0"))
            fill_price = (
                quote / quantity if quote > 0 and quantity > 0 else public_price
            )
            identity = str(selected["opportunity_id"])
            position = {
                **selected,
                "status": "MANAGING",
                "quantity": str(quantity),
                "entry_price": str(fill_price),
                "opened_at": now.isoformat(),
                "entry_spread_bps": str(book.get("spread_bps") or "0"),
                "entry_bid_depth_eur_top_10": str(
                    book.get("bid_depth_eur_top_10") or "0"
                ),
                "initial_risk": str(
                    abs(
                        fill_price
                        - Decimal(str(selected["stop_loss"]))
                    )
                ),
                "tp1_reached": False,
                "autoscale": False,
            }
            if not quantity_is_protectable_at_stop(
                quantity=quantity,
                stop_price=protective_trigger,
                rules=rules,
            ):
                raise ReconciliationRequired(
                    "verified entry fill is below the protective stop minimum"
                )
            protective_intent = _protective_stop_intent(
                position,
                quantity=quantity,
                trigger_price=protective_trigger,
            )
            protective_order = await client.submit_order(
                protective_intent,
                capability=capability,
                estimated_price=fill_price,
                reconciled_owned_quantity=quantity,
                exchange_minimum_order_eur=rules.minimum_order_value_eur,
            )
            protective_status = str(
                protective_order.get("status") or ""
            ).replace("_", "").replace("-", "").casefold()
            if protective_status not in {"new", "awaitingtrigger"}:
                raise ReconciliationRequired(
                    "native protective stop was not accepted as active"
                )
            position.update(
                {
                    "protective_stop_order_id": str(
                        protective_order["orderId"]
                    ),
                    "protective_stop_client_order_id": (
                        client.client_order_id_for(
                            protective_intent.idempotency_key
                        )
                    ),
                    "protective_stop_status": str(
                        protective_order.get("status")
                    ),
                    "protective_stop_trigger": str(protective_trigger),
                    "local_hard_stop_active": True,
                }
            )
            positions[identity] = position
            state["orders_generated_this_cycle"] += 1
            state["orders_submitted_this_cycle"] += 1
            state["events"].append(
                {
                    "event": "LIVE_NATIVE_PROTECTIVE_STOP_ACCEPTED",
                    "opportunity_id": identity,
                    "market": market,
                    "status": protective_order.get("status"),
                    "trigger_price": str(protective_trigger),
                }
            )
            state["fills_verified_this_cycle"] += 1
            state["events"].append(
                {
                    "event": "LIVE_POSITION_FILLED",
                    "opportunity_id": identity,
                    "market": market,
                    "quantity": str(quantity),
                    "price": str(fill_price),
                }
            )
            state["status"] = "POSITION_OPENED"
            state["reason_code"] = "EVENT_PLAYBOOK_LIVE_FILL_VERIFIED"
        else:
            # Both maker and fallback attempts are terminal here.  Never carry
            # an untracked pending entry into the next supervisor cycle.
            state["status"] = "ENTRY_NOT_FILLED"
            state["reason_code"] = "BOUNDED_ENTRY_ATTEMPTS_TERMINAL_NO_FILL"
        state["positions"] = positions
        state["orders_generated"] = int(state.get("orders_generated") or 0) + int(
            state["orders_generated_this_cycle"]
        )
        state["orders_submitted"] = int(state.get("orders_submitted") or 0) + int(
            state["orders_submitted_this_cycle"]
        )
        state["fills_verified"] = int(state.get("fills_verified") or 0) + int(
            state["fills_verified_this_cycle"]
        )
        _save(settings, state)
        return state


__all__ = [
    "DEFAULT_NEW_FAMILY_EVIDENCE_MULTIPLIER",
    "MAXIMUM_ORDER_EUR",
    "approve_playbook_live",
    "approval_phrase",
    "deactivate_playbook_live",
    "execute_event_driven_live_once",
    "execution_block_reason_code",
    "execution_block_requires_authority_deactivation",
    "is_playbook_opportunity_authorized",
    "migrate_playbook_live_capital_level_2",
    "playbook_authority_status",
    "playbook_catalog",
]
