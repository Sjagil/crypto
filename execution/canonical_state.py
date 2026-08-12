"""Canonical, deterministic financial execution state.

The append-only execution ledger is the economic source of truth.  This module
normalizes its immutable records, deduplicates economic identities and reduces
them into restart-safe orders, fills, positions, protection and capital read
models.  It deliberately performs no I/O, network, database or clock calls in
the reducer.

Accounting convention
---------------------
Long spot entry fees are capitalized into FIFO lot cost.  Exit fees are
deducted from realized proceeds.  A sell without a reconstructable local lot
is retained as an evidence gap; cost basis and realized PnL are never invented.
Confirmed protective coverage requires an acknowledged exchange stop in an
active venue state.  Unprotected quantity is conservatively treated as having
its full known cost basis at risk.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Iterable, Mapping

from core.contracts import normalize_market
from utils.common import stable_hash, stable_json

ZERO = Decimal("0")
SCHEMA_VERSION = "canonical_execution_state_v1"


class CanonicalOrderStatus(StrEnum):
    INTENT_CREATED = "INTENT_CREATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class ProtectionState(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    MISSING = "MISSING"
    INTENDED = "PROTECTION_INTENDED"
    CONFIRMED_ACTIVE = "PROTECTION_CONFIRMED_AT_EXCHANGE"
    PARTIAL = "PARTIAL_PROTECTION_CONFIRMED_AT_EXCHANGE"
    REJECTED = "PROTECTION_REJECTED"
    CANCELLED = "PROTECTION_CANCELLED"
    FILLED = "PROTECTION_FILLED"
    UNKNOWN = "PROTECTION_UNKNOWN"
    CONFLICT = "PROTECTION_CONFLICT"


class OwnershipState(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    MIXED = "MIXED"


TERMINAL_STATUSES = {
    CanonicalOrderStatus.FILLED,
    CanonicalOrderStatus.CANCELLED,
    CanonicalOrderStatus.REJECTED,
    CanonicalOrderStatus.EXPIRED,
}
ACTIVE_STATUSES = {
    CanonicalOrderStatus.ACKNOWLEDGED,
    CanonicalOrderStatus.OPEN,
    CanonicalOrderStatus.PARTIALLY_FILLED,
    CanonicalOrderStatus.CANCEL_REQUESTED,
}


def _decimal(value: Any, *, default: Decimal = ZERO) -> Decimal:
    if value in (None, ""):
        return default
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default
    return selected if selected.is_finite() else default


def _timestamp(value: Any, *, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) >= 100_000_000_000:
            numeric /= 1_000
        parsed = datetime.fromtimestamp(numeric, tz=UTC)
    elif value not in (None, ""):
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    elif fallback is not None:
        parsed = fallback
    else:
        # Historical ledger records always supply recorded_at.  A fixed epoch
        # keeps malformed fixtures deterministic without reading wall time.
        parsed = datetime(1970, 1, 1, tzinfo=UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalized_status(value: Any) -> CanonicalOrderStatus:
    text = str(value or "").replace("_", "").replace("-", "").casefold()
    return {
        "created": CanonicalOrderStatus.INTENT_CREATED,
        "submitted": CanonicalOrderStatus.SUBMITTED,
        "acknowledged": CanonicalOrderStatus.ACKNOWLEDGED,
        "new": CanonicalOrderStatus.OPEN,
        "open": CanonicalOrderStatus.OPEN,
        "awaitingtrigger": CanonicalOrderStatus.OPEN,
        "partiallyfilled": CanonicalOrderStatus.PARTIALLY_FILLED,
        "filled": CanonicalOrderStatus.FILLED,
        "cancelrequested": CanonicalOrderStatus.CANCEL_REQUESTED,
        "canceled": CanonicalOrderStatus.CANCELLED,
        "cancelled": CanonicalOrderStatus.CANCELLED,
        "rejected": CanonicalOrderStatus.REJECTED,
        "expired": CanonicalOrderStatus.EXPIRED,
        "unknown": CanonicalOrderStatus.UNKNOWN,
        "orderstateunknown": CanonicalOrderStatus.UNKNOWN,
    }.get(text, CanonicalOrderStatus.UNKNOWN)


def _event_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def canonical_event_id(raw: Mapping[str, Any]) -> str:
    """Return a deterministic economic identity for a raw ledger record."""

    if raw.get("event_id"):
        return str(raw["event_id"])
    event_type = str(raw.get("event_type") or "UNKNOWN").upper()
    payload = _event_payload(raw)
    identity: Any
    identity_event_type = event_type
    if event_type in {"FILL", "FILL_OBSERVED", "FILL_RECONCILED"}:
        identity_event_type = "FILL"
        identity = payload.get("fill_id") or [
            payload.get("order_id"),
            payload.get("exchange_fill_id"),
            payload.get("cumulative_quantity"),
            payload.get("cumulative_quote_amount_eur"),
        ]
    elif event_type == "ORDER_INTENT":
        identity = payload.get("idempotency_key") or payload.get("intent_id")
    elif event_type == "ORDER_STATUS_OBSERVED":
        identity = payload.get("checkpoint_id") or [
            payload.get("order_id"),
            payload.get("status"),
            payload.get("cumulative_quantity"),
            payload.get("cumulative_quote_amount_eur"),
        ]
    elif event_type in {"ORDER_ACKNOWLEDGED", "ORDER_OPEN", "ORDER_RESULT"}:
        record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
        identity = [
            payload.get("order_id") or record.get("order_id"),
            payload.get("client_order_id"),
            payload.get("intent_id"),
            payload.get("status") or record.get("status"),
            record.get("filled_quantity"),
        ]
    elif event_type in {"CANCEL_REQUESTED", "CANCEL_STATE_UNKNOWN", "CANCEL_RESOLVED"}:
        identity = payload.get("cancellation_id") or [
            payload.get("order_id"),
            payload.get("reason_code"),
            payload.get("resolution"),
        ]
    elif event_type in {
        "ORDER_CANCELLED",
        "ORDER_REJECTED",
        "ORDER_STATE_UNKNOWN",
        "ORDER_EXPIRED",
    }:
        identity = [
            payload.get("cancellation_id"),
            payload.get("order_id"),
            payload.get("client_order_id"),
            payload.get("intent_id"),
            payload.get("status"),
            payload.get("code") or payload.get("venue_error_code"),
            payload.get("reason_code"),
        ]
    elif event_type in {"RECONCILIATION_OBSERVED", "RECONCILIATION_CORRECTION"}:
        identity = payload.get("reconciliation_id") or payload
    else:
        identity = payload
    return stable_hash(
        [SCHEMA_VERSION, identity_event_type, identity],
        length=48,
    )


def _event_time(raw: Mapping[str, Any]) -> datetime:
    payload = _event_payload(raw)
    recorded = _timestamp(raw.get("recorded_at"))
    for key in (
        "filled_at",
        "exchange_updated_at",
        "exchange_created_at",
        "created_at",
        "submission_started_at",
        "cancellation_started_at",
        "received_at",
        "checked_at",
    ):
        if payload.get(key) not in (None, ""):
            try:
                return _timestamp(payload[key], fallback=recorded)
            except (OverflowError, OSError, TypeError, ValueError):
                continue
    return recorded


@dataclass(frozen=True)
class CanonicalExecutionEvent:
    event_id: str
    event_type: str
    event_timestamp: datetime
    observed_timestamp: datetime
    payload: dict[str, Any]
    source: str
    source_index: int

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        *,
        source_index: int,
    ) -> CanonicalExecutionEvent:
        payload = _event_payload(raw)
        observed = _timestamp(raw.get("recorded_at"), fallback=_event_time(raw))
        return cls(
            event_id=canonical_event_id(raw),
            event_type=str(raw.get("event_type") or "UNKNOWN").upper(),
            event_timestamp=_event_time(raw),
            observed_timestamp=observed,
            payload=copy.deepcopy(payload),
            source=str(
                payload.get("source")
                or payload.get("venue")
                or "local_execution_ledger"
            ),
            source_index=source_index,
        )


@dataclass
class CanonicalFill:
    fill_id: str
    order_id: str
    intent_id: str
    market: str
    side: str
    quantity: Decimal
    price: Decimal
    quote_amount_eur: Decimal
    fee_eur: Decimal
    fee_known: bool
    fee_asset: str | None
    filled_at: datetime
    strategy_id: str | None
    strategy_dna_hash: str | None
    signal_id: str | None
    setup_id: str | None
    source: str


@dataclass
class CanonicalRealizedPnLEvent:
    fill_id: str
    market: str
    strategy_id: str | None
    realized_pnl_eur: Decimal | None
    complete: bool
    filled_at: datetime


@dataclass
class CanonicalLot:
    lot_id: str
    market: str
    quantity: Decimal
    cost_basis_eur: Decimal
    cost_basis_known: bool
    strategy_id: str | None
    strategy_dna_hash: str | None
    signal_id: str | None
    setup_id: str | None
    entry_intent_id: str | None
    opened_at: datetime


@dataclass
class CanonicalOrder:
    canonical_id: str
    intent_id: str | None = None
    idempotency_key: str | None = None
    client_order_id: str | None = None
    exchange_order_id: str | None = None
    market: str | None = None
    side: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    status: CanonicalOrderStatus = CanonicalOrderStatus.INTENT_CREATED
    requested_quantity: Decimal = ZERO
    filled_quantity: Decimal = ZERO
    filled_quote_eur: Decimal = ZERO
    average_fill_price: Decimal = ZERO
    fee_eur: Decimal = ZERO
    fee_complete: bool = True
    estimated_price: Decimal = ZERO
    maximum_notional_eur: Decimal | None = None
    trigger_price: Decimal | None = None
    strategy_id: str | None = None
    strategy_dna_hash: str | None = None
    signal_id: str | None = None
    setup_id: str | None = None
    portfolio_decision_id: str | None = None
    portfolio_target_id: str | None = None
    risk_approval_id: str | None = None
    execution_intent_id: str | None = None
    ownership_state: OwnershipState = OwnershipState.UNKNOWN
    protective: bool = False
    source: str = "local_execution_ledger"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    unknown_reason: str | None = None

    @property
    def remaining_quantity(self) -> Decimal:
        return max(ZERO, self.requested_quantity - self.filled_quantity)


@dataclass
class CanonicalPosition:
    market: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = ZERO
    average_entry_price: Decimal = ZERO
    cost_basis_eur: Decimal = ZERO
    cost_basis_known: bool = True
    total_fees_eur: Decimal = ZERO
    realized_pnl_eur: Decimal = ZERO
    realized_pnl_complete: bool = True
    ownership_state: OwnershipState = OwnershipState.UNKNOWN
    strategy_id: str | None = None
    strategy_dna_hash: str | None = None
    signal_id: str | None = None
    setup_id: str | None = None
    lots: list[CanonicalLot] = field(default_factory=list)
    protected_quantity: Decimal = ZERO
    unprotected_quantity: Decimal = ZERO
    reserved_quantity: Decimal = ZERO
    available_quantity: Decimal = ZERO
    protection_state: ProtectionState = ProtectionState.NOT_REQUIRED
    effective_stop_price: Decimal | None = None
    planned_risk_eur: Decimal | None = None
    submitted_risk_eur: Decimal | None = None
    filled_exposure_eur: Decimal | None = None
    protected_risk_eur: Decimal | None = None
    unprotected_risk_eur: Decimal | None = None
    open_risk_eur: Decimal | None = None
    opened_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class CanonicalExecutionState:
    schema_version: str = SCHEMA_VERSION
    orders: dict[str, CanonicalOrder] = field(default_factory=dict)
    fills: dict[str, CanonicalFill] = field(default_factory=dict)
    realized_pnl_events: dict[str, CanonicalRealizedPnLEvent] = field(
        default_factory=dict
    )
    positions: dict[str, CanonicalPosition] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    portfolio_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_intents: dict[str, dict[str, Any]] = field(default_factory=dict)
    eur_total: Decimal | None = None
    eur_exchange_available: Decimal | None = None
    eur_reserved_for_orders: Decimal = ZERO
    eur_economically_committed: Decimal = ZERO
    eur_available_after_local_reservations: Decimal | None = None
    last_event_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        def ready(value: Any) -> Any:
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, datetime):
                return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, dict):
                return {str(key): ready(item) for key, item in sorted(value.items())}
            if isinstance(value, (list, tuple)):
                return [ready(item) for item in value]
            if hasattr(value, "__dataclass_fields__"):
                return ready(asdict(value))
            return value

        payload = ready(asdict(self))
        # Unknown economics must remain explicitly unknown at the serialization
        # boundary.  Internal Decimal zero placeholders are never presented as
        # observed cost basis or price evidence.
        for market, position in self.positions.items():
            if position.cost_basis_known:
                continue
            serialized = payload["positions"][market]
            serialized["cost_basis_eur"] = None
            serialized["average_entry_price"] = None
            for lot_index, lot in enumerate(position.lots):
                if not lot.cost_basis_known:
                    serialized["lots"][lot_index]["cost_basis_eur"] = None
        payload["state_hash"] = stable_hash(payload)
        payload["state_source"] = "CANONICAL_EXECUTION_LEDGER_REPLAY"
        payload["read_model_mutable"] = False
        return payload

    @property
    def state_hash(self) -> str:
        return str(self.to_dict()["state_hash"])


_EVENT_PRIORITY = {
    "PORTFOLIO_TARGET": 5,
    "RISK_APPROVAL": 7,
    "EXECUTION_INTENT": 9,
    "ORDER_INTENT": 10,
    "PROTECTIVE_ORDER_INTENT": 10,
    "ORDER_SUBMITTED": 20,
    "PROTECTIVE_ORDER_SUBMITTED": 20,
    "ORDER_ACKNOWLEDGED": 30,
    "PROTECTIVE_ORDER_ACKNOWLEDGED": 30,
    "ORDER_OPEN": 35,
    "ORDER_STATUS_OBSERVED": 40,
    "FILL": 50,
    "FILL_OBSERVED": 50,
    "FILL_RECONCILED": 50,
    "CANCEL_REQUESTED": 60,
    "ORDER_CANCELLED": 70,
    "ORDER_REJECTED": 70,
    "ORDER_EXPIRED": 70,
    "RECONCILIATION_OBSERVED": 80,
    "RECONCILIATION_CORRECTION": 90,
}


def normalize_execution_events(
    raw_events: Iterable[Mapping[str, Any]],
) -> list[CanonicalExecutionEvent]:
    events = [
        CanonicalExecutionEvent.from_raw(raw, source_index=index)
        for index, raw in enumerate(raw_events)
    ]
    return sorted(
        events,
        key=lambda event: (
            event.event_timestamp,
            _EVENT_PRIORITY.get(event.event_type, 45),
            event.observed_timestamp,
            event.event_id,
        ),
    )


def _order_from_payload(
    state: CanonicalExecutionState,
    payload: Mapping[str, Any],
    *,
    event: CanonicalExecutionEvent,
) -> CanonicalOrder:
    record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
    intent = record.get("intent") if isinstance(record.get("intent"), Mapping) else {}
    intent_id = str(payload.get("intent_id") or intent.get("intent_id") or "") or None
    client_id = str(payload.get("client_order_id") or intent.get("client_order_id") or "") or None
    exchange_id = str(payload.get("order_id") or record.get("order_id") or "") or None
    idempotency_key = str(payload.get("idempotency_key") or intent.get("idempotency_key") or "") or None
    for order in state.orders.values():
        if exchange_id and order.exchange_order_id == exchange_id:
            return order
        if client_id and order.client_order_id == client_id:
            return order
        if intent_id and order.intent_id == intent_id:
            return order
        if idempotency_key and order.idempotency_key == idempotency_key:
            return order
    canonical_id = stable_hash(
        ["CANONICAL_ORDER", intent_id, client_id, exchange_id, idempotency_key],
        length=32,
    )
    order = CanonicalOrder(canonical_id=canonical_id, created_at=event.event_timestamp)
    state.orders[canonical_id] = order
    return order


def _enrich_order(
    order: CanonicalOrder,
    payload: Mapping[str, Any],
    *,
    event: CanonicalExecutionEvent,
) -> None:
    record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
    intent = record.get("intent") if isinstance(record.get("intent"), Mapping) else {}

    def pick(key: str, *alternatives: str) -> Any:
        for selected in (key, *alternatives):
            if payload.get(selected) not in (None, ""):
                return payload[selected]
            if intent.get(selected) not in (None, ""):
                return intent[selected]
            if record.get(selected) not in (None, ""):
                return record[selected]
        return None

    order.intent_id = str(pick("intent_id") or order.intent_id or "") or None
    order.idempotency_key = str(pick("idempotency_key") or order.idempotency_key or "") or None
    order.client_order_id = str(pick("client_order_id") or order.client_order_id or "") or None
    order.exchange_order_id = str(pick("order_id") or order.exchange_order_id or "") or None
    raw_market = pick("market")
    if raw_market:
        order.market = normalize_market(str(raw_market))
    raw_side = pick("side")
    if raw_side:
        order.side = str(raw_side).upper()
    raw_type = pick("order_type")
    if raw_type:
        order.order_type = str(raw_type).upper()
    raw_tif = pick("time_in_force")
    if raw_tif:
        order.time_in_force = str(raw_tif).upper()
    quantity = pick("quantity")
    if quantity not in (None, ""):
        order.requested_quantity = max(order.requested_quantity, _decimal(quantity))
    estimate = pick("estimated_price", "limit_price")
    if estimate not in (None, ""):
        order.estimated_price = _decimal(estimate)
    maximum = pick("maximum_notional_eur")
    if maximum not in (None, ""):
        order.maximum_notional_eur = _decimal(maximum)
    trigger = pick("trigger_price", "trigger_amount")
    if trigger not in (None, ""):
        order.trigger_price = _decimal(trigger)
    for attribute, key in (
        ("strategy_id", "strategy_id"),
        ("strategy_dna_hash", "strategy_dna_hash"),
        ("signal_id", "signal_id"),
        ("setup_id", "setup_id"),
        ("portfolio_decision_id", "portfolio_decision_id"),
        ("portfolio_target_id", "portfolio_target_id"),
        ("risk_approval_id", "risk_approval_id"),
        ("execution_intent_id", "execution_intent_id"),
    ):
        value = pick(key)
        if value not in (None, ""):
            setattr(order, attribute, str(value))
    order.ownership_state = (
        OwnershipState.KNOWN if order.strategy_id else OwnershipState.UNKNOWN
    )
    reason_codes = {str(value).upper() for value in (pick("reason_codes") or [])}
    order.protective = bool(
        order.protective
        or order.order_type == "STOP_LOSS"
        or "PROTECTIVE" in str(order.idempotency_key or "").upper()
        or any("PROTECT" in value for value in reason_codes)
        or event.event_type.startswith("PROTECTIVE_ORDER_")
    )
    order.updated_at = max(order.updated_at or event.event_timestamp, event.event_timestamp)
    order.source = event.source


def _apply_status(order: CanonicalOrder, candidate: CanonicalOrderStatus) -> None:
    current = order.status
    # A final fill is the strongest economic fact.  A late cancellation or
    # stale open callback cannot remove already observed fills.
    if current is CanonicalOrderStatus.FILLED:
        return
    if candidate is CanonicalOrderStatus.FILLED:
        order.status = candidate
        return
    if current in {CanonicalOrderStatus.CANCELLED, CanonicalOrderStatus.REJECTED, CanonicalOrderStatus.EXPIRED}:
        if candidate in ACTIVE_STATUSES:
            return
    if current is CanonicalOrderStatus.PARTIALLY_FILLED and candidate in {
        CanonicalOrderStatus.INTENT_CREATED,
        CanonicalOrderStatus.SUBMITTED,
        CanonicalOrderStatus.ACKNOWLEDGED,
        CanonicalOrderStatus.OPEN,
    }:
        return
    lifecycle_rank = {
        CanonicalOrderStatus.INTENT_CREATED: 0,
        CanonicalOrderStatus.SUBMITTED: 1,
        CanonicalOrderStatus.ACKNOWLEDGED: 2,
        CanonicalOrderStatus.OPEN: 3,
        CanonicalOrderStatus.CANCEL_REQUESTED: 4,
        CanonicalOrderStatus.PARTIALLY_FILLED: 5,
    }
    if current in lifecycle_rank and candidate in lifecycle_rank:
        if lifecycle_rank[candidate] < lifecycle_rank[current]:
            return
    order.status = candidate


def _ownership_key(
    strategy_id: str | None,
    strategy_dna_hash: str | None,
    signal_id: str | None,
    setup_id: str | None,
) -> tuple[str, str, str, str]:
    return (
        strategy_id or "UNKNOWN",
        strategy_dna_hash or "UNKNOWN",
        signal_id or "UNKNOWN",
        setup_id or "UNKNOWN",
    )


def _position(state: CanonicalExecutionState, market: str) -> CanonicalPosition:
    normalized = normalize_market(market)
    if normalized not in state.positions:
        base, quote = normalized.split("-")
        state.positions[normalized] = CanonicalPosition(
            market=normalized,
            base_asset=base,
            quote_asset=quote,
        )
    return state.positions[normalized]


def _refresh_position(position: CanonicalPosition, state: CanonicalExecutionState) -> None:
    position.lots = [lot for lot in position.lots if lot.quantity > ZERO]
    position.quantity = sum((lot.quantity for lot in position.lots), ZERO)
    position.cost_basis_eur = sum((lot.cost_basis_eur for lot in position.lots), ZERO)
    position.cost_basis_known = all(lot.cost_basis_known for lot in position.lots)
    position.average_entry_price = (
        position.cost_basis_eur / position.quantity
        if position.quantity > ZERO and position.cost_basis_known
        else ZERO
    )
    owners = {
        _ownership_key(lot.strategy_id, lot.strategy_dna_hash, lot.signal_id, lot.setup_id)
        for lot in position.lots
    }
    if not owners or any(owner[0] == "UNKNOWN" for owner in owners):
        position.ownership_state = OwnershipState.UNKNOWN
    elif len(owners) == 1:
        position.ownership_state = OwnershipState.KNOWN
    else:
        position.ownership_state = OwnershipState.MIXED
    if len(owners) == 1:
        owner = next(iter(owners))
        position.strategy_id = None if owner[0] == "UNKNOWN" else owner[0]
        position.strategy_dna_hash = None if owner[1] == "UNKNOWN" else owner[1]
        position.signal_id = None if owner[2] == "UNKNOWN" else owner[2]
        position.setup_id = None if owner[3] == "UNKNOWN" else owner[3]
    else:
        position.strategy_id = None
        position.strategy_dna_hash = None
        position.signal_id = None
        position.setup_id = None

    open_sells = [
        order
        for order in state.orders.values()
        if order.market == position.market
        and order.side == "SELL"
        and order.status in ACTIVE_STATUSES
        and order.remaining_quantity > ZERO
    ]
    raw_reserved = sum((order.remaining_quantity for order in open_sells), ZERO)
    position.reserved_quantity = min(position.quantity, raw_reserved)
    position.available_quantity = max(ZERO, position.quantity - position.reserved_quantity)
    if raw_reserved > position.quantity:
        state.evidence_gaps.append(
            {
                "code": "SELL_RESERVATION_EXCEEDS_POSITION",
                "market": position.market,
                "position_quantity": str(position.quantity),
                "raw_reserved_quantity": str(raw_reserved),
            }
        )

    protective = [order for order in open_sells if order.protective]
    unknown_protective = [
        order
        for order in state.orders.values()
        if order.market == position.market
        and order.protective
        and order.status is CanonicalOrderStatus.UNKNOWN
    ]
    intended_protective = [
        order
        for order in state.orders.values()
        if order.market == position.market
        and order.protective
        and order.status in {
            CanonicalOrderStatus.INTENT_CREATED,
            CanonicalOrderStatus.SUBMITTED,
        }
    ]
    if position.quantity <= ZERO:
        position.protected_quantity = ZERO
        position.unprotected_quantity = ZERO
        position.protection_state = ProtectionState.NOT_REQUIRED
        position.effective_stop_price = None
    elif unknown_protective:
        position.protected_quantity = ZERO
        position.unprotected_quantity = position.quantity
        position.protection_state = ProtectionState.UNKNOWN
        position.effective_stop_price = None
    elif protective:
        raw_protected = sum((order.remaining_quantity for order in protective), ZERO)
        position.protected_quantity = min(position.quantity, raw_protected)
        position.unprotected_quantity = max(ZERO, position.quantity - position.protected_quantity)
        stop_prices = {order.trigger_price for order in protective if order.trigger_price is not None}
        position.effective_stop_price = max(stop_prices) if stop_prices else None
        if raw_protected > position.quantity or len(stop_prices) > 1:
            position.protection_state = ProtectionState.CONFLICT
        elif position.protected_quantity == position.quantity and position.effective_stop_price:
            position.protection_state = ProtectionState.CONFIRMED_ACTIVE
        else:
            position.protection_state = ProtectionState.PARTIAL
    elif intended_protective:
        position.protected_quantity = ZERO
        position.unprotected_quantity = position.quantity
        position.protection_state = ProtectionState.INTENDED
        position.effective_stop_price = None
    else:
        historical = [
            order
            for order in state.orders.values()
            if order.market == position.market and order.protective
        ]
        position.protected_quantity = ZERO
        position.unprotected_quantity = position.quantity
        position.effective_stop_price = None
        if any(order.status is CanonicalOrderStatus.REJECTED for order in historical):
            position.protection_state = ProtectionState.REJECTED
        elif any(order.status is CanonicalOrderStatus.CANCELLED for order in historical):
            position.protection_state = ProtectionState.CANCELLED
        elif any(order.status is CanonicalOrderStatus.FILLED for order in historical):
            position.protection_state = ProtectionState.FILLED
        else:
            position.protection_state = ProtectionState.MISSING

    position.filled_exposure_eur = (
        position.cost_basis_eur if position.cost_basis_known else None
    )
    if position.cost_basis_known:
        if position.effective_stop_price is not None:
            position.protected_risk_eur = max(
                ZERO,
                position.average_entry_price - position.effective_stop_price,
            ) * position.protected_quantity
        else:
            position.protected_risk_eur = ZERO
        position.unprotected_risk_eur = (
            position.average_entry_price * position.unprotected_quantity
        )
        position.open_risk_eur = (
            position.protected_risk_eur + position.unprotected_risk_eur
        )
    else:
        position.protected_risk_eur = None
        position.unprotected_risk_eur = None
        position.open_risk_eur = None


def _apply_fill(
    state: CanonicalExecutionState,
    event: CanonicalExecutionEvent,
) -> None:
    payload = event.payload
    fill_id = str(payload.get("fill_id") or event.event_id)
    if fill_id in state.fills:
        return
    order = _order_from_payload(state, payload, event=event)
    _enrich_order(order, payload, event=event)
    market = normalize_market(str(payload.get("market") or order.market or ""))
    side = str(payload.get("side") or order.side or "").upper()
    quantity = _decimal(payload.get("quantity"))
    price = _decimal(payload.get("price"))
    if quantity <= ZERO or price <= ZERO or side not in {"BUY", "SELL"}:
        state.evidence_gaps.append(
            {"code": "INVALID_FILL_EVIDENCE", "event_id": event.event_id}
        )
        return
    quote = _decimal(payload.get("quote_amount_eur"), default=quantity * price)
    fee = _decimal(payload.get("fee_eur"))
    fee_known = payload.get("fee_known") is not False
    fill = CanonicalFill(
        fill_id=fill_id,
        order_id=str(payload.get("order_id") or order.exchange_order_id or order.canonical_id),
        intent_id=str(payload.get("intent_id") or order.intent_id or "UNKNOWN"),
        market=market,
        side=side,
        quantity=quantity,
        price=price,
        quote_amount_eur=quote,
        fee_eur=fee,
        fee_known=fee_known,
        fee_asset=str(payload.get("fee_asset") or "EUR") if fee_known else None,
        filled_at=_timestamp(payload.get("filled_at"), fallback=event.event_timestamp),
        strategy_id=str(payload.get("strategy_id") or order.strategy_id or "") or None,
        strategy_dna_hash=str(payload.get("strategy_dna_hash") or order.strategy_dna_hash or "") or None,
        signal_id=str(payload.get("signal_id") or order.signal_id or "") or None,
        setup_id=str(payload.get("setup_id") or order.setup_id or "") or None,
        source=event.source,
    )
    state.fills[fill_id] = fill
    order.exchange_order_id = fill.order_id
    order.filled_quantity += quantity
    order.filled_quote_eur += quote
    order.fee_eur += fee
    order.fee_complete = order.fee_complete and fee_known
    order.average_fill_price = (
        order.filled_quote_eur / order.filled_quantity
        if order.filled_quantity > ZERO
        else ZERO
    )
    if order.requested_quantity > ZERO and order.filled_quantity >= order.requested_quantity:
        _apply_status(order, CanonicalOrderStatus.FILLED)
    else:
        _apply_status(order, CanonicalOrderStatus.PARTIALLY_FILLED)
    order.updated_at = event.event_timestamp

    position = _position(state, market)
    position.total_fees_eur += fee
    position.updated_at = fill.filled_at
    if side == "BUY":
        position.lots.append(
            CanonicalLot(
                lot_id=fill_id,
                market=market,
                quantity=quantity,
                cost_basis_eur=quote + fee,
                cost_basis_known=fee_known,
                strategy_id=fill.strategy_id,
                strategy_dna_hash=fill.strategy_dna_hash,
                signal_id=fill.signal_id,
                setup_id=fill.setup_id,
                entry_intent_id=fill.intent_id,
                opened_at=fill.filled_at,
            )
        )
        position.opened_at = min(
            [lot.opened_at for lot in position.lots],
            default=fill.filled_at,
        )
    else:
        remaining = quantity
        realized = ZERO
        realized_complete = fee_known
        for lot in position.lots:
            if remaining <= ZERO:
                break
            consumed = min(lot.quantity, remaining)
            if consumed <= ZERO:
                continue
            ratio = consumed / quantity
            allocated_exit_fee = fee * ratio
            unit_cost = (
                lot.cost_basis_eur / lot.quantity
                if lot.quantity > ZERO and lot.cost_basis_known
                else ZERO
            )
            if lot.cost_basis_known and fee_known:
                realized += consumed * price - allocated_exit_fee - consumed * unit_cost
            else:
                position.realized_pnl_complete = False
                realized_complete = False
            original_quantity = lot.quantity
            lot.quantity -= consumed
            if original_quantity > ZERO:
                lot.cost_basis_eur *= lot.quantity / original_quantity
            remaining -= consumed
        if remaining > ZERO:
            position.realized_pnl_complete = False
            realized_complete = False
            state.evidence_gaps.append(
                {
                    "code": "SELL_WITHOUT_RECONSTRUCTABLE_COST_BASIS",
                    "fill_id": fill_id,
                    "market": market,
                    "unmatched_quantity": str(remaining),
                    "strategy_id": fill.strategy_id,
                }
            )
        position.realized_pnl_eur += realized
        state.realized_pnl_events[fill_id] = CanonicalRealizedPnLEvent(
            fill_id=fill_id,
            market=market,
            strategy_id=fill.strategy_id,
            realized_pnl_eur=realized if realized_complete else None,
            complete=realized_complete,
            filled_at=fill.filled_at,
        )
    _refresh_position(position, state)


def _apply_reconciliation(
    state: CanonicalExecutionState,
    event: CanonicalExecutionEvent,
) -> None:
    payload = event.payload
    balances = payload.get("balances")
    if isinstance(balances, list):
        rows = [row for row in balances if isinstance(row, Mapping)]
    elif isinstance(balances, Mapping):
        rows = [
            {"symbol": symbol, **(dict(value) if isinstance(value, Mapping) else {"available": value})}
            for symbol, value in balances.items()
        ]
    else:
        rows = []
    for row in rows:
        symbol = str(row.get("symbol") or row.get("asset") or "").upper()
        available = _decimal(row.get("available"))
        in_order = _decimal(row.get("inOrder") or row.get("in_order"))
        total = available + in_order
        if symbol == "EUR":
            state.eur_total = total
            state.eur_exchange_available = available
            continue
        market = f"{symbol}-EUR"
        position = _position(state, market)
        if total > position.quantity:
            difference = total - position.quantity
            position.lots.append(
                CanonicalLot(
                    lot_id=f"reconciliation:{event.event_id}:{symbol}",
                    market=market,
                    quantity=difference,
                    cost_basis_eur=ZERO,
                    cost_basis_known=False,
                    strategy_id=None,
                    strategy_dna_hash=None,
                    signal_id=None,
                    setup_id=None,
                    entry_intent_id=None,
                    opened_at=event.event_timestamp,
                )
            )
            state.evidence_gaps.append(
                {
                    "code": "EXCHANGE_BALANCE_WITH_UNKNOWN_OWNERSHIP",
                    "market": market,
                    "quantity": str(difference),
                    "reconciliation_event_id": event.event_id,
                }
            )
        elif total < position.quantity:
            state.evidence_gaps.append(
                {
                    "code": "EXCHANGE_BALANCE_BELOW_CANONICAL_FILLS",
                    "market": market,
                    "canonical_quantity": str(position.quantity),
                    "exchange_quantity": str(total),
                    "reconciliation_event_id": event.event_id,
                }
            )
        _refresh_position(position, state)

    remote_orders = payload.get("remote_open_orders")
    if isinstance(remote_orders, list):
        observed_orders = [
            row for row in remote_orders if isinstance(row, Mapping)
        ]
    else:
        observed_orders = []
    for observed in observed_orders:
        exchange_id = str(observed.get("order_id") or "") or None
        client_id = str(observed.get("client_order_id") or "") or None
        matched_local_order = any(
            (exchange_id and order.exchange_order_id == exchange_id)
            or (client_id and order.client_order_id == client_id)
            for order in state.orders.values()
        )
        order = _order_from_payload(state, observed, event=event)
        _enrich_order(order, observed, event=event)
        observed_filled = _decimal(observed.get("filled_quantity"))
        order.filled_quantity = max(order.filled_quantity, observed_filled)
        status = _normalized_status(observed.get("status"))
        _apply_status(
            order,
            status
            if status is not CanonicalOrderStatus.UNKNOWN
            else CanonicalOrderStatus.OPEN,
        )
        if not matched_local_order:
            gap = {
                "code": "REMOTE_OPEN_ORDER_WITH_UNKNOWN_LOCAL_INTENT",
                "order_id": exchange_id,
                "client_order_id": client_id,
                "market": order.market,
                "side": order.side,
                "reconciliation_event_id": event.event_id,
            }
            if gap not in state.evidence_gaps:
                state.evidence_gaps.append(gap)
        if observed_filled > ZERO and not any(
            fill.order_id == exchange_id for fill in state.fills.values()
        ):
            state.evidence_gaps.append(
                {
                    "code": "CUMULATIVE_FILL_WITHOUT_FILL_EVENTS",
                    "order_id": exchange_id,
                    "observed_filled_quantity": str(observed_filled),
                    "reconciliation_event_id": event.event_id,
                }
            )


def _refresh_capital(state: CanonicalExecutionState) -> None:
    reserved = ZERO
    for order in state.orders.values():
        if order.side != "BUY" or order.status not in {
            *ACTIVE_STATUSES,
            CanonicalOrderStatus.UNKNOWN,
        }:
            continue
        requested = order.maximum_notional_eur
        if requested is None:
            requested = order.requested_quantity * order.estimated_price
        reserved += max(ZERO, requested - order.filled_quote_eur)
    state.eur_reserved_for_orders = reserved
    state.eur_economically_committed = sum(
        (
            position.cost_basis_eur
            for position in state.positions.values()
            if position.cost_basis_known
        ),
        ZERO,
    )
    state.eur_available_after_local_reservations = (
        max(ZERO, state.eur_exchange_available - reserved)
        if state.eur_exchange_available is not None
        else None
    )


def _assert_invariants(state: CanonicalExecutionState) -> None:
    if state.eur_reserved_for_orders < ZERO:
        raise ValueError("canonical reserved capital became negative")
    for position in state.positions.values():
        if position.quantity < ZERO:
            raise ValueError(f"canonical position became negative: {position.market}")
        if position.protected_quantity < ZERO:
            raise ValueError(f"canonical protected quantity became negative: {position.market}")
        if position.protected_quantity > position.quantity:
            raise ValueError(f"canonical protection exceeds position: {position.market}")
        if position.reserved_quantity < ZERO:
            raise ValueError(f"canonical reserved quantity became negative: {position.market}")
        if position.quantity == ZERO and position.open_risk_eur not in (None, ZERO):
            raise ValueError(f"closed position retains economic risk: {position.market}")
        if position.quantity == ZERO and position.protected_quantity != ZERO:
            raise ValueError(f"closed position retains protection: {position.market}")


def reduce_execution_event(
    previous: CanonicalExecutionState,
    event: CanonicalExecutionEvent,
) -> CanonicalExecutionState:
    """Pure deterministic reducer: previous state + immutable event -> state."""

    if event.event_id in previous.processed_event_ids:
        return previous
    state = copy.deepcopy(previous)
    event_type = event.event_type
    payload = event.payload
    if event_type == "PORTFOLIO_TARGET":
        target_id = str(payload.get("target_id") or "")
        if not target_id:
            state.evidence_gaps.append(
                {"code": "PORTFOLIO_TARGET_ID_MISSING", "event_id": event.event_id}
            )
        else:
            state.portfolio_targets.setdefault(target_id, copy.deepcopy(payload))
    elif event_type == "RISK_APPROVAL":
        approval_id = str(payload.get("approval_id") or "")
        target_id = str(payload.get("target_id") or "")
        if not approval_id or target_id not in state.portfolio_targets:
            state.evidence_gaps.append(
                {
                    "code": "RISK_APPROVAL_TARGET_UNKNOWN",
                    "event_id": event.event_id,
                    "approval_id": approval_id or None,
                    "target_id": target_id or None,
                }
            )
        else:
            state.risk_approvals.setdefault(approval_id, copy.deepcopy(payload))
    elif event_type == "EXECUTION_INTENT":
        execution_intent_id = str(payload.get("execution_intent_id") or "")
        target_id = str(payload.get("target_id") or "")
        approval_id = str(payload.get("risk_approval_id") or "")
        if (
            not execution_intent_id
            or target_id not in state.portfolio_targets
            or approval_id not in state.risk_approvals
        ):
            state.evidence_gaps.append(
                {
                    "code": "EXECUTION_INTENT_CHAIN_INCOMPLETE",
                    "event_id": event.event_id,
                    "execution_intent_id": execution_intent_id or None,
                    "target_id": target_id or None,
                    "risk_approval_id": approval_id or None,
                }
            )
        else:
            state.execution_intents.setdefault(
                execution_intent_id,
                copy.deepcopy(payload),
            )
    elif event_type in {
        "ORDER_INTENT",
        "ORDER_SUBMITTED",
        "PROTECTIVE_ORDER_INTENT",
        "PROTECTIVE_ORDER_SUBMITTED",
    }:
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(
            order,
            CanonicalOrderStatus.SUBMITTED
            if event_type.endswith("SUBMITTED")
            else CanonicalOrderStatus.INTENT_CREATED,
        )
    elif event_type in {"ORDER_ACKNOWLEDGED", "PROTECTIVE_ORDER_ACKNOWLEDGED"}:
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        status = _normalized_status(payload.get("status"))
        _apply_status(
            order,
            status if status is not CanonicalOrderStatus.UNKNOWN else CanonicalOrderStatus.ACKNOWLEDGED,
        )
    elif event_type == "ORDER_RESULT":
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
        _apply_status(order, _normalized_status(record.get("status")))
    elif event_type == "ORDER_OPEN":
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.OPEN)
    elif event_type == "ORDER_STATUS_OBSERVED":
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, _normalized_status(payload.get("status")))
    elif event_type in {"FILL", "FILL_OBSERVED", "FILL_RECONCILED"}:
        _apply_fill(state, event)
    elif event_type == "CANCEL_REQUESTED":
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.CANCEL_REQUESTED)
    elif event_type in {"ORDER_CANCELLED", "PROTECTIVE_ORDER_CANCELLED"}:
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.CANCELLED)
    elif event_type in {"ORDER_REJECTED", "PROTECTIVE_ORDER_REJECTED"}:
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.REJECTED)
    elif event_type == "ORDER_EXPIRED":
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.EXPIRED)
    elif event_type in {"ORDER_STATE_UNKNOWN", "CANCEL_STATE_UNKNOWN"}:
        order = _order_from_payload(state, payload, event=event)
        _enrich_order(order, payload, event=event)
        _apply_status(order, CanonicalOrderStatus.UNKNOWN)
        order.unknown_reason = str(payload.get("reason_code") or "UNKNOWN")
    elif event_type in {"RECONCILIATION_OBSERVED", "RECONCILIATION_CORRECTION"}:
        _apply_reconciliation(state, event)

    state.processed_event_ids.add(event.event_id)
    state.last_event_timestamp = max(
        state.last_event_timestamp or event.event_timestamp,
        event.event_timestamp,
    )
    for position in state.positions.values():
        _refresh_position(position, state)
    _refresh_capital(state)
    _assert_invariants(state)
    return state


def replay_execution_events(
    raw_events: Iterable[Mapping[str, Any]],
    *,
    initial_state: CanonicalExecutionState | None = None,
) -> CanonicalExecutionState:
    state = copy.deepcopy(initial_state) if initial_state is not None else CanonicalExecutionState()
    for event in normalize_execution_events(raw_events):
        state = reduce_execution_event(state, event)
    return state


def canonical_state_fingerprint(state: CanonicalExecutionState) -> str:
    return stable_hash(state.to_dict())


def assert_replay_deterministic(raw_events: Iterable[Mapping[str, Any]]) -> str:
    materialized = [copy.deepcopy(dict(event)) for event in raw_events]
    first = replay_execution_events(materialized)
    second = replay_execution_events(materialized)
    first_json = stable_json(first.to_dict())
    second_json = stable_json(second.to_dict())
    if first_json != second_json:
        raise AssertionError("canonical execution replay is not deterministic")
    return first.state_hash


__all__ = [
    "CanonicalExecutionEvent",
    "CanonicalExecutionState",
    "CanonicalFill",
    "CanonicalLot",
    "CanonicalOrder",
    "CanonicalOrderStatus",
    "CanonicalPosition",
    "CanonicalRealizedPnLEvent",
    "OwnershipState",
    "ProtectionState",
    "assert_replay_deterministic",
    "canonical_event_id",
    "canonical_state_fingerprint",
    "normalize_execution_events",
    "reduce_execution_event",
    "replay_execution_events",
]
