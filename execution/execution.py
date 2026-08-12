"""Paper broker and capability-gated Bitvavo crypto spot execution."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import aiohttp
from pydantic import SecretStr

from config.settings import Settings
from core.contracts import (
    ExecutionBlocked,
    Fill,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
    normalize_market,
    utc_now,
)
from portfolio.targets import CanonicalExecutionChain, validate_order_against_chain
from utils.common import append_jsonl, atomic_write_json, stable_hash

BITVAVO_BASE_URL = "https://api.bitvavo.com/v2"


def _normalized_venue_timestamp(value: object) -> str | None:
    """Normalize Bitvavo ISO or epoch-millisecond timestamps to UTC ISO."""

    if value is None or value == "":
        return None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        numeric = None
    if numeric is not None and numeric.is_finite():
        # Bitvavo documents order timestamps as Unix milliseconds.  Accept
        # seconds as a defensive fallback for fixtures and archived payloads.
        seconds = numeric / (Decimal("1000") if abs(numeric) >= Decimal("1e11") else Decimal("1"))
        try:
            return datetime.fromtimestamp(float(seconds), tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    try:
        selected = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if selected.tzinfo is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class ExecutionMarketRules:
    minimum_order_amount: Decimal = Decimal("0")
    minimum_order_value_eur: Decimal = Decimal("5")
    quantity_decimals: int = 8
    notional_decimals: int = 2
    tick_size: Decimal = Decimal("0.00000001")
    supported_order_types: tuple[str, ...] = (
        "market",
        "limit",
        "stopLoss",
    )

    def amount(self, value: Decimal) -> Decimal:
        quantum = Decimal(1).scaleb(-self.quantity_decimals)
        return value.quantize(quantum, rounding=ROUND_DOWN)

    def amount_up(self, value: Decimal) -> Decimal:
        quantum = Decimal(1).scaleb(-self.quantity_decimals)
        return value.quantize(quantum, rounding=ROUND_UP)

    def price(self, value: Decimal) -> Decimal:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        ticks = (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN)
        return ticks * self.tick_size

    def price_up(self, value: Decimal) -> Decimal:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        ticks = (value / self.tick_size).to_integral_value(rounding=ROUND_UP)
        return ticks * self.tick_size


@dataclass(frozen=True)
class EntryOrderPlan:
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None
    time_in_force: OrderTimeInForce
    planned_notional_eur: Decimal
    execution_policy: str
    fallback_reason: str | None = None


def minimum_protectable_entry_notional(
    *,
    entry_price: Decimal,
    stop_price: Decimal,
    rules: ExecutionMarketRules,
    quote_minimum_buffer: Decimal = Decimal("1.15"),
) -> Decimal:
    """Return the smallest entry notional whose full stop remains sellable."""

    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        raise ExecutionBlocked("protective stop prices are invalid")
    if quote_minimum_buffer < 1:
        raise ExecutionBlocked("protective quote-minimum buffer is invalid")
    safe_quote_minimum = (
        rules.minimum_order_value_eur * quote_minimum_buffer
    )
    minimum_quantity = max(
        rules.minimum_order_amount,
        safe_quote_minimum / stop_price,
    )
    quantity = rules.amount_up(minimum_quantity)
    quantum = Decimal(1).scaleb(-rules.notional_decimals)
    return (quantity * entry_price).quantize(quantum, rounding=ROUND_UP)


def quantity_is_protectable_at_stop(
    *,
    quantity: Decimal,
    stop_price: Decimal,
    rules: ExecutionMarketRules,
    quote_minimum_buffer: Decimal = Decimal("1.15"),
) -> bool:
    """Validate both base and buffered quote minima at the actual stop."""

    return (
        quantity >= rules.minimum_order_amount
        and stop_price > 0
        and quantity * stop_price
        >= rules.minimum_order_value_eur * quote_minimum_buffer
    )


def market_rules_from_bitvavo_metadata(
    payload: Mapping[str, Any],
) -> ExecutionMarketRules:
    """Normalize current Bitvavo tick/amount rules into execution rules."""

    try:
        quantity_decimals = int(payload.get("quantityDecimals") or 8)
        notional_decimals = int(payload.get("notionalDecimals") or 2)
        tick_size = Decimal(str(payload.get("tickSize") or "0.00000001"))
        minimum_base = Decimal(
            str(payload.get("minOrderInBaseAsset") or "0")
        )
        minimum_quote = Decimal(
            str(payload.get("minOrderInQuoteAsset") or "5")
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExecutionBlocked("invalid Bitvavo market execution rules") from exc
    if (
        quantity_decimals < 0
        or notional_decimals < 0
        or tick_size <= 0
        or minimum_base < 0
        or minimum_quote <= 0
    ):
        raise ExecutionBlocked("invalid Bitvavo market execution rules")
    order_types = tuple(
        str(value)
        for value in payload.get("orderTypes") or ("market", "limit")
    )
    return ExecutionMarketRules(
        minimum_order_amount=minimum_base,
        minimum_order_value_eur=minimum_quote,
        quantity_decimals=quantity_decimals,
        notional_decimals=notional_decimals,
        tick_size=tick_size,
        supported_order_types=order_types,
    )


def plan_bounded_entry_order(
    *,
    requested_notional_eur: Decimal,
    best_ask: Decimal,
    estimated_average_price: Decimal,
    maximum_slippage_bps: Decimal,
    rules: ExecutionMarketRules,
    limit_enabled: bool,
    time_in_force: OrderTimeInForce = OrderTimeInForce.IOC,
    price_buffer_bps: Decimal = Decimal("1"),
    market_fallback_enabled: bool = True,
) -> EntryOrderPlan:
    """Choose a bounded GTC/IOC/FOK limit when venue rounding permits it.

    The planner never increases the requested notional to satisfy a venue
    minimum.  At the exact EUR 5 micro-canary cap this often means using the
    existing quote-denominated market buy instead of silently breaching risk.
    """

    if requested_notional_eur <= 0 or best_ask <= 0:
        raise ExecutionBlocked("entry order plan has non-positive inputs")
    if maximum_slippage_bps < 0 or price_buffer_bps < 0:
        raise ExecutionBlocked("entry order plan has invalid price limits")
    market_quantity = requested_notional_eur / best_ask
    if not limit_enabled:
        return EntryOrderPlan(
            order_type=OrderType.MARKET,
            quantity=market_quantity,
            limit_price=None,
            time_in_force=OrderTimeInForce.GTC,
            planned_notional_eur=requested_notional_eur,
            execution_policy="QUOTE_MARKET_LIQUIDITY_PREFLIGHT",
            fallback_reason="LIMIT_ENTRIES_DISABLED",
        )
    if time_in_force not in {
        OrderTimeInForce.GTC,
        OrderTimeInForce.IOC,
        OrderTimeInForce.FOK,
    }:
        raise ExecutionBlocked(
            "autonomous entry limit must be GTC, IOC or FOK"
        )
    reference = max(best_ask, estimated_average_price)
    maximum_price = best_ask * (
        Decimal("1") + maximum_slippage_bps / Decimal("10000")
    )
    buffered = reference * (
        Decimal("1") + price_buffer_bps / Decimal("10000")
    )
    limit_price = rules.price_up(min(buffered, maximum_price))
    if limit_price > maximum_price:
        limit_price = rules.price(maximum_price)
    quantity = rules.amount(requested_notional_eur / limit_price)
    notional = quantity * limit_price
    limit_feasible = (
        limit_price >= best_ask
        and quantity >= rules.minimum_order_amount
        and notional >= rules.minimum_order_value_eur
        and notional <= requested_notional_eur
    )
    if limit_feasible:
        return EntryOrderPlan(
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=time_in_force,
            planned_notional_eur=notional,
            execution_policy="BOUNDED_MARKETABLE_LIMIT",
        )
    if not market_fallback_enabled:
        raise ExecutionBlocked("venue rounding makes bounded limit infeasible")
    return EntryOrderPlan(
        order_type=OrderType.MARKET,
        quantity=market_quantity,
        limit_price=None,
        time_in_force=OrderTimeInForce.GTC,
        planned_notional_eur=requested_notional_eur,
        execution_policy="QUOTE_MARKET_LIQUIDITY_PREFLIGHT",
        fallback_reason="VENUE_MINIMUM_OR_ROUNDING_REQUIRES_QUOTE_MARKET",
    )


@dataclass(frozen=True)
class ReconciliationResult:
    healthy: bool
    reason_codes: tuple[str, ...]
    checked_at: datetime
    local_open_orders: int
    remote_open_orders: int | None


@dataclass(frozen=True)
class LiveCapability:
    token: str
    checked_at: datetime
    allowed_markets: tuple[str, ...]
    maximum_order_eur: Decimal
    maximum_total_eur: Decimal
    maximum_open_positions: int
    maximum_new_orders_per_day: int = 1


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    failures: tuple[str, ...]
    capability: LiveCapability | None
    checked_at: datetime


class DurableLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.path,
            {
                "event_type": event_type,
                "recorded_at": datetime.now(UTC),
                "payload": payload,
            },
        )

    def events(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                raise ReconciliationRequired("execution ledger contains invalid JSON")
            events.append(payload)
        return events

    def idempotency_keys(self) -> set[str]:
        return {
            str(event["payload"].get("idempotency_key"))
            for event in self.events()
            if event.get("event_type") == "ORDER_INTENT"
        }

    def canonical_state(self):
        """Deterministically reduce this immutable ledger into financial truth."""

        from execution.canonical_state import replay_execution_events

        return replay_execution_events(self.events())

    def persist_canonical_state(self, path: Path) -> dict[str, Any]:
        """Write a rebuildable read model; the ledger remains authoritative."""

        state = self.canonical_state()
        payload = state.to_dict()
        atomic_write_json(path, payload)
        return payload

    def canonical_replay_report(self) -> dict[str, Any]:
        from execution.canonical_state import assert_replay_deterministic

        started = time.perf_counter()
        events = self.events()
        state_hash = assert_replay_deterministic(events)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        state = self.canonical_state()
        return {
            "schema_version": "canonical_execution_replay_report_v1",
            "source": str(self.path),
            "raw_event_count": len(events),
            "unique_event_count": len(state.processed_event_ids),
            "order_count": len(state.orders),
            "fill_count": len(state.fills),
            "position_count": sum(
                position.quantity > 0 for position in state.positions.values()
            ),
            "evidence_gap_count": len(state.evidence_gaps),
            "state_hash": state_hash,
            "deterministic": True,
            "elapsed_ms": elapsed_ms,
            "orders_generated": 0,
            "orders_submitted": 0,
            "private_exchange_requests": 0,
        }


class PaperBroker:
    """Deterministic spot broker; it has no network or real-order capability."""

    def __init__(
        self,
        *,
        initial_balances: dict[str, Decimal] | None = None,
        market_rules: dict[str, ExecutionMarketRules] | None = None,
        fee_fraction: Decimal = Decimal("0.0025"),
        slippage_bps: Decimal = Decimal("8"),
        spread_bps: Decimal = Decimal("5"),
        latency_ms: int = 100,
        ledger_path: Path | None = None,
    ) -> None:
        self.initial_balances = {
            key.upper(): Decimal(value)
            for key, value in (initial_balances or {"EUR": Decimal("2000")}).items()
        }
        self.balances = dict(self.initial_balances)
        self.market_rules = market_rules or {}
        self.fee_fraction = fee_fraction
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.latency_ms = max(0, latency_ms)
        self.ledger = DurableLedger(ledger_path or Path("output/checkpoints/paper_execution.jsonl"))
        self.orders: dict[str, OrderRecord] = {}
        self.fills: list[Fill] = []
        self.by_idempotency: dict[str, str] = {}
        self.open_orders: dict[str, OrderIntent] = {}
        self.reserved_eur: dict[str, Decimal] = {}
        self.reserved_assets: dict[str, tuple[str, Decimal]] = {}
        self._replay_ledger()

    def __repr__(self) -> str:
        return f"PaperBroker(balances={self.balances!r}, open_orders={len(self.open_orders)})"

    def _replay_ledger(self) -> None:
        for event in self.ledger.events():
            event_type = event.get("event_type")
            payload = event.get("payload") or {}
            if event_type == "FILL":
                try:
                    fill = Fill(
                        fill_id=str(payload["fill_id"]),
                        order_id=str(payload["order_id"]),
                        intent_id=str(payload["intent_id"]),
                        market=str(payload["market"]),
                        side=OrderSide(str(payload["side"])),
                        quantity=Decimal(str(payload["quantity"])),
                        price=Decimal(str(payload["price"])),
                        fee_eur=Decimal(str(payload["fee_eur"])),
                        filled_at=datetime.fromisoformat(
                            str(payload.get("filled_at") or event.get("recorded_at")).replace(
                                "Z", "+00:00"
                            )
                        ),
                        venue="paper",
                    )
                except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
                    raise ReconciliationRequired("paper ledger contains an invalid fill") from exc
                self.fills.append(fill)
                base, _ = fill.market.split("-")
                notional = fill.quantity * fill.price
                if fill.side is OrderSide.BUY:
                    self.balances["EUR"] = (
                        self.balances.get("EUR", Decimal("0")) - notional - fill.fee_eur
                    )
                    self.balances[base] = self.balances.get(base, Decimal("0")) + fill.quantity
                else:
                    self.balances[base] = self.balances.get(base, Decimal("0")) - fill.quantity
                    self.balances["EUR"] = (
                        self.balances.get("EUR", Decimal("0")) + notional - fill.fee_eur
                    )
            elif event_type == "ORDER_RESULT":
                try:
                    record = OrderRecord.model_validate(payload["record"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReconciliationRequired(
                        "paper ledger contains an invalid order result"
                    ) from exc
                self.orders[record.order_id] = record
                self.by_idempotency[record.intent.idempotency_key] = record.order_id
                if record.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}:
                    self.open_orders[record.order_id] = record.intent
                else:
                    self.open_orders.pop(record.order_id, None)
        if any(balance < 0 for balance in self.balances.values()):
            raise ReconciliationRequired("paper ledger reconstructs a negative balance")

    def _rules(self, market: str) -> ExecutionMarketRules:
        return self.market_rules.get(market, ExecutionMarketRules())

    def _available_eur(self) -> Decimal:
        return self.balances.get("EUR", Decimal("0")) - sum(
            self.reserved_eur.values(), Decimal("0")
        )

    def _available_asset(self, asset: str) -> Decimal:
        reserved = sum(
            amount
            for selected_asset, amount in self.reserved_assets.values()
            if selected_asset == asset
        )
        return self.balances.get(asset, Decimal("0")) - reserved

    def balance_snapshot(self) -> dict[str, dict[str, Decimal]]:
        assets = set(self.balances) | {asset for asset, _ in self.reserved_assets.values()}
        result: dict[str, dict[str, Decimal]] = {}
        for asset in sorted(assets):
            in_order = (
                sum(self.reserved_eur.values(), Decimal("0"))
                if asset == "EUR"
                else sum(
                    amount
                    for selected_asset, amount in self.reserved_assets.values()
                    if selected_asset == asset
                )
            )
            result[asset] = {
                "available": self.balances.get(asset, Decimal("0")) - in_order,
                "in_order": in_order,
                "total": self.balances.get(asset, Decimal("0")),
            }
        return result

    def _reject(self, intent: OrderIntent, code: str) -> OrderRecord:
        order_id = stable_hash(
            {"intent": intent.intent_id, "rejection": code},
            length=24,
        )
        record = OrderRecord(
            order_id=order_id,
            intent=intent,
            status=OrderStatus.REJECTED,
            rejection_code=code,
        )
        self.orders[order_id] = record
        self.by_idempotency[intent.idempotency_key] = order_id
        self.ledger.append(
            "ORDER_REJECTED",
            {
                "order_id": order_id,
                "idempotency_key": intent.idempotency_key,
                "code": code,
            },
        )
        self._record_result(record)
        return record

    def _record_result(self, record: OrderRecord) -> None:
        self.ledger.append(
            "ORDER_RESULT",
            {"record": record.model_dump(mode="json")},
        )

    def _release_reservation(self, order_id: str) -> None:
        self.reserved_eur.pop(order_id, None)
        self.reserved_assets.pop(order_id, None)

    def _reserve(self, order_id: str, intent: OrderIntent) -> bool:
        base, quote = intent.market.split("-")
        if quote != "EUR":
            return False
        if intent.side is OrderSide.BUY:
            if intent.limit_price is None:
                return False
            required = intent.quantity * intent.limit_price * (Decimal("1") + self.fee_fraction)
            if required > self._available_eur():
                return False
            self.reserved_eur[order_id] = required
        else:
            if intent.quantity > self._available_asset(base):
                return False
            self.reserved_assets[order_id] = (base, intent.quantity)
        return True

    def _fill_price(
        self,
        intent: OrderIntent,
        market_price: Decimal,
    ) -> Decimal:
        impact = (self.slippage_bps + self.spread_bps / Decimal("2")) / Decimal("10000")
        if intent.side is OrderSide.BUY:
            impacted = market_price * (Decimal("1") + impact)
            if intent.order_type is OrderType.LIMIT and intent.limit_price is not None:
                return min(intent.limit_price, impacted)
            return impacted
        impacted = market_price * (Decimal("1") - impact)
        if intent.order_type is OrderType.LIMIT and intent.limit_price is not None:
            return max(intent.limit_price, impacted)
        return impacted

    def _is_marketable(
        self,
        intent: OrderIntent,
        market_price: Decimal,
    ) -> bool:
        if intent.order_type is OrderType.MARKET:
            return True
        if intent.order_type is OrderType.STOP_LOSS:
            assert intent.trigger_price is not None
            return (
                market_price >= intent.trigger_price
                if intent.side is OrderSide.BUY
                else market_price <= intent.trigger_price
            )
        assert intent.limit_price is not None
        return (
            intent.limit_price >= market_price
            if intent.side is OrderSide.BUY
            else intent.limit_price <= market_price
        )

    def submit(
        self,
        intent: OrderIntent,
        *,
        market_price: Decimal,
        available_liquidity: Decimal | None = None,
    ) -> OrderRecord:
        existing_id = self.by_idempotency.get(intent.idempotency_key)
        if existing_id:
            return self.orders[existing_id]
        normalized = normalize_market(intent.market)
        if normalized != intent.market or not normalized.endswith("-EUR"):
            return self._reject(intent, "SPOT_EUR_MARKET_REQUIRED")
        rules = self._rules(normalized)
        quantity = rules.amount(intent.quantity)
        if quantity <= 0:
            return self._reject(intent, "INVALID_PRECISION")
        selected_intent = intent.model_copy(update={"quantity": quantity})
        reference_price = (
            selected_intent.limit_price if selected_intent.limit_price is not None else market_price
        )
        if (
            quantity < rules.minimum_order_amount
            or quantity * reference_price < rules.minimum_order_value_eur
        ):
            return self._reject(selected_intent, "MINIMUM_ORDER")
        base, _ = normalized.split("-")
        if selected_intent.side is OrderSide.SELL and quantity > self._available_asset(base):
            return self._reject(selected_intent, "INSUFFICIENT_OWNED_UNITS")
        estimated_buy_cost = (
            quantity
            * self._fill_price(selected_intent, market_price)
            * (Decimal("1") + self.fee_fraction)
        )
        if selected_intent.side is OrderSide.BUY and estimated_buy_cost > self._available_eur():
            return self._reject(selected_intent, "INSUFFICIENT_EUR")

        order_id = stable_hash(
            {"intent": selected_intent.intent_id, "key": selected_intent.idempotency_key},
            length=24,
        )
        self.by_idempotency[selected_intent.idempotency_key] = order_id
        self.ledger.append(
            "ORDER_INTENT",
            {
                "order_id": order_id,
                "intent_id": selected_intent.intent_id,
                "idempotency_key": selected_intent.idempotency_key,
                "market": selected_intent.market,
                "side": selected_intent.side.value,
                "order_type": selected_intent.order_type.value,
                "quantity": str(selected_intent.quantity),
                "limit_price": (
                    str(selected_intent.limit_price)
                    if selected_intent.limit_price is not None
                    else None
                ),
                "time_in_force": selected_intent.time_in_force.value,
                "post_only": selected_intent.post_only,
            },
        )
        if not self._is_marketable(selected_intent, market_price):
            if not self._reserve(order_id, selected_intent):
                return self._reject(selected_intent, "INSUFFICIENT_BALANCE_TO_RESERVE")
            record = OrderRecord(
                order_id=order_id,
                intent=selected_intent,
                status=OrderStatus.OPEN,
            )
            self.orders[order_id] = record
            self.open_orders[order_id] = selected_intent
            self.ledger.append(
                "ORDER_OPEN",
                {"order_id": order_id, "status": record.status.value},
            )
            self._record_result(record)
            return record
        return self._execute_fill(
            order_id,
            selected_intent,
            market_price=market_price,
            available_liquidity=available_liquidity,
        )

    def _execute_fill(
        self,
        order_id: str,
        intent: OrderIntent,
        *,
        market_price: Decimal,
        available_liquidity: Decimal | None,
    ) -> OrderRecord:
        self._release_reservation(order_id)
        rules = self._rules(intent.market)
        previous = self.orders.get(order_id)
        already_filled = previous.filled_quantity if previous is not None else Decimal("0")
        remaining_quantity = intent.quantity - already_filled
        fill_quantity = rules.amount(
            min(remaining_quantity, available_liquidity)
            if available_liquidity is not None
            else remaining_quantity
        )
        if fill_quantity <= 0:
            return self._reject(intent, "NO_LIQUIDITY")
        price = rules.price(self._fill_price(intent, market_price))
        notional = fill_quantity * price
        fee = notional * self.fee_fraction
        base, _ = intent.market.split("-")
        if intent.side is OrderSide.BUY:
            total = notional + fee
            if total > self._available_eur():
                return self._reject(intent, "INSUFFICIENT_EUR_AT_FILL")
            self.balances["EUR"] = self.balances.get("EUR", Decimal("0")) - total
            self.balances[base] = self.balances.get(base, Decimal("0")) + fill_quantity
        else:
            if fill_quantity > self._available_asset(base):
                return self._reject(intent, "INSUFFICIENT_OWNED_UNITS_AT_FILL")
            self.balances[base] = self.balances.get(base, Decimal("0")) - fill_quantity
            self.balances["EUR"] = self.balances.get("EUR", Decimal("0")) + notional - fee
        filled_at = max(
            utc_now(),
            intent.created_at + timedelta(milliseconds=self.latency_ms),
        )
        fill = Fill(
            fill_id=stable_hash(
                {
                    "order_id": order_id,
                    "fill_number": 1 + sum(item.order_id == order_id for item in self.fills),
                    "quantity": str(fill_quantity),
                    "price": str(price),
                },
                length=24,
            ),
            order_id=order_id,
            intent_id=intent.intent_id,
            market=intent.market,
            side=intent.side,
            quantity=fill_quantity,
            price=price,
            fee_eur=fee,
            filled_at=filled_at,
            venue="paper",
        )
        self.fills.append(fill)
        cumulative_quantity = already_filled + fill_quantity
        status = (
            OrderStatus.FILLED
            if cumulative_quantity == intent.quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        previous_notional = (
            already_filled * previous.average_fill_price
            if previous is not None and previous.average_fill_price is not None
            else Decimal("0")
        )
        average_price = (previous_notional + fill_quantity * price) / cumulative_quantity
        record = OrderRecord(
            order_id=order_id,
            intent=intent,
            status=status,
            filled_quantity=cumulative_quantity,
            average_fill_price=average_price,
            updated_at=filled_at,
        )
        self.orders[order_id] = record
        if status is OrderStatus.FILLED:
            self.open_orders.pop(order_id, None)
        else:
            self.open_orders[order_id] = intent
        self.ledger.append(
            "FILL",
            {
                "fill_id": fill.fill_id,
                "order_id": order_id,
                "intent_id": intent.intent_id,
                "market": intent.market,
                "side": intent.side.value,
                "quantity": str(fill_quantity),
                "price": str(price),
                "fee_eur": str(fee),
                "status": status.value,
                "filled_at": filled_at.isoformat(),
            },
        )
        self._record_result(record)
        return record

    def update_market(
        self,
        market: str,
        *,
        market_price: Decimal,
        available_liquidity: Decimal | None = None,
    ) -> tuple[OrderRecord, ...]:
        normalized = normalize_market(market)
        updated: list[OrderRecord] = []
        for order_id, intent in tuple(self.open_orders.items()):
            if intent.market == normalized and self._is_marketable(intent, market_price):
                updated.append(
                    self._execute_fill(
                        order_id,
                        intent,
                        market_price=market_price,
                        available_liquidity=available_liquidity,
                    )
                )
        return tuple(updated)

    def cancel(self, order_id: str) -> OrderRecord:
        intent = self.open_orders.pop(order_id, None)
        if intent is None:
            raise KeyError(f"open paper order not found: {order_id}")
        self._release_reservation(order_id)
        previous = self.orders[order_id]
        record = previous.model_copy(
            update={"status": OrderStatus.CANCELLED, "updated_at": utc_now()}
        )
        self.orders[order_id] = record
        self.ledger.append("ORDER_CANCELLED", {"order_id": order_id})
        self._record_result(record)
        return record

    def reconcile(self) -> ReconciliationResult:
        reasons: list[str] = []
        if any(balance < 0 for balance in self.balances.values()):
            reasons.append("NEGATIVE_BALANCE")
        if len(self.by_idempotency) != len(set(self.by_idempotency)):
            reasons.append("DUPLICATE_IDEMPOTENCY_KEY")
        if any(record.filled_quantity > record.intent.quantity for record in self.orders.values()):
            reasons.append("OVERFILLED_ORDER")
        try:
            self.ledger.events()
        except ReconciliationRequired:
            reasons.append("LEDGER_UNREADABLE")
        return ReconciliationResult(
            healthy=not reasons,
            reason_codes=tuple(reasons or ["RECONCILED"]),
            checked_at=utc_now(),
            local_open_orders=len(self.open_orders),
            remote_open_orders=None,
        )


class LivePreflight:
    @staticmethod
    def evaluate(
        settings: Settings,
        *,
        markets: tuple[str, ...],
        strategy_status: ResearchStatus,
        data_healthy: bool,
        risk_manager_healthy: bool,
        exchange_healthy: bool,
        reconciliation_healthy: bool,
        kill_switch_active: bool,
        canary_exception_approved: bool = False,
        operator_canary_authorized: bool = False,
        cap_limits: Mapping[str, Any] | None = None,
        portfolio_canary: bool = False,
    ) -> PreflightResult:
        failures = list(settings.static_live_preflight_failures())
        if operator_canary_authorized:
            # A scoped, persisted per-DNA authority replaces the legacy global
            # mode toggles. Credential, IP, scope, market and risk failures
            # remain non-overridable.
            overridable = {
                "LIVE_BLOCKED_NOT_PRODUCTION",
                "LIVE_BLOCKED_MODE_NOT_LIVE",
                "LIVE_BLOCKED_DISABLED",
                "LIVE_BLOCKED_MANUAL_APPROVAL",
                "LIVE_BLOCKED_CANARY_DISABLED",
            }
            failures = [reason for reason in failures if reason not in overridable]
        if strategy_status is not ResearchStatus.PAPER_CANDIDATE and not canary_exception_approved:
            failures.append("LIVE_BLOCKED_STRATEGY_NOT_PAPER_CANDIDATE")
        if not data_healthy:
            failures.append("LIVE_BLOCKED_DATA_UNHEALTHY")
        if not risk_manager_healthy:
            failures.append("LIVE_BLOCKED_RISK_MANAGER_UNHEALTHY")
        if not exchange_healthy:
            failures.append("LIVE_BLOCKED_EXCHANGE_UNHEALTHY")
        if not reconciliation_healthy:
            failures.append("LIVE_BLOCKED_RECONCILIATION")
        if kill_switch_active:
            failures.append("LIVE_BLOCKED_KILL_SWITCH")
        normalized_markets = tuple(normalize_market(market) for market in markets)
        for market in normalized_markets:
            if settings.shariah.eligibility(market).status.value != "ALLOWED":
                failures.append(f"LIVE_BLOCKED_ELIGIBILITY:{market}")
        selected_caps = dict(cap_limits or {})
        maximum_order_eur = Decimal(
            str(
                selected_caps.get(
                    "max_order_eur",
                    settings.execution.maximum_live_order_eur,
                )
            )
        )
        maximum_total_eur = Decimal(
            str(
                selected_caps.get(
                    "max_exposure_eur",
                    settings.execution.maximum_live_total_eur,
                )
            )
        )
        maximum_open_positions = int(
            selected_caps.get(
                "max_positions",
                settings.execution.maximum_live_open_positions,
            )
        )
        maximum_new_orders_per_day = int(
            selected_caps.get(
                "max_new_orders_per_day",
                settings.execution.maximum_live_new_orders_per_day,
            )
        )
        capital_level = int(selected_caps.get("capital_level") or 1)
        fixed_level_caps = {
            1: (Decimal("10"), Decimal("10"), 1),
            2: (Decimal("25"), Decimal("75"), 3),
            3: (Decimal("100"), Decimal("300"), 3),
        }
        invalid_caps = (
            maximum_order_eur <= 0
            or maximum_total_eur < maximum_order_eur
            or not 1 <= maximum_open_positions <= 3
            or not 1 <= maximum_new_orders_per_day <= 3
        )
        if portfolio_canary and capital_level == 1:
            invalid_caps = invalid_caps or (
                maximum_order_eur > Decimal("10")
                or maximum_total_eur > Decimal("15")
                or maximum_open_positions > 3
                or maximum_new_orders_per_day > 3
            )
        elif capital_level in fixed_level_caps:
            order_cap, total_cap, position_cap = fixed_level_caps[capital_level]
            invalid_caps = invalid_caps or (
                maximum_order_eur > order_cap
                or maximum_total_eur > total_cap
                or maximum_open_positions > position_cap
            )
        elif capital_level == 4:
            account_equity_eur = Decimal(
                str(selected_caps.get("account_equity_eur") or "0")
            )
            maximum_exposure_pct = Decimal(
                str(selected_caps.get("max_exposure_pct") or "0")
            )
            maximum_risk_pct = Decimal(
                str(selected_caps.get("max_risk_per_trade_pct") or "0")
            )
            invalid_caps = invalid_caps or (
                account_equity_eur <= 0
                or maximum_exposure_pct <= 0
                or maximum_exposure_pct > Decimal("5")
                or maximum_risk_pct <= 0
                or maximum_risk_pct > Decimal("0.25")
                or maximum_total_eur
                > account_equity_eur * maximum_exposure_pct / Decimal("100")
            )
        else:
            invalid_caps = True
        if invalid_caps:
            failures.append("LIVE_BLOCKED_INVALID_CAPITAL_AUTHORITY")
        failures = list(dict.fromkeys(failures))
        checked_at = utc_now()
        if failures:
            return PreflightResult(
                passed=False,
                failures=tuple(failures),
                capability=None,
                checked_at=checked_at,
            )
        capability = LiveCapability(
            token=stable_hash(
                {
                    "checked_at": checked_at,
                    "markets": normalized_markets,
                    "maximum_order": str(maximum_order_eur),
                    "maximum_total": str(maximum_total_eur),
                    "maximum_positions": maximum_open_positions,
                    "maximum_new_orders_per_day": maximum_new_orders_per_day,
                    "portfolio_canary": portfolio_canary,
                    "nonce": uuid.uuid4().hex,
                },
                length=32,
            ),
            checked_at=checked_at,
            allowed_markets=normalized_markets,
            maximum_order_eur=maximum_order_eur,
            maximum_total_eur=maximum_total_eur,
            maximum_open_positions=maximum_open_positions,
            maximum_new_orders_per_day=maximum_new_orders_per_day,
        )
        return PreflightResult(
            passed=True,
            failures=(),
            capability=capability,
            checked_at=checked_at,
        )


class BitvavoSpotClient:
    """Minimal live client. Funding and withdrawal endpoints do not exist here."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        api_key: SecretStr,
        api_secret: SecretStr,
        operator_id: int,
        ledger: DurableLedger,
        access_window_ms: int = 10_000,
    ) -> None:
        self.session = session
        self._api_key = api_key
        self._api_secret = api_secret
        self.operator_id = operator_id
        self.ledger = ledger
        self.access_window_ms = min(60_000, max(1_000, access_window_ms))

    def __repr__(self) -> str:
        return (
            "BitvavoSpotClient(api_key=SecretStr('**********'), api_secret=SecretStr('**********'))"
        )

    @staticmethod
    def _fee_eur_from_order(
        payload: Mapping[str, Any],
        *,
        fill_price: Decimal,
    ) -> tuple[Decimal, bool]:
        """Extract an actual paid fee and normalize a base-asset fee to EUR."""

        fee_value = payload.get("feePaid")
        fee_currency = str(payload.get("feeCurrency") or "").upper()
        if fee_value is not None and fee_currency:
            try:
                fee = Decimal(str(fee_value))
            except ArithmeticError:
                return Decimal("0"), False
            if fee_currency == "EUR":
                return fee, True
            if fee_currency == str(payload.get("market") or "").split("-")[0].upper():
                return fee * fill_price, True
        fees = payload.get("fees")
        if isinstance(fees, list) and fees:
            total = Decimal("0")
            for item in fees:
                if not isinstance(item, Mapping):
                    return Decimal("0"), False
                value = item.get("fee", item.get("amount"))
                currency = str(item.get("currency") or "").upper()
                try:
                    fee = Decimal(str(value))
                except (ArithmeticError, TypeError):
                    return Decimal("0"), False
                if currency == "EUR":
                    total += fee
                elif currency == str(payload.get("market") or "").split("-")[0].upper():
                    total += fee * fill_price
                else:
                    return Decimal("0"), False
            return total, True
        return Decimal("0"), False

    def _recorded_fill_totals(
        self,
        order_id: str,
    ) -> tuple[Decimal, Decimal, Decimal, bool]:
        """Return canonical cumulative quantity, quote and known EUR fees."""

        quantity = Decimal("0")
        quote = Decimal("0")
        fee = Decimal("0")
        all_fees_known = True
        found = False
        for event in self.ledger.events():
            if event.get("event_type") != "FILL":
                continue
            fill = dict(event.get("payload") or {})
            if str(fill.get("order_id") or "") != order_id:
                continue
            found = True
            try:
                fill_quantity = Decimal(str(fill.get("quantity") or "0"))
                fill_price = Decimal(str(fill.get("price") or "0"))
                fill_quote = Decimal(
                    str(
                        fill.get("quote_amount_eur")
                        or fill_quantity * fill_price
                    )
                )
                fill_fee = Decimal(str(fill.get("fee_eur") or "0"))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ReconciliationRequired(
                    "canonical fill ledger contains invalid cumulative values"
                ) from exc
            quantity += fill_quantity
            quote += fill_quote
            fee += fill_fee
            all_fees_known = (
                all_fees_known and fill.get("fee_known") is True
            )
        return quantity, quote, fee, found and all_fees_known

    def _record_order_status_observation(
        self,
        *,
        order_id: str,
        client_order_id: str | None,
        market: str,
        status: str,
        cumulative_quantity: Decimal,
        cumulative_quote: Decimal,
        received_at: datetime | str | None,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist a deduplicated venue-status checkpoint for restart recovery."""

        normalized_status = status.upper() or "UNKNOWN"
        checkpoint = stable_hash(
            [
                "BITVAVO_ORDER_STATUS",
                order_id,
                normalized_status,
                str(cumulative_quantity),
                str(cumulative_quote),
            ],
            length=24,
        )
        if any(
            event.get("event_type") == "ORDER_STATUS_OBSERVED"
            and str((event.get("payload") or {}).get("checkpoint_id") or "")
            == checkpoint
            for event in self.ledger.events()
        ):
            return
        self.ledger.append(
            "ORDER_STATUS_OBSERVED",
            {
                "checkpoint_id": checkpoint,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "market": market,
                "status": normalized_status,
                "cumulative_quantity": str(cumulative_quantity),
                "cumulative_quote_amount_eur": str(cumulative_quote),
                "exchange_created_at": _normalized_venue_timestamp(
                    payload.get("created")
                ),
                "exchange_updated_at": _normalized_venue_timestamp(
                    payload.get("updated")
                ),
                "received_at": (
                    received_at.isoformat()
                    if isinstance(received_at, datetime)
                    else received_at
                ),
                "venue": "bitvavo",
            },
        )

    def record_order_fill_progress(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_market: str,
        fallback_side: OrderSide,
        fallback_quantity: Decimal,
        fallback_price: Decimal,
        allow_terminal_partial: bool = False,
        allow_open_partial: bool = True,
        received_at: datetime | str | None = None,
    ) -> bool:
        """Persist only the new delta of a cumulative Bitvavo fill update.

        Bitvavo order payloads report cumulative amounts.  Replayed REST or
        WebSocket updates therefore cannot be appended directly: doing so
        would double inventory, fees and PnL.  Each durable FILL represents
        exactly the positive delta since the previous checkpoint.
        """

        status = str(payload.get("status") or "").replace("_", "").replace("-", "").casefold()
        try:
            reported_quantity = Decimal(
                str(payload.get("filledAmount") or "0")
            )
            reported_quote = Decimal(
                str(payload.get("filledAmountQuote") or "0")
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ReconciliationRequired(
                "live order reports invalid cumulative fill values"
            ) from exc
        partially_executed_terminal = (
            allow_terminal_partial
            and status in {"canceled", "cancelled", "partiallyfilled"}
            and reported_quantity > 0
        )
        open_partial = (
            allow_open_partial
            and reported_quantity > 0
            and status in {"new", "awaitingtrigger", "partiallyfilled"}
        )
        if status != "filled" and not partially_executed_terminal and not open_partial:
            return False
        order_id = str(payload.get("orderId") or "")
        if not order_id:
            raise ReconciliationRequired("filled live order lacks order identity")
        market = normalize_market(str(payload.get("market") or fallback_market))
        side = OrderSide(str(payload.get("side") or fallback_side.value).upper())
        cumulative_quantity = (
            reported_quantity
            if reported_quantity > 0
            else fallback_quantity
        )
        cumulative_quote = (
            reported_quote
            if reported_quote > 0
            else cumulative_quantity
            * Decimal(str(payload.get("price") or fallback_price))
        )
        cumulative_price = (
            cumulative_quote / cumulative_quantity
            if cumulative_quote > 0 and cumulative_quantity > 0
            else Decimal(str(payload.get("price") or fallback_price))
        )
        if cumulative_quantity <= 0 or cumulative_price <= 0:
            raise ReconciliationRequired("filled live order has invalid quantity or price")
        prior_quantity, prior_quote, prior_fee, _ = self._recorded_fill_totals(
            order_id
        )
        if cumulative_quantity < prior_quantity or cumulative_quote < prior_quote:
            raise ReconciliationRequired(
                "venue cumulative fill regressed below canonical ledger"
            )
        quantity = cumulative_quantity - prior_quantity
        quote_amount = cumulative_quote - prior_quote
        cumulative_fee, cumulative_fee_known = self._fee_eur_from_order(
            payload,
            fill_price=cumulative_price,
        )
        if cumulative_fee_known and cumulative_fee < prior_fee:
            raise ReconciliationRequired(
                "venue cumulative fee regressed below canonical ledger"
            )
        fee_eur = (
            cumulative_fee - prior_fee
            if cumulative_fee_known
            else Decimal("0")
        )
        fee_known = cumulative_fee_known
        ledger_events = self.ledger.events()
        acknowledged = next(
            (
                dict(event.get("payload") or {})
                for event in reversed(ledger_events)
                if event.get("event_type") == "ORDER_ACKNOWLEDGED"
                and str((event.get("payload") or {}).get("order_id") or "")
                == order_id
            ),
            {},
        )
        client_order_id = str(
            payload.get("clientOrderId")
            or acknowledged.get("client_order_id")
            or ""
        )
        intent_id = str(acknowledged.get("intent_id") or "")
        matching_intent = next(
            (
                dict(event.get("payload") or {})
                for event in reversed(ledger_events)
                if event.get("event_type") == "ORDER_INTENT"
                and (
                    (
                        intent_id
                        and str(
                            (event.get("payload") or {}).get("intent_id")
                            or ""
                        )
                        == intent_id
                    )
                    or (
                        client_order_id
                        and str(
                            (event.get("payload") or {}).get(
                                "client_order_id"
                            )
                            or ""
                        )
                        == client_order_id
                    )
                )
            ),
            {},
        )
        self._record_order_status_observation(
            order_id=order_id,
            client_order_id=client_order_id or None,
            market=market,
            status=status,
            cumulative_quantity=cumulative_quantity,
            cumulative_quote=cumulative_quote,
            received_at=received_at,
            payload=payload,
        )
        if quantity == 0:
            return False
        if quantity < 0 or quote_amount <= 0:
            raise ReconciliationRequired(
                "live fill delta has invalid quantity or quote amount"
            )
        price = quote_amount / quantity
        self.ledger.append(
            "FILL",
            {
                "fill_id": stable_hash(
                    [
                        "BITVAVO",
                        order_id,
                        str(cumulative_quantity),
                        str(cumulative_quote),
                    ],
                    length=24,
                ),
                "order_id": order_id,
                "client_order_id": client_order_id or None,
                "intent_id": matching_intent.get("intent_id"),
                "idempotency_key": matching_intent.get("idempotency_key"),
                "strategy_id": matching_intent.get("strategy_id"),
                "strategy_dna_hash": matching_intent.get("strategy_dna_hash"),
                "signal_id": matching_intent.get("signal_id"),
                "portfolio_decision_id": matching_intent.get(
                    "portfolio_decision_id"
                ),
                "reason_codes": list(matching_intent.get("reason_codes") or []),
                "market": market,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
                "quote_amount_eur": str(quote_amount),
                "cumulative_quantity": str(cumulative_quantity),
                "cumulative_quote_amount_eur": str(cumulative_quote),
                "fee_eur": str(fee_eur),
                "fee_known": fee_known,
                "cumulative_fee_eur": (
                    str(cumulative_fee) if cumulative_fee_known else None
                ),
                "status": (
                    "PARTIALLY_FILLED_FINAL"
                    if partially_executed_terminal
                    else "FILLED"
                    if status == "filled"
                    else "PARTIALLY_FILLED_PROGRESS"
                ),
                "filled_at": (
                    _normalized_venue_timestamp(payload.get("updated"))
                    or _normalized_venue_timestamp(payload.get("created"))
                    or utc_now().isoformat()
                ),
                "exchange_created_at": _normalized_venue_timestamp(
                    payload.get("created")
                ),
                "exchange_updated_at": _normalized_venue_timestamp(
                    payload.get("updated")
                ),
                "received_at": (
                    received_at.isoformat()
                    if isinstance(received_at, datetime)
                    else received_at
                ),
                "venue": "bitvavo",
            },
        )
        return True

    def record_final_fill(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_market: str,
        fallback_side: OrderSide,
        fallback_quantity: Decimal,
        fallback_price: Decimal,
        allow_terminal_partial: bool = False,
        received_at: datetime | str | None = None,
    ) -> bool:
        """Compatibility wrapper for terminal fills and terminal partials."""

        return self.record_order_fill_progress(
            payload,
            fallback_market=fallback_market,
            fallback_side=fallback_side,
            fallback_quantity=fallback_quantity,
            fallback_price=fallback_price,
            allow_terminal_partial=allow_terminal_partial,
            allow_open_partial=False,
            received_at=received_at,
        )

    @staticmethod
    def client_order_id_for(idempotency_key: str) -> str:
        """Return the deterministic public client identity used at the venue."""

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"crypto:{idempotency_key}",
            )
        )

    def _headers(self, method: str, path: str, body: str) -> dict[str, str]:
        timestamp = str(int(time.time() * 1_000))
        payload = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self._api_secret.get_secret_value().encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Bitvavo-Access-Key": self._api_key.get_secret_value(),
            "Bitvavo-Access-Signature": signature,
            "Bitvavo-Access-Timestamp": timestamp,
            "Bitvavo-Access-Window": str(self.access_window_ms),
        }

    async def _private_get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        attempts: int = 3,
    ) -> Any:
        query = urlencode(sorted((params or {}).items()))
        signed_path = f"{path}?{query}" if query else path
        for attempt in range(attempts):
            headers = self._headers("GET", signed_path, "")
            try:
                async with self.session.get(
                    f"https://api.bitvavo.com{signed_path}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status >= 500 or response.status == 429:
                        if attempt + 1 < attempts:
                            await asyncio.sleep(0.5 * 2**attempt)
                            continue
                        raise ReconciliationRequired(
                            "Bitvavo private read temporarily unavailable"
                        )
                    if response.status >= 400:
                        raise ReconciliationRequired(
                            "Bitvavo private read rejected; retry after health recovery"
                        )
                    return await response.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt + 1 >= attempts:
                    raise ReconciliationRequired(
                        "Bitvavo private read temporarily unavailable"
                    ) from exc
                await asyncio.sleep(0.5 * 2**attempt)
        raise ReconciliationRequired(
            "Bitvavo private read temporarily unavailable"
        )

    async def balances(self) -> list[dict[str, Any]]:
        payload = await self._private_get("/v2/balance")
        if not isinstance(payload, list):
            raise ReconciliationRequired("Bitvavo balance response is ambiguous")
        return payload

    async def server_time_ms(self) -> int:
        """Read the venue clock without authentication or account state."""

        try:
            async with self.session.get(
                f"{BITVAVO_BASE_URL}/time",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status >= 400:
                    raise ReconciliationRequired(
                        "Bitvavo server time temporarily unavailable"
                    )
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ReconciliationRequired(
                "Bitvavo server time temporarily unavailable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ReconciliationRequired("Bitvavo server time is ambiguous")
        try:
            value = int(payload.get("time"))
        except (TypeError, ValueError) as exc:
            raise ReconciliationRequired("Bitvavo server time is invalid") from exc
        if value <= 0:
            raise ReconciliationRequired("Bitvavo server time is invalid")
        return value

    async def account_fees(
        self,
        market: str | None = None,
    ) -> dict[str, Decimal]:
        """Read current maker/taker rates, optionally for one market category."""

        if market is None:
            payload = await self._private_get("/v2/account")
            if not isinstance(payload, Mapping):
                raise ReconciliationRequired("Bitvavo account response is ambiguous")
            fees = payload.get("fees")
            if not isinstance(fees, Mapping):
                raise ReconciliationRequired("Bitvavo account fees are missing")
        else:
            normalized = normalize_market(market)
            payload = await self._private_get(
                "/v2/account/fees",
                params={"market": normalized},
            )
            if not isinstance(payload, Mapping):
                raise ReconciliationRequired("Bitvavo market fees are ambiguous")
            fees = payload
        try:
            maker = Decimal(str(fees.get("maker")))
            taker = Decimal(str(fees.get("taker")))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ReconciliationRequired("Bitvavo account fees are invalid") from exc
        if maker < 0 or taker < 0 or maker > 1 or taker > 1:
            raise ReconciliationRequired("Bitvavo account fees are invalid")
        return {"maker": maker, "taker": taker}

    async def transaction_history(
        self,
        *,
        from_date_ms: int,
        to_date_ms: int,
        maximum_pages: int = 20,
    ) -> dict[str, Any]:
        """Read a bounded, paginated account history for cash reconciliation."""

        if from_date_ms < 0 or to_date_ms < from_date_ms:
            raise ValueError("invalid Bitvavo transaction-history interval")
        page_limit = max(1, min(100, int(maximum_pages)))
        items: list[dict[str, Any]] = []
        total_pages = 1
        current_page = 1
        while current_page <= min(total_pages, page_limit):
            payload = await self._private_get(
                "/v2/account/history",
                params={
                    "fromDate": from_date_ms,
                    "toDate": to_date_ms,
                    "page": current_page,
                    "maxItems": 100,
                },
            )
            if not isinstance(payload, Mapping):
                raise ReconciliationRequired(
                    "Bitvavo transaction history is ambiguous"
                )
            page_items = payload.get("items")
            if not isinstance(page_items, list):
                raise ReconciliationRequired(
                    "Bitvavo transaction history items are ambiguous"
                )
            items.extend(dict(row) for row in page_items if isinstance(row, Mapping))
            try:
                total_pages = max(1, int(payload.get("totalPages") or 1))
            except (TypeError, ValueError) as exc:
                raise ReconciliationRequired(
                    "Bitvavo transaction history pagination is invalid"
                ) from exc
            current_page += 1
        return {
            "items": items,
            "complete": total_pages <= page_limit,
            "pages_read": min(total_pages, page_limit),
            "total_pages": total_pages,
        }

    async def arm_cancel_on_disconnect(
        self,
        *,
        group_id: str,
        expiry_after_seconds: int = 30,
    ) -> dict[str, Any]:
        """Arm venue cancellation for entry orders in one bounded group."""

        if not group_id or len(group_id) > 64:
            raise ExecutionBlocked("cancel-on-disconnect group is invalid")
        expiry = max(10, min(300, int(expiry_after_seconds)))
        body = json.dumps(
            {
                "codGroupId": group_id,
                "expiryAfterSeconds": expiry,
            },
            separators=(",", ":"),
        )
        headers = self._headers("POST", "/v2/cancelOrdersAfter", body)
        try:
            async with self.session.post(
                f"{BITVAVO_BASE_URL}/cancelOrdersAfter",
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status >= 400:
                    raise ExecutionBlocked(
                        "Bitvavo cancel-on-disconnect could not be armed"
                    )
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ReconciliationRequired(
                "cancel-on-disconnect state is ambiguous"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ReconciliationRequired(
                "cancel-on-disconnect response is ambiguous"
            )
        self.ledger.append(
            "CANCEL_ON_DISCONNECT_ARMED",
            {
                "group_id_hash": stable_hash(group_id, length=24),
                "expiry_after_seconds": expiry,
                "orders_submitted": 0,
            },
        )
        return dict(payload)

    async def execution_market_rules(
        self,
        market: str,
    ) -> ExecutionMarketRules:
        """Read current public tick, amount and minimum-order constraints."""

        normalized = normalize_market(market)
        try:
            async with self.session.get(
                f"{BITVAVO_BASE_URL}/markets",
                params={"market": normalized},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status >= 400:
                    raise ExecutionBlocked(
                        "Bitvavo market execution rules are unavailable"
                    )
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ExecutionBlocked(
                "Bitvavo market execution rules are unavailable"
            ) from exc
        rows = payload if isinstance(payload, list) else [payload]
        selected = next(
            (
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("market")
                and normalize_market(str(row.get("market") or ""))
                == normalized
            ),
            None,
        )
        if not isinstance(selected, Mapping):
            raise ExecutionBlocked("Bitvavo market execution rules are missing")
        if str(selected.get("status") or "").casefold() != "trading":
            raise ExecutionBlocked("Bitvavo market is not trading")
        return market_rules_from_bitvavo_metadata(selected)

    async def open_orders(self, market: str) -> list[dict[str, Any]]:
        payload = await self._private_get(
            "/v2/ordersOpen",
            params={"market": normalize_market(market)},
        )
        if not isinstance(payload, list):
            raise ReconciliationRequired("Bitvavo open-order response is ambiguous")
        return payload

    async def get_order(
        self,
        *,
        market: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        payload = await self._private_get(
            "/v2/order",
            params={
                "market": normalize_market(market),
                "clientOrderId": client_order_id,
            },
        )
        if not isinstance(payload, dict) or "orderId" not in payload:
            raise ReconciliationRequired("Bitvavo order response is ambiguous")
        return payload

    async def recent_orders(
        self,
        market: str,
        *,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return authoritative recent order history for ambiguity recovery."""

        payload = await self._private_get(
            "/v2/orders",
            params={
                "market": normalize_market(market),
                "limit": max(1, min(int(limit), 1000)),
            },
        )
        if not isinstance(payload, list):
            raise ReconciliationRequired(
                "Bitvavo recent-order response is ambiguous"
            )
        if any(not isinstance(order, dict) for order in payload):
            raise ReconciliationRequired(
                "Bitvavo recent-order response contains an invalid order"
            )
        return payload

    def _record_unknown_order_state(
        self,
        *,
        intent: OrderIntent,
        client_order_id: str,
        reason_code: str,
    ) -> None:
        """Persist an ambiguous submission without ever retrying the write."""

        if any(
            event.get("event_type") == "ORDER_STATE_UNKNOWN"
            and str((event.get("payload") or {}).get("client_order_id") or "")
            == client_order_id
            for event in self.ledger.events()
        ):
            return
        self.ledger.append(
            "ORDER_STATE_UNKNOWN",
            {
                "intent_id": intent.intent_id,
                "client_order_id": client_order_id,
                "market": intent.market,
                "side": intent.side.value,
                "strategy_id": intent.strategy_id,
                "strategy_dna_hash": intent.strategy_dna_hash,
                "signal_id": intent.signal_id,
                "portfolio_decision_id": intent.portfolio_decision_id,
                "portfolio_target_id": intent.portfolio_target_id,
                "risk_approval_id": intent.risk_approval_id,
                "execution_intent_id": intent.execution_intent_id,
                "reason_code": reason_code,
                "execution_blocked_until_reconciled": True,
            },
        )

    def _ledger_order_context(
        self,
        order_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve a venue order to its canonical acknowledgement and intent."""

        events = self.ledger.events()
        acknowledgement = next(
            (
                dict(event.get("payload") or {})
                for event in reversed(events)
                if event.get("event_type") == "ORDER_ACKNOWLEDGED"
                and str((event.get("payload") or {}).get("order_id") or "")
                == order_id
            ),
            {},
        )
        intent_id = str(acknowledgement.get("intent_id") or "")
        client_order_id = str(
            acknowledgement.get("client_order_id") or ""
        )
        intent = next(
            (
                dict(event.get("payload") or {})
                for event in reversed(events)
                if event.get("event_type") == "ORDER_INTENT"
                and (
                    (
                        intent_id
                        and str(
                            (event.get("payload") or {}).get("intent_id")
                            or ""
                        )
                        == intent_id
                    )
                    or (
                        client_order_id
                        and str(
                            (event.get("payload") or {}).get(
                                "client_order_id"
                            )
                            or ""
                        )
                        == client_order_id
                    )
                )
            ),
            {},
        )
        return acknowledgement, intent

    def _record_cancel_race_fill(
        self,
        payload: Mapping[str, Any],
        *,
        order_id: str,
        market: str,
        received_at: datetime,
    ) -> bool:
        """Persist a terminal fill discovered while canceling an own order."""

        acknowledgement, intent = self._ledger_order_context(order_id)
        if not intent:
            return False
        fallback_price = Decimal(
            str(
                intent.get("limit_price")
                or intent.get("estimated_price")
                or intent.get("trigger_price")
                or "0"
            )
        )
        return self.record_final_fill(
            payload,
            fallback_market=market,
            fallback_side=OrderSide(
                str(
                    payload.get("side")
                    or intent.get("side")
                    or acknowledgement.get("side")
                    or ""
                ).upper()
            ),
            fallback_quantity=Decimal(str(intent.get("quantity") or "0")),
            fallback_price=fallback_price,
            allow_terminal_partial=True,
            received_at=received_at,
        )

    def _record_unknown_cancellation(
        self,
        *,
        cancellation_id: str,
        order_id: str,
        client_order_id: str | None,
        market: str,
        reason_code: str,
    ) -> None:
        """Persist an ambiguous cancellation without retrying the DELETE."""

        if any(
            event.get("event_type") == "CANCEL_STATE_UNKNOWN"
            and str((event.get("payload") or {}).get("cancellation_id") or "")
            == cancellation_id
            for event in self.ledger.events()
        ):
            return
        self.ledger.append(
            "CANCEL_STATE_UNKNOWN",
            {
                "cancellation_id": cancellation_id,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "market": market,
                "reason_code": reason_code,
                "replacement_blocked_until_reconciled": True,
            },
        )

    async def cancel_order(
        self,
        *,
        market: str,
        order_id: str,
        capability: LiveCapability,
    ) -> dict[str, Any]:
        """Cancel a Bitvavo spot order only with a fresh preflight capability."""
        if (
            len(capability.token) != 32
            or utc_now() - capability.checked_at > timedelta(minutes=5)
            or normalize_market(market) not in capability.allowed_markets
        ):
            raise ExecutionBlocked("live capability is invalid for cancellation")
        normalized_market = normalize_market(market)
        ledger_events = self.ledger.events()
        resolved_cancellations = {
            str((event.get("payload") or {}).get("cancellation_id") or "")
            for event in ledger_events
            if event.get("event_type") in {"ORDER_CANCELLED", "CANCEL_RESOLVED"}
        }
        if any(
            event.get("event_type") in {"ORDER_CANCELLED", "FILL"}
            and str((event.get("payload") or {}).get("order_id") or "")
            == order_id
            for event in ledger_events
        ):
            raise ExecutionBlocked("order is already terminal; cancellation forbidden")
        pending_cancellation = next(
            (
                dict(event.get("payload") or {})
                for event in reversed(ledger_events)
                if event.get("event_type") == "CANCEL_REQUESTED"
                and str((event.get("payload") or {}).get("order_id") or "")
                == order_id
                and str(
                    (event.get("payload") or {}).get("cancellation_id") or ""
                )
                not in resolved_cancellations
            ),
            None,
        )
        if pending_cancellation is not None:
            raise ReconciliationRequired(
                "cancellation state is unresolved; duplicate cancel forbidden"
            )
        acknowledgement, _ = self._ledger_order_context(order_id)
        client_order_id = str(
            acknowledgement.get("client_order_id") or ""
        ) or None
        cancellation_started_at = utc_now()
        cancellation_id = stable_hash(
            {
                "market": normalized_market,
                "order_id": order_id,
                "started_at": cancellation_started_at.isoformat(),
            },
            length=24,
        )
        self.ledger.append(
            "CANCEL_REQUESTED",
            {
                "cancellation_id": cancellation_id,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "market": normalized_market,
                "cancellation_started_at": cancellation_started_at,
            },
        )
        params = {"market": normalized_market, "orderId": order_id}
        query = urlencode(sorted(params.items()))
        signed_path = f"/v2/order?{query}"
        headers = self._headers("DELETE", signed_path, "")
        try:
            async with self.session.delete(
                f"https://api.bitvavo.com{signed_path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status >= 500 or response.status == 429:
                    self._record_unknown_cancellation(
                        cancellation_id=cancellation_id,
                        order_id=order_id,
                        client_order_id=client_order_id,
                        market=normalized_market,
                        reason_code=f"AMBIGUOUS_CANCEL_HTTP_{response.status}",
                    )
                    raise ReconciliationRequired(
                        "ambiguous cancellation state; reconciliation required"
                    )
                if response.status >= 400:
                    venue_error_code: str | None = None
                    try:
                        rejection = await response.json(content_type=None)
                    except (aiohttp.ClientError, ValueError):
                        rejection = None
                    if isinstance(rejection, Mapping):
                        raw_code = rejection.get("errorCode")
                        if raw_code is not None:
                            venue_error_code = str(raw_code)[:64]
                    suffix = (
                        f"_CODE_{venue_error_code}"
                        if venue_error_code
                        else ""
                    )
                    self._record_unknown_cancellation(
                        cancellation_id=cancellation_id,
                        order_id=order_id,
                        client_order_id=client_order_id,
                        market=normalized_market,
                        reason_code=(
                            f"CANCEL_HTTP_{response.status}{suffix}"
                        ),
                    )
                    raise ReconciliationRequired(
                        "cancellation rejection requires order-state reconciliation"
                    )
                payload = await response.json(content_type=None)
        except ValueError as exc:
            self._record_unknown_cancellation(
                cancellation_id=cancellation_id,
                order_id=order_id,
                client_order_id=client_order_id,
                market=normalized_market,
                reason_code="AMBIGUOUS_CANCEL_RESPONSE_DECODE_FAILURE",
            )
            raise ReconciliationRequired(
                "ambiguous cancellation response; reconciliation required"
            ) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._record_unknown_cancellation(
                cancellation_id=cancellation_id,
                order_id=order_id,
                client_order_id=client_order_id,
                market=normalized_market,
                reason_code="AMBIGUOUS_CANCEL_TRANSPORT_FAILURE",
            )
            raise ReconciliationRequired(
                "ambiguous cancellation state; reconciliation required"
            ) from exc
        if not isinstance(payload, dict) or str(
            payload.get("status") or ""
        ).replace("_", "").replace("-", "").casefold() not in {
            "canceled",
            "cancelled",
        }:
            self._record_unknown_cancellation(
                cancellation_id=cancellation_id,
                order_id=order_id,
                client_order_id=client_order_id,
                market=normalized_market,
                reason_code="AMBIGUOUS_CANCEL_RESPONSE_STATE",
            )
            raise ReconciliationRequired("ambiguous cancellation response")
        cancellation_received_at = utc_now()
        try:
            self._record_cancel_race_fill(
                payload,
                order_id=order_id,
                market=normalized_market,
                received_at=cancellation_received_at,
            )
        except (
            ArithmeticError,
            ReconciliationRequired,
            TypeError,
            ValueError,
        ) as exc:
            self._record_unknown_cancellation(
                cancellation_id=cancellation_id,
                order_id=order_id,
                client_order_id=client_order_id,
                market=normalized_market,
                reason_code="CANCEL_TERMINAL_FILL_RECORDING_FAILED",
            )
            raise ReconciliationRequired(
                "cancellation fill evidence requires reconciliation"
            ) from exc
        self.ledger.append(
            "ORDER_CANCELLED",
            {
                "cancellation_id": cancellation_id,
                "market": normalized_market,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": payload.get("status"),
                "cancellation_started_at": cancellation_started_at,
                "cancellation_received_at": cancellation_received_at,
                "exchange_created_at": _normalized_venue_timestamp(
                    payload.get("created")
                ),
                "exchange_updated_at": _normalized_venue_timestamp(
                    payload.get("updated")
                ),
            },
        )
        return payload

    async def submit_order(
        self,
        intent: OrderIntent,
        *,
        capability: LiveCapability,
        estimated_price: Decimal,
        reconciled_owned_quantity: Decimal,
        reconciled_total_exposure_eur: Decimal | None = None,
        reconciled_open_positions: int = 0,
        exchange_minimum_order_eur: Decimal = Decimal("5"),
        canonical_chain: CanonicalExecutionChain | None = None,
    ) -> dict[str, Any]:
        if (
            len(capability.token) != 32
            or capability.checked_at.tzinfo is None
            or utc_now() - capability.checked_at > timedelta(minutes=5)
            or capability.checked_at > utc_now() + timedelta(seconds=5)
        ):
            raise ExecutionBlocked("live capability is invalid or expired")
        if intent.side is OrderSide.BUY:
            if canonical_chain is None:
                raise ExecutionBlocked(
                    "new live entry requires canonical portfolio target and risk approval"
                )
            try:
                validate_order_against_chain(intent, canonical_chain)
            except ValueError as exc:
                raise ExecutionBlocked(
                    "canonical portfolio execution chain is invalid"
                ) from exc
            if canonical_chain.execution.expires_at < utc_now():
                raise ExecutionBlocked("canonical execution intent is expired")
        if intent.market not in capability.allowed_markets:
            raise ExecutionBlocked("live capability does not include this market")
        notional_price = estimated_price
        if (
            intent.side is OrderSide.BUY
            and intent.order_type is OrderType.LIMIT
            and intent.limit_price is not None
        ):
            # A buy can fill at any price up to its limit.  Cap checks must
            # therefore reserve the worst permitted price, not the ticker.
            notional_price = max(notional_price, intent.limit_price)
        estimated_notional = intent.quantity * notional_price
        if (
            intent.side is OrderSide.BUY
            and intent.order_type is OrderType.MARKET
            and intent.maximum_notional_eur is not None
        ):
            # Quote-denominated market buys are sent as ``amountQuote``.
            # Their base quantity is only an estimate and can contain a
            # recurring decimal (EUR 5 / price), so multiplying it back can
            # become EUR 5.000...001 and falsely breach the exact canary cap.
            # Reserve and transmit the explicit quote cap as the authority.
            estimated_notional = intent.maximum_notional_eur
        if intent.side is OrderSide.BUY:
            if reconciled_total_exposure_eur is None:
                raise ExecutionBlocked("live total exposure is not reconciled")
            if reconciled_total_exposure_eur < 0 or reconciled_open_positions < 0:
                raise ExecutionBlocked("live exposure reconciliation is invalid")
            if reconciled_open_positions >= capability.maximum_open_positions:
                raise ExecutionBlocked("live canary position limit reached")
            if reconciled_total_exposure_eur + estimated_notional > capability.maximum_total_eur:
                raise ExecutionBlocked("order exceeds total live canary cap")
            if estimated_notional < exchange_minimum_order_eur:
                raise ExecutionBlocked("order is below exchange minimum; autoscale forbidden")
        if intent.side is OrderSide.BUY and (
            estimated_notional > capability.maximum_order_eur
            or (
                intent.maximum_notional_eur is not None
                and estimated_notional > intent.maximum_notional_eur
            )
        ):
            raise ExecutionBlocked("order exceeds explicit live notional cap")
        if intent.side is OrderSide.SELL and intent.quantity > reconciled_owned_quantity:
            raise ExecutionBlocked("sell order exceeds reconciled owned units")
        if intent.idempotency_key in self.ledger.idempotency_keys():
            raise ExecutionBlocked("duplicate live order intent")
        today = utc_now().date().isoformat()
        intents_today = sum(
            1
            for event in self.ledger.events()
            if event.get("event_type") == "ORDER_INTENT"
            and str(event.get("recorded_at") or "")[:10] == today
            and str((event.get("payload") or {}).get("side")) == OrderSide.BUY.value
        )
        if intent.side is OrderSide.BUY and intents_today >= capability.maximum_new_orders_per_day:
            raise ExecutionBlocked("live canary daily new-order limit reached")
        client_order_id = self.client_order_id_for(
            intent.idempotency_key
        )
        venue_order_type = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stopLoss",
        }[intent.order_type]
        body: dict[str, Any] = {
            "market": intent.market,
            "side": intent.side.value.lower(),
            "orderType": venue_order_type,
            "operatorId": self.operator_id,
            "clientOrderId": client_order_id,
            "responseRequired": True,
            "selfTradePrevention": "decrementAndCancel",
        }
        if intent.cancel_on_disconnect_group:
            body["codGroupId"] = intent.cancel_on_disconnect_group
        if intent.side is OrderSide.BUY and intent.order_type is OrderType.MARKET:
            # A quote-denominated market buy keeps the canary at its exact
            # EUR notional cap instead of drifting below the venue minimum
            # after base-quantity rounding.
            body["amountQuote"] = str(estimated_notional)
        else:
            body["amount"] = str(intent.quantity)
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            body["price"] = str(intent.limit_price)
            body["timeInForce"] = intent.time_in_force.value
            body["postOnly"] = intent.post_only
        elif intent.order_type is OrderType.STOP_LOSS:
            assert intent.trigger_price is not None
            body["triggerAmount"] = str(intent.trigger_price)
            body["triggerType"] = "price"
            body["triggerReference"] = intent.trigger_reference
        serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        submission_started_at = utc_now()
        if canonical_chain is not None:
            self.ledger.append(
                "PORTFOLIO_TARGET",
                canonical_chain.target.model_dump(mode="json"),
            )
            self.ledger.append(
                "RISK_APPROVAL",
                canonical_chain.approval.model_dump(mode="json"),
            )
            self.ledger.append(
                "EXECUTION_INTENT",
                canonical_chain.execution.model_dump(mode="json"),
            )
        self.ledger.append(
            "ORDER_INTENT",
            {
                "intent_id": intent.intent_id,
                "idempotency_key": intent.idempotency_key,
                "client_order_id": client_order_id,
                "market": intent.market,
                "side": intent.side.value,
                "order_type": intent.order_type.value,
                "time_in_force": intent.time_in_force.value,
                "post_only": intent.post_only,
                "trigger_price": (
                    str(intent.trigger_price)
                    if intent.trigger_price is not None
                    else None
                ),
                "trigger_reference": (
                    intent.trigger_reference
                    if intent.order_type is OrderType.STOP_LOSS
                    else None
                ),
                "quantity": str(intent.quantity),
                "estimated_price": str(estimated_price),
                "limit_price": (
                    str(intent.limit_price)
                    if intent.limit_price is not None
                    else None
                ),
                "strategy_id": intent.strategy_id,
                "strategy_dna_hash": intent.strategy_dna_hash,
                "signal_id": intent.signal_id,
                "portfolio_decision_id": intent.portfolio_decision_id,
                "cancel_on_disconnect_group_hash": (
                    stable_hash(intent.cancel_on_disconnect_group, length=24)
                    if intent.cancel_on_disconnect_group
                    else None
                ),
                "reason_codes": list(intent.reason_codes),
                "maximum_notional_eur": (
                    str(intent.maximum_notional_eur)
                    if intent.maximum_notional_eur is not None
                    else None
                ),
                "created_at": intent.created_at,
                "submission_started_at": submission_started_at,
            },
        )
        headers = self._headers("POST", "/v2/order", serialized)
        try:
            async with self.session.post(
                f"{BITVAVO_BASE_URL}/order",
                data=serialized,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status >= 500 or response.status == 429:
                    self._record_unknown_order_state(
                        intent=intent,
                        client_order_id=client_order_id,
                        reason_code=f"AMBIGUOUS_HTTP_{response.status}",
                    )
                    raise ReconciliationRequired(
                        "ambiguous live order state; reconcile clientOrderId"
                    )
                if response.status >= 400:
                    venue_error_code: str | None = None
                    try:
                        rejection = await response.json(content_type=None)
                    except (aiohttp.ClientError, ValueError):
                        rejection = None
                    if isinstance(rejection, Mapping):
                        raw_code = rejection.get("errorCode")
                        if raw_code is not None:
                            venue_error_code = str(raw_code)[:64]
                    self.ledger.append(
                        "ORDER_REJECTED",
                        {
                            "intent_id": intent.intent_id,
                            "client_order_id": client_order_id,
                            "market": intent.market,
                            "side": intent.side.value,
                            "strategy_id": intent.strategy_id,
                            "strategy_dna_hash": intent.strategy_dna_hash,
                            "signal_id": intent.signal_id,
                            "portfolio_decision_id": (
                                intent.portfolio_decision_id
                            ),
                            "http_status": response.status,
                            "venue_error_code": venue_error_code,
                            "definitive": True,
                        },
                    )
                    detail = (
                        f" code {venue_error_code}"
                        if venue_error_code
                        else ""
                    )
                    raise ExecutionBlocked(
                        "Bitvavo rejected order with HTTP "
                        f"{response.status}{detail}"
                    )
                payload = await response.json(content_type=None)
        except ValueError as exc:
            self._record_unknown_order_state(
                intent=intent,
                client_order_id=client_order_id,
                reason_code="AMBIGUOUS_RESPONSE_DECODE_FAILURE",
            )
            raise ReconciliationRequired(
                "ambiguous live order response; reconcile clientOrderId"
            ) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._record_unknown_order_state(
                intent=intent,
                client_order_id=client_order_id,
                reason_code="AMBIGUOUS_TRANSPORT_FAILURE",
            )
            raise ReconciliationRequired(
                "ambiguous live order state; reconcile clientOrderId"
            ) from exc
        if not isinstance(payload, dict) or "orderId" not in payload:
            self._record_unknown_order_state(
                intent=intent,
                client_order_id=client_order_id,
                reason_code="AMBIGUOUS_RESPONSE_PAYLOAD",
            )
            raise ReconciliationRequired("ambiguous live order response; reconciliation required")
        acknowledgement_received_at = utc_now()
        self.ledger.append(
            "ORDER_ACKNOWLEDGED",
            {
                "intent_id": intent.intent_id,
                "client_order_id": client_order_id,
                "order_id": payload["orderId"],
                "status": payload.get("status"),
                "market": intent.market,
                "side": intent.side.value,
                "strategy_id": intent.strategy_id,
                "strategy_dna_hash": intent.strategy_dna_hash,
                "signal_id": intent.signal_id,
                "portfolio_decision_id": intent.portfolio_decision_id,
                "submission_started_at": submission_started_at,
                "acknowledgement_received_at": acknowledgement_received_at,
                "exchange_created_at": _normalized_venue_timestamp(
                    payload.get("created")
                ),
                "exchange_updated_at": _normalized_venue_timestamp(
                    payload.get("updated")
                ),
            },
        )
        self.record_order_fill_progress(
            payload,
            fallback_market=intent.market,
            fallback_side=intent.side,
            fallback_quantity=intent.quantity,
            fallback_price=estimated_price,
            allow_terminal_partial=(
                intent.order_type is OrderType.LIMIT
                and intent.time_in_force
                in {OrderTimeInForce.IOC, OrderTimeInForce.FOK}
            ),
            allow_open_partial=True,
            received_at=acknowledgement_received_at,
        )
        return payload

    async def reconcile(
        self,
        *,
        markets: tuple[str, ...],
    ) -> ReconciliationResult:
        reasons: list[str] = []
        remote_orders: list[dict[str, Any]] = []
        balance_rows: list[dict[str, Any]] = []
        try:
            balance_rows = await self.balances()
            for market in markets:
                remote_orders.extend(await self.open_orders(market))
        except (ExecutionBlocked, ReconciliationRequired):
            reasons.append("REMOTE_RECONCILIATION_FAILED")
        ledger_events = self.ledger.events()
        local_intents = [
            event
            for event in ledger_events
            if event.get("event_type") == "ORDER_INTENT"
        ]
        local_client_order_ids = {
            str((event.get("payload") or {}).get("client_order_id") or "")
            for event in local_intents
            if (event.get("payload") or {}).get("client_order_id")
        }
        if "REMOTE_RECONCILIATION_FAILED" not in reasons:
            for remote_order in remote_orders:
                client_order_id = str(
                    remote_order.get("clientOrderId") or ""
                )
                if client_order_id not in local_client_order_ids:
                    continue
                order_id = str(remote_order.get("orderId") or "")
                acknowledgement, intent_payload = self._ledger_order_context(
                    order_id
                )
                if not intent_payload:
                    continue
                try:
                    fallback_price = Decimal(
                        str(
                            intent_payload.get("limit_price")
                            or intent_payload.get("estimated_price")
                            or intent_payload.get("trigger_price")
                            or "0"
                        )
                    )
                    self.record_order_fill_progress(
                        remote_order,
                        fallback_market=str(
                            remote_order.get("market")
                            or intent_payload.get("market")
                            or acknowledgement.get("market")
                            or ""
                        ),
                        fallback_side=OrderSide(
                            str(
                                remote_order.get("side")
                                or intent_payload.get("side")
                                or acknowledgement.get("side")
                                or ""
                            ).upper()
                        ),
                        fallback_quantity=Decimal(
                            str(intent_payload.get("quantity") or "0")
                        ),
                        fallback_price=fallback_price,
                        allow_open_partial=True,
                        received_at=utc_now(),
                    )
                except (
                    ArithmeticError,
                    ReconciliationRequired,
                    TypeError,
                    ValueError,
                ):
                    reasons.append("OPEN_ORDER_FILL_PROGRESS_INVALID")
        resolved = {
            event["payload"].get("client_order_id")
            for event in ledger_events
            if event.get("event_type")
            in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}
        }
        unresolved_intents = [
            event
            for event in local_intents
            if str((event.get("payload") or {}).get("client_order_id") or "")
            not in resolved
        ]
        recent_by_market: dict[str, list[dict[str, Any]]] = {}
        if unresolved_intents and "REMOTE_RECONCILIATION_FAILED" not in reasons:
            remote_by_client = {
                str(order.get("clientOrderId") or ""): order
                for order in remote_orders
                if str(order.get("clientOrderId") or "")
            }
            for event in unresolved_intents:
                intent_payload = dict(event.get("payload") or {})
                client_order_id = str(
                    intent_payload.get("client_order_id") or ""
                )
                market = normalize_market(
                    str(intent_payload.get("market") or "")
                )
                if not client_order_id or not market:
                    reasons.append("UNKNOWN_ORDER_STATE")
                    continue
                found = remote_by_client.get(client_order_id)
                try:
                    if found is None:
                        if market not in recent_by_market:
                            recent_by_market[market] = await self.recent_orders(
                                market,
                                limit=1000,
                            )
                        found = next(
                            (
                                order
                                for order in recent_by_market[market]
                                if str(order.get("clientOrderId") or "")
                                == client_order_id
                            ),
                            None,
                        )
                except (ExecutionBlocked, ReconciliationRequired):
                    reasons.append("UNKNOWN_ORDER_STATE")
                    reasons.append("UNKNOWN_ORDER_LOOKUP_FAILED")
                    continue
                if found is None:
                    raw_created_at = (
                        intent_payload.get("created_at")
                        or event.get("recorded_at")
                    )
                    try:
                        created_at = datetime.fromisoformat(
                            str(raw_created_at).replace("Z", "+00:00")
                        )
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=UTC)
                    except (TypeError, ValueError):
                        created_at = utc_now()
                    age = utc_now() - created_at.astimezone(UTC)
                    history_is_complete = (
                        age >= timedelta(seconds=60)
                        and age <= timedelta(hours=24)
                        and len(recent_by_market.get(market, [])) < 1000
                    )
                    if not history_is_complete:
                        reasons.append("UNKNOWN_ORDER_STATE")
                        continue
                    self.ledger.append(
                        "ORDER_REJECTED",
                        {
                            "intent_id": intent_payload.get("intent_id"),
                            "client_order_id": client_order_id,
                            "market": market,
                            "side": intent_payload.get("side"),
                            "strategy_id": intent_payload.get("strategy_id"),
                            "strategy_dna_hash": intent_payload.get(
                                "strategy_dna_hash"
                            ),
                            "signal_id": intent_payload.get("signal_id"),
                            "portfolio_decision_id": intent_payload.get(
                                "portfolio_decision_id"
                            ),
                            "definitive": True,
                            "recovered": True,
                            "reason_code": (
                                "NOT_FOUND_IN_COMPLETE_RECENT_ORDER_HISTORY"
                            ),
                        },
                    )
                    continue
                order_id = str(found.get("orderId") or "")
                if not order_id:
                    reasons.append("UNKNOWN_ORDER_STATE")
                    continue
                try:
                    recovery_received_at = utc_now()
                    fallback_price = Decimal(
                        str(
                            intent_payload.get("limit_price")
                            or intent_payload.get("estimated_price")
                            or intent_payload.get("trigger_price")
                            or "0"
                        )
                    )
                    self.record_final_fill(
                        found,
                        fallback_market=market,
                        fallback_side=OrderSide(
                            str(intent_payload.get("side") or "").upper()
                        ),
                        fallback_quantity=Decimal(
                            str(intent_payload.get("quantity") or "0")
                        ),
                        fallback_price=fallback_price,
                        allow_terminal_partial=True,
                        received_at=recovery_received_at,
                    )
                except (
                    ArithmeticError,
                    ReconciliationRequired,
                    TypeError,
                    ValueError,
                ):
                    reasons.append("UNKNOWN_ORDER_FILL_RECOVERY_FAILED")
                    continue
                self.ledger.append(
                    "ORDER_ACKNOWLEDGED",
                    {
                        "intent_id": intent_payload.get("intent_id"),
                        "client_order_id": client_order_id,
                        "order_id": order_id,
                        "status": found.get("status"),
                        "market": market,
                        "side": intent_payload.get("side"),
                        "strategy_id": intent_payload.get("strategy_id"),
                        "strategy_dna_hash": intent_payload.get(
                            "strategy_dna_hash"
                        ),
                        "signal_id": intent_payload.get("signal_id"),
                        "portfolio_decision_id": intent_payload.get(
                            "portfolio_decision_id"
                        ),
                        "recovered": True,
                        "acknowledgement_received_at": recovery_received_at,
                        "exchange_created_at": _normalized_venue_timestamp(
                            found.get("created")
                        ),
                        "exchange_updated_at": _normalized_venue_timestamp(
                            found.get("updated")
                        ),
                    },
                )

        ledger_events = self.ledger.events()
        resolved_cancellation_ids = {
            str((event.get("payload") or {}).get("cancellation_id") or "")
            for event in ledger_events
            if event.get("event_type") in {"ORDER_CANCELLED", "CANCEL_RESOLVED"}
            and (event.get("payload") or {}).get("cancellation_id")
        }
        pending_cancellations = [
            event
            for event in ledger_events
            if event.get("event_type") == "CANCEL_REQUESTED"
            and str(
                (event.get("payload") or {}).get("cancellation_id") or ""
            )
            not in resolved_cancellation_ids
        ]
        if (
            pending_cancellations
            and "REMOTE_RECONCILIATION_FAILED" not in reasons
        ):
            remote_by_order = {
                str(order.get("orderId") or ""): order
                for order in remote_orders
                if str(order.get("orderId") or "")
            }
            for event in pending_cancellations:
                cancellation = dict(event.get("payload") or {})
                cancellation_id = str(
                    cancellation.get("cancellation_id") or ""
                )
                order_id = str(cancellation.get("order_id") or "")
                market = normalize_market(
                    str(cancellation.get("market") or "")
                )
                if not cancellation_id or not order_id or not market:
                    reasons.append("UNKNOWN_CANCELLATION_STATE")
                    continue
                found = remote_by_order.get(order_id)
                try:
                    if found is None:
                        if market not in recent_by_market:
                            recent_by_market[market] = await self.recent_orders(
                                market,
                                limit=1000,
                            )
                        found = next(
                            (
                                order
                                for order in recent_by_market[market]
                                if str(order.get("orderId") or "") == order_id
                            ),
                            None,
                        )
                except (ExecutionBlocked, ReconciliationRequired):
                    reasons.append("UNKNOWN_CANCELLATION_STATE")
                    reasons.append("CANCELLATION_LOOKUP_FAILED")
                    continue
                if found is None:
                    reasons.append("UNKNOWN_CANCELLATION_STATE")
                    reasons.append("CANCELLED_ORDER_MISSING_FROM_HISTORY")
                    continue
                status = (
                    str(found.get("status") or "")
                    .replace("_", "")
                    .replace("-", "")
                    .casefold()
                )
                received_at = utc_now()
                try:
                    filled_quantity = Decimal(
                        str(found.get("filledAmount") or "0")
                    )
                except (InvalidOperation, TypeError, ValueError):
                    reasons.append("UNKNOWN_CANCELLATION_STATE")
                    reasons.append("INVALID_CANCELLED_ORDER_FILL_QUANTITY")
                    continue
                if status in {
                    "filled",
                    "canceled",
                    "cancelled",
                } and filled_quantity > 0:
                    fill_recovery_failed = False
                    try:
                        self._record_cancel_race_fill(
                            found,
                            order_id=order_id,
                            market=market,
                            received_at=received_at,
                        )
                    except (
                        ArithmeticError,
                        ReconciliationRequired,
                        TypeError,
                        ValueError,
                    ):
                        fill_recovery_failed = True
                    try:
                        recorded_quantity = self._recorded_fill_totals(
                            order_id
                        )[0]
                    except ReconciliationRequired:
                        fill_recovery_failed = True
                        recorded_quantity = Decimal("0")
                    if (
                        fill_recovery_failed
                        or recorded_quantity != filled_quantity
                    ):
                        reasons.append("CANCELLATION_FILL_RECOVERY_FAILED")
                        continue
                if status in {"canceled", "cancelled"}:
                    self.ledger.append(
                        "ORDER_CANCELLED",
                        {
                            "cancellation_id": cancellation_id,
                            "market": market,
                            "order_id": order_id,
                            "client_order_id": cancellation.get(
                                "client_order_id"
                            ),
                            "status": found.get("status"),
                            "cancellation_started_at": cancellation.get(
                                "cancellation_started_at"
                            ),
                            "cancellation_received_at": received_at,
                            "exchange_created_at": (
                                _normalized_venue_timestamp(
                                    found.get("created")
                                )
                            ),
                            "exchange_updated_at": (
                                _normalized_venue_timestamp(
                                    found.get("updated")
                                )
                            ),
                            "recovered": True,
                        },
                    )
                    continue
                if status in {"filled", "expired", "rejected"}:
                    self.ledger.append(
                        "CANCEL_RESOLVED",
                        {
                            "cancellation_id": cancellation_id,
                            "market": market,
                            "order_id": order_id,
                            "client_order_id": cancellation.get(
                                "client_order_id"
                            ),
                            "terminal_order_status": found.get("status"),
                            "resolution": "ORDER_TERMINAL_BEFORE_CANCELLATION",
                            "recovered": True,
                        },
                    )
                    continue
                if status == "partiallyfilled":
                    reasons.append("UNKNOWN_CANCELLATION_STATE")
                    reasons.append("CANCELLATION_PARTIAL_FILL_STILL_OPEN")
                    continue
                if status in {"new", "awaitingtrigger"}:
                    raw_started_at = (
                        cancellation.get("cancellation_started_at")
                        or event.get("recorded_at")
                    )
                    try:
                        started_at = datetime.fromisoformat(
                            str(raw_started_at).replace("Z", "+00:00")
                        )
                        if started_at.tzinfo is None:
                            started_at = started_at.replace(tzinfo=UTC)
                    except (TypeError, ValueError):
                        started_at = utc_now()
                    if utc_now() - started_at.astimezone(UTC) < timedelta(
                        seconds=5
                    ):
                        reasons.append("UNKNOWN_CANCELLATION_STATE")
                        continue
                    self.ledger.append(
                        "CANCEL_RESOLVED",
                        {
                            "cancellation_id": cancellation_id,
                            "market": market,
                            "order_id": order_id,
                            "client_order_id": cancellation.get(
                                "client_order_id"
                            ),
                            "terminal_order_status": found.get("status"),
                            "resolution": "CANCELLATION_NOT_APPLIED_ORDER_STILL_OPEN",
                            "recovered": True,
                        },
                    )
                    continue
                reasons.append("UNKNOWN_CANCELLATION_STATE")
                reasons.append("UNRECOGNIZED_CANCELLED_ORDER_STATUS")

        ledger_events = self.ledger.events()
        resolved = {
            (event.get("payload") or {}).get("client_order_id")
            for event in ledger_events
            if event.get("event_type")
            in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}
        }
        for event in local_intents:
            client_order_id = event["payload"].get("client_order_id")
            if client_order_id and client_order_id not in resolved:
                reasons.append("UNACKNOWLEDGED_LOCAL_INTENT")
        if any(
            str(order.get("clientOrderId") or "") not in local_client_order_ids
            for order in remote_orders
        ):
            reasons.append("UNKNOWN_REMOTE_OPEN_ORDER")
        cancelled_order_ids = {
            str((event.get("payload") or {}).get("order_id") or "")
            for event in ledger_events
            if event.get("event_type") == "ORDER_CANCELLED"
        }
        terminal_fill_client_ids = {
            str((event.get("payload") or {}).get("client_order_id") or "")
            for event in ledger_events
            if event.get("event_type") == "FILL"
            and str((event.get("payload") or {}).get("status") or "").upper()
            in {"FILLED", "PARTIALLY_FILLED_FINAL"}
        }
        terminal_cancel_order_ids = {
            str((event.get("payload") or {}).get("order_id") or "")
            for event in ledger_events
            if event.get("event_type") == "CANCEL_RESOLVED"
            and str(
                (event.get("payload") or {}).get("terminal_order_status")
                or ""
            )
            .replace("_", "")
            .replace("-", "")
            .casefold()
            in {"filled", "expired", "rejected"}
        }
        latest_ack_by_client: dict[str, tuple[str, str]] = {}
        for event in ledger_events:
            if event.get("event_type") not in {
                "ORDER_ACKNOWLEDGED",
                "ORDER_STATUS_OBSERVED",
            }:
                continue
            payload = event.get("payload") or {}
            client_order_id = str(payload.get("client_order_id") or "")
            if client_order_id:
                latest_ack_by_client[client_order_id] = (
                    str(payload.get("status") or "")
                    .replace("_", "")
                    .replace("-", "")
                    .casefold(),
                    str(payload.get("order_id") or ""),
                )
        local_open_client_ids = {
            client_order_id
            for client_order_id, (status, order_id) in latest_ack_by_client.items()
            if status in {"new", "awaitingtrigger", "partiallyfilled"}
            and client_order_id not in terminal_fill_client_ids
            and order_id not in cancelled_order_ids
            and order_id not in terminal_cancel_order_ids
        }
        if "REMOTE_RECONCILIATION_FAILED" not in reasons:
            canonical_balances = sorted(
                (
                    {
                        "symbol": str(row.get("symbol") or "").upper(),
                        "available": str(row.get("available") or "0"),
                        "inOrder": str(row.get("inOrder") or "0"),
                    }
                    for row in balance_rows
                    if isinstance(row, Mapping) and row.get("symbol")
                ),
                key=lambda row: row["symbol"],
            )
            reconciliation_id = stable_hash(
                [
                    "BITVAVO_RECONCILIATION",
                    canonical_balances,
                    sorted(
                        (
                            str(order.get("orderId") or ""),
                            str(order.get("status") or ""),
                            str(order.get("filledAmount") or "0"),
                        )
                        for order in remote_orders
                    ),
                ],
                length=32,
            )
            if not any(
                event.get("event_type") == "RECONCILIATION_OBSERVED"
                and str(
                    (event.get("payload") or {}).get("reconciliation_id")
                    or ""
                )
                == reconciliation_id
                for event in self.ledger.events()
            ):
                self.ledger.append(
                    "RECONCILIATION_OBSERVED",
                    {
                        "reconciliation_id": reconciliation_id,
                        "balances": canonical_balances,
                        "remote_open_orders": [
                            {
                                "order_id": order.get("orderId"),
                                "client_order_id": order.get("clientOrderId"),
                                "market": order.get("market"),
                                "side": order.get("side"),
                                "order_type": order.get("orderType"),
                                "status": order.get("status"),
                                "quantity": order.get("amount"),
                                "filled_quantity": order.get("filledAmount"),
                                "remaining_quantity": order.get(
                                    "amountRemaining"
                                ),
                                "estimated_price": order.get("price"),
                                "trigger_price": order.get("triggerAmount"),
                            }
                            for order in remote_orders
                        ],
                        "reason_codes": list(
                            dict.fromkeys(reasons or ["RECONCILED"])
                        ),
                        "venue": "bitvavo",
                        "source": "private_rest_reconciliation",
                    },
                )
        return ReconciliationResult(
            healthy=not reasons,
            reason_codes=tuple(dict.fromkeys(reasons or ["RECONCILED"])),
            checked_at=utc_now(),
            local_open_orders=len(local_open_client_ids),
            remote_open_orders=len(remote_orders),
        )


def build_live_client(
    settings: Settings,
    *,
    session: aiohttp.ClientSession,
    ledger_path: Path,
) -> BitvavoSpotClient:
    key = settings.providers.bitvavo_trade_api_key
    secret = settings.providers.bitvavo_trade_api_secret
    operator_id = settings.providers.bitvavo_operator_id
    if key is None or secret is None or operator_id is None:
        raise ExecutionBlocked("live trading credentials or operator ID are missing")
    if settings.providers.unsafe_trade_scope():
        raise ExecutionBlocked("LIVE_BLOCKED_UNSAFE_CREDENTIAL_SCOPE")
    return BitvavoSpotClient(
        session=session,
        api_key=key,
        api_secret=secret,
        operator_id=operator_id,
        ledger=DurableLedger(ledger_path),
    )


__all__ = [
    "BitvavoSpotClient",
    "DurableLedger",
    "EntryOrderPlan",
    "ExecutionMarketRules",
    "LiveCapability",
    "LivePreflight",
    "PaperBroker",
    "PreflightResult",
    "ReconciliationResult",
    "build_live_client",
    "market_rules_from_bitvavo_metadata",
    "plan_bounded_entry_order",
]
