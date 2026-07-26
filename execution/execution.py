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
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any
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
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
    normalize_market,
    utc_now,
)
from utils.common import append_jsonl, stable_hash

BITVAVO_BASE_URL = "https://api.bitvavo.com/v2"


@dataclass(frozen=True)
class ExecutionMarketRules:
    minimum_order_amount: Decimal = Decimal("0")
    minimum_order_value_eur: Decimal = Decimal("5")
    quantity_decimals: int = 8
    notional_decimals: int = 2
    tick_size: Decimal = Decimal("0.00000001")

    def amount(self, value: Decimal) -> Decimal:
        quantum = Decimal(1).scaleb(-self.quantity_decimals)
        return value.quantize(quantum, rounding=ROUND_DOWN)

    def price(self, value: Decimal) -> Decimal:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        ticks = (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN)
        return ticks * self.tick_size


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
        self.ledger = DurableLedger(
            ledger_path or Path("output/checkpoints/paper_execution.jsonl")
        )
        self.orders: dict[str, OrderRecord] = {}
        self.fills: list[Fill] = []
        self.by_idempotency: dict[str, str] = {}
        self.open_orders: dict[str, OrderIntent] = {}
        self.reserved_eur: dict[str, Decimal] = {}
        self.reserved_assets: dict[str, tuple[str, Decimal]] = {}
        self._replay_ledger()

    def __repr__(self) -> str:
        return (
            f"PaperBroker(balances={self.balances!r}, "
            f"open_orders={len(self.open_orders)})"
        )

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
                            str(
                                payload.get("filled_at")
                                or event.get("recorded_at")
                            ).replace("Z", "+00:00")
                        ),
                        venue="paper",
                    )
                except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
                    raise ReconciliationRequired(
                        "paper ledger contains an invalid fill"
                    ) from exc
                self.fills.append(fill)
                base, _ = fill.market.split("-")
                notional = fill.quantity * fill.price
                if fill.side is OrderSide.BUY:
                    self.balances["EUR"] = (
                        self.balances.get("EUR", Decimal("0"))
                        - notional
                        - fill.fee_eur
                    )
                    self.balances[base] = (
                        self.balances.get(base, Decimal("0")) + fill.quantity
                    )
                else:
                    self.balances[base] = (
                        self.balances.get(base, Decimal("0")) - fill.quantity
                    )
                    self.balances["EUR"] = (
                        self.balances.get("EUR", Decimal("0"))
                        + notional
                        - fill.fee_eur
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
        assets = set(self.balances) | {
            asset for asset, _ in self.reserved_assets.values()
        }
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
        if quote != "EUR" or intent.limit_price is None:
            return False
        if intent.side is OrderSide.BUY:
            required = (
                intent.quantity * intent.limit_price * (Decimal("1") + self.fee_fraction)
            )
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
        impact = (
            self.slippage_bps + self.spread_bps / Decimal("2")
        ) / Decimal("10000")
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
            selected_intent.limit_price
            if selected_intent.limit_price is not None
            else market_price
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
        already_filled = (
            previous.filled_quantity if previous is not None else Decimal("0")
        )
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
                    "fill_number": 1
                    + sum(item.order_id == order_id for item in self.fills),
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
            if previous is not None
            and previous.average_fill_price is not None
            else Decimal("0")
        )
        average_price = (
            previous_notional + fill_quantity * price
        ) / cumulative_quantity
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
        if any(
            record.filled_quantity > record.intent.quantity
            for record in self.orders.values()
        ):
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
    ) -> PreflightResult:
        failures = list(settings.static_live_preflight_failures())
        if strategy_status is not ResearchStatus.PAPER_CANDIDATE:
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
                    "maximum_order": settings.execution.maximum_live_order_eur,
                    "nonce": uuid.uuid4().hex,
                },
                length=32,
            ),
            checked_at=checked_at,
            allowed_markets=normalized_markets,
            maximum_order_eur=Decimal(str(settings.execution.maximum_live_order_eur)),
            maximum_total_eur=Decimal(
                str(settings.execution.maximum_live_total_eur)
            ),
            maximum_open_positions=(
                settings.execution.maximum_live_open_positions
            ),
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
            "BitvavoSpotClient(api_key=SecretStr('**********'), "
            "api_secret=SecretStr('**********'))"
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
                    if response.status >= 400:
                        raise ExecutionBlocked(
                            f"Bitvavo private read failed with HTTP {response.status}"
                        )
                    return await response.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt + 1 >= attempts:
                    raise ExecutionBlocked("Bitvavo private read failed") from exc
                await asyncio.sleep(0.5 * 2**attempt)
        raise ExecutionBlocked("Bitvavo private read exhausted retries")

    async def balances(self) -> list[dict[str, Any]]:
        payload = await self._private_get("/v2/balance")
        if not isinstance(payload, list):
            raise ReconciliationRequired("Bitvavo balance response is ambiguous")
        return payload

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
        params = {"market": normalize_market(market), "orderId": order_id}
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
                    raise ReconciliationRequired(
                        "ambiguous cancellation state; reconciliation required"
                    )
                if response.status >= 400:
                    raise ExecutionBlocked(
                        f"Bitvavo rejected cancellation with HTTP {response.status}"
                    )
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ReconciliationRequired(
                "ambiguous cancellation state; reconciliation required"
            ) from exc
        if not isinstance(payload, dict):
            raise ReconciliationRequired("ambiguous cancellation response")
        self.ledger.append(
            "ORDER_CANCELLED",
            {
                "market": normalize_market(market),
                "order_id": order_id,
                "status": payload.get("status"),
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
    ) -> dict[str, Any]:
        if (
            len(capability.token) != 32
            or capability.checked_at.tzinfo is None
            or utc_now() - capability.checked_at > timedelta(minutes=5)
            or capability.checked_at > utc_now() + timedelta(seconds=5)
        ):
            raise ExecutionBlocked("live capability is invalid or expired")
        if intent.market not in capability.allowed_markets:
            raise ExecutionBlocked("live capability does not include this market")
        estimated_notional = intent.quantity * estimated_price
        if intent.side is OrderSide.BUY:
            if reconciled_total_exposure_eur is None:
                raise ExecutionBlocked(
                    "live total exposure is not reconciled"
                )
            if (
                reconciled_total_exposure_eur < 0
                or reconciled_open_positions < 0
            ):
                raise ExecutionBlocked(
                    "live exposure reconciliation is invalid"
                )
            if (
                reconciled_open_positions
                >= capability.maximum_open_positions
            ):
                raise ExecutionBlocked(
                    "live canary position limit reached"
                )
            if (
                reconciled_total_exposure_eur + estimated_notional
                > capability.maximum_total_eur
            ):
                raise ExecutionBlocked(
                    "order exceeds total live canary cap"
                )
            if estimated_notional < exchange_minimum_order_eur:
                raise ExecutionBlocked(
                    "order is below exchange minimum; autoscale forbidden"
                )
        if (
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
        client_order_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"crypto:{intent.idempotency_key}")
        )
        body: dict[str, Any] = {
            "market": intent.market,
            "side": intent.side.value.lower(),
            "orderType": intent.order_type.value.lower(),
            "operatorId": self.operator_id,
            "clientOrderId": client_order_id,
            "amount": str(intent.quantity),
            "responseRequired": True,
            "selfTradePrevention": "decrementAndCancel",
        }
        if intent.order_type is OrderType.LIMIT:
            assert intent.limit_price is not None
            body["price"] = str(intent.limit_price)
            body["timeInForce"] = "GTC"
        serialized = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        self.ledger.append(
            "ORDER_INTENT",
            {
                "intent_id": intent.intent_id,
                "idempotency_key": intent.idempotency_key,
                "client_order_id": client_order_id,
                "market": intent.market,
                "side": intent.side.value,
                "quantity": str(intent.quantity),
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
                    raise ReconciliationRequired(
                        "ambiguous live order state; reconcile clientOrderId"
                    )
                if response.status >= 400:
                    raise ExecutionBlocked(
                        f"Bitvavo rejected order with HTTP {response.status}"
                    )
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ReconciliationRequired(
                "ambiguous live order state; reconcile clientOrderId"
            ) from exc
        if not isinstance(payload, dict) or "orderId" not in payload:
            raise ReconciliationRequired(
                "ambiguous live order response; reconciliation required"
            )
        self.ledger.append(
            "ORDER_ACKNOWLEDGED",
            {
                "intent_id": intent.intent_id,
                "client_order_id": client_order_id,
                "order_id": payload["orderId"],
                "status": payload.get("status"),
            },
        )
        return payload

    async def reconcile(
        self,
        *,
        markets: tuple[str, ...],
    ) -> ReconciliationResult:
        reasons: list[str] = []
        remote_orders: list[dict[str, Any]] = []
        try:
            await self.balances()
            for market in markets:
                remote_orders.extend(await self.open_orders(market))
        except (ExecutionBlocked, ReconciliationRequired):
            reasons.append("REMOTE_RECONCILIATION_FAILED")
        local_intents = [
            event
            for event in self.ledger.events()
            if event.get("event_type") == "ORDER_INTENT"
        ]
        acknowledged = {
            event["payload"].get("client_order_id")
            for event in self.ledger.events()
            if event.get("event_type") == "ORDER_ACKNOWLEDGED"
        }
        for event in local_intents:
            client_order_id = event["payload"].get("client_order_id")
            if client_order_id and client_order_id not in acknowledged:
                reasons.append("UNACKNOWLEDGED_LOCAL_INTENT")
        return ReconciliationResult(
            healthy=not reasons,
            reason_codes=tuple(dict.fromkeys(reasons or ["RECONCILED"])),
            checked_at=utc_now(),
            local_open_orders=len(local_intents),
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
    "ExecutionMarketRules",
    "LiveCapability",
    "LivePreflight",
    "PaperBroker",
    "PreflightResult",
    "ReconciliationResult",
    "build_live_client",
]
