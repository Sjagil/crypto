"""Idempotent long-only spot position and PnL tracking."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from core.contracts import Fill, OrderSide, normalize_market
from utils.common import atomic_write_json, read_json, utc_now

ZERO = Decimal("0")
LOGGER = logging.getLogger("crypto.position_tracker")


@dataclass
class Position:
    market: str
    base_asset: str
    quote_asset: str
    owned_quantity: Decimal = ZERO
    available_quantity: Decimal = ZERO
    reserved_quantity: Decimal = ZERO
    average_entry_price: Decimal = ZERO
    cost_basis: Decimal = ZERO
    entry_fees: Decimal = ZERO
    mark_price: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    unrealized_pnl_percentage: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    total_fees: Decimal = ZERO
    strategy_id: str = ""
    parameter_hash: str = ""
    opened_at: datetime | None = None
    updated_at: datetime | None = None
    stop_price: Decimal | None = None
    target_price: Decimal | None = None
    trailing_stop: Decimal | None = None
    initial_risk: Decimal = ZERO
    current_open_risk: Decimal = ZERO
    maximum_favorable_excursion: Decimal = ZERO
    maximum_adverse_excursion: Decimal = ZERO

    def update_derived(self) -> None:
        self.cost_basis = self.average_entry_price * self.owned_quantity
        self.unrealized_pnl = (
            (self.mark_price - self.average_entry_price) * self.owned_quantity
            if self.mark_price > 0 and self.owned_quantity > 0
            else ZERO
        )
        self.unrealized_pnl_percentage = (
            self.unrealized_pnl / self.cost_basis if self.cost_basis else ZERO
        )
        self.current_open_risk = (
            max(ZERO, self.mark_price - self.stop_price) * self.owned_quantity
            if self.stop_price is not None and self.mark_price > 0
            else ZERO
        )
        if self.cost_basis:
            excursion = self.unrealized_pnl / self.cost_basis
            self.maximum_favorable_excursion = max(
                self.maximum_favorable_excursion, excursion
            )
            self.maximum_adverse_excursion = min(
                self.maximum_adverse_excursion, excursion
            )


class PositionTracker:
    def __init__(self, snapshot_path: Path | str | None = None) -> None:
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.positions: dict[str, Position] = {}
        self.fill_ids: set[str] = set()
        self.realized_events: list[dict[str, Any]] = []
        if self.snapshot_path and self.snapshot_path.is_file():
            self._load()

    def _position(self, market: str) -> Position:
        normalized = normalize_market(market)
        if normalized not in self.positions:
            base, quote = normalized.split("-")
            self.positions[normalized] = Position(
                market=normalized,
                base_asset=base,
                quote_asset=quote,
            )
        return self.positions[normalized]

    def ingest_fill(
        self,
        fill: Fill,
        *,
        strategy_id: str = "",
        parameter_hash: str = "",
        stop_price: Decimal | None = None,
        target_price: Decimal | None = None,
        trailing_stop: Decimal | None = None,
        initial_risk: Decimal = ZERO,
    ) -> Position:
        if fill.fill_id in self.fill_ids:
            return self._position(fill.market)
        position = self._position(fill.market)
        quantity = fill.quantity
        fee = fill.fee_eur
        if fill.side is OrderSide.BUY:
            prior_quantity = position.owned_quantity
            total_cost = position.average_entry_price * prior_quantity
            total_cost += fill.price * quantity + fee
            new_quantity = prior_quantity + quantity
            position.average_entry_price = total_cost / new_quantity
            position.owned_quantity = new_quantity
            position.available_quantity += quantity
            position.entry_fees += fee
            if prior_quantity == 0:
                position.opened_at = fill.filled_at
                position.strategy_id = strategy_id
                position.parameter_hash = parameter_hash
                position.stop_price = stop_price
                position.target_price = target_price
                position.trailing_stop = trailing_stop
                position.initial_risk = initial_risk
        else:
            if quantity > position.owned_quantity:
                raise ValueError("sell fill would create a negative spot position")
            realized = (fill.price - position.average_entry_price) * quantity - fee
            position.realized_pnl += realized
            position.owned_quantity -= quantity
            position.available_quantity = max(
                ZERO, position.available_quantity - quantity
            )
            self.realized_events.append(
                {
                    "fill_id": fill.fill_id,
                    "market": fill.market,
                    "strategy_id": position.strategy_id,
                    "realized_pnl": str(realized),
                    "filled_at": fill.filled_at.isoformat(),
                }
            )
            if position.owned_quantity == 0:
                position.available_quantity = ZERO
                position.reserved_quantity = ZERO
                position.average_entry_price = ZERO
                position.opened_at = None
        position.total_fees += fee
        if position.mark_price <= 0:
            position.mark_price = fill.price
        position.updated_at = fill.filled_at
        position.update_derived()
        self.fill_ids.add(fill.fill_id)
        LOGGER.info(
            "fill ingested and PnL updated",
            extra={
                "component": "position_tracker",
                "provider": fill.venue,
                "market": fill.market,
                "operation": "ingest_fill",
                "status": "PASSED",
                "reason_code": "FILL_APPLIED",
                "correlation_id": fill.intent_id,
            },
        )
        self.persist()
        return position

    def mark_to_market(
        self,
        market: str,
        mark_price: Decimal | float | str,
        *,
        observed_at: datetime | None = None,
    ) -> Position:
        selected = Decimal(str(mark_price))
        if selected <= 0:
            raise ValueError("mark price must be positive")
        position = self._position(market)
        position.mark_price = selected
        position.updated_at = observed_at or utc_now()
        position.update_derived()
        self.persist()
        return position

    def reserve(self, market: str, quantity: Decimal | float | str) -> Position:
        selected = Decimal(str(quantity))
        position = self._position(market)
        if selected < 0 or selected > position.available_quantity:
            raise ValueError("invalid reserved quantity")
        position.available_quantity -= selected
        position.reserved_quantity += selected
        position.updated_at = utc_now()
        self.persist()
        return position

    def reconcile_balances(
        self,
        balances: dict[str, Decimal | float | str],
        *,
        tolerance: Decimal = Decimal("0.00000001"),
    ) -> dict[str, Any]:
        discrepancies: list[dict[str, str]] = []
        for market, position in self.positions.items():
            exchange = Decimal(str(balances.get(position.base_asset, ZERO)))
            difference = exchange - position.owned_quantity
            if abs(difference) > tolerance:
                discrepancies.append(
                    {
                        "market": market,
                        "tracked": str(position.owned_quantity),
                        "exchange": str(exchange),
                        "difference": str(difference),
                    }
                )
        return {
            "healthy": not discrepancies,
            "reason_code": "RECONCILED" if not discrepancies else "BALANCE_DISCREPANCY",
            "discrepancies": discrepancies,
            "checked_at": utc_now().isoformat(),
        }

    def portfolio_pnl(self) -> dict[str, Decimal]:
        return {
            "realized_pnl": sum(
                (item.realized_pnl for item in self.positions.values()), ZERO
            ),
            "unrealized_pnl": sum(
                (item.unrealized_pnl for item in self.positions.values()), ZERO
            ),
            "total_fees": sum(
                (item.total_fees for item in self.positions.values()), ZERO
            ),
        }

    def pnl_by_strategy(self) -> dict[str, Decimal]:
        result: dict[str, Decimal] = {}
        for position in self.positions.values():
            key = position.strategy_id or "unassigned"
            result[key] = result.get(key, ZERO) + position.realized_pnl + position.unrealized_pnl
        return result

    def pnl_by_symbol(self) -> dict[str, Decimal]:
        return {
            market: position.realized_pnl + position.unrealized_pnl
            for market, position in self.positions.items()
        }

    def daily_pnl(self, selected_date: date | None = None) -> Decimal:
        selected = selected_date or datetime.now(UTC).date()
        return sum(
            (
                Decimal(item["realized_pnl"])
                for item in self.realized_events
                if datetime.fromisoformat(item["filled_at"]).date() == selected
            ),
            ZERO,
        )

    def drawdown_inputs(self) -> dict[str, Any]:
        pnl = self.portfolio_pnl()
        return {
            "realized_pnl": pnl["realized_pnl"],
            "unrealized_pnl": pnl["unrealized_pnl"],
            "daily_pnl": self.daily_pnl(),
            "positions": self.pnl_by_symbol(),
            "strategies": self.pnl_by_strategy(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "positions": {
                market: self._serialize_position(position)
                for market, position in self.positions.items()
            },
            "fill_ids": sorted(self.fill_ids),
            "realized_events": self.realized_events,
            "updated_at": utc_now().isoformat(),
        }

    @staticmethod
    def _serialize_position(position: Position) -> dict[str, Any]:
        return {
            key: (
                value.isoformat()
                if isinstance(value, datetime)
                else str(value)
                if isinstance(value, Decimal)
                else value
            )
            for key, value in asdict(position).items()
        }

    def persist(self) -> None:
        if self.snapshot_path:
            atomic_write_json(self.snapshot_path, self.snapshot())

    def _load(self) -> None:
        payload = read_json(self.snapshot_path)
        decimal_fields = {
            "owned_quantity",
            "available_quantity",
            "reserved_quantity",
            "average_entry_price",
            "cost_basis",
            "entry_fees",
            "mark_price",
            "unrealized_pnl",
            "unrealized_pnl_percentage",
            "realized_pnl",
            "total_fees",
            "stop_price",
            "target_price",
            "trailing_stop",
            "initial_risk",
            "current_open_risk",
            "maximum_favorable_excursion",
            "maximum_adverse_excursion",
        }
        datetime_fields = {"opened_at", "updated_at"}
        for market, raw in payload.get("positions", {}).items():
            values = dict(raw)
            for field in decimal_fields:
                if values.get(field) is not None:
                    values[field] = Decimal(values[field])
            for field in datetime_fields:
                if values.get(field):
                    values[field] = datetime.fromisoformat(values[field])
            self.positions[market] = Position(**values)
        self.fill_ids = set(payload.get("fill_ids", []))
        self.realized_events = list(payload.get("realized_events", []))


__all__ = ["Position", "PositionTracker"]
