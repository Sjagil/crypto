"""Async-safe, validated Decimal Level 2 order book."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from utils.common import utc_now

Level = tuple[Decimal, Decimal]
SnapshotRefresh = Callable[[], Awaitable[None]]
ChecksumValidator = Callable[[tuple[Level, ...], tuple[Level, ...], Any], bool]
LOGGER = logging.getLogger("crypto.orderbook")


class SequenceGap(RuntimeError):
    pass


class InvalidOrderBook(RuntimeError):
    pass


def _decimal(value: Any) -> Decimal:
    selected = Decimal(str(value))
    if not selected.is_finite():
        raise ValueError("order-book values must be finite")
    return selected


class Level2OrderBook:
    def __init__(
        self,
        *,
        provider: str,
        market: str,
        maximum_depth: int = 100,
        stale_after: timedelta = timedelta(seconds=30),
        refresh_snapshot: SnapshotRefresh | None = None,
        checksum_validator: ChecksumValidator | None = None,
    ) -> None:
        if maximum_depth < 1:
            raise ValueError("maximum_depth must be positive")
        self.provider = provider
        self.market = market
        self.maximum_depth = maximum_depth
        self.stale_after = stale_after
        self.refresh_snapshot = refresh_snapshot
        self.checksum_validator = checksum_validator
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._lock = asyncio.Lock()
        self.sequence: int | None = None
        self.valid = False
        self.last_valid_update: datetime | None = None
        self._message_ids: set[str] = set()
        self._previous_depth = Decimal("0")
        self._resiliency = Decimal("1")
        self.statistics: dict[str, int] = {
            "snapshots": 0,
            "deltas": 0,
            "duplicates": 0,
            "out_of_order": 0,
            "sequence_gaps": 0,
            "checksum_failures": 0,
            "invalidations": 0,
            "refresh_requests": 0,
        }

    @staticmethod
    def _levels(values: Iterable[Iterable[Any]]) -> dict[Decimal, Decimal]:
        result: dict[Decimal, Decimal] = {}
        for item in values:
            price, quantity = item
            selected_price = _decimal(price)
            selected_quantity = _decimal(quantity)
            if selected_price <= 0 or selected_quantity < 0:
                raise ValueError("price must be positive and quantity non-negative")
            if selected_quantity:
                result[selected_price] = selected_quantity
        return result

    def _trim(self) -> None:
        self._bids = dict(
            sorted(self._bids.items(), reverse=True)[: self.maximum_depth]
        )
        self._asks = dict(sorted(self._asks.items())[: self.maximum_depth])

    def _validate_cross(self) -> None:
        if self._bids and self._asks and max(self._bids) >= min(self._asks):
            raise InvalidOrderBook("book is crossed or locked")

    async def initialize(
        self,
        *,
        bids: Iterable[Iterable[Any]],
        asks: Iterable[Iterable[Any]],
        sequence: int | None = None,
        checksum: Any | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        async with self._lock:
            self._bids = self._levels(bids)
            self._asks = self._levels(asks)
            self._trim()
            try:
                self._validate_cross()
                self._validate_checksum(checksum)
            except Exception:
                self._invalidate_locked()
                raise
            self.sequence = sequence
            self.valid = True
            self.last_valid_update = timestamp or utc_now()
            self._previous_depth = self.cumulative_bid_depth + self.cumulative_ask_depth
            self._message_ids.clear()
            self.statistics["snapshots"] += 1
            LOGGER.info(
                "order book snapshot initialized",
                extra={
                    "component": "orderbook",
                    "provider": self.provider,
                    "market": self.market,
                    "operation": "snapshot_rebuild",
                    "status": "PASSED",
                    "reason_code": "SNAPSHOT_VALID",
                },
            )

    async def apply_delta(
        self,
        *,
        bids: Iterable[Iterable[Any]] = (),
        asks: Iterable[Iterable[Any]] = (),
        sequence: int | None = None,
        previous_sequence: int | None = None,
        message_id: str | None = None,
        checksum: Any | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        refresh = False
        async with self._lock:
            if not self.valid:
                raise InvalidOrderBook("book requires a fresh snapshot")
            if message_id and message_id in self._message_ids:
                self.statistics["duplicates"] += 1
                return False
            if sequence is not None and self.sequence is not None:
                if sequence <= self.sequence:
                    self.statistics["out_of_order"] += 1
                    return False
                expected = self.sequence + 1
                if sequence != expected or (
                    previous_sequence is not None and previous_sequence != self.sequence
                ):
                    self.statistics["sequence_gaps"] += 1
                    LOGGER.error(
                        "order book sequence gap",
                        extra={
                            "component": "orderbook",
                            "provider": self.provider,
                            "market": self.market,
                            "operation": "apply_delta",
                            "status": "FAILED",
                            "reason_code": "SEQUENCE_GAP",
                        },
                    )
                    self._invalidate_locked()
                    refresh = True
            if refresh:
                pass
            else:
                prior_bids = self._bids.copy()
                prior_asks = self._asks.copy()
                prior_depth = self.cumulative_bid_depth + self.cumulative_ask_depth
                try:
                    self._apply(self._bids, bids)
                    self._apply(self._asks, asks)
                    self._trim()
                    self._validate_cross()
                    self._validate_checksum(checksum)
                except Exception:
                    self._bids = prior_bids
                    self._asks = prior_asks
                    self._invalidate_locked()
                    refresh = True
                if not refresh:
                    if sequence is not None:
                        self.sequence = sequence
                    if message_id:
                        self._message_ids.add(message_id)
                        if len(self._message_ids) > 20_000:
                            self._message_ids = set(list(self._message_ids)[-10_000:])
                    self.last_valid_update = timestamp or utc_now()
                    new_depth = self.cumulative_bid_depth + self.cumulative_ask_depth
                    if prior_depth > 0:
                        self._resiliency = min(Decimal("1"), new_depth / prior_depth)
                    self._previous_depth = new_depth
                    self.statistics["deltas"] += 1
                    return True
        if refresh:
            await self._request_refresh()
            raise SequenceGap("book invalidated; snapshot refresh requested")
        return False

    @staticmethod
    def _apply(
        side: dict[Decimal, Decimal], values: Iterable[Iterable[Any]]
    ) -> None:
        for item in values:
            price, quantity = item
            selected_price = _decimal(price)
            selected_quantity = _decimal(quantity)
            if selected_price <= 0 or selected_quantity < 0:
                raise ValueError("invalid book level")
            if selected_quantity == 0:
                side.pop(selected_price, None)
            else:
                side[selected_price] = selected_quantity

    def _validate_checksum(self, checksum: Any | None) -> None:
        if checksum is None or self.checksum_validator is None:
            return
        if not self.checksum_validator(self.bids, self.asks, checksum):
            self.statistics["checksum_failures"] += 1
            raise InvalidOrderBook("provider checksum mismatch")

    def _invalidate_locked(self) -> None:
        self.valid = False
        self.statistics["invalidations"] += 1

    async def invalidate(self, reason: str = "MANUAL_INVALIDATION") -> None:
        del reason
        async with self._lock:
            self._invalidate_locked()
        await self._request_refresh()

    async def _request_refresh(self) -> None:
        if self.refresh_snapshot is not None:
            self.statistics["refresh_requests"] += 1
            await self.refresh_snapshot()

    @property
    def bids(self) -> tuple[Level, ...]:
        return tuple(sorted(self._bids.items(), reverse=True))

    @property
    def asks(self) -> tuple[Level, ...]:
        return tuple(sorted(self._asks.items()))

    @property
    def best_bid(self) -> Decimal | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return min(self._asks) if self._asks else None

    @property
    def mid_price(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> Decimal | None:
        if not self.mid_price or self.spread is None:
            return None
        return self.spread / self.mid_price * Decimal("10000")

    @property
    def microprice(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        bid_size = self._bids[self.best_bid]
        ask_size = self._asks[self.best_ask]
        total = bid_size + ask_size
        if total == 0:
            return self.mid_price
        return (self.best_ask * bid_size + self.best_bid * ask_size) / total

    @property
    def top_level_imbalance(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        bid = self._bids[self.best_bid]
        ask = self._asks[self.best_ask]
        return (bid - ask) / (bid + ask) if bid + ask else Decimal("0")

    @property
    def cumulative_bid_depth(self) -> Decimal:
        return sum(self._bids.values(), Decimal("0"))

    @property
    def cumulative_ask_depth(self) -> Decimal:
        return sum(self._asks.values(), Decimal("0"))

    @property
    def book_pressure(self) -> Decimal:
        total = self.cumulative_bid_depth + self.cumulative_ask_depth
        if total == 0:
            return Decimal("0")
        return (self.cumulative_bid_depth - self.cumulative_ask_depth) / total

    @property
    def resiliency(self) -> Decimal:
        return self._resiliency

    def depth_imbalance(self, band_bps: Decimal | float | str) -> Decimal | None:
        mid = self.mid_price
        if mid is None:
            return None
        fraction = _decimal(band_bps) / Decimal("10000")
        lower, upper = mid * (1 - fraction), mid * (1 + fraction)
        bids = sum(
            (quantity for price, quantity in self._bids.items() if price >= lower),
            Decimal("0"),
        )
        asks = sum(
            (quantity for price, quantity in self._asks.items() if price <= upper),
            Decimal("0"),
        )
        return (bids - asks) / (bids + asks) if bids + asks else Decimal("0")

    def weighted_average_execution_price(
        self, *, side: str, quantity: Decimal | float | str
    ) -> Decimal | None:
        remaining = _decimal(quantity)
        if remaining <= 0:
            raise ValueError("quantity must be positive")
        levels = self.asks if side.casefold() == "buy" else self.bids
        cost = Decimal("0")
        filled = Decimal("0")
        for price, available in levels:
            amount = min(remaining, available)
            cost += price * amount
            filled += amount
            remaining -= amount
            if remaining == 0:
                break
        if remaining > 0 or filled == 0:
            return None
        return cost / filled

    def estimated_slippage(
        self, *, side: str, quantity: Decimal | float | str
    ) -> Decimal | None:
        price = self.weighted_average_execution_price(side=side, quantity=quantity)
        reference = self.best_ask if side.casefold() == "buy" else self.best_bid
        if price is None or reference is None:
            return None
        direction = Decimal("1") if side.casefold() == "buy" else Decimal("-1")
        return direction * (price - reference) / reference

    def estimated_market_impact(
        self, *, side: str, quantity: Decimal | float | str
    ) -> Decimal | None:
        price = self.weighted_average_execution_price(side=side, quantity=quantity)
        mid = self.mid_price
        if price is None or mid is None:
            return None
        direction = Decimal("1") if side.casefold() == "buy" else Decimal("-1")
        return direction * (price - mid) / mid

    def is_stale(self, now: datetime | None = None) -> bool:
        if not self.valid or self.last_valid_update is None:
            return True
        return (now or utc_now()) - self.last_valid_update > self.stale_after

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "market": self.market,
            "valid": self.valid,
            "stale": self.is_stale(),
            "sequence": self.sequence,
            "bid_levels": len(self._bids),
            "ask_levels": len(self._asks),
            "spread_bps": str(self.spread_bps) if self.spread_bps is not None else None,
            "last_valid_update": (
                self.last_valid_update.isoformat() if self.last_valid_update else None
            ),
            **self.statistics,
        }


__all__ = ["InvalidOrderBook", "Level2OrderBook", "SequenceGap"]
