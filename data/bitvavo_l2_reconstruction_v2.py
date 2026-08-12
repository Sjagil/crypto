"""Deterministic, fail-closed Bitvavo L2 reconstruction (V2).

Raw source evidence is never modified.  A book becomes research-valid only
after a trusted snapshot and remains valid only while every venue nonce is
exactly contiguous.  Features are absent outside explicitly valid intervals.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Mapping, Sequence

from utils.common import stable_hash, utc_iso

L2_RECONSTRUCTION_VERSION = "L2_RECONSTRUCTION_V2"
L2_FEATURE_SCHEMA_VERSION = "bitvavo_l2_features_v2"


class BitvavoBookState(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    WAITING_FOR_SNAPSHOT = "WAITING_FOR_SNAPSHOT"
    SYNCING = "SYNCING"
    VALID = "VALID"
    GAPPED = "GAPPED"
    STALE = "STALE"
    RESEED_REQUIRED = "RESEED_REQUIRED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class BookTransition:
    at: datetime
    previous: BitvavoBookState
    current: BitvavoBookState
    reason: str
    sequence: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "at": utc_iso(self.at),
            "previous": self.previous.value,
            "current": self.current.value,
        }


@dataclass(frozen=True, slots=True)
class BookValidityInterval:
    market: str
    start: datetime
    end: datetime
    start_sequence: int
    end_sequence: int
    snapshot_reference: str
    closing_reason: str

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "start": utc_iso(self.start),
            "end": utc_iso(self.end),
            "duration_seconds": self.duration_seconds,
            "interval_hash": stable_hash(asdict(self)),
        }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _decimal(value: Any, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("INVALID_DECIMAL") from exc
    if not selected.is_finite():
        raise ValueError("NONFINITE_DECIMAL")
    if positive and selected <= 0:
        raise ValueError("NONPOSITIVE_PRICE")
    if nonnegative and selected < 0:
        raise ValueError("NEGATIVE_QUANTITY")
    return selected


def _parse_levels(rows: Sequence[Any]) -> list[tuple[Decimal, Decimal]]:
    parsed: list[tuple[Decimal, Decimal]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError("MALFORMED_LEVEL")
        parsed.append(
            (
                _decimal(row[0], positive=True),
                _decimal(row[1], nonnegative=True),
            )
        )
    return parsed


class BitvavoL2StateMachine:
    """One-market Bitvavo price-aggregated L2 state machine."""

    def __init__(
        self,
        market: str,
        *,
        maximum_levels: int = 500,
        stale_after: timedelta = timedelta(seconds=30),
        recent_event_limit: int = 20_000,
    ) -> None:
        if maximum_levels < 1 or stale_after <= timedelta(0) or recent_event_limit < 1:
            raise ValueError("invalid book bounds")
        self.market = market.upper()
        self.maximum_levels = maximum_levels
        self.stale_after = stale_after
        self.state = BitvavoBookState.UNINITIALIZED
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.sequence: int | None = None
        self.snapshot_reference: str | None = None
        self.last_event_at: datetime | None = None
        self.last_known_at: datetime | None = None
        self.transitions: list[BookTransition] = []
        self.intervals: list[BookValidityInterval] = []
        self.failure_counts: Counter[str] = Counter()
        self.applied_deltas = 0
        self.duplicate_events = 0
        self.pre_snapshot_discarded = 0
        self.reseed_count = 0
        self._valid_start: datetime | None = None
        self._valid_start_sequence: int | None = None
        self._recent_ids: deque[str] = deque(maxlen=recent_event_limit)
        self._recent_set: set[str] = set()
        self._transition(
            BitvavoBookState.WAITING_FOR_SNAPSHOT, datetime(1970, 1, 1, tzinfo=UTC), "INITIALIZE"
        )

    def _transition(self, state: BitvavoBookState, at: datetime, reason: str) -> None:
        selected = _utc(at)
        previous = self.state
        if previous is BitvavoBookState.VALID and state is not BitvavoBookState.VALID:
            self._close_valid_interval(selected, reason)
        if state is previous and self.transitions:
            return
        self.state = state
        self.transitions.append(BookTransition(selected, previous, state, reason, self.sequence))
        if reason not in {"INITIALIZE", "SNAPSHOT_ACCEPTED", "SYNC_COMPLETE", "DELTA_APPLIED"}:
            self.failure_counts[reason] += 1

    def _close_valid_interval(self, at: datetime, reason: str) -> None:
        if (
            self._valid_start is None
            or self._valid_start_sequence is None
            or self.sequence is None
            or self.snapshot_reference is None
        ):
            return
        selected = max(_utc(at), self._valid_start)
        self.intervals.append(
            BookValidityInterval(
                market=self.market,
                start=self._valid_start,
                end=selected,
                start_sequence=self._valid_start_sequence,
                end_sequence=self.sequence,
                snapshot_reference=self.snapshot_reference,
                closing_reason=reason,
            )
        )
        self._valid_start = None
        self._valid_start_sequence = None

    def _remember(self, event_id: str) -> bool:
        if event_id in self._recent_set:
            self.duplicate_events += 1
            return False
        if len(self._recent_ids) == self._recent_ids.maxlen:
            self._recent_set.discard(self._recent_ids[0])
        self._recent_ids.append(event_id)
        self._recent_set.add(event_id)
        return True

    def _trim(self, bids: dict[Decimal, Decimal], asks: dict[Decimal, Decimal]) -> None:
        selected_bids = sorted(bids.items(), reverse=True)[: self.maximum_levels]
        selected_asks = sorted(asks.items())[: self.maximum_levels]
        bids.clear()
        asks.clear()
        bids.update(selected_bids)
        asks.update(selected_asks)

    @staticmethod
    def _validate_book(bids: Mapping[Decimal, Decimal], asks: Mapping[Decimal, Decimal]) -> None:
        if not bids or not asks:
            raise ValueError("EMPTY_BOOK_SIDE")
        if max(bids) >= min(asks):
            raise ValueError("LOCKED_OR_CROSSED_BOOK")

    def _clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.sequence = None
        self.snapshot_reference = None

    def require_reseed(self, at: datetime, reason: str) -> None:
        selected = _utc(at)
        if self.state is BitvavoBookState.VALID:
            self._transition(BitvavoBookState.GAPPED, selected, reason)
        elif self.state not in {BitvavoBookState.GAPPED, BitvavoBookState.INVALID}:
            self._transition(BitvavoBookState.RESEED_REQUIRED, selected, reason)
        self._clear()
        if self.state is not BitvavoBookState.RESEED_REQUIRED:
            self._transition(BitvavoBookState.RESEED_REQUIRED, selected, "FRESH_SNAPSHOT_REQUIRED")

    def on_disconnect(self, at: datetime) -> None:
        self.require_reseed(at, "RECONNECT_RESET")

    def seed_snapshot(
        self,
        *,
        bids: Sequence[Any],
        asks: Sequence[Any],
        sequence: int | None,
        event_at: datetime,
        known_at: datetime,
        snapshot_reference: str,
    ) -> bool:
        selected_event = _utc(event_at)
        selected_known = _utc(known_at)
        self._transition(BitvavoBookState.SYNCING, selected_known, "SNAPSHOT_ACCEPTED")
        try:
            if sequence is None or int(sequence) < 0:
                raise ValueError("SNAPSHOT_SEQUENCE_MISSING")
            next_bids = {price: quantity for price, quantity in _parse_levels(bids) if quantity > 0}
            next_asks = {price: quantity for price, quantity in _parse_levels(asks) if quantity > 0}
            self._trim(next_bids, next_asks)
            self._validate_book(next_bids, next_asks)
        except ValueError as exc:
            self._clear()
            self._transition(BitvavoBookState.INVALID, selected_known, str(exc))
            self._transition(
                BitvavoBookState.RESEED_REQUIRED, selected_known, "FRESH_SNAPSHOT_REQUIRED"
            )
            return False
        self.bids = next_bids
        self.asks = next_asks
        self.sequence = int(sequence)
        self.snapshot_reference = str(snapshot_reference)
        self.last_event_at = selected_event
        self.last_known_at = selected_known
        self.reseed_count += 1
        self._valid_start = selected_known
        self._valid_start_sequence = self.sequence
        self._transition(BitvavoBookState.VALID, selected_known, "SYNC_COMPLETE")
        return True

    @staticmethod
    def _apply_side(
        side: dict[Decimal, Decimal],
        rows: Sequence[Any],
    ) -> None:
        for price, quantity in _parse_levels(rows):
            if quantity == 0:
                side.pop(price, None)
            else:
                side[price] = quantity

    def apply_delta(
        self,
        *,
        bids: Sequence[Any],
        asks: Sequence[Any],
        sequence: int | None,
        event_at: datetime,
        known_at: datetime,
        event_id: str,
        buffered_after_snapshot: bool = False,
    ) -> bool:
        selected_event = _utc(event_at)
        selected_known = _utc(known_at)
        if not self._remember(str(event_id)):
            return False
        if self.state is not BitvavoBookState.VALID or self.sequence is None:
            self.require_reseed(selected_known, "DELTA_WITHOUT_VALID_SNAPSHOT")
            return False
        if sequence is None:
            self.require_reseed(selected_known, "DELTA_SEQUENCE_MISSING")
            return False
        selected_sequence = int(sequence)
        if selected_sequence <= self.sequence:
            if buffered_after_snapshot:
                # Bitvavo's documented sync procedure explicitly discards
                # buffered events at or before the accepted snapshot nonce.
                self.pre_snapshot_discarded += 1
                return False
            self.require_reseed(selected_known, "OUT_OF_ORDER_NONCE")
            return False
        if selected_sequence != self.sequence + 1:
            self.require_reseed(selected_known, "NONCE_GAP")
            return False
        next_bids, next_asks = self.bids.copy(), self.asks.copy()
        try:
            self._apply_side(next_bids, bids)
            self._apply_side(next_asks, asks)
            self._trim(next_bids, next_asks)
            self._validate_book(next_bids, next_asks)
        except ValueError as exc:
            self._transition(BitvavoBookState.INVALID, selected_known, str(exc))
            self._clear()
            self._transition(
                BitvavoBookState.RESEED_REQUIRED, selected_known, "FRESH_SNAPSHOT_REQUIRED"
            )
            return False
        self.bids = next_bids
        self.asks = next_asks
        self.sequence = selected_sequence
        self.last_event_at = selected_event
        self.last_known_at = selected_known
        self.applied_deltas += 1
        return True

    def check_stale(self, at: datetime) -> BitvavoBookState:
        selected = _utc(at)
        if (
            self.state is BitvavoBookState.VALID
            and self.last_known_at is not None
            and selected - self.last_known_at > self.stale_after
        ):
            self._transition(BitvavoBookState.STALE, selected, "BOOK_STALE")
            self._clear()
            self._transition(BitvavoBookState.RESEED_REQUIRED, selected, "FRESH_SNAPSHOT_REQUIRED")
        return self.state

    def features(self, at: datetime | None = None) -> dict[str, Any] | None:
        selected = _utc(at or self.last_known_at or datetime(1970, 1, 1, tzinfo=UTC))
        self.check_stale(selected)
        if self.state is not BitvavoBookState.VALID:
            return None
        self._validate_book(self.bids, self.asks)
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        bid_size = self.bids[best_bid]
        ask_size = self.asks[best_ask]
        mid = (best_bid + best_ask) / Decimal("2")
        spread = best_ask - best_bid
        total = bid_size + ask_size
        top_imbalance = (bid_size - ask_size) / total if total else Decimal("0")
        microprice = (best_ask * bid_size + best_bid * ask_size) / total if total else mid
        body = {
            "schema_version": L2_FEATURE_SCHEMA_VERSION,
            "reconstruction_version": L2_RECONSTRUCTION_VERSION,
            "market": self.market,
            "available_at": utc_iso(selected),
            "sequence": self.sequence,
            "snapshot_reference": self.snapshot_reference,
            "book_state": self.state.value,
            "best_bid": str(best_bid),
            "best_ask": str(best_ask),
            "spread": str(spread),
            "spread_bps": str(spread / mid * Decimal(10_000)),
            "mid": str(mid),
            "microprice": str(microprice),
            "top_level_imbalance": str(top_imbalance),
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "execution_authority": False,
            "orders_generated": 0,
        }
        return {**body, "feature_hash": stable_hash(body)}

    def finalize(self, at: datetime, reason: str = "REPLAY_END") -> None:
        selected = _utc(at)
        if self.state is BitvavoBookState.VALID:
            self._close_valid_interval(selected, reason)

    def snapshot(self, at: datetime | None = None) -> dict[str, Any]:
        selected = _utc(at or self.last_known_at or datetime.now(UTC))
        closed_valid = sum(interval.duration_seconds for interval in self.intervals)
        open_valid = (
            max(0.0, (selected - self._valid_start).total_seconds())
            if self.state is BitvavoBookState.VALID and self._valid_start is not None
            else 0.0
        )
        return {
            "schema_version": L2_RECONSTRUCTION_VERSION,
            "market": self.market,
            "state": self.state.value,
            "sequence": self.sequence,
            "snapshot_reference": self.snapshot_reference,
            "applied_deltas": self.applied_deltas,
            "duplicate_events": self.duplicate_events,
            "pre_snapshot_discarded": self.pre_snapshot_discarded,
            "reseed_count": self.reseed_count,
            "valid_seconds": closed_valid + open_valid,
            "valid_interval_count": len(self.intervals) + int(open_valid > 0),
            "failure_counts": dict(self.failure_counts),
            "transition_count": len(self.transitions),
            "feature_available": self.features(selected) is not None,
            "execution_authority": False,
            "orders_generated": 0,
        }


def transition_trace(machine: BitvavoL2StateMachine) -> list[dict[str, Any]]:
    return [row.to_dict() for row in machine.transitions]


def valid_intervals(machine: BitvavoL2StateMachine) -> list[dict[str, Any]]:
    return [row.to_dict() for row in machine.intervals]


__all__ = [
    "BitvavoBookState",
    "BitvavoL2StateMachine",
    "BookTransition",
    "BookValidityInterval",
    "L2_FEATURE_SCHEMA_VERSION",
    "L2_RECONSTRUCTION_VERSION",
    "transition_trace",
    "valid_intervals",
]
