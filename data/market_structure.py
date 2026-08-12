"""Point-in-time, replayable cross-exchange spot market-structure data.

P1.2 is a data-foundation module.  It deliberately has no broker client, order
route, strategy promotion or portfolio-allocation surface.  Bitvavo is recorded
as the sole execution venue; every other venue is research context only.

The design is conceptually informed by the read-only reference repositories:

* NautilusTrader: distinct event/initialization timestamps and snapshot/delta
  order-book replay (LGPL-3.0; concepts only, no implementation copied).
* Freqtrade: future-perturbation checks for detecting lookahead (GPL-3.0;
  concepts only, no implementation copied).
* LEAN: explicit instrument and quote-currency identity (Apache-2.0).
* Qlib: immutable, versioned dataset/feature identities (MIT).

vectorbt and PyBroker are intentionally deferred to later hypothesis screening
and validation.  Their local copies carry Commons-Clause terms, so no code is
used here.
"""

from __future__ import annotations

import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from core.contracts import normalize_market
from utils.common import sha256_file, stable_hash, stable_json

MARKET_STRUCTURE_SCHEMA_VERSION = "market_structure_platform_v1"
INSTRUMENT_SCHEMA_VERSION = "canonical_cross_venue_instrument_v1"
RAW_EVENT_SCHEMA_VERSION = "market_structure_raw_event_v1"
BOOK_REPLAY_SCHEMA_VERSION = "l2_book_replay_v1"
FEATURE_SCHEMA_VERSION = "market_structure_feature_schema_v1"
LABEL_SCHEMA_VERSION = "market_structure_research_label_v1"
DATASET_MANIFEST_SCHEMA_VERSION = "market_structure_dataset_manifest_v1"
SOURCE_CONFIG_VERSION = "p1_2_bounded_tier1_sources_v1"
DEFAULT_PRIMARY_ASSETS = ("BTC", "ETH", "SOL")
DEPTH_BANDS_BPS = (5, 10, 25, 50, 100)
ALLOWED_BUCKET_SECONDS = (1, 5, 15, 30, 60, 300)


class VenueRole(StrEnum):
    EXECUTION_PRIMARY = "EXECUTION_PRIMARY"
    SPOT_REFERENCE = "SPOT_REFERENCE"
    DERIVATIVES_CONTEXT = "DERIVATIVES_CONTEXT"
    HISTORICAL_ONLY = "HISTORICAL_ONLY"
    DISABLED = "DISABLED"
    UNRELIABLE = "UNRELIABLE"


class MarketType(StrEnum):
    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class DataLayer(StrEnum):
    RAW = "RAW"
    NORMALIZED = "NORMALIZED"
    FEATURE = "FEATURE"
    RESEARCH_LABEL = "RESEARCH_LABEL"


class EventType(StrEnum):
    TRADE = "TRADE"
    BOOK_SNAPSHOT = "BOOK_SNAPSHOT"
    BOOK_DELTA = "BOOK_DELTA"
    TICKER = "TICKER"
    FX_RATE = "FX_RATE"
    LIQUIDATION = "LIQUIDATION"
    DERIVATIVES_CONTEXT = "DERIVATIVES_CONTEXT"


class TimestampQuality(StrEnum):
    EXCHANGE_REPORTED = "EXCHANGE_REPORTED"
    REST_OBSERVED_ONLY = "REST_OBSERVED_ONLY"
    INFERRED = "INFERRED"
    INVALID = "INVALID"


class ClockQuality(StrEnum):
    CLOCK_OK = "CLOCK_OK"
    CLOCK_SUSPECT = "CLOCK_SUSPECT"
    CLOCK_INVALID = "CLOCK_INVALID"
    CLOCK_NOT_EVALUABLE = "CLOCK_NOT_EVALUABLE"


class BookQuality(StrEnum):
    BOOK_UNINITIALIZED = "BOOK_UNINITIALIZED"
    BOOK_SYNCING = "BOOK_SYNCING"
    BOOK_VALID = "BOOK_VALID"
    BOOK_GAPPED = "BOOK_GAPPED"
    BOOK_STALE = "BOOK_STALE"
    BOOK_INVALID = "BOOK_INVALID"


class AggressorSemantics(StrEnum):
    AGGRESSOR_EXCHANGE_REPORTED = "AGGRESSOR_EXCHANGE_REPORTED"
    AGGRESSOR_INFERRED = "AGGRESSOR_INFERRED"
    AGGRESSOR_UNKNOWN = "AGGRESSOR_UNKNOWN"


class Missingness(StrEnum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    VENUE_OFFLINE = "VENUE_OFFLINE"
    UNSUPPORTED = "UNSUPPORTED"
    QUALITY_INVALID = "QUALITY_INVALID"


class HistoricalAvailability(StrEnum):
    TRUE_HISTORICAL_EVENT_DATA = "TRUE_HISTORICAL_EVENT_DATA"
    RECONSTRUCTED_DATA = "RECONSTRUCTED_DATA"
    CURRENT_COLLECTION_ONLY = "CURRENT_COLLECTION_ONLY"
    HISTORICAL_BARS_ONLY = "HISTORICAL_BARS_ONLY"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ReadinessState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    COLLECTING = "COLLECTING"
    PARTIAL = "PARTIAL"
    QUALITY_FAILED = "QUALITY_FAILED"
    RESEARCH_USABLE = "RESEARCH_USABLE"
    ROBUSTNESS_USABLE = "ROBUSTNESS_USABLE"


def _utc(value: datetime | str | int | float) -> datetime:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) >= 1e17:
            numeric /= 1e9
        elif abs(numeric) >= 1e14:
            numeric /= 1e6
        elif abs(numeric) >= 1e11:
            numeric /= 1e3
        selected = datetime.fromtimestamp(numeric, tz=UTC)
    else:
        selected = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if selected.tzinfo is None:
        raise ValueError("canonical market-structure timestamps require a timezone")
    return selected.astimezone(UTC)


def _decimal(value: Any, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("numeric market-structure value is invalid") from exc
    if not selected.is_finite():
        raise ValueError("numeric market-structure value must be finite")
    if positive and selected <= 0:
        raise ValueError("numeric market-structure value must be positive")
    if nonnegative and selected < 0:
        raise ValueError("numeric market-structure value must be non-negative")
    return selected


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(selected) for key, selected in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(selected) for selected in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True, slots=True)
class CanonicalInstrument:
    canonical_asset_id: str
    base_asset: str
    quote_asset: str
    venue: str
    venue_symbol: str
    market_type: MarketType
    venue_role: VenueRole
    price_precision: int | None = None
    quantity_precision: int | None = None
    minimum_quantity: Decimal | None = None
    minimum_notional: Decimal | None = None
    market_status: str = "UNKNOWN"
    listing_timestamp: datetime | None = None
    liquidity_tier: str = "TIER_3"
    execution_allowed: bool = False
    shariah_spot_eligible: bool = False

    def __post_init__(self) -> None:
        venue = self.venue.casefold()
        base = self.base_asset.upper()
        quote = self.quote_asset.upper()
        if not self.canonical_asset_id or not base or not quote or not self.venue_symbol:
            raise ValueError("instrument identity fields cannot be empty")
        if self.price_precision is not None and self.price_precision < 0:
            raise ValueError("price precision cannot be negative")
        if self.quantity_precision is not None and self.quantity_precision < 0:
            raise ValueError("quantity precision cannot be negative")
        if self.liquidity_tier not in {"TIER_1", "TIER_2", "TIER_3"}:
            raise ValueError("unsupported liquidity tier")
        if self.execution_allowed and (
            venue != "bitvavo"
            or self.venue_role is not VenueRole.EXECUTION_PRIMARY
            or self.market_type is not MarketType.SPOT
            or not self.shariah_spot_eligible
        ):
            raise ValueError("only Shariah-eligible Bitvavo spot can be execution-enabled")
        if venue != "bitvavo" and self.execution_allowed:
            raise ValueError("reference venues have zero execution authority")
        if self.market_type is not MarketType.SPOT and self.execution_allowed:
            raise ValueError("derivatives can never receive execution authority")
        if self.listing_timestamp is not None:
            _utc(self.listing_timestamp)

    @property
    def canonical_market(self) -> str:
        return normalize_market(f"{self.base_asset}-{self.quote_asset}")

    @property
    def instrument_id(self) -> str:
        return stable_hash(
            {
                "schema": INSTRUMENT_SCHEMA_VERSION,
                "asset": self.canonical_asset_id,
                "venue": self.venue.casefold(),
                "symbol": self.venue_symbol,
                "market_type": self.market_type.value,
                "base": self.base_asset.upper(),
                "quote": self.quote_asset.upper(),
            },
            length=48,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**_json_safe(asdict(self)), "instrument_id": self.instrument_id}


class InstrumentRegistry:
    def __init__(self, instruments: Sequence[CanonicalInstrument]) -> None:
        if not instruments:
            raise ValueError("instrument registry cannot be empty")
        by_id: dict[str, CanonicalInstrument] = {}
        by_venue_symbol: dict[tuple[str, str, MarketType], CanonicalInstrument] = {}
        for instrument in instruments:
            if instrument.instrument_id in by_id:
                raise ValueError("duplicate canonical instrument identity")
            key = (
                instrument.venue.casefold(),
                instrument.venue_symbol.upper(),
                instrument.market_type,
            )
            if key in by_venue_symbol:
                raise ValueError("duplicate venue symbol mapping")
            by_id[instrument.instrument_id] = instrument
            by_venue_symbol[key] = instrument
        execution = [row for row in instruments if row.execution_allowed]
        if any(row.venue.casefold() != "bitvavo" for row in execution):
            raise ValueError("Bitvavo must remain the only execution venue")
        self._instruments = tuple(instruments)
        self._by_id = by_id
        self._by_venue_symbol = by_venue_symbol

    @property
    def instruments(self) -> tuple[CanonicalInstrument, ...]:
        return self._instruments

    @property
    def registry_hash(self) -> str:
        return stable_hash([row.to_dict() for row in self._instruments], length=64)

    def resolve(
        self,
        venue: str,
        venue_symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> CanonicalInstrument:
        key = (venue.casefold(), venue_symbol.upper(), market_type)
        if key not in self._by_venue_symbol:
            raise KeyError(f"unknown venue instrument: {key}")
        return self._by_venue_symbol[key]

    def by_id(self, instrument_id: str) -> CanonicalInstrument:
        return self._by_id[instrument_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INSTRUMENT_SCHEMA_VERSION,
            "registry_hash": self.registry_hash,
            "instruments": [row.to_dict() for row in self._instruments],
        }


def initial_instrument_registry() -> InstrumentRegistry:
    """Return the bounded P1.2 registry; metadata gaps remain explicit."""

    instruments: list[CanonicalInstrument] = []
    for asset, tier in (
        ("BTC", "TIER_1"),
        ("ETH", "TIER_1"),
        ("SOL", "TIER_1"),
        ("LINK", "TIER_2"),
    ):
        instruments.extend(
            [
                CanonicalInstrument(
                    canonical_asset_id=f"CRYPTO:{asset}",
                    base_asset=asset,
                    quote_asset="EUR",
                    venue="bitvavo",
                    venue_symbol=f"{asset}-EUR",
                    market_type=MarketType.SPOT,
                    venue_role=VenueRole.EXECUTION_PRIMARY,
                    liquidity_tier=tier,
                    execution_allowed=True,
                    shariah_spot_eligible=True,
                ),
                CanonicalInstrument(
                    canonical_asset_id=f"CRYPTO:{asset}",
                    base_asset=asset,
                    quote_asset="EUR",
                    venue="kraken",
                    venue_symbol=("XBT/EUR" if asset == "BTC" else f"{asset}/EUR"),
                    market_type=MarketType.SPOT,
                    venue_role=VenueRole.SPOT_REFERENCE,
                    liquidity_tier=tier,
                    execution_allowed=False,
                    shariah_spot_eligible=True,
                ),
                CanonicalInstrument(
                    canonical_asset_id=f"CRYPTO:{asset}",
                    base_asset=asset,
                    quote_asset="USDT",
                    venue="mexc",
                    venue_symbol=f"{asset}USDT",
                    market_type=MarketType.SPOT,
                    venue_role=VenueRole.SPOT_REFERENCE,
                    liquidity_tier=tier,
                    execution_allowed=False,
                    shariah_spot_eligible=True,
                ),
            ]
        )
    for asset in ("BTC", "ETH", "SOL", "LINK"):
        instruments.append(
            CanonicalInstrument(
                canonical_asset_id=f"CRYPTO:{asset}",
                base_asset=asset,
                quote_asset="USDT",
                venue="mexc",
                venue_symbol=f"{asset}_USDT",
                market_type=MarketType.PERPETUAL,
                venue_role=VenueRole.DERIVATIVES_CONTEXT,
                execution_allowed=False,
                shariah_spot_eligible=False,
            )
        )
    for asset in ("BTC", "ETH"):
        instruments.append(
            CanonicalInstrument(
                canonical_asset_id=f"CRYPTO:{asset}",
                base_asset=asset,
                quote_asset="USD",
                venue="deribit",
                venue_symbol=f"{asset}-DERIVATIVES-CONTEXT",
                market_type=MarketType.OPTION,
                venue_role=VenueRole.DERIVATIVES_CONTEXT,
                execution_allowed=False,
                shariah_spot_eligible=False,
            )
        )
    return InstrumentRegistry(instruments)


@dataclass(frozen=True, slots=True)
class EventTimestamps:
    exchange_event_timestamp: datetime | None
    local_receive_timestamp: datetime
    normalized_event_timestamp: datetime
    persisted_timestamp: datetime
    quality: TimestampQuality
    request_start: datetime | None = None
    response_received: datetime | None = None

    def __post_init__(self) -> None:
        receive = _utc(self.local_receive_timestamp)
        normalized = _utc(self.normalized_event_timestamp)
        persisted = _utc(self.persisted_timestamp)
        event = _utc(self.exchange_event_timestamp) if self.exchange_event_timestamp else None
        request = _utc(self.request_start) if self.request_start else None
        response = _utc(self.response_received) if self.response_received else None
        if persisted < receive:
            raise ValueError("persisted timestamp cannot precede receive timestamp")
        if request and response and response < request:
            raise ValueError("REST response cannot precede request start")
        if self.quality is TimestampQuality.EXCHANGE_REPORTED and event is None:
            raise ValueError("exchange-reported timestamp quality requires an event timestamp")
        if self.quality is TimestampQuality.REST_OBSERVED_ONLY and (
            request is None or response is None
        ):
            raise ValueError("REST-observed timestamps require request and response times")
        if event is None and normalized != receive:
            raise ValueError("events without exchange time normalize to local receive time")

    @property
    def available_at(self) -> datetime:
        return _utc(self.local_receive_timestamp)

    @property
    def observed_latency_ms(self) -> float | None:
        if self.exchange_event_timestamp is None:
            return None
        return (
            _utc(self.local_receive_timestamp) - _utc(self.exchange_event_timestamp)
        ).total_seconds() * 1_000.0

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class ClockAssessment:
    venue: str
    quality: ClockQuality
    inferred_offset_ms: float | None
    network_roundtrip_ms: float | None
    sample_count: int
    maximum_absolute_offset_ms: float | None
    negative_latency_count: int
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class ClockMonitor:
    def __init__(
        self,
        *,
        ok_absolute_offset_ms: float = 2_000.0,
        invalid_absolute_offset_ms: float = 10_000.0,
    ) -> None:
        if ok_absolute_offset_ms <= 0 or invalid_absolute_offset_ms <= ok_absolute_offset_ms:
            raise ValueError("clock-quality thresholds are invalid")
        self.ok_absolute_offset_ms = ok_absolute_offset_ms
        self.invalid_absolute_offset_ms = invalid_absolute_offset_ms
        self._samples: dict[str, list[EventTimestamps]] = defaultdict(list)

    def observe(self, venue: str, timestamps: EventTimestamps) -> None:
        self._samples[venue.casefold()].append(timestamps)

    def assess(self, venue: str) -> ClockAssessment:
        rows = self._samples.get(venue.casefold(), [])
        latencies = [row.observed_latency_ms for row in rows if row.observed_latency_ms is not None]
        rest_roundtrips = [
            (_utc(row.response_received) - _utc(row.request_start)).total_seconds() * 1_000
            for row in rows
            if row.request_start is not None and row.response_received is not None
        ]
        if not latencies:
            return ClockAssessment(
                venue=venue.casefold(),
                quality=ClockQuality.CLOCK_NOT_EVALUABLE,
                inferred_offset_ms=None,
                network_roundtrip_ms=(
                    statistics.median(rest_roundtrips) if rest_roundtrips else None
                ),
                sample_count=len(rows),
                maximum_absolute_offset_ms=None,
                negative_latency_count=0,
                reason_codes=("NO_EXCHANGE_EVENT_TIMESTAMPS",),
            )
        median = float(statistics.median(latencies))
        maximum = max(abs(float(value)) for value in latencies)
        negative = sum(float(value) < -self.ok_absolute_offset_ms for value in latencies)
        reasons: list[str] = []
        if negative:
            reasons.append("EXCHANGE_CLOCK_AHEAD_OF_RECEIVE_BOUND")
        if maximum > self.invalid_absolute_offset_ms:
            reasons.append("OFFSET_EXCEEDS_INVALID_BOUND")
        elif maximum > self.ok_absolute_offset_ms:
            reasons.append("OFFSET_EXCEEDS_OK_BOUND")
        quality = (
            ClockQuality.CLOCK_INVALID
            if negative or maximum > self.invalid_absolute_offset_ms
            else (
                ClockQuality.CLOCK_SUSPECT
                if maximum > self.ok_absolute_offset_ms
                else ClockQuality.CLOCK_OK
            )
        )
        return ClockAssessment(
            venue=venue.casefold(),
            quality=quality,
            inferred_offset_ms=median,
            network_roundtrip_ms=(statistics.median(rest_roundtrips) if rest_roundtrips else None),
            sample_count=len(rows),
            maximum_absolute_offset_ms=maximum,
            negative_latency_count=negative,
            reason_codes=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class RawMarketEvent:
    venue: str
    instrument_id: str
    event_type: EventType
    timestamps: EventTimestamps
    payload: Mapping[str, Any]
    exchange_event_id: str | None = None
    sequence: int | None = None
    previous_sequence: int | None = None
    source_config_version: str = SOURCE_CONFIG_VERSION

    def __post_init__(self) -> None:
        if not self.venue or not self.instrument_id:
            raise ValueError("raw event venue and instrument are required")
        if not isinstance(self.payload, Mapping):
            raise ValueError("raw event payload must be a mapping")

    @property
    def raw_payload_hash(self) -> str:
        return stable_hash(self.payload, length=64)

    @property
    def event_id(self) -> str:
        identity: Any = self.exchange_event_id
        if not identity:
            identity = {
                "instrument": self.instrument_id,
                "type": self.event_type.value,
                "event_time": self.timestamps.normalized_event_timestamp,
                "receive_time": self.timestamps.local_receive_timestamp,
                "sequence": self.sequence,
                "payload": self.raw_payload_hash,
            }
        return stable_hash(
            [RAW_EVENT_SCHEMA_VERSION, self.venue.casefold(), self.event_type.value, identity],
            length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RAW_EVENT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "venue": self.venue.casefold(),
            "instrument_id": self.instrument_id,
            "event_type": self.event_type.value,
            **self.timestamps.to_dict(),
            "available_at": self.timestamps.available_at.isoformat(),
            "exchange_event_id": self.exchange_event_id,
            "sequence": self.sequence,
            "previous_sequence": self.previous_sequence,
            "raw_payload_hash": self.raw_payload_hash,
            "raw_payload": stable_json(self.payload),
            "source_config_version": self.source_config_version,
            "orders_generated": 0,
        }


@dataclass(frozen=True, slots=True)
class TradeEvent:
    raw: RawMarketEvent
    price: Decimal
    quantity: Decimal
    quote_notional: Decimal
    aggressor_side: str | None
    aggressor_semantics: AggressorSemantics
    aggressor_inference_method: str | None = None

    def __post_init__(self) -> None:
        if self.raw.event_type is not EventType.TRADE:
            raise ValueError("TradeEvent requires a raw TRADE event")
        _decimal(self.price, positive=True)
        _decimal(self.quantity, positive=True)
        _decimal(self.quote_notional, positive=True)
        side = str(self.aggressor_side or "").casefold()
        if side and side not in {"buy", "sell"}:
            raise ValueError("aggressor side must be buy, sell or unknown")
        if self.aggressor_semantics is AggressorSemantics.AGGRESSOR_INFERRED and not (
            self.aggressor_inference_method
        ):
            raise ValueError("inferred aggressor side requires an inference method")
        if self.aggressor_semantics is AggressorSemantics.AGGRESSOR_UNKNOWN and side:
            raise ValueError("unknown aggressor semantics cannot carry a side")

    @property
    def trade_identity(self) -> str:
        if self.raw.exchange_event_id:
            identity: Any = [self.raw.venue.casefold(), self.raw.exchange_event_id]
        else:
            identity = [
                self.raw.venue.casefold(),
                self.raw.instrument_id,
                self.raw.timestamps.normalized_event_timestamp,
                str(self.price),
                str(self.quantity),
                self.aggressor_side,
                self.raw.sequence,
            ]
        return stable_hash(["venue_trade_identity_v1", identity], length=64)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.raw.to_dict(),
            "trade_identity": self.trade_identity,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "quote_notional": str(self.quote_notional),
            "aggressor_side": self.aggressor_side,
            "aggressor_semantics": self.aggressor_semantics.value,
            "aggressor_inference_method": self.aggressor_inference_method,
        }


def infer_aggressor_side(
    *,
    trade_price: Decimal | float | str,
    exchange_side: str | None = None,
    best_bid: Decimal | float | str | None = None,
    best_ask: Decimal | float | str | None = None,
    previous_trade_price: Decimal | float | str | None = None,
) -> tuple[str | None, AggressorSemantics, str | None]:
    if exchange_side and exchange_side.casefold() in {"buy", "sell"}:
        return (
            exchange_side.casefold(),
            AggressorSemantics.AGGRESSOR_EXCHANGE_REPORTED,
            None,
        )
    price = _decimal(trade_price, positive=True)
    if best_ask is not None and price >= _decimal(best_ask, positive=True):
        return "buy", AggressorSemantics.AGGRESSOR_INFERRED, "QUOTE_TEST_AT_ASK"
    if best_bid is not None and price <= _decimal(best_bid, positive=True):
        return "sell", AggressorSemantics.AGGRESSOR_INFERRED, "QUOTE_TEST_AT_BID"
    if previous_trade_price is not None:
        previous = _decimal(previous_trade_price, positive=True)
        if price > previous:
            return "buy", AggressorSemantics.AGGRESSOR_INFERRED, "TICK_TEST_UP"
        if price < previous:
            return "sell", AggressorSemantics.AGGRESSOR_INFERRED, "TICK_TEST_DOWN"
    return None, AggressorSemantics.AGGRESSOR_UNKNOWN, None


def deduplicate_trades(
    trades: Iterable[TradeEvent],
) -> tuple[tuple[TradeEvent, ...], tuple[dict[str, Any], ...]]:
    unique: list[TradeEvent] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trade in trades:
        identity = trade.trade_identity
        if identity in seen:
            rejected.append(
                {
                    "trade_identity": identity,
                    "reason": "DUPLICATE_TRADE",
                    "event_id": trade.raw.event_id,
                }
            )
            continue
        seen.add(identity)
        unique.append(trade)
    return tuple(unique), tuple(rejected)


@dataclass(frozen=True, slots=True)
class BookEvent:
    raw: RawMarketEvent
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]

    def __post_init__(self) -> None:
        if self.raw.event_type not in {EventType.BOOK_SNAPSHOT, EventType.BOOK_DELTA}:
            raise ValueError("BookEvent requires snapshot or delta raw event")
        for price, quantity in (*self.bids, *self.asks):
            _decimal(price, positive=True)
            _decimal(quantity, nonnegative=True)


@dataclass(frozen=True, slots=True)
class BookFeatureSnapshot:
    instrument_id: str
    venue: str
    sequence: int | None
    event_timestamp: datetime
    available_at: datetime
    quality: BookQuality
    best_bid: Decimal | None
    best_ask: Decimal | None
    mid: Decimal | None
    spread: Decimal | None
    spread_bps: Decimal | None
    microprice: Decimal | None
    top_level_imbalance: Decimal | None
    depth: Mapping[str, Decimal | None]
    book_type: str = "L2_PRICE_AGGREGATED"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    @property
    def feature_hash(self) -> str:
        return stable_hash(_json_safe(asdict(self)), length=64)

    def to_dict(self) -> dict[str, Any]:
        return {**_json_safe(asdict(self)), "feature_hash": self.feature_hash}


class OrderBookReplayer:
    """Deterministic L2 snapshot/delta replay with fail-closed gaps."""

    def __init__(
        self,
        *,
        instrument_id: str,
        venue: str,
        maximum_levels: int = 500,
        stale_after: timedelta = timedelta(seconds=30),
    ) -> None:
        if maximum_levels < 1 or stale_after <= timedelta(0):
            raise ValueError("book replay bounds are invalid")
        self.instrument_id = instrument_id
        self.venue = venue.casefold()
        self.maximum_levels = maximum_levels
        self.stale_after = stale_after
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.sequence: int | None = None
        self.state = BookQuality.BOOK_UNINITIALIZED
        self.last_event_timestamp: datetime | None = None
        self.last_available_at: datetime | None = None
        self.gap_count = 0
        self.reconnect_count = 0
        self.applied_event_ids: set[str] = set()

    @staticmethod
    def _apply_side(
        side: dict[Decimal, Decimal],
        levels: Sequence[tuple[Decimal, Decimal]],
    ) -> None:
        for raw_price, raw_quantity in levels:
            price = _decimal(raw_price, positive=True)
            quantity = _decimal(raw_quantity, nonnegative=True)
            if quantity == 0:
                side.pop(price, None)
            else:
                side[price] = quantity

    def _trim(self) -> None:
        self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.maximum_levels])
        self.asks = dict(sorted(self.asks.items())[: self.maximum_levels])

    def _validate(self) -> None:
        if not self.bids or not self.asks:
            raise ValueError("order book must contain both sides")
        if max(self.bids) >= min(self.asks):
            raise ValueError("order book is locked or crossed")

    def reconnect_reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.sequence = None
        self.state = BookQuality.BOOK_SYNCING
        self.last_event_timestamp = None
        self.last_available_at = None
        self.applied_event_ids.clear()
        self.reconnect_count += 1

    def apply(self, event: BookEvent) -> bool:
        if (
            event.raw.instrument_id != self.instrument_id
            or event.raw.venue.casefold() != self.venue
        ):
            raise ValueError("book event does not match replay identity")
        if event.raw.event_id in self.applied_event_ids:
            return False
        if event.raw.event_type is EventType.BOOK_SNAPSHOT:
            next_bids: dict[Decimal, Decimal] = {}
            next_asks: dict[Decimal, Decimal] = {}
            self._apply_side(next_bids, event.bids)
            self._apply_side(next_asks, event.asks)
            self.bids, self.asks = next_bids, next_asks
            self._trim()
            try:
                self._validate()
            except ValueError:
                self.state = BookQuality.BOOK_INVALID
                self.bids.clear()
                self.asks.clear()
                raise
            self.sequence = event.raw.sequence
            self.state = BookQuality.BOOK_VALID
        else:
            if self.state is not BookQuality.BOOK_VALID:
                raise RuntimeError("book delta requires a valid snapshot")
            if event.raw.sequence is not None and self.sequence is not None:
                expected = self.sequence + 1
                previous_matches = (
                    event.raw.previous_sequence is None
                    or event.raw.previous_sequence == self.sequence
                )
                if event.raw.sequence != expected or not previous_matches:
                    self.state = BookQuality.BOOK_GAPPED
                    self.gap_count += 1
                    self.bids.clear()
                    self.asks.clear()
                    raise RuntimeError("book sequence gap; fresh snapshot required")
            prior_bids, prior_asks = self.bids.copy(), self.asks.copy()
            try:
                self._apply_side(self.bids, event.bids)
                self._apply_side(self.asks, event.asks)
                self._trim()
                self._validate()
            except ValueError:
                self.bids, self.asks = prior_bids, prior_asks
                self.state = BookQuality.BOOK_INVALID
                raise
            if event.raw.sequence is not None:
                self.sequence = event.raw.sequence
        self.last_event_timestamp = event.raw.timestamps.normalized_event_timestamp
        self.last_available_at = event.raw.timestamps.available_at
        self.applied_event_ids.add(event.raw.event_id)
        return True

    def quality_at(self, now: datetime) -> BookQuality:
        selected = _utc(now)
        if self.state is BookQuality.BOOK_VALID and self.last_available_at is not None:
            if selected - self.last_available_at > self.stale_after:
                return BookQuality.BOOK_STALE
        return self.state

    def features(self, *, now: datetime | None = None) -> BookFeatureSnapshot:
        # An uninitialized replay must not acquire wall-clock-dependent output.
        selected_now = _utc(now or self.last_available_at or datetime(1970, 1, 1, tzinfo=UTC))
        quality = self.quality_at(selected_now)
        if quality is not BookQuality.BOOK_VALID:
            return BookFeatureSnapshot(
                instrument_id=self.instrument_id,
                venue=self.venue,
                sequence=self.sequence,
                event_timestamp=self.last_event_timestamp or selected_now,
                available_at=self.last_available_at or selected_now,
                quality=quality,
                best_bid=None,
                best_ask=None,
                mid=None,
                spread=None,
                spread_bps=None,
                microprice=None,
                top_level_imbalance=None,
                depth={
                    f"{side}_depth_{band}bps_quote": None
                    for band in DEPTH_BANDS_BPS
                    for side in ("bid", "ask")
                },
            )
        best_bid = max(self.bids)
        best_ask = min(self.asks)
        bid_size = self.bids[best_bid]
        ask_size = self.asks[best_ask]
        mid = (best_bid + best_ask) / Decimal("2")
        spread = best_ask - best_bid
        top_total = bid_size + ask_size
        microprice = (best_ask * bid_size + best_bid * ask_size) / top_total if top_total else mid
        top_imbalance = (bid_size - ask_size) / top_total if top_total else Decimal("0")
        depth: dict[str, Decimal | None] = {}
        for band in DEPTH_BANDS_BPS:
            fraction = Decimal(band) / Decimal(10_000)
            lower, upper = mid * (Decimal("1") - fraction), mid * (Decimal("1") + fraction)
            bid_quote = sum(
                (price * quantity for price, quantity in self.bids.items() if price >= lower),
                Decimal("0"),
            )
            ask_quote = sum(
                (price * quantity for price, quantity in self.asks.items() if price <= upper),
                Decimal("0"),
            )
            total = bid_quote + ask_quote
            depth[f"bid_depth_{band}bps_quote"] = bid_quote
            depth[f"ask_depth_{band}bps_quote"] = ask_quote
            depth[f"depth_imbalance_{band}bps"] = (
                (bid_quote - ask_quote) / total if total else Decimal("0")
            )
        bid_distances = [float((mid - price) / mid) for price in self.bids]
        ask_distances = [float((price - mid) / mid) for price in self.asks]
        bid_cumulative = np.cumsum([float(value) for value in self.bids.values()])
        ask_cumulative = np.cumsum([float(value) for value in self.asks.values()])
        depth["bid_book_slope_l2"] = (
            Decimal(str(np.polyfit(bid_distances, bid_cumulative, 1)[0]))
            if len(bid_distances) >= 2
            else None
        )
        depth["ask_book_slope_l2"] = (
            Decimal(str(np.polyfit(ask_distances, ask_cumulative, 1)[0]))
            if len(ask_distances) >= 2
            else None
        )
        return BookFeatureSnapshot(
            instrument_id=self.instrument_id,
            venue=self.venue,
            sequence=self.sequence,
            event_timestamp=self.last_event_timestamp or selected_now,
            available_at=self.last_available_at or selected_now,
            quality=quality,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            spread=spread,
            spread_bps=spread / mid * Decimal(10_000),
            microprice=microprice,
            top_level_imbalance=top_imbalance,
            depth=depth,
        )


@dataclass(frozen=True, slots=True)
class FXObservation:
    base_currency: str
    quote_currency: str
    rate: Decimal
    event_timestamp: datetime
    available_at: datetime
    source: str
    quality: str = "DIRECT_MARKET_OBSERVATION"

    def __post_init__(self) -> None:
        if self.base_currency.upper() == self.quote_currency.upper():
            raise ValueError("FX observations require distinct currencies")
        _decimal(self.rate, positive=True)
        if _utc(self.available_at) < _utc(self.event_timestamp) - timedelta(days=7):
            raise ValueError("FX availability timestamp is implausibly early")

    @property
    def observation_id(self) -> str:
        return stable_hash(_json_safe(asdict(self)), length=64)


class PointInTimeFXBook:
    def __init__(self, observations: Iterable[FXObservation]) -> None:
        rows = sorted(
            observations,
            key=lambda row: (row.base_currency, row.quote_currency, _utc(row.available_at)),
        )
        self._rows: dict[tuple[str, str], list[FXObservation]] = defaultdict(list)
        for row in rows:
            self._rows[(row.base_currency.upper(), row.quote_currency.upper())].append(row)

    def rate(
        self,
        base_currency: str,
        quote_currency: str,
        *,
        at: datetime,
        maximum_age: timedelta,
    ) -> tuple[Decimal | None, Missingness, FXObservation | None]:
        base, quote = base_currency.upper(), quote_currency.upper()
        selected_at = _utc(at)
        if base == quote:
            return Decimal("1"), Missingness.PRESENT, None
        direct = [
            row
            for row in self._rows.get((base, quote), [])
            if _utc(row.available_at) <= selected_at
        ]
        inverse = [
            row
            for row in self._rows.get((quote, base), [])
            if _utc(row.available_at) <= selected_at
        ]
        if direct:
            selected = direct[-1]
            value = selected.rate
        elif inverse:
            selected = inverse[-1]
            value = Decimal("1") / selected.rate
        else:
            return None, Missingness.UNAVAILABLE, None
        age = selected_at - _utc(selected.available_at)
        if age > maximum_age:
            return None, Missingness.STALE, selected
        return value, Missingness.PRESENT, selected


@dataclass(frozen=True, slots=True)
class VenueQuote:
    instrument_id: str
    canonical_asset_id: str
    venue: str
    quote_currency: str
    best_bid: Decimal
    best_ask: Decimal
    timestamps: EventTimestamps
    clock_quality: ClockQuality
    source_kind: str = "WEBSOCKET_L1"

    def __post_init__(self) -> None:
        bid = _decimal(self.best_bid, positive=True)
        ask = _decimal(self.best_ask, positive=True)
        if bid >= ask:
            raise ValueError("venue quote is locked or crossed")

    @property
    def mid(self) -> Decimal:
        return (self.best_bid + self.best_ask) / Decimal("2")


def normalize_quote_to_eur(
    quote: VenueQuote,
    *,
    fx: PointInTimeFXBook,
    maximum_fx_age: timedelta = timedelta(minutes=5),
) -> dict[str, Any]:
    rate, missingness, observation = fx.rate(
        quote.quote_currency,
        "EUR",
        at=quote.timestamps.available_at,
        maximum_age=maximum_fx_age,
    )
    normalized = quote.mid * rate if rate is not None else None
    return {
        "instrument_id": quote.instrument_id,
        "canonical_asset_id": quote.canonical_asset_id,
        "venue": quote.venue.casefold(),
        "raw_bid": str(quote.best_bid),
        "raw_ask": str(quote.best_ask),
        "raw_mid": str(quote.mid),
        "quote_currency": quote.quote_currency.upper(),
        "conversion_rate": str(rate) if rate is not None else None,
        "conversion_timestamp": (
            observation.event_timestamp.isoformat()
            if observation is not None
            else quote.timestamps.available_at.isoformat()
        ),
        "conversion_source": observation.source if observation is not None else "IDENTITY_EUR",
        "normalized_eur_mid": str(normalized) if normalized is not None else None,
        "fx_missingness": missingness.value,
        "available_at": quote.timestamps.available_at.isoformat(),
        "clock_quality": quote.clock_quality.value,
        "source_kind": quote.source_kind,
    }


def align_cross_venue_quotes(
    quotes: Sequence[VenueQuote],
    *,
    fx: PointInTimeFXBook,
    primary_venue: str = "bitvavo",
    bucket_seconds: int = 5,
    maximum_freshness: timedelta = timedelta(seconds=5),
) -> pd.DataFrame:
    if bucket_seconds not in ALLOWED_BUCKET_SECONDS:
        raise ValueError("unsupported lead/lag bucket resolution")
    if not quotes:
        return pd.DataFrame()
    normalized = [normalize_quote_to_eur(row, fx=fx) for row in quotes]
    frame = pd.DataFrame(normalized)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["normalized_eur_mid"] = pd.to_numeric(frame["normalized_eur_mid"], errors="coerce")
    frame["bucket"] = frame["available_at"].dt.floor(f"{bucket_seconds}s")
    rows: list[dict[str, Any]] = []
    for (asset, bucket), selected in frame.groupby(["canonical_asset_id", "bucket"], sort=True):
        bucket_end = pd.Timestamp(bucket.to_pydatetime() + timedelta(seconds=bucket_seconds))
        by_venue: dict[str, dict[str, Any]] = {}
        for venue, venue_rows in selected.groupby("venue"):
            latest = venue_rows.sort_values("available_at").iloc[-1]
            age = (bucket_end - latest["available_at"]).total_seconds()
            by_venue[str(venue)] = {
                "instrument_id": latest["instrument_id"],
                "raw_mid": latest["raw_mid"],
                "normalized_eur_mid": (
                    float(latest["normalized_eur_mid"])
                    if pd.notna(latest["normalized_eur_mid"])
                    else None
                ),
                "quote_currency": latest["quote_currency"],
                "conversion_rate": latest["conversion_rate"],
                "available_at": latest["available_at"].isoformat(),
                "age_seconds": float(age),
                "freshness": (
                    Missingness.PRESENT.value
                    if age <= maximum_freshness.total_seconds()
                    else Missingness.STALE.value
                ),
                "clock_quality": latest["clock_quality"],
            }
        primary = by_venue.get(primary_venue.casefold())
        references = [
            float(value["normalized_eur_mid"])
            for venue, value in by_venue.items()
            if venue != primary_venue.casefold()
            and value["normalized_eur_mid"] is not None
            and value["freshness"] == Missingness.PRESENT.value
            and value["clock_quality"] == ClockQuality.CLOCK_OK.value
        ]
        primary_mid = (
            float(primary["normalized_eur_mid"])
            if primary and primary["normalized_eur_mid"] is not None
            else None
        )
        reference = float(statistics.median(references)) if references else None
        premium_bps = (
            (primary_mid / reference - 1.0) * 10_000.0
            if primary_mid is not None
            and reference
            and primary
            and primary["freshness"] == Missingness.PRESENT.value
            else None
        )
        rows.append(
            {
                "canonical_asset_id": asset,
                "bucket_start": bucket.isoformat(),
                "bucket_seconds": bucket_seconds,
                "venues": by_venue,
                "primary_venue": primary_venue.casefold(),
                "primary_eur_mid": primary_mid,
                "reference_eur_mid": reference,
                "primary_reference_premium_bps": premium_bps,
                "reference_venue_count": len(references),
                "alignment_quality": (
                    "ALIGNED" if premium_bps is not None else "REFERENCE_DATA_UNAVAILABLE"
                ),
                "execution_authority": False,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    feature_name: str
    group: str
    version: str
    source: tuple[str, ...]
    lookback: str
    timestamp_semantics: str
    required_quality_state: str
    units: str
    missingness_semantics: str

    @property
    def feature_id(self) -> str:
        return stable_hash(asdict(self), length=48)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "feature_id": self.feature_id}


def market_structure_feature_schema() -> tuple[FeatureDefinition, ...]:
    rows = (
        ("mid_price", "PRICE", ("L1_OR_L2",), "NONE", "BOOK_VALID", "quote/base"),
        ("spread_bps", "SPREAD", ("L1_OR_L2",), "NONE", "BOOK_VALID", "bps"),
        ("depth_5bps_quote", "DEPTH", ("L2",), "NONE", "BOOK_VALID", "quote"),
        ("depth_25bps_quote", "DEPTH", ("L2",), "NONE", "BOOK_VALID", "quote"),
        ("depth_100bps_quote", "DEPTH", ("L2",), "NONE", "BOOK_VALID", "quote"),
        ("top_level_imbalance", "BOOK_IMBALANCE", ("L1",), "NONE", "BOOK_VALID", "fraction"),
        ("depth_imbalance_25bps", "BOOK_IMBALANCE", ("L2",), "NONE", "BOOK_VALID", "fraction"),
        ("microprice", "MICROPRICE", ("L1",), "NONE", "BOOK_VALID", "quote/base"),
        ("aggressive_buy_notional", "TRADE_FLOW", ("TRADE",), "BUCKET", "TRADE_FRESH", "quote"),
        ("aggressive_sell_notional", "TRADE_FLOW", ("TRADE",), "BUCKET", "TRADE_FRESH", "quote"),
        ("volume_delta", "TRADE_FLOW", ("TRADE",), "BUCKET", "TRADE_FRESH", "quote"),
        ("cvd", "CVD", ("TRADE",), "EXPANDING_PAST_ONLY", "TRADE_FRESH", "quote"),
        (
            "trade_intensity",
            "TRADE_INTENSITY",
            ("TRADE",),
            "BUCKET",
            "TRADE_FRESH",
            "trades/second",
        ),
        (
            "liquidity_shock",
            "LIQUIDITY_SHOCK",
            ("SPREAD", "DEPTH", "TRADE"),
            "TRAILING",
            "PIT_VALID",
            "boolean",
        ),
        (
            "venue_premium_bps",
            "CROSS_VENUE_PRICE",
            ("MULTI_VENUE_L1", "FX"),
            "ASOF",
            "CLOCK_OK_AND_FRESH",
            "bps",
        ),
        (
            "reference_price_eur",
            "REFERENCE_PRICE",
            ("MULTI_VENUE_L1", "FX"),
            "ASOF",
            "CLOCK_OK_AND_FRESH",
            "EUR/base",
        ),
        (
            "funding_rate",
            "DERIVATIVES_CONTEXT",
            ("MEXC_PERPETUAL",),
            "ASOF",
            "CONTEXT_FRESH",
            "fraction",
        ),
    )
    return tuple(
        FeatureDefinition(
            feature_name=name,
            group=group,
            version="v1",
            source=source,
            lookback=lookback,
            timestamp_semantics="AVAILABLE_AT_BUCKET_END_NO_CENTERING",
            required_quality_state=quality,
            units=units,
            missingness_semantics="NULL_PLUS_EXPLICIT_REASON_NEVER_ZERO_FILLED",
        )
        for name, group, source, lookback, quality, units in rows
    )


def build_trade_flow_buckets(
    trades: Sequence[TradeEvent],
    *,
    bucket_seconds: int = 60,
    trailing_normalization_buckets: int = 20,
) -> pd.DataFrame:
    if bucket_seconds not in ALLOWED_BUCKET_SECONDS:
        raise ValueError("unsupported trade-flow bucket resolution")
    if trailing_normalization_buckets < 2:
        raise ValueError("trade-flow normalization requires trailing history")
    unique, _ = deduplicate_trades(trades)
    if not unique:
        return pd.DataFrame()
    rows = []
    for trade in unique:
        available = trade.raw.timestamps.available_at
        rows.append(
            {
                "instrument_id": trade.raw.instrument_id,
                "venue": trade.raw.venue.casefold(),
                "available_at": available,
                "bucket": pd.Timestamp(available).floor(f"{bucket_seconds}s"),
                "price": float(trade.price),
                "quantity": float(trade.quantity),
                "notional": float(trade.quote_notional),
                "side": trade.aggressor_side,
                "semantics": trade.aggressor_semantics.value,
            }
        )
    frame = pd.DataFrame(rows).sort_values("available_at")
    output = []
    for (instrument, venue, bucket), selected in frame.groupby(
        ["instrument_id", "venue", "bucket"], sort=True
    ):
        buys = selected.loc[selected["side"] == "buy", "notional"]
        sells = selected.loc[selected["side"] == "sell", "notional"]
        known = selected["side"].isin(["buy", "sell"])
        total_notional = float(selected["notional"].sum())
        vwap = (
            float((selected["price"] * selected["quantity"]).sum() / selected["quantity"].sum())
            if float(selected["quantity"].sum()) > 0
            else None
        )
        large_threshold = float(selected["notional"].median()) if len(selected) else 0.0
        output.append(
            {
                "instrument_id": instrument,
                "venue": venue,
                "bucket_start": bucket,
                "available_at": pd.Timestamp(
                    bucket.to_pydatetime() + timedelta(seconds=bucket_seconds)
                ),
                "bucket_seconds": bucket_seconds,
                "aggressive_buy_notional": float(buys.sum()),
                "aggressive_sell_notional": float(sells.sum()),
                "volume_delta": float(buys.sum() - sells.sum()),
                "total_notional": total_notional,
                "trade_count": int(len(selected)),
                "known_aggressor_count": int(known.sum()),
                "unknown_aggressor_count": int((~known).sum()),
                "trade_count_imbalance": (
                    float(
                        ((selected["side"] == "buy").sum() - (selected["side"] == "sell").sum())
                        / known.sum()
                    )
                    if known.sum()
                    else None
                ),
                "trade_intensity": len(selected) / bucket_seconds,
                "large_trade_share": (
                    float(
                        selected.loc[selected["notional"] > large_threshold, "notional"].sum()
                        / total_notional
                    )
                    if total_notional > 0
                    else None
                ),
                "vwap": vwap,
                "missingness": Missingness.PRESENT.value,
            }
        )
    result = pd.DataFrame(output).sort_values(["instrument_id", "venue", "bucket_start"])
    result["cvd"] = result.groupby(["instrument_id", "venue"])["volume_delta"].cumsum()
    for column in ("total_notional", "trade_intensity"):
        grouped = result.groupby(["instrument_id", "venue"])[column]
        prior_median = grouped.transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    trailing_normalization_buckets,
                    min_periods=max(2, trailing_normalization_buckets // 4),
                )
                .median()
            )
        )
        result[f"{column}_trailing_ratio"] = result[column] / prior_median.replace(0.0, np.nan)
    result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    return result.reset_index(drop=True)


def detect_liquidity_shocks(
    frame: pd.DataFrame,
    *,
    trailing_buckets: int = 20,
    spread_multiple: float = 2.0,
    depth_fraction: float = 0.5,
    intensity_multiple: float = 2.0,
) -> pd.DataFrame:
    required = {"spread_bps", "depth_25bps_quote", "trade_intensity"}
    if not required <= set(frame.columns):
        raise ValueError(
            f"liquidity shock frame is missing {sorted(required - set(frame.columns))}"
        )
    if trailing_buckets < 2:
        raise ValueError("liquidity shock detection requires trailing history")
    output = frame.copy()
    medians = {
        column: output[column]
        .shift(1)
        .rolling(
            trailing_buckets,
            min_periods=max(2, trailing_buckets // 4),
        )
        .median()
        for column in required
    }
    output["spread_widening"] = output["spread_bps"] > medians["spread_bps"] * spread_multiple
    output["depth_withdrawal"] = (
        output["depth_25bps_quote"] < medians["depth_25bps_quote"] * depth_fraction
    )
    output["trade_intensity_spike"] = (
        output["trade_intensity"] > medians["trade_intensity"] * intensity_multiple
    )
    output["liquidity_shock"] = output[
        ["spread_widening", "depth_withdrawal", "trade_intensity_spike"]
    ].any(axis=1)
    output["liquidity_shock_definition"] = "CONTEMPORANEOUS_TRAILING_MEDIAN_V1"
    return output


def feature_redundancy_report(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    warning_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    selected = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    correlation = selected.corr()
    rows = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            value = correlation.loc[first, second]
            rows.append(
                {
                    "feature_a": first,
                    "feature_b": second,
                    "correlation": float(value) if pd.notna(value) else None,
                    "redundancy_warning": pd.notna(value)
                    and abs(float(value)) >= warning_threshold,
                }
            )
    return rows


def build_research_labels(
    quotes: pd.DataFrame,
    *,
    horizons_seconds: Sequence[int] = (300, 900, 1_800, 3_600, 14_400, 86_400),
    fee_bps_roundtrip: float = 50.0,
) -> pd.DataFrame:
    required = {"available_at", "best_bid", "best_ask", "instrument_id", "venue"}
    if not required <= set(quotes.columns):
        raise ValueError(f"label input missing {sorted(required - set(quotes.columns))}")
    if any(horizon <= 0 for horizon in horizons_seconds):
        raise ValueError("label horizons must be positive")
    frame = quotes.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame.sort_values(["instrument_id", "venue", "available_at"]).reset_index(drop=True)
    output = frame[["instrument_id", "venue", "available_at"]].copy()
    for horizon in horizons_seconds:
        future_rows: list[float | None] = [None] * len(frame)
        executable: list[float | None] = [None] * len(frame)
        mfe: list[float | None] = [None] * len(frame)
        mae: list[float | None] = [None] * len(frame)
        for _, positions in frame.groupby(["instrument_id", "venue"], sort=False).groups.items():
            indices = list(positions)
            group = frame.loc[indices]
            times = group["available_at"].astype("int64").to_numpy()
            bids = pd.to_numeric(group["best_bid"]).to_numpy(dtype=float)
            asks = pd.to_numeric(group["best_ask"]).to_numpy(dtype=float)
            mids = (bids + asks) / 2.0
            for local, global_index in enumerate(indices):
                target = times[local] + horizon * 1_000_000_000
                future_local = int(np.searchsorted(times, target, side="left"))
                if future_local >= len(indices):
                    continue
                future_rows[global_index] = mids[future_local] / mids[local] - 1.0
                executable[global_index] = (
                    bids[future_local] / asks[local] - 1.0 - fee_bps_roundtrip / 10_000.0
                )
                path = mids[local + 1 : future_local + 1]
                if len(path):
                    mfe[global_index] = float(np.max(path / asks[local] - 1.0))
                    mae[global_index] = float(np.min(path / asks[local] - 1.0))
        output[f"future_mid_return_{horizon}s"] = future_rows
        output[f"executable_return_{horizon}s"] = executable
        output[f"mfe_{horizon}s"] = mfe
        output[f"mae_{horizon}s"] = mae
    output["layer"] = DataLayer.RESEARCH_LABEL.value
    output["schema_version"] = LABEL_SCHEMA_VERSION
    output["feature_columns_present"] = False
    output["execution_authority"] = False
    return output


def assert_future_invariance(
    builder: Any,
    rows: Sequence[Any],
    *,
    cutoff: int,
) -> dict[str, Any]:
    if cutoff <= 0 or cutoff >= len(rows):
        raise ValueError("future-invariance cutoff must split the input")
    baseline = builder(rows[:cutoff])
    full = builder(rows)
    if isinstance(baseline, pd.DataFrame) and isinstance(full, pd.DataFrame):
        prior = full.iloc[: len(baseline)].reset_index(drop=True)
        left = baseline.reset_index(drop=True)
        safe = stable_hash(_json_safe(left.to_dict("records"))) == stable_hash(
            _json_safe(prior.to_dict("records"))
        )
    else:
        safe = stable_hash(_json_safe(baseline)) == stable_hash(_json_safe(full[: len(baseline)]))
    return {
        "status": "PASSED" if safe else "HARD_REJECT",
        "future_rows_added": len(rows) - cutoff,
        "prior_rows_unchanged": safe,
    }


class LayeredParquetStore:
    """Content-addressed, append-only partitions for RAW/NORMALIZED/FEATURE/LABEL."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def write_rows(
        self,
        *,
        layer: DataLayer,
        venue: str,
        canonical_asset_id: str,
        data_type: str,
        date: str,
        rows: Sequence[Mapping[str, Any]],
        schema_version: str,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot persist an empty partition")
        if layer is DataLayer.RAW and schema_version != RAW_EVENT_SCHEMA_VERSION:
            raise ValueError("RAW partitions require the raw event schema")
        if layer is DataLayer.FEATURE and any(
            any(
                str(key).startswith(("future_", "mfe_", "mae_", "executable_return_"))
                for key in row
            )
            for row in rows
        ):
            raise ValueError("future-derived labels cannot be stored in FEATURE")
        if layer is DataLayer.RAW and any(
            "label" in str(key).casefold() for row in rows for key in row
        ):
            raise ValueError("RAW partitions cannot contain research labels")
        canonical_rows = [_json_safe(dict(row)) for row in rows]
        content_hash = stable_hash(
            {
                "layer": layer.value,
                "schema": schema_version,
                "rows": canonical_rows,
            },
            length=64,
        )
        directory = (
            self.root
            / layer.value.casefold()
            / venue.casefold()
            / data_type.casefold()
            / canonical_asset_id.replace(":", "_")
            / date
        )
        target = directory / f"part-{content_hash}.parquet"
        if target.is_file():
            return {
                "path": str(target.resolve()),
                "content_hash": content_hash,
                "file_sha256": sha256_file(target),
                "row_count": len(canonical_rows),
                "reused": True,
                "layer": layer.value,
            }
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
        table = pa.Table.from_pylist(canonical_rows)
        pq.write_table(table, temporary, compression="zstd", write_statistics=True)
        # Windows does not permit ``fsync`` on a read-only descriptor.
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return {
            "path": str(target.resolve()),
            "content_hash": content_hash,
            "file_sha256": sha256_file(target),
            "row_count": len(canonical_rows),
            "reused": False,
            "layer": layer.value,
        }


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    venue: str
    canonical_asset_id: str
    data_type: str
    availability: HistoricalAvailability
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    event_count: int
    file_count: int
    gap_count: int | None
    coverage_percentage: float | None
    timestamp_basis: str
    quality_status: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def parquet_coverage(
    path: Path | str,
    *,
    venue: str,
    canonical_asset_id: str,
    data_type: str,
    availability: HistoricalAvailability,
    expected_interval: timedelta | None = None,
) -> CoverageRecord:
    root = Path(path)
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else ([root] if root.is_file() else [])
    if not files:
        return CoverageRecord(
            venue=venue,
            canonical_asset_id=canonical_asset_id,
            data_type=data_type,
            availability=HistoricalAvailability.NOT_AVAILABLE,
            start_timestamp=None,
            end_timestamp=None,
            event_count=0,
            file_count=0,
            gap_count=None,
            coverage_percentage=0.0,
            timestamp_basis="NONE",
            quality_status="NOT_EVALUABLE",
            reason_codes=("NO_LOCAL_PARTITIONS",),
        )
    dataset = pads.dataset([str(file) for file in files], format="parquet")
    names = set(dataset.schema.names)
    timestamp_column = "observed_at" if "observed_at" in names else "timestamp"
    # ``raw_hash`` identifies the immutable source response and is therefore
    # intentionally shared by every normalized row emitted from that response;
    # it is not a row-level duplicate key. Trade-event deduplication uses venue
    # trade identity instead (see ``TradeEvent.dedup_key``).
    table = dataset.to_table(columns=[timestamp_column])
    frame = table.to_pandas()
    timestamps = (
        pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").dropna().sort_values()
    )
    reasons: list[str] = []
    if timestamps.empty:
        reasons.append("NO_VALID_TIMESTAMPS")
    gap_count: int | None = None
    coverage: float | None = None
    if expected_interval is not None and len(timestamps) >= 2:
        deltas = timestamps.diff().dropna()
        gap_count = int((deltas > expected_interval * 1.5).sum())
        duration = max(
            expected_interval.total_seconds(),
            (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds(),
        )
        expected = max(1, int(duration / expected_interval.total_seconds()) + 1)
        coverage = min(100.0, len(timestamps) / expected * 100.0)
        if gap_count:
            reasons.append("MISSING_INTERVALS")
    return CoverageRecord(
        venue=venue.casefold(),
        canonical_asset_id=canonical_asset_id,
        data_type=data_type,
        availability=availability,
        start_timestamp=(timestamps.iloc[0].to_pydatetime() if len(timestamps) else None),
        end_timestamp=(timestamps.iloc[-1].to_pydatetime() if len(timestamps) else None),
        event_count=len(frame),
        file_count=len(files),
        gap_count=gap_count,
        coverage_percentage=coverage,
        timestamp_basis=timestamp_column.upper(),
        quality_status=("PARTIAL" if reasons else "OBSERVED_DATA_PRESENT"),
        reason_codes=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    minimum_days_research: float = 14.0
    minimum_days_robustness: float = 60.0
    minimum_valid_observations: int = 100_000
    maximum_gap_rate: float = 0.01
    minimum_venue_overlap: int = 2
    require_clock_ok_for_lead_lag: bool = True

    def __post_init__(self) -> None:
        if (
            self.minimum_days_research <= 0
            or self.minimum_days_robustness < self.minimum_days_research
            or self.minimum_valid_observations < 1
            or not 0 <= self.maximum_gap_rate < 1
            or self.minimum_venue_overlap < 1
        ):
            raise ValueError("research-readiness policy is invalid")


def assess_readiness(
    coverage: CoverageRecord,
    *,
    policy: ReadinessPolicy,
    venue_overlap: int,
    clock_quality: ClockQuality,
    requires_book: bool = False,
    book_valid_fraction: float | None = None,
) -> dict[str, Any]:
    if (
        coverage.event_count == 0
        or coverage.start_timestamp is None
        or coverage.end_timestamp is None
    ):
        state = ReadinessState.NOT_STARTED
        reasons = ["NO_OBSERVATIONS"]
        days = 0.0
    else:
        days = max(
            0.0, (coverage.end_timestamp - coverage.start_timestamp).total_seconds() / 86_400.0
        )
        gap_rate = (
            float(coverage.gap_count or 0) / max(1, coverage.event_count)
            if coverage.gap_count is not None
            else None
        )
        reasons = []
        if days < policy.minimum_days_research:
            reasons.append("INSUFFICIENT_DAYS")
        if coverage.event_count < policy.minimum_valid_observations:
            reasons.append("INSUFFICIENT_OBSERVATIONS")
        if gap_rate is not None and gap_rate > policy.maximum_gap_rate:
            reasons.append("EXCESSIVE_GAP_RATE")
        if venue_overlap < policy.minimum_venue_overlap:
            reasons.append("INSUFFICIENT_VENUE_OVERLAP")
        if policy.require_clock_ok_for_lead_lag and clock_quality is not ClockQuality.CLOCK_OK:
            reasons.append("CLOCK_NOT_OK")
        if requires_book and (book_valid_fraction is None or book_valid_fraction < 0.99):
            reasons.append("BOOK_VALIDITY_INSUFFICIENT")
        if any(
            code in reasons
            for code in ("EXCESSIVE_GAP_RATE", "CLOCK_NOT_OK", "BOOK_VALIDITY_INSUFFICIENT")
        ):
            state = ReadinessState.QUALITY_FAILED
        elif reasons:
            state = ReadinessState.PARTIAL
        elif days >= policy.minimum_days_robustness:
            state = ReadinessState.ROBUSTNESS_USABLE
        else:
            state = ReadinessState.RESEARCH_USABLE
    return {
        "state": state.value,
        "coverage_days": days,
        "event_count": coverage.event_count,
        "venue_overlap": venue_overlap,
        "clock_quality": clock_quality.value,
        "requires_book": requires_book,
        "book_valid_fraction": book_valid_fraction,
        "reason_codes": reasons,
        "policy": _json_safe(asdict(policy)),
        "execution_authority": False,
    }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    schema_version: str
    source_config_version: str
    sources: tuple[Mapping[str, Any], ...]
    venues: tuple[str, ...]
    assets: tuple[str, ...]
    start: datetime | None
    end: datetime | None
    raw_event_hashes: tuple[str, ...]
    normalized_partitions: tuple[Mapping[str, Any], ...]
    feature_version: str
    clock_quality: Mapping[str, Any]
    coverage: tuple[Mapping[str, Any], ...]
    missingness: Mapping[str, Any]
    gaps: Mapping[str, Any]
    rejected_rows: Mapping[str, int]
    build_commit: str | None
    build_timestamp: datetime
    collection_start_timestamp: datetime | None
    layer_separation: Mapping[str, str]
    replay_hash: str
    authority: str = "RESEARCH_DATA_ONLY"

    @classmethod
    def build(
        cls,
        *,
        sources: Sequence[Mapping[str, Any]],
        venues: Sequence[str],
        assets: Sequence[str],
        coverage: Sequence[CoverageRecord],
        raw_event_hashes: Sequence[str],
        normalized_partitions: Sequence[Mapping[str, Any]],
        clock_quality: Mapping[str, Any],
        missingness: Mapping[str, Any],
        gaps: Mapping[str, Any],
        rejected_rows: Mapping[str, int],
        build_commit: str | None,
        collection_start_timestamp: datetime | None,
        replay_hash: str,
    ) -> DatasetManifest:
        starts = [row.start_timestamp for row in coverage if row.start_timestamp]
        ends = [row.end_timestamp for row in coverage if row.end_timestamp]
        identity = {
            "schema": DATASET_MANIFEST_SCHEMA_VERSION,
            "source_config": SOURCE_CONFIG_VERSION,
            "sources": _json_safe(sources),
            "venues": sorted(set(venues)),
            "assets": sorted(set(assets)),
            "coverage": [row.to_dict() for row in coverage],
            "raw_hashes": sorted(set(raw_event_hashes)),
            "partitions": _json_safe(normalized_partitions),
            "feature_version": FEATURE_SCHEMA_VERSION,
            "replay_hash": replay_hash,
        }
        return cls(
            dataset_id=stable_hash(identity, length=64),
            schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
            source_config_version=SOURCE_CONFIG_VERSION,
            sources=tuple(dict(row) for row in sources),
            venues=tuple(sorted(set(venue.casefold() for venue in venues))),
            assets=tuple(sorted(set(assets))),
            start=min(starts) if starts else None,
            end=max(ends) if ends else None,
            raw_event_hashes=tuple(sorted(set(raw_event_hashes))),
            normalized_partitions=tuple(dict(row) for row in normalized_partitions),
            feature_version=FEATURE_SCHEMA_VERSION,
            clock_quality=dict(clock_quality),
            coverage=tuple(row.to_dict() for row in coverage),
            missingness=dict(missingness),
            gaps=dict(gaps),
            rejected_rows={str(key): int(value) for key, value in rejected_rows.items()},
            build_commit=build_commit,
            build_timestamp=datetime.now(tz=UTC),
            collection_start_timestamp=collection_start_timestamp,
            layer_separation={
                "RAW": "IMMUTABLE_OBSERVED_EVENTS_NO_LABELS",
                "NORMALIZED": "IDENTITY_TIMESTAMP_AND_UNIT_NORMALIZATION",
                "FEATURE": "POINT_IN_TIME_ONLY",
                "RESEARCH_LABEL": "FUTURE_DERIVED_OFFLINE_ONLY",
            },
            replay_hash=replay_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def deterministic_replay_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (_json_safe(dict(row)) for row in rows),
        key=lambda row: (
            str(row.get("available_at") or row.get("local_receive_timestamp") or ""),
            str(row.get("event_id") or row.get("feature_hash") or ""),
        ),
    )
    return stable_hash({"schema": MARKET_STRUCTURE_SCHEMA_VERSION, "rows": ordered}, length=64)


def market_data_health(
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    research = {}
    for name in (
        "BITVAVO_TRADES",
        "BITVAVO_BOOK",
        "REFERENCE_TRADES",
        "REFERENCE_BOOK",
        "CLOCK_HEALTH",
        "CROSS_VENUE_ALIGNMENT",
        "STORAGE_WRITER",
        "FEATURE_PIPELINE",
    ):
        row = dict(components.get(name) or {})
        research[name] = {
            "status": row.get("status", "NOT_STARTED"),
            "last_update": row.get("last_update"),
            "freshness_seconds": row.get("freshness_seconds"),
            "gap_count": int(row.get("gap_count") or 0),
            "error_count": int(row.get("error_count") or 0),
        }
    return {
        "schema_version": "market_structure_health_v1",
        "execution_health": "UNCHANGED_SEPARATE_CANONICAL_SYSTEM",
        "research_data_health": research,
        "optional_reference_failure_stops_execution": False,
        "orders_generated": 0,
    }


def benchmark_market_structure_pipeline(
    trades: Sequence[TradeEvent],
    *,
    repetitions: int = 3,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    timings = []
    row_counts = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = build_trade_flow_buckets(trades, bucket_seconds=60)
        timings.append(time.perf_counter() - started)
        row_counts.append(len(result))
    median = statistics.median(timings)
    input_count = len(trades)
    serialized_bytes = len(stable_json([row.to_dict() for row in trades]).encode("utf-8"))
    return {
        "input_events": input_count,
        "output_feature_rows": max(row_counts, default=0),
        "median_elapsed_seconds": median,
        "events_per_second": input_count / median if median else None,
        "estimated_raw_bytes_per_event": serialized_bytes / input_count if input_count else None,
        "estimated_storage_growth_per_day_at_100_events_second_gb": (
            serialized_bytes / input_count * 100 * 86_400 / 1024**3 if input_count else None
        ),
        "benchmark_scope": "DETERMINISTIC_SYNTHETIC_CPU_ONLY",
    }


def reference_repository_provenance(project_root: Path | str) -> list[dict[str, Any]]:
    root = Path(project_root) / "crypto-references"
    rows = [
        ("nautilus_trader", "LGPL-3.0", "EVENT_INIT_TIMESTAMPS_AND_BOOK_REPLAY_CONCEPTS"),
        ("freqtrade", "GPL-3.0", "FUTURE_PERTURBATION_BIAS_CHECK_CONCEPT"),
        ("lean", "Apache-2.0", "INSTRUMENT_AND_QUOTE_IDENTITY_CONCEPT"),
        ("qlib", "MIT", "VERSIONED_PIT_DATASET_IDENTITY_CONCEPT"),
        ("vectorbt", "COMMONS_CLAUSE", "DEFERRED_LATER_SCREENING_NO_CODE_USED"),
        ("pybroker", "COMMONS_CLAUSE", "DEFERRED_LATER_VALIDATION_NO_CODE_USED"),
    ]
    output = []
    for name, license_name, concept in rows:
        repository = root / name
        license_candidates = sorted(repository.glob("LICENSE*"))
        output.append(
            {
                "repository": name,
                "path": str(repository.resolve()),
                "present": repository.is_dir(),
                "license": license_name,
                "license_sha256": (
                    sha256_file(license_candidates[0]) if license_candidates else None
                ),
                "concept_used": concept,
                "source_code_copied": False,
                "read_only": True,
            }
        )
    return output


def stage0_exact_divergence_evidence(alpha_artifact: Mapping[str, Any]) -> dict[str, Any]:
    family = "CROSS_SECTIONAL_MOMENTUM"
    family_rows = {
        str(row.get("family")): row
        for row in ((alpha_artifact.get("stage0") or {}).get("family_results") or [])
    }
    stage0 = (family_rows.get(family) or {}).get("best_result") or {}
    exact = (alpha_artifact.get("exact_results") or {}).get(family) or {}
    exact_metrics = (exact.get("final_test") or {}).get("metrics") or {}
    return {
        "family": family,
        "stage0_profit_factor": stage0.get("profit_factor"),
        "stage0_net_expectancy_eur_per_100_eur_episode": stage0.get("net_expectancy_eur"),
        "exact_profit_factor": exact_metrics.get("profit_factor"),
        "exact_net_expectancy_r": exact_metrics.get("net_expectancy_r"),
        "units_are_not_directly_comparable": True,
        "proven_semantic_differences": [
            "STAGE0_FIXED_100_EUR_EPISODE_ACCOUNTING_VS_NATIVE_RISK_SIZING",
            "STAGE0_TARGET_WEIGHT_EPISODES_VS_NATIVE_PORTFOLIO_CAPITAL_AND_CANDIDATE_COMPETITION",
            "NATIVE_ADAPTER_ATR_STOP_TARGET_TRAILING_AND_MAX_HOLD",
            "NATIVE_NEXT_ELIGIBLE_EXECUTION_AND_ORDER_CONSTRAINTS",
            "NATIVE_POSITION_OVERLAP_AND_CAPITAL_SEQUENCING",
        ],
        "market_structure_calibration_dimensions": [
            "ENTRY_REACHABLE_ASK",
            "EXIT_REACHABLE_BID",
            "OBSERVED_SPREAD",
            "DEPTH_CONSTRAINED_SLIPPAGE",
            "SIGNAL_TO_RECEIVE_DELAY",
            "BOOK_OR_REFERENCE_UNAVAILABLE",
        ],
        "root_cause_fully_attributed": False,
        "status": "RECORDED_FOR_FUTURE_STAGE0_CALIBRATION_NO_SIMULATOR_MUTATION",
    }


def source_inventory() -> list[dict[str, Any]]:
    return [
        {
            "source": "bitvavo",
            "classification": VenueRole.EXECUTION_PRIMARY.value,
            "supported": [
                "REST_TRADES",
                "REST_L2_SNAPSHOT",
                "WS_TRADES",
                "WS_L2_DELTA",
                "WS_TICKER",
            ],
            "activation": "EXISTING_PRIMARY_PUBLIC_COLLECTION",
            "execution_authority": "SOLE_EXISTING_EXECUTION_VENUE_UNCHANGED",
        },
        {
            "source": "kraken",
            "classification": VenueRole.SPOT_REFERENCE.value,
            "supported": ["HISTORICAL_OHLCV", "WS_TRADES", "WS_L2", "WS_TICKER"],
            "activation": "ADAPTER_PRESENT_EVENT_COLLECTION_NOT_STARTED",
            "execution_authority": "ZERO",
        },
        {
            "source": "mexc",
            "classification": VenueRole.DERIVATIVES_CONTEXT.value,
            "supported": ["HISTORICAL_SPOT_OHLCV", "WS_SPOT_TRADES", "WS_SPOT_L2", "FUNDING_OI"],
            "activation": "DERIVATIVES_CONTEXT_COLLECTED_SPOT_EVENTS_NOT_STARTED",
            "execution_authority": "ZERO",
        },
        {
            "source": "deribit",
            "classification": VenueRole.DERIVATIVES_CONTEXT.value,
            "supported": ["OPTIONS_PUBLIC_CONTEXT_ADAPTER"],
            "activation": "NO_LOCAL_RAW_PARTITIONS",
            "execution_authority": "ZERO",
        },
    ]


__all__ = [
    "AggressorSemantics",
    "BookEvent",
    "BookFeatureSnapshot",
    "BookQuality",
    "CanonicalInstrument",
    "ClockAssessment",
    "ClockMonitor",
    "ClockQuality",
    "CoverageRecord",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DataLayer",
    "DatasetManifest",
    "EventTimestamps",
    "EventType",
    "FEATURE_SCHEMA_VERSION",
    "FXObservation",
    "FeatureDefinition",
    "HistoricalAvailability",
    "InstrumentRegistry",
    "LABEL_SCHEMA_VERSION",
    "LayeredParquetStore",
    "MARKET_STRUCTURE_SCHEMA_VERSION",
    "MarketType",
    "Missingness",
    "OrderBookReplayer",
    "PointInTimeFXBook",
    "RAW_EVENT_SCHEMA_VERSION",
    "RawMarketEvent",
    "ReadinessPolicy",
    "ReadinessState",
    "TimestampQuality",
    "TradeEvent",
    "VenueQuote",
    "VenueRole",
    "align_cross_venue_quotes",
    "assert_future_invariance",
    "assess_readiness",
    "benchmark_market_structure_pipeline",
    "build_research_labels",
    "build_trade_flow_buckets",
    "deduplicate_trades",
    "detect_liquidity_shocks",
    "deterministic_replay_hash",
    "feature_redundancy_report",
    "infer_aggressor_side",
    "initial_instrument_registry",
    "market_data_health",
    "market_structure_feature_schema",
    "normalize_quote_to_eur",
    "parquet_coverage",
    "reference_repository_provenance",
    "source_inventory",
    "stage0_exact_divergence_evidence",
]
