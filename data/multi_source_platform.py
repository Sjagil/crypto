"""Point-in-time multi-source research-data contracts for P1.2.1.

The module deliberately contains no broker, order, capital, position, or
strategy-promotion surface.  Every source except Bitvavo is structurally
research-only, and even Bitvavo observations cannot route orders from here.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
import zlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.market_structure import BookQuality, ClockQuality
from utils.common import (
    atomic_write_json,
    parse_utc,
    read_json,
    sha256_file,
    stable_hash,
    stable_json,
    utc_iso,
    utc_now,
)

MULTI_SOURCE_SCHEMA_VERSION = "multi_source_pit_platform_v1"
OBSERVATION_SCHEMA_VERSION = "source_neutral_observation_v1"
KRAKEN_COLLECTOR_VERSION = "kraken_spot_public_v2_checksum_v1"
KRAKEN_BOOK_SCHEMA_VERSION = "kraken_l2_crc32_book_v1"
FEATURE_STORE_SCHEMA_VERSION = "multi_source_pit_feature_store_v1"
FREEZE_SCHEMA_VERSION = "multi_source_dataset_freeze_v1"
READINESS_SCHEMA_VERSION = "multi_source_family_readiness_v1"
API_BUDGET_SCHEMA_VERSION = "paid_source_budget_ledger_v1"
EVENT_SCHEMA_VERSION = "governed_public_event_v1"
PRIMARY_ASSETS = ("BTC", "ETH", "SOL")
QUOTE_ASSET_IDENTITIES = {
    "EUR": "FIAT:EUR",
    "USD": "FIAT:USD",
    "USDT": "CRYPTO:USDT",
    "USDC": "CRYPTO:USDC",
}


class SourceAuthority(StrEnum):
    EXECUTION_TRUTH = "EXECUTION_TRUTH"
    PRIMARY_MARKET_STRUCTURE = "PRIMARY_MARKET_STRUCTURE"
    PRIMARY_SPOT_MARKET = "PRIMARY_SPOT_MARKET"
    REFERENCE_SPOT_MARKET = "REFERENCE_SPOT_MARKET"
    REFERENCE_MARKET_STRUCTURE = "REFERENCE_MARKET_STRUCTURE"
    DERIVATIVES_CONTEXT = "DERIVATIVES_CONTEXT"
    MARKET_METADATA = "MARKET_METADATA"
    MARKET_CAP = "MARKET_CAP"
    DOMINANCE = "DOMINANCE"
    MARKET_BREADTH = "MARKET_BREADTH"
    UNIVERSE_CONTEXT = "UNIVERSE_CONTEXT"
    HISTORICAL_VALIDATION = "HISTORICAL_VALIDATION"
    SECONDARY_OHLCV = "SECONDARY_OHLCV"
    CROSS_SOURCE_SANITY_CHECK = "CROSS_SOURCE_SANITY_CHECK"
    EVENT_INTELLIGENCE = "EVENT_INTELLIGENCE"
    MACRO_CONTEXT = "MACRO_CONTEXT"


class TimestampResolution(StrEnum):
    EVENT_EXACT = "EVENT_EXACT"
    EVENT_SECOND = "EVENT_SECOND"
    BAR_BOUNDARY = "BAR_BOUNDARY"
    PROVIDER_SNAPSHOT = "PROVIDER_SNAPSHOT"
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"


class SourceQuality(StrEnum):
    PRIMARY_OFFICIAL = "PRIMARY_OFFICIAL"
    HIGH_QUALITY_STRUCTURED = "HIGH_QUALITY_STRUCTURED"
    SECONDARY_REPUTABLE = "SECONDARY_REPUTABLE"
    UNVERIFIED = "UNVERIFIED"


class DataClassification(StrEnum):
    TRUE_HISTORICAL_SOURCE = "TRUE_HISTORICAL_SOURCE"
    PROSPECTIVE_COLLECTION = "PROSPECTIVE_COLLECTION"
    RECONSTRUCTED = "RECONSTRUCTED"
    DERIVED = "DERIVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class DataFamily(StrEnum):
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    CROSS_VENUE = "CROSS_VENUE"
    MARKET_BREADTH = "MARKET_BREADTH"
    UNIVERSE = "UNIVERSE"
    DERIVATIVES_CONTEXT = "DERIVATIVES_CONTEXT"
    MACRO_CONTEXT = "MACRO_CONTEXT"
    EVENT_INTELLIGENCE = "EVENT_INTELLIGENCE"
    HISTORICAL_VALIDATION = "HISTORICAL_VALIDATION"


class FamilyReadiness(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    COLLECTING = "COLLECTING"
    PARTIAL = "PARTIAL"
    EXPLORATORY_USABLE = "EXPLORATORY_USABLE"
    RESEARCH_USABLE = "RESEARCH_USABLE"
    ROBUSTNESS_USABLE = "ROBUSTNESS_USABLE"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    QUALITY_FAILED = "QUALITY_FAILED"


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    source: str
    authorities: tuple[SourceAuthority, ...]
    execution_allowed: bool
    public_data_only: bool
    may_influence: tuple[str, ...]
    forbidden_influences: tuple[str, ...] = (
        "ORDER_SUBMISSION",
        "ORDER_CANCELLATION",
        "POSITION_AUTHORITY",
        "CAPITAL_RESERVATION",
        "PORTFOLIO_AUTHORITY",
        "LIVE_STRATEGY_AUTHORITY",
        "RISK_LIMITS",
    )

    def __post_init__(self) -> None:
        source = self.source.casefold()
        if self.execution_allowed and source != "bitvavo":
            raise ValueError("only Bitvavo may retain existing execution authority")
        if source != "bitvavo" and SourceAuthority.EXECUTION_TRUTH in self.authorities:
            raise ValueError("reference providers cannot be execution truth")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "authorities": [value.value for value in self.authorities],
        }


def source_authority_registry() -> dict[str, SourcePolicy]:
    policies = (
        SourcePolicy(
            "bitvavo",
            (
                SourceAuthority.EXECUTION_TRUTH,
                SourceAuthority.PRIMARY_MARKET_STRUCTURE,
                SourceAuthority.PRIMARY_SPOT_MARKET,
            ),
            True,
            False,
            ("EXISTING_GATED_EXECUTION_ECONOMICS", "PRIMARY_SPOT_FACTS", "TCA"),
            forbidden_influences=(),
        ),
        SourcePolicy(
            "kraken",
            (
                SourceAuthority.REFERENCE_SPOT_MARKET,
                SourceAuthority.REFERENCE_MARKET_STRUCTURE,
            ),
            False,
            True,
            ("REFERENCE_PRICE", "REFERENCE_FLOW", "REFERENCE_L2", "QUALITY_ORACLE"),
        ),
        SourcePolicy(
            "mexc_spot",
            (
                SourceAuthority.REFERENCE_SPOT_MARKET,
                SourceAuthority.REFERENCE_MARKET_STRUCTURE,
            ),
            False,
            True,
            ("REFERENCE_PRICE", "REFERENCE_FLOW", "REFERENCE_L2"),
        ),
        SourcePolicy(
            "mexc_derivatives",
            (SourceAuthority.DERIVATIVES_CONTEXT,),
            False,
            True,
            ("FUNDING_CONTEXT", "OPEN_INTEREST_CONTEXT", "BASIS_CONTEXT"),
        ),
        SourcePolicy(
            "coinmarketcap",
            (
                SourceAuthority.MARKET_METADATA,
                SourceAuthority.MARKET_CAP,
                SourceAuthority.DOMINANCE,
                SourceAuthority.MARKET_BREADTH,
                SourceAuthority.UNIVERSE_CONTEXT,
            ),
            False,
            True,
            ("BREADTH", "RANK", "SUPPLY", "MARKET_CAP", "DOMINANCE"),
        ),
        SourcePolicy(
            "eodhd",
            (
                SourceAuthority.HISTORICAL_VALIDATION,
                SourceAuthority.SECONDARY_OHLCV,
                SourceAuthority.CROSS_SOURCE_SANITY_CHECK,
                SourceAuthority.MACRO_CONTEXT,
            ),
            False,
            True,
            ("OHLCV_VALIDATION", "MACRO_CONTEXT", "PROVIDER_DISAGREEMENT"),
        ),
        SourcePolicy(
            "scrapers",
            (SourceAuthority.EVENT_INTELLIGENCE,),
            False,
            True,
            ("TIMESTAMPED_PUBLIC_EVENTS", "EXCHANGE_ANNOUNCEMENTS", "INCIDENT_CONTEXT"),
        ),
    )
    return {row.source: row for row in policies}


@dataclass(frozen=True, slots=True)
class CanonicalAssetIdentity:
    canonical_asset_id: str
    symbol: str
    name: str
    cmc_id: int | None
    provider_identifiers: Mapping[str, str]
    contract_network: str | None = None
    contract_address: str | None = None

    def __post_init__(self) -> None:
        if not self.canonical_asset_id.startswith("CRYPTO:"):
            raise ValueError("canonical asset IDs must use CRYPTO namespace")
        if not self.symbol or not self.name:
            raise ValueError("asset symbol and name are required")
        if not self.provider_identifiers:
            raise ValueError("provider identifiers cannot be empty")

    @property
    def identity_hash(self) -> str:
        return stable_hash(asdict(self), length=64)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "identity_hash": self.identity_hash}


class CanonicalAssetRegistry:
    def __init__(self, identities: Sequence[CanonicalAssetIdentity]) -> None:
        self._by_id: dict[str, CanonicalAssetIdentity] = {}
        self._by_provider_id: dict[tuple[str, str], CanonicalAssetIdentity] = {}
        self._by_cmc_id: dict[int, CanonicalAssetIdentity] = {}
        for identity in identities:
            if identity.canonical_asset_id in self._by_id:
                raise ValueError("duplicate canonical asset ID")
            self._by_id[identity.canonical_asset_id] = identity
            if identity.cmc_id is not None:
                if identity.cmc_id in self._by_cmc_id:
                    raise ValueError("duplicate CoinMarketCap ID")
                self._by_cmc_id[identity.cmc_id] = identity
            for provider, provider_id in identity.provider_identifiers.items():
                key = (provider.casefold(), str(provider_id).upper())
                if key in self._by_provider_id:
                    raise ValueError(f"provider identity collision: {key}")
                self._by_provider_id[key] = identity

    def resolve(self, provider: str, provider_id: str) -> CanonicalAssetIdentity:
        key = (provider.casefold(), str(provider_id).upper())
        if key not in self._by_provider_id:
            raise KeyError(f"unmapped provider asset identity: {key}")
        return self._by_provider_id[key]

    def resolve_cmc(self, cmc_id: int) -> CanonicalAssetIdentity:
        return self._by_cmc_id[cmc_id]

    @property
    def identities(self) -> tuple[CanonicalAssetIdentity, ...]:
        return tuple(self._by_id.values())

    @property
    def registry_hash(self) -> str:
        return stable_hash([row.to_dict() for row in self.identities], length=64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "canonical_multi_source_asset_registry_v1",
            "registry_hash": self.registry_hash,
            "assets": [row.to_dict() for row in self.identities],
        }


def initial_multi_source_asset_registry() -> CanonicalAssetRegistry:
    rows = (
        CanonicalAssetIdentity(
            "CRYPTO:BTC",
            "BTC",
            "Bitcoin",
            1,
            {
                "bitvavo": "BTC-EUR",
                "kraken": "BTC/EUR",
                "mexc_spot": "BTCUSDT",
                "coinmarketcap": "1",
                "eodhd": "BTC-USD.CC",
                "scrapers": "bitcoin",
            },
        ),
        CanonicalAssetIdentity(
            "CRYPTO:ETH",
            "ETH",
            "Ethereum",
            1027,
            {
                "bitvavo": "ETH-EUR",
                "kraken": "ETH/EUR",
                "mexc_spot": "ETHUSDT",
                "coinmarketcap": "1027",
                "eodhd": "ETH-USD.CC",
                "scrapers": "ethereum",
            },
        ),
        CanonicalAssetIdentity(
            "CRYPTO:SOL",
            "SOL",
            "Solana",
            5426,
            {
                "bitvavo": "SOL-EUR",
                "kraken": "SOL/EUR",
                "mexc_spot": "SOLUSDT",
                "coinmarketcap": "5426",
                "eodhd": "SOL-USD.CC",
                "scrapers": "solana",
            },
        ),
        CanonicalAssetIdentity(
            "CRYPTO:LINK",
            "LINK",
            "Chainlink",
            1975,
            {
                "bitvavo": "LINK-EUR",
                "kraken": "LINK/EUR",
                "mexc_spot": "LINKUSDT",
                "coinmarketcap": "1975",
                "eodhd": "LINK-USD.CC",
                "scrapers": "chainlink",
            },
        ),
    )
    return CanonicalAssetRegistry(rows)


def normalize_quote_asset(provider: str, quote_id: str) -> dict[str, Any]:
    """Normalize quote identity without pretending stablecoins are fiat."""

    normalized = str(quote_id).strip().upper()
    if normalized not in QUOTE_ASSET_IDENTITIES:
        raise KeyError(f"unmapped quote asset for {provider.casefold()}: {normalized}")
    canonical_id = QUOTE_ASSET_IDENTITIES[normalized]
    return {
        "provider": provider.casefold(),
        "provider_quote_id": normalized,
        "canonical_quote_asset_id": canonical_id,
        "is_stablecoin": canonical_id.startswith("CRYPTO:USD"),
        "fiat_equivalence_assumed": False,
        "fx_conversion_required": canonical_id != "FIAT:EUR",
    }


def mexc_semantic_source(market_kind: str) -> str:
    selected = market_kind.strip().casefold()
    if selected == "spot":
        return "mexc_spot"
    if selected in {"derivative", "derivatives", "futures", "perpetual"}:
        return "mexc_derivatives"
    raise ValueError("MEXC market kind must explicitly be spot or derivatives")


def _utc(value: datetime | str | None, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise ValueError("timestamp is required")
        return None
    selected = parse_utc(value) if isinstance(value, str) else value
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return selected.astimezone(UTC)


def _decimal(value: Any, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid decimal value") from exc
    if not selected.is_finite():
        raise ValueError("decimal value must be finite")
    if positive and selected <= 0:
        raise ValueError("decimal value must be positive")
    if nonnegative and selected < 0:
        raise ValueError("decimal value must be nonnegative")
    return selected


@dataclass(frozen=True, slots=True)
class SourceNeutralObservation:
    source: str
    source_type: str
    canonical_asset_id: str | None
    data_type: str
    local_receive_timestamp: datetime
    normalized_timestamp: datetime
    persisted_timestamp: datetime
    raw_payload: Any
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    venue: str | None = None
    venue_instrument_id: str | None = None
    exchange_event_timestamp: datetime | None = None
    provider_timestamp: datetime | None = None
    timestamp_resolution: TimestampResolution = TimestampResolution.RETRIEVAL_ONLY
    quality_state: str = "PRESENT"
    freshness_seconds: float = 0.0
    source_event_id: str | None = None
    classification: DataClassification = DataClassification.PROSPECTIVE_COLLECTION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        receive = _utc(self.local_receive_timestamp, required=True)
        normalized = _utc(self.normalized_timestamp, required=True)
        persisted = _utc(self.persisted_timestamp, required=True)
        event = _utc(self.exchange_event_timestamp)
        provider = _utc(self.provider_timestamp)
        if persisted < receive:
            raise ValueError("persisted timestamp cannot precede receive timestamp")
        if event is None and provider is None and normalized != receive:
            raise ValueError("retrieval-only observations normalize to receive time")
        if self.freshness_seconds < 0 or not math.isfinite(self.freshness_seconds):
            raise ValueError("freshness must be a nonnegative finite number")
        if not self.source or not self.data_type:
            raise ValueError("source and data type are required")

    @property
    def known_at(self) -> datetime:
        return self.local_receive_timestamp.astimezone(UTC)

    @property
    def raw_payload_hash(self) -> str:
        return stable_hash(self.raw_payload, length=64)

    @property
    def observation_id(self) -> str:
        identity: Any = self.source_event_id
        if identity is None:
            identity = {
                "source": self.source.casefold(),
                "instrument": self.venue_instrument_id,
                "asset": self.canonical_asset_id,
                "data_type": self.data_type,
                "event_time": self.exchange_event_timestamp or self.provider_timestamp,
                "known_at": self.known_at,
                "payload": self.raw_payload_hash,
            }
        return stable_hash(
            [self.schema_version, self.source.casefold(), self.data_type, identity],
            length=64,
        )

    def to_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        output = {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "source": self.source.casefold(),
            "source_type": self.source_type,
            "venue": self.venue.casefold() if self.venue else None,
            "canonical_asset_id": self.canonical_asset_id,
            "venue_instrument_id": self.venue_instrument_id,
            "data_type": self.data_type,
            "exchange_event_timestamp": utc_iso(self.exchange_event_timestamp)
            if self.exchange_event_timestamp
            else None,
            "provider_timestamp": utc_iso(self.provider_timestamp)
            if self.provider_timestamp
            else None,
            "local_receive_timestamp": utc_iso(self.local_receive_timestamp),
            "normalized_timestamp": utc_iso(self.normalized_timestamp),
            "persisted_timestamp": utc_iso(self.persisted_timestamp),
            "known_at": utc_iso(self.known_at),
            "raw_payload_hash": self.raw_payload_hash,
            "timestamp_resolution": self.timestamp_resolution.value,
            "quality_state": self.quality_state,
            "freshness_seconds": self.freshness_seconds,
            "source_event_id": self.source_event_id,
            "classification": self.classification.value,
            "metadata": dict(self.metadata),
        }
        if include_raw:
            output["raw_payload"] = self.raw_payload
        return output


class ImmutableSourceLedger:
    """Crash-recoverable append-only raw ledger with a per-source hash chain."""

    def __init__(self, root: Path | str, source: str, checkpoint_path: Path | str) -> None:
        self.root = Path(root).resolve()
        self.source = source.casefold()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(read_json(self.checkpoint_path)) if self.checkpoint_path.is_file() else {}
        self.record_count = int(checkpoint.get("record_count") or 0)
        self.root_hash = str(checkpoint.get("root_hash") or "0" * 64)
        self.first_known_at = checkpoint.get("first_known_at")
        self.last_known_at = checkpoint.get("last_known_at")
        self._recent_ids: deque[str] = deque(
            (str(value) for value in checkpoint.get("recent_observation_ids") or []),
            maxlen=20_000,
        )
        self._recent_set = set(self._recent_ids)
        recovery = self.recover_tail()
        self._reconcile_checkpoint(force=bool(recovery.get("repaired_bytes")))

    def _segment(self, known_at: datetime) -> Path:
        selected = known_at.astimezone(UTC)
        return (
            self.root
            / f"schema={OBSERVATION_SCHEMA_VERSION}"
            / f"date={selected:%Y-%m-%d}"
            / f"event_hour={selected:%H}"
            / "events.jsonl"
        )

    def recover_tail(self) -> dict[str, Any]:
        repaired_bytes = 0
        files = sorted(self.root.rglob("events.jsonl"))
        if not files:
            return {"status": "EMPTY", "repaired_bytes": 0}
        latest = files[-1]
        raw = latest.read_bytes()
        last_newline = raw.rfind(b"\n")
        if raw and last_newline != len(raw) - 1:
            durable = raw[: last_newline + 1] if last_newline >= 0 else b""
            repaired_bytes = len(raw) - len(durable)
            with latest.open("wb") as stream:
                stream.write(durable)
                stream.flush()
                os.fsync(stream.fileno())
        return {
            "status": "TAIL_REPAIRED" if repaired_bytes else "CLEAN",
            "path": str(latest),
            "repaired_bytes": repaired_bytes,
        }

    def _checkpoint_payload(self, target: Path) -> dict[str, Any]:
        return {
            "schema_version": "immutable_source_ledger_checkpoint_v1",
            "source": self.source,
            "record_count": self.record_count,
            "root_hash": self.root_hash,
            "first_known_at": self.first_known_at,
            "last_known_at": self.last_known_at,
            "last_segment": str(target.resolve()),
            "last_segment_size_bytes": target.stat().st_size,
            # A bounded exact-replay window protects the hot path without
            # turning every checkpoint into a multi-megabyte rewrite.
            "recent_observation_ids": list(self._recent_ids)[-2_000:],
            "orders_generated": 0,
            "private_exchange_requests": 0,
        }

    def _write_checkpoint(self, target: Path) -> None:
        atomic_write_json(self.checkpoint_path, self._checkpoint_payload(target))

    def _reconcile_checkpoint(self, *, force: bool = False) -> None:
        files = sorted(self.root.rglob("events.jsonl"))
        if not files:
            return
        checkpoint = dict(read_json(self.checkpoint_path)) if self.checkpoint_path.is_file() else {}
        latest = files[-1].resolve()
        checkpoint_target = Path(str(checkpoint.get("last_segment") or ""))
        checkpoint_matches_disk = (
            checkpoint_target.is_file()
            and checkpoint_target.resolve() == latest
            and int(checkpoint.get("last_segment_size_bytes") or -1) == latest.stat().st_size
        )
        if checkpoint_matches_disk and not force:
            return

        previous = "0" * 64
        count = 0
        first: str | None = None
        last: str | None = None
        recent: deque[str] = deque(maxlen=20_000)
        for path in files:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = dict(json.loads(line))
                    claimed = str(record.pop("record_hash", ""))
                    if record.get("previous_record_hash") != previous:
                        raise RuntimeError("immutable source ledger chain break during recovery")
                    expected = stable_hash(record, length=64)
                    if not claimed or claimed != expected:
                        raise RuntimeError("immutable source ledger hash failure during recovery")
                    previous = claimed
                    count += 1
                    known = str(record.get("known_at") or "") or None
                    first = first or known
                    last = known or last
                    observation_id = str(record.get("observation_id") or "")
                    if observation_id:
                        recent.append(observation_id)
        self.record_count = count
        self.root_hash = previous
        self.first_known_at = first
        self.last_known_at = last
        self._recent_ids = recent
        self._recent_set = set(recent)
        self._write_checkpoint(latest)

    def append(self, observation: SourceNeutralObservation) -> dict[str, Any]:
        if observation.source.casefold() != self.source:
            raise ValueError("observation source does not match ledger")
        if observation.observation_id in self._recent_set:
            return {
                "status": "DUPLICATE_REJECTED",
                "observation_id": observation.observation_id,
                "record_count": self.record_count,
                "root_hash": self.root_hash,
            }
        body = observation.to_dict(include_raw=True)
        body.update(
            {
                "previous_record_hash": self.root_hash,
                "ledger_source": self.source,
                "orders_generated": 0,
                "private_exchange_requests": 0,
            }
        )
        record_hash = stable_hash(body, length=64)
        record = {**body, "record_hash": record_hash}
        target = self._segment(observation.known_at)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(stable_json(record))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.record_count += 1
        self.root_hash = record_hash
        known = utc_iso(observation.known_at)
        self.first_known_at = self.first_known_at or known
        self.last_known_at = known
        self._recent_ids.append(observation.observation_id)
        self._recent_set.add(observation.observation_id)
        while len(self._recent_set) > self._recent_ids.maxlen:
            self._recent_set = set(self._recent_ids)
        self._write_checkpoint(target)
        return {
            "status": "APPENDED",
            "path": str(target),
            "observation_id": observation.observation_id,
            "record_hash": record_hash,
            "record_count": self.record_count,
            "root_hash": self.root_hash,
        }

    def append_many(
        self,
        observations: Sequence[SourceNeutralObservation],
    ) -> dict[str, Any]:
        """Append a durable batch while preserving the exact event hash chain."""

        accepted: list[tuple[Path, dict[str, Any], str]] = []
        duplicate_count = 0
        batch_ids: set[str] = set()
        next_root = self.root_hash
        ordered_observations = sorted(
            observations,
            key=lambda row: (row.known_at, row.observation_id),
        )
        for observation in ordered_observations:
            if observation.source.casefold() != self.source:
                raise ValueError("observation source does not match ledger")
            observation_id = observation.observation_id
            if observation_id in self._recent_set or observation_id in batch_ids:
                duplicate_count += 1
                continue
            body = observation.to_dict(include_raw=True)
            body.update(
                {
                    "previous_record_hash": next_root,
                    "ledger_source": self.source,
                    "orders_generated": 0,
                    "private_exchange_requests": 0,
                }
            )
            record_hash = stable_hash(body, length=64)
            accepted.append(
                (
                    self._segment(observation.known_at),
                    {**body, "record_hash": record_hash},
                    observation_id,
                )
            )
            next_root = record_hash
            batch_ids.add(observation_id)
        if not accepted:
            return {
                "status": "DUPLICATES_ONLY",
                "appended": 0,
                "duplicates": duplicate_count,
                "record_count": self.record_count,
                "root_hash": self.root_hash,
            }

        groups: dict[Path, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        batch_storage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"events": 0, "raw_bytes": 0}
        )
        for target, record, observation_id in accepted:
            groups[target].append((record, observation_id))
        for target, records in groups.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as stream:
                for record, _ in records:
                    serialized = stable_json(record)
                    stream.write(serialized)
                    stream.write("\n")
                    key = "|".join(
                        (
                            self.source,
                            str(record.get("canonical_asset_id") or "GLOBAL_OR_UNMAPPED"),
                            str(record.get("data_type") or "UNKNOWN"),
                        )
                    )
                    batch_storage[key]["events"] += 1
                    batch_storage[key]["raw_bytes"] += len(
                        f"{serialized}\n".encode("utf-8")
                    )
                stream.flush()
                os.fsync(stream.fileno())

        for _, record, observation_id in accepted:
            known = str(record["known_at"])
            self.record_count += 1
            self.root_hash = str(record["record_hash"])
            self.first_known_at = self.first_known_at or known
            self.last_known_at = known
            self._recent_ids.append(observation_id)
            self._recent_set.add(observation_id)
        if len(self._recent_set) > self._recent_ids.maxlen:
            self._recent_set = set(self._recent_ids)
        last_target = accepted[-1][0]
        self._write_checkpoint(last_target)
        return {
            "status": "APPENDED",
            "appended": len(accepted),
            "duplicates": duplicate_count,
            "record_count": self.record_count,
            "root_hash": self.root_hash,
            "last_segment": str(last_target.resolve()),
            "storage_by_source_asset_type": dict(batch_storage),
        }

    def checkpoint(self) -> dict[str, Any]:
        return (
            dict(read_json(self.checkpoint_path))
            if self.checkpoint_path.is_file()
            else {"source": self.source, "record_count": 0, "root_hash": "0" * 64}
        )


def verify_source_ledger(root: Path | str, source: str) -> dict[str, Any]:
    previous = "0" * 64
    count = 0
    failures: list[str] = []
    first = None
    last = None
    files = sorted(Path(root).rglob("events.jsonl"))
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = dict(json.loads(line))
                except (TypeError, ValueError):
                    failures.append(f"INVALID_JSON:{path}:{line_number}")
                    continue
                claimed = str(record.pop("record_hash", ""))
                if record.get("ledger_source") != source.casefold():
                    failures.append(f"SOURCE_MISMATCH:{path}:{line_number}")
                if record.get("previous_record_hash") != previous:
                    failures.append(f"CHAIN_BREAK:{path}:{line_number}")
                expected = stable_hash(record, length=64)
                if claimed != expected:
                    failures.append(f"HASH_MISMATCH:{path}:{line_number}")
                previous = claimed or expected
                known = record.get("known_at")
                first = first or known
                last = known or last
                count += 1
    return {
        "schema_version": "immutable_source_ledger_audit_v1",
        "source": source.casefold(),
        "status": "PASSED" if not failures else "FAILED",
        "record_count": count,
        "segment_count": len(files),
        "root_hash": previous,
        "first_known_at": first,
        "last_known_at": last,
        "integrity_failures": failures,
        "orders_generated": 0,
    }


def compact_source_ledger(
    root: Path | str,
    destination: Path | str,
    source: str,
    *,
    closed_before: datetime | None = None,
) -> dict[str, Any]:
    """Create immutable Zstd-Parquet replicas bound to raw segment hashes."""

    raw_root = Path(root)
    compact_root = Path(destination) / source.casefold()
    segments: list[dict[str, Any]] = []
    for raw_path in sorted(raw_root.rglob("events.jsonl")):
        if closed_before is not None:
            parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in raw_path.parts if "=" in part}
            try:
                segment_start = datetime.fromisoformat(
                    f"{parts['date']}T{parts['event_hour']}:00:00+00:00"
                )
            except (KeyError, ValueError):
                continue
            if segment_start + timedelta(hours=1) > closed_before.astimezone(UTC):
                continue
        raw_sha = sha256_file(raw_path)
        relative = raw_path.relative_to(raw_root)
        target_dir = compact_root.joinpath(*relative.parts[:-1])
        target = target_dir / f"events-{raw_sha[:24]}.parquet"
        manifest_path = target.with_suffix(".manifest.json")
        if target.is_file() and manifest_path.is_file():
            segments.append(dict(read_json(manifest_path)))
            continue
        rows: list[dict[str, Any]] = []
        with raw_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = dict(json.loads(line))
                rows.append(
                    {
                        "source": source.casefold(),
                        "canonical_asset_id": record.get("canonical_asset_id"),
                        "data_type": record.get("data_type"),
                        "known_at": record.get("known_at"),
                        "observation_id": record.get("observation_id"),
                        "record_hash": record.get("record_hash"),
                        "raw_line_number": line_number,
                        "record_json": stable_json(record),
                    }
                )
        if not rows:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        temporary = target_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        parquet_rows = pq.read_metadata(target).num_rows
        if parquet_rows != len(rows):
            raise RuntimeError("Parquet compaction row-count verification failed")
        manifest = {
            "schema_version": "immutable_source_parquet_compaction_v1",
            "source": source.casefold(),
            "raw_segment": str(raw_path.resolve()),
            "raw_sha256": raw_sha,
            "raw_bytes": raw_path.stat().st_size,
            "row_count": len(rows),
            "parquet_path": str(target.resolve()),
            "parquet_sha256": sha256_file(target),
            "parquet_bytes": target.stat().st_size,
            "compression": "ZSTD",
            "compression_ratio": target.stat().st_size / raw_path.stat().st_size,
            "raw_deleted": False,
            "orders_generated": 0,
        }
        atomic_write_json(manifest_path, manifest)
        segments.append(manifest)
    return {
        "schema_version": "immutable_source_compaction_audit_v1",
        "source": source.casefold(),
        "status": "PASSED" if segments else "NOT_EVALUABLE",
        "segment_count": len(segments),
        "row_count": sum(int(row["row_count"]) for row in segments),
        "raw_bytes": sum(int(row["raw_bytes"]) for row in segments),
        "parquet_bytes": sum(int(row["parquet_bytes"]) for row in segments),
        "compression_ratio": (
            sum(int(row["parquet_bytes"]) for row in segments)
            / sum(int(row["raw_bytes"]) for row in segments)
            if segments
            else None
        ),
        "raw_immutable_and_preserved": True,
        "segments": segments,
    }


def _kraken_checksum_text(value: Decimal) -> str:
    normalized = format(value, "f").replace(".", "").lstrip("0")
    return normalized or "0"


class KrakenL2Book:
    """Kraken spot WebSocket v2 L2 state using the venue CRC32 contract."""

    def __init__(self, symbol: str, *, depth: int = 100, stale_seconds: float = 30.0) -> None:
        if depth not in {10, 25, 100, 500, 1000}:
            raise ValueError("unsupported Kraken book depth")
        self.symbol = symbol
        self.depth = depth
        self.stale_seconds = stale_seconds
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.state = BookQuality.BOOK_UNINITIALIZED
        self.last_timestamp: datetime | None = None
        self.last_receive_timestamp: datetime | None = None
        self.last_checksum: int | None = None
        self.checksum_failures = 0
        self.out_of_order = 0
        self.duplicates = 0
        self.reconnects = 0
        self.applied_message_ids: deque[str] = deque(maxlen=20_000)
        self._applied_set: set[str] = set()

    @staticmethod
    def _levels(values: Iterable[Any]) -> list[tuple[Decimal, Decimal]]:
        output: list[tuple[Decimal, Decimal]] = []
        for row in values:
            if isinstance(row, Mapping):
                price, quantity = row.get("price"), row.get("qty")
            elif isinstance(row, (tuple, list)) and len(row) >= 2:
                price, quantity = row[0], row[1]
            else:
                raise ValueError("invalid Kraken price level")
            output.append((_decimal(price, positive=True), _decimal(quantity, nonnegative=True)))
        return output

    @staticmethod
    def _apply(side: dict[Decimal, Decimal], levels: Sequence[tuple[Decimal, Decimal]]) -> None:
        for price, quantity in levels:
            if quantity == 0:
                side.pop(price, None)
            else:
                side[price] = quantity

    def checksum(self) -> int:
        asks = sorted(self.asks.items())[:10]
        bids = sorted(self.bids.items(), reverse=True)[:10]
        text = "".join(
            f"{_kraken_checksum_text(price)}{_kraken_checksum_text(quantity)}"
            for price, quantity in (*asks, *bids)
        )
        return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF

    def disconnect(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.state = BookQuality.BOOK_SYNCING
        self.last_timestamp = None
        self.last_receive_timestamp = None
        self.last_checksum = None
        self.reconnects += 1

    def _fail_closed(self, state: BookQuality) -> None:
        self.bids.clear()
        self.asks.clear()
        self.state = state

    def apply(
        self,
        *,
        kind: str,
        bids: Iterable[Any],
        asks: Iterable[Any],
        checksum: int,
        event_timestamp: datetime,
        receive_timestamp: datetime,
        message_id: str,
    ) -> dict[str, Any]:
        if message_id in self._applied_set:
            self.duplicates += 1
            return {"status": "DUPLICATE_REJECTED", "state": self.state.value}
        event_at = _utc(event_timestamp, required=True)
        receive_at = _utc(receive_timestamp, required=True)
        if self.last_timestamp is not None and event_at < self.last_timestamp:
            self.out_of_order += 1
            self._fail_closed(BookQuality.BOOK_GAPPED)
            return {"status": "OUT_OF_ORDER_FAIL_CLOSED", "state": self.state.value}
        bid_levels = self._levels(bids)
        ask_levels = self._levels(asks)
        if kind == "snapshot":
            next_bids: dict[Decimal, Decimal] = {}
            next_asks: dict[Decimal, Decimal] = {}
            self._apply(next_bids, bid_levels)
            self._apply(next_asks, ask_levels)
            self.bids, self.asks = next_bids, next_asks
        elif kind == "update":
            if self.state is not BookQuality.BOOK_VALID:
                return {"status": "SNAPSHOT_REQUIRED", "state": self.state.value}
            self._apply(self.bids, bid_levels)
            self._apply(self.asks, ask_levels)
        else:
            raise ValueError("Kraken book kind must be snapshot or update")
        self.bids = dict(sorted(self.bids.items(), reverse=True)[: self.depth])
        self.asks = dict(sorted(self.asks.items())[: self.depth])
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            self._fail_closed(BookQuality.BOOK_INVALID)
            return {"status": "INVALID_OR_CROSSED_FAIL_CLOSED", "state": self.state.value}
        actual = self.checksum()
        if actual != int(checksum):
            self.checksum_failures += 1
            self._fail_closed(BookQuality.BOOK_GAPPED)
            return {
                "status": "CHECKSUM_FAILURE_FAIL_CLOSED",
                "expected_checksum": int(checksum),
                "actual_checksum": actual,
                "state": self.state.value,
            }
        self.state = BookQuality.BOOK_VALID
        self.last_timestamp = event_at
        self.last_receive_timestamp = receive_at
        self.last_checksum = actual
        self.applied_message_ids.append(message_id)
        self._applied_set.add(message_id)
        if len(self._applied_set) > self.applied_message_ids.maxlen:
            self._applied_set = set(self.applied_message_ids)
        return {"status": "APPLIED", "state": self.state.value, "checksum": actual}

    def quality_at(self, at: datetime) -> BookQuality:
        selected = _utc(at, required=True)
        if (
            self.state is BookQuality.BOOK_VALID
            and self.last_receive_timestamp is not None
            and (selected - self.last_receive_timestamp).total_seconds() > self.stale_seconds
        ):
            return BookQuality.BOOK_STALE
        return self.state

    def snapshot(self, at: datetime | None = None) -> dict[str, Any]:
        selected = at or self.last_receive_timestamp or datetime(1970, 1, 1, tzinfo=UTC)
        quality = self.quality_at(selected)
        if quality is not BookQuality.BOOK_VALID:
            return {
                "schema_version": KRAKEN_BOOK_SCHEMA_VERSION,
                "symbol": self.symbol,
                "quality": quality.value,
                "best_bid": None,
                "best_ask": None,
                "spread_bps": None,
                "checksum": self.last_checksum,
            }
        best_bid, best_ask = max(self.bids), min(self.asks)
        mid = (best_bid + best_ask) / Decimal("2")
        bid_qty, ask_qty = self.bids[best_bid], self.asks[best_ask]
        total = bid_qty + ask_qty
        return {
            "schema_version": KRAKEN_BOOK_SCHEMA_VERSION,
            "symbol": self.symbol,
            "quality": quality.value,
            "best_bid": str(best_bid),
            "best_ask": str(best_ask),
            "mid": str(mid),
            "spread_bps": str((best_ask - best_bid) / mid * Decimal("10000")),
            "microprice": str((best_ask * bid_qty + best_bid * ask_qty) / total if total else mid),
            "top_imbalance": str((bid_qty - ask_qty) / total if total else Decimal("0")),
            "checksum": self.last_checksum,
            "event_timestamp": utc_iso(self.last_timestamp) if self.last_timestamp else None,
            "known_at": utc_iso(self.last_receive_timestamp)
            if self.last_receive_timestamp
            else None,
        }


@dataclass(slots=True)
class TradeCoverage:
    source: str
    canonical_asset_id: str
    trade_count: int = 0
    valid_trade_count: int = 0
    duplicate_rejected: int = 0
    out_of_order_count: int = 0
    gap_indicators: int = 0
    total_notional: Decimal = Decimal("0")
    first_known_at: datetime | None = None
    last_known_at: datetime | None = None
    last_event_at: datetime | None = None
    latencies_ms: list[float] = field(default_factory=list)
    aggressor_known: int = 0
    aggressor_inferred: int = 0
    aggressor_unknown: int = 0

    def observe(
        self,
        *,
        known_at: datetime,
        event_at: datetime,
        notional: Decimal,
        aggressor_semantics: str,
        duplicate: bool = False,
    ) -> None:
        if duplicate:
            self.duplicate_rejected += 1
            return
        if self.last_event_at is not None and event_at < self.last_event_at:
            self.out_of_order_count += 1
        self.trade_count += 1
        self.valid_trade_count += 1
        self.total_notional += notional
        self.first_known_at = self.first_known_at or known_at
        self.last_known_at = max(self.last_known_at or known_at, known_at)
        self.last_event_at = max(self.last_event_at or event_at, event_at)
        self.latencies_ms.append((known_at - event_at).total_seconds() * 1000)
        if "INFERRED" in aggressor_semantics:
            self.aggressor_inferred += 1
        elif "EXCHANGE" in aggressor_semantics:
            self.aggressor_known += 1
        else:
            self.aggressor_unknown += 1

    def to_dict(self, now: datetime | None = None) -> dict[str, Any]:
        selected = now or utc_now()
        active = (
            (self.last_known_at - self.first_known_at).total_seconds()
            if self.first_known_at and self.last_known_at
            else 0.0
        )
        total = max(1, self.valid_trade_count)
        latencies = self.latencies_ms[-100_000:]
        return {
            "source": self.source,
            "canonical_asset_id": self.canonical_asset_id,
            "trade_count": self.trade_count,
            "valid_trade_count": self.valid_trade_count,
            "duplicate_rejected": self.duplicate_rejected,
            "out_of_order_count": self.out_of_order_count,
            "gap_indicators": self.gap_indicators,
            "notional": str(self.total_notional),
            "active_duration_seconds": active,
            "first_known_at": utc_iso(self.first_known_at) if self.first_known_at else None,
            "last_known_at": utc_iso(self.last_known_at) if self.last_known_at else None,
            "freshness_seconds": (
                (selected - self.last_known_at).total_seconds() if self.last_known_at else None
            ),
            "latency_ms_p50": statistics.median(latencies) if latencies else None,
            "latency_ms_p95": (
                sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
                if latencies
                else None
            ),
            "aggressor_known_share": self.aggressor_known / total,
            "aggressor_inferred_share": self.aggressor_inferred / total,
            "aggressor_unknown_share": self.aggressor_unknown / total,
        }


@dataclass(slots=True)
class BookCoverage:
    source: str
    canonical_asset_id: str
    collection_started_at: datetime
    last_transition_at: datetime
    current_state: BookQuality = BookQuality.BOOK_UNINITIALIZED
    durations: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def transition(self, state: BookQuality, at: datetime, reason: str | None = None) -> None:
        selected = _utc(at, required=True)
        elapsed = max(0.0, (selected - self.last_transition_at).total_seconds())
        self.durations[self.current_state.value] += elapsed
        self.current_state = state
        self.last_transition_at = selected
        if reason:
            self.failures[reason] += 1

    def to_dict(self, at: datetime | None = None) -> dict[str, Any]:
        selected = at or utc_now()
        durations = dict(self.durations)
        durations[self.current_state.value] = durations.get(self.current_state.value, 0.0) + max(
            0.0, (selected - self.last_transition_at).total_seconds()
        )
        total = max(0.0, (selected - self.collection_started_at).total_seconds())
        valid = durations.get(BookQuality.BOOK_VALID.value, 0.0)
        return {
            "source": self.source,
            "canonical_asset_id": self.canonical_asset_id,
            "collection_started_at": utc_iso(self.collection_started_at),
            "total_collection_seconds": total,
            "book_valid_seconds": valid,
            "book_invalid_seconds": durations.get(BookQuality.BOOK_INVALID.value, 0.0),
            "book_stale_seconds": durations.get(BookQuality.BOOK_STALE.value, 0.0),
            "book_gap_seconds": durations.get(BookQuality.BOOK_GAPPED.value, 0.0),
            "book_syncing_seconds": durations.get(BookQuality.BOOK_SYNCING.value, 0.0),
            "book_valid_fraction": valid / total if total else 0.0,
            "current_state": self.current_state.value,
            "failure_counts": dict(self.failures),
        }


def interval_overlap(
    first_start: datetime | None,
    first_end: datetime | None,
    second_start: datetime | None,
    second_end: datetime | None,
) -> dict[str, Any]:
    if not all((first_start, first_end, second_start, second_end)):
        return {
            "start": None,
            "end": None,
            "seconds": 0.0,
            "minutes": 0.0,
            "hours": 0.0,
            "days": 0.0,
            "status": "NO_OVERLAP_EVIDENCE",
        }
    start = max(first_start, second_start)
    end = min(first_end, second_end)
    seconds = max(0.0, (end - start).total_seconds())
    return {
        "start": utc_iso(start) if seconds > 0 else None,
        "end": utc_iso(end) if seconds > 0 else None,
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
        "days": seconds / 86400,
        "status": "OVERLAP_PRESENT" if seconds > 0 else "NO_OVERLAP_EVIDENCE",
    }


@dataclass(frozen=True, slots=True)
class ReadinessThreshold:
    family: str
    research_minimum_days: float
    robustness_minimum_days: float
    minimum_valid_observations: int
    minimum_valid_overlap_days: float
    maximum_gap_rate: float
    minimum_book_valid_fraction: float | None
    maximum_freshness_seconds: float
    minimum_assets: int
    clock_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_readiness_thresholds() -> dict[str, ReadinessThreshold]:
    rows = (
        ReadinessThreshold("TRADE_FLOW", 14, 60, 100_000, 0, 0.01, None, 10, 3, False),
        ReadinessThreshold("L2", 14, 60, 10_000, 0, 0.01, 0.99, 5, 3, False),
        ReadinessThreshold("CROSS_VENUE_PRICE", 14, 60, 100_000, 14, 0.01, 0.95, 5, 3, True),
        ReadinessThreshold("LEAD_LAG", 30, 90, 500_000, 30, 0.005, 0.99, 2, 3, True),
        ReadinessThreshold("CMC_BREADTH", 30, 180, 720, 0, 0.02, None, 7200, 30, False),
        ReadinessThreshold("CMC_UNIVERSE", 30, 180, 30, 0, 0.02, None, 90000, 100, False),
        ReadinessThreshold("EODHD_HISTORICAL", 365, 1825, 365, 0, 0.01, None, 172800, 3, False),
        ReadinessThreshold("EVENT_DATA", 30, 180, 50, 0, 0.10, None, 172800, 1, False),
        ReadinessThreshold("DERIVATIVES_CONTEXT", 30, 180, 720, 0, 0.05, None, 7200, 3, False),
    )
    return {row.family: row for row in rows}


def assess_family_readiness(
    threshold: ReadinessThreshold,
    *,
    duration_days: float,
    valid_observations: int,
    overlap_days: float,
    gap_rate: float | None,
    book_valid_fraction: float | None,
    freshness_seconds: float | None,
    asset_count: int,
    clock_quality: ClockQuality,
) -> dict[str, Any]:
    reasons: list[str] = []
    if valid_observations <= 0:
        state = FamilyReadiness.NOT_STARTED
        reasons.append("NO_VALID_OBSERVATIONS")
    else:
        if duration_days < threshold.research_minimum_days:
            reasons.append("INSUFFICIENT_WALL_CLOCK_HISTORY")
        if valid_observations < threshold.minimum_valid_observations:
            reasons.append("INSUFFICIENT_VALID_OBSERVATIONS")
        if overlap_days < threshold.minimum_valid_overlap_days:
            reasons.append("INSUFFICIENT_VALID_OVERLAP")
        if gap_rate is None or gap_rate > threshold.maximum_gap_rate:
            reasons.append("GAP_RATE_NOT_PROVEN_ACCEPTABLE")
        if threshold.minimum_book_valid_fraction is not None and (
            book_valid_fraction is None
            or book_valid_fraction < threshold.minimum_book_valid_fraction
        ):
            reasons.append("BOOK_VALID_FRACTION_INSUFFICIENT")
        if freshness_seconds is None or freshness_seconds > threshold.maximum_freshness_seconds:
            reasons.append("FRESHNESS_INSUFFICIENT")
        if asset_count < threshold.minimum_assets:
            reasons.append("MARKET_DIVERSITY_INSUFFICIENT")
        if threshold.clock_required and clock_quality is not ClockQuality.CLOCK_OK:
            reasons.append("CLOCK_NOT_OK")
        quality_failure = any(
            code in reasons
            for code in (
                "GAP_RATE_NOT_PROVEN_ACCEPTABLE",
                "BOOK_VALID_FRACTION_INSUFFICIENT",
                "CLOCK_NOT_OK",
            )
        )
        if not reasons:
            state = (
                FamilyReadiness.ROBUSTNESS_USABLE
                if duration_days >= threshold.robustness_minimum_days
                else FamilyReadiness.RESEARCH_USABLE
            )
        elif quality_failure:
            state = FamilyReadiness.QUALITY_FAILED
        else:
            state = FamilyReadiness.PARTIAL
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "family": threshold.family,
        "state": state.value,
        "duration_days": duration_days,
        "valid_observations": valid_observations,
        "valid_overlap_days": overlap_days,
        "gap_rate": gap_rate,
        "book_valid_fraction": book_valid_fraction,
        "freshness_seconds": freshness_seconds,
        "asset_count": asset_count,
        "clock_quality": clock_quality.value,
        "reason_codes": reasons,
        "threshold": threshold.to_dict(),
        "automatic_alpha_start": False,
        "execution_authority": False,
    }


@dataclass(frozen=True, slots=True)
class ApiBudgetRule:
    provider: str
    daily_credit_limit: int
    monthly_credit_limit: int
    minimum_interval_seconds: int
    endpoint_costs: Mapping[str, int]


class ApiBudgetLedger:
    def __init__(self, path: Path | str, rules: Sequence[ApiBudgetRule]) -> None:
        self.path = Path(path)
        self.rules = {rule.provider.casefold(): rule for rule in rules}
        self._state = (
            dict(read_json(self.path))
            if self.path.is_file()
            else {
                "schema_version": API_BUDGET_SCHEMA_VERSION,
                "providers": {},
                "orders_generated": 0,
            }
        )

    def _provider(self, provider: str) -> dict[str, Any]:
        state = self._state.setdefault("providers", {}).setdefault(
            provider.casefold(),
            {
                "requests": 0,
                "credits": 0,
                "cache_hits": 0,
                "duplicate_requests_avoided": 0,
                "by_day": {},
                "by_endpoint": {},
                "last_request_at": {},
                "rate_limit_events": 0,
                "quota_exhaustion_events": 0,
                "failed_requests": 0,
            },
        )
        state.setdefault("failed_requests", 0)
        return state

    def authorize(
        self,
        provider: str,
        endpoint: str,
        *,
        units: int = 1,
        at: datetime | None = None,
    ) -> dict[str, Any]:
        if units < 1:
            raise ValueError("API authorization units must be positive")
        selected = at or utc_now()
        name = provider.casefold()
        rule = self.rules[name]
        state = self._provider(name)
        unit_cost = int(rule.endpoint_costs.get(endpoint, 1))
        cost = unit_cost * units
        day = selected.astimezone(UTC).date().isoformat()
        day_credits = int((state["by_day"].get(day) or {}).get("credits") or 0)
        month = day[:7]
        month_credits = sum(
            int(row.get("credits") or 0)
            for key, row in state["by_day"].items()
            if key.startswith(month)
        )
        last = state["last_request_at"].get(endpoint)
        if last and (selected - parse_utc(last)).total_seconds() < rule.minimum_interval_seconds:
            state["duplicate_requests_avoided"] += 1
            self._persist()
            return {
                "allowed": False,
                "reason": "CACHE_OR_INTERVAL_REQUIRED",
                "cost": 0,
                "unit_cost": unit_cost,
            }
        if day_credits + cost > rule.daily_credit_limit:
            state["quota_exhaustion_events"] += 1
            self._persist()
            return {
                "allowed": False,
                "reason": "DAILY_CREDIT_LIMIT",
                "cost": cost,
                "unit_cost": unit_cost,
            }
        if month_credits + cost > rule.monthly_credit_limit:
            state["quota_exhaustion_events"] += 1
            self._persist()
            return {
                "allowed": False,
                "reason": "MONTHLY_CREDIT_LIMIT",
                "cost": cost,
                "unit_cost": unit_cost,
            }
        return {
            "allowed": True,
            "reason": "BUDGET_AVAILABLE",
            "cost": cost,
            "unit_cost": unit_cost,
        }

    def record_request(
        self,
        provider: str,
        endpoint: str,
        *,
        credits: int,
        requests: int = 1,
        at: datetime | None = None,
    ) -> None:
        if requests < 1:
            raise ValueError("recorded API requests must be positive")
        selected = at or utc_now()
        state = self._provider(provider)
        day = selected.astimezone(UTC).date().isoformat()
        daily = state["by_day"].setdefault(day, {"requests": 0, "credits": 0})
        endpoint_state = state["by_endpoint"].setdefault(endpoint, {"requests": 0, "credits": 0})
        for target in (state, daily, endpoint_state):
            target["requests"] = int(target.get("requests") or 0) + int(requests)
            target["credits"] = int(target.get("credits") or 0) + int(credits)
        state["last_request_at"][endpoint] = utc_iso(selected)
        self._persist()

    def record_cache_hit(self, provider: str) -> None:
        self._provider(provider)["cache_hits"] += 1
        self._persist()

    def record_rate_limit(self, provider: str) -> None:
        self._provider(provider)["rate_limit_events"] += 1
        self._persist()

    def record_failure(self, provider: str) -> None:
        self._provider(provider)["failed_requests"] += 1
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(self.path, self._state)

    def status(self) -> dict[str, Any]:
        payload = json.loads(stable_json(self._state))
        for provider, rule in self.rules.items():
            state = payload.setdefault("providers", {}).setdefault(provider, {})
            state["daily_credit_limit"] = rule.daily_credit_limit
            state["monthly_credit_limit"] = rule.monthly_credit_limit
            state["minimum_interval_seconds"] = rule.minimum_interval_seconds
        return payload


def default_api_budget_rules() -> tuple[ApiBudgetRule, ...]:
    return (
        ApiBudgetRule(
            "coinmarketcap",
            daily_credit_limit=500,
            monthly_credit_limit=15_000,
            minimum_interval_seconds=900,
            endpoint_costs={"global_metrics": 1, "rankings": 1, "metadata": 1},
        ),
        ApiBudgetRule(
            "eodhd",
            daily_credit_limit=1_000,
            monthly_credit_limit=30_000,
            minimum_interval_seconds=21_600,
            endpoint_costs={"eod": 1, "macro": 1, "economic_events": 1},
        ),
    )


def compute_cmc_breadth(
    records: Sequence[Mapping[str, Any]],
    *,
    known_at: datetime,
    global_context: Mapping[str, Any] | None = None,
    previous_breadth: Mapping[str, Any] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    rows = []
    for record in records:
        rank = record.get("cmc_rank")
        market_cap = record.get("market_cap")
        return_24h = record.get("percent_change_24h")
        return_7d = record.get("percent_change_7d")
        volume = record.get("volume_24h")
        if rank is None:
            continue
        rows.append(
            {
                "rank": int(rank),
                "symbol": str(record.get("symbol") or "").upper(),
                "market_cap": float(market_cap) if market_cap is not None else None,
                "return_24h": float(return_24h) if return_24h is not None else None,
                "return_7d": float(return_7d) if return_7d is not None else None,
                "volume_24h": float(volume) if volume is not None else None,
                "market_cap_dominance": (
                    float(record["market_cap_dominance"]) / 100
                    if record.get("market_cap_dominance") is not None
                    and float(record["market_cap_dominance"]) > 1
                    else (
                        float(record["market_cap_dominance"])
                        if record.get("market_cap_dominance") is not None
                        else None
                    )
                ),
            }
        )
    positive_24 = [row["return_24h"] for row in rows if row["return_24h"] is not None]
    positive_7d = [row["return_7d"] for row in rows if row["return_7d"] is not None]
    weights = [row for row in rows if row["market_cap"] and row["return_24h"] is not None]
    weighted = (
        sum(row["market_cap"] * row["return_24h"] for row in weights)
        / sum(row["market_cap"] for row in weights)
        if weights
        else None
    )
    caps = sorted((row["market_cap"] for row in rows if row["market_cap"]), reverse=True)
    total_cap = sum(caps)
    eligible_returns = [row for row in rows if row["return_24h"] is not None]
    selected_top = [row for row in eligible_returns if row["rank"] <= top_n]
    stable_symbols = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE"}
    altcoins = [
        row
        for row in eligible_returns
        if row["symbol"] not in {"BTC", "ETH", *stable_symbols}
    ]
    volume_rows = [
        row
        for row in eligible_returns
        if row["volume_24h"] is not None and row["volume_24h"] >= 0
    ]
    volume_total = sum(row["volume_24h"] for row in volume_rows)
    global_values = dict(global_context or {})

    def dominance(symbol: str, global_key: str) -> float | None:
        global_value = global_values.get(global_key)
        if global_value is not None:
            selected = float(global_value)
            return selected / 100 if selected > 1 else selected
        return next(
            (
                row["market_cap_dominance"]
                for row in rows
                if row["symbol"] == symbol and row["market_cap_dominance"] is not None
            ),
            None,
        )

    btc_dominance = dominance("BTC", "btc_dominance")
    eth_dominance = dominance("ETH", "eth_dominance")
    prior_btc = (previous_breadth or {}).get("btc_dominance")
    return {
        "schema_version": "cmc_point_in_time_breadth_v2",
        "known_at": utc_iso(known_at),
        "eligible_asset_count": len(rows),
        "positive_24h_fraction": (
            sum(value > 0 for value in positive_24) / len(positive_24) if positive_24 else None
        ),
        "positive_7d_fraction": (
            sum(value > 0 for value in positive_7d) / len(positive_7d) if positive_7d else None
        ),
        "median_return_24h": statistics.median(positive_24) if positive_24 else None,
        "cross_sectional_dispersion_24h": (
            statistics.pstdev(positive_24) if len(positive_24) >= 2 else None
        ),
        "market_cap_weighted_return_24h": weighted,
        "equal_weight_market_return_24h": (
            statistics.fmean(positive_24) if positive_24 else None
        ),
        "top_n": top_n,
        "top_n_positive_24h_fraction": (
            sum(row["return_24h"] > 0 for row in selected_top) / len(selected_top)
            if selected_top
            else None
        ),
        "altcoin_eligible_count": len(altcoins),
        "altcoin_positive_24h_fraction": (
            sum(row["return_24h"] > 0 for row in altcoins) / len(altcoins)
            if altcoins
            else None
        ),
        "top10_market_cap_concentration": sum(caps[:10]) / total_cap if total_cap else None,
        "btc_dominance": btc_dominance,
        "eth_dominance": eth_dominance,
        "btc_dominance_change": (
            btc_dominance - float(prior_btc)
            if btc_dominance is not None and prior_btc is not None
            else None
        ),
        "positive_return_volume_fraction": (
            sum(row["volume_24h"] for row in volume_rows if row["return_24h"] > 0)
            / volume_total
            if volume_total
            else None
        ),
        "total_market_cap": global_values.get("total_market_cap") or total_cap or None,
        "total_volume_24h": global_values.get("total_volume_24h") or volume_total or None,
        "definitions_version": "cmc_breadth_definitions_v2",
        "raw_inputs_stored_separately": True,
        "future_universe_membership_used": False,
        "execution_authority": False,
    }


def compare_ohlcv_sources(
    series: Mapping[str, pd.DataFrame],
    *,
    relative_tolerance: float = 0.03,
) -> list[dict[str, Any]]:
    normalized: dict[str, pd.Series] = {}
    for source, frame in series.items():
        if "close" not in frame:
            continue
        selected = frame.copy()
        selected.index = pd.to_datetime(selected.index, utc=True)
        normalized[source] = pd.to_numeric(selected["close"], errors="coerce")
    sources = sorted(normalized)
    output: list[dict[str, Any]] = []
    for first_index, first in enumerate(sources):
        for second in sources[first_index + 1 :]:
            aligned = pd.concat(
                [normalized[first].rename("first"), normalized[second].rename("second")],
                axis=1,
            )
            missing = int(aligned.isna().any(axis=1).sum())
            complete = aligned.dropna()
            if complete.empty:
                status = "SOURCE_GAP"
                maximum = None
            elif (complete <= 0).any().any():
                status = "SOURCE_INVALID"
                maximum = None
            else:
                relative = (complete["first"] / complete["second"] - 1).abs()
                maximum = float(relative.max())
                status = (
                    "SOURCE_DISAGREEMENT" if maximum > relative_tolerance else "SOURCE_AGREEMENT_OK"
                )
            output.append(
                {
                    "source_a": first,
                    "source_b": second,
                    "overlap_rows": len(complete),
                    "missing_rows": missing,
                    "maximum_relative_close_difference": maximum,
                    "status": status,
                    "arbitrary_source_override": False,
                }
            )
    return output


@dataclass(frozen=True, slots=True)
class GovernedEvent:
    source_name: str
    source_url: str
    source_quality: SourceQuality
    data_type: str
    first_observed_at: datetime
    ingested_at: datetime
    parser_version: str
    content_hash: str
    parsed_fields: Mapping[str, Any]
    published_at: datetime | None = None
    event_occurred_at: datetime | None = None
    updated_at: datetime | None = None
    deduplication_key: str | None = None

    def __post_init__(self) -> None:
        first = _utc(self.first_observed_at, required=True)
        ingested = _utc(self.ingested_at, required=True)
        published = _utc(self.published_at)
        if ingested < first:
            raise ValueError("event ingestion cannot precede first observation")
        if published and published > first:
            raise ValueError("future publication time cannot be historically known")

    @property
    def event_id(self) -> str:
        return stable_hash(
            [
                EVENT_SCHEMA_VERSION,
                self.deduplication_key or [self.source_name, self.source_url, self.content_hash],
            ],
            length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_quality": self.source_quality.value,
            "data_type": self.data_type,
            "event_occurred_at": utc_iso(self.event_occurred_at)
            if self.event_occurred_at
            else None,
            "published_at": utc_iso(self.published_at) if self.published_at else None,
            "first_observed_at": utc_iso(self.first_observed_at),
            "updated_at": utc_iso(self.updated_at) if self.updated_at else None,
            "ingested_at": utc_iso(self.ingested_at),
            "parser_version": self.parser_version,
            "content_hash": self.content_hash,
            "parsed_fields": dict(self.parsed_fields),
            "full_text_stored": False,
        }


class PointInTimeFeatureStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def append(self, rows: Sequence[Mapping[str, Any]], *, family: DataFamily) -> dict[str, Any]:
        if not rows:
            raise ValueError("feature store append cannot be empty")
        required = {
            "canonical_asset_id",
            "feature",
            "value",
            "source_timestamp",
            "known_at",
            "source",
            "quality",
        }
        normalized = []
        for row in rows:
            if not required <= set(row):
                raise ValueError(f"PIT feature row missing {sorted(required - set(row))}")
            source_time = _utc(row["source_timestamp"], required=True)
            known_at = _utc(row["known_at"], required=True)
            if known_at < source_time - timedelta(days=7):
                raise ValueError("known-at time is implausibly before source time")
            normalized.append(
                {
                    **dict(row),
                    "source_timestamp": utc_iso(source_time),
                    "known_at": utc_iso(known_at),
                    "family": family.value,
                    "schema_version": FEATURE_STORE_SCHEMA_VERSION,
                }
            )
        content_hash = stable_hash(normalized, length=64)
        known = min(parse_utc(str(row["known_at"])) for row in normalized)
        directory = self.root / family.value.casefold() / f"date={known:%Y-%m-%d}"
        target = directory / f"part-{content_hash}.parquet"
        if target.is_file():
            return {"status": "REUSED", "path": str(target), "content_hash": content_hash}
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
        pq.write_table(pa.Table.from_pylist(normalized), temporary, compression="zstd")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return {
            "status": "APPENDED",
            "path": str(target.resolve()),
            "content_hash": content_hash,
            "file_sha256": sha256_file(target),
            "row_count": len(normalized),
        }

    def as_of(
        self,
        canonical_asset_id: str,
        at: datetime,
        *,
        families: Sequence[DataFamily] | None = None,
    ) -> list[dict[str, Any]]:
        cutoff = _utc(at, required=True)
        selected_families = families or tuple(DataFamily)
        frames = []
        for family in selected_families:
            files = sorted((self.root / family.value.casefold()).rglob("*.parquet"))
            for path in files:
                frames.append(pd.read_parquet(path))
        if not frames:
            return []
        frame = pd.concat(frames, ignore_index=True)
        frame["known_at"] = pd.to_datetime(frame["known_at"], utc=True)
        selected = frame.loc[
            (frame["canonical_asset_id"] == canonical_asset_id) & (frame["known_at"] <= cutoff)
        ].sort_values(["feature", "source", "known_at"])
        latest = selected.groupby(["feature", "source"], as_index=False).tail(1)
        return latest.to_dict("records")


class DatasetFreezeManager:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def freeze(
        self,
        *,
        family: DataFamily,
        collection_epoch: datetime,
        data_end: datetime,
        source_manifests: Sequence[Mapping[str, Any]],
        readiness: Mapping[str, Any],
        holdout_fraction: float = 0.20,
        minimum_holdout_days: int = 7,
    ) -> dict[str, Any]:
        if not 0.10 <= holdout_fraction <= 0.50:
            raise ValueError("holdout fraction must be between 10% and 50%")
        start = _utc(collection_epoch, required=True)
        end = _utc(data_end, required=True)
        if end <= start:
            raise ValueError("dataset end must follow collection epoch")
        duration = end - start
        holdout_duration = max(
            timedelta(days=minimum_holdout_days),
            duration * holdout_fraction,
        )
        if holdout_duration >= duration:
            raise ValueError("insufficient history to reserve the required holdout")
        holdout_start = end - holdout_duration
        identity = {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "family": family.value,
            "collection_epoch": utc_iso(start),
            "trainable_cutoff": utc_iso(holdout_start),
            "holdout_start": utc_iso(holdout_start),
            "data_end": utc_iso(end),
            "source_manifests": list(source_manifests),
            "readiness": dict(readiness),
        }
        dataset_id = stable_hash(identity, length=64)
        target = self.root / family.value.casefold() / dataset_id / "manifest.json"
        payload = {
            **identity,
            "dataset_id": dataset_id,
            "holdout_status": "RESERVED_UNTOUCHED",
            "target_economics_inspected": False,
            "immutable": True,
            "automatic_alpha_started": False,
            "created_at": utc_iso(),
        }
        if target.is_file():
            existing = dict(read_json(target))
            existing_identity = {key: existing.get(key) for key in identity}
            if stable_hash(existing_identity) != stable_hash(identity):
                raise RuntimeError("immutable dataset freeze collision")
            return existing
        atomic_write_json(target, payload)
        return payload


def hypothesis_specific_readiness(families: Mapping[str, str]) -> dict[str, Any]:
    def usable(name: str) -> bool:
        return families.get(name) in {
            FamilyReadiness.RESEARCH_USABLE.value,
            FamilyReadiness.ROBUSTNESS_USABLE.value,
        }

    requirements = {
        "CROSS_VENUE_LEAD_LAG": ("BITVAVO_FLOW", "KRAKEN_FLOW", "CROSS_VENUE"),
        "FLOW_CONFIRMED_SWING": ("BITVAVO_FLOW", "BITVAVO_L2", "HISTORICAL_4H_1H"),
        "BREADTH_CONDITIONED_ALPHA": ("CMC_BREADTH", "CMC_UNIVERSE", "BITVAVO_PRICE"),
        "THREE_VENUE_DISLOCATION": ("BITVAVO_L2", "KRAKEN_L2", "MEXC_SPOT"),
        "EVENT_CONDITIONED_SWING": ("EVENT_DATA", "BITVAVO_PRICE", "HISTORICAL_4H_1H"),
    }
    return {
        hypothesis: {
            "required_families": list(required),
            "ready": all(usable(name) for name in required),
            "missing_or_unready": [name for name in required if not usable(name)],
            "automatic_research_start": False,
        }
        for hypothesis, required in requirements.items()
    }


def combined_manifest_integrity(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in audits]
    passed = bool(rows) and all(row.get("status") == "PASSED" for row in rows)
    return {
        "status": "PASSED" if passed else "FAILED",
        "sources": [row.get("source") for row in rows],
        "root_hashes": {str(row.get("source")): row.get("root_hash") for row in rows},
        "combined_manifest_hash": stable_hash(rows, length=64),
        "execution_health_dependency": False,
    }


__all__ = [
    "API_BUDGET_SCHEMA_VERSION",
    "CanonicalAssetIdentity",
    "CanonicalAssetRegistry",
    "DataClassification",
    "DataFamily",
    "DatasetFreezeManager",
    "FamilyReadiness",
    "GovernedEvent",
    "ImmutableSourceLedger",
    "KrakenL2Book",
    "MULTI_SOURCE_SCHEMA_VERSION",
    "PointInTimeFeatureStore",
    "ReadinessThreshold",
    "SourceAuthority",
    "SourceNeutralObservation",
    "SourcePolicy",
    "SourceQuality",
    "TimestampResolution",
    "TradeCoverage",
    "BookCoverage",
    "ApiBudgetLedger",
    "ApiBudgetRule",
    "assess_family_readiness",
    "combined_manifest_integrity",
    "compact_source_ledger",
    "compare_ohlcv_sources",
    "compute_cmc_breadth",
    "default_api_budget_rules",
    "default_readiness_thresholds",
    "hypothesis_specific_readiness",
    "initial_multi_source_asset_registry",
    "mexc_semantic_source",
    "normalize_quote_asset",
    "interval_overlap",
    "source_authority_registry",
    "verify_source_ledger",
]
