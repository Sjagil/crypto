"""Independent public/read-only P1.2.1 multi-source collection runtime.

This composition root deliberately imports no execution module and exposes no
order, account, balance, position, allocator, or strategy-promotion method.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.contracts import NormalizedDataRecord, NormalizedStreamEvent, StreamEventType
from data.bitvavo_l2_reconstruction_v2 import (
    L2_RECONSTRUCTION_VERSION,
    BitvavoBookState,
    BitvavoL2StateMachine,
)
from data.data_loader import DataLoader
from data.market_structure import BookQuality
from data.multi_source_maturation import (
    CollectorLease,
    CrossVenueAlignmentMonitor,
    FamilyFreezeManager,
    ReadinessHistoryStore,
    RuntimePerformanceMonitor,
    StorageGrowthMonitor,
    api_usage_report,
    assess_readiness,
    bitvavo_l2_maturation,
    classify_event,
    hypothesis_readiness,
    mexc_derivatives_maturation,
    research_readiness_policy_v1,
)
from data.multi_source_platform import (
    API_BUDGET_SCHEMA_VERSION,
    KRAKEN_COLLECTOR_VERSION,
    MULTI_SOURCE_SCHEMA_VERSION,
    ApiBudgetLedger,
    BookCoverage,
    DataClassification,
    DataFamily,
    GovernedEvent,
    ImmutableSourceLedger,
    KrakenL2Book,
    PointInTimeFeatureStore,
    SourceNeutralObservation,
    SourceQuality,
    TimestampResolution,
    TradeCoverage,
    compact_source_ledger,
    compute_cmc_breadth,
    default_api_budget_rules,
    initial_multi_source_asset_registry,
    interval_overlap,
    source_authority_registry,
)
from data.websocket_manager import WebSocketManager
from scrapers.rss import collect_registered_feeds
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso, utc_now

RUNTIME_SCHEMA_VERSION = "multi_source_public_collector_runtime_v1"
KRAKEN_EPOCH_SCHEMA_VERSION = "kraken_collection_epoch_v1"
PRIMARY_ASSETS = ("BTC", "ETH", "SOL")
PUBLIC_STREAM_SUBSCRIPTIONS: dict[str, dict[str, Any]] = {
    "bitvavo": {
        "trades": ["BTC-EUR", "ETH-EUR", "SOL-EUR"],
        "book": ["BTC-EUR", "ETH-EUR", "SOL-EUR"],
        "ticker": ["BTC-EUR", "ETH-EUR", "SOL-EUR"],
    },
    "kraken": {
        "trade": {
            "markets": ["BTC/EUR", "ETH/EUR", "SOL/EUR"],
            "snapshot": False,
        },
        "book": {
            "markets": ["BTC/EUR", "ETH/EUR", "SOL/EUR"],
            "depth": 100,
            "snapshot": True,
        },
        "ticker": ["BTC/EUR", "ETH/EUR", "SOL/EUR"],
    },
    "mexc": {
        "trades": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "book": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "ticker": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    },
}


def _positive_decimal(value: Any) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() and selected > 0 else None


def _source_name(provider: str) -> str:
    return "mexc_spot" if provider.casefold() == "mexc" else provider.casefold()


def _source_partition(settings: Settings, source: str) -> Path:
    mapping = {
        "bitvavo": settings.paths.raw_data_dir / "bitvavo" / "prospective_pit",
        "kraken": settings.paths.raw_data_dir / "kraken" / "prospective_pit",
        "mexc_spot": settings.paths.raw_data_dir / "mexc" / "spot_prospective_pit",
        "coinmarketcap": settings.paths.raw_data_dir / "coinmarketcap" / "prospective_pit",
        "eodhd": settings.paths.raw_data_dir / "eodhd" / "prospective_pit",
        "scrapers": settings.paths.raw_data_dir / "scrapers" / "governed_events",
    }
    return mapping[source]


class MultiSourceCollector:
    """Supervise independent public streams and low-frequency context reads."""

    def __init__(
        self,
        settings: Settings,
        *,
        status_interval_seconds: float = 10.0,
        batch_interval_seconds: float = 0.5,
        context_interval_seconds: float = 3_600.0,
        rss_interval_seconds: float = 1_800.0,
        websocket_manager: WebSocketManager | None = None,
        data_loader: DataLoader | None = None,
        notifier: Any | None = None,
        lease: CollectorLease | None = None,
    ) -> None:
        self.settings = settings
        self.output_root = settings.paths.output_dir / "multi_source"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = self.output_root / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_root / "status.json"
        self.heartbeat_path = self.output_root / "heartbeat.json"
        self.stop_path = self.output_root / "STOP"
        self.lease = lease or CollectorLease(
            self.output_root / "collector.lock", self.output_root / "ownership_history"
        )
        self.status_interval_seconds = status_interval_seconds
        self.batch_interval_seconds = batch_interval_seconds
        self.context_interval_seconds = context_interval_seconds
        self.rss_interval_seconds = rss_interval_seconds
        previous_status = dict(read_json(self.status_path)) if self.status_path.is_file() else {}
        self.process_started_at = utc_now()
        self.registry = initial_multi_source_asset_registry()
        self.policies = source_authority_registry()
        self.manager = websocket_manager or WebSocketManager(
            queue_size=20_000,
            maximum_connection_attempts=20,
            inactivity_timeout=45,
            ticker_minimum_interval_seconds=0.25,
        )
        self.loader = data_loader or DataLoader(settings)
        self.notifier = notifier
        self.ledgers = {
            source: ImmutableSourceLedger(
                _source_partition(settings, source),
                source,
                self.checkpoint_root / f"{source}.json",
            )
            for source in (
                "bitvavo",
                "kraken",
                "mexc_spot",
                "coinmarketcap",
                "eodhd",
                "scrapers",
            )
        }
        self.budget = ApiBudgetLedger(
            self.output_root / "api_budget.json",
            default_api_budget_rules(),
        )
        started = (
            datetime.fromisoformat(str(previous_status["started_at"]).replace("Z", "+00:00"))
            if previous_status.get("started_at")
            else utc_now()
        )
        now = utc_now()
        self.started_at = started
        previous_trades = dict(previous_status.get("trade_coverage") or {})
        self.trade_coverage: dict[tuple[str, str], TradeCoverage] = {}
        for source in ("bitvavo", "kraken", "mexc_spot"):
            for asset in PRIMARY_ASSETS:
                asset_id = f"CRYPTO:{asset}"
                prior = dict(previous_trades.get(f"{source}:{asset_id}") or {})
                first_known = prior.get("first_known_at")
                last_known = prior.get("last_known_at")
                self.trade_coverage[(source, asset_id)] = TradeCoverage(
                    source,
                    asset_id,
                    trade_count=int(prior.get("trade_count") or 0),
                    valid_trade_count=int(prior.get("valid_trade_count") or 0),
                    duplicate_rejected=int(prior.get("duplicate_rejected") or 0),
                    out_of_order_count=int(prior.get("out_of_order_count") or 0),
                    gap_indicators=int(prior.get("gap_indicators") or 0),
                    total_notional=Decimal(str(prior.get("notional") or "0")),
                    first_known_at=(
                        datetime.fromisoformat(str(first_known).replace("Z", "+00:00"))
                        if first_known
                        else None
                    ),
                    last_known_at=(
                        datetime.fromisoformat(str(last_known).replace("Z", "+00:00"))
                        if last_known
                        else None
                    ),
                    last_event_at=(
                        datetime.fromisoformat(str(last_known).replace("Z", "+00:00"))
                        if last_known
                        else None
                    ),
                )
        previous_books = dict(previous_status.get("book_coverage") or {})
        self.book_coverage: dict[tuple[str, str], BookCoverage] = {}
        for source in ("bitvavo", "kraken", "mexc_spot"):
            for asset in PRIMARY_ASSETS:
                asset_id = f"CRYPTO:{asset}"
                prior = dict(previous_books.get(f"{source}:{asset_id}") or {})
                collection_start = prior.get("collection_started_at")
                coverage = BookCoverage(
                    source,
                    asset_id,
                    (
                        datetime.fromisoformat(str(collection_start).replace("Z", "+00:00"))
                        if collection_start
                        else started
                    ),
                    now,
                    current_state=BookQuality.BOOK_SYNCING,
                )
                coverage.durations.update(
                    {
                        BookQuality.BOOK_VALID.value: float(prior.get("book_valid_seconds") or 0),
                        BookQuality.BOOK_INVALID.value: float(
                            prior.get("book_invalid_seconds") or 0
                        ),
                        BookQuality.BOOK_STALE.value: float(prior.get("book_stale_seconds") or 0),
                        BookQuality.BOOK_GAPPED.value: float(prior.get("book_gap_seconds") or 0),
                        BookQuality.BOOK_SYNCING.value: float(
                            prior.get("book_syncing_seconds") or 0
                        ),
                    }
                )
                coverage.failures.update(dict(prior.get("failure_counts") or {}))
                self.book_coverage[(source, asset_id)] = coverage
        self.kraken_books = {
            symbol: KrakenL2Book(symbol, depth=100, stale_seconds=30)
            for symbol in ("BTC/EUR", "ETH/EUR", "SOL/EUR")
        }
        # Never restore an in-memory Bitvavo book across a process boundary.
        # Each process starts fail-closed and must subscribe, buffer, then
        # acquire a fresh public REST snapshot before any feature is valid.
        self.bitvavo_books = {
            f"{asset}-EUR": BitvavoL2StateMachine(
                f"{asset}-EUR",
                maximum_levels=min(
                    500,
                    int(self.settings.market_data.orderbook_maximum_depth),
                ),
                stale_after=timedelta(seconds=30),
            )
            for asset in PRIMARY_ASSETS
        }
        self._bitvavo_sync_buffering: dict[str, bool] = {
            market: False for market in self.bitvavo_books
        }
        self._bitvavo_reseeding: set[str] = set()
        self._bitvavo_buffers: dict[str, deque[NormalizedStreamEvent]] = {
            market: deque() for market in self.bitvavo_books
        }
        self._bitvavo_buffer_limit = 20_000
        self._bitvavo_buffer_overflows: Counter[str] = Counter()
        self._bitvavo_reseed_attempts: Counter[str] = Counter()
        self._bitvavo_next_reseed_at: dict[str, datetime] = {}
        self.pending: dict[str, list[SourceNeutralObservation]] = defaultdict(list)
        previous_sources = dict(previous_status.get("source_status") or {})
        self.source_status: dict[str, dict[str, Any]] = {
            source: {
                "state": ("AVAILABLE" if self.ledgers[source].record_count > 0 else "STARTING"),
                "last_success": (
                    (previous_sources.get(source) or {}).get("last_success")
                    or self.ledgers[source].last_known_at
                ),
                "error_count": int((previous_sources.get(source) or {}).get("error_count") or 0),
                "last_error": (previous_sources.get(source) or {}).get("last_error"),
                "rate_limit": "WITHIN_LOCAL_BUDGET",
            }
            for source in self.ledgers
        }
        self.context_counts: dict[str, int] = defaultdict(
            int,
            {
                str(key): int(value)
                for key, value in dict(previous_status.get("context_counts") or {}).items()
            },
        )
        self.latest_cmc_breadth: dict[str, Any] | None = previous_status.get("latest_cmc_breadth")
        self.latest_cmc_global: dict[str, Any] | None = previous_status.get("latest_cmc_global")
        self.readiness_policy = research_readiness_policy_v1()
        self.readiness_history = ReadinessHistoryStore(self.output_root / "readiness")
        self.freeze_manager = FamilyFreezeManager(self.output_root / "freezes")
        self.alignment = CrossVenueAlignmentMonitor(self.output_root / "alignment.json")
        self.storage_monitor = StorageGrowthMonitor(self.output_root / "storage")
        self.storage_cumulative: dict[str, dict[str, int]] = {
            str(key): {
                "events": int(value.get("events") or 0),
                "raw_bytes": int(value.get("raw_bytes") or 0),
                "compressed_events": int(value.get("compressed_events") or 0),
                "compressed_bytes": int(value.get("compressed_bytes") or 0),
            }
            for key, value in dict(previous_status.get("storage_cumulative") or {}).items()
        }
        self.storage_measurement_started_at = str(
            previous_status.get("storage_measurement_started_at") or utc_iso()
        )
        self.performance_monitor = RuntimePerformanceMonitor()
        self.performance: dict[str, Any] = dict(previous_status.get("performance") or {})
        self.compaction: dict[str, Any] = dict(previous_status.get("compaction") or {})
        self.storage: dict[str, Any] = dict(previous_status.get("storage") or {})
        self._performance_event_baseline = sum(
            ledger.record_count for ledger in self.ledgers.values()
        )
        self._storage_critical_notified = self.storage.get("status") == "STORAGE_CRITICAL"
        self.feature_store = PointInTimeFeatureStore(
            settings.paths.data_dir / "features" / "multi_source_pit"
        )
        self._last_compaction_at: datetime | None = utc_now()
        self._last_storage_at: datetime | None = None
        self._event_evidence: dict[str, Any] = {
            "unique_ids": set(),
            "mapped_assets": set(),
            "categories": set(),
            "high_value_count": 0,
            "mapped_event_count": 0,
            "first_known_min": None,
            "first_known_max": None,
        }
        self._restore_context_evidence()
        self._last_restart: dict[str, datetime] = {}
        self._stop = asyncio.Event()
        self._epoch = self._ensure_kraken_epoch()

    def _restore_context_evidence(self) -> None:
        """Recover low-frequency counters from immutable ledgers after restarts."""

        latest_cmc_run: str | None = None
        latest_cmc_rows: list[dict[str, Any]] = []
        cmc_runs: set[str] = set()
        reconstructed: dict[str, int] = defaultdict(int)
        for source in ("coinmarketcap", "eodhd", "scrapers"):
            root = _source_partition(self.settings, source)
            for path in sorted(root.rglob("events.jsonl")):
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        record = dict(json.loads(line))
                        data_type = str(record.get("data_type") or "")
                        metadata = dict(record.get("metadata") or {})
                        if source == "coinmarketcap" and data_type == "UNIVERSE_RANKING":
                            run_id = str(metadata.get("retrieval_run_id") or "")
                            if run_id:
                                cmc_runs.add(run_id)
                            if run_id != latest_cmc_run:
                                latest_cmc_run = run_id
                                latest_cmc_rows = []
                            latest_cmc_rows.append(dict(metadata.get("values") or {}))
                        elif source == "coinmarketcap" and data_type == "MACRO_OBSERVATION":
                            reconstructed["cmc_global"] += 1
                            self.latest_cmc_global = dict(metadata.get("values") or {})
                        elif source == "eodhd":
                            instrument = str(record.get("venue_instrument_id") or "")
                            asset = instrument.split("-", 1)[0].casefold()
                            if asset:
                                reconstructed[f"eodhd_{asset}_bars"] += 1
                        elif source == "scrapers" and data_type == "EVENT_INTELLIGENCE":
                            reconstructed["governed_events"] += 1
                            raw_event = dict(record.get("raw_payload") or {})
                            parsed = dict(raw_event.get("parsed_fields") or {})
                            classification = classify_event(
                                source=str(raw_event.get("source_name") or ""),
                                title=str(parsed.get("title") or ""),
                                summary=str(parsed.get("summary") or ""),
                                existing_categories=parsed.get("event_categories")
                                or parsed.get("categories")
                                or (),
                            )
                            evidence = self._event_evidence
                            evidence["unique_ids"].add(str(record.get("source_event_id") or ""))
                            evidence["mapped_assets"].update(classification["canonical_asset_ids"])
                            evidence["categories"].update(classification["event_categories"])
                            evidence["high_value_count"] += int(classification["high_value_event"])
                            evidence["mapped_event_count"] += int(
                                bool(classification["canonical_asset_ids"])
                            )
                            known = record.get("known_at")
                            evidence["first_known_min"] = evidence["first_known_min"] or known
                            evidence["first_known_max"] = known or evidence["first_known_max"]
        for key, value in reconstructed.items():
            self.context_counts[key] = value
        if latest_cmc_rows:
            self.context_counts["cmc_rankings"] = len(latest_cmc_rows)
            self.context_counts["cmc_breadth_snapshots"] = len(cmc_runs)
            if self.latest_cmc_breadth is None:
                self.latest_cmc_breadth = compute_cmc_breadth(
                    latest_cmc_rows,
                    known_at=utc_now(),
                    global_context=self.latest_cmc_global,
                )

    def _ensure_kraken_epoch(self) -> dict[str, Any]:
        target = self.output_root / "KRAKEN_COLLECTION_EPOCH.json"
        identity = {
            "schema_version": KRAKEN_EPOCH_SCHEMA_VERSION,
            "source": "kraken",
            "collection_semantics": "PUBLIC_SPOT_WEBSOCKET_V2_PROSPECTIVE_ONLY",
            "collector_version": KRAKEN_COLLECTOR_VERSION,
            "symbols": ["BTC/EUR", "ETH/EUR", "SOL/EUR"],
            "book_depth": 100,
            "checksum_contract": "KRAKEN_CRC32_TOP10_V2",
            "execution_authority": False,
            "private_api_used": False,
        }
        if target.is_file():
            existing = dict(read_json(target))
            if any(existing.get(key) != value for key, value in identity.items()):
                raise RuntimeError("immutable Kraken collection epoch contract changed")
            return existing
        payload = {**identity, "collection_epoch": utc_iso()}
        atomic_write_json(target, payload)
        return payload

    def _resolve_asset(self, event: NormalizedStreamEvent) -> str | None:
        if event.event_type is StreamEventType.CONNECTION_STATUS:
            return None
        identity = self.registry.resolve(_source_name(event.provider), event.source_symbol)
        return identity.canonical_asset_id

    def _stream_observation(self, event: NormalizedStreamEvent) -> SourceNeutralObservation:
        source = _source_name(event.provider)
        canonical_asset_id = self._resolve_asset(event)
        raw = event.payload.get("provider_payload", event.payload)
        freshness = max(0.0, (event.observed_at - event.timestamp).total_seconds())
        return SourceNeutralObservation(
            source=source,
            source_type="PUBLIC_WEBSOCKET",
            venue=event.provider,
            canonical_asset_id=canonical_asset_id,
            venue_instrument_id=event.source_symbol,
            data_type=event.event_type.value.upper(),
            exchange_event_timestamp=event.timestamp,
            local_receive_timestamp=event.observed_at,
            normalized_timestamp=event.timestamp,
            persisted_timestamp=utc_now(),
            raw_payload=raw,
            timestamp_resolution=TimestampResolution.EVENT_EXACT,
            quality_state=self._event_quality(event),
            freshness_seconds=freshness,
            source_event_id=event.message_id,
            classification=DataClassification.PROSPECTIVE_COLLECTION,
            metadata={
                "canonical_market": event.canonical_market,
                "source_sequence": event.sequence,
                "raw_payload_hash": event.payload.get("raw_payload_hash"),
                "public_only": True,
                "orders_generated": 0,
                "private_exchange_requests": 0,
            },
        )

    @staticmethod
    def _event_quality(event: NormalizedStreamEvent) -> str:
        if event.event_type is StreamEventType.CONNECTION_STATUS:
            return str(event.payload.get("state") or "UNKNOWN")
        if event.provider == "mexc" and event.event_type is StreamEventType.ORDERBOOK_DELTA:
            return "REFERENCE_DELTA_UNRECONSTRUCTED"
        return "PRESENT"

    def _observe_trade(self, event: NormalizedStreamEvent, asset_id: str) -> None:
        payload = event.payload
        price = _positive_decimal(payload.get("price"))
        quantity = _positive_decimal(payload.get("base_quantity") or payload.get("quantity"))
        if price is None or quantity is None:
            return
        semantics = str(
            payload.get("aggressor_semantics")
            or ("EXCHANGE_REPORTED_TAKER_SIDE" if payload.get("side") else "UNKNOWN")
        )
        self.trade_coverage[(_source_name(event.provider), asset_id)].observe(
            known_at=event.observed_at,
            event_at=event.timestamp,
            notional=price * quantity,
            aggressor_semantics=semantics,
        )
        self.alignment.observe(
            source=_source_name(event.provider),
            canonical_asset_id=asset_id,
            event_at=event.timestamp,
            receive_at=event.observed_at,
        )

    def _handle_connection_status(self, event: NormalizedStreamEvent) -> None:
        state = str(event.payload.get("state") or "UNKNOWN").upper()
        source = _source_name(event.provider)
        self.source_status[source]["state"] = state
        if state in {"DISCONNECTED", "RECONNECTING", "FAILED"}:
            self.alignment.record_reconnect(source)
            if source == "kraken":
                for symbol, book in self.kraken_books.items():
                    asset = symbol.split("/", 1)[0]
                    coverage = self.book_coverage[("kraken", f"CRYPTO:{asset}")]
                    book.disconnect()
                    coverage.transition(
                        BookQuality.BOOK_SYNCING,
                        event.observed_at,
                        "RECONNECT_RESET",
                    )
            elif source == "bitvavo":
                for market, book in self.bitvavo_books.items():
                    asset = market.split("-", 1)[0]
                    book.on_disconnect(event.observed_at)
                    self._bitvavo_sync_buffering[market] = False
                    self.book_coverage[("bitvavo", f"CRYPTO:{asset}")].transition(
                        BookQuality.BOOK_GAPPED,
                        event.observed_at,
                        "RECONNECT_RESET",
                    )

    def _observe_book(self, event: NormalizedStreamEvent, asset_id: str) -> None:
        source = _source_name(event.provider)
        coverage = self.book_coverage[(source, asset_id)]
        if source == "bitvavo":
            market = event.canonical_market
            if market in self._bitvavo_reseeding:
                buffer = self._bitvavo_buffers[market]
                if len(buffer) >= self._bitvavo_buffer_limit:
                    self._bitvavo_buffer_overflows[market] += 1
                else:
                    buffer.append(event)
                return
            book = self.bitvavo_books[market]
            applied = book.apply_delta(
                bids=event.payload.get("bids") or [],
                asks=event.payload.get("asks") or [],
                sequence=event.sequence,
                event_at=event.timestamp,
                known_at=event.observed_at,
                event_id=event.message_id,
                buffered_after_snapshot=self._bitvavo_sync_buffering[market],
            )
            if applied:
                self._bitvavo_sync_buffering[market] = False
            if book.state is BitvavoBookState.VALID:
                desired = BookQuality.BOOK_VALID
                reason = None
            elif book.state is BitvavoBookState.STALE:
                desired = BookQuality.BOOK_STALE
                reason = "BOOK_STALE"
            elif book.state in {
                BitvavoBookState.GAPPED,
                BitvavoBookState.RESEED_REQUIRED,
            }:
                desired = BookQuality.BOOK_GAPPED
                reason = next(reversed(book.failure_counts), "RESEED_REQUIRED")
            else:
                desired = BookQuality.BOOK_INVALID
                reason = book.state.value
            if coverage.current_state is not desired:
                coverage.transition(desired, event.observed_at, reason)
            return
        if source != "kraken":
            # Raw deltas are retained, but validity is never invented without
            # the venue-specific snapshot/replay contract.
            desired = BookQuality.BOOK_SYNCING
            if coverage.current_state is not desired:
                coverage.transition(desired, event.observed_at, "VENUE_REPLAYER_NOT_PROVEN")
            return
        book = self.kraken_books[event.source_symbol]
        payload = event.payload
        kind = "snapshot" if event.event_type is StreamEventType.ORDERBOOK_SNAPSHOT else "update"
        checksum = payload.get("checksum")
        if checksum is None:
            coverage.transition(BookQuality.BOOK_INVALID, event.observed_at, "CHECKSUM_MISSING")
            return
        result = book.apply(
            kind=kind,
            bids=payload.get("bids") or [],
            asks=payload.get("asks") or [],
            checksum=int(checksum),
            event_timestamp=event.timestamp,
            receive_timestamp=event.observed_at,
            message_id=event.message_id,
        )
        desired = book.quality_at(event.observed_at)
        if coverage.current_state is not desired:
            reason = None if result["status"] == "APPLIED" else result["status"]
            coverage.transition(desired, event.observed_at, reason)

    def _bitvavo_snapshot_observation(
        self,
        record: NormalizedDataRecord,
        asset_id: str,
    ) -> SourceNeutralObservation:
        payload = record.raw_payload or {
            "market": record.canonical_market,
            "nonce": record.values.get("sequence"),
            "bids": record.values.get("bids") or [],
            "asks": record.values.get("asks") or [],
        }
        return SourceNeutralObservation(
            source="bitvavo",
            source_type="PUBLIC_REST",
            venue="bitvavo",
            canonical_asset_id=asset_id,
            venue_instrument_id=record.source_symbol,
            data_type="ORDERBOOK_SNAPSHOT",
            provider_timestamp=record.timestamp,
            local_receive_timestamp=record.observed_at,
            normalized_timestamp=record.timestamp,
            persisted_timestamp=utc_now(),
            raw_payload=payload,
            timestamp_resolution=TimestampResolution.RETRIEVAL_ONLY,
            quality_state="TRUSTED_RESEED_SNAPSHOT",
            freshness_seconds=max(
                0.0,
                (record.observed_at - record.timestamp).total_seconds(),
            ),
            source_event_id=record.raw_hash,
            classification=DataClassification.PROSPECTIVE_COLLECTION,
            metadata={
                "canonical_market": record.canonical_market,
                "source_sequence": record.values.get("sequence"),
                "raw_payload_hash": record.raw_hash,
                "reconstruction_version": L2_RECONSTRUCTION_VERSION,
                "public_only": True,
                "orders_generated": 0,
                "private_exchange_requests": 0,
            },
        )

    async def _reseed_bitvavo_market(self, market: str) -> bool:
        asset_id = f"CRYPTO:{market.split('-', 1)[0]}"
        book = self.bitvavo_books[market]
        coverage = self.book_coverage[("bitvavo", asset_id)]
        self._bitvavo_reseeding.add(market)
        self._bitvavo_buffers[market].clear()
        overflow_before = self._bitvavo_buffer_overflows[market]
        try:
            record = await self.loader.download_orderbook_snapshot(
                provider="bitvavo",
                market=market,
                depth=min(
                    500,
                    int(self.settings.market_data.orderbook_maximum_depth),
                ),
                persist=False,
                mode="public_research_l2_v2_reseed",
            )
            self.pending["bitvavo"].append(self._bitvavo_snapshot_observation(record, asset_id))
            accepted = book.seed_snapshot(
                bids=record.values.get("bids") or [],
                asks=record.values.get("asks") or [],
                sequence=record.values.get("sequence"),
                event_at=record.timestamp,
                known_at=record.observed_at,
                snapshot_reference=record.raw_hash,
            )
            buffered = sorted(
                self._bitvavo_buffers[market],
                key=lambda row: (
                    row.sequence if row.sequence is not None else -1,
                    row.observed_at,
                    row.message_id,
                ),
            )
            if self._bitvavo_buffer_overflows[market] > overflow_before:
                book.require_reseed(utc_now(), "RESEED_BUFFER_OVERFLOW")
                accepted = False
            elif accepted:
                self._bitvavo_sync_buffering[market] = True
                for event in buffered:
                    applied = book.apply_delta(
                        bids=event.payload.get("bids") or [],
                        asks=event.payload.get("asks") or [],
                        sequence=event.sequence,
                        event_at=event.timestamp,
                        known_at=event.observed_at,
                        event_id=event.message_id,
                        buffered_after_snapshot=True,
                    )
                    if applied:
                        self._bitvavo_sync_buffering[market] = False
                accepted = book.state is BitvavoBookState.VALID
            desired = BookQuality.BOOK_VALID if accepted else BookQuality.BOOK_GAPPED
            coverage.transition(
                desired,
                utc_now(),
                None if accepted else "RESEED_FAILED_CLOSED",
            )
            return accepted
        except Exception as exc:
            self._record_error("bitvavo", exc)
            coverage.transition(
                BookQuality.BOOK_GAPPED,
                utc_now(),
                "RESEED_PROVIDER_FAILURE",
            )
            return False
        finally:
            self._bitvavo_buffers[market].clear()
            self._bitvavo_reseeding.discard(market)

    async def _bitvavo_reseed_loop(self) -> None:
        while not self._stop.is_set():
            now = utc_now()
            health = self.manager.health("bitvavo")
            connected = str(health.get("state") or "").upper() == "CONNECTED"
            if connected:
                selected = [
                    market
                    for market, book in self.bitvavo_books.items()
                    if book.state is not BitvavoBookState.VALID
                    and now
                    >= self._bitvavo_next_reseed_at.get(
                        market,
                        datetime(1970, 1, 1, tzinfo=UTC),
                    )
                ]
                if selected:
                    results = await asyncio.gather(
                        *(self._reseed_bitvavo_market(market) for market in selected),
                        return_exceptions=True,
                    )
                    for market, result in zip(selected, results, strict=True):
                        if result is True:
                            self._bitvavo_reseed_attempts[market] = 0
                            self._bitvavo_next_reseed_at.pop(market, None)
                        else:
                            self._bitvavo_reseed_attempts[market] += 1
                            attempt = self._bitvavo_reseed_attempts[market]
                            delay = min(30.0, 0.5 * (2 ** min(attempt - 1, 6)))
                            self._bitvavo_next_reseed_at[market] = utc_now() + timedelta(
                                seconds=delay
                            )
            for market, book in self.bitvavo_books.items():
                before = book.state
                book.check_stale(utc_now())
                if before is BitvavoBookState.VALID and book.state is not before:
                    asset_id = f"CRYPTO:{market.split('-', 1)[0]}"
                    self.book_coverage[("bitvavo", asset_id)].transition(
                        BookQuality.BOOK_STALE,
                        utc_now(),
                        "BOOK_STALE",
                    )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.5)
            except TimeoutError:
                continue

    def _account_append_result(self, result: dict[str, Any]) -> None:
        for key, values in dict(result.get("storage_by_source_asset_type") or {}).items():
            cumulative = self.storage_cumulative.setdefault(
                key,
                {
                    "events": 0,
                    "raw_bytes": 0,
                    "compressed_events": 0,
                    "compressed_bytes": 0,
                },
            )
            cumulative["events"] += int(values.get("events") or 0)
            cumulative["raw_bytes"] += int(values.get("raw_bytes") or 0)

    async def _flush(self) -> None:
        for source, observations in tuple(self.pending.items()):
            if not observations:
                continue
            selected = observations[:]
            observations.clear()
            try:
                result = await asyncio.to_thread(self.ledgers[source].append_many, selected)
            except Exception as exc:  # source failure must not stop other sources
                observations[:0] = selected
                self._record_error(source, exc)
                continue
            if result.get("appended", 0) > 0:
                self._account_append_result(result)
                self._record_success(source)

    async def _stream_loop(self) -> None:
        await self.manager.start(PUBLIC_STREAM_SUBSCRIPTIONS)
        last_flush = asyncio.get_running_loop().time()
        while not self._stop.is_set():
            try:
                event = await self.manager.next_event(timeout=0.25)
            except TimeoutError:
                event = None
            if event is not None:
                try:
                    observation = self._stream_observation(event)
                    source = observation.source
                    self.pending[source].append(observation)
                    asset_id = observation.canonical_asset_id
                    if event.event_type is StreamEventType.CONNECTION_STATUS:
                        self._handle_connection_status(event)
                    elif asset_id and event.event_type is StreamEventType.TRADE:
                        self._observe_trade(event, asset_id)
                    elif asset_id and event.event_type in {
                        StreamEventType.ORDERBOOK_SNAPSHOT,
                        StreamEventType.ORDERBOOK_DELTA,
                    }:
                        self._observe_book(event, asset_id)
                except Exception as exc:
                    self._record_error(_source_name(event.provider), exc)
            now = asyncio.get_running_loop().time()
            if (
                now - last_flush >= self.batch_interval_seconds
                or sum(len(rows) for rows in self.pending.values()) >= 500
            ):
                await self._flush()
                last_flush = now
        await self._flush()

    def _record_success(self, source: str) -> None:
        state = self.source_status[source]
        state["state"] = "AVAILABLE"
        state["last_success"] = utc_iso()
        state["last_error"] = None

    def _record_error(self, source: str, exc: BaseException) -> None:
        state = self.source_status[source]
        state["state"] = "DEGRADED"
        state["error_count"] = int(state.get("error_count") or 0) + 1
        state["last_error"] = f"{type(exc).__name__}:REDACTED_PROVIDER_FAILURE"

    def _api_observation(
        self,
        source: str,
        record: NormalizedDataRecord,
        *,
        classification: DataClassification,
    ) -> SourceNeutralObservation:
        canonical_id: str | None = None
        provider_id = record.source_symbol
        try:
            if source == "coinmarketcap":
                canonical_id = self.registry.resolve_cmc(int(provider_id)).canonical_asset_id
            elif source == "eodhd":
                canonical_id = self.registry.resolve(source, provider_id).canonical_asset_id
        except (KeyError, TypeError, ValueError):
            canonical_id = None
        receive = record.observed_at
        return SourceNeutralObservation(
            source=source,
            source_type="PAID_READ_ONLY_API",
            canonical_asset_id=canonical_id,
            venue_instrument_id=provider_id,
            data_type=record.data_kind.upper(),
            provider_timestamp=record.timestamp,
            local_receive_timestamp=receive,
            normalized_timestamp=record.timestamp,
            persisted_timestamp=utc_now(),
            raw_payload=record.raw_payload if record.raw_payload is not None else record.values,
            timestamp_resolution=(
                TimestampResolution.PROVIDER_SNAPSHOT
                if source == "coinmarketcap"
                else TimestampResolution.BAR_BOUNDARY
            ),
            quality_state="PRESENT",
            freshness_seconds=max(0.0, (receive - record.timestamp).total_seconds()),
            source_event_id=stable_hash(
                [source, record.data_kind, provider_id, record.raw_hash, utc_iso(receive)],
                length=64,
            ),
            classification=classification,
            metadata={
                "available_at": utc_iso(record.available_at or receive),
                "retrieval_run_id": record.retrieval_run_id,
                "raw_hash": record.raw_hash,
                "values": record.values,
                "public_or_paid_read_only": True,
                "orders_generated": 0,
            },
        )

    async def _collect_cmc(self) -> None:
        decisions = {
            endpoint: self.budget.authorize("coinmarketcap", endpoint)
            for endpoint in ("rankings", "global_metrics")
        }
        if not any(row["allowed"] for row in decisions.values()):
            self.source_status["coinmarketcap"]["rate_limit"] = "CACHE_OR_BUDGET_GATED"
            return
        observations: list[SourceNeutralObservation] = []
        ranking_values: list[dict[str, Any]] = []
        if decisions["rankings"]["allowed"]:
            records = await self.loader.download_cmc_rankings(limit=250, persist=False)
            actual_credits = max(
                [int(row.values.get("response_credit_count") or 0) for row in records] or [0]
            ) or int(decisions["rankings"]["cost"])
            self.budget.record_request("coinmarketcap", "rankings", credits=actual_credits)
            observations.extend(
                self._api_observation(
                    "coinmarketcap", row, classification=DataClassification.PROSPECTIVE_COLLECTION
                )
                for row in records
            )
            ranking_values = [dict(row.values) for row in records]
            self.context_counts["cmc_rankings"] = len(records)
        if decisions["global_metrics"]["allowed"]:
            records = await self.loader.download_macro_series(
                provider="coinmarketcap", series="GLOBAL", persist=False
            )
            actual_credits = max(
                [int(row.values.get("response_credit_count") or 0) for row in records] or [0]
            ) or int(decisions["global_metrics"]["cost"])
            self.budget.record_request("coinmarketcap", "global_metrics", credits=actual_credits)
            observations.extend(
                self._api_observation(
                    "coinmarketcap", row, classification=DataClassification.PROSPECTIVE_COLLECTION
                )
                for row in records
            )
            if records:
                self.latest_cmc_global = dict(records[-1].values)
            self.context_counts["cmc_global"] += len(records)
        if ranking_values:
            known_at = utc_now()
            previous = self.latest_cmc_breadth
            self.latest_cmc_breadth = compute_cmc_breadth(
                ranking_values,
                known_at=known_at,
                global_context=self.latest_cmc_global,
                previous_breadth=previous,
            )
            feature_rows = [
                {
                    "canonical_asset_id": "CRYPTO:MARKET",
                    "feature": key,
                    "value": value,
                    "source_timestamp": known_at,
                    "known_at": known_at,
                    "source": "coinmarketcap",
                    "quality": "PIT_DEFINITION_V2",
                }
                for key, value in self.latest_cmc_breadth.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if feature_rows:
                await asyncio.to_thread(
                    self.feature_store.append,
                    feature_rows,
                    family=DataFamily.MARKET_BREADTH,
                )
            self.context_counts["cmc_breadth_snapshots"] += 1
        if observations:
            result = await asyncio.to_thread(
                self.ledgers["coinmarketcap"].append_many, observations
            )
            self._account_append_result(result)
            self._record_success("coinmarketcap")

    async def _collect_eodhd(self) -> None:
        decision = self.budget.authorize("eodhd", "eod", units=len(PRIMARY_ASSETS))
        if not decision["allowed"]:
            self.source_status["eodhd"]["rate_limit"] = "CACHE_OR_BUDGET_GATED"
            return
        end = utc_now()
        start = end - timedelta(days=14)
        observations: list[SourceNeutralObservation] = []
        successful_requests = 0
        for asset in PRIMARY_ASSETS:
            symbol = self.registry.resolve("eodhd", f"{asset}-USD.CC").provider_identifiers["eodhd"]
            try:
                records = await self.loader.download_macro_series(
                    provider="eodhd",
                    series=symbol,
                    start=start,
                    end=end,
                    persist=False,
                )
            except Exception as exc:
                self.budget.record_failure("eodhd")
                self._record_error("eodhd", exc)
                continue
            successful_requests += 1
            observations.extend(
                self._api_observation(
                    "eodhd", row, classification=DataClassification.TRUE_HISTORICAL_SOURCE
                )
                for row in records
            )
            self.context_counts[f"eodhd_{asset.casefold()}_bars"] += len(records)
        if successful_requests:
            self.budget.record_request(
                "eodhd",
                "eod",
                credits=int(decision["unit_cost"]) * successful_requests,
                requests=successful_requests,
            )
        if observations:
            result = await asyncio.to_thread(self.ledgers["eodhd"].append_many, observations)
            self._account_append_result(result)
            self._record_success("eodhd")

    async def _context_loop(self) -> None:
        while not self._stop.is_set():
            for source, collector in (
                ("coinmarketcap", self._collect_cmc),
                ("eodhd", self._collect_eodhd),
            ):
                if self.storage.get("status") == "STORAGE_CRITICAL":
                    self.source_status[source]["state"] = "PAUSED_STORAGE_CRITICAL"
                    continue
                try:
                    await collector()
                except Exception as exc:
                    self.budget.record_failure(source)
                    if "429" in str(exc):
                        self.budget.record_rate_limit(source)
                    self._record_error(source, exc)
            try:
                await asyncio.wait_for(self._stop.wait(), self.context_interval_seconds)
            except TimeoutError:
                continue

    async def _collect_rss(self) -> None:
        collection = await collect_registered_feeds(
            maximum_concurrency=self.settings.scrapers.maximum_concurrency,
            timeout_seconds=self.settings.scrapers.request_timeout_seconds,
            maximum_retries=self.settings.scrapers.maximum_retries,
            raw_dir=self.settings.paths.raw_data_dir / "scrapers" / "rss_payloads",
        )
        observations: list[SourceNeutralObservation] = []
        for record in collection.records:
            ingested = utc_now()
            classification = classify_event(
                source=record.source,
                title=record.title,
                summary=record.summary,
                existing_categories=record.categories,
            )
            event = GovernedEvent(
                source_name=record.source,
                source_url=record.url,
                source_quality=SourceQuality(classification["source_quality"]),
                data_type="PUBLIC_NEWS_OR_ANNOUNCEMENT",
                published_at=record.published_at,
                first_observed_at=record.observed_at,
                ingested_at=ingested,
                parser_version="rss_normalizer_v1",
                content_hash=record.raw_hash,
                parsed_fields={
                    "title": record.title,
                    "summary": record.summary,
                    "categories": list(record.categories),
                    "language": record.language,
                    "timestamp_quality": record.timestamp_quality.value,
                    "historical_coverage": record.historical_coverage.value,
                    "event_categories": classification["event_categories"],
                    "canonical_asset_ids": classification["canonical_asset_ids"],
                    "high_value_event": classification["high_value_event"],
                },
                deduplication_key=record.event_id,
            )
            observations.append(
                SourceNeutralObservation(
                    source="scrapers",
                    source_type="GOVERNED_PUBLIC_SCRAPER",
                    canonical_asset_id=(
                        classification["canonical_asset_ids"][0]
                        if len(classification["canonical_asset_ids"]) == 1
                        else None
                    ),
                    venue_instrument_id=record.event_id,
                    data_type="EVENT_INTELLIGENCE",
                    provider_timestamp=record.published_at,
                    local_receive_timestamp=record.observed_at,
                    normalized_timestamp=record.published_at or record.observed_at,
                    persisted_timestamp=ingested,
                    raw_payload=event.to_dict(),
                    timestamp_resolution=(
                        TimestampResolution.EVENT_SECOND
                        if record.published_at
                        else TimestampResolution.RETRIEVAL_ONLY
                    ),
                    quality_state=record.deduplication_status,
                    freshness_seconds=max(
                        0.0,
                        (
                            record.observed_at - (record.published_at or record.observed_at)
                        ).total_seconds(),
                    ),
                    source_event_id=event.event_id,
                    classification=(
                        DataClassification.TRUE_HISTORICAL_SOURCE
                        if record.published_at
                        else DataClassification.PROSPECTIVE_COLLECTION
                    ),
                    metadata={
                        "full_text_stored": False,
                        "first_known_at": utc_iso(record.observed_at),
                        "event_categories": classification["event_categories"],
                        "canonical_asset_ids": classification["canonical_asset_ids"],
                        "source_quality": classification["source_quality"],
                        "high_value_event": classification["high_value_event"],
                        "social_noise_source": False,
                        "orders_generated": 0,
                    },
                )
            )
            evidence = self._event_evidence
            evidence["unique_ids"].add(event.event_id)
            evidence["mapped_assets"].update(classification["canonical_asset_ids"])
            evidence["categories"].update(classification["event_categories"])
            evidence["high_value_count"] += int(classification["high_value_event"])
            evidence["mapped_event_count"] += int(bool(classification["canonical_asset_ids"]))
            known = utc_iso(record.observed_at)
            evidence["first_known_min"] = evidence["first_known_min"] or known
            evidence["first_known_max"] = known
        if observations:
            result = await asyncio.to_thread(self.ledgers["scrapers"].append_many, observations)
            self._account_append_result(result)
            self.context_counts["governed_events"] = self.ledgers["scrapers"].record_count
            self._record_success("scrapers")
        if collection.status == "FAILED":
            raise ConnectionError("all registered RSS feeds failed")

    async def _rss_loop(self) -> None:
        if not (self.settings.scrapers.scrapers_enabled and self.settings.scrapers.rss_enabled):
            self.source_status["scrapers"]["state"] = "DISABLED_BY_CONFIG"
            return
        while not self._stop.is_set():
            if self.storage.get("status") == "STORAGE_CRITICAL":
                self.source_status["scrapers"]["state"] = "PAUSED_STORAGE_CRITICAL"
                try:
                    await asyncio.wait_for(self._stop.wait(), self.rss_interval_seconds)
                except TimeoutError:
                    continue
                continue
            try:
                await self._collect_rss()
            except Exception as exc:
                self._record_error("scrapers", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), self.rss_interval_seconds)
            except TimeoutError:
                continue

    async def _restart_failed_streams(self) -> None:
        now = utc_now()
        health = self.manager.health()
        for provider, row in health.items():
            if row.get("state") != "FAILED":
                continue
            last = self._last_restart.get(provider)
            if last and now - last < timedelta(seconds=30):
                continue
            if provider == "kraken":
                for book in self.kraken_books.values():
                    book.disconnect()
            self._last_restart[provider] = now
            await self.manager.start({provider: PUBLIC_STREAM_SUBSCRIPTIONS[provider]})

    @staticmethod
    def _duration_days(first: str | None, last: str | None) -> float:
        if not first or not last:
            return 0.0
        return max(
            0.0,
            (
                datetime.fromisoformat(last.replace("Z", "+00:00"))
                - datetime.fromisoformat(first.replace("Z", "+00:00"))
            ).total_seconds()
            / 86400,
        )

    def _flow_metrics(self, source: str, now: datetime) -> dict[str, Any]:
        rows = [
            self.trade_coverage[(source, f"CRYPTO:{asset}")].to_dict(now)
            for asset in PRIMARY_ASSETS
        ]
        observations = sum(int(row["valid_trade_count"]) for row in rows)
        failures = sum(int(row["gap_indicators"]) + int(row["out_of_order_count"]) for row in rows)
        assets = [row["canonical_asset_id"] for row in rows if row["valid_trade_count"]]
        duration = min(
            [float(row["active_duration_seconds"]) for row in rows if row["valid_trade_count"]]
            or [0.0]
        )
        quality = {"TIMESTAMP_PASS", "AGGRESSOR_SEMANTICS_DECLARED"}
        if duration >= 3 * 86400:
            quality.add("MULTIPLE_PERIODS")
        return {
            "history_days": duration / 86400,
            "observations": observations,
            "valid_fraction": observations / max(1, observations + failures),
            "gap_fraction": failures / max(1, observations + failures),
            "assets": assets,
            "quality": sorted(quality),
            "freshness_seconds": max(
                [
                    float(row["freshness_seconds"])
                    for row in rows
                    if row["freshness_seconds"] is not None
                ]
                or [0.0]
            )
            if observations
            else None,
            "per_asset": rows,
        }

    def _kraken_l2_metrics(self, now: datetime) -> dict[str, Any]:
        rows = {
            asset: self.book_coverage[("kraken", f"CRYPTO:{asset}")].to_dict(now)
            for asset in PRIMARY_ASSETS
        }
        observations = sum(int(row["book_valid_seconds"]) for row in rows.values())
        total = sum(float(row["total_collection_seconds"]) for row in rows.values())
        valid = sum(float(row["book_valid_seconds"]) for row in rows.values())
        failures = sum(sum(row["failure_counts"].values()) for row in rows.values())
        assets = [f"CRYPTO:{asset}" for asset, row in rows.items() if row["book_valid_seconds"]]
        quality = {"TIMESTAMP_PASS"}
        if len(assets) == len(PRIMARY_ASSETS):
            quality.add("BOOK_REPLAY_VALID")
        if min([row["total_collection_seconds"] for row in rows.values()] or [0]) >= 3 * 86400:
            quality.add("MULTIPLE_PERIODS")
        return {
            "history_days": min(
                [float(row["total_collection_seconds"]) for row in rows.values()] or [0]
            )
            / 86400,
            "observations": observations,
            "valid_fraction": valid / total if total else None,
            "gap_fraction": failures / max(1, observations + failures),
            "assets": assets,
            "quality": sorted(quality),
            "per_asset": rows,
            "checksum_failures": sum(
                int(book.checksum_failures) for book in self.kraken_books.values()
            ),
            "reconnect_count": sum(int(book.reconnects) for book in self.kraken_books.values()),
        }

    def _family_metrics(self, now: datetime) -> dict[str, dict[str, Any]]:
        bitvavo_flow = self._flow_metrics("bitvavo", now)
        kraken_flow = self._flow_metrics("kraken", now)
        bitvavo_l2 = bitvavo_l2_maturation(
            self.settings.paths.data_dir / "context" / "microstructure_hourly"
        )
        bitvavo_l2_metrics = {
            "history_days": bitvavo_l2["history_days"],
            "observations": bitvavo_l2["book_samples"],
            "valid_fraction": bitvavo_l2["book_valid_fraction"],
            "gap_fraction": 1 - bitvavo_l2["book_valid_fraction"],
            "assets": [
                f"CRYPTO:{market.split('-', 1)[0]}"
                for market, row in bitvavo_l2["assets"].items()
                if row["closed_intervals"]
            ],
            "quality": ["TIMESTAMP_PASS"],
            "quality_failure": bitvavo_l2["quality_failure"],
            "quality_failure_reasons": bitvavo_l2["quality_failure_reasons"],
            "maturation": bitvavo_l2,
        }
        if bitvavo_l2["book_valid_fraction"] >= 0.97:
            bitvavo_l2_metrics["quality"].append("BOOK_REPLAY_VALID")
        if bitvavo_l2["history_days"] >= 3:
            bitvavo_l2_metrics["quality"].append("MULTIPLE_PERIODS")

        alignment = self.alignment.snapshot()
        one_second = dict((alignment.get("resolutions") or {}).get("1s") or {})
        matched = int(one_second.get("matched_buckets") or 0)
        clock_valid = int(one_second.get("clock_valid_buckets") or 0)
        freshness_valid = int(one_second.get("freshness_valid_buckets") or 0)
        per_asset = dict(one_second.get("per_asset") or {})
        overlap_by_asset = [
            float((per_asset.get(f"CRYPTO:{asset}") or {}).get("trade_overlap_seconds") or 0)
            for asset in PRIMARY_ASSETS
        ]
        cross_assets = [
            f"CRYPTO:{asset}" for asset, seconds in zip(PRIMARY_ASSETS, overlap_by_asset) if seconds
        ]
        cross_quality = []
        if bitvavo_flow["observations"]:
            cross_quality.append("BITVAVO_CLOCK_PASS")
        if kraken_flow["observations"]:
            cross_quality.append("KRAKEN_CLOCK_PASS")
        if clock_valid:
            cross_quality.append("CLOCK_VALID_OVERLAP")
        cross_metrics = {
            "history_days": min(overlap_by_asset or [0]) / 86400,
            "observations": matched,
            "valid_fraction": clock_valid / matched if matched else None,
            "gap_fraction": one_second.get("gap_rate"),
            "assets": cross_assets,
            "quality": cross_quality,
            "freshness_valid_fraction": freshness_valid / matched if matched else None,
            "multi_resolution": alignment,
        }

        htf_present = all(
            (
                self.settings.paths.data_dir
                / "cache"
                / "bitvavo"
                / f"{asset}-EUR_{timeframe}_ohlcv.parquet"
            ).is_file()
            for asset in PRIMARY_ASSETS
            for timeframe in ("1h", "4h")
        )
        flow_swing_quality = []
        if htf_present:
            flow_swing_quality.append("HTF_CANDLES_PRESENT")
        if bitvavo_flow["valid_fraction"] >= 0.97:
            flow_swing_quality.append("BITVAVO_FLOW_VALID")
        if bitvavo_flow["history_days"] >= 3:
            flow_swing_quality.append("MULTIPLE_PERIODS")
        flow_swing = {
            **{key: value for key, value in bitvavo_flow.items() if key != "quality"},
            "quality": flow_swing_quality,
            "htf_candles_present": htf_present,
        }

        cmc_snapshots = int(self.context_counts.get("cmc_breadth_snapshots") or 0)
        cmc_history = self._duration_days(
            self.ledgers["coinmarketcap"].first_known_at,
            self.ledgers["coinmarketcap"].last_known_at,
        )
        cmc_assets = int((self.latest_cmc_breadth or {}).get("eligible_asset_count") or 0)
        cmc_quality = []
        if self.latest_cmc_breadth:
            cmc_quality.extend(["PIT_UNIVERSE", "NO_FUTURE_MEMBERSHIP"])
        if (self.latest_cmc_breadth or {}).get("definitions_version"):
            cmc_quality.append("CONSISTENT_DEFINITION")

        evidence = self._event_evidence
        event_count = len(evidence["unique_ids"])
        mapped_fraction = int(evidence["mapped_event_count"]) / max(1, event_count)
        event_quality = ["FIRST_KNOWN_TIME", "SOURCE_QUALITY"] if event_count else []
        if evidence["mapped_assets"]:
            event_quality.append("ASSET_MAPPING")
        if len(evidence["categories"]) >= 4:
            event_quality.append("CATEGORY_DIVERSITY")
        event_history = self._duration_days(
            evidence["first_known_min"], evidence["first_known_max"]
        )
        derivatives_metrics = mexc_derivatives_maturation(self.settings.paths.data_dir / "context")

        liquidity = {
            **bitvavo_l2_metrics,
            "quality": [
                value
                for value, present in (
                    ("BOOK_REPLAY_VALID", "BOOK_REPLAY_VALID" in bitvavo_l2_metrics["quality"]),
                    ("SPREAD_DEPTH_PRESENT", bitvavo_l2["book_samples"] > 0),
                    ("FLOW_VALID", bitvavo_flow["valid_fraction"] >= 0.97),
                )
                if present
            ],
        }
        return {
            "BITVAVO_FLOW": bitvavo_flow,
            "BITVAVO_L2": bitvavo_l2_metrics,
            "CROSS_VENUE_LEAD_LAG": cross_metrics,
            "MULTI_VENUE_DISLOCATION": {
                **cross_metrics,
                "quality": ["TIMESTAMP_PASS"] if clock_valid else [],
            },
            "FLOW_CONFIRMED_SWING": flow_swing,
            "LIQUIDITY_SHOCK": liquidity,
            "CMC_BREADTH": {
                "history_days": cmc_history,
                "observations": cmc_snapshots,
                "valid_fraction": 1.0 if cmc_snapshots else None,
                "gap_fraction": 0.0 if cmc_snapshots else None,
                "assets": [],
                "quality": cmc_quality,
                "eligible_asset_count": cmc_assets,
                "latest": self.latest_cmc_breadth,
            },
            "BTC_MARKET_REGIME": {
                "history_days": min(cmc_history, bitvavo_flow["history_days"]),
                "observations": cmc_snapshots,
                "valid_fraction": 1.0 if cmc_snapshots and htf_present else None,
                "gap_fraction": 0.0 if cmc_snapshots and htf_present else None,
                "assets": ["CRYPTO:BTC"] if bitvavo_flow["observations"] else [],
                "quality": [
                    *(["PIT_MARKET_CONTEXT"] if cmc_snapshots else []),
                    *(["BTC_PRICE_PRESENT"] if htf_present else []),
                    *(["MULTIPLE_PERIODS"] if cmc_history >= 14 else []),
                ],
            },
            "MEXC_DERIVATIVES_CONTEXT": derivatives_metrics,
            "EVENT_INTELLIGENCE": {
                "history_days": event_history,
                "observations": int(evidence["high_value_count"]),
                "valid_fraction": mapped_fraction if event_count else None,
                "gap_fraction": None,
                "assets": sorted(evidence["mapped_assets"]),
                "quality": event_quality,
                "unique_event_count": event_count,
                "high_value_event_count": int(evidence["high_value_count"]),
                "event_categories": sorted(evidence["categories"]),
                "social_noise_expansion": False,
            },
        }

    def _notify_operational(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            notifier = self.notifier
            if notifier is None:
                from notifications.telegram import TelegramNotifier

                notifier = TelegramNotifier(
                    self.settings.telegram,
                    output_directory=self.settings.paths.output_dir / "notifications",
                    allowed_markets=("BTC-EUR", "ETH-EUR", "SOL-EUR"),
                )
                self.notifier = notifier
            return dict(notifier.notify_system_event(event_type, payload))
        except Exception as exc:
            return {
                "delivery_status": "FAILED_ISOLATED",
                "reason_code": type(exc).__name__,
                "orders_generated": 0,
            }

    def _readiness(self) -> dict[str, Any]:
        now = utc_now()
        metrics = self._family_metrics(now)
        assessments = {
            family: assess_readiness(self.readiness_policy[family], values)
            for family, values in metrics.items()
        }
        transitions: dict[str, Any] = {}
        freezes: dict[str, Any] = {}
        notifications: dict[str, Any] = {}
        feature_names = {
            "BITVAVO_FLOW": ("CVD", "TRADE_INTENSITY", "FLOW_GAPS"),
            "BITVAVO_L2": ("SPREAD", "DEPTH", "IMBALANCE", "MICROPRICE"),
            "CROSS_VENUE_LEAD_LAG": ("CLOCK_VALID_FLOW_ALIGNMENT",),
            "MULTI_VENUE_DISLOCATION": ("NORMALIZED_REFERENCE_PRICE",),
            "FLOW_CONFIRMED_SWING": ("HTF_SETUP", "BITVAVO_FLOW"),
            "LIQUIDITY_SHOCK": ("SPREAD", "DEPTH", "FLOW"),
            "CMC_BREADTH": tuple(
                key
                for key, value in (self.latest_cmc_breadth or {}).items()
                if isinstance(value, int | float)
            ),
            "BTC_MARKET_REGIME": ("BTC_PRICE", "MARKET_BREADTH"),
            "MEXC_DERIVATIVES_CONTEXT": ("FUNDING", "OPEN_INTEREST", "BASIS"),
            "EVENT_INTELLIGENCE": ("CATEGORY", "ASSET_MAPPING", "FIRST_KNOWN_TIME"),
        }
        source_map = {
            "BITVAVO_FLOW": ("bitvavo",),
            "BITVAVO_L2": ("bitvavo",),
            "CROSS_VENUE_LEAD_LAG": ("bitvavo", "kraken"),
            "MULTI_VENUE_DISLOCATION": ("bitvavo", "kraken", "mexc_spot"),
            "FLOW_CONFIRMED_SWING": ("bitvavo",),
            "LIQUIDITY_SHOCK": ("bitvavo",),
            "CMC_BREADTH": ("coinmarketcap",),
            "BTC_MARKET_REGIME": ("bitvavo", "coinmarketcap"),
            "MEXC_DERIVATIVES_CONTEXT": (),
            "EVENT_INTELLIGENCE": ("scrapers",),
        }
        for family, assessment in assessments.items():
            transition_result = self.readiness_history.record(assessment, at=now)
            transitions[family] = transition_result
            transition = (
                transition_result.get("transition")
                if transition_result.get("status") == "TRANSITION_RECORDED"
                else None
            )
            source_names = source_map[family]
            starts = [
                self.ledgers[name].first_known_at
                for name in source_names
                if self.ledgers[name].first_known_at
            ]
            ends = [
                self.ledgers[name].last_known_at
                for name in source_names
                if self.ledgers[name].last_known_at
            ]
            if starts and ends:
                freeze = self.freeze_manager.maybe_freeze(
                    assessment=assessment,
                    transition=transition,
                    source_manifests=[self.ledgers[name].checkpoint() for name in source_names],
                    assets=assessment["metrics"].get("assets") or (),
                    features=feature_names[family],
                    collection_start=min(
                        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in starts
                    ),
                    data_end=max(
                        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in ends
                    ),
                    coverage=assessment["metrics"],
                    clock_metrics=(assessment["metrics"].get("multi_resolution") or {}),
                    build_commit=os.getenv("GIT_COMMIT"),
                )
            else:
                freeze = {"status": "NOT_ELIGIBLE", "family": family}
            freezes[family] = freeze
            if freeze.get("status") == "FREEZE_CREATED":
                notifications[family] = self._notify_operational(
                    "DATASET_READY_FOR_RESEARCH",
                    {
                        "status": "DATASET READY FOR RESEARCH",
                        "reason_code": family,
                        "candidate_id": (freeze.get("freeze") or {}).get("dataset_id"),
                    },
                )
        hypotheses = hypothesis_readiness(
            assessments,
            frozen_families=self.freeze_manager.frozen_families(),
            spot_candidate_available=False,
        )
        return {
            "schema_version": "p1_2_2_independent_family_readiness_v1",
            "policy_version": "research_readiness_policy_v1",
            "policy": {name: row.to_dict() for name, row in self.readiness_policy.items()},
            "families": assessments,
            "family_states": {name: row["state"] for name, row in assessments.items()},
            "hypotheses": hypotheses,
            "transitions": transitions,
            "freeze_candidates": freezes,
            "notifications": notifications,
            "automatic_alpha_started": False,
            "automatic_stage0_started": False,
            "automatic_backtest_started": False,
            "automatic_ml_training_started": False,
            "automatic_strategy_promotion": False,
            "ml_authority": "SHADOW_ONLY",
            "next_expected_milestone": "FIRST_FAMILY_RESEARCH_USABLE",
        }

    def _cross_venue_overlap(self) -> dict[str, Any]:
        per_asset: dict[str, Any] = {}
        starts: list[datetime] = []
        ends: list[datetime] = []
        total_seconds = 0.0
        for asset in PRIMARY_ASSETS:
            bitvavo = self.trade_coverage[("bitvavo", f"CRYPTO:{asset}")]
            kraken = self.trade_coverage[("kraken", f"CRYPTO:{asset}")]
            overlap = interval_overlap(
                bitvavo.first_known_at,
                bitvavo.last_known_at,
                kraken.first_known_at,
                kraken.last_known_at,
            )
            per_asset[asset] = overlap
            if overlap["start"]:
                starts.append(datetime.fromisoformat(overlap["start"]))
                ends.append(datetime.fromisoformat(overlap["end"]))
                total_seconds += float(overlap["seconds"])
        aggregate = {
            "start": utc_iso(min(starts)) if starts else None,
            "end": utc_iso(max(ends)) if ends else None,
            "seconds": total_seconds,
            "minutes": total_seconds / 60,
            "hours": total_seconds / 3600,
            "days": total_seconds / 86400,
            "status": "OVERLAP_PRESENT" if total_seconds > 0 else "NO_OVERLAP_EVIDENCE",
            "aggregation": "SUM_OF_PER_ASSET_SIMULTANEOUS_INTERVALS",
        }
        return {"per_asset": per_asset, "aggregate": aggregate}

    async def _compact_closed_segments(self, now: datetime) -> dict[str, Any]:
        started = asyncio.get_running_loop().time()
        results: dict[str, Any] = {}
        for source, ledger in self.ledgers.items():
            try:
                result = await asyncio.to_thread(
                    compact_source_ledger,
                    ledger.root,
                    self.settings.paths.data_dir / "compacted" / "multi_source",
                    source,
                    closed_before=now,
                )
            except Exception as exc:
                result = {
                    "status": "FAILED_ISOLATED",
                    "source": source,
                    "reason_code": type(exc).__name__,
                    "orders_generated": 0,
                }
            results[source] = result
            if result.get("status") == "PASSED":
                key = f"{source}|ALL_ASSETS|ALL_TYPES"
                cumulative = self.storage_cumulative.setdefault(
                    key,
                    {
                        "events": 0,
                        "raw_bytes": 0,
                        "compressed_events": 0,
                        "compressed_bytes": 0,
                    },
                )
                cumulative["compressed_events"] = int(result.get("row_count") or 0)
                cumulative["compressed_bytes"] = int(result.get("parquet_bytes") or 0)
        elapsed = max(1e-9, asyncio.get_running_loop().time() - started)
        return {
            "schema_version": "multi_source_compaction_runtime_v1",
            "observed_at": utc_iso(now),
            "elapsed_seconds": elapsed,
            "rows_per_second": sum(int(row.get("row_count") or 0) for row in results.values())
            / elapsed,
            "closed_hours_only": True,
            "raw_deleted": False,
            "sources": results,
        }

    async def _maintenance_loop(self) -> None:
        while not self._stop.is_set():
            now = utc_now()
            try:
                self.alignment.persist()
                self.storage = await asyncio.to_thread(
                    self.storage_monitor.observe,
                    self.storage_cumulative,
                    disk_path=self.settings.paths.data_dir,
                    at=now,
                )
                if (
                    self.storage.get("status") == "STORAGE_CRITICAL"
                    and not self._storage_critical_notified
                ):
                    self._notify_operational(
                        "STORAGE_CRITICAL",
                        {
                            "status": "STORAGE_CRITICAL",
                            "reason_code": "OPTIONAL_RESEARCH_SOURCES_PAUSED",
                        },
                    )
                    self._storage_critical_notified = True
                if self.storage.get("status") != "STORAGE_CRITICAL":
                    self._storage_critical_notified = False
                if self._last_compaction_at is None or now - self._last_compaction_at >= timedelta(
                    hours=1
                ):
                    self.compaction = await self._compact_closed_segments(now)
                    self._last_compaction_at = now
            except Exception as exc:
                self.compaction = {
                    "status": "MAINTENANCE_FAILURE_ISOLATED",
                    "reason_code": type(exc).__name__,
                    "orders_generated": 0,
                }
            try:
                await asyncio.wait_for(self._stop.wait(), 60.0)
            except TimeoutError:
                continue

    def snapshot(self) -> dict[str, Any]:
        now = utc_now()
        stream_health = self.manager.health()
        for provider, row in stream_health.items():
            source = _source_name(provider)
            if row.get("state") in {"CONNECTED", "STALE", "FAILED", "RECONNECTING"}:
                self.source_status[source]["state"] = row["state"]
        checkpoints = {source: ledger.checkpoint() for source, ledger in self.ledgers.items()}
        readiness = self._readiness()
        session_events = max(
            0,
            sum(ledger.record_count for ledger in self.ledgers.values())
            - self._performance_event_baseline,
        )
        api_usage = api_usage_report(self.budget.status(), observed_at=now)
        self.performance = self.performance_monitor.snapshot(
            total_events=session_events,
            queue_size=self.manager.queue.qsize(),
            queue_capacity=self.manager.queue.maxsize,
            compaction=self.compaction,
            api_scheduler=api_usage,
        )
        ownership = (
            dict(read_json(self.lease.path))
            if self.lease.acquired and self.lease.path.is_file()
            else {"status": "NOT_ACQUIRED", "instance_id": self.lease.instance_id}
        )
        hypothesis_rows = dict(readiness.get("hypotheses") or {})
        if (hypothesis_rows.get("H1_FLOW_CONFIRMED_SWING_READY") or {}).get("ready"):
            next_action = "P1.3_FLOW_CONFIRMED_SWING_RESEARCH"
        elif any(
            (hypothesis_rows.get(name) or {}).get("ready")
            for name in (
                "H3_CROSS_VENUE_LEAD_LAG_READY",
                "H4_CROSS_VENUE_FLOW_READY",
                "H5_LIQUIDITY_SHOCK_READY",
            )
        ):
            next_action = "P1.3_CROSS_VENUE_MARKET_STRUCTURE_RESEARCH"
        elif (hypothesis_rows.get("H6_BREADTH_MOMENTUM_READY") or {}).get("ready"):
            next_action = "P1.3_BREADTH_CONDITIONED_RESEARCH"
        else:
            next_action = "CONTINUE_PROSPECTIVE_COLLECTION"
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "platform_schema_version": MULTI_SOURCE_SCHEMA_VERSION,
            "observed_at": utc_iso(now),
            "started_at": utc_iso(self.started_at),
            "collection_uptime_seconds": max(0.0, (now - self.started_at).total_seconds()),
            "process_started_at": utc_iso(self.process_started_at),
            "process_uptime_seconds": max(0.0, (now - self.process_started_at).total_seconds()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "mode": "PUBLIC_READ_ONLY_COLLECTION",
            "ownership": ownership,
            "kraken_collection_epoch": self._epoch,
            "source_status": self.source_status,
            "stream_health": stream_health,
            "trade_coverage": {
                f"{source}:{asset}": row.to_dict(now)
                for (source, asset), row in self.trade_coverage.items()
            },
            "book_coverage": {
                f"{source}:{asset}": row.to_dict(now)
                for (source, asset), row in self.book_coverage.items()
            },
            "kraken_books": {
                symbol: book.snapshot(now) for symbol, book in self.kraken_books.items()
            },
            "bitvavo_l2_v2": {
                "schema_version": L2_RECONSTRUCTION_VERSION,
                "books": {
                    market: book.snapshot(now) for market, book in self.bitvavo_books.items()
                },
                "features": {
                    market: book.features(now) for market, book in self.bitvavo_books.items()
                },
                "reseed_attempts": dict(self._bitvavo_reseed_attempts),
                "buffer_overflows": dict(self._bitvavo_buffer_overflows),
                "execution_authority": False,
                "orders_generated": 0,
            },
            "cross_venue_overlap": self._cross_venue_overlap(),
            "cross_venue_multi_resolution": self.alignment.snapshot(),
            "context_counts": dict(self.context_counts),
            "latest_cmc_breadth": self.latest_cmc_breadth,
            "latest_cmc_global": self.latest_cmc_global,
            "api_budget": self.budget.status(),
            "api_usage": api_usage,
            "api_budget_schema": API_BUDGET_SCHEMA_VERSION,
            "ledger_checkpoints": checkpoints,
            "readiness": readiness,
            "storage_cumulative": self.storage_cumulative,
            "storage_measurement_started_at": self.storage_measurement_started_at,
            "storage": self.storage,
            "compaction": self.compaction,
            "performance": self.performance,
            "next_exact_action": next_action,
            "authority": {name: policy.to_dict() for name, policy in self.policies.items()},
            "asset_registry_hash": self.registry.registry_hash,
            "execution": {
                "sole_execution_authority": "bitvavo",
                "new_exchange_mutations": 0,
                "orders_generated": 0,
                "orders_submitted": 0,
                "private_kraken_requests": 0,
                "private_mexc_requests": 0,
                "live_authority_increased": False,
                "execution_health_dependency": False,
            },
        }

    async def _status_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._restart_failed_streams()
                self.lease.heartbeat()
                payload = self.snapshot()
                atomic_write_json(self.status_path, payload)
                atomic_write_json(
                    self.heartbeat_path,
                    {
                        "status": "RUNNING",
                        "observed_at": payload["observed_at"],
                        "pid": os.getpid(),
                        "orders_generated": 0,
                    },
                )
            except Exception as exc:
                self.source_status["bitvavo"]["last_runtime_error"] = (
                    f"{type(exc).__name__}:REDACTED_RUNTIME_FAILURE"
                )
                if "COLLECTOR_LEASE" in str(exc):
                    self._stop.set()
            try:
                await asyncio.wait_for(self._stop.wait(), self.status_interval_seconds)
            except TimeoutError:
                continue

    async def run(self, *, duration_seconds: float | None = None) -> dict[str, Any]:
        if not self.lease.acquired:
            self.lease.acquire()
        tasks: list[asyncio.Task[Any]] = []
        try:
            if self.stop_path.exists():
                self.stop_path.unlink()
            tasks = [
                asyncio.create_task(self._stream_loop(), name="multi-source-streams"),
                asyncio.create_task(self._context_loop(), name="multi-source-context"),
                asyncio.create_task(self._rss_loop(), name="multi-source-rss"),
                asyncio.create_task(self._status_loop(), name="multi-source-status"),
                asyncio.create_task(
                    self._bitvavo_reseed_loop(),
                    name="multi-source-bitvavo-l2-v2-reseed",
                ),
                asyncio.create_task(self._maintenance_loop(), name="multi-source-maintenance"),
            ]
            try:
                if duration_seconds is None:
                    while not self._stop.is_set():
                        if self.stop_path.exists():
                            self._stop.set()
                            break
                        await asyncio.sleep(1)
                else:
                    try:
                        await asyncio.wait_for(self._stop.wait(), duration_seconds)
                    except TimeoutError:
                        self._stop.set()
            finally:
                self._stop.set()
                await self.manager.stop()
                await asyncio.gather(*tasks, return_exceptions=True)
                await self._flush()
                payload = self.snapshot()
                payload["runtime_status"] = "STOPPED"
                atomic_write_json(self.status_path, payload)
                atomic_write_json(
                    self.heartbeat_path,
                    {
                        "status": "STOPPED",
                        "observed_at": payload["observed_at"],
                        "pid": os.getpid(),
                        "orders_generated": 0,
                    },
                )
            return payload
        finally:
            self.lease.release()


async def run_multi_source_collector(
    settings: Settings,
    *,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    output_root = settings.paths.output_dir / "multi_source"
    lease = CollectorLease(output_root / "collector.lock", output_root / "ownership_history")
    lease.acquire()
    try:
        collector = MultiSourceCollector(settings, lease=lease)
        return await collector.run(duration_seconds=duration_seconds)
    except BaseException:
        lease.release()
        raise


__all__ = [
    "MultiSourceCollector",
    "PUBLIC_STREAM_SUBSCRIPTIONS",
    "RUNTIME_SCHEMA_VERSION",
    "run_multi_source_collector",
]
