from __future__ import annotations

import inspect
import zlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import SecretStr

from data.data_loader import DataLoader
from data.market_structure import BookQuality, ClockQuality
from data.multi_source_platform import (
    ApiBudgetLedger,
    ApiBudgetRule,
    CanonicalAssetIdentity,
    CanonicalAssetRegistry,
    DataClassification,
    DataFamily,
    DatasetFreezeManager,
    FamilyReadiness,
    GovernedEvent,
    ImmutableSourceLedger,
    KrakenL2Book,
    PointInTimeFeatureStore,
    SourceNeutralObservation,
    SourceQuality,
    TimestampResolution,
    assess_family_readiness,
    compact_source_ledger,
    compare_ohlcv_sources,
    compute_cmc_breadth,
    default_readiness_thresholds,
    hypothesis_specific_readiness,
    initial_multi_source_asset_registry,
    mexc_semantic_source,
    normalize_quote_asset,
    source_authority_registry,
    verify_source_ledger,
)
from data.multi_source_runtime import MultiSourceCollector
from data.websocket_manager import WebSocketManager


def _observation(source: str, index: int) -> SourceNeutralObservation:
    event = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
    receive = event + timedelta(milliseconds=20)
    return SourceNeutralObservation(
        source=source,
        source_type="PUBLIC_WEBSOCKET",
        venue=source,
        canonical_asset_id="CRYPTO:BTC",
        venue_instrument_id="BTC/EUR",
        data_type="TRADE",
        exchange_event_timestamp=event,
        local_receive_timestamp=receive,
        normalized_timestamp=event,
        persisted_timestamp=receive + timedelta(milliseconds=1),
        raw_payload={"trade_id": index, "price": "100", "qty": "1"},
        timestamp_resolution=TimestampResolution.EVENT_EXACT,
        source_event_id=f"BTC/EUR:{index}",
        classification=DataClassification.PROSPECTIVE_COLLECTION,
    )


def test_authority_matrix_is_explicit_and_bitvavo_only_execution() -> None:
    registry = source_authority_registry()
    assert set(registry) == {
        "bitvavo",
        "kraken",
        "mexc_spot",
        "mexc_derivatives",
        "coinmarketcap",
        "eodhd",
        "scrapers",
    }
    assert [name for name, row in registry.items() if row.execution_allowed] == ["bitvavo"]
    for name in registry.keys() - {"bitvavo"}:
        assert "ORDER_SUBMISSION" in registry[name].forbidden_influences


def test_asset_identity_collision_and_stablecoin_semantics() -> None:
    registry = initial_multi_source_asset_registry()
    assert registry.resolve("kraken", "BTC/EUR").canonical_asset_id == "CRYPTO:BTC"
    assert registry.resolve("mexc_spot", "BTCUSDT").canonical_asset_id == "CRYPTO:BTC"
    usdt = normalize_quote_asset("mexc_spot", "USDT")
    assert usdt["canonical_quote_asset_id"] == "CRYPTO:USDT"
    assert usdt["fiat_equivalence_assumed"] is False
    assert normalize_quote_asset("bitvavo", "EUR")["canonical_quote_asset_id"] == "FIAT:EUR"
    with pytest.raises(KeyError):
        normalize_quote_asset("unit", "UNKNOWN")
    duplicate = CanonicalAssetIdentity(
        "CRYPTO:OTHER",
        "OTHER",
        "Other",
        999,
        {"kraken": "BTC/EUR"},
    )
    with pytest.raises(ValueError, match="collision"):
        CanonicalAssetRegistry([registry.identities[0], duplicate])


def test_mexc_spot_and_derivatives_cannot_silently_merge() -> None:
    policies = source_authority_registry()
    assert mexc_semantic_source("spot") == "mexc_spot"
    assert mexc_semantic_source("perpetual") == "mexc_derivatives"
    assert policies["mexc_spot"].authorities != policies["mexc_derivatives"].authorities
    assert "FUNDING_CONTEXT" not in policies["mexc_spot"].may_influence
    assert "REFERENCE_PRICE" not in policies["mexc_derivatives"].may_influence
    with pytest.raises(ValueError):
        mexc_semantic_source("unknown")


def test_observation_requires_timezone_awareness_and_preserves_known_at() -> None:
    row = _observation("kraken", 1)
    assert row.known_at == datetime(2026, 1, 1, 0, 0, 1, 20_000, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        SourceNeutralObservation(
            source="kraken",
            source_type="PUBLIC_WEBSOCKET",
            canonical_asset_id="CRYPTO:BTC",
            data_type="TRADE",
            exchange_event_timestamp=datetime(2026, 1, 1),
            local_receive_timestamp=datetime(2026, 1, 1),
            normalized_timestamp=datetime(2026, 1, 1),
            persisted_timestamp=datetime(2026, 1, 1),
            raw_payload={},
        )


def test_immutable_ledger_batch_dedup_recovery_and_integrity(tmp_path) -> None:
    root = tmp_path / "raw"
    checkpoint = tmp_path / "checkpoint.json"
    ledger = ImmutableSourceLedger(root, "kraken", checkpoint)
    result = ledger.append_many([_observation("kraken", 1), _observation("kraken", 2)])
    assert result["appended"] == 2
    assert ledger.append_many([_observation("kraken", 2)])["status"] == "DUPLICATES_ONLY"
    audit = verify_source_ledger(root, "kraken")
    assert audit["status"] == "PASSED"
    segment = next(root.rglob("events.jsonl"))
    with segment.open("ab") as stream:
        stream.write(b'{"partial":')
    recovered = ImmutableSourceLedger(root, "kraken", checkpoint)
    assert recovered.record_count == 2
    assert verify_source_ledger(root, "kraken")["status"] == "PASSED"
    compacted = compact_source_ledger(root, tmp_path / "compacted", "kraken")
    assert compacted["status"] == "PASSED"
    assert compacted["row_count"] == 2
    assert compacted["raw_immutable_and_preserved"] is True
    assert segment.is_file()


def test_kraken_crc32_book_is_fail_closed() -> None:
    book = KrakenL2Book("BTC/EUR", depth=100)
    bids = [{"price": Decimal("100.0"), "qty": Decimal("3.00")}, {"price": "99", "qty": "4"}]
    asks = [{"price": "101.0", "qty": "1.00"}, {"price": "102", "qty": "2"}]
    checksum_text = "101010010221000300994"
    checksum = zlib.crc32(checksum_text.encode()) & 0xFFFFFFFF
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    applied = book.apply(
        kind="snapshot",
        bids=bids,
        asks=asks,
        checksum=checksum,
        event_timestamp=timestamp,
        receive_timestamp=timestamp + timedelta(milliseconds=10),
        message_id="snapshot-1",
    )
    assert applied["status"] == "APPLIED"
    assert book.state is BookQuality.BOOK_VALID
    assert (
        book.apply(
            kind="snapshot",
            bids=bids,
            asks=asks,
            checksum=checksum,
            event_timestamp=timestamp,
            receive_timestamp=timestamp,
            message_id="snapshot-1",
        )["status"]
        == "DUPLICATE_REJECTED"
    )
    failed = book.apply(
        kind="update",
        bids=[{"price": "100.0", "qty": "5"}],
        asks=[],
        checksum=1,
        event_timestamp=timestamp + timedelta(seconds=1),
        receive_timestamp=timestamp + timedelta(seconds=1, milliseconds=10),
        message_id="update-bad",
    )
    assert failed["status"] == "CHECKSUM_FAILURE_FAIL_CLOSED"
    assert book.state is BookQuality.BOOK_GAPPED
    assert book.snapshot()["best_bid"] is None


def test_kraken_parser_preserves_taker_semantics_and_book_precision() -> None:
    manager = WebSocketManager()
    trade = manager.parse_message(
        "kraken",
        {
            "channel": "trade",
            "type": "update",
            "data": [
                {
                    "symbol": "BTC/EUR",
                    "trade_id": 7,
                    "price": Decimal("60000.0100"),
                    "qty": Decimal("0.00100"),
                    "side": "buy",
                    "timestamp": "2026-01-01T00:00:00.000000Z",
                }
            ],
        },
    )[0]
    assert trade.message_id == "BTC/EUR:7"
    assert trade.payload["aggressor_semantics"] == "EXCHANGE_REPORTED_TAKER_SIDE"
    assert isinstance(trade.payload["provider_payload"]["price"], Decimal)


def test_cmc_breadth_and_source_disagreement_are_observable() -> None:
    rows = [
        {
            "cmc_rank": 1,
            "market_cap": 100,
            "percent_change_24h": 5,
            "percent_change_7d": 3,
            "volume_24h": 10,
        },
        {
            "cmc_rank": 2,
            "market_cap": 50,
            "percent_change_24h": -2,
            "percent_change_7d": 1,
            "volume_24h": 5,
        },
    ]
    breadth = compute_cmc_breadth(rows, known_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert breadth["positive_24h_fraction"] == 0.5
    assert breadth["eligible_asset_count"] == 2
    index = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    disagreement = compare_ohlcv_sources(
        {
            "bitvavo": pd.DataFrame({"close": [100, 101]}, index=index),
            "eodhd": pd.DataFrame({"close": [100, 120]}, index=index),
        },
        relative_tolerance=0.03,
    )[0]
    assert disagreement["status"] == "SOURCE_DISAGREEMENT"
    assert disagreement["arbitrary_source_override"] is False


@pytest.mark.asyncio
async def test_cmc_parsing_includes_pit_breadth_fields_and_credit_count(isolated_settings) -> None:
    providers = isolated_settings.providers.model_copy(
        update={"coinmarketcap_api_key": SecretStr("unit")}
    )
    settings = isolated_settings.model_copy(update={"providers": providers})

    async def request(method, url, params, headers):
        del method, url, params, headers
        return {
            "status": {"timestamp": "2026-01-01T00:00:01Z", "credit_count": 3},
            "data": [
                {
                    "id": 1,
                    "cmc_rank": 1,
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "slug": "bitcoin",
                    "last_updated": "2026-01-01T00:00:00Z",
                    "quote": {
                        "EUR": {
                            "price": 100,
                            "market_cap": 1000,
                            "volume_24h": 50,
                            "percent_change_1h": 1,
                            "percent_change_24h": 2,
                            "percent_change_7d": 3,
                            "market_cap_dominance": 55,
                        }
                    },
                }
            ],
        }

    record = (await DataLoader(settings, requester=request).download_cmc_rankings(limit=1))[0]
    assert record.available_at == record.observed_at
    assert record.values["percent_change_24h"] == 2
    assert record.values["market_cap_dominance"] == 55
    assert record.values["response_credit_count"] == 3


@pytest.mark.asyncio
async def test_eodhd_history_is_forward_only_validation_not_microstructure(
    isolated_settings,
) -> None:
    providers = isolated_settings.providers.model_copy(update={"eodhd_api_key": SecretStr("unit")})
    settings = isolated_settings.model_copy(update={"providers": providers})

    async def request(method, url, params, headers):
        del method, url, params, headers
        return [
            {"date": "2026-01-01", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}
        ]

    record = (
        await DataLoader(settings, requester=request).download_macro_series(
            provider="eodhd", series="BTC-USD.CC"
        )
    )[0]
    assert record.source_symbol == "BTC-USD.CC"
    assert record.values["point_in_time_status"] == "FORWARD_ONLY"
    assert "REFERENCE_L2" not in source_authority_registry()["eodhd"].may_influence


def test_governed_event_first_known_and_dedup_contract() -> None:
    first = datetime(2026, 1, 2, tzinfo=UTC)
    event = GovernedEvent(
        source_name="Kraken",
        source_url="https://example.test/event",
        source_quality=SourceQuality.PRIMARY_OFFICIAL,
        data_type="EXCHANGE_ANNOUNCEMENT",
        published_at=first - timedelta(minutes=5),
        first_observed_at=first,
        ingested_at=first + timedelta(seconds=1),
        parser_version="unit_v1",
        content_hash="a" * 64,
        parsed_fields={"title": "Maintenance"},
        deduplication_key="kraken-maintenance-1",
    )
    duplicate = GovernedEvent(
        source_name="Kraken",
        source_url="https://example.test/changed-url",
        source_quality=SourceQuality.PRIMARY_OFFICIAL,
        data_type="EXCHANGE_ANNOUNCEMENT",
        published_at=first - timedelta(minutes=5),
        first_observed_at=first,
        ingested_at=first + timedelta(seconds=2),
        parser_version="unit_v1",
        content_hash="b" * 64,
        parsed_fields={"title": "Maintenance updated"},
        deduplication_key="kraken-maintenance-1",
    )
    assert event.event_id == duplicate.event_id
    assert event.to_dict()["full_text_stored"] is False
    with pytest.raises(ValueError, match="future publication"):
        GovernedEvent(
            source_name="unit",
            source_url="https://example.test",
            source_quality=SourceQuality.UNVERIFIED,
            data_type="EVENT",
            published_at=first + timedelta(seconds=1),
            first_observed_at=first,
            ingested_at=first,
            parser_version="unit",
            content_hash="c" * 64,
            parsed_fields={},
        )


def test_api_budget_quota_and_interval_gates(tmp_path) -> None:
    ledger = ApiBudgetLedger(
        tmp_path / "budget.json",
        [ApiBudgetRule("cmc", 1, 2, 1, {"rankings": 1})],
    )
    at = datetime(2026, 1, 1, tzinfo=UTC)
    assert ledger.authorize("cmc", "rankings", at=at)["allowed"]
    ledger.record_request("cmc", "rankings", credits=1, at=at)
    assert ledger.authorize("cmc", "rankings", at=at)["reason"] == "CACHE_OR_INTERVAL_REQUIRED"
    assert (
        ledger.authorize("cmc", "rankings", at=at + timedelta(seconds=2))["reason"]
        == "DAILY_CREDIT_LIMIT"
    )
    assert ledger.status()["orders_generated"] == 0


def test_point_in_time_lookup_is_invariant_to_future_append(tmp_path) -> None:
    store = PointInTimeFeatureStore(tmp_path / "features")
    first = datetime(2026, 1, 1, tzinfo=UTC)
    store.append(
        [
            {
                "canonical_asset_id": "CRYPTO:BTC",
                "feature": "cmc_rank",
                "value": 1,
                "source_timestamp": first,
                "known_at": first + timedelta(minutes=1),
                "source": "coinmarketcap",
                "quality": "PRESENT",
            }
        ],
        family=DataFamily.UNIVERSE,
    )
    cutoff = first + timedelta(hours=1)
    before = store.as_of("CRYPTO:BTC", cutoff, families=[DataFamily.UNIVERSE])
    store.append(
        [
            {
                "canonical_asset_id": "CRYPTO:BTC",
                "feature": "cmc_rank",
                "value": 2,
                "source_timestamp": first + timedelta(days=1),
                "known_at": first + timedelta(days=1, minutes=1),
                "source": "coinmarketcap",
                "quality": "PRESENT",
            }
        ],
        family=DataFamily.UNIVERSE,
    )
    after = store.as_of("CRYPTO:BTC", cutoff, families=[DataFamily.UNIVERSE])
    assert before[0]["value"] == after[0]["value"] == 1


def test_readiness_hypotheses_freeze_and_holdout_are_non_promotional(tmp_path) -> None:
    threshold = default_readiness_thresholds()["TRADE_FLOW"]
    ready = assess_family_readiness(
        threshold,
        duration_days=30,
        valid_observations=200_000,
        overlap_days=0,
        gap_rate=0,
        book_valid_fraction=None,
        freshness_seconds=1,
        asset_count=3,
        clock_quality=ClockQuality.CLOCK_OK,
    )
    assert ready["state"] == FamilyReadiness.RESEARCH_USABLE.value
    assert ready["automatic_alpha_start"] is False
    hypotheses = hypothesis_specific_readiness(
        {
            "CMC_BREADTH": "RESEARCH_USABLE",
            "CMC_UNIVERSE": "RESEARCH_USABLE",
            "BITVAVO_PRICE": "RESEARCH_USABLE",
        }
    )
    assert hypotheses["BREADTH_CONDITIONED_ALPHA"]["ready"]
    assert not hypotheses["CROSS_VENUE_LEAD_LAG"]["ready"]
    freezer = DatasetFreezeManager(tmp_path / "freezes")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frozen = freezer.freeze(
        family=DataFamily.MARKET_BREADTH,
        collection_epoch=start,
        data_end=start + timedelta(days=60),
        source_manifests=[{"source": "coinmarketcap", "root_hash": "a" * 64}],
        readiness=ready,
    )
    reused = freezer.freeze(
        family=DataFamily.MARKET_BREADTH,
        collection_epoch=start,
        data_end=start + timedelta(days=60),
        source_manifests=[{"source": "coinmarketcap", "root_hash": "a" * 64}],
        readiness=ready,
    )
    assert reused["dataset_id"] == frozen["dataset_id"]
    assert frozen["holdout_status"] == "RESERVED_UNTOUCHED"
    assert frozen["target_economics_inspected"] is False
    assert frozen["automatic_alpha_started"] is False


def test_runtime_has_no_execution_import_and_source_errors_are_isolated(
    isolated_settings,
    tmp_path,
) -> None:
    source = inspect.getsource(__import__("data.multi_source_runtime", fromlist=["*"]))
    assert "from execution" not in source
    assert "import execution" not in source
    paths = isolated_settings.paths.model_copy(
        update={
            "raw_data_dir": tmp_path / "raw",
            "output_dir": tmp_path / "output",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": paths})
    collector = MultiSourceCollector(settings)
    bitvavo_before = dict(collector.source_status["bitvavo"])
    collector._record_error("kraken", ConnectionError("unit"))
    assert collector.source_status["kraken"]["state"] == "DEGRADED"
    assert collector.source_status["bitvavo"] == bitvavo_before
    assert collector.snapshot()["execution"]["new_exchange_mutations"] == 0
