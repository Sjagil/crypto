from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from data.market_structure import (
    FEATURE_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    RAW_EVENT_SCHEMA_VERSION,
    AggressorSemantics,
    BookEvent,
    BookQuality,
    CanonicalInstrument,
    ClockMonitor,
    ClockQuality,
    CoverageRecord,
    DataLayer,
    DatasetManifest,
    EventTimestamps,
    EventType,
    FXObservation,
    HistoricalAvailability,
    LayeredParquetStore,
    MarketType,
    Missingness,
    OrderBookReplayer,
    PointInTimeFXBook,
    RawMarketEvent,
    ReadinessPolicy,
    TimestampQuality,
    TradeEvent,
    VenueQuote,
    VenueRole,
    align_cross_venue_quotes,
    assert_future_invariance,
    assess_readiness,
    benchmark_market_structure_pipeline,
    build_research_labels,
    build_trade_flow_buckets,
    deduplicate_trades,
    detect_liquidity_shocks,
    deterministic_replay_hash,
    infer_aggressor_side,
    initial_instrument_registry,
    market_data_health,
    market_structure_feature_schema,
    normalize_quote_to_eur,
    parquet_coverage,
    reference_repository_provenance,
    source_inventory,
    stage0_exact_divergence_evidence,
)
from reporting.market_structure_platform import build_market_structure_platform

ORIGIN = datetime(2026, 1, 1, tzinfo=UTC)


def timestamps(index: int = 0, *, latency_ms: int = 20) -> EventTimestamps:
    event_at = ORIGIN + timedelta(seconds=index)
    receive = event_at + timedelta(milliseconds=latency_ms)
    return EventTimestamps(
        exchange_event_timestamp=event_at,
        local_receive_timestamp=receive,
        normalized_event_timestamp=event_at,
        persisted_timestamp=receive + timedelta(milliseconds=5),
        quality=TimestampQuality.EXCHANGE_REPORTED,
    )


def raw_event(
    event_type: EventType,
    index: int = 0,
    *,
    sequence: int | None = None,
    previous_sequence: int | None = None,
    exchange_event_id: str | None = None,
) -> RawMarketEvent:
    instrument = initial_instrument_registry().resolve("bitvavo", "BTC-EUR")
    return RawMarketEvent(
        venue="bitvavo",
        instrument_id=instrument.instrument_id,
        event_type=event_type,
        timestamps=timestamps(index),
        payload={"index": index, "event_type": event_type.value},
        sequence=sequence,
        previous_sequence=previous_sequence,
        exchange_event_id=exchange_event_id,
    )


def trade(index: int, side: str | None = "buy", *, event_id: str | None = None) -> TradeEvent:
    semantics = (
        AggressorSemantics.AGGRESSOR_EXCHANGE_REPORTED
        if side
        else AggressorSemantics.AGGRESSOR_UNKNOWN
    )
    price = Decimal(50_000 + index)
    return TradeEvent(
        raw=raw_event(EventType.TRADE, index, exchange_event_id=event_id or f"trade-{index}"),
        price=price,
        quantity=Decimal("0.01"),
        quote_notional=price * Decimal("0.01"),
        aggressor_side=side,
        aggressor_semantics=semantics,
    )


def quote(
    venue: str,
    quote_currency: str,
    index: int,
    bid: str,
    ask: str,
) -> VenueQuote:
    registry = initial_instrument_registry()
    symbol = "BTC-EUR" if venue == "bitvavo" else ("XBT/EUR" if venue == "kraken" else "BTCUSDT")
    instrument = registry.resolve(venue, symbol)
    return VenueQuote(
        instrument_id=instrument.instrument_id,
        canonical_asset_id="CRYPTO:BTC",
        venue=venue,
        quote_currency=quote_currency,
        best_bid=Decimal(bid),
        best_ask=Decimal(ask),
        timestamps=timestamps(index),
        clock_quality=ClockQuality.CLOCK_OK,
    )


def test_canonical_instrument_identity_is_stable_and_venue_specific() -> None:
    registry = initial_instrument_registry()
    bitvavo = registry.resolve("bitvavo", "BTC-EUR")
    kraken = registry.resolve("kraken", "XBT/EUR")
    assert bitvavo.canonical_asset_id == kraken.canonical_asset_id == "CRYPTO:BTC"
    assert bitvavo.instrument_id != kraken.instrument_id
    assert registry.registry_hash == initial_instrument_registry().registry_hash


def test_only_bitvavo_spot_has_execution_authority() -> None:
    registry = initial_instrument_registry()
    enabled = [row for row in registry.instruments if row.execution_allowed]
    assert enabled
    assert {row.venue for row in enabled} == {"bitvavo"}
    assert {row.market_type for row in enabled} == {MarketType.SPOT}


def test_reference_venue_cannot_be_execution_enabled() -> None:
    with pytest.raises(ValueError, match="only Shariah-eligible Bitvavo"):
        CanonicalInstrument(
            canonical_asset_id="CRYPTO:BTC",
            base_asset="BTC",
            quote_asset="EUR",
            venue="kraken",
            venue_symbol="XBT/EUR",
            market_type=MarketType.SPOT,
            venue_role=VenueRole.SPOT_REFERENCE,
            execution_allowed=True,
            shariah_spot_eligible=True,
        )


def test_derivatives_are_context_only() -> None:
    registry = initial_instrument_registry()
    derivatives = [row for row in registry.instruments if row.market_type is not MarketType.SPOT]
    assert derivatives
    assert not any(row.execution_allowed for row in derivatives)
    assert {row.venue_role for row in derivatives} == {VenueRole.DERIVATIVES_CONTEXT}


def test_timestamp_model_distinguishes_event_receive_and_persist() -> None:
    selected = timestamps()
    assert selected.exchange_event_timestamp != selected.local_receive_timestamp
    assert selected.available_at == selected.local_receive_timestamp
    assert selected.observed_latency_ms == pytest.approx(20.0)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        EventTimestamps(
            exchange_event_timestamp=None,
            local_receive_timestamp=datetime(2026, 1, 1),
            normalized_event_timestamp=datetime(2026, 1, 1),
            persisted_timestamp=datetime(2026, 1, 1),
            quality=TimestampQuality.INFERRED,
        )


def test_rest_timestamp_requires_request_response_and_normalizes_to_receive() -> None:
    receive = ORIGIN + timedelta(seconds=2)
    selected = EventTimestamps(
        exchange_event_timestamp=None,
        local_receive_timestamp=receive,
        normalized_event_timestamp=receive,
        persisted_timestamp=receive + timedelta(milliseconds=1),
        quality=TimestampQuality.REST_OBSERVED_ONLY,
        request_start=ORIGIN,
        response_received=receive,
    )
    assert selected.observed_latency_ms is None


def test_clock_monitor_ok_suspect_invalid_and_not_evaluable() -> None:
    ok = ClockMonitor()
    ok.observe("bitvavo", timestamps(latency_ms=20))
    assert ok.assess("bitvavo").quality is ClockQuality.CLOCK_OK
    suspect = ClockMonitor()
    suspect.observe("x", timestamps(latency_ms=3_000))
    assert suspect.assess("x").quality is ClockQuality.CLOCK_SUSPECT
    invalid = ClockMonitor()
    invalid.observe("x", timestamps(latency_ms=20_000))
    assert invalid.assess("x").quality is ClockQuality.CLOCK_INVALID
    assert ok.assess("kraken").quality is ClockQuality.CLOCK_NOT_EVALUABLE


def test_raw_event_identity_and_payload_hash_are_deterministic() -> None:
    first = raw_event(EventType.TRADE, exchange_event_id="same")
    second = raw_event(EventType.TRADE, exchange_event_id="same")
    assert first.event_id == second.event_id
    assert first.raw_payload_hash == second.raw_payload_hash


def test_trade_deduplication_uses_venue_trade_identity() -> None:
    first = trade(1, event_id="same")
    duplicate = trade(2, event_id="same")
    unique, rejected = deduplicate_trades([first, duplicate])
    assert len(unique) == 1
    assert len(rejected) == 1


@pytest.mark.parametrize(
    ("kwargs", "expected_side", "semantics"),
    [
        ({"exchange_side": "BUY"}, "buy", AggressorSemantics.AGGRESSOR_EXCHANGE_REPORTED),
        ({"best_bid": "99", "best_ask": "100"}, "buy", AggressorSemantics.AGGRESSOR_INFERRED),
        ({"previous_trade_price": "101"}, "sell", AggressorSemantics.AGGRESSOR_INFERRED),
        ({}, None, AggressorSemantics.AGGRESSOR_UNKNOWN),
    ],
)
def test_aggressor_side_semantics(
    kwargs: dict[str, str], expected_side: str | None, semantics: AggressorSemantics
) -> None:
    side, actual, _ = infer_aggressor_side(trade_price="100", **kwargs)
    assert side == expected_side
    assert actual is semantics


def test_orderbook_snapshot_delta_spread_depth_microprice_and_imbalance() -> None:
    replay = OrderBookReplayer(
        instrument_id=raw_event(EventType.BOOK_SNAPSHOT).instrument_id,
        venue="bitvavo",
    )
    replay.apply(
        BookEvent(
            raw=raw_event(EventType.BOOK_SNAPSHOT, sequence=10),
            bids=((Decimal("99"), Decimal("2")), (Decimal("98"), Decimal("3"))),
            asks=((Decimal("101"), Decimal("4")), (Decimal("102"), Decimal("5"))),
        )
    )
    replay.apply(
        BookEvent(
            raw=raw_event(EventType.BOOK_DELTA, 1, sequence=11, previous_sequence=10),
            bids=((Decimal("99"), Decimal("4")),),
            asks=((Decimal("101"), Decimal("2")),),
        )
    )
    features = replay.features(now=timestamps(1).available_at)
    assert features.quality is BookQuality.BOOK_VALID
    assert features.mid == Decimal("100")
    assert features.spread == Decimal("2")
    assert features.microprice == pytest.approx(Decimal("100.3333333333333333333333333"))
    assert features.top_level_imbalance == Decimal("0.3333333333333333333333333333")
    assert features.depth["bid_depth_100bps_quote"] == Decimal("396")


def test_orderbook_sequence_gap_fails_closed() -> None:
    replay = OrderBookReplayer(
        instrument_id=raw_event(EventType.BOOK_SNAPSHOT).instrument_id, venue="bitvavo"
    )
    replay.apply(
        BookEvent(
            raw=raw_event(EventType.BOOK_SNAPSHOT, sequence=10),
            bids=((Decimal("99"), Decimal("1")),),
            asks=((Decimal("101"), Decimal("1")),),
        )
    )
    with pytest.raises(RuntimeError, match="sequence gap"):
        replay.apply(
            BookEvent(
                raw=raw_event(EventType.BOOK_DELTA, 1, sequence=12, previous_sequence=10),
                bids=(),
                asks=(),
            )
        )
    assert replay.state is BookQuality.BOOK_GAPPED
    assert not replay.bids and not replay.asks


def test_orderbook_reconnect_requires_fresh_snapshot() -> None:
    replay = OrderBookReplayer(
        instrument_id=raw_event(EventType.BOOK_SNAPSHOT).instrument_id, venue="bitvavo"
    )
    replay.reconnect_reset()
    assert replay.state is BookQuality.BOOK_SYNCING
    with pytest.raises(RuntimeError, match="valid snapshot"):
        replay.apply(BookEvent(raw=raw_event(EventType.BOOK_DELTA, sequence=1), bids=(), asks=()))


def test_orderbook_crossed_snapshot_is_invalid() -> None:
    replay = OrderBookReplayer(
        instrument_id=raw_event(EventType.BOOK_SNAPSHOT).instrument_id, venue="bitvavo"
    )
    with pytest.raises(ValueError, match="locked or crossed"):
        replay.apply(
            BookEvent(
                raw=raw_event(EventType.BOOK_SNAPSHOT, sequence=1),
                bids=((Decimal("101"), Decimal("1")),),
                asks=((Decimal("100"), Decimal("1")),),
            )
        )
    assert replay.state is BookQuality.BOOK_INVALID


def test_orderbook_staleness_is_explicit() -> None:
    replay = OrderBookReplayer(
        instrument_id=raw_event(EventType.BOOK_SNAPSHOT).instrument_id,
        venue="bitvavo",
        stale_after=timedelta(seconds=1),
    )
    replay.apply(
        BookEvent(
            raw=raw_event(EventType.BOOK_SNAPSHOT, sequence=1),
            bids=((Decimal("99"), Decimal("1")),),
            asks=((Decimal("101"), Decimal("1")),),
        )
    )
    assert (
        replay.features(now=timestamps().available_at + timedelta(seconds=2)).quality
        is BookQuality.BOOK_STALE
    )


def test_uninitialized_orderbook_output_is_deterministic() -> None:
    replay = OrderBookReplayer(instrument_id="x", venue="bitvavo")
    assert replay.features().feature_hash == replay.features().feature_hash
    assert replay.features().event_timestamp == datetime(1970, 1, 1, tzinfo=UTC)


def test_trade_flow_cvd_and_unknown_aggressor_are_explicit() -> None:
    rows = [trade(1, "buy"), trade(2, "sell"), trade(3, None)]
    frame = build_trade_flow_buckets(rows, bucket_seconds=60)
    assert frame.loc[0, "unknown_aggressor_count"] == 1
    assert frame.loc[0, "cvd"] == pytest.approx(
        float(rows[0].quote_notional - rows[1].quote_notional)
    )


def test_trade_flow_trailing_normalization_uses_prior_buckets_only() -> None:
    rows = [trade(index, "buy") for index in range(1, 241)]
    baseline = build_trade_flow_buckets(rows[:180], bucket_seconds=60)
    expanded = build_trade_flow_buckets(rows, bucket_seconds=60)
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1].reset_index(drop=True),
        expanded.iloc[: len(baseline) - 1].reset_index(drop=True),
    )


def test_future_invariance_detects_safe_prefix_builder() -> None:
    result = assert_future_invariance(lambda values: list(values), [1, 2, 3, 4], cutoff=2)
    assert result["status"] == "PASSED"


def test_liquidity_shock_uses_trailing_not_centered_statistics() -> None:
    frame = pd.DataFrame(
        {
            "spread_bps": [1.0] * 5 + [5.0],
            "depth_25bps_quote": [100.0] * 5 + [20.0],
            "trade_intensity": [1.0] * 5 + [5.0],
        }
    )
    result = detect_liquidity_shocks(frame, trailing_buckets=4)
    assert bool(result.iloc[-1]["liquidity_shock"])
    assert not bool(result.iloc[0]["liquidity_shock"])


def test_point_in_time_fx_direct_inverse_missing_and_stale() -> None:
    observation = FXObservation("USDT", "EUR", Decimal("0.9"), ORIGIN, ORIGIN, "OBSERVED")
    book = PointInTimeFXBook([observation])
    assert book.rate("USDT", "EUR", at=ORIGIN, maximum_age=timedelta(minutes=5))[0] == Decimal(
        "0.9"
    )
    assert book.rate("EUR", "USDT", at=ORIGIN, maximum_age=timedelta(minutes=5))[0] == Decimal(
        "1"
    ) / Decimal("0.9")
    assert (
        book.rate("USD", "EUR", at=ORIGIN, maximum_age=timedelta(minutes=5))[1]
        is Missingness.UNAVAILABLE
    )
    assert (
        book.rate("USDT", "EUR", at=ORIGIN + timedelta(hours=1), maximum_age=timedelta(minutes=5))[
            1
        ]
        is Missingness.STALE
    )


def test_stablecoin_conversion_never_assumes_fixed_parity() -> None:
    result = normalize_quote_to_eur(
        quote("mexc", "USDT", 0, "50000", "50002"),
        fx=PointInTimeFXBook([]),
    )
    assert result["normalized_eur_mid"] is None
    assert result["fx_missingness"] == Missingness.UNAVAILABLE.value


def test_cross_venue_alignment_preserves_venue_identity_and_fx() -> None:
    fx = PointInTimeFXBook(
        [FXObservation("USDT", "EUR", Decimal("0.9"), ORIGIN, ORIGIN, "OBSERVED")]
    )
    quotes = [
        quote("bitvavo", "EUR", 0, "44999", "45001"),
        quote("mexc", "USDT", 0, "49999", "50001"),
    ]
    result = align_cross_venue_quotes(quotes, fx=fx, bucket_seconds=5)
    assert result.loc[0, "reference_venue_count"] == 1
    assert set(result.loc[0, "venues"]) == {"bitvavo", "mexc"}
    assert result.loc[0, "primary_reference_premium_bps"] == pytest.approx(0.0)


def test_feature_schema_labels_l1_l2_but_never_l3() -> None:
    features = market_structure_feature_schema()
    assert features
    assert all("L3" not in source for row in features for source in row.source)
    assert all(
        row.timestamp_semantics == "AVAILABLE_AT_BUCKET_END_NO_CENTERING" for row in features
    )


def test_research_labels_are_offline_and_executable_ask_to_future_bid() -> None:
    quotes = pd.DataFrame(
        {
            "instrument_id": ["x", "x"],
            "venue": ["bitvavo", "bitvavo"],
            "available_at": [ORIGIN, ORIGIN + timedelta(seconds=60)],
            "best_bid": [99.0, 109.0],
            "best_ask": [101.0, 111.0],
        }
    )
    labels = build_research_labels(quotes, horizons_seconds=(60,), fee_bps_roundtrip=0)
    assert labels.loc[0, "executable_return_60s"] == pytest.approx(109 / 101 - 1)
    assert labels.loc[0, "layer"] == DataLayer.RESEARCH_LABEL.value
    assert not bool(labels.loc[0, "execution_authority"])


def test_layered_store_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    store = LayeredParquetStore(tmp_path)
    kwargs = dict(
        layer=DataLayer.RAW,
        venue="bitvavo",
        canonical_asset_id="CRYPTO:BTC",
        data_type="TRADE",
        date="2026-01-01",
        rows=[raw_event(EventType.TRADE).to_dict()],
        schema_version=RAW_EVENT_SCHEMA_VERSION,
    )
    first = store.write_rows(**kwargs)
    second = store.write_rows(**kwargs)
    assert first["path"] == second["path"]
    assert second["reused"] is True
    assert first["file_sha256"] == second["file_sha256"]


def test_feature_layer_rejects_future_labels(tmp_path: Path) -> None:
    store = LayeredParquetStore(tmp_path)
    with pytest.raises(ValueError, match="future-derived"):
        store.write_rows(
            layer=DataLayer.FEATURE,
            venue="bitvavo",
            canonical_asset_id="CRYPTO:BTC",
            data_type="FEATURE",
            date="2026-01-01",
            rows=[{"future_mid_return_60s": 0.1}],
            schema_version=FEATURE_SCHEMA_VERSION,
        )


def test_raw_layer_rejects_labels(tmp_path: Path) -> None:
    store = LayeredParquetStore(tmp_path)
    with pytest.raises(ValueError, match="research labels"):
        store.write_rows(
            layer=DataLayer.RAW,
            venue="bitvavo",
            canonical_asset_id="CRYPTO:BTC",
            data_type="TRADE",
            date="2026-01-01",
            rows=[{"label": 1}],
            schema_version=RAW_EVENT_SCHEMA_VERSION,
        )


def test_parquet_coverage_reports_absence_and_present_partition(tmp_path: Path) -> None:
    absent = parquet_coverage(
        tmp_path / "absent",
        venue="kraken",
        canonical_asset_id="CRYPTO:BTC",
        data_type="TRADE",
        availability=HistoricalAvailability.NOT_AVAILABLE,
    )
    assert absent.event_count == 0
    store = LayeredParquetStore(tmp_path / "store")
    store.write_rows(
        layer=DataLayer.RAW,
        venue="bitvavo",
        canonical_asset_id="CRYPTO:BTC",
        data_type="TRADE",
        date="2026-01-01",
        rows=[{"observed_at": ORIGIN.isoformat(), "value": 1}],
        schema_version=RAW_EVENT_SCHEMA_VERSION,
    )
    present = parquet_coverage(
        tmp_path / "store",
        venue="bitvavo",
        canonical_asset_id="CRYPTO:BTC",
        data_type="TRADE",
        availability=HistoricalAvailability.CURRENT_COLLECTION_ONLY,
    )
    assert present.event_count == 1
    assert present.availability is HistoricalAvailability.CURRENT_COLLECTION_ONLY


def test_research_readiness_states_are_evidence_based() -> None:
    policy = ReadinessPolicy(
        minimum_days_research=1,
        minimum_days_robustness=10,
        minimum_valid_observations=2,
        minimum_venue_overlap=2,
    )
    base = dict(
        venue="bitvavo",
        canonical_asset_id="CRYPTO:BTC",
        data_type="TRADE",
        availability=HistoricalAvailability.CURRENT_COLLECTION_ONLY,
        file_count=1,
        gap_count=0,
        coverage_percentage=100.0,
        timestamp_basis="OBSERVED_AT",
        quality_status="OK",
        reason_codes=(),
    )
    absent = CoverageRecord(start_timestamp=None, end_timestamp=None, event_count=0, **base)
    assert (
        assess_readiness(
            absent, policy=policy, venue_overlap=0, clock_quality=ClockQuality.CLOCK_NOT_EVALUABLE
        )["state"]
        == "NOT_STARTED"
    )
    partial = CoverageRecord(
        start_timestamp=ORIGIN, end_timestamp=ORIGIN + timedelta(hours=1), event_count=2, **base
    )
    assert (
        assess_readiness(
            partial, policy=policy, venue_overlap=2, clock_quality=ClockQuality.CLOCK_OK
        )["state"]
        == "PARTIAL"
    )
    usable = CoverageRecord(
        start_timestamp=ORIGIN, end_timestamp=ORIGIN + timedelta(days=2), event_count=2, **base
    )
    assert (
        assess_readiness(
            usable, policy=policy, venue_overlap=2, clock_quality=ClockQuality.CLOCK_OK
        )["state"]
        == "RESEARCH_USABLE"
    )
    failed = CoverageRecord(
        start_timestamp=ORIGIN,
        end_timestamp=ORIGIN + timedelta(days=2),
        event_count=2,
        **{**base, "gap_count": 2},
    )
    assert (
        assess_readiness(
            failed, policy=policy, venue_overlap=2, clock_quality=ClockQuality.CLOCK_OK
        )["state"]
        == "QUALITY_FAILED"
    )


def test_dataset_manifest_identity_is_stable_despite_build_timestamp() -> None:
    coverage = CoverageRecord(
        "bitvavo",
        "CRYPTO:BTC",
        "TRADE",
        HistoricalAvailability.CURRENT_COLLECTION_ONLY,
        ORIGIN,
        ORIGIN + timedelta(days=2),
        10,
        1,
        0,
        100.0,
        "OBSERVED_AT",
        "OK",
        (),
    )
    kwargs = dict(
        sources=source_inventory(),
        venues=["bitvavo"],
        assets=["BTC"],
        coverage=[coverage],
        raw_event_hashes=["a" * 64],
        normalized_partitions=[],
        clock_quality={"bitvavo": "CLOCK_OK"},
        missingness={},
        gaps={},
        rejected_rows={},
        build_commit="abc",
        collection_start_timestamp=ORIGIN,
        replay_hash="b" * 64,
    )
    assert DatasetManifest.build(**kwargs).dataset_id == DatasetManifest.build(**kwargs).dataset_id


def test_deterministic_replay_is_order_independent_for_unique_rows() -> None:
    rows = [
        {"available_at": ORIGIN.isoformat(), "event_id": "b"},
        {"available_at": (ORIGIN + timedelta(seconds=1)).isoformat(), "event_id": "a"},
    ]
    assert deterministic_replay_hash(rows) == deterministic_replay_hash(list(reversed(rows)))


def test_market_health_keeps_execution_and_research_separate() -> None:
    health = market_data_health({"REFERENCE_BOOK": {"status": "FAILED", "error_count": 1}})
    assert health["execution_health"] == "UNCHANGED_SEPARATE_CANONICAL_SYSTEM"
    assert health["optional_reference_failure_stops_execution"] is False
    assert health["orders_generated"] == 0


def test_reference_provenance_is_read_only_and_no_code_copied() -> None:
    rows = reference_repository_provenance(Path.cwd())
    assert {row["repository"] for row in rows} == {
        "nautilus_trader",
        "freqtrade",
        "lean",
        "qlib",
        "vectorbt",
        "pybroker",
    }
    assert all(row["read_only"] and not row["source_code_copied"] for row in rows)


def test_stage0_exact_divergence_records_semantic_noncomparability() -> None:
    evidence = stage0_exact_divergence_evidence(
        {
            "stage0": {
                "family_results": [
                    {"family": "CROSS_SECTIONAL_MOMENTUM", "best_result": {"profit_factor": 2.0}}
                ]
            },
            "exact_results": {
                "CROSS_SECTIONAL_MOMENTUM": {"final_test": {"metrics": {"profit_factor": 0.9}}}
            },
        }
    )
    assert evidence["stage0_profit_factor"] == 2.0
    assert evidence["exact_profit_factor"] == 0.9
    assert evidence["units_are_not_directly_comparable"] is True
    assert evidence["root_cause_fully_attributed"] is False


def test_performance_benchmark_has_storage_and_throughput_metrics() -> None:
    result = benchmark_market_structure_pipeline(
        [trade(index) for index in range(120)], repetitions=1
    )
    assert result["input_events"] == 120
    assert result["events_per_second"] > 0
    assert result["estimated_storage_growth_per_day_at_100_events_second_gb"] > 0


def test_platform_builder_emits_a_to_v_and_zero_authority(tmp_path: Path) -> None:
    result = build_market_structure_platform(
        Path.cwd(), scan_local_data=False, output_root=tmp_path
    )
    payload = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert tuple(payload["sections"]) == tuple("ABCDEFGHIJKLMNOPQRSTUV")
    assert payload["sections"]["U"]["private_bitvavo_mutations"] == 0
    assert payload["sections"]["V"]["automatically_started"] is False
    assert payload["ml_authority"] == "SHADOW_ONLY"
    assert payload["portfolio_allocator_built"] is False
    assert payload["sections"]["I"]["status"] == "NOT_EVALUABLE"
    assert len(payload["sections"]["E"]["matrix"]) == 4 * 3 * 6
    assert LABEL_SCHEMA_VERSION == payload["sections"]["N"]["schema_version"]
