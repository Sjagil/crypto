from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from data.multi_source_maturation import (
    CollectorAlreadyActive,
    CollectorLease,
    CrossVenueAlignmentMonitor,
    FamilyFreezeManager,
    ReadinessHistoryStore,
    ReadinessLevel,
    StorageGrowthMonitor,
    api_usage_report,
    assess_readiness,
    classify_event,
    hypothesis_readiness,
    mexc_derivatives_maturation,
    research_readiness_policy_v1,
    verify_restart_continuity,
)
from data.multi_source_platform import (
    ApiBudgetLedger,
    DataClassification,
    ImmutableSourceLedger,
    SourceNeutralObservation,
    TimestampResolution,
    compute_cmc_breadth,
    verify_source_ledger,
)
from data.multi_source_runtime import MultiSourceCollector
from utils.common import atomic_write_json


def _observation(source: str, at: datetime, identity: str) -> SourceNeutralObservation:
    return SourceNeutralObservation(
        source=source,
        source_type="PUBLIC_WEBSOCKET",
        venue=source,
        canonical_asset_id="CRYPTO:BTC",
        venue_instrument_id="BTC/EUR",
        data_type="TRADE",
        exchange_event_timestamp=at,
        local_receive_timestamp=at + timedelta(milliseconds=20),
        normalized_timestamp=at,
        persisted_timestamp=at + timedelta(milliseconds=30),
        raw_payload={"price": "100", "qty": "1"},
        timestamp_resolution=TimestampResolution.EVENT_EXACT,
        source_event_id=identity,
        classification=DataClassification.PROSPECTIVE_COLLECTION,
    )


def _ready_metrics(policy, level: str) -> dict:
    return {
        "history_days": getattr(policy, f"{level}_minimum_history_days"),
        "observations": getattr(policy, f"{level}_minimum_observations"),
        "valid_fraction": getattr(policy, f"{level}_minimum_valid_fraction"),
        "gap_fraction": getattr(policy, f"{level}_maximum_gap_fraction"),
        "assets": list(policy.required_assets),
        "quality": list(policy.required_quality),
    }


def test_single_collector_atomic_ownership_and_stale_recovery(tmp_path) -> None:
    lock = tmp_path / "collector.lock"
    history = tmp_path / "history"
    first = CollectorLease(lock, history)
    second = CollectorLease(lock, history)
    first.acquire()
    with pytest.raises(CollectorAlreadyActive, match="COLLECTOR_ALREADY_ACTIVE"):
        second.acquire()
    first.release()

    atomic_write_json(lock, {"pid": 2_147_483_647, "instance_id": "stale"})
    recovered = CollectorLease(lock, history)
    payload = recovered.acquire()
    assert payload["instance_id"] == recovered.instance_id
    assert list(history.glob("stale-*.json"))
    recovered.release()


def test_hash_continuation_and_duplicate_prevention_after_restart(tmp_path) -> None:
    root = tmp_path / "raw"
    checkpoint = tmp_path / "checkpoint.json"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = ImmutableSourceLedger(root, "bitvavo", checkpoint)
    first.append_many([_observation("bitvavo", start, "one")])
    previous_status = {"ledger_checkpoints": {"bitvavo": first.checkpoint()}}
    prior_hash = first.root_hash
    restarted = ImmutableSourceLedger(root, "bitvavo", checkpoint)
    assert restarted.root_hash == prior_hash
    assert restarted.append_many([_observation("bitvavo", start, "one")])["appended"] == 0
    restarted.append_many([_observation("bitvavo", start + timedelta(seconds=1), "two")])
    assert restarted.root_hash != prior_hash
    assert verify_source_ledger(root, "bitvavo")["status"] == "PASSED"
    current_status = {"ledger_checkpoints": {"bitvavo": restarted.checkpoint()}}
    continuity = verify_restart_continuity(previous_status, current_status)
    assert continuity["status"] == "PASSED"
    assert continuity["sources"]["bitvavo"]["added_records"] == 1


def test_versioned_policy_has_ten_independent_families_and_monotonic_states() -> None:
    policies = research_readiness_policy_v1()
    assert len(policies) == 10
    assert all(row.policy_version == "research_readiness_policy_v1" for row in policies.values())
    policy = policies["BITVAVO_FLOW"]
    assert assess_readiness(policy, {"observations": 0})["state"] == "NOT_STARTED"
    assert (
        assess_readiness(policy, {"observations": 1})["state"]
        == ReadinessLevel.COLLECTING
    )
    partial = {
        "history_days": policy.exploratory_minimum_history_days * 0.25,
        "observations": 1,
    }
    assert assess_readiness(policy, partial)["state"] == ReadinessLevel.PARTIAL
    assert (
        assess_readiness(policy, _ready_metrics(policy, "exploratory"))["state"]
        == ReadinessLevel.EXPLORATORY_USABLE
    )
    assert (
        assess_readiness(policy, _ready_metrics(policy, "research"))["state"]
        == ReadinessLevel.RESEARCH_USABLE
    )
    assert (
        assess_readiness(policy, _ready_metrics(policy, "robustness"))["state"]
        == ReadinessLevel.ROBUSTNESS_USABLE
    )


def test_hypothesis_readiness_requires_freeze_and_spot_candidate() -> None:
    policies = research_readiness_policy_v1()
    assessments = {
        name: assess_readiness(policy, _ready_metrics(policy, "research"))
        for name, policy in policies.items()
    }
    blocked = hypothesis_readiness(assessments)
    assert blocked["H3_CROSS_VENUE_LEAD_LAG_READY"]["ready"] is False
    assert "UNTOUCHED_HOLDOUT_NOT_FROZEN" in blocked[
        "H3_CROSS_VENUE_LEAD_LAG_READY"
    ]["additional_blockers"]
    assert blocked["H7_DERIVATIVES_MODIFIER_READY"]["ready"] is False
    ready = hypothesis_readiness(
        assessments,
        frozen_families=("CROSS_VENUE_LEAD_LAG",),
        spot_candidate_available=True,
    )
    assert ready["H3_CROSS_VENUE_LEAD_LAG_READY"]["ready"] is True
    assert ready["H7_DERIVATIVES_MODIFIER_READY"]["ready"] is True
    assert all(row["trade_ready"] is False for row in ready.values())


def test_immutable_transition_history_freeze_holdout_and_forward_partition(tmp_path) -> None:
    policy = research_readiness_policy_v1()["BITVAVO_FLOW"]
    assessment = assess_readiness(policy, _ready_metrics(policy, "research"))
    history = ReadinessHistoryStore(tmp_path / "readiness")
    first = history.record(assessment, at=datetime(2026, 3, 1, tzinfo=UTC))
    assert first["status"] == "TRANSITION_RECORDED"
    assert history.record(assessment)["status"] == "UNCHANGED"
    assert len(list((tmp_path / "readiness").rglob("history/*.json"))) == 1

    freezes = FamilyFreezeManager(tmp_path / "freezes")
    frozen = freezes.maybe_freeze(
        assessment=assessment,
        transition=first["transition"],
        source_manifests=[{"source": "bitvavo", "root_hash": "a" * 64}],
        assets=("CRYPTO:BTC", "CRYPTO:ETH", "CRYPTO:SOL"),
        features=("CVD", "TRADE_INTENSITY"),
        collection_start=datetime(2026, 1, 1, tzinfo=UTC),
        data_end=datetime(2026, 3, 1, tzinfo=UTC),
        coverage=assessment["metrics"],
        clock_metrics={"status": "PASS"},
        build_commit="unit",
    )
    manifest = frozen["freeze"]
    assert frozen["status"] == "FREEZE_CREATED"
    assert manifest["holdout_status"] == "RESERVED_UNTOUCHED"
    assert manifest["automatic_stage0_started"] is False
    assert manifest["automatic_backtest_started"] is False
    assert manifest["automatic_ml_training_started"] is False
    assert manifest["automatic_strategy_promotion"] is False
    assert manifest["live_authority_changed"] is False
    assert freezes.maybe_freeze(
        assessment=assessment,
        transition=first["transition"],
        source_manifests=[],
        assets=(),
        features=(),
        collection_start=datetime(2026, 1, 1, tzinfo=UTC),
        data_end=datetime(2026, 3, 1, tzinfo=UTC),
        coverage={},
        clock_metrics={},
        build_commit=None,
    )["status"] == "ALREADY_FROZEN"
    after_freeze = datetime.fromisoformat(
        manifest["post_freeze_forward_data_start"].replace("Z", "+00:00")
    ) + timedelta(seconds=1)
    assert (
        freezes.classify_timestamp("BITVAVO_FLOW", after_freeze)
        == "POST_FREEZE_FORWARD_DATA"
    )


def test_storage_growth_uses_observed_windows_and_disk_thresholds(tmp_path, monkeypatch) -> None:
    monitor = StorageGrowthMonitor(tmp_path / "storage", minimum_sample_interval_seconds=0)
    monkeypatch.setattr(
        "data.multi_source_maturation.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1_000, used=950, free=50),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    monitor.observe(
        {"bitvavo|CRYPTO:BTC|TRADE": {"events": 10, "raw_bytes": 100}},
        disk_path=tmp_path,
        force=True,
        at=start,
    )
    report = monitor.observe(
        {"bitvavo|CRYPTO:BTC|TRADE": {"events": 110, "raw_bytes": 1_100}},
        disk_path=tmp_path,
        force=True,
        at=start + timedelta(hours=1),
    )
    rate = report["windows"]["1h"]["rates"]["bitvavo|CRYPTO:BTC|TRADE"]
    assert rate["event_delta"] == 100
    assert rate["raw_byte_delta"] == 1_000
    assert report["status"] == "STORAGE_CRITICAL"
    assert report["bitvavo_execution_data_may_pause"] is False


def test_cross_venue_overlap_is_independent_at_each_resolution(tmp_path) -> None:
    monitor = CrossVenueAlignmentMonitor(tmp_path / "alignment.json")
    at = datetime.now(UTC).replace(microsecond=100_000)
    for source in ("bitvavo", "kraken"):
        monitor.observe(
            source=source,
            canonical_asset_id="CRYPTO:BTC",
            event_at=at,
            receive_at=at + timedelta(milliseconds=20),
        )
    monitor.persist()
    report = monitor.snapshot()
    assert report["resolutions"]["1s"]["matched_buckets"] == 1
    assert report["resolutions"]["5s"]["matched_buckets"] == 1
    assert report["resolutions"]["1s"]["readiness_is_resolution_specific"] is True
    assert report["arbitrary_cross_resolution_promotion"] is False


def test_cmc_breadth_definitions_are_pit_and_inputs_are_separate() -> None:
    rows = [
        {
            "cmc_rank": 1,
            "symbol": "BTC",
            "market_cap": 100,
            "percent_change_24h": 2,
            "percent_change_7d": 3,
            "volume_24h": 10,
            "market_cap_dominance": 60,
        },
        {
            "cmc_rank": 2,
            "symbol": "SOL",
            "market_cap": 20,
            "percent_change_24h": -1,
            "percent_change_7d": 4,
            "volume_24h": 5,
            "market_cap_dominance": 10,
        },
    ]
    result = compute_cmc_breadth(
        rows,
        known_at=datetime(2026, 1, 1, tzinfo=UTC),
        global_context={"btc_dominance": 0.61, "eth_dominance": 0.12},
        previous_breadth={"btc_dominance": 0.60},
    )
    assert result["equal_weight_market_return_24h"] == 0.5
    assert result["altcoin_positive_24h_fraction"] == 0
    assert result["btc_dominance_change"] == pytest.approx(0.01)
    assert result["raw_inputs_stored_separately"] is True
    assert result["future_universe_membership_used"] is False


def test_event_governance_prioritizes_high_value_and_never_social_noise() -> None:
    event = classify_event(
        source="Kraken",
        title="Bitcoin network upgrade announcement",
        summary="Official release",
    )
    assert event["canonical_asset_ids"] == ["CRYPTO:BTC"]
    assert "NETWORK_UPGRADE" in event["event_categories"]
    assert event["high_value_event"] is True
    assert event["social_noise_source"] is False


def test_existing_mexc_derivatives_context_is_pit_information_only(tmp_path) -> None:
    import pandas as pd

    root = tmp_path / "context"
    root.mkdir()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for asset in ("BTC", "ETH", "SOL"):
        frame = pd.DataFrame(
            {
                "available_at": [start, start + timedelta(days=15)],
                "point_in_time_status": ["SOURCE_AVAILABLE_AT"] * 2,
                "funding_rate": [0.0001, 0.0002],
                "open_interest": [100.0, 110.0],
                "basis": [1.0, 2.0],
                "execution_permitted": [False, False],
                "canonical_market": [f"{asset}-USDT"] * 2,
            }
        )
        frame.to_parquet(root / f"derivatives_mexc_{asset}.parquet", index=False)
    result = mexc_derivatives_maturation(root)
    assert result["observations"] == 6
    assert result["history_days"] == 15
    assert result["valid_fraction"] == 1
    assert set(result["assets"]) == {"CRYPTO:BTC", "CRYPTO:ETH", "CRYPTO:SOL"}
    assert "PIT_DERIVATIVES_CONTEXT" in result["quality"]
    assert "INFORMATION_ONLY" in result["quality"]
    assert result["execution_authority"] is False


def test_api_failure_accounting_and_secret_free_usage_report(tmp_path) -> None:
    from data.multi_source_platform import default_api_budget_rules

    ledger = ApiBudgetLedger(tmp_path / "budget.json", default_api_budget_rules())
    at = datetime(2026, 1, 1, 12, tzinfo=UTC)
    ledger.record_request("coinmarketcap", "rankings", credits=3, at=at)
    ledger.record_failure("coinmarketcap")
    ledger.record_rate_limit("coinmarketcap")
    report = api_usage_report(ledger.status(), observed_at=at)
    provider = report["providers"]["coinmarketcap"]
    assert provider["credits_today"] == 3
    assert provider["failed_requests"] == 1
    assert provider["rate_limit_events"] == 1
    assert report["credentials_serialized"] is False


class _Notifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def notify_system_event(self, event_type: str, payload: dict) -> dict:
        self.calls.append((event_type, payload))
        return {"delivery_status": "SENT", "orders_generated": 0}


def test_runtime_freeze_notification_is_once_and_never_trade_ready(
    isolated_settings, tmp_path, monkeypatch
) -> None:
    paths = isolated_settings.paths.model_copy(
        update={
            "raw_data_dir": tmp_path / "raw",
            "output_dir": tmp_path / "output",
            "data_dir": tmp_path / "data",
        }
    )
    notifier = _Notifier()
    collector = MultiSourceCollector(
        isolated_settings.model_copy(update={"paths": paths}), notifier=notifier
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    collector.ledgers["bitvavo"].append_many(
        [
            _observation("bitvavo", start, "start"),
            _observation("bitvavo", start + timedelta(days=60), "end"),
        ]
    )
    empty = {
        name: {
            "history_days": 0,
            "observations": 0,
            "valid_fraction": None,
            "gap_fraction": None,
            "assets": [],
            "quality": [],
        }
        for name in collector.readiness_policy
    }
    empty["BITVAVO_FLOW"] = _ready_metrics(
        collector.readiness_policy["BITVAVO_FLOW"], "research"
    )
    monkeypatch.setattr(collector, "_family_metrics", lambda _now: empty)
    first = collector._readiness()
    second = collector._readiness()
    assert first["families"]["BITVAVO_FLOW"]["dataset_ready_for_research"] is True
    assert first["families"]["BITVAVO_FLOW"]["trade_ready"] is False
    assert first["freeze_candidates"]["BITVAVO_FLOW"]["status"] == "FREEZE_CREATED"
    assert second["freeze_candidates"]["BITVAVO_FLOW"]["status"] == "ALREADY_FROZEN"
    assert len(notifier.calls) == 1
    assert notifier.calls[0][1]["status"] == "DATASET READY FOR RESEARCH"


@pytest.mark.asyncio
async def test_collector_startup_fails_safe_when_lease_is_owned(
    isolated_settings, tmp_path
) -> None:
    paths = isolated_settings.paths.model_copy(
        update={"raw_data_dir": tmp_path / "raw", "output_dir": tmp_path / "output"}
    )
    settings = isolated_settings.model_copy(update={"paths": paths})
    owner = MultiSourceCollector(settings)
    contender = MultiSourceCollector(settings)
    owner.lease.acquire()
    try:
        with pytest.raises(CollectorAlreadyActive, match="COLLECTOR_ALREADY_ACTIVE"):
            await contender.run(duration_seconds=0.01)
    finally:
        owner.lease.release()


@pytest.mark.asyncio
async def test_optional_source_and_compactor_failures_are_isolated(
    isolated_settings, tmp_path, monkeypatch
) -> None:
    paths = isolated_settings.paths.model_copy(
        update={
            "raw_data_dir": tmp_path / "raw",
            "output_dir": tmp_path / "output",
            "data_dir": tmp_path / "data",
        }
    )
    collector = MultiSourceCollector(isolated_settings.model_copy(update={"paths": paths}))
    bitvavo_before = dict(collector.source_status["bitvavo"])
    for source in ("kraken", "mexc_spot", "coinmarketcap", "eodhd", "scrapers"):
        collector._record_error(source, RuntimeError("secret-value-must-not-appear"))
    assert collector.source_status["bitvavo"] == bitvavo_before
    assert all(
        "secret-value" not in str(collector.source_status[source])
        for source in ("kraken", "mexc_spot", "coinmarketcap", "eodhd", "scrapers")
    )

    def fail_compaction(*_args, **_kwargs):
        raise OSError("unit")

    monkeypatch.setattr("data.multi_source_runtime.compact_source_ledger", fail_compaction)
    report = await collector._compact_closed_segments(datetime.now(UTC))
    assert all(
        row["status"] == "FAILED_ISOLATED" for row in report["sources"].values()
    )
    assert report["raw_deleted"] is False
    assert collector.source_status["bitvavo"] == bitvavo_before
