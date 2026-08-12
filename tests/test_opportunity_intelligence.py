from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from config.settings import PathSettings, Settings
from core.event_driven_live import _soft_exit_confirmed
from core.opportunity_intelligence import (
    FEATURE_SCHEMA_VERSION,
    bayesian_playbook_weights,
    build_canonical_ml_dataset,
    build_training_dataset,
    confirmation_independence,
    deduplicate_opportunities,
    estimate_roundtrip_economics,
    freeze_feature_snapshot,
    record_active_swing_forward_snapshots,
    train_canonical_shadow_models,
    train_shadow_models,
)


def test_active_swing_forward_snapshots_are_deduplicated_and_labelable(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    selected = _settings(isolated_settings, tmp_path)
    decision = datetime(2026, 8, 1, 12, tzinfo=UTC)
    candidate = {
        "market": "BTC-EUR",
        "strategy": "TACTICAL_15M_BREAKOUT_RETEST",
        "strategy_dna_hash": "abc123",
        "family": "BREAKOUT_RETEST",
        "entry_timeframe": "15m",
        "confirmation_timeframe": "1h",
        "regime_timeframe": "4h",
        "signal_timestamp": "2026-08-01T11:45:00+00:00",
        "current_price": 100.0,
        "stop": 98.0,
        "target_1": 103.0,
        "target_2": 106.0,
        "estimated_fee_fraction": 0.0025,
        "estimated_slippage_bps": 5.0,
        "opportunity_id": "source-id",
        "status": "ACTIONABLE",
    }

    first = record_active_swing_forward_snapshots(
        selected,
        [candidate],
        observed_at=decision,
    )
    second = record_active_swing_forward_snapshots(
        selected,
        [candidate],
        observed_at=decision + timedelta(minutes=15),
    )
    same_setup_next_candle = record_active_swing_forward_snapshots(
        selected,
        [{**candidate, "signal_timestamp": "2026-08-01T11:59:00+00:00"}],
        observed_at=decision + timedelta(minutes=30),
    )
    ignored_watch = record_active_swing_forward_snapshots(
        selected,
        [{**candidate, "strategy_dna_hash": "watch-only", "status": "WATCH"}],
        observed_at=decision + timedelta(minutes=45),
    )

    assert first["written_this_scan"] == 1
    assert second["written_this_scan"] == 0
    assert second["duplicates_this_scan"] == 1
    assert same_setup_next_candle["written_this_scan"] == 0
    assert same_setup_next_candle["duplicates_this_scan"] == 1
    assert ignored_watch["written_this_scan"] == 0
    assert ignored_watch["ignored_non_candidates_this_scan"] == 1
    ledger = (
        tmp_path
        / "output"
        / "intelligence"
        / "active_swing_decision_snapshots.jsonl"
    )
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(events) == 1
    snapshot = events[0]["feature_snapshot"]
    assert snapshot["point_in_time_timing_valid"] is True
    assert snapshot["values"]["strategy_id"] == candidate["strategy"]
    assert snapshot["values"]["entry_timeframe"] == "15m"

    selected.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(
        decision + timedelta(minutes=15),
        decision + timedelta(hours=25),
        freq="15min",
        tz="UTC",
    )
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    ).to_parquet(selected.paths.processed_data_dir / "BTC-EUR_15m.parquet")

    dataset = build_training_dataset(selected)

    assert dataset["prospective_snapshot_count"] == 1
    assert dataset["row_count"] == 1
    assert dataset["canonical_point_in_time_rows"] == 1


def _settings(settings: Settings, tmp_path) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_correlated_flow_confirmations_are_not_independent() -> None:
    result = confirmation_independence(
        {
            "valid_structure": True,
            "relative_volume": True,
            "trade_intensity": True,
            "taker_flow": True,
            "cvd": True,
            "ofi": True,
            "mlobi": False,
            "microprice": False,
        }
    )
    assert result["raw_count"] == 6
    # Executed prints (taker/CVD) remain one group.  L2 depth-flow OFI is a
    # separate source and therefore contributes one additional group.
    assert result["independent_count"] == 4


def test_cost_model_rejects_target_consumed_by_fees() -> None:
    result = estimate_roundtrip_economics(
        entry_price=100.0,
        stop_loss=99.8,
        take_profit_1=100.25,
        take_profit_2=100.40,
        spread_bps=10.0,
        observed_slippage_bps=8.0,
    )
    assert result["roundtrip_cost_bps"] > result["gross_target_2_bps"]
    assert result["positive_after_costs"] is False


def test_conservative_rule_ev_is_explicit_and_cannot_raise_caps() -> None:
    result = estimate_roundtrip_economics(
        entry_price=100.0,
        stop_loss=98.0,
        take_profit_1=103.0,
        take_profit_2=106.0,
        spread_bps=5.0,
        observed_slippage_bps=2.0,
        conservative_win_probability=0.40,
    )
    assert result["rule_based_ev_status"] == "CONSERVATIVE_FIXED_PRIOR"
    assert result["expected_net_value_bps"] > 0
    assert result["cost_to_target_2_ratio"] < 0.25
    assert result["edge_cost_ratio_target_2"] > 4.0
    assert result["may_raise_risk_caps"] is False


def test_deduplication_keeps_best_market_event_and_snapshot() -> None:
    base = {
        "market": "BTC-EUR",
        "detected_at": "2026-08-04T15:04:00+00:00",
        "state": "ENTRY_READY",
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit_1": 102.0,
        "take_profit_2": 103.0,
        "realtime_inputs": {"ofi_1m": 0.1},
        "execution_economics": {"positive_after_costs": True},
    }
    rows = deduplicate_opportunities(
        [
            {
                **base,
                "opportunity_id": "low",
                "playbook_id": "A",
                "score": 60.0,
            },
            {
                **base,
                "opportunity_id": "high",
                "playbook_id": "B",
                "score": 70.0,
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["opportunity_id"] == "high"
    assert rows[0]["cluster_size"] == 2
    assert rows[0]["economic_bet_id"] == rows[0]["cluster_id"]
    assert rows[0]["decision_trace"]["deduplicated_decision"] == (
        "SELECTED_CLUSTER_WINNER"
    )
    assert rows[0]["decision_trace"]["duplicates_suppressed"] == 1
    assert rows[0]["feature_snapshot"]["feature_schema_version"] == (
        FEATURE_SCHEMA_VERSION
    )


def test_feature_snapshot_is_deterministic() -> None:
    row = {
        "market": "ETH-EUR",
        "family": "BREAKOUT",
        "detected_at": "2026-08-04T15:00:00+00:00",
        "score": 70.0,
        "realtime_inputs": {"ofi_1m": 0.2},
    }
    assert freeze_feature_snapshot(row)["feature_hash"] == (
        freeze_feature_snapshot(row)["feature_hash"]
    )


def test_feature_snapshot_records_explicit_point_in_time_provenance() -> None:
    snapshot = freeze_feature_snapshot(
        {
            "market": "ETH-EUR",
            "context_timeframe": "15m",
            "detected_at": "2026-08-04T15:00:01+00:00",
            "market_event_ts": "2026-08-04T15:00:00+00:00",
            "last_updated_at": "2026-08-04T15:00:01+00:00",
            "realtime_inputs": {"ofi_1m": 0.2},
            "synthetic_data_used": False,
        }
    )

    assert snapshot["event_time"] == "2026-08-04T15:00:00+00:00"
    assert snapshot["available_at"] == "2026-08-04T15:00:01+00:00"
    assert snapshot["is_final"] is True
    assert snapshot["point_in_time_timing_valid"] is True
    assert snapshot["provenance"]["synthetic_data_used"] is False


def test_feature_snapshot_preserves_realtime_observation_timeframe() -> None:
    snapshot = freeze_feature_snapshot(
        {
            "market": "ETH-EUR",
            "detected_at": "2026-08-04T15:00:01+00:00",
            "market_event_ts": "2026-08-04T15:00:00+00:00",
            "observation_timeframe": "1m",
            "realtime_inputs": {"ofi_1m": 0.2},
            "synthetic_data_used": False,
        }
    )

    assert snapshot["values"]["context_timeframe"] is None
    assert snapshot["values"]["observation_timeframe"] == "1m"


def _canonical_training_row(index: int) -> dict[str, object]:
    decision = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=index)
    return {
        "cluster_id": f"cluster-{index}",
        "signal_id": f"signal-{index}",
        "decision_timestamp": decision.isoformat(),
        "event_time": (decision - timedelta(minutes=1)).isoformat(),
        "available_at": decision.isoformat(),
        "is_final": True,
        "provenance": {"producer": "prospective-test-collector"},
        "canonical_point_in_time_ready": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": f"{index:064x}",
        "features": {
            "market": "BTC-EUR" if index % 2 else "ETH-EUR",
            "context_timeframe": "15m",
            "family": "BREAKOUT",
            "score": 50.0 + index % 20,
        },
        "net_profitable": index % 2,
        "net_return_r": 1.0 if index % 2 else -1.0,
        "label_uses_future_features": False,
        "label_start": decision.isoformat(),
        "label_end": (decision + timedelta(hours=24)).isoformat(),
        "evidence_source": "SHADOW_CLOSED_CANDLE",
    }


def test_canonical_dataset_excludes_legacy_rows_without_backfill(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    result = build_canonical_ml_dataset(
        settings,
        rows=[
            _canonical_training_row(0),
            {
                "decision_timestamp": "2026-08-04T00:00:00Z",
                "features": {"market": "BTC-EUR", "context_timeframe": "15m"},
                "net_profitable": 1,
            },
        ],
    )

    assert result["status"] == "REGISTERED_RESEARCH_ONLY"
    assert result["canonical_row_count"] == 1
    assert result["excluded_row_count"] == 1
    assert result["historical_metadata_backfilled"] is False
    assert Path(result["manifest"]).is_file()
    assert Path(result["rows_artifact"]).is_file()


def test_canonical_shadow_model_uses_purged_exact_splits_and_stays_shadow(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    result = train_canonical_shadow_models(
        settings,
        rows=[_canonical_training_row(index) for index in range(600)],
    )

    assert result["status"] == "REGISTERED_SHADOW_NO_LIVE_AUTHORITY"
    assert result["model_registered"] is True
    assert result["purged_walk_forward"] is True
    assert len(result["validation_folds"]) == 5
    assert result["promotion_evaluation"]["permitted"] is True
    assert result["live_decision_influence"] is False
    assert result["orders_submitted"] == 0
    for fold in result["validation_folds"]:
        assert fold["train_end"] < fold["validation_start"]
        assert fold["validation_end"] < fold["test_start"]


def test_training_dataset_links_fill_through_order_result_intent(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    opportunity_id = "prospective-opportunity"
    snapshot = freeze_feature_snapshot(
        {
            "opportunity_id": opportunity_id,
            "market": "ETH-EUR",
            "family": "LIQUIDITY_SWEEP_RECLAIM",
            "detected_at": "2026-08-04T15:00:00+00:00",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit_1": 103.0,
            "take_profit_2": 105.0,
        }
    )
    lifecycle = (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl"
    )
    lifecycle.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.write_text(
        json.dumps(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": opportunity_id,
                "feature_snapshot": snapshot,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paper = (
        settings.paths.output_dir
        / "paper"
        / "event_driven_playbook_execution.jsonl"
    )
    paper.parent.mkdir(parents=True, exist_ok=True)
    events = []
    for intent_id, side, price in (
        ("buy-intent", "BUY", "100"),
        ("sell-intent", "SELL", "104"),
    ):
        events.extend(
            [
                {
                    "event_type": "FILL",
                    "payload": {
                        "intent_id": intent_id,
                        "side": side,
                        "price": price,
                        "quantity": "1",
                        "fee_eur": "0.1",
                    },
                },
                {
                    "event_type": "ORDER_RESULT",
                    "payload": {
                        "record": {
                            "intent": {
                                "intent_id": intent_id,
                                "signal_id": opportunity_id,
                            }
                        }
                    },
                },
            ]
        )
    paper.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    result = build_training_dataset(settings)

    assert result["row_count"] == 1
    assert result["complete_labeled_episodes"] == 1
    assert result["incomplete_snapshotted_episodes"] == 0
    assert result["feature_coverage"]["market"]["coverage_fraction"] == 1.0
    assert result["weighted_context_collection"] == {
        "status": "PENDING_FIRST_POST_DEPLOYMENT_EPISODE",
        "snapshots_with_weighted_context": 0,
        "legacy_snapshots_without_weighted_context": 1,
        "historical_snapshots_backfilled": False,
        "reason": (
            "Decision-time features are immutable; pre-deployment snapshots "
            "are not rewritten with values that were not recorded at t0."
        ),
    }
    dataset = json.loads(
        Path(result["json_artifact"]).read_text(encoding="utf-8")
    )
    assert dataset["rows"][0]["signal_id"] == opportunity_id
    assert dataset["rows"][0]["net_profitable"] == 1


def test_live_training_label_uses_canonical_realized_pnl(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    opportunity_id = "live-canonical-opportunity"
    snapshot = freeze_feature_snapshot(
        {
            "opportunity_id": opportunity_id,
            "market": "ETH-EUR",
            "family": "LIQUIDITY_SWEEP_RECLAIM",
            "detected_at": "2026-08-04T15:00:00+00:00",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit_1": 103.0,
            "take_profit_2": 105.0,
        }
    )
    lifecycle = (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl"
    )
    lifecycle.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.write_text(
        json.dumps(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": opportunity_id,
                "feature_snapshot": snapshot,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-04T15:01:00Z",
            "payload": {
                "fill_id": "live-buy",
                "order_id": "buy-order",
                "intent_id": "buy-intent",
                "signal_id": opportunity_id,
                "strategy_id": "TEST_STRATEGY",
                "market": "ETH-EUR",
                "side": "BUY",
                "price": "100",
                "quantity": "1",
                "fee_eur": "0.1",
                "filled_at": "2026-08-04T15:01:00Z",
            },
        },
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-04T16:00:00Z",
            "payload": {
                "fill_id": "live-sell",
                "order_id": "sell-order",
                "intent_id": "sell-intent",
                "signal_id": opportunity_id,
                "strategy_id": "TEST_STRATEGY",
                "market": "ETH-EUR",
                "side": "SELL",
                "price": "104",
                "quantity": "1",
                "fee_eur": "0.1",
                "filled_at": "2026-08-04T16:00:00Z",
            },
        },
    ]
    ledger.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    result = build_training_dataset(settings)
    dataset = json.loads(Path(result["json_artifact"]).read_text(encoding="utf-8"))
    row = dataset["rows"][0]

    assert row["evidence_source"] == "LIVE_CANONICAL_EXECUTION_STATE"
    assert row["economic_label_source"] == (
        "CANONICAL_EXECUTION_STATE_REALIZED_PNL"
    )
    assert row["net_pnl_eur"] == 3.8
    assert len(row["canonical_state_hash"]) == 64


def test_training_dataset_labels_unfilled_shadow_episode_from_future_candles(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    opportunity_id = "shadow-opportunity"
    decision = datetime(2026, 8, 1, 12, tzinfo=UTC)
    snapshot = freeze_feature_snapshot(
        {
            "opportunity_id": opportunity_id,
            "market": "ETH-EUR",
            "family": "FAILED_BREAKDOWN_REVERSAL",
            "detected_at": decision.isoformat(),
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit_1": 103.0,
            "take_profit_2": 105.0,
            "execution_economics": {"roundtrip_cost_bps": 20.0},
        }
    )
    lifecycle = (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl"
    )
    lifecycle.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.write_text(
        json.dumps(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": opportunity_id,
                "feature_snapshot": snapshot,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = pd.date_range(
        decision + timedelta(minutes=15),
        periods=120,
        freq="15min",
        tz="UTC",
    )
    close = pd.Series(
        [100.0 + 8.0 * position / (len(index) - 1) for position in range(len(index))],
        index=index,
    )
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "high": close.to_numpy() + 0.10,
            "low": close.to_numpy() - 0.10,
            "close": close.to_numpy(),
        }
    )
    path = settings.paths.processed_data_dir / "ETH-EUR_15m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)

    result = build_training_dataset(settings)

    assert result["row_count"] == 1
    assert result["fill_labeled_episodes"] == 0
    assert result["shadow_labeled_episodes"] == 1
    dataset = json.loads(Path(result["json_artifact"]).read_text(encoding="utf-8"))
    row = dataset["rows"][0]
    assert row["evidence_source"] == "SHADOW_CLOSED_CANDLE"
    assert row["tp2_before_stop"] is True
    assert row["net_profitable"] == 1
    assert row["returns_by_horizon"]["24h"] > 0
    assert row["label_uses_future_features"] is False


def test_recent_canonical_snapshot_is_reported_as_awaiting_label_horizon(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    decision = datetime.now(UTC) - timedelta(minutes=5)
    opportunity_id = "canonical-label-pending"
    snapshot = freeze_feature_snapshot(
        {
            "opportunity_id": opportunity_id,
            "market": "ETH-EUR",
            "context_timeframe": "15m",
            "family": "FAILED_BREAKDOWN_REVERSAL",
            "detected_at": decision.isoformat(),
            "market_event_ts": (decision - timedelta(seconds=1)).isoformat(),
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit_1": 103.0,
            "take_profit_2": 105.0,
        }
    )
    lifecycle = (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl"
    )
    lifecycle.parent.mkdir(parents=True, exist_ok=True)
    lifecycle.write_text(
        json.dumps(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": opportunity_id,
                "feature_snapshot": snapshot,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_training_dataset(settings)

    assert result["row_count"] == 0
    assert result["canonical_feature_ready_incomplete_count"] == 1
    assert result["canonical_pending_label_horizon_count"] == 1
    assert result["canonical_label_horizon_mature_unresolved_count"] == 0
    assert result["canonical_incomplete_missing_timeframe_count"] == 0
    assert result["next_canonical_label_due_at"] is not None

    canonical = build_canonical_ml_dataset(settings)
    assert canonical["status"] == "DATA_PENDING"
    assert canonical["canonical_pending_label_horizon_count"] == 1
    assert canonical["reason"] == (
        "PROSPECTIVE_POINT_IN_TIME_ROWS_AWAITING_24H_LABEL_HORIZON"
    )


def test_shadow_training_fails_safe_with_small_sample(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    result = train_shadow_models(
        settings,
        rows=[
            {
                "decision_timestamp": f"2026-08-04T00:{index:02d}:00Z",
                "net_profitable": index % 2,
                "features": {"score": 50 + index},
            }
            for index in range(10)
        ],
    )
    assert result["status"] == "DATA_PENDING"
    assert result["authority"] == "SHADOW_ONLY"
    assert result["live_decision_influence"] is False
    assert result["orders_submitted"] == 0


def test_one_hundred_rows_are_pipeline_smoke_not_promotable_intelligence(
    isolated_settings: Settings, tmp_path
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    result = train_shadow_models(
        settings,
        rows=[
            {
                "decision_timestamp": (
                    datetime(2026, 1, 1, tzinfo=UTC)
                    + timedelta(minutes=index)
                ).isoformat(),
                "net_profitable": index % 2,
                "features": {
                    "score": 45 + index % 30,
                    "market": "BTC-EUR" if index % 2 else "ETH-EUR",
                },
            }
            for index in range(100)
        ],
    )
    assert result["status"] == "PIPELINE_SMOKE_ONLY"
    assert result["promotion_evaluation_ready"] is False
    assert result["promotion_gate"]["all_gates_passed"] is False
    assert result["live_decision_influence"] is False


def test_bayesian_weighting_cannot_raise_risk_caps() -> None:
    rows = bayesian_playbook_weights(
        [
            {
                "family": "BREAKOUT",
                "timeframe": "15m",
                "regime": "RISK_ON",
                "net_return_r": 1.0,
            }
        ]
    )
    assert rows[0]["ranking_multiplier"] <= 1.0
    assert rows[0]["can_raise_risk_caps"] is False


def test_soft_exit_requires_persistence_and_price_confirmation() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=UTC)
    position = {"entry_price": "100"}
    realtime = {"windows": {"1m": {"return": -0.001}}}
    assert not _soft_exit_confirmed(
        position,
        reason="ORDERFLOW_EXHAUSTION",
        realtime=realtime,
        price=Decimal("99.9"),
        now=now,
    )
    assert not _soft_exit_confirmed(
        position,
        reason="ORDERFLOW_EXHAUSTION",
        realtime=realtime,
        price=Decimal("99.9"),
        now=now + timedelta(seconds=5),
    )
    assert _soft_exit_confirmed(
        position,
        reason="ORDERFLOW_EXHAUSTION",
        realtime=realtime,
        price=Decimal("99.9"),
        now=now + timedelta(seconds=11),
    )


def test_soft_exit_persistence_adapts_to_slow_thin_higher_timeframe() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=UTC)
    position = {"entry_price": "100", "context_timeframe": "4h"}
    realtime = {
        "windows": {"1m": {"return": -0.001, "trade_intensity": 0.5}},
        "book": {"spread_bps": 20, "bid_depth_eur_top_10": 100},
    }
    for seconds in (0, 6, 12):
        assert not _soft_exit_confirmed(
            position,
            reason="ORDERFLOW_EXHAUSTION",
            realtime=realtime,
            price=Decimal("99.9"),
            now=now + timedelta(seconds=seconds),
        )
    assert _soft_exit_confirmed(
        position,
        reason="ORDERFLOW_EXHAUSTION",
        realtime=realtime,
        price=Decimal("99.9"),
        now=now + timedelta(seconds=21),
    )
    assert position["soft_exit_required_seconds"] == 20.0


def test_fast_market_soft_exit_still_requires_ten_seconds() -> None:
    now = datetime(2026, 8, 4, 15, 0, tzinfo=UTC)
    position = {"entry_price": "100", "context_timeframe": "15m"}
    realtime = {
        "windows": {"1m": {"return": -0.001, "trade_intensity": 5.0}},
        "book": {"spread_bps": 2, "bid_depth_eur_top_10": 5_000},
    }
    for seconds in (0, 4, 9):
        assert not _soft_exit_confirmed(
            position,
            reason="ORDERFLOW_EXHAUSTION",
            realtime=realtime,
            price=Decimal("99.9"),
            now=now + timedelta(seconds=seconds),
        )
    assert _soft_exit_confirmed(
        position,
        reason="ORDERFLOW_EXHAUSTION",
        realtime=realtime,
        price=Decimal("99.9"),
        now=now + timedelta(seconds=11),
    )
    assert position["soft_exit_required_seconds"] == 10.0
