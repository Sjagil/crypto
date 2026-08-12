from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config.settings import PathSettings, Settings
from core.opportunity_audit import build_daily_opportunity_audit


def test_daily_audit_explains_significant_missed_move(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(
        "2026-08-03T00:00:00Z",
        periods=6,
        freq="15min",
    )
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 1.01, 1.02, 1.04, 1.06, 1.08],
            "high": [1.01, 1.02, 1.04, 1.06, 1.09, 1.10],
            "low": [0.99, 1.0, 1.01, 1.03, 1.05, 1.07],
            "close": [1.0, 1.01, 1.03, 1.05, 1.08, 1.09],
        }
    ).to_parquet(settings.paths.processed_data_dir / "ADA-EUR_15m.parquet")
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        json.dumps(
            {
                "recorded_at": "2026-08-03T00:40:00+00:00",
                "opportunity_id": "ada-1",
                "market": "ADA-EUR",
                "to_state": "ARMED",
                "hard_blockers": ["INSUFFICIENT_REALTIME_CONFIRMATIONS"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "ada-1": {
                        "market": "ADA-EUR",
                        "state": "ARMED",
                        "hard_blockers": [
                            "INSUFFICIENT_REALTIME_CONFIRMATIONS"
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    assert report["significant_move_count"] == 1
    assert report["opportunities_detected"] == 1
    assert report["orders_submitted"] == 0
    assert report["missed_move_rate"] == 1.0
    missed = report["significant_moves"][0]
    assert missed["market"] == "ADA-EUR"
    assert missed["maximum_move_pct"] == 10.0
    assert missed["furthest_lifecycle_state"] == "ARMED"
    assert missed["miss_reason"] == "INSUFFICIENT_REALTIME_CONFIRMATIONS"
    counterfactual = missed["opportunity_outcomes"][0]["counterfactual"]
    assert counterfactual["status"] == "COUNTERFACTUAL_NOT_A_FILL"
    assert counterfactual["theoretical_entry_price"] == 1.05
    assert counterfactual["maximum_favorable_move_after_detection_pct"] > 4.0
    assert report["trade_cadence"]["windows"]["6h"][
        "raw_opportunities"
    ] == 1
    assert report["trade_cadence"]["windows"]["6h"]["diagnosis"].startswith(
        "ENTRY_GATE_BOTTLENECK:"
    )
    cadence = report["trade_cadence"]["windows"]["6h"]
    assert "near_entry_to_ready_rate" not in cadence
    assert cadence["conversion_metric_semantics"] == {
        "populations_nested": False,
        "operational_decision_eligible": False,
        "reason": (
            "entry_ready and near_entry are independent lifecycle observations "
            "inside the cadence window; this ratio may exceed 1 and is not a "
            "conversion probability"
        ),
    }
    assert report["trade_cadence"]["minimum_daily_trade_quota"] is None
    assert report["trigger_leakage"]["rejected_opportunity_count"] == 1
    assert report["trigger_leakage"]["automatic_threshold_changes"] is False
    assert (
        settings.paths.output_dir
        / "operations"
        / "daily_opportunity_audit.json"
    ).is_file()


def test_trade_cadence_excludes_operator_inventory_and_strategy_exits(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        json.dumps(
            {
                "recorded_at": "2026-08-03T01:00:00Z",
                "opportunity_id": "real-signal",
                "market": "ADA-EUR",
                "to_state": "ENTRY_READY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps({"opportunities": {}}),
        encoding="utf-8",
    )
    settings.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "recorded_at": "2026-08-03T01:01:00Z",
            "event_type": "ORDER_ACKNOWLEDGED",
            "payload": {
                "intent_id": "inventory",
                "market": "TAO-EUR",
                "side": "SELL",
                "strategy_id": "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE",
            },
        },
        {
            "recorded_at": "2026-08-03T01:02:00Z",
            "event_type": "ORDER_ACKNOWLEDGED",
            "payload": {
                "intent_id": "entry",
                "signal_id": "real-signal",
                "market": "ADA-EUR",
                "side": "BUY",
                "strategy_id": "TEST_STRATEGY",
            },
        },
        {
            "recorded_at": "2026-08-03T01:03:00Z",
            "event_type": "FILL",
            "payload": {
                "intent_id": "entry",
                "signal_id": "real-signal",
                "market": "ADA-EUR",
                "side": "BUY",
                "strategy_id": "TEST_STRATEGY",
            },
        },
        {
            "recorded_at": "2026-08-03T01:04:00Z",
            "event_type": "FILL",
            "payload": {
                "intent_id": "exit",
                "signal_id": "real-signal",
                "market": "ADA-EUR",
                "side": "SELL",
                "strategy_id": "TEST_STRATEGY",
            },
        },
    ]
    (settings.paths.checkpoints_dir / "live_execution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    cadence = report["trade_cadence"]["windows"]["6h"]
    assert cadence["orders_submitted"] == 1
    assert cadence["fills"] == 1
    assert cadence["execution_count_scope"] == "LIVE_STRATEGY_BUY_ENTRIES_ONLY"
    assert report["opportunity_utilization"] == {
        "scope": "DAILY_CAUSAL_ENTRY_READY_AFTER_COSTS",
        "economically_valid_opportunities": 1,
        "executed_valid_opportunities": 1,
        "ratio": 1.0,
        "trade_quota_enforced": False,
    }


def test_daily_audit_reports_current_authority_leakage(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    events = settings.paths.output_dir / "live" / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        json.dumps(
            {
                "recorded_at": "2026-08-03T01:00:00Z",
                "opportunity_id": "candidate-1",
                "market": "ADA-EUR",
                "to_state": "ENTRY_READY",
                "hard_blockers": [],
                "execution_economics": {"positive_after_costs": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (settings.paths.output_dir / "live" / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "candidate-1": {
                        "opportunity_id": "candidate-1",
                        "market": "ADA-EUR",
                        "family": "UNAPPROVED",
                        "state": "ENTRY_READY",
                        "hard_blockers": [],
                        "execution_economics": {
                            "positive_after_costs": True
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    assert report["authority_leakage"]["blocked_only_by_missing_authority"] == 1
    assert report["authority_leakage"]["ratio"] == 1.0
    assert report["authority_leakage"]["automatic_authority_changes"] is False


def test_daily_authority_leakage_keeps_paper_filled_candidate_visible(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    events = settings.paths.output_dir / "live" / "events"
    events.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "recorded_at": "2026-08-03T01:00:00Z",
            "opportunity_id": "paper-candidate",
            "market": "CC-EUR",
            "to_state": "ENTRY_READY",
            "hard_blockers": [],
            "execution_economics": {"positive_after_costs": True},
        },
        {
            "recorded_at": "2026-08-03T01:01:00Z",
            "opportunity_id": "paper-candidate",
            "market": "CC-EUR",
            "to_state": "FILLED",
            "reason_codes": ["PAPER_ONLY"],
            "details": {"paper_event": {"event": "PAPER_POSITION_OPENED"}},
        },
    ]
    (events / "opportunity_lifecycle.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    live = settings.paths.output_dir / "live"
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "paper-candidate": {
                        "opportunity_id": "paper-candidate",
                        "market": "CC-EUR",
                        "playbook_id": "MOMENTUM_BREAKOUT_V1",
                        "family": "MOMENTUM_BREAKOUT",
                        "playbook_dna": "unapproved-dna",
                        "parameter_band_hash": "unapproved-band",
                        "state": "FILLED",
                        "hard_blockers": [],
                        "execution_economics": {
                            "positive_after_costs": True
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("CC-EUR",),
    )

    leakage = report["authority_leakage"]
    assert leakage["economically_valid_candidates"] == 1
    assert leakage["blocked_only_by_missing_authority"] == 1
    assert leakage["ratio"] == 1.0
    assert leakage["current_projection"]["economically_valid_candidates"] == 0
    assert leakage["candidates"] == [
        {
            "opportunity_id": "paper-candidate",
            "market": "CC-EUR",
            "playbook_id": "MOMENTUM_BREAKOUT_V1",
            "family": "MOMENTUM_BREAKOUT",
            "projection_state": "FILLED",
            "paper_lifecycle_is_not_live_execution": True,
        }
    ]
    assert leakage["automatic_authority_changes"] is False


def test_daily_audit_attributes_rejected_gate_counterfactual_after_costs(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-08-03T00:00:00Z", periods=6, freq="15min")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 6,
            "high": [1.0, 1.01, 1.03, 1.06, 1.08, 1.09],
            "low": [0.995, 0.997, 1.0, 1.02, 1.04, 1.06],
            "close": [1.0, 1.005, 1.025, 1.05, 1.07, 1.08],
        }
    ).to_parquet(settings.paths.processed_data_dir / "ADA-EUR_15m.parquet")
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        json.dumps(
            {
                "recorded_at": "2026-08-03T00:15:00Z",
                "opportunity_id": "ada-cost-reject",
                "market": "ADA-EUR",
                "to_state": "DISCOVERED",
                "hard_blockers": [
                    "CONSERVATIVE_EXPECTED_VALUE_NOT_POSITIVE"
                ],
                "execution_economics": {
                    "net_target_1_bps": 200.0,
                    "roundtrip_cost_bps": 40.0,
                    "stop_bps": 100.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "ada-cost-reject": {
                        "market": "ADA-EUR",
                        "entry_price": 1.0,
                        "stop_loss": 0.99,
                        "take_profit_1": 1.02,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    attribution = report["gate_counterfactual_attribution"]["ECR_COST"]
    assert attribution["rejected_opportunity_count"] == 1
    assert attribution["resolved_boundary_count"] == 1
    assert attribution["tp1_before_stop_count"] == 1
    assert attribution["counterfactual_mean_net_return_pct"] == 2.0
    assert round(attribution["counterfactual_mean_net_r"], 6) == round(
        200 / 140,
        6,
    )
    assert attribution["assessment"] == (
        "INSUFFICIENT_RESOLVED_COUNTERFACTUALS"
    )
    assert report["gate_counterfactual_method"]["execution_claim"] is False


def test_daily_audit_does_not_count_paper_fill_as_live_conversion(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-08-03T00:00:00Z", periods=6, freq="15min")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 6,
            "high": [1.0, 1.01, 1.03, 1.06, 1.09, 1.1],
            "low": [0.99] * 6,
            "close": [1.0, 1.01, 1.03, 1.06, 1.08, 1.09],
        }
    ).to_parquet(settings.paths.processed_data_dir / "ADA-EUR_15m.parquet")
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "recorded_at": "2026-08-03T00:40:00+00:00",
            "opportunity_id": "ada-paper",
            "market": "ADA-EUR",
            "to_state": "ENTRY_READY",
        },
        {
            "recorded_at": "2026-08-03T00:41:00+00:00",
            "opportunity_id": "ada-paper",
            "market": "ADA-EUR",
            "to_state": "FILLED",
            "reason_codes": ["PAPER_POSITION_OPENED", "PAPER_ONLY"],
            "details": {"paper_event": {"event": "PAPER_POSITION_OPENED"}},
        },
        {
            "recorded_at": "2026-08-03T00:42:00+00:00",
            "opportunity_id": "ada-stale-sibling",
            "market": "ADA-EUR",
            "to_state": "INVALIDATED",
            "hard_blockers": ["STALE_REALTIME_DATA"],
        },
    ]
    (events / "opportunity_lifecycle.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps({"opportunities": {"ada-paper": {"market": "ADA-EUR"}}}),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    assert report["orders_submitted"] == 0
    assert report["opportunities_filled"] == 0
    assert report["paper_opportunities_filled"] == 1
    move = report["significant_moves"][0]
    assert move["paper_converted"] is True
    assert move["converted_to_order"] is False
    assert move["furthest_live_lifecycle_state"] == "ENTRY_READY"
    assert move["miss_reason"] == "PLAYBOOK_AUTHORITY_MISSING"


def test_daily_audit_separates_processing_latency_from_market_move_delay(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-08-03T00:00:00Z", periods=8, freq="15min")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 8,
            "high": [1.0, 1.01, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
            "low": [0.99] * 8,
            "close": [1.0, 1.01, 1.035, 1.04, 1.05, 1.06, 1.07, 1.08],
        }
    ).to_parquet(settings.paths.processed_data_dir / "ADA-EUR_15m.parquet")
    events = settings.paths.output_dir / "live" / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "recorded_at": "2026-08-03T00:00:00Z",
                    "opportunity_id": "coverage-marker",
                    "market": "BTC-EUR",
                    "to_state": "INVALIDATED",
                },
                {
                    "recorded_at": "2026-08-03T01:00:00.250000Z",
                    "opportunity_id": "ada-latency",
                    "market": "ADA-EUR",
                    "to_state": "INVALIDATED",
                    "reason_codes": ["NO_STALE_ENTRY_REUSE"],
                    "feature_snapshot": {
                        "decision_timestamp": "2026-08-03T01:00:00Z"
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    live = settings.paths.output_dir / "live"
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps({"opportunities": {}}),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    move = report["significant_moves"][0]
    assert move["detection_latency_seconds"] == 0.25
    assert move["move_to_first_opportunity_seconds"] > 0
    assert move["miss_reason"] == "SETUP_NEVER_REACHED_ENTRY_READY"
    assert report["average_detection_latency_seconds"] == 0.25


def test_daily_audit_reconciles_closed_paper_pnl_and_slippage(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-08-03T00:00:00Z", periods=6, freq="15min")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 6,
            "high": [1.0, 1.02, 1.05, 1.08, 1.1, 1.12],
            "low": [0.99, 1.0, 1.01, 1.04, 1.07, 1.09],
            "close": [1.0, 1.01, 1.04, 1.07, 1.09, 1.11],
        }
    ).to_parquet(settings.paths.processed_data_dir / "ADA-EUR_15m.parquet")
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "opportunity_lifecycle.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "recorded_at": "2026-08-03T00:20:00Z",
                    "opportunity_id": "ada-paper-closed",
                    "market": "ADA-EUR",
                    "to_state": "ENTRY_READY",
                },
                {
                    "recorded_at": "2026-08-03T00:21:00Z",
                    "opportunity_id": "ada-paper-closed",
                    "market": "ADA-EUR",
                    "to_state": "FILLED",
                    "reason_codes": ["PAPER_ONLY"],
                },
                {
                    "recorded_at": "2026-08-03T00:50:00Z",
                    "opportunity_id": "ada-paper-closed",
                    "market": "ADA-EUR",
                    "to_state": "CLOSED",
                    "reason_codes": ["PAPER_ONLY"],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "ada-paper-closed": {
                        "market": "ADA-EUR",
                        "playbook_id": "TEST_PLAYBOOK",
                        "playbook_dna": "dna",
                        "macro_regime": "RISK_ON",
                        "context_timeframe": "15m",
                        "entry_price": 1.0,
                        "stop_loss": 0.98,
                        "take_profit_1": 1.05,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    paper = settings.paths.output_dir / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    ledger_rows = [
        {
            "event_type": "ORDER_RESULT",
            "payload": {
                "record": {
                    "intent": {
                        "intent_id": "buy-intent",
                        "signal_id": "ada-paper-closed",
                        "strategy_id": "TEST_PLAYBOOK",
                        "strategy_dna_hash": "dna",
                        "reason_codes": ["PAPER_ONLY"],
                    }
                }
            },
        },
        {
            "event_type": "FILL",
            "payload": {
                "intent_id": "buy-intent",
                "market": "ADA-EUR",
                "side": "BUY",
                "price": "1.001",
                "quantity": "5",
                "fee_eur": "0.01",
            },
        },
        {
            "event_type": "ORDER_RESULT",
            "payload": {
                "record": {
                    "intent": {
                        "intent_id": "sell-intent",
                        "signal_id": "ada-paper-closed",
                        "strategy_id": "TEST_PLAYBOOK",
                        "strategy_dna_hash": "dna",
                        "reason_codes": ["TAKE_PROFIT_1", "PAPER_ONLY"],
                    }
                }
            },
        },
        {
            "event_type": "FILL",
            "payload": {
                "intent_id": "sell-intent",
                "market": "ADA-EUR",
                "side": "SELL",
                "price": "1.05",
                "quantity": "5",
                "fee_eur": "0.01",
            },
        },
    ]
    (paper / "event_driven_playbook_execution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )

    report = build_daily_opportunity_audit(
        settings,
        observed_at=datetime(2026, 8, 3, 2, tzinfo=UTC),
        markets=("ADA-EUR",),
    )

    assert report["paper_execution_evidence"]["closed_round_trips"] == 1
    attribution = report["paper_execution_evidence"]["by_playbook"][
        "TEST_PLAYBOOK"
    ]
    assert attribution["closed_round_trips"] == 1
    assert attribution["winning_round_trips"] == 1
    assert attribution["win_rate"] == 1.0
    assert attribution["paper_net_expectancy_eur"] == 0.225
    assert attribution["closed_position_profit_factor"] is None
    dna_attribution = report["paper_execution_evidence"]["by_playbook_dna"][
        "dna"
    ]
    assert dna_attribution["playbook_id"] == "TEST_PLAYBOOK"
    assert dna_attribution["closed_round_trips"] == 1
    assert dna_attribution["paper_net_expectancy_eur"] == 0.225
    assert report["paper_net_expectancy_eur"] == 0.225
    assert report["average_slippage_bps"] == 10.0
    assert report["pnl_by_playbook"] == {"TEST_PLAYBOOK": 0.225}
    assert report["pnl_by_regime"] == {"RISK_ON": 0.225}
    assert report["pnl_by_timeframe"] == {"15m": 0.225}
