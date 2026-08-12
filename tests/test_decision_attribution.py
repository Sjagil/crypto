from __future__ import annotations

import json
from pathlib import Path

from core.decision_attribution import build_decision_execution_attribution


def test_decision_attribution_uses_only_canonical_fills_and_redacts_ids(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "output" / "checkpoints"
    checkpoint.mkdir(parents=True)
    events = [
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-08-01T10:00:01Z",
            "payload": {
                "intent_id": "secret-intent-buy",
                "signal_id": "signal-1",
                "strategy_id": "TEST_V1",
                "order_type": "LIMIT",
                "time_in_force": "GTC",
                "reason_codes": ["CAUSAL_ENTRY"],
            },
        },
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-01T10:00:03Z",
            "payload": {
                "intent_id": "secret-intent-buy",
                "fill_id": "secret-fill-buy",
                "order_id": "secret-order-buy",
                "client_order_id": "secret-client-buy",
                "signal_id": "signal-1",
                "strategy_id": "TEST_V1",
                "strategy_dna_hash": "dna-1",
                "market": "BTC-EUR",
                "side": "BUY",
                "price": "101",
                "quantity": "2",
                "fee_eur": "0.2",
                "fee_known": True,
            },
        },
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-08-01T11:00:00Z",
            "payload": {
                "intent_id": "secret-intent-sell",
                "signal_id": "signal-1",
                "strategy_id": "TEST_V1",
                "order_type": "LIMIT",
                "time_in_force": "IOC",
                "reason_codes": ["STRATEGY_EXIT"],
            },
        },
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-01T11:00:01Z",
            "payload": {
                "intent_id": "secret-intent-sell",
                "fill_id": "secret-fill-sell",
                "order_id": "secret-order-sell",
                "signal_id": "signal-1",
                "strategy_id": "TEST_V1",
                "strategy_dna_hash": "dna-1",
                "market": "BTC-EUR",
                "side": "SELL",
                "price": "103",
                "quantity": "2",
                "fee_eur": "0.2",
                "fee_known": True,
                "reason_codes": ["STRATEGY_EXIT"],
            },
        },
    ]
    (checkpoint / "live_execution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )
    live = tmp_path / "output" / "live"
    live.mkdir(parents=True)
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "signal-1": {
                        "playbook_id": "TEST_V1",
                        "playbook_dna": "dna-1",
                        "family": "TEST_FAMILY",
                        "context_timeframe": "15m",
                        "macro_regime": "RECOVERY",
                        "entry_price": 100,
                        "score": 72,
                        "feature_snapshot": {
                            "decision_timestamp": "2026-08-01T10:00:00Z",
                            "feature_hash": "feature-hash",
                            "feature_schema_version": "features-v1",
                            "values": {"entry_price": 100},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_decision_execution_attribution(tmp_path)

    assert result["status"] == "READY"
    assert result["trade_count"] == 1
    assert result["closed_round_trips"] == 1
    trade = result["trades"][0]
    assert trade["position_status"] == "CLOSED"
    assert trade["decision_price"] == "100"
    assert trade["entry_slippage_bps"] == "100.00"
    assert trade["entry_price_shortfall_eur"] == "2"
    assert trade["realized_gross_pnl_eur"] == "4"
    assert trade["realized_net_pnl_eur"] == "3.6"
    assert trade["submission_to_first_fill_seconds"] == 2.0
    assert trade["decision_to_first_fill_seconds"] == 3.0
    assert trade["decision_context"]["feature_hash"] == "feature-hash"
    serialized = json.dumps(result)
    for forbidden in (
        "secret-intent-buy",
        "secret-fill-buy",
        "secret-order-buy",
        "secret-client-buy",
        "signal-1",
    ):
        assert forbidden not in serialized
    assert result["privacy"]["exchange_order_ids_serialized"] is False
    cached = build_decision_execution_attribution(tmp_path)
    assert cached["artifact"].endswith(
        "decision_execution_attribution.json"
    )
    assert result["orders_submitted"] == 0


def test_missing_independent_decision_price_is_not_zero_slippage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "output" / "checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint / "live_execution.jsonl").write_text(
        json.dumps(
            {
                "event_type": "FILL",
                "recorded_at": "2026-08-01T10:00:03Z",
                "payload": {
                    "intent_id": "intent",
                    "signal_id": "generated-signal",
                    "strategy_id": "GENERATED_V1",
                    "strategy_dna_hash": "dna-generated",
                    "market": "ETH-EUR",
                    "side": "BUY",
                    "price": "1500",
                    "quantity": "0.01",
                    "fee_eur": "0.03",
                    "fee_known": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    live = tmp_path / "output" / "live"
    live.mkdir(parents=True)
    (live / "generated_strategy_live_state.json").write_text(
        json.dumps(
            {
                "positions": {
                    "dna-generated": {
                        "signal_id": "generated-signal",
                        "strategy_id": "GENERATED_V1",
                        "strategy_dna_hash": "dna-generated",
                        "timeframe": "2h",
                        "entry_price": "1500",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_decision_execution_attribution(tmp_path)
    trade = result["trades"][0]

    assert trade["position_status"] == "OPEN"
    assert trade["decision_price"] is None
    assert trade["entry_slippage_bps"] is None
    assert trade["decision_price_status"] == (
        "UNAVAILABLE_INDEPENDENT_PRE_FILL_REFERENCE"
    )
    assert result["decision_price_mapping_ratio"] == 0.0


def test_point_in_time_execution_timing_uses_exchange_and_receive_clocks(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "output" / "checkpoints"
    checkpoint.mkdir(parents=True)
    events = [
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-08-01T10:00:01.100Z",
            "payload": {
                "intent_id": "intent-timing",
                "signal_id": "signal-timing",
                "strategy_id": "TIMING_V1",
                "submission_started_at": "2026-08-01T10:00:01.000Z",
                "order_type": "LIMIT",
                "time_in_force": "IOC",
            },
        },
        {
            "event_type": "ORDER_ACKNOWLEDGED",
            "recorded_at": "2026-08-01T10:00:01.500Z",
            "payload": {
                "intent_id": "intent-timing",
                "acknowledgement_received_at": (
                    "2026-08-01T10:00:01.400Z"
                ),
                "exchange_created_at": "2026-08-01T10:00:01.200Z",
            },
        },
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-01T10:00:03.000Z",
            "payload": {
                "intent_id": "intent-timing",
                "signal_id": "signal-timing",
                "strategy_id": "TIMING_V1",
                "strategy_dna_hash": "dna-timing",
                "market": "ETH-EUR",
                "side": "BUY",
                "price": "100",
                "quantity": "0.1",
                "fee_eur": "0.025",
                "fee_known": True,
                "filled_at": "2026-08-01T10:00:02.500Z",
                "received_at": "2026-08-01T10:00:02.800Z",
            },
        },
    ]
    (checkpoint / "live_execution.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )
    live = tmp_path / "output" / "live"
    live.mkdir(parents=True)
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "signal-timing": {
                        "playbook_id": "TIMING_V1",
                        "playbook_dna": "dna-timing",
                        "context_timeframe": "15m",
                        "feature_snapshot": {
                            "decision_timestamp": "2026-08-01T10:00:00Z",
                            "values": {"entry_price": 99},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_decision_execution_attribution(tmp_path)
    trade = result["trades"][0]

    assert trade["decision_to_first_fill_seconds"] == 2.5
    assert trade["submission_to_first_fill_seconds"] == 1.5
    assert trade["submission_to_ack_seconds"] == 0.4
    assert trade["exchange_fill_to_local_receive_seconds"] == 0.3
    assert trade["timestamp_evidence"] == {
        "decision_at_available": True,
        "submission_started_at_source": "EXPLICIT",
        "acknowledgement_received_at_source": "EXPLICIT",
        "fill_exchange_at_source": "EXCHANGE_EVENT",
        "fill_received_at_source": "EXPLICIT",
        "acknowledgement_recovered": False,
    }
    assert result["timing_coverage"]["explicit_submission_count"] == 1
    assert result["timing_coverage"]["explicit_exchange_fill_count"] == 1
