from __future__ import annotations

import json
from decimal import Decimal

from core.live_strategy_accounting import rebuild_live_strategy_accounting
from utils.common import append_jsonl, atomic_write_json


def _approval(tmp_path) -> None:
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "live_strategy_approvals.yaml").write_text(
        """
version: 1
default_policy: FAIL_CLOSED
strategies:
  TEST_STRATEGY:
    strategy_dna_hash: dna-1
    strategy_family: TEST_FAMILY
    timeframe: 1h
    approved_markets: [BTC-EUR]
    approved_for_live: true
    maximum_order_eur: 5
    maximum_total_exposure_eur: 10
    autoscale: false
""".strip(),
        encoding="utf-8",
    )


def test_live_strategy_accounting_rebuilds_virtual_strategy_lot(tmp_path) -> None:
    _approval(tmp_path)
    ledger = tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    append_jsonl(
        ledger,
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-07-30T00:00:00Z",
            "payload": {
                "intent_id": "intent-1",
                "client_order_id": "client-1",
                "idempotency_key": "idem-1",
                "strategy_id": "TEST_STRATEGY",
                "strategy_dna_hash": "dna-1",
                "signal_id": "signal-1",
                "portfolio_decision_id": "decision-1",
            },
        },
    )
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-07-30T00:01:00Z",
            "payload": {
                "fill_id": "fill-buy",
                "order_id": "order-buy",
                "client_order_id": "client-1",
                "market": "BTC-EUR",
                "side": "BUY",
                "quantity": "1",
                "price": "100",
                "fee_eur": "1",
                "filled_at": "2026-07-30T00:01:00Z",
            },
        },
    )
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-07-30T01:01:00Z",
            "payload": {
                "fill_id": "fill-sell",
                "order_id": "order-sell",
                "strategy_id": "TEST_STRATEGY",
                "strategy_dna_hash": "dna-1",
                "signal_id": "signal-1",
                "portfolio_decision_id": "decision-2",
                "client_order_id": "client-2",
                "market": "BTC-EUR",
                "side": "SELL",
                "quantity": "1",
                "price": "110",
                "fee_eur": "1",
                "filled_at": "2026-07-30T01:01:00Z",
            },
        },
    )

    result = rebuild_live_strategy_accounting(
        tmp_path,
        ledger_path=ledger,
        price_by_market={"BTC-EUR": Decimal("110")},
    )

    assert result["integrity_status"] == "PASSED"
    strategy = result["strategies"][0]
    assert strategy["strategy_id"] == "TEST_STRATEGY"
    assert strategy["closed_trade_count"] == 1
    assert strategy["open_trade_count"] == 0
    assert strategy["realised_pnl_eur"] == "8"
    assert strategy["fees_paid_eur"] == "2"
    assert strategy["unique_signal_count"] == 1
    assert strategy["unique_portfolio_decision_count"] == 2
    assert strategy["last_closed_trade"] == {
        "market": "BTC-EUR",
        "opened_at": "2026-07-30T00:01:00Z",
        "closed_at": "2026-07-30T01:01:00Z",
        "holding_seconds": 3600,
        "entry_price_eur": "100",
        "exit_price_eur": "110",
        "quantity": "1",
        "fees_eur": "2",
        "net_pnl_eur": "8",
        "average_slippage_bps": None,
    }

    decisions = (
        tmp_path
        / "output"
        / "live"
        / "strategy_accounting_decisions.jsonl"
    )
    first_count = len(decisions.read_text(encoding="utf-8").splitlines())
    rebuild_live_strategy_accounting(
        tmp_path,
        ledger_path=ledger,
        price_by_market={"BTC-EUR": Decimal("110")},
    )
    assert len(decisions.read_text(encoding="utf-8").splitlines()) == first_count


def test_live_strategy_accounting_fails_closed_on_unattributed_fill(tmp_path) -> None:
    _approval(tmp_path)
    ledger = tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-07-30T00:01:00Z",
            "payload": {
                "fill_id": "fill-without-strategy",
                "order_id": "order-1",
                "market": "BTC-EUR",
                "side": "BUY",
                "quantity": "1",
                "price": "100",
                "fee_eur": "1",
                "filled_at": "2026-07-30T00:01:00Z",
            },
        },
    )

    result = rebuild_live_strategy_accounting(tmp_path, ledger_path=ledger)

    assert result["integrity_status"] == "FAILED"
    assert "UNATTRIBUTED_LIVE_FILLS" in result["hard_blockers"]
    assert result["unattributed_fill_count"] == 1
    stored = json.loads(
        (
            tmp_path / "output" / "live" / "strategy_accounts.json"
        ).read_text(encoding="utf-8")
    )
    assert stored["secrets_serialized"] is False


def test_operator_inventory_fill_is_excluded_from_strategy_accounting(
    tmp_path,
) -> None:
    _approval(tmp_path)
    ledger = tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    append_jsonl(
        ledger,
        {
            "event_type": "ORDER_INTENT",
            "recorded_at": "2026-07-30T00:00:00Z",
            "payload": {
                "intent_id": "inventory-intent",
                "client_order_id": "inventory-client",
                "strategy_id": (
                    "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE"
                ),
            },
        },
    )
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-07-30T00:01:00Z",
            "payload": {
                "fill_id": "inventory-fill",
                "client_order_id": "inventory-client",
                "market": "NPC-EUR",
                "side": "SELL",
                "quantity": "100",
                "price": "0.005",
                "fee_eur": "0.001",
            },
        },
    )

    result = rebuild_live_strategy_accounting(tmp_path, ledger_path=ledger)

    assert result["integrity_status"] == "PASSED"
    assert result["hard_blockers"] == []
    assert result["excluded_non_strategy_fill_count"] == 1
    assert result["excluded_non_strategy_fill_ids"] == ["inventory-fill"]
    assert not any(
        row["strategy_id"]
        == "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE"
        for row in result["strategies"]
    )


def test_generated_positive_authority_is_included_in_strategy_books(
    tmp_path,
) -> None:
    authority = (
        tmp_path
        / "output"
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    atomic_write_json(
        authority,
        {
            "schema_version": "positive_strategy_live_authority_v1",
            "active": True,
            "maximum_order_eur": "5",
            "maximum_total_exposure_eur": "15",
            "approved_candidates": [
                {
                    "strategy_id": "EXACT_POSITIVE_TEST",
                    "strategy_dna_hash": "a" * 64,
                    "timeframe": "4h",
                    "approved_markets": ["BTC-EUR"],
                }
            ],
        },
    )

    result = rebuild_live_strategy_accounting(tmp_path)

    strategy = next(
        row
        for row in result["strategies"]
        if row["strategy_id"] == "EXACT_POSITIVE_TEST"
    )
    assert strategy["operator_approved"] is True
    assert strategy["strategy_dna"] == "a" * 64
    assert strategy["allocated_capital_eur"] == "5"
    assert strategy["allocation_mode"] == "SHARED_PORTFOLIO_CAP"
    assert strategy["shared_portfolio_cap_eur"] == "15"


def test_declared_playbook_identity_migration_preserves_historical_fills(
    tmp_path,
) -> None:
    current_dna = "c" * 64
    previous_dna = "b" * 64
    atomic_write_json(
        tmp_path / "config" / "live_playbook_authority.json",
        {
            "schema_version": "event_driven_playbook_authority_v1",
            "active": True,
            "approved_playbooks": [
                {
                    "active": True,
                    "playbook_id": "MIGRATED_PLAYBOOK",
                    "playbook_dna": current_dna,
                    "previous_playbook_dna": previous_dna,
                    "execution_timeframes": ["15m"],
                    "markets": ["BTC-EUR"],
                    "maximum_order_eur": "10",
                    "maximum_total_exposure_eur": "10",
                }
            ],
        },
    )
    ledger = tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-01T00:00:00Z",
            "payload": {
                "fill_id": "legacy-buy",
                "strategy_id": "MIGRATED_PLAYBOOK",
                "strategy_dna_hash": previous_dna,
                "market": "BTC-EUR",
                "side": "BUY",
                "quantity": "1",
                "price": "100",
                "fee_eur": "1",
            },
        },
    )
    append_jsonl(
        ledger,
        {
            "event_type": "FILL",
            "recorded_at": "2026-08-01T01:00:00Z",
            "payload": {
                "fill_id": "legacy-sell",
                "strategy_id": "MIGRATED_PLAYBOOK",
                "strategy_dna_hash": previous_dna,
                "market": "BTC-EUR",
                "side": "SELL",
                "quantity": "1",
                "price": "110",
                "fee_eur": "1",
            },
        },
    )

    result = rebuild_live_strategy_accounting(tmp_path, ledger_path=ledger)

    assert result["integrity_status"] == "PASSED"
    account = next(
        row
        for row in result["strategies"]
        if row["strategy_id"] == "MIGRATED_PLAYBOOK"
    )
    assert account["strategy_dna"] == current_dna
    assert account["closed_trade_count"] == 1
    assert account["realised_pnl_eur"] == "8"
