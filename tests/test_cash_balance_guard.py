from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from core.cash_balance_guard import evaluate_eur_cash_continuity
from core.daily_profit_target import record_external_capital_flow


def _settings(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
        )
    )


def test_unexplained_cash_change_remains_blocked_until_operator_records_flow(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    first_at = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
    first = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="100",
        observed_at=first_at,
    )
    assert first["status"] == "BASELINE_ESTABLISHED"

    second_at = first_at + timedelta(minutes=10)
    unexplained = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="65",
        observed_at=second_at,
    )
    assert unexplained["status"] == "UNEXPLAINED_EUR_BALANCE_CHANGE"
    assert unexplained["new_entries_blocked"] is True
    assert unexplained["protective_exits_allowed"] is True
    assert unexplained["accepted_eur_available"] == "100"
    assert unexplained["orders_submitted"] == 0

    record_external_capital_flow(
        settings,
        amount_eur="-35",
        reason="TRANSFER",
        effective_at=first_at + timedelta(minutes=5),
        note="operator-confirmed test transfer",
    )
    resolved = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="65",
        observed_at=second_at,
    )
    assert resolved["status"] == "READY_EXPLAINED_CHANGE"
    assert resolved["new_entries_blocked"] is False
    assert resolved["accepted_eur_available"] == "65"


def test_complete_exchange_history_explains_external_cash_change(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    first_at = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
    evaluate_eur_cash_continuity(
        settings,
        current_eur_available="100",
        observed_at=first_at,
    )

    resolved = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="65",
        observed_at=first_at + timedelta(minutes=10),
        exchange_external_cash_delta_eur="-35",
        exchange_history_complete=True,
    )

    assert resolved["status"] == "READY_EXPLAINED_CHANGE"
    assert resolved["exchange_external_cash_delta_eur"] == "-35"
    assert resolved["exchange_transaction_history_complete"] is True
    assert resolved["new_entries_blocked"] is False


def test_private_external_trade_fills_explain_cash_but_do_not_adopt_inventory(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    first_at = datetime(2026, 8, 11, 10, 14, 48, tzinfo=UTC)
    evaluate_eur_cash_continuity(
        settings,
        current_eur_available="268.06",
        observed_at=first_at,
    )
    fills = tmp_path / "output" / "live" / "events" / "fills.jsonl"
    fills.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event": "BITVAVO_ACCOUNT_FILL",
            "event_id": "external-buy",
            "recorded_at": "2026-08-11T10:14:50Z",
            "payload": {
                "client_order_public_id": None,
                "fill_price": "7.4846",
                "amount": "35.72535606",
                "fee": "0.670000033324",
                "fee_currency": "EUR",
                "side": "BUY",
            },
        },
        {
            "event": "BITVAVO_ACCOUNT_FILL",
            "event_id": "external-sell",
            "recorded_at": "2026-08-11T11:03:25Z",
            "payload": {
                "client_order_public_id": None,
                "fill_price": "7.5175",
                "amount": "13.33555039",
                "fee": "0.260000056825",
                "fee_currency": "EUR",
                "side": "SELL",
            },
        },
        {
            "event": "BITVAVO_ACCOUNT_FILL",
            "event_id": "bot-fill-mirror",
            "recorded_at": "2026-08-11T11:04:00Z",
            "payload": {
                "client_order_public_id": "client_masked",
                "fill_price": "7.5",
                "amount": "1",
                "fee": "0.02",
                "fee_currency": "EUR",
                "side": "BUY",
            },
        },
    ]
    fills.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    resolved = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="99.99",
        observed_at=first_at + timedelta(hours=2),
        exchange_history_complete=True,
    )

    assert resolved["status"] == "READY_EXPLAINED_CHANGE"
    assert resolved["external_account_fill_cash_delta_eur"] == (
        "-168.070000000000"
    )
    assert resolved["external_account_fill_event_ids"] == [
        "external-buy",
        "external-sell",
    ]
    assert resolved["external_fills_do_not_adopt_inventory_or_grant_exit_authority"] is True
    assert resolved["new_entries_blocked"] is False

    stable = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="99.99",
        observed_at=first_at + timedelta(hours=2, minutes=1),
        exchange_history_complete=True,
    )
    assert stable["status"] == "READY_STABLE"
    assert stable["external_account_fill_cash_delta_eur"] == "0"


def test_guard_bootstraps_prior_distinct_reconciliation_snapshot(tmp_path) -> None:
    settings = _settings(tmp_path)
    events = tmp_path / "output" / "live" / "events"
    events.mkdir(parents=True)
    rows = [
        {
            "event": "ACCOUNT_RECONCILIATION",
            "recorded_at": "2026-08-02T10:00:00Z",
            "account": {"eur_available": "100"},
        },
        {
            "event": "ACCOUNT_RECONCILIATION",
            "recorded_at": "2026-08-02T10:10:00Z",
            "account": {"eur_available": "65"},
        },
    ]
    (events / "positions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="65",
        observed_at=datetime(2026, 8, 2, 10, 11, tzinfo=UTC),
    )

    assert result["bootstrapped_from_reconciliation_events"] is True
    assert result["accepted_eur_available"] == "100"
    assert result["pending_unexplained_eur_available"] == "65"
    assert result["new_entries_blocked"] is True


def test_canonical_live_fill_explains_cash_change_without_mirror(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.paths.checkpoints_dir = tmp_path / "output" / "checkpoints"
    first_at = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
    evaluate_eur_cash_continuity(
        settings,
        current_eur_available="20.32",
        observed_at=first_at,
    )
    settings.paths.checkpoints_dir.mkdir(parents=True)
    (settings.paths.checkpoints_dir / "live_execution.jsonl").write_text(
        json.dumps(
            {
                "event_type": "FILL",
                "recorded_at": "2026-08-02T10:05:00+00:00",
                "payload": {
                    "market": "BTC-EUR",
                    "side": "BUY",
                    "price": "55541",
                    "quantity": "0.00018004",
                    "fee_eur": "0.02039836",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="10.30",
        observed_at=first_at + timedelta(minutes=10),
    )

    assert resolved["status"] == "READY_EXPLAINED_CHANGE"
    assert resolved["canonical_fill_cash_delta_eur"] == "-10.02000000"
    assert resolved["new_entries_blocked"] is False


def test_fill_just_before_stale_cash_baseline_is_reconciled_once(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.paths.checkpoints_dir = tmp_path / "output" / "checkpoints"
    settings.paths.checkpoints_dir.mkdir(parents=True)
    fill_at = datetime(2026, 8, 7, 18, 4, 15, tzinfo=UTC)
    (settings.paths.checkpoints_dir / "live_execution.jsonl").write_text(
        json.dumps(
            {
                "event_type": "FILL",
                "recorded_at": fill_at.isoformat(),
                "payload": {
                    "fill_id": "fill-race-1",
                    "market": "ETH-EUR",
                    "side": "BUY",
                    "price": "1654.42",
                    "quantity": "0.00604379",
                    "fee_eur": "0.0310329482",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # The account snapshot is stale for a moment and still reports pre-fill
    # cash even though the canonical fill has already been persisted.
    evaluate_eur_cash_continuity(
        settings,
        current_eur_available="278.46",
        observed_at=fill_at + timedelta(seconds=2),
    )
    resolved = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="268.43",
        observed_at=fill_at + timedelta(seconds=20),
    )

    assert resolved["status"] == "READY_EXPLAINED_CHANGE"
    assert resolved["new_entries_blocked"] is False
    assert resolved["fill_settlement_lookback_applied"] is True
    assert resolved["canonical_fill_event_ids"] == ["fill-race-1"]
    assert "fill-race-1" in resolved["consumed_canonical_fill_ids"]

    stable = evaluate_eur_cash_continuity(
        settings,
        current_eur_available="268.43",
        observed_at=fill_at + timedelta(seconds=30),
    )
    assert stable["status"] == "READY_STABLE"
    assert stable["canonical_fill_cash_delta_eur"] == "0"
