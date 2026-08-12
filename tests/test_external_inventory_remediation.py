from __future__ import annotations

import json
from types import SimpleNamespace

from core.cli import build_parser
from reporting.external_inventory_remediation import (
    build_external_inventory_migration_contract,
    build_external_inventory_remediation,
    verify_external_inventory_migration_contract,
)
from utils.common import atomic_write_json


def _settings(tmp_path):
    return SimpleNamespace(paths=SimpleNamespace(output_dir=tmp_path / "output"))


def test_external_inventory_plan_is_orderless_and_exact(tmp_path) -> None:
    settings = _settings(tmp_path)
    live = settings.paths.output_dir / "live"
    events = live / "events"
    events.mkdir(parents=True)
    atomic_write_json(
        live / "generated_strategy_live_state.json",
        {
            "positions": {
                "dna": {
                    "market": "LINK-EUR",
                    "status": "OPEN",
                    "quantity": "1.39429176",
                    "native_protective_stop_active": True,
                    "protective_stop_status": "awaitingTrigger",
                    "protective_stop_trigger": "7.01840",
                }
            }
        },
    )
    rows = [
        {
            "event": "BITVAVO_ACCOUNT_FILL",
            "event_id": "buy",
            "payload": {
                "market": "LINK-EUR",
                "side": "BUY",
                "amount": "35.72535606",
                "fill_price": "7.4846",
                "fee": "0.670000033324",
                "fee_currency": "EUR",
                "client_order_public_id": None,
            },
        },
        {
            "event": "BITVAVO_ACCOUNT_FILL",
            "event_id": "sell",
            "payload": {
                "market": "LINK-EUR",
                "side": "SELL",
                "amount": "13.33555039",
                "fill_price": "7.5175",
                "fee": "0.260000056825",
                "fee_currency": "EUR",
                "client_order_public_id": None,
            },
        },
    ]
    (events / "fills.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    health = {
        "managed_position_protection_eligible": True,
        "account": {
            "portfolio_heat": {
                "inventory_classification": [
                    {
                        "market": "LINK-EUR",
                        "classification": "MANAGED_POSITION",
                        "total_quantity": "23.78409743",
                        "managed_quantity": "1.39429176",
                        "external_quantity": "22.38980567",
                    }
                ]
            },
            "portfolio_valuation": {
                "holdings": [{"market": "LINK-EUR", "price_eur": "7.5"}]
            },
        },
    }

    result = build_external_inventory_remediation(settings, health)

    position = result["positions"][0]
    assert result["status"] == "OPERATOR_DECISION_REQUIRED"
    assert result["orders_generated"] == result["orders_submitted"] == 0
    assert result["external_exit_authority_granted"] is False
    assert position["external_fill_totals"] == {
        "net_quantity": "22.38980567",
        "eur_cash_delta": "-168.070000000000",
    }
    assert position["fill_evidence_matches_external_quantity"] is True
    assert position["managed_protection"]["managed_quantity"] == "1.39429176"
    assert position["external_quantity_has_bot_exit_authority"] is False


def test_external_inventory_plan_cli_is_registered() -> None:
    args = build_parser().parse_args(
        ["live", "external-inventory-plan", "--markets", "LINK-EUR"]
    )
    assert args.command == "live"
    assert args.live_command == "external-inventory-plan"
    assert args.markets == "LINK-EUR"


def test_external_inventory_migration_draft_is_fail_closed(tmp_path) -> None:
    settings = _settings(tmp_path)
    remediation = {
        "positions": [
            {
                "market": "LINK-EUR",
                "external_quantity": "22.38980567",
                "managed_quantity": "1.39429176",
                "mark_price_eur": "7.5",
                "external_fill_totals": {"net_quantity": "22.38980567"},
                "fill_evidence_matches_external_quantity": True,
            }
        ]
    }

    contract = build_external_inventory_migration_contract(
        settings, remediation, market="LINK-EUR"
    )

    assert contract["status"] == "DRAFT_OPERATOR_INPUT_REQUIRED"
    assert contract["external_quantity"] == "22.38980567"
    assert contract["managed_quantity_excluded"] == "1.39429176"
    assert contract["external_exit_authority_granted"] is False
    assert contract["activation_implemented"] is False
    assert contract["orders_generated"] == contract["orders_submitted"] == 0
    assert "EXPLICIT_MIGRATION_DECISION_MISSING" in contract["verification"]["failures"]


def test_external_inventory_migration_detects_quantity_drift(tmp_path) -> None:
    settings = _settings(tmp_path)
    remediation = {
        "positions": [
            {
                "market": "LINK-EUR",
                "external_quantity": "2",
                "managed_quantity": "1",
                "mark_price_eur": "10",
                "external_fill_totals": {"net_quantity": "2"},
                "fill_evidence_matches_external_quantity": True,
            }
        ]
    }
    contract = build_external_inventory_migration_contract(
        settings, remediation, market="LINK-EUR"
    )
    remediation["positions"][0]["external_quantity"] = "3"

    verification = verify_external_inventory_migration_contract(contract, remediation)

    assert "EXTERNAL_QUANTITY_CHANGED" in verification["failures"]
    assert "REMEDIATION_SNAPSHOT_CHANGED" in verification["failures"]
    assert verification["authority_granted"] is False


def test_external_inventory_migration_contract_cli_is_registered() -> None:
    args = build_parser().parse_args(
        ["live", "external-inventory-migration-contract", "--market", "LINK-EUR"]
    )

    assert args.live_command == "external-inventory-migration-contract"
    assert args.market == "LINK-EUR"
