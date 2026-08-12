from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.cli import build_parser
from core.live_asset_preflight import (
    _exchange_external_eur_delta,
    _market_exceptions,
    _public_market_projection,
    _safe_balance_projection,
    _transaction_history_window,
    live_asset_preflight,
)
from utils.common import atomic_write_json


def test_public_market_projection_contains_only_execution_metadata() -> None:
    projected = _public_market_projection(
        {
            "market": "TAO-EUR",
            "status": "trading",
            "minOrderInBaseAsset": "0.02",
            "minOrderInQuoteAsset": "5",
            "pricePrecision": 5,
            "amountPrecision": 6,
            "orderTypes": ["market", "limit"],
            "private": "must-not-be-projected",
        }
    )
    assert projected["venue_available"] is True
    assert projected["minimum_order_quote"] == "5"
    assert "private" not in projected


@pytest.mark.asyncio
async def test_tao_npc_preflight_fails_closed_without_dna_or_signal(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    universe = output / "universe"
    universe.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    atomic_write_json(
        universe / "top50_eligibility.json",
        {
            "source_snapshot_hash": "snapshot",
            "rows": [
                {
                    "rank": 34,
                    "symbol": "TAO",
                    "eur_spot_market": "TAO-EUR",
                    "execution_eligibility": "NOT_EXECUTION_ELIGIBLE",
                    "execution_reason": "SHARIAH_REVIEW_REQUIRED",
                }
            ],
        },
    )

    async def fake_metadata(_settings):
        return {
            "TAO-EUR": {
                "market": "TAO-EUR",
                "status": "trading",
                "minOrderInQuoteAsset": "5",
            },
            "NPC-EUR": {
                "market": "NPC-EUR",
                "status": "trading",
                "minOrderInQuoteAsset": "5",
            },
        }

    monkeypatch.setattr(
        "core.live_asset_preflight._public_bitvavo_market_metadata",
        fake_metadata,
    )
    settings = SimpleNamespace(
        paths=SimpleNamespace(
            project_root=tmp_path,
            output_dir=output,
        )
    )
    payload = await live_asset_preflight(
        settings,
        markets=("TAO-EUR", "NPC-EUR"),
    )
    assert payload["status"] == "BLOCKED"
    assert payload["privacy_and_authority"]["private_exchange_requests"] == 0
    assert payload["privacy_and_authority"]["orders_submitted"] == 0
    tao, npc = payload["markets"]
    assert "SHARIAH_REVIEW_REQUIRED" in tao["blockers"]
    assert "NO_APPROVED_STRATEGY_DNA_FOR_MARKET" in tao["blockers"]
    assert (
        "NOT_IN_POINT_IN_TIME_TOP50_EXECUTION_UNIVERSE"
        in npc["blockers"]
    )


@pytest.mark.asyncio
async def test_preflight_recognizes_event_playbook_authority_but_keeps_signal_gate(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    universe = output / "universe"
    live = output / "live"
    config = tmp_path / "config"
    universe.mkdir(parents=True)
    live.mkdir(parents=True)
    config.mkdir()
    atomic_write_json(
        universe / "top50_eligibility.json",
        {
            "rows": [
                {
                    "rank": 34,
                    "symbol": "TAO",
                    "eur_spot_market": "TAO-EUR",
                    "execution_eligibility": "LIVE_ELIGIBLE",
                    "execution_reason": "PASSED",
                }
            ]
        },
    )
    atomic_write_json(
        config / "live_playbook_authority.json",
        {
            "active": True,
            "maximum_order_eur": 10,
            "approved_playbooks": [
                {
                    "active": True,
                    "playbook_id": "FAILED_BREAKDOWN_REVERSAL_V1",
                    "playbook_dna": "a" * 64,
                    "execution_timeframes": ["15m", "1h"],
                    "markets": ["TAO-EUR"],
                }
            ],
        },
    )
    atomic_write_json(
        live / "autonomous_live_authority.json",
        {"active": True, "markets": ["TAO-EUR"]},
    )

    async def fake_metadata(_settings):
        return {
            "TAO-EUR": {
                "market": "TAO-EUR",
                "status": "trading",
                "minOrderInQuoteAsset": "5",
            }
        }

    monkeypatch.setattr(
        "core.live_asset_preflight._public_bitvavo_market_metadata",
        fake_metadata,
    )
    settings = SimpleNamespace(
        paths=SimpleNamespace(project_root=tmp_path, output_dir=output)
    )
    payload = await live_asset_preflight(settings, markets=("TAO-EUR",))
    row = payload["markets"][0]

    assert row["matching_operator_authority"] is True
    assert row["matching_authority_type"] == "EVENT_PLAYBOOK"
    assert row["approved_playbooks"][0]["playbook_id"] == (
        "FAILED_BREAKDOWN_REVERSAL_V1"
    )
    assert "NO_APPROVED_STRATEGY_DNA_FOR_MARKET" not in row["blockers"]
    assert row["blockers"] == ["NO_NATURAL_APPROVED_STRATEGY_SIGNAL"]


def test_live_asset_preflight_cli_is_registered() -> None:
    args = build_parser().parse_args(
        ["live", "asset-preflight", "--markets", "TAO-EUR,NPC-EUR"]
    )
    assert args.command == "live"
    assert args.live_command == "asset-preflight"
    assert args.markets == "TAO-EUR,NPC-EUR"


def test_live_account_health_cli_is_registered() -> None:
    args = build_parser().parse_args(
        ["live", "account-health", "--markets", "ETH-EUR,TAO-EUR"]
    )
    assert args.command == "live"
    assert args.live_command == "account-health"
    assert args.markets == "ETH-EUR,TAO-EUR"


def test_safe_balance_projection_exposes_no_private_fields() -> None:
    projected = _safe_balance_projection(
        [
            {
                "symbol": "EUR",
                "available": "50.00",
                "inOrder": "0",
                "accountId": "must-not-leak",
            },
            {
                "symbol": "TAO",
                "available": "0.2",
                "inOrder": "0.1",
                "apiKey": "must-not-leak",
            },
            {
                "symbol": "ZERO",
                "available": "0",
                "inOrder": "0",
            },
        ]
    )
    assert projected["eur_available"] == "50.00"
    assert projected["non_eur_holding_count"] == 1
    assert projected["non_eur_holdings"] == [
        {
            "symbol": "TAO",
            "available": "0.2",
            "in_order": "0.1",
            "total": "0.3",
        }
    ]
    serialized = str(projected)
    assert "accountId" not in serialized
    assert "apiKey" not in serialized
    assert "must-not-leak" not in serialized


def test_transaction_projection_uses_only_external_eur_movements() -> None:
    result = _exchange_external_eur_delta(
        [
            {
                "transactionId": "deposit-secret-id",
                "executedAt": "2026-08-01T10:00:00Z",
                "type": "deposit",
                "sentCurrency": "",
                "sentAmount": "0",
                "receivedCurrency": "EUR",
                "receivedAmount": "30",
                "feesCurrency": "EUR",
                "feesAmount": "0",
                "address": "must-not-leak",
            },
            {
                "transactionId": "trade-id",
                "type": "buy",
                "sentCurrency": "EUR",
                "sentAmount": "10",
                "receivedCurrency": "BTC",
                "receivedAmount": "0.0001",
            },
        ]
    )
    assert result["external_eur_delta"] == "30"
    assert result["matched_external_transactions"] == 1
    serialized = str(result)
    assert "deposit-secret-id" not in serialized
    assert "must-not-leak" not in serialized


def test_transaction_history_window_treats_tiny_concurrent_skew_as_empty() -> None:
    now = datetime(2026, 8, 7, 13, 36, 58, tzinfo=UTC)
    current_ms = int(now.timestamp() * 1_000)

    assert _transaction_history_window(
        now + timedelta(milliseconds=250),
        current_ms=current_ms,
        maximum_clock_drift_ms=5_000,
    ) is None


def test_transaction_history_window_rejects_material_future_baseline() -> None:
    now = datetime(2026, 8, 7, 13, 36, 58, tzinfo=UTC)
    current_ms = int(now.timestamp() * 1_000)

    with pytest.raises(ValueError, match="materially in the future"):
        _transaction_history_window(
            now + timedelta(seconds=10),
            current_ms=current_ms,
            maximum_clock_drift_ms=5_000,
        )


def test_operator_market_exceptions_keep_strategy_and_signal_gates(
    tmp_path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "execution_market_exceptions.yaml").write_text(
        """
version: 1
default_policy: FAIL_CLOSED
markets:
  NPC-EUR:
    approved: true
    approved_at: "2026-07-28T00:00:00+02:00"
    approval_reference: explicit_test
    allow_outside_top50: true
    spot_only: true
    maximum_order_eur: 5.0
    maximum_total_exposure_eur: 10.0
    requires_approved_strategy_dna: true
    requires_natural_signal: true
""".strip(),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        paths=SimpleNamespace(project_root=tmp_path)
    )
    exception = _market_exceptions(settings)["NPC-EUR"]
    assert exception["allow_outside_top50"] is True
    assert exception["requires_approved_strategy_dna"] is True
    assert exception["requires_natural_signal"] is True
