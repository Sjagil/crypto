from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pytest

from config.settings import PathSettings, Settings
from core.live_capital import (
    LiveEntryReservationBusy,
    capital_level_2_capacity,
    live_entry_reservation,
    managed_live_portfolio,
    submit_level_2_buy_atomically,
)


def _settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _write_state(path: Path, positions: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"positions": positions}), encoding="utf-8")


def test_level_2_capacity_is_shared_across_live_engines(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "generated-eth": {
                "market": "ETH-EUR",
                "status": "OPEN",
                "quantity": "0.01",
                "entry_price": "2500",
            }
        },
    )
    _write_state(
        live / "event_driven_execution_state.json",
        {
            "event-btc": {
                "market": "BTC-EUR",
                "status": "ENTRY_PENDING",
                "quantity": "0.00025",
                "entry_price": "100000",
            }
        },
    )

    snapshot = managed_live_portfolio(settings)
    allowed, reason, _ = capital_level_2_capacity(
        settings,
        requested_notional_eur=Decimal("25"),
    )

    assert snapshot["managed_position_count"] == 2
    assert snapshot["managed_exposure_eur"] == "50.00000"
    assert snapshot["current_position_count"] == 1
    assert snapshot["current_position_exposure_eur"] == "25.00"
    assert snapshot["pending_entry_order_count"] == 1
    assert snapshot["pending_entry_exposure_eur"] == "25.00000"
    assert {
        row["market"]: row["exposure_class"] for row in snapshot["positions"]
    } == {
        "ETH-EUR": "CURRENT_POSITION",
        "BTC-EUR": "PENDING_ENTRY",
    }
    assert allowed is True
    assert reason == "CAPITAL_LEVEL_2_CAPACITY_AVAILABLE"


def test_level_2_shared_position_and_exposure_caps_fail_closed(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "one": {
                "market": "ETH-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "25",
            },
            "two": {
                "market": "SOL-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "25",
            },
        },
    )
    _write_state(
        live / "event_driven_execution_state.json",
        {
            "three": {
                "market": "BTC-EUR",
                "status": "EXIT_PENDING",
                "quantity": "1",
                "entry_price": "25",
            }
        },
    )

    allowed, reason, snapshot = capital_level_2_capacity(
        settings,
        requested_notional_eur=Decimal("1"),
    )

    assert allowed is False
    assert reason == "MANAGED_POSITION_LIMIT_REACHED"
    assert snapshot["managed_exposure_eur"] == "75"


def test_managing_event_position_reserves_shared_capacity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(live / "generated_strategy_live_state.json", {})
    _write_state(
        live / "event_driven_execution_state.json",
        {
            "managed-event-position": {
                "market": "SOL-EUR",
                "status": "MANAGING",
                "quantity": "2",
                "entry_price": "12.50",
            }
        },
    )

    snapshot = managed_live_portfolio(settings)

    assert snapshot["managed_position_count"] == 1
    assert snapshot["current_position_count"] == 1
    assert snapshot["pending_entry_order_count"] == 0
    assert snapshot["managed_exposure_eur"] == "25.00"
    assert snapshot["positions"][0]["exposure_class"] == "CURRENT_POSITION"


@pytest.mark.parametrize(
    ("root_status", "exposure_class"),
    [
        ("OPEN", "CURRENT_POSITION"),
        ("OPEN_PENDING_RECONCILIATION", "PENDING_ENTRY"),
        ("PARTIALLY_REDUCED", "CURRENT_POSITION"),
        ("EXIT_PENDING_RECONCILIATION", "CURRENT_POSITION"),
    ],
)
def test_primary_rr_position_is_in_shared_level_2_portfolio(
    isolated_settings: Settings,
    tmp_path: Path,
    root_status: str,
    exposure_class: str,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_state(
        settings.paths.output_dir
        / "live"
        / "generated_strategy_live_state.json",
        {},
    )
    _write_state(
        settings.paths.output_dir
        / "live"
        / "event_driven_execution_state.json",
        {},
    )
    rr_path = settings.paths.output_dir / "reports" / "current_position.json"
    rr_path.parent.mkdir(parents=True, exist_ok=True)
    rr_path.write_text(
        json.dumps(
            {
                "status": root_status,
                "position": {
                    "strategy_id": "RR_B60_H5_Z20",
                    "strategy_dna_hash": "4" * 64,
                    "market": "ETH-EUR",
                    "quantity": "0.01",
                    "entry_price": "1000",
                    "entry_client_order_id": "rr-client",
                    "entry_opportunity_id": "rr-signal",
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = managed_live_portfolio(settings)

    assert snapshot["managed_position_count"] == 1
    assert snapshot["managed_exposure_eur"] == "10.00"
    assert snapshot["positions"][0]["source"] == "RR_PRIMARY"
    assert snapshot["positions"][0]["exposure_class"] == exposure_class
    assert snapshot["positions"][0]["client_order_public_id"]
    assert "client_order_id" not in snapshot["positions"][0]


@pytest.mark.parametrize("ack_status", [None, "new", "partiallyFilled"])
def test_orphaned_canonical_buy_intent_reserves_crash_window_exposure(
    isolated_settings: Settings,
    tmp_path: Path,
    ack_status: str | None,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "eth": {
                "market": "ETH-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "25",
            },
            "sol": {
                "market": "SOL-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "25",
            },
        },
    )
    _write_state(live / "event_driven_execution_state.json", {})
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_type": "ORDER_INTENT",
            "payload": {
                "client_order_id": "orphan-client",
                "signal_id": "orphan-signal",
                "market": "BTC-EUR",
                "side": "BUY",
                "quantity": "0.00025",
                "limit_price": "100000",
                "maximum_notional_eur": "25",
            },
        }
    ]
    if ack_status is not None:
        events.append(
            {
                "event_type": "ORDER_ACKNOWLEDGED",
                "payload": {
                    "client_order_id": "orphan-client",
                    "order_id": "orphan-order",
                    "status": ack_status,
                },
            }
        )
    ledger.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    allowed, reason, snapshot = capital_level_2_capacity(
        settings,
        requested_notional_eur=Decimal("1"),
    )

    assert snapshot["status"] == "READY"
    assert snapshot["ledger_recovered_pending_order_count"] == 1
    assert snapshot["private_order_identifiers_serialized"] is False
    assert "client_order_id" not in snapshot["positions"][-1]
    assert len(snapshot["positions"][-1]["client_order_public_id"]) == 20
    assert snapshot["pending_entry_exposure_eur"] == "25"
    assert snapshot["managed_exposure_eur"] == "75"
    assert snapshot["managed_position_count"] == 3
    assert allowed is False
    assert reason == "MANAGED_POSITION_LIMIT_REACHED"


def test_engine_state_deduplicates_same_pending_ledger_intent(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "pending": {
                "market": "BTC-EUR",
                "status": "ENTRY_PENDING_RECONCILIATION",
                "quantity": "0.00025",
                "entry_price": "100000",
                "client_order_id": "same-client",
                "signal_id": "same-signal",
            }
        },
    )
    _write_state(live / "event_driven_execution_state.json", {})
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event_type": "ORDER_INTENT",
                    "payload": {
                        "client_order_id": "same-client",
                        "signal_id": "same-signal",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "quantity": "0.00025",
                        "limit_price": "100000",
                    },
                },
                {
                    "event_type": "ORDER_ACKNOWLEDGED",
                    "payload": {
                        "client_order_id": "same-client",
                        "order_id": "same-order",
                        "status": "new",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = managed_live_portfolio(settings)

    assert snapshot["managed_position_count"] == 1
    assert snapshot["managed_exposure_eur"] == "25.00000"
    assert snapshot["ledger_recovered_pending_order_count"] == 0


def test_terminal_canonical_buy_does_not_reserve_pending_capacity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(live / "generated_strategy_live_state.json", {})
    _write_state(live / "event_driven_execution_state.json", {})
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event_type": "ORDER_INTENT",
                    "payload": {
                        "client_order_id": "filled-client",
                        "signal_id": "filled-signal",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "quantity": "0.00025",
                        "limit_price": "100000",
                    },
                },
                {
                    "event_type": "ORDER_ACKNOWLEDGED",
                    "payload": {
                        "client_order_id": "filled-client",
                        "order_id": "filled-order",
                        "status": "filled",
                    },
                },
                {
                    "event_type": "FILL",
                    "payload": {
                        "client_order_id": "filled-client",
                        "order_id": "filled-order",
                        "status": "FILLED",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = managed_live_portfolio(settings)

    assert snapshot["managed_position_count"] == 0
    assert snapshot["ledger_recovered_pending_order_count"] == 0


def test_invalid_canonical_ledger_blocks_new_capacity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{invalid-json\n", encoding="utf-8")

    allowed, reason, snapshot = capital_level_2_capacity(
        settings,
        requested_notional_eur=Decimal("10"),
    )

    assert allowed is False
    assert reason == "PENDING_ORDER_EXPOSURE_UNRECONCILED"
    assert snapshot["status"] == "PENDING_ORDER_EXPOSURE_UNRECONCILED"


def test_live_entry_reservation_is_nonblocking_and_exclusive(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)

    with live_entry_reservation(settings):
        with pytest.raises(
            LiveEntryReservationBusy,
            match="LIVE_ENTRY_RESERVATION_BUSY",
        ):
            with live_entry_reservation(settings):
                pass

    with live_entry_reservation(settings):
        pass


@pytest.mark.asyncio
async def test_atomic_buy_reservation_prevents_concurrent_cap_overshoot(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "eth": {
                "market": "ETH-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "30",
            },
            "sol": {
                "market": "SOL-EUR",
                "status": "OPEN",
                "quantity": "1",
                "entry_price": "30",
            },
        },
    )
    _write_state(live / "event_driven_execution_state.json", {})
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    first_submitting = asyncio.Event()
    release_first = asyncio.Event()

    async def first_submit(_snapshot: object) -> dict[str, str]:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event_type": "ORDER_INTENT",
                        "payload": {
                            "client_order_id": "atomic-first",
                            "signal_id": "atomic-first-signal",
                            "market": "BTC-EUR",
                            "side": "BUY",
                            "quantity": "0.00015",
                            "limit_price": "100000",
                            "maximum_notional_eur": "15",
                        },
                    }
                )
                + "\n"
            )
        first_submitting.set()
        await release_first.wait()
        return {"status": "new"}

    first = asyncio.create_task(
        submit_level_2_buy_atomically(
            settings,
            requested_notional_eur=Decimal("15"),
            submit_order=first_submit,
        )
    )
    await first_submitting.wait()

    second_called = False

    async def second_submit(_snapshot: object) -> dict[str, str]:
        nonlocal second_called
        second_called = True
        return {"status": "new"}

    second_approved, second_reason, _, second_order = (
        await submit_level_2_buy_atomically(
            settings,
            requested_notional_eur=Decimal("15"),
            submit_order=second_submit,
        )
    )

    assert second_approved is False
    assert second_reason == "LIVE_ENTRY_RESERVATION_BUSY"
    assert second_order is None
    assert second_called is False

    release_first.set()
    first_approved, _, _, first_order = await first
    assert first_approved is True
    assert first_order == {"status": "new"}

    retry_approved, retry_reason, retry_snapshot, retry_order = (
        await submit_level_2_buy_atomically(
            settings,
            requested_notional_eur=Decimal("15"),
            submit_order=second_submit,
        )
    )
    assert retry_approved is False
    assert retry_reason == "MANAGED_POSITION_LIMIT_REACHED"
    assert retry_snapshot["managed_exposure_eur"] == "75"
    assert retry_order is None
    assert second_called is False


@pytest.mark.asyncio
async def test_atomic_reprice_replaces_exact_existing_reservation(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    live = settings.paths.output_dir / "live"
    _write_state(
        live / "generated_strategy_live_state.json",
        {
            "replace-me": {
                "market": "BTC-EUR",
                "status": "ENTRY_PENDING_RECONCILIATION",
                "quantity": "0.0002",
                "entry_price": "50000",
            },
            "keep-me": {
                "market": "ETH-EUR",
                "status": "OPEN",
                "quantity": "0.01",
                "entry_price": "1000",
            },
        },
    )
    _write_state(live / "event_driven_execution_state.json", {})
    submitted_snapshot: dict[str, object] = {}

    async def submit(snapshot: Mapping[str, Any]) -> dict[str, str]:
        submitted_snapshot.update(dict(snapshot))
        return {"status": "new"}

    approved, reason, snapshot, order = await submit_level_2_buy_atomically(
        settings,
        requested_notional_eur=Decimal("10"),
        submit_order=submit,
        replacing_source="GENERATED_DNA",
        replacing_identity="replace-me",
    )

    assert approved is True
    assert reason == "CAPITAL_LEVEL_2_CAPACITY_AVAILABLE"
    assert order == {"status": "new"}
    assert snapshot["managed_position_count"] == 2
    assert submitted_snapshot["capacity_managed_position_count"] == 1
    assert Decimal(
        str(submitted_snapshot["capacity_managed_exposure_eur"])
    ) == Decimal("10")
    assert submitted_snapshot["capacity_replacement"] == {
        "source": "GENERATED_DNA",
        "identity": "replace-me",
    }


@pytest.mark.asyncio
async def test_atomic_reprice_requires_exact_existing_identity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    called = False

    async def submit(_snapshot: object) -> dict[str, str]:
        nonlocal called
        called = True
        return {"status": "new"}

    approved, reason, _, order = await submit_level_2_buy_atomically(
        settings,
        requested_notional_eur=Decimal("10"),
        submit_order=submit,
        replacing_source="GENERATED_DNA",
        replacing_identity="missing",
    )

    assert approved is False
    assert reason == "REPLACED_POSITION_RESERVATION_NOT_FOUND"
    assert order is None
    assert called is False
