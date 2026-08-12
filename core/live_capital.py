"""Shared, operator-approved live capital limits and portfolio accounting."""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, TypeVar

from config.settings import Settings
from utils.common import read_json, stable_hash

CAPITAL_LEVEL = 2
MAXIMUM_ORDER_EUR = Decimal("25")
MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR = Decimal("75")
MAXIMUM_MANAGED_POSITIONS = 3
MAXIMUM_NEW_ORDERS_PER_DAY = 3
MAXIMUM_RISK_PER_TRADE_EUR = Decimal("2")
AUTOSCALE = False
APPROVAL_PHRASE = "I APPROVE LIVE CAPITAL LEVEL 2"

_T = TypeVar("_T")


class LiveEntryReservationBusy(RuntimeError):
    """Raised when another engine owns the atomic live BUY reservation."""


class LiveEntryReservation(AbstractContextManager["LiveEntryReservation"]):
    """Cross-process, crash-releasing lock for the final BUY cap check.

    The operating system owns the byte-range/file lock.  A process crash
    therefore releases it without relying on a potentially stale PID file.
    The lock deliberately covers BUY reservation only; exits and native
    protection never wait behind entry allocation.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "LiveEntryReservation":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if self.path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - production runtime is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LiveEntryReservationBusy(
                "LIVE_ENTRY_RESERVATION_BUSY"
            ) from exc
        self._handle = handle
        return self

    def __exit__(self, *exc_info: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - production runtime is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def live_entry_reservation(settings: Settings) -> LiveEntryReservation:
    return LiveEntryReservation(
        settings.paths.output_dir / "live" / "entry_reservation.lock"
    )


def _exposure_class(row: Mapping[str, Any]) -> str | None:
    """Classify actual and potential exposure conservatively.

    Pending entries reserve capacity before a fill.  Positions being managed or
    exited remain actual exposure until reconciliation proves them closed.
    """

    status = str(row.get("status") or "").upper()
    if status.startswith("ENTRY_PENDING"):
        return "PENDING_ENTRY"
    if status.startswith(
        ("OPEN", "MANAGING", "PARTIALLY_REDUCED", "EXIT_PENDING")
    ):
        return "CURRENT_POSITION"
    return None


def _open_position(row: Mapping[str, Any]) -> bool:
    return _exposure_class(row) is not None


def _notional(row: Mapping[str, Any]) -> Decimal:
    try:
        return Decimal(str(row.get("quantity") or "0")) * Decimal(
            str(row.get("entry_price") or "0")
        )
    except (ArithmeticError, ValueError):
        return Decimal("0")


def _normalized_order_status(value: Any) -> str:
    return str(value or "").replace("_", "").replace("-", "").casefold()


def _intent_notional(payload: Mapping[str, Any]) -> Decimal:
    """Reserve the worst authorized BUY notional from an order intent."""

    try:
        explicit = Decimal(str(payload.get("maximum_notional_eur") or "0"))
        if explicit > 0:
            return explicit
        quantity = Decimal(str(payload.get("quantity") or "0"))
        price = Decimal(
            str(
                payload.get("limit_price")
                or payload.get("estimated_price")
                or payload.get("trigger_price")
                or "0"
            )
        )
    except (ArithmeticError, ValueError):
        return Decimal("0")
    return max(Decimal("0"), quantity * price)


def _ledger_pending_buy_reservations(
    settings: Settings,
    *,
    represented_client_ids: set[str],
    represented_signal_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recover open or ambiguous BUY exposure missing from engine state.

    The canonical order ledger is written before the venue POST.  It therefore
    closes the crash window where an order may exist at Bitvavo while neither
    live-engine state file has persisted its pending position yet.
    """

    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    if not ledger.is_file():
        return [], []
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"INVALID_LIVE_LEDGER_JSON_LINE:{line_number}")
            continue
        if not isinstance(event, dict):
            failures.append(f"INVALID_LIVE_LEDGER_EVENT:{line_number}")
            continue
        events.append(event)

    intents: dict[str, dict[str, Any]] = {}
    status_by_client: dict[str, str] = {}
    order_to_client: dict[str, str] = {}
    terminal_clients: set[str] = set()
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = dict(event.get("payload") or {})
        client_id = str(payload.get("client_order_id") or "")
        order_id = str(payload.get("order_id") or "")
        if event_type == "ORDER_INTENT":
            if str(payload.get("side") or "").upper() != "BUY":
                continue
            if not client_id:
                failures.append("BUY_ORDER_INTENT_MISSING_CLIENT_ID")
                continue
            intents[client_id] = payload
            status_by_client.setdefault(client_id, "orderstateunknown")
            continue
        if order_id and client_id:
            order_to_client[order_id] = client_id
        if not client_id and order_id:
            client_id = order_to_client.get(order_id, "")
        if not client_id:
            continue
        if event_type in {"ORDER_ACKNOWLEDGED", "ORDER_STATUS_OBSERVED"}:
            status_by_client[client_id] = _normalized_order_status(
                payload.get("status")
            )
        if event_type in {"ORDER_REJECTED", "ORDER_CANCELLED"}:
            terminal_clients.add(client_id)
        if event_type == "FILL" and str(payload.get("status") or "").upper() in {
            "FILLED",
            "PARTIALLY_FILLED_FINAL",
        }:
            terminal_clients.add(client_id)
        if event_type == "CANCEL_RESOLVED" and _normalized_order_status(
            payload.get("terminal_order_status")
        ) in {"filled", "expired", "rejected"}:
            terminal_clients.add(client_id)

    terminal_statuses = {
        "filled",
        "canceled",
        "cancelled",
        "expired",
        "rejected",
    }
    reservations: list[dict[str, Any]] = []
    for client_id, intent in intents.items():
        signal_id = str(intent.get("signal_id") or "")
        if (
            client_id in represented_client_ids
            or (signal_id and signal_id in represented_signal_ids)
            or client_id in terminal_clients
        ):
            continue
        status = status_by_client.get(client_id, "orderstateunknown")
        if status in terminal_statuses:
            continue
        notional = _intent_notional(intent)
        if notional <= 0:
            failures.append(
                "PENDING_BUY_NOTIONAL_UNKNOWN:"
                + stable_hash(client_id, length=16)
            )
            continue
        reservations.append(
            {
                "source": "CANONICAL_LEDGER",
                "identity": stable_hash(client_id, length=20),
                "client_order_public_id": stable_hash(
                    client_id,
                    length=20,
                ),
                "signal_id": signal_id or None,
                "market": str(intent.get("market") or ""),
                "status": status.upper(),
                "exposure_class": "PENDING_ENTRY",
                "notional_eur": str(notional),
                "recovered_from_ledger": True,
            }
        )
    return reservations, failures


def managed_live_portfolio(settings: Settings) -> dict[str, Any]:
    """Return shared actual and potential exposure across both live engines.

    The legacy ``managed_*`` totals intentionally remain conservative potential
    totals so existing Level-2 gates continue to reserve capacity for pending
    entry orders.
    """

    sources = {
        "GENERATED_DNA": (
            settings.paths.output_dir / "live" / "generated_strategy_live_state.json"
        ),
        "EVENT_PLAYBOOK": (
            settings.paths.output_dir / "live" / "event_driven_execution_state.json"
        ),
    }
    rows: list[dict[str, Any]] = []
    represented_client_ids: set[str] = set()
    represented_signal_ids: set[str] = set()
    for source, path in sources.items():
        state = dict(read_json(path)) if path.is_file() else {}
        for identity, raw in dict(state.get("positions") or {}).items():
            row = dict(raw)
            exposure_class = _exposure_class(row)
            if exposure_class is None:
                continue
            client_order_id = str(row.get("client_order_id") or "")
            signal_id = str(
                row.get("signal_id") or row.get("opportunity_id") or ""
            )
            if client_order_id:
                represented_client_ids.add(client_order_id)
            if signal_id:
                represented_signal_ids.add(signal_id)
            rows.append(
                {
                    "source": source,
                    "identity": str(identity),
                    "market": str(row.get("market") or ""),
                    "status": str(row.get("status") or ""),
                    "client_order_public_id": (
                        stable_hash(client_order_id, length=20)
                        if client_order_id
                        else None
                    ),
                    "signal_id": signal_id or None,
                    "exposure_class": exposure_class,
                    "notional_eur": str(_notional(row)),
                }
            )
    rr_path = settings.paths.output_dir / "reports" / "current_position.json"
    rr_state = dict(read_json(rr_path)) if rr_path.is_file() else {}
    rr_position = dict(rr_state.get("position") or {})
    rr_status = str(rr_state.get("status") or "")
    if rr_position and rr_status in {
        "OPEN",
        "OPEN_PENDING_RECONCILIATION",
        "PARTIALLY_REDUCED",
        "EXIT_PENDING_RECONCILIATION",
    }:
        rr_row = {
            **rr_position,
            "status": (
                "ENTRY_PENDING_RECONCILIATION"
                if rr_status == "OPEN_PENDING_RECONCILIATION"
                else rr_status
            ),
        }
        client_order_id = str(
            rr_position.get("entry_client_order_id") or ""
        )
        signal_id = str(
            rr_position.get("entry_opportunity_id") or ""
        )
        if client_order_id:
            represented_client_ids.add(client_order_id)
        if signal_id:
            represented_signal_ids.add(signal_id)
        exposure_class = _exposure_class(rr_row)
        if exposure_class is not None:
            rows.append(
                {
                    "source": "RR_PRIMARY",
                    "identity": str(
                        rr_position.get("strategy_dna_hash")
                        or "RR_B60_H5_Z20"
                    ),
                    "market": str(rr_position.get("market") or ""),
                    "status": rr_status,
                    "client_order_public_id": (
                        stable_hash(client_order_id, length=20)
                        if client_order_id
                        else None
                    ),
                    "signal_id": signal_id or None,
                    "exposure_class": exposure_class,
                    "notional_eur": str(_notional(rr_row)),
                }
            )
    ledger_rows, ledger_failures = _ledger_pending_buy_reservations(
        settings,
        represented_client_ids=represented_client_ids,
        represented_signal_ids=represented_signal_ids,
    )
    rows.extend(ledger_rows)
    potential_exposure = sum(
        (Decimal(row["notional_eur"]) for row in rows),
        Decimal("0"),
    )
    current_rows = [
        row for row in rows if row["exposure_class"] == "CURRENT_POSITION"
    ]
    pending_rows = [
        row for row in rows if row["exposure_class"] == "PENDING_ENTRY"
    ]
    current_exposure = sum(
        (Decimal(row["notional_eur"]) for row in current_rows), Decimal("0")
    )
    pending_exposure = sum(
        (Decimal(row["notional_eur"]) for row in pending_rows), Decimal("0")
    )
    return {
        "schema_version": "managed_live_portfolio_v4",
        "private_order_identifiers_serialized": False,
        "status": (
            "READY"
            if not ledger_failures
            else "PENDING_ORDER_EXPOSURE_UNRECONCILED"
        ),
        "failures": ledger_failures,
        "capital_level": CAPITAL_LEVEL,
        "current_position_count": len(current_rows),
        "current_position_exposure_eur": str(current_exposure),
        "pending_entry_order_count": len(pending_rows),
        "pending_entry_exposure_eur": str(pending_exposure),
        "ledger_recovered_pending_order_count": len(ledger_rows),
        "ledger_recovered_pending_exposure_eur": str(
            sum(
                (Decimal(row["notional_eur"]) for row in ledger_rows),
                Decimal("0"),
            )
        ),
        "potential_managed_position_count": len(rows),
        "potential_managed_exposure_eur": str(potential_exposure),
        "managed_position_count": len(rows),
        "managed_exposure_eur": str(potential_exposure),
        "positions": rows,
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_managed_exposure_eur": str(
            MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR
        ),
        "maximum_managed_positions": MAXIMUM_MANAGED_POSITIONS,
        "maximum_risk_per_trade_eur": str(MAXIMUM_RISK_PER_TRADE_EUR),
        "autoscale": AUTOSCALE,
    }


def capital_level_2_capacity(
    settings: Settings,
    *,
    requested_notional_eur: Decimal,
    replacing_source: str | None = None,
    replacing_identity: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    snapshot = managed_live_portfolio(settings)
    if snapshot["status"] != "READY":
        return False, "PENDING_ORDER_EXPOSURE_UNRECONCILED", snapshot
    count = int(snapshot["managed_position_count"])
    exposure = Decimal(snapshot["managed_exposure_eur"])
    if replacing_source is not None or replacing_identity is not None:
        matches = [
            row
            for row in snapshot["positions"]
            if row.get("source") == replacing_source
            and row.get("identity") == replacing_identity
        ]
        if len(matches) != 1:
            return (
                False,
                "REPLACED_POSITION_RESERVATION_NOT_FOUND",
                snapshot,
            )
        count -= 1
        exposure -= Decimal(str(matches[0]["notional_eur"]))
    snapshot["capacity_managed_position_count"] = count
    snapshot["capacity_managed_exposure_eur"] = str(exposure)
    snapshot["capacity_replacement"] = (
        {
            "source": replacing_source,
            "identity": replacing_identity,
        }
        if replacing_source is not None
        else None
    )
    if count >= MAXIMUM_MANAGED_POSITIONS:
        return False, "MANAGED_POSITION_LIMIT_REACHED", snapshot
    if exposure + requested_notional_eur > MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR:
        return False, "MANAGED_EXPOSURE_LIMIT_REACHED", snapshot
    return True, "CAPITAL_LEVEL_2_CAPACITY_AVAILABLE", snapshot


async def submit_level_2_buy_atomically(
    settings: Settings,
    *,
    requested_notional_eur: Decimal,
    submit_order: Callable[[Mapping[str, Any]], Awaitable[_T]],
    replacing_source: str | None = None,
    replacing_identity: str | None = None,
) -> tuple[bool, str, dict[str, Any], _T | None]:
    """Recheck shared capacity and durably submit one BUY under one lock.

    The callback must append its canonical ``ORDER_INTENT`` before issuing the
    venue POST (``BitvavoSpotClient.submit_order`` has this invariant).  Once
    the callback returns or raises, the ledger reservation is visible to every
    subsequent engine and the operating-system lock can safely be released.
    """

    try:
        with live_entry_reservation(settings):
            approved, reason, snapshot = capital_level_2_capacity(
                settings,
                requested_notional_eur=requested_notional_eur,
                replacing_source=replacing_source,
                replacing_identity=replacing_identity,
            )
            if not approved:
                return False, reason, snapshot, None
            result = await submit_order(snapshot)
            return True, reason, snapshot, result
    except LiveEntryReservationBusy:
        return (
            False,
            "LIVE_ENTRY_RESERVATION_BUSY",
            managed_live_portfolio(settings),
            None,
        )


__all__ = [
    "APPROVAL_PHRASE",
    "AUTOSCALE",
    "CAPITAL_LEVEL",
    "MAXIMUM_MANAGED_POSITIONS",
    "MAXIMUM_NEW_ORDERS_PER_DAY",
    "MAXIMUM_ORDER_EUR",
    "MAXIMUM_RISK_PER_TRADE_EUR",
    "MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR",
    "LiveEntryReservation",
    "LiveEntryReservationBusy",
    "capital_level_2_capacity",
    "live_entry_reservation",
    "managed_live_portfolio",
    "submit_level_2_buy_atomically",
]
