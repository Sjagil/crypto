"""Fail-closed continuity checks for the live EUR cash balance.

The exchange reconciliation endpoint proves that open orders agree, but a
closed external order, transfer, or manual account mutation can still change
cash without leaving an open-order mismatch.  This guard keeps the last
accepted EUR balance and requires a material change to reconcile to either a
canonical fill or an operator-confirmed external capital flow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from config.settings import Settings
from utils.common import atomic_write_json, read_json, utc_iso, utc_now

ZERO = Decimal("0")
DEFAULT_TOLERANCE_EUR = Decimal("0.50")
TAIL_BYTES = 4 * 1024 * 1024
FILL_SETTLEMENT_LOOKBACK = timedelta(minutes=2)


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _tail_jsonl(path: Path, *, maximum_bytes: int = TAIL_BYTES) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - maximum_bytes)
        handle.seek(start)
        raw = handle.read()
    if start:
        _, _, raw = raw.partition(b"\n")
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _prior_distinct_cash_snapshot(
    settings: Settings,
    current: Decimal,
) -> tuple[Decimal, datetime] | None:
    path = settings.paths.output_dir / "live" / "events" / "positions.jsonl"
    snapshots: list[tuple[Decimal, datetime]] = []
    for row in _tail_jsonl(path):
        if row.get("event") != "ACCOUNT_RECONCILIATION":
            continue
        amount = _decimal((row.get("account") or {}).get("eur_available"))
        observed = _timestamp(row.get("recorded_at"))
        if amount is not None and observed is not None:
            snapshots.append((amount, observed))
    for amount, observed in reversed(snapshots):
        if amount != current:
            return amount, observed
    return None


def _confirmed_capital_flow(
    settings: Settings,
    *,
    after: datetime,
    through: datetime,
) -> Decimal:
    path = (
        settings.paths.output_dir
        / "portfolio"
        / "external_capital_flows.jsonl"
    )
    total = ZERO
    for row in _tail_jsonl(path):
        if row.get("operator_confirmed") is not True:
            continue
        effective = _timestamp(row.get("effective_at"))
        amount = _decimal(row.get("amount_eur"))
        if (
            effective is not None
            and amount is not None
            and after < effective <= through
        ):
            total += amount
    return total


def _canonical_fill_cash_delta(
    settings: Settings,
    *,
    after: datetime,
    through: datetime,
    excluded_event_ids: set[str] | None = None,
) -> tuple[Decimal, list[str]]:
    checkpoints = getattr(settings.paths, "checkpoints_dir", None)
    canonical_path = (
        Path(checkpoints) / "live_execution.jsonl"
        if checkpoints is not None
        else None
    )
    path = (
        canonical_path
        if canonical_path is not None and canonical_path.is_file()
        else settings.paths.output_dir / "live" / "events" / "fills.jsonl"
    )
    direct_canonical = canonical_path is not None and path == canonical_path
    total = ZERO
    event_ids: list[str] = []
    excluded = excluded_event_ids or set()
    for row in _tail_jsonl(path):
        if (
            row.get("event_type") != "FILL"
            if direct_canonical
            else row.get("event") != "CANONICAL_FILL"
        ):
            continue
        observed = _timestamp(row.get("recorded_at"))
        if observed is None or not after < observed <= through:
            continue
        payload = row.get("payload") or {}
        event_id = str(
            payload.get("fill_id")
            or row.get("event_id")
            or payload.get("idempotency_key")
            or ""
        ).strip()
        if event_id and event_id in excluded:
            continue
        price = _decimal(payload.get("price"))
        quantity = _decimal(payload.get("quantity"))
        fee = _decimal(payload.get("fee_eur")) or ZERO
        side = str(payload.get("side") or "").upper()
        if price is None or quantity is None or side not in {"BUY", "SELL"}:
            continue
        notional = price * quantity
        total += notional - fee if side == "SELL" else -(notional + fee)
        if event_id:
            event_ids.append(event_id)
    return total, event_ids


def _external_account_fill_cash_delta(
    settings: Settings,
    *,
    after: datetime,
    through: datetime,
    excluded_event_ids: set[str] | None = None,
) -> tuple[Decimal, list[str]]:
    """Value privately observed fills which were not created by this bot.

    Bitvavo account-stream fills generated by this system always carry a
    client-order identity.  A fill without one is external account activity
    (for example a manual app order).  It is valid broker evidence for EUR
    continuity, but deliberately remains *external inventory*: recognizing its
    cash effect neither adopts the asset nor grants the bot exit authority.
    """

    path = settings.paths.output_dir / "live" / "events" / "fills.jsonl"
    total = ZERO
    event_ids: list[str] = []
    excluded = excluded_event_ids or set()
    for row in _tail_jsonl(path):
        if row.get("event") != "BITVAVO_ACCOUNT_FILL":
            continue
        observed = _timestamp(row.get("recorded_at"))
        if observed is None or not after < observed <= through:
            continue
        payload = row.get("payload") or {}
        if payload.get("client_order_public_id"):
            continue
        event_id = str(
            row.get("event_id")
            or payload.get("raw_payload_hash")
            or payload.get("fill_public_id")
            or ""
        ).strip()
        if event_id and event_id in excluded:
            continue
        price = _decimal(payload.get("fill_price") or payload.get("price"))
        quantity = _decimal(
            payload.get("amount") or payload.get("filled_amount")
        )
        side = str(payload.get("side") or "").upper()
        fee = (
            _decimal(payload.get("fee")) or ZERO
            if str(payload.get("fee_currency") or "").upper() == "EUR"
            else ZERO
        )
        if price is None or quantity is None or side not in {"BUY", "SELL"}:
            continue
        notional = price * quantity
        total += notional - fee if side == "SELL" else -(notional + fee)
        if event_id:
            event_ids.append(event_id)
    return total, event_ids


def _within_tolerance(
    observed: Decimal,
    explained: Decimal,
    tolerance: Decimal,
) -> bool:
    return abs(observed - explained) <= tolerance


def evaluate_eur_cash_continuity(
    settings: Settings,
    *,
    current_eur_available: Decimal | float | str,
    observed_at: datetime | None = None,
    tolerance_eur: Decimal = DEFAULT_TOLERANCE_EUR,
    exchange_external_cash_delta_eur: Decimal | float | str = ZERO,
    exchange_history_complete: bool = False,
) -> dict[str, Any]:
    """Persist and reconcile material EUR balance changes without orders."""

    current = _decimal(current_eur_available)
    exchange_external_delta = (
        _decimal(exchange_external_cash_delta_eur) or ZERO
    )
    now = (observed_at or utc_now()).astimezone(UTC)
    path = settings.paths.output_dir / "operations" / "eur_cash_continuity.json"
    previous = dict(read_json(path)) if path.is_file() else {}
    consumed_fill_ids = {
        str(value)
        for value in (previous.get("consumed_canonical_fill_ids") or [])
        if str(value).strip()
    }
    consumed_external_fill_ids = {
        str(value)
        for value in (previous.get("consumed_external_fill_event_ids") or [])
        if str(value).strip()
    }
    accepted = _decimal(previous.get("accepted_eur_available"))
    accepted_at = _timestamp(previous.get("accepted_at"))
    bootstrapped_from_events = False
    if accepted is None or accepted_at is None:
        historical = (
            _prior_distinct_cash_snapshot(settings, current)
            if current is not None
            else None
        )
        if historical is not None:
            accepted, accepted_at = historical
            bootstrapped_from_events = True

    if current is None:
        status = "EUR_BALANCE_INVALID"
        blocking = True
        delta = None
        capital_flow = ZERO
        fill_delta = ZERO
        fill_event_ids: list[str] = []
        external_fill_delta = ZERO
        external_fill_event_ids: list[str] = []
        settlement_lookback_applied = False
    elif accepted is None or accepted_at is None:
        status = "BASELINE_ESTABLISHED"
        blocking = False
        accepted = current
        accepted_at = now
        delta = ZERO
        capital_flow = ZERO
        fill_delta = ZERO
        fill_event_ids = []
        external_fill_delta = ZERO
        external_fill_event_ids = []
        settlement_lookback_applied = False
    else:
        delta = current - accepted
        capital_flow = _confirmed_capital_flow(
            settings,
            after=accepted_at,
            through=now,
        )
        fill_delta, fill_event_ids = _canonical_fill_cash_delta(
            settings,
            after=accepted_at,
            through=now,
            excluded_event_ids=consumed_fill_ids,
        )
        external_fill_delta, external_fill_event_ids = (
            _external_account_fill_cash_delta(
                settings,
                after=accepted_at,
                through=now,
                excluded_event_ids=consumed_external_fill_ids,
            )
        )
        # Complete exchange history is the broker-of-record evidence and takes
        # precedence over the operator ledger, preventing the same deposit or
        # withdrawal from being counted twice.
        explained_external = (
            exchange_external_delta
            if exchange_history_complete
            else capital_flow
        )
        explained = explained_external + fill_delta + external_fill_delta
        settlement_lookback_applied = False
        if (
            abs(delta) > tolerance_eur
            and not _within_tolerance(delta, explained, tolerance_eur)
        ):
            # A marketable limit can fill milliseconds before the account
            # snapshot used to establish the cash baseline is persisted. The
            # balance endpoint may still return its pre-fill value during that
            # short settlement race. Reconsider only unconsumed canonical
            # fills in a tightly bounded window and only accept them when they
            # reconcile the observed cash delta within the normal tolerance.
            candidate_delta, candidate_ids = _canonical_fill_cash_delta(
                settings,
                after=accepted_at - FILL_SETTLEMENT_LOOKBACK,
                through=now,
                excluded_event_ids=consumed_fill_ids,
            )
            candidate_external_delta, candidate_external_ids = (
                _external_account_fill_cash_delta(
                    settings,
                    after=accepted_at - FILL_SETTLEMENT_LOOKBACK,
                    through=now,
                    excluded_event_ids=consumed_external_fill_ids,
                )
            )
            candidate_explained = (
                explained_external
                + candidate_delta
                + candidate_external_delta
            )
            if _within_tolerance(delta, candidate_explained, tolerance_eur):
                fill_delta = candidate_delta
                fill_event_ids = candidate_ids
                external_fill_delta = candidate_external_delta
                external_fill_event_ids = candidate_external_ids
                explained = candidate_explained
                settlement_lookback_applied = True
        if abs(delta) <= tolerance_eur:
            status = "READY_STABLE"
            blocking = False
        elif _within_tolerance(delta, explained, tolerance_eur):
            status = "READY_EXPLAINED_CHANGE"
            blocking = False
        else:
            status = "UNEXPLAINED_EUR_BALANCE_CHANGE"
            blocking = True
        if not blocking:
            accepted = current
            accepted_at = now
            consumed_fill_ids.update(fill_event_ids)
            consumed_external_fill_ids.update(external_fill_event_ids)

    payload = {
        "schema_version": "eur_cash_continuity_v1",
        "status": status,
        "checked_at": utc_iso(now),
        "accepted_at": utc_iso(accepted_at) if accepted_at else None,
        "accepted_eur_available": str(accepted) if accepted is not None else None,
        "current_eur_available": str(current) if current is not None else None,
        "pending_unexplained_eur_available": (
            str(current) if blocking and current is not None else None
        ),
        "observed_delta_eur": str(delta) if delta is not None else None,
        "confirmed_capital_flow_eur": str(capital_flow),
        "canonical_fill_cash_delta_eur": str(fill_delta),
        "canonical_fill_event_ids": fill_event_ids,
        "consumed_canonical_fill_ids": sorted(consumed_fill_ids)[-500:],
        "external_account_fill_cash_delta_eur": str(external_fill_delta),
        "external_account_fill_event_ids": external_fill_event_ids,
        "consumed_external_fill_event_ids": sorted(
            consumed_external_fill_ids
        )[-500:],
        "external_fills_change_cash_evidence_only": True,
        "external_fills_do_not_adopt_inventory_or_grant_exit_authority": True,
        "fill_settlement_lookback_applied": settlement_lookback_applied,
        "exchange_external_cash_delta_eur": str(exchange_external_delta),
        "exchange_transaction_history_complete": exchange_history_complete,
        "tolerance_eur": str(tolerance_eur),
        "new_entries_blocked": blocking,
        "protective_exits_allowed": True,
        "bootstrapped_from_reconciliation_events": bootstrapped_from_events,
        "resolution_command_template": (
            ".\\.venv\\Scripts\\python.exe .\\main.py capital record-flow "
            "--amount-eur <SIGNED_AMOUNT> --reason <DEPOSIT|WITHDRAWAL|TRANSFER|CORRECTION> "
            "--effective-at <UTC_TIMESTAMP> --note \"<OPERATOR_EXPLANATION>\""
            if blocking
            else None
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
        "secrets_serialized": False,
    }
    atomic_write_json(path, payload)
    return {**payload, "artifact": str(path)}


__all__: Iterable[str] = ["evaluate_eur_cash_continuity"]
