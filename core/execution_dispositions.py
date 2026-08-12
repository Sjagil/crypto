"""Deterministic final dispositions for realtime entry-ready candidates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from utils.common import stable_hash


def _hard_block_disposition(blockers: Iterable[object]) -> str:
    combined = " ".join(str(value).upper() for value in blockers)
    mappings = (
        (("STALE", "DATA", "SEQUENCE", "BOOK"), "REJECTED_DATA"),
        (("SPREAD",), "REJECTED_LIQUIDITY"),
        (("LIQUID", "DEPTH", "SLIPPAGE", "IMPACT"), "REJECTED_LIQUIDITY"),
        (("STOP", "EXIT"), "REJECTED_STOP"),
        (("ECONOMIC", "EXPECTANCY", "ECR", "NET_RR", "COST"), "REJECTED_ECONOMICS"),
        (("RISK", "EXPOSURE", "POSITION", "DAILY_LOSS", "DRAWDOWN"), "REJECTED_RISK"),
        (("HOSTILE", "ORDERFLOW"), "REJECTED_HOSTILE_FLOW"),
        (("AUTHORITY", "DNA"), "REJECTED_AUTHORITY"),
    )
    return next(
        (
            disposition
            for markers, disposition in mappings
            if any(marker in combined for marker in markers)
        ),
        "REJECTED_HARD_POLICY",
    )


def _cycle_disposition(status: str, reason: str) -> str:
    combined = f"{status} {reason}".upper()
    mappings = (
        (("DATA_BLOCKED", "REALTIME_ENTRY_FACTS"), "REJECTED_DATA"),
        (("RECONCILIATION", "PREFLIGHT"), "REJECTED_RECONCILIATION"),
        (("BALANCE", "EXPOSURE", "PORTFOLIO_HEAT", "POSITION_LIMIT"), "REJECTED_RISK"),
        (("MINIMUM", "LIQUIDITY", "MARKET_RULES", "BOUNDED_LIMIT"), "REJECTED_LIQUIDITY"),
        (("STOP", "PROTECT"), "REJECTED_STOP"),
        (("ECONOMIC", "ECR", "NET_RR", "COST"), "REJECTED_ECONOMICS"),
        (("AUTHORITY", "ENTRIES_DISABLED", "ACCOUNT"), "REJECTED_AUTHORITY_OR_ACCOUNT"),
        (("DAILY", "DRAWDOWN", "KILL_SWITCH", "RISK"), "REJECTED_RISK"),
    )
    return next(
        (
            disposition
            for markers, disposition in mappings
            if any(marker in combined for marker in markers)
        ),
        "EXECUTION_INCIDENT_UNEXPLAINED_NO_SUBMIT",
    )


def build_entry_ready_dispositions(
    opportunities: Iterable[Mapping[str, Any]],
    execution: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> dict[str, Any]:
    """Assign every fresh ENTRY_READY row an explicit cycle disposition."""

    rows = [
        dict(row)
        for row in opportunities
        if str(row.get("state") or "") == "ENTRY_READY"
    ]
    events = [dict(row) for row in execution.get("events") or []]
    events_by_identity: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        identity = str(event.get("opportunity_id") or "")
        if identity:
            events_by_identity.setdefault(identity, []).append(event)
    selected_identity = next(
        (
            identity
            for identity, selected_events in events_by_identity.items()
            if any(
                str(event.get("event") or "")
                in {"LIVE_ORDER_INTENT_CREATED", "LIVE_ORDER_SUBMITTED"}
                for event in selected_events
            )
        ),
        "",
    )
    cycle_status = str(execution.get("status") or "")
    reason_code = str(execution.get("reason_code") or "")
    dispositions: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row.get("opportunity_id") or "")
        selected_events = events_by_identity.get(identity, [])
        event_names = [str(event.get("event") or "") for event in selected_events]
        if "LIVE_POSITION_FILLED" in event_names:
            disposition = "FILLED"
        elif "LIVE_ORDER_PARTIALLY_FILLED" in event_names:
            disposition = "ORDER_PARTIALLY_FILLED"
        elif "LIVE_ORDER_SUBMITTED" in event_names:
            disposition = "ORDER_SUBMITTED"
        elif "LIVE_ORDER_INTENT_CREATED" in event_names:
            disposition = _cycle_disposition(cycle_status, reason_code)
        elif row.get("hard_blockers"):
            disposition = _hard_block_disposition(row["hard_blockers"])
        elif row.get("live_authority_granted") is not True:
            disposition = "REJECTED_AUTHORITY"
        elif selected_identity and identity != selected_identity:
            disposition = "SUPERSEDED_BY_HIGHER_RANKED_CANDIDATE"
        else:
            disposition = _cycle_disposition(cycle_status, reason_code)
        body = {
            "observed_at": observed_at.astimezone(UTC).isoformat(),
            "opportunity_id": identity,
            "market": row.get("market"),
            "playbook_id": row.get("playbook_id"),
            "state": "ENTRY_READY",
            "disposition": disposition,
            "execution_status": cycle_status,
            "reason_code": reason_code,
            "hard_blockers": list(row.get("hard_blockers") or []),
            "exchange_order_id": next(
                (
                    event.get("order_id")
                    for event in selected_events
                    if event.get("order_id")
                ),
                None,
            ),
        }
        dispositions.append(
            {
                **body,
                "disposition_id": stable_hash(
                    [
                        identity,
                        disposition,
                        reason_code,
                        body["exchange_order_id"],
                    ],
                    length=40,
                ),
            }
        )
    submitted = sum(
        row["disposition"]
        in {"ORDER_SUBMITTED", "ORDER_PARTIALLY_FILLED", "FILLED"}
        for row in dispositions
    )
    unexplained = sum(
        row["disposition"] == "EXECUTION_INCIDENT_UNEXPLAINED_NO_SUBMIT"
        for row in dispositions
    )
    return {
        "schema_version": "entry_ready_dispositions_v1",
        "observed_at": observed_at.astimezone(UTC).isoformat(),
        "entry_ready_count": len(dispositions),
        "order_submitted_count": submitted,
        "unexplained_no_submit_count": unexplained,
        "execution_incident": bool(dispositions and unexplained),
        "rows": dispositions,
    }


__all__ = ["build_entry_ready_dispositions"]
