from __future__ import annotations

from datetime import UTC, datetime

from core.execution_dispositions import build_entry_ready_dispositions

NOW = datetime(2026, 8, 8, 20, tzinfo=UTC)


def _row(**updates: object) -> dict[str, object]:
    return {
        "opportunity_id": "op-1",
        "market": "SOL-EUR",
        "playbook_id": "BREAKOUT_PULLBACK_V1",
        "state": "ENTRY_READY",
        "live_authority_granted": True,
        "hard_blockers": [],
        **updates,
    }


def test_submitted_entry_ready_has_exchange_disposition() -> None:
    report = build_entry_ready_dispositions(
        [_row()],
        {
            "status": "ENTRY_SUBMITTED",
            "reason_code": "NATURAL_ENTRY",
            "events": [
                {
                    "event": "LIVE_ORDER_SUBMITTED",
                    "opportunity_id": "op-1",
                    "order_id": "venue-1",
                }
            ],
        },
        observed_at=NOW,
    )

    assert report["entry_ready_count"] == 1
    assert report["order_submitted_count"] == 1
    assert report["execution_incident"] is False
    assert report["rows"][0]["disposition"] == "ORDER_SUBMITTED"


def test_hard_data_blocker_is_not_reported_as_generic_no_entry() -> None:
    report = build_entry_ready_dispositions(
        [_row(hard_blockers=["ORDERBOOK_SEQUENCE_INVALID"])],
        {"status": "DATA_BLOCKED", "reason_code": "REALTIME_ENTRY_FACTS_NOT_READY"},
        observed_at=NOW,
    )

    assert report["rows"][0]["disposition"] == "REJECTED_DATA"
    assert report["execution_incident"] is False


def test_authority_failure_is_explicit() -> None:
    report = build_entry_ready_dispositions(
        [_row(live_authority_granted=False)],
        {"status": "READY", "reason_code": "NO_APPROVED_EVENT_ENTRY_READY"},
        observed_at=NOW,
    )

    assert report["rows"][0]["disposition"] == "REJECTED_AUTHORITY"
    assert report["execution_incident"] is False


def test_unexplained_entry_ready_without_submit_is_incident() -> None:
    report = build_entry_ready_dispositions(
        [_row()],
        {"status": "READY", "reason_code": ""},
        observed_at=NOW,
    )

    assert report["rows"][0]["disposition"] == (
        "EXECUTION_INCIDENT_UNEXPLAINED_NO_SUBMIT"
    )
    assert report["execution_incident"] is True
