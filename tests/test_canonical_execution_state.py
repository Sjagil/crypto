from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from execution.canonical_state import (
    CanonicalExecutionState,
    CanonicalOrderStatus,
    OwnershipState,
    ProtectionState,
    assert_replay_deterministic,
    normalize_execution_events,
    reduce_execution_event,
    replay_execution_events,
)
from execution.position_tracker import PositionTracker
from execution.state_migration import build_execution_divergence_report


def event(event_type: str, recorded_at: str, **payload: object) -> dict[str, object]:
    return {
        "event_type": event_type,
        "recorded_at": recorded_at,
        "payload": payload,
    }


def intent(
    *,
    recorded_at: str = "2026-01-01T00:00:00Z",
    intent_id: str = "intent-1",
    market: str = "BTC-EUR",
    side: str = "BUY",
    quantity: str = "2",
    price: str = "100",
    order_type: str = "LIMIT",
    trigger_price: str | None = None,
    strategy_id: str = "STRATEGY_A",
    strategy_dna_hash: str = "dna-a",
    signal_id: str = "signal-a",
) -> dict[str, object]:
    return event(
        "ORDER_INTENT",
        recorded_at,
        intent_id=intent_id,
        idempotency_key=f"key:{intent_id}",
        client_order_id=f"client:{intent_id}",
        market=market,
        side=side,
        quantity=quantity,
        estimated_price=price,
        limit_price=price if order_type == "LIMIT" else None,
        order_type=order_type,
        trigger_price=trigger_price,
        strategy_id=strategy_id,
        strategy_dna_hash=strategy_dna_hash,
        signal_id=signal_id,
    )


def acknowledgement(
    *,
    recorded_at: str = "2026-01-01T00:00:01Z",
    intent_id: str = "intent-1",
    order_id: str = "order-1",
    market: str = "BTC-EUR",
    side: str = "BUY",
    status: str = "new",
) -> dict[str, object]:
    return event(
        "ORDER_ACKNOWLEDGED",
        recorded_at,
        intent_id=intent_id,
        client_order_id=f"client:{intent_id}",
        order_id=order_id,
        market=market,
        side=side,
        status=status,
    )


def fill(
    *,
    recorded_at: str = "2026-01-01T00:00:02Z",
    filled_at: str | None = None,
    fill_id: str = "fill-1",
    intent_id: str = "intent-1",
    order_id: str = "order-1",
    market: str = "BTC-EUR",
    side: str = "BUY",
    quantity: str = "2",
    price: str = "100",
    fee: str = "1",
    strategy_id: str = "STRATEGY_A",
    strategy_dna_hash: str = "dna-a",
    signal_id: str = "signal-a",
) -> dict[str, object]:
    return event(
        "FILL",
        recorded_at,
        fill_id=fill_id,
        intent_id=intent_id,
        order_id=order_id,
        client_order_id=f"client:{intent_id}",
        market=market,
        side=side,
        quantity=quantity,
        price=price,
        quote_amount_eur=str(Decimal(quantity) * Decimal(price)),
        fee_eur=fee,
        fee_known=True,
        filled_at=filled_at or recorded_at,
        strategy_id=strategy_id,
        strategy_dna_hash=strategy_dna_hash,
        signal_id=signal_id,
        venue="bitvavo",
    )


def protective_events(
    *,
    quantity: str = "2",
    trigger: str = "90",
    status: str = "awaitingTrigger",
) -> list[dict[str, object]]:
    return [
        intent(
            recorded_at="2026-01-01T00:00:03Z",
            intent_id="stop-1",
            side="SELL",
            quantity=quantity,
            price=trigger,
            order_type="STOP_LOSS",
            trigger_price=trigger,
        ),
        acknowledgement(
            recorded_at="2026-01-01T00:00:04Z",
            intent_id="stop-1",
            order_id="stop-order-1",
            side="SELL",
            status=status,
        ),
    ]


def order_by_intent(state: CanonicalExecutionState, intent_id: str):
    return next(order for order in state.orders.values() if order.intent_id == intent_id)


def test_clean_entry_and_confirmed_protection_are_canonical() -> None:
    events = [intent(), acknowledgement(status="filled"), fill(), *protective_events()]
    state = replay_execution_events(events)
    position = state.positions["BTC-EUR"]

    assert position.quantity == Decimal("2")
    assert position.cost_basis_eur == Decimal("201")
    assert position.average_entry_price == Decimal("100.5")
    assert position.strategy_id == "STRATEGY_A"
    assert position.ownership_state is OwnershipState.KNOWN
    assert position.protection_state is ProtectionState.CONFIRMED_ACTIVE
    assert position.protected_quantity == Decimal("2")
    assert position.unprotected_quantity == 0
    assert position.effective_stop_price == Decimal("90")
    assert position.open_risk_eur == Decimal("21.0")
    assert position.available_quantity == 0
    assert position.reserved_quantity == Decimal("2")


def test_multiple_partial_and_duplicate_fills_apply_once() -> None:
    first = fill(fill_id="fill-a", quantity="0.5", price="100", fee="0.1")
    private_duplicate = deepcopy(first)
    private_duplicate["payload"]["source"] = "private_websocket"
    rest_duplicate = deepcopy(first)
    rest_duplicate["event_type"] = "FILL_OBSERVED"
    rest_duplicate["payload"]["source"] = "rest"
    reconciliation_duplicate = deepcopy(first)
    reconciliation_duplicate["event_type"] = "FILL_RECONCILED"
    reconciliation_duplicate["payload"]["source"] = "reconciliation"
    second = fill(
        recorded_at="2026-01-01T00:00:03Z",
        fill_id="fill-b",
        quantity="1.5",
        price="102",
        fee="0.3",
    )
    events = [
        intent(),
        acknowledgement(),
        first,
        private_duplicate,
        rest_duplicate,
        reconciliation_duplicate,
        second,
    ]
    state = replay_execution_events(events)
    position = state.positions["BTC-EUR"]
    order = order_by_intent(state, "intent-1")

    assert len(state.fills) == 2
    assert len(state.processed_event_ids) == 4
    assert position.quantity == Decimal("2.0")
    assert position.cost_basis_eur == Decimal("203.4")
    assert order.filled_quantity == Decimal("2.0")
    assert order.fee_eur == Decimal("0.4")
    assert order.status is CanonicalOrderStatus.FILLED


@pytest.mark.parametrize("fill_first", [False, True])
def test_fill_and_status_callback_order_converge(fill_first: bool) -> None:
    opened = event(
        "ORDER_STATUS_OBSERVED",
        "2026-01-01T00:00:03Z",
        checkpoint_id="open-check",
        order_id="order-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
        status="new",
        exchange_updated_at="2026-01-01T00:00:01Z",
    )
    executed = fill(
        recorded_at="2026-01-01T00:00:02Z",
        filled_at="2026-01-01T00:00:02Z",
    )
    tail = [executed, opened] if fill_first else [opened, executed]
    state = replay_execution_events([intent(), acknowledgement(), *tail])
    order = order_by_intent(state, "intent-1")

    assert order.status is CanonicalOrderStatus.FILLED
    assert state.positions["BTC-EUR"].quantity == Decimal("2")


def test_cancel_fill_race_keeps_final_fill_as_strongest_fact() -> None:
    cancelled = event(
        "ORDER_CANCELLED",
        "2026-01-01T00:00:04Z",
        cancellation_id="cancel-1",
        intent_id="intent-1",
        order_id="order-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
        status="cancelled",
    )
    state = replay_execution_events([intent(), acknowledgement(), fill(), cancelled])

    assert order_by_intent(state, "intent-1").status is CanonicalOrderStatus.FILLED
    assert state.positions["BTC-EUR"].quantity == Decimal("2")


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("ORDER_REJECTED", CanonicalOrderStatus.REJECTED),
        ("ORDER_EXPIRED", CanonicalOrderStatus.EXPIRED),
        ("ORDER_STATE_UNKNOWN", CanonicalOrderStatus.UNKNOWN),
    ],
)
def test_rejected_expired_and_unknown_states_are_explicit(event_type, expected) -> None:
    terminal = event(
        event_type,
        "2026-01-01T00:00:02Z",
        intent_id="intent-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
        reason_code="FIXTURE",
    )
    state = replay_execution_events([intent(), terminal])
    order = order_by_intent(state, "intent-1")

    assert order.status is expected
    assert state.eur_reserved_for_orders == (
        Decimal("200") if expected is CanonicalOrderStatus.UNKNOWN else Decimal("0")
    )


def test_replay_is_deterministic_and_previous_state_is_not_mutated() -> None:
    events = [intent(), acknowledgement(), fill(), *protective_events()]
    normalized = normalize_execution_events(events)
    initial = CanonicalExecutionState()
    next_state = reduce_execution_event(initial, normalized[0])

    assert not initial.processed_event_ids
    assert next_state.processed_event_ids
    assert assert_replay_deterministic(events) == replay_execution_events(events).state_hash
    read_model = next_state.to_dict()
    read_model["orders"].clear()
    assert next_state.orders


def test_partial_exit_uses_fifo_actual_fills_and_fees() -> None:
    events = [
        intent(quantity="2"),
        acknowledgement(status="filled"),
        fill(quantity="2", fee="2"),
        intent(
            recorded_at="2026-01-01T00:01:00Z",
            intent_id="exit-1",
            side="SELL",
            quantity="1",
            price="120",
        ),
        acknowledgement(
            recorded_at="2026-01-01T00:01:01Z",
            intent_id="exit-1",
            order_id="exit-order-1",
            side="SELL",
            status="filled",
        ),
        fill(
            recorded_at="2026-01-01T00:01:02Z",
            fill_id="exit-fill-1",
            intent_id="exit-1",
            order_id="exit-order-1",
            side="SELL",
            quantity="1",
            price="120",
            fee="1",
        ),
    ]
    state = replay_execution_events(events)
    position = state.positions["BTC-EUR"]

    assert position.quantity == Decimal("1")
    assert position.cost_basis_eur == Decimal("101")
    assert position.realized_pnl_eur == Decimal("18")
    assert position.realized_pnl_complete is True
    assert position.strategy_id == "STRATEGY_A"


def test_full_exit_clears_risk_and_protection() -> None:
    events = [intent(), acknowledgement(status="filled"), fill(), *protective_events()]
    events.append(
        fill(
            recorded_at="2026-01-01T00:01:00Z",
            fill_id="stop-fill",
            intent_id="stop-1",
            order_id="stop-order-1",
            side="SELL",
            quantity="2",
            price="90",
            fee="1",
        )
    )
    state = replay_execution_events(events)
    position = state.positions["BTC-EUR"]

    assert position.quantity == 0
    assert position.open_risk_eur == 0
    assert position.protected_quantity == 0
    assert position.protection_state is ProtectionState.NOT_REQUIRED


def test_stop_and_discretionary_exit_race_cannot_create_negative_position() -> None:
    discretionary_exit = fill(
        recorded_at="2026-01-01T00:01:00Z",
        fill_id="discretionary-exit",
        intent_id="exit-1",
        order_id="exit-order-1",
        side="SELL",
        quantity="2",
        price="110",
        fee="1",
    )
    late_stop_fill = fill(
        recorded_at="2026-01-01T00:01:01Z",
        fill_id="late-stop-fill",
        intent_id="stop-1",
        order_id="stop-order-1",
        side="SELL",
        quantity="2",
        price="90",
        fee="1",
    )
    state = replay_execution_events(
        [
            intent(),
            acknowledgement(status="filled"),
            fill(),
            *protective_events(),
            discretionary_exit,
            late_stop_fill,
        ]
    )
    position = state.positions["BTC-EUR"]

    assert position.quantity == 0
    assert position.open_risk_eur == 0
    assert position.protected_quantity == 0
    assert position.protection_state is ProtectionState.NOT_REQUIRED
    assert any(
        gap["code"] == "SELL_WITHOUT_RECONSTRUCTABLE_COST_BASIS"
        and gap["fill_id"] == "late-stop-fill"
        for gap in state.evidence_gaps
    )


def test_partial_protection_and_rejection_fail_closed() -> None:
    base = [intent(), acknowledgement(status="filled"), fill()]
    partial = replay_execution_events([*base, *protective_events(quantity="1")])
    assert partial.positions["BTC-EUR"].protection_state is ProtectionState.PARTIAL
    assert partial.positions["BTC-EUR"].unprotected_quantity == Decimal("1")
    assert partial.positions["BTC-EUR"].open_risk_eur == Decimal("111")

    rejected_events = [*base, protective_events()[0]]
    rejected_events.append(
        event(
            "ORDER_REJECTED",
            "2026-01-01T00:00:04Z",
            intent_id="stop-1",
            client_order_id="client:stop-1",
            market="BTC-EUR",
            reason_code="VENUE_REJECTED",
        )
    )
    rejected = replay_execution_events(rejected_events)
    position = rejected.positions["BTC-EUR"]
    assert position.protection_state is ProtectionState.REJECTED
    assert position.open_risk_eur == position.cost_basis_eur


def test_multiple_strategy_ownership_is_explicit_and_partial_exit_preserves_lots() -> None:
    second_intent = intent(
        recorded_at="2026-01-01T00:01:00Z",
        intent_id="intent-2",
        quantity="1",
        strategy_id="STRATEGY_B",
        strategy_dna_hash="dna-b",
        signal_id="signal-b",
    )
    second_ack = acknowledgement(
        recorded_at="2026-01-01T00:01:01Z",
        intent_id="intent-2",
        order_id="order-2",
        status="filled",
    )
    second_fill = fill(
        recorded_at="2026-01-01T00:01:02Z",
        fill_id="fill-2",
        intent_id="intent-2",
        order_id="order-2",
        quantity="1",
        strategy_id="STRATEGY_B",
        strategy_dna_hash="dna-b",
        signal_id="signal-b",
    )
    state = replay_execution_events(
        [intent(quantity="1"), acknowledgement(status="filled"), fill(quantity="1"), second_intent, second_ack, second_fill]
    )
    position = state.positions["BTC-EUR"]

    assert position.quantity == Decimal("2")
    assert position.ownership_state is OwnershipState.MIXED
    assert position.strategy_id is None
    assert {lot.strategy_id for lot in position.lots} == {"STRATEGY_A", "STRATEGY_B"}


def test_unknown_exchange_balance_and_manual_sell_do_not_fabricate_ownership_or_pnl(
    tmp_path,
) -> None:
    observed = event(
        "RECONCILIATION_OBSERVED",
        "2026-01-01T00:00:00Z",
        reconciliation_id="recon-1",
        balances=[
            {"symbol": "EUR", "available": "500", "inOrder": "10"},
            {"symbol": "TAO", "available": "2", "inOrder": "0"},
        ],
    )
    manual_sell = fill(
        recorded_at="2026-01-01T00:00:02Z",
        fill_id="manual-sell",
        intent_id="manual-intent",
        order_id="manual-order",
        market="NPC-EUR",
        side="SELL",
        quantity="5",
        strategy_id="OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE",
    )
    state = replay_execution_events([observed, manual_sell])

    assert state.eur_total == Decimal("510")
    assert state.eur_exchange_available == Decimal("500")
    assert state.positions["TAO-EUR"].quantity == Decimal("2")
    assert state.positions["TAO-EUR"].ownership_state is OwnershipState.UNKNOWN
    assert state.positions["TAO-EUR"].open_risk_eur is None
    assert any(gap["code"] == "SELL_WITHOUT_RECONSTRUCTABLE_COST_BASIS" for gap in state.evidence_gaps)
    tracker = PositionTracker(tmp_path / "positions.json")
    tracker.apply_canonical_state(state)
    assert tracker.positions["TAO-EUR"].current_open_risk is None
    assert tracker.positions["TAO-EUR"].risk_complete is False


def test_reserved_capital_survives_partial_fill_and_releases_on_rejection() -> None:
    open_state = replay_execution_events([intent(quantity="2"), acknowledgement()])
    assert open_state.eur_reserved_for_orders == Decimal("200")

    partial_state = replay_execution_events(
        [intent(quantity="2"), acknowledgement(), fill(quantity="0.5", fee="0.1")]
    )
    assert partial_state.eur_reserved_for_orders == Decimal("150.0")

    rejected = event(
        "ORDER_REJECTED",
        "2026-01-01T00:00:03Z",
        intent_id="intent-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
    )
    rejected_state = replay_execution_events([intent(), rejected])
    assert rejected_state.eur_reserved_for_orders == 0


def test_reconciliation_discovers_manual_open_order_without_fabricating_owner() -> None:
    observed = event(
        "RECONCILIATION_OBSERVED",
        "2026-01-01T00:00:00Z",
        reconciliation_id="manual-order-recon",
        balances=[{"symbol": "EUR", "available": "500", "inOrder": "50"}],
        remote_open_orders=[
            {
                "order_id": "manual-order-1",
                "client_order_id": None,
                "market": "ETH-EUR",
                "side": "buy",
                "order_type": "limit",
                "status": "new",
                "quantity": "0.02",
                "filled_quantity": "0",
                "estimated_price": "2500",
            }
        ],
    )

    state = replay_execution_events([observed, deepcopy(observed)])
    order = next(iter(state.orders.values()))

    assert order.status is CanonicalOrderStatus.OPEN
    assert order.exchange_order_id == "manual-order-1"
    assert order.strategy_id is None
    assert order.ownership_state is OwnershipState.UNKNOWN
    assert state.eur_reserved_for_orders == Decimal("50.00")
    assert state.eur_available_after_local_reservations == Decimal("450.00")
    assert len(state.orders) == 1
    assert any(
        gap["code"] == "REMOTE_OPEN_ORDER_WITH_UNKNOWN_LOCAL_INTENT"
        for gap in state.evidence_gaps
    )


def test_unknown_cost_basis_serializes_as_none_never_fictive_zero() -> None:
    observed = event(
        "RECONCILIATION_OBSERVED",
        "2026-01-01T00:00:00Z",
        reconciliation_id="unknown-basis",
        balances=[{"symbol": "TAO", "available": "2", "inOrder": "0"}],
    )
    state = replay_execution_events([observed])
    serialized = state.to_dict()["positions"]["TAO-EUR"]

    assert serialized["cost_basis_known"] is False
    assert serialized["cost_basis_eur"] is None
    assert serialized["average_entry_price"] is None
    assert serialized["lots"][0]["cost_basis_eur"] is None


def test_order_retains_portfolio_risk_and_execution_provenance() -> None:
    local_intent = intent()
    local_intent["payload"].update(
        {
            "portfolio_target_id": "portfolio-target-1",
            "risk_approval_id": "risk-approval-1",
            "execution_intent_id": "execution-intent-1",
        }
    )
    state = replay_execution_events([local_intent])
    order = next(iter(state.orders.values()))

    assert order.ownership_state is OwnershipState.KNOWN
    assert order.portfolio_target_id == "portfolio-target-1"
    assert order.risk_approval_id == "risk-approval-1"
    assert order.execution_intent_id == "execution-intent-1"


def test_portfolio_target_risk_and_execution_chain_replays_before_order() -> None:
    target = event(
        "PORTFOLIO_TARGET",
        "2026-01-01T00:00:00Z",
        target_id="target-1",
        market="BTC-EUR",
    )
    approval = event(
        "RISK_APPROVAL",
        "2026-01-01T00:00:01Z",
        approval_id="approval-1",
        target_id="target-1",
        approved=True,
    )
    execution_intent = event(
        "EXECUTION_INTENT",
        "2026-01-01T00:00:02Z",
        execution_intent_id="execution-1",
        target_id="target-1",
        risk_approval_id="approval-1",
    )
    order_intent = intent(recorded_at="2026-01-01T00:00:03Z")
    order_intent["payload"].update(
        {
            "portfolio_target_id": "target-1",
            "risk_approval_id": "approval-1",
            "execution_intent_id": "execution-1",
        }
    )
    state = replay_execution_events(
        [target, approval, execution_intent, order_intent]
    )

    assert set(state.portfolio_targets) == {"target-1"}
    assert set(state.risk_approvals) == {"approval-1"}
    assert set(state.execution_intents) == {"execution-1"}
    assert not state.evidence_gaps


def test_unknown_cannot_be_promoted_to_open_by_stale_status() -> None:
    unknown = event(
        "ORDER_STATE_UNKNOWN",
        "2026-01-01T00:00:03Z",
        intent_id="intent-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
        reason_code="AMBIGUOUS_TRANSPORT_FAILURE",
    )
    stale_open = event(
        "ORDER_STATUS_OBSERVED",
        "2026-01-01T00:00:04Z",
        checkpoint_id="stale-open",
        order_id="order-1",
        client_order_id="client:intent-1",
        market="BTC-EUR",
        status="new",
        exchange_updated_at="2026-01-01T00:00:01Z",
    )
    state = replay_execution_events([intent(), unknown, stale_open])

    assert order_by_intent(state, "intent-1").status is CanonicalOrderStatus.UNKNOWN
    assert state.eur_reserved_for_orders == Decimal("200")


def test_invalid_negative_or_overprotected_economics_never_escape_invariants() -> None:
    events = [intent(quantity="1"), acknowledgement(status="filled"), fill(quantity="1"), *protective_events(quantity="2")]
    state = replay_execution_events(events)
    position = state.positions["BTC-EUR"]

    assert position.protected_quantity == position.quantity
    assert position.protection_state is ProtectionState.CONFLICT
    assert position.quantity >= 0
    assert state.eur_reserved_for_orders >= 0


def test_position_tracker_is_a_derived_canonical_read_model(tmp_path) -> None:
    state = replay_execution_events(
        [intent(), acknowledgement(status="filled"), fill(), *protective_events()]
    )
    tracker = PositionTracker(tmp_path / "positions.json")
    snapshot = tracker.apply_canonical_state(state)
    position = tracker.positions["BTC-EUR"]

    assert snapshot["state_source"] == "CANONICAL_EXECUTION_STATE"
    assert snapshot["canonical_state_hash"] == state.state_hash
    assert snapshot["read_model_only"] is True
    assert position.strategy_id == "STRATEGY_A"
    assert position.stop_price == Decimal("90")
    assert position.protected_quantity == Decimal("2")
    assert position.current_open_risk == Decimal("21.0")
    assert position.state_source == "CANONICAL_EXECUTION_STATE"


def test_dual_read_divergence_classifies_lost_ownership_stop_and_risk() -> None:
    state = replay_execution_events(
        [intent(), acknowledgement(status="filled"), fill(), *protective_events()]
    )
    legacy = {
        "positions": {
            "BTC-EUR": {
                "owned_quantity": "2",
                "strategy_id": "",
                "stop_price": None,
                "current_open_risk": "0",
                "realized_pnl": "0",
            }
        }
    }
    report = build_execution_divergence_report(legacy, state)
    fields = {row["field"]: row for row in report["divergences"]}

    assert fields["strategy_ownership"]["classification"] == "REAL_DEFECT"
    assert fields["effective_stop_price"]["classification"] == "REAL_DEFECT"
    assert fields["protected_quantity"]["classification"] == "EXPECTED_SCHEMA_DIFFERENCE"
    assert fields["open_risk_eur"]["classification"] == "REAL_DEFECT"
    assert report["legacy_can_overwrite_canonical"] is False
