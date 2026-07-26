from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from core.contracts import (
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    ResearchStatus,
)
from data.database import Database
from execution.execution import (
    BitvavoSpotClient,
    ExecutionMarketRules,
    LiveCapability,
    LivePreflight,
    PaperBroker,
)
from risk.risk_manager import OperationalDegradation
from utils.common import utc_now


def intent(
    key: str,
    *,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    price: Decimal | None = None,
) -> OrderIntent:
    return OrderIntent(
        intent_id=f"intent-{key}",
        idempotency_key=key,
        market="BTC-EUR",
        side=side,
        order_type=order_type,
        quantity=Decimal("0.01"),
        limit_price=price,
        strategy_id="unit",
    )


def test_paper_broker_idempotency_partial_fill_and_no_short(tmp_path) -> None:
    ledger = tmp_path / "paper.jsonl"
    broker = PaperBroker(
        initial_balances={"EUR": Decimal("1000")},
        market_rules={
            "BTC-EUR": ExecutionMarketRules(minimum_order_value_eur=Decimal("5"))
        },
        ledger_path=ledger,
    )
    order = broker.submit(
        intent("one"),
        market_price=Decimal("10000"),
        available_liquidity=Decimal("0.004"),
    )
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert broker.submit(intent("one"), market_price=Decimal("10000")).order_id == order.order_id
    restarted = PaperBroker(ledger_path=ledger)
    assert restarted.submit(intent("one"), market_price=Decimal("10000")).order_id == order.order_id
    sell = broker.submit(
        intent("sell", side=OrderSide.SELL),
        market_price=Decimal("10000"),
    )
    assert sell.status is OrderStatus.REJECTED
    assert sell.rejection_code == "INSUFFICIENT_OWNED_UNITS"
    assert broker.reconcile().healthy


def test_paper_broker_multi_fill_and_restart_rebuilds_balances(tmp_path) -> None:
    ledger = tmp_path / "paper-multifill.jsonl"
    initial = {"EUR": Decimal("1000")}
    broker = PaperBroker(
        initial_balances=initial,
        market_rules={
            "BTC-EUR": ExecutionMarketRules(minimum_order_value_eur=Decimal("5"))
        },
        ledger_path=ledger,
    )
    partial = broker.submit(
        intent("multi"),
        market_price=Decimal("10000"),
        available_liquidity=Decimal("0.004"),
    )
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    completed = broker.update_market(
        "BTC-EUR",
        market_price=Decimal("10010"),
        available_liquidity=Decimal("0.006"),
    )[0]
    assert completed.status is OrderStatus.FILLED
    assert completed.filled_quantity == Decimal("0.01000000")
    assert len(broker.fills) == 2
    restarted = PaperBroker(
        initial_balances=initial,
        market_rules={
            "BTC-EUR": ExecutionMarketRules(minimum_order_value_eur=Decimal("5"))
        },
        ledger_path=ledger,
    )
    assert restarted.balances == broker.balances
    assert len(restarted.fills) == 2
    assert restarted.reconcile().healthy
    assert not restarted.open_orders


def test_isolated_test_only_paper_acceptance_populates_durable_tables(
    tmp_path,
) -> None:
    database = Database(sqlite_path=tmp_path / "paper-acceptance.db")
    database.migrate()
    broker = PaperBroker(
        initial_balances={"EUR": Decimal("1000")},
        market_rules={
            "BTC-EUR": ExecutionMarketRules(minimum_order_value_eur=Decimal("5"))
        },
        ledger_path=tmp_path / "TEST_ONLY_paper.jsonl",
    )
    order = broker.submit(
        intent("TEST_ONLY-acceptance"),
        market_price=Decimal("10000"),
        available_liquidity=Decimal("0.004"),
    )
    order = broker.update_market(
        "BTC-EUR",
        market_price=Decimal("10010"),
        available_liquidity=Decimal("0.006"),
    )[0]
    candidate_id = "TEST_ONLY_NEVER_REAL_CANDIDATE"
    database.upsert_records(
        "orders",
        [
            {
                **order.model_dump(mode="json"),
                "candidate_id": candidate_id,
                "mode": "paper",
                "status": order.status.value,
            }
        ],
    )
    database.upsert_records(
        "fills",
        [
            {
                **fill.model_dump(mode="json"),
                "candidate_id": candidate_id,
                "mode": "paper",
                "status": "FILLED",
            }
            for fill in broker.fills
        ],
    )
    balances = broker.balance_snapshot()
    database.upsert_records(
        "balances",
        [
            {
                "external_id": f"TEST_ONLY:{asset}",
                "candidate_id": candidate_id,
                "mode": "paper",
                "asset": asset,
                **{key: str(value) for key, value in values.items()},
                "status": "RECONCILED",
            }
            for asset, values in balances.items()
        ],
    )
    database.upsert_records(
        "positions",
        [
            {
                "external_id": "TEST_ONLY:BTC-EUR",
                "candidate_id": candidate_id,
                "mode": "paper",
                "market": "BTC-EUR",
                "quantity": str(broker.balances["BTC"]),
                "status": "OPEN",
            }
        ],
    )
    equity = broker.balances["EUR"] + broker.balances["BTC"] * Decimal("10010")
    database.upsert_records(
        "pnl_snapshots",
        [
            {
                "external_id": "TEST_ONLY:pnl",
                "candidate_id": candidate_id,
                "mode": "paper",
                "numeric_value": float(equity - Decimal("1000")),
                "status": "MARKED",
            }
        ],
    )
    reconciliation = broker.reconcile()
    database.upsert_records(
        "risk_events",
        [
            {
                "external_id": "TEST_ONLY:reconciliation",
                "candidate_id": candidate_id,
                "mode": "paper",
                "status": "PASSED" if reconciliation.healthy else "FAILED",
                "reason_codes": reconciliation.reason_codes,
            }
        ],
    )
    degradation = OperationalDegradation(
        state_path=tmp_path / "TEST_ONLY_degradation.json",
        audit_path=tmp_path / "TEST_ONLY_degradation.jsonl",
    )
    killed = degradation.evaluate(kill_switch=("NEGATIVE_BALANCE",))
    database.upsert_records(
        "kill_switch_events",
        [
            {
                "external_id": "TEST_ONLY:kill-switch",
                "candidate_id": candidate_id,
                "mode": "paper",
                "status": killed["state"],
                "reason_codes": killed["reason_codes"],
            }
        ],
    )
    counts = database.health()["table_counts"]
    assert counts["orders"] == 1
    assert counts["fills"] == 2
    assert counts["balances"] == 2
    assert counts["positions"] == 1
    assert counts["pnl_snapshots"] == 1
    assert counts["risk_events"] == 1
    assert counts["kill_switch_events"] == 1
    database.close()


def test_limit_reservation_and_cancel(tmp_path) -> None:
    broker = PaperBroker(ledger_path=tmp_path / "paper.jsonl")
    order = broker.submit(
        intent("limit", order_type=OrderType.LIMIT, price=Decimal("9000")),
        market_price=Decimal("10000"),
    )
    assert order.status is OrderStatus.OPEN
    assert broker.balance_snapshot()["EUR"]["in_order"] > 0
    assert broker.cancel(order.order_id).status is OrderStatus.CANCELLED
    assert broker.balance_snapshot()["EUR"]["in_order"] == 0


def test_live_preflight_is_fail_closed(isolated_settings) -> None:
    result = LivePreflight.evaluate(
        isolated_settings,
        markets=("BTC-EUR",),
        strategy_status=ResearchStatus.PAPER_CANDIDATE,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
    )
    assert not result.passed
    assert result.capability is None
    assert not ({"withdraw", "withdrawal", "transfer"} & set(BitvavoSpotClient.__dict__))


@pytest.mark.asyncio
async def test_expired_capability_never_reaches_network(tmp_path) -> None:
    class Session:
        def post(self, *args, **kwargs):
            raise AssertionError("network must not be reached")

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now() - timedelta(minutes=6),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("100"),
        maximum_total_eur=Decimal("100"),
        maximum_open_positions=1,
    )
    with pytest.raises(ExecutionBlocked, match="expired"):
        await client.submit_order(
            intent("live"),
            capability=capability,
            estimated_price=Decimal("10000"),
            reconciled_owned_quantity=Decimal("0"),
        )


@pytest.mark.asyncio
async def test_live_cancel_requires_capability_and_is_audited(tmp_path) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {"orderId": "order-1", "status": "canceled"}

    class Session:
        def delete(self, *args, **kwargs):
            del args, kwargs
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "live.jsonl")
    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("100"),
        maximum_total_eur=Decimal("100"),
        maximum_open_positions=1,
    )
    result = await client.cancel_order(
        market="BTC-EUR",
        order_id="order-1",
        capability=capability,
    )
    assert result["status"] == "canceled"
    assert ledger.events()[-1]["event_type"] == "ORDER_CANCELLED"
