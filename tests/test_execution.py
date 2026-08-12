from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal

import pytest

from core.contracts import (
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
)
from data.database import Database
from execution.execution import (
    BitvavoSpotClient,
    ExecutionMarketRules,
    LiveCapability,
    LivePreflight,
    PaperBroker,
    market_rules_from_bitvavo_metadata,
    minimum_protectable_entry_notional,
    plan_bounded_entry_order,
    quantity_is_protectable_at_stop,
)
from portfolio.contracts import (
    ExecutionIntent,
    ExecutionStyle,
    PortfolioTarget,
    RiskApproval,
)
from portfolio.targets import CanonicalExecutionChain
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


def attach_canonical_chain(
    order: OrderIntent,
    *,
    estimated_price: Decimal,
) -> tuple[OrderIntent, CanonicalExecutionChain]:
    now = utc_now()
    notional = order.quantity * estimated_price
    target = PortfolioTarget.create(
        market=order.market,
        current_quantity=Decimal("0"),
        current_notional_eur=Decimal("0"),
        target_weight=min(Decimal("1"), notional / Decimal("100000")),
        target_notional_eur=notional,
        target_quantity=order.quantity,
        source_intent_ids=(f"strategy-intent:{order.intent_id}",),
        source_strategies=(order.strategy_id,),
        confidence=Decimal("1"),
        expected_net_edge=Decimal("0.01"),
        risk_budget_eur=notional,
        cluster=None,
        generated_at=now,
        expires_at=now + timedelta(minutes=10),
        portfolio_state_hash="unit-test-portfolio-state",
        cost_model_version="unit-test-cost-v1",
        reason_codes=("UNIT_TEST_APPROVED_TARGET",),
    )
    approval = RiskApproval.create(
        target_id=target.target_id,
        approved=True,
        approved_delta_quantity=target.delta_quantity,
        risk_eur=notional,
        reason_codes=("APPROVED",),
        policy_version="unit-test-risk-v1",
        account_state_hash="unit-test-account-state",
        approved_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    execution_intent = ExecutionIntent.create(
        target=target,
        approval=approval,
        style=(
            ExecutionStyle.MARKET_WITHIN_BOUNDS
            if order.order_type is OrderType.MARKET
            else ExecutionStyle.PASSIVE_LIMIT
        ),
        maximum_notional_eur=order.maximum_notional_eur or notional,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        reason_codes=("UNIT_TEST_CHAIN",),
    )
    chain = CanonicalExecutionChain(
        target=target,
        approval=approval,
        execution=execution_intent,
    )
    linked = order.model_copy(
        update={
            "portfolio_target_id": target.target_id,
            "risk_approval_id": approval.approval_id,
            "execution_intent_id": execution_intent.execution_intent_id,
        }
    )
    return linked, chain


def test_protectable_minimum_is_valid_at_entry_and_stop() -> None:
    rules = ExecutionMarketRules(
        minimum_order_amount=Decimal("0.01"),
        minimum_order_value_eur=Decimal("5"),
        quantity_decimals=3,
        notional_decimals=2,
    )
    minimum = minimum_protectable_entry_notional(
        entry_price=Decimal("100"),
        stop_price=Decimal("80"),
        rules=rules,
    )
    assert minimum == Decimal("7.20")
    quantity = rules.amount(minimum / Decimal("100"))
    assert quantity_is_protectable_at_stop(
        quantity=quantity,
        stop_price=Decimal("80"),
        rules=rules,
    )


def test_protectable_minimum_rejects_invalid_stop() -> None:
    with pytest.raises(ExecutionBlocked, match="protective stop prices"):
        minimum_protectable_entry_notional(
            entry_price=Decimal("100"),
            stop_price=Decimal("100"),
            rules=ExecutionMarketRules(),
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


def test_bounded_limit_planner_uses_ioc_only_when_venue_rounding_fits() -> None:
    rules = market_rules_from_bitvavo_metadata(
        {
            "quantityDecimals": 4,
            "notionalDecimals": 2,
            "tickSize": "0.01",
            "minOrderInBaseAsset": "0.001",
            "minOrderInQuoteAsset": "5",
        }
    )
    limit = plan_bounded_entry_order(
        requested_notional_eur=Decimal("25"),
        best_ask=Decimal("100"),
        estimated_average_price=Decimal("100.02"),
        maximum_slippage_bps=Decimal("20"),
        rules=rules,
        limit_enabled=True,
    )
    micro = plan_bounded_entry_order(
        requested_notional_eur=Decimal("5"),
        best_ask=Decimal("1667.60"),
        estimated_average_price=Decimal("1667.60"),
        maximum_slippage_bps=Decimal("25"),
        rules=ExecutionMarketRules(
            minimum_order_amount=Decimal("0.00304033"),
            minimum_order_value_eur=Decimal("5"),
            quantity_decimals=8,
            notional_decimals=2,
            tick_size=Decimal("0.01"),
        ),
        limit_enabled=True,
    )

    assert limit.order_type is OrderType.LIMIT
    assert limit.time_in_force is OrderTimeInForce.IOC
    assert limit.planned_notional_eur <= Decimal("25")
    assert micro.order_type is OrderType.MARKET
    assert micro.fallback_reason == (
        "VENUE_MINIMUM_OR_ROUNDING_REQUIRES_QUOTE_MARKET"
    )


def test_bounded_limit_planner_supports_persistent_gtc_entry() -> None:
    plan = plan_bounded_entry_order(
        requested_notional_eur=Decimal("25"),
        best_ask=Decimal("100"),
        estimated_average_price=Decimal("100.02"),
        maximum_slippage_bps=Decimal("20"),
        rules=ExecutionMarketRules(
            minimum_order_amount=Decimal("0.001"),
            minimum_order_value_eur=Decimal("5"),
            quantity_decimals=4,
            notional_decimals=2,
            tick_size=Decimal("0.01"),
        ),
        limit_enabled=True,
        time_in_force=OrderTimeInForce.GTC,
    )

    assert plan.order_type is OrderType.LIMIT
    assert plan.time_in_force is OrderTimeInForce.GTC
    assert plan.planned_notional_eur <= Decimal("25")


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
async def test_live_buy_cannot_bypass_portfolio_target_and_risk_chain(tmp_path) -> None:
    class Session:
        def post(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("network must not be reached")

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-chain-required.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("100"),
        maximum_total_eur=Decimal("100"),
        maximum_open_positions=1,
    )

    with pytest.raises(ExecutionBlocked, match="portfolio target"):
        await client.submit_order(
            intent("bypass-attempt"),
            capability=capability,
            estimated_price=Decimal("10000"),
            reconciled_owned_quantity=Decimal("0"),
            reconciled_total_exposure_eur=Decimal("0"),
            reconciled_open_positions=0,
        )
    assert client.ledger.events() == []


@pytest.mark.asyncio
async def test_live_daily_buy_limit_blocks_before_network(
    tmp_path,
) -> None:
    class Session:
        def post(self, *args, **kwargs):
            raise AssertionError("network must not be reached")

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "live.jsonl")
    ledger.append(
        "ORDER_INTENT",
        {
            "intent_id": "prior",
            "idempotency_key": "prior",
            "client_order_id": "prior",
            "market": "BTC-EUR",
            "side": "BUY",
            "quantity": "0.001",
        },
    )
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
        maximum_order_eur=Decimal("5"),
        maximum_total_eur=Decimal("10"),
        maximum_open_positions=1,
        maximum_new_orders_per_day=1,
    )
    second, second_chain = attach_canonical_chain(
        intent("second").model_copy(
            update={"quantity": Decimal("0.0005")}
        ),
        estimated_price=Decimal("10000"),
    )
    with pytest.raises(ExecutionBlocked, match="daily new-order"):
        await client.submit_order(
            second,
            capability=capability,
            estimated_price=Decimal("10000"),
            reconciled_owned_quantity=Decimal("0"),
            reconciled_total_exposure_eur=Decimal("0"),
            reconciled_open_positions=0,
            canonical_chain=second_chain,
        )


@pytest.mark.asyncio
async def test_market_buy_uses_exact_quote_notional(tmp_path) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {
                "orderId": "order-1",
                "clientOrderId": "client-1",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "filled",
                "filledAmount": "0.0005",
                "filledAmountQuote": "5",
                "feePaid": "0.0125",
                "feeCurrency": "EUR",
                "created": 1786250400000,
                "updated": 1786250400250,
            }

    class Session:
        body = None

        def post(self, *args, **kwargs):
            del args
            self.body = kwargs["data"]
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("5"),
        maximum_total_eur=Decimal("10"),
        maximum_open_positions=1,
    )
    order = intent("quote-buy").model_copy(
        update={
            # Deliberately rounds back above EUR 5.  The quote-denominated
            # cap, not this estimated base quantity, is canonical.
            "quantity": Decimal("0.0005000000000000000000000001"),
            "maximum_notional_eur": Decimal("5"),
        }
    )
    order, chain = attach_canonical_chain(
        order,
        estimated_price=Decimal("10000"),
    )
    await client.submit_order(
        order,
        capability=capability,
        estimated_price=Decimal("10000"),
        reconciled_owned_quantity=Decimal("0"),
        reconciled_total_exposure_eur=Decimal("0"),
        reconciled_open_positions=0,
        canonical_chain=chain,
    )
    payload = json.loads(session.body)
    assert Decimal(payload["amountQuote"]) == Decimal("5")
    assert "amount" not in payload
    fill = next(
        row for row in client.ledger.events() if row["event_type"] == "FILL"
    )
    assert fill["payload"]["fee_known"] is True
    assert fill["payload"]["fee_eur"] == "0.0125"
    assert Decimal(fill["payload"]["price"]) == Decimal("10000")
    assert fill["payload"]["strategy_id"] == order.strategy_id
    assert fill["payload"]["intent_id"] == order.intent_id
    assert fill["payload"]["filled_at"].endswith("+00:00")
    assert fill["payload"]["exchange_created_at"].endswith("+00:00")
    assert fill["payload"]["exchange_updated_at"].endswith("+00:00")
    assert fill["payload"]["received_at"] is not None
    events = client.ledger.events()
    order_intent = next(
        row for row in events if row["event_type"] == "ORDER_INTENT"
    )
    acknowledgement = next(
        row for row in events if row["event_type"] == "ORDER_ACKNOWLEDGED"
    )
    assert order_intent["payload"]["submission_started_at"] is not None
    assert acknowledgement["payload"][
        "acknowledgement_received_at"
    ] is not None
    assert acknowledgement["payload"]["exchange_created_at"] == fill[
        "payload"
    ]["exchange_created_at"]


@pytest.mark.asyncio
async def test_ioc_limit_serializes_policy_and_records_terminal_partial(
    tmp_path,
) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {
                "orderId": "ioc-order-1",
                "clientOrderId": "ioc-client-1",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "canceled",
                "filledAmount": "0.001",
                "filledAmountQuote": "10.01",
                "feePaid": "0.025025",
                "feeCurrency": "EUR",
                "timeInForce": "IOC",
            }

    class Session:
        body = None

        def post(self, *args, **kwargs):
            del args
            self.body = kwargs["data"]
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-ioc.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("25"),
        maximum_total_eur=Decimal("25"),
        maximum_open_positions=1,
    )
    order = intent(
        "ioc-entry",
        order_type=OrderType.LIMIT,
        price=Decimal("10010"),
    ).model_copy(
        update={
            "quantity": Decimal("0.002"),
            "time_in_force": OrderTimeInForce.IOC,
            "maximum_notional_eur": Decimal("25"),
        }
    )
    order, chain = attach_canonical_chain(
        order,
        estimated_price=Decimal("10000"),
    )

    await client.submit_order(
        order,
        capability=capability,
        estimated_price=Decimal("10000"),
        reconciled_owned_quantity=Decimal("0"),
        reconciled_total_exposure_eur=Decimal("0"),
        reconciled_open_positions=0,
        canonical_chain=chain,
    )

    payload = json.loads(session.body)
    assert payload["orderType"] == "limit"
    assert payload["timeInForce"] == "IOC"
    assert payload["postOnly"] is False
    assert payload["price"] == "10010"
    fill = next(
        event
        for event in client.ledger.events()
        if event["event_type"] == "FILL"
    )
    assert fill["payload"]["quantity"] == "0.001"
    assert fill["payload"]["status"] == "PARTIALLY_FILLED_FINAL"
    intent_event = next(
        event["payload"]
        for event in client.ledger.events()
        if event["event_type"] == "ORDER_INTENT"
    )
    assert intent_event["time_in_force"] == "IOC"


@pytest.mark.asyncio
async def test_live_sell_can_close_appreciated_position_above_buy_cap(
    tmp_path,
) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {
                "orderId": "sell-order-1",
                "clientOrderId": "sell-client-1",
                "market": "BTC-EUR",
                "status": "filled",
            }

    class Session:
        body = None

        def post(self, *args, **kwargs):
            del args
            self.body = kwargs["data"]
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-sell.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("5"),
        maximum_total_eur=Decimal("10"),
        maximum_open_positions=1,
    )
    sell = intent("close-appreciated", side=OrderSide.SELL).model_copy(
        update={"quantity": Decimal("0.001")}
    )
    await client.submit_order(
        sell,
        capability=capability,
        estimated_price=Decimal("6000"),
        reconciled_owned_quantity=Decimal("0.001"),
        reconciled_total_exposure_eur=Decimal("6"),
        reconciled_open_positions=1,
    )
    payload = json.loads(session.body)
    assert payload["side"] == "sell"
    assert payload["amount"] == "0.001"
    assert "amountQuote" not in payload


@pytest.mark.asyncio
async def test_live_native_stop_loss_uses_bitvavo_trigger_contract(
    tmp_path,
    monkeypatch,
) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {
                "orderId": "stop-order-1",
                "market": "BTC-EUR",
                "side": "sell",
                "status": "awaitingTrigger",
                "filledAmount": "0",
            }

    class Session:
        body = None

        def post(self, *args, **kwargs):
            del args
            self.body = kwargs["data"]
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-native-stop.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("10"),
        maximum_total_eur=Decimal("10"),
        maximum_open_positions=1,
    )
    stop = intent("native-stop", side=OrderSide.SELL).model_copy(
        update={
            "order_type": OrderType.STOP_LOSS,
            "quantity": Decimal("0.001"),
            "trigger_price": Decimal("9500"),
            "trigger_reference": "bestBid",
        }
    )

    result = await client.submit_order(
        stop,
        capability=capability,
        estimated_price=Decimal("10000"),
        reconciled_owned_quantity=Decimal("0.001"),
        reconciled_total_exposure_eur=Decimal("10"),
        reconciled_open_positions=1,
    )

    payload = json.loads(session.body)
    assert result["status"] == "awaitingTrigger"
    assert payload["orderType"] == "stopLoss"
    assert payload["triggerAmount"] == "9500"
    assert payload["triggerType"] == "price"
    assert payload["triggerReference"] == "bestBid"
    assert payload["side"] == "sell"
    assert "price" not in payload

    async def balances():
        return []

    async def open_orders(_market):
        return [
            {
                "orderId": "stop-order-1",
                "clientOrderId": client.client_order_id_for(
                    stop.idempotency_key
                ),
                "status": "awaitingTrigger",
            }
        ]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    reconciliation = await client.reconcile(markets=("BTC-EUR",))
    assert reconciliation.healthy is True
    assert reconciliation.local_open_orders == 1
    assert reconciliation.remote_open_orders == 1


@pytest.mark.asyncio
async def test_definitive_live_rejection_is_terminal_for_reconciliation(
    tmp_path,
    monkeypatch,
) -> None:
    class Response:
        status = 400

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self, content_type=None):
            del content_type
            return {"errorCode": 205}

    class Session:
        def post(self, *args, **kwargs):
            del args, kwargs
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-rejected.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("5"),
        maximum_total_eur=Decimal("10"),
        maximum_open_positions=1,
    )
    sell = intent("definitive-reject", side=OrderSide.SELL)

    with pytest.raises(ExecutionBlocked):
        await client.submit_order(
            sell,
            capability=capability,
            estimated_price=Decimal("1000"),
            reconciled_owned_quantity=Decimal("0.01"),
        )

    event_types = [
        event["event_type"] for event in client.ledger.events()
    ]
    assert event_types == ["ORDER_INTENT", "ORDER_REJECTED"]
    rejected = client.ledger.events()[-1]["payload"]
    assert rejected["definitive"] is True
    assert rejected["http_status"] == 400
    assert rejected["venue_error_code"] == "205"

    async def balances():
        return []

    async def open_orders(_market):
        return []

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    reconciliation = await client.reconcile(markets=("BTC-EUR",))
    assert reconciliation.healthy is True
    assert reconciliation.local_open_orders == 0


@pytest.mark.asyncio
async def test_private_read_rate_limit_is_recoverable_not_definitive(
    tmp_path,
) -> None:
    class Response:
        status = 429

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class Session:
        def get(self, *args, **kwargs):
            del args, kwargs
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-private-read.jsonl"),
    )

    with pytest.raises(ReconciliationRequired, match="temporarily unavailable"):
        await client._private_get("/v2/balance", attempts=1)


@pytest.mark.asyncio
async def test_ambiguous_live_submission_is_explicitly_unknown(
    tmp_path,
) -> None:
    class Response:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            del args
            return None

    class Session:
        def post(self, *args, **kwargs):
            del args, kwargs
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    client = BitvavoSpotClient(
        session=Session(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-unknown.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("25"),
        maximum_total_eur=Decimal("75"),
        maximum_open_positions=3,
    )

    ambiguous, ambiguous_chain = attach_canonical_chain(
        intent("ambiguous"),
        estimated_price=Decimal("1000"),
    )
    with pytest.raises(ReconciliationRequired, match="clientOrderId"):
        await client.submit_order(
            ambiguous,
            capability=capability,
            estimated_price=Decimal("1000"),
            reconciled_owned_quantity=Decimal("0"),
            reconciled_total_exposure_eur=Decimal("0"),
            reconciled_open_positions=0,
            canonical_chain=ambiguous_chain,
        )

    events = client.ledger.events()
    assert [event["event_type"] for event in events] == [
        "PORTFOLIO_TARGET",
        "RISK_APPROVAL",
        "EXECUTION_INTENT",
        "ORDER_INTENT",
        "ORDER_STATE_UNKNOWN",
    ]
    assert events[-1]["payload"]["execution_blocked_until_reconciled"] is True
    assert events[-1]["payload"]["reason_code"] == "AMBIGUOUS_HTTP_503"


@pytest.mark.asyncio
async def test_unreadable_success_response_is_unknown_not_retried(
    tmp_path,
) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            del args
            return None

        async def json(self, content_type=None):
            del content_type
            raise ValueError("invalid JSON")

    class Session:
        post_calls = 0

        def post(self, *args, **kwargs):
            del args, kwargs
            self.post_calls += 1
            return Response()

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=DurableLedger(tmp_path / "live-unreadable.jsonl"),
    )
    capability = LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("25"),
        maximum_total_eur=Decimal("75"),
        maximum_open_positions=3,
    )

    unreadable, unreadable_chain = attach_canonical_chain(
        intent("unreadable-response"),
        estimated_price=Decimal("1000"),
    )
    with pytest.raises(ReconciliationRequired, match="clientOrderId"):
        await client.submit_order(
            unreadable,
            capability=capability,
            estimated_price=Decimal("1000"),
            reconciled_owned_quantity=Decimal("0"),
            reconciled_total_exposure_eur=Decimal("0"),
            reconciled_open_positions=0,
            canonical_chain=unreadable_chain,
        )

    assert session.post_calls == 1
    events = client.ledger.events()
    assert [event["event_type"] for event in events] == [
        "PORTFOLIO_TARGET",
        "RISK_APPROVAL",
        "EXECUTION_INTENT",
        "ORDER_INTENT",
        "ORDER_STATE_UNKNOWN",
    ]
    assert events[-1]["payload"]["reason_code"] == (
        "AMBIGUOUS_RESPONSE_DECODE_FAILURE"
    )


def _append_unknown_live_intent(
    ledger,
    *,
    client_order_id: str,
    created_at=None,
) -> None:
    ledger.append(
        "ORDER_INTENT",
        {
            "intent_id": "intent-recovery",
            "idempotency_key": "recovery-key",
            "client_order_id": client_order_id,
            "market": "BTC-EUR",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": "0.01",
            "estimated_price": "1000",
            "strategy_id": "unit-recovery",
            "strategy_dna_hash": "dna-recovery",
            "signal_id": "signal-recovery",
            "created_at": created_at or utc_now(),
        },
    )
    ledger.append(
        "ORDER_STATE_UNKNOWN",
        {
            "intent_id": "intent-recovery",
            "client_order_id": client_order_id,
            "market": "BTC-EUR",
            "reason_code": "AMBIGUOUS_TRANSPORT_FAILURE",
        },
    )


@pytest.mark.asyncio
async def test_reconcile_recovers_unknown_open_order_by_client_id(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "recover-open.jsonl")
    client_id = "client-open"
    _append_unknown_live_intent(ledger, client_order_id=client_id)
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    async def balances():
        return []

    async def open_orders(_market):
        return [
            {
                "orderId": "venue-open",
                "clientOrderId": client_id,
                "market": "BTC-EUR",
                "side": "buy",
                "status": "new",
            }
        ]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is True
    assert result.local_open_orders == 1
    assert result.remote_open_orders == 1
    recovered = [
        event
        for event in ledger.events()
        if event["event_type"] == "ORDER_ACKNOWLEDGED"
    ]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["recovered"] is True


@pytest.mark.asyncio
async def test_reconcile_recovers_unknown_fill_once(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "recover-fill.jsonl")
    client_id = "client-filled"
    _append_unknown_live_intent(ledger, client_order_id=client_id)
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    filled = {
        "orderId": "venue-filled",
        "clientOrderId": client_id,
        "market": "BTC-EUR",
        "side": "buy",
        "status": "filled",
        "filledAmount": "0.01",
        "filledAmountQuote": "10.01",
        "feePaid": "0.025",
        "feeCurrency": "EUR",
    }

    async def balances():
        return []

    async def open_orders(_market):
        return []

    async def recent_orders(_market, *, limit=1000):
        assert limit == 1000
        return [filled]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    monkeypatch.setattr(client, "recent_orders", recent_orders)

    first = await client.reconcile(markets=("BTC-EUR",))
    second = await client.reconcile(markets=("BTC-EUR",))

    assert first.healthy is True
    assert second.healthy is True
    fills = [event for event in ledger.events() if event["event_type"] == "FILL"]
    acknowledgements = [
        event
        for event in ledger.events()
        if event["event_type"] == "ORDER_ACKNOWLEDGED"
    ]
    assert len(fills) == 1
    assert len(acknowledgements) == 1
    assert fills[0]["payload"]["price"] == "1001"
    assert fills[0]["payload"]["fee_known"] is True


@pytest.mark.asyncio
async def test_reconcile_keeps_unknown_state_when_lookup_fails(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "recover-failed.jsonl")
    _append_unknown_live_intent(ledger, client_order_id="client-failed")
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    async def balances():
        return []

    async def open_orders(_market):
        return []

    async def recent_orders(_market, *, limit=1000):
        del limit
        raise ReconciliationRequired("lookup unavailable")

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    monkeypatch.setattr(client, "recent_orders", recent_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is False
    assert "UNKNOWN_ORDER_STATE" in result.reason_codes
    assert "UNKNOWN_ORDER_LOOKUP_FAILED" in result.reason_codes
    assert "UNACKNOWLEDGED_LOCAL_INTENT" in result.reason_codes


@pytest.mark.asyncio
async def test_reconcile_definitively_resolves_recent_absent_unknown_order(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "recover-absent.jsonl")
    _append_unknown_live_intent(
        ledger,
        client_order_id="client-absent",
        created_at=utc_now() - timedelta(minutes=2),
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    async def balances():
        return []

    async def open_orders(_market):
        return []

    async def recent_orders(_market, *, limit=1000):
        del limit
        return []

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    monkeypatch.setattr(client, "recent_orders", recent_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is True
    rejected = [
        event
        for event in ledger.events()
        if event["event_type"] == "ORDER_REJECTED"
    ]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["definitive"] is True
    assert rejected[0]["payload"]["recovered"] is True


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
    with pytest.raises(ExecutionBlocked, match="already terminal"):
        await client.cancel_order(
            market="BTC-EUR",
            order_id="order-1",
            capability=capability,
        )


def _append_acknowledged_live_order(
    ledger,
    *,
    order_id: str,
    client_order_id: str,
    status: str = "new",
) -> None:
    ledger.append(
        "ORDER_INTENT",
        {
            "intent_id": f"intent-{order_id}",
            "idempotency_key": f"key-{order_id}",
            "client_order_id": client_order_id,
            "market": "BTC-EUR",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": "0.01",
            "limit_price": "1000",
            "estimated_price": "1000",
            "strategy_id": "cancel-race-test",
            "strategy_dna_hash": "cancel-race-dna",
            "signal_id": "cancel-race-signal",
            "created_at": utc_now(),
        },
    )
    ledger.append(
        "ORDER_ACKNOWLEDGED",
        {
            "intent_id": f"intent-{order_id}",
            "client_order_id": client_order_id,
            "order_id": order_id,
            "market": "BTC-EUR",
            "side": "BUY",
            "status": status,
            "strategy_id": "cancel-race-test",
            "strategy_dna_hash": "cancel-race-dna",
            "signal_id": "cancel-race-signal",
        },
    )


def _cancel_capability() -> LiveCapability:
    return LiveCapability(
        token="a" * 32,
        checked_at=utc_now(),
        allowed_markets=("BTC-EUR",),
        maximum_order_eur=Decimal("25"),
        maximum_total_eur=Decimal("75"),
        maximum_open_positions=3,
    )


@pytest.mark.asyncio
async def test_ambiguous_cancel_is_durable_and_duplicate_delete_is_forbidden(
    tmp_path,
) -> None:
    import asyncio

    from pydantic import SecretStr

    from execution.execution import DurableLedger

    class Session:
        delete_calls = 0

        def delete(self, *args, **kwargs):
            del args, kwargs
            self.delete_calls += 1
            raise asyncio.TimeoutError

    ledger = DurableLedger(tmp_path / "cancel-unknown.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="cancel-order",
        client_order_id="cancel-client",
    )
    session = Session()
    client = BitvavoSpotClient(
        session=session,  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    with pytest.raises(ReconciliationRequired, match="ambiguous cancellation"):
        await client.cancel_order(
            market="BTC-EUR",
            order_id="cancel-order",
            capability=_cancel_capability(),
        )
    with pytest.raises(ReconciliationRequired, match="duplicate cancel"):
        await client.cancel_order(
            market="BTC-EUR",
            order_id="cancel-order",
            capability=_cancel_capability(),
        )

    assert session.delete_calls == 1
    event_types = [event["event_type"] for event in ledger.events()]
    assert event_types[-2:] == ["CANCEL_REQUESTED", "CANCEL_STATE_UNKNOWN"]
    assert ledger.events()[-1]["payload"][
        "replacement_blocked_until_reconciled"
    ] is True


@pytest.mark.asyncio
async def test_reconcile_recovers_ambiguous_cancel_as_cancelled(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "cancel-recovered.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="cancel-recovered",
        client_order_id="cancel-recovered-client",
    )
    ledger.append(
        "CANCEL_REQUESTED",
        {
            "cancellation_id": "cancellation-recovered",
            "order_id": "cancel-recovered",
            "client_order_id": "cancel-recovered-client",
            "market": "BTC-EUR",
            "cancellation_started_at": utc_now() - timedelta(seconds=10),
        },
    )
    ledger.append(
        "CANCEL_STATE_UNKNOWN",
        {
            "cancellation_id": "cancellation-recovered",
            "order_id": "cancel-recovered",
            "market": "BTC-EUR",
        },
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    async def balances():
        return []

    async def open_orders(_market):
        return []

    async def recent_orders(_market, *, limit=1000):
        del limit
        return [
            {
                "orderId": "cancel-recovered",
                "clientOrderId": "cancel-recovered-client",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "canceled",
                "filledAmount": "0",
            }
        ]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    monkeypatch.setattr(client, "recent_orders", recent_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is True
    assert result.local_open_orders == 0
    assert result.remote_open_orders == 0
    cancelled = [
        event
        for event in ledger.events()
        if event["event_type"] == "ORDER_CANCELLED"
    ]
    assert len(cancelled) == 1
    assert cancelled[0]["payload"]["recovered"] is True


@pytest.mark.asyncio
async def test_reconcile_cancel_fill_race_records_fill_exactly_once(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "cancel-fill-race.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="cancel-filled",
        client_order_id="cancel-filled-client",
    )
    ledger.append(
        "CANCEL_REQUESTED",
        {
            "cancellation_id": "cancellation-filled",
            "order_id": "cancel-filled",
            "client_order_id": "cancel-filled-client",
            "market": "BTC-EUR",
            "cancellation_started_at": utc_now() - timedelta(seconds=10),
        },
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    filled = {
        "orderId": "cancel-filled",
        "clientOrderId": "cancel-filled-client",
        "market": "BTC-EUR",
        "side": "buy",
        "status": "filled",
        "filledAmount": "0.01",
        "filledAmountQuote": "10.01",
        "feePaid": "0.025",
        "feeCurrency": "EUR",
    }

    async def balances():
        return []

    async def open_orders(_market):
        return []

    async def recent_orders(_market, *, limit=1000):
        del limit
        return [filled]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    monkeypatch.setattr(client, "recent_orders", recent_orders)
    first = await client.reconcile(markets=("BTC-EUR",))
    second = await client.reconcile(markets=("BTC-EUR",))

    assert first.healthy is True
    assert second.healthy is True
    assert first.local_open_orders == 0
    assert len(
        [event for event in ledger.events() if event["event_type"] == "FILL"]
    ) == 1
    resolved = [
        event
        for event in ledger.events()
        if event["event_type"] == "CANCEL_RESOLVED"
    ]
    assert len(resolved) == 1
    assert resolved[0]["payload"]["terminal_order_status"] == "filled"


@pytest.mark.asyncio
async def test_reconcile_cancel_not_applied_keeps_order_open(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "cancel-not-applied.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="still-open",
        client_order_id="still-open-client",
    )
    ledger.append(
        "CANCEL_REQUESTED",
        {
            "cancellation_id": "cancellation-still-open",
            "order_id": "still-open",
            "client_order_id": "still-open-client",
            "market": "BTC-EUR",
            "cancellation_started_at": utc_now() - timedelta(seconds=10),
        },
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    open_order = {
        "orderId": "still-open",
        "clientOrderId": "still-open-client",
        "market": "BTC-EUR",
        "side": "buy",
        "status": "new",
        "filledAmount": "0",
    }

    async def balances():
        return []

    async def open_orders(_market):
        return [open_order]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is True
    assert result.local_open_orders == 1
    assert result.remote_open_orders == 1
    resolved = [
        event
        for event in ledger.events()
        if event["event_type"] == "CANCEL_RESOLVED"
    ]
    assert resolved[-1]["payload"]["resolution"] == (
        "CANCELLATION_NOT_APPLIED_ORDER_STILL_OPEN"
    )


@pytest.mark.asyncio
async def test_reconcile_cancel_partial_fill_stays_unknown_until_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "cancel-partial-open.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="partial-open",
        client_order_id="partial-open-client",
        status="partiallyFilled",
    )
    ledger.append(
        "CANCEL_REQUESTED",
        {
            "cancellation_id": "cancellation-partial-open",
            "order_id": "partial-open",
            "client_order_id": "partial-open-client",
            "market": "BTC-EUR",
            "cancellation_started_at": utc_now() - timedelta(seconds=10),
        },
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    partial = {
        "orderId": "partial-open",
        "clientOrderId": "partial-open-client",
        "market": "BTC-EUR",
        "side": "buy",
        "status": "partiallyFilled",
        "filledAmount": "0.005",
        "filledAmountQuote": "5",
    }

    async def balances():
        return []

    async def open_orders(_market):
        return [partial]

    monkeypatch.setattr(client, "balances", balances)
    monkeypatch.setattr(client, "open_orders", open_orders)
    result = await client.reconcile(markets=("BTC-EUR",))

    assert result.healthy is False
    assert result.local_open_orders == 1
    assert result.remote_open_orders == 1
    assert "UNKNOWN_CANCELLATION_STATE" in result.reason_codes
    assert "CANCELLATION_PARTIAL_FILL_STILL_OPEN" in result.reason_codes
    fills = [
        event for event in ledger.events() if event["event_type"] == "FILL"
    ]
    assert len(fills) == 1
    assert fills[0]["payload"]["quantity"] == "0.005"
    assert fills[0]["payload"]["status"] == "PARTIALLY_FILLED_PROGRESS"
    assert not any(
        event["event_type"] == "CANCEL_RESOLVED"
        for event in ledger.events()
    )


def test_incremental_fill_progress_is_cumulative_idempotent_and_fee_exact(
    tmp_path,
) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "incremental-fills.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="incremental-order",
        client_order_id="incremental-client",
        status="new",
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )

    def record(
        *,
        status: str,
        quantity: str,
        quote: str,
        fee: str,
    ) -> bool:
        return client.record_order_fill_progress(
            {
                "orderId": "incremental-order",
                "clientOrderId": "incremental-client",
                "market": "BTC-EUR",
                "side": "buy",
                "status": status,
                "filledAmount": quantity,
                "filledAmountQuote": quote,
                "feePaid": fee,
                "feeCurrency": "EUR",
            },
            fallback_market="BTC-EUR",
            fallback_side=OrderSide.BUY,
            fallback_quantity=Decimal("0.01"),
            fallback_price=Decimal("1000"),
        )

    assert record(
        status="partiallyFilled",
        quantity="0.004",
        quote="4",
        fee="0.010",
    )
    assert not record(
        status="partiallyFilled",
        quantity="0.004",
        quote="4",
        fee="0.010",
    )
    assert record(
        status="partiallyFilled",
        quantity="0.006",
        quote="6.3",
        fee="0.015",
    )
    assert record(
        status="filled",
        quantity="0.010",
        quote="10.8",
        fee="0.027",
    )

    fills = [
        event["payload"]
        for event in ledger.events()
        if event["event_type"] == "FILL"
    ]
    assert [row["quantity"] for row in fills] == ["0.004", "0.002", "0.004"]
    assert [row["quote_amount_eur"] for row in fills] == ["4", "2.3", "4.5"]
    assert [row["fee_eur"] for row in fills] == ["0.010", "0.005", "0.012"]
    assert [row["status"] for row in fills] == [
        "PARTIALLY_FILLED_PROGRESS",
        "PARTIALLY_FILLED_PROGRESS",
        "FILLED",
    ]
    assert sum(Decimal(row["quantity"]) for row in fills) == Decimal("0.010")
    assert sum(Decimal(row["quote_amount_eur"]) for row in fills) == Decimal(
        "10.8"
    )
    assert sum(Decimal(row["fee_eur"]) for row in fills) == Decimal("0.027")


def test_incremental_fill_progress_rejects_cumulative_regression(tmp_path) -> None:
    from pydantic import SecretStr

    from execution.execution import DurableLedger

    ledger = DurableLedger(tmp_path / "fill-regression.jsonl")
    _append_acknowledged_live_order(
        ledger,
        order_id="regression-order",
        client_order_id="regression-client",
        status="new",
    )
    client = BitvavoSpotClient(
        session=object(),  # type: ignore[arg-type]
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        operator_id=1,
        ledger=ledger,
    )
    base = {
        "orderId": "regression-order",
        "clientOrderId": "regression-client",
        "market": "BTC-EUR",
        "side": "buy",
        "status": "partiallyFilled",
        "feeCurrency": "EUR",
    }
    client.record_order_fill_progress(
        {
            **base,
            "filledAmount": "0.006",
            "filledAmountQuote": "6",
            "feePaid": "0.015",
        },
        fallback_market="BTC-EUR",
        fallback_side=OrderSide.BUY,
        fallback_quantity=Decimal("0.01"),
        fallback_price=Decimal("1000"),
    )

    with pytest.raises(ReconciliationRequired, match="regressed"):
        client.record_order_fill_progress(
            {
                **base,
                "filledAmount": "0.005",
                "filledAmountQuote": "5",
                "feePaid": "0.0125",
            },
            fallback_market="BTC-EUR",
            fallback_side=OrderSide.BUY,
            fallback_quantity=Decimal("0.01"),
            fallback_price=Decimal("1000"),
        )
