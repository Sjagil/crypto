"""Persistent, restart-safe paper lifecycle for frozen actionable strategies."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.autonomous_trading import (
    PRIMARY_STRATEGY_DNA,
    PRIMARY_STRATEGY_ID,
    build_autonomous_control_plane,
    decide_managed_position_action,
)
from core.contracts import OrderIntent, OrderSide, OrderStatus, OrderType
from execution.execution import ExecutionMarketRules, PaperBroker
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso, utc_now


def _paths(settings: Settings) -> tuple[Path, Path]:
    directory = settings.paths.output_dir / "paper"
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "paper_execution.jsonl"
    ledger.touch(exist_ok=True)
    return directory / "paper_state.json", ledger


def _broker(settings: Settings) -> PaperBroker:
    _, ledger = _paths(settings)
    return PaperBroker(
        initial_balances={
            "EUR": Decimal(str(settings.paper_automation.initial_capital_eur))
        },
        market_rules={
            "ETH-EUR": ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
            )
        },
        fee_fraction=Decimal(str(settings.costs.default_fee)),
        slippage_bps=Decimal(str(settings.costs.slippage_bps)),
        spread_bps=Decimal(str(settings.costs.spread_bps)),
        ledger_path=ledger,
    )


def _state(settings: Settings) -> dict[str, Any]:
    state_path, _ = _paths(settings)
    if state_path.is_file():
        return dict(read_json(state_path))
    return {
        "schema_version": "paper_lifecycle_v1",
        "autotrade_enabled": settings.paper_automation.autotrade_enabled,
        "status": "READY",
        "positions": {},
        "real_orders_placed": 0,
        "paper_orders_placed": 0,
        "real_exchange_requests": 0,
        "last_cycle_at": None,
    }


def _save(settings: Settings, state: dict[str, Any]) -> None:
    state_path, _ = _paths(settings)
    atomic_write_json(state_path, state)


def activate_paper_auto(settings: Settings) -> dict[str, Any]:
    state = _state(settings)
    state.update(
        {
            "autotrade_enabled": True,
            "status": "READY",
            "activated_at": utc_iso(),
            "real_orders_placed": 0,
            "real_exchange_requests": 0,
        }
    )
    _save(settings, state)
    return paper_status(settings)


def _new_orders_today(broker: PaperBroker) -> int:
    today = utc_now().date().isoformat()
    return sum(
        1
        for event in broker.ledger.events()
        if event.get("event_type") == "ORDER_INTENT"
        and str(event.get("recorded_at") or "")[:10] == today
        and str((event.get("payload") or {}).get("side")) == "BUY"
    )


def paper_status(settings: Settings) -> dict[str, Any]:
    state = _state(settings)
    broker = _broker(settings)
    fills = broker.fills
    realized = Decimal("0")
    buy_cost: dict[str, Decimal] = {}
    for fill in fills:
        if fill.side is OrderSide.BUY:
            buy_cost[fill.market] = buy_cost.get(fill.market, Decimal("0")) + (
                fill.quantity * fill.price + fill.fee_eur
            )
        else:
            realized += fill.quantity * fill.price - fill.fee_eur
            realized -= buy_cost.get(fill.market, Decimal("0"))
            buy_cost[fill.market] = Decimal("0")
    return {
        **state,
        "balances": broker.balance_snapshot(),
        "open_positions": len(state.get("positions") or {}),
        "paper_orders": len(broker.orders),
        "paper_fills": len(fills),
        "new_buy_orders_today": _new_orders_today(broker),
        "realized_pnl_eur": str(realized),
        "reconciliation": broker.reconcile(),
        "real_orders_placed": 0,
        "real_exchange_requests": 0,
    }


def _submit(
    settings: Settings,
    broker: PaperBroker,
    *,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    reason: str,
    identity: str,
) -> Any:
    intent = OrderIntent(
        intent_id=f"paper-rr-{stable_hash({'identity': identity, 'side': side.value}, length=16)}",
        idempotency_key=stable_hash(
            {
                "paper": PRIMARY_STRATEGY_DNA,
                "identity": identity,
                "side": side.value,
            },
            length=32,
        ),
        market="ETH-EUR",
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_id=PRIMARY_STRATEGY_ID,
        maximum_notional_eur=(
            Decimal(str(settings.paper_automation.initial_capital_eur))
            if side is OrderSide.BUY
            else None
        ),
        reason_codes=(reason,),
    )
    return broker.submit(intent, market_price=price)


def run_paper_once(
    settings: Settings,
    *,
    control_plane: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate and manage the frozen RR strategy without private API calls."""

    state = _state(settings)
    if not state.get("autotrade_enabled", settings.paper_automation.autotrade_enabled):
        return {
            **paper_status(settings),
            "cycle_status": "DISABLED",
            "reason_code": "PAPER_AUTOTRADE_DISABLED",
            "orders_generated_this_cycle": 0,
        }
    control = control_plane or build_autonomous_control_plane(settings)
    opportunity = dict(control["live"].get("natural_signal") or {})
    price = Decimal(str(opportunity.get("entry_price") or "0"))
    if price <= 0:
        state.update(
            {
                "status": "DATA_BLOCKED",
                "last_cycle_at": utc_iso(),
                "last_reason": "PAPER_MARKET_PRICE_MISSING",
            }
        )
        _save(settings, state)
        return {
            **paper_status(settings),
            "cycle_status": "NO_TRADE",
            "reason_code": "PAPER_MARKET_PRICE_MISSING",
            "orders_generated_this_cycle": 0,
        }
    broker = _broker(settings)
    positions = dict(state.get("positions") or {})
    position = dict(positions.get(PRIMARY_STRATEGY_DNA) or {})
    base_balance = broker.balances.get("ETH", Decimal("0"))
    equity = broker.balances.get("EUR", Decimal("0")) + base_balance * price
    today = utc_now().date().isoformat()
    day_start_equity = Decimal(
        str(
            state.get("day_start_equity_eur")
            if state.get("risk_date") == today
            else equity
        )
    )
    peak_equity = max(
        equity,
        Decimal(str(state.get("peak_equity_eur") or equity)),
    )
    daily_loss_pct = (
        max(Decimal("0"), day_start_equity - equity)
        / day_start_equity
        * Decimal("100")
        if day_start_equity > 0
        else Decimal("0")
    )
    drawdown_pct = (
        max(Decimal("0"), peak_equity - equity)
        / peak_equity
        * Decimal("100")
        if peak_equity > 0
        else Decimal("0")
    )
    state.update(
        {
            "risk_date": today,
            "day_start_equity_eur": str(day_start_equity),
            "peak_equity_eur": str(peak_equity),
            "current_equity_eur": str(equity),
            "daily_loss_pct": str(daily_loss_pct),
            "drawdown_pct": str(drawdown_pct),
        }
    )
    risk_exit = bool(
        daily_loss_pct
        >= Decimal(str(settings.paper_automation.max_daily_loss_pct))
        or drawdown_pct
        >= Decimal(str(settings.paper_automation.max_drawdown_pct))
    )
    order = None
    reason = "NO_NATURAL_NEW_ENTRY"
    if position:
        if base_balance <= 0:
            state.update(
                {
                    "status": "RECONCILIATION_BLOCKED",
                    "last_cycle_at": utc_iso(),
                    "last_reason": "PAPER_POSITION_BALANCE_MISMATCH",
                }
            )
            _save(settings, state)
            return {
                **paper_status(settings),
                "cycle_status": "NO_TRADE",
                "reason_code": "PAPER_POSITION_BALANCE_MISMATCH",
                "orders_generated_this_cycle": 0,
            }
        decision = decide_managed_position_action(
            position,
            market_price=float(price),
            strategy_action=(
                "EXIT"
                if risk_exit
                else str(opportunity.get("action") or "NO_SIGNAL")
            ),
            owned_quantity=base_balance,
        )
        reason = decision.reason_code
        if decision.action == "UPDATE_ONLY":
            if decision.updated_stop_loss is not None:
                position["stop_loss"] = decision.updated_stop_loss
            if decision.tp1_reached is not None:
                position["tp1_reached"] = decision.tp1_reached
            positions[PRIMARY_STRATEGY_DNA] = position
        elif decision.action == "SELL_FULL":
            order = _submit(
                settings,
                broker,
                side=OrderSide.SELL,
                quantity=base_balance,
                price=price,
                reason=decision.reason_code,
                identity=f"{position['entry_opportunity_id']}:{decision.reason_code}",
            )
            if order.status is OrderStatus.FILLED:
                positions.pop(PRIMARY_STRATEGY_DNA, None)
                state["last_closed_position"] = {
                    **position,
                    "exit_price": str(order.average_fill_price),
                    "exit_reason": decision.reason_code,
                    "closed_at": utc_iso(),
                }
    elif (
        opportunity.get("actionable") is True
        and opportunity.get("action") == "BUY"
        and not opportunity.get("blockers")
    ):
        if risk_exit:
            reason = "PAPER_RISK_LIMIT_REACHED"
        elif len(positions) >= settings.paper_automation.max_open_positions:
            reason = "PAPER_MAX_OPEN_POSITIONS"
        elif _new_orders_today(broker) >= settings.paper_automation.max_new_orders_per_day:
            reason = "PAPER_MAX_NEW_ORDERS_PER_DAY"
        else:
            entry = price
            stop = Decimal(str(opportunity["stop_loss"]))
            risk_budget = Decimal(
                str(
                    settings.paper_automation.initial_capital_eur
                    * settings.paper_automation.max_risk_per_trade_pct
                    / 100.0
                )
            )
            risk_distance = entry - stop
            if risk_distance <= 0:
                reason = "PAPER_INVALID_STOP"
            else:
                risk_quantity = risk_budget / risk_distance
                maximum_notional = Decimal(
                    str(
                        settings.paper_automation.initial_capital_eur
                        * settings.paper_automation.max_total_exposure_pct
                        / 100.0
                    )
                )
                affordable = broker.balances.get("EUR", Decimal("0")) / (
                    entry * (Decimal("1") + Decimal(str(settings.costs.default_fee)))
                )
                quantity = min(risk_quantity, maximum_notional / entry, affordable)
                order = _submit(
                    settings,
                    broker,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    price=entry,
                    reason="NATURAL_RR_ENTRY",
                    identity=str(opportunity["opportunity_id"]),
                )
                reason = (
                    "PAPER_RR_ENTRY_FILLED"
                    if order.status is OrderStatus.FILLED
                    else str(order.rejection_code or order.status.value)
                )
                if order.status is OrderStatus.FILLED:
                    positions[PRIMARY_STRATEGY_DNA] = {
                        "strategy_id": PRIMARY_STRATEGY_ID,
                        "strategy_dna": PRIMARY_STRATEGY_DNA,
                        "market": "ETH-EUR",
                        "timeframe": "1d",
                        "entry_opportunity_id": opportunity["opportunity_id"],
                        "entry_price": str(order.average_fill_price),
                        "quantity": str(order.filled_quantity),
                        "stop_loss": opportunity["stop_loss"],
                        "take_profit_1": opportunity["take_profit_1"],
                        "take_profit_2": opportunity["take_profit_2"],
                        "tp1_reached": False,
                        "opened_at": utc_iso(),
                    }
    state.update(
        {
            "schema_version": "paper_lifecycle_v1",
            "status": "ACTIVE" if positions else "READY",
            "positions": positions,
            "paper_orders_placed": len(broker.orders),
            "real_orders_placed": 0,
            "real_exchange_requests": 0,
            "last_cycle_at": utc_iso(),
            "last_signal": {
                key: opportunity.get(key)
                for key in (
                    "opportunity_id",
                    "action",
                    "actionable",
                    "entry_price",
                    "stop_loss",
                    "take_profit_1",
                    "take_profit_2",
                    "blockers",
                )
            },
            "last_reason": reason,
        }
    )
    _save(settings, state)
    return {
        **paper_status(settings),
        "cycle_status": (
            "ORDER_FILLED"
            if order is not None and order.status is OrderStatus.FILLED
            else "NO_TRADE"
        ),
        "reason_code": reason,
        "orders_generated_this_cycle": int(order is not None),
        "real_orders_placed": 0,
        "real_exchange_requests": 0,
    }


__all__ = [
    "activate_paper_auto",
    "paper_status",
    "run_paper_once",
]
