"""Restart-safe paper lifecycle for event-driven playbook opportunities."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from config.settings import Settings
from core.contracts import OrderIntent, OrderSide, OrderStatus, OrderType
from execution.execution import ExecutionMarketRules, PaperBroker
from reporting.canonical_economics import (
    ECONOMIC_SCHEMA_VERSION,
    canonical_family,
)
from utils.common import atomic_write_json, read_json, stable_hash, utc_now

MAXIMUM_PAPER_ORDER_EUR = Decimal("25")
MAXIMUM_PAPER_POSITIONS = 5


def load_canonical_entry_economics_gate(settings: Settings) -> dict[str, Any]:
    """Load the immutable canonical-economics decision for paper entries.

    No evidence means paper discovery may continue.  Once a latest pointer
    exists, malformed, out-of-scope, or hash-mismatched evidence fails closed
    for *new* entries.  Position management never consumes this gate.
    """

    root = (settings.paths.output_dir / "economics").resolve()
    latest_path = root / "latest.json"
    base = {
        "status": "NOT_AVAILABLE",
        "new_entries_allowed": True,
        "block_all_new_entries": False,
        "paused_families": [],
        "live_entry_families": [],
        "live_entry_strategy_dna_hashes": [],
        "live_new_entries_allowed": False,
        "artifact_path": None,
        "artifact_hash": None,
    }
    if not latest_path.is_file():
        return base
    try:
        latest = read_json(latest_path)
        if not isinstance(latest, Mapping):
            raise ValueError("latest economics pointer is not an object")
        artifact_path = Path(str(latest.get("artifact_path") or "")).resolve()
        if not artifact_path.is_relative_to(root / "runs"):
            raise ValueError("economics artifact is outside the immutable run root")
        artifact = read_json(artifact_path)
        if not isinstance(artifact, Mapping):
            raise ValueError("economics artifact is not an object")
        pointer_hash = str(latest.get("artifact_hash") or "")
        artifact_hash = str(artifact.get("artifact_hash") or "")
        computed_hash = stable_hash(
            {
                key: value
                for key, value in artifact.items()
                if key not in {"artifact_hash", "created_at"}
            },
            length=64,
        )
        if not pointer_hash or pointer_hash != artifact_hash:
            raise ValueError("economics latest pointer hash mismatch")
        if artifact_hash != computed_hash:
            raise ValueError("economics artifact content hash mismatch")
        if artifact.get("schema_version") != ECONOMIC_SCHEMA_VERSION:
            raise ValueError("unsupported canonical economics schema")
        paused = sorted(
            {
                str(row.get("strategy_family") or "").upper()
                for row in artifact.get("promotion_recommendations") or []
                if isinstance(row, Mapping)
                and (
                    row.get("promotion_status")
                    == "BLOCKED_NEGATIVE_EXPECTANCY"
                    or row.get("recommendation") == "PAUSE_PAPER_GENERATION"
                )
                and row.get("strategy_family")
            }
        )
        live_families = sorted(
            {
                str(row.get("strategy_family") or "").upper()
                for row in artifact.get("promotion_recommendations") or []
                if isinstance(row, Mapping)
                and row.get("live_validated") is True
                and row.get("strategy_family")
            }
        )
        live_dna = sorted(
            {
                str(value).lower()
                for value in artifact.get(
                    "live_validated_strategy_dna_hashes"
                )
                or []
                if value
            }
        )
        return {
            **base,
            "status": "READY",
            "artifact_path": str(artifact_path),
            "artifact_hash": artifact_hash,
            "paused_families": paused,
            "live_entry_families": live_families,
            "live_entry_strategy_dna_hashes": live_dna,
            "live_new_entries_allowed": bool(live_families or live_dna),
        }
    except (OSError, TypeError, ValueError):
        return {
            **base,
            "status": "INVALID_EVIDENCE_FAIL_CLOSED",
            "new_entries_allowed": False,
            "block_all_new_entries": True,
        }


def _paths(settings: Settings) -> tuple[Path, Path]:
    directory = settings.paths.output_dir / "paper"
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory / "event_driven_playbook_state.json",
        directory / "event_driven_playbook_execution.jsonl",
    )


def _state(settings: Settings) -> dict[str, Any]:
    path, _ = _paths(settings)
    if path.is_file():
        return dict(read_json(path))
    return {
        "schema_version": "event_driven_playbook_paper_state_v1",
        "status": "READY",
        "positions": {},
        "completed_opportunities": [],
        "paper_orders": 0,
        "paper_fills": 0,
        "real_orders": 0,
        "private_exchange_requests": 0,
    }


def _broker(
    settings: Settings,
    markets: Iterable[str],
) -> PaperBroker:
    _, ledger = _paths(settings)
    return PaperBroker(
        initial_balances={
            "EUR": Decimal(str(settings.paper_automation.initial_capital_eur))
        },
        market_rules={
            market: ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
            )
            for market in markets
        },
        fee_fraction=Decimal(str(settings.costs.default_fee)),
        slippage_bps=Decimal(str(settings.costs.slippage_bps)),
        spread_bps=Decimal(str(settings.costs.spread_bps)),
        ledger_path=ledger,
    )


def _intent(
    *,
    opportunity: Mapping[str, Any],
    side: OrderSide,
    quantity: Decimal,
    reason: str,
) -> OrderIntent:
    identity = stable_hash(
        {
            "opportunity_id": opportunity["opportunity_id"],
            "state": opportunity.get("state"),
            "side": side.value,
            "reason": reason,
        },
        length=40,
    )
    return OrderIntent(
        intent_id=identity[:32],
        idempotency_key=f"event-paper:{identity}",
        market=str(opportunity["market"]),
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        strategy_id=str(opportunity["playbook_id"]),
        strategy_dna_hash=str(opportunity["playbook_dna"]),
        signal_id=str(opportunity["opportunity_id"]),
        portfolio_decision_id=identity,
        maximum_notional_eur=(
            MAXIMUM_PAPER_ORDER_EUR if side is OrderSide.BUY else None
        ),
        reason_codes=(reason, "PAPER_ONLY"),
    )


def run_event_driven_paper_once(
    settings: Settings,
    *,
    opportunities: Iterable[Mapping[str, Any]],
    realtime_snapshot: Mapping[str, Any],
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Fill natural paper entries and manage their complete exit lifecycle."""

    now = (observed_at or utc_now()).astimezone(UTC)
    rows = [dict(row) for row in opportunities]
    prices = {
        str(row.get("market")): Decimal(str(row["price"]))
        for row in realtime_snapshot.get("markets") or []
        if row.get("market") and row.get("price") is not None
    }
    markets = tuple(sorted({*prices, *(str(row["market"]) for row in rows)}))
    state = _state(settings)
    broker = _broker(settings, markets)
    positions = {
        str(key): dict(value)
        for key, value in (state.get("positions") or {}).items()
    }
    completed = set(state.get("completed_opportunities") or [])
    events: list[dict[str, Any]] = []

    for key, position in list(positions.items()):
        market = str(position["market"])
        price = prices.get(market)
        if price is None or price <= 0:
            continue
        entry = Decimal(str(position["entry_price"]))
        stop = Decimal(str(position["stop_loss"]))
        tp1 = Decimal(str(position["take_profit_1"]))
        tp2 = Decimal(str(position["take_profit_2"]))
        quantity = Decimal(str(position["quantity"]))
        matching = max(
            (
                row
                for row in rows
                if row.get("market") == market
                and row.get("playbook_dna") == position.get("playbook_dna")
            ),
            key=lambda row: float(row.get("score") or 0),
            default={},
        )
        buy_ratio = float(
            (matching.get("realtime_inputs") or {}).get("taker_buy_ratio_1m")
            or 0.5
        )
        ofi = float(
            (matching.get("realtime_inputs") or {}).get("ofi_1m") or 0.0
        )
        opened_at = datetime.fromisoformat(
            str(position["opened_at"]).replace("Z", "+00:00")
        ).astimezone(UTC)
        time_review_due = (now - opened_at).total_seconds() >= (
            int(position.get("time_stop_minutes") or 30) * 60
        )
        flow_exhausted = buy_ratio < 0.45 and ofi < 0
        structure_no_longer_supported = (
            not matching
            or bool(matching.get("hard_blockers"))
            or str(matching.get("state") or "").upper()
            in {"INVALIDATED", "EXPIRED"}
        )
        time_exit_confirmed = (
            time_review_due
            and not position.get("tp1_reached")
            and price <= entry
            and (structure_no_longer_supported or flow_exhausted)
        )
        reason: str | None = None
        exit_quantity = quantity
        if price <= stop:
            reason = "HARD_STOP"
        elif price >= tp2:
            reason = "TAKE_PROFIT_2"
        elif time_exit_confirmed:
            reason = "TIME_AND_STRUCTURE_EXIT"
        elif flow_exhausted:
            reason = "ORDERFLOW_EXHAUSTION"
        elif price >= tp1 and not position.get("tp1_reached"):
            reason = "TAKE_PROFIT_1"
            exit_quantity = quantity / Decimal("2")
        if reason is None:
            continue
        order = broker.submit(
            _intent(
                opportunity={
                    **position,
                    "opportunity_id": position["opportunity_id"],
                    "playbook_id": position["playbook_id"],
                    "playbook_dna": position["playbook_dna"],
                    "state": "EXITING",
                },
                side=OrderSide.SELL,
                quantity=exit_quantity,
                reason=reason,
            ),
            market_price=price,
        )
        if order.status is not OrderStatus.FILLED:
            events.append(
                {
                    "event": "PAPER_EXIT_REJECTED",
                    "opportunity_id": position["opportunity_id"],
                    "reason": order.rejection_code,
                }
            )
            continue
        if reason == "TAKE_PROFIT_1":
            position["quantity"] = str(quantity - order.filled_quantity)
            position["stop_loss"] = str(entry)
            position["tp1_reached"] = True
            positions[key] = position
            lifecycle_state = "MANAGING"
        else:
            positions.pop(key, None)
            completed.add(str(position["opportunity_id"]))
            lifecycle_state = "CLOSED"
        events.append(
            {
                "event": "PAPER_POSITION_UPDATE",
                "opportunity_id": position["opportunity_id"],
                "market": market,
                "state": lifecycle_state,
                "reason": reason,
                "quantity": str(order.filled_quantity),
                "price": str(order.average_fill_price),
            }
        )

    occupied_markets = {str(row["market"]) for row in positions.values()}
    entry_candidates = [
        row
        for row in sorted(
            rows,
            key=lambda item: float(item.get("score") or 0),
            reverse=True,
        )
        if row.get("state") == "ENTRY_READY"
        and not row.get("hard_blockers")
        and str(row.get("market")) not in occupied_markets
        and str(row.get("opportunity_id")) not in positions
        and str(row.get("opportunity_id")) not in completed
    ]
    economics_gate = load_canonical_entry_economics_gate(settings)
    paused_families = set(economics_gate["paused_families"])
    blocked_candidates = [
        row
        for row in entry_candidates
        if economics_gate["block_all_new_entries"]
        or canonical_family(
            str(row.get("playbook_id") or ""),
            str(row.get("strategy_family") or ""),
        )[0]
        in paused_families
    ]
    blocked_ids = {
        str(row.get("opportunity_id") or "") for row in blocked_candidates
    }
    eligible_candidates = [
        row
        for row in entry_candidates
        if str(row.get("opportunity_id") or "") not in blocked_ids
    ]
    entry = eligible_candidates[0] if eligible_candidates else None
    economics_gate.update(
        {
            "entry_candidate_count": len(entry_candidates),
            "blocked_entry_candidate_count": len(blocked_candidates),
            "eligible_entry_candidate_count": len(eligible_candidates),
            "blocked_candidate_families": sorted(
                {
                    canonical_family(
                        str(row.get("playbook_id") or ""),
                        str(row.get("strategy_family") or ""),
                    )[0]
                    for row in blocked_candidates
                }
            ),
        }
    )
    if blocked_candidates:
        events.append(
            {
                "event": "PAPER_ENTRY_COHORT_GATED",
                "status": economics_gate["status"],
                "artifact_hash": economics_gate["artifact_hash"],
                "blocked_entry_candidate_count": len(blocked_candidates),
                "blocked_candidate_families": economics_gate[
                    "blocked_candidate_families"
                ],
                "position_management_affected": False,
            }
        )
    if entry is not None and len(positions) < MAXIMUM_PAPER_POSITIONS:
        market = str(entry["market"])
        price = prices.get(market)
        stop = Decimal(str(entry["stop_loss"]))
        if price is not None and price > stop > 0:
            capital = Decimal(str(settings.paper_automation.initial_capital_eur))
            playbook_risk_multiplier = max(
                Decimal("0"),
                min(
                    Decimal("1"),
                    Decimal(
                        str(entry.get("playbook_risk_multiplier") or "1")
                    ),
                ),
            )
            risk_budget = (
                capital * Decimal("0.005") * playbook_risk_multiplier
            )
            risk_quantity = risk_budget / (price - stop)
            quantity = min(
                risk_quantity,
                (
                    MAXIMUM_PAPER_ORDER_EUR
                    * playbook_risk_multiplier
                    / price
                ),
            )
            identity = str(entry["opportunity_id"])
            events.append(
                {
                    "event": "PAPER_ORDER_INTENT_CREATED",
                    "opportunity_id": identity,
                    "market": market,
                    "state": "ORDER_INTENT_CREATED",
                    "maximum_notional_eur": str(MAXIMUM_PAPER_ORDER_EUR),
                }
            )
            order = broker.submit(
                _intent(
                    opportunity=entry,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    reason="EVENT_DRIVEN_ENTRY_READY",
                ),
                market_price=price,
            )
            if order.status is OrderStatus.FILLED:
                positions[identity] = {
                    **entry,
                    "state": "MANAGING",
                    "entry_price": str(order.average_fill_price),
                    "quantity": str(order.filled_quantity),
                    "opened_at": now.isoformat(),
                    "tp1_reached": False,
                    "paper_only": True,
                }
                events.append(
                    {
                        "event": "PAPER_POSITION_OPENED",
                        "opportunity_id": identity,
                        "market": market,
                        "state": "FILLED",
                        "quantity": str(order.filled_quantity),
                        "price": str(order.average_fill_price),
                    }
                )

    state.update(
        {
            "status": "ACTIVE" if positions else "READY",
            "positions": positions,
            "completed_opportunities": sorted(completed),
            "last_cycle_at": now.isoformat(),
            "paper_orders": len(broker.orders),
            "paper_fills": len(broker.fills),
            "balances": broker.balance_snapshot(),
            "reconciliation": asdict(broker.reconcile()),
            "events": events,
            "entry_economics_gate": economics_gate,
            "real_orders": 0,
            "private_exchange_requests": 0,
        }
    )
    state_path, _ = _paths(settings)
    atomic_write_json(state_path, state)
    return state


__all__ = [
    "load_canonical_entry_economics_gate",
    "run_event_driven_paper_once",
]
