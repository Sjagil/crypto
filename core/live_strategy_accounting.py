"""Reproducible per-strategy accounting for canonical live Bitvavo fills.

The exchange owns the real balances.  This module rebuilds virtual strategy
lots from the append-only execution ledger so several approved strategy DNA
instances can share one venue balance without sharing performance attribution.
It never calls an exchange and has no order-submission capability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.strategy_degradation import evaluate_strategy_degradation
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    stable_hash,
    utc_iso,
)

ZERO = Decimal("0")
NON_STRATEGY_EXECUTION_IDS = frozenset(
    {
        "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE",
        "EXECUTION_SMOKE_NOT_STRATEGY_TRADE",
    }
)


def _decimal(value: Any, *, default: Decimal = ZERO) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default
    return result if result.is_finite() else default


def _holding_seconds(opened_at: str | None, closed_at: str | None) -> int | None:
    if not opened_at or not closed_at:
        return None
    try:
        opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        closed = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=UTC)
    if closed.tzinfo is None:
        closed = closed.replace(tzinfo=UTC)
    return max(0, int((closed.astimezone(UTC) - opened.astimezone(UTC)).total_seconds()))


def _load_approvals(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "config" / "live_strategy_approvals.yaml"
    payload = (
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if path.is_file()
        else {}
    )
    strategies = payload.get("strategies") or {}
    approvals = {
        str(strategy_id): dict(values)
        for strategy_id, values in strategies.items()
        if isinstance(values, Mapping)
    }
    playbook_path = root / "config" / "live_playbook_authority.json"
    if playbook_path.is_file():
        playbook_authority = dict(read_json(playbook_path))
        if (
            playbook_authority.get("active") is True
            and playbook_authority.get("schema_version")
            == "event_driven_playbook_authority_v1"
        ):
            for row in playbook_authority.get("approved_playbooks") or []:
                if not isinstance(row, Mapping) or row.get("active") is not True:
                    continue
                strategy_id = str(row.get("playbook_id") or "")
                dna = str(row.get("playbook_dna") or "")
                if not strategy_id or len(dna) != 64:
                    continue
                approvals.setdefault(
                    strategy_id,
                    {
                        "strategy_dna_hash": dna,
                        "accepted_historical_dna_hashes": list(
                            dict.fromkeys(
                                value
                                for value in (
                                    dna,
                                    str(row.get("previous_playbook_dna") or ""),
                                )
                                if len(value) == 64
                            )
                        ),
                        "strategy_family": row.get("family"),
                        "timeframe": "/".join(
                            str(value)
                            for value in row.get("execution_timeframes") or []
                        ),
                        "approved_markets": list(row.get("markets") or []),
                        "approved_for_live": True,
                        "maximum_order_eur": row.get("maximum_order_eur"),
                        "maximum_total_exposure_eur": row.get(
                            "maximum_total_exposure_eur"
                        ),
                        "autoscale": False,
                    },
                )
    portfolio_path = (
        root
        / "output"
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    if portfolio_path.is_file():
        portfolio = dict(read_json(portfolio_path))
        if (
            portfolio.get("active") is True
            and portfolio.get("schema_version")
            == "positive_strategy_live_authority_v1"
        ):
            for row in portfolio.get("approved_candidates") or []:
                if not isinstance(row, Mapping):
                    continue
                strategy_id = str(row.get("strategy_id") or "")
                dna = str(row.get("strategy_dna_hash") or "")
                if not strategy_id or len(dna) != 64:
                    continue
                approvals.setdefault(
                    strategy_id,
                    {
                        "strategy_dna_hash": dna,
                        "strategy_family": "EXACT_POSITIVE_GENERATED",
                        "timeframe": row.get("timeframe"),
                        "approved_markets": list(
                            row.get("approved_markets") or []
                        ),
                        "approved_for_live": True,
                        "maximum_order_eur": portfolio.get(
                            "maximum_order_eur"
                        ),
                        "maximum_total_exposure_eur": portfolio.get(
                            "maximum_order_eur"
                        ),
                        "allocation_mode": "SHARED_PORTFOLIO_CAP",
                        "shared_portfolio_cap_eur": portfolio.get(
                            "maximum_total_exposure_eur"
                        ),
                        "autoscale": False,
                    },
                )
    return approvals


def _market_prices(root: Path) -> dict[str, Decimal]:
    health_path = root / "output" / "operations" / "live_account_health.json"
    if not health_path.is_file():
        return {}
    health = dict(read_json(health_path))
    holdings = (
        health.get("account", {})
        .get("portfolio_valuation", {})
        .get("holdings", [])
    )
    return {
        str(row.get("market") or ""): _decimal(row.get("price_eur"))
        for row in holdings
        if isinstance(row, Mapping) and row.get("market")
    }


def _intent_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "ORDER_INTENT":
            continue
        payload = dict(event.get("payload") or {})
        for field in ("intent_id", "client_order_id", "idempotency_key"):
            identity = str(payload.get(field) or "")
            if identity:
                index[identity] = payload
    return index


def _fill_attribution(
    payload: Mapping[str, Any],
    intents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    attribution = {
        "strategy_id": payload.get("strategy_id"),
        "strategy_dna_hash": payload.get("strategy_dna_hash"),
        "signal_id": payload.get("signal_id"),
        "portfolio_decision_id": payload.get("portfolio_decision_id"),
    }
    if attribution["strategy_id"]:
        return attribution
    for field in ("intent_id", "client_order_id", "idempotency_key"):
        identity = str(payload.get(field) or "")
        if not identity or identity not in intents:
            continue
        intent = intents[identity]
        for key in attribution:
            attribution[key] = attribution[key] or intent.get(key)
        break
    return attribution


def _empty_account(
    strategy_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "strategy_dna": str(metadata.get("strategy_dna_hash") or ""),
        "accepted_historical_dna_hashes": set(
            str(value)
            for value in metadata.get("accepted_historical_dna_hashes") or []
            if value
        ),
        "strategy_family": str(metadata.get("strategy_family") or "UNKNOWN"),
        "timeframe": str(metadata.get("timeframe") or "UNKNOWN"),
        "approved_markets": list(metadata.get("approved_markets") or []),
        "authority_level": "LIVE_CANARY",
        "operator_approved": bool(metadata.get("approved_for_live")),
        "maximum_order_eur": _decimal(metadata.get("maximum_order_eur")),
        "maximum_total_exposure_eur": _decimal(
            metadata.get("maximum_total_exposure_eur")
        ),
        "autoscale": bool(metadata.get("autoscale", False)),
        "allocation_mode": str(
            metadata.get("allocation_mode") or "DEDICATED_STRATEGY_CAP"
        ),
        "shared_portfolio_cap_eur": _decimal(
            metadata.get("shared_portfolio_cap_eur")
        ),
        "quantity_by_market": defaultdict(lambda: ZERO),
        "cost_basis_by_market": defaultdict(lambda: ZERO),
        "entry_quantity_by_market": defaultdict(lambda: ZERO),
        "entry_notional_by_market": defaultdict(lambda: ZERO),
        "entry_fee_by_market": defaultdict(lambda: ZERO),
        "exit_fee_by_market": defaultdict(lambda: ZERO),
        "realised_pnl_by_market": defaultdict(lambda: ZERO),
        "opened_at_by_market": {},
        "slippage_bps_by_market": defaultdict(list),
        "last_price_by_market": defaultdict(lambda: ZERO),
        "open_trade_count": 0,
        "closed_trade_count": 0,
        "realized_round_trips": [],
        "closed_trades": [],
        "realised_pnl": ZERO,
        "fees_paid": ZERO,
        "gross_profit": ZERO,
        "gross_loss": ZERO,
        "filled_buy_count": 0,
        "filled_sell_count": 0,
        "fill_count": 0,
        "first_fill_at": None,
        "last_fill_at": None,
        "signal_ids": set(),
        "decision_ids": set(),
        "client_order_ids": set(),
    }


def _serialize_account(
    account: dict[str, Any],
    prices: Mapping[str, Decimal],
) -> dict[str, Any]:
    open_value = ZERO
    unrealised = ZERO
    open_markets: list[dict[str, Any]] = []
    for market, quantity in sorted(account["quantity_by_market"].items()):
        if quantity <= ZERO:
            continue
        price = prices.get(market) or account["last_price_by_market"][market]
        basis = account["cost_basis_by_market"][market]
        value = quantity * price
        market_unrealised = value - basis
        open_value += value
        unrealised += market_unrealised
        open_markets.append(
            {
                "market": market,
                "quantity": str(quantity),
                "cost_basis_eur": str(basis),
                "mark_price_eur": str(price),
                "market_value_eur": str(value),
                "unrealised_pnl_eur": str(market_unrealised),
            }
        )
    round_trips = list(account["realized_round_trips"])
    realised = account["realised_pnl"]
    gross_profit = account["gross_profit"]
    gross_loss = account["gross_loss"]
    profit_factor = (
        float(gross_profit / gross_loss)
        if gross_loss > ZERO
        else None
    )
    expectancy = (
        float(realised / len(round_trips)) if round_trips else None
    )
    wins = sum(value > ZERO for value in round_trips)
    losses = sum(value < ZERO for value in round_trips)
    cumulative = ZERO
    peak = ZERO
    maximum_drawdown = ZERO
    for value in round_trips:
        cumulative += value
        peak = max(peak, cumulative)
        maximum_drawdown = max(maximum_drawdown, peak - cumulative)
    allocated = account["maximum_total_exposure_eur"]
    used = open_value
    lifecycle = (
        "SCALING"
        if str(account["authority_level"]).upper() not in {"LIVE_CANARY", "LEVEL_3"}
        else "ACTIVE"
        if round_trips
        else "VALIDATING"
    )
    return {
        "strategy_id": account["strategy_id"],
        "strategy_dna": account["strategy_dna"],
        "strategy_family": account["strategy_family"],
        "timeframe": account["timeframe"],
        "approved_markets": account["approved_markets"],
        "authority_level": account["authority_level"],
        "lifecycle_state": lifecycle,
        "operator_approved": account["operator_approved"],
        "autoscale": account["autoscale"],
        "allocation_mode": account["allocation_mode"],
        "shared_portfolio_cap_eur": str(
            account["shared_portfolio_cap_eur"]
        ),
        "allocated_capital_eur": str(allocated),
        "used_capital_eur": str(used),
        "available_capital_eur": str(max(ZERO, allocated - used)),
        "open_position_value_eur": str(open_value),
        "open_trade_count": len(open_markets),
        "closed_trade_count": account["closed_trade_count"],
        "fill_count": account["fill_count"],
        "filled_buy_count": account["filled_buy_count"],
        "filled_sell_count": account["filled_sell_count"],
        "gross_profit_eur": str(gross_profit),
        "gross_loss_eur": str(gross_loss),
        "realised_pnl_eur": str(realised),
        "unrealised_pnl_eur": str(unrealised),
        "net_pnl_eur": str(realised + unrealised),
        "fees_paid_eur": str(account["fees_paid"]),
        "profit_factor": profit_factor,
        "expectancy_eur": expectancy,
        "win_rate": (
            wins / len(round_trips) if round_trips else None
        ),
        "winning_round_trips": wins,
        "losing_round_trips": losses,
        "maximum_drawdown_eur": str(maximum_drawdown),
        "first_fill_at": account["first_fill_at"],
        "last_fill_at": account["last_fill_at"],
        "last_closed_trade": (
            account["closed_trades"][-1]
            if account["closed_trades"]
            else None
        ),
        "unique_signal_count": len(account["signal_ids"]),
        "unique_portfolio_decision_count": len(account["decision_ids"]),
        "unique_client_order_count": len(account["client_order_ids"]),
        "open_markets": open_markets,
        "performance_sources": ["LIVE_CANARY"],
    }


def rebuild_live_strategy_accounting(
    root: Path,
    *,
    ledger_path: Path | None = None,
    price_by_market: Mapping[str, Decimal | str | float] | None = None,
) -> dict[str, Any]:
    """Rebuild deterministic strategy books and write sanitized live artifacts."""

    root = root.resolve()
    ledger = ledger_path or (
        root / "output" / "checkpoints" / "live_execution.jsonl"
    )
    output = root / "output" / "live"
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "strategy_accounting_decisions.jsonl"
    approvals = _load_approvals(root)
    prices = _market_prices(root)
    prices.update(
        {
            str(market): _decimal(value)
            for market, value in (price_by_market or {}).items()
        }
    )
    events: list[dict[str, Any]] = []
    invalid_lines: list[int] = []
    if ledger.is_file():
        for line_number, line in enumerate(
            ledger.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if isinstance(event, dict):
                events.append(event)
    intents = _intent_index(events)
    accounts = {
        strategy_id: _empty_account(strategy_id, metadata)
        for strategy_id, metadata in approvals.items()
        if metadata.get("approved_for_live") is True
    }
    failures: list[str] = [
        f"INVALID_LEDGER_JSON_LINE:{line}" for line in invalid_lines
    ]
    unattributed_fill_ids: list[str] = []
    excluded_non_strategy_fill_ids: list[str] = []
    for event in events:
        if event.get("event_type") != "FILL":
            continue
        payload = dict(event.get("payload") or {})
        attribution = _fill_attribution(payload, intents)
        strategy_id = str(attribution.get("strategy_id") or "")
        if strategy_id in NON_STRATEGY_EXECUTION_IDS:
            excluded_non_strategy_fill_ids.append(
                str(payload.get("fill_id") or "UNKNOWN")
            )
            continue
        if not strategy_id:
            fill_id = str(payload.get("fill_id") or "UNKNOWN")
            unattributed_fill_ids.append(fill_id)
            continue
        if strategy_id not in accounts:
            accounts[strategy_id] = _empty_account(
                strategy_id,
                {
                    "strategy_dna_hash": attribution.get(
                        "strategy_dna_hash"
                    ),
                    "approved_markets": [payload.get("market")],
                    "approved_for_live": False,
                },
            )
        account = accounts[strategy_id]
        expected_dna = str(account["strategy_dna"] or "")
        accepted_dnas = set(account["accepted_historical_dna_hashes"])
        if expected_dna:
            accepted_dnas.add(expected_dna)
        fill_dna = str(attribution.get("strategy_dna_hash") or "")
        if accepted_dnas and fill_dna and fill_dna not in accepted_dnas:
            failures.append(f"STRATEGY_DNA_MISMATCH:{strategy_id}")
            continue
        market = str(payload.get("market") or "")
        side = str(payload.get("side") or "").upper()
        quantity = _decimal(payload.get("quantity"))
        price = _decimal(payload.get("price"))
        fee = _decimal(payload.get("fee_eur"))
        slippage_bps = _decimal(
            payload.get("actual_slippage_bps")
            if payload.get("actual_slippage_bps") is not None
            else payload.get("slippage_bps"),
            default=Decimal("NaN"),
        )
        if not market or side not in {"BUY", "SELL"} or quantity <= ZERO or price <= ZERO:
            failures.append(f"INVALID_FILL:{payload.get('fill_id') or 'UNKNOWN'}")
            continue
        account["fill_count"] += 1
        account["fees_paid"] += fee
        account["last_price_by_market"][market] = price
        recorded_at = str(
            payload.get("filled_at") or event.get("recorded_at") or ""
        )
        account["first_fill_at"] = account["first_fill_at"] or recorded_at
        account["last_fill_at"] = recorded_at
        if slippage_bps.is_finite():
            account["slippage_bps_by_market"][market].append(slippage_bps)
        for key, target in (
            ("signal_id", "signal_ids"),
            ("portfolio_decision_id", "decision_ids"),
            ("client_order_id", "client_order_ids"),
        ):
            value = str(attribution.get(key) or payload.get(key) or "")
            if value:
                account[target].add(value)
        existing_quantity = account["quantity_by_market"][market]
        existing_basis = account["cost_basis_by_market"][market]
        if side == "BUY":
            if existing_quantity <= ZERO:
                account["open_trade_count"] += 1
                account["opened_at_by_market"][market] = recorded_at
            account["filled_buy_count"] += 1
            account["quantity_by_market"][market] = existing_quantity + quantity
            account["cost_basis_by_market"][market] = (
                existing_basis + quantity * price + fee
            )
            account["entry_quantity_by_market"][market] += quantity
            account["entry_notional_by_market"][market] += quantity * price
            account["entry_fee_by_market"][market] += fee
            continue
        account["filled_sell_count"] += 1
        if existing_quantity <= ZERO or quantity > existing_quantity + Decimal("1e-12"):
            failures.append(
                f"UNMATCHED_SELL_FILL:{strategy_id}:{payload.get('fill_id') or 'UNKNOWN'}"
            )
            continue
        sold_fraction = min(Decimal("1"), quantity / existing_quantity)
        allocated_basis = existing_basis * sold_fraction
        pnl = quantity * price - fee - allocated_basis
        account["realised_pnl"] += pnl
        account["realised_pnl_by_market"][market] += pnl
        account["exit_fee_by_market"][market] += fee
        if pnl > ZERO:
            account["gross_profit"] += pnl
        elif pnl < ZERO:
            account["gross_loss"] += abs(pnl)
        remaining_quantity = max(ZERO, existing_quantity - quantity)
        remaining_basis = max(ZERO, existing_basis - allocated_basis)
        account["quantity_by_market"][market] = remaining_quantity
        account["cost_basis_by_market"][market] = remaining_basis
        if remaining_quantity <= Decimal("1e-12"):
            opened_at = account["opened_at_by_market"].get(market)
            entry_notional = account["entry_notional_by_market"][market]
            entry_quantity = account["entry_quantity_by_market"][market]
            entry_price = (
                entry_notional / entry_quantity
                if entry_quantity > ZERO
                else ZERO
            )
            trade_pnl = account["realised_pnl_by_market"][market]
            trade_fees = (
                account["entry_fee_by_market"][market]
                + account["exit_fee_by_market"][market]
            )
            slippages = account["slippage_bps_by_market"][market]
            average_slippage = (
                sum(slippages, ZERO) / Decimal(len(slippages))
                if slippages
                else None
            )
            account["quantity_by_market"][market] = ZERO
            account["cost_basis_by_market"][market] = ZERO
            account["closed_trade_count"] += 1
            account["realized_round_trips"].append(trade_pnl)
            account["closed_trades"].append(
                {
                    "market": market,
                    "opened_at": opened_at,
                    "closed_at": recorded_at,
                    "holding_seconds": _holding_seconds(
                        opened_at,
                        recorded_at,
                    ),
                    "entry_price_eur": str(entry_price),
                    "exit_price_eur": str(price),
                    "quantity": str(entry_quantity),
                    "fees_eur": str(trade_fees),
                    "net_pnl_eur": str(trade_pnl),
                    "average_slippage_bps": (
                        str(average_slippage)
                        if average_slippage is not None
                        else None
                    ),
                }
            )
            account["entry_quantity_by_market"][market] = ZERO
            account["entry_notional_by_market"][market] = ZERO
            account["entry_fee_by_market"][market] = ZERO
            account["exit_fee_by_market"][market] = ZERO
            account["realised_pnl_by_market"][market] = ZERO
            account["opened_at_by_market"].pop(market, None)
            account["slippage_bps_by_market"][market].clear()
    if unattributed_fill_ids:
        failures.append("UNATTRIBUTED_LIVE_FILLS")
    serialized = [
        _serialize_account(account, prices)
        for _, account in sorted(accounts.items())
    ]
    integrity = "PASSED" if not failures else "FAILED"
    generated_at = utc_iso()
    source_hash = stable_hash(
        {
            "ledger_events": events,
            "approvals": approvals,
            "prices": {key: str(value) for key, value in prices.items()},
        }
    )
    account_payload = {
        "schema_version": "live_strategy_accounts_v1",
        "generated_at": generated_at,
        "source_ledger": str(ledger),
        "source_hash": source_hash,
        "integrity_status": integrity,
        "hard_blockers": sorted(set(failures)),
        "unattributed_fill_count": len(unattributed_fill_ids),
        "unattributed_fill_ids": unattributed_fill_ids,
        "excluded_non_strategy_fill_count": len(
            excluded_non_strategy_fill_ids
        ),
        "excluded_non_strategy_fill_ids": excluded_non_strategy_fill_ids,
        "live_strategy_account_count": len(serialized),
        "strategies": serialized,
        "orders_generated": 0,
        "orders_submitted": 0,
        "secrets_serialized": False,
    }
    performance_payload = {
        "schema_version": "live_strategy_performance_v1",
        "generated_at": generated_at,
        "integrity_status": integrity,
        "strategies": [
            {
                key: row[key]
                for key in (
                    "strategy_id",
                    "strategy_dna",
                    "strategy_family",
                    "timeframe",
                    "authority_level",
                    "lifecycle_state",
                    "closed_trade_count",
                    "open_trade_count",
                    "realised_pnl_eur",
                    "unrealised_pnl_eur",
                    "net_pnl_eur",
                    "fees_paid_eur",
                    "profit_factor",
                    "expectancy_eur",
                    "win_rate",
                    "maximum_drawdown_eur",
                )
            }
            for row in serialized
        ],
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    allocations_payload = {
        "schema_version": "live_strategy_allocations_v1",
        "generated_at": generated_at,
        "allocation_policy": "OPERATOR_CAPS_WITH_EVIDENCE_BASED_ELIGIBILITY",
        "automatic_cap_increases": False,
        "strategies": [
            {
                key: row[key]
                for key in (
                    "strategy_id",
                    "strategy_dna",
                    "authority_level",
                    "allocated_capital_eur",
                    "used_capital_eur",
                    "available_capital_eur",
                    "operator_approved",
                    "autoscale",
                )
            }
            for row in serialized
        ],
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    degradation_payload = evaluate_strategy_degradation(
        accounts,
        integrity_failures=failures,
        generated_at=generated_at,
    )
    atomic_write_json(output / "strategy_accounts.json", account_payload)
    atomic_write_json(output / "strategy_performance.json", performance_payload)
    atomic_write_json(output / "strategy_allocations.json", allocations_payload)
    operations = root / "output" / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        operations / "strategy_degradation.json",
        degradation_payload,
    )
    decision_hash = stable_hash(
        {
            "source_hash": source_hash,
            "integrity": integrity,
            "strategies": serialized,
        }
    )
    previous_hash = None
    if decisions_path.is_file():
        for line in reversed(
            decisions_path.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            try:
                previous_hash = json.loads(line).get("decision_hash")
            except json.JSONDecodeError:
                previous_hash = None
            break
    if decision_hash != previous_hash:
        append_jsonl(
            decisions_path,
            {
                "schema_version": "live_strategy_accounting_decision_v1",
                "recorded_at": datetime.now(UTC),
                "decision_hash": decision_hash,
                "source_hash": source_hash,
                "integrity_status": integrity,
                "live_strategy_account_count": len(serialized),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
    return account_payload


__all__ = ["rebuild_live_strategy_accounting"]
