"""Controlled sell-only reallocation for pre-existing spot inventory.

This module deliberately does not create strategy evidence.  It exists for an
operator-authorised reduction of inventory that predates the canonical live
strategy ledger, so wallet capacity can be moved back to EUR without pretending
that the reduction was a strategy exit.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp

from config.settings import Settings
from core.contracts import (
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderTimeInForce,
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
)
from core.live_asset_preflight import _market_exceptions, live_account_health
from notifications.telegram import TelegramNotifier
from risk.risk_manager import KillSwitch, PortfolioSnapshot, PositionExposure, RiskManager
from utils.common import append_jsonl, atomic_write_json, stable_hash, utc_iso

REALLOCATION_STRATEGY_ID = "OPERATOR_INVENTORY_REALLOCATION_NOT_STRATEGY_TRADE"
REALLOCATION_REASON = "OPERATOR_AUTHORISED_PREEXISTING_INVENTORY_REALLOCATION"
SCHEMA_VERSION = "inventory_reallocation_v1"


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def target_reallocation_quantity(
    *,
    available_quantity: Decimal,
    mark_price_eur: Decimal,
    account_equity_eur: Decimal,
    target_weight: Decimal | None,
) -> Decimal:
    """Return only the inventory units above an explicit target weight."""

    if available_quantity <= 0:
        return Decimal("0")
    if target_weight is None:
        return available_quantity
    if target_weight < 0 or target_weight >= 1:
        raise ValueError("target weight must be in [0, 1)")
    if mark_price_eur <= 0 or account_equity_eur <= 0:
        raise ValueError("positive mark price and account equity are required")
    target_units = account_equity_eur * target_weight / mark_price_eur
    return max(Decimal("0"), available_quantity - target_units)


def validate_reallocation_authority(
    settings: Settings,
    *,
    market: str,
    approval_reference: str,
) -> dict[str, Any]:
    """Validate a persisted, market-specific operator exception."""

    normalized = str(market).strip().upper().replace("/", "-").replace("_", "-")
    failures: list[str] = []
    if len(normalized.split("-")) != 2 or not normalized.endswith("-EUR"):
        failures.append("EUR_SPOT_MARKET_REQUIRED")
    exception = _market_exceptions(settings).get(normalized)
    if exception is None:
        failures.append("MARKET_REALLOCATION_AUTHORITY_MISSING")
    else:
        if not exception.get("spot_only"):
            failures.append("SPOT_ONLY_AUTHORITY_REQUIRED")
        if str(exception.get("approval_reference") or "") != str(
            approval_reference
        ).strip():
            failures.append("APPROVAL_REFERENCE_MISMATCH")
    return {
        "market": normalized,
        "approved": not failures,
        "failures": failures,
        "approval_reference": (
            str(exception.get("approval_reference"))
            if exception is not None
            else None
        ),
        "approval_phrase_stored": False,
        "strategy_performance_attribution": False,
        "spot_only": True,
    }


def evaluate_sell_book(
    *,
    market: str,
    quantity: Decimal,
    bids: Sequence[Sequence[Any]],
    asks: Sequence[Sequence[Any]],
    quote_volume_24h_eur: Decimal,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate a market sell against the visible bid book."""

    parsed_bids = sorted(
        (
            (_decimal(row[0]), _decimal(row[1]))
            for row in bids
            if len(row) >= 2
        ),
        reverse=True,
    )
    parsed_asks = sorted(
        (
            (_decimal(row[0]), _decimal(row[1]))
            for row in asks
            if len(row) >= 2
        ),
    )
    parsed_bids = [
        (price, amount)
        for price, amount in parsed_bids
        if price > 0 and amount > 0
    ]
    parsed_asks = [
        (price, amount)
        for price, amount in parsed_asks
        if price > 0 and amount > 0
    ]
    if quantity <= 0:
        raise ValueError("reallocation quantity must be positive")
    if not parsed_bids or not parsed_asks:
        raise ValueError("public order book must contain bids and asks")

    best_bid = parsed_bids[0][0]
    best_ask = parsed_asks[0][0]
    midpoint = (best_bid + best_ask) / Decimal("2")
    if best_ask <= best_bid or midpoint <= 0:
        raise ValueError("public order book is crossed or invalid")

    remaining = quantity
    filled = Decimal("0")
    gross_eur = Decimal("0")
    marketable_limit_price = Decimal("0")
    for price, amount in parsed_bids:
        if remaining <= 0:
            break
        take = min(remaining, amount)
        filled += take
        gross_eur += take * price
        if take > 0:
            marketable_limit_price = price
        remaining -= take
    average_price = gross_eur / filled if filled > 0 else Decimal("0")
    spread_bps = (best_ask - best_bid) / midpoint * Decimal("10000")
    slippage_bps = (
        (Decimal("1") - average_price / best_bid) * Decimal("10000")
        if average_price > 0
        else Decimal("Infinity")
    )
    visible_bid_depth_eur = sum(
        price * amount for price, amount in parsed_bids
    )
    participation_pct = (
        gross_eur / visible_bid_depth_eur * Decimal("100")
        if visible_bid_depth_eur > 0
        else Decimal("Infinity")
    )
    blockers: list[str] = []
    if remaining > Decimal("0.00000001"):
        blockers.append("INSUFFICIENT_VISIBLE_BID_LIQUIDITY")
    if spread_bps > _decimal(limits.get("maximum_spread_bps")):
        blockers.append("SPREAD_LIMIT_EXCEEDED")
    if visible_bid_depth_eur < _decimal(
        limits.get("minimum_visible_ask_depth_eur")
    ):
        blockers.append("MINIMUM_VISIBLE_BID_DEPTH_NOT_MET")
    if quote_volume_24h_eur < _decimal(
        limits.get("minimum_24h_quote_volume_eur")
    ):
        blockers.append("MINIMUM_24H_QUOTE_VOLUME_NOT_MET")
    if slippage_bps > _decimal(limits.get("maximum_slippage_bps")):
        blockers.append("ESTIMATED_SELL_SLIPPAGE_LIMIT_EXCEEDED")
    if participation_pct > _decimal(
        limits.get("maximum_visible_liquidity_participation_pct")
    ):
        blockers.append("VISIBLE_LIQUIDITY_PARTICIPATION_LIMIT_EXCEEDED")
    return {
        "status": "PASSED" if not blockers else "BLOCKED",
        "market": market,
        "quantity": str(quantity),
        "best_bid": str(best_bid),
        "best_ask": str(best_ask),
        "marketable_limit_price": str(marketable_limit_price),
        "spread_bps": str(spread_bps),
        "estimated_average_sell_price": str(average_price),
        "estimated_gross_eur": str(gross_eur),
        "estimated_sell_slippage_bps": str(slippage_bps),
        "visible_bid_depth_eur": str(visible_bid_depth_eur),
        "visible_liquidity_participation_pct": str(participation_pct),
        "quote_volume_24h_eur": str(quote_volume_24h_eur),
        "unfilled_quantity": str(remaining),
        "limits": dict(limits),
        "blocking_reasons": blockers,
    }


async def _current_sell_liquidity(
    session: aiohttp.ClientSession,
    *,
    settings: Settings,
    market: str,
    quantity: Decimal,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=10)
    async with session.get(
        f"https://api.bitvavo.com/v2/{market}/book",
        params={"depth": 100},
        timeout=timeout,
    ) as response:
        if response.status >= 400:
            raise RuntimeError(f"PUBLIC_BOOK_HTTP_{response.status}")
        book = await response.json(content_type=None)
    async with session.get(
        "https://api.bitvavo.com/v2/ticker/24h",
        params={"market": market},
        timeout=timeout,
    ) as response:
        if response.status >= 400:
            raise RuntimeError(f"PUBLIC_TICKER_HTTP_{response.status}")
        ticker = await response.json(content_type=None)
    if not isinstance(book, dict) or not isinstance(ticker, dict):
        raise RuntimeError("PUBLIC_LIQUIDITY_INVALID_RESPONSE")
    last = _decimal(ticker.get("last"))
    base_volume = _decimal(ticker.get("volume"))
    quote_volume = _decimal(
        ticker.get("volumeQuote")
        or ticker.get("quoteVolume")
        or base_volume * last
    )
    return evaluate_sell_book(
        market=market,
        quantity=quantity,
        bids=list(book.get("bids") or []),
        asks=list(book.get("asks") or []),
        quote_volume_24h_eur=quote_volume,
        limits=settings.autonomous_live.liquidity_limits(market),
    )


async def _quantity_decimals(
    session: aiohttp.ClientSession,
    *,
    market: str,
) -> int:
    timeout = aiohttp.ClientTimeout(total=10)
    async with session.get(
        "https://api.bitvavo.com/v2/markets",
        timeout=timeout,
    ) as response:
        if response.status >= 400:
            raise RuntimeError(f"BITVAVO_MARKETS_HTTP_{response.status}")
        payload = await response.json(content_type=None)
    if not isinstance(payload, list):
        raise RuntimeError("BITVAVO_MARKETS_INVALID_RESPONSE")
    row = next(
        (
            value
            for value in payload
            if isinstance(value, Mapping)
            and str(value.get("market") or "").upper() == market
        ),
        None,
    )
    if row is None:
        raise RuntimeError("BITVAVO_MARKET_METADATA_MISSING")
    decimals = int(row.get("quantityDecimals"))
    if decimals < 0 or decimals > 18:
        raise RuntimeError("BITVAVO_QUANTITY_DECIMALS_INVALID")
    return decimals


def _artifact_paths(settings: Settings) -> tuple[Path, Path]:
    root = settings.paths.output_dir / "operations"
    return (
        root / "inventory_reallocation.json",
        root / "inventory_reallocations.jsonl",
    )


def _write_result(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    current, ledger = _artifact_paths(settings)
    current.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(current, payload)
    append_jsonl(ledger, payload)
    payload["artifact"] = str(current)
    payload["ledger"] = str(ledger)
    return payload


async def _repair_unacknowledged_reallocation_intents(
    settings: Settings,
    *,
    market: str,
) -> dict[str, Any]:
    """Resolve prior definitive rejects before allowing a new attempt."""

    from execution.execution import build_live_client

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20)
    ) as session:
        client = build_live_client(
            settings,
            session=session,
            ledger_path=(
                settings.paths.checkpoints_dir / "live_execution.jsonl"
            ),
        )
        events = client.ledger.events()
        resolved = {
            str((event.get("payload") or {}).get("client_order_id") or "")
            for event in events
            if event.get("event_type")
            in {"ORDER_ACKNOWLEDGED", "ORDER_REJECTED"}
        }
        pending = [
            dict(event.get("payload") or {})
            for event in events
            if event.get("event_type") == "ORDER_INTENT"
            and str((event.get("payload") or {}).get("market") or "")
            == market
            and str(
                (event.get("payload") or {}).get("strategy_id") or ""
            )
            == REALLOCATION_STRATEGY_ID
            and str(
                (event.get("payload") or {}).get("client_order_id") or ""
            )
            not in resolved
        ]
        repaired = 0
        for intent in pending:
            client_order_id = str(intent.get("client_order_id") or "")
            try:
                order = await client.get_order(
                    market=market,
                    client_order_id=client_order_id,
                )
            except ExecutionBlocked as exc:
                if "HTTP 404" not in str(exc):
                    return {
                        "status": "BLOCKED",
                        "reason_code": (
                            "PRIOR_INTENT_REMOTE_STATE_UNAVAILABLE"
                        ),
                        "repaired": repaired,
                    }
                client.ledger.append(
                    "ORDER_REJECTED",
                    {
                        "intent_id": intent.get("intent_id"),
                        "client_order_id": client_order_id,
                        "market": market,
                        "side": "SELL",
                        "strategy_id": REALLOCATION_STRATEGY_ID,
                        "portfolio_decision_id": intent.get(
                            "portfolio_decision_id"
                        ),
                        "http_status": 404,
                        "venue_error_code": "CONFIRMED_NOT_FOUND",
                        "definitive": True,
                        "recovered_after_restart": True,
                    },
                )
                repaired += 1
                continue
            except ReconciliationRequired:
                return {
                    "status": "BLOCKED",
                    "reason_code": "PRIOR_INTENT_REMOTE_STATE_AMBIGUOUS",
                    "repaired": repaired,
                }
            client.ledger.append(
                "ORDER_ACKNOWLEDGED",
                {
                    "intent_id": intent.get("intent_id"),
                    "client_order_id": client_order_id,
                    "order_id": order.get("orderId"),
                    "status": order.get("status"),
                    "market": market,
                    "side": "SELL",
                    "strategy_id": REALLOCATION_STRATEGY_ID,
                    "portfolio_decision_id": intent.get(
                        "portfolio_decision_id"
                    ),
                    "recovered_after_restart": True,
                },
            )
            client.record_final_fill(
                order,
                fallback_market=market,
                fallback_side=OrderSide.SELL,
                fallback_quantity=_decimal(intent.get("quantity")),
                fallback_price=_decimal(order.get("price")),
            )
            return {
                "status": "REMOTE_ORDER_RECOVERED",
                "reason_code": "PRIOR_INTENT_EXISTS_AT_VENUE",
                "repaired": repaired,
            }
        return {
            "status": "READY",
            "reason_code": (
                "DEFINITIVE_REJECTION_REPAIRED"
                if repaired
                else "NO_UNACKNOWLEDGED_REALLOCATION_INTENT"
            ),
            "repaired": repaired,
        }


async def reallocate_preexisting_inventory(
    settings: Settings,
    *,
    market: str,
    approval_reference: str,
    submit: bool,
    target_weight: Decimal | None = None,
) -> dict[str, Any]:
    """Preflight or sell pre-existing units down to a target weight."""

    from execution.execution import LivePreflight, build_live_client

    authority = validate_reallocation_authority(
        settings,
        market=market,
        approval_reference=approval_reference,
    )
    normalized = str(authority["market"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": utc_iso(),
        "status": "BLOCKED",
        "mode": "SUBMIT" if submit else "PREFLIGHT_ONLY",
        "market": normalized,
        "side": "SELL",
        "order_type": "LIMIT",
        "time_in_force": "IOC",
        "market_fallback_enabled": False,
        "authority": authority,
        "strategy_id": REALLOCATION_STRATEGY_ID,
        "strategy_performance_attribution": False,
        "risk_reduction_only": True,
        "target_weight": (
            str(target_weight) if target_weight is not None else "0"
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
        "withdrawals_attempted": 0,
        "secrets_serialized": False,
        "exchange_identifiers_masked": True,
    }
    if not authority["approved"]:
        result["failures"] = list(authority["failures"])
        return _write_result(settings, result)

    ledger_repair = await _repair_unacknowledged_reallocation_intents(
        settings,
        market=normalized,
    )
    result["ledger_repair"] = ledger_repair
    if ledger_repair["status"] not in {"READY"}:
        result["failures"] = [str(ledger_repair["reason_code"])]
        return _write_result(settings, result)

    health = await live_account_health(
        settings,
        markets=(normalized,),
    )
    result["account_health_status"] = health.get("status")
    result["reconciliation"] = health.get("reconciliation")
    if (
        health.get("status") != "READY"
        or health.get("risk_reduction_allowed") is not True
        or not (health.get("reconciliation") or {}).get("healthy")
    ):
        result["failures"] = list(
            health.get("failures") or ["LIVE_ACCOUNT_HEALTH_BLOCKED"]
        )
        return _write_result(settings, result)

    base = normalized.split("-")[0]
    holding = next(
        (
            row
            for row in (
                (health.get("account") or {}).get("non_eur_holdings")
                or []
            )
            if str(row.get("symbol") or "").upper() == base
        ),
        None,
    )
    available_quantity = _decimal((holding or {}).get("available"))
    if available_quantity <= 0:
        result["failures"] = ["NO_AVAILABLE_INVENTORY_TO_REALLOCATE"]
        return _write_result(settings, result)
    valuation = (health.get("account") or {}).get("portfolio_valuation") or {}
    holding_valuation = next(
        (
            row
            for row in valuation.get("holdings") or []
            if str(row.get("symbol") or "").upper() == base
        ),
        None,
    )
    mark_price = _decimal((holding_valuation or {}).get("price_eur"))
    account_equity = _decimal(valuation.get("estimated_total_equity_eur"))
    try:
        quantity = target_reallocation_quantity(
            available_quantity=available_quantity,
            mark_price_eur=mark_price,
            account_equity_eur=account_equity,
            target_weight=target_weight,
        )
    except ValueError as exc:
        result["failures"] = [
            "INVALID_REALLOCATION_TARGET_" + type(exc).__name__.upper()
        ]
        return _write_result(settings, result)
    result.update(
        {
            "available_quantity": str(available_quantity),
            "mark_price_eur": str(mark_price),
            "account_equity_eur": str(account_equity),
            "current_weight": str(
                available_quantity * mark_price / account_equity
                if account_equity > 0
                else Decimal("0")
            ),
            "target_value_eur": str(
                account_equity * target_weight
                if target_weight is not None
                else Decimal("0")
            ),
        }
    )
    if quantity <= 0:
        result.update(
            {
                "status": "NO_ACTION_REQUIRED",
                "quantity": "0",
                "failures": [],
            }
        )
        return _write_result(settings, result)

    timeout = aiohttp.ClientTimeout(
        total=min(20.0, settings.market_data.request_timeout_seconds)
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        quantity_decimals = await _quantity_decimals(
            session,
            market=normalized,
        )
        quantum = Decimal(1).scaleb(-quantity_decimals)
        tradable_quantity = quantity.quantize(
            quantum,
            rounding=ROUND_DOWN,
        )
        rounding_dust_quantity = quantity - tradable_quantity
        if tradable_quantity <= 0:
            result["failures"] = [
                "INVENTORY_BELOW_EXCHANGE_QUANTITY_PRECISION"
            ]
            return _write_result(settings, result)
        liquidity = await _current_sell_liquidity(
            session,
            settings=settings,
            market=normalized,
            quantity=tradable_quantity,
        )
        result["quantity"] = str(tradable_quantity)
        result["quantity_decimals"] = quantity_decimals
        result["rounding_dust_quantity"] = str(rounding_dust_quantity)
        result["estimated_remaining_inventory_quantity"] = str(
            available_quantity - tradable_quantity
        )
        result["liquidity"] = liquidity
        if liquidity["status"] != "PASSED":
            result["failures"] = list(liquidity["blocking_reasons"])
            return _write_result(settings, result)

        client = build_live_client(
            settings,
            session=session,
            ledger_path=(
                settings.paths.checkpoints_dir / "live_execution.jsonl"
            ),
        )
        balances = await client.balances()
        market_rules = await client.execution_market_rules(normalized)
        reconciled = await client.reconcile(markets=(normalized,))
        latest_owned = next(
            (
                _decimal(row.get("available"))
                for row in balances
                if str(row.get("symbol") or "").upper() == base
            ),
            Decimal("0"),
        )
        if not reconciled.healthy:
            result["failures"] = ["LIVE_RECONCILIATION_FAILED"]
            return _write_result(settings, result)
        if latest_owned != available_quantity:
            result["failures"] = ["INVENTORY_CHANGED_DURING_PREFLIGHT"]
            return _write_result(settings, result)
        limit_price = market_rules.price(
            _decimal(liquidity["marketable_limit_price"])
        )
        if limit_price <= 0:
            result["failures"] = ["INVALID_MARKETABLE_LIMIT_PRICE"]
            return _write_result(settings, result)
        result["limit_price"] = str(limit_price)
        result["time_in_force"] = OrderTimeInForce.IOC.value

        kill_switch = KillSwitch(
            settings.paths.checkpoints_dir / "kill_switch.json"
        )
        equity = _decimal(
            (
                (health.get("account") or {}).get("portfolio_valuation")
                or {}
            ).get("estimated_total_equity_eur")
        )
        price = _decimal(liquidity["estimated_average_sell_price"])
        snapshot = PortfolioSnapshot(
            equity_eur=float(equity),
            cash_eur=float(
                _decimal((health.get("account") or {}).get("eur_available"))
            ),
            day_start_equity_eur=float(equity),
            peak_equity_eur=float(equity),
            trades_today=0,
            positions=(
                PositionExposure(
                    market=normalized,
                    quantity=float(available_quantity),
                    mark_price=float(price),
                    open_risk_eur=0.0,
                ),
            ),
            reconciled=True,
        )
        risk = RiskManager.from_settings(
            settings,
            kill_switch_path=(
                settings.paths.checkpoints_dir / "kill_switch.json"
            ),
        ).assess_exit(
            market=normalized,
            requested_quantity=float(tradable_quantity),
            snapshot=snapshot,
        )
        result["risk"] = {
            "approved": risk.approved,
            "reason_codes": [
                getattr(value, "value", str(value))
                for value in risk.reason_codes
            ],
            "kill_switch_active": kill_switch.active,
            "sell_reduces_risk": True,
        }
        if not risk.approved:
            result["failures"] = ["CENTRAL_RISK_MANAGER_REJECTED_EXIT"]
            return _write_result(settings, result)

        preflight = LivePreflight.evaluate(
            settings,
            markets=(normalized,),
            strategy_status=ResearchStatus.PAPER_CANDIDATE,
            data_healthy=True,
            risk_manager_healthy=True,
            exchange_healthy=True,
            reconciliation_healthy=True,
            # A kill switch blocks entries, but must not block a risk-reducing
            # sell.  All other static authority and credential checks remain.
            kill_switch_active=False,
            canary_exception_approved=True,
            operator_canary_authorized=True,
            cap_limits={
                "capital_level": 1,
                "max_order_eur": "10",
                "max_exposure_eur": "10",
                "max_positions": 1,
                "max_new_orders_per_day": 1,
            },
        )
        result["live_preflight"] = {
            "passed": preflight.passed,
            "failures": list(preflight.failures),
        }
        if not preflight.passed or preflight.capability is None:
            result["failures"] = list(preflight.failures)
            return _write_result(settings, result)

        decision_hash = stable_hash(
            [
                SCHEMA_VERSION,
                normalized,
                str(tradable_quantity),
                authority["approval_reference"],
                "TARGET_WEIGHT",
                str(target_weight if target_weight is not None else "0"),
            ],
            length=40,
        )
        result["decision_id"] = decision_hash
        prior_rejections = sum(
            1
            for event in client.ledger.events()
            if event.get("event_type") == "ORDER_REJECTED"
            and str(
                (event.get("payload") or {}).get(
                    "portfolio_decision_id"
                )
                or ""
            )
            == decision_hash
        )
        attempt_number = prior_rejections + 1
        idempotency_key = (
            f"inventory-reallocation:{decision_hash}:"
            f"attempt:{attempt_number}"
        )
        result["attempt_number"] = attempt_number
        result["idempotency_key_hash"] = stable_hash(
            idempotency_key,
            length=24,
        )
        result["estimated_gross_eur"] = liquidity["estimated_gross_eur"]
        if not submit:
            result["status"] = "READY_TO_SUBMIT"
            result["failures"] = []
            return _write_result(settings, result)

        intent = OrderIntent(
            intent_id=decision_hash[:32],
            idempotency_key=idempotency_key,
            market=normalized,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=tradable_quantity,
            limit_price=limit_price,
            time_in_force=OrderTimeInForce.IOC,
            strategy_id=REALLOCATION_STRATEGY_ID,
            portfolio_decision_id=decision_hash,
            reason_codes=(REALLOCATION_REASON,),
        )
        notifier = TelegramNotifier(
            settings.telegram,
            output_directory=(
                settings.paths.output_dir / "notifications"
            ),
            allowed_markets=tuple(
                dict.fromkeys((*settings.operational.markets, normalized))
            ),
        )
        result["telegram_pre_submit"] = notifier.notify_order_event(
            "ORDER_SUBMITTING",
            {
                "intent_id": intent.intent_id,
                "market": normalized,
                "side": "SELL",
                "order_type": "LIMIT",
                "time_in_force": "IOC",
                "price": float(limit_price),
                "quantity": str(tradable_quantity),
                "notional_eur": float(
                    _decimal(liquidity["estimated_gross_eur"])
                ),
                "strategy_id": REALLOCATION_STRATEGY_ID,
            },
        )
        result["orders_generated"] = 1
        try:
            order = await client.submit_order(
                intent,
                capability=preflight.capability,
                estimated_price=price,
                reconciled_owned_quantity=latest_owned,
                reconciled_total_exposure_eur=None,
                reconciled_open_positions=0,
                exchange_minimum_order_eur=Decimal("5"),
            )
        except ReconciliationRequired:
            result.update(
                {
                    "status": "RECONCILIATION_REQUIRED",
                    "orders_submitted": 1,
                    "failures": [
                        "AMBIGUOUS_ORDER_STATE_RECONCILE_CLIENT_ID"
                    ],
                }
            )
            result["telegram_post_submit"] = notifier.notify_system_event(
                "RECONCILIATION_MISMATCH",
                {
                    "market": normalized,
                    "status": "ORDER_STATE_AMBIGUOUS",
                    "reason_code": (
                        "EXACT_CLIENT_ORDER_RECONCILIATION_REQUIRED"
                    ),
                },
            )
            return _write_result(settings, result)
        except ExecutionBlocked as exc:
            result.update(
                {
                    "status": "ORDER_REJECTED",
                    "failures": [type(exc).__name__],
                    "orders_submitted": 0,
                }
            )
            result["telegram_post_submit"] = notifier.notify_order_event(
                "ORDER_REJECTED",
                {
                    "intent_id": intent.intent_id,
                    "market": normalized,
                    "side": "SELL",
                    "order_type": "LIMIT",
                    "time_in_force": "IOC",
                    "quantity": str(tradable_quantity),
                    "strategy_id": REALLOCATION_STRATEGY_ID,
                    "reason_code": type(exc).__name__,
                },
            )
            return _write_result(settings, result)

        order_status = (
            str(order.get("status") or "UNKNOWN")
            .replace("_", "")
            .replace("-", "")
            .upper()
        )
        public_order_id = stable_hash(
            ["public-inventory-order", order.get("orderId")],
            length=16,
        )
        result.update(
            {
                "status": (
                    "FILLED"
                    if order_status == "FILLED"
                    else "PARTIALLY_FILLED"
                    if order_status == "PARTIALLYFILLED"
                    else "SUBMITTED"
                ),
                "order": {
                    "public_order_id": public_order_id,
                    "status": order.get("status"),
                    "market": normalized,
                    "side": "SELL",
                    "order_type": "LIMIT",
                    "time_in_force": "IOC",
                    "limit_price": str(limit_price),
                    "filled_quantity": str(
                        order.get("filledAmount") or "0"
                    ),
                    "filled_quote_eur": str(
                        order.get("filledAmountQuote") or "0"
                    ),
                    "estimated_remaining_inventory_quantity": result[
                        "estimated_remaining_inventory_quantity"
                    ],
                    "fee_eur": str(order.get("feePaid") or "0"),
                },
                "orders_submitted": 1,
                "failures": [],
            }
        )
        result["telegram_post_submit"] = notifier.notify_order_event(
            (
                "ORDER_FILLED"
                if result["status"] == "FILLED"
                else "ORDER_PARTIALLY_FILLED"
            ),
            {
                "order_id": public_order_id,
                "market": normalized,
                "side": "SELL",
                "order_type": "LIMIT",
                "time_in_force": "IOC",
                "price": float(limit_price),
                "quantity": result["order"]["filled_quantity"],
                "notional_eur": result["order"]["filled_quote_eur"],
                "fee_eur": result["order"]["fee_eur"],
                "strategy_id": REALLOCATION_STRATEGY_ID,
            },
        )
        result["post_submit_reconciliation"] = asdict(
            await client.reconcile(markets=(normalized,))
        )
        return _write_result(settings, result)


__all__ = [
    "REALLOCATION_STRATEGY_ID",
    "evaluate_sell_book",
    "reallocate_preexisting_inventory",
    "target_reallocation_quantity",
    "validate_reallocation_authority",
]
