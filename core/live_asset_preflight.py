"""Read-only market and authority preflight for user-held spot assets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import aiohttp

from config.settings import Settings
from core.autonomous_trading import LiveStrategyApprovalRegistry
from core.inventory_risk_override import evaluate_inventory_risk_override
from core.market_exceptions import load_execution_market_exceptions
from core.practical_governance import live_canary_authority
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso


def _market_exceptions(settings: Settings) -> dict[str, dict[str, Any]]:
    """Backward-compatible alias for the shared strict loader."""

    return load_execution_market_exceptions(settings)


async def _public_bitvavo_market_metadata(
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(
        total=min(15.0, settings.market_data.request_timeout_seconds)
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("https://api.bitvavo.com/v2/markets") as response:
            if response.status >= 400:
                raise RuntimeError(f"BITVAVO_MARKETS_HTTP_{response.status}")
            payload = await response.json(content_type=None)
    if not isinstance(payload, list):
        raise RuntimeError("BITVAVO_MARKETS_INVALID")
    return {
        str(row.get("market") or "").upper(): dict(row)
        for row in payload
        if isinstance(row, dict) and row.get("market")
    }


async def _public_bitvavo_eur_prices(
    settings: Settings,
) -> dict[str, Decimal]:
    timeout = aiohttp.ClientTimeout(
        total=min(15.0, settings.market_data.request_timeout_seconds)
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            "https://api.bitvavo.com/v2/ticker/price"
        ) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"BITVAVO_TICKER_PRICE_HTTP_{response.status}"
                )
            payload = await response.json(content_type=None)
    if not isinstance(payload, list):
        raise RuntimeError("BITVAVO_TICKER_PRICE_INVALID")
    prices: dict[str, Decimal] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        market = str(row.get("market") or "").upper()
        price = _decimal_or_zero(row.get("price"))
        if market.endswith("-EUR") and price > 0:
            prices[market] = price
    return prices


def _public_market_projection(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "venue_available": False,
            "venue_status": "MISSING",
            "minimum_order_base": None,
            "minimum_order_quote": None,
            "price_precision": None,
            "amount_precision": None,
            "order_types": [],
        }
    status = str(row.get("status") or "unknown")
    return {
        "venue_available": status.casefold() in {"trading", "active"},
        "venue_status": status,
        "minimum_order_base": row.get("minOrderInBaseAsset"),
        "minimum_order_quote": row.get("minOrderInQuoteAsset"),
        "price_precision": row.get("pricePrecision"),
        "amount_precision": row.get("amountPrecision"),
        "order_types": list(row.get("orderTypes") or []),
    }


def _decimal_or_zero(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


_EXTERNAL_TRANSACTION_TYPES = {
    "deposit",
    "withdrawal",
    "internal_transfer",
    "withdrawal_cancelled",
    "external_transferred_funds",
    "manually_assigned",
    "manually_assigned_bitvavo",
}


def _exchange_external_eur_delta(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project only explicit non-trading EUR movements from broker history."""

    delta = Decimal("0")
    matched = 0
    evidence: list[dict[str, Any]] = []
    for row in rows:
        transaction_type = str(row.get("type") or "").casefold()
        if transaction_type not in _EXTERNAL_TRANSACTION_TYPES:
            continue
        sent = (
            _decimal_or_zero(row.get("sentAmount"))
            if str(row.get("sentCurrency") or "").upper() == "EUR"
            else Decimal("0")
        )
        received = (
            _decimal_or_zero(row.get("receivedAmount"))
            if str(row.get("receivedCurrency") or "").upper() == "EUR"
            else Decimal("0")
        )
        fee = (
            _decimal_or_zero(row.get("feesAmount"))
            if str(row.get("feesCurrency") or "").upper() == "EUR"
            else Decimal("0")
        )
        eur_delta = received - sent - fee
        if eur_delta == 0:
            continue
        delta += eur_delta
        matched += 1
        evidence.append(
            {
                "transaction_hash": stable_hash(
                    [
                        "BITVAVO_TRANSACTION",
                        str(row.get("transactionId") or ""),
                    ],
                    length=24,
                ),
                "executed_at": row.get("executedAt"),
                "type": transaction_type,
                "eur_delta": str(eur_delta),
            }
        )
    return {
        "external_eur_delta": str(delta),
        "matched_external_transactions": matched,
        "evidence": evidence,
    }


def _transaction_history_window(
    accepted_at: datetime,
    *,
    current_ms: int,
    maximum_clock_drift_ms: int,
) -> tuple[int, int] | None:
    """Return a safe incremental history window.

    Concurrent health/reconciliation cycles can update the accepted cash
    baseline after an older cycle captured its initial wall clock.  A tiny
    forward skew therefore means that there is simply no new interval to
    query; it is not a malformed broker request.  A baseline materially in
    the future remains fail-closed.
    """

    from_ms = int(accepted_at.timestamp() * 1_000) + 1
    if from_ms <= current_ms:
        return from_ms, current_ms
    if from_ms - current_ms <= maximum_clock_drift_ms + 1:
        return None
    raise ValueError("cash continuity baseline is materially in the future")


def _safe_balance_projection(
    balances: list[dict[str, Any]],
) -> dict[str, Any]:
    holdings: list[dict[str, str]] = []
    for row in balances:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or not symbol.replace("-", "").isalnum():
            continue
        available = _decimal_or_zero(row.get("available"))
        in_order = _decimal_or_zero(row.get("inOrder"))
        total = available + in_order
        if total <= 0:
            continue
        holdings.append(
            {
                "symbol": symbol,
                "available": str(available),
                "in_order": str(in_order),
                "total": str(total),
            }
        )
    holdings.sort(key=lambda row: row["symbol"])
    by_symbol = {row["symbol"]: row for row in holdings}
    non_eur = [row for row in holdings if row["symbol"] != "EUR"]
    return {
        "eur_available": by_symbol.get("EUR", {}).get("available", "0"),
        "non_eur_holdings": non_eur,
        "non_eur_holding_count": len(non_eur),
        "positive_symbol_count": len(holdings),
    }


def _managed_inventory_quantities(output_dir: Path) -> dict[str, Decimal]:
    """Project only canonically managed open positions from durable state."""

    quantities: dict[str, Decimal] = {}
    state_paths = (
        output_dir / "live" / "event_driven_execution_state.json",
        output_dir / "live" / "generated_strategy_live_state.json",
    )
    for path in state_paths:
        if not path.is_file():
            continue
        try:
            payload = dict(read_json(path))
        except (OSError, TypeError, ValueError):
            continue
        positions = payload.get("positions") or {}
        rows = positions.values() if isinstance(positions, Mapping) else positions
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            market = str(row.get("market") or "").upper()
            symbol = market.split("-", 1)[0]
            quantity = _decimal_or_zero(row.get("quantity"))
            if symbol and quantity > 0:
                quantities[symbol] = quantities.get(symbol, Decimal("0")) + quantity
    return quantities


async def live_account_health(
    settings: Settings,
    *,
    markets: tuple[str, ...] = ("ETH-EUR",),
    adopt_inventory: bool = False,
) -> dict[str, Any]:
    """Perform a sanitized private read without creating an order intent."""

    from core.contracts import ExecutionBlocked, ReconciliationRequired
    from execution.execution import build_live_client

    normalized = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-").replace("_", "-")
            for market in markets
            if str(market).strip()
        )
    )
    if not normalized or any(
        len(market.split("-")) != 2 or not market.endswith("-EUR")
        for market in normalized
    ):
        raise ValueError("account health accepts one or more EUR spot markets")

    provider = settings.providers
    configuration_checks = {
        "trade_credentials_present": provider.has_trade_credentials(),
        "operator_id_present": bool(provider.bitvavo_operator_id),
        "trade_scope_safe": not provider.unsafe_trade_scope(),
        "withdrawal_permission": bool(
            provider.bitvavo_withdrawal_permission
        ),
        "ip_whitelist_confirmed": bool(
            provider.bitvavo_ip_whitelist_confirmed
        ),
    }
    failures: list[str] = []
    if not configuration_checks["trade_credentials_present"]:
        failures.append("TRADE_CREDENTIALS_MISSING")
    if not configuration_checks["operator_id_present"]:
        failures.append("BITVAVO_OPERATOR_ID_MISSING")
    if not configuration_checks["trade_scope_safe"]:
        failures.append("UNSAFE_API_SCOPE")
    if configuration_checks["withdrawal_permission"]:
        failures.append("WITHDRAWAL_PERMISSION_ENABLED")
    if not configuration_checks["ip_whitelist_confirmed"]:
        failures.append("IP_WHITELIST_NOT_CONFIRMED")

    projection = {
        "eur_available": "0",
        "non_eur_holdings": [],
        "non_eur_holding_count": 0,
        "positive_symbol_count": 0,
    }
    raw_balances: list[dict[str, Any]] = []
    reconciliation: dict[str, Any] | None = None
    account_fees: dict[str, str] | None = None
    market_fee_rates: dict[str, dict[str, str]] = {}
    clock_sync: dict[str, Any] = {"status": "NOT_CHECKED"}
    transaction_reconciliation: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "complete": False,
        "external_eur_delta": "0",
        "matched_external_transactions": 0,
        "evidence": [],
    }
    private_requests = 0
    public_valuation_requests = 0
    if not failures:
        timeout = aiohttp.ClientTimeout(
            total=min(20.0, settings.market_data.request_timeout_seconds)
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                client = build_live_client(
                    settings,
                    session=session,
                    ledger_path=(
                        settings.paths.checkpoints_dir
                        / "live_execution.jsonl"
                    ),
                )
                local_before_ms = int(datetime.now(UTC).timestamp() * 1_000)
                venue_time_ms = await client.server_time_ms()
                local_after_ms = int(datetime.now(UTC).timestamp() * 1_000)
                public_valuation_requests += 1
                local_midpoint_ms = (local_before_ms + local_after_ms) // 2
                clock_drift_ms = venue_time_ms - local_midpoint_ms
                maximum_clock_drift_ms = min(
                    5_000,
                    max(1_000, client.access_window_ms // 2),
                )
                clock_sync = {
                    "status": (
                        "READY"
                        if abs(clock_drift_ms) <= maximum_clock_drift_ms
                        else "CLOCK_DRIFT_EXCEEDS_SAFE_WINDOW"
                    ),
                    "drift_ms": clock_drift_ms,
                    "maximum_drift_ms": maximum_clock_drift_ms,
                }
                if clock_sync["status"] != "READY":
                    failures.append("CLOCK_DRIFT_EXCEEDS_SAFE_WINDOW")
                balances = await client.balances()
                raw_balances = balances
                private_requests += 1
                projection = _safe_balance_projection(balances)
                fee_rates = await client.account_fees()
                private_requests += 1
                account_fees = {
                    "maker_rate": str(fee_rates["maker"]),
                    "taker_rate": str(fee_rates["taker"]),
                    "source": "BITVAVO_PRIVATE_ACCOUNT_FEES",
                }
                for market in normalized:
                    rates = await client.account_fees(market)
                    private_requests += 1
                    market_fee_rates[market] = {
                        "maker_rate": str(rates["maker"]),
                        "taker_rate": str(rates["taker"]),
                        "source": "BITVAVO_PRIVATE_MARKET_FEES",
                    }
                continuity_path = (
                    settings.paths.output_dir
                    / "operations"
                    / "eur_cash_continuity.json"
                )
                continuity_state = (
                    dict(read_json(continuity_path))
                    if continuity_path.is_file()
                    else {}
                )
                accepted_at_raw = continuity_state.get("accepted_at")
                if accepted_at_raw:
                    try:
                        accepted_at = datetime.fromisoformat(
                            str(accepted_at_raw).replace("Z", "+00:00")
                        ).astimezone(UTC)
                    except ValueError:
                        transaction_reconciliation["status"] = (
                            "INVALID_LOCAL_BASELINE_TIMESTAMP"
                        )
                    else:
                        history_to_ms = int(
                            datetime.now(UTC).timestamp() * 1_000
                        )
                        try:
                            history_window = _transaction_history_window(
                                accepted_at,
                                current_ms=history_to_ms,
                                maximum_clock_drift_ms=(
                                    maximum_clock_drift_ms
                                ),
                            )
                        except ValueError:
                            transaction_reconciliation["status"] = (
                                "INVALID_LOCAL_BASELINE_TIMESTAMP"
                            )
                            failures.append(
                                "PRIVATE_ACCOUNT_RESPONSE_AMBIGUOUS"
                            )
                        else:
                            history = (
                                await client.transaction_history(
                                    from_date_ms=history_window[0],
                                    to_date_ms=history_window[1],
                                )
                                if history_window is not None
                                else {
                                    "items": [],
                                    "complete": True,
                                    "pages_read": 0,
                                    "total_pages": 0,
                                }
                            )
                            private_requests += int(history["pages_read"])
                            transaction_reconciliation = {
                                "status": (
                                    "READY"
                                    if history["complete"]
                                    else "INCOMPLETE_PAGINATION"
                                ),
                                "complete": bool(history["complete"]),
                                **_exchange_external_eur_delta(
                                    history["items"]
                                ),
                                "pages_read": history["pages_read"],
                                "total_pages": history["total_pages"],
                                "from_date": accepted_at.isoformat(),
                                "through_date": datetime.fromtimestamp(
                                    history_to_ms / 1_000,
                                    tz=UTC,
                                ).isoformat(),
                            }
                reconciled = await client.reconcile(markets=normalized)
                private_requests += 1 + len(normalized)
                reconciliation = asdict(reconciled)
                if not reconciled.healthy:
                    failures.append("RECONCILIATION_MISMATCH")
        except ExecutionBlocked:
            failures.append("PRIVATE_ACCOUNT_READ_BLOCKED")
        except ReconciliationRequired:
            failures.append("PRIVATE_ACCOUNT_RESPONSE_AMBIGUOUS")
        except (aiohttp.ClientError, TimeoutError):
            failures.append("PRIVATE_ACCOUNT_NETWORK_ERROR")

    estimated_equity = _decimal_or_zero(projection["eur_available"])
    valuations: list[dict[str, str | None]] = []
    valuation_status = "EUR_ONLY"
    if projection["non_eur_holdings"]:
        try:
            eur_prices = await _public_bitvavo_eur_prices(settings)
            public_valuation_requests += 1
            unpriced_symbols: list[str] = []
            for holding in projection["non_eur_holdings"]:
                symbol = str(holding["symbol"])
                quantity = _decimal_or_zero(holding["total"])
                market = f"{symbol}-EUR"
                price = eur_prices.get(market)
                value = quantity * price if price is not None else None
                if value is None:
                    unpriced_symbols.append(symbol)
                else:
                    estimated_equity += value
                valuations.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "quantity": str(quantity),
                        "price_eur": str(price) if price is not None else None,
                        "estimated_value_eur": (
                            str(value) if value is not None else None
                        ),
                    }
                )
            valuation_status = (
                "PARTIAL_MARK_TO_MARKET"
                if unpriced_symbols
                else "COMPLETE_MARK_TO_MARKET"
            )
        except (aiohttp.ClientError, TimeoutError, RuntimeError):
            valuation_status = "PUBLIC_PRICE_VALUATION_UNAVAILABLE"

    projection["portfolio_valuation"] = {
        "status": valuation_status,
        "estimated_total_equity_eur": str(estimated_equity),
        "holdings": valuations,
    }

    from core.account_inventory import (
        classify_account_inventory,
        expected_inventory_after_canonical_fills,
        grandfathered_inventory_from_expected,
        load_inventory_baseline,
        reconcile_inventory,
        write_inventory_baseline,
    )

    authority_active, authority, authority_failures = live_canary_authority(
        settings.paths.project_root
    )
    inventory: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "adopted": False,
        "authority_failures": list(authority_failures),
    }
    baseline: dict[str, Decimal] = {}
    if projection["non_eur_holding_count"]:
        if adopt_inventory:
            if (
                failures
                or not authority_active
                or authority is None
                or reconciliation is None
                or not reconciliation.get("healthy")
            ):
                failures.append("INVENTORY_BASELINE_ADOPTION_BLOCKED")
            else:
                baseline_artifact = write_inventory_baseline(
                    settings,
                    holdings=projection["non_eur_holdings"],
                    authority=authority,
                )
                inventory = {
                    "status": "ADOPTED_AND_RECONCILED",
                    "adopted": True,
                    "artifact": baseline_artifact["artifact"],
                    "inventory_hash": baseline_artifact[
                        "inventory_hash"
                    ],
                    "authority_failures": [],
                }
                baseline = {
                    str(row.get("symbol") or "").upper(): _decimal_or_zero(
                        row.get("total")
                    )
                    for row in projection["non_eur_holdings"]
                    if row.get("symbol")
                }
        elif authority_active and authority is not None and raw_balances:
            baseline, baseline_failures = load_inventory_baseline(
                settings,
                authority=authority,
            )
            baseline = expected_inventory_after_canonical_fills(
                settings,
                baseline,
            )
            inventory_reconciliation = reconcile_inventory(
                raw_balances,
                baseline,
                prices_eur={
                    str(row.get("symbol") or "").upper(): _decimal_or_zero(
                        row.get("price_eur")
                    )
                    for row in valuations
                    if row.get("symbol")
                },
                minimum_material_excess_eur=Decimal("1"),
            )
            inventory = {
                "status": (
                    "RECONCILED"
                    if not baseline_failures
                    and inventory_reconciliation["reconciled"]
                    else "REQUIRES_ADOPTION_OR_RECONCILIATION"
                ),
                "adopted": False,
                "baseline_failures": list(baseline_failures),
                "excess": inventory_reconciliation["excess"],
                "ignored_dust_excess": inventory_reconciliation[
                    "ignored_dust_excess"
                ],
                "minimum_material_excess_eur": inventory_reconciliation[
                    "minimum_material_excess_eur"
                ],
                "missing_or_reduced": inventory_reconciliation[
                    "missing_or_reduced"
                ],
                "authority_failures": [],
            }

    managed_quantities = _managed_inventory_quantities(
        settings.paths.output_dir
    )
    classification_baseline = grandfathered_inventory_from_expected(
        baseline,
        managed_quantities,
    )
    inventory_classification = classify_account_inventory(
        raw_balances,
        baseline=classification_baseline,
        managed_quantities=managed_quantities,
    )
    classification_by_symbol = {
        row["symbol"]: row for row in inventory_classification
    }
    for valuation in valuations:
        classification = classification_by_symbol.get(
            str(valuation.get("symbol") or "")
        )
        if classification:
            valuation["position_classification"] = classification[
                "classification"
            ]
            valuation["autonomous_exit_authority"] = classification[
                "autonomous_exit_authority"
            ]
            valuation["autonomous_exit_authority_quantity"] = classification[
                "autonomous_exit_authority_quantity"
            ]
            valuation["external_quantity"] = classification[
                "external_quantity"
            ]
            valuation["mixed_ownership"] = classification["mixed_ownership"]
            valuation["autonomous_exit_authority_applies_to_full_quantity"] = (
                classification["external_quantity"] == "0"
            )

    safe_minimums: dict[str, str] = {}
    try:
        venue_metadata = await _public_bitvavo_market_metadata(settings)
        public_valuation_requests += 1
        for market in normalized:
            minimum = _decimal_or_zero(
                (venue_metadata.get(market) or {}).get(
                    "minOrderInQuoteAsset"
                )
            )
            if minimum > 0:
                safe_minimums[market] = str(
                    (minimum * Decimal("1.15")).quantize(
                        Decimal("0.01")
                    )
                )
    except (aiohttp.ClientError, TimeoutError, RuntimeError):
        pass
    minimum_order_eur = min(
        (_decimal_or_zero(value) for value in safe_minimums.values()),
        default=Decimal(str(settings.execution.maximum_live_order_eur)),
    )
    eur_available = _decimal_or_zero(projection["eur_available"])
    entry_blockers: list[str] = []
    wallet_asset_exposure = sum(
        (
            _decimal_or_zero(row.get("estimated_value_eur"))
            for row in valuations
        ),
        Decimal("0"),
    )
    largest_holding = max(
        valuations,
        key=lambda row: _decimal_or_zero(row.get("estimated_value_eur")),
        default=None,
    )
    crypto_exposure_fraction = (
        wallet_asset_exposure / estimated_equity
        if estimated_equity > 0
        else Decimal("0")
    )
    largest_asset_fraction = (
        _decimal_or_zero((largest_holding or {}).get("estimated_value_eur"))
        / estimated_equity
        if estimated_equity > 0
        else Decimal("0")
    )
    maximum_crypto_fraction = Decimal(
        str(settings.autonomous_live.maximum_total_crypto_exposure_pct)
    ) / Decimal("100")
    inventory_risk_override = evaluate_inventory_risk_override(
        settings,
        raw_balances,
    )
    projection["portfolio_heat"] = {
        "managed_strategy_positions": sum(
            row["classification"] == "MANAGED_POSITION"
            for row in inventory_classification
        ),
        "inventory_classification": inventory_classification,
        "total_wallet_asset_exposure_eur": str(wallet_asset_exposure),
        "total_wallet_crypto_exposure_fraction": str(
            crypto_exposure_fraction
        ),
        "largest_asset": (largest_holding or {}).get("symbol"),
        "largest_asset_exposure_fraction": str(largest_asset_fraction),
        "maximum_total_crypto_exposure_fraction": str(
            maximum_crypto_fraction
        ),
        "protective_exits_allowed": True,
        "inventory_risk_override": inventory_risk_override,
    }
    if valuation_status not in {"EUR_ONLY", "COMPLETE_MARK_TO_MARKET"}:
        entry_blockers.append("TOTAL_WALLET_EXPOSURE_VALUATION_INCOMPLETE")
    if (
        crypto_exposure_fraction > maximum_crypto_fraction
        and not inventory_risk_override["active"]
    ):
        entry_blockers.append("TOTAL_WALLET_CRYPTO_EXPOSURE_LIMIT")
    from core.cash_balance_guard import evaluate_eur_cash_continuity

    cash_continuity = evaluate_eur_cash_continuity(
        settings,
        current_eur_available=eur_available,
        exchange_external_cash_delta_eur=transaction_reconciliation[
            "external_eur_delta"
        ],
        exchange_history_complete=bool(
            transaction_reconciliation["complete"]
        ),
    )
    if cash_continuity["new_entries_blocked"]:
        entry_blockers.append("UNEXPLAINED_EUR_BALANCE_CHANGE")
    if eur_available < minimum_order_eur:
        # Lack of quote cash prevents a new BUY, but it is not a broken
        # reconciliation.  Monitoring and protective SELLs must stay active.
        entry_blockers.append("INSUFFICIENT_AVAILABLE_EUR_FOR_MINIMUM_ORDER")
    if (
        projection["non_eur_holding_count"]
        and inventory["status"]
        not in {"ADOPTED_AND_RECONCILED", "RECONCILED"}
    ):
        failures.append("EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION")

    inventory_classification = list(
        projection.get("portfolio_heat", {}).get("inventory_classification")
        or []
    )
    managed_position_present = any(
        str(row.get("autonomous_exit_authority") or "").lower() == "true"
        and _decimal_or_zero(row.get("managed_quantity")) > 0
        for row in inventory_classification
    )
    scoped_protection_failures = [
        failure
        for failure in failures
        if failure != "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"
    ]
    managed_position_protection_eligible = bool(
        managed_position_present
        and reconciliation is not None
        and reconciliation.get("healthy") is True
        and not scoped_protection_failures
    )

    payload = {
        "schema_version": "live_account_health_v1",
        "checked_at": utc_iso(),
        "status": "READY" if not failures else "BLOCKED",
        "configuration": configuration_checks,
        "markets_checked": list(normalized),
        "account": projection,
        "account_fee_rates": account_fees,
        "market_fee_rates": market_fee_rates,
        "clock_sync": clock_sync,
        "transaction_reconciliation": transaction_reconciliation,
        "preexisting_inventory": inventory,
        "reconciliation": reconciliation,
        "minimum_required_eur": str(minimum_order_eur),
        "venue_safe_minimums_eur": safe_minimums,
        "failures": list(dict.fromkeys(failures)),
        "entry_allowed": not failures and not entry_blockers,
        "entry_blockers": list(dict.fromkeys(entry_blockers)),
        "eur_cash_continuity": cash_continuity,
        "risk_reduction_allowed": not failures,
        "managed_position_protection_eligible": (
            managed_position_protection_eligible
        ),
        "managed_position_protection_scope": (
            "CANONICAL_MANAGED_POSITIONS_ONLY"
            if managed_position_protection_eligible
            else "NONE"
        ),
        "external_inventory_actions_allowed": False,
        "scoped_protection_failures": scoped_protection_failures,
        "privacy_and_authority": {
            "private_exchange_requests": private_requests,
            "public_valuation_requests": public_valuation_requests,
            "orders_generated": 0,
            "orders_submitted": 0,
            "withdrawals_attempted": 0,
            "secrets_serialized": False,
            "account_identifiers_serialized": False,
        },
    }
    from core.execution_authority import build_execution_authority_matrix

    payload["execution_authority"] = build_execution_authority_matrix(
        settings,
        account_health=payload,
    )
    path = (
        settings.paths.output_dir
        / "operations"
        / "live_account_health.json"
    )
    atomic_write_json(path, payload)
    from core.daily_profit_target import update_daily_profit_target

    payload["daily_profit_target"] = update_daily_profit_target(
        settings,
        estimated_equity_eur=estimated_equity,
        valuation_status=valuation_status,
    )
    atomic_write_json(path, payload)
    payload["artifact"] = str(path)
    return payload


async def live_asset_preflight(
    settings: Settings,
    *,
    markets: tuple[str, ...],
) -> dict[str, Any]:
    """Audit assets without balances, private requests, signals or orders."""

    normalized = tuple(
        dict.fromkeys(
            str(market).strip().upper().replace("/", "-").replace("_", "-")
            for market in markets
            if str(market).strip()
        )
    )
    if not normalized:
        raise ValueError("asset preflight requires at least one market")
    if any(
        len(market.split("-")) != 2 or not market.endswith("-EUR")
        for market in normalized
    ):
        raise ValueError("asset preflight accepts EUR spot markets only")

    venue = await _public_bitvavo_market_metadata(settings)
    eligibility_path = (
        settings.paths.output_dir
        / "universe"
        / "top50_eligibility.json"
    )
    eligibility = (
        dict(read_json(eligibility_path))
        if eligibility_path.is_file()
        else {"rows": []}
    )
    eligibility_by_market = {
        str(row.get("eur_spot_market") or "").upper(): dict(row)
        for row in eligibility.get("rows") or []
        if row.get("eur_spot_market")
    }
    registry = LiveStrategyApprovalRegistry(
        settings.paths.project_root
        / "config"
        / "live_strategy_approvals.yaml"
    ).load()
    approved_by_market: dict[str, list[dict[str, Any]]] = {}
    for approval in registry.values():
        for market in approval.approved_markets:
            approved_by_market.setdefault(market, []).append(
                asdict(approval)
            )
    authority_active, authority, authority_failures = live_canary_authority(
        settings.paths.project_root
    )
    playbook_path = (
        settings.paths.project_root
        / "config"
        / "live_playbook_authority.json"
    )
    playbook_authority = (
        dict(read_json(playbook_path)) if playbook_path.is_file() else {}
    )
    service_path = (
        settings.paths.output_dir
        / "live"
        / "autonomous_live_authority.json"
    )
    service_authority = (
        dict(read_json(service_path)) if service_path.is_file() else {}
    )
    approved_playbooks_by_market: dict[str, list[dict[str, Any]]] = {}
    if playbook_authority.get("active") is True:
        for raw in playbook_authority.get("approved_playbooks") or []:
            approval = dict(raw)
            if approval.get("active") is not True:
                continue
            for approved_market in approval.get("markets") or []:
                approved_playbooks_by_market.setdefault(
                    str(approved_market).upper(), []
                ).append(approval)
    market_exceptions = _market_exceptions(settings)

    rows: list[dict[str, Any]] = []
    for market in normalized:
        public = _public_market_projection(venue.get(market))
        universe = eligibility_by_market.get(market)
        approvals = approved_by_market.get(market, [])
        playbook_approvals = approved_playbooks_by_market.get(market, [])
        exception = market_exceptions.get(market)
        blockers: list[str] = []
        if not public["venue_available"]:
            blockers.append("VENUE_MARKET_NOT_TRADING")
        if universe is None and not (
            exception and exception.get("allow_outside_top50")
        ):
            blockers.append("NOT_IN_POINT_IN_TIME_TOP50_EXECUTION_UNIVERSE")
        elif (
            universe is not None
            and universe.get("execution_eligibility") != "LIVE_ELIGIBLE"
            and exception is None
        ):
            blockers.append(
                str(universe.get("execution_reason") or "EXECUTION_INELIGIBLE")
            )
        if not approvals and not playbook_approvals:
            blockers.append("NO_APPROVED_STRATEGY_DNA_FOR_MARKET")
        exact_matching_authority = bool(
            authority_active
            and authority
            and str(authority.get("market") or "").upper() == market
            and any(
                approval["strategy_dna_hash"]
                == authority.get("strategy_dna")
                for approval in approvals
            )
        )
        playbook_matching_authority = bool(
            playbook_approvals
            and service_authority.get("active") is True
            and market in set(service_authority.get("markets") or [])
        )
        matching_authority = (
            exact_matching_authority or playbook_matching_authority
        )
        if not matching_authority:
            blockers.append("NO_MATCHING_ACTIVE_OPERATOR_AUTHORITY")
        # A held balance is not an entry/exit signal.  Private balance access is
        # intentionally skipped until all public, identity and strategy gates
        # are green.
        blockers.append("NO_NATURAL_APPROVED_STRATEGY_SIGNAL")
        rows.append(
            {
                "market": market,
                **public,
                "point_in_time_universe": universe,
                "operator_market_exception": (
                    {
                        "active": True,
                        "allow_outside_top50": bool(
                            exception.get("allow_outside_top50")
                        ),
                        "spot_only": True,
                        "maximum_order_eur": float(
                            exception["maximum_order_eur"]
                        ),
                        "maximum_total_exposure_eur": float(
                            exception["maximum_total_exposure_eur"]
                        ),
                        "requires_approved_strategy_dna": True,
                        "requires_natural_signal": True,
                        "approval_reference": exception.get(
                            "approval_reference"
                        ),
                    }
                    if exception
                    else {"active": False}
                ),
                "approved_strategies": [
                    {
                        "strategy_id": approval["strategy_id"],
                        "strategy_dna_hash": approval[
                            "strategy_dna_hash"
                        ],
                        "timeframe": approval["timeframe"],
                        "maximum_order_eur": approval[
                            "maximum_order_eur"
                        ],
                        "maximum_total_exposure_eur": approval[
                            "maximum_total_exposure_eur"
                        ],
                        "autoscale": approval["autoscale"],
                    }
                    for approval in approvals
                ],
                "approved_playbooks": [
                    {
                        "playbook_id": approval.get("playbook_id"),
                        "playbook_dna": approval.get("playbook_dna"),
                        "execution_timeframes": approval.get(
                            "execution_timeframes"
                        ),
                        "maximum_order_eur": approval.get(
                            "maximum_order_eur",
                            playbook_authority.get("maximum_order_eur", 10),
                        ),
                        "autoscale": False,
                    }
                    for approval in playbook_approvals
                ],
                "matching_authority_type": (
                    "EXACT_STRATEGY"
                    if exact_matching_authority
                    else "EVENT_PLAYBOOK"
                    if playbook_matching_authority
                    else None
                ),
                "matching_operator_authority": matching_authority,
                "live_trade_ready": not blockers,
                "blockers": blockers,
            }
        )

    payload = {
        "schema_version": "live_asset_preflight_v1",
        "generated_at": utc_iso(),
        "markets": rows,
        "summary": {
            "requested_market_count": len(rows),
            "venue_trading_count": sum(
                row["venue_available"] for row in rows
            ),
            "live_trade_ready_count": sum(
                row["live_trade_ready"] for row in rows
            ),
            "active_operator_authority": authority_active,
            "authority_failures": authority_failures,
        },
        "privacy_and_authority": {
            "private_exchange_requests": 0,
            "balance_requests": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
            "withdrawals_attempted": 0,
            "secrets_serialized": False,
            "possession_is_not_a_strategy_signal": True,
            "preflight_hash": stable_hash(
                {
                    "markets": normalized,
                    "eligibility_snapshot": eligibility.get(
                        "source_snapshot_hash"
                    ),
                    "authority_active": authority_active,
                    "market_exceptions": sorted(market_exceptions),
                },
                length=64,
            ),
        },
        "status": (
            "READY"
            if rows and all(row["live_trade_ready"] for row in rows)
            else "BLOCKED"
        ),
    }
    path = (
        settings.paths.output_dir
        / "operations"
        / "live_asset_preflight.json"
    )
    atomic_write_json(path, payload)
    payload["artifact"] = str(path)
    return payload


__all__ = [
    "_safe_balance_projection",
    "live_account_health",
    "live_asset_preflight",
]
