from __future__ import annotations

import asyncio
import sys
import threading
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from core.live_asset_preflight import live_account_health
from core.live_capital import (
    APPROVAL_PHRASE as LEVEL_2_APPROVAL_PHRASE,
    AUTOSCALE as LEVEL_2_AUTOSCALE,
    CAPITAL_LEVEL,
    MAXIMUM_MANAGED_POSITIONS,
    MAXIMUM_NEW_ORDERS_PER_DAY,
    MAXIMUM_ORDER_EUR,
    MAXIMUM_RISK_PER_TRADE_EUR,
    MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR,
    managed_live_portfolio,
    submit_level_2_buy_atomically,
)
from execution.execution import (
    LivePreflight,
    build_live_client,
    minimum_protectable_entry_notional,
    quantity_is_protectable_at_stop,
)
from portfolio.buy_chain import canonicalize_approved_buy_order
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA = "swing_layer_live_canary_v1"
AUTHORITY_PATH = ROOT / "config" / "live_swing_layer_authority.json"
DEFAULT_MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        result = Decimal(str(default if value in (None, "") else value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)
    return result if result.is_finite() else Decimal(default)


def _settings() -> Settings:
    return Settings.load(
        env_file=ROOT / ".env",
        create_directories=True,
    )


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    failure: list[BaseException] = []

    def target() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            failure.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


def _defaults() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "active": False,
        "capital_level": CAPITAL_LEVEL,
        "markets": list(DEFAULT_MARKETS),
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR),
        "maximum_open_positions": MAXIMUM_MANAGED_POSITIONS,
        "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        "maximum_risk_per_trade_eur": str(MAXIMUM_RISK_PER_TRADE_EUR),
        "autoscale": bool(LEVEL_2_AUTOSCALE),
        "spot_only": True,
        "margin": False,
        "leverage": False,
        "shorting": False,
        "withdrawals": False,
        "approval_phrase_stored": False,
    }

def _authority() -> dict[str, Any]:
    payload = _defaults()
    if AUTHORITY_PATH.is_file():
        try:
            raw = read_json(AUTHORITY_PATH)
            if isinstance(raw, Mapping):
                payload.update(dict(raw))
        except (OSError, TypeError, ValueError):
            pass
    payload["schema_version"] = SCHEMA
    payload["approval_phrase_stored"] = False
    payload["autoscale"] = False
    payload["spot_only"] = True
    payload["margin"] = False
    payload["leverage"] = False
    payload["shorting"] = False
    payload["withdrawals"] = False
    payload["markets"] = list(
        dict.fromkeys(
            str(market).strip().upper()
            for market in payload.get("markets") or DEFAULT_MARKETS
            if str(market).strip()
        )
    )
    return payload


def _persist_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {**_defaults(), **dict(payload)}
    body["approval_phrase_stored"] = False
    body.pop("approval", None)
    body.pop("approval_phrase", None)
    AUTHORITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(AUTHORITY_PATH, body)
    return body


def _persisted_health() -> dict[str, Any]:
    path = ROOT / "output" / "operations" / "live_account_health.json"
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _health(markets: tuple[str, ...]) -> dict[str, Any]:
    try:
        payload = _run(
            live_account_health(
                _settings(),
                markets=markets,
                adopt_inventory=False,
            )
        )
    except (ExecutionBlocked, ReconciliationRequired) as exc:
        return {
            "schema_version": "live_account_health_v1",
            "status": "BLOCKED",
            "failures": [f"{type(exc).__name__}:{str(exc)[:240]}"],
            "entry_allowed": False,
            "entry_blockers": ["PRIVATE_ACCOUNT_READ_BLOCKED"],
            "risk_reduction_allowed": False,
            "reconciliation": {
                "healthy": False,
                "reason_codes": ["REMOTE_RECONCILIATION_FAILED"],
            },
            "privacy_and_authority": {
                "orders_generated": 0,
                "orders_submitted": 0,
                "secrets_serialized": False,
            },
        }
    return dict(payload)


def _portfolio_raw() -> dict[str, Any]:
    try:
        return dict(managed_live_portfolio(_settings()))
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "managed_live_portfolio_v4",
            "status": "BLOCKED",
            "failures": [f"{type(exc).__name__}:{str(exc)[:240]}"],
            "positions": [],
            "managed_position_count": 0,
            "managed_exposure_eur": "0",
        }


def swing_layer_portfolio() -> dict[str, Any]:
    raw = _portfolio_raw()
    return {
        "schema_version": SCHEMA,
        "status": "READY" if raw.get("status") == "READY" else "BLOCKED",
        "positions": {
            f"{row.get('source')}:{row.get('identity')}": row
            for row in raw.get("positions") or []
            if isinstance(row, Mapping)
        },
        "position_count": int(raw.get("managed_position_count") or 0),
        "managed_exposure_eur": str(raw.get("managed_exposure_eur") or "0"),
        "canonical_state_hash": stable_hash(raw, length=64),
        "evidence_gap_count": len(raw.get("failures") or []),
        "canonical": raw,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _equity_and_cash(health: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    account = dict(health.get("account") or {})
    valuation = dict(account.get("portfolio_valuation") or {})
    equity = _decimal(valuation.get("estimated_total_equity_eur"))
    cash = _decimal(account.get("eur_available"))
    return equity, cash


def swing_layer_account_snapshot(
    markets: tuple[str, ...] = DEFAULT_MARKETS,
) -> dict[str, Any]:
    normalized = tuple(
        dict.fromkeys(
            str(value).strip().upper()
            for value in markets
            if str(value).strip()
        )
    )
    health = _health(normalized)
    portfolio = _portfolio_raw()
    equity, cash = _equity_and_cash(health)
    failures = list(health.get("failures") or [])
    if portfolio.get("status") != "READY":
        failures.extend(
            portfolio.get("failures") or ["MANAGED_PORTFOLIO_NOT_READY"]
        )
    return {
        "status": (
            "READY"
            if health.get("status") == "READY" and not failures
            else "BLOCKED"
        ),
        "failures": list(dict.fromkeys(str(value) for value in failures)),
        "entry_allowed": bool(health.get("entry_allowed")) and not failures,
        "entry_blockers": list(health.get("entry_blockers") or []),
        "equity_eur": str(equity),
        "cash_eur": str(cash),
        "exposure_eur": str(portfolio.get("managed_exposure_eur") or "0"),
        "risk_reduction_allowed": (
            bool(health.get("risk_reduction_allowed")) and not failures
        ),
        "managed_position_protection_eligible": bool(
            health.get("managed_position_protection_eligible")
        ),
        "canonical_health": health,
        "canonical_portfolio": portfolio,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def reconcile_swing_layer_live(
    markets: tuple[str, ...] = DEFAULT_MARKETS,
) -> dict[str, Any]:
    snapshot = swing_layer_account_snapshot(markets)
    health = dict(snapshot.get("canonical_health") or {})
    reconciliation = dict(health.get("reconciliation") or {})
    healthy = bool(
        snapshot.get("status") == "READY"
        and reconciliation.get("healthy") is True
    )
    reasons = list(reconciliation.get("reason_codes") or [])
    if not healthy and not reasons:
        reasons.append("REMOTE_RECONCILIATION_FAILED")
    portfolio = swing_layer_portfolio()
    return {
        "status": "READY" if healthy else "BLOCKED",
        "healthy": healthy,
        "reason_codes": reasons or ["RECONCILED"],
        "local_open_orders": int(
            reconciliation.get("local_open_orders") or 0
        ),
        "remote_open_orders": int(
            reconciliation.get("remote_open_orders") or 0
        ),
        "portfolio": portfolio,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def swing_layer_authority_status() -> dict[str, Any]:
    authority = _authority()
    health = _persisted_health()
    reconciliation = dict(health.get("reconciliation") or {})
    portfolio = swing_layer_portfolio()
    environment_ready = bool(
        authority.get("active") is True
        and health.get("status") == "READY"
        and reconciliation.get("healthy") is True
    )
    return {
        **authority,
        "execution_environment_ready": environment_ready,
        "state_status": (
            "READY"
            if environment_ready
            else "RECONCILIATION_REQUIRED"
            if authority.get("active")
            else "DISABLED"
        ),
        "managed_positions": portfolio["position_count"],
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def approve_swing_layer_canary(
    *,
    markets: tuple[str, ...],
    approval: str,
) -> dict[str, Any]:
    settings = _settings()
    if approval.strip() != LEVEL_2_APPROVAL_PHRASE:
        raise PermissionError("Level-2 swing canary approval phrase mismatch")
    normalized = tuple(
        dict.fromkeys(
            str(value).strip().upper()
            for value in markets
            if str(value).strip()
        )
    )
    if not normalized:
        raise ValueError("swing canary requires at least one market")
    for market in normalized:
        if settings.shariah.eligibility(market).status.value != "ALLOWED":
            raise PermissionError(f"market is not eligible: {market}")
    body = _defaults()
    body.update(
        {
            "active": True,
            "activated_at": utc_iso(),
            "markets": list(normalized),
            "approval_phrase_stored": False,
            "operator_approval_reference": "explicit_capital_level_2_swing_canary",
        }
    )
    return {
        **_persist_authority(body),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def deactivate_swing_layer_canary() -> dict[str, Any]:
    body = _authority()
    body.update(
        {
            "active": False,
            "deactivated_at": utc_iso(),
            "approval_phrase_stored": False,
        }
    )
    return {
        **_persist_authority(body),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _validate_intent(
    intent: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    authority = _authority()
    blockers: list[str] = []
    market = str(intent.get("market") or "").upper()
    if authority.get("active") is not True:
        blockers.append("SWING_CANARY_AUTHORITY_INACTIVE")
    if str(intent.get("authority") or "").upper() != "LIVE":
        blockers.append("INTENT_AUTHORITY_NOT_LIVE")
    if market not in set(authority.get("markets") or []):
        blockers.append("MARKET_OUTSIDE_SWING_AUTHORITY")
    notional = _decimal(intent.get("notional_eur"))
    if notional <= 0:
        blockers.append("INVALID_NOTIONAL")
    if notional > _decimal(authority.get("maximum_order_eur"), "10"):
        blockers.append("ORDER_CAP_EXCEEDED")
    if _decimal(intent.get("net_edge_bps")) <= 0:
        blockers.append("NON_POSITIVE_NET_EDGE")
    stop_pct = _decimal(intent.get("stop_pct"))
    if stop_pct <= 0 or stop_pct >= 1:
        blockers.append("INVALID_STOP_DISTANCE")
    expires = str(intent.get("expires_at") or "")
    try:
        expiry = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        if expiry.astimezone(UTC) <= datetime.now(UTC):
            blockers.append("INTENT_EXPIRED")
    except ValueError:
        blockers.append("INVALID_INTENT_EXPIRY")
    return authority, blockers


def _live_preflight(
    intent: Mapping[str, Any],
) -> tuple[dict[str, Any], Any | None, dict[str, Any], dict[str, Any]]:
    authority, blockers = _validate_intent(intent)
    market = str(intent.get("market") or "").upper()
    if blockers:
        return (
            {
                "status": "BLOCKED",
                "accepted": False,
                "blockers": blockers,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
            None,
            {},
            {},
        )

    snapshot = swing_layer_account_snapshot((market,))
    health = dict(snapshot.get("canonical_health") or {})
    portfolio = dict(snapshot.get("canonical_portfolio") or {})
    reconciliation = dict(health.get("reconciliation") or {})
    if (
        snapshot.get("status") != "READY"
        or snapshot.get("entry_allowed") is not True
    ):
        blockers.extend(snapshot.get("failures") or [])
        blockers.extend(snapshot.get("entry_blockers") or [])

    current_exposure = _decimal(portfolio.get("managed_exposure_eur"))
    requested = _decimal(intent.get("notional_eur"))
    if current_exposure + requested > _decimal(
        authority.get("maximum_total_exposure_eur"), str(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR)
    ):
        blockers.append("TOTAL_EXPOSURE_CAP_EXCEEDED")
    if int(portfolio.get("managed_position_count") or 0) >= int(
        authority.get("maximum_open_positions") or MAXIMUM_MANAGED_POSITIONS
    ):
        blockers.append("POSITION_CAP_EXCEEDED")

    settings = _settings()
    matrix = dict(health.get("execution_authority") or {})
    kill = dict(matrix.get("kill_switch") or {})
    preflight = LivePreflight.evaluate(
        settings,
        markets=(market,),
        strategy_status=ResearchStatus.PAPER_CANDIDATE,
        data_healthy=health.get("status") == "READY",
        risk_manager_healthy=not health.get("failures"),
        exchange_healthy=health.get("status") == "READY",
        reconciliation_healthy=reconciliation.get("healthy") is True,
        kill_switch_active=kill.get("active") is True,
        canary_exception_approved=True,
        operator_canary_authorized=authority.get("active") is True,
        cap_limits={
            "capital_level": CAPITAL_LEVEL,
            "max_order_eur": authority.get("maximum_order_eur", str(MAXIMUM_ORDER_EUR)),
            "max_exposure_eur": authority.get(
                "maximum_total_exposure_eur", str(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR)
            ),
            "max_positions": authority.get("maximum_open_positions", MAXIMUM_MANAGED_POSITIONS),
            "max_new_orders_per_day": authority.get(
                "maximum_new_orders_per_day", MAXIMUM_NEW_ORDERS_PER_DAY
            ),
        },
    )
    blockers.extend(preflight.failures)
    blockers = list(dict.fromkeys(str(value) for value in blockers if value))
    result = {
        "status": (
            "READY"
            if not blockers and preflight.passed
            else "BLOCKED"
        ),
        "accepted": bool(not blockers and preflight.passed),
        "blockers": blockers,
        "account_status": snapshot.get("status"),
        "reconciliation_status": (
            "READY"
            if reconciliation.get("healthy") is True
            else "BLOCKED"
        ),
        "authority_active": authority.get("active") is True,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return result, preflight.capability, health, portfolio


def swing_layer_live_preflight(intent: Mapping[str, Any]) -> dict[str, Any]:
    result, _capability, _health_payload, _portfolio = _live_preflight(intent)
    return result


async def _public_price(session, market: str) -> Decimal:
    import aiohttp

    async with session.get(
        "https://api.bitvavo.com/v2/ticker/price",
        params={"market": market},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as response:
        if response.status >= 400:
            raise ExecutionBlocked("public Bitvavo price unavailable")
        payload = await response.json(content_type=None)
    price = _decimal(
        dict(payload).get("price")
        if isinstance(payload, Mapping)
        else None
    )
    if price <= 0:
        raise ExecutionBlocked("public Bitvavo price invalid")
    return price


def _managed_quantity(
    health: Mapping[str, Any],
    market: str,
) -> Decimal:
    symbol = market.split("-", 1)[0]
    account = dict(health.get("account") or {})
    heat = dict(account.get("portfolio_heat") or {})
    rows = heat.get("inventory_classification") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("symbol") or "").upper() == symbol:
            return _decimal(row.get("managed_quantity"))
    return Decimal("0")


async def _submit_buy_async(
    intent: Mapping[str, Any],
    capability,
    health: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    import aiohttp

    settings = _settings()
    authority = _authority()
    market = str(intent["market"]).upper()
    notional = _decimal(intent["notional_eur"])
    stop_pct = _decimal(intent["stop_pct"])
    risk_eur = notional * stop_pct
    if risk_eur <= 0 or risk_eur > _decimal(
        authority.get("maximum_risk_per_trade_eur"), "1"
    ):
        raise ExecutionBlocked(
            "swing intent exceeds risk-per-trade authority"
        )

    equity, _cash = _equity_and_cash(health)
    if equity <= 0:
        raise ExecutionBlocked("reconciled account equity is unavailable")

    metadata = dict(intent.get("metadata") or {})
    confidence = _decimal(metadata.get("signal_confidence"), "0.5")
    confidence = min(Decimal("1"), max(Decimal("0.01"), confidence))
    net_edge = _decimal(intent.get("net_edge_bps")) / Decimal("10000")
    current_quantity = _managed_quantity(health, market)

    async with aiohttp.ClientSession() as session:
        client = build_live_client(
            settings,
            session=session,
            ledger_path=(
                settings.paths.checkpoints_dir
                / "live_execution.jsonl"
            ),
        )
        price = await _public_price(session, market)
        rules = await client.execution_market_rules(market)
        stop_price = rules.price(
            price * (Decimal("1") - stop_pct)
        )
        minimum_notional = minimum_protectable_entry_notional(
            entry_price=price,
            stop_price=stop_price,
            rules=rules,
        )
        if notional < minimum_notional:
            raise ExecutionBlocked(
                "swing canary notional is too small for a safely "
                "protectable native stop"
            )
        quantity = rules.amount(notional / price)
        if quantity <= 0:
            raise ExecutionBlocked(
                "swing canary quantity rounded to zero"
            )

        order = OrderIntent(
            intent_id=str(
                intent.get("intent_id")
                or stable_hash(intent, length=24)
            ),
            idempotency_key=stable_hash(
                [
                    "SWING_LAYER_BUY",
                    str(intent.get("intent_id") or ""),
                ],
                length=32,
            ),
            market=market,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=OrderTimeInForce.GTC,
            strategy_id=str(
                intent.get("strategy")
                or "crypto_ai_swing_layer"
            ),
            strategy_dna_hash=(
                str(metadata.get("strategy_dna_hash"))
                if metadata.get("strategy_dna_hash")
                else None
            ),
            signal_id=str(intent.get("intent_id") or ""),
            maximum_notional_eur=notional,
            reason_codes=("SWING_LAYER_CANONICAL_BUY",),
        )
        plan = canonicalize_approved_buy_order(
            settings,
            order,
            mark_price=price,
            current_quantity=current_quantity,
            equity_eur=equity,
            approved_risk_eur=risk_eur,
            expected_net_edge=net_edge,
            confidence=confidence,
            family="CRYPTO_AI_SWING_LAYER",
            evidence_id=stable_hash(intent, length=64),
            policy_version="swing_layer_live_bridge_v1",
            account_state=health,
            portfolio_state=portfolio,
            horizon_seconds=int(
                metadata.get("horizon_seconds")
                or 24 * 3600
            ),
        )

        async def submit_with_fresh_portfolio(fresh_portfolio):
            return await client.submit_order(
                plan.order,
                capability=capability,
                estimated_price=price,
                reconciled_owned_quantity=current_quantity,
                reconciled_total_exposure_eur=_decimal(
                    fresh_portfolio.get("managed_exposure_eur")
                ),
                reconciled_open_positions=int(
                    fresh_portfolio.get(
                        "managed_position_count"
                    )
                    or 0
                ),
                exchange_minimum_order_eur=(
                    rules.minimum_order_value_eur
                ),
                canonical_chain=plan.chain,
            )

        approved, reason, fresh_portfolio, result = (
            await submit_level_2_buy_atomically(
                settings,
                requested_notional_eur=notional,
                submit_order=submit_with_fresh_portfolio,
            )
        )
        if not approved or result is None:
            return {
                "accepted": False,
                "status": "BLOCKED",
                "reason_code": reason,
                "canonical_portfolio": fresh_portfolio,
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        payload = (
            result.model_dump(mode="json")
            if hasattr(result, "model_dump")
            else dict(result)
            if isinstance(result, Mapping)
            else {"result": str(result)}
        )
        filled_quantity = _decimal(
            payload.get("filled_quantity")
            or payload.get("filledAmount")
            or payload.get("quantity")
        )
        protective: dict[str, Any] = {
            "status": "NOT_REQUIRED_UNTIL_FILL",
            "native_stop_required": True,
        }
        submitted_count = 1

        if filled_quantity > 0:
            protected_quantity = rules.amount(filled_quantity)
            if not quantity_is_protectable_at_stop(
                quantity=protected_quantity,
                stop_price=stop_price,
                rules=rules,
            ):
                raise ReconciliationRequired(
                    "filled swing quantity cannot support "
                    "required native stop"
                )
            stop_intent = OrderIntent(
                intent_id=stable_hash(
                    ["SWING_LAYER_STOP", plan.order.intent_id],
                    length=32,
                ),
                idempotency_key=stable_hash(
                    [
                        "SWING_LAYER_STOP",
                        plan.order.idempotency_key,
                    ],
                    length=32,
                ),
                market=market,
                side=OrderSide.SELL,
                order_type=OrderType.STOP_LOSS,
                quantity=protected_quantity,
                trigger_price=stop_price,
                trigger_reference="bestBid",
                time_in_force=OrderTimeInForce.GTC,
                strategy_id=plan.order.strategy_id,
                strategy_dna_hash=plan.order.strategy_dna_hash,
                signal_id=plan.order.signal_id,
                reason_codes=("NATIVE_PROTECTIVE_STOP",),
            )
            stop_result = await client.submit_order(
                stop_intent,
                capability=capability,
                estimated_price=stop_price,
                reconciled_owned_quantity=filled_quantity,
                reconciled_total_exposure_eur=_decimal(
                    fresh_portfolio.get("managed_exposure_eur")
                ),
                reconciled_open_positions=int(
                    fresh_portfolio.get(
                        "managed_position_count"
                    )
                    or 0
                ),
                exchange_minimum_order_eur=(
                    rules.minimum_order_value_eur
                ),
                canonical_chain=None,
            )
            protective = (
                stop_result.model_dump(mode="json")
                if hasattr(stop_result, "model_dump")
                else dict(stop_result)
                if isinstance(stop_result, Mapping)
                else {"result": str(stop_result)}
            )
            protective["native_stop_required"] = True
            submitted_count = 2

        return {
            "accepted": True,
            "status": "SUBMITTED",
            "reason_code": reason,
            "order": payload,
            "native_protective_stop": protective,
            "orders_generated": submitted_count,
            "orders_submitted": submitted_count,
        }


def submit_swing_layer_buy(
    intent: Mapping[str, Any],
    *,
    execute: bool = False,
) -> dict[str, Any]:
    preflight, capability, health, portfolio = _live_preflight(intent)
    if preflight.get("accepted") is not True or capability is None:
        return preflight
    if not execute:
        return {
            **preflight,
            "status": "READY_NOT_SUBMITTED",
            "accepted": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    try:
        return _run(
            _submit_buy_async(
                intent,
                capability,
                health,
                portfolio,
            )
        )
    except (ExecutionBlocked, ReconciliationRequired) as exc:
        return {
            "accepted": False,
            "status": "BLOCKED",
            "reason_code": type(exc).__name__,
            "reason": str(exc)[:300],
            "orders_generated": 0,
            "orders_submitted": 0,
        }


async def _submit_exit_async(
    *,
    market: str,
    reason: str,
    requested_quantity: Decimal | None,
) -> dict[str, Any]:
    import aiohttp

    settings = _settings()
    authority = _authority()
    health = _health((market,))
    if (
        health.get("risk_reduction_allowed") is not True
        and health.get(
            "managed_position_protection_eligible"
        )
        is not True
    ):
        raise ExecutionBlocked(
            "canonical risk-reduction authority is not ready"
        )
    managed_quantity = _managed_quantity(health, market)
    quantity = min(
        managed_quantity,
        (
            requested_quantity
            if requested_quantity is not None
            else managed_quantity
        ),
    )
    if quantity <= 0:
        raise ExecutionBlocked(
            "no canonically managed quantity is available to exit"
        )

    reconciliation = dict(health.get("reconciliation") or {})
    portfolio = _portfolio_raw()
    preflight = LivePreflight.evaluate(
        settings,
        markets=(market,),
        strategy_status=ResearchStatus.PAPER_CANDIDATE,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=health.get("status") == "READY",
        reconciliation_healthy=(
            reconciliation.get("healthy") is True
        ),
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=(
            authority.get("active") is True
        ),
        cap_limits={
            "capital_level": CAPITAL_LEVEL,
            "max_order_eur": authority.get(
                "maximum_order_eur", "10"
            ),
            "max_exposure_eur": authority.get(
                "maximum_total_exposure_eur", str(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR)
            ),
            "max_positions": authority.get(
                "maximum_open_positions", 1
            ),
            "max_new_orders_per_day": authority.get(
                "maximum_new_orders_per_day", MAXIMUM_NEW_ORDERS_PER_DAY
            ),
        },
    )
    if not preflight.passed or preflight.capability is None:
        raise ExecutionBlocked(
            "canonical exit preflight blocked: "
            + ",".join(preflight.failures)
        )

    async with aiohttp.ClientSession() as session:
        client = build_live_client(
            settings,
            session=session,
            ledger_path=(
                settings.paths.checkpoints_dir
                / "live_execution.jsonl"
            ),
        )
        price = await _public_price(session, market)
        exit_intent = OrderIntent(
            intent_id=stable_hash(
                ["SWING_LAYER_EXIT", market, reason, utc_iso()],
                length=32,
            ),
            idempotency_key=stable_hash(
                [
                    "SWING_LAYER_EXIT",
                    market,
                    reason,
                    str(quantity),
                ],
                length=32,
            ),
            market=market,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            strategy_id="crypto_ai_swing_layer",
            reason_codes=(
                "RISK_REDUCING_EXIT",
                str(reason)[:80],
            ),
        )
        result = await client.submit_order(
            exit_intent,
            capability=preflight.capability,
            estimated_price=price,
            reconciled_owned_quantity=managed_quantity,
            reconciled_total_exposure_eur=_decimal(
                portfolio.get("managed_exposure_eur")
            ),
            reconciled_open_positions=int(
                portfolio.get("managed_position_count") or 0
            ),
            canonical_chain=None,
        )
        payload = (
            result.model_dump(mode="json")
            if hasattr(result, "model_dump")
            else dict(result)
            if isinstance(result, Mapping)
            else {"result": str(result)}
        )
        return {
            "accepted": True,
            "status": "SUBMITTED",
            "order": payload,
            "orders_generated": 1,
            "orders_submitted": 1,
        }


def submit_swing_layer_exit(
    *,
    market: str,
    reason: str,
    requested_quantity: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    normalized = str(market).strip().upper()
    quantity = (
        _decimal(requested_quantity)
        if requested_quantity not in (None, "")
        else None
    )
    if not execute:
        health = _health((normalized,))
        managed = _managed_quantity(health, normalized)
        allowed = bool(
            managed > 0
            and (
                health.get("risk_reduction_allowed") is True
                or health.get(
                    "managed_position_protection_eligible"
                )
                is True
            )
        )
        return {
            "accepted": allowed,
            "status": (
                "READY_NOT_SUBMITTED"
                if allowed
                else "BLOCKED"
            ),
            "managed_quantity": str(managed),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    try:
        return _run(
            _submit_exit_async(
                market=normalized,
                reason=reason,
                requested_quantity=quantity,
            )
        )
    except (ExecutionBlocked, ReconciliationRequired) as exc:
        return {
            "accepted": False,
            "status": "BLOCKED",
            "reason_code": type(exc).__name__,
            "reason": str(exc)[:300],
            "orders_generated": 0,
            "orders_submitted": 0,
        }


__all__ = [
    "approve_swing_layer_canary",
    "deactivate_swing_layer_canary",
    "reconcile_swing_layer_live",
    "submit_swing_layer_buy",
    "submit_swing_layer_exit",
    "swing_layer_account_snapshot",
    "swing_layer_authority_status",
    "swing_layer_live_preflight",
    "swing_layer_portfolio",
]
