"""Fresh active-portfolio constraints for the canonical live execution stack.

This module does not own broker authority.  It turns the private-account-derived
Round-34 live plan into additional entry/exit constraints consumed by the
existing canonical execution engines.  When the Round-35 orchestrator is not
enabled, the pre-existing execution policy is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from config.settings import Settings
from utils.common import read_json

ZERO = Decimal("0")
DEFAULT_MAXIMUM_PLAN_AGE_SECONDS = 20 * 60


def _decimal(value: Any) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return ZERO
    return selected if selected.is_finite() else ZERO


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _paths(settings: Settings) -> tuple[Path, Path]:
    root = settings.paths.output_dir / "portfolio"
    return (
        root / "active_live_plan.json",
        root / "active_live_orchestrator.json",
    )


def active_live_orchestrator_enabled(settings: Settings) -> bool:
    _, marker = _paths(settings)
    if not marker.is_file():
        return False
    try:
        payload = dict(read_json(marker))
    except (OSError, TypeError, ValueError):
        return False
    return payload.get("enabled") is True


def load_active_live_plan(
    settings: Settings,
    *,
    maximum_age_seconds: int = DEFAULT_MAXIMUM_PLAN_AGE_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    plan_path, _ = _paths(settings)
    orchestrator_enabled = active_live_orchestrator_enabled(settings)
    if not orchestrator_enabled:
        return {
            "status": "NOT_APPLIED",
            "reason": "ACTIVE_LIVE_ORCHESTRATOR_DISABLED",
            "orchestrator_enabled": False,
            "fresh": False,
            "directives": {},
        }
    missing_status = "BLOCKED"
    if not plan_path.is_file():
        return {
            "status": missing_status,
            "reason": "ACTIVE_LIVE_PLAN_MISSING",
            "orchestrator_enabled": orchestrator_enabled,
            "fresh": False,
            "directives": {},
        }
    try:
        payload = dict(read_json(plan_path))
    except (OSError, TypeError, ValueError):
        return {
            "status": missing_status,
            "reason": "ACTIVE_LIVE_PLAN_UNREADABLE",
            "orchestrator_enabled": orchestrator_enabled,
            "fresh": False,
            "directives": {},
        }
    generated_at = _parse_time(payload.get("generated_at"))
    observed = (now or datetime.now(UTC)).astimezone(UTC)
    if generated_at is None:
        return {
            "status": missing_status,
            "reason": "ACTIVE_LIVE_PLAN_TIMESTAMP_INVALID",
            "orchestrator_enabled": orchestrator_enabled,
            "fresh": False,
            "directives": {},
        }
    age_seconds = (observed - generated_at).total_seconds()
    if age_seconds < -5 or age_seconds > maximum_age_seconds:
        return {
            "status": missing_status,
            "reason": "ACTIVE_LIVE_PLAN_STALE",
            "orchestrator_enabled": orchestrator_enabled,
            "fresh": False,
            "age_seconds": age_seconds,
            "generated_at": generated_at.isoformat(),
            "directives": {},
        }
    if payload.get("mode") != "live-plan":
        return {
            "status": missing_status,
            "reason": "ACTIVE_LIVE_PLAN_MODE_INVALID",
            "orchestrator_enabled": orchestrator_enabled,
            "fresh": False,
            "age_seconds": age_seconds,
            "directives": {},
        }
    directives: dict[str, dict[str, Any]] = {}
    for raw in payload.get("decisions") or []:
        if not isinstance(raw, Mapping):
            continue
        market = str(raw.get("market") or "").strip().upper()
        if market:
            directives[market] = dict(raw)
    return {
        "status": "READY",
        "reason": "ACTIVE_LIVE_PLAN_READY",
        "orchestrator_enabled": orchestrator_enabled,
        "fresh": True,
        "age_seconds": age_seconds,
        "generated_at": generated_at.isoformat(),
        "plan_status": payload.get("status"),
        "directives": directives,
    }


def active_entry_budget_eur(
    settings: Settings,
    market: str,
    *,
    default_budget_eur: Decimal,
) -> tuple[Decimal, str, dict[str, Any]]:
    """Return an additional EUR cap for a new canonical live BUY."""

    default_budget = max(ZERO, Decimal(default_budget_eur))
    plan = load_active_live_plan(settings)
    if plan["status"] == "NOT_APPLIED":
        return default_budget, "ACTIVE_PORTFOLIO_NOT_APPLIED", plan
    if plan["status"] != "READY":
        return ZERO, str(plan["reason"]), plan

    directive = dict(plan["directives"].get(str(market).upper()) or {})
    if not directive:
        return ZERO, "ACTIVE_PORTFOLIO_MARKET_NOT_TARGETED", plan
    action = str(directive.get("action") or "").upper()
    if directive.get("live_plan_eligible") is not True:
        return ZERO, "ACTIVE_PORTFOLIO_ENTRY_NOT_ELIGIBLE", plan
    if action not in {"ENTRY", "ADD"}:
        return ZERO, f"ACTIVE_PORTFOLIO_ACTION_{action or 'UNKNOWN'}", plan

    delta = _decimal(directive.get("delta_notional_eur"))
    if delta <= ZERO:
        return ZERO, "ACTIVE_PORTFOLIO_NO_POSITIVE_DELTA", plan
    return min(default_budget, delta), "ACTIVE_PORTFOLIO_ENTRY_BUDGET_APPLIED", plan


def active_exit_quantity(
    settings: Settings,
    market: str,
    *,
    current_quantity: Decimal,
    mark_price: Decimal,
) -> tuple[Decimal, str | None, dict[str, Any]]:
    """Return a risk-reducing SELL quantity requested by the fresh live plan."""

    current = max(ZERO, Decimal(current_quantity))
    price = max(ZERO, Decimal(mark_price))
    plan = load_active_live_plan(settings)
    if plan["status"] != "READY" or current <= ZERO or price <= ZERO:
        return ZERO, None, plan
    directive = dict(plan["directives"].get(str(market).upper()) or {})
    if not directive or directive.get("live_plan_eligible") is not True:
        return ZERO, None, plan

    action = str(directive.get("action") or "").upper()
    if action == "FULL_EXIT":
        return current, action, plan
    if action != "REDUCE":
        return ZERO, None, plan

    target_notional = max(ZERO, _decimal(directive.get("target_notional_eur")))
    target_quantity = target_notional / price
    requested = max(ZERO, current - target_quantity)
    return min(current, requested), action, plan


__all__ = [
    "DEFAULT_MAXIMUM_PLAN_AGE_SECONDS",
    "active_entry_budget_eur",
    "active_exit_quantity",
    "active_live_orchestrator_enabled",
    "load_active_live_plan",
]
