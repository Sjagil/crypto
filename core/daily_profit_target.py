"""Non-binding, capital-scaled daily profit target reporting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from config.settings import Settings
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    stable_hash,
    utc_iso,
)

ZERO = Decimal("0")


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def daily_profit_target_path(settings: Settings) -> Path:
    return settings.paths.output_dir / "portfolio" / "daily_profit_target.json"


def capital_flow_ledger_path(settings: Settings) -> Path:
    return settings.paths.output_dir / "portfolio" / "external_capital_flows.jsonl"


def confirmed_capital_flow_for_date(settings: Settings, date_utc: str) -> Decimal:
    """Sum operator-confirmed deposits and withdrawals for one UTC date."""

    path = capital_flow_ledger_path(settings)
    if not path.is_file():
        return ZERO
    total = ZERO
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(record, dict)
            or record.get("operator_confirmed") is not True
            or record.get("date_utc") != date_utc
        ):
            continue
        amount = _decimal(record.get("amount_eur"))
        if amount is not None:
            total += amount
    return total


def record_external_capital_flow(
    settings: Settings,
    *,
    amount_eur: Decimal | float | str,
    reason: str,
    effective_at: datetime | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Record an operator-confirmed deposit/withdrawal without touching execution."""

    amount = _decimal(amount_eur)
    if amount is None or amount == ZERO:
        raise ValueError("CAPITAL_FLOW_AMOUNT_MUST_BE_NON_ZERO")
    selected_reason = str(reason).strip().upper()
    if selected_reason not in {"DEPOSIT", "WITHDRAWAL", "TRANSFER", "CORRECTION"}:
        raise ValueError("CAPITAL_FLOW_REASON_INVALID")
    when = (effective_at or datetime.now(UTC)).astimezone(UTC)
    flow_id = stable_hash(
        ["external-capital-flow", when.isoformat(), str(amount), selected_reason, note],
        length=40,
    )
    payload = {
        "schema_version": "external_capital_flow_v1",
        "flow_id": flow_id,
        "recorded_at": utc_iso(),
        "effective_at": utc_iso(when),
        "date_utc": when.date().isoformat(),
        "amount_eur": str(amount),
        "reason": selected_reason,
        "note": str(note or "")[:160],
        "operator_confirmed": True,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    path = capital_flow_ledger_path(settings)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(existing, dict) and existing.get("flow_id") == flow_id:
                return {
                    **existing,
                    "status": "SKIPPED_DUPLICATE",
                    "artifact": str(path),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
    append_jsonl(path, payload)
    return {**payload, "status": "RECORDED", "artifact": str(path)}


def update_daily_profit_target(
    settings: Settings,
    *,
    estimated_equity_eur: Decimal | float | str | None,
    observed_at: datetime | None = None,
    valuation_status: str = "MARK_TO_MARKET_ESTIMATE",
) -> dict[str, Any]:
    """Persist target progress without changing signals, sizing or execution."""

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    target_settings = settings.daily_profit_target
    path = daily_profit_target_path(settings)
    previous = dict(read_json(path)) if path.is_file() else {}
    equity = _decimal(estimated_equity_eur)
    reference_equity = Decimal(str(target_settings.reference_equity_eur))
    reference_target = Decimal(str(target_settings.reference_target_eur))
    target_fraction = reference_target / reference_equity
    same_day = previous.get("date_utc") == now.date().isoformat()
    previous_start = _decimal(previous.get("day_start_equity_eur"))
    day_start = (
        previous_start
        if same_day and previous_start is not None and previous_start > ZERO
        else equity
    )
    confirmed_flow = confirmed_capital_flow_for_date(
        settings,
        now.date().isoformat(),
    )
    risk_adjusted_start = (
        day_start + confirmed_flow if day_start is not None else None
    )

    if not target_settings.enabled:
        status = "DISABLED"
        scaled_target = None
        pnl = None
        progress = None
    elif equity is None or day_start is None or day_start <= ZERO:
        status = "VALUATION_PENDING"
        scaled_target = None
        pnl = None
        progress = None
    else:
        scaled_target = day_start * target_fraction
        pnl = equity - day_start
        progress = pnl / scaled_target if scaled_target > ZERO else ZERO
        status = "TARGET_REACHED" if pnl >= scaled_target else "IN_PROGRESS"

    adjusted_pnl = (
        equity - risk_adjusted_start
        if equity is not None
        and risk_adjusted_start is not None
        else None
    )
    if (
        target_settings.enabled
        and adjusted_pnl is not None
        and scaled_target is not None
    ):
        progress = (
            adjusted_pnl / scaled_target
            if scaled_target > ZERO
            else ZERO
        )
        status = (
            "TARGET_REACHED"
            if adjusted_pnl >= scaled_target
            else "IN_PROGRESS"
        )

    payload = {
        "schema_version": "daily_profit_target_v1",
        "status": status,
        "date_utc": now.date().isoformat(),
        "updated_at": utc_iso(now),
        "enabled": target_settings.enabled,
        "reference_equity_eur": str(reference_equity),
        "reference_target_eur": str(reference_target),
        "target_fraction": str(target_fraction),
        "day_start_equity_eur": str(day_start) if day_start is not None else None,
        "risk_adjusted_day_start_equity_eur": (
            str(risk_adjusted_start) if risk_adjusted_start is not None else None
        ),
        "external_capital_flow_eur": str(confirmed_flow),
        "current_estimated_equity_eur": str(equity) if equity is not None else None,
        "scaled_daily_target_eur": (
            str(scaled_target) if scaled_target is not None else None
        ),
        "mark_to_market_pnl_eur": str(pnl) if pnl is not None else None,
        "cash_flow_adjusted_pnl_eur": (
            str(adjusted_pnl) if adjusted_pnl is not None else None
        ),
        "progress_fraction": str(progress) if progress is not None else None,
        "valuation_status": valuation_status,
        "non_binding": True,
        "force_trades": False,
        "override_risk_limits": False,
        "may_increase_position_size": False,
        "may_relax_stops": False,
        "may_bypass_kill_switch": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(path, payload)
    return {**payload, "artifact": str(path)}


__all__ = [
    "capital_flow_ledger_path",
    "confirmed_capital_flow_for_date",
    "daily_profit_target_path",
    "record_external_capital_flow",
    "update_daily_profit_target",
]
