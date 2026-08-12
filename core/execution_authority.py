"""Explicit execution-authority semantics for the canonical live engine.

This module is observational and deterministic.  It does not place orders and
does not replace the per-market data, cost, strategy-DNA or risk gates.  Its
purpose is to prevent an entry pause or kill switch from being confused with
the authority to protect or reduce an already open position.
"""

from __future__ import annotations

from typing import Any, Mapping

from config.settings import Settings
from risk.risk_manager import KillSwitch
from utils.common import read_json, utc_iso


def _read(path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_execution_authority_matrix(
    settings: Settings,
    *,
    account_health: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a four-authority summary from canonical persisted state."""

    live = settings.paths.output_dir / "live"
    operations = settings.paths.output_dir / "operations"
    control = _read(live / "autonomous_live_state.json")
    service_authority = _read(live / "autonomous_live_authority.json")
    runtime = _read(live / "autonomous_live_status.json")
    heartbeat = _read(live / "heartbeat.json")
    orderflow = _read(operations / "orderflow_stream_health.json")
    kill_switch = KillSwitch(
        settings.paths.checkpoints_dir / "kill_switch.json"
    )

    control_state = str(control.get("state") or "DISABLED").upper()
    service_authority_active = service_authority.get("active") is True
    public_stream = dict(runtime.get("websocket") or {})
    private_stream = dict(runtime.get("private_account_websocket") or {})
    heartbeat_public = dict(heartbeat.get("websocket") or {})
    heartbeat_private = dict(
        heartbeat.get("private_account_websocket") or {}
    )
    if public_stream.get("state") != "CONNECTED" and heartbeat_public:
        public_stream = heartbeat_public
    if not private_stream.get("ready_for_new_entries") and heartbeat_private:
        private_stream = heartbeat_private
    public_ready = public_stream.get("state") == "CONNECTED"
    private_ready = bool(private_stream.get("ready_for_new_entries"))
    orderflow_provider = dict(orderflow.get("provider") or {})
    orderflow_ready = (
        orderflow.get("status") == "HEALTHY"
        and orderflow.get("ledger_integrity_status") == "PASSED"
        and orderflow_provider.get("state") == "CONNECTED"
        and not orderflow.get("reason_codes")
    )

    entry_blockers = list(account_health.get("entry_blockers") or [])
    entry_blockers.extend(account_health.get("failures") or [])
    if control_state != "ENABLED":
        entry_blockers.append(f"CONTROL_STATE_{control_state}")
    if not service_authority_active:
        entry_blockers.append("LIVE_SERVICE_AUTHORITY_INACTIVE")
    if kill_switch.active:
        entry_blockers.append("KILL_SWITCH_ACTIVE")
    if not public_ready:
        entry_blockers.append("PUBLIC_STREAM_NOT_READY")
    if not private_ready:
        entry_blockers.append("PRIVATE_STREAM_NOT_READY")
    entry_blockers = _unique(entry_blockers)

    risk_reduction_blockers: list[str] = []
    if account_health.get("risk_reduction_allowed") is not True:
        risk_reduction_blockers.extend(account_health.get("failures") or [])
        if not risk_reduction_blockers:
            risk_reduction_blockers.append("ACCOUNT_RISK_REDUCTION_NOT_READY")
    if not service_authority_active:
        risk_reduction_blockers.append("LIVE_SERVICE_AUTHORITY_INACTIVE")
    if not private_ready:
        risk_reduction_blockers.append("PRIVATE_EXECUTION_CHANNEL_NOT_READY")
    risk_reduction_blockers = _unique(risk_reduction_blockers)
    risk_reduction_allowed = not risk_reduction_blockers

    managed_protection_blockers: list[str] = []
    managed_protection_eligible = (
        account_health.get("managed_position_protection_eligible") is True
    )
    if not risk_reduction_allowed and not managed_protection_eligible:
        managed_protection_blockers.extend(
            account_health.get("scoped_protection_failures")
            or account_health.get("failures")
            or ["MANAGED_POSITION_PROTECTION_NOT_RECONCILED"]
        )
    if not service_authority_active:
        managed_protection_blockers.append("LIVE_SERVICE_AUTHORITY_INACTIVE")
    if not private_ready:
        managed_protection_blockers.append(
            "PRIVATE_EXECUTION_CHANNEL_NOT_READY"
        )
    managed_protection_blockers = _unique(managed_protection_blockers)
    managed_protection_allowed = not managed_protection_blockers and (
        risk_reduction_allowed or managed_protection_eligible
    )

    closed_candle_blockers = list(entry_blockers)
    orderflow_blockers = list(entry_blockers)
    if not orderflow_ready:
        orderflow_blockers.append("ORDERFLOW_STREAM_NOT_READY")

    kill_mode = "NONE"
    if kill_switch.active:
        kill_mode = "SOFT_ENTRY_KILL_PRESERVE_PROTECTION"

    return {
        "schema_version": "execution_authority_matrix_v1",
        "generated_at": utc_iso(),
        "summary_is_not_a_substitute_for_per_market_gates": True,
        "control_state": control_state,
        "kill_switch": {
            "active": kill_switch.active,
            "mode": kill_mode,
            "reason": kill_switch.reason or None,
            "entries_blocked": kill_switch.active,
            "risk_reducing_exits_preserved": True,
            "native_protection_preserved": True,
        },
        "entry_authority": {
            "allowed": not entry_blockers,
            "state": "ALLOWED" if not entry_blockers else "BLOCKED",
            "blockers": entry_blockers,
        },
        "exit_authority": {
            "allowed": risk_reduction_allowed,
            "state": "ALLOWED" if risk_reduction_allowed else "BLOCKED",
            "risk_increasing_actions": False,
            "blockers": risk_reduction_blockers,
        },
        "protection_authority": {
            "allowed": managed_protection_allowed,
            "state": (
                "ALLOWED" if managed_protection_allowed else "BLOCKED"
            ),
            "native_stop_management": True,
            "scope": (
                "ALL_CANONICALLY_RECONCILED_POSITIONS"
                if risk_reduction_allowed
                else "CANONICAL_MANAGED_POSITIONS_ONLY"
                if managed_protection_allowed
                else "NONE"
            ),
            "external_inventory_actions_allowed": False,
            "blockers": managed_protection_blockers,
        },
        "managed_position_exit_authority": {
            "allowed": managed_protection_allowed,
            "state": (
                "ALLOWED" if managed_protection_allowed else "BLOCKED"
            ),
            "scope": "CANONICAL_MANAGED_POSITIONS_ONLY",
            "risk_increasing_actions": False,
            "external_inventory_actions_allowed": False,
            "blockers": managed_protection_blockers,
        },
        "emergency_authority": {
            "state": "MANUAL_ONLY",
            "available": risk_reduction_allowed,
            "automatic_flatten_authorized": False,
            "blockers": risk_reduction_blockers,
        },
        "strategy_health_profiles": {
            "CLOSED_CANDLE_SWING": {
                "entry_allowed": not closed_candle_blockers,
                "orderflow_required": False,
                "orderflow_may_confirm_or_veto_when_available": True,
                "blockers": closed_candle_blockers,
            },
            "ORDERFLOW_TIMED_SWING": {
                "entry_allowed": not orderflow_blockers,
                "orderflow_required": True,
                "blockers": _unique(orderflow_blockers),
            },
        },
        "runtime_dependencies": {
            "public_stream_ready": public_ready,
            "private_stream_ready": private_ready,
            "orderflow_stream_ready": orderflow_ready,
            "service_authority_active": service_authority_active,
            "per_market_data_health_still_enforced": True,
            "reconciliation_still_enforced": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = ["build_execution_authority_matrix"]
