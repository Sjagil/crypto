"""Dual-read migration and explicit economic divergence reporting."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from execution.canonical_state import CanonicalExecutionState, ProtectionState
from utils.common import stable_hash, utc_iso

ZERO = Decimal("0")


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _legacy_positions(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    positions = snapshot.get("positions")
    return {
        str(market): dict(value)
        for market, value in dict(positions or {}).items()
        if isinstance(value, Mapping)
    }


def build_execution_divergence_report(
    legacy_snapshot: Mapping[str, Any],
    canonical: CanonicalExecutionState,
    *,
    quantity_tolerance: Decimal = Decimal("0.00000001"),
    money_tolerance_eur: Decimal = Decimal("0.01"),
) -> dict[str, Any]:
    """Compare legacy read state with canonical replay without mutating either."""

    legacy = _legacy_positions(legacy_snapshot)
    rows: list[dict[str, Any]] = []

    def add(
        market: str,
        field: str,
        legacy_value: Any,
        canonical_value: Any,
        classification: str,
        reason: str,
    ) -> None:
        rows.append(
            {
                "market": market,
                "field": field,
                "legacy_value": legacy_value,
                "canonical_value": canonical_value,
                "classification": classification,
                "reason": reason,
            }
        )

    markets = sorted(set(legacy) | set(canonical.positions))
    for market in markets:
        old = legacy.get(market, {})
        current = canonical.positions.get(market)
        old_quantity = _decimal(old.get("owned_quantity")) or ZERO
        new_quantity = current.quantity if current is not None else ZERO
        quantity_difference = abs(old_quantity - new_quantity)
        if quantity_difference > ZERO:
            add(
                market,
                "position_quantity",
                str(old_quantity),
                str(new_quantity),
                (
                    "ROUNDING_DIFFERENCE"
                    if quantity_difference <= quantity_tolerance
                    else "MISSING_HISTORICAL_EVIDENCE"
                    if current is not None and not current.cost_basis_known
                    else "REAL_DEFECT"
                ),
                "legacy and canonical quantities differ",
            )

        old_owner = str(old.get("strategy_id") or "") or None
        new_owner = current.strategy_id if current is not None else None
        if old_owner != new_owner:
            add(
                market,
                "strategy_ownership",
                old_owner,
                new_owner,
                (
                    "MISSING_HISTORICAL_EVIDENCE"
                    if current is not None and current.ownership_state.value == "UNKNOWN"
                    else "REAL_DEFECT"
                ),
                "ledger ownership and legacy position ownership disagree",
            )

        old_stop = _decimal(old.get("stop_price"))
        new_stop = current.effective_stop_price if current is not None else None
        if old_stop != new_stop:
            add(
                market,
                "effective_stop_price",
                str(old_stop) if old_stop is not None else None,
                str(new_stop) if new_stop is not None else None,
                (
                    "REAL_DEFECT"
                    if current is not None
                    and current.protection_state
                    is ProtectionState.CONFIRMED_ACTIVE
                    else "EXPECTED_SCHEMA_DIFFERENCE"
                ),
                "legacy stop is not the confirmed canonical exchange protection",
            )

        old_protected = _decimal(old.get("protected_quantity"))
        new_protected = current.protected_quantity if current is not None else ZERO
        if old_protected is None:
            if new_protected > ZERO:
                add(
                    market,
                    "protected_quantity",
                    None,
                    str(new_protected),
                    "EXPECTED_SCHEMA_DIFFERENCE",
                    "legacy schema did not represent confirmed protected quantity",
                )
        elif abs(old_protected - new_protected) > quantity_tolerance:
            add(
                market,
                "protected_quantity",
                str(old_protected),
                str(new_protected),
                "REAL_DEFECT",
                "economically protected quantity differs",
            )

        old_risk = _decimal(old.get("current_open_risk"))
        new_risk = current.open_risk_eur if current is not None else ZERO
        if new_risk is None:
            if old_risk is not None:
                add(
                    market,
                    "open_risk_eur",
                    str(old_risk),
                    None,
                    "MISSING_HISTORICAL_EVIDENCE",
                    "canonical cost/protection evidence is incomplete",
                )
        elif (
            current is not None
            and current.quantity <= ZERO
            and old_risk is None
            and market not in legacy
        ):
            # A replay-only closed shell carries no live economic risk. Its
            # absent legacy counterpart is not an economic divergence.
            pass
        elif old_risk is None or abs(old_risk - new_risk) > money_tolerance_eur:
            add(
                market,
                "open_risk_eur",
                str(old_risk) if old_risk is not None else None,
                str(new_risk),
                "REAL_DEFECT",
                "legacy risk is not based on actual fills and confirmed protection",
            )

        old_realized = _decimal(old.get("realized_pnl")) or ZERO
        new_realized = current.realized_pnl_eur if current is not None else ZERO
        if abs(old_realized - new_realized) > money_tolerance_eur:
            add(
                market,
                "realized_pnl_eur",
                str(old_realized),
                str(new_realized),
                (
                    "MISSING_HISTORICAL_EVIDENCE"
                    if current is not None and not current.realized_pnl_complete
                    else "REAL_DEFECT"
                ),
                "actual-fill realized PnL differs",
            )

    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    report = {
        "schema_version": "canonical_execution_divergence_v1",
        "generated_at": utc_iso(),
        "canonical_state_hash": canonical.state_hash,
        "legacy_state_source": str(
            legacy_snapshot.get("state_source") or "LEGACY_POSITION_TRACKER"
        ),
        "canonical_state_source": "CANONICAL_EXECUTION_LEDGER_REPLAY",
        "comparison_fields": [
            "position_quantity",
            "strategy_ownership",
            "effective_stop_price",
            "protected_quantity",
            "open_risk_eur",
            "realized_pnl_eur",
        ],
        "divergence_count": len(rows),
        "classification_counts": counts,
        "divergences": rows,
        "legacy_can_overwrite_canonical": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    report["report_hash"] = stable_hash(report)
    return report


def execution_state_migration_status() -> dict[str, Any]:
    return {
        "schema_version": "execution_state_migration_status_v1",
        "canonical_state": "CANONICAL_EXECUTION_LEDGER_REPLAY",
        "canonical_consumers": [
            "position_tracker",
            "supervisor_positions_read_model",
            "system_execution_audit",
            "opportunity_intelligence_live_economic_labels",
        ],
        "legacy_temporary_consumers": [
            "generated_strategy_runtime_state",
            "event_driven_runtime_state",
            "historical_dashboard_artifacts",
        ],
        "legacy_may_overwrite_canonical": False,
        "removal_gate": "DUAL_READ_EQUIVALENCE_AND_RESTART_REPLAY_REQUIRED",
    }


__all__ = [
    "build_execution_divergence_report",
    "execution_state_migration_status",
]
