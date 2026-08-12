"""Unambiguous signal, simulated-execution and actual-live evidence layers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from core.decision_attribution import build_decision_execution_attribution
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {}


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return result if result.is_finite() else Decimal("0")


def _sum(rows: list[Mapping[str, Any]], field: str) -> Decimal:
    return sum((_decimal(row.get(field)) for row in rows), Decimal("0"))


def build_execution_evidence_layers(root: Path) -> dict[str, Any]:
    """Build a read-only evidence artifact without equating unlike PnLs."""

    root = root.resolve()
    operations = root / "output" / "operations"
    live = root / "output" / "live"
    audit_path = operations / "daily_opportunity_audit.json"
    performance_path = live / "strategy_performance.json"
    audit = _read(audit_path)
    live_performance = _read(performance_path)
    paper = dict(audit.get("paper_execution_evidence") or {})
    paper_rows = [
        dict(row)
        for row in dict(paper.get("by_playbook") or {}).values()
        if isinstance(row, Mapping)
    ]
    live_rows = [
        dict(row)
        for row in live_performance.get("strategies") or []
        if isinstance(row, Mapping)
    ]
    active_live_rows = [
        row
        for row in live_rows
        if int(row.get("closed_trade_count") or 0)
        or int(row.get("open_trade_count") or 0)
        or _decimal(row.get("fees_paid_eur")) != 0
    ]
    live_integrity = str(
        live_performance.get("integrity_status") or "NOT_AVAILABLE"
    )
    resolved = int(audit.get("resolved_counterfactual_count") or 0)
    paper_round_trips = int(paper.get("closed_round_trips") or 0)
    decision_attribution = build_decision_execution_attribution(root)
    live_round_trips = sum(
        int(row.get("closed_trade_count") or 0) for row in live_rows
    )

    payload: dict[str, Any] = {
        "schema_version": "execution_evidence_layers_v1",
        "generated_at": utc_iso(),
        "theoretical_signal_pnl": {
            "status": (
                "COUNTERFACTUAL_EVIDENCE_AVAILABLE"
                if resolved
                else "INSUFFICIENT_RESOLVED_COUNTERFACTUALS"
            ),
            "resolved_episode_count": resolved,
            "false_breakout_rate": audit.get("false_breakout_rate"),
            "mfe_distribution_pct": audit.get("mfe_distribution_pct"),
            "mae_distribution_pct": audit.get("mae_distribution_pct"),
            "gross_or_net_pnl_eur": None,
            "aggregation_note": (
                "No EUR PnL is invented: rejected-gate counterfactuals are "
                "multi-label and are not exchange fills."
            ),
            "source": str(audit_path),
        },
        "simulated_execution_pnl": {
            "status": (
                "VERIFIED_CLOSED_PAPER_ROUND_TRIPS"
                if paper_round_trips
                else "NO_CLOSED_PAPER_ROUND_TRIPS"
            ),
            "closed_round_trips": paper_round_trips,
            "gross_expectancy_eur": paper.get("paper_gross_expectancy_eur"),
            "net_expectancy_eur": paper.get("paper_net_expectancy_eur"),
            "net_pnl_eur": str(_sum(paper_rows, "paper_net_pnl_eur")),
            "fees_eur": str(_sum(paper_rows, "paper_fees_eur")),
            "by_playbook": paper.get("by_playbook", {}),
            "by_playbook_dna": paper.get("by_playbook_dna", {}),
            "source": paper.get("source"),
        },
        "actual_live_pnl": {
            "status": (
                "VERIFIED_CANONICAL_LIVE_ACCOUNTING"
                if live_integrity == "PASSED"
                else "LIVE_ACCOUNTING_NOT_VERIFIED"
            ),
            "integrity_status": live_integrity,
            "closed_round_trips": live_round_trips,
            "open_positions": sum(
                int(row.get("open_trade_count") or 0) for row in live_rows
            ),
            "realised_pnl_eur": str(_sum(live_rows, "realised_pnl_eur")),
            "unrealised_pnl_eur": str(
                _sum(live_rows, "unrealised_pnl_eur")
            ),
            "net_pnl_eur": str(_sum(live_rows, "net_pnl_eur")),
            "fees_eur": str(_sum(live_rows, "fees_paid_eur")),
            "live_strategy_account_count": len(live_rows),
            "active_strategy_count": len(active_live_rows),
            "active_strategies": active_live_rows,
            "source": str(performance_path),
        },
        "comparison_policy": {
            "layers_are_not_interchangeable": True,
            "paper_is_not_reported_as_live": True,
            "counterfactual_is_not_reported_as_fill": True,
            "live_requires_canonical_exchange_fill_attribution": True,
        },
        "decision_execution_attribution": {
            "status": decision_attribution.get("status"),
            "trade_count": decision_attribution.get("trade_count"),
            "closed_round_trips": decision_attribution.get("closed_round_trips"),
            "open_positions": decision_attribution.get("open_positions"),
            "decision_price_mapped_count": decision_attribution.get(
                "decision_price_mapped_count"
            ),
            "decision_price_mapping_ratio": decision_attribution.get(
                "decision_price_mapping_ratio"
            ),
            "artifact": decision_attribution.get("artifact")
            or str(
                root
                / "output"
                / "operations"
                / "decision_execution_attribution.json"
            ),
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    payload["evidence_hash"] = stable_hash(payload, length=64)
    output = operations / "execution_evidence_layers.json"
    atomic_write_json(output, payload)
    payload["artifact"] = str(output)
    return payload


__all__ = ["build_execution_evidence_layers"]
