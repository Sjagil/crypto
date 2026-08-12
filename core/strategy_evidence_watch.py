"""Read-only, sample-aware promotion and demotion evidence watch."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from utils.common import atomic_write_json, read_json, utc_iso

ZERO = Decimal("0")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {}


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _recommendation(
    *,
    paper_round_trips: int,
    paper_expectancy: Decimal | None,
    live_round_trips: int,
    live_expectancy: Decimal | None,
    live_profit_factor: float | None,
    integrity_ok: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not integrity_ok:
        return "DISABLE_NEW_ENTRIES_RECOMMENDED", [
            "LIVE_ACCOUNTING_INTEGRITY_NOT_PASSED"
        ]
    if (
        live_round_trips >= 30
        and live_expectancy is not None
        and live_expectancy < ZERO
        and live_profit_factor is not None
        and live_profit_factor < 0.80
    ):
        return "SHADOW_ONLY_REVIEW_RECOMMENDED", [
            "LIVE_30_MATERIALLY_NEGATIVE"
        ]
    if (
        live_round_trips >= 20
        and live_expectancy is not None
        and live_expectancy < ZERO
        and live_profit_factor is not None
        and live_profit_factor < 0.90
    ):
        return "PAPER_ONLY_REVIEW_RECOMMENDED", ["LIVE_20_NEGATIVE"]
    if (
        live_round_trips >= 10
        and live_expectancy is not None
        and live_expectancy < ZERO
        and live_profit_factor is not None
        and live_profit_factor < 0.75
    ):
        return "REDUCE_CANARY_REVIEW_RECOMMENDED", [
            "LIVE_10_EARLY_WARNING"
        ]
    if live_round_trips >= 10 and live_expectancy is not None:
        if live_expectancy >= ZERO:
            return "OPERATOR_PROMOTION_REVIEW_ELIGIBLE", [
                "LIVE_SAMPLE_NON_NEGATIVE"
            ]
        return "HOLD_CANARY", ["LIVE_SAMPLE_MIXED"]
    if paper_round_trips >= 20 and paper_expectancy is not None:
        if paper_expectancy < ZERO:
            return "PAPER_NEGATIVE_REVIEW_REQUIRED", [
                "PAPER_20_PLUS_NEGATIVE"
            ]
        return "PRIORITIZE_LIVE_CANARY_EVIDENCE", [
            "PAPER_20_PLUS_POSITIVE"
        ]
    if paper_round_trips and paper_expectancy is not None:
        if paper_expectancy > ZERO:
            return "PRIORITIZE_MORE_EVIDENCE", [
                "PAPER_EARLY_POSITIVE",
                "SAMPLE_BELOW_20",
            ]
        reasons.extend(["PAPER_EARLY_NEGATIVE", "SAMPLE_BELOW_20"])
        return "HOLD_SMALL_CANARY_AND_MEASURE", reasons
    return "COLLECT_PAPER_AND_SHADOW", ["NO_CLOSED_PAPER_EVIDENCE"]


def build_strategy_evidence_watch(
    root: Path,
    *,
    execution_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare unlike evidence explicitly without changing live authority."""

    root = root.resolve()
    authority = _read(root / "config" / "live_playbook_authority.json")
    evidence = dict(
        execution_evidence
        or _read(
            root
            / "output"
            / "operations"
            / "execution_evidence_layers.json"
        )
    )
    live_performance = _read(
        root / "output" / "live" / "strategy_performance.json"
    )
    paper_by_playbook = dict(
        dict(evidence.get("simulated_execution_pnl") or {}).get(
            "by_playbook"
        )
        or {}
    )
    paper_by_playbook_dna = dict(
        dict(evidence.get("simulated_execution_pnl") or {}).get(
            "by_playbook_dna"
        )
        or {}
    )
    live_rows = {
        str(row.get("strategy_id") or ""): dict(row)
        for row in live_performance.get("strategies") or []
        if isinstance(row, Mapping) and row.get("strategy_id")
    }
    integrity_ok = (
        str(live_performance.get("integrity_status") or "") == "PASSED"
    )
    rows: list[dict[str, Any]] = []
    approved_ids: set[str] = set()
    for approved in authority.get("approved_playbooks") or []:
        if not isinstance(approved, Mapping):
            continue
        playbook_id = str(approved.get("playbook_id") or "")
        if not playbook_id:
            continue
        approved_ids.add(playbook_id)
        playbook_dna = str(approved.get("playbook_dna") or "")
        paper_family_history = dict(
            paper_by_playbook.get(playbook_id) or {}
        )
        paper = dict(paper_by_playbook_dna.get(playbook_dna) or {})
        live = dict(live_rows.get(playbook_id) or {})
        paper_round_trips = int(paper.get("closed_round_trips") or 0)
        live_round_trips = int(live.get("closed_trade_count") or 0)
        paper_expectancy = _decimal(paper.get("paper_net_expectancy_eur"))
        live_expectancy = _decimal(live.get("expectancy_eur"))
        live_pf_raw = live.get("profit_factor")
        live_profit_factor = (
            float(live_pf_raw) if live_pf_raw is not None else None
        )
        recommendation, reasons = _recommendation(
            paper_round_trips=paper_round_trips,
            paper_expectancy=paper_expectancy,
            live_round_trips=live_round_trips,
            live_expectancy=live_expectancy,
            live_profit_factor=live_profit_factor,
            integrity_ok=integrity_ok,
        )
        rows.append(
            {
                "strategy_id": playbook_id,
                "family": approved.get("family"),
                "strategy_dna": playbook_dna,
                "currently_live_authorized": bool(approved.get("active")),
                "evidence_multiplier": str(
                    approved.get("evidence_multiplier") or "1"
                ),
                "paper": {
                    "identity_scope": "EXACT_CURRENT_DNA",
                    "closed_round_trips": paper_round_trips,
                    "net_expectancy_eur": (
                        str(paper_expectancy)
                        if paper_expectancy is not None
                        else None
                    ),
                    "net_pnl_eur": paper.get("paper_net_pnl_eur"),
                    "fees_eur": paper.get("paper_fees_eur"),
                },
                "paper_family_history": {
                    "informational_only": True,
                    "may_not_demote_current_dna": True,
                    "closed_round_trips": int(
                        paper_family_history.get("closed_round_trips") or 0
                    ),
                    "net_expectancy_eur": paper_family_history.get(
                        "paper_net_expectancy_eur"
                    ),
                    "net_pnl_eur": paper_family_history.get(
                        "paper_net_pnl_eur"
                    ),
                },
                "live": {
                    "closed_round_trips": live_round_trips,
                    "open_positions": int(live.get("open_trade_count") or 0),
                    "net_expectancy_eur": (
                        str(live_expectancy)
                        if live_expectancy is not None
                        else None
                    ),
                    "profit_factor": live_profit_factor,
                    "net_pnl_eur": live.get("net_pnl_eur"),
                },
                "recommendation": recommendation,
                "reason_codes": reasons,
                "automatic_authority_change": False,
                "protective_exits_unaffected": True,
            }
        )

    # Exact-DNA canaries can be live without family authority configuration.
    for strategy_id, live in sorted(live_rows.items()):
        if strategy_id in approved_ids:
            continue
        live_round_trips = int(live.get("closed_trade_count") or 0)
        open_positions = int(live.get("open_trade_count") or 0)
        if not live_round_trips and not open_positions:
            continue
        live_expectancy = _decimal(live.get("expectancy_eur"))
        pf_raw = live.get("profit_factor")
        live_pf = float(pf_raw) if pf_raw is not None else None
        recommendation, reasons = _recommendation(
            paper_round_trips=0,
            paper_expectancy=None,
            live_round_trips=live_round_trips,
            live_expectancy=live_expectancy,
            live_profit_factor=live_pf,
            integrity_ok=integrity_ok,
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "family": live.get("strategy_family"),
                "currently_live_authorized": True,
                "authority_source": "EXACT_DNA_LIVE_ACCOUNTING",
                "paper": {"closed_round_trips": 0},
                "live": {
                    "closed_round_trips": live_round_trips,
                    "open_positions": open_positions,
                    "net_expectancy_eur": (
                        str(live_expectancy)
                        if live_expectancy is not None
                        else None
                    ),
                    "profit_factor": live_pf,
                    "net_pnl_eur": live.get("net_pnl_eur"),
                },
                "recommendation": recommendation,
                "reason_codes": reasons,
                "automatic_authority_change": False,
                "protective_exits_unaffected": True,
            }
        )

    payload = {
        "schema_version": "strategy_evidence_watch_v1",
        "generated_at": utc_iso(),
        "live_accounting_integrity": (
            "PASSED" if integrity_ok else "NOT_PASSED"
        ),
        "policy": {
            "automatic_live_promotion": False,
            "automatic_cap_increase": False,
            "automatic_authority_changes": False,
            "paper_and_live_evidence_are_not_interchangeable": True,
            "single_loss_never_disables": True,
            "operator_review_required_for_capital_increase": True,
        },
        "watched_strategy_evidence_row_count": len(rows),
        "recommendation_counts": {
            recommendation: sum(
                row["recommendation"] == recommendation for row in rows
            )
            for recommendation in sorted(
                {row["recommendation"] for row in rows}
            )
        },
        "strategies": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = (
        root / "output" / "operations" / "strategy_evidence_watch.json"
    )
    atomic_write_json(output, payload)
    payload["artifact"] = str(output)
    return payload


__all__ = ["build_strategy_evidence_watch"]
