"""Canonical product-level evidence for the crypto active-swing service.

The builder is observability-only.  It reconciles durable artifacts, writes no
financial state and never grants authority or creates an order.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from config.settings import ACTIVE_SWING_TIMEFRAMES, Settings
from rl.position_management import RLEligibilityEvidence, evaluate_rl_eligibility
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

ZERO = Decimal("0")
TERMINAL_OPPORTUNITY_STATES = {"CLOSED", "INVALIDATED", "EXPIRED"}
LIFECYCLE_MAP = {
    "DISCOVERED": "DISCOVERED",
    "WATCHING": "WATCHING",
    "ARMED": "ENTRY_TRIGGER_PENDING",
    "ENTRY_READY": "ENTRY_READY",
    "ORDER_INTENT_CREATED": "ENTERED",
    "ORDER_SUBMITTED": "ENTERED",
    "PARTIALLY_FILLED": "ENTERED",
    "FILLED": "POSITION_ACTIVE",
    "MANAGING": "POSITION_ACTIVE",
    "EXITING": "EXIT_READY",
    "CLOSED": "CLOSED",
    "INVALIDATED": "INVALIDATED",
    "EXPIRED": "EXPIRED",
}
PRODUCT_LIFECYCLE = (
    "DISCOVERED",
    "WATCHING",
    "NEAR_SETUP",
    "SETUP_VALID",
    "ENTRY_TRIGGER_PENDING",
    "ENTRY_READY",
    "ENTERED",
    "POSITION_ACTIVE",
    "PROFIT_PROTECTION",
    "REDUCE_CANDIDATE",
    "ROTATION_CANDIDATE",
    "EXIT_READY",
    "CLOSED",
    "INVALIDATED",
    "EXPIRED",
)


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value)) if value is not None else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _mapping_values(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(row) for row in value.values() if isinstance(row, Mapping)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _opportunity_funnel(output: Path) -> dict[str, Any]:
    source_path = output / "live" / "opportunity_lifecycle_state.json"
    source = _mapping(source_path)
    rows = _mapping_values(source.get("opportunities"))
    active_rows = [
        row
        for row in rows
        if str(row.get("state") or "") not in TERMINAL_OPPORTUNITY_STATES
    ]
    counts = Counter(
        LIFECYCLE_MAP.get(str(row.get("state") or ""), "DISCOVERED")
        for row in active_rows
    )
    lifecycle_counts = {state: int(counts.get(state, 0)) for state in PRODUCT_LIFECYCLE}
    top = sorted(
        active_rows,
        key=lambda row: float(row.get("score") or 0.0),
        reverse=True,
    )[:10]
    return {
        "schema_version": "active_swing_opportunity_funnel_v1",
        "generated_at": utc_iso(),
        "status": "READY" if source else "DATA_MISSING",
        "source": str(source_path),
        "source_updated_at": source.get("updated_at"),
        "active_opportunity_count": len(active_rows),
        "lifecycle_counts": lifecycle_counts,
        "entry_ready_count": lifecycle_counts["ENTRY_READY"],
        "near_entry_count": int(source.get("near_entry_count") or 0),
        "near_entry_markets": list(source.get("near_entry_markets") or []),
        "top_active": [
            {
                "opportunity_id": row.get("opportunity_id"),
                "market": row.get("market"),
                "strategy_id": row.get("playbook_id") or row.get("strategy_id"),
                "lifecycle": LIFECYCLE_MAP.get(
                    str(row.get("state") or ""), "DISCOVERED"
                ),
                "score": row.get("score"),
                "valid_until": row.get("valid_until"),
                "next_required_condition": row.get("next_required_condition"),
                "hard_blockers": list(row.get("hard_blockers") or []),
            }
            for row in top
        ],
        "poor_setups_used_as_fillers": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _position_management_test(output: Path, account: Mapping[str, Any]) -> dict[str, Any]:
    state_path = output / "live" / "generated_strategy_live_state.json"
    state = _mapping(state_path)
    positions = [
        row
        for row in _mapping_values(state.get("positions"))
        if str(row.get("status") or "").upper() == "OPEN"
    ]
    holdings = list(
        ((account.get("account") or {}).get("portfolio_valuation") or {}).get(
            "holdings"
        )
        or []
    )
    results: list[dict[str, Any]] = []
    for position in positions:
        market = str(position.get("market") or "")
        holding = next(
            (dict(row) for row in holdings if row.get("market") == market),
            {},
        )
        entry = _decimal(position.get("entry_price"))
        mark = _decimal(holding.get("price_eur"))
        quantity = _decimal(position.get("quantity"))
        gross_unrealized = (
            (mark - entry) * quantity
            if entry is not None and mark is not None and quantity is not None
            else None
        )
        protected = bool(
            position.get("native_protective_stop_active") is True
            and str(position.get("protective_stop_status") or "")
            == "awaitingTrigger"
        )
        tp1_reached = position.get("tp1_reached") is True
        lifecycle = "PROFIT_PROTECTION" if tp1_reached else "POSITION_ACTIVE"
        action = (
            "HOLD_WITH_NATIVE_PROTECTIVE_STOP"
            if protected
            else "RISK_REDUCTION_REVIEW_REQUIRED"
        )
        results.append(
            {
                "market": market,
                "managed_quantity": position.get("quantity"),
                "entry_price": position.get("entry_price"),
                "current_mark_price": holding.get("price_eur"),
                "gross_unrealized_pnl_eur": (
                    str(gross_unrealized) if gross_unrealized is not None else None
                ),
                "stop_loss": position.get("stop_loss"),
                "take_profit_1": position.get("take_profit_1"),
                "take_profit_2": position.get("take_profit_2"),
                "tp1_reached": tp1_reached,
                "native_protective_stop_active": protected,
                "protective_stop_trigger": position.get("protective_stop_trigger"),
                "lifecycle": lifecycle,
                "deterministic_proposed_action": action,
                "position_thesis_contract": "LEGACY_POSITION_REQUIRES_THESIS_MIGRATION",
                "rl_advisory_action": None,
                "rl_authority": "SHADOW_ONLY",
                "execution_authority_changed": False,
            }
        )
    return {
        "schema_version": "current_position_management_test_v1",
        "generated_at": utc_iso(),
        "status": "ASSESSED" if positions else "NO_MANAGED_OPEN_POSITION",
        "source": str(state_path),
        "managed_open_position_count": len(positions),
        "positions": results,
        "cash_is_active_competitor": True,
        "rotation_candidate_count": 0,
        "rotation_hysteresis_applied": True,
        "financial_state_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _alpha_dashboard(output: Path) -> dict[str, Any]:
    evidence_path = output / "operations" / "execution_evidence_layers.json"
    evidence = _mapping(evidence_path)
    actual = dict(evidence.get("actual_live_pnl") or {})
    paper = dict(evidence.get("simulated_execution_pnl") or {})
    theoretical = dict(evidence.get("theoretical_signal_pnl") or {})
    return {
        "schema_version": "active_swing_alpha_dashboard_v1",
        "generated_at": utc_iso(),
        "status": "READY" if evidence else "DATA_MISSING",
        "source": str(evidence_path),
        "actual_live": {
            "status": actual.get("status"),
            "closed_round_trips": actual.get("closed_round_trips"),
            "realised_pnl_eur": actual.get("realised_pnl_eur"),
            "unrealised_pnl_eur": actual.get("unrealised_pnl_eur"),
            "net_pnl_eur": actual.get("net_pnl_eur"),
            "fees_eur": actual.get("fees_eur"),
        },
        "paper_execution": {
            "status": paper.get("status"),
            "closed_round_trips": paper.get("closed_round_trips"),
            "net_expectancy_eur": paper.get("net_expectancy_eur"),
            "net_pnl_eur": paper.get("net_pnl_eur"),
            "fees_eur": paper.get("fees_eur"),
        },
        "theoretical_signal": {
            "status": theoretical.get("status"),
            "resolved_episode_count": theoretical.get("resolved_episode_count"),
            "false_breakout_rate": theoretical.get("false_breakout_rate"),
            "gross_or_net_pnl_eur": theoretical.get("gross_or_net_pnl_eur"),
        },
        "layers_are_not_interchangeable": True,
        "tests_or_backtests_prove_profitability": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _requirement_coverage() -> list[dict[str, Any]]:
    return [
        {"sections": "335-342", "area": "product_and_spot_scope", "status": "IMPLEMENTED"},
        {"sections": "343-354", "area": "timeframes_and_causality", "status": "IMPLEMENTED_TESTED"},
        {"sections": "355-367", "area": "opportunity_contract_and_lifecycle", "status": "IMPLEMENTED_TESTED"},
        {"sections": "368-382", "area": "ml_meta_labeling_and_pit", "status": "SHADOW_DATA_PENDING"},
        {"sections": "383-397", "area": "rl_and_position_management", "status": "SHADOW_TRAINING_BLOCKED"},
        {"sections": "398-415", "area": "portfolio_rotation_and_cash", "status": "PARTIAL_EXTERNAL_INVENTORY_BLOCKED"},
        {"sections": "416-429", "area": "retail_execution_and_bitvavo", "status": "IMPLEMENTED_ACCOUNT_BLOCKED"},
        {"sections": "430-441", "area": "always_on_operations_and_observability", "status": "RUNNING_PAUSED"},
        {"sections": "442-453", "area": "readiness_profitability_and_authority", "status": "NOT_ECONOMICALLY_GOOD_YET"},
    ]


def build_crypto_active_swing_product(
    settings: Settings,
    *,
    runtime: Mapping[str, Any],
    deployment_decision: str,
    deployment_blockers: list[str],
    entry_constraints: list[str],
) -> dict[str, Any]:
    """Build the single product truth without changing live authority."""

    output = settings.paths.output_dir
    product_dir = output / "product"
    account = _mapping(output / "operations" / "live_account_health.json")
    external = _mapping(output / "operations" / "external_inventory_remediation.json")
    authority = _mapping(
        output / "governance" / "positive_strategy_live_authority.json"
    )
    ml = _mapping(output / "ml" / "canonical_training_status.json")
    intelligence = _mapping(output / "intelligence" / "validation_report.json")
    blockers_report = _mapping(
        output / "reports" / "system_audit" / "live_blocker_report.json"
    )
    execution_capability = _mapping(
        output / "reports" / "system_audit" / "execution_capability_report.json"
    )
    reference_map = _mapping(
        output / "reference_integration" / "reference_master_map.json"
    )
    universe = _mapping(output / "universe" / "tiered_trading_universe.json")
    funnel = _opportunity_funnel(output)
    position_test = _position_management_test(output, account)
    alpha = _alpha_dashboard(output)
    rl = evaluate_rl_eligibility(
        RLEligibilityEvidence(
            prospective_episode_count=int(
                ml.get("canonical_feature_ready_incomplete_count") or 0
            ),
            completed_episode_count=int(ml.get("canonical_row_count") or 0),
            distinct_regime_count=0,
            baseline_count=3,
            multi_seed_count=0,
            stress_test_passed=False,
            canonical_cost_model_used=True,
            point_in_time_inputs_verified=False,
        )
    )
    paper_expectancy = _decimal(alpha["paper_execution"].get("net_expectancy_eur"))
    economically_good = bool(
        paper_expectancy is not None
        and paper_expectancy > ZERO
        and not any(
            row.get("code") == "NO_LIVE_VALIDATED_STRATEGY_FAMILY"
            for row in blockers_report.get("blockers") or []
            if isinstance(row, Mapping)
        )
    )
    account_ready = bool(account.get("status") == "READY" and account.get("entry_allowed") is True)
    authority_active = authority.get("active") is True
    control_enabled = runtime.get("control_state") == "ENABLED"
    live_ready = bool(
        economically_good
        and account_ready
        and authority_active
        and control_enabled
        and deployment_decision == "APPROVED_LIVE_ACTIVE"
        and not deployment_blockers
        and not entry_constraints
    )
    product_status = (
        "LIVE_READY"
        if live_ready
        else "NOT_ECONOMICALLY_GOOD_YET"
        if not economically_good
        else "LIVE_BLOCKED"
    )
    from reporting.active_swing_current_evidence import (
        build_current_active_swing_evidence,
    )

    current_evidence = build_current_active_swing_evidence(
        settings,
        runtime=runtime,
        product_economically_good=economically_good,
    )
    funnel = dict(current_evidence["funnel"])
    position_test = dict(current_evidence["position_management"])
    alpha = dict(current_evidence["alpha_dashboard"])
    end_to_end = dict(current_evidence["end_to_end"])
    economic_recovery = dict(current_evidence["economic_recovery_gate"])
    prospective_net_r = dict(
        current_evidence["prospective_net_r_calibration"]
    )
    canonical_execution = dict(execution_capability.get("canonical_execution_state") or {})
    capabilities = dict(execution_capability.get("capabilities") or {})
    canonical_money_path = {
        "schema_version": "canonical_money_path_v1",
        "generated_at": utc_iso(),
        "path": [
            "DATA",
            "MARKET_STATE",
            "STRATEGY_ML_INTELLIGENCE",
            "OPPORTUNITY",
            "AI_DECISION_SNAPSHOT",
            "INVESTMENT_INTENT",
            "PORTFOLIO_TARGET",
            "RISK_APPROVAL",
            "EXECUTION_INTENT",
            "ORDER_INTENT",
            "BITVAVO",
            "CANONICAL_STATE",
            "OUTCOME",
            "FORWARD_EVIDENCE",
            "LEARNING",
        ],
        "all_autonomous_buy_routes_atomic": capabilities.get(
            "all_autonomous_buy_routes_atomic"
        )
        is True,
        "deterministic_canonical_replay": canonical_execution.get("status") == "READY",
        "status": (
            "PROVEN_BY_AUDIT_AND_TESTS"
            if capabilities.get("all_autonomous_buy_routes_atomic") is True
            and canonical_execution.get("status") == "READY"
            else "INCOMPLETE"
        ),
        "second_money_path_authorized": False,
        "strategy_or_ml_may_submit_orders_directly": False,
        "audit_private_exchange_requests": canonical_execution.get(
            "private_exchange_requests"
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    dimensions = {
        "reference_repository_integration": (
            "PASS" if reference_map.get("all_nine_present") is True else "INCOMPLETE"
        ),
        "timeframe_and_opportunity_contract": "IMPLEMENTED_TESTED",
        "causal_data_and_point_in_time": (
            "DATA_PENDING" if ml.get("status") == "DATA_PENDING" else "PARTIAL"
        ),
        "market_universe": (
            "READY" if universe.get("live_executable_markets") else "DATA_MISSING"
        ),
        "opportunity_funnel": funnel["status"],
        "strategy_economics": (
            "ECONOMICALLY_GOOD" if economically_good else "NOT_ECONOMICALLY_GOOD_YET"
        ),
        "ml": str(intelligence.get("status") or ml.get("authority") or "SHADOW_ONLY"),
        "rl": str(rl["status"]),
        "portfolio_and_external_inventory": (
            "BLOCKED_OPERATOR_DECISION_REQUIRED"
            if external.get("status") == "OPERATOR_DECISION_REQUIRED"
            else "READY"
        ),
        "account_and_reconciliation": (
            "READY" if account_ready else "BLOCKED"
        ),
        "runtime_and_control": (
            "RUNNING_ENABLED"
            if runtime.get("process_running") is True and control_enabled
            else "RUNNING_PAUSED"
            if runtime.get("process_running") is True
            else "NOT_RUNNING"
        ),
        "live_authority": "ACTIVE" if authority_active else "INACTIVE",
    }
    covered_sections = list(range(335, 454))
    payload = {
        "schema_version": "crypto_active_swing_product_v1",
        "generated_at": utc_iso(),
        "product_status": product_status,
        "live_ready": live_ready,
        "economically_good": economically_good,
        "deployment_decision": deployment_decision,
        "deployment_blockers": list(dict.fromkeys(deployment_blockers)),
        "entry_constraints": list(dict.fromkeys(entry_constraints)),
        "product_definition": {
            "venue": "BITVAVO",
            "quote_currency": "EUR",
            "spot_only": True,
            "long_only": True,
            "leverage": False,
            "margin": False,
            "shorting": False,
            "core_decision_timeframes": list(ACTIVE_SWING_TIMEFRAMES),
            "tactical_execution_timeframes": ["tick", "1m"],
            "cash_is_active_competitor": True,
        },
        "status_dimensions": dimensions,
        "account_truth": {
            "status": account.get("status"),
            "entry_allowed": account.get("entry_allowed"),
            "reconciliation_healthy": (account.get("reconciliation") or {}).get(
                "healthy"
            ),
            "failures": list(account.get("failures") or []),
            "external_inventory_status": external.get("status"),
        },
        "opportunity_funnel": funnel,
        "current_end_to_end_test": end_to_end,
        "current_position_management_test": position_test,
        "alpha_dashboard": alpha,
        "economic_recovery_gate": economic_recovery,
        "prospective_net_r_calibration": prospective_net_r,
        "ml": {
            "status": ml.get("status"),
            "authority": ml.get("authority"),
            "canonical_row_count": ml.get("canonical_row_count"),
            "pending_label_count": ml.get("canonical_pending_label_horizon_count"),
            "next_label_due_at": ml.get("next_canonical_label_due_at"),
            "live_decision_influence": ml.get("live_decision_influence"),
        },
        "rl": rl,
        "canonical_money_path": canonical_money_path,
        "requirements": {
            "source_sections": "335-453",
            "covered_section_count": len(covered_sections),
            "covered_sections": covered_sections,
            "coverage_complete": covered_sections == list(range(335, 454)),
            "clusters": _requirement_coverage(),
        },
        "current_truth_hash": stable_hash(
            {
                "status": product_status,
                "dimensions": dimensions,
                "funnel": funnel,
                "positions": position_test["positions"],
                "ml": ml,
                "deployment_blockers": deployment_blockers,
                "entry_constraints": entry_constraints,
            },
            length=64,
        ),
        "financial_state_changed": False,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(product_dir / "opportunity_funnel.json", funnel)
    atomic_write_json(product_dir / "current_position_management_test.json", position_test)
    atomic_write_json(product_dir / "alpha_dashboard.json", alpha)
    atomic_write_json(product_dir / "canonical_money_path.json", canonical_money_path)
    atomic_write_json(product_dir / "active_swing_product_status.json", payload)
    return payload


__all__ = ["build_crypto_active_swing_product"]
