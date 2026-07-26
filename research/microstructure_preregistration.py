"""Preregister one small positioning family before sufficient data exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.common import atomic_write_json, read_json, stable_hash

FAMILY_ID = "CROWDING_AVOIDANCE_V1"
RESEARCH_STAGES = (
    "technical_feature_validation",
    "preliminary_research",
    "formal_regime_assessment",
)


def crowding_avoidance_plan() -> dict[str, Any]:
    """Return fixed DNA and gates; no historical result is inspected."""

    body: dict[str, Any] = {
        "schema_version": "economic_hypothesis_card_v1",
        "family_id": FAMILY_ID,
        "status": "PREREGISTERED_WAITING_FOR_PROSPECTIVE_DATA",
        "economic_mechanism": (
            "A price advance led by leveraged perpetual positioning, "
            "without confirming spot demand, is more fragile than a "
            "spot-led advance. The signal only blocks new spot longs."
        ),
        "who_causes_edge": (
            "crowded leveraged perpetual longs and their forced unwind"
        ),
        "who_pays": (
            "late leveraged buyers entering without confirming spot flow"
        ),
        "why_edge_can_persist": (
            "position unwinds, inventory transfer and cross-venue "
            "arbitrage require time and balance-sheet capacity"
        ),
        "expected_work_regimes": [
            "positive price trend with extreme positive funding",
            "rapid open-interest growth",
            "perpetual volume dominance",
            "weak or non-confirming spot CVD",
        ],
        "expected_failure_regimes": [
            "genuine spot-led accumulation",
            "neutral funding and stable open interest",
            "structural supply shocks",
        ],
        "decision_horizon": "4h",
        "execution": "NEXT_OPEN_AFTER_COMPLETED_4H_BAR",
        "action": "BLOCK_NEW_LONG_ONLY",
        "markets": [
            "BTC-EUR",
            "ETH-EUR",
            "SOL-EUR",
            "LINK-EUR",
        ],
        "required_point_in_time_fields": [
            "funding_rate",
            "funding_zscore",
            "open_interest",
            "open_interest_change",
            "perpetual_spot_volume_ratio",
            "spot_cvd",
            "spot_cvd_zscore",
            "event_time",
            "arrival_time",
            "available_at",
            "raw_payload_hash",
        ],
        "primary_dna": [
            {
                "id": "CA_V1_DNA_01",
                "funding_z_min": 2.0,
                "open_interest_change_min": 0.10,
                "perpetual_spot_volume_ratio_min": 1.5,
                "spot_cvd_z_max": 0.0,
            },
            {
                "id": "CA_V1_DNA_02",
                "funding_z_min": 2.5,
                "open_interest_change_min": 0.10,
                "perpetual_spot_volume_ratio_min": 2.0,
                "spot_cvd_z_max": 0.0,
            },
            {
                "id": "CA_V1_DNA_03",
                "funding_z_min": 2.0,
                "open_interest_change_min": 0.15,
                "perpetual_spot_volume_ratio_min": 1.5,
                "spot_cvd_z_max": -0.5,
            },
            {
                "id": "CA_V1_DNA_04",
                "funding_z_min": 2.5,
                "open_interest_change_min": 0.15,
                "perpetual_spot_volume_ratio_min": 2.0,
                "spot_cvd_z_max": -0.5,
            },
        ],
        "adaptive_expansion_permitted": False,
        "threshold_changes_after_results_permitted": False,
        "minimum_data_days": {
            "technical_feature_validation": 90,
            "preliminary_research": 180,
            "formal_regime_assessment": 365,
        },
        "economic_gates": {
            "validation_profit_factor_minimum": 1.15,
            "confirmation_profit_factor_minimum": 1.15,
            "stressed_confirmation_profit_factor_minimum": 1.0,
            "maximum_drawdown": 0.20,
            "paired_alpha_ci_lower_must_be_positive": True,
        },
        "statistical_gates": {
            "deflated_sharpe_minimum": 0.95,
            "pbo_maximum_when_applicable": 0.10,
            "white_reality_check_pvalue_maximum": 0.10,
            "hansen_spa_pvalue_maximum": 0.05,
            "positive_walk_forward_folds_minimum": "5_OF_6",
        },
        "forward_gates": {
            "minimum_new_closed_days": 365,
            "minimum_actual_target_changes": 30,
            "normal_return_positive": True,
            "stressed_return_positive": True,
            "ci_lower_positive": True,
            "integrity_failures_allowed": 0,
        },
        "data_readiness": "INSUFFICIENT_NOT_BACKTESTED",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
    }
    return {
        **body,
        "plan_hash": stable_hash(body, length=64),
    }


def write_crowding_avoidance_plan(path: Path) -> dict[str, Any]:
    plan = crowding_avoidance_plan()
    if path.is_file():
        if read_json(path) != plan:
            raise RuntimeError(
                "MICROSTRUCTURE_PLAN_HISTORY_REVISION"
            )
    else:
        atomic_write_json(path, plan)
    return {**plan, "plan_path": str(path)}


def microstructure_research_gate(
    readiness: dict[str, Any],
    *,
    requested_stage: str,
) -> dict[str, Any]:
    """Fail closed until the requested prospective milestone is proven."""

    if requested_stage not in RESEARCH_STAGES:
        raise ValueError("UNKNOWN_MICROSTRUCTURE_RESEARCH_STAGE")
    milestones = dict(readiness.get("milestones") or {})
    milestone = dict(milestones.get(requested_stage) or {})
    snapshot_audit = dict(readiness.get("snapshot_audit") or {})
    permitted = bool(
        milestone.get("eligible")
        and readiness.get("synthetic_data_used") is False
        and snapshot_audit.get("latest_snapshot_eligible") is True
        and (
            requested_stage != "technical_feature_validation"
            or readiness.get("backtest_permitted") is True
        )
    )
    body = {
        "schema_version": "microstructure_research_gate_v1",
        "family_id": FAMILY_ID,
        "requested_stage": requested_stage,
        "status": "PERMITTED" if permitted else "BLOCKED",
        "reason_code": (
            "PROSPECTIVE_MILESTONE_PROVEN"
            if permitted
            else "INSUFFICIENT_VERIFIED_PROSPECTIVE_HISTORY"
        ),
        "required_complete_hours": milestone.get(
            "required_complete_hours"
        ),
        "remaining_complete_hours": milestone.get(
            "remaining_complete_hours"
        ),
        "consecutive_complete_hours": readiness.get(
            "consecutive_complete_hours",
            0,
        ),
        "excluded_snapshot_count": snapshot_audit.get(
            "excluded_snapshot_count",
            0,
        ),
        "readiness_hash": readiness.get("readiness_hash"),
        "backtest_permitted": (
            permitted
            if requested_stage == "technical_feature_validation"
            else bool(readiness.get("backtest_permitted"))
        ),
        "paper_permitted": False,
        "live_permitted": False,
        "orders_generated": 0,
    }
    return {**body, "gate_hash": stable_hash(body, length=64)}


__all__ = [
    "FAMILY_ID",
    "RESEARCH_STAGES",
    "crowding_avoidance_plan",
    "microstructure_research_gate",
    "write_crowding_avoidance_plan",
]
