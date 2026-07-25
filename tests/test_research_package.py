from __future__ import annotations

import pytest

from reporting.research_package import build_acceptance_summary


def _inputs() -> dict:
    identity = "frozen-identity"
    dna_hash = "frozen-dna"
    stochastic = {
        "passed": False,
        "checks": {
            "normal_monte_carlo": False,
            "normal_dirichlet": True,
            "stressed_monte_carlo": False,
            "stressed_dirichlet": True,
        },
    }
    quality = {
        "ruff": {"passed": True},
        "pytest": {"passed": True, "passed_count": 1},
    }
    return {
        "source_commit": "a" * 40,
        "lead": {
            "status": "FROZEN_RESEARCH_LEAD",
            "immutable_identity": identity,
            "strategy_dna_hash": dna_hash,
            "candidate_type": "ECONOMIC_RESEARCH_LEAD_NOT_PAPER_APPROVED",
            "parameters": {"momentum_lookback": 20},
            "selection_bias": "CONTAMINATED_BY_PRIOR_EXPLORATION",
            "robustness": {
                "economic_gates_passed": True,
                "statistical_gates_passed": False,
            },
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "campaign": {
            "status": "COMPLETED",
            "campaign": "ENSEMBLE",
            "joint_parameter_trials": 160,
            "total_known_family_trials": 1_240,
            "positive_all_three_periods_descriptive_only": 119,
            "survivor_count": 12,
            "economic_research_lead_count": 1,
            "statistically_qualified_count": 0,
            "multiple_testing": {"known_trial_count": 1_240},
            "survivors": [
                {
                    "strategy_dna_hash": dna_hash,
                    "robustness": {"stochastic_validation": stochastic},
                }
            ],
        },
        "external": {
            "candidate_identity": identity,
            "strategy_dna_hash": dna_hash,
            "status": "EXTERNAL_ECONOMIC_PASS_STATISTICAL_PARTIAL",
            "global_checks": {
                "all_views_net_positive": True,
                "all_views_stressed_positive": True,
                "white_reality_check": True,
                "hansen_spa": True,
                "pbo": False,
                "at_least_one_dsr_pass": False,
            },
            "multiple_testing": {"known_trial_count": 1_245},
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "forward": {
            "candidate_identity": identity,
            "strategy_dna_hash": dna_hash,
            "status": "COLLECTING_FORWARD_DATA",
            "reason_code": "INSUFFICIENT_NEW_CLOSED_DAILY_OBSERVATIONS",
            "required_closed_daily_observations": 365,
            "required_rebalances": 30,
            "required_regime_coverage": {"minimum_decisions_per_state": 5},
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "institutional_audit": {
            "source_candidate_identity": identity,
            "status": "STRICT_ALLOWED_POLICY_ECONOMIC_PASS",
            "execution_identity": "strict-execution",
            "economic_gates_passed": True,
            "historical_statistical_gates_passed": False,
            "portfolio_policy": {"maximum_position_exposure": 0.20},
            "exposure_semantics": {"hard_minimum_cash": 0.60},
            "checks": {"minimum_cash": True},
            "normal": {"metrics": {"net_return": 1.5}},
            "stressed": {"metrics": {"net_return": 1.2}},
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "continuation": {
            "status": "COMPLETED",
            "joint_parameter_trials": 48,
            "prior_exploratory_trials_accounted": 1_245,
            "total_known_family_trials": 1_293,
            "positive_all_three_periods_descriptive_only": 44,
            "economic_research_lead_count": 0,
            "statistically_qualified_count": 0,
            "survivors": [
                {
                    "strategy_dna_hash": "continuation-dna",
                    "robustness": {"stochastic_validation": stochastic},
                }
            ],
        },
        "observer": {
            "source_candidate_identity": identity,
            "status": "FROZEN_FORWARD_RESEARCH",
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        },
        "quality": quality,
    }


def test_acceptance_summary_keeps_positive_research_fail_closed() -> None:
    summary = build_acceptance_summary(**_inputs())

    assert summary["validation"]["economic_gates_passed"]
    assert not summary["validation"]["statistical_gates_passed"]
    assert not summary["validation"]["forward_passed"]
    assert not summary["promotion"]["shadow_candidate"]
    assert not summary["promotion"]["paper_candidate_permitted"]
    assert not summary["promotion"]["live_ready"]


def test_acceptance_summary_rejects_cross_candidate_evidence() -> None:
    values = _inputs()
    values["external"]["candidate_identity"] = "different"

    with pytest.raises(ValueError, match="identity mismatch"):
        build_acceptance_summary(**values)


def test_acceptance_summary_includes_orderless_autopilot_evidence() -> None:
    values = _inputs()
    values["autopilot_state"] = {
        "status": "COMPLETED_ORDERLESS",
        "cycle_count": 4,
        "last_cycle_id": "AUTO_TEST",
        "last_completed_at": "2026-07-25T00:00:00Z",
        "last_data_fingerprint": "data-hash",
        "last_research_at": "2026-07-25T00:00:00Z",
        "last_feature_store_dataset_id": "tensor-id",
        "research_ran": False,
        "research_reason": "DATA_UNCHANGED",
        "degradation": {"status": "INSUFFICIENT_FORWARD_DATA"},
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    values["autopilot_degradation"] = {
        "system_degraded": False,
        "status": "HEALTHY",
    }
    values["feature_store"] = {
        "schema_version": "portfolio_daily_causal_v1",
        "dataset_id": "tensor-id",
        "frequency": "1d",
        "assets": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
        "feature_names": ["log_return_1"],
        "shapes": {"features": [10, 4, 1]},
        "per_asset": {},
        "causality": {"closed_candles_only": True},
        "tensor_sha256": "tensor-sha",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    values["breakout_forward_observers"] = {
        "observer.json": {
            "source_candidate_identity": "frozen-identity",
            "policy_name": "TURTLE_TEST",
            "strategy_dna_hash": "breakout-dna",
            "forward_observer_schema_version": "forward-v1",
            "forward_summary": {
                "status": "COLLECTING_FORWARD_DATA",
                "closed_daily_observations": 0,
                "forward_performance_pass": False,
            },
            "degradation_observation": None,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    }

    summary = build_acceptance_summary(**values)

    assert summary["autopilot"]["status"] == "COMPLETED_ORDERLESS"
    assert summary["autopilot"]["cycle_count"] == 4
    assert not summary["autopilot"]["persistent_kill_switch"][
        "system_degraded"
    ]
    assert summary["autopilot"]["orders_generated"] == 0
    assert (
        summary["autopilot"]["last_feature_store_dataset_id"]
        == "tensor-id"
    )
    assert summary["feature_store"]["dataset_id"] == "tensor-id"
    assert summary["feature_store"]["research_only"] is True
    assert summary["breakout_forward_observers"]["policy_count"] == 1
    assert not summary["breakout_forward_observers"][
        "all_formal_performance_pass"
    ]


def test_acceptance_summary_rejects_autopilot_live_permission() -> None:
    values = _inputs()
    values["autopilot_state"] = {
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": True,
    }

    with pytest.raises(ValueError, match="live permission"):
        build_acceptance_summary(**values)


def test_acceptance_summary_rejects_feature_store_identity_mismatch() -> None:
    values = _inputs()
    values["autopilot_state"] = {
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
        "last_feature_store_dataset_id": "expected",
    }
    values["feature_store"] = {
        "dataset_id": "different",
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }

    with pytest.raises(ValueError, match="identity mismatch"):
        build_acceptance_summary(**values)


def test_acceptance_summary_includes_fail_closed_plateau_campaign() -> None:
    values = _inputs()
    values["absolute_momentum_plateau"] = {
        "status": "COMPLETED_NOT_PROMOTED",
        "campaign": "ABSOLUTE_MOMENTUM_PLATEAU_V1",
        "engine_version": "1.0.0",
        "generated_trial_count": 117,
        "registered_unique_plateau_trials": 117,
        "total_known_trials": 16_832,
        "plateau_eligible_count": 81,
        "primary_strategy_id": "AMPS_P01_V90_T04",
        "multiple_testing": {
            "probability_of_backtest_overfitting": 0.4857,
            "plateau_selection_pbo": 0.5143,
        },
        "trial_registry": {
            "status": "PASSED",
            "unique_trial_count": 117,
        },
        "primary_result": {
            "gates": {
                "stochastic_validation": {
                    "passed": True,
                },
                "research_pass": False,
            },
        },
        "holdout_status": (
            "NO_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS"
        ),
        "orders_generated": 0,
        "live_ready": False,
    }

    summary = build_acceptance_summary(**values)

    plateau = summary["absolute_momentum_plateau"]
    assert plateau["registered_unique_plateau_trials"] == 117
    assert plateau["trial_registry"]["status"] == "PASSED"
    assert not plateau["primary_result"]["gates"]["research_pass"]
    assert plateau["orders_generated"] == 0
    assert plateau["live_ready"] is False
