"""Immutable source-and-evidence package for the frozen rotation research lead."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from config.settings import Settings
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    stable_hash,
    utc_iso,
)

_PYTEST_COUNTS = re.compile(
    r"(?P<passed>\d+) passed"
    r"(?:, (?P<deselected>\d+) deselected)?"
    r"(?:, (?P<warnings>\d+) warnings?)?"
)


def _run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _quality_evidence(project_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, arguments in {
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
    }.items():
        completed = _run(arguments, cwd=project_root, check=False)
        output = "\n".join(
            value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "return_code": completed.returncode,
            "command": arguments,
            "summary": output[-8_000:],
        }
    match = _PYTEST_COUNTS.search(results["pytest"]["summary"])
    if match:
        results["pytest"].update(
            {
                "passed_count": int(match.group("passed")),
                "deselected_count": int(match.group("deselected") or 0),
                "warning_count": int(match.group("warnings") or 0),
            }
        )
    failed = [name for name, result in results.items() if not result["passed"]]
    if failed:
        raise RuntimeError(f"package quality checks failed: {failed}")
    return results


def build_acceptance_summary(
    *,
    source_commit: str,
    lead: dict[str, Any],
    campaign: dict[str, Any],
    external: dict[str, Any],
    forward: dict[str, Any],
    institutional_audit: dict[str, Any],
    continuation: dict[str, Any],
    observer: dict[str, Any],
    quality: dict[str, Any],
    capital_utilization: dict[str, Any] | None = None,
    diversified_rotation: dict[str, Any] | None = None,
    portfolio_breakout: dict[str, Any] | None = None,
    absolute_momentum: dict[str, Any] | None = None,
    absolute_momentum_plateau: dict[str, Any] | None = None,
    volatility_contraction: dict[str, Any] | None = None,
    multi_alpha_ensemble: dict[str, Any] | None = None,
    trend_pullback: dict[str, Any] | None = None,
    portfolio_storm: dict[str, Any] | None = None,
    signal_synthesis_storm: dict[str, Any] | None = None,
    autopilot_state: dict[str, Any] | None = None,
    autopilot_degradation: dict[str, Any] | None = None,
    feature_store: dict[str, Any] | None = None,
    ai_governance: dict[str, Any] | None = None,
    breakout_forward_observers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cross-check evidence identities and state the strongest honest conclusion."""

    def require_stochastic_validation(
        label: str,
        rows: list[dict[str, Any]],
        *,
        container: str,
    ) -> None:
        missing = [
            str(row.get("policy_name") or row.get("strategy_dna_hash") or index)
            for index, row in enumerate(rows)
            if not isinstance(row.get(container), dict)
            or not isinstance(row[container].get("stochastic_validation"), dict)
        ]
        if missing:
            raise ValueError(
                f"{label} rows lack formal Monte Carlo/Dirichlet evidence: {missing}"
            )

    identity = str(lead["immutable_identity"])
    dna_hash = str(lead["strategy_dna_hash"])
    if lead.get("status") != "FROZEN_RESEARCH_LEAD":
        raise ValueError("candidate is not a frozen research lead")
    if campaign.get("status") != "COMPLETED":
        raise ValueError("source ensemble campaign is incomplete")
    require_stochastic_validation(
        "source rotation campaign",
        campaign["survivors"],
        container="robustness",
    )
    require_stochastic_validation(
        "institutional rotation continuation",
        continuation["survivors"],
        container="robustness",
    )
    for label, artifact in (("external", external), ("forward", forward)):
        if artifact.get("candidate_identity") != identity:
            raise ValueError(f"{label} candidate identity mismatch")
        if artifact.get("strategy_dna_hash") != dna_hash:
            raise ValueError(f"{label} strategy DNA mismatch")
    if institutional_audit.get("source_candidate_identity") != identity:
        raise ValueError("institutional audit candidate identity mismatch")
    if observer.get("source_candidate_identity") != identity:
        raise ValueError("forward observer candidate identity mismatch")
    if capital_utilization is not None:
        if capital_utilization.get("source_candidate_identity") != identity:
            raise ValueError("capital utilization candidate identity mismatch")
        if capital_utilization.get("strategy_dna_hash") != dna_hash:
            raise ValueError("capital utilization strategy DNA mismatch")
        require_stochastic_validation(
            "capital utilization",
            capital_utilization["policy_results"],
            container="gates",
        )
    if diversified_rotation is not None:
        if diversified_rotation.get("source_candidate_identity") != identity:
            raise ValueError("diversified rotation candidate identity mismatch")
        if diversified_rotation.get("source_frozen_strategy_dna_hash") != dna_hash:
            raise ValueError("diversified rotation source DNA mismatch")
        require_stochastic_validation(
            "diversified rotation",
            diversified_rotation["policy_results"],
            container="gates",
        )
    if portfolio_breakout is not None:
        if portfolio_breakout.get("source_candidate_identity") != identity:
            raise ValueError("portfolio breakout candidate identity mismatch")
        if portfolio_breakout.get("source_frozen_strategy_dna_hash") != dna_hash:
            raise ValueError("portfolio breakout source DNA mismatch")
        require_stochastic_validation(
            "portfolio breakout",
            portfolio_breakout["policy_results"],
            container="gates",
        )
    if absolute_momentum is not None:
        if absolute_momentum.get("campaign") != "ABSOLUTE_MOMENTUM_V1":
            raise ValueError("absolute momentum campaign identity mismatch")
        if int(absolute_momentum.get("orders_generated") or 0) != 0:
            raise ValueError("absolute momentum campaign contains orders")
        if bool(absolute_momentum.get("live_ready", False)):
            raise ValueError("absolute momentum campaign contains live permission")
        require_stochastic_validation(
            "absolute momentum",
            absolute_momentum["policy_results"],
            container="gates",
        )
    if absolute_momentum_plateau is not None:
        if (
            absolute_momentum_plateau.get("campaign")
            != "ABSOLUTE_MOMENTUM_PLATEAU_V1"
        ):
            raise ValueError(
                "absolute momentum plateau campaign identity mismatch"
            )
        if int(
            absolute_momentum_plateau.get("orders_generated") or 0
        ) != 0:
            raise ValueError(
                "absolute momentum plateau campaign contains orders"
            )
        if bool(
            absolute_momentum_plateau.get("live_ready", False)
        ):
            raise ValueError(
                "absolute momentum plateau campaign contains live permission"
            )
        registry = absolute_momentum_plateau.get("trial_registry") or {}
        if registry.get("status") != "PASSED":
            raise ValueError(
                "absolute momentum plateau registry audit failed"
            )
        if int(registry.get("unique_trial_count") or 0) != int(
            absolute_momentum_plateau.get(
                "registered_unique_plateau_trials"
            )
            or 0
        ):
            raise ValueError(
                "absolute momentum plateau trial count mismatch"
            )
        primary = absolute_momentum_plateau.get("primary_result")
        if not isinstance(primary, dict):
            raise ValueError(
                "absolute momentum plateau primary result missing"
            )
        require_stochastic_validation(
            "absolute momentum plateau",
            [primary],
            container="gates",
        )
    if volatility_contraction is not None:
        if (
            volatility_contraction.get("campaign")
            != "VOLATILITY_CONTRACTION_V1"
        ):
            raise ValueError(
                "volatility contraction campaign identity mismatch"
            )
        if int(
            volatility_contraction.get("orders_generated") or 0
        ) != 0:
            raise ValueError(
                "volatility contraction campaign contains orders"
            )
        if bool(volatility_contraction.get("live_ready", False)):
            raise ValueError(
                "volatility contraction campaign contains live permission"
            )
        contraction_registry = (
            volatility_contraction.get("trial_registry") or {}
        )
        if contraction_registry.get("status") != "PASSED":
            raise ValueError(
                "volatility contraction registry audit failed"
            )
        if int(
            contraction_registry.get("unique_trial_count") or 0
        ) != int(
            volatility_contraction.get(
                "registered_unique_trials"
            )
            or 0
        ):
            raise ValueError(
                "volatility contraction trial count mismatch"
            )
        contraction_primary = volatility_contraction.get(
            "primary_result"
        )
        if not isinstance(contraction_primary, dict):
            raise ValueError(
                "volatility contraction primary result missing"
            )
        require_stochastic_validation(
            "volatility contraction",
            [contraction_primary],
            container="gates",
        )
    if multi_alpha_ensemble is not None:
        if (
            multi_alpha_ensemble.get("campaign")
            != "MULTI_ALPHA_ENSEMBLE_V1"
        ):
            raise ValueError(
                "multi-alpha ensemble campaign identity mismatch"
            )
        if int(multi_alpha_ensemble.get("orders_generated") or 0) != 0:
            raise ValueError(
                "multi-alpha ensemble campaign contains orders"
            )
        if bool(multi_alpha_ensemble.get("live_ready", False)):
            raise ValueError(
                "multi-alpha ensemble campaign contains live permission"
            )
        ensemble_registry = multi_alpha_ensemble.get("trial_registry") or {}
        if ensemble_registry.get("status") != "PASSED":
            raise ValueError(
                "multi-alpha ensemble registry audit failed"
            )
        if int(ensemble_registry.get("unique_trial_count") or 0) != int(
            multi_alpha_ensemble.get("registered_unique_trials") or 0
        ):
            raise ValueError(
                "multi-alpha ensemble trial count mismatch"
            )
        ensemble_primary = multi_alpha_ensemble.get("primary_result")
        if not isinstance(ensemble_primary, dict):
            raise ValueError(
                "multi-alpha ensemble primary result missing"
            )
        require_stochastic_validation(
            "multi-alpha ensemble",
            [ensemble_primary],
            container="gates",
        )
    if trend_pullback is not None:
        if trend_pullback.get("campaign") != "TREND_PULLBACK_V1":
            raise ValueError(
                "trend pullback campaign identity mismatch"
            )
        if int(trend_pullback.get("orders_generated") or 0) != 0:
            raise ValueError(
                "trend pullback campaign contains orders"
            )
        if bool(trend_pullback.get("live_ready", False)):
            raise ValueError(
                "trend pullback campaign contains live permission"
            )
        pullback_registry = trend_pullback.get("trial_registry") or {}
        if pullback_registry.get("status") != "PASSED":
            raise ValueError(
                "trend pullback registry audit failed"
            )
        if int(pullback_registry.get("unique_trial_count") or 0) != int(
            trend_pullback.get("registered_unique_trials") or 0
        ):
            raise ValueError(
                "trend pullback trial count mismatch"
            )
        pullback_primary = trend_pullback.get("primary_result")
        if not isinstance(pullback_primary, dict):
            raise ValueError(
                "trend pullback primary result missing"
            )
        require_stochastic_validation(
            "trend pullback",
            [pullback_primary],
            container="gates",
        )
    if portfolio_storm is not None:
        if portfolio_storm.get("status") != "COMPLETED_NOT_PROMOTED":
            raise ValueError("portfolio storm is incomplete")
        if portfolio_storm.get("source_candidate_identity") != identity:
            raise ValueError("portfolio storm candidate identity mismatch")
        if int(portfolio_storm.get("orders_generated") or 0) != 0:
            raise ValueError("portfolio storm contains generated orders")
        if bool(portfolio_storm.get("live_ready", False)):
            raise ValueError("portfolio storm contains live permission")
    if signal_synthesis_storm is not None:
        if signal_synthesis_storm.get("status") != "COMPLETED_SCREENING_NOT_PROMOTED":
            raise ValueError("signal synthesis storm is incomplete")
        if signal_synthesis_storm.get("source_candidate_identity") != identity:
            raise ValueError("signal synthesis storm candidate identity mismatch")
        if int(signal_synthesis_storm.get("orders_generated") or 0) != 0:
            raise ValueError("signal synthesis storm contains generated orders")
        if bool(signal_synthesis_storm.get("live_ready", False)):
            raise ValueError("signal synthesis storm contains live permission")
    if autopilot_state is not None:
        if int(autopilot_state.get("orders_generated") or 0) != 0:
            raise ValueError("autopilot state contains generated orders")
        if bool(autopilot_state.get("paper_candidate_permitted", False)):
            raise ValueError("autopilot state contains paper permission")
        if bool(autopilot_state.get("live_ready", False)):
            raise ValueError("autopilot state contains live permission")
    if feature_store is not None:
        if int(feature_store.get("orders_generated") or 0) != 0:
            raise ValueError("feature store contains generated orders")
        if bool(feature_store.get("paper_candidate_permitted", False)):
            raise ValueError("feature store contains paper permission")
        if bool(feature_store.get("live_ready", False)):
            raise ValueError("feature store contains live permission")
        if autopilot_state is not None and autopilot_state.get(
            "last_feature_store_dataset_id"
        ) != feature_store.get("dataset_id"):
            raise ValueError("autopilot feature store identity mismatch")
    if ai_governance is not None:
        if ai_governance.get("status") != "AI_DEVELOPMENT_EMBARGOED":
            raise ValueError("AI development must remain embargoed")
        if bool(ai_governance.get("eligible", False)):
            raise ValueError("research package cannot contain AI eligibility")
        if bool(
            ai_governance.get(
                "automatic_promotion_permitted",
                False,
            )
        ):
            raise ValueError("AI governance permits automatic promotion")
    if breakout_forward_observers is not None:
        for name, observer_payload in breakout_forward_observers.items():
            if observer_payload.get("source_candidate_identity") != identity:
                raise ValueError(f"breakout forward observer identity mismatch: {name}")
            if int(observer_payload.get("orders_generated") or 0) != 0:
                raise ValueError(f"breakout forward observer contains orders: {name}")
            if bool(
                observer_payload.get(
                    "paper_candidate_permitted",
                    False,
                )
            ) or bool(observer_payload.get("live_ready", False)):
                raise ValueError(f"breakout forward observer contains permission: {name}")
    if any(
        bool(artifact.get(field))
        for artifact in (
            lead,
            external,
            forward,
            institutional_audit,
            observer,
        )
        for field in ("paper_candidate_permitted", "live_ready")
    ):
        raise ValueError("research package cannot contain paper/live permission")
    if not all(bool(result.get("passed")) for result in quality.values()):
        raise ValueError("quality evidence is not fully passing")

    robustness = dict(lead["robustness"])
    economic_pass = bool(
        robustness["economic_gates_passed"]
        and institutional_audit["economic_gates_passed"]
        and external["global_checks"]["all_views_net_positive"]
        and external["global_checks"]["all_views_stressed_positive"]
    )
    statistical_pass = bool(
        robustness["statistical_gates_passed"]
        and institutional_audit["historical_statistical_gates_passed"]
        and all(
            external["global_checks"][name]
            for name in (
                "white_reality_check",
                "hansen_spa",
                "pbo",
                "at_least_one_dsr_pass",
            )
        )
    )
    forward_pass = forward["status"] == "FORWARD_PASS"
    summary = {
        "generated_at": utc_iso(),
        "source_commit": source_commit,
        "status": (
            "RESEARCH_PROMOTION_GATES_PASS"
            if economic_pass and statistical_pass and forward_pass
            else "ECONOMIC_RESEARCH_PASS_STATISTICAL_FAILED_FORWARD_COLLECTING"
        ),
        "candidate": {
            "immutable_identity": identity,
            "strategy_dna_hash": dna_hash,
            "candidate_type": lead["candidate_type"],
            "parameters": lead["parameters"],
            "selection_bias": lead["selection_bias"],
        },
        "source_campaign": {
            "campaign": campaign["campaign"],
            "joint_parameter_trials": campaign["joint_parameter_trials"],
            "total_known_family_trials": campaign["total_known_family_trials"],
            "positive_all_three_periods_descriptive_only": campaign[
                "positive_all_three_periods_descriptive_only"
            ],
            "survivor_count": campaign["survivor_count"],
            "economic_research_lead_count": campaign["economic_research_lead_count"],
            "statistically_qualified_count": campaign["statistically_qualified_count"],
        },
        "strict_policy_reproduction": {
            "status": institutional_audit["status"],
            "execution_identity": institutional_audit["execution_identity"],
            "portfolio_policy": institutional_audit["portfolio_policy"],
            "exposure_semantics": institutional_audit["exposure_semantics"],
            "checks": institutional_audit["checks"],
            "metrics": institutional_audit["normal"]["metrics"],
            "stressed_metrics": institutional_audit["stressed"]["metrics"],
        },
        "continuation_family": {
            "status": continuation["status"],
            "joint_parameter_trials": continuation["joint_parameter_trials"],
            "prior_trials_accounted": continuation["prior_exploratory_trials_accounted"],
            "total_known_trials": continuation["total_known_family_trials"],
            "positive_all_three_periods_descriptive_only": continuation[
                "positive_all_three_periods_descriptive_only"
            ],
            "economic_research_lead_count": continuation["economic_research_lead_count"],
            "statistically_qualified_count": continuation["statistically_qualified_count"],
        },
        "validation": {
            "economic_gates_passed": economic_pass,
            "statistical_gates_passed": statistical_pass,
            "forward_passed": forward_pass,
            "historical_multiple_testing": campaign["multiple_testing"],
            "external_status": external["status"],
            "external_multiple_testing": external["multiple_testing"],
            "forward_status": forward["status"],
            "forward_reason_code": forward.get("reason_code"),
            "forward_requirements": {
                "closed_daily_observations": forward["required_closed_daily_observations"],
                "rebalances": forward["required_rebalances"],
                "regime_coverage": forward["required_regime_coverage"],
            },
            "observer_status": observer["status"],
            "observer_orders_generated": observer["orders_generated"],
            "observer_orders_submitted": observer["orders_submitted"],
        },
        "promotion": {
            "research_lead": economic_pass,
            "research_pass": economic_pass and statistical_pass and forward_pass,
            "shadow_candidate": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "reason": (
                "Economic evidence is positive under strict costs and portfolio "
                "limits, but multiple-testing and genuine future-forward gates fail."
            ),
        },
        "quality": quality,
        "live_orders": 0,
    }
    if capital_utilization is not None:
        summary["capital_utilization"] = {
            "status": capital_utilization["status"],
            "campaign": capital_utilization["campaign"],
            "policies_tested": capital_utilization["policies_tested"],
            "prior_trials_accounted": capital_utilization["prior_trials_accounted"],
            "total_known_trials": capital_utilization["total_known_trials"],
            "multiple_testing": capital_utilization["multiple_testing"],
            "paired_block_bootstrap_vs_frozen_control": capital_utilization[
                "paired_block_bootstrap_vs_frozen_control"
            ],
            "policy_results": [
                {
                    "policy_name": row["policy_name"],
                    "current_operational_limits_compatible": row[
                        "current_operational_limits_compatible"
                    ],
                    "metrics": row["normal"]["metrics"],
                    "gates": row["gates"],
                }
                for row in capital_utilization["policy_results"]
            ],
            "paper_candidates": 0,
            "live_orders": 0,
            "live_ready": False,
        }
    if diversified_rotation is not None:
        summary["diversified_rotation"] = {
            "status": diversified_rotation["status"],
            "campaign": diversified_rotation["campaign"],
            "policies_tested": diversified_rotation["policies_tested"],
            "prior_trials_accounted": diversified_rotation["prior_trials_accounted"],
            "total_known_trials": diversified_rotation["total_known_trials"],
            "multiple_testing": diversified_rotation["multiple_testing"],
            "policy_results": [
                {
                    "policy_name": row["policy_name"],
                    "strategy_dna_hash": row["strategy_dna_hash"],
                    "metrics": row["normal"]["metrics"],
                    "gates": row["gates"],
                }
                for row in diversified_rotation["policy_results"]
            ],
            "paper_candidates": 0,
            "live_orders": 0,
            "live_ready": False,
        }
    if portfolio_breakout is not None:
        summary["portfolio_breakout"] = {
            "status": portfolio_breakout["status"],
            "campaign": portfolio_breakout["campaign"],
            "parameters_tested": portfolio_breakout["parameters_tested"],
            "prior_trials_accounted": portfolio_breakout["prior_trials_accounted"],
            "total_known_trials": portfolio_breakout["total_known_trials"],
            "multiple_testing": portfolio_breakout["multiple_testing"],
            "return_path_audit": portfolio_breakout["return_path_audit"],
            "economic_research_lead_count": portfolio_breakout["economic_research_lead_count"],
            "statistically_qualified_count": portfolio_breakout["statistically_qualified_count"],
            "policy_results": [
                {
                    "policy_name": row["policy_name"],
                    "strategy_dna_hash": row["strategy_dna_hash"],
                    "metrics": row["normal"]["metrics"],
                    "gates": row["gates"],
                }
                for row in portfolio_breakout["policy_results"]
            ],
            "paper_candidates": 0,
            "live_orders": 0,
            "live_ready": False,
        }
    if absolute_momentum is not None:
        summary["absolute_momentum"] = {
            "status": absolute_momentum["status"],
            "campaign": absolute_momentum["campaign"],
            "primary_policy_name": absolute_momentum["primary_policy_name"],
            "primary_strategy_dna_hash": absolute_momentum[
                "primary_strategy_dna_hash"
            ],
            "formal_risk_budget_paths": absolute_momentum[
                "formal_risk_budget_paths"
            ],
            "total_known_trials": absolute_momentum["total_known_trials"],
            "multiple_testing": absolute_momentum["multiple_testing"],
            "primary_result": absolute_momentum["primary_result"],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if absolute_momentum_plateau is not None:
        summary["absolute_momentum_plateau"] = {
            "status": absolute_momentum_plateau["status"],
            "campaign": absolute_momentum_plateau["campaign"],
            "engine_version": absolute_momentum_plateau[
                "engine_version"
            ],
            "generated_trial_count": absolute_momentum_plateau[
                "generated_trial_count"
            ],
            "registered_unique_plateau_trials": (
                absolute_momentum_plateau[
                    "registered_unique_plateau_trials"
                ]
            ),
            "total_known_trials": absolute_momentum_plateau[
                "total_known_trials"
            ],
            "plateau_eligible_count": absolute_momentum_plateau[
                "plateau_eligible_count"
            ],
            "primary_strategy_id": absolute_momentum_plateau[
                "primary_strategy_id"
            ],
            "multiple_testing": absolute_momentum_plateau[
                "multiple_testing"
            ],
            "trial_registry": absolute_momentum_plateau[
                "trial_registry"
            ],
            "primary_result": absolute_momentum_plateau[
                "primary_result"
            ],
            "holdout_status": absolute_momentum_plateau[
                "holdout_status"
            ],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if volatility_contraction is not None:
        summary["volatility_contraction"] = {
            "status": volatility_contraction["status"],
            "campaign": volatility_contraction["campaign"],
            "engine_version": volatility_contraction[
                "engine_version"
            ],
            "generated_trial_count": volatility_contraction[
                "generated_trial_count"
            ],
            "registered_unique_trials": (
                volatility_contraction[
                    "registered_unique_trials"
                ]
            ),
            "total_known_trials": volatility_contraction[
                "total_known_trials"
            ],
            "primary_strategy_id": volatility_contraction[
                "primary_strategy_id"
            ],
            "multiple_testing": volatility_contraction[
                "multiple_testing"
            ],
            "trial_registry": volatility_contraction[
                "trial_registry"
            ],
            "primary_result": volatility_contraction[
                "primary_result"
            ],
            "holdout_status": volatility_contraction[
                "holdout_status"
            ],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if multi_alpha_ensemble is not None:
        summary["multi_alpha_ensemble"] = {
            "status": multi_alpha_ensemble["status"],
            "campaign": multi_alpha_ensemble["campaign"],
            "engine_version": multi_alpha_ensemble["engine_version"],
            "generated_trial_count": multi_alpha_ensemble[
                "generated_trial_count"
            ],
            "registered_unique_trials": multi_alpha_ensemble[
                "registered_unique_trials"
            ],
            "total_known_trials": multi_alpha_ensemble[
                "total_known_trials"
            ],
            "primary_strategy_id": multi_alpha_ensemble[
                "primary_strategy_id"
            ],
            "multiple_testing": multi_alpha_ensemble[
                "multiple_testing"
            ],
            "trial_registry": multi_alpha_ensemble["trial_registry"],
            "primary_result": multi_alpha_ensemble["primary_result"],
            "inherited_selection_bias_pass": multi_alpha_ensemble[
                "inherited_selection_bias_pass"
            ],
            "holdout_status": multi_alpha_ensemble["holdout_status"],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if trend_pullback is not None:
        summary["trend_pullback"] = {
            "status": trend_pullback["status"],
            "campaign": trend_pullback["campaign"],
            "engine_version": trend_pullback["engine_version"],
            "generated_trial_count": trend_pullback[
                "generated_trial_count"
            ],
            "registered_unique_trials": trend_pullback[
                "registered_unique_trials"
            ],
            "total_known_trials": trend_pullback[
                "total_known_trials"
            ],
            "primary_strategy_id": trend_pullback[
                "primary_strategy_id"
            ],
            "multiple_testing": trend_pullback["multiple_testing"],
            "trial_registry": trend_pullback["trial_registry"],
            "primary_result": trend_pullback["primary_result"],
            "holdout_status": trend_pullback["holdout_status"],
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if portfolio_storm is not None:
        summary["portfolio_storm"] = {
            "status": portfolio_storm["status"],
            "campaign": portfolio_storm["campaign"],
            "trial_count": portfolio_storm["trial_count"],
            "prior_known_trials": portfolio_storm["prior_known_trials"],
            "total_known_trials": portfolio_storm["total_known_trials"],
            "pareto_survivor_count": portfolio_storm["pareto_survivor_count"],
            "multiple_testing": portfolio_storm["multiple_testing"],
            "research_pass": False,
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if signal_synthesis_storm is not None:
        summary["signal_synthesis_storm"] = {
            "status": signal_synthesis_storm["status"],
            "campaign": signal_synthesis_storm["campaign"],
            "engine_version": signal_synthesis_storm["engine_version"],
            "trial_count": signal_synthesis_storm["trial_count"],
            "prior_known_trials": signal_synthesis_storm["prior_known_trials"],
            "total_known_trials": signal_synthesis_storm["total_known_trials"],
            "registered_signal_blocks": signal_synthesis_storm["registered_signal_blocks"],
            "executable_signal_blocks": signal_synthesis_storm["executable_signal_blocks"],
            "development_screen": signal_synthesis_storm["development_screen"],
            "pareto_survivor_count": signal_synthesis_storm["pareto_survivor_count"],
            "positive_validation_survivors": (
                signal_synthesis_storm["positive_validation_survivors"]
            ),
            "positive_confirmation_survivors": (
                signal_synthesis_storm["positive_confirmation_survivors"]
            ),
            "canonical_exact_audit": signal_synthesis_storm.get("canonical_exact_audit"),
            "multiple_testing": signal_synthesis_storm["multiple_testing"],
            "research_pass": False,
            "paper_candidates": 0,
            "orders_generated": 0,
            "live_ready": False,
        }
    if autopilot_state is not None:
        summary["autopilot"] = {
            "status": autopilot_state["status"],
            "cycle_count": autopilot_state["cycle_count"],
            "last_cycle_id": autopilot_state.get("last_cycle_id"),
            "last_completed_at": autopilot_state.get("last_completed_at"),
            "last_data_fingerprint": autopilot_state.get("last_data_fingerprint"),
            "last_research_at": autopilot_state.get("last_research_at"),
            "last_research_data_fingerprint": autopilot_state.get("last_research_data_fingerprint"),
            "last_feature_store_dataset_id": autopilot_state.get("last_feature_store_dataset_id"),
            "research_ran": autopilot_state.get("research_ran"),
            "research_reason": autopilot_state.get("research_reason"),
            "degradation": autopilot_state.get("degradation"),
            "persistent_kill_switch": autopilot_degradation,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    if feature_store is not None:
        summary["feature_store"] = {
            "schema_version": feature_store["schema_version"],
            "dataset_id": feature_store["dataset_id"],
            "frequency": feature_store["frequency"],
            "assets": feature_store["assets"],
            "feature_names": feature_store["feature_names"],
            "shapes": feature_store["shapes"],
            "per_asset": feature_store["per_asset"],
            "causality": feature_store["causality"],
            "tensor_sha256": feature_store["tensor_sha256"],
            "research_only": True,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    if ai_governance is not None:
        summary["ai_governance"] = ai_governance
    if breakout_forward_observers is not None:
        per_policy = {
            str(payload.get("policy_name") or name): {
                "strategy_dna_hash": payload["strategy_dna_hash"],
                "forward_observer_schema_version": payload.get("forward_observer_schema_version"),
                "forward_summary": payload.get("forward_summary"),
                "degradation_observation": payload.get("degradation_observation"),
                "orders_generated": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
            for name, payload in sorted(breakout_forward_observers.items())
        }
        summary["breakout_forward_observers"] = {
            "status": "FROZEN_FORWARD_RESEARCH",
            "policy_count": len(per_policy),
            "per_policy": per_policy,
            "all_formal_performance_pass": bool(per_policy)
            and all(
                bool(
                    (row.get("forward_summary") or {}).get(
                        "forward_performance_pass",
                        False,
                    )
                )
                for row in per_policy.values()
            ),
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    return summary


def _artifact_paths(settings: Settings) -> dict[str, Path]:
    reports = settings.paths.lab_dir / "reports"
    candidates = settings.paths.lab_dir / "candidates"
    paths = {
        "rotation_research_lead_v1.json": (candidates / "rotation_research_lead_v1.json"),
        "cross_sectional_ensemble_v1.json": (reports / "cross_sectional_ensemble_v1.json"),
        "cross_sectional_ensemble_v1.csv": (reports / "cross_sectional_ensemble_v1.csv"),
        "rotation_external_holdouts_v1.json": (reports / "rotation_external_holdouts_v1.json"),
        "rotation_forward_validation_v1.json": (reports / "rotation_forward_validation_v1.json"),
        "rotation_institutional_audit_v2.json": (reports / "rotation_institutional_audit_v2.json"),
        "rotation_institutional_audit_v2.csv": (reports / "rotation_institutional_audit_v2.csv"),
        "cross_sectional_institutional_v2.json": (
            reports / "cross_sectional_institutional_v2.json"
        ),
        "cross_sectional_institutional_v2.csv": (reports / "cross_sectional_institutional_v2.csv"),
        "rotation_forward_observer_v2.json": (reports / "rotation_forward_observer_v2.json"),
        "capital_utilization_campaign_v1.json": (reports / "capital_utilization_campaign_v1.json"),
        "capital_utilization_campaign_v1.csv": (reports / "capital_utilization_campaign_v1.csv"),
        "diversified_rotation_campaign_v1.json": (
            reports / "diversified_rotation_campaign_v1.json"
        ),
        "diversified_rotation_campaign_v1.csv": (reports / "diversified_rotation_campaign_v1.csv"),
        "portfolio_breakout_campaign_v1.json": (reports / "portfolio_breakout_campaign_v1.json"),
        "portfolio_breakout_campaign_v1.csv": (reports / "portfolio_breakout_campaign_v1.csv"),
        "absolute_momentum_campaign_v1.json": (
            reports / "absolute_momentum_campaign_v1.json"
        ),
        "absolute_momentum_campaign_v1.csv": (
            reports / "absolute_momentum_campaign_v1.csv"
        ),
        "absolute_momentum_plateau_campaign_v1.json": (
            reports / "absolute_momentum_plateau_campaign_v1.json"
        ),
        "absolute_momentum_plateau_campaign_v1.csv": (
            reports / "absolute_momentum_plateau_campaign_v1.csv"
        ),
        "absolute_momentum_plateau_plan_v1.json": (
            reports / "absolute_momentum_plateau_plan_v1.json"
        ),
        "volatility_contraction_campaign_v1.json": (
            reports / "volatility_contraction_campaign_v1.json"
        ),
        "volatility_contraction_campaign_v1.csv": (
            reports / "volatility_contraction_campaign_v1.csv"
        ),
        "volatility_contraction_plan_v1.json": (
            reports / "volatility_contraction_plan_v1.json"
        ),
        "multi_alpha_ensemble_campaign_v1.json": (
            reports / "multi_alpha_ensemble_campaign_v1.json"
        ),
        "multi_alpha_ensemble_campaign_v1.csv": (
            reports / "multi_alpha_ensemble_campaign_v1.csv"
        ),
        "multi_alpha_ensemble_plan_v1.json": (
            reports / "multi_alpha_ensemble_plan_v1.json"
        ),
        "trend_pullback_campaign_v1.json": (
            reports / "trend_pullback_campaign_v1.json"
        ),
        "trend_pullback_campaign_v1.csv": (
            reports / "trend_pullback_campaign_v1.csv"
        ),
        "trend_pullback_plan_v1.json": (
            reports / "trend_pullback_plan_v1.json"
        ),
        "forward_ledger_preflight_v1.json": (
            reports / "forward_ledger_preflight_v1.json"
        ),
        "portfolio_breakout_forward_observer_v1.json": (
            reports / "portfolio_breakout_forward_observer_v1.json"
        ),
        "portfolio_storm_plan_v1.json": (reports / "portfolio_storm_plan_v1.json"),
        "portfolio_storm_report_v1.json": (reports / "portfolio_storm_report_v1.json"),
        "portfolio_storm_returns_v1.npz": (reports / "portfolio_storm_returns_v1.npz"),
        "signal_synthesis_storm_plan_v1.json": (reports / "signal_synthesis_storm_plan_v1.json"),
        "signal_synthesis_storm_report_v1.json": (
            reports / "signal_synthesis_storm_report_v1.json"
        ),
        "signal_synthesis_storm_returns_v1.npz": (
            reports / "signal_synthesis_storm_returns_v1.npz"
        ),
        "signal_synthesis_storm_plan_v2.json": (reports / "signal_synthesis_storm_plan_v2.json"),
        "signal_synthesis_storm_report_v2.json": (
            reports / "signal_synthesis_storm_report_v2.json"
        ),
        "signal_synthesis_storm_returns_v2.npz": (
            reports / "signal_synthesis_storm_returns_v2.npz"
        ),
    }
    observer_directory = settings.paths.lab_dir / "observers" / "capital_utilization_v1"
    for observer in sorted(observer_directory.glob("*.json")):
        paths[f"capital_observer_{observer.name}"] = observer
    diversified_observer_directory = (
        settings.paths.lab_dir / "observers" / "diversified_rotation_v1"
    )
    for observer in sorted(diversified_observer_directory.glob("*.json")):
        paths[f"diversified_observer_{observer.name}"] = observer
    breakout_observer_directory = settings.paths.lab_dir / "observers" / "portfolio_breakout_v1"
    for observer in sorted(breakout_observer_directory.glob("*.json")):
        paths[f"breakout_observer_{observer.name}"] = observer
    absolute_momentum_observer_directory = (
        settings.paths.lab_dir / "observers" / "absolute_momentum_v1"
    )
    for observer in sorted(absolute_momentum_observer_directory.glob("*.json")):
        paths[f"absolute_momentum_observer_{observer.name}"] = observer
    plateau_observer_directory = (
        settings.paths.lab_dir
        / "observers"
        / "absolute_momentum_plateau_v1"
    )
    for observer in sorted(plateau_observer_directory.glob("*.json")):
        paths[f"absolute_momentum_plateau_observer_{observer.name}"] = observer
    plateau_registry_directory = (
        settings.paths.lab_dir
        / "strategy_registry"
        / "absolute_momentum_plateau_v1"
    )
    registry_index = plateau_registry_directory / "index.json"
    paths["absolute_momentum_plateau_registry_index.json"] = (
        registry_index
    )
    for record in sorted(
        (plateau_registry_directory / "records").glob("*.json")
    ):
        paths[f"absolute_momentum_plateau_trial_{record.name}"] = record
    contraction_observer_directory = (
        settings.paths.lab_dir
        / "observers"
        / "volatility_contraction_v1"
    )
    for observer in sorted(
        contraction_observer_directory.glob("*.json")
    ):
        paths[f"volatility_contraction_observer_{observer.name}"] = (
            observer
        )
    contraction_registry_directory = (
        settings.paths.lab_dir
        / "strategy_registry"
        / "volatility_contraction_v1"
    )
    paths["volatility_contraction_registry_index.json"] = (
        contraction_registry_directory / "index.json"
    )
    for record in sorted(
        (contraction_registry_directory / "records").glob("*.json")
    ):
        paths[f"volatility_contraction_trial_{record.name}"] = (
            record
        )
    ensemble_observer_directory = (
        settings.paths.lab_dir
        / "observers"
        / "multi_alpha_ensemble_v1"
    )
    for observer in sorted(ensemble_observer_directory.glob("*.json")):
        paths[f"multi_alpha_ensemble_observer_{observer.name}"] = observer
    ensemble_registry_directory = (
        settings.paths.lab_dir
        / "strategy_registry"
        / "multi_alpha_ensemble_v1"
    )
    paths["multi_alpha_ensemble_registry_index.json"] = (
        ensemble_registry_directory / "index.json"
    )
    for record in sorted(
        (ensemble_registry_directory / "records").glob("*.json")
    ):
        paths[f"multi_alpha_ensemble_trial_{record.name}"] = record
    pullback_observer_directory = (
        settings.paths.lab_dir
        / "observers"
        / "trend_pullback_v1"
    )
    for observer in sorted(pullback_observer_directory.glob("*.json")):
        paths[f"trend_pullback_observer_{observer.name}"] = observer
    pullback_registry_directory = (
        settings.paths.lab_dir
        / "strategy_registry"
        / "trend_pullback_v1"
    )
    paths["trend_pullback_registry_index.json"] = (
        pullback_registry_directory / "index.json"
    )
    for record in sorted(
        (pullback_registry_directory / "records").glob("*.json")
    ):
        paths[f"trend_pullback_trial_{record.name}"] = record
    autopilot_directory = settings.paths.lab_dir / "autopilot"
    autopilot_state = autopilot_directory / "state.json"
    autopilot_degradation = autopilot_directory / "degradation_state.json"
    if autopilot_state.is_file():
        paths["autopilot_state.json"] = autopilot_state
        state = read_json(autopilot_state)
        last_cycle = Path(str(state.get("last_cycle_path") or ""))
        if last_cycle.is_file():
            paths["autopilot_latest_cycle.json"] = last_cycle
    if autopilot_degradation.is_file():
        paths["autopilot_degradation_state.json"] = autopilot_degradation
    feature_store_directory = settings.paths.lab_dir / "feature_store" / "portfolio_daily_v1"
    feature_store_manifest = feature_store_directory / "latest.manifest.json"
    feature_store_tensor = feature_store_directory / "latest.npz"
    if feature_store_manifest.is_file() and feature_store_tensor.is_file():
        paths["feature_store_latest.manifest.json"] = feature_store_manifest
        paths["feature_store_latest.npz"] = feature_store_tensor
    ai_governance_report = reports / "ai_governance_status_v1.json"
    if ai_governance_report.is_file():
        paths["ai_governance_status_v1.json"] = ai_governance_report
    storm_epoch_index = settings.paths.lab_dir / "storm_epochs" / "index.json"
    if storm_epoch_index.is_file():
        paths["portfolio_storm_epoch_index.json"] = storm_epoch_index
    signal_storm_epoch_index = settings.paths.lab_dir / "signal_storm_epochs" / "index.json"
    if signal_storm_epoch_index.is_file():
        paths["signal_storm_epoch_index.json"] = signal_storm_epoch_index
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"acceptance evidence is missing: {missing}")
    return paths


def _readme(summary: dict[str, Any]) -> str:
    metrics = summary["strict_policy_reproduction"]["metrics"]
    return f"""# Crypto rotation research acceptance package

This package preserves the committed source and complete evidence for the
frozen cross-sectional rotation research lead.

## Verified outcome

- Strict-policy net return: {metrics["net_return"]:.4%}
- Annualized return: {metrics["annualized_return"]:.4%}
- Sharpe ratio: {metrics["sharpe"]:.4f}
- Maximum drawdown: {metrics["maximum_drawdown"]:.4%}
- Weekly effective sample size: {metrics["portfolio_period_effective_sample_size"]}
- Closed holding episodes: {metrics["closed_position_episodes"]}

The result is economically positive after normal and stressed costs. Statistical
multiple-testing gates and genuine future-forward requirements do not pass.
Shadow, paper and live promotion therefore remain disabled.

`ACCEPTANCE.json` is the machine-readable conclusion. `MANIFEST.json` inventories
all evidence with SHA-256. `SOURCE_COMMIT.txt` and `source-*.zip` identify the
exact committed implementation.
"""


def build_rotation_acceptance_package(settings: Settings) -> dict[str, Any]:
    """Run quality checks and package a clean committed source with evidence."""

    project_root = settings.paths.project_root
    source_commit = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
    ).stdout.strip()
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("commit all source changes before acceptance packaging")
    paths = _artifact_paths(settings)
    quality = _quality_evidence(project_root)
    payloads = {name: read_json(path) for name, path in paths.items() if path.suffix == ".json"}
    summary = build_acceptance_summary(
        source_commit=source_commit,
        lead=payloads["rotation_research_lead_v1.json"],
        campaign=payloads["cross_sectional_ensemble_v1.json"],
        external=payloads["rotation_external_holdouts_v1.json"],
        forward=payloads["rotation_forward_validation_v1.json"],
        institutional_audit=payloads["rotation_institutional_audit_v2.json"],
        continuation=payloads["cross_sectional_institutional_v2.json"],
        observer=payloads["rotation_forward_observer_v2.json"],
        quality=quality,
        capital_utilization=payloads["capital_utilization_campaign_v1.json"],
        diversified_rotation=payloads["diversified_rotation_campaign_v1.json"],
        portfolio_breakout=payloads["portfolio_breakout_campaign_v1.json"],
        absolute_momentum=payloads["absolute_momentum_campaign_v1.json"],
        absolute_momentum_plateau=payloads[
            "absolute_momentum_plateau_campaign_v1.json"
        ],
        volatility_contraction=payloads[
            "volatility_contraction_campaign_v1.json"
        ],
        multi_alpha_ensemble=payloads[
            "multi_alpha_ensemble_campaign_v1.json"
        ],
        trend_pullback=payloads["trend_pullback_campaign_v1.json"],
        portfolio_storm=payloads["portfolio_storm_report_v1.json"],
        signal_synthesis_storm=payloads["signal_synthesis_storm_report_v2.json"],
        autopilot_state=payloads.get("autopilot_state.json"),
        autopilot_degradation=payloads.get("autopilot_degradation_state.json"),
        feature_store=payloads.get("feature_store_latest.manifest.json"),
        ai_governance=payloads.get("ai_governance_status_v1.json"),
        breakout_forward_observers={
            name: payload
            for name, payload in payloads.items()
            if name.startswith("breakout_observer_")
        },
    )
    evidence_hash = stable_hash(
        {
            "source_commit": source_commit,
            "artifacts": {name: sha256_file(path) for name, path in sorted(paths.items())},
        },
        length=12,
    )
    packages = settings.paths.output_dir / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    package_name = f"crypto_rotation_research_{source_commit[:7]}_{evidence_hash[:8]}"
    target = packages / package_name
    archive = packages / f"{package_name}.zip"
    checksum = packages / f"{package_name}.zip.sha256"
    if target.is_dir() and archive.is_file() and checksum.is_file():
        return {
            "status": summary["status"],
            "package": str(target),
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "source_commit": source_commit,
            "reused": True,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }

    temporary = Path(tempfile.mkdtemp(prefix=f".{package_name}.", dir=packages))
    try:
        artifact_dir = temporary / "artifacts"
        artifact_dir.mkdir(parents=True)
        for name, source in paths.items():
            shutil.copy2(source, artifact_dir / name)
        atomic_write_json(temporary / "ACCEPTANCE.json", summary)
        atomic_write_text(temporary / "README.md", _readme(summary))
        atomic_write_text(temporary / "SOURCE_COMMIT.txt", f"{source_commit}\n")
        source_archive = temporary / f"source-{source_commit[:7]}.zip"
        _run(
            [
                "git",
                "archive",
                "--format=zip",
                "--output",
                str(source_archive),
                source_commit,
            ],
            cwd=project_root,
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        atomic_write_json(
            temporary / "MANIFEST.json",
            {
                "source_commit": source_commit,
                "evidence_hash": evidence_hash,
                "files": [
                    {
                        "path": path.relative_to(temporary).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in files
                ],
            },
        )
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    created_archive = Path(
        shutil.make_archive(
            str(packages / package_name),
            "zip",
            root_dir=packages,
            base_dir=package_name,
        )
    )
    archive_hash = sha256_file(created_archive)
    atomic_write_text(checksum, f"{archive_hash}  {created_archive.name}\n")
    return {
        "status": summary["status"],
        "package": str(target),
        "archive": str(created_archive),
        "archive_sha256": archive_hash,
        "source_commit": source_commit,
        "reused": False,
        "quality": quality,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


__all__ = ["build_acceptance_summary", "build_rotation_acceptance_package"]
