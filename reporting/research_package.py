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
            value.strip()
            for value in (completed.stdout, completed.stderr)
            if value.strip()
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
    autopilot_state: dict[str, Any] | None = None,
    autopilot_degradation: dict[str, Any] | None = None,
    feature_store: dict[str, Any] | None = None,
    breakout_forward_observers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cross-check evidence identities and state the strongest honest conclusion."""

    identity = str(lead["immutable_identity"])
    dna_hash = str(lead["strategy_dna_hash"])
    if lead.get("status") != "FROZEN_RESEARCH_LEAD":
        raise ValueError("candidate is not a frozen research lead")
    if campaign.get("status") != "COMPLETED":
        raise ValueError("source ensemble campaign is incomplete")
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
    if diversified_rotation is not None:
        if diversified_rotation.get("source_candidate_identity") != identity:
            raise ValueError("diversified rotation candidate identity mismatch")
        if (
            diversified_rotation.get("source_frozen_strategy_dna_hash")
            != dna_hash
        ):
            raise ValueError("diversified rotation source DNA mismatch")
    if portfolio_breakout is not None:
        if portfolio_breakout.get("source_candidate_identity") != identity:
            raise ValueError("portfolio breakout candidate identity mismatch")
        if portfolio_breakout.get("source_frozen_strategy_dna_hash") != dna_hash:
            raise ValueError("portfolio breakout source DNA mismatch")
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
        if (
            autopilot_state is not None
            and autopilot_state.get("last_feature_store_dataset_id")
            != feature_store.get("dataset_id")
        ):
            raise ValueError("autopilot feature store identity mismatch")
    if breakout_forward_observers is not None:
        for name, observer_payload in breakout_forward_observers.items():
            if (
                observer_payload.get("source_candidate_identity")
                != identity
            ):
                raise ValueError(
                    f"breakout forward observer identity mismatch: {name}"
                )
            if int(observer_payload.get("orders_generated") or 0) != 0:
                raise ValueError(
                    f"breakout forward observer contains orders: {name}"
                )
            if bool(
                observer_payload.get(
                    "paper_candidate_permitted",
                    False,
                )
            ) or bool(observer_payload.get("live_ready", False)):
                raise ValueError(
                    f"breakout forward observer contains permission: {name}"
                )
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
            "prior_trials_accounted": continuation[
                "prior_exploratory_trials_accounted"
            ],
            "total_known_trials": continuation["total_known_family_trials"],
            "positive_all_three_periods_descriptive_only": continuation[
                "positive_all_three_periods_descriptive_only"
            ],
            "economic_research_lead_count": continuation[
                "economic_research_lead_count"
            ],
            "statistically_qualified_count": continuation[
                "statistically_qualified_count"
            ],
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
                "closed_daily_observations": forward[
                    "required_closed_daily_observations"
                ],
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
            "prior_trials_accounted": capital_utilization[
                "prior_trials_accounted"
            ],
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
            "prior_trials_accounted": diversified_rotation[
                "prior_trials_accounted"
            ],
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
            "prior_trials_accounted": portfolio_breakout[
                "prior_trials_accounted"
            ],
            "total_known_trials": portfolio_breakout["total_known_trials"],
            "multiple_testing": portfolio_breakout["multiple_testing"],
            "return_path_audit": portfolio_breakout["return_path_audit"],
            "economic_research_lead_count": portfolio_breakout[
                "economic_research_lead_count"
            ],
            "statistically_qualified_count": portfolio_breakout[
                "statistically_qualified_count"
            ],
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
    if autopilot_state is not None:
        summary["autopilot"] = {
            "status": autopilot_state["status"],
            "cycle_count": autopilot_state["cycle_count"],
            "last_cycle_id": autopilot_state.get("last_cycle_id"),
            "last_completed_at": autopilot_state.get(
                "last_completed_at"
            ),
            "last_data_fingerprint": autopilot_state.get(
                "last_data_fingerprint"
            ),
            "last_research_at": autopilot_state.get("last_research_at"),
            "last_research_data_fingerprint": autopilot_state.get(
                "last_research_data_fingerprint"
            ),
            "last_feature_store_dataset_id": autopilot_state.get(
                "last_feature_store_dataset_id"
            ),
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
    if breakout_forward_observers is not None:
        per_policy = {
            str(payload.get("policy_name") or name): {
                "strategy_dna_hash": payload["strategy_dna_hash"],
                "forward_observer_schema_version": payload.get(
                    "forward_observer_schema_version"
                ),
                "forward_summary": payload.get("forward_summary"),
                "degradation_observation": payload.get(
                    "degradation_observation"
                ),
                "orders_generated": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
            for name, payload in sorted(
                breakout_forward_observers.items()
            )
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
        "rotation_research_lead_v1.json": (
            candidates / "rotation_research_lead_v1.json"
        ),
        "cross_sectional_ensemble_v1.json": (
            reports / "cross_sectional_ensemble_v1.json"
        ),
        "cross_sectional_ensemble_v1.csv": (
            reports / "cross_sectional_ensemble_v1.csv"
        ),
        "rotation_external_holdouts_v1.json": (
            reports / "rotation_external_holdouts_v1.json"
        ),
        "rotation_forward_validation_v1.json": (
            reports / "rotation_forward_validation_v1.json"
        ),
        "rotation_institutional_audit_v2.json": (
            reports / "rotation_institutional_audit_v2.json"
        ),
        "rotation_institutional_audit_v2.csv": (
            reports / "rotation_institutional_audit_v2.csv"
        ),
        "cross_sectional_institutional_v2.json": (
            reports / "cross_sectional_institutional_v2.json"
        ),
        "cross_sectional_institutional_v2.csv": (
            reports / "cross_sectional_institutional_v2.csv"
        ),
        "rotation_forward_observer_v2.json": (
            reports / "rotation_forward_observer_v2.json"
        ),
        "capital_utilization_campaign_v1.json": (
            reports / "capital_utilization_campaign_v1.json"
        ),
        "capital_utilization_campaign_v1.csv": (
            reports / "capital_utilization_campaign_v1.csv"
        ),
        "diversified_rotation_campaign_v1.json": (
            reports / "diversified_rotation_campaign_v1.json"
        ),
        "diversified_rotation_campaign_v1.csv": (
            reports / "diversified_rotation_campaign_v1.csv"
        ),
        "portfolio_breakout_campaign_v1.json": (
            reports / "portfolio_breakout_campaign_v1.json"
        ),
        "portfolio_breakout_campaign_v1.csv": (
            reports / "portfolio_breakout_campaign_v1.csv"
        ),
        "portfolio_breakout_forward_observer_v1.json": (
            reports / "portfolio_breakout_forward_observer_v1.json"
        ),
    }
    observer_directory = (
        settings.paths.lab_dir / "observers" / "capital_utilization_v1"
    )
    for observer in sorted(observer_directory.glob("*.json")):
        paths[f"capital_observer_{observer.name}"] = observer
    diversified_observer_directory = (
        settings.paths.lab_dir / "observers" / "diversified_rotation_v1"
    )
    for observer in sorted(diversified_observer_directory.glob("*.json")):
        paths[f"diversified_observer_{observer.name}"] = observer
    breakout_observer_directory = (
        settings.paths.lab_dir / "observers" / "portfolio_breakout_v1"
    )
    for observer in sorted(breakout_observer_directory.glob("*.json")):
        paths[f"breakout_observer_{observer.name}"] = observer
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
    feature_store_directory = (
        settings.paths.lab_dir
        / "feature_store"
        / "portfolio_daily_v1"
    )
    feature_store_manifest = (
        feature_store_directory / "latest.manifest.json"
    )
    feature_store_tensor = feature_store_directory / "latest.npz"
    if feature_store_manifest.is_file() and feature_store_tensor.is_file():
        paths["feature_store_latest.manifest.json"] = (
            feature_store_manifest
        )
        paths["feature_store_latest.npz"] = feature_store_tensor
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
    payloads = {
        name: read_json(path)
        for name, path in paths.items()
        if path.suffix == ".json"
    }
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
        autopilot_state=payloads.get("autopilot_state.json"),
        autopilot_degradation=payloads.get(
            "autopilot_degradation_state.json"
        ),
        feature_store=payloads.get(
            "feature_store_latest.manifest.json"
        ),
        breakout_forward_observers={
            name: payload
            for name, payload in payloads.items()
            if name.startswith("breakout_observer_")
        },
    )
    evidence_hash = stable_hash(
        {
            "source_commit": source_commit,
            "artifacts": {
                name: sha256_file(path) for name, path in sorted(paths.items())
            },
        },
        length=12,
    )
    packages = settings.paths.output_dir / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    package_name = (
        f"crypto_rotation_research_{source_commit[:7]}_{evidence_hash[:8]}"
    )
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
