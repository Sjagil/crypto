"""Nine-repository usage registry and reproducible master-map artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reporting.reference_integration_phase_a import (
    REFERENCE_ASSIGNMENTS,
    snapshot_reference,
)
from utils.common import atomic_write_json, sha256_file, stable_hash, utc_iso

SCHEMA_VERSION = "reference_repository_master_map_v1"


@dataclass(frozen=True, slots=True)
class ReferenceUsage:
    repo: str
    algorithms_or_patterns: tuple[str, ...]
    compatibility: str
    native_call_sites: tuple[str, ...]
    test_ids: tuple[str, ...]
    usage_events: tuple[str, ...]
    measurable_value: str
    status: str = "APPLIED_NATIVE"

    @property
    def usage_count(self) -> int:
        return len(self.usage_events)


REFERENCE_USAGE_REGISTRY: tuple[ReferenceUsage, ...] = (
    ReferenceUsage(
        "nautilus_trader",
        ("event-sourced execution state", "idempotent fill reconciliation", "deterministic replay"),
        "Native Python reducer; no Rust/LGPL runtime dependency.",
        ("execution/canonical_state.py", "execution/state_migration.py"),
        ("tests/test_canonical_execution_state.py",),
        ("canonical_order_state", "duplicate_fill_guard", "replay_reconciliation"),
        "One canonical replayable financial-state owner with duplicate-fill suppression.",
    ),
    ReferenceUsage(
        "lean",
        ("Alpha-to-PortfolioTarget boundary", "target delta semantics", "risk before execution"),
        "Native Pydantic spot contracts; no .NET runtime dependency.",
        ("portfolio/contracts.py", "portfolio/targets.py", "portfolio/buy_chain.py"),
        ("tests/test_portfolio_contracts.py", "tests/test_portfolio_targets.py"),
        ("investment_intent_contract", "portfolio_target_delta", "risk_execution_boundary"),
        "Strategies can express desired exposure without receiving order authority.",
    ),
    ReferenceUsage(
        "vectorbt",
        ("vectorized Stage-0 screen", "parameter broadcasting", "cost-aware early rejection"),
        "Native pandas/numpy screen; Commons-Clause code is not copied or imported.",
        ("research/research_factory.py",),
        ("tests/test_research_factory.py", "tests/test_reference_integrations.py"),
        ("stage0_signal_matrix", "stage0_cost_screen", "numeric_book_probe"),
        "Cheap candidates are rejected before exact event-driven validation.",
    ),
    ReferenceUsage(
        "pybroker",
        ("chronological walk-forward", "per-window retraining", "train/test isolation"),
        "Native validation manifest; Commons-Clause code is only probed in isolation.",
        ("research/research_factory.py", "research/optimization.py"),
        ("tests/test_research_factory.py", "tests/test_reference_integrations.py"),
        ("walk_forward_manifest", "purge_embargo_folds", "return_vector_probe"),
        "Random-shuffle leakage is excluded and fold boundaries are reproducible.",
    ),
    ReferenceUsage(
        "qlib",
        ("Dataset-Experiment-Recorder lifecycle", "content-addressed model artifacts"),
        "Native immutable registry; MIT reference remains optional.",
        ("ml/contracts.py", "ml/registry.py", "ml/lifecycle.py"),
        ("tests/test_ml_lifecycle.py", "tests/test_reference_integrations.py"),
        ("dataset_manifest", "model_registry", "indexed_aggregation_probe"),
        "Dataset, label, model and prediction provenance is separated and hash-addressed.",
    ),
    ReferenceUsage(
        "freqtrade",
        ("lookahead analysis", "recursive warmup stability", "model expiry"),
        "Native causal lifecycle; GPL source is not copied or imported by production.",
        ("ml/lifecycle.py", "research/research_factory.py"),
        ("tests/test_ml_lifecycle.py", "tests/test_reference_integrations.py"),
        ("point_in_time_gate", "recursive_stability_gate", "timeframe_probe"),
        "Stale, leaky or warmup-unstable models fail closed to SHADOW/no-trade.",
    ),
    ReferenceUsage(
        "finrl-trading",
        ("explicit RL state/action/reward", "deterministic evaluation", "baseline comparison"),
        "Dependency-free native environment now; Gymnasium/PyTorch/SB3 remain optional and blocked.",
        ("rl/position_management.py",),
        ("tests/test_rl_position_management.py", "tests/test_reference_integrations.py"),
        ("position_state_contract", "monotone_spot_actions", "rl_eligibility_gate", "robust_zscore_probe"),
        "RL cannot add leverage or short and cannot start training before evidence/dependency gates pass.",
    ),
    ReferenceUsage(
        "rd-agent",
        ("Hypothesis-Experiment-Feedback loop", "append-only trace", "research memory"),
        "Native immutable Pydantic ledger; no RD-Agent/LLM runtime dependency.",
        ("research/autonomous_rd.py",),
        ("tests/test_autonomous_research_loop.py", "tests/test_reference_integrations.py"),
        ("hypothesis_record", "preregistered_experiment", "feedback_trace", "feedback_probe"),
        "Autonomous iteration is reproducible but has zero strategy-promotion or exchange authority.",
    ),
    ReferenceUsage(
        "tradingagents",
        ("bull/bear evidence separation", "risk challenge", "typed decision hand-off"),
        "Native deterministic schemas; no graph, prompt, provider or agent runtime dependency.",
        ("core/structured_market_intelligence.py",),
        ("tests/test_structured_market_intelligence.py", "tests/test_reference_integrations.py"),
        ("multi_perspective_claims", "conflict_fail_closed", "shadow_ai_intent_boundary", "schema_probe"),
        "Structured intelligence enters the canonical chain only as a SHADOW NO_TRADE intent.",
    ),
)


def _count_python_files(root: Path) -> tuple[int, int]:
    files = tuple(root.rglob("*.py"))
    tests = tuple(path for path in files if "test" in path.name.lower() or "tests" in path.parts)
    return len(files), len(tests)


def build_reference_master_map(
    workspace: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    usages = {item.repo: item for item in REFERENCE_USAGE_REGISTRY}
    assignments = {item.repo: item for item in REFERENCE_ASSIGNMENTS}
    if set(usages) != set(assignments):
        raise ValueError("reference assignments and usage registry must cover the same repositories")
    repositories: list[dict[str, Any]] = []
    for repo in sorted(assignments):
        assignment = assignments[repo]
        usage = usages[repo]
        snapshot = snapshot_reference(workspace, assignment)
        python_files, test_files = _count_python_files(Path(snapshot["path"]))
        missing_call_sites = [
            path for path in usage.native_call_sites if not (workspace / path).is_file()
        ]
        missing_test_ids = [path for path in usage.test_ids if not (workspace / path).is_file()]
        row = {
            "repo": repo,
            "commit": snapshot["actual_commit"],
            "tree_hash": snapshot["actual_tree"],
            "branch": snapshot["actual_branch"],
            "remote": snapshot["actual_remote"],
            "license": assignment.license,
            "license_sha256": snapshot["license_sha256"],
            "python_file_count": python_files,
            "test_file_count": test_files,
            "relevant_source_evidence": snapshot["source_evidence"],
            "primary_responsibility": assignment.primary_responsibility,
            "integration_mode": assignment.integration_mode,
            **asdict(usage),
            "usage_count": usage.usage_count,
            "missing_native_call_sites": missing_call_sites,
            "missing_test_ids": missing_test_ids,
            "native_call_site_sha256": {
                path: sha256_file(workspace / path)
                for path in usage.native_call_sites
                if (workspace / path).is_file()
            },
            "test_sha256": {
                path: sha256_file(workspace / path)
                for path in usage.test_ids
                if (workspace / path).is_file()
            },
            "reference_unchanged": bool(
                snapshot["reference_clean"]
                and snapshot["commit_verified"]
                and snapshot["tree_verified"]
            ),
            "complete": bool(
                usage.usage_count > 0
                and not missing_call_sites
                and not missing_test_ids
                and snapshot["source_symbols_verified"]
                and snapshot["reference_clean"]
                and snapshot["commit_verified"]
                and snapshot["tree_verified"]
                and snapshot["license_verified"]
            ),
        }
        repositories.append(row)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "workspace": str(workspace),
        "repository_count": len(repositories),
        "repositories": repositories,
        "all_nine_present": len(repositories) == 9,
        "all_usage_counts_positive": all(row["usage_count"] > 0 for row in repositories),
        "all_call_sites_and_tests_present": all(
            not row["missing_native_call_sites"] and not row["missing_test_ids"]
            for row in repositories
        ),
        "all_references_unchanged": all(row["reference_unchanged"] for row in repositories),
        "all_integrations_complete": all(row["complete"] for row in repositories),
        "execution_authority": False,
        "orders_generated": 0,
        "orders_submitted": 0,
        "automatic_live_launch_permitted": False,
    }
    payload["artifact_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "generated_at"},
        length=64,
    )
    destination = output_path or (
        workspace / "output" / "reference_integration" / "reference_master_map.json"
    )
    atomic_write_json(destination, payload)
    return {**payload, "artifact_path": str(destination.resolve())}


__all__ = [
    "REFERENCE_USAGE_REGISTRY",
    "ReferenceUsage",
    "SCHEMA_VERSION",
    "build_reference_master_map",
]
