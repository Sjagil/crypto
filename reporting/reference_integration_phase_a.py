"""Forensic Phase A inventory for the nine local reference repositories.

This module is intentionally read-only with respect to the reference clones and
all exchange/runtime state.  It records exact local provenance, inspects named
upstream source locations, and maps reference concepts onto existing native
owners before any migration work is allowed to start.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from utils.common import atomic_write_json, sha256_file, stable_hash, utc_iso

SCHEMA_VERSION = "reference_integration_phase_a_v2"


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    path: str
    symbols: tuple[str, ...]
    concept: str


@dataclass(frozen=True, slots=True)
class ReferenceAssignment:
    repo: str
    expected_commit: str
    expected_tree: str
    expected_branch: str
    expected_remote: str
    license_file: str
    license: str
    expected_license_sha256: str
    primary_responsibility: str
    integration_mode: str
    used_concepts: tuple[str, ...]
    forbidden_copying: str
    runtime_dependency: bool
    fallback: str
    tests: tuple[str, ...]
    evidence: tuple[SourceEvidence, ...]


REFERENCE_ASSIGNMENTS: tuple[ReferenceAssignment, ...] = (
    ReferenceAssignment(
        repo="nautilus_trader",
        expected_commit="e8be4522cd12a4a65a4d1350f791d414ad246439",
        expected_tree="24bf52d8e0a6c4395e2b927f81d802a5c2436551",
        expected_branch="develop",
        expected_remote="https://github.com/nautechsystems/nautilus_trader.git",
        license_file="LICENSE",
        license="LGPL-3.0",
        expected_license_sha256=(
            "ee907919ec88c9c017b1f8b608db20960b6598aefcc4fe58820bde955d65ed3c"
        ),
        primary_responsibility="execution state and reconciliation invariants",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "event-driven execution state",
            "duplicate fill suppression",
            "venue reconciliation as events",
            "deterministic replay and derived read models",
        ),
        forbidden_copying="No upstream implementation copied; LGPL source is read as design evidence only.",
        runtime_dependency=False,
        fallback="existing execution.canonical_state ledger/reducer",
        tests=(
            "tests/test_canonical_execution_state.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "crates/execution/src/reconciliation/orders.rs",
                ("reconcile_order_report", "reconcile_fill_report"),
                "deduplicate fills and translate venue reports into state events",
            ),
            SourceEvidence(
                "crates/execution/src/engine/mod.rs",
                ("ExecutionEngine", "reconcile_execution_mass_status"),
                "event-driven execution and mass-status reconciliation",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="lean",
        expected_commit="c6cc3b743ed7b65d5e0b9fa2bfc18b7d3ac2aea0",
        expected_tree="f6ad336753d096b6cf123cd755e94d8d6e7aca9a",
        expected_branch="master",
        expected_remote="https://github.com/QuantConnect/Lean.git",
        license_file="LICENSE",
        license="Apache-2.0",
        expected_license_sha256=(
            "1c752ade7a9db6c100bf9a9f57225b52e138a92d2cd9dfe71c0af09af85366f7"
        ),
        primary_responsibility="strategy intent to portfolio target boundary",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "Alpha to PortfolioTarget separation",
            "target quantity versus current holdings delta",
            "risk-adjusted targets before execution",
        ),
        forbidden_copying="No LEAN classes copied; native crypto contracts preserve repository ownership.",
        runtime_dependency=False,
        fallback="fail closed with no executable target delta",
        tests=(
            "tests/test_reference_integration_phase_a.py",
            "future tests/test_portfolio_targets.py",
        ),
        evidence=(
            SourceEvidence(
                "Common/Algorithm/Framework/Portfolio/PortfolioTarget.cs",
                ("PortfolioTarget", "Percent"),
                "represent desired holdings independently of order submission",
            ),
            SourceEvidence(
                "Common/Algorithm/Framework/Portfolio/PortfolioTargetCollection.cs",
                ("PortfolioTargetCollection",),
                "collect and order target deltas",
            ),
            SourceEvidence(
                "Algorithm/Execution/ImmediateExecutionModel.cs",
                ("ImmediateExecutionModel", "Execute"),
                "execution consumes targets instead of strategy signals",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="vectorbt",
        expected_commit="34b6d5935e3ea3eccd549e2592bc0f455b8045f5",
        expected_tree="c33e6ed2845bf7643ed529899b82e38583bab3a0",
        expected_branch="master",
        expected_remote="https://github.com/polakowo/vectorbt.git",
        license_file="LICENSE.md",
        license="Apache-2.0 with Commons Clause",
        expected_license_sha256=(
            "8543a6018b754731325d61ed366763d5b6800bba1d5ddf56fb464e5ac7a8246e"
        ),
        primary_responsibility="approximate vectorized Stage-0 research screening",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "broadcasted parameter arrays",
            "signal matrices",
            "fast cost-aware rejection before exact validation",
        ),
        forbidden_copying="Commons Clause source is not copied or made a production dependency.",
        runtime_dependency=False,
        fallback="native research.research_factory Stage-0 screen",
        tests=(
            "tests/test_research_factory.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "vectorbt/portfolio/base.py",
                ("Portfolio", "from_signals"),
                "broadcastable signal simulation with explicit fees and slippage",
            ),
            SourceEvidence(
                "vectorbt/indicators/factory.py",
                ("IndicatorFactory", "run_combs"),
                "parameter-combination vectorization",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="pybroker",
        expected_commit="e0e7b08886343274efb05b96f7399ca3de280aa5",
        expected_tree="a02357b99b5ec4cb6bbacaa08d420bbed489cfc7",
        expected_branch="master",
        expected_remote="https://github.com/edtechre/pybroker.git",
        license_file="LICENSE",
        license="Apache-2.0 with Commons Clause",
        expected_license_sha256=(
            "8ef0c743408bcf350e8f9318f9a6447faeeda40d81fb973dfa52cbf6610f7951"
        ),
        primary_responsibility="chronological walk-forward validation contract",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "chronological walk-forward windows",
            "per-window retraining boundaries",
            "train/test segmentation without random shuffle",
        ),
        forbidden_copying="Commons Clause source is not copied or made a production dependency.",
        runtime_dependency=False,
        fallback="native WalkForwardManifest and exact validation engine",
        tests=(
            "tests/test_research_factory.py",
            "tests/test_optimization_risk.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "src/pybroker/strategy.py",
                ("WalkforwardMixin", "walkforward_split", "walkforward"),
                "ordered train/test windows and explicit split boundaries",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="qlib",
        expected_commit="79633dd9506ea689e5400dea0197717b5b3d74b7",
        expected_tree="0fdc22ba6b0d651a22d15f0142e27866a9f36908",
        expected_branch="main",
        expected_remote="https://github.com/microsoft/qlib.git",
        license_file="LICENSE",
        license="MIT",
        expected_license_sha256=(
            "9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b"
        ),
        primary_responsibility="immutable ML dataset experiment and model lifecycle",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "Dataset and DatasetH separation",
            "experiment recorder",
            "prediction and portfolio-analysis provenance",
        ),
        forbidden_copying="No Qlib implementation copied; MIT code remains an optional reference only.",
        runtime_dependency=False,
        fallback="native content-addressed dataset and evidence artifacts",
        tests=(
            "tests/test_feature_store.py",
            "tests/test_reference_integration_phase_a.py",
            "future tests/test_model_registry.py",
        ),
        evidence=(
            SourceEvidence(
                "qlib/data/dataset/__init__.py",
                ("Dataset", "DatasetH"),
                "dataset preparation and handler-backed data segments",
            ),
            SourceEvidence(
                "qlib/workflow/recorder.py",
                ("Recorder",),
                "experiment parameters metrics and artifact provenance",
            ),
            SourceEvidence(
                "qlib/workflow/record_temp.py",
                ("SignalRecord", "PortAnaRecord"),
                "prediction and economic-analysis records",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="freqtrade",
        expected_commit="89d469fe638eaf116d45a8f92598aeed4d9f6dde",
        expected_tree="046f7c3f5025069b8a9757c092d6edee26f60c3f",
        expected_branch="develop",
        expected_remote="https://github.com/freqtrade/freqtrade.git",
        license_file="LICENSE",
        license="GPL-3.0",
        expected_license_sha256=(
            "53927bd0b739d38c87a0a82236fd9b070c2dfff11c0c119be50372005d5047ad"
        ),
        primary_responsibility="crypto ML lifecycle and causal bias controls",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "truncated-history lookahead comparison",
            "recursive warmup stability",
            "model expiry and rolling retraining boundaries",
            "closed-candle and mutable-universe controls",
        ),
        forbidden_copying="GPL implementation is not copied, imported by production, or made execution owner.",
        runtime_dependency=False,
        fallback="native bias checks; failed or stale ML remains SHADOW/blocked",
        tests=(
            "tests/test_research_factory.py",
            "tests/test_reference_integration_phase_a.py",
            "future tests/test_ml_lifecycle.py",
        ),
        evidence=(
            SourceEvidence(
                "freqtrade/optimize/analysis/lookahead.py",
                ("LookaheadAnalysis",),
                "full-history versus sliced-run bias analysis",
            ),
            SourceEvidence(
                "freqtrade/optimize/analysis/recursive.py",
                ("RecursiveAnalysis", "analyze_indicators_lookahead"),
                "startup-history stability and indicator lookahead checks",
            ),
            SourceEvidence(
                "freqtrade/freqai/data_kitchen.py",
                ("FreqaiDataKitchen", "check_if_model_expired", "check_if_new_training_required"),
                "model expiry, retraining and warmup windows",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="finrl-trading",
        expected_commit="e65d6f0483ead7d2ef4a5fc940cdf960392a25c1",
        expected_tree="25c9d3151c0e4b415b7676a2fdb0f1c3b4bcc6db",
        expected_branch="master",
        expected_remote="https://github.com/AI4Finance-Foundation/FinRL-Trading.git",
        license_file="LICENSE",
        license="Apache-2.0",
        expected_license_sha256=(
            "afae3377fdbd0537635360e91585f3c5b478ffe8eb5308f1ddcb37b76a7325d2"
        ),
        primary_responsibility="bounded position-management RL environment and baseline contract",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "explicit state and action spaces",
            "cost-aware reward accounting",
            "deterministic out-of-sample prediction",
            "multiple non-RL baselines before agent comparison",
        ),
        forbidden_copying="No FinRL environment, agent or execution implementation copied or imported at runtime.",
        runtime_dependency=False,
        fallback="deterministic HOLD/REDUCE/EXIT baselines; RL remains SHADOW_ONLY",
        tests=(
            "tests/test_rl_position_management.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "src/strategies/rl_model.py",
                ("DRLAgent", "train_model", "DRL_prediction", "deterministic=True"),
                "separate training and deterministic evaluation of multiple agents",
            ),
            SourceEvidence(
                "src/strategies/fundamental_portfolio_drl.py",
                ("state_space", "action_space", "transaction_cost_pct", "reward_scaling"),
                "explicit environment dimensions, costs and reward scaling",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="rd-agent",
        expected_commit="6762f84f9bc0f5c6486c50a00e128a57ac6c3683",
        expected_tree="e9eea57867e58c4424a346684edc16bd22c4e736",
        expected_branch="main",
        expected_remote="https://github.com/microsoft/RD-Agent.git",
        license_file="LICENSE",
        license="MIT",
        expected_license_sha256=(
            "9906940f61b1f0b533fa7d99baf55178b2808fbe113ea51dfbfad8572ccd5f2b"
        ),
        primary_responsibility="preregistered autonomous research hypothesis experiment feedback memory loop",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "Hypothesis to Experiment to Feedback lifecycle",
            "append-only trace and knowledge memory",
            "explicit accept or reject decision",
            "bounded iterative research with no promotion authority",
        ),
        forbidden_copying="No RD-Agent orchestration or LLM implementation copied or made a runtime dependency.",
        runtime_dependency=False,
        fallback="native deterministic research ledger; failed experiments remain rejected",
        tests=(
            "tests/test_autonomous_research_loop.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "rdagent/core/proposal.py",
                ("Hypothesis", "ExperimentFeedback", "HypothesisFeedback", "Trace"),
                "typed hypothesis feedback and chronological trace concepts",
            ),
            SourceEvidence(
                "rdagent/components/workflow/rd_loop.py",
                ("RDLoop", "_propose", "_exp_gen", "feedback", "record"),
                "bounded propose experiment feedback record loop",
            ),
        ),
    ),
    ReferenceAssignment(
        repo="tradingagents",
        expected_commit="a33fd4c0f134485a43553a2c23a63cb14adbd88f",
        expected_tree="fc9e0a673a22f7d5d755e2e6af7ecbdf2ba43a0b",
        expected_branch="main",
        expected_remote="https://github.com/TauricResearch/TradingAgents.git",
        license_file="LICENSE",
        license="Apache-2.0",
        expected_license_sha256=(
            "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6"
        ),
        primary_responsibility="structured multi-perspective market intelligence evidence",
        integration_mode="C_CONCEPT_REFERENCE_ONLY_NATIVE_IMPLEMENTATION",
        used_concepts=(
            "bull and bear evidence separation",
            "research synthesis before trader proposal",
            "independent risk challenge",
            "typed structured output with provenance",
        ),
        forbidden_copying="No TradingAgents graph, prompts or agents copied or connected to execution.",
        runtime_dependency=False,
        fallback="neutral NO_TRADE intelligence snapshot when evidence is missing or conflicted",
        tests=(
            "tests/test_structured_market_intelligence.py",
            "tests/test_reference_integration_phase_a.py",
        ),
        evidence=(
            SourceEvidence(
                "tradingagents/agents/schemas.py",
                ("ResearchPlan", "TraderProposal", "PortfolioDecision", "PortfolioRating"),
                "structured research trading and portfolio decision payloads",
            ),
            SourceEvidence(
                "tradingagents/agents/utils/agent_states.py",
                ("InvestDebateState", "RiskDebateState", "AgentState"),
                "separate debate, risk and final-decision state",
            ),
        ),
    ),
)


CAPABILITY_GAPS: tuple[dict[str, str], ...] = (
    {
        "capability": "canonical execution ledger and replay",
        "current_implementation": "execution/canonical_state.py",
        "reference": "nautilus_trader",
        "reference_concept": "events, idempotent fills, reconciliation, deterministic replay",
        "current_gap": "core invariants exist; reconciliation semantics still span several live modules",
        "action": "KEEP_AND_HARDEN",
        "owner_after_migration": "execution.canonical_state",
    },
    {
        "capability": "strategy intent to portfolio target boundary",
        "current_implementation": "portfolio.contracts plus portfolio.targets and canonical buy-chain adapters",
        "reference": "lean",
        "reference_concept": "Alpha -> PortfolioTarget -> Risk -> Execution",
        "current_gap": "canonical owner exists; legacy direct OrderIntent callers remain migration debt",
        "action": "KEEP_CANONICAL_OWNER_AND_MIGRATE_REMAINING_CALLERS",
        "owner_after_migration": "portfolio.targets",
    },
    {
        "capability": "fast approximate Stage-0 research",
        "current_implementation": "research/research_factory.py Stage0Signals and simulate_stage0",
        "reference": "vectorbt",
        "reference_concept": "broadcasted signal and parameter arrays",
        "current_gap": "native screen exists; benchmark agreement and sampled false-negative evidence need consolidation",
        "action": "KEEP_NATIVE_AND_BENCHMARK",
        "owner_after_migration": "research.research_factory",
    },
    {
        "capability": "purged chronological walk-forward",
        "current_implementation": "research/research_factory.py WalkForwardManifest plus research/optimization.py",
        "reference": "pybroker",
        "reference_concept": "ordered train/test windows and retraining boundaries",
        "current_gap": "strong primitives exist in more than one validation path",
        "action": "CONSOLIDATE_SHARED_VALIDATION_CONTRACT",
        "owner_after_migration": "research.validation",
    },
    {
        "capability": "immutable ML dataset and model lifecycle",
        "current_implementation": "data/feature_store.py, data/multi_source_platform.py, ExperimentContract, distributed SHADOW markers",
        "reference": "qlib",
        "reference_concept": "Dataset -> Experiment -> Recorder -> Model -> Prediction",
        "current_gap": "feature/dataset identities exist but no single canonical label store or model registry",
        "action": "CONSOLIDATE_DATASET_CONTRACT_AND_BUILD_REGISTRY",
        "owner_after_migration": "ml.registry",
    },
    {
        "capability": "crypto ML lifecycle and leakage controls",
        "current_implementation": "research_factory causality/static/warmup checks and SHADOW-only inference modules",
        "reference": "freqtrade",
        "reference_concept": "lookahead, recursive warmup, expiry and rolling inference",
        "current_gap": "checks exist but are not one mandatory lifecycle contract",
        "action": "CONSOLIDATE_FAIL_CLOSED_LIFECYCLE",
        "owner_after_migration": "ml.lifecycle",
    },
    {
        "capability": "position-management reinforcement-learning research",
        "current_implementation": "rl.position_management deterministic environment and baselines",
        "reference": "finrl-trading",
        "reference_concept": "explicit environment state action reward and deterministic evaluation",
        "current_gap": "no Gymnasium/SB3 runtime and insufficient prospective episodes for agent training",
        "action": "KEEP_SHADOW_BASELINES_AND_COLLECT_EPISODES",
        "owner_after_migration": "rl.position_management",
    },
    {
        "capability": "autonomous research iteration and experiment memory",
        "current_implementation": "research.autonomous_rd append-only hypothesis experiment feedback ledger",
        "reference": "rd-agent",
        "reference_concept": "Hypothesis -> Experiment -> Feedback -> Trace",
        "current_gap": "research loop is deliberately unable to promote or mutate live strategy authority",
        "action": "KEEP_BOUNDED_RESEARCH_ONLY",
        "owner_after_migration": "research.autonomous_rd",
    },
    {
        "capability": "structured multi-perspective market intelligence",
        "current_implementation": "core.structured_market_intelligence typed evidence synthesis",
        "reference": "tradingagents",
        "reference_concept": "bull/bear debate, research synthesis and independent risk challenge",
        "current_gap": "no approved model provider; deterministic synthesis remains SHADOW_ONLY",
        "action": "KEEP_STRUCTURED_SHADOW_EVIDENCE",
        "owner_after_migration": "core.structured_market_intelligence",
    },
    {
        "capability": "canonical cost and expectancy",
        "current_implementation": "research.research_factory.SharedCostModel and research.backtest.CostModel",
        "reference": "lean",
        "reference_concept": "portfolio construction and execution consume the same economic assumptions",
        "current_gap": "multiple cost representations can drift",
        "action": "CONSOLIDATE_CANONICAL_COST_CONTRACT",
        "owner_after_migration": "core.economics",
    },
    {
        "capability": "reference adapter failure isolation",
        "current_implementation": "research/reference_integrations.py probes execute in one aggregate call",
        "reference": "all",
        "reference_concept": "references inform bounded subsystems only",
        "current_gap": "one failed probe aborts the aggregate and roles do not match the new primary assignments",
        "action": "RECLASSIFY_AS_NON_AUTHORITATIVE_PROBES_AND_ISOLATE_FAILURES",
        "owner_after_migration": "reporting.reference_health",
    },
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _source_evidence(root: Path, rows: Sequence[SourceEvidence]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        path = root / row.path
        source = path.read_text(encoding="utf-8")
        evidence.append(
            {
                **asdict(row),
                "exists": True,
                "sha256": sha256_file(path),
                "symbol_presence": {symbol: symbol in source for symbol in row.symbols},
            }
        )
    return evidence


def snapshot_reference(workspace: Path, assignment: ReferenceAssignment) -> dict[str, Any]:
    root = workspace / "crypto-references" / assignment.repo
    if not root.is_dir():
        raise FileNotFoundError(f"missing reference repository: {root}")
    license_path = root / assignment.license_file
    actual_commit = _git(root, "rev-parse", "HEAD")
    actual_tree = _git(root, "rev-parse", "HEAD^{tree}")
    actual_branch = _git(root, "branch", "--show-current")
    actual_remote = _git(root, "remote", "get-url", "origin")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    license_hash = sha256_file(license_path)
    evidence = _source_evidence(root, assignment.evidence)
    row = {
        **asdict(assignment),
        "path": str(root.resolve()),
        "actual_commit": actual_commit,
        "actual_tree": actual_tree,
        "actual_branch": actual_branch,
        "actual_remote": actual_remote,
        "license_sha256": license_hash,
        "reference_clean": not bool(status),
        "dirty_entries": status.splitlines(),
        "commit_verified": actual_commit == assignment.expected_commit,
        "tree_verified": actual_tree == assignment.expected_tree,
        "branch_verified": actual_branch == assignment.expected_branch,
        "remote_verified": actual_remote == assignment.expected_remote,
        "license_verified": license_hash == assignment.expected_license_sha256,
        "source_evidence": evidence,
    }
    row["source_symbols_verified"] = all(
        all(item["symbol_presence"].values()) for item in evidence
    )
    return row


def _production_baseline(workspace: Path) -> dict[str, Any]:
    direct_order_paths: list[dict[str, Any]] = []
    for relative in (
        "core/autonomous_trading.py",
        "core/event_driven_live.py",
        "core/generated_strategy_live.py",
        "core/inventory_reallocation.py",
        "core/cli.py",
    ):
        path = workspace / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        direct_order_paths.append(
            {
                "path": relative,
                "order_intent_mentions": source.count("OrderIntent("),
                "submit_order_mentions": source.count("submit_order("),
                "sha256": sha256_file(path),
            }
        )
    return {
        "composition_root": "main.py -> core.cli.main",
        "data_to_live_features": (
            "data.multi_source_runtime -> data.multi_source_platform.PointInTimeFeatureStore"
        ),
        "data_to_research": "processed Parquet -> research.features -> research.research_factory",
        "research_authority": "exact native research.backtest.BacktestEngine",
        "execution_owner": "execution.canonical_state append-only replay reducer",
        "exchange_owner": "execution.execution Bitvavo client",
        "risk_owners": "risk.risk_manager plus core execution-authority gates",
        "ml_authority": "SHADOW_ONLY",
        "portfolio_target_owner": "portfolio.targets",
        "direct_order_paths": direct_order_paths,
        "missing_requested_directories": [
            name
            for name in ("strategies", "portfolio")
            if not (workspace / name).is_dir()
        ],
    }


def build_phase_a_inventory(
    workspace: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Build and persist one content-addressed, audit-only Phase A artifact."""

    workspace = workspace.resolve()
    snapshots = [snapshot_reference(workspace, row) for row in REFERENCE_ASSIGNMENTS]
    primary_roles = [row.primary_responsibility for row in REFERENCE_ASSIGNMENTS]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workspace": str(workspace),
        "generated_at": utc_iso(),
        "phase": "A",
        "phase_status": "PASSED" if all(
            row["reference_clean"]
            and row["commit_verified"]
            and row["tree_verified"]
            and row["branch_verified"]
            and row["remote_verified"]
            and row["license_verified"]
            and row["source_symbols_verified"]
            for row in snapshots
        ) and len(primary_roles) == len(set(primary_roles)) else "FAILED",
        "repositories": snapshots,
        "reference_contribution_matrix": [
            {
                "repo": row.repo,
                "primary_responsibility": row.primary_responsibility,
                "integration_mode": row.integration_mode,
                "used_concepts": list(row.used_concepts),
                "native_owner": next(
                    (
                        gap["owner_after_migration"]
                        for gap in CAPABILITY_GAPS
                        if gap["reference"] == row.repo
                    ),
                    "cross-cutting documented owner",
                ),
            }
            for row in REFERENCE_ASSIGNMENTS
        ],
        "capability_gap_matrix": list(CAPABILITY_GAPS),
        "production_baseline": _production_baseline(workspace),
        "architecture_invariants": {
            "references_are_read_only": True,
            "one_primary_responsibility_per_reference": len(primary_roles)
            == len(set(primary_roles)),
            "reference_runtime_dependency_allowed": False,
            "reference_execution_authority": False,
            "exact_native_validation_authoritative": True,
            "risk_highest_authority": True,
            "ml_authority": "SHADOW_ONLY",
        },
        "phase_b_entry_gates": {
            "all_repositories_clean": all(row["reference_clean"] for row in snapshots),
            "all_commits_verified": all(row["commit_verified"] for row in snapshots),
            "all_trees_verified": all(row["tree_verified"] for row in snapshots),
            "all_licenses_verified": all(row["license_verified"] for row in snapshots),
            "all_source_symbols_verified": all(
                row["source_symbols_verified"] for row in snapshots
            ),
            "unique_primary_roles": len(primary_roles) == len(set(primary_roles)),
        },
        "side_effects": {
            "orders_generated": 0,
            "orders_submitted": 0,
            "orders_cancelled": 0,
            "fills_created": 0,
            "private_exchange_mutations": 0,
            "authority_changes": 0,
            "risk_changes": 0,
            "shariah_changes": 0,
            "supervisor_or_collector_restarts": 0,
            "reference_files_modified": 0,
        },
        "legal_note": (
            "Repository-local license classification is an engineering boundary, not legal advice. "
            "All nine integrations are intentionally restricted to concept/reference-only native work."
        ),
    }
    artifact_hash = stable_hash(
        {key: value for key, value in body.items() if key != "generated_at"},
        length=64,
    )
    body["artifact_hash"] = artifact_hash
    root = output_root or workspace / "output" / "reference_integration" / "phase_a"
    artifact_path = root / "runs" / artifact_hash / "reference_integration_phase_a.json"
    if artifact_path.is_file():
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        if existing.get("artifact_hash") != artifact_hash:
            raise ValueError("existing Phase A artifact identity mismatch")
    else:
        atomic_write_json(artifact_path, body)
    latest = {**body, "artifact_path": str(artifact_path.resolve())}
    atomic_write_json(root / "latest.json", latest)
    return latest


__all__ = [
    "CAPABILITY_GAPS",
    "REFERENCE_ASSIGNMENTS",
    "SCHEMA_VERSION",
    "build_phase_a_inventory",
    "snapshot_reference",
]
