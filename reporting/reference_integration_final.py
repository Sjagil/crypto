"""Render the required A-AD final reference-integration evidence report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reporting.reference_integration_health import build_reference_integration_health
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "reference_integration_final_report_v1"

ADDED_FILES = (
    "core/economics.py",
    "core/structured_market_intelligence.py",
    "docs/ARCHITECTURE_OWNERSHIP.md",
    "docs/ML_MODEL_LIFECYCLE.md",
    "docs/REFERENCE_INTEGRATION_GAP_AUDIT.md",
    "docs/REFERENCE_INTEGRATION_POLICY.md",
    "docs/REFERENCE_REPOSITORY_MASTER_MAP.md",
    "ml/__init__.py",
    "ml/contracts.py",
    "ml/labels.py",
    "ml/lifecycle.py",
    "ml/registry.py",
    "portfolio/__init__.py",
    "portfolio/buy_chain.py",
    "portfolio/contracts.py",
    "portfolio/targets.py",
    "research/autonomous_rd.py",
    "reporting/ml_legacy_assessment.py",
    "reporting/reference_integration_final.py",
    "reporting/reference_integration_health.py",
    "reporting/reference_integration_phase_a.py",
    "reporting/reference_repository_master_map.py",
    "rl/__init__.py",
    "rl/position_management.py",
    "scripts/build_reference_master_map.py",
    "scripts/build_reference_integration_final.py",
    "scripts/build_reference_integration_health.py",
    "scripts/build_reference_integration_phase_a.py",
    "tests/test_canonical_cost_model.py",
    "tests/test_canonical_buy_chain.py",
    "tests/test_buy_producer_migration.py",
    "tests/test_ml_lifecycle.py",
    "tests/test_portfolio_contracts.py",
    "tests/test_portfolio_targets.py",
    "tests/test_reference_integration_health.py",
    "tests/test_reference_integration_final.py",
    "tests/test_reference_integration_phase_a.py",
    "tests/test_reference_repository_master_map.py",
    "tests/test_rl_position_management.py",
    "tests/test_autonomous_research_loop.py",
    "tests/test_structured_market_intelligence.py",
)

MODIFIED_FILES = (
    "core/autonomous_trading.py",
    "core/cli.py",
    "core/contracts.py",
    "core/event_driven_live.py",
    "core/generated_strategy_live.py",
    "core/opportunity_intelligence.py",
    "execution/canonical_state.py",
    "execution/execution.py",
    "reporting/system_audit.py",
    "research/backtest.py",
    "research/research_factory.py",
    "tests/test_canonical_execution_state.py",
    "tests/test_execution.py",
    "ui/server.py",
)


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    return dict(value) if isinstance(value, Mapping) else {}


def _earliest_phase_a(root: Path) -> dict[str, Any]:
    artifacts = sorted(
        (root / "output" / "reference_integration" / "phase_a" / "runs").glob(
            "*/reference_integration_phase_a.json"
        )
    )
    rows = [_mapping(path) for path in artifacts]
    rows = [row for row in rows if row]
    return min(rows, key=lambda row: str(row.get("generated_at") or "")) if rows else {}


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Reference integration final report",
        "",
        f"Status: **{payload['status']}**  ",
        f"Live readiness: **{payload['live_readiness']}**  ",
        f"Artifact: `{payload['artifact_hash']}`",
        "",
    ]
    for key, section in payload["sections"].items():
        lines.extend((f"## {key}. {section['title']}", ""))
        body = section["body"]
        if isinstance(body, str):
            lines.extend((body, ""))
        elif isinstance(body, list):
            for value in body:
                lines.append(f"- `{value}`" if isinstance(value, str) else f"- {value}")
            lines.append("")
        else:
            lines.extend(("```json", _compact_json(body), "```", ""))
    return "\n".join(lines)


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _content_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _content_identity(item)
            for key, item in sorted(value.items())
            if key not in {"generated_at", "artifact_hash"}
        }
    if isinstance(value, (list, tuple)):
        return [_content_identity(item) for item in value]
    return value


def build_reference_integration_final(
    project_root: Path,
    *,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact A-AD report and its content-addressed artifacts."""

    root = project_root.resolve()
    output = root / "output"
    health_result = build_reference_integration_health(root)
    health = health_result["payload"]
    latest_phase_a = _mapping(output / "reference_integration" / "phase_a" / "latest.json")
    phase_a = _mapping(Path(str(latest_phase_a.get("artifact_path") or "")))
    master_map = _mapping(
        output / "reference_integration" / "reference_master_map.json"
    )
    reference_probes = _mapping(output / "reference_integrations" / "latest.json")
    baseline = _earliest_phase_a(root).get("production_baseline") or {}
    research_pointer = _mapping(output / "research_factory" / "latest.json")
    research = _mapping(Path(str(research_pointer.get("artifact_path") or "")))
    economics_pointer = _mapping(output / "economics" / "latest.json")
    economics = _mapping(
        Path(str(economics_pointer.get("artifact_path") or ""))
    )
    references = [
        {
            "repo": row.get("repo"),
            "commit": row.get("actual_commit"),
            "tree_hash": row.get("actual_tree"),
            "branch": row.get("actual_branch"),
            "license": row.get("license"),
            "license_sha256": row.get("license_sha256"),
            "clean": row.get("reference_clean"),
        }
        for row in phase_a.get("repositories") or []
    ]
    tests_ok = all(
        verification.get(key) is True
        for key in (
            "targeted_passed",
            "integration_passed",
            "full_suite_passed",
            "lint_passed",
            "compile_passed",
            "diff_check_passed",
        )
    )
    side_effects = dict(health["side_effects"])
    sections = {
        "A": {"title": "Baseline vóór integratie", "body": baseline},
        "B": {"title": "Exacte commits/tree-hashes/licenses van alle negen references", "body": references},
        "C": {"title": "Reference contribution matrix en usage registry", "body": {
            "contributions": phase_a.get("reference_contribution_matrix") or [],
            "usage_registry": master_map.get("repositories") or [],
        }},
        "D": {
            "title": "Architecture ownership vóór/na",
            "body": {
                "before": baseline,
                "after": {
                    "intent": "portfolio.contracts.InvestmentIntent",
                    "target": "portfolio.contracts.PortfolioTarget",
                    "risk": "portfolio.contracts.RiskApproval plus risk.risk_manager",
                    "execution_intent": "portfolio.contracts.ExecutionIntent",
                    "orders": "execution.execution",
                    "financial_state": "execution.canonical_state",
                    "costs": "core.economics.CanonicalCostModel",
                    "ml_registry": "ml.registry",
                },
            },
        },
        "E": {"title": "Wat van NautilusTrader is toegepast", "body": "Event-driven canonical order/fill/position state, duplicate-fill suppression, reconciliation events and deterministic replay; implemented natively."},
        "F": {"title": "Wat van LEAN is toegepast", "body": "The explicit InvestmentIntent -> PortfolioTarget -> RiskApproval -> ExecutionIntent boundary and target delta semantics; implemented natively."},
        "G": {"title": "Wat van vectorbt is toegepast", "body": "Approximate vectorized Stage-0 parameter screening and sampled false-negative review. Exact native validation remains authoritative."},
        "H": {"title": "Wat van PyBroker is toegepast", "body": "Chronological walk-forward manifests with immutable folds, explicit purge and embargo boundaries and final-test isolation."},
        "I": {"title": "Wat van Qlib is toegepast", "body": "Content-addressed dataset/model manifests, experiment provenance and immutable registries. The legacy model is not migrated without proof."},
        "J": {"title": "Wat van Freqtrade, FinRL, RD-Agent en TradingAgents is toegepast", "body": {
            "freqtrade": "Native point-in-time, closed-input, lookahead, recursive warmup, model expiry, drift and deterministic fallback checks.",
            "finrl": "Een native spot-only SHADOW position-management environment met expliciete state/action/reward, kosten, deterministic baselines en een harde trainingseligibilitygate.",
            "rd_agent": "Een immutable Hypothesis -> PreregisteredExperiment -> Feedback trace zonder live-promotie- of exchangebevoegdheid.",
            "tradingagents": "Getypeerde bull/bear/risk evidence, conflict-detectie en een AIDecisionSnapshot die als SHADOW uitsluitend NO_TRADE aan InvestmentIntent mag doorgeven.",
        }},
        "K": {"title": "Welke code native is geschreven", "body": list(ADDED_FILES)},
        "L": {"title": "Welke code bewust NIET is gekopieerd wegens license/ownership", "body": "No implementation from the nine references was copied or imported. GPL/LGPL/Commons-Clause sources were used only as bounded design evidence; all runtime owners remain native."},
        "M": {"title": "Canonical execution state status", "body": health["phases"]["C"]},
        "N": {"title": "Portfolio target status", "body": health["phases"]["D"]},
        "O": {"title": "Research factory status", "body": health["research_state"]},
        "P": {"title": "Walk-forward status", "body": {"phase": health["phases"]["F"], "manifest": research.get("validation_manifest")}},
        "Q": {"title": "Feature/dataset/label architecture", "body": {"canonical_contracts": health["phases"]["G"], "canonical_dataset": health["canonical_model_state"], "legacy_dataset": health["model_state"]["dataset"]}},
        "R": {"title": "ML/model registry status", "body": {"canonical": health["canonical_model_state"], "legacy_assessment": health["model_state"]}},
        "S": {"title": "Bias/leakage audit", "body": {"canonical_lifecycle": health["phases"]["H"], "research_bias_checks": research.get("bias_checks")}},
        "T": {"title": "Cost/expectancy architecture", "body": {"canonical_owner": "core.economics.CanonicalCostModel", "research_cost_model": research.get("shared_cost_model"), "current_economics": economics}},
        "U": {"title": "Portfolio/risk architecture", "body": {"chain": "InvestmentIntent -> PortfolioTarget -> RiskApproval", "risk_highest_authority": True, "correlation_cluster": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES"}},
        "V": {"title": "Execution/reconciliation architecture", "body": health["execution_state"]},
        "W": {"title": "Performance benchmark vóór/na", "body": research.get("benchmark") or {"status": "NOT_EVALUABLE_NO_RESEARCH_BENCHMARK"}},
        "X": {"title": "Tests", "body": dict(verification)},
        "Y": {"title": "Failures / NOT_EVALUABLE punten", "body": {
            "legacy_buy_callers": health["execution_state"]["legacy_callers"],
            "legacy_ml": health["model_state"]["model"]["blockers"],
            "portfolio_correlation_cluster": "NOT_EVALUABLE_NO_ROBUST_FORWARD_CANDIDATES",
            "profitability": "NOT_PROVEN_CURRENT_CANONICAL_ECONOMICS_NEGATIVE",
        }},
        "Z": {"title": "Live side effects", "body": side_effects},
        "AA": {"title": "Exacte bestanden toegevoegd/gewijzigd", "body": {"added": list(ADDED_FILES), "modified": list(MODIFIED_FILES)}},
        "AB": {"title": "Artifact IDs + hashes", "body": {
            "phase_a": latest_phase_a.get("artifact_hash"),
            "reference_master_map": master_map.get("artifact_hash"),
            "reference_probe_evidence": reference_probes.get("evidence_hash"),
            "health": health_result["artifact_hash"],
            "research_factory": research_pointer.get("artifact_hash"),
            "canonical_economics": economics_pointer.get("artifact_hash")
            or economics.get("artifact_hash")
            or economics.get("reconciliation_hash"),
        }},
        "AC": {"title": "Resterende technische schuld", "body": [
            f"Collect enough new prospective feature snapshots with explicit event_time, available_at, finality and provenance; do not backfill the {health['canonical_model_state'].get('excluded_row_count', 'existing')} legacy rows.",
            "Allow the canonical trainer to register a SHADOW model only after its 500-row and class-support gates plus five purged folds pass.",
            "Resolve external/grandfathered inventory ownership, account readiness and the paused control state before considering any new live entry authority.",
            "Do not build correlation/cluster allocation until robust forward candidates exist.",
        ]},
        "AD": {"title": "Exact één beste volgende taak", "body": "Run prospective opportunity collection until at least 500 causally timed and fully labeled rows with at least 100 examples per class exist; then rerun the canonical SHADOW trainer and inspect its untouched test/calibration evidence."},
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": "PARTIAL_NOT_LIVE_READY" if health["status"] != "PASSED" else "PASSED",
        "live_readiness": health["live_readiness"],
        "verification_status": "PASSED" if tests_ok else "INCOMPLETE_OR_FAILED",
        "sections": sections,
        "side_effects": side_effects,
    }
    artifact_hash = stable_hash(_content_identity(payload), length=64)
    payload["artifact_hash"] = artifact_hash
    run = output / "reference_integration" / "final" / "runs" / artifact_hash
    json_path = run / "reference_integration_final.json"
    markdown_path = run / "reference_integration_final.md"
    if json_path.is_file():
        existing = _mapping(json_path)
        if stable_hash(_content_identity(existing), length=64) != artifact_hash:
            raise ValueError("final report artifact identity collision")
        payload = existing
    else:
        atomic_write_json(json_path, payload)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    pointer = {
        "schema_version": "reference_integration_final_pointer_v1",
        "artifact_hash": artifact_hash,
        "json_path": str(json_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "status": payload["status"],
        "live_readiness": payload["live_readiness"],
        "verification_status": payload["verification_status"],
    }
    atomic_write_json(output / "reference_integration" / "final" / "latest.json", pointer)
    return pointer


__all__ = [
    "ADDED_FILES",
    "MODIFIED_FILES",
    "SCHEMA_VERSION",
    "build_reference_integration_final",
]
