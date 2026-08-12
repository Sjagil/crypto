"""Sequential, evidence-based maturity ladder for the crypto platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEVEL_RANGES = {
    "BEGINNER": range(1, 5),
    "INTERMEDIATE": range(5, 19),
    "ADVANCED": range(19, 30),
    "EXPERT": range(30, 34),
}


PROJECT_NAMES = (
    "Multi-source crypto data platform",
    "Data quality and causal candle controls",
    "Crypto performance and risk analyzer",
    "Read-only operations dashboard",
    "Event-driven backtester",
    "Transaction-cost and slippage model",
    "Reproducible strategy laboratory",
    "Walk-forward validation framework",
    "Execution foundation and authority gates",
    "Paper-trading simulator",
    "Reconciliation and crash recovery",
    "Shadow execution",
    "Tiny live canary with human approval",
    "Monte Carlo and stochastic validation",
    "Cross-sectional research framework",
    "Momentum and rotation strategies",
    "Market-breadth model",
    "BTC-dominance allocator",
    "Volatility models",
    "Portfolio construction",
    "Portfolio risk engine",
    "Regime detector",
    "Regime-aware allocator",
    "Order-flow and microstructure research",
    "Liquidity and market-impact model",
    "Cross-exchange research",
    "Derivatives context",
    "On-chain and stablecoin context",
    "News and event intelligence",
    "Probabilistic ML meta-model",
    "Alpha combination and dynamic allocation",
    "Execution analytics and adaptive routing",
    "Autonomous manager with hard governance",
)


EVIDENCE: dict[int, tuple[str, ...]] = {
    1: ("data/multi_source_platform.py", "output/multi_source/status.json"),
    2: ("data/market_data.py", "tests/test_market_data.py"),
    3: ("research/crypto_performance.py", "tests/test_crypto_performance.py"),
    4: ("ui/server.py", "tests/test_multi_timeframe_expansion.py"),
    5: ("research/backtest.py", "tests/test_backtest_math.py"),
    6: ("reporting/canonical_economics.py", "tests/test_canonical_economics.py"),
    7: ("research/simple_strategy_lab.py", "tests/test_simple_strategy_lab.py"),
    8: ("research/stochastic_validation.py", "tests/test_stochastic_validation.py"),
    9: ("core/execution_authority.py", "tests/test_execution_authority.py"),
    10: ("core/event_driven_paper.py", "tests/test_event_driven_paper.py"),
    11: ("execution/canonical_state.py", "tests/test_canonical_execution_state.py"),
    12: ("core/execution_evidence.py", "tests/test_execution_evidence.py"),
    13: ("risk/canary_guard.py", "tests/test_canary_guard.py"),
    14: ("research/stochastic_validation.py", "tests/test_stochastic_validation.py"),
    15: ("research/portfolio_selection.py", "tests/test_portfolio_selection.py"),
    16: ("research/absolute_momentum.py", "tests/test_absolute_momentum.py"),
    17: ("research/macro_context.py", "tests/test_market_intelligence.py"),
    18: ("research/macro_liquidity_rotation.py", "tests/test_macro_liquidity_rotation.py"),
    19: ("research/volatility_contraction.py", "tests/test_volatility_contraction.py"),
    20: ("research/portfolio_selection.py", "tests/test_portfolio_selection.py"),
    21: ("risk/risk_manager.py", "tests/test_portfolio_controls.py"),
    22: ("research/hmm_regime_manager.py", "tests/test_hmm_regime_manager.py"),
    23: ("research/regime_router.py", "tests/test_regime_router.py"),
    24: ("research/microstructure_observer.py", "tests/test_microstructure_observer.py"),
    25: ("core/market_mechanics.py", "tests/test_market_mechanics.py"),
    26: ("data/multi_source_platform.py", "tests/test_multi_source_platform.py"),
    27: ("data/derivatives_context.py", "tests/test_database_macro_derivatives.py"),
    28: ("research/macro_context.py", "tests/test_database_macro_derivatives.py"),
    29: ("core/event_driven_playbooks.py", "tests/test_event_driven_playbooks.py"),
    30: ("research/hmm_strategy_comparison.py", "tests/test_hmm_strategy_comparison.py"),
    31: ("research/multi_alpha_ensemble.py", "tests/test_multi_alpha_ensemble.py"),
    32: ("core/decision_attribution.py", "tests/test_decision_attribution.py"),
    33: ("core/autonomous_trading.py", "tests/test_autonomous_trading.py"),
}


def _level(project_id: int) -> str:
    return next(name for name, ids in LEVEL_RANGES.items() if project_id in ids)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _operational_readiness(workspace: Path) -> dict[str, Any]:
    """Keep implementation maturity separate from financial/live authority."""

    output = workspace / "output"
    account = _read_mapping(output / "operations" / "live_account_health.json")
    external = _read_mapping(
        output / "operations" / "external_inventory_remediation.json"
    )
    telegram = _read_mapping(
        output / "operations" / "telegram_signal_evidence.json"
    )
    economics = _read_mapping(output / "economics" / "latest.json")

    blockers: list[str] = []
    if account.get("status") != "READY" or account.get("entry_allowed") is not True:
        blockers.append("LIVE_ACCOUNT_ENTRY_NOT_READY")
    control_state = (
        (account.get("execution_authority") or {}).get("control_state")
    )
    if control_state != "RUNNING":
        blockers.append("AUTONOMOUS_CONTROL_NOT_RUNNING")
    if external.get("status") == "OPERATOR_DECISION_REQUIRED":
        blockers.append("EXTERNAL_INVENTORY_DECISION_REQUIRED")
    if not economics or economics.get("best_validated_family") is None:
        blockers.append("NO_FINANCIALLY_VALIDATED_STRATEGY_FAMILY")

    telegram_claim = telegram.get("claim_under_test") or {}
    telegram_gate = telegram.get("paper_shadow_gate") or {}
    telegram_ready = (
        telegram_claim.get("status") == "CONFIRMED"
        and telegram_gate.get("status") == "PROMOTION_ELIGIBLE"
    )
    telegram_blockers: list[str] = []
    if not telegram:
        telegram_blockers.append("TELEGRAM_EXACT_EVIDENCE_MISSING")
    elif not telegram_ready:
        telegram_blockers.append("TELEGRAM_95PCT_TP2_CLAIM_NOT_CONFIRMED")

    return {
        "schema_version": "crypto_operational_readiness_v1",
        "autonomous_live_status": "READY" if not blockers else "BLOCKED",
        "autonomous_live_blockers": blockers,
        "telegram_signal_authority": (
            "ELIGIBLE_FOR_SEPARATE_OPERATOR_REVIEW"
            if telegram_ready
            else "PAPER_SHADOW_ONLY"
        ),
        "telegram_signal_blockers": telegram_blockers,
        "profitability_proven": bool(
            economics and economics.get("best_validated_family") is not None
        ),
        "account_health_status": account.get("status", "MISSING"),
        "control_state": control_state or "UNKNOWN",
        "external_inventory_status": external.get("status", "MISSING"),
        "sources": {
            "account_health": str(
                output / "operations" / "live_account_health.json"
            ),
            "external_inventory": str(
                output / "operations" / "external_inventory_remediation.json"
            ),
            "telegram_evidence": str(
                output / "operations" / "telegram_signal_evidence.json"
            ),
            "canonical_economics": str(output / "economics" / "latest.json"),
        },
    }


def build_maturity_ladder(
    workspace: Path,
    *,
    beginner_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if beginner_artifact is None:
        path = workspace / "output" / "roadmap" / "beginner_foundation.json"
        beginner_artifact = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    certifications: dict[str, dict[str, Any]] = {"BEGINNER": beginner_artifact}
    for level in ("INTERMEDIATE", "ADVANCED", "EXPERT"):
        path = (
            workspace
            / "output"
            / "roadmap"
            / f"{level.lower()}_certification.json"
        )
        certifications[level] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
    projects: list[dict[str, Any]] = []
    prior_level_certified = True
    for project_id, name in enumerate(PROJECT_NAMES, start=1):
        level = _level(project_id)
        paths = EVIDENCE[project_id]
        present = [path for path in paths if (workspace / path).exists()]
        certification = certifications[level]
        gate = next(
            (
                item
                for item in certification.get("project_gates", [])
                if item.get("project_id") == project_id
            ),
            {},
        )
        level_certified = certification.get("status") == f"{level}_CERTIFIED"
        if level_certified and prior_level_certified and gate.get("passed"):
            status = "CERTIFIED"
        elif project_id <= 4:
            status = "NOT_CERTIFIED"
        elif len(present) == len(paths):
            status = "IMPLEMENTED_AWAITING_LEVEL_CERTIFICATION"
        elif present:
            status = "PARTIAL_IMPLEMENTATION"
        else:
            status = "NOT_STARTED"
        projects.append(
            {
                "project_id": project_id,
                "level": level,
                "name": name,
                "status": status,
                "evidence_present": present,
                "evidence_required": list(paths),
            }
        )
        if project_id == max(LEVEL_RANGES[level]):
            prior_level_certified = prior_level_certified and level_certified

    levels: list[dict[str, Any]] = []
    prior_certified = True
    for name, ids in LEVEL_RANGES.items():
        items = [projects[index - 1] for index in ids]
        certified = bool(items) and all(item["status"] == "CERTIFIED" for item in items)
        if certified:
            status = "CERTIFIED"
        elif prior_certified:
            status = "ACTIVE"
        else:
            status = "WAITING_FOR_PREVIOUS_LEVEL"
        levels.append({"name": name, "status": status})
        prior_certified = prior_certified and certified

    next_project = next(
        (item for item in projects if item["status"] != "CERTIFIED"),
        None,
    )
    current = next((item["name"] for item in levels if item["status"] == "ACTIVE"), "COMPLETE")
    implementation_complete = all(
        item["status"] == "CERTIFIED" for item in levels
    )
    operational = _operational_readiness(workspace)
    return {
        "schema_version": "crypto_maturity_ladder_v1",
        "roadmap_order": "execution_early",
        "current_level": current,
        "certification_scope": "IMPLEMENTATION_AND_BOUNDED_TESTS_ONLY",
        "implementation_complete": implementation_complete,
        "current_operational_stage": (
            "LIVE_READY"
            if operational["autonomous_live_status"] == "READY"
            else "LIVE_VALIDATION_BLOCKED"
        ),
        "operational_readiness": operational,
        "levels": levels,
        "next_project": next_project,
        "projects": projects,
        "sequential_certification_enforced": True,
        "live_canary_enabled": False,
        "implementation_certification_grants_live_authority": False,
        "side_effects": {
            "orders_submitted": 0,
            "exchange_mutations": 0,
            "trading_authority_changed": False,
        },
    }
