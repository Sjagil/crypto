"""Certify roadmap levels with bounded tests and sequential governance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reporting.crypto_maturity_ladder import EVIDENCE, LEVEL_RANGES, build_maturity_ladder

LEVEL_TESTS: dict[str, tuple[str, ...]] = {
    "INTERMEDIATE": (
        "tests/test_backtest_math.py",
        "tests/test_canonical_economics.py",
        "tests/test_simple_strategy_lab.py",
        "tests/test_optimization_risk.py",
        "tests/test_execution_authority.py",
        "tests/test_event_driven_paper.py",
        "tests/test_canonical_execution_state.py",
        "tests/test_execution_evidence.py",
        "tests/test_canary_guard.py",
        "tests/test_stochastic_validation.py",
        "tests/test_portfolio_selection.py",
        "tests/test_absolute_momentum.py",
        "tests/test_market_intelligence.py",
        "tests/test_macro_liquidity_rotation.py",
    ),
    "ADVANCED": (
        "tests/test_volatility_contraction.py",
        "tests/test_portfolio_selection.py",
        "tests/test_portfolio_controls.py",
        "tests/test_hmm_regime_manager.py",
        "tests/test_regime_router.py",
        "tests/test_microstructure_observer.py",
        "tests/test_market_mechanics.py",
        "tests/test_multi_source_platform.py",
        "tests/test_database_macro_derivatives.py",
        "tests/test_event_driven_playbooks.py",
    ),
    "EXPERT": (
        "tests/test_hmm_strategy_comparison.py",
        "tests/test_multi_alpha_ensemble.py",
        "tests/test_decision_attribution.py",
        "tests/test_autonomous_trading.py",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _previous_level_certified(workspace: Path, level: str) -> bool:
    sequence = list(LEVEL_RANGES)
    index = sequence.index(level)
    if index == 0:
        return True
    previous = sequence[index - 1]
    filename = (
        "beginner_foundation.json"
        if previous == "BEGINNER"
        else f"{previous.lower()}_certification.json"
    )
    expected = f"{previous}_CERTIFIED"
    return _read_json(workspace / "output" / "roadmap" / filename).get("status") == expected


def run_level_tests(workspace: Path, level: str) -> dict[str, Any]:
    tests = LEVEL_TESTS[level]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "command": ["python", "-m", "pytest", *tests, "-q"],
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "output_tail": output[-6000:],
        "test_files": list(tests),
    }


def build_level_certification(
    workspace: Path,
    level: str,
    *,
    validation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = workspace.resolve()
    level = level.upper()
    if level not in LEVEL_TESTS:
        raise ValueError(f"unsupported certifiable level: {level}")
    prior_certified = _previous_level_certified(workspace, level)
    validation = validation or run_level_tests(workspace, level)
    project_gates: list[dict[str, Any]] = []
    for project_id in LEVEL_RANGES[level]:
        required = EVIDENCE[project_id]
        checks = {
            "all_evidence_present": all((workspace / path).exists() for path in required),
            "bounded_validation_passed": validation.get("passed") is True,
            "previous_level_certified": prior_certified,
        }
        project_gates.append(
            {
                "project_id": project_id,
                "passed": all(checks.values()),
                "checks": checks,
                "evidence_required": list(required),
            }
        )

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    all_passed = all(item["passed"] for item in project_gates)
    artifact: dict[str, Any] = {
        "schema_version": "crypto_level_certification_v1",
        "level": level,
        "generated_at": generated_at,
        "status": f"{level}_CERTIFIED" if all_passed else f"{level}_BLOCKED",
        "previous_level_certified": prior_certified,
        "project_gates": project_gates,
        "validation": validation,
        "claims": {
            "live_profitability_proven": False,
            "live_trading_enabled_by_certification": False,
            "telegram_signal_accuracy_validated": False,
        },
        "side_effects": {
            "orders_submitted": 0,
            "exchange_mutations": 0,
            "trading_authority_changed": False,
        },
    }
    digest = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact["artifact_id"] = f"{level}_{digest[:16]}"
    artifact["content_hash"] = digest
    output_root = workspace / "output" / "roadmap"
    filename = f"{level.lower()}_certification.json"
    _write_json(output_root / "runs" / artifact["artifact_id"] / filename, artifact)
    _write_json(output_root / filename, artifact)
    ladder = build_maturity_ladder(workspace)
    _write_json(output_root / "crypto_maturity_ladder.json", ladder)
    return artifact, ladder
