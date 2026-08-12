"""Run ordered P1.2.4 validation without touching the live collector."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utils.common import atomic_write_json, stable_hash, utc_iso


def _run(workspace: Path, name: str, arguments: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        arguments,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    return {
        "name": name,
        "command": arguments,
        "return_code": result.returncode,
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "elapsed_seconds": time.perf_counter() - started,
        "output_hash": stable_hash(output),
        "output_tail": output[-4000:],
    }


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    python = sys.executable
    p13 = "tests/test_p1_3_governance.py"
    checks = [
        _run(
            workspace,
            "RUFF",
            [
                python,
                "-m",
                "ruff",
                "check",
                "research/p1_3_governance.py",
                "reporting/p1_2_4_preregistration.py",
                "scripts/p1_3_governance.py",
                "scripts/validate_p1_3_governance.py",
                "scripts/build_p1_2_4_artifact.py",
                p13,
                "tests/test_p1_2_4_preregistration.py",
            ],
        ),
        _run(
            workspace,
            "COMPILE_IMPORT",
            [
                python,
                "-m",
                "compileall",
                "-q",
                "research/p1_3_governance.py",
                "reporting/p1_2_4_preregistration.py",
                "scripts/p1_3_governance.py",
            ],
        ),
        _run(
            workspace,
            "P1_2_2_READINESS_REGRESSIONS",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_multi_source_maturation.py",
                "tests/test_multi_source_platform.py",
            ],
        ),
        _run(
            workspace,
            "P1_2_3_L2_REGRESSIONS",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_bitvavo_l2_reconstruction_v2.py",
                "tests/test_bitvavo_l2_recovery_report.py",
                "tests/test_bitvavo_l2_performance.py",
                "tests/test_bitvavo_l2_p1_2_3_artifact.py",
                "tests/test_multi_source_runtime_bitvavo_l2.py",
                "tests/test_reference_integrations.py",
            ],
        ),
        _run(
            workspace,
            "PREREGISTRATION_TESTS",
            [python, "-m", "pytest", "-q", p13],
        ),
        _run(
            workspace,
            "IMMUTABILITY_TESTS",
            [python, "-m", "pytest", "-q", p13, "-k", "immutable or modification or tamper"],
        ),
        _run(
            workspace,
            "TARGET_ACCESS_TESTS",
            [python, "-m", "pytest", "-q", p13, "-k", "target"],
        ),
        _run(
            workspace,
            "HOLDOUT_PROTECTION_TESTS",
            [python, "-m", "pytest", "-q", p13, "-k", "holdout"],
        ),
        _run(
            workspace,
            "RESEARCH_LEDGER_TESTS",
            [python, "-m", "pytest", "-q", p13, "-k", "ledger or failed"],
        ),
        _run(
            workspace,
            "AUTHORITY_TESTS",
            [python, "-m", "pytest", "-q", p13, "-k", "authority"],
        ),
        _run(
            workspace,
            "BOUNDED_REGRESSION_SUITE",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_research_factory.py",
                "tests/test_alpha_discovery.py",
                "tests/test_multi_source_maturation.py",
                "tests/test_multi_source_platform.py",
                p13,
                "tests/test_p1_2_4_preregistration.py",
            ],
        ),
    ]
    passed = all(row["status"] == "PASSED" for row in checks)
    body = {
        "schema_version": "p1_2_4_ordered_validation_v1",
        "generated_at": utc_iso(),
        "status": "PASSED" if passed else "FAILED",
        "collector_stop_requested": False,
        "collector_restart_requested": False,
        "checks": checks,
    }
    run_id = stable_hash(body)
    payload = {**body, "run_id": run_id}
    target = (
        workspace
        / "output"
        / "research_governance"
        / "p1_3"
        / "validation"
        / f"{run_id}.json"
    )
    atomic_write_json(target, payload)
    print(target.resolve())
    print(payload["status"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
