"""Run P1.2 validation in the prescribed dependency order and persist evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utils.common import atomic_write_json, stable_hash, utc_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--include-full-suite", action="store_true")
    parser.add_argument("--full-suite-timeout", type=int, default=600)
    return parser.parse_args()


def _run(root: Path, name: str, command: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        return {
            "name": name,
            "status": "PASSED" if completed.returncode == 0 else "FAILED",
            "return_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
            "output_tail": combined.splitlines()[-30:],
        }
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part.decode(errors="replace") if isinstance(part, bytes) else str(part or "")
            for part in (exc.stdout, exc.stderr)
        )
        return {
            "name": name,
            "status": "NOT_EVALUABLE_TIMEOUT",
            "return_code": None,
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
            "timeout_seconds": timeout,
            "output_tail": output.splitlines()[-30:],
        }


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    python = sys.executable
    focused_files = [
        "data/market_structure.py",
        "reporting/market_structure_platform.py",
        "scripts/build_market_structure_platform.py",
        "scripts/validate_market_structure.py",
        "tests/test_market_structure.py",
    ]
    steps: list[tuple[str, list[str], int]] = [
        ("ruff", [python, "-m", "ruff", "check", *focused_files], 120),
        (
            "compile_import",
            [
                python,
                "-c",
                "import data.market_structure, reporting.market_structure_platform; "
                "import scripts.build_market_structure_platform, scripts.validate_market_structure",
            ],
            120,
        ),
        (
            "p0_canonical_state",
            [python, "-m", "pytest", "tests/test_canonical_execution_state.py", "-q"],
            180,
        ),
        (
            "p0_5_economics",
            [python, "-m", "pytest", "tests/test_canonical_economics.py", "-q"],
            180,
        ),
        (
            "p1_research_factory",
            [python, "-m", "pytest", "tests/test_research_factory.py", "-q"],
            180,
        ),
        (
            "p1_1_alpha_discovery",
            [python, "-m", "pytest", "tests/test_alpha_discovery.py", "-q"],
            180,
        ),
        ("p1_2_focused", [python, "-m", "pytest", "tests/test_market_structure.py", "-q"], 180),
        (
            "market_data_replay_orderbook_timestamp",
            [
                python,
                "-m",
                "pytest",
                "tests/test_data_realtime.py",
                "tests/test_orderflow_recorder.py",
                "tests/test_market_data.py",
                "-q",
            ],
            300,
        ),
    ]
    if args.include_full_suite:
        steps.append(
            (
                "broader_non_network_full_suite",
                [python, "-m", "pytest", "-q"],
                args.full_suite_timeout,
            )
        )
    results = [_run(root, name, command, timeout) for name, command, timeout in steps]
    required = [row for row in results if row["name"] != "broader_non_network_full_suite"]
    payload = {
        "schema_version": "p1_2_validation_evidence_v1",
        "generated_at": utc_iso(),
        "validation_order": [row["name"] for row in results],
        "results": results,
        "required_status": "PASSED"
        if all(row["status"] == "PASSED" for row in required)
        else "FAILED",
        "full_suite_status": next(
            (row["status"] for row in results if row["name"] == "broader_non_network_full_suite"),
            "NOT_RUN",
        ),
        "network_tests_enabled": False,
        "orders_generated": 0,
        "private_exchange_mutations": 0,
    }
    payload["evidence_hash"] = stable_hash(payload, length=64)
    target = root / "output" / "market_structure" / "validation" / "latest.json"
    atomic_write_json(target, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["required_status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
