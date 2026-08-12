"""Build the audit-only reference-integration Phase A artifact."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from reporting.reference_integration_phase_a import (  # noqa: E402
    build_phase_a_inventory,
)


def main() -> int:
    result = build_phase_a_inventory(WORKSPACE)
    print(result["artifact_path"])
    print(result["artifact_hash"])
    print(result["phase_status"])
    return 0 if result["phase_status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
