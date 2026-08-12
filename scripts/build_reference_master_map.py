"""Build the machine-readable nine-repository reference master map."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from reporting.reference_repository_master_map import (  # noqa: E402
    build_reference_master_map,
)


def main() -> int:
    result = build_reference_master_map(WORKSPACE)
    print(result["artifact_path"])
    print(result["artifact_hash"])
    print("PASSED" if result["all_integrations_complete"] else "FAILED")
    return 0 if result["all_integrations_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
