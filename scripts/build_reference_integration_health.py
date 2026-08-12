"""Build the audit-only A-J reference integration health artifact."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from reporting.reference_integration_health import (  # noqa: E402
    build_reference_integration_health,
)


def main() -> int:
    result = build_reference_integration_health(WORKSPACE)
    print(result["artifact_path"])
    print(result["artifact_hash"])
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
