"""Build the required A-AD final integration report from verification evidence."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from reporting.reference_integration_final import (  # noqa: E402
    build_reference_integration_final,
)
from utils.common import read_json  # noqa: E402


def main() -> int:
    verification_path = (
        WORKSPACE / "output" / "reference_integration" / "verification.json"
    )
    if not verification_path.is_file():
        raise SystemExit(f"verification artifact missing: {verification_path}")
    result = build_reference_integration_final(
        WORKSPACE,
        verification=read_json(verification_path),
    )
    print(result["markdown_path"])
    print(result["artifact_hash"])
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
