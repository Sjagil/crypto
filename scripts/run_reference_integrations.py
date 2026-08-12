"""Execute all pinned local reference functions and persist hashed evidence."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from research.reference_integrations import run_reference_integration_probes  # noqa: E402
from utils.common import atomic_write_json  # noqa: E402


def main() -> int:
    workspace = WORKSPACE
    result = run_reference_integration_probes(workspace)
    target = workspace / "output" / "reference_integrations" / "latest.json"
    atomic_write_json(target, result)
    print(target)
    print(result["evidence_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
