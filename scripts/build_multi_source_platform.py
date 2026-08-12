"""Build the immutable P1.2.1 A-W evidence artifact."""

from __future__ import annotations

import json

from config.settings import Settings
from reporting.multi_source_data_platform import build_multi_source_evidence


def main() -> int:
    payload = build_multi_source_evidence(Settings.load())
    print(
        json.dumps(
            {
                "status": "PASSED" if payload["validation_passed"] else "FAILED",
                "artifact_path": payload["artifact_path"],
                "artifact_sha256": payload["artifact_sha256"],
                "artifact_hash": payload["artifact_hash"],
                "hard_completion_criteria_passed": payload["hard_completion_criteria_passed"],
                "exact_next_task": payload["sections"]["W"]["task"],
                "orders_generated": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
