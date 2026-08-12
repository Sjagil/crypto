from __future__ import annotations

import json
from pathlib import Path

from reporting.crypto_beginner_foundation import build_beginner_foundation


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    artifact, ladder = build_beginner_foundation(workspace)
    print(
        json.dumps(
            {
                "artifact_id": artifact["artifact_id"],
                "status": artifact["status"],
                "current_level": ladder["current_level"],
                "next_project": ladder["next_project"],
                "orders_submitted": 0,
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "BEGINNER_CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
