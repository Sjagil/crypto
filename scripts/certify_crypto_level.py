from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporting.crypto_level_certification import build_level_certification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        required=True,
        choices=("INTERMEDIATE", "ADVANCED", "EXPERT"),
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    artifact, ladder = build_level_certification(workspace, args.level)
    print(
        json.dumps(
            {
                "artifact_id": artifact["artifact_id"],
                "status": artifact["status"],
                "validation_passed": artifact["validation"]["passed"],
                "current_level": ladder["current_level"],
                "orders_submitted": 0,
            },
            indent=2,
        )
    )
    return 0 if artifact["status"].endswith("_CERTIFIED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
