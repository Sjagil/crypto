"""Build the immutable P1.2.2 A-V evidence report."""

from __future__ import annotations

import argparse
import json

from config.settings import Settings
from reporting.multi_source_maturation_platform import build_maturation_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    payload = build_maturation_evidence(Settings.load())
    selected = (
        {
            "run_id": payload["run_id"],
            "evidence_path": payload["evidence_path"],
            "latest": payload["latest"],
            "exact_next_action": payload["sections"]["V"]["action"],
            "family_states": payload["sections"]["O"]["states"],
            "test_status": payload["sections"]["T"]["p1_2_2"]["status"],
        }
        if args.summary
        else payload
    )
    print(json.dumps(selected, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
