"""Build the immutable P1.2.4 A-V completion artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from reporting.p1_2_4_preregistration import build_p1_2_4_evidence
from utils.common import atomic_write_json, read_json, stable_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    artifact = build_p1_2_4_evidence(workspace, args.preregistration, args.validation)
    root = workspace / "output" / "research_governance" / "p1_2_4"
    immutable = root / "runs" / artifact["run_id"] / "p1_2_4_evidence.json"
    if immutable.is_file():
        if stable_hash(read_json(immutable)) != stable_hash(artifact):
            raise RuntimeError("immutable P1.2.4 artifact collision")
    else:
        atomic_write_json(immutable, artifact)
    atomic_write_json(
        root / "latest.json",
        {
            "schema_version": "p1_2_4_latest_pointer_v1",
            "run_id": artifact["run_id"],
            "artifact_path": str(immutable.resolve()),
            "artifact_hash": artifact["artifact_hash"],
        },
    )
    print(immutable.resolve())
    print(artifact["artifact_hash"])
    return 0 if artifact["definition_of_done_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
