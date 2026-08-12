from __future__ import annotations

import argparse
from pathlib import Path

from reporting.bitvavo_l2_p1_2_3_artifact import build_p1_2_3_artifact
from utils.common import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=Path("output/multi_source/p1_2_3/deploy_20260811T0000CEST"),
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    artifact = build_p1_2_3_artifact(workspace, workspace / args.deploy_root)
    root = workspace / "output" / "multi_source" / "p1_2_3"
    immutable_path = root / f"{artifact['run_id']}.json"
    atomic_write_json(immutable_path, artifact)
    atomic_write_json(root / "P1_2_3_FINAL_LATEST.json", artifact)
    print(immutable_path)
    print(artifact["artifact_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
