"""Verify a controlled multi-source collector restart against a saved status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import Settings
from data.multi_source_maturation import (
    record_deployment_event,
    verify_restart_continuity,
)
from utils.common import atomic_write_json, read_json, stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-status", type=Path, required=True)
    parser.add_argument("--reason", default="P1.2.2_CONTROLLED_DEPLOYMENT")
    args = parser.parse_args()
    settings = Settings.load()
    previous = dict(read_json(args.previous_status))
    current = dict(read_json(settings.paths.output_dir / "multi_source" / "status.json"))
    continuity = verify_restart_continuity(previous, current)
    audit_id = stable_hash(continuity, length=32)
    audit_path = (
        settings.paths.output_dir
        / "multi_source"
        / "deployments"
        / f"continuity-{audit_id}.json"
    )
    atomic_write_json(audit_path, continuity)
    deployment = record_deployment_event(
        settings.paths.output_dir / "multi_source" / "deployments" / "history",
        instance_id=str((current.get("ownership") or {}).get("instance_id") or "UNKNOWN"),
        previous_status=previous,
        current_status=current,
        reason=args.reason,
        continuity=continuity,
    )
    payload = {
        "continuity": continuity,
        "continuity_path": str(audit_path.resolve()),
        "deployment": deployment,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if continuity["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
