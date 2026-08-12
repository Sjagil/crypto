"""Create the P1.3 preregistration or authorize a future frozen-data run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.p1_3_governance import (
    ALLOWED_SEEDS,
    P13ResearchRunner,
    PreregistrationStore,
    ResearchGateError,
    bindings_from_workspace,
    build_preregistration_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "preregister-p1-3",
        help="freeze the metadata-only P1.3 experiment plan",
    )
    run = commands.add_parser(
        "run-p1-3",
        help="guarded authorization only; does not execute research",
    )
    run.add_argument("--preregistration-id")
    run.add_argument("--dataset-freeze-id")
    run.add_argument("--seed", type=int, default=ALLOWED_SEEDS[0])
    run.add_argument(
        "--phase",
        choices=("DEVELOPMENT", "FINAL_HOLDOUT"),
        default="DEVELOPMENT",
    )
    run.add_argument("--candidate-hash")
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = args.workspace.resolve()
    governance_root = workspace / "output" / "research_governance" / "p1_3"
    if args.command == "preregister-p1-3":
        bindings = bindings_from_workspace(workspace)
        plan = build_preregistration_plan(bindings)
        payload = PreregistrationStore(governance_root).create(plan)
        path = (
            governance_root
            / "preregistrations"
            / payload["preregistration_id"]
            / "P1_3_PREREGISTRATION_V1.json"
        )
        print(path.resolve())
        print(payload["preregistration_id"])
        print(payload["content_hash"])
        return 0
    runner = P13ResearchRunner(workspace, governance_root)
    try:
        authorization = runner.authorize(
            preregistration_id=args.preregistration_id,
            dataset_freeze_id=args.dataset_freeze_id,
            seed=args.seed,
            phase=args.phase,
            candidate_hash=args.candidate_hash,
        )
    except ResearchGateError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(authorization, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
