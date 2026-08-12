"""Build the read-only P1.2 market-structure data-platform evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporting.market_structure_platform import build_market_structure_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--no-local-scan",
        action="store_true",
        help="Build contract-only evidence without inspecting local data partitions.",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_market_structure_platform(
        args.project_root,
        scan_local_data=not args.no_local_scan,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
