"""Run the bounded P1.1 economically grounded alpha-discovery campaign."""

from __future__ import annotations

import argparse
import json

from config.settings import get_settings
from research.alpha_discovery import build_alpha_discovery_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-rows-4h", type=int, default=6_000)
    parser.add_argument("--maximum-rows-1d", type=int, default=1_200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_alpha_discovery_artifact(
        get_settings(),
        maximum_rows_4h=args.maximum_rows_4h,
        maximum_rows_1d=args.maximum_rows_1d,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
