"""Run the bounded P1 research factory without granting trading authority."""

from __future__ import annotations

import argparse
import json

from config.settings import get_settings
from research.research_factory import build_research_factory_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maximum-rows",
        type=int,
        default=20_000,
        help="maximum closed candles frozen per market/timeframe (minimum 100)",
    )
    parser.add_argument(
        "--stage0-only",
        action="store_true",
        help="skip native exact validation even if Stage 0 finds a survivor",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_research_factory_artifact(
        get_settings(),
        maximum_rows=args.maximum_rows,
        execute_exact=not args.stage0_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
