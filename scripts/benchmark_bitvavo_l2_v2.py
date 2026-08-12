from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporting.bitvavo_l2_performance import benchmark_bitvavo_l2_v2
from utils.common import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/multi_source/p1_2_3/bitvavo_l2_performance_latest.json"),
    )
    args = parser.parse_args()
    result = benchmark_bitvavo_l2_v2(args.events)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
