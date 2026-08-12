"""Launch the independent P1.2.1 public/read-only collector."""

from __future__ import annotations

import argparse
import asyncio
import json

from config.settings import Settings
from data.multi_source_runtime import run_multi_source_collector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()
    payload = asyncio.run(
        run_multi_source_collector(
            Settings.load(),
            duration_seconds=args.duration_seconds,
        )
    )
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
