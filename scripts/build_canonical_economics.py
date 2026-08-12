"""Build the immutable P0.5 canonical strategy-family economics artifact."""

from __future__ import annotations

import json

from config.settings import get_settings
from reporting.canonical_economics import build_canonical_strategy_economics


def main() -> int:
    result = build_canonical_strategy_economics(get_settings())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
