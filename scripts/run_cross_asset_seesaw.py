"""Run the orderless cross-asset seesaw falsification campaign."""

from __future__ import annotations

import json

from config.settings import get_settings
from research.cross_asset_seesaw import run_cross_asset_seesaw_campaign


def main() -> int:
    result = run_cross_asset_seesaw_campaign(get_settings())
    summary = {
        key: result[key]
        for key in (
            "status",
            "trial_count",
            "paper_candidate_count",
            "live_ready",
            "orders_generated",
            "orders_submitted",
            "private_exchange_requests",
            "artifact_hash",
            "artifact_path",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
