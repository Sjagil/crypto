from __future__ import annotations

from collections import Counter
from typing import Any


def audit_entries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("deduplication_status") or row.get("status")) for row in rows)
    return {
        "status": "GO" if not statuses.get("RSS_TIMESTAMP_MISSING") else "PARTIAL",
        "entry_count": len(rows),
        "status_counts": dict(statuses),
        "historical_coverage": "FORWARD_ONLY",
        "publication_dates_reconstructed": False,
    }
