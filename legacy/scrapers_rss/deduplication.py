from __future__ import annotations

from typing import Any


def classify_versions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        key = str(row.get("entry_id_hash"))
        payload_hash = str(row.get("payload_hash"))
        if key not in seen:
            row["deduplication_status"] = "RSS_ENTRY_UNIQUE"
        elif seen[key] == payload_hash:
            row["deduplication_status"] = "RSS_ENTRY_DUPLICATE"
        else:
            row["deduplication_status"] = "RSS_ENTRY_UPDATED"
        seen[key] = payload_hash
        out.append(row)
    return out
