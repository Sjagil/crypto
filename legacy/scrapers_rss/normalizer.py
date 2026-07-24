from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any

from .contracts import FeedSpec


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def normalize_feed(content: bytes, spec: FeedSpec, *, first_seen_at: str | None = None) -> list[dict[str, Any]]:
    first_seen = first_seen_at or datetime.now(UTC).isoformat()
    root = ET.fromstring(content)
    rows: list[dict[str, Any]] = []
    entries = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for entry in entries:
        title = _text(entry, "title")
        link = _link(entry)
        entry_id = _text(entry, "guid") or _text(entry, "id") or link or title
        published = _text(entry, "pubDate") or _text(entry, "published")
        updated = _text(entry, "updated")
        identity = f"{spec.feed_id}|{entry_id}"
        row = {
            "feed_id": spec.feed_id,
            "publisher": spec.publisher,
            "feed_url_hash": _hash(spec.url),
            "entry_id_hash": _hash(identity),
            "title": title[:500],
            "link_hash": _hash(link) if link else None,
            "published_at": published or None,
            "updated_at": updated or None,
            "first_seen_at": first_seen,
            "symbols": [],
            "tags": list(spec.datasets),
            "language": spec.language,
            "historical_coverage": spec.historical_coverage,
        }
        row["payload_hash"] = _hash(str(sorted(row.items())))
        row["status"] = "RSS_FORWARD_ONLY" if published else "RSS_TIMESTAMP_MISSING"
        rows.append(row)
    return rows


def _text(entry: ET.Element, local_name: str) -> str:
    for child in entry.iter():
        if child.tag.rsplit("}", 1)[-1] == local_name and child.text:
            return child.text.strip()
    return ""


def _link(entry: ET.Element) -> str:
    for child in entry.iter():
        if child.tag.rsplit("}", 1)[-1] == "link":
            return (child.get("href") or child.text or "").strip()
    return ""
