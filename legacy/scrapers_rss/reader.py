from __future__ import annotations

import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable

from .deduplication import classify_versions
from .normalizer import normalize_feed
from .registry import registered_feeds


Fetcher = Callable[[str, dict[str, str], int], bytes]


def _fetch(url: str, headers: dict[str, str], timeout: int) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(2_000_000)


def collect_registered_feeds(*, fetcher: Fetcher = _fetch, timeout_seconds: int = 15) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    feeds: list[dict[str, Any]] = []
    headers = {"User-Agent": "StocksResearch/11.3 research-contact@example.com", "Accept": "application/rss+xml, application/atom+xml, text/xml"}
    for spec in registered_feeds():
        try:
            payload = fetcher(spec.url, headers, max(1, min(timeout_seconds, 30)))
            normalized = normalize_feed(payload, spec)
            rows.extend(normalized)
            feeds.append(spec.public_dict() | {"status": "RSS_FORWARD_ONLY", "record_count": len(normalized)})
        except (OSError, ValueError, ET.ParseError, urllib.error.URLError):
            feeds.append(spec.public_dict() | {"status": "RSS_FETCH_BLOCKED", "record_count": 0})
    return {"status": "GO" if rows else "PARTIAL", "rows": classify_versions(rows), "feeds": feeds}
