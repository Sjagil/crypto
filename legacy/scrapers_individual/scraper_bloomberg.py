from __future__ import annotations

from typing import Any

from .common import fetch_with_payload, parse_generic_articles, parse_tables_and_widgets

BASE = "https://www.bloomberg.com"
_HEADERS = {
    "Referer": "https://www.bloomberg.com/",
    "Cache-Control": "no-cache",
}


async def scrape_bloomberg_async() -> dict[str, Any]:
    sections = ["/europe", "/finance", "/economics", "/markets", "/technology"]
    output: dict[str, Any] = {
        "source": "bloomberg",
        "scraped_at": None,
        "sections": {},
    }

    for path in sections:
        url = f"{BASE}{path}"
        soup, payload = await fetch_with_payload(url=url, source="bloomberg", headers=_HEADERS)
        if soup is not None:
            payload["articles"] = parse_generic_articles(soup, limit=50)
            payload["market_data"] = parse_tables_and_widgets(soup)
            payload["counts"] = {
                "articles": len(payload["articles"]),
                "market_data": len(payload["market_data"]),
                "embedded_json": len(payload["embedded_json"]),
            }
        output["sections"][path.lstrip("/")] = payload
        output["scraped_at"] = payload["scraped_at"]

    return output
