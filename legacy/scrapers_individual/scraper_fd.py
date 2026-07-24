from __future__ import annotations

import re
from typing import Any

from .common import fetch_with_payload, parse_generic_articles
from src.utils import clean_text, parse_table

FD_BASE = "https://fd.nl"
BEURS_BASE = "https://beurs.fd.nl"
_HEADERS = {
    "Referer": "https://fd.nl/",
    "Accept-Language": "nl-NL,nl;q=0.9",
}


def _parse_financial_rows(soup: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for tbl in soup.find_all("table"):
        records.extend(parse_table(tbl))

    if records:
        return records

    row_cls = re.compile(r"(row|item|security|instrument|quote|stock|bond|option|fund)", re.I)
    for row in soup.find_all(["div", "li", "tr"], class_=row_cls):
        text = clean_text(row.get_text(" "))
        if not text:
            continue
        records.append({"raw": text[:300]})

    return records


async def scrape_fd_async() -> dict[str, Any]:
    endpoints = {
        "fd_beurs": f"{FD_BASE}/beurs",
        "aex_analyse": f"{BEURS_BASE}/analyse/amsterdam/aex/",
        "dj30": f"{BEURS_BASE}/aandelen/nyse/dj30/",
        "indices_amsterdam": f"{BEURS_BASE}/indices/amsterdam/",
        "obligaties": f"{BEURS_BASE}/obligaties/binnenland/",
        "opties": f"{BEURS_BASE}/derivaten/opties/?call=AEX.IWDA/O",
    }

    out: dict[str, Any] = {"source": "fd", "scraped_at": None, "pages": {}}
    for key, url in endpoints.items():
        soup, payload = await fetch_with_payload(url=url, source="fd", headers=_HEADERS)
        if soup is not None:
            payload["financial_rows"] = _parse_financial_rows(soup)
            payload["news_titles"] = parse_generic_articles(soup, limit=30)
            payload["count_rows"] = len(payload["financial_rows"])
        out["pages"][key] = payload
        out["scraped_at"] = payload["scraped_at"]
    return out
