from __future__ import annotations

import re
from typing import Any

from .common import fetch_with_payload, parse_generic_articles
from src.utils import clean_text, parse_table

BASE = "https://www.lseg.com/en"
_HEADERS = {"Referer": "https://www.lseg.com/"}


async def scrape_lseg_async() -> dict[str, Any]:
    soup, payload = await fetch_with_payload(url=BASE, source="lseg", headers=_HEADERS)
    if soup is None:
        return payload

    payload["news"] = parse_generic_articles(soup, limit=50)

    payload["products"] = []
    seen: set[str] = set()
    for item in soup.find_all(class_=re.compile(r"product|solution|service|capability|offering", re.I)):
        name = clean_text(item.get_text(" "))
        if not name or name in seen:
            continue
        seen.add(name)
        payload["products"].append({"name": name[:200]})

    payload["market_data"] = []
    for tbl in soup.find_all("table"):
        payload["market_data"].extend(parse_table(tbl))

    payload["key_links"] = []
    kw = re.compile(r"data|market|trading|analytics|index|fixed|equity|fx|commodity|workspace|feed", re.I)
    for a in soup.find_all("a", href=True):
        href = clean_text(a.get("href"))
        txt = clean_text(a.get_text())
        if kw.search(href) or kw.search(txt):
            payload["key_links"].append({"text": txt[:80], "url": href})

    payload["key_links"] = payload["key_links"][:50]
    return payload
