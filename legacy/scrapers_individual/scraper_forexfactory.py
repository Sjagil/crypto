from __future__ import annotations

import re
from typing import Any

from .common import fetch_with_payload, parse_generic_articles
from src.utils import clean_text, parse_table

BASE = "https://www.forexfactory.com"
_HEADERS = {"Referer": BASE + "/"}


def _calendar_events(soup: Any) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    table = soup.find("table", class_=re.compile(r"calendar", re.I))
    if not table:
        return events

    current_date = ""
    for row in table.find_all("tr", class_=re.compile(r"calendar_row|flexgrid-row", re.I)):
        date_td = row.find("td", class_=re.compile(r"\bdate\b", re.I))
        if date_td and clean_text(date_td.get_text()):
            current_date = clean_text(date_td.get_text())

        event = {
            "date": current_date,
            "time": clean_text((row.find("td", class_=re.compile(r"\btime\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\btime\b", re.I)) else ""),
            "currency": clean_text((row.find("td", class_=re.compile(r"\bcurrency\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\bcurrency\b", re.I)) else ""),
            "event": clean_text((row.find("td", class_=re.compile(r"\bevent\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\bevent\b", re.I)) else ""),
            "actual": clean_text((row.find("td", class_=re.compile(r"\bactual\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\bactual\b", re.I)) else ""),
            "forecast": clean_text((row.find("td", class_=re.compile(r"\bforecast\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\bforecast\b", re.I)) else ""),
            "previous": clean_text((row.find("td", class_=re.compile(r"\bprevious\b", re.I)) or {}).get_text() if row.find("td", class_=re.compile(r"\bprevious\b", re.I)) else ""),
        }
        if event["event"]:
            events.append(event)
    return events


async def scrape_forexfactory_async() -> dict[str, Any]:
    pages = {
        "calendar": f"{BASE}/calendar",
        "market_eurusd": f"{BASE}/market/eurusd",
        "news": f"{BASE}/news",
        "home": BASE,
    }

    out: dict[str, Any] = {"source": "forexfactory", "scraped_at": None, "pages": {}}
    for key, url in pages.items():
        soup, payload = await fetch_with_payload(url=url, source="forexfactory", headers=_HEADERS)
        if soup is not None:
            payload["articles"] = parse_generic_articles(soup, limit=40)
            payload["tables"] = []
            for tbl in soup.find_all("table"):
                payload["tables"].extend(parse_table(tbl))
            if key == "calendar":
                payload["calendar_events"] = _calendar_events(soup)
                payload["calendar_count"] = len(payload["calendar_events"])
        out["pages"][key] = payload
        out["scraped_at"] = payload["scraped_at"]
    return out
