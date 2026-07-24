from __future__ import annotations

from .contracts import FeedSpec


REGISTRY_VERSION = "stocks_phase11_3_rss_registry_v1"
FEEDS = (
    FeedSpec("SEC_PRESS_RELEASES", "SEC", "https://www.sec.gov/news/pressreleases.rss", "en", ("company_filings", "stock_news")),
    FeedSpec("ECB_PRESS_RELEASES", "ECB", "https://www.ecb.europa.eu/rss/press.html", "en", ("macro_events",)),
)


def registered_feeds() -> tuple[FeedSpec, ...]:
    return FEEDS
