from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeedSpec:
    feed_id: str
    publisher: str
    url: str
    language: str
    datasets: tuple[str, ...]
    historical_coverage: str = "FORWARD_ONLY"

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("url")
        return payload
