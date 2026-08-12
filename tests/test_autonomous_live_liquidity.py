from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from config.settings import Settings
from core.autonomous_trading import _bitvavo_entry_liquidity


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status = 200
        self._payload = payload

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self, **_: object) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self, *, book: dict[str, Any], ticker: dict[str, Any]) -> None:
        self.book = book
        self.ticker = ticker

    def get(self, url: str, **_: object) -> _Response:
        return _Response(self.book if url.endswith("/book") else self.ticker)


@pytest.mark.asyncio
async def test_entry_liquidity_passes_for_deep_tight_book(
    isolated_settings: Settings,
) -> None:
    session = _Session(
        book={
            "bids": [["1999", "10"]],
            "asks": [["2001", "10"], ["2002", "10"]],
        },
        ticker={"last": "2000", "volume": "10000"},
    )
    result = await _bitvavo_entry_liquidity(
        session,
        market="ETH-EUR",
        requested_notional_eur=Decimal("5"),
        settings=isolated_settings,
    )
    assert result["status"] == "PASSED"
    assert result["blocking_reasons"] == []


@pytest.mark.asyncio
async def test_entry_liquidity_blocks_wide_shallow_npc_book(
    isolated_settings: Settings,
) -> None:
    session = _Session(
        book={
            "bids": [["0.0040", "100"]],
            "asks": [["0.0050", "100"]],
        },
        ticker={"last": "0.0045", "volume": "1000"},
    )
    result = await _bitvavo_entry_liquidity(
        session,
        market="NPC-EUR",
        requested_notional_eur=Decimal("5"),
        settings=isolated_settings,
    )
    assert result["status"] == "BLOCKED"
    assert "SPREAD_LIMIT_EXCEEDED" in result["blocking_reasons"]
    assert "MINIMUM_VISIBLE_ASK_DEPTH_NOT_MET" in result["blocking_reasons"]
