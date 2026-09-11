

from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from data.bitvavo_market_data_pro import (
    BitvavoMarketDataProManager,
    L3_STATUS,
)
from data.orderbook_l2 import Level2OrderBook


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def test_mdpro_auth_signature_is_documented_hmac():
    manager = BitvavoMarketDataProManager(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        markets=("BTC-EUR",),
    )
    timestamp = 1548175200641
    message = manager.authentication_message(timestamp_ms=timestamp)
    expected = hmac.new(
        b"secret",
        f"{timestamp}GET/v2/websocket".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert message["signature"] == expected
    assert "secret" not in str(manager.health("bitvavo"))


@pytest.mark.asyncio
async def test_range_sequence_delta_advances_to_end_sequence():
    book = Level2OrderBook(provider="unit", market="BTC-EUR", maximum_depth=1000)
    await book.initialize(
        bids=[["100", "2"]],
        asks=[["101", "2"]],
        sequence=100,
    )
    applied = await book.apply_delta_range(
        bids=[["100", "3"]],
        asks=[["101", "1.5"]],
        start_sequence=101,
        end_sequence=103,
    )
    assert applied is True
    assert book.sequence == 103
    assert book.valid is True


@pytest.mark.asyncio
async def test_manager_snapshot_then_buffered_event_is_causal():
    manager = BitvavoMarketDataProManager(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        markets=("BTC-EUR",),
    )
    manager._socket = FakeSocket()
    await manager._handle_book(
        {
            "event": "book",
            "market": "BTC-EUR",
            "bids": [["100", "2"]],
            "asks": [],
            "startMdSeqNo": 11,
            "endMdSeqNo": 11,
            "timestamp": 1752139200000000000,
        }
    )
    assert manager.queue.qsize() == 0
    await manager._handle_snapshot(
        {
            "market": "BTC-EUR",
            "bids": [["99", "2"]],
            "asks": [["101", "2"]],
            "mdSeqNo": 10,
            "timestamp": 1752139200000000000,
        }
    )
    event = await manager.next_event(timeout=0.1)
    assert event.sequence == 11
    assert event.payload["from_sequence"] == 11
    assert manager.snapshot()["markets"]["BTC-EUR"]["sequence"] == 11


def test_l3_is_explicitly_unsupported():
    manager = BitvavoMarketDataProManager(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        markets=("BTC-EUR",),
    )
    assert manager.health("bitvavo")["l3_status"] == L3_STATUS
    assert manager.health("bitvavo")["individual_order_ids_available"] is False


def test_mdpro_auth_success_shapes_are_strict_but_compatible():
    manager = BitvavoMarketDataProManager(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        markets=("BTC-EUR",),
    )
    assert manager._authentication_confirmed(
        {"event": "authenticate", "authenticated": True}
    )
    assert manager._authentication_confirmed(
        {"action": "authenticate", "response": {"authenticated": True}}
    )
    assert manager._authentication_confirmed(
        {"event": "authenticated", "authenticated": True}
    )
    assert not manager._authentication_confirmed(
        {"action": "authenticate", "response": {}}
    )
    assert not manager._authentication_confirmed(
        {"event": "authenticate", "authenticated": False}
    )
    assert not manager._authentication_confirmed(
        {"event": "ticker", "authenticated": True}
    )
