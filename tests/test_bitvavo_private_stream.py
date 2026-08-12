from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import aiohttp
import pytest
from pydantic import SecretStr

from data.bitvavo_private_stream import (
    BITVAVO_WEBSOCKET_PATH,
    BitvavoPrivateAccountStream,
    sanitize_account_event,
)


class _FakeSocket:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.sent: list[dict[str, object]] = []

    async def __aenter__(self) -> "_FakeSocket":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def receive(self) -> SimpleNamespace:
        if self.payloads:
            return SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(self.payloads.pop(0)),
            )
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data="")

    async def ping(self) -> None:
        return None


def test_bitvavo_websocket_auth_signature_and_health_are_secret_safe() -> None:
    stream = BitvavoPrivateAccountStream(
        api_key=SecretStr("private-key-value"),
        api_secret=SecretStr("private-secret-value"),
    )
    timestamp = 1_700_000_000_000
    message = stream.authentication_message(timestamp_ms=timestamp)
    expected = hmac.new(
        b"private-secret-value",
        f"{timestamp}GET{BITVAVO_WEBSOCKET_PATH}".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert message["signature"] == expected
    assert message["key"] == "private-key-value"
    serialized_health = json.dumps(stream.health())
    assert "private-key-value" not in serialized_health
    assert "private-secret-value" not in serialized_health
    assert stream.health()["secrets_serialized"] is False


def test_private_account_event_masks_all_venue_identifiers() -> None:
    raw = {
        "event": "fill",
        "orderId": "venue-order-123",
        "fillId": "venue-fill-456",
        "clientOrderId": "client-order-789",
        "market": "ETH-EUR",
        "side": "buy",
        "price": "1700",
        "amount": "0.002",
    }
    event = sanitize_account_event(raw)
    assert event is not None
    serialized = json.dumps(event)
    assert event["event"] == "FILL"
    assert event["market"] == "ETH-EUR"
    assert event["order_public_id"].startswith("ord_")
    assert event["fill_public_id"].startswith("fill_")
    assert "venue-order-123" not in serialized
    assert "venue-fill-456" not in serialized
    assert "client-order-789" not in serialized


@pytest.mark.asyncio
async def test_private_stream_authenticates_subscribes_and_queues_fill() -> None:
    socket = _FakeSocket(
        [
            {"event": "authenticate", "authenticated": True},
            {"event": "subscribed", "channels": [{"name": "account"}]},
            {
                "event": "fill",
                "orderId": "order-1",
                "fillId": "fill-1",
                "market": "ETH-EUR",
                "side": "buy",
                "price": "1700",
                "amount": "0.002",
            },
        ]
    )
    stream = BitvavoPrivateAccountStream(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        connect=lambda *args, **kwargs: socket,
    )
    with pytest.raises(ConnectionError, match="closed"):
        await stream._connection()
    assert socket.sent[0]["action"] == "authenticate"
    assert socket.sent[1] == {
        "action": "subscribe",
        "channels": [
            {
                "name": "account",
                "markets": ["ETH-EUR"],
            }
        ],
    }
    event = stream.queue.get_nowait()
    assert event["event"] == "FILL"
    assert event["market"] == "ETH-EUR"
    assert stream.ready is False
    assert stream.health()["messages"] == 1


@pytest.mark.asyncio
async def test_private_stream_rejects_account_event_before_authentication() -> None:
    socket = _FakeSocket(
        [
            {
                "event": "order",
                "orderId": "order-1",
                "market": "ETH-EUR",
                "status": "new",
            }
        ]
    )
    stream = BitvavoPrivateAccountStream(
        api_key=SecretStr("key"),
        api_secret=SecretStr("secret"),
        connect=lambda *args, **kwargs: socket,
    )
    with pytest.raises(PermissionError, match="before subscription"):
        await stream._connection()
    assert stream.queue.empty()
