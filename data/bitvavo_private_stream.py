"""Authenticated Bitvavo order and fill stream with secret-safe health state."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import aiohttp
from pydantic import SecretStr

from utils.common import sha256_text, stable_json, utc_iso

BITVAVO_WEBSOCKET_URL = "wss://ws.bitvavo.com/v2/"
BITVAVO_WEBSOCKET_PATH = "/v2/websocket"
LOGGER = logging.getLogger("crypto.bitvavo.private_stream")


def _public_identity(value: Any, *, prefix: str) -> str | None:
    """Return a stable non-reversible identity for a venue identifier."""

    text = str(value or "").strip()
    if not text:
        return None
    return f"{prefix}_{sha256_text(text)[:20]}"


def _timestamp(value: Any) -> str:
    if value in (None, ""):
        return utc_iso()
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) >= 100_000_000_000:
            number /= 1_000
        return datetime.fromtimestamp(number, tz=UTC).isoformat()
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return utc_iso()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def sanitize_account_event(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize an account event without preserving private venue identities."""

    event_name = str(
        payload.get("event")
        or payload.get("type")
        or payload.get("eventType")
        or ""
    ).casefold()
    if event_name not in {"order", "fill"}:
        return None
    body = payload.get(event_name)
    values = dict(body) if isinstance(body, Mapping) else dict(payload)
    order_id = values.get("orderId") or values.get("order_id")
    fill_id = values.get("fillId") or values.get("fill_id") or values.get("tradeId")
    client_order_id = values.get("clientOrderId") or values.get("client_order_id")
    market = str(values.get("market") or "").upper()
    return {
        "event": event_name.upper(),
        "received_at": utc_iso(),
        "venue": "bitvavo",
        "market": market or None,
        "status": str(values.get("status") or "").upper() or None,
        "side": str(values.get("side") or "").upper() or None,
        "order_type": str(values.get("orderType") or values.get("type") or "").upper()
        or None,
        "order_public_id": _public_identity(order_id, prefix="ord"),
        "fill_public_id": _public_identity(fill_id, prefix="fill"),
        "client_order_public_id": _public_identity(client_order_id, prefix="client"),
        "amount": values.get("amount") or values.get("quantity"),
        "amount_remaining": values.get("amountRemaining")
        or values.get("remainingAmount"),
        "filled_amount": values.get("filledAmount")
        or values.get("amountFilled"),
        "price": values.get("price"),
        "fill_price": values.get("fillPrice") or values.get("price"),
        "fee": values.get("fee") or values.get("feePaid"),
        "fee_currency": values.get("feeCurrency"),
        "venue_timestamp": _timestamp(
            values.get("updated")
            or values.get("created")
            or values.get("timestamp")
        ),
        "raw_payload_hash": sha256_text(stable_json(payload)),
    }


class BitvavoPrivateAccountStream:
    """Maintain the authenticated account channel used for entry authority."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        api_secret: SecretStr,
        markets: tuple[str, ...] = ("ETH-EUR",),
        access_window_ms: int = 10_000,
        queue_size: int = 1_000,
        heartbeat_seconds: float = 20.0,
        inactivity_timeout_seconds: float = 90.0,
        session: aiohttp.ClientSession | None = None,
        connect: Callable[..., Any] | None = None,
        seed: int = 73,
    ) -> None:
        if queue_size < 1:
            raise ValueError("private account queue size must be positive")
        self._api_key = api_key
        self._api_secret = api_secret
        self.markets = tuple(
            dict.fromkeys(
                str(market).strip().upper()
                for market in markets
                if str(market).strip()
            )
        )
        if not self.markets:
            raise ValueError("private account stream requires at least one market")
        self.access_window_ms = access_window_ms
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(queue_size)
        self.heartbeat_seconds = heartbeat_seconds
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.session = session
        self.connect_override = connect
        self.random = random.Random(seed)
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._state = "STOPPED"
        self._connection_attempts = 0
        self._connections = 0
        self._reconnects = 0
        self._messages = 0
        self._pings = 0
        self._dropped_messages = 0
        self._last_event_at: str | None = None
        self._last_connected_at: str | None = None
        self._last_error_code: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._state == "AUTHENTICATED"

    @property
    def running(self) -> bool:
        return bool(self._task is not None and not self._task.done())

    def authentication_message(self, *, timestamp_ms: int | None = None) -> dict[str, Any]:
        """Build the documented authentication frame; never expose it in health."""

        timestamp = timestamp_ms or int(time.time() * 1_000)
        signing_payload = f"{timestamp}GET{BITVAVO_WEBSOCKET_PATH}"
        signature = hmac.new(
            self._api_secret.get_secret_value().encode("utf-8"),
            signing_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "action": "authenticate",
            "key": self._api_key.get_secret_value(),
            "signature": signature,
            "timestamp": timestamp,
            "window": self.access_window_ms,
        }

    def health(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        connected_at = (
            datetime.fromisoformat(
                str(self._last_connected_at).replace("Z", "+00:00")
            ).astimezone(UTC)
            if self._last_connected_at
            else None
        )
        return {
            "provider": "bitvavo",
            "channel": "account",
            "state": self._state,
            "ready_for_new_entries": self.ready,
            "connection_attempts": self._connection_attempts,
            "connections": self._connections,
            "reconnects": self._reconnects,
            "messages": self._messages,
            "pings": self._pings,
            "dropped_messages": self._dropped_messages,
            "last_connected_at": self._last_connected_at,
            "last_connected_age_ms": (
                max(0.0, (now - connected_at).total_seconds() * 1_000)
                if connected_at is not None
                else None
            ),
            "last_event_at": self._last_event_at,
            "last_error_code": self._last_error_code,
            "secrets_serialized": False,
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self.run(),
            name="bitvavo-private-account-stream",
        )

    async def stop(self) -> None:
        self._stop.set()
        self._ready.clear()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._state = "STOPPED"

    async def wait_until_ready(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return self.ready

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                yield await asyncio.wait_for(self.queue.get(), timeout=0.25)
            except TimeoutError:
                continue

    async def run(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            self._connection_attempts += 1
            self._state = "CONNECTING"
            self._ready.clear()
            try:
                await self._connection()
                consecutive_failures = 0
                if not self._stop.is_set():
                    self._reconnects += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_failures += 1
                self._ready.clear()
                self._state = "RECONNECTING"
                self._last_error_code = type(exc).__name__.upper()
                LOGGER.warning(
                    "Bitvavo private account stream reconnect scheduled",
                    extra={
                        "component": "bitvavo_private_stream",
                        "operation": "reconnect",
                        "status": "RETRYING",
                        "reason_code": self._last_error_code,
                        "retry_number": consecutive_failures,
                    },
                )
                delay = min(30.0, 0.5 * 2 ** min(consecutive_failures - 1, 6))
                delay *= 0.75 + self.random.random() * 0.5
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    pass

    async def _connection(self) -> None:
        owned = self.session is None
        session = self.session or aiohttp.ClientSession()
        connector = self.connect_override or session.ws_connect
        try:
            async with connector(
                BITVAVO_WEBSOCKET_URL,
                heartbeat=self.heartbeat_seconds,
            ) as socket:
                await socket.send_json(self.authentication_message())
                authenticated = False
                subscribed = False
                while not self._stop.is_set():
                    try:
                        message = await asyncio.wait_for(
                            socket.receive(),
                            timeout=self.inactivity_timeout_seconds,
                        )
                    except TimeoutError:
                        self._pings += 1
                        await socket.ping()
                        continue
                    if message.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise ConnectionError("private account stream closed")
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = json.loads(message.data)
                    if not isinstance(payload, Mapping):
                        continue
                    event = str(payload.get("event") or "").casefold()
                    if event in {"authenticate", "authenticated"}:
                        if payload.get("authenticated") is not True:
                            raise PermissionError("Bitvavo WebSocket authentication failed")
                        authenticated = True
                        self._connections += 1
                        self._last_connected_at = utc_iso()
                        await socket.send_json(
                            {
                                "action": "subscribe",
                                "channels": [
                                    {
                                        "name": "account",
                                        "markets": list(self.markets),
                                    }
                                ],
                            }
                        )
                        continue
                    if event == "subscribed":
                        channels = payload.get("channels")
                        text = stable_json(channels).casefold()
                        if authenticated and ("account" in text or channels is None):
                            subscribed = True
                            self._state = "AUTHENTICATED"
                            self._last_error_code = None
                            self._ready.set()
                        continue
                    if event in {"error", "failed"} or payload.get("errorCode"):
                        raise ConnectionError("Bitvavo private channel returned an error")
                    normalized = sanitize_account_event(payload)
                    if normalized is None:
                        continue
                    if not (authenticated and subscribed):
                        raise PermissionError("account event arrived before subscription")
                    self._messages += 1
                    self._last_event_at = utc_iso()
                    if self.queue.full():
                        self.queue.get_nowait()
                        self._dropped_messages += 1
                    self.queue.put_nowait(normalized)
        finally:
            self._ready.clear()
            if owned:
                await session.close()


__all__ = [
    "BITVAVO_WEBSOCKET_PATH",
    "BITVAVO_WEBSOCKET_URL",
    "BitvavoPrivateAccountStream",
    "sanitize_account_event",
]
