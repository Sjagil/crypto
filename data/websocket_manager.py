"""Concurrent normalized public WebSocket streams for crypto spot data."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import aiohttp

from core.contracts import (
    NormalizedStreamEvent,
    StreamEventType,
    normalize_market,
)
from utils.common import sha256_text, stable_json, utc_now

BITVAVO_WS = "wss://ws.bitvavo.com/v2/"
KRAKEN_WS = "wss://ws.kraken.com/v2"
MEXC_WS = "wss://wbs-api.mexc.com/ws"
LOGGER = logging.getLogger("crypto.websocket")


def _protobuf_fields(payload: bytes) -> dict[int, list[int | bytes]]:
    """Decode the protobuf wire types used by MEXC's documented public schema."""
    position = 0
    fields: dict[int, list[int | bytes]] = {}

    def varint() -> int:
        nonlocal position
        result = 0
        shift = 0
        while position < len(payload):
            byte = payload[position]
            position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 63:
                raise ValueError("protobuf varint is too long")
        raise ValueError("truncated protobuf varint")

    while position < len(payload):
        key = varint()
        number, wire_type = key >> 3, key & 0x07
        if wire_type == 0:
            value: int | bytes = varint()
        elif wire_type == 2:
            length = varint()
            end = position + length
            if end > len(payload):
                raise ValueError("truncated protobuf field")
            value = payload[position:end]
            position = end
        elif wire_type == 1:
            value = payload[position : position + 8]
            position += 8
        elif wire_type == 5:
            value = payload[position : position + 4]
            position += 4
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        fields.setdefault(number, []).append(value)
    return fields


def _protobuf_text(value: int | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return value.decode("utf-8")


def decode_mexc_protobuf(payload: bytes) -> dict[str, Any]:
    """Decode current MEXC public ticker, deal and aggregated-depth pushes."""
    wrapper = _protobuf_fields(payload)
    channel = _protobuf_text(wrapper.get(1, [b""])[0])
    symbol = _protobuf_text(wrapper.get(3, [b""])[0])
    send_time = int(wrapper.get(6, [0])[0])
    result: dict[str, Any] = {
        "channel": channel,
        "symbol": symbol,
        "sendTime": send_time,
    }
    if 315 in wrapper:
        fields = _protobuf_fields(bytes(wrapper[315][0]))
        result["publicbookticker"] = {
            "bidPrice": _protobuf_text(fields.get(1, [b""])[0]),
            "bidQuantity": _protobuf_text(fields.get(2, [b""])[0]),
            "askPrice": _protobuf_text(fields.get(3, [b""])[0]),
            "askQuantity": _protobuf_text(fields.get(4, [b""])[0]),
            "version": _protobuf_text(fields.get(5, [b""])[0]),
        }
    elif 314 in wrapper:
        fields = _protobuf_fields(bytes(wrapper[314][0]))
        deals = []
        for encoded in fields.get(1, []):
            item = _protobuf_fields(bytes(encoded))
            deals.append(
                {
                    "price": _protobuf_text(item.get(1, [b""])[0]),
                    "quantity": _protobuf_text(item.get(2, [b""])[0]),
                    "tradeType": int(item.get(3, [0])[0]),
                    "time": int(item.get(4, [0])[0]),
                    "tradeId": _protobuf_text(item.get(5, [b""])[0]),
                }
            )
        result["publicdeals"] = {"deals": deals}
    elif 313 in wrapper:
        fields = _protobuf_fields(bytes(wrapper[313][0]))

        def levels(number: int) -> list[list[str]]:
            selected: list[list[str]] = []
            for encoded in fields.get(number, []):
                item = _protobuf_fields(bytes(encoded))
                selected.append(
                    [
                        _protobuf_text(item.get(1, [b""])[0]),
                        _protobuf_text(item.get(2, [b""])[0]),
                    ]
                )
            return selected

        result["publicdepth"] = {
            "asks": levels(1),
            "bids": levels(2),
            "fromVersion": _protobuf_text(fields.get(4, [b""])[0]),
            "toVersion": _protobuf_text(fields.get(5, [b""])[0]),
        }
    else:
        raise ValueError("unsupported MEXC public protobuf body")
    return result


def _timestamp(value: Any, *, milliseconds: bool = False) -> datetime:
    if value is None:
        return utc_now()
    if isinstance(value, (int, float)):
        number = float(value) / (1_000 if milliseconds or value > 10_000_000_000 else 1)
        return datetime.fromtimestamp(number, tz=UTC)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass
class StreamHealth:
    provider: str
    state: str = "STOPPED"
    connection_attempts: int = 0
    connections: int = 0
    reconnects: int = 0
    subscriptions: int = 0
    messages: int = 0
    duplicates: int = 0
    dropped_messages: int = 0
    parse_errors: int = 0
    sequence_gaps: int = 0
    pings: int = 0
    last_message_at: datetime | None = None
    last_error: str | None = None
    latencies_ms: list[float] = field(default_factory=list)
    started_at: datetime | None = None

    def snapshot(self, stale_after: timedelta) -> dict[str, Any]:
        now = utc_now()
        uptime = (now - self.started_at).total_seconds() if self.started_at else 0.0
        stale = (
            self.state == "CONNECTED"
            and (
                self.last_message_at is None
                or now - self.last_message_at > stale_after
            )
        )
        return {
            "provider": self.provider,
            "state": "STALE" if stale else self.state,
            "uptime_seconds": uptime,
            "connection_attempts": self.connection_attempts,
            "connections": self.connections,
            "reconnects": self.reconnects,
            "subscriptions": self.subscriptions,
            "messages": self.messages,
            "duplicates": self.duplicates,
            "dropped_messages": self.dropped_messages,
            "parse_errors": self.parse_errors,
            "sequence_gaps": self.sequence_gaps,
            "pings": self.pings,
            "throughput_per_second": self.messages / uptime if uptime else 0.0,
            "mean_latency_ms": (
                sum(self.latencies_ms) / len(self.latencies_ms)
                if self.latencies_ms
                else None
            ),
            "maximum_latency_ms": max(self.latencies_ms) if self.latencies_ms else None,
            "last_message_at": (
                self.last_message_at.isoformat() if self.last_message_at else None
            ),
            "last_error": self.last_error,
        }


class WebSocketManager:
    def __init__(
        self,
        *,
        queue_size: int = 2_000,
        backpressure_policy: Literal["drop_oldest", "drop_newest"] = "drop_oldest",
        maximum_connection_attempts: int = 5,
        inactivity_timeout: float = 45.0,
        heartbeat: float = 20.0,
        seed: int = 42,
        session: aiohttp.ClientSession | None = None,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if queue_size < 1 or maximum_connection_attempts < 1:
            raise ValueError("queue size and attempts must be positive")
        self.queue: asyncio.Queue[NormalizedStreamEvent] = asyncio.Queue(queue_size)
        self.backpressure_policy = backpressure_policy
        self.maximum_connection_attempts = maximum_connection_attempts
        self.inactivity_timeout = inactivity_timeout
        self.heartbeat = heartbeat
        self.random = random.Random(seed)
        self.session = session
        self.connect_override = connect
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop = asyncio.Event()
        self._message_ids: dict[str, set[str]] = {}
        self._sequence: dict[tuple[str, str, str], int] = {}
        self.health_state = {
            provider: StreamHealth(provider)
            for provider in ("bitvavo", "kraken", "mexc")
        }

    async def start(
        self,
        subscriptions: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> None:
        self._stop.clear()
        for provider, channels in subscriptions.items():
            if provider not in self.health_state:
                raise ValueError(f"unsupported WebSocket provider: {provider}")
            if provider in self._tasks and not self._tasks[provider].done():
                continue
            self._tasks[provider] = asyncio.create_task(
                self.run_provider(provider, channels),
                name=f"websocket-{provider}",
            )

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        for health in self.health_state.values():
            if health.state != "FAILED":
                health.state = "STOPPED"

    async def __aenter__(self) -> "WebSocketManager":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    async def events(self) -> AsyncIterator[NormalizedStreamEvent]:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                yield await asyncio.wait_for(self.queue.get(), timeout=0.25)
            except TimeoutError:
                continue

    async def next_event(self, timeout: float | None = None) -> NormalizedStreamEvent:
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout=timeout)

    async def run_provider(
        self, provider: str, channels: Mapping[str, Sequence[str]]
    ) -> None:
        health = self.health_state[provider]
        health.started_at = health.started_at or utc_now()
        attempts = 0
        while not self._stop.is_set() and attempts < self.maximum_connection_attempts:
            attempts += 1
            health.connection_attempts += 1
            health.state = "CONNECTING"
            try:
                await self._connection(provider, channels)
                if self._stop.is_set():
                    break
                health.reconnects += 1
                attempts = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                health.last_error = f"{type(exc).__name__}: {exc}"
                health.state = "RECONNECTING"
                LOGGER.exception(
                    "public WebSocket reconnect scheduled",
                    extra={
                        "component": "websocket",
                        "provider": provider,
                        "operation": "reconnect",
                        "status": "RETRYING",
                        "reason_code": type(exc).__name__,
                        "exception_type": type(exc).__name__,
                        "retry_number": attempts,
                    },
                )
                if attempts >= self.maximum_connection_attempts:
                    health.state = "FAILED"
                    break
                delay = min(30.0, 0.5 * 2 ** (attempts - 1))
                delay *= 0.75 + self.random.random() * 0.5
                await asyncio.sleep(delay)

    async def _connection(
        self, provider: str, channels: Mapping[str, Sequence[str]]
    ) -> None:
        health = self.health_state[provider]
        owned = self.session is None
        session = self.session or aiohttp.ClientSession()
        endpoint = {
            "bitvavo": BITVAVO_WS,
            "kraken": KRAKEN_WS,
            "mexc": MEXC_WS,
        }[provider]
        connector = self.connect_override or session.ws_connect
        try:
            async with connector(endpoint, heartbeat=self.heartbeat) as socket:
                health.state = "CONNECTED"
                health.connections += 1
                LOGGER.info(
                    "public WebSocket connected",
                    extra={
                        "component": "websocket",
                        "provider": provider,
                        "operation": "connect",
                        "status": "PASSED",
                        "reason_code": "CONNECTED",
                        "retry_number": health.connection_attempts - 1,
                    },
                )
                await self._emit_status(provider, "CONNECTED")
                for message in self.subscription_messages(provider, channels):
                    await socket.send_json(message)
                    health.subscriptions += 1
                    LOGGER.info(
                        "public WebSocket subscription sent",
                        extra={
                            "component": "websocket",
                            "provider": provider,
                            "operation": "subscribe",
                            "status": "PASSED",
                            "reason_code": "SUBSCRIBED",
                        },
                    )
                while not self._stop.is_set():
                    try:
                        message = await asyncio.wait_for(
                            socket.receive(), timeout=self.inactivity_timeout
                        )
                    except TimeoutError:
                        health.pings += 1
                        await socket.ping()
                        raise TimeoutError("stream inactivity timeout")
                    if message.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise ConnectionError("provider closed WebSocket")
                    if message.type == aiohttp.WSMsgType.BINARY:
                        if provider != "mexc":
                            health.parse_errors += 1
                            raise ValueError("unexpected binary provider message")
                        raw = decode_mexc_protobuf(message.data)
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        raw = json.loads(message.data)
                    else:
                        continue
                    if self._is_control_message(provider, raw):
                        continue
                    for event in self.parse_message(provider, raw):
                        await self._publish(event)
                        health.messages += 1
                        health.last_message_at = utc_now()
                        latency = max(
                            0.0,
                            (event.observed_at - event.timestamp).total_seconds() * 1_000,
                        )
                        health.latencies_ms.append(latency)
                        if len(health.latencies_ms) > 10_000:
                            del health.latencies_ms[:5_000]
        finally:
            if owned:
                await session.close()

    @staticmethod
    def subscription_messages(
        provider: str, channels: Mapping[str, Sequence[str]]
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if provider == "bitvavo":
            messages.append(
                {
                    "action": "subscribe",
                    "channels": [
                        {"name": channel, "markets": list(markets)}
                        for channel, markets in channels.items()
                    ],
                }
            )
        elif provider == "kraken":
            for channel, markets in channels.items():
                messages.append(
                    {
                        "method": "subscribe",
                        "params": {"channel": channel, "symbol": list(markets)},
                    }
                )
        elif provider == "mexc":
            channel_names = {
                "ticker": "spot@public.aggre.bookTicker.v3.api.pb@100ms",
                "trades": "spot@public.aggre.deals.v3.api.pb@100ms",
                "book": "spot@public.aggre.depth.v3.api.pb@100ms",
            }
            params = [
                f"{channel_names[channel]}@{market.replace('-', '').upper()}"
                for channel, markets in channels.items()
                if channel in channel_names
                for market in markets
            ]
            messages.append({"method": "SUBSCRIPTION", "params": params})
        return messages

    @staticmethod
    def _is_control_message(provider: str, payload: Mapping[str, Any]) -> bool:
        if provider == "bitvavo":
            return payload.get("event") in {"subscribed", "pong"}
        if provider == "kraken":
            return payload.get("method") in {"subscribe", "pong"} or payload.get(
                "channel"
            ) in {"status", "heartbeat"}
        return payload.get("msg") in {"PONG", "SUBSCRIPTION"} or "code" in payload

    def parse_message(
        self, provider: str, payload: Mapping[str, Any]
    ) -> list[NormalizedStreamEvent]:
        if provider == "bitvavo":
            return self._parse_bitvavo(payload)
        if provider == "kraken":
            return self._parse_kraken(payload)
        if provider == "mexc":
            return self._parse_mexc(payload)
        raise ValueError("unsupported provider")

    def _event(
        self,
        *,
        provider: str,
        event_type: StreamEventType,
        source_symbol: str,
        timestamp: datetime,
        payload: dict[str, Any],
        sequence: int | None = None,
        message_id: str | None = None,
    ) -> NormalizedStreamEvent:
        canonical = source_symbol.replace("_", "-").replace("/", "-").upper()
        if "-" not in canonical:
            for quote in ("USDT", "EUR", "USD"):
                if canonical.endswith(quote):
                    canonical = f"{canonical[:-len(quote)]}-{quote}"
                    break
        if canonical.startswith("XBT-"):
            canonical = f"BTC-{canonical.split('-', 1)[1]}"
        return NormalizedStreamEvent(
            event_type=event_type,
            provider=provider,
            source_symbol=source_symbol,
            canonical_market=normalize_market(canonical),
            timestamp=timestamp,
            observed_at=utc_now(),
            sequence=sequence,
            message_id=message_id
            or sha256_text(
                stable_json(
                    [
                        provider,
                        event_type.value,
                        source_symbol,
                        timestamp.isoformat(),
                        payload,
                    ]
                )
            ),
            payload=payload,
        )

    def _parse_bitvavo(
        self, raw: Mapping[str, Any]
    ) -> list[NormalizedStreamEvent]:
        event = raw.get("event")
        market = str(raw.get("market", ""))
        timestamp = _timestamp(raw.get("timestamp"), milliseconds=True)
        mapping = {
            "ticker": StreamEventType.TICKER,
            "trade": StreamEventType.TRADE,
            "candle": StreamEventType.CANDLE,
            "book": StreamEventType.ORDERBOOK_DELTA,
        }
        if event not in mapping or not market:
            return []
        if event == "candle":
            return [
                self._event(
                    provider="bitvavo",
                    event_type=StreamEventType.CANDLE,
                    source_symbol=market,
                    timestamp=_timestamp(item[0], milliseconds=True),
                    payload={
                        "interval": raw.get("interval"),
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "volume": item[5],
                    },
                )
                for item in raw.get("candle", [])
            ]
        if event == "ticker":
            payload = {
                "last_price": raw.get("price") or raw.get("last"),
                "best_bid": raw.get("bestBid"),
                "best_ask": raw.get("bestAsk"),
                "volume_24h": raw.get("volume"),
            }
        elif event == "trade":
            payload = {
                "trade_id": raw.get("id"),
                "price": raw.get("price"),
                "quantity": raw.get("amount"),
                "side": raw.get("side"),
            }
        else:
            payload = {
                "bids": raw.get("bids", []),
                "asks": raw.get("asks", []),
                "checksum": raw.get("checksum"),
            }
        return [
            self._event(
                provider="bitvavo",
                event_type=mapping[event],
                source_symbol=market,
                timestamp=timestamp,
                payload=payload,
                sequence=raw.get("nonce"),
                message_id=str(raw.get("id") or "") or None,
            )
        ]

    def _parse_kraken(
        self, raw: Mapping[str, Any]
    ) -> list[NormalizedStreamEvent]:
        channel = str(raw.get("channel", ""))
        mapping = {
            "ticker": StreamEventType.TICKER,
            "trade": StreamEventType.TRADE,
            "ohlc": StreamEventType.CANDLE,
            "book": (
                StreamEventType.ORDERBOOK_SNAPSHOT
                if raw.get("type") == "snapshot"
                else StreamEventType.ORDERBOOK_DELTA
            ),
        }
        if channel not in mapping:
            return []
        events: list[NormalizedStreamEvent] = []
        for item in raw.get("data", []):
            symbol = str(item.get("symbol", ""))
            timestamp = _timestamp(
                item.get("timestamp")
                or item.get("time")
                or item.get("interval_begin")
            )
            if channel == "ticker":
                payload = {
                    "last_price": item.get("last"),
                    "best_bid": item.get("bid"),
                    "best_ask": item.get("ask"),
                    "volume_24h": item.get("volume"),
                }
            elif channel == "trade":
                payload = {
                    "trade_id": item.get("trade_id"),
                    "price": item.get("price"),
                    "quantity": item.get("qty"),
                    "side": item.get("side"),
                }
            elif channel == "ohlc":
                payload = {
                    "interval": item.get("interval"),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume"),
                }
            else:
                payload = {
                    "bids": item.get("bids", []),
                    "asks": item.get("asks", []),
                    "checksum": item.get("checksum"),
                }
            events.append(
                self._event(
                    provider="kraken",
                    event_type=mapping[channel],
                    source_symbol=symbol,
                    timestamp=timestamp,
                    payload=payload,
                    sequence=item.get("sequence"),
                    message_id=str(item.get("trade_id") or "") or None,
                )
            )
        return events

    def _parse_mexc(
        self, raw: Mapping[str, Any]
    ) -> list[NormalizedStreamEvent]:
        channel = str(raw.get("c") or raw.get("channel") or "")
        symbol = str(raw.get("s") or raw.get("symbol") or "")
        data = (
            raw.get("publicdeals")
            or raw.get("publicdepth")
            or raw.get("publicbookticker")
            or raw.get("d")
            or raw.get("data")
            or {}
        )
        mapping = (
            StreamEventType.TRADE
            if "deal" in channel.casefold()
            else StreamEventType.ORDERBOOK_DELTA
            if "depth" in channel.casefold()
            else StreamEventType.TICKER
        )
        if not symbol:
            return []
        items = data.get("deals", []) if isinstance(data, dict) and data.get("deals") else [data]
        if mapping is StreamEventType.ORDERBOOK_DELTA and isinstance(data, dict):
            items = [
                {
                    "bids": data.get("bids", []),
                    "asks": data.get("asks", []),
                    "from_sequence": data.get("fromVersion"),
                    "to_sequence": data.get("toVersion"),
                }
            ]
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                normalized_items.append({"value": item})
            elif mapping is StreamEventType.TICKER:
                normalized_items.append(
                    {
                        "last_price": item.get("lastPrice"),
                        "best_bid": item.get("bidPrice"),
                        "best_ask": item.get("askPrice"),
                        "bid_quantity": item.get("bidQuantity"),
                        "ask_quantity": item.get("askQuantity"),
                    }
                )
            elif mapping is StreamEventType.TRADE:
                trade_type = item.get("tradeType")
                normalized_items.append(
                    {
                        "trade_id": item.get("tradeId"),
                        "price": item.get("price"),
                        "quantity": item.get("quantity"),
                        "side": "buy" if trade_type == 1 else "sell",
                    }
                )
            else:
                normalized_items.append(item)
        items = normalized_items
        return [
            self._event(
                provider="mexc",
                event_type=mapping,
                source_symbol=symbol,
                timestamp=_timestamp(
                    (
                        item.get("t")
                        or item.get("time")
                        or raw.get("t")
                        or raw.get("sendTime")
                    )
                    if isinstance(item, dict)
                    else raw.get("t"),
                    milliseconds=True,
                ),
                payload=item,
                sequence=(
                    int(item.get("to_sequence"))
                    if isinstance(item, dict) and item.get("to_sequence")
                    else int(data.get("version"))
                    if isinstance(data, dict) and data.get("version")
                    else raw.get("r") or raw.get("version")
                ),
                message_id=(
                    str(item.get("trade_id"))
                    if item.get("trade_id")
                    else None
                ),
            )
            for item in items
        ]

    async def _publish(self, event: NormalizedStreamEvent) -> None:
        ids = self._message_ids.setdefault(event.provider, set())
        if event.message_id in ids:
            self.health_state[event.provider].duplicates += 1
            return
        ids.add(event.message_id)
        if len(ids) > 20_000:
            self._message_ids[event.provider] = set(list(ids)[-10_000:])
        if (
            event.sequence is not None
            and event.event_type is StreamEventType.ORDERBOOK_DELTA
        ):
            key = (event.provider, event.canonical_market, event.event_type.value)
            previous = self._sequence.get(key)
            if previous is not None and event.sequence > previous + 1:
                self.health_state[event.provider].sequence_gaps += 1
            if previous is not None and event.sequence <= previous:
                self.health_state[event.provider].duplicates += 1
                return
            self._sequence[key] = event.sequence
        if self.queue.full():
            health = self.health_state[event.provider]
            health.dropped_messages += 1
            LOGGER.warning(
                "WebSocket queue backpressure",
                extra={
                    "component": "websocket",
                    "provider": event.provider,
                    "market": event.canonical_market,
                    "operation": "queue_publish",
                    "status": "PARTIAL",
                    "reason_code": "MESSAGE_DROPPED_BACKPRESSURE",
                },
            )
            if self.backpressure_policy == "drop_newest":
                return
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)

    async def _emit_status(self, provider: str, state: str) -> None:
        event = NormalizedStreamEvent(
            event_type=StreamEventType.CONNECTION_STATUS,
            provider=provider,
            source_symbol="SYSTEM-EUR",
            canonical_market="SYSTEM-EUR",
            timestamp=utc_now(),
            observed_at=utc_now(),
            message_id=str(uuid.uuid4()),
            payload={"state": state},
        )
        await self._publish(event)

    def health(self, provider: str | None = None) -> dict[str, Any]:
        stale_after = timedelta(seconds=self.inactivity_timeout)
        if provider:
            return self.health_state[provider].snapshot(stale_after)
        return {
            name: selected.snapshot(stale_after)
            for name, selected in self.health_state.items()
        }


__all__ = [
    "StreamHealth",
    "WebSocketManager",
    "decode_mexc_protobuf",
]
