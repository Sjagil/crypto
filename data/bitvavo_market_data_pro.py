

"""Authenticated Bitvavo WS Market Data Pro L1/L2 manager.

This module is market-data only.  It never submits orders and never invents
Level 3.  Bitvavo Market Data Pro exposes aggregated price-level L2 updates,
not market-by-order identities.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import random
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import aiohttp
from pydantic import SecretStr

from core.contracts import NormalizedStreamEvent, StreamEventType, normalize_market
from data.orderbook_l2 import Level2OrderBook, SequenceGap
from utils.common import sha256_text, stable_json, utc_now

BITVAVO_MDPRO_URL = "wss://ws-mdpro.bitvavo.com/v2/"
BITVAVO_WEBSOCKET_PATH = "/v2/websocket"
L3_STATUS = "L3_UNSUPPORTED_BY_EXECUTION_VENUE"


def _event_time(value: Any, observed_at: datetime) -> datetime:
    if value in (None, ""):
        return observed_at
    try:
        number = int(value)
    except (TypeError, ValueError):
        return observed_at
    magnitude = abs(number)
    if magnitude >= 100_000_000_000_000_000:
        seconds = number / 1_000_000_000
    elif magnitude >= 100_000_000_000_000:
        seconds = number / 1_000_000
    elif magnitude >= 100_000_000_000:
        seconds = number / 1_000
    else:
        seconds = number
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return observed_at


class BitvavoMarketDataProManager:
    """Non-conflated L2 manager with strict range-sequence synchronization."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        api_secret: SecretStr,
        markets: Iterable[str],
        depth: int = 1000,
        queue_size: int = 20_000,
        heartbeat_seconds: float = 20.0,
        inactivity_timeout_seconds: float = 60.0,
        access_window_ms: int = 10_000,
        session: aiohttp.ClientSession | None = None,
        connect: Callable[..., Any] | None = None,
        seed: int = 4401,
    ) -> None:
        selected = tuple(
            dict.fromkeys(
                normalize_market(str(market))
                for market in markets
                if str(market).strip()
            )
        )
        if not selected or len(selected) > 25:
            raise ValueError("Market Data Pro requires 1..25 tracked markets")
        if depth < 1 or depth > 1000:
            raise ValueError("Market Data Pro book depth must be 1..1000")
        self._api_key = api_key
        self._api_secret = api_secret
        self.markets = selected
        self.depth = int(depth)
        self.queue: asyncio.Queue[NormalizedStreamEvent] = asyncio.Queue(
            maxsize=max(100, int(queue_size))
        )
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.inactivity_timeout_seconds = float(inactivity_timeout_seconds)
        self.access_window_ms = int(access_window_ms)
        self.session = session
        self.connect_override = connect
        self.random = random.Random(seed)
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._socket: Any | None = None
        self._books: dict[str, Level2OrderBook] = {}
        self._buffers: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=50_000)
        )
        self._seed_callbacks: list[Callable[[Any], None]] = []
        self._request_id = 1000
        self._request_market: dict[int, str] = {}
        self._latest_ticker: dict[str, dict[str, Any]] = {}
        self._latest_trade: dict[str, dict[str, Any]] = {}
        self._state = "STOPPED"
        self._authenticated = False
        self._last_message_at: datetime | None = None
        self._last_error: str | None = None
        self._connections = 0
        self._reconnects = 0
        self._messages = 0
        self._book_updates = 0
        self._sequence_gaps = 0
        self._dropped_messages = 0
        self._resyncs = 0
        self._snapshots = 0

    def add_seed_callback(self, callback: Callable[[Any], None]) -> None:
        self._seed_callbacks.append(callback)

    def authentication_message(self, *, timestamp_ms: int | None = None) -> dict[str, Any]:
        timestamp = timestamp_ms or int(time.time() * 1_000)
        payload = f"{timestamp}GET{BITVAVO_WEBSOCKET_PATH}"
        signature = hmac.new(
            self._api_secret.get_secret_value().encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "action": "authenticate",
            "key": self._api_key.get_secret_value(),
            "signature": signature,
            "timestamp": timestamp,
            "window": self.access_window_ms,
        }

    @property
    def ready(self) -> bool:
        return self._state == "CONNECTED" and self._authenticated

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run(), name="bitvavo-mdpro")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._state = "STOPPED"
        self._authenticated = False

    async def next_event(self, timeout: float | None = None) -> NormalizedStreamEvent:
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout=timeout)

    async def _publish(self, event: NormalizedStreamEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            self._dropped_messages += 1
            raise RuntimeError("MDPRO_QUEUE_OVERFLOW_FAIL_CLOSED") from exc

    def health(self, provider: str | None = None) -> dict[str, Any]:
        now = utc_now()
        age = (
            max(0.0, (now - self._last_message_at).total_seconds())
            if self._last_message_at is not None
            else None
        )
        row = {
            "provider": "bitvavo",
            "feed": "WS_MARKET_DATA_PRO",
            "state": self._state,
            "authenticated": self._authenticated,
            "ready": self.ready,
            "markets": list(self.markets),
            "market_count": len(self.markets),
            "depth": self.depth,
            "messages": self._messages,
            "book_updates": self._book_updates,
            "snapshots": self._snapshots,
            "connections": self._connections,
            "reconnects": self._reconnects,
            "sequence_gaps": self._sequence_gaps,
            "dropped_messages": self._dropped_messages,
            "resyncs": self._resyncs,
            "queue_size": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
            "last_message_at": (
                self._last_message_at.isoformat()
                if self._last_message_at is not None
                else None
            ),
            "last_message_age_seconds": age,
            "last_error": self._last_error,
            "l1_status": "AVAILABLE_NATIVE",
            "l2_status": "AVAILABLE_NATIVE_NON_CONFLATED"
            if self.ready
            else "NOT_READY",
            "l3_status": L3_STATUS,
            "individual_order_ids_available": False,
            "synthetic_l3_used": False,
            "orders_generated": 0,
            "orders_submitted": 0,
            "sensitive_values_serialized": False,
        }
        return row if provider else {"bitvavo": row}

    def snapshot(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for market in self.markets:
            book = self._books.get(market)
            if book is None:
                rows[market] = {
                    "status": "WAITING_FOR_SNAPSHOT",
                    "l3_status": L3_STATUS,
                }
                continue
            bid = book.best_bid
            ask = book.best_ask
            mid = book.mid_price
            rows[market] = {
                "status": (
                    "READY"
                    if book.valid and not book.is_stale()
                    else "STALE_OR_INVALID"
                ),
                "sequence": book.sequence,
                "best_bid": float(bid) if bid is not None else None,
                "best_ask": float(ask) if ask is not None else None,
                "mid_price": float(mid) if mid is not None else None,
                "spread_bps": (
                    float(book.spread_bps)
                    if book.spread_bps is not None
                    else None
                ),
                "microprice": (
                    float(book.microprice)
                    if book.microprice is not None
                    else None
                ),
                "top_level_imbalance": (
                    float(book.top_level_imbalance)
                    if book.top_level_imbalance is not None
                    else None
                ),
                "book_pressure": float(book.book_pressure),
                "bid_depth_base": float(book.cumulative_bid_depth),
                "ask_depth_base": float(book.cumulative_ask_depth),
                "depth_imbalance_5bps": self._depth(book, 5),
                "depth_imbalance_10bps": self._depth(book, 10),
                "depth_imbalance_25bps": self._depth(book, 25),
                "depth_imbalance_50bps": self._depth(book, 50),
                "depth_imbalance_100bps": self._depth(book, 100),
                "bid_levels": len(book.bids),
                "ask_levels": len(book.asks),
                "ticker": self._latest_ticker.get(market),
                "last_trade": self._latest_trade.get(market),
                "book_health": book.health(),
                "l3_status": L3_STATUS,
                "individual_order_ids_available": False,
            }
        return {
            "schema_version": "bitvavo_mdpro_snapshot_v1",
            "generated_at": utc_now().isoformat(),
            "feed": "BITVAVO_WS_MARKET_DATA_PRO",
            "market_count": len(rows),
            "markets": rows,
            "health": self.health("bitvavo"),
            "l3_status": L3_STATUS,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    @staticmethod
    def _depth(book: Level2OrderBook, bps: int) -> float | None:
        value = book.depth_imbalance(Decimal(str(bps)))
        return float(value) if value is not None else None

    async def update_markets(self, markets: Iterable[str]) -> None:
        selected = tuple(
            dict.fromkeys(
                normalize_market(str(market))
                for market in markets
                if str(market).strip()
            )
        )[:25]
        if not selected or selected == self.markets:
            return
        removed = sorted(set(self.markets) - set(selected))
        added = sorted(set(selected) - set(self.markets))
        self.markets = selected
        for market in removed:
            self._books.pop(market, None)
            self._buffers.pop(market, None)
        socket = self._socket
        if socket is not None and self.ready:
            if removed:
                await socket.send_json(
                    {
                        "action": "unsubscribe",
                        "channels": [
                            {"name": name, "markets": removed}
                            for name in ("book", "ticker", "trades")
                        ],
                    }
                )
            if added:
                await socket.send_json(
                    {
                        "action": "subscribe",
                        "channels": [
                            {"name": name, "markets": added}
                            for name in ("book", "ticker", "trades")
                        ],
                    }
                )
                for market in added:
                    await self._request_snapshot(market)

    async def run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                await self._connection()
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                self._last_error = f"{type(exc).__name__}:{str(exc)[:300]}"
                self._state = "RECONNECTING"
                self._authenticated = False
                self._reconnects += 1
                await asyncio.sleep(
                    min(30.0, 0.5 * (2 ** min(failures, 6)))
                    * (0.8 + 0.4 * self.random.random())
                )

    async def _connection(self) -> None:
        owned = self.session is None
        session = self.session or aiohttp.ClientSession()
        try:
            connector = self.connect_override or session.ws_connect
            async with connector(
                BITVAVO_MDPRO_URL,
                heartbeat=self.heartbeat_seconds,
                receive_timeout=self.inactivity_timeout_seconds,
            ) as socket:
                self._socket = socket
                self._state = "AUTHENTICATING"
                self._authenticated = False
                self._books.clear()
                self._buffers.clear()
                await socket.send_json(self.authentication_message())
                await self._await_auth(socket)
                self._authenticated = True
                self._state = "CONNECTED"
                self._connections += 1
                await socket.send_json(
                    {
                        "action": "subscribe",
                        "channels": [
                            {"name": name, "markets": list(self.markets)}
                            for name in ("book", "ticker", "trades")
                        ],
                    }
                )
                for market in self.markets:
                    await self._request_snapshot(market)
                async for message in socket:
                    if self._stop.is_set():
                        break
                    if message.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(message.data)
                        self._messages += 1
                        self._last_message_at = utc_now()
                        await self._handle_message(payload)
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSING,
                    }:
                        raise RuntimeError("MDPRO_SOCKET_CLOSED")
        finally:
            self._socket = None
            if owned:
                await session.close()

    @staticmethod
    def _authentication_confirmed(payload: Mapping[str, Any]) -> bool:
        """Accept Bitvavo authentication success shapes without accepting ambiguity."""

        response = payload.get("response")
        response_map = response if isinstance(response, Mapping) else {}
        event = str(payload.get("event") or payload.get("action") or "").lower()
        response_event = str(
            response_map.get("event") or response_map.get("action") or ""
        ).lower()
        explicit = (
            payload.get("authenticated") is True
            or response_map.get("authenticated") is True
        )
        named_success = event in {"authenticate", "authenticated"} or response_event in {
            "authenticate",
            "authenticated",
        }
        return bool(explicit and named_success)

    async def _await_auth(self, socket: Any) -> None:
        for _ in range(20):
            message = await asyncio.wait_for(socket.receive(), timeout=10)
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            payload = json.loads(message.data)
            response = payload.get("response")
            response_map = response if isinstance(response, Mapping) else {}
            error = (
                payload.get("error")
                or payload.get("errorCode")
                or response_map.get("error")
                or response_map.get("errorCode")
            )
            if error:
                raise PermissionError(f"MDPRO_AUTH_ERROR:{error}")
            if self._authentication_confirmed(payload):
                return
        raise PermissionError("MDPRO_AUTHENTICATION_NOT_CONFIRMED")

    async def _request_snapshot(self, market: str) -> None:
        socket = self._socket
        if socket is None:
            return
        self._request_id += 1
        request_id = self._request_id
        self._request_market[request_id] = market
        await socket.send_json(
            {
                "action": "getBook",
                "requestId": request_id,
                "market": market,
                "depth": self.depth,
            }
        )

    async def _request_resync(self, market: str) -> None:
        self._resyncs += 1
        self._books.pop(market, None)
        self._buffers[market].clear()
        socket = self._socket
        if socket is None:
            return
        await socket.send_json(
            {
                "action": "unsubscribe",
                "channels": [{"name": "book", "markets": [market]}],
            }
        )
        await socket.send_json(
            {
                "action": "subscribe",
                "channels": [{"name": "book", "markets": [market]}],
            }
        )
        await self._request_snapshot(market)

    async def _handle_message(self, payload: Mapping[str, Any]) -> None:
        if payload.get("error") or payload.get("errorCode"):
            raise RuntimeError(
                f"MDPRO_ERROR:{payload.get('errorCode') or payload.get('error')}"
            )
        if payload.get("action") == "getBook" and isinstance(
            payload.get("response"), Mapping
        ):
            await self._handle_snapshot(dict(payload["response"]))
            return
        event = str(payload.get("event") or "").lower()
        if event == "book" and payload.get("market"):
            await self._handle_book(dict(payload))
        elif event == "ticker" and payload.get("market"):
            await self._handle_ticker(dict(payload))
        elif event in {"trade", "trades"} and payload.get("market"):
            await self._handle_trade(dict(payload))

    async def _handle_snapshot(self, response: dict[str, Any]) -> None:
        market = normalize_market(str(response["market"]))
        sequence = int(response["mdSeqNo"])
        book = Level2OrderBook(
            provider="bitvavo_mdpro",
            market=market,
            maximum_depth=self.depth,
        )
        observed = utc_now()
        await book.initialize(
            bids=response.get("bids") or (),
            asks=response.get("asks") or (),
            sequence=sequence,
            timestamp=observed,
        )
        self._books[market] = book
        self._snapshots += 1
        seed = SimpleNamespace(
            canonical_market=market,
            raw_hash=sha256_text(stable_json(response)),
            values={
                "bids": response.get("bids") or [],
                "asks": response.get("asks") or [],
                "sequence": sequence,
            },
        )
        for callback in self._seed_callbacks:
            callback(seed)
        buffered = sorted(
            list(self._buffers.get(market, ())),
            key=lambda row: int(row.get("startMdSeqNo") or -1),
        )
        self._buffers[market].clear()
        for update in buffered:
            end = int(update.get("endMdSeqNo") or -1)
            start = int(update.get("startMdSeqNo") or -1)
            if end <= sequence:
                continue
            if start <= sequence < end:
                await self._request_resync(market)
                return
            if start != sequence + 1:
                self._sequence_gaps += 1
                await self._request_resync(market)
                return
            applied = await self._apply_book_update(update)
            if not applied:
                return
            sequence = int(update["endMdSeqNo"])

    async def _handle_book(self, payload: dict[str, Any]) -> None:
        market = normalize_market(str(payload["market"]))
        if market not in self._books:
            self._buffers[market].append(payload)
            return
        await self._apply_book_update(payload)

    async def _apply_book_update(self, payload: dict[str, Any]) -> bool:
        market = normalize_market(str(payload["market"]))
        book = self._books.get(market)
        if book is None:
            self._buffers[market].append(payload)
            return False
        start = int(payload["startMdSeqNo"])
        end = int(payload["endMdSeqNo"])
        current = book.sequence
        if current is not None and end <= current:
            return True
        if current is None or start != current + 1:
            self._sequence_gaps += 1
            await self._request_resync(market)
            return False
        try:
            await book.apply_delta_range(
                bids=payload.get("bids") or (),
                asks=payload.get("asks") or (),
                start_sequence=start,
                end_sequence=end,
                timestamp=utc_now(),
            )
        except SequenceGap:
            self._sequence_gaps += 1
            await self._request_resync(market)
            return False
        self._book_updates += 1
        observed = utc_now()
        event_time = _event_time(payload.get("timestamp"), observed)
        await self._publish(
            NormalizedStreamEvent(
                event_type=StreamEventType.ORDERBOOK_DELTA,
                provider="bitvavo",
                source_symbol=market,
                canonical_market=market,
                timestamp=event_time,
                observed_at=observed,
                sequence=end,
                message_id=f"mdpro-book-{market}-{start}-{end}",
                payload={
                    "bids": payload.get("bids") or [],
                    "asks": payload.get("asks") or [],
                    "from_sequence": start,
                    "to_sequence": end,
                    "mdpro": True,
                    "source_kind": "BITVAVO_MARKET_DATA_PRO_L2",
                    "raw_payload_hash": sha256_text(stable_json(payload)),
                    "l3_status": L3_STATUS,
                },
            )
        )
        return True

    async def _handle_ticker(self, payload: dict[str, Any]) -> None:
        market = normalize_market(str(payload["market"]))
        observed = utc_now()
        self._latest_ticker[market] = {
            "observed_at": observed.isoformat(),
            "best_bid": payload.get("bestBid"),
            "best_bid_size": payload.get("bestBidSize"),
            "best_ask": payload.get("bestAsk"),
            "best_ask_size": payload.get("bestAskSize"),
            "last_price": payload.get("lastPrice"),
        }
        await self._publish(
            NormalizedStreamEvent(
                event_type=StreamEventType.TICKER,
                provider="bitvavo",
                source_symbol=market,
                canonical_market=market,
                timestamp=observed,
                observed_at=observed,
                message_id=f"mdpro-ticker-{market}-{time.time_ns()}",
                payload={
                    "best_bid": payload.get("bestBid"),
                    "best_ask": payload.get("bestAsk"),
                    "best_bid_size": payload.get("bestBidSize"),
                    "best_ask_size": payload.get("bestAskSize"),
                    "last_price": payload.get("lastPrice"),
                    "ticker_kind": "MDPRO_L1",
                    "raw_payload_hash": sha256_text(stable_json(payload)),
                },
            )
        )

    async def _handle_trade(self, payload: dict[str, Any]) -> None:
        market = normalize_market(str(payload["market"]))
        observed = utc_now()
        event_time = _event_time(
            payload.get("timestampNs") or payload.get("timestamp"),
            observed,
        )
        price = Decimal(str(payload.get("price") or "0"))
        amount = Decimal(str(payload.get("amount") or "0"))
        quote = price * amount
        trade_id = str(payload.get("id") or time.time_ns())
        self._latest_trade[market] = {
            "observed_at": observed.isoformat(),
            "timestamp": event_time.isoformat(),
            "id": trade_id,
            "price": str(price),
            "amount": str(amount),
            "side": payload.get("side"),
        }
        await self._publish(
            NormalizedStreamEvent(
                event_type=StreamEventType.TRADE,
                provider="bitvavo",
                source_symbol=market,
                canonical_market=market,
                timestamp=event_time,
                observed_at=observed,
                message_id=f"mdpro-trade-{trade_id}",
                payload={
                    "trade_id": trade_id,
                    "price": str(price),
                    "base_quantity": str(amount),
                    "quantity": str(amount),
                    "quote_quantity": str(quote),
                    "aggressor_side": str(payload.get("side") or "").lower(),
                    "side": str(payload.get("side") or "").lower(),
                    "raw_payload_hash": sha256_text(stable_json(payload)),
                },
            )
        )


__all__ = [
    "BITVAVO_MDPRO_URL",
    "BitvavoMarketDataProManager",
    "L3_STATUS",
]
