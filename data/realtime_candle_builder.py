"""Causal closed-candle projection from the existing public trade stream."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.common import atomic_write_json, utc_now


class RealtimeCandleBuilder:
    """Build 1m/5m/15m candles without fabricating empty intervals.

    A bar becomes closed only when a real trade in a later UTC bucket arrives.
    The projection is an execution freshness aid; immutable provider history
    and REST recovery remain the authoritative historical sources.
    """

    _SECONDS = {"1m": 60, "5m": 300, "15m": 900}

    def __init__(
        self,
        *,
        output_path: Path,
        maximum_closed_per_series: int = 256,
        persist_interval_seconds: float = 5.0,
    ) -> None:
        self.output_path = output_path
        self.maximum_closed_per_series = max(16, maximum_closed_per_series)
        self.persist_interval_seconds = max(1.0, float(persist_interval_seconds))
        self._current: dict[tuple[str, str], dict[str, Any]] = {}
        self._closed: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.maximum_closed_per_series)
        )
        self._late_events = 0
        self._trades_ingested = 0
        self._last_persisted_at: datetime | None = None

    @staticmethod
    def _bucket(timestamp: datetime, seconds: int) -> datetime:
        selected = timestamp.astimezone(UTC)
        epoch = int(selected.timestamp())
        return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)

    def ingest_trade(
        self,
        *,
        market: str,
        timestamp: datetime,
        observed_at: datetime,
        price: float,
        base_quantity: float,
        quote_quantity: float,
        aggressor_side: str,
    ) -> None:
        if price <= 0 or base_quantity < 0 or quote_quantity < 0:
            return
        self._trades_ingested += 1
        wrote_closed = False
        direction = 1 if aggressor_side.casefold() in {"buy", "bid"} else -1
        for timeframe, seconds in self._SECONDS.items():
            key = (market, timeframe)
            bucket = self._bucket(timestamp, seconds)
            current = self._current.get(key)
            if current is not None and bucket < current["timestamp"]:
                self._late_events += 1
                continue
            if current is None or bucket > current["timestamp"]:
                if current is not None:
                    closed = {
                        **current,
                        "timestamp": current["timestamp"].isoformat(),
                        "closed": True,
                        "available_at": observed_at.astimezone(UTC).isoformat(),
                    }
                    self._closed[key].append(closed)
                    wrote_closed = True
                current = {
                    "timestamp": bucket,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": base_quantity,
                    "quote_volume": quote_quantity,
                    "buy_quote_volume": quote_quantity if direction > 0 else 0.0,
                    "sell_quote_volume": quote_quantity if direction < 0 else 0.0,
                    "trade_count": 1,
                }
                self._current[key] = current
                continue
            current["high"] = max(float(current["high"]), price)
            current["low"] = min(float(current["low"]), price)
            current["close"] = price
            current["volume"] = float(current["volume"]) + base_quantity
            current["quote_volume"] = float(current["quote_volume"]) + quote_quantity
            if direction > 0:
                current["buy_quote_volume"] = (
                    float(current["buy_quote_volume"]) + quote_quantity
                )
            else:
                current["sell_quote_volume"] = (
                    float(current["sell_quote_volume"]) + quote_quantity
                )
            current["trade_count"] = int(current["trade_count"]) + 1
        persistence_due = bool(
            self._last_persisted_at is None
            or (
                observed_at.astimezone(UTC) - self._last_persisted_at
            ).total_seconds()
            >= self.persist_interval_seconds
        )
        # Persist a bounded open-candle execution projection as a liveness
        # signal as well as every newly closed candle.  Open rows are clearly
        # marked execution-only and can never become strategy truth.  This
        # also makes a dead trade subscription distinguishable from a quiet
        # candle interval without writing on every tick.
        if wrote_closed or persistence_due:
            self.persist(observed_at=observed_at)

    def snapshot(self, *, observed_at: datetime | None = None) -> dict[str, Any]:
        observed = (observed_at or utc_now()).astimezone(UTC)
        closed: dict[str, list[dict[str, Any]]] = {}
        for (market, timeframe), rows in sorted(self._closed.items()):
            closed[f"{market}:{timeframe}"] = list(rows)
        open_rows: dict[str, dict[str, Any]] = {}
        for (market, timeframe), row in sorted(self._current.items()):
            open_rows[f"{market}:{timeframe}"] = {
                **row,
                "timestamp": row["timestamp"].isoformat(),
                "closed": False,
            }
        return {
            "schema_version": "realtime_candle_projection_v1",
            "generated_at": observed.isoformat(),
            "timeframes": list(self._SECONDS),
            "closed_candles": closed,
            "open_candles_execution_only": open_rows,
            "trades_ingested": self._trades_ingested,
            "late_events_ignored": self._late_events,
            "synthetic_candles_created": 0,
            "forward_filled": False,
            "strategy_truth_uses_closed_only": True,
        }

    def persist(self, *, observed_at: datetime | None = None) -> None:
        observed = (observed_at or utc_now()).astimezone(UTC)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.output_path, self.snapshot(observed_at=observed))
        self._last_persisted_at = observed
