"""Tiered Bitvavo EUR spot universe and candle-health read model.

Discovery and research breadth are deliberately wider than execution
authority.  A market appearing in the shadow universe never grants order
authority; the canonical live universe remains the fail-closed source for
live execution.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp
import pandas as pd

from config.settings import Settings
from core.market_exceptions import load_execution_market_exceptions
from data.data_loader import TIMEFRAME_SECONDS
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "dynamic_live_universe_v1"
CANDLE_SCHEMA_VERSION = "multi_timeframe_candle_health_v1"
TIERED_SCHEMA_VERSION = "tiered_trading_universe_v1"
PREFERRED_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
    "NPC-EUR",
    "ADA-EUR",
)
REQUIRED_TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d", "1W")
LIVE_REQUIRED_TIMEFRAMES = ("15m", "1h", "4h", "1d")
RESEARCH_TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d")
RESEARCH_MINIMUM_ROWS = {
    "15m": 1_000,
    "1h": 500,
    "2h": 500,
    "4h": 365,
    "1d": 180,
}
SHADOW_ENTRY_TIMEFRAMES = ("15m", "1h")


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _paths(settings: Settings) -> dict[str, Path]:
    governance = settings.paths.output_dir / "governance"
    data = settings.paths.output_dir / "data"
    governance.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return {
        "universe": governance / "live_universe.json",
        "candles": data / "candle_health.json",
        "tiered": settings.paths.output_dir
        / "universe"
        / "tiered_trading_universe.json",
        "top50": settings.paths.output_dir / "universe" / "top50_current.json",
        "eligibility": settings.paths.output_dir
        / "universe"
        / "top50_eligibility.json",
    }


def _normalized_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns and frame.index.name == "timestamp":
        frame = frame.reset_index()
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if required - set(frame.columns):
        raise ValueError("MISSING_OHLCV_COLUMNS")
    result = frame.loc[:, sorted(required)].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.dropna()
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def candle_health(
    settings: Settings,
    *,
    markets: Sequence[str] = PREFERRED_MARKETS,
    timeframes: Sequence[str] = REQUIRED_TIMEFRAMES,
    now: datetime | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    """Audit local closed-candle chains without touching trading state."""

    current = pd.Timestamp(now or datetime.now(UTC))
    rows: list[dict[str, Any]] = []
    for market in markets:
        for timeframe in timeframes:
            path = (
                settings.paths.processed_data_dir
                / f"{market}_{timeframe}.parquet"
            )
            row: dict[str, Any] = {
                "market": market,
                "timeframe": timeframe,
                "path": str(path),
                "status": "BLOCKED",
                "reason_codes": [],
                "rows": 0,
            }
            if not path.is_file():
                row["reason_codes"].append("DATASET_MISSING")
                rows.append(row)
                continue
            try:
                frame = _normalized_frame(path)
            except (OSError, ValueError) as exc:
                row["reason_codes"].append(
                    f"DATASET_{type(exc).__name__.upper()}"
                )
                rows.append(row)
                continue
            seconds = int(TIMEFRAME_SECONDS[timeframe])
            expected_delta = pd.to_timedelta(seconds, unit="s")
            latest_open = frame.index[-1]
            latest_close = latest_open + expected_delta
            age_seconds = max(
                0.0,
                float((current - latest_close).total_seconds()),
            )
            duplicates = int(frame.index.duplicated().sum())
            non_monotone = not frame.index.is_monotonic_increasing
            invalid = (
                (frame["volume"] < 0)
                | (frame["high"] < frame[["open", "close"]].max(axis=1))
                | (frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
            )
            recent = frame.index[-min(len(frame), 1_000) :]
            differences = recent.to_series().diff().dropna()
            gap_count = int((differences > expected_delta * 1.5).sum())
            alignment_ok = True
            if timeframe == "15m":
                alignment_ok = bool(
                    (frame.index.minute % 15 == 0).all()
                    and (frame.index.second == 0).all()
                )
            elif timeframe == "2h":
                alignment_ok = bool(
                    (frame.index.minute == 0).all()
                    and (frame.index.second == 0).all()
                    and (frame.index.hour % 2 == 0).all()
                )
            elif timeframe in {"1h", "4h"}:
                hours = 4 if timeframe == "4h" else 1
                alignment_ok = bool(
                    (frame.index.minute == 0).all()
                    and (frame.index.second == 0).all()
                    and (frame.index.hour % hours == 0).all()
                )
            elif timeframe == "1d":
                alignment_ok = bool(
                    (frame.index.hour == 0).all()
                    and (frame.index.minute == 0).all()
                )
            elif timeframe == "1W":
                alignment_ok = bool(
                    (frame.index.dayofweek == 0).all()
                    and (frame.index.hour == 0).all()
                )
            if latest_close > current:
                row["reason_codes"].append("INCOMPLETE_LAST_CANDLE")
            if age_seconds > seconds * 2.25:
                row["reason_codes"].append("STALE_CANDLE_CHAIN")
            if duplicates:
                row["reason_codes"].append("DUPLICATE_TIMESTAMPS")
            if non_monotone:
                row["reason_codes"].append("NON_MONOTONE_TIMESTAMPS")
            if bool(invalid.any()):
                row["reason_codes"].append("INVALID_OHLCV")
            if not alignment_ok:
                row["reason_codes"].append("UTC_ALIGNMENT_FAILED")
            # Gaps are reported rather than silently filled.  Exchange outages
            # in older history do not block a current signal when the recent
            # chain and latest closed candle are healthy.
            row.update(
                {
                    "rows": int(len(frame)),
                    "start": frame.index[0].isoformat(),
                    "latest_open": latest_open.isoformat(),
                    "latest_close": latest_close.isoformat(),
                    "age_seconds": age_seconds,
                    "recent_gap_count": gap_count,
                    "duplicate_count": duplicates,
                    "alignment_ok": alignment_ok,
                    "closed_candle_only": latest_close <= current,
                    "status": (
                        "HEALTHY"
                        if not row["reason_codes"]
                        else "BLOCKED"
                    ),
                }
            )
            rows.append(row)
    healthy = sum(row["status"] == "HEALTHY" for row in rows)
    report = {
        "schema_version": CANDLE_SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "markets": list(markets),
        "timeframes": list(timeframes),
        "healthy_series": healthy,
        "total_series": len(rows),
        "all_healthy": healthy == len(rows),
        "rows": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    if write_artifact:
        atomic_write_json(_paths(settings)["candles"], report)
    return report


async def _public_snapshot(
    settings: Settings,
    markets: Sequence[str],
) -> dict[str, dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(
        total=min(20.0, settings.market_data.request_timeout_seconds),
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("https://api.bitvavo.com/v2/markets") as response:
            if response.status >= 400:
                raise RuntimeError(f"BITVAVO_MARKETS_HTTP_{response.status}")
            metadata_raw = await response.json(content_type=None)
        async with session.get(
            "https://api.bitvavo.com/v2/ticker/24h"
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"BITVAVO_TICKER_HTTP_{response.status}")
            tickers_raw = await response.json(content_type=None)
        metadata = {
            str(row.get("market") or "").upper(): dict(row)
            for row in metadata_raw
            if isinstance(row, dict)
        }
        tickers = {
            str(row.get("market") or "").upper(): dict(row)
            for row in tickers_raw
            if isinstance(row, dict)
        }

        async def book(market: str) -> tuple[str, dict[str, Any]]:
            async with session.get(
                f"https://api.bitvavo.com/v2/{market}/book",
                params={"depth": 100},
            ) as response:
                if response.status >= 400:
                    return market, {"status": response.status}
                payload = await response.json(content_type=None)
                return market, dict(payload) if isinstance(payload, dict) else {}

        books = dict(await asyncio.gather(*(book(market) for market in markets)))
    result: dict[str, dict[str, Any]] = {}
    for market in markets:
        ticker = tickers.get(market, {})
        market_metadata = metadata.get(market, {})
        bid = _decimal(ticker.get("bid"))
        ask = _decimal(ticker.get("ask"))
        midpoint = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal("0")
        spread_bps = (
            float((ask - bid) / midpoint * Decimal("10000"))
            if midpoint > 0
            else None
        )
        ask_depth = sum(
            _decimal(level[0]) * _decimal(level[1])
            for level in books.get(market, {}).get("asks") or []
            if isinstance(level, list) and len(level) >= 2
        )
        quote_volume = _decimal(
            ticker.get("volumeQuote")
            or ticker.get("quoteVolume")
            or ticker.get("volumeQuote24h")
        )
        if quote_volume <= 0:
            quote_volume = _decimal(ticker.get("volume")) * midpoint
        result[market] = {
            "venue_available": str(
                market_metadata.get("status") or ""
            ).casefold()
            in {"trading", "active"},
            "venue_status": market_metadata.get("status"),
            "bid": str(bid),
            "ask": str(ask),
            "midpoint": str(midpoint),
            "spread_bps": spread_bps,
            "visible_ask_depth_eur": float(ask_depth),
            "quote_volume_24h_eur": float(quote_volume),
            "minimum_order_quote": market_metadata.get(
                "minOrderInQuoteAsset"
            ),
            "amount_precision": market_metadata.get("amountPrecision"),
            "price_precision": market_metadata.get("pricePrecision"),
        }
    return result


def _top50_eligibility(settings: Settings) -> dict[str, dict[str, Any]]:
    path = _paths(settings)["eligibility"]
    if not path.is_file():
        return {}
    raw = dict(read_json(path))
    return {
        str(row.get("eur_spot_market") or "").upper(): dict(row)
        for row in raw.get("rows") or []
        if row.get("eur_spot_market")
    }


def _top50_rows(settings: Settings) -> list[dict[str, Any]]:
    """Return the freshest point-in-time rows without fabricating metadata."""

    path = _paths(settings)["top50"]
    if path.is_file():
        raw = dict(read_json(path))
        return [dict(row) for row in raw.get("rows") or []]
    return list(_top50_eligibility(settings).values())


def build_tiered_trading_universe(
    settings: Settings,
    *,
    candle_report: Mapping[str, Any] | None = None,
    live_report: Mapping[str, Any] | None = None,
    maximum_shadow_markets: int = 25,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a wide research universe while preserving live fail-closed scope.

    Research eligibility is intentionally independent of operational Shariah
    approval.  ``REVIEW_REQUIRED`` assets may be observed and backtested, but
    only markets already selected by :func:`select_live_universe` are labelled
    ``LIVE_EXECUTABLE``.
    """

    current = now or datetime.now(UTC)
    top50_rows = _top50_rows(settings)
    merged: dict[str, dict[str, Any]] = {}
    eligibility = _top50_eligibility(settings)
    for raw in top50_rows:
        market = str(raw.get("eur_spot_market") or "").upper()
        if not market:
            continue
        merged[market] = {**dict(raw), **dict(eligibility.get(market) or {})}

    exceptions = load_execution_market_exceptions(settings)
    for market, exception in exceptions.items():
        if market in merged or not exception.get("allow_outside_top50"):
            continue
        shariah_status = settings.shariah.eligibility(market).status.value
        merged[market] = {
            "rank": None,
            "symbol": market.split("-", maxsplit=1)[0],
            "eur_spot_market": market,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": "OPERATOR_EXCEPTION_RUNTIME_CHECKS_REQUIRED",
            "execution_reason": "APPROVED_OUTSIDE_TOP50_EXCEPTION",
            "shariah_status": shariah_status,
            "stablecoin": False,
            "wrapped": False,
            "leveraged_token": False,
            "staking_derivative": False,
            "available_at": exception.get("approved_at"),
            "market_cap": None,
            "volume_24h": None,
            "source": "OPERATOR_MARKET_EXCEPTION",
            "outside_top50_exception": True,
        }

    monitor_only_markets = tuple(
        settings.autonomous_live.monitor_only_markets
    )
    for market in monitor_only_markets:
        if market in merged:
            merged[market]["monitor_only"] = True
            continue
        shariah_status = settings.shariah.eligibility(market).status.value
        merged[market] = {
            "rank": None,
            "symbol": market.split("-", maxsplit=1)[0],
            "eur_spot_market": market,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": "MONITOR_ONLY_NOT_EXECUTION_ELIGIBLE",
            "execution_reason": "OPERATOR_MONITOR_ONLY",
            "shariah_status": shariah_status,
            "stablecoin": False,
            "wrapped": False,
            "leveraged_token": False,
            "staking_derivative": False,
            "available_at": current.isoformat(),
            "market_cap": None,
            "volume_24h": None,
            "source": "OPERATOR_MONITOR_ONLY",
            "outside_top50_exception": False,
            "monitor_only": True,
        }

    discovery_markets = sorted(
        merged,
        key=lambda market: (
            int(merged[market].get("rank") or 10_000),
            market,
        ),
    )
    research_markets = [
        market
        for market in discovery_markets
        if merged[market].get("research_eligibility") == "RESEARCH_ELIGIBLE"
        and not bool(merged[market].get("stablecoin"))
        and not bool(merged[market].get("wrapped"))
        and not bool(merged[market].get("leveraged_token"))
        and not bool(merged[market].get("staking_derivative"))
    ]
    inspected_candles = dict(candle_report or {})
    if not inspected_candles:
        inspected_candles = candle_health(
            settings,
            markets=research_markets,
            timeframes=RESEARCH_TIMEFRAMES,
            now=current,
            write_artifact=False,
        )
    candles_by_market: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in inspected_candles.get("rows") or []:
        row = dict(raw)
        candles_by_market.setdefault(str(row.get("market") or ""), {})[
            str(row.get("timeframe") or "")
        ] = row

    eligible_shadow: list[str] = []
    data_reasons: dict[str, list[str]] = {}
    timeframe_gap_reasons: dict[str, list[str]] = {}
    for market in research_markets:
        reasons: list[str] = []
        by_timeframe = candles_by_market.get(market, {})
        for timeframe in RESEARCH_TIMEFRAMES:
            row = by_timeframe.get(timeframe)
            if row is None:
                reasons.append(f"{timeframe.upper()}_DATASET_MISSING")
                continue
            if row.get("status") != "HEALTHY":
                reasons.append(f"{timeframe.upper()}_CANDLE_CHAIN_BLOCKED")
            if int(row.get("rows") or 0) < RESEARCH_MINIMUM_ROWS[timeframe]:
                reasons.append(f"{timeframe.upper()}_INSUFFICIENT_ROWS")
        entry_ready = any(
            by_timeframe.get(timeframe, {}).get("status") == "HEALTHY"
            and int(by_timeframe.get(timeframe, {}).get("rows") or 0)
            >= RESEARCH_MINIMUM_ROWS[timeframe]
            for timeframe in SHADOW_ENTRY_TIMEFRAMES
        )
        blocking_reasons = [] if entry_ready else ["NO_HEALTHY_ENTRY_TIMEFRAME"]
        timeframe_gap_reasons[market] = list(dict.fromkeys(reasons))
        data_reasons[market] = blocking_reasons
        if not blocking_reasons:
            eligible_shadow.append(market)

    live = dict(live_report or live_universe_status(settings))
    live_markets = [
        str(value).upper() for value in live.get("selected_markets") or []
    ]
    # The point-in-time universe is already capped at fifty assets and every
    # shadow market still has to pass the causal candle-health requirements.
    # Keeping an additional hard cap of twenty-five made the active scanner
    # ignore otherwise healthy top-50 EUR markets for no economic reason.
    limit = max(1, min(50, int(maximum_shadow_markets)))
    shadow_markets = eligible_shadow[:limit]
    # Retain every healthy live market in the observed set even when its CMC
    # rank falls below the research cap.  Replace the lowest ranked non-live
    # market instead of silently expanding beyond the configured limit.
    must_observe = tuple(dict.fromkeys((*live_markets, *monitor_only_markets)))
    for market in must_observe:
        if market not in eligible_shadow or market in shadow_markets:
            continue
        if len(shadow_markets) >= limit:
            removable = next(
                (
                    value
                    for value in reversed(shadow_markets)
                    if value not in must_observe
                ),
                None,
            )
            if removable is not None:
                shadow_markets.remove(removable)
        shadow_markets.append(market)
    shadow_markets.sort(
        key=lambda market: (int(merged[market].get("rank") or 10_000), market)
    )

    rows: list[dict[str, Any]] = []
    for market in discovery_markets:
        source = merged[market]
        tiers = ["DISCOVERY"]
        if market in research_markets:
            tiers.append("RESEARCH")
        if market in shadow_markets:
            tiers.append("SHADOW")
        if market in live_markets:
            tiers.append("LIVE_EXECUTABLE")
        by_timeframe = candles_by_market.get(market, {})
        rows.append(
            {
                "market": market,
                "symbol": source.get("symbol") or market.split("-", 1)[0],
                "rank": source.get("rank"),
                "tiers": tiers,
                "highest_tier": tiers[-1],
                "research_eligibility": source.get("research_eligibility"),
                "execution_eligibility": source.get("execution_eligibility"),
                "shariah_status": source.get("shariah_status"),
                "available_at": source.get("available_at"),
                "volume_24h": source.get("volume_24h"),
                "market_cap": source.get("market_cap"),
                "source": source.get("source") or "POINT_IN_TIME_TOP50",
                "outside_top50_exception": bool(
                    source.get("outside_top50_exception")
                ),
                "monitor_only": bool(source.get("monitor_only")),
                "research_candle_series_healthy": sum(
                    by_timeframe.get(timeframe, {}).get("status") == "HEALTHY"
                    and int(by_timeframe.get(timeframe, {}).get("rows") or 0)
                    >= RESEARCH_MINIMUM_ROWS[timeframe]
                    for timeframe in RESEARCH_TIMEFRAMES
                ),
                "research_candle_series_required": len(RESEARCH_TIMEFRAMES),
                "shadow_reason_codes": data_reasons.get(market, []),
                "timeframe_gap_reason_codes": timeframe_gap_reasons.get(
                    market,
                    [],
                ),
                "live_authority_inherited": market in live_markets,
                "live_authority_granted_by_tier_builder": False,
            }
        )

    context_only = [
        str(row.get("symbol") or "")
        for row in top50_rows
        if row.get("research_eligibility") == "CONTEXT_ONLY"
    ]
    payload = {
        "schema_version": TIERED_SCHEMA_VERSION,
        "generated_at": current.isoformat(),
        "top50_snapshot_available": bool(top50_rows),
        "discovery_markets": discovery_markets,
        "research_markets": research_markets,
        "shadow_markets": shadow_markets,
        "paper_markets": [],
        "live_executable_markets": live_markets,
        "context_only_symbols": context_only,
        "counts": {
            "discovery": len(discovery_markets),
            "research": len(research_markets),
            "shadow": len(shadow_markets),
            "paper": 0,
            "live_executable": len(live_markets),
            "context_only": len(context_only),
        },
        "research_timeframes": list(RESEARCH_TIMEFRAMES),
        "shadow_entry_timeframes": list(SHADOW_ENTRY_TIMEFRAMES),
        "shadow_minimum_healthy_entry_timeframes": 1,
        "shadow_required_timeframes": ["15m_OR_1h"],
        "maximum_shadow_markets": limit,
        "execution_authority_unchanged": True,
        "rows": rows,
        "selection_hash": stable_hash(
            {
                "shadow_markets": shadow_markets,
                "live_markets": live_markets,
                "top50": [
                    [row.get("symbol"), row.get("rank"), row.get("available_at")]
                    for row in top50_rows
                ],
            },
            length=64,
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(_paths(settings)["tiered"], payload)
    return payload


def tiered_trading_universe_status(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["tiered"]
    if not path.is_file():
        return {
            "schema_version": TIERED_SCHEMA_VERSION,
            "status": "NOT_REFRESHED",
            "shadow_markets": [],
            "live_executable_markets": [],
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    return dict(read_json(path))


def select_live_universe(
    settings: Settings,
    *,
    market_snapshot: Mapping[str, Mapping[str, Any]],
    candle_report: Mapping[str, Any],
    preferred_markets: Sequence[str] = PREFERRED_MARKETS,
    minimum_markets: int = 5,
) -> dict[str, Any]:
    """Pure, deterministic fail-closed market selector."""

    eligibility = _top50_eligibility(settings)
    exceptions = load_execution_market_exceptions(settings)
    candle_by_market: dict[str, list[dict[str, Any]]] = {}
    for raw in candle_report.get("rows") or []:
        row = dict(raw)
        candle_by_market.setdefault(str(row["market"]), []).append(row)
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for priority, market in enumerate(preferred_markets, start=1):
        public = dict(market_snapshot.get(market) or {})
        universe = dict(eligibility.get(market) or {})
        exception = dict(exceptions.get(market) or {})
        outside_top50_exception = bool(
            exception.get("approved") and exception.get("allow_outside_top50")
        )
        limits = settings.autonomous_live.liquidity_limits(market)
        candles = candle_by_market.get(market, [])
        reasons: list[str] = []
        if not public.get("venue_available"):
            reasons.append("VENUE_UNAVAILABLE")
        if (
            universe.get("execution_eligibility") != "LIVE_ELIGIBLE"
            and not outside_top50_exception
        ):
            reasons.append(
                str(
                    universe.get("execution_reason")
                    or "TOP50_EXECUTION_INELIGIBLE"
                )
            )
        if settings.shariah.eligibility(market).status.value != "ALLOWED":
            reasons.append("SHARIAH_NOT_ALLOWED")
        candle_status = {
            str(row.get("timeframe") or ""): str(row.get("status") or "")
            for row in candles
        }
        if any(
            candle_status.get(timeframe) != "HEALTHY"
            for timeframe in LIVE_REQUIRED_TIMEFRAMES
        ):
            reasons.append("CANDLE_CHAIN_NOT_HEALTHY")
        spread = public.get("spread_bps")
        if spread is None or float(spread) > limits["maximum_spread_bps"]:
            reasons.append("SPREAD_LIMIT")
        if (
            float(public.get("visible_ask_depth_eur") or 0.0)
            < limits["minimum_visible_ask_depth_eur"]
        ):
            reasons.append("ORDERBOOK_DEPTH_LIMIT")
        if (
            float(public.get("quote_volume_24h_eur") or 0.0)
            < limits["minimum_24h_quote_volume_eur"]
        ):
            reasons.append("VOLUME_24H_LIMIT")
        eligible = not reasons
        if eligible:
            selected.append(market)
        rows.append(
            {
                "priority": priority,
                "market": market,
                "symbol": market.split("-", maxsplit=1)[0],
                "status": "LIVE_ELIGIBLE" if eligible else "BLOCKED",
                "reason_codes": reasons,
                "rank": universe.get("rank"),
                "shariah_status": settings.shariah.eligibility(market).status.value,
                "execution_eligibility_basis": (
                    "APPROVED_OUTSIDE_TOP50_EXCEPTION"
                    if outside_top50_exception
                    else "POINT_IN_TIME_TOP50"
                ),
                "outside_top50_exception": outside_top50_exception,
                "strategy_dna_authority_granted": False,
                **public,
                "candle_timeframes_healthy": sum(
                    row.get("status") == "HEALTHY" for row in candles
                ),
                "required_candle_timeframes": len(
                    LIVE_REQUIRED_TIMEFRAMES
                ),
                "live_required_timeframes": list(
                    LIVE_REQUIRED_TIMEFRAMES
                ),
                "optional_context_timeframes": [
                    timeframe
                    for timeframe in REQUIRED_TIMEFRAMES
                    if timeframe not in LIVE_REQUIRED_TIMEFRAMES
                ],
            }
        )
    selected = selected[:25]
    status = "READY" if len(selected) >= minimum_markets else "BLOCKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": status,
        "minimum_markets": minimum_markets,
        "selected_markets": selected,
        "live_eligible_count": len(selected),
        "maximum_concurrent_positions": min(
            2,
            settings.operational.maximum_positions,
        ),
        "selection_hash": stable_hash(
            {
                "markets": selected,
                "rows": rows,
                "required_timeframes": REQUIRED_TIMEFRAMES,
                "live_required_timeframes": LIVE_REQUIRED_TIMEFRAMES,
            },
            length=64,
        ),
        "rows": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _dynamic_preferred_markets(settings: Settings) -> tuple[str, ...]:
    """Resolve every operator-reviewed top-50 EUR spot market by rank.

    Token-type, candle, venue and liquidity checks remain downstream and
    fail closed.  This only removes the old six-market discovery bottleneck.
    """

    rows = _top50_rows(settings)
    ranked = sorted(
        (
            dict(row)
            for row in rows
            if row.get("eur_spot_market")
            and not bool(row.get("stablecoin"))
            and not bool(row.get("wrapped"))
            and not bool(row.get("leveraged_token"))
            and not bool(row.get("staking_derivative"))
        ),
        key=lambda row: (
            int(row.get("rank") or 10_000),
            str(row.get("eur_spot_market") or ""),
        ),
    )
    ordered: list[str] = list(PREFERRED_MARKETS)
    for row in ranked:
        market = str(row["eur_spot_market"]).upper()
        if (
            settings.shariah.eligibility(market).status.value == "ALLOWED"
            and market not in ordered
        ):
            ordered.append(market)
    for market in load_execution_market_exceptions(settings):
        if market not in ordered:
            ordered.append(market)
    return tuple(ordered[:50])


async def refresh_live_universe(
    settings: Settings,
    *,
    preferred_markets: Sequence[str] | None = None,
    minimum_markets: int = 5,
) -> dict[str, Any]:
    resolved_markets = tuple(
        preferred_markets
        if preferred_markets is not None
        else _dynamic_preferred_markets(settings)
    )
    candles = candle_health(
        settings,
        markets=resolved_markets,
        timeframes=REQUIRED_TIMEFRAMES,
    )
    snapshot = await _public_snapshot(settings, resolved_markets)
    report = select_live_universe(
        settings,
        market_snapshot=snapshot,
        candle_report=candles,
        preferred_markets=resolved_markets,
        minimum_markets=minimum_markets,
    )
    atomic_write_json(_paths(settings)["universe"], report)
    if report.get("selected_markets"):
        candle_health(
            settings,
            markets=tuple(report["selected_markets"]),
            timeframes=REQUIRED_TIMEFRAMES,
        )
    return report


def live_universe_status(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["universe"]
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "NOT_REFRESHED",
            "selected_markets": [],
            "live_eligible_count": 0,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    return dict(read_json(path))


__all__ = [
    "PREFERRED_MARKETS",
    "LIVE_REQUIRED_TIMEFRAMES",
    "REQUIRED_TIMEFRAMES",
    "RESEARCH_TIMEFRAMES",
    "build_tiered_trading_universe",
    "candle_health",
    "_dynamic_preferred_markets",
    "live_universe_status",
    "refresh_live_universe",
    "select_live_universe",
    "tiered_trading_universe_status",
]
