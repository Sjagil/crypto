"""Forward-only GEX and spot-orderflow context for tactical decisions.

The module never manufactures historical microstructure.  Missing or stale
prospective observations lower confidence and keep orderflow-dependent entries
closed.  BTC/ETH options context is explicitly a market-regime overlay for
altcoins, not coin-specific gamma exposure.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.common import atomic_write_json, read_json, utc_now


def _finite(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if np.isfinite(selected) else None


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.tz_convert("UTC").to_pydatetime()


def load_gex_context(
    settings: Settings,
    *,
    underlying: str,
    now: datetime | None = None,
    maximum_age: timedelta = timedelta(hours=2),
) -> dict[str, Any]:
    observed = (now or utc_now()).astimezone(UTC)
    path = settings.paths.context_data_dir / f"gex_{underlying.upper()}.parquet"
    if not path.is_file():
        return {
            "underlying": underlying.upper(),
            "status": "DATA_PENDING",
            "reason_code": "GEX_HISTORY_MISSING",
            "fresh": False,
        }
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {
            "underlying": underlying.upper(),
            "status": "DATA_BLOCKED",
            "reason_code": "GEX_HISTORY_UNREADABLE",
            "fresh": False,
        }
    if frame.empty or "available_at" not in frame:
        return {
            "underlying": underlying.upper(),
            "status": "DATA_PENDING",
            "reason_code": "GEX_HISTORY_EMPTY",
            "fresh": False,
        }
    frame = frame.copy()
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="coerce"
    )
    frame = frame.loc[
        frame["available_at"].notna()
        & frame["available_at"].le(pd.Timestamp(observed))
    ].sort_values("available_at")
    if frame.empty:
        return {
            "underlying": underlying.upper(),
            "status": "DATA_PENDING",
            "reason_code": "NO_CAUSALLY_AVAILABLE_GEX",
            "fresh": False,
        }
    row = frame.iloc[-1].to_dict()
    available = pd.Timestamp(row["available_at"]).to_pydatetime()
    age = observed - available
    signed = _finite(
        row.get("convention_signed_gex", row.get("net_gex_proxy"))
    )
    absolute = _finite(row.get("absolute_gex", row.get("gross_gex_proxy")))
    normalized = (
        signed / absolute
        if signed is not None and absolute not in {None, 0.0}
        else None
    )
    regime = (
        "POSITIVE_GEX"
        if normalized is not None and normalized >= 0.05
        else "NEGATIVE_GEX"
        if normalized is not None and normalized <= -0.05
        else "NEUTRAL_GEX"
        if normalized is not None
        else "GEX_DATA_PENDING"
    )
    fresh = age <= maximum_age
    return {
        "underlying": underlying.upper(),
        "status": "READY" if fresh else "STALE",
        "provider": str(row.get("provider") or "deribit"),
        "available_at": available.isoformat(),
        "age_seconds": max(0.0, age.total_seconds()),
        "fresh": fresh,
        "regime": regime,
        "absolute_gex": absolute,
        "convention_signed_gex": signed,
        "normalized_signed_gex": normalized,
        "flow_adjusted_gex": _finite(row.get("flow_adjusted_gex")),
        "flow_adjusted_status": row.get("flow_adjusted_status"),
        "gamma_flip": _finite(row.get("gamma_flip_proxy")),
        "call_wall": _finite(row.get("call_wall")),
        "put_wall": _finite(row.get("put_wall")),
        "max_gamma_strike": _finite(
            row.get("max_gamma_strike", row.get("dominant_gamma_strike"))
        ),
        "gex_concentration_within_2pct": _finite(
            row.get("gex_concentration_within_2pct")
        ),
        "absolute_gex_change_1h": _finite(row.get("absolute_gex_change_1h")),
        "absolute_gex_change_4h": _finite(row.get("absolute_gex_change_4h")),
        "absolute_gex_change_24h": _finite(row.get("absolute_gex_change_24h")),
        "signed_gex_change_1h": _finite(row.get("signed_gex_change_1h")),
        "signed_gex_change_4h": _finite(row.get("signed_gex_change_4h")),
        "signed_gex_change_24h": _finite(row.get("signed_gex_change_24h")),
        "dealer_positioning_known": False,
        "point_in_time_status": str(
            row.get("point_in_time_status") or "FORWARD_ONLY"
        ),
    }


def _snapshot_rows(
    directory: Path,
    *,
    now: datetime,
    maximum_hours: int = 24,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff = now - timedelta(hours=maximum_hours)
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            payload = dict(read_json(path))
        except (OSError, TypeError, ValueError):
            continue
        hour_end = _timestamp(payload.get("hour_end"))
        if hour_end is None or hour_end > now:
            continue
        if hour_end < cutoff:
            break
        for raw in payload.get("markets") or []:
            row = dict(raw)
            row["hour_end"] = hour_end
            row["snapshot_status"] = payload.get("status")
            rows.append(row)
    return rows


def load_orderflow_context(
    settings: Settings,
    *,
    markets: Sequence[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    observed = (now or utc_now()).astimezone(UTC)
    directory = settings.paths.context_data_dir / "microstructure_hourly"
    raw = _snapshot_rows(directory, now=observed) if directory.is_dir() else []
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        market = str(row.get("market") or "").upper()
        if market in markets:
            by_market.setdefault(market, []).append(row)
    output: dict[str, Any] = {}
    for market in markets:
        rows = sorted(by_market.get(market, []), key=lambda item: item["hour_end"])
        if not rows:
            output[market] = {
                "status": "DATA_PENDING",
                "reason_codes": ["NO_PROSPECTIVE_ORDERFLOW"],
                "fresh": False,
            }
            continue
        latest = rows[-1]
        age = observed - latest["hour_end"]
        fresh = age <= timedelta(hours=2, minutes=15)
        latest_complete = latest.get("status") == "COMPLETE"
        horizons: dict[str, Any] = {}
        for label, hours in (("1h", 1), ("2h", 2), ("4h", 4), ("24h", 24)):
            selected = rows[-hours:]
            trade_delta = sum(
                _finite(item.get("trade_delta_base")) or 0.0 for item in selected
            )
            spot_volume = sum(
                _finite(item.get("spot_base_volume")) or 0.0 for item in selected
            )
            ofi_values = [
                value
                for item in selected
                if (value := _finite(item.get("order_flow_imbalance_normalized")))
                is not None
            ]
            horizons[label] = {
                "trade_delta_base": trade_delta,
                "trade_delta_percentage": (
                    trade_delta / spot_volume if spot_volume > 0 else None
                ),
                "cvd_slope_base_per_hour": trade_delta / max(1, len(selected)),
                "ofi_normalized_mean": (
                    float(np.mean(ofi_values)) if ofi_values else None
                ),
                "complete_hours": sum(
                    item.get("status") == "COMPLETE" for item in selected
                ),
                "required_hours": hours,
            }
        output[market] = {
            "status": "READY" if fresh and latest_complete else "DATA_GAP",
            "fresh": fresh,
            "available_at": latest["hour_end"].isoformat(),
            "age_seconds": max(0.0, age.total_seconds()),
            "reason_codes": list(latest.get("reason_codes") or []),
            "spot_cvd_cumulative_base": _finite(
                latest.get("spot_cvd_cumulative_base")
            ),
            "spot_cvd_robust_zscore": _finite(
                latest.get("spot_cvd_robust_zscore")
            ),
            "orderbook_imbalance_top_5": _finite(
                latest.get("orderbook_imbalance_top_5")
            ),
            "orderbook_imbalance_top_10": _finite(
                latest.get("orderbook_imbalance_top_10", latest.get("orderbook_imbalance"))
            ),
            "orderbook_imbalance_top_25": _finite(
                latest.get("orderbook_imbalance_top_25")
            ),
            "orderbook_imbalance_within_10bps": _finite(
                latest.get("orderbook_imbalance_within_10bps")
            ),
            "microprice": _finite(latest.get("microprice")),
            "spread_bps": _finite(latest.get("spread_bps")),
            "bullish_absorption_score": _finite(
                latest.get("bullish_absorption_score")
            ),
            "bearish_absorption_score": _finite(
                latest.get("bearish_absorption_score")
            ),
            "horizons": horizons,
            "synthetic_data_used": False,
        }
    status_counts = Counter(
        str(row.get("status") or "DATA_PENDING")
        for row in output.values()
    )
    requested_count = len(output)
    ready_count = int(status_counts.get("READY", 0))
    data_gap_count = int(status_counts.get("DATA_GAP", 0))
    data_pending_count = int(status_counts.get("DATA_PENDING", 0))
    reason_counts = Counter(
        str(reason)
        for row in output.values()
        for reason in row.get("reason_codes") or []
    )
    aggregate_status = (
        "READY"
        if requested_count and ready_count == requested_count
        else "PARTIAL"
        if ready_count
        else "DATA_GAP"
        if data_gap_count
        else "DATA_PENDING"
    )
    return {
        "status": aggregate_status,
        "observed_at": observed.isoformat(),
        "requested_market_count": requested_count,
        "ready_market_count": ready_count,
        "data_gap_market_count": data_gap_count,
        "data_pending_market_count": data_pending_count,
        "ready_fraction": (
            ready_count / requested_count if requested_count else 0.0
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "markets": output,
        "source": "PROSPECTIVE_HASH_CHAINED_BITVAVO_ORDERFLOW",
        "historical_backfill_permitted": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def load_orderflow_15m_context(
    settings: Settings,
    *,
    markets: Sequence[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Load only fully sealed prospective 15-minute spot-flow buckets."""

    observed = (now or utc_now()).astimezone(UTC)
    directory = settings.paths.context_data_dir / "microstructure_15m"
    raw = (
        _snapshot_rows(directory, now=observed, maximum_hours=4)
        if directory.is_dir()
        else []
    )
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        market = str(row.get("market") or "").upper()
        if market in markets:
            by_market.setdefault(market, []).append(row)

    output: dict[str, Any] = {}
    for market in markets:
        rows = sorted(
            by_market.get(market, []),
            key=lambda item: item["hour_end"],
        )
        if not rows:
            output[market] = {
                "status": "DATA_PENDING",
                "reason_codes": ["NO_PROSPECTIVE_15M_ORDERFLOW"],
                "fresh": False,
            }
            continue
        latest = rows[-1]
        age = observed - latest["hour_end"]
        fresh = age <= timedelta(minutes=35)
        latest_complete = latest.get("status") == "COMPLETE"
        horizons: dict[str, Any] = {}
        for label, periods in (("15m", 1), ("1h", 4)):
            selected = rows[-periods:]
            trade_delta = sum(
                _finite(item.get("trade_delta_base")) or 0.0
                for item in selected
            )
            spot_volume = sum(
                _finite(item.get("spot_base_volume")) or 0.0
                for item in selected
            )
            ofi_values = [
                value
                for item in selected
                if (
                    value := _finite(
                        item.get("order_flow_imbalance_normalized")
                    )
                )
                is not None
            ]
            hours = periods / 4.0
            horizons[label] = {
                "trade_delta_base": trade_delta,
                "trade_delta_percentage": (
                    trade_delta / spot_volume if spot_volume > 0 else None
                ),
                "cvd_slope_base_per_hour": trade_delta / max(hours, 0.25),
                "ofi_normalized_mean": (
                    float(np.mean(ofi_values)) if ofi_values else None
                ),
                "complete_periods": sum(
                    item.get("status") == "COMPLETE" for item in selected
                ),
                "required_periods": periods,
            }
        output[market] = {
            "status": "READY" if fresh and latest_complete else "DATA_GAP",
            "fresh": fresh,
            "available_at": latest["hour_end"].isoformat(),
            "age_seconds": max(0.0, age.total_seconds()),
            "reason_codes": list(latest.get("reason_codes") or []),
            "spot_cvd_cumulative_base": _finite(
                latest.get("spot_cvd_cumulative_base")
            ),
            "spot_cvd_robust_zscore": _finite(
                latest.get("spot_cvd_robust_zscore")
            ),
            "orderbook_imbalance_top_5": _finite(
                latest.get("orderbook_imbalance_top_5")
            ),
            "orderbook_imbalance_top_10": _finite(
                latest.get(
                    "orderbook_imbalance_top_10",
                    latest.get("orderbook_imbalance"),
                )
            ),
            "orderbook_imbalance_top_25": _finite(
                latest.get("orderbook_imbalance_top_25")
            ),
            "orderbook_imbalance_within_10bps": _finite(
                latest.get("orderbook_imbalance_within_10bps")
            ),
            "microprice": _finite(latest.get("microprice")),
            "spread_bps": _finite(latest.get("spread_bps")),
            "bullish_absorption_score": _finite(
                latest.get("bullish_absorption_score")
            ),
            "bearish_absorption_score": _finite(
                latest.get("bearish_absorption_score")
            ),
            "horizons": horizons,
            "synthetic_data_used": False,
        }

    status_counts = Counter(
        str(row.get("status") or "DATA_PENDING")
        for row in output.values()
    )
    requested_count = len(output)
    ready_count = int(status_counts.get("READY", 0))
    data_gap_count = int(status_counts.get("DATA_GAP", 0))
    data_pending_count = int(status_counts.get("DATA_PENDING", 0))
    aggregate_status = (
        "READY"
        if requested_count and ready_count == requested_count
        else "PARTIAL"
        if ready_count
        else "DATA_GAP"
        if data_gap_count
        else "DATA_PENDING"
    )
    return {
        "status": aggregate_status,
        "observed_at": observed.isoformat(),
        "requested_market_count": requested_count,
        "ready_market_count": ready_count,
        "data_gap_market_count": data_gap_count,
        "data_pending_market_count": data_pending_count,
        "ready_fraction": (
            ready_count / requested_count if requested_count else 0.0
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "markets": output,
        "source": "PROSPECTIVE_HASH_CHAINED_BITVAVO_ORDERFLOW_15M",
        "historical_backfill_permitted": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_market_mechanics_snapshot(
    settings: Settings,
    *,
    markets: Sequence[str],
    now: datetime | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    observed = (now or utc_now()).astimezone(UTC)
    btc = load_gex_context(settings, underlying="BTC", now=observed)
    eth = load_gex_context(settings, underlying="ETH", now=observed)
    orderflow = load_orderflow_context(settings, markets=markets, now=observed)
    orderflow_15m = load_orderflow_15m_context(
        settings,
        markets=markets,
        now=observed,
    )
    rows: dict[str, Any] = {}
    for market in markets:
        local_gex = eth if market == "ETH-EUR" else btc
        rows[market] = {
            "market": market,
            "gex": local_gex,
            "gex_scope": (
                "COIN_SPECIFIC_ETH" if market == "ETH-EUR" else
                "COIN_SPECIFIC_BTC" if market == "BTC-EUR" else
                "BTC_MARKET_REGIME_PROXY_FOR_ALTCOIN"
            ),
            "orderflow": (orderflow.get("markets") or {}).get(market, {}),
            "orderflow_15m": (orderflow_15m.get("markets") or {}).get(
                market,
                {},
            ),
        }
    payload = {
        "schema_version": "market_mechanics_snapshot_v1",
        "generated_at": observed.isoformat(),
        "btc_gex": btc,
        "eth_gex": eth,
        "orderflow": orderflow,
        "orderflow_15m": orderflow_15m,
        "markets": rows,
        "gex_is_direct_entry_signal": False,
        "orderflow_is_execution_confirmation": True,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    if write_artifact:
        target = settings.paths.output_dir / "active_trading" / "market_mechanics.json"
        atomic_write_json(target, payload)
    return payload


__all__ = [
    "build_market_mechanics_snapshot",
    "load_gex_context",
    "load_orderflow_15m_context",
    "load_orderflow_context",
]
