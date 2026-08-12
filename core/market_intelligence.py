"""Transparent point-in-time coin ranking and token-fundamental coverage."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.common import append_jsonl, atomic_write_json, read_json, stable_hash, utc_iso, utc_now

RANKING_SCHEMA_VERSION = "coin_ranking_v1"
TOKENOMICS_SCHEMA_VERSION = "token_fundamentals_v1"


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted((value, key) for key, value in values.items())
    denominator = max(1, len(ordered) - 1)
    return {
        key: 100.0 * index / denominator
        for index, (_, key) in enumerate(ordered)
    }


def _closed_frame(settings: Settings, market: str) -> pd.DataFrame:
    path = settings.paths.processed_data_dir / f"{market}_1h.parquet"
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(
            path,
            columns=["timestamp", "close", "volume"],
        )
    except (OSError, ValueError, KeyError):
        return pd.DataFrame()
    if "timestamp" not in frame or "close" not in frame:
        return pd.DataFrame()
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(
        frame.get("volume"),
        errors="coerce",
    )
    return (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["close"])
        .loc[lambda value: ~value.index.duplicated(keep="last")]
        .sort_index()
    )


def _technical_snapshot(
    settings: Settings,
    market: str | None,
) -> dict[str, Any]:
    if not market:
        return {
            "data_status": "CONTEXT_ONLY_NO_EUR_MARKET",
            "data_through": None,
            "rows": 0,
        }
    frame = _closed_frame(settings, market)
    if len(frame) < 25:
        return {
            "data_status": "INSUFFICIENT_CLOSED_1H_DATA",
            "data_through": (
                frame.index[-1].isoformat() if not frame.empty else None
            ),
            "rows": len(frame),
        }
    close = frame["close"]
    returns = close.pct_change().dropna()

    def trailing_return(hours: int) -> float | None:
        if len(close) <= hours:
            return None
        return float(close.iloc[-1] / close.iloc[-hours - 1] - 1.0)

    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = (
        float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        if len(close) >= 200
        else None
    )
    latest = float(close.iloc[-1])
    realised_volatility = (
        float(returns.iloc[-24 * 30 :].std(ddof=1) * math.sqrt(24 * 365))
        if len(returns) >= 48
        else None
    )
    now = pd.Timestamp(utc_now())
    age_hours = max(
        0.0,
        float((now - frame.index[-1]).total_seconds() / 3_600.0),
    )
    return {
        "data_status": (
            "READY_CLOSED_1H"
            if age_hours <= 2.25 and len(frame) >= 200
            else "STALE_OR_SHORT_HISTORY"
        ),
        "data_through": frame.index[-1].isoformat(),
        "data_age_hours": age_hours,
        "rows": len(frame),
        "latest_close": latest,
        "return_24h": trailing_return(24),
        "return_7d": trailing_return(24 * 7),
        "return_30d": trailing_return(24 * 30),
        "ema50": ema50,
        "ema200": ema200,
        "above_ema50": latest > ema50,
        "above_ema200": (latest > ema200 if ema200 is not None else None),
        "realised_volatility_annualized": realised_volatility,
    }


def _token_record(row: Mapping[str, Any]) -> dict[str, Any]:
    market_cap = _number(row.get("market_cap"))
    circulating_supply = _number(row.get("circulating_supply"))
    total_supply = _number(row.get("total_supply"))
    maximum_supply = _number(row.get("maximum_supply"))
    fully_diluted_valuation = _number(
        row.get("fully_diluted_valuation")
        or row.get("fdv")
    )
    required = {
        "circulating_supply": circulating_supply,
        "total_supply": total_supply,
        "maximum_supply": maximum_supply,
        "market_cap": market_cap,
        "fully_diluted_valuation": fully_diluted_valuation,
        "unlock_calendar": row.get("unlock_calendar"),
        "holder_concentration": row.get("holder_concentration"),
        "protocol_revenue": row.get("protocol_revenue"),
        "token_value_capture": row.get("token_value_capture"),
        "security_incidents": row.get("security_incidents"),
    }
    present = sum(value is not None for value in required.values())
    coverage = present / len(required)
    hard_ineligible = (
        bool(row.get("stablecoin"))
        or bool(row.get("wrapped"))
        or bool(row.get("leveraged_token"))
        or bool(row.get("staking_derivative"))
    )
    if hard_ineligible:
        status = "NOT_EXECUTION_ELIGIBLE"
    elif coverage >= 0.8:
        status = "COMPLETE"
    elif coverage >= 0.5:
        status = "PARTIAL"
    else:
        status = "REVIEW_REQUIRED"
    price_estimate = (
        market_cap / circulating_supply
        if market_cap is not None
        and circulating_supply is not None
        and circulating_supply > 0
        else None
    )
    venue_execution_eligibility = str(
        row.get("execution_eligibility")
        or "NOT_EXECUTION_ELIGIBLE"
    )
    if status == "NOT_EXECUTION_ELIGIBLE":
        execution_eligibility = "NOT_EXECUTION_ELIGIBLE"
    elif venue_execution_eligibility != "LIVE_ELIGIBLE":
        execution_eligibility = "NOT_EXECUTION_ELIGIBLE"
    elif status in {"COMPLETE", "PARTIAL"}:
        execution_eligibility = "LIVE_ELIGIBLE"
    else:
        execution_eligibility = "REVIEW_REQUIRED"
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "name": row.get("name"),
        "rank": row.get("rank"),
        "snapshot_timestamp": row.get("snapshot_timestamp"),
        "available_at": row.get("available_at"),
        "market_cap": market_cap,
        "circulating_supply": circulating_supply,
        "total_supply": total_supply,
        "maximum_supply": maximum_supply,
        "fully_diluted_valuation": fully_diluted_valuation,
        "implied_price_from_market_cap": price_estimate,
        "unlock_calendar": row.get("unlock_calendar"),
        "holder_concentration": row.get("holder_concentration"),
        "protocol_revenue": row.get("protocol_revenue"),
        "token_value_capture": row.get("token_value_capture"),
        "security_incidents": row.get("security_incidents"),
        "stablecoin": bool(row.get("stablecoin")),
        "wrapped": bool(row.get("wrapped")),
        "leveraged_token": bool(row.get("leveraged_token")),
        "staking_derivative": bool(row.get("staking_derivative")),
        "shariah_status": row.get("shariah_status"),
        "research_eligibility": row.get("research_eligibility"),
        "venue_execution_eligibility": venue_execution_eligibility,
        "execution_eligibility": execution_eligibility,
        "coverage_fraction": coverage,
        "missing_fields": sorted(
            key for key, value in required.items() if value is None
        ),
        "status": status,
    }


def refresh_token_fundamentals(settings: Settings) -> dict[str, Any]:
    """Materialize sanitized point-in-time fundamental coverage."""

    source_path = (
        settings.paths.output_dir / "universe" / "top50_current.json"
    )
    source = _mapping(source_path)
    rows = [
        _token_record(row)
        for row in source.get("rows") or []
        if isinstance(row, Mapping)
    ]
    rows.sort(key=lambda row: (int(row.get("rank") or 10_000), row["symbol"]))
    output = settings.paths.output_dir / "tokenomics"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TOKENOMICS_SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "source": "POINT_IN_TIME_TOP50_SNAPSHOT",
        "source_path": str(source_path),
        "source_snapshot_hash": source.get("source_snapshot_hash"),
        "asset_count": len(rows),
        "complete_count": sum(row["status"] == "COMPLETE" for row in rows),
        "partial_count": sum(row["status"] == "PARTIAL" for row in rows),
        "review_required_count": sum(
            row["status"] == "REVIEW_REQUIRED" for row in rows
        ),
        "execution_review_required_count": sum(
            row["execution_eligibility"] == "REVIEW_REQUIRED"
            for row in rows
        ),
        "live_execution_eligible_count": sum(
            row["execution_eligibility"] == "LIVE_ELIGIBLE"
            for row in rows
        ),
        "not_execution_eligible_count": sum(
            row["status"] == "NOT_EXECUTION_ELIGIBLE" for row in rows
        ),
        "missing_data_lowers_confidence": True,
        "missing_data_interpreted_as_positive": False,
        "assets": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    atomic_write_json(output / "current.json", payload)
    return {
        **payload,
        "artifact": str((output / "current.json").resolve()),
    }


def inspect_token_fundamentals(
    settings: Settings,
    asset: str,
) -> dict[str, Any]:
    path = settings.paths.output_dir / "tokenomics" / "current.json"
    payload = (
        _mapping(path)
        if path.is_file()
        else refresh_token_fundamentals(settings)
    )
    symbol = str(asset).upper().replace("-EUR", "")
    match = next(
        (
            dict(row)
            for row in payload.get("assets") or []
            if str(row.get("symbol") or "").upper() == symbol
        ),
        None,
    )
    return {
        "status": "FOUND" if match else "NOT_FOUND",
        "asset": match,
        "artifact": str(path.resolve()),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_coin_ranking(settings: Settings) -> dict[str, Any]:
    """Rank top-50 assets using transparent closed-data subscores."""

    source_path = (
        settings.paths.output_dir / "universe" / "top50_current.json"
    )
    source = _mapping(source_path)
    universe_rows = [
        dict(row)
        for row in source.get("rows") or []
        if isinstance(row, Mapping)
    ]
    if not universe_rows:
        raise ValueError("TOP50_POINT_IN_TIME_SNAPSHOT_MISSING")
    tokenomics = refresh_token_fundamentals(settings)
    token_by_symbol = {
        str(row["symbol"]): row
        for row in tokenomics["assets"]
    }
    technical_by_symbol = {
        str(row.get("symbol") or "").upper(): _technical_snapshot(
            settings,
            str(row.get("eur_spot_market") or "") or None,
        )
        for row in universe_rows
    }
    volume_values = {
        str(row.get("symbol") or "").upper(): math.log1p(
            max(0.0, _number(row.get("volume_24h")) or 0.0)
        )
        for row in universe_rows
    }
    market_cap_values = {
        str(row.get("symbol") or "").upper(): math.log1p(
            max(0.0, _number(row.get("market_cap")) or 0.0)
        )
        for row in universe_rows
    }
    momentum_values = {
        symbol: float(technical["return_7d"])
        for symbol, technical in technical_by_symbol.items()
        if technical.get("return_7d") is not None
    }
    liquidity_percentile = _percentile(volume_values)
    capacity_percentile = _percentile(market_cap_values)
    momentum_percentile = _percentile(momentum_values)
    output = settings.paths.output_dir / "ranking"
    output.mkdir(parents=True, exist_ok=True)
    previous = _mapping(output / "current.json")
    previous_scores = {
        str(row.get("symbol")): float(row.get("stable_score") or 0.0)
        for row in previous.get("rows") or []
        if row.get("symbol")
    }
    ranked: list[dict[str, Any]] = []
    for row in universe_rows:
        symbol = str(row.get("symbol") or "").upper()
        technical = technical_by_symbol[symbol]
        token = token_by_symbol.get(symbol) or {}
        trend_score = (
            100.0
            if technical.get("above_ema50") is True
            and technical.get("above_ema200") is True
            else 65.0
            if technical.get("above_ema50") is True
            else 35.0
            if technical.get("above_ema50") is False
            else 0.0
        )
        realised_volatility = _number(
            technical.get("realised_volatility_annualized")
        )
        volatility_fit = (
            80.0
            if realised_volatility is not None
            and 0.25 <= realised_volatility <= 1.50
            else 40.0
            if realised_volatility is not None
            else 0.0
        )
        execution_score = (
            100.0
            if row.get("execution_eligibility") == "LIVE_ELIGIBLE"
            else 40.0
            if row.get("venue_availability")
            else 0.0
        )
        shariah_score = (
            100.0 if row.get("shariah_status") == "ALLOWED" else 0.0
        )
        data_quality = (
            100.0
            if technical.get("data_status") == "READY_CLOSED_1H"
            else 40.0
            if int(technical.get("rows") or 0) >= 200
            else 0.0
        )
        tokenomics_quality = float(token.get("coverage_fraction") or 0.0) * 100.0
        venue_execution_eligible = (
            row.get("execution_eligibility") == "LIVE_ELIGIBLE"
            and row.get("shariah_status") == "ALLOWED"
        )
        tokenomics_status = str(token.get("status") or "REVIEW_REQUIRED")
        live_execution_eligible = (
            venue_execution_eligible
            and tokenomics_status in {"COMPLETE", "PARTIAL"}
        )
        live_execution_reason = (
            "PASSED"
            if live_execution_eligible
            else "TOKEN_FUNDAMENTALS_REVIEW_REQUIRED"
            if venue_execution_eligible
            and tokenomics_status == "REVIEW_REQUIRED"
            else "TOKEN_NOT_EXECUTION_ELIGIBLE"
            if tokenomics_status == "NOT_EXECUTION_ELIGIBLE"
            else "VENUE_SHARIAH_OR_UNIVERSE_NOT_ELIGIBLE"
        )
        subscores = {
            "liquidity": liquidity_percentile.get(symbol, 0.0),
            "market_capacity": capacity_percentile.get(symbol, 0.0),
            "momentum_7d": momentum_percentile.get(symbol, 50.0),
            "trend_quality": trend_score,
            "volatility_fit": volatility_fit,
            "execution_eligibility": execution_score,
            "shariah_eligibility": shariah_score,
            "data_quality": data_quality,
            "tokenomics_quality": tokenomics_quality,
            "correlation_penalty": 0.0,
            "regime_fit": 50.0,
        }
        weights = {
            "liquidity": 0.18,
            "market_capacity": 0.10,
            "momentum_7d": 0.18,
            "trend_quality": 0.14,
            "volatility_fit": 0.08,
            "execution_eligibility": 0.10,
            "shariah_eligibility": 0.10,
            "data_quality": 0.07,
            "tokenomics_quality": 0.03,
            "regime_fit": 0.02,
        }
        raw_score = sum(
            subscores[key] * weight for key, weight in weights.items()
        )
        previous_score = previous_scores.get(symbol)
        stable_score = (
            0.75 * raw_score + 0.25 * previous_score
            if previous_score is not None
            else raw_score
        )
        ranked.append(
            {
                "symbol": symbol,
                "name": row.get("name"),
                "market": row.get("eur_spot_market"),
                "source_rank": row.get("rank"),
                "raw_score": raw_score,
                "stable_score": stable_score,
                "subscores": subscores,
                "weights": weights,
                "technical": technical,
                "tokenomics_status": tokenomics_status,
                "execution_eligibility": row.get("execution_eligibility"),
                "research_eligibility": row.get("research_eligibility"),
                "shariah_status": row.get("shariah_status"),
                "ranking_eligible": (
                    row.get("research_eligibility") == "RESEARCH_ELIGIBLE"
                ),
                "venue_execution_eligible": venue_execution_eligible,
                "live_execution_eligible": live_execution_eligible,
                "live_execution_eligibility_reason": live_execution_reason,
                "operator_exception_may_override_review": True,
                "operator_exception_changes_fundamental_status": False,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["ranking_eligible"],
            row["stable_score"],
            -int(row.get("source_rank") or 10_000),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    generated_at = utc_iso()
    payload = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_snapshot_hash": source.get("source_snapshot_hash"),
        "source_snapshot_timestamp": source.get("source_collected_at")
        or source.get("generated_at"),
        "closed_candles_only": True,
        "hysteresis": {
            "enabled": True,
            "current_weight": 0.75,
            "previous_weight": 0.25,
        },
        "transparent_subscores": True,
        "missing_data_interpreted_as_positive": False,
        "row_count": len(ranked),
        "venue_execution_eligible_count": sum(
            row["venue_execution_eligible"] for row in ranked
        ),
        "live_execution_eligible_count": sum(
            row["live_execution_eligible"] for row in ranked
        ),
        "rows": ranked,
        "ranking_hash": stable_hash(ranked, length=64),
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
    }
    atomic_write_json(output / "current.json", payload)
    append_jsonl(
        output / "history.jsonl",
        {
            "generated_at": generated_at,
            "source_snapshot_hash": payload["source_snapshot_hash"],
            "ranking_hash": payload["ranking_hash"],
            "top_symbols": [row["symbol"] for row in ranked[:10]],
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    return {
        **payload,
        "artifact": str((output / "current.json").resolve()),
        "history": str((output / "history.jsonl").resolve()),
    }


def inspect_coin_ranking(
    settings: Settings,
    asset: str,
) -> dict[str, Any]:
    path = settings.paths.output_dir / "ranking" / "current.json"
    payload = _mapping(path) if path.is_file() else build_coin_ranking(settings)
    symbol = str(asset).upper().replace("-EUR", "")
    match = next(
        (
            dict(row)
            for row in payload.get("rows") or []
            if str(row.get("symbol") or "").upper() == symbol
        ),
        None,
    )
    return {
        "status": "FOUND" if match else "NOT_FOUND",
        "asset": match,
        "artifact": str(path.resolve()),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "build_coin_ranking",
    "inspect_coin_ranking",
    "inspect_token_fundamentals",
    "refresh_token_fundamentals",
]
