"""Canonical whole-market context for the integrated crypto stack.

Orderless by design: canonical crypto owns data/evidence/execution truth;
crypto-ai-swing-layer may only enrich this packet as advisory context.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.settings import ACTIVE_SWING_TIMEFRAMES, TIMEFRAME_SECONDS
from research.features import FeaturePipeline
from scrapers.intelligence import load_intelligence

SCHEMA_VERSION = "crypto_whole_market_context_v1"
WHOLE_STACK_FEATURE_ROWS = 1500
MIN_FEATURE_ROWS = 220
SWING_WEIGHTS = {"15m": 0.05, "1h": 0.35, "4h": 0.40, "1d": 0.20}
POSITION_WEIGHTS = {"1h": 0.10, "4h": 0.20, "1d": 0.45, "1W": 0.25}


def _utc(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _seconds(tf: str) -> int:
    key = "1W" if tf.lower() == "1w" else tf
    return int(TIMEFRAME_SECONDS[key])


def _provenance(path: Path) -> dict[str, Any]:
    target = Path(str(path) + ".provenance.json")
    if not target.is_file():
        return {"status": "MISSING", "path": str(target)}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "INVALID", "path": str(target), "error": f"{type(exc).__name__}:{exc}"}
    return {
        "status": "READY" if payload.get("source_type") == "REAL_PROVIDER_DATA" else "UNVERIFIED_SOURCE",
        "path": str(target),
        "source_type": payload.get("source_type"),
        "provider": payload.get("provider"),
        "rows": payload.get("rows"),
        "sha256": payload.get("sha256"),
        "source_segments": payload.get("source_segments"),
    }


def _closed_frame(path: Path, market: str, tf: str, as_of: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        return pd.DataFrame(), {"status": "MISSING", "path": str(path)}
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} requires timestamp column or DatetimeIndex")
    idx = pd.DatetimeIndex(frame.index)
    frame.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    seconds = _seconds(tf)
    close_times = frame.index + pd.to_timedelta(seconds, unit="s")
    frame = frame.loc[close_times <= pd.Timestamp(as_of)].copy()
    frame.attrs.update({"market": market, "timeframe": tf})
    provenance = _provenance(path)
    if frame.empty:
        return frame, {"status": "NO_CLOSED_CANDLES", "path": str(path), "provenance": provenance}
    last_open = frame.index[-1].to_pydatetime()
    last_close = last_open + timedelta(seconds=seconds)
    age = max(0.0, (as_of - last_close).total_seconds())
    fresh_limit = max(seconds * 3.0, 1800.0)
    blockers = []
    if provenance.get("status") != "READY":
        blockers.append("REAL_PROVIDER_PROVENANCE_NOT_READY")
    if age > fresh_limit:
        blockers.append("STALE_CANDLES")
    if len(frame) < MIN_FEATURE_ROWS:
        blockers.append("INSUFFICIENT_FEATURE_HISTORY")
    gaps = 0
    if len(frame) > 1:
        delta = np.diff(frame.index.view("i8")) / 1e9
        gaps = int(np.sum(delta > seconds * 1.5))
    return frame, {
        "status": "READY" if not blockers else "DEGRADED",
        "blockers": blockers,
        "path": str(path),
        "rows": len(frame),
        "last_open": last_open.isoformat(),
        "last_close": last_close.isoformat(),
        "age_seconds": age,
        "fresh": age <= fresh_limit,
        "freshness_limit_seconds": fresh_limit,
        "gap_count": gaps,
        "provenance": provenance,
    }


def _load_intelligence(root: Path, as_of: datetime) -> tuple[list[Any], dict[str, Any]]:
    directory = root / "data_store" / "intelligence"
    candidates = sorted(directory.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True) if directory.is_dir() else []
    failures = []
    for path in candidates:
        try:
            records = [r for r in load_intelligence(path) if getattr(r, "usable_at", None) is not None and _utc(r.usable_at) <= as_of]
            return records, {"status": "READY" if records else "EMPTY", "path": str(path), "record_count": len(records), "load_failures": failures}
        except Exception as exc:
            failures.append(f"{path.name}:{type(exc).__name__}")
    return [], {"status": "MISSING", "record_count": 0, "load_failures": failures}


def _news(records: list[Any], market: str, as_of: datetime) -> dict[str, Any]:
    cutoff = as_of - timedelta(hours=168)
    rows, weighted, total = [], 0.0, 0.0
    for r in records:
        markets = {str(x).upper() for x in getattr(r, "markets", ()) or ()}
        categories = {str(x) for x in getattr(r, "categories", ()) or ()}
        macro = bool(categories & {"macro_calendar", "macro_liquidity", "interest_rates", "inflation", "regulation"})
        if markets and market not in markets and not macro:
            continue
        ts = _utc(getattr(r, "usable_at"))
        if not cutoff <= ts <= as_of:
            continue
        relevance = max(0.0, _num(getattr(r, "relevance_score", None)) or 0.0)
        impact = max(0.0, _num(getattr(r, "impact_score", None)) or 0.0)
        sentiment = _num(getattr(r, "sentiment_score", None))
        age_h = max(0.0, (as_of - ts).total_seconds() / 3600.0)
        weight = max(0.05, relevance) * max(0.05, impact) * math.exp(-age_h / 48.0)
        if sentiment is not None:
            weighted += sentiment * weight
            total += weight
        rows.append({"source": getattr(r, "source", None), "title": getattr(r, "title", None), "url": getattr(r, "url", None), "usable_at": ts.isoformat(), "sentiment_score": sentiment, "relevance_score": relevance, "impact_score": impact})
    rows.sort(key=lambda x: x["usable_at"], reverse=True)
    return {
        "status": "READY" if rows else "MISSING",
        "record_count": len(rows),
        "sentiment_score": weighted / total if total > 0 else None,
        "confidence": float(np.clip(1.0 - math.exp(-len(rows) / 8.0), 0.0, 1.0)),
        "top_events": rows[:12],
    }


def _technical(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"status": "MISSING", "technical_score": None}
    row = features.iloc[-1]
    close, ema21, ema50, sma200 = map(_num, (row.get("close"), row.get("ema_21"), row.get("ema_50"), row.get("sma_200")))
    rsi, macd, mom, vr = map(_num, (row.get("rsi_14"), row.get("macd_hist"), row.get("momentum_10"), row.get("volume_ratio")))
    parts = []
    def add(v, w):
        if v is not None:
            parts.append((float(np.clip(v, -1, 1)), w))
    if close is not None and ema21 is not None: add(1 if close > ema21 else -1, .18)
    if ema21 is not None and ema50 is not None: add(1 if ema21 > ema50 else -1, .16)
    if close is not None and sma200 is not None: add(1 if close > sma200 else -1, .16)
    if rsi is not None: add((rsi - 50) / 25, .16)
    if macd is not None: add(1 if macd > 0 else -1 if macd < 0 else 0, .12)
    if mom is not None: add(float(np.tanh(mom / .02)), .10)
    if vr is not None: add(vr - 1, .04)
    breakout, breakdown = bool(row.get("breakout_20", False)), bool(row.get("breakdown_20", False))
    if breakout: add(1, .08)
    elif breakdown: add(-1, .08)
    denom = sum(w for _, w in parts)
    score = sum(v * w for v, w in parts) / denom if denom else None
    bb_upper = _num(row.get("bb_upper"))
    overextended = bool(rsi is not None and rsi >= 75 and close is not None and bb_upper is not None and close >= bb_upper)
    if breakout: state = "BREAKOUT_20"
    elif breakdown: state = "BREAKDOWN_20"
    elif score is not None and score >= .45 and rsi is not None and rsi >= 55: state = "TREND_CONTINUATION"
    elif score is not None and score >= .25 and rsi is not None and 40 <= rsi <= 55: state = "TREND_PULLBACK"
    else: state = "NEUTRAL"
    return {"status": "READY" if score is not None else "DEGRADED", "technical_score": score, "latest_at": features.index[-1].isoformat(), "close": close, "rsi_14": rsi, "ema_21": ema21, "ema_50": ema50, "sma_200": sma200, "macd_hist": macd, "momentum_10": mom, "volume_ratio": vr, "atr_14": _num(row.get("atr_14")), "rolling_volatility_20": _num(row.get("rolling_volatility_20")), "breakout_state": state, "overextended": overextended}


def _weighted(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    rows = [(scores[k], w) for k, w in weights.items() if scores.get(k) is not None]
    denom = sum(w for _, w in rows)
    return float(np.clip(sum(float(v) * w for v, w in rows) / denom, -1, 1)) if denom else None


def _mtf(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = {tf: _num(v.get("technical_score")) for tf, v in states.items()}
    available = [v for v in scores.values() if v is not None]
    if not available:
        return {"status": "MISSING", "scores": scores}
    pos, neg = sum(v > 0 for v in available), sum(v < 0 for v in available)
    return {"status": "READY", "swing_score": _weighted(scores, SWING_WEIGHTS), "position_score": _weighted(scores, POSITION_WEIGHTS), "all_timeframe_consensus": float(np.mean(available)), "alignment_fraction": max(pos, neg) / len(available), "alignment_score_01": max(pos, neg) / len(available), "direction": 1 if pos > neg else -1 if neg > pos else 0, "available_timeframes": [k for k, v in scores.items() if v is not None], "scores": scores, "weight_contract": {"swing": SWING_WEIGHTS, "position": POSITION_WEIGHTS, "all_timeframe_consensus": "UNWEIGHTED_DESCRIPTIVE_MEAN"}}


def _research(root: Path) -> dict[str, Any]:
    base = root / "output" / "research" / "operational-stack" / "acceptance"
    rows = []
    if base.is_dir():
        for path in sorted(base.glob("*/research_summary.json")):
            try: p = json.loads(path.read_text(encoding="utf-8"))
            except Exception: continue
            rows.append({"strategy_id": p.get("strategy_id") or path.parent.name, "status": p.get("status"), "passed": bool(p.get("passed", False)), "reasons": p.get("reasons") or [], "parameters": p.get("parameters") or {}, "artifact": str(path)})
    return {"status": "READY" if rows else "MISSING", "strategy_count": len(rows), "passed_gate_count": sum(r["passed"] for r in rows), "rejected_gate_count": sum(not r["passed"] for r in rows), "strategies": rows, "interpretation": "PROCESS_SUCCESS_IS_NOT_EDGE_APPROVAL"}


def build_whole_market_context(project_root: Path | str, *, markets: Iterable[str] = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"), timeframes: Iterable[str] = ACTIVE_SWING_TIMEFRAMES, as_of: datetime | str | None = None) -> dict[str, Any]:
    root, decision = Path(project_root).expanduser().resolve(), _utc(as_of)
    markets, timeframes = tuple(dict.fromkeys(str(x).upper() for x in markets)), tuple(dict.fromkeys(str(x) for x in timeframes))
    intelligence, intelligence_status = _load_intelligence(root, decision)
    frames, quality = {}, {}
    for market in markets:
        for tf in timeframes:
            path = root / "data_store" / "normalized" / f"{market}_{tf}.parquet"
            try: frames[(market, tf)], quality[(market, tf)] = _closed_frame(path, market, tf, decision)
            except Exception as exc: frames[(market, tf)], quality[(market, tf)] = pd.DataFrame(), {"status": "ERROR", "path": str(path), "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
    payload = {}
    for market in markets:
        states = {}
        for tf in timeframes:
            frame = frames[(market, tf)]
            if len(frame) < MIN_FEATURE_ROWS:
                states[tf] = {"status": "MISSING", "technical_score": None, "data_quality": quality[(market, tf)]}
                continue
            benchmark = frames.get(("BTC-EUR", tf))
            try:
                snapshot_frame = frame.tail(WHOLE_STACK_FEATURE_ROWS).copy()
                snapshot_benchmark = (
                    benchmark.tail(WHOLE_STACK_FEATURE_ROWS).copy()
                    if benchmark is not None and not benchmark.empty
                    else None
                )
                features = FeaturePipeline().build(
                    snapshot_frame,
                    market=market,
                    benchmark=snapshot_benchmark,
                    intelligence=intelligence or None,
                )
                tech = _technical(features)
            except Exception as exc:
                tech = {"status": "ERROR", "technical_score": None, "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
            states[tf] = {**tech, "data_quality": quality[(market, tf)]}
        payload[market] = {"timeframes": states, "mtf": _mtf(states), "news": _news(intelligence, market, decision), "fundamentals": {"status": "PENDING_EXTERNAL_ENRICHMENT", "source": "crypto-ai-swing-layer:CMCContextCollector"}}
    qrows = list(quality.values())
    if any(q.get("status") in {"MISSING", "ERROR", "NO_CLOSED_CANDLES"} for q in qrows): status = "BLOCKED_DATA"
    elif any(q.get("status") == "DEGRADED" for q in qrows): status = "DEGRADED_CONTEXT"
    else: status = "READY_FOR_ADVISORY_ENRICHMENT"
    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(), "decision_as_of": decision.isoformat(), "status": status, "markets": list(markets), "timeframes": list(timeframes), "canonical_crypto_root": str(root), "market_context": payload, "intelligence_dataset": intelligence_status, "research_evidence": _research(root), "authority": {"execution_authority": "SJAGIL_CRYPTO_ONLY", "swing_authority": "ADVISORY_ONLY", "live_decision_influence": False, "automatic_live_promotion": False, "autoscale": False}, "safety": {"closed_candles_only": True,
                "feature_context_rows": WHOLE_STACK_FEATURE_ROWS, "lookahead_policy": "candle_open + timeframe <= decision_as_of", "intelligence_lookahead_policy": "usable_at <= decision_as_of", "private_exchange_requests": 0, "orders_generated": 0, "orders_submitted": 0}}
