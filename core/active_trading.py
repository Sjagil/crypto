"""Active, causal multi-timeframe opportunity and capital control layer.

This module expands what the canonical live supervisor *observes*.  It never
grants strategy authority and never submits an order directly.  When
``execute=True`` it delegates exclusively to the existing generated-strategy
live executor, which enforces frozen DNA, reconciliation and risk limits.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from core.early_move_detector import detect_early_moves
from core.generated_strategy_live import execute_generated_strategy_live_once
from core.live_universe import (
    build_tiered_trading_universe,
    live_universe_status,
)
from core.market_mechanics import build_market_mechanics_snapshot
from core.regime_policy import regime_policy
from data.data_loader import TIMEFRAME_SECONDS, DataLoader
from data.market_data import drop_open_candles, resample_ohlcv
from research.backtest import BacktestConfig, BacktestEngine
from research.features import FeaturePipeline
from research.gex_orderflow_strategies import (
    evaluate_market_mechanics_strategy,
    market_mechanics_strategy_specs,
)
from research.stochastic_validation import (
    StochasticValidationPolicy,
    validate_strategy_return_paths,
)
from research.tactical_multitimeframe import (
    TacticalMultiTimeframeStrategy,
    TacticalStrategySpec,
    tactical_catalogue_payload,
    tactical_strategy_specs,
)
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    stable_hash,
    utc_iso,
    utc_now,
)

ACTIVE_TRADING_VERSION = "active_trading_v1"
TACTICAL_TIMEFRAMES = ("15m", "1h", "2h")
OBSERVED_TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d", "1W")
TIMEFRAME_DIRECTION_WEIGHTS = {
    "15m": 0.18,
    "1h": 0.30,
    "2h": 0.24,
    "4h": 0.18,
    "1d": 0.07,
    "1W": 0.03,
}
FAST_TIMEFRAMES = ("15m", "1h", "2h", "4h")
TIMEFRAME_DISAGREEMENT_PENALTY = 0.20

# Tactical research DNA never inherits exact-DNA execution authority.  These
# routes merely describe which separately operator-approved event playbook may
# consume the closed-candle setup after its own realtime, cost and risk checks.
TACTICAL_FAMILY_AUTHORITY_ROUTES: dict[str, tuple[str, ...]] = {
    "TREND_PULLBACK": (
        "TREND_PULLBACK_V1",
        "NORMAL_SWING_TREND_RETEST_V1",
    ),
    "RANGE_VWAP_REVERSION": ("RANGE_VWAP_REVERSION_V1",),
    "RANGE_MEAN_REVERSION": ("RANGE_VWAP_REVERSION_V1",),
    "VOLATILITY_COMPRESSION_BREAKOUT": ("COMPRESSION_BREAKOUT_V1",),
    "COMPRESSION_BREAKOUT": ("COMPRESSION_BREAKOUT_V1",),
    "RS_LEADER_PULLBACK": ("RS_LEADER_PULLBACK_V1",),
    "RELATIVE_STRENGTH_CONTINUATION": ("RS_LEADER_PULLBACK_V1",),
    "BREAKOUT_RETEST": ("NORMAL_SWING_BREAKOUT_RETEST_V1",),
    "NORMAL_SWING_TREND_RETEST": ("NORMAL_SWING_TREND_RETEST_V1",),
    "NORMAL_SWING_BREAKOUT_RETEST": (
        "NORMAL_SWING_BREAKOUT_RETEST_V1",
    ),
    "LIQUIDITY_SWEEP_RECLAIM": (
        "LIQUIDITY_SWEEP_RECLAIM_V1",
        "BEAR_SPOT_LIQUIDITY_RECOVERY_V1",
    ),
    "FAILED_BREAKDOWN_REVERSAL": (
        "FAILED_BREAKDOWN_REVERSAL_V1",
        "BEAR_SPOT_FAILED_BREAKDOWN_V1",
    ),
    "VWAP_RECLAIM": ("VWAP_RECLAIM_V1",),
    "RELATIVE_STRENGTH_ROTATION": ("RELATIVE_STRENGTH_ROTATION_V1",),
    "MOMENTUM_BREAKOUT": ("MOMENTUM_BREAKOUT_V1",),
    "BREAKOUT_PULLBACK": ("BREAKOUT_PULLBACK_V1",),
}


def _safe_read(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if math.isfinite(selected) else default


def _family_canary_authority(
    settings: Settings,
    *,
    family: str,
    market: str,
    entry_timeframe: str,
) -> dict[str, Any]:
    """Report a bounded family route without granting exact-DNA authority."""

    authority = _safe_read(
        settings.paths.project_root
        / "config"
        / "live_playbook_authority.json"
    )
    routed_ids = TACTICAL_FAMILY_AUTHORITY_ROUTES.get(
        str(family).upper(),
        (),
    )
    approved: list[dict[str, Any]] = []
    entry_route_eligible = entry_timeframe in {"15m", "1h"}
    if authority.get("active") is True and routed_ids and entry_route_eligible:
        for raw in authority.get("approved_playbooks") or []:
            row = dict(raw) if isinstance(raw, Mapping) else {}
            if row.get("active") is not True:
                continue
            if str(row.get("playbook_id") or "") not in routed_ids:
                continue
            markets = {str(value) for value in row.get("markets") or []}
            if markets and market not in markets:
                continue
            approved.append(row)
    maximum_order = _finite(authority.get("maximum_order_eur"), 0.0) or 0.0
    effective_orders = [
        maximum_order
        * max(
            0.0,
            min(1.0, _finite(row.get("evidence_multiplier"), 1.0) or 0.0),
        )
        for row in approved
    ]
    return {
        "available": bool(approved),
        "playbook_ids": [str(row["playbook_id"]) for row in approved],
        "strategy_roles": sorted(
            {
                str(row.get("strategy_role") or "VALIDATED_PLAYBOOK")
                for row in approved
            }
        ),
        "maximum_effective_order_eur": (
            max(effective_orders) if effective_orders else 0.0
        ),
        "requires_realtime_confirmation": bool(approved),
        "exact_dna_authority_granted": False,
        "entry_timeframe_route_eligible": entry_route_eligible,
    }


def _candle_timeframe_score(frame: pd.DataFrame) -> dict[str, Any] | None:
    """Score one closed-candle timeframe without inventing order flow."""

    required = {"high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns) or len(frame) < 55:
        return None
    source = frame.loc[:, sorted(required)].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    if len(source) < 55:
        return None
    close = source["close"]
    high = source["high"]
    low = source["low"]
    volume = source["volume"]
    prior_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    atr_scale = max(1e-12, 1.5 * atr)
    trend = float(np.tanh((ema20.iloc[-1] - ema50.iloc[-1]) / atr_scale))

    momentum_periods = min(8, len(close) - 1)
    log_returns = np.log(close).diff()
    realized = float(
        log_returns.iloc[-max(20, momentum_periods * 3) :].std(ddof=0)
    )
    normalized_momentum = (
        math.log(close.iloc[-1] / close.iloc[-momentum_periods - 1])
        / max(1e-12, realized * math.sqrt(momentum_periods))
    )
    momentum = float(np.tanh(normalized_momentum))

    prior_high = float(high.shift(1).rolling(20).max().iloc[-1])
    prior_low = float(low.shift(1).rolling(20).min().iloc[-1])
    range_width = max(1e-12, prior_high - prior_low)
    range_location = max(
        -1.0,
        min(1.0, 2.0 * (float(close.iloc[-1]) - prior_low) / range_width - 1.0),
    )
    ema_slope = float(
        np.tanh((ema20.iloc[-1] - ema20.iloc[-5]) / atr_scale)
    )
    structure = 0.60 * range_location + 0.40 * ema_slope

    baseline_volume = float(volume.shift(1).rolling(20).median().iloc[-1])
    relative_volume = (
        float(volume.iloc[-1]) / baseline_volume
        if baseline_volume > 0.0
        else 1.0
    )
    signed_direction = 1.0 if momentum >= 0.0 else -1.0
    volume_score = signed_direction * float(
        np.tanh(max(0.0, relative_volume - 1.0))
    )

    change = close.diff().abs()
    efficiency_denominator = float(change.iloc[-20:].sum())
    efficiency_ratio = (
        abs(float(close.iloc[-1] - close.iloc[-21]))
        / efficiency_denominator
        if efficiency_denominator > 0.0
        else 0.0
    )
    # Flow is intentionally absent here.  Its 20% component is only admitted
    # by the realtime event-driven layer when actual trades/books are fresh.
    available_components = {
        "trend": (trend, 0.25),
        "momentum": (momentum, 0.20),
        "structure": (structure, 0.20),
        "volume": (volume_score, 0.15),
    }
    available_weight = sum(weight for _, weight in available_components.values())
    score = sum(
        value * weight for value, weight in available_components.values()
    ) / available_weight
    return {
        "score": max(-1.0, min(1.0, float(score))),
        "components": {
            "trend": trend,
            "momentum_volatility_adjusted": momentum,
            "structure": structure,
            "volume": volume_score,
            "flow": None,
        },
        "efficiency_ratio_20": efficiency_ratio,
        "relative_volume_20": relative_volume,
        "flow_status": "DEFERRED_TO_REALTIME_EXECUTION_LAYER",
    }


def _weighted_timeframe_assessment(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    market: str,
) -> dict[str, Any]:
    per_timeframe: dict[str, dict[str, Any]] = {}
    for timeframe in OBSERVED_TIMEFRAMES:
        frame = frames.get((market, timeframe))
        if frame is None:
            continue
        result = _candle_timeframe_score(frame)
        if result is not None:
            per_timeframe[timeframe] = result
    available_weight = sum(
        TIMEFRAME_DIRECTION_WEIGHTS[timeframe]
        for timeframe in per_timeframe
    )
    composite = (
        sum(
            TIMEFRAME_DIRECTION_WEIGHTS[timeframe] * row["score"]
            for timeframe, row in per_timeframe.items()
        )
        / available_weight
        if available_weight > 0.0
        else 0.0
    )
    fast_rows = [
        per_timeframe[timeframe]["score"]
        for timeframe in FAST_TIMEFRAMES
        if timeframe in per_timeframe
    ]
    fast_weight = sum(
        TIMEFRAME_DIRECTION_WEIGHTS[timeframe]
        for timeframe in FAST_TIMEFRAMES
        if timeframe in per_timeframe
    )
    fast_score = (
        sum(
            TIMEFRAME_DIRECTION_WEIGHTS[timeframe]
            * per_timeframe[timeframe]["score"]
            for timeframe in FAST_TIMEFRAMES
            if timeframe in per_timeframe
        )
        / fast_weight
        if fast_weight > 0.0
        else 0.0
    )
    slow_rows = [
        per_timeframe[timeframe]["score"]
        for timeframe in ("1d", "1W")
        if timeframe in per_timeframe
    ]
    slow_score = float(np.mean(slow_rows)) if slow_rows else 0.0
    disagreement = float(np.std(fast_rows, ddof=0)) if fast_rows else 1.0
    adjusted = max(
        -1.0,
        min(
            1.0,
            composite - TIMEFRAME_DISAGREEMENT_PENALTY * disagreement,
        ),
    )
    routing_source = per_timeframe.get("1h") or per_timeframe.get("2h") or {}
    efficiency = float(routing_source.get("efficiency_ratio_20") or 0.0)
    market_mode = (
        "TREND"
        if efficiency > 0.35
        else "RANGE"
        if efficiency < 0.20
        else "MIXED"
    )
    if fast_score >= 0.35 and slow_score < -0.20:
        trade_type = "COUNTERTREND_LONG"
        threshold = 0.50
        risk_multiplier = 0.65
    elif fast_score >= 0.35 and slow_score >= 0.0:
        trade_type = "TREND_LONG"
        threshold = 0.35
        risk_multiplier = 1.0
    else:
        trade_type = "TACTICAL_LONG"
        threshold = 0.40
        risk_multiplier = 0.85
    conflicts = [
        f"{timeframe}_DIRECTION_NEGATIVE"
        for timeframe, row in per_timeframe.items()
        if float(row["score"]) < -0.10
    ]
    return {
        "score": adjusted,
        "raw_composite_score": composite,
        "fast_score": fast_score,
        "slow_score": slow_score,
        "disagreement": disagreement,
        "disagreement_penalty": TIMEFRAME_DISAGREEMENT_PENALTY * disagreement,
        "entry_threshold": threshold,
        "trade_type": trade_type,
        "market_mode": market_mode,
        "risk_multiplier": risk_multiplier,
        "per_timeframe": per_timeframe,
        "weights": dict(TIMEFRAME_DIRECTION_WEIGHTS),
        "conflicts": conflicts,
        "missing_timeframes": [
            timeframe
            for timeframe in OBSERVED_TIMEFRAMES
            if timeframe not in per_timeframe
        ],
        "hard_blocked_by_1d_or_1w": False,
    }


def _family_mode_fit(family: str, market_mode: str) -> float:
    selected = family.upper()
    reversal = any(
        token in selected
        for token in ("REVERS", "RECLAIM", "SWEEP", "RECOVERY", "RANGE")
    )
    trend = any(
        token in selected
        for token in ("TREND", "BREAKOUT", "MOMENTUM", "EXPANSION")
    )
    if market_mode == "TREND":
        return 1.0 if trend else 0.80 if reversal else 0.90
    if market_mode == "RANGE":
        return 1.0 if reversal else 0.75 if trend else 0.90
    return 0.90


def _strategy_directional_gate(
    family: str,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Route closed-candle triggers without reintroducing an HTF veto.

    The weighted timeframe model already assigns only ten percent to 1d/1W.
    Requiring a universal +0.40 composite after a deterministic 15m/1h
    trigger effectively counted the same higher-timeframe disagreement a
    second time.  Continuation families still need positive fast momentum;
    explicit reversal/recovery families may operate around neutral after a
    causal reclaim.  Deeply negative fast structure remains blocked.
    """

    selected = str(family or "").upper()
    score = float(assessment.get("score") or 0.0)
    fast_score = float(assessment.get("fast_score") or 0.0)
    base_threshold = float(assessment.get("entry_threshold") or 0.40)
    reversal = any(
        token in selected
        for token in (
            "REVERS",
            "RECLAIM",
            "SWEEP",
            "RECOVERY",
            "RANGE_MEAN_REVERSION",
        )
    )
    if reversal:
        effective_threshold = min(base_threshold, -0.20)
        fast_floor = -0.15
        basis = "CAUSAL_REVERSAL_TRIGGER_WITH_NON_HOSTILE_FAST_STRUCTURE"
    else:
        effective_threshold = min(base_threshold, 0.15)
        fast_floor = 0.20
        basis = "POSITIVE_FAST_STRUCTURE_WITH_SOFT_HIGHER_TIMEFRAMES"
    approved = bool(
        score >= effective_threshold
        and fast_score >= fast_floor
    )
    return {
        "approved": approved,
        "base_threshold": base_threshold,
        "effective_threshold": effective_threshold,
        "fast_score_floor": fast_floor,
        "basis": basis,
        "higher_timeframes_are_soft": True,
    }


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").to_pydatetime()


def _freshness(
    available_at: Any,
    *,
    maximum_age_hours: float,
    now: datetime,
) -> dict[str, Any]:
    parsed = _timestamp(available_at)
    if parsed is None:
        return {
            "available_at": None,
            "age_hours": None,
            "freshness": "MISSING",
            "fresh": False,
            "from_future": False,
        }
    raw_age = (now - parsed).total_seconds() / 3_600.0
    from_future = raw_age < 0.0
    age = max(0.0, raw_age)
    return {
        "available_at": parsed.isoformat(),
        "age_hours": age,
        "freshness": (
            "FUTURE"
            if from_future
            else "FRESH"
            if age <= maximum_age_hours
            else "STALE"
        ),
        "fresh": not from_future and age <= maximum_age_hours,
        "from_future": from_future,
    }


def _normalized_frame(
    settings: Settings,
    market: str,
    timeframe: str,
    *,
    maximum_rows: int,
    now: datetime,
    require_fresh: bool = True,
) -> pd.DataFrame:
    path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path.name}: timestamp column missing")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.loc[:, ["open", "high", "low", "close", "volume"]]
    frame = (
        frame.apply(pd.to_numeric, errors="coerce")
        .dropna()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
        .sort_index()
    )
    frame.attrs.update(
        {
            "market": market,
            "timeframe": timeframe,
            "data_provenance": {
                "source_type": "REAL_PROVIDER_DATA",
                "path": str(path),
            },
        }
    )
    frame = drop_open_candles(frame, timeframe=timeframe, now=now)
    if require_fresh:
        _assert_frame_fresh(frame, timeframe=timeframe, now=now)
    if len(frame) > maximum_rows:
        frame = frame.iloc[-maximum_rows:].copy()
        frame.attrs.update(
            {
                "market": market,
                "timeframe": timeframe,
                "data_provenance": {
                    "source_type": "REAL_PROVIDER_DATA",
                    "path": str(path),
                },
            }
        )
    return frame


def _records_to_ohlcv_frame(
    records: Sequence[Any],
    *,
    market: str,
    timeframe: str,
    now: datetime,
) -> pd.DataFrame:
    """Convert only real, closed provider records into a causal OHLCV tail."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if getattr(record, "closed", None) is not True:
            continue
        values = dict(getattr(record, "values", {}) or {})
        rows.append(
            {
                "timestamp": getattr(record, "timestamp", None),
                "open": values.get("open"),
                "high": values.get("high"),
                "low": values.get("low"),
                "close": values.get("close"),
                "volume": values.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    frame = (
        frame.loc[:, ["open", "high", "low", "close", "volume"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
        .sort_index()
    )
    frame = drop_open_candles(frame, timeframe=timeframe, now=now)
    frame.attrs.update(
        {
            "market": market,
            "timeframe": timeframe,
            "data_provenance": {
                "source_type": "BITVAVO_REST_CLOSED_CANDLE_TAIL",
                "synthetic_data_used": False,
            },
        }
    )
    return frame


def _frame_gap_summary(frame: pd.DataFrame, *, timeframe: str) -> dict[str, Any]:
    interval = int(TIMEFRAME_SECONDS[timeframe])
    if len(frame) < 2:
        return {"gap_count": 0, "maximum_gap_seconds": 0}
    differences = frame.index.to_series().diff().dt.total_seconds().dropna()
    gaps = differences[differences > interval * 1.5]
    return {
        "gap_count": int(len(gaps)),
        "maximum_gap_seconds": int(gaps.max()) if not gaps.empty else 0,
    }


def _realtime_closed_candle_frame(
    settings: Settings,
    *,
    market: str,
    timeframe: str,
    now: datetime,
) -> pd.DataFrame:
    projection = _safe_read(
        settings.paths.output_dir / "operations" / "realtime_candles.json"
    )
    rows = list(
        (projection.get("closed_candles") or {}).get(
            f"{market}:{timeframe}",
            [],
        )
    )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    frame = (
        frame.loc[:, ["open", "high", "low", "close", "volume"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
        .sort_index()
    )
    return drop_open_candles(frame, timeframe=timeframe, now=now)


async def _recover_recent_fast_frames(
    settings: Settings,
    markets: Sequence[str],
    *,
    frames: Mapping[tuple[str, str], pd.DataFrame],
    now: datetime,
    maximum_rows: int,
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, Any]]:
    """Repair missing/stale 15m and 1h tails without mutating canonical history.

    The continuous research synchronizer may take longer than one tactical
    scan.  This recovery path fetches a bounded native Bitvavo tail, keeps
    closed candles only and overlays it on the existing real history.  It
    never forward-fills gaps and never writes concurrently to Parquet.
    """

    hard_stale_after_seconds = {"15m": 1_500, "1h": 4_800}

    def requires_recovery(market: str, timeframe: str) -> bool:
        frame = frames.get((market, timeframe))
        if frame is None or frame.empty:
            return True
        interval = int(TIMEFRAME_SECONDS[timeframe])
        latest_close = (
            frame.index[-1].to_pydatetime().astimezone(UTC)
            + timedelta(seconds=interval)
        )
        return (now - latest_close).total_seconds() > (
            hard_stale_after_seconds[timeframe]
        )

    missing = [
        (market, timeframe)
        for market in markets
        for timeframe in ("15m", "1h")
        if requires_recovery(market, timeframe)
    ]
    report: dict[str, Any] = {
        "status": "NOT_REQUIRED" if not missing else "RUNNING",
        "requested": len(missing),
        "recovered": 0,
        "failed": 0,
        "rows": {},
        "synthetic_data_used": False,
        "canonical_files_mutated": False,
    }
    if not missing:
        return {}, report

    loader = DataLoader(settings)
    semaphore = asyncio.Semaphore(8)

    async def recover_one(
        market: str,
        timeframe: str,
    ) -> tuple[tuple[str, str], pd.DataFrame | None, dict[str, Any]]:
        key = (market, timeframe)
        history_days = 4 if timeframe == "15m" else 8
        try:
            try:
                base = _normalized_frame(
                    settings,
                    market,
                    timeframe,
                    maximum_rows=maximum_rows,
                    now=now,
                    require_fresh=False,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError):
                base = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
            websocket_tail = _realtime_closed_candle_frame(
                settings,
                market=market,
                timeframe=timeframe,
                now=now,
            )
            websocket_parts = [
                value for value in (base, websocket_tail) if not value.empty
            ]
            websocket_merged = (
                pd.concat(websocket_parts).sort_index()
                if websocket_parts
                else base.copy()
            )
            websocket_merged = websocket_merged.loc[
                ~websocket_merged.index.duplicated(keep="last")
            ]
            try:
                _assert_frame_fresh(
                    websocket_merged,
                    timeframe=timeframe,
                    now=now,
                )
            except ValueError:
                pass
            else:
                if len(websocket_merged) > maximum_rows:
                    websocket_merged = websocket_merged.iloc[-maximum_rows:].copy()
                websocket_merged.attrs.update(
                    {
                        "market": market,
                        "timeframe": timeframe,
                        "data_provenance": {
                            "source_type": "CANONICAL_PLUS_WEBSOCKET_CLOSED_CANDLES",
                            "synthetic_data_used": False,
                            "forward_filled": False,
                        },
                    }
                )
                latest_close = (
                    websocket_merged.index[-1].to_pydatetime().astimezone(UTC)
                    + timedelta(seconds=int(TIMEFRAME_SECONDS[timeframe]))
                )
                return key, websocket_merged, {
                    "status": "RECOVERED_FROM_WEBSOCKET",
                    "tail_rows": int(len(websocket_tail)),
                    "merged_rows": int(len(websocket_merged)),
                    "latest_closed_at": latest_close.isoformat(),
                    "age_seconds": max(
                        0,
                        int((now - latest_close).total_seconds()),
                    ),
                    **_frame_gap_summary(
                        websocket_merged,
                        timeframe=timeframe,
                    ),
                }
            async with semaphore:
                records = await loader.download_ohlcv(
                    provider="bitvavo",
                    market=market,
                    timeframe=timeframe,
                    start=now - timedelta(days=history_days),
                    end=now,
                    resume=False,
                    persist=False,
                )
            tail = _records_to_ohlcv_frame(
                records,
                market=market,
                timeframe=timeframe,
                now=now,
            )
            merged_parts = [
                value
                for value in (base, websocket_tail, tail)
                if not value.empty
            ]
            merged = (
                pd.concat(merged_parts).sort_index()
                if merged_parts
                else base.copy()
            )
            merged = merged.loc[~merged.index.duplicated(keep="last")]
            merged = drop_open_candles(merged, timeframe=timeframe, now=now)
            _assert_frame_fresh(merged, timeframe=timeframe, now=now)
            if len(merged) > maximum_rows:
                merged = merged.iloc[-maximum_rows:].copy()
            merged.attrs.update(
                {
                    "market": market,
                    "timeframe": timeframe,
                    "data_provenance": {
                        "source_type": "CANONICAL_PLUS_BITVAVO_REST_TAIL",
                        "synthetic_data_used": False,
                        "forward_filled": False,
                    },
                }
            )
            latest_close = merged.index[-1].to_pydatetime().astimezone(UTC) + timedelta(
                seconds=int(TIMEFRAME_SECONDS[timeframe])
            )
            detail = {
                "status": "RECOVERED",
                "tail_rows": int(len(tail)),
                "merged_rows": int(len(merged)),
                "latest_closed_at": latest_close.isoformat(),
                "age_seconds": max(0, int((now - latest_close).total_seconds())),
                **_frame_gap_summary(merged, timeframe=timeframe),
            }
            return key, merged, detail
        except Exception as exc:
            return key, None, {
                "status": "FAILED",
                "reason": str(exc) or type(exc).__name__,
                "exception_type": type(exc).__name__,
            }

    results = await asyncio.gather(
        *(recover_one(market, timeframe) for market, timeframe in missing)
    )
    recovered: dict[tuple[str, str], pd.DataFrame] = {}
    for key, frame, detail in results:
        label = f"{key[0]}:{key[1]}"
        report["rows"][label] = detail
        if frame is None:
            report["failed"] += 1
        else:
            recovered[key] = frame
            report["recovered"] += 1
    report["status"] = (
        "READY"
        if report["failed"] == 0
        else "PARTIAL"
        if report["recovered"] > 0
        else "FAILED"
    )
    return recovered, report


def _assert_frame_fresh(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    now: datetime,
) -> None:
    """Reject a tactical frame older than two complete candle intervals."""

    if frame.empty:
        raise ValueError(f"EMPTY_{timeframe.upper()}_DATA")
    interval_seconds = int(TIMEFRAME_SECONDS[timeframe])
    latest_close = frame.index[-1].to_pydatetime().astimezone(UTC) + timedelta(
        seconds=interval_seconds
    )
    age_seconds = max(
        0.0,
        (now.astimezone(UTC) - latest_close).total_seconds(),
    )
    if age_seconds > 2 * interval_seconds:
        raise ValueError(
            f"STALE_{timeframe.upper()}_DATA:age_seconds={age_seconds:.0f}"
        )


def _fast_data_health(
    settings: Settings,
    frames: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Expose candle-age SLOs independently from service process state."""

    thresholds = {
        "1m": (90, 180),
        "5m": (360, 600),
        "15m": (1_020, 1_500),
        "1h": (3_900, 4_800),
        "2h": (7_500, 8_400),
        "4h": (14_700, 15_600),
        "1d": (86_700, 87_600),
        "1W": (605_100, 606_000),
    }
    realtime = _safe_read(
        settings.paths.output_dir / "operations" / "realtime_candles.json"
    )
    closed_projection = dict(realtime.get("closed_candles") or {})
    timeframe_rows: dict[str, Any] = {}
    for timeframe, (warning_seconds, hard_seconds) in thresholds.items():
        rows: list[dict[str, Any]] = []
        interval = int(TIMEFRAME_SECONDS[timeframe])
        for market in markets:
            latest: datetime | None = None
            if timeframe in {"1m", "5m"}:
                projected = list(
                    closed_projection.get(f"{market}:{timeframe}") or []
                )
                if projected:
                    latest_timestamp = _timestamp(projected[-1].get("timestamp"))
                    if latest_timestamp is not None:
                        latest = latest_timestamp + timedelta(seconds=interval)
            else:
                frame = frames.get((market, timeframe))
                if frame is not None and not frame.empty:
                    latest = (
                        frame.index[-1].to_pydatetime().astimezone(UTC)
                        + timedelta(seconds=interval)
                    )
            age = (
                max(0.0, (now - latest).total_seconds())
                if latest is not None
                else None
            )
            status = (
                "MISSING"
                if age is None
                else "HARD_STALE"
                if age > hard_seconds
                else "WARNING"
                if age > warning_seconds
                else "FRESH"
            )
            rows.append(
                {
                    "market": market,
                    "latest_closed_at": latest.isoformat() if latest else None,
                    "age_seconds": age,
                    "status": status,
                }
            )
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in ("FRESH", "WARNING", "HARD_STALE", "MISSING")
        }
        timeframe_rows[timeframe] = {
            "warning_after_seconds": warning_seconds,
            "hard_stale_after_seconds": hard_seconds,
            "counts": counts,
            "affected_markets": [
                row
                for row in rows
                if row["status"] in {"WARNING", "HARD_STALE", "MISSING"}
            ][:50],
        }
    hard_dependencies = sum(
        details["counts"]["HARD_STALE"] + details["counts"]["MISSING"]
        for details in timeframe_rows.values()
    )
    return {
        "schema_version": "fast_market_data_health_v1",
        "observed_at": now.isoformat(),
        "status": "DEGRADED" if hard_dependencies else "HEALTHY",
        "process_running_is_not_health": True,
        "dependency_scoped_fail_closed": True,
        "hard_unhealthy_dependency_count": hard_dependencies,
        "timeframes": timeframe_rows,
        "websocket_projection_generated_at": realtime.get("generated_at"),
        "last_rest_recovery": None,
    }


def _market_opportunity_intensity(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    features: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    *,
    previous: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    returns: dict[str, list[float]] = {"15m": [], "1h": [], "4h": []}
    normalized_1h: list[float] = []
    volume_positive = 0
    volume_observed = 0
    impulses: dict[str, float | None] = {"BTC-EUR": None, "ETH-EUR": None}
    for market in markets:
        for timeframe in returns:
            frame = frames.get((market, timeframe))
            if frame is None or len(frame) < 2:
                continue
            close = pd.to_numeric(frame["close"], errors="coerce")
            if close.tail(2).isna().any() or float(close.iloc[-2]) <= 0:
                continue
            selected_return = float(close.iloc[-1] / close.iloc[-2] - 1.0)
            returns[timeframe].append(selected_return)
            if timeframe == "1h":
                feature = features.get((market, "1h"))
                atr = (
                    _finite(feature["atr_14"].iloc[-1])
                    if feature is not None and "atr_14" in feature
                    else None
                )
                atr_fraction = (atr / float(close.iloc[-1])) if atr else None
                if atr_fraction and atr_fraction > 0:
                    normalized_1h.append(selected_return / atr_fraction)
                if market in impulses:
                    impulses[market] = selected_return
        feature_15m = features.get((market, "15m"))
        if feature_15m is not None and "relative_volume_20" in feature_15m:
            value = _finite(feature_15m["relative_volume_20"].iloc[-1])
            if value is not None:
                volume_observed += 1
                volume_positive += int(value > 1.0)
    breadth = {
        timeframe: (
            sum(value > 0.0 for value in values) / len(values)
            if values
            else 0.0
        )
        for timeframe, values in returns.items()
    }
    median_returns = {
        timeframe: float(np.median(values)) if values else 0.0
        for timeframe, values in returns.items()
    }
    previous_intensity = dict(previous.get("market_opportunity_intensity") or {})
    previous_breadth = dict(previous_intensity.get("breadth") or {})
    breadth_acceleration = breadth["1h"] - float(
        previous_breadth.get("1h") or breadth["1h"]
    )
    volume_breadth = volume_positive / volume_observed if volume_observed else 0.0
    btc_eth_impulse = float(
        np.mean([value for value in impulses.values() if value is not None])
    ) if any(value is not None for value in impulses.values()) else 0.0
    dispersion = float(np.std(returns["1h"])) if returns["1h"] else 0.0
    normalized_median = float(np.median(normalized_1h)) if normalized_1h else 0.0
    signed_score = (
        0.20 * (2.0 * breadth["15m"] - 1.0)
        + 0.20 * (2.0 * breadth["1h"] - 1.0)
        + 0.15 * float(np.tanh(breadth_acceleration / 0.15))
        + 0.15 * float(np.tanh(normalized_median))
        + 0.10 * (2.0 * volume_breadth - 1.0)
        + 0.10 * float(np.tanh(btc_eth_impulse / 0.01))
        + 0.10 * float(np.tanh(dispersion / 0.015))
    )
    score = float(np.clip(50.0 * (signed_score + 1.0), 0.0, 100.0))
    level = (
        "EXTREME"
        if score >= 80.0
        else "HIGH"
        if score >= 65.0
        else "NORMAL"
        if score >= 40.0
        else "LOW"
    )
    return {
        "schema_version": "market_opportunity_intensity_v1",
        "observed_at": now.isoformat(),
        "score": score,
        "level": level,
        "opportunity_regime": (
            "RISK_ON_IMPULSE"
            if level in {"HIGH", "EXTREME"} and breadth["1h"] >= 0.60
            else "RECOVERY_OR_MIXED"
            if score >= 40.0
            else "LOW_OPPORTUNITY"
        ),
        "breadth": breadth,
        "breadth_acceleration_1h": breadth_acceleration,
        "median_returns": median_returns,
        "median_atr_normalized_return_1h": normalized_median,
        "volume_breadth": volume_breadth,
        "btc_eth_impulse_1h": impulses,
        "cross_sectional_dispersion_1h": dispersion,
        "market_count": len(markets),
    }


def _record_summary(record: Any) -> dict[str, Any]:
    values = dict(getattr(record, "values", {}) or {})
    return {
        "provider": str(getattr(record, "provider", "") or ""),
        "source_symbol": str(getattr(record, "source_symbol", "") or ""),
        "timestamp": (
            getattr(record, "timestamp", None).isoformat()
            if getattr(record, "timestamp", None)
            else None
        ),
        "observed_at": (
            getattr(record, "observed_at", None).isoformat()
            if getattr(record, "observed_at", None)
            else None
        ),
        "available_at": (
            getattr(record, "available_at", None).isoformat()
            if getattr(record, "available_at", None)
            else None
        ),
        "values": {
            key: value
            for key, value in values.items()
            if key
            in {
                "total_market_cap",
                "total_volume_24h",
                "btc_dominance",
                "eth_dominance",
                "fear_greed",
                "classification",
                "funding_rate",
                "annualized_funding",
                "open_interest",
                "perpetual_premium",
                "basis",
                "close",
                "adjusted_close",
                "volume",
                "rank",
                "cmc_rank",
                "symbol",
                "name",
                "market_cap",
                "circulating_supply",
                "total_supply",
                "volume_24h",
                "quote_volume_24h",
            }
        },
    }


async def _refresh_scraper_intelligence(
    settings: Settings,
    *,
    now: datetime,
    minimum_interval_hours: float = 4.0,
) -> dict[str, Any]:
    """Refresh causal web/RSS context on a bounded cadence."""

    if not settings.scrapers.scrapers_enabled:
        return {"status": "DISABLED", "records": 0, "sources": []}
    status_path = settings.paths.intelligence_dir / "scraper_status.json"
    previous = _safe_read(status_path)
    last_observed = _timestamp(previous.get("observed_at"))
    source_ids = {
        str(row.get("source_id") or "").upper()
        for row in previous.get("sources") or []
    }
    real_attempt = bool(source_ids - {"", "SELF_TEST"})
    if (
        last_observed is not None
        and real_attempt
        and (now - last_observed).total_seconds()
        <= minimum_interval_hours * 3_600.0
    ):
        return {
            "status": "CACHED",
            "observed_at": last_observed.isoformat(),
            "records": int((previous.get("audit") or {}).get("record_count") or 0),
            "sources": sorted(source_ids),
        }
    try:
        from scrapers.intelligence import run_intelligence_pipeline

        result = await run_intelligence_pipeline(
            settings,
            observed_at=now,
            include_rss=True,
        )
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "error_code": type(exc).__name__,
            "records": 0,
            "sources": [],
        }
    return {
        "status": result.status,
        "observed_at": result.observed_at.isoformat(),
        "records": len(result.records),
        "sources": [row.source_id for row in result.sources],
    }


async def _capture(
    name: str,
    operation: Awaitable[Any],
) -> tuple[str, dict[str, Any]]:
    try:
        result = await operation
    except Exception as exc:
        return name, {
            "status": "UNAVAILABLE",
            "error_code": type(exc).__name__,
            "records": [],
        }
    records = result if isinstance(result, list) else [result]
    summaries = [
        _record_summary(record)
        for record in records
        if getattr(record, "values", None) is not None
    ]
    return name, {
        "status": "READY" if summaries else "EMPTY",
        "records": summaries,
    }


async def refresh_public_macro_context(settings: Settings) -> dict[str, Any]:
    """Fetch public context and append causal GEX observations.

    Most sources are read as ephemeral context. GEX is different: strategies
    require a point-in-time history, so every successful refresh is appended
    to the canonical BTC/ETH GEX parquet files. Failures remain isolated and
    are reported as context degradation; they never stop signal generation.
    """

    loader = DataLoader(settings)
    now = utc_now()
    start = now - timedelta(days=14)

    async def refresh_gex(underlying: str) -> tuple[str, dict[str, Any]]:
        name = f"deribit_{underlying.casefold()}_gex"
        try:
            summary = await loader.download_gex_context(
                underlying=underlying,
                persist=True,
            )
        except Exception as exc:
            return name, {
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "secrets_serialized": False,
            }
        return name, {
            "status": "READY",
            "available_at": summary.get("available_at")
            or summary.get("observed_at"),
            "provider": summary.get("provider") or "deribit",
            "point_in_time_history_appended": True,
            "secrets_serialized": False,
        }

    tasks = (
        _capture(
            "coinmarketcap_global",
            loader.download_macro_series(
                provider="coinmarketcap",
                series="GLOBAL",
                persist=False,
            ),
        ),
        _capture(
            "coinmarketcap_rankings",
            loader.download_cmc_rankings(
                limit=50,
                convert="EUR",
                persist=False,
            ),
        ),
        _capture(
            "fear_and_greed",
            loader.download_macro_series(
                provider="alternative_me",
                series="fear_and_greed",
                persist=False,
            ),
        ),
        _capture(
            "mexc_btc_derivatives",
            loader.download_derivatives_context(
                provider="mexc",
                market="BTC-USDT",
                persist=False,
            ),
        ),
        _capture(
            "mexc_eth_derivatives",
            loader.download_derivatives_context(
                provider="mexc",
                market="ETH-USDT",
                persist=False,
            ),
        ),
        _capture(
            "eodhd_vix",
            loader.download_macro_series(
                provider="eodhd",
                series="VIX.INDX",
                start=start,
                end=now,
                persist=False,
            ),
        ),
        _capture(
            "eodhd_ndx",
            loader.download_macro_series(
                provider="eodhd",
                series="NDX.INDX",
                start=start,
                end=now,
                persist=False,
            ),
        ),
        _capture(
            "eodhd_spx",
            loader.download_macro_series(
                provider="eodhd",
                series="GSPC.INDX",
                start=start,
                end=now,
                persist=False,
            ),
        ),
    )
    captured, gex, intelligence = await asyncio.gather(
        asyncio.gather(*tasks),
        asyncio.gather(
            refresh_gex("BTC"),
            refresh_gex("ETH"),
        ),
        _refresh_scraper_intelligence(settings, now=now),
    )
    return {
        "refreshed_at": now.isoformat(),
        "sources": {**dict(captured), **dict(gex)},
        "intelligence": intelligence,
        "secrets_serialized": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _latest_local_record(
    settings: Settings,
    filename: str,
) -> dict[str, Any]:
    path = settings.paths.context_data_dir / filename
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    sort_columns = [
        column
        for column in ("available_at", "timestamp", "observed_at")
        if column in frame
    ]
    row = frame.sort_values(sort_columns or [frame.columns[0]]).iloc[-1]
    return {
        "provider": row.get("provider"),
        "source_symbol": row.get("source_symbol"),
        "available_at": row.get("available_at"),
        "observed_at": row.get("observed_at"),
        "values": row.to_dict(),
        "path": str(path),
    }


def _public_latest(
    refreshed: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    rows = list(
        ((refreshed or {}).get("sources") or {}).get(source, {}).get("records")
        or []
    )
    if not rows:
        return {}

    def sort_key(row: Mapping[str, Any]) -> tuple[datetime, datetime]:
        available = _timestamp(row.get("available_at") or row.get("observed_at"))
        timestamp = _timestamp(row.get("timestamp"))
        minimum = datetime.min.replace(tzinfo=UTC)
        return available or minimum, timestamp or minimum

    return dict(max((dict(row) for row in rows), key=sort_key))


def _public_close_return(
    refreshed: Mapping[str, Any] | None,
    source: str,
    *,
    periods: int = 5,
) -> float | None:
    rows = list(
        ((refreshed or {}).get("sources") or {}).get(source, {}).get("records")
        or []
    )
    values: list[tuple[datetime, float]] = []
    for row in rows:
        timestamp = _timestamp(row.get("timestamp"))
        close = _finite(
            (row.get("values") or {}).get("close")
            or (row.get("values") or {}).get("adjusted_close")
        )
        if timestamp is not None and close is not None and close > 0.0:
            values.append((timestamp, close))
    unique = sorted(dict(values).items())
    if len(unique) <= periods:
        return None
    return float(unique[-1][1] / unique[-periods - 1][1] - 1.0)


def _stablecoin_liquidity_context(
    settings: Settings,
    *,
    refreshed: Mapping[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Build a causal live USDT/USDC and aggregate liquidity snapshot.

    CoinMarketCap changes use only observations that were actually available
    by the decision clock.  DefiLlama's historical reference series is useful
    for the current live state, but is explicitly marked as unavailable for
    retrospective backfills because its rows share the retrieval timestamp.
    """

    observed = now.astimezone(UTC)
    ranking_path = settings.paths.context_data_dir / "coinmarketcap_rankings.parquet"
    ranking_frames: list[pd.DataFrame] = []
    if ranking_path.is_file():
        try:
            local = pd.read_parquet(
                ranking_path,
                columns=[
                    "provider",
                    "available_at",
                    "observed_at",
                    "symbol",
                    "cmc_rank",
                    "market_cap",
                    "circulating_supply",
                    "volume_24h",
                ],
            )
        except (OSError, ValueError):
            local = pd.DataFrame()
        if not local.empty:
            ranking_frames.append(local)
    refreshed_rows: list[dict[str, Any]] = []
    for record in list(
        ((refreshed or {}).get("sources") or {})
        .get("coinmarketcap_rankings", {})
        .get("records")
        or []
    ):
        values = dict(record.get("values") or {})
        refreshed_rows.append(
            {
                "provider": record.get("provider") or "coinmarketcap",
                "available_at": record.get("available_at"),
                "observed_at": record.get("observed_at"),
                "symbol": values.get("symbol"),
                "cmc_rank": values.get("cmc_rank") or values.get("rank"),
                "market_cap": values.get("market_cap"),
                "circulating_supply": values.get("circulating_supply"),
                "volume_24h": values.get("volume_24h"),
            }
        )
    if refreshed_rows:
        ranking_frames.append(pd.DataFrame(refreshed_rows))
    rankings = (
        pd.concat(ranking_frames, ignore_index=True)
        if ranking_frames
        else pd.DataFrame()
    )
    if not rankings.empty:
        rankings["available_at"] = pd.to_datetime(
            rankings["available_at"], utc=True, errors="coerce"
        )
        rankings["symbol"] = rankings["symbol"].astype(str).str.upper()
        rankings = rankings.loc[
            rankings["available_at"].notna()
            & rankings["available_at"].le(pd.Timestamp(observed))
            & rankings["symbol"].isin({"USDT", "USDC"})
        ].copy()
        for column in ("market_cap", "circulating_supply", "volume_24h"):
            rankings[column] = pd.to_numeric(rankings[column], errors="coerce")
        rankings = rankings.sort_values("available_at").drop_duplicates(
            ["symbol", "available_at"], keep="last"
        )

    def asof_value(frame: pd.DataFrame, column: str, when: pd.Timestamp) -> float | None:
        eligible = frame.loc[frame["available_at"].le(when), column].dropna()
        if eligible.empty:
            return None
        return _finite(eligible.iloc[-1])

    stablecoins: dict[str, Any] = {}
    latest_available: list[datetime] = []
    for symbol in ("USDT", "USDC"):
        selected = (
            rankings.loc[rankings["symbol"].eq(symbol)].copy()
            if not rankings.empty
            else pd.DataFrame()
        )
        if selected.empty:
            stablecoins[symbol] = {
                "status": "MISSING",
                "provider": "coinmarketcap",
            }
            continue
        current = selected.iloc[-1]
        current_at = pd.Timestamp(current["available_at"])
        current_cap = _finite(current.get("market_cap"))
        latest_available.append(current_at.to_pydatetime())
        changes: dict[str, float | None] = {}
        for label, hours in (("1h", 1), ("6h", 6), ("24h", 24)):
            previous = asof_value(
                selected,
                "market_cap",
                current_at - timedelta(hours=hours),
            )
            changes[label] = (
                float(current_cap / previous - 1.0)
                if current_cap is not None and previous not in {None, 0.0}
                else None
            )
        stablecoins[symbol] = {
            "status": "READY",
            "provider": str(current.get("provider") or "coinmarketcap"),
            "available_at": current_at.isoformat(),
            "cmc_rank": int(current["cmc_rank"])
            if pd.notna(current.get("cmc_rank"))
            else None,
            "market_cap_eur": current_cap,
            "circulating_supply": _finite(current.get("circulating_supply")),
            "volume_24h_eur": _finite(current.get("volume_24h")),
            "market_cap_change_1h": changes["1h"],
            "market_cap_change_6h": changes["6h"],
            "market_cap_change_24h": changes["24h"],
        }

    def combined_change(hours: int) -> float | None:
        if rankings.empty:
            return None
        current_values: list[float] = []
        previous_values: list[float] = []
        for symbol in ("USDT", "USDC"):
            selected = rankings.loc[rankings["symbol"].eq(symbol)]
            if selected.empty:
                continue
            current_at = pd.Timestamp(selected["available_at"].iloc[-1])
            current = asof_value(selected, "market_cap", current_at)
            previous = asof_value(
                selected,
                "market_cap",
                current_at - timedelta(hours=hours),
            )
            if current is not None and previous not in {None, 0.0}:
                current_values.append(current)
                previous_values.append(previous)
        if not current_values or sum(previous_values) <= 0.0:
            return None
        return float(sum(current_values) / sum(previous_values) - 1.0)

    defillama_path = settings.paths.context_data_dir / "defillama_stablecoins.parquet"
    aggregate: dict[str, Any] = {
        "status": "MISSING",
        "provider": "defillama",
        "backtest_safe": False,
    }
    if defillama_path.is_file():
        try:
            total = pd.read_parquet(
                defillama_path,
                columns=[
                    "available_at",
                    "observation_time",
                    "stablecoin_market_cap",
                ],
            )
        except (OSError, ValueError):
            total = pd.DataFrame()
        if not total.empty:
            total["available_at"] = pd.to_datetime(
                total["available_at"], utc=True, errors="coerce"
            )
            total["observation_time"] = pd.to_datetime(
                total["observation_time"], utc=True, errors="coerce"
            )
            total["stablecoin_market_cap"] = pd.to_numeric(
                total["stablecoin_market_cap"], errors="coerce"
            )
            total = total.loc[
                total["available_at"].notna()
                & total["available_at"].le(pd.Timestamp(observed))
                & total["observation_time"].notna()
                & total["stablecoin_market_cap"].gt(0.0)
            ]
            if not total.empty:
                batch_at = total["available_at"].max()
                batch = (
                    total.loc[total["available_at"].eq(batch_at)]
                    .sort_values("observation_time")
                    .drop_duplicates("observation_time", keep="last")
                )
                current = batch.iloc[-1]
                current_value = float(current["stablecoin_market_cap"])

                def historical_change(days: int) -> float | None:
                    target = current["observation_time"] - timedelta(
                        days=days
                    )
                    prior = batch.loc[batch["observation_time"].le(target)]
                    if prior.empty:
                        return None
                    previous = float(prior.iloc[-1]["stablecoin_market_cap"])
                    return current_value / previous - 1.0 if previous > 0.0 else None

                latest_available.append(batch_at.to_pydatetime())
                aggregate = {
                    "status": "READY_LIVE_ONLY",
                    "provider": "defillama",
                    "available_at": batch_at.isoformat(),
                    "historical_reference_at": current["observation_time"].isoformat(),
                    "total_market_cap_usd": current_value,
                    "change_1d": historical_change(1),
                    "change_7d": historical_change(7),
                    "change_30d": historical_change(30),
                    "backtest_safe": False,
                    "point_in_time_note": (
                        "CURRENT_LIVE_SNAPSHOT_WITH_HISTORICAL_REFERENCE; "
                        "DO_NOT_RETROFIT_IN_BACKTESTS"
                    ),
                }

    combined_1h = combined_change(1)
    combined_6h = combined_change(6)
    combined_24h = combined_change(24)
    aggregate_1d = _finite(aggregate.get("change_1d"))
    evidence = [
        value
        for value in (combined_6h, combined_24h, aggregate_1d)
        if value is not None
    ]
    if evidence and (
        min(evidence) <= -0.0025
        or sum(value < -0.001 for value in evidence) >= 2
    ):
        state = "DRAINING"
        score = -1.0
        risk_multiplier = 0.75
    elif evidence and (
        max(evidence) >= 0.0025
        or sum(value > 0.001 for value in evidence) >= 2
    ):
        state = "EXPANDING"
        score = 1.0
        risk_multiplier = 1.05
    elif evidence and min(evidence) < 0.0 < max(evidence):
        state = "MIXED"
        score = 0.0
        risk_multiplier = 0.90
    elif evidence:
        state = "STABLE"
        score = 0.0
        risk_multiplier = 1.0
    else:
        state = "DATA_PENDING"
        score = 0.0
        risk_multiplier = 0.80
    available_at = max(latest_available).isoformat() if latest_available else None
    return {
        "status": "READY" if evidence else "DATA_PENDING",
        "state": state,
        "score": score,
        "risk_multiplier": risk_multiplier,
        "provider": "coinmarketcap+defillama",
        "observed_at": observed.isoformat(),
        "available_at": available_at,
        "usdt": stablecoins["USDT"],
        "usdc": stablecoins["USDC"],
        "combined_market_cap_change_1h": combined_1h,
        "combined_market_cap_change_6h": combined_6h,
        "combined_market_cap_change_24h": combined_24h,
        "aggregate": aggregate,
        "macro_only": True,
        "is_entry_signal": False,
    }


def _latest_intelligence_context(
    settings: Settings,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Summarize fresh non-synthetic forward intelligence as risk context."""

    path = settings.paths.intelligence_dir / "crypto_intelligence.parquet"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}
    if frame.empty or "usable_at" not in frame:
        return {}
    frame = frame.copy()
    frame["usable_at"] = pd.to_datetime(frame["usable_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(now - settings.scrapers.stale_news_after)
    selected = frame.loc[
        frame["usable_at"].notna()
        & frame["usable_at"].le(pd.Timestamp(now))
        & frame["usable_at"].ge(cutoff)
    ].copy()
    if "source" in selected:
        selected = selected.loc[
            ~selected["source"].astype(str).str.casefold().isin(
                {"self test", "self_test", "synthetic"}
            )
        ]
    if "url" in selected:
        selected = selected.loc[
            ~selected["url"].astype(str).str.contains(
                "example.test",
                case=False,
                na=False,
            )
        ]
    if selected.empty:
        return {}

    def categories(value: Any) -> set[str]:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                decoded = [value]
        else:
            decoded = value or []
        return {str(item).casefold() for item in decoded}

    risk_categories = {
        "exchange_risk",
        "regulation",
        "macro_calendar",
        "security_incident",
        "token_unlock",
    }
    risks: list[float] = []
    weighted_sentiment = 0.0
    sentiment_weight = 0.0
    for row in selected.to_dict(orient="records"):
        relevance = max(0.0, min(1.0, _finite(row.get("relevance_score"), 0.0) or 0.0))
        impact = max(0.0, min(1.0, _finite(row.get("impact_score"), 0.0) or 0.0))
        sentiment = max(-1.0, min(1.0, _finite(row.get("sentiment_score"), 0.0) or 0.0))
        category_set = categories(row.get("categories"))
        if category_set & risk_categories:
            risks.append(impact * relevance * (1.0 if sentiment <= 0.0 else 0.5))
        weighted_sentiment += sentiment * max(relevance, 0.01)
        sentiment_weight += max(relevance, 0.01)
    latest = selected["usable_at"].max().to_pydatetime()
    return {
        "provider": "multi_source_scraper",
        "available_at": latest.isoformat(),
        "observed_at": latest.isoformat(),
        "path": str(path),
        "values": {
            "event_count": int(len(selected)),
            "source_count": int(selected.get("source", pd.Series(dtype=str)).nunique()),
            "event_risk": max(risks, default=0.0),
            "sentiment": weighted_sentiment / sentiment_weight if sentiment_weight else 0.0,
            "sources": sorted(
                selected.get("source", pd.Series(dtype=str)).astype(str).unique().tolist()
            ),
        },
    }


def build_crypto_macro_snapshot(
    settings: Settings,
    *,
    refreshed: Mapping[str, Any] | None = None,
    market_frames: Mapping[tuple[str, str], pd.DataFrame] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a nuanced, freshness-aware crypto regime snapshot."""

    observed = (now or utc_now()).astimezone(UTC)
    cmc = _public_latest(refreshed, "coinmarketcap_global") or _latest_local_record(
        settings,
        "coinmarketcap_global.parquet",
    )
    fear = _public_latest(refreshed, "fear_and_greed") or _latest_local_record(
        settings,
        "alternative_me_fear_and_greed.parquet",
    )
    rankings = _public_latest(refreshed, "coinmarketcap_rankings") or _latest_local_record(
        settings,
        "coinmarketcap_rankings.parquet",
    )
    derivatives = _public_latest(
        refreshed,
        "mexc_btc_derivatives",
    ) or _latest_local_record(settings, "derivatives_mexc_BTC.parquet")
    eth_derivatives = _public_latest(
        refreshed,
        "mexc_eth_derivatives",
    ) or _latest_local_record(settings, "derivatives_mexc_ETH.parquet")
    vix = _public_latest(refreshed, "eodhd_vix") or _latest_local_record(
        settings,
        "eodhd_vix_indx.parquet",
    )
    ndx = _public_latest(refreshed, "eodhd_ndx") or _latest_local_record(
        settings,
        "eodhd_ndx_indx.parquet",
    )
    spx = _public_latest(refreshed, "eodhd_spx") or _latest_local_record(
        settings,
        "eodhd_gspc_indx.parquet",
    )
    dff = _latest_local_record(settings, "fred_dff.parquet")
    dgs10 = _latest_local_record(settings, "fred_dgs10.parquet")
    intelligence = _latest_intelligence_context(settings, now=observed)
    stablecoin_liquidity = _stablecoin_liquidity_context(
        settings,
        refreshed=refreshed,
        now=observed,
    )
    sources = {
        "coinmarketcap_global": {
            **_freshness(
                cmc.get("available_at"),
                maximum_age_hours=6.0,
                now=observed,
            ),
            "provider": cmc.get("provider") or "coinmarketcap",
            "confidence": 0.90,
        },
        "fear_and_greed": {
            **_freshness(
                fear.get("available_at"),
                maximum_age_hours=36.0,
                now=observed,
            ),
            "provider": fear.get("provider") or "alternative_me",
            "confidence": 0.65,
        },
        "coinmarketcap_rankings": {
            **_freshness(
                rankings.get("available_at"),
                maximum_age_hours=6.0,
                now=observed,
            ),
            "provider": rankings.get("provider") or "coinmarketcap",
            "confidence": 0.70,
        },
        "derivatives": {
            **_freshness(
                derivatives.get("available_at"),
                maximum_age_hours=2.0,
                now=observed,
            ),
            "provider": derivatives.get("provider") or "mexc",
            "confidence": 0.75,
        },
        "eth_derivatives": {
            **_freshness(
                eth_derivatives.get("available_at"),
                maximum_age_hours=2.0,
                now=observed,
            ),
            "provider": eth_derivatives.get("provider") or "mexc",
            "confidence": 0.70,
        },
        "vix": {
            **_freshness(
                vix.get("available_at"),
                maximum_age_hours=36.0,
                now=observed,
            ),
            "provider": vix.get("provider") or "eodhd",
            "confidence": 0.70,
        },
        "nasdaq_100": {
            **_freshness(
                ndx.get("available_at"),
                maximum_age_hours=36.0,
                now=observed,
            ),
            "provider": ndx.get("provider") or "eodhd",
            "confidence": 0.65,
        },
        "sp500": {
            **_freshness(
                spx.get("available_at"),
                maximum_age_hours=36.0,
                now=observed,
            ),
            "provider": spx.get("provider") or "eodhd",
            "confidence": 0.65,
        },
        "short_rate": {
            **_freshness(
                dff.get("available_at"),
                maximum_age_hours=240.0,
                now=observed,
            ),
            "provider": dff.get("provider") or "fred",
            "confidence": 0.75,
        },
        "long_rate": {
            **_freshness(
                dgs10.get("available_at"),
                maximum_age_hours=240.0,
                now=observed,
            ),
            "provider": dgs10.get("provider") or "fred",
            "confidence": 0.75,
        },
        "scraper_intelligence": {
            **_freshness(
                intelligence.get("available_at"),
                maximum_age_hours=(
                    settings.scrapers.stale_news_after.total_seconds() / 3_600.0
                ),
                now=observed,
            ),
            "provider": intelligence.get("provider") or "multi_source_scraper",
            "confidence": 0.45,
        },
        "stablecoin_liquidity": {
            **_freshness(
                stablecoin_liquidity.get("available_at"),
                maximum_age_hours=6.0,
                now=observed,
            ),
            "provider": stablecoin_liquidity.get("provider")
            or "coinmarketcap+defillama",
            "confidence": 0.85,
        },
    }
    frames = dict(market_frames or {})
    btc_1h = frames.get(("BTC-EUR", "1h"))
    btc_4h = frames.get(("BTC-EUR", "4h"))
    btc_1d = frames.get(("BTC-EUR", "1d"))

    def trend(frame: pd.DataFrame | None, span: int) -> bool | None:
        if frame is None or len(frame) < span:
            return None
        close = pd.to_numeric(frame["close"], errors="coerce")
        average = close.ewm(span=span, adjust=False).mean()
        return bool(close.iloc[-1] > average.iloc[-1])

    btc_4h_up = trend(btc_4h, 50)
    btc_1d_up = trend(btc_1d, 50)
    btc_24h = (
        float(btc_1h["close"].iloc[-1] / btc_1h["close"].iloc[-25] - 1.0)
        if btc_1h is not None and len(btc_1h) >= 25
        else None
    )
    btc_volatility = None
    volatility_percentile = None
    if btc_1h is not None and len(btc_1h) >= 120:
        returns = pd.to_numeric(btc_1h["close"], errors="coerce").pct_change()
        rolling = returns.rolling(24).std()
        btc_volatility = _finite(rolling.iloc[-1])
        history = rolling.dropna().iloc[-720:]
        if btc_volatility is not None and len(history):
            volatility_percentile = float((history <= btc_volatility).mean())
    breadth_states = []
    for (market, timeframe), frame in frames.items():
        if timeframe != "4h" or len(frame) < 50:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        breadth_states.append(bool(close.iloc[-1] > close.ewm(span=50, adjust=False).mean().iloc[-1]))
    breadth = float(np.mean(breadth_states)) if breadth_states else None
    cmc_values = dict(cmc.get("values") or {})
    fear_values = dict(fear.get("values") or {})
    derivative_values = dict(derivatives.get("values") or {})
    eth_derivative_values = dict(eth_derivatives.get("values") or {})
    vix_values = dict(vix.get("values") or {})
    ndx_values = dict(ndx.get("values") or {})
    spx_values = dict(spx.get("values") or {})
    intelligence_values = dict(intelligence.get("values") or {})
    btc_dominance = _finite(cmc_values.get("btc_dominance"))
    if btc_dominance is not None and btc_dominance > 1.0:
        btc_dominance /= 100.0
    fear_greed = _finite(fear_values.get("fear_greed"))
    funding = _finite(derivative_values.get("funding_rate"))
    eth_funding = _finite(eth_derivative_values.get("funding_rate"))
    vix_close = _finite(vix_values.get("close") or vix_values.get("adjusted_close"))
    ndx_close = _finite(ndx_values.get("close") or ndx_values.get("adjusted_close"))
    spx_close = _finite(spx_values.get("close") or spx_values.get("adjusted_close"))
    ndx_return_5d = _public_close_return(refreshed, "eodhd_ndx", periods=5)
    spx_return_5d = _public_close_return(refreshed, "eodhd_spx", periods=5)
    event_risk = _finite(intelligence_values.get("event_risk"), 0.0) or 0.0
    event_source_count = int(intelligence_values.get("source_count") or 0)
    stablecoin_state = str(
        stablecoin_liquidity.get("state") or "DATA_PENDING"
    )
    stablecoin_total_change_1d = _finite(
        (stablecoin_liquidity.get("aggregate") or {}).get("change_1d")
    )
    if btc_1h is None or btc_4h is None or btc_1d is None:
        regime = "DATA_BLOCKED"
    elif btc_24h is not None and btc_24h <= -0.06:
        regime = "DELEVERAGING"
    elif any(
        value is not None and abs(value) >= 0.001
        for value in (funding, eth_funding)
    ):
        regime = "LIQUIDATION_STRESS"
    elif event_risk >= 0.75 and event_source_count >= 2:
        regime = "MACRO_RISK_OFF"
    elif (
        stablecoin_state == "DRAINING"
        and stablecoin_total_change_1d is not None
        and stablecoin_total_change_1d <= -0.0025
        and (btc_1d_up is False or (breadth or 0.0) < 0.35)
    ):
        regime = "MACRO_RISK_OFF"
    elif (
        (vix_close is not None and vix_close >= 30.0)
        or (
            ndx_return_5d is not None
            and spx_return_5d is not None
            and ndx_return_5d <= -0.04
            and spx_return_5d <= -0.03
        )
        or (btc_1d_up is False and (breadth or 0.0) < 0.35)
    ):
        regime = "MACRO_RISK_OFF"
    elif btc_4h_up and not btc_1d_up:
        regime = "RECOVERY"
    elif (
        btc_4h_up
        and btc_1d_up
        and (breadth or 0.0) >= 0.65
        and (fear_greed or 50.0) >= 55.0
        and stablecoin_state != "DRAINING"
    ):
        regime = "STRONG_RISK_ON"
    elif (
        btc_4h_up
        and btc_1d_up
        and (breadth or 0.0) >= 0.50
        and stablecoin_state != "DRAINING"
    ):
        regime = "MODERATE_RISK_ON"
    elif (
        btc_4h_up
        and (breadth or 0.0) < 0.50
        and (btc_dominance or 0.0) >= 0.55
    ):
        regime = "BTC_LED_RISK_ON"
    elif (
        (breadth or 0.0) >= 0.60
        and (btc_dominance or 1.0) < 0.58
        and stablecoin_state != "DRAINING"
    ):
        regime = "ALTCOIN_ROTATION"
    elif volatility_percentile is not None and volatility_percentile >= 0.80:
        regime = "VOLATILITY_EXPANSION"
    elif volatility_percentile is not None and volatility_percentile <= 0.30:
        regime = "RANGE_LOW_VOL"
    elif volatility_percentile is not None:
        regime = "RANGE_HIGH_VOL"
    else:
        regime = "UNCERTAIN"
    fresh_weight = sum(
        float(row["confidence"])
        for row in sources.values()
        if row["fresh"]
    )
    total_weight = sum(float(row["confidence"]) for row in sources.values())
    confidence = fresh_weight / total_weight if total_weight else 0.0
    payload = {
        "schema_version": "crypto_macro_snapshot_v1",
        "observed_at": observed.isoformat(),
        "available_at": observed.isoformat(),
        "regime": regime,
        "confidence": confidence,
        "features": {
            "btc_4h_trend_up": btc_4h_up,
            "btc_1d_trend_up": btc_1d_up,
            "btc_return_24h": btc_24h,
            "btc_volatility_24h": btc_volatility,
            "btc_volatility_percentile": volatility_percentile,
            "altcoin_breadth": breadth,
            "btc_dominance": btc_dominance,
            "eth_dominance": _finite(cmc_values.get("eth_dominance")),
            "coinmarketcap_rank": _finite((rankings.get("values") or {}).get("rank")),
            "total_crypto_market_cap": _finite(
                cmc_values.get("total_market_cap")
            ),
            "fear_greed": fear_greed,
            "fear_greed_classification": fear_values.get("classification"),
            "aggregate_funding_proxy": funding,
            "eth_funding_proxy": eth_funding,
            "open_interest_proxy": _finite(
                derivative_values.get("open_interest")
            ),
            "eth_open_interest_proxy": _finite(
                eth_derivative_values.get("open_interest")
            ),
            "perpetual_premium": _finite(
                derivative_values.get("perpetual_premium")
            ),
            "vix": vix_close,
            "nasdaq_100": ndx_close,
            "sp500": spx_close,
            "nasdaq_100_return_5d": ndx_return_5d,
            "sp500_return_5d": spx_return_5d,
            "short_rate": _finite((dff.get("values") or {}).get("value")),
            "long_rate": _finite((dgs10.get("values") or {}).get("value")),
            "scraper_event_count": int(intelligence_values.get("event_count") or 0),
            "scraper_source_count": event_source_count,
            "scraper_event_risk": event_risk,
            "scraper_sentiment": _finite(intelligence_values.get("sentiment")),
            "stablecoin_liquidity_state": stablecoin_state,
            "stablecoin_liquidity_score": _finite(
                stablecoin_liquidity.get("score")
            ),
            "stablecoin_liquidity_risk_multiplier": _finite(
                stablecoin_liquidity.get("risk_multiplier")
            ),
            "usdt_market_cap_eur": _finite(
                (stablecoin_liquidity.get("usdt") or {}).get("market_cap_eur")
            ),
            "usdt_market_cap_change_1h": _finite(
                (stablecoin_liquidity.get("usdt") or {}).get(
                    "market_cap_change_1h"
                )
            ),
            "usdt_market_cap_change_24h": _finite(
                (stablecoin_liquidity.get("usdt") or {}).get(
                    "market_cap_change_24h"
                )
            ),
            "usdc_market_cap_eur": _finite(
                (stablecoin_liquidity.get("usdc") or {}).get("market_cap_eur")
            ),
            "usdc_market_cap_change_1h": _finite(
                (stablecoin_liquidity.get("usdc") or {}).get(
                    "market_cap_change_1h"
                )
            ),
            "usdc_market_cap_change_24h": _finite(
                (stablecoin_liquidity.get("usdc") or {}).get(
                    "market_cap_change_24h"
                )
            ),
            "stablecoin_total_market_cap_usd": _finite(
                (stablecoin_liquidity.get("aggregate") or {}).get(
                    "total_market_cap_usd"
                )
            ),
            "stablecoin_total_change_1d": stablecoin_total_change_1d,
            "stablecoin_total_change_7d": _finite(
                (stablecoin_liquidity.get("aggregate") or {}).get("change_7d")
            ),
        },
        "stablecoin_liquidity": stablecoin_liquidity,
        "sources": sources,
        "provider_refresh": {
            name: str((details or {}).get("status") or "UNKNOWN")
            for name, details in ((refreshed or {}).get("sources") or {}).items()
        },
        "scraper_refresh": dict((refreshed or {}).get("intelligence") or {}),
        "missing_data_lowers_confidence": True,
        "missing_data_blocks_backfill": True,
        "macro_is_entry_signal": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return payload


def _regime_policy(
    regime: str,
    family: str,
    market: str,
) -> tuple[str, float, str]:
    """Compatibility wrapper around the shared live regime policy."""

    return regime_policy(regime, family, market)


def _alignment(
    features: pd.DataFrame,
    spec: TacticalStrategySpec,
) -> tuple[float, list[str]]:
    latest = features.iloc[-1]
    conflicts: list[str] = []
    values: list[bool] = []
    for timeframe in (
        spec.confirmation_timeframe,
        spec.regime_timeframe,
    ):
        key = f"htf_{timeframe}_trend_bullish"
        value = bool(latest.get(key, False))
        values.append(value)
        if not value:
            conflicts.append(f"{timeframe}_TREND_NOT_BULLISH")
    return float(np.mean(values)) if values else 0.0, conflicts


def _trigger(
    features: pd.DataFrame,
    spec: TacticalStrategySpec,
) -> tuple[float, str]:
    close = pd.to_numeric(features["close"], errors="coerce")
    mechanism = spec.mechanism
    if mechanism in {"donchian_breakout", "compression_breakout"}:
        return float(features["donchian_high_20"].iloc[-1]), "CLOSE_ABOVE_DONCHIAN"
    if mechanism == "donchian_atr_fractal":
        return float(features["high"].shift(1).rolling(120).max().iloc[-1]), "CLOSE_ABOVE_120_BAR_HIGH"
    if mechanism == "fractal_breakout":
        level = pd.to_numeric(
            features["confirmed_fractal_high_price"],
            errors="coerce",
        ).ffill()
        return float(level.iloc[-1]), "CLOSE_ABOVE_CONFIRMED_FRACTAL"
    if mechanism in {"trend_pullback", "mtf_trend_continuation"}:
        return float(features["ema_20"].iloc[-1]), "EMA20_RECLAIM"
    if mechanism == "vwap_reclaim":
        return float(features["vwap_20"].iloc[-1]), "VWAP_RECLAIM"
    if mechanism in {"relative_strength", "cross_sectional_momentum"}:
        return float(close.iloc[-1]), "BTC_RELATIVE_MOMENTUM_CROSS"
    if mechanism == "range_reversion":
        return float(features["bollinger_lower"].iloc[-1]), "RSI_RECOVERY_FROM_LOWER_BAND"
    if mechanism in {"liquidity_sweep", "failed_breakout"}:
        return float(close.iloc[-1]), "CONFIRMED_SWEEP_RECLAIM"
    if mechanism == "post_liquidation_recovery":
        return float(features["high"].shift(1).iloc[-1]), "RECOVER_PRIOR_SHOCK_HIGH"
    if mechanism == "momentum_acceleration":
        return float(close.iloc[-1]), "MOMENTUM_ACCELERATION_POSITIVE"
    if mechanism in {"volume_expansion", "volatility_expansion"}:
        return float(features["high"].shift(1).iloc[-1]), "RANGE_AND_VOLUME_EXPANSION"
    if mechanism == "breakout_retest":
        return float(features["donchian_high_20"].iloc[-1]), "BREAKOUT_LEVEL_RETEST"
    if mechanism == "structure_continuation":
        return float(close.iloc[-1]), "BOS_OR_CHOCH_CONFIRMATION"
    if mechanism == "range_breakout":
        return float(features["high"].shift(1).rolling(24).max().iloc[-1]), "CLOSE_ABOVE_RANGE"
    if mechanism == "defensive_recovery":
        return float(features["ema_20"].iloc[-1]), "RSI_AND_EMA_RECOVERY"
    return float(close.iloc[-1]), "MECHANISM_CONFIRMATION"


def _readiness_score(
    features: pd.DataFrame,
    *,
    trigger: float | None,
    alignment: float,
) -> float:
    latest = features.iloc[-1]
    close = float(latest["close"])
    distance = (
        abs(trigger / close - 1.0)
        if close > 0 and trigger is not None and trigger > 0
        else 1.0
    )
    proximity = max(0.0, 1.0 - distance / 0.04)
    relative_volume = min(
        1.0,
        max(0.0, _finite(latest.get("relative_volume_20"), 0.0) or 0.0) / 1.25,
    )
    trend = float(
        close > (_finite(latest.get("ema_50"), close) or close)
    )
    return 100.0 * (
        0.40 * proximity
        + 0.25 * alignment
        + 0.20 * relative_volume
        + 0.15 * trend
    )


def _load_scan_frames(
    settings: Settings,
    markets: Sequence[str],
    *,
    now: datetime,
    maximum_rows: int = 3_000,
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], str],
]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    failures: dict[tuple[str, str], str] = {}
    for market in markets:
        for timeframe in OBSERVED_TIMEFRAMES:
            try:
                frames[(market, timeframe)] = _normalized_frame(
                    settings,
                    market,
                    timeframe,
                    maximum_rows=maximum_rows,
                    now=now,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                failures[(market, timeframe)] = str(exc) or type(exc).__name__
        # The continuous live data service prioritises native 15m/1h/4h/1d
        # updates. Derive 2h and 1W only from those fresh, closed source bars
        # when their cached target is missing or stale. This keeps the tactical
        # chain current without fabricating candles or forward filling gaps.
        for target, source in (("2h", "1h"), ("1W", "1d")):
            if (market, target) in frames:
                continue
            source_frame = frames.get((market, source))
            if source_frame is None:
                continue
            try:
                derived = resample_ohlcv(
                    source_frame,
                    source_timeframe=source,
                    target_timeframe=target,
                    drop_incomplete=True,
                )
                derived = drop_open_candles(
                    derived,
                    timeframe=target,
                    now=now,
                )
                _assert_frame_fresh(derived, timeframe=target, now=now)
                if len(derived) > maximum_rows:
                    derived = derived.iloc[-maximum_rows:].copy()
                derived.attrs.update(
                    {
                        "market": market,
                        "timeframe": target,
                        "data_provenance": {
                            "source_type": "CAUSAL_CLOSED_CANDLE_RESAMPLE",
                            "source_timeframe": source,
                        },
                    }
                )
                frames[(market, target)] = derived
                failures.pop((market, target), None)
            except (OSError, TypeError, ValueError) as exc:
                failures[(market, target)] = str(exc) or type(exc).__name__
    return frames, failures


def _feature_frames(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
) -> tuple[
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], str],
]:
    result: dict[tuple[str, str], pd.DataFrame] = {}
    failures: dict[tuple[str, str], str] = {}
    pipeline = FeaturePipeline()
    for timeframe in TACTICAL_TIMEFRAMES:
        benchmark = frames.get(("BTC-EUR", timeframe))
        for market in markets:
            source = frames.get((market, timeframe))
            if source is None or benchmark is None:
                failures[(market, timeframe)] = "EXECUTION_OR_BENCHMARK_MISSING"
                continue
            higher = {
                selected: frames[(market, selected)]
                for selected in ("1h", "2h", "4h", "1d")
                if (market, selected) in frames
                and TIMEFRAME_SECONDS[selected] > TIMEFRAME_SECONDS[timeframe]
            }
            try:
                result[(market, timeframe)] = pipeline.build(
                    source,
                    market=market,
                    benchmark=benchmark,
                    higher_timeframes=higher,
                )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                failures[(market, timeframe)] = type(exc).__name__
    return result, failures


def build_rotation_ranking(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    *,
    regime: str,
) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, float]] = {}
    for market in markets:
        values: dict[str, float] = {}
        for timeframe, periods in (
            ("15m", 4),
            ("1h", 24),
            ("2h", 12),
            ("4h", 12),
            ("1d", 7),
        ):
            frame = frames.get((market, timeframe))
            if frame is not None and len(frame) > periods:
                close = pd.to_numeric(frame["close"], errors="coerce")
                values[f"return_{timeframe}"] = float(
                    close.iloc[-1] / close.iloc[-periods - 1] - 1.0
                )
        if values:
            raw[market] = values
    if not raw:
        return []
    btc = raw.get("BTC-EUR", {})
    score_parts: dict[str, float] = {}
    for market, values in raw.items():
        momentum = sum(
            values.get(field, 0.0) * weight
            for field, weight in (
                ("return_1h", 0.20),
                ("return_2h", 0.20),
                ("return_4h", 0.25),
                ("return_1d", 0.35),
            )
        )
        relative = sum(
            (values.get(field, 0.0) - btc.get(field, 0.0)) * weight
            for field, weight in (
                ("return_1h", 0.20),
                ("return_2h", 0.20),
                ("return_4h", 0.25),
                ("return_1d", 0.35),
            )
        )
        frame = frames.get((market, "4h"))
        trend_quality = 0.0
        volatility_penalty = 0.0
        if frame is not None and len(frame) >= 50:
            close = pd.to_numeric(frame["close"], errors="coerce")
            trend_quality = float(
                close.iloc[-1] > close.ewm(span=50, adjust=False).mean().iloc[-1]
            )
            volatility_penalty = min(
                1.0,
                float(close.pct_change().iloc[-30:].std()) / 0.08,
            )
        score_parts[market] = (
            35.0 * relative
            + 25.0 * momentum
            + 0.30 * trend_quality
            - 0.10 * volatility_penalty
        )
    ordered = sorted(score_parts.items(), key=lambda item: item[1])
    denominator = max(1, len(ordered) - 1)
    percentiles = {
        market: 100.0 * index / denominator
        for index, (market, _) in enumerate(ordered)
    }
    rows = [
        {
            "market": market,
            "rank": 0,
            "rotation_score": percentiles[market],
            "relative_strength_score": score_parts[market],
            "returns": raw[market],
            "regime": regime,
            "decision": (
                "FAVOUR"
                if percentiles[market] >= 70.0
                else "NEUTRAL"
                if percentiles[market] >= 35.0
                else "UNDERWEIGHT"
            ),
        }
        for market in score_parts
    ]
    rows.sort(key=lambda row: row["rotation_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _scan_tactical_opportunities(
    settings: Settings,
    features: Mapping[tuple[str, str], pd.DataFrame],
    frames: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    *,
    macro: Mapping[str, Any],
    rotation: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rotation_by_market = {
        str(row["market"]): dict(row) for row in rotation
    }
    rows: list[dict[str, Any]] = []
    evaluated_by_timeframe = {
        timeframe: 0 for timeframe in TACTICAL_TIMEFRAMES
    }
    for spec in tactical_strategy_specs():
        strategy = TacticalMultiTimeframeStrategy(spec)
        for market in markets:
            frame = features.get((market, spec.timeframe))
            if frame is None or frame.empty:
                continue
            evaluated_by_timeframe[spec.timeframe] += 1
            output = strategy.generate(frame)
            latest = frame.iloc[-1]
            close = float(latest["close"])
            raw_trigger, trigger_reason = _trigger(frame, spec)
            trigger = _finite(raw_trigger)
            legacy_alignment, _ = _alignment(frame, spec)
            timeframe_assessment = _weighted_timeframe_assessment(
                frames,
                market,
            )
            directional_score = float(timeframe_assessment["score"])
            alignment = max(0.0, min(1.0, (directional_score + 1.0) / 2.0))
            conflicts = list(timeframe_assessment["conflicts"])
            policy, risk_multiplier, policy_reason = _regime_policy(
                str(macro["regime"]),
                spec.family,
                market,
            )
            family_mode_fit = _family_mode_fit(
                spec.family,
                str(timeframe_assessment["market_mode"]),
            )
            execution_risk_multiplier = (
                risk_multiplier
                * float(timeframe_assessment["risk_multiplier"])
                * family_mode_fit
            )
            readiness = _readiness_score(
                frame,
                trigger=trigger,
                alignment=alignment,
            )
            rotation_score = float(
                rotation_by_market.get(market, {}).get(
                    "rotation_score",
                    50.0,
                )
            )
            score = (
                0.55 * readiness
                + 0.25 * rotation_score
                + 0.20 * 100.0 * alignment
            ) * family_mode_fit
            entry_trigger_confirmed = bool(output.entry.iloc[-1])
            directional_gate = _strategy_directional_gate(
                spec.family,
                timeframe_assessment,
            )
            entry_threshold = float(
                directional_gate["effective_threshold"]
            )
            actionable = (
                entry_trigger_confirmed
                and trigger is not None
                and policy in {"ENABLE", "REDUCE"}
                and directional_gate["approved"] is True
            )
            family_authority = _family_canary_authority(
                settings,
                family=spec.family,
                market=market,
                entry_timeframe=spec.timeframe,
            )
            distance = (
                (trigger / close - 1.0)
                if close > 0.0 and trigger is not None and trigger > 0.0
                else None
            )
            if policy in {"BLOCK", "SHADOW_ONLY"}:
                status = "AVOID"
            elif actionable:
                status = "ACTIONABLE"
            elif score >= 70.0 or (
                distance is not None and abs(distance) <= 0.0075
            ):
                status = "NEAR_ENTRY"
            elif score >= 50.0 or (
                distance is not None and abs(distance) <= 0.02
            ):
                status = "APPROACHING"
            else:
                status = "WATCH"
            atr = _finite(latest.get("atr_14"), 0.0) or 0.0
            stop_distance = atr * spec.stop_atr
            target_1_distance = atr * min(spec.target_atr, spec.stop_atr * 1.5)
            target_2_distance = atr * spec.target_atr
            # The closed candle establishes the causal strategy thesis.  A
            # separate realtime trigger may subsequently time execution.  Do
            # not make a still-open candle part of the strategy truth, but do
            # keep a qualified near-entry warm instead of requiring the
            # closed candle itself to contain the final fill trigger.
            setup_valid_on_closed_candle = bool(
                status in {"ACTIONABLE", "NEAR_ENTRY"}
                and policy in {"ENABLE", "REDUCE"}
                and directional_gate["approved"] is True
                and trigger is not None
                and atr > 0.0
                and stop_distance > 0.0
                and target_2_distance > 0.0
            )
            signal_timestamp = frame.index[-1].isoformat()
            opportunity_id = stable_hash(
                [
                    spec.dna_hash,
                    market,
                    signal_timestamp,
                    status,
                    (
                        round(trigger, 10)
                        if trigger is not None
                        else "NO_VALID_TRIGGER"
                    ),
                ],
                length=32,
            )
            reason_not_entered = None
            if not actionable:
                if trigger is None:
                    reason_not_entered = "INVALID_OR_MISSING_CAUSAL_TRIGGER"
                elif policy in {"BLOCK", "SHADOW_ONLY"}:
                    reason_not_entered = policy_reason
                elif not entry_trigger_confirmed:
                    reason_not_entered = (
                        "ENTRY_TRIGGER_NOT_CONFIRMED_ON_LATEST_CLOSED_CANDLE"
                    )
                else:
                    reason_not_entered = (
                        "WEIGHTED_TIMEFRAME_SCORE_BELOW_ENTRY_THRESHOLD"
                    )
            next_required_condition = (
                (
                    "REALTIME_FAMILY_ENVELOPE_CONFIRMATION"
                    if family_authority["available"]
                    else "STRATEGY_AUTHORITY"
                )
                if actionable
                else reason_not_entered
                or "ENTRY_TRIGGER_NOT_CONFIRMED_ON_LATEST_CLOSED_CANDLE"
            )
            rows.append(
                {
                    "opportunity_id": opportunity_id,
                    "rank": 0,
                    "market": market,
                    "strategy": spec.strategy_id,
                    "strategy_dna_hash": spec.dna_hash,
                    "family": spec.family,
                    "timeframe": spec.timeframe,
                    "entry_timeframe": spec.timeframe,
                    "confirmation_timeframe": spec.confirmation_timeframe,
                    "regime_timeframe": spec.regime_timeframe,
                    "regime": macro["regime"],
                    "regime_policy": policy,
                    "regime_risk_multiplier": execution_risk_multiplier,
                    "macro_risk_multiplier": risk_multiplier,
                    "timeframe_risk_multiplier": timeframe_assessment[
                        "risk_multiplier"
                    ],
                    "family_mode_fit": family_mode_fit,
                    "trade_type": timeframe_assessment["trade_type"],
                    "market_mode": timeframe_assessment["market_mode"],
                    "status": status,
                    "entry_trigger_confirmed": entry_trigger_confirmed,
                    "setup_valid_on_closed_candle": (
                        setup_valid_on_closed_candle
                    ),
                    "score": score,
                    "confidence": min(
                        100.0,
                        0.65 * readiness
                        + 35.0 * float(macro.get("confidence") or 0.0),
                    ),
                    "current_price": close,
                    "entry_zone": (
                        [trigger * 0.998, trigger * 1.002]
                        if trigger is not None
                        else [None, None]
                    ),
                    "entry_atr": atr,
                    "trigger": trigger,
                    "trigger_data_valid": trigger is not None,
                    "trigger_reason": trigger_reason,
                    "stop": close - stop_distance,
                    "target_1": close + target_1_distance,
                    "target_2": close + target_2_distance,
                    "expected_holding_period": spec.expected_holding_period,
                    "estimated_fee_fraction": 0.0025,
                    "estimated_slippage_bps": 8.0,
                    "liquidity_status": "PRECHECK_REQUIRED_AT_ORDER_TIME",
                    "spread_status": "PRECHECK_REQUIRED_AT_ORDER_TIME",
                    "timeframe_alignment_score": alignment,
                    "legacy_binary_alignment_score": legacy_alignment,
                    "weighted_timeframe_score": directional_score,
                    "raw_weighted_timeframe_score": timeframe_assessment[
                        "raw_composite_score"
                    ],
                    "fast_timeframe_score": timeframe_assessment["fast_score"],
                    "slow_timeframe_score": timeframe_assessment["slow_score"],
                    "timeframe_disagreement": timeframe_assessment[
                        "disagreement"
                    ],
                    "timeframe_disagreement_penalty": timeframe_assessment[
                        "disagreement_penalty"
                    ],
                    "weighted_entry_threshold": entry_threshold,
                    "base_weighted_entry_threshold": directional_gate[
                        "base_threshold"
                    ],
                    "fast_timeframe_score_floor": directional_gate[
                        "fast_score_floor"
                    ],
                    "strategy_directional_gate": directional_gate,
                    "timeframe_score_details": timeframe_assessment[
                        "per_timeframe"
                    ],
                    "timeframe_score_weights": timeframe_assessment["weights"],
                    "missing_timeframe_scores": timeframe_assessment[
                        "missing_timeframes"
                    ],
                    "hard_blocked_by_1d_or_1w": False,
                    "timeframe_conflicts": conflicts,
                    "higher_timeframe_parent_valid": bool(
                        (
                            spec.timeframe == "15m"
                            and spec.confirmation_timeframe
                            in {"1h", "2h", "4h"}
                            and spec.regime_timeframe in {"4h", "1d", "1W"}
                        )
                        or (
                            spec.timeframe == "1h"
                            and spec.confirmation_timeframe in {"4h", "1d"}
                            and spec.regime_timeframe in {"1d", "1W"}
                        )
                    ),
                    "parent_setup_timeframes": [
                        spec.confirmation_timeframe,
                        spec.regime_timeframe,
                    ],
                    "macro_support": policy,
                    "crypto_context_support": macro["regime"],
                    "rotation_score": rotation_score,
                    "token_risk_status": "USE_LIVE_UNIVERSE_PREFLIGHT",
                    "reason_not_yet_entered": reason_not_entered,
                    "next_required_condition": next_required_condition,
                    "near_entry": status == "NEAR_ENTRY",
                    "gate_matrix": {
                        "SETUP": (
                            "PASS"
                            if setup_valid_on_closed_candle
                            else "WAIT"
                        ),
                        "CLOSED_CANDLE_ENTRY_TRIGGER": (
                            "PASS" if entry_trigger_confirmed else "WAIT"
                        ),
                        "TRIGGER_DATA": (
                            "PASS" if trigger is not None else "FAIL"
                        ),
                        "MTF": (
                            "PASS"
                            if directional_gate["approved"] is True
                            else "FAIL"
                        ),
                        "STOP": "PASS" if stop_distance > 0.0 else "FAIL",
                        "TARGET": (
                            "PASS" if target_2_distance > 0.0 else "FAIL"
                        ),
                        "MACRO": (
                            "PASS"
                            if policy in {"ENABLE", "REDUCE"}
                            else "FAIL"
                        ),
                        "EXACT_DNA_AUTHORITY": "FAIL",
                        "FAMILY_ENVELOPE_AUTHORITY": (
                            "PASS"
                            if family_authority["available"]
                            else "FAIL"
                        ),
                        "STRATEGY_AUTHORITY": (
                            "ROUTED_TO_FAMILY_ENVELOPE"
                            if family_authority["available"]
                            else "FAIL"
                        ),
                    },
                    "distance_to_trigger": distance,
                    "signal_timestamp": signal_timestamp,
                    "closed_candle_only": True,
                    "next_open_execution_required": True,
                    "live_authority_granted": False,
                    "family_canary_authority_available": family_authority[
                        "available"
                    ],
                    "authorized_event_playbook_ids": family_authority[
                        "playbook_ids"
                    ],
                    "family_authority_strategy_roles": family_authority[
                        "strategy_roles"
                    ],
                    "family_authority_maximum_effective_order_eur": (
                        family_authority["maximum_effective_order_eur"]
                    ),
                    "family_authority_requires_realtime_confirmation": (
                        family_authority["requires_realtime_confirmation"]
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            row["status"] == "ACTIONABLE",
            row["status"] == "NEAR_ENTRY",
            row["score"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, evaluated_by_timeframe


def _scan_market_mechanics_opportunities(
    features: Mapping[tuple[str, str], pd.DataFrame],
    frames: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    mechanics: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Evaluate 15m/1h entries with forward-only 2h/4h and flow context."""

    rows: list[dict[str, Any]] = []
    evaluations = 0
    mechanics_by_market = dict(mechanics.get("markets") or {})
    for spec in market_mechanics_strategy_specs():
        for market in markets:
            fifteen_minute = features.get((market, "15m"))
            one_hour = features.get((market, "1h"))
            two_hour = features.get((market, "2h"))
            four_hour = frames.get((market, "4h"))
            if (
                one_hour is None
                or two_hour is None
                or four_hour is None
                or one_hour.empty
                or two_hour.empty
                or four_hour.empty
            ):
                continue
            evaluations += 1
            result = evaluate_market_mechanics_strategy(
                spec,
                market=market,
                one_hour=one_hour,
                two_hour=two_hour,
                four_hour=four_hour,
                fifteen_minute=fifteen_minute,
                mechanics=dict(mechanics_by_market.get(market) or {}),
            )
            entry_timeframe = str(result.get("entry_timeframe") or "1h")
            entry_frame = (
                fifteen_minute
                if entry_timeframe == "15m" and fifteen_minute is not None
                else one_hour
            )
            close = float(result.get("entry") or entry_frame["close"].iloc[-1])
            status = str(result.get("status") or "DATA_PENDING")
            score = 100.0 * float(result.get("score") or 0.0)
            blockers = [str(value) for value in result.get("blockers") or []]
            signal_timestamp = entry_frame.index[-1].isoformat()
            rows.append(
                {
                    "opportunity_id": stable_hash(
                        [spec.dna_hash, market, signal_timestamp, status],
                        length=32,
                    ),
                    "rank": 0,
                    "market": market,
                    "strategy": spec.strategy_id,
                    "strategy_dna_hash": spec.dna_hash,
                    "family": spec.family,
                    "timeframe": entry_timeframe,
                    "entry_timeframe": entry_timeframe,
                    "setup_timeframe": "2h",
                    "confirmation_timeframe": "2h",
                    "regime_timeframe": "4h",
                    "status": status,
                    "entry_trigger_confirmed": status == "ACTIONABLE",
                    "score": score,
                    "confidence": score,
                    "current_price": close,
                    "entry_zone": [close * 0.998, close * 1.002],
                    "trigger": close,
                    "trigger_reason": (
                        f"{entry_timeframe.upper()}_CLOSE_PLUS_"
                        "PROSPECTIVE_ORDERFLOW"
                    ),
                    "stop": result.get("stop"),
                    "target_1": result.get("target_1"),
                    "target_2": result.get("target_2"),
                    "expected_holding_period": result.get(
                        "expected_holding_period"
                    ),
                    "estimated_fee_fraction": 0.0025,
                    "estimated_slippage_bps": 8.0,
                    "liquidity_status": (
                        "ORDERFLOW_CONFIRMED"
                        if status == "ACTIONABLE"
                        else "PROSPECTIVE_ORDERFLOW_REQUIRED"
                    ),
                    "spread_status": (
                        "CONFIRMED"
                        if status == "ACTIONABLE"
                        else "FLOW_DATA_PENDING_OR_UNCONFIRMED"
                    ),
                    "timeframe_alignment_score": result.get(
                        "score_components", {}
                    ).get("trend_4h"),
                    "timeframe_conflicts": blockers,
                    "gex_scope": result.get("gex_scope"),
                    "gex_regime": result.get("gex_regime"),
                    "score_components": result.get("score_components", {}),
                    "reason_not_yet_entered": (
                        None if status == "ACTIONABLE" else ",".join(blockers)
                    ),
                    "distance_to_trigger": 0.0,
                    "signal_timestamp": signal_timestamp,
                    "closed_candle_only": True,
                    "next_open_execution_required": True,
                    "historical_backtest_status": result.get(
                        "historical_backtest_status"
                    ),
                    "execution_scope": (
                        "MARKET_MECHANICS_DNA_REQUIRES_OPERATOR_APPROVAL"
                    ),
                    "live_authority_granted": False,
                    "operator_dna_approval_required": True,
                }
            )
    return rows, evaluations


def _execution_funnel(
    *,
    opportunities: Sequence[Mapping[str, Any]],
    frame_failures: Mapping[tuple[str, str], str],
    feature_failures: Mapping[tuple[str, str], str],
    execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Explain every stage between a detected signal and a live fill.

    Counts are deliberately observational.  They never grant execution
    authority or turn a near-entry/watchlist row into a signal.
    """

    rows = [dict(row) for row in opportunities]
    confirmed = [
        row for row in rows if row.get("entry_trigger_confirmed") is True
    ]
    macro_rejected = [
        row
        for row in confirmed
        if str(row.get("regime_policy") or row.get("macro_support") or "")
        in {"BLOCK", "SHADOW_ONLY"}
    ]
    actionable = [row for row in rows if row.get("status") == "ACTIONABLE"]
    authority_rejected = [
        row
        for row in actionable
        if row.get("live_authority_granted") is not True
        and row.get("family_canary_authority_available") is not True
    ]
    data_pending = [row for row in rows if row.get("status") == "DATA_PENDING"]
    all_data_failures = [
        str(value)
        for value in (*frame_failures.values(), *feature_failures.values())
    ]
    freshness_rejected = sum(
        "STALE" in value.upper() for value in all_data_failures
    )
    execution_state = dict(execution or {})
    execution_status = str(execution_state.get("status") or "NOT_DELEGATED")
    ranked = list(execution_state.get("ranked_natural_entries") or [])
    live_fills_this_cycle = int(
        execution_state.get("fills_verified_this_cycle") or 0
    )
    setups = [
        row
        for row in rows
        if str(row.get("status") or "")
        in {
            "ACTIONABLE",
            "NEAR_ENTRY",
            "APPROACHING",
            "PULLBACK_PENDING",
            "EARLY_MOMENTUM_ALERT",
        }
    ]
    mtf_qualified = [
        row
        for row in setups
        if (
            (row.get("strategy_directional_gate") or {}).get("approved")
            is True
            or (row.get("gate_matrix") or {}).get("MTF") == "PASS"
        )
    ]
    near_entries = [
        row
        for row in rows
        if row.get("near_entry") is True
        or str(row.get("status") or "") == "NEAR_ENTRY"
    ]
    economics_evaluated = [
        row for row in rows if row.get("execution_economics") is not None
    ]
    economics_pass = [
        row
        for row in economics_evaluated
        if (row.get("execution_economics") or {}).get(
            "positive_after_costs"
        )
        is True
    ]
    microstructure_evaluated = [
        row for row in rows if row.get("microstructure_state") is not None
    ]
    microstructure_pass = [
        row
        for row in microstructure_evaluated
        if row.get("microstructure_state") in {"NEUTRAL", "SUPPORTIVE"}
    ]
    authority_pass = [
        row
        for row in rows
        if row.get("live_authority_granted") is True
        or row.get("family_canary_authority_available") is True
    ]
    family_authority_routed = [
        row
        for row in rows
        if row.get("family_canary_authority_available") is True
        and row.get("live_authority_granted") is not True
    ]
    blocker_counts: dict[str, int] = {}
    for row in rows:
        reasons = list(row.get("hard_blockers") or [])
        if not reasons and row.get("next_required_condition"):
            reasons = [str(row["next_required_condition"])]
        if not reasons and row.get("reason_not_yet_entered"):
            reasons = [
                value.strip()
                for value in str(row["reason_not_yet_entered"]).split(",")
                if value.strip()
            ]
        for reason in dict.fromkeys(reasons):
            blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
    dominant_blocker = (
        max(blocker_counts.items(), key=lambda item: item[1])[0]
        if blocker_counts
        else None
    )
    stage_counts = {
        "evaluations": len(rows),
        "raw_signals": len(confirmed),
        "strategy_setups": len(setups),
        "mtf_qualified": len(mtf_qualified),
        "near_entry": len(near_entries),
        "economic_evaluated": len(economics_evaluated),
        "economic_pass": len(economics_pass),
        "microstructure_evaluated": len(microstructure_evaluated),
        "microstructure_pass": len(microstructure_pass),
        "authority_pass": len(authority_pass),
        "family_authority_routed": len(family_authority_routed),
        "entry_ready": len(ranked),
        "orders_submitted": int(
            execution_state.get("orders_submitted_this_cycle") or 0
        ),
        "fills": live_fills_this_cycle,
    }

    def conversion(numerator: str, denominator: str) -> float | None:
        base = stage_counts[denominator]
        return stage_counts[numerator] / base if base else None

    return {
        "schema_version": "active_execution_funnel_v1",
        "strategy_market_evaluations": len(rows),
        "entry_triggers_confirmed_before_overlays": len(confirmed),
        "rejected_by_freshness": freshness_rejected,
        "rejected_by_macro_regime": len(macro_rejected),
        "rejected_by_live_authority": len(authority_rejected),
        "routed_to_family_canary_authority": len(family_authority_routed),
        "rejected_by_liquidity": int(execution_status == "LIQUIDITY_BLOCKED"),
        "rejected_by_risk_sizing": int(
            execution_status
            in {"RISK_BLOCKED", "PORTFOLIO_CAP_BLOCKED", "WEEKLY_BUDGET_BLOCKED"}
        ),
        "rejected_by_missing_data": len(data_pending)
        + len(all_data_failures),
        "execution_ready": len(ranked),
        "live_orders_submitted": int(
            execution_state.get("orders_submitted_this_cycle") or 0
        ),
        "live_orders_cancelled": int(
            execution_state.get("orders_cancelled_this_cycle") or 0
        ),
        "live_orders_repriced": int(
            execution_state.get("orders_repriced_this_cycle") or 0
        ),
        "live_fills_verified": live_fills_this_cycle,
        "execution_status": execution_status,
        "execution_reason": execution_state.get("last_reason"),
        "stage_counts": stage_counts,
        "conversion_ratios": {
            "setup_per_evaluation": conversion(
                "strategy_setups", "evaluations"
            ),
            "near_entry_per_setup": conversion(
                "near_entry", "strategy_setups"
            ),
            "entry_ready_per_near_entry_observation_ratio": conversion(
                "entry_ready", "near_entry"
            ),
            "fill_per_entry_ready": conversion("fills", "entry_ready"),
        },
        "conversion_metric_semantics": {
            "populations_nested": False,
            "operational_decision_eligible": False,
            "reason": (
                "stage observations are counted independently per cycle and "
                "must not be interpreted as nested conversion probabilities"
            ),
        },
        "blocker_counts": dict(
            sorted(
                blocker_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "dominant_blocker": dominant_blocker,
        "macro_rejected_examples": [
            {
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "timeframe": row.get("timeframe"),
                "reason": row.get("reason_not_yet_entered"),
            }
            for row in macro_rejected[:10]
        ],
        "authority_rejected_examples": [
            {
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "strategy_dna_hash": row.get("strategy_dna_hash"),
                "timeframe": row.get("timeframe"),
            }
            for row in authority_rejected[:10]
        ],
    }


def build_proactive_allocation_plan(
    settings: Settings,
    *,
    rotation: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    regime: str,
    live_markets: set[str],
) -> dict[str, Any]:
    """Publish a bounded allocation plan without creating execution authority.

    Existing wallet inventory is marked separately from strategy-owned exposure.
    A target weight is actionable only after a frozen, approved DNA emits a
    natural signal; rankings alone never create an order.
    """

    utilization = build_capital_utilization(settings)
    equity = _finite(utilization.get("account_equity_eur"), 0.0) or 0.0
    cash = _finite(utilization.get("eur_cash"), 0.0) or 0.0
    current_values: dict[str, float] = {"EUR": max(0.0, cash)}
    for row in utilization.get("material_inventory") or []:
        market = str(row.get("market") or "")
        if market:
            current_values[market] = max(
                0.0,
                _finite(row.get("estimated_value_eur"), 0.0) or 0.0,
            )

    risk_off = {"MACRO_RISK_OFF", "RISK_OFF", "CAPITULATION"}
    recovery = {"RECOVERY", "BROAD_RISK_ON", "BTC_BULL", "ALT_BULL"}
    target_cash_fraction = (
        0.60 if regime in risk_off else 0.20 if regime in recovery else 0.40
    )
    best_by_market: dict[str, Mapping[str, Any]] = {}
    status_rank = {"ACTIONABLE": 4, "NEAR_ENTRY": 3, "APPROACHING": 2}
    for row in opportunities:
        market = str(row.get("market") or "")
        if market not in live_markets:
            continue
        if str(row.get("status") or "") not in status_rank:
            continue
        current = best_by_market.get(market)
        candidate_key = (
            status_rank.get(str(row.get("status") or ""), 0),
            _finite(row.get("score"), 0.0) or 0.0,
        )
        current_key = (
            status_rank.get(str((current or {}).get("status") or ""), 0),
            _finite((current or {}).get("score"), 0.0) or 0.0,
        )
        if current is None or candidate_key > current_key:
            best_by_market[market] = row

    ranked: list[dict[str, Any]] = []
    for row in rotation:
        market = str(row.get("market") or "")
        if market not in live_markets or row.get("decision") != "FAVOUR":
            continue
        opportunity = best_by_market.get(market)
        if opportunity is None:
            continue
        rotation_score = _finite(row.get("rotation_score"), 0.0) or 0.0
        opportunity_score = _finite(opportunity.get("score"), 0.0) or 0.0
        ranked.append(
            {
                "market": market,
                "allocation_score": 0.55 * rotation_score
                + 0.45 * opportunity_score,
                "rotation_score": rotation_score,
                "signal_score": opportunity_score,
                "signal_status": opportunity.get("status"),
                "strategy_id": opportunity.get("strategy"),
                "strategy_dna_hash": opportunity.get("strategy_dna_hash"),
                "timeframe": opportunity.get("timeframe"),
                "live_authority_granted": bool(
                    opportunity.get("live_authority_granted")
                ),
            }
        )
    ranked.sort(key=lambda row: row["allocation_score"], reverse=True)
    selected = ranked[:3]
    maximum_single_coin_fraction = 0.20
    risk_asset_budget = max(0.0, 1.0 - target_cash_fraction)
    target_weights: dict[str, float] = {"EUR": 1.0}

    # Existing eligible inventory is not silently assigned to a strategy, but
    # a bounded sleeve is retained in the advisory target.  Without this, any
    # externally held coin was assigned an unrealistic 0% target even when a
    # staged deconcentration to the configured single-coin cap was intended.
    retained_inventory_budget = risk_asset_budget
    for market, value in sorted(
        (
            (market, value)
            for market, value in current_values.items()
            if market != "EUR" and market in live_markets and value > 0.0
        ),
        key=lambda item: item[1],
        reverse=True,
    ):
        current_weight = value / equity if equity > 0.0 else 0.0
        retained = min(
            maximum_single_coin_fraction,
            current_weight,
            retained_inventory_budget,
        )
        if retained > 0.0:
            target_weights[market] = retained
            retained_inventory_budget -= retained

    remaining_signal_budget = max(
        0.0,
        risk_asset_budget
        - sum(
            weight
            for market, weight in target_weights.items()
            if market != "EUR"
        ),
    )
    unallocated_selected = [
        row for row in selected if row["market"] not in target_weights
    ]
    if unallocated_selected and remaining_signal_budget > 0.0:
        raw_total = sum(
            max(1.0, row["allocation_score"])
            for row in unallocated_selected
        )
        for row in unallocated_selected:
            raw_weight = (
                remaining_signal_budget
                * max(1.0, row["allocation_score"])
                / raw_total
            )
            target_weights[row["market"]] = min(
                maximum_single_coin_fraction,
                raw_weight,
            )
    allocated = sum(
        weight
        for market, weight in target_weights.items()
        if market != "EUR"
    )
    target_weights["EUR"] = max(target_cash_fraction, 1.0 - allocated)

    rows: list[dict[str, Any]] = []
    all_assets = sorted(set(current_values) | set(target_weights))
    selected_by_market = {row["market"]: row for row in selected}
    for asset in all_assets:
        current_value = current_values.get(asset, 0.0)
        current_weight = current_value / equity if equity > 0.0 else 0.0
        target_weight = target_weights.get(asset, 0.0)
        delta = target_weight - current_weight
        candidate = selected_by_market.get(asset, {})
        if asset == "EUR":
            action = "CASH_BUFFER"
        elif current_weight > target_weight + 0.02:
            action = "REDUCE_EXTERNAL_INVENTORY_REVIEW"
        elif delta > 0.02 and candidate.get("live_authority_granted"):
            action = "ACCUMULATE_ON_APPROVED_NATURAL_SIGNAL"
        elif delta > 0.02:
            action = "WAIT_FOR_DNA_APPROVAL_AND_NATURAL_SIGNAL"
        else:
            action = "HOLD"
        rows.append(
            {
                "asset": asset,
                "current_value_eur": current_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "target_value_eur": equity * target_weight,
                "delta_value_eur": equity * delta,
                "action": action,
                "signal_status": candidate.get("signal_status"),
                "strategy_id": candidate.get("strategy_id"),
                "timeframe": candidate.get("timeframe"),
                "live_authority_granted": bool(
                    candidate.get("live_authority_granted")
                ),
            }
        )
    payload = {
        "schema_version": "proactive_allocation_v1",
        "generated_at": utc_iso(),
        "regime": regime,
        "account_equity_eur": equity,
        "target_cash_fraction": target_weights["EUR"],
        "maximum_single_coin_target_fraction": maximum_single_coin_fraction,
        "external_inventory_retention_is_strategy_authority": False,
        "maximum_selected_markets": 3,
        "ranked_candidates": selected,
        "rows": rows,
        "advisory_only": True,
        "does_not_expand_execution_authority": True,
        "external_inventory_not_claimed": True,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = settings.paths.output_dir / "portfolio"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "proactive_allocation.json", payload)
    return payload


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(result.metrics).items()
        if key
        in {
            "net_return",
            "cagr",
            "maximum_drawdown",
            "sharpe",
            "sortino",
            "calmar",
            "omega",
            "trade_count",
            "profit_factor",
            "net_expectancy_r",
            "net_expectancy_eur",
            "effective_sample_size",
            "average_exposure",
            "turnover",
            "transaction_costs_eur",
            "monte_carlo_p95_drawdown",
            "probability_of_loss",
        }
    }


def _equity_returns(result: Any) -> np.ndarray:
    return (
        pd.to_numeric(
            result.equity_curve["equity"],
            errors="coerce",
        )
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )


def validate_tactical_catalogue(
    settings: Settings,
    *,
    markets: Sequence[str] | None = None,
    maximum_rows: int = 8_000,
    simulations: int = 1_000,
) -> dict[str, Any]:
    """Backtest every tactical DNA under normal/stressed/OOS costs."""

    now = utc_now()
    selected_markets = tuple(markets or ())
    if not selected_markets:
        selected_markets = tuple(
            str(value)
            for value in live_universe_status(settings).get(
                "selected_markets",
                [],
            )
        )
    frames, frame_failures = _load_scan_frames(
        settings,
        selected_markets,
        now=now,
        maximum_rows=maximum_rows,
    )
    features, feature_failures = _feature_frames(
        frames,
        selected_markets,
    )
    normal_config = BacktestConfig.from_settings(
        settings,
        initial_cash_eur=10_000.0,
        stressed=False,
    )
    stressed_config = BacktestConfig.from_settings(
        settings,
        initial_cash_eur=10_000.0,
        stressed=True,
    )
    stochastic_policy = StochasticValidationPolicy(
        simulations=max(100, int(simulations)),
        expected_block_length=12,
        maximum_drawdown=0.50,
        maximum_drawdown_breach_probability=0.20,
        maximum_terminal_loss_probability=0.20,
        minimum_p05_total_return=-0.20,
        dirichlet_blocks=8,
        minimum_observations=30,
        seed=int(settings.app.random_seed),
        batch_size=128,
    )
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(tactical_strategy_specs()):
        strategy = TacticalMultiTimeframeStrategy(spec)
        selected_features = {
            market: features[(market, spec.timeframe)]
            for market in selected_markets
            if (market, spec.timeframe) in features
        }
        if not selected_features:
            rows.append(
                {
                    "strategy_id": spec.strategy_id,
                    "strategy_dna_hash": spec.dna_hash,
                    "family": spec.family,
                    "timeframe": spec.timeframe,
                    "status": "DATA_BLOCKED",
                    "hard_blockers": ["NO_VALID_FEATURE_FRAMES"],
                    "capital_warnings": [],
                    "live_authority_granted": False,
                }
            )
            continue
        normal = BacktestEngine(
            normal_config,
            settings=settings,
        ).run(selected_features, strategy)
        stressed = BacktestEngine(
            stressed_config,
            settings=settings,
        ).run(selected_features, strategy)
        oos_features: dict[str, pd.DataFrame] = {}
        for market, frame in selected_features.items():
            start = max(0, int(len(frame) * 0.70) - 250)
            oos = frame.iloc[start:].copy()
            oos.attrs.update(frame.attrs)
            oos_features[market] = oos
        out_of_sample = BacktestEngine(
            normal_config,
            settings=settings,
        ).run(oos_features, strategy)
        stochastic = validate_strategy_return_paths(
            _equity_returns(normal),
            _equity_returns(stressed),
            policy=stochastic_policy,
            seed_offset=index * 10_000,
        )
        normal_metrics = _result_summary(normal)
        stressed_metrics = _result_summary(stressed)
        oos_metrics = _result_summary(out_of_sample)
        integrity = {
            "normal": dict(normal.integrity),
            "stressed": dict(stressed.integrity),
            "out_of_sample": dict(out_of_sample.integrity),
        }
        hard_integrity = all(
            result.integrity.get("valid_data") is True
            and result.integrity.get("no_lookahead") is True
            and result.integrity.get("no_repainting") is True
            and result.integrity.get("closed_candle_integrity") is True
            and result.integrity.get("next_open_execution") is True
            for result in (normal, stressed, out_of_sample)
        )
        research_positive = bool(
            hard_integrity
            and float(normal_metrics.get("net_return") or 0.0) > 0.0
            and float(normal_metrics.get("profit_factor") or 0.0) > 1.0
            and float(normal_metrics.get("net_expectancy_r") or 0.0) > 0.0
        )
        paper_eligible = bool(
            research_positive
            and int(normal_metrics.get("trade_count") or 0) >= 5
        )
        micro_eligible = bool(
            paper_eligible
            and float(normal_metrics.get("profit_factor") or 0.0) >= 1.10
            and float(stressed_metrics.get("profit_factor") or 0.0) > 1.0
            and float(oos_metrics.get("net_return") or 0.0) > 0.0
            and int(normal_metrics.get("trade_count") or 0) >= 20
        )
        status = (
            "LIVE_MICRO_ELIGIBLE_REQUIRES_OPERATOR_DNA_APPROVAL"
            if micro_eligible
            else "PAPER_ELIGIBLE"
            if paper_eligible
            else "RESEARCH_POSITIVE"
            if research_positive
            else "SHADOW_ONLY"
        )
        capital_warnings = []
        if stochastic.get("passed") is not True:
            capital_warnings.append("MONTE_CARLO_OR_DIRICHLET_WARNING")
        if float(stressed_metrics.get("profit_factor") or 0.0) <= 1.0:
            capital_warnings.append("STRESSED_PROFIT_FACTOR_NOT_POSITIVE")
        if float(oos_metrics.get("net_return") or 0.0) <= 0.0:
            capital_warnings.append("OUT_OF_SAMPLE_NET_RETURN_NOT_POSITIVE")
        rows.append(
            {
                "strategy_id": spec.strategy_id,
                "strategy_dna_hash": spec.dna_hash,
                "family": spec.family,
                "timeframe": spec.timeframe,
                "status": status,
                "research_positive": research_positive,
                "paper_eligible": paper_eligible,
                "micro_live_eligible": micro_eligible,
                "live_authority_granted": False,
                "normal": normal_metrics,
                "stressed": stressed_metrics,
                "out_of_sample": oos_metrics,
                "stochastic_validation": stochastic,
                "integrity": integrity,
                "hard_blockers": (
                    [] if hard_integrity else ["INTEGRITY_CHECK_FAILED"]
                ),
                "capital_warnings": capital_warnings,
                "parameters": {
                    "stop_atr": spec.stop_atr,
                    "target_atr": spec.target_atr,
                    "trailing_atr": spec.trailing_atr,
                    "maximum_holding_bars": spec.maximum_holding_bars,
                },
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("micro_live_eligible")),
            bool(row.get("paper_eligible")),
            bool(row.get("research_positive")),
            float((row.get("out_of_sample") or {}).get("net_return") or -1.0),
            float((row.get("normal") or {}).get("profit_factor") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "tactical_multitimeframe_validation_v1",
        "generated_at": now.isoformat(),
        "markets": list(selected_markets),
        "maximum_rows_per_market": maximum_rows,
        "evaluated_tactical_strategy_instance_count": len(rows),
        "research_positive_count": sum(
            bool(row.get("research_positive")) for row in rows
        ),
        "paper_eligible_count": sum(
            bool(row.get("paper_eligible")) for row in rows
        ),
        "micro_live_eligible_count": sum(
            bool(row.get("micro_live_eligible")) for row in rows
        ),
        "live_authority_granted_count": 0,
        "frame_failures": {
            f"{market}:{timeframe}": reason
            for (market, timeframe), reason in frame_failures.items()
        },
        "feature_failures": {
            f"{market}:{timeframe}": reason
            for (market, timeframe), reason in feature_failures.items()
        },
        "stochastic_policy": {
            "policy_hash": stochastic_policy.policy_hash,
            "simulations": stochastic_policy.simulations,
            "dirichlet_blocks": stochastic_policy.dirichlet_blocks,
            "interpretation": (
                "Monte Carlo and Dirichlet are capital-confidence evidence, "
                "not integrity overrides."
            ),
        },
        "strategies": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = settings.paths.output_dir / "active_trading"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "tactical_validation.json"
    csv_path = output / "tactical_validation.csv"
    atomic_write_json(json_path, payload)
    pd.DataFrame(
        [
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "family": row["family"],
                "timeframe": row["timeframe"],
                "status": row["status"],
                "research_positive": row.get("research_positive"),
                "paper_eligible": row.get("paper_eligible"),
                "micro_live_eligible": row.get("micro_live_eligible"),
                "net_return": (row.get("normal") or {}).get("net_return"),
                "profit_factor": (row.get("normal") or {}).get(
                    "profit_factor"
                ),
                "trade_count": (row.get("normal") or {}).get("trade_count"),
                "stressed_profit_factor": (
                    row.get("stressed") or {}
                ).get("profit_factor"),
                "out_of_sample_net_return": (
                    row.get("out_of_sample") or {}
                ).get("net_return"),
                "monte_carlo_passed": (
                    row.get("stochastic_validation") or {}
                )
                .get("normal", {})
                .get("monte_carlo", {})
                .get("passed"),
                "dirichlet_passed": (
                    row.get("stochastic_validation") or {}
                )
                .get("normal", {})
                .get("dirichlet", {})
                .get("passed"),
                "live_authority_granted": False,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    return {
        **payload,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
        },
    }


def build_capital_utilization(settings: Settings) -> dict[str, Any]:
    health = _safe_read(
        settings.paths.output_dir
        / "operations"
        / "live_account_health.json"
    )
    account = dict(health.get("account") or {})
    valuation = dict(account.get("portfolio_valuation") or {})
    equity = _finite(valuation.get("estimated_total_equity_eur"), 0.0) or 0.0
    cash = _finite(account.get("eur_available"), 0.0) or 0.0
    holdings = [
        dict(row)
        for row in valuation.get("holdings") or []
        if isinstance(row, Mapping)
    ]
    material_inventory = [
        row
        for row in holdings
        if (_finite(row.get("estimated_value_eur"), 0.0) or 0.0) >= 5.0
    ]
    inventory_exposure = sum(
        _finite(row.get("estimated_value_eur"), 0.0) or 0.0
        for row in material_inventory
    )
    live_state = _safe_read(
        settings.paths.output_dir / "live" / "generated_strategy_live_state.json"
    )
    strategy_positions = [
        dict(row)
        for row in (live_state.get("positions") or {}).values()
        if isinstance(row, Mapping)
    ]
    strategy_exposure = sum(
        (_finite(row.get("quantity"), 0.0) or 0.0)
        * (_finite(row.get("entry_price"), 0.0) or 0.0)
        for row in strategy_positions
    )
    used = min(equity, inventory_exposure + strategy_exposure)
    cash_fraction = cash / equity if equity > 0.0 else 0.0
    utilization = used / equity if equity > 0.0 else 0.0
    authority = _safe_read(
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    maximum_strategy_exposure = _finite(
        authority.get("maximum_total_exposure_eur"),
        0.0,
    ) or 0.0
    unused_risk_budget = max(0.0, maximum_strategy_exposure - strategy_exposure)
    payload = {
        "schema_version": "capital_utilization_v1",
        "generated_at": utc_iso(),
        "valuation_status": valuation.get("status"),
        "account_equity_eur": equity,
        "eur_cash": cash,
        "cash_percentage": cash_fraction,
        "inventory_exposure_eur": inventory_exposure,
        "strategy_exposure_eur": strategy_exposure,
        "capital_utilization": utilization,
        "unused_strategy_risk_budget_eur": unused_risk_budget,
        "blocked_capital_eur": inventory_exposure,
        "material_inventory": material_inventory,
        "capital_allocated_by_timeframe": {
            str(row.get("timeframe") or "UNKNOWN"): (
                _finite(row.get("quantity"), 0.0) or 0.0
            )
            * (_finite(row.get("entry_price"), 0.0) or 0.0)
            for row in strategy_positions
        },
        "current_stage": "MICRO",
        "stage_caps": {
            "maximum_order_eur": authority.get("maximum_order_eur"),
            "maximum_strategy_exposure_eur": authority.get(
                "maximum_total_exposure_eur"
            ),
            "maximum_open_positions": authority.get(
                "maximum_open_positions"
            ),
        },
        "next_stage": "SMALL",
        "next_stage_requirements": [
            "OPERATOR_CAPITAL_APPROVAL",
            "AT_LEAST_THREE_FLAWLESS_LIVE_ROUND_TRIPS",
            "NO_RECONCILIATION_OR_DUPLICATE_ORDER_INCIDENTS",
            "LIVE_SLIPPAGE_WITHIN_MODEL_BAND",
            "NON_NEGATIVE_NET_LIVE_EXPECTANCY",
        ],
        "autoscale": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = settings.paths.output_dir / "portfolio"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "capital_utilization.json", payload)
    return payload


def build_tao_inventory_policy(settings: Settings) -> dict[str, Any]:
    utilization = build_capital_utilization(settings)
    match = next(
        (
            dict(row)
            for row in utilization.get("material_inventory") or []
            if str(row.get("market") or "") == "TAO-EUR"
            or str(row.get("symbol") or "") == "TAO"
        ),
        {},
    )
    value = _finite(match.get("estimated_value_eur"), 0.0) or 0.0
    equity = _finite(utilization.get("account_equity_eur"), 0.0) or 0.0
    concentration = value / equity if equity > 0.0 else 0.0
    payload = {
        "schema_version": "tao_inventory_policy_v1",
        "generated_at": utc_iso(),
        "market": "TAO-EUR",
        "classification": (
            "REVIEW_REQUIRED" if value >= 5.0 else "NO_MATERIAL_INVENTORY"
        ),
        "position_owner": "EXTERNAL_INVENTORY",
        "bot_managed": False,
        "estimated_cost_basis": None,
        "current_value_eur": value,
        "account_concentration": concentration,
        "unrealized_pnl": None,
        "strategy_thesis": None,
        "regime_compatibility": "REQUIRES_EXPLICIT_CLAIM_AND_POLICY",
        "stop_policy": "NO_AUTOMATIC_STOP_UNTIL_OPERATOR_CLAIM",
        "reduction_policy": (
            "CONCENTRATION_COUNTS_AGAINST_NEW_RISK_BUDGET"
            if concentration >= 0.25
            else "MONITOR"
        ),
        "full_exit_policy": "NO_AUTOMATIC_EXIT_WITHOUT_OWNERSHIP_TRANSFER",
        "maximum_allowed_exposure": 0.10,
        "concentration_warning": concentration > 0.10,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = settings.paths.output_dir / "inventory"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "TAO-EUR_policy.json", payload)
    return payload


def build_lower_timeframe_candidate_queue(settings: Settings) -> dict[str, Any]:
    """Reconcile frozen 1h/2h DNA with paper state and explicit live authority."""

    frozen = _safe_read(
        settings.paths.output_dir
        / "strategies"
        / "frozen_classical_paper_candidates.json"
    )
    paper = _safe_read(
        settings.paths.output_dir / "paper" / "generated_strategy_state.json"
    )
    authority = _safe_read(
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    evaluations = dict(paper.get("evaluations") or {})
    authorized_by_dna = {
        str(row.get("strategy_dna_hash") or ""): dict(row)
        for row in authority.get("approved_candidates") or []
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for candidate in frozen.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        timeframe = str(candidate.get("timeframe") or "")
        if timeframe not in TACTICAL_TIMEFRAMES:
            continue
        dna = str(candidate.get("strategy_dna_hash") or "")
        metrics = dict(candidate.get("metrics") or {})
        profit_factor = _finite(metrics.get("profit_factor"), 0.0) or 0.0
        net_return = _finite(metrics.get("net_return"), 0.0) or 0.0
        expectancy = _finite(metrics.get("net_expectancy_r"), 0.0) or 0.0
        trade_count = max(0, int(_finite(metrics.get("trade_count"), 0.0) or 0))
        sample_weight = trade_count / (trade_count + 50.0) if trade_count else 0.0
        adjusted_profit_factor = sample_weight * min(5.0, profit_factor) + (
            1.0 - sample_weight
        )
        paper_evaluation = dict(evaluations.get(dna) or {})
        paper_evaluated = paper_evaluation.get("status") == "EVALUATED"
        authorized = dna in authorized_by_dna
        hard_blockers: list[str] = []
        if len(dna) != 64:
            hard_blockers.append("STRATEGY_DNA_INVALID")
        if not candidate.get("frozen_candidate_hash"):
            hard_blockers.append("FROZEN_IDENTITY_MISSING")
        if profit_factor <= 1.0:
            hard_blockers.append("NORMAL_COST_PROFIT_FACTOR_NOT_POSITIVE")
        if net_return <= 0.0:
            hard_blockers.append("NORMAL_COST_NET_RETURN_NOT_POSITIVE")
        if expectancy <= 0.0:
            hard_blockers.append("NORMAL_COST_EXPECTANCY_NOT_POSITIVE")
        warnings: list[str] = []
        if not paper_evaluated:
            warnings.append("PAPER_EVALUATION_PENDING")
        stressed_pf = _finite(metrics.get("stressed_profit_factor"))
        holdout_pf = _finite(metrics.get("holdout_profit_factor"))
        drawdown = abs(_finite(metrics.get("maximum_drawdown"), 0.0) or 0.0)
        raw_monte_carlo_drawdown = _finite(
            metrics.get("monte_carlo_p95_drawdown")
        )
        monte_carlo_drawdown = (
            abs(raw_monte_carlo_drawdown)
            if raw_monte_carlo_drawdown is not None
            else None
        )
        if trade_count < 30:
            warnings.append("SMALL_SAMPLE_BELOW_30_TRADES")
        if stressed_pf is None:
            warnings.append("STRESSED_COST_EVIDENCE_MISSING")
        elif stressed_pf <= 1.0:
            warnings.append("STRESSED_COST_EDGE_NOT_POSITIVE")
        if holdout_pf is None:
            warnings.append("HOLDOUT_EVIDENCE_MISSING")
        elif holdout_pf <= 1.0:
            warnings.append("HOLDOUT_EDGE_NOT_POSITIVE")
        if drawdown > 0.25:
            warnings.append("HISTORICAL_DRAWDOWN_ABOVE_25_PERCENT")
        if monte_carlo_drawdown is None:
            warnings.append("MONTE_CARLO_EVIDENCE_MISSING")
        elif monte_carlo_drawdown > 0.25:
            warnings.append("MONTE_CARLO_P95_DRAWDOWN_ABOVE_25_PERCENT")
        paper_eligible = not hard_blockers
        micro_eligible = (
            paper_eligible
            and paper_evaluated
            and profit_factor >= 1.10
        )
        if authorized:
            status = "LIVE_MICRO_AUTHORIZED"
        elif micro_eligible:
            status = "LIVE_MICRO_ELIGIBLE_REQUIRES_OPERATOR_DNA_APPROVAL"
        elif paper_eligible and paper_evaluated:
            status = "PAPER_ACTIVE"
        elif paper_eligible:
            status = "PAPER_PENDING_EVALUATION"
        else:
            status = "PAPER_BLOCKED"
        double_cost_pf = _finite(metrics.get("double_cost_profit_factor"))
        robustness_values = (stressed_pf, double_cost_pf, holdout_pf)
        robustness_count = sum(
            value is not None and value > 1.0
            for value in robustness_values
        )
        robustness_observed = sum(
            value is not None for value in robustness_values
        )
        # Extreme PF values from tiny samples must not dominate the operator
        # approval queue.  This score changes only prioritisation: practical
        # micro-live eligibility remains governed by the explicit hard
        # blockers above and still requires a separate operator phrase.
        edge_score = 25.0 * min(
            1.0,
            max(0.0, (profit_factor - 1.0) / 0.50),
        )
        sample_score = 25.0 * min(1.0, trade_count / 100.0)
        robustness_score = 15.0 * robustness_count / 3.0
        evidence_score = 5.0 * robustness_observed / 3.0
        drawdown_score = 20.0 * (
            1.0 - min(1.0, drawdown / 0.25)
        )
        monte_carlo_score = (
            10.0 * (1.0 - min(1.0, monte_carlo_drawdown / 0.30))
            if monte_carlo_drawdown is not None
            else 0.0
        )
        penalties = 0.0
        if trade_count < 30:
            penalties += 12.0
        if stressed_pf is not None and stressed_pf <= 1.0:
            penalties += 10.0
        if holdout_pf is not None and holdout_pf <= 1.0:
            penalties += 15.0
        if drawdown > 0.25:
            penalties += 10.0
        if (
            monte_carlo_drawdown is not None
            and monte_carlo_drawdown > 0.25
        ):
            penalties += 8.0
        score = max(
            0.0,
            min(
                100.0,
                edge_score
                + sample_score
                + robustness_score
                + evidence_score
                + drawdown_score
                + monte_carlo_score
                - penalties,
            ),
        )
        if authorized:
            approval_priority = "AUTHORIZED"
        elif not micro_eligible:
            approval_priority = "NOT_ELIGIBLE"
        elif holdout_pf is not None and holdout_pf <= 1.0:
            approval_priority = "DEFER_NEGATIVE_HOLDOUT"
        elif (
            trade_count < 20
            or drawdown > 0.25
            or monte_carlo_drawdown is None
            or monte_carlo_drawdown > 0.25
        ):
            approval_priority = "DEFER_WEAK_EVIDENCE"
        elif (
            trade_count >= 40
            and drawdown <= 0.15
            and monte_carlo_drawdown is not None
            and monte_carlo_drawdown <= 0.20
            and (stressed_pf is None or stressed_pf > 1.0)
        ):
            approval_priority = "PRIORITY_MICRO"
        else:
            approval_priority = "SECONDARY_MICRO"
        strategy_id = str(candidate.get("strategy_id") or "")
        approval_command = (
            ".\\.venv\\Scripts\\python.exe .\\main.py "
            "live approve-positive-dna "
            f"--strategy-id {strategy_id} "
            f'--approval "LIVE POSITIVE DNA {strategy_id} CONFIRMED"'
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_dna_hash": dna,
                "frozen_candidate_hash": candidate.get(
                    "frozen_candidate_hash"
                ),
                "timeframe": timeframe,
                "family": candidate.get("economic_hypothesis_family"),
                "markets": list(candidate.get("markets") or []),
                "source": candidate.get("source"),
                "status": status,
                "paper_evaluated": paper_evaluated,
                "paper_eligible": paper_eligible,
                "live_authorized": authorized,
                "micro_live_eligible": micro_eligible,
                "operator_dna_approval_required": bool(
                    micro_eligible and not authorized
                ),
                "profit_factor": profit_factor,
                "adjusted_profit_factor": adjusted_profit_factor,
                "stressed_profit_factor": stressed_pf,
                "double_cost_profit_factor": double_cost_pf,
                "holdout_profit_factor": holdout_pf,
                "net_return": net_return,
                "net_expectancy_r": expectancy,
                "trade_count": trade_count,
                "sample_weight": sample_weight,
                "maximum_drawdown": drawdown,
                "monte_carlo_p95_drawdown": monte_carlo_drawdown,
                "hard_blockers": hard_blockers,
                "capital_scaling_warnings": warnings,
                "selection_score": score,
                "approval_readiness_score": score,
                "approval_priority": approval_priority,
                "approval_command": approval_command,
                "unknown_dna_fails_closed": True,
            }
        )
    priority_order = {
        "AUTHORIZED": 5,
        "PRIORITY_MICRO": 4,
        "SECONDARY_MICRO": 3,
        "DEFER_NEGATIVE_HOLDOUT": 2,
        "DEFER_WEAK_EVIDENCE": 1,
        "NOT_ELIGIBLE": 0,
    }
    rows.sort(
        key=lambda row: (
            bool(row["live_authorized"]),
            bool(row["micro_live_eligible"]),
            priority_order.get(str(row["approval_priority"]), 0),
            float(row["selection_score"]),
            int(row["trade_count"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    payload = {
        "schema_version": "lower_timeframe_candidate_queue_v3",
        "generated_at": utc_iso(),
        "timeframes": list(TACTICAL_TIMEFRAMES),
        "candidate_count": len(rows),
        "paper_evaluated_count": sum(row["paper_evaluated"] for row in rows),
        "live_authorized_count": sum(row["live_authorized"] for row in rows),
        "pending_operator_dna_approval_count": sum(
            row["operator_dna_approval_required"] for row in rows
        ),
        "priority_micro_candidate_count": sum(
            row["approval_priority"] == "PRIORITY_MICRO"
            for row in rows
        ),
        "deferred_candidate_count": sum(
            str(row["approval_priority"]).startswith("DEFER_")
            for row in rows
        ),
        "independent_family_count": len(
            {str(row.get("family") or "UNKNOWN") for row in rows}
        ),
        "auto_live_promotion": False,
        "unknown_dna_fail_closed": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "candidates": rows,
    }
    output = settings.paths.output_dir / "governance"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "lower_timeframe_candidate_queue.json",
        payload,
    )
    return payload


def _timeframe_status(
    settings: Settings,
    *,
    evaluations: Mapping[str, int],
    opportunities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authority = _safe_read(
        settings.paths.output_dir
        / "governance"
        / "multi_timeframe_authority.json"
    )
    authority_rows = [
        dict(row)
        for row in authority.get("strategies") or []
        if isinstance(row, Mapping)
    ]
    candidate_queue = build_lower_timeframe_candidate_queue(settings)
    queued_candidates = [
        dict(row)
        for row in candidate_queue.get("candidates") or []
        if isinstance(row, Mapping)
    ]
    validation = _safe_read(
        settings.paths.output_dir
        / "active_trading"
        / "tactical_validation.json"
    )
    validation_by_dna = {
        str(row.get("strategy_dna_hash")): dict(row)
        for row in validation.get("strategies") or []
        if isinstance(row, Mapping)
    }
    catalogue = tactical_catalogue_payload()
    rows: dict[str, Any] = {}
    for timeframe in TACTICAL_TIMEFRAMES:
        live = [
            row
            for row in authority_rows
            if str(row.get("timeframe")) == timeframe
            and str(row.get("authority")) == "LIVE_MICRO"
        ]
        tactical = [
            {
                **row,
                "validation_status": validation_by_dna.get(
                    str(row.get("strategy_dna_hash")),
                    {},
                ).get("status", "VALIDATION_PENDING"),
                "validation_metrics": validation_by_dna.get(
                    str(row.get("strategy_dna_hash")),
                    {},
                ).get("normal", {}),
                "stressed_metrics": validation_by_dna.get(
                    str(row.get("strategy_dna_hash")),
                    {},
                ).get("stressed", {}),
                "out_of_sample_metrics": validation_by_dna.get(
                    str(row.get("strategy_dna_hash")),
                    {},
                ).get("out_of_sample", {}),
                "paper_eligible": bool(
                    validation_by_dna.get(
                        str(row.get("strategy_dna_hash")),
                        {},
                    ).get("paper_eligible")
                ),
                "micro_live_eligible_pending_operator_approval": bool(
                    validation_by_dna.get(
                        str(row.get("strategy_dna_hash")),
                        {},
                    ).get("micro_live_eligible")
                ),
            }
            for row in catalogue["strategies"]
            if row["timeframe"] == timeframe
        ]
        frozen_positive = [
            row
            for row in queued_candidates
            if str(row.get("timeframe")) == timeframe
        ]
        pending_approval = [
            row
            for row in frozen_positive
            if row.get("operator_dna_approval_required") is True
        ]
        selected_opportunities = [
            row
            for row in opportunities
            if str(row.get("timeframe")) == timeframe
        ]
        rows[timeframe] = {
            "total_strategies": len(tactical) + len(frozen_positive),
            "independent_families": len(
                {str(row.get("family") or "UNKNOWN") for row in tactical}
                | {
                    str(row.get("family") or "UNKNOWN")
                    for row in frozen_positive
                }
            ),
            "positive_historical_candidates": len(frozen_positive) + sum(
                row["validation_status"]
                in {
                    "RESEARCH_POSITIVE",
                    "PAPER_ELIGIBLE",
                    "LIVE_MICRO_ELIGIBLE_REQUIRES_OPERATOR_DNA_APPROVAL",
                }
                for row in tactical
            ),
            "validation_pending": sum(
                row["validation_status"] == "VALIDATION_PENDING"
                for row in tactical
            ),
            "economic_blocked": sum(
                row["validation_status"] in {"SHADOW_ONLY", "DATA_BLOCKED"}
                for row in tactical
            ),
            "shadow": sum(not row["paper_eligible"] for row in tactical),
            "paper": sum(
                row.get("paper_evaluated") is True
                for row in frozen_positive
            ) + sum(row["paper_eligible"] for row in tactical),
            "micro_live_eligible_pending_operator_approval": sum(
                row["micro_live_eligible_pending_operator_approval"]
                for row in tactical
            ) + len(pending_approval),
            "micro_live": sum(
                row.get("live_authorized") is True
                for row in frozen_positive
            ),
            "normal_live": 0,
            "blocked": sum(
                row.get("status") == "AVOID"
                for row in selected_opportunities
            ),
            "markets_scanned": len(
                {row["market"] for row in selected_opportunities}
            ),
            "strategies_evaluated": int(evaluations.get(timeframe) or 0),
            "valid_signals": sum(
                row.get("status") in {"ACTIONABLE", "NEAR_ENTRY"}
                for row in selected_opportunities
            ),
            "near_entries": sum(
                row.get("status") == "NEAR_ENTRY"
                for row in selected_opportunities
            ),
            "actionable_entries": sum(
                row.get("status") == "ACTIONABLE"
                for row in selected_opportunities
            ),
            "live_strategies": live,
            "frozen_positive_strategies": frozen_positive,
            "pending_live_dna_approval": pending_approval,
            "shadow_strategies": tactical,
        }
    payload = {
        "schema_version": "active_timeframe_status_v1",
        "generated_at": utc_iso(),
        "timeframes": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    output = settings.paths.output_dir / "active_trading"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "timeframes.json", payload)
    return payload


async def scan_all(
    settings: Settings,
    *,
    refresh_external: bool = True,
    execute: bool = False,
    notify: bool = True,
    maximum_rows: int = 3_000,
) -> dict[str, Any]:
    """Run a full causal scan and optionally delegate approved live execution."""

    scan_started_at = utc_now()
    previous = active_trading_status(settings)
    previous_regime = str(
        (previous.get("macro") or {}).get("regime") or ""
    ) or None
    universe = live_universe_status(settings)
    tiered_universe = await asyncio.to_thread(
        build_tiered_trading_universe,
        settings,
        live_report=universe,
        maximum_shadow_markets=50,
        now=scan_started_at,
    )
    markets = tuple(
        str(value)
        for value in tiered_universe.get("shadow_markets") or []
    )
    live_markets = {
        str(value)
        for value in tiered_universe.get("live_executable_markets") or []
    }
    if not markets:
        markets = tuple(
            str(value)
            for value in universe.get("selected_markets") or []
        ) or tuple(settings.operational.markets)
    refreshed = (
        await refresh_public_macro_context(settings)
        if refresh_external
        else None
    )
    # The decision clock starts only after every external observation has been
    # received.  This guarantees available_at <= decision time for live use.
    now = utc_now()
    frames, frame_failures = await asyncio.to_thread(
        _load_scan_frames,
        settings,
        markets,
        now=now,
        maximum_rows=maximum_rows,
    )
    recovered_frames, fast_data_recovery = await _recover_recent_fast_frames(
        settings,
        markets,
        frames=frames,
        now=now,
        maximum_rows=maximum_rows,
    )
    frames.update(recovered_frames)
    for key in recovered_frames:
        frame_failures.pop(key, None)
    # A repaired native 1h tail must also refresh its causal 2h context in
    # this same decision cycle.  Never retain a stale cached 2h frame behind
    # a newly recovered execution frame.
    for market in markets:
        source = recovered_frames.get((market, "1h"))
        if source is None:
            continue
        try:
            derived = resample_ohlcv(
                source,
                source_timeframe="1h",
                target_timeframe="2h",
                drop_incomplete=True,
            )
            derived = drop_open_candles(derived, timeframe="2h", now=now)
            _assert_frame_fresh(derived, timeframe="2h", now=now)
            if len(derived) > maximum_rows:
                derived = derived.iloc[-maximum_rows:].copy()
            derived.attrs.update(
                {
                    "market": market,
                    "timeframe": "2h",
                    "data_provenance": {
                        "source_type": "RECOVERED_CAUSAL_1H_RESAMPLE",
                        "source_timeframe": "1h",
                        "synthetic_data_used": False,
                    },
                }
            )
            frames[(market, "2h")] = derived
            frame_failures.pop((market, "2h"), None)
        except (OSError, TypeError, ValueError) as exc:
            frame_failures[(market, "2h")] = str(exc) or type(exc).__name__
    features, feature_failures = await asyncio.to_thread(
        _feature_frames,
        frames,
        markets,
    )
    data_health = _fast_data_health(
        settings,
        frames,
        markets,
        now=now,
    )
    data_health["last_rest_recovery"] = fast_data_recovery
    market_opportunity_intensity = _market_opportunity_intensity(
        frames,
        features,
        markets,
        previous=previous,
        now=now,
    )
    macro = build_crypto_macro_snapshot(
        settings,
        refreshed=refreshed,
        market_frames=frames,
        now=now,
    )
    rotation = build_rotation_ranking(
        frames,
        markets,
        regime=str(macro["regime"]),
    )
    opportunities, evaluations = _scan_tactical_opportunities(
        settings,
        features,
        frames,
        markets,
        macro=macro,
        rotation=rotation,
    )
    market_mechanics = build_market_mechanics_snapshot(
        settings,
        markets=markets,
        now=now,
    )
    mechanics_opportunities, mechanics_evaluations = (
        _scan_market_mechanics_opportunities(
            features,
            frames,
            markets,
            market_mechanics,
        )
    )
    opportunities.extend(mechanics_opportunities)
    for timeframe in ("15m", "1h"):
        evaluations[timeframe] = int(evaluations.get(timeframe) or 0) + sum(
            str(row.get("entry_timeframe") or "1h") == timeframe
            for row in mechanics_opportunities
        )
    early_moves = detect_early_moves(
        frames,
        features,
        markets,
        rotation=rotation,
        mechanics=market_mechanics,
        regime=str(macro["regime"]),
    )
    opportunities.extend(early_moves)
    evaluations["15m"] = int(evaluations.get("15m") or 0) + len(early_moves)
    opportunities.sort(
        key=lambda row: (
            row.get("status") == "ACTIONABLE",
            row.get("status") == "EARLY_MOMENTUM_ALERT",
            row.get("status") == "NEAR_ENTRY",
            row.get("status") == "PULLBACK_PENDING",
            row.get("status") == "APPROACHING",
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(opportunities, start=1):
        row["rank"] = rank
    tier_by_market = {
        str(row.get("market")): str(row.get("highest_tier") or "DISCOVERY")
        for row in tiered_universe.get("rows") or []
    }
    for row in opportunities:
        market = str(row.get("market") or "")
        row["market_tier"] = tier_by_market.get(market, "DISCOVERY")
        row["live_market_executable"] = market in live_markets
        if str(row.get("family") or "") == "EARLY_MOVE_DETECTION":
            row["execution_scope"] = (
                "EARLY_MOVE_ALERT_ONLY_NEW_DNA_REQUIRES_APPROVAL"
                if market in live_markets
                else "SHADOW_RESEARCH_ONLY"
            )
        elif row.get("operator_dna_approval_required") is True:
            row["execution_scope"] = (
                "MARKET_MECHANICS_DNA_REQUIRES_OPERATOR_APPROVAL"
                if market in live_markets
                else "SHADOW_RESEARCH_ONLY"
            )
        else:
            if row.get("family_canary_authority_available") is True:
                row["execution_scope"] = (
                    "APPROVED_FAMILY_ENVELOPE_REQUIRES_REALTIME_CONFIRMATION"
                )
            else:
                row["execution_scope"] = (
                    "TACTICAL_DNA_REQUIRES_SEPARATE_OPERATOR_APPROVAL"
                    if market in live_markets
                    else "SHADOW_RESEARCH_ONLY"
                )
    for row in rotation:
        market = str(row.get("market") or "")
        row["market_tier"] = tier_by_market.get(market, "DISCOVERY")
        row["live_market_executable"] = market in live_markets
    actionable = [
        row for row in opportunities if row["status"] == "ACTIONABLE"
    ]
    near_entry = [
        row for row in opportunities if row["status"] == "NEAR_ENTRY"
    ]
    early_move_alerts = [
        row
        for row in opportunities
        if row["status"]
        in {
            "EARLY_MOMENTUM_ALERT",
            "PULLBACK_PENDING",
            "FIRST_PULLBACK_AFTER_IMPULSE",
            "EXTENDED_MOVE_WAIT_FOR_PULLBACK",
        }
    ]
    breakout = [
        row
        for row in opportunities
        if "BREAKOUT" in str(row["family"])
    ]
    pullback = [
        row
        for row in opportunities
        if "PULLBACK" in str(row["family"])
    ]
    relative_strength = [
        row
        for row in opportunities
        if "RELATIVE_STRENGTH" in str(row["family"])
        or "CROSS_SECTIONAL" in str(row["family"])
    ]
    defensive = [
        row
        for row in opportunities
        if any(
            key in str(row["family"])
            for key in ("DEFENSIVE", "RECOVERY", "REVERSION")
        )
    ]
    capital = build_capital_utilization(settings)
    allocation = build_proactive_allocation_plan(
        settings,
        rotation=rotation,
        opportunities=opportunities,
        regime=str(macro["regime"]),
        live_markets=live_markets,
    )
    tao = build_tao_inventory_policy(settings)
    from core.opportunity_intelligence import (
        record_active_swing_forward_snapshots,
    )

    forward_evidence = await asyncio.to_thread(
        record_active_swing_forward_snapshots,
        settings,
        opportunities,
        observed_at=now,
    )
    timeframes = _timeframe_status(
        settings,
        evaluations=evaluations,
        opportunities=opportunities,
    )
    output = settings.paths.output_dir / "active_trading"
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "status": output / "status.json",
        "opportunities": output / "opportunities.json",
        "early_moves": output / "early_moves.json",
        "macro": output / "macro_crypto.json",
        "rotation": output / "rotation.json",
        "market_mechanics": output / "market_mechanics.json",
        "data_health": output / "data_health.json",
        "market_opportunity_intensity": (
            output / "market_opportunity_intensity.json"
        ),
        "missed_market_move_incident": (
            output / "missed_market_move_incident.json"
        ),
        "opportunity_capture": output / "opportunity_capture.json",
    }
    opportunity_payload = {
        "schema_version": "active_opportunities_v1",
        "generated_at": now.isoformat(),
        "top_5_actionable": actionable[:5],
        "top_5_near_entry": near_entry[:5],
        "top_5_early_moves": early_move_alerts[:5],
        "top_5_breakout_candidates": breakout[:5],
        "top_5_pullback_candidates": pullback[:5],
        "top_5_relative_strength": relative_strength[:5],
        "top_5_defensive": defensive[:5],
        "top_5_rotation": rotation[:5],
        "all": opportunities,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(files["opportunities"], opportunity_payload)
    atomic_write_json(
        files["early_moves"],
        {
            "schema_version": "early_move_observations_v1",
            "generated_at": now.isoformat(),
            "rows": early_move_alerts,
            "closed_candle_only": True,
            "live_authority_granted": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    atomic_write_json(files["macro"], macro)
    atomic_write_json(
        files["rotation"],
        {
            "schema_version": "active_rotation_v1",
            "generated_at": now.isoformat(),
            "rows": rotation,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    # Persist the causal context before delegating to the canonical execution
    # engine.  The live evaluator loads this artifact fail-closed; writing it
    # afterwards would make an execution-enabled scan use the previous hourly
    # snapshot instead of the observations fetched for this decision clock.
    execution = None
    if execute:
        execution = await execute_generated_strategy_live_once(
            settings,
            submit=True,
            allow_new_entry=True,
            observed_at=now,
        )
    orders_generated = int(
        (execution or {}).get("orders_generated_this_cycle") or 0
    )
    orders_submitted = int(
        (execution or {}).get("orders_submitted_this_cycle") or 0
    )
    execution_funnel = _execution_funnel(
        opportunities=opportunities,
        frame_failures=frame_failures,
        feature_failures=feature_failures,
        execution=execution,
    )
    entry_ready_count = int(
        (execution_funnel.get("stage_counts") or {}).get("entry_ready") or 0
    )
    missed_market_move_incident = {
        "schema_version": "missed_market_move_incident_v1",
        "observed_at": now.isoformat(),
        "active": bool(
            market_opportunity_intensity["level"] in {"HIGH", "EXTREME"}
            and market_opportunity_intensity["breadth"]["1h"] >= 0.60
            and entry_ready_count == 0
        ),
        "market_opportunity_intensity": market_opportunity_intensity["level"],
        "breadth_1h": market_opportunity_intensity["breadth"]["1h"],
        "entry_ready_count": entry_ready_count,
        "dominant_blocker": execution_funnel.get("dominant_blocker"),
        "blocker_counts": execution_funnel.get("blocker_counts", {}),
        "data_health": data_health["status"],
        "automatic_threshold_relaxation": False,
        "forced_trade": False,
    }
    live_dispositions = _safe_read(
        settings.paths.output_dir
        / "live"
        / "entry_ready_dispositions_status.json"
    )
    stage_counts = dict(execution_funnel.get("stage_counts") or {})
    valid_setups = int(stage_counts.get("strategy_setups") or 0)
    near_entry_count = int(stage_counts.get("near_entry") or 0)
    submitted_count = int(
        live_dispositions.get("order_submitted_count")
        or stage_counts.get("orders_submitted")
        or 0
    )
    opportunity_capture = {
        "schema_version": "opportunity_capture_dashboard_v1",
        "observed_at": now.isoformat(),
        "market_opportunity_intensity": market_opportunity_intensity,
        "data_health": data_health,
        "current_funnel": stage_counts,
        "valid_signal_count": valid_setups,
        "near_entry_count": near_entry_count,
        "entry_ready_count": entry_ready_count,
        "order_intent_count": sum(
            str(row.get("disposition") or "")
            in {
                "ORDER_SUBMITTED",
                "ORDER_PARTIALLY_FILLED",
                "FILLED",
            }
            for row in live_dispositions.get("rows") or []
        ),
        "order_submitted_count": submitted_count,
        "fill_count": int(
            (execution or {}).get("fills_verified_this_cycle") or 0
        ),
        "conversion": {
            "setup_to_near_entry": (
                near_entry_count / valid_setups if valid_setups else None
            ),
            "entry_ready_per_near_entry_observation_ratio": (
                entry_ready_count / near_entry_count
                if near_entry_count
                else None
            ),
            "ready_to_order": (
                submitted_count / entry_ready_count
                if entry_ready_count
                else None
            ),
        },
        "conversion_metric_semantics": {
            "populations_nested": False,
            "operational_decision_eligible": False,
            "reason": (
                "near-entry and entry-ready observations are not guaranteed "
                "to represent the same nested cohort"
            ),
        },
        "leakage": {
            "data": int(execution_funnel.get("rejected_by_freshness") or 0),
            "authority": int(
                execution_funnel.get("rejected_by_live_authority") or 0
            ),
            "trigger": int(
                (execution_funnel.get("blocker_counts") or {}).get(
                    "ENTRY_TRIGGER_NOT_CONFIRMED_ON_LATEST_CLOSED_CANDLE", 0
                )
            ),
            "execution_unexplained": int(
                live_dispositions.get("unexplained_no_submit_count") or 0
            ),
        },
        "decision_latency": {
            "status": "COLLECTING_FROM_EVENT_DISPOSITION_LEDGER",
            "market_event_to_detection_ms": None,
            "trigger_to_intent_ms": None,
            "intent_to_submit_ms": None,
            "submit_to_ack_ms": None,
            "ack_to_fill_ms": None,
        },
        "missed_market_move_incident": missed_market_move_incident,
        "forced_trade": False,
    }
    status, status_reason = _scan_runtime_status(
        orders_submitted=orders_submitted,
        actionable_count=len(actionable),
        early_move_count=len(early_move_alerts),
        execution_delegated=execute,
    )
    if notify:
        telegram = await asyncio.to_thread(
            _notify_regime_change,
            settings,
            previous_regime=previous_regime,
            current_regime=str(macro["regime"]),
            markets=markets,
        )
        telegram_opportunities = await asyncio.to_thread(
            _notify_opportunity_update,
            settings,
            opportunities=(
                *actionable[:3],
                *early_move_alerts[:3],
                *near_entry[:3],
            ),
            markets=markets,
        )
    else:
        telegram = {
            "delivery_status": "SKIPPED_AUDIT_NO_NOTIFY",
        }
        telegram_opportunities = {
            "delivery_status": "SKIPPED_AUDIT_NO_NOTIFY",
        }
    payload = {
        "schema_version": ACTIVE_TRADING_VERSION,
        "scan_started_at": scan_started_at.isoformat(),
        "generated_at": now.isoformat(),
        "scan_interval_minutes": int(
            settings.autonomous_live.active_trading_scan_minutes
        ),
        "scan_poll_seconds": int(
            settings.autonomous_live.active_trading_poll_seconds
        ),
        "scan_maximum_rows": int(maximum_rows),
        "notifications_enabled": bool(notify),
        "status": status,
        "reason": status_reason,
        "markets_scanned": list(markets),
        "market_count": len(markets),
        "universe_tiers": tiered_universe.get("counts", {}),
        "live_executable_markets": sorted(live_markets),
        "scan_universe_expands_execution_authority": False,
        "timeframes_scanned": list(OBSERVED_TIMEFRAMES),
        "strategy_catalogue": {
            "strategy_count_15m": tactical_catalogue_payload()[
                "strategy_count_15m"
            ],
            "strategy_count_1h": tactical_catalogue_payload()[
                "strategy_count_1h"
            ],
            "strategy_count_2h": tactical_catalogue_payload()[
                "strategy_count_2h"
            ],
            "independent_families_15m": tactical_catalogue_payload()[
                "independent_families_15m"
            ],
            "independent_families_1h": tactical_catalogue_payload()[
                "independent_families_1h"
            ],
            "independent_families_2h": tactical_catalogue_payload()[
                "independent_families_2h"
            ],
            "forward_only_gex_orderflow_strategies": len(
                market_mechanics_strategy_specs()
            ),
            "early_move_detection_formulas": 1,
        },
        "evaluations": evaluations,
        "fast_data_recovery": fast_data_recovery,
        "data_health": data_health,
        "market_opportunity_intensity": market_opportunity_intensity,
        "missed_market_move_incident": missed_market_move_incident,
        "opportunity_capture": opportunity_capture,
        "forward_evidence": forward_evidence,
        "frame_failures": {
            f"{market}:{timeframe}": reason
            for (market, timeframe), reason in frame_failures.items()
        },
        "feature_failures": {
            f"{market}:{timeframe}": reason
            for (market, timeframe), reason in feature_failures.items()
        },
        "macro": macro,
        "market_mechanics": {
            "status": (market_mechanics.get("orderflow") or {}).get(
                "status"
            ),
            "status_1h": (market_mechanics.get("orderflow") or {}).get(
                "status"
            ),
            "status_15m": (
                market_mechanics.get("orderflow_15m") or {}
            ).get("status"),
            "btc_gex": market_mechanics.get("btc_gex"),
            "eth_gex": market_mechanics.get("eth_gex"),
            "strategy_evaluations": mechanics_evaluations,
            "historical_backfill_used": False,
        },
        "top_5_actionable": actionable[:5],
        "top_5_near_entry": near_entry[:5],
        "top_5_early_moves": early_move_alerts[:5],
        "top_5_rotation": rotation[:5],
        "capital_utilization": capital,
        "proactive_allocation": allocation,
        "tao_inventory": tao,
        "timeframe_status": timeframes,
        "execution": execution,
        "execution_funnel": execution_funnel,
        "telegram_regime_update": telegram,
        "telegram_opportunity_update": telegram_opportunities,
        "execution_delegated_to_canonical_live_engine": execute,
        "new_tactical_dna_live_authority_granted": False,
        "orders_generated": orders_generated,
        "orders_submitted": orders_submitted,
        "artifacts": {key: str(path) for key, path in files.items()},
        "tiered_universe_artifact": str(
            settings.paths.output_dir
            / "universe"
            / "tiered_trading_universe.json"
        ),
    }
    atomic_write_json(files["status"], payload)
    atomic_write_json(files["data_health"], data_health)
    atomic_write_json(
        files["market_opportunity_intensity"],
        market_opportunity_intensity,
    )
    atomic_write_json(
        files["missed_market_move_incident"],
        missed_market_move_incident,
    )
    atomic_write_json(files["opportunity_capture"], opportunity_capture)
    append_jsonl(output / "scan_funnel.jsonl", {
        "schema_version": "active_scan_funnel_event_v1",
        "scan_started_at": scan_started_at.isoformat(),
        "generated_at": now.isoformat(),
        "scan_interval_minutes": payload["scan_interval_minutes"],
        "market_count": payload["market_count"],
        "evaluations": evaluations,
        "execution_funnel": execution_funnel,
        "orders_generated": orders_generated,
        "orders_submitted": orders_submitted,
    })
    return payload


def _scan_runtime_status(
    *,
    orders_submitted: int,
    actionable_count: int,
    early_move_count: int,
    execution_delegated: bool,
) -> tuple[str, str]:
    """Describe scan evidence without implying unavailable live authority."""

    if orders_submitted > 0:
        return "CANONICAL_ORDER_SUBMITTED", "NATURAL_APPROVED_ENTRY_SUBMITTED"
    if actionable_count > 0:
        return (
            (
                "ACTIONABLE_SIGNAL_DELEGATED_NO_ORDER"
                if execution_delegated
                else "ACTIONABLE_SIGNAL_OBSERVED_NO_EXECUTION"
            ),
            (
                "CANONICAL_ENGINE_DID_NOT_AUTHORIZE_OR_SUBMIT"
                if execution_delegated
                else "TACTICAL_SIGNAL_HAS_NO_EXECUTION_DELEGATION"
            ),
        )
    if early_move_count > 0:
        return (
            "EARLY_MOVE_OBSERVED_NO_EXECUTION",
            "EARLY_MOVE_OBSERVED_NO_EXECUTION_AUTHORITY",
        )
    return "NO_CURRENT_ENTRY", "NO_VALID_ENTRY_AFTER_FULL_SCAN"


def active_trading_status(settings: Settings) -> dict[str, Any]:
    path = settings.paths.output_dir / "active_trading" / "status.json"
    payload = _safe_read(path)
    if payload:
        raw_status = str(payload.get("status") or "")
        if raw_status.startswith("LIVE_ACTIVE") and int(
            payload.get("orders_submitted") or 0
        ) == 0:
            normalized, reason = _scan_runtime_status(
                orders_submitted=0,
                actionable_count=len(payload.get("top_5_actionable") or []),
                early_move_count=len(payload.get("top_5_early_moves") or []),
                execution_delegated=bool(
                    payload.get("execution_delegated_to_canonical_live_engine")
                ),
            )
            return {
                **payload,
                "raw_writer_status": raw_status,
                "status": normalized,
                "reason": reason,
                "status_semantics_normalized_on_read": True,
            }
        return payload
    return {
        "schema_version": ACTIVE_TRADING_VERSION,
        "status": "NOT_SCANNED",
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def opportunity_status(settings: Settings) -> dict[str, Any]:
    path = settings.paths.output_dir / "active_trading" / "opportunities.json"
    return _safe_read(path) or {
        "status": "NOT_SCANNED",
        "top_5_actionable": [],
        "top_5_near_entry": [],
        "top_5_early_moves": [],
        "all": [],
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _notify_regime_change(
    settings: Settings,
    *,
    previous_regime: str | None,
    current_regime: str,
    markets: Sequence[str],
) -> dict[str, Any]:
    if not previous_regime or previous_regime == current_regime:
        return {
            "delivery_status": "SKIPPED_DUPLICATE",
            "reason_code": "REGIME_UNCHANGED",
        }
    try:
        from notifications.telegram import TelegramNotifier

        return TelegramNotifier(
            settings.telegram,
            output_directory=(
                settings.paths.output_dir / "notifications"
            ),
            allowed_markets=markets,
        ).notify_system_event(
            "REGIME_CHANGED",
            {
                "status": current_regime,
                "reason": f"{previous_regime} -> {current_regime}",
                "mode": "ACTIVE_TRADING",
            },
        )
    except Exception as exc:
        return {
            "delivery_status": "FAILED_ISOLATED",
            "reason_code": type(exc).__name__,
        }


def _notify_opportunity_update(
    settings: Settings,
    *,
    opportunities: Sequence[Mapping[str, Any]],
    markets: Sequence[str],
) -> dict[str, Any]:
    """Notify material tactical changes without granting order authority."""

    try:
        from notifications.telegram import TelegramNotifier

        return TelegramNotifier(
            settings.telegram,
            output_directory=(
                settings.paths.output_dir / "notifications"
            ),
            allowed_markets=markets,
        ).notify_opportunity_update(opportunities)
    except Exception as exc:
        return {
            "delivery_status": "FAILED_ISOLATED",
            "reason_code": type(exc).__name__,
            "orders_generated": 0,
            "orders_submitted": 0,
        }


__all__ = [
    "ACTIVE_TRADING_VERSION",
    "active_trading_status",
    "build_capital_utilization",
    "build_crypto_macro_snapshot",
    "build_lower_timeframe_candidate_queue",
    "build_rotation_ranking",
    "build_tao_inventory_policy",
    "opportunity_status",
    "refresh_public_macro_context",
    "scan_all",
]
