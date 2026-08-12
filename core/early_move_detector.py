"""Causal early-move detection for the active 15-minute market scan.

The detector is intentionally an observation layer, not a trading strategy.
It identifies price/volume acceleration early, distinguishes a fresh impulse
from an already extended move, and publishes bounded reference levels.  A row
from this module can trigger an alert or research queue entry, but can never
grant live authority or submit an order.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.common import stable_hash

EARLY_MOVE_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class EarlyMovePolicy:
    minimum_score: float = 55.0
    minimum_confirmations: int = 4
    minimum_return_15m: float = 0.003
    minimum_return_1h: float = 0.008
    minimum_relative_volume: float = 1.25
    maximum_extension_atr: float = 2.25
    maximum_return_1h_before_pullback: float = 0.06
    maximum_return_4h_before_pullback: float = 0.12
    minimum_breadth_for_first_pullback: float = 0.50
    maximum_first_pullback_atr: float = 0.90
    stop_atr: float = 1.50
    target_1_r: float = 1.50
    target_2_r: float = 2.50

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "engine_version": EARLY_MOVE_ENGINE_VERSION,
                "policy": asdict(self),
                "closed_candle_only": True,
                "observation_only": True,
            },
            length=64,
        )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if math.isfinite(selected) else default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _flow_component(mechanics: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    flow = dict(mechanics.get("orderflow_15m") or {})
    if not flow:
        flow = dict(mechanics.get("orderflow") or {})
    synthetic = bool(flow.get("synthetic_data_used", False))
    fresh = bool(flow.get("fresh", False))
    status = str(flow.get("status") or "MISSING")
    horizons = dict(flow.get("horizons") or {})
    horizon = dict(horizons.get("15m") or horizons.get("1h") or {})
    ofi = _finite(horizon.get("ofi_normalized_mean"))
    trade_delta = _finite(horizon.get("trade_delta_percentage"))
    imbalance = _finite(flow.get("orderbook_imbalance_top_10"))
    usable = fresh and not synthetic and status == "READY"
    component = (
        float(
            np.mean(
                [
                    _clip01((ofi + 0.20) / 0.40),
                    _clip01((trade_delta + 0.20) / 0.40),
                    _clip01((imbalance + 0.20) / 0.40),
                ]
            )
        )
        if usable
        else 0.0
    )
    return component, {
        "status": status,
        "fresh": fresh,
        "synthetic_data_used": synthetic,
        "usable_for_confirmation": usable,
        "ofi_normalized": ofi if usable else None,
        "trade_delta_percentage": trade_delta if usable else None,
        "orderbook_imbalance_top_10": imbalance if usable else None,
        "component": component,
    }


def _rotation_context(
    rotation: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], float]:
    by_market = {str(row.get("market") or ""): row for row in rotation}
    btc = by_market.get("BTC-EUR", {})
    btc_return_1h = _finite((btc.get("returns") or {}).get("return_1h"))
    return by_market, btc_return_1h


def detect_early_moves(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    features: Mapping[tuple[str, str], pd.DataFrame],
    markets: Sequence[str],
    *,
    rotation: Sequence[Mapping[str, Any]],
    mechanics: Mapping[str, Any],
    regime: str,
    policy: EarlyMovePolicy | None = None,
) -> list[dict[str, Any]]:
    """Return fresh-impulse and pullback-pending observations.

    All calculations use the last fully closed 15-minute candle and strictly
    earlier baselines.  Missing orderflow contributes no positive score.
    """

    selected_policy = policy or EarlyMovePolicy()
    rotation_by_market, btc_return_1h = _rotation_context(rotation)
    breadth_inputs = [
        _finite((row.get("returns") or {}).get("return_1h"))
        for row in rotation
        if str(row.get("market") or "") != "BTC-EUR"
    ]
    positive_breadth = (
        sum(value > 0.0 for value in breadth_inputs) / len(breadth_inputs)
        if breadth_inputs
        else 0.0
    )
    mechanics_by_market = dict(mechanics.get("markets") or {})
    rows: list[dict[str, Any]] = []

    for market in markets:
        frame = frames.get((market, "15m"))
        feature_frame = features.get((market, "15m"))
        if frame is None or feature_frame is None or len(frame) < 100:
            continue
        source = frame.iloc[-100:].copy()
        close = pd.to_numeric(source["close"], errors="coerce")
        high = pd.to_numeric(source["high"], errors="coerce")
        low = pd.to_numeric(source["low"], errors="coerce")
        volume = pd.to_numeric(source["volume"], errors="coerce")
        if close.tail(22).isna().any() or volume.tail(22).isna().any():
            continue

        current = float(close.iloc[-1])
        if current <= 0.0:
            continue
        return_15m = current / float(close.iloc[-2]) - 1.0
        return_30m = current / float(close.iloc[-3]) - 1.0
        return_1h = current / float(close.iloc[-5]) - 1.0
        earlier_returns = close.pct_change().iloc[-5:-1]
        acceleration = return_15m - float(earlier_returns.mean())
        prior_volume = volume.iloc[-21:-1]
        volume_median = float(prior_volume.median())
        relative_volume = (
            float(volume.iloc[-1]) / volume_median
            if volume_median > 0.0
            else 0.0
        )
        volume_mad = float((prior_volume - volume_median).abs().median())
        volume_robust_z = (
            (float(volume.iloc[-1]) - volume_median)
            / max(1e-12, 1.4826 * volume_mad)
        )
        prior_high = float(high.iloc[-21:-1].max())
        breakout_distance = current / prior_high - 1.0 if prior_high > 0.0 else -1.0
        candle_range = float(high.iloc[-1] - low.iloc[-1])
        close_location = (
            float((current - low.iloc[-1]) / candle_range)
            if candle_range > 0.0
            else 0.5
        )
        latest_features = feature_frame.iloc[-1]
        atr = max(_finite(latest_features.get("atr_14")), current * 0.0025)
        ema_20 = _finite(latest_features.get("ema_20"), current)
        extension_atr = (current - ema_20) / atr if atr > 0.0 else 0.0
        four_hour = frames.get((market, "4h"))
        return_4h = 0.0
        if four_hour is not None and len(four_hour) >= 2:
            four_hour_close = pd.to_numeric(four_hour["close"], errors="coerce")
            if four_hour_close.tail(2).notna().all():
                return_4h = float(
                    four_hour_close.iloc[-1] / four_hour_close.iloc[-2] - 1.0
                )
        rotation_row = rotation_by_market.get(market, {})
        rotation_score = _finite(rotation_row.get("rotation_score"), 50.0)
        relative_strength_1h = return_1h - btc_return_1h
        flow_score, flow = _flow_component(
            dict(mechanics_by_market.get(market) or {})
        )

        components = {
            "return_15m": _clip01(return_15m / 0.012),
            "return_1h": _clip01(return_1h / 0.040),
            "acceleration": _clip01(acceleration / 0.010),
            "relative_volume": _clip01((relative_volume - 1.0) / 1.50),
            "breakout_proximity": _clip01(1.0 + breakout_distance / 0.02),
            "close_location": _clip01(close_location),
            "btc_relative_strength": _clip01(
                (relative_strength_1h + 0.005) / 0.030
            ),
            "prospective_orderflow": flow_score,
        }
        score = 100.0 * (
            0.18 * components["return_15m"]
            + 0.14 * components["return_1h"]
            + 0.15 * components["acceleration"]
            + 0.19 * components["relative_volume"]
            + 0.12 * components["breakout_proximity"]
            + 0.08 * components["close_location"]
            + 0.08 * components["btc_relative_strength"]
            + 0.06 * components["prospective_orderflow"]
        )
        confirmations = {
            "positive_15m_impulse": return_15m >= selected_policy.minimum_return_15m,
            "positive_1h_impulse": return_1h >= selected_policy.minimum_return_1h,
            "positive_acceleration": acceleration >= 0.0015,
            "relative_volume_expansion": (
                relative_volume >= selected_policy.minimum_relative_volume
            ),
            "near_or_above_20_bar_high": breakout_distance >= -0.001,
            "strong_candle_close": close_location >= 0.70,
            "btc_relative_strength_positive": relative_strength_1h > 0.0,
            "prospective_buy_flow": (
                bool(flow["usable_for_confirmation"]) and flow_score >= 0.55
            ),
        }
        confirmation_count = sum(confirmations.values())
        prior_return_15m = float(close.iloc[-2] / close.iloc[-3] - 1.0)
        prior_return_1h = float(close.iloc[-2] / close.iloc[-6] - 1.0)
        prior_volume_baseline = float(volume.iloc[-22:-2].median())
        prior_relative_volume = (
            float(volume.iloc[-2]) / prior_volume_baseline
            if prior_volume_baseline > 0.0
            else 0.0
        )
        prior_range = float(high.iloc[-2] - low.iloc[-2])
        prior_close_location = (
            float((close.iloc[-2] - low.iloc[-2]) / prior_range)
            if prior_range > 0.0
            else 0.5
        )
        pullback_depth_atr = (
            float((close.iloc[-2] - current) / atr) if atr > 0.0 else 0.0
        )
        first_pullback = (
            prior_return_15m >= selected_policy.minimum_return_15m
            and prior_return_1h >= selected_policy.minimum_return_1h
            and prior_relative_volume >= selected_policy.minimum_relative_volume
            and prior_close_location >= 0.65
            and return_15m <= 0.002
            and pullback_depth_atr >= 0.0
            and pullback_depth_atr <= selected_policy.maximum_first_pullback_atr
            and current > ema_20
            and positive_breadth
            >= selected_policy.minimum_breadth_for_first_pullback
            and regime.upper()
            in {"RECOVERY", "BROAD_RISK_ON", "BTC_BULL", "ALT_BULL"}
        )
        detected = (
            return_15m > 0.0
            and score >= selected_policy.minimum_score
            and confirmation_count >= selected_policy.minimum_confirmations
        )
        if not detected and not first_pullback:
            continue
        extended = (
            extension_atr >= selected_policy.maximum_extension_atr
            or return_1h >= selected_policy.maximum_return_1h_before_pullback
            or return_4h >= selected_policy.maximum_return_4h_before_pullback
        )
        status = (
            "FIRST_PULLBACK_AFTER_IMPULSE"
            if first_pullback
            else "EXTENDED_MOVE_WAIT_FOR_PULLBACK"
            if extended
            else "EARLY_MOMENTUM_ALERT"
        )
        if first_pullback:
            # Rank the quality of the completed impulse and the controlled
            # pullback; realtime flow still has to trigger the actual entry.
            score = max(
                score,
                100.0
                * (
                    0.30 * _clip01(prior_return_15m / 0.012)
                    + 0.25 * _clip01(prior_return_1h / 0.04)
                    + 0.20 * _clip01((prior_relative_volume - 1.0) / 1.5)
                    + 0.15 * _clip01(prior_close_location)
                    + 0.10 * _clip01(positive_breadth)
                ),
            )
        entry_reference = (
            max(ema_20, current - 0.75 * atr)
            if extended or first_pullback
            else max(prior_high, current)
        )
        risk_distance = max(selected_policy.stop_atr * atr, entry_reference * 0.005)
        stop = max(1e-12, entry_reference - risk_distance)
        signal_timestamp = frame.index[-1].isoformat()
        rows.append(
            {
                "opportunity_id": stable_hash(
                    [selected_policy.dna_hash, market, signal_timestamp, status],
                    length=32,
                ),
                "rank": 0,
                "market": market,
                "strategy": "EARLY_MOVE_VOLUME_FLOW_15M",
                "strategy_dna_hash": selected_policy.dna_hash,
                "family": (
                    "FIRST_PULLBACK_AFTER_IMPULSE"
                    if first_pullback
                    else "EARLY_MOVE_DETECTION"
                ),
                "timeframe": "15m",
                "entry_timeframe": "15m",
                "confirmation_timeframe": "1h",
                "regime_timeframe": "4h",
                "regime": regime,
                "regime_policy": "OBSERVE_ONLY",
                "status": status,
                "entry_trigger_confirmed": False,
                "score": score,
                "confidence": min(100.0, score + 2.5 * confirmation_count),
                "current_price": current,
                "entry_zone": [entry_reference * 0.998, entry_reference * 1.002],
                "trigger": prior_high,
                "trigger_reason": "15M_PRICE_VOLUME_ACCELERATION",
                "stop": stop,
                "target_1": entry_reference
                + selected_policy.target_1_r * risk_distance,
                "target_2": entry_reference
                + selected_policy.target_2_r * risk_distance,
                "expected_holding_period": "30m-24h",
                "estimated_fee_fraction": 0.0025,
                "estimated_slippage_bps": 8.0,
                "liquidity_status": "PRECHECK_REQUIRED_AT_ORDER_TIME",
                "spread_status": "PRECHECK_REQUIRED_AT_ORDER_TIME",
                "timeframe_alignment_score": components["btc_relative_strength"],
                "timeframe_conflicts": (
                    ["MOVE_EXTENDED_ABOVE_ATR_ENVELOPE"] if extended else []
                ),
                "rotation_score": rotation_score,
                "distance_to_trigger": breakout_distance,
                "signal_timestamp": signal_timestamp,
                "reason_not_yet_entered": (
                    "REALTIME_RECLAIM_AND_NON_HOSTILE_FLOW_REQUIRED"
                    if first_pullback
                    else
                    "MOVE_EXTENDED_WAIT_FOR_PULLBACK_AND_NEW_CLOSED_CANDLE"
                    if extended
                    else "EARLY_ALERT_REQUIRES_CLOSED_CANDLE_ENTRY_CONFIRMATION"
                ),
                "formula_version": EARLY_MOVE_ENGINE_VERSION,
                "formula": {
                    "score_components": components,
                    "confirmations": confirmations,
                    "confirmation_count": confirmation_count,
                    "return_15m": return_15m,
                    "return_30m": return_30m,
                    "return_1h": return_1h,
                    "return_4h": return_4h,
                    "momentum_acceleration": acceleration,
                    "relative_volume_20": relative_volume,
                    "volume_robust_zscore": volume_robust_z,
                    "breakout_distance": breakout_distance,
                    "close_location": close_location,
                    "extension_atr": extension_atr,
                    "prior_return_15m": prior_return_15m,
                    "prior_return_1h": prior_return_1h,
                    "prior_relative_volume": prior_relative_volume,
                    "prior_close_location": prior_close_location,
                    "first_pullback_depth_atr": pullback_depth_atr,
                    "positive_altcoin_breadth_1h": positive_breadth,
                    "btc_relative_strength_1h": relative_strength_1h,
                    "orderflow": flow,
                },
                "closed_candle_only": True,
                "setup_valid_on_closed_candle": first_pullback,
                "next_open_execution_required": True,
                "historical_backtest_status": "REQUIRES_PREREGISTERED_VALIDATION",
                "execution_scope": (
                    "EXISTING_BREAKOUT_PULLBACK_FAMILY_PRECHECK_REQUIRED"
                    if first_pullback
                    else "ALERT_ONLY_NEW_DNA_REQUIRES_APPROVAL"
                ),
                "live_authority_granted": False,
                "operator_dna_approval_required": True,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


__all__ = [
    "EARLY_MOVE_ENGINE_VERSION",
    "EarlyMovePolicy",
    "detect_early_moves",
]
