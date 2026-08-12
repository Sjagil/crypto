"""Forward-only 4h-regime, 2h-setup and 15m/1h flow strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.common import stable_hash

ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class MarketMechanicsStrategySpec:
    strategy_id: str
    family: str
    mechanism: str
    gex_policy: str
    market_scope: str
    stop_atr: float
    target_1_atr: float
    target_2_atr: float
    expected_holding_period: str
    entry_timeframe: str = "1h"

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "engine_version": ENGINE_VERSION,
                **asdict(self),
                "regime_timeframe": "4h",
                "setup_timeframe": "2h",
                "entry_timeframe": self.entry_timeframe,
                "closed_candle_only": True,
                "next_open_execution": True,
                "prospective_orderflow_only": True,
                "long_only_spot": True,
            },
            length=64,
        )


def market_mechanics_strategy_specs() -> tuple[MarketMechanicsStrategySpec, ...]:
    return (
        MarketMechanicsStrategySpec(
            "GEX_FLOW_NEGATIVE_BREAKOUT_4H2H1H",
            "NEGATIVE_GEX_BREAKOUT",
            "negative_gex_breakout",
            "NEGATIVE_GEX",
            "ALL_LIQUID_MARKETS_WITH_BTC_PROXY",
            2.5,
            3.75,
            6.0,
            "6h-4d",
        ),
        MarketMechanicsStrategySpec(
            "GEX_FLOW_POSITIVE_SUPPORT_RECLAIM_4H2H1H",
            "POSITIVE_GEX_SUPPORT_RECLAIM",
            "positive_gex_support_reclaim",
            "POSITIVE_GEX",
            "BTC_ETH_OR_ALT_WITH_BTC_PROXY",
            1.75,
            2.5,
            4.0,
            "3h-2d",
        ),
        MarketMechanicsStrategySpec(
            "GEX_FLOW_GAMMA_FLIP_RECLAIM_4H2H1H",
            "GAMMA_FLIP_RECLAIM",
            "gamma_flip_reclaim",
            "ANY_FRESH_GEX",
            "BTC_ETH_ONLY",
            2.0,
            3.0,
            5.0,
            "4h-3d",
        ),
        MarketMechanicsStrategySpec(
            "FLOW_TREND_PULLBACK_4H2H1H",
            "ORDERFLOW_TREND_PULLBACK",
            "orderflow_trend_pullback",
            "ANY_FRESH_GEX",
            "ALL_LIQUID_MARKETS_WITH_BTC_PROXY",
            2.0,
            3.0,
            5.0,
            "4h-3d",
        ),
        MarketMechanicsStrategySpec(
            "FLOW_ALT_RELATIVE_STRENGTH_4H2H1H",
            "ALTCOIN_RELATIVE_STRENGTH_CONTINUATION",
            "altcoin_relative_strength",
            "NOT_EXTREME_NEGATIVE_GEX",
            "ALTCOINS_ONLY_WITH_BTC_PROXY",
            2.25,
            3.5,
            5.5,
            "6h-4d",
        ),
        MarketMechanicsStrategySpec(
            "GEX_FLOW_NEGATIVE_BREAKOUT_4H2H15M",
            "NEGATIVE_GEX_BREAKOUT",
            "negative_gex_breakout",
            "NEGATIVE_GEX",
            "ALL_LIQUID_MARKETS_WITH_BTC_PROXY",
            2.25,
            3.5,
            5.0,
            "2h-2d",
            "15m",
        ),
        MarketMechanicsStrategySpec(
            "GEX_FLOW_POSITIVE_SUPPORT_RECLAIM_4H2H15M",
            "POSITIVE_GEX_SUPPORT_RECLAIM",
            "positive_gex_support_reclaim",
            "POSITIVE_GEX",
            "BTC_ETH_OR_ALT_WITH_BTC_PROXY",
            1.5,
            2.25,
            3.5,
            "1h-18h",
            "15m",
        ),
        MarketMechanicsStrategySpec(
            "GEX_FLOW_GAMMA_FLIP_RECLAIM_4H2H15M",
            "GAMMA_FLIP_RECLAIM",
            "gamma_flip_reclaim",
            "ANY_FRESH_GEX",
            "BTC_ETH_ONLY",
            1.75,
            2.75,
            4.0,
            "1h-24h",
            "15m",
        ),
        MarketMechanicsStrategySpec(
            "FLOW_TREND_PULLBACK_4H2H15M",
            "ORDERFLOW_TREND_PULLBACK",
            "orderflow_trend_pullback",
            "ANY_FRESH_GEX",
            "ALL_LIQUID_MARKETS_WITH_BTC_PROXY",
            1.75,
            2.75,
            4.0,
            "1h-24h",
            "15m",
        ),
        MarketMechanicsStrategySpec(
            "FLOW_ALT_RELATIVE_STRENGTH_4H2H15M",
            "ALTCOIN_RELATIVE_STRENGTH_CONTINUATION",
            "altcoin_relative_strength",
            "NOT_EXTREME_NEGATIVE_GEX",
            "ALTCOINS_ONLY_WITH_BTC_PROXY",
            2.0,
            3.0,
            4.5,
            "2h-2d",
            "15m",
        ),
    )


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _series(frame: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    if key not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").fillna(default)


def _boolean(frame: pd.DataFrame, key: str) -> pd.Series:
    if key not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[key].astype("boolean").fillna(False).astype(bool)


def _ema(frame: pd.DataFrame, span: int) -> pd.Series:
    """Use an existing causal EMA or derive it from closed candles."""

    key = f"ema_{span}"
    if key in frame:
        return _series(frame, key)
    return _series(frame, "close").ewm(span=span, adjust=False).mean()


def evaluate_market_mechanics_strategy(
    spec: MarketMechanicsStrategySpec,
    *,
    market: str,
    one_hour: pd.DataFrame,
    two_hour: pd.DataFrame,
    four_hour: pd.DataFrame,
    mechanics: Mapping[str, Any],
    fifteen_minute: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate one strategy without inventing unavailable flow observations."""

    market = market.upper()
    entry_frame = fifteen_minute if spec.entry_timeframe == "15m" else one_hour
    if (
        entry_frame is None
        or entry_frame.empty
        or two_hour.empty
        or four_hour.empty
    ):
        return {
            "strategy_id": spec.strategy_id,
            "strategy_dna_hash": spec.dna_hash,
            "family": spec.family,
            "market": market,
            "status": "DATA_PENDING",
            "actionable": False,
            "score": 0.0,
            "blockers": ["MULTI_TIMEFRAME_DATA_MISSING"],
            "historical_backtest_status": "PROSPECTIVE_DATA_ACCUMULATING",
            "live_authority_granted": False,
            "operator_dna_approval_required": True,
        }
    gex = dict(mechanics.get("gex") or {})
    flow_key = (
        "orderflow_15m"
        if spec.entry_timeframe == "15m"
        else "orderflow"
    )
    raw_flow = mechanics.get(flow_key)
    # Preserve compatibility with older research fixtures, but never fall
    # back to hourly data when a live snapshot explicitly reports that its
    # 15-minute bucket is pending or incomplete.
    legacy_hourly_flow = (
        raw_flow is None and spec.entry_timeframe == "15m"
    )
    if legacy_hourly_flow:
        raw_flow = mechanics.get("orderflow")
    flow = dict(raw_flow or {})
    scope_blocked = (
        spec.market_scope == "BTC_ETH_ONLY" and market not in {"BTC-EUR", "ETH-EUR"}
    ) or (
        spec.market_scope == "ALTCOINS_ONLY_WITH_BTC_PROXY"
        and market in {"BTC-EUR", "ETH-EUR"}
    )
    data_ready = gex.get("fresh") is True and flow.get("status") == "READY"
    close_entry = _series(entry_frame, "close")
    close_2h = _series(two_hour, "close")
    close_4h = _series(four_hour, "close")
    ema20_entry = _ema(entry_frame, 20)
    ema20_2h = _ema(two_hour, 20)
    ema50_2h = _ema(two_hour, 50)
    ema50_4h = _ema(four_hour, 50)
    relative_volume = _series(entry_frame, "relative_volume_20")
    relative_strength = _series(entry_frame, "btc_relative_momentum_20")
    atr = max(0.0, _number(entry_frame.iloc[-1], "atr_14"))
    if atr <= 0.0:
        true_range = pd.concat(
            [
                (_series(entry_frame, "high") - _series(entry_frame, "low")).abs(),
                (_series(entry_frame, "high") - close_entry.shift(1)).abs(),
                (_series(entry_frame, "low") - close_entry.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = max(0.0, float(true_range.rolling(14, min_periods=1).mean().iloc[-1]))
    trend_4h = bool(close_4h.iloc[-1] > ema50_4h.iloc[-1])
    setup_2h = bool(close_2h.iloc[-1] >= ema20_2h.iloc[-1] * 0.985)
    cvd_z = _number(flow, "spot_cvd_robust_zscore")
    flow_horizon = (
        "15m"
        if spec.entry_timeframe == "15m" and not legacy_hourly_flow
        else "1h"
    )
    selected_horizon = dict(
        (flow.get("horizons") or {}).get(flow_horizon) or {}
    )
    cvd_slope = _number(selected_horizon, "cvd_slope_base_per_hour")
    ofi = _number(selected_horizon, "ofi_normalized_mean")
    obi = _number(flow, "orderbook_imbalance_top_10")
    absorption = _number(flow, "bullish_absorption_score")
    spread = _number(flow, "spread_bps", default=999.0)
    gex_regime = str(gex.get("regime") or "GEX_DATA_PENDING")
    normalized_gex = _number(gex, "normalized_signed_gex")
    gex_policy_match = (
        spec.gex_policy == "ANY_FRESH_GEX"
        or gex_regime == spec.gex_policy
        or (
            spec.gex_policy == "NOT_EXTREME_NEGATIVE_GEX"
            and normalized_gex > -0.35
        )
    )
    donchian = _series(entry_frame, "donchian_high_20", default=np.nan)
    crossed_donchian = bool(
        len(entry_frame) >= 2
        and close_entry.iloc[-2] <= donchian.iloc[-2]
        and close_entry.iloc[-1] > donchian.iloc[-1]
    )
    vwap_reclaim = bool(_boolean(entry_frame, "vwap_reclaim").iloc[-1])
    gamma_flip = gex.get("gamma_flip")
    flip_reclaim = bool(
        gamma_flip is not None
        and len(close_entry) >= 2
        and close_entry.iloc[-2] <= float(gamma_flip)
        and close_entry.iloc[-1] > float(gamma_flip)
    )
    price_trigger = False
    if spec.mechanism == "negative_gex_breakout":
        price_trigger = crossed_donchian and relative_volume.iloc[-1] >= 1.4
    elif spec.mechanism == "positive_gex_support_reclaim":
        price_trigger = vwap_reclaim and close_2h.iloc[-1] > ema50_2h.iloc[-1]
    elif spec.mechanism == "gamma_flip_reclaim":
        price_trigger = flip_reclaim and setup_2h
    elif spec.mechanism == "orderflow_trend_pullback":
        price_trigger = (
            close_entry.iloc[-1] > ema20_entry.iloc[-1]
            and setup_2h
            and absorption >= 0.20
        )
    elif spec.mechanism == "altcoin_relative_strength":
        price_trigger = (
            relative_strength.iloc[-1] > 0.0
            and crossed_donchian
            and relative_volume.iloc[-1] >= 1.1
        )
    components = {
        "trend_4h": 1.0 if trend_4h else 0.0,
        "gex_regime": 1.0 if gex_policy_match else 0.0,
        "relative_strength": float(
            np.clip(0.5 + relative_strength.iloc[-1] * 10.0, 0.0, 1.0)
        ),
        "cvd": float(np.clip(0.5 + 0.20 * cvd_z + 0.25 * np.sign(cvd_slope), 0.0, 1.0)),
        "ofi": float(np.clip(0.5 + ofi, 0.0, 1.0)),
        "absorption": float(np.clip(absorption, 0.0, 1.0)),
        "obi": float(np.clip(0.5 + obi, 0.0, 1.0)),
        "liquidity_quality": float(np.clip(1.0 - spread / 20.0, 0.0, 1.0)),
    }
    score = (
        0.20 * components["trend_4h"]
        + 0.15 * components["gex_regime"]
        + 0.15 * components["relative_strength"]
        + 0.15 * components["cvd"]
        + 0.15 * components["ofi"]
        + 0.10 * components["absorption"]
        + 0.05 * components["obi"]
        + 0.05 * components["liquidity_quality"]
    )
    orderflow_confirmed = cvd_slope > 0.0 and ofi > 0.0 and spread <= 15.0
    confirmed = bool(
        not scope_blocked
        and data_ready
        and trend_4h
        and setup_2h
        and gex_policy_match
        and price_trigger
        and orderflow_confirmed
        and score >= 0.65
    )
    blockers: list[str] = []
    if scope_blocked:
        blockers.append("MARKET_OUTSIDE_STRATEGY_SCOPE")
    if not gex.get("fresh"):
        blockers.append("GEX_NOT_FRESH")
    if flow.get("status") != "READY":
        blockers.append("PROSPECTIVE_ORDERFLOW_NOT_READY")
    if not trend_4h:
        blockers.append("4H_REGIME_NOT_BULLISH")
    if not setup_2h:
        blockers.append("2H_SETUP_NOT_ACTIVE")
    if not gex_policy_match:
        blockers.append("GEX_POLICY_MISMATCH")
    if not price_trigger:
        blockers.append(f"{spec.entry_timeframe.upper()}_PRICE_TRIGGER_NOT_CONFIRMED")
    if not orderflow_confirmed:
        blockers.append("PROSPECTIVE_ORDERFLOW_NOT_CONFIRMED")
    if score < 0.65:
        blockers.append("COMPOSITE_SCORE_BELOW_0_65")
    close = float(close_entry.iloc[-1])
    return {
        "strategy_id": spec.strategy_id,
        "strategy_dna_hash": spec.dna_hash,
        "family": spec.family,
        "mechanism": spec.mechanism,
        "market": market,
        "regime_timeframe": "4h",
        "setup_timeframe": "2h",
        "entry_timeframe": spec.entry_timeframe,
        "orderflow_confirmation_horizon": flow_horizon,
        "gex_scope": mechanics.get("gex_scope"),
        "gex_regime": gex_regime,
        "score": score,
        "score_components": components,
        "status": "ACTIONABLE" if confirmed else "DATA_PENDING" if not data_ready else "WATCH",
        "actionable": confirmed,
        "entry": close,
        "stop": close - spec.stop_atr * atr,
        "target_1": close + spec.target_1_atr * atr,
        "target_2": close + spec.target_2_atr * atr,
        "expected_holding_period": spec.expected_holding_period,
        "blockers": blockers,
        "historical_backtest_status": "PROSPECTIVE_DATA_ACCUMULATING",
        "live_authority_granted": False,
        "operator_dna_approval_required": True,
        "closed_candle_only": True,
        "next_open_execution_required": True,
    }


__all__ = [
    "ENGINE_VERSION",
    "MarketMechanicsStrategySpec",
    "evaluate_market_mechanics_strategy",
    "market_mechanics_strategy_specs",
]
