"""Causal event-driven spot playbooks and persistent opportunity lifecycle.

This module does not submit orders.  It converts the prospective realtime
trade/book stream plus the existing multi-timeframe context into auditable
opportunities.  Execution authority remains a separate, explicit concern.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.active_swing_contracts import normalize_live_opportunity
from core.opportunity_intelligence import (
    confirmation_independence,
    estimate_roundtrip_economics,
    freeze_feature_snapshot,
)
from utils.common import atomic_write_json, stable_hash, utc_now


class OpportunityState(StrEnum):
    DISCOVERED = "DISCOVERED"
    WATCHING = "WATCHING"
    ARMED = "ARMED"
    ENTRY_READY = "ENTRY_READY"
    ORDER_INTENT_CREATED = "ORDER_INTENT_CREATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    MANAGING = "MANAGING"
    EXITING = "EXITING"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


TERMINAL_STATES = {
    OpportunityState.CLOSED,
    OpportunityState.INVALIDATED,
    OpportunityState.EXPIRED,
}
PRE_ENTRY_STATES = {
    OpportunityState.DISCOVERED,
    OpportunityState.WATCHING,
    OpportunityState.ARMED,
    OpportunityState.ENTRY_READY,
}
EXECUTION_OWNED_STATES = {
    OpportunityState.ORDER_INTENT_CREATED,
    OpportunityState.ORDER_SUBMITTED,
    OpportunityState.PARTIALLY_FILLED,
    OpportunityState.FILLED,
    OpportunityState.MANAGING,
    OpportunityState.EXITING,
    *TERMINAL_STATES,
}
COMPACTABLE_TERMINAL_STATES = {
    OpportunityState.INVALIDATED,
    OpportunityState.EXPIRED,
}
TERMINAL_TOMBSTONE_FIELDS = (
    "opportunity_id",
    "episode_id",
    "market",
    "playbook_id",
    "family",
    "playbook_dna",
    "state",
    "detected_at",
    "last_seen_at",
    "last_updated_at",
    "valid_until",
    "reason_codes",
)
ENTRY_PERSISTENCE_SECONDS = 5.0
PREFERRED_NET_SWING_UPSIDE_BPS = 300.0
NORMAL_MINIMUM_NET_TARGET_1_BPS = 125.0
NORMAL_MINIMUM_NET_TARGET_2_BPS = 225.0
TIER_A_MINIMUM_NET_TARGET_2_BPS = 160.0
MAXIMUM_COST_TO_TARGET_RATIO = 0.25
CONSERVATIVE_WIN_PROBABILITY = 0.40
MAXIMUM_BOOK_AGE_SECONDS = 10.0
MAXIMUM_EXECUTED_FLOW_AGE_SECONDS = 90.0
MINIMUM_EXECUTED_QUOTE_VOLUME_EUR = 5.0
MINIMUM_VOLUME_CONFIRMATION_TRADES = 3
MINIMUM_BOOK_PERSISTENCE_UPDATES = 3


def _swing_time_stop_minutes(context_timeframe: str | None) -> int:
    """Return a generous review horizon, not an automatic same-day exit."""

    return {
        "15m": 1_440,
        "1h": 4_320,
        "4h": 10_080,
        "1d": 43_200,
    }.get(str(context_timeframe or "").casefold(), 1_440)


def _setup_validity_minutes(context_timeframe: str | None) -> int:
    """Keep an unfilled setup alive for several closed execution bars."""

    return {
        "15m": 120,
        "1h": 480,
        "4h": 1_440,
        "1d": 4_320,
    }.get(str(context_timeframe or "").casefold(), 120)


@dataclass(frozen=True)
class PlaybookSpec:
    playbook_id: str
    family: str
    hypothesis: str
    execution_timeframes: tuple[str, ...]
    context_timeframes: tuple[str, ...]
    expected_regimes: tuple[str, ...]
    entry_logic: str
    orderflow_confirmation: str
    liquidity_rule: str
    invalidation: str
    stop_method: str
    take_profit_method: str
    trailing_method: str
    time_stop: str
    risk_policy: str
    parameter_band: Mapping[str, tuple[float, float]]

    @property
    def dna(self) -> str:
        return stable_hash(asdict(self), length=64)


PLAYBOOKS: tuple[PlaybookSpec, ...] = (
    PlaybookSpec(
        "MOMENTUM_BREAKOUT_V1",
        "MOMENTUM_BREAKOUT",
        "Fresh price acceleration with expanding participation can persist.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Break above recent structure while 1m/5m momentum is positive.",
        "Positive CVD/taker flow plus OFI or MLOBI confirmation.",
        "Spread and bounded-order slippage must remain below market caps.",
        "Acceleration, CVD and bid support reverse before entry.",
        "Structure or volatility stop below the breakout base.",
        "TP1 at 1.5R and TP2 at 2.5R with a runner when flow persists.",
        "Trail below microstructure higher lows after TP1.",
        "Review after the timeframe horizon; hold while structure remains valid.",
        "Micro-live cap; size reduced in weak macro regimes.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "BREAKOUT_PULLBACK_V1",
        "BREAKOUT_PULLBACK",
        "A confirmed breakout retest can improve entry quality and reduce chase.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Retest of a prior breakout level followed by positive acceleration.",
        "Bid replenishment and positive taker-flow reversal on the reclaim.",
        "The retest may not widen spread or exhaust visible depth.",
        "Closed price loses the breakout level or flow remains negative.",
        "Below the reclaimed structure and sweep low.",
        "TP1 at 1.5R and TP2 at the prior expansion projection.",
        "Trail behind reclaimed structure after TP1.",
        "Review after the timeframe horizon; hold while structure remains valid.",
        "Prefer reduced risk over late breakout chasing.",
        {"score": (54.0, 100.0), "pullback_pct": (0.1, 1.5)},
    ),
    PlaybookSpec(
        "VOLATILITY_EXPANSION_V1",
        "VOLATILITY_EXPANSION",
        "Range expansion plus abnormal volume can mark a new information regime.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "1m and 5m range/return expansion with RVOL and trade intensity.",
        "CVD and orderbook pressure agree with price direction.",
        "Reject expansion produced by a thin book or excessive slippage.",
        "Return compression or opposing CVD/OFI before fill.",
        "Below the expansion origin or 1.2 times recent volatility.",
        "Scale at 1.25R and 2.25R.",
        "Orderflow-aware trailing stop after the first scale.",
        "Review after the timeframe horizon; exit only on confirmed invalidation.",
        "Use volatility-proportional risk within the micro-live cap.",
        {"score": (54.0, 100.0), "rvol": (1.3, 8.0)},
    ),
    PlaybookSpec(
        "LIQUIDITY_SWEEP_RECLAIM_V1",
        "LIQUIDITY_SWEEP_RECLAIM",
        "A failed downside liquidity sweep can precede a sharp spot reversal.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("SIDEWAYS_HIGH_VOL", "CAPITULATION", "RECOVERY"),
        "Downside impulse reverses and reclaims structure with acceleration.",
        "CVD improves, ask liquidity depletes and bids replenish.",
        "Require executable spread and enough depth for a bounded exit.",
        "Price revisits the sweep low or CVD makes a new low.",
        "Below the confirmed sweep low.",
        "TP1 at range midpoint and TP2 at opposite liquidity.",
        "Trail below successive reclaimed micro-lows.",
        "Review after the timeframe horizon; exit only on confirmed invalidation.",
        "Micro-only when the higher-timeframe trend is still bearish.",
        {"score": (54.0, 100.0), "reversal_1m": (0.05, 3.0)},
    ),
    PlaybookSpec(
        "FAILED_BREAKDOWN_REVERSAL_V1",
        "FAILED_BREAKDOWN_REVERSAL",
        "A failed breakdown traps aggressive sellers and can mean-revert.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "RECOVERY"),
        "Reclaim after negative impulse with rising short-window return.",
        "Taker buy ratio and OFI flip positive after seller exhaustion.",
        "Avoid illiquid markets and unrecovered spread spikes.",
        "Second loss of the reclaimed level.",
        "Below the failed-breakdown low.",
        "TP1 at mean/VWAP and TP2 at prior range high.",
        "Trail only after the mean is reclaimed.",
        "Review after the timeframe horizon; hold if recovery structure persists.",
        "Reduced risk against a bearish 4h structure.",
        {"score": (54.0, 100.0), "flow_flip": (0.05, 1.0)},
    ),
    PlaybookSpec(
        "FAILED_BREAKOUT_REVERSAL_V1",
        "FAILED_BREAKOUT_REVERSAL",
        "A rejected first breakout followed by a demand-backed reclaim can "
        "shake out weak buyers before continuation.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "RECOVERY", "BROAD_RISK_ON"),
        "Reclaim the failed breakout level with positive short-window acceleration.",
        "Taker buying, CVD and OFI recover while asks deplete.",
        "The second reclaim must retain executable spread, depth and bounded impact.",
        "Price loses the reclaimed level again or demand flow reverses.",
        "Below the second-reclaim micro-low.",
        "TP1 at the failed-breakout high and TP2 at a 2R extension.",
        "Trail behind reclaimed structure and exit on renewed flow exhaustion.",
        "Review after the timeframe horizon; exit only if the reclaim fails.",
        "Micro-only until the second reclaim has prospective fill evidence.",
        {"score": (54.0, 100.0), "reclaim_confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "POST_LIQUIDATION_RECOVERY_V1",
        "POST_LIQUIDATION_RECOVERY",
        "Forced selling followed by flow normalization can create convex recovery.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("CAPITULATION", "RECOVERY"),
        "Sharp negative impulse stabilizes and reclaims the first micro-high.",
        "CVD divergence, bid replenishment and falling sell intensity.",
        "Depth must recover before entry; never catch a disorderly book.",
        "New low or renewed sell-flow acceleration.",
        "Below the liquidation low.",
        "TP1 at 50 percent retrace and TP2 at pre-event VWAP.",
        "Trail under recovery higher lows.",
        "Review after the timeframe horizon; exit only if recovery invalidates.",
        "Always micro-sized in capitulation.",
        {"score": (62.0, 100.0), "sell_impulse": (0.5, 8.0)},
    ),
    PlaybookSpec(
        "RELATIVE_STRENGTH_ROTATION_V1",
        "RELATIVE_STRENGTH_ROTATION",
        "Leaders that outperform BTC with confirmed participation may persist.",
        ("5m", "15m"),
        ("1h", "4h"),
        ("ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Positive 5m relative return versus BTC and valid trend context.",
        "Positive CVD and bid-side imbalance confirm genuine demand.",
        "Liquidity-adjusted score must pass before rotation.",
        "Relative strength turns negative or the leader loses structure.",
        "Below the most recent 5m swing.",
        "TP1 at 1.5R and retain a relative-strength runner.",
        "Trail when relative strength or CVD deteriorates.",
        "Review after the timeframe horizon; hold while relative strength persists.",
        "Cap correlated altcoin exposure at portfolio level.",
        {"score": (54.0, 100.0), "relative_return_5m": (0.1, 5.0)},
    ),
    PlaybookSpec(
        "LEADER_FOLLOWER_V1",
        "LEADER_FOLLOWER",
        "Lagging liquid peers can follow a causally observed leader impulse.",
        ("1m", "5m"),
        ("15m", "1h"),
        ("ALT_BULL", "BROAD_RISK_ON"),
        "Leader breaks out first; follower confirms volume before entry.",
        "Follower CVD and bid support must turn positive independently.",
        "Follower spread/depth must satisfy its own execution limits.",
        "Leader reverses or follower confirmation does not arrive.",
        "Below follower pre-confirmation structure.",
        "TP1 at 1.25R and TP2 at leader-relative projection.",
        "Trail on leader or follower flow deterioration.",
        "Expire after 10 minutes.",
        "Micro-only until live latency evidence exists.",
        {"score": (62.0, 100.0), "leader_lag_seconds": (5.0, 600.0)},
    ),
    PlaybookSpec(
        "VWAP_RECLAIM_V1",
        "VWAP_RECLAIM",
        "Reclaiming session VWAP with demand can signal intraday control change.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("SIDEWAYS_LOW_VOL", "RECOVERY", "BROAD_RISK_ON"),
        "Price reclaims causal session VWAP and holds above it.",
        "CVD, taker buy ratio and OFI confirm the reclaim.",
        "Spread and impact stay normal during the reclaim.",
        "Price closes back below VWAP with negative flow.",
        "Below VWAP reclaim low.",
        "TP1 at prior high and TP2 at 2R.",
        "Trail under VWAP or higher lows after TP1.",
        "Review after the timeframe horizon; hold while VWAP structure persists.",
        "Reduce risk when 4h trend disagrees.",
        {"score": (54.0, 100.0), "vwap_distance_pct": (0.0, 1.0)},
    ),
    PlaybookSpec(
        "TREND_PULLBACK_V1",
        "TREND_PULLBACK",
        "Pullbacks in established trends can offer asymmetric continuation entries.",
        ("5m", "15m"),
        ("1h", "4h", "1d"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON"),
        "Higher-timeframe uptrend plus pullback stabilization and reacceleration.",
        "Selling intensity fades; CVD and bid support turn positive.",
        "Reject pullbacks with structural liquidity deterioration.",
        "Higher-timeframe structure breaks.",
        "Below pullback low or causal ATR stop.",
        "TP1 at prior high and TP2 at trend extension.",
        "Trail under higher lows.",
        "Review after the timeframe horizon; hold while trend structure persists.",
        "Risk follows higher-timeframe alignment.",
        {"score": (54.0, 100.0), "pullback_atr": (0.25, 2.0)},
    ),
    PlaybookSpec(
        "RANGE_VWAP_REVERSION_V1",
        "RANGE_VWAP_REVERSION",
        "A low-efficiency range excursion can revert after a causal VWAP-side reclaim.",
        ("5m", "15m", "1h"),
        ("1h", "4h"),
        ("SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "RECOVERY"),
        "Range efficiency is low, price is below VWAP and a closed candle confirms recovery.",
        "Microstructure may be neutral or supportive but never hostile.",
        "Fees, spread and P75 slippage must leave positive net expectancy to VWAP.",
        "Range floor fails or downside acceleration resumes.",
        "Below the causal range low with an ATR buffer.",
        "TP1 at VWAP and TP2 in the upper half of the established range.",
        "No trailing before VWAP; trail only the residual position.",
        "Review after 4-8 execution candles when mean reversion does not progress.",
        "Range-only; disabled during an accelerating directional collapse.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "COMPRESSION_BREAKOUT_V1",
        "VOLATILITY_COMPRESSION_BREAKOUT",
        "Volatility compression followed by participation can precede repricing.",
        ("5m", "15m", "1h"),
        ("1h", "4h"),
        ("SIDEWAYS_LOW_VOL", "BROAD_RISK_ON", "ALT_BULL", "RECOVERY"),
        "A closed candle exits a causal compression range with expanding volume.",
        "Orderflow times the entry and may be neutral or supportive, never hostile.",
        "Reject wide spreads, insufficient depth and targets consumed by friction.",
        "Price closes back inside compression or loses the breakout base.",
        "Below the compression base with an ATR buffer.",
        "TP1 at 1.5R and TP2 at the measured compression range.",
        "Trail below higher lows after TP1.",
        "Review after 4-8 execution candles if expansion fails to continue.",
        "Do not chase an already extended impulse candle.",
        {"score": (56.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "RS_LEADER_PULLBACK_V1",
        "RS_LEADER_PULLBACK",
        "A BTC-relative leader can resume after a contained intraday pullback.",
        ("5m", "15m", "1h"),
        ("1h", "4h"),
        ("ALT_BULL", "BROAD_RISK_ON", "RECOVERY", "UNKNOWN"),
        "Relative strength remains positive while a closed candle reclaims pullback structure.",
        "Demand timing may be neutral or supportive; hostile flow vetoes entry.",
        "Both rotation legs, spread and slippage must preserve positive net edge.",
        "Relative strength turns negative or the pullback structure fails.",
        "Below the contained pullback low with an ATR buffer.",
        "TP1 at the prior high and TP2 at a relative-strength extension.",
        "Trail when relative strength or 1h structure deteriorates.",
        "Review after 4-8 execution candles if leadership does not resume.",
        "One shared position per market prevents overlapping playbook accumulation.",
        {"score": (56.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "RANGE_EXPANSION_VOLUME_V1",
        "RANGE_EXPANSION_VOLUME",
        "A range break backed by new volume can reveal repricing.",
        ("1m", "5m", "15m"),
        ("1h", "4h"),
        ("BROAD_RISK_ON", "ALT_BULL", "RECOVERY"),
        "Return/range expansion with RVOL above its recent baseline.",
        "Trade intensity, CVD and OFI must confirm.",
        "Book depth cannot collapse as price expands.",
        "Return enters range while flow reverses.",
        "Below expansion base.",
        "TP1 at 1.5R and TP2 at measured range move.",
        "Trail with flow and micro-lows.",
        "Review after the timeframe horizon; exit only on confirmed compression.",
        "Volatility-proportional micro sizing.",
        {"score": (54.0, 100.0), "relative_volume": (1.25, 10.0)},
    ),
    PlaybookSpec(
        "ORDERFLOW_CONTINUATION_V1",
        "ORDERFLOW_CONTINUATION",
        "Persistent aggressive buying plus replenishing bids can sustain price.",
        ("1m", "3m", "5m"),
        ("15m", "1h"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Positive micro-return with persistent demand and valid structure.",
        "CVD, OFI, MLOBI and taker ratio align.",
        "Impact estimate and spread remain within cap.",
        "CVD divergence, negative OFI or bid withdrawal.",
        "Below the latest orderflow-supported micro-low.",
        "Partial at 1.25R and runner at 2.25R.",
        "Tighten on flow exhaustion.",
        "Review after the timeframe horizon; exit only on persistent flow reversal.",
        "Requires sequence-valid prospective orderflow.",
        {"score": (62.0, 100.0), "flow_confirmations": (3.0, 5.0)},
    ),
    PlaybookSpec(
        "NORMAL_SWING_TREND_RETEST_V1",
        "NORMAL_SWING_TREND_RETEST",
        "A 15m reclaim inside aligned 1h/4h/1d trend can improve swing entry quality.",
        ("15m", "1h"),
        ("1h", "4h", "1d", "1W"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Closed 15m or 1h pullback reclaims trend structure after higher-timeframe alignment.",
        "1m flow confirms execution only; it never creates the swing idea.",
        "Expected spread, fees and P75 slippage must leave positive net expectancy.",
        "The reclaimed 15m structure or causal 4h trend is lost.",
        "Below the confirmed pullback low or causal ATR invalidation.",
        "TP1 at the prior swing high and TP2 at a 2.5R trend extension.",
        "Trail below confirmed 1h higher lows after TP1.",
        "Review after the timeframe horizon; hold while 1h/4h structure persists.",
        "Normal spot-long risk only after full 1d/4h macro alignment.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "NORMAL_SWING_BREAKOUT_RETEST_V1",
        "NORMAL_SWING_BREAKOUT_RETEST",
        "A 15m breakout-retest backed by volume can continue within a 4h/1d uptrend.",
        ("15m", "1h"),
        ("1h", "4h", "1d", "1W"),
        ("BTC_BULL", "ALT_BULL", "BROAD_RISK_ON", "RECOVERY"),
        "Closed-bar breakout and retest hold above the causal range boundary.",
        "Positive flow and replenishing bids confirm only the execution window.",
        "Reject thin books, wide spreads and targets consumed by roundtrip costs.",
        "Close back inside the broken range or loss of the retest low.",
        "Below the retest low with a bounded ATR floor.",
        "TP1 at 1.5R and TP2 at the measured range projection.",
        "Trail below 1h structure after TP1.",
        "Review after the timeframe horizon; hold while breakout structure persists.",
        "Spot-long only; never chase an unconfirmed 15m candle.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "BEAR_SPOT_LIQUIDITY_RECOVERY_V1",
        "BEAR_SPOT_LIQUIDITY_RECOVERY",
        "A downside sweep with a causal 4h recovery can mean-revert inside a bearish 1d regime.",
        ("15m", "1h"),
        ("1h", "4h", "1d", "1W"),
        ("RISK_OFF", "MACRO_RISK_OFF", "CAPITULATION", "RECOVERY"),
        "15m sweep low is reclaimed while the last closed 4h trend has turned positive.",
        "Demand flow, CVD and bid replenishment confirm the entry timing.",
        "Require conservative P75 costs and at least 1.5 net reward-to-risk at TP2.",
        "Price loses the sweep low or the 4h recovery invalidates.",
        "Immediately below the confirmed sweep low.",
        "TP1 at VWAP/range midpoint and TP2 at opposing range liquidity.",
        "Trail only after TP1; the hard stop remains immediate.",
        "Review after the timeframe horizon; hold while recovery remains valid.",
        "Long-only micro risk capped at 40 percent of normal sizing.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "BEAR_SPOT_FAILED_BREAKDOWN_V1",
        "BEAR_SPOT_FAILED_BREAKDOWN",
        "A failed breakdown plus 4h local recovery can trap sellers in a bearish daily regime.",
        ("15m", "1h"),
        ("1h", "4h", "1d", "1W"),
        ("RISK_OFF", "MACRO_RISK_OFF", "SIDEWAYS_HIGH_VOL", "RECOVERY"),
        "15m closes back above the breakdown boundary with an improving 4h trend.",
        "Positive executed flow confirms the reclaim without defining it.",
        "Spread, depth, fees and P75 slippage must support a clean exit.",
        "Second close below the reclaimed boundary or renewed sell acceleration.",
        "Below the failed-breakdown low.",
        "TP1 at VWAP and TP2 at the previous range high.",
        "Trail after VWAP is reclaimed; never widen the hard stop.",
        "Review after the timeframe horizon; hold while the reclaim remains valid.",
        "Long-only micro risk capped at 40 percent of normal sizing.",
        {"score": (54.0, 100.0), "confirmations": (3.0, 7.0)},
    ),
    PlaybookSpec(
        "ORDERFLOW_EXHAUSTION_EXIT_V1",
        "ORDERFLOW_EXHAUSTION_EXIT",
        "Flow exhaustion is an exit overlay, never a standalone long entry.",
        ("1m", "3m", "5m"),
        ("15m", "1h"),
        ("UNKNOWN",),
        "Only evaluate for an existing long position.",
        "CVD divergence, negative OFI, ask replenishment or bid withdrawal.",
        "Use a bounded exit when spread is normal; otherwise reconcile first.",
        "Exit condition clears when demand and book support recover.",
        "Existing position hard stop remains authoritative.",
        "May reduce or close before static targets.",
        "Tighten the trailing stop on partial exhaustion.",
        "Immediate when multiple exhaustion facts agree.",
        "Cannot increase exposure.",
        {"score": (54.0, 100.0), "exit_confirmations": (2.0, 5.0)},
    ),
)


PLAYBOOK_BY_FAMILY = {item.family: item for item in PLAYBOOKS}

BEAR_SPOT_RECOVERY_FAMILIES = {
    "BEAR_SPOT_LIQUIDITY_RECOVERY",
    "BEAR_SPOT_FAILED_BREAKDOWN",
}


def _number(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if selected == selected else None


def _scale(value: float | None, lower: float, upper: float) -> float:
    if value is None or upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def macro_risk_multiplier(regime: str | None) -> float:
    selected = str(regime or "UNKNOWN").upper()
    if selected in {"STRONG_RISK_ON", "ALT_BULL"}:
        return 1.15
    if selected in {"RISK_ON", "BTC_BULL", "BROAD_RISK_ON", "RECOVERY"}:
        return 1.0
    if selected in {"NEUTRAL", "SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL"}:
        return 0.85
    if selected in {"RISK_OFF", "MACRO_RISK_OFF"}:
        return 0.65
    if selected in {"EXTREME_RISK_OFF", "CAPITULATION"}:
        return 0.40
    return 0.85


def macro_long_lock(
    regime: str | None,
    macro_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a soft higher-timeframe modifier plus macro-data integrity gate.

    Directional disagreement is already represented by the weighted
    15m/1h/2h/4h/1d/1w score.  Re-applying the old binary 1d/4h permission
    tree here would double count the same evidence and suppress valid tactical
    rebounds.  Missing macro data remains fail closed; a bearish observation
    only reduces the maximum risk multiplier.
    """

    if not macro_context:
        return {
            "status": "NOT_EVALUATED",
            "long_allowed": True,
            "blockers": [],
            "warnings": ["MACRO_CONTEXT_NOT_AVAILABLE_SIZE_REDUCTION"],
            "btc_1d_trend_up": None,
            "btc_4h_trend_up": None,
            "provider_status": None,
            "risk_multiplier_cap": 0.65,
        }
    features = dict(macro_context.get("features") or {})
    btc_1d = features.get("btc_1d_trend_up")
    btc_4h = features.get("btc_4h_trend_up")
    provider_refresh = dict(macro_context.get("provider_refresh") or {})
    provider_status = provider_refresh.get("coinmarketcap_global")
    blockers: list[str] = []
    warnings: list[str] = []
    if provider_status != "READY":
        blockers.append("CMC_MACRO_DATA_NOT_READY_LONG_LOCK")
    if btc_1d is not True:
        warnings.append(
            "MACRO_1D_BEARISH_SIZE_REDUCTION"
            if btc_1d is False
            else "MACRO_1D_UNKNOWN_SIZE_REDUCTION"
        )
    if btc_4h is not True:
        warnings.append(
            "MACRO_4H_BEARISH_SIZE_REDUCTION"
            if btc_4h is False
            else "MACRO_4H_UNKNOWN_SIZE_REDUCTION"
        )
    selected_regime = str(regime or "UNKNOWN").upper()
    if selected_regime in {
        "RISK_OFF",
        "MACRO_RISK_OFF",
        "EXTREME_RISK_OFF",
        "CAPITULATION",
    }:
        warnings.append("MACRO_RISK_OFF_SIZE_REDUCTION")
    risk_cap = 1.0
    if btc_1d is not True:
        risk_cap = min(risk_cap, 0.75)
    if btc_4h is not True:
        risk_cap = min(risk_cap, 0.65)
    if selected_regime in {"RISK_OFF", "MACRO_RISK_OFF"}:
        risk_cap = min(risk_cap, 0.65)
    elif selected_regime in {"EXTREME_RISK_OFF", "CAPITULATION"}:
        risk_cap = min(risk_cap, 0.40)
    return {
        "status": "PASS" if not blockers else "BLOCKED_DATA_INTEGRITY",
        "long_allowed": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "btc_1d_trend_up": btc_1d,
        "btc_4h_trend_up": btc_4h,
        "provider_status": provider_status,
        "available_at": macro_context.get("available_at"),
        "risk_multiplier_cap": risk_cap if not blockers else 0.0,
    }


def macro_playbook_gate(
    playbook: PlaybookSpec,
    regime: str | None,
    macro_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply normal trend lock or the bounded bearish-recovery exception.

    The exception is not a relaxation for every long strategy.  It is limited
    to separately identified long-only recovery DNA and requires an explicitly
    bearish closed 1d state plus a positive last-closed 4h recovery state.
    """

    if playbook.family not in BEAR_SPOT_RECOVERY_FAMILIES:
        gate = macro_long_lock(regime, macro_context)
        return {**gate, "policy": "WEIGHTED_SOFT_REGIME"}
    if not macro_context:
        return {
            "status": "BLOCKED",
            "long_allowed": False,
            "blockers": ["BEAR_RECOVERY_MACRO_CONTEXT_MISSING"],
            "btc_1d_trend_up": None,
            "btc_4h_trend_up": None,
            "provider_status": None,
            "policy": "BEARISH_RECOVERY_LONG_ONLY",
            "risk_multiplier_cap": 0.40,
        }
    features = dict(macro_context.get("features") or {})
    provider_status = dict(
        macro_context.get("provider_refresh") or {}
    ).get("coinmarketcap_global")
    btc_1d = features.get("btc_1d_trend_up")
    btc_4h = features.get("btc_4h_trend_up")
    selected_regime = str(regime or "UNKNOWN").upper()
    blockers: list[str] = []
    if provider_status != "READY":
        blockers.append("CMC_MACRO_DATA_NOT_READY_BEAR_RECOVERY")
    if btc_1d is not False:
        blockers.append("BEAR_RECOVERY_REQUIRES_BEARISH_1D")
    if btc_4h is not True:
        blockers.append("BEAR_RECOVERY_REQUIRES_POSITIVE_4H")
    if selected_regime not in {
        "RISK_OFF",
        "MACRO_RISK_OFF",
        "EXTREME_RISK_OFF",
        "CAPITULATION",
        "RECOVERY",
        "SIDEWAYS_HIGH_VOL",
    }:
        blockers.append("BEAR_RECOVERY_REGIME_NOT_ELIGIBLE")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "long_allowed": not blockers,
        "blockers": blockers,
        "btc_1d_trend_up": btc_1d,
        "btc_4h_trend_up": btc_4h,
        "provider_status": provider_status,
        "available_at": macro_context.get("available_at"),
        "policy": "BEARISH_RECOVERY_LONG_ONLY",
        "risk_multiplier_cap": 0.40,
    }


def _tier(score: float) -> str:
    if score >= 88:
        return "A"
    if score >= 82:
        return "B"
    if score >= 75:
        return "C"
    if score >= 65:
        return "WATCH"
    return "REJECT"


def _best_context(
    market: str,
    opportunities: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    selected = [row for row in opportunities if row.get("market") == market]
    if not selected:
        return {}
    return max(
        selected,
        key=lambda row: (
            bool(row.get("setup_valid_on_closed_candle")),
            bool(row.get("entry_trigger_confirmed")),
            _number(row.get("confidence")) or 0.0,
        ),
    )


def _higher_timeframe_parent(context: Mapping[str, Any]) -> dict[str, Any]:
    """Require a causal chain while treating higher timeframes as soft context.

    A 15m trigger must inherit a closed 1h/2h/4h setup and 4h-or-higher
    regime.  A 1h trigger is itself a valid swing-entry timeframe when it
    inherits a closed 4h/1d setup and 1d-or-higher regime.  Higher-timeframe
    entries remain context-only in this execution path.
    """

    entry_timeframe = str(
        context.get("entry_timeframe") or context.get("timeframe") or ""
    )
    confirmation_timeframe = str(
        context.get("confirmation_timeframe") or ""
    )
    regime_timeframe = str(context.get("regime_timeframe") or "")
    alignment = _number(context.get("timeframe_alignment_score")) or 0.0
    weighted_score = _number(context.get("weighted_timeframe_score"))
    weighted_threshold = (
        _number(context.get("weighted_entry_threshold")) or 0.35
    )
    explicit = context.get("higher_timeframe_parent_valid")
    causal_chain = bool(
        (
            entry_timeframe == "15m"
            and confirmation_timeframe in {"1h", "2h", "4h"}
            and regime_timeframe in {"4h", "1d", "1W"}
        )
        or (
            entry_timeframe == "1h"
            and confirmation_timeframe in {"4h", "1d"}
            and regime_timeframe in {"1d", "1W"}
        )
    )
    directional_valid = (
        weighted_score >= weighted_threshold
        if weighted_score is not None
        else alignment >= 0.5
    )
    closed_candle_setup_valid = bool(
        context.get("setup_valid_on_closed_candle") is True
        or context.get("entry_trigger_confirmed") is True
    )
    valid = bool(
        (explicit is True or causal_chain)
        and directional_valid
        and closed_candle_setup_valid
        and context.get("closed_candle_only") is True
    )
    return {
        "valid": valid,
        "entry_timeframe": entry_timeframe or None,
        "confirmation_timeframe": confirmation_timeframe or None,
        "regime_timeframe": regime_timeframe or None,
        "alignment_score": alignment,
        "weighted_timeframe_score": weighted_score,
        "weighted_entry_threshold": weighted_threshold,
        "directional_score_valid": directional_valid,
        "trade_type": context.get("trade_type"),
        "hard_blocked_by_1d_or_1w": False,
        "closed_candle_only": context.get("closed_candle_only") is True,
        "closed_candle_setup_valid": closed_candle_setup_valid,
        "closed_candle_entry_trigger_confirmed": (
            context.get("entry_trigger_confirmed") is True
        ),
        "execution_trigger_policy": (
            "CLOSED_CANDLE_OR_REALTIME_MICROSTRUCTURE"
        ),
        "blocker": None if valid else "MISSING_VALID_1H_4H_PARENT_SETUP",
    }


def _execution_scorecard(
    *,
    context: Mapping[str, Any],
    scoring: Mapping[str, Any],
    economics: Mapping[str, Any],
    parent_setup: Mapping[str, Any],
    entry_chase_atr: float | None,
) -> dict[str, float]:
    """Return the fixed 100-point active-swing scorecard."""

    components = dict(scoring.get("components") or {})
    alignment = max(
        0.0,
        min(1.0, _number(context.get("timeframe_alignment_score")) or 0.0),
    )
    confidence = max(
        0.0,
        min(1.0, (_number(context.get("confidence")) or 0.0) / 100.0),
    )
    trigger = context.get("entry_trigger_confirmed") is True
    chase_quality = (
        max(0.0, 1.0 - entry_chase_atr / 0.35)
        if entry_chase_atr is not None
        else 0.0
    )
    net_tp1 = float(economics.get("net_target_1_bps") or 0.0)
    net_tp2 = float(economics.get("net_target_2_bps") or 0.0)
    rr1 = float(economics.get("net_rr_target_1") or 0.0)
    rr2 = float(economics.get("net_rr_target_2") or 0.0)
    ev_positive = float(
        float(economics.get("expected_net_value_bps") or 0.0) > 0.0
    )
    target_quality = min(
        1.0,
        0.30 * _scale(net_tp1, 0.0, 150.0)
        + 0.30 * _scale(net_tp2, 0.0, 300.0)
        + 0.15 * _scale(rr1, 0.0, 1.25)
        + 0.15 * _scale(rr2, 0.0, 2.0)
        + 0.10 * ev_positive,
    )
    return {
        "structure_4h": 20.0
        * (0.65 * alignment + 0.35 * float(bool(parent_setup.get("valid")))),
        "confirmation_1h": 15.0 * (0.65 * confidence + 0.35 * alignment),
        "entry_15m": 15.0 * (0.70 * float(trigger) + 0.30 * chase_quality),
        "target_net_rr": 15.0 * target_quality,
        "relative_strength": float(components.get("relative_strength") or 0.0),
        "volume_trade_intensity": 8.0
        * min(1.0, float(components.get("volume_acceleration") or 0.0) / 20.0),
        "orderflow": 7.0
        * min(1.0, float(components.get("executed_flow_cvd") or 0.0) / 20.0),
        "macro_breadth": float(components.get("macro") or 0.0),
        "friction_liquidity": 5.0
        * min(1.0, float(components.get("orderbook_liquidity") or 0.0) / 15.0),
    }


def score_realtime_market(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    btc_row: Mapping[str, Any] | None = None,
    macro_regime: str = "UNKNOWN",
    strategy_evidence: float = 0.5,
) -> dict[str, Any]:
    """Apply the preregistered 20/20/20/15/10/10/5 score."""

    context = context or {}
    windows = dict(row.get("windows") or {})
    one = dict(windows.get("1m") or {})
    five = dict(windows.get("5m") or {})
    fifteen = dict(windows.get("15m") or {})
    book = dict(row.get("book") or {})
    return_1m = _number(one.get("return"))
    return_5m = _number(five.get("return"))
    return_15m = _number(fifteen.get("return"))
    acceleration = _number(row.get("acceleration_1m"))
    rvol = _number(row.get("relative_volume_1m"))
    intensity = _number(row.get("trade_intensity_1m"))
    buy_ratio = _number(one.get("taker_buy_ratio"))
    cvd = _number(one.get("cvd_quote_eur"))
    trade_count = int(_number(one.get("trade_count")) or 0)
    executed_quote_volume = _number(one.get("quote_volume_eur")) or 0.0
    trade_age = _number(row.get("trade_age_seconds"))
    book_age = _number(row.get("book_age_seconds"))
    volume_sample_sufficient = (
        trade_count >= MINIMUM_VOLUME_CONFIRMATION_TRADES
    )
    executed_flow_sample_sufficient = (
        trade_count >= 1
        and executed_quote_volume >= MINIMUM_EXECUTED_QUOTE_VOLUME_EUR
    )
    ofi = _number(row.get("ofi_1m"))
    ofi_windows = {
        str(label): _number(value)
        for label, value in dict(row.get("ofi_windows") or {}).items()
    }
    mlobi = _number(book.get("mlobi_top_10"))
    weighted_book_imbalance = _number(
        book.get("distance_weighted_imbalance_top_10")
    )
    depth_imbalance_10bps = _number(
        book.get("depth_imbalance_within_10_bps")
    )
    microprice_edge = _number(row.get("microprice_edge_bps"))
    book_update_count_10s = int(
        _number(row.get("book_update_count_10s")) or 0
    )
    raw_book_persistence = _number(
        row.get("mlobi_positive_persistence_10s")
    )
    book_persistence = (
        raw_book_persistence
        if book_update_count_10s >= MINIMUM_BOOK_PERSISTENCE_UPDATES
        else None
    )
    bid_replenishment_ratio = _number(
        row.get("bid_replenishment_ratio_1m")
    )
    ask_depletion_ratio = _number(row.get("ask_depletion_ratio_1m"))
    absorption = _number(row.get("bullish_absorption_score_1m"))
    sweep_reclaim = bool(row.get("downside_sweep_reclaim_1m"))
    spread = _number(book.get("spread_bps"))
    slippage = _number(row.get("estimated_buy_slippage_bps"))
    bid_depth = _number(book.get("bid_depth_eur_top_10"))
    dynamic_spread_cap = _number(book.get("dynamic_spread_cap_bps"))
    spread_within_dynamic_cap = book.get("spread_within_dynamic_cap")
    btc_return_5m = _number(
        ((btc_row or {}).get("windows") or {}).get("5m", {}).get("return")
    )
    relative_5m = (
        return_5m - btc_return_5m
        if return_5m is not None and btc_return_5m is not None
        else return_5m
    )
    trigger = bool(context.get("entry_trigger_confirmed"))
    alignment = _number(context.get("timeframe_alignment_score")) or 0.0
    technical = 20.0 * min(
        1.0,
        0.35 * float(trigger)
        + 0.35 * _scale(return_5m, 0.0, 0.012)
        + 0.20 * _scale(return_15m, 0.0, 0.025)
        + 0.10 * alignment,
    )
    volume = 20.0 * min(
        1.0,
        0.40
        * _scale(rvol if volume_sample_sufficient else None, 1.0, 3.0)
        + 0.25
        * _scale(
            intensity if volume_sample_sufficient else None,
            1.0,
            3.0,
        )
        + 0.20 * _scale(acceleration, 0.0, 0.006)
        + 0.15 * _scale(return_1m, 0.0, 0.006),
    )
    flow = 20.0 * min(
        1.0,
        0.35 * _scale(buy_ratio, 0.50, 0.75)
        + 0.25 * _scale(cvd, 0.0, 250.0)
        + 0.20 * _scale(ofi, 0.0, 0.35)
        + 0.10 * _scale(mlobi, 0.0, 0.35)
        + 0.05 * _scale(book_persistence, 0.3, 0.9)
        + 0.05
        * max(
            _scale(bid_replenishment_ratio, 0.0, 0.15),
            _scale(ask_depletion_ratio, 0.0, 0.15),
        ),
    )
    liquidity = 15.0 * min(
        1.0,
        0.40 * (1.0 - _scale(spread, 4.0, 35.0))
        + 0.25 * (1.0 - _scale(slippage, 2.0, 25.0))
        + 0.25 * _scale(bid_depth, 250.0, 10_000.0)
        + 0.05 * _scale(mlobi, -0.10, 0.25)
        + 0.05 * _scale(microprice_edge, -1.0, 3.0),
    )
    relative = 10.0 * _scale(relative_5m, -0.002, 0.012)
    evidence = 10.0 * max(0.0, min(1.0, float(strategy_evidence)))
    macro_multiplier = macro_risk_multiplier(macro_regime)
    macro = 5.0 * min(1.0, macro_multiplier)
    components = {
        "technical": technical,
        "volume_acceleration": volume,
        "executed_flow_cvd": flow,
        "orderbook_liquidity": liquidity,
        "relative_strength": relative,
        "strategy_evidence": evidence,
        "macro": macro,
    }
    persistent_ofi_values = [
        value
        for label in ("30s", "90s", "300s")
        if (value := ofi_windows.get(label)) is not None
    ]
    persistent_ofi_confirmed = (
        sum(value > 0.03 for value in persistent_ofi_values) >= 2
        if len(persistent_ofi_values) >= 2
        else ofi is not None and ofi > 0.03
    )
    confirmations = {
        "relative_volume": (
            volume_sample_sufficient and rvol is not None and rvol >= 1.3
        ),
        "trade_intensity": (
            volume_sample_sufficient
            and intensity is not None
            and intensity >= 1.25
        ),
        "taker_flow": (
            executed_flow_sample_sufficient
            and buy_ratio is not None
            and buy_ratio >= 0.56
        ),
        "cvd": (
            executed_flow_sample_sufficient and cvd is not None and cvd > 0
        ),
        "ofi": persistent_ofi_confirmed,
        "mlobi": mlobi is not None and mlobi > 0.03,
        "microprice": (
            microprice_edge is not None and microprice_edge > 0.25
        ),
        "absorption_or_sweep": (
            (absorption is not None and absorption >= 0.05)
            or sweep_reclaim
        ),
        "replenishment": (
            (
                bid_replenishment_ratio is not None
                and bid_replenishment_ratio >= 0.03
            )
            or (
                ask_depletion_ratio is not None
                and ask_depletion_ratio >= 0.03
            )
        ),
        "valid_structure": trigger or alignment >= 0.5,
    }
    confirmation_quality = confirmation_independence(confirmations)
    confirmation_count = int(confirmation_quality["raw_count"])
    independent_confirmation_count = int(
        confirmation_quality["independent_count"]
    )
    raw_score = sum(components.values())
    # Correlated flow features are useful, but are not independent evidence.
    redundancy_penalty = min(
        6.0,
        max(0, confirmation_count - independent_confirmation_count) * 0.75,
    )
    score = max(0.0, min(100.0, raw_score - redundancy_penalty))
    ofi_reference = (
        sum(persistent_ofi_values) / len(persistent_ofi_values)
        if persistent_ofi_values
        else ofi
    )
    execution_quality_parts = {
        "ofi": (
            0.20,
            _scale(ofi_reference, -0.15, 0.25)
            if ofi_reference is not None
            else None,
        ),
        "cvd": (
            0.15,
            _scale(cvd, -250.0, 250.0) if cvd is not None else None,
        ),
        "microprice": (
            0.10,
            _scale(microprice_edge, -2.0, 2.0)
            if microprice_edge is not None
            else None,
        ),
        "spread": (
            0.15,
            (
                max(0.0, 1.0 - spread / dynamic_spread_cap)
                if spread is not None
                and dynamic_spread_cap is not None
                and dynamic_spread_cap > 0
                else None
            ),
        ),
        "depth": (
            0.10,
            _scale(
                depth_imbalance_10bps
                if depth_imbalance_10bps is not None
                else weighted_book_imbalance,
                -0.35,
                0.35,
            )
            if depth_imbalance_10bps is not None
            or weighted_book_imbalance is not None
            else None,
        ),
        "replenishment": (
            0.10,
            max(
                _scale(bid_replenishment_ratio, -0.15, 0.15),
                _scale(ask_depletion_ratio, -0.15, 0.15),
            )
            if bid_replenishment_ratio is not None
            or ask_depletion_ratio is not None
            else None,
        ),
        "absorption": (
            0.10,
            _scale(absorption, 0.0, 0.20)
            if absorption is not None
            else None,
        ),
        "book_persistence": (
            0.10,
            _scale(book_persistence, 0.0, 1.0)
            if book_persistence is not None
            else None,
        ),
    }
    available_execution_weight = sum(
        weight
        for weight, value in execution_quality_parts.values()
        if value is not None
    )
    execution_quality_score = (
        sum(
            weight * float(value)
            for weight, value in execution_quality_parts.values()
            if value is not None
        )
        / available_execution_weight
        if available_execution_weight > 0
        else None
    )
    executed_flow_hostile = bool(
        executed_flow_sample_sufficient
        and (
            (
                buy_ratio is not None
                and buy_ratio < 0.42
                and cvd is not None
                and cvd < 0.0
            )
            or (ofi_reference is not None and ofi_reference < -0.15)
        )
    )
    orderbook_hostile = bool(
        (
            depth_imbalance_10bps is not None
            and depth_imbalance_10bps < -0.25
        )
        and microprice_edge is not None
        and microprice_edge < -1.0
    )
    hostile_groups = [
        name
        for name, active in (
            ("EXECUTED_FLOW", executed_flow_hostile),
            ("ORDERBOOK", orderbook_hostile),
        )
        if active
    ]
    if len(hostile_groups) >= 2:
        microstructure_state = "HOSTILE"
    elif (
        execution_quality_score is not None
        and execution_quality_score >= 0.55
        and independent_confirmation_count >= 2
    ):
        microstructure_state = "SUPPORTIVE"
    else:
        microstructure_state = "NEUTRAL"
    hard_blockers: list[str] = []
    if not row.get("fresh"):
        hard_blockers.append("STALE_REALTIME_DATA")
    if book_age is None or book_age > MAXIMUM_BOOK_AGE_SECONDS:
        hard_blockers.append("STALE_ORDERBOOK_DATA")
    if (
        trade_age is None
        or trade_age > MAXIMUM_EXECUTED_FLOW_AGE_SECONDS
        or not executed_flow_sample_sufficient
    ):
        hard_blockers.append("NO_RECENT_EXECUTED_TRADES")
    if not row.get("sequence_valid"):
        hard_blockers.append("ORDERBOOK_SEQUENCE_INVALID")
    if spread is None or spread > 35.0:
        hard_blockers.append("SPREAD_ABOVE_MICRO_LIVE_CAP")
    if slippage is None or slippage > 25.0:
        hard_blockers.append("SLIPPAGE_ABOVE_MICRO_LIVE_CAP")
    if bid_depth is None or bid_depth < 25.0:
        hard_blockers.append("INSUFFICIENT_EXIT_DEPTH")
    if microstructure_state == "HOSTILE":
        hard_blockers.append("HOSTILE_MICROSTRUCTURE")
    return {
        "score": score,
        "raw_score": raw_score,
        "confirmation_redundancy_penalty": redundancy_penalty,
        "tier": _tier(score),
        "components": components,
        "confirmations": confirmations,
        "confirmation_count": confirmation_count,
        "independent_confirmation_count": independent_confirmation_count,
        "confirmation_groups": confirmation_quality["groups"],
        "confirmation_independence_ratio": confirmation_quality[
            "independence_ratio"
        ],
        "execution_quality_score": execution_quality_score,
        "execution_quality_components": {
            name: value
            for name, (_weight, value) in execution_quality_parts.items()
        },
        "execution_quality_observed_weight": available_execution_weight,
        "microstructure_state": microstructure_state,
        "microstructure_hostile_groups": hostile_groups,
        "microstructure_policy": (
            "HOSTILE_BLOCKS_NEUTRAL_SIZES_NORMALLY_SUPPORTIVE_STRENGTHENS"
        ),
        "microstructure_quality": {
            "book_fresh": (
                book_age is not None
                and book_age <= MAXIMUM_BOOK_AGE_SECONDS
            ),
            "executed_flow_fresh": (
                trade_age is not None
                and trade_age <= MAXIMUM_EXECUTED_FLOW_AGE_SECONDS
                and executed_flow_sample_sufficient
            ),
            "volume_sample_sufficient": volume_sample_sufficient,
            "book_persistence_sample_sufficient": (
                book_update_count_10s
                >= MINIMUM_BOOK_PERSISTENCE_UPDATES
            ),
            "trade_count_1m": trade_count,
            "executed_quote_volume_eur_1m": executed_quote_volume,
            "trade_age_seconds": trade_age,
            "book_age_seconds": book_age,
            "book_update_count_10s": book_update_count_10s,
        },
        "hard_blockers": hard_blockers,
        "macro_regime": macro_regime,
        "macro_risk_multiplier": macro_multiplier,
        "relative_return_5m": relative_5m,
        "inputs": {
            "return_1m": return_1m,
            "return_5m": return_5m,
            "return_15m": return_15m,
            "acceleration_1m": acceleration,
            "relative_volume_1m": rvol,
            "trade_intensity_1m": intensity,
            "taker_buy_ratio_1m": buy_ratio,
            "cvd_quote_eur_1m": cvd,
            "trade_count_1m": trade_count,
            "executed_quote_volume_eur_1m": executed_quote_volume,
            "trade_age_seconds": trade_age,
            "book_age_seconds": book_age,
            "ofi_1m": ofi,
            "ofi_windows": ofi_windows,
            "mlobi_top_10": mlobi,
            "distance_weighted_imbalance_top_10": weighted_book_imbalance,
            "depth_imbalance_within_10_bps": depth_imbalance_10bps,
            "microprice_edge_bps": microprice_edge,
            "mlobi_positive_persistence_10s": book_persistence,
            "book_update_count_10s": book_update_count_10s,
            "bid_replenishment_ratio_1m": bid_replenishment_ratio,
            "ask_depletion_ratio_1m": ask_depletion_ratio,
            "bullish_absorption_score_1m": absorption,
            "downside_sweep_reclaim_1m": sweep_reclaim,
            "spread_bps": spread,
            "dynamic_spread_cap_bps": dynamic_spread_cap,
            "spread_within_dynamic_cap": spread_within_dynamic_cap,
            "estimated_buy_slippage_bps": slippage,
            "bid_depth_eur_top_10": bid_depth,
        },
    }


def _matching_playbooks(
    scoring: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[PlaybookSpec]:
    inputs = dict(scoring.get("inputs") or {})
    return_1m = _number(inputs.get("return_1m")) or 0.0
    return_5m = _number(inputs.get("return_5m")) or 0.0
    rvol = _number(inputs.get("relative_volume_1m")) or 0.0
    acceleration = _number(inputs.get("acceleration_1m")) or 0.0
    buy_ratio = _number(inputs.get("taker_buy_ratio_1m")) or 0.0
    ofi = _number(inputs.get("ofi_1m")) or 0.0
    sweep_reclaim = bool(inputs.get("downside_sweep_reclaim_1m"))
    participation_confirmed = bool(
        (scoring.get("confirmation_groups") or {}).get("participation")
    )
    context_family = str(context.get("family") or "").upper()
    status = str(context.get("status") or "").upper()
    families: set[str] = set()
    if (
        return_1m > 0.0015
        and return_5m > 0.004
        and rvol >= 1.3
        and participation_confirmed
    ):
        families.update({"MOMENTUM_BREAKOUT", "RANGE_EXPANSION_VOLUME"})
    if rvol >= 1.5 and acceleration > 0.001 and participation_confirmed:
        families.add("VOLATILITY_EXPANSION")
    if (
        status in {"PULLBACK_PENDING", "FIRST_PULLBACK_AFTER_IMPULSE"}
        and return_1m > 0
        and ofi > 0
    ):
        families.add("BREAKOUT_PULLBACK")
    if "LIQUIDITY_SWEEP" in context_family and return_1m > 0:
        families.add("LIQUIDITY_SWEEP_RECLAIM")
        families.add("BEAR_SPOT_LIQUIDITY_RECOVERY")
    if sweep_reclaim:
        families.update(
            {"LIQUIDITY_SWEEP_RECLAIM", "FAILED_BREAKDOWN_REVERSAL"}
        )
    if "FAILED_BREAKDOWN" in context_family and return_1m > 0:
        families.add("FAILED_BREAKDOWN_REVERSAL")
        families.add("BEAR_SPOT_FAILED_BREAKDOWN")
    if (
        "FAILED_BREAKOUT" in context_family
        and return_1m > 0
        and buy_ratio >= 0.55
        and ofi > 0
    ):
        families.add("FAILED_BREAKOUT_REVERSAL")
    if "VWAP" in context_family and return_1m > 0 and buy_ratio >= 0.55:
        families.add("VWAP_RECLAIM")
    if "TREND_PULLBACK" in context_family and acceleration > 0:
        families.add("TREND_PULLBACK")
        families.add("NORMAL_SWING_TREND_RETEST")
    if (
        ("RANGE_VWAP_REVERSION" in context_family
         or "RANGE_MEAN_REVERSION" in context_family)
        and return_1m > 0
    ):
        families.add("RANGE_VWAP_REVERSION")
    if (
        "COMPRESSION_BREAKOUT" in context_family
        and return_1m > 0
        and rvol >= 1.0
    ):
        families.add("VOLATILITY_COMPRESSION_BREAKOUT")
    relative_return_5m = _number(scoring.get("relative_return_5m"))
    if (
        ("RS_LEADER_PULLBACK" in context_family
         or "RELATIVE_STRENGTH_CONTINUATION" in context_family)
        and relative_return_5m is not None
        and relative_return_5m > 0
        and return_1m > 0
    ):
        families.add("RS_LEADER_PULLBACK")
    if (
        "BREAKOUT_RETEST" in context_family
        and return_1m > 0
        and rvol >= 1.3
        and participation_confirmed
    ):
        families.add("NORMAL_SWING_BREAKOUT_RETEST")
    if "NORMAL_SWING_TREND_RETEST" in context_family and acceleration > 0:
        families.add("NORMAL_SWING_TREND_RETEST")
    if (
        "NORMAL_SWING_BREAKOUT_RETEST" in context_family
        and return_1m > 0
        and rvol >= 1.3
        and participation_confirmed
    ):
        families.add("NORMAL_SWING_BREAKOUT_RETEST")
    if (
        "BEAR_SPOT_LIQUIDITY_RECOVERY" in context_family
        and sweep_reclaim
        and return_1m > 0
    ):
        families.add("BEAR_SPOT_LIQUIDITY_RECOVERY")
    if (
        "BEAR_SPOT_FAILED_BREAKDOWN" in context_family
        and return_1m > 0
        and ofi > 0
    ):
        families.add("BEAR_SPOT_FAILED_BREAKDOWN")
    if (
        _number(scoring.get("relative_return_5m")) is not None
        and float(scoring["relative_return_5m"]) > 0.003
    ):
        families.add("RELATIVE_STRENGTH_ROTATION")
    if buy_ratio >= 0.58 and ofi > 0.03 and return_1m > 0:
        families.add("ORDERFLOW_CONTINUATION")
    return [PLAYBOOK_BY_FAMILY[name] for name in sorted(families)]


def _validated_playbook_band(
    playbook: PlaybookSpec,
    scoring: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate entry facts against the playbook's frozen live band.

    The catalog contains family-specific score and confirmation floors.  They
    are execution constraints, not merely documentation.  Observed market
    features such as RVOL remain inputs to the playbook matcher; only frozen
    governance parameters that are present for every opportunity are checked
    here so an unavailable optional feature cannot silently veto an entry.
    """

    parameter_band = dict(playbook.parameter_band)
    score = float(scoring.get("score") or 0.0)
    confirmation_count = int(
        scoring.get("independent_confirmation_count") or 0
    )
    failures: list[str] = []
    observed: dict[str, float] = {"score": score}
    score_band = parameter_band.get("score")
    if score_band is not None:
        lower, upper = (float(value) for value in score_band)
        if not lower <= score <= upper:
            failures.append("PLAYBOOK_SCORE_OUTSIDE_VALIDATED_BAND")
    confirmation_keys = (
        "confirmations",
        "flow_confirmations",
        "reclaim_confirmations",
    )
    for key in confirmation_keys:
        band = parameter_band.get(key)
        if band is None:
            continue
        lower, upper = (float(value) for value in band)
        observed[key] = float(confirmation_count)
        # More independent confirmations strengthen an entry.  The upper
        # catalog value describes the preregistered search range, not a reason
        # to reject a live observation that supplies additional evidence.
        if confirmation_count < lower:
            failures.append(
                "PLAYBOOK_CONFIRMATIONS_OUTSIDE_VALIDATED_BAND"
            )
        break
    return {
        "status": "VALIDATED" if not failures else "OUTSIDE_BAND",
        "failures": failures,
        "observed": observed,
        "parameter_band": {
            key: list(value) for key, value in parameter_band.items()
        },
        "parameter_band_hash": stable_hash(parameter_band, length=64),
    }


def build_event_driven_opportunities(
    realtime_snapshot: Mapping[str, Any],
    *,
    tactical_opportunities: Iterable[Mapping[str, Any]] = (),
    macro_regime: str = "UNKNOWN",
    macro_context: Mapping[str, Any] | None = None,
    evidence_by_family: Mapping[str, float] | None = None,
    maker_fee_bps: float = 15.0,
    taker_fee_bps: float = 25.0,
) -> list[dict[str, Any]]:
    rows = list(realtime_snapshot.get("markets") or [])
    btc_row = next(
        (row for row in rows if row.get("market") == "BTC-EUR"),
        None,
    )
    tactical = [
        dict(row)
        for row in tactical_opportunities
        if str(
            row.get("entry_timeframe") or row.get("timeframe") or ""
        ).casefold()
        in {"15m", "1h"}
    ]
    evidence = dict(evidence_by_family or {})
    output: list[dict[str, Any]] = []
    observed_at = str(
        realtime_snapshot.get("observed_at") or utc_now().isoformat()
    )
    observed_datetime = datetime.fromisoformat(
        observed_at.replace("Z", "+00:00")
    ).astimezone(UTC)
    for row in rows:
        market = str(row.get("market") or "")
        context = _best_context(market, tactical)
        parent_setup = _higher_timeframe_parent(context)
        family_evidence = evidence.get(str(context.get("family") or ""), 0.5)
        scoring = score_realtime_market(
            row,
            context=context,
            btc_row=btc_row,
            macro_regime=macro_regime,
            strategy_evidence=family_evidence,
        )
        for playbook in _matching_playbooks(scoring, context):
            macro_gate = macro_playbook_gate(
                playbook,
                macro_regime,
                macro_context,
            )
            band_validation = _validated_playbook_band(playbook, scoring)
            price = _number(row.get("price"))
            if price is None or price <= 0:
                continue
            return_5m = abs(
                _number(scoring["inputs"].get("return_5m")) or 0.002
            )
            stop_fraction = max(0.0035, min(0.02, return_5m * 1.5))
            fallback_stop = price * (1.0 - stop_fraction)
            context_stop = _number(
                context.get("stop") or context.get("stop_loss")
            )
            context_target_1 = _number(
                context.get("target_1") or context.get("take_profit_1")
            )
            context_target_2 = _number(
                context.get("target_2") or context.get("take_profit_2")
            )
            stop = (
                context_stop
                if context_stop is not None and 0 < context_stop < price
                else fallback_stop
            )
            risk = price - stop
            take_profit_1 = (
                context_target_1
                if context_target_1 is not None and context_target_1 > price
                else price + risk * 1.5
            )
            take_profit_2 = (
                context_target_2
                if context_target_2 is not None
                and context_target_2 > take_profit_1
                else price + risk * 2.5
            )
            economics = estimate_roundtrip_economics(
                entry_price=price,
                stop_loss=stop,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                spread_bps=scoring["inputs"].get("spread_bps"),
                observed_slippage_bps=scoring["inputs"].get(
                    "estimated_buy_slippage_bps"
                ),
                maker_fee_bps=maker_fee_bps,
                taker_fee_bps=taker_fee_bps,
                conservative_win_probability=(
                    CONSERVATIVE_WIN_PROBABILITY
                ),
            )
            entry_zone = list(context.get("entry_zone") or [])
            entry_atr = _number(context.get("entry_atr"))
            zone_low = _number(entry_zone[0]) if len(entry_zone) >= 2 else None
            zone_high = _number(entry_zone[1]) if len(entry_zone) >= 2 else None
            ideal_entry = (
                (zone_low + zone_high) / 2.0
                if zone_low is not None and zone_high is not None
                else price
            )
            entry_chase_atr: float | None = None
            if len(entry_zone) >= 2 and entry_atr and entry_atr > 0:
                if zone_high is not None:
                    entry_chase_atr = max(0.0, price - zone_high) / entry_atr
            maximum_by_chase = (
                zone_high + 0.35 * entry_atr
                if zone_high is not None and entry_atr and entry_atr > 0
                else price
            )
            maximum_by_rr_1 = (take_profit_1 + stop) / 2.0
            maximum_by_rr_2 = (take_profit_2 + 1.5 * stop) / 2.5
            maximum_acceptable_entry = min(
                maximum_by_chase,
                maximum_by_rr_1,
                maximum_by_rr_2,
            )
            edge_denominator = max(1e-12, take_profit_2 - ideal_entry)
            edge_consumed = max(
                0.0,
                min(1.0, (price - ideal_entry) / edge_denominator),
            )
            execution_scorecard = _execution_scorecard(
                context=context,
                scoring=scoring,
                economics=economics,
                parent_setup=parent_setup,
                entry_chase_atr=entry_chase_atr,
            )
            adjusted_score = max(
                0.0, min(100.0, sum(execution_scorecard.values()))
            )
            cost_penalty = max(
                0.0,
                15.0 - execution_scorecard["target_net_rr"],
            )
            adjusted_tier = _tier(adjusted_score)
            hard_blockers = list(scoring["hard_blockers"])
            hard_blockers.extend(band_validation["failures"])
            hard_blockers.extend(macro_gate["blockers"])
            if not parent_setup["valid"]:
                hard_blockers.append(str(parent_setup["blocker"]))
            if not economics["positive_after_costs"]:
                hard_blockers.append("NO_POSITIVE_EXIT_PATH_AFTER_ALL_IN_COSTS")
            net_target_1_bps = float(
                economics.get("net_target_1_bps") or 0.0
            )
            net_target_2_bps = float(
                economics.get("net_target_2_bps") or 0.0
            )
            net_rr_1 = float(economics.get("net_rr_target_1") or 0.0)
            net_rr_2 = float(economics.get("net_rr_target_2") or 0.0)
            expected_net_value_bps = float(
                economics.get("expected_net_value_bps") or 0.0
            )
            cost_to_target = float(
                economics.get("cost_to_target_2_ratio") or 1.0
            )
            tier_a_exception = bool(
                adjusted_tier == "A"
                and net_target_2_bps >= TIER_A_MINIMUM_NET_TARGET_2_BPS
                and net_rr_2 >= 2.0
                and expected_net_value_bps > 0.0
                and parent_setup["valid"]
                and entry_chase_atr is not None
                and entry_chase_atr <= 0.35
            )
            advisory_warnings: list[str] = []
            if net_target_2_bps < PREFERRED_NET_SWING_UPSIDE_BPS:
                advisory_warnings.append(
                    "NET_SWING_UPSIDE_BELOW_PREFERRED_3_PERCENT"
                )
            if scoring["inputs"].get("spread_within_dynamic_cap") is False:
                advisory_warnings.append(
                    "SPREAD_ABOVE_MARKET_NORMAL_P75_ADVISORY"
                )
            if not tier_a_exception:
                if net_target_1_bps < NORMAL_MINIMUM_NET_TARGET_1_BPS:
                    hard_blockers.append("NET_TARGET_1_BELOW_1_25_PERCENT")
                if net_target_2_bps < NORMAL_MINIMUM_NET_TARGET_2_BPS:
                    hard_blockers.append("NET_TARGET_2_BELOW_2_25_PERCENT")
                if net_rr_1 < 1.0:
                    hard_blockers.append("NET_RR_TARGET_1_BELOW_1_0")
                if net_rr_2 < 1.5:
                    hard_blockers.append("NET_RR_TARGET_2_BELOW_1_5")
            if expected_net_value_bps <= 0.0:
                hard_blockers.append("CONSERVATIVE_EXPECTED_VALUE_NOT_POSITIVE")
            if cost_to_target > MAXIMUM_COST_TO_TARGET_RATIO:
                hard_blockers.append("ROUNDTRIP_COST_ABOVE_25_PERCENT_OF_TARGET")
            if entry_chase_atr is None:
                hard_blockers.append("ENTRY_CHASE_ATR_NOT_AVAILABLE")
            elif entry_chase_atr > 0.35:
                hard_blockers.append("ENTRY_CHASE_ABOVE_0_35_ATR")
            if price > maximum_acceptable_entry:
                hard_blockers.append("MAXIMUM_ACCEPTABLE_ENTRY_EXCEEDED")
            state = (
                OpportunityState.ENTRY_READY
                if adjusted_tier in {"A", "B"}
                and not hard_blockers
                else OpportunityState.ARMED
                if adjusted_score >= 75
                else OpportunityState.WATCHING
                if adjusted_score >= 65
                else OpportunityState.DISCOVERED
            )
            hard_blockers = list(dict.fromkeys(hard_blockers))
            near_entry = bool(
                state is OpportunityState.ARMED
                or (
                    state is OpportunityState.WATCHING
                    and adjusted_score >= 65.0
                )
            )
            gate_matrix = {
                "SETUP": "PASS",
                "MTF": "PASS" if parent_setup["valid"] else "FAIL",
                "STOP": "PASS" if 0.0 < stop < price else "FAIL",
                "TARGET": (
                    "PASS"
                    if take_profit_2 > take_profit_1 > price
                    else "FAIL"
                ),
                "SPREAD": (
                    "FAIL"
                    if "SPREAD_ABOVE_MICRO_LIVE_CAP" in hard_blockers
                    else "PASS"
                ),
                "LIQUIDITY": (
                    "FAIL"
                    if any(
                        reason in hard_blockers
                        for reason in (
                            "SLIPPAGE_ABOVE_MICRO_LIVE_CAP",
                            "INSUFFICIENT_EXIT_DEPTH",
                        )
                    )
                    else "PASS"
                ),
                "ORDERFLOW": scoring["microstructure_state"],
                "NET_EV": (
                    "PASS" if expected_net_value_bps > 0.0 else "FAIL"
                ),
                "STRATEGY_AUTHORITY": "PENDING_RUNTIME_CHECK",
            }
            next_required_condition = (
                hard_blockers[0]
                if hard_blockers
                else "ENTRY_PERSISTENCE"
                if state is OpportunityState.ENTRY_READY
                else "SCORE_OR_SETUP_STRENGTHENING"
            )
            episode_seconds = max(
                300,
                _setup_validity_minutes(context.get("timeframe")) * 60,
            )
            episode_bucket = int(observed_datetime.timestamp()) // episode_seconds
            episode_id = stable_hash(
                {
                    "market": market,
                    "playbook_dna": playbook.dna,
                    "context_strategy": context.get("strategy"),
                    "episode_bucket": episode_bucket,
                },
                length=40,
            )
            identity = stable_hash(
                {
                    "market": market,
                    "playbook_dna": playbook.dna,
                    "event_bucket": (
                        context.get("signal_timestamp")
                        or observed_datetime.replace(
                            minute=(observed_datetime.minute // 15) * 15,
                            second=0,
                            microsecond=0,
                        ).isoformat()
                    ),
                },
                length=40,
            )
            original_blockers = [
                reason
                for reason in scoring["hard_blockers"]
                if reason != "INSUFFICIENT_REALTIME_CONFIRMATIONS"
            ]
            if int(scoring["confirmation_count"]) < 3:
                original_blockers.append("INSUFFICIENT_REALTIME_CONFIRMATIONS")
            original_entry_ready = (
                _tier(float(scoring["raw_score"])) in {"A", "B"}
                and not original_blockers
                and not band_validation["failures"]
            )
            blocked_filter = next(
                (
                    reason
                    for reason in (
                        "INSUFFICIENT_REALTIME_CONFIRMATIONS"
                        if int(scoring["independent_confirmation_count"]) < 3
                        else None,
                        "NO_POSITIVE_EXIT_PATH_AFTER_ALL_IN_COSTS"
                        if not economics["positive_after_costs"]
                        else None,
                        *(hard_blockers or []),
                    )
                    if reason
                ),
                None,
            )
            opportunity = {
                    "opportunity_id": identity,
                    "episode_id": episode_id,
                    "episode_bucket_seconds": episode_seconds,
                    "market": market,
                    "playbook_id": playbook.playbook_id,
                    "family": playbook.family,
                    "playbook_dna": playbook.dna,
                    "state": state.value,
                    "score": adjusted_score,
                    "raw_score": scoring["raw_score"],
                    "rule_score_before_costs": scoring["score"],
                    "confirmation_redundancy_penalty": scoring[
                        "confirmation_redundancy_penalty"
                    ],
                    "cost_penalty": cost_penalty,
                    "tier": adjusted_tier,
                    "score_components": scoring["components"],
                    "execution_scorecard": execution_scorecard,
                    "execution_scorecard_version": "ACTIVE_SWING_100_V1",
                    "execution_quality_score": scoring[
                        "execution_quality_score"
                    ],
                    "execution_quality_components": scoring[
                        "execution_quality_components"
                    ],
                    "microstructure_state": scoring[
                        "microstructure_state"
                    ],
                    "microstructure_hostile_groups": scoring[
                        "microstructure_hostile_groups"
                    ],
                    "microstructure_quality": scoring[
                        "microstructure_quality"
                    ],
                    "confirmations": scoring["confirmations"],
                    "confirmation_count": scoring["confirmation_count"],
                    "independent_confirmation_count": scoring[
                        "independent_confirmation_count"
                    ],
                    "confirmation_groups": scoring["confirmation_groups"],
                    "confirmation_independence_ratio": scoring[
                        "confirmation_independence_ratio"
                    ],
                    "hard_blockers": hard_blockers,
                    "near_entry": near_entry,
                    "warm_candidate": near_entry,
                    "gate_matrix": gate_matrix,
                    "next_required_condition": next_required_condition,
                    "advisory_warnings": advisory_warnings,
                    "tier_a_economic_exception": tier_a_exception,
                    "parameter_band_status": band_validation["status"],
                    "validated_parameters": band_validation["observed"],
                    "parameter_band": band_validation["parameter_band"],
                    "parameter_band_hash": band_validation[
                        "parameter_band_hash"
                    ],
                    "macro_regime": macro_regime,
                    "macro_risk_multiplier": scoring[
                        "macro_risk_multiplier"
                    ],
                    "macro_long_lock": macro_gate,
                    "macro_policy": macro_gate["policy"],
                    "playbook_risk_multiplier": min(
                        float(scoring["macro_risk_multiplier"]),
                        float(macro_gate["risk_multiplier_cap"]),
                    ),
                    "entry_price": price,
                    "ideal_entry": ideal_entry,
                    "entry_zone_low": zone_low,
                    "entry_zone_high": zone_high,
                    "maximum_acceptable_entry": maximum_acceptable_entry,
                    "edge_consumed": edge_consumed,
                    "edge_consumed_at_trigger": (
                        edge_consumed
                        if state is OpportunityState.ENTRY_READY
                        else None
                    ),
                    "stop_loss": stop,
                    "take_profit_1": take_profit_1,
                    "take_profit_2": take_profit_2,
                    "execution_economics": economics,
                    "preferred_net_swing_upside_bps": (
                        PREFERRED_NET_SWING_UPSIDE_BPS
                    ),
                    "normal_minimum_net_target_1_bps": (
                        NORMAL_MINIMUM_NET_TARGET_1_BPS
                    ),
                    "normal_minimum_net_target_2_bps": (
                        NORMAL_MINIMUM_NET_TARGET_2_BPS
                    ),
                    "maximum_cost_to_target_ratio": (
                        MAXIMUM_COST_TO_TARGET_RATIO
                    ),
                    "higher_timeframe_parent": parent_setup,
                    "entry_chase_atr": entry_chase_atr,
                    "urgency_class": (
                        "FAST"
                        if playbook.family
                        in {
                            "MOMENTUM_BREAKOUT",
                            "RANGE_EXPANSION_VOLUME",
                            "VOLATILITY_EXPANSION",
                            "BREAKOUT_PULLBACK",
                            "RS_LEADER_PULLBACK",
                            "RELATIVE_STRENGTH_ROTATION",
                        }
                        else "MEDIUM"
                        if playbook.family
                        in {
                            "VWAP_RECLAIM",
                            "TREND_PULLBACK",
                            "NORMAL_SWING_BREAKOUT_RETEST",
                        }
                        else "SLOW"
                    ),
                    "decision_trace": {
                        "original_rule_decision": (
                            "ENTRY_READY" if original_entry_ready else "BLOCKED"
                        ),
                        "deduplicated_decision": "PENDING_CLUSTER_SELECTION",
                        "cost_adjusted_decision": state.value,
                        "final_live_decision": "PENDING_AUTHORITY_AND_PORTFOLIO_RISK",
                        "would_have_entered_before_changes": original_entry_ready,
                        "would_have_entered_after_changes": (
                            state is OpportunityState.ENTRY_READY
                        ),
                        "filter_that_blocked_trade": blocked_filter,
                    },
                    "time_stop_minutes": _swing_time_stop_minutes(
                        context.get("timeframe")
                    ),
                    "detected_at": observed_at,
                    "market_event_ts": (
                        row.get("last_event_at")
                        or row.get("observed_at")
                        or observed_at
                    ),
                    "setup_detected_ts": (
                        context.get("signal_timestamp") or observed_at
                    ),
                    "near_entry_ts": observed_at if near_entry else None,
                    "trigger_ts": (
                        observed_at
                        if state is OpportunityState.ENTRY_READY
                        else None
                    ),
                    "last_updated_at": observed_at,
                    "valid_until": (
                        datetime.fromisoformat(
                            observed_at.replace("Z", "+00:00")
                        )
                        + timedelta(
                            minutes=_setup_validity_minutes(
                                context.get("timeframe")
                            )
                        )
                    ).astimezone(UTC).isoformat(),
                    "context_strategy": context.get("strategy"),
                    "context_timeframe": context.get("timeframe"),
                    # The realtime score is always built from the closed 1m
                    # observation window in ``score_realtime_market``.  Keep
                    # that provenance explicit when a dynamic mover has no
                    # tactical closed-candle parent context yet.  It is not a
                    # substitute for the missing parent gate; such entries
                    # remain blocked by ``MISSING_VALID_1H_4H_PARENT_SETUP``.
                    "observation_timeframe": "1m",
                    "trade_type": context.get("trade_type"),
                    "market_mode": context.get("market_mode"),
                    "weighted_timeframe_score": context.get(
                        "weighted_timeframe_score"
                    ),
                    "fast_timeframe_score": context.get(
                        "fast_timeframe_score"
                    ),
                    "slow_timeframe_score": context.get(
                        "slow_timeframe_score"
                    ),
                    "timeframe_disagreement": context.get(
                        "timeframe_disagreement"
                    ),
                    "hard_blocked_by_1d_or_1w": False,
                    "context_trigger_confirmed": bool(
                        context.get("entry_trigger_confirmed")
                    ),
                    "closed_candle_setup_valid": bool(
                        parent_setup.get("closed_candle_setup_valid")
                    ),
                    "execution_trigger_source": (
                        "CLOSED_CANDLE_AND_REALTIME"
                        if context.get("entry_trigger_confirmed") is True
                        else "REALTIME_AFTER_CLOSED_CANDLE_SETUP"
                    ),
                    "realtime_inputs": scoring["inputs"],
                    "synthetic_data_used": False,
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            opportunity["feature_snapshot"] = freeze_feature_snapshot(
                opportunity
            )
            opportunity["active_swing_contract"] = normalize_live_opportunity(
                opportunity,
                decision_time=observed_at,
            ).model_dump(mode="json")
            output.append(opportunity)
    return sorted(
        output,
        key=lambda item: (
            item["state"] == OpportunityState.ENTRY_READY.value,
            item["score"],
        ),
        reverse=True,
    )


class OpportunityLifecycleLedger:
    """Append-only, restart-safe opportunity state with a current projection."""

    def __init__(self, *, ledger_path: Path, state_path: Path) -> None:
        self.ledger_path = ledger_path
        self.state_path = state_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.ledger_path.is_file():
            return "0" * 64
        try:
            with self.ledger_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                position = stream.tell() - 1
                while position >= 0:
                    stream.seek(position)
                    if stream.read(1) not in {b"\n", b"\r"}:
                        break
                    position -= 1
                end = position + 1
                while position >= 0:
                    stream.seek(position)
                    if stream.read(1) == b"\n":
                        position += 1
                        break
                    position -= 1
                stream.seek(max(0, position))
                last = stream.read(end - max(0, position)).decode("utf-8")
                return str(json.loads(last).get("record_hash") or "0" * 64)
        except (OSError, TypeError, ValueError):
            return "0" * 64

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_path.is_file():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        opportunities: dict[str, dict[str, Any]] = {}
        for key, value in (payload.get("opportunities") or {}).items():
            row = dict(value)
            # A live-ready projection created before parameter-band authority
            # existed must be revalidated from fresh facts.  Never inherit an
            # executable state across an execution-governance migration.
            if (
                row.get("state") == OpportunityState.ENTRY_READY.value
                and not row.get("parameter_band_hash")
            ):
                row["state"] = OpportunityState.ARMED.value
                row["parameter_band_status"] = "REVALIDATION_PENDING"
                row["hard_blockers"] = list(
                    dict.fromkeys(
                        [
                            *(row.get("hard_blockers") or []),
                            "PLAYBOOK_BAND_REVALIDATION_PENDING",
                        ]
                    )
                )
                row["persistence_pending"] = True
            opportunities[str(key)] = row
        return opportunities

    @staticmethod
    def _terminal_timestamp(row: Mapping[str, Any]) -> datetime | None:
        for key in ("last_updated_at", "last_seen_at", "detected_at"):
            raw = row.get(key)
            if raw is None:
                continue
            try:
                selected = datetime.fromisoformat(
                    str(raw).replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if selected.tzinfo is None:
                selected = selected.replace(tzinfo=UTC)
            return selected.astimezone(UTC)
        return None

    def _compact_historical_terminal_rows(self) -> int:
        """Replace old rejected setups with duplicate-safe tombstones.

        The append-only ledger remains the historical source of truth.  The
        JSON projection is restart state, not a second event archive.  Keeping
        multi-kilobyte feature snapshots for every expired 15-minute identity
        caused each small state transition to rewrite hundreds of megabytes.
        Current-day rows stay complete for the daily audit; executed/closed
        rows stay complete for fill attribution.  Only older opportunities
        that never reached execution are compacted.
        """

        today = utc_now().astimezone(UTC).date()
        compacted = 0
        for identity, row in list(self._state.items()):
            try:
                state = OpportunityState(str(row.get("state")))
            except ValueError:
                continue
            if state not in COMPACTABLE_TERMINAL_STATES:
                continue
            timestamp = self._terminal_timestamp(row)
            if timestamp is None or timestamp.date() >= today:
                continue
            if row.get("terminal_projection_compacted") is True:
                continue
            tombstone = {
                key: row.get(key)
                for key in TERMINAL_TOMBSTONE_FIELDS
                if row.get(key) is not None
            }
            tombstone.setdefault("opportunity_id", identity)
            tombstone["terminal_projection_compacted"] = True
            tombstone["historical_detail_source"] = str(self.ledger_path)
            self._state[identity] = tombstone
            compacted += 1
        return compacted

    def _append(self, payload: Mapping[str, Any]) -> None:
        body = {
            "schema_version": "opportunity_lifecycle_event_v1",
            "recorded_at": utc_now().isoformat(),
            "previous_hash": self._last_hash,
            **dict(payload),
        }
        record = {**body, "record_hash": stable_hash(body, length=64)}
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._last_hash = record["record_hash"]

    def upsert(self, opportunity: Mapping[str, Any]) -> bool:
        identity = str(opportunity["opportunity_id"])
        current = self._state.get(identity)
        # Once execution owns an opportunity, scanner refreshes may never move
        # it back to an entry state.  Execution/reconciliation must perform all
        # later transitions explicitly.  This also prevents a closed identity
        # from becoming a second live order when the same market facts recur.
        if current is not None and str(current.get("state")) in {
            state.value for state in EXECUTION_OWNED_STATES
        }:
            return False
        selected = dict(opportunity)
        proposed_state = str(selected["state"])
        selected_state = proposed_state
        now = utc_now().astimezone(UTC)
        selected["last_seen_at"] = now.isoformat()
        current_score = float(selected.get("score") or 0.0)
        previous_score = (
            float(current.get("score") or 0.0) if current is not None else None
        )
        score_delta = (
            current_score - previous_score
            if previous_score is not None
            else 0.0
        )
        previous_delta = (
            float(current.get("score_delta") or 0.0)
            if current is not None
            else 0.0
        )
        selected["score_delta"] = score_delta
        selected["score_acceleration"] = score_delta - previous_delta
        if proposed_state == OpportunityState.ENTRY_READY.value:
            candidate_since = (
                current.get("entry_ready_candidate_since")
                if current
                else None
            )
            if not candidate_since:
                candidate_since = now.isoformat()
            try:
                candidate_at = datetime.fromisoformat(
                    str(candidate_since).replace("Z", "+00:00")
                ).astimezone(UTC)
            except ValueError:
                candidate_at = now
                candidate_since = now.isoformat()
            selected["entry_ready_candidate_since"] = candidate_since
            selected["entry_persistence_seconds"] = max(
                0.0,
                (now - candidate_at).total_seconds(),
            )
            if selected["entry_persistence_seconds"] < ENTRY_PERSISTENCE_SECONDS:
                selected_state = OpportunityState.ARMED.value
                selected["state"] = selected_state
                selected["persistence_pending"] = True
            else:
                selected["persistence_pending"] = False
        else:
            selected.pop("entry_ready_candidate_since", None)
            selected.pop("entry_persistence_seconds", None)
            selected["persistence_pending"] = False
        changed = (
            current is None
            or current.get("state") != selected_state
            or current.get("tier") != selected.get("tier")
            or current.get("parameter_band_status")
            != selected.get("parameter_band_status")
            or current.get("parameter_band_hash")
            != selected.get("parameter_band_hash")
            or tuple(current.get("hard_blockers") or ())
            != tuple(selected.get("hard_blockers") or ())
            or abs(float(current.get("score") or 0) - float(selected.get("score") or 0))
            >= 3
        )
        if not changed:
            # Fresh observations must still refresh liveness even when no
            # material lifecycle field changed.  This timestamp is used to
            # invalidate a setup that disappears from the realtime matcher;
            # it deliberately does not create a ledger event every second.
            current["last_seen_at"] = now.isoformat()
            return False
        self._append(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": identity,
                "market": selected.get("market"),
                "playbook_id": selected.get("playbook_id"),
                "playbook_dna": selected.get("playbook_dna"),
                "from_state": current.get("state") if current else None,
                "to_state": selected_state,
                "score": selected.get("score"),
                "tier": selected.get("tier"),
                "hard_blockers": selected.get("hard_blockers"),
                "next_required_condition": selected.get(
                    "next_required_condition"
                ),
                "microstructure_state": selected.get(
                    "microstructure_state"
                ),
                "episode_id": selected.get("episode_id"),
                "score_delta": selected.get("score_delta"),
                "score_acceleration": selected.get("score_acceleration"),
                "cluster_id": selected.get("cluster_id"),
                "economic_bet_id": selected.get("economic_bet_id"),
                "cluster_size": selected.get("cluster_size"),
                "execution_economics": selected.get(
                    "execution_economics"
                ),
                "decision_trace": selected.get("decision_trace"),
                "feature_snapshot": selected.get("feature_snapshot"),
            }
        )
        self._state[identity] = selected
        self.write_projection()
        return True

    def invalidate_absent(
        self,
        current_opportunity_ids: Iterable[str],
        *,
        observed_at: datetime | None = None,
        grace_seconds: float = ENTRY_PERSISTENCE_SECONDS,
    ) -> list[dict[str, Any]]:
        """Invalidate pre-entry setups no longer confirmed by realtime facts."""

        now = (observed_at or utc_now()).astimezone(UTC)
        seen = {str(value) for value in current_opportunity_ids}
        invalidated: list[dict[str, Any]] = []
        for identity, row in list(self._state.items()):
            if identity in seen or str(row.get("state")) not in {
                item.value for item in PRE_ENTRY_STATES
            }:
                continue
            timestamp = (
                row.get("last_seen_at")
                or row.get("last_updated_at")
                or row.get("detected_at")
            )
            try:
                last_seen = datetime.fromisoformat(
                    str(timestamp).replace("Z", "+00:00")
                ).astimezone(UTC)
            except (TypeError, ValueError):
                last_seen = now - timedelta(seconds=grace_seconds)
            if (now - last_seen).total_seconds() < grace_seconds:
                continue
            invalidated.append(
                self.transition(
                    identity,
                    OpportunityState.INVALIDATED,
                    reason_codes=(
                        "REALTIME_PLAYBOOK_FACTS_NO_LONGER_MATCH",
                        "NO_STALE_ENTRY_REUSE",
                    ),
                    details={
                        "last_seen_at": last_seen.isoformat(),
                        "invalidated_at": now.isoformat(),
                    },
                )
            )
        return invalidated

    def transition(
        self,
        opportunity_id: str,
        state: OpportunityState,
        *,
        reason_codes: Iterable[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self._state.get(opportunity_id)
        if current is None:
            raise KeyError(f"unknown opportunity: {opportunity_id}")
        prior = str(current.get("state"))
        updated = {
            **current,
            "state": state.value,
            "last_updated_at": utc_now().isoformat(),
            "reason_codes": list(reason_codes),
            **dict(details or {}),
        }
        self._append(
            {
                "event_type": "OPPORTUNITY_TRANSITION",
                "opportunity_id": opportunity_id,
                "market": current.get("market"),
                "playbook_id": current.get("playbook_id"),
                "playbook_dna": current.get("playbook_dna"),
                "from_state": prior,
                "to_state": state.value,
                "reason_codes": list(reason_codes),
                "details": dict(details or {}),
            }
        )
        self._state[opportunity_id] = updated
        self.write_projection()
        return updated

    def expire(self, *, observed_at: datetime | None = None) -> int:
        now = (observed_at or utc_now()).astimezone(UTC)
        expired = 0
        for identity, row in list(self._state.items()):
            # Entry validity applies only before an order intent exists.  A
            # submitted, partially filled or managed position remains owned by
            # execution until reconciliation records an explicit terminal
            # transition; it must never silently expire on an entry deadline.
            if str(row.get("state")) not in {
                item.value for item in PRE_ENTRY_STATES
            }:
                continue
            valid_until = datetime.fromisoformat(
                str(row["valid_until"]).replace("Z", "+00:00")
            ).astimezone(UTC)
            if valid_until < now:
                self.transition(
                    identity,
                    OpportunityState.EXPIRED,
                    reason_codes=("OPPORTUNITY_TIME_LIMIT_EXPIRED",),
                )
                expired += 1
        return expired

    def recover_orphan_order_intents(
        self,
        *,
        reconciliation_ready: bool,
        observed_at: datetime | None = None,
    ) -> int:
        """Release intents that reconciliation proves never reached Bitvavo.

        `ORDER_INTENT_CREATED` is execution-owned so scanner refreshes cannot
        accidentally submit it twice.  If the process dies before the venue
        call returns, that protection would otherwise strand the opportunity
        forever.  A green startup reconciliation proves there is no unknown
        remote order/position; the opportunity can therefore return to
        WATCHING for a completely fresh persistence and microstructure check.
        """

        if not reconciliation_ready:
            return 0
        now = (observed_at or utc_now()).astimezone(UTC)
        recovered = 0
        for identity, row in list(self._state.items()):
            if str(row.get("state")) != OpportunityState.ORDER_INTENT_CREATED.value:
                continue
            try:
                valid_until = datetime.fromisoformat(
                    str(row["valid_until"]).replace("Z", "+00:00")
                ).astimezone(UTC)
            except (KeyError, ValueError):
                valid_until = now - timedelta(seconds=1)
            if valid_until < now:
                target = OpportunityState.EXPIRED
                reason = "ORPHAN_INTENT_EXPIRED_AFTER_RECONCILIATION"
            else:
                target = OpportunityState.WATCHING
                reason = "ORPHAN_INTENT_RELEASED_FOR_FRESH_REVALIDATION"
            self.transition(
                identity,
                target,
                reason_codes=(
                    reason,
                    "NO_REMOTE_ORDER_OR_POSITION_AFTER_RECONCILIATION",
                ),
                details={
                    "orphan_intent_recovered": True,
                    "duplicate_submission_allowed": False,
                    "fresh_persistence_required": target is OpportunityState.WATCHING,
                },
            )
            recovered += 1
        return recovered

    def write_projection(self) -> None:
        compacted_terminal_count = self._compact_historical_terminal_rows()
        active = {
            key: value
            for key, value in self._state.items()
            if str(value.get("state"))
            not in {item.value for item in TERMINAL_STATES}
        }
        near_entry = sorted(
            (
                dict(value)
                for value in active.values()
                if value.get("near_entry") is True
                or (
                    str(value.get("state"))
                    in {
                        OpportunityState.ARMED.value,
                        OpportunityState.WATCHING.value,
                    }
                    and float(value.get("score") or 0.0) >= 65.0
                )
            ),
            key=lambda row: float(row.get("score") or 0.0),
            reverse=True,
        )
        atomic_write_json(
            self.state_path,
            {
                "schema_version": "opportunity_lifecycle_state_v1",
                "updated_at": utc_now().isoformat(),
                "opportunities": self._state,
                "active_count": len(active),
                "entry_ready_count": sum(
                    row.get("state") == OpportunityState.ENTRY_READY.value
                    for row in active.values()
                ),
                "near_entry_count": len(near_entry),
                "near_entry_markets": list(
                    dict.fromkeys(
                        str(row.get("market")) for row in near_entry
                    )
                ),
                "near_entry": near_entry,
                "compacted_terminal_count": sum(
                    row.get("terminal_projection_compacted") is True
                    for row in self._state.values()
                ),
                "compacted_terminal_count_this_write": compacted_terminal_count,
                "historical_detail_source": str(self.ledger_path),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )

    @property
    def state(self) -> Mapping[str, Mapping[str, Any]]:
        return self._state


__all__ = [
    "OpportunityLifecycleLedger",
    "OpportunityState",
    "PLAYBOOKS",
    "PlaybookSpec",
    "build_event_driven_opportunities",
    "macro_playbook_gate",
    "macro_risk_multiplier",
    "score_realtime_market",
]
