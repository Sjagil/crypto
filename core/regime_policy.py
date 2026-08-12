"""Shared causal macro-regime policy for tactical and live selection.

Macro context is deliberately an overlay: it may reduce or block an already
valid strategy signal, but it must never manufacture an entry.
"""

from __future__ import annotations


def regime_policy(
    regime: str,
    family: str,
    market: str,
) -> tuple[str, float, str]:
    """Return execution policy, sizing multiplier and an audit reason."""

    selected_regime = str(regime or "DATA_BLOCKED").upper()
    selected_family = str(family or "UNKNOWN").upper()
    selected_market = str(market or "").upper()
    recovery = any(
        key in selected_family
        for key in ("RECOVERY", "REVERSION", "FAILED_BREAKOUT", "SWEEP")
    )
    trend = any(
        key in selected_family
        for key in (
            "TREND",
            "BREAKOUT",
            "MOMENTUM",
            "RELATIVE_STRENGTH",
            "DONCHIAN",
            "TURTLE",
        )
    )
    if selected_regime == "DATA_BLOCKED":
        return "REDUCE", 0.85, "MACRO_CONTEXT_UNAVAILABLE_RISK_REDUCED"
    if selected_regime == "LIQUIDATION_STRESS":
        if recovery:
            return "REDUCE", 0.40, "CONFIRMED_EXHAUSTION_MICRO_RISK"
        return "REDUCE", 0.40, "EXTREME_RISK_OFF_MICRO_RISK"
    if selected_regime == "DELEVERAGING":
        if recovery and selected_market in {"BTC-EUR", "ETH-EUR"}:
            return "REDUCE", 0.40, "DEFENSIVE_RECOVERY_MICRO_RISK"
        return "REDUCE", 0.40, "DELEVERAGING_MICRO_RISK"
    if selected_regime == "MACRO_RISK_OFF":
        # Macro is a sizing overlay only.  It may never erase a causal setup;
        # integrity, liquidity, execution and portfolio risk remain the hard
        # gates.  This prevents the prior failure mode where an ADA/NPC mover
        # was observed correctly but assigned zero execution capacity solely
        # because broad-market context was defensive.
        return "REDUCE", 0.65, "MACRO_RISK_OFF_POSITION_SIZE_REDUCED"
    if selected_regime == "RANGE_LOW_VOL":
        if recovery:
            return "ENABLE", 0.75, "RANGE_FAMILY_FAVOURED"
        if trend:
            return "REDUCE", 0.50, "BREAKOUT_REQUIRES_EXPANSION_CONFIRMATION"
    if selected_regime == "RANGE_HIGH_VOL" and trend:
        return "REDUCE", 0.60, "CHOP_RISK"
    if selected_regime in {"RECOVERY", "VOLATILITY_EXPANSION"}:
        return "ENABLE", 0.75, "SELECTIVE_FAMILY_ENABLED"
    if selected_regime in {
        "STRONG_RISK_ON",
        "MODERATE_RISK_ON",
        "BTC_LED_RISK_ON",
        "ALTCOIN_ROTATION",
    }:
        return "ENABLE", 1.0, "REGIME_SUPPORTIVE"
    return "REDUCE", 0.50, "UNCERTAIN_CONTEXT"


__all__ = ["regime_policy"]
