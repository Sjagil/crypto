"""Current, orderless evidence for active-swing product sections 440-443.

Every value comes from a durable point-in-time artifact.  Missing evidence is
reported as missing; this module never fills it with a favourable assumption.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from config.settings import ACTIVE_SWING_TIMEFRAMES, Settings
from core.active_trading import active_trading_status
from reporting.prospective_net_r import build_prospective_net_r_calibration
from utils.common import atomic_write_json, read_json, utc_iso

ZERO = Decimal("0")
TERMINAL_STATES = {"CLOSED", "INVALIDATED", "EXPIRED"}
LIFECYCLE_MAP = {
    "DISCOVERED": "DISCOVERED",
    "WATCHING": "WATCHING",
    "ARMED": "ENTRY_TRIGGER_PENDING",
    "ENTRY_READY": "ENTRY_READY",
    "ORDER_INTENT_CREATED": "ENTERED",
    "ORDER_SUBMITTED": "ENTERED",
    "PARTIALLY_FILLED": "ENTERED",
    "FILLED": "POSITION_ACTIVE",
    "MANAGING": "POSITION_ACTIVE",
    "EXITING": "EXIT_READY",
    "CLOSED": "CLOSED",
    "INVALIDATED": "INVALIDATED",
    "EXPIRED": "EXPIRED",
}
PRODUCT_LIFECYCLE = (
    "DISCOVERED",
    "WATCHING",
    "NEAR_SETUP",
    "SETUP_VALID",
    "ENTRY_TRIGGER_PENDING",
    "ENTRY_READY",
    "ENTERED",
    "POSITION_ACTIVE",
    "PROFIT_PROTECTION",
    "REDUCE_CANDIDATE",
    "ROTATION_CANDIDATE",
    "EXIT_READY",
    "CLOSED",
    "INVALIDATED",
    "EXPIRED",
)


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(row) for row in value.values() if isinstance(row, Mapping)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _age_seconds(value: Any, *, observed_at: Any) -> int | None:
    start = _timestamp(value)
    end = _timestamp(observed_at)
    if start is None or end is None or start > end:
        return None
    return int((end - start).total_seconds())


def _market_row(universe: Mapping[str, Any], market: str) -> dict[str, Any]:
    return next(
        (row for row in _rows(universe.get("rows")) if row.get("market") == market),
        {},
    )


def _mechanics_row(mechanics: Mapping[str, Any], market: str) -> dict[str, Any]:
    markets = mechanics.get("markets")
    if isinstance(markets, Mapping) and isinstance(markets.get(market), Mapping):
        return dict(markets[market])
    return next((row for row in _rows(markets) if row.get("market") == market), {})


def _roundtrip_costs(
    candidate: Mapping[str, Any],
    account: Mapping[str, Any],
    mechanics: Mapping[str, Any],
) -> dict[str, Any]:
    market = str(candidate.get("market") or "")
    market_fees = dict((account.get("market_fee_rates") or {}).get(market) or {})
    taker = _decimal(
        market_fees.get("taker_rate")
        or candidate.get("estimated_fee_fraction")
    )
    slippage_bps = _decimal(candidate.get("estimated_slippage_bps"))
    market_mechanics = _mechanics_row(mechanics, market)
    orderflow = dict(market_mechanics.get("orderflow_15m") or {})
    spread_bps = _decimal(orderflow.get("spread_bps"))
    roundtrip_fee = taker * Decimal("2") if taker is not None else None
    roundtrip_slippage = (
        slippage_bps * Decimal("2") / Decimal("10000")
        if slippage_bps is not None
        else None
    )
    spread_fraction = (
        spread_bps / Decimal("10000") if spread_bps is not None else None
    )
    parts = (roundtrip_fee, roundtrip_slippage, spread_fraction)
    all_in = sum(parts, ZERO) if all(value is not None for value in parts) else None
    return {
        "fee_rate_one_way": str(taker) if taker is not None else None,
        "roundtrip_fee_fraction": (
            str(roundtrip_fee) if roundtrip_fee is not None else None
        ),
        "slippage_bps_one_way": (
            str(slippage_bps) if slippage_bps is not None else None
        ),
        "roundtrip_slippage_fraction": (
            str(roundtrip_slippage) if roundtrip_slippage is not None else None
        ),
        "spread_bps": str(spread_bps) if spread_bps is not None else None,
        "all_in_cost_fraction": str(all_in) if all_in is not None else None,
        "quote_status": orderflow.get("status") or "MISSING",
        "quote_available_at": orderflow.get("available_at"),
        "orderflow_reason_codes": list(orderflow.get("reason_codes") or []),
    }


def _candidate_economics(
    candidate: Mapping[str, Any],
    costs: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    entry = _decimal(candidate.get("current_price"))
    stop = _decimal(candidate.get("stop"))
    target_1 = _decimal(candidate.get("target_1"))
    target_2 = _decimal(candidate.get("target_2"))
    all_in_cost = _decimal(costs.get("all_in_cost_fraction"))
    valid = bool(
        entry is not None
        and stop is not None
        and target_1 is not None
        and target_2 is not None
        and ZERO < stop < entry < target_1 < target_2
    )
    risk = entry - stop if valid and entry is not None and stop is not None else None
    gross_target_return = (
        (target_2 - entry) / entry
        if valid and entry is not None and target_2 is not None
        else None
    )
    conservative_net_target_return = (
        gross_target_return - all_in_cost
        if gross_target_return is not None and all_in_cost is not None
        else None
    )
    net_reward_risk = (
        conservative_net_target_return / (risk / entry + all_in_cost)
        if conservative_net_target_return is not None
        and conservative_net_target_return > ZERO
        and risk is not None
        and entry is not None
        and all_in_cost is not None
        else None
    )
    return {
        "levels_valid": valid,
        "risk_per_unit": str(risk) if risk is not None else None,
        "gross_target_2_return": (
            str(gross_target_return) if gross_target_return is not None else None
        ),
        "conservative_net_target_2_return": (
            str(conservative_net_target_return)
            if conservative_net_target_return is not None
            else None
        ),
        "net_reward_risk_to_target_2": (
            str(net_reward_risk) if net_reward_risk is not None else None
        ),
        "expected_net_r": calibration.get("expected_net_r"),
        "expected_net_r_status": calibration.get("status")
        or "NO_PROSPECTIVE_CALIBRATED_EXPECTATION",
        "expected_net_r_calibration": dict(calibration),
        "positive_target_after_costs": bool(
            conservative_net_target_return is not None
            and conservative_net_target_return > ZERO
        ),
    }


def _current_candidate_test(
    *,
    active: Mapping[str, Any],
    account: Mapping[str, Any],
    external: Mapping[str, Any],
    universe: Mapping[str, Any],
    mechanics: Mapping[str, Any],
    ml: Mapping[str, Any],
    runtime: Mapping[str, Any],
    net_r_calibration: Mapping[str, Any],
    product_economically_good: bool,
) -> dict[str, Any]:
    observed_at = active.get("generated_at")
    macro = dict(active.get("macro") or {})
    candidates = [
        *_rows(active.get("top_5_actionable")),
        *_rows(active.get("top_5_near_entry")),
    ]
    seen: set[str] = set()
    evaluations: list[dict[str, Any]] = []
    for candidate in candidates:
        opportunity_id = str(candidate.get("opportunity_id") or "")
        if not opportunity_id or opportunity_id in seen:
            continue
        seen.add(opportunity_id)
        market = str(candidate.get("market") or "")
        market_info = _market_row(universe, market)
        costs = _roundtrip_costs(candidate, account, mechanics)
        calibration = dict(
            (
                net_r_calibration.get("estimates_by_opportunity_id")
                or {}
            ).get(opportunity_id)
            or {}
        )
        economics = _candidate_economics(candidate, costs, calibration)
        minimum_eur = _decimal(
            (account.get("venue_safe_minimums_eur") or {}).get(market)
        )
        theoretical_eur = Decimal("25")
        entry = _decimal(candidate.get("current_price"))
        theoretical_quantity = (
            theoretical_eur / entry if entry is not None and entry > ZERO else None
        )
        market_timeframes = dict(candidate.get("timeframe_score_details") or {})
        missing_timeframes = [
            timeframe
            for timeframe in ACTIVE_SWING_TIMEFRAMES
            if timeframe not in market_timeframes
        ]
        blockers: list[str] = []
        if candidate.get("live_authority_granted") is not True:
            blockers.append("STRATEGY_AUTHORITY_NOT_GRANTED")
        if account.get("entry_allowed") is not True:
            blockers.append("ACCOUNT_ENTRY_NOT_ALLOWED")
        blockers.extend(str(value) for value in account.get("failures") or [])
        if external.get("status") == "OPERATOR_DECISION_REQUIRED":
            blockers.append("EXTERNAL_INVENTORY_OPERATOR_DECISION_REQUIRED")
        if runtime.get("control_state") != "ENABLED":
            blockers.append(f"CONTROL_STATE_{runtime.get('control_state') or 'UNKNOWN'}")
        if not product_economically_good:
            blockers.append("PRODUCT_ECONOMICS_NOT_POSITIVE")
        if economics["expected_net_r"] is None:
            blockers.append("EXPECTED_NET_R_NOT_CALIBRATED")
        elif _decimal(economics["expected_net_r"]) <= ZERO:
            blockers.append("EXPECTED_NET_R_NOT_POSITIVE")
        if costs["quote_status"] != "READY":
            blockers.append("FRESH_EXECUTION_QUOTE_NOT_READY")
        if missing_timeframes:
            blockers.append("REQUIRED_TIMEFRAME_STATE_MISSING")
        if market_info.get("shariah_status") != "ALLOWED":
            blockers.append("SHARIAH_NOT_ALLOWED")
        blockers = list(dict.fromkeys(blockers))
        final_decision = "NO_TRADE" if blockers else "ELIGIBLE_FOR_CANONICAL_CHAIN"
        evaluations.append(
            {
                "rank": len(evaluations) + 1,
                "market": market,
                "asset": market.split("-")[0] if "-" in market else market,
                "opportunity_id": opportunity_id,
                "discovery_reason": candidate.get("trigger_reason"),
                "strategy_id": candidate.get("strategy"),
                "strategy_family": candidate.get("family"),
                "strategy_dna_hash": candidate.get("strategy_dna_hash"),
                "entry_timeframe": candidate.get("entry_timeframe"),
                "setup_timeframe": candidate.get("confirmation_timeframe"),
                "structural_timeframe": candidate.get("regime_timeframe"),
                "higher_timeframe_context": market_timeframes,
                "missing_timeframes": missing_timeframes,
                "btc_context": {
                    "regime": macro.get("regime"),
                    "status": macro.get("status"),
                    "confidence": macro.get("confidence"),
                    "btc_4h_trend_up": (macro.get("features") or {}).get(
                        "btc_4h_trend_up"
                    ),
                    "btc_1d_trend_up": (macro.get("features") or {}).get(
                        "btc_1d_trend_up"
                    ),
                    "btc_return_24h": (macro.get("features") or {}).get(
                        "btc_return_24h"
                    ),
                },
                "signal_timestamp": candidate.get("signal_timestamp"),
                "signal_age_seconds": _age_seconds(
                    candidate.get("signal_timestamp"), observed_at=observed_at
                ),
                "setup_age_seconds": _age_seconds(
                    candidate.get("signal_timestamp"), observed_at=observed_at
                ),
                "entry_price_reference": candidate.get("current_price"),
                "stop": candidate.get("stop"),
                "target_1": candidate.get("target_1"),
                "target_2": candidate.get("target_2"),
                "economics": economics,
                "costs": costs,
                "ml": {
                    "evaluated": False,
                    "prediction": None,
                    "authority": ml.get("authority") or "SHADOW_ONLY",
                    "status": "NOT_MAPPED_TO_CURRENT_CANDIDATE",
                },
                "rl": {
                    "evaluated": False,
                    "advisory": None,
                    "authority": "SHADOW_ONLY",
                    "status": "NOT_ELIGIBLE",
                },
                "uncertainty": "HIGH",
                "theoretical_allocation_eur_before_gates": str(theoretical_eur),
                "requested_eur_allocation": "0",
                "theoretical_fractional_quantity": (
                    str(theoretical_quantity)
                    if theoretical_quantity is not None
                    else None
                ),
                "bitvavo_realizable_quantity": None,
                "quantity_precision_status": "NOT_EVALUATED_BECAUSE_PRE_RISK_BLOCKED",
                "minimum_order_eur": (
                    str(minimum_eur) if minimum_eur is not None else None
                ),
                "minimum_order_feasible_before_gates": bool(
                    minimum_eur is not None and theoretical_eur >= minimum_eur
                ),
                "retail_realizable": False,
                "portfolio_fit": "BLOCKED",
                "correlation_cluster": "CRYPTO_COMMON_BETA_UNRESOLVED",
                "shariah_status": market_info.get("shariah_status") or "UNKNOWN",
                "authority_status": "BLOCKED" if blockers else "READY",
                "reconciliation_healthy": (account.get("reconciliation") or {}).get(
                    "healthy"
                ),
                "final_decision": final_decision,
                "blockers": blockers,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        )
    return {
        "schema_version": "current_active_swing_end_to_end_test_v1",
        "generated_at": utc_iso(),
        "source_observed_at": observed_at,
        "status": "ASSESSED" if evaluations else "NO_CURRENT_CANDIDATE",
        "candidate_count": len(evaluations),
        "candidates": evaluations[:5],
        "natural_signal_forced": False,
        "financial_state_changed": False,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _position_management(
    *,
    output: Path,
    active: Mapping[str, Any],
    account: Mapping[str, Any],
    mechanics: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = output / "live" / "generated_strategy_live_state.json"
    state = _mapping(state_path)
    positions = [
        row
        for row in _rows(state.get("positions"))
        if str(row.get("status") or "").upper() == "OPEN"
    ]
    holdings = list(
        ((account.get("account") or {}).get("portfolio_valuation") or {}).get(
            "holdings"
        )
        or []
    )
    current_candidates = [
        *_rows(active.get("top_5_actionable")),
        *_rows(active.get("top_5_near_entry")),
    ]
    macro = dict(active.get("macro") or {})
    results: list[dict[str, Any]] = []
    for position in positions:
        market = str(position.get("market") or "")
        holding = next(
            (dict(row) for row in holdings if row.get("market") == market), {}
        )
        candidate = next(
            (row for row in current_candidates if row.get("market") == market), {}
        )
        mechanics_row = _mechanics_row(mechanics, market)
        orderflow = dict(mechanics_row.get("orderflow_15m") or {})
        entry = _decimal(position.get("entry_price"))
        stop = _decimal(position.get("stop_loss"))
        target_2 = _decimal(position.get("take_profit_2"))
        mark = _decimal(holding.get("price_eur"))
        quantity = _decimal(position.get("quantity"))
        original_risk = entry - stop if entry and stop and entry > stop else None
        unrealized_r = (
            (mark - entry) / original_risk
            if mark is not None and entry is not None and original_risk
            else None
        )
        remaining_gross_r = (
            (target_2 - mark) / (mark - stop)
            if target_2 is not None
            and mark is not None
            and stop is not None
            and target_2 > mark > stop
            else None
        )
        gross_pnl = (
            (mark - entry) * quantity
            if mark is not None and entry is not None and quantity is not None
            else None
        )
        protected = bool(
            position.get("native_protective_stop_active") is True
            and position.get("protective_stop_status") == "awaitingTrigger"
        )
        tp1_reached = position.get("tp1_reached") is True
        replacements = [
            {
                "market": row.get("market"),
                "strategy_id": row.get("strategy"),
                "score": row.get("score"),
                "live_authority_granted": row.get("live_authority_granted"),
            }
            for row in _rows(active.get("top_5_actionable"))
            if row.get("market") != market
        ][:3]
        results.append(
            {
                "market": market,
                "entry_strategy": position.get("strategy_id"),
                "strategy_dna_hash": position.get("strategy_dna_hash"),
                "entry_timeframe": position.get("timeframe"),
                "setup_timeframe": "NOT_PERSISTED_LEGACY_POSITION",
                "entry_thesis": "NOT_PERSISTED_LEGACY_POSITION",
                "current_thesis": (
                    "TP1_REACHED_AND_NATIVE_STOP_ACTIVE"
                    if tp1_reached and protected
                    else "PROTECTION_REVIEW_REQUIRED"
                ),
                "timeframe_state": {
                    timeframe: (candidate.get("timeframe_score_details") or {}).get(
                        timeframe
                    )
                    for timeframe in ("15m", "1h", "4h", "1d")
                },
                "btc_context": {
                    "regime": macro.get("regime"),
                    "status": macro.get("status"),
                    "btc_4h_trend_up": (macro.get("features") or {}).get(
                        "btc_4h_trend_up"
                    ),
                    "btc_1d_trend_up": (macro.get("features") or {}).get(
                        "btc_1d_trend_up"
                    ),
                },
                "market_context": {
                    "trade_type": candidate.get("trade_type"),
                    "timeframe_conflicts": list(
                        candidate.get("timeframe_conflicts") or []
                    ),
                    "macro_support": candidate.get("macro_support"),
                },
                "managed_quantity": position.get("quantity"),
                "entry_price": position.get("entry_price"),
                "current_mark_price": holding.get("price_eur"),
                "gross_unrealized_pnl_eur": (
                    str(gross_pnl) if gross_pnl is not None else None
                ),
                "unrealized_r": str(unrealized_r) if unrealized_r is not None else None,
                "updated_expected_net_r": None,
                "updated_expected_net_r_status": (
                    "NO_PROSPECTIVE_CALIBRATED_EXPECTATION"
                ),
                "remaining_gross_reward_risk": (
                    str(remaining_gross_r) if remaining_gross_r is not None else None
                ),
                "stop_loss": position.get("stop_loss"),
                "take_profit_1": position.get("take_profit_1"),
                "take_profit_2": position.get("take_profit_2"),
                "tp1_reached": tp1_reached,
                "spread_bps": orderflow.get("spread_bps"),
                "liquidity_status": orderflow.get("status") or "MISSING",
                "event_state": "NO_POSITION_SPECIFIC_EVENT_EVIDENCE",
                "better_replacement_candidates": replacements,
                "native_protective_stop_active": protected,
                "protective_stop_trigger": position.get("protective_stop_trigger"),
                "lifecycle": "PROFIT_PROTECTION" if tp1_reached else "POSITION_ACTIVE",
                "proposed_action": (
                    "KEEP" if protected else "REDUCE_OR_EXIT_REVIEW_REQUIRED"
                ),
                "deterministic_proposed_action": (
                    "HOLD_WITH_NATIVE_PROTECTIVE_STOP"
                    if protected
                    else "RISK_REDUCTION_REVIEW_REQUIRED"
                ),
                "action_detail": (
                    "HOLD_WITH_NATIVE_PROTECTIVE_STOP"
                    if protected
                    else "NO_CONFIRMED_NATIVE_PROTECTION"
                ),
                "rl_advisory": None,
                "rl_authority": "SHADOW_ONLY",
                "execution_authority_changed": False,
            }
        )
    return {
        "schema_version": "current_position_management_test_v2",
        "generated_at": utc_iso(),
        "status": "ASSESSED" if positions else "NO_OPEN_MANAGED_POSITION_REQUIRED",
        "source": str(state_path),
        "managed_open_position_count": len(positions),
        "positions": results,
        "cash_is_active_competitor": True,
        "rotation_hysteresis_applied": True,
        "financial_state_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _funnel(
    *,
    output: Path,
    active: Mapping[str, Any],
    account: Mapping[str, Any],
    universe: Mapping[str, Any],
    end_to_end: Mapping[str, Any],
) -> dict[str, Any]:
    data_health = dict(active.get("data_health") or {})
    timeframe_health = dict(data_health.get("timeframes") or {})
    stage_counts = dict((active.get("execution_funnel") or {}).get("stage_counts") or {})
    candidates = _rows(end_to_end.get("candidates"))
    blocker_counts = Counter(
        blocker for row in candidates for blocker in row.get("blockers") or []
    )
    lifecycle_source = _mapping(output / "live" / "opportunity_lifecycle_state.json")
    lifecycle_rows = [
        row
        for row in _rows(lifecycle_source.get("opportunities"))
        if str(row.get("state") or "") not in TERMINAL_STATES
    ]
    lifecycle_counter = Counter(
        LIFECYCLE_MAP.get(str(row.get("state") or ""), "DISCOVERED")
        for row in lifecycle_rows
    )
    lifecycle_counts = {
        state: int(lifecycle_counter.get(state, 0)) for state in PRODUCT_LIFECYCLE
    }
    return {
        "schema_version": "active_swing_current_funnel_v2",
        "generated_at": utc_iso(),
        "status": "READY" if active else "DATA_MISSING",
        "source_observed_at": active.get("generated_at"),
        "eligible_markets_scanned": len(universe.get("live_executable_markets") or []),
        "markets_scanned": int(active.get("market_count") or 0),
        "lifecycle_counts": lifecycle_counts,
        "timeframe_freshness": {
            timeframe: dict(timeframe_health.get(timeframe) or {})
            for timeframe in ACTIVE_SWING_TIMEFRAMES
        },
        "setups": int(stage_counts.get("strategy_setups") or 0),
        "near_setups": int(stage_counts.get("near_entry") or 0),
        "entry_triggers": int(stage_counts.get("entry_ready") or 0),
        "ml_evaluated": 0,
        "ml_supported": 0,
        "rl_evaluated": 0,
        "positive_net_target_after_costs": sum(
            bool((row.get("economics") or {}).get("positive_target_after_costs"))
            for row in candidates
        ),
        "positive_expected_net_r": sum(
            (_decimal((row.get("economics") or {}).get("expected_net_r")) or ZERO)
            > ZERO
            for row in candidates
        ),
        "retail_realizable": sum(row.get("retail_realizable") is True for row in candidates),
        "portfolio_feasible": sum(row.get("portfolio_fit") == "PASS" for row in candidates),
        "shariah_valid": sum(row.get("shariah_status") == "ALLOWED" for row in candidates),
        "risk_valid": sum(
            bool((row.get("economics") or {}).get("levels_valid")) for row in candidates
        ),
        "authority_valid": sum(row.get("authority_status") == "READY" for row in candidates),
        "quote_valid": sum((row.get("costs") or {}).get("quote_status") == "READY" for row in candidates),
        "reconciliation_valid": (account.get("reconciliation") or {}).get("healthy") is True,
        "submitted": int(active.get("orders_submitted") or 0),
        "filled": int((active.get("execution") or {}).get("fills_verified_this_cycle") or 0),
        "top_rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in blocker_counts.most_common(10)
        ],
        "poor_setups_used_as_fillers": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _economic_gate(execution_evidence: Mapping[str, Any]) -> dict[str, Any]:
    paper = dict(execution_evidence.get("simulated_execution_pnl") or {})
    thresholds = {
        "minimum_closed_round_trips": 30,
        "minimum_profit_factor": 1.20,
        "minimum_net_expectancy_eur": "0",
    }
    rows: list[dict[str, Any]] = []
    for family, values in sorted(dict(paper.get("by_playbook") or {}).items()):
        metrics = dict(values or {})
        trades = int(metrics.get("closed_round_trips") or 0)
        profit_factor = _decimal(metrics.get("closed_position_profit_factor"))
        expectancy = _decimal(metrics.get("paper_net_expectancy_eur"))
        checks = {
            "minimum_sample": trades >= thresholds["minimum_closed_round_trips"],
            "profit_factor": bool(
                profit_factor is not None
                and profit_factor >= Decimal(str(thresholds["minimum_profit_factor"]))
            ),
            "net_expectancy": bool(expectancy is not None and expectancy > ZERO),
        }
        rows.append(
            {
                "strategy_family": family,
                "closed_round_trips": trades,
                "profit_factor": (
                    str(profit_factor) if profit_factor is not None else None
                ),
                "net_expectancy_eur": str(expectancy) if expectancy is not None else None,
                "checks": checks,
                "paper_forward_eligible": all(checks.values()),
                "live_authority_granted": False,
            }
        )
    eligible = [row for row in rows if row["paper_forward_eligible"]]
    aggregate_expectancy = _decimal(paper.get("net_expectancy_eur"))
    return {
        "schema_version": "active_swing_economic_recovery_gate_v1",
        "generated_at": utc_iso(),
        "status": (
            "PAPER_FORWARD_CANDIDATE_REVIEW_REQUIRED"
            if eligible
            else "NOT_ECONOMICALLY_GOOD_YET"
        ),
        "thresholds": thresholds,
        "aggregate": {
            "closed_round_trips": paper.get("closed_round_trips"),
            "net_expectancy_eur": paper.get("net_expectancy_eur"),
            "net_pnl_eur": paper.get("net_pnl_eur"),
            "positive": bool(aggregate_expectancy is not None and aggregate_expectancy > ZERO),
        },
        "families": rows,
        "eligible_family_count": len(eligible),
        "eligible_families": [row["strategy_family"] for row in eligible],
        "automatic_live_promotion_permitted": False,
        "operator_review_required_after_all_gates": True,
        "tests_or_backtests_are_live_profitability_proof": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _alpha_dashboard(
    *,
    active: Mapping[str, Any],
    opportunities: Mapping[str, Any],
    execution_evidence: Mapping[str, Any],
    position_management: Mapping[str, Any],
    end_to_end: Mapping[str, Any],
    account: Mapping[str, Any],
) -> dict[str, Any]:
    actual = dict(execution_evidence.get("actual_live_pnl") or {})
    paper = dict(execution_evidence.get("simulated_execution_pnl") or {})
    theoretical = dict(execution_evidence.get("theoretical_signal_pnl") or {})
    deterministic = _rows(end_to_end.get("candidates"))
    families = Counter(
        str(row.get("strategy_family") or "UNKNOWN") for row in deterministic
    )
    return {
        "schema_version": "active_swing_alpha_dashboard_v2",
        "generated_at": utc_iso(),
        "status": "READY" if active else "DATA_MISSING",
        "source_observed_at": active.get("generated_at"),
        "top_deterministic_candidates": deterministic,
        "top_ml_candidates": [],
        "top_ml_candidates_status": "NO_CURRENT_CANDIDATE_LEVEL_ML_PREDICTION",
        "top_cross_sectional_ranks": _rows(
            opportunities.get("top_5_relative_strength")
        )[:5],
        "top_emerging_leaders": _rows(opportunities.get("top_5_early_moves"))[:5],
        "top_gem_candidates": [],
        "top_gem_candidates_status": "NO_GEM_PASSED_FULL_ECONOMIC_CONTRACT",
        "top_event_candidates": _rows(opportunities.get("top_5_defensive"))[:5],
        "strategy_family_leadership": dict(families),
        "current_positions": list(position_management.get("positions") or []),
        "rotation_candidates": _rows(active.get("top_5_rotation"))[:5],
        "eur_cash_decision": {
            "available_eur": (account.get("account") or {}).get("eur_available"),
            "decision": "KEEP_CASH_AVAILABLE",
            "reason": "NO_CANDIDATE_PASSED_ECONOMICS_PORTFOLIO_AUTHORITY_AND_ACCOUNT_GATES",
        },
        "actual_live": {
            "status": actual.get("status"),
            "closed_round_trips": actual.get("closed_round_trips"),
            "realised_pnl_eur": actual.get("realised_pnl_eur"),
            "unrealised_pnl_eur": actual.get("unrealised_pnl_eur"),
            "net_pnl_eur": actual.get("net_pnl_eur"),
            "fees_eur": actual.get("fees_eur"),
        },
        "paper_execution": {
            "status": paper.get("status"),
            "closed_round_trips": paper.get("closed_round_trips"),
            "net_expectancy_eur": paper.get("net_expectancy_eur"),
            "net_pnl_eur": paper.get("net_pnl_eur"),
            "fees_eur": paper.get("fees_eur"),
        },
        "theoretical_signal": {
            "status": theoretical.get("status"),
            "resolved_episode_count": theoretical.get("resolved_episode_count"),
            "false_breakout_rate": theoretical.get("false_breakout_rate"),
            "gross_or_net_pnl_eur": theoretical.get("gross_or_net_pnl_eur"),
        },
        "layers_are_not_interchangeable": True,
        "dashboard_mutates_portfolio_state": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_current_active_swing_evidence(
    settings: Settings,
    *,
    runtime: Mapping[str, Any],
    product_economically_good: bool,
) -> dict[str, Any]:
    """Build all current evidence artifacts without entering the money path."""

    output = settings.paths.output_dir
    product = output / "product"
    active = active_trading_status(settings)
    opportunities = _mapping(output / "active_trading" / "opportunities.json")
    mechanics = _mapping(output / "active_trading" / "market_mechanics.json")
    account = _mapping(output / "operations" / "live_account_health.json")
    external = _mapping(output / "operations" / "external_inventory_remediation.json")
    universe = _mapping(output / "universe" / "tiered_trading_universe.json")
    ml = _mapping(output / "ml" / "canonical_training_status.json")
    execution_evidence = _mapping(
        output / "operations" / "execution_evidence_layers.json"
    )
    current_candidates = [
        *_rows(active.get("top_5_actionable")),
        *_rows(active.get("top_5_near_entry")),
    ]
    net_r_calibration = build_prospective_net_r_calibration(
        settings,
        current_candidates,
        as_of=_timestamp(active.get("generated_at")) or datetime.now(UTC),
    )
    end_to_end = _current_candidate_test(
        active=active,
        account=account,
        external=external,
        universe=universe,
        mechanics=mechanics,
        ml=ml,
        runtime=runtime,
        net_r_calibration=net_r_calibration,
        product_economically_good=product_economically_good,
    )
    positions = _position_management(
        output=output,
        active=active,
        account=account,
        mechanics=mechanics,
    )
    funnel = _funnel(
        output=output,
        active=active,
        account=account,
        universe=universe,
        end_to_end=end_to_end,
    )
    economics = _economic_gate(execution_evidence)
    alpha = _alpha_dashboard(
        active=active,
        opportunities=opportunities,
        execution_evidence=execution_evidence,
        position_management=positions,
        end_to_end=end_to_end,
        account=account,
    )
    payload = {
        "schema_version": "active_swing_current_evidence_v1",
        "generated_at": utc_iso(),
        "end_to_end": end_to_end,
        "position_management": positions,
        "funnel": funnel,
        "alpha_dashboard": alpha,
        "economic_recovery_gate": economics,
        "prospective_net_r_calibration": net_r_calibration,
        "financial_state_changed": False,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(product / "current_end_to_end_test.json", end_to_end)
    atomic_write_json(product / "current_position_management_test.json", positions)
    atomic_write_json(product / "opportunity_funnel.json", funnel)
    atomic_write_json(product / "alpha_dashboard.json", alpha)
    atomic_write_json(product / "economic_recovery_gate.json", economics)
    atomic_write_json(product / "current_evidence.json", payload)
    return payload


__all__ = ["build_current_active_swing_evidence"]
