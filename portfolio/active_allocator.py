"""Active portfolio construction from the unified whole-stack packet.

Paper mode may actively allocate, rotate, reduce and exit a virtual wallet.
Live-plan mode is stricter: it can create canonical InvestmentIntent and
PortfolioTarget objects, but it never approves risk or submits an exchange order.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping


from config.settings import Settings
from core.live_capital import (
    MAXIMUM_ORDER_EUR,
    MAXIMUM_RISK_PER_TRADE_EUR,
    MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR,
)
from portfolio.contracts import InvestmentDirection, InvestmentIntent
from portfolio.targets import construct_portfolio_target
from utils.common import stable_hash


DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": "active_portfolio_policy_v1",
    "paper": {
        "entry_score": 0.20,
        "reduce_score": 0.00,
        "exit_score": -0.25,
        "minimum_alignment": 0.66,
        "maximum_positions": 3,
        "required_entry_timeframes": ["15m", "1h", "4h"],
        "risk_reduce_fraction": 0.50,
    },
    "live": {
        "require_selector_qualified": True,
        "require_market_selector_pass": True,
        "require_research_survivor": True,
        "require_positive_stressed_edge": True,
        "required_entry_timeframes": ["15m", "1h", "4h"],
    },
}


def _f(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def _d(value: Any) -> Decimal:
    try:
        selected = Decimal(str(value))
    except Exception:
        return Decimal("0")
    return selected if selected.is_finite() else Decimal("0")


def load_policy(path: Path | None = None) -> dict[str, Any]:
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if path is None or not path.is_file():
        return policy
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("active portfolio policy must be a JSON object")
    for section in ("paper", "live"):
        if isinstance(payload.get(section), dict):
            policy[section].update(payload[section])
    return policy


@dataclass(frozen=True)
class WalletPosition:
    market: str
    quantity: float
    mark_price: float

    @property
    def notional_eur(self) -> float:
        return max(0.0, self.quantity) * max(0.0, self.mark_price)


@dataclass(frozen=True)
class WalletState:
    equity_eur: float
    cash_eur: float
    positions: tuple[WalletPosition, ...]

    @property
    def exposure_eur(self) -> float:
        return sum(row.notional_eur for row in self.positions)

    def by_market(self) -> dict[str, WalletPosition]:
        return {row.market: row for row in self.positions}


def wallet_from_mapping(payload: Mapping[str, Any]) -> WalletState:
    raw_positions = payload.get("positions") or {}
    rows: list[WalletPosition] = []
    if isinstance(raw_positions, Mapping):
        iterable = [
            {"market": market, **dict(value)}
            for market, value in raw_positions.items()
            if isinstance(value, Mapping)
        ]
    else:
        iterable = [dict(value) for value in raw_positions if isinstance(value, Mapping)]
    for row in iterable:
        market = str(row.get("market") or "").upper()
        quantity = _f(row.get("quantity")) or 0.0
        mark = _f(row.get("mark_price") or row.get("price")) or 0.0
        if market and quantity >= 0 and mark > 0:
            rows.append(WalletPosition(market, quantity, mark))
    equity = _f(payload.get("equity_eur")) or 0.0
    cash = _f(payload.get("cash_eur")) or 0.0
    if equity <= 0 or cash < 0:
        raise ValueError("wallet requires positive equity and non-negative cash")
    return WalletState(equity, cash, tuple(rows))


def _market_packet(packet: Mapping[str, Any], market: str) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = dict(packet.get("canonical_context") or {})
    advisory = dict(packet.get("swing_advisory") or {})
    canonical_market = dict((canonical.get("market_context") or {}).get(market) or {})
    advisory_market = dict((advisory.get("market_advisory") or {}).get(market) or {})
    return canonical_market, advisory_market


def _timeframe_fresh(canonical_market: Mapping[str, Any], timeframe: str) -> bool:
    state = dict((canonical_market.get("timeframes") or {}).get(timeframe) or {})
    quality = dict(state.get("data_quality") or {})
    return bool(quality.get("fresh")) and quality.get("provenance", {}).get("status") == "READY"


def _latest_price(canonical_market: Mapping[str, Any]) -> float | None:
    states = dict(canonical_market.get("timeframes") or {})
    for timeframe in ("15m", "1h", "2h", "4h", "1d"):
        value = _f(dict(states.get(timeframe) or {}).get("close"))
        if value is not None and value > 0:
            return value
    return None


def _signal_row(
    packet: Mapping[str, Any],
    market: str,
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    canonical, advisory = _market_packet(packet, market)
    mtf = dict(canonical.get("mtf") or {})
    states = dict(canonical.get("timeframes") or {})
    descriptive = dict(advisory.get("descriptive_advisory") or {})
    selector = dict(advisory.get("prospective_selector") or {})
    score = _f(descriptive.get("score"))
    alignment = _f(mtf.get("alignment_score_01")) or 0.0
    h1 = _f(dict(states.get("1h") or {}).get("technical_score"))
    h4 = _f(dict(states.get("4h") or {}).get("technical_score"))
    d1 = _f(dict(states.get("1d") or {}).get("technical_score"))
    price = _latest_price(canonical)
    required = tuple(policy["paper"]["required_entry_timeframes"])
    fresh = all(_timeframe_fresh(canonical, tf) for tf in required)
    cmc_ready = dict(advisory.get("fundamentals_cmc") or {}).get("status") == "READY"
    news = dict(advisory.get("news") or {})
    news_confidence = _f(news.get("confidence")) or 0.0

    trend_confirmed = (
        h1 is not None
        and h4 is not None
        and h1 > 0
        and h4 > 0
    )
    strength = 0.0
    if score is not None and score > 0:
        strength = score * max(0.25, alignment)
        if d1 is not None and d1 > 0:
            strength *= 1.10
        if cmc_ready:
            strength *= 1.05
        if news.get("status") == "READY":
            strength *= 1.0 + min(0.10, news_confidence * 0.10)

    paper_candidate = bool(
        fresh
        and score is not None
        and score >= float(policy["paper"]["entry_score"])
        and alignment >= float(policy["paper"]["minimum_alignment"])
        and trend_confirmed
        and price is not None
    )
    hard_exit = bool(
        score is not None
        and (
            score <= float(policy["paper"]["exit_score"])
            or (
                h1 is not None
                and h4 is not None
                and h1 <= -0.45
                and h4 <= -0.45
            )
            or str(dict(states.get("4h") or {}).get("breakout_state") or "") == "BREAKDOWN_20"
            or str(dict(states.get("1d") or {}).get("breakout_state") or "") == "BREAKDOWN_20"
        )
    )
    reduce = bool(
        not hard_exit
        and score is not None
        and (
            score <= float(policy["paper"]["reduce_score"])
            or (
                h1 is not None
                and h4 is not None
                and h1 < 0
                and h4 < 0
            )
        )
    )
    return {
        "market": market,
        "price": price,
        "score": score,
        "strength": strength,
        "alignment": alignment,
        "h1": h1,
        "h4": h4,
        "d1": d1,
        "fresh_for_entry": fresh,
        "cmc_ready": cmc_ready,
        "news_status": news.get("status", "MISSING"),
        "selector_status": selector.get("status"),
        "selector_action": selector.get("action"),
        "selector_passes": bool(selector.get("passes")),
        "paper_candidate": paper_candidate,
        "reduce": reduce,
        "hard_exit": hard_exit,
    }


def _waterfill_targets(
    candidates: list[dict[str, Any]],
    *,
    equity_eur: float,
    exposure_limit_eur: float,
    position_cap_eur: float,
) -> dict[str, float]:
    if not candidates or exposure_limit_eur <= 0 or position_cap_eur <= 0:
        return {}
    remaining = min(exposure_limit_eur, equity_eur)
    targets = {row["market"]: 0.0 for row in candidates}
    active = list(candidates)
    for _ in range(len(candidates) + 2):
        if remaining <= 1e-9 or not active:
            break
        total_strength = sum(max(1e-9, float(row["strength"])) for row in active)
        next_active: list[dict[str, Any]] = []
        spent = 0.0
        for row in active:
            market = row["market"]
            share = remaining * max(1e-9, float(row["strength"])) / total_strength
            capacity = max(0.0, position_cap_eur - targets[market])
            allocation = min(capacity, share)
            targets[market] += allocation
            spent += allocation
            if targets[market] + 1e-9 < position_cap_eur:
                next_active.append(row)
        remaining -= spent
        if spent <= 1e-9:
            break
        active = next_active
    return targets


def build_active_portfolio_plan(
    packet: Mapping[str, Any],
    wallet: WalletState,
    settings: Settings,
    *,
    mode: str = "paper",
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"paper", "live-plan"}:
        raise ValueError("mode must be paper or live-plan")
    selected_policy = dict(policy or DEFAULT_POLICY)
    markets = [str(value).upper() for value in packet.get("markets") or []]
    signals = [
        _signal_row(packet, market, policy=selected_policy)
        for market in markets
    ]
    by_signal = {row["market"]: row for row in signals}
    current = wallet.by_market()

    maximum_positions = min(
        int(selected_policy["paper"]["maximum_positions"]),
        int(settings.risk.maximum_portfolio_exposure / max(settings.risk.maximum_position_fraction, 1e-12)),
    )
    maximum_positions = max(1, min(maximum_positions, 3))
    candidates = sorted(
        [row for row in signals if row["paper_candidate"]],
        key=lambda row: (-float(row["strength"]), row["market"]),
    )[:maximum_positions]

    if mode == "paper":
        exposure_cap = wallet.equity_eur * settings.risk.maximum_portfolio_exposure
        position_cap = wallet.equity_eur * settings.risk.maximum_position_fraction
    else:
        exposure_cap = float(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR)
        position_cap = float(MAXIMUM_ORDER_EUR)

    desired = _waterfill_targets(
        candidates,
        equity_eur=wallet.equity_eur,
        exposure_limit_eur=exposure_cap,
        position_cap_eur=position_cap,
    )

    # Existing positions are not silently liquidated just because they did not
    # rank into the entry set. They move only on explicit reduction/exit state.
    for market, position in current.items():
        signal = by_signal.get(market)
        current_notional = position.notional_eur
        if signal is None:
            desired.setdefault(market, current_notional)
            continue
        if signal["hard_exit"]:
            desired[market] = 0.0
        elif signal["reduce"]:
            desired[market] = min(
                desired.get(market, current_notional),
                current_notional * float(selected_policy["paper"]["risk_reduce_fraction"]),
            )
        else:
            desired.setdefault(market, current_notional)

    selector_policy = dict((packet.get("swing_advisory") or {}).get("selector_policy") or {})
    research = dict((packet.get("canonical_context") or {}).get("research_evidence") or {})
    stressed_bps = _f(selector_policy.get("mean_stressed_net_bps"))
    selector_qualified = bool(selector_policy.get("qualified"))
    research_survivor = int(research.get("passed_gate_count") or 0) > 0

    decisions: list[dict[str, Any]] = []
    live_targets: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    validity = now + timedelta(minutes=15)
    portfolio_hash = stable_hash(
        {
            "equity_eur": wallet.equity_eur,
            "cash_eur": wallet.cash_eur,
            "positions": [asdict(row) for row in wallet.positions],
        },
        length=64,
    )

    all_markets = sorted(set(desired) | set(current))
    for market in all_markets:
        signal = by_signal.get(market) or {"price": None}
        price = _f(signal.get("price"))
        old_position = current.get(market)
        current_notional = old_position.notional_eur if old_position else 0.0
        target_notional = max(0.0, float(desired.get(market, current_notional)))
        delta = target_notional - current_notional
        tolerance = max(1.0, wallet.equity_eur * 0.001)
        action = (
            "HOLD"
            if abs(delta) <= tolerance
            else "ENTRY"
            if current_notional <= tolerance and delta > 0
            else "ADD"
            if delta > 0
            else "FULL_EXIT"
            if target_notional <= tolerance
            else "REDUCE"
        )

        live_reasons: list[str] = []
        if action in {"ENTRY", "ADD"}:
            if not bool(signal.get("fresh_for_entry")):
                live_reasons.append("ENTRY_DATA_NOT_FRESH")
            if not selector_qualified:
                live_reasons.append("SELECTOR_POLICY_NOT_QUALIFIED")
            if not bool(signal.get("selector_passes")):
                live_reasons.append("MARKET_SELECTOR_ABSTAINED")
            if not research_survivor:
                live_reasons.append("NO_CANONICAL_RESEARCH_SURVIVOR")
            if stressed_bps is None or stressed_bps <= 0:
                live_reasons.append("POSITIVE_STRESSED_EDGE_NOT_PROVEN")
        elif action in {"REDUCE", "FULL_EXIT"}:
            # Risk-reducing actions do not need positive entry edge, but they
            # still need current market/account/reconciliation evidence before
            # an exchange executor may act.
            if price is None or price <= 0:
                live_reasons.append("EXIT_MARK_PRICE_MISSING")

        row = {
            "market": market,
            "action": action,
            "current_notional_eur": current_notional,
            "target_notional_eur": target_notional,
            "delta_notional_eur": delta,
            "mark_price": price,
            "signal": signal,
            "paper_exploration": action in {"ENTRY", "ADD"} and not (
                selector_qualified and research_survivor
            ),
            "live_plan_eligible": not live_reasons,
            "live_blockers": live_reasons,
        }
        decisions.append(row)

        if mode == "live-plan" and not live_reasons and price and price > 0 and action != "HOLD":
            current_qty = old_position.quantity if old_position else 0.0
            if action in {"ENTRY", "ADD"}:
                expected_return = Decimal(str(stressed_bps / 10_000.0))
                direction = InvestmentDirection.LONG
            elif action == "FULL_EXIT":
                expected_return = Decimal("0")
                direction = InvestmentDirection.FLAT
            else:
                expected_return = Decimal("0")
                direction = InvestmentDirection.REDUCE

            intent = InvestmentIntent.create(
                market=market,
                direction=direction,
                confidence=Decimal(str(min(1.0, max(0.0, float(signal.get("alignment") or 0.0))))),
                expected_return=expected_return,
                expected_risk=None,
                horizon_seconds=24 * 60 * 60,
                strategy_id="ACTIVE_PORTFOLIO_UNIFIED_V1",
                family="WHOLE_STACK_PORTFOLIO",
                generated_at=now,
                valid_until=validity,
                evidence_id=stable_hash(
                    {
                        "whole_stack": packet.get("generated_at"),
                        "market": market,
                        "selector_policy": selector_policy,
                        "signal": signal,
                    },
                    length=40,
                ),
                reason_codes=("UNIFIED_WHOLE_STACK_PORTFOLIO_PLAN",),
            )
            decision = construct_portfolio_target(
                (intent,),
                current_quantity=Decimal(str(current_qty)),
                current_notional_eur=Decimal(str(current_notional)),
                equity_eur=Decimal(str(wallet.equity_eur)),
                mark_price=Decimal(str(price)),
                proposed_target_weight=Decimal(str(target_notional / wallet.equity_eur)),
                risk_budget_eur=MAXIMUM_RISK_PER_TRADE_EUR,
                cluster="CRYPTO_MAJOR",
                decision_time=now,
                expires_at=validity,
                portfolio_state_hash=portfolio_hash,
                cost_model_version="WHOLE_STACK_ACTIVE_PORTFOLIO_V1",
                quantity_step=Decimal("0.00000001"),
            )
            live_targets.append(
                {
                    "investment_intent": intent.model_dump(mode="json"),
                    "portfolio_decision": decision.model_dump(mode="json"),
                }
            )

    deployed_target = sum(row["target_notional_eur"] for row in decisions)
    return {
        "schema_version": "active_portfolio_plan_v1",
        "generated_at": now.isoformat(),
        "mode": mode,
        "wallet": {
            "equity_eur": wallet.equity_eur,
            "cash_eur": wallet.cash_eur,
            "current_exposure_eur": wallet.exposure_eur,
        },
        "limits": {
            "maximum_positions": maximum_positions,
            "target_exposure_cap_eur": exposure_cap,
            "target_position_cap_eur": position_cap,
            "reserve_cash_fraction": settings.risk.reserve_cash_fraction,
            "capital_level_2_order_cap_eur": float(MAXIMUM_ORDER_EUR),
            "capital_level_2_total_cap_eur": float(MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR),
        },
        "target_exposure_eur": deployed_target,
        "target_cash_eur": max(0.0, wallet.equity_eur - deployed_target),
        "signals": signals,
        "decisions": decisions,
        "canonical_live_targets": live_targets,
        "live_entry_gate": {
            "selector_qualified": selector_qualified,
            "research_survivor_available": research_survivor,
            "mean_stressed_net_bps": stressed_bps,
        },
        "authority": {
            "paper_can_trade": mode == "paper",
            "live_orders_submitted": 0,
            "risk_approval_performed": False,
            "exchange_authority": "SJAGIL_CRYPTO_ONLY",
        },
    }


def apply_paper_plan(
    wallet_payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wallet = wallet_from_mapping(wallet_payload)
    positions = {
        row.market: {"quantity": row.quantity, "mark_price": row.mark_price}
        for row in wallet.positions
    }
    cash = wallet.cash_eur
    fills: list[dict[str, Any]] = []
    fee = float(settings.costs.default_fee)
    slippage = float(settings.costs.slippage_bps) / 10_000.0

    # Sells first release wallet capacity.
    ordered = sorted(
        plan.get("decisions") or [],
        key=lambda row: 0 if row.get("delta_notional_eur", 0) < 0 else 1,
    )
    for row in ordered:
        delta = float(row.get("delta_notional_eur") or 0.0)
        price = _f(row.get("mark_price"))
        market = str(row.get("market") or "")
        if not market or price is None or price <= 0 or abs(delta) < 1.0:
            continue
        position = positions.get(market, {"quantity": 0.0, "mark_price": price})
        quantity = float(position.get("quantity") or 0.0)

        if delta < 0 and quantity > 0:
            requested_notional = min(-delta, quantity * price)
            sell_price = price * (1.0 - slippage)
            sell_qty = min(quantity, requested_notional / max(price, 1e-12))
            gross = sell_qty * sell_price
            fee_eur = gross * fee
            cash += gross - fee_eur
            quantity -= sell_qty
            positions[market] = {"quantity": max(0.0, quantity), "mark_price": price}
            fills.append(
                {
                    "market": market,
                    "side": "SELL",
                    "quantity": sell_qty,
                    "price": sell_price,
                    "gross_eur": gross,
                    "fee_eur": fee_eur,
                    "reason": row.get("action"),
                }
            )
        elif delta > 0:
            buy_price = price * (1.0 + slippage)
            affordable = cash / max(buy_price * (1.0 + fee), 1e-12)
            requested_qty = delta / max(price, 1e-12)
            buy_qty = min(affordable, requested_qty)
            if buy_qty <= 0:
                continue
            gross = buy_qty * buy_price
            fee_eur = gross * fee
            total = gross + fee_eur
            if total > cash + 1e-9:
                continue
            cash -= total
            quantity += buy_qty
            positions[market] = {"quantity": quantity, "mark_price": price}
            fills.append(
                {
                    "market": market,
                    "side": "BUY",
                    "quantity": buy_qty,
                    "price": buy_price,
                    "gross_eur": gross,
                    "fee_eur": fee_eur,
                    "reason": row.get("action"),
                    "exploration_only": bool(row.get("paper_exploration")),
                }
            )

    marked = 0.0
    for market, row in positions.items():
        price = _f(row.get("mark_price")) or 0.0
        marked += float(row.get("quantity") or 0.0) * price
    result = {
        "schema_version": "active_paper_wallet_v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "cash_eur": cash,
        "equity_eur": cash + marked,
        "positions": positions,
        "fill_count": int(wallet_payload.get("fill_count") or 0) + len(fills),
    }
    return result, fills
