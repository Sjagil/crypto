"""Fail-closed portfolio risk orchestration for long-only crypto spot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import RiskSettings, Settings, ShariahSettings
from core.contracts import EligibilityStatus, RiskDecision, normalize_market
from research.trading_math import calculate_position_size
from utils.common import append_jsonl, atomic_write_json, read_json, utc_iso


class RiskReason(StrEnum):
    APPROVED = "APPROVED"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    RISK_MANAGER_UNHEALTHY = "RISK_MANAGER_UNHEALTHY"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    DATA_UNHEALTHY = "DATA_UNHEALTHY"
    INTELLIGENCE_TIMING_UNHEALTHY = "INTELLIGENCE_TIMING_UNHEALTHY"
    MARKET_NOT_ALLOWED = "MARKET_NOT_ALLOWED"
    INVALID_PRICE_OR_STOP = "INVALID_PRICE_OR_STOP"
    MINIMUM_STOP_DISTANCE = "MINIMUM_STOP_DISTANCE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    MAXIMUM_TRADES_PER_DAY = "MAXIMUM_TRADES_PER_DAY"
    MAXIMUM_OPEN_RISK = "MAXIMUM_OPEN_RISK"
    MAXIMUM_PORTFOLIO_EXPOSURE = "MAXIMUM_PORTFOLIO_EXPOSURE"
    MAXIMUM_POSITION_FRACTION = "MAXIMUM_POSITION_FRACTION"
    CASH_RESERVE = "CASH_RESERVE"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    ZERO_SIZE = "ZERO_SIZE"
    INSUFFICIENT_OWNED_UNITS = "INSUFFICIENT_OWNED_UNITS"


@dataclass(frozen=True)
class PositionExposure:
    market: str
    quantity: float
    mark_price: float
    open_risk_eur: float

    @property
    def notional_eur(self) -> float:
        return self.quantity * self.mark_price


@dataclass(frozen=True)
class PortfolioSnapshot:
    equity_eur: float
    cash_eur: float
    day_start_equity_eur: float
    peak_equity_eur: float
    trades_today: int
    positions: tuple[PositionExposure, ...] = ()
    reconciled: bool = True
    risk_manager_healthy: bool = True
    data_healthy: bool = True
    intelligence_timing_healthy: bool = True
    returns_by_market: dict[str, pd.Series] = field(default_factory=dict)
    drawdown_state: str = "NORMAL"

    @property
    def exposure_eur(self) -> float:
        return sum(position.notional_eur for position in self.positions)

    @property
    def open_risk_eur(self) -> float:
        return sum(position.open_risk_eur for position in self.positions)

    @property
    def daily_loss_fraction(self) -> float:
        if self.day_start_equity_eur <= 0:
            return 1.0
        return max(0.0, 1.0 - self.equity_eur / self.day_start_equity_eur)

    @property
    def drawdown_fraction(self) -> float:
        if self.peak_equity_eur <= 0:
            return 1.0
        return max(0.0, 1.0 - self.equity_eur / self.peak_equity_eur)


class KillSwitch:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.active = False
        self.reason = ""
        self.activated_at: str | None = None
        if path and path.is_file():
            try:
                payload = read_json(path)
                self.active = bool(payload.get("active", True))
                self.reason = str(payload.get("reason", "UNKNOWN_PERSISTED_STATE"))
                self.activated_at = payload.get("activated_at")
            except (OSError, ValueError, TypeError):
                self.active = True
                self.reason = "UNREADABLE_KILL_SWITCH_STATE"

    def _persist(self) -> None:
        if self.path:
            atomic_write_json(
                self.path,
                {
                    "active": self.active,
                    "reason": self.reason,
                    "activated_at": self.activated_at,
                },
            )

    def activate(self, reason: str) -> None:
        selected = reason.strip() or "MANUAL_KILL_SWITCH"
        self.active = True
        self.reason = selected
        self.activated_at = utc_iso()
        self._persist()

    def reset(self, *, approval_phrase: str, required_phrase: str) -> None:
        if approval_phrase != required_phrase:
            raise PermissionError("kill switch reset approval phrase is invalid")
        self.active = False
        self.reason = ""
        self.activated_at = None
        self._persist()


def reliability_multiplier(evidence: dict[str, float | int | bool | None]) -> float:
    """Conservative 0..1 sizing multiplier based only on OOS/operational evidence."""

    direct = (
        "holdout_expectancy_score",
        "stressed_profit_factor_score",
        "positive_walk_forward_fold_ratio",
        "cpcv_path_consistency",
        "parameter_stability",
        "effective_sample_score",
        "recent_operational_score",
        "data_completeness",
        "provider_health",
    )
    inverse = (
        "monte_carlo_probability_of_loss",
        "symbol_concentration",
        "regime_concentration",
    )
    scores = [
        min(1.0, max(0.0, float(evidence[name])))
        for name in direct
        if evidence.get(name) is not None
    ]
    scores.extend(
        1.0 - min(1.0, max(0.0, float(evidence[name])))
        for name in inverse
        if evidence.get(name) is not None
    )
    if not scores:
        return 0.0
    # A weak mandatory dimension should reduce risk materially; a single good
    # metric can never compensate for missing or poor evidence.
    return float(min(1.0, max(0.0, np.prod(scores) ** (1.0 / len(scores)))))


class OperationalDegradation:
    """Persistent severity state without a second health-monitor subsystem."""

    _rank = {
        "NORMAL": 0,
        "WARNING": 1,
        "REDUCE_RISK": 2,
        "BLOCK_NEW_ENTRIES": 3,
        "KILL_SWITCH": 4,
    }
    _hard_kill_reasons = {
        "HARD_DRAWDOWN_LIMIT",
        "DUPLICATE_ORDER_RISK",
        "NEGATIVE_BALANCE",
        "UNKNOWN_ORDER_STATE",
        "LEDGER_CORRUPTION",
        "SEVERE_RECONCILIATION_FAILURE",
        "EXECUTION_VENUE_DATA_INVALID",
        "UNSAFE_CREDENTIAL_SCOPE",
        "RISK_GATE_BYPASS",
    }

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        audit_path: Path | None = None,
        persistence: int = 2,
    ) -> None:
        if persistence < 1:
            raise ValueError("degradation persistence must be positive")
        self.state_path = state_path
        self.audit_path = audit_path
        self.persistence = persistence
        self.state = "NORMAL"
        self.reason_codes: tuple[str, ...] = ("WITHIN_LIMITS",)
        self.counters: dict[str, int] = {}
        self.manual_reset_required = False
        if state_path and state_path.is_file():
            try:
                payload = read_json(state_path)
                self.state = str(payload["state"])
                self.reason_codes = tuple(payload.get("reason_codes") or ())
                self.counters = {
                    str(key): int(value)
                    for key, value in (payload.get("counters") or {}).items()
                }
                self.manual_reset_required = bool(
                    payload.get("manual_reset_required", False)
                )
            except (OSError, ValueError, TypeError, KeyError):
                self.state = "KILL_SWITCH"
                self.reason_codes = ("UNREADABLE_DEGRADATION_STATE",)
                self.manual_reset_required = True

    def evaluate(
        self,
        *,
        warning: tuple[str, ...] = (),
        reduce_risk: tuple[str, ...] = (),
        block_new_entries: tuple[str, ...] = (),
        kill_switch: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        active = set(warning + reduce_risk + block_new_entries + kill_switch)
        for reason in set(self.counters) | active:
            self.counters[reason] = self.counters.get(reason, 0) + 1 if reason in active else 0
        hard = sorted(active & self._hard_kill_reasons)
        desired = "NORMAL"
        reasons: tuple[str, ...] = ("WITHIN_LIMITS",)
        levels = (
            ("WARNING", warning),
            ("REDUCE_RISK", reduce_risk),
            ("BLOCK_NEW_ENTRIES", block_new_entries),
            ("KILL_SWITCH", kill_switch),
        )
        for level, level_reasons in levels:
            persistent = tuple(
                reason
                for reason in level_reasons
                if self.counters.get(reason, 0) >= self.persistence
            )
            if persistent:
                desired, reasons = level, persistent
        if hard:
            desired, reasons = "KILL_SWITCH", tuple(hard)
        if self.manual_reset_required and self.state in {
            "BLOCK_NEW_ENTRIES",
            "KILL_SWITCH",
        }:
            desired, reasons = self.state, ("MANUAL_RESET_REQUIRED",)
        if desired != self.state:
            previous = self.state
            self.state = desired
            self.reason_codes = reasons
            if desired in {"BLOCK_NEW_ENTRIES", "KILL_SWITCH"}:
                self.manual_reset_required = True
            self._audit(
                {
                    "event": "DEGRADATION_TRANSITION",
                    "previous_state": previous,
                    "state": desired,
                    "reason_codes": reasons,
                    "timestamp": utc_iso(),
                }
            )
        self._persist()
        return self.status()

    def manual_reset(
        self,
        *,
        confirmed: bool,
        reason: str,
        resolved_health_checks: bool,
    ) -> dict[str, Any]:
        if not confirmed:
            raise PermissionError("manual degradation reset requires confirmation")
        if not reason.strip():
            raise ValueError("manual degradation reset requires a reason")
        if not resolved_health_checks:
            raise RuntimeError("health checks are not resolved")
        previous = self.state
        self.state = "NORMAL"
        self.reason_codes = ("MANUAL_RESET",)
        self.counters.clear()
        self.manual_reset_required = False
        self._audit(
            {
                "event": "MANUAL_DEGRADATION_RESET",
                "previous_state": previous,
                "state": self.state,
                "reason": reason.strip(),
                "timestamp": utc_iso(),
                "resolved_health_checks": True,
            }
        )
        self._persist()
        return self.status()

    def status(self) -> dict[str, Any]:
        multiplier = {
            "NORMAL": 1.0,
            "WARNING": 0.75,
            "REDUCE_RISK": 0.5,
            "BLOCK_NEW_ENTRIES": 0.0,
            "KILL_SWITCH": 0.0,
        }[self.state]
        return {
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "risk_multiplier": multiplier,
            "new_entries_allowed": multiplier > 0,
            "manual_reset_required": self.manual_reset_required,
            "counters": dict(self.counters),
            "checked_at": utc_iso(),
        }

    def _persist(self) -> None:
        if self.state_path:
            atomic_write_json(self.state_path, self.status())

    def _audit(self, event: dict[str, Any]) -> None:
        if self.audit_path:
            append_jsonl(self.audit_path, event)


class RiskManager:
    def __init__(
        self,
        *,
        risk: RiskSettings,
        shariah: ShariahSettings,
        fee_fraction_per_side: float,
        slippage_fraction_per_side: float,
        kill_switch: KillSwitch | None = None,
        maximum_correlation: float = 0.85,
    ) -> None:
        self.risk = risk
        self.shariah = shariah
        self.fee_fraction_per_side = fee_fraction_per_side
        self.slippage_fraction_per_side = slippage_fraction_per_side
        self.kill_switch = kill_switch or KillSwitch()
        if not 0 <= maximum_correlation <= 1:
            raise ValueError("maximum_correlation must be between zero and one")
        self.maximum_correlation = maximum_correlation

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        kill_switch_path: Path | None = None,
    ) -> "RiskManager":
        return cls(
            risk=settings.risk,
            shariah=settings.shariah,
            fee_fraction_per_side=settings.costs.default_fee,
            slippage_fraction_per_side=(
                settings.costs.slippage_bps + settings.costs.spread_bps / 2.0
            )
            / 10_000.0,
            kill_switch=KillSwitch(kill_switch_path),
        )

    def _static_reasons(
        self,
        *,
        market: str,
        snapshot: PortfolioSnapshot,
    ) -> list[RiskReason]:
        reasons: list[RiskReason] = []
        if self.kill_switch.active:
            reasons.append(RiskReason.KILL_SWITCH_ACTIVE)
        if not snapshot.risk_manager_healthy:
            reasons.append(RiskReason.RISK_MANAGER_UNHEALTHY)
        if not snapshot.reconciled:
            reasons.append(RiskReason.RECONCILIATION_REQUIRED)
        if not snapshot.data_healthy:
            reasons.append(RiskReason.DATA_UNHEALTHY)
        if not snapshot.intelligence_timing_healthy:
            reasons.append(RiskReason.INTELLIGENCE_TIMING_UNHEALTHY)
        if self.shariah.eligibility(market).status is not EligibilityStatus.ALLOWED:
            reasons.append(RiskReason.MARKET_NOT_ALLOWED)
        if snapshot.daily_loss_fraction >= self.risk.maximum_daily_loss:
            reasons.append(RiskReason.DAILY_LOSS_LIMIT)
        if (
            snapshot.drawdown_state
            in {"BLOCK_NEW_ENTRIES", "KILL_SWITCH"}
            or snapshot.drawdown_fraction >= self.risk.maximum_portfolio_drawdown
        ):
            reasons.append(RiskReason.DRAWDOWN_LIMIT)
        if snapshot.trades_today >= self.risk.maximum_trades_per_day:
            reasons.append(RiskReason.MAXIMUM_TRADES_PER_DAY)
        return reasons

    def _correlation_blocked(
        self,
        market: str,
        snapshot: PortfolioSnapshot,
    ) -> bool:
        if snapshot.returns_by_market:
            combined = pd.DataFrame(snapshot.returns_by_market).dropna(how="all")
            if isinstance(combined.index, pd.DatetimeIndex):
                from risk.correlation_analyzer import CorrelationAnalyzer

                analyzer = CorrelationAnalyzer(
                    minimum_samples=20,
                    cluster_threshold=self.maximum_correlation,
                )
                weights = {
                    position.market: (
                        position.notional_eur / snapshot.equity_eur
                        if snapshot.equity_eur > 0
                        else 1.0
                    )
                    for position in snapshot.positions
                }
                decision = analyzer.assess_proposal(
                    market=market,
                    proposed_weight=self.risk.maximum_position_fraction,
                    existing_weights=weights,
                    returns=combined,
                    correlated_risk_cap=(
                        self.risk.maximum_portfolio_exposure * 0.5
                    ),
                    large_position_threshold=min(
                        0.10, self.risk.maximum_position_fraction
                    ),
                )
                return not decision.approved
        requested = snapshot.returns_by_market.get(market)
        if requested is None or requested.dropna().empty:
            return False
        correlated_exposure = 0.0
        for position in snapshot.positions:
            other = snapshot.returns_by_market.get(position.market)
            if other is None:
                continue
            aligned = pd.concat([requested, other], axis=1).dropna()
            if len(aligned) < 20:
                continue
            correlation = float(aligned.corr().iloc[0, 1])
            if np.isfinite(correlation) and correlation >= self.maximum_correlation:
                correlated_exposure += position.notional_eur
        return correlated_exposure >= (
            snapshot.equity_eur * self.risk.maximum_portfolio_exposure * 0.5
        )

    def assess_entry(
        self,
        *,
        market: str,
        entry_price: float,
        stop_price: float,
        snapshot: PortfolioSnapshot,
        size_multiplier: float = 1.0,
        reliability_evidence: dict[str, float | int | bool | None] | None = None,
        volatility_multiplier: float = 1.0,
        correlation_multiplier: float = 1.0,
        liquidity_multiplier: float = 1.0,
        drawdown_multiplier: float = 1.0,
        live_mode: bool = False,
    ) -> RiskDecision:
        normalized = normalize_market(market)
        reasons = self._static_reasons(market=normalized, snapshot=snapshot)
        if (
            entry_price <= 0
            or stop_price <= 0
            or stop_price >= entry_price
            or snapshot.equity_eur <= 0
            or snapshot.cash_eur < 0
        ):
            reasons.append(RiskReason.INVALID_PRICE_OR_STOP)
        stop_fraction = (
            (entry_price - stop_price) / entry_price
            if entry_price > 0 and stop_price > 0
            else 0.0
        )
        if stop_fraction < self.risk.minimum_stop_distance:
            reasons.append(RiskReason.MINIMUM_STOP_DISTANCE)
        if self._correlation_blocked(normalized, snapshot):
            reasons.append(RiskReason.CORRELATION_LIMIT)
        if reasons:
            return RiskDecision(
                approved=False,
                reason_codes=tuple(reason.value for reason in dict.fromkeys(reasons)),
            )

        mode_cap = (
            self.risk.maximum_live_risk_per_trade
            if live_mode
            else self.risk.maximum_research_risk_per_trade
        )
        requested_risk = min(self.risk.risk_per_trade, mode_cap)
        reliability = (
            reliability_multiplier(reliability_evidence)
            if reliability_evidence is not None
            else 1.0
        )
        multiplier = float(
            np.prod(
                [
                    min(1.0, max(0.0, size_multiplier)),
                    reliability,
                    min(1.0, max(0.0, volatility_multiplier)),
                    min(1.0, max(0.0, correlation_multiplier)),
                    min(1.0, max(0.0, liquidity_multiplier)),
                    min(1.0, max(0.0, drawdown_multiplier)),
                ]
            )
        )
        size = calculate_position_size(
            snapshot.equity_eur,
            requested_risk * multiplier,
            entry_price,
            stop_price,
            fee_fraction_per_side=self.fee_fraction_per_side,
            slippage_fraction_per_side=self.slippage_fraction_per_side,
            max_position_fraction=self.risk.maximum_position_fraction,
        )
        units = size.units
        notional = units * entry_price
        risk_eur = size.actual_risk
        available_cash = max(
            0.0,
            snapshot.cash_eur - snapshot.equity_eur * self.risk.reserve_cash_fraction,
        )
        if notional > available_cash:
            units *= available_cash / max(notional, 1e-12)
            notional = units * entry_price
            risk_eur *= available_cash / max(size.position_notional, 1e-12)
        remaining_exposure = (
            snapshot.equity_eur * self.risk.maximum_portfolio_exposure
            - snapshot.exposure_eur
        )
        if notional > remaining_exposure:
            scale = max(0.0, remaining_exposure) / max(notional, 1e-12)
            units *= scale
            notional *= scale
            risk_eur *= scale
        remaining_risk = (
            snapshot.equity_eur * self.risk.maximum_total_open_risk
            - snapshot.open_risk_eur
        )
        if risk_eur > remaining_risk:
            scale = max(0.0, remaining_risk) / max(risk_eur, 1e-12)
            units *= scale
            notional *= scale
            risk_eur *= scale

        post_reasons: list[RiskReason] = []
        if available_cash <= 0:
            post_reasons.append(RiskReason.CASH_RESERVE)
        if remaining_exposure <= 0:
            post_reasons.append(RiskReason.MAXIMUM_PORTFOLIO_EXPOSURE)
        if remaining_risk <= 0:
            post_reasons.append(RiskReason.MAXIMUM_OPEN_RISK)
        if notional > snapshot.equity_eur * self.risk.maximum_position_fraction + 1e-8:
            post_reasons.append(RiskReason.MAXIMUM_POSITION_FRACTION)
        if units <= 0 or not np.isfinite(units):
            post_reasons.append(RiskReason.ZERO_SIZE)
        if post_reasons:
            return RiskDecision(
                approved=False,
                reason_codes=tuple(reason.value for reason in post_reasons),
            )
        return RiskDecision(
            approved=True,
            reason_codes=(RiskReason.APPROVED.value,),
            approved_quantity=Decimal(str(units)),
            risk_eur=Decimal(str(risk_eur)),
        )

    def assess_exit(
        self,
        *,
        market: str,
        requested_quantity: float,
        snapshot: PortfolioSnapshot,
    ) -> RiskDecision:
        normalized = normalize_market(market)
        owned = sum(
            position.quantity
            for position in snapshot.positions
            if position.market == normalized
        )
        if requested_quantity <= 0 or requested_quantity > owned + 1e-12:
            return RiskDecision(
                approved=False,
                reason_codes=(RiskReason.INSUFFICIENT_OWNED_UNITS.value,),
            )
        return RiskDecision(
            approved=True,
            reason_codes=(RiskReason.APPROVED.value,),
            approved_quantity=Decimal(str(requested_quantity)),
        )

    def status(self, snapshot: PortfolioSnapshot) -> dict[str, Any]:
        return {
            "healthy": (
                not self.kill_switch.active
                and snapshot.risk_manager_healthy
                and snapshot.reconciled
                and snapshot.data_healthy
                and snapshot.intelligence_timing_healthy
            ),
            "kill_switch_active": self.kill_switch.active,
            "kill_switch_reason": self.kill_switch.reason,
            "daily_loss_fraction": snapshot.daily_loss_fraction,
            "drawdown_fraction": snapshot.drawdown_fraction,
            "exposure_fraction": (
                snapshot.exposure_eur / snapshot.equity_eur
                if snapshot.equity_eur > 0
                else 1.0
            ),
            "open_risk_fraction": (
                snapshot.open_risk_eur / snapshot.equity_eur
                if snapshot.equity_eur > 0
                else 1.0
            ),
            "checked_at": datetime.now(UTC).isoformat(),
        }


__all__ = [
    "KillSwitch",
    "OperationalDegradation",
    "PortfolioSnapshot",
    "PositionExposure",
    "RiskManager",
    "RiskReason",
    "reliability_multiplier",
]
