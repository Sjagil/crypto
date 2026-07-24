"""Shariah-aware EUR dust reporting and explicitly gated consolidation planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from config.settings import ShariahSettings
from core.contracts import EligibilityStatus
from utils.common import stable_hash, utc_now


@dataclass(frozen=True)
class DustItem:
    asset: str
    quantity: Decimal
    estimated_eur_value: Decimal
    market: str
    eligibility: EligibilityStatus
    status: str
    reason_code: str


@dataclass(frozen=True)
class DustPlan:
    plan_id: str
    asset: str
    market: str
    quantity: Decimal
    estimated_eur_value: Decimal
    mode: Literal["paper", "live"]
    action: Literal["SELL_TO_EUR"]
    status: str
    reason_code: str


class DustSweeper:
    def __init__(
        self,
        *,
        shariah: ShariahSettings,
        dust_threshold_eur: Decimal = Decimal("10"),
        live_conversion_enabled: bool = False,
    ) -> None:
        if dust_threshold_eur <= 0:
            raise ValueError("dust threshold must be positive")
        self.shariah = shariah
        self.dust_threshold_eur = dust_threshold_eur
        self.live_conversion_enabled = live_conversion_enabled
        self._plan_ids: set[str] = set()
        self.audit_events: list[dict[str, Any]] = []

    def identify(
        self,
        balances: Mapping[str, Decimal | float | str],
        eur_prices: Mapping[str, Decimal | float | str],
    ) -> list[DustItem]:
        items: list[DustItem] = []
        for asset, raw_quantity in balances.items():
            selected_asset = asset.upper()
            if selected_asset == "EUR":
                continue
            quantity = Decimal(str(raw_quantity))
            if quantity <= 0:
                continue
            market = f"{selected_asset}-EUR"
            price = Decimal(str(eur_prices.get(selected_asset, "0")))
            value = quantity * price
            if value >= self.dust_threshold_eur:
                continue
            eligibility = self.shariah.eligibility(market).status
            if eligibility is EligibilityStatus.BLOCKED:
                status, reason = "IGNORED", "BLOCKED_ASSET"
            elif eligibility is EligibilityStatus.REVIEW_REQUIRED:
                status, reason = "IGNORED", "REVIEW_REQUIRED_ASSET"
            elif price <= 0:
                status, reason = "REPORT_ONLY", "MISSING_EUR_PRICE"
            else:
                status, reason = "REPORT_ONLY", "ALLOWED_DUST"
            items.append(
                DustItem(
                    asset=selected_asset,
                    quantity=quantity,
                    estimated_eur_value=value,
                    market=market,
                    eligibility=eligibility,
                    status=status,
                    reason_code=reason,
                )
            )
        return items

    def plan_direct_eur_consolidation(
        self,
        item: DustItem,
        *,
        supported_markets: set[str],
        minimum_order_eur: Mapping[str, Decimal | float | str],
        mode: Literal["paper", "live"] = "paper",
        live_preflight: Callable[[], bool] | None = None,
        risk_approval: Callable[[DustItem], bool] | None = None,
    ) -> DustPlan | None:
        if item.eligibility is not EligibilityStatus.ALLOWED:
            return None
        if item.market not in supported_markets:
            self._audit(item, "NO_ACTION", "DIRECT_EUR_MARKET_UNAVAILABLE")
            return None
        minimum = Decimal(str(minimum_order_eur.get(item.market, "Infinity")))
        if item.estimated_eur_value < minimum:
            self._audit(item, "NO_ACTION", "BELOW_EXCHANGE_MINIMUM")
            return None
        if mode == "live":
            if not self.live_conversion_enabled:
                self._audit(item, "BLOCKED", "LIVE_DUST_CONVERSION_DISABLED")
                return None
            if live_preflight is None or not live_preflight():
                self._audit(item, "BLOCKED", "LIVE_PREFLIGHT_FAILED")
                return None
            if risk_approval is None or not risk_approval(item):
                self._audit(item, "BLOCKED", "RISK_REJECTED")
                return None
        plan_id = stable_hash(
            [
                item.asset,
                item.market,
                str(item.quantity),
                str(item.estimated_eur_value),
                mode,
            ],
            length=32,
        )
        if plan_id in self._plan_ids:
            self._audit(item, "DUPLICATE", "IDEMPOTENT_PLAN_EXISTS")
            return None
        self._plan_ids.add(plan_id)
        plan = DustPlan(
            plan_id=plan_id,
            asset=item.asset,
            market=item.market,
            quantity=item.quantity,
            estimated_eur_value=item.estimated_eur_value,
            mode=mode,
            action="SELL_TO_EUR",
            status="SIMULATED" if mode == "paper" else "PLANNED_NOT_SUBMITTED",
            reason_code="DIRECT_EUR_CONSOLIDATION",
        )
        self.audit_events.append(
            {
                "timestamp": utc_now().isoformat(),
                "event": "DUST_PLAN",
                "plan_id": plan_id,
                "asset": item.asset,
                "market": item.market,
                "mode": mode,
                "status": plan.status,
                "reason_code": plan.reason_code,
                "automatic_submission": False,
                "intermediary_asset": None,
            }
        )
        return plan

    def report(
        self,
        balances: Mapping[str, Decimal | float | str],
        eur_prices: Mapping[str, Decimal | float | str],
    ) -> dict[str, Any]:
        items = self.identify(balances, eur_prices)
        return {
            "status": "REPORT_ONLY",
            "generated_at": utc_now().isoformat(),
            "live_conversion_enabled": self.live_conversion_enabled,
            "items": [
                {
                    "asset": item.asset,
                    "quantity": str(item.quantity),
                    "estimated_eur_value": str(item.estimated_eur_value),
                    "market": item.market,
                    "eligibility": item.eligibility.value,
                    "status": item.status,
                    "reason_code": item.reason_code,
                }
                for item in items
            ],
            "prohibited_intermediaries": ["BNB", "USDT"],
            "withdrawals_permitted": False,
        }

    def _audit(self, item: DustItem, status: str, reason: str) -> None:
        self.audit_events.append(
            {
                "timestamp": utc_now().isoformat(),
                "event": "DUST_PLAN_EVALUATION",
                "asset": item.asset,
                "market": item.market,
                "status": status,
                "reason_code": reason,
            }
        )


__all__ = ["DustItem", "DustPlan", "DustSweeper"]
