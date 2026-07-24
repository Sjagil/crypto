from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings, ShariahSettings
from core.contracts import EligibilityRecord, EligibilityStatus, Fill, OrderSide
from execution.dust_sweeper import DustSweeper
from execution.position_tracker import PositionTracker
from risk.correlation_analyzer import CorrelationAnalyzer
from risk.drawdown_protection import (
    DrawdownProtection,
    DrawdownState,
    DrawdownThresholds,
)


def fill(
    fill_id: str,
    side: OrderSide,
    quantity: str,
    price: str,
    fee: str = "0",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"order-{fill_id}",
        intent_id=f"intent-{fill_id}",
        market="BTC-EUR",
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        fee_eur=Decimal(fee),
        filled_at=datetime.now(UTC),
        venue="paper",
    )


def test_position_partial_fills_realized_unrealized_and_persistence(tmp_path) -> None:
    path = tmp_path / "positions.json"
    tracker = PositionTracker(path)
    tracker.ingest_fill(fill("buy-1", OrderSide.BUY, "1", "100"))
    tracker.ingest_fill(fill("buy-2", OrderSide.BUY, "1", "120"))
    position = tracker.mark_to_market("BTC-EUR", "130")
    assert position.owned_quantity == Decimal("2")
    assert position.average_entry_price == Decimal("110")
    assert position.unrealized_pnl == Decimal("40")
    position = tracker.ingest_fill(fill("sell-1", OrderSide.SELL, "0.5", "140", "1"))
    assert position.owned_quantity == Decimal("1.5")
    assert position.realized_pnl == Decimal("14")
    duplicate = tracker.ingest_fill(fill("sell-1", OrderSide.SELL, "0.5", "140", "1"))
    assert duplicate.owned_quantity == Decimal("1.5")
    restored = PositionTracker(path)
    assert restored.positions["BTC-EUR"].realized_pnl == Decimal("14")
    assert restored.portfolio_pnl()["unrealized_pnl"] == Decimal("30")


def test_negative_position_prevention_and_reconciliation() -> None:
    tracker = PositionTracker()
    tracker.ingest_fill(fill("buy", OrderSide.BUY, "1", "100"))
    with pytest.raises(ValueError):
        tracker.ingest_fill(fill("oversell", OrderSide.SELL, "2", "100"))
    discrepancy = tracker.reconcile_balances({"BTC": "0.9"})
    assert not discrepancy["healthy"]
    assert discrepancy["reason_code"] == "BALANCE_DISCREPANCY"


def correlated_returns(end: datetime) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=100, freq="h")
    randomizer = np.random.default_rng(42)
    btc = randomizer.normal(0, 0.01, len(index))
    return pd.DataFrame(
        {
            "BTC-EUR": btc,
            "ETH-EUR": btc * 0.95 + randomizer.normal(0, 0.001, len(index)),
            "SOL-EUR": randomizer.normal(0, 0.01, len(index)),
        },
        index=index,
    )


def test_correlation_concentration_and_cap_rejection() -> None:
    analyzer = CorrelationAnalyzer(maximum_age=timedelta(hours=2))
    returns = correlated_returns(datetime.now(UTC))
    statistics = analyzer.risk_statistics(
        returns, {"BTC-EUR": 0.5, "ETH-EUR": 0.5}
    )
    assert statistics["effective_position_count"] == pytest.approx(2)
    assert statistics["concentration_score"] > 0.9
    decision = analyzer.assess_proposal(
        market="ETH-EUR",
        proposed_weight=0.2,
        existing_weights={"BTC-EUR": 0.4},
        returns=returns,
        correlated_risk_cap=0.5,
    )
    assert not decision.approved
    assert decision.reason_codes == ("CORRELATED_RISK_CAP_EXCEEDED",)


def test_stale_correlation_large_position_fails_closed() -> None:
    analyzer = CorrelationAnalyzer(maximum_age=timedelta(hours=1))
    decision = analyzer.assess_proposal(
        market="ETH-EUR",
        proposed_weight=0.2,
        existing_weights={},
        returns=correlated_returns(datetime.now(UTC) - timedelta(days=1)),
        correlated_risk_cap=0.5,
    )
    assert not decision.approved
    assert decision.reason_codes == ("STALE_CORRELATION_FAIL_CLOSED",)


def test_drawdown_transitions_persistent_kill_switch_and_manual_reset(tmp_path) -> None:
    state = tmp_path / "drawdown.json"
    audit = tmp_path / "drawdown.jsonl"
    protection = DrawdownProtection(
        state_path=state,
        audit_path=audit,
        thresholds=DrawdownThresholds(
            warning=0.02,
            reduce_risk=0.04,
            block_new_entries=0.06,
            kill_switch=0.08,
        ),
        cooldown=timedelta(0),
    )
    index = pd.date_range(end=datetime.now(UTC), periods=5, freq="h")
    warning = protection.evaluate(
        portfolio_equity=pd.Series([100, 101, 102, 101, 99], index=index)
    )
    assert warning["state"] == DrawdownState.WARNING
    killed = protection.evaluate(
        portfolio_equity=pd.Series([100, 101, 102, 95, 90], index=index)
    )
    assert killed["state"] == DrawdownState.KILL_SWITCH
    restored = DrawdownProtection(state_path=state, audit_path=audit)
    assert restored.state is DrawdownState.KILL_SWITCH
    recovered = restored.evaluate(
        portfolio_equity=pd.Series([100, 101, 102, 103, 104], index=index)
    )
    assert recovered["state"] == DrawdownState.KILL_SWITCH
    with pytest.raises(ValueError):
        restored.manual_reset(reason="")
    restored.manual_reset(reason="operator reviewed exposure and reconciliation")
    assert restored.state is DrawdownState.NORMAL
    assert audit.read_text(encoding="utf-8").count("\n") >= 3


def shariah_with_blocked(settings: Settings) -> ShariahSettings:
    markets = dict(settings.shariah.markets)
    markets["DOGE-EUR"] = EligibilityRecord(
        market="DOGE-EUR",
        status=EligibilityStatus.BLOCKED,
        reason="UNIT_TEST_BLOCK",
    )
    return ShariahSettings(
        source_path=settings.shariah.source_path,
        version=settings.shariah.version,
        markets=markets,
    )


def test_dust_report_only_blocked_and_direct_eur_plan(
    isolated_settings: Settings,
) -> None:
    sweeper = DustSweeper(
        shariah=shariah_with_blocked(isolated_settings),
        dust_threshold_eur=Decimal("10"),
    )
    items = sweeper.identify(
        {"BTC": "0.0003", "DOGE": "100", "UNKNOWN": "1"},
        {"BTC": "20000", "DOGE": "0.05", "UNKNOWN": "1"},
    )
    by_asset = {item.asset: item for item in items}
    assert by_asset["DOGE"].status == "IGNORED"
    assert by_asset["UNKNOWN"].reason_code == "REVIEW_REQUIRED_ASSET"
    plan = sweeper.plan_direct_eur_consolidation(
        by_asset["BTC"],
        supported_markets={"BTC-EUR"},
        minimum_order_eur={"BTC-EUR": "5"},
        mode="paper",
    )
    assert plan is not None
    assert plan.status == "SIMULATED"
    assert plan.action == "SELL_TO_EUR"
    assert (
        sweeper.plan_direct_eur_consolidation(
            by_asset["BTC"],
            supported_markets={"BTC-EUR"},
            minimum_order_eur={"BTC-EUR": "5"},
            mode="paper",
        )
        is None
    )
    report = sweeper.report({"BTC": "0.0003"}, {"BTC": "20000"})
    assert report["status"] == "REPORT_ONLY"
    assert not report["withdrawals_permitted"]
