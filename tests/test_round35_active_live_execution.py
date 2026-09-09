from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from config.settings import PathSettings, Settings
from core.active_portfolio_live_bridge import (
    active_entry_budget_eur,
    active_exit_quantity,
)
from utils.common import atomic_write_json


def _settings(tmp_path: Path) -> Settings:
    base = Settings.load(create_directories=False)
    return base.model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def _write_plan(settings: Settings, decisions: list[dict]) -> None:
    path = settings.paths.output_dir / "portfolio" / "active_live_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        settings.paths.output_dir / "portfolio" / "active_live_orchestrator.json",
        {"enabled": True, "mode": "test"},
    )
    atomic_write_json(
        path,
        {
            "schema_version": "active_portfolio_plan_v1",
            "mode": "live-plan",
            "status": "LIVE_TARGETS_READY_FOR_RISK_APPROVAL",
            "generated_at": datetime.now(UTC).isoformat(),
            "decisions": decisions,
        },
    )


def test_fresh_live_plan_caps_actual_entry_budget(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_plan(
        settings,
        [
            {
                "market": "BTC-EUR",
                "action": "ENTRY",
                "delta_notional_eur": 12.5,
                "target_notional_eur": 12.5,
                "live_plan_eligible": True,
                "live_blockers": [],
            }
        ],
    )
    budget, reason, _ = active_entry_budget_eur(
        settings,
        "BTC-EUR",
        default_budget_eur=Decimal("25"),
    )
    assert budget == Decimal("12.5")
    assert reason == "ACTIVE_PORTFOLIO_ENTRY_BUDGET_APPLIED"


def test_reduce_plan_requests_only_excess_owned_quantity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_plan(
        settings,
        [
            {
                "market": "ETH-EUR",
                "action": "REDUCE",
                "target_notional_eur": 60,
                "delta_notional_eur": -40,
                "live_plan_eligible": True,
                "live_blockers": [],
            }
        ],
    )
    quantity, action, _ = active_exit_quantity(
        settings,
        "ETH-EUR",
        current_quantity=Decimal("1"),
        mark_price=Decimal("100"),
    )
    assert quantity == Decimal("0.4")
    assert action == "REDUCE"


def test_enabled_orchestrator_fails_closed_without_plan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    marker = settings.paths.output_dir / "portfolio" / "active_live_orchestrator.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, {"enabled": True})
    budget, reason, _ = active_entry_budget_eur(
        settings,
        "BTC-EUR",
        default_budget_eur=Decimal("25"),
    )
    assert budget == 0
    assert reason == "ACTIVE_LIVE_PLAN_MISSING"


def test_round35_live_source_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "core" / "event_driven_live.py").read_text(encoding="utf-8")
    assert "ACTIVE_PORTFOLIO_FULL_EXIT" in source
    assert "ACTIVE_PORTFOLIO_REDUCE" in source
    assert "LIVE_TAKE_PROFIT_1_FILLED" in source
    assert "REARM_NATIVE_STOP_AFTER_PARTIAL_EXIT" in source
    assert "OrderType.STOP_LOSS" in source
    assert 'reason = "TAKE_PROFIT_2"' in source
    assert "submit_level_2_buy_atomically" in source


def test_round34_cycle_bootstraps_repo_root_before_config_import() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_active_portfolio_cycle.py").read_text(
        encoding="utf-8"
    )
    assert "sys.path.insert(0, str(ROOT))" in source
    assert source.index("sys.path.insert(0, str(ROOT))") < source.index(
        "from config.settings import Settings"
    )


def test_tp1_partial_sell_requires_both_legs_to_remain_venue_safe() -> None:
    from decimal import ROUND_DOWN

    from core.event_driven_live import _tp1_partial_quantity

    class Rules:
        minimum_order_value_eur = Decimal("5")
        minimum_order_amount = Decimal("0.01")

        @staticmethod
        def amount(value: Decimal) -> Decimal:
            return value.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    rules = Rules()
    assert _tp1_partial_quantity(
        quantity=Decimal("0.20"),
        best_bid=Decimal("100"),
        breakeven_stop=Decimal("100"),
        rules=rules,
    ) == Decimal("0.10")
    assert _tp1_partial_quantity(
        quantity=Decimal("0.10"),
        best_bid=Decimal("100"),
        breakeven_stop=Decimal("100"),
        rules=rules,
    ) == Decimal("0")
