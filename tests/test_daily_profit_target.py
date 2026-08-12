from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from config.settings import PathSettings, Settings
from core.daily_profit_target import (
    record_external_capital_flow,
    update_daily_profit_target,
)


def _isolated_target_settings(
    settings: Settings,
    tmp_path,
) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_daily_profit_target_scales_with_equity(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    isolated_settings = _isolated_target_settings(
        isolated_settings,
        tmp_path,
    )
    observed = datetime(2026, 7, 30, 8, tzinfo=UTC)
    payload = update_daily_profit_target(
        isolated_settings,
        estimated_equity_eur=Decimal("50"),
        observed_at=observed,
    )
    assert Decimal(payload["scaled_daily_target_eur"]) == Decimal("0.250")
    assert payload["non_binding"] is True
    assert payload["force_trades"] is False
    assert payload["override_risk_limits"] is False
    assert payload["orders_generated"] == 0


def test_daily_profit_target_preserves_start_equity_and_tracks_progress(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    isolated_settings = _isolated_target_settings(
        isolated_settings,
        tmp_path,
    )
    observed = datetime(2026, 7, 31, 8, tzinfo=UTC)
    update_daily_profit_target(
        isolated_settings,
        estimated_equity_eur=Decimal("50000"),
        observed_at=observed,
    )
    payload = update_daily_profit_target(
        isolated_settings,
        estimated_equity_eur=Decimal("50250"),
        observed_at=observed + timedelta(hours=2),
    )
    assert payload["day_start_equity_eur"] == "50000"
    assert payload["mark_to_market_pnl_eur"] == "250"
    assert payload["progress_fraction"] == "1"
    assert payload["status"] == "TARGET_REACHED"


def test_daily_profit_target_cannot_be_made_binding(tmp_path) -> None:
    env = tmp_path / "unsafe.env"
    env.write_text("DAILY_PROFIT_TARGET_FORCE_TRADES=true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must remain non-binding"):
        Settings.load(env_file=env, create_directories=False)


def test_confirmed_withdrawal_is_not_counted_as_trading_loss(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    isolated_settings = _isolated_target_settings(isolated_settings, tmp_path)
    observed = datetime(2026, 8, 2, 8, tzinfo=UTC)
    update_daily_profit_target(
        isolated_settings,
        estimated_equity_eur="500",
        observed_at=observed,
    )
    record_external_capital_flow(
        isolated_settings,
        amount_eur="-100",
        reason="WITHDRAWAL",
        effective_at=observed + timedelta(hours=1),
    )

    payload = update_daily_profit_target(
        isolated_settings,
        estimated_equity_eur="405",
        observed_at=observed + timedelta(hours=2),
    )

    assert payload["mark_to_market_pnl_eur"] == "-95"
    assert payload["cash_flow_adjusted_pnl_eur"] == "5"
    assert payload["risk_adjusted_day_start_equity_eur"] == "400"
    assert payload["external_capital_flow_eur"] == "-100"
    assert payload["status"] == "TARGET_REACHED"


def test_external_capital_flow_is_append_only_and_orderless(
    isolated_settings: Settings,
    tmp_path,
) -> None:
    isolated_settings = _isolated_target_settings(isolated_settings, tmp_path)

    payload = record_external_capital_flow(
        isolated_settings,
        amount_eur="-50",
        reason="WITHDRAWAL",
        effective_at=datetime(2026, 7, 31, 15, 18, tzinfo=UTC),
        note="Operator confirmed wallet withdrawal",
    )

    assert payload["amount_eur"] == "-50"
    assert payload["operator_confirmed"] is True
    assert payload["status"] == "RECORDED"
    assert payload["orders_generated"] == 0
    assert payload["orders_submitted"] == 0

    duplicate = record_external_capital_flow(
        isolated_settings,
        amount_eur="-50",
        reason="WITHDRAWAL",
        effective_at=datetime(2026, 7, 31, 15, 18, tzinfo=UTC),
        note="Operator confirmed wallet withdrawal",
    )
    assert duplicate["status"] == "SKIPPED_DUPLICATE"
