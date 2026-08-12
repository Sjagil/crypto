from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from config.settings import PathSettings, Settings
from core.cli import build_parser
from core.swing_trading import (
    SwingCooldownManager,
    WeeklyTradeBudgetManager,
    execution_timeframe_allowed,
    material_position_limit,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    )
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_active_swing_timeframe_policy_enables_15m_and_blocks_5m() -> None:
    assert execution_timeframe_allowed("15m") is True
    assert execution_timeframe_allowed("1h") is True
    assert execution_timeframe_allowed("2h") is True
    assert execution_timeframe_allowed("4h") is True
    assert execution_timeframe_allowed("1d") is True
    assert execution_timeframe_allowed("1W") is True
    assert execution_timeframe_allowed("5m") is False


def test_material_position_limit_uses_wallet_equity_tiers() -> None:
    assert material_position_limit("999.99") == 2
    assert material_position_limit("1000") == 3
    assert material_position_limit("9999.99") == 3
    assert material_position_limit("10000") == 5
    assert material_position_limit("250000") == 12


def test_weekly_budget_caps_unique_entries_and_keeps_exits_available(
    tmp_path: Path,
) -> None:
    manager = WeeklyTradeBudgetManager(_settings(tmp_path))
    observed = datetime(2026, 7, 30, 12, tzinfo=UTC)
    for index in range(20):
        result = manager.record_entry(
            entry_identity=f"entry-{index}",
            strategy_id="STRATEGY",
            strategy_dna_hash="a" * 64,
            market="BTC-EUR",
            timeframe="1h",
            regime="BULL",
            order_status="filled",
            observed_at=observed,
        )
        assert result["recorded"] is True
    duplicate = manager.record_entry(
        entry_identity="entry-0",
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        regime="BULL",
        order_status="partial",
        observed_at=observed,
    )
    status = manager.status(observed_at=observed)
    assert duplicate["recorded"] is False
    assert status["new_entries"] == 20
    assert status["remaining_entry_budget"] == 0
    assert status["new_entries_blocked"] is True
    assert status["protective_exits_always_allowed"] is True
    assert manager.assess_entry(score=100, observed_at=observed)[
        "approved"
    ] is False


def test_weekly_budget_resets_on_new_iso_week(tmp_path: Path) -> None:
    manager = WeeklyTradeBudgetManager(_settings(tmp_path))
    manager.record_entry(
        entry_identity="old-week",
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="4h",
        regime=None,
        order_status="filled",
        observed_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    status = manager.status(
        observed_at=datetime(2026, 8, 3, tzinfo=UTC)
    )
    assert status["new_entries"] == 0
    assert status["remaining_entry_budget"] == 20


def test_swing_cooldown_uses_closed_bars_and_blocks_same_setup(
    tmp_path: Path,
) -> None:
    manager = SwingCooldownManager(_settings(tmp_path))
    exited_at = datetime(2026, 7, 30, 8, tzinfo=UTC)
    manager.record_exit(
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        reason="STRATEGY_EXIT",
        observed_at=exited_at,
    )
    blocked = manager.assess_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        signal_candle_at=exited_at + timedelta(hours=5),
        observed_at=exited_at + timedelta(hours=5),
    )
    assert blocked["approved"] is False
    assert blocked["required_closed_bars"] == 6
    allowed_at = exited_at + timedelta(hours=6)
    assert manager.assess_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        signal_candle_at=allowed_at,
        observed_at=allowed_at,
    )["approved"] is True
    manager.record_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        signal_candle_at=allowed_at,
        observed_at=allowed_at,
    )
    duplicate = manager.assess_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="a" * 64,
        market="BTC-EUR",
        timeframe="1h",
        signal_candle_at=allowed_at,
        observed_at=allowed_at + timedelta(hours=1),
    )
    assert duplicate["approved"] is False
    assert duplicate["reason_code"] == "DUPLICATE_SETUP_SUPPRESSED"


def test_stop_loss_uses_longer_timeframe_cooldown(tmp_path: Path) -> None:
    manager = SwingCooldownManager(_settings(tmp_path))
    exited_at = datetime(2026, 7, 30, 8, tzinfo=UTC)
    manager.record_exit(
        strategy_id="STRATEGY",
        strategy_dna_hash="b" * 64,
        market="ETH-EUR",
        timeframe="2h",
        reason="STOP_LOSS_REACHED",
        observed_at=exited_at,
    )
    blocked = manager.assess_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="b" * 64,
        market="ETH-EUR",
        timeframe="2h",
        signal_candle_at=exited_at + timedelta(hours=14),
        observed_at=exited_at + timedelta(hours=14),
    )
    assert blocked["approved"] is False
    assert blocked["required_closed_bars"] == 8
    assert manager.assess_entry(
        strategy_id="STRATEGY",
        strategy_dna_hash="b" * 64,
        market="ETH-EUR",
        timeframe="2h",
        signal_candle_at=exited_at + timedelta(hours=16),
        observed_at=exited_at + timedelta(hours=16),
    )["approved"] is True


def test_live_active_swing_cli_routes_are_registered() -> None:
    parser = build_parser()
    for command in (
        "weekly-budget",
        "opportunities",
        "performance",
        "deployment-audit",
        "verify",
        "pause",
        "resume",
        "shutdown",
    ):
        args = parser.parse_args(["live", command])
        assert args.command == "live"
        assert args.live_command == command
