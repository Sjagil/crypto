from __future__ import annotations

from pathlib import Path

from config.settings import PathSettings, Settings
from core.execution_authority import build_execution_authority_matrix
from utils.common import atomic_write_json


def _settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _ready_runtime(settings: Settings) -> None:
    live = settings.paths.output_dir / "live"
    operations = settings.paths.output_dir / "operations"
    atomic_write_json(
        live / "autonomous_live_state.json", {"state": "ENABLED"}
    )
    atomic_write_json(
        live / "autonomous_live_authority.json", {"active": True}
    )
    atomic_write_json(
        live / "autonomous_live_status.json",
        {
            "websocket": {"state": "CONNECTED"},
            "private_account_websocket": {"ready_for_new_entries": True},
        },
    )
    atomic_write_json(
        operations / "orderflow_stream_health.json",
        {
            "status": "HEALTHY",
            "ledger_integrity_status": "PASSED",
            "reason_codes": [],
            "provider": {"state": "CONNECTED"},
        },
    )


def test_authority_matrix_separates_entry_exit_and_protection(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "entry_allowed": True,
            "entry_blockers": [],
            "failures": [],
            "risk_reduction_allowed": True,
        },
    )

    assert matrix["entry_authority"]["allowed"] is True
    assert matrix["exit_authority"]["allowed"] is True
    assert matrix["protection_authority"]["allowed"] is True
    assert matrix["strategy_health_profiles"]["CLOSED_CANDLE_SWING"][
        "entry_allowed"
    ] is True
    assert matrix["strategy_health_profiles"]["ORDERFLOW_TIMED_SWING"][
        "entry_allowed"
    ] is True


def test_kill_switch_blocks_entries_but_preserves_risk_reduction(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)
    atomic_write_json(
        settings.paths.checkpoints_dir / "kill_switch.json",
        {"active": True, "reason": "TEST_ENTRY_KILL"},
    )

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "entry_allowed": True,
            "entry_blockers": [],
            "failures": [],
            "risk_reduction_allowed": True,
        },
    )

    assert matrix["entry_authority"]["allowed"] is False
    assert "KILL_SWITCH_ACTIVE" in matrix["entry_authority"]["blockers"]
    assert matrix["exit_authority"]["allowed"] is True
    assert matrix["protection_authority"]["allowed"] is True
    assert matrix["emergency_authority"]["automatic_flatten_authorized"] is False


def test_orderflow_failure_is_scoped_to_orderflow_timed_strategies(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)
    atomic_write_json(
        settings.paths.output_dir
        / "operations"
        / "orderflow_stream_health.json",
        {
            "status": "DEGRADED",
            "ledger_integrity_status": "PASSED",
            "reason_codes": ["STALE_ORDERBOOK"],
            "provider": {"state": "CONNECTED"},
        },
    )

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "entry_allowed": True,
            "entry_blockers": [],
            "failures": [],
            "risk_reduction_allowed": True,
        },
    )

    assert matrix["strategy_health_profiles"]["CLOSED_CANDLE_SWING"][
        "entry_allowed"
    ] is True
    flow = matrix["strategy_health_profiles"]["ORDERFLOW_TIMED_SWING"]
    assert flow["entry_allowed"] is False
    assert "ORDERFLOW_STREAM_NOT_READY" in flow["blockers"]


def test_current_heartbeat_recovers_stale_startup_stream_projection(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)
    atomic_write_json(
        settings.paths.output_dir / "live" / "autonomous_live_status.json",
        {
            "websocket": {"state": "STARTING"},
            "private_account_websocket": {"ready_for_new_entries": False},
        },
    )
    atomic_write_json(
        settings.paths.output_dir / "live" / "heartbeat.json",
        {
            "websocket": {"state": "CONNECTED"},
            "private_account_websocket": {"ready_for_new_entries": True},
        },
    )

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "entry_allowed": True,
            "entry_blockers": [],
            "failures": [],
            "risk_reduction_allowed": True,
        },
    )

    assert matrix["entry_authority"]["allowed"] is True
    assert matrix["exit_authority"]["allowed"] is True


def test_external_inventory_blocks_entries_but_not_managed_stop_maintenance(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "entry_allowed": False,
            "entry_blockers": [],
            "failures": [
                "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"
            ],
            "risk_reduction_allowed": False,
            "managed_position_protection_eligible": True,
            "managed_position_protection_scope": (
                "CANONICAL_MANAGED_POSITIONS_ONLY"
            ),
            "scoped_protection_failures": [],
        },
    )

    assert matrix["entry_authority"]["allowed"] is False
    assert matrix["exit_authority"]["allowed"] is False
    protection = matrix["protection_authority"]
    assert protection["allowed"] is True
    assert protection["scope"] == "CANONICAL_MANAGED_POSITIONS_ONLY"
    assert protection["external_inventory_actions_allowed"] is False
    assert matrix["managed_position_exit_authority"]["allowed"] is True


def test_managed_stop_scope_still_fails_closed_without_private_channel(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _ready_runtime(settings)
    atomic_write_json(
        settings.paths.output_dir / "live" / "autonomous_live_status.json",
        {
            "websocket": {"state": "CONNECTED"},
            "private_account_websocket": {"ready_for_new_entries": False},
        },
    )

    matrix = build_execution_authority_matrix(
        settings,
        account_health={
            "failures": [
                "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"
            ],
            "risk_reduction_allowed": False,
            "managed_position_protection_eligible": True,
            "scoped_protection_failures": [],
        },
    )

    assert matrix["protection_authority"]["allowed"] is False
    assert "PRIVATE_EXECUTION_CHANNEL_NOT_READY" in matrix[
        "protection_authority"
    ]["blockers"]
