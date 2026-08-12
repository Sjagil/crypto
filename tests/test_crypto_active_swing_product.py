from __future__ import annotations

from config.settings import PathSettings, Settings
from reporting.crypto_active_swing_product import build_crypto_active_swing_product
from utils.common import atomic_write_json


def _settings(tmp_path) -> Settings:
    return Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def test_product_status_is_fail_closed_when_economics_and_inventory_fail(tmp_path) -> None:
    settings = _settings(tmp_path)
    output = settings.paths.output_dir
    atomic_write_json(
        output / "operations" / "live_account_health.json",
        {"status": "BLOCKED", "entry_allowed": False, "account": {}},
    )
    atomic_write_json(
        output / "operations" / "external_inventory_remediation.json",
        {"status": "OPERATOR_DECISION_REQUIRED"},
    )
    atomic_write_json(
        output / "operations" / "execution_evidence_layers.json",
        {
            "actual_live_pnl": {"closed_round_trips": 1},
            "simulated_execution_pnl": {
                "status": "VERIFIED_CLOSED_PAPER_ROUND_TRIPS",
                "closed_round_trips": 20,
                "net_expectancy_eur": -0.10,
            },
            "theoretical_signal_pnl": {},
        },
    )
    atomic_write_json(
        output / "ml" / "canonical_training_status.json",
        {
            "status": "DATA_PENDING",
            "authority": "SHADOW_ONLY",
            "canonical_row_count": 0,
            "canonical_feature_ready_incomplete_count": 10,
        },
    )
    atomic_write_json(
        output / "live" / "opportunity_lifecycle_state.json",
        {
            "updated_at": "2026-08-12T00:00:00+00:00",
            "opportunities": {
                "one": {"opportunity_id": "one", "state": "ARMED", "score": 80}
            },
        },
    )

    result = build_crypto_active_swing_product(
        settings,
        runtime={"process_running": True, "control_state": "PAUSED"},
        deployment_decision="HALTED",
        deployment_blockers=["ACCOUNT_STATUS_BLOCKED"],
        entry_constraints=["ACCOUNT_ENTRY_NOT_ALLOWED"],
    )

    assert result["product_status"] == "NOT_ECONOMICALLY_GOOD_YET"
    assert result["live_ready"] is False
    assert result["opportunity_funnel"]["lifecycle_counts"][
        "ENTRY_TRIGGER_PENDING"
    ] == 1
    assert result["requirements"]["coverage_complete"] is True
    assert result["canonical_money_path"]["second_money_path_authorized"] is False
    assert result["prospective_net_r_calibration"]["orders_submitted"] == 0
    assert result["orders_generated"] == result["orders_submitted"] == 0


def test_position_management_test_protects_only_managed_quantity(tmp_path) -> None:
    settings = _settings(tmp_path)
    output = settings.paths.output_dir
    atomic_write_json(
        output / "operations" / "live_account_health.json",
        {
            "status": "BLOCKED",
            "entry_allowed": False,
            "account": {
                "portfolio_valuation": {
                    "holdings": [{"market": "LINK-EUR", "price_eur": "7.5"}]
                }
            },
        },
    )
    atomic_write_json(
        output / "live" / "generated_strategy_live_state.json",
        {
            "positions": {
                "dna": {
                    "market": "LINK-EUR",
                    "status": "OPEN",
                    "quantity": "1.4",
                    "entry_price": "7.0",
                    "tp1_reached": True,
                    "native_protective_stop_active": True,
                    "protective_stop_status": "awaitingTrigger",
                    "protective_stop_trigger": "6.8",
                }
            }
        },
    )

    result = build_crypto_active_swing_product(
        settings,
        runtime={"process_running": True, "control_state": "PAUSED"},
        deployment_decision="HALTED",
        deployment_blockers=[],
        entry_constraints=[],
    )

    position = result["current_position_management_test"]["positions"][0]
    assert position["lifecycle"] == "PROFIT_PROTECTION"
    assert position["deterministic_proposed_action"] == (
        "HOLD_WITH_NATIVE_PROTECTIVE_STOP"
    )
    assert position["execution_authority_changed"] is False
    assert result["current_position_management_test"]["orders_submitted"] == 0
