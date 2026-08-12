from __future__ import annotations

from config.settings import PathSettings, Settings
from reporting.active_swing_current_evidence import (
    build_current_active_swing_evidence,
)
from utils.common import atomic_write_json


def _settings(tmp_path) -> Settings:
    return Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def test_current_candidate_is_complete_but_fail_closed(tmp_path) -> None:
    settings = _settings(tmp_path)
    output = settings.paths.output_dir
    candidate = {
        "opportunity_id": "candidate-1",
        "market": "LINK-EUR",
        "strategy": "TACTICAL_2H_RANGE_BREAKOUT",
        "family": "RANGE_BREAKOUT",
        "strategy_dna_hash": "a" * 64,
        "entry_timeframe": "2h",
        "confirmation_timeframe": "4h",
        "regime_timeframe": "1d",
        "signal_timestamp": "2026-08-12T00:00:00+00:00",
        "current_price": 10,
        "stop": 9,
        "target_1": 12,
        "target_2": 14,
        "estimated_fee_fraction": 0.0025,
        "estimated_slippage_bps": 8,
        "live_authority_granted": False,
        "trigger_reason": "CLOSE_ABOVE_RANGE",
        "timeframe_score_details": {
            timeframe: {"score": 0.5}
            for timeframe in ("15m", "1h", "2h", "4h", "1d", "1W")
        },
    }
    atomic_write_json(
        output / "active_trading" / "status.json",
        {
            "generated_at": "2026-08-12T01:00:00+00:00",
            "market_count": 1,
            "top_5_actionable": [candidate],
            "top_5_near_entry": [],
            "top_5_rotation": [],
            "macro": {"status": "FRESH", "regime": "RISK_OFF", "features": {}},
            "data_health": {
                "timeframes": {
                    timeframe: {"counts": {"FRESH": 1}}
                    for timeframe in ("15m", "1h", "2h", "4h", "1d", "1W")
                }
            },
            "execution_funnel": {
                "stage_counts": {"strategy_setups": 1, "entry_ready": 1}
            },
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    atomic_write_json(
        output / "active_trading" / "opportunities.json",
        {"top_5_relative_strength": [], "top_5_early_moves": []},
    )
    atomic_write_json(
        output / "active_trading" / "market_mechanics.json",
        {
            "markets": {
                "LINK-EUR": {
                    "orderflow_15m": {
                        "status": "READY",
                        "spread_bps": 2,
                        "available_at": "2026-08-12T00:59:00+00:00",
                    }
                }
            }
        },
    )
    atomic_write_json(
        output / "operations" / "live_account_health.json",
        {
            "status": "BLOCKED",
            "entry_allowed": False,
            "failures": ["EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"],
            "reconciliation": {"healthy": True},
            "market_fee_rates": {"LINK-EUR": {"taker_rate": "0.0025"}},
            "venue_safe_minimums_eur": {"LINK-EUR": "5.75"},
            "account": {"eur_available": "100", "portfolio_valuation": {}},
        },
    )
    atomic_write_json(
        output / "operations" / "external_inventory_remediation.json",
        {"status": "OPERATOR_DECISION_REQUIRED"},
    )
    atomic_write_json(
        output / "universe" / "tiered_trading_universe.json",
        {
            "live_executable_markets": ["LINK-EUR"],
            "rows": [{"market": "LINK-EUR", "shariah_status": "ALLOWED"}],
        },
    )
    atomic_write_json(
        output / "ml" / "canonical_training_status.json",
        {"authority": "SHADOW_ONLY"},
    )
    atomic_write_json(
        output / "operations" / "execution_evidence_layers.json",
        {
            "simulated_execution_pnl": {
                "closed_round_trips": 40,
                "net_expectancy_eur": 0.1,
                "by_playbook": {
                    "RANGE_BREAKOUT": {
                        "closed_round_trips": 40,
                        "closed_position_profit_factor": 1.3,
                        "paper_net_expectancy_eur": 0.1,
                    }
                },
            }
        },
    )

    result = build_current_active_swing_evidence(
        settings,
        runtime={"control_state": "PAUSED"},
        product_economically_good=False,
    )

    row = result["end_to_end"]["candidates"][0]
    assert row["final_decision"] == "NO_TRADE"
    assert row["requested_eur_allocation"] == "0"
    assert row["economics"]["positive_target_after_costs"] is True
    assert row["economics"]["expected_net_r"] is None
    assert row["ml"]["authority"] == "SHADOW_ONLY"
    assert "PRODUCT_ECONOMICS_NOT_POSITIVE" in row["blockers"]
    assert result["funnel"]["eligible_markets_scanned"] == 1
    assert result["economic_recovery_gate"]["eligible_family_count"] == 1
    assert (
        result["economic_recovery_gate"]["automatic_live_promotion_permitted"]
        is False
    )
    assert result["orders_generated"] == result["orders_submitted"] == 0


def test_managed_position_uses_real_position_and_never_fabricates_rl(tmp_path) -> None:
    settings = _settings(tmp_path)
    output = settings.paths.output_dir
    atomic_write_json(
        output / "operations" / "live_account_health.json",
        {
            "account": {
                "portfolio_valuation": {
                    "holdings": [{"market": "LINK-EUR", "price_eur": "11"}]
                }
            }
        },
    )
    atomic_write_json(
        output / "live" / "generated_strategy_live_state.json",
        {
            "positions": {
                "dna": {
                    "market": "LINK-EUR",
                    "status": "OPEN",
                    "strategy_id": "SIMPLE",
                    "strategy_dna_hash": "b" * 64,
                    "timeframe": "4h",
                    "quantity": "2",
                    "entry_price": "10",
                    "stop_loss": "9",
                    "take_profit_1": "11",
                    "take_profit_2": "13",
                    "tp1_reached": True,
                    "native_protective_stop_active": True,
                    "protective_stop_status": "awaitingTrigger",
                    "protective_stop_trigger": "9",
                }
            }
        },
    )

    result = build_current_active_swing_evidence(
        settings,
        runtime={"control_state": "PAUSED"},
        product_economically_good=False,
    )

    position = result["position_management"]["positions"][0]
    assert position["unrealized_r"] == "1"
    assert position["proposed_action"] == "KEEP"
    assert position["rl_advisory"] is None
    assert position["execution_authority_changed"] is False
