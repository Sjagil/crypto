from __future__ import annotations

from config.settings import PathSettings, Settings
from reporting.active_swing_deployment import (
    _account_deployment_gates,
    _runtime_research_and_universe_truth,
    _strategy_truth,
)
from utils.common import atomic_write_json


def test_strategy_truth_merges_dedicated_and_portfolio_live_authority(
    tmp_path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    strategies = tmp_path / "output" / "strategies"
    governance = tmp_path / "output" / "governance"
    config = tmp_path / "config"
    atomic_write_json(
        strategies / "backtest_positive.json",
        [
            {
                "strategy_id": "RR",
                "strategy_dna": "a" * 64,
                "timeframe": "1d",
            },
            {
                "strategy_id": "PORTFOLIO",
                "strategy_dna": "b" * 64,
                "timeframe": "4h",
            },
        ],
    )
    atomic_write_json(
        strategies / "paper_active.json",
        [],
    )
    atomic_write_json(
        governance / "positive_strategy_live_authority.json",
        {
            "active": True,
            "approved_candidates": [
                {
                    "strategy_id": "PORTFOLIO",
                    "strategy_dna_hash": "b" * 64,
                    "timeframe": "4h",
                    "approved_markets": ["BTC-EUR"],
                }
            ],
        },
    )
    config.mkdir(parents=True, exist_ok=True)
    (config / "live_strategy_approvals.yaml").write_text(
        """
version: 1
strategies:
  RR:
    strategy_dna_hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    timeframe: 1d
    approved_markets: [ETH-EUR]
    approved_for_live: true
""".strip(),
        encoding="utf-8",
    )

    truth = _strategy_truth(settings)

    assert len(truth["live"]) == 2
    assert truth["live_by_timeframe"] == {"4h": 1, "1d": 1}
    assert truth["shadow"] == []


def test_runtime_truth_uses_continuous_research_and_market_exceptions(
    tmp_path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    output = settings.paths.output_dir
    atomic_write_json(
        output / "live" / "heartbeat.json",
        {
            "research": {
                "status": "RUNNING",
                "continuous_research": {
                    "status": "RUNNING",
                    "running": True,
                    "pid": 123,
                },
            }
        },
    )
    atomic_write_json(
        output / "autopilot" / "status.json",
        {
            "stages": {
                "universe": {
                    "execution_eligible": 5,
                }
            }
        },
    )
    live_markets = [
        "BTC-EUR",
        "ETH-EUR",
        "SOL-EUR",
        "LINK-EUR",
        "TAO-EUR",
        "NPC-EUR",
    ]
    atomic_write_json(
        output / "universe" / "tiered_trading_universe.json",
        {
            "live_executable_markets": live_markets,
            "selection_hash": "selection-1",
        },
    )

    research, continuous, universe = (
        _runtime_research_and_universe_truth(settings)
    )

    assert continuous["running"] is True
    assert universe["live_executable_markets"] == live_markets
    normalized = research["stages"]["universe"]
    assert normalized["execution_eligible"] == 6
    assert normalized["execution_eligible_markets"] == live_markets
    assert normalized["execution_eligibility_basis"] == (
        "TIERED_LIVE_UNIVERSE_WITH_EXPLICIT_EXCEPTIONS"
    )


def test_account_deployment_gates_block_unready_external_inventory() -> None:
    blockers, constraints = _account_deployment_gates(
        account={
            "status": "BLOCKED",
            "entry_allowed": False,
            "failures": ["EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION"],
        },
        external_inventory={"status": "OPERATOR_DECISION_REQUIRED"},
    )

    assert blockers == [
        "ACCOUNT_STATUS_BLOCKED",
        "EXTERNAL_INVENTORY_OPERATOR_DECISION_REQUIRED",
    ]
    assert constraints == [
        "EXISTING_NON_EUR_POSITION_REQUIRES_RECONCILIATION",
        "ACCOUNT_ENTRY_NOT_ALLOWED",
        "EXTERNAL_INVENTORY_NOT_CANONICAL",
    ]


def test_account_deployment_gates_allow_only_explicit_ready_account() -> None:
    assert _account_deployment_gates(
        account={"status": "READY", "entry_allowed": True},
        external_inventory={"status": "NO_EXTERNAL_INVENTORY"},
    ) == ([], [])
