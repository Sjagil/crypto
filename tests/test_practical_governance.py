from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from config.settings import SUPPORTED_TIMEFRAMES, PathSettings, Settings
from core.contracts import ResearchStatus
from core.practical_governance import (
    PRIMARY_STRATEGY_DNA,
    PRIMARY_STRATEGY_ID,
    _strategy_record,
    activate_live_canary_authority,
    approve_capital_level,
    build_portfolio_artifacts,
    capital_level,
    capital_scaling_status,
    capital_scaling_status_from_ledger,
    deactivate_live_canary_authority,
    live_canary_authority,
    live_capital_evidence,
    reclassify_existing_strategies,
)
from execution.execution import DurableLedger, LivePreflight
from utils.common import atomic_write_json


def candidate(**overrides):
    payload = {
        "strategy_name": PRIMARY_STRATEGY_ID,
        "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
        "strategy_family": "BTC_REGIME_BETA_RESIDUAL_MEAN_REVERSION",
        "family_cluster": "residual_mean_reversion",
        "assets_universe": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
        "timeframe": "1d",
        "normal_profit_factor": 1.979,
        "stressed_profit_factor": 1.787,
        "double_cost_profit_factor": 1.787,
        "net_total_return": 0.458,
        "net_cagr": 0.056,
        "maximum_drawdown": 0.043,
        "expectancy": 0.001,
        "sample_count": 65,
        "sample_unit": "DECISION_OR_REBALANCE_EVENTS",
        "costs_included": True,
        "lookahead_status": "PASSED",
        "repainting_status": "PASSED",
        "entry_logic": "causal residual entry",
        "exit_logic": "normalization or time exit",
        "integrity": {"synthetic_data_used": False},
        "bitvavo_spot_long_only_compatible": True,
        "scores": {"historical_performance": 70.0, "composite": 77.0},
        "statistical_evidence": {
            "test_passes": {
                "deflated_sharpe": False,
                "ordinary_pbo": False,
                "selection_pbo": False,
                "white_reality_check": False,
                "hansen_spa": False,
            }
        },
        "holdout_status": "NO_GLOBALLY_UNTOUCHED_HISTORICAL_HOLDOUT_REMAINS",
        "forward_observations": 0,
        "parameters": {"frozen": True},
        "evidence": [],
    }
    payload.update(overrides)
    return payload


def test_academic_failures_do_not_block_paper_or_canary() -> None:
    settings = Settings.load()
    record = _strategy_record(
        candidate(),
        approved_live_dna=set(),
        settings=settings,
    )
    assert record["research_positive"] is True
    assert record["paper_active"] is True
    assert record["live_canary_eligible"] is True
    assert record["hard_blockers"] == []
    assert {
        "DSR_FAILED",
        "PBO_FAILED",
        "WHITE_REALITY_CHECK_FAILED",
        "HANSEN_SPA_FAILED",
        "UNTOUCHED_HOLDOUT_MISSING",
    }.issubset(set(record["capital_scaling_warnings"]))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("lookahead_status", "FAILED", "LOOKAHEAD"),
        ("repainting_status", "FAILED", "REPAINTING"),
        ("normal_profit_factor", 1.0, "PROFIT_FACTOR_NOT_ABOVE_ONE"),
        ("net_total_return", 0.0, "NET_RETURN_NOT_POSITIVE"),
        ("entry_logic", "", "ENTRY_MISSING"),
        ("exit_logic", "", "EXIT_MISSING"),
    ],
)
def test_non_overridable_strategy_blockers(
    field: str,
    value,
    reason: str,
) -> None:
    record = _strategy_record(
        candidate(**{field: value}),
        approved_live_dna=set(),
        settings=Settings.load(),
    )
    assert record["research_positive"] is False
    assert reason in record["hard_blockers"]


def test_scoped_operator_authority_never_stores_phrase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load()
    atomic_write_json(
        tmp_path / "output" / "governance" / "reclassified_strategies.json",
        {
            "records": [
                {
                    "strategy_id": PRIMARY_STRATEGY_ID,
                    "strategy_dna": PRIMARY_STRATEGY_DNA,
                    "markets": ["ETH-EUR"],
                    "live_canary_eligible": True,
                }
            ]
        },
    )
    monkeypatch.setattr(
        "core.practical_governance._load_live_approved_dna",
        lambda _root: {PRIMARY_STRATEGY_DNA},
    )
    payload = activate_live_canary_authority(
        tmp_path,
        settings,
        strategy_id=PRIMARY_STRATEGY_ID,
        approval_phrase=settings.execution.required_manual_approval_phrase,
    )
    serialized = (tmp_path / "output" / "governance" / "live_canary_authority.json").read_text(
        encoding="utf-8"
    )
    assert payload["active"] is True
    assert payload["maximum_order_eur"] == 10.0
    assert payload["maximum_total_exposure_eur"] == 10.0
    assert settings.execution.required_manual_approval_phrase not in serialized
    assert live_canary_authority(tmp_path)[0] is True
    deactivate_live_canary_authority(tmp_path, reason="TEST")
    assert live_canary_authority(tmp_path)[0] is False


def test_operator_authority_only_overrides_legacy_global_toggles() -> None:
    settings = Settings.load()
    passed = LivePreflight.evaluate(
        settings,
        markets=("ETH-EUR",),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
    )
    assert passed.passed is True
    blocked = LivePreflight.evaluate(
        settings,
        markets=("ETH-EUR",),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=False,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
    )
    assert blocked.passed is False
    assert "LIVE_BLOCKED_EXCHANGE_UNHEALTHY" in blocked.failures


def test_capital_levels_require_operator_approval_and_never_autoscale() -> None:
    level = capital_level(
        flawless_round_trips=12,
        net_live_expectancy=0.01,
        operator_approved_level=1,
    )
    assert level["eligible_level"] == 3
    assert level["active_level"] == 1
    assert level["operator_approval_required_to_raise"] is True
    assert level["autoscale"] is False
    assert level["caps"]["max_order_eur"] == 10.0


def test_capital_level_approval_requires_evidence_and_exact_phrase(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="not evidence-eligible"):
        approve_capital_level(
            tmp_path,
            strategy_id=PRIMARY_STRATEGY_ID,
            requested_level=2,
            approval_phrase=(
                f"I APPROVE CAPITAL LEVEL 2 FOR {PRIMARY_STRATEGY_ID}"
            ),
            flawless_round_trips=2,
            net_live_expectancy=1.0,
        )
    approved = approve_capital_level(
        tmp_path,
        strategy_id=PRIMARY_STRATEGY_ID,
        requested_level=2,
        approval_phrase=f"I APPROVE CAPITAL LEVEL 2 FOR {PRIMARY_STRATEGY_ID}",
        flawless_round_trips=3,
        net_live_expectancy=1.0,
    )
    assert approved["active_level"] == 2
    assert approved["caps"]["max_order_eur"] == 25.0
    assert approved["autoscale"] is False
    serialized = (
        tmp_path / "output" / "governance" / "capital_level_authority.json"
    ).read_text(encoding="utf-8")
    assert "I APPROVE CAPITAL" not in serialized
    assert capital_scaling_status(
        tmp_path,
        strategy_id=PRIMARY_STRATEGY_ID,
        flawless_round_trips=0,
        net_live_expectancy=None,
    )["active_level"] == 1


def test_live_capital_evidence_requires_actual_known_fees(
    tmp_path: Path,
) -> None:
    ledger = DurableLedger(
        tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    )
    ledger.append(
        "FILL",
        {
            "order_id": "buy-1",
            "market": "ETH-EUR",
            "side": "BUY",
            "quantity": "1",
            "price": "100",
            "fee_eur": "1",
            "fee_known": True,
        },
    )
    ledger.append(
        "FILL",
        {
            "order_id": "sell-1",
            "market": "ETH-EUR",
            "side": "SELL",
            "quantity": "1",
            "price": "110",
            "fee_eur": "1",
            "fee_known": True,
        },
    )
    evidence = live_capital_evidence(tmp_path)
    assert evidence["flawless_round_trips"] == 1
    assert evidence["net_live_expectancy"] == pytest.approx(8.0)
    assert evidence["critical_incidents"] == []
    assert evidence["evidence_scope_strategy_id"] == PRIMARY_STRATEGY_ID
    assert evidence["out_of_scope_fill_count"] == 0
    status = capital_scaling_status_from_ledger(tmp_path)
    assert status["eligible_level"] == 1
    assert status["active_level"] == 1


def test_unknown_live_fill_fee_blocks_scaling_evidence(
    tmp_path: Path,
) -> None:
    ledger = DurableLedger(
        tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    )
    ledger.append(
        "FILL",
        {
            "order_id": "buy-unknown-fee",
            "market": "ETH-EUR",
            "side": "BUY",
            "quantity": "0.01",
            "price": "2000",
            "fee_eur": "0",
            "fee_known": False,
        },
    )
    evidence = live_capital_evidence(tmp_path)
    assert evidence["flawless_round_trips"] == 0
    assert "UNKNOWN_LIVE_FILL_FEE" in evidence["critical_incidents"]


def test_live_capital_evidence_ignores_other_strategy_fills(
    tmp_path: Path,
) -> None:
    ledger = DurableLedger(
        tmp_path / "output" / "checkpoints" / "live_execution.jsonl"
    )
    ledger.append(
        "FILL",
        {
            "order_id": "other-playbook-buy",
            "market": "BTC-EUR",
            "strategy_id": "LIQUIDITY_SWEEP_RECLAIM_V1",
            "side": "BUY",
            "quantity": "0.001",
            "price": "50000",
            "fee_eur": "0.1",
            "fee_known": True,
        },
    )
    ledger.append(
        "FILL",
        {
            "order_id": "other-exact-dna-buy",
            "market": "ETH-EUR",
            "strategy_id": "SIMPLE_EXACT_other",
            "side": "BUY",
            "quantity": "0.01",
            "price": "2000",
            "fee_eur": "0.05",
            "fee_known": True,
        },
    )

    evidence = live_capital_evidence(
        tmp_path,
        strategy_id=PRIMARY_STRATEGY_ID,
    )

    assert evidence["critical_incidents"] == []
    assert evidence["open_quantity"] == 0.0
    assert evidence["flawless_round_trips"] == 0
    assert evidence["out_of_scope_fill_count"] == 2


def test_capital_level_two_preflight_caps_are_enforced() -> None:
    settings = Settings.load()
    allowed = LivePreflight.evaluate(
        settings,
        markets=("ETH-EUR",),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
        cap_limits={
            "capital_level": 2,
            "max_order_eur": 25,
            "max_exposure_eur": 75,
            "max_positions": 2,
            "max_new_orders_per_day": 1,
        },
    )
    assert allowed.passed is True
    assert allowed.capability is not None
    assert allowed.capability.maximum_order_eur == Decimal("25")
    blocked = LivePreflight.evaluate(
        settings,
        markets=("ETH-EUR",),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
        cap_limits={
            "capital_level": 2,
            "max_order_eur": 26,
            "max_exposure_eur": 75,
            "max_positions": 2,
            "max_new_orders_per_day": 1,
        },
    )
    assert blocked.passed is False
    assert "LIVE_BLOCKED_INVALID_CAPITAL_AUTHORITY" in blocked.failures


def test_all_requested_timeframes_are_supported() -> None:
    required = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "3h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "2d",
        "3d",
        "1W",
        "1mo",
    }
    assert required.issubset(set(SUPPORTED_TIMEFRAMES))


def test_portfolio_artifacts_use_reconciled_positions_and_wallet_evidence(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    atomic_write_json(
        tmp_path / "output" / "paper" / "generated_strategy_state.json",
        {
            "reconciliation": {"healthy": True},
            "positions": {
                "paper-dna": {
                    "strategy_dna": "paper-dna",
                    "market": "BTC-EUR",
                    "entry_price": "50000",
                    "quantity": "0.001",
                }
            },
        },
    )
    atomic_write_json(
        tmp_path / "output" / "live" / "generated_strategy_live_state.json",
        {
            "positions": {
                "live-dna": {
                    "strategy_dna": "live-dna",
                    "market": "ETH-EUR",
                    "entry_price": "2000",
                    "quantity": "0.05",
                }
            }
        },
    )
    atomic_write_json(
        tmp_path / "output" / "operations" / "live_account_health.json",
        {
            "reconciliation": {"healthy": True},
            "account": {
                "eur_available": "80",
                "portfolio_valuation": {
                    "status": "COMPLETE_MARK_TO_MARKET",
                    "estimated_total_equity_eur": "565",
                    "holdings": [
                        {
                            "market": "ETH-EUR",
                            "estimated_value_eur": "100",
                        },
                        {
                            "market": "TAO-EUR",
                            "estimated_value_eur": "385",
                        },
                    ],
                },
            },
        },
    )
    governance = {
        "records": [
            {
                "paper_active": True,
                "strategy_id": "TEST",
                "strategy_dna": "paper-dna",
                "timeframe": "1h",
                "paper_risk_multiplier": 0.25,
                "composite_score": 50.0,
                "live_canary_active": False,
            }
        ]
    }

    result = build_portfolio_artifacts(tmp_path, settings, governance)
    current = result["current_allocation"]

    assert len(current["paper_positions"]) == 1
    assert current["paper_positions"][0]["source"] == (
        "GENERATED_PAPER_BROKER_RECONCILED"
    )
    assert len(current["live_positions"]) == 1
    assert current["total_live_exposure_eur"] == 100.0
    assert current["cash_eur"] == 80.0
    assert current["cash_fraction"] == pytest.approx(80.0 / 565.0)
    holdings = {
        row["market"]: row for row in current["wallet_holdings"]
    }
    assert holdings["ETH-EUR"]["managed_strategy_position"] is True
    assert holdings["TAO-EUR"]["managed_strategy_position"] is False


def test_simple_lab_exact_positive_is_visible_as_paper_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.practical_governance.collect_longlist",
        lambda _root: [],
    )
    monkeypatch.setattr(
        "core.practical_governance.score_candidates",
        lambda _rows: [],
    )
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    atomic_write_json(
        tmp_path
        / "output"
        / "strategies"
        / "simple_lab_backtest_positive.json",
        {
            "candidates": [
                {
                    "strategy_dna_hash": "simple-paper-dna",
                    "economic_hypothesis_family": "BREAKOUT",
                    "block_ids": ["donchian20_breakout"],
                    "markets": ["BTC-EUR"],
                    "timeframe": "4h",
                    "metrics": {
                        "net_return": 0.05,
                        "profit_factor": 1.2,
                        "net_expectancy_r": 0.1,
                        "trade_count": 12,
                    },
                    "integrity": {
                        "no_lookahead": True,
                        "no_repainting": True,
                    },
                    "frozen_candidate_hash": "frozen-simple-paper",
                    "parameters": {},
                }
            ]
        },
    )

    result = reclassify_existing_strategies(
        tmp_path,
        settings,
    )

    record = next(
        row
        for row in result["records"]
        if row["strategy_dna"] == "simple-paper-dna"
    )
    assert record["lifecycle_state"] == "PAPER_ACTIVE"
    assert record["paper_active"] is True
    assert record["live_canary_eligible"] is False
    assert record["operator_approval_required_for_live"] is True
    assert record["stressed_profit_factor"] is None


def test_tactical_and_gex_implementations_are_registered_without_live_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.practical_governance.collect_longlist",
        lambda _root: [],
    )
    monkeypatch.setattr(
        "core.practical_governance.score_candidates",
        lambda _rows: [],
    )
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    ).model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )

    reclassify_existing_strategies(tmp_path, settings)
    registry = __import__("json").loads(
        (
            tmp_path / "output" / "strategies" / "all_strategy_dna.json"
        ).read_text(encoding="utf-8")
    )
    pending = {
        row["strategy_id"]: row
        for row in registry["registered_pending"]
    }

    tactical = pending["TACTICAL_15M_DONCHIAN_BREAKOUT"]
    assert tactical["timeframe"] == "15m"
    assert tactical["lifecycle_state"] == "DATA_PENDING"
    assert tactical["paper_active"] is False
    assert tactical["live_canary_eligible"] is False

    gex = pending["GEX_FLOW_NEGATIVE_BREAKOUT_4H2H15M"]
    assert gex["timeframe"] == "15m"
    assert gex["lifecycle_state"] == "DATA_PENDING"
    assert gex["paper_active"] is False
    assert gex["live_canary_eligible"] is False
    assert gex["metadata"]["prospective_orderflow_only"] is True
