from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PathSettings, get_settings
from core.autonomous_trading import (
    PRIMARY_STRATEGY_DNA,
    PRIMARY_STRATEGY_ID,
    LiveApprovalStatus,
    LiveStrategyApprovalRegistry,
    MarketRegime,
    MarketRegimeClassifier,
    Opportunity,
    OpportunityScanner,
    RegimeSnapshot,
    StrategyRegimeRouter,
    _load_primary_frames,
    decide_managed_position_action,
)


def _daily_frame(*, direction: float = 1.0) -> pd.DataFrame:
    index = pd.date_range(
        "2025-12-01",
        periods=240,
        freq="D",
        tz="UTC",
    )
    close = 100.0 + direction * np.arange(len(index)) * 0.25
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def test_primary_frame_loader_uses_timestamp_column_not_range_index(
    tmp_path: Path,
) -> None:
    settings = get_settings().model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    source = _daily_frame().reset_index(names="timestamp")
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"):
        source.to_parquet(
            settings.paths.processed_data_dir / f"{market}_1d.parquet",
            index=False,
        )

    frames, hashes, failures = _load_primary_frames(settings)

    assert not failures
    assert set(hashes) == {"BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"}
    assert all(frame.index[0].year == 2025 for frame in frames.values())
    assert all(isinstance(frame.index, pd.DatetimeIndex) for frame in frames.values())


def test_live_approval_registry_is_human_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.yaml"
    path.write_text(
        """
version: 1
default_policy: FAIL_CLOSED
strategies:
  RR_B60_H5_Z20:
    strategy_dna_hash: 4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0
    strategy_family: BTC_REGIME_BETA_RESIDUAL_MEAN_REVERSION
    timeframe: 1d
    approved_markets: [ETH-EUR]
    approved_for_live: false
    approval_reference: null
    approved_at: null
    maximum_order_eur: 5
    maximum_total_exposure_eur: 10
    maximum_open_positions: 1
    maximum_new_orders_per_day: 1
    autoscale: false
    spot_only: true
""",
        encoding="utf-8",
    )
    status, record, reason = LiveStrategyApprovalRegistry(path).assess(
        PRIMARY_STRATEGY_ID,
        PRIMARY_STRATEGY_DNA,
    )
    assert status is LiveApprovalStatus.NOT_APPROVED
    assert record is not None
    assert reason == "LIVE_APPROVAL_HUMAN_CONFIRMATION_REQUIRED"


def test_live_approval_registry_rejects_dna_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.yaml"
    path.write_text(
        """
version: 1
default_policy: FAIL_CLOSED
strategies:
  RR_B60_H5_Z20:
    strategy_dna_hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    strategy_family: TEST
    timeframe: 1d
    approved_markets: [ETH-EUR]
    approved_for_live: false
    approval_reference: null
    approved_at: null
    maximum_order_eur: 5
    maximum_total_exposure_eur: 10
    maximum_open_positions: 1
    maximum_new_orders_per_day: 1
    autoscale: false
    spot_only: true
""",
        encoding="utf-8",
    )
    status, _, reason = LiveStrategyApprovalRegistry(path).assess(
        PRIMARY_STRATEGY_ID,
        PRIMARY_STRATEGY_DNA,
    )
    assert status is LiveApprovalStatus.DNA_MISMATCH
    assert reason == "LIVE_APPROVAL_DNA_MISMATCH"


def test_regime_classifier_uses_closed_daily_history() -> None:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    result = MarketRegimeClassifier.classify(
        _daily_frame(direction=1.0),
        observed_at=now,
    )
    assert result.primary_regime is MarketRegime.TREND_UP
    assert MarketRegime.RISK_ON in result.active_regimes
    assert result.data_fresh


def test_regime_classifier_blocks_insufficient_history() -> None:
    result = MarketRegimeClassifier.classify(
        _daily_frame().iloc[:50],
        observed_at=datetime(2026, 1, 20, tzinfo=UTC),
    )
    assert result.primary_regime is MarketRegime.UNCLASSIFIED
    assert not result.data_fresh


def test_router_blocks_risk_off_even_with_a_structure_match() -> None:
    snapshot = RegimeSnapshot(
        observed_at="2026-07-27T00:00:00+00:00",
        data_through="2026-07-26T00:00:00+00:00",
        primary_regime=MarketRegime.MEAN_REVERSION,
        active_regimes=(
            MarketRegime.MEAN_REVERSION,
            MarketRegime.RISK_OFF,
        ),
        confidence=75.0,
        metrics={},
        reason_codes=(),
        data_fresh=True,
    )
    primary = StrategyRegimeRouter().route(snapshot)[0]
    assert primary.strategy_id == PRIMARY_STRATEGY_ID
    assert not primary.eligible
    assert "RISK_OFF_BLOCK" in primary.reason_codes


def test_opportunity_ranking_prefers_actionable_over_raw_score() -> None:
    common = {
        "strategy_dna_hash": PRIMARY_STRATEGY_DNA,
        "market": "ETH-EUR",
        "timeframe": "1d",
        "action": "BUY",
        "confidence": 70.0,
        "reward_risk": 1.5,
        "regime_fit": 0.7,
        "liquidity_score": 0.9,
        "robustness_score": 0.7,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit_1": 107.5,
        "take_profit_2": 112.5,
        "valid_until": "2026-07-28T00:00:00+00:00",
    }
    blocked = Opportunity(
        opportunity_id="blocked",
        strategy_id="BLOCKED",
        score=99.0,
        blockers=("STALE_DATA",),
        actionable=False,
        **common,
    )
    actionable = Opportunity(
        opportunity_id="actionable",
        strategy_id=PRIMARY_STRATEGY_ID,
        score=70.0,
        blockers=(),
        actionable=True,
        **common,
    )
    ranked = OpportunityScanner.rank([blocked, actionable])
    assert ranked[0].opportunity_id == "actionable"


def test_opportunity_score_is_bounded() -> None:
    score = OpportunityScanner.score(
        confidence=10_000,
        reward_risk=100,
        regime_fit=100,
        liquidity_score=100,
        robustness_score=100,
    )
    assert score == 100.0


def _managed_position() -> dict[str, object]:
    return {
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit_1": 110.0,
        "take_profit_2": 120.0,
        "tp1_reached": False,
    }


def test_managed_position_stops_out_full_owned_quantity() -> None:
    decision = decide_managed_position_action(
        _managed_position(),
        market_price=94.99,
        strategy_action="HOLD",
        owned_quantity=Decimal("0.05"),
    )
    assert decision.action == "SELL_FULL"
    assert decision.reason_code == "STOP_LOSS_REACHED"
    assert decision.quantity_fraction == 1.0


def test_managed_position_tp1_moves_stop_to_breakeven_once() -> None:
    position = _managed_position()
    decision = decide_managed_position_action(
        position,
        market_price=110.0,
        strategy_action="HOLD",
        owned_quantity=Decimal("0.05"),
    )
    assert decision.action == "UPDATE_ONLY"
    assert decision.updated_stop_loss == 100.0
    assert decision.tp1_reached is True

    position["tp1_reached"] = True
    position["stop_loss"] = decision.updated_stop_loss
    repeated = decide_managed_position_action(
        position,
        market_price=111.0,
        strategy_action="HOLD",
        owned_quantity=Decimal("0.05"),
    )
    assert repeated.action == "HOLD"


def test_managed_position_tp2_and_strategy_exit_sell_full() -> None:
    take_profit = decide_managed_position_action(
        _managed_position(),
        market_price=120.0,
        strategy_action="HOLD",
        owned_quantity=Decimal("0.05"),
    )
    assert take_profit.action == "SELL_FULL"
    assert take_profit.reason_code == "TP2_REACHED"

    strategy_exit = decide_managed_position_action(
        _managed_position(),
        market_price=105.0,
        strategy_action="EXIT",
        owned_quantity=Decimal("0.05"),
    )
    assert strategy_exit.action == "SELL_FULL"
    assert strategy_exit.reason_code == "STRATEGY_EXIT"


def test_managed_position_requires_positive_owned_balance() -> None:
    decision = decide_managed_position_action(
        _managed_position(),
        market_price=105.0,
        strategy_action="HOLD",
        owned_quantity=Decimal("0"),
    )
    assert decision.action == "BLOCK"
    assert decision.reason_code == "MANAGED_POSITION_BALANCE_MISSING"
