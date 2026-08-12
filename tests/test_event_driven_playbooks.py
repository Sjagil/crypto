from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.event_driven_playbooks import (
    PLAYBOOKS,
    OpportunityLifecycleLedger,
    OpportunityState,
    _higher_timeframe_parent,
    build_event_driven_opportunities,
    macro_long_lock,
    score_realtime_market,
)


def test_weighted_intraday_parent_can_offset_bearish_higher_timeframe() -> None:
    selected = context()
    selected.update(
        {
            "timeframe_alignment_score": 0.0,
            "weighted_timeframe_score": 0.61,
            "weighted_entry_threshold": 0.50,
            "trade_type": "COUNTERTREND_LONG",
        }
    )

    parent = _higher_timeframe_parent(selected)

    assert parent["valid"] is True
    assert parent["trade_type"] == "COUNTERTREND_LONG"
    assert parent["hard_blocked_by_1d_or_1w"] is False


def realtime_row(market: str = "ADA-EUR") -> dict:
    btc = market == "BTC-EUR"
    return {
        "market": market,
        "fresh": True,
        "sequence_valid": True,
        "trade_age_seconds": 1.0,
        "book_age_seconds": 0.5,
        "price": 0.18 if market == "ADA-EUR" else 55_000.0,
        "windows": {
            "1m": {
                "return": 0.004,
                "quote_volume_eur": 5_000.0,
                "trade_count": 80,
                "taker_buy_ratio": 0.68,
                "cvd_quote_eur": 1_500.0,
            },
            "5m": {"return": 0.002 if btc else 0.012},
            "15m": {"return": 0.004 if btc else 0.021},
        },
        "acceleration_1m": 0.006,
        "relative_volume_1m": 4.0,
        "relative_volume_5m": 2.2,
        "trade_intensity_1m": 4.0,
        "ofi_1m": 0.20,
        "microprice_edge_bps": 3.0,
        "mlobi_positive_persistence_10s": 0.9,
        "book_update_count_10s": 5,
        "book_update_count_1m": 20,
        "bid_replenishment_ratio_1m": 0.08,
        "ask_depletion_ratio_1m": 0.04,
        "bullish_absorption_score_1m": 0.2,
        "book": {
            "mlobi_top_10": 0.24,
            "distance_weighted_imbalance_top_10": 0.3,
            "depth_imbalance_within_10_bps": 0.3,
            "spread_bps": 7.0,
            "dynamic_spread_cap_bps": 15.0,
            "spread_within_dynamic_cap": True,
            "bid_depth_eur_top_10": 25_000.0,
            "ask_depth_eur_top_10": 15_000.0,
        },
        "estimated_buy_slippage_bps": 1.0,
    }


def context(market: str = "ADA-EUR") -> dict:
    return {
        "market": market,
        "family": "VOLUME_EXPANSION",
        "strategy": "TACTICAL_15M_VOLUME_EXPANSION",
        "timeframe": "15m",
        "entry_timeframe": "15m",
        "confirmation_timeframe": "1h",
        "regime_timeframe": "4h",
        "entry_trigger_confirmed": True,
        "closed_candle_only": True,
        "higher_timeframe_parent_valid": True,
        "timeframe_alignment_score": 1.0,
        "confidence": 90.0,
        "status": "ACTIONABLE",
        "entry_zone": (
            [0.179, 0.181]
            if market == "ADA-EUR"
            else [54_900.0, 55_100.0]
        ),
        "entry_atr": 0.005 if market == "ADA-EUR" else 1_000.0,
        "stop": 0.175 if market == "ADA-EUR" else 53_500.0,
        "target_1": 0.187 if market == "ADA-EUR" else 57_500.0,
        "target_2": 0.195 if market == "ADA-EUR" else 59_000.0,
    }


def test_macro_risk_off_reduces_size_but_is_not_a_score_veto() -> None:
    scoring = score_realtime_market(
        realtime_row(),
        context=context(),
        btc_row=realtime_row("BTC-EUR"),
        macro_regime="MACRO_RISK_OFF",
        strategy_evidence=0.8,
    )

    assert scoring["score"] >= 54
    assert scoring["tier"] in {"A", "B", "C", "WATCH"}
    assert scoring["macro_risk_multiplier"] == 0.65
    assert "MACRO_RISK_OFF" not in scoring["hard_blockers"]
    assert scoring["confirmation_count"] >= 3


def test_neutral_microstructure_is_not_a_standalone_entry_veto() -> None:
    row = realtime_row()
    row["ofi_1m"] = 0.0
    row["ofi_windows"] = {"30s": 0.0, "90s": 0.0, "300s": 0.0}
    row["windows"]["1m"]["taker_buy_ratio"] = 0.50
    row["windows"]["1m"]["cvd_quote_eur"] = 0.0
    row["book"]["mlobi_top_10"] = 0.0
    row["book"]["depth_imbalance_within_10_bps"] = 0.0
    row["microprice_edge_bps"] = 0.0
    row["bid_replenishment_ratio_1m"] = 0.0
    row["ask_depletion_ratio_1m"] = 0.0
    row["bullish_absorption_score_1m"] = 0.0

    scoring = score_realtime_market(row, context=context())

    assert scoring["microstructure_state"] == "NEUTRAL"
    assert "HOSTILE_MICROSTRUCTURE" not in scoring["hard_blockers"]
    assert "INSUFFICIENT_REALTIME_CONFIRMATIONS" not in scoring["hard_blockers"]


def test_independently_hostile_flow_and_book_remain_fail_closed() -> None:
    row = realtime_row()
    row["ofi_1m"] = -0.30
    row["ofi_windows"] = {"30s": -0.30, "90s": -0.30, "300s": -0.30}
    row["windows"]["1m"]["taker_buy_ratio"] = 0.30
    row["windows"]["1m"]["cvd_quote_eur"] = -500.0
    row["book"]["depth_imbalance_within_10_bps"] = -0.50
    row["microprice_edge_bps"] = -2.0

    scoring = score_realtime_market(row, context=context())

    assert scoring["microstructure_state"] == "HOSTILE"
    assert scoring["microstructure_hostile_groups"] == [
        "EXECUTED_FLOW",
        "ORDERBOOK",
    ]
    assert "HOSTILE_MICROSTRUCTURE" in scoring["hard_blockers"]


def test_event_builder_turns_confirmed_flow_into_entry_ready() -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        macro_regime="MACRO_RISK_OFF",
        evidence_by_family={"VOLUME_EXPANSION": 0.8},
    )

    ada = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert ada
    assert all(row["episode_id"] for row in ada)
    assert all(row["gate_matrix"]["ORDERFLOW"] for row in ada)
    assert all(row["next_required_condition"] for row in ada)
    assert any(row["state"] == "ENTRY_READY" for row in ada)
    assert any(row["family"] == "VOLATILITY_EXPANSION" for row in ada)
    assert all(row["stop_loss"] < row["entry_price"] for row in ada)
    assert all(row["take_profit_2"] > row["take_profit_1"] for row in ada)
    assert any(row["parameter_band_status"] == "VALIDATED" for row in ada)
    assert all(len(row["parameter_band_hash"]) == 64 for row in ada)
    assert all(row["time_stop_minutes"] == 1_440 for row in ada)
    assert all(
        row["active_swing_contract"]["execution_authority"] is False
        for row in ada
    )
    assert all(
        row["active_swing_contract"]["timeframe_contract"][
            "entry_timeframe"
        ]
        == "15m"
        for row in ada
    )
    assert all(
        row["execution_economics"]["net_target_2_bps"] >= 225
        for row in ada
    )
    assert all(
        row["execution_quality_score"] is None
        or 0.0 <= row["execution_quality_score"] <= 1.0
        for row in ada
    )


def test_swing_entry_is_blocked_when_dynamic_net_economics_fail() -> None:
    selected_context = {
        **context(),
        "stop": 0.178,
        "target_1": 0.181,
        "target_2": 0.182,
    }
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[selected_context],
        macro_regime="BROAD_RISK_ON",
    )

    selected = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert selected
    assert all(row["state"] != "ENTRY_READY" for row in selected)
    assert all(
        "NET_TARGET_2_BELOW_2_25_PERCENT" in row["hard_blockers"]
        for row in selected
    )


def test_two_point_nine_eight_percent_is_not_blocked_by_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.event_driven_playbooks as playbooks

    original = playbooks.estimate_roundtrip_economics

    def economics_with_rounding_tolerance(**kwargs: object) -> dict:
        result = original(**kwargs)
        return {
            **result,
            "net_target_2_bps": 298.0,
            "net_target_1_bps": 150.0,
            "net_rr_target_1": 1.1,
            "net_rr_target_2": 2.0,
            "expected_net_value_bps": 25.0,
            "cost_to_target_2_ratio": 0.10,
            "positive_after_costs": True,
        }

    monkeypatch.setattr(
        playbooks,
        "estimate_roundtrip_economics",
        economics_with_rounding_tolerance,
    )
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        macro_regime="BROAD_RISK_ON",
    )

    selected = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert selected
    assert all(
        "NET_TARGET_2_BELOW_2_25_PERCENT" not in row["hard_blockers"]
        for row in selected
    )
    assert all(row["normal_minimum_net_target_2_bps"] == 225.0 for row in selected)
    assert all(
        "NET_SWING_UPSIDE_BELOW_PREFERRED_3_PERCENT" in row["advisory_warnings"]
        for row in selected
    )


def test_bearish_1d_is_soft_modifier_not_standalone_long_blocker() -> None:
    macro = {
        "available_at": "2026-08-04T20:00:00+00:00",
        "features": {
            "btc_1d_trend_up": False,
            "btc_4h_trend_up": True,
        },
        "provider_refresh": {"coinmarketcap_global": "READY"},
    }
    lock = macro_long_lock("MACRO_RISK_OFF", macro)
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        macro_regime="MACRO_RISK_OFF",
        macro_context=macro,
    )

    assert lock["long_allowed"] is True
    assert lock["blockers"] == []
    assert "MACRO_1D_BEARISH_SIZE_REDUCTION" in lock["warnings"]
    assert lock["risk_multiplier_cap"] == 0.65
    assert opportunities
    assert all(
        "MACRO_1D_BEARISH_LONG_LOCK" not in row["hard_blockers"]
        for row in opportunities
    )
    assert all(row["macro_policy"] == "WEIGHTED_SOFT_REGIME" for row in opportunities)


def test_live_signal_context_uses_only_15m_or_1h_entry_generators() -> None:
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[
            {
                **context(),
                "timeframe": "4h",
                "entry_timeframe": "4h",
                "strategy": "FORBIDDEN_4H_ENTRY",
            },
            {**context(), "strategy": "ALLOWED_15M_ENTRY"},
        ],
        macro_regime="BROAD_RISK_ON",
    )

    assert opportunities
    assert all(
        row["context_strategy"] == "ALLOWED_15M_ENTRY"
        for row in opportunities
        if row["market"] == "ADA-EUR"
    )


def test_closed_1h_entry_with_4h_parent_and_1d_regime_is_valid() -> None:
    one_hour_context = {
        **context(),
        "strategy": "ALLOWED_1H_ENTRY",
        "timeframe": "1h",
        "entry_timeframe": "1h",
        "confirmation_timeframe": "4h",
        "regime_timeframe": "1d",
        "higher_timeframe_parent_valid": True,
    }
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[one_hour_context],
        macro_regime="BROAD_RISK_ON",
    )

    selected = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert selected
    assert all(row["context_strategy"] == "ALLOWED_1H_ENTRY" for row in selected)
    assert all(
        "MISSING_VALID_1H_4H_PARENT_SETUP" not in row["hard_blockers"]
        for row in selected
    )
    assert any(row["state"] == "ENTRY_READY" for row in selected)


def test_unconfirmed_1h_entry_cannot_be_made_valid_by_explicit_parent_flag() -> None:
    unconfirmed = {
        **context(),
        "timeframe": "1h",
        "entry_timeframe": "1h",
        "confirmation_timeframe": "4h",
        "regime_timeframe": "1d",
        "higher_timeframe_parent_valid": True,
        "entry_trigger_confirmed": False,
    }
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[unconfirmed],
        macro_regime="BROAD_RISK_ON",
    )

    selected = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert selected
    assert all(
        "MISSING_VALID_1H_4H_PARENT_SETUP" in row["hard_blockers"]
        for row in selected
    )


def test_realtime_trigger_can_complete_a_closed_candle_near_entry_setup() -> None:
    near_entry = {
        **context(),
        "entry_trigger_confirmed": False,
        "setup_valid_on_closed_candle": True,
        "status": "NEAR_ENTRY",
    }

    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-08T18:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[near_entry],
        macro_regime="RECOVERY",
    )

    selected = [row for row in opportunities if row["market"] == "ADA-EUR"]
    assert selected
    assert all(row["higher_timeframe_parent"]["valid"] for row in selected)
    assert all(row["closed_candle_setup_valid"] is True for row in selected)
    assert all(
        row["execution_trigger_source"]
        == "REALTIME_AFTER_CLOSED_CANDLE_SETUP"
        for row in selected
    )
    assert all(
        "MISSING_VALID_1H_4H_PARENT_SETUP" not in row["hard_blockers"]
        for row in selected
    )


def test_playbook_specific_score_floor_blocks_weak_orderflow_entry() -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    row = realtime_row()
    row["windows"]["1m"]["return"] = 0.0001
    row["windows"]["1m"]["taker_buy_ratio"] = 0.59
    row["windows"]["1m"]["cvd_quote_eur"] = 1.0
    row["windows"]["5m"]["return"] = 0.0001
    row["windows"]["15m"]["return"] = 0.0001
    row["relative_volume_1m"] = 1.0
    row["trade_intensity_1m"] = 1.0
    row["acceleration_1m"] = 0.0
    row["ofi_1m"] = 0.031
    row["book"]["mlobi_top_10"] = 0.0
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), row],
        },
        tactical_opportunities=[],
        macro_regime="MACRO_RISK_OFF",
    )

    orderflow = next(
        item
        for item in opportunities
        if item["family"] == "ORDERFLOW_CONTINUATION"
        and item["market"] == "ADA-EUR"
    )
    assert orderflow["score"] < 62
    assert orderflow["state"] != "ENTRY_READY"
    assert orderflow["parameter_band_status"] == "OUTSIDE_BAND"
    assert orderflow["context_timeframe"] is None
    assert orderflow["observation_timeframe"] == "1m"
    assert (
        orderflow["feature_snapshot"]["values"]["observation_timeframe"]
        == "1m"
    )
    assert (
        "MISSING_VALID_1H_4H_PARENT_SETUP"
        in orderflow["hard_blockers"]
    )
    assert (
        "PLAYBOOK_SCORE_OUTSIDE_VALIDATED_BAND"
        in orderflow["hard_blockers"]
    )


def test_failed_breakout_reclaim_has_distinct_long_spot_playbook() -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    selected_context = {
        **context(),
        "family": "FAILED_BREAKOUT_RECLAIM",
        "strategy": "TACTICAL_FAILED_BREAKOUT_RECLAIM",
    }
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[selected_context],
    )

    assert any(
        row["family"] == "FAILED_BREAKOUT_REVERSAL"
        for row in opportunities
        if row["market"] == "ADA-EUR"
    )


def test_stale_or_invalid_book_remains_fail_closed() -> None:
    row = realtime_row()
    row["fresh"] = False
    row["sequence_valid"] = False
    scoring = score_realtime_market(row, context=context())

    assert "STALE_REALTIME_DATA" in scoring["hard_blockers"]
    assert "ORDERBOOK_SEQUENCE_INVALID" in scoring["hard_blockers"]


def test_fresh_ticker_cannot_hide_stale_book_or_missing_trades() -> None:
    row = realtime_row()
    row["fresh"] = True
    row["book_age_seconds"] = 11.0
    row["trade_age_seconds"] = 91.0
    row["windows"]["1m"]["trade_count"] = 0
    row["windows"]["1m"]["quote_volume_eur"] = 0.0
    row["windows"]["1m"]["taker_buy_ratio"] = None
    scoring = score_realtime_market(row, context=context())

    assert "STALE_ORDERBOOK_DATA" in scoring["hard_blockers"]
    assert "NO_RECENT_EXECUTED_TRADES" in scoring["hard_blockers"]
    assert scoring["microstructure_quality"]["book_fresh"] is False
    assert scoring["microstructure_quality"]["executed_flow_fresh"] is False


def test_replenishment_is_a_book_group_confirmation_not_extra_flow() -> None:
    row = realtime_row()
    row["book"]["mlobi_top_10"] = -0.1
    row["microprice_edge_bps"] = -1.0
    row["ofi_1m"] = -0.1
    row["ofi_windows"] = {"30s": -0.1, "90s": -0.1, "300s": -0.1}
    row["bid_replenishment_ratio_1m"] = 0.10
    scoring = score_realtime_market(row, context=context())

    assert scoring["confirmations"]["replenishment"] is True
    assert scoring["confirmation_groups"]["orderbook_flow"] is True
    assert scoring["confirmation_groups"]["executed_flow"] is True
    assert scoring["execution_quality_components"]["replenishment"] > 0.5


def test_single_large_trade_cannot_create_volume_participation_score() -> None:
    row = realtime_row()
    row["windows"]["1m"]["trade_count"] = 1
    row["windows"]["1m"]["quote_volume_eur"] = 10_000.0
    row["relative_volume_1m"] = 500.0
    row["trade_intensity_1m"] = 100.0
    scoring = score_realtime_market(row, context=context())

    assert scoring["microstructure_quality"]["volume_sample_sufficient"] is False
    assert scoring["confirmation_groups"]["participation"] is False
    assert scoring["components"]["volume_acceleration"] < 8.0


def test_playbook_catalog_has_formal_dna_and_exit_plan() -> None:
    assert len(PLAYBOOKS) == 21
    assert len({item.dna for item in PLAYBOOKS}) == len(PLAYBOOKS)
    assert all(len(item.dna) == 64 for item in PLAYBOOKS)
    assert all(item.stop_method for item in PLAYBOOKS)
    assert all(item.take_profit_method for item in PLAYBOOKS)
    assert all(item.time_stop for item in PLAYBOOKS)


def test_bearish_daily_allows_only_explicit_micro_recovery_playbook() -> None:
    macro = {
        "available_at": "2026-08-04T20:00:00+00:00",
        "features": {
            "btc_1d_trend_up": False,
            "btc_4h_trend_up": True,
        },
        "provider_refresh": {"coinmarketcap_global": "READY"},
    }
    row = realtime_row()
    row["downside_sweep_reclaim_1m"] = True
    selected_context = {
        **context(),
        "family": "LIQUIDITY_SWEEP_RECOVERY",
        "strategy": "TACTICAL_15M_LIQUIDITY_SWEEP",
    }
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), row],
        },
        tactical_opportunities=[selected_context],
        macro_regime="MACRO_RISK_OFF",
        macro_context=macro,
    )

    recovery = next(
        item
        for item in opportunities
        if item["market"] == "ADA-EUR"
        and item["family"] == "BEAR_SPOT_LIQUIDITY_RECOVERY"
    )
    assert recovery["macro_policy"] == "BEARISH_RECOVERY_LONG_ONLY"
    assert recovery["playbook_risk_multiplier"] == 0.40
    assert "MACRO_1D_BEARISH_LONG_LOCK" not in recovery["hard_blockers"]
    assert recovery["stop_loss"] < recovery["entry_price"]
    assert recovery["take_profit_2"] > recovery["take_profit_1"]

    ordinary = [
        item
        for item in opportunities
        if item["market"] == "ADA-EUR"
        and item["family"] != "BEAR_SPOT_LIQUIDITY_RECOVERY"
    ]
    assert ordinary
    assert all(
        "MACRO_1D_BEARISH_LONG_LOCK" not in item["hard_blockers"]
        for item in ordinary
    )
    assert all(item["macro_policy"] == "WEIGHTED_SOFT_REGIME" for item in ordinary)


@pytest.mark.parametrize(
    ("provider_status", "trend_4h", "expected_blocker"),
    [
        ("STALE", True, "CMC_MACRO_DATA_NOT_READY_BEAR_RECOVERY"),
        ("READY", False, "BEAR_RECOVERY_REQUIRES_POSITIVE_4H"),
    ],
)
def test_bear_recovery_remains_fail_closed_on_macro_integrity(
    provider_status: str,
    trend_4h: bool,
    expected_blocker: str,
) -> None:
    row = realtime_row()
    row["downside_sweep_reclaim_1m"] = True
    opportunities = build_event_driven_opportunities(
        {
            "observed_at": "2026-08-04T20:00:00+00:00",
            "markets": [realtime_row("BTC-EUR"), row],
        },
        tactical_opportunities=[
            {
                **context(),
                "family": "LIQUIDITY_SWEEP_RECOVERY",
            }
        ],
        macro_regime="MACRO_RISK_OFF",
        macro_context={
            "features": {
                "btc_1d_trend_up": False,
                "btc_4h_trend_up": trend_4h,
            },
            "provider_refresh": {
                "coinmarketcap_global": provider_status,
            },
        },
    )
    recovery = next(
        item
        for item in opportunities
        if item["family"] == "BEAR_SPOT_LIQUIDITY_RECOVERY"
    )
    assert recovery["state"] != "ENTRY_READY"
    assert expected_blocker in recovery["hard_blockers"]


def test_lifecycle_is_append_only_restart_safe_and_does_not_expire_fills(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "lifecycle.jsonl"
    state_path = tmp_path / "state.json"
    ledger = OpportunityLifecycleLedger(
        ledger_path=ledger_path,
        state_path=state_path,
    )
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        evidence_by_family={"VOLUME_EXPANSION": 0.8},
    )[0]
    assert ledger.upsert(opportunity) is True
    assert ledger.upsert(opportunity) is False
    identity = opportunity["opportunity_id"]
    ledger.transition(identity, OpportunityState.ORDER_INTENT_CREATED)
    ledger.transition(identity, OpportunityState.ORDER_SUBMITTED)
    ledger.transition(identity, OpportunityState.PARTIALLY_FILLED)
    ledger.transition(identity, OpportunityState.FILLED)
    restarted = OpportunityLifecycleLedger(
        ledger_path=ledger_path,
        state_path=state_path,
    )
    assert restarted.state[identity]["state"] == "FILLED"
    expired = restarted.expire(observed_at=observed + timedelta(hours=1))
    assert expired == 0
    assert restarted.state[identity]["state"] == "FILLED"
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(records) == 5
    assert all(record["record_hash"] for record in records)


def test_lifecycle_expires_only_pre_entry_opportunities(tmp_path: Path) -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        evidence_by_family={"VOLUME_EXPANSION": 0.8},
    )[0]
    identity = opportunity["opportunity_id"]

    assert ledger.upsert(opportunity) is True
    assert ledger.expire(observed_at=observed + timedelta(hours=3)) == 1
    assert ledger.state[identity]["state"] == "EXPIRED"


def test_orphan_intent_requires_green_reconciliation_then_fresh_validation(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
    )[0]
    identity = opportunity["opportunity_id"]
    assert ledger.upsert(opportunity) is True
    ledger.transition(identity, OpportunityState.ORDER_INTENT_CREATED)

    assert (
        ledger.recover_orphan_order_intents(
            reconciliation_ready=False,
            observed_at=observed + timedelta(minutes=1),
        )
        == 0
    )
    assert ledger.state[identity]["state"] == "ORDER_INTENT_CREATED"
    assert (
        ledger.recover_orphan_order_intents(
            reconciliation_ready=True,
            observed_at=observed + timedelta(minutes=1),
        )
        == 1
    )
    assert ledger.state[identity]["state"] == "WATCHING"
    assert ledger.state[identity]["orphan_intent_recovered"] is True
    assert ledger.state[identity]["duplicate_submission_allowed"] is False


def test_expired_orphan_intent_cannot_be_retried(tmp_path: Path) -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
    )[0]
    identity = opportunity["opportunity_id"]
    ledger.upsert(opportunity)
    ledger.transition(identity, OpportunityState.ORDER_INTENT_CREATED)

    assert (
        ledger.recover_orphan_order_intents(
            reconciliation_ready=True,
            observed_at=observed + timedelta(hours=3),
        )
        == 1
    )
    assert ledger.state[identity]["state"] == "EXPIRED"


@pytest.mark.parametrize(
    "execution_state",
    [
        OpportunityState.ORDER_INTENT_CREATED,
        OpportunityState.ORDER_SUBMITTED,
        OpportunityState.PARTIALLY_FILLED,
        OpportunityState.FILLED,
        OpportunityState.MANAGING,
        OpportunityState.EXITING,
        OpportunityState.CLOSED,
    ],
)
def test_scanner_upsert_cannot_regress_execution_owned_state(
    tmp_path: Path,
    execution_state: OpportunityState,
) -> None:
    observed = datetime(2026, 8, 3, 12, tzinfo=UTC)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / f"{execution_state.value}.jsonl",
        state_path=tmp_path / f"{execution_state.value}.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": observed.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
    )[0]
    identity = opportunity["opportunity_id"]
    assert ledger.upsert(opportunity) is True
    ledger.transition(identity, execution_state)

    refreshed = {
        **opportunity,
        "state": OpportunityState.ENTRY_READY.value,
        "score": float(opportunity["score"]) + 20.0,
    }
    assert ledger.upsert(refreshed) is False
    assert ledger.state[identity]["state"] == execution_state.value


def test_entry_ready_requires_five_seconds_of_persistent_realtime_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.event_driven_playbooks as module

    start = datetime(2026, 8, 3, 12, tzinfo=UTC)
    selected_now = start
    monkeypatch.setattr(module, "utc_now", lambda: selected_now)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": start.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        evidence_by_family={"VOLUME_EXPANSION": 0.8},
    )[0]
    identity = opportunity["opportunity_id"]

    assert ledger.upsert(opportunity) is True
    assert ledger.state[identity]["state"] == "ARMED"
    assert ledger.state[identity]["persistence_pending"] is True
    selected_now = start + timedelta(seconds=4)
    assert ledger.upsert(opportunity) is False
    assert ledger.state[identity]["state"] == "ARMED"
    selected_now = start + timedelta(seconds=5)
    assert ledger.upsert(opportunity) is True
    assert ledger.state[identity]["state"] == "ENTRY_READY"
    assert ledger.state[identity]["persistence_pending"] is False


def test_entry_ready_invalidates_when_realtime_match_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.event_driven_playbooks as module

    start = datetime(2026, 8, 3, 12, tzinfo=UTC)
    selected_now = start
    monkeypatch.setattr(module, "utc_now", lambda: selected_now)
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = build_event_driven_opportunities(
        {
            "observed_at": start.isoformat(),
            "markets": [realtime_row("BTC-EUR"), realtime_row()],
        },
        tactical_opportunities=[context()],
        evidence_by_family={"VOLUME_EXPANSION": 0.8},
    )[0]
    identity = opportunity["opportunity_id"]
    assert ledger.upsert(opportunity) is True
    selected_now = start + timedelta(seconds=5)
    assert ledger.upsert(opportunity) is True
    assert ledger.state[identity]["state"] == "ENTRY_READY"

    selected_now = start + timedelta(seconds=9)
    assert ledger.invalidate_absent((), observed_at=selected_now) == []
    assert ledger.state[identity]["state"] == "ENTRY_READY"
    selected_now = start + timedelta(seconds=10)
    invalidated = ledger.invalidate_absent((), observed_at=selected_now)

    assert len(invalidated) == 1
    assert invalidated[0]["state"] == "INVALIDATED"
    assert invalidated[0]["reason_codes"] == [
        "REALTIME_PLAYBOOK_FACTS_NO_LONGER_MATCH",
        "NO_STALE_ENTRY_REUSE",
    ]


def test_lifecycle_migrates_existing_entry_when_band_changes(
    tmp_path: Path,
) -> None:
    ledger = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=tmp_path / "state.json",
    )
    opportunity = {
        **build_event_driven_opportunities(
            {
                "observed_at": datetime(2026, 8, 3, 12, tzinfo=UTC).isoformat(),
                "markets": [realtime_row("BTC-EUR"), realtime_row()],
            },
            tactical_opportunities=[context()],
        )[0],
        "state": "ENTRY_READY",
    }
    assert ledger.upsert(opportunity) is True
    identity = opportunity["opportunity_id"]
    updated = {
        **opportunity,
        "state": "ARMED",
        "parameter_band_status": "OUTSIDE_BAND",
        "hard_blockers": ["PLAYBOOK_SCORE_OUTSIDE_VALIDATED_BAND"],
    }

    assert ledger.upsert(updated) is True
    assert ledger.state[identity]["state"] == "ARMED"
    assert ledger.state[identity]["parameter_band_status"] == "OUTSIDE_BAND"


def test_restart_demotes_legacy_entry_ready_without_band_hash(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "opportunities": {
                    "legacy": {
                        "opportunity_id": "legacy",
                        "state": "ENTRY_READY",
                        "hard_blockers": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    restarted = OpportunityLifecycleLedger(
        ledger_path=tmp_path / "lifecycle.jsonl",
        state_path=state_path,
    )

    assert restarted.state["legacy"]["state"] == "ARMED"
    assert (
        "PLAYBOOK_BAND_REVALIDATION_PENDING"
        in restarted.state["legacy"]["hard_blockers"]
    )


def test_projection_compacts_only_prior_day_nonexecuted_terminal_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.event_driven_playbooks as module

    now = datetime(2026, 8, 11, 12, tzinfo=UTC)
    monkeypatch.setattr(module, "utc_now", lambda: now)
    state_path = tmp_path / "state.json"
    ledger_path = tmp_path / "lifecycle.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "opportunities": {
                    "old-rejected": {
                        "opportunity_id": "old-rejected",
                        "state": "INVALIDATED",
                        "market": "BTC-EUR",
                        "playbook_id": "PB",
                        "last_updated_at": "2026-08-10T23:59:59+00:00",
                        "feature_snapshot": {"large": "x" * 10_000},
                    },
                    "today-rejected": {
                        "opportunity_id": "today-rejected",
                        "state": "EXPIRED",
                        "last_updated_at": "2026-08-11T00:00:00+00:00",
                        "feature_snapshot": {"kept": True},
                    },
                    "old-closed": {
                        "opportunity_id": "old-closed",
                        "state": "CLOSED",
                        "last_updated_at": "2026-08-10T12:00:00+00:00",
                        "feature_snapshot": {"fill_attribution": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    ledger = OpportunityLifecycleLedger(
        ledger_path=ledger_path,
        state_path=state_path,
    )

    ledger.write_projection()

    projection = json.loads(state_path.read_text(encoding="utf-8"))
    old_rejected = projection["opportunities"]["old-rejected"]
    assert old_rejected["terminal_projection_compacted"] is True
    assert "feature_snapshot" not in old_rejected
    assert projection["opportunities"]["today-rejected"][
        "feature_snapshot"
    ] == {"kept": True}
    assert projection["opportunities"]["old-closed"][
        "feature_snapshot"
    ] == {"fill_attribution": True}
    assert projection["compacted_terminal_count"] == 1

    restarted = OpportunityLifecycleLedger(
        ledger_path=ledger_path,
        state_path=state_path,
    )
    assert restarted.state["old-rejected"]["state"] == "INVALIDATED"
    assert (
        restarted.upsert(
            {
                "opportunity_id": "old-rejected",
                "state": "ENTRY_READY",
            }
        )
        is False
    )
