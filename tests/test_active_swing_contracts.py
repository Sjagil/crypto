from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.active_swing_contracts import (
    TimeframeObservation,
    causal_asof_join,
    normalize_live_opportunity,
)


def test_causal_asof_join_excludes_future_and_unavailable_bars() -> None:
    decision = datetime(2026, 8, 12, 12, tzinfo=UTC)
    rows = [
        TimeframeObservation(
            market="BTC-EUR",
            timeframe="15m",
            bar_close_time=decision - timedelta(minutes=15),
            available_at=decision - timedelta(minutes=14, seconds=50),
            source="BITVAVO",
            quality="VALID",
            features={"close": 100},
        ),
        TimeframeObservation(
            market="BTC-EUR",
            timeframe="15m",
            bar_close_time=decision,
            available_at=decision + timedelta(seconds=1),
            source="BITVAVO",
            quality="VALID",
            features={"close": 101},
        ),
    ]

    selected = causal_asof_join(
        decision_time=decision,
        observations=rows,
        required_timeframes=("15m",),
    )

    assert selected["15m"].features["close"] == 100


def test_causal_asof_join_fails_when_required_timeframe_is_missing() -> None:
    with pytest.raises(ValueError, match="missing required causal timeframes"):
        causal_asof_join(
            decision_time=datetime(2026, 8, 12, 12, tzinfo=UTC),
            observations=(),
            required_timeframes=("15m", "1h"),
        )


def test_live_normalization_accepts_utc_string_but_never_grants_authority() -> None:
    row = {
        "opportunity_id": "source-1",
        "episode_id": "episode-1",
        "market": "ADA-EUR",
        "playbook_id": "TACTICAL_15M_VOLUME_EXPANSION",
        "family": "VOLUME_EXPANSION",
        "playbook_dna": "a" * 64,
        "state": "ENTRY_READY",
        "entry_price": 0.18,
        "stop_loss": 0.175,
        "take_profit_1": 0.19,
        "take_profit_2": 0.20,
        "execution_economics": {
            "expected_net_value_bps": 20,
            "net_rr_target_2": 2.5,
            "roundtrip_fee_bps": 50,
        },
        "higher_timeframe_parent": {
            "entry_timeframe": "15m",
            "confirmation_timeframe": "1h",
            "regime_timeframe": "4h",
            "alignment_score": 0.8,
        },
        "detected_at": "2026-08-12T12:00:00+00:00",
        "setup_detected_ts": "2026-08-12T11:00:00+00:00",
        "valid_until": "2026-08-12T16:00:00+00:00",
        "time_stop_minutes": 1440,
        "feature_snapshot": {"snapshot_id": "pit-1"},
        "realtime_inputs": {"spread_bps": 7, "ask_depth_eur_top_10": 1000},
    }

    contract = normalize_live_opportunity(
        row,
        decision_time="2026-08-12T12:00:00+00:00",
    )

    assert contract.lifecycle.value == "ENTRY_READY"
    assert contract.timeframe_contract.required_timeframes == (
        "15m",
        "1h",
        "4h",
    )
    assert contract.execution_authority is False
    assert contract.retail_realizable is False
    assert contract.bitvavo_realizable_quantity is None
