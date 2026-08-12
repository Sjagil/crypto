from __future__ import annotations

import numpy as np
import pandas as pd

from core.early_move_detector import detect_early_moves


def _inputs(
    *,
    extended: bool = False,
    orderflow_ready: bool = True,
    first_pullback: bool = False,
):
    index = pd.date_range("2026-07-30", periods=120, freq="15min", tz="UTC")
    close = np.full(120, 100.0)
    close[-5:] = [100.0, 100.1, 100.2, 100.35, 101.6]
    volume = [100.0] * 119 + [350.0]
    if first_pullback:
        close[-6:] = [100.0, 100.1, 100.2, 100.35, 101.6, 101.2]
        volume = [100.0] * 118 + [350.0, 120.0]
    frame = pd.DataFrame(
        {
            "open": close - 0.08,
            "high": close + 0.10,
            "low": close - 0.35,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    feature_frame = pd.DataFrame(
        {
            "atr_14": np.full(120, 0.50),
            "ema_20": np.full(120, 100.9 if not extended else 99.9),
        },
        index=index,
    )
    four_hour = pd.DataFrame(
        {
            "close": [100.0, 102.0 if not extended else 113.0],
        },
        index=pd.date_range("2026-07-29", periods=2, freq="4h", tz="UTC"),
    )
    flow = {
        "status": "READY" if orderflow_ready else "DATA_PENDING",
        "fresh": orderflow_ready,
        "synthetic_data_used": False,
        "horizons": {
            "15m": {
                "ofi_normalized_mean": 0.15,
                "trade_delta_percentage": 0.12,
            }
        },
        "orderbook_imbalance_top_10": 0.10,
    }
    frames = {("ETH-EUR", "15m"): frame, ("ETH-EUR", "4h"): four_hour}
    features = {("ETH-EUR", "15m"): feature_frame}
    rotation = [
        {"market": "BTC-EUR", "returns": {"return_1h": 0.001}},
        {
            "market": "ETH-EUR",
            "rotation_score": 75.0,
            "returns": {"return_1h": 0.016},
        },
    ]
    mechanics = {"markets": {"ETH-EUR": {"orderflow_15m": flow}}}
    return frames, features, rotation, mechanics


def test_detects_fresh_price_volume_acceleration_without_execution_authority():
    frames, features, rotation, mechanics = _inputs()

    rows = detect_early_moves(
        frames,
        features,
        ["ETH-EUR"],
        rotation=rotation,
        mechanics=mechanics,
        regime="BROAD_RISK_ON",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "EARLY_MOMENTUM_ALERT"
    assert row["formula"]["relative_volume_20"] == 3.5
    assert row["formula"]["confirmation_count"] >= 4
    assert row["formula"]["orderflow"]["usable_for_confirmation"] is True
    assert row["closed_candle_only"] is True
    assert row["entry_trigger_confirmed"] is False
    assert row["live_authority_granted"] is False
    assert row["orders_generated"] == 0
    assert row["orders_submitted"] == 0


def test_extended_move_is_pullback_pending_instead_of_chase_entry():
    frames, features, rotation, mechanics = _inputs(extended=True)

    row = detect_early_moves(
        frames,
        features,
        ["ETH-EUR"],
        rotation=rotation,
        mechanics=mechanics,
        regime="BROAD_RISK_ON",
    )[0]

    assert row["status"] == "EXTENDED_MOVE_WAIT_FOR_PULLBACK"
    assert "MOVE_EXTENDED_ABOVE_ATR_ENVELOPE" in row["timeframe_conflicts"]
    assert "WAIT_FOR_PULLBACK" in row["reason_not_yet_entered"]
    assert row["live_authority_granted"] is False


def test_first_controlled_pullback_is_warm_for_realtime_family_trigger():
    frames, features, rotation, mechanics = _inputs(first_pullback=True)

    row = detect_early_moves(
        frames,
        features,
        ["ETH-EUR"],
        rotation=rotation,
        mechanics=mechanics,
        regime="RECOVERY",
    )[0]

    assert row["status"] == "FIRST_PULLBACK_AFTER_IMPULSE"
    assert row["family"] == "FIRST_PULLBACK_AFTER_IMPULSE"
    assert row["setup_valid_on_closed_candle"] is True
    assert row["entry_trigger_confirmed"] is False
    assert row["formula"]["positive_altcoin_breadth_1h"] == 1.0
    assert row["execution_scope"] == (
        "EXISTING_BREAKOUT_PULLBACK_FAMILY_PRECHECK_REQUIRED"
    )


def test_missing_prospective_orderflow_never_adds_positive_flow_score():
    frames, features, rotation, mechanics = _inputs(orderflow_ready=False)

    row = detect_early_moves(
        frames,
        features,
        ["ETH-EUR"],
        rotation=rotation,
        mechanics=mechanics,
        regime="UNKNOWN",
    )[0]

    flow = row["formula"]["orderflow"]
    assert flow["usable_for_confirmation"] is False
    assert flow["component"] == 0.0
    assert row["formula"]["confirmations"]["prospective_buy_flow"] is False
    assert row["orders_submitted"] == 0
