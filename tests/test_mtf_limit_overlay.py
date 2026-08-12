from __future__ import annotations

import pandas as pd

from research.mtf_limit_overlay import (
    LimitOverlayParameters,
    simulate_limit_overlay_market,
)
from research.multi_timeframe_authority import MultiTimeframeParameters


def test_limit_overlay_waits_for_future_15m_bar() -> None:
    index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=12,
        freq="15min",
        name="timestamp",
    )
    frame = pd.DataFrame(
        {
            "open": [101.0] * 12,
            "high": [102.0] * 12,
            "low": [101.0] * 6 + [99.0] + [101.0] * 5,
            "close": [101.0] * 12,
            "volume": [10.0] * 12,
        },
        index=index,
    )
    featured = pd.DataFrame(
        {
            "decision_at": [index[4]],
            "entry_signal": [True],
            "exit_signal": [False],
            "entry_level": [100.0],
            "close": [101.0],
            "atr": [2.0],
            "confirmed_fractal_low": [95.0],
        },
        index=pd.DatetimeIndex([index[0]], name="timestamp"),
    )
    parameters = LimitOverlayParameters(
        parent=MultiTimeframeParameters(
            timeframe="1h",
            entry_lookback=2,
            exit_lookback=2,
        ),
        entry_window_15m_bars=4,
    )

    trades = simulate_limit_overlay_market(
        frame,
        parameters,
        parent_featured=featured,
    )

    assert len(trades) == 1
    assert trades[0]["entry_timestamp"] == index[6].isoformat()
    assert trades[0]["entry_price"] == 100.0
    assert pd.Timestamp(trades[0]["entry_timestamp"]) > index[4]


def test_limit_overlay_dna_includes_parent_identity() -> None:
    first = LimitOverlayParameters(
        parent=MultiTimeframeParameters(
            timeframe="1h",
            entry_lookback=180,
            exit_lookback=60,
        ),
        entry_window_15m_bars=4,
    )
    second = LimitOverlayParameters(
        parent=MultiTimeframeParameters(
            timeframe="2h",
            entry_lookback=240,
            exit_lookback=72,
        ),
        entry_window_15m_bars=8,
    )

    assert len(first.dna_hash) == 64
    assert first.dna_hash != second.dna_hash
