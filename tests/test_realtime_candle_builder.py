from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from data.realtime_candle_builder import RealtimeCandleBuilder


def test_builder_closes_on_next_real_trade_without_filling_gaps(
    tmp_path: Path,
) -> None:
    builder = RealtimeCandleBuilder(output_path=tmp_path / "candles.json")
    builder.ingest_trade(
        market="ETH-EUR",
        timestamp=datetime(2026, 8, 8, 12, 1, tzinfo=UTC),
        observed_at=datetime(2026, 8, 8, 12, 1, 1, tzinfo=UTC),
        price=100.0,
        base_quantity=1.0,
        quote_quantity=100.0,
        aggressor_side="buy",
    )
    builder.ingest_trade(
        market="ETH-EUR",
        timestamp=datetime(2026, 8, 8, 12, 16, tzinfo=UTC),
        observed_at=datetime(2026, 8, 8, 12, 16, 1, tzinfo=UTC),
        price=102.0,
        base_quantity=1.0,
        quote_quantity=102.0,
        aggressor_side="sell",
    )

    snapshot = builder.snapshot()
    fifteen = snapshot["closed_candles"]["ETH-EUR:15m"]
    assert len(fifteen) == 1
    assert fifteen[0]["timestamp"] == "2026-08-08T12:00:00+00:00"
    assert snapshot["synthetic_candles_created"] == 0
    assert snapshot["forward_filled"] is False


def test_builder_keeps_open_candle_out_of_strategy_projection(
    tmp_path: Path,
) -> None:
    builder = RealtimeCandleBuilder(output_path=tmp_path / "candles.json")
    builder.ingest_trade(
        market="BTC-EUR",
        timestamp=datetime(2026, 8, 8, 12, 4, tzinfo=UTC),
        observed_at=datetime(2026, 8, 8, 12, 4, 1, tzinfo=UTC),
        price=65_000.0,
        base_quantity=0.01,
        quote_quantity=650.0,
        aggressor_side="buy",
    )

    snapshot = builder.snapshot()
    assert "BTC-EUR:15m" not in snapshot["closed_candles"]
    assert snapshot["open_candles_execution_only"]["BTC-EUR:15m"]["closed"] is False
    assert snapshot["strategy_truth_uses_closed_only"] is True


def test_builder_uses_utc_boundaries_and_ignores_late_events(
    tmp_path: Path,
) -> None:
    builder = RealtimeCandleBuilder(output_path=tmp_path / "candles.json")
    common = {
        "market": "SOL-EUR",
        "base_quantity": 1.0,
        "quote_quantity": 100.0,
        "aggressor_side": "buy",
    }
    builder.ingest_trade(
        **common,
        timestamp=datetime(2026, 8, 8, 12, 15, tzinfo=UTC),
        observed_at=datetime(2026, 8, 8, 12, 15, 1, tzinfo=UTC),
        price=100.0,
    )
    builder.ingest_trade(
        **common,
        timestamp=datetime(2026, 8, 8, 12, 14, tzinfo=UTC),
        observed_at=datetime(2026, 8, 8, 12, 15, 2, tzinfo=UTC),
        price=99.0,
    )

    assert builder.snapshot()["late_events_ignored"] == 3

