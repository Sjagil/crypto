from __future__ import annotations

import pandas as pd

from core.cli import build_parser
from research.combinatorial_lab import _synthetic_ohlcv
from research.features import FeaturePipeline
from research.trade_count_audit import (
    _multi_timeframe_definition,
    _simulate_round_trips,
    _volume_signal_definition,
)
from research.volume_strategy_campaign import (
    VolumeStrategyDNA,
    _parameter_path,
)


def _row(
    archetype: str,
    *,
    market: str = "BTC-EUR",
    timeframe: str = "1h",
    coordinate: int = 2,
) -> VolumeStrategyDNA:
    return VolumeStrategyDNA(
        market=market,
        timeframe=timeframe,
        archetype=archetype,
        coordinate=coordinate,
        parameters=_parameter_path(archetype, coordinate),
    )


def test_every_volume_family_has_exact_audit_conditions() -> None:
    frame = _synthetic_ohlcv(rows=600, timeframe="1h", seed=9_111)
    for archetype in (
        "DONCHIAN_RVOL_BREAKOUT",
        "TREND_PULLBACK_DRYUP_RECOVERY",
        "VOLUME_CONTRACTION_BREAKOUT",
        "OBV_CMF_CONTINUATION",
        "VWAP_MFI_RECLAIM",
    ):
        definition = _volume_signal_definition(frame, _row(archetype))
        assert definition.conditions
        assert definition.entry_signal.index.equals(frame.index)
        assert definition.exit_signal.index.equals(frame.index)


def test_round_trip_definition_counts_entries_and_terminal_liquidation() -> None:
    frame = _synthetic_ohlcv(rows=8, timeframe="1h", seed=8_222)
    entry = pd.Series(
        [False, True, True, False, False, True, False, False],
        index=frame.index,
    )
    exit_ = pd.Series(
        [False, False, False, True, False, False, False, False],
        index=frame.index,
    )
    result = _simulate_round_trips(
        frame,
        entry_signal=entry,
        exit_signal=exit_,
    )
    assert result["filled_entry_count"] == 2
    assert result["filled_exit_count"] == 2
    assert result["completed_round_trip_count"] == 2
    assert result["blocked_existing_position"] == 1
    assert result["terminal_liquidation_count"] == 1


def test_multi_timeframe_filter_waits_for_daily_close() -> None:
    frame_4h = _synthetic_ohlcv(rows=1_500, timeframe="4h", seed=7_333)
    frame_1d = (
        frame_4h.resample("1D", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    definition, alignment = _multi_timeframe_definition(
        frame_4h,
        frame_1d,
        _row("DONCHIAN_RVOL_BREAKOUT", timeframe="4h"),
    )
    assert alignment["causal"]
    assert not alignment["incomplete_context_visible"]
    assert alignment["execution_rows_before_join"] == len(frame_4h)
    assert alignment["execution_rows_after_join"] == len(frame_4h)
    assert "last_fully_closed_1d_above_ema200" in definition.conditions


def test_trade_count_audit_is_a_canonical_main_cli_command() -> None:
    args = build_parser().parse_args(["trade-count-audit"])
    assert args.command == "trade-count-audit"


def test_arbitrary_supported_higher_timeframe_is_closed_bar_aligned() -> None:
    base = _synthetic_ohlcv(rows=1_000, timeframe="15m", seed=6_444)
    higher = (
        base.resample("2h", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    higher.attrs.update(market="BTC-EUR", timeframe="2h")
    result = FeaturePipeline().build(
        base,
        market="BTC-EUR",
        higher_timeframes={"2h": higher},
    )
    assert "htf_2h_regime_bullish" in result
    source = result["htf_2h_source_timestamp"].dropna()
    source_timestamps = pd.DatetimeIndex(
        pd.to_datetime(source, utc=True)
    )
    decision_timestamps = pd.DatetimeIndex(
        pd.to_datetime(source.index, utc=True)
    )
    assert bool(
        (
            source_timestamps.asi8
            + 2 * 60 * 60 * 1_000_000_000
            <= decision_timestamps.asi8
            + 15 * 60 * 1_000_000_000
        ).all()
    )
