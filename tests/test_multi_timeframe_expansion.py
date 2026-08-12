from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.settings import PathSettings, Settings
from core.live_universe import (
    LIVE_REQUIRED_TIMEFRAMES,
    REQUIRED_TIMEFRAMES,
    RESEARCH_MINIMUM_ROWS,
    RESEARCH_TIMEFRAMES,
    _dynamic_preferred_markets,
    build_tiered_trading_universe,
    select_live_universe,
)
from research.multi_timeframe_authority import (
    MultiTimeframeParameters,
    _confirmed_fractal_low,
    _feature_frame,
    _load_frame,
    _metrics,
)
from ui.server import (
    HTML_DOCUMENT,
    _trend_label,
    build_daily_pnl_calendar,
    build_multi_timeframe_matrix,
    build_paper_read_model,
    build_trending_read_model,
    build_ui_snapshot,
)
from utils.common import append_jsonl, atomic_write_json


def _settings(tmp_path: Path) -> Settings:
    base = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    )
    return base.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_confirmed_fractal_is_not_visible_before_right_bars_close() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
    low = pd.Series([5.0, 4.0, 1.0, 4.0, 5.0, 4.0, 3.0, 4.0], index=index)

    confirmed = _confirmed_fractal_low(low, span=2)

    assert pd.isna(confirmed.iloc[2])
    assert pd.isna(confirmed.iloc[3])
    assert confirmed.iloc[4] == 1.0


def test_multi_timeframe_loader_accepts_canonical_timestamp_index(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    path = settings.paths.processed_data_dir / "BTC-EUR_15m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
            "volume": [10.0, 11.0, 12.0],
        },
        index=pd.Index(index, name="timestamp"),
    ).to_parquet(path)

    loaded = _load_frame(settings, "BTC-EUR", "15m")

    assert list(loaded.columns) == ["close", "high", "low", "open", "volume"]
    assert loaded.index.equals(index)


def test_dynamic_live_universe_includes_reviewed_top50_eur_markets(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    universe = tmp_path / "output" / "universe"
    universe.mkdir(parents=True)
    atomic_write_json(
        universe / "top50_current.json",
        {
            "rows": [
                {
                    "rank": 8,
                    "symbol": "DOGE",
                    "eur_spot_market": "DOGE-EUR",
                    "stablecoin": False,
                    "wrapped": False,
                    "leveraged_token": False,
                    "staking_derivative": False,
                },
                {
                    "rank": 4,
                    "symbol": "USDT",
                    "eur_spot_market": "USDT-EUR",
                    "stablecoin": True,
                    "wrapped": False,
                    "leveraged_token": False,
                    "staking_derivative": False,
                },
            ]
        },
    )

    markets = _dynamic_preferred_markets(settings)

    assert "DOGE-EUR" in markets
    assert "USDT-EUR" not in markets


def test_multi_timeframe_dna_changes_with_timeframe() -> None:
    one_hour = MultiTimeframeParameters(
        timeframe="1h",
        entry_lookback=180,
        exit_lookback=60,
    )
    two_hour = MultiTimeframeParameters(
        timeframe="2h",
        entry_lookback=180,
        exit_lookback=60,
    )

    assert len(one_hour.dna_hash) == 64
    assert one_hour.dna_hash != two_hour.dna_hash


def test_15m_mtf_dna_has_hourly_confirmation_and_is_distinct() -> None:
    fifteen_minute = MultiTimeframeParameters(
        timeframe="15m",
        entry_lookback=96,
        exit_lookback=32,
        daily_ema_period=2,
    )
    one_hour = MultiTimeframeParameters(
        timeframe="1h",
        entry_lookback=96,
        exit_lookback=32,
        daily_ema_period=2,
    )

    assert "15M_H1" in fifteen_minute.strategy_id
    assert fifteen_minute.dna_hash != one_hour.dna_hash


def test_15m_hourly_confirmation_uses_only_fully_closed_hour() -> None:
    index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=24,
        freq="15min",
        name="timestamp",
    )
    close = pd.Series(
        [100.0] * 4 + [101.0] * 4 + [102.0] * 16,
        index=index,
    )
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        },
        index=index,
    )
    parameters = MultiTimeframeParameters(
        timeframe="15m",
        entry_lookback=2,
        exit_lookback=2,
        daily_ema_period=1,
        atr_period=2,
    )

    featured = _feature_frame(frame, parameters)

    assert pd.isna(featured.loc[index[2], "hourly_available_at"])
    assert featured.loc[index[3], "hourly_available_at"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert featured.loc[index[3], "hourly_close"] == 100.0


def test_metrics_include_profit_factor_and_drawdown() -> None:
    metrics = _metrics(
        [
            {"net_return": 0.10},
            {"net_return": -0.05},
            {"net_return": 0.04},
        ]
    )

    assert metrics["trade_count"] == 3
    assert metrics["profit_factor"] == pytest.approx(2.8)
    assert metrics["expectancy"] > 0
    assert metrics["maximum_drawdown"] < 0


def test_candle_health_accepts_named_timestamp_index(
    tmp_path: Path,
) -> None:
    from core.live_universe import candle_health

    settings = _settings(tmp_path)
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(
        "2026-07-31T10:00:00Z",
        periods=5,
        freq="1h",
        name="timestamp",
    )
    pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [10.0] * 5,
        },
        index=index,
    ).to_parquet(settings.paths.processed_data_dir / "NPC-EUR_1h.parquet")

    report = candle_health(
        settings,
        markets=("NPC-EUR",),
        timeframes=("1h",),
        now=pd.Timestamp("2026-07-31T16:00:00Z").to_pydatetime(),
        write_artifact=False,
    )

    assert report["healthy_series"] == 1
    assert report["rows"][0]["status"] == "HEALTHY"


def test_live_universe_selects_five_healthy_allowed_markets(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    markets = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR", "TAO-EUR"]
    eligibility_path = (
        settings.paths.output_dir / "universe" / "top50_eligibility.json"
    )
    atomic_write_json(
        eligibility_path,
        {
            "rows": [
                {
                    "eur_spot_market": market,
                    "execution_eligibility": "LIVE_ELIGIBLE",
                    "execution_reason": "PASSED",
                    "shariah_status": "ALLOWED",
                    "rank": rank,
                }
                for rank, market in enumerate(markets, start=1)
            ]
        },
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": "HEALTHY",
            }
            for market in markets
            for timeframe in REQUIRED_TIMEFRAMES
        ]
    }
    public = {
        market: {
            "venue_available": True,
            "spread_bps": 0.1,
            "visible_ask_depth_eur": 1_000_000.0,
            "quote_volume_24h_eur": 10_000_000.0,
        }
        for market in markets
    }

    result = select_live_universe(
        settings,
        market_snapshot=public,
        candle_report=candles,
        preferred_markets=markets,
    )

    assert result["status"] == "READY"
    assert result["selected_markets"] == markets
    assert result["live_eligible_count"] == 5
    assert result["maximum_concurrent_positions"] == 2


def test_live_universe_accepts_only_explicit_outside_top50_exception(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "execution_market_exceptions.yaml").write_text(
        """
version: 1
default_policy: FAIL_CLOSED
markets:
  NPC-EUR:
    approved: true
    approved_at: "2026-07-28T00:00:00+02:00"
    approval_reference: explicit_test
    allow_outside_top50: true
    spot_only: true
    maximum_order_eur: 5.0
    maximum_total_exposure_eur: 10.0
    requires_approved_strategy_dna: true
    requires_natural_signal: true
""".strip(),
        encoding="utf-8",
    )
    candles = {
        "rows": [
            {
                "market": "NPC-EUR",
                "timeframe": timeframe,
                "status": "HEALTHY",
            }
            for timeframe in REQUIRED_TIMEFRAMES
        ]
    }
    public = {
        "NPC-EUR": {
            "venue_available": True,
            "spread_bps": 1.0,
            "visible_ask_depth_eur": 10_000.0,
            "quote_volume_24h_eur": 1_000_000.0,
        }
    }

    result = select_live_universe(
        settings,
        market_snapshot=public,
        candle_report=candles,
        preferred_markets=("NPC-EUR",),
        minimum_markets=1,
    )

    assert result["status"] == "READY"
    assert result["selected_markets"] == ["NPC-EUR"]
    assert result["rows"][0]["outside_top50_exception"] is True
    assert result["rows"][0]["strategy_dna_authority_granted"] is False
    assert (
        result["rows"][0]["execution_eligibility_basis"]
        == "APPROVED_OUTSIDE_TOP50_EXCEPTION"
    )


def test_optional_weekly_context_does_not_block_intraday_live_market(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    market = "BTC-EUR"
    universe = tmp_path / "output" / "universe"
    universe.mkdir(parents=True)
    atomic_write_json(
        universe / "top50_eligibility.json",
        {
            "rows": [
                {
                    "eur_spot_market": market,
                    "execution_eligibility": "LIVE_ELIGIBLE",
                    "execution_reason": "PASSED",
                    "rank": 1,
                }
            ]
        },
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": (
                    "HEALTHY"
                    if timeframe in LIVE_REQUIRED_TIMEFRAMES
                    else "BLOCKED"
                ),
            }
            for timeframe in REQUIRED_TIMEFRAMES
        ]
    }
    result = select_live_universe(
        settings,
        market_snapshot={
            market: {
                "venue_available": True,
                "spread_bps": 0.2,
                "visible_ask_depth_eur": 100_000.0,
                "quote_volume_24h_eur": 10_000_000.0,
            }
        },
        candle_report=candles,
        preferred_markets=(market,),
        minimum_markets=1,
    )

    assert result["status"] == "READY"
    assert result["selected_markets"] == [market]
    assert result["rows"][0]["optional_context_timeframes"] == ["2h", "1W"]


def test_tiered_universe_expands_shadow_without_granting_live_authority(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    rows = [
        {
            "rank": rank,
            "symbol": market.split("-", 1)[0],
            "eur_spot_market": market,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": (
                "LIVE_ELIGIBLE" if market == "BTC-EUR" else "NOT_EXECUTION_ELIGIBLE"
            ),
            "shariah_status": "ALLOWED" if market == "BTC-EUR" else "REVIEW_REQUIRED",
            "stablecoin": False,
            "wrapped": False,
            "leveraged_token": False,
            "staking_derivative": False,
            "available_at": "2026-07-31T10:00:00+00:00",
        }
        for rank, market in enumerate(
            ("BTC-EUR", "ETH-EUR", "XRP-EUR"),
            start=1,
        )
    ]
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_current.json",
        {"rows": rows},
    )
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_eligibility.json",
        {"rows": rows},
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": "HEALTHY",
                "rows": RESEARCH_MINIMUM_ROWS[timeframe],
            }
            for market in ("BTC-EUR", "ETH-EUR", "XRP-EUR")
            for timeframe in RESEARCH_TIMEFRAMES
        ]
    }

    result = build_tiered_trading_universe(
        settings,
        candle_report=candles,
        live_report={"selected_markets": ["BTC-EUR"]},
        maximum_shadow_markets=2,
    )

    assert result["counts"] == {
        "discovery": 4,
        "research": 4,
        "shadow": 2,
        "paper": 0,
        "live_executable": 1,
        "context_only": 0,
    }
    assert result["shadow_markets"] == ["BTC-EUR", "ETH-EUR"]
    assert result["live_executable_markets"] == ["BTC-EUR"]
    eth = next(row for row in result["rows"] if row["market"] == "ETH-EUR")
    assert eth["highest_tier"] == "SHADOW"
    assert eth["live_authority_inherited"] is False
    assert result["execution_authority_unchanged"] is True
    assert result["orders_submitted"] == 0


def test_shadow_universe_keeps_1h_market_when_only_15m_chain_is_blocked(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    rows = [
        {
            "rank": rank,
            "symbol": market.split("-", 1)[0],
            "eur_spot_market": market,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": "NOT_EXECUTION_ELIGIBLE",
            "shariah_status": "REVIEW_REQUIRED",
            "stablecoin": False,
            "wrapped": False,
            "leveraged_token": False,
            "staking_derivative": False,
            "available_at": "2026-07-31T10:00:00+00:00",
        }
        for rank, market in enumerate(("XRP-EUR", "DOGE-EUR"), start=1)
    ]
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_current.json",
        {"rows": rows},
    )
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_eligibility.json",
        {"rows": rows},
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": (
                    "BLOCKED" if timeframe == "15m" else "HEALTHY"
                ),
                "rows": RESEARCH_MINIMUM_ROWS[timeframe],
            }
            for market in ("XRP-EUR", "DOGE-EUR")
            for timeframe in RESEARCH_TIMEFRAMES
        ]
    }

    result = build_tiered_trading_universe(
        settings,
        candle_report=candles,
        live_report={"selected_markets": []},
        maximum_shadow_markets=2,
    )

    assert result["shadow_markets"] == ["XRP-EUR", "DOGE-EUR"]
    assert result["shadow_required_timeframes"] == ["15m_OR_1h"]
    assert result["shadow_entry_timeframes"] == ["15m", "1h"]
    assert result["shadow_minimum_healthy_entry_timeframes"] == 1
    xrp = next(row for row in result["rows"] if row["market"] == "XRP-EUR")
    assert xrp["shadow_reason_codes"] == []
    assert "15M_CANDLE_CHAIN_BLOCKED" in xrp["timeframe_gap_reason_codes"]
    assert xrp["live_authority_inherited"] is False
    assert result["orders_submitted"] == 0


def test_tiered_universe_can_scan_every_healthy_top50_eur_market(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    markets = [f"COIN{index}-EUR" for index in range(30)]
    rows = [
        {
            "rank": index + 1,
            "symbol": market.split("-", 1)[0],
            "eur_spot_market": market,
            "research_eligibility": "RESEARCH_ELIGIBLE",
            "execution_eligibility": "NOT_EXECUTION_ELIGIBLE",
            "shariah_status": "REVIEW_REQUIRED",
            "stablecoin": False,
            "wrapped": False,
            "leveraged_token": False,
            "staking_derivative": False,
            "available_at": "2026-07-31T10:00:00+00:00",
        }
        for index, market in enumerate(markets)
    ]
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_current.json",
        {"rows": rows},
    )
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_eligibility.json",
        {"rows": rows},
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": "HEALTHY",
                "rows": RESEARCH_MINIMUM_ROWS[timeframe],
            }
            for market in markets
            for timeframe in RESEARCH_TIMEFRAMES
        ]
    }

    result = build_tiered_trading_universe(
        settings,
        candle_report=candles,
        live_report={"selected_markets": []},
        maximum_shadow_markets=50,
    )

    assert result["maximum_shadow_markets"] == 50
    assert result["counts"]["shadow"] == 30
    assert result["shadow_markets"] == markets
    assert result["execution_authority_unchanged"] is True
    assert result["orders_submitted"] == 0


def test_tiered_universe_includes_outside_top50_exception_without_authority(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "execution_market_exceptions.yaml").write_text(
        """
version: 1
default_policy: FAIL_CLOSED
markets:
  NPC-EUR:
    approved: true
    approved_at: "2026-07-28T00:00:00+02:00"
    approval_reference: explicit_test
    allow_outside_top50: true
    spot_only: true
    maximum_order_eur: 5.0
    maximum_total_exposure_eur: 10.0
    requires_approved_strategy_dna: true
    requires_natural_signal: true
""".strip(),
        encoding="utf-8",
    )
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_current.json",
        {
            "rows": [
                {
                    "rank": 1,
                    "symbol": "BTC",
                    "eur_spot_market": "BTC-EUR",
                    "research_eligibility": "RESEARCH_ELIGIBLE",
                    "execution_eligibility": "LIVE_ELIGIBLE",
                    "shariah_status": "ALLOWED",
                    "stablecoin": False,
                    "wrapped": False,
                    "leveraged_token": False,
                    "staking_derivative": False,
                    "available_at": "2026-07-31T10:00:00+00:00",
                }
            ]
        },
    )
    candles = {
        "rows": [
            {
                "market": market,
                "timeframe": timeframe,
                "status": "HEALTHY",
                "rows": RESEARCH_MINIMUM_ROWS[timeframe],
            }
            for market in ("BTC-EUR", "NPC-EUR")
            for timeframe in RESEARCH_TIMEFRAMES
        ]
    }

    result = build_tiered_trading_universe(
        settings,
        candle_report=candles,
        live_report={"selected_markets": ["BTC-EUR"]},
        maximum_shadow_markets=2,
    )

    assert result["discovery_markets"] == [
        "BTC-EUR",
        "NPC-EUR",
        "PYR-EUR",
    ]
    assert result["research_markets"] == [
        "BTC-EUR",
        "NPC-EUR",
        "PYR-EUR",
    ]
    assert result["shadow_markets"] == ["BTC-EUR", "NPC-EUR"]
    npc = next(row for row in result["rows"] if row["market"] == "NPC-EUR")
    assert npc["source"] == "OPERATOR_MARKET_EXCEPTION"
    assert npc["outside_top50_exception"] is True
    assert npc["highest_tier"] == "SHADOW"
    assert npc["live_authority_inherited"] is False
    assert result["orders_submitted"] == 0


def test_tiered_universe_includes_monitor_only_market_without_authority(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    atomic_write_json(
        settings.paths.output_dir / "universe" / "top50_current.json",
        {"rows": []},
    )
    candles = {
        "rows": [
            {
                "market": "PYR-EUR",
                "timeframe": timeframe,
                "status": "HEALTHY",
                "rows": RESEARCH_MINIMUM_ROWS[timeframe],
            }
            for timeframe in RESEARCH_TIMEFRAMES
        ]
    }

    result = build_tiered_trading_universe(
        settings,
        candle_report=candles,
        live_report={"selected_markets": []},
        maximum_shadow_markets=1,
    )

    assert result["research_markets"] == ["PYR-EUR"]
    assert result["shadow_markets"] == ["PYR-EUR"]
    assert result["live_executable_markets"] == []
    pyr = result["rows"][0]
    assert pyr["monitor_only"] is True
    assert pyr["highest_tier"] == "SHADOW"
    assert pyr["execution_eligibility"] == (
        "MONITOR_ONLY_NOT_EXECUTION_ELIGIBLE"
    )
    assert pyr["live_authority_inherited"] is False
    assert result["execution_authority_unchanged"] is True
    assert result["orders_submitted"] == 0


def test_ui_escapes_dynamic_table_values_and_has_no_order_control() -> None:
    assert "replace(/[&<>\"']/g" in HTML_DOCUMENT
    assert "direct-order" not in HTML_DOCUMENT
    assert "emergency-stop" in HTML_DOCUMENT
    assert "P&L Calendar" in HTML_DOCUMENT
    assert "EUR-cashcontinuïteit" in HTML_DOCUMENT
    assert 'id="signalCards"' in HTML_DOCUMENT
    assert 'id="stablecoinCards"' in HTML_DOCUMENT
    assert 'id="mtfMatrix"' in HTML_DOCUMENT
    assert 'id="paperPositions"' in HTML_DOCUMENT
    assert "Automatische paperposities" in HTML_DOCUMENT
    assert "Bitvavo accountinventory" in HTML_DOCUMENT
    assert "geen market fallback" in HTML_DOCUMENT
    assert "Zacht dagdoel (geschaald)" in HTML_DOCUMENT
    assert 'id="inventoryReallocation"' in HTML_DOCUMENT
    assert 'id="maturity"' in HTML_DOCUMENT
    assert "snapshot.crypto_maturity" in HTML_DOCUMENT
    assert "Operator-only inventoryreallocatie" in HTML_DOCUMENT
    assert "Entry-orderpolicy" in HTML_DOCUMENT
    assert "Actuele prijs" in HTML_DOCUMENT
    assert "Entryzone" in HTML_DOCUMENT
    assert "TP1 / TP2" in HTML_DOCUMENT
    assert "GEEN LIVE AUTHORITY" in HTML_DOCUMENT
    assert "overflow-wrap:anywhere" in HTML_DOCUMENT
    assert "ACTIEF — GEEN GELDIGE ENTRY" in HTML_DOCUMENT


def test_paper_read_model_marks_positions_without_mixing_live_state() -> None:
    result = build_paper_read_model(
        {
            "open_positions": 1,
            "positions": {
                "dna-1": {
                    "market": "ETH-EUR",
                    "entry_price": "100",
                    "quantity": "0.5",
                    "stop_loss": "90",
                    "paper_only": True,
                }
            },
        },
        {"rows": [{"market": "ETH-EUR", "midpoint": "110"}]},
    )

    position = result["positions"]["dna-1"]
    assert position["current_price"] == 110.0
    assert position["gross_unrealized_pnl_eur"] == pytest.approx(5.0)
    assert position["gross_return_fraction"] == pytest.approx(0.10)
    assert result["gross_unrealized_pnl_eur"] == pytest.approx(5.0)
    assert result["marked_positions"] == 1
    assert position["paper_only"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "DATA_PENDING"),
        (0.006, "BULLISH"),
        (-0.006, "BEARISH"),
        (0.001, "NEUTRAL"),
    ],
)
def test_ui_trend_label_is_explicit(
    value: float | None,
    expected: str,
) -> None:
    assert _trend_label(value) == expected


def test_multi_timeframe_matrix_uses_closed_weekly_data_without_orders(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "close": [100.0, 101.0, 103.0],
        },
        index=pd.date_range(
            "2026-07-13",
            periods=3,
            freq="7D",
            tz="UTC",
        ),
    ).to_parquet(
        settings.paths.processed_data_dir / "BTC-EUR_1W.parquet"
    )
    rotation = {
        "rows": [
            {
                "rank": 1,
                "market": "BTC-EUR",
                "market_tier": "LIVE_EXECUTABLE",
                "live_market_executable": True,
                "rotation_score": 80.0,
                "returns": {
                    "return_1h": 0.01,
                    "return_2h": 0.02,
                    "return_4h": 0.03,
                    "return_1d": 0.04,
                },
            }
        ]
    }
    opportunities = {
        "all": [
            {
                "market": "BTC-EUR",
                "strategy": "MTF_DONCHIAN",
                "family": "BREAKOUT",
                "status": "NEAR_ENTRY",
            }
        ]
    }

    matrix = build_multi_timeframe_matrix(
        settings,
        rotation,
        opportunities,
    )

    row = matrix["rows"][0]
    assert row["trends"] == {
        "15m": "DATA_PENDING",
        "1h": "BULLISH",
        "2h": "BULLISH",
        "4h": "BULLISH",
        "1d": "BULLISH",
        "1W": "BULLISH",
    }
    assert row["alignment_score"] == 100.0
    assert row["active_families"] == ["BREAKOUT"]
    assert matrix["closed_candle_only"] is True
    assert matrix["orders_generated"] == 0
    assert matrix["orders_submitted"] == 0


def test_daily_pnl_calendar_uses_latest_account_snapshot_per_day(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    path = settings.paths.output_dir / "live" / "events" / "pnl.jsonl"
    for recorded_at, pnl in (
        ("2026-07-30T10:00:00+00:00", 4.0),
        ("2026-07-30T18:00:00+00:00", 7.5),
        ("2026-07-31T10:00:00+00:00", -2.0),
    ):
        append_jsonl(
            path,
            {
                "event": "DAILY_PNL_TARGET_SNAPSHOT",
                "recorded_at": recorded_at,
                "state": {
                    "date_utc": recorded_at[:10],
                    "day_start_equity_eur": 100.0,
                    "current_estimated_equity_eur": 100.0 + pnl,
                    "mark_to_market_pnl_eur": pnl,
                    "scaled_daily_target_eur": 1.0,
                    "non_binding": True,
                },
            },
        )

    calendar = build_daily_pnl_calendar(settings)

    assert calendar["observed_days"] == 2
    assert calendar["rows"][0]["account_wide_mtm_pnl_eur"] == 7.5
    assert calendar["rows"][1]["status"] == "LOSS"
    assert calendar["scope"] == "ACCOUNT_WIDE_INCLUDING_EXTERNAL_INVENTORY"
    assert calendar["strategy_only_pnl_available"] is False


def test_daily_pnl_calendar_flags_large_unexplained_equity_discontinuity(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    path = settings.paths.output_dir / "live" / "events" / "pnl.jsonl"
    for recorded_at, equity in (
        ("2026-07-31T10:00:00+00:00", 750.0),
        ("2026-07-31T10:06:00+00:00", 700.0),
    ):
        append_jsonl(
            path,
            {
                "event": "DAILY_PNL_TARGET_SNAPSHOT",
                "recorded_at": recorded_at,
                "state": {
                    "date_utc": "2026-07-31",
                    "day_start_equity_eur": 750.0,
                    "current_estimated_equity_eur": equity,
                    "mark_to_market_pnl_eur": equity - 750.0,
                    "orders_submitted": 0,
                },
            },
        )

    calendar = build_daily_pnl_calendar(settings)
    row = calendar["rows"][0]

    assert calendar["days_with_unexplained_discontinuity"] == 1
    assert row["pnl_quality"] == "UNEXPLAINED_CAPITAL_FLOW_OR_VALUATION_JUMP"
    assert row["unexplained_equity_step_eur"] == -50.0
    assert row["external_cash_flow_adjusted"] is False
    assert row["raw_cash_flow_adjusted_pnl_eur"] == -50.0
    assert row["cash_flow_adjusted_pnl_eur"] is None
    assert row["return_fraction"] is None
    assert row["status"] == "UNVERIFIED"
    assert calendar["negative_days"] == 0
    assert calendar["unverified_days"] == 1
    assert calendar["months"][0]["pnl_eur"] == 0.0


def test_daily_pnl_calendar_ignores_temporary_missing_public_valuation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    path = settings.paths.output_dir / "live" / "events" / "pnl.jsonl"
    for recorded_at, equity, valuation_status in (
        (
            "2026-07-31T10:00:00+00:00",
            750.0,
            "COMPLETE_MARK_TO_MARKET",
        ),
        (
            "2026-07-31T10:06:00+00:00",
            100.0,
            "PUBLIC_PRICE_VALUATION_UNAVAILABLE",
        ),
        (
            "2026-07-31T10:12:00+00:00",
            752.0,
            "COMPLETE_MARK_TO_MARKET",
        ),
    ):
        append_jsonl(
            path,
            {
                "event": "DAILY_PNL_TARGET_SNAPSHOT",
                "recorded_at": recorded_at,
                "state": {
                    "date_utc": "2026-07-31",
                    "day_start_equity_eur": 750.0,
                    "current_estimated_equity_eur": equity,
                    "mark_to_market_pnl_eur": equity - 750.0,
                    "valuation_status": valuation_status,
                },
            },
        )

    calendar = build_daily_pnl_calendar(settings)
    row = calendar["rows"][0]

    assert row["status"] == "PROFIT"
    assert row["cash_flow_adjusted_pnl_eur"] == 2.0
    assert row["valuation_status"] == "COMPLETE_MARK_TO_MARKET"
    assert calendar["days_with_unexplained_discontinuity"] == 0


def test_daily_pnl_calendar_does_not_double_count_reconciled_fill_fee(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    fill_path = settings.paths.output_dir / "live" / "events" / "fills.jsonl"
    for event, payload in (
        (
            "BITVAVO_ACCOUNT_FILL",
            {"fee": "0.25", "received_at": "2026-07-31T10:00:00Z"},
        ),
        (
            "CANONICAL_FILL",
            {"fee_eur": "0.25", "received_at": "2026-07-31T10:00:01Z"},
        ),
    ):
        append_jsonl(
            fill_path,
            {
                "event": event,
                "recorded_at": payload["received_at"],
                "payload": payload,
            },
        )
    append_jsonl(
        settings.paths.output_dir / "live" / "events" / "pnl.jsonl",
        {
            "event": "DAILY_PNL_TARGET_SNAPSHOT",
            "recorded_at": "2026-07-31T18:00:00Z",
            "state": {
                "date_utc": "2026-07-31",
                "day_start_equity_eur": 100.0,
                "current_estimated_equity_eur": 101.0,
                "mark_to_market_pnl_eur": 1.0,
                "valuation_status": "COMPLETE_MARK_TO_MARKET",
            },
        },
    )

    row = build_daily_pnl_calendar(settings)["rows"][0]

    assert row["fees_eur"] == 0.25
    assert row["fill_events"] == 1


def test_daily_pnl_calendar_adjusts_operator_confirmed_withdrawal(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    flow_path = settings.paths.output_dir / "portfolio" / "external_capital_flows.jsonl"
    append_jsonl(
        flow_path,
        {
            "operator_confirmed": True,
            "date_utc": "2026-07-31",
            "effective_at": "2026-07-31T10:05:00+00:00",
            "amount_eur": "-50",
        },
    )
    pnl_path = settings.paths.output_dir / "live" / "events" / "pnl.jsonl"
    for recorded_at, equity in (
        ("2026-07-31T10:00:00+00:00", 750.0),
        ("2026-07-31T10:06:00+00:00", 700.0),
    ):
        append_jsonl(
            pnl_path,
            {
                "event": "DAILY_PNL_TARGET_SNAPSHOT",
                "recorded_at": recorded_at,
                "state": {
                    "date_utc": "2026-07-31",
                    "day_start_equity_eur": 750.0,
                    "current_estimated_equity_eur": equity,
                    "mark_to_market_pnl_eur": equity - 750.0,
                },
            },
        )

    calendar = build_daily_pnl_calendar(settings)
    row = calendar["rows"][0]

    assert row["external_capital_flow_eur"] == -50.0
    assert row["cash_flow_adjusted_pnl_eur"] == 0.0
    assert row["pnl_quality"] == "OPERATOR_CONFIRMED_CASH_FLOW_ADJUSTED"
    assert calendar["days_with_unexplained_discontinuity"] == 0


def test_trending_advice_never_grants_live_authority_from_hotness() -> None:
    active = {
        "top_5_rotation": [
            {
                "rank": 1,
                "market": "ETH-EUR",
                "rotation_score": 90.0,
                "decision": "FAVOUR",
                "returns": {"return_1h": 0.02},
                "live_market_executable": True,
            }
        ]
    }
    opportunities = {
        "all": [
            {
                "market": "ETH-EUR",
                "status": "ACTIONABLE",
                "score": 80.0,
                "trigger": 3_000.0,
                "stop": 2_900.0,
                "target_1": 3_200.0,
                "target_2": 3_300.0,
                "live_authority_granted": False,
            }
        ]
    }

    trending = build_trending_read_model(
        active,
        opportunities,
        {"regime": "TRENDING_NEUTRAL", "stablecoin_liquidity": {"state": "DRAINING"}},
    )

    row = trending["rows"][0]
    assert row["action"] == "RESEARCH_SIGNAL"
    assert row["live_authority_granted"] is False
    assert row["stop_loss"] == 2_900.0
    assert row["take_profit_2"] == 3_300.0
    assert "sizing verlagen" in row["advice"]
    assert trending["advice_is_entry_signal"] is False
    assert trending["orders_submitted"] == 0


def test_early_move_is_visible_without_rotation_rank_or_live_authority() -> None:
    opportunities = {
        "all": [
            {
                "market": "NPC-EUR",
                "status": "PULLBACK_PENDING",
                "score": 88.0,
                "rotation_score": 70.0,
                "strategy": "EARLY_MOVE_VOLUME_FLOW_15M",
                "timeframe": "15m",
                "current_price": 0.041,
                "entry_zone": [0.038, 0.039],
                "trigger": 0.040,
                "stop": 0.036,
                "target_1": 0.044,
                "target_2": 0.048,
                "reason_not_yet_entered": (
                    "MOVE_EXTENDED_WAIT_FOR_PULLBACK_AND_NEW_CLOSED_CANDLE"
                ),
                "live_authority_granted": False,
                "formula": {
                    "return_15m": 0.04,
                    "return_1h": 0.08,
                    "return_4h": 0.15,
                    "relative_volume_20": 4.2,
                    "volume_robust_zscore": 5.0,
                    "extension_atr": 3.1,
                },
            }
        ]
    }

    trending = build_trending_read_model(
        {"top_5_rotation": []},
        opportunities,
        {"regime": "ALT_BULL", "stablecoin_liquidity": {"state": "STABLE"}},
    )

    row = trending["rows"][0]
    assert row["market"] == "NPC-EUR"
    assert row["action"] == "PULLBACK_PENDING"
    assert "niet jagen" not in row["advice"]
    assert "wacht op pullback" in row["advice"]
    assert row["early_move_formula"]["relative_volume_20"] == 4.2
    assert row["live_authority_granted"] is False
    assert trending["advice_is_entry_signal"] is False
    assert trending["orders_submitted"] == 0


def test_ui_snapshot_exposes_active_scan_cadence_and_timestamp(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    status_path = (
        settings.paths.output_dir / "active_trading" / "status.json"
    )
    atomic_write_json(
        status_path,
        {
            "generated_at": "2026-08-01T19:15:00+00:00",
            "scan_interval_minutes": 15,
            "status": "LIVE_ACTIVE_NO_CURRENT_ENTRY",
            "markets_scanned": ["BTC-EUR", "ETH-EUR"],
        },
    )
    atomic_write_json(
        settings.paths.output_dir / "roadmap" / "crypto_maturity_ladder.json",
        {
            "schema_version": "crypto_maturity_ladder_v1",
            "current_level": "INTERMEDIATE",
            "projects": [{"project_id": 1, "status": "CERTIFIED"}],
        },
    )

    snapshot = build_ui_snapshot(settings)
    active = snapshot["active_trading"]

    assert active["generated_at"] == "2026-08-01T19:15:00+00:00"
    assert active["scan_interval_minutes"] == 15
    assert active["scan_poll_seconds"] == 30
    assert active["scan_maximum_rows"] == 1_500
    assert active["markets_scanned"] == ["BTC-EUR", "ETH-EUR"]
    assert snapshot["crypto_maturity"]["current_level"] == "INTERMEDIATE"
