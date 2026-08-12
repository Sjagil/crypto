from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PathSettings, Settings
from core.market_mechanics import (
    load_orderflow_15m_context,
    load_orderflow_context,
)
from research.gex_orderflow_strategies import (
    evaluate_market_mechanics_strategy,
    market_mechanics_strategy_specs,
)


def _frame(frequency: str, rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-07-01", periods=rows, freq=frequency, tz="UTC")
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 100.0),
            "atr_14": np.full(rows, 1.0),
            "relative_volume_20": np.full(rows, 1.5),
            "btc_relative_momentum_20": np.full(rows, 0.01),
        },
        index=index,
    )


def _mechanics(*, flow_status: str = "READY") -> dict:
    return {
        "gex_scope": "BTC_MARKET_REGIME_PROXY_FOR_ALTCOIN",
        "gex": {
            "fresh": True,
            "regime": "POSITIVE_GEX",
            "normalized_signed_gex": 0.2,
        },
        "orderflow": {
            "status": flow_status,
            "spot_cvd_robust_zscore": 1.0,
            "orderbook_imbalance_top_10": 0.1,
            "bullish_absorption_score": 0.8,
            "spread_bps": 1.0,
            "horizons": {
                "1h": {
                    "cvd_slope_base_per_hour": 2.0,
                    "ofi_normalized_mean": 0.2,
                }
            },
        },
    }


def _settings(tmp_path: Path) -> Settings:
    base = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    )
    return base.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_orderflow_context_reports_exact_market_coverage(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    directory = settings.paths.context_data_dir / "microstructure_hourly"
    directory.mkdir(parents=True)
    hour_start = datetime(2026, 8, 1, 18, tzinfo=UTC)
    payload = {
        "hour_start": hour_start.isoformat(),
        "hour_end": (hour_start + timedelta(hours=1)).isoformat(),
        "status": "PARTIAL",
        "markets": [
            {
                "market": "BTC-EUR",
                "status": "COMPLETE",
                "reason_codes": [],
                "spot_base_volume": 10.0,
                "trade_delta_base": 1.0,
            },
            {
                "market": "ETH-EUR",
                "status": "DATA_GAP",
                "reason_codes": ["RECORDER_STARTED_MID_HOUR"],
            },
        ],
    }
    (directory / "20260801T180000Z.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    context = load_orderflow_context(
        settings,
        markets=("BTC-EUR", "ETH-EUR", "SOL-EUR"),
        now=datetime(2026, 8, 1, 19, 30, tzinfo=UTC),
    )

    assert context["status"] == "PARTIAL"
    assert context["requested_market_count"] == 3
    assert context["ready_market_count"] == 1
    assert context["data_gap_market_count"] == 1
    assert context["data_pending_market_count"] == 1
    assert context["ready_fraction"] == 1 / 3
    assert context["reason_counts"] == {
        "NO_PROSPECTIVE_ORDERFLOW": 1,
        "RECORDER_STARTED_MID_HOUR": 1,
    }


def test_orderflow_context_distinguishes_all_gap_from_all_pending(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    pending = load_orderflow_context(
        settings,
        markets=("BTC-EUR",),
        now=datetime(2026, 8, 1, 19, 30, tzinfo=UTC),
    )
    assert pending["status"] == "DATA_PENDING"
    assert pending["data_pending_market_count"] == 1


def test_15m_orderflow_context_requires_sealed_complete_bucket(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    directory = settings.paths.context_data_dir / "microstructure_15m"
    directory.mkdir(parents=True)
    interval_start = datetime(2026, 8, 1, 18, 15, tzinfo=UTC)
    payload = {
        "hour_start": interval_start.isoformat(),
        "hour_end": (interval_start + timedelta(minutes=15)).isoformat(),
        "interval_minutes": 15,
        "status": "COMPLETE",
        "markets": [
            {
                "market": "BTC-EUR",
                "status": "COMPLETE",
                "reason_codes": [],
                "spot_base_volume": 10.0,
                "trade_delta_base": 1.0,
                "order_flow_imbalance_normalized": 0.2,
                "spread_bps": 1.0,
            }
        ],
    }
    (directory / "20260801T181500Z.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    context = load_orderflow_15m_context(
        settings,
        markets=("BTC-EUR", "ETH-EUR"),
        now=datetime(2026, 8, 1, 18, 31, tzinfo=UTC),
    )
    assert context["status"] == "PARTIAL"
    assert context["ready_market_count"] == 1
    assert context["data_pending_market_count"] == 1
    assert context["markets"]["BTC-EUR"]["horizons"]["15m"][
        "trade_delta_base"
    ] == 1.0


def test_flow_strategy_requires_prospective_orderflow() -> None:
    spec = next(
        row
        for row in market_mechanics_strategy_specs()
        if row.strategy_id == "FLOW_TREND_PULLBACK_4H2H1H"
    )
    result = evaluate_market_mechanics_strategy(
        spec,
        market="TAO-EUR",
        one_hour=_frame("1h"),
        two_hour=_frame("2h"),
        four_hour=_frame("4h"),
        mechanics=_mechanics(flow_status="DATA_GAP"),
    )
    assert result["status"] == "DATA_PENDING"
    assert not result["actionable"]
    assert "PROSPECTIVE_ORDERFLOW_NOT_READY" in result["blockers"]


def test_4h_2h_1h_flow_strategy_emits_bounded_trade_plan() -> None:
    spec = next(
        row
        for row in market_mechanics_strategy_specs()
        if row.strategy_id == "FLOW_TREND_PULLBACK_4H2H1H"
    )
    result = evaluate_market_mechanics_strategy(
        spec,
        market="TAO-EUR",
        one_hour=_frame("1h"),
        two_hour=_frame("2h"),
        four_hour=_frame("4h"),
        mechanics=_mechanics(),
    )
    assert result["status"] == "ACTIONABLE"
    assert result["score"] >= 0.65
    assert result["stop"] < result["entry"] < result["target_1"]
    assert result["target_1"] < result["target_2"]
    assert result["live_authority_granted"] is False


def test_4h_2h_15m_flow_strategy_uses_closed_micro_entry_frame() -> None:
    spec = next(
        row
        for row in market_mechanics_strategy_specs()
        if row.strategy_id == "FLOW_TREND_PULLBACK_4H2H15M"
    )
    mechanics = _mechanics()
    mechanics["orderflow_15m"] = {
        **mechanics["orderflow"],
        "horizons": {
            "15m": {
                "cvd_slope_base_per_hour": 2.0,
                "ofi_normalized_mean": 0.2,
            }
        },
    }
    result = evaluate_market_mechanics_strategy(
        spec,
        market="TAO-EUR",
        fifteen_minute=_frame("15min"),
        one_hour=_frame("1h"),
        two_hour=_frame("2h"),
        four_hour=_frame("4h"),
        mechanics=mechanics,
    )
    assert result["status"] == "ACTIONABLE"
    assert result["entry_timeframe"] == "15m"
    assert result["orderflow_confirmation_horizon"] == "15m"
    assert result["stop"] < result["entry"] < result["target_1"]
    assert result["closed_candle_only"] is True


def test_15m_strategy_does_not_fall_back_when_live_15m_bucket_is_gap() -> None:
    spec = next(
        row
        for row in market_mechanics_strategy_specs()
        if row.strategy_id == "FLOW_TREND_PULLBACK_4H2H15M"
    )
    mechanics = _mechanics()
    mechanics["orderflow_15m"] = {
        "status": "DATA_GAP",
        "horizons": {},
    }
    result = evaluate_market_mechanics_strategy(
        spec,
        market="TAO-EUR",
        fifteen_minute=_frame("15min"),
        one_hour=_frame("1h"),
        two_hour=_frame("2h"),
        four_hour=_frame("4h"),
        mechanics=mechanics,
    )
    assert result["status"] == "DATA_PENDING"
    assert "PROSPECTIVE_ORDERFLOW_NOT_READY" in result["blockers"]
