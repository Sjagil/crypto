from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PathSettings, Settings
from core.market_intelligence import (
    build_coin_ranking,
    inspect_coin_ranking,
    inspect_token_fundamentals,
    refresh_token_fundamentals,
)


def _settings(
    isolated_settings: Settings,
    tmp_path: Path,
) -> Settings:
    return isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _universe(settings: Settings) -> None:
    path = settings.paths.output_dir / "universe" / "top50_current.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_snapshot_hash": "snapshot",
                "source_collected_at": "2026-07-30T00:00:00Z",
                "rows": [
                    {
                        "rank": 1,
                        "symbol": "BTC",
                        "name": "Bitcoin",
                        "eur_spot_market": "BTC-EUR",
                        "market_cap": 1_000_000_000_000,
                        "circulating_supply": 20_000_000,
                        "volume_24h": 20_000_000_000,
                        "venue_availability": True,
                        "research_eligibility": "RESEARCH_ELIGIBLE",
                        "execution_eligibility": "LIVE_ELIGIBLE",
                        "shariah_status": "ALLOWED",
                        "stablecoin": False,
                        "wrapped": False,
                        "leveraged_token": False,
                        "staking_derivative": False,
                    },
                    {
                        "rank": 2,
                        "symbol": "ETH",
                        "name": "Ethereum",
                        "eur_spot_market": "ETH-EUR",
                        "market_cap": 300_000_000_000,
                        "circulating_supply": 120_000_000,
                        "volume_24h": 10_000_000_000,
                        "venue_availability": True,
                        "research_eligibility": "RESEARCH_ELIGIBLE",
                        "execution_eligibility": "LIVE_ELIGIBLE",
                        "shariah_status": "ALLOWED",
                        "stablecoin": False,
                        "wrapped": False,
                        "leveraged_token": False,
                        "staking_derivative": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _closed_data(settings: Settings, market: str, slope: float) -> None:
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(1, unit="h")
    index = pd.date_range(end=end, periods=900, freq="h")
    close = 100.0 + np.arange(len(index)) * slope
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        }
    )
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        settings.paths.processed_data_dir / f"{market}_1h.parquet",
        index=False,
    )


def test_point_in_time_ranking_and_tokenomics_are_transparent_and_orderless(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _universe(settings)
    _closed_data(settings, "BTC-EUR", 0.15)
    _closed_data(settings, "ETH-EUR", 0.05)

    tokenomics = refresh_token_fundamentals(settings)
    ranking = build_coin_ranking(settings)

    assert tokenomics["asset_count"] == 2
    assert tokenomics["review_required_count"] == 2
    assert tokenomics["execution_review_required_count"] == 2
    assert tokenomics["live_execution_eligible_count"] == 0
    assert tokenomics["assets"][0]["venue_execution_eligibility"] == (
        "LIVE_ELIGIBLE"
    )
    assert tokenomics["assets"][0]["execution_eligibility"] == (
        "REVIEW_REQUIRED"
    )
    assert tokenomics["missing_data_interpreted_as_positive"] is False
    assert ranking["row_count"] == 2
    assert ranking["closed_candles_only"] is True
    assert ranking["transparent_subscores"] is True
    assert ranking["orders_generated"] == 0
    assert ranking["orders_submitted"] == 0
    assert ranking["rows"][0]["symbol"] == "BTC"
    assert ranking["venue_execution_eligible_count"] == 2
    assert ranking["live_execution_eligible_count"] == 0
    assert ranking["rows"][0]["venue_execution_eligible"] is True
    assert ranking["rows"][0]["live_execution_eligible"] is False
    assert (
        ranking["rows"][0]["live_execution_eligibility_reason"]
        == "TOKEN_FUNDAMENTALS_REVIEW_REQUIRED"
    )
    assert set(ranking["rows"][0]["subscores"]) >= {
        "liquidity",
        "momentum_7d",
        "trend_quality",
        "data_quality",
        "tokenomics_quality",
    }
    assert inspect_coin_ranking(settings, "BTC")["status"] == "FOUND"
    assert inspect_token_fundamentals(settings, "ETH-EUR")["status"] == "FOUND"
    assert (
        settings.paths.output_dir / "ranking" / "history.jsonl"
    ).is_file()
