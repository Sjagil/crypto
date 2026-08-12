from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import PathSettings, Settings
from research.cross_asset_seesaw import (
    SeesawParameters,
    chronological_metrics,
    episode_metrics,
    simulate_portfolio_episodes,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(
        env_file=tmp_path / "missing.env",
        create_directories=False,
    )
    return settings.model_copy(update={"paths": PathSettings(project_root=tmp_path)})


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    leaders = {}
    for market in ("BTC-EUR", "ETH-EUR", "XRP-EUR", "LTC-EUR", "BCH-EUR"):
        close = [100.0] * len(index)
        close[4] = 97.0
        leaders[market] = pd.DataFrame(
            {"open": close, "close": close},
            index=index,
        )
    targets = {}
    for offset, market in enumerate(
        (
            "SOL-EUR",
            "LINK-EUR",
            "ADA-EUR",
            "DOGE-EUR",
            "TRX-EUR",
            "HYPE-EUR",
            "AVAX-EUR",
            "BNB-EUR",
        )
    ):
        opened = [100.0 + offset] * len(index)
        opened[7] = opened[5] * 1.02
        targets[market] = pd.DataFrame(
            {"open": opened, "close": opened},
            index=index,
        )
    return {**leaders, **targets}


def test_seesaw_uses_closed_leader_then_next_open_and_clusters_targets() -> None:
    parameters = SeesawParameters(
        direction="NEGATIVE_SEESAW",
        leader_threshold=0.02,
        holding_bars=2,
    )

    episodes = simulate_portfolio_episodes(
        _frames(),
        parameters,
        round_trip_cost=0.01,
    )

    assert len(episodes) == 1
    assert episodes[0]["signal_at"] == "2026-01-01T04:00:00+00:00"
    assert episodes[0]["entry_at"] == "2026-01-01T05:00:00+00:00"
    assert episodes[0]["exit_at"] == "2026-01-01T07:00:00+00:00"
    assert episodes[0]["target_count"] == 8
    assert abs(episodes[0]["portfolio_return"] - 0.01) < 1e-12


def test_episode_statistics_keep_chronological_splits() -> None:
    episodes = [
        {
            "entry_at": f"2026-01-{day:02d}T00:00:00+00:00",
            "portfolio_return": value,
        }
        for day, value in enumerate(
            (0.01, -0.02, 0.03, 0.01, 0.02, -0.01, 0.04, 0.01, -0.01, 0.02),
            start=1,
        )
    ]

    metrics = episode_metrics(episodes)
    periods = chronological_metrics(episodes)

    assert metrics["episode_count"] == 10
    assert abs(float(metrics["profit_factor"]) - 3.5) < 1e-12
    assert periods["development"]["episode_count"] == 6
    assert periods["validation"]["episode_count"] == 2
    assert periods["holdout"]["episode_count"] == 2


def test_seesaw_parameters_reject_overlapping_universes(tmp_path: Path) -> None:
    _settings(tmp_path)
    try:
        SeesawParameters(
            direction="NEGATIVE_SEESAW",
            leader_threshold=0.02,
            holding_bars=24,
            leader_markets=("BTC-EUR",),
            target_markets=("BTC-EUR", "ETH-EUR"),
        )
    except ValueError as exc:
        assert "must not overlap" in str(exc)
    else:
        raise AssertionError("overlapping leader and target universes were accepted")
