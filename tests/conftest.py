from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from research.features import FeaturePipeline


def make_ohlcv(
    rows: int = 700,
    *,
    seed: int = 42,
    market: str = "BTC-EUR",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=rows, freq="1h", tz="UTC")
    drift = np.linspace(0.0, 0.22, rows)
    cycle = 0.035 * np.sin(np.arange(rows) / 16.0)
    noise = rng.normal(0.0, 0.006, rows).cumsum()
    close = 20_000.0 * np.exp(drift + cycle + noise)
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0, 0.001, rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.008, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.008, rows))
    volume = rng.lognormal(5.0, 0.35, rows)
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    frame.index.name = "timestamp"
    frame.attrs["market"] = market
    return frame


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return FeaturePipeline().build(ohlcv, market="BTC-EUR")


@pytest.fixture
def isolated_settings(tmp_path) -> Settings:
    return Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    )
