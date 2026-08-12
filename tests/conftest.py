from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import EligibilityRecord, EligibilityStatus, Settings
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


@pytest.fixture
def restrictive_settings(isolated_settings: Settings) -> Settings:
    """Explicit fail-closed policy for tests that exercise review semantics."""

    markets = {
        market: EligibilityRecord(
            market=market,
            status=EligibilityStatus.ALLOWED,
            reason="TEST_ALLOWLIST",
        )
        for market in ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
    }
    markets["ICP-EUR"] = EligibilityRecord(
        market="ICP-EUR",
        status=EligibilityStatus.REVIEW_REQUIRED,
        reason="TEST_REVIEW_REQUIRED",
    )
    shariah = isolated_settings.shariah.model_copy(
        update={
            "markets": markets,
            "operator_reviewed_all_eur_spot": False,
            "operator_review_reason": None,
        }
    )
    return isolated_settings.model_copy(update={"shariah": shariah})
