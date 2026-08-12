from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_crypto_campaign import _load_frames


def _write_frame(path: Path, index: pd.DatetimeIndex) -> None:
    close = np.linspace(100.0, 150.0, len(index))
    pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    ).to_parquet(path)


def _write_column_timestamp_frame(path: Path, index: pd.DatetimeIndex) -> None:
    close = np.linspace(100.0, 150.0, len(index))
    pd.DataFrame(
        {
            "timestamp": index,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        }
    ).to_parquet(path)


def test_adaptive_panel_preserves_pre_listing_history_without_imputation(
    tmp_path: Path,
) -> None:
    benchmark_index = pd.date_range(
        "2022-01-01",
        periods=2_000,
        freq="4h",
        tz="UTC",
    )
    young_index = benchmark_index[-1_300:]
    _write_frame(tmp_path / "BTC-EUR_4h.parquet", benchmark_index)
    _write_frame(tmp_path / "TAO-EUR_4h.parquet", young_index)

    frames, hashes = _load_frames(
        tmp_path,
        ("BTC-EUR", "TAO-EUR"),
        "4h",
    )

    assert len(frames["BTC-EUR"]) == 2_000
    assert len(frames["TAO-EUR"]) == 1_300
    assert frames["TAO-EUR"].index[0] > frames["BTC-EUR"].index[0]
    assert "__point_in_time_panel__" in hashes
    assert "__common_segment__" not in hashes


def test_adaptive_panel_uses_timestamp_column_instead_of_range_index(
    tmp_path: Path,
) -> None:
    index = pd.date_range("2022-01-01", periods=1_300, freq="4h", tz="UTC")
    _write_column_timestamp_frame(tmp_path / "BTC-EUR_4h.parquet", index)

    frames, _ = _load_frames(tmp_path, ("BTC-EUR",), "4h")

    assert frames["BTC-EUR"].index.equals(index)
    assert "timestamp" not in frames["BTC-EUR"].columns
    assert frames["BTC-EUR"].index[0].year == 2022
