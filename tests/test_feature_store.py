from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.feature_store import (
    FEATURE_NAMES,
    STRICT_PORTFOLIO_MARKETS,
    build_and_persist_feature_store,
    build_feature_tensors,
    persist_feature_store,
)
from utils.common import read_json, sha256_file


def _frame(
    *,
    start: str,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=rows, freq="1D", tz="UTC")
    log_returns = rng.normal(0.0005, 0.025, rows)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_price = close * np.exp(rng.normal(0.0, 0.005, rows))
    high = np.maximum(open_price, close) * (
        1.0 + rng.uniform(0.001, 0.02, rows)
    )
    low = np.minimum(open_price, close) * (
        1.0 - rng.uniform(0.001, 0.02, rows)
    )
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(10.0, 0.5, rows),
        },
        index=index,
    )


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "BTC-EUR": _frame(start="2020-01-01", rows=360, seed=1),
        "ETH-EUR": _frame(start="2020-01-01", rows=360, seed=2),
        "SOL-EUR": _frame(start="2020-03-01", rows=300, seed=3),
        "LINK-EUR": _frame(start="2020-01-20", rows=341, seed=4),
    }


def test_feature_tensor_is_masked_stationary_and_strict():
    bundle = build_feature_tensors(_frames())

    assert bundle.assets == STRICT_PORTFOLIO_MARKETS
    assert bundle.feature_names == FEATURE_NAMES
    assert bundle.features.shape == (360, 4, len(FEATURE_NAMES))
    assert bundle.feature_mask.shape == (360, 4)
    assert bundle.targets.shape == (360, 4)
    assert np.isfinite(bundle.features).all()
    assert np.isfinite(bundle.targets).all()
    assert np.all(bundle.features[~bundle.feature_mask] == 0.0)
    assert np.all(bundle.targets[~bundle.target_mask] == 0.0)
    assert not bundle.feature_mask[:199, 0].any()
    assert not bundle.feature_mask[:259, 2].any()
    assert bundle.feature_mask[-1].all()
    assert not bundle.target_mask[-1].any()
    assert bundle.manifest["causality"]["full_sample_normalization"] is False
    assert bundle.manifest["causality"]["targets_physically_separate"] is True
    assert bundle.manifest["orders_generated"] == 0
    assert bundle.manifest["live_ready"] is False


def test_future_price_shock_cannot_change_prior_features():
    frames = _frames()
    baseline = build_feature_tensors(frames)
    shocked = {market: frame.copy() for market, frame in frames.items()}
    last = shocked["BTC-EUR"].index[-1]
    shocked["BTC-EUR"].loc[last, "open"] *= 4.0
    shocked["BTC-EUR"].loc[last, "close"] *= 5.0
    shocked["BTC-EUR"].loc[last, "high"] = (
        shocked["BTC-EUR"].loc[last, ["open", "close"]].max() * 1.01
    )
    shocked["BTC-EUR"].loc[last, "low"] *= 0.99
    changed = build_feature_tensors(shocked)

    np.testing.assert_array_equal(
        baseline.features[:-1],
        changed.features[:-1],
    )
    np.testing.assert_array_equal(
        baseline.feature_mask[:-1],
        changed.feature_mask[:-1],
    )
    np.testing.assert_array_equal(
        baseline.targets[:-2],
        changed.targets[:-2],
    )


def test_target_available_at_uses_next_own_candle_across_gap():
    frames = _frames()
    market = "SOL-EUR"
    gap_at = frames[market].index[250]
    prior_at = frames[market].index[249]
    next_at = frames[market].index[251]
    frames[market] = frames[market].drop(index=gap_at)
    bundle = build_feature_tensors(frames)
    timeline = pd.to_datetime(bundle.timestamps_ns, utc=True)
    prior_position = int(np.flatnonzero(timeline == prior_at)[0])
    asset_position = bundle.assets.index(market)

    assert bundle.target_mask[prior_position, asset_position]
    assert (
        bundle.target_available_at_ns[prior_position, asset_position]
        == next_at.value
    )


def test_feature_store_is_content_addressed_and_checksum_verified(tmp_path):
    bundle = build_feature_tensors(
        _frames(),
        source_hashes={market: market for market in STRICT_PORTFOLIO_MARKETS},
    )
    first = persist_feature_store(bundle, tmp_path)
    second = persist_feature_store(bundle, tmp_path)

    assert first["status"] == "CREATED"
    assert second["status"] == "REUSED"
    assert first["dataset_id"] == second["dataset_id"]
    manifest = read_json(Path(second["manifest"]))
    assert manifest["tensor_sha256"] == sha256_file(second["tensor"])
    with np.load(second["tensor"], allow_pickle=False) as archive:
        assert set(archive.files) == {
            "features",
            "feature_mask",
            "targets",
            "target_mask",
            "timestamps_ns",
            "target_available_at_ns",
            "assets",
            "feature_names",
        }
        assert archive["features"].shape == bundle.features.shape


def test_unknown_or_missing_assets_fail_closed():
    frames = _frames()
    frames["DOGE-EUR"] = frames.pop("LINK-EUR")
    with pytest.raises(ValueError, match="exact strict universe"):
        build_feature_tensors(frames)


def test_build_from_parquet_records_source_hashes(tmp_path):
    source = tmp_path / "normalized"
    source.mkdir()
    paths: dict[str, Path] = {}
    for market, frame in _frames().items():
        path = source / f"{market}_1d.parquet"
        frame.to_parquet(path)
        paths[market] = path

    result = build_and_persist_feature_store(
        paths,
        tmp_path / "features",
    )
    manifest = read_json(result["manifest"])

    assert result["status"] == "CREATED"
    assert manifest["source_hashes"] == {
        market: sha256_file(path)
        for market, path in sorted(paths.items())
    }
