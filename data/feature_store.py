"""Causal, point-in-time tensor storage for portfolio research.

The store is intentionally research-only. Features are derived from completed
daily candles, targets are physically separate, and listing gaps require an
explicit mask. No global/full-sample normalization is applied.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.common import (
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
    utc_iso,
)

FEATURE_STORE_SCHEMA_VERSION = "portfolio_daily_causal_v1"
STRICT_PORTFOLIO_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
)
FEATURE_NAMES = (
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "log_return_60",
    "log_return_90",
    "realized_volatility_20",
    "realized_volatility_60",
    "donchian_upper_distance_20",
    "donchian_upper_distance_55",
    "donchian_lower_distance_10",
    "donchian_lower_distance_20",
    "ema_distance_50",
    "ema_distance_200",
    "intraday_log_range",
    "close_to_open_log_return",
    "volume_log_change_1",
    "volume_zscore_20",
    "btc_relative_momentum_20",
    "btc_relative_momentum_90",
    "cross_sectional_momentum_rank_90",
    "market_breadth_positive_momentum_90",
    "btc_ema200_distance",
)


@dataclass(frozen=True, slots=True)
class FeatureStorePolicy:
    """Immutable causal and point-in-time tensor policy."""

    markets: tuple[str, ...] = STRICT_PORTFOLIO_MARKETS
    minimum_history_observations: int = 200
    frequency: str = "1d"
    schema_version: str = FEATURE_STORE_SCHEMA_VERSION
    clip_absolute_value: float = 20.0

    def __post_init__(self) -> None:
        if tuple(self.markets) != STRICT_PORTFOLIO_MARKETS:
            raise ValueError("feature store markets must match strict allowlist")
        if self.minimum_history_observations < 200:
            raise ValueError(
                "minimum_history_observations must preserve EMA200 warmup"
            )
        if self.frequency != "1d":
            raise ValueError("portfolio feature store supports 1d only")
        if self.clip_absolute_value <= 0:
            raise ValueError("clip_absolute_value must be positive")


@dataclass(frozen=True, slots=True)
class FeatureTensorBundle:
    """In-memory causal tensors plus their machine-readable manifest."""

    features: np.ndarray
    feature_mask: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    timestamps_ns: np.ndarray
    target_available_at_ns: np.ndarray
    assets: tuple[str, ...]
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]


def _validate_frame(frame: pd.DataFrame, *, market: str) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{market} is missing OHLCV columns: {missing}")
    selected = frame.loc[:, required].copy()
    if not isinstance(selected.index, pd.DatetimeIndex):
        raise TypeError(f"{market} index must be a DatetimeIndex")
    if selected.index.tz is None:
        selected.index = selected.index.tz_localize("UTC")
    else:
        selected.index = selected.index.tz_convert("UTC")
    selected = selected.sort_index()
    if selected.index.has_duplicates:
        raise ValueError(f"{market} contains duplicate timestamps")
    numeric = selected.astype(float)
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    positive = (numeric.loc[:, required] > 0).all(axis=1).to_numpy()
    if not bool(np.all(finite & positive)):
        raise ValueError(f"{market} contains invalid OHLCV rows")
    if not bool(
        (
            (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1))
            & (
                numeric["low"]
                <= numeric[["open", "close", "high"]].min(axis=1)
            )
        ).all()
    ):
        raise ValueError(f"{market} contains inconsistent OHLC values")
    return numeric


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator
    return np.log(ratio.where((ratio > 0) & np.isfinite(ratio)))


def _asset_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    log_close = np.log(close)
    log_return = log_close.diff()
    log_volume = np.log(volume)
    features = pd.DataFrame(index=frame.index)
    for horizon in (1, 5, 20, 60, 90):
        features[f"log_return_{horizon}"] = log_close.diff(horizon)
    for horizon in (20, 60):
        features[f"realized_volatility_{horizon}"] = (
            log_return.rolling(horizon, min_periods=horizon).std(ddof=1)
            * np.sqrt(365.0)
        )
    for horizon in (20, 55):
        prior_high = high.rolling(
            horizon,
            min_periods=horizon,
        ).max().shift(1)
        features[f"donchian_upper_distance_{horizon}"] = _log_ratio(
            close,
            prior_high,
        )
    for horizon in (10, 20):
        prior_low = low.rolling(
            horizon,
            min_periods=horizon,
        ).min().shift(1)
        features[f"donchian_lower_distance_{horizon}"] = _log_ratio(
            close,
            prior_low,
        )
    for horizon in (50, 200):
        ema = close.ewm(
            span=horizon,
            adjust=False,
            min_periods=horizon,
        ).mean()
        features[f"ema_distance_{horizon}"] = _log_ratio(close, ema)
    features["intraday_log_range"] = _log_ratio(high, low)
    features["close_to_open_log_return"] = _log_ratio(
        close,
        frame["open"],
    )
    features["volume_log_change_1"] = log_volume.diff()
    prior_volume_mean = log_volume.rolling(
        20,
        min_periods=20,
    ).mean().shift(1)
    prior_volume_std = log_volume.rolling(
        20,
        min_periods=20,
    ).std(ddof=1).shift(1)
    features["volume_zscore_20"] = (
        (log_volume - prior_volume_mean) / prior_volume_std.replace(0, np.nan)
    )
    return features


def _aligned_feature_panels(
    frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeatureStorePolicy,
) -> tuple[
    pd.DatetimeIndex,
    dict[str, pd.DataFrame],
    dict[str, pd.Series],
    dict[str, pd.Series],
]:
    received = {str(market) for market in frames}
    expected = set(policy.markets)
    if received != expected:
        raise ValueError(
            "feature store requires exact strict universe; "
            f"missing={sorted(expected - received)}, "
            f"unknown={sorted(received - expected)}"
        )
    validated = {
        market: _validate_frame(frames[market], market=market)
        for market in policy.markets
    }
    timeline = pd.DatetimeIndex(
        sorted(
            set().union(
                *(set(frame.index) for frame in validated.values())
            )
        ),
        tz="UTC",
        name="timestamp",
    )
    features = {
        market: _asset_features(frame).reindex(timeline)
        for market, frame in validated.items()
    }
    closes = {
        market: frame["close"].reindex(timeline)
        for market, frame in validated.items()
    }
    next_returns = {
        market: np.log(frame["close"].shift(-1) / frame["close"]).reindex(
            timeline
        )
        for market, frame in validated.items()
    }
    btc_20 = features["BTC-EUR"]["log_return_20"]
    btc_90 = features["BTC-EUR"]["log_return_90"]
    btc_ema = features["BTC-EUR"]["ema_distance_200"]
    momentum_90 = pd.concat(
        {
            market: panel["log_return_90"]
            for market, panel in features.items()
        },
        axis=1,
    )
    ranks = momentum_90.rank(
        axis=1,
        method="average",
        pct=True,
        na_option="keep",
    )
    breadth = (momentum_90 > 0).sum(axis=1) / momentum_90.notna().sum(
        axis=1
    ).replace(0, np.nan)
    for market, panel in features.items():
        panel["btc_relative_momentum_20"] = (
            panel["log_return_20"] - btc_20
        )
        panel["btc_relative_momentum_90"] = (
            panel["log_return_90"] - btc_90
        )
        panel["cross_sectional_momentum_rank_90"] = ranks[market]
        panel["market_breadth_positive_momentum_90"] = breadth
        panel["btc_ema200_distance"] = btc_ema
        features[market] = panel.loc[:, FEATURE_NAMES]
    return timeline, features, closes, next_returns


def build_feature_tensors(
    frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeatureStorePolicy | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> FeatureTensorBundle:
    """Build deterministic causal tensors from strict daily OHLCV frames."""

    selected_policy = policy or FeatureStorePolicy()
    timeline, panels, closes, next_returns = _aligned_feature_panels(
        frames,
        policy=selected_policy,
    )
    time_count = len(timeline)
    asset_count = len(selected_policy.markets)
    feature_count = len(FEATURE_NAMES)
    tensor = np.zeros(
        (time_count, asset_count, feature_count),
        dtype=np.float32,
    )
    feature_mask = np.zeros((time_count, asset_count), dtype=bool)
    targets = np.zeros((time_count, asset_count), dtype=np.float32)
    target_mask = np.zeros((time_count, asset_count), dtype=bool)
    available_at = np.full(
        (time_count, asset_count),
        np.iinfo(np.int64).min,
        dtype=np.int64,
    )
    per_asset: dict[str, Any] = {}
    timestamps_ns = timeline.asi8.astype(np.int64, copy=True)

    for asset_index, market in enumerate(selected_policy.markets):
        panel = panels[market].replace([np.inf, -np.inf], np.nan)
        raw = panel.to_numpy(dtype=float)
        valid = np.isfinite(raw).all(axis=1)
        own_history = closes[market].notna().cumsum()
        valid &= (
            own_history.to_numpy()
            >= selected_policy.minimum_history_observations
        )
        clipped = np.clip(
            np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0),
            -selected_policy.clip_absolute_value,
            selected_policy.clip_absolute_value,
        )
        tensor[:, asset_index, :] = clipped.astype(np.float32)
        tensor[~valid, asset_index, :] = 0.0
        feature_mask[:, asset_index] = valid

        raw_target = next_returns[market].to_numpy(dtype=float)
        target_valid = valid & np.isfinite(raw_target)
        targets[:, asset_index] = np.nan_to_num(
            raw_target,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        targets[~target_valid, asset_index] = 0.0
        target_mask[:, asset_index] = target_valid
        own_positions = np.flatnonzero(
            closes[market].notna().to_numpy()
        )
        own_next_position = {
            int(current): int(future)
            for current, future in zip(
                own_positions[:-1],
                own_positions[1:],
                strict=True,
            )
        }
        for target_position in np.flatnonzero(target_valid):
            next_position = own_next_position.get(int(target_position))
            if next_position is not None:
                available_at[target_position, asset_index] = timestamps_ns[
                    next_position
                ]
        per_asset[market] = {
            "source_rows": int(closes[market].notna().sum()),
            "feature_rows": int(valid.sum()),
            "target_rows": int(target_valid.sum()),
            "first_feature_at": (
                timeline[np.flatnonzero(valid)[0]].isoformat()
                if valid.any()
                else None
            ),
            "last_feature_at": (
                timeline[np.flatnonzero(valid)[-1]].isoformat()
                if valid.any()
                else None
            ),
        }

    feature_hash = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    feature_mask_hash = hashlib.sha256(
        feature_mask.tobytes(order="C")
    ).hexdigest()
    target_hash = hashlib.sha256(targets.tobytes(order="C")).hexdigest()
    target_mask_hash = hashlib.sha256(
        target_mask.tobytes(order="C")
    ).hexdigest()
    target_available_at_hash = hashlib.sha256(
        available_at.tobytes(order="C")
    ).hexdigest()
    lineage = {
        "schema_version": selected_policy.schema_version,
        "policy": asdict(selected_policy),
        "assets": list(selected_policy.markets),
        "feature_names": list(FEATURE_NAMES),
        "source_hashes": dict(sorted((source_hashes or {}).items())),
        "timeline_start": timeline[0].isoformat() if len(timeline) else None,
        "timeline_end": timeline[-1].isoformat() if len(timeline) else None,
        "timestamps_hash": hashlib.sha256(
            timestamps_ns.tobytes(order="C")
        ).hexdigest(),
        "feature_tensor_hash": feature_hash,
        "feature_mask_hash": feature_mask_hash,
        "target_tensor_hash": target_hash,
        "target_mask_hash": target_mask_hash,
        "target_available_at_hash": target_available_at_hash,
    }
    dataset_id = stable_hash(lineage, length=64)
    manifest = {
        **lineage,
        "dataset_id": dataset_id,
        "generated_at": utc_iso(),
        "frequency": "1d",
        "shapes": {
            "features": list(tensor.shape),
            "feature_mask": list(feature_mask.shape),
            "targets": list(targets.shape),
            "target_mask": list(target_mask.shape),
            "timestamps_ns": list(timestamps_ns.shape),
            "target_available_at_ns": list(available_at.shape),
        },
        "dtypes": {
            "features": str(tensor.dtype),
            "feature_mask": str(feature_mask.dtype),
            "targets": str(targets.dtype),
            "target_mask": str(target_mask.dtype),
            "timestamps_ns": str(timestamps_ns.dtype),
            "target_available_at_ns": str(available_at.dtype),
        },
        "per_asset": per_asset,
        "causality": {
            "closed_candles_only": True,
            "features_known_at": "SOURCE_DAILY_CLOSE",
            "execution_earliest_at": "NEXT_AVAILABLE_OPEN",
            "rolling_channels_shifted": True,
            "full_sample_normalization": False,
            "point_in_time_listing_mask": True,
            "targets_physically_separate": True,
            "target_known_at": "NEXT_SOURCE_DAILY_CLOSE",
            "masked_tensor_fill_value": 0.0,
        },
        "research_only": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    return FeatureTensorBundle(
        features=tensor,
        feature_mask=feature_mask,
        targets=targets,
        target_mask=target_mask,
        timestamps_ns=timestamps_ns,
        target_available_at_ns=available_at,
        assets=selected_policy.markets,
        feature_names=FEATURE_NAMES,
        manifest=manifest,
    )


def _write_npz(path: Path, bundle: FeatureTensorBundle) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                features=bundle.features,
                feature_mask=bundle.feature_mask,
                targets=bundle.targets,
                target_mask=bundle.target_mask,
                timestamps_ns=bundle.timestamps_ns,
                target_available_at_ns=bundle.target_available_at_ns,
                assets=np.asarray(bundle.assets, dtype="U16"),
                feature_names=np.asarray(
                    bundle.feature_names,
                    dtype="U64",
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def persist_feature_store(
    bundle: FeatureTensorBundle,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Persist a content-addressed snapshot plus atomic latest aliases."""

    root = Path(output_dir)
    snapshots = root / "snapshots"
    dataset_id = str(bundle.manifest["dataset_id"])
    snapshot_npz = snapshots / f"{dataset_id}.npz"
    snapshot_manifest = snapshots / f"{dataset_id}.manifest.json"
    reused = snapshot_npz.is_file() and snapshot_manifest.is_file()
    if not reused:
        _write_npz(snapshot_npz, bundle)
        snapshot_payload = {
            **bundle.manifest,
            "tensor_path": str(snapshot_npz),
            "tensor_sha256": sha256_file(snapshot_npz),
        }
        atomic_write_json(snapshot_manifest, snapshot_payload)
    else:
        snapshot_payload = read_json(snapshot_manifest)
        if snapshot_payload.get("dataset_id") != dataset_id:
            raise RuntimeError("feature store snapshot identity mismatch")
        if snapshot_payload.get("tensor_sha256") != sha256_file(
            snapshot_npz
        ):
            raise RuntimeError("feature store snapshot checksum mismatch")

    latest_npz = root / "latest.npz"
    latest_manifest = root / "latest.manifest.json"
    _write_npz(latest_npz, bundle)
    latest_payload = {
        **bundle.manifest,
        "tensor_path": str(latest_npz),
        "tensor_sha256": sha256_file(latest_npz),
        "snapshot_tensor_path": str(snapshot_npz),
        "snapshot_manifest_path": str(snapshot_manifest),
        "reused_snapshot": reused,
    }
    atomic_write_json(latest_manifest, latest_payload)
    return {
        "status": "REUSED" if reused else "CREATED",
        "dataset_id": dataset_id,
        "tensor": str(latest_npz),
        "manifest": str(latest_manifest),
        "snapshot_tensor": str(snapshot_npz),
        "snapshot_manifest": str(snapshot_manifest),
        "tensor_sha256": latest_payload["tensor_sha256"],
        "shapes": bundle.manifest["shapes"],
        "per_asset": bundle.manifest["per_asset"],
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def build_and_persist_feature_store(
    source_paths: Mapping[str, Path | str],
    output_dir: Path | str,
    *,
    policy: FeatureStorePolicy | None = None,
) -> dict[str, Any]:
    """Load strict normalized OHLCV, build tensors and persist lineage."""

    selected_policy = policy or FeatureStorePolicy()
    received = {str(market) for market in source_paths}
    if received != set(selected_policy.markets):
        raise ValueError("source paths must match exact strict universe")
    paths = {
        market: Path(source_paths[market])
        for market in selected_policy.markets
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"feature store sources missing: {missing}")
    frames = {
        market: pd.read_parquet(path)
        for market, path in paths.items()
    }
    bundle = build_feature_tensors(
        frames,
        policy=selected_policy,
        source_hashes={
            market: sha256_file(path)
            for market, path in paths.items()
        },
    )
    return persist_feature_store(bundle, output_dir)
