from __future__ import annotations

from pathlib import Path

from data import data_loader
from research.seven_year import _is_canonical_ohlcv_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_macos_safe_windows_process_abstraction_exists() -> None:
    assert callable(data_loader._is_windows)


def test_active_tests_do_not_mutate_global_os_name() -> None:
    offenders = []
    for path in (ROOT / "tests").rglob("test*.py"):
        text = path.read_text(encoding="utf-8")
        needle = "data.data_loader.os" + '.name", "nt"'
        if needle in text and "monkeypatch.setattr" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_feature_frame_is_not_historical_ohlcv_dataset() -> None:
    assert not _is_canonical_ohlcv_dataset(Path("BTC-EUR_features.parquet"))
    assert _is_canonical_ohlcv_dataset(Path("BTC-EUR_1h.parquet"))
    assert _is_canonical_ohlcv_dataset(Path("ETH-EUR_1d.csv"))
