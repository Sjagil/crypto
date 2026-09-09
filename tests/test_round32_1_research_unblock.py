from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.downloader import CanonicalDownloader
from reporting.top_existing_strategies import (
    _campaign_candidates,
    _volume_candidates,
)


def _frame(start: datetime, rows: int = 12) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="1h", tz="UTC")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close + 0.25,
            "volume": 10.0,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


class _Provider:
    name = "bitvavo"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    async def fetch_candles(self, market, timeframe, *, start, end):
        result = self.frame.loc[
            (self.frame.index >= pd.Timestamp(start))
            & (self.frame.index <= pd.Timestamp(end))
        ].copy()
        result.attrs["market"] = market
        return result


def test_downloader_writes_real_provider_provenance(isolated_settings, tmp_path) -> None:
    paths = isolated_settings.paths.model_copy(
        update={"processed_data_dir": (tmp_path / "normalized").resolve()}
    )
    settings = isolated_settings.model_copy(update={"paths": paths})
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=12)
    frame = _frame(start)

    result = asyncio.run(
        CanonicalDownloader(settings).download_one(
            market="BTC-EUR",
            timeframe="1h",
            provider=_Provider(frame),
            start=start,
            end=end,
            resume=True,
        )
    )

    provenance_path = result.output_path.with_suffix(
        f"{result.output_path.suffix}.provenance.json"
    )
    assert provenance_path.is_file()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["source_type"] == "REAL_PROVIDER_DATA"
    assert payload["market"] == "BTC-EUR"
    assert payload["timeframe"] == "1h"
    assert payload["closed_candles_only"] is True
    assert payload["providers_used"] == ["bitvavo"]
    assert payload["data_sha256"]


def test_missing_campaign_reports_are_not_fabricated(tmp_path) -> None:
    (tmp_path / "output" / "lab" / "reports").mkdir(parents=True)
    assert _campaign_candidates(tmp_path) == []
    assert _volume_candidates(tmp_path) == []


def test_windows_pid_test_no_longer_mutates_global_os_name() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "tests" / "test_autonomous_live.py").read_text(encoding="utf-8")
    assert 'monkeypatch.setattr(os, "name", "nt")' not in text
    assert "core.autonomous_live._platform_name" in text
