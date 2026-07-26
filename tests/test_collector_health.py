from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from data.collector_health import collector_health_report
from utils.common import atomic_write_json


class FakeDatabase:
    def __init__(self, now: datetime, *, gaps: int = 0) -> None:
        self.now = now
        self.gaps = gaps

    def fetch_recent_records(self, table, *, limit):
        assert limit == 1
        return [
            {
                "payload": {
                    "timestamp": self.now.isoformat(),
                    "observed_at": self.now.isoformat(),
                    "sequence": 10 if table == "orderbook_snapshots" else None,
                }
            }
        ]

    def fetch_records(self, table):
        assert table == "provider_health"
        return [
            {
                "payload": {
                    "reconnect_count": 2,
                    "sequence_gaps": self.gaps,
                }
            }
        ]


def _settings(tmp_path):
    paths = SimpleNamespace(
        checkpoints_dir=tmp_path / "checkpoints",
        data_dir=tmp_path / "data",
    )
    paths.checkpoints_dir.mkdir()
    (paths.data_dir / "raw").mkdir(parents=True)
    (paths.data_dir / "raw" / "event.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return SimpleNamespace(
        paths=paths,
        operational=SimpleNamespace(cycle_seconds=60.0),
        market_data=SimpleNamespace(minimum_free_disk_gb=0.0),
    )


def test_collector_health_reports_required_operational_fields(
    tmp_path,
) -> None:
    now = datetime(2026, 7, 26, 17, 0, tzinfo=UTC)
    settings = _settings(tmp_path)
    atomic_write_json(
        settings.paths.checkpoints_dir
        / "operate-shadow_heartbeat.json",
        {
            "pid": 123,
            "heartbeat_at": now.isoformat(),
        },
    )
    atomic_write_json(
        settings.paths.checkpoints_dir / "data_service.lock",
        {"pid": 123, "hostname": "host"},
    )

    report = collector_health_report(
        settings=settings,
        database=FakeDatabase(now),
        service_id="operate-shadow",
        observed_at=now,
    )

    assert report["status"] == "HEALTHY"
    assert report["process_id"] == 123
    assert report["host"] == "host"
    assert report["last_trade_timestamp"]
    assert report["last_orderbook_timestamp"]
    assert report["last_sequence_number"] == 10
    assert report["reconnect_count"] == 2
    assert report["gap_count"] == 0
    assert report["bytes_written"] > 0
    assert report["checksum"]


def test_collector_health_fails_closed_on_sequence_gap(
    tmp_path,
) -> None:
    now = datetime(2026, 7, 26, 17, 0, tzinfo=UTC)
    settings = _settings(tmp_path)
    atomic_write_json(
        settings.paths.checkpoints_dir
        / "operate-shadow_heartbeat.json",
        {"pid": 123, "heartbeat_at": now.isoformat()},
    )

    report = collector_health_report(
        settings=settings,
        database=FakeDatabase(now, gaps=1),
        service_id="operate-shadow",
        observed_at=now,
    )
    assert report["status"] == "GAP_DETECTED"
    assert "GAP_DETECTED" in report["reason_codes"]
