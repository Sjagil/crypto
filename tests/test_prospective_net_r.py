from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from config.settings import PathSettings, Settings
from reporting.prospective_net_r import build_prospective_net_r_calibration
from utils.common import atomic_write_json


def _settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def test_prospective_net_r_requires_diverse_point_in_time_support(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    as_of = datetime(2026, 8, 12, tzinfo=UTC)
    rows = []
    for index in range(40):
        decision = as_of - timedelta(days=12) + timedelta(hours=index * 6)
        rows.append(
            {
                "decision_timestamp": decision.isoformat(),
                "label_available_at": (decision + timedelta(hours=24)).isoformat(),
                "canonical_point_in_time_ready": True,
                "label_uses_future_features": False,
                "net_return_r": 1.0 if index % 2 == 0 else -0.25,
                "features": {
                    "market": ("BTC-EUR", "ETH-EUR", "SOL-EUR")[index % 3],
                    "family": "BREAKOUT_RETEST",
                    "entry_timeframe": "15m",
                    "setup_timeframe": "1h",
                },
            }
        )
    rows_path = tmp_path / "output" / "ml" / "datasets" / "test" / "rows.json"
    atomic_write_json(rows_path, {"rows": rows})
    atomic_write_json(
        tmp_path / "output" / "ml" / "canonical_training_status.json",
        {
            "dataset_registered": True,
            "dataset_id": "test",
            "canonical_row_count": len(rows),
            "rows_artifact": str(rows_path),
        },
    )
    candidate = {
        "opportunity_id": "candidate-1",
        "family": "BREAKOUT_RETEST",
        "entry_timeframe": "15m",
        "confirmation_timeframe": "1h",
    }

    result = build_prospective_net_r_calibration(
        settings,
        [candidate],
        as_of=as_of,
    )
    estimate = result["estimates_by_opportunity_id"]["candidate-1"]

    assert estimate["status"] == "CALIBRATED_PROSPECTIVE_EXPECTATION"
    assert estimate["row_count"] == 40
    assert estimate["distinct_market_count"] == 3
    assert estimate["label_span_days"] >= 7
    assert estimate["expected_net_r"] is not None
    assert estimate["expected_net_r"] > 0
    assert estimate["may_change_live_authority"] is False


def test_prospective_net_r_excludes_labels_unavailable_at_decision(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    as_of = datetime(2026, 8, 12, tzinfo=UTC)
    rows_path = tmp_path / "output" / "ml" / "datasets" / "test" / "rows.json"
    atomic_write_json(
        rows_path,
        {
            "rows": [
                {
                    "decision_timestamp": "2026-08-11T12:00:00+00:00",
                    "label_available_at": "2026-08-12T12:00:00+00:00",
                    "canonical_point_in_time_ready": True,
                    "label_uses_future_features": False,
                    "net_return_r": 10.0,
                    "features": {
                        "market": "BTC-EUR",
                        "family": "BREAKOUT_RETEST",
                        "entry_timeframe": "15m",
                        "setup_timeframe": "1h",
                    },
                }
            ]
        },
    )
    atomic_write_json(
        tmp_path / "output" / "ml" / "canonical_training_status.json",
        {
            "dataset_registered": True,
            "dataset_id": "test",
            "canonical_row_count": 1,
            "rows_artifact": str(rows_path),
        },
    )

    result = build_prospective_net_r_calibration(
        settings,
        [
            {
                "opportunity_id": "candidate-1",
                "family": "BREAKOUT_RETEST",
                "entry_timeframe": "15m",
                "confirmation_timeframe": "1h",
            }
        ],
        as_of=as_of,
    )
    estimate = result["estimates_by_opportunity_id"]["candidate-1"]

    assert estimate["row_count"] == 0
    assert estimate["future_labels_excluded"] == 1
    assert estimate["expected_net_r"] is None
    assert estimate["status"] == "INSUFFICIENT_PROSPECTIVE_SUPPORT"
