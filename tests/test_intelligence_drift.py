from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.intelligence_drift import build_intelligence_drift_report


def _rows(*, shifted: bool) -> list[dict]:
    rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(200):
        recent_shift = 8.0 if shifted and index >= 140 else 0.0
        rows.append(
            {
                "decision_timestamp": (start + timedelta(minutes=index)).isoformat(),
                "net_profitable": index % 2,
                "features": {
                    "score": float(index % 20) + recent_shift,
                    "spread_bps": float(index % 5) + recent_shift,
                    "market": "BTC-EUR" if index % 2 else "ETH-EUR",
                    "family": "TREND" if index % 3 else "REVERSAL",
                },
            }
        )
    return rows


def test_drift_monitor_is_shadow_only_and_orderless(tmp_path) -> None:
    report = build_intelligence_drift_report(
        tmp_path,
        rows=_rows(shifted=False),
        oos_predictions=[
            {
                "decision_timestamp": row["decision_timestamp"],
                "prediction": 0.7 if row["net_profitable"] else 0.3,
                "label": row["net_profitable"],
            }
            for row in _rows(shifted=False)
        ],
        training_data_hash="fixture",
    )

    assert report["authority"] == "SHADOW_ONLY"
    assert report["live_decision_influence"] is False
    assert report["orders_generated"] == 0
    assert report["orders_submitted"] == 0
    assert report["interpretation"]["drift_cannot_veto_deterministic_entries"] is True
    assert (tmp_path / "output" / "intelligence" / "drift_report.json").is_file()


def test_drift_monitor_detects_large_recent_feature_shift(tmp_path) -> None:
    report = build_intelligence_drift_report(
        tmp_path,
        rows=_rows(shifted=True),
        training_data_hash="fixture-shift",
    )

    assert report["status"] == "CRITICAL_DRIFT_SHADOW_WARNING"
    assert "score" in report["critical_features"]
    assert report["feature_drift"]["score"]["value"] >= 0.25
