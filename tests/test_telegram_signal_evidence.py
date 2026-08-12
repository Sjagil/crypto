from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import PathSettings
from reporting.telegram_signal_evidence import build_telegram_signal_evidence
from utils.common import (
    append_jsonl,
    atomic_write_json,
    stable_hash,
    utc_iso,
)

ALERTED_AT = datetime(2026, 8, 10, 12, 5, tzinfo=UTC)


def _settings(isolated_settings, tmp_path: Path):
    return isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _write_candles(
    root: Path,
    market: str,
    rows: list[tuple[datetime, float, float, float, float]],
) -> None:
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close"],
    )
    frame["volume"] = 1.0
    path = root / "data_store" / "normalized" / f"{market}_15m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _row(opportunity_id: str, market: str) -> dict[str, object]:
    return {
        "opportunity_id": opportunity_id,
        "signal_timestamp": utc_iso(ALERTED_AT - timedelta(minutes=5)),
        "market": market,
        "status": "ACTIONABLE",
        "strategy": "TEST_BREAKOUT",
        "strategy_dna_hash": f"dna-{opportunity_id}",
        "timeframe": "15m",
        "current_price": 99.0,
        "trigger": 100.0,
        "stop": 98.0,
        "target_1": 101.0,
        "target_2": 102.0,
        "entry_condition": "HIGH_AT_OR_ABOVE_TRIGGER",
        "entry_condition_source": "ALERT_CURRENT_PRICE",
        "evaluation_roundtrip_cost_bps": 35.0,
        "live_authority_granted": False,
    }


def _append_event(path: Path, rows: list[dict[str, object]], previous: str) -> str:
    body = {
        "schema_version": "telegram_opportunity_evidence_event_v1",
        "recorded_at": utc_iso(ALERTED_AT),
        "notification_id": stable_hash(rows, length=40),
        "delivery_status_at_capture": "PENDING",
        "rows": rows,
        "previous_hash": previous,
    }
    record_hash = stable_hash(body, length=64)
    append_jsonl(path, {**body, "record_hash": record_hash})
    return record_hash


def test_exact_prospective_cohort_resolves_tp2_stop_ambiguity_and_open(
    isolated_settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    evidence_path = (
        settings.paths.output_dir
        / "notifications"
        / "telegram_opportunity_evidence.jsonl"
    )
    rows = [
        _row("tp2", "BTC-EUR"),
        _row("stop", "ETH-EUR"),
        _row("ambiguous", "SOL-EUR"),
        _row("not-triggered", "LINK-EUR"),
        _row("future-excluded", "AVAX-EUR"),
    ]
    first_hash = _append_event(evidence_path, rows, "GENESIS")
    _append_event(evidence_path, [_row("tp2", "BTC-EUR")], first_hash)
    atomic_write_json(
        settings.paths.output_dir
        / "notifications"
        / "telegram_preview.json",
        [],
    )

    entry_candle = (ALERTED_AT.replace(minute=15), 99.0, 100.5, 98.5, 100.2)
    _write_candles(
        tmp_path,
        "BTC-EUR",
        [
            entry_candle,
            (ALERTED_AT.replace(minute=30), 100.2, 102.5, 99.5, 102.0),
        ],
    )
    _write_candles(
        tmp_path,
        "ETH-EUR",
        [
            entry_candle,
            (ALERTED_AT.replace(minute=30), 100.2, 100.5, 97.5, 98.0),
        ],
    )
    _write_candles(
        tmp_path,
        "SOL-EUR",
        [(ALERTED_AT.replace(minute=15), 99.0, 102.5, 97.5, 100.0)],
    )
    untriggered = []
    cursor = ALERTED_AT.replace(minute=15)
    while cursor <= ALERTED_AT + timedelta(hours=24):
        untriggered.append((cursor, 99.0, 99.5, 98.5, 99.0))
        cursor += timedelta(minutes=15)
    _write_candles(tmp_path, "LINK-EUR", untriggered)
    _write_candles(
        tmp_path,
        "AVAX-EUR",
        [
            entry_candle,
            (ALERTED_AT.replace(minute=45), 100.2, 102.5, 99.5, 102.0),
        ],
    )

    result = build_telegram_signal_evidence(
        settings,
        observed_at=ALERTED_AT + timedelta(minutes=45),
        force=True,
    )
    outcomes = {
        row["opportunity_id"]: row["outcome"]
        for row in result["prospective_exact_evidence"]["outcomes"]
    }
    assert outcomes == {
        "tp2": "TP2_BEFORE_STOP",
        "stop": "STOP_BEFORE_TP2",
        "ambiguous": "AMBIGUOUS_SAME_CANDLE",
        "not-triggered": "OPEN_NOT_TRIGGERED",
        "future-excluded": "OPEN_TRIGGERED",
    }
    summary = result["prospective_exact_evidence"]["summary"]
    assert summary["alert_count"] == 5
    assert summary["boundary_resolved_count"] == 3
    assert summary["tp2_before_stop_count"] == 1
    assert summary["tp2_before_stop_fraction"] == 1 / 3
    assert result["paper_shadow_gate"]["status"] == (
        "COLLECT_EXACT_PROSPECTIVE_EVIDENCE"
    )
    assert result["paper_shadow_gate"]["automatic_live_authority_changes"] is False
    assert result["paper_shadow_gate"]["live_order_authority_granted"] is False
    assert result["orders_submitted"] == 0


def test_corrupt_hash_chain_fails_closed(isolated_settings, tmp_path: Path) -> None:
    settings = _settings(isolated_settings, tmp_path)
    evidence_path = (
        settings.paths.output_dir
        / "notifications"
        / "telegram_opportunity_evidence.jsonl"
    )
    _append_event(evidence_path, [_row("tp2", "BTC-EUR")], "WRONG")

    result = build_telegram_signal_evidence(
        settings,
        observed_at=ALERTED_AT + timedelta(hours=25),
        force=True,
    )

    exact = result["prospective_exact_evidence"]
    assert exact["hash_chain_status"] == "INVALID"
    assert exact["event_count"] == 0
    assert exact["summary"]["alert_count"] == 0
    assert result["paper_shadow_gate"]["conditions"]["integrity_clean"] is False
    assert result["paper_shadow_gate"]["live_order_authority_granted"] is False
