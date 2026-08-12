from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from reporting.bitvavo_l2_recovery import protocol_semantics_matrix, replay_source_neutral_l2
from utils.common import stable_hash


def _write_rows(root: Path, rows: list[dict]) -> None:
    target = (
        root
        / "schema=source_neutral_observation_v1"
        / "date=2026-08-10"
        / "event_hour=12"
        / "events.jsonl"
    )
    target.parent.mkdir(parents=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(kind: str, nonce: int, at: datetime, *, bids: list, asks: list) -> dict:
    payload = {"market": "BTC-EUR", "nonce": nonce, "bids": bids, "asks": asks}
    return {
        "data_type": kind,
        "venue_instrument_id": "BTC-EUR",
        "known_at": at.isoformat().replace("+00:00", "Z"),
        "provider_timestamp": at.isoformat().replace("+00:00", "Z"),
        "exchange_event_timestamp": at.isoformat().replace("+00:00", "Z"),
        "raw_payload": payload,
        "raw_payload_hash": stable_hash(payload),
        "observation_id": stable_hash([kind, nonce, at]),
        "metadata": {"source_sequence": nonce},
    }


def test_protocol_matrix_is_official_and_complete() -> None:
    rows = protocol_semantics_matrix()
    assert len(rows) >= 6
    assert all(row["status"] == "MATCH" for row in rows)
    assert all(str(row["source"]).startswith("https://docs.bitvavo.com/") for row in rows)


def test_replay_uses_snapshot_and_suppresses_post_gap_features(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    seeds = tmp_path / "seeds"
    at = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_rows(
        seeds,
        [_row("ORDERBOOK_SNAPSHOT", 10, at, bids=[["100", "2"]], asks=[["101", "3"]])],
    )
    _write_rows(
        raw,
        [
            _row("ORDERBOOK_DELTA", 11, at + timedelta(seconds=1), bids=[["100", "2.1"]], asks=[]),
            _row("ORDERBOOK_DELTA", 13, at + timedelta(seconds=2), bids=[], asks=[["101", "4"]]),
        ],
    )
    result = replay_source_neutral_l2(
        workspace=Path(__file__).resolve().parents[1],
        raw_root=raw,
        auxiliary_snapshot_root=seeds,
    )
    btc = result["assets"]["BTC-EUR"]
    assert result["source_segment_bounds"]
    assert all(
        len(row["sha256_at_capture"]) == 64
        and row["hash_scope"] == "EXACT_BOUNDED_PREFIX"
        for row in result["source_segment_bounds"]
    )
    assert btc["data_counts"]["applied_deltas"] == 1
    assert btc["failure_counts"]["NONCE_GAP"] == 1
    assert btc["state"] == "RESEED_REQUIRED"
    assert btc["feature_count"] == 1
    assert not result["historical_missing_evidence_fabricated"]


def test_replay_causally_orders_equal_receive_batch_by_nonce(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    at = datetime(2026, 8, 10, 12, tzinfo=UTC)
    rows = [
        _row("ORDERBOOK_SNAPSHOT", 10, at, bids=[["100", "2"]], asks=[["101", "3"]]),
        _row("ORDERBOOK_DELTA", 12, at + timedelta(seconds=1), bids=[["100", "4"]], asks=[]),
        _row("ORDERBOOK_DELTA", 11, at + timedelta(seconds=1), bids=[["100", "3"]], asks=[]),
    ]
    _write_rows(raw, rows)
    result = replay_source_neutral_l2(
        workspace=Path(__file__).resolve().parents[1],
        raw_root=raw,
    )
    btc = result["assets"]["BTC-EUR"]
    assert btc["data_counts"]["applied_deltas"] == 2
    assert btc["failure_counts"] == {}
    assert result["causal_reorder_policy"]["cross_known_at_reordering"] is False
    assert (
        result["causal_reorder_policy"]["observable_stats"][
            "same_receive_batches_reordered"
        ]
        == 1
    )
