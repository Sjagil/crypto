"""Deterministic synthetic throughput probe for Bitvavo L2 V2."""

from __future__ import annotations

import os
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from typing import Any

from data.bitvavo_l2_reconstruction_v2 import BitvavoL2StateMachine
from utils.common import stable_hash, utc_iso


def benchmark_bitvavo_l2_v2(event_count: int = 100_000) -> dict[str, Any]:
    if event_count < 1:
        raise ValueError("event_count must be positive")
    started_at = datetime.now(UTC)
    book = BitvavoL2StateMachine("BTC-EUR", stale_after=timedelta(days=1))
    assert book.seed_snapshot(
        bids=[["100", "2"]],
        asks=[["101", "3"]],
        sequence=1,
        event_at=started_at,
        known_at=started_at,
        snapshot_reference="synthetic-performance-probe",
    )
    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for index in range(event_count):
        event_at = started_at + timedelta(microseconds=index + 1)
        applied = book.apply_delta(
            bids=[["100", str(2 + (index % 10) / 10)]],
            asks=[],
            sequence=index + 2,
            event_at=event_at,
            known_at=event_at,
            event_id=f"synthetic:{index + 2}",
        )
        if not applied:
            raise RuntimeError(f"synthetic event {index} was rejected")
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    body = {
        "schema_version": "bitvavo_l2_v2_performance_v1",
        "observed_at": utc_iso(),
        "pid": os.getpid(),
        "scope": "DETERMINISTIC_SYNTHETIC_CPU_AND_MEMORY_PROBE",
        "events": event_count,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "events_per_second": event_count / wall_seconds if wall_seconds else None,
        "cpu_utilization_single_core_fraction": cpu_seconds / wall_seconds if wall_seconds else None,
        "peak_traced_memory_bytes": peak_bytes,
        "bounded_recent_event_capacity": 20_000,
        "queue_capacity": 20_000,
        "queue_drops": 0,
        "final_sequence": book.sequence,
        "final_state": book.state.value,
        "features_available": book.features(started_at + timedelta(seconds=1)) is not None,
        "execution_authority": False,
        "orders_generated": 0,
    }
    return {**body, "benchmark_hash": stable_hash(body)}


__all__ = ["benchmark_bitvavo_l2_v2"]
