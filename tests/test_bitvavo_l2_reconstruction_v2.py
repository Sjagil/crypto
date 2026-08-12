from __future__ import annotations

from datetime import UTC, datetime, timedelta

from data.bitvavo_l2_reconstruction_v2 import (
    BitvavoBookState,
    BitvavoL2StateMachine,
)

T0 = datetime(2026, 8, 10, 12, tzinfo=UTC)


def machine() -> BitvavoL2StateMachine:
    return BitvavoL2StateMachine("BTC-EUR", stale_after=timedelta(seconds=10))


def seed(book: BitvavoL2StateMachine, sequence: int = 10) -> None:
    assert book.seed_snapshot(
        bids=[["100", "2"], ["99", "1"]],
        asks=[["101", "3"], ["102", "1"]],
        sequence=sequence,
        event_at=T0,
        known_at=T0,
        snapshot_reference="s" * 64,
    )


def test_perfect_sequence_and_decimal_delete_semantics() -> None:
    book = machine()
    seed(book)
    assert book.apply_delta(
        bids=[["100", "0"], ["100.5", "1.25"]],
        asks=[],
        sequence=11,
        event_at=T0 + timedelta(seconds=1),
        known_at=T0 + timedelta(seconds=1),
        event_id="11",
    )
    feature = book.features(T0 + timedelta(seconds=1))
    assert feature is not None
    assert feature["best_bid"] == "100.5"
    assert feature["best_ask"] == "101"
    assert feature["book_state"] == "VALID"


def test_missing_delta_fails_closed_and_suppresses_features() -> None:
    book = machine()
    seed(book)
    assert not book.apply_delta(
        bids=[],
        asks=[["102", "2"]],
        sequence=12,
        event_at=T0 + timedelta(seconds=2),
        known_at=T0 + timedelta(seconds=2),
        event_id="12",
    )
    assert book.state is BitvavoBookState.RESEED_REQUIRED
    assert book.features(T0 + timedelta(seconds=2)) is None
    assert book.failure_counts["NONCE_GAP"] == 1


def test_reconnect_and_restart_require_fresh_snapshot() -> None:
    book = machine()
    seed(book)
    book.on_disconnect(T0 + timedelta(seconds=1))
    assert book.state is BitvavoBookState.RESEED_REQUIRED
    assert not book.apply_delta(
        bids=[],
        asks=[],
        sequence=11,
        event_at=T0,
        known_at=T0 + timedelta(seconds=2),
        event_id="11",
    )
    assert book.features(T0 + timedelta(seconds=2)) is None
    seed(book, sequence=20)
    assert book.state is BitvavoBookState.VALID
    assert book.reseed_count == 2


def test_duplicate_is_idempotent_but_out_of_order_nonce_fails_closed() -> None:
    book = machine()
    seed(book)
    kwargs = dict(
        bids=[["100", "2.1"]],
        asks=[],
        sequence=11,
        event_at=T0 + timedelta(seconds=1),
        known_at=T0 + timedelta(seconds=1),
        event_id="same",
    )
    assert book.apply_delta(**kwargs)
    assert not book.apply_delta(**kwargs)
    assert book.state is BitvavoBookState.VALID
    assert book.duplicate_events == 1
    assert not book.apply_delta(
        bids=[],
        asks=[],
        sequence=10,
        event_at=T0 + timedelta(seconds=2),
        known_at=T0 + timedelta(seconds=2),
        event_id="old",
    )
    assert book.state is BitvavoBookState.RESEED_REQUIRED


def test_buffered_delta_at_snapshot_nonce_is_discarded_without_invalidating() -> None:
    book = machine()
    seed(book, sequence=20)
    assert not book.apply_delta(
        bids=[["100", "9"]],
        asks=[],
        sequence=20,
        event_at=T0 - timedelta(milliseconds=1),
        known_at=T0 + timedelta(milliseconds=1),
        event_id="buffered-20",
        buffered_after_snapshot=True,
    )
    assert book.state is BitvavoBookState.VALID
    assert book.pre_snapshot_discarded == 1
    assert book.features(T0 + timedelta(milliseconds=1)) is not None


def test_crossed_book_and_negative_quantity_are_invalid() -> None:
    for rows, sequence in [([[["102", "1"]], []], 11), ([[["100", "-1"]], []], 11)]:
        book = machine()
        seed(book)
        assert not book.apply_delta(
            bids=rows[0],
            asks=rows[1],
            sequence=sequence,
            event_at=T0 + timedelta(seconds=1),
            known_at=T0 + timedelta(seconds=1),
            event_id=str(rows),
        )
        assert book.state is BitvavoBookState.RESEED_REQUIRED
        assert book.features(T0 + timedelta(seconds=1)) is None


def test_stale_book_closes_valid_interval_and_requires_reseed() -> None:
    book = machine()
    seed(book)
    assert book.features(T0 + timedelta(seconds=11)) is None
    assert book.state is BitvavoBookState.RESEED_REQUIRED
    assert len(book.intervals) == 1
    assert book.intervals[0].closing_reason == "BOOK_STALE"


def test_determinism_partition_independence_and_future_invariance() -> None:
    events = [
        dict(
            bids=[["100", str(2 + index / 10)]],
            asks=[],
            sequence=11 + index,
            event_at=T0 + timedelta(seconds=index + 1),
            known_at=T0 + timedelta(seconds=index + 1),
            event_id=str(11 + index),
        )
        for index in range(4)
    ]
    left, right = machine(), machine()
    seed(left)
    seed(right)
    for event in events:
        assert left.apply_delta(**event)
    for partition in (events[:2], events[2:]):
        for event in partition:
            assert right.apply_delta(**event)
    at = T0 + timedelta(seconds=4)
    assert left.features(at) == right.features(at)
    before = left.features(at)
    assert left.apply_delta(
        bids=[],
        asks=[["101", "4"]],
        sequence=15,
        event_at=T0 + timedelta(seconds=5),
        known_at=T0 + timedelta(seconds=5),
        event_id="15",
    )
    assert before == right.features(at)


def test_bad_snapshot_never_creates_a_valid_interval() -> None:
    book = machine()
    assert not book.seed_snapshot(
        bids=[["101", "1"]],
        asks=[["100", "1"]],
        sequence=10,
        event_at=T0,
        known_at=T0,
        snapshot_reference="bad",
    )
    assert book.state is BitvavoBookState.RESEED_REQUIRED
    assert book.intervals == []
