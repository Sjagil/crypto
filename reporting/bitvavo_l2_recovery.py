"""Forensic and replay reporting for Bitvavo L2 reconstruction V2."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Iterator

from data.bitvavo_l2_reconstruction_v2 import (
    L2_RECONSTRUCTION_VERSION,
    BitvavoBookState,
    BitvavoL2StateMachine,
    transition_trace,
    valid_intervals,
)
from data.multi_source_maturation import bitvavo_l2_maturation
from research.reference_integrations import (
    freqtrade_timeframe_seconds,
    lean_sharpe_ratio,
    nautilus_top_of_book,
    pybroker_returns,
    qlib_sum_by_index,
    vectorbt_book_invariants,
)
from utils.common import stable_hash, utc_iso, utc_now

BITVAVO_L2_RECOVERY_REPORT_SCHEMA = "bitvavo_l2_recovery_report_v1"
PRIMARY_MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR")

LOCAL_REASONS = {
    "RECORDER_STARTED_MID_HOUR",
    "BOOK_SEQUENCE_INVALID",
    "NO_VALID_ORDERBOOK",
    "POSITIONING_CONTEXT_TIMEOUT",
}
SOURCE_OR_TRANSPORT_REASONS = {
    "NO_STREAM_EVENTS",
    "STREAM_NOT_HEALTHY",
    "SEQUENCE_GAP",
    "DROPPED_MESSAGES",
    "ARRIVAL_END_EARLY",
    "ARRIVAL_START_LATE",
    "NO_TRADES",
    "NO_TICKER",
}
UNRELATED_CONTEXT_REASONS = {
    "DERIVATIVES_CONTEXT_INCOMPLETE",
    "PERPETUAL_SPOT_VOLUME_RATIO_UNAVAILABLE",
    "PERPETUAL_SPOT_QUOTE_VOLUME_RATIO_UNAVAILABLE",
}


def protocol_semantics_matrix() -> list[dict[str, Any]]:
    manage = "https://docs.bitvavo.com/docs/manage-order-book/"
    rest = "https://docs.bitvavo.com/docs/rest-api/get-order-book/"
    subscription = "https://docs.bitvavo.com/docs/websocket-api/book-subscription/"
    return [
        {
            "topic": "subscription_before_snapshot",
            "official_semantics": "subscribe to book, then buffer events ordered by nonce",
            "v2_behavior": "subscribe/buffer before every startup, restart and reconnect seed",
            "source": manage,
            "status": "MATCH",
        },
        {
            "topic": "trusted_snapshot",
            "official_semantics": "GET /{market}/book returns bids, asks and a version nonce",
            "v2_behavior": "public REST response is persisted as immutable ORDERBOOK_SNAPSHOT evidence",
            "source": rest,
            "status": "MATCH",
        },
        {
            "topic": "buffer_discard",
            "official_semantics": "discard buffered events with nonce at or below snapshot nonce",
            "v2_behavior": "only the bounded post-subscription sync buffer may discard <= snapshot nonce",
            "source": manage,
            "status": "MATCH",
        },
        {
            "topic": "continuity",
            "official_semantics": "each next event nonce must be exactly local nonce + 1",
            "v2_behavior": "any other new nonce fails closed and requires a fresh snapshot",
            "source": manage,
            "status": "MATCH",
        },
        {
            "topic": "level_update",
            "official_semantics": "replace a price size; size zero deletes the level",
            "v2_behavior": "exact Decimal replace/delete semantics applied transactionally",
            "source": subscription,
            "status": "MATCH",
        },
        {
            "topic": "depth",
            "official_semantics": "REST returns at most 1000 levels per side",
            "v2_behavior": "configured bounded depth, sorted independently per side",
            "source": rest,
            "status": "MATCH",
        },
    ]


def v1_forensics(snapshot_root: Path) -> dict[str, Any]:
    maturation = bitvavo_l2_maturation(snapshot_root)
    reason_totals: Counter[str] = Counter()
    state_totals: Counter[str] = Counter()
    for market in PRIMARY_MARKETS:
        row = maturation["assets"][market]
        reason_totals.update(row.get("reason_counts") or {})
        state_totals.update(row.get("state_counts") or {})

    def selected(reasons: set[str]) -> dict[str, int]:
        return {
            reason: reason_totals[reason] for reason in sorted(reasons) if reason_totals[reason]
        }

    classified_total = sum(reason_totals.values())
    return {
        "schema_version": "bitvavo_l2_v1_forensics_v1",
        "observed_at": utc_iso(),
        "baseline": maturation,
        "state_totals": dict(state_totals),
        "failure_taxonomy": {
            "LOCAL_COLLECTOR_OR_POLICY": selected(LOCAL_REASONS),
            "SOURCE_OR_TRANSPORT_EVIDENCE": selected(SOURCE_OR_TRANSPORT_REASONS),
            "UNRELATED_CONTEXT_CONFLATION": selected(UNRELATED_CONTEXT_REASONS),
            "UNCLASSIFIED": {
                reason: count
                for reason, count in sorted(reason_totals.items())
                if reason not in LOCAL_REASONS
                and reason not in SOURCE_OR_TRANSPORT_REASONS
                and reason not in UNRELATED_CONTEXT_REASONS
            },
        },
        "classified_reason_occurrences": classified_total,
        "root_causes": [
            {
                "code": "V1_SAMPLED_DELTA_PERSISTENCE",
                "classification": "LOCAL",
                "finding": (
                    "V1 applied every delta in memory but persisted only periodic five-second book states; "
                    "therefore pre-multi-source history cannot support exact raw-delta V2 replay."
                ),
                "recovery": "PROSPECTIVE_ONLY_WHERE_RAW_DELTAS_AND_TRUSTED_SNAPSHOT_EXIST",
            },
            {
                "code": "LIFECYCLE_REASON_CONFLATION",
                "classification": "LOCAL",
                "finding": (
                    "RECORDER_STARTED_MID_HOUR and shared stream-health counters invalidate whole hourly "
                    "asset rows even when a book later has an exact valid sub-interval."
                ),
                "recovery": "EXPLICIT_VALIDITY_INTERVALS",
            },
            {
                "code": "UNRELATED_CONTEXT_CONFLATION",
                "classification": "LOCAL",
                "finding": "derivatives/positioning completeness is mixed into an L2 status row",
                "recovery": "SEPARATE_EXECUTION_FLOW_AND_L2_HEALTH",
            },
            {
                "code": "SNAPSHOT_EVIDENCE_NOT_IN_SOURCE_NEUTRAL_LEDGER",
                "classification": "LOCAL",
                "finding": (
                    "the continuous source-neutral delta ledger started without corresponding immutable "
                    "ORDERBOOK_SNAPSHOT observations"
                ),
                "recovery": "V2_PERSISTS_EVERY_RESEED_SNAPSHOT",
            },
            {
                "code": "SOURCE_NEUTRAL_BATCH_TIEBREAK_ORDER",
                "classification": "LOCAL",
                "finding": (
                    "the immutable writer orders simultaneously known observations by identity, "
                    "so nonce order inside one receive batch is not a replay-order guarantee; no "
                    "events are lost"
                ),
                "recovery": (
                    "BOUNDED_CAUSAL_NONCE_ORDER_WITHIN_IDENTICAL_KNOWN_AT_BATCH_ONLY"
                ),
            },
            {
                "code": "REAL_TRANSPORT_GAPS_PRESENT",
                "classification": "SOURCE_OR_TRANSPORT",
                "finding": "stream-unhealthy, dropped-message and arrival-boundary evidence is present",
                "recovery": "FAIL_CLOSED_RESEED_WITH_BOUNDED_BACKOFF",
            },
        ],
        "raw_history_repair_policy": {
            "overwrite_v1": False,
            "fabricate_missing_deltas": False,
            "infer_book_from_trades": False,
            "pre_evidence_intervals": "IRRECOVERABLE_FOR_EXACT_L2_RESEARCH",
        },
    }


def _sha256_prefix(path: Path, size_bytes: int) -> str:
    """Hash exactly the immutable byte prefix selected for this replay."""

    digest = hashlib.sha256()
    remaining = size_bytes
    with path.open("rb") as stream:
        while remaining > 0:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    if remaining:
        raise OSError(f"replay segment shrank during capture: {path}")
    return digest.hexdigest()


def _bounded_jsonl_records(root: Path) -> tuple[Iterator[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(root.rglob("events.jsonl"))
    bounds = []
    for path in files:
        size_bytes = path.stat().st_size
        bounds.append(
            {
                "path": str(path.resolve()),
                "size_bytes": size_bytes,
                "sha256_at_capture": _sha256_prefix(path, size_bytes),
                "hash_scope": "EXACT_BOUNDED_PREFIX",
            }
        )

    def iterator() -> Iterator[dict[str, Any]]:
        for path, bound in zip(files, bounds, strict=True):
            remaining = int(bound["size_bytes"])
            with path.open("rb") as stream:
                while remaining > 0:
                    line = stream.readline(remaining)
                    if not line:
                        break
                    remaining -= len(line)
                    if not line.endswith(b"\n"):
                        break
                    try:
                        yield dict(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return iterator(), bounds


def _load_auxiliary_snapshots(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.is_dir():
        return []
    rows, _ = _bounded_jsonl_records(root)
    return sorted(
        [row for row in rows if row.get("data_type") == "ORDERBOOK_SNAPSHOT"],
        key=lambda row: str(row.get("known_at") or ""),
    )


def _causally_order_l2_records(
    rows: Iterator[dict[str, Any]],
    stats: Counter[str],
    *,
    maximum_same_receive_batch: int = 20_000,
) -> Iterator[dict[str, Any]]:
    """Restore nonce order only inside one causally simultaneous receive batch.

    The immutable batch writer sorts equal-known-at observations by identity,
    which can scramble venue nonce order without losing any event.  Events in
    the same receive batch became knowable simultaneously, so bounded ordering
    by the source nonce is causal; ordering across known-at boundaries remains
    forbidden.
    """

    selected = (
        row
        for row in rows
        if row.get("data_type") in {"ORDERBOOK_DELTA", "ORDERBOOK_SNAPSHOT"}
    )
    for _, grouped in groupby(selected, key=lambda row: str(row.get("known_at") or "")):
        batch = list(grouped)
        if len(batch) > maximum_same_receive_batch:
            raise RuntimeError("same-receive L2 batch exceeds deterministic reorder bound")
        ordered = sorted(
            batch,
            key=lambda row: (
                0 if row.get("data_type") == "ORDERBOOK_SNAPSHOT" else 1,
                str(row.get("venue_instrument_id") or ""),
                int(
                    (row.get("raw_payload") or {}).get("nonce")
                    or (row.get("metadata") or {}).get("source_sequence")
                    or -1
                ),
                str(row.get("observation_id") or row.get("source_event_id") or ""),
            ),
        )
        if [row.get("observation_id") for row in ordered] != [
            row.get("observation_id") for row in batch
        ]:
            stats["same_receive_batches_reordered"] += 1
            stats["events_in_reordered_batches"] += len(batch)
        stats["maximum_same_receive_batch_seen"] = max(
            stats["maximum_same_receive_batch_seen"], len(batch)
        )
        yield from ordered


def replay_source_neutral_l2(
    *,
    workspace: Path,
    raw_root: Path,
    auxiliary_snapshot_root: Path | None = None,
) -> dict[str, Any]:
    machines = {
        market: BitvavoL2StateMachine(market, maximum_levels=500) for market in PRIMARY_MARKETS
    }
    sync_buffer = {market: False for market in PRIMARY_MARKETS}
    features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_bucket: dict[str, int] = {}
    feature_cadence_seconds = freqtrade_timeframe_seconds(workspace, "5m") // 60
    first_known: dict[str, datetime] = {}
    last_known: dict[str, datetime] = {}
    data_counts: dict[str, Counter[str]] = defaultdict(Counter)
    snapshots = _load_auxiliary_snapshots(auxiliary_snapshot_root)
    snapshot_index = 0
    raw_records, segment_bounds = _bounded_jsonl_records(raw_root)
    reorder_stats: Counter[str] = Counter()

    def parse_time(value: Any) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)

    def apply_snapshot(row: dict[str, Any]) -> None:
        nonlocal snapshot_index
        market = str(row.get("venue_instrument_id") or "").upper()
        if market not in machines:
            return
        known = parse_time(row["known_at"])
        payload = dict(row.get("raw_payload") or {})
        machine = machines[market]
        if machine.state is BitvavoBookState.VALID:
            return
        accepted = machine.seed_snapshot(
            bids=payload.get("bids") or [],
            asks=payload.get("asks") or [],
            sequence=payload.get("nonce") or (row.get("metadata") or {}).get("source_sequence"),
            event_at=parse_time(row.get("provider_timestamp") or row["known_at"]),
            known_at=known,
            snapshot_reference=str(row.get("raw_payload_hash") or row.get("observation_id")),
        )
        sync_buffer[market] = accepted
        data_counts[market]["snapshots"] += 1

    for row in _causally_order_l2_records(raw_records, reorder_stats):
        market = str(row.get("venue_instrument_id") or "").upper()
        if market not in machines:
            continue
        known = parse_time(row["known_at"])
        first_known.setdefault(market, known)
        last_known[market] = known
        while (
            snapshot_index < len(snapshots)
            and parse_time(snapshots[snapshot_index]["known_at"]) <= known
        ):
            apply_snapshot(snapshots[snapshot_index])
            snapshot_index += 1
        if row.get("data_type") == "ORDERBOOK_SNAPSHOT":
            apply_snapshot(row)
            continue
        payload = dict(row.get("raw_payload") or {})
        machine = machines[market]
        machine.check_stale(known)
        applied = machine.apply_delta(
            bids=payload.get("bids") or [],
            asks=payload.get("asks") or [],
            sequence=payload.get("nonce") or (row.get("metadata") or {}).get("source_sequence"),
            event_at=parse_time(row.get("exchange_event_timestamp") or row["known_at"]),
            known_at=known,
            event_id=str(row.get("observation_id") or row.get("source_event_id")),
            buffered_after_snapshot=sync_buffer[market],
        )
        if applied:
            sync_buffer[market] = False
            data_counts[market]["applied_deltas"] += 1
            bucket = int(known.timestamp()) // feature_cadence_seconds
            if feature_bucket.get(market) != bucket:
                feature_bucket[market] = bucket
                feature = machine.features(known)
                if feature is not None:
                    features[market].append(feature)
        else:
            data_counts[market]["rejected_or_discarded_deltas"] += 1

    for market, machine in machines.items():
        if market in last_known:
            machine.finalize(last_known[market])

    assets: dict[str, Any] = {}
    all_returns: list[dict[int, float]] = []
    for market, machine in machines.items():
        window = (
            max(0.0, (last_known[market] - first_known[market]).total_seconds())
            if market in first_known and market in last_known
            else 0.0
        )
        valid_seconds = sum(interval.duration_seconds for interval in machine.intervals)
        gap_durations: list[float] = []
        invalid_at: datetime | None = None
        for transition in machine.transitions:
            if (
                transition.current
                in {
                    BitvavoBookState.GAPPED,
                    BitvavoBookState.RESEED_REQUIRED,
                    BitvavoBookState.STALE,
                    BitvavoBookState.INVALID,
                }
                and invalid_at is None
            ):
                invalid_at = transition.at
            elif transition.current is BitvavoBookState.VALID and invalid_at is not None:
                gap_durations.append(max(0.0, (transition.at - invalid_at).total_seconds()))
                invalid_at = None
        mids = [float(row["mid"]) for row in features[market]]
        returns = pybroker_returns(workspace, mids) if mids else []
        finite_returns = [value for value in returns if value is not None]
        all_returns.append(
            {index: value for index, value in enumerate(returns) if value is not None}
        )
        reference_validation: dict[str, Any] | None = None
        latest = features[market][-1] if features[market] else None
        if latest:
            best_bid = float(latest["best_bid"])
            best_ask = float(latest["best_ask"])
            reference_validation = {
                "vectorbt": vectorbt_book_invariants(workspace, best_bid, best_ask),
                "nautilus_trader": nautilus_top_of_book(
                    workspace,
                    best_bid,
                    best_ask,
                    1.0,
                    1.0,
                ),
                "pybroker_return_observations": sum(value is not None for value in returns),
                "lean_sharpe_ratio": (
                    lean_sharpe_ratio(
                        workspace,
                        statistics.fmean(finite_returns),
                        statistics.stdev(finite_returns),
                    )
                    if len(finite_returns) >= 2 and statistics.stdev(finite_returns) > 0
                    else None
                ),
            }
        assets[market] = {
            **machine.snapshot(last_known.get(market) or utc_now()),
            "first_known_at": utc_iso(first_known[market]) if market in first_known else None,
            "last_known_at": utc_iso(last_known[market]) if market in last_known else None,
            "total_observed_seconds": window,
            "valid_seconds": valid_seconds,
            "book_valid_fraction": valid_seconds / window if window else 0.0,
            "feature_count": len(features[market]),
            "gap_duration_seconds": gap_durations,
            "reseed_latency_seconds_p50": (
                sorted(gap_durations)[len(gap_durations) // 2] if gap_durations else None
            ),
            "data_counts": dict(data_counts[market]),
            "valid_intervals": valid_intervals(machine),
            "transition_trace": transition_trace(machine),
            "latest_feature": latest,
            "reference_validation": reference_validation,
            "historical_missing_evidence_fabricated": False,
            "execution_authority": False,
            "orders_generated": 0,
        }

    qlib_aggregate = (
        qlib_sum_by_index(
            workspace,
            all_returns,
            sorted({index for row in all_returns for index in row}),
        )
        if any(all_returns)
        else {}
    )
    body = {
        "schema_version": "bitvavo_l2_source_neutral_replay_v2",
        "reconstruction_version": L2_RECONSTRUCTION_VERSION,
        "generated_at": utc_iso(),
        "assets": assets,
        "source_segment_bounds": segment_bounds,
        "causal_reorder_policy": {
            "scope": "IDENTICAL_KNOWN_AT_BATCH_ONLY",
            "maximum_batch_events": 20_000,
            "cross_known_at_reordering": False,
            "source_nonce_required": True,
            "observable_stats": dict(reorder_stats),
        },
        "auxiliary_snapshot_root": str(auxiliary_snapshot_root)
        if auxiliary_snapshot_root
        else None,
        "qlib_indexed_return_aggregate": qlib_aggregate,
        "freqtrade_feature_cadence_seconds": feature_cadence_seconds,
        "raw_overwritten": False,
        "v1_overwritten": False,
        "historical_missing_evidence_fabricated": False,
        "private_exchange_requests": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return {**body, "replay_hash": stable_hash(body)}


def build_recovery_report(
    *,
    workspace: Path,
    v1_snapshot_root: Path,
    raw_root: Path,
    auxiliary_snapshot_root: Path | None = None,
) -> dict[str, Any]:
    forensics = v1_forensics(v1_snapshot_root)
    replay = replay_source_neutral_l2(
        workspace=workspace,
        raw_root=raw_root,
        auxiliary_snapshot_root=auxiliary_snapshot_root,
    )
    v1 = forensics["baseline"]
    comparisons = {}
    for market in PRIMARY_MARKETS:
        comparisons[market] = {
            "v1_closed_hour_intervals": v1["assets"][market]["closed_intervals"],
            "v1_valid_hour_intervals": v1["assets"][market]["valid_intervals"],
            "v1_valid_fraction": v1["assets"][market]["book_valid_fraction"],
            "v2_exact_valid_interval_count": replay["assets"][market]["valid_interval_count"],
            "v2_exact_valid_seconds": replay["assets"][market]["valid_seconds"],
            "v2_valid_fraction": replay["assets"][market]["book_valid_fraction"],
            "comparison_scope": "V1_HOURLY_POLICY_VS_V2_EXACT_PROSPECTIVE_EVIDENCE",
        }
    body = {
        "schema_version": BITVAVO_L2_RECOVERY_REPORT_SCHEMA,
        "generated_at": utc_iso(),
        "official_protocol_matrix": protocol_semantics_matrix(),
        "v1_forensics": forensics,
        "v2_replay": replay,
        "v1_v2_comparison": comparisons,
        "health_separation": {
            "execution_orderflow_health": "UNCHANGED_AND_INDEPENDENT",
            "flow_research_health": "TRADES_REMAIN_VALID_INDEPENDENT_OF_L2",
            "l2_research_health": "VALID_ONLY_INSIDE_V2_EXPLICIT_INTERVALS",
        },
        "readiness_policy": {
            "p1_2_2_authoritative": True,
            "thresholds_lowered": False,
            "v2_auto_promotion": False,
        },
        "next_exact_action": "CONTINUE_PROSPECTIVE_COLLECTION",
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return {**body, "report_hash": stable_hash(body)}


__all__ = [
    "BITVAVO_L2_RECOVERY_REPORT_SCHEMA",
    "build_recovery_report",
    "protocol_semantics_matrix",
    "replay_source_neutral_l2",
    "v1_forensics",
]
