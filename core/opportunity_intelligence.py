"""Fail-safe opportunity intelligence for the existing event-driven engine.

The deterministic playbook remains the signal authority.  This module owns
deduplication, immutable decision snapshots and optional *shadow-only* models.
Missing, stale or incompatible models always fall back to the rule engine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config.settings import Settings
from core.intelligence_drift import build_intelligence_drift_report
from execution.canonical_state import replay_execution_events
from utils.common import (
    append_jsonl,
    atomic_write_json,
    stable_hash,
    utc_now,
)

FEATURE_SCHEMA_VERSION = "opportunity_features_v1"
LABEL_SCHEMA_VERSION = "opportunity_labels_v1"
PIPELINE_SMOKE_ROWS = 100
MINIMUM_SHADOW_EVALUATION_ROWS = 500
MINIMUM_CLASS_ROWS = 100
CANONICAL_LABEL_HORIZON = timedelta(hours=24)
ACTIVE_SWING_FORWARD_STATES = {
    "ACTIONABLE",
    "NEAR_ENTRY",
    "APPROACHING",
    "EARLY_MOMENTUM_ALERT",
    "PULLBACK_PENDING",
    "FIRST_PULLBACK_AFTER_IMPULSE",
    "EXTENDED_MOVE_WAIT_FOR_PULLBACK",
}
_SETUP_TIMEFRAME_SECONDS = {
    "15m": 900,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
    "1W": 604800,
}

CONFIRMATION_GROUPS: dict[str, tuple[str, ...]] = {
    "structure": ("valid_structure", "absorption_or_sweep"),
    "participation": ("relative_volume", "trade_intensity"),
    # Taker flow and CVD are both derived from actual executions.  Keep them
    # in one group so the same prints cannot count twice.
    "executed_flow": ("taker_flow", "cvd"),
    # OFI is derived from L2 depth changes in the Bitvavo feed.  It belongs
    # with the other book-pressure facts, not with executed trade flow.
    "orderbook_flow": (
        "ofi",
        "mlobi",
        "microprice",
        "replenishment",
    ),
}


def confirmation_independence(
    confirmations: Mapping[str, Any],
) -> dict[str, Any]:
    """Collapse correlated confirmations into economic evidence groups."""

    groups = {
        name: any(bool(confirmations.get(key)) for key in members)
        for name, members in CONFIRMATION_GROUPS.items()
    }
    raw_count = sum(bool(value) for value in confirmations.values())
    independent_count = sum(groups.values())
    return {
        "groups": groups,
        "raw_count": raw_count,
        "independent_count": independent_count,
        "independence_ratio": (
            independent_count / raw_count if raw_count else 0.0
        ),
    }


def estimate_roundtrip_economics(
    *,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    take_profit_2: float,
    spread_bps: float | None,
    observed_slippage_bps: float | None,
    maker_fee_bps: float = 15.0,
    taker_fee_bps: float = 25.0,
    minimum_profit_buffer_bps: float = 5.0,
    conservative_win_probability: float | None = None,
) -> dict[str, Any]:
    """Estimate conservative all-in economics without predicting direction."""

    entry = max(float(entry_price), 1e-12)
    spread = max(0.0, float(spread_bps or 0.0))
    observed = max(0.0, float(observed_slippage_bps or 0.0))
    # P75 proxy until enough immutable fills exist for a learned quantile model.
    slippage_p75 = max(observed * 1.25, observed + 1.0)
    roundtrip_cost_bps = (
        max(0.0, float(maker_fee_bps))
        + max(0.0, float(taker_fee_bps))
        + spread
        + slippage_p75 * 2.0
    )
    stop_bps = abs(entry - float(stop_loss)) / entry * 10_000.0
    tp1_bps = max(0.0, float(take_profit_1) - entry) / entry * 10_000.0
    tp2_bps = max(0.0, float(take_profit_2) - entry) / entry * 10_000.0
    net_tp1_bps = tp1_bps - roundtrip_cost_bps
    net_tp2_bps = tp2_bps - roundtrip_cost_bps
    net_risk_bps = stop_bps + roundtrip_cost_bps
    cost_to_target_2_ratio = (
        roundtrip_cost_bps / tp2_bps if tp2_bps > 0 else None
    )
    edge_cost_ratio_1 = (
        tp1_bps / roundtrip_cost_bps if roundtrip_cost_bps > 0 else None
    )
    edge_cost_ratio_2 = (
        tp2_bps / roundtrip_cost_bps if roundtrip_cost_bps > 0 else None
    )
    probability = (
        max(0.0, min(1.0, float(conservative_win_probability)))
        if conservative_win_probability is not None
        else None
    )
    expected_net_value_bps = (
        probability * net_tp2_bps
        - (1.0 - probability) * net_risk_bps
        if probability is not None
        else None
    )
    positive_exit_paths = [
        name
        for name, value in (
            ("TP1", net_tp1_bps),
            ("TP2", net_tp2_bps),
        )
        if value > minimum_profit_buffer_bps
    ]
    # Outcome probabilities are deliberately not invented.  Until clean
    # prospective labels exist, this gate only asks whether at least one
    # bounded exit path can clear estimated round-trip costs.  Expected value
    # remains uncalibrated and therefore cannot raise size or execution caps.
    return {
        "cost_model": "CONSERVATIVE_RULE_P75_V1",
        "slippage_p75_bps": round(slippage_p75, 8),
        "roundtrip_cost_bps": round(roundtrip_cost_bps, 8),
        "minimum_profit_buffer_bps": float(minimum_profit_buffer_bps),
        "stop_bps": round(stop_bps, 8),
        "gross_target_1_bps": round(tp1_bps, 8),
        "gross_target_2_bps": round(tp2_bps, 8),
        "net_target_1_bps": round(net_tp1_bps, 8),
        "net_target_2_bps": round(net_tp2_bps, 8),
        "net_rr_target_1": round(net_tp1_bps / net_risk_bps, 8)
        if net_risk_bps > 0
        else None,
        "net_rr_target_2": round(net_tp2_bps / net_risk_bps, 8)
        if net_risk_bps > 0
        else None,
        "cost_to_stop_ratio": round(roundtrip_cost_bps / stop_bps, 8)
        if stop_bps > 0
        else None,
        "cost_to_target_2_ratio": round(cost_to_target_2_ratio, 8)
        if cost_to_target_2_ratio is not None
        else None,
        "edge_cost_ratio_target_1": round(edge_cost_ratio_1, 8)
        if edge_cost_ratio_1 is not None
        else None,
        "edge_cost_ratio_target_2": round(edge_cost_ratio_2, 8)
        if edge_cost_ratio_2 is not None
        else None,
        "positive_exit_paths": positive_exit_paths,
        "positive_after_costs": bool(positive_exit_paths),
        "rule_based_ev_status": (
            "CONSERVATIVE_FIXED_PRIOR"
            if probability is not None
            else "UNCALIBRATED_OUTCOME_PROBABILITIES"
        ),
        "conservative_win_probability": probability,
        "expected_net_value_bps": (
            round(expected_net_value_bps, 8)
            if expected_net_value_bps is not None
            else None
        ),
        "may_raise_risk_caps": False,
    }


def _utc_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def _snapshot_point_in_time_metadata(
    opportunity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return explicit timing/provenance without backfilling legacy records.

    A newly frozen snapshot is an immutable as-of observation.  The event time
    identifies the newest source event represented by the values; available-at
    identifies when the complete vector was observable to the decision engine.
    "is_final" means the vector itself is frozen and will not be revised, not
    that future market prices are known.
    """

    decision = _utc_timestamp(
        opportunity.get("detected_at") or opportunity.get("decision_timestamp")
    ) or utc_now()
    event = _utc_timestamp(
        opportunity.get("feature_event_time")
        or opportunity.get("market_event_ts")
        or opportunity.get("setup_detected_ts")
        or decision
    )
    available = _utc_timestamp(
        opportunity.get("feature_available_at")
        or opportunity.get("last_updated_at")
        or decision
    )
    provenance = opportunity.get("feature_provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        provenance = {
            "producer": "core.opportunity_intelligence.freeze_feature_snapshot",
            "market_source": "EVENT_DRIVEN_REALTIME_AS_OF"
            if opportunity.get("realtime_inputs")
            else "OPPORTUNITY_DECISION_RECORD",
            "context_source": "CLOSED_CANDLE_CONTEXT"
            if opportunity.get("context_timeframe")
            else "NOT_PRESENT",
            "synthetic_data_used": bool(opportunity.get("synthetic_data_used")),
        }
    timing_valid = bool(
        event is not None
        and available is not None
        and event <= available <= decision
    )
    return {
        "event_time": event.isoformat() if event is not None else None,
        "available_at": available.isoformat() if available is not None else None,
        "is_final": bool(
            opportunity.get("feature_is_final", timing_valid)
        ),
        "provenance": dict(provenance),
        "point_in_time_timing_valid": timing_valid,
    }


def freeze_feature_snapshot(opportunity: Mapping[str, Any]) -> dict[str, Any]:
    """Create a compact immutable point-in-time feature vector."""

    realtime = dict(opportunity.get("realtime_inputs") or {})
    economics = dict(opportunity.get("execution_economics") or {})
    values = {
        "market": opportunity.get("market"),
        "strategy_id": opportunity.get("strategy_id"),
        "strategy_dna_hash": opportunity.get("strategy_dna_hash"),
        "family": opportunity.get("family"),
        "entry_timeframe": opportunity.get("entry_timeframe"),
        "setup_timeframe": opportunity.get("setup_timeframe"),
        "structural_timeframe": opportunity.get("structural_timeframe"),
        "signal_status": opportunity.get("signal_status"),
        "context_timeframe": opportunity.get("context_timeframe"),
        "observation_timeframe": opportunity.get(
            "observation_timeframe"
        ),
        "trade_type": opportunity.get("trade_type"),
        "market_mode": opportunity.get("market_mode"),
        "weighted_timeframe_score": opportunity.get(
            "weighted_timeframe_score"
        ),
        "fast_timeframe_score": opportunity.get("fast_timeframe_score"),
        "slow_timeframe_score": opportunity.get("slow_timeframe_score"),
        "timeframe_disagreement": opportunity.get(
            "timeframe_disagreement"
        ),
        "macro_regime": opportunity.get("macro_regime"),
        "macro_1d_trend_up": (
            opportunity.get("macro_long_lock") or {}
        ).get("btc_1d_trend_up"),
        "macro_4h_trend_up": (
            opportunity.get("macro_long_lock") or {}
        ).get("btc_4h_trend_up"),
        "macro_long_allowed": (
            opportunity.get("macro_long_lock") or {}
        ).get("long_allowed"),
        "score": opportunity.get("score"),
        "entry_price": opportunity.get("entry_price"),
        "stop_loss": opportunity.get("stop_loss"),
        "take_profit_1": opportunity.get("take_profit_1"),
        "take_profit_2": opportunity.get("take_profit_2"),
        "independent_confirmation_count": opportunity.get(
            "independent_confirmation_count"
        ),
        "return_1m": realtime.get("return_1m"),
        "return_5m": realtime.get("return_5m"),
        "return_15m": realtime.get("return_15m"),
        "relative_volume_1m": realtime.get("relative_volume_1m"),
        "trade_intensity_1m": realtime.get("trade_intensity_1m"),
        "taker_buy_ratio_1m": realtime.get("taker_buy_ratio_1m"),
        "cvd_quote_eur_1m": realtime.get("cvd_quote_eur_1m"),
        "ofi_1m": realtime.get("ofi_1m"),
        "mlobi_top_10": realtime.get("mlobi_top_10"),
        "microprice_edge_bps": realtime.get("microprice_edge_bps"),
        "spread_bps": realtime.get("spread_bps"),
        "estimated_buy_slippage_bps": realtime.get(
            "estimated_buy_slippage_bps"
        ),
        "bid_depth_eur_top_10": realtime.get("bid_depth_eur_top_10"),
        "roundtrip_cost_bps": economics.get("roundtrip_cost_bps"),
        "net_target_1_bps": economics.get("net_target_1_bps"),
        "net_target_2_bps": economics.get("net_target_2_bps"),
    }
    decision_timestamp = opportunity.get("detected_at") or utc_now().isoformat()
    body = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "decision_timestamp": decision_timestamp,
        **_snapshot_point_in_time_metadata(
            {**dict(opportunity), "detected_at": decision_timestamp}
        ),
        "values": values,
    }
    return {**body, "feature_hash": stable_hash(body, length=64)}


def _active_swing_setup_identity(
    *,
    market: str,
    strategy_dna_hash: str,
    signal_timestamp: datetime,
    setup_timeframe: Any,
    setup_origin_timestamp: Any = None,
) -> str:
    origin = _utc_timestamp(setup_origin_timestamp)
    if origin is None:
        seconds = _SETUP_TIMEFRAME_SECONDS.get(str(setup_timeframe), 900)
        epoch = int(signal_timestamp.timestamp())
        origin = datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)
    return stable_hash(
        {
            "market": market,
            "strategy_dna_hash": strategy_dna_hash,
            "setup_origin_timestamp": origin.isoformat(),
            "setup_timeframe": str(setup_timeframe or ""),
        },
        length=64,
    )


def record_active_swing_forward_snapshots(
    settings: Settings,
    candidates: Iterable[Mapping[str, Any]],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist one immutable decision snapshot per natural tactical setup.

    The first observation of a stable market/strategy/signal identity wins.
    Repeated 15-minute scans therefore update observability elsewhere without
    silently creating multiple economic labels for the same setup.
    """

    decision = observed_at or utc_now()
    if decision.tzinfo is None:
        raise ValueError("active-swing snapshot decision time must be timezone-aware")
    decision = decision.astimezone(UTC)
    root = settings.paths.output_dir / "intelligence"
    ledger = root / "active_swing_decision_snapshots.jsonl"
    index_path = root / "active_swing_snapshot_index.json"
    known: set[str] = set()
    ledger_events = _read_jsonl(ledger)
    for event in ledger_events:
        snapshot = dict(event.get("feature_snapshot") or {})
        values = dict(snapshot.get("values") or {})
        signal = _utc_timestamp(snapshot.get("event_time"))
        market = str(values.get("market") or "")
        strategy_dna = str(values.get("strategy_dna_hash") or "")
        if market and strategy_dna and signal is not None:
            known.add(
                _active_swing_setup_identity(
                    market=market,
                    strategy_dna_hash=strategy_dna,
                    signal_timestamp=signal,
                    setup_timeframe=(
                        values.get("setup_timeframe")
                        or values.get("context_timeframe")
                    ),
                )
            )
        else:
            identity = str(event.get("opportunity_id") or "")
            if identity:
                known.add(identity)

    written = 0
    duplicates = 0
    ignored_non_candidates = 0
    rejected: dict[str, int] = defaultdict(int)
    for source in candidates:
        candidate = dict(source)
        source_status = str(candidate.get("status") or "")
        if source_status not in ACTIVE_SWING_FORWARD_STATES:
            ignored_non_candidates += 1
            continue
        market = str(candidate.get("market") or "")
        strategy_id = str(candidate.get("strategy") or "")
        strategy_dna = str(candidate.get("strategy_dna_hash") or "")
        signal = _utc_timestamp(candidate.get("signal_timestamp"))
        entry = _finite_decimal(
            candidate.get("current_price") or candidate.get("trigger")
        )
        stop = _finite_decimal(candidate.get("stop"))
        target_1 = _finite_decimal(candidate.get("target_1"))
        target_2 = _finite_decimal(candidate.get("target_2"))
        if not market or not strategy_id or not strategy_dna or signal is None:
            rejected["IDENTITY_OR_SIGNAL_TIME_MISSING"] += 1
            continue
        if signal > decision:
            rejected["SIGNAL_AFTER_DECISION_TIME"] += 1
            continue
        if (
            entry is None
            or stop is None
            or target_1 is None
            or target_2 is None
            or not (Decimal("0") < stop < entry < target_1 <= target_2)
        ):
            rejected["LABEL_LEVELS_INVALID"] += 1
            continue
        setup_timeframe = candidate.get("confirmation_timeframe") or candidate.get(
            "setup_timeframe"
        )
        identity = _active_swing_setup_identity(
            market=market,
            strategy_dna_hash=strategy_dna,
            signal_timestamp=signal,
            setup_timeframe=setup_timeframe,
            setup_origin_timestamp=candidate.get("setup_origin_timestamp"),
        )
        if identity in known:
            duplicates += 1
            continue
        fee_fraction = _finite_decimal(candidate.get("estimated_fee_fraction"))
        slippage_bps = _finite_decimal(candidate.get("estimated_slippage_bps"))
        roundtrip_cost_bps = (
            fee_fraction * Decimal("20000")
            + slippage_bps * Decimal("2")
            if fee_fraction is not None and slippage_bps is not None
            else None
        )
        gross_target_1_bps = (target_1 / entry - Decimal("1")) * Decimal("10000")
        gross_target_2_bps = (target_2 / entry - Decimal("1")) * Decimal("10000")
        normalized = {
            "market": market,
            "strategy_id": strategy_id,
            "strategy_dna_hash": strategy_dna,
            "family": candidate.get("family"),
            "entry_timeframe": candidate.get("entry_timeframe")
            or candidate.get("timeframe"),
            "setup_timeframe": candidate.get("confirmation_timeframe"),
            "structural_timeframe": candidate.get("regime_timeframe"),
            "signal_status": source_status,
            "context_timeframe": candidate.get("confirmation_timeframe")
            or candidate.get("regime_timeframe"),
            "observation_timeframe": candidate.get("entry_timeframe")
            or candidate.get("timeframe"),
            "trade_type": candidate.get("trade_type"),
            "market_mode": candidate.get("market_mode"),
            "weighted_timeframe_score": candidate.get(
                "weighted_timeframe_score"
            ),
            "fast_timeframe_score": candidate.get("fast_timeframe_score"),
            "slow_timeframe_score": candidate.get("slow_timeframe_score"),
            "timeframe_disagreement": candidate.get("timeframe_disagreement"),
            "macro_regime": candidate.get("regime"),
            "score": candidate.get("score"),
            "entry_price": str(entry),
            "stop_loss": str(stop),
            "take_profit_1": str(target_1),
            "take_profit_2": str(target_2),
            "execution_economics": {
                "roundtrip_cost_bps": (
                    str(roundtrip_cost_bps)
                    if roundtrip_cost_bps is not None
                    else None
                ),
                "net_target_1_bps": (
                    str(gross_target_1_bps - roundtrip_cost_bps)
                    if roundtrip_cost_bps is not None
                    else None
                ),
                "net_target_2_bps": (
                    str(gross_target_2_bps - roundtrip_cost_bps)
                    if roundtrip_cost_bps is not None
                    else None
                ),
            },
            "detected_at": decision.isoformat(),
            "feature_event_time": signal.isoformat(),
            "feature_available_at": decision.isoformat(),
            "feature_is_final": True,
            "feature_provenance": {
                "producer": (
                    "core.opportunity_intelligence."
                    "record_active_swing_forward_snapshots"
                ),
                "market_source": "CLOSED_CANDLE_ACTIVE_SWING_SCAN",
                "context_source": "CAUSAL_MULTI_TIMEFRAME_AS_OF",
                "synthetic_data_used": False,
            },
        }
        snapshot = freeze_feature_snapshot(normalized)
        append_jsonl(
            ledger,
            {
                "schema_version": "active_swing_forward_snapshot_v1",
                "recorded_at": decision.isoformat(),
                "opportunity_id": identity,
                "source_opportunity_id": candidate.get("opportunity_id"),
                "source_status": candidate.get("status"),
                "feature_snapshot": snapshot,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        known.add(identity)
        written += 1
    atomic_write_json(
        index_path,
        {
            "schema_version": "active_swing_snapshot_index_v1",
            "updated_at": decision.isoformat(),
            "snapshot_count": len(known),
            "ledger_record_count": len(ledger_events) + written,
            "snapshot_ids": sorted(known),
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    return {
        "status": "RECORDED" if written else "NO_NEW_SNAPSHOTS",
        "ledger": str(ledger),
        "snapshot_count": len(known),
        "ledger_record_count": len(ledger_events) + written,
        "written_this_scan": written,
        "duplicates_this_scan": duplicates,
        "ignored_non_candidates_this_scan": ignored_non_candidates,
        "rejected_this_scan": dict(sorted(rejected.items())),
        "financial_state_changed": False,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def deduplicate_opportunities(
    opportunities: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one economically best playbook for each market event cluster."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in opportunities:
        row = dict(source)
        event_time = str(row.get("detected_at") or "")
        try:
            parsed = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            parsed = parsed.astimezone(UTC).replace(
                minute=(parsed.minute // 15) * 15,
                second=0,
                microsecond=0,
            )
            bucket = parsed.isoformat()
        except ValueError:
            bucket = event_time[:16]
        cluster_id = stable_hash(
            [row.get("market"), "LONG", bucket], length=40
        )
        row["cluster_id"] = cluster_id
        grouped[cluster_id].append(row)

    selected: list[dict[str, Any]] = []
    state_rank = {"ENTRY_READY": 4, "ARMED": 3, "WATCHING": 2, "DISCOVERED": 1}
    for cluster_id, rows in grouped.items():
        winner = max(
            rows,
            key=lambda row: (
                state_rank.get(str(row.get("state")), 0),
                bool(
                    (row.get("execution_economics") or {}).get(
                        "positive_after_costs"
                    )
                ),
                float(row.get("score") or 0.0),
            ),
        )
        winner = dict(winner)
        winner["cluster_id"] = cluster_id
        # This is the canonical economic-bet identity: overlapping playbooks
        # on the same directional market move share one risk budget/order.
        winner["economic_bet_id"] = cluster_id
        winner["cluster_size"] = len(rows)
        winner["deduplication_suppressed_count"] = len(rows) - 1
        winner["alternative_playbooks"] = sorted(
            str(row.get("playbook_id"))
            for row in rows
            if row.get("opportunity_id") != winner.get("opportunity_id")
        )
        trace = dict(winner.get("decision_trace") or {})
        trace["deduplicated_decision"] = "SELECTED_CLUSTER_WINNER"
        trace["duplicates_suppressed"] = len(rows) - 1
        winner["decision_trace"] = trace
        winner["feature_snapshot"] = freeze_feature_snapshot(winner)
        selected.append(winner)
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("state")) == "ENTRY_READY",
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )


def bayesian_playbook_weights(
    rows: Iterable[Mapping[str, Any]], *, prior_strength: float = 20.0
) -> list[dict[str, Any]]:
    """Shrink sparse family/timeframe/regime expectancy toward the global mean."""

    clean = [
        dict(row)
        for row in rows
        if row.get("net_return_r") is not None
    ]
    if not clean:
        return []
    global_mean = sum(float(row["net_return_r"]) for row in clean) / len(clean)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in clean:
        key = (
            str(row.get("family") or "UNKNOWN"),
            str(row.get("timeframe") or "UNKNOWN"),
            str(row.get("regime") or "UNKNOWN"),
        )
        groups[key].append(float(row["net_return_r"]))
    output = []
    for key, values in sorted(groups.items()):
        local_mean = sum(values) / len(values)
        shrunk = (
            prior_strength * global_mean + len(values) * local_mean
        ) / (prior_strength + len(values))
        multiplier = max(0.5, min(1.0, 0.75 + shrunk * 0.25))
        output.append(
            {
                "family": key[0],
                "timeframe": key[1],
                "regime": key[2],
                "sample_size": len(values),
                "local_mean_net_r": local_mean,
                "shrunk_mean_net_r": shrunk,
                "ranking_multiplier": multiplier,
                "can_raise_risk_caps": False,
            }
        )
    return output


def intelligence_status(settings: Settings) -> dict[str, Any]:
    root = settings.paths.output_dir / "intelligence"
    model_path = root / "model_status.json"
    legacy = (
        json.loads(model_path.read_text(encoding="utf-8"))
        if model_path.is_file()
        else {}
    )
    canonical_path = (
        settings.paths.output_dir / "ml" / "canonical_training_status.json"
    )
    canonical = (
        json.loads(canonical_path.read_text(encoding="utf-8"))
        if canonical_path.is_file()
        else {}
    )
    return {
        "status": canonical.get("status") or "DATA_COLLECTION",
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "pipeline_smoke_rows": PIPELINE_SMOKE_ROWS,
        "minimum_shadow_evaluation_rows": MINIMUM_SHADOW_EVALUATION_ROWS,
        "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        "models": (
            {
                "meta_labeler": "CANONICAL_LOGISTIC_SHADOW",
                "fill_slippage": "DATA_PENDING",
                "reversal_persistence": "DATA_PENDING",
                "anomaly_detector": "DATA_PENDING",
            }
            if canonical.get("model_registered") is True
            else None
        )
        or legacy.get("models")
        or {
            "meta_labeler": "DATA_PENDING",
            "fill_slippage": "DATA_PENDING",
            "reversal_persistence": "DATA_PENDING",
            "anomaly_detector": "DATA_PENDING",
        },
        "drift_monitor": legacy.get("drift_monitor")
        or {
            "status": "DATA_PENDING",
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
        },
        "canonical_pipeline": canonical or {"status": "NOT_RUN"},
        "legacy_compatibility_pipeline": {
            "status": legacy.get("status") or "NOT_PRESENT",
            "canonical_registration_permitted": False,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def score_shadow_opportunity(
    settings: Settings,
    opportunity: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one opportunity without changing its state, score or blockers."""

    root = settings.paths.output_dir / "intelligence"
    bundle_path = root / "model_bundle.joblib"
    snapshot = dict(
        opportunity.get("feature_snapshot")
        or freeze_feature_snapshot(opportunity)
    )
    if (
        not bundle_path.is_file()
        or snapshot.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
    ):
        return {
            "status": "DATA_PENDING",
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        }
    try:
        import joblib
        import pandas as pd

        bundle = joblib.load(bundle_path)
        if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("FEATURE_SCHEMA_MISMATCH")
        frame = pd.DataFrame([dict(snapshot.get("values") or {})])
        probability = float(
            bundle["logistic_meta_labeler"].predict_proba(frame)[0, 1]
        )
        transformed = bundle["preprocessor"].transform(frame)
        transformed = (
            transformed.toarray()
            if hasattr(transformed, "toarray")
            else transformed
        )
        anomaly = int(bundle["anomaly_detector"].predict(transformed)[0])
        return {
            "status": "SCORED_SHADOW",
            "authority": "SHADOW_ONLY",
            "p_net_profitable": probability,
            # A losing roundtrip is not automatically a false breakout.  This
            # stays unavailable until prospective MFE/invalidation labels exist.
            "p_false_breakout": None,
            "market_state_anomaly": anomaly == -1,
            "live_decision_influence": False,
            "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        }
    except Exception as exc:
        return {
            "status": "FALLBACK_RULE_ENGINE",
            "authority": "SHADOW_ONLY",
            "reason_code": type(exc).__name__,
            "live_decision_influence": False,
            "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        }


def write_intelligence_status(settings: Settings) -> dict[str, Any]:
    payload = {
        **intelligence_status(settings),
        "generated_at": utc_now().isoformat(),
        "reason": (
            "Legacy opportunities lack immutable decision snapshots; clean "
            "prospective labels are being collected before training."
        ),
    }
    path = settings.paths.output_dir / "intelligence" / "model_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return {**payload, "artifact": str(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    selected = Path(path)
    if not selected.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in selected.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _finite_decimal(value: Any) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _intent_signal_map(events: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Recover the canonical signal identity recorded with an order intent.

    Execution FILL events intentionally contain only venue/fill facts.  The
    durable strategy identity lives in ORDER_RESULT.record.intent, linked by
    ``intent_id``.  Reading the FILL payload alone therefore loses the signal
    identity and previously produced an empty training dataset even when
    prospectively snapshotted paper roundtrips existed.
    """

    identities: dict[str, str] = {}
    for event in events:
        payload = dict(event.get("payload") or {})
        candidates: list[Mapping[str, Any]] = [payload]
        record = payload.get("record")
        if isinstance(record, Mapping):
            intent = record.get("intent")
            if isinstance(intent, Mapping):
                candidates.append(intent)
        embedded_intent = payload.get("intent")
        if isinstance(embedded_intent, Mapping):
            candidates.append(embedded_intent)
        for candidate in candidates:
            intent_id = str(candidate.get("intent_id") or "")
            signal_id = str(candidate.get("signal_id") or "")
            if intent_id and signal_id:
                identities[intent_id] = signal_id
    return identities


SHADOW_LABEL_HORIZONS = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "12h": timedelta(hours=12),
    "24h": timedelta(hours=24),
}


def _load_shadow_label_frame(
    settings: Settings,
    market: str,
    cache: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    if market in cache:
        return cache[market]
    selected: pd.DataFrame | None = None
    for timeframe in ("15m", "1h"):
        path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
        if not path.is_file():
            continue
        try:
            frame = pd.read_parquet(path)
            if "timestamp" in frame.columns:
                frame["timestamp"] = pd.to_datetime(
                    frame["timestamp"],
                    utc=True,
                )
                frame = frame.set_index("timestamp")
            else:
                frame.index = pd.to_datetime(frame.index, utc=True)
            required = ["high", "low", "close"]
            selected = (
                frame.loc[:, required]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .loc[lambda value: ~value.index.duplicated(keep="last")]
                .sort_index()
            )
        except (KeyError, OSError, TypeError, ValueError):
            selected = None
        if selected is not None and not selected.empty:
            break
    cache[market] = selected
    return selected


def _shadow_episode_label(
    settings: Settings,
    snapshot: Mapping[str, Any],
    *,
    cache: dict[str, pd.DataFrame | None],
) -> dict[str, Any] | None:
    """Resolve one immutable decision against later closed market candles."""

    values = dict(snapshot.get("values") or {})
    market = str(values.get("market") or "")
    frame = _load_shadow_label_frame(settings, market, cache)
    if frame is None or frame.empty:
        return None
    try:
        decision = pd.Timestamp(snapshot.get("decision_timestamp"))
    except (TypeError, ValueError):
        return None
    if pd.isna(decision):
        return None
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    final_cutoff = decision + SHADOW_LABEL_HORIZONS["24h"]
    if frame.index[-1] < final_cutoff:
        return None
    entry = _finite_decimal(values.get("entry_price"))
    stop = _finite_decimal(values.get("stop_loss"))
    target_1 = _finite_decimal(values.get("take_profit_1"))
    target_2 = _finite_decimal(values.get("take_profit_2"))
    if (
        entry is None
        or stop is None
        or target_1 is None
        or target_2 is None
        or entry <= 0
        or stop >= entry
        or target_1 <= entry
        or target_2 < target_1
    ):
        return None
    path = frame.loc[(frame.index > decision) & (frame.index <= final_cutoff)]
    if path.empty:
        return None
    returns_by_horizon: dict[str, float | None] = {}
    for label, delta in SHADOW_LABEL_HORIZONS.items():
        horizon = path.loc[path.index <= decision + delta]
        returns_by_horizon[label] = (
            float(Decimal(str(horizon["close"].iloc[-1])) / entry - 1)
            if not horizon.empty
            else None
        )
    first_stop_at: str | None = None
    first_tp1_at: str | None = None
    first_tp2_at: str | None = None
    ambiguous_same_candle = False
    for timestamp, candle in path.iterrows():
        low = Decimal(str(candle["low"]))
        high = Decimal(str(candle["high"]))
        stop_hit = low <= stop
        tp1_hit = high >= target_1
        tp2_hit = high >= target_2
        if stop_hit and (tp1_hit or tp2_hit):
            ambiguous_same_candle = True
        if stop_hit and first_stop_at is None:
            first_stop_at = timestamp.isoformat()
        if tp1_hit and first_tp1_at is None:
            first_tp1_at = timestamp.isoformat()
        if tp2_hit and first_tp2_at is None:
            first_tp2_at = timestamp.isoformat()
    stop_before_tp1 = bool(
        first_stop_at
        and (first_tp1_at is None or first_stop_at <= first_tp1_at)
    )
    tp2_before_stop = bool(
        first_tp2_at
        and (first_stop_at is None or first_tp2_at < first_stop_at)
    )
    if ambiguous_same_candle and first_stop_at == first_tp2_at:
        tp2_before_stop = False
        stop_before_tp1 = True
    if tp2_before_stop:
        exit_price = target_2
        outcome = "TP2_BEFORE_STOP"
    elif stop_before_tp1:
        exit_price = stop
        outcome = "STOP_BEFORE_TP1"
    else:
        exit_price = Decimal(str(path["close"].iloc[-1]))
        outcome = "TIME_EXIT_24H"
    roundtrip_cost_bps = _finite_decimal(values.get("roundtrip_cost_bps")) or Decimal("0")
    gross_return = exit_price / entry - 1
    net_return = gross_return - roundtrip_cost_bps / Decimal("10000")
    initial_risk_fraction = abs(entry - stop) / entry
    mfe = Decimal(str(path["high"].max())) / entry - 1
    mae = Decimal(str(path["low"].min())) / entry - 1
    return {
        "net_pnl_eur": None,
        "net_profitable": int(net_return > 0),
        "net_return_r": (
            float(net_return / initial_risk_fraction)
            if initial_risk_fraction > 0
            else None
        ),
        "net_return_fraction": float(net_return),
        "gross_return_fraction": float(gross_return),
        "false_breakout": int(stop_before_tp1),
        "evidence_source": "SHADOW_CLOSED_CANDLE",
        "label_horizon_hours": 24,
        "returns_by_horizon": returns_by_horizon,
        "maximum_favorable_excursion": float(mfe),
        "maximum_adverse_excursion": float(mae),
        "tp1_hit": first_tp1_at is not None,
        "tp2_hit": first_tp2_at is not None,
        "stop_hit": first_stop_at is not None,
        "first_tp1_at": first_tp1_at,
        "first_tp2_at": first_tp2_at,
        "first_stop_at": first_stop_at,
        "tp2_before_stop": tp2_before_stop,
        "outcome": outcome,
        "same_candle_path_ambiguous": ambiguous_same_candle,
        "label_uses_future_features": False,
        "label_start": decision.isoformat(),
        "label_end": final_cutoff.isoformat(),
        "label_available_at": final_cutoff.isoformat(),
    }


def _training_row_point_in_time(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only evidence recorded in the immutable decision snapshot."""

    decision = _utc_timestamp(snapshot.get("decision_timestamp"))
    event = _utc_timestamp(snapshot.get("event_time"))
    available = _utc_timestamp(snapshot.get("available_at"))
    provenance = snapshot.get("provenance")
    timing_valid = bool(
        decision is not None
        and event is not None
        and available is not None
        and event <= available <= decision
    )
    canonical_ready = bool(
        timing_valid
        and snapshot.get("is_final") is True
        and isinstance(provenance, Mapping)
        and bool(provenance)
        and snapshot.get("feature_hash")
    )
    return {
        "event_time": event.isoformat() if event is not None else None,
        "available_at": available.isoformat() if available is not None else None,
        "is_final": snapshot.get("is_final") is True,
        "provenance": dict(provenance) if isinstance(provenance, Mapping) else None,
        "canonical_point_in_time_ready": canonical_ready,
    }


def build_training_dataset(settings: Settings) -> dict[str, Any]:
    """Build one leakage-safe row per prospectively snapshotted cluster.

    Legacy lifecycle records intentionally remain excluded because they do not
    contain immutable decision-time features.
    """

    lifecycle_paths = (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl",
        settings.paths.output_dir
        / "intelligence"
        / "active_swing_decision_snapshots.jsonl",
    )
    snapshots: dict[str, dict[str, Any]] = {}
    for lifecycle_path in lifecycle_paths:
        for event in _read_jsonl(lifecycle_path):
            if (
                lifecycle_path.name == "active_swing_decision_snapshots.jsonl"
                and str(event.get("source_status") or "")
                not in ACTIVE_SWING_FORWARD_STATES
            ):
                continue
            snapshot = event.get("feature_snapshot")
            identity = str(event.get("opportunity_id") or "")
            if (
                identity
                and isinstance(snapshot, Mapping)
                and snapshot.get("feature_schema_version")
                == FEATURE_SCHEMA_VERSION
            ):
                snapshots.setdefault(identity, dict(snapshot))

    ledgers = (
        settings.paths.checkpoints_dir / "live_execution.jsonl",
        settings.paths.output_dir
        / "paper"
        / "event_driven_playbook_execution.jsonl",
    )
    fills_by_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ledger in ledgers:
        evidence = "LIVE" if "live_execution" in ledger.name else "PAPER"
        events = _read_jsonl(ledger)
        intent_signals = _intent_signal_map(events)
        if evidence == "LIVE":
            canonical = replay_execution_events(events)
            for fill_id, fill in canonical.fills.items():
                signal_id = str(
                    fill.signal_id
                    or intent_signals.get(fill.intent_id)
                    or ""
                )
                if not signal_id:
                    continue
                realized = canonical.realized_pnl_events.get(fill_id)
                fills_by_signal[signal_id].append(
                    {
                        "fill_id": fill.fill_id,
                        "intent_id": fill.intent_id,
                        "market": fill.market,
                        "side": fill.side,
                        "price": str(fill.price),
                        "quantity": str(fill.quantity),
                        "fee_eur": str(fill.fee_eur),
                        "filled_at": fill.filled_at.isoformat(),
                        "evidence_source": "LIVE_CANONICAL_EXECUTION_STATE",
                        "canonical_state_hash": canonical.state_hash,
                        "canonical_realized_pnl_eur": (
                            str(realized.realized_pnl_eur)
                            if realized is not None
                            and realized.realized_pnl_eur is not None
                            else None
                        ),
                        "canonical_realized_pnl_complete": bool(
                            realized is not None and realized.complete
                        ),
                    }
                )
            continue
        for event in events:
            if event.get("event_type") != "FILL":
                continue
            payload = dict(event.get("payload") or {})
            signal_id = str(
                payload.get("signal_id")
                or intent_signals.get(str(payload.get("intent_id") or ""))
                or ""
            )
            if signal_id:
                fills_by_signal[signal_id].append(
                    {
                        **payload,
                        "filled_at": payload.get("filled_at")
                        or event.get("recorded_at"),
                        "evidence_source": evidence,
                    }
                )

    rows: list[dict[str, Any]] = []
    shadow_frame_cache: dict[str, pd.DataFrame | None] = {}
    shadow_labeled_count = 0
    fill_labeled_count = 0
    for signal_id, snapshot in snapshots.items():
        fills = fills_by_signal.get(signal_id) or []
        buys = [row for row in fills if str(row.get("side")).upper() == "BUY"]
        sells = [row for row in fills if str(row.get("side")).upper() == "SELL"]
        if not buys or not sells:
            shadow_label = _shadow_episode_label(
                settings,
                snapshot,
                cache=shadow_frame_cache,
            )
            if shadow_label is not None:
                rows.append(
                    {
                        "cluster_id": signal_id,
                        "signal_id": signal_id,
                        "decision_timestamp": snapshot.get(
                            "decision_timestamp"
                        ),
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_hash": snapshot.get("feature_hash"),
                        "features": dict(snapshot.get("values") or {}),
                        **_training_row_point_in_time(snapshot),
                        **shadow_label,
                    }
                )
                shadow_labeled_count += 1
            continue
        buy_notional = sum(
            (_finite_decimal(row.get("price")) or Decimal("0"))
            * (_finite_decimal(row.get("quantity")) or Decimal("0"))
            for row in buys
        )
        sell_notional = sum(
            (_finite_decimal(row.get("price")) or Decimal("0"))
            * (_finite_decimal(row.get("quantity")) or Decimal("0"))
            for row in sells
        )
        fees = sum(
            (_finite_decimal(row.get("fee_eur")) or Decimal("0"))
            for row in fills
        )
        canonical_exit_pnl = [
            _finite_decimal(row.get("canonical_realized_pnl_eur"))
            for row in sells
            if row.get("canonical_realized_pnl_complete") is True
        ]
        uses_complete_canonical_pnl = (
            bool(sells)
            and len(canonical_exit_pnl) == len(sells)
            and all(value is not None for value in canonical_exit_pnl)
        )
        net_pnl = (
            sum(
                (value for value in canonical_exit_pnl if value is not None),
                Decimal("0"),
            )
            if uses_complete_canonical_pnl
            else sell_notional - buy_notional - fees
        )
        values = dict(snapshot.get("values") or {})
        entry = _finite_decimal(values.get("entry_price"))
        stop = _finite_decimal(values.get("stop_loss"))
        quantity = sum(
            (_finite_decimal(row.get("quantity")) or Decimal("0"))
            for row in buys
        )
        initial_risk = (
            abs(entry - stop) * quantity
            if entry is not None and stop is not None
            else Decimal("0")
        )
        net_r = float(net_pnl / initial_risk) if initial_risk > 0 else None
        rows.append(
            {
                "cluster_id": signal_id,
                "signal_id": signal_id,
                "decision_timestamp": snapshot.get("decision_timestamp"),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_hash": snapshot.get("feature_hash"),
                "features": values,
                **_training_row_point_in_time(snapshot),
                "net_pnl_eur": float(net_pnl),
                "net_profitable": int(net_pnl > 0),
                "net_return_r": net_r,
                "false_breakout": None,
                "evidence_source": fills[0].get("evidence_source"),
                "economic_label_source": (
                    "CANONICAL_EXECUTION_STATE_REALIZED_PNL"
                    if uses_complete_canonical_pnl
                    else "LEGACY_FILL_PAIR_ACCOUNTING"
                ),
                "canonical_state_hash": fills[0].get(
                    "canonical_state_hash"
                ),
                "label_uses_future_features": False,
                "label_start": snapshot.get("decision_timestamp"),
                "label_end": max(
                    (
                        str(row.get("filled_at") or row.get("recorded_at") or "")
                        for row in sells
                    ),
                    default="",
                )
                or None,
            }
        )
        fill_labeled_count += 1
    rows.sort(key=lambda row: str(row.get("decision_timestamp") or ""))
    complete_signal_ids = {str(row["signal_id"]) for row in rows}
    incomplete_snapshot_ids = sorted(set(snapshots) - complete_signal_ids)
    canonical_feature_ready_incomplete_count = 0
    canonical_pending_label_horizon_count = 0
    canonical_label_horizon_mature_unresolved_count = 0
    canonical_incomplete_missing_timeframe_count = 0
    pending_label_due_times: list[pd.Timestamp] = []
    observed_at = pd.Timestamp(utc_now())
    for signal_id in incomplete_snapshot_ids:
        snapshot = snapshots[signal_id]
        point_in_time = _training_row_point_in_time(snapshot)
        if point_in_time["canonical_point_in_time_ready"] is not True:
            continue
        values = dict(snapshot.get("values") or {})
        if not (
            values.get("context_timeframe")
            or values.get("observation_timeframe")
        ):
            canonical_incomplete_missing_timeframe_count += 1
            continue
        canonical_feature_ready_incomplete_count += 1
        decision = _utc_timestamp(snapshot.get("decision_timestamp"))
        if decision is None:
            continue
        label_due_at = decision + CANONICAL_LABEL_HORIZON
        if label_due_at > observed_at:
            canonical_pending_label_horizon_count += 1
            pending_label_due_times.append(label_due_at)
        else:
            canonical_label_horizon_mature_unresolved_count += 1
    next_canonical_label_due_at = (
        min(pending_label_due_times).isoformat()
        if pending_label_due_times
        else None
    )
    feature_names = sorted(
        {
            str(name)
            for snapshot in snapshots.values()
            for name in dict(snapshot.get("values") or {})
        }
    )
    feature_non_missing = {
        name: sum(
            dict(snapshot.get("values") or {}).get(name) is not None
            for snapshot in snapshots.values()
        )
        for name in feature_names
    }
    snapshot_count = len(snapshots)
    feature_coverage = {
        name: {
            "non_missing": count,
            "missing": snapshot_count - count,
            "coverage_fraction": (
                count / snapshot_count if snapshot_count else 0.0
            ),
        }
        for name, count in feature_non_missing.items()
    }
    weighted_context_snapshot_count = feature_non_missing.get(
        "weighted_timeframe_score", 0
    )
    weighted_context_collection = {
        "status": (
            "COLLECTING_PROSPECTIVELY"
            if weighted_context_snapshot_count > 0
            else "PENDING_FIRST_POST_DEPLOYMENT_EPISODE"
        ),
        "snapshots_with_weighted_context": weighted_context_snapshot_count,
        "legacy_snapshots_without_weighted_context": (
            snapshot_count - weighted_context_snapshot_count
        ),
        "historical_snapshots_backfilled": False,
        "reason": (
            "Decision-time features are immutable; pre-deployment snapshots "
            "are not rewritten with values that were not recorded at t0."
        ),
    }
    canonical_point_in_time_rows = sum(
        row.get("canonical_point_in_time_ready") is True for row in rows
    )
    root = settings.paths.output_dir / "intelligence"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "opportunity_training_rows.json"
    csv_path = root / "opportunity_training_rows.csv"
    atomic_write_json(
        json_path,
        {
            "schema_version": LABEL_SCHEMA_VERSION,
            "generated_at": utc_now().isoformat(),
            "row_count": len(rows),
            "complete_labeled_episodes": len(rows),
            "fill_labeled_episodes": fill_labeled_count,
            "shadow_labeled_episodes": shadow_labeled_count,
            "incomplete_snapshotted_episodes": len(incomplete_snapshot_ids),
            "prospective_snapshot_count": snapshot_count,
            "fill_signal_count": len(fills_by_signal),
            "feature_coverage": feature_coverage,
            "weighted_context_collection": weighted_context_collection,
            "legacy_rows_excluded": True,
            "canonical_point_in_time_rows": canonical_point_in_time_rows,
            "canonical_feature_ready_incomplete_count": (
                canonical_feature_ready_incomplete_count
            ),
            "canonical_pending_label_horizon_count": (
                canonical_pending_label_horizon_count
            ),
            "canonical_label_horizon_mature_unresolved_count": (
                canonical_label_horizon_mature_unresolved_count
            ),
            "canonical_incomplete_missing_timeframe_count": (
                canonical_incomplete_missing_timeframe_count
            ),
            "next_canonical_label_due_at": next_canonical_label_due_at,
            "noncanonical_legacy_rows": len(rows) - canonical_point_in_time_rows,
            "historical_point_in_time_metadata_backfilled": False,
            "rows": rows,
        },
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "features"
            }
            | {
                f"feature__{key}": value
                for key, value in row["features"].items()
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)
    status = (
        "SHADOW_EVALUATION_READY"
        if len(rows) >= MINIMUM_SHADOW_EVALUATION_ROWS
        else "PIPELINE_SMOKE_READY"
        if len(rows) >= PIPELINE_SMOKE_ROWS
        else "DATA_PENDING"
    )
    return {
        "status": status,
        "row_count": len(rows),
        "complete_labeled_episodes": len(rows),
        "fill_labeled_episodes": fill_labeled_count,
        "shadow_labeled_episodes": shadow_labeled_count,
        "incomplete_snapshotted_episodes": len(incomplete_snapshot_ids),
        "prospective_snapshot_count": snapshot_count,
        "fill_signal_count": len(fills_by_signal),
        "feature_coverage": feature_coverage,
        "weighted_context_collection": weighted_context_collection,
        "canonical_point_in_time_rows": canonical_point_in_time_rows,
        "canonical_feature_ready_incomplete_count": (
            canonical_feature_ready_incomplete_count
        ),
        "canonical_pending_label_horizon_count": (
            canonical_pending_label_horizon_count
        ),
        "canonical_label_horizon_mature_unresolved_count": (
            canonical_label_horizon_mature_unresolved_count
        ),
        "canonical_incomplete_missing_timeframe_count": (
            canonical_incomplete_missing_timeframe_count
        ),
        "next_canonical_label_due_at": next_canonical_label_due_at,
        "noncanonical_legacy_rows": len(rows) - canonical_point_in_time_rows,
        "historical_point_in_time_metadata_backfilled": False,
        "pipeline_smoke_rows": PIPELINE_SMOKE_ROWS,
        "minimum_shadow_evaluation_rows": MINIMUM_SHADOW_EVALUATION_ROWS,
        "json_artifact": str(json_path),
        "csv_artifact": str(csv_path),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _canonical_row_failures(row: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    decision = _utc_timestamp(row.get("decision_timestamp"))
    event = _utc_timestamp(row.get("event_time"))
    available = _utc_timestamp(row.get("available_at"))
    label_start = _utc_timestamp(row.get("label_start"))
    label_end = _utc_timestamp(row.get("label_end"))
    if decision is None or event is None or available is None:
        failures.append("EXPLICIT_FEATURE_TIMESTAMPS_MISSING")
    elif not event <= available <= decision:
        failures.append("FEATURE_TIMING_NOT_CAUSAL")
    if row.get("is_final") is not True:
        failures.append("FEATURE_SNAPSHOT_NOT_FINAL")
    if not isinstance(row.get("provenance"), Mapping) or not row.get("provenance"):
        failures.append("FEATURE_PROVENANCE_MISSING")
    if row.get("label_uses_future_features") is not False:
        failures.append("FEATURE_LABEL_SEPARATION_NOT_PROVEN")
    if label_start is None or label_end is None:
        failures.append("LABEL_WINDOW_MISSING")
    elif decision is None or label_start < decision or label_end < label_start:
        failures.append("LABEL_WINDOW_NOT_CAUSAL")
    features = row.get("features")
    if not isinstance(features, Mapping) or not features:
        failures.append("FEATURE_VALUES_MISSING")
    else:
        if not features.get("market"):
            failures.append("MARKET_MISSING")
        if not (
            features.get("context_timeframe")
            or features.get("observation_timeframe")
        ):
            failures.append("TIMEFRAME_MISSING")
    if not row.get("feature_hash"):
        failures.append("FEATURE_HASH_MISSING")
    return failures


def build_canonical_ml_dataset(
    settings: Settings,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Register only prospectively proven point-in-time training rows.

    Legacy rows remain in the compatibility artifact and are counted by reason;
    this function never fabricates missing timestamps or provenance.
    """

    from ml.contracts import CanonicalDatasetManifest
    from ml.registry import ImmutableDatasetRegistry

    root = settings.paths.output_dir / "ml"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "canonical_training_status.json"
    if rows is None:
        legacy = build_training_dataset(settings)
        source_path = Path(str(legacy["json_artifact"]))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        candidate_rows = [
            dict(row)
            for row in source.get("rows") or []
            if isinstance(row, Mapping)
        ]
    else:
        legacy = {}
        candidate_rows = [dict(row) for row in rows]
        source_path = None

    exclusions: dict[str, int] = defaultdict(int)
    canonical_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        failures = _canonical_row_failures(row)
        if failures:
            for failure in failures:
                exclusions[failure] += 1
            continue
        canonical_rows.append(row)
    canonical_rows.sort(key=lambda row: str(row["decision_timestamp"]))

    base = {
        "schema_version": "canonical_opportunity_dataset_status_v1",
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "candidate_row_count": len(candidate_rows),
        "canonical_row_count": len(canonical_rows),
        "excluded_row_count": len(candidate_rows) - len(canonical_rows),
        "exclusions": dict(sorted(exclusions.items())),
        "historical_metadata_backfilled": False,
        "canonical_feature_ready_incomplete_count": int(
            legacy.get("canonical_feature_ready_incomplete_count") or 0
        ),
        "canonical_pending_label_horizon_count": int(
            legacy.get("canonical_pending_label_horizon_count") or 0
        ),
        "canonical_label_horizon_mature_unresolved_count": int(
            legacy.get("canonical_label_horizon_mature_unresolved_count") or 0
        ),
        "canonical_incomplete_missing_timeframe_count": int(
            legacy.get("canonical_incomplete_missing_timeframe_count") or 0
        ),
        "next_canonical_label_due_at": legacy.get(
            "next_canonical_label_due_at"
        ),
        "source_artifact": str(source_path.resolve()) if source_path else None,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    if not canonical_rows:
        if base["canonical_pending_label_horizon_count"]:
            reason = "PROSPECTIVE_POINT_IN_TIME_ROWS_AWAITING_24H_LABEL_HORIZON"
        elif base["canonical_feature_ready_incomplete_count"]:
            reason = "PROSPECTIVE_POINT_IN_TIME_ROWS_LABEL_UNRESOLVED"
        else:
            reason = (
                "NO_PROSPECTIVE_ROWS_WITH_COMPLETE_POINT_IN_TIME_AND_LABEL_"
                "PROVENANCE"
            )
        payload = {
            **base,
            "status": "DATA_PENDING",
            "dataset_registered": False,
            "reason": reason,
        }
        atomic_write_json(status_path, payload)
        return payload

    feature_names = sorted(
        {
            str(name)
            for row in canonical_rows
            for name in dict(row["features"])
        }
    )
    row_count = len(canonical_rows)
    missingness = {
        name: Decimal(
            sum(dict(row["features"]).get(name) is None for row in canonical_rows)
        )
        / Decimal(row_count)
        for name in feature_names
    }
    decisions = [
        _utc_timestamp(row["decision_timestamp"]) for row in canonical_rows
    ]
    if any(value is None for value in decisions):
        raise ValueError("canonical row lost its validated decision timestamp")
    decision_times = [value for value in decisions if value is not None]
    source_hash = stable_hash(canonical_rows, length=64)
    manifest = CanonicalDatasetManifest.create(
        schema_version="canonical_opportunity_dataset_v1",
        feature_version=FEATURE_SCHEMA_VERSION,
        label_version=LABEL_SCHEMA_VERSION,
        source_hashes={"canonical_rows": source_hash},
        created_from=(
            "core.opportunity_intelligence.freeze_feature_snapshot",
            "execution.canonical_state.replay_execution_events",
            "closed_market_candle_labels_after_decision_time",
        ),
        time_start=min(decision_times),
        time_end=max(decision_times),
        symbols=tuple(
            sorted({str(dict(row["features"])["market"]) for row in canonical_rows})
        ),
        timeframes=tuple(
            sorted(
                {
                    str(
                        dict(row["features"]).get("context_timeframe")
                        or dict(row["features"])["observation_timeframe"]
                    )
                    for row in canonical_rows
                }
            )
        ),
        feature_count=len(feature_names),
        row_count=row_count,
        missingness_profile=missingness,
        point_in_time_policy={
            "available_at_lte_decision_time": True,
            "event_time_lte_available_at": True,
            "features_labels_separated": True,
            "final_immutable_snapshots_only": True,
            "historical_metadata_backfilled": False,
        },
    )
    registry = ImmutableDatasetRegistry(root / "datasets")
    manifest_path = registry.register(manifest)
    rows_path = manifest_path.parent / "rows.json"
    rows_payload = {
        "schema_version": "canonical_opportunity_rows_v1",
        "dataset_id": manifest.dataset_id,
        "content_hash": manifest.content_hash,
        "row_count": row_count,
        "rows": canonical_rows,
    }
    if rows_path.is_file():
        existing = json.loads(rows_path.read_text(encoding="utf-8"))
        if existing != rows_payload:
            raise ValueError("immutable canonical dataset row collision")
    else:
        atomic_write_json(rows_path, rows_payload)
    payload = {
        **base,
        "status": "REGISTERED_RESEARCH_ONLY",
        "dataset_registered": True,
        "dataset_id": manifest.dataset_id,
        "dataset_content_hash": manifest.content_hash,
        "manifest": str(manifest_path.resolve()),
        "rows_artifact": str(rows_path.resolve()),
        "point_in_time_policy": manifest.point_in_time_policy,
    }
    atomic_write_json(status_path, payload)
    return payload


def _expected_calibration_error(
    labels: Any,
    probabilities: Any,
    *,
    bins: int = 10,
) -> float:
    import numpy as np

    labels_array = np.asarray(labels, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (probability_array >= lower) & (
            probability_array <= upper
            if index == bins - 1
            else probability_array < upper
        )
        if not selected.any():
            continue
        error += float(selected.mean()) * abs(
            float(labels_array[selected].mean())
            - float(probability_array[selected].mean())
        )
    return error


def train_canonical_shadow_models(
    settings: Settings,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train/register a purged PIT model or fail closed as DATA_PENDING."""

    dataset = build_canonical_ml_dataset(settings, rows=rows)
    root = settings.paths.output_dir / "ml"
    status_path = root / "canonical_training_status.json"
    if dataset.get("dataset_registered") is not True:
        return dataset
    source = json.loads(
        Path(str(dataset["rows_artifact"])).read_text(encoding="utf-8")
    )
    selected_rows = [dict(row) for row in source.get("rows") or []]
    positives = sum(int(row.get("net_profitable") or 0) for row in selected_rows)
    negatives = len(selected_rows) - positives
    if (
        len(selected_rows) < MINIMUM_SHADOW_EVALUATION_ROWS
        or min(positives, negatives) < MINIMUM_CLASS_ROWS
    ):
        payload = {
            **dataset,
            "status": "DATA_PENDING",
            "model_registered": False,
            "positive_rows": positives,
            "negative_rows": negatives,
            "minimum_rows": MINIMUM_SHADOW_EVALUATION_ROWS,
            "minimum_class_rows": MINIMUM_CLASS_ROWS,
            "reason": "CANONICAL_SAMPLE_OR_CLASS_SUPPORT_INSUFFICIENT",
        }
        atomic_write_json(status_path, payload)
        return payload

    import joblib
    import numpy as np
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from ml.contracts import ModelArtifactManifest, ModelStatus
    from ml.registry import ModelRegistry, evaluate_model_promotion
    from utils.common import sha256_file

    selected_rows.sort(key=lambda row: str(row["decision_timestamp"]))
    decision_times = [
        _utc_timestamp(row["decision_timestamp"]) for row in selected_rows
    ]
    if any(value is None for value in decision_times):
        raise ValueError("canonical training row has invalid decision timestamp")
    times = [value for value in decision_times if value is not None]
    validation_boundary = times[int(len(times) * 0.60)]
    test_boundary = times[int(len(times) * 0.80)]
    train_index = [
        index
        for index, value in enumerate(times)
        if value < validation_boundary - CANONICAL_LABEL_HORIZON
    ]
    validation_index = [
        index
        for index, value in enumerate(times)
        if validation_boundary <= value < test_boundary - CANONICAL_LABEL_HORIZON
    ]
    test_index = [
        index for index, value in enumerate(times) if value >= test_boundary
    ]
    if min(map(len, (train_index, validation_index, test_index))) < 20:
        payload = {
            **dataset,
            "status": "DATA_PENDING",
            "model_registered": False,
            "purge_horizon_hours": 24,
            "split_rows": {
                "train": len(train_index),
                "validation": len(validation_index),
                "test": len(test_index),
            },
            "reason": "PURGED_TRAIN_VALIDATION_TEST_SPLITS_TOO_SMALL",
        }
        atomic_write_json(status_path, payload)
        return payload

    import pandas as pd

    frame = pd.DataFrame([dict(row["features"]) for row in selected_rows])
    labels = np.asarray([int(row["net_profitable"]) for row in selected_rows])
    categorical = [
        name
        for name in (
            "market",
            "family",
            "context_timeframe",
            "observation_timeframe",
            "macro_regime",
            "trade_type",
            "market_mode",
        )
        if name in frame.columns
    ]
    numeric = [name for name in frame.columns if name not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("canonical dataset has no trainable feature columns")
    estimator = Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                    max_iter=2_000,
                ),
            ),
        ]
    )

    development_index = [
        index
        for index, value in enumerate(times)
        if value < test_boundary - CANONICAL_LABEL_HORIZON
    ]
    fold_metrics: list[dict[str, Any]] = []
    splitter = TimeSeriesSplit(n_splits=5)
    for fold, (raw_train, raw_validation) in enumerate(
        splitter.split(development_index), 1
    ):
        fold_validation = [development_index[int(index)] for index in raw_validation]
        fold_validation_start = times[fold_validation[0]]
        fold_train = [
            development_index[int(index)]
            for index in raw_train
            if times[development_index[int(index)]]
            < fold_validation_start - CANONICAL_LABEL_HORIZON
        ]
        if len(fold_train) < 20 or len(set(labels[fold_train])) < 2:
            continue
        model = clone(estimator)
        model.fit(frame.iloc[fold_train], labels[fold_train])
        probabilities = model.predict_proba(frame.iloc[fold_validation])[:, 1]
        fold_labels = labels[fold_validation]
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "train_start": times[fold_train[0]].isoformat(),
                "train_end": times[fold_train[-1]].isoformat(),
                "validation_start": times[fold_validation[0]].isoformat(),
                "validation_end": times[fold_validation[-1]].isoformat(),
                "test_start": times[test_index[0]].isoformat(),
                "test_end": times[test_index[-1]].isoformat(),
                "purge_hours": 24,
                "roc_auc": float(roc_auc_score(fold_labels, probabilities))
                if len(set(fold_labels)) == 2
                else None,
                "brier": float(brier_score_loss(fold_labels, probabilities)),
            }
        )
    if len(fold_metrics) != 5 or len(set(labels[train_index])) < 2:
        payload = {
            **dataset,
            "status": "DATA_PENDING",
            "model_registered": False,
            "purged_walk_forward_folds": fold_metrics,
            "reason": "FIVE_VALID_PURGED_WALK_FORWARD_FOLDS_NOT_AVAILABLE",
        }
        atomic_write_json(status_path, payload)
        return payload

    estimator.fit(frame.iloc[train_index], labels[train_index])
    validation_probability = estimator.predict_proba(frame.iloc[validation_index])[:, 1]
    test_probability = estimator.predict_proba(frame.iloc[test_index])[:, 1]
    validation_labels = labels[validation_index]
    test_labels = labels[test_index]
    validation_brier = float(
        brier_score_loss(validation_labels, validation_probability)
    )
    test_brier = float(brier_score_loss(test_labels, test_probability))
    validation_ece = _expected_calibration_error(
        validation_labels, validation_probability
    )
    test_ece = _expected_calibration_error(test_labels, test_probability)

    bundle = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "dataset_id": dataset["dataset_id"],
        "model": estimator,
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "purge_horizon_hours": 24,
    }
    staging = root / "models" / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary_bundle = staging / f"{dataset['dataset_id']}.joblib"
    joblib.dump(bundle, temporary_bundle)
    binary_hash = sha256_file(temporary_bundle)
    trained_at = utc_now()
    manifest = ModelArtifactManifest.create(
        model_id="canonical_opportunity_meta_labeler",
        dataset_id=str(dataset["dataset_id"]),
        feature_schema=FEATURE_SCHEMA_VERSION,
        label_schema=LABEL_SCHEMA_VERSION,
        algorithm="LOGISTIC_REGRESSION_BALANCED_BASELINE",
        hyperparameters={
            "solver": "liblinear",
            "class_weight": "balanced",
            "random_state": 42,
            "binary_sha256": binary_hash,
            "purge_horizon_hours": 24,
        },
        train_range=(times[train_index[0]], times[train_index[-1]]),
        validation_range=(
            times[validation_index[0]],
            times[validation_index[-1]],
        ),
        test_range=(times[test_index[0]], times[test_index[-1]]),
        code_commit=f"WORKTREE_SHA256:{sha256_file(Path(__file__))}",
        metrics={
            "validation_roc_auc": float(
                roc_auc_score(validation_labels, validation_probability)
            )
            if len(set(validation_labels)) == 2
            else None,
            "test_roc_auc": float(roc_auc_score(test_labels, test_probability))
            if len(set(test_labels)) == 2
            else None,
            "validation_brier": validation_brier,
            "test_brier": test_brier,
            "row_count": len(selected_rows),
        },
        economic_metrics={
            "status": None,
            "test_net_expectancy": None,
        },
        calibration={
            "validation_ece": validation_ece,
            "test_ece": test_ece,
            "bins": 10,
        },
        regime_metrics={"status": "NOT_EVALUABLE_BASELINE_SHADOW"},
        status=ModelStatus.SHADOW,
        trained_at=trained_at,
        expires_at=trained_at + timedelta(days=30),
        live_decision_influence=False,
    )
    model_directory = (
        root / "models" / manifest.model_id / manifest.artifact_hash
    )
    model_directory.mkdir(parents=True, exist_ok=True)
    bundle_path = model_directory / "model.joblib"
    if bundle_path.is_file():
        if sha256_file(bundle_path) != binary_hash:
            raise ValueError("immutable canonical model binary collision")
        temporary_bundle.unlink()
    else:
        temporary_bundle.replace(bundle_path)
    manifest_path = ModelRegistry(root / "models").register(manifest)
    promotion = evaluate_model_promotion(
        current_status=ModelStatus.RESEARCH_ONLY,
        requested_status=ModelStatus.SHADOW,
        evidence={
            "dataset_immutable": True,
            "point_in_time_passed": True,
            "lookahead_passed": True,
            "purged_walk_forward_passed": True,
        },
    )
    if promotion["permitted"] is not True:
        raise ValueError("canonical shadow promotion evidence unexpectedly failed")
    payload = {
        **dataset,
        "status": "REGISTERED_SHADOW_NO_LIVE_AUTHORITY",
        "model_registered": True,
        "model_id": manifest.model_id,
        "model_artifact_hash": manifest.artifact_hash,
        "model_manifest": str(manifest_path.resolve()),
        "model_binary": str(bundle_path.resolve()),
        "model_binary_sha256": binary_hash,
        "purged_walk_forward": True,
        "purge_horizon_hours": 24,
        "validation_folds": fold_metrics,
        "calibration_report": manifest.calibration,
        "promotion_evaluation": promotion,
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(status_path, payload)
    return payload


def train_shadow_models(
    settings: Settings,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chronologically validate compact tabular challengers in shadow mode."""

    if rows is None:
        dataset = build_training_dataset(settings)
        source = json.loads(
            Path(dataset["json_artifact"]).read_text(encoding="utf-8")
        )
        selected_rows = list(source.get("rows") or [])
    else:
        selected_rows = [dict(row) for row in rows]
    root = settings.paths.output_dir / "intelligence"
    root.mkdir(parents=True, exist_ok=True)
    positives = sum(int(row.get("net_profitable") or 0) for row in selected_rows)
    negatives = len(selected_rows) - positives
    if (
        len(selected_rows) < PIPELINE_SMOKE_ROWS
        or min(positives, negatives) < min(20, MINIMUM_CLASS_ROWS)
    ):
        drift = build_intelligence_drift_report(
            settings.paths.project_root,
            rows=selected_rows,
        )
        payload = {
            "status": "DATA_PENDING",
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "row_count": len(selected_rows),
            "positive_rows": positives,
            "negative_rows": negatives,
            "pipeline_smoke_rows": PIPELINE_SMOKE_ROWS,
            "minimum_shadow_evaluation_rows": MINIMUM_SHADOW_EVALUATION_ROWS,
            "minimum_class_rows_for_shadow_evaluation": MINIMUM_CLASS_ROWS,
            "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
            "models": {
                "meta_labeler": "DATA_PENDING",
                "fill_slippage": "DATA_PENDING",
                "reversal_persistence": "DATA_PENDING",
                "anomaly_detector": "DATA_PENDING",
            },
            "drift_monitor": {
                "status": drift["status"],
                "artifact": drift["artifact"],
                "authority": "SHADOW_ONLY",
                "live_decision_influence": False,
            },
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(root / "model_status.json", payload)
        return payload

    import joblib
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    selected_rows.sort(key=lambda row: str(row.get("decision_timestamp") or ""))
    frame_rows = [dict(row.get("features") or {}) for row in selected_rows]
    import pandas as pd

    frame = pd.DataFrame(frame_rows)
    labels = np.asarray([int(row["net_profitable"]) for row in selected_rows])
    categorical = [
        name
        for name in (
            "market",
            "family",
            "context_timeframe",
            "observation_timeframe",
            "macro_regime",
            "trade_type",
            "market_mode",
        )
        if name in frame.columns
    ]
    numeric = [name for name in frame.columns if name not in categorical]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    logistic = Pipeline(
        [
            ("features", preprocessing),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=0.2,
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    split = TimeSeriesSplit(n_splits=5, gap=max(1, len(frame) // 100))
    fold_metrics: list[dict[str, Any]] = []
    oos_predictions: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(split.split(frame), 1):
        logistic.fit(frame.iloc[train_index], labels[train_index])
        probability = logistic.predict_proba(frame.iloc[test_index])[:, 1]
        y_test = labels[test_index]
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "roc_auc": float(roc_auc_score(y_test, probability))
                if len(set(y_test)) == 2
                else None,
                "brier": float(brier_score_loss(y_test, probability)),
            }
        )
        oos_predictions.extend(
            {
                "fold": fold,
                "decision_timestamp": selected_rows[int(row_index)].get(
                    "decision_timestamp"
                ),
                "prediction": float(predicted),
                "label": int(labels[int(row_index)]),
            }
            for row_index, predicted in zip(
                test_index, probability, strict=True
            )
        )
    logistic.fit(frame, labels)
    transformed = preprocessing.fit_transform(frame)
    challenger = HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, learning_rate=0.05, random_state=42
    )
    challenger.fit(transformed.toarray() if hasattr(transformed, "toarray") else transformed, labels)
    anomaly = IsolationForest(contamination="auto", random_state=42)
    anomaly.fit(transformed.toarray() if hasattr(transformed, "toarray") else transformed)
    bundle = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "trained_until_timestamp": selected_rows[-1].get("decision_timestamp"),
        "logistic_meta_labeler": logistic,
        "preprocessor": preprocessing,
        "gradient_boosting_challenger": challenger,
        "anomaly_detector": anomaly,
        "live_decision_influence": False,
    }
    joblib.dump(bundle, root / "model_bundle.joblib")
    evaluation_ready = (
        len(selected_rows) >= MINIMUM_SHADOW_EVALUATION_ROWS
        and min(positives, negatives) >= MINIMUM_CLASS_ROWS
    )
    training_data_hash = stable_hash(selected_rows, length=64)
    drift = build_intelligence_drift_report(
        settings.paths.project_root,
        rows=selected_rows,
        oos_predictions=oos_predictions,
        training_data_hash=training_data_hash,
    )
    payload = {
        "status": "SHADOW_ONLY" if evaluation_ready else "PIPELINE_SMOKE_ONLY",
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "row_count": len(selected_rows),
        "trained_until_timestamp": selected_rows[-1].get("decision_timestamp"),
        "training_data_hash": training_data_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "chronological_validation": True,
        "validation_folds": fold_metrics,
        "promotion_evaluation_ready": evaluation_ready,
        "promotion_gate": {
            "clean_deduplicated_rows_minimum": MINIMUM_SHADOW_EVALUATION_ROWS,
            "chronological_splits_minimum": 5,
            "profitable_splits_minimum": 4,
            "stressed_net_expectancy_positive": True,
            "net_expectancy_improvement_vs_baseline_minimum": 0.15,
            "false_breakout_relative_reduction_minimum": 0.15,
            "drawdown_not_worse_than_baseline": True,
            "single_asset_profit_contribution_maximum": 0.35,
            "single_period_profit_contribution_maximum": 0.35,
            "probability_calibration_required": True,
            "all_gates_passed": False,
        },
        "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        "models": {
            "meta_labeler": "LOGISTIC_ELASTIC_NET_SHADOW",
            "meta_labeler_challenger": "HIST_GRADIENT_BOOSTING_SHADOW",
            "fill_slippage": "DATA_PENDING",
            "reversal_persistence": "DATA_PENDING",
            "anomaly_detector": "ISOLATION_FOREST_SHADOW",
        },
        "drift_monitor": {
            "status": drift["status"],
            "artifact": drift["artifact"],
            "critical_feature_count": len(drift.get("critical_features") or []),
            "warning_feature_count": len(drift.get("warning_features") or []),
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(root / "model_status.json", payload)
    atomic_write_json(root / "validation_report.json", payload)
    return payload
