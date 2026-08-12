"""Canonical decision-to-fill attribution without inventing missing evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from utils.common import atomic_write_json, read_json, sha256_file, stable_hash, utc_iso

SCHEMA_VERSION = "decision_execution_attribution_v2"
ZERO = Decimal("0")


def _decimal(value: object) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        selected = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if selected.tzinfo is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _weighted_average(
    rows: list[Mapping[str, Any]],
) -> tuple[Decimal | None, Decimal]:
    quantity = sum(
        (_decimal(row.get("quantity")) or ZERO for row in rows),
        ZERO,
    )
    if quantity <= ZERO:
        return None, ZERO
    notional = sum(
        (
            (_decimal(row.get("price")) or ZERO)
            * (_decimal(row.get("quantity")) or ZERO)
            for row in rows
        ),
        ZERO,
    )
    return notional / quantity, quantity


def _signal_context(root: Path, signal_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Resolve only live signal identities; do not load research-wide history."""

    contexts: dict[str, dict[str, Any]] = {}
    generated = _read_mapping(root / "output" / "live" / "generated_strategy_live_state.json")
    for position in dict(generated.get("positions") or {}).values():
        if not isinstance(position, Mapping):
            continue
        signal_id = str(position.get("signal_id") or "")
        if signal_id not in signal_ids:
            continue
        contexts[signal_id] = {
            "source": "GENERATED_STRATEGY_LIVE_STATE",
            "strategy_id": position.get("strategy_id"),
            "strategy_dna": position.get("strategy_dna_hash"),
            "timeframe": position.get("timeframe"),
            "decision_timestamp": position.get("signal_timestamp"),
            # This value is persisted after execution and is therefore not an
            # independent pre-fill benchmark.  It must never imply zero slip.
            "decision_price": None,
            "decision_price_status": "UNAVAILABLE_INDEPENDENT_PRE_FILL_REFERENCE",
            "candidate_metrics": dict(position.get("candidate_metrics") or {}),
        }

    projection_path = root / "output" / "live" / "opportunity_lifecycle_state.json"
    projection = _read_mapping(projection_path)
    for signal_id, raw in dict(projection.get("opportunities") or {}).items():
        if signal_id not in signal_ids or not isinstance(raw, Mapping):
            continue
        opportunity = dict(raw)
        snapshot = dict(opportunity.get("feature_snapshot") or {})
        values = dict(snapshot.get("values") or {})
        contexts[signal_id] = {
            "source": "IMMUTABLE_OPPORTUNITY_PROJECTION",
            "strategy_id": opportunity.get("playbook_id"),
            "strategy_dna": opportunity.get("playbook_dna"),
            "strategy_family": opportunity.get("family"),
            "timeframe": opportunity.get("context_timeframe"),
            "regime": opportunity.get("macro_regime"),
            "decision_timestamp": (
                snapshot.get("decision_timestamp")
                or opportunity.get("detected_at")
            ),
            "decision_price": (
                values.get("entry_price")
                if values.get("entry_price") is not None
                else opportunity.get("entry_price")
            ),
            "decision_price_status": "CAUSAL_PRE_FILL_REFERENCE",
            "setup_score": opportunity.get("score"),
            "execution_quality_score": opportunity.get("execution_quality_score"),
            "execution_scorecard": dict(opportunity.get("execution_scorecard") or {}),
            "score_components": dict(opportunity.get("score_components") or {}),
            "gate_matrix": dict(opportunity.get("gate_matrix") or {}),
            "feature_hash": snapshot.get("feature_hash"),
            "feature_schema_version": snapshot.get("feature_schema_version"),
        }
    return contexts


def _strategy_aggregates(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("strategy_id") or "UNKNOWN")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for strategy_id, selected in sorted(grouped.items()):
        closed = [row for row in selected if row.get("position_status") == "CLOSED"]
        net_values = [
            _decimal(row.get("realized_net_pnl_eur"))
            for row in closed
            if _decimal(row.get("realized_net_pnl_eur")) is not None
        ]
        fees = sum(
            (_decimal(row.get("fees_eur")) or ZERO for row in selected),
            ZERO,
        )
        slippages = [
            _decimal(row.get("entry_slippage_bps"))
            for row in selected
            if _decimal(row.get("entry_slippage_bps")) is not None
        ]
        acknowledgement_latencies = [
            _decimal(row.get("submission_to_ack_seconds"))
            for row in selected
            if _decimal(row.get("submission_to_ack_seconds")) is not None
        ]
        fill_latencies = [
            _decimal(row.get("submission_to_first_fill_seconds"))
            for row in selected
            if _decimal(row.get("submission_to_first_fill_seconds")) is not None
        ]
        output[strategy_id] = {
            "strategy_id": strategy_id,
            "strategy_dna_values": sorted(
                {
                    str(row.get("strategy_dna") or "")
                    for row in selected
                    if row.get("strategy_dna")
                }
            ),
            "trade_count": len(selected),
            "closed_round_trips": len(closed),
            "open_positions": sum(
                row.get("position_status") == "OPEN" for row in selected
            ),
            "fees_eur": str(fees),
            "realized_net_pnl_eur": str(sum(net_values, ZERO)),
            "average_entry_slippage_bps": (
                str(sum(slippages, ZERO) / Decimal(len(slippages)))
                if slippages
                else None
            ),
            "slippage_observation_count": len(slippages),
            "average_submission_to_ack_seconds": (
                str(
                    sum(acknowledgement_latencies, ZERO)
                    / Decimal(len(acknowledgement_latencies))
                )
                if acknowledgement_latencies
                else None
            ),
            "average_submission_to_first_fill_seconds": (
                str(sum(fill_latencies, ZERO) / Decimal(len(fill_latencies)))
                if fill_latencies
                else None
            ),
        }
    return output


def build_decision_execution_attribution(root: Path) -> dict[str, Any]:
    """Build sanitized attribution from canonical live fills only.

    The builder is cached by the canonical execution-ledger hash.  It performs
    no exchange request and never serializes venue, client, intent or fill IDs.
    """

    root = root.resolve()
    source = root / "output" / "checkpoints" / "live_execution.jsonl"
    output = root / "output" / "operations" / "decision_execution_attribution.json"
    source_hash = sha256_file(source) if source.is_file() else None
    cached = _read_mapping(output)
    if (
        cached.get("schema_version") == SCHEMA_VERSION
        and cached.get("source_ledger_hash") == source_hash
    ):
        cached["artifact"] = str(output)
        return cached

    events = _read_jsonl(source)
    intents: dict[str, dict[str, Any]] = {}
    acknowledgements: dict[str, dict[str, Any]] = {}
    fills_by_signal: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        payload = dict(event.get("payload") or {})
        event_type = str(event.get("event_type") or "").upper()
        intent_id = str(payload.get("intent_id") or "")
        if event_type == "ORDER_INTENT" and intent_id:
            intents[intent_id] = {
                "recorded_at": event.get("recorded_at"),
                "submission_started_at": payload.get(
                    "submission_started_at"
                ),
                "order_type": payload.get("order_type"),
                "time_in_force": payload.get("time_in_force"),
                "reason_codes": list(payload.get("reason_codes") or []),
            }
        if event_type == "ORDER_ACKNOWLEDGED" and intent_id:
            acknowledgements[intent_id] = {
                "recorded_at": event.get("recorded_at"),
                "acknowledgement_received_at": payload.get(
                    "acknowledgement_received_at"
                ),
                "exchange_created_at": payload.get("exchange_created_at"),
                "exchange_updated_at": payload.get("exchange_updated_at"),
                "recovered": payload.get("recovered") is True,
            }
        if event_type != "FILL":
            continue
        signal_id = str(payload.get("signal_id") or "")
        strategy_id = str(payload.get("strategy_id") or "")
        if not signal_id or not strategy_id or strategy_id.startswith("OPERATOR_"):
            continue
        fills_by_signal[signal_id].append(
            {
                "recorded_at": event.get("recorded_at"),
                "filled_at": payload.get("filled_at"),
                "received_at": payload.get("received_at"),
                "exchange_created_at": payload.get("exchange_created_at"),
                "exchange_updated_at": payload.get("exchange_updated_at"),
                "intent_id": intent_id,
                "market": payload.get("market"),
                "side": str(payload.get("side") or "").upper(),
                "price": payload.get("price"),
                "quantity": payload.get("quantity"),
                "fee_eur": payload.get("fee_eur"),
                "fee_known": payload.get("fee_known") is True,
                "strategy_id": strategy_id,
                "strategy_dna": payload.get("strategy_dna_hash"),
                "reason_codes": list(payload.get("reason_codes") or []),
            }
        )

    contexts = _signal_context(root, set(fills_by_signal))
    trades: list[dict[str, Any]] = []
    for signal_id, fills in sorted(fills_by_signal.items()):
        fills.sort(key=lambda row: str(row.get("recorded_at") or ""))
        buys = [row for row in fills if row.get("side") == "BUY"]
        sells = [row for row in fills if row.get("side") == "SELL"]
        buy_average, buy_quantity = _weighted_average(buys)
        sell_average, sell_quantity = _weighted_average(sells)
        matched_quantity = min(buy_quantity, sell_quantity)
        open_quantity = max(ZERO, buy_quantity - sell_quantity)
        buy_fees = sum(
            (_decimal(row.get("fee_eur")) or ZERO for row in buys), ZERO
        )
        sell_fees = sum(
            (_decimal(row.get("fee_eur")) or ZERO for row in sells), ZERO
        )
        allocated_buy_fees = (
            buy_fees * matched_quantity / buy_quantity
            if buy_quantity > ZERO
            else ZERO
        )
        gross_realized = (
            matched_quantity * (sell_average - buy_average)
            if matched_quantity > ZERO
            and buy_average is not None
            and sell_average is not None
            else None
        )
        net_realized = (
            gross_realized - allocated_buy_fees - sell_fees
            if gross_realized is not None
            else None
        )
        context = dict(contexts.get(signal_id) or {})
        decision_price = _decimal(context.get("decision_price"))
        entry_slippage_bps = (
            (buy_average / decision_price - Decimal("1")) * Decimal("10000")
            if buy_average is not None
            and decision_price is not None
            and decision_price > ZERO
            else None
        )
        entry_shortfall = (
            (buy_average - decision_price) * buy_quantity
            if buy_average is not None and decision_price is not None
            else None
        )
        first_buy_recorded_at = (
            _timestamp(buys[0].get("recorded_at")) if buys else None
        )
        first_buy_exchange_at = (
            _timestamp(buys[0].get("filled_at")) if buys else None
        ) or first_buy_recorded_at
        first_buy_received_at = (
            _timestamp(buys[0].get("received_at")) if buys else None
        ) or first_buy_recorded_at
        decision_at = _timestamp(context.get("decision_timestamp"))
        entry_intent = intents.get(str(buys[0].get("intent_id") or ""), {}) if buys else {}
        entry_acknowledgement = (
            acknowledgements.get(str(buys[0].get("intent_id") or ""), {})
            if buys
            else {}
        )
        submitted_at = _timestamp(
            entry_intent.get("submission_started_at")
        ) or _timestamp(entry_intent.get("recorded_at"))
        acknowledged_at = _timestamp(
            entry_acknowledgement.get("acknowledgement_received_at")
        ) or _timestamp(entry_acknowledgement.get("recorded_at"))
        closed = buy_quantity > ZERO and open_quantity <= buy_quantity * Decimal("0.000001")
        fees_known = all(row.get("fee_known") is True for row in fills)
        if closed and decision_price is not None and fees_known:
            completeness = "CLOSED_FULL_EXECUTION_ATTRIBUTION"
        elif open_quantity > ZERO and fees_known:
            completeness = "OPEN_EXECUTION_ATTRIBUTION"
        else:
            completeness = "PARTIAL_ATTRIBUTION_MISSING_EVIDENCE"
        strategy_id = str(fills[0].get("strategy_id") or context.get("strategy_id") or "UNKNOWN")
        strategy_dna = str(fills[0].get("strategy_dna") or context.get("strategy_dna") or "")
        trades.append(
            {
                "trade_public_id": stable_hash(
                    {"signal_id": signal_id, "strategy_id": strategy_id},
                    length=20,
                ),
                "evidence": "CANONICAL_LIVE_FILL_LEDGER",
                "attribution_status": completeness,
                "market": fills[0].get("market"),
                "strategy_id": strategy_id,
                "strategy_dna": strategy_dna,
                "strategy_family": context.get("strategy_family"),
                "timeframe": context.get("timeframe"),
                "regime": context.get("regime"),
                "position_status": "CLOSED" if closed else "OPEN",
                "buy_fill_count": len(buys),
                "sell_fill_count": len(sells),
                "buy_quantity": str(buy_quantity),
                "sell_quantity": str(sell_quantity),
                "open_quantity": str(open_quantity),
                "average_buy_fill_price": str(buy_average) if buy_average is not None else None,
                "average_sell_fill_price": str(sell_average) if sell_average is not None else None,
                "decision_price": str(decision_price) if decision_price is not None else None,
                "decision_price_status": context.get("decision_price_status", "UNAVAILABLE"),
                "entry_slippage_bps": str(entry_slippage_bps) if entry_slippage_bps is not None else None,
                "entry_price_shortfall_eur": str(entry_shortfall) if entry_shortfall is not None else None,
                "fees_eur": str(buy_fees + sell_fees),
                "fees_known": fees_known,
                "realized_gross_pnl_eur": str(gross_realized) if gross_realized is not None else None,
                "realized_net_pnl_eur": str(net_realized) if net_realized is not None else None,
                "decision_to_first_fill_seconds": (
                    max(
                        0.0,
                        (first_buy_exchange_at - decision_at).total_seconds(),
                    )
                    if first_buy_exchange_at is not None and decision_at is not None
                    else None
                ),
                "submission_to_first_fill_seconds": (
                    max(
                        0.0,
                        (first_buy_exchange_at - submitted_at).total_seconds(),
                    )
                    if first_buy_exchange_at is not None and submitted_at is not None
                    else None
                ),
                "submission_to_ack_seconds": (
                    max(0.0, (acknowledged_at - submitted_at).total_seconds())
                    if acknowledged_at is not None and submitted_at is not None
                    else None
                ),
                "exchange_fill_to_local_receive_seconds": (
                    max(
                        0.0,
                        (
                            first_buy_received_at - first_buy_exchange_at
                        ).total_seconds(),
                    )
                    if first_buy_received_at is not None
                    and first_buy_exchange_at is not None
                    else None
                ),
                "timestamp_evidence": {
                    "decision_at_available": decision_at is not None,
                    "submission_started_at_source": (
                        "EXPLICIT"
                        if entry_intent.get("submission_started_at")
                        else "LEDGER_RECORDED_AT_FALLBACK"
                    ),
                    "acknowledgement_received_at_source": (
                        "EXPLICIT"
                        if entry_acknowledgement.get(
                            "acknowledgement_received_at"
                        )
                        else (
                            "LEDGER_RECORDED_AT_FALLBACK"
                            if entry_acknowledgement
                            else "UNAVAILABLE"
                        )
                    ),
                    "fill_exchange_at_source": (
                        "EXCHANGE_EVENT"
                        if buys and buys[0].get("filled_at")
                        else "LEDGER_RECORDED_AT_FALLBACK"
                    ),
                    "fill_received_at_source": (
                        "EXPLICIT"
                        if buys and buys[0].get("received_at")
                        else "LEDGER_RECORDED_AT_FALLBACK"
                    ),
                    "acknowledgement_recovered": (
                        entry_acknowledgement.get("recovered") is True
                    ),
                },
                "entry_order_type": entry_intent.get("order_type"),
                "entry_time_in_force": entry_intent.get("time_in_force"),
                "entry_reason_codes": list(entry_intent.get("reason_codes") or []),
                "exit_reason_codes": sorted(
                    {
                        str(reason)
                        for row in sells
                        for reason in row.get("reason_codes") or []
                    }
                ),
                "decision_context": {
                    key: context.get(key)
                    for key in (
                        "source",
                        "setup_score",
                        "execution_quality_score",
                        "execution_scorecard",
                        "score_components",
                        "gate_matrix",
                        "candidate_metrics",
                        "feature_hash",
                        "feature_schema_version",
                    )
                    if context.get(key) is not None
                },
            }
        )

    mapped_decisions = sum(
        row.get("decision_price") is not None for row in trades
    )
    explicit_submission_count = sum(
        row.get("timestamp_evidence", {}).get(
            "submission_started_at_source"
        )
        == "EXPLICIT"
        for row in trades
    )
    explicit_exchange_fill_count = sum(
        row.get("timestamp_evidence", {}).get("fill_exchange_at_source")
        == "EXCHANGE_EVENT"
        for row in trades
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "source_ledger_hash": source_hash,
        "source": str(source),
        "status": "READY" if trades else "NO_CANONICAL_STRATEGY_FILLS",
        "trade_count": len(trades),
        "closed_round_trips": sum(
            row.get("position_status") == "CLOSED" for row in trades
        ),
        "open_positions": sum(
            row.get("position_status") == "OPEN" for row in trades
        ),
        "decision_price_mapped_count": mapped_decisions,
        "decision_price_mapping_ratio": (
            mapped_decisions / len(trades) if trades else None
        ),
        "timing_coverage": {
            "explicit_submission_count": explicit_submission_count,
            "explicit_exchange_fill_count": explicit_exchange_fill_count,
            "trade_count": len(trades),
            "new_events_are_point_in_time_complete": True,
            "legacy_events_use_explicitly_labeled_fallbacks": True,
        },
        "strategy_attribution": _strategy_aggregates(trades),
        "trades": trades,
        "privacy": {
            "exchange_order_ids_serialized": False,
            "client_order_ids_serialized": False,
            "intent_ids_serialized": False,
            "fill_ids_serialized": False,
            "secrets_serialized": False,
        },
        "interpretation": {
            "missing_decision_price_is_not_zero_slippage": True,
            "paper_is_not_live": True,
            "unrealized_pnl_is_not_realized_pnl": True,
            "context_components_are_not_claimed_as_euro_pnl": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(output, payload)
    payload["artifact"] = str(output)
    return payload


__all__ = ["build_decision_execution_attribution"]
