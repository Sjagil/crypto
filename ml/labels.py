"""Causal label construction kept physically and logically outside features."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.contracts import CanonicalLabelRecord, LabelOutcome, LabelSchema
from utils.common import atomic_write_json, stable_hash

ZERO = Decimal("0")


def build_triple_barrier_label(
    *,
    candidate_id: str,
    market: str,
    decision_time: datetime,
    feature_cutoff: datetime,
    entry_price: Decimal,
    future_bars: Sequence[Mapping[str, Any]],
    schema: LabelSchema,
    fees_fraction: Decimal = ZERO,
    spread_fraction: Decimal = ZERO,
    slippage_fraction: Decimal = ZERO,
) -> CanonicalLabelRecord:
    if entry_price <= ZERO:
        raise ValueError("entry price must be positive")
    cutoff = decision_time + timedelta(seconds=schema.maximum_holding_seconds)
    selected: list[tuple[datetime, Decimal, Decimal, Decimal]] = []
    for bar in future_bars:
        timestamp = bar["timestamp"]
        if not isinstance(timestamp, datetime):
            raise TypeError("label bar timestamp must be datetime")
        if timestamp < decision_time or timestamp > cutoff:
            continue
        selected.append(
            (
                timestamp,
                Decimal(str(bar["high"])),
                Decimal(str(bar["low"])),
                Decimal(str(bar["close"])),
            )
        )
    selected.sort(key=lambda row: row[0])
    target_price = entry_price * (Decimal("1") + schema.profit_barrier_fraction)
    stop_price = entry_price * (Decimal("1") - schema.stop_barrier_fraction)
    maximum = entry_price
    minimum = entry_price
    outcome = LabelOutcome.NOT_EVALUABLE
    exit_price: Decimal | None = None
    exit_time = decision_time
    for timestamp, high, low, close in selected:
        del close
        maximum = max(maximum, high)
        minimum = min(minimum, low)
        target_hit = high >= target_price
        stop_hit = low <= stop_price
        if target_hit and stop_hit:
            outcome = LabelOutcome.AMBIGUOUS_SAME_BAR
            exit_price = stop_price if schema.conservative_same_bar_policy else None
            exit_time = timestamp
            break
        if target_hit:
            outcome = LabelOutcome.TARGET_FIRST
            exit_price = target_price
            exit_time = timestamp
            break
        if stop_hit:
            outcome = LabelOutcome.STOP_FIRST
            exit_price = stop_price
            exit_time = timestamp
            break
    if outcome is LabelOutcome.NOT_EVALUABLE and selected:
        outcome = LabelOutcome.TIMEOUT
        exit_time, _, _, exit_price = selected[-1]

    gross_return = (
        (exit_price / entry_price) - Decimal("1")
        if exit_price is not None
        else None
    )
    costs = fees_fraction + spread_fraction + slippage_fraction
    net_return = gross_return - costs if gross_return is not None else None
    mfe = (maximum / entry_price) - Decimal("1") if selected else None
    mae = (minimum / entry_price) - Decimal("1") if selected else None
    values = {
        "label_version": schema.label_version,
        "candidate_id": candidate_id,
        "market": market,
        "decision_time": decision_time,
        "feature_cutoff": feature_cutoff,
        "label_start": decision_time,
        "label_end": exit_time,
        "outcome": outcome,
        "target_first": True if outcome is LabelOutcome.TARGET_FIRST else False if selected else None,
        "stop_first": True if outcome in {LabelOutcome.STOP_FIRST, LabelOutcome.AMBIGUOUS_SAME_BAR} else False if selected else None,
        "timeout": outcome is LabelOutcome.TIMEOUT,
        "gross_return": gross_return,
        "net_return": net_return,
        "mae": mae,
        "mfe": mfe,
        "holding_seconds": int((exit_time - decision_time).total_seconds()) if selected else None,
        "fees_fraction": fees_fraction,
        "spread_fraction": spread_fraction,
        "slippage_fraction": slippage_fraction,
        "exit_reason": outcome.value,
    }
    identity = {
        key: str(value) if isinstance(value, (datetime, Decimal, LabelOutcome)) else value
        for key, value in values.items()
    }
    return CanonicalLabelRecord(
        label_id=f"label_{stable_hash(identity, length=48)}",
        **values,
    )


def freeze_labels(
    records: Sequence[CanonicalLabelRecord],
    output_root: Path,
) -> dict[str, Any]:
    payload = [record.model_dump(mode="json") for record in sorted(records, key=lambda row: row.label_id)]
    freeze_hash = stable_hash(payload, length=64)
    target = output_root / "freezes" / freeze_hash / "labels.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if stable_hash(existing, length=64) != freeze_hash:
            raise ValueError("existing label freeze identity mismatch")
    else:
        atomic_write_json(target, payload)
    return {
        "label_freeze_id": f"labels_{freeze_hash}",
        "hash": freeze_hash,
        "path": str(target.resolve()),
        "row_count": len(records),
        "immutable": True,
    }


__all__ = ["build_triple_barrier_label", "freeze_labels"]
