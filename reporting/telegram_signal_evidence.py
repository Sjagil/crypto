"""Prospective, orderless TP2/stop evidence for tactical Telegram alerts.

The outbound Telegram ledger proves delivery, not signal quality.  This module
therefore evaluates only exact rows captured at notification time against
fully closed local candles.  Legacy preview messages are reconstructed only as
an explicitly non-promotable diagnostic because their displayed prices were
rounded and they lack canonical opportunity identities.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config.settings import Settings
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso, utc_now

SCHEMA_VERSION = "telegram_signal_evidence_v1"
EVIDENCE_EVENT_SCHEMA = "telegram_opportunity_evidence_event_v1"
HORIZON = timedelta(hours=24)
DEFAULT_ROUNDTRIP_COST_BPS = Decimal("35")
MINIMUM_RESOLVED_ALERTS = 30
MINIMUM_WILSON_LOWER_BOUND = 0.50
BOUNDARY_OUTCOMES = frozenset(
    {"TP2_BEFORE_STOP", "STOP_BEFORE_TP2", "AMBIGUOUS_SAME_CANDLE"}
)
TERMINAL_RETURN_OUTCOMES = BOUNDARY_OUTCOMES | {"TIME_EXIT_24H"}
PRICE_RE = r"(?:€\s*)?([^\n]+)"
LEGACY_ROW_RE = re.compile(
    rf"(?m)^(\d+)\.\s+([A-Z0-9-]+)\s+·\s+([^\n·]+?)\s+·\s+([^\n]+)\n"
    rf"Trigger:\s*{PRICE_RE}\n"
    rf"Stop:\s*{PRICE_RE}\n"
    rf"TP1 / TP2:\s*{PRICE_RE}\s*/\s*{PRICE_RE}$"
)


def _timestamp(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        selected = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError):
        return None
    if selected.tzinfo is None or selected.utcoffset() is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def _decimal(value: object) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _legacy_price(value: str) -> Decimal | None:
    rendered = str(value).strip().replace("€", "").replace(" ", "")
    if not rendered or rendered.lower() in {"n.b.", "nb", "none"}:
        return None
    if "," in rendered:
        rendered = rendered.replace(".", "").replace(",", ".")
    try:
        selected = Decimal(rendered)
    except InvalidOperation:
        return None
    return selected if selected.is_finite() else None


def _read_json_rows(path: Path) -> tuple[list[Any], list[str]]:
    if not path.is_file():
        return [], []
    rows: list[Any] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(f"INVALID_JSON_LINE_{line_number}")
    return rows, errors


def _verified_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = _read_json_rows(path)
    verified: list[dict[str, Any]] = []
    expected_previous = "GENESIS"
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            errors.append(f"NON_OBJECT_EVENT_{index}")
            break
        record = dict(raw)
        record_hash = str(record.pop("record_hash", ""))
        if record.get("schema_version") != EVIDENCE_EVENT_SCHEMA:
            errors.append(f"UNSUPPORTED_SCHEMA_EVENT_{index}")
            break
        if str(record.get("previous_hash") or "") != expected_previous:
            errors.append(f"PREVIOUS_HASH_MISMATCH_EVENT_{index}")
            break
        calculated = stable_hash(record, length=64)
        if not record_hash or calculated != record_hash:
            errors.append(f"RECORD_HASH_MISMATCH_EVENT_{index}")
            break
        verified.append({**record, "record_hash": record_hash})
        expected_previous = record_hash
    return verified, errors


def _prospective_cohort(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cohort: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        alerted_at = _timestamp(event.get("recorded_at"))
        notification_id = str(event.get("notification_id") or "")
        if alerted_at is None or not notification_id:
            continue
        for raw in event.get("rows") or []:
            if not isinstance(raw, Mapping):
                continue
            opportunity_id = str(raw.get("opportunity_id") or "")
            if not opportunity_id or opportunity_id in seen:
                continue
            seen.add(opportunity_id)
            cohort.append(
                {
                    **dict(raw),
                    "alerted_at": utc_iso(alerted_at),
                    "notification_id": notification_id,
                    "evidence_source": "EXACT_PROSPECTIVE_LEDGER",
                }
            )
    return cohort


def _legacy_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = read_json(path)
    except (OSError, TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for message in payload:
        if not isinstance(message, Mapping) or str(
            message.get("message_type")
        ) != "TACTICAL_OPPORTUNITY_UPDATE":
            continue
        alerted_at = _timestamp(message.get("created_at"))
        notification_id = str(message.get("notification_id") or "")
        if alerted_at is None or not notification_id:
            continue
        for match in LEGACY_ROW_RE.finditer(str(message.get("message") or "")):
            index, market, timeframe, strategy, trigger, stop, target_1, target_2 = (
                match.groups()
            )
            normalized_strategy = strategy.strip()
            semantic_pullback = any(
                token in normalized_strategy.upper()
                for token in ("PULLBACK", "RETEST", "DIP")
            )
            rows.append(
                {
                    "opportunity_id": stable_hash(
                        ["legacy-preview", notification_id, index], length=32
                    ),
                    "notification_id": notification_id,
                    "alerted_at": utc_iso(alerted_at),
                    "signal_timestamp": None,
                    "market": market,
                    "timeframe": timeframe.strip(),
                    "strategy": normalized_strategy,
                    "strategy_dna_hash": "",
                    "trigger": float(trigger_value)
                    if (trigger_value := _legacy_price(trigger)) is not None
                    else None,
                    "stop": float(stop_value)
                    if (stop_value := _legacy_price(stop)) is not None
                    else None,
                    "target_1": float(target_1_value)
                    if (target_1_value := _legacy_price(target_1)) is not None
                    else None,
                    "target_2": float(target_2_value)
                    if (target_2_value := _legacy_price(target_2)) is not None
                    else None,
                    "entry_condition": (
                        "LOW_AT_OR_BELOW_TRIGGER"
                        if semantic_pullback
                        else "HIGH_AT_OR_ABOVE_TRIGGER"
                    ),
                    "entry_condition_source": "LEGACY_STRATEGY_NAME_INFERENCE",
                    "evaluation_roundtrip_cost_bps": float(
                        DEFAULT_ROUNDTRIP_COST_BPS
                    ),
                    "evidence_source": "LEGACY_PREVIEW_ROUNDED_LEVELS",
                }
            )
    return rows


def _timeframe_delta(label: str) -> timedelta:
    normalized = str(label).strip().lower()
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return timedelta(minutes=int(normalized[:-1]))
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return timedelta(hours=int(normalized[:-1]))
    return timedelta(minutes=15)


def _load_candles(
    settings: Settings,
    market: str,
    cache: dict[str, tuple[pd.DataFrame, str, timedelta]],
) -> tuple[pd.DataFrame, str, timedelta]:
    if market in cache:
        return cache[market]
    for timeframe in ("15m", "1h"):
        path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
        if not path.is_file():
            continue
        try:
            frame = pd.read_parquet(
                path,
                columns=["timestamp", "open", "high", "low", "close"],
            )
        except (OSError, ValueError):
            continue
        if "timestamp" not in frame.columns:
            continue
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True, errors="coerce"
        )
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=["timestamp", "high", "low", "close"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        result = (frame, timeframe, _timeframe_delta(timeframe))
        cache[market] = result
        return result
    result = (pd.DataFrame(), "NONE", timedelta(minutes=15))
    cache[market] = result
    return result


def _entry_hit(condition: str, low: Decimal, high: Decimal, trigger: Decimal) -> bool:
    if condition == "LOW_AT_OR_BELOW_TRIGGER":
        return low <= trigger
    if condition == "CANDLE_TOUCH_TRIGGER":
        return low <= trigger <= high
    return high >= trigger


def _evaluate_row(
    settings: Settings,
    row: Mapping[str, Any],
    *,
    observed_at: datetime,
    cache: dict[str, tuple[pd.DataFrame, str, timedelta]],
) -> dict[str, Any]:
    market = str(row.get("market") or "").upper()
    alerted_at = _timestamp(row.get("alerted_at"))
    entry = _decimal(row.get("trigger"))
    stop = _decimal(row.get("stop"))
    target_1 = _decimal(row.get("target_1"))
    target_2 = _decimal(row.get("target_2"))
    base = {
        "opportunity_id": str(row.get("opportunity_id") or ""),
        "notification_id": str(row.get("notification_id") or ""),
        "evidence_source": str(row.get("evidence_source") or ""),
        "market": market,
        "strategy": str(row.get("strategy") or "UNKNOWN"),
        "strategy_dna_hash": str(row.get("strategy_dna_hash") or ""),
        "timeframe": str(row.get("timeframe") or ""),
        "alerted_at": utc_iso(alerted_at) if alerted_at else None,
        "signal_timestamp": row.get("signal_timestamp"),
        "entry_condition": str(
            row.get("entry_condition") or "HIGH_AT_OR_ABOVE_TRIGGER"
        ),
        "trigger": float(entry) if entry is not None else None,
        "stop": float(stop) if stop is not None else None,
        "target_1": float(target_1) if target_1 is not None else None,
        "target_2": float(target_2) if target_2 is not None else None,
    }
    if (
        alerted_at is None
        or not market
        or entry is None
        or stop is None
        or target_1 is None
        or target_2 is None
        or entry <= 0
        or stop <= 0
        or not stop < entry < target_1 <= target_2
    ):
        return {
            **base,
            "outcome": "INVALID_LEVELS",
            "resolved_for_tp2_claim": False,
            "tp2_claim_success": False,
            "net_return_fraction": None,
            "candle_source_timeframe": None,
        }

    frame, candle_timeframe, interval = _load_candles(settings, market, cache)
    if frame.empty:
        return {
            **base,
            "outcome": "MISSING_MARKET_DATA",
            "resolved_for_tp2_claim": False,
            "tp2_claim_success": False,
            "net_return_fraction": None,
            "candle_source_timeframe": candle_timeframe,
        }

    horizon_end = alerted_at + HORIZON
    evaluation_end = min(observed_at, horizon_end)
    alerted_timestamp = pd.Timestamp(alerted_at)
    interval_pd = pd.Timedelta(interval)
    first_open = alerted_timestamp.ceil(interval_pd)
    if first_open == alerted_timestamp and alerted_at.microsecond:
        first_open += interval_pd
    candle_close = frame["timestamp"] + interval_pd
    selected = frame.loc[
        (frame["timestamp"] >= first_open)
        & (candle_close <= pd.Timestamp(evaluation_end))
    ]
    latest_available_close = (
        frame["timestamp"].iloc[-1].to_pydatetime() + interval
    )
    coverage_complete = latest_available_close >= horizon_end
    horizon_complete = observed_at >= horizon_end

    entered_at: datetime | None = None
    tp1_hit_at: datetime | None = None
    boundary_at: datetime | None = None
    outcome: str | None = None
    condition = str(base["entry_condition"])
    last_close: Decimal | None = None
    for candle in selected.itertuples(index=False):
        timestamp = candle.timestamp.to_pydatetime()
        low = Decimal(str(candle.low))
        high = Decimal(str(candle.high))
        last_close = Decimal(str(candle.close))
        if entered_at is None:
            if not _entry_hit(condition, low, high, entry):
                continue
            entered_at = timestamp
            stop_hit = low <= stop
            tp1_hit = high >= target_1
            tp2_hit = high >= target_2
            if tp1_hit:
                tp1_hit_at = timestamp
            if stop_hit and tp2_hit:
                outcome = "AMBIGUOUS_SAME_CANDLE"
            elif stop_hit:
                # The candle may have visited the stop before entry; treating
                # it as ambiguous is conservative and avoids invented order.
                outcome = "AMBIGUOUS_SAME_CANDLE"
            elif tp2_hit and condition != "HIGH_AT_OR_ABOVE_TRIGGER":
                outcome = "AMBIGUOUS_SAME_CANDLE"
            elif tp2_hit:
                outcome = "TP2_BEFORE_STOP"
            if outcome:
                boundary_at = timestamp
                break
            continue
        stop_hit = low <= stop
        tp1_hit = high >= target_1
        tp2_hit = high >= target_2
        if tp1_hit and tp1_hit_at is None:
            tp1_hit_at = timestamp
        if stop_hit and tp2_hit:
            outcome = "AMBIGUOUS_SAME_CANDLE"
        elif stop_hit:
            outcome = "STOP_BEFORE_TP2"
        elif tp2_hit:
            outcome = "TP2_BEFORE_STOP"
        if outcome:
            boundary_at = timestamp
            break

    if outcome is None:
        if horizon_complete and not coverage_complete:
            outcome = "MISSING_MARKET_DATA"
        elif entered_at is None:
            outcome = (
                "NOT_TRIGGERED_24H" if horizon_complete else "OPEN_NOT_TRIGGERED"
            )
        else:
            outcome = "TIME_EXIT_24H" if horizon_complete else "OPEN_TRIGGERED"

    exit_price: Decimal | None = None
    if outcome == "TP2_BEFORE_STOP":
        exit_price = target_2
    elif outcome in {"STOP_BEFORE_TP2", "AMBIGUOUS_SAME_CANDLE"}:
        exit_price = stop
    elif outcome == "TIME_EXIT_24H":
        exit_price = last_close
    gross_return = exit_price / entry - 1 if exit_price is not None else None
    cost_bps = _decimal(row.get("evaluation_roundtrip_cost_bps"))
    if cost_bps is None or cost_bps < 0:
        cost_bps = DEFAULT_ROUNDTRIP_COST_BPS
    net_return = (
        gross_return - cost_bps / Decimal("10000")
        if gross_return is not None
        else None
    )
    return {
        **base,
        "outcome": outcome,
        "entry_at": utc_iso(entered_at) if entered_at else None,
        "tp1_hit_at": utc_iso(tp1_hit_at) if tp1_hit_at else None,
        "boundary_at": utc_iso(boundary_at) if boundary_at else None,
        "horizon_end": utc_iso(horizon_end),
        "horizon_complete": horizon_complete,
        "coverage_complete": coverage_complete,
        "latest_available_candle_close": utc_iso(latest_available_close),
        "candle_source_timeframe": candle_timeframe,
        "roundtrip_cost_bps": float(cost_bps),
        "gross_return_fraction": float(gross_return)
        if gross_return is not None
        else None,
        "net_return_fraction": float(net_return)
        if net_return is not None
        else None,
        "resolved_for_tp2_claim": outcome in BOUNDARY_OUTCOMES,
        "tp2_claim_success": outcome == "TP2_BEFORE_STOP",
    }


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    counts = Counter(str(row.get("outcome") or "UNKNOWN") for row in selected)
    resolved = [row for row in selected if row.get("resolved_for_tp2_claim")]
    successes = sum(bool(row.get("tp2_claim_success")) for row in resolved)
    returns = [
        float(value)
        for row in selected
        if str(row.get("outcome")) in TERMINAL_RETURN_OUTCOMES
        and (value := row.get("net_return_fraction")) is not None
    ]
    return {
        "alert_count": len(selected),
        "outcome_counts": dict(sorted(counts.items())),
        "boundary_resolved_count": len(resolved),
        "tp2_before_stop_count": successes,
        "tp2_before_stop_fraction": successes / len(resolved) if resolved else None,
        "wilson_95pct_lower_bound": _wilson_lower(successes, len(resolved)),
        "terminal_net_return_count": len(returns),
        "mean_net_return_fraction": sum(returns) / len(returns) if returns else None,
    }


def _source_fingerprint(
    settings: Settings,
    *,
    evidence_path: Path,
    preview_path: Path,
    rows: Iterable[Mapping[str, Any]],
    observed_at: datetime,
) -> str:
    markets = sorted({str(row.get("market") or "") for row in rows if row.get("market")})
    paths = [evidence_path, preview_path]
    for market in markets:
        for timeframe in ("15m", "1h"):
            path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            if path.is_file():
                paths.append(path)
                break
    files = []
    for path in paths:
        try:
            stat = path.stat()
            files.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
        except OSError:
            files.append({"path": str(path.resolve()), "missing": True})
    cutoff = observed_at.replace(
        minute=(observed_at.minute // 15) * 15,
        second=0,
        microsecond=0,
    )
    return stable_hash({"files": files, "closed_candle_cutoff": utc_iso(cutoff)})


def build_telegram_signal_evidence(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build exact prospective evidence without changing any execution state."""

    selected_at = _timestamp(observed_at or utc_now()) or utc_now()
    notifications = settings.paths.output_dir / "notifications"
    evidence_path = notifications / "telegram_opportunity_evidence.jsonl"
    preview_path = notifications / "telegram_preview.json"
    artifact = settings.paths.output_dir / "operations" / "telegram_signal_evidence.json"
    events, integrity_errors = _verified_events(evidence_path)
    cohort = _prospective_cohort(events)
    legacy = _legacy_rows(preview_path)
    fingerprint = _source_fingerprint(
        settings,
        evidence_path=evidence_path,
        preview_path=preview_path,
        rows=[*cohort, *legacy],
        observed_at=selected_at,
    )
    if not force and artifact.is_file():
        try:
            existing = read_json(artifact)
        except (OSError, TypeError, ValueError):
            existing = {}
        if existing.get("source_fingerprint") == fingerprint:
            return existing

    cache: dict[str, tuple[pd.DataFrame, str, timedelta]] = {}
    prospective_outcomes = [
        _evaluate_row(settings, row, observed_at=selected_at, cache=cache)
        for row in cohort
    ]
    legacy_outcomes = [
        _evaluate_row(settings, row, observed_at=selected_at, cache=cache)
        for row in legacy
    ]
    prospective_summary = _summary(prospective_outcomes)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prospective_outcomes:
        by_strategy[str(row.get("strategy") or "UNKNOWN")].append(row)
    strategy_summaries = {
        strategy: _summary(rows) for strategy, rows in sorted(by_strategy.items())
    }
    lower_bound = prospective_summary["wilson_95pct_lower_bound"]
    mean_net = prospective_summary["mean_net_return_fraction"]
    gate_conditions = {
        "integrity_clean": not integrity_errors,
        "minimum_30_boundary_resolved_alerts": (
            prospective_summary["boundary_resolved_count"]
            >= MINIMUM_RESOLVED_ALERTS
        ),
        "positive_mean_net_return_after_35bps": (
            mean_net is not None and mean_net > 0.0
        ),
        "wilson_95pct_lower_bound_at_least_50pct": (
            lower_bound is not None
            and lower_bound >= MINIMUM_WILSON_LOWER_BOUND
        ),
    }
    eligible = all(gate_conditions.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(selected_at),
        "artifact": str(artifact.resolve()),
        "source_fingerprint": fingerprint,
        "claim_under_test": {
            "claim": "Approximately 95 percent of Telegram signals reach TP2",
            "status": (
                "SUPPORTED_BY_EXACT_PROSPECTIVE_EVIDENCE"
                if eligible
                and prospective_summary["tp2_before_stop_fraction"] is not None
                and prospective_summary["tp2_before_stop_fraction"] >= 0.95
                else "NOT_CONFIRMED"
            ),
            "reason": (
                "EXACT_PROSPECTIVE_SAMPLE_NOT_YET_SUFFICIENT"
                if prospective_summary["boundary_resolved_count"]
                < MINIMUM_RESOLVED_ALERTS
                else "EXACT_PROSPECTIVE_RESULTS_DO_NOT_SUPPORT_95_PERCENT"
            ),
        },
        "prospective_exact_evidence": {
            "ledger": str(evidence_path.resolve()),
            "hash_chain_status": "VALID" if not integrity_errors else "INVALID",
            "integrity_errors": integrity_errors,
            "event_count": len(events),
            "cohort_rule": "FIRST_HASH_VALID_ALERT_PER_OPPORTUNITY_ID",
            "measurement_rule": (
                "FIRST_FULLY_CLOSED_15M_CANDLE_AFTER_ALERT_THROUGH_24H;_"
                "SAME_CANDLE_AMBIGUITY_IS_A_CONSERVATIVE_FAILURE"
            ),
            "summary": prospective_summary,
            "by_strategy": strategy_summaries,
            "outcomes": prospective_outcomes,
        },
        "legacy_preview_diagnostic": {
            "status": "INDICATIVE_ONLY_ROUNDED_LEVELS",
            "excluded_from_all_promotion_and_authority_decisions": True,
            "limitations": [
                "DISPLAYED_PRICES_ARE_ROUNDED",
                "NO_CANONICAL_OPPORTUNITY_ID_IN_PREVIEW",
                "ENTRY_DIRECTION_INFERRED_FROM_STRATEGY_NAME",
                "PREVIEW_CACHE_IS_NOT_A_COMPLETE_IMMUTABLE_POPULATION",
            ],
            "summary": _summary(legacy_outcomes),
            "by_strategy": {
                strategy: _summary(rows)
                for strategy, rows in sorted(
                    (
                        (strategy, [row for row in legacy_outcomes if row["strategy"] == strategy])
                        for strategy in sorted(
                            {str(row["strategy"]) for row in legacy_outcomes}
                        )
                    ),
                    key=lambda item: item[0],
                )
            },
            "outcomes": legacy_outcomes,
        },
        "paper_shadow_gate": {
            "status": (
                "ELIGIBLE_FOR_PAPER_SHADOW_REVIEW"
                if eligible
                else "COLLECT_EXACT_PROSPECTIVE_EVIDENCE"
            ),
            "conditions": gate_conditions,
            "automatic_live_authority_changes": False,
            "live_order_authority_granted": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
        "execution_mutations": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    payload["evidence_hash"] = stable_hash(payload, length=64)
    atomic_write_json(artifact, payload)
    return payload


__all__ = ["build_telegram_signal_evidence"]
