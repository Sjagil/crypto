"""Conservative, point-in-time NetR calibration for active-swing candidates."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping

from config.settings import Settings
from utils.common import atomic_write_json, read_json, utc_iso

MINIMUM_EXACT_GROUP_ROWS = 30
MINIMUM_POSITIVE_ROWS = 5
MINIMUM_NEGATIVE_ROWS = 5
MINIMUM_DISTINCT_MARKETS = 3
MINIMUM_LABEL_SPAN_DAYS = 7.0
ZERO_PRIOR_STRENGTH = 20
ONE_SIDED_95_Z = 1.645


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("family") or candidate.get("strategy_family") or ""),
        str(candidate.get("entry_timeframe") or candidate.get("timeframe") or ""),
        str(
            candidate.get("confirmation_timeframe")
            or candidate.get("setup_timeframe")
            or ""
        ),
    )


def _load_canonical_rows(settings: Settings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status_path = settings.paths.output_dir / "ml" / "canonical_training_status.json"
    if not status_path.is_file():
        return {}, []
    try:
        status_value = read_json(status_path)
    except (OSError, TypeError, ValueError):
        return {}, []
    status = dict(status_value) if isinstance(status_value, Mapping) else {}
    rows_artifact = status.get("rows_artifact")
    if status.get("dataset_registered") is not True or not rows_artifact:
        return status, []
    path = Path(str(rows_artifact))
    if not path.is_file():
        return status, []
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return status, []
    if not isinstance(value, Mapping):
        return status, []
    return status, [
        dict(row)
        for row in value.get("rows") or []
        if isinstance(row, Mapping)
    ]


def _calibrate_group(
    rows: Iterable[Mapping[str, Any]],
    *,
    family: str,
    entry_timeframe: str,
    setup_timeframe: str,
    as_of: datetime,
) -> dict[str, Any]:
    selected: list[tuple[dict[str, Any], float, datetime]] = []
    future_labels_excluded = 0
    for source in rows:
        row = dict(source)
        values = dict(row.get("features") or {})
        if (
            str(values.get("family") or "") != family
            or str(
                values.get("entry_timeframe")
                or values.get("observation_timeframe")
                or ""
            )
            != entry_timeframe
            or str(
                values.get("setup_timeframe")
                or values.get("context_timeframe")
                or ""
            )
            != setup_timeframe
        ):
            continue
        label_available = _timestamp(
            row.get("label_available_at") or row.get("label_end")
        )
        decision = _timestamp(row.get("decision_timestamp"))
        net_r = _finite(row.get("net_return_r"))
        if label_available is None or decision is None or net_r is None:
            continue
        if label_available > as_of:
            future_labels_excluded += 1
            continue
        if (
            row.get("canonical_point_in_time_ready") is not True
            or row.get("label_uses_future_features") is not False
        ):
            continue
        selected.append((row, net_r, decision))
    values = [net_r for _, net_r, _ in selected]
    positives = sum(value > 0 for value in values)
    negatives = sum(value <= 0 for value in values)
    markets = {
        str(dict(row.get("features") or {}).get("market") or "")
        for row, _, _ in selected
    }
    markets.discard("")
    decisions = [decision for _, _, decision in selected]
    span_days = (
        (max(decisions) - min(decisions)).total_seconds() / 86400
        if len(decisions) > 1
        else 0.0
    )
    failures: list[str] = []
    if len(values) < MINIMUM_EXACT_GROUP_ROWS:
        failures.append("MINIMUM_EXACT_GROUP_ROWS_NOT_MET")
    if positives < MINIMUM_POSITIVE_ROWS:
        failures.append("MINIMUM_POSITIVE_ROWS_NOT_MET")
    if negatives < MINIMUM_NEGATIVE_ROWS:
        failures.append("MINIMUM_NEGATIVE_ROWS_NOT_MET")
    if len(markets) < MINIMUM_DISTINCT_MARKETS:
        failures.append("MINIMUM_DISTINCT_MARKETS_NOT_MET")
    if span_days < MINIMUM_LABEL_SPAN_DAYS:
        failures.append("MINIMUM_LABEL_SPAN_NOT_MET")
    mean = fmean(values) if values else None
    sample_std = stdev(values) if len(values) > 1 else None
    lower_bound = (
        mean - ONE_SIDED_95_Z * sample_std / math.sqrt(len(values))
        if mean is not None and sample_std is not None
        else None
    )
    shrunk_mean = (
        mean * len(values) / (len(values) + ZERO_PRIOR_STRENGTH)
        if mean is not None
        else None
    )
    conservative = (
        min(lower_bound, shrunk_mean)
        if lower_bound is not None and shrunk_mean is not None and not failures
        else None
    )
    return {
        "status": (
            "CALIBRATED_PROSPECTIVE_EXPECTATION"
            if conservative is not None
            else "INSUFFICIENT_PROSPECTIVE_SUPPORT"
        ),
        "family": family,
        "entry_timeframe": entry_timeframe,
        "setup_timeframe": setup_timeframe,
        "row_count": len(values),
        "positive_rows": positives,
        "negative_rows": negatives,
        "distinct_market_count": len(markets),
        "label_span_days": span_days,
        "future_labels_excluded": future_labels_excluded,
        "sample_mean_net_r": mean,
        "sample_std_net_r": sample_std,
        "zero_prior_shrunk_mean_net_r": shrunk_mean,
        "one_sided_95_lower_net_r": lower_bound,
        "expected_net_r": conservative,
        "positive_expected_net_r": bool(
            conservative is not None and conservative > 0
        ),
        "failures": failures,
        "point_in_time_only": True,
        "historical_backfill_used": False,
        "may_change_live_authority": False,
    }


def build_prospective_net_r_calibration(
    settings: Settings,
    candidates: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Calibrate exact family/timeframe NetR without future-label leakage."""

    observed_at = as_of or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("NetR calibration as_of must be timezone-aware")
    observed_at = observed_at.astimezone(UTC)
    status, rows = _load_canonical_rows(settings)
    estimates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        identity = str(candidate.get("opportunity_id") or "")
        if not identity or identity in estimates:
            continue
        family, entry_timeframe, setup_timeframe = _candidate_key(candidate)
        estimates[identity] = _calibrate_group(
            rows,
            family=family,
            entry_timeframe=entry_timeframe,
            setup_timeframe=setup_timeframe,
            as_of=observed_at,
        )
    payload = {
        "schema_version": "prospective_net_r_calibration_v1",
        "generated_at": utc_iso(),
        "as_of": observed_at.isoformat(),
        "status": (
            "CALIBRATED"
            if any(row.get("expected_net_r") is not None for row in estimates.values())
            else "DATA_PENDING"
        ),
        "canonical_dataset_registered": status.get("dataset_registered") is True,
        "canonical_dataset_id": status.get("dataset_id"),
        "canonical_row_count": int(status.get("canonical_row_count") or 0),
        "minimums": {
            "exact_group_rows": MINIMUM_EXACT_GROUP_ROWS,
            "positive_rows": MINIMUM_POSITIVE_ROWS,
            "negative_rows": MINIMUM_NEGATIVE_ROWS,
            "distinct_markets": MINIMUM_DISTINCT_MARKETS,
            "label_span_days": MINIMUM_LABEL_SPAN_DAYS,
        },
        "estimates_by_opportunity_id": estimates,
        "financial_state_changed": False,
        "execution_authority_changed": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    artifact = settings.paths.output_dir / "product" / "prospective_net_r.json"
    atomic_write_json(artifact, payload)
    return payload


__all__ = ["build_prospective_net_r_calibration"]
