"""Chronological shadow-only drift diagnostics for opportunity intelligence."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.common import atomic_write_json, stable_hash, utc_iso

SCHEMA_VERSION = "opportunity_intelligence_drift_v1"
CATEGORICAL_FEATURES = {
    "market",
    "family",
    "context_timeframe",
    "macro_regime",
    "trade_type",
    "market_mode",
}


def _finite(value: object) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * min(1.0, max(0.0, fraction))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _psi(reference: list[float], recent: list[float]) -> float | None:
    if len(reference) < 20 or len(recent) < 20:
        return None
    internal = sorted(
        {
            _quantile(reference, index / 10.0)
            for index in range(1, 10)
        }
    )
    boundaries = [-math.inf, *internal, math.inf]
    if len(boundaries) < 3:
        return 0.0

    def proportions(values: list[float]) -> list[float]:
        counts = [0] * (len(boundaries) - 1)
        for value in values:
            for index, (left, right) in enumerate(
                zip(boundaries, boundaries[1:], strict=True)
            ):
                if left < value <= right:
                    counts[index] += 1
                    break
        epsilon = 1e-6
        return [max(epsilon, count / len(values)) for count in counts]

    expected = proportions(reference)
    actual = proportions(recent)
    return sum(
        (observed - baseline) * math.log(observed / baseline)
        for baseline, observed in zip(expected, actual, strict=True)
    )


def _total_variation(reference: list[str], recent: list[str]) -> float | None:
    if len(reference) < 20 or len(recent) < 20:
        return None
    reference_counts = Counter(reference)
    recent_counts = Counter(recent)
    categories = set(reference_counts) | set(recent_counts)
    return 0.5 * sum(
        abs(
            reference_counts[category] / len(reference)
            - recent_counts[category] / len(recent)
        )
        for category in categories
    )


def _brier(rows: list[Mapping[str, Any]]) -> float | None:
    observations = [
        (_finite(row.get("prediction")), _finite(row.get("label")))
        for row in rows
    ]
    valid = [(prediction, label) for prediction, label in observations if prediction is not None and label is not None]
    if not valid:
        return None
    return sum((prediction - label) ** 2 for prediction, label in valid) / len(valid)


def _expected_calibration_error(
    rows: list[Mapping[str, Any]], *, bins: int = 10
) -> float | None:
    valid = [
        (prediction, label)
        for row in rows
        if (prediction := _finite(row.get("prediction"))) is not None
        and (label := _finite(row.get("label"))) is not None
    ]
    if not valid:
        return None
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            (prediction, label)
            for prediction, label in valid
            if lower <= prediction < upper
            or (index == bins - 1 and prediction == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(row[0] for row in selected) / len(selected)
        observed = sum(row[1] for row in selected) / len(selected)
        error += len(selected) / len(valid) * abs(confidence - observed)
    return error


def _level(value: float | None) -> str:
    if value is None:
        return "INSUFFICIENT"
    if value >= 0.25:
        return "CRITICAL"
    if value >= 0.10:
        return "WARNING"
    return "STABLE"


def build_intelligence_drift_report(
    root: Path,
    *,
    rows: Iterable[Mapping[str, Any]],
    oos_predictions: Iterable[Mapping[str, Any]] = (),
    training_data_hash: str | None = None,
) -> dict[str, Any]:
    """Measure chronological drift without obtaining execution authority."""

    selected = sorted(
        (dict(row) for row in rows),
        key=lambda row: str(row.get("decision_timestamp") or ""),
    )
    predictions = sorted(
        (dict(row) for row in oos_predictions),
        key=lambda row: str(row.get("decision_timestamp") or ""),
    )
    output = root / "output" / "intelligence" / "drift_report.json"
    minimum_rows = 100
    if len(selected) < minimum_rows:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_iso(),
            "status": "INSUFFICIENT_DATA",
            "authority": "SHADOW_ONLY",
            "live_decision_influence": False,
            "row_count": len(selected),
            "minimum_rows": minimum_rows,
            "training_data_hash": training_data_hash,
            "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(output, payload)
        return {**payload, "artifact": str(output)}

    split_at = max(50, min(len(selected) - 50, int(len(selected) * 0.70)))
    reference = selected[:split_at]
    recent = selected[split_at:]
    feature_names = sorted(
        {
            str(name)
            for row in selected
            for name in dict(row.get("features") or {})
        }
    )
    feature_drift: dict[str, dict[str, Any]] = {}
    critical_features: list[str] = []
    warning_features: list[str] = []
    for feature in feature_names:
        reference_raw = [dict(row.get("features") or {}).get(feature) for row in reference]
        recent_raw = [dict(row.get("features") or {}).get(feature) for row in recent]
        reference_missing = sum(value is None for value in reference_raw) / len(reference_raw)
        recent_missing = sum(value is None for value in recent_raw) / len(recent_raw)
        missingness_delta = recent_missing - reference_missing
        if feature in CATEGORICAL_FEATURES:
            metric = _total_variation(
                [str(value) for value in reference_raw if value is not None],
                [str(value) for value in recent_raw if value is not None],
            )
            metric_name = "total_variation_distance"
        else:
            metric = _psi(
                [value for raw in reference_raw if (value := _finite(raw)) is not None],
                [value for raw in recent_raw if (value := _finite(raw)) is not None],
            )
            metric_name = "population_stability_index"
        level = _level(metric)
        if abs(missingness_delta) >= 0.25:
            level = "CRITICAL"
        elif abs(missingness_delta) >= 0.10 and level == "STABLE":
            level = "WARNING"
        if level == "CRITICAL":
            critical_features.append(feature)
        elif level == "WARNING":
            warning_features.append(feature)
        feature_drift[feature] = {
            "metric": metric_name,
            "value": metric,
            "level": level,
            "reference_missing_fraction": reference_missing,
            "recent_missing_fraction": recent_missing,
            "missingness_delta": missingness_delta,
        }

    reference_labels = [
        value
        for row in reference
        if (value := _finite(row.get("net_profitable"))) is not None
    ]
    recent_labels = [
        value
        for row in recent
        if (value := _finite(row.get("net_profitable"))) is not None
    ]
    reference_positive_rate = (
        sum(reference_labels) / len(reference_labels) if reference_labels else None
    )
    recent_positive_rate = (
        sum(recent_labels) / len(recent_labels) if recent_labels else None
    )
    positive_rate_delta = (
        recent_positive_rate - reference_positive_rate
        if reference_positive_rate is not None and recent_positive_rate is not None
        else None
    )

    prediction_split = max(1, int(len(predictions) * 0.60))
    reference_predictions = predictions[:prediction_split]
    recent_predictions = predictions[prediction_split:]
    prediction_psi = _psi(
        [value for row in reference_predictions if (value := _finite(row.get("prediction"))) is not None],
        [value for row in recent_predictions if (value := _finite(row.get("prediction"))) is not None],
    )
    reference_brier = _brier(reference_predictions)
    recent_brier = _brier(recent_predictions)
    reference_ece = _expected_calibration_error(reference_predictions)
    recent_ece = _expected_calibration_error(recent_predictions)
    brier_delta = (
        recent_brier - reference_brier
        if reference_brier is not None and recent_brier is not None
        else None
    )
    calibration_delta = (
        recent_ece - reference_ece
        if reference_ece is not None and recent_ece is not None
        else None
    )

    status = "STABLE"
    if critical_features or _level(prediction_psi) == "CRITICAL":
        status = "CRITICAL_DRIFT_SHADOW_WARNING"
    elif warning_features or _level(prediction_psi) == "WARNING":
        status = "WARNING_DRIFT_SHADOW"
    if positive_rate_delta is not None and abs(positive_rate_delta) >= 0.20:
        status = "CRITICAL_DRIFT_SHADOW_WARNING"
    elif (
        positive_rate_delta is not None
        and abs(positive_rate_delta) >= 0.10
        and status == "STABLE"
    ):
        status = "WARNING_DRIFT_SHADOW"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": status,
        "authority": "SHADOW_ONLY",
        "live_decision_influence": False,
        "fallback_policy": "DETERMINISTIC_RULE_ENGINE",
        "row_count": len(selected),
        "reference_rows": len(reference),
        "recent_rows": len(recent),
        "training_data_hash": training_data_hash,
        "feature_drift": feature_drift,
        "critical_features": critical_features,
        "warning_features": warning_features,
        "label_drift": {
            "reference_positive_rate": reference_positive_rate,
            "recent_positive_rate": recent_positive_rate,
            "positive_rate_delta": positive_rate_delta,
        },
        "prediction_drift": {
            "chronological_oos_prediction_count": len(predictions),
            "population_stability_index": prediction_psi,
            "level": _level(prediction_psi),
            "reference_brier": reference_brier,
            "recent_brier": recent_brier,
            "brier_delta": brier_delta,
            "reference_expected_calibration_error": reference_ece,
            "recent_expected_calibration_error": recent_ece,
            "calibration_error_delta": calibration_delta,
        },
        "thresholds": {
            "warning": 0.10,
            "critical": 0.25,
            "label_rate_warning": 0.10,
            "label_rate_critical": 0.20,
        },
        "interpretation": {
            "drift_is_not_alpha": True,
            "drift_cannot_create_orders": True,
            "drift_cannot_veto_deterministic_entries": True,
            "drift_cannot_change_sizing_or_authority": True,
            "chronological_split": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    payload["report_hash"] = stable_hash(payload, length=64)
    atomic_write_json(output, payload)
    return {**payload, "artifact": str(output)}


__all__ = ["build_intelligence_drift_report"]
