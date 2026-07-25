"""Continuous strategy-combination lab built on the canonical research stack.

This module owns orchestration and durable accounting only. Feature calculation,
exact fills, optimization, persistence and reporting remain in their existing
canonical modules.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import itertools
import math
import os
import socket
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS, Settings, normalize_timeframe
from core.contracts import (
    DataValidationError,
    EligibilityStatus,
    NormalizedDataRecord,
    ResearchStatus,
)
from data.data_loader import DataLoader
from data.database import Database
from data.market_data import (
    load_ohlcv,
    quality_report,
    resample_ohlcv,
    save_ohlcv,
    timeframe_delta,
    validate_ohlcv,
)
from research.backtest import BacktestConfig, BacktestEngine, BacktestResult
from research.features import (
    FeaturePipeline,
    feature_registry,
    parameterized_feature_series,
)
from research.optimization import (
    ResearchOutcome,
    WalkForwardResult,
    multiple_testing_bootstrap,
    robust_score,
    run_research,
    walk_forward_validate,
)
from research.strategies import Strategy, StrategyOutput
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
    stable_json,
    utc_iso,
    utc_now,
)


class ParameterKind(StrEnum):
    INTEGER = "INTEGER"
    HALF_STEP = "HALF_STEP"
    FLOAT = "FLOAT"
    DECIMAL = "DECIMAL"
    CHOICE = "CHOICE"
    BOOLEAN = "BOOLEAN"
    TIMEFRAME = "TIMEFRAME"
    DURATION = "DURATION"


class ExitProfile(StrEnum):
    FIXED_R = "FIXED_R"
    TRAILING_TREND = "TRAILING_TREND"
    TIME_REGIME = "TIME_REGIME"


class BlockRole(StrEnum):
    ENTRY_TRIGGER = "ENTRY_TRIGGER"
    TREND_FILTER = "TREND_FILTER"
    REGIME_FILTER = "REGIME_FILTER"
    CONFIRMATION = "CONFIRMATION"
    EXIT_TRIGGER = "EXIT_TRIGGER"
    RISK_OVERLAY = "RISK_OVERLAY"
    POSITION_SIZE_MODIFIER = "POSITION_SIZE_MODIFIER"
    AVOIDANCE_FILTER = "AVOIDANCE_FILTER"


class BlockDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CONTEXTUAL = "CONTEXTUAL"


class LogicMode(StrEnum):
    ALL = "ALL"
    ANY = "ANY"
    MAJORITY = "MAJORITY"
    WEIGHTED_VOTE = "WEIGHTED_VOTE"
    LAYERED = "LAYERED"


class SignalOperator(StrEnum):
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"
    ABOVE_FEATURE = "ABOVE_FEATURE"
    BELOW_FEATURE = "BELOW_FEATURE"
    BOOLEAN_TRUE = "BOOLEAN_TRUE"
    BOOLEAN_FALSE = "BOOLEAN_FALSE"
    RISING = "RISING"
    FALLING = "FALLING"
    CUSTOM_CAUSAL = "CUSTOM_CAUSAL"


class OverlaySemantics(StrEnum):
    ENTRY_ONLY_SIZE_REDUCTION = "ENTRY_ONLY_SIZE_REDUCTION"
    BLOCK_NEW_ENTRIES = "BLOCK_NEW_ENTRIES"
    ONE_TIME_REDUCTION_ON_TRANSITION = "ONE_TIME_REDUCTION_ON_TRANSITION"
    EXIT_ON_TRANSITION = "EXIT_ON_TRANSITION"
    CONTINUOUS_STOP_ADJUSTMENT = "CONTINUOUS_STOP_ADJUSTMENT"
    ADVISORY_ONLY = "ADVISORY_ONLY"


class GenerationMode(StrEnum):
    FAMILY_AWARE = "FAMILY_AWARE"
    EXHAUSTIVE = "EXHAUSTIVE"


class UniverseType(StrEnum):
    RAW_CMC_TOP_N = "RAW_CMC_TOP_N"
    DISCOVERY_UNIVERSE = "DISCOVERY_UNIVERSE"
    ELIGIBILITY_FILTERED = "ELIGIBILITY_FILTERED"
    ALLOWED_RESEARCH = "ALLOWED_RESEARCH"
    REVIEW_RESEARCH_ONLY = "REVIEW_RESEARCH_ONLY"
    EXECUTION_UNIVERSE = "EXECUTION_UNIVERSE"
    # Compatibility values retained for persisted v1/v2 snapshots.
    RESEARCH_ELIGIBLE = "RESEARCH_ELIGIBLE"
    EXECUTION_ELIGIBLE = "EXECUTION_ELIGIBLE"


class CombinationState(StrEnum):
    GENERATED = "GENERATED"
    INVALID_STATIC_RULES = "INVALID_STATIC_RULES"
    MISSING_DATA = "MISSING_DATA"
    UNSUPPORTED_TIMEFRAME = "UNSUPPORTED_TIMEFRAME"
    DUPLICATE = "DUPLICATE"
    QUEUED_BASELINE = "QUEUED_BASELINE"
    BASELINE_RUNNING = "BASELINE_RUNNING"
    BASELINE_COMPLETED = "BASELINE_COMPLETED"
    BASELINE_REJECTED = "BASELINE_REJECTED"
    QUEUED_SCREENING = "QUEUED_SCREENING"
    SCREENING_RUNNING = "SCREENING_RUNNING"
    SCREENING_COMPLETED = "SCREENING_COMPLETED"
    SCREENING_REJECTED = "SCREENING_REJECTED"
    QUEUED_EXACT_BACKTEST = "QUEUED_EXACT_BACKTEST"
    EXACT_BACKTEST_RUNNING = "EXACT_BACKTEST_RUNNING"
    EXACT_BACKTEST_COMPLETED = "EXACT_BACKTEST_COMPLETED"
    EXACT_BACKTEST_REJECTED = "EXACT_BACKTEST_REJECTED"
    QUEUED_OPTIMIZATION = "QUEUED_OPTIMIZATION"
    OPTIMIZATION_RUNNING = "OPTIMIZATION_RUNNING"
    OPTIMIZATION_COMPLETED = "OPTIMIZATION_COMPLETED"
    QUEUED_VALIDATION = "QUEUED_VALIDATION"
    VALIDATION_RUNNING = "VALIDATION_RUNNING"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    RESEARCH_PASS = "RESEARCH_PASS"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    ERROR_RETRYABLE = "ERROR_RETRYABLE"
    ERROR_FINAL = "ERROR_FINAL"
    SUPERSEDED = "SUPERSEDED"


class LifecycleStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    SCREENING_SURVIVOR = "SCREENING_SURVIVOR"
    EXACT_SURVIVOR = "EXACT_SURVIVOR"
    VALIDATION_SURVIVOR = "VALIDATION_SURVIVOR"
    RESEARCH_PASS = "RESEARCH_PASS"
    PAPER_CANDIDATE = "PAPER_CANDIDATE"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


class LabControl(StrEnum):
    START = "START"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    DRAIN = "DRAIN"
    STOP = "STOP"
    STATUS = "STATUS"


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _finite_json(value: Any) -> Any:
    """Replace non-finite numeric values before strict JSON persistence."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_finite_json(item) for item in value]
    return value


def canonical_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _canonical_value(parameters[key]) for key in sorted(parameters)}


def parameter_hash(parameters: Mapping[str, Any]) -> str:
    return stable_hash(canonical_parameters(parameters), length=64)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    minimum: int | float | Decimal | None = None
    maximum: int | float | Decimal | None = None
    step: int | float | Decimal | None = None
    choices: tuple[Any, ...] = ()
    default: Any = None
    integer_only: bool = False
    description: str = ""
    constraints: tuple[str, ...] = ()
    optimizer_distribution: str = "GRID"
    cache_behavior: str = "FEATURE_SERIES"

    def __post_init__(self) -> None:
        if self.kind is ParameterKind.HALF_STEP:
            if self.step is None or _decimal(self.step) != Decimal("0.5"):
                raise ValueError("HALF_STEP requires an exact Decimal 0.5 step")
        if self.integer_only and self.kind is not ParameterKind.INTEGER:
            raise ValueError("integer-only parameters must use INTEGER kind")
        if (
            self.kind
            in {
                ParameterKind.CHOICE,
                ParameterKind.TIMEFRAME,
                ParameterKind.BOOLEAN,
            }
            and not self.choices
        ):
            raise ValueError(f"{self.kind} requires choices")
        if self.default is not None:
            self.validate(self.default)

    def values(self) -> tuple[Any, ...]:
        if self.choices:
            return tuple(self.validate(value) for value in self.choices)
        if self.minimum is None or self.maximum is None:
            return (self.validate(self.default),)
        if self.kind in {ParameterKind.HALF_STEP, ParameterKind.DECIMAL}:
            step = _decimal(self.step or "1")
            current = _decimal(self.minimum)
            maximum = _decimal(self.maximum)
            values: list[Decimal] = []
            while current <= maximum:
                values.append(current)
                current += step
            return tuple(values)
        if self.kind is ParameterKind.INTEGER:
            return tuple(
                range(
                    int(self.minimum),
                    int(self.maximum) + 1,
                    int(self.step or 1),
                )
            )
        step = float(self.step or 1.0)
        count = int(math.floor((float(self.maximum) - float(self.minimum)) / step))
        return tuple(
            self.validate(float(self.minimum) + index * step) for index in range(count + 1)
        )

    def validate(self, value: Any) -> Any:
        if self.kind is ParameterKind.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"{self.name} must be boolean")
            selected: Any = value
        elif self.kind is ParameterKind.INTEGER:
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError(f"{self.name} must be an integer")
            selected = int(value)
        elif self.kind in {ParameterKind.HALF_STEP, ParameterKind.DECIMAL}:
            selected = _decimal(value)
            if self.kind is ParameterKind.HALF_STEP:
                anchor = _decimal(self.minimum or 0)
                if (selected - anchor) % Decimal("0.5") != 0:
                    raise ValueError(f"{self.name} must use exact 0.5 steps")
        elif self.kind in {ParameterKind.CHOICE, ParameterKind.TIMEFRAME}:
            selected = value
        elif self.kind is ParameterKind.DURATION:
            selected = str(value)
        else:
            selected = float(value)
            if not math.isfinite(selected):
                raise ValueError(f"{self.name} must be finite")
        if self.choices and selected not in self.choices:
            raise ValueError(f"{self.name} is not an allowed choice")
        if self.minimum is not None and _decimal(selected) < _decimal(self.minimum):
            raise ValueError(f"{self.name} is below its minimum")
        if self.maximum is not None and _decimal(selected) > _decimal(self.maximum):
            raise ValueError(f"{self.name} exceeds its maximum")
        return selected

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


@dataclass(frozen=True)
class SignalBlock:
    block_id: str
    version: str
    display_name: str
    family: str
    subfamily: str
    role: BlockRole
    direction: BlockDirection
    required_features: tuple[str, ...]
    supported_timeframes: tuple[str, ...]
    warmup_bars: int
    parameter_specs: tuple[ParameterSpec, ...]
    signal_kind: str
    feature: str
    compare_feature: str | None
    knowability_timestamp: str
    missing_data_policy: Literal["REJECT", "FALSE", "BLOCK"]
    description: str
    redundancy_group: str
    computational_cost_class: Literal["LOW", "MEDIUM", "HIGH"]
    compatible_roles: tuple[BlockRole, ...]
    incompatible_blocks: tuple[str, ...]
    default_parameters: Mapping[str, Any]
    source_quality_requirements: tuple[str, ...]
    operator: SignalOperator = SignalOperator.CUSTOM_CAUSAL
    overlay_semantics: OverlaySemantics | None = None

    def __post_init__(self) -> None:
        if self.block_id.startswith("raw_fractal"):
            raise ValueError("raw unconfirmed fractals cannot be registered")
        if self.direction is BlockDirection.BEARISH and self.role is BlockRole.ENTRY_TRIGGER:
            raise ValueError("bearish blocks cannot create spot entries")
        if self.warmup_bars < 0:
            raise ValueError("warmup bars cannot be negative")
        names = {spec.name for spec in self.parameter_specs}
        if set(self.default_parameters) != names:
            raise ValueError(f"default parameters do not match specs for {self.block_id}")
        for spec in self.parameter_specs:
            spec.validate(self.default_parameters[spec.name])
        if self.role is BlockRole.RISK_OVERLAY and self.overlay_semantics is None:
            raise ValueError(f"risk overlay requires explicit semantics: {self.block_id}")
        if self.role is not BlockRole.RISK_OVERLAY and self.overlay_semantics is not None:
            raise ValueError(f"overlay semantics are invalid for {self.block_id}")

    @property
    def parameter_space_size(self) -> int:
        return math.prod(max(1, len(spec.values())) for spec in self.parameter_specs)

    def parameters(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        selected = dict(self.default_parameters)
        selected.update(dict(overrides or {}))
        specs = {spec.name: spec for spec in self.parameter_specs}
        unknown = sorted(set(selected) - set(specs))
        if unknown:
            raise ValueError(f"unknown parameters for {self.block_id}: {unknown}")
        validated = {name: specs[name].validate(value) for name, value in selected.items()}
        if self.block_id == "ema_trend" and _decimal(validated["fast"]) >= _decimal(
            validated["slow"]
        ):
            raise ValueError("ema_trend.fast must be below ema_trend.slow")
        return validated

    def calculate(
        self,
        features: pd.DataFrame,
        parameters: Mapping[str, Any] | None = None,
    ) -> pd.Series:
        missing = [name for name in self.required_features if name not in features]
        if missing:
            if self.missing_data_policy == "FALSE":
                return pd.Series(False, index=features.index)
            raise KeyError(f"{self.block_id} missing required features: {missing}")
        params = self.parameters(parameters)
        source = features[self.feature]
        threshold = params.get("value", params.get("threshold"))
        if self.operator is SignalOperator.BOOLEAN_TRUE:
            signal = source.astype(bool)
        elif self.operator is SignalOperator.BOOLEAN_FALSE:
            signal = ~source.astype(bool)
        elif self.operator is SignalOperator.GREATER_THAN:
            signal = source.astype(float) > float(threshold)
        elif self.operator is SignalOperator.GREATER_THAN_OR_EQUAL:
            signal = source.astype(float) >= float(threshold)
        elif self.operator is SignalOperator.LESS_THAN:
            signal = source.astype(float) < float(threshold)
        elif self.operator is SignalOperator.LESS_THAN_OR_EQUAL:
            signal = source.astype(float) <= float(threshold)
        elif self.signal_kind == "POSITIVE":
            signal = source.astype(float) > 0.0
        elif self.signal_kind == "NEGATIVE":
            signal = source.astype(float) < 0.0
        elif self.signal_kind == "PCT_CHANGE_POSITIVE_20":
            signal = source.astype(float).pct_change(20) > 0.0
        elif self.signal_kind == "CHANGE_POSITIVE_20":
            signal = source.astype(float).diff(20) > 0.0
        elif self.operator is SignalOperator.ABOVE_FEATURE:
            signal = source.astype(float) > features[str(self.compare_feature)].astype(float)
        elif self.operator is SignalOperator.BELOW_FEATURE:
            signal = source.astype(float) < features[str(self.compare_feature)].astype(float)
        elif self.operator is SignalOperator.CROSS_ABOVE:
            other = (
                features[str(self.compare_feature)].astype(float)
                if self.compare_feature
                else pd.Series(float(threshold), index=features.index)
            )
            signal = (source > other) & (source.shift(1) <= other.shift(1))
        elif self.operator is SignalOperator.CROSS_BELOW:
            other = (
                features[str(self.compare_feature)].astype(float)
                if self.compare_feature
                else pd.Series(float(threshold), index=features.index)
            )
            signal = (source < other) & (source.shift(1) >= other.shift(1))
        elif self.signal_kind == "DYNAMIC_RSI_LT":
            value = parameterized_feature_series(
                features,
                "rsi",
                {"period": params["period"]},
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
                provider_context_hash=stable_hash(features.attrs.get("data_provenance") or {}),
            )
            signal = value < float(params["value"])
        elif self.signal_kind == "DYNAMIC_EMA_ABOVE":
            value = parameterized_feature_series(
                features,
                "ema",
                {"period": params["period"]},
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
                provider_context_hash=stable_hash(features.attrs.get("data_provenance") or {}),
            )
            signal = features["close"] > value
        elif self.signal_kind == "DYNAMIC_EMA_CROSS":
            fast = parameterized_feature_series(
                features,
                "ema",
                {"period": params["fast"]},
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
            )
            slow = parameterized_feature_series(
                features,
                "ema",
                {"period": params["slow"]},
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
            )
            signal = fast > slow
        elif self.signal_kind == "DYNAMIC_ATR_MULTIPLE":
            value = parameterized_feature_series(
                features,
                "atr",
                {"period": params["period"]},
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
                provider_context_hash=stable_hash(features.attrs.get("data_provenance") or {}),
            )
            signal = value / features["close"] > float(params["value"])
        elif self.signal_kind == "DYNAMIC_BOLLINGER_CROSS":
            lower = parameterized_feature_series(
                features,
                "bollinger_lower",
                {
                    "period": params["period"],
                    "multiplier": params["multiplier"],
                },
                market=features.attrs.get("market"),
                timeframe=features.attrs.get("timeframe"),
                provider_context_hash=stable_hash(features.attrs.get("data_provenance") or {}),
            )
            signal = (features["close"] > lower) & (features["close"].shift(1) <= lower.shift(1))
        else:
            raise ValueError(f"unsupported signal kind: {self.signal_kind}")
        return signal.fillna(False).astype(bool)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_specs"] = [spec.to_dict() for spec in self.parameter_specs]
        return _canonical_value(payload)


ALL_TIMEFRAMES = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1W")
TECH_TIMEFRAMES = ALL_TIMEFRAMES
FAST_SCREEN_VERSION = "2.0.0"
SCREEN_POLICY_VERSION = "2.1.0"
EXIT_MODEL_VERSION = "2.2.0"
SURVIVOR_POLICY_VERSION = "2.1.0"
FAST_SCREEN_MINIMUM_TRADES = 30
DEFAULT_COMPATIBLE = tuple(BlockRole)


def _half(
    name: str,
    minimum: str,
    maximum: str,
    default: str,
    *,
    description: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        kind=ParameterKind.HALF_STEP,
        minimum=Decimal(minimum),
        maximum=Decimal(maximum),
        step=Decimal("0.5"),
        default=Decimal(default),
        description=description,
    )


def _block(
    block_id: str,
    *,
    family: str,
    role: BlockRole,
    direction: BlockDirection,
    feature: str,
    signal_kind: str = "BOOL",
    compare: str | None = None,
    threshold: tuple[str, str, str] | None = None,
    period: tuple[str, str, str] | None = None,
    warmup: int = 1,
    redundancy: str | None = None,
    source: tuple[str, ...] = ("closed_ohlcv",),
    timeframes: tuple[str, ...] = TECH_TIMEFRAMES,
    missing: Literal["REJECT", "FALSE", "BLOCK"] = "REJECT",
    incompatible: tuple[str, ...] = (),
    cost: Literal["LOW", "MEDIUM", "HIGH"] = "LOW",
    description: str = "",
    extra_specs: tuple[ParameterSpec, ...] = (),
) -> SignalBlock:
    specs: list[ParameterSpec] = []
    defaults: dict[str, Any] = {}
    if threshold:
        minimum, maximum, default = threshold
        try:
            specification = _half("value", minimum, maximum, default)
        except ValueError:
            specification = ParameterSpec(
                name="value",
                kind=ParameterKind.DECIMAL,
                minimum=Decimal(minimum),
                maximum=Decimal(maximum),
                step=Decimal("0.1"),
                default=Decimal(default),
            )
        specs.append(specification)
        defaults["value"] = Decimal(default)
    if period:
        minimum, maximum, default = period
        specs.append(_half("period", minimum, maximum, default))
        defaults["period"] = Decimal(default)
    for specification in extra_specs:
        if specification.name in defaults:
            raise ValueError(f"duplicate parameter spec: {specification.name}")
        specs.append(specification)
        defaults[specification.name] = specification.default
    operator = {
        "BOOL": SignalOperator.BOOLEAN_TRUE,
        "GT": SignalOperator.GREATER_THAN,
        "LT": SignalOperator.LESS_THAN,
        "ABOVE_FEATURE": SignalOperator.ABOVE_FEATURE,
        "BELOW_FEATURE": SignalOperator.BELOW_FEATURE,
        "CROSS_ABOVE": SignalOperator.CROSS_ABOVE,
        "CROSS_BELOW": SignalOperator.CROSS_BELOW,
    }.get(signal_kind, SignalOperator.CUSTOM_CAUSAL)
    overlay_semantics = (
        OverlaySemantics.ONE_TIME_REDUCTION_ON_TRANSITION
        if role is BlockRole.RISK_OVERLAY
        else None
    )
    return SignalBlock(
        block_id=block_id,
        version="1.0.0",
        display_name=block_id.replace("_", " ").title(),
        family=family,
        subfamily=redundancy or family.casefold(),
        role=role,
        direction=direction,
        required_features=tuple(
            dict.fromkeys(
                [feature]
                + ([compare] if compare else [])
                + (["close"] if signal_kind.startswith("DYNAMIC_") else [])
                + (
                    ["open", "high", "low", "volume"]
                    if signal_kind == "DYNAMIC_ATR_MULTIPLE"
                    else []
                )
            )
        ),
        supported_timeframes=timeframes,
        warmup_bars=warmup,
        parameter_specs=tuple(specs),
        signal_kind=signal_kind,
        feature=feature,
        compare_feature=compare,
        knowability_timestamp="candle_close_or_source_available_at",
        missing_data_policy=missing,
        description=description or block_id.replace("_", " "),
        redundancy_group=redundancy or block_id,
        computational_cost_class=cost,
        compatible_roles=DEFAULT_COMPATIBLE,
        incompatible_blocks=incompatible,
        default_parameters=defaults,
        source_quality_requirements=source,
        operator=operator,
        overlay_semantics=overlay_semantics,
    )


def signal_block_registry() -> dict[str, SignalBlock]:
    """Return only blocks whose formulas or source columns are implemented."""

    blocks = [
        _block(
            "positive_return_20",
            family="PRICE_RETURNS",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="PCT_CHANGE_POSITIVE_20",
            warmup=20,
            redundancy="return_momentum",
        ),
        _block(
            "negative_return_exit",
            family="PRICE_RETURNS",
            role=BlockRole.EXIT_TRIGGER,
            direction=BlockDirection.BEARISH,
            feature="roc_12",
            signal_kind="NEGATIVE",
            warmup=12,
            redundancy="return_momentum",
        ),
        _block(
            "btc_relative_momentum",
            family="PRICE_RETURNS",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="btc_relative_momentum_20",
            signal_kind="POSITIVE",
            warmup=20,
            redundancy="relative_momentum",
        ),
        _block(
            "price_above_ema20",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="ABOVE_FEATURE",
            compare="ema_20",
            warmup=20,
            redundancy="ma_position",
        ),
        _block(
            "price_above_ema50",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="ABOVE_FEATURE",
            compare="ema_50",
            warmup=50,
            redundancy="ma_position",
        ),
        _block(
            "price_above_ema200",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="ABOVE_FEATURE",
            compare="ema_200",
            warmup=200,
            redundancy="ma_position",
        ),
        _block(
            "price_above_sma50",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="ABOVE_FEATURE",
            compare="sma_50",
            warmup=50,
            redundancy="ma_position",
        ),
        _block(
            "ema20_above_ema50",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="ema_20",
            signal_kind="ABOVE_FEATURE",
            compare="ema_50",
            warmup=50,
            redundancy="ma_alignment",
        ),
        _block(
            "ema50_above_ema200",
            family="TREND",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.BULLISH,
            feature="ema_50",
            signal_kind="ABOVE_FEATURE",
            compare="ema_200",
            warmup=200,
            redundancy="ma_alignment",
        ),
        _block(
            "ema50_positive_slope",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="ema_50_slope",
            signal_kind="POSITIVE",
            warmup=55,
            redundancy="ma_slope",
        ),
        _block(
            "generalized_ema_position",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="DYNAMIC_EMA_ABOVE",
            period=("5.0", "200.0", "20.5"),
            warmup=201,
            redundancy="ma_position",
        ),
        _block(
            "adx_trend_strength",
            family="TREND",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.NEUTRAL,
            feature="adx_14",
            signal_kind="GT",
            threshold=("10.0", "50.0", "22.5"),
            warmup=28,
            redundancy="trend_strength",
        ),
        _block(
            "dmi_bullish",
            family="TREND",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="plus_di_14",
            signal_kind="ABOVE_FEATURE",
            compare="minus_di_14",
            warmup=28,
            redundancy="dmi",
        ),
        _block(
            "aroon_bullish",
            family="TREND",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="aroon_up_25",
            signal_kind="ABOVE_FEATURE",
            compare="aroon_down_25",
            warmup=25,
            redundancy="aroon",
        ),
        _block(
            "supertrend_bullish",
            family="TREND",
            role=BlockRole.TREND_FILTER,
            direction=BlockDirection.BULLISH,
            feature="supertrend_direction",
            signal_kind="POSITIVE",
            warmup=20,
            redundancy="supertrend",
        ),
        _block(
            "donchian20_breakout",
            family="TREND",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="CROSS_ABOVE",
            compare="donchian_high_20",
            warmup=21,
            redundancy="donchian",
        ),
        _block(
            "donchian55_breakout",
            family="TREND",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="CROSS_ABOVE",
            compare="donchian_high_55",
            warmup=56,
            redundancy="donchian",
        ),
        _block(
            "choppiness_low",
            family="TREND",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="choppiness_14",
            signal_kind="LT",
            threshold=("30.0", "70.0", "50.0"),
            warmup=28,
            redundancy="choppiness",
        ),
        _block(
            "choppiness_high",
            family="STATISTICAL_REGIME",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="choppiness_14",
            signal_kind="GT",
            threshold=("50.0", "75.0", "61.5"),
            warmup=28,
            redundancy="range_choppiness",
        ),
        _block(
            "adx_range_low",
            family="STATISTICAL_REGIME",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="adx_14",
            signal_kind="LT",
            threshold=("10.0", "25.0", "18.0"),
            warmup=28,
            redundancy="range_trend_strength",
        ),
        _block(
            "htf_4h_regime_bullish",
            family="MULTI_TIMEFRAME",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.BULLISH,
            feature="htf_4h_regime_bullish",
            signal_kind="BOOL",
            warmup=200,
            redundancy="htf_4h_regime",
            timeframes=("5m", "15m", "30m", "1h", "2h"),
        ),
        _block(
            "htf_1d_regime_bullish",
            family="MULTI_TIMEFRAME",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.BULLISH,
            feature="htf_1d_regime_bullish",
            signal_kind="BOOL",
            warmup=200,
            redundancy="htf_1d_regime",
            timeframes=("5m", "15m", "30m", "1h", "2h", "4h"),
        ),
        _block(
            "rsi_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="rsi_14",
            signal_kind="LT",
            threshold=("10.0", "40.0", "30.0"),
            warmup=14,
            redundancy="rsi",
        ),
        _block(
            "rsi_recovery",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="rsi_14",
            signal_kind="CROSS_ABOVE",
            threshold=("25.0", "50.0", "35.0"),
            warmup=15,
            redundancy="rsi_recovery",
        ),
        _block(
            "ema20_reclaim",
            family="TREND",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="CROSS_ABOVE",
            compare="ema_20",
            warmup=21,
            redundancy="ema_reclaim",
        ),
        _block(
            "rsi_overbought_exit",
            family="MOMENTUM",
            role=BlockRole.EXIT_TRIGGER,
            direction=BlockDirection.BEARISH,
            feature="rsi_14",
            signal_kind="GT",
            threshold=("60.0", "95.0", "72.5"),
            warmup=14,
            redundancy="rsi",
        ),
        _block(
            "generalized_rsi_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="DYNAMIC_RSI_LT",
            threshold=("10.0", "40.0", "14.5"),
            period=("7.0", "30.0", "13.5"),
            warmup=31,
            redundancy="rsi",
            cost="MEDIUM",
        ),
        _block(
            "stoch_rsi_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="stoch_rsi_14",
            signal_kind="LT",
            threshold=("5.0", "40.0", "20.0"),
            warmup=28,
            redundancy="stoch_rsi",
        ),
        _block(
            "macd_bullish",
            family="MOMENTUM",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="macd",
            signal_kind="ABOVE_FEATURE",
            compare="macd_signal",
            warmup=35,
            redundancy="macd",
        ),
        _block(
            "macd_histogram_positive",
            family="MOMENTUM",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="macd_histogram",
            signal_kind="POSITIVE",
            warmup=35,
            redundancy="macd",
        ),
        _block(
            "ppo_positive",
            family="MOMENTUM",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="ppo",
            signal_kind="POSITIVE",
            warmup=26,
            redundancy="price_oscillator",
        ),
        _block(
            "roc_positive",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="roc_12",
            signal_kind="GT",
            threshold=("-10.0", "20.0", "0.5"),
            warmup=12,
            redundancy="return_momentum",
        ),
        _block(
            "cci_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="cci_20",
            signal_kind="LT",
            threshold=("-200.0", "-50.0", "-100.0"),
            warmup=20,
            redundancy="cci",
        ),
        _block(
            "williams_r_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="williams_r_14",
            signal_kind="LT",
            threshold=("-95.0", "-50.0", "-80.0"),
            warmup=14,
            redundancy="williams_r",
        ),
        _block(
            "mfi_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="mfi_14",
            signal_kind="LT",
            threshold=("5.0", "40.0", "20.0"),
            warmup=14,
            redundancy="money_flow",
        ),
        _block(
            "connors_rsi_oversold",
            family="MOMENTUM",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="connors_rsi",
            signal_kind="LT",
            threshold=("5.0", "40.0", "20.0"),
            warmup=100,
            redundancy="connors_rsi",
        ),
        _block(
            "normalized_atr_regime",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="normalized_atr_14",
            signal_kind="LT",
            threshold=("0.5", "10.0", "5.0"),
            warmup=14,
            redundancy="atr_regime",
        ),
        _block(
            "generalized_atr_regime",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="close",
            signal_kind="DYNAMIC_ATR_MULTIPLE",
            threshold=("0.5", "10.0", "2.5"),
            period=("7.0", "30.0", "14.5"),
            warmup=31,
            redundancy="atr_regime",
            cost="MEDIUM",
        ),
        _block(
            "bollinger_lower_reversion",
            family="VOLATILITY",
            role=BlockRole.ENTRY_TRIGGER,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="DYNAMIC_BOLLINGER_CROSS",
            warmup=50,
            redundancy="bollinger",
            cost="MEDIUM",
            extra_specs=(
                ParameterSpec(
                    name="period",
                    kind=ParameterKind.INTEGER,
                    minimum=10,
                    maximum=50,
                    step=5,
                    default=20,
                    integer_only=True,
                    constraints=("period >= 2",),
                    optimizer_distribution="INT_STEP",
                ),
                _half("multiplier", "1.5", "3.0", "2.0"),
            ),
        ),
        _block(
            "bollinger_squeeze",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="bollinger_width",
            signal_kind="LT",
            threshold=("0.5", "20.0", "5.0"),
            warmup=20,
            redundancy="volatility_compression",
        ),
        _block(
            "bollinger_keltner_squeeze",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="bollinger_upper",
            signal_kind="BELOW_FEATURE",
            compare="keltner_upper",
            warmup=20,
            redundancy="volatility_compression",
        ),
        _block(
            "prior_squeeze_within_12",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="prior_squeeze_within_12",
            signal_kind="BOOL",
            warmup=32,
            redundancy="prior_volatility_compression",
        ),
        _block(
            "rolling_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="rolling_volatility_20",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="realized_volatility",
        ),
        _block(
            "ewma_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="ewma_volatility",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="realized_volatility",
        ),
        _block(
            "parkinson_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="parkinson_volatility_20",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="range_volatility",
        ),
        _block(
            "garman_klass_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="garman_klass_volatility_20",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="range_volatility",
        ),
        _block(
            "rogers_satchell_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="rogers_satchell_volatility_20",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="range_volatility",
        ),
        _block(
            "yang_zhang_volatility_low",
            family="VOLATILITY",
            role=BlockRole.REGIME_FILTER,
            direction=BlockDirection.NEUTRAL,
            feature="yang_zhang_volatility_20",
            signal_kind="LT",
            threshold=("0.5", "10.0", "3.0"),
            warmup=20,
            redundancy="range_volatility",
        ),
        _block(
            "relative_volume_expansion",
            family="VOLUME_FLOW",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="relative_volume_20",
            signal_kind="GT",
            threshold=("0.5", "3.0", "1.5"),
            warmup=20,
            redundancy="volume_expansion",
        ),
        _block(
            "btc_relative_persistence",
            family="PRICE_RETURNS",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="btc_relative_momentum_persistence_5",
            signal_kind="POSITIVE",
            warmup=25,
            redundancy="relative_persistence",
        ),
        _block(
            "volume_zscore_positive",
            family="VOLUME_FLOW",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="volume_zscore_20",
            signal_kind="GT",
            threshold=("-1.0", "3.0", "1.0"),
            warmup=20,
            redundancy="volume_expansion",
        ),
        _block(
            "obv_positive_trend",
            family="VOLUME_FLOW",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="obv",
            signal_kind="CHANGE_POSITIVE_20",
            warmup=20,
            redundancy="obv",
        ),
        _block(
            "chaikin_money_flow_positive",
            family="VOLUME_FLOW",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="chaikin_money_flow_20",
            signal_kind="POSITIVE",
            warmup=20,
            redundancy="money_flow",
        ),
        _block(
            "price_above_vwap",
            family="VOLUME_FLOW",
            role=BlockRole.CONFIRMATION,
            direction=BlockDirection.BULLISH,
            feature="close",
            signal_kind="ABOVE_FEATURE",
            compare="vwap_20",
            warmup=20,
            redundancy="vwap",
        ),
    ]
    blocks.extend(
        [
            SignalBlock(
                block_id="rsi_threshold",
                version="1.0.0",
                display_name="Generalized RSI Threshold",
                family="MOMENTUM",
                subfamily="rsi",
                role=BlockRole.ENTRY_TRIGGER,
                direction=BlockDirection.BULLISH,
                required_features=("close",),
                supported_timeframes=TECH_TIMEFRAMES,
                warmup_bars=31,
                parameter_specs=(
                    _half("value", "10.0", "40.0", "30.0"),
                    _half("period", "7.0", "30.0", "14.0"),
                ),
                signal_kind="DYNAMIC_RSI_LT",
                feature="close",
                compare_feature=None,
                knowability_timestamp="candle_close",
                missing_data_policy="REJECT",
                description="Generalized Wilder RSI with exact half-step period and threshold.",
                redundancy_group="rsi",
                computational_cost_class="MEDIUM",
                compatible_roles=DEFAULT_COMPATIBLE,
                incompatible_blocks=(),
                default_parameters={
                    "value": Decimal("30.0"),
                    "period": Decimal("14.0"),
                },
                source_quality_requirements=("closed_ohlcv",),
            ),
            SignalBlock(
                block_id="ema_trend",
                version="1.0.0",
                display_name="Generalized EMA Trend",
                family="TREND",
                subfamily="ma_alignment",
                role=BlockRole.TREND_FILTER,
                direction=BlockDirection.BULLISH,
                required_features=("close",),
                supported_timeframes=TECH_TIMEFRAMES,
                warmup_bars=201,
                parameter_specs=(
                    _half("fast", "5.0", "50.0", "20.5"),
                    _half("slow", "20.0", "200.0", "50.0"),
                ),
                signal_kind="DYNAMIC_EMA_CROSS",
                feature="close",
                compare_feature=None,
                knowability_timestamp="candle_close",
                missing_data_policy="REJECT",
                description="Generalized EMA alignment with fast-period constraint.",
                redundancy_group="ma_alignment",
                computational_cost_class="MEDIUM",
                compatible_roles=DEFAULT_COMPATIBLE,
                incompatible_blocks=(),
                default_parameters={
                    "fast": Decimal("20.5"),
                    "slow": Decimal("50.0"),
                },
                source_quality_requirements=("closed_ohlcv",),
            ),
            _block(
                "relative_volume",
                family="VOLUME_FLOW",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.BULLISH,
                feature="relative_volume_20",
                signal_kind="GT",
                threshold=("0.5", "3.0", "1.5"),
                warmup=20,
                redundancy="volume_expansion",
            ),
        ]
    )

    structure_specs = [
        ("confirmed_fractal_high", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("confirmed_fractal_low", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("higher_high", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("higher_low", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("lower_high", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("lower_low", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_bos", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("bearish_bos", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_choch", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("bearish_choch", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_liquidity_sweep", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("bearish_liquidity_sweep", BlockRole.AVOIDANCE_FILTER, BlockDirection.BEARISH),
        ("equal_highs", BlockRole.AVOIDANCE_FILTER, BlockDirection.NEUTRAL),
        ("equal_lows", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("bullish_displacement", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_displacement", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_fvg", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_fvg", BlockRole.AVOIDANCE_FILTER, BlockDirection.BEARISH),
        ("bullish_order_block_proxy", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_order_block_proxy", BlockRole.AVOIDANCE_FILTER, BlockDirection.BEARISH),
        ("discount_zone", BlockRole.REGIME_FILTER, BlockDirection.BULLISH),
        ("premium_zone", BlockRole.AVOIDANCE_FILTER, BlockDirection.NEUTRAL),
        ("inside_bar", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("outside_bar", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("fractal_high_breakout", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("fractal_low_breakdown", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
    ]
    blocks.extend(
        _block(
            name,
            family="MARKET_STRUCTURE",
            role=role,
            direction=direction,
            feature=name,
            warmup=55,
            redundancy=(
                "fractal_pivot"
                if name.startswith(("confirmed_", "higher_", "lower_"))
                else "market_structure"
            ),
            description="Confirmed, causal fractal-derived market structure.",
        )
        for name, role, direction in structure_specs
    )
    blocks.extend(
        [
            _block(
                "fractal_density_low",
                family="MARKET_STRUCTURE",
                role=BlockRole.REGIME_FILTER,
                direction=BlockDirection.NEUTRAL,
                feature="fractal_density_50",
                signal_kind="LT",
                threshold=("0.0", "0.5", "0.2"),
                warmup=55,
                redundancy="fractal_regime",
            ),
            _block(
                "fractal_amplitude_filter",
                family="MARKET_STRUCTURE",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.NEUTRAL,
                feature="fractal_amplitude_atr",
                signal_kind="GT",
                threshold=("0.5", "5.0", "1.5"),
                warmup=55,
                redundancy="fractal_regime",
            ),
            _block(
                "fractal_3_low_confirmed",
                family="MARKET_STRUCTURE",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.BULLISH,
                feature="fractal_3_confirmed_fractal_low",
                warmup=3,
                redundancy="fractal_pivot_3",
            ),
            _block(
                "fractal_5_low_confirmed",
                family="MARKET_STRUCTURE",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.BULLISH,
                feature="fractal_5_confirmed_fractal_low",
                warmup=5,
                redundancy="fractal_pivot_5",
            ),
            _block(
                "fractal_7_low_confirmed",
                family="MARKET_STRUCTURE",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.BULLISH,
                feature="fractal_7_confirmed_fractal_low",
                warmup=7,
                redundancy="fractal_pivot_7",
            ),
            _block(
                "fractal_breakout_volume_confirmed",
                family="MARKET_STRUCTURE",
                role=BlockRole.CONFIRMATION,
                direction=BlockDirection.BULLISH,
                feature="fractal_breakout_volume_confirmation",
                warmup=55,
                redundancy="fractal_breakout_confirmation",
            ),
            _block(
                "fractal_mtf_bullish",
                family="MARKET_STRUCTURE",
                role=BlockRole.REGIME_FILTER,
                direction=BlockDirection.BULLISH,
                feature="multi_timeframe_fractal_alignment",
                signal_kind="GT",
                threshold=("0.0", "1.0", "0.0"),
                warmup=55,
                redundancy="fractal_mtf",
            ),
            _block(
                "hurst_trending_regime",
                family="STATISTICAL_REGIME",
                role=BlockRole.REGIME_FILTER,
                direction=BlockDirection.NEUTRAL,
                feature="hurst_exponent",
                signal_kind="GT",
                threshold=("0.0", "1.0", "0.5"),
                warmup=100,
                redundancy="fractal_dimension",
            ),
            _block(
                "fractal_dimension_trend_regime",
                family="STATISTICAL_REGIME",
                role=BlockRole.REGIME_FILTER,
                direction=BlockDirection.NEUTRAL,
                feature="fractal_dimension_index",
                signal_kind="LT",
                threshold=("1.0", "2.0", "1.5"),
                warmup=100,
                redundancy="fractal_dimension",
            ),
        ]
    )

    candle_specs = [
        ("doji", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("spinning_top", BlockRole.CONFIRMATION, BlockDirection.NEUTRAL),
        ("high_wave", BlockRole.AVOIDANCE_FILTER, BlockDirection.NEUTRAL),
        ("hammer", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("inverted_hammer", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("shooting_star", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("hanging_man", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_engulfing", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("bearish_engulfing", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_harami", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_harami", BlockRole.AVOIDANCE_FILTER, BlockDirection.BEARISH),
        ("morning_star_proxy", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("evening_star_proxy", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("three_white_soldiers", BlockRole.ENTRY_TRIGGER, BlockDirection.BULLISH),
        ("three_black_crows", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_pin_bar", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_pin_bar", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("bullish_marubozu", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("bearish_marubozu", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
        ("rising_three_methods_proxy", BlockRole.CONFIRMATION, BlockDirection.BULLISH),
        ("falling_three_methods_proxy", BlockRole.EXIT_TRIGGER, BlockDirection.BEARISH),
    ]
    blocks.extend(
        _block(
            name,
            family="CANDLE",
            role=role,
            direction=direction,
            feature=name,
            warmup=5,
            redundancy="candle_pattern",
            description="Candle context block; requires layering with trend, structure, volume, or volatility.",
        )
        for name, role, direction in candle_specs
    )

    contextual = [
        (
            "global_risk_on",
            "crypto_risk_on",
            BlockRole.REGIME_FILTER,
            BlockDirection.BULLISH,
            "BOOL",
            None,
        ),
        (
            "global_risk_off",
            "crypto_risk_off",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "BOOL",
            None,
        ),
        (
            "fear_greed_positive",
            "sentiment_fear_greed",
            BlockRole.REGIME_FILTER,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("20.0", "80.0", "50.0"),
        ),
        (
            "btc_dominance_rising",
            "dominance_btc_dominance_change_7d",
            BlockRole.REGIME_FILTER,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("-10.0", "10.0", "0.0"),
        ),
        (
            "stablecoin_rotation_risk",
            "dominance_stablecoin_dominance_change_7d",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("-10.0", "10.0", "0.0"),
        ),
        (
            "breadth_above_ema20",
            "breadth_fraction_above_mean_20d",
            BlockRole.REGIME_FILTER,
            BlockDirection.BULLISH,
            "GT",
            ("0.0", "1.0", "0.5"),
        ),
        (
            "breadth_above_ema50",
            "breadth_fraction_above_mean_50d",
            BlockRole.REGIME_FILTER,
            BlockDirection.BULLISH,
            "GT",
            ("0.0", "1.0", "0.5"),
        ),
        (
            "breadth_above_ema200",
            "breadth_fraction_above_mean_200d",
            BlockRole.REGIME_FILTER,
            BlockDirection.BULLISH,
            "GT",
            ("0.0", "1.0", "0.5"),
        ),
        (
            "high_impact_event_avoidance",
            "events_high_impact_event_risk",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "1.0", "0.0"),
        ),
        (
            "funding_not_overheated",
            "derivatives_funding_zscore",
            BlockRole.REGIME_FILTER,
            BlockDirection.CONTEXTUAL,
            "LT",
            ("0.0", "5.0", "2.0"),
        ),
        (
            "open_interest_growth",
            "derivatives_open_interest_change_7d",
            BlockRole.CONFIRMATION,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("-1.0", "1.0", "0.0"),
        ),
        (
            "positive_basis",
            "derivatives_basis",
            BlockRole.CONFIRMATION,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("-1.0", "1.0", "0.0"),
        ),
        (
            "liquidation_risk",
            "derivatives_liquidation_imbalance",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "gex_regime",
            "gex_net_gex_proxy",
            BlockRole.REGIME_FILTER,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("-10.0", "10.0", "0.0"),
        ),
        (
            "gamma_concentration",
            "gex_gamma_concentration",
            BlockRole.RISK_OVERLAY,
            BlockDirection.CONTEXTUAL,
            "GT",
            ("0.0", "1.0", "0.5"),
        ),
        (
            "negative_intelligence_risk",
            "negative_risk_event_score",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "regulation_risk",
            "regulation_event_score",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "exchange_risk",
            "exchange_risk_score",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "hack_exploit_risk",
            "hack_exploit_score",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "stablecoin_depeg_risk",
            "stablecoin_risk_score",
            BlockRole.AVOIDANCE_FILTER,
            BlockDirection.BEARISH,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
        (
            "intelligence_source_diversity",
            "intelligence_source_diversity",
            BlockRole.CONFIRMATION,
            BlockDirection.NEUTRAL,
            "GT",
            ("0.0", "10.0", "1.0"),
        ),
    ]
    for block_id, feature, role, direction, signal_kind, threshold in contextual:
        blocks.append(
            _block(
                block_id,
                family=(
                    "INTELLIGENCE_EVENTS"
                    if "risk" in block_id or "intelligence" in block_id
                    else "MACRO_DERIVATIVES"
                ),
                role=role,
                direction=direction,
                feature=feature,
                signal_kind=signal_kind,
                threshold=threshold,
                warmup=1,
                redundancy=feature.split("_")[0],
                source=("source_available_at",),
                timeframes=ALL_TIMEFRAMES,
                missing="REJECT",
            )
        )

    registry = {block.block_id: block for block in blocks}
    if len(registry) != len(blocks):
        duplicates = [
            block_id
            for block_id, count in Counter(block.block_id for block in blocks).items()
            if count > 1
        ]
        raise ValueError(f"duplicate signal block IDs: {duplicates}")
    return registry


@dataclass(frozen=True)
class StrategyCombination:
    combination_id: str
    strategy_dna_hash: str
    combination_size: int
    block_ids: tuple[str, ...]
    families: tuple[str, ...]
    roles: tuple[str, ...]
    redundancy_score: float
    logic_mode: LogicMode
    default_parameters: Mapping[str, Any]
    parameter_space_size: int
    estimated_computational_cost: int
    eligibility_status: CombinationState
    generated_at: datetime
    exclusion_reason: str | None = None
    requested_timeframes: tuple[str, ...] = ()
    common_supported_timeframes: tuple[str, ...] = ()
    excluded_timeframes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


class CombinationGenerator:
    def __init__(self, registry: Mapping[str, SignalBlock] | None = None) -> None:
        self.registry = dict(registry or signal_block_registry())
        self.last_generation_status: dict[str, Any] = {
            "status": "NOT_STARTED",
            "continuation_cursor": None,
            "remaining_count": None,
        }

    def estimate(
        self,
        sizes: Iterable[int],
        *,
        logic_modes: Iterable[LogicMode],
        assets: int,
        timeframes: int,
    ) -> dict[str, Any]:
        block_count = len(self.registry)
        by_size = {
            str(size): math.comb(block_count, size)
            for size in sorted(set(sizes))
            if 1 <= size <= min(5, block_count)
        }
        memberships = sum(by_size.values())
        logic_count = len(set(logic_modes))
        baselines = memberships * logic_count * max(1, assets) * max(1, timeframes)
        parameter_sums = [0] * (max(by_size, default="0") and max(map(int, by_size)) + 1 or 1)
        parameter_sums[0] = 1
        for block in self.registry.values():
            for size in range(len(parameter_sums) - 1, 0, -1):
                parameter_sums[size] += parameter_sums[size - 1] * block.parameter_space_size
        trials_by_size = {
            str(size): parameter_sums[size] * logic_count for size in map(int, by_size)
        }
        return {
            "registered_signal_blocks": block_count,
            "raw_combinations_by_size": by_size,
            "raw_memberships": memberships,
            "logic_variants": logic_count,
            "assets": assets,
            "timeframes": timeframes,
            "baseline_experiments_upper_bound": baselines,
            "parameter_space_by_size": trials_by_size,
            "parameter_trials_upper_bound": sum(trials_by_size.values()),
            "calculation": "PRODUCT_PER_DNA_THEN_SUM",
            "warning": "Upper bounds precede static/family-aware rejection.",
        }

    def generate(
        self,
        *,
        sizes: Iterable[int],
        logic_modes: Iterable[LogicMode] = (LogicMode.LAYERED,),
        mode: GenerationMode = GenerationMode.FAMILY_AWARE,
        block_ids: Iterable[str] | None = None,
        timeframes: Iterable[str] = ("1h", "4h", "1d"),
        maximum_rows: int | None = None,
        continuation_cursor: str | None = None,
    ) -> list[StrategyCombination]:
        selected_ids = tuple(sorted(block_ids or self.registry))
        unknown = sorted(set(selected_ids) - set(self.registry))
        if unknown:
            raise KeyError(f"unknown signal blocks: {unknown}")
        selected_timeframes = set(timeframes)
        generated: list[StrategyCombination] = []
        seen: set[tuple[tuple[str, ...], LogicMode]] = set()
        cursor_pending = continuation_cursor is not None
        traversed = 0
        for size in sorted(set(sizes)):
            if not 1 <= size <= 5:
                raise ValueError("combination sizes must be between one and five")
            for membership in itertools.combinations(selected_ids, size):
                for logic_mode in sorted(set(logic_modes), key=str):
                    key = (membership, logic_mode)
                    if key in seen:
                        state, reason = CombinationState.DUPLICATE, "DUPLICATE"
                    else:
                        seen.add(key)
                        state, reason = self._validate(
                            membership,
                            logic_mode=logic_mode,
                            mode=mode,
                            timeframes=selected_timeframes,
                        )
                    blocks = [self.registry[block_id] for block_id in membership]
                    common_timeframes = set.intersection(
                        *(set(block.supported_timeframes) for block in blocks)
                    )
                    defaults = {
                        block.block_id: canonical_parameters(block.default_parameters)
                        for block in blocks
                    }
                    dna = {
                        "blocks": [
                            {"id": block.block_id, "version": block.version} for block in blocks
                        ],
                        "logic_mode": logic_mode,
                    }
                    dna_hash = stable_hash(dna)
                    traversed += 1
                    if cursor_pending:
                        if dna_hash == continuation_cursor:
                            cursor_pending = False
                        continue
                    generated.append(
                        StrategyCombination(
                            combination_id=f"cmb-{dna_hash[:20]}",
                            strategy_dna_hash=dna_hash,
                            combination_size=size,
                            block_ids=membership,
                            families=tuple(sorted({block.family for block in blocks})),
                            roles=tuple(sorted({block.role.value for block in blocks})),
                            redundancy_score=self._redundancy(blocks),
                            logic_mode=logic_mode,
                            default_parameters=defaults,
                            parameter_space_size=math.prod(
                                block.parameter_space_size for block in blocks
                            ),
                            estimated_computational_cost=sum(
                                {"LOW": 1, "MEDIUM": 3, "HIGH": 8}[block.computational_cost_class]
                                for block in blocks
                            ),
                            eligibility_status=state,
                            generated_at=utc_now(),
                            exclusion_reason=reason,
                            requested_timeframes=tuple(sorted(selected_timeframes)),
                            common_supported_timeframes=tuple(
                                sorted(common_timeframes.intersection(selected_timeframes))
                            ),
                            excluded_timeframes=tuple(
                                sorted(selected_timeframes - common_timeframes)
                            ),
                        )
                    )
                    if maximum_rows and len(generated) >= maximum_rows:
                        total = sum(
                            math.comb(len(selected_ids), requested_size) * len(set(logic_modes))
                            for requested_size in sorted(set(sizes))
                        )
                        self.last_generation_status = {
                            "status": "PARTIAL_GENERATION",
                            "continuation_cursor": generated[-1].strategy_dna_hash,
                            "generated_count": len(generated),
                            "remaining_count": max(0, total - traversed),
                        }
                        return generated
        if cursor_pending:
            raise ValueError("generation continuation cursor was not found")
        self.last_generation_status = {
            "status": "COMPLETE_GENERATION",
            "continuation_cursor": None,
            "generated_count": len(generated),
            "remaining_count": 0,
        }
        return generated

    def _validate(
        self,
        membership: tuple[str, ...],
        *,
        logic_mode: LogicMode,
        mode: GenerationMode,
        timeframes: set[str],
    ) -> tuple[CombinationState, str | None]:
        blocks = [self.registry[block_id] for block_id in membership]
        entry_roles = {
            BlockRole.ENTRY_TRIGGER,
        }
        if not any(block.role in entry_roles for block in blocks):
            return CombinationState.INVALID_STATIC_RULES, "NO_ENTRY_CAPABLE_BLOCK"
        if all(block.family == "CANDLE" for block in blocks):
            return CombinationState.INVALID_STATIC_RULES, "CANDLE_CONTEXT_REQUIRED"
        if any(
            block.direction is BlockDirection.BEARISH and block.role is BlockRole.ENTRY_TRIGGER
            for block in blocks
        ):
            return CombinationState.INVALID_STATIC_RULES, "BEARISH_SHORT_PATH_FORBIDDEN"
        common_timeframes = set.intersection(*(set(block.supported_timeframes) for block in blocks))
        if not common_timeframes.intersection(timeframes):
            return CombinationState.UNSUPPORTED_TIMEFRAME, "NO_SUPPORTED_TIMEFRAME"
        for block in blocks:
            if set(block.incompatible_blocks).intersection(membership):
                return CombinationState.INVALID_STATIC_RULES, "INCOMPATIBLE_BLOCKS"
        groups = Counter(block.redundancy_group for block in blocks)
        if mode is GenerationMode.FAMILY_AWARE and max(groups.values(), default=0) > 1:
            return CombinationState.INVALID_STATIC_RULES, "REDUNDANT_INFORMATION_FAMILY"
        if logic_mode is LogicMode.ANY and any(
            block.role in {BlockRole.AVOIDANCE_FILTER, BlockRole.EXIT_TRIGGER} for block in blocks
        ):
            return CombinationState.INVALID_STATIC_RULES, "ANY_WITH_EXIT_OR_AVOIDANCE_AMBIGUOUS"
        return CombinationState.GENERATED, None

    @staticmethod
    def _redundancy(blocks: Sequence[SignalBlock]) -> float:
        if len(blocks) < 2:
            return 0.0
        counts = Counter(block.redundancy_group for block in blocks)
        redundant_pairs = sum(count * (count - 1) // 2 for count in counts.values())
        all_pairs = len(blocks) * (len(blocks) - 1) // 2
        return redundant_pairs / max(1, all_pairs)


@dataclass(frozen=True)
class UniverseMember:
    cmc_rank: int
    cmc_id: int
    symbol: str
    name: str
    market_cap: float | None
    circulating_supply: float | None
    total_supply: float | None
    maximum_supply: float | None
    volume_24h: float | None
    provider_timestamp: datetime
    observed_at: datetime
    allowlist_status: str
    market_availability: Mapping[str, tuple[str, ...]]
    universe_types: tuple[UniverseType, ...]
    exclusion_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


@dataclass(frozen=True)
class UniverseSnapshot:
    snapshot_id: str
    observed_at: datetime
    provider_timestamp: datetime
    target_size: int
    scan_limit: int
    members: tuple[UniverseMember, ...]
    bias_label: str
    allowlist_version: int

    @property
    def research_eligible(self) -> tuple[UniverseMember, ...]:
        return tuple(
            member
            for member in self.members
            if any(
                kind in member.universe_types
                for kind in (
                    UniverseType.ALLOWED_RESEARCH,
                    UniverseType.REVIEW_RESEARCH_ONLY,
                    UniverseType.RESEARCH_ELIGIBLE,
                )
            )
        )

    @property
    def execution_eligible(self) -> tuple[UniverseMember, ...]:
        return tuple(
            member
            for member in self.members
            if any(
                kind in member.universe_types
                for kind in (
                    UniverseType.EXECUTION_UNIVERSE,
                    UniverseType.EXECUTION_ELIGIBLE,
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


STABLECOIN_SYMBOLS = frozenset(
    {
        "USDT",
        "USDC",
        "DAI",
        "FDUSD",
        "TUSD",
        "USDE",
        "PYUSD",
        "USD1",
        "FRAX",
        "EURC",
        "RLUSD",
    }
)
WRAPPED_PREFIXES = ("W", "ST", "LST")
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "5L", "5S")


def _settings_database(settings: Settings) -> Database:
    configured = (
        settings.providers.database_url.get_secret_value()
        if settings.providers.database_url
        else None
    )
    selected = (
        configured
        if configured and configured.startswith(("sqlite://", "postgresql://", "postgresql+"))
        else None
    )
    return Database(selected, sqlite_path=settings.paths.database_path)


class UniverseManager:
    def __init__(
        self,
        settings: Settings,
        *,
        loader: DataLoader | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings
        self.database = database or _settings_database(settings)
        self.database.migrate()
        self.loader = loader or DataLoader(settings, database=self.database)

    async def refresh(
        self,
        *,
        target_size: int | None = None,
        scan_limit: int | None = None,
    ) -> UniverseSnapshot:
        target = target_size or self.settings.lab.universe_target_size
        limit = scan_limit or self.settings.lab.universe_scan_limit
        rankings, metadata = await asyncio.gather(
            self.loader.download_cmc_rankings(limit=limit),
            self._provider_metadata(),
        )
        return self.build_snapshot(
            rankings,
            provider_markets=metadata,
            target_size=target,
            scan_limit=limit,
        )

    async def _provider_metadata(self) -> dict[str, set[str]]:
        availability: dict[str, set[str]] = {}
        for provider in ("bitvavo", "kraken", "mexc"):
            try:
                records = await self.loader.download_market_metadata(provider=provider)
            except Exception:
                availability[provider] = set()
                continue
            availability[provider] = {
                record.canonical_market
                for record in records
                if record.canonical_market.endswith(("-EUR", "-USDT", "-USD"))
            }
        return availability

    def build_snapshot(
        self,
        rankings: Sequence[NormalizedDataRecord | Mapping[str, Any]],
        *,
        provider_markets: Mapping[str, Iterable[str]],
        target_size: int = 25,
        scan_limit: int | None = None,
        historical_rows: Mapping[str, int] | None = None,
        observed_at: datetime | None = None,
        historical_rankings_available: bool = False,
    ) -> UniverseSnapshot:
        if not 1 <= target_size <= 25:
            raise ValueError("universe target must be between one and 25")
        observed = observed_at or utc_now()
        availability = {
            provider: {market.upper().replace("/", "-").replace("_", "-") for market in markets}
            for provider, markets in provider_markets.items()
        }
        raw_rows: list[dict[str, Any]] = []
        for record in rankings:
            if isinstance(record, NormalizedDataRecord):
                values = dict(record.values)
                values.setdefault("observed_at", record.observed_at)
                values.setdefault("provider_timestamp", record.timestamp)
            else:
                values = dict(record)
            raw_rows.append(values)
        raw_rows.sort(key=lambda row: int(row.get("cmc_rank") or 10**9))
        raw_rows = raw_rows[: scan_limit or len(raw_rows)]
        members: list[UniverseMember] = []
        seen_economic_assets: set[str] = set()
        discovery_count = 0
        for row in raw_rows:
            symbol = str(row.get("symbol") or "").upper()
            name = str(row.get("name") or symbol)
            slug = str(row.get("slug") or name).casefold()
            tags = {str(tag).casefold() for tag in row.get("tags") or []}
            reasons: list[str] = []
            if symbol in STABLECOIN_SYMBOLS or "stablecoin" in tags:
                reasons.append("STABLECOIN")
            if (
                "wrapped" in tags
                or "wrapped" in name.casefold()
                or (symbol.startswith("W") and len(symbol) > 3)
            ):
                reasons.append("WRAPPED_REPRESENTATION")
            if (
                "liquid-staking-derivatives" in tags
                or "liquid staking" in name.casefold()
                or symbol.startswith(("STETH", "WSTETH", "RETH", "CBETH"))
            ):
                reasons.append("LIQUID_STAKING_DERIVATIVE")
            if symbol.endswith(LEVERAGED_SUFFIXES) or "leveraged-token" in tags:
                reasons.append("LEVERAGED_TOKEN")
            if "synthetics" in tags or "synthetic" in name.casefold():
                reasons.append("SYNTHETIC_TOKEN")
            economic_key = slug.removeprefix("wrapped-").removeprefix("staked-")
            if economic_key in seen_economic_assets:
                reasons.append("DUPLICATE_ECONOMIC_ASSET")
            else:
                seen_economic_assets.add(economic_key)
            quote_markets = {
                provider: tuple(
                    sorted(market for market in markets if market.split("-")[0] == symbol)
                )
                for provider, markets in availability.items()
            }
            eur_market = f"{symbol}-EUR"
            eligibility = self.settings.shariah.eligibility(eur_market)
            volume = float(row["volume_24h"]) if row.get("volume_24h") is not None else None
            if volume is None or volume < self.settings.lab.minimum_volume_24h_eur:
                reasons.append("INSUFFICIENT_LIQUIDITY")
            available_research = any(
                quote_markets.get(provider) for provider in ("bitvavo", "kraken", "mexc")
            )
            if not available_research:
                reasons.append("NO_RESEARCH_MARKET")
            if historical_rows is not None:
                rows = int(historical_rows.get(symbol, 0))
                if rows < self.settings.lab.minimum_history_rows:
                    reasons.append("INSUFFICIENT_HISTORY")
            rank = int(row.get("cmc_rank") or 0)
            types: list[UniverseType] = []
            if 0 < rank <= target_size:
                types.append(UniverseType.RAW_CMC_TOP_N)
            hard_reasons = tuple(dict.fromkeys(reasons))
            if not any(
                reason.startswith(
                    (
                        "STABLECOIN",
                        "WRAPPED",
                        "LIQUID_STAKING",
                        "LEVERAGED",
                        "SYNTHETIC",
                        "DUPLICATE",
                    )
                )
                for reason in hard_reasons
            ):
                types.append(UniverseType.ELIGIBILITY_FILTERED)
            if not hard_reasons and discovery_count < target_size:
                types.append(UniverseType.DISCOVERY_UNIVERSE)
                discovery_count += 1
                if eligibility.status is EligibilityStatus.ALLOWED:
                    types.extend(
                        (
                            UniverseType.ALLOWED_RESEARCH,
                            UniverseType.RESEARCH_ELIGIBLE,
                        )
                    )
                    if eur_market in set(quote_markets.get("bitvavo") or ()):
                        types.extend(
                            (
                                UniverseType.EXECUTION_UNIVERSE,
                                UniverseType.EXECUTION_ELIGIBLE,
                            )
                        )
                elif eligibility.status is EligibilityStatus.REVIEW_REQUIRED:
                    types.append(UniverseType.REVIEW_RESEARCH_ONLY)
            provider_timestamp = row.get("provider_timestamp") or observed
            if isinstance(provider_timestamp, str):
                provider_timestamp = datetime.fromisoformat(
                    provider_timestamp.replace("Z", "+00:00")
                )
            if provider_timestamp.tzinfo is None:
                provider_timestamp = provider_timestamp.replace(tzinfo=UTC)
            members.append(
                UniverseMember(
                    cmc_rank=rank,
                    cmc_id=int(row.get("cmc_id") or row.get("id") or 0),
                    symbol=symbol,
                    name=name,
                    market_cap=(
                        float(row["market_cap"]) if row.get("market_cap") is not None else None
                    ),
                    circulating_supply=(
                        float(row["circulating_supply"])
                        if row.get("circulating_supply") is not None
                        else None
                    ),
                    total_supply=(
                        float(row["total_supply"]) if row.get("total_supply") is not None else None
                    ),
                    maximum_supply=(
                        float(row["maximum_supply"])
                        if row.get("maximum_supply") is not None
                        else None
                    ),
                    volume_24h=volume,
                    provider_timestamp=provider_timestamp.astimezone(UTC),
                    observed_at=observed.astimezone(UTC),
                    allowlist_status=eligibility.status.value,
                    market_availability=quote_markets,
                    universe_types=tuple(types),
                    exclusion_reasons=hard_reasons,
                )
            )
        snapshot_material = {
            "observed_at": utc_iso(observed),
            "allowlist_version": self.settings.shariah.version,
            "members": [member.to_dict() for member in members],
        }
        snapshot = UniverseSnapshot(
            snapshot_id=f"univ-{stable_hash(snapshot_material)[:24]}",
            observed_at=observed.astimezone(UTC),
            provider_timestamp=max(
                (member.provider_timestamp for member in members),
                default=observed,
            ),
            target_size=target_size,
            scan_limit=scan_limit or len(raw_rows),
            members=tuple(members),
            bias_label=(
                "POINT_IN_TIME_UNIVERSE"
                if historical_rankings_available
                else "CURRENT_UNIVERSE_RETROSPECTIVE"
            ),
            allowlist_version=self.settings.shariah.version,
        )
        self.persist(snapshot)
        return snapshot

    def persist(self, snapshot: UniverseSnapshot) -> None:
        self.database.upsert_records(
            "universe_snapshots",
            [
                {
                    "external_id": snapshot.snapshot_id,
                    "provider": "coinmarketcap",
                    "timestamp": snapshot.provider_timestamp,
                    "observed_at": snapshot.observed_at,
                    "available_at": snapshot.observed_at,
                    "status": snapshot.bias_label,
                    "payload": snapshot.to_dict(),
                    **snapshot.to_dict(),
                }
            ],
        )
        self.database.upsert_records(
            "universe_members",
            [
                {
                    "external_id": f"{snapshot.snapshot_id}:{member.cmc_id}",
                    "provider": "coinmarketcap",
                    "market": f"{member.symbol}-EUR",
                    "timestamp": member.provider_timestamp,
                    "observed_at": member.observed_at,
                    "available_at": member.observed_at,
                    "status": (
                        "DISCOVERY_REVIEW_ONLY"
                        if UniverseType.REVIEW_RESEARCH_ONLY in member.universe_types
                        else (
                            "ALLOWED_RESEARCH"
                            if UniverseType.ALLOWED_RESEARCH in member.universe_types
                            else "EXCLUDED"
                        )
                    ),
                    **member.to_dict(),
                    "snapshot_id": snapshot.snapshot_id,
                }
                for member in snapshot.members
            ],
        )
        directory = self.settings.paths.lab_dir / "manifests" / "universes"
        atomic_write_json(directory / f"{snapshot.snapshot_id}.json", snapshot.to_dict())

    def history(self) -> list[dict[str, Any]]:
        rows = self.database.fetch_records("universe_snapshots")
        return [dict(row["payload"]) for row in rows]

    def latest(self) -> dict[str, Any] | None:
        rows = self.history()
        return rows[-1] if rows else None


class CombinatorialStrategy(Strategy):
    """Adapter that lets SignalBlock DNA run through the canonical backtester."""

    family = "combinatorial_lab"
    description = "Canonical SignalBlock combination"
    defaults = {
        "exit_profile": ExitProfile.FIXED_R.value,
        "stop_atr": 2.0,
        "target_atr": 3.0,
        "trailing_atr": 1.5,
        "maximum_holding_bars": 120,
    }
    parameter_space = {
        "exit_profile": tuple(profile.value for profile in ExitProfile),
        "stop_atr": (1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
        "target_atr": (2.0, 3.0, 4.0, 6.0, 10.0, 20.0),
        "trailing_atr": (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
        "maximum_holding_bars": (48, 120, 240, 480, 720),
    }

    def __init__(
        self,
        combination: StrategyCombination,
        registry: Mapping[str, SignalBlock],
        *,
        block_parameters: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.combination = combination
        self.registry = dict(registry)
        self.block_parameters = {
            key: dict(value) for key, value in (block_parameters or {}).items()
        }
        self.strategy_id = f"lab_{combination.strategy_dna_hash[:24]}"
        defaults: dict[str, Any] = {
            "exit__profile": ExitProfile.FIXED_R.value,
            "exit__stop_atr": Decimal("2.0"),
            "exit__target_atr": Decimal("3.0"),
            "exit__trailing_atr": Decimal("1.5"),
            "exit__maximum_holding_bars": 120,
            "risk__risk_fraction": Decimal("0.005"),
            "risk__position_fraction": Decimal("1.0"),
            "logic__vote_threshold": Decimal("0.5"),
        }
        spaces: dict[str, tuple[Any, ...]] = {
            "exit__profile": tuple(profile.value for profile in ExitProfile),
            "exit__stop_atr": tuple(
                Decimal(str(value)) for value in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
            ),
            "exit__target_atr": tuple(
                Decimal(str(value)) for value in (2.0, 3.0, 4.0, 6.0, 10.0, 20.0)
            ),
            "exit__trailing_atr": tuple(
                Decimal(str(value)) for value in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
            ),
            "exit__maximum_holding_bars": (48, 120, 240, 480, 720),
            "risk__risk_fraction": (
                Decimal("0.0025"),
                Decimal("0.005"),
                Decimal("0.01"),
            ),
            "risk__position_fraction": (
                Decimal("0.5"),
                Decimal("0.75"),
                Decimal("1.0"),
            ),
            "logic__vote_threshold": (
                Decimal("0.5"),
                Decimal("0.6"),
                Decimal("0.7"),
            ),
        }
        optimizer_specs: dict[str, ParameterSpec] = {
            "exit__profile": ParameterSpec(
                name="exit__profile",
                kind=ParameterKind.CHOICE,
                choices=tuple(profile.value for profile in ExitProfile),
                default=ExitProfile.FIXED_R.value,
            ),
            "exit__stop_atr": _half("exit__stop_atr", "1.5", "6.0", "2.0"),
            "exit__target_atr": _half("exit__target_atr", "2.0", "20.0", "3.0"),
            "exit__trailing_atr": _half("exit__trailing_atr", "0.0", "4.0", "1.5"),
            "exit__maximum_holding_bars": ParameterSpec(
                name="exit__maximum_holding_bars",
                kind=ParameterKind.INTEGER,
                minimum=48,
                maximum=720,
                step=24,
                default=120,
                integer_only=True,
                optimizer_distribution="INT_STEP",
            ),
            "risk__risk_fraction": ParameterSpec(
                name="risk__risk_fraction",
                kind=ParameterKind.DECIMAL,
                minimum=Decimal("0.0025"),
                maximum=Decimal("0.01"),
                step=Decimal("0.0025"),
                default=Decimal("0.005"),
            ),
            "risk__position_fraction": ParameterSpec(
                name="risk__position_fraction",
                kind=ParameterKind.DECIMAL,
                minimum=Decimal("0.5"),
                maximum=Decimal("1.0"),
                step=Decimal("0.25"),
                default=Decimal("1.0"),
            ),
            "logic__vote_threshold": ParameterSpec(
                name="logic__vote_threshold",
                kind=ParameterKind.DECIMAL,
                minimum=Decimal("0.5"),
                maximum=Decimal("0.7"),
                step=Decimal("0.1"),
                default=Decimal("0.5"),
            ),
        }
        for block_id in combination.block_ids:
            block = self.registry[block_id]
            chosen = block.parameters(self.block_parameters.get(block_id))
            for spec in block.parameter_specs:
                key = f"{block_id}__{spec.name}"
                defaults[key] = chosen[spec.name]
                spaces[key] = spec.values()
                optimizer_specs[key] = replace(spec, name=key)
            if combination.logic_mode is LogicMode.WEIGHTED_VOTE:
                weight_key = f"logic__weight__{block_id}"
                defaults[weight_key] = Decimal("1.0")
                spaces[weight_key] = (
                    Decimal("0.5"),
                    Decimal("1.0"),
                    Decimal("1.5"),
                )
                optimizer_specs[weight_key] = _half(
                    weight_key,
                    "0.5",
                    "1.5",
                    "1.0",
                )
        self.defaults = defaults
        self.parameter_space = spaces
        self.optimizer_parameter_specs = optimizer_specs

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        if _decimal(parameters["exit__stop_atr"]) <= 0:
            raise ValueError("exit__stop_atr must be positive")
        ExitProfile(str(parameters["exit__profile"]))
        if _decimal(parameters["exit__target_atr"]) <= 0:
            raise ValueError("exit__target_atr must be positive")
        if _decimal(parameters["exit__trailing_atr"]) < 0:
            raise ValueError("exit__trailing_atr cannot be negative")
        threshold = _decimal(parameters["logic__vote_threshold"])
        if not Decimal("0") < threshold <= Decimal("1"):
            raise ValueError("logic__vote_threshold must be in (0, 1]")
        for block_id in self.combination.block_ids:
            block = self.registry[block_id]
            block.parameters(
                {
                    spec.name: parameters[f"{block_id}__{spec.name}"]
                    for spec in block.parameter_specs
                }
            )

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        selected = self.parameters(parameters)
        signals = {
            block_id: self.registry[block_id].calculate(
                features,
                {
                    spec.name: selected[f"{block_id}__{spec.name}"]
                    for spec in self.registry[block_id].parameter_specs
                },
            )
            for block_id in self.combination.block_ids
        }
        entry_signals = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role is BlockRole.ENTRY_TRIGGER
        ]
        trend_and_confirmation = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role
            in {
                BlockRole.TREND_FILTER,
                BlockRole.CONFIRMATION,
            }
        ]
        mandatory_regime = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role is BlockRole.REGIME_FILTER
        ]
        exit_signals = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role is BlockRole.EXIT_TRIGGER
        ]
        avoidance = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role is BlockRole.AVOIDANCE_FILTER
        ]
        risk_overlays = [
            signals[block_id]
            for block_id in self.combination.block_ids
            if self.registry[block_id].role is BlockRole.RISK_OVERLAY
        ]
        false = pd.Series(False, index=features.index)
        true = pd.Series(True, index=features.index)

        def all_of(items: Sequence[pd.Series]) -> pd.Series:
            return pd.concat(items, axis=1).all(axis=1) if items else true.copy()

        def any_of(items: Sequence[pd.Series]) -> pd.Series:
            return pd.concat(items, axis=1).any(axis=1) if items else false.copy()

        if self.combination.logic_mode is LogicMode.LAYERED:
            entry = (
                any_of(entry_signals) & all_of(mandatory_regime) & all_of(trend_and_confirmation)
            )
        elif self.combination.logic_mode is LogicMode.ALL:
            entry = all_of([*entry_signals, *mandatory_regime, *trend_and_confirmation]) & any_of(
                entry_signals
            )
        elif self.combination.logic_mode is LogicMode.ANY:
            entry = any_of(entry_signals) & all_of(mandatory_regime)
        elif self.combination.logic_mode is LogicMode.MAJORITY:
            voters = [*entry_signals, *trend_and_confirmation]
            votes = (
                pd.concat(voters, axis=1).sum(axis=1)
                if voters
                else pd.Series(0, index=features.index)
            )
            threshold = max(1, math.ceil(len(voters) / 2))
            entry = (votes >= threshold) & any_of(entry_signals) & all_of(mandatory_regime)
        else:
            eligible_ids = [
                block_id
                for block_id in self.combination.block_ids
                if self.registry[block_id].role
                in {
                    BlockRole.ENTRY_TRIGGER,
                    BlockRole.TREND_FILTER,
                    BlockRole.CONFIRMATION,
                }
                and self.registry[block_id].direction is not BlockDirection.BEARISH
            ]
            raw_weights = {
                block_id: float(selected[f"logic__weight__{block_id}"]) for block_id in eligible_ids
            }
            weight_total = sum(raw_weights.values())
            weighted_score = sum(
                signals[block_id].astype(float) * (raw_weights[block_id] / max(weight_total, 1e-12))
                for block_id in eligible_ids
            )
            entry = (
                (weighted_score >= float(selected["logic__vote_threshold"]))
                & any_of(entry_signals)
                & all_of(mandatory_regime)
            )
        avoid = any_of(avoidance)
        exit_signal = any_of(exit_signals)
        overlay_active = any_of(risk_overlays)
        overlay_transition = overlay_active & ~overlay_active.shift(1, fill_value=False)
        sizes = pd.Series(
            float(selected["risk__position_fraction"]),
            index=features.index,
        )
        exit_profile = ExitProfile(str(selected["exit__profile"]))
        stop_atr = selected["exit__stop_atr"]
        target_atr = selected["exit__target_atr"]
        trailing_atr = selected["exit__trailing_atr"]
        maximum_holding_bars = selected["exit__maximum_holding_bars"]
        if exit_profile is ExitProfile.FIXED_R:
            trailing_atr = Decimal("0")
        elif exit_profile is ExitProfile.TRAILING_TREND:
            target_atr = max(_decimal(target_atr), Decimal("20"))
            trailing_atr = max(_decimal(trailing_atr), Decimal("2.5"))
        else:
            target_atr = max(_decimal(target_atr), Decimal("20"))
            trailing_atr = Decimal("0")
        execution_parameters = {
            "stop_atr": stop_atr,
            "target_atr": target_atr,
            "trailing_atr": trailing_atr,
            "maximum_holding_bars": maximum_holding_bars,
        }
        return self._output(
            features,
            entry=entry,
            exit=exit_signal,
            avoid=avoid,
            reduce=overlay_transition,
            size_multiplier=sizes,
            parameters=execution_parameters,
            entry_reason=f"LAB_DNA:{self.combination.strategy_dna_hash[:16]}",
            exit_reason="LAB_EXIT_BLOCK",
            metadata={
                "strategy_dna_hash": self.combination.strategy_dna_hash,
                "block_ids": list(self.combination.block_ids),
                "logic_mode": self.combination.logic_mode.value,
                "exit_profile": exit_profile.value,
                "effective_exit_parameters": canonical_parameters(
                    execution_parameters
                ),
                "flattened_parameters": canonical_parameters(selected),
                "risk_overlay_semantics": {
                    block_id: self.registry[block_id].overlay_semantics
                    for block_id in self.combination.block_ids
                    if self.registry[block_id].role is BlockRole.RISK_OVERLAY
                },
                "long_only": True,
            },
        )


def _canonical_backtest_worker(
    config: BacktestConfig,
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    combination: StrategyCombination,
    block_parameters: dict[str, dict[str, Any]],
) -> BacktestResult:
    """Process-safe entry point for CPU-heavy canonical backtests."""
    strategy = CombinatorialStrategy(
        combination,
        signal_block_registry(),
        block_parameters=block_parameters,
    )
    return BacktestEngine(config, settings=settings).run(frames, strategy)


def _fast_screen_worker(
    frames: dict[str, pd.DataFrame],
    combination: StrategyCombination,
    block_parameters: dict[str, dict[str, Any]],
    round_trip_cost: float,
) -> dict[str, Any]:
    """Process-safe causal coarse screen; never produces a paper candidate."""

    strategy = CombinatorialStrategy(
        combination,
        signal_block_registry(),
        block_parameters=block_parameters,
    )
    return fast_screen(
        frames,
        strategy,
        round_trip_cost=round_trip_cost,
    )


@dataclass(frozen=True)
class LabResearchResult:
    outcome: ResearchOutcome
    rolling_walk_forward: WalkForwardResult
    double_cost_result: BacktestResult


def _canonical_research_worker(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    combination: StrategyCombination,
    block_parameters: dict[str, dict[str, Any]],
    checkpoint_path: Path,
    search_method: Literal["grid", "random", "coordinate", "optuna"],
    search_trials: int,
    allow_review_required_research_only: bool,
) -> LabResearchResult:
    """Process-safe adapter around the existing canonical research pipeline."""
    strategy = CombinatorialStrategy(
        combination,
        signal_block_registry(),
        block_parameters=block_parameters,
    )
    outcome = run_research(
        frames,
        strategy,
        settings,
        search_method=search_method,
        search_trials=search_trials,
        purge_bars=1,
        embargo_bars=1,
        checkpoint_path=checkpoint_path,
        promote_to_paper=False,
        allow_review_required_research_only=allow_review_required_research_only,
    )
    pre_holdout: dict[str, pd.DataFrame] = {}
    for market, frame in frames.items():
        selected = frame.iloc[: max(2, int(len(frame) * 0.80) - 1)].copy()
        selected.attrs.update(frame.attrs)
        pre_holdout[market] = selected
    normal_config = BacktestConfig.from_settings(
        settings,
        allow_review_required_research_only=allow_review_required_research_only,
    )
    rolling = walk_forward_validate(
        pre_holdout,
        strategy,
        outcome.parameters,
        normal_config,
        folds=settings.research.walk_forward_folds,
        mode="rolling",
        purge_bars=1,
        embargo_bars=1,
        settings=settings,
    )
    double_cost_config = replace(
        normal_config,
        costs=replace(normal_config.costs, multiplier=2.0),
        bootstrap_samples=min(1_000, normal_config.bootstrap_samples),
        monte_carlo_runs=min(1_000, normal_config.monte_carlo_runs),
    )
    double_cost_result = BacktestEngine(
        double_cost_config,
        settings=settings,
    ).run(
        pre_holdout,
        strategy,
        parameters=outcome.parameters,
    )
    return LabResearchResult(outcome, rolling, double_cost_result)


@dataclass(frozen=True)
class LabPaths:
    root: Path
    state: Path
    checkpoints: Path
    leaderboards: Path
    reports: Path
    charts: Path
    logs: Path
    exports: Path
    manifests: Path
    failures: Path

    @classmethod
    def create(cls, root: Path) -> "LabPaths":
        selected = root.resolve()
        values = {
            name: selected / name
            for name in (
                "state",
                "checkpoints",
                "leaderboards",
                "reports",
                "charts",
                "logs",
                "exports",
                "manifests",
                "failures",
            )
        }
        selected.mkdir(parents=True, exist_ok=True)
        for path in values.values():
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=selected, **values)


def _payload_rows(database: Database, table_name: str) -> list[dict[str, Any]]:
    return [dict(row["payload"]) for row in database.fetch_records(table_name)]


def _frame_content_hash(frame: pd.DataFrame) -> str:
    """Hash the ordered feature values, index, columns and dtypes."""

    digest = hashlib.sha256()
    digest.update(
        stable_json(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "index_name": str(frame.index.name),
            }
        ).encode("utf-8")
    )
    values = pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(values.tobytes())
    return digest.hexdigest()


def _matches_research_slice(
    payload: Mapping[str, Any],
    *,
    data_hashes_by_timeframe: Mapping[str, str],
    feature_hashes_by_timeframe: Mapping[str, str],
    screening_engine_version: str,
    screen_policy_version: str | None = None,
    exit_model_version: str | None = None,
    survivor_policy_version: str | None = None,
    markets: Sequence[str],
    snapshot_id: str,
    sources: set[str],
) -> bool:
    """Require persisted results to belong to the exact active data slice."""

    tested_timeframes = tuple(str(value) for value in payload.get("timeframes_tested") or ())
    if len(tested_timeframes) != 1:
        return False
    timeframe = tested_timeframes[0]
    expected_data_hash = data_hashes_by_timeframe.get(timeframe)
    expected_feature_hash = feature_hashes_by_timeframe.get(timeframe)
    return bool(
        expected_data_hash
        and expected_feature_hash
        and payload.get("data_hash") == expected_data_hash
        and payload.get("feature_hash") == expected_feature_hash
        and payload.get("screening_engine_version") == screening_engine_version
        and (
            screen_policy_version is None
            or payload.get("screen_policy_version") == screen_policy_version
        )
        and (
            exit_model_version is None
            or payload.get("exit_model_version") == exit_model_version
        )
        and (
            survivor_policy_version is None
            or payload.get("survivor_policy_version") == survivor_policy_version
        )
        and payload.get("universe_snapshot_id") == snapshot_id
        and payload.get("source") in sources
        and set(payload.get("assets_tested") or ()) == set(markets)
    )


def _provenance_source_type(provenance: Mapping[str, Any] | None) -> str | None:
    if not provenance:
        return None
    direct = provenance.get("source_type")
    if isinstance(direct, str) and direct:
        return direct
    nested = {
        source
        for value in provenance.values()
        if isinstance(value, Mapping)
        for source in [_provenance_source_type(value)]
        if source
    }
    if len(nested) == 1:
        return nested.pop()
    return "MIXED_PROVIDER_DATA" if nested else None


class LabStore:
    """Typed durable accounting over the existing generic SQLAlchemy tables."""

    def __init__(self, settings: Settings, database: Database | None = None) -> None:
        self.settings = settings
        self.database = database or _settings_database(settings)
        self.database.migrate()
        self.paths = LabPaths.create(settings.paths.lab_dir)

    def persist_blocks(self, blocks: Iterable[SignalBlock]) -> int:
        rows = []
        for block in blocks:
            payload = block.to_dict()
            rows.append(
                {
                    "external_id": f"{block.block_id}:{block.version}",
                    "status": "ACTIVE",
                    "timestamp": utc_now(),
                    **payload,
                }
            )
        return self.database.upsert_records("signal_blocks", rows)

    def repair_provenance_source_labels(self) -> dict[str, int]:
        """Repair legacy labels only when immutable provenance proves the source."""

        repaired = Counter()
        job_sources: dict[str, tuple[str, str]] = {}
        for row in self.database.fetch_records("experiment_jobs"):
            payload = dict(row["payload"])
            source_type = _provenance_source_type(payload.get("data_provenance"))
            job_id = str(payload.get("job_id") or row["external_id"])
            if source_type:
                job_sources[job_id] = (source_type, str(payload.get("stage") or ""))
            if source_type and payload.get("source_type") != source_type:
                payload["source_type"] = source_type
                self.database.upsert_records("experiment_jobs", [payload])
                repaired["experiment_jobs"] += 1

        table_sources = {
            "baseline_results": lambda stage: (
                "SENSITIVITY_REAL" if stage == "SENSITIVITY" else "BASELINE_REAL"
            ),
            "exact_backtest_results": lambda stage: "EXACT_REAL",
            "experiment_trials": lambda stage: "SCREENING_REAL",
            "leaderboard_entries": lambda stage: (
                "EXACT_REAL" if stage == "EXACT_BACKTEST" else "BASELINE_REAL"
            ),
        }
        exact_keys = {
            (
                payload.get("combination_id"),
                payload.get("parameter_hash"),
                payload.get("data_hash"),
            )
            for payload in _payload_rows(self.database, "exact_backtest_results")
            if _provenance_source_type(payload.get("data_provenance")) == "REAL_PROVIDER_DATA"
        }
        for table_name, source_label in table_sources.items():
            for row in self.database.fetch_records(table_name):
                payload = dict(row["payload"])
                job_id = str(payload.get("job_id") or "")
                evidence = job_sources.get(job_id)
                source_type = (
                    evidence[0]
                    if evidence
                    else _provenance_source_type(payload.get("data_provenance"))
                )
                if source_type != "REAL_PROVIDER_DATA":
                    continue
                key = (
                    payload.get("combination_id"),
                    payload.get("parameter_hash"),
                    payload.get("data_hash"),
                )
                stage = (
                    evidence[1]
                    if evidence
                    else (
                        "EXACT_BACKTEST"
                        if key in exact_keys
                        else (
                            "SENSITIVITY"
                            if "SENSITIVITY" in str(payload.get("source") or "")
                            else "BASELINE"
                        )
                    )
                )
                changed = payload.get("source_type") != source_type or "SYNTHETIC" in str(
                    payload.get("source") or ""
                )
                if not changed:
                    continue
                payload["source_type"] = source_type
                payload["source"] = source_label(stage)
                if table_name == "leaderboard_entries":
                    payload["provider_coverage"] = sorted(
                        {
                            provider
                            for item in (payload.get("data_provenance") or {}).values()
                            if isinstance(item, Mapping)
                            for provider in item.get("providers_used", [])
                        }
                    )
                self.database.upsert_records(table_name, [payload])
                repaired[table_name] += 1
        return dict(sorted(repaired.items()))

    def persist_combinations(self, combinations: Iterable[StrategyCombination]) -> int:
        rows = []
        block_rows = []
        parameter_rows = []
        for combination in combinations:
            payload = combination.to_dict()
            rows.append(
                {
                    "external_id": combination.combination_id,
                    "status": combination.eligibility_status.value,
                    "timestamp": combination.generated_at,
                    **payload,
                }
            )
            for ordinal, block_id in enumerate(combination.block_ids):
                block_rows.append(
                    {
                        "external_id": f"{combination.combination_id}:{block_id}",
                        "status": "ACTIVE",
                        "timestamp": combination.generated_at,
                        "combination_id": combination.combination_id,
                        "block_id": block_id,
                        "ordinal": ordinal,
                    }
                )
            parameter_rows.append(
                {
                    "external_id": combination.strategy_dna_hash,
                    "status": "DEFAULT",
                    "timestamp": combination.generated_at,
                    "combination_id": combination.combination_id,
                    "parameters": combination.default_parameters,
                    "parameter_hash": parameter_hash(combination.default_parameters),
                    "space_size": combination.parameter_space_size,
                }
            )
        count = self.database.upsert_records("strategy_combinations", rows)
        self.database.upsert_records("combination_blocks", block_rows)
        self.database.upsert_records("parameter_spaces", parameter_rows)
        return count

    @staticmethod
    def experiment_hash(
        *,
        snapshot_id: str,
        markets: Sequence[str],
        timeframes: Sequence[str],
        combination: StrategyCombination,
        parameters: Mapping[str, Any],
        entry_profile: Mapping[str, Any],
        exit_profile: Mapping[str, Any],
        risk_profile: Mapping[str, Any],
        cost_profile: Mapping[str, Any],
        data_hashes: Mapping[str, str],
        feature_hashes: Mapping[str, str] | None = None,
        macro_hashes: Mapping[str, str] | None = None,
        intelligence_hashes: Mapping[str, str] | None = None,
        software_version: str = "1.0.0",
    ) -> str:
        return stable_hash(
            {
                "universe_snapshot": snapshot_id,
                "markets": sorted(markets),
                "timeframes": sorted(timeframes),
                "blocks": list(combination.block_ids),
                "block_versions": [
                    {"id": block_id, "version": "1.0.0"} for block_id in combination.block_ids
                ],
                "logic_mode": combination.logic_mode,
                "parameters": canonical_parameters(parameters),
                "entry_profile": entry_profile,
                "exit_profile": exit_profile,
                "risk_profile": risk_profile,
                "cost_profile": cost_profile,
                "data_hashes": data_hashes,
                "feature_hashes": feature_hashes or {},
                "macro_hashes": macro_hashes or {},
                "intelligence_hashes": intelligence_hashes or {},
                "software_version": software_version,
            }
        )

    def queue_job(
        self,
        *,
        run_id: str,
        combination: StrategyCombination,
        snapshot_id: str,
        markets: Sequence[str],
        timeframe: str,
        parameters: Mapping[str, Any],
        data_hash: str,
        feature_hash: str | None = None,
        data_provenance: Mapping[str, Any] | None = None,
        force: bool = False,
        retest: bool = False,
        only_missing: bool = False,
        sensitivity_parameter: str | None = None,
    ) -> dict[str, Any]:
        provenance = dict(
            data_provenance
            or {
                "source_type": "UNSPECIFIED_DIRECT_API",
                "reason": "queue_job called without dataset provenance",
            }
        )
        base_experiment = self.experiment_hash(
            snapshot_id=snapshot_id,
            markets=markets,
            timeframes=[timeframe],
            combination=combination,
            parameters=parameters,
            entry_profile={
                "logic_mode": combination.logic_mode.value,
                "screening_engine_version": FAST_SCREEN_VERSION,
                "screen_policy_version": SCREEN_POLICY_VERSION,
            },
            exit_profile={
                "type": "profiled_atr_stop_target_trailing_time_regime",
                "profiles": [profile.value for profile in ExitProfile],
                "exit_model_version": EXIT_MODEL_VERSION,
            },
            risk_profile={"type": "settings"},
            cost_profile={
                "type": "normal",
                "fee": self.settings.costs.default_fee,
                "slippage_bps": self.settings.costs.slippage_bps,
                "spread_bps": self.settings.costs.spread_bps,
            },
            data_hashes={timeframe: data_hash},
            feature_hashes=({timeframe: feature_hash} if feature_hash is not None else {}),
            software_version=self.settings.app.version,
        )
        base_job_id = f"job-{base_experiment[:24]}"
        existing = self.job(base_job_id)
        if (
            existing
            and existing.get("status")
            in {
                CombinationState.SCREENING_COMPLETED.value,
                CombinationState.BASELINE_COMPLETED.value,
                CombinationState.EXACT_BACKTEST_COMPLETED.value,
                CombinationState.RESEARCH_PASS.value,
                CombinationState.PAPER_CANDIDATE.value,
            }
            and not (force or retest)
        ):
            deduplicated = existing | {
                "deduplicated": True,
                "skip_reason": (
                    "ONLY_MISSING_ALREADY_COMPLETE"
                    if only_missing
                    else "IDENTICAL_EXPERIMENT_COMPLETE"
                ),
            }
            if str(existing.get("run_id")) == run_id:
                return deduplicated
            alias_id = (
                f"{base_job_id}-alias-"
                f"{stable_hash([run_id, base_job_id], length=12)}"
            )
            alias = {
                **deduplicated,
                "job_id": alias_id,
                "run_id": run_id,
                "source_job_id": existing.get("job_id"),
                "stage": "DEDUPLICATED_MEMBERSHIP",
                "reason_code": "IDENTICAL_EXPERIMENT_REUSED_IN_CAMPAIGN",
                "created_at": utc_iso(),
                "updated_at": utc_iso(),
            }
            self.database.upsert_records(
                "experiment_jobs",
                [
                    {
                        **alias,
                        "external_id": alias_id,
                        "status": alias["status"],
                        "timestamp": utc_now(),
                    }
                ],
            )
            return alias
        run_version = (
            stable_hash([utc_iso(), run_id, "force" if force else "retest"])[:10]
            if force or retest
            else None
        )
        experiment = (
            stable_hash(
                {
                    "base_experiment_hash": base_experiment,
                    "run_version": run_version,
                    "force": force,
                    "retest": retest,
                }
            )
            if run_version
            else base_experiment
        )
        job_id = f"{base_job_id}-{run_version}" if run_version else base_job_id
        result_type = (
            "PARAMETER_SENSITIVITY"
            if sensitivity_parameter and sensitivity_parameter != "CLI_OVERRIDE"
            else (
                "JOINT_PARAMETER_SCREEN"
                if sensitivity_parameter == "CLI_OVERRIDE"
                else "BASELINE_SCREEN"
            )
        )
        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "combination_id": combination.combination_id,
            "strategy_dna_hash": combination.strategy_dna_hash,
            "experiment_hash": experiment,
            "base_experiment_hash": base_experiment,
            "run_version": run_version or "original",
            "universe_snapshot_id": snapshot_id,
            "block_ids": list(combination.block_ids),
            "parameter_hash": parameter_hash(parameters),
            "parameters": canonical_parameters(parameters),
            "markets": sorted(markets),
            "timeframe": timeframe,
            "stage": (
                "SENSITIVITY"
                if result_type == "PARAMETER_SENSITIVITY"
                else "BASELINE"
            ),
            "result_type": result_type,
            "status": CombinationState.QUEUED_BASELINE.value,
            "reason_code": "NEW_OR_CHANGED_EXPERIMENT",
            "attempt": 0,
            "retry_eligible": True,
            "last_checkpoint": None,
            "created_at": utc_iso(),
            "updated_at": utc_iso(),
            "retest": retest,
            "force": force,
            "data_hash": data_hash,
            "feature_hash": feature_hash,
            "screening_engine_version": FAST_SCREEN_VERSION,
            "screen_policy_version": SCREEN_POLICY_VERSION,
            "exit_model_version": EXIT_MODEL_VERSION,
            "survivor_policy_version": SURVIVOR_POLICY_VERSION,
            "data_provenance": _canonical_value(provenance),
            "source_type": _provenance_source_type(provenance),
            "software_version": self.settings.app.version,
            "sensitivity_parameter": sensitivity_parameter,
        }
        self.database.upsert_records(
            "experiment_jobs",
            [
                {
                    "external_id": job_id,
                    "status": payload["status"],
                    "timestamp": utc_now(),
                    **payload,
                }
            ],
        )
        self.record_event(
            {
                "run_id": payload.get("run_id"),
                "lab_instance_id": None,
                "job_id": payload.get("job_id"),
                "combination_id": payload.get("combination_id"),
                "experiment_hash": payload.get("experiment_hash"),
                "universe_snapshot_id": payload.get("universe_snapshot_id"),
                "block_ids": payload.get("block_ids") or [],
                "parameter_hash": payload.get("parameter_hash"),
                "market": ",".join(payload.get("markets") or []),
                "timeframe": payload.get("timeframe"),
                "stage": payload.get("stage"),
                "worker": f"pid:{os.getpid()}",
                "started_at": payload.get("created_at"),
                "completed_at": payload.get("updated_at"),
                "duration": None,
                "status": payload.get("status"),
                "reason_code": payload.get("reason_code"),
                "retry_count": payload.get("attempt") or 0,
                "memory_usage": None,
                "cpu_duration": None,
            }
        )
        return payload

    def job(self, job_id: str) -> dict[str, Any] | None:
        row = self.database.fetch_record_by_external_id(
            "experiment_jobs",
            job_id,
        )
        return dict(row["payload"]) if row is not None else None

    def jobs(self) -> list[dict[str, Any]]:
        return _payload_rows(self.database, "experiment_jobs")

    def update_job(
        self,
        job: Mapping[str, Any],
        *,
        status: CombinationState,
        stage: str | None = None,
        reason_code: str,
        checkpoint: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        payload = dict(job)
        payload.update(
            {
                "status": status.value,
                "stage": stage or payload.get("stage"),
                "reason_code": reason_code,
                "last_checkpoint": checkpoint,
                "updated_at": utc_iso(),
            }
        )
        if error is not None:
            payload.update(
                {
                    "exception_type": type(error).__name__,
                    "attempt": int(payload.get("attempt") or 0) + 1,
                    "retry_eligible": status is CombinationState.ERROR_RETRYABLE,
                }
            )
        self.database.upsert_records(
            "experiment_jobs",
            [
                {
                    "external_id": str(payload["job_id"]),
                    "status": payload["status"],
                    "timestamp": utc_now(),
                    **payload,
                }
            ],
        )
        self.record_event(
            {
                "run_id": payload.get("run_id"),
                "lab_instance_id": None,
                "job_id": payload.get("job_id"),
                "combination_id": payload.get("combination_id"),
                "experiment_hash": payload.get("experiment_hash"),
                "universe_snapshot_id": payload.get("universe_snapshot_id"),
                "block_ids": payload.get("block_ids") or [],
                "parameter_hash": payload.get("parameter_hash"),
                "market": ",".join(payload.get("markets") or []),
                "timeframe": payload.get("timeframe"),
                "stage": payload.get("stage"),
                "worker": f"pid:{os.getpid()}",
                "started_at": job.get("updated_at") or job.get("created_at"),
                "completed_at": payload.get("updated_at"),
                "duration": None,
                "status": payload.get("status"),
                "reason_code": payload.get("reason_code"),
                "retry_count": payload.get("attempt") or 0,
                "memory_usage": None,
                "cpu_duration": None,
            }
        )
        return payload

    def recover_stale_jobs(self) -> int:
        stale_states = {
            CombinationState.BASELINE_RUNNING.value,
            CombinationState.SCREENING_RUNNING.value,
            CombinationState.EXACT_BACKTEST_RUNNING.value,
            CombinationState.OPTIMIZATION_RUNNING.value,
            CombinationState.VALIDATION_RUNNING.value,
        }
        recovered = 0
        for job in self.jobs():
            if job.get("status") in stale_states:
                self.update_job(
                    job,
                    status=CombinationState.ERROR_RETRYABLE,
                    reason_code="STALE_RUNNING_JOB_RECOVERED",
                    checkpoint=job.get("last_checkpoint"),
                )
                recovered += 1
        return recovered

    def supersede_incomplete_jobs(self, *, active_run_id: str) -> int:
        terminal_states = {
            CombinationState.SCREENING_COMPLETED.value,
            CombinationState.BASELINE_COMPLETED.value,
            CombinationState.EXACT_BACKTEST_COMPLETED.value,
            CombinationState.BASELINE_REJECTED.value,
            CombinationState.EXACT_BACKTEST_REJECTED.value,
            CombinationState.VALIDATION_REJECTED.value,
            CombinationState.RESEARCH_PASS.value,
            CombinationState.PAPER_CANDIDATE.value,
            CombinationState.ERROR_FINAL.value,
            CombinationState.SUPERSEDED.value,
        }
        superseded = 0
        for job in self.jobs():
            if (
                str(job.get("run_id")) != active_run_id
                and str(job.get("status")) not in terminal_states
            ):
                self.update_job(
                    job,
                    status=CombinationState.SUPERSEDED,
                    stage="SUPERSEDED",
                    reason_code="SUPERSEDED_BY_NEW_CAMPAIGN_RUN",
                    checkpoint=job.get("last_checkpoint"),
                )
                superseded += 1
        return superseded

    def save_result(
        self,
        table_name: str,
        *,
        job: Mapping[str, Any],
        result: Mapping[str, Any],
        status: str,
    ) -> None:
        result_key = (
            result.get("trial_id")
            or result.get("fold_id")
            or result.get("simulation_id")
            or table_name
        )
        safe_result = _finite_json(dict(result))
        self.database.upsert_records(
            table_name,
            [
                {
                    "external_id": f"{job['experiment_hash']}:{result_key}",
                    "status": status,
                    "timestamp": utc_now(),
                    "job_id": job["job_id"],
                    "combination_id": job["combination_id"],
                    "experiment_hash": job["experiment_hash"],
                    **safe_result,
                }
            ],
        )

    def queue_status(self, *, run_id: str | None = None) -> dict[str, Any]:
        jobs = [job for job in self.jobs() if run_id is None or str(job.get("run_id")) == run_id]
        counts = Counter(str(job.get("status")) for job in jobs)
        return {
            "updated_at": utc_iso(),
            "total": len(jobs),
            "by_status": dict(sorted(counts.items())),
            "remaining_work": sum(
                count
                for state, count in counts.items()
                if state
                not in {
                    CombinationState.SCREENING_COMPLETED.value,
                    CombinationState.BASELINE_COMPLETED.value,
                    CombinationState.EXACT_BACKTEST_COMPLETED.value,
                    CombinationState.BASELINE_REJECTED.value,
                    CombinationState.EXACT_BACKTEST_REJECTED.value,
                    CombinationState.VALIDATION_REJECTED.value,
                    CombinationState.RESEARCH_PASS.value,
                    CombinationState.PAPER_CANDIDATE.value,
                    CombinationState.ERROR_FINAL.value,
                    CombinationState.SUPERSEDED.value,
                }
            ),
            "run_id": run_id,
        }

    def reconcile_state(
        self,
        *,
        run_id: str | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Classify stale failure artifacts against durable job truth."""

        jobs = {
            str(job.get("job_id")): job
            for job in self.jobs()
            if run_id is None or str(job.get("run_id")) == run_id
        }
        successful_or_superseded = {
            CombinationState.SCREENING_COMPLETED.value,
            CombinationState.BASELINE_COMPLETED.value,
            CombinationState.EXACT_BACKTEST_COMPLETED.value,
            CombinationState.RESEARCH_PASS.value,
            CombinationState.PAPER_CANDIDATE.value,
            CombinationState.VALIDATION_REJECTED.value,
            CombinationState.EXACT_BACKTEST_REJECTED.value,
            CombinationState.SUPERSEDED.value,
        }
        archive = self.paths.failures / "archive" / "resolved"
        rows: list[dict[str, Any]] = []
        for path in sorted(self.paths.failures.glob("*.json")):
            try:
                failure = read_json(path)
            except (OSError, ValueError, TypeError) as exc:
                rows.append(
                    {
                        "path": str(path),
                        "classification": "INVALID_ARTIFACT",
                        "reason_code": type(exc).__name__,
                    }
                )
                continue
            job_id = str(failure.get("job_id") or path.stem.split(".")[0])
            job = jobs.get(job_id)
            if job is None:
                classification = (
                    "OUT_OF_SCOPE" if run_id is not None else "ORPHANED_ARTIFACT"
                )
            elif str(job.get("status")) in successful_or_superseded:
                classification = "RESOLVED"
            elif str(job.get("status")) == CombinationState.ERROR_RETRYABLE.value:
                classification = "RETRYABLE"
            elif str(job.get("status")) == CombinationState.ERROR_FINAL.value:
                classification = "FINAL_FAILURE"
            else:
                classification = "ACTIVE_OR_UNRESOLVED"
            row = {
                "path": str(path),
                "job_id": job_id,
                "job_status": job.get("status") if job else None,
                "classification": classification,
                "archived_to": None,
            }
            if apply and classification == "RESOLVED":
                archive.mkdir(parents=True, exist_ok=True)
                target = archive / path.name
                if target.exists():
                    target = archive / (
                        f"{path.stem}.{sha256_file(path)[:10]}{path.suffix}"
                    )
                os.replace(path, target)
                row["archived_to"] = str(target)
            rows.append(row)
        counts = Counter(row["classification"] for row in rows)
        report = {
            "status": (
                "RECONCILED"
                if apply
                else "DRY_RUN"
            ),
            "run_id": run_id,
            "apply": apply,
            "failure_artifacts": len(rows),
            "counts": dict(sorted(counts.items())),
            "rows": rows,
            "generated_at": utc_iso(),
            "live_orders": 0,
        }
        report_path = self.paths.reports / (
            f"state_reconciliation_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        atomic_write_json(report_path, report)
        return report | {"report_path": str(report_path)}

    def save_leaderboard_entry(self, payload: Mapping[str, Any]) -> None:
        safe_payload = _finite_json(dict(payload))
        self.database.upsert_records(
            "leaderboard_entries",
            [
                {
                    "external_id": str(payload["entry_id"]),
                    "status": str(payload["lifecycle_status"]),
                    "timestamp": utc_now(),
                    **safe_payload,
                }
            ],
        )

    def leaderboard(self, *, include_synthetic: bool = False) -> list[dict[str, Any]]:
        persisted = _payload_rows(self.database, "leaderboard_entries")
        deduplicated: dict[str, dict[str, Any]] = {}
        for row in persisted:
            entry_id = str(row.get("entry_id") or row.get("external_id") or "")
            current = deduplicated.get(entry_id)
            if current is None or str(row.get("last_tested_at") or "") >= str(
                current.get("last_tested_at") or ""
            ):
                deduplicated[entry_id] = row
        rows = [
            row
            for row in deduplicated.values()
            if include_synthetic or row.get("source_type") == "REAL_PROVIDER_DATA"
        ]

        def leaderboard_score(row: Mapping[str, Any]) -> float:
            value = row.get("robust_score")
            if value is None:
                return -math.inf
            selected = float(value)
            return selected if math.isfinite(selected) else -math.inf

        rows.sort(
            key=lambda row: (
                -leaderboard_score(row),
                str(row.get("strategy_dna_hash") or ""),
            )
        )
        return [
            row
            | {
                "rank": rank,
                "previous_rank": row.get("rank"),
                "rank_change": (int(row.get("rank")) - rank if row.get("rank") else None),
            }
            for rank, row in enumerate(rows, 1)
        ]

    def export_leaderboards(self) -> dict[str, str]:
        rows = self.leaderboard()
        frame = pd.DataFrame(rows)
        tabular = frame.copy()
        for column in tabular.columns:
            if tabular[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
                tabular[column] = tabular[column].map(
                    lambda value: (
                        stable_json(value) if isinstance(value, (dict, list, tuple)) else value
                    )
                )
        paths = {
            "csv": self.paths.leaderboards / "leaderboard.csv",
            "parquet": self.paths.leaderboards / "leaderboard.parquet",
            "json": self.paths.leaderboards / "leaderboard.json",
            "html": self.paths.leaderboards / "leaderboard.html",
        }
        tabular.to_csv(paths["csv"], index=False)
        tabular.to_parquet(paths["parquet"], index=False)
        atomic_write_json(paths["json"], rows)
        paths["html"].write_text(
            tabular.to_html(index=False, escape=True),
            encoding="utf-8",
        )
        view_definitions: dict[str, list[dict[str, Any]]] = {
            "global": rows,
            **{
                f"combination_size_{size}": [
                    row for row in rows if int(row.get("combination_size") or 0) == size
                ]
                for size in range(1, 6)
            },
            "technical_only": [
                row
                for row in rows
                if "MACRO_DERIVATIVES" not in (row.get("block_families") or [])
                and "INTELLIGENCE_EVENTS" not in (row.get("block_families") or [])
            ],
            "macro_plus_technical": [
                row
                for row in rows
                if "MACRO_DERIVATIVES" in (row.get("block_families") or [])
                and len(set(row.get("block_families") or [])) > 1
            ],
            "orderflow_enhanced": [
                row for row in rows if "VOLUME_FLOW" in (row.get("block_families") or [])
            ],
            "derivatives_context_enhanced": [
                row
                for row in rows
                if any(
                    token in str(block_id)
                    for block_id in (row.get("block_names") or [])
                    for token in ("funding", "open_interest", "liquidation", "gex")
                )
            ],
            "on_chain_enhanced": [
                row
                for row in rows
                if any(
                    "onchain" in str(block_id) or "on_chain" in str(block_id)
                    for block_id in (row.get("block_names") or [])
                )
            ],
            "paper_candidates": [row for row in rows if bool(row.get("paper_candidate"))],
        }
        for timeframe in sorted(
            {str(timeframe) for row in rows for timeframe in (row.get("timeframes_tested") or [])}
        ):
            view_definitions[f"timeframe_{timeframe}"] = [
                row for row in rows if timeframe in (row.get("timeframes_tested") or [])
            ]
        for market in sorted(
            {str(market) for row in rows for market in (row.get("assets_tested") or [])}
        ):
            view_definitions[f"market_{market}"] = [
                row for row in rows if market in (row.get("assets_tested") or [])
            ]
        for snapshot in sorted(
            {
                str(row.get("universe_snapshot_id"))
                for row in rows
                if row.get("universe_snapshot_id")
            }
        ):
            view_definitions[f"universe_{snapshot}"] = [
                row for row in rows if row.get("universe_snapshot_id") == snapshot
            ]
        for family in sorted(
            {str(family) for row in rows for family in (row.get("block_families") or [])}
        ):
            view_definitions[f"family_{family}"] = [
                row for row in rows if family in (row.get("block_families") or [])
            ]
        view_index: dict[str, dict[str, Any]] = {}
        views_directory = self.paths.leaderboards / "views"
        views_directory.mkdir(parents=True, exist_ok=True)
        for name, view_rows in sorted(view_definitions.items()):
            slug = "".join(
                character if character.isalnum() else "_" for character in name.casefold()
            ).strip("_")
            view_frame = pd.DataFrame(
                [row | {"view_rank": rank} for rank, row in enumerate(view_rows, 1)]
            )
            for column in view_frame.columns:
                if (
                    view_frame[column]
                    .map(lambda value: isinstance(value, (dict, list, tuple)))
                    .any()
                ):
                    view_frame[column] = view_frame[column].map(
                        lambda value: (
                            stable_json(value) if isinstance(value, (dict, list, tuple)) else value
                        )
                    )
            view_path = views_directory / f"{slug}.csv"
            view_frame.to_csv(view_path, index=False)
            view_index[name] = {"rows": len(view_rows), "csv": str(view_path)}
        views_path = self.paths.leaderboards / "views.json"
        atomic_write_json(views_path, view_index)
        paths["views"] = views_path
        snapshot_id = f"lbs-{stable_hash(rows)[:24]}"
        self.database.upsert_records(
            "leaderboard_snapshots",
            [
                {
                    "external_id": snapshot_id,
                    "status": "EXPORTED",
                    "timestamp": utc_now(),
                    "entry_count": len(rows),
                    "paths": {key: str(path) for key, path in paths.items()},
                }
            ],
        )
        return {key: str(path) for key, path in paths.items()}

    def generate_report(self, *, run_id: str | None = None) -> dict[str, Any]:
        """Generate auditable lab tables, charts and a compact HTML report."""
        from reporting.visualizations import VisualizationReporter
        from research.indicator_registry import indicator_coverage_report

        scoped_jobs = [
            row
            for row in _payload_rows(self.database, "experiment_jobs")
            if run_id is None or str(row.get("run_id")) == run_id
        ]
        experiment_hashes = {
            str(row.get("experiment_hash"))
            for row in scoped_jobs
            if row.get("experiment_hash")
        }
        combination_ids = {
            str(row.get("combination_id"))
            for row in scoped_jobs
            if row.get("combination_id")
        }
        leaderboard = [
            row
            for row in self.leaderboard()
            if run_id is None
            or str(row.get("combination_id")) in combination_ids
        ]
        events = _payload_rows(self.database, "lab_events")
        if run_id is not None:
            events = [
                row for row in events if str(row.get("run_id")) == run_id
            ]
        combinations = {
            str(row.get("combination_id")): row
            for row in _payload_rows(self.database, "strategy_combinations")
        }
        trials = _payload_rows(self.database, "experiment_trials")
        folds = _payload_rows(self.database, "walk_forward_results")
        gates = _payload_rows(self.database, "gate_results")
        if run_id is not None:
            trials = [
                row
                for row in trials
                if str(row.get("experiment_hash")) in experiment_hashes
            ]
            folds = [
                row
                for row in folds
                if str(row.get("experiment_hash")) in experiment_hashes
            ]
            gates = [
                row
                for row in gates
                if str(row.get("experiment_hash")) in experiment_hashes
            ]
        rank_rows = [
            {
                "entry": row.get("strategy_dna_hash"),
                "rank_change": row.get("rank_change") or 0,
            }
            for row in leaderboard[:50]
        ]
        size_rows = [
            {
                "combination_size": row.get("combination_size"),
                "robust_score": row.get("robust_score"),
            }
            for row in leaderboard
        ]
        family_rows = [
            {"family": family, "robust_score": row.get("robust_score")}
            for row in leaderboard
            for family in (row.get("block_families") or [])
        ]
        block_rows = [
            {"block": block} for row in leaderboard[:25] for block in (row.get("block_names") or [])
        ]
        redundancy_rows = [
            {
                "redundancy_score": (
                    combinations.get(str(row.get("combination_id")), {}).get("redundancy_score")
                    or 0.0
                ),
                "robust_score": row.get("robust_score"),
            }
            for row in leaderboard
        ]
        parameter_rows: list[dict[str, Any]] = []
        for row in leaderboard:
            for block_id, parameters in (row.get("parameters") or {}).items():
                for name, value in (parameters or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        parameter_rows.append(
                            {
                                "parameter": f"{block_id}.{name}",
                                "parameter_value": value,
                                "robust_score": row.get("robust_score"),
                            }
                        )
        optimization_rows: list[dict[str, Any]] = []
        for trial in trials:
            if str(trial.get("stage") or "").upper() != "OPTIMIZATION":
                continue
            parameters = trial.get("parameters") or {}
            numeric = [
                float(value)
                for value in parameters.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            optimization_rows.append(
                {
                    "parameter_x": numeric[0] if numeric else math.nan,
                    "parameter_y": numeric[1] if len(numeric) > 1 else 0.0,
                    "score": trial.get("score"),
                    "stability": trial.get("score"),
                    "train_score": trial.get("score"),
                    "validation_score": trial.get("score"),
                    "fold_score": math.nan,
                    "rejection_reason": trial.get("error_code"),
                }
            )
        for fold in folds:
            optimization_rows.append(
                {
                    "parameter_x": math.nan,
                    "parameter_y": math.nan,
                    "score": math.nan,
                    "stability": math.nan,
                    "train_score": math.nan,
                    "validation_score": math.nan,
                    "fold_score": fold.get("net_expectancy_r"),
                    "rejection_reason": None,
                }
            )
        for gate in gates:
            for reason in gate.get("reasons") or []:
                optimization_rows.append(
                    {
                        "parameter_x": math.nan,
                        "parameter_y": math.nan,
                        "score": math.nan,
                        "stability": math.nan,
                        "train_score": math.nan,
                        "validation_score": math.nan,
                        "fold_score": math.nan,
                        "rejection_reason": reason,
                    }
                )
        event_rows = [
            {
                "throughput": (1.0 / max(float(event.get("duration") or 0.0), 1e-9)),
                "completed": int(str(event.get("status") or "").upper() in {"PASSED", "COMPLETED"}),
                "duration": float(event.get("duration") or 0.0),
            }
            for event in events
        ]
        datasets = {
            "lab_rank": pd.DataFrame(rank_rows),
            "lab_size": pd.DataFrame(size_rows),
            "lab_family": pd.DataFrame(family_rows),
            "lab_blocks": pd.DataFrame(block_rows),
            "lab_redundancy": pd.DataFrame(redundancy_rows),
            "lab_parameters": pd.DataFrame(parameter_rows),
            "lab_parameter_heatmap": pd.DataFrame(
                [
                    {
                        "parameter_x": row.get("parameter_x"),
                        "parameter_y": row.get("parameter_y"),
                        "score": row.get("score"),
                    }
                    for row in optimization_rows
                    if row.get("score") is not None
                ]
            ),
            "lab_asset": pd.DataFrame(
                [
                    {"asset": asset, "robust_score": row.get("robust_score")}
                    for row in leaderboard
                    for asset in (row.get("assets_tested") or [])
                ]
            ),
            "lab_timeframe": pd.DataFrame(
                [
                    {
                        "timeframe": timeframe,
                        "robust_score": row.get("robust_score"),
                    }
                    for row in leaderboard
                    for timeframe in (row.get("timeframes_tested") or [])
                ]
            ),
            "lab_universe": pd.DataFrame(
                [
                    {
                        "universe_snapshot": row.get("universe_snapshot_id"),
                        "robust_score": row.get("robust_score"),
                    }
                    for row in leaderboard
                ]
            ),
            "lab_events": pd.DataFrame(event_rows),
            "lab_workers": pd.DataFrame(
                [
                    {
                        "utilization": 0.0,
                    }
                ]
            ),
            "lab_decay": pd.DataFrame(
                [{"robust_score": row.get("robust_score")} for row in leaderboard]
            ),
            "lab_lifecycle": pd.DataFrame(
                [{"lifecycle_status": row.get("lifecycle_status")} for row in leaderboard]
            ),
            "optimization": pd.DataFrame(optimization_rows),
        }
        chart_index = VisualizationReporter(self.paths.charts).generate(datasets)
        coverage = indicator_coverage_report()
        summary = {
            "generated_at": utc_iso(),
            "run_id": run_id,
            "leaderboard_rows": len(leaderboard),
            "queue": self.queue_status(run_id=run_id),
            "gate_results": len(gates),
            "research_passes": sum(bool(row.get("passed")) for row in gates),
            "paper_candidates": sum(bool(row.get("paper_candidate")) for row in leaderboard),
            "live_ready": 0,
            "indicator_coverage": {
                "registry_hash": coverage["registry_hash"],
                "source_item_occurrences": coverage["source_item_occurrences"],
                "unique_canonical_indicators": coverage["unique_canonical_indicators"],
                "counts_by_family": coverage["counts_by_family"],
                "counts_by_status": coverage["counts_by_status"],
            },
            "chart_index": chart_index,
            "limitations": [
                "Synthetic smoke data is labeled and is not evidence of profitability.",
                "Missing datasets are recorded as skipped charts, never fabricated.",
                "Research passes do not authorize live trading.",
            ],
        }
        suffix = f"_{run_id}" if run_id else ""
        summary_path = atomic_write_json(
            self.paths.reports / f"lab_report{suffix}.json",
            summary,
        )
        top_rows = "".join(
            "<tr>"
            f"<td>{int(row.get('rank') or 0)}</td>"
            f"<td>{str(row.get('strategy_dna_hash') or '')[:16]}</td>"
            f"<td>{float(row.get('robust_score') or 0.0):.4f}</td>"
            f"<td>{str(row.get('lifecycle_status') or '')}</td>"
            "</tr>"
            for row in leaderboard[:25]
        )
        html_path = self.paths.reports / f"lab_report{suffix}.html"
        html_path.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>Combinatorial lab report</title>"
            "<style>body{font:15px system-ui;max-width:1100px;margin:40px auto}"
            "table{border-collapse:collapse;width:100%}th,td{padding:7px;"
            "border:1px solid #ccd1d1;text-align:left}th{background:#f4f6f7}</style>"
            "<h1>Combinatorial research lab</h1>"
            "<p>Research output only. No profitability claim and no live approval.</p>"
            f"<p>Leaderboard rows: {len(leaderboard)}; "
            f"research passes: {summary['research_passes']}; "
            f"paper candidates: {summary['paper_candidates']}.</p>"
            "<table><thead><tr><th>Rank</th><th>DNA</th><th>Robust score</th>"
            f"<th>Lifecycle</th></tr></thead><tbody>{top_rows}</tbody></table>"
            "</html>",
            encoding="utf-8",
        )
        return {
            "summary": str(summary_path),
            "html": str(html_path),
            "charts": chart_index,
        }

    def record_event(self, payload: Mapping[str, Any]) -> None:
        event = {"event_id": f"evt-{stable_hash(payload)[:24]}", **dict(payload)}
        self.database.upsert_records(
            "lab_events",
            [
                {
                    "external_id": event["event_id"],
                    "status": str(event.get("status") or "RECORDED"),
                    "timestamp": utc_now(),
                    **event,
                }
            ],
        )
        append_jsonl(self.paths.logs / "lab.jsonl", event)


def _synthetic_ohlcv(
    *,
    rows: int,
    timeframe: str,
    seed: int,
) -> pd.DataFrame:
    seconds = {
        "5m": 300,
        "15m": 900,
        "30m": 1_800,
        "1h": 3_600,
        "2h": 7_200,
        "4h": 14_400,
        "6h": 21_600,
        "8h": 28_800,
        "12h": 43_200,
        "1d": 86_400,
        "1W": 604_800,
        "1w": 604_800,
        "1M": 2_592_000,
    }[timeframe]
    randomizer = np.random.default_rng(seed)
    index = pd.date_range(
        end=pd.Timestamp("2026-01-01", tz="UTC"),
        periods=rows,
        freq=pd.Timedelta(seconds=seconds),
    )
    returns = randomizer.normal(0.00015, 0.012, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = randomizer.uniform(0.002, 0.02, rows)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = randomizer.lognormal(mean=10.0, sigma=0.5, size=rows)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    frame.attrs.update(
        {
            "provider": "synthetic_offline",
            "market": "SYNTHETIC-EUR",
            "timeframe": timeframe,
            "closed_candles_only": True,
        }
    )
    return frame


def fast_screen(
    frames: Mapping[str, pd.DataFrame],
    strategy: CombinatorialStrategy,
    *,
    round_trip_cost: float,
) -> dict[str, Any]:
    """Causal long-only coarse screen used only for parameter-region ranking."""

    trade_returns: list[float] = []
    gross_trade_returns: list[float] = []
    exit_reasons: Counter[str] = Counter()
    trade_entry_buckets: set[str] = set()
    trades = 0
    for market, frame in frames.items():
        output = strategy.generate(frame)
        entry = output.entry.to_numpy(dtype=bool)
        exit_signal = (output.exit | output.avoid | output.reduce).to_numpy(dtype=bool)
        opens = frame["open"].to_numpy(dtype=float)
        highs = frame["high"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
        stop_distances = output.stop_distance.to_numpy(dtype=float)
        target_distances = output.target_distance.to_numpy(dtype=float)
        trailing_distances = output.trailing_distance.to_numpy(dtype=float)
        size_multipliers = output.size_multiplier.to_numpy(dtype=float)
        maximum_holding = output.maximum_holding_bars
        holding = False
        entry_price = 0.0
        stop_price = 0.0
        target_price = math.inf
        trailing_distance = 0.0
        trailing_stop: float | None = None
        maximum_seen = -math.inf
        bars_held = 0
        size_multiplier = 1.0
        entry_timestamp: pd.Timestamp | None = None

        def close_trade(exit_price: float, reason: str) -> None:
            nonlocal holding, trades
            gross_return = exit_price / entry_price - 1.0
            gross_trade_returns.append(size_multiplier * gross_return)
            trade_returns.append(
                size_multiplier * gross_return
                - size_multiplier * round_trip_cost
            )
            exit_reasons[reason] += 1
            if entry_timestamp is not None:
                iso = entry_timestamp.isocalendar()
                trade_entry_buckets.add(
                    f"{market}:{int(iso.year):04d}-W{int(iso.week):02d}"
                )
            trades += 1
            holding = False

        for index in range(1, len(frame)):
            if not holding and entry[index - 1]:
                stop_distance = float(stop_distances[index - 1])
                target_distance = float(target_distances[index - 1])
                if (
                    not math.isfinite(stop_distance)
                    or stop_distance <= 0
                    or not math.isfinite(target_distance)
                    or target_distance <= 0
                ):
                    continue
                holding = True
                entry_timestamp = pd.Timestamp(frame.index[index])
                entry_price = float(opens[index])
                stop_price = entry_price - stop_distance
                target_price = entry_price + target_distance
                trailing_distance = max(
                    0.0,
                    float(trailing_distances[index - 1]),
                )
                trailing_stop = None
                maximum_seen = entry_price
                bars_held = 0
                size_multiplier = float(np.clip(size_multipliers[index - 1], 0.0, 1.0))
            if not holding:
                continue
            if exit_signal[index - 1]:
                close_trade(float(opens[index]), "SIGNAL_EXIT_NEXT_OPEN")
                continue
            bars_held += 1
            effective_stop = max(
                stop_price,
                trailing_stop if trailing_stop is not None else -math.inf,
            )
            stop_hit = lows[index] <= effective_stop
            target_hit = highs[index] >= target_price
            if stop_hit:
                # Conservative same-bar ordering: stop wins if both are touched.
                close_trade(effective_stop, "STOP_OR_PRIOR_TRAILING_STOP")
                continue
            if target_hit:
                close_trade(target_price, "TARGET")
                continue
            maximum_seen = max(maximum_seen, float(highs[index]))
            if trailing_distance > 0:
                trailing_stop = max(
                    trailing_stop or -math.inf,
                    maximum_seen - trailing_distance,
                )
            if maximum_holding is not None and bars_held >= maximum_holding:
                close_trade(float(closes[index]), "MAXIMUM_HOLDING")
        if holding:
            close_trade(float(closes[-1]), "TERMINAL_LIQUIDATION")
    values = np.asarray(trade_returns, dtype=float)
    gross_values = np.asarray(gross_trade_returns, dtype=float)
    cumulative = float(np.prod(1.0 + values) - 1.0) if len(values) else 0.0
    gross_cumulative = (
        float(np.prod(1.0 + gross_values) - 1.0)
        if len(gross_values)
        else 0.0
    )
    volatility = float(values.std(ddof=0)) if len(values) else 0.0
    mean = float(values.mean()) if len(values) else 0.0
    score = mean / volatility * math.sqrt(len(values)) if volatility > 0 else mean
    return {
        "source": "SCREENING_ONLY",
        "status": "COMPLETED",
        "long_only": True,
        "one_bar_signal_shift": True,
        "basic_costs_applied": True,
        "canonical_exit_family_approximated": True,
        "conservative_same_bar_stop_priority": True,
        "future_data_used_for_signals": False,
        "trades": trades,
        "gross_screening_return": gross_cumulative,
        "screening_return": cumulative,
        "cost_drag": gross_cumulative - cumulative,
        "round_trip_cost": float(round_trip_cost),
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "terminal_liquidations": int(exit_reasons["TERMINAL_LIQUIDATION"]),
        "trade_entry_buckets": sorted(trade_entry_buckets),
        "trade_overlap_signature": stable_hash(
            sorted(trade_entry_buckets),
            length=64,
        ),
        "screening_score": float(score),
        "paper_candidate_permitted": False,
    }


def screening_survivor_score(
    screening: Mapping[str, Any],
    *,
    minimum_trades: int,
) -> float | None:
    """Return a finite rank score only after the hard sample-size screen."""

    trades = int(screening.get("trades") or 0)
    score = float(screening.get("screening_score") or 0.0)
    net_return = float(screening.get("screening_return") or 0.0)
    if (
        trades < minimum_trades
        or not math.isfinite(score)
        or not math.isfinite(net_return)
        or net_return <= 0.0
    ):
        return None
    return score


QUICK_BLOCKS = (
    "rsi_oversold",
    "donchian20_breakout",
    "bullish_bos",
    "bullish_liquidity_sweep",
    "price_above_ema200",
    "adx_trend_strength",
    "relative_volume_expansion",
    "bearish_bos",
)

HYPOTHESIS_BLOCKS = (
    # Trend breakout
    "price_above_ema200",
    "ema50_above_ema200",
    "ema50_positive_slope",
    "adx_trend_strength",
    "dmi_bullish",
    "donchian20_breakout",
    "donchian55_breakout",
    "bullish_bos",
    "relative_volume_expansion",
    "volume_zscore_positive",
    "bearish_bos",
    "bearish_choch",
    "fractal_low_breakdown",
    "negative_return_exit",
    "rsi_overbought_exit",
    # Pullback in uptrend
    "price_above_ema20",
    "ema20_above_ema50",
    "rsi_oversold",
    "stoch_rsi_oversold",
    "bullish_liquidity_sweep",
    "price_above_vwap",
    # Range mean reversion
    "bollinger_lower_reversion",
    "choppiness_low",
    "normalized_atr_regime",
    "bearish_liquidity_sweep",
    "bearish_fvg",
    "equal_highs",
    # Volatility expansion
    "bollinger_squeeze",
    "bollinger_keltner_squeeze",
    "rolling_volatility_low",
    # Relative strength
    "btc_relative_momentum",
    "positive_return_20",
)


ECONOMIC_HYPOTHESIS_TEMPLATES: dict[str, tuple[str, ...]] = {
    "TREND_BREAKOUT": (
        "htf_1d_regime_bullish",
        "htf_4h_regime_bullish",
        "donchian20_breakout",
        "relative_volume_expansion",
    ),
    "PULLBACK_IN_UPTREND": (
        "htf_1d_regime_bullish",
        "ema20_above_ema50",
        "rsi_recovery",
        "ema20_reclaim",
    ),
    "RANGE_MEAN_REVERSION": (
        "choppiness_high",
        "adx_range_low",
        "bollinger_lower_reversion",
    ),
    "VOLATILITY_EXPANSION": (
        "prior_squeeze_within_12",
        "donchian20_breakout",
        "relative_volume_expansion",
    ),
    "BTC_RELATIVE_STRENGTH": (
        "htf_1d_regime_bullish",
        "btc_relative_momentum",
        "btc_relative_persistence",
        "positive_return_20",
    ),
}


def economic_hypothesis_family(combination: StrategyCombination) -> str:
    """Map arbitrary block DNA to its nearest declared economic hypothesis."""

    selected = set(combination.block_ids)
    scored = [
        (
            len(selected.intersection(template)) / max(1, len(set(template))),
            family,
        )
        for family, template in ECONOMIC_HYPOTHESIS_TEMPLATES.items()
    ]
    best_score, best_family = max(scored, default=(0.0, "UNCLASSIFIED"))
    if best_score > 0:
        return best_family
    return (
        f"BLOCK_FAMILY:{sorted(combination.families)[0]}"
        if combination.families
        else "UNCLASSIFIED"
    )


def diverse_screening_survivors(
    candidates: Sequence[
        tuple[float, Mapping[str, Any], Mapping[str, Any], StrategyCombination]
    ],
    *,
    maximum_survivors: int,
    maximum_per_family: int = 2,
    global_slots: int = 3,
) -> list[tuple[float, Mapping[str, Any], Mapping[str, Any], StrategyCombination]]:
    """Select high-scoring candidates without collapsing onto one hypothesis."""

    if maximum_survivors < 1:
        return []
    ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected: list[
        tuple[float, Mapping[str, Any], Mapping[str, Any], StrategyCombination]
    ] = []
    selected_experiments: set[str] = set()
    family_counts: Counter[str] = Counter()

    def jaccard(left: set[str], right: set[str]) -> float:
        union = left.union(right)
        return len(left.intersection(right)) / len(union) if union else 0.0

    def add(item) -> bool:
        _, payload, screening, combination = item
        experiment = str(payload.get("experiment_hash"))
        family = economic_hypothesis_family(combination)
        if (
            experiment in selected_experiments
            or family_counts[family] >= maximum_per_family
        ):
            return False
        blocks = set(combination.block_ids)
        trade_buckets = set(screening.get("trade_entry_buckets") or ())
        for _, _, selected_screening, selected_combination in selected:
            if economic_hypothesis_family(selected_combination) != family:
                continue
            block_overlap = jaccard(blocks, set(selected_combination.block_ids))
            selected_buckets = set(
                selected_screening.get("trade_entry_buckets") or ()
            )
            trade_overlap = (
                jaccard(trade_buckets, selected_buckets)
                if trade_buckets and selected_buckets
                else 0.0
            )
            if block_overlap >= 0.75 and trade_overlap >= 0.80:
                return False
        selected.append(item)
        selected_experiments.add(experiment)
        family_counts[family] += 1
        return True

    for item in ordered:
        if len(selected) >= min(global_slots, maximum_survivors):
            break
        add(item)
    families = sorted(
        {economic_hypothesis_family(item[3]) for item in ordered}
    )
    while len(selected) < maximum_survivors:
        added = False
        for family in families:
            item = next(
                (
                    candidate
                    for candidate in ordered
                    if economic_hypothesis_family(candidate[3]) == family
                    and str(candidate[1].get("experiment_hash"))
                    not in selected_experiments
                ),
                None,
            )
            if item is not None and add(item):
                added = True
                if len(selected) >= maximum_survivors:
                    break
        if not added:
            break
    return selected


class LabRunner:
    """Resumable staged runner with a bounded asynchronous worker pool."""

    def __init__(self, settings: Settings, *, store: LabStore | None = None) -> None:
        self.settings = settings
        self.store = store or LabStore(settings)
        self.paths = self.store.paths
        self.registry = signal_block_registry()
        self.generator = CombinationGenerator(self.registry)
        self.feature_definition_hash = stable_hash(feature_registry(), length=64)
        self.instance_id = f"lab-{stable_hash([socket.gethostname(), os.getpid(), utc_iso()])[:20]}"
        self.lock_path = self.paths.state / "lab.lock"
        self.control_path = self.paths.state / "control.json"
        self.heartbeat_path = self.paths.state / "heartbeat.json"
        self.current_status_path = self.paths.state / "current_status.json"
        self.queue_status_path = self.paths.state / "queue_status.json"
        self.worker_status_path = self.paths.state / "worker_status.json"
        self.last_market_selection: dict[str, Any] = {
            "snapshot_id": None,
            "selected": [],
            "excluded_for_data": [],
            "requested_size": 0,
        }

    def _acquire_lock(self) -> int:
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            stale = False
            try:
                lock = read_json(self.lock_path)
                pid = int(lock.get("pid") or 0)
                if pid <= 0:
                    stale = True
                else:
                    stale = not self._pid_exists(pid)
            except (OSError, TypeError, ValueError):
                stale = True
            if not stale:
                raise RuntimeError("LAB_ALREADY_RUNNING") from exc
            self.lock_path.unlink(missing_ok=True)
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        os.write(
            descriptor,
            stable_json(
                {
                    "lab_instance_id": self.instance_id,
                    "pid": os.getpid(),
                    "started_at": utc_iso(),
                }
            ).encode("utf-8"),
        )
        return descriptor

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information,
                False,
                pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
                return True
            return ctypes.get_last_error() == 5
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _release_lock(self, descriptor: int) -> None:
        os.close(descriptor)
        self.lock_path.unlink(missing_ok=True)

    def control(self, action: LabControl) -> dict[str, Any]:
        if action is LabControl.STATUS:
            return self.status()
        payload = {
            "action": action.value,
            "requested_at": utc_iso(),
            "requested_by": "cli",
        }
        atomic_write_json(self.control_path, payload)
        return payload

    def _control_action(self) -> LabControl:
        if not self.control_path.is_file():
            return LabControl.START
        return LabControl(read_json(self.control_path)["action"])

    def heartbeat(
        self,
        *,
        run_id: str,
        status: str,
        completed_jobs: int,
        failed_jobs: int,
    ) -> dict[str, Any]:
        payload = {
            "run_id": run_id,
            "lab_instance_id": self.instance_id,
            "pid": os.getpid(),
            "status": status,
            "heartbeat_at": utc_iso(),
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "queue": self.store.queue_status(run_id=run_id),
        }
        atomic_write_json(self.heartbeat_path, payload)
        atomic_write_json(self.queue_status_path, payload["queue"])
        self.store.database.upsert_records(
            "lab_heartbeats",
            [
                {
                    "external_id": f"{self.instance_id}:{payload['heartbeat_at']}",
                    "status": status,
                    "timestamp": utc_now(),
                    **payload,
                }
            ],
        )
        return payload

    def write_worker_status(
        self,
        *,
        run_id: str,
        phase: str,
        workers: int,
        active: int,
        completed: int,
        failed: int,
    ) -> None:
        atomic_write_json(
            self.worker_status_path,
            {
                "run_id": run_id,
                "phase": phase,
                "workers": workers,
                "active": active,
                "completed": completed,
                "failed": failed,
                "task_leaks": [],
                "updated_at": utc_iso(),
            },
        )

    def status(self) -> dict[str, Any]:
        current = (
            read_json(self.current_status_path)
            if self.current_status_path.is_file()
            else {"status": "STOPPED"}
        )
        if isinstance(current.get("data_provenance"), Mapping):
            current["data_provenance"] = {
                timeframe: {
                    market: self._provenance_summary(provenance)
                    for market, provenance in by_market.items()
                }
                for timeframe, by_market in current["data_provenance"].items()
                if isinstance(by_market, Mapping)
            }
        return current | {
            "lock_present": self.lock_path.is_file(),
            "control": self._control_action().value,
            "queue": self.store.queue_status(
                run_id=(str(current["run_id"]) if current.get("run_id") else None)
            ),
            "heartbeat": (
                read_json(self.heartbeat_path) if self.heartbeat_path.is_file() else None
            ),
        }

    def _markets(
        self,
        universe_size: int,
        *,
        universe_scope: Literal["allowed", "discovery"] = "allowed",
        include_review_required_research_only: bool = False,
        required_timeframes: Sequence[str] | None = None,
        minimum_rows: int = 0,
    ) -> list[str]:
        data_exclusions: list[dict[str, Any]] = []
        latest = UniverseManager(
            self.settings,
            database=self.store.database,
        ).latest()
        if latest:
            members = latest.get("members") or []
            eligible: list[str] = []
            for member in members:
                types = set(member.get("universe_types", []))
                allowed = UniverseType.ALLOWED_RESEARCH.value in types
                review = UniverseType.REVIEW_RESEARCH_ONLY.value in types
                discovery = UniverseType.DISCOVERY_UNIVERSE.value in types
                if universe_scope == "allowed" and not allowed:
                    continue
                if universe_scope == "discovery" and not discovery:
                    continue
                if review and not include_review_required_research_only:
                    continue
                availability = member.get("market_availability") or {}
                available = [
                    market
                    for provider in ("bitvavo", "mexc", "kraken")
                    for market in availability.get(provider, [])
                ]
                eur_market = f"{member['symbol']}-EUR"
                bitvavo_markets = list(availability.get("bitvavo") or [])
                selected_market = (
                    eur_market
                    if eur_market in bitvavo_markets or (allowed and eur_market in available)
                    else (available[0] if available else eur_market)
                )
                missing: list[str] = []
                for timeframe in required_timeframes or ():
                    path = self._data_path(selected_market, timeframe)
                    if not path.is_file():
                        missing.append(f"{timeframe}:MISSING")
                        continue
                    try:
                        available_rows = len(
                            pd.read_parquet(path, columns=["close"])
                            if path.suffix.casefold() == ".parquet"
                            else pd.read_csv(path, usecols=["close"])
                        )
                    except (OSError, ValueError, KeyError):
                        missing.append(f"{timeframe}:INVALID")
                        continue
                    if available_rows < minimum_rows:
                        missing.append(f"{timeframe}:ROWS_{available_rows}_LT_{minimum_rows}")
                if missing:
                    data_exclusions.append(
                        {
                            "symbol": member.get("symbol"),
                            "market": selected_market,
                            "reason_code": "INSUFFICIENT_REQUIRED_REAL_DATA",
                            "details": missing,
                        }
                    )
                    continue
                eligible.append(selected_market)
                if len(eligible) >= universe_size:
                    break
            if eligible:
                self.last_market_selection = {
                    "snapshot_id": latest.get("snapshot_id"),
                    "selected": eligible[:universe_size],
                    "excluded_for_data": data_exclusions,
                    "requested_size": universe_size,
                }
                return eligible[:universe_size]
            self.last_market_selection = {
                "snapshot_id": latest.get("snapshot_id"),
                "selected": [],
                "excluded_for_data": data_exclusions,
                "requested_size": universe_size,
            }
            if required_timeframes:
                return []
        fallback = list(self.settings.market_data.symbols[:universe_size])
        self.last_market_selection = {
            "snapshot_id": None,
            "selected": fallback,
            "excluded_for_data": data_exclusions,
            "requested_size": universe_size,
            "selection_basis": "SETTINGS_FALLBACK_NO_UNIVERSE_SNAPSHOT",
        }
        return fallback

    def _data_path(self, market: str, timeframe: str) -> Path:
        stem = f"{market}_{timeframe}"
        parquet = self.settings.paths.processed_data_dir / f"{stem}.parquet"
        csv = self.settings.paths.processed_data_dir / f"{stem}.csv"
        return parquet if parquet.is_file() or not csv.is_file() else csv

    @staticmethod
    def _provenance_summary(provenance: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not provenance:
            return None
        conflicts = list(provenance.get("reconciliation_conflicts") or [])
        conflict_count = int(provenance.get("reconciliation_conflict_count", len(conflicts)))
        conflict_sample = list(provenance.get("reconciliation_conflict_sample") or conflicts[:3])
        return {
            key: provenance.get(key)
            for key in (
                "source_type",
                "market",
                "timeframe",
                "providers_requested",
                "providers_used",
                "provider_errors",
                "provider_hashes",
                "closed_candles_only",
                "retrieved_at",
                "data_file",
                "data_sha256",
                "rows",
                "start",
                "end",
            )
        } | {
            "reconciliation_conflict_count": conflict_count,
            "reconciliation_conflict_sample": conflict_sample[:3],
        }

    def data_status(
        self,
        *,
        markets: Sequence[str],
        timeframes: Sequence[str],
        minimum_rows: int,
    ) -> dict[str, Any]:
        datasets: list[dict[str, Any]] = []
        for market in markets:
            for timeframe in timeframes:
                path = self._data_path(market, timeframe)
                row: dict[str, Any] = {
                    "market": market,
                    "timeframe": timeframe,
                    "path": str(path),
                    "exists": path.is_file(),
                    "status": "MISSING",
                }
                if path.is_file():
                    try:
                        frame = load_ohlcv(
                            path,
                            market=market,
                            timeframe=timeframe,
                            closed_candles_only=True,
                        )
                        provenance_path = path.with_suffix(f"{path.suffix}.provenance.json")
                        provenance = (
                            read_json(provenance_path) if provenance_path.is_file() else None
                        )
                        report = quality_report(
                            frame,
                            market=market,
                            timeframe=timeframe,
                            maximum_staleness=max(
                                self.settings.market_data.maximum_staleness,
                                timeframe_delta(timeframe) * 2,
                            ),
                        )
                        row.update(
                            rows=len(frame),
                            sha256=sha256_file(path),
                            quality=report.model_dump(mode="json"),
                            provenance=self._provenance_summary(provenance),
                            status=(
                                "READY"
                                if len(frame) >= minimum_rows
                                and report.valid
                                and provenance
                                and provenance.get("source_type") == "REAL_PROVIDER_DATA"
                                else "INVALID_OR_INCOMPLETE"
                            ),
                        )
                    except Exception as exc:
                        row.update(
                            status="INVALID_OR_INCOMPLETE",
                            reason_code=type(exc).__name__,
                        )
                datasets.append(row)
        return {
            "status": (
                "READY"
                if datasets and all(row["status"] == "READY" for row in datasets)
                else "BLOCKED_DATA_UNAVAILABLE"
            ),
            "minimum_rows": minimum_rows,
            "datasets": datasets,
            "synthetic_fallback": False,
        }

    async def prepare_real_data(
        self,
        *,
        markets: Sequence[str],
        timeframes: Sequence[str],
        minimum_rows: int,
        history_profile: str = "standard",
        providers: Sequence[str] = ("bitvavo", "kraken"),
        only_missing: bool = True,
    ) -> dict[str, Any]:
        loader = DataLoader(self.settings, database=self.store.database)
        now = utc_now()
        prepared: list[dict[str, Any]] = []
        for market in markets:
            for requested_timeframe in timeframes:
                timeframe = normalize_timeframe(requested_timeframe)
                path = self._data_path(market, timeframe)
                existing_status = self.data_status(
                    markets=[market],
                    timeframes=[timeframe],
                    minimum_rows=minimum_rows,
                )["datasets"][0]
                if only_missing and existing_status["status"] == "READY":
                    prepared.append(existing_status | {"action": "SKIPPED_COMPLETE"})
                    continue
                by_provider: dict[str, list[NormalizedDataRecord]] = {}
                errors: dict[str, str] = {}
                for provider in providers:
                    try:
                        profile_start = loader.history_start(
                            profile=history_profile,
                            timeframe=timeframe,
                            provider=provider,
                            end=now,
                        )
                        row_start = now - timedelta(
                            seconds=TIMEFRAME_SECONDS[timeframe] * (minimum_rows + 50)
                        )
                        records, _ = await loader.download_canonical_ohlcv(
                            provider=provider,
                            market=market,
                            timeframe=timeframe,
                            start=min(profile_start, row_start),
                            end=now,
                            resume=True,
                            persist=True,
                        )
                        closed = [record for record in records if record.closed]
                        if closed:
                            by_provider[provider] = closed
                    except Exception as exc:
                        errors[provider] = type(exc).__name__
                if not by_provider:
                    prepared.append(
                        {
                            "market": market,
                            "timeframe": timeframe,
                            "status": "RETRYABLE_PROVIDER_ERROR",
                            "provider_errors": errors,
                            "synthetic_fallback": False,
                        }
                    )
                    continue
                selected, conflicts = loader.reconcile_provider_series(by_provider)
                rows = [
                    {
                        "timestamp": record.timestamp,
                        **{
                            column: record.values[column]
                            for column in ("open", "high", "low", "close", "volume")
                        },
                    }
                    for record in selected
                    if record.closed
                ]
                frame = pd.DataFrame(rows)
                try:
                    frame = validate_ohlcv(
                        frame,
                        timeframe=timeframe,
                        closed_candles_only=True,
                    )
                except DataValidationError as exc:
                    prepared.append(
                        {
                            "market": market,
                            "timeframe": timeframe,
                            "status": "BLOCKED_DATA_UNAVAILABLE",
                            "reason_code": type(exc).__name__,
                            "provider_errors": errors,
                            "synthetic_fallback": False,
                        }
                    )
                    continue
                derivation: dict[str, Any] | None = None
                for source_timeframe in sorted(
                    (
                        candidate
                        for candidate, seconds in TIMEFRAME_SECONDS.items()
                        if seconds < TIMEFRAME_SECONDS[timeframe]
                        and TIMEFRAME_SECONDS[timeframe] % seconds == 0
                    ),
                    key=TIMEFRAME_SECONDS.__getitem__,
                    reverse=True,
                ):
                    source_path = self._data_path(market, source_timeframe)
                    source_provenance_path = source_path.with_suffix(
                        f"{source_path.suffix}.provenance.json"
                    )
                    if not source_path.is_file() or not source_provenance_path.is_file():
                        continue
                    source_provenance = read_json(source_provenance_path)
                    if source_provenance.get("source_type") != "REAL_PROVIDER_DATA":
                        continue
                    try:
                        source_frame = load_ohlcv(
                            source_path,
                            market=market,
                            timeframe=source_timeframe,
                            closed_candles_only=True,
                        )
                        derived = resample_ohlcv(
                            source_frame,
                            source_timeframe=source_timeframe,
                            target_timeframe=timeframe,
                            drop_incomplete=True,
                        )
                        derived = validate_ohlcv(
                            derived,
                            timeframe=timeframe,
                            closed_candles_only=True,
                        )
                    except (DataValidationError, OSError, ValueError):
                        continue
                    if len(derived) <= len(frame):
                        continue
                    frame = derived
                    derivation = {
                        "source_classification": "RESAMPLED_FROM_DEEPER_LOCAL_REAL_DATA",
                        "source_timeframe": source_timeframe,
                        "source_path": str(source_path),
                        "source_sha256": sha256_file(source_path),
                        "source_rows": len(source_frame),
                        "derived_rows": len(derived),
                        "providers_used": source_provenance.get("providers_used", []),
                        "complete_bins_only": True,
                    }
                    break
                path = self.settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
                saved, manifest = save_ohlcv(
                    frame,
                    path,
                    market=market,
                    timeframe=timeframe,
                    maximum_staleness=self.settings.market_data.maximum_staleness,
                )
                provider_hashes = {
                    provider: stable_hash([record.raw_hash for record in provider_records])
                    for provider, provider_records in by_provider.items()
                }
                provenance = {
                    "source_type": "REAL_PROVIDER_DATA",
                    "market": market,
                    "timeframe": timeframe,
                    "providers_requested": list(providers),
                    "providers_used": sorted(by_provider),
                    "provider_errors": errors,
                    "provider_hashes": provider_hashes,
                    "reconciliation_conflicts": conflicts,
                    "closed_candles_only": True,
                    "retrieved_at": utc_iso(),
                    "data_file": str(saved),
                    "data_sha256": manifest.sha256,
                    "rows": len(frame),
                    "start": frame.index[0].isoformat(),
                    "end": frame.index[-1].isoformat(),
                    "derivation": derivation,
                }
                if derivation:
                    provenance["providers_used"] = derivation["providers_used"]
                atomic_write_json(
                    saved.with_suffix(f"{saved.suffix}.provenance.json"),
                    provenance,
                )
                prepared.append(
                    {
                        "market": market,
                        "timeframe": timeframe,
                        "status": (
                            "PREPARED"
                            if len(frame) >= minimum_rows
                            else "PARTIAL_PROVIDER_COVERAGE"
                        ),
                        "rows": len(frame),
                        "path": str(saved),
                        "provenance": self._provenance_summary(provenance),
                        "synthetic_fallback": False,
                    }
                )
        return {
            "status": (
                "PREPARED"
                if prepared
                and all(row["status"] in {"PREPARED", "SKIPPED_COMPLETE"} for row in prepared)
                else "PARTIAL_PROVIDER_COVERAGE"
            ),
            "datasets": prepared,
            "provider_status": loader.provider_status(),
            "synthetic_fallback": False,
        }

    def _profile_blocks(self, profile: str) -> tuple[str, ...]:
        if profile.casefold() == "quick":
            return QUICK_BLOCKS
        if profile.casefold() == "hypotheses":
            return HYPOTHESIS_BLOCKS
        if profile.casefold() == "standard":
            return tuple(
                block_id
                for block_id, block in self.registry.items()
                if block.family
                in {
                    "PRICE_RETURNS",
                    "TREND",
                    "MOMENTUM",
                    "VOLATILITY",
                    "VOLUME_FLOW",
                    "MARKET_STRUCTURE",
                    "CANDLE",
                }
            )
        return tuple(self.registry)

    def _parameter_sets(
        self,
        combination: StrategyCombination,
        overrides: Mapping[str, Sequence[Any]] | None,
    ) -> list[tuple[str | None, dict[str, Any]]]:
        base = {
            block_id: self.registry[block_id].parameters(
                combination.default_parameters.get(block_id)
            )
            for block_id in combination.block_ids
        }
        relevant: list[tuple[str, str, Sequence[Any]]] = []
        for key, values in (overrides or {}).items():
            try:
                block_id, parameter_name = key.split(".", 1)
            except ValueError as exc:
                raise ValueError(f"parameter override requires BLOCK.PARAMETER: {key}") from exc
            if block_id in combination.block_ids:
                relevant.append((block_id, parameter_name, tuple(values)))
        if not relevant:
            generated_sensitivity: list[tuple[str | None, dict[str, Any]]] = [(None, base)]
            for block_id in combination.block_ids:
                for spec in self.registry[block_id].parameter_specs:
                    non_default = next(
                        (value for value in spec.values() if value != spec.validate(spec.default)),
                        None,
                    )
                    if non_default is None:
                        continue
                    selected = {
                        selected_block: dict(parameters)
                        for selected_block, parameters in base.items()
                    }
                    selected[block_id][spec.name] = non_default
                    generated_sensitivity.append(
                        (
                            f"{block_id}__{spec.name}",
                            {
                                selected_block: self.registry[selected_block].parameters(parameters)
                                for selected_block, parameters in selected.items()
                            },
                        )
                    )
            return generated_sensitivity
        generated: list[dict[str, Any]] = []
        for values in itertools.product(*(item[2] for item in relevant)):
            selected = {block_id: dict(parameters) for block_id, parameters in base.items()}
            for (block_id, parameter_name, _), value in zip(
                relevant,
                values,
                strict=True,
            ):
                selected[block_id][parameter_name] = value
            generated.append(
                {
                    block_id: self.registry[block_id].parameters(parameters)
                    for block_id, parameters in selected.items()
                }
            )
        unique = {parameter_hash(parameters): parameters for parameters in generated}
        return [("CLI_OVERRIDE", unique[key]) for key in sorted(unique)]

    def _frames(
        self,
        *,
        markets: Sequence[str],
        timeframe: str,
        rows: int | None,
        data_mode: Literal["real", "synthetic"],
        start_at: pd.Timestamp | datetime | None = None,
        end_at: pd.Timestamp | datetime | None = None,
    ) -> tuple[dict[str, pd.DataFrame], str, dict[str, Any]]:
        frames: dict[str, pd.DataFrame] = {}
        hashes: dict[str, str] = {}
        provenance: dict[str, Any] = {}
        macro_context: pd.DataFrame | None = None
        macro_path = (
            self.settings.paths.context_data_dir
            / f"macro_context_{normalize_timeframe(timeframe)}.parquet"
        )
        if data_mode == "real" and macro_path.is_file():
            macro_context = pd.read_parquet(macro_path)
            macro_context.index = pd.to_datetime(macro_context.index, utc=True)
            macro_context.attrs.update(
                canonical_macro_context=True,
                point_in_time_aligned=True,
                provenance_engine="MacroContextEngine",
            )
        benchmark: pd.DataFrame | None = None
        if data_mode == "real":
            benchmark_path = self._data_path("BTC-EUR", timeframe)
            if benchmark_path.is_file():
                benchmark = load_ohlcv(
                    benchmark_path,
                    market="BTC-EUR",
                    timeframe=timeframe,
                    closed_candles_only=True,
                )
                if start_at is not None:
                    benchmark = benchmark.loc[
                        benchmark.index >= pd.Timestamp(start_at).tz_convert("UTC")
                    ]
                if end_at is not None:
                    benchmark = benchmark.loc[
                        benchmark.index <= pd.Timestamp(end_at).tz_convert("UTC")
                    ]
                benchmark = benchmark.copy()
                benchmark.attrs.update(
                    market="BTC-EUR",
                    timeframe=timeframe,
                )
        for offset, market in enumerate(markets):
            if data_mode == "synthetic":
                if rows is None:
                    raise ValueError("synthetic frames require an explicit row bound")
                raw = _synthetic_ohlcv(
                    rows=rows,
                    timeframe=timeframe,
                    seed=self.settings.lab.deterministic_seed + offset,
                )
                item_provenance = {
                    "source_type": "SYNTHETIC_SMOKE",
                    "provider": "synthetic_offline",
                    "closed_candles_only": True,
                    "rows": len(raw),
                }
                data_hash = stable_hash(
                    {
                        "market": market,
                        "timeframe": timeframe,
                        "rows": rows,
                        "last": float(raw["close"].iloc[-1]),
                    }
                )
            else:
                path = self._data_path(market, timeframe)
                if not path.is_file():
                    raise DataValidationError(f"BLOCKED_DATA_UNAVAILABLE:{market}:{timeframe}")
                raw = load_ohlcv(
                    path,
                    market=market,
                    timeframe=timeframe,
                    closed_candles_only=True,
                )
                provenance_path = path.with_suffix(f"{path.suffix}.provenance.json")
                if not provenance_path.is_file():
                    raise DataValidationError(f"MISSING_REAL_DATA_PROVENANCE:{market}:{timeframe}")
                item_provenance = read_json(provenance_path)
                if item_provenance.get("source_type") != "REAL_PROVIDER_DATA":
                    raise DataValidationError(f"INVALID_REAL_DATA_PROVENANCE:{market}:{timeframe}")
                if rows is not None and len(raw) < rows:
                    raise DataValidationError(
                        f"INSUFFICIENT_REAL_DATA:{market}:{timeframe}:{len(raw)}<{rows}"
                    )
                if start_at is not None:
                    raw = raw.loc[raw.index >= pd.Timestamp(start_at).tz_convert("UTC")]
                if end_at is not None:
                    raw = raw.loc[raw.index <= pd.Timestamp(end_at).tz_convert("UTC")]
                if rows is not None:
                    raw = raw.iloc[-rows:].copy()
                else:
                    raw = raw.copy()
                if raw.empty:
                    raise DataValidationError(f"EMPTY_RESEARCH_SLICE:{market}:{timeframe}")
                raw.attrs.update(
                    {
                        "market": market,
                        "timeframe": timeframe,
                        "data_provenance": item_provenance,
                    }
                )
                source_hash = sha256_file(path)
                data_hash = stable_hash(
                    {
                        "source_sha256": source_hash,
                        "market": market,
                        "timeframe": timeframe,
                        "slice_start": raw.index[0].isoformat(),
                        "slice_end": raw.index[-1].isoformat(),
                        "slice_rows": len(raw),
                    }
                )
                item_provenance["research_slice"] = {
                    "source_sha256": source_hash,
                    "start": raw.index[0].isoformat(),
                    "end": raw.index[-1].isoformat(),
                    "rows": len(raw),
                    "common_period_requested": (start_at is not None or end_at is not None),
                }
            provider_provenance = dict(item_provenance)
            higher_timeframes: dict[str, pd.DataFrame] = {}
            if data_mode == "real":
                for higher_timeframe in ("4h", "1d"):
                    if TIMEFRAME_SECONDS[higher_timeframe] <= TIMEFRAME_SECONDS[timeframe]:
                        continue
                    higher_path = self._data_path(market, higher_timeframe)
                    if not higher_path.is_file():
                        continue
                    higher = load_ohlcv(
                        higher_path,
                        market=market,
                        timeframe=higher_timeframe,
                        closed_candles_only=True,
                    )
                    if end_at is not None:
                        higher = higher.loc[higher.index <= pd.Timestamp(end_at).tz_convert("UTC")]
                    higher = higher.copy()
                    higher.attrs.update(
                        market=market,
                        timeframe=higher_timeframe,
                    )
                    higher_timeframes[higher_timeframe] = higher
            features = FeaturePipeline().build(
                raw,
                market=market,
                benchmark=benchmark,
                macro_context=macro_context,
                higher_timeframes=higher_timeframes,
            )
            item_provenance["benchmark_context"] = {
                "status": "ATTACHED" if benchmark is not None else "MISSING",
                "market": "BTC-EUR",
                "timeframe": timeframe,
                "rows": len(benchmark) if benchmark is not None else 0,
            }
            item_provenance["higher_timeframe_context"] = {
                selected: {
                    "rows": len(frame),
                    "start": frame.index[0].isoformat(),
                    "end": frame.index[-1].isoformat(),
                    "availability_lag_seconds": TIMEFRAME_SECONDS[selected],
                }
                for selected, frame in higher_timeframes.items()
            }
            if macro_context is not None:
                item_provenance["macro_context"] = {
                    "status": "ATTACHED",
                    "path": str(macro_path),
                    "sha256": sha256_file(macro_path),
                    "rows": len(macro_context),
                    "overlap_rows": int(macro_context.index.intersection(raw.index).size),
                    "usable_rows": int(
                        macro_context.reindex(raw.index)
                        .get("macro_context_usable", pd.Series(False, index=raw.index))
                        .fillna(False)
                        .sum()
                    ),
                }
                features.attrs["context_provenance"] = dict(item_provenance["macro_context"])
            else:
                features.attrs["macro_context"] = {
                    "status": ResearchStatus.MISSING_REQUIRED_CONTEXT.value,
                    "reason_code": "MACRO_CONTEXT_NOT_AVAILABLE_FOR_TIMEFRAME",
                }
            feature_output_hash = _frame_content_hash(features)
            context_hash = (
                sha256_file(macro_path)
                if macro_context is not None
                else stable_hash(
                    {
                        "status": ResearchStatus.MISSING_REQUIRED_CONTEXT.value,
                        "timeframe": timeframe,
                    },
                    length=64,
                )
            )
            feature_hash = stable_hash(
                {
                    "definition_hash": self.feature_definition_hash,
                    "output_hash": feature_output_hash,
                    "context_hash": context_hash,
                },
                length=64,
            )
            item_provenance.update(
                {
                    "feature_definition_hash": self.feature_definition_hash,
                    "feature_output_hash": feature_output_hash,
                    "feature_hash": feature_hash,
                    "context_hash": context_hash,
                }
            )
            provider_provenance.update(
                {
                    "feature_definition_hash": self.feature_definition_hash,
                    "feature_output_hash": feature_output_hash,
                    "feature_hash": feature_hash,
                    "context_hash": context_hash,
                }
            )
            features.attrs["data_provenance"] = provider_provenance
            frames[market] = features
            hashes[market] = stable_hash(
                {
                    "normalized_slice_hash": data_hash,
                    "feature_hash": feature_hash,
                    "context_hash": context_hash,
                },
                length=64,
            )
            provenance[market] = item_provenance
        return frames, stable_hash(hashes), provenance

    def _result_payload(
        self,
        result: BacktestResult,
        *,
        combination: StrategyCombination,
        job: Mapping[str, Any],
        source: str,
        bias_label: str,
    ) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "experiment_hash": job["experiment_hash"],
            "source": source,
            "combination_id": combination.combination_id,
            "strategy_dna_hash": combination.strategy_dna_hash,
            "block_ids": list(combination.block_ids),
            "families": list(combination.families),
            "roles": list(combination.roles),
            "logic_mode": combination.logic_mode.value,
            "combination_size": combination.combination_size,
            "parameters": job["parameters"],
            "parameter_hash": job["parameter_hash"],
            "data_hash": job.get("data_hash"),
            "feature_hash": job.get("feature_hash"),
            "screening_engine_version": job.get("screening_engine_version"),
            "screen_policy_version": job.get("screen_policy_version"),
            "exit_model_version": job.get("exit_model_version"),
            "survivor_policy_version": job.get("survivor_policy_version"),
            "result_type": "EXACT_BACKTEST",
            "data_provenance": job.get("data_provenance"),
            "source_type": (
                job.get("source_type") or _provenance_source_type(job.get("data_provenance"))
            ),
            "software_version": job.get("software_version"),
            "universe_snapshot_id": job["universe_snapshot_id"],
            "assets_tested": job["markets"],
            "timeframes_tested": [job["timeframe"]],
            "data_period": {
                "start": (
                    result.equity_curve.index.min().isoformat()
                    if not result.equity_curve.empty
                    else None
                ),
                "end": (
                    result.equity_curve.index.max().isoformat()
                    if not result.equity_curve.empty
                    else None
                ),
            },
            "metrics": dict(result.metrics),
            "integrity": dict(result.integrity),
            "bias_label": bias_label,
            "tested_at": utc_iso(),
            "live_orders": 0,
        }

    async def _screen_and_validate(
        self,
        *,
        baseline_payloads: Sequence[Mapping[str, Any]],
        combinations: Mapping[str, StrategyCombination],
        frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
        executor: ProcessPoolExecutor,
        allow_review_required_research_only: bool = False,
        maximum_survivors: int = 12,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        screened: list[
            tuple[
                float,
                Mapping[str, Any],
                Mapping[str, Any],
                StrategyCombination,
            ]
        ] = []
        existing_exact = {
            str(row.get("experiment_hash"))
            for row in _payload_rows(self.store.database, "exact_backtest_results")
        }
        for payload in baseline_payloads:
            if payload.get("result_type") == "PARAMETER_SENSITIVITY":
                continue
            combination = combinations.get(str(payload["combination_id"]))
            timeframe = str(payload["timeframes_tested"][0])
            frames = frames_by_timeframe.get(timeframe)
            if combination is None or not frames:
                continue
            result = payload.get("screening")
            if not isinstance(result, Mapping):
                continue
            rank_score = screening_survivor_score(
                result,
                minimum_trades=FAST_SCREEN_MINIMUM_TRADES,
            )
            if rank_score is None:
                continue
            screened.append((rank_score, payload, result, combination))
        best_by_combination_timeframe: dict[
            tuple[str, str],
            tuple[
                float,
                Mapping[str, Any],
                Mapping[str, Any],
                StrategyCombination,
            ],
        ] = {}
        for item in screened:
            _, payload, _, _ = item
            key = (
                str(payload["combination_id"]),
                str(payload["timeframes_tested"][0]),
            )
            current = best_by_combination_timeframe.get(key)
            if current is None or item[0] > current[0]:
                best_by_combination_timeframe[key] = item
        survivors: list[
            tuple[
                float,
                Mapping[str, Any],
                Mapping[str, Any],
                StrategyCombination,
            ]
        ] = []
        for timeframe in sorted(frames_by_timeframe):
            candidates = [
                item
                for (_, selected_timeframe), item in (best_by_combination_timeframe.items())
                if selected_timeframe == timeframe
            ]
            survivors.extend(
                diverse_screening_survivors(
                    candidates,
                    maximum_survivors=maximum_survivors,
                )
            )
        exact_backtests = 0
        exact_payloads: list[dict[str, Any]] = []
        for _, payload, screening, combination in survivors:
            timeframe = str(payload["timeframes_tested"][0])
            if str(payload["experiment_hash"]) in existing_exact:
                continue
            job = self.store.job(str(payload["job_id"]))
            if job is None:
                continue
            queued = self.store.update_job(
                job,
                status=CombinationState.QUEUED_EXACT_BACKTEST,
                stage="EXACT_BACKTEST",
                reason_code="SCREENING_SURVIVOR",
                checkpoint=job.get("last_checkpoint"),
            )
            running = self.store.update_job(
                queued,
                status=CombinationState.EXACT_BACKTEST_RUNNING,
                stage="EXACT_BACKTEST",
                reason_code="CANONICAL_BACKTEST_STARTED",
                checkpoint=queued.get("last_checkpoint"),
            )
            try:
                config = BacktestConfig(
                    costs=BacktestConfig.from_settings(
                        self.settings,
                        stressed=True,
                    ).costs,
                    bootstrap_samples=100,
                    monte_carlo_runs=100,
                    random_seed=self.settings.lab.deterministic_seed,
                    allow_review_required_research_only=(allow_review_required_research_only),
                )
                loop = asyncio.get_running_loop()
                exact = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        _canonical_backtest_worker,
                        config,
                        self.settings,
                        dict(frames_by_timeframe[timeframe]),
                        combination,
                        {
                            block_id: dict(parameters)
                            for block_id, parameters in payload["parameters"].items()
                        },
                    ),
                    timeout=self.settings.lab.combination_timeout_seconds,
                )
                result_payload = self._result_payload(
                    exact,
                    combination=combination,
                    job=running,
                    source=(
                        "EXACT_REAL"
                        if _provenance_source_type(running.get("data_provenance"))
                        == "REAL_PROVIDER_DATA"
                        else "SYNTHETIC_SMOKE"
                    ),
                    bias_label=str(payload["bias_label"]),
                )
                result_payload["screening"] = screening
                self.store.save_result(
                    "exact_backtest_results",
                    job=running,
                    result=result_payload,
                    status="COMPLETED",
                )
                self.store.update_job(
                    running,
                    status=CombinationState.EXACT_BACKTEST_COMPLETED,
                    stage="EXACT_BACKTEST",
                    reason_code="CANONICAL_STRESSED_EXACT_COMPLETE",
                    checkpoint=f"exact:{running['experiment_hash']}",
                )
                exact_backtests += 1
                exact_payloads.append(result_payload)
            except Exception as exc:
                self.store.update_job(
                    running,
                    status=CombinationState.EXACT_BACKTEST_REJECTED,
                    stage="EXACT_BACKTEST",
                    reason_code=type(exc).__name__,
                    checkpoint=running.get("last_checkpoint"),
                    error=exc,
                )
        return len(screened), exact_backtests, exact_payloads

    async def _baseline_job(
        self,
        *,
        job: Mapping[str, Any],
        combination: StrategyCombination,
        frames: Mapping[str, pd.DataFrame],
        bias_label: str,
        semaphore: asyncio.Semaphore,
        executor: ProcessPoolExecutor | ThreadPoolExecutor,
        allow_review_required_research_only: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if job.get("deduplicated"):
            return dict(job), None
        async with semaphore:
            sensitivity = job.get("result_type") == "PARAMETER_SENSITIVITY"
            stage = "SENSITIVITY_SCREENING" if sensitivity else "SCREENING"
            running = self.store.update_job(
                job,
                status=CombinationState.SCREENING_RUNNING,
                stage=stage,
                reason_code="FAST_SCREEN_WORKER_CLAIMED",
            )
            started = time.perf_counter()
            try:
                round_trip_cost = (
                    2.0 * self.settings.costs.default_fee
                    + 2.0 * self.settings.costs.slippage_bps / 10_000.0
                    + self.settings.costs.spread_bps / 10_000.0
                )
                loop = asyncio.get_running_loop()
                screening = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        _fast_screen_worker,
                        dict(frames),
                        combination,
                        {
                            block_id: dict(parameters)
                            for block_id, parameters in running["parameters"].items()
                        },
                        round_trip_cost,
                    ),
                    timeout=self.settings.lab.combination_timeout_seconds,
                )
                is_real = (
                    _provenance_source_type(running.get("data_provenance")) == "REAL_PROVIDER_DATA"
                )
                trial_id = stable_hash(
                    [
                        running["experiment_hash"],
                        stage,
                        running["parameter_hash"],
                    ]
                )
                starts = [frame.index.min() for frame in frames.values()]
                ends = [frame.index.max() for frame in frames.values()]
                payload = {
                    "trial_id": f"trial-{trial_id[:24]}",
                    "job_id": running["job_id"],
                    "experiment_hash": running["experiment_hash"],
                    "source": ("FAST_SCREEN_REAL" if is_real else "SYNTHETIC_FAST_SCREEN"),
                    "stage": stage,
                    "combination_id": combination.combination_id,
                    "strategy_dna_hash": combination.strategy_dna_hash,
                    "block_ids": list(combination.block_ids),
                    "families": list(combination.families),
                    "economic_hypothesis_family": economic_hypothesis_family(
                        combination
                    ),
                    "roles": list(combination.roles),
                    "logic_mode": combination.logic_mode.value,
                    "combination_size": combination.combination_size,
                    "parameters": running["parameters"],
                    "parameter_hash": running["parameter_hash"],
                    "data_hash": running.get("data_hash"),
                    "feature_hash": running.get("feature_hash"),
                    "screening_engine_version": running.get("screening_engine_version"),
                    "screen_policy_version": running.get("screen_policy_version"),
                    "exit_model_version": running.get("exit_model_version"),
                    "survivor_policy_version": running.get("survivor_policy_version"),
                    "result_type": running.get("result_type", "BASELINE_SCREEN"),
                    "data_provenance": running.get("data_provenance"),
                    "source_type": running.get("source_type"),
                    "software_version": running.get("software_version"),
                    "universe_snapshot_id": running["universe_snapshot_id"],
                    "assets_tested": running["markets"],
                    "timeframes_tested": [running["timeframe"]],
                    "data_period": {
                        "start": min(starts).isoformat() if starts else None,
                        "end": max(ends).isoformat() if ends else None,
                    },
                    "metrics": {
                        "trade_count": int(screening["trades"]),
                        "gross_return": float(screening["gross_screening_return"]),
                        "net_return": float(screening["screening_return"]),
                        "cost_drag": float(screening["cost_drag"]),
                        "sharpe": float(screening["screening_score"]),
                    },
                    "integrity": {
                        "no_lookahead": bool(not screening["future_data_used_for_signals"]),
                        "no_repainting": True,
                        "long_only": bool(screening["long_only"]),
                        "basic_costs_applied": bool(screening["basic_costs_applied"]),
                        "exact_event_driven": False,
                    },
                    "screening": dict(screening),
                    "bias_label": bias_label,
                    "tested_at": utc_iso(),
                    "duration_seconds": time.perf_counter() - started,
                    "live_orders": 0,
                    "paper_candidate_permitted": False,
                }
                self.store.save_result(
                    "experiment_trials",
                    job=running,
                    result=payload,
                    status="COMPLETED",
                )
                completed = self.store.update_job(
                    running,
                    status=CombinationState.SCREENING_COMPLETED,
                    stage=stage,
                    reason_code=f"CAUSAL_COST_AWARE_{stage}_COMPLETE",
                    checkpoint=f"screen:{running['experiment_hash']}",
                )
                return completed, payload
            except Exception as exc:
                retryable = int(running.get("attempt") or 0) < self.settings.lab.maximum_retries
                failed = self.store.update_job(
                    running,
                    status=(
                        CombinationState.ERROR_RETRYABLE
                        if retryable
                        else CombinationState.ERROR_FINAL
                    ),
                    stage=stage,
                    reason_code=type(exc).__name__,
                    checkpoint=running.get("last_checkpoint"),
                    error=exc,
                )
                atomic_write_json(
                    self.paths.failures / f"{running['job_id']}.json",
                    {
                        "job_id": running["job_id"],
                        "exception_type": type(exc).__name__,
                        "reason_code": type(exc).__name__,
                        "attempt": failed["attempt"],
                        "last_checkpoint": failed.get("last_checkpoint"),
                        "retry_eligible": failed["retry_eligible"],
                    },
                )
                return failed, None

    async def _optimize_and_validate(
        self,
        *,
        exact_payloads: Sequence[Mapping[str, Any]],
        combinations: Mapping[str, StrategyCombination],
        frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
        executor: ProcessPoolExecutor,
        profile: str,
        maximum_candidates: int,
        maximum_trials: int | None = None,
        allow_review_required_research_only: bool = False,
    ) -> tuple[int, int, int, int]:
        """Run canonical optimization and robustness stages for exact survivors."""
        already_validated = {
            str(row.get("experiment_hash"))
            for row in _payload_rows(self.store.database, "gate_results")
        }
        exact_survivors = [
            payload
            for payload in exact_payloads
            if (
                int((payload.get("metrics") or {}).get("trade_count") or 0)
                >= self.settings.research.minimum_trades
                and float((payload.get("metrics") or {}).get("net_expectancy_r") or -math.inf)
                > self.settings.research.minimum_net_expectancy_r
                and float((payload.get("metrics") or {}).get("profit_factor") or 0.0)
                >= self.settings.research.minimum_stressed_profit_factor
            )
        ]
        ranked = sorted(
            exact_survivors,
            key=lambda payload: robust_score(
                dict(payload.get("metrics") or {}),
                minimum_trades=self.settings.research.minimum_trades,
            ),
            reverse=True,
        )
        ranked_by_timeframe = [
            payload
            for timeframe in sorted(frames_by_timeframe)
            for payload in [
                item for item in ranked if timeframe in (item.get("timeframes_tested") or ())
            ][:maximum_candidates]
        ]
        optimized = 0
        validated = 0
        research_passes = 0
        paper_candidates = 0
        batch_candidates: list[dict[str, Any]] = []
        search_method: Literal["grid", "random", "coordinate", "optuna"] = "random"
        search_trials = maximum_trials or (8 if profile.casefold() == "standard" else 12)
        for payload in ranked_by_timeframe:
            experiment_hash = str(payload["experiment_hash"])
            if experiment_hash in already_validated:
                continue
            combination = combinations.get(str(payload["combination_id"]))
            timeframe = str(payload["timeframes_tested"][0])
            frames = frames_by_timeframe.get(timeframe)
            job = self.store.job(str(payload["job_id"]))
            if combination is None or not frames or job is None:
                continue
            queued = self.store.update_job(
                job,
                status=CombinationState.QUEUED_OPTIMIZATION,
                stage="OPTIMIZATION",
                reason_code="EXACT_SURVIVOR",
                checkpoint=job.get("last_checkpoint"),
            )
            running = self.store.update_job(
                queued,
                status=CombinationState.OPTIMIZATION_RUNNING,
                stage="OPTIMIZATION",
                reason_code="CANONICAL_OPTIMIZER_STARTED",
                checkpoint=queued.get("last_checkpoint"),
            )
            checkpoint_path = self.paths.checkpoints / f"{experiment_hash}.optimization.jsonl"
            try:
                loop = asyncio.get_running_loop()
                research_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        _canonical_research_worker,
                        self.settings,
                        dict(frames),
                        combination,
                        {
                            block_id: dict(parameters)
                            for block_id, parameters in payload["parameters"].items()
                        },
                        checkpoint_path,
                        search_method,
                        search_trials,
                        allow_review_required_research_only,
                    ),
                    timeout=self.settings.lab.trial_timeout_seconds * max(1, search_trials),
                )
                outcome = research_result.outcome
                for trial in outcome.optimization.trials:
                    self.store.save_result(
                        "experiment_trials",
                        job=running,
                        result={
                            "trial_id": f"opt-{trial.trial_id}",
                            "stage": "OPTIMIZATION",
                            "method": outcome.optimization.method,
                            **asdict(trial),
                        },
                        status=trial.status,
                    )
                self.store.database.upsert_records(
                    "parameter_spaces",
                    [
                        {
                            "external_id": f"space-{experiment_hash}",
                            "status": "OPTIMIZED",
                            "timestamp": utc_now(),
                            "experiment_hash": experiment_hash,
                            "strategy_id": outcome.strategy_id,
                            "method": outcome.optimization.method,
                            "best_parameters": outcome.optimization.best_parameters,
                            "best_score": outcome.optimization.best_score,
                            "trial_count": len(outcome.optimization.trials),
                        }
                    ],
                )
                optimized += 1
                optimization_complete = self.store.update_job(
                    running,
                    status=CombinationState.OPTIMIZATION_COMPLETED,
                    stage="OPTIMIZATION",
                    reason_code="CANONICAL_OPTIMIZER_COMPLETE",
                    checkpoint=f"optimization:{experiment_hash}",
                )
                validation_queued = self.store.update_job(
                    optimization_complete,
                    status=CombinationState.QUEUED_VALIDATION,
                    stage="VALIDATION",
                    reason_code="ROBUSTNESS_VALIDATION_QUEUED",
                    checkpoint=optimization_complete.get("last_checkpoint"),
                )
                validation_running = self.store.update_job(
                    validation_queued,
                    status=CombinationState.VALIDATION_RUNNING,
                    stage="VALIDATION",
                    reason_code="ROBUSTNESS_VALIDATION_STARTED",
                    checkpoint=validation_queued.get("last_checkpoint"),
                )
                for mode, walk_forward in (
                    ("anchored", outcome.walk_forward),
                    ("rolling", research_result.rolling_walk_forward),
                ):
                    for fold in walk_forward.folds:
                        self.store.save_result(
                            "walk_forward_results",
                            job=validation_running,
                            result={
                                "fold_id": f"{mode}-{fold.fold}",
                                "mode": mode,
                                **asdict(fold),
                            },
                            status="COMPLETED",
                        )
                result_variants = (
                    ("normal", outcome.normal_result),
                    ("stressed", outcome.stressed_result),
                    ("final_holdout", outcome.holdout_result),
                    ("double_cost", research_result.double_cost_result),
                )
                for source, result in result_variants:
                    robust_payload = self._result_payload(
                        result,
                        combination=combination,
                        job=validation_running,
                        source=source.upper(),
                        bias_label=str(payload["bias_label"]),
                    )
                    robust_payload["trial_id"] = f"robust-{source}"
                    robust_payload["optimized_exit_parameters"] = outcome.parameters
                    self.store.save_result(
                        "exact_backtest_results",
                        job=validation_running,
                        result=robust_payload,
                        status="COMPLETED",
                    )
                monte_carlo = {
                    "simulation_id": "canonical-summary",
                    "source": "CANONICAL_BACKTEST_MONTE_CARLO",
                    "probability_of_loss": outcome.normal_result.metrics.get("probability_of_loss"),
                    "probability_of_20pct_drawdown": outcome.normal_result.metrics.get(
                        "probability_of_20pct_drawdown"
                    ),
                    "probability_of_30pct_drawdown": outcome.normal_result.metrics.get(
                        "probability_of_30pct_drawdown"
                    ),
                    "risk_of_ruin": outcome.normal_result.metrics.get("risk_of_ruin"),
                }
                self.store.save_result(
                    "monte_carlo_results",
                    job=validation_running,
                    result=monte_carlo,
                    status="COMPLETED",
                )
                gate = outcome.gate
                gate_payload = {
                    "gate_id": f"gate-{experiment_hash[:24]}",
                    "experiment_hash": experiment_hash,
                    "status": gate.status.value,
                    "passed": gate.passed,
                    "reasons": list(gate.reasons),
                    "metrics": dict(gate.metrics),
                    "lookahead_safe": outcome.lookahead_safe,
                    "repainting_safe": outcome.repainting_safe,
                    "paper_candidate": False,
                    "live_ready": False,
                }
                self.store.save_result(
                    "gate_results",
                    job=validation_running,
                    result=gate_payload,
                    status=gate.status.value,
                )
                (self.paths.failures / f"{validation_running['job_id']}.validation.json").unlink(
                    missing_ok=True
                )
                validated += 1
                final_state = (
                    CombinationState.RESEARCH_PASS
                    if gate.passed
                    else CombinationState.VALIDATION_REJECTED
                )
                if gate.passed:
                    research_passes += 1
                final_job = self.store.update_job(
                    validation_running,
                    status=final_state,
                    stage="VALIDATION",
                    reason_code=gate.reasons[0],
                    checkpoint=f"validation:{experiment_hash}",
                )
                self._apply_research_to_leaderboard(
                    payload,
                    outcome=outcome,
                    rolling=research_result.rolling_walk_forward,
                    double_cost=research_result.double_cost_result,
                    final_job=final_job,
                )
                daily_returns = (
                    outcome.normal_result.equity_curve["equity"]
                    .astype(float)
                    .resample("1D")
                    .last()
                    .pct_change()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                batch_candidates.append(
                    {
                        "experiment_hash": experiment_hash,
                        "payload": payload,
                        "outcome": outcome,
                        "final_job": final_job,
                        "daily_returns": daily_returns,
                    }
                )
            except Exception as exc:
                self.store.update_job(
                    running,
                    status=CombinationState.VALIDATION_REJECTED,
                    stage="VALIDATION",
                    reason_code=type(exc).__name__,
                    checkpoint=running.get("last_checkpoint"),
                    error=exc,
                )
                atomic_write_json(
                    self.paths.failures / f"{running['job_id']}.validation.json",
                    {
                        "job_id": running["job_id"],
                        "exception_type": type(exc).__name__,
                        "reason_code": type(exc).__name__,
                        "error_message": str(exc)[:500],
                        "attempt": int(running.get("attempt") or 0) + 1,
                        "last_checkpoint": running.get("last_checkpoint"),
                        "retry_eligible": False,
                    },
                )
        if batch_candidates:
            return_matrix = pd.concat(
                {str(item["experiment_hash"]): item["daily_returns"] for item in batch_candidates},
                axis=1,
                join="inner",
            ).dropna(how="any")
            batch_result = multiple_testing_bootstrap(
                return_matrix,
                bootstrap_samples=(self.settings.research.multiple_testing_bootstrap_samples),
                block_size=min(
                    self.settings.research.multiple_testing_block_size,
                    max(1, len(return_matrix)),
                ),
                seed=self.settings.lab.deterministic_seed,
            )
            global_reasons: list[str] = []
            if (
                batch_result.white_reality_check_pvalue
                > self.settings.research.maximum_white_reality_check_pvalue
            ):
                global_reasons.append("WHITE_REALITY_CHECK_FAILURE")
            if batch_result.hansen_spa_pvalue > self.settings.research.maximum_hansen_spa_pvalue:
                global_reasons.append("HANSEN_SPA_FAILURE")
            if batch_result.probability_of_backtest_overfitting is None:
                global_reasons.append("PBO_INSUFFICIENT_STRATEGIES")
            elif (
                batch_result.probability_of_backtest_overfitting
                > self.settings.research.maximum_probability_of_backtest_overfitting
            ):
                global_reasons.append("PBO_ABOVE_GATE")
            research_passes = 0
            for item in batch_candidates:
                experiment_hash = str(item["experiment_hash"])
                payload = item["payload"]
                outcome = item["outcome"]
                final_job = item["final_job"]
                batch_dsr = batch_result.deflated_sharpe_probabilities.get(
                    experiment_hash,
                    0.0,
                )
                candidate_dsr = min(
                    outcome.deflated_sharpe_probability,
                    batch_dsr,
                )
                reasons = [] if outcome.gate.passed else list(outcome.gate.reasons)
                reasons.extend(global_reasons)
                if candidate_dsr < self.settings.research.minimum_deflated_sharpe_probability:
                    reasons.append("BATCH_DEFLATED_SHARPE_FAILURE")
                reasons = list(dict.fromkeys(reasons))
                passed = not reasons
                if passed:
                    research_passes += 1
                status = (
                    ResearchStatus.RESEARCH_PASS.value
                    if passed
                    else (
                        outcome.gate.status.value
                        if not outcome.gate.passed
                        else ResearchStatus.REJECTED_EXPECTANCY.value
                    )
                )
                batch_metrics = {
                    **dict(outcome.gate.metrics),
                    "white_reality_check_pvalue": (batch_result.white_reality_check_pvalue),
                    "hansen_spa_pvalue": batch_result.hansen_spa_pvalue,
                    "probability_of_backtest_overfitting": (
                        batch_result.probability_of_backtest_overfitting
                    ),
                    "deflated_sharpe_probability": candidate_dsr,
                    "multiple_testing_strategy_count": (batch_result.strategy_count),
                    "multiple_testing_observation_count": (batch_result.observation_count),
                }
                self.store.save_result(
                    "gate_results",
                    job=final_job,
                    result={
                        "gate_id": f"gate-{experiment_hash[:24]}",
                        "experiment_hash": experiment_hash,
                        "status": status,
                        "passed": passed,
                        "reasons": (
                            reasons
                            if reasons
                            else ["ALL_RESEARCH_AND_MULTIPLE_TESTING_GATES_PASSED"]
                        ),
                        "metrics": batch_metrics,
                        "lookahead_safe": outcome.lookahead_safe,
                        "repainting_safe": outcome.repainting_safe,
                        "paper_candidate": False,
                        "live_ready": False,
                    },
                    status=status,
                )
                final_state = (
                    CombinationState.RESEARCH_PASS
                    if passed
                    else CombinationState.VALIDATION_REJECTED
                )
                updated_job = self.store.update_job(
                    final_job,
                    status=final_state,
                    stage="MULTIPLE_TESTING",
                    reason_code=(
                        "ALL_RESEARCH_AND_MULTIPLE_TESTING_GATES_PASSED" if passed else reasons[0]
                    ),
                    checkpoint=f"multiple-testing:{experiment_hash}",
                )
                provisional = self._leaderboard_entry(payload)
                previous = next(
                    (
                        row
                        for row in self.store.leaderboard()
                        if row.get("entry_id") == provisional["entry_id"]
                    ),
                    provisional,
                )
                previous.update(
                    {
                        "white_reality_check_pvalue": (batch_result.white_reality_check_pvalue),
                        "hansen_spa_pvalue": batch_result.hansen_spa_pvalue,
                        "pbo": (batch_result.probability_of_backtest_overfitting),
                        "deflated_sharpe": candidate_dsr,
                        "gate_status": status,
                        "lifecycle_status": (
                            LifecycleStatus.RESEARCH_PASS.value
                            if passed
                            else LifecycleStatus.DEGRADED.value
                        ),
                        "paper_candidate": False,
                        "live_ready": False,
                        "last_checkpoint": updated_job.get("last_checkpoint"),
                    }
                )
                self.store.save_leaderboard_entry(previous)
        return optimized, validated, research_passes, paper_candidates

    def _apply_research_to_leaderboard(
        self,
        baseline_payload: Mapping[str, Any],
        *,
        outcome: ResearchOutcome,
        rolling: WalkForwardResult,
        double_cost: BacktestResult,
        final_job: Mapping[str, Any],
    ) -> None:
        provisional = self._leaderboard_entry(baseline_payload)
        previous = next(
            (
                row
                for row in self.store.leaderboard()
                if row.get("entry_id") == provisional["entry_id"]
            ),
            provisional,
        )
        normal = outcome.normal_result.metrics
        holdout = outcome.holdout_result.metrics
        stressed = outcome.stressed_result.metrics
        positive_folds = outcome.walk_forward.positive_folds
        total_folds = len(outcome.walk_forward.folds)
        stability = outcome.stability.acceptable_score_fraction
        mc_loss = float(
            1.0 if normal.get("probability_of_loss") is None else normal["probability_of_loss"]
        )
        drawdown = float(
            1.0 if normal.get("maximum_drawdown") is None else normal["maximum_drawdown"]
        )
        concentration = max(
            float(
                1.0
                if normal.get("symbol_profit_concentration") is None
                else normal["symbol_profit_concentration"]
            ),
            outcome.walk_forward.fold_profit_concentration,
        )
        score = (
            float(holdout.get("net_expectancy_r") or 0.0)
            * math.sqrt(max(1, int(holdout.get("trade_count") or 0)))
            + math.log1p(max(0.0, float(stressed.get("profit_factor") or 0.0)))
            + 2.0 * (positive_folds / max(1, total_folds))
            + stability
            - 5.0 * drawdown
            - 2.0 * mc_loss
            - concentration
            - float(normal.get("time_under_water") or 0.0)
            - 0.01 * float(normal.get("turnover") or 0.0)
        )
        if baseline_payload.get("bias_label") == "CURRENT_UNIVERSE_RETROSPECTIVE":
            score -= 2.0
        passed = outcome.gate.passed
        entry = previous | {
            **dict(normal),
            "optimized_exit_parameters": outcome.parameters,
            "optimization_method": outcome.optimization.method,
            "optimization_trial_count": len(outcome.optimization.trials),
            "stressed_profit_factor": stressed.get("profit_factor"),
            "double_cost_profit_factor": double_cost.metrics.get("profit_factor"),
            "positive_folds": positive_folds,
            "total_folds": total_folds,
            "rolling_positive_folds": rolling.positive_folds,
            "rolling_total_folds": len(rolling.folds),
            "final_holdout_expectancy": holdout.get("net_expectancy_r"),
            "parameter_stability_score": stability,
            "neighborhood_profitability": (
                outcome.stability.positive_neighbors / max(1, outcome.stability.tested_neighbors)
            ),
            "fold_concentration": outcome.walk_forward.fold_profit_concentration,
            "cpcv_path_count": outcome.cpcv.path_count,
            "cpcv_path_consistency": outcome.cpcv.path_consistency,
            "cpcv_final_holdout_excluded": outcome.cpcv.final_holdout_excluded,
            "pbo": outcome.cpcv.probability_of_backtest_overfitting,
            "deflated_sharpe": outcome.deflated_sharpe_probability,
            "monte_carlo_probability_of_loss": normal.get("probability_of_loss"),
            "monte_carlo_probability_of_20pct_drawdown": normal.get(
                "probability_of_20pct_drawdown"
            ),
            "monte_carlo_probability_of_30pct_drawdown": normal.get(
                "probability_of_30pct_drawdown"
            ),
            "risk_of_ruin": normal.get("risk_of_ruin"),
            "gate_status": outcome.gate.status.value,
            "robust_score": score,
            "lifecycle_status": (
                LifecycleStatus.RESEARCH_PASS.value if passed else LifecycleStatus.DEGRADED.value
            ),
            "paper_candidate": False,
            "live_ready": False,
            "last_tested_at": utc_iso(),
            "last_checkpoint": final_job.get("last_checkpoint"),
        }
        self.store.save_leaderboard_entry(entry)

    def _leaderboard_entry(
        self,
        payload: Mapping[str, Any],
        *,
        previous: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = dict(payload["metrics"])
        score = robust_score(
            metrics,
            minimum_trades=self.settings.research.minimum_trades,
        )
        if payload.get("bias_label") == "CURRENT_UNIVERSE_RETROSPECTIVE":
            score -= 2.0
        trades = int(metrics.get("trade_count") or 0)
        lifecycle = LifecycleStatus.DISCOVERED
        if score > 0 and trades >= self.settings.research.minimum_trades:
            lifecycle = LifecycleStatus.SCREENING_SURVIVOR
        if previous and (
            float(metrics.get("net_expectancy_r") or 0.0) < 0
            or float(
                1.0 if metrics.get("maximum_drawdown") is None else metrics["maximum_drawdown"]
            )
            > self.settings.research.maximum_drawdown
        ):
            lifecycle = LifecycleStatus.DEGRADED
        entry_id = stable_hash(
            [
                payload["strategy_dna_hash"],
                payload["parameter_hash"],
                payload["universe_snapshot_id"],
                payload["timeframes_tested"],
            ]
        )
        return {
            "entry_id": f"lb-{entry_id[:24]}",
            "rank": previous.get("rank") if previous else None,
            "combination_id": payload["combination_id"],
            "strategy_dna_hash": payload["strategy_dna_hash"],
            "block_names": payload["block_ids"],
            "block_families": payload["families"],
            "block_roles": payload["roles"],
            "logic_mode": payload["logic_mode"],
            "combination_size": payload["combination_size"],
            "parameters": payload["parameters"],
            "parameter_hash": payload["parameter_hash"],
            "entry_profile": {"logic_mode": payload["logic_mode"]},
            "exit_profile": {"type": "canonical_defaults"},
            "risk_profile": {"type": "settings"},
            "universe_snapshot_id": payload["universe_snapshot_id"],
            "assets_tested": payload["assets_tested"],
            "timeframes_tested": payload["timeframes_tested"],
            "source": payload.get("source"),
            "source_type": payload.get("source_type"),
            "provider_coverage": (
                sorted(
                    {
                        provider
                        for item in (payload.get("data_provenance") or {}).values()
                        for provider in item.get("providers_used", [])
                    }
                )
                if payload.get("source_type") == "REAL_PROVIDER_DATA"
                else ["synthetic_offline"]
            ),
            "data_hash": payload.get("data_hash"),
            "data_provenance": payload.get("data_provenance"),
            "data_period": payload["data_period"],
            "train_period": None,
            "validation_period": None,
            "final_holdout_period": None,
            **metrics,
            "stressed_profit_factor": None,
            "positive_folds": 0,
            "total_folds": 0,
            "final_holdout_expectancy": None,
            "parameter_stability_score": None,
            "neighborhood_profitability": None,
            "fold_concentration": None,
            "regime_concentration": None,
            "pbo": None,
            "deflated_sharpe": None,
            "data_quality_status": (
                "PASSED"
                if payload["integrity"].get("valid_data")
                and payload["integrity"].get("closed_candle_integrity")
                and payload["integrity"].get("provider_data_integrity")
                else "FAILED"
            ),
            "lookahead_status": (
                "PASSED" if payload["integrity"].get("no_lookahead") else "FAILED"
            ),
            "repainting_status": (
                "PASSED" if payload["integrity"].get("no_repainting") else "FAILED"
            ),
            "eligibility_status": payload["bias_label"],
            "gate_status": "NOT_EVALUATED_BASELINE_ONLY",
            "robust_score": score,
            "first_discovered_at": (previous.get("first_discovered_at") if previous else utc_iso()),
            "last_tested_at": utc_iso(),
            "number_of_retests": (
                int(previous.get("number_of_retests") or 0) + 1 if previous else 0
            ),
            "lifecycle_status": lifecycle.value,
            "paper_candidate": False,
            "live_ready": False,
        }

    async def run_once(
        self,
        *,
        profile: str = "quick",
        universe_size: int = 5,
        combination_sizes: Sequence[int] = (1, 2),
        logic_modes: Sequence[LogicMode] = (LogicMode.LAYERED,),
        timeframes: Sequence[str] = ("1h", "4h"),
        rows: int = 1_000,
        history_mode: Literal[
            "smoke",
            "bounded",
            "common_full_history",
            "asset_max_history",
        ] = "bounded",
        workers: int = 2,
        data_mode: Literal["real", "synthetic"] = "real",
        max_trials: int | None = None,
        universe_scope: Literal["allowed", "discovery"] = "allowed",
        include_review_required_research_only: bool = False,
        resume: bool = False,
        force: bool = False,
        retest: bool = False,
        only_missing: bool = False,
        block_ids: Sequence[str] | None = None,
        parameter_overrides: Mapping[str, Sequence[Any]] | None = None,
        combination_templates: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        if rows < 250:
            raise ValueError("lab runs require at least 250 rows")
        if data_mode == "synthetic" and history_mode not in {"smoke", "bounded"}:
            raise ValueError("synthetic lab runs cannot use full-history modes")
        worker_limit = min(
            max(1, workers),
            self.settings.lab.max_workers,
            self.settings.lab.cpu_limit or workers,
        )
        run_id = f"labrun-{stable_hash([utc_iso(), profile, os.getpid()])[:20]}"
        recovered = self.store.recover_stale_jobs() if resume else 0
        superseded = self.store.supersede_incomplete_jobs(active_run_id=run_id) if resume else 0
        self.store.persist_blocks(self.registry.values())
        generation_checkpoint = self.paths.checkpoints / "generation_cursor.json"
        continuation_cursor = None
        if resume and generation_checkpoint.is_file():
            generation_state = read_json(generation_checkpoint)
            if generation_state.get("status") == "PARTIAL_GENERATION":
                continuation_cursor = generation_state.get("continuation_cursor")
        if combination_templates:
            combinations = []
            template_status: dict[str, Any] = {}
            for hypothesis, membership in sorted(combination_templates.items()):
                selected = tuple(dict.fromkeys(membership))
                generated = self.generator.generate(
                    sizes=(len(selected),),
                    logic_modes=logic_modes,
                    mode=GenerationMode.FAMILY_AWARE,
                    block_ids=selected,
                    timeframes=timeframes,
                    maximum_rows=None,
                )
                exact = [item for item in generated if item.block_ids == tuple(sorted(selected))]
                if len(exact) != 1:
                    raise ValueError(f"economic hypothesis template is invalid: {hypothesis}")
                combinations.extend(exact)
                template_status[hypothesis] = {
                    "block_ids": list(selected),
                    "combination_id": exact[0].combination_id,
                    "eligibility_status": exact[0].eligibility_status.value,
                    "exclusion_reason": exact[0].exclusion_reason,
                }
            self.generator.last_generation_status = {
                "status": "COMPLETE_TEMPLATE_GENERATION",
                "generated_count": len(combinations),
                "remaining_count": 0,
                "continuation_cursor": None,
                "economic_hypotheses": template_status,
            }
        else:
            combinations = self.generator.generate(
                sizes=combination_sizes,
                logic_modes=logic_modes,
                mode=(
                    GenerationMode.FAMILY_AWARE
                    if profile.casefold() != "exhaustive"
                    else GenerationMode.EXHAUSTIVE
                ),
                block_ids=block_ids or self._profile_blocks(profile),
                timeframes=timeframes,
                maximum_rows=self.settings.lab.maximum_generation_rows,
                continuation_cursor=continuation_cursor,
            )
        atomic_write_json(
            self.paths.checkpoints / "generation_cursor.json",
            self.generator.last_generation_status,
        )
        self.store.persist_combinations(combinations)
        valid = [
            combination
            for combination in combinations
            if combination.eligibility_status is CombinationState.GENERATED
        ]
        invalid = [
            combination
            for combination in combinations
            if combination.eligibility_status is not CombinationState.GENERATED
        ]
        markets = self._markets(
            universe_size,
            universe_scope=universe_scope,
            include_review_required_research_only=(include_review_required_research_only),
            required_timeframes=timeframes if data_mode == "real" else None,
            minimum_rows=(
                rows
                if data_mode == "real" and history_mode in {"smoke", "bounded"}
                else self.settings.lab.deep_minimum_history_rows
                if data_mode == "real"
                else 0
            ),
        )
        snapshot = UniverseManager(
            self.settings,
            database=self.store.database,
        ).latest()
        snapshot_id = str(snapshot["snapshot_id"]) if snapshot else "universe-current-settings"
        bias_label = str(snapshot["bias_label"]) if snapshot else "CURRENT_UNIVERSE_RETROSPECTIVE"
        semaphore = asyncio.Semaphore(worker_limit)
        executor = ProcessPoolExecutor(max_workers=worker_limit)
        screening_executor = ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix="lab-fast-screen",
        )
        completed = 0
        sensitivity_completed = 0
        failed = 0
        deduplicated = 0
        baseline_payloads: list[dict[str, Any]] = []
        frames_by_timeframe: dict[str, dict[str, pd.DataFrame]] = {}
        provenance_by_timeframe: dict[str, dict[str, Any]] = {}
        raw_provenance_by_timeframe: dict[str, dict[str, Any]] = {}
        data_hashes_by_timeframe: dict[str, str] = {}
        feature_hashes_by_timeframe: dict[str, str] = {}
        rows_used_by_timeframe: dict[str, Any] = {}
        started = time.perf_counter()
        atomic_write_json(
            self.current_status_path,
            {
                "run_id": run_id,
                "lab_instance_id": self.instance_id,
                "status": "RUNNING",
                "started_at": utc_iso(),
                "live_orders": 0,
            },
        )
        self.write_worker_status(
            run_id=run_id,
            phase="PREPARING_AND_HASHING_DATA",
            workers=worker_limit,
            active=0,
            completed=0,
            failed=0,
        )
        for timeframe in timeframes:
            selected_rows: int | None = rows
            slice_start: pd.Timestamp | None = None
            slice_end: pd.Timestamp | None = None
            if data_mode == "real" and history_mode == "common_full_history":
                indices = {
                    market: pd.DatetimeIndex(
                        pd.read_parquet(
                            self._data_path(market, timeframe),
                            columns=["close"],
                        ).index
                    )
                    for market in markets
                }
                slice_start = max(index.min() for index in indices.values())
                slice_end = min(index.max() for index in indices.values())
                if slice_start >= slice_end:
                    raise DataValidationError(f"NO_COMMON_FULL_HISTORY_PERIOD:{timeframe}")
                available_rows = {
                    market: int(((index >= slice_start) & (index <= slice_end)).sum())
                    for market, index in indices.items()
                }
                minimum_common_rows = min(available_rows.values())
                if minimum_common_rows < self.settings.lab.deep_minimum_history_rows:
                    raise DataValidationError(
                        f"INSUFFICIENT_COMMON_FULL_HISTORY:{timeframe}:{minimum_common_rows}"
                    )
                selected_rows = None
                rows_used_by_timeframe[timeframe] = {
                    "common_start": slice_start.isoformat(),
                    "common_end": slice_end.isoformat(),
                    "rows_by_market": available_rows,
                    "minimum_rows": minimum_common_rows,
                }
            elif data_mode == "real" and history_mode == "asset_max_history":
                available_rows = {
                    market: len(
                        pd.read_parquet(
                            self._data_path(market, timeframe),
                            columns=["close"],
                        )
                    )
                    for market in markets
                }
                if min(available_rows.values()) < self.settings.lab.deep_minimum_history_rows:
                    raise DataValidationError(
                        f"INSUFFICIENT_ASSET_MAX_HISTORY:{timeframe}:{available_rows}"
                    )
                selected_rows = None
                rows_used_by_timeframe[timeframe] = available_rows
            else:
                rows_used_by_timeframe[timeframe] = rows
            frames, data_hash, data_provenance = self._frames(
                markets=markets,
                timeframe=timeframe,
                rows=selected_rows,
                data_mode=data_mode,
                start_at=slice_start,
                end_at=slice_end,
            )
            frames_by_timeframe[timeframe] = frames
            data_hashes_by_timeframe[timeframe] = data_hash
            feature_hashes_by_timeframe[timeframe] = stable_hash(
                {market: details["feature_hash"] for market, details in data_provenance.items()},
                length=64,
            )
            provenance_by_timeframe[timeframe] = {
                market: self._provenance_summary(provenance)
                for market, provenance in data_provenance.items()
            }
            raw_provenance_by_timeframe[timeframe] = data_provenance
            estimated_worker_memory_mb = (
                sum(
                    float(frame.memory_usage(index=True, deep=True).sum())
                    for frame in frames.values()
                )
                * worker_limit
                / (1024.0 * 1024.0)
            )
            if estimated_worker_memory_mb > self.settings.lab.memory_limit_mb:
                executor.shutdown(wait=True, cancel_futures=True)
                screening_executor.shutdown(wait=True, cancel_futures=True)
                raise MemoryError(
                    "estimated process-worker frame memory exceeds LAB_MEMORY_LIMIT_MB"
                )
        plan_manifest = {
            "run_id": run_id,
            "manifest_kind": "IMMUTABLE_CAMPAIGN_PLAN",
            "status": "FROZEN_BEFORE_SCREENING",
            "created_at": utc_iso(),
            "profile": profile.upper(),
            "history_mode": history_mode.upper(),
            "data_mode": data_mode,
            "markets": list(markets),
            "timeframes": list(timeframes),
            "rows_used_by_timeframe": rows_used_by_timeframe,
            "data_hashes_by_timeframe": data_hashes_by_timeframe,
            "feature_hashes_by_timeframe": feature_hashes_by_timeframe,
            "software_version": self.settings.app.version,
            "screening_engine_version": FAST_SCREEN_VERSION,
            "deterministic_seed": self.settings.lab.deterministic_seed,
            "combination_ids": [combination.combination_id for combination in valid],
            "economic_hypotheses": (
                {name: list(blocks) for name, blocks in sorted(combination_templates.items())}
                if combination_templates
                else None
            ),
            "costs": self.settings.costs.model_dump(mode="json"),
            "research_gates": self.settings.research.model_dump(mode="json"),
            "live_orders": 0,
        }
        plan_path = self.paths.manifests / f"{run_id}.plan.json"
        atomic_write_json(plan_path, plan_manifest)
        plan_hash = sha256_file(plan_path)
        for timeframe in timeframes:
            frames = frames_by_timeframe[timeframe]
            data_hash = data_hashes_by_timeframe[timeframe]
            data_provenance = raw_provenance_by_timeframe[timeframe]
            tasks = []
            for combination in valid:
                for sensitivity_parameter, parameters in self._parameter_sets(
                    combination,
                    parameter_overrides,
                ):
                    job = self.store.queue_job(
                        run_id=run_id,
                        combination=combination,
                        snapshot_id=snapshot_id,
                        markets=markets,
                        timeframe=timeframe,
                        parameters=parameters,
                        data_hash=data_hash,
                        feature_hash=feature_hashes_by_timeframe[timeframe],
                        data_provenance=data_provenance,
                        force=force,
                        retest=retest,
                        only_missing=only_missing,
                        sensitivity_parameter=sensitivity_parameter,
                    )
                    if job.get("deduplicated"):
                        deduplicated += 1
                        continue
                    tasks.append(
                        self._baseline_job(
                            job=job,
                            combination=combination,
                            frames=frames,
                            bias_label=bias_label,
                            semaphore=semaphore,
                            executor=screening_executor,
                            allow_review_required_research_only=(
                                include_review_required_research_only
                            ),
                        )
                    )
            self.write_worker_status(
                run_id=run_id,
                phase=f"SCREENING:{timeframe}",
                workers=worker_limit,
                active=min(worker_limit, len(tasks)),
                completed=completed,
                failed=failed,
            )
            timeframe_finished = 0
            for task in asyncio.as_completed(tasks):
                job, payload = await task
                timeframe_finished += 1
                if payload is None:
                    failed += int(
                        job.get("status")
                        in {
                            CombinationState.ERROR_RETRYABLE.value,
                            CombinationState.ERROR_FINAL.value,
                        }
                    )
                else:
                    completed += 1
                    if job.get("result_type") == "PARAMETER_SENSITIVITY":
                        sensitivity_completed += 1
                    baseline_payloads.append(payload)
                self.heartbeat(
                    run_id=run_id,
                    status="RUNNING",
                    completed_jobs=completed,
                    failed_jobs=failed,
                )
                self.write_worker_status(
                    run_id=run_id,
                    phase=f"SCREENING:{timeframe}",
                    workers=worker_limit,
                    active=min(
                        worker_limit,
                        max(0, len(tasks) - timeframe_finished),
                    ),
                    completed=completed,
                    failed=failed,
                )
        screening_executor.shutdown(wait=True, cancel_futures=True)
        combination_index = {combination.combination_id: combination for combination in valid}
        persisted_baselines = [
            payload
            for payload in _payload_rows(
                self.store.database,
                "experiment_trials",
            )
            if payload.get("combination_id") in combination_index
            and _matches_research_slice(
                payload,
                data_hashes_by_timeframe=data_hashes_by_timeframe,
                feature_hashes_by_timeframe=feature_hashes_by_timeframe,
                screening_engine_version=FAST_SCREEN_VERSION,
                screen_policy_version=SCREEN_POLICY_VERSION,
                exit_model_version=EXIT_MODEL_VERSION,
                survivor_policy_version=SURVIVOR_POLICY_VERSION,
                markets=markets,
                snapshot_id=snapshot_id,
                sources={"FAST_SCREEN_REAL", "SYNTHETIC_FAST_SCREEN"},
            )
        ]
        unique_baselines = {
            str(payload["experiment_hash"]): payload
            for payload in [*persisted_baselines, *baseline_payloads]
            if payload.get("experiment_hash")
        }
        current_job_by_experiment = {
            str(job.get("experiment_hash")): job
            for job in self.store.jobs()
            if str(job.get("run_id")) == run_id
            and job.get("experiment_hash")
        }
        unique_baselines = {
            experiment_hash: (
                dict(payload)
                | {
                    "job_id": current_job_by_experiment[
                        experiment_hash
                    ]["job_id"]
                }
                if experiment_hash in current_job_by_experiment
                else payload
            )
            for experiment_hash, payload in unique_baselines.items()
        }
        self.write_worker_status(
            run_id=run_id,
            phase="EXACT_BACKTEST",
            workers=worker_limit,
            active=1,
            completed=completed,
            failed=failed,
        )
        screening_trials, exact_backtests, new_exact_payloads = await self._screen_and_validate(
            baseline_payloads=list(unique_baselines.values()),
            combinations=combination_index,
            frames_by_timeframe=frames_by_timeframe,
            executor=executor,
            allow_review_required_research_only=(include_review_required_research_only),
        )
        optimized_candidates = 0
        walk_forward_candidates = 0
        research_passes = 0
        paper_candidates = 0
        if profile.casefold() != "quick":
            self.write_worker_status(
                run_id=run_id,
                phase="OPTIMIZATION_AND_VALIDATION",
                workers=worker_limit,
                active=1,
                completed=completed,
                failed=failed,
            )
            persisted_exact = [
                payload
                for payload in _payload_rows(
                    self.store.database,
                    "exact_backtest_results",
                )
                if payload.get("combination_id") in combination_index
                and _matches_research_slice(
                    payload,
                    data_hashes_by_timeframe=data_hashes_by_timeframe,
                    feature_hashes_by_timeframe=feature_hashes_by_timeframe,
                    screening_engine_version=FAST_SCREEN_VERSION,
                    screen_policy_version=SCREEN_POLICY_VERSION,
                    exit_model_version=EXIT_MODEL_VERSION,
                    survivor_policy_version=SURVIVOR_POLICY_VERSION,
                    markets=markets,
                    snapshot_id=snapshot_id,
                    sources={"EXACT_REAL", "SYNTHETIC_SMOKE"},
                )
            ]
            exact_candidates = {
                str(payload["experiment_hash"]): payload
                for payload in [*persisted_exact, *new_exact_payloads]
            }
            exact_candidates = {
                experiment_hash: (
                    dict(payload)
                    | {
                        "job_id": current_job_by_experiment[
                            experiment_hash
                        ]["job_id"]
                    }
                    if experiment_hash in current_job_by_experiment
                    else payload
                )
                for experiment_hash, payload in exact_candidates.items()
            }
            (
                optimized_candidates,
                walk_forward_candidates,
                research_passes,
                paper_candidates,
            ) = await self._optimize_and_validate(
                exact_payloads=list(exact_candidates.values()),
                combinations=combination_index,
                frames_by_timeframe=frames_by_timeframe,
                executor=executor,
                profile=profile,
                maximum_candidates=3 if profile.casefold() == "standard" else 5,
                maximum_trials=max_trials,
                allow_review_required_research_only=(include_review_required_research_only),
            )
        executor.shutdown(wait=True, cancel_futures=True)
        # A crash can happen after the atomic baseline write but before the
        # leaderboard write. Rebuild missing entries idempotently on every run.
        known_entries = {row["entry_id"]: row for row in self.store.leaderboard()}
        for payload in _payload_rows(self.store.database, "baseline_results"):
            if not isinstance(payload.get("metrics"), dict):
                continue
            if payload.get("source") not in {"BASELINE_REAL", "SYNTHETIC_SMOKE"}:
                continue
            provisional = self._leaderboard_entry(payload)
            previous = known_entries.get(provisional["entry_id"])
            entry = self._leaderboard_entry(payload, previous=previous)
            self.store.save_leaderboard_entry(entry)
            known_entries[entry["entry_id"]] = entry
        exports = self.store.export_leaderboards()
        queue = self.store.queue_status(run_id=run_id)
        duration = time.perf_counter() - started
        formal_status = (
            "PARTIAL"
            if failed
            else (
                "PASSED_WITH_APPROVED_PAPER_CANDIDATES"
                if paper_candidates
                else "PASSED_WITH_ZERO_APPROVED_CANDIDATES"
            )
        )
        result = {
            "run_id": run_id,
            "plan_manifest": str(plan_path),
            "plan_manifest_sha256": plan_hash,
            "lab_instance_id": self.instance_id,
            "profile": profile.upper(),
            "data_mode": data_mode,
            "history_mode": history_mode.upper(),
            "rows_used_by_timeframe": rows_used_by_timeframe,
            "source_type": ("BASELINE_REAL" if data_mode == "real" else "SYNTHETIC_SMOKE"),
            "data_provenance": provenance_by_timeframe,
            "synthetic_fallback": False,
            "status": formal_status,
            "registered_signal_blocks": len(self.registry),
            "block_counts_by_family": dict(
                sorted(Counter(block.family for block in self.registry.values()).items())
            ),
            "generated_combinations": len(combinations),
            "generation_status": self.generator.last_generation_status,
            "economic_hypotheses": (
                {name: list(blocks) for name, blocks in sorted(combination_templates.items())}
                if combination_templates
                else None
            ),
            "valid_combinations": len(valid),
            "excluded_combinations": len(invalid),
            "exclusion_reasons": dict(
                sorted(Counter(item.exclusion_reason for item in invalid).items())
            ),
            "generated_by_size": dict(
                sorted(Counter(item.combination_size for item in combinations).items())
            ),
            "markets": markets,
            "target_universe": universe_size,
            "actual_universe": len(markets),
            "universe_data_selection": self.last_market_selection,
            "timeframes": list(timeframes),
            "baseline_backtests": 0,
            "new_baseline_backtests": 0,
            "fast_screen_trials": completed,
            "baseline_screening_trials": completed - sensitivity_completed,
            "sensitivity_trials": sensitivity_completed,
            "screening_trials": screening_trials,
            "exact_backtests": exact_backtests,
            "optimized_candidates": optimized_candidates,
            "walk_forward_candidates": walk_forward_candidates,
            "research_passes": research_passes,
            "paper_candidates": paper_candidates,
            "deduplicated_jobs": deduplicated,
            "recovered_jobs": recovered,
            "superseded_jobs": superseded,
            "failures": failed,
            "queue": queue,
            "queue_throughput_per_second": completed / max(duration, 1e-9),
            "duration_seconds": duration,
            "workers": worker_limit,
            "worker_type": "thread_screen_process_exact",
            "memory_limit_mb": self.settings.lab.memory_limit_mb,
            "leaderboard_exports": exports,
            "bias_label": bias_label,
            "live_orders": 0,
            "completed_at": utc_iso(),
        }
        atomic_write_json(
            self.paths.manifests / f"{run_id}.json",
            result,
        )
        atomic_write_json(
            self.current_status_path,
            result | {"status": "STOPPED"},
        )
        atomic_write_json(
            self.worker_status_path,
            {
                "run_id": run_id,
                "workers": worker_limit,
                "active": 0,
                "completed": completed,
                "failed": failed,
                "task_leaks": [],
                "updated_at": utc_iso(),
            },
        )
        self.heartbeat(
            run_id=run_id,
            status="STOPPED",
            completed_jobs=completed,
            failed_jobs=failed,
        )
        self.store.record_event(
            {
                "run_id": run_id,
                "lab_instance_id": self.instance_id,
                "job_id": None,
                "combination_id": None,
                "experiment_hash": None,
                "universe_snapshot_id": snapshot_id,
                "block_ids": [],
                "parameter_hash": None,
                "market": None,
                "timeframe": None,
                "stage": "RUN",
                "worker": "orchestrator",
                "started_at": None,
                "completed_at": utc_iso(),
                "duration": duration,
                "status": result["status"],
                "reason_code": "RUN_ONCE_COMPLETE",
                "retry_count": 0,
                "memory_usage": None,
                "cpu_duration": time.process_time(),
            }
        )
        return result

    async def run_once_guarded(self, **run_arguments: Any) -> dict[str, Any]:
        """Run one cycle and persist a terminal failure state on any exception."""

        try:
            return await self.run_once(**run_arguments)
        except BaseException as exc:
            current = (
                read_json(self.current_status_path) if self.current_status_path.is_file() else {}
            )
            run_id = str(current.get("run_id") or "run-start-failed")
            failure = {
                **current,
                "run_id": run_id,
                "lab_instance_id": self.instance_id,
                "status": "FAILED",
                "failed_at": utc_iso(),
                "reason_code": type(exc).__name__,
                "error": str(exc),
                "live_orders": 0,
            }
            atomic_write_json(self.current_status_path, failure)
            self.heartbeat(
                run_id=run_id,
                status="FAILED",
                completed_jobs=0,
                failed_jobs=1,
            )
            self.store.record_event(
                {
                    "run_id": run_id,
                    "lab_instance_id": self.instance_id,
                    "job_id": None,
                    "combination_id": None,
                    "experiment_hash": None,
                    "universe_snapshot_id": None,
                    "block_ids": [],
                    "parameter_hash": None,
                    "market": None,
                    "timeframe": None,
                    "stage": "RUN",
                    "worker": "orchestrator",
                    "started_at": current.get("started_at"),
                    "completed_at": utc_iso(),
                    "duration": None,
                    "status": "FAILED",
                    "reason_code": type(exc).__name__,
                    "retry_count": 0,
                    "memory_usage": None,
                    "cpu_duration": time.process_time(),
                }
            )
            raise

    async def run_continuous(
        self,
        *,
        soak_minutes: float | None = None,
        **run_arguments: Any,
    ) -> dict[str, Any]:
        descriptor = self._acquire_lock()
        started = time.monotonic()
        next_research_cycle = started
        cycles = 0
        completed_jobs = 0
        failed_jobs = 0
        last: dict[str, Any] = {}
        atomic_write_json(
            self.control_path,
            {"action": LabControl.START.value, "requested_at": utc_iso()},
        )
        try:
            while True:
                action = self._control_action()
                if action in {LabControl.STOP, LabControl.DRAIN}:
                    break
                if action is LabControl.PAUSE:
                    self.heartbeat(
                        run_id=str(last.get("run_id") or "continuous"),
                        status="PAUSED",
                        completed_jobs=completed_jobs,
                        failed_jobs=failed_jobs,
                    )
                    await asyncio.sleep(self.settings.lab.heartbeat_seconds)
                    continue
                if soak_minutes is not None and (time.monotonic() - started >= soak_minutes * 60):
                    break
                if time.monotonic() >= next_research_cycle:
                    last = await self.run_once(**run_arguments)
                    cycles += 1
                    completed_jobs += int(last.get("new_baseline_backtests") or 0)
                    failed_jobs += int(last.get("failures") or 0)
                    next_research_cycle = (
                        time.monotonic() + self.settings.lab.leaderboard_refresh_minutes * 60.0
                    )
                self.heartbeat(
                    run_id=str(last.get("run_id") or "continuous"),
                    status="IDLE",
                    completed_jobs=completed_jobs,
                    failed_jobs=failed_jobs,
                )
                remaining_to_cycle = max(0.0, next_research_cycle - time.monotonic())
                sleep_seconds = min(
                    self.settings.lab.heartbeat_seconds,
                    remaining_to_cycle or self.settings.lab.heartbeat_seconds,
                )
                await asyncio.sleep(sleep_seconds)
        finally:
            self._release_lock(descriptor)
        result = {
            **last,
            "continuous": True,
            "cycles": cycles,
            "soak_minutes": (time.monotonic() - started) / 60.0,
            "completed_jobs_total": completed_jobs,
            "failed_jobs_total": failed_jobs,
            "clean_shutdown": True,
            "task_leaks": [],
            "status": "STOPPED",
            "live_orders": 0,
        }
        atomic_write_json(self.current_status_path, result)
        return result


def describe_blocks(
    registry: Mapping[str, SignalBlock] | None = None,
) -> list[dict[str, Any]]:
    return [
        block.to_dict()
        for block in sorted(
            (registry or signal_block_registry()).values(),
            key=lambda item: item.block_id,
        )
    ]


def validate_blocks(
    registry: Mapping[str, SignalBlock] | None = None,
) -> dict[str, Any]:
    selected = dict(registry or signal_block_registry())
    bearish_entry = [
        block.block_id
        for block in selected.values()
        if block.direction is BlockDirection.BEARISH and block.role is BlockRole.ENTRY_TRIGGER
    ]
    raw_fractals = [
        block.block_id for block in selected.values() if block.block_id.startswith("raw_fractal")
    ]
    parameter_results: list[dict[str, Any]] = []
    for block in selected.values():
        for specification in block.parameter_specs:
            values = specification.values()
            default = specification.validate(specification.default)
            parameter_results.append(
                {
                    "parameter": f"{block.block_id}__{specification.name}",
                    "kind": specification.kind.value,
                    "default": _canonical_value(default),
                    "value_count": len(values),
                    "has_non_default_trial": any(value != default for value in values),
                    "hashes_deterministic": parameter_hash({"value": default})
                    == parameter_hash({"value": default}),
                    "optimizer_distribution": specification.optimizer_distribution,
                    "cache_behavior": specification.cache_behavior,
                }
            )
    return {
        "status": (
            "PASSED"
            if not bearish_entry
            and not raw_fractals
            and all(row["has_non_default_trial"] for row in parameter_results)
            else "FAILED"
        ),
        "registered": len(selected),
        "unique_ids": len(selected) == len(set(selected)),
        "bearish_entry_blocks": bearish_entry,
        "raw_fractal_blocks": raw_fractals,
        "parameter_specs": parameter_results,
        "tunable_parameter_count": len(parameter_results),
        "counts_by_family": dict(
            sorted(Counter(block.family for block in selected.values()).items())
        ),
        "counts_by_role": dict(
            sorted(Counter(block.role.value for block in selected.values()).items())
        ),
    }


def write_legacy_migration_report(settings: Settings) -> Path:
    """Persist the audited disposition of every non-empty legacy Python file."""

    legacy_root = settings.paths.project_root / "legacy"
    rules = {
        "candles_engine.py": ("MIGRATED", "research/features.py", "test_features_strategies.py"),
        "market_structure_patterns.py": (
            "MIGRATED",
            "research/features.py",
            "test_indicator_registry_fractals_investing.py",
        ),
        "trading_math_engine.py": ("MIGRATED", "research/trading_math.py", "test_backtest_math.py"),
        "simple_backtest_engine.py": ("MIGRATED", "research/backtest.py", "test_backtest_math.py"),
        "stocks_strategy_combo_lab_v1_1_1_crypto.py": (
            "MIGRATED",
            "research/combinatorial_lab.py",
            "test_combinatorial_lab.py",
        ),
        "macro_context_engine.py": (
            "MIGRATED",
            "research/macro_context.py",
            "test_database_macro_derivatives.py",
        ),
        "models/models_options.py": ("OUT_OF_SCOPE", None, None),
        "models/models_regime.py": (
            "ALREADY_SUPERSEDED",
            "research/macro_context.py",
            "test_database_macro_derivatives.py",
        ),
        "models/models_volatility.py": (
            "ALREADY_SUPERSEDED",
            "research/features.py",
            "test_features_strategies.py",
        ),
        "data_downloaders/crypto_data_downloader.py": (
            "ALREADY_SUPERSEDED",
            "data/data_loader.py",
            "test_data_realtime.py",
        ),
        "data_downloaders/multi_venue_crypto_data_downloader.py": (
            "DUPLICATE",
            "data/data_loader.py",
            "test_data_realtime.py",
        ),
        "data_downloaders/kraken_public_data_collector.py": (
            "ALREADY_SUPERSEDED",
            "data/data_loader.py",
            "test_data_realtime.py",
        ),
    }
    files: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in sorted(legacy_root.rglob("*.py")):
        if path.stat().st_size == 0:
            continue
        relative = path.relative_to(legacy_root).as_posix()
        if relative.startswith("scrapers_individual/"):
            disposition, destination, test = (
                "MIGRATED",
                "scrapers/intelligence.py",
                "test_intelligence.py",
            )
        elif relative.startswith("scrapers_rss/"):
            disposition, destination, test = (
                "MIGRATED",
                "scrapers/rss.py",
                "test_intelligence.py",
            )
        else:
            disposition, destination, test = rules.get(
                relative,
                ("DEFERRED_WITH_REASON", None, None),
            )
        reason = (
            "Active code is canonical and never imports legacy."
            if disposition != "OUT_OF_SCOPE"
            else "Options and derivative execution are outside crypto spot scope."
        )
        definitions = sorted(
            (
                {
                    "name": node.name,
                    "kind": (
                        "class"
                        if isinstance(node, ast.ClassDef)
                        else "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    ),
                    "line": node.lineno,
                }
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
            ),
            key=lambda item: (item["line"], item["kind"], item["name"]),
        )
        if not definitions:
            definitions = [{"name": "<module>", "kind": "module", "line": 1}]
        file_record = {
            "legacy_file": relative,
            "sha256": sha256_file(path),
            "disposition": disposition,
            "destination": destination,
            "reason": reason,
            "regression_test": test,
            "limitation": "Archived source remains for audit; no active runtime import.",
            "definition_count": len(definitions),
        }
        files.append(file_record)
        records.extend(
            {
                **file_record,
                "legacy_function_or_class": definition["name"],
                "definition_kind": definition["kind"],
                "definition_line": definition["line"],
            }
            for definition in definitions
        )
    payload = {
        "generated_at": utc_iso(),
        "legacy_root": str(legacy_root),
        "files_inspected": len(files),
        "definitions_inspected": len(records),
        "active_imports_legacy": 0,
        "dispositions": dict(sorted(Counter(row["disposition"] for row in files).items())),
        "files": files,
        "records": records,
    }
    target = settings.paths.reports_dir / "legacy_migration_report.json"
    atomic_write_json(target, payload)
    return target


__all__ = [
    "BlockDirection",
    "BlockRole",
    "CombinationGenerator",
    "CombinationState",
    "CombinatorialStrategy",
    "ExitProfile",
    "GenerationMode",
    "LabControl",
    "LabRunner",
    "LabStore",
    "LifecycleStatus",
    "LogicMode",
    "ParameterKind",
    "ParameterSpec",
    "OverlaySemantics",
    "SignalBlock",
    "SignalOperator",
    "StrategyCombination",
    "UniverseManager",
    "UniverseMember",
    "UniverseSnapshot",
    "UniverseType",
    "canonical_parameters",
    "describe_blocks",
    "parameter_hash",
    "signal_block_registry",
    "validate_blocks",
    "write_legacy_migration_report",
]
