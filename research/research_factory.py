"""Evidence-bound research factory with fast, authority-free falsification.

Stage 0 is deliberately approximate and orderless.  It can reject research
hypotheses or route a bounded candidate to the repository's native exact
backtester, but it cannot create paper/live authority or exchange actions.
"""

from __future__ import annotations

import inspect
import itertools
import json
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS, Settings
from core.economics import CanonicalCostModel
from research.backtest import BacktestConfig, BacktestEngine, CostModel
from research.features import FeaturePipeline
from research.optimization import (
    chronological_split,
    strategy_lookahead_test,
    strategy_repainting_test,
    walk_forward_optimize,
)
from research.stochastic_validation import stationary_bootstrap_monte_carlo
from research.strategies import Strategy, StrategyOutput
from utils.common import atomic_write_json, stable_hash, utc_iso

RESEARCH_FACTORY_SCHEMA_VERSION = "crypto_research_factory_v1"
DATASET_SCHEMA_VERSION = "immutable_ohlcv_dataset_v1"
EXPERIMENT_SCHEMA_VERSION = "canonical_research_experiment_v1"
VALIDATION_MANIFEST_SCHEMA_VERSION = "standard_walk_forward_manifest_v1"
STAGE0_SCHEMA_VERSION = "approximate_stage0_falsification_v1"
DEFAULT_STAGE0_NOTIONAL_EUR = 100.0


class PromotionState(StrEnum):
    IDEA = "IDEA"
    STAGE0_REJECTED = "STAGE0_REJECTED"
    STAGE0_PROMISING = "STAGE0_PROMISING"
    EXACT_VALIDATION = "EXACT_VALIDATION"
    EXACT_REJECTED = "EXACT_REJECTED"
    WALK_FORWARD_PASS = "WALK_FORWARD_PASS"
    WALK_FORWARD_FAIL = "WALK_FORWARD_FAIL"
    STRESS_PASS = "STRESS_PASS"
    STRESS_FAIL = "STRESS_FAIL"
    FORWARD_CANDIDATE = "FORWARD_CANDIDATE"
    PAPER_VALIDATING = "PAPER_VALIDATING"
    PAPER_POSITIVE = "PAPER_POSITIVE"
    LIVE_VALIDATED = "LIVE_VALIDATED"
    DEMOTED = "DEMOTED"


class RejectionReason(StrEnum):
    NEGATIVE_GROSS_EXPECTANCY = "NEGATIVE_GROSS_EXPECTANCY"
    NEGATIVE_NET_EXPECTANCY = "NEGATIVE_NET_EXPECTANCY"
    COST_FRAGILE = "COST_FRAGILE"
    PARAMETER_FRAGILE = "PARAMETER_FRAGILE"
    LOOKAHEAD_BIAS = "LOOKAHEAD_BIAS"
    RECURSIVE_INSTABILITY = "RECURSIVE_INSTABILITY"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    ASSET_OVERFIT = "ASSET_OVERFIT"
    REGIME_OVERFIT = "REGIME_OVERFIT"
    TIMEFRAME_OVERFIT = "TIMEFRAME_OVERFIT"
    EXCESSIVE_DRAWDOWN = "EXCESSIVE_DRAWDOWN"
    EXCESSIVE_TURNOVER = "EXCESSIVE_TURNOVER"
    DUPLICATE_ALPHA = "DUPLICATE_ALPHA"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    FORWARD_FAILURE = "FORWARD_FAILURE"
    P0_5_GROSS_NEGATIVE_FAMILY = "P0_5_GROSS_NEGATIVE_FAMILY"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"


@dataclass(frozen=True)
class ProspectiveSnapshot:
    snapshot_id: str
    candidate_id: str
    signal_timestamp: str
    feature_hash: str
    strategy_version: str
    parameter_version: str
    market_context_hash: str
    cost_prediction_version: str
    entry_plan_hash: str
    stop_targets_hash: str
    canonical_outcome_source: str = "P0_CANONICAL_FINANCIAL_STATE"
    future_information: bool = False

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        signal_timestamp: str,
        features: Mapping[str, Any],
        strategy_version: str,
        parameters: Mapping[str, Any],
        market_context: Mapping[str, Any],
        cost_prediction_version: str,
        entry_plan: Mapping[str, Any],
        stop_targets: Mapping[str, Any],
    ) -> "ProspectiveSnapshot":
        values = {
            "candidate_id": candidate_id,
            "signal_timestamp": _iso_timestamp(signal_timestamp),
            "feature_hash": stable_hash(features, length=40),
            "strategy_version": strategy_version,
            "parameter_version": stable_hash(parameters, length=40),
            "market_context_hash": stable_hash(market_context, length=40),
            "cost_prediction_version": cost_prediction_version,
            "entry_plan_hash": stable_hash(entry_plan, length=40),
            "stop_targets_hash": stable_hash(stop_targets, length=40),
        }
        return cls(snapshot_id=stable_hash(values, length=40), **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    status: str
    missing_bars: int
    duplicate_timestamps: int
    stale_or_non_increasing_bars: int
    null_ohlcv_rows: int
    invalid_ohlc_rows: int
    extreme_return_rows: int
    timestamp_discontinuities: int
    source_changes: str
    point_in_time_universe_status: str
    listing_age_status: str


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    schema_version: str
    provider: str
    market: str
    asset: str
    timeframe: str
    start: str
    end: str
    rows: int
    timestamp_convention: str
    data_hash: str
    source_file_hash: str
    source_byte_count: int
    adjustment_normalization_version: str
    missing_data_policy: str
    data_cutoff: str
    quality: DataQualityReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SharedCostModel = CanonicalCostModel


@dataclass(frozen=True)
class Stage0Hypothesis:
    hypothesis_id: str
    strategy_family: str
    strategy_implementation: str
    strategy_version: str
    candidate_origin: str
    rationale: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    parameter_space: Mapping[str, tuple[Any, ...]]
    supported_timeframes: tuple[str, ...]
    side: str
    holding_semantics: str
    p0_5_classification: str
    stage0_authority: str = "APPROXIMATE_RESEARCH_ONLY"

    def __post_init__(self) -> None:
        if self.stage0_authority != "APPROXIMATE_RESEARCH_ONLY":
            raise ValueError("Stage 0 can only have APPROXIMATE_RESEARCH_ONLY authority")
        if self.side != "LONG_ONLY":
            raise ValueError("P1 Stage 0 currently supports long-only spot hypotheses")


@dataclass(frozen=True)
class ExperimentContract:
    run_id: str
    experiment_id: str
    schema_version: str
    strategy_family: str
    strategy_implementation: str
    strategy_version: str
    parameter_set: Mapping[str, Any]
    asset_universe: str
    assets: tuple[str, ...]
    timeframe: str
    data_version: str
    data_cutoff: str
    feature_schema_version: str
    cost_model_version: str
    universe_version: str
    signal_version: str
    validation_manifest_id: str
    code_commit_hash: str | None
    created_at: str
    candidate_origin: str
    final_test_generation: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage0Signals:
    entry: pd.Series
    exit: pd.Series
    stop_distance: pd.Series
    target_distance: pd.Series

    def validate(self, index: pd.Index) -> "Stage0Signals":
        for value in (self.entry, self.exit, self.stop_distance, self.target_distance):
            if not value.index.equals(index):
                raise ValueError("Stage-0 signal output index mismatch")
        if (self.stop_distance.dropna() <= 0).any():
            raise ValueError("Stage-0 stop distances must be positive")
        if (self.target_distance.dropna() <= 0).any():
            raise ValueError("Stage-0 target distances must be positive")
        return self


@dataclass(frozen=True)
class Stage0Trade:
    entry_timestamp: str
    exit_timestamp: str
    signal_timestamp: str
    raw_entry_price: float
    raw_exit_price: float
    gross_pnl_eur: float
    fees_eur: float
    spread_cost_eur: float
    slippage_cost_eur: float
    net_pnl_eur: float
    holding_bars: int
    exit_reason: str


@dataclass(frozen=True)
class Stage0Result:
    result_id: str
    experiment_id: str
    schema_version: str
    hypothesis_id: str
    dataset_id: str
    market: str
    timeframe: str
    parameter_set: Mapping[str, Any]
    parameter_hash: str
    signal_count: int
    trade_count: int
    gross_pnl_eur: float
    estimated_fees_eur: float
    estimated_spread_eur: float
    estimated_slippage_eur: float
    net_pnl_eur: float
    gross_expectancy_eur: float | None
    net_expectancy_eur: float | None
    profit_factor: float | None
    sharpe: float | None
    maximum_drawdown_eur: float
    turnover_eur: float
    average_holding_bars: float | None
    exposure_fraction: float
    win_rate: float | None
    sample_size_status: str
    data_quality_status: str
    cost_model_version: str
    execution_delay_bars: int
    stage0_authority: str
    rejection_reasons: tuple[str, ...]
    trades: tuple[Stage0Trade, ...] = field(repr=False)

    def to_dict(self, *, include_trades: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_trades:
            value.pop("trades", None)
        return value


@dataclass(frozen=True)
class PlateauResult:
    parameter_hash: str
    stable: bool
    tested_neighbor_count: int
    positive_neighbor_fraction: float
    median_neighbor_expectancy_fraction: float | None
    isolated_optimum: bool


@dataclass(frozen=True)
class ValidationFold:
    fold: int
    train_start: str
    train_end: str
    purge_start: str
    purge_end: str
    validation_start: str
    validation_end: str
    embargo_start: str
    embargo_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class WalkForwardManifest:
    manifest_id: str
    schema_version: str
    strategy_id: str
    parameter_search_scope: Mapping[str, Any]
    asset_universe: tuple[str, ...]
    timeframe: str
    purge_period_bars: int
    embargo_period_bars: int
    regime_holdout: str
    asset_holdout: tuple[str, ...]
    parameter_selection_rule: str
    cost_assumptions: Mapping[str, Any]
    data_version: str
    code_version: str | None
    folds: tuple[ValidationFold, ...]
    final_test_immutable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SignalBuilder = Callable[[pd.DataFrame, Mapping[str, Any]], Stage0Signals]


def _iso_timestamp(value: Any) -> str:
    selected = pd.Timestamp(value)
    if selected.tzinfo is None:
        selected = selected.tz_localize("UTC")
    else:
        selected = selected.tz_convert("UTC")
    return selected.isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return default
    return selected if math.isfinite(selected) else default


def _frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return sha256(hashed.tobytes()).hexdigest()


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return pd.to_timedelta(int(TIMEFRAME_SECONDS[timeframe]), unit="s")


def validate_stage0_data(frame: pd.DataFrame, timeframe: str) -> DataQualityReport:
    required = ["open", "high", "low", "close", "volume"]
    if any(name not in frame.columns for name in required):
        return DataQualityReport(
            status="INVALID_DATA",
            missing_bars=0,
            duplicate_timestamps=0,
            stale_or_non_increasing_bars=0,
            null_ohlcv_rows=len(frame),
            invalid_ohlc_rows=len(frame),
            extreme_return_rows=0,
            timestamp_discontinuities=0,
            source_changes="NOT_EVALUABLE",
            point_in_time_universe_status="PIT_UNIVERSE_PARTIAL",
            listing_age_status="NOT_EVALUABLE",
        )
    duplicate = int(frame.index.duplicated().sum())
    differences = frame.index.to_series().diff()
    stale = int((differences.dropna() <= pd.Timedelta(0)).sum())
    expected = _timeframe_delta(timeframe)
    missing = (
        ((differences / expected) - 1.0).fillna(0.0).clip(lower=0.0)
        if expected > pd.Timedelta(0)
        else pd.Series(0.0, index=frame.index)
    )
    null_rows = int(frame[required].isna().any(axis=1).sum())
    invalid = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (frame["volume"] < 0)
        ).sum()
    )
    returns = frame["close"].astype(float).pct_change()
    extreme = int((returns.abs() > 0.80).sum())
    duration_days = max(
        0.0,
        (frame.index[-1] - frame.index[0]).total_seconds() / 86_400.0,
    ) if len(frame) >= 2 else 0.0
    critical = duplicate + stale + null_rows + invalid
    return DataQualityReport(
        status="READY" if critical == 0 else "INVALID_DATA",
        missing_bars=int(missing.sum()),
        duplicate_timestamps=duplicate,
        stale_or_non_increasing_bars=stale,
        null_ohlcv_rows=null_rows,
        invalid_ohlc_rows=invalid,
        extreme_return_rows=extreme,
        timestamp_discontinuities=int((missing > 0).sum()),
        source_changes="NOT_EVALUABLE_FROM_SINGLE_DATASET",
        point_in_time_universe_status="PIT_UNIVERSE_PARTIAL",
        listing_age_status=(
            "MATURE_HISTORY" if duration_days >= 730 else "SHORT_OR_PARTIAL_HISTORY"
        ),
    )


def load_immutable_ohlcv(
    path: Path,
    *,
    provider: str,
    market: str,
    timeframe: str,
    maximum_rows: int | None = None,
) -> tuple[pd.DataFrame, DatasetIdentity]:
    """Freeze one file at read and derive a reproducible selected dataset."""

    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_bytes()
    source_hash = sha256(content).hexdigest()
    frame = pd.read_parquet(BytesIO(content))
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"dataset has no timestamp index: {path}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index().loc[:, ["open", "high", "low", "close", "volume"]]
    frame = frame.astype(float)
    if maximum_rows is not None:
        if maximum_rows < 100:
            raise ValueError("maximum_rows must preserve at least 100 observations")
        frame = frame.iloc[-maximum_rows:].copy()
    quality = validate_stage0_data(frame, timeframe)
    data_hash = _frame_hash(frame)
    identity_inputs = {
        "schema": DATASET_SCHEMA_VERSION,
        "provider": provider,
        "market": market,
        "timeframe": timeframe,
        "start": _iso_timestamp(frame.index[0]),
        "end": _iso_timestamp(frame.index[-1]),
        "rows": len(frame),
        "data_hash": data_hash,
        "source_file_hash": source_hash,
    }
    dataset_id = stable_hash(identity_inputs, length=40)
    frame.attrs.update(
        {
            "market": market,
            "timeframe": timeframe,
            "provider": provider,
            "dataset_id": dataset_id,
            "closed_candles_only": True,
            "point_in_time": True,
        }
    )
    return frame, DatasetIdentity(
        dataset_id=dataset_id,
        schema_version=DATASET_SCHEMA_VERSION,
        provider=provider,
        market=market,
        asset=market.split("-")[0],
        timeframe=timeframe,
        start=_iso_timestamp(frame.index[0]),
        end=_iso_timestamp(frame.index[-1]),
        rows=len(frame),
        timestamp_convention="UTC_CANDLE_OPEN_TIMESTAMP_CLOSED_BAR_ONLY",
        data_hash=data_hash,
        source_file_hash=source_hash,
        source_byte_count=len(content),
        adjustment_normalization_version="normalized_ohlcv_v1",
        missing_data_policy="PRESERVE_GAPS_AND_REPORT_DO_NOT_FORWARD_FILL",
        data_cutoff=_iso_timestamp(frame.index[-1]),
        quality=quality,
    )


def generate_parameter_grid(
    parameter_space: Mapping[str, Sequence[Any]],
) -> tuple[dict[str, Any], ...]:
    if not parameter_space:
        return ({},)
    names = tuple(sorted(parameter_space))
    values: list[tuple[Any, ...]] = []
    for name in names:
        selected = tuple(parameter_space[name])
        if not selected:
            raise ValueError(f"parameter has no values: {name}")
        values.append(selected)
    return tuple(
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*values)
    )


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def failed_breakdown_reversal_signals(
    frame: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> Stage0Signals:
    """Causal structural surrogate for the existing event-driven playbook."""

    lookback = int(parameters["lookback"])
    breach = float(parameters["breach_bps"]) / 10_000.0
    reclaim = float(parameters["reclaim_bps"]) / 10_000.0
    volume_multiple = float(parameters["volume_multiple"])
    prior_low = frame["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    prior_mean = frame["close"].rolling(lookback, min_periods=lookback).mean().shift(1)
    prior_volume = frame["volume"].rolling(lookback, min_periods=lookback).mean().shift(1)
    entry = (
        (frame["low"] <= prior_low * (1.0 - breach))
        & (frame["close"] >= prior_low * (1.0 + reclaim))
        & (frame["close"] > frame["open"])
        & (frame["volume"] >= prior_volume * volume_multiple)
        & (frame["close"].pct_change() > 0)
    ).fillna(False)
    exit_signal = (
        (frame["close"] < prior_low)
        | (frame["close"] >= prior_mean)
    ).fillna(False)
    atr = _atr(frame)
    return Stage0Signals(
        entry=entry.astype(bool),
        exit=exit_signal.astype(bool),
        stop_distance=atr * float(parameters["stop_atr"]),
        target_distance=atr * float(parameters["target_atr"]),
    ).validate(frame.index)


class FailedBreakdownReversalResearchAdapter(Strategy):
    """Exact-backtest adapter; it is not a new production strategy."""

    strategy_id = "FAILED_BREAKDOWN_REVERSAL_V1_RESEARCH_ADAPTER"
    family = "FAILED_BREAKDOWN_REVERSAL"
    description = "Causal OHLCV structural surrogate for exact research validation."
    defaults = {
        "lookback": 20,
        "breach_bps": 10.0,
        "reclaim_bps": 0.0,
        "volume_multiple": 1.0,
        "stop_atr": 1.5,
        "target_atr": 2.5,
        "trailing_atr": 0.0,
        "maximum_holding_bars": 32,
    }
    parameter_space = {
        "lookback": (20, 40),
        "breach_bps": (0.0, 10.0),
        "reclaim_bps": (0.0, 10.0),
        "volume_multiple": (1.0, 1.25),
        "stop_atr": (1.5, 2.0),
        "target_atr": (2.5, 3.5),
    }

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        super().validate_parameters(parameters)
        if int(parameters["lookback"]) < 2:
            raise ValueError("lookback must be at least two bars")
        if float(parameters["volume_multiple"]) <= 0:
            raise ValueError("volume_multiple must be positive")

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        selected = self.parameters(parameters)
        signals = failed_breakdown_reversal_signals(features, selected)
        return self._output(
            features,
            entry=signals.entry,
            exit=signals.exit,
            parameters=selected,
            entry_reason="FAILED_BREAKDOWN_RECLAIM_RESEARCH",
            exit_reason="RECLAIM_STRUCTURE_INVALIDATED_OR_MEAN_REACHED",
            metadata={
                "candidate_origin": "STRUCTURAL_VARIANT",
                "production_registration": False,
            },
        )


def _drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def simulate_stage0(
    frame: pd.DataFrame,
    signals: Stage0Signals,
    *,
    hypothesis: Stage0Hypothesis,
    dataset: DatasetIdentity,
    parameters: Mapping[str, Any],
    costs: SharedCostModel,
    execution_delay_bars: int = 0,
    notional_eur: float = DEFAULT_STAGE0_NOTIONAL_EUR,
    minimum_trades: int = 30,
) -> Stage0Result:
    """Approximate next-open long-only simulation with conservative intrabar ties."""

    if execution_delay_bars < 0:
        raise ValueError("execution delay cannot be negative")
    if notional_eur <= 0:
        raise ValueError("notional must be positive")
    signals.validate(frame.index)
    entry_values = signals.entry.to_numpy(dtype=bool)
    exit_values = signals.exit.to_numpy(dtype=bool)
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    stop_distances = signals.stop_distance.to_numpy(dtype=float)
    target_distances = signals.target_distance.to_numpy(dtype=float)
    maximum_holding = int(parameters.get("maximum_holding_bars") or 32)
    trades: list[Stage0Trade] = []
    occupied_until = -1
    for signal_index in np.flatnonzero(entry_values):
        if signal_index <= occupied_until:
            continue
        entry_index = int(signal_index) + 1 + execution_delay_bars
        if entry_index >= len(frame):
            continue
        stop_distance = float(stop_distances[signal_index])
        target_distance = float(target_distances[signal_index])
        if not (
            math.isfinite(stop_distance)
            and math.isfinite(target_distance)
            and stop_distance > 0
            and target_distance > 0
        ):
            continue
        raw_entry = float(opens[entry_index])
        stop_price = raw_entry - stop_distance
        target_price = raw_entry + target_distance
        exit_index = min(len(frame) - 1, entry_index + maximum_holding)
        raw_exit = float(closes[exit_index])
        reason = "MAXIMUM_HOLDING"
        for cursor in range(entry_index, exit_index + 1):
            if cursor > entry_index and exit_values[cursor - 1]:
                exit_index = cursor
                raw_exit = float(opens[cursor])
                reason = "SIGNAL_EXIT_NEXT_OPEN"
                break
            stop_hit = lows[cursor] <= stop_price
            target_hit = highs[cursor] >= target_price
            if stop_hit:
                exit_index = cursor
                raw_exit = stop_price
                reason = "STOP_FIRST_CONSERVATIVE" if target_hit else "STOP"
                break
            if target_hit:
                exit_index = cursor
                raw_exit = target_price
                reason = "TARGET"
                break
        quantity = notional_eur / raw_entry
        half_spread = costs.spread_bps / 2.0 / 10_000.0
        slippage = (
            costs.slippage_bps
            + costs.failed_execution_allowance_bps
            + costs.partial_fill_impact_bps
        ) / 10_000.0
        adjusted_entry = raw_entry * (1.0 + half_spread + slippage)
        adjusted_exit = raw_exit * (1.0 - half_spread - slippage)
        gross = quantity * (raw_exit - raw_entry)
        spread_cost = quantity * (raw_entry + raw_exit) * half_spread
        slippage_cost = quantity * (raw_entry + raw_exit) * slippage
        fees = (
            quantity * adjusted_entry * costs.taker_fee_fraction
            + quantity * adjusted_exit * costs.taker_fee_fraction
        )
        net = quantity * (adjusted_exit - adjusted_entry) - fees
        trades.append(
            Stage0Trade(
                entry_timestamp=_iso_timestamp(frame.index[entry_index]),
                exit_timestamp=_iso_timestamp(frame.index[exit_index]),
                signal_timestamp=_iso_timestamp(frame.index[signal_index]),
                raw_entry_price=raw_entry,
                raw_exit_price=raw_exit,
                gross_pnl_eur=gross,
                fees_eur=fees,
                spread_cost_eur=spread_cost,
                slippage_cost_eur=slippage_cost,
                net_pnl_eur=net,
                holding_bars=exit_index - entry_index + 1,
                exit_reason=reason,
            )
        )
        occupied_until = exit_index
    gross_values = [row.gross_pnl_eur for row in trades]
    net_values = [row.net_pnl_eur for row in trades]
    gross_total = float(sum(gross_values))
    net_total = float(sum(net_values))
    fees_total = float(sum(row.fees_eur for row in trades))
    spread_total = float(sum(row.spread_cost_eur for row in trades))
    slippage_total = float(sum(row.slippage_cost_eur for row in trades))
    wins = [value for value in net_values if value > 0]
    losses = [value for value in net_values if value < 0]
    profit_factor = (
        sum(wins) / abs(sum(losses))
        if losses
        else None
    )
    sharpe = (
        statistics.fmean(net_values) / statistics.stdev(net_values)
        if len(net_values) >= 10 and statistics.stdev(net_values) > 0
        else None
    )
    rejection: list[str] = []
    if dataset.quality.status != "READY":
        rejection.append(RejectionReason.DATA_QUALITY_FAILURE)
    if len(trades) < minimum_trades:
        rejection.append(RejectionReason.INSUFFICIENT_SAMPLE)
    if trades and gross_total <= 0:
        rejection.append(RejectionReason.NEGATIVE_GROSS_EXPECTANCY)
    if trades and net_total <= 0:
        rejection.append(RejectionReason.NEGATIVE_NET_EXPECTANCY)
    if trades and profit_factor is not None and profit_factor < 1.0:
        if RejectionReason.NEGATIVE_NET_EXPECTANCY not in rejection:
            rejection.append(RejectionReason.NEGATIVE_NET_EXPECTANCY)
    parameter_hash = stable_hash(parameters, length=32)
    result_identity = {
        "schema": STAGE0_SCHEMA_VERSION,
        "hypothesis": hypothesis.hypothesis_id,
        "dataset": dataset.dataset_id,
        "parameters": parameters,
        "costs": costs.cost_model_version,
        "delay": execution_delay_bars,
    }
    experiment_id = stable_hash(
        {
            "schema": EXPERIMENT_SCHEMA_VERSION,
            "hypothesis": hypothesis.hypothesis_id,
            "dataset": dataset.dataset_id,
            "parameters": parameters,
            "costs": costs.cost_model_version,
            "signal_version": STAGE0_SCHEMA_VERSION,
        },
        length=40,
    )
    return Stage0Result(
        result_id=stable_hash(result_identity, length=40),
        experiment_id=experiment_id,
        schema_version=STAGE0_SCHEMA_VERSION,
        hypothesis_id=hypothesis.hypothesis_id,
        dataset_id=dataset.dataset_id,
        market=dataset.market,
        timeframe=dataset.timeframe,
        parameter_set=dict(parameters),
        parameter_hash=parameter_hash,
        signal_count=int(signals.entry.sum()),
        trade_count=len(trades),
        gross_pnl_eur=gross_total,
        estimated_fees_eur=fees_total,
        estimated_spread_eur=spread_total,
        estimated_slippage_eur=slippage_total,
        net_pnl_eur=net_total,
        gross_expectancy_eur=(gross_total / len(trades) if trades else None),
        net_expectancy_eur=(net_total / len(trades) if trades else None),
        profit_factor=profit_factor,
        sharpe=sharpe,
        maximum_drawdown_eur=_drawdown(net_values),
        turnover_eur=len(trades) * notional_eur * 2.0,
        average_holding_bars=(
            statistics.fmean(row.holding_bars for row in trades) if trades else None
        ),
        exposure_fraction=(
            sum(row.holding_bars for row in trades) / len(frame) if len(frame) else 0.0
        ),
        win_rate=(len(wins) / len(trades) if trades else None),
        sample_size_status=(
            "SUFFICIENT" if len(trades) >= minimum_trades else "INSUFFICIENT_SAMPLE"
        ),
        data_quality_status=dataset.quality.status,
        cost_model_version=costs.cost_model_version,
        execution_delay_bars=execution_delay_bars,
        stage0_authority="APPROXIMATE_RESEARCH_ONLY",
        rejection_reasons=tuple(dict.fromkeys(str(value) for value in rejection)),
        trades=tuple(trades),
    )


def stage0_causality_check(
    frame: pd.DataFrame,
    builder: SignalBuilder,
    parameters: Mapping[str, Any],
    *,
    cutoff_fraction: float = 0.80,
) -> dict[str, Any]:
    """Prove that changing future OHLCV cannot change earlier signals."""

    if not 0.5 <= cutoff_fraction < 1.0:
        raise ValueError("cutoff_fraction must be in [0.5, 1.0)")
    cutoff = max(2, int(len(frame) * cutoff_fraction))
    baseline = builder(frame.copy(), parameters)
    perturbed = frame.copy()
    columns = [name for name in ("open", "high", "low", "close", "volume") if name in frame]
    perturbed.loc[perturbed.index[cutoff:], columns] = (
        perturbed.loc[perturbed.index[cutoff:], columns] * 7.0 + 123.0
    )
    candidate = builder(perturbed, parameters)
    entry_safe = baseline.entry.iloc[:cutoff].equals(candidate.entry.iloc[:cutoff])
    exit_safe = baseline.exit.iloc[:cutoff].equals(candidate.exit.iloc[:cutoff])
    stop_safe = np.allclose(
        baseline.stop_distance.iloc[:cutoff],
        candidate.stop_distance.iloc[:cutoff],
        equal_nan=True,
    )
    target_safe = np.allclose(
        baseline.target_distance.iloc[:cutoff],
        candidate.target_distance.iloc[:cutoff],
        equal_nan=True,
    )
    return {
        "status": "PASSED" if all((entry_safe, exit_safe, stop_safe, target_safe)) else "FAILED",
        "cutoff_index": cutoff,
        "entry_causal": bool(entry_safe),
        "exit_causal": bool(exit_safe),
        "stop_causal": bool(stop_safe),
        "target_causal": bool(target_safe),
        "future_rows_modified": len(frame) - cutoff,
    }


def static_lookahead_audit(source: str | Callable[..., Any]) -> dict[str, Any]:
    """Conservative source-pattern audit complementing dynamic prefix tests."""

    text = inspect.getsource(source) if callable(source) else str(source)
    patterns = {
        "NEGATIVE_SHIFT": ("shift(-", ".shift(periods=-"),
        "CENTERED_ROLLING": ("center=True", "center = True"),
        "BACKFILL": (".bfill(", "method=\"bfill\"", "method='bfill'"),
        "FUTURE_INDEX_ACCESS": ("iloc[-1]", "iat[-1]"),
        "WHOLE_SERIES_NORMALIZATION": ("expanding().max()", "expanding().min()"),
    }
    findings = [
        name
        for name, tokens in patterns.items()
        if any(token in text for token in tokens)
    ]
    return {
        "status": "HARD_REJECT" if findings else "PASSED",
        "findings": findings,
        "checked_patterns": sorted(patterns),
        "scope": "STATIC_HEURISTIC_PLUS_DYNAMIC_CAUSALITY_REQUIRED",
    }


def recursive_warmup_stability(
    frame: pd.DataFrame,
    builder: SignalBuilder,
    parameters: Mapping[str, Any],
    *,
    startup_sizes: Sequence[int] = (200, 500, 1_000),
    comparison_rows: int = 50,
) -> dict[str, Any]:
    """Compare final signals after different causal startup histories."""

    if comparison_rows < 1:
        raise ValueError("comparison_rows must be positive")
    sizes = sorted({int(value) for value in startup_sizes if int(value) > comparison_rows})
    sizes = [value for value in sizes if value <= len(frame)]
    if len(sizes) < 2:
        return {
            "status": "NOT_EVALUABLE_INSUFFICIENT_HISTORY",
            "startup_sizes": sizes,
            "stable": None,
        }
    outputs = []
    for size in sizes:
        selected = frame.iloc[-size:].copy()
        selected.attrs.update(frame.attrs)
        outputs.append((size, builder(selected, parameters)))
    reference_size, reference = outputs[-1]
    comparisons: list[dict[str, Any]] = []
    stable = True
    for size, candidate in outputs[:-1]:
        shared = reference.entry.index.intersection(candidate.entry.index)[-comparison_rows:]
        entry_equal = reference.entry.loc[shared].equals(candidate.entry.loc[shared])
        exit_equal = reference.exit.loc[shared].equals(candidate.exit.loc[shared])
        stop_equal = np.allclose(
            reference.stop_distance.loc[shared],
            candidate.stop_distance.loc[shared],
            equal_nan=True,
        )
        target_equal = np.allclose(
            reference.target_distance.loc[shared],
            candidate.target_distance.loc[shared],
            equal_nan=True,
        )
        current = all((entry_equal, exit_equal, stop_equal, target_equal))
        stable &= current
        comparisons.append(
            {
                "startup_size": size,
                "reference_startup_size": reference_size,
                "comparison_rows": len(shared),
                "stable": current,
            }
        )
    return {
        "status": "PASSED" if stable else "FAILED",
        "startup_sizes": sizes,
        "stable": stable,
        "comparisons": comparisons,
    }


def aggregate_stage0_by_parameter(
    results: Sequence[Stage0Result],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Stage0Result]] = defaultdict(list)
    for result in results:
        grouped[result.parameter_hash].append(result)
    rows: list[dict[str, Any]] = []
    for parameter_hash, selected in grouped.items():
        trades = [trade for result in selected for trade in result.trades]
        net = [trade.net_pnl_eur for trade in trades]
        gross = [trade.gross_pnl_eur for trade in trades]
        wins = [value for value in net if value > 0]
        losses = [value for value in net if value < 0]
        positive_assets = {
            result.market
            for result in selected
            if result.net_pnl_eur > 0 and result.trade_count > 0
        }
        positive_timeframes = {
            result.timeframe
            for result in selected
            if result.net_pnl_eur > 0 and result.trade_count > 0
        }
        rows.append(
            {
                "parameter_hash": parameter_hash,
                "parameter_set": dict(selected[0].parameter_set),
                "dataset_count": len(selected),
                "asset_count": len({row.market for row in selected}),
                "timeframe_count": len({row.timeframe for row in selected}),
                "positive_asset_count": len(positive_assets),
                "positive_timeframe_count": len(positive_timeframes),
                "trade_count": len(trades),
                "gross_pnl_eur": float(sum(gross)),
                "net_pnl_eur": float(sum(net)),
                "gross_expectancy_eur": statistics.fmean(gross) if gross else None,
                "net_expectancy_eur": statistics.fmean(net) if net else None,
                "profit_factor": (
                    sum(wins) / abs(sum(losses)) if losses else None
                ),
                "maximum_drawdown_eur": _drawdown(net),
                "win_rate": len(wins) / len(net) if net else None,
                "turnover_eur": sum(result.turnover_eur for result in selected),
                "rejected_dataset_count": sum(bool(row.rejection_reasons) for row in selected),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _safe_float(row.get("net_expectancy_eur"), -math.inf),
            _safe_float(row.get("net_pnl_eur"), -math.inf),
        ),
        reverse=True,
    )


def _parameter_neighbors(
    parameters: Mapping[str, Any],
    parameter_space: Mapping[str, Sequence[Any]],
) -> set[str]:
    neighbors: set[str] = set()
    for name, options in parameter_space.items():
        values = list(options)
        if name not in parameters or parameters[name] not in values:
            continue
        position = values.index(parameters[name])
        for candidate in (position - 1, position + 1):
            if 0 <= candidate < len(values):
                modified = dict(parameters)
                modified[name] = values[candidate]
                neighbors.add(stable_hash(modified, length=32))
    return neighbors


def parameter_plateaus(
    aggregate_rows: Sequence[Mapping[str, Any]],
    parameter_space: Mapping[str, Sequence[Any]],
) -> list[PlateauResult]:
    by_hash = {str(row["parameter_hash"]): row for row in aggregate_rows}
    output: list[PlateauResult] = []
    for row in aggregate_rows:
        parameters = dict(row["parameter_set"])
        neighbor_hashes = _parameter_neighbors(parameters, parameter_space)
        neighbors = [by_hash[value] for value in neighbor_hashes if value in by_hash]
        base = _safe_float(row.get("net_expectancy_eur"))
        positive = [
            selected
            for selected in neighbors
            if _safe_float(selected.get("net_expectancy_eur")) > 0
        ]
        fraction = len(positive) / len(neighbors) if neighbors else 0.0
        median_neighbor = (
            statistics.median(
                _safe_float(selected.get("net_expectancy_eur"))
                for selected in neighbors
            )
            if neighbors
            else None
        )
        median_fraction = (
            median_neighbor / base
            if median_neighbor is not None and base > 0
            else None
        )
        stable = bool(neighbors) and base > 0 and fraction >= 0.60 and (
            median_fraction is not None and median_fraction >= 0.50
        )
        isolated = base > 0 and bool(neighbors) and fraction < 0.40
        output.append(
            PlateauResult(
                parameter_hash=str(row["parameter_hash"]),
                stable=stable,
                tested_neighbor_count=len(neighbors),
                positive_neighbor_fraction=fraction,
                median_neighbor_expectancy_fraction=median_fraction,
                isolated_optimum=isolated,
            )
        )
    return output


def multiple_testing_accounting(
    *,
    hypotheses: int,
    parameter_combinations: int,
    assets: int,
    timeframes: int,
    regimes: int,
    cost_scenarios: int,
) -> dict[str, Any]:
    values = {
        "strategy_hypotheses": hypotheses,
        "parameter_combinations": parameter_combinations,
        "assets": assets,
        "timeframes": timeframes,
        "regime_filters": regimes,
        "cost_scenarios": cost_scenarios,
    }
    if any(int(value) < 1 for value in values.values()):
        raise ValueError("multiple-testing dimensions must all be positive")
    total = math.prod(int(value) for value in values.values())
    return {
        **values,
        "total_tested_variants": total,
        "warning": (
            "SELECTED_RESULTS_ARE_NOT_SINGLE_INDEPENDENT_TESTS"
            if total > 1
            else None
        ),
        "penalty_framework": "CONSERVATIVE_TRIAL_COUNT_DISCLOSURE_AND_PLATEAU_GATE",
        "false_precision": False,
    }


def prioritize_research_backlog(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank defensible research non-FIFO while deferring disproven families."""

    promise_scores = {
        "PAPER_POSITIVE": 100.0,
        "PROMISING": 80.0,
        "VALIDATION_CANDIDATE": 75.0,
        "INSUFFICIENT_SAMPLE": 50.0,
        "GROSS_POSITIVE_NET_POSITIVE": 70.0,
        "GROSS_POSITIVE_NET_NEGATIVE": 35.0,
        "CANONICALLY_NEGATIVE": -100.0,
        "GROSS_NEGATIVE": -100.0,
    }
    origin_scores = {
        "EXISTING_STRATEGY": 0.0,
        "PARAMETER_VARIANT": -10.0,
        "STRUCTURAL_VARIANT": 12.0,
        "NEW_HYPOTHESIS": 8.0,
    }
    output: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        promise = str(item.get("p0_5_classification") or "INSUFFICIENT_SAMPLE")
        origin = str(item.get("candidate_origin") or "EXISTING_STRATEGY")
        data = min(1.0, max(0.0, _safe_float(item.get("data_availability"))))
        sample = min(1.0, max(0.0, _safe_float(item.get("sample_availability"))))
        diversification = min(
            1.0, max(0.0, _safe_float(item.get("diversification_potential")))
        )
        validation_cost = min(
            1.0, max(0.0, _safe_float(item.get("validation_cost"), default=0.5))
        )
        disproven = bool(item.get("already_disproven")) or promise in {
            "CANONICALLY_NEGATIVE",
            "GROSS_NEGATIVE",
        }
        priority_score = (
            promise_scores.get(promise, 0.0)
            + origin_scores.get(origin, 0.0)
            + data * 20.0
            + sample * 10.0
            + diversification * 10.0
            - validation_cost * 10.0
        )
        output.append(
            {
                **item,
                "priority_score": priority_score,
                "scheduler_state": (
                    "DEFER_DO_NOT_RESCUE" if disproven else "ELIGIBLE_BOUNDED_RESEARCH"
                ),
                "fifo": False,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["scheduler_state"] == "ELIGIBLE_BOUNDED_RESEARCH",
            _safe_float(row["priority_score"]),
            str(row.get("backlog_id") or row.get("strategy_family") or ""),
        ),
        reverse=True,
    )


def promotion_state_for(
    *,
    stage0_survivor_count: int,
    exact_status: str | None,
) -> PromotionState:
    if exact_status == "ROBUST_EXACT_PASS":
        return PromotionState.FORWARD_CANDIDATE
    if exact_status in {"EXACT_REJECTED", "HARD_REJECT", "EXACT_VALIDATION_ERROR"}:
        return PromotionState.EXACT_REJECTED
    if stage0_survivor_count > 0:
        return PromotionState.EXACT_VALIDATION
    return PromotionState.STAGE0_REJECTED


def build_walk_forward_manifest(
    datasets: Sequence[DatasetIdentity],
    *,
    strategy_id: str,
    parameter_search_scope: Mapping[str, Any],
    timeframe: str,
    costs: SharedCostModel,
    code_version: str | None,
    folds: int = 3,
    purge_bars: int = 40,
    embargo_bars: int = 2,
    asset_holdout: Sequence[str] = (),
) -> WalkForwardManifest:
    if folds < 2:
        raise ValueError("walk-forward manifest requires at least two folds")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge and embargo cannot be negative")
    selected = [row for row in datasets if row.timeframe == timeframe]
    if not selected:
        raise ValueError("manifest requires at least one matching dataset")
    start = max(pd.Timestamp(row.start) for row in selected)
    end = min(pd.Timestamp(row.end) for row in selected)
    if start >= end:
        raise ValueError("datasets have no shared chronological window")
    timeline = pd.date_range(start, end, periods=folds * 3 + 1)
    bar_delta = _timeframe_delta(timeframe)
    purge_delta = bar_delta * purge_bars
    embargo_delta = bar_delta * embargo_bars
    fold_rows: list[ValidationFold] = []
    for fold in range(folds):
        base = fold * 3
        train_start = timeline[0]
        train_end = timeline[base + 1]
        purge_start = train_end
        validation_start = train_end + purge_delta
        purge_end = validation_start
        validation_end = timeline[base + 2]
        embargo_start = validation_end
        test_start = validation_end + embargo_delta
        embargo_end = test_start
        test_end = timeline[base + 3]
        if not train_end <= validation_start < validation_end <= test_start < test_end:
            raise ValueError("purge or embargo leaves an empty validation/test window")
        fold_rows.append(
            ValidationFold(
                fold=fold + 1,
                train_start=_iso_timestamp(train_start),
                train_end=_iso_timestamp(train_end),
                purge_start=_iso_timestamp(purge_start),
                purge_end=_iso_timestamp(purge_end),
                validation_start=_iso_timestamp(validation_start),
                validation_end=_iso_timestamp(validation_end),
                embargo_start=_iso_timestamp(embargo_start),
                embargo_end=_iso_timestamp(embargo_end),
                test_start=_iso_timestamp(test_start),
                test_end=_iso_timestamp(test_end),
            )
        )
    identity = {
        "schema": VALIDATION_MANIFEST_SCHEMA_VERSION,
        "strategy": strategy_id,
        "datasets": [row.dataset_id for row in selected],
        "scope": parameter_search_scope,
        "timeframe": timeframe,
        "purge": purge_bars,
        "embargo": embargo_bars,
        "assets": sorted(row.market for row in selected),
        "holdout": sorted(asset_holdout),
        "cost_model": costs.cost_model_version,
        "folds": [asdict(row) for row in fold_rows],
    }
    return WalkForwardManifest(
        manifest_id=stable_hash(identity, length=40),
        schema_version=VALIDATION_MANIFEST_SCHEMA_VERSION,
        strategy_id=strategy_id,
        parameter_search_scope=dict(parameter_search_scope),
        asset_universe=tuple(sorted(row.market for row in selected)),
        timeframe=timeframe,
        purge_period_bars=purge_bars,
        embargo_period_bars=embargo_bars,
        regime_holdout="CAUSAL_LABELS_ONLY_OR_NOT_EVALUABLE",
        asset_holdout=tuple(sorted(asset_holdout)),
        parameter_selection_rule=(
            "TRAIN_VALIDATION_ONLY_STABLE_PLATEAU_NO_FINAL_TEST_RETUNING"
        ),
        cost_assumptions=asdict(costs),
        data_version=stable_hash([row.dataset_id for row in selected], length=40),
        code_version=code_version,
        folds=tuple(fold_rows),
        final_test_immutable=True,
    )


class ResearchCache:
    """Small immutable JSON cache with fully versioned keys."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(
        *,
        dataset_ids: Sequence[str],
        parameters: Mapping[str, Any],
        strategy_version: str,
        timeframe: str,
        cost_model_version: str,
        feature_schema_version: str,
    ) -> str:
        return stable_hash(
            {
                "datasets": sorted(dataset_ids),
                "parameters": parameters,
                "strategy_version": strategy_version,
                "timeframe": timeframe,
                "cost_model_version": cost_model_version,
                "feature_schema_version": feature_schema_version,
            },
            length=48,
        )

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("cache_key") != key:
            raise ValueError("research cache identity mismatch")
        return dict(value)

    def store(self, key: str, payload: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        selected = {"cache_key": key, **dict(payload)}
        if path.is_file():
            current = json.loads(path.read_text(encoding="utf-8"))
            if stable_hash(current) != stable_hash(selected):
                raise FileExistsError(f"immutable research cache collision: {path}")
            return path
        atomic_write_json(path, selected)
        return path


def derive_dataset_identity(
    frame: pd.DataFrame,
    parent: DatasetIdentity,
    *,
    purpose: str,
) -> DatasetIdentity:
    """Identify an immutable chronological subset without losing provenance."""

    if frame.empty:
        raise ValueError("derived dataset cannot be empty")
    selected = frame.sort_index().copy()
    quality = validate_stage0_data(selected, parent.timeframe)
    data_hash = _frame_hash(selected)
    identity = {
        "schema": DATASET_SCHEMA_VERSION,
        "parent": parent.dataset_id,
        "purpose": purpose,
        "start": _iso_timestamp(selected.index[0]),
        "end": _iso_timestamp(selected.index[-1]),
        "rows": len(selected),
        "data_hash": data_hash,
    }
    return DatasetIdentity(
        dataset_id=stable_hash(identity, length=40),
        schema_version=DATASET_SCHEMA_VERSION,
        provider=parent.provider,
        market=parent.market,
        asset=parent.asset,
        timeframe=parent.timeframe,
        start=_iso_timestamp(selected.index[0]),
        end=_iso_timestamp(selected.index[-1]),
        rows=len(selected),
        timestamp_convention=parent.timestamp_convention,
        data_hash=data_hash,
        source_file_hash=parent.source_file_hash,
        source_byte_count=parent.source_byte_count,
        adjustment_normalization_version=(
            f"{parent.adjustment_normalization_version}:{purpose}"
        ),
        missing_data_policy=parent.missing_data_policy,
        data_cutoff=_iso_timestamp(selected.index[-1]),
        quality=quality,
    )


def _aggregate_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    trades = int(row.get("trade_count") or 0)
    gross = _safe_float(row.get("gross_expectancy_eur"))
    net = _safe_float(row.get("net_expectancy_eur"))
    pf = row.get("profit_factor")
    if trades < 100:
        reasons.append(RejectionReason.INSUFFICIENT_SAMPLE)
    if gross <= 0:
        reasons.append(RejectionReason.NEGATIVE_GROSS_EXPECTANCY)
    if net <= 0:
        reasons.append(RejectionReason.NEGATIVE_NET_EXPECTANCY)
    if pf is not None and _safe_float(pf) < 1.0:
        if RejectionReason.NEGATIVE_NET_EXPECTANCY not in reasons:
            reasons.append(RejectionReason.NEGATIVE_NET_EXPECTANCY)
    if int(row.get("positive_asset_count") or 0) < 2:
        reasons.append(RejectionReason.ASSET_OVERFIT)
    return [str(value) for value in reasons]


def stage0_cost_stress(
    *,
    frames: Mapping[str, pd.DataFrame],
    datasets: Mapping[str, DatasetIdentity],
    hypothesis: Stage0Hypothesis,
    builder: SignalBuilder,
    parameters: Mapping[str, Any],
    costs: SharedCostModel,
    stress_bps: Sequence[float] = (0.0, 10.0, 25.0, 50.0),
    execution_delay_bars: int = 0,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    signal_cache = {
        key: builder(frame, parameters) for key, frame in frames.items()
    }
    for stress in stress_bps:
        stressed = costs.stressed(additional_roundtrip_bps=float(stress))
        rows = [
            simulate_stage0(
                frame,
                signal_cache[key],
                hypothesis=hypothesis,
                dataset=datasets[key],
                parameters=parameters,
                costs=stressed,
                execution_delay_bars=execution_delay_bars,
            )
            for key, frame in frames.items()
        ]
        trades = [trade for row in rows for trade in row.trades]
        output.append(
            {
                "scenario": "BASE" if stress == 0 else f"BASE_PLUS_{int(stress)}_BPS",
                "additional_roundtrip_bps": float(stress),
                "trade_count": len(trades),
                "net_pnl_eur": float(sum(row.net_pnl_eur for row in trades)),
                "net_expectancy_eur": (
                    statistics.fmean(row.net_pnl_eur for row in trades)
                    if trades
                    else None
                ),
                "profit_factor": (
                    sum(row.net_pnl_eur for row in trades if row.net_pnl_eur > 0)
                    / abs(sum(row.net_pnl_eur for row in trades if row.net_pnl_eur < 0))
                    if any(row.net_pnl_eur < 0 for row in trades)
                    else None
                ),
                "cost_model_version": stressed.cost_model_version,
            }
        )
    return output


def stage0_delay_liquidity_stress(
    *,
    frames: Mapping[str, pd.DataFrame],
    datasets: Mapping[str, DatasetIdentity],
    hypothesis: Stage0Hypothesis,
    builder: SignalBuilder,
    parameters: Mapping[str, Any],
    costs: SharedCostModel,
) -> list[dict[str, Any]]:
    scenarios = (
        ("NEXT_OPEN_BASE", 0, 1.0, 1.0),
        ("ONE_EXTRA_BAR_DELAY", 1, 1.0, 1.0),
        ("TWO_EXTRA_BARS_DELAY", 2, 1.0, 1.0),
        ("LIQUIDITY_2X", 0, 2.0, 2.0),
        ("LIQUIDITY_3X", 0, 3.0, 3.0),
    )
    signals = {key: builder(frame, parameters) for key, frame in frames.items()}
    output: list[dict[str, Any]] = []
    for name, delay, spread_multiplier, slippage_multiplier in scenarios:
        selected_costs = costs.stressed(
            spread_multiplier=spread_multiplier,
            slippage_multiplier=slippage_multiplier,
        )
        results = [
            simulate_stage0(
                frame,
                signals[key],
                hypothesis=hypothesis,
                dataset=datasets[key],
                parameters=parameters,
                costs=selected_costs,
                execution_delay_bars=delay,
            )
            for key, frame in frames.items()
        ]
        trades = [trade for result in results for trade in result.trades]
        output.append(
            {
                "scenario": name,
                "execution_delay_bars": delay,
                "spread_multiplier": spread_multiplier,
                "slippage_multiplier": slippage_multiplier,
                "trade_count": len(trades),
                "net_pnl_eur": float(sum(row.net_pnl_eur for row in trades)),
                "net_expectancy_eur": (
                    statistics.fmean(row.net_pnl_eur for row in trades)
                    if trades
                    else None
                ),
                "approximation_authority": "RESEARCH_ONLY",
            }
        )
    return output


def strategy_result_correlation(
    results: Sequence[Stage0Result],
) -> list[dict[str, Any]]:
    """Correlate daily Stage-0 PnL paths without assuming independent alpha."""

    by_parameter: dict[str, dict[pd.Timestamp, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for result in results:
        for trade in result.trades:
            day = pd.Timestamp(trade.exit_timestamp).floor("1d")
            by_parameter[result.parameter_hash][day] += trade.net_pnl_eur
    hashes = sorted(by_parameter)
    output: list[dict[str, Any]] = []
    for index, first in enumerate(hashes):
        for second in hashes[index + 1 :]:
            dates = sorted(set(by_parameter[first]) | set(by_parameter[second]))
            correlation: float | None = None
            if len(dates) >= 10:
                left = np.asarray([by_parameter[first].get(day, 0.0) for day in dates])
                right = np.asarray([by_parameter[second].get(day, 0.0) for day in dates])
                if float(left.std()) > 0 and float(right.std()) > 0:
                    correlation = float(np.corrcoef(left, right)[0, 1])
            output.append(
                {
                    "parameter_hash_a": first,
                    "parameter_hash_b": second,
                    "daily_pnl_correlation": correlation,
                    "observation_days": len(dates),
                    "duplicate_alpha_warning": (
                        correlation is not None and correlation >= 0.90
                    ),
                }
            )
    return output


def _serializable_exact_result(result: Any) -> dict[str, Any]:
    return {
        "strategy_id": result.strategy_id,
        "initial_cash_eur": result.initial_cash_eur,
        "ending_equity_eur": result.ending_equity_eur,
        "trade_count": len(result.trades),
        "order_count": len(result.orders),
        "metrics": dict(result.metrics),
        "integrity": dict(result.integrity),
    }


def run_exact_rejection_review(
    raw_frames: Mapping[str, pd.DataFrame],
    *,
    parameters: Mapping[str, Any],
    settings: Settings,
    stage0_parameter_hash: str,
) -> dict[str, Any]:
    """Audit one top rejection in the exact engine without allowing promotion."""

    strategy = FailedBreakdownReversalResearchAdapter()
    features = {
        market: FeaturePipeline().build(frame.copy(), market=market)
        for market, frame in raw_frames.items()
    }
    config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=2_000.0),
        bootstrap_samples=100,
        monte_carlo_runs=100,
    )
    result = BacktestEngine(config, settings=settings).run(
        features,
        strategy,
        parameters=dict(parameters),
    )
    exact_positive = (
        int(result.metrics.get("trade_count") or 0) >= 30
        and _safe_float(result.metrics.get("net_expectancy_r")) > 0
        and _safe_float(result.metrics.get("profit_factor")) >= 1.0
    )
    return {
        "sample_type": "TOP_RANKED_STAGE0_REJECTION",
        "sample_size": 1,
        "parameter_hash": stage0_parameter_hash,
        "development_data_only": True,
        "final_test_accessed": False,
        "promotion_authority": False,
        "exact_result": _serializable_exact_result(result),
        "false_negative_detected": exact_positive,
        "interpretation": (
            "STAGE0_FALSE_NEGATIVE_REQUIRES_GATE_REVIEW"
            if exact_positive
            else "REJECTION_DIRECTION_CONFIRMED_IN_SAMPLE"
        ),
    }


def run_exact_candidate_validation(
    raw_frames: Mapping[str, pd.DataFrame],
    *,
    parameters: Mapping[str, Any],
    settings: Settings,
    purge_bars: int,
    embargo_bars: int,
) -> dict[str, Any]:
    """Use the existing exact engine and its train/validation/test governance."""

    strategy = FailedBreakdownReversalResearchAdapter()
    features: dict[str, pd.DataFrame] = {}
    for market, frame in raw_frames.items():
        selected = frame.copy()
        selected.attrs.update(frame.attrs)
        features[market] = FeaturePipeline().build(selected, market=market)
    lookahead = all(
        strategy_lookahead_test(frame, strategy, dict(parameters))
        for frame in features.values()
    )
    repainting = all(
        strategy_repainting_test(frame, strategy, dict(parameters))
        for frame in features.values()
    )
    if not lookahead or not repainting:
        return {
            "status": "HARD_REJECT",
            "rejection_reasons": [RejectionReason.LOOKAHEAD_BIAS],
            "lookahead_safe": lookahead,
            "repainting_safe": repainting,
        }
    train, validation, holdout = chronological_split(
        features,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    del train  # Parameters are already frozen by Stage 0 before final-test use.
    config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=2_000.0),
        bootstrap_samples=100,
        monte_carlo_runs=100,
    )
    normal = BacktestEngine(config, settings=settings).run(
        validation,
        strategy,
        parameters=dict(parameters),
    )
    final = BacktestEngine(config, settings=settings).run(
        holdout,
        strategy,
        parameters=dict(parameters),
    )
    exact_cost_stress: list[dict[str, Any]] = []
    exact_stress_results: dict[float, Any] = {0.0: final}
    for additional_roundtrip_bps in (0.0, 10.0, 25.0, 50.0):
        if additional_roundtrip_bps == 0:
            stressed_result = final
        else:
            stressed_config = replace(
                config,
                costs=CostModel(
                    fee_fraction=config.costs.fee_fraction,
                    slippage_bps=(
                        config.costs.slippage_bps + additional_roundtrip_bps / 2.0
                    ),
                    spread_bps=config.costs.spread_bps,
                    multiplier=1.0,
                ),
            )
            stressed_result = BacktestEngine(stressed_config, settings=settings).run(
                holdout,
                strategy,
                parameters=dict(parameters),
            )
            exact_stress_results[additional_roundtrip_bps] = stressed_result
        exact_cost_stress.append(
            {
                "scenario": (
                    "BASE"
                    if additional_roundtrip_bps == 0
                    else f"BASE_PLUS_{int(additional_roundtrip_bps)}_BPS"
                ),
                "additional_roundtrip_bps": additional_roundtrip_bps,
                **_serializable_exact_result(stressed_result),
            }
        )
    stressed = exact_stress_results[25.0]
    walk_forward = walk_forward_optimize(
        {market: frame.iloc[: int(len(frame) * 0.80)].copy() for market, frame in features.items()},
        strategy,
        config,
        folds=3,
        mode="anchored",
        search_method="random",
        search_trials=20,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        settings=settings,
        minimum_trades=10,
    )
    trade_returns = [
        float(trade.net_pnl_eur)
        / max(1e-9, float(trade.entry_price * trade.quantity))
        for trade in final.trades
    ]
    from research.stochastic_validation import StochasticValidationPolicy

    monte_carlo = stationary_bootstrap_monte_carlo(
        trade_returns,
        policy=StochasticValidationPolicy(
            simulations=1_000,
            expected_block_length=5,
            maximum_drawdown=settings.research.maximum_drawdown,
            maximum_drawdown_breach_probability=(
                settings.research.maximum_monte_carlo_probability_of_20pct_drawdown
            ),
            maximum_terminal_loss_probability=(
                settings.research.maximum_dirichlet_probability_of_loss
            ),
            minimum_p05_total_return=settings.research.minimum_stochastic_p05_total_return,
            minimum_observations=30,
            confidence_level=settings.research.confidence_level,
            seed=settings.app.random_seed,
        ),
    )
    asset_holdout = {}
    for market, frame in holdout.items():
        result = BacktestEngine(config, settings=settings).run(
            {market: frame},
            strategy,
            parameters=dict(parameters),
        )
        asset_holdout[market] = _serializable_exact_result(result)
    final_positive = (
        int(final.metrics.get("trade_count") or 0) >= settings.research.minimum_trades
        and _safe_float(final.metrics.get("net_expectancy_r")) > 0
        and _safe_float(final.metrics.get("profit_factor")) >= settings.research.minimum_profit_factor
    )
    stress_positive = (
        _safe_float(stressed.metrics.get("net_expectancy_r")) >= 0
        and _safe_float(stressed.metrics.get("profit_factor"))
        >= settings.research.minimum_stressed_profit_factor
    )
    walk_positive = (
        walk_forward.valid
        and walk_forward.positive_folds >= min(3, settings.research.minimum_positive_folds)
    )
    return {
        "status": (
            "ROBUST_EXACT_PASS"
            if final_positive and stress_positive and walk_positive and monte_carlo.get("passed")
            else "EXACT_REJECTED"
        ),
        "parameters_frozen_before_final_test": True,
        "final_test_retuned": False,
        "lookahead_safe": lookahead,
        "repainting_safe": repainting,
        "validation": _serializable_exact_result(normal),
        "final_test": _serializable_exact_result(final),
        "stressed_final_test": _serializable_exact_result(stressed),
        "exact_cost_stress": exact_cost_stress,
        "cost_failure_threshold": next(
            (
                row["additional_roundtrip_bps"]
                for row in exact_cost_stress
                if _safe_float(row["metrics"].get("net_expectancy_r")) <= 0
            ),
            None,
        ),
        "walk_forward": asdict(walk_forward),
        "walk_forward_selection": (
            "PARAMETERS_SELECTED_ON_EACH_TRAIN_FOLD_AND_FROZEN_ON_NEXT_FOLD"
        ),
        "asset_holdout": asset_holdout,
        "regime_holdout": {
            "status": "NOT_EVALUABLE_NO_CAUSAL_REGIME_LABELS_IN_SELECTED_DATASETS"
        },
        "monte_carlo": monte_carlo,
        "exact_engine_authoritative": True,
        "stage0_economics_used_as_authority": False,
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _p0_5_branch(artifact: Mapping[str, Any]) -> dict[str, Any]:
    families = list(artifact.get("family_results") or [])
    gross_negative = [
        str(row.get("dimension_value"))
        for row in families
        if row.get("cost_classification") == "GROSS_NEGATIVE"
    ]
    gross_positive_net_negative = [
        str(row.get("dimension_value"))
        for row in families
        if row.get("cost_classification") == "GROSS_POSITIVE_NET_NEGATIVE"
    ]
    promising = [
        str(row.get("dimension_value"))
        for row in families
        if _safe_float(row.get("net_pnl_eur")) > 0
        and _safe_float(row.get("net_expectancy_eur")) > 0
    ]
    return {
        "decision": (
            "ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION"
            if len(gross_negative) >= max(1, len(families) // 2)
            else "PROMISING_EXISTING_ALPHA"
        ),
        "gross_negative_families": gross_negative,
        "gross_positive_net_negative_families": gross_positive_net_negative,
        "promising_but_not_validated_families": promising,
        "broad_parameter_rescue_allowed": False,
        "live_validated_family_count": 0,
    }


REFERENCE_CONCEPTS = (
    {
        "reference_repository": "crypto-references/vectorbt",
        "reference_file": "vectorbt/portfolio/base.py",
        "class_or_function": "Portfolio.from_signals",
        "concept": "broadcastable signal/parameter arrays for fast approximate screening",
        "our_native_implementation": "generate_parameter_grid + simulate_stage0",
        "why_it_fits": "rejects many combinations without replacing exact execution",
    },
    {
        "reference_repository": "crypto-references/pybroker",
        "reference_file": "src/pybroker/strategy.py",
        "class_or_function": "WalkforwardMixin.walkforward_split",
        "concept": "chronological windows with explicit lookahead separation",
        "our_native_implementation": "WalkForwardManifest + existing chronological_split",
        "why_it_fits": "makes purge, embargo and final-test boundaries immutable",
    },
    {
        "reference_repository": "crypto-references/freqtrade",
        "reference_file": "freqtrade/optimize/analysis/lookahead.py",
        "class_or_function": "LookaheadAnalysis",
        "concept": "compare full and truncated calculations for future-data bias",
        "our_native_implementation": "stage0_causality_check + static_lookahead_audit",
        "why_it_fits": "hard-rejects causal violations before exact validation",
    },
    {
        "reference_repository": "crypto-references/freqtrade",
        "reference_file": "freqtrade/optimize/analysis/recursive.py",
        "class_or_function": "RecursiveAnalysis",
        "concept": "indicator sensitivity to startup history",
        "our_native_implementation": "recursive_warmup_stability",
        "why_it_fits": "requires explicit stable warmup semantics",
    },
    {
        "reference_repository": "crypto-references/qlib",
        "reference_file": "qlib/workflow/recorder.py",
        "class_or_function": "Recorder",
        "concept": "immutable experiment identity, parameters, metrics and artifacts",
        "our_native_implementation": "DatasetIdentity + ExperimentContract + immutable run artifact",
        "why_it_fits": "reproducible local research without ML authority",
    },
    {
        "reference_repository": "crypto-references/lean",
        "reference_file": "Algorithm.Framework/",
        "class_or_function": "Alpha/Portfolio/Execution boundaries",
        "concept": "separate research signals from portfolio and execution authority",
        "our_native_implementation": "Stage0 authority invariant and exact-engine routing",
        "why_it_fits": "prevents approximate screening from placing or approving trades",
    },
    {
        "reference_repository": "crypto-references/nautilus_trader",
        "reference_file": "python/nautilus_trader/backtest/__init__.pyi",
        "class_or_function": "BacktestRunConfig / BacktestEngineConfig",
        "concept": "one explicit run configuration preserving semantic boundaries",
        "our_native_implementation": "ExperimentContract + WalkForwardManifest",
        "why_it_fits": "keeps exact native backtesting reproducible and authoritative",
    },
)


def _default_dataset_specs(settings: Settings) -> list[dict[str, Any]]:
    markets = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "ADA-EUR")
    timeframes = ("15m", "1h")
    return [
        {
            "path": settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet",
            "provider": "bitvavo",
            "market": market,
            "timeframe": timeframe,
        }
        for market in markets
        for timeframe in timeframes
    ]


def _strategy_data_coverage(
    p0_5: Mapping[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    episodes = [
        row
        for row in p0_5.get("episodes") or []
        if row.get("strategy_id") == "FAILED_BREAKDOWN_REVERSAL_V1"
    ]
    markets = sorted({str(row.get("market")) for row in episodes})
    rows = []
    for market in markets:
        available = [
            timeframe
            for timeframe in ("15m", "1h")
            if (
                settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            ).is_file()
        ]
        rows.append(
            {
                "market": market,
                "canonical_paper_episode_count": sum(
                    row.get("market") == market for row in episodes
                ),
                "available_research_timeframes": available,
                "status": "READY" if available else "MISSING_REQUIRED_OHLCV",
            }
        )
    return {
        "canonical_paper_episode_count": len(episodes),
        "canonical_paper_markets": markets,
        "market_coverage": rows,
        "exact_same_population_research_status": (
            "READY"
            if rows and all(row["status"] == "READY" for row in rows)
            else "NOT_EVALUABLE_MISSING_HISTORICAL_DATA"
        ),
    }


def _exact_timeframe_choice(
    results: Sequence[Stage0Result],
    parameter_hash: str,
) -> str | None:
    totals: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for row in results:
        if row.parameter_hash != parameter_hash:
            continue
        totals[row.timeframe] += row.net_pnl_eur
        counts[row.timeframe] += row.trade_count
    if not totals:
        return None
    return max(sorted(totals), key=lambda value: (totals[value], counts[value], value))


def _experiment_contracts(
    results: Sequence[Stage0Result],
    datasets: Mapping[str, DatasetIdentity],
    *,
    run_id: str,
    hypothesis: Stage0Hypothesis,
    manifest_id: str,
    code_commit: str | None,
    created_at: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        dataset = datasets[result.dataset_id]
        contract = ExperimentContract(
            run_id=run_id,
            experiment_id=result.experiment_id,
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            strategy_family=hypothesis.strategy_family,
            strategy_implementation=hypothesis.strategy_implementation,
            strategy_version=hypothesis.strategy_version,
            parameter_set=dict(result.parameter_set),
            asset_universe="P1_BOUNDED_CORE_EUR_SPOT_V1",
            assets=(result.market,),
            timeframe=result.timeframe,
            data_version=dataset.dataset_id,
            data_cutoff=dataset.data_cutoff,
            feature_schema_version="causal_ohlcv_structure_v1",
            cost_model_version=result.cost_model_version,
            universe_version="core_eur_spot_pit_partial_v1",
            signal_version=STAGE0_SCHEMA_VERSION,
            validation_manifest_id=manifest_id,
            code_commit_hash=code_commit,
            created_at=created_at,
            candidate_origin=hypothesis.candidate_origin,
            final_test_generation=1,
        )
        output.append(contract.to_dict())
    return output


def build_research_factory_artifact(
    settings: Settings,
    *,
    dataset_specs: Sequence[Mapping[str, Any]] | None = None,
    maximum_rows: int = 20_000,
    execute_exact: bool = True,
) -> dict[str, Any]:
    """Run the bounded P1 campaign and persist one immutable evidence artifact."""

    started = time.perf_counter()
    latest_economics = _read_json(settings.paths.output_dir / "economics" / "latest.json")
    p0_5_path = Path(str(latest_economics.get("artifact_path") or ""))
    if not p0_5_path.is_file():
        raise FileNotFoundError("latest immutable P0.5 artifact is unavailable")
    p0_5_bytes = p0_5_path.read_bytes()
    p0_5 = json.loads(p0_5_bytes)
    p0_path = settings.paths.output_dir / "live" / "event_driven_execution_state.json"
    p0_bytes = p0_path.read_bytes() if p0_path.is_file() else b""
    branch = _p0_5_branch(p0_5)
    if branch["decision"] != (
        "ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION"
    ):
        raise ValueError("P0.5 branch changed; bounded reset campaign is no longer valid")
    costs = SharedCostModel.from_settings(settings)
    promising_family = next(
        (
            row
            for row in p0_5.get("family_results") or []
            if row.get("dimension_value") == "FAILED_BREAKDOWN_REVERSAL"
        ),
        {},
    )
    promising_sample = int(promising_family.get("closed_episode_count") or 0)
    promising_profit_factor = _safe_float(promising_family.get("profit_factor"))
    hypothesis = Stage0Hypothesis(
        hypothesis_id="failed_breakdown_structural_falsification_v1",
        strategy_family="FAILED_BREAKDOWN_REVERSAL",
        strategy_implementation="FAILED_BREAKDOWN_REVERSAL_V1_RESEARCH_ADAPTER",
        strategy_version="1.0.0-research",
        candidate_origin="STRUCTURAL_VARIANT",
        rationale=(
            "The current immutable P0.5 artifact found FAILED_BREAKDOWN_REVERSAL "
            f"positive but not validated at N={promising_sample}/"
            f"PF={promising_profit_factor:.3f}; test a causal OHLCV-only surrogate "
            "on core assets without inheriting the paper evidence."
        ),
        required_inputs=("open", "high", "low", "close", "volume"),
        optional_inputs=("causal_orderflow", "causal_regime"),
        parameter_space=FailedBreakdownReversalResearchAdapter.parameter_space,
        supported_timeframes=("15m", "1h"),
        side="LONG_ONLY",
        holding_semantics="SIGNAL_AT_CLOSE_ENTRY_NEXT_OPEN_CONSERVATIVE_STOP_FIRST",
        p0_5_classification="PROMISING_BUT_INSUFFICIENT_SAMPLE",
    )
    specs = list(dataset_specs or _default_dataset_specs(settings))
    full_frames: dict[str, pd.DataFrame] = {}
    full_identities: dict[str, DatasetIdentity] = {}
    missing_sources: list[dict[str, Any]] = []
    for raw in specs:
        path = Path(raw["path"])
        key = f"{raw['market']}:{raw['timeframe']}"
        if not path.is_file():
            missing_sources.append(
                {
                    "market": str(raw["market"]),
                    "timeframe": str(raw["timeframe"]),
                    "path": str(path),
                    "status": "MISSING_REQUIRED_DATA",
                }
            )
            continue
        frame, identity = load_immutable_ohlcv(
            path,
            provider=str(raw.get("provider") or "unknown"),
            market=str(raw["market"]),
            timeframe=str(raw["timeframe"]),
            maximum_rows=maximum_rows,
        )
        full_frames[key] = frame
        full_identities[key] = identity
    if not full_frames:
        raise ValueError("no Stage-0 datasets are available")
    scheduler_items: list[dict[str, Any]] = [
        {
            "backlog_id": hypothesis.hypothesis_id,
            "strategy_family": hypothesis.strategy_family,
            "candidate_origin": hypothesis.candidate_origin,
            "p0_5_classification": "PROMISING",
            "data_availability": 1.0,
            "sample_availability": 0.25,
            "diversification_potential": 0.5,
            "validation_cost": 0.25,
            "already_disproven": False,
            "campaign_selected": True,
        }
    ]
    for family in p0_5.get("family_results") or []:
        classification = str(family.get("cost_classification") or "INSUFFICIENT_SAMPLE")
        scheduler_items.append(
            {
                "backlog_id": f"p0.5-family:{family.get('dimension_value')}",
                "strategy_family": str(family.get("dimension_value") or "UNKNOWN"),
                "candidate_origin": "EXISTING_STRATEGY",
                "p0_5_classification": classification,
                "data_availability": 0.5,
                "sample_availability": min(
                    1.0, int(family.get("closed_episode_count") or 0) / 100.0
                ),
                "diversification_potential": 0.25,
                "validation_cost": 0.75,
                "already_disproven": classification == "GROSS_NEGATIVE",
                "campaign_selected": False,
            }
        )
    research_scheduler = prioritize_research_backlog(scheduler_items)
    development_frames: dict[str, pd.DataFrame] = {}
    development_identities: dict[str, DatasetIdentity] = {}
    for key, frame in full_frames.items():
        cutoff = max(100, int(len(frame) * 0.80))
        development = frame.iloc[:cutoff].copy()
        development.attrs.update(frame.attrs)
        identity = derive_dataset_identity(
            development,
            full_identities[key],
            purpose="TRAIN_VALIDATION_ONLY_EXCLUDES_FINAL_20_PERCENT",
        )
        development.attrs["dataset_id"] = identity.dataset_id
        development_frames[key] = development
        development_identities[key] = identity
    default_parameters = FailedBreakdownReversalResearchAdapter().parameters()
    sample_key = sorted(development_frames)[0]
    bias = {
        "dynamic_causality": stage0_causality_check(
            development_frames[sample_key],
            failed_breakdown_reversal_signals,
            default_parameters,
        ),
        "static_source_audit": static_lookahead_audit(
            failed_breakdown_reversal_signals
        ),
        "recursive_warmup": recursive_warmup_stability(
            development_frames[sample_key],
            failed_breakdown_reversal_signals,
            default_parameters,
        ),
        "higher_timeframe_incomplete_candle_use": False,
        "future_universe_membership_use": False,
        "future_regime_use": False,
    }
    hard_bias_failure = (
        bias["dynamic_causality"]["status"] != "PASSED"
        or bias["static_source_audit"]["status"] != "PASSED"
        or bias["recursive_warmup"]["status"] == "FAILED"
    )
    grid = generate_parameter_grid(hypothesis.parameter_space)
    baseline_results: list[Stage0Result] = []
    if not hard_bias_failure:
        for parameters in grid:
            selected_parameters = {
                **parameters,
                "maximum_holding_bars": 32,
                "trailing_atr": 0.0,
            }
            for key, frame in development_frames.items():
                signals = failed_breakdown_reversal_signals(
                    frame,
                    selected_parameters,
                )
                baseline_results.append(
                    simulate_stage0(
                        frame,
                        signals,
                        hypothesis=hypothesis,
                        dataset=development_identities[key],
                        parameters=selected_parameters,
                        costs=costs,
                        minimum_trades=30,
                    )
                )
    stage0_elapsed = time.perf_counter() - started
    aggregate = aggregate_stage0_by_parameter(baseline_results)
    plateaus = parameter_plateaus(aggregate, hypothesis.parameter_space)
    plateau_by_hash = {row.parameter_hash: row for row in plateaus}
    rejection_table: list[dict[str, Any]] = []
    preliminary: list[dict[str, Any]] = []
    for row in aggregate:
        reasons = _aggregate_rejection_reasons(row)
        plateau = plateau_by_hash[str(row["parameter_hash"])]
        if not plateau.stable:
            reasons.append(str(RejectionReason.PARAMETER_FRAGILE))
        selected = {
            **dict(row),
            "plateau": asdict(plateau),
            "rejection_reasons": list(dict.fromkeys(reasons)),
        }
        if reasons:
            rejection_table.append(selected)
        else:
            preliminary.append(selected)
    stress_targets = (preliminary or aggregate[:1])[:10]
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    cost_survivors: list[dict[str, Any]] = []
    for candidate in stress_targets:
        parameter_hash = str(candidate["parameter_hash"])
        rows = stage0_cost_stress(
            frames=development_frames,
            datasets=development_identities,
            hypothesis=hypothesis,
            builder=failed_breakdown_reversal_signals,
            parameters=dict(candidate["parameter_set"]),
            costs=costs,
        )
        cost_stress[parameter_hash] = rows
        plus_ten = next(row for row in rows if row["scenario"] == "BASE_PLUS_10_BPS")
        if candidate in preliminary and _safe_float(plus_ten["net_expectancy_eur"]) > 0:
            cost_survivors.append(candidate)
        elif candidate in preliminary:
            rejected = {
                **candidate,
                "rejection_reasons": [str(RejectionReason.COST_FRAGILE)],
            }
            rejection_table.append(rejected)
    survivors = sorted(
        cost_survivors,
        key=lambda row: (
            _safe_float(row.get("net_expectancy_eur")),
            _safe_float(row.get("net_pnl_eur")),
        ),
        reverse=True,
    )[:3]
    robustness_target = survivors[0] if survivors else (aggregate[0] if aggregate else None)
    delay_liquidity = (
        stage0_delay_liquidity_stress(
            frames=development_frames,
            datasets=development_identities,
            hypothesis=hypothesis,
            builder=failed_breakdown_reversal_signals,
            parameters=dict(robustness_target["parameter_set"]),
            costs=costs,
        )
        if robustness_target is not None
        else []
    )
    timeframe_rows: list[dict[str, Any]] = []
    if robustness_target is not None:
        target_hash = str(robustness_target["parameter_hash"])
        for timeframe in sorted({row.timeframe for row in baseline_results}):
            selected_rows = [
                row
                for row in baseline_results
                if row.parameter_hash == target_hash and row.timeframe == timeframe
            ]
            aggregate_timeframe = aggregate_stage0_by_parameter(selected_rows)
            if aggregate_timeframe:
                timeframe_rows.append(
                    {"timeframe": timeframe, **aggregate_timeframe[0]}
                )
    selected_timeframe = (
        _exact_timeframe_choice(
            baseline_results,
            str(robustness_target["parameter_hash"]),
        )
        if robustness_target is not None
        else sorted({row.timeframe for row in full_identities.values()})[0]
    )
    code_commit = _git_commit(settings.paths.project_root)
    implementation_source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    code_version = (
        f"{code_commit}:{implementation_source_hash}"
        if code_commit
        else implementation_source_hash
    )
    manifest_datasets = [
        row for row in full_identities.values() if row.timeframe == selected_timeframe
    ]
    manifest = build_walk_forward_manifest(
        manifest_datasets,
        strategy_id=hypothesis.strategy_implementation,
        parameter_search_scope={
            name: list(values) for name, values in hypothesis.parameter_space.items()
        },
        timeframe=str(selected_timeframe),
        costs=costs,
        code_version=code_version,
        folds=3,
        purge_bars=40,
        embargo_bars=2,
        asset_holdout=(manifest_datasets[-1].market,) if len(manifest_datasets) > 1 else (),
    )
    exact_result: dict[str, Any] = {
        "status": "NOT_RUN_NO_STAGE0_SURVIVOR",
        "exact_engine_authoritative": True,
    }
    false_negative_review: dict[str, Any] = {
        "sample_type": "TOP_RANKED_STAGE0_REJECTION",
        "sample_size": 0,
        "status": "NOT_RUN_NO_REJECTION_OR_STAGE0_ONLY_MODE",
        "promotion_authority": False,
    }
    if aggregate and execute_exact:
        review_target = aggregate[0]
        review_frames = {
            identity.market: development_frames[key]
            for key, identity in development_identities.items()
            if identity.timeframe == selected_timeframe
        }
        try:
            false_negative_review = run_exact_rejection_review(
                review_frames,
                parameters=dict(review_target["parameter_set"]),
                settings=settings,
                stage0_parameter_hash=str(review_target["parameter_hash"]),
            )
        except (ValueError, PermissionError, ArithmeticError) as exc:
            false_negative_review = {
                "sample_type": "TOP_RANKED_STAGE0_REJECTION",
                "sample_size": 1,
                "status": "EXACT_REVIEW_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "promotion_authority": False,
            }
    if survivors and execute_exact:
        selected = survivors[0]
        exact_frames = {
            identity.market: full_frames[key]
            for key, identity in full_identities.items()
            if identity.timeframe == selected_timeframe
        }
        try:
            exact_result = run_exact_candidate_validation(
                exact_frames,
                parameters=dict(selected["parameter_set"]),
                settings=settings,
                purge_bars=manifest.purge_period_bars,
                embargo_bars=manifest.embargo_period_bars,
            )
        except (ValueError, PermissionError, ArithmeticError) as exc:
            exact_result = {
                "status": "EXACT_VALIDATION_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exact_engine_authoritative": True,
            }
    promotion_state = promotion_state_for(
        stage0_survivor_count=len(survivors),
        exact_status=exact_result.get("status"),
    )
    forward_candidates = (
        [
            {
                "strategy_id": hypothesis.strategy_implementation,
                "strategy_version": hypothesis.strategy_version,
                "parameter_set": dict(survivors[0]["parameter_set"]),
                "parameter_hash": survivors[0]["parameter_hash"],
                "market_context": "TO_BE_CAPTURED_PROSPECTIVELY",
                "cost_prediction_model": costs.cost_model_version,
                "canonical_outcome_source": "P0_CANONICAL_EXECUTION_STATE",
                "authority": "FORWARD_OBSERVATION_ONLY",
                "future_information": False,
            }
        ]
        if promotion_state == PromotionState.FORWARD_CANDIDATE
        else []
    )
    correlation_source_hashes = {
        str(row["parameter_hash"])
        for row in (survivors or aggregate[:5])
    }
    correlations = strategy_result_correlation(
        [row for row in baseline_results if row.parameter_hash in correlation_source_hashes]
    )
    duplicate_warnings = [row for row in correlations if row["duplicate_alpha_warning"]]
    data_version = stable_hash(
        sorted(row.dataset_id for row in development_identities.values()),
        length=40,
    )
    source_hashes = {
        "p0_current_canonical_read_model_sha256": (
            sha256(p0_bytes).hexdigest() if p0_bytes else None
        ),
        "p0_5_immutable_artifact_sha256": sha256(p0_5_bytes).hexdigest(),
        "datasets": {
            key: identity.source_file_hash
            for key, identity in sorted(full_identities.items())
        },
        "research_factory_source_sha256": implementation_source_hash,
    }
    run_identity = {
        "schema": RESEARCH_FACTORY_SCHEMA_VERSION,
        "p0_5_hash": source_hashes["p0_5_immutable_artifact_sha256"],
        "p0_hash": source_hashes["p0_current_canonical_read_model_sha256"],
        "datasets": [
            row.dataset_id for row in sorted(full_identities.values(), key=lambda value: value.dataset_id)
        ],
        "hypothesis": asdict(hypothesis),
        "cost_model": asdict(costs),
        "manifest": manifest.manifest_id,
        "code_version": code_version,
    }
    run_id = stable_hash(run_identity, length=32)
    created_at = utc_iso()
    experiment_created_at = max(
        identity.data_cutoff for identity in development_identities.values()
    )
    datasets_by_id = {
        row.dataset_id: row for row in development_identities.values()
    }
    experiment_contracts = _experiment_contracts(
        baseline_results,
        datasets_by_id,
        run_id=run_id,
        hypothesis=hypothesis,
        manifest_id=manifest.manifest_id,
        code_commit=code_commit,
        created_at=experiment_created_at,
    )
    elapsed = time.perf_counter() - started
    total_variants = len(baseline_results)
    status_counts = Counter(
        "STAGE0_PROMISING"
        if row["parameter_hash"] in {str(value["parameter_hash"]) for value in survivors}
        else "STAGE0_REJECTED"
        for row in aggregate
    )
    validation_backlog_before = int(p0_5.get("validation_backlog_count") or 0)
    payload: dict[str, Any] = {
        "schema_version": RESEARCH_FACTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "code_commit_hash": code_commit,
        "research_factory_source_sha256": implementation_source_hash,
        "p0_baseline": {
            "canonical_read_model_path": str(p0_path.resolve()),
            "canonical_state_version": p0_5.get("canonical_state_version"),
            "canonical_state_hash": p0_5.get("canonical_state_hash"),
            "deterministic_replay": p0_5.get("replay_deterministic") is True,
        },
        "p0_5_baseline": {
            "artifact_path": str(p0_5_path.resolve()),
            "artifact_hash": p0_5.get("artifact_hash"),
            "closed_episode_count": p0_5.get("closed_episode_count"),
            "net_pnl_eur": (p0_5.get("aggregate") or {}).get("net_pnl_eur"),
            "profit_factor": (p0_5.get("aggregate") or {}).get("profit_factor"),
            "live_validated_family_count": 0,
        },
        "p0_5_branch": branch,
        "strategy_data_coverage": _strategy_data_coverage(p0_5, settings),
        "hypothesis": asdict(hypothesis),
        "strategy_generation_freeze": {
            "active": True,
            "new_strategy_count": 0,
            "research_adapter_registration_count": 0,
            "candidate_origin": hypothesis.candidate_origin,
        },
        "research_scheduler": {
            "policy": "P0.5_PROMISE_DATA_DIVERSIFICATION_COST_NON_FIFO_V1",
            "fifo": False,
            "items": research_scheduler,
            "selected_backlog_ids": [
                row["backlog_id"] for row in research_scheduler if row["campaign_selected"]
            ],
            "gross_negative_parameter_rescue_allowed": False,
        },
        "shared_cost_model": asdict(costs),
        "datasets": [
            row.to_dict()
            for row in sorted(full_identities.values(), key=lambda value: (value.market, value.timeframe))
        ],
        "development_datasets": [
            row.to_dict()
            for row in sorted(
                development_identities.values(),
                key=lambda value: (value.market, value.timeframe),
            )
        ],
        "missing_dataset_sources": missing_sources,
        "data_version": data_version,
        "data_quality_gate": {
            "invalid_dataset_count": sum(
                row.quality.status != "READY" for row in full_identities.values()
            ),
            "pit_universe_status": "PIT_UNIVERSE_PARTIAL",
            "results_with_critical_data_failure_are_strategy_failures": False,
        },
        "bias_checks": bias,
        "stage0": {
            "schema_version": STAGE0_SCHEMA_VERSION,
            "authority": "APPROXIMATE_RESEARCH_ONLY",
            "live_eligible": False,
            "paper_promoted": False,
            "parameter_combination_count": len(grid),
            "dataset_count": len(development_frames),
            "tested_variant_count": total_variants,
            "results": [row.to_dict() for row in baseline_results],
            "aggregate_parameter_results": aggregate,
            "plateau_results": [asdict(row) for row in plateaus],
            "survivors": survivors,
            "status_counts": dict(status_counts),
        },
        "multiple_testing": multiple_testing_accounting(
            hypotheses=1,
            parameter_combinations=len(grid),
            assets=len({row.market for row in development_identities.values()}),
            timeframes=len({row.timeframe for row in development_identities.values()}),
            regimes=1,
            cost_scenarios=4,
        ),
        "cost_stress": cost_stress,
        "execution_delay_liquidity_stress": delay_liquidity,
        "validation_manifest": manifest.to_dict(),
        "exact_validation": exact_result,
        "exact_backtester_authority": {
            "engine": "research.backtest.BacktestEngine",
            "authoritative": True,
            "stage0_can_override_exact": False,
        },
        "asset_holdout": exact_result.get("asset_holdout") or {
            "status": "NOT_RUN_NO_EXACT_SURVIVOR"
        },
        "regime_holdout": exact_result.get("regime_holdout") or {
            "status": "NOT_EVALUABLE_NO_CAUSAL_REGIME_LABELS"
        },
        "timeframe_robustness": {
            "tested_timeframes": sorted(
                {row.timeframe for row in development_identities.values()}
            ),
            "classification": (
                "NO_POSITIVE_TIMEFRAME"
                if timeframe_rows
                and not any(_safe_float(row.get("net_expectancy_eur")) > 0 for row in timeframe_rows)
                else (
                    "NOT_EVALUABLE_NO_RESULTS"
                    if not timeframe_rows
                    else "REQUIRES_EXACT_CONFIRMATION"
                )
            ),
            "stage0_top_rejection_rows": timeframe_rows,
        },
        "monte_carlo_bootstrap": exact_result.get("monte_carlo") or {
            "status": "NOT_RUN_NO_WALK_FORWARD_SURVIVOR"
        },
        "strategy_correlation": {
            "rows": correlations,
            "duplicate_alpha_warning_count": len(duplicate_warnings),
            "promotion_impact": "HARD_REJECT_DUPLICATE_ALPHA" if duplicate_warnings else "NONE",
        },
        "incremental_portfolio_value": {
            "status": (
                "NOT_EVALUABLE_NO_ROBUST_EXACT_CANDIDATE"
                if promotion_state != PromotionState.FORWARD_CANDIDATE
                else "RESEARCH_ONLY_REQUIRES_SEPARATE_EXISTING_PORTFOLIO_PATH"
            ),
            "allocator_built": False,
        },
        "promotion_table": [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "strategy_family": hypothesis.strategy_family,
                "promotion_state": str(promotion_state),
                "stage0_survivor_count": len(survivors),
                "exact_status": exact_result.get("status"),
                "live_validated": False,
                "automatic_authority": False,
            }
        ],
        "rejection_table": rejection_table,
        "backlog": {
            "inventory_backlog_before": validation_backlog_before,
            "inventory_backlog_after": validation_backlog_before,
            "inventory_claimed_as_screened": False,
            "scoped_exact_jobs_before_stage0": total_variants,
            "scoped_exact_jobs_after_stage0": len(survivors),
            "scoped_reduction_count": max(0, total_variants - len(survivors)),
            "scoped_rejection_fraction": (
                (total_variants - len(survivors)) / total_variants
                if total_variants
                else None
            ),
            "deferred_inventory_reason": (
                "P0.5_ALPHA_RESET_BRANCH_AND_MISSING_UNIFORM_DATA_CONTRACTS; "
                "UNSCREENED_ITEMS_REMAIN_VISIBLE"
            ),
        },
        "compute_budget": {
            "CHEAP_STAGE0": total_variants,
            "MEDIUM_EXACT": len(survivors),
            "AUDIT_EXACT_REJECTION_REVIEW": int(
                int(false_negative_review.get("sample_size") or 0) > 0
            ),
            "EXPENSIVE_ROBUSTNESS": int(exact_result.get("status") not in {None, "NOT_RUN_NO_STAGE0_SURVIVOR"}),
            "VERY_EXPENSIVE_FORWARD": len(forward_candidates),
            "parallel_shared_mutable_state": False,
            "isolated_run_artifacts": True,
        },
        "benchmark": {
            "stage0_elapsed_seconds": stage0_elapsed,
            "total_pipeline_elapsed_seconds": elapsed,
            "stage0_variants_per_second": (
                total_variants / stage0_elapsed if stage0_elapsed else None
            ),
            "before_exact_validation_jobs": total_variants,
            "after_exact_validation_jobs": len(survivors),
            "false_negative_review_sample": false_negative_review,
        },
        "forward_candidates": forward_candidates,
        "prospective_snapshot_contract": {
            "immutable_fields": [
                "signal_snapshot",
                "features",
                "strategy_version",
                "parameter_version",
                "market_context",
                "cost_prediction",
                "entry_plan",
                "stop_targets",
            ],
            "canonical_outcome_source": "P0_CANONICAL_FINANCIAL_STATE",
            "telegram_delivery_is_trade_label": False,
            "retroactive_forward_evidence": False,
        },
        "ml_status": {
            "authority": "SHADOW_ONLY",
            "neural_networks": "NOT_EVALUABLE",
            "transformers": "NOT_EVALUABLE",
            "mixture_of_experts": "NOT_EVALUABLE",
            "reinforcement_learning": "NOT_EVALUABLE",
            "authority_changes": 0,
        },
        "experiment_contracts": experiment_contracts,
        "reference_concepts": list(REFERENCE_CONCEPTS),
        "source_hashes": source_hashes,
        "safety": {
            "real_orders_submitted": 0,
            "real_orders_cancelled": 0,
            "real_protective_orders_modified": 0,
            "private_bitvavo_mutations": 0,
            "live_authority_increase": False,
            "risk_limit_increase": False,
            "shariah_policy_weakening": False,
            "research_trades_placed": 0,
        },
    }
    payload["artifact_hash"] = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "artifact_hash", "benchmark"}
        },
        length=64,
    )
    root = settings.paths.output_dir / "research_factory"
    artifact_path = root / "runs" / run_id / "research_factory_evidence.json"
    if artifact_path.is_file():
        existing = _read_json(artifact_path)
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key not in {"created_at", "artifact_hash", "benchmark"}
        }
        comparable_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "artifact_hash", "benchmark"}
        }
        if stable_hash(comparable_existing) != stable_hash(comparable_payload):
            raise FileExistsError(f"immutable research artifact collision: {artifact_path}")
        payload = existing
    else:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact_path, payload)
    status = {
        "schema_version": "research_factory_operator_status_v1",
        "run_id": run_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": payload["artifact_hash"],
        "total_active_families": len(p0_5.get("family_results") or []),
        "total_implementations": len(p0_5.get("strategy_results") or []),
        "stage0_queue": 0,
        "stage0_rejected": max(0, len(aggregate) - len(survivors)),
        "exact_queue": len(survivors),
        "walk_forward_pass": int(exact_result.get("status") == "ROBUST_EXACT_PASS"),
        "stress_pass": int(exact_result.get("status") == "ROBUST_EXACT_PASS"),
        "forward_candidates": len(forward_candidates),
        "paper_positive": sum(
            row.get("promotion_status") == "PAPER_POSITIVE"
            for row in p0_5.get("promotion_recommendations") or []
        ),
        "live_validated": 0,
        "ml_authority": "SHADOW_ONLY",
        "live_authority_changed": False,
    }
    atomic_write_json(root / "latest.json", status)
    return {
        "status": "COMPLETE",
        "run_id": run_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": payload["artifact_hash"],
        "p0_5_branch": branch["decision"],
        "tested_variant_count": total_variants,
        "stage0_survivor_count": len(survivors),
        "exact_status": exact_result.get("status"),
        "forward_candidate_count": len(forward_candidates),
        "real_orders_submitted": 0,
        "private_bitvavo_mutations": 0,
    }


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EXPERIMENT_SCHEMA_VERSION",
    "RESEARCH_FACTORY_SCHEMA_VERSION",
    "STAGE0_SCHEMA_VERSION",
    "VALIDATION_MANIFEST_SCHEMA_VERSION",
    "DataQualityReport",
    "DatasetIdentity",
    "ExperimentContract",
    "FailedBreakdownReversalResearchAdapter",
    "PlateauResult",
    "PromotionState",
    "ProspectiveSnapshot",
    "REFERENCE_CONCEPTS",
    "RejectionReason",
    "ResearchCache",
    "SharedCostModel",
    "Stage0Hypothesis",
    "Stage0Result",
    "Stage0Signals",
    "Stage0Trade",
    "ValidationFold",
    "WalkForwardManifest",
    "aggregate_stage0_by_parameter",
    "build_research_factory_artifact",
    "build_walk_forward_manifest",
    "derive_dataset_identity",
    "failed_breakdown_reversal_signals",
    "generate_parameter_grid",
    "load_immutable_ohlcv",
    "multiple_testing_accounting",
    "parameter_plateaus",
    "prioritize_research_backlog",
    "promotion_state_for",
    "recursive_warmup_stability",
    "run_exact_candidate_validation",
    "run_exact_rejection_review",
    "simulate_stage0",
    "stage0_causality_check",
    "stage0_cost_stress",
    "stage0_delay_liquidity_stress",
    "static_lookahead_audit",
    "strategy_result_correlation",
    "validate_stage0_data",
]
