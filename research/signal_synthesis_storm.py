"""Deterministic cross-family signal-DNA synthesis research storm.

This module deliberately reuses the canonical :mod:`research.combinatorial_lab`
registry and signal formulas.  It only owns a broad, preregistered screening
layer: strategy DNA is fixed before evaluation, development data alone drives
Pareto selection, and every tested path remains in the multiple-testing
denominator.  The output is research evidence, never an executable candidate.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.combinatorial_lab import (
    BlockRole,
    ExitProfile,
    LogicMode,
    SignalBlock,
    canonical_parameters,
    parameter_hash,
    signal_block_registry,
)
from research.optimization import deflated_sharpe_ratio
from research.portfolio_storm import large_matrix_multiple_testing
from utils.common import stable_hash
from utils.pandas_time import sunday_week_end_labels

SIGNAL_STORM_ENGINE_VERSION = "1.1.0"
SIGNAL_STORM_TRIAL_COUNT = 5_000
SIGNAL_STORM_SEED = 20260725
SIGNAL_STORM_TIMEFRAMES = ("1h", "4h", "1d")
SIGNAL_STORM_MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
SIGNAL_STORM_ASSET_PAIRS = tuple(itertools.combinations(SIGNAL_STORM_MARKETS, 2))
SIGNAL_STORM_MAXIMUM_TOTAL_EXPOSURE = 0.40
SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE = 0.20
SIGNAL_STORM_MINIMUM_CASH = 0.60
SIGNAL_STORM_COMMON_WARMUP_BARS = 220
SIGNAL_STORM_BATCH_SIZE = 384
SIGNAL_STORM_MINIMUM_ACTIVE_DEVELOPMENT_WEEKS = 12

# This source column is intentionally absent until a timestamped event feed is
# connected.  Missing-data policy is REJECT, so the block may not be converted
# to a silent False signal.
SIGNAL_STORM_CONTEXT_BLOCKERS: Mapping[str, str] = {
    "high_impact_event_avoidance": ("MISSING_POINT_IN_TIME_COLUMN:events_high_impact_event_risk")
}


@dataclass(frozen=True, slots=True)
class SignalSynthesisDNA:
    """One immutable signal, timeframe, asset-pair and exit chromosome."""

    entry_block: str
    context_block: str
    confirmation_block: str | None
    avoidance_block: str | None
    exit_block: str
    overlay_block: str | None
    timeframe: str
    asset_pair: tuple[str, str]
    logic_mode: str
    vote_threshold: float
    exit_profile: str
    stop_atr: float
    target_atr: float
    trailing_atr: float
    maximum_holding_bars: int
    block_parameters: Mapping[str, Mapping[str, Any]]
    maximum_total_exposure: float = SIGNAL_STORM_MAXIMUM_TOTAL_EXPOSURE
    maximum_position_exposure: float = SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE
    minimum_cash: float = SIGNAL_STORM_MINIMUM_CASH

    def __post_init__(self) -> None:
        if self.timeframe not in SIGNAL_STORM_TIMEFRAMES:
            raise ValueError("signal storm timeframe is unsupported")
        if (
            len(self.asset_pair) != 2
            or len(set(self.asset_pair)) != 2
            or not set(self.asset_pair).issubset(SIGNAL_STORM_MARKETS)
        ):
            raise ValueError("signal storm asset pair is invalid")
        LogicMode(self.logic_mode)
        ExitProfile(self.exit_profile)
        if not 0.0 < self.vote_threshold <= 1.0:
            raise ValueError("vote threshold must be in (0, 1]")
        if min(self.stop_atr, self.target_atr) <= 0.0:
            raise ValueError("ATR stop and target must be positive")
        if self.trailing_atr < 0.0 or self.maximum_holding_bars < 1:
            raise ValueError("signal storm exit parameters are invalid")
        if self.maximum_total_exposure > 0.40 + 1e-12:
            raise ValueError("signal storm total exposure exceeds strict limit")
        if self.maximum_position_exposure > 0.20 + 1e-12:
            raise ValueError("signal storm position exposure exceeds strict limit")
        if self.minimum_cash < 0.60 - 1e-12:
            raise ValueError("signal storm minimum cash violates strict reserve")
        if self.maximum_total_exposure > 1.0 - self.minimum_cash + 1e-12:
            raise ValueError("signal storm exposure violates minimum cash")
        if (
            len(self.asset_pair) * self.maximum_position_exposure
            > self.maximum_total_exposure + 1e-12
        ):
            raise ValueError("signal storm pair allocation exceeds total limit")

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                block
                for block in (
                    self.entry_block,
                    self.context_block,
                    self.confirmation_block,
                    self.avoidance_block,
                    self.exit_block,
                    self.overlay_block,
                )
                if block is not None
            )
        )

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "family": "CROSS_FAMILY_SIGNAL_SYNTHESIS_STORM",
                "engine_version": SIGNAL_STORM_ENGINE_VERSION,
                "parameters": self.to_dict(),
            },
            length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_block": self.entry_block,
            "context_block": self.context_block,
            "confirmation_block": self.confirmation_block,
            "avoidance_block": self.avoidance_block,
            "exit_block": self.exit_block,
            "overlay_block": self.overlay_block,
            "timeframe": self.timeframe,
            "asset_pair": list(self.asset_pair),
            "logic_mode": self.logic_mode,
            "vote_threshold": self.vote_threshold,
            "exit_profile": self.exit_profile,
            "stop_atr": self.stop_atr,
            "target_atr": self.target_atr,
            "trailing_atr": self.trailing_atr,
            "maximum_holding_bars": self.maximum_holding_bars,
            "block_parameters": {
                block_id: canonical_parameters(parameters)
                for block_id, parameters in sorted(self.block_parameters.items())
            },
            "maximum_total_exposure": self.maximum_total_exposure,
            "maximum_position_exposure": self.maximum_position_exposure,
            "minimum_cash": self.minimum_cash,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SignalSynthesisDNA:
        payload = dict(values)
        payload["asset_pair"] = tuple(payload["asset_pair"])
        payload["block_parameters"] = {
            str(block_id): dict(parameters)
            for block_id, parameters in dict(payload["block_parameters"]).items()
        }
        return cls(**payload)


def _parameter_alleles(block: SignalBlock) -> tuple[dict[str, Any], ...]:
    """Return valid min/default/max Cartesian alleles for one block."""

    if not block.parameter_specs:
        return ({},)
    values_by_spec: list[tuple[Any, ...]] = []
    for specification in block.parameter_specs:
        values = specification.values()
        selected = tuple(
            dict.fromkeys(
                (
                    values[0],
                    specification.validate(specification.default),
                    values[-1],
                )
            )
        )
        values_by_spec.append(selected)
    valid: dict[str, dict[str, Any]] = {}
    for values in itertools.product(*values_by_spec):
        candidate = {
            specification.name: value
            for specification, value in zip(
                block.parameter_specs,
                values,
                strict=True,
            )
        }
        try:
            parameters = block.parameters(candidate)
        except ValueError:
            continue
        valid[parameter_hash(parameters)] = canonical_parameters(parameters)
    if not valid:
        raise RuntimeError(f"signal block has no valid parameter alleles: {block.block_id}")
    return tuple(valid[key] for key in sorted(valid))


def _role_pools(
    registry: Mapping[str, SignalBlock],
) -> dict[str, tuple[str, ...]]:
    executable = {
        block_id: block
        for block_id, block in registry.items()
        if block_id not in SIGNAL_STORM_CONTEXT_BLOCKERS
    }
    return {
        "entry": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role is BlockRole.ENTRY_TRIGGER
            )
        ),
        "context": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role in {BlockRole.TREND_FILTER, BlockRole.REGIME_FILTER}
            )
        ),
        "confirmation": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role is BlockRole.CONFIRMATION
            )
        ),
        "avoidance": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role is BlockRole.AVOIDANCE_FILTER
            )
        ),
        "exit": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role is BlockRole.EXIT_TRIGGER
            )
        ),
        "overlay": tuple(
            sorted(
                block_id
                for block_id, block in executable.items()
                if block.role is BlockRole.RISK_OVERLAY
            )
        ),
    }


def _timeframe_compatible(
    registry: Mapping[str, SignalBlock],
    block_ids: Sequence[str],
    timeframe: str,
) -> bool:
    return all(timeframe in registry[block_id].supported_timeframes for block_id in block_ids)


def preregistered_signal_dna(
    *,
    trial_count: int = SIGNAL_STORM_TRIAL_COUNT,
    seed: int = SIGNAL_STORM_SEED,
    registry: Mapping[str, SignalBlock] | None = None,
) -> tuple[SignalSynthesisDNA, ...]:
    """Create a deterministic, unique and role-stratified trial population."""

    if trial_count < 2:
        raise ValueError("signal synthesis storm requires at least two trials")
    selected_registry = dict(registry or signal_block_registry())
    pools = _role_pools(selected_registry)
    if any(not pool for pool in pools.values()):
        raise ValueError("signal synthesis registry is missing a required role")
    alleles = {
        block_id: _parameter_alleles(block)
        for block_id, block in selected_registry.items()
        if block_id not in SIGNAL_STORM_CONTEXT_BLOCKERS
    }
    generator = np.random.default_rng(seed)
    rows: list[SignalSynthesisDNA] = []
    seen: set[str] = set()
    maximum_attempts = max(10_000, trial_count * 30)
    attempts = 0
    while len(rows) < trial_count and attempts < maximum_attempts:
        attempts += 1
        timeframe = SIGNAL_STORM_TIMEFRAMES[int(generator.integers(len(SIGNAL_STORM_TIMEFRAMES)))]
        entry = pools["entry"][int(generator.integers(len(pools["entry"])))]
        context_candidates = tuple(
            block_id
            for block_id in pools["context"]
            if timeframe in selected_registry[block_id].supported_timeframes
        )
        if not context_candidates:
            continue
        context = context_candidates[int(generator.integers(len(context_candidates)))]
        confirmation = (
            pools["confirmation"][int(generator.integers(len(pools["confirmation"])))]
            if float(generator.random()) < 0.80
            else None
        )
        avoidance = (
            pools["avoidance"][int(generator.integers(len(pools["avoidance"])))]
            if float(generator.random()) < 0.65
            else None
        )
        exit_block = pools["exit"][int(generator.integers(len(pools["exit"])))]
        overlay = (
            pools["overlay"][int(generator.integers(len(pools["overlay"])))]
            if float(generator.random()) < 0.20
            else None
        )
        block_ids = tuple(
            block
            for block in (
                entry,
                context,
                confirmation,
                avoidance,
                exit_block,
                overlay,
            )
            if block is not None
        )
        if len(set(block_ids)) != len(block_ids):
            continue
        if not _timeframe_compatible(
            selected_registry,
            block_ids,
            timeframe,
        ):
            continue
        parameters = {
            block_id: alleles[block_id][int(generator.integers(len(alleles[block_id])))]
            for block_id in block_ids
        }
        logic_mode = (
            LogicMode.LAYERED,
            LogicMode.ALL,
            LogicMode.MAJORITY,
            LogicMode.WEIGHTED_VOTE,
        )[int(generator.integers(4))]
        profile = tuple(ExitProfile)[int(generator.integers(len(ExitProfile)))]
        row = SignalSynthesisDNA(
            entry_block=entry,
            context_block=context,
            confirmation_block=confirmation,
            avoidance_block=avoidance,
            exit_block=exit_block,
            overlay_block=overlay,
            timeframe=timeframe,
            asset_pair=SIGNAL_STORM_ASSET_PAIRS[
                int(generator.integers(len(SIGNAL_STORM_ASSET_PAIRS)))
            ],
            logic_mode=logic_mode.value,
            vote_threshold=(0.50, 0.60, 0.70)[int(generator.integers(3))],
            exit_profile=profile.value,
            stop_atr=(1.5, 2.0, 3.0)[int(generator.integers(3))],
            target_atr=(2.0, 3.0, 6.0)[int(generator.integers(3))],
            trailing_atr=(1.5, 2.5, 4.0)[int(generator.integers(3))],
            maximum_holding_bars=(48, 120, 240)[int(generator.integers(3))],
            block_parameters=parameters,
        )
        if row.dna_hash in seen:
            continue
        seen.add(row.dna_hash)
        rows.append(row)
    if len(rows) != trial_count:
        raise RuntimeError(
            f"could not generate requested unique signal DNA: {len(rows)}/{trial_count}"
        )
    if trial_count >= 1_000:
        covered = {block_id for row in rows for block_id in row.block_ids}
        executable = set(selected_registry) - set(SIGNAL_STORM_CONTEXT_BLOCKERS)
        missing = sorted(executable - covered)
        if missing:
            raise RuntimeError(f"signal storm role coverage incomplete: {missing}")
    return tuple(rows)


def signal_storm_plan(
    *,
    trial_count: int = SIGNAL_STORM_TRIAL_COUNT,
    seed: int = SIGNAL_STORM_SEED,
    registry: Mapping[str, SignalBlock] | None = None,
) -> dict[str, Any]:
    selected_registry = dict(registry or signal_block_registry())
    rows = preregistered_signal_dna(
        trial_count=trial_count,
        seed=seed,
        registry=selected_registry,
    )
    hashes = [row.dna_hash for row in rows]
    block_counts = Counter(block_id for row in rows for block_id in row.block_ids)
    return {
        "schema_version": "signal_synthesis_storm_plan_v2",
        "status": "PREREGISTERED_NOT_RUN",
        "campaign": "SIGNAL_SYNTHESIS_STORM_V1",
        "engine_version": SIGNAL_STORM_ENGINE_VERSION,
        "seed": seed,
        "trial_count": len(rows),
        "strategy_dna_hashes": hashes,
        "strategy_dna": [row.to_dict() for row in rows],
        "search_space_hash": stable_hash(hashes, length=64),
        "registered_signal_blocks": len(selected_registry),
        "executable_signal_blocks": len(
            set(selected_registry) - set(SIGNAL_STORM_CONTEXT_BLOCKERS)
        ),
        "blocked_signal_blocks": dict(SIGNAL_STORM_CONTEXT_BLOCKERS),
        "covered_executable_blocks": len(block_counts),
        "block_trial_counts": dict(sorted(block_counts.items())),
        "families_covered": sorted(
            {selected_registry[block_id].family for block_id in block_counts}
        ),
        "roles_covered": sorted(
            {selected_registry[block_id].role.value for block_id in block_counts}
        ),
        "timeframes": list(SIGNAL_STORM_TIMEFRAMES),
        "asset_pairs": [list(pair) for pair in SIGNAL_STORM_ASSET_PAIRS],
        "logic_modes": [
            LogicMode.LAYERED.value,
            LogicMode.ALL.value,
            LogicMode.MAJORITY.value,
            LogicMode.WEIGHTED_VOTE.value,
        ],
        "exit_profiles": [profile.value for profile in ExitProfile],
        "parameter_allele_policy": "MIN_DEFAULT_MAX_CARTESIAN_VALID",
        "selection_basis": "DEVELOPMENT_ONLY",
        "selection_integrity": {
            "development_returns_only": True,
            "development_turnover_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "objectives": {
            "maximize": ["portfolio_period_profit_factor"],
            "minimize": ["ulcer_index", "turnover_efficiency"],
        },
        "pre_pareto_development_gate": {
            "minimum_active_weeks": (SIGNAL_STORM_MINIMUM_ACTIVE_DEVELOPMENT_WEEKS),
            "minimum_net_return_exclusive": 0.0,
            "minimum_profit_factor_exclusive": 1.0,
            "validation_or_confirmation_used": False,
        },
        "maximum_total_exposure": SIGNAL_STORM_MAXIMUM_TOTAL_EXPOSURE,
        "maximum_position_exposure": (SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE),
        "minimum_cash": SIGNAL_STORM_MINIMUM_CASH,
        "next_open_execution": True,
        "screening_only": True,
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def _align_frames(
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    pd.DatetimeIndex,
]:
    if set(frames_by_timeframe) != set(SIGNAL_STORM_TIMEFRAMES):
        raise ValueError("signal storm requires exactly 1h, 4h and 1d frames")
    common_by_timeframe: dict[str, pd.DatetimeIndex] = {}
    for timeframe, frames in frames_by_timeframe.items():
        if set(frames) != set(SIGNAL_STORM_MARKETS):
            raise ValueError(f"signal storm requires strict four-asset universe: {timeframe}")
        common = pd.DatetimeIndex(
            sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
        )
        if len(common) <= SIGNAL_STORM_COMMON_WARMUP_BARS + 100:
            raise ValueError(f"insufficient common signal storm history: {timeframe}")
        common_by_timeframe[timeframe] = common
    common_start = max(
        index[SIGNAL_STORM_COMMON_WARMUP_BARS] for index in common_by_timeframe.values()
    )
    common_end = min(index[-1] for index in common_by_timeframe.values())
    if common_start >= common_end:
        raise ValueError("signal storm timeframes have no common research period")
    aligned: dict[str, dict[str, pd.DataFrame]] = {}
    for timeframe, frames in frames_by_timeframe.items():
        selected_index = common_by_timeframe[timeframe]
        selected_index = selected_index[
            (selected_index >= common_start) & (selected_index <= common_end)
        ]
        aligned[timeframe] = {}
        for market, frame in frames.items():
            selected = frame.loc[selected_index].copy()
            selected.attrs.update(frame.attrs)
            aligned[timeframe][market] = selected
    weekly_index = pd.date_range(
        start=pd.Timestamp(common_start).normalize(),
        end=pd.Timestamp(common_end).normalize(),
        freq="W-SUN",
        tz="UTC",
    )
    if len(weekly_index) < 40:
        raise ValueError("signal storm requires at least forty common weeks")
    return aligned, weekly_index


def _block_signal(
    block: SignalBlock,
    frame: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    missing = sorted(set(block.required_features) - set(frame.columns))
    if missing:
        raise ValueError(f"BLOCKED_MISSING_SIGNAL_FEATURE:{block.block_id}:{missing}")
    return block.calculate(frame, parameters).to_numpy(dtype=bool)


def _signal_matrix(
    rows: Sequence[SignalSynthesisDNA],
    *,
    frame: pd.DataFrame,
    registry: Mapping[str, SignalBlock],
    cache: dict[tuple[str, str], np.ndarray],
    market: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def values(block_id: str, parameters: Mapping[str, Any]) -> np.ndarray:
        key = (
            block_id,
            parameter_hash(parameters),
        )
        if key not in cache:
            cache[key] = _block_signal(
                registry[block_id],
                frame,
                parameters,
            )
        return cache[key]

    entries: list[np.ndarray] = []
    exits: list[np.ndarray] = []
    avoids: list[np.ndarray] = []
    reductions: list[np.ndarray] = []
    for row in rows:
        block_signals = {
            block_id: values(
                block_id,
                row.block_parameters[block_id],
            )
            for block_id in row.block_ids
        }
        raw_entry = block_signals[row.entry_block]
        context = block_signals[row.context_block]
        confirmation = (
            block_signals[row.confirmation_block]
            if row.confirmation_block
            else np.ones(len(frame), dtype=bool)
        )
        context_role = registry[row.context_block].role
        mandatory_regime = (
            context if context_role is BlockRole.REGIME_FILTER else np.ones(len(frame), dtype=bool)
        )
        voters = [raw_entry]
        if context_role is BlockRole.TREND_FILTER:
            voters.append(context)
        if row.confirmation_block:
            voters.append(confirmation)
        logic_mode = LogicMode(row.logic_mode)
        if logic_mode in {LogicMode.LAYERED, LogicMode.ALL}:
            entry = raw_entry & context & confirmation
        elif logic_mode is LogicMode.MAJORITY:
            votes = np.sum(np.vstack(voters), axis=0)
            entry = raw_entry & mandatory_regime & (votes >= math.ceil(len(voters) / 2))
        else:
            votes = np.mean(np.vstack(voters), axis=0)
            entry = raw_entry & mandatory_regime & (votes >= row.vote_threshold)
        avoid = (
            block_signals[row.avoidance_block]
            if row.avoidance_block
            else np.zeros(len(frame), dtype=bool)
        )
        overlay = (
            block_signals[row.overlay_block]
            if row.overlay_block
            else np.zeros(len(frame), dtype=bool)
        )
        transition = overlay & ~np.roll(overlay, 1)
        transition[0] = False
        entries.append(entry & ~avoid)
        exits.append(block_signals[row.exit_block])
        avoids.append(avoid)
        reductions.append(transition)
    return (
        np.column_stack(entries),
        np.column_stack(exits),
        np.column_stack(avoids),
        np.column_stack(reductions),
    )


def _effective_exit_arrays(
    rows: Sequence[SignalSynthesisDNA],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stop = np.asarray([row.stop_atr for row in rows], dtype=float)
    target = np.asarray([row.target_atr for row in rows], dtype=float)
    trailing = np.asarray([row.trailing_atr for row in rows], dtype=float)
    maximum_holding = np.asarray(
        [row.maximum_holding_bars for row in rows],
        dtype=int,
    )
    profiles = np.asarray([row.exit_profile for row in rows], dtype=object)
    fixed = profiles == ExitProfile.FIXED_R.value
    trailing_profile = profiles == ExitProfile.TRAILING_TREND.value
    time_profile = profiles == ExitProfile.TIME_REGIME.value
    trailing[fixed | time_profile] = 0.0
    target[trailing_profile | time_profile] = np.maximum(
        target[trailing_profile | time_profile],
        20.0,
    )
    trailing[trailing_profile] = np.maximum(
        trailing[trailing_profile],
        2.5,
    )
    return stop, target, trailing, maximum_holding


def _simulate_asset_batch(
    frame: pd.DataFrame,
    *,
    entry: np.ndarray,
    exit_signal: np.ndarray,
    avoid: np.ndarray,
    reduce: np.ndarray,
    stop_multiple: np.ndarray,
    target_multiple: np.ndarray,
    trailing_multiple: np.ndarray,
    maximum_holding: np.ndarray,
    one_way_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize trial state while preserving next-open and stop ordering."""

    observations, trials = entry.shape
    if any(matrix.shape != (observations, trials) for matrix in (exit_signal, avoid, reduce)):
        raise ValueError("signal matrices do not share one shape")
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    atr = frame["atr_14"].to_numpy(dtype=float)
    returns = np.zeros((observations - 1, trials), dtype=np.float32)
    turnover = np.zeros_like(returns)
    held = np.zeros(trials, dtype=bool)
    stop_price = np.zeros(trials, dtype=float)
    target_price = np.zeros(trials, dtype=float)
    trailing_distance = np.zeros(trials, dtype=float)
    trailing_stop = np.full(trials, -np.inf, dtype=float)
    maximum_seen = np.full(trials, -np.inf, dtype=float)
    bars_held = np.zeros(trials, dtype=int)

    for timestamp in range(1, observations):
        output = returns[timestamp - 1]
        flow = turnover[timestamp - 1]
        was_held = held.copy()
        forced = was_held & (
            exit_signal[timestamp - 1] | avoid[timestamp - 1] | reduce[timestamp - 1]
        )
        if forced.any():
            output[forced] = opens[timestamp] / closes[timestamp - 1] - 1.0 - one_way_cost
            flow[forced] += 1.0
            held[forced] = False

        active = was_held & ~forced
        effective_stop = np.maximum(stop_price, trailing_stop)
        stop_hit = active & (lows[timestamp] <= effective_stop)
        target_hit = active & ~stop_hit & (highs[timestamp] >= target_price)
        if stop_hit.any():
            output[stop_hit] = effective_stop[stop_hit] / closes[timestamp - 1] - 1.0 - one_way_cost
            flow[stop_hit] += 1.0
            held[stop_hit] = False
        if target_hit.any():
            output[target_hit] = (
                target_price[target_hit] / closes[timestamp - 1] - 1.0 - one_way_cost
            )
            flow[target_hit] += 1.0
            held[target_hit] = False
        carried = active & ~stop_hit & ~target_hit
        if carried.any():
            output[carried] = closes[timestamp] / closes[timestamp - 1] - 1.0
            bars_held[carried] += 1
            maximum_seen[carried] = np.maximum(
                maximum_seen[carried],
                highs[timestamp],
            )
            trailed = carried & (trailing_distance > 0.0)
            trailing_stop[trailed] = np.maximum(
                trailing_stop[trailed],
                maximum_seen[trailed] - trailing_distance[trailed],
            )
            time_exit = carried & (bars_held >= maximum_holding)
            if time_exit.any():
                output[time_exit] -= one_way_cost
                flow[time_exit] += 1.0
                held[time_exit] = False

        can_enter = (
            ~was_held & entry[timestamp - 1] & ~exit_signal[timestamp - 1] & ~avoid[timestamp - 1]
        )
        valid_atr = math.isfinite(float(atr[timestamp - 1])) and (atr[timestamp - 1] > 0.0)
        if can_enter.any() and valid_atr:
            stop_distance = atr[timestamp - 1] * stop_multiple
            target_distance = atr[timestamp - 1] * target_multiple
            entry_stop = opens[timestamp] - stop_distance
            entry_target = opens[timestamp] + target_distance
            same_stop = can_enter & (lows[timestamp] <= entry_stop)
            same_target = can_enter & ~same_stop & (highs[timestamp] >= entry_target)
            if same_stop.any():
                output[same_stop] = (
                    entry_stop[same_stop] / opens[timestamp] - 1.0 - 2.0 * one_way_cost
                )
                flow[same_stop] += 2.0
            if same_target.any():
                output[same_target] = (
                    entry_target[same_target] / opens[timestamp] - 1.0 - 2.0 * one_way_cost
                )
                flow[same_target] += 2.0
            opened = can_enter & ~same_stop & ~same_target
            if opened.any():
                output[opened] = closes[timestamp] / opens[timestamp] - 1.0 - one_way_cost
                flow[opened] += 1.0
                held[opened] = True
                stop_price[opened] = entry_stop[opened]
                target_price[opened] = entry_target[opened]
                trailing_distance[opened] = atr[timestamp - 1] * trailing_multiple[opened]
                maximum_seen[opened] = np.maximum(
                    opens[timestamp],
                    highs[timestamp],
                )
                trailing_stop[opened] = -np.inf
                bars_held[opened] = 1
                trailed = opened & (trailing_distance > 0.0)
                trailing_stop[trailed] = maximum_seen[trailed] - trailing_distance[trailed]

    if held.any():
        returns[-1, held] -= one_way_cost
        turnover[-1, held] += 1.0
    return returns, turnover


def _weekly(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    weekly_index: pd.DatetimeIndex,
    *,
    compound: bool,
) -> np.ndarray:
    frame = pd.DataFrame(values, index=index)
    labels = sunday_week_end_labels(frame.index)
    if compound:
        result = (1.0 + frame).groupby(labels).prod() - 1.0
    else:
        result = frame.groupby(labels).sum()
    result = result.reindex(weekly_index)
    if result.isna().any(axis=None):
        missing = int(result.isna().any(axis=1).sum())
        raise ValueError(f"signal storm weekly alignment has gaps: {missing}")
    return result.to_numpy(dtype=np.float32)


def _objectives(
    returns: np.ndarray,
    turnover: float,
) -> tuple[float, float, float, float]:
    positive = float(returns[returns > 0].sum())
    negative = float(abs(returns[returns < 0].sum()))
    profit_factor = positive / negative if negative > 0.0 else 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    ulcer = float(np.sqrt(np.mean(np.square(drawdown))))
    efficiency = turnover / max(float(equity[-1]), 1e-12)
    standard = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / standard) if standard > 0.0 else 0.0
    return profit_factor, ulcer, efficiency, sharpe


def _pareto_indices(values: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for index, row in enumerate(values):
        dominates = (
            (values[:, 0] >= row[0])
            & (values[:, 1] <= row[1])
            & (values[:, 2] <= row[2])
            & ((values[:, 0] > row[0]) | (values[:, 1] < row[1]) | (values[:, 2] < row[2]))
        )
        if not bool(dominates.any()):
            selected.append(index)
    return np.asarray(selected, dtype=int)


def _coverage(
    dna: Sequence[SignalSynthesisDNA],
    registry: Mapping[str, SignalBlock],
) -> dict[str, Any]:
    blocks = Counter(block for row in dna for block in row.block_ids)
    return {
        "block_trial_counts": dict(sorted(blocks.items())),
        "covered_executable_blocks": len(blocks),
        "family_trial_counts": dict(
            sorted(
                Counter(registry[block].family for row in dna for block in row.block_ids).items()
            )
        ),
        "role_trial_counts": dict(
            sorted(
                Counter(
                    registry[block].role.value for row in dna for block in row.block_ids
                ).items()
            )
        ),
        "timeframe_trial_counts": dict(sorted(Counter(row.timeframe for row in dna).items())),
        "asset_pair_trial_counts": dict(
            sorted(Counter("|".join(row.asset_pair) for row in dna).items())
        ),
        "logic_mode_trial_counts": dict(sorted(Counter(row.logic_mode for row in dna).items())),
        "exit_profile_trial_counts": dict(sorted(Counter(row.exit_profile for row in dna).items())),
    }


def run_signal_synthesis_storm(
    frames_by_timeframe: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
    dna: Sequence[SignalSynthesisDNA],
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    prior_known_trials: int,
    known_trial_count: int | None = None,
    bootstrap_samples: int = 2_000,
    maximum_survivors: int = 48,
    batch_size: int = SIGNAL_STORM_BATCH_SIZE,
    registry: Mapping[str, SignalBlock] | None = None,
) -> tuple[dict[str, Any], np.ndarray, pd.DatetimeIndex]:
    """Run every preregistered path and return orderless screen evidence."""

    if len(dna) < 2:
        raise ValueError("signal storm requires at least two DNA paths")
    if len({row.dna_hash for row in dna}) != len(dna):
        raise ValueError("signal storm DNA contains duplicates")
    if batch_size < 1:
        raise ValueError("signal storm batch size must be positive")
    total_known_trials = (
        int(known_trial_count)
        if known_trial_count is not None
        else prior_known_trials + len(dna)
    )
    if total_known_trials < max(prior_known_trials, len(dna)):
        raise ValueError(
            "known trial count cannot be below the prior or evaluated "
            "strategy count"
        )
    selected_registry = dict(registry or signal_block_registry())
    aligned, weekly_index = _align_frames(frames_by_timeframe)
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    weekly_returns = np.zeros(
        (len(weekly_index), len(dna)),
        dtype=np.float32,
    )
    weekly_turnover = np.zeros_like(weekly_returns)
    indices_by_group = {
        (timeframe, pair): [
            index
            for index, row in enumerate(dna)
            if row.timeframe == timeframe and row.asset_pair == pair
        ]
        for timeframe in SIGNAL_STORM_TIMEFRAMES
        for pair in SIGNAL_STORM_ASSET_PAIRS
    }
    signal_caches: dict[
        tuple[str, str],
        dict[tuple[str, str], np.ndarray],
    ] = {}
    for (timeframe, pair), trial_indices in indices_by_group.items():
        for offset in range(0, len(trial_indices), batch_size):
            selected_indices = trial_indices[offset : offset + batch_size]
            rows = [dna[index] for index in selected_indices]
            stop, target, trailing, maximum_holding = _effective_exit_arrays(rows)
            batch_return: np.ndarray | None = None
            batch_turnover: np.ndarray | None = None
            bar_index: pd.DatetimeIndex | None = None
            for market in pair:
                frame = aligned[timeframe][market]
                cache = signal_caches.setdefault((timeframe, market), {})
                entry, exit_signal, avoid, reduce = _signal_matrix(
                    rows,
                    frame=frame,
                    registry=selected_registry,
                    cache=cache,
                    market=market,
                )
                asset_return, asset_turnover = _simulate_asset_batch(
                    frame,
                    entry=entry,
                    exit_signal=exit_signal,
                    avoid=avoid,
                    reduce=reduce,
                    stop_multiple=stop,
                    target_multiple=target,
                    trailing_multiple=trailing,
                    maximum_holding=maximum_holding,
                    one_way_cost=one_way_cost,
                )
                weighted_return = asset_return * SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE
                weighted_turnover = asset_turnover * SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE
                batch_return = (
                    weighted_return if batch_return is None else batch_return + weighted_return
                )
                batch_turnover = (
                    weighted_turnover
                    if batch_turnover is None
                    else batch_turnover + weighted_turnover
                )
                bar_index = frame.index[1:]
            if batch_return is None or batch_turnover is None or bar_index is None:
                raise RuntimeError("signal storm batch did not evaluate")
            weekly_returns[:, selected_indices] = _weekly(
                batch_return,
                bar_index,
                weekly_index,
                compound=True,
            )
            weekly_turnover[:, selected_indices] = _weekly(
                batch_turnover,
                bar_index,
                weekly_index,
                compound=False,
            )
    if not np.isfinite(weekly_returns).all():
        raise ValueError("signal storm produced non-finite returns")
    observations = len(weekly_index)
    development_end = int(observations * 0.60)
    validation_end = int(observations * 0.80)
    if (
        min(
            development_end,
            validation_end - development_end,
            observations - validation_end,
        )
        < 8
    ):
        raise ValueError("signal storm split is too short")
    development = weekly_returns[:development_end].astype(float)
    objective_rows = np.asarray(
        [
            _objectives(
                development[:, index],
                float(weekly_turnover[:development_end, index].sum()),
            )
            for index in range(len(dna))
        ],
        dtype=float,
    )
    development_activity = np.count_nonzero(
        np.abs(development) > 1e-12,
        axis=0,
    )
    development_net = np.prod(1.0 + development, axis=0) - 1.0
    eligible = np.flatnonzero(
        (development_activity >= SIGNAL_STORM_MINIMUM_ACTIVE_DEVELOPMENT_WEEKS)
        & (development_net > 0.0)
        & (objective_rows[:, 0] > 1.0)
        & (weekly_turnover[:development_end].sum(axis=0) > 0.0)
    )
    pareto = (
        eligible[_pareto_indices(objective_rows[eligible, :3])]
        if len(eligible)
        else np.asarray([], dtype=int)
    )
    if len(pareto) > maximum_survivors:
        pf_rank = pd.Series(objective_rows[pareto, 0]).rank(pct=True)
        ui_rank = pd.Series(-objective_rows[pareto, 1]).rank(pct=True)
        te_rank = pd.Series(-objective_rows[pareto, 2]).rank(pct=True)
        robust = (pf_rank + ui_rank + te_rank).to_numpy(dtype=float)
        pareto = pareto[np.argsort(robust)[::-1][:maximum_survivors]]
    multiple_testing = large_matrix_multiple_testing(
        development,
        bootstrap_samples=bootstrap_samples,
        block_size=4,
        seed=SIGNAL_STORM_SEED,
    )
    weekly_standard = development.std(axis=0, ddof=1)
    trial_sharpes = np.divide(
        development.mean(axis=0),
        weekly_standard,
        out=np.zeros(len(dna), dtype=float),
        where=weekly_standard > 0.0,
    )
    survivors: list[dict[str, Any]] = []
    for index in pareto:
        row = dna[int(index)]
        validation = weekly_returns[
            development_end:validation_end,
            index,
        ].astype(float)
        confirmation = weekly_returns[validation_end:, index].astype(float)
        dsr = deflated_sharpe_ratio(
            pd.Series(development[:, index]),
            trial_sharpes,
            observed_sharpe=float(trial_sharpes[index]),
            total_trials=total_known_trials,
        )
        survivors.append(
            {
                "strategy_dna_hash": row.dna_hash,
                "parameters": row.to_dict(),
                "development": dict(
                    zip(
                        (
                            "portfolio_period_profit_factor",
                            "ulcer_index",
                            "turnover_efficiency",
                            "weekly_sharpe",
                        ),
                        objective_rows[index],
                        strict=True,
                    )
                ),
                "validation": {
                    "net_return": float(np.prod(1.0 + validation) - 1.0),
                    "mean_weekly_return": float(validation.mean()),
                },
                "confirmation": {
                    "net_return": float(np.prod(1.0 + confirmation) - 1.0),
                    "mean_weekly_return": float(confirmation.mean()),
                },
                "deflated_sharpe_probability": dsr,
                "canonical_exact_status": "REQUIRED_BEFORE_PROMOTION",
                "paper_candidate_permitted": False,
                "live_ready": False,
            }
        )
    coverage = _coverage(dna, selected_registry)
    report = {
        "schema_version": "signal_synthesis_storm_report_v2",
        "status": "COMPLETED_SCREENING_NOT_PROMOTED",
        "campaign": "SIGNAL_SYNTHESIS_STORM_V1",
        "engine_version": SIGNAL_STORM_ENGINE_VERSION,
        "trial_count": len(dna),
        "prior_known_trials": prior_known_trials,
        "new_strategy_trial_count": (
            total_known_trials - prior_known_trials
        ),
        "total_known_trials": total_known_trials,
        "search_space_hash": stable_hash(
            [row.dna_hash for row in dna],
            length=64,
        ),
        "selection_basis": "DEVELOPMENT_ONLY",
        "selection_integrity": {
            "development_returns_only": True,
            "development_turnover_only": True,
            "validation_used_for_selection": False,
            "confirmation_used_for_selection": False,
        },
        "screening_semantics": {
            "mark_to_market": True,
            "next_open_execution": True,
            "same_bar_stop_priority": True,
            "one_way_costs_on_each_fill": True,
            "risk_overlay_reduction": ("CONSERVATIVE_FULL_EXIT_IN_SCREEN"),
            "canonical_exact_backtest_required": True,
        },
        "development_screen": {
            "minimum_active_weeks": (SIGNAL_STORM_MINIMUM_ACTIVE_DEVELOPMENT_WEEKS),
            "minimum_net_return_exclusive": 0.0,
            "minimum_profit_factor_exclusive": 1.0,
            "eligible_trial_count": len(eligible),
            "ineligible_trial_count": len(dna) - len(eligible),
            "validation_or_confirmation_used": False,
        },
        "split": {
            "frequency": "W-SUN",
            "common_start": weekly_index[0].isoformat(),
            "common_end": weekly_index[-1].isoformat(),
            "development_observations": development_end,
            "validation_observations": (validation_end - development_end),
            "confirmation_observations": observations - validation_end,
        },
        "coverage": coverage,
        "registered_signal_blocks": len(selected_registry),
        "executable_signal_blocks": len(
            set(selected_registry) - set(SIGNAL_STORM_CONTEXT_BLOCKERS)
        ),
        "blocked_signal_blocks": dict(SIGNAL_STORM_CONTEXT_BLOCKERS),
        "pareto_survivor_count": len(survivors),
        "positive_validation_survivors": sum(
            row["validation"]["net_return"] > 0.0 for row in survivors
        ),
        "positive_confirmation_survivors": sum(
            row["confirmation"]["net_return"] > 0.0 for row in survivors
        ),
        "pareto_survivors": survivors,
        "multiple_testing": {
            **multiple_testing,
            "dsr_total_trial_denominator": total_known_trials,
            "white_reality_check_gate": (multiple_testing["white_reality_check_pvalue"] <= 0.10),
            "hansen_spa_gate": (multiple_testing["hansen_spa_pvalue"] <= 0.05),
            "pbo_gate": (
                multiple_testing["probability_of_backtest_overfitting"] is not None
                and multiple_testing["probability_of_backtest_overfitting"] <= 0.10
            ),
            "white_spa_status": ("FORMALLY_EVALUATED_ALL_SIGNAL_STORM_TRIALS"),
        },
        "maximum_total_exposure": (SIGNAL_STORM_MAXIMUM_TOTAL_EXPOSURE),
        "maximum_position_exposure": (SIGNAL_STORM_MAXIMUM_POSITION_EXPOSURE),
        "minimum_cash": SIGNAL_STORM_MINIMUM_CASH,
        "research_pass": False,
        "paper_candidates": 0,
        "orders_generated": 0,
        "live_ready": False,
    }
    return report, weekly_returns, weekly_index


__all__ = [
    "SIGNAL_STORM_CONTEXT_BLOCKERS",
    "SIGNAL_STORM_ENGINE_VERSION",
    "SIGNAL_STORM_SEED",
    "SIGNAL_STORM_TRIAL_COUNT",
    "SignalSynthesisDNA",
    "preregistered_signal_dna",
    "run_signal_synthesis_storm",
    "signal_storm_plan",
]
