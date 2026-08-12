"""Registry-driven, resumable simple-strategy research-space factory.

The factory separates exhaustive hypothesis registration from selective deep
validation.  Batch sizes control resource use only; they never define the end
of the content-addressed search space.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import (
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_SECONDS,
    Settings,
    normalize_timeframe,
)
from research.combinatorial_lab import (
    BlockDirection,
    BlockRole,
    CombinationGenerator,
    CombinationState,
    GenerationMode,
    LogicMode,
    SignalBlock,
    SignalOperator,
    StrategyCombination,
    _synthetic_ohlcv,
    signal_block_registry,
)
from research.features import FeaturePipeline
from research.indicator_registry import (
    CoverageStatus,
    IndicatorDefinition,
    IndicatorRole,
    canonical_slug,
    indicator_registry,
)
from utils.common import atomic_write_json, stable_hash, utc_iso

SIMPLE_LAB_SCHEMA_VERSION = "simple_strategy_lab_v2"
CANONICAL_HARVEST_VERSION = 2
DEFAULT_COMPLEXITIES = (1, 2, 3, 4, 5)
DEFAULT_TIMEFRAMES = tuple(SUPPORTED_TIMEFRAMES)
RESULT_FILES: dict[str, tuple[str, ...]] = {
    "single_indicator_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "indicator",
        "market",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "single_condition_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_id",
        "market",
        "timeframe",
        "status",
        "raw_signals",
        "completed_trades",
    ),
    "two_block_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "three_block_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "four_block_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "five_block_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "fractal_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
    ),
    "candlestick_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "block_ids",
        "market",
        "timeframe",
        "status",
    ),
    "timeframe_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "timeframe",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "asset_results.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "market",
        "status",
        "net_return",
        "profit_factor",
        "completed_trades",
    ),
    "frequency_buckets.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "market",
        "timeframe",
        "trades_per_year",
        "frequency_bucket",
    ),
    "signal_funnels.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "market",
        "timeframe",
        "tradable_bars",
        "raw_signals",
        "edge_triggered_signals",
        "blocked_existing_position",
        "blocked_risk",
        "completed_round_trips",
        "average_holding_bars",
    ),
    "ablation_results.csv": (
        "parent_strategy_dna_hash",
        "parent_strategy_variant_dna_hash",
        "ablation_strategy_dna_hash",
        "ablation_strategy_variant_dna_hash",
        "removed_block",
        "net_return_delta",
        "profit_factor_delta",
        "trade_count_delta",
    ),
    "rejection_reasons.csv": (
        "strategy_dna_hash",
        "strategy_variant_dna_hash",
        "status",
        "reason",
    ),
}
CANONICAL_RESULT_SOURCES = frozenset(
    {
        "FAST_SCREEN_REAL",
        "EXACT_REAL",
        "NORMAL",
        "STRESSED",
        "DOUBLE_COST",
        "FINAL_HOLDOUT",
    }
)
_EXECUTABLE_INDICATOR_STATUSES = frozenset(
    {
        CoverageStatus.IMPLEMENTED,
        CoverageStatus.IMPLEMENTED_AS_ALIAS,
        CoverageStatus.DERIVED_FROM_EXISTING_FEATURES,
    }
)
_BEARISH_TOKENS = frozenset(
    {
        "bearish",
        "lower_low",
        "lower_high",
        "breakdown",
        "shooting_star",
        "hanging_man",
        "three_black",
        "negative",
    }
)
_BULLISH_TOKENS = frozenset(
    {
        "bullish",
        "higher_low",
        "higher_high",
        "breakout",
        "hammer",
        "three_white",
        "positive",
        "reclaim",
    }
)


@lru_cache(maxsize=1)
def _canonical_feature_schema() -> tuple[
    tuple[str, str, float | None, float | None],
    ...,
]:
    """Discover executable feature columns without using synthetic performance."""

    frame = _synthetic_ohlcv(
        rows=1_200,
        timeframe="1h",
        seed=97_531,
    )
    features = FeaturePipeline().build(
        frame,
        market="SCHEMA-ONLY-EUR",
    )
    rows: list[tuple[str, str, float | None, float | None]] = []
    for column in features.columns:
        series = features[column]
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        rows.append(
            (
                str(column),
                str(series.dtype),
                float(finite.min()) if len(finite) else None,
                float(finite.max()) if len(finite) else None,
            )
        )
    return tuple(rows)


def _indicator_direction(
    definition: IndicatorDefinition,
) -> BlockDirection:
    name = canonical_slug(definition.canonical_name)
    if any(token in name for token in _BEARISH_TOKENS):
        return BlockDirection.BEARISH
    if any(token in name for token in _BULLISH_TOKENS):
        return BlockDirection.BULLISH
    return BlockDirection.NEUTRAL


def _role_mapping(
    indicator_role: IndicatorRole,
    *,
    direction: BlockDirection,
) -> BlockRole | None:
    if indicator_role is IndicatorRole.ENTRY:
        return None if direction is BlockDirection.BEARISH else BlockRole.ENTRY_TRIGGER
    if indicator_role is IndicatorRole.EXIT:
        return BlockRole.EXIT_TRIGGER
    if indicator_role is IndicatorRole.FILTER:
        return (
            BlockRole.AVOIDANCE_FILTER
            if direction is BlockDirection.BEARISH
            else BlockRole.CONFIRMATION
        )
    if indicator_role is IndicatorRole.REGIME:
        return BlockRole.REGIME_FILTER
    return None


def registry_driven_signal_blocks() -> dict[str, SignalBlock]:
    """Expand every executable feature-backed indicator into causal baselines."""

    registry = signal_block_registry()
    feature_schema = {
        name: {
            "dtype": dtype,
            "minimum": minimum,
            "maximum": maximum,
        }
        for name, dtype, minimum, maximum in _canonical_feature_schema()
    }
    existing_identities = {
        (
            block.feature,
            block.operator,
            block.role,
            block.signal_kind,
        )
        for block in registry.values()
    }
    for definition in indicator_registry().definitions():
        if (
            not definition.combinable
            or not definition.tradable
            or definition.repaints
            or definition.status not in _EXECUTABLE_INDICATOR_STATUSES
        ):
            continue
        matched_features = sorted(
            {
                feature
                for output in definition.output_columns
                for feature in feature_schema
                if (feature == str(output) or feature.startswith(f"{output}_"))
            }
        )
        if not matched_features:
            continue
        direction = _indicator_direction(definition)
        for feature in matched_features:
            schema = feature_schema[feature]
            is_boolean = schema["dtype"] == "bool"
            for indicator_role in definition.compatible_roles:
                role = _role_mapping(
                    indicator_role,
                    direction=direction,
                )
                if role is None:
                    continue
                if is_boolean:
                    conditions = (
                        (
                            "true",
                            SignalOperator.BOOLEAN_TRUE,
                            "BOOL",
                        ),
                    )
                elif role is BlockRole.EXIT_TRIGGER:
                    conditions = (
                        (
                            "falling",
                            SignalOperator.FALLING,
                            "FALLING",
                        ),
                    )
                    if (
                        schema["minimum"] is not None
                        and schema["maximum"] is not None
                        and float(schema["minimum"]) < 0.0 < float(schema["maximum"])
                    ):
                        conditions += (
                            (
                                "negative",
                                SignalOperator.CUSTOM_CAUSAL,
                                "NEGATIVE",
                            ),
                        )
                else:
                    conditions = (
                        (
                            "rising",
                            SignalOperator.RISING,
                            "RISING",
                        ),
                    )
                    if (
                        schema["minimum"] is not None
                        and schema["maximum"] is not None
                        and float(schema["minimum"]) < 0.0 < float(schema["maximum"])
                    ):
                        conditions += (
                            (
                                "positive",
                                SignalOperator.CUSTOM_CAUSAL,
                                "POSITIVE",
                            ),
                        )
                for condition, operator, signal_kind in conditions:
                    identity = (
                        feature,
                        operator,
                        role,
                        signal_kind,
                    )
                    if identity in existing_identities:
                        continue
                    block_id = (
                        "auto__"
                        f"{canonical_slug(definition.canonical_name)}__"
                        f"{canonical_slug(feature)}__"
                        f"{condition}__{role.value.casefold()}"
                    )
                    if block_id in registry:
                        continue
                    selected_timeframes = tuple(
                        timeframe
                        for timeframe in definition.supported_timeframes
                        if timeframe in SUPPORTED_TIMEFRAMES
                    )
                    registry[block_id] = SignalBlock(
                        block_id=block_id,
                        version="1.0.0",
                        display_name=(f"{definition.display_name} {condition} as {role.value}"),
                        family=definition.family,
                        subfamily=definition.subfamily,
                        role=role,
                        direction=direction,
                        required_features=(feature,),
                        supported_timeframes=(selected_timeframes or tuple(SUPPORTED_TIMEFRAMES)),
                        warmup_bars=max(
                            1,
                            int(definition.warmup_bars),
                        ),
                        parameter_specs=(),
                        signal_kind=signal_kind,
                        feature=feature,
                        compare_feature=None,
                        knowability_timestamp=("BAR_CLOSE_PLUS_CAUSAL_LAG"),
                        missing_data_policy="REJECT",
                        description=(
                            "Registry-derived simple baseline for "
                            f"{definition.canonical_name}; schema discovery "
                            "uses no performance observations."
                        ),
                        redundancy_group=definition.redundancy_group,
                        computational_cost_class=(
                            "HIGH"
                            if definition.warmup_bars > 250
                            else "MEDIUM"
                            if definition.warmup_bars > 50
                            else "LOW"
                        ),
                        compatible_roles=(role,),
                        incompatible_blocks=(),
                        default_parameters={},
                        source_quality_requirements=tuple(
                            dict.fromkeys(
                                (
                                    "closed_candles_only",
                                    *definition.input_columns,
                                )
                            )
                        ),
                        operator=operator,
                    )
                    existing_identities.add(identity)
    for context_timeframe in SUPPORTED_TIMEFRAMES:
        execution_timeframes = tuple(
            timeframe
            for timeframe in SUPPORTED_TIMEFRAMES
            if TIMEFRAME_SECONDS[timeframe] < TIMEFRAME_SECONDS[context_timeframe]
        )
        if not execution_timeframes:
            continue
        route_definitions = (
            (
                "regime_bullish",
                f"htf_{context_timeframe}_regime_bullish",
                BlockRole.REGIME_FILTER,
                "MULTI_TIMEFRAME_REGIME",
                "Causal bullish regime on the last fully closed context bar.",
            ),
            (
                "trend_bullish",
                f"htf_{context_timeframe}_trend_bullish",
                BlockRole.CONFIRMATION,
                "MULTI_TIMEFRAME_TREND",
                "Causal bullish trend on the last fully closed context bar.",
            ),
            (
                "fractal_positive",
                f"fractal_alignment_{context_timeframe}",
                BlockRole.CONFIRMATION,
                "MULTI_TIMEFRAME_FRACTAL",
                "Positive confirmed-fractal state from the closed context bar.",
            ),
        )
        for suffix, feature, role, subfamily, description in route_definitions:
            block_id = f"mtf__{canonical_slug(context_timeframe)}__{suffix}"
            if block_id in registry:
                continue
            signal_kind = "POSITIVE" if suffix == "fractal_positive" else "BOOL"
            operator = (
                SignalOperator.CUSTOM_CAUSAL
                if suffix == "fractal_positive"
                else SignalOperator.BOOLEAN_TRUE
            )
            registry[block_id] = SignalBlock(
                block_id=block_id,
                version="1.0.0",
                display_name=(f"{context_timeframe} {suffix.replace('_', ' ')}"),
                family="MULTI_TIMEFRAME",
                subfamily=subfamily,
                role=role,
                direction=BlockDirection.BULLISH,
                required_features=(feature,),
                supported_timeframes=execution_timeframes,
                warmup_bars=200,
                parameter_specs=(),
                signal_kind=signal_kind,
                feature=feature,
                compare_feature=None,
                knowability_timestamp=("LAST_FULLY_CLOSED_HIGHER_TIMEFRAME_BAR"),
                missing_data_policy="REJECT",
                description=description,
                redundancy_group=(f"mtf_{context_timeframe}_{subfamily.casefold()}"),
                computational_cost_class="LOW",
                compatible_roles=(role,),
                incompatible_blocks=(),
                default_parameters={},
                source_quality_requirements=(
                    "closed_candles_only",
                    "backward_asof_alignment",
                ),
                operator=operator,
            )
    return registry


@dataclass(frozen=True)
class SearchCursor:
    complexity_index: int = 0
    membership_rank: int = 0
    logic_index: int = 0
    positions: dict[int, int] = field(default_factory=dict)
    next_complexity_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity_index": self.complexity_index,
            "membership_rank": self.membership_rank,
            "logic_index": self.logic_index,
            "positions": {str(key): int(value) for key, value in sorted(self.positions.items())},
            "next_complexity_index": self.next_complexity_index,
            "scheduling": "DETERMINISTIC_COMPLEXITY_ROUND_ROBIN",
        }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    temporary.replace(path)


def _unrank_combination(
    item_count: int,
    size: int,
    rank: int,
) -> tuple[int, ...]:
    total = math.comb(item_count, size)
    if rank < 0 or rank >= total:
        raise IndexError("combination rank is outside the search space")
    selected: list[int] = []
    start = 0
    remaining_rank = rank
    for position in range(size):
        remaining_slots = size - position - 1
        for candidate in range(start, item_count):
            suffix_count = (
                math.comb(
                    item_count - candidate - 1,
                    remaining_slots,
                )
                if remaining_slots
                else 1
            )
            if remaining_rank < suffix_count:
                selected.append(candidate)
                start = candidate + 1
                break
            remaining_rank -= suffix_count
    return tuple(selected)


def frequency_bucket(trades_per_year: float) -> str:
    value = max(0.0, float(trades_per_year))
    if value < 3.0:
        return "ULTRA_LOW_FREQUENCY"
    if value < 12.0:
        return "LOW_FREQUENCY"
    if value < 52.0:
        return "MEDIUM_FREQUENCY"
    if value < 250.0:
        return "HIGH_FREQUENCY"
    return "VERY_HIGH_FREQUENCY"


class SimpleStrategyResearchFactory:
    """Persist the complete known search space in deterministic resource batches."""

    def __init__(
        self,
        settings: Settings,
        *,
        output_dir: Path | None = None,
        registry: dict[str, SignalBlock] | None = None,
    ) -> None:
        self.settings = settings
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else settings.paths.output_dir / "research" / "simple_strategy_lab"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Research evidence belongs to the factory instance's own research root.
        # This keeps temporary/offline factories from inheriting the durable
        # production sync or trade-audit state while preserving the canonical
        # production layout: output/research/simple_strategy_lab -> output/research.
        self.research_evidence_dir = self.output_dir.parent
        self.registry = dict(registry or registry_driven_signal_blocks())
        self.generator = CombinationGenerator(self.registry)
        self.indicators = indicator_registry().definitions()
        registry_prefix = self.registry_hash[:16]
        self.queue_path = self.output_dir / f"generation_queue_{registry_prefix}.sqlite3"
        self.cursor_path = self.output_dir / f"generation_cursor_{registry_prefix}.json"
        preserved_registry_queues = tuple(
            str(path.resolve())
            for path in sorted(self.output_dir.glob("generation_queue_*.sqlite3"))
            if path.resolve() != self.queue_path.resolve()
        )
        preserved_registry_cursors = tuple(
            str(path.resolve())
            for path in sorted(self.output_dir.glob("generation_cursor_*.json"))
            if path.resolve() != self.cursor_path.resolve()
        )
        atomic_write_json(
            self.output_dir / "current_registry_state.json",
            {
                "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                "registry_hash": self.registry_hash,
                "registered_signal_blocks": len(self.registry),
                "queue_path": str(self.queue_path.resolve()),
                "cursor_path": str(self.cursor_path.resolve()),
                "legacy_queue_preserved": (self.output_dir / "generation_queue.sqlite3").is_file(),
                "legacy_cursor_preserved": (self.output_dir / "generation_cursor.json").is_file(),
                "preserved_registry_queue_count": len(preserved_registry_queues),
                "preserved_registry_queues": list(preserved_registry_queues),
                "preserved_registry_cursor_count": len(preserved_registry_cursors),
                "preserved_registry_cursors": list(preserved_registry_cursors),
                "updated_at": utc_iso(),
            },
        )
        self._initialize_queue()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.queue_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_queue(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_queue (
                    strategy_dna_hash TEXT PRIMARY KEY,
                    combination_id TEXT NOT NULL,
                    complexity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_strategy_queue_status
                ON strategy_queue(status, complexity)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _normalize_complexities(values: Iterable[int]) -> tuple[int, ...]:
        selected = tuple(sorted(set(int(value) for value in values)))
        if not selected or any(value < 1 or value > 5 for value in selected):
            raise ValueError("simple-lab complexities must be between one and five")
        return selected

    @staticmethod
    def _normalize_timeframes(values: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(normalize_timeframe(str(value)) for value in values))
        unknown = sorted(set(selected) - set(SUPPORTED_TIMEFRAMES))
        if unknown:
            raise ValueError(f"unsupported timeframes: {unknown}")
        return selected

    def _read_cursor(self) -> SearchCursor:
        if not self.cursor_path.is_file():
            return SearchCursor()
        payload = json.loads(self.cursor_path.read_text(encoding="utf-8"))
        return SearchCursor(
            complexity_index=int(payload.get("complexity_index") or 0),
            membership_rank=int(payload.get("membership_rank") or 0),
            logic_index=int(payload.get("logic_index") or 0),
            positions={
                int(key): int(value) for key, value in (payload.get("positions") or {}).items()
            },
            next_complexity_index=int(payload.get("next_complexity_index") or 0),
        )

    def _write_cursor(
        self,
        cursor: SearchCursor,
        *,
        complexities: Sequence[int],
        timeframes: Sequence[str],
        logic_modes: Sequence[LogicMode],
        complete: bool,
    ) -> None:
        atomic_write_json(
            self.cursor_path,
            {
                "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                **cursor.to_dict(),
                "complexities": list(complexities),
                "timeframes": list(timeframes),
                "logic_modes": [mode.value for mode in logic_modes],
                "complete": bool(complete),
                "updated_at": utc_iso(),
            },
        )

    def raw_space_size(
        self,
        *,
        complexities: Iterable[int] = DEFAULT_COMPLEXITIES,
        logic_modes: Iterable[LogicMode] = (LogicMode.LAYERED,),
    ) -> dict[int, int]:
        sizes = self._normalize_complexities(complexities)
        logic_count = len(tuple(dict.fromkeys(logic_modes)))
        return {size: math.comb(len(self.registry), size) * logic_count for size in sizes}

    def materialize_batch(
        self,
        *,
        batch_size: int = 2_000,
        complexities: Iterable[int] = DEFAULT_COMPLEXITIES,
        timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
        logic_modes: Iterable[LogicMode] = (LogicMode.LAYERED,),
        resume: bool = True,
    ) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        sizes = self._normalize_complexities(complexities)
        selected_timeframes = self._normalize_timeframes(timeframes)
        modes = tuple(dict.fromkeys(logic_modes))
        if not modes:
            raise ValueError("at least one logic mode is required")
        block_ids = tuple(sorted(self.registry))
        cursor = self._read_cursor() if resume else SearchCursor()
        inserted = 0
        duplicates = 0
        exclusions: Counter[str] = Counter()
        complete = False
        total_positions = {size: math.comb(len(block_ids), size) * len(modes) for size in sizes}
        if cursor.positions:
            positions = {
                size: min(
                    total_positions[size],
                    max(0, int(cursor.positions.get(size, 0))),
                )
                for size in sizes
            }
            next_complexity_index = cursor.next_complexity_index % len(sizes)
        else:
            # Backward-compatible migration from the original sequential
            # cursor. Completed lower complexities retain their exact
            # position; all untouched complexities start at rank zero.
            positions = {}
            for index, size in enumerate(sizes):
                if index < cursor.complexity_index:
                    positions[size] = total_positions[size]
                elif index == cursor.complexity_index:
                    positions[size] = min(
                        total_positions[size],
                        cursor.membership_rank * len(modes) + cursor.logic_index,
                    )
                else:
                    positions[size] = 0
            next_complexity_index = cursor.complexity_index % len(sizes)
        with self._connect() as connection:
            while inserted + duplicates < batch_size:
                if all(positions[size] >= total_positions[size] for size in sizes):
                    complete = True
                    break
                selected_index: int | None = None
                for offset in range(len(sizes)):
                    candidate_index = (next_complexity_index + offset) % len(sizes)
                    candidate_size = sizes[candidate_index]
                    if positions[candidate_size] < total_positions[candidate_size]:
                        selected_index = candidate_index
                        break
                if selected_index is None:
                    complete = True
                    break
                size = sizes[selected_index]
                position = positions[size]
                membership_rank = position // len(modes)
                logic_index = position % len(modes)
                indexes = _unrank_combination(
                    len(block_ids),
                    size,
                    membership_rank,
                )
                membership = tuple(block_ids[index] for index in indexes)
                logic_mode = modes[logic_index]
                combination = self.generator.materialize_membership(
                    membership,
                    logic_mode=logic_mode,
                    mode=GenerationMode.FAMILY_AWARE,
                    timeframes=selected_timeframes,
                )
                status = (
                    "QUEUED"
                    if combination.eligibility_status is CombinationState.GENERATED
                    else "EXCLUDED"
                )
                payload = combination.to_dict()
                now = utc_iso()
                result = connection.execute(
                    """
                    INSERT OR IGNORE INTO strategy_queue (
                        strategy_dna_hash,
                        combination_id,
                        complexity,
                        status,
                        reason,
                        payload_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        combination.strategy_dna_hash,
                        combination.combination_id,
                        combination.combination_size,
                        status,
                        combination.exclusion_reason,
                        json.dumps(
                            payload,
                            sort_keys=True,
                            default=str,
                        ),
                        now,
                        now,
                    ),
                )
                if result.rowcount:
                    inserted += 1
                    if status == "EXCLUDED":
                        exclusions[
                            combination.exclusion_reason or combination.eligibility_status.value
                        ] += 1
                else:
                    duplicates += 1
                positions[size] += 1
                next_complexity_index = (selected_index + 1) % len(sizes)
            connection.execute(
                """
                INSERT OR REPLACE INTO metadata(key, value_json)
                VALUES ('search_definition', ?)
                """,
                (
                    json.dumps(
                        {
                            "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                            "registry_hash": self.registry_hash,
                            "block_count": len(block_ids),
                            "complexities": list(sizes),
                            "timeframes": list(selected_timeframes),
                            "logic_modes": [mode.value for mode in modes],
                        },
                        sort_keys=True,
                    ),
                ),
            )
        complete = complete or all(positions[size] >= total_positions[size] for size in sizes)
        first_incomplete_index = next(
            (index for index, size in enumerate(sizes) if positions[size] < total_positions[size]),
            len(sizes),
        )
        legacy_size = (
            sizes[first_incomplete_index] if first_incomplete_index < len(sizes) else sizes[-1]
        )
        legacy_position = positions[legacy_size]
        next_cursor = SearchCursor(
            complexity_index=first_incomplete_index,
            membership_rank=legacy_position // len(modes),
            logic_index=legacy_position % len(modes),
            positions=positions,
            next_complexity_index=next_complexity_index,
        )
        self._write_cursor(
            next_cursor,
            complexities=sizes,
            timeframes=selected_timeframes,
            logic_modes=modes,
            complete=complete,
        )
        artifacts = self.build_inventory(
            complexities=sizes,
            timeframes=selected_timeframes,
            logic_modes=modes,
        )
        return {
            "status": "COMPLETE" if complete else "BATCH_COMPLETE",
            "batch_size": batch_size,
            "inserted": inserted,
            "duplicates": duplicates,
            "exclusions": dict(sorted(exclusions.items())),
            "cursor": next_cursor.to_dict(),
            "search_space_complete": complete,
            "queue": self.queue_status(
                complexities=sizes,
                logic_modes=modes,
            ),
            "artifacts": artifacts,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    @property
    def registry_hash(self) -> str:
        return stable_hash(
            [
                {
                    "block_id": block.block_id,
                    "version": block.version,
                    "configuration": block.to_dict(),
                }
                for block in sorted(
                    self.registry.values(),
                    key=lambda item: item.block_id,
                )
            ],
            length=64,
        )

    def queue_status(
        self,
        *,
        complexities: Iterable[int] = DEFAULT_COMPLEXITIES,
        logic_modes: Iterable[LogicMode] = (LogicMode.LAYERED,),
    ) -> dict[str, Any]:
        raw_by_size = self.raw_space_size(
            complexities=complexities,
            logic_modes=logic_modes,
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, complexity, COUNT(*) AS count
                FROM strategy_queue
                GROUP BY status, complexity
                ORDER BY status, complexity
                """
            ).fetchall()
            exclusion_rows = connection.execute(
                """
                SELECT COALESCE(reason, 'UNSPECIFIED') AS reason,
                       COUNT(*) AS count
                FROM strategy_queue
                WHERE status = 'EXCLUDED'
                GROUP BY COALESCE(reason, 'UNSPECIFIED')
                ORDER BY reason
                """
            ).fetchall()
        by_status: Counter[str] = Counter()
        by_complexity: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            by_status[str(row["status"])] += int(row["count"])
            by_complexity[str(row["complexity"])][str(row["status"])] += int(row["count"])
        registered = sum(by_status.values())
        total = sum(raw_by_size.values())
        cursor = self._read_cursor()
        materialized_attempts = sum(
            min(
                raw_by_size[size],
                max(0, int(cursor.positions.get(size, 0))),
            )
            for size in raw_by_size
        )
        excluded = int(by_status.get("EXCLUDED", 0))
        accepted = max(0, registered - excluded)
        executed = sum(
            int(by_status.get(status, 0))
            for status in (
                "BASELINE_COMPLETED",
                "EXACT_BACKTEST_COMPLETED",
            )
        )
        exclusion_reason_counts = {str(row["reason"]): int(row["count"]) for row in exclusion_rows}
        return {
            "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
            "registry_hash": self.registry_hash,
            "queue_path": str(self.queue_path.resolve()),
            "cursor_path": str(self.cursor_path.resolve()),
            "registered_signal_blocks": len(self.registry),
            "raw_search_space_by_complexity": {
                str(key): value for key, value in raw_by_size.items()
            },
            "total_known_raw_memberships": total,
            "total_persisted": registered,
            "total_remaining_to_materialize": max(
                0,
                total - materialized_attempts,
            ),
            "total_materialized_attempts": materialized_attempts,
            "total_possible_registry_combinations": total,
            "total_materialized_statically_valid": accepted,
            "total_materialized_causally_valid": accepted,
            "total_unique_dna": registered,
            "total_accepted_into_queue": accepted,
            "total_currently_queued": int(by_status.get("QUEUED", 0)),
            "total_executed": executed,
            "total_deduplicated": max(
                0,
                materialized_attempts - registered,
            ),
            "total_excluded": excluded,
            "exclusion_reason_counts": exclusion_reason_counts,
            "status_counts": dict(sorted(by_status.items())),
            "complexity_status_counts": {
                key: dict(sorted(value.items())) for key, value in sorted(by_complexity.items())
            },
            "cursor": cursor.to_dict(),
            "deduplicated": True,
            "resumable": True,
            "content_limit": None,
            "resource_limit_only": True,
            "orders_generated": 0,
            "orders_submitted": 0,
            "updated_at": utc_iso(),
        }

    def queued_strategies(
        self,
        *,
        limit: int,
        complexity: int | None = None,
        family: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("queue selection limit must be positive")
        requested_family = str(family or "").strip().casefold()

        def matches_family(payload: Mapping[str, Any]) -> bool:
            return not requested_family or requested_family in {
                str(value).casefold() for value in payload.get("families", [])
            }

        def load_complexity(
            connection: sqlite3.Connection,
            selected_complexity: int,
        ) -> list[dict[str, Any]]:
            """Load a deterministic slice without starving rare families."""

            selected: list[dict[str, Any]] = []
            last_hash = ""
            page_size = max(128, limit * 8)
            while len(selected) < limit:
                rows = connection.execute(
                    """
                    SELECT strategy_dna_hash, payload_json
                    FROM strategy_queue
                    WHERE status = 'QUEUED'
                      AND complexity = ?
                      AND strategy_dna_hash > ?
                    ORDER BY strategy_dna_hash
                    LIMIT ?
                    """,
                    (
                        int(selected_complexity),
                        last_hash,
                        page_size,
                    ),
                ).fetchall()
                if not rows:
                    break
                last_hash = str(rows[-1]["strategy_dna_hash"])
                for row in rows:
                    payload = json.loads(str(row["payload_json"]))
                    if matches_family(payload):
                        selected.append(payload)
                        if len(selected) >= limit:
                            break
            return selected

        with self._connect() as connection:
            complexities = (
                (int(complexity),)
                if complexity is not None
                else tuple(
                    int(row["complexity"])
                    for row in connection.execute(
                        """
                        SELECT DISTINCT complexity
                        FROM strategy_queue
                        WHERE status = 'QUEUED'
                        ORDER BY complexity
                        """
                    ).fetchall()
                )
            )
            candidates = {
                selected_complexity: load_complexity(
                    connection,
                    selected_complexity,
                )
                for selected_complexity in complexities
            }

        if complexity is not None:
            return candidates.get(int(complexity), [])[:limit]

        # Deterministic round-robin selection prevents the enormous two-block
        # search space from delaying all three- and four-block hypotheses.
        # A resource batch is therefore representative of every materialized
        # complexity while the persistent queue remains exhaustive.
        selected: list[dict[str, Any]] = []
        offsets = {value: 0 for value in complexities}
        while len(selected) < limit:
            progress = False
            for selected_complexity in complexities:
                offset = offsets[selected_complexity]
                available = candidates[selected_complexity]
                if offset >= len(available):
                    continue
                selected.append(available[offset])
                offsets[selected_complexity] = offset + 1
                progress = True
                if len(selected) >= limit:
                    break
            if not progress:
                break
        return selected

    def validation_schedule(
        self,
        *,
        cycle: int,
        queue: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prioritize standalone evidence, then rotate every registry family."""

        if cycle < 1:
            raise ValueError("validation cycle must be positive")
        queue_state = dict(queue or self.queue_status())
        complexity_status = queue_state.get("complexity_status_counts") or {}
        one_block_queued = int(
            (complexity_status.get("1") or complexity_status.get(1) or {}).get("QUEUED") or 0
        )
        families = sorted(
            {str(block.family) for block in self.registry.values() if str(block.family)}
        )
        priority = (
            "MARKET_STRUCTURE",
            "FRACTAL",
            "CANDLE",
            "CANDLESTICK",
        )
        family_rotation = [
            *(family for family in priority if family in families),
            *(family for family in families if family not in priority),
        ]
        if one_block_queued > 0:
            return {
                "phase": "STANDALONE_FIRST",
                "complexity": 1,
                "family": None,
                "one_block_remaining": one_block_queued,
                "family_rotation": family_rotation,
            }
        family = family_rotation[(cycle - 1) % len(family_rotation)] if family_rotation else None
        return {
            "phase": "FAMILY_ROUND_ROBIN",
            "complexity": None,
            "family": family,
            "one_block_remaining": 0,
            "family_rotation": family_rotation,
        }

    def update_strategy_status(
        self,
        strategy_hashes: Iterable[str],
        *,
        status: str,
        reason: str | None = None,
    ) -> int:
        hashes = tuple(dict.fromkeys(str(value) for value in strategy_hashes))
        if not hashes:
            return 0
        updated = 0
        now = utc_iso()
        with self._connect() as connection:
            for strategy_hash in hashes:
                result = connection.execute(
                    """
                    UPDATE strategy_queue
                    SET status = ?, reason = ?, updated_at = ?
                    WHERE strategy_dna_hash = ?
                    """,
                    (status, reason, now, strategy_hash),
                )
                updated += int(result.rowcount)
        return updated

    @staticmethod
    def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, str):
            decoded = json.loads(payload)
            return dict(decoded) if isinstance(decoded, Mapping) else {}
        return {}

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        default: float = 0.0,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _period_years(payload: Mapping[str, Any]) -> float:
        period = payload.get("data_period") or {}
        try:
            start = pd.Timestamp(period["start"])
            end = pd.Timestamp(period["end"])
        except (KeyError, TypeError, ValueError):
            return 1.0
        return max(
            1.0 / 365.25,
            (end - start).total_seconds() / (365.25 * 86_400.0),
        )

    @staticmethod
    def _evidence_priority(payload: Mapping[str, Any]) -> int:
        source = str(payload.get("source") or "").upper()
        return {
            "NORMAL": 60,
            "FINAL_HOLDOUT": 50,
            "DOUBLE_COST": 45,
            "STRESSED": 40,
            "EXACT_REAL": 30,
            "FAST_SCREEN_REAL": 10,
        }.get(source, 0)

    def _recent_canonical_payloads(
        self,
        database: Any,
        table_name: str,
        *,
        strategy_hashes: set[str],
        recent_limit: int,
    ) -> list[dict[str, Any]]:
        fetch_by_payload = getattr(
            database,
            "fetch_records_by_payload_values",
            None,
        )
        records = (
            fetch_by_payload(
                table_name,
                key="strategy_dna_hash",
                values=strategy_hashes,
            )
            if callable(fetch_by_payload)
            else database.fetch_recent_records(
                table_name,
                limit=recent_limit,
            )
        )
        selected: list[dict[str, Any]] = []
        for record in records:
            payload = self._record_payload(record)
            if (
                str(payload.get("strategy_dna_hash") or "") in strategy_hashes
                and str(payload.get("source") or "").upper() in CANONICAL_RESULT_SOURCES
            ):
                selected.append(payload)
        return selected

    def _registered_strategy_hashes(
        self,
        strategy_hashes: Iterable[str],
    ) -> set[str]:
        """Return candidate hashes that exist in the durable generation queue."""

        candidates = sorted({str(value) for value in strategy_hashes if str(value)})
        if not candidates:
            return set()
        registered: set[str] = set()
        with self._connect() as connection:
            for offset in range(0, len(candidates), 400):
                batch = candidates[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT strategy_dna_hash
                    FROM strategy_queue
                    WHERE strategy_dna_hash IN ({placeholders})
                      AND status != 'EXCLUDED'
                    """,
                    batch,
                ).fetchall()
                registered.update(str(row["strategy_dna_hash"]) for row in rows)
        return registered

    def _register_canonical_payloads(
        self,
        payloads: Iterable[Mapping[str, Any]],
    ) -> set[str]:
        """Register hash-verified canonical DNA found ahead of enumeration."""

        discovered: dict[str, StrategyCombination] = {}
        for payload in payloads:
            strategy_hash = str(payload.get("strategy_dna_hash") or "")
            block_ids = tuple(str(value) for value in payload.get("block_ids") or ())
            if (
                not strategy_hash
                or not 1 <= len(block_ids) <= 5
                or any(block_id not in self.registry for block_id in block_ids)
            ):
                continue
            timeframes = tuple(
                str(value) for value in (payload.get("timeframes_tested") or DEFAULT_TIMEFRAMES)
            )
            configured_mode = str(payload.get("logic_mode") or "").upper()
            modes = (
                (LogicMode(configured_mode),)
                if configured_mode in {mode.value for mode in LogicMode}
                else tuple(LogicMode)
            )
            for logic_mode in modes:
                combination = self.generator.materialize_membership(
                    block_ids,
                    logic_mode=logic_mode,
                    mode=GenerationMode.EXHAUSTIVE,
                    timeframes=timeframes,
                )
                if (
                    combination.strategy_dna_hash == strategy_hash
                    and combination.eligibility_status is CombinationState.GENERATED
                ):
                    discovered[strategy_hash] = combination
                    break
        if not discovered:
            return set()
        now = utc_iso()
        with self._connect() as connection:
            for combination in discovered.values():
                payload_json = json.dumps(
                    combination.to_dict(),
                    sort_keys=True,
                    default=str,
                )
                connection.execute(
                    """
                    INSERT INTO strategy_queue (
                        strategy_dna_hash,
                        combination_id,
                        complexity,
                        status,
                        reason,
                        payload_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, 'QUEUED', ?, ?, ?, ?)
                    ON CONFLICT(strategy_dna_hash) DO UPDATE SET
                        status = CASE
                            WHEN strategy_queue.status = 'EXCLUDED'
                             AND strategy_queue.reason =
                                 'REDUNDANT_INFORMATION_FAMILY'
                            THEN 'QUEUED'
                            ELSE strategy_queue.status
                        END,
                        reason = CASE
                            WHEN strategy_queue.status = 'EXCLUDED'
                             AND strategy_queue.reason =
                                 'REDUNDANT_INFORMATION_FAMILY'
                            THEN excluded.reason
                            ELSE strategy_queue.reason
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        combination.strategy_dna_hash,
                        combination.combination_id,
                        combination.combination_size,
                        "CANONICAL_EVIDENCE_DISCOVERED_OUT_OF_SEQUENCE",
                        payload_json,
                        now,
                        now,
                    ),
                )
        return set(discovered)

    def reconcile_available_canonical_results(
        self,
        database: Any,
        *,
        recent_limit: int = 50_000,
    ) -> dict[str, Any]:
        """Harvest available canonical evidence while another lab run is active.

        The canonical runner remains the only backtest engine.  This method
        intersects recent and restart-safe historical immutable identities
        with the persistent simple-lab queue and writes the same reconciled
        artifacts used after a normal dispatch.
        """

        candidate_hashes: set[str] = set()
        candidate_payloads: list[dict[str, Any]] = []
        historical_cursor_path = (
            self.output_dir / f"canonical_harvest_cursor_{self.registry_hash[:16]}.json"
        )
        historical_cursor = self._read_json_mapping(historical_cursor_path)
        if int(historical_cursor.get("harvest_version") or 0) != CANONICAL_HARVEST_VERSION:
            historical_cursor = {}
        historical_positions = {
            str(key): int(value)
            for key, value in (historical_cursor.get("positions") or {}).items()
        }
        next_historical_positions = dict(historical_positions)
        historical_batches: dict[str, list[dict[str, Any]]] = {}
        fetch_after_id = getattr(
            database,
            "fetch_records_after_id",
            None,
        )
        if callable(fetch_after_id):
            for table_name in (
                "experiment_trials",
                "exact_backtest_results",
            ):
                batch = fetch_after_id(
                    table_name,
                    after_id=historical_positions.get(
                        table_name,
                        0,
                    ),
                    limit=min(recent_limit, 5_000),
                )
                historical_batches[table_name] = batch
                record_ids = [int(record.get("id") or 0) for record in batch]
                if record_ids:
                    next_historical_positions[table_name] = max(record_ids)
        for table_name in (
            "experiment_trials",
            "exact_backtest_results",
        ):
            recent_records = database.fetch_recent_records(
                table_name,
                limit=recent_limit,
            )
            records = [
                *recent_records,
                *historical_batches.get(table_name, ()),
            ]
            for record in records:
                payload = self._record_payload(record)
                source = str(payload.get("source") or "").upper()
                strategy_hash = str(payload.get("strategy_dna_hash") or "")
                if strategy_hash and source in CANONICAL_RESULT_SOURCES:
                    candidate_hashes.add(strategy_hash)
                    candidate_payloads.append(payload)
        discovered = self._register_canonical_payloads(candidate_payloads)
        registered = self._registered_strategy_hashes(candidate_hashes)
        if not registered:
            if callable(fetch_after_id):
                atomic_write_json(
                    historical_cursor_path,
                    {
                        "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                        "harvest_version": CANONICAL_HARVEST_VERSION,
                        "registry_hash": self.registry_hash,
                        "positions": next_historical_positions,
                        "updated_at": utc_iso(),
                    },
                )
            return {
                "status": "NO_REGISTERED_CANONICAL_EVIDENCE",
                "candidate_strategy_count": len(candidate_hashes),
                "strategy_count_with_evidence": 0,
                "historical_records_scanned": sum(
                    len(batch) for batch in historical_batches.values()
                ),
                "historical_cursor_path": str(historical_cursor_path.resolve()),
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        result = self.reconcile_canonical_results(
            database,
            strategy_hashes=registered,
            recent_limit=recent_limit,
        )
        if callable(fetch_after_id):
            atomic_write_json(
                historical_cursor_path,
                {
                    "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                    "harvest_version": CANONICAL_HARVEST_VERSION,
                    "registry_hash": self.registry_hash,
                    "positions": next_historical_positions,
                    "updated_at": utc_iso(),
                },
            )
        return {
            **result,
            "candidate_strategy_count": len(candidate_hashes),
            "registered_candidate_count": len(registered),
            "registered_from_canonical_count": len(discovered),
            "historical_records_scanned": sum(len(batch) for batch in historical_batches.values()),
            "historical_cursor_path": str(historical_cursor_path.resolve()),
        }

    def reconcile_canonical_results(
        self,
        database: Any,
        *,
        strategy_hashes: Iterable[str],
        recent_limit: int = 50_000,
    ) -> dict[str, Any]:
        """Reconcile dispatched queue identities to actual canonical evidence.

        Fast-screen rows are used for signal-funnel and per-market frequency
        accounting.  Exact normal-cost evidence supersedes screening metrics
        only for the same immutable experiment hash.  This method never
        promotes, generates, or submits an order.
        """

        requested = {str(value) for value in strategy_hashes if str(value)}
        if not requested:
            return {
                "status": "NO_STRATEGIES_REQUESTED",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        baseline_payloads = [
            payload
            for payload in self._recent_canonical_payloads(
                database,
                "experiment_trials",
                strategy_hashes=requested,
                recent_limit=recent_limit,
            )
            if (
                str(payload.get("source") or "").upper() == "FAST_SCREEN_REAL"
                and str(payload.get("result_type") or "") in {"", "BASELINE_SCREEN"}
            )
        ]
        exact_payloads = self._recent_canonical_payloads(
            database,
            "exact_backtest_results",
            strategy_hashes=requested,
            recent_limit=recent_limit,
        )
        exact_by_experiment: dict[str, dict[str, Any]] = {}
        for payload in exact_payloads:
            experiment_hash = str(payload.get("experiment_hash") or "")
            current = exact_by_experiment.get(experiment_hash)
            if experiment_hash and (
                current is None
                or self._evidence_priority(payload) > self._evidence_priority(current)
            ):
                exact_by_experiment[experiment_hash] = payload

        # Retesting the same immutable experiment is idempotent.  Keep one
        # baseline row per experiment and prefer the latest stored record.
        baselines_by_experiment: dict[str, dict[str, Any]] = {}
        for payload in reversed(baseline_payloads):
            experiment_hash = str(payload.get("experiment_hash") or "")
            if experiment_hash:
                baselines_by_experiment.setdefault(
                    experiment_hash,
                    payload,
                )
        reconciled: list[dict[str, Any]] = []
        for experiment_hash, baseline in sorted(baselines_by_experiment.items()):
            evidence = exact_by_experiment.get(
                experiment_hash,
                baseline,
            )
            screening = baseline.get("screening") or {}
            funnel = screening.get("signal_funnel") or {}
            metrics = evidence.get("metrics") or {}
            source = str(evidence.get("source") or "")
            block_ids = tuple(str(value) for value in baseline.get("block_ids") or ())
            markets = tuple(str(value) for value in baseline.get("assets_tested") or ())
            timeframes = tuple(str(value) for value in baseline.get("timeframes_tested") or ())
            net_return = self._finite_number(
                metrics.get(
                    "net_return",
                    screening.get("screening_return"),
                )
            )
            profit_factor = self._finite_number(
                metrics.get(
                    "profit_factor",
                    screening.get("profit_factor"),
                )
            )
            completed_trades = int(
                metrics.get(
                    "trade_count",
                    funnel.get(
                        "completed_round_trip_count",
                        screening.get("trades"),
                    ),
                )
                or 0
            )
            strategy_variant_dna_hash = str(
                baseline.get("strategy_variant_dna_hash")
                or stable_hash(
                    {
                        "block_strategy_dna_hash": baseline.get("strategy_dna_hash"),
                        "parameters": baseline.get("parameters") or {},
                        "exit_model_version": baseline.get("exit_model_version"),
                    }
                )
            )
            reconciled.append(
                {
                    "strategy_dna_hash": str(baseline.get("strategy_dna_hash") or ""),
                    "strategy_variant_dna_hash": (strategy_variant_dna_hash),
                    "experiment_hash": experiment_hash,
                    "parameter_hash": str(baseline.get("parameter_hash") or ""),
                    "block_ids": block_ids,
                    "families": tuple(str(value) for value in baseline.get("families") or ()),
                    "markets": markets,
                    "market_scope": "|".join(markets),
                    "timeframe": (timeframes[0] if len(timeframes) == 1 else ""),
                    "status": (
                        "EXACT_REAL" if source.upper() != "FAST_SCREEN_REAL" else "FAST_SCREEN_REAL"
                    ),
                    "source": source,
                    "net_return": net_return,
                    "profit_factor": profit_factor,
                    "completed_trades": completed_trades,
                    "screening": screening,
                    "signal_funnel": funnel,
                    "data_period": dict(baseline.get("data_period") or {}),
                    "period_years": self._period_years(baseline),
                    "integrity": dict(evidence.get("integrity") or baseline.get("integrity") or {}),
                }
            )

        evidence_hashes = {row["strategy_dna_hash"] for row in reconciled}
        exact_hashes = {
            str(row["strategy_dna_hash"]) for row in reconciled if row["status"] == "EXACT_REAL"
        }
        self.update_strategy_status(
            requested - evidence_hashes,
            status="QUEUED",
            reason="NO_CANONICAL_RESULT_FOUND_RETRYABLE",
        )
        self.update_strategy_status(
            evidence_hashes - exact_hashes,
            status="BASELINE_COMPLETED",
            reason="CANONICAL_FAST_SCREEN_RECONCILED",
        )
        self.update_strategy_status(
            evidence_hashes & exact_hashes,
            status="EXACT_BACKTEST_COMPLETED",
            reason="CANONICAL_EXACT_RESULT_RECONCILED",
        )
        cumulative_rows = self._merge_canonical_evidence(reconciled)
        artifacts = self._write_reconciled_result_artifacts(
            cumulative_rows,
        )
        summary = {
            "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
            "status": "COMPLETE",
            "requested_strategy_count": len(requested),
            "strategy_count_with_evidence": len(evidence_hashes),
            "experiment_count": len(reconciled),
            "cumulative_experiment_count": len(cumulative_rows),
            "exact_strategy_count": len(exact_hashes & evidence_hashes),
            "baseline_only_strategy_count": len(evidence_hashes - exact_hashes),
            "missing_result_strategy_count": len(requested - evidence_hashes),
            "positive_after_costs_count": sum(
                row["net_return"] > 0.0 and row["profit_factor"] > 1.0 for row in reconciled
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
            "artifacts": artifacts,
            "updated_at": utc_iso(),
        }
        atomic_write_json(
            self.output_dir / "result_reconciliation_summary.json",
            summary,
        )
        self.write_objective_completion_audit()
        return summary

    def _merge_canonical_evidence(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge a completed batch into the durable canonical evidence index."""

        evidence_path = self.output_dir / "canonical_result_evidence.json"
        existing_rows: list[dict[str, Any]] = []
        if evidence_path.is_file():
            try:
                decoded = json.loads(evidence_path.read_text(encoding="utf-8"))
                candidates = decoded.get("rows") if isinstance(decoded, Mapping) else None
                if isinstance(candidates, list):
                    existing_rows = [dict(item) for item in candidates if isinstance(item, Mapping)]
            except (OSError, json.JSONDecodeError):
                existing_rows = []

        merged: dict[str, dict[str, Any]] = {}
        for raw_row in (*existing_rows, *rows):
            row = dict(raw_row)
            experiment_hash = str(row.get("experiment_hash") or "")
            if not experiment_hash:
                continue
            if not row.get("strategy_variant_dna_hash"):
                row["strategy_variant_dna_hash"] = stable_hash(
                    {
                        "block_strategy_dna_hash": row.get("strategy_dna_hash"),
                        "parameter_hash": row.get("parameter_hash"),
                        "experiment_hash": experiment_hash,
                    }
                )
            current = merged.get(experiment_hash)
            if (
                current is not None
                and current.get("status") == "EXACT_REAL"
                and row.get("status") != "EXACT_REAL"
            ):
                continue
            merged[experiment_hash] = row
        return [merged[key] for key in sorted(merged)]

    def _write_reconciled_result_artifacts(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, str]:
        single_indicator_rows: list[dict[str, Any]] = []
        single_condition_rows: list[dict[str, Any]] = []
        two_block_rows: list[dict[str, Any]] = []
        three_block_rows: list[dict[str, Any]] = []
        four_block_rows: list[dict[str, Any]] = []
        five_block_rows: list[dict[str, Any]] = []
        fractal_rows: list[dict[str, Any]] = []
        candlestick_rows: list[dict[str, Any]] = []
        timeframe_rows: list[dict[str, Any]] = []
        asset_rows: list[dict[str, Any]] = []
        frequency_rows: list[dict[str, Any]] = []
        funnel_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = []

        for row in rows:
            strategy_hash = str(row["strategy_dna_hash"])
            strategy_variant_hash = str(row["strategy_variant_dna_hash"])
            block_ids = tuple(row["block_ids"])
            market_scope = str(row["market_scope"])
            timeframe = str(row["timeframe"])
            common = {
                "strategy_dna_hash": strategy_hash,
                "strategy_variant_dna_hash": strategy_variant_hash,
                "market": market_scope,
                "timeframe": timeframe,
                "status": row["status"],
                "net_return": row["net_return"],
                "profit_factor": row["profit_factor"],
                "completed_trades": row["completed_trades"],
            }
            if len(block_ids) == 1:
                block = self.registry.get(block_ids[0])
                single_indicator_rows.append(
                    {
                        **common,
                        "indicator": (block.feature if block is not None else block_ids[0]),
                    }
                )
                single_condition_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "block_id": block_ids[0],
                        "market": market_scope,
                        "timeframe": timeframe,
                        "status": row["status"],
                        "raw_signals": int(
                            row["signal_funnel"].get(
                                "raw_entry_signal_count",
                                0,
                            )
                            or 0
                        ),
                        "completed_trades": row["completed_trades"],
                    }
                )
            if len(block_ids) == 2:
                two_block_rows.append(
                    {
                        **common,
                        "block_ids": "|".join(block_ids),
                    }
                )
            if len(block_ids) == 3:
                three_block_rows.append(
                    {
                        **common,
                        "block_ids": "|".join(block_ids),
                    }
                )
            if len(block_ids) == 4:
                four_block_rows.append(
                    {
                        **common,
                        "block_ids": "|".join(block_ids),
                    }
                )
            if len(block_ids) == 5:
                five_block_rows.append(
                    {
                        **common,
                        "block_ids": "|".join(block_ids),
                    }
                )
            if any(
                "fractal" in value.casefold()
                for value in (
                    *block_ids,
                    *tuple(row["families"]),
                )
            ):
                fractal_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "block_ids": "|".join(block_ids),
                        "market": market_scope,
                        "timeframe": timeframe,
                        "status": row["status"],
                    }
                )
            if any(
                str(family).upper() in {"CANDLE", "CANDLESTICK"} for family in row["families"]
            ) or any(
                token in block_id.casefold()
                for block_id in block_ids
                for token in (
                    "engulf",
                    "doji",
                    "hammer",
                    "morning_star",
                    "pin_bar",
                    "three_white",
                )
            ):
                candlestick_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "block_ids": "|".join(block_ids),
                        "market": market_scope,
                        "timeframe": timeframe,
                        "status": row["status"],
                    }
                )
            timeframe_rows.append(
                {key: common[key] for key in RESULT_FILES["timeframe_results.csv"]}
            )
            per_market = row["signal_funnel"].get("per_market") or {}
            for market in row["markets"]:
                market_funnel = dict(per_market.get(market) or {})
                market_trades = int(
                    market_funnel.get(
                        "completed_round_trip_count",
                        0,
                    )
                    or 0
                )
                asset_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "market": market,
                        "status": row["status"],
                        "net_return": self._finite_number(market_funnel.get("net_return")),
                        "profit_factor": self._finite_number(market_funnel.get("profit_factor")),
                        "completed_trades": market_trades,
                    }
                )
                trades_per_year = market_trades / max(
                    1.0 / 365.25,
                    float(row["period_years"]),
                )
                frequency_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "market": market,
                        "timeframe": timeframe,
                        "trades_per_year": trades_per_year,
                        "frequency_bucket": frequency_bucket(trades_per_year),
                    }
                )
                funnel_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "market": market,
                        "timeframe": timeframe,
                        "tradable_bars": int(
                            market_funnel.get(
                                "tradable_bar_count",
                                0,
                            )
                            or 0
                        ),
                        "raw_signals": int(
                            market_funnel.get(
                                "raw_entry_signal_count",
                                0,
                            )
                            or 0
                        ),
                        "edge_triggered_signals": int(
                            market_funnel.get(
                                "edge_trigger_count",
                                0,
                            )
                            or 0
                        ),
                        "blocked_existing_position": int(
                            market_funnel.get(
                                "blocked_existing_position",
                                0,
                            )
                            or 0
                        ),
                        "blocked_risk": int(
                            market_funnel.get(
                                "blocked_risk",
                                0,
                            )
                            or 0
                        ),
                        "completed_round_trips": market_trades,
                        "average_holding_bars": self._finite_number(
                            market_funnel.get("average_holding_bars")
                        ),
                    }
                )
            if (
                row["completed_trades"] <= 0
                or row["net_return"] <= 0.0
                or row["profit_factor"] <= 1.0
            ):
                rejection_rows.append(
                    {
                        "strategy_dna_hash": strategy_hash,
                        "strategy_variant_dna_hash": (strategy_variant_hash),
                        "status": "FAST_SCREEN_REJECTED",
                        "reason": (
                            "NO_COMPLETED_ROUND_TRIPS"
                            if row["completed_trades"] <= 0
                            else "NON_POSITIVE_AFTER_COSTS"
                        ),
                    }
                )

        metric_index = {
            (
                tuple(sorted(row["block_ids"])),
                str(row["timeframe"]),
                tuple(sorted(row["markets"])),
            ): row
            for row in rows
        }
        ablation_rows: list[dict[str, Any]] = []
        for parent in rows:
            parent_blocks = tuple(sorted(parent["block_ids"]))
            if len(parent_blocks) < 2:
                continue
            for removed in parent_blocks:
                child_blocks = tuple(block_id for block_id in parent_blocks if block_id != removed)
                child = metric_index.get(
                    (
                        child_blocks,
                        str(parent["timeframe"]),
                        tuple(sorted(parent["markets"])),
                    )
                )
                if child is None:
                    continue
                ablation_rows.append(
                    {
                        "parent_strategy_dna_hash": parent["strategy_dna_hash"],
                        "parent_strategy_variant_dna_hash": parent["strategy_variant_dna_hash"],
                        "ablation_strategy_dna_hash": child["strategy_dna_hash"],
                        "ablation_strategy_variant_dna_hash": child["strategy_variant_dna_hash"],
                        "removed_block": removed,
                        "net_return_delta": (parent["net_return"] - child["net_return"]),
                        "profit_factor_delta": (parent["profit_factor"] - child["profit_factor"]),
                        "trade_count_delta": (
                            parent["completed_trades"] - child["completed_trades"]
                        ),
                    }
                )

        outputs = {
            "single_indicator_results.csv": single_indicator_rows,
            "single_condition_results.csv": single_condition_rows,
            "two_block_results.csv": two_block_rows,
            "three_block_results.csv": three_block_rows,
            "four_block_results.csv": four_block_rows,
            "five_block_results.csv": five_block_rows,
            "fractal_results.csv": fractal_rows,
            "candlestick_results.csv": candlestick_rows,
            "timeframe_results.csv": timeframe_rows,
            "asset_results.csv": asset_rows,
            "frequency_buckets.csv": frequency_rows,
            "signal_funnels.csv": funnel_rows,
            "ablation_results.csv": ablation_rows,
            "rejection_reasons.csv": rejection_rows,
        }
        for filename, output_rows in outputs.items():
            _write_csv(
                self.output_dir / filename,
                RESULT_FILES[filename],
                output_rows,
            )
        evidence_rows = [
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in row.items()
                if key not in {"screening", "signal_funnel"}
            }
            | {
                "screening": row["screening"],
                "signal_funnel": row["signal_funnel"],
            }
            for row in rows
        ]
        atomic_write_json(
            self.output_dir / "canonical_result_evidence.json",
            {
                "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                "rows": evidence_rows,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        self._write_result_leaderboard(rows)
        return {
            filename: str((self.output_dir / filename).resolve())
            for filename in (
                *outputs,
                "canonical_result_evidence.json",
                "leaderboard.html",
            )
        }

    def _write_result_leaderboard(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        ranked = sorted(
            rows,
            key=lambda row: (
                row["net_return"] > 0.0 and row["profit_factor"] > 1.0,
                row["profit_factor"],
                row["net_return"],
                row["completed_trades"],
            ),
            reverse=True,
        )
        table_rows = "".join(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><code>{str(row['strategy_dna_hash'])[:16]}</code></td>"
            f"<td><code>{str(row['strategy_variant_dna_hash'])[:16]}</code></td>"
            f"<td>{' + '.join(row['block_ids'])}</td>"
            f"<td>{row['timeframe']}</td>"
            f"<td>{row['market_scope']}</td>"
            f"<td>{float(row['net_return']):.4f}</td>"
            f"<td>{float(row['profit_factor']):.3f}</td>"
            f"<td>{int(row['completed_trades'])}</td>"
            f"<td>{row['status']}</td>"
            "</tr>"
            for index, row in enumerate(ranked[:500], start=1)
        )
        (self.output_dir / "leaderboard.html").write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Simple Strategy Lab — canonical results</title>"
            "<style>body{font-family:system-ui;max-width:1400px;"
            "margin:2rem auto;padding:0 1rem}table{border-collapse:"
            "collapse;width:100%}td,th{border:1px solid #ddd;"
            "padding:.35rem;text-align:left}code{font-size:.85em}"
            "</style></head><body><h1>Canonical real-data results</h1>"
            "<p>Fast screens are cost-aware; exact evidence is labelled "
            "separately. Orders generated: 0; orders submitted: 0.</p>"
            "<table><thead><tr><th>Rank</th><th>Block DNA</th>"
            "<th>Variant DNA</th><th>Blocks</th>"
            "<th>TF</th><th>Markets</th><th>Net return</th><th>PF</th>"
            "<th>Round trips</th><th>Evidence</th></tr></thead><tbody>"
            f"{table_rows}</tbody></table></body></html>",
            encoding="utf-8",
        )

    def _mapped_indicator_blocks(
        self,
        indicator: IndicatorDefinition,
    ) -> tuple[str, ...]:
        names = {
            indicator.canonical_name.casefold(),
            *(value.casefold() for value in indicator.output_columns),
        }
        return tuple(
            sorted(
                block.block_id
                for block in self.registry.values()
                if (
                    block.feature.casefold() in names
                    or block.block_id.casefold() in names
                    or indicator.canonical_name.casefold() in block.block_id.casefold()
                )
            )
        )

    @staticmethod
    def _read_json_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}

    @staticmethod
    def _csv_rows(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error):
            return []

    def write_objective_completion_audit(
        self,
        *,
        queue: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reconcile the automated research objective against durable evidence."""

        queue_state = dict(
            queue or self._read_json_mapping(self.output_dir / "generation_queue_status.json")
        )
        result_state = self._read_json_mapping(
            self.output_dir / "result_reconciliation_summary.json"
        )
        service_state = self._read_json_mapping(self.output_dir / "service_status.json")
        data_sync_state = self._read_json_mapping(
            self.research_evidence_dir / "data_sync_progress.json"
        )
        timeframe_rows = self._csv_rows(self.output_dir / "timeframe_results.csv")
        fractal_rows = self._csv_rows(self.output_dir / "fractal_results.csv")
        candlestick_rows = self._csv_rows(self.output_dir / "candlestick_results.csv")
        frequency_rows = self._csv_rows(self.output_dir / "frequency_buckets.csv")
        funnel_rows = self._csv_rows(self.output_dir / "signal_funnels.csv")
        ablation_rows = self._csv_rows(self.output_dir / "ablation_results.csv")
        trade_count_audit = self._read_json_mapping(
            self.research_evidence_dir / "trade_count_audit" / "summary.json"
        )
        complexity_names = {
            1: "single_condition",
            2: "two_block",
            3: "three_block",
            4: "four_block",
            5: "five_block",
        }
        complexity_rows = {
            complexity: self._csv_rows(
                self.output_dir / f"{complexity_names[complexity]}_results.csv"
            )
            for complexity in DEFAULT_COMPLEXITIES
        }
        per_timeframe: dict[str, dict[str, Any]] = {}
        for row in timeframe_rows:
            timeframe = str(row.get("timeframe") or "")
            if not timeframe:
                continue
            bucket = per_timeframe.setdefault(
                timeframe,
                {
                    "experiment_count": 0,
                    "completed_round_trips": 0,
                    "positive_after_costs_count": 0,
                },
            )
            bucket["experiment_count"] += 1
            try:
                bucket["completed_round_trips"] += int(float(row.get("completed_trades") or 0))
                net_return = float(row.get("net_return") or 0.0)
                profit_factor = float(row.get("profit_factor") or 0.0)
            except (TypeError, ValueError):
                continue
            if net_return > 0.0 and profit_factor > 1.0:
                bucket["positive_after_costs_count"] += 1

        expected_timeframes = set(DEFAULT_TIMEFRAMES)
        configured_timeframes = {
            str(value)
            for value in (
                self._read_json_mapping(self.output_dir / "generation_summary.json").get(
                    "timeframes"
                )
                or ()
            )
        }
        data_sync_complete = bool(data_sync_state) and (
            data_sync_state.get("status") == "COMPLETE"
            or (
                int(data_sync_state.get("total_operations") or 0) > 0
                and int(data_sync_state.get("completed_operations") or 0)
                == int(data_sync_state.get("total_operations") or 0)
                and int(data_sync_state.get("failure_count") or 0) == 0
            )
        )
        required_complexities = (1, 2, 3, 4)
        complexity_evidence = {
            str(complexity): {
                "result_rows": len(complexity_rows[complexity]),
                "evidence_present": bool(complexity_rows[complexity]),
            }
            for complexity in DEFAULT_COMPLEXITIES
        }
        one_block_status = (queue_state.get("complexity_status_counts") or {}).get("1") or {}

        def block_count(row: Mapping[str, Any]) -> int:
            value = str(row.get("block_ids") or "").strip()
            return len(value.split("|")) if value else 0

        invariants = {
            "content_limit_is_none": (queue_state.get("content_limit") is None),
            "resource_limits_only": bool(queue_state.get("resource_limit_only")),
            "resumable": bool(queue_state.get("resumable")),
            "deduplicated": bool(queue_state.get("deduplicated")),
            "all_supported_timeframes_configured": (expected_timeframes <= configured_timeframes),
            "one_to_four_block_evidence_present": all(
                complexity_evidence[str(complexity)]["evidence_present"]
                for complexity in required_complexities
            ),
            "standalone_queue_complete": (int(one_block_status.get("QUEUED") or 0) == 0),
            "required_simple_timeframes_evidenced": {
                "15m",
                "1h",
                "4h",
                "1d",
            }
            <= set(per_timeframe),
            "fractal_standalone_evidenced": any(block_count(row) == 1 for row in fractal_rows),
            "fractal_pair_evidenced": any(block_count(row) == 2 for row in fractal_rows),
            "candlestick_standalone_evidenced": any(
                block_count(row) == 1 for row in candlestick_rows
            ),
            "frequency_buckets_present": bool(frequency_rows),
            "signal_funnels_present": bool(funnel_rows),
            "ablation_present": bool(ablation_rows),
            "trade_count_audit_complete": (trade_count_audit.get("status") == "COMPLETE"),
            "canonical_results_reconciled": (result_state.get("status") == "COMPLETE"),
            "research_orders_zero": (
                int(queue_state.get("orders_generated") or 0) == 0
                and int(queue_state.get("orders_submitted") or 0) == 0
                and int(result_state.get("orders_generated") or 0) == 0
                and int(result_state.get("orders_submitted") or 0) == 0
                and int(data_sync_state.get("live_orders") or 0) == 0
            ),
            "maximum_data_sync_complete": data_sync_complete,
        }
        audit = {
            "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
            "status": ("COMPLETE" if all(invariants.values()) else "IN_PROGRESS"),
            "requirements": {
                "maximum_real_data_fetch": {
                    "status": ("COMPLETE" if data_sync_complete else "IN_PROGRESS"),
                    "evidence": data_sync_state,
                },
                "open_registry_generation": {
                    "registered_signal_blocks": int(
                        queue_state.get("registered_signal_blocks") or 0
                    ),
                    "known_raw_memberships": int(
                        queue_state.get("total_known_raw_memberships") or 0
                    ),
                    "persisted_unique_dna": int(queue_state.get("total_persisted") or 0),
                    "remaining_to_materialize": int(
                        queue_state.get("total_remaining_to_materialize") or 0
                    ),
                },
                "block_complexity_evidence": complexity_evidence,
                "simple_family_evidence": {
                    "fractal_rows": len(fractal_rows),
                    "fractal_standalone_rows": sum(block_count(row) == 1 for row in fractal_rows),
                    "fractal_pair_rows": sum(block_count(row) == 2 for row in fractal_rows),
                    "candlestick_rows": len(candlestick_rows),
                    "candlestick_standalone_rows": sum(
                        block_count(row) == 1 for row in candlestick_rows
                    ),
                },
                "timeframe_trade_evidence": per_timeframe,
                "multi_timeframe_routes": {
                    "route_count": len(
                        self._csv_rows(self.output_dir / "multi_timeframe_route_coverage.csv")
                    )
                },
                "canonical_validation": result_state,
                "trade_count_audit": trade_count_audit,
                "continuous_service": {
                    "status": service_state.get("status"),
                    "cycle": service_state.get("cycle"),
                    "pid": service_state.get("pid"),
                    "validation_schedule": service_state.get("validation_schedule"),
                },
            },
            "invariants": invariants,
            "orders_generated": 0,
            "orders_submitted": 0,
            "updated_at": utc_iso(),
        }
        atomic_write_json(
            self.output_dir / "objective_completion_audit.json",
            audit,
        )
        return audit

    def build_inventory(
        self,
        *,
        complexities: Iterable[int] = DEFAULT_COMPLEXITIES,
        timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
        logic_modes: Iterable[LogicMode] = (LogicMode.LAYERED,),
    ) -> dict[str, str]:
        sizes = self._normalize_complexities(complexities)
        selected_timeframes = self._normalize_timeframes(timeframes)
        modes = tuple(dict.fromkeys(logic_modes))
        queue = self.queue_status(
            complexities=sizes,
            logic_modes=modes,
        )
        definitions = sorted(
            self.indicators,
            key=lambda item: (item.family, item.canonical_name),
        )
        registry_payload = {
            "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
            "registry_hash": self.registry_hash,
            "signal_blocks": [
                block.to_dict()
                for block in sorted(
                    self.registry.values(),
                    key=lambda item: item.block_id,
                )
            ],
            "formal_indicators": [definition.to_dict() for definition in definitions],
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(self.output_dir / "registry.json", registry_payload)
        summary = {
            **queue,
            "formal_indicator_count": len(definitions),
            "tradable_indicator_count": sum(definition.tradable for definition in definitions),
            "combinable_indicator_count": sum(definition.combinable for definition in definitions),
            "timeframes": list(selected_timeframes),
            "logic_modes": [mode.value for mode in modes],
            "generation_is_separate_from_validation": True,
            "examples_are_not_whitelists": True,
        }
        atomic_write_json(
            self.output_dir / "generation_summary.json",
            summary,
        )
        atomic_write_json(
            self.output_dir / "complete_search_space_summary.json",
            summary,
        )
        atomic_write_json(
            self.output_dir / "generation_queue_status.json",
            queue,
        )

        indicator_rows = []
        for definition in definitions:
            mapped = self._mapped_indicator_blocks(definition)
            indicator_rows.append(
                {
                    "canonical_name": definition.canonical_name,
                    "family": definition.family,
                    "status": definition.status.value,
                    "tradable": definition.tradable,
                    "combinable": definition.combinable,
                    "compatible_roles": "|".join(
                        role.value for role in definition.compatible_roles
                    ),
                    "supported_timeframes": "|".join(definition.supported_timeframes),
                    "mapped_executable_blocks": "|".join(mapped),
                    "mapped_block_count": len(mapped),
                    "coverage_status": (
                        "EXECUTABLE_BLOCK_MAPPED"
                        if mapped
                        else (
                            "DATA_OR_IMPLEMENTATION_PENDING"
                            if definition.combinable
                            else definition.status.value
                        )
                    ),
                }
            )
        _write_csv(
            self.output_dir / "indicator_condition_coverage.csv",
            tuple(indicator_rows[0]) if indicator_rows else (),
            indicator_rows,
        )

        blocks_by_family: defaultdict[str, list[SignalBlock]] = defaultdict(list)
        for block in self.registry.values():
            blocks_by_family[block.family].append(block)
        family_pair_rows = []
        for left_index, left in enumerate(sorted(blocks_by_family)):
            for right in sorted(blocks_by_family)[left_index:]:
                left_count = len(blocks_by_family[left])
                right_count = len(blocks_by_family[right])
                possible = math.comb(left_count, 2) if left == right else left_count * right_count
                family_pair_rows.append(
                    {
                        "left_family": left,
                        "right_family": right,
                        "left_blocks": left_count,
                        "right_blocks": right_count,
                        "possible_pairs": possible,
                        "generation_policy": "REGISTRY_DRIVEN",
                    }
                )
        _write_csv(
            self.output_dir / "family_pair_coverage.csv",
            tuple(family_pair_rows[0]) if family_pair_rows else (),
            family_pair_rows,
        )

        role_counts = Counter(block.role.value for block in self.registry.values())
        role_rows = [
            {
                "role": role.value,
                "registered_blocks": role_counts[role.value],
                "supported_by_generator": True,
            }
            for role in BlockRole
        ]
        _write_csv(
            self.output_dir / "block_role_coverage.csv",
            tuple(role_rows[0]),
            role_rows,
        )

        parameter_rows = []
        for block in sorted(self.registry.values(), key=lambda item: item.block_id):
            if not block.parameter_specs:
                parameter_rows.append(
                    {
                        "block_id": block.block_id,
                        "family": block.family,
                        "parameter": "",
                        "minimum": "",
                        "maximum": "",
                        "step": "",
                        "value_count": 1,
                        "timeframe_aware": False,
                    }
                )
            for specification in block.parameter_specs:
                parameter_rows.append(
                    {
                        "block_id": block.block_id,
                        "family": block.family,
                        "parameter": specification.name,
                        "minimum": specification.minimum,
                        "maximum": specification.maximum,
                        "step": specification.step,
                        "value_count": len(specification.values()),
                        "timeframe_aware": (specification.kind.value == "TIMEFRAME"),
                    }
                )
        _write_csv(
            self.output_dir / "parameter_space_coverage.csv",
            tuple(parameter_rows[0]),
            parameter_rows,
        )

        timeframe_rows = []
        for timeframe in selected_timeframes:
            supported = [
                block for block in self.registry.values() if timeframe in block.supported_timeframes
            ]
            timeframe_rows.append(
                {
                    "timeframe": timeframe,
                    "registered_blocks": len(supported),
                    "entry_blocks": sum(
                        block.role is BlockRole.ENTRY_TRIGGER for block in supported
                    ),
                    "native_or_resample_required": True,
                    "closed_candle_only": True,
                    "coverage_status": ("GENERATABLE" if supported else "NO_REGISTERED_BLOCKS"),
                }
            )
        _write_csv(
            self.output_dir / "timeframe_coverage.csv",
            tuple(timeframe_rows[0]),
            timeframe_rows,
        )
        timeframe_routes = [
            {
                "execution_timeframe": execution,
                "context_timeframe": context,
                "execution_seconds": TIMEFRAME_SECONDS[execution],
                "context_seconds": TIMEFRAME_SECONDS[context],
                "alignment": "LAST_FULLY_CLOSED_BACKWARD_ASOF",
                "causal": True,
                "executable_block_ids": "|".join(
                    sorted(
                        block.block_id
                        for block in self.registry.values()
                        if (
                            block.family == "MULTI_TIMEFRAME"
                            and execution in block.supported_timeframes
                            and f"__{canonical_slug(context)}__" in block.block_id
                        )
                    )
                ),
                "generation_status": (
                    "EXECUTABLE_ROUTE"
                    if any(
                        block.family == "MULTI_TIMEFRAME"
                        and execution in block.supported_timeframes
                        and f"__{canonical_slug(context)}__" in block.block_id
                        for block in self.registry.values()
                    )
                    else "REGISTERED_ROUTE_DATA_PENDING"
                ),
            }
            for execution in selected_timeframes
            for context in selected_timeframes
            if TIMEFRAME_SECONDS[context] > TIMEFRAME_SECONDS[execution]
        ]
        _write_csv(
            self.output_dir / "multi_timeframe_route_coverage.csv",
            (
                tuple(timeframe_routes[0])
                if timeframe_routes
                else (
                    "execution_timeframe",
                    "context_timeframe",
                    "alignment",
                    "causal",
                    "generation_status",
                )
            ),
            timeframe_routes,
        )
        summary["multi_timeframe_route_count"] = len(timeframe_routes)
        summary["single_timeframe_count"] = len(selected_timeframes)
        summary["membership_timeframe_experiment_upper_bound"] = int(
            summary["total_known_raw_memberships"]
            * (len(selected_timeframes) + len(timeframe_routes))
        )
        atomic_write_json(
            self.output_dir / "generation_summary.json",
            summary,
        )
        atomic_write_json(
            self.output_dir / "complete_search_space_summary.json",
            summary,
        )

        exit_rows = [
            {
                "exit_family": "FIXED_R",
                "mechanism": "ATR stop plus ATR target",
                "parameters": ("stop_atr=1.5|2|2.5|3|4|6;target_atr=2|3|4|6|10|20"),
                "executed_by_canonical_strategy": True,
            },
            {
                "exit_family": "TRAILING_TREND",
                "mechanism": "ATR stop plus ATR trailing stop",
                "parameters": "trailing_atr=1|1.5|2|2.5|3|4",
                "executed_by_canonical_strategy": True,
            },
            {
                "exit_family": "TIME_REGIME",
                "mechanism": ("maximum holding bars or bearish regime"),
                "parameters": ("maximum_holding_bars=48|120|240|480|720"),
                "executed_by_canonical_strategy": True,
            },
            {
                "exit_family": "TIME_ONLY",
                "mechanism": "maximum holding bars with ATR risk stop",
                "parameters": ("maximum_holding_bars=48|120|240|480|720"),
                "executed_by_canonical_strategy": True,
            },
        ]
        profile_mechanisms = {
            "ATR_STOP_ONLY": "ATR stop; no reachable target or time exit",
            "ATR_TRAILING_ONLY": ("ATR stop plus ATR trailing; no reachable target"),
            "EMA20_EXIT": "close below EMA20",
            "EMA50_EXIT": "close below EMA50",
            "VWAP_EXIT": "close below rolling VWAP20",
            "SUPERTREND_EXIT": "Supertrend direction bearish",
            "RSI_EXTREME_EXIT": "RSI14 at or above 75",
            "MOMENTUM_LOSS_EXIT": "ROC12 below zero",
            "VOLUME_WEAKNESS_EXIT": ("relative volume below 0.75 and close below EMA20"),
            "ADX_WEAKNESS_EXIT": "ADX14 below 18",
            "N_BAR_LOW_EXIT": "close below prior Donchian 20-bar low",
            "FRACTAL_LOW_EXIT": ("close below last confirmed causal fractal low"),
            "REGIME_EXIT": "bear regime or close below EMA200",
        }
        exit_rows.extend(
            {
                "exit_family": family,
                "mechanism": mechanism,
                "parameters": "ATR risk stop retained",
                "executed_by_canonical_strategy": True,
            }
            for family, mechanism in profile_mechanisms.items()
        )
        _write_csv(
            self.output_dir / "exit_matrix_coverage.csv",
            tuple(exit_rows[0]),
            exit_rows,
        )

        indicator_family_counts = Counter(
            definition.family for definition in definitions if definition.combinable
        )
        executable_family_counts = Counter(block.family for block in self.registry.values())
        underexplored = [
            {
                "family": family,
                "combinable_indicators": count,
                "executable_blocks": executable_family_counts.get(
                    family,
                    0,
                ),
                "coverage_ratio": float(executable_family_counts.get(family, 0) / max(1, count)),
                "priority": ("HIGH" if executable_family_counts.get(family, 0) == 0 else "MEDIUM"),
            }
            for family, count in sorted(indicator_family_counts.items())
            if executable_family_counts.get(family, 0) < count
        ]
        atomic_write_json(
            self.output_dir / "underexplored_families.json",
            {
                "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                "rows": underexplored,
            },
        )
        atomic_write_json(
            self.output_dir / "deduplication_summary.json",
            {
                "schema_version": SIMPLE_LAB_SCHEMA_VERSION,
                "strategy_identity": "SHA256_CANONICAL_BLOCK_VERSIONS_AND_LOGIC",
                "database_primary_key": "strategy_dna_hash",
                "deduplicated": True,
                "persisted_unique": queue["total_persisted"],
                "materialized_attempts": queue["total_materialized_attempts"],
                "duplicates_suppressed": queue["total_deduplicated"],
                "content_limit": None,
            },
        )

        for filename, headers in RESULT_FILES.items():
            path = self.output_dir / filename
            if not path.is_file():
                _write_csv(path, headers, ())
        report = self._html_report(summary, underexplored)
        (self.output_dir / "report.html").write_text(report, encoding="utf-8")
        (self.output_dir / "leaderboard.html").write_text(
            self._html_report(
                summary,
                underexplored,
                title="Simple Strategy Lab — leaderboard pending executions",
            ),
            encoding="utf-8",
        )
        self.write_objective_completion_audit(queue=queue)
        return {
            path.name: str(path) for path in sorted(self.output_dir.iterdir()) if path.is_file()
        }

    @staticmethod
    def _html_report(
        summary: dict[str, Any],
        underexplored: Sequence[dict[str, Any]],
        *,
        title: str = "Simple Strategy Research Factory",
    ) -> str:
        rows = "".join(
            "<tr>"
            f"<td>{row['family']}</td>"
            f"<td>{row['combinable_indicators']}</td>"
            f"<td>{row['executable_blocks']}</td>"
            f"<td>{row['priority']}</td>"
            "</tr>"
            for row in underexplored
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;"
            "padding:0 1rem}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:.45rem;text-align:left}"
            "code{background:#eee;padding:.15rem .3rem}</style></head><body>"
            f"<h1>{title}</h1>"
            "<p>Generation is registry-driven and exhaustive in deterministic "
            "resource batches. Deep validation remains selective.</p>"
            f"<p>Blocks: <b>{summary['registered_signal_blocks']}</b>; "
            f"known raw 1–5 block memberships: "
            f"<b>{summary['total_known_raw_memberships']:,}</b>; persisted: "
            f"<b>{summary['total_persisted']:,}</b>; remaining: "
            f"<b>{summary['total_remaining_to_materialize']:,}</b>.</p>"
            "<p>Orders generated: <b>0</b>; orders submitted: <b>0</b>.</p>"
            "<h2>Underexplored formal indicator families</h2>"
            "<table><thead><tr><th>Family</th><th>Combinable indicators</th>"
            "<th>Executable blocks</th><th>Priority</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></body></html>"
        )


__all__ = [
    "DEFAULT_COMPLEXITIES",
    "DEFAULT_TIMEFRAMES",
    "SIMPLE_LAB_SCHEMA_VERSION",
    "SearchCursor",
    "SimpleStrategyResearchFactory",
    "frequency_bucket",
]
