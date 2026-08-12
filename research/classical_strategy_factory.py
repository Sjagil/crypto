"""Bounded, deterministic classical strategy-DNA factory.

The factory owns preregistration and grammar validation only.  Signal
calculation, fills, costs, exact backtests, statistics, ledgers and lifecycle
remain in their canonical modules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from config.settings import TIMEFRAME_SECONDS
from research.combinatorial_lab import (
    CLASSICAL_DISABLED_FAMILY_INTERFACES,
    CLASSICAL_ECONOMIC_FAMILY_TEMPLATES,
    BlockRole,
    ExitProfile,
    signal_block_registry,
)
from utils.common import stable_hash

CLASSICAL_FACTORY_VERSION = "1.0.0"
CLASSICAL_FACTORY_SEED = 20260728
CLASSICAL_FACTORY_DEFAULT_TRIALS = 2_000
CLASSICAL_RESEARCH_UNIVERSE = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
    "NPC-EUR",
)
CLASSICAL_PROMOTION_UNIVERSE = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
    "NPC-EUR",
)


@dataclass(frozen=True, slots=True)
class TimeframeRoute:
    route_id: str
    regime_timeframe: str
    setup_timeframe: str
    signal_timeframe: str


TIMEFRAME_ROUTES = (
    TimeframeRoute("MTF_1W_1D_4H", "1W", "1d", "4h"),
    TimeframeRoute("MTF_1D_4H_1H", "1d", "4h", "1h"),
    TimeframeRoute("MTF_4H_1H_15M", "4h", "1h", "15m"),
    TimeframeRoute("MTF_1H_15M_5M", "1h", "15m", "5m"),
    TimeframeRoute("SINGLE_1D", "1d", "1d", "1d"),
    TimeframeRoute("SINGLE_4H", "4h", "4h", "4h"),
    TimeframeRoute("SINGLE_1H", "1h", "1h", "1h"),
    TimeframeRoute("SINGLE_15M", "15m", "15m", "15m"),
    TimeframeRoute("SINGLE_5M", "5m", "5m", "5m"),
)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    policy_id: str
    exit_profile: str
    stop_atr: float
    target_atr: float
    trailing_atr: float
    maximum_holding_bars: int
    sizing_policy: str
    risk_fraction: float
    position_fraction: float


EXECUTION_POLICIES = (
    ExecutionPolicy("FIXED_R_15_30", ExitProfile.FIXED_R.value, 1.5, 3.0, 0.0, 120, "RISK_PER_STOP", 0.0025, 1.0),
    ExecutionPolicy("FIXED_R_20_30", ExitProfile.FIXED_R.value, 2.0, 3.0, 0.0, 120, "RISK_PER_STOP", 0.0050, 1.0),
    ExecutionPolicy("FIXED_R_20_40", ExitProfile.FIXED_R.value, 2.0, 4.0, 0.0, 240, "RISK_PER_STOP", 0.0050, 0.75),
    ExecutionPolicy("FIXED_R_25_40", ExitProfile.FIXED_R.value, 2.5, 4.0, 0.0, 240, "MAXIMUM_LOSS_CONSTRAINED", 0.0050, 0.75),
    ExecutionPolicy("TRAILING_20_25", ExitProfile.TRAILING_TREND.value, 2.0, 20.0, 2.5, 480, "RISK_PER_STOP", 0.0050, 1.0),
    ExecutionPolicy("TRAILING_30_40", ExitProfile.TRAILING_TREND.value, 3.0, 20.0, 4.0, 720, "MAXIMUM_LOSS_CONSTRAINED", 0.0025, 1.0),
    ExecutionPolicy("SIGNAL_TIME_20", ExitProfile.TIME_REGIME.value, 2.0, 20.0, 0.0, 48, "FIXED_POSITION_FRACTION", 0.0025, 0.50),
    ExecutionPolicy("SIGNAL_TIME_25", ExitProfile.TIME_REGIME.value, 2.5, 20.0, 0.0, 120, "FIXED_POSITION_FRACTION", 0.0050, 0.75),
)


@dataclass(frozen=True, slots=True)
class ClassicalStrategyDNA:
    family: str
    hypothesis: str
    block_ids: tuple[str, ...]
    universe: tuple[str, ...]
    route: TimeframeRoute
    execution_policy: ExecutionPolicy
    regime_block: str | None
    trigger_block: str
    confirmation_blocks: tuple[str, ...]
    exit_block: str | None
    stop_method: str
    execution_rule: str = "SIGNAL_CLOSE_NEXT_OPEN"
    cost_profile: str = "NORMAL_AND_STRESSED"
    cooldown_bars: int = 1
    promotion_eligibility: str = "PRACTICAL_GOVERNANCE_AFTER_EXACT_BACKTEST"

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "factory_version": CLASSICAL_FACTORY_VERSION,
                "dna": self.to_dict(include_hash=False),
            },
            length=64,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "hypothesis": self.hypothesis,
            "block_ids": list(self.block_ids),
            "universe": list(self.universe),
            "signal_timeframe": self.route.signal_timeframe,
            "setup_timeframe": self.route.setup_timeframe,
            "regime_timeframe": self.route.regime_timeframe,
            "timeframe_route": self.route.route_id,
            "regime": self.regime_block,
            "setup": self.family,
            "trigger": self.trigger_block,
            "confirmations": list(self.confirmation_blocks),
            "exit": self.exit_block or self.execution_policy.exit_profile,
            "stop": self.stop_method,
            "sizing": self.execution_policy.sizing_policy,
            "execution_policy": asdict(self.execution_policy),
            "cost_profile": self.cost_profile,
            "execution_rule": self.execution_rule,
            "cooldown_bars": self.cooldown_bars,
            "maximum_holding_bars": self.execution_policy.maximum_holding_bars,
            "promotion_eligibility": self.promotion_eligibility,
        }
        if include_hash:
            payload["strategy_dna"] = self.dna_hash
        return payload


def _route_compatible(block_ids: tuple[str, ...], route: TimeframeRoute) -> bool:
    signal_seconds = TIMEFRAME_SECONDS[route.signal_timeframe]
    for block_id in block_ids:
        if block_id.startswith("htf_"):
            source = block_id.split("_", 2)[1]
            if TIMEFRAME_SECONDS[source] <= signal_seconds:
                return False
    return True


def _classify_blocks(
    block_ids: tuple[str, ...],
) -> tuple[str | None, str, tuple[str, ...], str | None]:
    registry = signal_block_registry()
    entries = [
        block_id
        for block_id in block_ids
        if registry[block_id].role is BlockRole.ENTRY_TRIGGER
    ]
    regimes = [
        block_id
        for block_id in block_ids
        if registry[block_id].role is BlockRole.REGIME_FILTER
    ]
    confirmations = [
        block_id
        for block_id in block_ids
        if registry[block_id].role
        in {BlockRole.CONFIRMATION, BlockRole.TREND_FILTER}
    ]
    exits = [
        block_id
        for block_id in block_ids
        if registry[block_id].role is BlockRole.EXIT_TRIGGER
    ]
    if len(entries) != 1:
        raise ValueError(f"classical family requires exactly one trigger: {block_ids}")
    if len(regimes) > 1:
        raise ValueError(f"classical family has multiple regimes: {block_ids}")
    if len(confirmations) > 2:
        raise ValueError(f"classical family has more than two confirmations: {block_ids}")
    if len(exits) > 1:
        raise ValueError(f"classical family has multiple signal exits: {block_ids}")
    groups = [registry[block_id].redundancy_group for block_id in confirmations]
    if len(groups) != len(set(groups)):
        raise ValueError(f"classical family repeats a confirmation information group: {block_ids}")
    return (
        regimes[0] if regimes else None,
        entries[0],
        tuple(confirmations),
        exits[0] if exits else None,
    )


def generate_classical_strategy_dna(
    *,
    trial_count: int = CLASSICAL_FACTORY_DEFAULT_TRIALS,
) -> tuple[ClassicalStrategyDNA, ...]:
    """Generate a fixed prefix of the deterministic family/route/policy grid."""

    if trial_count < 1:
        raise ValueError("classical strategy trial count must be positive")
    rows: list[ClassicalStrategyDNA] = []
    seen: set[str] = set()
    for family, raw_blocks in sorted(CLASSICAL_ECONOMIC_FAMILY_TEMPLATES.items()):
        blocks = tuple(dict.fromkeys(raw_blocks))
        regime, trigger, confirmations, exit_block = _classify_blocks(blocks)
        for route in TIMEFRAME_ROUTES:
            if not _route_compatible(blocks, route):
                continue
            for policy in EXECUTION_POLICIES:
                row = ClassicalStrategyDNA(
                    family=family,
                    hypothesis=(
                        f"{family.replace('_', ' ').title()} may retain positive "
                        "net expectancy in its declared regime after realistic costs."
                    ),
                    block_ids=blocks,
                    universe=CLASSICAL_RESEARCH_UNIVERSE,
                    route=route,
                    execution_policy=policy,
                    regime_block=regime,
                    trigger_block=trigger,
                    confirmation_blocks=confirmations,
                    exit_block=exit_block,
                    stop_method=f"ATR_{policy.stop_atr:.1f}",
                )
                if row.dna_hash in seen:
                    raise RuntimeError("duplicate classical strategy DNA")
                seen.add(row.dna_hash)
                rows.append(row)
                if len(rows) == trial_count:
                    return tuple(rows)
    if len(rows) < trial_count:
        raise ValueError(
            f"classical catalog contains only {len(rows)} valid DNA paths; "
            f"requested {trial_count}"
        )
    return tuple(rows)


def classical_factory_plan(
    *,
    trial_count: int = CLASSICAL_FACTORY_DEFAULT_TRIALS,
) -> dict[str, Any]:
    rows = generate_classical_strategy_dna(trial_count=trial_count)
    hashes = [row.dna_hash for row in rows]
    return {
        "schema_version": "classical_strategy_factory_plan_v1",
        "status": "PREREGISTERED_NOT_FULLY_EVALUATED",
        "campaign": "CLASSICAL_STRATEGY_FACTORY_V1",
        "factory_version": CLASSICAL_FACTORY_VERSION,
        "deterministic_seed": CLASSICAL_FACTORY_SEED,
        "trial_count": len(rows),
        "search_space_hash": stable_hash(hashes, length=64),
        "strategy_dna_hashes": hashes,
        "strategy_dna": [row.to_dict() for row in rows],
        "economic_family_count": len(CLASSICAL_ECONOMIC_FAMILY_TEMPLATES),
        "disabled_data_interface_count": len(CLASSICAL_DISABLED_FAMILY_INTERFACES),
        "disabled_data_interfaces": dict(sorted(CLASSICAL_DISABLED_FAMILY_INTERFACES.items())),
        "timeframe_routes": [asdict(route) for route in TIMEFRAME_ROUTES],
        "execution_policies": [asdict(policy) for policy in EXECUTION_POLICIES],
        "family_trial_counts": dict(sorted(Counter(row.family for row in rows).items())),
        "signal_timeframe_trial_counts": dict(
            sorted(Counter(row.route.signal_timeframe for row in rows).items())
        ),
        "sizing_policy_trial_counts": dict(
            sorted(Counter(row.execution_policy.sizing_policy for row in rows).items())
        ),
        "research_universe": list(CLASSICAL_RESEARCH_UNIVERSE),
        "promotion_universe": list(CLASSICAL_PROMOTION_UNIVERSE),
        "grammar": {
            "maximum_regimes": 1,
            "setups": 1,
            "triggers": 1,
            "maximum_confirmations": 2,
            "exits": 1,
            "stops": 1,
            "sizing_policies": 1,
            "redundant_confirmation_groups_forbidden": True,
        },
        "evaluation_contract": {
            "real_provider_data_only": True,
            "closed_candles_only": True,
            "next_open_execution": True,
            "normal_stressed_and_double_costs": True,
            "walk_forward": True,
            "monte_carlo": True,
            "dirichlet_time_concentration": True,
            "regime_attribution": True,
            "return_path_deduplication": True,
            "academic_tests_are_capital_warnings_for_canary": True,
            "automatic_live_promotion": False,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def classical_family_catalog() -> dict[str, Any]:
    registry = signal_block_registry()
    executable = []
    for family, blocks in sorted(CLASSICAL_ECONOMIC_FAMILY_TEMPLATES.items()):
        regime, trigger, confirmations, exit_block = _classify_blocks(tuple(blocks))
        executable.append(
            {
                "family": family,
                "status": "EXECUTABLE",
                "blocks": list(blocks),
                "regime": regime,
                "trigger": trigger,
                "confirmations": list(confirmations),
                "exit": exit_block or "CANONICAL_ATR_TARGET_TRAIL_TIME_EXIT",
                "source_requirements": sorted(
                    {
                        source
                        for block_id in blocks
                        for source in registry[block_id].source_quality_requirements
                    }
                ),
            }
        )
    return {
        "schema_version": "classical_family_catalog_v1",
        "factory_version": CLASSICAL_FACTORY_VERSION,
        "executable_family_count": len(executable),
        "disabled_family_count": len(CLASSICAL_DISABLED_FAMILY_INTERFACES),
        "executable_families": executable,
        "disabled_families": [
            {"family": family, "status": "DATA_PENDING", "reason_code": reason}
            for family, reason in sorted(CLASSICAL_DISABLED_FAMILY_INTERFACES.items())
        ],
        "synthetic_orderflow_used": False,
        "synthetic_derivatives_used": False,
    }


__all__ = [
    "CLASSICAL_FACTORY_DEFAULT_TRIALS",
    "CLASSICAL_FACTORY_SEED",
    "CLASSICAL_FACTORY_VERSION",
    "CLASSICAL_PROMOTION_UNIVERSE",
    "CLASSICAL_RESEARCH_UNIVERSE",
    "EXECUTION_POLICIES",
    "TIMEFRAME_ROUTES",
    "ClassicalStrategyDNA",
    "ExecutionPolicy",
    "TimeframeRoute",
    "classical_factory_plan",
    "classical_family_catalog",
    "generate_classical_strategy_dna",
]
