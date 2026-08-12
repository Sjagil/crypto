"""P1.1 economically grounded, zero-authority crypto alpha discovery.

This module owns hypothesis preregistration, economic edge diagnostics and a
bounded cross-sectional Stage-0 screen. It does not own exchange connectivity,
portfolio allocation, paper promotion or live authority.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from research.backtest import BacktestConfig, BacktestEngine, CostModel
from research.features import FeaturePipeline
from research.optimization import chronological_split
from research.research_factory import (
    SharedCostModel,
    derive_dataset_identity,
    load_immutable_ohlcv,
    parameter_plateaus,
)
from research.strategies import Strategy, StrategyOutput
from utils.common import atomic_write_json, stable_hash, utc_iso

ALPHA_DISCOVERY_SCHEMA_VERSION = "economically_grounded_alpha_discovery_v1"
HYPOTHESIS_CARD_SCHEMA_VERSION = "economic_hypothesis_card_v1"
PANEL_STAGE0_SCHEMA_VERSION = (
    "cross_sectional_panel_stage0_v3_fixed_exposure_deferred_missing_bar_exit"
)
ECONOMIC_EDGE_POLICY_VERSION = "expected_move_to_roundtrip_cost_v1"
STAGE0_MAXIMUM_RESEARCH_EXPOSURE = 0.20


class CandidateOrigin(StrEnum):
    STRUCTURAL_VARIANT = "STRUCTURAL_VARIANT"
    NEW_HYPOTHESIS = "NEW_HYPOTHESIS"
    EXISTING_CAUSAL_ENGINE = "EXISTING_CAUSAL_ENGINE"


class FamilyClassification(StrEnum):
    UNTESTED = "UNTESTED"
    GROSS_NEGATIVE = "GROSS_NEGATIVE"
    GROSS_POSITIVE_NET_NEGATIVE = "GROSS_POSITIVE_NET_NEGATIVE"
    STAGE0_PROMISING = "STAGE0_PROMISING"
    EXACT_REJECTED = "EXACT_REJECTED"
    EXACT_POSITIVE_NOT_ROBUST = "EXACT_POSITIVE_NOT_ROBUST"
    FORWARD_CANDIDATE = "FORWARD_CANDIDATE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True, slots=True)
class HypothesisCard:
    hypothesis_id: str
    family: str
    economic_mechanism: str
    why_it_might_exist: str
    expected_holding_period: str
    expected_turnover: str
    expected_market_regime: tuple[str, ...]
    expected_failure_regime: tuple[str, ...]
    required_data: tuple[str, ...]
    target_assets: tuple[str, ...]
    target_timeframes: tuple[str, ...]
    parameters_to_test: Mapping[str, tuple[Any, ...]]
    falsification_criteria: tuple[str, ...]
    known_risks: tuple[str, ...]
    reference_concepts: tuple[str, ...]
    candidate_origin: CandidateOrigin
    primary_entry_timeframe: str
    context_timeframes: tuple[str, ...]
    information_only_inputs: tuple[str, ...] = ()
    stage0_authority: str = "APPROXIMATE_RESEARCH_ONLY"

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.family:
            raise ValueError("hypothesis ID and family are required")
        if len(self.economic_mechanism.strip()) < 40:
            raise ValueError("hypothesis requires a substantive economic mechanism")
        if len(self.why_it_might_exist.strip()) < 30:
            raise ValueError("hypothesis requires a substantive persistence rationale")
        if not self.expected_market_regime or not self.expected_failure_regime:
            raise ValueError("expected success and failure regimes are required")
        if not self.required_data or not self.target_assets or not self.target_timeframes:
            raise ValueError("data, assets and timeframes are required")
        if self.primary_entry_timeframe == "15m":
            raise ValueError("15m cannot be the primary P1.1 alpha timeframe")
        if self.primary_entry_timeframe not in self.target_timeframes:
            raise ValueError("primary timeframe must be included in target timeframes")
        if not self.parameters_to_test:
            raise ValueError("bounded parameter regions are required")
        combinations = 1
        for name, values in self.parameters_to_test.items():
            if not name or not values:
                raise ValueError("every parameter requires a named bounded region")
            combinations *= len(values)
        if combinations > 256:
            raise ValueError("one hypothesis card cannot exceed 256 parameter regions")
        if not self.falsification_criteria or not self.known_risks:
            raise ValueError("falsification criteria and known risks are required")
        if not self.reference_concepts:
            raise ValueError("reference concepts are required")
        if self.stage0_authority != "APPROXIMATE_RESEARCH_ONLY":
            raise ValueError("hypothesis cards cannot grant execution authority")
        if self.information_only_inputs and not all(
            value.startswith("INFORMATION_ONLY:") for value in self.information_only_inputs
        ):
            raise ValueError("context-only data must be explicitly INFORMATION_ONLY")

    @property
    def parameter_region_count(self) -> int:
        return math.prod(len(values) for values in self.parameters_to_test.values())

    @property
    def card_hash(self) -> str:
        return stable_hash(
            {
                "schema": HYPOTHESIS_CARD_SCHEMA_VERSION,
                "card": asdict(self),
            },
            length=48,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_CARD_SCHEMA_VERSION,
            **asdict(self),
            "parameter_region_count": self.parameter_region_count,
            "card_hash": self.card_hash,
        }


class HypothesisRegistry:
    """Immutable-ID registry rejecting duplicate mechanisms and cards."""

    def __init__(self, cards: Sequence[HypothesisCard]) -> None:
        ids = [card.hypothesis_id for card in cards]
        hashes = [card.card_hash for card in cards]
        mechanisms = [
            stable_hash(
                {
                    "family": card.family,
                    "mechanism": " ".join(card.economic_mechanism.lower().split()),
                },
                length=40,
            )
            for card in cards
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate hypothesis ID")
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate hypothesis card")
        if len(set(mechanisms)) != len(mechanisms):
            raise ValueError("duplicate family economic mechanism")
        self._cards = tuple(cards)

    @property
    def cards(self) -> tuple[HypothesisCard, ...]:
        return self._cards

    @property
    def registry_hash(self) -> str:
        return stable_hash([card.card_hash for card in self._cards], length=48)


@dataclass(frozen=True, slots=True)
class EconomicEdgePolicy:
    version: str
    roundtrip_cost_bps: float
    minimum_move_cost_ratio: float
    calibration: Mapping[str, Any]

    @classmethod
    def from_costs(
        cls,
        costs: SharedCostModel,
        *,
        holding_period_hours: float,
        liquidity_tier: str,
    ) -> "EconomicEdgePolicy":
        if holding_period_hours <= 0:
            raise ValueError("holding period must be positive")
        if liquidity_tier not in {"TIER_1", "TIER_2", "TIER_3"}:
            raise ValueError("unknown liquidity tier")
        one_way_bps = (
            costs.taker_fee_fraction * 10_000.0
            + costs.spread_bps / 2.0
            + costs.slippage_bps
            + costs.failed_execution_allowance_bps
            + costs.partial_fill_impact_bps
        )
        roundtrip = 2.0 * one_way_bps
        liquidity_margin = {"TIER_1": 0.5, "TIER_2": 1.0, "TIER_3": 2.0}[
            liquidity_tier
        ]
        horizon_margin = 1.0 if holding_period_hours >= 24 else 1.5
        minimum_ratio = 1.0 + liquidity_margin + horizon_margin
        calibration = {
            "cost_model_version": costs.cost_model_version,
            "holding_period_hours": holding_period_hours,
            "liquidity_tier": liquidity_tier,
            "one_way_cost_bps": one_way_bps,
            "margin_components": {
                "cost_recovery": 1.0,
                "liquidity_uncertainty": liquidity_margin,
                "holding_horizon_uncertainty": horizon_margin,
            },
        }
        return cls(
            version=f"{ECONOMIC_EDGE_POLICY_VERSION}:{stable_hash(calibration, length=20)}",
            roundtrip_cost_bps=roundtrip,
            minimum_move_cost_ratio=minimum_ratio,
            calibration=calibration,
        )

    def assess(self, expected_favorable_move_bps: float | None) -> dict[str, Any]:
        ratio = (
            float(expected_favorable_move_bps) / self.roundtrip_cost_bps
            if expected_favorable_move_bps is not None and self.roundtrip_cost_bps > 0
            else None
        )
        return {
            "policy_version": self.version,
            "expected_favorable_move_bps": expected_favorable_move_bps,
            "roundtrip_cost_bps": self.roundtrip_cost_bps,
            "expected_move_to_roundtrip_cost_ratio": ratio,
            "minimum_required_ratio": self.minimum_move_cost_ratio,
            "economically_large_enough": (
                ratio is not None and ratio >= self.minimum_move_cost_ratio
            ),
        }


@dataclass(frozen=True, slots=True)
class LiquidityProfile:
    market: str
    median_quote_turnover: float
    history_rows: int
    first_available: str
    last_available: str
    tier: str
    point_in_time_status: str = "PIT_FROM_FIRST_AVAILABLE_CANDLE_PARTIAL"


def liquidity_profiles(frames: Mapping[str, pd.DataFrame]) -> tuple[LiquidityProfile, ...]:
    observations: list[tuple[str, float]] = []
    for market, frame in frames.items():
        turnover = (frame["close"].astype(float) * frame["volume"].astype(float)).replace(
            [np.inf, -np.inf], np.nan
        )
        observations.append((market, float(turnover.median())))
    values = np.asarray([value for _, value in observations], dtype=float)
    upper = float(np.quantile(values, 0.60))
    lower = float(np.quantile(values, 0.25))
    output = []
    for market, value in observations:
        tier = "TIER_1" if value >= upper else ("TIER_2" if value >= lower else "TIER_3")
        frame = frames[market]
        output.append(
            LiquidityProfile(
                market=market,
                median_quote_turnover=value,
                history_rows=len(frame),
                first_available=pd.Timestamp(frame.index[0]).isoformat(),
                last_available=pd.Timestamp(frame.index[-1]).isoformat(),
                tier=tier,
            )
        )
    return tuple(sorted(output, key=lambda row: (-row.median_quote_turnover, row.market)))


def diagnose_p1_alpha_failure(p1: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = list((p1.get("stage0") or {}).get("aggregate_parameter_results") or [])
    if not aggregate:
        raise ValueError("P1 artifact has no Stage-0 aggregate evidence")
    best = aggregate[0]
    trades = int(best.get("trade_count") or 0)
    gross = float(best.get("gross_pnl_eur") or 0.0)
    net = float(best.get("net_pnl_eur") or 0.0)
    gross_per_trade = gross / trades if trades else None
    net_per_trade = net / trades if trades else None
    cost_per_trade = (gross - net) / trades if trades else None
    ratio = (
        cost_per_trade / gross_per_trade
        if cost_per_trade is not None and gross_per_trade is not None and gross_per_trade > 0
        else None
    )
    exact_review = ((p1.get("benchmark") or {}).get("false_negative_review_sample") or {})
    exact_metrics = ((exact_review.get("exact_result") or {}).get("metrics") or {})
    reasons = ["EDGE_TOO_SMALL_FOR_COSTS", "EXCESSIVE_TURNOVER"]
    if int(best.get("positive_asset_count") or 0) == 0:
        reasons.append("ASSET_GENERALIZATION_FAILURE")
    if float(best.get("profit_factor") or 0.0) < 1.0:
        reasons.append("POOR_ENTRY_OR_EXIT_CAPTURE")
    return {
        "classification": reasons,
        "best_parameter_hash": best.get("parameter_hash"),
        "trade_count": trades,
        "gross_pnl_eur": gross,
        "net_pnl_eur": net,
        "gross_edge_per_trade_eur": gross_per_trade,
        "net_edge_per_trade_eur": net_per_trade,
        "estimated_roundtrip_cost_per_trade_eur": cost_per_trade,
        "cost_to_positive_gross_edge_ratio": ratio,
        "turnover_eur": best.get("turnover_eur"),
        "positive_asset_count": best.get("positive_asset_count"),
        "positive_timeframe_count": best.get("positive_timeframe_count"),
        "exact_rejection_review": {
            "trade_count": exact_metrics.get("trade_count"),
            "net_expectancy_eur": exact_metrics.get("net_expectancy_eur"),
            "profit_factor": exact_metrics.get("profit_factor"),
            "maximum_drawdown": exact_metrics.get("maximum_drawdown"),
            "average_mfe_r": exact_metrics.get("average_mfe_r"),
            "average_mae_r": exact_metrics.get("average_mae_r"),
        },
        "average_holding_period": "NOT_REPORTED_BY_P1_AGGREGATE",
        "false_breakout_frequency": "NOT_EVALUABLE_FROM_CANONICAL_P1_FIELDS",
        "constraint_for_p1_1": (
            "PRIORITIZE_4H_1D_LOW_TURNOVER_MOVES_WITH_EXPLICIT_MOVE_COST_MARGIN"
        ),
    }


DEFAULT_ALPHA_ASSETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "ADA-EUR",
    "XRP-EUR",
    "LINK-EUR",
    "DOGE-EUR",
    "LTC-EUR",
)


def initial_hypothesis_cards() -> tuple[HypothesisCard, ...]:
    """Return eight preregistered mechanisms; only supported cards are run."""

    common_falsification = (
        "broad reasonable parameter region is gross negative",
        "net expectancy is non-positive under baseline Bitvavo costs",
        "expected favorable move lacks sufficient roundtrip-cost margin",
        "positive evidence is an isolated parameter optimum",
    )
    common_risks = (
        "crypto market beta can masquerade as alpha",
        "point-in-time universe reconstruction is partial",
        "cross-sectional assets are correlated rather than independent",
    )
    assets = DEFAULT_ALPHA_ASSETS
    return (
        HypothesisCard(
            hypothesis_id="p1.1_cross_sectional_momentum_v1",
            family="CROSS_SECTIONAL_MOMENTUM",
            economic_mechanism=(
                "Capital and attention rotate unevenly across liquid crypto assets, so persistent "
                "risk-adjusted leadership may continue after ranking while laggards are avoided."
            ),
            why_it_might_exist=(
                "Fragmented investor attention, reflexive flows and gradual information diffusion "
                "can make relative rather than standalone momentum persistent."
            ),
            expected_holding_period="3 to 14 days",
            expected_turnover="weekly or twice-weekly top-N rotation",
            expected_market_regime=("broad risk-on", "leadership dispersion"),
            expected_failure_regime=("violent reversals", "high-correlation deleveraging"),
            required_data=("causal OHLCV", "PIT eligible universe", "BTC benchmark"),
            target_assets=assets,
            target_timeframes=("4h", "1d"),
            parameters_to_test={
                "momentum_days": (7, 14, 28),
                "top_n": (1, 3),
                "holding_days": (3, 7),
                "minimum_rank_persistence": (1, 2),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks,
            reference_concepts=("cross-sectional ranking", "volatility-adjusted momentum"),
            candidate_origin=CandidateOrigin.EXISTING_CAUSAL_ENGINE,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_medium_term_trend_pullback_v1",
            family="MEDIUM_TERM_TREND_PULLBACK",
            economic_mechanism=(
                "Persistent multi-day crypto trends can survive temporary profit-taking; entering "
                "a controlled pullback seeks a larger continuation move with improved location."
            ),
            why_it_might_exist=(
                "Slow institutional rebalancing and reflexive trend participation can extend moves "
                "after shallow retracements without requiring high-frequency prediction."
            ),
            expected_holding_period="2 to 10 days",
            expected_turnover="at most a few entries per asset per month",
            expected_market_regime=("persistent directional trend", "moderate volatility"),
            expected_failure_regime=("sideways chop", "gap-like trend reversal"),
            required_data=("causal OHLCV", "4h trend structure", "1d context"),
            target_assets=assets,
            target_timeframes=("1h", "4h"),
            parameters_to_test={
                "trend_days": (20, 40),
                "pullback_atr": (0.5, 1.0),
                "continuation_days": (3, 7),
                "exit_days": (5, 10),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("pullback depth can signal trend failure",),
            reference_concepts=("time-series momentum", "pullback continuation"),
            candidate_origin=CandidateOrigin.NEW_HYPOTHESIS,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_volatility_contraction_expansion_v1",
            family="VOLATILITY_CONTRACTION_EXPANSION",
            economic_mechanism=(
                "Volatility clusters, and an unusually compressed range can precede repricing; a "
                "directional expansion after compression targets a move larger than its friction."
            ),
            why_it_might_exist=(
                "Dormant positioning and liquidity can accumulate during compression before a new "
                "information shock or flow imbalance produces sustained expansion."
            ),
            expected_holding_period="2 to 14 days",
            expected_turnover="only after uncommon compression and expansion events",
            expected_market_regime=("low-volatility compression ending", "directional repricing"),
            expected_failure_regime=("false expansion", "systemic reversal"),
            required_data=("causal OHLCV", "prior volatility distribution", "volume"),
            target_assets=assets,
            target_timeframes=("4h", "1d"),
            parameters_to_test={
                "compression_days": (5, 10),
                "baseline_days": (60, 120),
                "compression_quantile": (0.20, 0.30),
                "breakout_days": (10, 20),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("compression definitions can be data-mined",),
            reference_concepts=("volatility clustering", "contraction then expansion"),
            candidate_origin=CandidateOrigin.EXISTING_CAUSAL_ENGINE,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_quality_consolidation_breakout_v1",
            family="QUALITY_CONSOLIDATION_BREAKOUT",
            economic_mechanism=(
                "A multi-day consolidation with declining range, persistent relative leadership "
                "and renewed volume may represent absorbed supply before a durable repricing."
            ),
            why_it_might_exist=(
                "Patient accumulation can cap volatility while supply is absorbed, after which new "
                "demand must travel farther to find liquidity than in an unconstrained breakout."
            ),
            expected_holding_period="3 to 14 days",
            expected_turnover="rare qualified consolidations only",
            expected_market_regime=("constructive breadth", "leadership persistence"),
            expected_failure_regime=("broad risk-off", "volume-less breakout"),
            required_data=("causal OHLCV", "consolidation quality", "relative rank", "volume"),
            target_assets=assets,
            target_timeframes=("4h",),
            parameters_to_test={
                "consolidation_days": (5, 10),
                "maximum_range_atr": (2.0, 3.0),
                "relative_rank_minimum": (0.60, 0.80),
                "volume_multiple": (1.0, 1.5),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("breakout gap can erase entry economics",),
            reference_concepts=("supply absorption", "relative-strength confirmed breakout"),
            candidate_origin=CandidateOrigin.STRUCTURAL_VARIANT,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_btc_relative_alt_rotation_v1",
            family="BTC_RELATIVE_ALT_ROTATION",
            economic_mechanism=(
                "Altcoins with persistent positive residual performance versus BTC may be receiving "
                "asset-specific flows that continue while systemic BTC conditions remain tolerable."
            ),
            why_it_might_exist=(
                "Sector narratives and token-specific adoption diffuse gradually, producing relative "
                "leadership distinct from common BTC beta."
            ),
            expected_holding_period="3 to 21 days",
            expected_turnover="weekly top-relative-strength rotation",
            expected_market_regime=("BTC stable or rising", "cross-sectional dispersion"),
            expected_failure_regime=("BTC crash", "altcoin correlation spike"),
            required_data=("causal OHLCV", "BTC-relative returns", "liquidity"),
            target_assets=tuple(asset for asset in assets if asset != "BTC-EUR"),
            target_timeframes=("4h", "1d"),
            parameters_to_test={
                "relative_days": (7, 14, 28),
                "top_n": (1, 3),
                "btc_regime": ("SOFT", "HARD"),
                "holding_days": (3, 7),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("residual leadership can be unstable",),
            reference_concepts=("BTC residual momentum", "long-only relative rotation"),
            candidate_origin=CandidateOrigin.EXISTING_CAUSAL_ENGINE,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_breadth_conditioned_momentum_v1",
            family="BREADTH_CONDITIONED_MOMENTUM",
            economic_mechanism=(
                "Momentum is more durable when participation is broad; causal market breadth may "
                "separate healthy trend continuation from narrow, fragile leadership."
            ),
            why_it_might_exist=(
                "Broad participation indicates distributed demand and lower dependence on a single "
                "asset, while narrow rallies are more exposed to common reversals."
            ),
            expected_holding_period="3 to 14 days",
            expected_turnover="weekly, with soft exposure reduction in narrow breadth",
            expected_market_regime=("broad participation", "positive dispersion"),
            expected_failure_regime=("breadth whipsaw", "sudden systemic shock"),
            required_data=("PIT universe OHLCV", "causal breadth", "momentum ranks"),
            target_assets=assets,
            target_timeframes=("4h", "1d"),
            parameters_to_test={
                "momentum_days": (14, 28),
                "breadth_days": (20, 40),
                "breadth_floor": (0.40, 0.60),
                "breadth_mode": ("SOFT", "HARD"),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("breadth adds no incremental value over beta",),
            reference_concepts=("advance participation", "regime-conditioned momentum"),
            candidate_origin=CandidateOrigin.NEW_HYPOTHESIS,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_slow_volume_accumulation_v1",
            family="SLOW_VOLUME_ACCUMULATION",
            economic_mechanism=(
                "Persistent price resilience on expanding positive volume may reveal gradual demand "
                "accumulation before price fully reflects the flow."
            ),
            why_it_might_exist=(
                "Large participants split orders through time, so volume-price confirmation may lead "
                "slower multi-day continuation without orderflow churn."
            ),
            expected_holding_period="3 to 21 days",
            expected_turnover="low-frequency multi-bar accumulation episodes",
            expected_market_regime=("orderly accumulation", "adequate liquidity"),
            expected_failure_regime=("distribution disguised as volume", "risk-off shock"),
            required_data=("causal OHLCV", "relative volume", "price resilience"),
            target_assets=assets,
            target_timeframes=("4h", "1d"),
            parameters_to_test={
                "accumulation_days": (5, 10),
                "relative_volume": (1.0, 1.5),
                "resilience_days": (3, 7),
            },
            falsification_criteria=common_falsification,
            known_risks=common_risks + ("reported volume may change across venues",),
            reference_concepts=("volume-price confirmation", "multi-bar accumulation"),
            candidate_origin=CandidateOrigin.NEW_HYPOTHESIS,
            primary_entry_timeframe="4h",
            context_timeframes=("1d",),
        ),
        HypothesisCard(
            hypothesis_id="p1.1_derivatives_context_modifier_v1",
            family="DERIVATIVES_CONTEXT_MODIFIER",
            economic_mechanism=(
                "Extreme leverage positioning can alter spot trend continuation and exhaustion risk, "
                "so derivatives state may improve timing without becoming a traded instrument."
            ),
            why_it_might_exist=(
                "Funding, open interest and liquidations reveal leveraged crowding whose unwind can "
                "affect spot prices through arbitrage and forced flow."
            ),
            expected_holding_period="1 to 10 days",
            expected_turnover="modifier only; cannot originate execution",
            expected_market_regime=("measurable leverage crowding",),
            expected_failure_regime=("missing PIT history", "venue-specific distortions"),
            required_data=("PIT funding", "PIT open interest", "spot hypothesis"),
            target_assets=("BTC-EUR", "ETH-EUR"),
            target_timeframes=("1h", "4h"),
            parameters_to_test={"context_window": (24, 72), "mode": ("SOFT",)},
            falsification_criteria=(
                "point-in-time derivatives history is incomplete",
                "modifier does not add net economics to a frozen spot baseline",
            ),
            known_risks=("venue mismatch", "timestamp leakage", "crowding can persist"),
            reference_concepts=("leverage crowding", "information-only context"),
            candidate_origin=CandidateOrigin.NEW_HYPOTHESIS,
            primary_entry_timeframe="4h",
            context_timeframes=("1h",),
            information_only_inputs=(
                "INFORMATION_ONLY:funding",
                "INFORMATION_ONLY:open_interest",
                "INFORMATION_ONLY:liquidations",
            ),
        ),
    )


def rank_hypotheses(
    cards: Sequence[HypothesisCard],
    *,
    supported_families: Sequence[str],
) -> list[dict[str, Any]]:
    supported = set(supported_families)
    priority = {
        "CROSS_SECTIONAL_MOMENTUM": 100,
        "MEDIUM_TERM_TREND_PULLBACK": 95,
        "VOLATILITY_CONTRACTION_EXPANSION": 90,
        "QUALITY_CONSOLIDATION_BREAKOUT": 85,
        "BTC_RELATIVE_ALT_ROTATION": 80,
        "BREADTH_CONDITIONED_MOMENTUM": 75,
        "SLOW_VOLUME_ACCUMULATION": 65,
        "DERIVATIVES_CONTEXT_MODIFIER": 40,
    }
    rows = []
    for card in cards:
        executable = card.family in supported
        rows.append(
            {
                "hypothesis_id": card.hypothesis_id,
                "family": card.family,
                "priority_score": priority.get(card.family, 0),
                "parameter_region_count": card.parameter_region_count,
                "data_support": "READY" if executable else "NOT_EVALUABLE",
                "campaign_state": "BOUNDED_STAGE0" if executable else "DEFERRED_DATA_OR_ENGINE",
                "reason": (
                    "ECONOMICALLY_DISTINCT_AND_CAUSALLY_SUPPORTED"
                    if executable
                    else "NO_COMPLETE_CAUSAL_STAGE0_INPUT_OR_ENGINE"
                ),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["priority_score"]), row["family"]))


@dataclass(frozen=True)
class PanelData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    eligible: pd.DataFrame
    timeframe: str
    bars_per_day: int
    data_hash: str

    @property
    def markets(self) -> tuple[str, ...]:
        return tuple(self.close.columns)


def build_point_in_time_panel(
    frames: Mapping[str, pd.DataFrame],
    *,
    timeframe: str,
    minimum_history_bars: int = 120,
) -> PanelData:
    if timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("P1.1 panel supports 1h, 4h and 1d")
    if minimum_history_bars < 20:
        raise ValueError("minimum PIT history must be at least 20 bars")
    markets = tuple(sorted(frames))
    if "BTC-EUR" not in markets or len(markets) < 3:
        raise ValueError("cross-sectional panel requires BTC and at least three markets")
    union = pd.DatetimeIndex(
        sorted(set().union(*(set(frame.index) for frame in frames.values())))
    )
    if union.tz is None:
        union = union.tz_localize("UTC")
    else:
        union = union.tz_convert("UTC")
    fields: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        fields[field] = pd.concat(
            {
                market: frames[market][field].astype(float).reindex(union)
                for market in markets
            },
            axis=1,
        )
        fields[field].columns = markets
    present = fields["close"].notna() & fields["open"].notna()
    history = present.astype(int).cumsum()
    eligible = present & (history >= minimum_history_bars)
    hashes = {
        name: stable_hash(
            pd.util.hash_pandas_object(value, index=True).astype(str).tolist(), length=32
        )
        for name, value in fields.items()
    }
    return PanelData(
        **fields,
        eligible=eligible,
        timeframe=timeframe,
        bars_per_day={"1h": 24, "4h": 6, "1d": 1}[timeframe],
        data_hash=stable_hash(
            {
                "timeframe": timeframe,
                "markets": markets,
                "fields": hashes,
                "minimum_history_bars": minimum_history_bars,
            },
            length=48,
        ),
    )


def parameter_grid(card: HypothesisCard) -> tuple[dict[str, Any], ...]:
    names = tuple(sorted(card.parameters_to_test))
    return tuple(
        dict(zip(names, values, strict=True))
        for values in product(*(card.parameters_to_test[name] for name in names))
    )


def _atr_panel(panel: PanelData, period: int = 14) -> pd.DataFrame:
    previous = panel.close.shift(1)
    return pd.DataFrame(
        np.maximum.reduce(
            [
                (panel.high - panel.low).to_numpy(),
                (panel.high - previous).abs().to_numpy(),
                (panel.low - previous).abs().to_numpy(),
            ]
        ),
        index=panel.close.index,
        columns=panel.close.columns,
    ).rolling(period, min_periods=period).mean()


def _top_weights(
    score: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_n: int,
    exposure_scale: pd.Series | None = None,
) -> pd.DataFrame:
    valid = score.where(eligible & score.gt(0.0))
    selected = valid.rank(axis=1, ascending=False, method="first") <= top_n
    counts = selected.sum(axis=1).replace(0, np.nan)
    weights = (
        selected.astype(float).div(counts, axis=0).fillna(0.0)
        * STAGE0_MAXIMUM_RESEARCH_EXPOSURE
    )
    if exposure_scale is not None:
        scale = exposure_scale.astype(float).clip(0.0, 1.0).fillna(0.0)
        weights = weights.mul(scale, axis=0)
    return weights


def _scheduled_weights(
    desired: pd.DataFrame,
    *,
    every_bars: int,
) -> pd.DataFrame:
    if every_bars < 1:
        raise ValueError("rebalance interval must be positive")
    scheduled = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    scheduled.iloc[::every_bars] = desired.iloc[::every_bars]
    return scheduled.ffill().fillna(0.0)


def family_desired_weights(
    panel: PanelData,
    *,
    family: str,
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    bars = panel.bars_per_day
    close = panel.close
    returns = close.pct_change(fill_method=None)
    atr = _atr_panel(panel)
    eligible = panel.eligible.copy()
    top_n = int(parameters.get("top_n") or 2)
    holding_days = int(parameters.get("holding_days") or 7)

    if family == "CROSS_SECTIONAL_MOMENTUM":
        lookback = int(parameters["momentum_days"]) * bars
        raw = close.pct_change(lookback, fill_method=None)
        volatility = returns.rolling(lookback, min_periods=lookback).std()
        score = raw / volatility.replace(0.0, np.nan)
        ranks = score.rank(axis=1, pct=True)
        persistence = int(parameters["minimum_rank_persistence"])
        persistent = (
            ranks.rolling(persistence, min_periods=persistence).min()
            >= 1.0 - top_n / max(1, len(close.columns))
        )
        desired = _top_weights(score, eligible & persistent, top_n=top_n)
    elif family == "MEDIUM_TERM_TREND_PULLBACK":
        trend = int(parameters["trend_days"]) * bars
        continuation = int(parameters["continuation_days"]) * bars
        exit_days = int(parameters["exit_days"]) * bars
        ema = close.ewm(span=trend, adjust=False, min_periods=trend).mean()
        recent_high = close.rolling(continuation, min_periods=continuation).max().shift(1)
        pullback = (recent_high - close) / atr.replace(0.0, np.nan)
        trend_strength = close / ema - 1.0
        continuing = close > close.shift(max(1, exit_days // 2))
        setup = (
            (close > ema)
            & (trend_strength > 0)
            & (pullback >= float(parameters["pullback_atr"]))
            & (pullback <= float(parameters["pullback_atr"]) + 2.0)
            & continuing
        )
        desired = _top_weights(trend_strength, eligible & setup, top_n=2)
    elif family == "VOLATILITY_CONTRACTION_EXPANSION":
        compression = int(parameters["compression_days"]) * bars
        baseline = int(parameters["baseline_days"]) * bars
        breakout = int(parameters["breakout_days"]) * bars
        realized = returns.rolling(compression, min_periods=compression).std()
        threshold = realized.shift(1).rolling(baseline, min_periods=baseline).quantile(
            float(parameters["compression_quantile"])
        )
        was_compressed = (realized <= threshold).shift(1).rolling(
            compression, min_periods=1
        ).max().astype(bool)
        prior_high = panel.high.rolling(breakout, min_periods=breakout).max().shift(1)
        expansion = close / prior_high - 1.0
        setup = was_compressed & (close > prior_high) & (returns > 0)
        desired = _top_weights(expansion, eligible & setup, top_n=2)
    elif family == "QUALITY_CONSOLIDATION_BREAKOUT":
        consolidation = int(parameters["consolidation_days"]) * bars
        prior_high = panel.high.rolling(consolidation, min_periods=consolidation).max().shift(1)
        prior_low = panel.low.rolling(consolidation, min_periods=consolidation).min().shift(1)
        range_atr = (prior_high - prior_low) / atr.shift(1).replace(0.0, np.nan)
        relative = close.pct_change(20 * bars, fill_method=None).rank(axis=1, pct=True)
        prior_volume = panel.volume.rolling(consolidation, min_periods=consolidation).mean().shift(1)
        setup = (
            (range_atr <= float(parameters["maximum_range_atr"]))
            & (close > prior_high)
            & (relative >= float(parameters["relative_rank_minimum"]))
            & (panel.volume >= prior_volume * float(parameters["volume_multiple"]))
        )
        score = (close / prior_high - 1.0) + relative
        desired = _top_weights(score, eligible & setup, top_n=2)
    elif family == "BTC_RELATIVE_ALT_ROTATION":
        lookback = int(parameters["relative_days"]) * bars
        absolute = close.pct_change(lookback, fill_method=None)
        btc = absolute["BTC-EUR"]
        score = absolute.sub(btc, axis=0)
        score["BTC-EUR"] = np.nan
        btc_ema = close["BTC-EUR"].ewm(
            span=40 * bars, adjust=False, min_periods=40 * bars
        ).mean()
        btc_score = (close["BTC-EUR"] / btc_ema - 1.0).clip(-0.20, 0.20)
        if str(parameters["btc_regime"]) == "HARD":
            eligible = eligible & (btc_score > 0).to_numpy()[:, None]
            scale = None
        else:
            scale = (0.5 + 2.5 * btc_score).clip(0.0, 1.0)
        desired = _top_weights(score, eligible, top_n=top_n, exposure_scale=scale)
    elif family == "BREADTH_CONDITIONED_MOMENTUM":
        lookback = int(parameters["momentum_days"]) * bars
        breadth_window = int(parameters["breadth_days"]) * bars
        raw = close.pct_change(lookback, fill_method=None)
        ema = close.ewm(
            span=breadth_window, adjust=False, min_periods=breadth_window
        ).mean()
        breadth = ((close > ema) & eligible).sum(axis=1) / eligible.sum(axis=1).replace(
            0, np.nan
        )
        floor = float(parameters["breadth_floor"])
        if str(parameters["breadth_mode"]) == "HARD":
            family_eligible = eligible & (breadth >= floor).fillna(False).to_numpy()[:, None]
            scale = None
        else:
            family_eligible = eligible
            scale = (breadth / max(floor, 1e-9)).clip(0.0, 1.0)
        desired = _top_weights(raw, family_eligible, top_n=3, exposure_scale=scale)
    elif family == "SLOW_VOLUME_ACCUMULATION":
        window = int(parameters["accumulation_days"]) * bars
        resilience = int(parameters["resilience_days"]) * bars
        signed_volume = panel.volume * np.sign(returns).fillna(0.0)
        accumulation = signed_volume.rolling(window, min_periods=window).sum() / panel.volume.rolling(
            window, min_periods=window
        ).sum().replace(0.0, np.nan)
        relative_volume = panel.volume / panel.volume.rolling(
            window, min_periods=window
        ).mean().shift(1)
        resilient = close.pct_change(resilience, fill_method=None) > returns.mean(axis=1).rolling(
            resilience, min_periods=resilience
        ).sum().to_numpy()[:, None]
        setup = (
            (accumulation > 0)
            & (relative_volume >= float(parameters["relative_volume"]))
            & resilient
        )
        desired = _top_weights(accumulation, eligible & setup, top_n=2)
    else:
        raise ValueError(f"unsupported Stage-0 family: {family}")
    return _scheduled_weights(desired, every_bars=max(1, holding_days * bars))


def panel_causality_check(
    panel: PanelData,
    *,
    family: str,
    parameters: Mapping[str, Any],
    cutoff_fraction: float = 0.80,
) -> dict[str, Any]:
    cutoff = int(len(panel.close) * cutoff_fraction)
    baseline = family_desired_weights(panel, family=family, parameters=parameters)
    shocked_fields = {}
    for name in ("open", "high", "low", "close", "volume"):
        value = getattr(panel, name).copy()
        value.iloc[cutoff:] = value.iloc[cutoff:] * 3.0 + 17.0
        shocked_fields[name] = value
    shocked = PanelData(
        **shocked_fields,
        eligible=panel.eligible,
        timeframe=panel.timeframe,
        bars_per_day=panel.bars_per_day,
        data_hash="CAUSALITY_SHOCK",
    )
    candidate = family_desired_weights(shocked, family=family, parameters=parameters)
    safe = np.allclose(
        baseline.iloc[:cutoff].to_numpy(),
        candidate.iloc[:cutoff].to_numpy(),
        equal_nan=True,
    )
    return {
        "status": "PASSED" if safe else "HARD_REJECT",
        "prior_weights_unchanged": bool(safe),
        "future_rows_modified": len(panel.close) - cutoff,
    }


@dataclass(frozen=True)
class PanelStage0Result:
    result_id: str
    hypothesis_id: str
    family: str
    parameter_set: Mapping[str, Any]
    parameter_hash: str
    data_hash: str
    timeframe: str
    market_count: int
    signal_count: int
    trade_count: int
    gross_pnl_eur: float
    net_pnl_eur: float
    gross_expectancy_eur: float | None
    net_expectancy_eur: float | None
    profit_factor: float | None
    maximum_drawdown: float
    turnover: float
    annualized_turnover: float
    trades_per_week: float
    average_holding_bars: float | None
    median_mfe_bps: float | None
    median_mae_bps: float | None
    median_holding_return_bps: float | None
    favorable_move_cost_coverage: float | None
    expected_move_cost_ratio: float | None
    cost_as_fraction_of_gross_opportunity: float | None
    classification: FamilyClassification
    rejection_reasons: tuple[str, ...]
    authority: str = "APPROXIMATE_RESEARCH_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position_episodes(
    panel: PanelData,
    weights: pd.DataFrame,
    *,
    roundtrip_cost_fraction: float,
) -> list[dict[str, float]]:
    episodes: list[dict[str, float]] = []
    for market in weights.columns:
        active = weights[market] > 1e-12
        starts = np.flatnonzero(active & ~active.shift(1, fill_value=False))
        for start in starts:
            later = np.flatnonzero(~active.iloc[start + 1 :].to_numpy())
            stop = start + 1 + int(later[0]) if len(later) else len(active) - 1
            entry = float(panel.open[market].iloc[start])
            if stop <= start or not math.isfinite(entry) or entry <= 0.0:
                continue
            exit_candidates = panel.open[market].iloc[stop:]
            exit_candidates = exit_candidates[
                np.isfinite(exit_candidates) & exit_candidates.gt(0.0)
            ]
            if exit_candidates.empty:
                continue
            exit_time = exit_candidates.index[0]
            exit_position = int(panel.open.index.get_loc(exit_time))
            exit_price = float(exit_candidates.iloc[0])
            path_high_values = panel.high[market].iloc[start : exit_position + 1]
            path_low_values = panel.low[market].iloc[start : exit_position + 1]
            path_high_values = path_high_values[
                np.isfinite(path_high_values) & path_high_values.gt(0.0)
            ]
            path_low_values = path_low_values[
                np.isfinite(path_low_values) & path_low_values.gt(0.0)
            ]
            if path_high_values.empty or path_low_values.empty:
                continue
            path_high = float(path_high_values.max())
            path_low = float(path_low_values.min())
            gross_return = exit_price / entry - 1.0
            episodes.append(
                {
                    "gross_return": gross_return,
                    "net_return": gross_return - roundtrip_cost_fraction,
                    "mfe_bps": (path_high / entry - 1.0) * 10_000.0,
                    "mae_bps": (path_low / entry - 1.0) * 10_000.0,
                    "holding_bars": float(exit_position - start),
                }
            )
    return episodes


def simulate_panel_stage0(
    panel: PanelData,
    weights: pd.DataFrame,
    *,
    card: HypothesisCard,
    parameters: Mapping[str, Any],
    costs: SharedCostModel,
    edge_policy: EconomicEdgePolicy,
    initial_equity_eur: float = 1_000.0,
) -> PanelStage0Result:
    if not weights.index.equals(panel.open.index) or tuple(weights.columns) != panel.markets:
        raise ValueError("Stage-0 weights do not match panel identity")
    executed = weights.shift(1).fillna(0.0)
    forward_open_return = panel.open.shift(-1) / panel.open - 1.0
    gross_returns = (executed * forward_open_return).sum(axis=1, min_count=1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    one_way = (
        costs.taker_fee_fraction
        + (costs.spread_bps / 2.0 + costs.slippage_bps) / 10_000.0
    )
    net_returns = gross_returns - turnover * one_way
    net_equity = initial_equity_eur * (1.0 + net_returns).cumprod()
    drawdown = net_equity / net_equity.cummax() - 1.0
    roundtrip = edge_policy.roundtrip_cost_bps / 10_000.0
    episodes = _position_episodes(
        panel,
        executed,
        roundtrip_cost_fraction=roundtrip,
    )
    gross_trade = [row["gross_return"] * 100.0 for row in episodes]
    net_trade = [row["net_return"] * 100.0 for row in episodes]
    gains = sum(value for value in net_trade if value > 0)
    losses = abs(sum(value for value in net_trade if value < 0))
    median_mfe = statistics.median(row["mfe_bps"] for row in episodes) if episodes else None
    move_assessment = edge_policy.assess(median_mfe)
    gross_opportunity = sum(max(0.0, value) for value in gross_trade)
    cost_drag = sum(gross_trade) - sum(net_trade)
    elapsed_days = max(
        1.0,
        (panel.close.index[-1] - panel.close.index[0]).total_seconds() / 86_400.0,
    )
    gross_total = float(sum(gross_trade))
    net_total = float(sum(net_trade))
    reasons = []
    if not episodes or gross_total <= 0:
        reasons.append("NEGATIVE_GROSS_EXPECTANCY")
    if not episodes or net_total <= 0:
        reasons.append("NEGATIVE_NET_EXPECTANCY")
    if len(episodes) < 30:
        reasons.append("INSUFFICIENT_SAMPLE")
    if not move_assessment["economically_large_enough"]:
        reasons.append("EXPECTED_MOVE_TOO_SMALL_FOR_COSTS")
    classification = (
        FamilyClassification.NOT_EVALUABLE
        if not episodes
        else (
            FamilyClassification.GROSS_NEGATIVE
            if gross_total <= 0
            else (
                FamilyClassification.GROSS_POSITIVE_NET_NEGATIVE
                if net_total <= 0
                else FamilyClassification.STAGE0_PROMISING
            )
        )
    )
    identity = {
        "schema": PANEL_STAGE0_SCHEMA_VERSION,
        "card": card.card_hash,
        "data": panel.data_hash,
        "parameters": parameters,
        "costs": costs.cost_model_version,
    }
    return PanelStage0Result(
        result_id=stable_hash(identity, length=48),
        hypothesis_id=card.hypothesis_id,
        family=card.family,
        parameter_set=dict(parameters),
        parameter_hash=stable_hash(parameters, length=32),
        data_hash=panel.data_hash,
        timeframe=panel.timeframe,
        market_count=len(panel.markets),
        signal_count=int((weights.diff().abs().sum(axis=1) > 0).sum()),
        trade_count=len(episodes),
        gross_pnl_eur=gross_total,
        net_pnl_eur=net_total,
        gross_expectancy_eur=(gross_total / len(episodes) if episodes else None),
        net_expectancy_eur=(net_total / len(episodes) if episodes else None),
        profit_factor=(gains / losses if losses else None),
        maximum_drawdown=abs(float(drawdown.min())),
        turnover=float(turnover.sum()),
        annualized_turnover=float(turnover.sum() / (elapsed_days / 365.25)),
        trades_per_week=float(len(episodes) / elapsed_days * 7.0),
        average_holding_bars=(
            statistics.fmean(row["holding_bars"] for row in episodes) if episodes else None
        ),
        median_mfe_bps=median_mfe,
        median_mae_bps=(
            statistics.median(row["mae_bps"] for row in episodes) if episodes else None
        ),
        median_holding_return_bps=(
            statistics.median(row["gross_return"] for row in episodes) * 10_000.0
            if episodes
            else None
        ),
        favorable_move_cost_coverage=(
            sum(
                row["mfe_bps"]
                >= edge_policy.roundtrip_cost_bps * edge_policy.minimum_move_cost_ratio
                for row in episodes
            )
            / len(episodes)
            if episodes
            else None
        ),
        expected_move_cost_ratio=move_assessment[
            "expected_move_to_roundtrip_cost_ratio"
        ],
        cost_as_fraction_of_gross_opportunity=(
            cost_drag / gross_opportunity if gross_opportunity > 0 else None
        ),
        classification=classification,
        rejection_reasons=tuple(reasons),
    )


def panel_net_return_path(
    panel: PanelData,
    weights: pd.DataFrame,
    *,
    costs: SharedCostModel,
) -> pd.Series:
    executed = weights.shift(1).fillna(0.0)
    forward_open_return = panel.open.shift(-1) / panel.open - 1.0
    gross = (executed * forward_open_return).sum(axis=1, min_count=1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    one_way = costs.taker_fee_fraction + (
        costs.spread_bps / 2.0 + costs.slippage_bps
    ) / 10_000.0
    return (gross - turnover * one_way).rename("net_return")


def panel_asset_and_regime_diagnostics(
    panel: PanelData,
    weights: pd.DataFrame,
    *,
    costs: SharedCostModel,
) -> dict[str, Any]:
    executed = weights.shift(1).fillna(0.0)
    forward = panel.open.shift(-1) / panel.open - 1.0
    turnover = executed.diff().abs().fillna(executed.abs())
    one_way = costs.taker_fee_fraction + (
        costs.spread_bps / 2.0 + costs.slippage_bps
    ) / 10_000.0
    contributions = executed * forward - turnover * one_way
    by_asset = {
        market: {
            "net_return_contribution": float(contributions[market].sum()),
            "positive": float(contributions[market].sum()) > 0,
            "active_bars": int((executed[market] > 0).sum()),
        }
        for market in panel.markets
    }
    portfolio = contributions.sum(axis=1)
    btc = panel.close["BTC-EUR"]
    btc_ema = btc.ewm(
        span=40 * panel.bars_per_day,
        adjust=False,
        min_periods=40 * panel.bars_per_day,
    ).mean()
    btc_vol = btc.pct_change(fill_method=None).rolling(
        20 * panel.bars_per_day,
        min_periods=20 * panel.bars_per_day,
    ).std()
    volatility_median = btc_vol.expanding(min_periods=100).median().shift(1)
    masks = {
        "BTC_TREND_UP": btc > btc_ema,
        "BTC_TREND_DOWN": btc <= btc_ema,
        "HIGH_VOLATILITY": btc_vol > volatility_median,
        "LOW_VOLATILITY": btc_vol <= volatility_median,
    }
    regimes = {}
    for name, mask in masks.items():
        values = portfolio.where(mask).dropna()
        regimes[name] = {
            "observations": len(values),
            "net_return_sum": float(values.sum()),
            "mean_bar_return": float(values.mean()) if len(values) else None,
        }
    positive_assets = sum(row["positive"] for row in by_asset.values())
    active_bars = int((executed > 0.0).sum().sum())
    return {
        "asset_results": by_asset,
        "asset_classification": (
            "NOT_EVALUABLE"
            if active_bars == 0
            else (
                "MULTI_ASSET_ROBUST"
                if positive_assets >= max(2, len(by_asset) // 2)
                else ("ASSET_SPECIFIC" if positive_assets >= 1 else "ASSET_FRAGILE")
            )
        ),
        "positive_asset_count": positive_assets,
        "regime_results": regimes,
        "regime_classification": (
            "NOT_EVALUABLE"
            if active_bars == 0
            else (
                "REGIME_FRAGILE"
                if any(
                    row["observations"] >= 100
                    and float(row["net_return_sum"] or 0.0) < 0
                    for row in regimes.values()
                )
                else "NO_CATASTROPHIC_COMMON_REGIME_FAILURE_OBSERVED"
            )
        ),
    }


def stage0_baselines(
    panel: PanelData,
    *,
    costs: SharedCostModel,
) -> dict[str, Any]:
    forward = panel.open.shift(-1) / panel.open - 1.0
    btc = forward["BTC-EUR"].fillna(0.0)
    eligible_count = panel.eligible.sum(axis=1).replace(0, np.nan)
    equal = (forward.where(panel.eligible).sum(axis=1) / eligible_count).fillna(0.0)
    one_way = costs.taker_fee_fraction + (
        costs.spread_bps / 2.0 + costs.slippage_bps
    ) / 10_000.0

    def metrics(values: pd.Series, turnover: float) -> dict[str, Any]:
        net = values.copy()
        if len(net):
            net.iloc[0] -= turnover * one_way
        equity = (1.0 + net).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        return {
            "net_return": float(equity.iloc[-1] - 1.0),
            "maximum_drawdown": abs(float(drawdown.min())),
            "turnover": turnover,
        }

    return {
        "BTC_BUY_AND_HOLD": metrics(btc, 1.0),
        "PIT_EQUAL_WEIGHT_ELIGIBLE": metrics(equal, 1.0),
        "BTC_EXPOSURE_MATCHED_20PCT": metrics(
            btc * STAGE0_MAXIMUM_RESEARCH_EXPOSURE,
            STAGE0_MAXIMUM_RESEARCH_EXPOSURE,
        ),
        "PIT_EQUAL_WEIGHT_EXPOSURE_MATCHED_20PCT": metrics(
            equal * STAGE0_MAXIMUM_RESEARCH_EXPOSURE,
            STAGE0_MAXIMUM_RESEARCH_EXPOSURE,
        ),
        "CASH": {"net_return": 0.0, "maximum_drawdown": 0.0, "turnover": 0.0},
    }


@dataclass(frozen=True, slots=True)
class FailedHypothesisRecord:
    hypothesis_id: str
    card_hash: str
    family: str
    tested_parameter_hashes: tuple[str, ...]
    economic_results_hash: str
    rejection_reasons: tuple[str, ...]
    data_version: str
    cost_model_version: str
    stage0_engine_version: str
    recorded_at_data_cutoff: str
    retest_requires: tuple[str, ...] = (
        "NEW_CAUSAL_DATA",
        "NEW_EXTERNAL_EVIDENCE",
        "MATERIALLY_DIFFERENT_ECONOMIC_MECHANISM",
    )

    @property
    def record_id(self) -> str:
        return stable_hash(asdict(self), length=48)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "record_id": self.record_id}


class FailedFamilyArchive:
    def __init__(self, path: Any) -> None:
        from pathlib import Path

        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        rows = value.get("records") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("failed-family archive schema mismatch")
        return [dict(row) for row in rows]

    def append(self, record: FailedHypothesisRecord) -> None:
        rows = self.records()
        if any(row.get("record_id") == record.record_id for row in rows):
            return
        same_experiment = [
            row
            for row in rows
            if row.get("card_hash") == record.card_hash
            and row.get("data_version") == record.data_version
            and row.get("cost_model_version") == record.cost_model_version
            and row.get("stage0_engine_version") == record.stage0_engine_version
        ]
        if same_experiment:
            raise ValueError("failed hypothesis already archived for identical evidence")
        rows.append(record.to_dict())
        payload = {
            "schema_version": "failed_hypothesis_archive_v1",
            "record_count": len(rows),
            "records": rows,
            "archive_hash": stable_hash(rows, length=64),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, payload)


def forward_candidate_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "exact_native_positive": evidence.get("exact_status") == "ROBUST_EXACT_PASS",
        "positive_exact_net_expectancy": float(evidence.get("net_expectancy") or 0.0) > 0,
        "acceptable_profit_factor": float(evidence.get("profit_factor") or 0.0) >= 1.10,
        "bounded_drawdown": float(evidence.get("maximum_drawdown") or 1.0) <= 0.30,
        "walk_forward": int(evidence.get("positive_walk_forward_folds") or 0) >= 2,
        "parameter_plateau": evidence.get("parameter_plateau") is True,
        "cost_plus_25_bps": float(evidence.get("cost_plus_25_net_expectancy") or 0.0) > 0,
        "liquidity_ready": evidence.get("liquidity_status") in {"TIER_1", "TIER_2"},
        "lookahead_safe": evidence.get("lookahead_safe") is True,
        "asset_robust": evidence.get("asset_status") in {
            "MULTI_ASSET_ROBUST",
            "ASSET_SPECIFIC",
        },
        "common_regime_safe": evidence.get("regime_status") not in {
            "CATASTROPHIC_COMMON_REGIME_FAILURE",
            None,
        },
    }
    passed = all(checks.values())
    return {
        "state": "FORWARD_CANDIDATE" if passed else "NOT_PROMOTED",
        "checks": checks,
        "stage0_only_promotion": False,
        "automatic_authority": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(selected) for key, selected in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(selected) for selected in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prior_exact_evidence(settings: Settings) -> dict[str, dict[str, Any]]:
    report_root = settings.paths.lab_dir / "reports"
    specifications = {
        "CROSS_SECTIONAL_MOMENTUM": (
            "rotation_institutional_audit_v2.json",
            "MATCHING_CAUSAL_PORTFOLIO_ENGINE_PRIOR_NOT_QUALIFIED",
        ),
        "MEDIUM_TERM_TREND_PULLBACK": (
            "absolute_momentum_campaign_v1.json",
            "RELATED_TREND_BASELINE_NOT_PULLBACK_EXACT_MATCH",
        ),
        "VOLATILITY_CONTRACTION_EXPANSION": (
            "volatility_contraction_campaign_v1.json",
            "MATCHING_CAUSAL_PORTFOLIO_ENGINE_PRIOR_NOT_PROMOTED",
        ),
        "QUALITY_CONSOLIDATION_BREAKOUT": (
            "portfolio_breakout_campaign_v1.json",
            "NAIVE_BREAKOUT_COMPARATOR_NOT_STRUCTURAL_EXACT_MATCH",
        ),
        "BTC_RELATIVE_ALT_ROTATION": (
            "residual_momentum_campaign_v1.json",
            "MATCHING_RESIDUAL_MOMENTUM_ENGINE_PRIOR_NOT_PROMOTED",
        ),
        "BREADTH_CONDITIONED_MOMENTUM": (
            "rotation_institutional_audit_v2.json",
            "BREADTH_ABLATION_CONTEXT_NOT_ISOLATED_EXACT_MATCH",
        ),
    }
    output = {}
    for family, (filename, relationship) in specifications.items():
        path = report_root / filename
        if not path.is_file():
            output[family] = {
                "status": "MISSING_PRIOR_EXACT_EVIDENCE",
                "relationship": relationship,
            }
            continue
        content = path.read_bytes()
        artifact = json.loads(content)
        primary = artifact.get("primary_result") or {}
        normal = primary.get("normal") or artifact.get("normal") or {}
        gates = primary.get("gates") or artifact.get("checks") or {}
        output[family] = {
            "artifact_path": str(path.resolve()),
            "artifact_sha256": sha256(content).hexdigest(),
            "artifact_status": artifact.get("status"),
            "relationship": relationship,
            "exact_metrics": normal.get("metrics"),
            "exact_cost_breakdown": normal.get("cost_breakdown"),
            "gates": gates,
            "live_ready": artifact.get("live_ready") is True,
            "paper_candidates": int(artifact.get("paper_candidates") or 0),
            "orders_generated": int(artifact.get("orders_generated") or 0),
        }
    return output


def _exact_family_status(prior: Mapping[str, Any] | None) -> str:
    if not prior or prior.get("status") == "MISSING_PRIOR_EXACT_EVIDENCE":
        return "NOT_EVALUABLE_NO_MATCHING_EXACT_ENGINE"
    relationship = str(prior.get("relationship") or "")
    if not relationship.startswith("MATCHING_"):
        return "NOT_EVALUABLE_PRIOR_ENGINE_NOT_EXACT_MATCH"
    if prior.get("live_ready") or int(prior.get("paper_candidates") or 0) > 0:
        return "PRIOR_EXACT_ROBUST_REQUIRES_CURRENT_DATA_REVALIDATION"
    metrics = prior.get("exact_metrics") or {}
    if float(metrics.get("net_return") or 0.0) > 0:
        return "PRIOR_EXACT_POSITIVE_NOT_ROBUST"
    return "PRIOR_EXACT_REJECTED"


class FrozenPanelSignalStrategy(Strategy):
    """Native exact-engine adapter for a preregistered panel selection path."""

    strategy_id = "P1_1_FROZEN_PANEL_SIGNAL"
    family = "P1_1_ALPHA_DISCOVERY"
    description = "Execute a causally precomputed panel-selection transition in the native engine."
    defaults = {
        "stop_atr": 3.0,
        "target_atr": 6.0,
        "trailing_atr": 3.0,
        "maximum_holding_bars": 84,
    }
    parameter_space = {
        "stop_atr": (2.5, 3.0, 3.5),
        "target_atr": (5.0, 6.0, 7.0),
    }

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        selected = self.parameters(parameters)
        if "alpha_weight" not in features:
            raise ValueError("frozen panel alpha_weight is missing")
        active = features["alpha_weight"].fillna(0.0) > 1e-12
        entry = active & ~active.shift(1, fill_value=False)
        exit_ = ~active & active.shift(1, fill_value=False)
        size = (features["alpha_weight"].fillna(0.0) / STAGE0_MAXIMUM_RESEARCH_EXPOSURE).clip(
            0.0, 1.0
        )
        return self._output(
            features,
            entry=entry,
            exit=exit_,
            parameters=selected,
            size_multiplier=size,
            entry_reason="P1_1_FROZEN_PANEL_SELECTION",
            exit_reason="P1_1_FROZEN_PANEL_DESELECTION",
            metadata={
                "production_registration": False,
                "authority": "EXACT_RESEARCH_ONLY",
            },
        )


def _exact_summary(result: Any) -> dict[str, Any]:
    return {
        "strategy_id": result.strategy_id,
        "trade_count": len(result.trades),
        "order_count": len(result.orders),
        "initial_cash_eur": result.initial_cash_eur,
        "ending_equity_eur": result.ending_equity_eur,
        "metrics": dict(result.metrics),
        "integrity": dict(result.integrity),
    }


def _native_feature_frames(
    raw_frames: Mapping[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return _attach_alpha_weights(_base_native_features(raw_frames), weights)


def _base_native_features(
    raw_frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output = {}
    for market, frame in raw_frames.items():
        selected = frame.copy()
        selected.attrs.update(frame.attrs)
        features = FeaturePipeline().build(selected, market=market)
        features.attrs.update(selected.attrs)
        output[market] = features
    return output


def _attach_alpha_weights(
    base_features: Mapping[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    output = {}
    for market, base in base_features.items():
        features = base.copy()
        features.attrs.update(base.attrs)
        features["alpha_weight"] = weights[market].reindex(features.index).fillna(0.0)
        output[market] = features
    return output


def _select_fold_parameters(
    panel: PanelData,
    *,
    card: HypothesisCard,
    costs: SharedCostModel,
    edge_policy: EconomicEdgePolicy,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    results = []
    for parameters in parameter_grid(card):
        weights = family_desired_weights(panel, family=card.family, parameters=parameters)
        results.append(
            simulate_panel_stage0(
                panel,
                weights,
                card=card,
                parameters=parameters,
                costs=costs,
                edge_policy=edge_policy,
            )
        )
    plateaus = parameter_plateaus(
        [row.to_dict() for row in results], card.parameters_to_test
    )
    stable = {row.parameter_hash for row in plateaus if row.stable}
    eligible = [
        row
        for row in results
        if row.parameter_hash in stable
        and row.classification == FamilyClassification.STAGE0_PROMISING
        and row.trade_count >= 20
        and float(row.profit_factor or 0.0) >= 1.0
    ]
    if not eligible:
        return None, {
            "status": "NO_STABLE_TRAIN_FOLD_PARAMETER",
            "tested_parameter_regions": len(results),
        }
    best = max(
        eligible,
        key=lambda row: (
            float(row.net_expectancy_eur or -math.inf),
            float(row.profit_factor or 0.0),
        ),
    )
    return dict(best.parameter_set), {
        "status": "PARAMETER_SELECTED_ON_TRAIN_ONLY",
        "parameter_hash": best.parameter_hash,
        "train_stage0": best.to_dict(),
        "tested_parameter_regions": len(results),
    }


def run_native_exact_alpha_validation(
    full_raw_frames: Mapping[str, pd.DataFrame],
    development_raw_frames: Mapping[str, pd.DataFrame],
    *,
    card: HypothesisCard,
    frozen_parameters: Mapping[str, Any],
    costs: SharedCostModel,
    edge_policy: EconomicEdgePolicy,
    settings: Settings,
    folds: int = 3,
    purge_bars: int = 40,
    embargo_bars: int = 2,
) -> dict[str, Any]:
    """Validate one Stage-0 survivor through the native event-driven engine."""

    strategy = FrozenPanelSignalStrategy()
    config = replace(
        BacktestConfig.from_settings(settings, initial_cash_eur=2_000.0),
        bootstrap_samples=100,
        monte_carlo_runs=100,
    )
    full_panel = build_point_in_time_panel(
        full_raw_frames, timeframe="4h", minimum_history_bars=120
    )
    frozen_weights = family_desired_weights(
        full_panel,
        family=card.family,
        parameters=frozen_parameters,
    )
    full_features = _attach_alpha_weights(
        _base_native_features(full_raw_frames), frozen_weights
    )
    _, validation, final_test = chronological_split(
        full_features,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    validation_result = BacktestEngine(config, settings=settings).run(
        validation, strategy, parameters=dict(strategy.defaults)
    )
    final_result = BacktestEngine(config, settings=settings).run(
        final_test, strategy, parameters=dict(strategy.defaults)
    )
    exact_cost_stress = []
    stressed_results = {}
    for additional_roundtrip_bps in (0.0, 10.0, 25.0, 50.0):
        if additional_roundtrip_bps == 0:
            result = final_result
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
            result = BacktestEngine(stressed_config, settings=settings).run(
                final_test, strategy, parameters=dict(strategy.defaults)
            )
        stressed_results[additional_roundtrip_bps] = result
        exact_cost_stress.append(
            {
                "scenario": "BASE" if additional_roundtrip_bps == 0 else f"BASE_PLUS_{int(additional_roundtrip_bps)}_BPS",
                "additional_roundtrip_bps": additional_roundtrip_bps,
                "result": _exact_summary(result),
            }
        )

    development_panel = build_point_in_time_panel(
        development_raw_frames, timeframe="4h", minimum_history_bars=120
    )
    development_base_features = _base_native_features(development_raw_frames)
    count = len(development_panel.close)
    initial_train = count // 3
    test_size = (count - initial_train) // folds
    nested_folds = []
    for fold in range(folds):
        boundary = initial_train + fold * test_size
        train_stop = max(120, boundary - purge_bars)
        test_start = min(count - 1, boundary + embargo_bars)
        test_stop = count if fold == folds - 1 else min(count, test_start + test_size)
        train_end_time = development_panel.close.index[train_stop - 1]
        test_start_time = development_panel.close.index[test_start]
        test_end_time = development_panel.close.index[test_stop - 1]
        train_frames = {
            market: frame.loc[:train_end_time].copy()
            for market, frame in development_raw_frames.items()
        }
        train_panel = build_point_in_time_panel(
            train_frames, timeframe="4h", minimum_history_bars=120
        )
        selected_parameters, selection = _select_fold_parameters(
            train_panel,
            card=card,
            costs=costs,
            edge_policy=edge_policy,
        )
        if selected_parameters is None:
            nested_folds.append(
                {
                    "fold": fold + 1,
                    "train_end": train_end_time.isoformat(),
                    "test_start": test_start_time.isoformat(),
                    "test_end": test_end_time.isoformat(),
                    "selection": selection,
                    "status": "FOLD_REJECTED_NO_TRAIN_PARAMETER",
                }
            )
            continue
        fold_weights = family_desired_weights(
            development_panel,
            family=card.family,
            parameters=selected_parameters,
        )
        fold_features = _attach_alpha_weights(development_base_features, fold_weights)
        test_features = {
            market: frame.loc[test_start_time:test_end_time].copy()
            for market, frame in fold_features.items()
        }
        result = BacktestEngine(config, settings=settings).run(
            test_features, strategy, parameters=dict(strategy.defaults)
        )
        metrics = result.metrics
        positive = (
            int(metrics.get("trade_count") or 0) > 0
            and float(metrics.get("net_expectancy_r") or 0.0) > 0
            and float(metrics.get("profit_factor") or 0.0) > 1.0
        )
        nested_folds.append(
            {
                "fold": fold + 1,
                "train_end": train_end_time.isoformat(),
                "purge_bars": purge_bars,
                "embargo_bars": embargo_bars,
                "test_start": test_start_time.isoformat(),
                "test_end": test_end_time.isoformat(),
                "selection": selection,
                "selected_parameters": selected_parameters,
                "result": _exact_summary(result),
                "positive": positive,
                "status": "POSITIVE" if positive else "NEGATIVE",
            }
        )
    positive_folds = sum(row.get("positive") is True for row in nested_folds)
    asset_holdout = {}
    for market, frame in final_test.items():
        result = BacktestEngine(config, settings=settings).run(
            {market: frame}, strategy, parameters=dict(strategy.defaults)
        )
        asset_holdout[market] = _exact_summary(result)
    final_metrics = final_result.metrics
    plus_25_metrics = stressed_results[25.0].metrics
    positive_assets = sum(
        float((row.get("metrics") or {}).get("net_expectancy_r") or 0.0) > 0
        for row in asset_holdout.values()
    )
    robust = (
        int(final_metrics.get("trade_count") or 0) >= settings.research.minimum_trades
        and float(final_metrics.get("net_expectancy_r") or 0.0) > 0
        and float(final_metrics.get("profit_factor") or 0.0)
        >= settings.research.minimum_profit_factor
        and float(final_metrics.get("maximum_drawdown") or 1.0)
        <= settings.research.maximum_drawdown
        and positive_folds >= 2
        and float(plus_25_metrics.get("net_expectancy_r") or 0.0) > 0
        and positive_assets >= 2
    )
    return {
        "status": "ROBUST_EXACT_PASS" if robust else "EXACT_REJECTED",
        "exact_engine": "research.backtest.BacktestEngine",
        "exact_engine_authoritative": True,
        "strategy_adapter": strategy.strategy_id,
        "frozen_family_parameters": dict(frozen_parameters),
        "parameters_frozen_before_final_test": True,
        "final_test_retuned": False,
        "validation": _exact_summary(validation_result),
        "final_test": _exact_summary(final_result),
        "exact_cost_stress": exact_cost_stress,
        "nested_walk_forward": {
            "mode": "ANCHORED_TRAIN_SELECT_THEN_NEXT_FOLD_EXACT",
            "folds": nested_folds,
            "positive_folds": positive_folds,
            "valid_fold_count": len(nested_folds),
            "final_20_percent_excluded": True,
        },
        "asset_holdout": asset_holdout,
        "positive_asset_count": positive_assets,
        "stage0_has_execution_authority": False,
        "orders_submitted": 0,
    }


def build_alpha_discovery_artifact(
    settings: Settings,
    *,
    markets: Sequence[str] = DEFAULT_ALPHA_ASSETS,
    maximum_rows_4h: int = 6_000,
    maximum_rows_1d: int = 1_200,
) -> dict[str, Any]:
    """Run the bounded P1.1 Stage-0 campaign and persist immutable evidence."""

    started = time.perf_counter()
    p0_5_latest = _read_json(settings.paths.output_dir / "economics" / "latest.json")
    p1_latest = _read_json(settings.paths.output_dir / "research_factory" / "latest.json")
    p0_5_path = Path(str(p0_5_latest.get("artifact_path") or ""))
    p1_path = Path(str(p1_latest.get("artifact_path") or ""))
    if not p0_5_path.is_file() or not p1_path.is_file():
        raise FileNotFoundError("binding P0.5 and P1 artifacts are required")
    p0_5_bytes = p0_5_path.read_bytes()
    p1_bytes = p1_path.read_bytes()
    p1 = json.loads(p1_bytes)
    if (p1.get("p0_5_branch") or {}).get("decision") != (
        "ALPHA_RESEARCH_RESET_REQUIRED_WITH_BOUNDED_PROMISING_EXCEPTION"
    ):
        raise ValueError("P1 reset branch is not binding")

    selected_markets = tuple(dict.fromkeys(str(market) for market in markets))
    disallowed = [
        market
        for market in selected_markets
        if settings.shariah.eligibility(market).status.value != "ALLOWED"
    ]
    if disallowed:
        raise ValueError(f"research universe contains non-allowed markets: {disallowed}")
    full_frames_by_timeframe: dict[str, dict[str, pd.DataFrame]] = {"4h": {}, "1d": {}}
    frames_by_timeframe: dict[str, dict[str, pd.DataFrame]] = {"4h": {}, "1d": {}}
    identities: list[dict[str, Any]] = []
    development_identities: list[dict[str, Any]] = []
    missing = []
    for timeframe, maximum_rows in (("4h", maximum_rows_4h), ("1d", maximum_rows_1d)):
        for market in selected_markets:
            path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            if not path.is_file():
                missing.append(
                    {"market": market, "timeframe": timeframe, "path": str(path)}
                )
                continue
            frame, identity = load_immutable_ohlcv(
                path,
                provider="bitvavo",
                market=market,
                timeframe=timeframe,
                maximum_rows=maximum_rows,
            )
            cutoff = max(100, int(len(frame) * 0.80))
            development = frame.iloc[:cutoff].copy()
            development.attrs.update(frame.attrs)
            development_identity = derive_dataset_identity(
                development,
                identity,
                purpose="P1.1_TRAIN_VALIDATION_ONLY_EXCLUDES_FINAL_20_PERCENT",
            )
            development.attrs["dataset_id"] = development_identity.dataset_id
            full_frames_by_timeframe[timeframe][market] = frame
            frames_by_timeframe[timeframe][market] = development
            identities.append(identity.to_dict())
            development_identities.append(development_identity.to_dict())
    if set(frames_by_timeframe["4h"]) != set(selected_markets):
        raise ValueError("complete 4h research universe is required")

    cards = initial_hypothesis_cards()
    registry = HypothesisRegistry(cards)
    supported_families = tuple(card.family for card in cards[:-1])
    ranking = rank_hypotheses(cards, supported_families=supported_families)
    costs = SharedCostModel.from_settings(settings)
    panel_4h = build_point_in_time_panel(
        frames_by_timeframe["4h"], timeframe="4h", minimum_history_bars=120
    )
    panel_1d = (
        build_point_in_time_panel(
            frames_by_timeframe["1d"], timeframe="1d", minimum_history_bars=120
        )
        if set(frames_by_timeframe["1d"]) == set(selected_markets)
        else None
    )
    liquidity = liquidity_profiles(frames_by_timeframe["4h"])
    median_tier = sorted(row.tier for row in liquidity)[len(liquidity) // 2]
    edge_policy = EconomicEdgePolicy.from_costs(
        costs,
        holding_period_hours=72.0,
        liquidity_tier=median_tier,
    )

    causality: dict[str, Any] = {}
    results: list[PanelStage0Result] = []
    family_runtime: dict[str, float] = {}
    for card in cards:
        if card.family not in supported_families:
            causality[card.family] = {
                "status": "NOT_EVALUABLE_INFORMATION_ONLY_DATA_UNAVAILABLE"
            }
            continue
        grid = parameter_grid(card)
        causality[card.family] = panel_causality_check(
            panel_4h,
            family=card.family,
            parameters=grid[0],
        )
        if causality[card.family]["status"] != "PASSED":
            continue
        family_started = time.perf_counter()
        for parameters in grid:
            weights = family_desired_weights(
                panel_4h,
                family=card.family,
                parameters=parameters,
            )
            results.append(
                simulate_panel_stage0(
                    panel_4h,
                    weights,
                    card=card,
                    parameters=parameters,
                    costs=costs,
                    edge_policy=edge_policy,
                )
            )
        family_runtime[card.family] = time.perf_counter() - family_started
    stage0_elapsed = time.perf_counter() - started

    family_results: list[dict[str, Any]] = []
    stage0_survivors: list[dict[str, Any]] = []
    best_weights: dict[str, pd.DataFrame] = {}
    best_paths: dict[str, pd.Series] = {}
    timeframe_results: dict[str, Any] = {}
    cost_stress: dict[str, Any] = {}
    plateau_rows: dict[str, Any] = {}
    for card in cards:
        selected = [row for row in results if row.family == card.family]
        if not selected:
            family_results.append(
                {
                    "hypothesis_id": card.hypothesis_id,
                    "family": card.family,
                    "classification": "NOT_EVALUABLE",
                    "reason": "INFORMATION_ONLY_DATA_OR_CAUSAL_ENGINE_UNAVAILABLE",
                    "tested_parameter_regions": 0,
                }
            )
            continue
        ordered = sorted(
            selected,
            key=lambda row: (
                float(row.net_expectancy_eur or -math.inf),
                float(row.profit_factor or 0.0),
            ),
            reverse=True,
        )
        best = ordered[0]
        plateau = parameter_plateaus(
            [row.to_dict() for row in selected],
            card.parameters_to_test,
        )
        plateau_by_hash = {row.parameter_hash: row for row in plateau}
        best_plateau = plateau_by_hash[best.parameter_hash]
        plateau_rows[card.family] = [asdict(row) for row in plateau]
        survivor = (
            best.classification == FamilyClassification.STAGE0_PROMISING
            and best.trade_count >= 30
            and float(best.profit_factor or 0.0) >= 1.05
            and best.maximum_drawdown <= 0.35
            and best_plateau.stable
            and best.expected_move_cost_ratio is not None
            and best.expected_move_cost_ratio >= edge_policy.minimum_move_cost_ratio
        )
        weights = family_desired_weights(
            panel_4h,
            family=card.family,
            parameters=best.parameter_set,
        )
        best_weights[card.family] = weights
        best_paths[card.family] = panel_net_return_path(
            panel_4h, weights, costs=costs
        )
        diagnostics = panel_asset_and_regime_diagnostics(
            panel_4h, weights, costs=costs
        )
        family_row = {
            "hypothesis_id": card.hypothesis_id,
            "family": card.family,
            "classification": str(best.classification),
            "tested_parameter_regions": len(selected),
            "best_result": best.to_dict(),
            "best_plateau": asdict(best_plateau),
            "stage0_survivor": survivor,
            **diagnostics,
        }
        family_results.append(family_row)
        if survivor:
            stage0_survivors.append(family_row)
        scenarios = []
        for additional_bps in (0.0, 10.0, 25.0, 50.0):
            selected_costs = costs.stressed(additional_roundtrip_bps=additional_bps)
            scenario = simulate_panel_stage0(
                panel_4h,
                weights,
                card=card,
                parameters=best.parameter_set,
                costs=selected_costs,
                edge_policy=EconomicEdgePolicy.from_costs(
                    selected_costs,
                    holding_period_hours=72.0,
                    liquidity_tier=median_tier,
                ),
            )
            scenarios.append(
                {
                    "scenario": "BASE" if additional_bps == 0 else f"BASE_PLUS_{int(additional_bps)}_BPS",
                    "additional_roundtrip_bps": additional_bps,
                    "net_expectancy_eur": scenario.net_expectancy_eur,
                    "profit_factor": scenario.profit_factor,
                    "classification": str(scenario.classification),
                }
            )
        cost_stress[card.family] = scenarios
        timeframe_rows = {"4h": best.to_dict()}
        if panel_1d is not None and "1d" in card.target_timeframes:
            daily_weights = family_desired_weights(
                panel_1d,
                family=card.family,
                parameters=best.parameter_set,
            )
            daily_result = simulate_panel_stage0(
                panel_1d,
                daily_weights,
                card=card,
                parameters=best.parameter_set,
                costs=costs,
                edge_policy=EconomicEdgePolicy.from_costs(
                    costs,
                    holding_period_hours=24.0 * 7.0,
                    liquidity_tier=median_tier,
                ),
            )
            timeframe_rows["1d"] = daily_result.to_dict()
        timeframe_results[card.family] = timeframe_rows

    card_by_family = {card.family: card for card in cards}
    development_cutoff = max(str(row["data_cutoff"]) for row in development_identities)
    failed_records: list[FailedHypothesisRecord] = []
    for row in family_results:
        if row.get("stage0_survivor") is True or not row.get("best_result"):
            continue
        best_result = dict(row["best_result"])
        reasons = list(best_result.get("rejection_reasons") or [])
        plateau = row.get("best_plateau") or {}
        if not plateau.get("stable"):
            reasons.append("PARAMETER_FRAGILE")
        if float(best_result.get("maximum_drawdown") or 0.0) > 0.35:
            reasons.append("EXCESSIVE_DRAWDOWN")
        if row.get("regime_classification") == "REGIME_FRAGILE":
            reasons.append("REGIME_FRAGILE")
        failed_records.append(
            FailedHypothesisRecord(
                hypothesis_id=str(row["hypothesis_id"]),
                card_hash=card_by_family[str(row["family"])].card_hash,
                family=str(row["family"]),
                tested_parameter_hashes=tuple(
                    sorted(
                        result.parameter_hash
                        for result in results
                        if result.family == row["family"]
                    )
                ),
                economic_results_hash=stable_hash(row, length=48),
                rejection_reasons=tuple(dict.fromkeys(reasons)),
                data_version=panel_4h.data_hash,
                cost_model_version=costs.cost_model_version,
                stage0_engine_version=PANEL_STAGE0_SCHEMA_VERSION,
                recorded_at_data_cutoff=development_cutoff,
            )
        )

    correlations = []
    families = sorted(best_paths)
    for index, first in enumerate(families):
        for second in families[index + 1 :]:
            aligned = pd.concat([best_paths[first], best_paths[second]], axis=1).dropna()
            correlation = None
            if (
                len(aligned) >= 30
                and float(aligned.iloc[:, 0].std(ddof=0)) > 0
                and float(aligned.iloc[:, 1].std(ddof=0)) > 0
            ):
                selected_correlation = float(
                    aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                )
                correlation = selected_correlation if math.isfinite(selected_correlation) else None
            correlations.append(
                {
                    "family_a": first,
                    "family_b": second,
                    "net_return_correlation": correlation,
                    "observation_count": len(aligned),
                    "duplicate_alpha_warning": correlation is not None and correlation >= 0.90,
                }
            )

    prior_exact = _prior_exact_evidence(settings)
    exact_results: dict[str, dict[str, Any]] = {}
    family_by_name = {str(row["family"]): row for row in family_results}
    for family, family_row in family_by_name.items():
        if not family_row.get("best_result"):
            continue
        if family_row.get("stage0_survivor") is True:
            try:
                current_exact = run_native_exact_alpha_validation(
                    full_frames_by_timeframe["4h"],
                    frames_by_timeframe["4h"],
                    card=card_by_family[family],
                    frozen_parameters=dict(family_row["best_result"]["parameter_set"]),
                    costs=costs,
                    edge_policy=edge_policy,
                    settings=settings,
                )
            except (ValueError, ArithmeticError, PermissionError) as exc:
                current_exact = {
                    "status": "EXACT_VALIDATION_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "exact_engine_authoritative": True,
                    "orders_submitted": 0,
                }
            exact_results[family] = {
                **current_exact,
                "prior_evidence": prior_exact.get(family),
                "current_stage0_survivor": True,
                "current_parameter_exactly_validated": current_exact.get("status")
                != "EXACT_VALIDATION_ERROR",
                "forward_authority": False,
            }
        else:
            exact_results[family] = {
                "status": "NOT_RUN_NO_STAGE0_SURVIVOR",
                "prior_evidence_status": _exact_family_status(prior_exact.get(family)),
                "prior_evidence": prior_exact.get(family),
                "current_stage0_survivor": False,
                "current_parameter_exactly_validated": False,
                "forward_authority": False,
            }
    promotion = {}
    for family, value in exact_results.items():
        family_row = family_by_name[family]
        final_metrics = ((value.get("final_test") or {}).get("metrics") or {})
        plus_25 = next(
            (
                row
                for row in value.get("exact_cost_stress") or []
                if row.get("scenario") == "BASE_PLUS_25_BPS"
            ),
            {},
        )
        plus_25_metrics = ((plus_25.get("result") or {}).get("metrics") or {})
        asset_status = str(family_row.get("asset_classification") or "NOT_EVALUABLE")
        regime_status = (
            "CATASTROPHIC_COMMON_REGIME_FAILURE"
            if family_row.get("regime_classification") == "REGIME_FRAGILE"
            else str(family_row.get("regime_classification") or "NOT_EVALUABLE")
        )
        promotion[family] = forward_candidate_gate(
            {
                "exact_status": value.get("status"),
                "net_expectancy": final_metrics.get("net_expectancy_r"),
                "profit_factor": final_metrics.get("profit_factor"),
                "maximum_drawdown": final_metrics.get("maximum_drawdown"),
                "positive_walk_forward_folds": (
                    (value.get("nested_walk_forward") or {}).get("positive_folds")
                ),
                "parameter_plateau": (family_row.get("best_plateau") or {}).get(
                    "stable"
                ),
                "cost_plus_25_net_expectancy": plus_25_metrics.get("net_expectancy_r"),
                "liquidity_status": median_tier,
                "lookahead_safe": causality.get(family, {}).get("status") == "PASSED",
                "asset_status": asset_status,
                "regime_status": regime_status,
            }
        )
    forward_candidates = [
        family for family, value in promotion.items() if value["state"] == "FORWARD_CANDIDATE"
    ]
    for family, value in exact_results.items():
        if (
            value.get("current_stage0_survivor") is not True
            or value.get("status") == "ROBUST_EXACT_PASS"
        ):
            continue
        failed_checks = [
            f"EXACT_GATE_FAILED:{name}"
            for name, passed in (promotion.get(family, {}).get("checks") or {}).items()
            if passed is not True
        ]
        failed_records.append(
            FailedHypothesisRecord(
                hypothesis_id=card_by_family[family].hypothesis_id,
                card_hash=card_by_family[family].card_hash,
                family=family,
                tested_parameter_hashes=tuple(
                    sorted(
                        result.parameter_hash
                        for result in results
                        if result.family == family
                    )
                ),
                economic_results_hash=stable_hash(value, length=48),
                rejection_reasons=tuple(
                    failed_checks
                    or [f"EXACT_STATUS:{value.get('status') or 'UNKNOWN'}"]
                ),
                data_version=panel_4h.data_hash,
                cost_model_version=costs.cost_model_version,
                stage0_engine_version=(
                    f"{PANEL_STAGE0_SCHEMA_VERSION}+native_backtest_exact_v1"
                ),
                recorded_at_data_cutoff=development_cutoff,
            )
        )

    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    source_evidence = {
        "p0_5_artifact_sha256": sha256(p0_5_bytes).hexdigest(),
        "p1_artifact_sha256": sha256(p1_bytes).hexdigest(),
        "alpha_discovery_source_sha256": source_hash,
    }
    run_identity = {
        "schema": ALPHA_DISCOVERY_SCHEMA_VERSION,
        "sources": source_evidence,
        "registry": registry.registry_hash,
        "development_data": sorted(row["dataset_id"] for row in development_identities),
        "cost_model": costs.cost_model_version,
    }
    run_id = stable_hash(run_identity, length=32)
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": ALPHA_DISCOVERY_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_iso(),
        "factory_version": PANEL_STAGE0_SCHEMA_VERSION,
        "source_evidence": {
            **source_evidence,
            "p0_5_artifact_path": str(p0_5_path.resolve()),
            "p1_artifact_path": str(p1_path.resolve()),
        },
        "current_alpha_failure_diagnosis": diagnose_p1_alpha_failure(p1),
        "timeframe_priority": {
            "primary": "4h",
            "context": ["1d", "1w where available"],
            "tactical": "1h where economically justified",
            "15m": "EXECUTION_REFINEMENT_ONLY_NOT_PRIMARY_ALPHA",
            "evidence_driven_not_permanent": True,
        },
        "hypothesis_registry": {
            "registry_hash": registry.registry_hash,
            "cards": [card.to_dict() for card in cards],
        },
        "research_priority_ranking": ranking,
        "search_space": {
            "hypothesis_count": len(cards),
            "stage0_hypothesis_count": len(supported_families),
            "parameter_region_count": len(results),
            "concrete_asset_timeframe_variants": len(results) * len(panel_4h.markets),
            "markets": list(panel_4h.markets),
            "primary_timeframe": "4h",
            "blind_indicator_combination_search": False,
        },
        "datasets": identities,
        "development_datasets": development_identities,
        "missing_datasets": missing,
        "pit_universe": {
            "status": "PIT_UNIVERSE_PARTIAL",
            "eligibility_from_first_available_candle": True,
            "current_shariah_allowlist_used": True,
            "historical_shariah_reconstruction": "NOT_EVALUABLE",
        },
        "liquidity_profiles": [asdict(row) for row in liquidity],
        "economic_edge_policy": asdict(edge_policy),
        "bias_checks": causality,
        "stage0": {
            "authority": "APPROXIMATE_RESEARCH_ONLY",
            "maximum_research_exposure": STAGE0_MAXIMUM_RESEARCH_EXPOSURE,
            "exposure_is_fixed_not_optimized": True,
            "results": [row.to_dict() for row in results],
            "family_results": family_results,
            "survivor_count": len(stage0_survivors),
            "survivors": stage0_survivors,
            "parameter_plateaus": plateau_rows,
        },
        "gross_vs_net_economics": [
            {
                "family": row["family"],
                "gross_pnl_eur": (row.get("best_result") or {}).get("gross_pnl_eur"),
                "net_pnl_eur": (row.get("best_result") or {}).get("net_pnl_eur"),
                "classification": row.get("classification"),
            }
            for row in family_results
        ],
        "turnover_and_expected_move": [
            {
                "family": row["family"],
                **{
                    key: (row.get("best_result") or {}).get(key)
                    for key in (
                        "annualized_turnover",
                        "trades_per_week",
                        "average_holding_bars",
                        "median_mfe_bps",
                        "median_mae_bps",
                        "median_holding_return_bps",
                        "favorable_move_cost_coverage",
                        "expected_move_cost_ratio",
                        "cost_as_fraction_of_gross_opportunity",
                    )
                },
            }
            for row in family_results
        ],
        "baselines": stage0_baselines(panel_4h, costs=costs),
        "timeframe_results": timeframe_results,
        "cost_stress": cost_stress,
        "prior_exact_evidence": prior_exact,
        "exact_results": exact_results,
        "walk_forward_results": {
            family: value.get("nested_walk_forward")
            or "NOT_RUN_NO_STAGE0_SURVIVOR"
            for family, value in exact_results.items()
        },
        "strategy_correlation": correlations,
        "promotion_gates": promotion,
        "forward_candidates": forward_candidates,
        "failed_hypothesis_archive": {
            "path": str((settings.paths.output_dir / "alpha_discovery" / "failed_hypotheses.json").resolve()),
            "records": [row.to_dict() for row in failed_records],
            "record_count": len(failed_records),
            "retest_requires_new_evidence": True,
        },
        "benchmark": {
            "stage0_elapsed_seconds": stage0_elapsed,
            "total_elapsed_seconds": elapsed,
            "parameter_regions_per_second": len(results) / stage0_elapsed if stage0_elapsed else None,
            "family_runtime_seconds": family_runtime,
        },
        "ml_status": {"authority": "SHADOW_ONLY", "authority_changes": 0},
        "portfolio_allocator": {"built": False, "reason": "ALPHA_REQUIRED_FIRST"},
        "safety": {
            "new_real_orders_submitted": 0,
            "real_orders_cancelled": 0,
            "real_protective_orders_modified": 0,
            "private_bitvavo_mutations_caused_by_research": 0,
            "authority_increases": 0,
            "risk_increases": 0,
            "shariah_weakening": 0,
        },
    }
    payload = _json_safe(payload)
    payload["artifact_hash"] = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "artifact_hash", "benchmark"}
        },
        length=64,
    )
    root = settings.paths.output_dir / "alpha_discovery"
    artifact_path = root / "runs" / run_id / "alpha_discovery_evidence.json"
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
            raise FileExistsError(f"immutable alpha-discovery collision: {artifact_path}")
        payload = existing
    else:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact_path, payload)
    archive = FailedFamilyArchive(root / "failed_hypotheses.json")
    for record in failed_records:
        archive.append(record)
    status = {
        "schema_version": "alpha_discovery_operator_status_v1",
        "run_id": run_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": payload["artifact_hash"],
        "hypothesis_count": len(cards),
        "tested_parameter_regions": len(results),
        "stage0_survivors": len(stage0_survivors),
        "exact_current_parameter_survivors": sum(
            value.get("status") == "ROBUST_EXACT_PASS" for value in exact_results.values()
        ),
        "forward_candidates": len(forward_candidates),
        "ml_authority": "SHADOW_ONLY",
        "live_authority_changed": False,
    }
    atomic_write_json(root / "latest.json", status)
    return {"status": "COMPLETE", **status}


__all__ = [
    "ALPHA_DISCOVERY_SCHEMA_VERSION",
    "ECONOMIC_EDGE_POLICY_VERSION",
    "PANEL_STAGE0_SCHEMA_VERSION",
    "STAGE0_MAXIMUM_RESEARCH_EXPOSURE",
    "CandidateOrigin",
    "EconomicEdgePolicy",
    "FailedFamilyArchive",
    "FailedHypothesisRecord",
    "FamilyClassification",
    "HypothesisCard",
    "HypothesisRegistry",
    "LiquidityProfile",
    "PanelData",
    "PanelStage0Result",
    "build_point_in_time_panel",
    "build_alpha_discovery_artifact",
    "diagnose_p1_alpha_failure",
    "family_desired_weights",
    "forward_candidate_gate",
    "initial_hypothesis_cards",
    "liquidity_profiles",
    "panel_causality_check",
    "panel_asset_and_regime_diagnostics",
    "panel_net_return_path",
    "parameter_grid",
    "rank_hypotheses",
    "simulate_panel_stage0",
    "stage0_baselines",
]
