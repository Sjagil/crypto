"""Classical fail-closed regime routing for approved strategy sleeves.

The router is deliberately not a strategy optimizer and has no execution
authority. It classifies the latest fully closed daily market state with
backward-looking price features, applies entry hysteresis, and can allocate
risk only to sleeves whose independent lifecycle evidence already satisfies
the requested operating mode. If evidence or regime confidence is missing,
the only valid route is cash.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.portfolio_selection import (
    RotationPortfolioPolicy,
    _validated_panel,
)
from utils.common import stable_hash

REGIME_ROUTER_VERSION = "1.0.0"
GENESIS_HASH = "0" * 64


class MarketRegime(StrEnum):
    TREND_RISK_ON = "TREND_RISK_ON"
    RANGE_RISK_ON = "RANGE_RISK_ON"
    RISK_OFF = "RISK_OFF"
    UNCERTAIN = "UNCERTAIN"


class SleeveStyle(StrEnum):
    TREND = "TREND"
    MEAN_REVERSION = "MEAN_REVERSION"
    EVENT_CONTINUATION = "EVENT_CONTINUATION"
    LIQUIDITY_RECOVERY = "LIQUIDITY_RECOVERY"


class RouterMode(StrEnum):
    RESEARCH_OBSERVER = "RESEARCH_OBSERVER"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True, slots=True)
class RegimeRouterPolicy:
    """Immutable classical regime and allocation thresholds."""

    btc_ema_period: int = 200
    ema_slope_lookback: int = 20
    choppiness_period: int = 14
    trend_choppiness_maximum: float = 45.0
    range_choppiness_minimum: float = 55.0
    volatility_lookback: int = 30
    volatility_baseline_lookback: int = 252
    breadth_minimum: float = 0.50
    risk_on_confirmation_periods: int = 3
    maximum_total_exposure: float = 0.40
    maximum_sleeve_exposure: float = 0.20
    minimum_cash: float = 0.60

    def __post_init__(self) -> None:
        if self.btc_ema_period != 200:
            raise ValueError("router BTC EMA is fixed at 200")
        if self.ema_slope_lookback != 20:
            raise ValueError("router EMA slope lookback is fixed at 20")
        if self.choppiness_period != 14:
            raise ValueError("router choppiness period is fixed at 14")
        if not (
            0.0
            < self.trend_choppiness_maximum
            < self.range_choppiness_minimum
            < 100.0
        ):
            raise ValueError("invalid choppiness thresholds")
        if self.volatility_lookback != 30:
            raise ValueError("router volatility lookback is fixed at 30")
        if self.volatility_baseline_lookback != 252:
            raise ValueError(
                "router volatility baseline is fixed at 252"
            )
        if not 0.0 < self.breadth_minimum <= 1.0:
            raise ValueError("invalid breadth minimum")
        if self.risk_on_confirmation_periods < 2:
            raise ValueError("risk-on hysteresis must be at least two")
        if not 0.0 < self.maximum_total_exposure <= 1.0:
            raise ValueError("invalid total exposure")
        if not 0.0 < self.maximum_sleeve_exposure <= 1.0:
            raise ValueError("invalid sleeve exposure")
        if not 0.0 <= self.minimum_cash < 1.0:
            raise ValueError("invalid cash floor")
        if (
            self.maximum_total_exposure
            > 1.0 - self.minimum_cash + 1e-12
        ):
            raise ValueError("total exposure violates cash floor")

    @property
    def policy_hash(self) -> str:
        return stable_hash(asdict(self), length=64)


@dataclass(frozen=True, slots=True)
class StrategySleeve:
    """Fail-closed lifecycle evidence for one frozen strategy sleeve."""

    strategy_id: str
    family: str
    style: SleeveStyle
    strategy_dna_hash: str
    research_pass: bool
    forward_pass: bool
    shadow_candidate_permitted: bool
    paper_candidate_permitted: bool
    live_ready: bool
    orders_generated: int = 0

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id is required")
        if not self.family.strip():
            raise ValueError("family is required")
        if len(self.strategy_dna_hash) != 64:
            raise ValueError("strategy DNA hash must contain 64 chars")
        if self.orders_generated != 0:
            raise ValueError("router sleeves must be orderless")
        if self.live_ready and not self.paper_candidate_permitted:
            raise ValueError("live sleeve must first permit paper")
        if (
            self.paper_candidate_permitted
            and not self.shadow_candidate_permitted
        ):
            raise ValueError("paper sleeve must first permit shadow")
        if (
            self.shadow_candidate_permitted
            and not (self.research_pass and self.forward_pass)
        ):
            raise ValueError(
                "shadow sleeve needs research and forward passes"
            )


def _ohlc_panels(
    frames: Mapping[str, pd.DataFrame],
    *,
    policy: RotationPortfolioPolicy,
    benchmark: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens, closes = _validated_panel(
        frames,
        benchmark_market=benchmark,
        portfolio_policy=policy,
    )
    normalized = {
        market.upper().replace("/", "-").replace("_", "-"): frame
        for market, frame in frames.items()
    }
    panels: dict[str, pd.DataFrame] = {}
    for field in ("high", "low"):
        values: dict[str, pd.Series] = {}
        for market in closes.columns:
            frame = normalized[market]
            if field not in frame:
                raise ValueError(f"{market} missing router {field}")
            series = pd.to_numeric(
                frame[field],
                errors="raise",
            ).copy()
            series.index = pd.to_datetime(series.index, utc=True)
            values[market] = (
                series[~series.index.duplicated(keep="last")]
                .sort_index()
                .reindex(closes.index)
            )
        panel = pd.DataFrame(values, index=closes.index)
        finite = panel.stack().to_numpy(dtype=float)
        if not np.isfinite(finite).all() or (finite <= 0.0).any():
            raise ValueError(f"invalid router {field} values")
        panels[field] = panel
    return opens, closes, panels["high"], panels["low"]


def _choppiness(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int,
) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high.sub(low),
            high.sub(previous_close).abs(),
            low.sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    range_sum = true_range.rolling(
        period,
        min_periods=period,
    ).sum()
    high_max = high.rolling(period, min_periods=period).max()
    low_min = low.rolling(period, min_periods=period).min()
    denominator = high_max.sub(low_min).replace(0.0, np.nan)
    return (
        100.0
        * np.log10(range_sum.div(denominator))
        / math.log10(period)
    )


def classify_latest_regime(
    frames: Mapping[str, pd.DataFrame],
    *,
    portfolio_policy: RotationPortfolioPolicy,
    router_policy: RegimeRouterPolicy,
    benchmark_market: str = "BTC-EUR",
) -> dict[str, Any]:
    """Classify the latest closed daily state without future data."""

    benchmark = (
        benchmark_market.upper().replace("/", "-").replace("_", "-")
    )
    _, closes, highs, lows = _ohlc_panels(
        frames,
        policy=portfolio_policy,
        benchmark=benchmark,
    )
    minimum_rows = max(
        router_policy.btc_ema_period
        + router_policy.ema_slope_lookback,
        router_policy.volatility_lookback
        + router_policy.volatility_baseline_lookback,
        portfolio_policy.minimum_history_observations,
    )
    if len(closes) < minimum_rows:
        raise ValueError("insufficient history for regime router")
    ema = closes.ewm(
        span=router_policy.btc_ema_period,
        adjust=False,
        min_periods=router_policy.btc_ema_period,
    ).mean()
    btc_close = closes[benchmark]
    btc_ema = ema[benchmark]
    btc_ema_slope = btc_ema.div(
        btc_ema.shift(router_policy.ema_slope_lookback)
    ).sub(1.0)
    choppiness = _choppiness(
        highs[benchmark],
        lows[benchmark],
        btc_close,
        period=router_policy.choppiness_period,
    )
    log_returns = np.log(btc_close.where(btc_close > 0.0)).diff()
    volatility = log_returns.rolling(
        router_policy.volatility_lookback,
        min_periods=router_policy.volatility_lookback,
    ).std(ddof=0) * math.sqrt(365.25)
    volatility_baseline = volatility.rolling(
        router_policy.volatility_baseline_lookback,
        min_periods=router_policy.volatility_baseline_lookback,
    ).median()
    breadth = closes.gt(ema).mean(axis=1)
    timestamp = closes.index[-1]
    values = {
        "btc_close": float(btc_close.loc[timestamp]),
        "btc_ema200": float(btc_ema.loc[timestamp]),
        "btc_ema_slope_20": float(btc_ema_slope.loc[timestamp]),
        "choppiness_14": float(choppiness.loc[timestamp]),
        "annualized_volatility_30": float(
            volatility.loc[timestamp]
        ),
        "volatility_baseline_252": float(
            volatility_baseline.loc[timestamp]
        ),
        "breadth_above_ema200": float(breadth.loc[timestamp]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raw_regime = MarketRegime.UNCERTAIN
        reason = "NON_FINITE_CAUSAL_FEATURE"
    else:
        above_trend = values["btc_close"] > values["btc_ema200"]
        slope_positive = values["btc_ema_slope_20"] > 0.0
        high_volatility = (
            values["annualized_volatility_30"]
            > values["volatility_baseline_252"]
        )
        broad = (
            values["breadth_above_ema200"]
            >= router_policy.breadth_minimum
        )
        if (not above_trend) or (high_volatility and not broad):
            raw_regime = MarketRegime.RISK_OFF
            reason = "BTC_TREND_OR_HIGH_VOLATILITY_BREADTH_FAILURE"
        elif (
            slope_positive
            and broad
            and values["choppiness_14"]
            <= router_policy.trend_choppiness_maximum
        ):
            raw_regime = MarketRegime.TREND_RISK_ON
            reason = "CAUSAL_TREND_AND_BREADTH_CONFIRMED"
        elif (
            broad
            and values["choppiness_14"]
            >= router_policy.range_choppiness_minimum
        ):
            raw_regime = MarketRegime.RANGE_RISK_ON
            reason = "CAUSAL_RANGE_AND_BREADTH_CONFIRMED"
        else:
            raw_regime = MarketRegime.UNCERTAIN
            reason = "NO_PREDECLARED_REGIME_DOMINATES"
    return {
        "decision_at": timestamp.isoformat(),
        "raw_regime": raw_regime.value,
        "reason": reason,
        "features": values,
        "feature_causality": (
            "COMPLETED_DAILY_CANDLES_EXECUTABLE_NEXT_OPEN"
        ),
    }


def apply_regime_hysteresis(
    raw_regime: MarketRegime,
    *,
    previous: Mapping[str, Any] | None,
    policy: RegimeRouterPolicy,
) -> dict[str, Any]:
    """Require repeated risk-on evidence; risk-off always switches now."""

    previous = dict(previous or {})
    previous_active = MarketRegime(
        str(previous.get("active_regime") or MarketRegime.UNCERTAIN)
    )
    previous_pending = str(previous.get("pending_regime") or "")
    previous_count = int(previous.get("pending_count") or 0)
    if raw_regime is MarketRegime.RISK_OFF:
        return {
            "active_regime": MarketRegime.RISK_OFF.value,
            "pending_regime": None,
            "pending_count": 0,
            "transition": "IMMEDIATE_RISK_OFF",
        }
    if raw_regime is MarketRegime.UNCERTAIN:
        return {
            "active_regime": MarketRegime.UNCERTAIN.value,
            "pending_regime": None,
            "pending_count": 0,
            "transition": "IMMEDIATE_UNCERTAIN_TO_CASH",
        }
    if raw_regime is previous_active:
        return {
            "active_regime": previous_active.value,
            "pending_regime": None,
            "pending_count": 0,
            "transition": "REGIME_MAINTAINED",
        }
    pending_count = (
        previous_count + 1
        if previous_pending == raw_regime.value
        else 1
    )
    if pending_count >= policy.risk_on_confirmation_periods:
        return {
            "active_regime": raw_regime.value,
            "pending_regime": None,
            "pending_count": 0,
            "transition": "RISK_ON_HYSTERESIS_CONFIRMED",
        }
    return {
        "active_regime": previous_active.value,
        "pending_regime": raw_regime.value,
        "pending_count": pending_count,
        "transition": "RISK_ON_CONFIRMATION_PENDING",
    }


def _mode_eligibility(
    sleeve: StrategySleeve,
    mode: RouterMode,
) -> tuple[bool, list[str]]:
    checks = {
        "research_pass": sleeve.research_pass,
        "forward_pass": sleeve.forward_pass,
    }
    if mode in {RouterMode.SHADOW, RouterMode.PAPER, RouterMode.LIVE}:
        checks["shadow_candidate_permitted"] = (
            sleeve.shadow_candidate_permitted
        )
    if mode in {RouterMode.PAPER, RouterMode.LIVE}:
        checks["paper_candidate_permitted"] = (
            sleeve.paper_candidate_permitted
        )
    if mode is RouterMode.LIVE:
        checks["live_ready"] = sleeve.live_ready
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def _styles_for_regime(
    regime: MarketRegime,
) -> frozenset[SleeveStyle]:
    if regime is MarketRegime.TREND_RISK_ON:
        return frozenset(
            {
                SleeveStyle.TREND,
                SleeveStyle.EVENT_CONTINUATION,
            }
        )
    if regime is MarketRegime.RANGE_RISK_ON:
        return frozenset(
            {
                SleeveStyle.MEAN_REVERSION,
                SleeveStyle.LIQUIDITY_RECOVERY,
            }
        )
    return frozenset()


def route_approved_sleeves(
    sleeves: Sequence[StrategySleeve],
    *,
    active_regime: MarketRegime,
    mode: RouterMode,
    policy: RegimeRouterPolicy,
) -> dict[str, Any]:
    """Allocate only approved, regime-compatible sleeves or remain cash."""

    permitted_styles = _styles_for_regime(active_regime)
    audit: list[dict[str, Any]] = []
    eligible: list[StrategySleeve] = []
    for sleeve in sorted(sleeves, key=lambda row: row.strategy_id):
        lifecycle_eligible, failed = _mode_eligibility(sleeve, mode)
        style_eligible = sleeve.style in permitted_styles
        if not style_eligible:
            failed = [*failed, "regime_style_match"]
        accepted = lifecycle_eligible and style_eligible
        audit.append(
            {
                "strategy_id": sleeve.strategy_id,
                "family": sleeve.family,
                "style": sleeve.style.value,
                "strategy_dna_hash": sleeve.strategy_dna_hash,
                "eligible": accepted,
                "failed_checks": failed,
            }
        )
        if accepted:
            eligible.append(sleeve)
    allocations: dict[str, float] = {}
    if eligible:
        per_sleeve = min(
            policy.maximum_sleeve_exposure,
            policy.maximum_total_exposure / len(eligible),
        )
        allocations = {
            sleeve.strategy_id: float(per_sleeve)
            for sleeve in eligible
        }
    total_exposure = float(sum(allocations.values()))
    cash = float(1.0 - total_exposure)
    if (
        total_exposure > policy.maximum_total_exposure + 1e-12
        or cash < policy.minimum_cash - 1e-12
        or any(
            weight > policy.maximum_sleeve_exposure + 1e-12
            for weight in allocations.values()
        )
    ):
        raise RuntimeError("regime router exposure invariant violated")
    if not permitted_styles:
        status = "CASH_ONLY_REGIME_BLOCKED"
    elif not eligible:
        status = "CASH_ONLY_NO_APPROVED_STRATEGIES"
    else:
        status = "APPROVED_STRATEGIES_ROUTED_ORDERLESS"
    return {
        "status": status,
        "mode": mode.value,
        "active_regime": active_regime.value,
        "permitted_styles": sorted(
            style.value for style in permitted_styles
        ),
        "allocations": allocations,
        "total_exposure": total_exposure,
        "cash_fraction": cash,
        "sleeves_considered": len(sleeves),
        "eligible_sleeves": len(eligible),
        "eligibility_audit": audit,
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def sleeve_from_campaign_report(
    report: Mapping[str, Any],
    *,
    style: SleeveStyle,
) -> StrategySleeve:
    """Extract lifecycle evidence conservatively from a campaign report."""

    primary = dict(report.get("primary_result") or {})
    normal = dict(primary.get("normal") or {})
    gates = dict(primary.get("gates") or {})
    forward_summaries = dict(report.get("forward_summaries") or {})
    forward_pass = any(
        bool(summary.get("forward_performance_pass"))
        and all(
            bool(value)
            for value in dict(summary.get("checks") or {}).values()
        )
        for summary in forward_summaries.values()
    )
    strategy_id = str(
        report.get("primary_strategy_id")
        or primary.get("strategy_id")
        or "UNKNOWN"
    )
    dna_hash = str(
        primary.get("strategy_dna_hash")
        or normal.get("strategy_dna_hash")
        or ""
    )
    if len(dna_hash) != 64:
        dna_hash = stable_hash(
            {
                "campaign": report.get("campaign"),
                "strategy_id": strategy_id,
                "missing_dna": True,
            },
            length=64,
        )
    research_pass = bool(gates.get("research_pass", False))
    shadow_permitted = bool(
        gates.get("shadow_candidate_permitted", False)
        or report.get("shadow_candidate_permitted", False)
    )
    paper_permitted = bool(
        gates.get("paper_candidate_permitted", False)
        or report.get("paper_candidate_permitted", False)
    )
    live_ready = bool(
        gates.get("live_ready", False)
        and report.get("live_ready", False)
    )
    return StrategySleeve(
        strategy_id=strategy_id,
        family=str(
            report.get("strategy_family")
            or report.get("campaign")
            or "UNKNOWN"
        ),
        style=style,
        strategy_dna_hash=dna_hash,
        research_pass=research_pass,
        forward_pass=forward_pass,
        shadow_candidate_permitted=shadow_permitted,
        paper_candidate_permitted=paper_permitted,
        live_ready=live_ready,
        orders_generated=int(report.get("orders_generated") or 0),
    )


def append_router_decision(
    existing: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one deduplicated cryptographic router decision."""

    existing = dict(existing or {})
    decisions = list(existing.get("decisions") or [])
    audit_router_decision_chain(decisions)
    decision_at = str(record["decision_at"])
    if decisions and str(decisions[-1].get("decision_at")) == decision_at:
        return {
            "decisions": decisions,
            "decision_count": len(decisions),
            "chain_root_hash": str(
                decisions[-1].get("record_hash") or GENESIS_HASH
            ),
            "deduplicated": True,
        }
    previous_hash = (
        str(decisions[-1]["record_hash"])
        if decisions
        else GENESIS_HASH
    )
    payload = {
        **dict(record),
        "sequence_number": len(decisions),
        "previous_hash": previous_hash,
    }
    payload["record_hash"] = stable_hash(
        {
            "previous_hash": previous_hash,
            "record": payload,
        },
        length=64,
    )
    decisions.append(payload)
    return {
        "decisions": decisions,
        "decision_count": len(decisions),
        "chain_root_hash": payload["record_hash"],
        "deduplicated": False,
    }


def audit_router_decision_chain(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify sequence, previous hashes, and every decision record hash."""

    expected_previous = GENESIS_HASH
    for expected_sequence, raw in enumerate(decisions):
        record = dict(raw)
        stored_hash = str(record.pop("record_hash", ""))
        if int(record.get("sequence_number", -1)) != expected_sequence:
            raise RuntimeError(
                f"REGIME_ROUTER_SEQUENCE_BREAK:{expected_sequence}"
            )
        if str(record.get("previous_hash")) != expected_previous:
            raise RuntimeError(
                f"REGIME_ROUTER_PREVIOUS_HASH_BREAK:{expected_sequence}"
            )
        computed = stable_hash(
            {
                "previous_hash": expected_previous,
                "record": record,
            },
            length=64,
        )
        if stored_hash != computed:
            raise RuntimeError(
                f"REGIME_ROUTER_RECORD_HASH_BREAK:{expected_sequence}"
            )
        expected_previous = stored_hash
    return {
        "status": "PASSED",
        "decision_count": len(decisions),
        "chain_root_hash": expected_previous,
    }


__all__ = [
    "GENESIS_HASH",
    "REGIME_ROUTER_VERSION",
    "MarketRegime",
    "RegimeRouterPolicy",
    "RouterMode",
    "SleeveStyle",
    "StrategySleeve",
    "append_router_decision",
    "audit_router_decision_chain",
    "apply_regime_hysteresis",
    "classify_latest_regime",
    "route_approved_sleeves",
    "sleeve_from_campaign_report",
]
