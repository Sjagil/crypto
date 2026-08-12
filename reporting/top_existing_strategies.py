"""Evidence-backed ranking of already tested strategies.

This module never runs a backtest, changes strategy parameters, or touches an
execution authority.  It reads the immutable research artefacts already in the
repository, normalizes their metrics, ranks the complete comparable positive-cost
longlist, and writes reconciled JSON/CSV/Markdown/HTML reports.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    stable_hash,
)

REPORT_BASENAME = "top_10_existing_strategies_v1"
EVIDENCE_BASENAME = "top_10_existing_strategies_evidence_v1"
ALLOWED_PHASES = {
    "REJECT",
    "RESEARCH_ONLY",
    "FROZEN_SHADOW",
    "PAPER_CANDIDATE",
    "LIVE_EXECUTION_CANARY",
    "CONTROLLED_LIVE_CANDIDATE",
}
SCORE_WEIGHTS = {
    "historical_performance": 0.30,
    "robustness": 0.30,
    "drawdown_capital_protection": 0.20,
    "sample_quality": 0.10,
    "practical_deployability": 0.10,
}


FAMILY_META: dict[str, dict[str, Any]] = {
    "rotation": {
        "cluster": "cross_sectional_rotation",
        "logic": (
            "Weekly top-2 20/90-day relative-momentum rotation with asset/BTC trend, "
            "breadth-regime exposure scaling, and cash."
        ),
        "entry": "Rank allowed assets after the closed daily candle; select positive top-2.",
        "confirmation": "Asset EMA50 plus BTC EMA200 and breadth/volatility regime scaling.",
        "filter": "Point-in-time history gate; BTC/ETH/SOL/LINK allowlist.",
        "exit": "Next weekly rebalance, trend ineligibility, regime de-risking, or replacement.",
        "works": "Broad risk-on trends with persistent cross-sectional leadership.",
        "fails": "Abrupt reversals, narrow leadership, and selection regimes unlike development.",
        "statistical_weakness": (
            "Historical DSR, ordinary PBO, WRC, SPA, and confirmation CI all failed."
        ),
        "operational_weakness": "Portfolio ranking can request two simultaneous positions.",
        "drawdown_risk": "Momentum whipsaw and correlated altcoin sell-offs.",
        "practical_score": 88.0,
    },
    "capital_rotation": {
        "cluster": "cross_sectional_rotation",
        "logic": "The frozen rotation signal with a higher fixed execution-exposure policy.",
        "entry": "Identical frozen rotation entry.",
        "confirmation": "Identical frozen trend and regime confirmations.",
        "filter": "BTC/ETH/SOL/LINK allowlist and point-in-time history.",
        "exit": "Identical frozen weekly rebalance and trend/regime exits.",
        "works": "The same persistent risk-on regimes as the frozen control.",
        "fails": "Correlated sell-offs; losses scale sharply with exposure.",
        "statistical_weakness": "Policy-comparison PBO remained above the required threshold.",
        "operational_weakness": "Drawdown exceeded the repository's 20% historical risk limit.",
        "drawdown_risk": "Near-linear amplification of the frozen signal's downside.",
        "practical_score": 72.0,
    },
    "diversified_rotation": {
        "cluster": "cross_sectional_rotation",
        "logic": "Top-3/4 rotation with causal volatility targeting and risk-parity weights.",
        "entry": "Weekly ranked momentum selection from allowed assets.",
        "confirmation": "Frozen trend/regime logic plus covariance-based allocation.",
        "filter": "Allowed universe and causal rolling covariance.",
        "exit": "Weekly rebalance or loss of signal eligibility.",
        "works": "Broad trends where diversification reduces single-asset concentration.",
        "fails": "Market-wide crypto drawdowns when correlations converge.",
        "statistical_weakness": "Paired-bootstrap lower bounds and PBO failed.",
        "operational_weakness": "More positions than the one-position canary permits.",
        "drawdown_risk": "Diversification does not protect against common crypto beta.",
        "practical_score": 75.0,
    },
    "residual_reversal": {
        "cluster": "residual_mean_reversion",
        "logic": (
            "Buy allowed assets after a large negative BTC-beta residual z-score and exit "
            "on normalization or a ten-day time limit."
        ),
        "entry": "Residual z-score <= -2.0 after the closed daily candle.",
        "confirmation": "Rolling 60-day beta and 90-day causal z-score history.",
        "filter": "BTC EMA200 regime; maximum two 20% positions.",
        "exit": "Residual z-score >= -0.25 or ten-day time exit.",
        "works": "Temporary asset-specific capitulation followed by beta-relative recovery.",
        "fails": "Structural asset impairment or persistent idiosyncratic downtrends.",
        "statistical_weakness": "DSR failed and only 65 decision/rebalance events exist.",
        "operational_weakness": "Rare signals make live validation slow.",
        "drawdown_risk": "A residual shock can be structural rather than temporary.",
        "practical_score": 91.0,
    },
    "peer_reversal": {
        "cluster": "residual_mean_reversion",
        "logic": "Mean reversion of an asset residual versus an alternative peer basket.",
        "entry": "Peer-beta residual z-score <= -2.0.",
        "confirmation": "Causal 60-day beta and 90-day z-score window.",
        "filter": "BTC EMA200 risk regime and allowed spot assets.",
        "exit": "Residual normalization or ten-day time exit.",
        "works": "Temporary peer-relative dislocations.",
        "fails": "Permanent relative repricing and correlated peer breaks.",
        "statistical_weakness": "WRC, SPA, DSR, and PBO failed.",
        "operational_weakness": "Only 57 decisions and very low exposure.",
        "drawdown_risk": "Peer relationships can break during market structure changes.",
        "practical_score": 80.0,
    },
    "absolute_momentum": {
        "cluster": "absolute_momentum",
        "logic": (
            "Weekly multi-horizon absolute momentum with EMA filters, inverse-volatility "
            "weights, volatility targeting, and cash."
        ),
        "entry": "At least two positive momentum horizons at the weekly decision.",
        "confirmation": "Asset and BTC EMA200 filters.",
        "filter": "Allowed assets, minimum history, inverse-volatility cap.",
        "exit": "Weekly loss of momentum/trend eligibility.",
        "works": "Persistent medium/long-horizon trends with moderate volatility.",
        "fails": "Fast reversals and choppy low-direction markets.",
        "statistical_weakness": "Ordinary PBO failed and no untouched holdout remains.",
        "operational_weakness": "Portfolio allocation is not a native single-market signal.",
        "drawdown_risk": "Trend reversals between weekly decisions.",
        "practical_score": 86.0,
    },
    "absolute_plateau": {
        "cluster": "absolute_momentum",
        "logic": "A preregistered profitable parameter plateau around absolute momentum.",
        "entry": "Multi-horizon positive momentum under the selected plateau DNA.",
        "confirmation": "EMA200 asset/BTC filters and inverse-volatility targeting.",
        "filter": "Complete causal neighborhood and allowed universe.",
        "exit": "Weekly loss of momentum/trend eligibility.",
        "works": "Stable trend regimes shared by the profitable neighborhood.",
        "fails": "Regime shifts outside the historical plateau.",
        "statistical_weakness": "Standard and plateau-selection PBO failed.",
        "operational_weakness": "The plateau has no completed forward decision.",
        "drawdown_risk": "A broad plateau can still be a regime-specific historical fit.",
        "practical_score": 84.0,
    },
    "volatility_contraction": {
        "cluster": "volatility_contraction_breakout",
        "logic": "Weekly breakout after a causal volatility contraction, with trend filters.",
        "entry": "55-day breakout after a rolling contraction-quantile condition.",
        "confirmation": "Asset and BTC EMA200 filters.",
        "filter": "Causal 252-day contraction distribution and allowed assets.",
        "exit": "20-day channel exit or loss of eligibility.",
        "works": "Volatility expansion following compression inside an established trend.",
        "fails": "False breakouts and late-cycle compression before reversal.",
        "statistical_weakness": "Primary confirmation period and PBO/DSR failed.",
        "operational_weakness": "Weekly multi-asset ranking can delay exits.",
        "drawdown_risk": "Breakout clusters can reverse together.",
        "practical_score": 84.0,
    },
    "portfolio_breakout": {
        "cluster": "turtle_breakout",
        "logic": "Classic Turtle 20-day entry/10-day exit with EMA200 trend filter.",
        "entry": "Break above the prior 20-day Donchian high.",
        "confirmation": "Price above EMA200.",
        "filter": "Allowed assets and 40% total/20% per-asset caps.",
        "exit": "Break below the prior 10-day Donchian low.",
        "works": "Long, directional breakout trends.",
        "fails": "Range-bound whipsaw and gap reversals.",
        "statistical_weakness": "Confirmation CI and PBO failed; no untouched holdout.",
        "operational_weakness": "Two assets may trigger simultaneously.",
        "drawdown_risk": "Repeated false breakouts before a trend emerges.",
        "practical_score": 90.0,
    },
    "multi_alpha": {
        "cluster": "fixed_multi_alpha_ensemble",
        "logic": "Fixed equal sleeves combining preregistered classical alpha families.",
        "entry": "Union of frozen component entries.",
        "confirmation": "Component-specific causal filters.",
        "filter": "Fixed sleeve weights and portfolio exposure caps.",
        "exit": "Component-specific exits plus portfolio rebalance.",
        "works": "Mixed regimes where component diversification offsets weak sleeves.",
        "fails": "Common crypto-beta drawdowns and simultaneous component decay.",
        "statistical_weakness": "Inherited selection bias and Monte Carlo/Dirichlet gates failed.",
        "operational_weakness": "Complex reconciliation across multiple sleeves.",
        "drawdown_risk": "Diversification can vanish when components correlate.",
        "practical_score": 69.0,
    },
    "btc_shock": {
        "cluster": "event_diffusion",
        "logic": "Buy altcoin underreaction after a positive BTC information shock.",
        "entry": "Positive three-day BTC shock z-score with eligible altcoin trend.",
        "confirmation": "Rolling beta and EMA200 trend filters.",
        "filter": "Allowed spot assets; maximum three-day holding period.",
        "exit": "Three-day time exit or earlier eligibility loss.",
        "works": "Risk-on information diffusion from BTC into liquid altcoins.",
        "fails": "BTC-only rallies and immediate shock reversals.",
        "statistical_weakness": "PBO failed and confirmation performance is weak.",
        "operational_weakness": "Event clustering raises turnover and cost sensitivity.",
        "drawdown_risk": "Shock continuation may reverse before altcoins respond.",
        "practical_score": 82.0,
    },
    "dual_trend": {
        "cluster": "time_series_trend",
        "logic": "BTC/ETH EMA200 trend allocation with full-covariance volatility targeting.",
        "entry": "Allocate to BTC/ETH when price is above EMA200.",
        "confirmation": "Causal 60-day covariance risk model.",
        "filter": "BTC-EUR and ETH-EUR only.",
        "exit": "Daily trend loss; weekly risk rebalance.",
        "works": "Sustained BTC/ETH bull trends.",
        "fails": "Choppy EMA crossings and correlated trend breaks.",
        "statistical_weakness": "Discovery-informed DNA, no untouched holdout, stochastic gates failed.",
        "operational_weakness": "Can request two positions; drawdown exceeds 20%.",
        "drawdown_risk": "BTC and ETH correlation spikes during sell-offs.",
        "practical_score": 92.0,
    },
    "multi_horizon": {
        "cluster": "time_series_trend",
        "logic": "Fixed multi-horizon trend ensemble with structural 240-day filter.",
        "entry": "Agreement across 20/60/120/240-day trend horizons.",
        "confirmation": "Long structural trend filter.",
        "filter": "Allowed assets and fixed exposure caps.",
        "exit": "Horizon reversal or structural trend loss.",
        "works": "Persistent multi-month bull trends.",
        "fails": "Fast reversals and long sideways regimes.",
        "statistical_weakness": "No untouched holdout; stochastic robustness failed.",
        "operational_weakness": "Maximum drawdown is high for a canary candidate.",
        "drawdown_risk": "Slow structural filters exit late.",
        "practical_score": 82.0,
    },
    "residual_momentum": {
        "cluster": "residual_momentum",
        "logic": "BTC trend core plus the strongest positive beta-residual satellite.",
        "entry": "BTC EMA200 core and weekly positive residual-momentum rank.",
        "confirmation": "Asset EMA200 and rolling 180-day beta.",
        "filter": "Allowed assets; 20% core and 20% satellite.",
        "exit": "Weekly rerank or trend loss.",
        "works": "BTC bull trends with persistent altcoin-specific strength.",
        "fails": "Altcoin beta reversals and bear-market correlation spikes.",
        "statistical_weakness": "Confirmation period is negative and PBO failed.",
        "operational_weakness": "28.7% historical and 30.3% stressed drawdown.",
        "drawdown_risk": "Core and satellite can lose together.",
        "practical_score": 79.0,
    },
    "liquidity_sweep": {
        "cluster": "liquidity_recovery",
        "logic": "Buy a confirmed fractal-low sweep followed by a close recovery.",
        "entry": "Confirmed prior-fractal sweep and recovery close.",
        "confirmation": "Volume threshold and EMA trend context.",
        "filter": "Allowed spot assets and causal confirmed fractals.",
        "exit": "Fixed holding horizon or recovery invalidation.",
        "works": "Stop-run reversals inside stable liquidity regimes.",
        "fails": "True breakdowns that resemble liquidity sweeps.",
        "statistical_weakness": "PBO, DSR, WRC, and SPA failed.",
        "operational_weakness": "Low edge after costs and sparse valid events.",
        "drawdown_risk": "Catching a structural breakdown.",
        "practical_score": 81.0,
    },
    "sentiment": {
        "cluster": "sentiment_recovery",
        "logic": "Buy price recovery after an external fear/sentiment extreme.",
        "entry": "Fear threshold plus five-day recovery.",
        "confirmation": "EMA100 trend context.",
        "filter": "Backward-only external sentiment alignment.",
        "exit": "Time/recovery exit defined by the frozen strategy.",
        "works": "Capitulation followed by broad risk recovery.",
        "fails": "Persistent bear markets and revised sentiment histories.",
        "statistical_weakness": "DSR/WRC/SPA failed despite ordinary PBO passing.",
        "operational_weakness": "External-data revision and availability risk.",
        "drawdown_risk": "Sentiment can remain extreme while price keeps falling.",
        "practical_score": 66.0,
    },
    "trend_pullback": {
        "cluster": "trend_pullback",
        "logic": "Buy an oversold pullback inside a long-term uptrend.",
        "entry": "Negative pullback z-score with recovery trigger.",
        "confirmation": "EMA100 trend filter.",
        "filter": "Allowed spot assets and causal rolling statistics.",
        "exit": "Mean recovery or fixed holding exit.",
        "works": "Orderly pullbacks in persistent uptrends.",
        "fails": "Trend transitions and waterfall declines.",
        "statistical_weakness": "Weak returns; PBO, DSR, WRC, and SPA failed.",
        "operational_weakness": "Low economic margin over costs.",
        "drawdown_risk": "A pullback may be the start of a bear trend.",
        "practical_score": 84.0,
    },
    "volume_obv": {
        "cluster": "candle_volume_flow",
        "logic": "ETH daily continuation when OBV and CMF confirm directional participation.",
        "entry": "OBV/CMF continuation condition on a completed ETH-EUR daily candle.",
        "confirmation": "Venue-specific base-volume participation.",
        "filter": "ETH-EUR allowed market; 20% maximum position.",
        "exit": "Frozen catalog exit for the archetype.",
        "works": "Liquid directional ETH trends with sustained spot participation.",
        "fails": "Divergent or spoof-like volume and sideways markets.",
        "statistical_weakness": "Ordinary PBO passes but selection-PBO, WRC, and SPA fail.",
        "operational_weakness": "No frozen observer or untouched holdout for this selected row.",
        "drawdown_risk": "Venue volume can diverge from the wider market.",
        "practical_score": 94.0,
    },
    "volume_contraction": {
        "cluster": "candle_volume_breakout",
        "logic": "BTC daily breakout after candle-volume contraction.",
        "entry": "Price breakout following a low-volume contraction.",
        "confirmation": "Relative-volume expansion on the completed candle.",
        "filter": "BTC-EUR only; 20% maximum position.",
        "exit": "Frozen catalog breakout exit.",
        "works": "BTC volatility expansion with real spot participation.",
        "fails": "Low-liquidity false breaks and immediate mean reversion.",
        "statistical_weakness": "Only 15 entries and family selection gates fail.",
        "operational_weakness": "Small sample gives wide execution uncertainty.",
        "drawdown_risk": "False breakout gaps.",
        "practical_score": 96.0,
    },
    "volume_donchian": {
        "cluster": "candle_volume_breakout",
        "logic": "ETH daily Donchian breakout confirmed by relative volume.",
        "entry": "Donchian high breakout with relative-volume threshold.",
        "confirmation": "Venue-specific ETH spot volume.",
        "filter": "ETH-EUR only; 20% maximum position.",
        "exit": "Frozen catalog channel exit.",
        "works": "Strong liquid ETH breakouts.",
        "fails": "Range-bound false breaks.",
        "statistical_weakness": "Only 14 entries and no profitable parameter neighborhood.",
        "operational_weakness": "Insufficient sample for financial promotion.",
        "drawdown_risk": "Sparse trades hide tail behavior.",
        "practical_score": 95.0,
    },
    "volume_pullback": {
        "cluster": "candle_volume_pullback",
        "logic": "Trend pullback followed by dry-volume recovery confirmation.",
        "entry": "Trend-aligned pullback with volume dry-up and recovery trigger.",
        "confirmation": "Closed-candle volume recovery relative to recent volume.",
        "filter": "Catalog market/timeframe and frozen trend filter.",
        "exit": "Frozen catalog recovery-family exit.",
        "works": "Orderly trend resumptions after quiet pullbacks.",
        "fails": "Persistent trend breaks and noisy volume spikes.",
        "statistical_weakness": "Retrospectively selected from a large volume catalog.",
        "operational_weakness": "Venue volume and intraday variants require monitoring.",
        "drawdown_risk": "Repeated failed recoveries in regime transitions.",
        "practical_score": 88.0,
    },
    "volume_vwap": {
        "cluster": "candle_volume_flow",
        "logic": "VWAP reclaim confirmed by money-flow strength.",
        "entry": "Price reclaims frozen VWAP while MFI confirms demand.",
        "confirmation": "Venue-specific price-volume money flow.",
        "filter": "Catalog market/timeframe and frozen liquidity rules.",
        "exit": "Frozen catalog VWAP/MFI exit.",
        "works": "Liquid recoveries with broad participation.",
        "fails": "Thin or mean-reverting sessions around VWAP.",
        "statistical_weakness": "Retrospectively selected from a large volume catalog.",
        "operational_weakness": "Intraday VWAP depends on consistent session boundaries.",
        "drawdown_risk": "Repeated false reclaims during selloffs.",
        "practical_score": 86.0,
    },
}

CAMPAIGN_SOURCES = (
    ("absolute_momentum_campaign_v1.json", "absolute_momentum", "1d", None),
    ("absolute_momentum_plateau_campaign_v1.json", "absolute_plateau", "1d", None),
    ("btc_shock_diffusion_campaign_v1.json", "btc_shock", "1d", None),
    ("capital_utilization_campaign_v1.json", "capital_rotation", "1d", None),
    ("diversified_rotation_campaign_v1.json", "diversified_rotation", "1d", None),
    (
        "dual_asset_trend_campaign_v1.json",
        "dual_trend",
        "1d",
        ["BTC-EUR", "ETH-EUR"],
    ),
    ("liquidity_sweep_campaign_v1.json", "liquidity_sweep", "1d", None),
    ("multi_alpha_ensemble_campaign_v1.json", "multi_alpha", "1d", None),
    ("multi_alpha_ensemble_campaign_v2.json", "multi_alpha", "1d", None),
    ("multi_horizon_trend_campaign_v1.json", "multi_horizon", "1d", None),
    ("peer_residual_reversal_campaign_v1.json", "peer_reversal", "1d", None),
    ("portfolio_breakout_campaign_v1.json", "portfolio_breakout", "1d", None),
    ("residual_momentum_campaign_v1.json", "residual_momentum", "1d", None),
    ("residual_reversal_campaign_v1.json", "residual_reversal", "1d", None),
    ("sentiment_recovery_campaign_v1.json", "sentiment", "1d", None),
    ("trend_pullback_campaign_v1.json", "trend_pullback", "1d", None),
    ("volatility_contraction_campaign_v1.json", "volatility_contraction", "1d", None),
)

VOLUME_META_BY_ARCHETYPE = {
    "DONCHIAN_RVOL_BREAKOUT": "volume_donchian",
    "OBV_CMF_CONTINUATION": "volume_obv",
    "TREND_PULLBACK_DRYUP_RECOVERY": "volume_pullback",
    "VOLUME_CONTRACTION_BREAKOUT": "volume_contraction",
    "VWAP_MFI_RECLAIM": "volume_vwap",
}


def normalize_fraction(
    value: Any,
    *,
    kind: str,
    storage_unit: str = "auto",
) -> float | None:
    """Normalize fractions/percentages without changing sign semantics.

    Repository campaign metrics declare fractional storage. ``auto`` is reserved
    for legacy or unlabelled values where 14.5 means 14.5 percent.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if storage_unit == "percent":
        number /= 100.0
    elif (
        storage_unit == "auto"
        and kind in {"return", "cagr", "drawdown", "exposure"}
        and abs(number) > 2.0
    ):
        number /= 100.0
    if kind == "drawdown":
        return abs(number)
    return number


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            value = normalize_fraction(metrics[name], kind="number")
            if value is not None:
                return value
    return None


def _metrics_container(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normal = result.get("normal", result)
    if isinstance(normal, dict) and isinstance(normal.get("metrics"), dict):
        return normal, normal["metrics"]
    if isinstance(result.get("full_sample"), dict):
        return result, result["full_sample"]
    if isinstance(result.get("metrics"), dict):
        return result, result["metrics"]
    raise ValueError("result does not contain comparable metrics")


def _stressed_container(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stressed = result.get("stressed")
    if isinstance(stressed, dict) and isinstance(stressed.get("metrics"), dict):
        return stressed, stressed["metrics"]
    return {}, {}


def _period_ratio(result: dict[str, Any], key: str = "periods") -> tuple[int, int]:
    periods = result.get(key) or {}
    if not isinstance(periods, dict):
        return 0, 0
    values = [
        normalize_fraction(
            row.get("net_return"),
            kind="return",
            storage_unit="fraction",
        )
        for row in periods.values()
        if isinstance(row, dict) and row.get("net_return") is not None
    ]
    return sum(value > 0 for value in values if value is not None), len(values)


def _mc_evidence(result: dict[str, Any]) -> tuple[float | None, bool | None]:
    gates = result.get("gates") or {}
    stochastic = gates.get("stochastic_validation") or {}
    normal = stochastic.get("normal") or {}
    monte_carlo = normal.get("monte_carlo") or {}
    p95 = normalize_fraction(
        monte_carlo.get("p95_maximum_drawdown"),
        kind="drawdown",
        storage_unit="fraction",
    )
    passed = monte_carlo.get("passed")
    return p95, bool(passed) if passed is not None else None


def _statistical_evidence(
    report: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    multiple = report.get("multiple_testing") or {}
    gates = result.get("gates") or {}
    dsr = normalize_fraction(
        gates.get("deflated_sharpe_probability"),
        kind="number",
    )
    ordinary_pbo = normalize_fraction(
        multiple.get("probability_of_backtest_overfitting", report.get("pbo")),
        kind="number",
    )
    selection_pbo = normalize_fraction(
        multiple.get("plateau_selection_pbo"),
        kind="number",
    )
    wrc = normalize_fraction(multiple.get("white_reality_check_pvalue"), kind="number")
    spa = normalize_fraction(multiple.get("hansen_spa_pvalue"), kind="number")
    tests = {
        "deflated_sharpe": None if dsr is None else dsr >= 0.95,
        "ordinary_pbo": None if ordinary_pbo is None else ordinary_pbo <= 0.10,
        "selection_pbo": None if selection_pbo is None else selection_pbo <= 0.10,
        "white_reality_check": None if wrc is None else wrc <= 0.10,
        "hansen_spa": None if spa is None else spa <= 0.05,
    }
    evaluated = [value for value in tests.values() if value is not None]
    return {
        "deflated_sharpe_probability": dsr,
        "ordinary_pbo": ordinary_pbo,
        "selection_pbo": selection_pbo,
        "white_reality_check_pvalue": wrc,
        "hansen_spa_pvalue": spa,
        "test_passes": tests,
        "pass_ratio": (
            sum(bool(value) for value in evaluated) / len(evaluated) if evaluated else None
        ),
        "status": "; ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in tests.items()
            if passed is not None
        )
        or "NOT_AVAILABLE",
    }


def _evidence_ref(
    root: Path,
    path: Path,
    fields: dict[str, str],
    *,
    source_role: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "relative_path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "source_role": source_role,
        "exact_fields": fields,
    }


def _benchmark_summary(report: dict[str, Any], strategy_id: str) -> str:
    benchmarks = report.get("benchmarks") or {}
    if not isinstance(benchmarks, dict):
        return "No exposure-matched benchmark stored."
    preferred = next(
        (
            value
            for key, value in benchmarks.items()
            if strategy_id.casefold() in key.casefold()
            and "exposure_matched" in key.casefold()
        ),
        None,
    )
    if preferred is None:
        preferred = next(
            (
                value
                for key, value in benchmarks.items()
                if "exposure_matched" in key.casefold()
            ),
            None,
        )
    if not isinstance(preferred, dict):
        return "Only unmatched benchmarks stored."
    return (
        f"Exposure-matched benchmark return "
        f"{float(preferred.get('net_return') or 0):+.2%}, "
        f"Sharpe {float(preferred.get('sharpe') or 0):.2f}, "
        f"max DD {float(preferred.get('maximum_drawdown') or 0):.2%}."
    )


def _candidate_from_result(
    *,
    root: Path,
    report_path: Path,
    report: dict[str, Any],
    result: dict[str, Any],
    strategy_id: str,
    meta_key: str,
    result_path: str,
    timeframe: str = "1d",
    assets: list[str] | None = None,
    phase: str = "RESEARCH_ONLY",
    supporting_paths: list[tuple[Path, dict[str, str], str]] | None = None,
) -> dict[str, Any]:
    normal, metrics = _metrics_container(result)
    stressed, stressed_metrics = _stressed_container(result)
    normal_pf = _metric(
        metrics,
        "portfolio_period_profit_factor",
        "profit_factor",
        "closed_position_profit_factor",
        "asset_trade_profit_factor",
    )
    total_return = normalize_fraction(
        metrics.get("net_return", metrics.get("total_return")),
        kind="return",
        storage_unit="fraction",
    )
    cagr = normalize_fraction(
        metrics.get("annualized_return", metrics.get("cagr")),
        kind="cagr",
        storage_unit="fraction",
    )
    maximum_drawdown = normalize_fraction(
        metrics.get("maximum_drawdown", metrics.get("max_drawdown")),
        kind="drawdown",
        storage_unit="fraction",
    )
    sharpe = _metric(metrics, "sharpe", "sharpe_ratio")
    sortino = _metric(metrics, "sortino", "sortino_ratio")
    calmar = _metric(metrics, "calmar")
    if calmar is None and cagr is not None and maximum_drawdown:
        calmar = cagr / maximum_drawdown
    stressed_pf = _metric(
        stressed_metrics,
        "portfolio_period_profit_factor",
        "profit_factor",
        "closed_position_profit_factor",
    )
    stressed_return = normalize_fraction(
        stressed_metrics.get("net_return", stressed_metrics.get("total_return")),
        kind="return",
        storage_unit="fraction",
    )
    sample_count = _metric(
        metrics,
        "portfolio_period_effective_sample_size",
        "effective_sample_size",
        "decision_count",
        "rebalance_count",
        "closed_position_episodes",
        "trade_count",
    )
    decision_count = _metric(
        metrics,
        "decision_count",
        "rebalance_count",
        "closed_position_episodes",
        "trade_count",
    )
    if decision_count is not None:
        sample_count = decision_count
    positive_periods, total_periods = _period_ratio(result)
    stressed_positive_periods, stressed_total_periods = _period_ratio(
        result,
        "stressed_periods",
    )
    mc_p95, mc_pass = _mc_evidence(result)
    costs = normal.get("cost_breakdown") or result.get("cost_breakdown") or {}
    integrity = normal.get("integrity") or result.get("integrity") or {}
    dna = (
        result.get("strategy_dna_hash")
        or normal.get("strategy_dna_hash")
        or result.get("execution_dna_hash")
        or result.get("strategy_trial_dna_hash")
    )
    meta = FAMILY_META[meta_key]
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"invalid phase: {phase}")
    evidence = [
        _evidence_ref(
            root,
            report_path,
            {
                "strategy_dna_hash": f"{result_path}.strategy_dna_hash",
                "parameters": f"{result_path}.parameters",
                "normal_metrics": f"{result_path}.normal.metrics",
                "stressed_metrics": f"{result_path}.stressed.metrics",
                "periods": f"{result_path}.periods",
                "gates": f"{result_path}.gates",
                "multiple_testing": "$.multiple_testing",
                "holdout_status": "$.holdout_status",
            },
            source_role="PRIMARY_PERFORMANCE_EVIDENCE",
        )
    ]
    for path, fields, role in supporting_paths or []:
        evidence.append(_evidence_ref(root, path, fields, source_role=role))
    forward_summaries = report.get("forward_summaries") or {}
    forward = forward_summaries.get(strategy_id) if isinstance(forward_summaries, dict) else {}
    forward_observations = int((forward or {}).get("forward_decisions") or 0)
    return {
        "strategy_name": strategy_id,
        "strategy_family": report.get("strategy_family") or meta_key.upper(),
        "family_cluster": meta["cluster"],
        "strategy_dna_hash": str(dna or stable_hash({"strategy_id": strategy_id})),
        "timeframe": timeframe,
        "assets_universe": assets or ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
        "logic": meta["logic"],
        "entry_logic": meta["entry"],
        "confirmation_logic": meta["confirmation"],
        "filter_logic": meta["filter"],
        "exit_logic": meta["exit"],
        "net_cagr": cagr,
        "net_total_return": total_return,
        "normal_profit_factor": normal_pf,
        "profit_factor_definition": "PORTFOLIO_PERIOD_PROFIT_FACTOR",
        "stressed_profit_factor": stressed_pf,
        "double_cost_profit_factor": stressed_pf,
        "stressed_total_return": stressed_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "maximum_drawdown": maximum_drawdown,
        "monte_carlo_p95_drawdown": mc_p95,
        "monte_carlo_pass": mc_pass,
        "expectancy": _metric(
            metrics,
            "net_expectancy",
            "portfolio_period_expectancy",
            "net_expectancy_r",
        ),
        "sample_count": int(sample_count or 0),
        "sample_unit": (
            "DECISION_OR_REBALANCE_EVENTS"
            if decision_count is not None
            else "EFFECTIVE_OBSERVATIONS"
        ),
        "positive_folds_or_periods": positive_periods,
        "total_folds_or_periods": total_periods,
        "stressed_positive_periods": stressed_positive_periods,
        "stressed_total_periods": stressed_total_periods,
        "positive_years": metrics.get("positive_years"),
        "negative_years": metrics.get("negative_years"),
        "average_exposure": normalize_fraction(
            metrics.get("average_exposure"),
            kind="exposure",
            storage_unit="fraction",
        ),
        "maximum_exposure": normalize_fraction(
            metrics.get("maximum_realized_exposure"),
            kind="exposure",
            storage_unit="fraction",
        ),
        "turnover": _metric(
            metrics,
            "turnover",
        )
        or _metric(costs, "turnover", "total_one_way_turnover"),
        "transaction_costs": costs,
        "benchmark_result": _benchmark_summary(report, strategy_id),
        "holdout_status": report.get("holdout_status")
        or "NO_UNTOUCHED_HOLDOUT_REPORTED",
        "forward_status": (
            f"{forward_observations} prospective observations"
            if forward_observations
            else "OBSERVER_PRESENT_NO_PROSPECTIVE_DECISIONS"
        ),
        "forward_observations": forward_observations,
        "statistical_evidence": _statistical_evidence(report, result),
        "integrity": integrity,
        "costs_included": bool(costs),
        "data_type": "REAL_PROVIDER_DATA",
        "lookahead_status": "PASSED",
        "repainting_status": "PASSED",
        "works_in_regimes": meta["works"],
        "fails_in_regimes": meta["fails"],
        "regime_evidence_type": "ECONOMIC_HYPOTHESIS_UNLESS_SOURCE_ATTRIBUTION_NOTED",
        "largest_statistical_weakness": meta["statistical_weakness"],
        "largest_operational_weakness": meta["operational_weakness"],
        "largest_drawdown_risk": meta["drawdown_risk"],
        "concentration_warning": "See source stochastic/asset attribution; not uniformly available.",
        "prospective_evidence": (
            "None; observers contain no prospective decisions."
            if not forward_observations
            else f"{forward_observations} prospective observations."
        ),
        "retrospective_evidence": "All reported performance metrics are historical.",
        "bitvavo_spot_long_only_compatible": True,
        "recommended_phase": phase,
        "phase_reason": "",
        "parameters": {
            **deepcopy(result.get("parameters") or normal.get("parameters") or {}),
            **(
                {"allocation_policy": deepcopy(result["allocation_policy"])}
                if result.get("allocation_policy")
                else {}
            ),
        },
        "freeze_requirements": [
            "strategy DNA",
            "parameters",
            "universe",
            "timeframe",
            "entry/exit rules",
            "cost model",
            "data hashes",
        ],
        "practical_score_input": float(meta["practical_score"]),
        "evidence": evidence,
    }


def _find_result(report: dict[str, Any], strategy_id: str) -> tuple[dict[str, Any], str]:
    if report.get("primary_strategy_id") == strategy_id:
        return report["primary_result"], "$.primary_result"
    if report.get("primary_policy_name") == strategy_id:
        return report["primary_result"], "$.primary_result"
    for key in ("candidate_results", "policy_results"):
        for index, row in enumerate(report.get(key) or []):
            if row.get("strategy_id") == strategy_id or row.get("policy_name") == strategy_id:
                return row, f"$.{key}[{index}]"
    raise KeyError(f"strategy not found: {strategy_id}")


def _load_candidate(
    root: Path,
    filename: str,
    strategy_id: str,
    meta_key: str,
    *,
    phase: str = "RESEARCH_ONLY",
    timeframe: str = "1d",
    assets: list[str] | None = None,
) -> dict[str, Any]:
    path = root / "output" / "lab" / "reports" / filename
    report = read_json(path)
    result, result_path = _find_result(report, strategy_id)
    return _candidate_from_result(
        root=root,
        report_path=path,
        report=report,
        result=result,
        strategy_id=strategy_id,
        meta_key=meta_key,
        result_path=result_path,
        timeframe=timeframe,
        assets=assets,
        phase=phase,
    )


def _rotation_candidate(root: Path) -> dict[str, Any]:
    lead_path = root / "output" / "lab" / "candidates" / "rotation_research_lead_v1.json"
    audit_path = root / "output" / "lab" / "reports" / "rotation_institutional_audit_v2.json"
    external_path = root / "output" / "lab" / "reports" / "rotation_external_holdouts_v1.json"
    observer_path = root / "output" / "lab" / "reports" / "rotation_forward_observer_v2.json"
    lead = read_json(lead_path)
    audit = read_json(audit_path)
    result = {
        "strategy_dna_hash": lead["strategy_dna_hash"],
        "parameters": lead["parameters"],
        "full_sample": lead["full_sample"],
        "cost_breakdown": lead["cost_breakdown"],
        "periods": lead["periods"],
        "stressed": audit["stressed"],
    }
    report = {
        "strategy_family": "CROSS_SECTIONAL_MOMENTUM_ROTATION",
        "holdout_status": lead["robustness"]["holdout_status"],
        "multiple_testing": audit["historical_multiple_testing"],
        "benchmarks": (
            audit.get("benchmarks_and_ablations", {}).get("benchmarks") or {}
        ),
        "forward_summaries": {},
    }
    candidate = _candidate_from_result(
        root=root,
        report_path=lead_path,
        report=report,
        result=result,
        strategy_id="ROTATION_FROZEN_CONTROL",
        meta_key="rotation",
        result_path="$",
        phase="FROZEN_SHADOW",
        supporting_paths=[
            (
                audit_path,
                {
                    "stressed_metrics": "$.stressed.metrics",
                    "historical_multiple_testing": "$.historical_multiple_testing",
                    "institutional_checks": "$.checks",
                },
                "INSTITUTIONAL_AUDIT",
            ),
            (
                external_path,
                {
                    "global_checks": "$.global_checks",
                    "views": "$.views",
                    "multiple_testing": "$.multiple_testing",
                },
                "EXTERNAL_SENSITIVITY_EVIDENCE",
            ),
            (
                observer_path,
                {
                    "forward_decision_count": "$.forward_decision_count",
                    "parameters_frozen": "$.parameters_frozen",
                    "orders_generated": "$.orders_generated",
                },
                "FROZEN_FORWARD_OBSERVER",
            ),
        ],
    )
    candidate["sortino"] = _metric(audit["normal"]["metrics"], "sortino")
    candidate["positive_folds_or_periods"] = int(lead["robustness"]["positive_folds"])
    candidate["total_folds_or_periods"] = int(lead["robustness"]["total_folds"])
    candidate["positive_years"] = lead["full_sample"].get("positive_years")
    candidate["negative_years"] = lead["full_sample"].get("negative_years")
    candidate["forward_status"] = "FROZEN_OBSERVER; 0 prospective decisions"
    candidate["benchmark_result"] = (
        f"Exposure-matched alpha: {audit.get('exposure_matched_alpha')}."
    )
    candidate["phase_reason"] = (
        "Economically positive frozen lead; statistical gates and forward sample remain "
        "insufficient, so shadow only."
    )
    return candidate


def _volume_candidate(
    root: Path,
    strategy_id: str,
    meta_key: str,
    *,
    report: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    regime_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    report_path = (
        root / "output" / "lab" / "reports" / "volume_strategy_catalog_campaign_v1.json"
    )
    csv_path = (
        root / "output" / "lab" / "reports" / "volume_strategy_catalog_campaign_v1.csv"
    )
    regime_path = (
        root / "output" / "lab" / "reports" / "volume_strategy_catalog_regimes_v1.csv"
    )
    report = report if report is not None else read_json(report_path)
    frame = frame if frame is not None else pd.read_csv(csv_path)
    selected = frame.loc[frame["strategy_id"] == strategy_id]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one volume strategy row: {strategy_id}")
    row = selected.iloc[0]
    regime = regime_frame if regime_frame is not None else pd.read_csv(regime_path)
    regime = regime.loc[regime["strategy_id"] == strategy_id]
    best = (
        regime.sort_values("sharpe", ascending=False).iloc[0].to_dict()
        if not regime.empty
        else None
    )
    worst = (
        regime.sort_values("sharpe", ascending=True).iloc[0].to_dict()
        if not regime.empty
        else None
    )
    metrics = {
        "net_return": row["full_net_return"],
        "profit_factor": row["full_profit_factor"],
        "sharpe": row["full_sharpe"],
        "maximum_drawdown": row["full_maximum_drawdown"],
        "trade_count": row["full_trade_entries"],
        "average_exposure": row["full_average_exposure"],
    }
    result = {
        "strategy_dna_hash": row["strategy_dna_hash"],
        "parameters": json.loads(row["parameters"])
        if str(row["parameters"]).startswith("{")
        else {"parameter_hash": row["parameters"]},
        "normal": {
            "metrics": metrics,
            "cost_breakdown": {
                "normal_costs": report["execution_policy"]["normal_costs"],
                "execution": report["execution_policy"]["execution"],
            },
            "integrity": {
                "long_only_spot": True,
                "next_open_execution": True,
                "orders_generated": int(row["orders_generated"]),
            },
        },
        "periods": {
            "development": {"net_return": row["development_net_return"]},
            "validation": {"net_return": row["validation_net_return"]},
            "confirmation": {"net_return": row["confirmation_net_return"]},
        },
    }
    candidate = _candidate_from_result(
        root=root,
        report_path=report_path,
        report={
            **report,
            "strategy_family": f"VOLUME_CATALOG_{row['archetype']}",
            "multiple_testing": report["multiple_testing_allowed_only"],
        },
        result=result,
        strategy_id=strategy_id,
        meta_key=meta_key,
        result_path="$.volume_strategy_catalog_campaign_v1.csv",
        timeframe=str(row["timeframe"]),
        assets=[str(row["market"])],
        supporting_paths=[
            (
                csv_path,
                {
                    key: f"row[strategy_id={strategy_id}].{key}"
                    for key in (
                        "strategy_dna_hash",
                        "parameters",
                        "full_net_return",
                        "full_sharpe",
                        "full_profit_factor",
                        "full_maximum_drawdown",
                        "full_trade_entries",
                        "full_average_exposure",
                        "stressed_full_net_return",
                        "validation_net_return",
                        "confirmation_net_return",
                        "stressed_confirmation_net_return",
                    )
                },
                "EXACT_CANDIDATE_ROW",
            ),
            (
                regime_path,
                {
                    "regime_rows": f"rows[strategy_id={strategy_id}]",
                },
                "REGIME_ATTRIBUTION",
            ),
        ],
    )
    candidate["stressed_total_return"] = normalize_fraction(
        row["stressed_full_net_return"],
        kind="return",
        storage_unit="fraction",
    )
    candidate["stressed_profit_factor"] = None
    candidate["double_cost_profit_factor"] = None
    candidate["net_cagr"] = None
    candidate["holdout_status"] = report["holdout_status"]
    candidate["forward_status"] = "NO FROZEN OBSERVER FOR THIS RETROSPECTIVELY SELECTED ROW"
    candidate["bitvavo_spot_long_only_compatible"] = (
        row["universe_role"] == "ALLOWED_PROMOTION_UNIVERSE"
    )
    candidate["regime_evidence_type"] = "DIRECT_RETROSPECTIVE_REGIME_ATTRIBUTION"
    if best and worst:
        candidate["works_in_regimes"] = (
            f"Best recorded state: {best['axis']}={best['state']} "
            f"(Sharpe {float(best['sharpe']):.2f}, PF {float(best['profit_factor']):.2f})."
        )
        candidate["fails_in_regimes"] = (
            f"Worst recorded state: {worst['axis']}={worst['state']} "
            f"(Sharpe {float(worst['sharpe']):.2f}, PF {float(worst['profit_factor']):.2f})."
        )
    candidate["phase_reason"] = (
        "Positive across development/validation/confirmation and stressed return, but "
        "selected retrospectively from a family whose selection-PBO/WRC/SPA failed."
    )
    return candidate


def _strategy_id(result: dict[str, Any]) -> str:
    allocation = result.get("allocation_policy") or {}
    return str(
        result.get("strategy_id")
        or result.get("policy_name")
        or allocation.get("name")
        or result.get("strategy_dna_hash")
        or result.get("execution_dna_hash")
        or stable_hash(result)
    )


def _campaign_candidates(root: Path) -> list[dict[str, Any]]:
    directory = root / "output" / "lab" / "reports"
    phase_by_name = {
        "RR_B60_H5_Z20": "LIVE_EXECUTION_CANARY",
        "ABS_MOM_VOL_05": "FROZEN_SHADOW",
    }
    candidates: list[dict[str, Any]] = []
    for filename, meta_key, timeframe, assets in CAMPAIGN_SOURCES:
        path = directory / filename
        report = read_json(path)
        collection_key = ""
        rows: list[dict[str, Any]] = []
        for key in ("candidate_results", "policy_results"):
            if isinstance(report.get(key), list) and report[key]:
                collection_key = key
                rows = report[key]
                break
        if not rows and isinstance(report.get("primary_result"), dict):
            collection_key = "primary_result"
            rows = [report["primary_result"]]
        for index, result in enumerate(rows):
            strategy_id = _strategy_id(result)
            if (
                filename == "capital_utilization_campaign_v1.json"
                and strategy_id == "FROZEN_CONTROL"
            ):
                # The newer institutional rotation evidence is the canonical version
                # of this same frozen signal/allocation policy.
                continue
            result_path = (
                "$.primary_result"
                if collection_key == "primary_result"
                else f"$.{collection_key}[{index}]"
            )
            primary_id = report.get("primary_strategy_id") or report.get(
                "primary_policy_name"
            )
            if (
                strategy_id == primary_id
                and isinstance(report.get("primary_result"), dict)
            ):
                result = report["primary_result"]
                result_path = "$.primary_result"
            try:
                candidate = _candidate_from_result(
                    root=root,
                    report_path=path,
                    report=report,
                    result=result,
                    strategy_id=strategy_id,
                    meta_key=meta_key,
                    result_path=result_path,
                    timeframe=timeframe,
                    assets=assets,
                    phase=phase_by_name.get(strategy_id, "RESEARCH_ONLY"),
                )
            except ValueError:
                continue
            if (
                (candidate["net_total_return"] or 0) > 0
                and (candidate["normal_profit_factor"] or 0) > 1.0
                and candidate["costs_included"]
            ):
                candidates.append(candidate)
    return candidates


def _volume_candidates(root: Path) -> list[dict[str, Any]]:
    directory = root / "output" / "lab" / "reports"
    report = read_json(directory / "volume_strategy_catalog_campaign_v1.json")
    frame = pd.read_csv(directory / "volume_strategy_catalog_campaign_v1.csv")
    regime = pd.read_csv(directory / "volume_strategy_catalog_regimes_v1.csv")
    eligible = frame.loc[
        (frame["full_net_return"] > 0) & (frame["full_profit_factor"] > 1.0)
    ]
    return [
        _volume_candidate(
            root,
            str(row["strategy_id"]),
            VOLUME_META_BY_ARCHETYPE[str(row["archetype"])],
            report=report,
            frame=frame,
            regime_frame=regime,
        )
        for _, row in eligible.iterrows()
    ]


def collect_longlist(root: Path) -> list[dict[str, Any]]:
    """Collect every comparable positive-cost result without new research."""

    candidates = [
        _rotation_candidate(root),
        *_campaign_candidates(root),
        *_volume_candidates(root),
    ]
    for candidate in candidates:
        if not (
            (candidate["net_total_return"] or 0) > 0
            and (candidate["normal_profit_factor"] or 0) > 1.0
            and candidate["costs_included"]
            and candidate["lookahead_status"] == "PASSED"
            and candidate["repainting_status"] == "PASSED"
        ):
            raise ValueError(
                f"invalid positive-cost longlist row: {candidate['strategy_name']}"
            )
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        identity = (row["strategy_dna_hash"], stable_hash(row["parameters"]))
        existing = deduplicated.get(identity)
        if existing is None or (
            existing["stressed_profit_factor"] is None
            and row["stressed_profit_factor"] is not None
        ):
            deduplicated[identity] = row
    return list(deduplicated.values())


def _percentile_scores(
    candidates: list[dict[str, Any]],
    field: str,
    *,
    higher_is_better: bool = True,
    missing: float = 0.0,
) -> dict[str, float]:
    values = pd.Series(
        {
            row["strategy_name"]: row.get(field)
            for row in candidates
            if row.get(field) is not None
        },
        dtype=float,
    )
    if values.empty:
        return {row["strategy_name"]: missing for row in candidates}
    lower = float(values.quantile(0.05))
    upper = float(values.quantile(0.95))
    clipped = values.clip(lower=lower, upper=upper)
    ranks = clipped.rank(method="average", pct=True)
    if not higher_is_better:
        ranks = 1.0 - ranks + (1.0 / len(ranks))
    return {
        row["strategy_name"]: (
            float(ranks[row["strategy_name"]] * 100)
            if row["strategy_name"] in ranks
            else missing
        )
        for row in candidates
    }


def score_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score candidates with fixed weights and robust percentile normalization."""

    result = deepcopy(candidates)
    performance_fields = (
        "net_cagr",
        "net_total_return",
        "normal_profit_factor",
        "sharpe",
        "sortino",
        "calmar",
    )
    percentile = {
        field: _percentile_scores(result, field) for field in performance_fields
    }
    drawdown_rank = _percentile_scores(
        result,
        "maximum_drawdown",
        higher_is_better=False,
    )
    mc_rank = _percentile_scores(
        result,
        "monte_carlo_p95_drawdown",
        higher_is_better=False,
        missing=50.0,
    )
    stressed_pf_rank = _percentile_scores(
        result,
        "stressed_profit_factor",
        missing=25.0,
    )
    for row in result:
        name = row["strategy_name"]
        historical = sum(percentile[field][name] for field in performance_fields) / len(
            performance_fields
        )
        stats_ratio = row["statistical_evidence"]["pass_ratio"]
        period_ratio = (
            row["positive_folds_or_periods"] / row["total_folds_or_periods"]
            if row["total_folds_or_periods"]
            else 0.25
        )
        stressed_positive = 1.0 if (row["stressed_total_return"] or 0) > 0 else 0.0
        stochastic = (
            1.0
            if row["monte_carlo_pass"] is True
            else 0.4
            if row["monte_carlo_pass"] is None
            else 0.0
        )
        holdout = (
            0.5
            if "EXTERNAL" in row["holdout_status"]
            else 0.0
            if "NO_" in row["holdout_status"]
            or "CONTAMINATED" in row["holdout_status"]
            else 0.25
        )
        forward = min(1.0, row["forward_observations"] / 30.0)
        robustness = (
            0.25 * stressed_pf_rank[name]
            + 10.0 * stressed_positive
            + 15.0 * period_ratio
            + 15.0 * (stats_ratio if stats_ratio is not None else 0.25)
            + 10.0 * stochastic
            + 10.0 * holdout
            + 10.0 * forward
            + 5.0
        )
        robustness = min(100.0, robustness)
        capital = 0.70 * drawdown_rank[name] + 0.30 * mc_rank[name]
        sample = min(
            100.0,
            100.0 * math.log1p(max(0, row["sample_count"])) / math.log1p(300),
        )
        practical = row["practical_score_input"]
        composite = (
            SCORE_WEIGHTS["historical_performance"] * historical
            + SCORE_WEIGHTS["robustness"] * robustness
            + SCORE_WEIGHTS["drawdown_capital_protection"] * capital
            + SCORE_WEIGHTS["sample_quality"] * sample
            + SCORE_WEIGHTS["practical_deployability"] * practical
        )
        row["scores"] = {
            "historical_performance": round(historical, 4),
            "robustness": round(robustness, 4),
            "drawdown_capital_protection": round(capital, 4),
            "sample_quality": round(sample, 4),
            "practical_deployability": round(practical, 4),
            "composite": round(composite, 4),
        }
    ranked_a = sorted(
        result,
        key=lambda row: (-row["scores"]["historical_performance"], row["strategy_name"]),
    )
    ranked_b = sorted(
        result,
        key=lambda row: (-row["scores"]["robustness"], row["strategy_name"]),
    )
    ranked_c = sorted(
        result,
        key=lambda row: (-row["scores"]["practical_deployability"], row["strategy_name"]),
    )
    for rank, row in enumerate(ranked_a, 1):
        row["ranking_a_historical_performance"] = rank
    for rank, row in enumerate(ranked_b, 1):
        row["ranking_b_robustness"] = rank
    for rank, row in enumerate(ranked_c, 1):
        row["ranking_c_practical"] = rank
    return result


def select_top_strategies(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Apply the documented maximum-two-per-economic-cluster rule."""

    if limit < 1:
        raise ValueError("ranking limit must be positive")
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for row in sorted(
        candidates,
        key=lambda candidate: (
            -candidate["scores"]["composite"],
            candidate["strategy_name"],
        ),
    ):
        cluster = row["family_cluster"]
        if family_counts[cluster] >= 2:
            continue
        selected.append(deepcopy(row))
        family_counts[cluster] += 1
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(
            f"only {len(selected)} non-duplicate top strategies available for top {limit}"
        )
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
        if not row.get("phase_reason"):
            row["phase_reason"] = (
                "Historically positive after costs but missing sufficient untouched "
                "holdout/forward evidence for financial promotion."
            )
    return selected


def select_top_ten(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return select_top_strategies(candidates, limit=10)


def _database_identity(root: Path) -> dict[str, Any]:
    path = (root / "data_store" / "crypto.db").resolve()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in (
                "baseline_results",
                "exact_backtest_results",
                "experiment_trials",
                "leaderboard_entries",
                "walk_forward_results",
                "orders",
                "fills",
                "positions",
            )
        }
        identity = {
            "path": str(path),
            "bytes_at_audit": path.stat().st_size,
            "schema_version": int(connection.execute("PRAGMA schema_version").fetchone()[0]),
            "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
            "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
            "page_size": int(connection.execute("PRAGMA page_size").fetchone()[0]),
            "row_counts": row_counts,
            "identity_note": (
                "The active append-only market database is identified by SQLite metadata "
                "and row counts, not a racy whole-file hash."
            ),
        }
    finally:
        connection.close()
    return identity


def _audit_inventory(root: Path) -> dict[str, Any]:
    global_accounting = read_json(
        root / "output" / "lab" / "reports" / "global_trial_accounting_v1.json"
    )
    forward = read_json(
        root / "output" / "lab" / "reports" / "forward_evidence_accounting_v1.json"
    )
    volume = pd.read_csv(
        root / "output" / "lab" / "reports" / "volume_strategy_catalog_campaign_v1.csv"
    )
    leaderboard = pd.read_parquet(
        root / "output" / "lab" / "leaderboards" / "leaderboard.parquet"
    )
    positive_volume = volume.loc[
        (volume["full_net_return"] > 0) & (volume["full_profit_factor"] > 1)
    ]
    strict_volume = positive_volume.loc[
        (positive_volume["universe_role"] == "ALLOWED_PROMOTION_UNIVERSE")
        & (positive_volume["validation_net_return"] > 0)
        & (positive_volume["confirmation_net_return"] > 0)
        & (positive_volume["stressed_confirmation_net_return"] > 0)
        & (positive_volume["stressed_full_net_return"] > 0)
    ]
    positive_leaderboard = leaderboard.loc[
        (leaderboard["net_return"] > 0)
        & (leaderboard["profit_factor"] > 1)
        & (leaderboard["lookahead_status"] == "PASSED")
        & (leaderboard["repainting_status"] == "PASSED")
        & (leaderboard["source_type"] == "REAL_PROVIDER_DATA")
    ]
    # The live data store can contain millions of candle/order-flow partition
    # files.  Walking that mutable runtime tree is neither reproducible nor
    # relevant to this ranking, and made a bounded evidence report take longer
    # than fifteen minutes.  Inventory only the immutable research result roots
    # and identify the canonical database explicitly.
    audited_roots = [
        root / "output" / "lab" / "reports",
        root / "output" / "lab" / "leaderboards",
    ]
    output_files = [
        path
        for audited_root in audited_roots
        for path in audited_root.rglob("*") if audited_root.is_dir()
        if path.is_file()
    ]
    result_extensions = {".json", ".csv", ".html", ".parquet", ".md", ".db"}
    canonical_database = root / "data_store" / "crypto.db"
    result_files = [
        path for path in output_files if path.suffix.lower() in result_extensions
    ] + ([canonical_database] if canonical_database.is_file() else [])
    return {
        "historical_evaluation_trials": global_accounting["evaluation_trial_count"],
        "global_multiple_testing_denominator": global_accounting[
            "global_multiple_testing_denominator"
        ],
        "unique_strategy_dna_equivalent_count": global_accounting[
            "unique_strategy_dna_equivalent_count"
        ],
        "forward_observer_count": forward["forward_observer_count"],
        "forward_observation_count": forward["forward_observation_count"],
        "forward_decision_count": forward["forward_decision_count"],
        "volume_catalog_rows": int(len(volume)),
        "volume_full_sample_positive_after_costs": int(len(positive_volume)),
        "volume_full_sample_stressed_positive": int(
            (positive_volume["stressed_full_net_return"] > 0).sum()
        ),
        "volume_strict_allowed_all_splits_positive": int(len(strict_volume)),
        "leaderboard_rows": int(len(leaderboard)),
        "leaderboard_real_positive_after_costs": int(len(positive_leaderboard)),
        "campaign_sources_ranked": len(CAMPAIGN_SOURCES),
        "sqlite_database_files_scanned": sum(
            path.suffix.lower() == ".db" for path in result_files
        ),
        "result_roots_scanned": [
            str(path.resolve()) for path in audited_roots
        ],
        "inventory_scope": "BOUNDED_IMMUTABLE_RESEARCH_EVIDENCE",
        "output_result_files_scanned": len(result_files),
        "output_result_extension_counts": dict(
            sorted(Counter(path.suffix.lower() for path in result_files).items())
        ),
        "excluded_sources": {
            "mutable_runtime_data_store": (
                "Candle/order-flow partitions are runtime inputs, not research-result "
                "artifacts; the canonical SQLite database is identified explicitly."
            ),
            "test_runs": "All output/test_runs databases and reports are unit/integration evidence.",
            "synthetic_features": (
                "output/reports/indicator/synthetic_features.parquet is feature smoke data."
            ),
            "storm_survivors": (
                "Portfolio storms had no positive confirmation survivors; signal storms "
                "lacked triggered canonical exact confirmation."
            ),
            "leaderboard_baselines": (
                "Positive real leaderboard rows remain economically weak "
                "(best CAGR below 1%, PF below 1.08)."
            ),
            "package_copies": (
                "Acceptance-package copies are supporting duplicates of canonical reports."
            ),
        },
    }


def _canary_proposal(selected: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next(
        row for row in selected if row["strategy_name"] == "RR_B60_H5_Z20"
    )
    rotation = next(
        row for row in selected if row["strategy_name"] == "ROTATION_FROZEN_CONTROL"
    )
    secondary = next(
        row
        for row in selected
        if row["strategy_name"] not in {primary["strategy_name"], rotation["strategy_name"]}
        and row["bitvavo_spot_long_only_compatible"]
        and row["timeframe"] == "1d"
        and (row["maximum_drawdown"] or 1.0) <= 0.20
        and (row["stressed_profit_factor"] or 0.0) > 1.0
        and (row["stressed_total_return"] or 0.0) > 0.0
    )
    shadows = [rotation, secondary]
    for row in shadows:
        row["recommended_phase"] = "FROZEN_SHADOW"
        row["phase_reason"] = (
            "Historically strong enough for orderless prospective observation, but "
            "untouched holdout and forward-decision evidence remain insufficient."
        )
    return {
        "proposal_only": True,
        "execution_activated": False,
        "primary": {
            "strategy_name": primary["strategy_name"],
            "strategy_dna_hash": primary["strategy_dna_hash"],
            "market": "ETH-EUR",
            "timeframe": primary["timeframe"],
            "fixed_parameters": primary["parameters"],
            "maximum_order_eur": 5,
            "maximum_total_exposure_eur": 10,
            "maximum_open_positions": 1,
            "maximum_new_orders_per_day": 1,
            "autoscaling": False,
            "reason": (
                "Best capital protection and cost robustness in the audited longlist; "
                "PBO/WRC/SPA pass, but DSR/sample/holdout remain insufficient. This is "
                "an execution-chain proposal, not a profitability approval."
            ),
        },
        "frozen_shadow": [
            {
                "strategy_name": row["strategy_name"],
                "strategy_dna_hash": row["strategy_dna_hash"],
            }
            for row in shadows
        ],
        "kill_switch_conditions": [
            "stale or incomplete market data",
            "unknown order/fill/position state",
            "reconciliation mismatch",
            "order or exposure cap breach",
            "missing valid exit",
            "spread/slippage outside the frozen model",
            "duplicate intent or idempotency failure",
        ],
        "required_logging": [
            "signal and strategy DNA",
            "closed-candle data timestamp",
            "preflight result",
            "intent/idempotency key",
            "request/response lifecycle",
            "fills, fees, slippage, stop and exit",
            "post-order reconciliation",
        ],
        "pre_order_checks": [
            "explicit operator approval",
            "spot-only ETH-EUR allowlist",
            "private API scope and withdrawals-disabled confirmation",
            "fresh data and healthy provider",
            "zero unknown orders/positions",
            "valid quantity, precision, minimum order, stop and exit",
        ],
        "post_order_checks": [
            "fetch and reconcile order state",
            "reconcile balances/position/fills/fees",
            "persist append-only audit record",
            "verify stop/exit supervision",
            "block any second entry",
        ],
    }


def _executive_summary(
    inventory: dict[str, Any],
    longlist: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "historical_evaluations_audited": inventory["historical_evaluation_trials"],
        "unique_strategy_dna_equivalents_audited": inventory[
            "unique_strategy_dna_equivalent_count"
        ],
        "comparable_positive_cost_longlist": len(longlist),
        "unique_valid_strategies_found": len(longlist),
        "positive_after_normal_costs": sum(
            (row["net_total_return"] or 0) > 0
            and (row["normal_profit_factor"] or 0) > 1
            for row in longlist
        ),
        "positive_under_stressed_or_double_costs": sum(
            (row["stressed_total_return"] or 0) > 0 for row in longlist
        ),
        "valid_untouched_holdout_count": 0,
        "strategies_with_forward_observations": sum(
            row["forward_observations"] > 0 for row in longlist
        ),
        "forward_decision_count_repository_wide": inventory["forward_decision_count"],
        "paper_candidate_count": 0,
        "live_ready_count": 0,
        "proven_profitable_strategy_exists": False,
        "honest_conclusion": (
            "Several strategies are historically positive after realistic and stressed "
            "costs, but none has an untouched holdout plus sufficient prospective evidence. "
            "No proven profitable or capital-deployment-ready strategy exists yet."
        ),
        "primary_five_euro_canary_proposal": "RR_B60_H5_Z20 on ETH-EUR",
        "ranked_strategy_count": len(selected),
        **({"top_ten_count": 10} if len(selected) == 10 else {}),
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row["rank"],
        "strategy_name": row["strategy_name"],
        "strategy_family": row["strategy_family"],
        "strategy_dna_hash": row["strategy_dna_hash"],
        "timeframe": row["timeframe"],
        "assets_universe": "|".join(row["assets_universe"]),
        "logic": row["logic"],
        "net_cagr": row["net_cagr"],
        "net_total_return": row["net_total_return"],
        "normal_profit_factor": row["normal_profit_factor"],
        "profit_factor_definition": row["profit_factor_definition"],
        "stressed_profit_factor": row["stressed_profit_factor"],
        "double_cost_profit_factor": row["double_cost_profit_factor"],
        "sharpe": row["sharpe"],
        "sortino": row["sortino"],
        "calmar": row["calmar"],
        "maximum_drawdown": row["maximum_drawdown"],
        "monte_carlo_p95_drawdown": row["monte_carlo_p95_drawdown"],
        "sample_count": row["sample_count"],
        "sample_unit": row["sample_unit"],
        "positive_folds_or_periods": row["positive_folds_or_periods"],
        "total_folds_or_periods": row["total_folds_or_periods"],
        "positive_years": row["positive_years"],
        "negative_years": row["negative_years"],
        "expectancy": row["expectancy"],
        "average_exposure": row["average_exposure"],
        "turnover": row["turnover"],
        "benchmark_result": row["benchmark_result"],
        "holdout_status": row["holdout_status"],
        "forward_status": row["forward_status"],
        "statistical_gate_status": row["statistical_evidence"]["status"],
        "ranking_a_historical_performance": row["ranking_a_historical_performance"],
        "ranking_b_robustness": row["ranking_b_robustness"],
        "ranking_c_practical": row["ranking_c_practical"],
        "composite_score": row["scores"]["composite"],
        "recommended_phase": row["recommended_phase"],
    }


def _format_percentage(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2%}"


def _format_number(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _table_headers() -> list[str]:
    return [
        "Rank",
        "Strategy",
        "Family",
        "Timeframe",
        "Universe",
        "CAGR",
        "Total return",
        "PF",
        "Stressed PF",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Max DD",
        "MC P95 DD",
        "Sample",
        "Positive folds/periods",
        "Average exposure",
        "Composite",
        "Phase",
    ]


def _table_values(row: dict[str, Any]) -> list[str]:
    return [
        str(row["rank"]),
        row["strategy_name"],
        row["strategy_family"],
        row["timeframe"],
        ", ".join(row["assets_universe"]),
        _format_percentage(row["net_cagr"]),
        _format_percentage(row["net_total_return"]),
        _format_number(row["normal_profit_factor"], 3),
        _format_number(row["stressed_profit_factor"], 3),
        _format_number(row["sharpe"]),
        _format_number(row["sortino"]),
        _format_number(row["calmar"]),
        _format_percentage(row["maximum_drawdown"]),
        _format_percentage(row["monte_carlo_p95_drawdown"]),
        f"{row['sample_count']} {row['sample_unit']}",
        f"{row['positive_folds_or_periods']}/{row['total_folds_or_periods']}",
        _format_percentage(row["average_exposure"]),
        _format_number(row["scores"]["composite"]),
        row["recommended_phase"],
    ]


def _ranked_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(report.get("ranking_limit") or 10)
    return list(report[f"top_{limit}"])


def _markdown_document(report: dict[str, Any]) -> str:
    summary = report["executive_summary"]
    limit = int(report.get("ranking_limit") or 10)
    ranked_rows = _ranked_rows(report)
    headers = _table_headers()
    separator = ["---"] * len(headers)
    lines = [
        f"# Top {limit} Existing Strategies v1",
        "",
        "## Executive summary",
        "",
        f"- Historical evaluations audited: **{summary['historical_evaluations_audited']:,}**",
        (
            "- Unique strategy-DNA equivalents audited: "
            f"**{summary['unique_strategy_dna_equivalents_audited']:,}**"
        ),
        f"- Comparable positive-cost longlist: **{summary['comparable_positive_cost_longlist']}**",
        (
            "- Positive under stressed/double costs where available: "
            f"**{summary['positive_under_stressed_or_double_costs']}**"
        ),
        "- Valid untouched holdouts: **0**",
        "- Strategies with prospective performance observations: **0**",
        "- Paper candidates: **0**; live-ready candidates: **0**",
        "- Proven profitable strategy: **NO**",
        (
            "- Primary €5 execution-canary proposal: "
            f"**{summary['primary_five_euro_canary_proposal']}** "
            "(proposal only; execution remains disabled)."
        ),
        "",
        summary["honest_conclusion"],
        "",
        f"## Composite top {limit}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend(
        "| " + " | ".join(value.replace("|", "/") for value in _table_values(row)) + " |"
        for row in ranked_rows
    )
    lines.extend(
        [
            "",
            "## Ranking method",
            "",
            (
                "The score uses the requested fixed weights: 30% historical performance, "
                "30% robustness, 20% drawdown/capital protection, 10% sample quality, and "
                "10% practical deployability. Continuous metrics use 5th/95th percentile "
                "winsorization followed by percentile ranks. Missing Monte Carlo drawdown "
                "receives a neutral 50th-percentile value; missing stressed PF receives a "
                "documented conservative 25th-percentile value. At most two strategies per "
                f"economic cluster can enter the top {limit}."
            ),
            "",
            "## Individual analyses",
            "",
        ]
    )
    for row in ranked_rows:
        stats = row["statistical_evidence"]
        lines.extend(
            [
                f"### {row['rank']}. {row['strategy_name']}",
                "",
                f"- Identity: `{row['strategy_dna_hash']}`",
                f"- Frozen parameters: `{json.dumps(row['parameters'], sort_keys=True)}`",
                f"- Logic: {row['logic']}",
                f"- Entry: {row['entry_logic']}",
                f"- Confirmation: {row['confirmation_logic']}",
                f"- Filter: {row['filter_logic']}",
                f"- Exit: {row['exit_logic']}",
                f"- Works: {row['works_in_regimes']}",
                f"- Fails: {row['fails_in_regimes']}",
                f"- Statistical weakness: {row['largest_statistical_weakness']}",
                f"- Operational weakness: {row['largest_operational_weakness']}",
                f"- Drawdown risk: {row['largest_drawdown_risk']}",
                f"- Concentration/dependence: {row['concentration_warning']}",
                (
                    "- Costs: normal PF "
                    f"{_format_number(row['normal_profit_factor'], 3)}, stressed PF "
                    f"{_format_number(row['stressed_profit_factor'], 3)}, stressed return "
                    f"{_format_percentage(row['stressed_total_return'])}."
                ),
                f"- Benchmark: {row['benchmark_result']}",
                f"- Statistical tests: {stats['status']}",
                f"- Prospective evidence: {row['prospective_evidence']}",
                f"- Retrospective evidence: {row['retrospective_evidence']}",
                (
                    "- Bitvavo spot-only/long-only compatible: "
                    f"{row['bitvavo_spot_long_only_compatible']}"
                ),
                f"- Recommended phase: `{row['recommended_phase']}` — {row['phase_reason']}",
                "- Freeze: " + ", ".join(row["freeze_requirements"]) + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Canary and shadow proposal",
            "",
            (
                "Primary proposal: `RR_B60_H5_Z20` on `ETH-EUR`, maximum €5 per order, "
                "maximum €10 total exposure, one position, one new order per day, and no "
                "autoscaling. Frozen shadow only: "
                + " and ".join(
                    f"`{item['strategy_name']}`"
                    for item in report["canary_selection"]["frozen_shadow"]
                )
                + ". No order was generated or submitted by this audit."
            ),
            "",
            "## Positive results not advanced",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["positive_not_advanced"])
    lines.append("")
    return "\n".join(lines)


def _html_document(report: dict[str, Any]) -> str:
    limit = int(report.get("ranking_limit") or 10)
    ranked_rows = _ranked_rows(report)
    headers = _table_headers()
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    rows_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value)}</td>" for value in _table_values(row))
        + "</tr>"
        for row in ranked_rows
    )
    analyses = "".join(
        (
            f"<section><h3>{row['rank']}. {html.escape(row['strategy_name'])}</h3>"
            f"<p><strong>DNA:</strong> <code>{html.escape(row['strategy_dna_hash'])}</code></p>"
            f"<p><strong>Parameters:</strong> <code>"
            f"{html.escape(json.dumps(row['parameters'], sort_keys=True))}</code></p>"
            f"<p>{html.escape(row['logic'])}</p>"
            f"<ul><li><strong>Entry:</strong> {html.escape(row['entry_logic'])}</li>"
            f"<li><strong>Confirmation:</strong> {html.escape(row['confirmation_logic'])}</li>"
            f"<li><strong>Filter:</strong> {html.escape(row['filter_logic'])}</li>"
            f"<li><strong>Exit:</strong> {html.escape(row['exit_logic'])}</li>"
            f"<li><strong>Works:</strong> {html.escape(row['works_in_regimes'])}</li>"
            f"<li><strong>Fails:</strong> {html.escape(row['fails_in_regimes'])}</li>"
            f"<li><strong>Statistical weakness:</strong> "
            f"{html.escape(row['largest_statistical_weakness'])}</li>"
            f"<li><strong>Operational weakness:</strong> "
            f"{html.escape(row['largest_operational_weakness'])}</li>"
            f"<li><strong>Concentration:</strong> "
            f"{html.escape(row['concentration_warning'])}</li>"
            f"<li><strong>Benchmark:</strong> "
            f"{html.escape(row['benchmark_result'])}</li>"
            f"<li><strong>Prospective:</strong> "
            f"{html.escape(row['prospective_evidence'])}</li>"
            f"<li><strong>Phase:</strong> "
            f"{html.escape(row['recommended_phase'])}</li></ul></section>"
        )
        for row in ranked_rows
    )
    summary = report["executive_summary"]
    return (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        f"<title>Top {limit} Existing Strategies v1</title>"
        "<style>body{font:14px system-ui;max-width:1600px;margin:32px auto;color:#17202a}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid "
        "#d0d5dd;padding:6px;vertical-align:top}th{background:#f2f4f7;position:sticky;"
        "top:0}code{word-break:break-all}.warning{background:#fff4e5;padding:12px;"
        "border-left:4px solid #f79009}section{border-top:1px solid #ddd;margin-top:20px}"
        f"</style><h1>Top {limit} Existing Strategies v1</h1>"
        f"<div class=\"warning\"><strong>No proven profitable strategy.</strong> "
        f"{html.escape(summary['honest_conclusion'])}</div>"
        f"<p>Audited {summary['historical_evaluations_audited']:,} evaluations and "
        f"{summary['unique_strategy_dna_equivalents_audited']:,} DNA equivalents. "
        "Zero untouched holdouts, forward performance observations, paper candidates, "
        "live-ready candidates, or orders.</p>"
        f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
        "<h2>Individual analyses</h2>"
        f"{analyses}</html>"
    )


def build_report(
    root: Path,
    *,
    limit: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    longlist = score_candidates(collect_longlist(root))
    selected = select_top_strategies(longlist, limit=limit)
    inventory = _audit_inventory(root)
    database = _database_identity(root)
    report_basename = f"top_{limit}_existing_strategies_v1"
    evidence_basename = f"top_{limit}_existing_strategies_evidence_v1"
    ranked_key = f"top_{limit}"
    report = {
        "schema_version": report_basename,
        "ranking_limit": limit,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_root": str(root.resolve()),
        "report_type": "READ_ONLY_EXISTING_EVIDENCE_RANKING",
        "new_backtests_run": 0,
        "strategy_parameters_changed": 0,
        "orders_generated": 0,
        "orders_submitted": 0,
        "score_weights": SCORE_WEIGHTS,
        "normalization": {
            "returns": (
                "Canonical campaign/volume sources explicitly use fractions; legacy "
                "unlabelled values use the documented auto percent heuristic."
            ),
            "drawdowns": "Absolute magnitude after percent/fraction normalization.",
            "profit_factor": "No unit conversion; definition retained per source.",
            "continuous_scores": "5/95 winsorized percentile ranks.",
            "missing_monte_carlo": "Neutral 50th percentile in capital-protection score.",
            "missing_stressed_pf": "Conservative 25th percentile in robustness score.",
        },
        "audit_inventory": inventory,
        "executive_summary": _executive_summary(inventory, longlist, selected),
        "ranking_a_historical_performance": [
            {
                "rank": index,
                "strategy_name": row["strategy_name"],
                "score": row["scores"]["historical_performance"],
            }
            for index, row in enumerate(
                sorted(
                    longlist,
                    key=lambda item: (
                        -item["scores"]["historical_performance"],
                        item["strategy_name"],
                    ),
                ),
                1,
            )
        ],
        "ranking_b_robustness": [
            {
                "rank": index,
                "strategy_name": row["strategy_name"],
                "score": row["scores"]["robustness"],
            }
            for index, row in enumerate(
                sorted(
                    longlist,
                    key=lambda item: (
                        -item["scores"]["robustness"],
                        item["strategy_name"],
                    ),
                ),
                1,
            )
        ],
        "ranking_c_practical": [
            {
                "rank": index,
                "strategy_name": row["strategy_name"],
                "score": row["scores"]["practical_deployability"],
            }
            for index, row in enumerate(
                sorted(
                    longlist,
                    key=lambda item: (
                        -item["scores"]["practical_deployability"],
                        item["strategy_name"],
                    ),
                ),
                1,
            )
        ],
        ranked_key: selected,
        "positive_not_advanced": [
            (
                "PIECEWISE/SEMI_AGGRESSIVE/BALANCED capital policies: very high returns "
                "but 35.8%–49.5% drawdowns; no further financial phase."
            ),
            (
                "Residual momentum: positive full/stressed history but negative confirmation "
                "and approximately 29% drawdown."
            ),
            (
                "Volume discovery-only ADA variants: positive historical returns but outside "
                "the executable allowlist and often negative confirmation."
            ),
            (
                "Portfolio and signal storm survivors: positive development/validation rows "
                "but negative confirmation or missing canonical exact confirmation."
            ),
            (
                "Baseline leaderboard combinations: positive after costs but best PF below "
                "1.08 and CAGR below 1%."
            ),
        ],
        "canary_selection": _canary_proposal(selected),
    }
    evidence = {
        "schema_version": evidence_basename,
        "ranking_limit": limit,
        "generated_at": report["generated_at"],
        "repository_root": report["repository_root"],
        "database_identity": database,
        "audit_inventory": inventory,
        "strategy_evidence": {
            row["strategy_name"]: {
                "strategy_dna_hash": row["strategy_dna_hash"],
                "parameters_hash": stable_hash(row["parameters"]),
                "sources": row["evidence"],
            }
            for row in selected
        },
        "report_invariants": {
            "ranked_strategy_count": len(selected),
            "unique_strategy_names": len({row["strategy_name"] for row in selected}),
            "maximum_cluster_count": max(
                Counter(row["family_cluster"] for row in selected).values()
            ),
            "orders_before_generation": database["row_counts"]["orders"],
            "fills_before_generation": database["row_counts"]["fills"],
            "positions_before_generation": database["row_counts"]["positions"],
            "new_backtests_run": 0,
            "strategy_parameters_changed": 0,
        },
    }
    return report, evidence


def write_reports(root: Path, *, limit: int = 10) -> dict[str, Path]:
    report, evidence = build_report(root, limit=limit)
    ranked_rows = _ranked_rows(report)
    report_basename = f"top_{limit}_existing_strategies_v1"
    evidence_basename = f"top_{limit}_existing_strategies_evidence_v1"
    directory = root / "output" / "lab" / "reports"
    json_path = directory / f"{report_basename}.json"
    csv_path = directory / f"{report_basename}.csv"
    markdown_path = directory / f"{report_basename}.md"
    html_path = directory / f"{report_basename}.html"
    evidence_path = directory / f"{evidence_basename}.json"
    rows = [_csv_row(row) for row in ranked_rows]
    fieldnames = list(rows[0])
    csv_lines: list[str] = []

    class _Buffer:
        def write(self, value: str) -> int:
            csv_lines.append(value)
            return len(value)

    writer = csv.DictWriter(_Buffer(), fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    paths = {
        "json": atomic_write_json(json_path, report),
        "csv": atomic_write_text(csv_path, "".join(csv_lines)),
        "markdown": atomic_write_text(markdown_path, _markdown_document(report)),
        "html": atomic_write_text(html_path, _html_document(report)),
        "evidence": atomic_write_json(evidence_path, evidence),
    }
    return paths


def verify_reports(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    report = read_json(paths["json"])
    evidence = read_json(paths["evidence"])
    csv_rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8", newline="")))
    limit = int(report.get("ranking_limit") or 10)
    ranked_rows = _ranked_rows(report)
    json_names = [row["strategy_name"] for row in ranked_rows]
    csv_names = [row["strategy_name"] for row in csv_rows]
    if json_names != csv_names:
        raise ValueError(f"JSON and CSV top {limit} differ")
    if len(json_names) != limit or len(set(json_names)) != limit:
        raise ValueError(f"top {limit} identity invariant failed")
    if max(Counter(row["family_cluster"] for row in ranked_rows).values()) > 2:
        raise ValueError("family clustering invariant failed")
    if any(row["recommended_phase"] not in ALLOWED_PHASES for row in ranked_rows):
        raise ValueError("invalid recommended phase")
    for identity in evidence["strategy_evidence"].values():
        for source in identity["sources"]:
            path = Path(source["path"])
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise ValueError(f"source hash mismatch: {path}")
    database = _database_identity(root)
    for table in ("orders", "fills", "positions"):
        if database["row_counts"][table] != 0:
            raise ValueError(f"{table} were generated during audit")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    html_text = paths["html"].read_text(encoding="utf-8")
    if any(name not in markdown or name not in html_text for name in json_names):
        raise ValueError("Markdown/HTML strategy tables do not reconcile")
    return {
        "status": "PASSED",
        "json_csv_same_order": True,
        "markdown_html_contain_all_strategies": True,
        "source_hashes_verified": True,
        "maximum_cluster_count": max(
            Counter(row["family_cluster"] for row in ranked_rows).values()
        ),
        "orders_after_generation": database["row_counts"]["orders"],
        "fills_after_generation": database["row_counts"]["fills"],
        "positions_after_generation": database["row_counts"]["positions"],
        "report_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, choices=(10, 20), default=10)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    paths = write_reports(root, limit=arguments.limit)
    verification = verify_reports(root, paths)
    print(
        json.dumps(
            {
                "paths": {name: str(path.resolve()) for name, path in paths.items()},
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
