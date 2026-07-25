"""Append-only forward evidence for frozen portfolio research.

Forward observations are reconstructed from frozen next-open decisions and
realized open-to-open market moves. Historical observations are immutable:
data corrections or logic drift that alter an existing record fail closed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.portfolio_selection import _validated_panel
from utils.common import stable_hash

FORWARD_OBSERVER_SCHEMA_VERSION = "breakout_forward_observer_v1"
PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION = "portfolio_forward_observer_v1"


class ForwardHistoryRevisionError(RuntimeError):
    """Raised when already-recorded forward evidence changes."""


@dataclass(frozen=True, slots=True)
class BreakoutForwardEvidence:
    """Candidate immutable observations and their aggregate state."""

    observations: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    degradation_observation: dict[str, Any] | None
    schema_version: str = FORWARD_OBSERVER_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ForwardPerformanceGatePolicy:
    """Frozen thresholds applied only after sample requirements are met."""

    minimum_profit_factor: float = 1.15
    minimum_stressed_profit_factor: float = 1.05
    maximum_drawdown: float = 0.20
    minimum_effective_sample_size: int = 100
    stressed_cost_multiplier: float = 2.0
    bootstrap_samples: int = 2_000
    bootstrap_block_size: int = 10
    bootstrap_seed: int = 42

    def __post_init__(self) -> None:
        if self.minimum_profit_factor <= 1:
            raise ValueError("minimum_profit_factor must exceed one")
        if self.minimum_stressed_profit_factor < 1:
            raise ValueError(
                "minimum_stressed_profit_factor must be at least one"
            )
        if not 0 < self.maximum_drawdown < 1:
            raise ValueError("maximum_drawdown must be in (0, 1)")
        if self.minimum_effective_sample_size < 2:
            raise ValueError(
                "minimum_effective_sample_size must be at least two"
            )
        if self.stressed_cost_multiplier < 1:
            raise ValueError(
                "stressed_cost_multiplier must be at least one"
            )
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if self.bootstrap_block_size < 1:
            raise ValueError("bootstrap_block_size must be positive")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _finite_prices(row: pd.Series) -> dict[str, float]:
    return {
        str(market): float(value)
        for market, value in row.items()
        if math.isfinite(float(value))
    }


def _regime_at(
    closes: pd.DataFrame,
    *,
    prior_position: int,
) -> dict[str, Any]:
    history = closes.iloc[: prior_position + 1]
    btc = history["BTC-EUR"].dropna()
    if len(btc) < 200:
        btc_trend = "UNKNOWN"
        btc_distance = None
    else:
        ema = btc.ewm(
            span=200,
            adjust=False,
            min_periods=200,
        ).mean()
        btc_distance = float(btc.iloc[-1] / ema.iloc[-1] - 1.0)
        btc_trend = "UP" if btc_distance >= 0 else "DOWN"

    btc_returns = np.log(btc).diff()
    rolling_volatility = btc_returns.rolling(
        20,
        min_periods=20,
    ).std(ddof=1)
    causal_median = rolling_volatility.expanding(
        min_periods=60,
    ).median().shift(1)
    current_volatility = (
        float(rolling_volatility.iloc[-1])
        if len(rolling_volatility)
        and math.isfinite(float(rolling_volatility.iloc[-1]))
        else None
    )
    median_volatility = (
        float(causal_median.iloc[-1])
        if len(causal_median)
        and math.isfinite(float(causal_median.iloc[-1]))
        else None
    )
    volatility_state = (
        "UNKNOWN"
        if current_volatility is None or median_volatility is None
        else "HIGH"
        if current_volatility >= median_volatility
        else "LOW"
    )

    momentum = history.iloc[-1] / history.shift(90).iloc[-1] - 1.0
    eligible = momentum.replace([np.inf, -np.inf], np.nan).dropna()
    breadth = (
        float((eligible > 0).mean())
        if not eligible.empty
        else None
    )
    breadth_state = (
        "UNKNOWN"
        if breadth is None
        else "BROAD"
        if breadth >= 0.5
        else "NARROW"
    )
    return {
        "btc_trend": btc_trend,
        "volatility": volatility_state,
        "breadth": breadth_state,
        "btc_ema200_distance": btc_distance,
        "btc_realized_volatility_20": current_volatility,
        "causal_volatility_median": median_volatility,
        "positive_momentum_breadth_90": breadth,
    }


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"observation_hash"}
    }
    return stable_hash(payload, length=64)


def build_forward_hash_chain(
    observations: list[dict[str, Any]]
    | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build a deterministic chain without mutating immutable observations."""

    previous_hash = "0" * 64
    entries: list[dict[str, Any]] = []
    for sequence_number, record in enumerate(observations, start=1):
        observation_hash = str(record.get("observation_hash") or "")
        if observation_hash != _record_hash(record):
            raise ForwardHistoryRevisionError(
                "FORWARD_HASH_CHAIN_OBSERVATION_CHECKSUM_INVALID:"
                f"{record.get('observation_id')}"
            )
        chain_hash = stable_hash(
            {
                "sequence_number": sequence_number,
                "previous_record_hash": previous_hash,
                "observation_id": str(record["observation_id"]),
                "observation_hash": observation_hash,
            },
            length=64,
        )
        entries.append(
            {
                "sequence_number": sequence_number,
                "observation_id": str(record["observation_id"]),
                "observation_hash": observation_hash,
                "previous_record_hash": previous_hash,
                "record_hash": chain_hash,
            }
        )
        previous_hash = chain_hash
    return {
        "schema_version": "forward_hash_chain_v1",
        "record_count": len(entries),
        "genesis_hash": "0" * 64,
        "root_hash": previous_hash,
        "entries": entries,
    }


def _coverage(
    observations: list[dict[str, Any]],
    *,
    minimum_per_state: int,
) -> dict[str, Any]:
    required = {
        "btc_trend": ("UP", "DOWN"),
        "volatility": ("HIGH", "LOW"),
        "breadth": ("BROAD", "NARROW"),
    }
    counts = {
        axis: {
            state: sum(
                item["regime"][axis] == state
                for item in observations
            )
            for state in states
        }
        for axis, states in required.items()
    }
    checks = {
        f"{axis}_{state}": counts[axis][state] >= minimum_per_state
        for axis, states in required.items()
        for state in states
    }
    return {
        "minimum_per_state": minimum_per_state,
        "counts": counts,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _degradation_evidence(
    *,
    forward_returns: list[float],
    historical_returns: list[float],
    window: int = 30,
) -> dict[str, Any] | None:
    if len(forward_returns) < window:
        return None
    historical = pd.Series(historical_returns, dtype=float)
    historical_windows = (
        (1.0 + historical)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        - 1.0
    ).dropna()
    if len(historical_windows) < 30:
        return None
    cv_std = float(historical_windows.std(ddof=1))
    if not math.isfinite(cv_std) or cv_std <= 0:
        return {
            "live_return": float(
                np.prod(1.0 + np.asarray(forward_returns[-window:])) - 1.0
            ),
            "cv_mean": float(historical_windows.mean()),
            "cv_std": cv_std,
            "observation_count": len(forward_returns),
            "window": f"{window}d",
            "source": "BREAKOUT_APPEND_ONLY_FORWARD_OBSERVER",
        }
    return {
        "live_return": float(
            np.prod(1.0 + np.asarray(forward_returns[-window:])) - 1.0
        ),
        "cv_mean": float(historical_windows.mean()),
        "cv_std": cv_std,
        "observation_count": len(forward_returns),
        "window": f"{window}d",
        "source": "BREAKOUT_APPEND_ONLY_FORWARD_OBSERVER",
    }


def _profit_factor(returns: np.ndarray) -> float | None:
    positive = float(returns[returns > 0].sum())
    negative = float(abs(returns[returns < 0].sum()))
    if negative <= 0:
        return None
    return positive / negative


def _effective_sample_size(returns: np.ndarray) -> dict[str, Any]:
    count = len(returns)
    if count < 2:
        return {
            "raw_observations": count,
            "lag_one_autocorrelation": None,
            "effective_sample_size": count,
        }
    lag_one = float(pd.Series(returns).autocorr(lag=1))
    if not math.isfinite(lag_one) or abs(lag_one) >= 1:
        lag_one = 0.0
    estimate = count * (1.0 - lag_one) / (1.0 + lag_one)
    return {
        "raw_observations": count,
        "lag_one_autocorrelation": lag_one,
        "effective_sample_size": int(
            min(count, max(1.0, math.floor(estimate)))
        ),
    }


def _maximum_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return float(drawdown.min(initial=0.0))


def _block_bootstrap_mean_ci_lower(
    returns: np.ndarray,
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> float | None:
    count = len(returns)
    if count < max(2, block_size):
        return None
    generator = np.random.default_rng(seed)
    block = min(block_size, count)
    starts = np.arange(0, count - block + 1)
    blocks_needed = math.ceil(count / block)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = generator.choice(
            starts,
            size=blocks_needed,
            replace=True,
        )
        draw = np.concatenate(
            [returns[start : start + block] for start in selected]
        )[:count]
        means[sample] = float(draw.mean())
    return float(np.quantile(means, 0.025))


def _formal_performance(
    observations: list[dict[str, Any]],
    *,
    policy: ForwardPerformanceGatePolicy,
) -> tuple[dict[str, Any], dict[str, bool]]:
    normal = np.asarray(
        [float(item["net_return"]) for item in observations],
        dtype=float,
    )
    stressed = np.asarray(
        [
            (
                1.0
                - float(item["turnover"])
                * (
                    float(item["expected_cost_fraction"])
                    / max(float(item["turnover"]), 1e-30)
                )
                * policy.stressed_cost_multiplier
            )
            * (1.0 + float(item["gross_return"]))
            - 1.0
            if float(item["turnover"]) > 1e-30
            else float(item["gross_return"])
            for item in observations
        ],
        dtype=float,
    )
    normal_equity = float(np.prod(1.0 + normal))
    stressed_equity = float(np.prod(1.0 + stressed))
    normal_pf = _profit_factor(normal)
    stressed_pf = _profit_factor(stressed)
    sample = _effective_sample_size(normal)
    ci_lower = _block_bootstrap_mean_ci_lower(
        normal,
        samples=policy.bootstrap_samples,
        block_size=policy.bootstrap_block_size,
        seed=policy.bootstrap_seed,
    )
    metrics = {
        "net_return": normal_equity - 1.0,
        "stressed_net_return": stressed_equity - 1.0,
        "profit_factor": normal_pf,
        "stressed_profit_factor": stressed_pf,
        "maximum_drawdown": _maximum_drawdown(normal),
        "stressed_maximum_drawdown": _maximum_drawdown(stressed),
        "mean_daily_return": float(normal.mean()),
        "stressed_mean_daily_return": float(stressed.mean()),
        "mean_daily_return_ci_lower_95": ci_lower,
        **sample,
    }
    checks = {
        "net_positive": metrics["net_return"] > 0,
        "stressed_net_positive": metrics["stressed_net_return"] > 0,
        "profit_factor": (
            normal_pf is None
            or normal_pf >= policy.minimum_profit_factor
        ),
        "stressed_profit_factor": (
            stressed_pf is None
            or stressed_pf >= policy.minimum_stressed_profit_factor
        ),
        "maximum_drawdown": (
            abs(float(metrics["maximum_drawdown"]))
            <= policy.maximum_drawdown
        ),
        "effective_sample_size": (
            int(metrics["effective_sample_size"])
            >= policy.minimum_effective_sample_size
        ),
        "confidence_interval_lower_positive": (
            ci_lower is not None and ci_lower > 0
        ),
    }
    return metrics, checks


def _build_portfolio_forward_evidence(
    result: Any,
    frames: Mapping[str, pd.DataFrame],
    *,
    forward_start: Any,
    schema_version: str,
    minimum_observations: int = 365,
    minimum_rebalances: int = 30,
    minimum_regime_decisions: int = 5,
    performance_policy: ForwardPerformanceGatePolicy | None = None,
) -> BreakoutForwardEvidence:
    """Reconstruct immutable realized observations from frozen decisions."""

    start = _utc(forward_start)
    opens, closes = _validated_panel(
        frames,
        benchmark_market="BTC-EUR",
        portfolio_policy=result.portfolio_policy,
    )
    decisions = result.decisions[
        result.decisions["reason"] != "TERMINAL_LIQUIDATION"
    ].copy()
    decisions["executed_at"] = pd.to_datetime(
        decisions["executed_at"],
        utc=True,
    )
    if decisions["executed_at"].duplicated().any():
        raise ValueError("breakout result contains duplicate execution times")
    by_execution = {
        _utc(row["executed_at"]): row.to_dict()
        for _, row in decisions.iterrows()
    }
    weights = pd.Series(0.0, index=opens.columns, dtype=float)
    observations: list[dict[str, Any]] = []
    forward_decisions: list[dict[str, Any]] = []
    historical_returns: list[float] = []
    one_way_cost = float(result.cost_breakdown["one_way_cost_rate"])

    for position in range(len(opens) - 1):
        execution_at = _utc(opens.index[position])
        realization_at = _utc(opens.index[position + 1])
        decision = by_execution.get(execution_at)
        turnover = 0.0
        expected_cost_fraction = 0.0
        decision_at: str | None = None
        reason = "HOLD_UNCHANGED"
        if decision is not None:
            target = pd.Series(
                0.0,
                index=opens.columns,
                dtype=float,
            )
            for market, weight in dict(
                decision.get("target_weights") or {}
            ).items():
                if market not in target.index:
                    raise ValueError(
                        f"forward decision contains unknown market: {market}"
                    )
                target[market] = float(weight)
            turnover = float((target - weights).abs().sum())
            recorded_turnover = float(decision.get("turnover") or 0.0)
            if not math.isclose(
                turnover,
                recorded_turnover,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError("forward turnover reconstruction mismatch")
            weights = target
            expected_cost_fraction = turnover * one_way_cost
            decision_at = _utc(decision["decision_at"]).isoformat()
            reason = str(decision["reason"])

        start_prices = opens.iloc[position]
        end_prices = opens.iloc[position + 1]
        held = weights[weights > 1e-12].index
        if not (
            start_prices.reindex(held).notna().all()
            and end_prices.reindex(held).notna().all()
        ):
            raise ValueError("held asset lacks forward open price")
        asset_returns = end_prices / start_prices - 1.0
        gross_return = float((weights * asset_returns).sum())
        net_return = float(
            (1.0 - expected_cost_fraction) * (1.0 + gross_return)
            - 1.0
        )
        if execution_at < start:
            historical_returns.append(net_return)
            continue

        prior_position = max(0, position - 1)
        regime = _regime_at(
            closes,
            prior_position=prior_position,
        )
        record = {
            "schema_version": schema_version,
            "observation_id": stable_hash(
                [
                    result.parameters.dna_hash,
                    execution_at.isoformat(),
                    realization_at.isoformat(),
                ],
                length=64,
            ),
            "strategy_dna_hash": result.parameters.dna_hash,
            "execution_identity": result.summary()[
                "execution_identity"
            ],
            "decision_at": decision_at,
            "execution_at": execution_at.isoformat(),
            "realization_at": realization_at.isoformat(),
            "decision_event": decision is not None,
            "reason": reason,
            "target_weights": {
                market: float(weight)
                for market, weight in weights.items()
                if float(weight) > 1e-12
            },
            "cash_fraction": float(1.0 - weights.sum()),
            "turnover": turnover,
            "expected_cost_fraction": expected_cost_fraction,
            "gross_return": gross_return,
            "net_return": net_return,
            "source_open_prices": {
                "start": _finite_prices(start_prices),
                "end": _finite_prices(end_prices),
            },
            "regime": regime,
            "execution_instruction": (
                "NEXT_AVAILABLE_OPEN_HYPOTHETICAL_ONLY"
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        record["observation_hash"] = _record_hash(record)
        observations.append(record)
        if decision is not None:
            forward_decisions.append(
                {
                    "observation_id": record["observation_id"],
                    "observation_hash": record["observation_hash"],
                    "decision_at": decision_at,
                    "execution_at": execution_at.isoformat(),
                    "reason": reason,
                    "turnover": turnover,
                    "target_weights": record["target_weights"],
                    "regime": regime,
                }
            )

    forward_returns = [
        float(item["net_return"]) for item in observations
    ]
    equity = float(
        np.prod(1.0 + np.asarray(forward_returns, dtype=float))
        if forward_returns
        else 1.0
    )
    rebalances = sum(
        float(item["turnover"]) > 1e-12 for item in observations
    )
    coverage = _coverage(
        observations,
        minimum_per_state=minimum_regime_decisions,
    )
    checks = {
        "minimum_closed_daily_observations": (
            len(observations) >= minimum_observations
        ),
        "minimum_rebalances": rebalances >= minimum_rebalances,
        "minimum_regime_coverage": coverage["passed"],
    }
    selected_performance_policy = (
        performance_policy or ForwardPerformanceGatePolicy()
    )
    sample_requirements_met = all(checks.values())
    performance_metrics: dict[str, Any] | None = None
    performance_checks: dict[str, bool] | None = None
    if sample_requirements_met:
        performance_metrics, performance_checks = _formal_performance(
            observations,
            policy=selected_performance_policy,
        )
    performance_pass = bool(
        performance_checks
        and all(performance_checks.values())
    )
    summary = {
        "status": (
            "FORWARD_PERFORMANCE_PASS"
            if performance_pass
            else "FORWARD_PERFORMANCE_NOT_QUALIFIED"
            if sample_requirements_met
            else "COLLECTING_FORWARD_DATA"
        ),
        "forward_start": start.isoformat(),
        "latest_realization_at": (
            observations[-1]["realization_at"]
            if observations
            else None
        ),
        "closed_daily_observations": len(observations),
        "required_closed_daily_observations": minimum_observations,
        "remaining_closed_daily_observations": max(
            0,
            minimum_observations - len(observations),
        ),
        "forward_decisions": len(forward_decisions),
        "forward_rebalances": rebalances,
        "required_forward_rebalances": minimum_rebalances,
        "remaining_forward_rebalances": max(
            0,
            minimum_rebalances - rebalances,
        ),
        "forward_net_return": equity - 1.0,
        "regime_coverage": coverage,
        "checks": checks,
        "performance_gate_policy": {
            "minimum_profit_factor": (
                selected_performance_policy.minimum_profit_factor
            ),
            "minimum_stressed_profit_factor": (
                selected_performance_policy.minimum_stressed_profit_factor
            ),
            "maximum_drawdown": (
                selected_performance_policy.maximum_drawdown
            ),
            "minimum_effective_sample_size": (
                selected_performance_policy.minimum_effective_sample_size
            ),
            "stressed_cost_multiplier": (
                selected_performance_policy.stressed_cost_multiplier
            ),
        },
        "formal_performance_gates_evaluated": (
            sample_requirements_met
        ),
        "performance_metrics": performance_metrics,
        "performance_checks": performance_checks,
        "forward_performance_pass": performance_pass,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    return BreakoutForwardEvidence(
        observations=tuple(observations),
        decisions=tuple(forward_decisions),
        summary=summary,
        degradation_observation=_degradation_evidence(
            forward_returns=forward_returns,
            historical_returns=historical_returns,
        ),
        schema_version=schema_version,
    )


def build_breakout_forward_evidence(
    result: Any,
    frames: Mapping[str, pd.DataFrame],
    *,
    forward_start: Any,
    minimum_observations: int = 365,
    minimum_rebalances: int = 30,
    minimum_regime_decisions: int = 5,
    performance_policy: ForwardPerformanceGatePolicy | None = None,
) -> BreakoutForwardEvidence:
    """Build append-only evidence for a frozen breakout portfolio."""

    return _build_portfolio_forward_evidence(
        result,
        frames,
        forward_start=forward_start,
        schema_version=FORWARD_OBSERVER_SCHEMA_VERSION,
        minimum_observations=minimum_observations,
        minimum_rebalances=minimum_rebalances,
        minimum_regime_decisions=minimum_regime_decisions,
        performance_policy=performance_policy,
    )


def build_rotation_forward_evidence(
    result: Any,
    frames: Mapping[str, pd.DataFrame],
    *,
    forward_start: Any,
    minimum_observations: int = 365,
    minimum_rebalances: int = 30,
    minimum_regime_decisions: int = 5,
    performance_policy: ForwardPerformanceGatePolicy | None = None,
) -> BreakoutForwardEvidence:
    """Build append-only evidence for a frozen rotation allocation policy."""

    return _build_portfolio_forward_evidence(
        result,
        frames,
        forward_start=forward_start,
        schema_version=PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION,
        minimum_observations=minimum_observations,
        minimum_rebalances=minimum_rebalances,
        minimum_regime_decisions=minimum_regime_decisions,
        performance_policy=performance_policy,
    )


def merge_forward_observations(
    existing: list[dict[str, Any]],
    candidate: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Append new records and reject any mutation of existing evidence."""

    by_id: dict[str, dict[str, Any]] = {}
    for record in existing:
        observation_id = str(record["observation_id"])
        expected_hash = _record_hash(record)
        if record.get("observation_hash") != expected_hash:
            raise ForwardHistoryRevisionError(
                f"stored observation checksum is invalid: {observation_id}"
            )
        if observation_id in by_id:
            raise ForwardHistoryRevisionError(
                f"duplicate stored observation: {observation_id}"
            )
        by_id[observation_id] = dict(record)
    candidate_ids = {
        str(record["observation_id"]) for record in candidate
    }
    missing_from_candidate = set(by_id) - candidate_ids
    if missing_from_candidate:
        raise ForwardHistoryRevisionError(
            "FORWARD_SOURCE_TRUNCATION_DETECTED:"
            f"{sorted(missing_from_candidate)}"
        )
    for record in candidate:
        observation_id = str(record["observation_id"])
        previous = by_id.get(observation_id)
        if previous is not None:
            if previous.get("observation_hash") != record.get(
                "observation_hash"
            ):
                raise ForwardHistoryRevisionError(
                    "FORWARD_HISTORY_REVISION_DETECTED:"
                    f"{observation_id}"
                )
            continue
        by_id[observation_id] = dict(record)
    return sorted(
        by_id.values(),
        key=lambda item: (
            str(item["execution_at"]),
            str(item["observation_id"]),
        ),
    )


def merge_portfolio_forward_manifest(
    existing: Mapping[str, Any],
    evidence: BreakoutForwardEvidence,
    *,
    source_candidate_identity: str,
    strategy_dna_hash: str,
    execution_identity: str,
    forward_start: Any,
) -> dict[str, Any]:
    """Merge forward evidence while enforcing immutable observer identity."""

    identity_checks = validate_forward_manifest_identity(
        existing,
        source_candidate_identity=source_candidate_identity,
        strategy_dna_hash=strategy_dna_hash,
        execution_identity=execution_identity,
        forward_start=forward_start,
    )
    existing_observations = list(
        existing.get("forward_observations") or []
    )
    existing_chain = existing.get("forward_hash_chain")
    if existing_chain is not None:
        expected_existing_chain = build_forward_hash_chain(
            existing_observations
        )
        if dict(existing_chain) != expected_existing_chain:
            raise ForwardHistoryRevisionError(
                "FORWARD_HASH_CHAIN_REVISION_DETECTED"
            )
    merged = merge_forward_observations(
        existing_observations,
        evidence.observations,
    )
    forward_hash_chain = build_forward_hash_chain(merged)
    merged_ids = {str(item["observation_id"]) for item in merged}
    decisions = [
        {
            "observation_id": item["observation_id"],
            "observation_hash": item["observation_hash"],
            "decision_at": item["decision_at"],
            "execution_at": item["execution_at"],
            "reason": item["reason"],
            "turnover": item["turnover"],
            "target_weights": item["target_weights"],
            "regime": item["regime"],
        }
        for item in merged
        if bool(item.get("decision_event"))
        and str(item["observation_id"]) in merged_ids
    ]
    payload = dict(existing)
    payload.update(
        {
            **identity_checks,
            "status": "FROZEN_FORWARD_RESEARCH",
            "forward_observer_schema_version": evidence.schema_version,
            "forward_observations": merged,
            "forward_hash_chain": forward_hash_chain,
            "forward_decisions": decisions,
            "forward_summary": {
                **evidence.summary,
                "closed_daily_observations": len(merged),
            },
            "degradation_observation": (
                evidence.degradation_observation
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
            "candidate_promotion_implied": False,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
    )
    return payload


def merge_breakout_forward_manifest(
    existing: Mapping[str, Any],
    evidence: BreakoutForwardEvidence,
    *,
    source_candidate_identity: str,
    strategy_dna_hash: str,
    execution_identity: str,
    forward_start: Any,
) -> dict[str, Any]:
    """Backward-compatible breakout wrapper around the generic merge."""

    return merge_portfolio_forward_manifest(
        existing,
        evidence,
        source_candidate_identity=source_candidate_identity,
        strategy_dna_hash=strategy_dna_hash,
        execution_identity=execution_identity,
        forward_start=forward_start,
    )


def validate_forward_manifest_identity(
    existing: Mapping[str, Any],
    *,
    source_candidate_identity: str,
    strategy_dna_hash: str,
    execution_identity: str,
    forward_start: Any,
) -> dict[str, str]:
    """Validate and return the canonical immutable identity fields."""

    identity_checks = {
        "source_candidate_identity": str(source_candidate_identity),
        "strategy_dna_hash": str(strategy_dna_hash),
        "execution_identity": str(execution_identity),
        "forward_start": _utc(forward_start).isoformat(),
    }
    for field, expected in identity_checks.items():
        current = existing.get(field)
        if current is not None and str(current) != expected:
            raise ForwardHistoryRevisionError(
                f"forward observer identity mismatch: {field}"
            )
    return identity_checks


__all__ = [
    "FORWARD_OBSERVER_SCHEMA_VERSION",
    "PORTFOLIO_FORWARD_OBSERVER_SCHEMA_VERSION",
    "BreakoutForwardEvidence",
    "ForwardPerformanceGatePolicy",
    "ForwardHistoryRevisionError",
    "build_breakout_forward_evidence",
    "build_forward_hash_chain",
    "build_rotation_forward_evidence",
    "merge_breakout_forward_manifest",
    "merge_portfolio_forward_manifest",
    "merge_forward_observations",
    "validate_forward_manifest_identity",
]
