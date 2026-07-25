"""Fail-closed stochastic robustness gates for strategy return paths.

The stationary bootstrap preserves short-range serial dependence while the
Dirichlet test stresses concentration across chronological market blocks.
Neither test changes strategy parameters or the multiple-testing trial count.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from utils.common import stable_hash

STOCHASTIC_VALIDATION_VERSION = "stochastic-validation-v1"


@dataclass(frozen=True)
class StochasticValidationPolicy:
    """Immutable thresholds used by the formal stochastic promotion gates."""

    simulations: int = 10_000
    expected_block_length: int = 10
    maximum_drawdown: float = 0.20
    maximum_drawdown_breach_probability: float = 0.01
    maximum_terminal_loss_probability: float = 0.05
    minimum_p05_total_return: float = 0.0
    dirichlet_blocks: int = 12
    dirichlet_concentrations: tuple[float, ...] = (0.5, 1.0, 5.0)
    minimum_observations: int = 30
    confidence_level: float = 0.95
    seed: int = 42
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.simulations < 100:
            raise ValueError("stochastic simulations must be at least 100")
        if self.expected_block_length < 1:
            raise ValueError("expected block length must be positive")
        if not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("maximum drawdown must be in (0, 1)")
        if not 0.0 <= self.maximum_drawdown_breach_probability <= 1.0:
            raise ValueError("drawdown breach probability must be in [0, 1]")
        if not 0.0 <= self.maximum_terminal_loss_probability <= 1.0:
            raise ValueError("terminal loss probability must be in [0, 1]")
        if self.minimum_p05_total_return <= -1.0:
            raise ValueError("minimum p05 total return must exceed -100%")
        if self.dirichlet_blocks < 4:
            raise ValueError("Dirichlet validation requires at least four blocks")
        if not self.dirichlet_concentrations or any(
            concentration <= 0.0 for concentration in self.dirichlet_concentrations
        ):
            raise ValueError("Dirichlet concentrations must be positive")
        if self.minimum_observations < 8:
            raise ValueError("minimum observations must be at least eight")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be in (0.5, 1)")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")

    @property
    def policy_hash(self) -> str:
        return stable_hash(
            {
                "version": STOCHASTIC_VALIDATION_VERSION,
                **asdict(self),
            },
            length=64,
        )


def _validated_returns(returns: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(
        list(returns) if not isinstance(returns, np.ndarray) else returns,
        dtype=float,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("return path is empty")
    if not np.isfinite(values).all():
        raise ValueError("return path contains non-finite values")
    if np.any(values <= -1.0):
        raise ValueError("arithmetic returns must be greater than -100%")
    return values


def _invalid_result(
    *,
    test: str,
    observation_count: int,
    policy: StochasticValidationPolicy,
    reason: str,
) -> dict[str, Any]:
    return {
        "test": test,
        "version": STOCHASTIC_VALIDATION_VERSION,
        "policy_hash": policy.policy_hash,
        "observation_count": observation_count,
        "passed": False,
        "reason_codes": [reason],
    }


def _wilson_upper_probability(
    events: int,
    total: int,
    *,
    confidence_level: float,
) -> float:
    """Return a one-sided Wilson upper confidence bound for a binomial rate."""

    if total < 1:
        return 1.0
    probability = events / total
    z_score = NormalDist().inv_cdf(confidence_level)
    denominator = 1.0 + z_score**2 / total
    center = probability + z_score**2 / (2.0 * total)
    radius = z_score * math.sqrt(
        probability * (1.0 - probability) / total
        + z_score**2 / (4.0 * total**2)
    )
    return float(min(1.0, (center + radius) / denominator))


def stationary_bootstrap_monte_carlo(
    returns: Iterable[float] | np.ndarray,
    *,
    policy: StochasticValidationPolicy,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Simulate dependent paths using geometric stationary-bootstrap restarts."""

    try:
        values = _validated_returns(returns)
    except ValueError as exc:
        return _invalid_result(
            test="STATIONARY_BOOTSTRAP_MONTE_CARLO",
            observation_count=0,
            policy=policy,
            reason=f"INVALID_RETURN_PATH:{exc}",
        )
    observation_count = int(values.size)
    if observation_count < policy.minimum_observations:
        return _invalid_result(
            test="STATIONARY_BOOTSTRAP_MONTE_CARLO",
            observation_count=observation_count,
            policy=policy,
            reason="INSUFFICIENT_OBSERVATIONS",
        )

    rng = np.random.default_rng(policy.seed + seed_offset)
    restart_probability = min(1.0, 1.0 / float(policy.expected_block_length))
    terminal_returns = np.empty(policy.simulations, dtype=float)
    maximum_drawdowns = np.empty(policy.simulations, dtype=float)

    written = 0
    while written < policy.simulations:
        batch = min(policy.batch_size, policy.simulations - written)
        indices = np.empty((batch, observation_count), dtype=np.int64)
        indices[:, 0] = rng.integers(0, observation_count, size=batch)
        for column in range(1, observation_count):
            restart = rng.random(batch) < restart_probability
            fresh = rng.integers(0, observation_count, size=batch)
            continued = (indices[:, column - 1] + 1) % observation_count
            indices[:, column] = np.where(restart, fresh, continued)

        sampled = values[indices]
        equity = np.cumprod(1.0 + sampled, axis=1)
        peaks = np.maximum.accumulate(
            np.concatenate((np.ones((batch, 1)), equity), axis=1),
            axis=1,
        )[:, 1:]
        drawdowns = 1.0 - np.divide(
            equity,
            peaks,
            out=np.zeros_like(equity),
            where=peaks > 0.0,
        )
        terminal_returns[written : written + batch] = equity[:, -1] - 1.0
        maximum_drawdowns[written : written + batch] = np.max(drawdowns, axis=1)
        written += batch

    terminal_loss_events = int(np.sum(terminal_returns <= 0.0))
    drawdown_breach_events = int(
        np.sum(maximum_drawdowns > policy.maximum_drawdown)
    )
    terminal_loss_probability = terminal_loss_events / policy.simulations
    breach_probability = drawdown_breach_events / policy.simulations
    terminal_loss_probability_upper = _wilson_upper_probability(
        terminal_loss_events,
        policy.simulations,
        confidence_level=policy.confidence_level,
    )
    breach_probability_upper = _wilson_upper_probability(
        drawdown_breach_events,
        policy.simulations,
        confidence_level=policy.confidence_level,
    )
    p05_total_return, median_total_return, p95_total_return = (
        float(value) for value in np.quantile(terminal_returns, (0.05, 0.50, 0.95))
    )
    median_drawdown, p95_drawdown = (
        float(value) for value in np.quantile(maximum_drawdowns, (0.50, 0.95))
    )
    checks = {
        "terminal_loss_probability": (
            terminal_loss_probability_upper <= policy.maximum_terminal_loss_probability
        ),
        "p05_total_return": p05_total_return >= policy.minimum_p05_total_return,
        "drawdown_breach_probability": (
            breach_probability_upper <= policy.maximum_drawdown_breach_probability
        ),
    }
    return {
        "test": "STATIONARY_BOOTSTRAP_MONTE_CARLO",
        "version": STOCHASTIC_VALIDATION_VERSION,
        "policy_hash": policy.policy_hash,
        "observation_count": observation_count,
        "simulations": policy.simulations,
        "seed": policy.seed + seed_offset,
        "expected_block_length": policy.expected_block_length,
        "maximum_drawdown_threshold": policy.maximum_drawdown,
        "terminal_loss_probability": terminal_loss_probability,
        "terminal_loss_probability_upper_confidence_bound": (
            terminal_loss_probability_upper
        ),
        "maximum_drawdown_breach_probability": breach_probability,
        "maximum_drawdown_breach_probability_upper_confidence_bound": (
            breach_probability_upper
        ),
        "probability_confidence_level": policy.confidence_level,
        "p05_total_return": p05_total_return,
        "median_total_return": median_total_return,
        "p95_total_return": p95_total_return,
        "median_maximum_drawdown": median_drawdown,
        "p95_maximum_drawdown": p95_drawdown,
        "checks": checks,
        "passed": all(checks.values()),
        "reason_codes": [
            name.upper() + "_GATE_FAILED" for name, passed in checks.items() if not passed
        ],
    }


def dirichlet_time_concentration_stress(
    returns: Iterable[float] | np.ndarray,
    *,
    policy: StochasticValidationPolicy,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Stress whether total growth depends on a small set of time blocks.

    Each simulation draws weights over chronological blocks. Concentration
    alpha 0.5 emphasizes regime dependence; alpha 5.0 approaches balanced
    history. Growth is calculated in log space and mapped back to arithmetic
    total return over the original sample length.
    """

    try:
        values = _validated_returns(returns)
    except ValueError as exc:
        return _invalid_result(
            test="DIRICHLET_TIME_CONCENTRATION_STRESS",
            observation_count=0,
            policy=policy,
            reason=f"INVALID_RETURN_PATH:{exc}",
        )
    observation_count = int(values.size)
    if observation_count < policy.minimum_observations:
        return _invalid_result(
            test="DIRICHLET_TIME_CONCENTRATION_STRESS",
            observation_count=observation_count,
            policy=policy,
            reason="INSUFFICIENT_OBSERVATIONS",
        )

    block_count = min(
        policy.dirichlet_blocks,
        max(4, observation_count // 5),
    )
    blocks = np.array_split(np.log1p(values), block_count)
    block_log_growth_rates = np.array(
        [float(np.mean(block)) for block in blocks],
        dtype=float,
    )
    profiles: list[dict[str, Any]] = []
    all_profiles_passed = True

    for profile_index, concentration in enumerate(policy.dirichlet_concentrations):
        rng = np.random.default_rng(policy.seed + seed_offset + 10_000 + profile_index)
        weights = rng.dirichlet(
            np.full(block_count, concentration, dtype=float),
            size=policy.simulations,
        )
        simulated_log_growth = (weights @ block_log_growth_rates) * observation_count
        simulated_total_returns = np.expm1(np.clip(simulated_log_growth, -745.0, 709.0))
        terminal_loss_events = int(np.sum(simulated_total_returns <= 0.0))
        terminal_loss_probability = terminal_loss_events / policy.simulations
        terminal_loss_probability_upper = _wilson_upper_probability(
            terminal_loss_events,
            policy.simulations,
            confidence_level=policy.confidence_level,
        )
        p05, median, p95 = (
            float(value) for value in np.quantile(simulated_total_returns, (0.05, 0.50, 0.95))
        )
        checks = {
            "terminal_loss_probability": (
                terminal_loss_probability_upper
                <= policy.maximum_terminal_loss_probability
            ),
            "p05_total_return": p05 >= policy.minimum_p05_total_return,
        }
        profile_passed = all(checks.values())
        all_profiles_passed = all_profiles_passed and profile_passed
        profiles.append(
            {
                "concentration_alpha": concentration,
                "terminal_loss_probability": terminal_loss_probability,
                "terminal_loss_probability_upper_confidence_bound": (
                    terminal_loss_probability_upper
                ),
                "probability_positive": 1.0 - terminal_loss_probability,
                "p05_total_return": p05,
                "median_total_return": median,
                "p95_total_return": p95,
                "mean_weight_herfindahl": float(np.mean(np.sum(weights**2, axis=1))),
                "checks": checks,
                "passed": profile_passed,
            }
        )

    return {
        "test": "DIRICHLET_TIME_CONCENTRATION_STRESS",
        "version": STOCHASTIC_VALIDATION_VERSION,
        "policy_hash": policy.policy_hash,
        "observation_count": observation_count,
        "simulations_per_profile": policy.simulations,
        "seed": policy.seed + seed_offset,
        "chronological_block_count": block_count,
        "block_observation_counts": [len(block) for block in blocks],
        "profiles": profiles,
        "passed": all_profiles_passed,
        "reason_codes": (
            [] if all_profiles_passed else ["ONE_OR_MORE_DIRICHLET_CONCENTRATION_PROFILES_FAILED"]
        ),
    }


def validate_strategy_return_paths(
    normal_returns: Iterable[float] | np.ndarray,
    stressed_returns: Iterable[float] | np.ndarray,
    *,
    policy: StochasticValidationPolicy,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Apply both independent gates to normal and stressed-cost return paths."""

    normal_monte_carlo = stationary_bootstrap_monte_carlo(
        normal_returns,
        policy=policy,
        seed_offset=seed_offset,
    )
    normal_dirichlet = dirichlet_time_concentration_stress(
        normal_returns,
        policy=policy,
        seed_offset=seed_offset,
    )
    stressed_monte_carlo = stationary_bootstrap_monte_carlo(
        stressed_returns,
        policy=policy,
        seed_offset=seed_offset + 1_000_000,
    )
    stressed_dirichlet = dirichlet_time_concentration_stress(
        stressed_returns,
        policy=policy,
        seed_offset=seed_offset + 1_000_000,
    )
    checks = {
        "normal_monte_carlo": bool(normal_monte_carlo["passed"]),
        "normal_dirichlet": bool(normal_dirichlet["passed"]),
        "stressed_monte_carlo": bool(stressed_monte_carlo["passed"]),
        "stressed_dirichlet": bool(stressed_dirichlet["passed"]),
    }
    return {
        "version": STOCHASTIC_VALIDATION_VERSION,
        "policy": asdict(policy),
        "policy_hash": policy.policy_hash,
        "normal": {
            "monte_carlo": normal_monte_carlo,
            "dirichlet": normal_dirichlet,
        },
        "stressed": {
            "monte_carlo": stressed_monte_carlo,
            "dirichlet": stressed_dirichlet,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "reason_codes": [name.upper() + "_FAILED" for name, passed in checks.items() if not passed],
        "interpretation": (
            "Independent fail-closed robustness evidence. These simulations do "
            "not change strategy DNA, p-values, DSR, PBO or the known-trial count."
        ),
    }


def policy_from_research_settings(
    research: Any,
    *,
    seed: int,
    expected_block_length: int,
) -> StochasticValidationPolicy:
    """Build the immutable gate policy without coupling this module to Pydantic."""

    return StochasticValidationPolicy(
        simulations=int(research.monte_carlo_runs),
        expected_block_length=expected_block_length,
        maximum_drawdown=float(research.maximum_drawdown),
        maximum_drawdown_breach_probability=float(
            research.maximum_monte_carlo_probability_of_20pct_drawdown
        ),
        maximum_terminal_loss_probability=float(research.maximum_dirichlet_probability_of_loss),
        minimum_p05_total_return=float(research.minimum_stochastic_p05_total_return),
        dirichlet_blocks=int(research.dirichlet_block_count),
        confidence_level=float(research.confidence_level),
        seed=seed,
    )


__all__ = [
    "STOCHASTIC_VALIDATION_VERSION",
    "StochasticValidationPolicy",
    "dirichlet_time_concentration_stress",
    "policy_from_research_settings",
    "stationary_bootstrap_monte_carlo",
    "validate_strategy_return_paths",
]
