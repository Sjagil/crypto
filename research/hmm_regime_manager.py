"""Causal multi-timeframe Hidden Markov regime inference.

The HMM is a regime controller, never an entry strategy or execution authority.
Models are fitted only on observations strictly before the observation being
classified.  Posterior probabilities are produced with an explicit forward
filter (``P(S_t | x_1:t)``); full-sample smoothed probabilities are forbidden.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from config.settings import TIMEFRAME_SECONDS, HMMRegimeSettings
from utils.common import stable_hash

HMM_ENGINE_VERSION = "1.1.0"
HSMM_DURATION_ENGINE_VERSION = "1.0.0"
HMM_TIMEFRAMES = ("1W", "1d", "4h", "1h", "15m")
FEATURE_COLUMNS = (
    "log_return",
    "realized_volatility",
    "atr_fraction",
    "trend_slope",
    "range_efficiency",
    "volume_zscore",
    "breadth",
    "average_cross_asset_correlation",
)
STATE_COUNTS = {"1W": 3, "1d": 4, "4h": 4, "1h": 4, "15m": 3}
MINIMUM_TRAINING = {"1W": 104, "1d": 365, "4h": 500, "1h": 720, "15m": 1_000}


@dataclass(frozen=True)
class HMMFitSnapshot:
    """Immutable parameters needed for causal forward inference."""

    timeframe: str
    fitted_through: pd.Timestamp
    training_started_at: pd.Timestamp
    feature_columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    start_probability: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    state_labels: tuple[str, ...]
    converged: bool
    iterations: int
    model_hash: str


@dataclass(frozen=True)
class HMMInference:
    """Causal probability path and diagnostics for one timeframe."""

    timeframe: str
    probabilities: pd.DataFrame
    dominant_state: pd.Series
    posterior_entropy: pd.Series
    risk_multiplier: pd.Series
    expected_duration: dict[str, float]
    current_forecasts: dict[str, dict[str, float]]
    fit_history: tuple[dict[str, Any], ...]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class DurationFilterStep:
    """One normalized explicit-duration forward-filter update."""

    state_probability: np.ndarray
    state_age_probability: np.ndarray
    log_predictive_density: float
    dominant_state: int
    dominant_state_age: int


def shifted_poisson_duration_distribution(
    expected_duration: float,
    maximum_duration: int,
) -> np.ndarray:
    """Return ``P(D=d)`` for ``D=1+Poisson(expected_duration-1)``.

    The final bucket includes the right tail.  This makes the finite support a
    proper probability distribution and guarantees a terminal hazard of one.
    ``expected_duration`` is a mean, not a hard minimum.
    """

    mean = float(expected_duration)
    maximum = int(maximum_duration)
    if not math.isfinite(mean) or mean < 1.0:
        raise ValueError("expected HSMM duration must be finite and at least one")
    if maximum < 2:
        raise ValueError("maximum HSMM duration must be at least two")
    poisson_mean = mean - 1.0
    probabilities = np.zeros(maximum, dtype=float)
    probability = math.exp(-poisson_mean)
    for offset in range(maximum - 1):
        if offset:
            probability *= poisson_mean / float(offset)
        probabilities[offset] = probability
    probabilities[-1] = max(0.0, 1.0 - float(probabilities[:-1].sum()))
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid shifted-Poisson duration distribution")
    return probabilities / total


def duration_hazard(probability_mass: np.ndarray) -> np.ndarray:
    """Convert duration probability mass to the discrete exit hazard."""

    mass = np.asarray(probability_mass, dtype=float)
    if mass.ndim != 1 or len(mass) < 2:
        raise ValueError("duration probability mass must be one-dimensional")
    if bool((mass < 0.0).any()) or not bool(np.isfinite(mass).all()):
        raise ValueError("duration probability mass must be finite and non-negative")
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("duration probability mass is empty")
    mass = mass / total
    survival = np.flip(np.cumsum(np.flip(mass)))
    hazard = np.divide(
        mass,
        survival,
        out=np.ones_like(mass),
        where=survival > 1e-15,
    )
    hazard[-1] = 1.0
    return np.clip(hazard, 0.0, 1.0)


class ExplicitDurationHSMMFilter:
    """Causal explicit-duration forward filter over ``(state, age)``.

    This is a diagnostic HSMM filter, not a trading signal.  The diagonal of
    the supplied transition matrix is removed because persistence is governed
    by the explicit duration hazard rather than by a geometric self-transition.
    """

    def __init__(
        self,
        *,
        start_probability: np.ndarray,
        transition_matrix: np.ndarray,
        expected_durations: float | np.ndarray,
        maximum_duration: int,
        transition_shrinkage: float = 0.0,
    ) -> None:
        start = np.asarray(start_probability, dtype=float)
        transition = np.asarray(transition_matrix, dtype=float)
        if start.ndim != 1 or transition.shape != (len(start), len(start)):
            raise ValueError("HSMM start/transition dimensions do not reconcile")
        if (
            not bool(np.isfinite(start).all())
            or not bool(np.isfinite(transition).all())
            or bool((start < 0.0).any())
            or bool((transition < 0.0).any())
        ):
            raise ValueError("HSMM probabilities must be finite and non-negative")
        if not 0.0 <= float(transition_shrinkage) <= 1.0:
            raise ValueError("HSMM transition shrinkage must be in [0, 1]")
        start_total = float(start.sum())
        if start_total <= 0.0:
            raise ValueError("HSMM start probability is empty")
        self.start_probability = start / start_total
        self.state_count = len(start)
        self.maximum_duration = int(maximum_duration)
        durations = np.asarray(expected_durations, dtype=float)
        if durations.ndim == 0:
            durations = np.repeat(float(durations), self.state_count)
        if durations.shape != (self.state_count,):
            raise ValueError("HSMM expected durations must match the state count")
        self.expected_durations = durations
        self.duration_probability = np.vstack(
            [
                shifted_poisson_duration_distribution(
                    float(duration),
                    self.maximum_duration,
                )
                for duration in durations
            ]
        )
        self.hazard = np.vstack(
            [duration_hazard(row) for row in self.duration_probability]
        )

        off_diagonal = transition.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        uniform = np.ones_like(off_diagonal) - np.eye(self.state_count)
        uniform /= np.maximum(uniform.sum(axis=1, keepdims=True), 1.0)
        row_total = off_diagonal.sum(axis=1, keepdims=True)
        off_diagonal = np.divide(
            off_diagonal,
            row_total,
            out=uniform.copy(),
            where=row_total > 1e-15,
        )
        shrinkage = float(transition_shrinkage)
        self.off_diagonal_transition = (
            (1.0 - shrinkage) * off_diagonal + shrinkage * uniform
        )
        self._state_age_probability: np.ndarray | None = None

    @property
    def state_age_probability(self) -> np.ndarray | None:
        """Return a defensive copy of the current expanded posterior."""

        if self._state_age_probability is None:
            return None
        return self._state_age_probability.copy()

    def reset(self) -> None:
        self._state_age_probability = None

    def _prior(self) -> np.ndarray:
        if self._state_age_probability is None:
            prior = np.zeros(
                (self.state_count, self.maximum_duration),
                dtype=float,
            )
            prior[:, 0] = self.start_probability
            return prior

        previous = self._state_age_probability
        prior = np.zeros_like(previous)
        exit_mass = previous * self.hazard
        stay_mass = previous * (1.0 - self.hazard)
        prior[:, 1:] = stay_mass[:, :-1]
        prior[:, 0] = (
            exit_mass.sum(axis=1) @ self.off_diagonal_transition
        )
        return prior

    def step(self, emission_log_likelihood: np.ndarray) -> DurationFilterStep:
        """Consume one closed-candle emission vector without future data."""

        log_likelihood = np.asarray(emission_log_likelihood, dtype=float)
        if log_likelihood.shape != (self.state_count,) or not bool(
            np.isfinite(log_likelihood).all()
        ):
            raise ValueError("HSMM emission likelihood has invalid shape or values")
        prior = self._prior()
        maximum = float(np.max(log_likelihood))
        emission = np.exp(log_likelihood - maximum)
        unnormalized = prior * emission[:, None]
        normalizer = float(unnormalized.sum())
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise RuntimeError("HSMM forward normalization failed")
        posterior = unnormalized / normalizer
        self._state_age_probability = posterior
        state_probability = posterior.sum(axis=1)
        dominant_state = int(np.argmax(state_probability))
        dominant_age = int(np.argmax(posterior[dominant_state])) + 1
        return DurationFilterStep(
            state_probability=state_probability.copy(),
            state_age_probability=posterior.copy(),
            log_predictive_density=float(maximum + math.log(normalizer)),
            dominant_state=dominant_state,
            dominant_state_age=dominant_age,
        )


def _validated_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"HMM OHLCV missing columns: {sorted(missing)}")
    result = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("HMM OHLCV requires a DatetimeIndex")
    if result.index.tz is None:
        raise ValueError("HMM OHLCV timestamps must be timezone-aware")
    result.index = result.index.tz_convert("UTC")
    result = result.sort_index()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("HMM OHLCV timestamps must be unique and increasing")
    result = result.apply(pd.to_numeric, errors="coerce")
    if bool((result[["open", "high", "low", "close"]] <= 0.0).any().any()):
        raise ValueError("HMM prices must be positive")
    if bool((result["volume"] < 0.0).any()):
        raise ValueError("HMM volume cannot be negative")
    return result


def causal_market_context(
    frames: Mapping[str, pd.DataFrame],
    *,
    benchmark_market: str = "BTC-EUR",
    correlation_lookback: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """Return closed-candle breadth and average correlation without filling gaps."""

    closes = {
        market: _validated_ohlcv(frame)["close"]
        for market, frame in frames.items()
    }
    # Calculate returns on each asset's own native close sequence before
    # cross-asset alignment.  Differencing an outer-joined panel would insert
    # artificial NaNs when one provider stamps weekly bars on another weekday.
    returns = pd.concat(
        {
            market: np.log(series.where(series > 0.0)).diff()
            for market, series in closes.items()
        },
        axis=1,
    ).sort_index()
    positive = returns.gt(0.0)
    available = returns.notna().sum(axis=1).replace(0, np.nan)
    breadth = positive.sum(axis=1).div(available)

    benchmark = benchmark_market.upper().replace("/", "-").replace("_", "-")
    if benchmark not in returns:
        correlation = pd.Series(np.nan, index=returns.index)
    else:
        rolling = [
            returns[column].rolling(
                correlation_lookback,
                min_periods=max(10, correlation_lookback // 2),
            ).corr(returns[benchmark])
            for column in returns.columns
            if column != benchmark
        ]
        correlation = (
            pd.concat(rolling, axis=1).mean(axis=1)
            if rolling
            else pd.Series(np.nan, index=returns.index)
        )
    breadth.name = "breadth"
    correlation.name = "average_cross_asset_correlation"
    return breadth, correlation


def causal_hmm_features(
    frame: pd.DataFrame,
    *,
    breadth: pd.Series | None = None,
    average_correlation: pd.Series | None = None,
    volatility_lookback: int = 20,
    trend_lookback: int = 20,
) -> pd.DataFrame:
    """Build a stationary, closed-candle-only HMM feature matrix."""

    ohlcv = _validated_ohlcv(frame)
    close = ohlcv["close"]
    log_close = np.log(close)
    log_return = log_close.diff()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            ohlcv["high"] - ohlcv["low"],
            (ohlcv["high"] - previous_close).abs(),
            (ohlcv["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    realized = log_return.rolling(
        volatility_lookback,
        min_periods=volatility_lookback,
    ).std(ddof=0)
    atr_fraction = true_range.rolling(
        volatility_lookback,
        min_periods=volatility_lookback,
    ).mean().div(close)
    trend_slope = log_close.diff(trend_lookback).div(trend_lookback)
    directional = close.diff(trend_lookback).abs()
    travelled = close.diff().abs().rolling(
        trend_lookback,
        min_periods=trend_lookback,
    ).sum()
    range_efficiency = directional.div(travelled.replace(0.0, np.nan))
    volume_mean = ohlcv["volume"].rolling(
        volatility_lookback,
        min_periods=volatility_lookback,
    ).mean()
    volume_std = ohlcv["volume"].rolling(
        volatility_lookback,
        min_periods=volatility_lookback,
    ).std(ddof=0)
    volume_zscore = ohlcv["volume"].sub(volume_mean).div(
        volume_std.replace(0.0, np.nan)
    )
    result = pd.DataFrame(
        {
            "log_return": log_return,
            "realized_volatility": realized,
            "atr_fraction": atr_fraction,
            "trend_slope": trend_slope,
            "range_efficiency": range_efficiency,
            "volume_zscore": volume_zscore,
            "breadth": (
                breadth.reindex(ohlcv.index)
                if breadth is not None
                else pd.Series(0.5, index=ohlcv.index)
            ),
            "average_cross_asset_correlation": (
                average_correlation.reindex(ohlcv.index)
                if average_correlation is not None
                else pd.Series(0.0, index=ohlcv.index)
            ),
        },
        index=ohlcv.index,
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    lower = np.nanpercentile(values, 25.0, axis=0)
    upper = np.nanpercentile(values, 75.0, axis=0)
    scale = upper - lower
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
    transformed = np.clip((values - center) / scale, -8.0, 8.0)
    return transformed, center, scale


def _economic_state_labels(
    timeframe: str,
    means: np.ndarray,
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    direction = (
        means[:, feature_columns.index("log_return")]
        + means[:, feature_columns.index("trend_slope")]
    )
    volatility = (
        means[:, feature_columns.index("realized_volatility")]
        + means[:, feature_columns.index("atr_fraction")]
    )
    volume = means[:, feature_columns.index("volume_zscore")]
    labels = [""] * len(means)
    available = set(range(len(means)))

    def assign(index: int, label: str) -> None:
        labels[index] = label
        available.discard(index)

    def available_extreme(values: np.ndarray, *, maximum: bool) -> int:
        if not available:
            raise RuntimeError("no HMM state remains available")
        return int(
            max(available, key=lambda index: values[index])
            if maximum
            else min(available, key=lambda index: values[index])
        )

    if timeframe == "1W":
        assign(available_extreme(direction, maximum=True), "STRUCTURAL_BULL")
        assign(available_extreme(direction, maximum=False), "STRUCTURAL_RISK_OFF")
        for index in tuple(available):
            assign(index, "STRUCTURAL_RANGE")
    elif timeframe == "1d":
        assign(available_extreme(direction, maximum=True), "TREND_UP")
        assign(available_extreme(direction, maximum=False), "TREND_DOWN")
        remaining = sorted(available, key=lambda index: volatility[index])
        assign(remaining[0], "LOW_VOL_RANGE")
        assign(remaining[-1], "HIGH_VOL_TRANSITION")
    elif timeframe in {"4h", "1h"}:
        assign(available_extreme(direction, maximum=True), "CONTINUATION")
        assign(available_extreme(direction, maximum=False), "PULLBACK")
        remaining = sorted(available, key=lambda index: volatility[index])
        assign(remaining[0], "MEAN_REVERSION")
        assign(remaining[-1], "VOLATILITY_EXPANSION")
    elif timeframe == "15m":
        volume_state = available_extreme(volume, maximum=True)
        assign(volume_state, "LIQUIDITY_VOLUME_CONFIRMATION")
        remaining = sorted(available, key=lambda index: abs(direction[index]), reverse=True)
        assign(remaining[0], "ENTRY_TIMING")
        assign(remaining[-1], "MICRO_NOISE")
    else:
        raise ValueError(f"unsupported HMM timeframe: {timeframe}")
    if any(not label for label in labels):
        raise RuntimeError("HMM state labelling did not cover every state")
    return tuple(labels)


def _diag_variances(model: GaussianHMM) -> np.ndarray:
    raw = np.asarray(model._covars_, dtype=float)  # noqa: SLF001
    if raw.ndim == 3:
        raw = np.diagonal(raw, axis1=1, axis2=2)
    if raw.ndim != 2:
        raise ValueError("unexpected diagonal HMM covariance shape")
    return np.maximum(raw, 1e-8)


def _emission_log_likelihood(snapshot: HMMFitSnapshot, observation: np.ndarray) -> np.ndarray:
    scaled = np.clip((observation - snapshot.center) / snapshot.scale, -8.0, 8.0)
    delta = scaled[None, :] - snapshot.means
    return -0.5 * (
        np.sum(np.log(2.0 * math.pi * snapshot.variances), axis=1)
        + np.sum((delta * delta) / snapshot.variances, axis=1)
    )


def emission_log_likelihood(
    snapshot: HMMFitSnapshot,
    observation: np.ndarray,
) -> np.ndarray:
    """Public, read-only emission kernel for observer validation."""

    return _emission_log_likelihood(snapshot, observation)


def _forward_step(
    snapshot: HMMFitSnapshot,
    observation: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    prior = (
        snapshot.start_probability
        if previous is None
        else previous @ snapshot.transition_matrix
    )
    log_alpha = np.log(np.maximum(prior, 1e-300)) + _emission_log_likelihood(
        snapshot,
        observation,
    )
    log_alpha -= logsumexp(log_alpha)
    return np.exp(log_alpha)


def forward_probability_step(
    snapshot: HMMFitSnapshot,
    observation: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    """Public causal HMM forward step used by validation campaigns."""

    return _forward_step(snapshot, observation, previous)


class InstitutionalHMMRegimeManager:
    """Fit and evaluate a hierarchy of causal HMM regime models."""

    def __init__(self, settings: HMMRegimeSettings) -> None:
        self.settings = settings
        if not settings.observer_only:
            raise ValueError("HMM manager has no execution-enabled mode")

    def _refit_interval(self, timeframe: str) -> int:
        return {
            "15m": self.settings.refit_interval_15m,
            "1h": self.settings.refit_interval_1h,
            "4h": self.settings.refit_interval_4h,
            "1d": self.settings.refit_interval_1d,
            "1W": self.settings.refit_interval_1w,
        }[timeframe]

    def fit(
        self,
        features: pd.DataFrame,
        *,
        timeframe: str,
    ) -> HMMFitSnapshot:
        if timeframe not in HMM_TIMEFRAMES:
            raise ValueError(f"unsupported HMM timeframe: {timeframe}")
        selected = features.loc[:, FEATURE_COLUMNS].dropna()
        minimum = MINIMUM_TRAINING[timeframe]
        if len(selected) < minimum:
            raise ValueError(
                f"insufficient HMM training history for {timeframe}: "
                f"{len(selected)} < {minimum}"
            )
        selected = selected.iloc[-self.settings.maximum_training_observations :]
        scaled, center, scale = _robust_scale(selected.to_numpy(dtype=float))
        model = GaussianHMM(
            n_components=STATE_COUNTS[timeframe],
            covariance_type="diag",
            n_iter=self.settings.maximum_iterations,
            tol=self.settings.convergence_tolerance,
            min_covar=1e-6,
            random_state=self.settings.random_seed,
            params="stmc",
            init_params="stmc",
        )
        hmm_logger = logging.getLogger("hmmlearn.base")
        previous_level = hmm_logger.level
        try:
            hmm_logger.setLevel(logging.ERROR)
            model.fit(scaled)
        finally:
            hmm_logger.setLevel(previous_level)
        transition = np.asarray(model.transmat_, dtype=float)
        transition = transition / transition.sum(axis=1, keepdims=True)
        means = np.asarray(model.means_, dtype=float)
        variances = _diag_variances(model)
        labels = _economic_state_labels(
            timeframe,
            means,
            tuple(selected.columns),
        )
        identity = {
            "engine_version": HMM_ENGINE_VERSION,
            "timeframe": timeframe,
            "fitted_through": selected.index[-1].isoformat(),
            "training_started_at": selected.index[0].isoformat(),
            "feature_columns": list(selected.columns),
            "start_probability": model.startprob_.tolist(),
            "transition_matrix": transition.tolist(),
            "means": means.tolist(),
            "variances": variances.tolist(),
            "state_labels": labels,
        }
        return HMMFitSnapshot(
            timeframe=timeframe,
            fitted_through=selected.index[-1],
            training_started_at=selected.index[0],
            feature_columns=tuple(selected.columns),
            center=center,
            scale=scale,
            start_probability=np.asarray(model.startprob_, dtype=float),
            transition_matrix=transition,
            means=means,
            variances=variances,
            state_labels=labels,
            converged=bool(model.monitor_.converged),
            iterations=int(model.monitor_.iter),
            model_hash=stable_hash(identity, length=64),
        )

    def walk_forward(
        self,
        features: pd.DataFrame,
        *,
        timeframe: str,
    ) -> HMMInference:
        """Classify each eligible observation using strictly prior fitted data."""

        selected = features.loc[:, FEATURE_COLUMNS].dropna().copy()
        minimum = MINIMUM_TRAINING[timeframe]
        if len(selected) <= minimum:
            raise ValueError(f"insufficient HMM inference history for {timeframe}")
        probabilities = pd.DataFrame(
            np.nan,
            index=selected.index,
            columns=sorted(
                {
                    label
                    for label_set in (
                        ("STRUCTURAL_BULL", "STRUCTURAL_RANGE", "STRUCTURAL_RISK_OFF"),
                        ("TREND_UP", "TREND_DOWN", "LOW_VOL_RANGE", "HIGH_VOL_TRANSITION"),
                        ("CONTINUATION", "PULLBACK", "MEAN_REVERSION", "VOLATILITY_EXPANSION"),
                        ("ENTRY_TIMING", "LIQUIDITY_VOLUME_CONFIRMATION", "MICRO_NOISE"),
                    )
                    for label in label_set
                }
            ),
            dtype=float,
        )
        snapshot: HMMFitSnapshot | None = None
        previous: np.ndarray | None = None
        fit_history: list[dict[str, Any]] = []
        interval = self._refit_interval(timeframe)
        last_refit = -interval
        values = selected.to_numpy(dtype=float)

        for position in range(minimum, len(selected)):
            if snapshot is None or position - last_refit >= interval:
                snapshot = self.fit(selected.iloc[:position], timeframe=timeframe)
                training = selected.loc[
                    snapshot.training_started_at : snapshot.fitted_through
                ].to_numpy(dtype=float)
                previous = None
                for observation in training:
                    previous = _forward_step(snapshot, observation, previous)
                last_refit = position
                fit_history.append(
                    {
                        "inference_starts_at": selected.index[position].isoformat(),
                        "fitted_through": snapshot.fitted_through.isoformat(),
                        "training_started_at": snapshot.training_started_at.isoformat(),
                        "model_hash": snapshot.model_hash,
                        "state_labels": list(snapshot.state_labels),
                        "converged": snapshot.converged,
                        "iterations": snapshot.iterations,
                        "strictly_prior_to_inference": bool(
                            snapshot.fitted_through < selected.index[position]
                        ),
                    }
                )
            assert snapshot is not None
            previous = _forward_step(snapshot, values[position], previous)
            for state, label in enumerate(snapshot.state_labels):
                probabilities.iat[position, probabilities.columns.get_loc(label)] = previous[state]

        probabilities = probabilities.dropna(how="all").fillna(0.0)
        row_total = probabilities.sum(axis=1)
        probabilities = probabilities.div(row_total.replace(0.0, np.nan), axis=0).dropna()
        dominant = probabilities.idxmax(axis=1).rename("dominant_state")
        entropy = (
            -(probabilities.clip(lower=self.settings.probability_floor)
              * np.log(probabilities.clip(lower=self.settings.probability_floor))).sum(axis=1)
            / math.log(STATE_COUNTS[timeframe])
        ).rename("posterior_entropy")
        risk = self._risk_multiplier(probabilities, timeframe=timeframe)
        assert snapshot is not None
        durations = {
            label: float(1.0 / max(1e-9, 1.0 - snapshot.transition_matrix[state, state]))
            for state, label in enumerate(snapshot.state_labels)
        }
        current_vector = np.array(
            [probabilities.iloc[-1].get(label, 0.0) for label in snapshot.state_labels]
        )
        forecasts: dict[str, dict[str, float]] = {}
        for horizon in (1, 3, 5, 10):
            forecast = current_vector @ np.linalg.matrix_power(
                snapshot.transition_matrix,
                horizon,
            )
            forecasts[str(horizon)] = {
                label: float(forecast[state])
                for state, label in enumerate(snapshot.state_labels)
            }
        integrity = {
            "filtered_not_smoothed": True,
            "strictly_prior_model_fit": all(
                row["strictly_prior_to_inference"] for row in fit_history
            ),
            "closed_candle_features": True,
            "backward_asof_required": True,
            "stationary_features_only": True,
            "diagonal_covariance": True,
            "economic_label_set_stable": len(
                {
                    tuple(sorted(row["state_labels"]))
                    for row in fit_history
                }
            )
            == 1,
            "observer_only": True,
            "orders_generated": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
        }
        return HMMInference(
            timeframe=timeframe,
            probabilities=probabilities,
            dominant_state=dominant,
            posterior_entropy=entropy,
            risk_multiplier=risk,
            expected_duration=durations,
            current_forecasts=forecasts,
            fit_history=tuple(fit_history),
            integrity=integrity,
        )

    @staticmethod
    def _risk_multiplier(probabilities: pd.DataFrame, *, timeframe: str) -> pd.Series:
        if timeframe == "1W":
            value = (
                probabilities.get("STRUCTURAL_BULL", 0.0)
                + 0.55 * probabilities.get("STRUCTURAL_RANGE", 0.0)
                + 0.10 * probabilities.get("STRUCTURAL_RISK_OFF", 0.0)
            )
        elif timeframe == "1d":
            value = (
                probabilities.get("TREND_UP", 0.0)
                + 0.65 * probabilities.get("LOW_VOL_RANGE", 0.0)
                + 0.25 * probabilities.get("HIGH_VOL_TRANSITION", 0.0)
                + 0.10 * probabilities.get("TREND_DOWN", 0.0)
            )
        elif timeframe in {"4h", "1h"}:
            value = (
                probabilities.get("CONTINUATION", 0.0)
                + 0.70 * probabilities.get("MEAN_REVERSION", 0.0)
                + 0.45 * probabilities.get("PULLBACK", 0.0)
                + 0.30 * probabilities.get("VOLATILITY_EXPANSION", 0.0)
            )
        else:
            value = (
                0.85 * probabilities.get("ENTRY_TIMING", 0.0)
                + probabilities.get("LIQUIDITY_VOLUME_CONFIRMATION", 0.0)
                + 0.25 * probabilities.get("MICRO_NOISE", 0.0)
            )
        return pd.Series(value, index=probabilities.index, name="risk_multiplier").clip(0.0, 1.0)

    def backward_asof_align(
        self,
        target_index: pd.DatetimeIndex,
        inferences: Mapping[str, HMMInference],
        *,
        target_timeframe: str,
        target_is_available_at: bool = False,
    ) -> pd.DataFrame:
        """Align states using only source candles closed by the decision time.

        Feature-frame indices identify candle opens, so their decision timestamp
        becomes available one target interval later. Portfolio execution paths
        already identify an executable instant and set ``target_is_available_at``.
        """

        if target_index.tz is None:
            raise ValueError("target HMM timestamps must be timezone-aware")
        if target_timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"unsupported HMM target timeframe: {target_timeframe}")
        target_offset = timedelta(
            seconds=int(TIMEFRAME_SECONDS[target_timeframe])
        )
        decision_available_at = target_index.tz_convert("UTC")
        if not target_is_available_at:
            decision_available_at = decision_available_at + target_offset
        target = pd.DataFrame(
            {
                "timestamp": target_index.tz_convert("UTC"),
                "decision_available_at": decision_available_at,
            }
        ).sort_values("decision_available_at")
        aligned = pd.DataFrame(index=target_index)
        for timeframe, inference in inferences.items():
            if timeframe not in TIMEFRAME_SECONDS:
                raise ValueError(f"unsupported HMM source timeframe: {timeframe}")
            source_offset = timedelta(
                seconds=int(TIMEFRAME_SECONDS[timeframe])
            )
            source = pd.DataFrame(
                {
                    "source_timestamp": inference.risk_multiplier.index,
                    "available_at": inference.risk_multiplier.index + source_offset,
                    f"{timeframe}_risk_multiplier": inference.risk_multiplier.to_numpy(),
                    f"{timeframe}_state": inference.dominant_state.to_numpy(),
                    f"{timeframe}_entropy": inference.posterior_entropy.to_numpy(),
                }
            ).sort_values("available_at")
            merged = pd.merge_asof(
                target,
                source,
                left_on="decision_available_at",
                right_on="available_at",
                direction="backward",
                allow_exact_matches=True,
            ).set_index("timestamp")
            for column in source.columns:
                if column not in {"available_at", "source_timestamp"}:
                    aligned[column] = merged[column].reindex(aligned.index)
            aligned[f"{timeframe}_available_at"] = merged["available_at"].reindex(
                aligned.index
            )
            aligned[f"{timeframe}_source_timestamp"] = merged[
                "source_timestamp"
            ].reindex(aligned.index)
        aligned["decision_available_at"] = target.set_index("timestamp")[
            "decision_available_at"
        ].reindex(aligned.index)
        return aligned

    def blended_risk_multiplier(self, aligned: pd.DataFrame) -> pd.Series:
        """Hierarchical blend; uncertainty reduces rather than increases risk."""

        weights = {"1W": 0.30, "1d": 0.30, "4h": 0.18, "1h": 0.14, "15m": 0.08}
        numerator = pd.Series(0.0, index=aligned.index)
        denominator = pd.Series(0.0, index=aligned.index)
        entropy_penalty = pd.Series(0.0, index=aligned.index)
        for timeframe, weight in weights.items():
            risk_column = f"{timeframe}_risk_multiplier"
            entropy_column = f"{timeframe}_entropy"
            if risk_column not in aligned:
                continue
            valid = aligned[risk_column].notna()
            numerator = numerator.add(aligned[risk_column].fillna(0.0) * weight)
            denominator = denominator.add(valid.astype(float) * weight)
            if entropy_column in aligned:
                entropy_penalty = entropy_penalty.add(
                    aligned[entropy_column].fillna(1.0) * weight
                )
        blended = numerator.div(denominator.replace(0.0, np.nan))
        uncertainty = entropy_penalty.div(denominator.replace(0.0, np.nan))
        result = (blended * (1.0 - 0.35 * uncertainty)).clip(0.0, 1.0)
        return result.rename("hmm_blended_risk_multiplier")

    def apply_risk_multiplier(
        self,
        target_weights: pd.DataFrame,
        multiplier: pd.Series,
    ) -> pd.DataFrame:
        """Scale long-only targets while preserving the 40/20/60 hard limits."""

        numeric = target_weights.apply(pd.to_numeric, errors="coerce")
        if bool(numeric.isna().any().any()) or bool((numeric < -1e-12).any().any()):
            raise ValueError("HMM target weights must be finite and long-only")
        scaled = numeric.mul(
            multiplier.reindex(numeric.index).fillna(0.0).clip(0.0, 1.0),
            axis=0,
        ).clip(lower=0.0, upper=self.settings.maximum_asset_exposure)
        exposure = scaled.sum(axis=1)
        over = exposure > self.settings.maximum_total_exposure
        if bool(over.any()):
            scaled.loc[over] = scaled.loc[over].mul(
                self.settings.maximum_total_exposure / exposure.loc[over],
                axis=0,
            )
        if bool(
            (
                scaled.sum(axis=1)
                > 1.0 - self.settings.minimum_cash_fraction + 1e-12
            ).any()
        ):
            raise RuntimeError("HMM minimum-cash invariant failed")
        return scaled


__all__ = [
    "DurationFilterStep",
    "ExplicitDurationHSMMFilter",
    "FEATURE_COLUMNS",
    "HMM_ENGINE_VERSION",
    "HMM_TIMEFRAMES",
    "HSMM_DURATION_ENGINE_VERSION",
    "HMMFitSnapshot",
    "HMMInference",
    "InstitutionalHMMRegimeManager",
    "causal_hmm_features",
    "causal_market_context",
    "duration_hazard",
    "emission_log_likelihood",
    "forward_probability_step",
    "shifted_poisson_duration_distribution",
]
