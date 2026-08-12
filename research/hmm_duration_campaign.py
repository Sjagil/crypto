"""Duration-aware HMM validation and bounded observer-only HPO.

The supplied HMM posterior is not made "sticky" by multiplying every state by
the same scalar.  Instead, this campaign evaluates a finite explicit-duration
forward filter over the expanded ``(state, age)`` state space.  Hyperparameters
are selected only from out-of-sample predictive folds.  Trading returns are
never an HPO objective and the selected candidate receives no paper/live
authority.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from config.settings import Settings
from research.global_trial_accounting import audit_global_trial_accounting
from research.hmm_regime_campaign import ALLOWED_MARKETS
from research.hmm_regime_manager import (
    FEATURE_COLUMNS,
    HMM_TIMEFRAMES,
    HSMM_DURATION_ENGINE_VERSION,
    MINIMUM_TRAINING,
    ExplicitDurationHSMMFilter,
    HMMFitSnapshot,
    InstitutionalHMMRegimeManager,
    causal_hmm_features,
    causal_market_context,
    emission_log_likelihood,
)
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    stable_hash,
    utc_now,
)

plt.switch_backend("Agg")


HMM_DURATION_HPO_CAMPAIGN_ID = "hmm_duration_hpo_v1"
DEFAULT_EXPECTED_DWELL = {
    "15m": 96,
    "1h": 72,
    "4h": 48,
    "1d": 14,
    "1W": 8,
}
VALIDATION_OBSERVATIONS = {
    "15m": 672,
    "1h": 336,
    "4h": 168,
    "1d": 90,
    "1W": 26,
}
MAXIMUM_FEATURE_OBSERVATIONS = {
    "15m": 5_000,
    "1h": 4_000,
    "4h": 3_000,
    "1d": 2_500,
    "1W": 1_200,
}


@dataclass(frozen=True)
class PreparedFold:
    """Immutable emissions and OOS base diagnostics for one fold."""

    timeframe: str
    fold: int
    snapshot: HMMFitSnapshot
    training_log_likelihoods: np.ndarray
    validation_log_likelihoods: np.ndarray
    validation_index: pd.DatetimeIndex
    base_negative_log_predictive_density: float
    base_churn: float
    base_switch_rate: float
    base_mean_entropy: float
    base_occupancy_entropy: float


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    state_count = values.shape[1]
    return (
        -np.sum(
            np.clip(values, 1e-15, 1.0)
            * np.log(np.clip(values, 1e-15, 1.0)),
            axis=1,
        )
        / math.log(state_count)
    )


def _path_diagnostics(
    probabilities: np.ndarray,
    log_predictive_density: Iterable[float],
) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=float)
    logscore = np.asarray(tuple(log_predictive_density), dtype=float)
    dominant = np.argmax(values, axis=1)
    occupancy = np.bincount(
        dominant,
        minlength=values.shape[1],
    ).astype(float)
    occupancy /= max(1.0, float(occupancy.sum()))
    occupancy_entropy = float(
        -np.sum(
            np.clip(occupancy, 1e-15, 1.0)
            * np.log(np.clip(occupancy, 1e-15, 1.0))
        )
        / math.log(values.shape[1])
    )
    if len(values) <= 1:
        churn = 0.0
        switch_rate = 0.0
    else:
        # Total-variation distance is bounded in [0, 1].
        churn = float(
            np.mean(np.sum(np.abs(np.diff(values, axis=0)), axis=1) / 2.0)
        )
        switch_rate = float(np.mean(dominant[1:] != dominant[:-1]))
    return {
        "negative_log_predictive_density": float(-np.mean(logscore)),
        "churn": churn,
        "switch_rate": switch_rate,
        "mean_entropy": float(np.mean(_normalized_entropy(values))),
        "occupancy_entropy": occupancy_entropy,
        "effective_states": float(
            math.exp(
                -np.sum(
                    np.clip(occupancy, 1e-15, 1.0)
                    * np.log(np.clip(occupancy, 1e-15, 1.0))
                )
            )
        ),
    }


def _base_forward_path(
    snapshot: HMMFitSnapshot,
    training_log_likelihoods: np.ndarray,
    validation_log_likelihoods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous: np.ndarray | None = None
    for log_likelihood in training_log_likelihoods:
        # Reconstruct an observation-independent forward update from the
        # already calculated emissions.
        prior = (
            snapshot.start_probability
            if previous is None
            else previous @ snapshot.transition_matrix
        )
        normalized = np.log(np.maximum(prior, 1e-300)) + log_likelihood
        normalized -= logsumexp(normalized)
        previous = np.exp(normalized)
    probabilities: list[np.ndarray] = []
    logscore: list[float] = []
    for log_likelihood in validation_log_likelihoods:
        prior = (
            snapshot.start_probability
            if previous is None
            else previous @ snapshot.transition_matrix
        )
        joint = np.log(np.maximum(prior, 1e-300)) + log_likelihood
        logscore.append(float(logsumexp(joint)))
        normalized = joint - logsumexp(joint)
        previous = np.exp(normalized)
        probabilities.append(previous.copy())
    return np.vstack(probabilities), np.asarray(logscore, dtype=float)


def _duration_forward_path(
    fold: PreparedFold,
    *,
    expected_dwell: int,
    maximum_duration_factor: int,
    transition_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_duration = max(
        2,
        int(math.ceil(expected_dwell * maximum_duration_factor)),
    )
    engine = ExplicitDurationHSMMFilter(
        start_probability=fold.snapshot.start_probability,
        transition_matrix=fold.snapshot.transition_matrix,
        expected_durations=float(expected_dwell),
        maximum_duration=maximum_duration,
        transition_shrinkage=transition_shrinkage,
    )
    for log_likelihood in fold.training_log_likelihoods:
        engine.step(log_likelihood)
    probabilities: list[np.ndarray] = []
    logscore: list[float] = []
    for log_likelihood in fold.validation_log_likelihoods:
        step = engine.step(log_likelihood)
        probabilities.append(step.state_probability)
        logscore.append(step.log_predictive_density)
    return np.vstack(probabilities), np.asarray(logscore, dtype=float)


def _load_features(settings: Settings, timeframe: str) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for market in ALLOWED_MARKETS:
        path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"HMM_DURATION_REAL_DATA_MISSING:{path}")
        frames[market] = pd.read_parquet(path)
        hashes[market] = sha256_file(path)
    breadth, correlation = causal_market_context(frames)
    features = causal_hmm_features(
        frames["BTC-EUR"],
        breadth=breadth,
        average_correlation=correlation,
    )
    maximum = MAXIMUM_FEATURE_OBSERVATIONS[timeframe]
    return features.iloc[-maximum:].copy(), hashes


def _prepare_folds(
    settings: Settings,
    timeframes: tuple[str, ...],
    *,
    fold_count: int,
) -> tuple[list[PreparedFold], dict[str, dict[str, str]]]:
    manager = InstitutionalHMMRegimeManager(settings.hmm_regime)
    prepared: list[PreparedFold] = []
    hashes: dict[str, dict[str, str]] = {}
    for timeframe in timeframes:
        features, hashes[timeframe] = _load_features(settings, timeframe)
        validation_size = VALIDATION_OBSERVATIONS[timeframe]
        minimum = MINIMUM_TRAINING[timeframe]
        required = minimum + fold_count * validation_size
        if len(features) < required:
            raise ValueError(
                f"HMM_DURATION_INSUFFICIENT_HISTORY:{timeframe}:"
                f"{len(features)}<{required}"
            )
        for fold_number in range(fold_count):
            remaining = (fold_count - fold_number - 1) * validation_size
            validation_end = len(features) - remaining
            validation_start = validation_end - validation_size
            training = features.iloc[:validation_start]
            validation = features.iloc[validation_start:validation_end]
            snapshot = manager.fit(training, timeframe=timeframe)
            snapshot_training = training.loc[
                snapshot.training_started_at : snapshot.fitted_through,
                FEATURE_COLUMNS,
            ].to_numpy(dtype=float)
            validation_values = validation.loc[:, FEATURE_COLUMNS].to_numpy(
                dtype=float
            )
            training_log = np.vstack(
                [
                    emission_log_likelihood(snapshot, observation)
                    for observation in snapshot_training
                ]
            )
            validation_log = np.vstack(
                [
                    emission_log_likelihood(snapshot, observation)
                    for observation in validation_values
                ]
            )
            base_probability, base_logscore = _base_forward_path(
                snapshot,
                training_log,
                validation_log,
            )
            base = _path_diagnostics(base_probability, base_logscore)
            prepared.append(
                PreparedFold(
                    timeframe=timeframe,
                    fold=fold_number + 1,
                    snapshot=snapshot,
                    training_log_likelihoods=training_log,
                    validation_log_likelihoods=validation_log,
                    validation_index=validation.index,
                    base_negative_log_predictive_density=base[
                        "negative_log_predictive_density"
                    ],
                    base_churn=base["churn"],
                    base_switch_rate=base["switch_rate"],
                    base_mean_entropy=base["mean_entropy"],
                    base_occupancy_entropy=base["occupancy_entropy"],
                )
            )
    return prepared, hashes


def evaluate_duration_candidate(
    prepared: Iterable[PreparedFold],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one frozen duration candidate on all OOS folds."""

    rows: list[dict[str, Any]] = []
    for fold in prepared:
        probability, logscore = _duration_forward_path(
            fold,
            expected_dwell=int(parameters[f"expected_dwell_{fold.timeframe}"]),
            maximum_duration_factor=int(parameters["maximum_duration_factor"]),
            transition_shrinkage=float(parameters["transition_shrinkage"]),
        )
        metrics = _path_diagnostics(probability, logscore)
        rows.append(
            {
                "timeframe": fold.timeframe,
                "fold": fold.fold,
                **metrics,
                "base_negative_log_predictive_density": (
                    fold.base_negative_log_predictive_density
                ),
                "base_churn": fold.base_churn,
                "base_switch_rate": fold.base_switch_rate,
                "base_mean_entropy": fold.base_mean_entropy,
                "base_occupancy_entropy": fold.base_occupancy_entropy,
            }
        )
    frame = pd.DataFrame(rows)
    occupancy_entropy = float(frame["occupancy_entropy"].mean())
    collapse_penalty = max(0.0, 0.35 - occupancy_entropy) * 10.0
    return {
        "negative_log_predictive_density": float(
            frame["negative_log_predictive_density"].mean()
        ),
        "base_negative_log_predictive_density": float(
            frame["base_negative_log_predictive_density"].mean()
        ),
        "predictive_nll_delta_vs_hmm": float(
            (
                frame["negative_log_predictive_density"]
                - frame["base_negative_log_predictive_density"]
            ).mean()
        ),
        "churn": float(frame["churn"].mean()),
        "base_churn": float(frame["base_churn"].mean()),
        "churn_delta_vs_hmm": float(
            (frame["churn"] - frame["base_churn"]).mean()
        ),
        "switch_rate": float(frame["switch_rate"].mean()),
        "base_switch_rate": float(frame["base_switch_rate"].mean()),
        "switch_rate_delta_vs_hmm": float(
            (frame["switch_rate"] - frame["base_switch_rate"]).mean()
        ),
        "mean_entropy": float(frame["mean_entropy"].mean()),
        "base_mean_entropy": float(frame["base_mean_entropy"].mean()),
        "occupancy_entropy": occupancy_entropy,
        "collapse_penalty": collapse_penalty,
        "noncollapsed": bool(occupancy_entropy >= 0.35),
        "folds": rows,
    }


def _parameter_ranges(
    trial: Any,
    timeframes: tuple[str, ...],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "maximum_duration_factor": trial.suggest_int(
            "maximum_duration_factor",
            3,
            5,
        ),
        "transition_shrinkage": trial.suggest_float(
            "transition_shrinkage",
            0.0,
            0.5,
            step=0.1,
        ),
    }
    for timeframe in timeframes:
        default = DEFAULT_EXPECTED_DWELL[timeframe]
        parameters[f"expected_dwell_{timeframe}"] = trial.suggest_int(
            f"expected_dwell_{timeframe}",
            max(2, default // 2),
            default * 2,
        )
    return parameters


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def _chart(
    path: Path,
    trials: pd.DataFrame,
    selected_number: int,
    *,
    selected_accepted: bool,
) -> None:
    selected_color = "#1b7f3a" if selected_accepted else "#e08b18"
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    colors = np.where(
        trials["trial_number"].eq(selected_number),
        selected_color,
        np.where(trials["noncollapsed"], "#2878b5", "#c7362f"),
    )
    axes[0].scatter(
        trials["churn"],
        trials["negative_log_predictive_density"],
        c=colors,
        alpha=0.8,
    )
    outcome = "accepted" if selected_accepted else "rejected"
    axes[0].set_title(
        "Duration HPO: OOS predictive loss versus posterior churn "
        f"(diagnostic best {outcome})"
    )
    axes[0].set_xlabel("Mean total-variation churn (lower is calmer)")
    axes[0].set_ylabel("Negative log predictive density (lower is better)")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(
        trials["switch_rate"],
        trials["occupancy_entropy"],
        c=colors,
        alpha=0.8,
    )
    axes[1].axhline(0.35, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_title("Collapse guard: state occupancy versus switches")
    axes[1].set_xlabel("Dominant-state switch rate")
    axes[1].set_ylabel("Normalized occupancy entropy")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _apply_predictive_acceptance_gate(
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    selection = dict(result["selection"])
    metrics = dict(selection["selected_metrics"])
    margin = max(
        0.01,
        abs(float(metrics["base_negative_log_predictive_density"])) * 0.01,
    )
    predictive_noninferior = bool(
        float(metrics["predictive_nll_delta_vs_hmm"]) <= margin
    )
    accepted = bool(
        predictive_noninferior
        and metrics["noncollapsed"]
        and float(metrics["churn_delta_vs_hmm"]) <= 0.0
    )
    selection.update(
        {
            "predictive_noninferiority_margin": margin,
            "predictive_noninferior": predictive_noninferior,
            "accepted_candidate": accepted,
            "status": (
                str(selection["status"])
                if accepted
                else "NO_PREDICTIVELY_NONINFERIOR_DURATION_CANDIDATE"
            ),
        }
    )
    result["selection"] = selection
    result["status"] = (
        "COMPLETED_OBSERVER_ONLY_ACCEPTED_DIAGNOSTIC"
        if accepted
        else "COMPLETED_OBSERVER_ONLY_NO_ACCEPTED_CANDIDATE"
    )
    return result


def _write_human_reports(
    payload: dict[str, Any],
    trials_frame: pd.DataFrame,
    *,
    markdown_path: Path,
    html_path: Path,
    chart_path: Path,
) -> None:
    selection = payload["selection"]
    selected_number = int(selection["selected_trial_number"])
    summary_frame = trials_frame.sort_values(
        ["negative_log_predictive_density", "churn"],
    ).head(20)
    summary_columns = [
        "trial_number",
        "negative_log_predictive_density",
        "predictive_nll_delta_vs_hmm",
        "churn",
        "churn_delta_vs_hmm",
        "switch_rate",
        "occupancy_entropy",
        "noncollapsed",
    ]
    markdown = "\n\n".join(
        [
            "# Duration-aware HMM/HSMM HPO v1",
            (
                "Observer-only. Parameters are selected on out-of-sample "
                "predictive folds; trading performance is not optimized."
            ),
            (
                f"Diagnostic-best trial: `{selected_number}` "
                f"({selection['status']}). Accepted: "
                f"**{selection['accepted_candidate']}**."
            ),
            _markdown_table(summary_frame.loc[:, summary_columns].round(6)),
            "Orders generated: **0**. Live authority changed: **no**.",
        ]
    )
    atomic_write_text(markdown_path, markdown)
    atomic_write_text(
        html_path,
        (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Duration-aware HMM HPO</title></head><body>"
            "<h1>Duration-aware HMM/HSMM HPO v1</h1>"
            "<p>Observer-only; OOS predictive selection; zero order authority.</p>"
            f"<p>Diagnostic-best trial: <code>{selected_number}</code> "
            f"({html.escape(str(selection['status']))}). "
            "Accepted: "
            f"<strong>{selection['accepted_candidate']}</strong>.</p>"
            f"{summary_frame.loc[:, summary_columns].round(6).to_html(index=False)}"
            f"<img src='{html.escape(chart_path.name)}' alt='HMM duration HPO chart'>"
            "</body></html>"
        ),
    )


def run_hmm_duration_hpo(
    settings: Settings,
    *,
    timeframes: tuple[str, ...] = HMM_TIMEFRAMES,
    trials: int = 20,
    folds: int = 3,
) -> dict[str, Any]:
    """Run bounded, deterministic, OOS HPO with zero execution authority."""

    if not settings.hmm_regime.enabled or not settings.hmm_regime.observer_only:
        raise RuntimeError("HMM_DURATION_REQUIRES_OBSERVER_ONLY_MODE")
    if trials < 4 or trials > 250:
        raise ValueError("HMM duration HPO trials must be between 4 and 250")
    if folds < 2 or folds > 5:
        raise ValueError("HMM duration HPO folds must be between 2 and 5")
    normalized = tuple(dict.fromkeys(timeframes))
    if not normalized or any(value not in HMM_TIMEFRAMES for value in normalized):
        raise ValueError("unsupported HMM duration HPO timeframe")

    output = settings.paths.output_dir / "hmm"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "hmm_duration_hpo_v1.csv"
    json_path = reports / "hmm_duration_hpo_v1.json"
    markdown_path = reports / "hmm_duration_hpo_v1.md"
    html_path = reports / "hmm_duration_hpo_v1.html"
    chart_path = reports / "hmm_duration_hpo_v1.png"
    if json_path.is_file() and csv_path.is_file():
        existing = dict(read_json(json_path))
        identity_matches = (
            existing.get("campaign_id") == HMM_DURATION_HPO_CAMPAIGN_ID
            and tuple(existing.get("timeframes") or ()) == normalized
            and int(existing.get("fold_count") or 0) == folds
            and int(existing.get("trial_count") or 0) == trials
        )
        current_hashes = {
            timeframe: {
                market: sha256_file(
                    settings.paths.processed_data_dir
                    / f"{market}_{timeframe}.parquet"
                )
                for market in ALLOWED_MARKETS
            }
            for timeframe in normalized
        }
        if identity_matches and current_hashes == existing.get("data_hashes"):
            reconciled = _apply_predictive_acceptance_gate(existing)
            trials_frame = pd.read_csv(csv_path)
            atomic_write_json(json_path, reconciled)
            _write_human_reports(
                reconciled,
                trials_frame,
                markdown_path=markdown_path,
                html_path=html_path,
                chart_path=chart_path,
            )
            _chart(
                chart_path,
                trials_frame,
                int(reconciled["selection"]["selected_trial_number"]),
                selected_accepted=bool(
                    reconciled["selection"]["accepted_candidate"]
                ),
            )
            return reconciled

    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError("Optuna is required for HMM duration HPO") from exc

    prepared, data_hashes = _prepare_folds(
        settings,
        normalized,
        fold_count=folds,
    )
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: Any) -> tuple[float, float]:
        parameters = _parameter_ranges(trial, normalized)
        metrics = evaluate_duration_candidate(prepared, parameters)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("metrics", metrics)
        # Predictive quality is primary. Collapse cannot be rewarded merely
        # because a static state has low churn.
        return (
            float(metrics["negative_log_predictive_density"]),
            float(metrics["churn"] + metrics["collapse_penalty"]),
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=("minimize", "minimize"),
        sampler=optuna.samplers.NSGAIISampler(
            population_size=min(20, trials),
            seed=settings.hmm_regime.random_seed,
        ),
    )
    study.optimize(objective, n_trials=trials, n_jobs=1)
    for trial in study.trials:
        metrics = dict(trial.user_attrs["metrics"])
        parameters = dict(trial.user_attrs["parameters"])
        trial_rows.append(
            {
                "trial_number": int(trial.number),
                **parameters,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "folds"
                },
                "objective_predictive": float(trial.values[0]),
                "objective_persistence": float(trial.values[1]),
            }
        )
    trials_frame = pd.DataFrame(trial_rows)
    predictive_best = float(
        trials_frame["negative_log_predictive_density"].min()
    )
    plateau_tolerance = max(0.01, abs(predictive_best) * 0.01)
    plateau = trials_frame.loc[
        trials_frame["noncollapsed"]
        & (
            trials_frame["negative_log_predictive_density"]
            <= predictive_best + plateau_tolerance
        )
    ]
    selection_status = "SELECTED_NONCOLLAPSED_PREDICTIVE_PLATEAU"
    if plateau.empty:
        plateau = trials_frame.nsmallest(
            1,
            "negative_log_predictive_density",
        )
        selection_status = "FALLBACK_PREDICTIVE_BEST_COLLAPSE_WARNING"
    selected_row = plateau.sort_values(
        ["churn", "negative_log_predictive_density", "mean_entropy"],
        ascending=[True, True, True],
    ).iloc[0]
    selected_number = int(selected_row["trial_number"])
    selected_trial = study.trials[selected_number]
    selected_parameters = dict(selected_trial.user_attrs["parameters"])
    selected_metrics = dict(selected_trial.user_attrs["metrics"])

    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / HMM_DURATION_HPO_CAMPAIGN_ID,
        campaign_id=HMM_DURATION_HPO_CAMPAIGN_ID,
    )
    data_fingerprint = stable_hash(
        {
            "data_hashes": data_hashes,
            "timeframes": normalized,
            "folds": folds,
            "validation_observations": {
                timeframe: VALIDATION_OBSERVATIONS[timeframe]
                for timeframe in normalized
            },
            "feature_columns": FEATURE_COLUMNS,
        },
        length=64,
    )
    registrations: list[dict[str, Any]] = []
    for trial in study.trials:
        parameters = dict(trial.user_attrs["parameters"])
        metrics = dict(trial.user_attrs["metrics"])
        dna = stable_hash(
            {
                "campaign": HMM_DURATION_HPO_CAMPAIGN_ID,
                "engine_version": HSMM_DURATION_ENGINE_VERSION,
                "parameters": parameters,
                "timeframes": normalized,
                "objective": "OOS_PREDICTIVE_NLL_AND_CHURN_WITH_COLLAPSE_GUARD",
            },
            length=64,
        )
        registrations.append(
            registry.register(
                data_fingerprint=data_fingerprint,
                strategy_family="HMM_DURATION_OBSERVER_HPO",
                strategy_dna_hash=dna,
                parameters=parameters,
                metrics_at_birth={
                    key: value
                    for key, value in metrics.items()
                    if key != "folds"
                },
                return_path_hash=stable_hash(
                    metrics["folds"],
                    length=64,
                ),
                selection_metadata={
                    "trial_number": int(trial.number),
                    "observer_only": True,
                    "trading_return_not_an_objective": True,
                    "selected": bool(trial.number == selected_number),
                    "orders_generated": 0,
                },
            )
        )
    registry_audit = registry.audit()
    global_accounting = audit_global_trial_accounting(
        settings.paths.lab_dir,
        persist=True,
    )

    trials_frame.to_csv(csv_path, index=False)
    payload = _apply_predictive_acceptance_gate({
        "schema_version": "hmm_duration_hpo_report_v1",
        "campaign_id": HMM_DURATION_HPO_CAMPAIGN_ID,
        "generated_at": utc_now().isoformat(),
        "status": "COMPLETED_OBSERVER_ONLY",
        "hypothesis": (
            "Explicit state-duration hazards may reduce regime churn without "
            "sacrificing out-of-sample predictive density."
        ),
        "mathematics": {
            "duration": "D_i = 1 + Poisson(mu_i - 1)",
            "hazard": "h_i(d) = P(D_i=d) / P(D_i>=d)",
            "expanded_state": "(regime, age)",
            "posterior": "P(S_t, age_t | x_1:t)",
            "smoothed_posterior_forbidden": "P(S_t | x_1:T)",
        },
        "selection": {
            "status": selection_status,
            "selected_trial_number": selected_number,
            "selected_parameters": selected_parameters,
            "selected_metrics": selected_metrics,
            "predictive_plateau_tolerance": plateau_tolerance,
            "trading_returns_used_as_objective": False,
            "automatic_strategy_promotion": False,
        },
        "timeframes": list(normalized),
        "fold_count": folds,
        "trial_count": trials,
        "data_hashes": data_hashes,
        "data_fingerprint": data_fingerprint,
        "trials": trial_rows,
        "trial_registrations": registrations,
        "trial_registry": registry_audit,
        "global_trial_accounting": global_accounting,
        "authority": {
            "observer_only": True,
            "orders_generated": 0,
            "orders_submitted": 0,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "live_authority_changed": False,
            "tao_npc_authority_added": False,
        },
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "html": str(html_path),
            "chart": str(chart_path),
        },
    })
    atomic_write_json(json_path, payload)
    _chart(
        chart_path,
        trials_frame,
        selected_number,
        selected_accepted=bool(
            payload["selection"]["accepted_candidate"]
        ),
    )
    _write_human_reports(
        payload,
        trials_frame,
        markdown_path=markdown_path,
        html_path=html_path,
        chart_path=chart_path,
    )
    return payload


def hmm_duration_hpo_status(settings: Settings) -> dict[str, Any]:
    path = (
        settings.paths.output_dir
        / "hmm"
        / "reports"
        / "hmm_duration_hpo_v1.json"
    )
    if not path.is_file():
        return {
            "status": "NOT_RUN",
            "campaign_id": HMM_DURATION_HPO_CAMPAIGN_ID,
            "observer_only": True,
            "orders_generated": 0,
            "live_ready": False,
        }
    report = dict(read_json(path))
    selected_metrics = dict(
        report.get("selection", {}).get("selected_metrics") or {}
    )
    selected_metrics.pop("folds", None)
    return {
        "status": report.get("status"),
        "campaign_id": report.get("campaign_id"),
        "selected_trial_number": report.get("selection", {}).get(
            "selected_trial_number"
        ),
        "selected_parameters": report.get("selection", {}).get(
            "selected_parameters"
        ),
        "selected_metrics": selected_metrics,
        "trial_registry": report.get("trial_registry"),
        "global_multiple_testing_denominator": report.get(
            "global_trial_accounting",
            {},
        ).get("global_multiple_testing_denominator"),
        "report": str(path),
        "observer_only": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "live_ready": False,
    }


__all__ = [
    "DEFAULT_EXPECTED_DWELL",
    "HMM_DURATION_HPO_CAMPAIGN_ID",
    "PreparedFold",
    "evaluate_duration_candidate",
    "hmm_duration_hpo_status",
    "run_hmm_duration_hpo",
]
