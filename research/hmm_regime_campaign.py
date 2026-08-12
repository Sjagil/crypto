"""Orderless HMM regime observer and frozen RR comparison campaign."""

from __future__ import annotations

import html
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from config.settings import Settings
from research.global_trial_accounting import audit_global_trial_accounting
from research.hmm_regime_manager import (
    HMM_ENGINE_VERSION,
    HMM_TIMEFRAMES,
    MINIMUM_TRAINING,
    HMMInference,
    InstitutionalHMMRegimeManager,
    causal_hmm_features,
    causal_market_context,
)
from research.portfolio_selection import RotationPortfolioPolicy, _validated_panel
from research.residual_reversal import (
    ResidualReversalParameters,
    backtest_residual_reversal,
)
from research.stochastic_validation import (
    StochasticValidationPolicy,
    validate_strategy_return_paths,
)
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
    utc_iso,
)

# v1 was registered before the numerical reproducibility quantizer was added.
# Keep that immutable evidence and use a new campaign identity for the amended,
# deterministic matrix.
HMM_CAMPAIGN_ID = "hmm_regime_controller_v1_3"
RR_STRATEGY_ID = "RR_B60_H5_Z20"
RR_STRATEGY_DNA = "4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0"
ALLOWED_MARKETS = ("BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR")
HMM_POLICY_MATRIX = (
    {
        "policy_id": "HMM_SOFT_40",
        "floor": 0.40,
        "slope": 0.60,
        "gate": None,
        "hypothesis": "HMM uncertainty scales exposure but never fully suppresses RR.",
    },
    {
        "policy_id": "HMM_BALANCED_15",
        "floor": 0.15,
        "slope": 0.85,
        "gate": None,
        "hypothesis": "A stronger causal regime multiplier improves downside efficiency.",
    },
    {
        "policy_id": "HMM_GATE_45",
        "floor": 0.0,
        "slope": 1.0,
        "gate": 0.45,
        "hypothesis": "RR is active only when the preregistered HMM score is at least 0.45.",
    },
)
HMM_LAYER_ROLES = {
    "1W": "Structural capital/risk-off overlay; never an intrabar entry.",
    "1d": "Daily swing regime and family enable/size input.",
    "4h": "Swing continuation, pullback and volatility condition.",
    "1h": "Active swing setup confirmation and exit-risk context.",
    "15m": "Entry timing and liquidity/volume confirmation; micro-noise filter.",
}


def _load_frames(settings: Settings, timeframe: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for market in ALLOWED_MARKETS:
        path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"HMM_REAL_DATA_MISSING:{path}")
        frames[market] = pd.read_parquet(path)
    return frames


def _performance(
    equity: pd.Series,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
) -> dict[str, Any]:
    years = max(
        (equity.index[-1] - equity.index[0]).total_seconds()
        / (365.25 * 24.0 * 3_600.0),
        1.0 / 365.25,
    )
    net_return = float(equity.iloc[-1] - 1.0)
    annualized = (
        float((1.0 + net_return) ** (1.0 / years) - 1.0)
        if net_return > -1.0
        else -1.0
    )
    standard = float(returns.std(ddof=0))
    downside = float(returns.where(returns < 0.0, 0.0).std(ddof=0))
    gains = float(returns.where(returns > 0.0, 0.0).sum())
    losses = float(abs(returns.where(returns < 0.0, 0.0).sum()))
    drawdown = equity.div(equity.cummax()).sub(1.0)
    maximum_drawdown = float(drawdown.min())
    return {
        "net_return": net_return,
        "annualized_return": annualized,
        "portfolio_period_profit_factor": (
            gains / losses if losses > 0.0 else float("inf")
        ),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(365.25))
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(returns.mean() / downside * math.sqrt(365.25))
            if downside > 0.0
            else 0.0
        ),
        "maximum_drawdown": maximum_drawdown,
        "calmar": (
            annualized / abs(maximum_drawdown)
            if maximum_drawdown < 0.0
            else 0.0
        ),
        "average_exposure": float(weights.sum(axis=1).mean()),
        "maximum_exposure": float(weights.sum(axis=1).max()),
        "maximum_asset_exposure": float(weights.max(axis=1).max()),
        "turnover": float(turnover.sum()),
        "observations": int(len(returns)),
        "positive_periods": int((returns > 0.0).sum()),
        "negative_periods": int((returns < 0.0).sum()),
    }


def _weighted_path(
    frames: dict[str, pd.DataFrame],
    base_weights: pd.DataFrame,
    multiplier: pd.Series,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    policy: RotationPortfolioPolicy,
) -> dict[str, Any]:
    opens, closes = _validated_panel(
        frames,
        benchmark_market="BTC-EUR",
        portfolio_policy=policy,
    )
    weights = base_weights.mul(
        multiplier.reindex(base_weights.index).fillna(0.0),
        axis=0,
    ).clip(lower=0.0, upper=policy.maximum_position_exposure)
    exposure = weights.sum(axis=1)
    if float(exposure.max()) > policy.maximum_total_exposure + 1e-12:
        raise RuntimeError("HMM_TOTAL_EXPOSURE_BREACH")
    open_returns = opens.shift(-1).div(opens).sub(1.0)
    open_returns.iloc[-1] = closes.iloc[-1].div(opens.iloc[-1]).sub(1.0)
    open_returns = open_returns.reindex(weights.index).fillna(0.0)
    gross_returns = (weights * open_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    turnover.iloc[-1] += float(weights.iloc[-1].sum())
    one_way_cost = (
        fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    )
    net_returns = (1.0 - turnover * one_way_cost) * (1.0 + gross_returns) - 1.0
    equity = (1.0 + net_returns).cumprod()
    return {
        "metrics": _performance(equity, net_returns, weights, turnover),
        "equity": equity,
        "returns": net_returns,
        "weights": weights,
        "turnover": turnover,
        "integrity": {
            "maximum_total_exposure_respected": bool(
                exposure.max() <= policy.maximum_total_exposure + 1e-12
            ),
            "maximum_asset_exposure_respected": bool(
                weights.max(axis=1).max()
                <= policy.maximum_position_exposure + 1e-12
            ),
            "minimum_cash_respected": bool(
                exposure.max() <= 1.0 - policy.minimum_cash + 1e-12
            ),
            "long_only": bool((weights >= -1e-12).all().all()),
            "orders_generated": 0,
        },
    }


def _fold_consistency(returns: pd.Series, folds: int = 6) -> dict[str, Any]:
    parts = np.array_split(returns, folds)
    fold_rows = []
    for number, part in enumerate(parts, start=1):
        total = float((1.0 + part).prod() - 1.0)
        fold_rows.append(
            {
                "fold": number,
                "start": part.index[0].isoformat(),
                "end": part.index[-1].isoformat(),
                "net_return": total,
                "positive": total > 0.0,
            }
        )
    positives = sum(bool(row["positive"]) for row in fold_rows)
    return {
        "positive_folds": positives,
        "total_folds": len(fold_rows),
        "paper_rights_by_regime_consistency": positives >= 4,
        "folds": fold_rows,
    }


def _regime_attribution(
    returns: pd.Series,
    daily: HMMInference,
) -> dict[str, Any]:
    state = daily.dominant_state.reindex(returns.index, method="ffill").shift(1)
    rows: dict[str, Any] = {}
    for regime, values in returns.groupby(state):
        if not isinstance(regime, str) or values.empty:
            continue
        equity = (1.0 + values).cumprod()
        gains = float(values.where(values > 0.0, 0.0).sum())
        losses = float(abs(values.where(values < 0.0, 0.0).sum()))
        rows[regime] = {
            "observations": int(len(values)),
            "net_return": float(equity.iloc[-1] - 1.0),
            "profit_factor": gains / losses if losses > 0.0 else float("inf"),
            "sharpe": (
                float(values.mean() / values.std(ddof=0) * math.sqrt(365.25))
                if float(values.std(ddof=0)) > 0.0
                else 0.0
            ),
            "maximum_drawdown": float(
                equity.div(equity.cummax()).sub(1.0).min()
            ),
        }
    return rows


def _inference_summary(inference: HMMInference) -> dict[str, Any]:
    current_probabilities = {
        key: float(value)
        for key, value in inference.probabilities.iloc[-1].items()
        if float(value) > 1e-12
    }
    changes = inference.dominant_state.ne(
        inference.dominant_state.shift(1)
    )
    if not changes.empty:
        changes.iloc[0] = False
    return {
        "timeframe": inference.timeframe,
        "observations_classified": int(len(inference.probabilities)),
        "current_timestamp": inference.probabilities.index[-1].isoformat(),
        "current_state": str(inference.dominant_state.iloc[-1]),
        "current_probabilities": current_probabilities,
        "current_entropy": float(inference.posterior_entropy.iloc[-1]),
        "average_entropy": float(inference.posterior_entropy.mean()),
        "p95_entropy": float(inference.posterior_entropy.quantile(0.95)),
        "current_risk_multiplier": float(inference.risk_multiplier.iloc[-1]),
        "state_change_count": int(changes.sum()),
        "state_change_rate": float(changes.mean()),
        "expected_duration": inference.expected_duration,
        "k_step_forecasts": inference.current_forecasts,
        "fit_count": len(inference.fit_history),
        "latest_fit": inference.fit_history[-1],
        "integrity": inference.integrity,
    }


def _chart(
    output: Path,
    control: pd.Series,
    variants: dict[str, pd.Series],
    daily: HMMInference,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    axes[0].plot(control.index, control, label="RR control", linewidth=2.0)
    for name, equity in variants.items():
        axes[0].plot(equity.index, equity, label=name, alpha=0.85)
    axes[0].set_title("Frozen RR control versus causal HMM risk policies")
    axes[0].set_ylabel("Net equity (start = 1)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    visible_probabilities = daily.probabilities.loc[
        :,
        daily.probabilities.sum(axis=0) > 1e-9,
    ]
    visible_probabilities.plot.area(
        ax=axes[1],
        stacked=True,
        linewidth=0.0,
        alpha=0.8,
    )
    axes[1].set_title("Causal daily filtered HMM probabilities")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Probability")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            str(row.get(column, "")).replace("|", "\\|")
            for column in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def run_hmm_regime_campaign(settings: Settings) -> dict[str, Any]:
    """Run the preregistered HMM matrix and persist orderless evidence."""

    if not settings.hmm_regime.enabled:
        return {"status": "DISABLED", "orders_generated": 0}
    if not settings.hmm_regime.observer_only:
        raise RuntimeError("HMM_OBSERVER_AUTHORITY_VIOLATION")

    output = settings.paths.output_dir / "hmm"
    reports = output / "reports"
    posteriors = output / "posteriors"
    reports.mkdir(parents=True, exist_ok=True)
    posteriors.mkdir(parents=True, exist_ok=True)
    manager = InstitutionalHMMRegimeManager(settings.hmm_regime)
    inferences: dict[str, HMMInference] = {}
    data_hashes: dict[str, dict[str, str]] = {}

    for timeframe in HMM_TIMEFRAMES:
        frames = _load_frames(settings, timeframe)
        data_hashes[timeframe] = {
            market: sha256_file(
                settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            )
            for market in ALLOWED_MARKETS
        }
        breadth, correlation = causal_market_context(frames)
        features = causal_hmm_features(
            frames["BTC-EUR"],
            breadth=breadth,
            average_correlation=correlation,
        )
        # Daily and weekly paths are used in the historical comparison. Lower
        # layers are current causal observers in v1, not retroactive gates.
        lower_refit = {
            "15m": settings.hmm_regime.refit_interval_15m,
            "1h": settings.hmm_regime.refit_interval_1h,
            "4h": settings.hmm_regime.refit_interval_4h,
        }
        selected = (
            features
            if timeframe in {"1W", "1d"}
            else features.iloc[
                -(
                    MINIMUM_TRAINING[timeframe]
                    + 2 * lower_refit[timeframe]
                    + 1
                ) :
            ]
        )
        inference = manager.walk_forward(selected, timeframe=timeframe)
        inferences[timeframe] = inference
        posterior = inference.probabilities.copy()
        posterior["dominant_state"] = inference.dominant_state
        posterior["posterior_entropy"] = inference.posterior_entropy
        posterior["risk_multiplier"] = inference.risk_multiplier
        posterior.to_parquet(posteriors / f"{timeframe}_filtered.parquet")

    daily_frames = _load_frames(settings, "1d")
    policy = RotationPortfolioPolicy(
        allowed_markets=ALLOWED_MARKETS,
        maximum_total_exposure=settings.hmm_regime.maximum_total_exposure,
        maximum_position_exposure=settings.hmm_regime.maximum_asset_exposure,
        minimum_cash=settings.hmm_regime.minimum_cash_fraction,
        minimum_history_observations=200,
    )
    parameters = ResidualReversalParameters(
        beta_lookback=60,
        residual_horizon=5,
        entry_zscore=-2.0,
    )
    if parameters.dna_hash != RR_STRATEGY_DNA:
        raise RuntimeError("FROZEN_RR_DNA_MISMATCH")
    normal_control = backtest_residual_reversal(
        daily_frames,
        parameters,
        fee_rate=settings.costs.taker_fee,
        slippage_bps=settings.costs.slippage_bps,
        spread_bps=settings.costs.spread_bps,
        portfolio_policy=policy,
    )
    stressed_control = backtest_residual_reversal(
        daily_frames,
        parameters,
        fee_rate=settings.costs.taker_fee * 2.0,
        slippage_bps=settings.costs.slippage_bps * 2.0,
        spread_bps=settings.costs.spread_bps * 2.0,
        portfolio_policy=policy,
    )
    aligned = manager.backward_asof_align(
        normal_control.executed_weights.index,
        {"1W": inferences["1W"], "1d": inferences["1d"]},
        target_timeframe="1d",
        target_is_available_at=True,
    )
    base_multiplier = manager.blended_risk_multiplier(aligned).round(6)
    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir / "strategy_registry" / HMM_CAMPAIGN_ID,
        campaign_id=HMM_CAMPAIGN_ID,
    )
    data_fingerprint = stable_hash(data_hashes, length=64)
    results: list[dict[str, Any]] = []
    equity_paths: dict[str, pd.Series] = {}

    stochastic_policy = StochasticValidationPolicy(
        simulations=min(10_000, int(settings.research.monte_carlo_runs)),
        expected_block_length=10,
        maximum_drawdown=float(settings.research.maximum_drawdown),
        maximum_drawdown_breach_probability=float(
            settings.research.maximum_monte_carlo_probability_of_20pct_drawdown
        ),
        maximum_terminal_loss_probability=float(
            settings.research.maximum_dirichlet_probability_of_loss
        ),
        minimum_p05_total_return=float(
            settings.research.minimum_stochastic_p05_total_return
        ),
        dirichlet_blocks=int(settings.research.dirichlet_block_count),
        confidence_level=float(settings.research.confidence_level),
        seed=settings.hmm_regime.random_seed,
    )
    for index, profile in enumerate(HMM_POLICY_MATRIX):
        score = (
            profile["floor"] + profile["slope"] * base_multiplier
        ).clip(0.0, 1.0).round(6)
        if profile["gate"] is not None:
            score = score.where(base_multiplier >= profile["gate"], 0.0)
        normal = _weighted_path(
            daily_frames,
            normal_control.executed_weights,
            score,
            fee_rate=settings.costs.taker_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            policy=policy,
        )
        stressed = _weighted_path(
            daily_frames,
            stressed_control.executed_weights,
            score,
            fee_rate=settings.costs.taker_fee * 2.0,
            slippage_bps=settings.costs.slippage_bps * 2.0,
            spread_bps=settings.costs.spread_bps * 2.0,
            policy=policy,
        )
        dna = stable_hash(
            {
                "family": "HMM_CONDITIONED_RESIDUAL_REVERSAL",
                "engine_version": HMM_ENGINE_VERSION,
                "base_strategy_dna": RR_STRATEGY_DNA,
                "policy": profile,
                "timeframes": ["1W", "1d"],
                "feature_columns": list(inferences["1d"].probabilities.columns),
                "observer_only": True,
            },
            length=64,
        )
        stochastic = validate_strategy_return_paths(
            normal["returns"].to_numpy(),
            stressed["returns"].to_numpy(),
            policy=stochastic_policy,
            seed_offset=index * 100,
        )
        fold_consistency = _fold_consistency(normal["returns"])
        attribution = _regime_attribution(normal["returns"], inferences["1d"])
        registration = registry.register(
            data_fingerprint=data_fingerprint,
            strategy_family="HMM_CONDITIONED_RESIDUAL_REVERSAL",
            strategy_dna_hash=dna,
            parameters={
                "base_strategy_id": RR_STRATEGY_ID,
                "base_strategy_dna": RR_STRATEGY_DNA,
                "hmm_policy": profile,
                "timeframes": ["1W", "1d"],
                "observer_only": True,
            },
            metrics_at_birth=normal["metrics"],
            return_path_hash=stable_hash(
                normal["returns"].round(15).tolist(),
                length=64,
            ),
            selection_metadata={
                "preregistered_policy_index": index,
                "matrix_size": len(HMM_POLICY_MATRIX),
                "selected_after_results": False,
                "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
            },
        )
        row = {
            "policy_id": profile["policy_id"],
            "strategy_dna_hash": dna,
            "hypothesis": profile["hypothesis"],
            "parameters": profile,
            "normal": normal["metrics"],
            "stressed": stressed["metrics"],
            "integrity": normal["integrity"],
            "regime_attribution": attribution,
            "fold_consistency": fold_consistency,
            "stochastic_validation": stochastic,
            "registration": registration,
            "paper_candidate_permitted": False,
            "live_ready": False,
            "orders_generated": 0,
        }
        results.append(row)
        equity_paths[str(profile["policy_id"])] = normal["equity"]

    registry_audit = registry.audit()
    global_accounting = audit_global_trial_accounting(settings.paths.lab_dir)
    control_returns = normal_control.equity_curve.pct_change(fill_method=None).fillna(0.0)
    control = {
        "strategy_id": RR_STRATEGY_ID,
        "strategy_dna_hash": RR_STRATEGY_DNA,
        "normal": normal_control.summary(),
        "stressed": stressed_control.summary(),
        "fold_consistency": _fold_consistency(control_returns),
        "regime_attribution": _regime_attribution(control_returns, inferences["1d"]),
    }
    chart_path = reports / "hmm_rr_comparison_v1.png"
    _chart(
        chart_path,
        normal_control.equity_curve,
        equity_paths,
        inferences["1d"],
    )
    payload = {
        "schema_version": "hmm_regime_campaign_v1",
        "campaign_id": HMM_CAMPAIGN_ID,
        "generated_at": utc_iso(),
        "status": "OBSERVER_COMPLETE",
        "ai_development_status": "AI_DEVELOPMENT_EMBARGOED",
        "causality_contract": {
            "posterior": "P(S_t|x_1:t)",
            "smoothed_posterior_forbidden": "P(S_t|x_1:T)",
            "higher_timeframe_alignment": "BACKWARD_ASOF_LAST_CLOSED_CANDLE",
            "fit_window": "STRICTLY_PRIOR_TO_CLASSIFIED_OBSERVATION",
        },
        "observer": {
            timeframe: _inference_summary(inference)
            for timeframe, inference in inferences.items()
        },
        "multi_timeframe_roles": HMM_LAYER_ROLES,
        "control": control,
        "hmm_policy_matrix": list(HMM_POLICY_MATRIX),
        "candidate_results": results,
        "trial_registry": registry_audit,
        "global_trial_accounting": {
            "evaluation_trial_count": global_accounting["evaluation_trial_count"],
            "global_multiple_testing_denominator": global_accounting[
                "global_multiple_testing_denominator"
            ],
            "status": global_accounting["status"],
        },
        "artifacts": {
            "chart": str(chart_path),
            "posterior_directory": str(posteriors),
        },
        "exposure_policy": asdict(policy),
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    json_path = reports / "hmm_regime_campaign_v1.json"
    atomic_write_json(json_path, payload)
    csv_rows = [
        {
            "policy_id": row["policy_id"],
            "strategy_dna_hash": row["strategy_dna_hash"],
            "net_return": row["normal"]["net_return"],
            "annualized_return": row["normal"]["annualized_return"],
            "profit_factor": row["normal"]["portfolio_period_profit_factor"],
            "sharpe": row["normal"]["sharpe"],
            "maximum_drawdown": row["normal"]["maximum_drawdown"],
            "stressed_profit_factor": row["stressed"][
                "portfolio_period_profit_factor"
            ],
            "positive_folds": row["fold_consistency"]["positive_folds"],
            "monte_carlo_passed": row["stochastic_validation"]["normal"][
                "monte_carlo"
            ]["passed"],
            "dirichlet_passed": row["stochastic_validation"]["normal"][
                "dirichlet"
            ]["passed"],
            "orders_generated": 0,
        }
        for row in results
    ]
    csv_path = reports / "hmm_regime_campaign_v1.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    markdown_path = reports / "hmm_regime_campaign_v1.md"
    table = _markdown_table(csv_rows)
    atomic_write_text(
        markdown_path,
        "\n".join(
            [
                "# Causal HMM regime campaign v1",
                "",
                "Observer-only. Filtered probabilities only; zero order authority.",
                "",
                table,
                "",
                f"Global multiple-testing denominator: "
                f"{payload['global_trial_accounting']['global_multiple_testing_denominator']}",
                "",
                f"Chart: `{chart_path}`",
            ]
        ),
    )
    html_path = reports / "hmm_regime_campaign_v1.html"
    atomic_write_text(
        html_path,
        (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Causal HMM regime campaign v1</title>"
            "<style>body{font-family:system-ui;margin:2rem;max-width:1400px}"
            "table{border-collapse:collapse}th,td{padding:.45rem;border:1px solid #bbb}"
            "img{max-width:100%}</style></head><body>"
            "<h1>Causal HMM regime campaign v1</h1>"
            "<p>Observer-only; filtered probabilities; zero order authority.</p>"
            f"{pd.DataFrame(csv_rows).to_html(index=False, escape=True)}"
            f"<p>Global denominator: {html.escape(str(payload['global_trial_accounting']['global_multiple_testing_denominator']))}</p>"
            f"<img src='{html.escape(chart_path.name)}' alt='HMM strategy comparison'>"
            "</body></html>"
        ),
    )
    status_path = output / "status.json"
    atomic_write_json(
        status_path,
        {
            "status": "OBSERVER_COMPLETE",
            "campaign_id": HMM_CAMPAIGN_ID,
            "updated_at": payload["generated_at"],
            "current_states": {
                timeframe: payload["observer"][timeframe]["current_state"]
                for timeframe in HMM_TIMEFRAMES
            },
            "report": str(json_path),
            "orders_generated": 0,
            "orders_submitted": 0,
            "observer_only": True,
            "live_ready": False,
        },
    )
    return payload


def hmm_regime_status(settings: Settings) -> dict[str, Any]:
    path = settings.paths.output_dir / "hmm" / "status.json"
    if not path.is_file():
        return {
            "status": "NOT_RUN",
            "enabled": settings.hmm_regime.enabled,
            "observer_only": settings.hmm_regime.observer_only,
            "orders_generated": 0,
        }
    from utils.common import read_json

    return dict(read_json(path))


__all__ = [
    "HMM_CAMPAIGN_ID",
    "HMM_POLICY_MATRIX",
    "hmm_regime_status",
    "run_hmm_regime_campaign",
]
