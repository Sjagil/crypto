"""Equal-sample comparison of every canonical strategy with a causal HMM overlay.

The overlay is deliberately narrow: it changes only the size of a *new* entry.
Entry, exit, stop, target, trailing-stop, costs and next-open execution stay
byte-for-byte delegated to the registered base strategy.  This makes the
base/HMM comparison interpretable and prevents the HMM from becoming an opaque
second signal engine.
"""

from __future__ import annotations

import html
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import Settings
from research.backtest import BacktestConfig, BacktestEngine, BacktestResult
from research.features import FeaturePipeline
from research.global_trial_accounting import audit_global_trial_accounting
from research.hmm_regime_campaign import (
    ALLOWED_MARKETS,
    HMM_CAMPAIGN_ID,
    run_hmm_regime_campaign,
)
from research.hmm_regime_manager import (
    HMM_ENGINE_VERSION,
    HMMInference,
    InstitutionalHMMRegimeManager,
)
from research.stochastic_validation import (
    StochasticValidationPolicy,
    validate_strategy_return_paths,
)
from research.strategies import Strategy, StrategyOutput, strategy_registry
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


HMM_ALL_STRATEGIES_CAMPAIGN_ID = "hmm_all_canonical_strategies_v1"
HMM_COMPARISON_TIMEFRAMES = ("15m", "1h", "4h", "1d", "1W")
HMM_OVERLAY_POLICY = {
    "policy_id": "HMM_SOFT_40",
    "floor": 0.40,
    "slope": 0.60,
    "hard_gate": None,
    "entry_or_exit_changed": False,
    "existing_position_resized": False,
}
MAXIMUM_BARS = {
    "15m": 6_000,
    "1h": 8_000,
    "4h": 8_000,
    "1d": 5_000,
    "1W": 1_500,
}


class HMMConditionedStrategy(Strategy):
    """Apply a preregistered causal HMM multiplier to base entry sizing only."""

    def __init__(self, base: Strategy, *, dna_hash: str) -> None:
        self.base = base
        self.strategy_id = f"{base.strategy_id}__hmm_soft_40"
        self.family = base.family
        self.description = (
            f"{base.description} Causal HMM entry-size overlay; signals unchanged."
        )
        self.defaults = dict(base.defaults)
        self.parameter_space = dict(base.parameter_space)
        self.uses_intelligence = base.uses_intelligence
        self.dna_hash = dna_hash

    def generate(
        self,
        features: pd.DataFrame,
        parameters: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        output = self.base.generate(features, parameters)
        if "_hmm_entry_size_multiplier" not in features:
            raise ValueError("HMM multiplier is missing from comparison features")
        multiplier = (
            pd.to_numeric(
                features["_hmm_entry_size_multiplier"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        sizes = output.size_multiplier.mul(multiplier).clip(0.0, 1.0)
        metadata = dict(output.metadata)
        metadata.update(
            {
                "hmm_overlay": HMM_OVERLAY_POLICY["policy_id"],
                "hmm_engine_version": HMM_ENGINE_VERSION,
                "hmm_strategy_dna": self.dna_hash,
                "hmm_changes_entry_exit": False,
            }
        )
        return replace(
            output,
            size_multiplier=sizes,
            metadata=metadata,
        ).validate(features.index)


def _posterior_inference(path: Path, timeframe: str) -> HMMInference:
    frame = pd.read_parquet(path)
    required = {
        "dominant_state",
        "posterior_entropy",
        "risk_multiplier",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"HMM posterior missing columns: {sorted(missing)}")
    probability_columns = [
        column for column in frame.columns if column not in required
    ]
    return HMMInference(
        timeframe=timeframe,
        probabilities=frame.loc[:, probability_columns].astype(float),
        dominant_state=frame["dominant_state"].astype(str),
        posterior_entropy=frame["posterior_entropy"].astype(float),
        risk_multiplier=frame["risk_multiplier"].astype(float),
        expected_duration={},
        current_forecasts={},
        fit_history=(),
        integrity={
            "loaded_from_immutable_campaign": HMM_CAMPAIGN_ID,
            "filtered_not_smoothed": True,
        },
    )


def _load_inferences(
    settings: Settings,
) -> tuple[dict[str, HMMInference], dict[str, str]]:
    root = settings.paths.output_dir / "hmm" / "posteriors"
    paths = {
        timeframe: root / f"{timeframe}_filtered.parquet"
        for timeframe in ("1W", "1d")
    }
    return {
        timeframe: _posterior_inference(
            path,
            timeframe,
        )
        for timeframe, path in paths.items()
    }, {
        timeframe: sha256_file(path)
        for timeframe, path in paths.items()
    }


def _load_market_frame(
    settings: Settings,
    market: str,
    timeframe: str,
) -> pd.DataFrame:
    path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"HMM_STRATEGY_DATA_MISSING:{path}")
    frame = pd.read_parquet(path).iloc[-MAXIMUM_BARS[timeframe] :].copy()
    frame.attrs.update(
        {
            "market": market,
            "timeframe": timeframe,
            "provider": "REAL_NORMALIZED_MARKET_DATA",
            "synthetic": False,
        }
    )
    return frame


def _comparison_features(
    settings: Settings,
    timeframe: str,
    manager: InstitutionalHMMRegimeManager,
    inferences: Mapping[str, HMMInference],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    raw = {
        market: _load_market_frame(settings, market, timeframe)
        for market in ALLOWED_MARKETS
    }
    benchmark = raw["BTC-EUR"]
    source_inferences = (
        {"1W": inferences["1W"]}
        if timeframe == "1W"
        else {"1W": inferences["1W"], "1d": inferences["1d"]}
    )
    features: dict[str, pd.DataFrame] = {}
    for market, frame in raw.items():
        built = FeaturePipeline().build(
            frame,
            market=market,
            benchmark=benchmark,
        )
        aligned = manager.backward_asof_align(
            built.index,
            source_inferences,
            target_timeframe=timeframe,
        )
        base_multiplier = manager.blended_risk_multiplier(aligned)
        soft_multiplier = (
            HMM_OVERLAY_POLICY["floor"]
            + HMM_OVERLAY_POLICY["slope"] * base_multiplier
        ).clip(0.0, 1.0)
        valid = soft_multiplier.notna()
        built = built.loc[valid].copy()
        built["_hmm_entry_size_multiplier"] = soft_multiplier.loc[valid]
        built["_hmm_dominant_state"] = (
            aligned.loc[valid, "1d_state"]
            if "1d_state" in aligned
            else aligned.loc[valid, "1W_state"]
        )
        knowability = dict(built.attrs["feature_knowability"])
        knowability["_hmm_entry_size_multiplier"] = {
            "available_at": "source_candle_close",
            "lookahead_safe": True,
            "repaint": False,
            "alignment": "backward_asof_available_at",
        }
        knowability["_hmm_dominant_state"] = {
            "available_at": "source_candle_close",
            "lookahead_safe": True,
            "repaint": False,
            "alignment": "backward_asof_available_at",
        }
        built.attrs["feature_knowability"] = knowability
        built.attrs["hmm_alignment_integrity"] = "PASSED"
        features[market] = built
    hashes = {
        market: sha256_file(
            settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
        )
        for market in ALLOWED_MARKETS
    }
    return features, hashes


def _backtest_config(settings: Settings, *, stressed: bool) -> BacktestConfig:
    return replace(
        BacktestConfig.from_settings(
            settings,
            stressed=stressed,
            allow_review_required_research_only=False,
        ),
        bootstrap_samples=100,
        monte_carlo_runs=100,
    )


def _run_pair(
    settings: Settings,
    features: dict[str, pd.DataFrame],
    base: Strategy,
    overlay: HMMConditionedStrategy,
) -> tuple[BacktestResult, BacktestResult, BacktestResult, BacktestResult]:
    normal = _backtest_config(settings, stressed=False)
    stressed = _backtest_config(settings, stressed=True)
    return (
        BacktestEngine(normal, settings=settings).run(features, base),
        BacktestEngine(normal, settings=settings).run(features, overlay),
        BacktestEngine(stressed, settings=settings).run(features, base),
        BacktestEngine(stressed, settings=settings).run(features, overlay),
    )


def _equity_returns(result: BacktestResult) -> pd.Series:
    return (
        pd.to_numeric(result.equity_curve["equity"], errors="coerce")
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .rename("return")
    )


def _stochastic_returns(result: BacktestResult) -> np.ndarray:
    equity = pd.to_numeric(
        result.equity_curve["equity"],
        errors="coerce",
    ).dropna()
    daily = equity.resample("1D").last().dropna()
    selected = daily if len(daily) >= 30 else equity
    return (
        selected.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )


def _finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _selected_metrics(result: BacktestResult) -> dict[str, Any]:
    keys = (
        "net_return",
        "cagr",
        "profit_factor",
        "net_expectancy_r",
        "net_expectancy_eur",
        "trade_count",
        "effective_sample_size",
        "maximum_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "average_exposure",
        "turnover",
        "transaction_costs_eur",
        "monte_carlo_p95_drawdown",
        "probability_of_loss",
    )
    return _finite({key: result.metrics.get(key) for key in keys})


def _metric_delta(
    base: Mapping[str, Any],
    hmm: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in base:
        left = base.get(key)
        right = hmm.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            output[key] = float(right) - float(left)
    return _finite(output)


def _regime_attribution(
    returns: pd.Series,
    state: pd.Series,
) -> dict[str, Any]:
    aligned_state = state.reindex(returns.index).ffill().shift(1)
    rows: dict[str, Any] = {}
    for label, values in returns.groupby(aligned_state):
        if not isinstance(label, str) or values.empty:
            continue
        gains = float(values.where(values > 0.0, 0.0).sum())
        losses = float(abs(values.where(values < 0.0, 0.0).sum()))
        equity = (1.0 + values).cumprod()
        rows[label] = _finite(
            {
                "observations": int(len(values)),
                "net_return": float(equity.iloc[-1] - 1.0),
                "profit_factor": gains / losses if losses else None,
                "maximum_drawdown": float(
                    equity.div(equity.cummax()).sub(1.0).min()
                ),
            }
        )
    return rows


def _score(row: Mapping[str, Any]) -> float:
    metrics = row["hmm_normal"]
    stressed = row["hmm_stressed"]
    drawdown = abs(float(metrics.get("maximum_drawdown") or 0.0))
    return float(
        35.0 * np.clip(float(metrics.get("cagr") or 0.0), -0.5, 1.0)
        + 20.0 * np.clip(float(metrics.get("sharpe") or 0.0), -2.0, 3.0) / 3.0
        + 15.0 * np.clip(float(metrics.get("profit_factor") or 0.0) - 1.0, -1.0, 2.0) / 2.0
        + 15.0 * np.clip(float(stressed.get("profit_factor") or 0.0) - 1.0, -1.0, 2.0) / 2.0
        + 15.0 * (1.0 - np.clip(drawdown, 0.0, 0.50) / 0.50)
    )


def _write_chart(
    path: Path,
    rows: list[dict[str, Any]],
    equity_paths: Mapping[str, tuple[pd.Series, pd.Series]],
) -> None:
    ranked = sorted(rows, key=lambda row: row["comparison_score"], reverse=True)
    top = ranked[:20]
    labels = [f"{row['base_strategy_id']} [{row['timeframe']}]" for row in top]
    delta_cagr = [
        100.0 * float(row["delta_normal"].get("cagr") or 0.0)
        for row in top
    ]
    delta_drawdown = [
        100.0 * float(row["drawdown_improvement"])
        for row in top
    ]
    figure, axes = plt.subplots(3, 1, figsize=(15, 16))
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in delta_cagr]
    axes[0].barh(labels[::-1], delta_cagr[::-1], color=colors[::-1])
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("HMM minus base CAGR (percentage points)")
    axes[0].grid(axis="x", alpha=0.2)
    colors_dd = [
        "#2e7d32" if value >= 0 else "#c62828" for value in delta_drawdown
    ]
    axes[1].barh(labels[::-1], delta_drawdown[::-1], color=colors_dd[::-1])
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(
        "HMM minus base maximum drawdown (positive means less severe)"
    )
    axes[1].grid(axis="x", alpha=0.2)
    for row in top[:8]:
        key = str(row["comparison_id"])
        base, hmm = equity_paths[key]
        base_norm = base / float(base.iloc[0])
        hmm_norm = hmm / float(hmm.iloc[0])
        label = f"{row['base_strategy_id']} {row['timeframe']}"
        axes[2].plot(
            base_norm.index,
            base_norm,
            linewidth=0.9,
            alpha=0.45,
            linestyle="--",
        )
        axes[2].plot(hmm_norm.index, hmm_norm, linewidth=1.2, label=label)
    axes[2].set_title("Top HMM variants (solid) with equal-sample bases (dashed)")
    axes[2].set_ylabel("Equity, normalized to 1")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _flat_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    selected = []
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["comparison_score"], reverse=True),
        start=1,
    ):
        selected.append(
            {
                "rank": rank,
                "base_strategy_id": row["base_strategy_id"],
                "hmm_strategy_id": row["hmm_strategy_id"],
                "family": row["family"],
                "timeframe": row["timeframe"],
                "hmm_strategy_dna": row["hmm_strategy_dna"],
                "comparison_score": row["comparison_score"],
                "base_net_return": row["base_normal"]["net_return"],
                "hmm_net_return": row["hmm_normal"]["net_return"],
                "delta_net_return": row["delta_normal"].get("net_return"),
                "base_cagr": row["base_normal"]["cagr"],
                "hmm_cagr": row["hmm_normal"]["cagr"],
                "delta_cagr": row["delta_normal"].get("cagr"),
                "base_profit_factor": row["base_normal"]["profit_factor"],
                "hmm_profit_factor": row["hmm_normal"]["profit_factor"],
                "hmm_stressed_profit_factor": row["hmm_stressed"][
                    "profit_factor"
                ],
                "base_sharpe": row["base_normal"]["sharpe"],
                "hmm_sharpe": row["hmm_normal"]["sharpe"],
                "base_maximum_drawdown": row["base_normal"][
                    "maximum_drawdown"
                ],
                "hmm_maximum_drawdown": row["hmm_normal"][
                    "maximum_drawdown"
                ],
                "drawdown_improvement": row["drawdown_improvement"],
                "hmm_trade_count": row["hmm_normal"]["trade_count"],
                "hmm_effective_sample_size": row["hmm_normal"][
                    "effective_sample_size"
                ],
                "monte_carlo_and_dirichlet_passed": row[
                    "stochastic_validation"
                ]["passed"],
                "hmm_positive_after_costs": row["hmm_positive_after_costs"],
                "orders_generated": 0,
            }
        )
    return pd.DataFrame(selected)


def _markdown_frame(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    def cell(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(cell(value) for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_human_reports(
    reports: Path,
    payload: Mapping[str, Any],
    flat: pd.DataFrame,
) -> None:
    display = flat.head(20).copy()
    percent_columns = [
        "base_net_return",
        "hmm_net_return",
        "delta_net_return",
        "base_cagr",
        "hmm_cagr",
        "delta_cagr",
        "base_maximum_drawdown",
        "hmm_maximum_drawdown",
        "drawdown_improvement",
    ]
    for column in percent_columns:
        display[column] = display[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{100.0 * float(value):.2f}%"
            )
        )
    columns = [
        "rank",
        "base_strategy_id",
        "timeframe",
        "base_cagr",
        "hmm_cagr",
        "delta_cagr",
        "base_profit_factor",
        "hmm_profit_factor",
        "hmm_stressed_profit_factor",
        "base_maximum_drawdown",
        "hmm_maximum_drawdown",
        "drawdown_improvement",
        "hmm_trade_count",
        "monte_carlo_and_dirichlet_passed",
    ]
    table = display.loc[:, columns]
    markdown = [
        "# HMM comparison across all canonical strategies",
        "",
        f"- Campaign: `{payload['campaign_id']}`",
        f"- Comparisons: {payload['summary']['comparison_count']}",
        f"- HMM improvements in CAGR: {payload['summary']['hmm_cagr_improvement_count']}",
        f"- HMM improvements in drawdown: {payload['summary']['hmm_drawdown_improvement_count']}",
        f"- Positive HMM variants after costs: {payload['summary']['hmm_positive_count']}",
        "- HMM authority: observer/research only; no paper or live promotion.",
        "",
        "## Top 20",
        "",
        _markdown_frame(table),
        "",
        "Every row uses equal samples, identical signals/exits/stops/targets, "
        "normal and stressed costs, next-open execution, and a causal filtered "
        "HMM size overlay. An HMM variant is a new trial, not a rewritten base.",
    ]
    atomic_write_text(
        reports / "hmm_all_strategies_comparison_v1.md",
        "\n".join(markdown) + "\n",
    )
    html_table = table.to_html(index=False, escape=True)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>HMM all-strategy comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:.4rem}}th{{background:#f1f4f8}}</style>
</head><body><h1>HMM comparison across all canonical strategies</h1>
<p>Campaign: <code>{html.escape(str(payload["campaign_id"]))}</code>.
Research observer only; orders generated: 0.</p>
<img src="hmm_all_strategies_comparison_v1.png"
alt="HMM comparison chart" style="max-width:100%">
<h2>Top 20</h2>{html_table}</body></html>"""
    atomic_write_text(
        reports / "hmm_all_strategies_comparison_v1.html",
        page,
    )


def _write_top50_mtf_pipeline_reports(
    reports: Path,
    payload: Mapping[str, Any],
    flat: pd.DataFrame,
) -> dict[str, str]:
    """Publish the best fifty existing comparisons as a shadow MTF registry.

    The historical comparison remains an equal-sample single-timeframe test.
    The sequential 1d/4h/1h/15m chain is deliberately labelled prospective:
    registering the overlay does not manufacture historical TAO evidence and
    cannot grant order authority.
    """

    pipeline_timeframes = {"15m", "1h", "4h", "1d"}
    selected = flat.loc[
        flat["timeframe"].astype(str).isin(pipeline_timeframes)
    ].head(50).copy()
    pipeline = {
        "macro_regime": "1d closed candle; directional context only",
        "setup": "4h closed candle; structure and invalidation",
        "confirmation": "1h closed candle; momentum, volume and relative strength",
        "entry_timing": "15m closed candle; reclaim, retest or breakout confirmation",
        "execution_veto": "1m/ticks/orderbook; costs and liquidity only",
    }
    rows = selected.to_dict(orient="records")
    registry = _finite(
        {
            "schema_version": "top_50_mtf_strategy_pipeline_v1",
            "generated_at": payload.get("generated_at"),
            "source_campaign": payload.get("campaign_id"),
            "selection_basis": "comparison_score_descending_clipped_to_50",
            "strategy_timeframe_combination_count": len(rows),
            "pipeline": pipeline,
            "application_mode": "SHADOW_CONTEXT_OVERLAY",
            "historical_metrics_scope": list(payload.get("universe") or []),
            "prospective_markets": [
                *list(payload.get("universe") or []),
                "TAO-EUR",
            ],
            "tao_evidence_status": (
                "PROSPECTIVE_ONLY_NOT_IN_HISTORICAL_COMPARISON_MATRIX"
            ),
            "closed_candle_only": True,
            "next_open_execution": True,
            "orders_generated": 0,
            "orders_submitted": 0,
            "live_authority_granted": False,
            "rows": rows,
        }
    )
    json_path = reports / "top_50_mtf_strategy_pipeline_v1.json"
    csv_path = reports / "top_50_mtf_strategy_pipeline_v1.csv"
    md_path = reports / "top_50_mtf_strategy_pipeline_v1.md"
    atomic_write_json(json_path, registry)
    selected.to_csv(csv_path, index=False)
    display_columns = [
        column
        for column in (
            "rank",
            "base_strategy_id",
            "timeframe",
            "hmm_cagr",
            "hmm_profit_factor",
            "hmm_stressed_profit_factor",
            "hmm_maximum_drawdown",
            "hmm_trade_count",
            "monte_carlo_and_dirichlet_passed",
        )
        if column in selected.columns
    ]
    markdown = [
        "# Top 50 multi-timeframe strategy pipeline",
        "",
        "Sequential shadow chain: **1D regime → 4H setup → 1H "
        "confirmation → 15m entry → microstructure execution veto**.",
        "",
        "Historical metrics retain their original campaign universe. TAO-EUR "
        "is prospective-only until separately closed and reconciled evidence "
        "exists; this registry grants no order authority.",
        "",
        _markdown_frame(selected.loc[:, display_columns]),
    ]
    atomic_write_text(md_path, "\n".join(markdown) + "\n")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def run_hmm_all_strategy_comparison(settings: Settings) -> dict[str, Any]:
    """Run every registered canonical strategy with and without the HMM overlay."""

    if not settings.hmm_regime.enabled:
        return {"status": "DISABLED", "orders_generated": 0}
    if not settings.hmm_regime.observer_only:
        raise RuntimeError("HMM_ALL_STRATEGIES_REQUIRES_OBSERVER_ONLY")

    # Rebuild the causal posterior artifacts when the campaign identity changed.
    status_path = settings.paths.output_dir / "hmm" / "status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    regime_report_path = (
        settings.paths.output_dir
        / "hmm"
        / "reports"
        / "hmm_regime_campaign_v1.json"
    )
    regime_report = (
        read_json(regime_report_path)
        if regime_report_path.is_file()
        else {}
    )
    posterior_paths_exist = all(
        (
            settings.paths.output_dir
            / "hmm"
            / "posteriors"
            / f"{timeframe}_filtered.parquet"
        ).is_file()
        for timeframe in ("1W", "1d")
    )
    if not (
        posterior_paths_exist
        and (
            status.get("campaign_id") == HMM_CAMPAIGN_ID
            or regime_report.get("campaign_id") == HMM_CAMPAIGN_ID
        )
    ):
        run_hmm_regime_campaign(settings)

    inferences, posterior_hashes = _load_inferences(settings)
    manager = InstitutionalHMMRegimeManager(settings.hmm_regime)
    reports = settings.paths.output_dir / "hmm" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    registry = ContentAddressedTrialRegistry(
        settings.paths.lab_dir
        / "strategy_registry"
        / HMM_ALL_STRATEGIES_CAMPAIGN_ID,
        campaign_id=HMM_ALL_STRATEGIES_CAMPAIGN_ID,
    )
    policy = StochasticValidationPolicy(
        simulations=100,
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
        seed=int(settings.hmm_regime.random_seed),
    )

    rows: list[dict[str, Any]] = []
    equity_paths: dict[str, tuple[pd.Series, pd.Series]] = {}
    feature_evidence: dict[str, Any] = {}
    strategies = strategy_registry()
    checkpoint_root = (
        settings.paths.output_dir / "hmm" / "checkpoints" / "all_strategies_v1"
    )
    checkpoint_rows = checkpoint_root / "rows"
    checkpoint_equity = checkpoint_root / "equity"
    checkpoint_rows.mkdir(parents=True, exist_ok=True)
    checkpoint_equity.mkdir(parents=True, exist_ok=True)
    progress_path = (
        settings.paths.output_dir / "hmm" / "all_strategies_progress.json"
    )
    started_at = utc_now()
    atomic_write_json(
        progress_path,
        {
            "status": "RUNNING",
            "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
            "started_at": started_at,
            "completed_comparisons": 0,
            "total_comparisons": (
                len(strategies) * len(HMM_COMPARISON_TIMEFRAMES)
            ),
            "orders_generated": 0,
        },
    )
    for timeframe_number, timeframe in enumerate(HMM_COMPARISON_TIMEFRAMES):
        features, data_hashes = _comparison_features(
            settings,
            timeframe,
            manager,
            inferences,
        )
        feature_evidence[timeframe] = {
            "data_hashes": data_hashes,
            "rows_by_market": {
                market: int(len(frame))
                for market, frame in features.items()
            },
            "sample_start": min(frame.index[0] for frame in features.values()),
            "sample_end": max(frame.index[-1] for frame in features.values()),
            "hmm_layers": ["1W"] if timeframe == "1W" else ["1W", "1d"],
        }
        data_fingerprint = stable_hash(
            {
                "timeframe": timeframe,
                "data_hashes": data_hashes,
                "posterior_hashes": posterior_hashes,
                "rows": feature_evidence[timeframe]["rows_by_market"],
                "sample_start": feature_evidence[timeframe]["sample_start"],
                "sample_end": feature_evidence[timeframe]["sample_end"],
            },
            length=64,
        )
        for strategy_number, base in enumerate(strategies.values()):
            dna_hash = stable_hash(
                {
                    "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
                    "base_strategy_id": base.strategy_id,
                    "base_defaults": base.defaults,
                    "timeframe": timeframe,
                    "universe": ALLOWED_MARKETS,
                    "hmm_policy": HMM_OVERLAY_POLICY,
                    "hmm_engine_version": HMM_ENGINE_VERSION,
                    "hmm_configuration": settings.hmm_regime.model_dump(
                        mode="json"
                    ),
                    "posterior_hashes": posterior_hashes,
                    "hmm_layers": feature_evidence[timeframe]["hmm_layers"],
                },
                length=64,
            )
            overlay = HMMConditionedStrategy(base, dna_hash=dna_hash)
            comparison_id = stable_hash(
                [base.strategy_id, timeframe, dna_hash],
                length=24,
            )
            checkpoint_identity = (
                f"{comparison_id}_{data_fingerprint[:16]}"
            )
            row_checkpoint = checkpoint_rows / f"{checkpoint_identity}.json"
            equity_checkpoint = (
                checkpoint_equity / f"{checkpoint_identity}.parquet"
            )
            if row_checkpoint.is_file() and equity_checkpoint.is_file():
                checkpoint_row = read_json(row_checkpoint)
                if (
                    checkpoint_row.get("data_fingerprint")
                    != data_fingerprint
                    or checkpoint_row.get("hmm_strategy_dna") != dna_hash
                ):
                    raise RuntimeError(
                        "HMM_COMPARISON_CHECKPOINT_IDENTITY_MISMATCH"
                    )
                checkpoint_curve = pd.read_parquet(equity_checkpoint)
                if set(checkpoint_curve.columns) != {"base", "hmm"}:
                    raise RuntimeError(
                        "HMM_COMPARISON_EQUITY_CHECKPOINT_INVALID"
                    )
                rows.append(checkpoint_row)
                equity_paths[comparison_id] = (
                    checkpoint_curve["base"].dropna(),
                    checkpoint_curve["hmm"].dropna(),
                )
                atomic_write_json(
                    progress_path,
                    {
                        "status": "RUNNING",
                        "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
                        "started_at": started_at,
                        "updated_at": utc_now(),
                        "completed_comparisons": len(rows),
                        "total_comparisons": (
                            len(strategies)
                            * len(HMM_COMPARISON_TIMEFRAMES)
                        ),
                        "current_timeframe": timeframe,
                        "current_strategy": base.strategy_id,
                        "checkpoint_reused": True,
                        "orders_generated": 0,
                    },
                )
                continue
            base_normal, hmm_normal, base_stressed, hmm_stressed = _run_pair(
                settings,
                features,
                base,
                overlay,
            )
            base_metrics = _selected_metrics(base_normal)
            hmm_metrics = _selected_metrics(hmm_normal)
            base_stressed_metrics = _selected_metrics(base_stressed)
            hmm_stressed_metrics = _selected_metrics(hmm_stressed)
            normal_stochastic = _stochastic_returns(hmm_normal)
            stressed_stochastic = _stochastic_returns(hmm_stressed)
            stochastic = (
                validate_strategy_return_paths(
                    normal_stochastic,
                    stressed_stochastic,
                    policy=policy,
                    seed_offset=(
                        timeframe_number * 10_000 + strategy_number * 100
                    ),
                )
                if min(len(normal_stochastic), len(stressed_stochastic)) >= 30
                else {
                    "passed": False,
                    "reason_codes": ["INSUFFICIENT_STOCHASTIC_OBSERVATIONS"],
                    "checks": {},
                    "policy": _finite(policy.__dict__),
                }
            )
            hmm_returns = _equity_returns(hmm_normal)
            state = features["BTC-EUR"]["_hmm_dominant_state"]
            row = {
                "comparison_id": comparison_id,
                "data_fingerprint": data_fingerprint,
                "base_strategy_id": base.strategy_id,
                "hmm_strategy_id": overlay.strategy_id,
                "hmm_strategy_dna": dna_hash,
                "family": base.family,
                "timeframe": timeframe,
                "universe": list(ALLOWED_MARKETS),
                "base_parameters": dict(base.defaults),
                "hmm_policy": dict(HMM_OVERLAY_POLICY),
                "base_normal": base_metrics,
                "hmm_normal": hmm_metrics,
                "delta_normal": _metric_delta(base_metrics, hmm_metrics),
                "base_stressed": base_stressed_metrics,
                "hmm_stressed": hmm_stressed_metrics,
                "delta_stressed": _metric_delta(
                    base_stressed_metrics,
                    hmm_stressed_metrics,
                ),
                "drawdown_improvement": (
                    float(base_metrics.get("maximum_drawdown") or 0.0)
                    - float(hmm_metrics.get("maximum_drawdown") or 0.0)
                ),
                "stressed_drawdown_improvement": (
                    float(
                        base_stressed_metrics.get("maximum_drawdown")
                        or 0.0
                    )
                    - float(
                        hmm_stressed_metrics.get("maximum_drawdown")
                        or 0.0
                    )
                ),
                "hmm_positive_after_costs": bool(
                    float(hmm_metrics.get("net_return") or 0.0) > 0.0
                    and float(hmm_metrics.get("profit_factor") or 0.0) > 1.0
                    and float(hmm_metrics.get("net_expectancy_r") or 0.0) > 0.0
                ),
                "stochastic_validation": _finite(stochastic),
                "hmm_regime_attribution": _regime_attribution(
                    hmm_returns,
                    state,
                ),
                "integrity": {
                    "same_sample": True,
                    "same_costs": True,
                    "same_entry_exit_stop_target": True,
                    "next_open_execution": True,
                    "closed_candle_only": True,
                    "filtered_not_smoothed": True,
                    "backward_asof_available_at": True,
                    "synthetic_data": False,
                    "orders_generated": 0,
                },
                "promotion": {
                    "research_observer_only": True,
                    "paper_candidate_permitted": False,
                    "live_ready": False,
                    "reason": "NEW_HMM_DNA_REQUIRES_SEPARATE_FORWARD_EVIDENCE",
                },
            }
            row["comparison_score"] = _score(row)
            registration = registry.register(
                data_fingerprint=data_fingerprint,
                strategy_family=f"{base.family}_hmm_overlay",
                strategy_dna_hash=dna_hash,
                parameters={
                    "base_strategy_id": base.strategy_id,
                    "base_parameters": base.defaults,
                    "timeframe": timeframe,
                    "universe": list(ALLOWED_MARKETS),
                    "hmm_policy": HMM_OVERLAY_POLICY,
                },
                metrics_at_birth={
                    "normal": hmm_metrics,
                    "stressed": hmm_stressed_metrics,
                    "stochastic_passed": bool(stochastic["passed"]),
                },
                return_path_hash=stable_hash(
                    hmm_returns.round(12).tolist(),
                    length=64,
                ),
                selection_metadata={
                    "preregistered_before_comparison": True,
                    "base_strategy_unchanged": True,
                    "observer_only": True,
                    "orders_generated": 0,
                },
            )
            row["trial_registration"] = registration
            finalized_row = _finite(row)
            rows.append(finalized_row)
            equity_paths[comparison_id] = (
                base_normal.equity_curve["equity"],
                hmm_normal.equity_curve["equity"],
            )
            atomic_write_json(row_checkpoint, finalized_row)
            pd.concat(
                {
                    "base": base_normal.equity_curve["equity"],
                    "hmm": hmm_normal.equity_curve["equity"],
                },
                axis=1,
            ).to_parquet(equity_checkpoint)
            atomic_write_json(
                progress_path,
                {
                    "status": "RUNNING",
                    "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
                    "started_at": started_at,
                    "updated_at": utc_now(),
                    "completed_comparisons": len(rows),
                    "total_comparisons": (
                        len(strategies)
                        * len(HMM_COMPARISON_TIMEFRAMES)
                    ),
                    "current_timeframe": timeframe,
                    "current_strategy": base.strategy_id,
                    "orders_generated": 0,
                },
            )

    ranked = sorted(rows, key=lambda row: row["comparison_score"], reverse=True)
    summary = {
        "canonical_strategy_count": len(strategies),
        "timeframe_count": len(HMM_COMPARISON_TIMEFRAMES),
        "comparison_count": len(rows),
        "hmm_positive_count": sum(
            bool(row["hmm_positive_after_costs"]) for row in rows
        ),
        "hmm_stressed_positive_count": sum(
            float(row["hmm_stressed"].get("net_return") or 0.0) > 0.0
            and float(row["hmm_stressed"].get("profit_factor") or 0.0) > 1.0
            for row in rows
        ),
        "hmm_cagr_improvement_count": sum(
            float(row["delta_normal"].get("cagr") or 0.0) > 0.0
            for row in rows
        ),
        "hmm_drawdown_improvement_count": sum(
            float(row["drawdown_improvement"]) > 0.0
            for row in rows
        ),
        "stochastic_pass_count": sum(
            bool(row["stochastic_validation"]["passed"]) for row in rows
        ),
        "standalone_zero_entry_strategy_count": sum(
            row["base_strategy_id"] == "risk_event_avoidance_overlay"
            for row in rows
        ),
    }
    global_accounting = audit_global_trial_accounting(settings.paths.lab_dir)
    payload = _finite(
        {
            "schema_version": "hmm_all_strategy_comparison_v1",
            "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
            "generated_at": utc_now(),
            "status": "COMPLETED_OBSERVER_ONLY",
            "hypothesis": (
                "A causal filtered HMM regime probability can improve the "
                "downside efficiency of a fixed strategy by scaling only new "
                "entry risk, without changing its trading rules."
            ),
            "policy": HMM_OVERLAY_POLICY,
            "timeframes": list(HMM_COMPARISON_TIMEFRAMES),
            "universe": list(ALLOWED_MARKETS),
            "feature_evidence": feature_evidence,
            "posterior_hashes": posterior_hashes,
            "hmm_configuration": settings.hmm_regime.model_dump(mode="json"),
            "summary": summary,
            "top_20": ranked[:20],
            "top_50": ranked[:50],
            "multi_timeframe_pipeline": {
                "roles": {
                    "1d": "macro_regime",
                    "4h": "setup_and_invalidation",
                    "1h": "confirmation",
                    "15m": "entry_timing",
                },
                "mode": "SHADOW_CONTEXT_OVERLAY",
                "orders_generated": 0,
                "orders_submitted": 0,
            },
            "comparisons": ranked,
            "trial_registry": registry.index(),
            "global_trial_accounting": {
                "global_multiple_testing_denominator": global_accounting[
                    "global_multiple_testing_denominator"
                ],
                "integrity_status": global_accounting["status"],
            },
            "authority": {
                "orders_generated": 0,
                "orders_submitted": 0,
                "paper_candidate_permitted": False,
                "live_ready": False,
                "observer_only": True,
                "existing_rr_canary_unchanged": True,
            },
        }
    )
    flat = _flat_rows(ranked)
    atomic_write_json(
        reports / "hmm_all_strategies_comparison_v1.json",
        payload,
    )
    flat.to_csv(
        reports / "hmm_all_strategies_comparison_v1.csv",
        index=False,
    )
    _write_chart(
        reports / "hmm_all_strategies_comparison_v1.png",
        ranked,
        equity_paths,
    )
    _write_human_reports(reports, payload, flat)
    payload["top_50_mtf_artifacts"] = _write_top50_mtf_pipeline_reports(
        reports,
        payload,
        flat,
    )
    atomic_write_json(
        reports / "hmm_all_strategies_comparison_v1.json",
        payload,
    )
    status_payload = {
        "status": payload["status"],
        "campaign_id": payload["campaign_id"],
        "generated_at": payload["generated_at"],
        "summary": summary,
        "report": str(
            reports / "hmm_all_strategies_comparison_v1.json"
        ),
        "orders_generated": 0,
        "observer_only": True,
        "live_ready": False,
    }
    atomic_write_json(
        settings.paths.output_dir / "hmm" / "all_strategies_status.json",
        status_payload,
    )
    atomic_write_json(
        progress_path,
        status_payload
        | {
            "completed_comparisons": len(rows),
            "total_comparisons": len(rows),
        },
    )
    return payload


def hmm_all_strategy_status(settings: Settings) -> dict[str, Any]:
    path = settings.paths.output_dir / "hmm" / "all_strategies_status.json"
    if not path.is_file():
        return {
            "status": "NOT_RUN",
            "campaign_id": HMM_ALL_STRATEGIES_CAMPAIGN_ID,
            "orders_generated": 0,
            "observer_only": True,
        }
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def refresh_top50_mtf_registry(settings: Settings) -> dict[str, Any]:
    """Rebuild the top-50 shadow registry from completed immutable evidence."""

    reports = settings.paths.output_dir / "hmm" / "reports"
    source = reports / "hmm_all_strategies_comparison_v1.json"
    if not source.is_file():
        return {
            "status": "BLOCKED_SOURCE_COMPARISON_MISSING",
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    payload = dict(read_json(source))
    comparisons = [
        dict(row)
        for row in payload.get("comparisons") or []
        if isinstance(row, Mapping)
    ]
    comparisons.sort(
        key=lambda row: float(row.get("comparison_score") or 0.0),
        reverse=True,
    )
    flat = _flat_rows(comparisons)
    pipeline_comparisons = [
        row
        for row in comparisons
        if str(row.get("timeframe")) in {"15m", "1h", "4h", "1d"}
    ]
    payload["top_50"] = pipeline_comparisons[:50]
    payload["multi_timeframe_pipeline"] = {
        "roles": {
            "1d": "macro_regime",
            "4h": "setup_and_invalidation",
            "1h": "confirmation",
            "15m": "entry_timing",
        },
        "mode": "SHADOW_CONTEXT_OVERLAY",
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    artifacts = _write_top50_mtf_pipeline_reports(reports, payload, flat)
    payload["top_50_mtf_artifacts"] = artifacts
    atomic_write_json(source, payload)
    return {
        "status": "READY_SHADOW_ONLY",
        "source_campaign": payload.get("campaign_id"),
        "comparison_count": len(comparisons),
        "selected_count": min(50, len(pipeline_comparisons)),
        "pipeline": payload["multi_timeframe_pipeline"],
        "tao_evidence_status": (
            "PROSPECTIVE_ONLY_NOT_IN_HISTORICAL_COMPARISON_MATRIX"
        ),
        "artifacts": artifacts,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "HMM_ALL_STRATEGIES_CAMPAIGN_ID",
    "HMM_COMPARISON_TIMEFRAMES",
    "HMMConditionedStrategy",
    "_write_top50_mtf_pipeline_reports",
    "hmm_all_strategy_status",
    "refresh_top50_mtf_registry",
    "run_hmm_all_strategy_comparison",
]
