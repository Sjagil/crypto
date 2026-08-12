"""Per-strategy robustness charts and causal market-regime attribution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from research.backtest import BacktestResult
from research.seven_year import _rolling_twelve_month_metrics
from research.trading_math import bootstrap_expectancy
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    utc_iso,
)


def _daily_returns(result: BacktestResult) -> pd.Series:
    equity = result.equity_curve["equity"].astype(float)
    return (
        equity.resample("1D")
        .last()
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _regime_attribution(
    strategy_returns: pd.Series,
    benchmark: pd.DataFrame,
) -> dict[str, Any]:
    close = benchmark["close"].astype(float).resample("1D").last().dropna()
    trend = close > close.ewm(span=200, adjust=False, min_periods=60).mean()
    volatility = close.pct_change().rolling(20, min_periods=10).std()
    high_volatility = volatility > volatility.rolling(180, min_periods=40).median()
    regimes = pd.Series("UNKNOWN", index=close.index, dtype="object")
    regimes.loc[trend & ~high_volatility] = "BTC_BULL_LOW_VOL"
    regimes.loc[trend & high_volatility] = "BTC_BULL_HIGH_VOL"
    regimes.loc[~trend & ~high_volatility] = "RISK_OFF_LOW_VOL"
    regimes.loc[~trend & high_volatility] = "RISK_OFF_HIGH_VOL"
    aligned = pd.concat(
        [strategy_returns.rename("strategy_return"), regimes.rename("regime")],
        axis=1,
        join="inner",
    ).dropna()
    rows: list[dict[str, Any]] = []
    for regime, values in aligned.groupby("regime", sort=True):
        returns = values["strategy_return"].astype(float)
        standard_deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        rows.append(
            {
                "regime": str(regime),
                "observations": int(len(returns)),
                "total_return": float(np.prod(1.0 + returns) - 1.0),
                "mean_daily_return": float(returns.mean()),
                "annualized_sharpe": (
                    float(returns.mean() / standard_deviation * math.sqrt(365.0))
                    if standard_deviation > 0.0
                    else 0.0
                ),
                "positive_period_fraction": float((returns > 0.0).mean()),
            }
        )
    ordered = sorted(rows, key=lambda row: row["total_return"], reverse=True)
    return {
        "classification": (
            "Causal daily BTC EMA200 trend crossed with backward-looking "
            "20-day realized-volatility state."
        ),
        "rows": rows,
        "best_regime": ordered[0]["regime"] if ordered else None,
        "failure_regime": ordered[-1]["regime"] if ordered else None,
    }


def generate_strategy_evidence_bundle(
    output_directory: Path,
    *,
    strategy_dna: str,
    timeframe: str,
    normal_result: BacktestResult,
    stressed_result: BacktestResult,
    stochastic: Mapping[str, Any],
    benchmark: pd.DataFrame,
) -> dict[str, Any]:
    """Persist one reconciled PNG/CSV/JSON evidence bundle."""

    selected = output_directory.resolve() / strategy_dna[:24]
    selected.mkdir(parents=True, exist_ok=True)
    normal_equity = normal_result.equity_curve["equity"].astype(float)
    stressed_equity = stressed_result.equity_curve["equity"].astype(float)
    common = normal_equity.index.intersection(stressed_equity.index)
    normal_equity = normal_equity.reindex(common)
    stressed_equity = stressed_equity.reindex(common)
    normal_returns = _daily_returns(normal_result)
    attribution = _regime_attribution(normal_returns, benchmark)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(normal_equity.index, normal_equity, label="normal costs")
    axes[0, 0].plot(stressed_equity.index, stressed_equity, label="stressed costs")
    axes[0, 0].set_title(f"Equity — {strategy_dna[:12]} — {timeframe}")
    axes[0, 0].legend()
    normal_drawdown = normal_equity / normal_equity.cummax() - 1.0
    stressed_drawdown = stressed_equity / stressed_equity.cummax() - 1.0
    axes[0, 1].plot(normal_drawdown.index, normal_drawdown, label="normal")
    axes[0, 1].plot(stressed_drawdown.index, stressed_drawdown, label="stressed")
    axes[0, 1].set_title("Drawdown")
    axes[0, 1].legend()

    labels = ("p05", "median", "p95")
    normal_mc = stochastic["normal"]["monte_carlo"]
    stressed_mc = stochastic["stressed"]["monte_carlo"]
    axes[1, 0].bar(
        np.arange(3) - 0.18,
        [
            normal_mc.get("p05_total_return", 0.0),
            normal_mc.get("median_total_return", 0.0),
            normal_mc.get("p95_total_return", 0.0),
        ],
        width=0.36,
        label="normal",
    )
    axes[1, 0].bar(
        np.arange(3) + 0.18,
        [
            stressed_mc.get("p05_total_return", 0.0),
            stressed_mc.get("median_total_return", 0.0),
            stressed_mc.get("p95_total_return", 0.0),
        ],
        width=0.36,
        label="stressed",
    )
    axes[1, 0].set_xticks(np.arange(3), labels)
    axes[1, 0].set_title("Stationary-bootstrap terminal return")
    axes[1, 0].legend()

    normal_profiles = stochastic["normal"]["dirichlet"].get("profiles") or []
    stressed_profiles = stochastic["stressed"]["dirichlet"].get("profiles") or []
    concentrations = [row["concentration_alpha"] for row in normal_profiles]
    axes[1, 1].plot(
        concentrations,
        [row["probability_positive"] for row in normal_profiles],
        marker="o",
        label="normal",
    )
    axes[1, 1].plot(
        concentrations,
        [row["probability_positive"] for row in stressed_profiles],
        marker="o",
        label="stressed",
    )
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_title("Dirichlet probability positive")
    axes[1, 1].set_xlabel("concentration alpha")
    axes[1, 1].legend()

    chart_path = selected / "strategy_robustness.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    regime_csv = selected / "regime_attribution.csv"
    pd.DataFrame(attribution["rows"]).to_csv(regime_csv, index=False)
    payload = {
        "generated_at": utc_iso(),
        "strategy_dna": strategy_dna,
        "timeframe": timeframe,
        "chart": str(chart_path),
        "regime_csv": str(regime_csv),
        "regime_attribution": attribution,
        "stochastic_policy_hash": stochastic.get("policy_hash"),
        "monte_carlo_passed": bool(
            stochastic["normal"]["monte_carlo"]["passed"]
            and stochastic["stressed"]["monte_carlo"]["passed"]
        ),
        "dirichlet_passed": bool(
            stochastic["normal"]["dirichlet"]["passed"]
            and stochastic["stressed"]["dirichlet"]["passed"]
        ),
    }
    json_path = selected / "strategy_evidence.json"
    atomic_write_json(json_path, payload)
    return {**payload, "json": str(json_path)}


def generate_campaign_stochastic_chart(
    report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Visualize already persisted Monte Carlo and Dirichlet campaign evidence."""

    source = report_path.resolve()
    report = dict(read_json(source))
    primary = dict(report["primary_result"])
    stochastic = dict(primary["gates"]["stochastic_validation"])
    strategy_dna = str(primary["strategy_dna_hash"])
    strategy_id = str(primary["strategy_id"])
    selected = output_directory.resolve() / strategy_dna[:24]
    selected.mkdir(parents=True, exist_ok=True)
    normal = dict(stochastic["normal"])
    stressed = dict(stochastic["stressed"])
    normal_mc = dict(normal["monte_carlo"])
    stressed_mc = dict(stressed["monte_carlo"])
    normal_dirichlet = list(normal["dirichlet"].get("profiles") or [])
    stressed_dirichlet = list(stressed["dirichlet"].get("profiles") or [])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = ("p05", "median", "p95")
    positions = np.arange(3)
    axes[0, 0].bar(
        positions - 0.18,
        [
            normal_mc["p05_total_return"],
            normal_mc["median_total_return"],
            normal_mc["p95_total_return"],
        ],
        width=0.36,
        label="normal costs",
    )
    axes[0, 0].bar(
        positions + 0.18,
        [
            stressed_mc["p05_total_return"],
            stressed_mc["median_total_return"],
            stressed_mc["p95_total_return"],
        ],
        width=0.36,
        label="stressed costs",
    )
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_title("Stationary-bootstrap terminal return")
    axes[0, 0].legend()

    axes[0, 1].bar(
        ("normal median", "normal p95", "stressed median", "stressed p95"),
        (
            normal_mc["median_maximum_drawdown"],
            normal_mc["p95_maximum_drawdown"],
            stressed_mc["median_maximum_drawdown"],
            stressed_mc["p95_maximum_drawdown"],
        ),
    )
    axes[0, 1].set_title("Monte Carlo maximum drawdown")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].plot(
        [row["concentration_alpha"] for row in normal_dirichlet],
        [row["probability_positive"] for row in normal_dirichlet],
        marker="o",
        label="normal costs",
    )
    axes[1, 0].plot(
        [row["concentration_alpha"] for row in stressed_dirichlet],
        [row["probability_positive"] for row in stressed_dirichlet],
        marker="o",
        label="stressed costs",
    )
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set_xlabel("Dirichlet concentration alpha")
    axes[1, 0].set_title("Probability of positive terminal return")
    axes[1, 0].legend()

    period_names = ("development", "validation", "confirmation")
    periods = dict(primary["periods"])
    stressed_periods = dict(primary["stressed_periods"])
    axes[1, 1].bar(
        positions - 0.18,
        [periods[name]["net_return"] for name in period_names],
        width=0.36,
        label="normal costs",
    )
    axes[1, 1].bar(
        positions + 0.18,
        [stressed_periods[name]["net_return"] for name in period_names],
        width=0.36,
        label="stressed costs",
    )
    axes[1, 1].set_xticks(positions, period_names)
    axes[1, 1].set_title("Chronological period net return")
    axes[1, 1].legend()
    figure.suptitle(f"{strategy_id} — persisted robustness evidence")

    chart_path = selected / "stochastic_robustness.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    payload = {
        "schema_version": "campaign_stochastic_chart_v1",
        "generated_at": utc_iso(),
        "strategy_id": strategy_id,
        "strategy_dna": strategy_dna,
        "source_report": str(source),
        "source_sha256": sha256_file(source),
        "chart": str(chart_path),
        "monte_carlo": {
            "normal_passed": bool(normal_mc["passed"]),
            "stressed_passed": bool(stressed_mc["passed"]),
            "simulations": int(normal_mc["simulations"]),
        },
        "dirichlet": {
            "normal_passed": bool(normal["dirichlet"]["passed"]),
            "stressed_passed": bool(stressed["dirichlet"]["passed"]),
            "simulations_per_profile": int(
                normal["dirichlet"]["simulations_per_profile"]
            ),
        },
        "orders_generated": 0,
    }
    json_path = selected / "stochastic_robustness.json"
    atomic_write_json(json_path, payload)
    return {**payload, "json": str(json_path)}


def generate_seven_year_run_evidence(
    result_path: Path,
) -> dict[str, Any]:
    """Build reconciled CSV and PNG evidence from one persisted 7y run."""

    source = result_path.resolve()
    report = dict(read_json(source))
    directory = source.parent
    mandatory_path = directory / "mandatory_statistics.json"
    rolling_path = directory / "rolling_12m_metrics.csv"
    annual = list(report.get("annual_returns") or [])
    regimes = list(report.get("regime_performance") or [])
    walk_forward = dict(report.get("walk_forward") or {})
    folds: list[dict[str, Any]] = []
    for mode in ("anchored", "rolling"):
        for row in dict(walk_forward.get(mode) or {}).get("folds") or []:
            folds.append({"mode": mode, **dict(row)})
    stress_rows = []
    for profile in ("normal_costs", "stressed_costs", "double_costs"):
        metrics = dict(report.get(profile) or {}).get("metrics") or {}
        stress_rows.append(
            {
                "cost_profile": profile,
                **{
                    key: metrics.get(key)
                    for key in (
                        "net_return",
                        "cagr",
                        "profit_factor",
                        "sharpe",
                        "sortino",
                        "calmar",
                        "maximum_drawdown",
                        "trade_count",
                        "transaction_costs_eur",
                    )
                },
            }
        )
    exports = {
        "annual_returns_csv": directory / "annual_returns.csv",
        "regime_performance_csv": directory / "regime_performance.csv",
        "walk_forward_csv": directory / "walk_forward.csv",
        "stress_results_csv": directory / "stress_results.csv",
        "capacity_csv": directory / "capacity.csv",
        "rolling_12m_metrics_csv": rolling_path,
        "mandatory_statistics_json": mandatory_path,
    }
    frames = {
        "annual_returns_csv": pd.DataFrame(annual),
        "regime_performance_csv": pd.DataFrame(regimes),
        "walk_forward_csv": pd.DataFrame(folds),
        "stress_results_csv": pd.DataFrame(stress_rows),
        "capacity_csv": pd.DataFrame(report.get("capacity") or []),
    }
    for name, frame in frames.items():
        atomic_write_text(
            exports[name],
            frame.to_csv(index=False, lineterminator="\n"),
        )

    equity_path = directory / "normal_costs_equity.csv"
    mandatory: dict[str, Any] = {
        "schema_version": "seven_year_mandatory_statistics_v1",
        "generated_at": utc_iso(),
        "strategy_id": report.get("strategy_id"),
        "strategy_dna_hash": report.get("strategy_dna_hash"),
        "status": report.get("status"),
        "available": equity_path.is_file(),
        "source_result": str(source),
        "source_sha256": sha256_file(source),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    if equity_path.is_file():
        equity_frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        equity = equity_frame["equity"].astype(float)
        returns = equity.pct_change().dropna()
        intervals = (
            equity.index.to_series().diff().dropna().dt.total_seconds()
        )
        median_seconds = (
            float(intervals.median()) if not intervals.empty else math.nan
        )
        periods_per_year = (
            365.2425 * 86_400.0 / median_seconds
            if math.isfinite(median_seconds) and median_seconds > 0
            else math.nan
        )
        elapsed_years = max(
            (equity.index[-1] - equity.index[0]).total_seconds()
            / (365.2425 * 86_400.0),
            1.0 / 365.2425,
        )
        trades_path = directory / "normal_costs_trades.csv"
        orders_path = directory / "normal_costs_orders.csv"
        trades = (
            pd.read_csv(trades_path)
            if trades_path.is_file()
            else pd.DataFrame()
        )
        orders = (
            pd.read_csv(orders_path)
            if orders_path.is_file()
            else pd.DataFrame()
        )
        bootstrap = None
        if "r_multiple" in trades and not trades.empty:
            r_values = trades["r_multiple"].astype(float).to_numpy()
            bootstrap = bootstrap_expectancy(
                r_values,
                bootstrap_samples=500,
                block_size=min(
                    max(1, int(round(len(r_values) ** (1 / 3)))),
                    len(r_values),
                ),
                seed=42,
            ).to_dict()
        rolling_frame, rolling_summary = _rolling_twelve_month_metrics(
            equity
        )
        atomic_write_text(
            rolling_path,
            rolling_frame.to_csv(index=True, lineterminator="\n"),
        )
        normal = dict(
            dict(report.get("normal_costs") or {}).get("metrics") or {}
        )
        costs = float(normal.get("transaction_costs_eur") or 0.0)
        initial = float(
            dict(report.get("normal_costs") or {}).get(
                "initial_cash_eur"
            )
            or 0.0
        )
        ending = float(
            dict(report.get("normal_costs") or {}).get(
                "ending_equity_eur"
            )
            or 0.0
        )
        fees = (
            float(orders["fee_eur"].astype(float).sum())
            if "fee_eur" in orders
            else None
        )
        slippage = (
            float(
                (
                    (
                        orders["fill_price"].astype(float)
                        - orders["raw_price"].astype(float)
                    ).abs()
                    * orders["quantity"].astype(float)
                ).sum()
            )
            if {
                "fill_price",
                "raw_price",
                "quantity",
            }.issubset(orders.columns)
            else None
        )
        full_expectancy = float(normal.get("net_expectancy_r") or 0.0)
        oos: dict[str, Any] = {}
        for mode in ("anchored", "rolling"):
            selected = dict(
                dict(report.get("walk_forward") or {}).get(mode) or {}
            )
            selected_folds = list(selected.get("folds") or [])
            oos_trades = sum(
                int(row.get("trade_count") or 0)
                for row in selected_folds
            )
            oos_expectancy = (
                sum(
                    float(row.get("net_expectancy_r") or 0.0)
                    * int(row.get("trade_count") or 0)
                    for row in selected_folds
                )
                / oos_trades
                if oos_trades
                else 0.0
            )
            oos[mode] = {
                "fold_count": len(selected_folds),
                "positive_folds": selected.get("positive_folds"),
                "oos_trade_count": oos_trades,
                "oos_weighted_expectancy_r": oos_expectancy,
                "oos_net_pnl_eur": float(
                    sum(
                        float(row.get("net_pnl_eur") or 0.0)
                        for row in selected_folds
                    )
                ),
                "walk_forward_efficiency": (
                    oos_expectancy / full_expectancy
                    if full_expectancy > 0
                    else None
                ),
                "valid": selected.get("valid"),
            }
        required_regimes = {
            "BULL_MARKET",
            "BEAR_MARKET",
            "SIDEWAYS_MARKET",
            "HIGH_VOLATILITY",
            "LOW_VOLATILITY",
            "LIQUIDITY_STRESS",
            "CRASH_PERIOD",
            "RECOVERY_PERIOD",
        }
        available_regimes = {
            str(row.get("regime")) for row in regimes
        }
        mandatory.update(
            {
                "normal_metrics": {
                    **normal,
                    "gross_return_estimate": (
                        (ending + costs) / initial - 1.0
                        if initial > 0
                        else None
                    ),
                    "annualized_volatility": (
                        float(
                            returns.std(ddof=1)
                            * math.sqrt(periods_per_year)
                        )
                        if len(returns) > 1
                        and math.isfinite(periods_per_year)
                        else 0.0
                    ),
                    "trades_per_year": (
                        len(trades) / elapsed_years
                    ),
                    "time_in_market": (
                        float(
                            (
                                equity_frame[
                                    "exposure_fraction"
                                ].astype(float)
                                > 1e-12
                            ).mean()
                        )
                        if "exposure_fraction" in equity_frame
                        else normal.get("average_exposure")
                    ),
                    "total_fees_eur": fees,
                    "total_slippage_eur": slippage,
                    "bootstrap_confidence_interval_lower_r": (
                        bootstrap["lower_expectancy_r"]
                        if bootstrap
                        else None
                    ),
                    "bootstrap_confidence_interval_upper_r": (
                        bootstrap["upper_expectancy_r"]
                        if bootstrap
                        else None
                    ),
                },
                "stress_profit_factor": dict(
                    dict(report.get("stressed_costs") or {}).get(
                        "metrics"
                    )
                    or {}
                ).get("profit_factor"),
                "double_cost_profit_factor": dict(
                    dict(report.get("double_costs") or {}).get(
                        "metrics"
                    )
                    or {}
                ).get("profit_factor"),
                "bootstrap_expectancy": bootstrap,
                "rolling_12m": rolling_summary,
                "annual_summary": {
                    "best_year": max(
                        (
                            float(row.get("net_return") or 0.0)
                            for row in annual
                        ),
                        default=None,
                    ),
                    "worst_year": min(
                        (
                            float(row.get("net_return") or 0.0)
                            for row in annual
                        ),
                        default=None,
                    ),
                    "positive_years": sum(
                        float(row.get("net_return") or 0.0) > 0.0
                        for row in annual
                    ),
                    "negative_years": sum(
                        float(row.get("net_return") or 0.0) <= 0.0
                        for row in annual
                    ),
                },
                "out_of_sample_performance": oos,
                "parameter_stability": report.get(
                    "parameter_stability"
                ),
                "neighborhood_stability": report.get(
                    "neighborhood_stability"
                )
                or report.get("parameter_stability"),
                "capacity": report.get("capacity"),
                "regime_coverage": {
                    "required": sorted(required_regimes),
                    "available": sorted(available_regimes),
                    "missing": sorted(
                        required_regimes - available_regimes
                    ),
                    "complete": required_regimes.issubset(
                        available_regimes
                    ),
                    "legacy_missing_labels_are_not_fabricated": True,
                },
                "definitions": {
                    "gross_return_estimate": (
                        "NET_ENDING_EQUITY_PLUS_RECORDED_TRANSACTION_COSTS"
                    ),
                    "profit_factor": "CLOSED_TRADE_R_MULTIPLES",
                    "rolling_profit_factor": (
                        "POSITIVE_PERIOD_RETURNS_DIVIDED_BY_ABSOLUTE_NEGATIVE_PERIOD_RETURNS"
                    ),
                    "ranking_uses_net_results_only": True,
                },
            }
        )
    else:
        mandatory["unavailable_reason"] = (
            "NO_EXECUTABLE_SEVEN_YEAR_EQUITY_CURVE"
        )
        atomic_write_text(rolling_path, "")
    atomic_write_json(mandatory_path, mandatory)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    if equity_path.is_file():
        equity_frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        equity = equity_frame["equity"].astype(float)
        axes[0, 0].plot(equity.index, equity, color="#1864ab")
        axes[0, 0].set_title("Net equity — normal costs")
        drawdown = equity / equity.cummax() - 1.0
        axes[0, 1].fill_between(
            drawdown.index,
            drawdown.to_numpy(dtype=float),
            0.0,
            color="#c92a2a",
            alpha=0.35,
        )
        axes[0, 1].set_title("Drawdown")
    else:
        reason = ", ".join(report.get("status_reasons") or [])
        axes[0, 0].text(
            0.5,
            0.5,
            f"No executable 7y curve\n{report.get('status')}\n{reason}",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 1].axis("off")

    if annual:
        annual_frame = pd.DataFrame(annual)
        axes[1, 0].bar(
            annual_frame["year"].astype(str),
            annual_frame["net_return"].astype(float),
            color=[
                "#2b8a3e" if value >= 0 else "#c92a2a"
                for value in annual_frame["net_return"].astype(float)
            ],
        )
        axes[1, 0].set_title("Calendar-year net return")
        axes[1, 0].tick_params(axis="x", rotation=35)
    else:
        axes[1, 0].axis("off")
    if regimes:
        regime_frame = pd.DataFrame(regimes)
        axes[1, 1].barh(
            regime_frame["regime"].astype(str),
            regime_frame["compounded_return"].astype(float),
            color="#5f3dc4",
        )
        axes[1, 1].set_title("Retrospective regime attribution")
    else:
        axes[1, 1].axis("off")
    figure.suptitle(
        f"{report.get('strategy_id')} — {report.get('market')} "
        f"{report.get('timeframe')} — {report.get('status')}"
    )
    chart_path = directory / "seven_year_evidence.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)

    payload = {
        "schema_version": "seven_year_run_evidence_v1",
        "generated_at": utc_iso(),
        "strategy_id": report.get("strategy_id"),
        "strategy_dna_hash": report.get("strategy_dna_hash"),
        "status": report.get("status"),
        "source_result": str(source),
        "source_sha256": sha256_file(source),
        "chart": str(chart_path.resolve()),
        "exports": {
            name: str(path.resolve()) for name, path in exports.items()
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    evidence_path = directory / "seven_year_evidence.json"
    atomic_write_json(evidence_path, payload)
    return {**payload, "evidence_json": str(evidence_path.resolve())}


def generate_seven_year_evidence_tree(
    directory: Path,
) -> dict[str, Any]:
    """Generate evidence for every persisted run, including exclusions."""

    root = directory.resolve()
    rows = [
        generate_seven_year_run_evidence(path)
        for path in sorted((root / "runs").glob("**/seven_year_result.json"))
    ]
    payload = {
        "schema_version": "seven_year_evidence_index_v1",
        "generated_at": utc_iso(),
        "run_count": len(rows),
        "runs": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    index_path = root / "evidence_index.json"
    atomic_write_json(index_path, payload)
    return {**payload, "path": str(index_path)}


__all__ = [
    "generate_campaign_stochastic_chart",
    "generate_seven_year_evidence_tree",
    "generate_seven_year_run_evidence",
    "generate_strategy_evidence_bundle",
]
