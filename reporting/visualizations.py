"""Deterministic batch chart generation with explicit missing-data statuses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.common import atomic_write_json, utc_now

PlotKind = Literal["line", "bar", "distribution", "scatter", "heatmap", "area"]
LOGGER = logging.getLogger("crypto.visualizations")

PLOT_SPECS: dict[str, tuple[str, PlotKind, tuple[str, ...]]] = {
    # Market data
    "price_and_volume": ("market", "line", ("close", "volume")),
    "missing_bars_by_provider": ("market", "bar", ("missing_bars",)),
    "provider_coverage": ("provider", "bar", ("coverage",)),
    "provider_latency": ("provider", "distribution", ("latency_ms",)),
    "websocket_message_throughput": ("websocket", "line", ("messages_per_second",)),
    "reconnect_timeline": ("websocket", "bar", ("reconnects",)),
    "spread_history": ("orderbook", "line", ("spread_bps",)),
    "orderbook_imbalance": ("orderbook", "line", ("imbalance",)),
    "depth_profile": ("depth", "line", ("bid_depth", "ask_depth")),
    "estimated_slippage_by_order_size": ("depth", "line", ("slippage_bps",)),
    # Research
    "equity_curve": ("research", "line", ("equity",)),
    "drawdown_curve": ("research", "line", ("drawdown",)),
    "underwater_curve": ("research", "area", ("drawdown",)),
    "rolling_return": ("research", "line", ("rolling_return",)),
    "rolling_volatility": ("research", "line", ("rolling_volatility",)),
    "rolling_sharpe": ("research", "line", ("rolling_sharpe",)),
    "trade_return_distribution": ("trades", "distribution", ("return",)),
    "r_multiple_distribution": ("trades", "distribution", ("r_multiple",)),
    "win_and_loss_streak_distribution": ("trades", "distribution", ("streak",)),
    "mae_versus_mfe": ("trades", "scatter", ("mae", "mfe")),
    "monthly_returns": ("monthly", "heatmap", ("return",)),
    "strategy_contribution": ("contribution", "bar", ("strategy", "value")),
    "symbol_contribution": ("contribution", "bar", ("symbol", "value")),
    "regime_contribution": ("contribution", "bar", ("regime", "value")),
    "transaction_cost_contribution": ("contribution", "bar", ("cost_type", "value")),
    # Optimization
    "parameter_heatmap": ("optimization", "heatmap", ("parameter_x", "parameter_y", "score")),
    "parameter_neighborhood_stability": ("optimization", "line", ("stability",)),
    "train_versus_validation_performance": (
        "optimization",
        "scatter",
        ("train_score", "validation_score"),
    ),
    "walk_forward_fold_performance": ("optimization", "bar", ("fold_score",)),
    "monte_carlo_terminal_equity_distribution": (
        "monte_carlo",
        "distribution",
        ("terminal_equity",),
    ),
    "monte_carlo_drawdown_distribution": (
        "monte_carlo",
        "distribution",
        ("maximum_drawdown",),
    ),
    "risk_of_ruin_curve": ("monte_carlo", "line", ("risk_of_ruin",)),
    "candidate_rejection_reason_counts": ("optimization", "bar", ("rejection_reason",)),
    # Continuous combinatorial lab
    "leaderboard_rank_change": ("lab_rank", "bar", ("entry", "rank_change")),
    "combination_size_performance": (
        "lab_size",
        "bar",
        ("combination_size", "robust_score"),
    ),
    "family_contribution": ("lab_family", "bar", ("family", "robust_score")),
    "block_frequency_among_top_strategies": (
        "lab_blocks",
        "bar",
        ("block",),
    ),
    "redundancy_versus_performance": (
        "lab_redundancy",
        "scatter",
        ("redundancy_score", "robust_score"),
    ),
    "parameter_distributions": (
        "lab_parameters",
        "distribution",
        ("parameter_value",),
    ),
    "parameter_heatmaps_lab": (
        "lab_parameter_heatmap",
        "heatmap",
        ("parameter_x", "parameter_y", "score"),
    ),
    "performance_by_asset": ("lab_asset", "bar", ("asset", "robust_score")),
    "performance_by_timeframe": (
        "lab_timeframe",
        "bar",
        ("timeframe", "robust_score"),
    ),
    "performance_by_universe_snapshot": (
        "lab_universe",
        "bar",
        ("universe_snapshot", "robust_score"),
    ),
    "queue_throughput": ("lab_events", "line", ("throughput",)),
    "jobs_completed_per_hour": ("lab_events", "bar", ("completed",)),
    "worker_utilization": ("lab_workers", "line", ("utilization",)),
    "experiment_duration_distribution": (
        "lab_events",
        "distribution",
        ("duration",),
    ),
    "leaderboard_decay": ("lab_decay", "line", ("robust_score",)),
    "strategy_lifecycle_transitions": (
        "lab_lifecycle",
        "bar",
        ("lifecycle_status",),
    ),
    # Portfolio
    "allocation": ("portfolio", "bar", ("allocation",)),
    "open_risk": ("portfolio", "bar", ("open_risk",)),
    "correlation_heatmap": ("correlation", "heatmap", ()),
    "rolling_btc_beta": ("portfolio", "line", ("btc_beta",)),
    "marginal_risk_contribution": ("portfolio", "bar", ("marginal_risk",)),
    "realized_and_unrealized_pnl": (
        "portfolio",
        "line",
        ("realized_pnl", "unrealized_pnl"),
    ),
    "daily_pnl": ("portfolio", "bar", ("daily_pnl",)),
    "exposure_through_time": ("portfolio", "line", ("exposure",)),
    # Macro
    "fear_and_greed": ("macro", "line", ("sentiment_fear_greed",)),
    "btc_dominance": ("macro", "line", ("dominance_btc_dominance",)),
    "stablecoin_dominance": ("macro", "line", ("dominance_stablecoin_dominance",)),
    "breadth": ("macro", "line", ("breadth_fraction_above_mean_50d",)),
    "funding": ("macro", "line", ("derivatives_funding_rate",)),
    "open_interest": ("macro", "line", ("derivatives_open_interest",)),
    "liquidations": (
        "macro",
        "line",
        ("derivatives_long_liquidations", "derivatives_short_liquidations"),
    ),
    "global_risk_score": ("macro", "line", ("crypto_risk_score",)),
    "event_risk_windows": ("macro", "area", ("events_high_impact_event_risk",)),
    "gex_by_strike": ("gex_strike", "bar", ("strike", "gross_gex")),
    "gex_by_expiry": ("gex_expiry", "bar", ("expiry", "gross_gex")),
    "net_gex_proxy_through_time": ("macro", "line", ("gex_net_gex_proxy",)),
    "spot_distance_from_dominant_gamma_levels": (
        "macro",
        "line",
        ("gex_spot_distance_from_dominant_gamma",),
    ),
}


@dataclass(frozen=True)
class PlotResult:
    name: str
    category: str
    status: str
    reason_code: str
    files: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()


class VisualizationReporter:
    def __init__(
        self,
        output_dir: Path | str,
        *,
        save_svg: bool = False,
        dpi: int = 120,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_svg = save_svg
        self.dpi = dpi
        self.results: list[PlotResult] = []
        sns.set_theme(style="whitegrid")

    def generate(
        self, datasets: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        self.results = []
        for name, (category, kind, columns) in PLOT_SPECS.items():
            frame = datasets.get(category)
            if frame is None or frame.empty:
                self.results.append(
                    PlotResult(name, category, "SKIPPED", "MISSING_DATASET")
                )
                continue
            missing = tuple(column for column in columns if column not in frame)
            if missing:
                self.results.append(
                    PlotResult(
                        name,
                        category,
                        "SKIPPED",
                        f"MISSING_COLUMNS:{','.join(missing)}",
                    )
                )
                continue
            try:
                files = self.plot(
                    frame,
                    name=name,
                    title=name.replace("_", " ").title(),
                    kind=kind,
                    columns=columns,
                )
                self.results.append(
                    PlotResult(
                        name,
                        category,
                        "PASSED",
                        "GENERATED",
                        tuple(str(item) for item in files),
                        columns,
                    )
                )
            except Exception as exc:
                self.results.append(
                    PlotResult(
                        name,
                        category,
                        "FAILED",
                        f"{type(exc).__name__}:{exc}",
                        columns=columns,
                    )
                )
        return self.write_index()

    def plot(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        title: str,
        kind: PlotKind,
        columns: tuple[str, ...],
    ) -> tuple[Path, ...]:
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        try:
            selected = frame.copy()
            if isinstance(selected.index, pd.DatetimeIndex):
                selected = selected.sort_index()
                axis.set_xlabel("UTC time")
            if kind == "line":
                for column in columns:
                    axis.plot(selected.index, selected[column], label=column)
                if len(columns) > 1:
                    axis.legend()
            elif kind == "area":
                column = columns[0]
                values = pd.to_numeric(selected[column], errors="coerce")
                axis.fill_between(selected.index, values, 0, alpha=0.45, label=column)
                axis.legend()
            elif kind == "distribution":
                sns.histplot(
                    data=selected,
                    x=columns[0],
                    kde=True,
                    ax=axis,
                )
            elif kind == "scatter":
                sns.scatterplot(data=selected, x=columns[0], y=columns[1], ax=axis)
                finite = selected[list(columns[:2])].dropna()
                if len(finite) >= 2:
                    lower = min(finite.iloc[:, 0].min(), finite.iloc[:, 1].min())
                    upper = max(finite.iloc[:, 0].max(), finite.iloc[:, 1].max())
                    if "train_versus" in name:
                        axis.plot([lower, upper], [lower, upper], linestyle="--", color="grey")
            elif kind == "heatmap":
                matrix = self._heatmap_matrix(selected, columns)
                sns.heatmap(matrix, cmap="vlag", center=0, ax=axis)
            elif kind == "bar":
                self._bar(selected, columns, axis)
            else:
                raise ValueError(f"unsupported plot kind: {kind}")
            axis.set_title(title)
            axis.set_ylabel(", ".join(columns) if columns else "value")
            if kind not in {"heatmap", "distribution"}:
                axis.grid(True, alpha=0.25)
            else:
                axis.grid(False)
            return self._save(figure, name)
        finally:
            plt.close(figure)

    @staticmethod
    def _heatmap_matrix(
        frame: pd.DataFrame, columns: tuple[str, ...]
    ) -> pd.DataFrame:
        if not columns:
            return frame.select_dtypes(include=[np.number])
        if len(columns) == 1:
            return frame[[columns[0]]].T
        if len(columns) == 3:
            return frame.pivot_table(
                index=columns[1],
                columns=columns[0],
                values=columns[2],
                aggfunc="mean",
            )
        return frame[list(columns)].corr()

    @staticmethod
    def _bar(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        axis: plt.Axes,
    ) -> None:
        if len(columns) >= 2 and not pd.api.types.is_numeric_dtype(frame[columns[0]]):
            grouped = frame.groupby(columns[0])[columns[-1]].sum()
            grouped.plot.bar(ax=axis)
        elif columns and not pd.api.types.is_numeric_dtype(frame[columns[0]]):
            frame[columns[0]].value_counts().plot.bar(ax=axis)
        else:
            values = frame[list(columns)] if columns else frame
            values.tail(50).plot.bar(ax=axis)

    def _save(self, figure: plt.Figure, name: str) -> tuple[Path, ...]:
        files = [self.output_dir / f"{name}.png"]
        figure.savefig(files[0], dpi=self.dpi, format="png")
        if self.save_svg:
            files.append(self.output_dir / f"{name}.svg")
            figure.savefig(files[-1], format="svg")
        return tuple(files)

    def write_index(self) -> dict[str, Any]:
        payload = {
            "generated_at": utc_now().isoformat(),
            "backend": matplotlib.get_backend(),
            "plot_count": len(self.results),
            "passed": sum(item.status == "PASSED" for item in self.results),
            "failed": sum(item.status == "FAILED" for item in self.results),
            "skipped": sum(item.status == "SKIPPED" for item in self.results),
            "plots": [
                {
                    "name": item.name,
                    "category": item.category,
                    "status": item.status,
                    "reason_code": item.reason_code,
                    "files": list(item.files),
                    "columns": list(item.columns),
                }
                for item in self.results
            ],
        }
        atomic_write_json(self.output_dir / "index.json", payload)
        LOGGER.info(
            "chart index generated",
            extra={
                "component": "reporting",
                "operation": "chart_generation",
                "status": "FAILED" if payload["failed"] else "PASSED",
                "reason_code": "PLOT_FAILURES" if payload["failed"] else "CHARTS_GENERATED",
            },
        )
        return payload


__all__ = ["PLOT_SPECS", "PlotResult", "VisualizationReporter"]
