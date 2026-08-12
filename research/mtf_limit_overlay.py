"""Causal 15m limit-entry overlays for frozen positive 1h/2h DNA.

The parent strategy remains the alpha source.  After a fully closed parent
candle signals, a bounded buy limit rests at the already-known breakout level
for a small number of *future* 15m bars.  This module is research-only and
cannot grant live authority or submit an order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from research.multi_timeframe_authority import (
    DEFAULT_MARKETS,
    MultiTimeframeParameters,
    _feature_frame,
    _load_frame,
    _metrics,
    _panel_net_return,
    _split_oos,
    _trade_return,
    _walk_forward,
)
from research.stochastic_validation import (
    StochasticValidationPolicy,
    validate_strategy_return_paths,
)
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "mtf_15m_limit_overlay_v1"


@dataclass(frozen=True, slots=True)
class LimitOverlayParameters:
    parent: MultiTimeframeParameters
    entry_window_15m_bars: int
    normal_side_cost_bps: float = 27.0
    stressed_side_cost_bps: float = 50.0

    @property
    def strategy_id(self) -> str:
        return f"LIMIT15M_{self.parent.strategy_id}"

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "strategy_id": self.strategy_id,
                "parent_strategy_id": self.parent.strategy_id,
                "parent_strategy_dna": self.parent.dna_hash,
                "parameters": {
                    "parent": asdict(self.parent),
                    "entry_window_15m_bars": self.entry_window_15m_bars,
                    "normal_side_cost_bps": self.normal_side_cost_bps,
                    "stressed_side_cost_bps": self.stressed_side_cost_bps,
                },
                "entry": (
                    "after closed parent breakout, bounded buy limit at the "
                    "causally known parent Donchian level on future 15m bars"
                ),
                "execution": "15m limit fill, no chase and no market fallback",
                "exit": "parent exit at next 15m open or bounded intrabar stop",
                "closed_candle_only": True,
                "long_only_spot": True,
            },
            length=64,
        )


def overlay_parameter_set() -> tuple[LimitOverlayParameters, ...]:
    return (
        LimitOverlayParameters(
            parent=MultiTimeframeParameters(
                timeframe="1h",
                entry_lookback=180,
                exit_lookback=60,
            ),
            entry_window_15m_bars=4,
        ),
        LimitOverlayParameters(
            parent=MultiTimeframeParameters(
                timeframe="2h",
                entry_lookback=240,
                exit_lookback=72,
            ),
            entry_window_15m_bars=8,
        ),
    )


def _parent_frame(fifteen_minute: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    bars = 4 if timeframe == "1h" else 8 if timeframe == "2h" else 0
    if not bars:
        raise ValueError("limit overlay parent must be 1h or 2h")
    grouped = fifteen_minute.resample(
        timeframe,
        origin="epoch",
        label="left",
        closed="left",
    )
    parent = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    return parent.loc[counts.eq(bars)].dropna()


def _overlay_stop(signal: Mapping[str, Any], fill: float, parameters: LimitOverlayParameters) -> float:
    atr_stop = float(signal["close"]) - parameters.parent.atr_stop_multiple * float(
        signal["atr"]
    )
    fractal = float(signal.get("confirmed_fractal_low") or np.nan)
    stop = atr_stop
    if (
        parameters.parent.use_confirmed_fractal_stop
        and np.isfinite(fractal)
        and 0.0 < fractal < fill
    ):
        stop = max(stop, fractal)
    return min(stop, fill * 0.999)


def simulate_limit_overlay_market(
    fifteen_minute: pd.DataFrame,
    parameters: LimitOverlayParameters,
    *,
    side_cost_bps: float | None = None,
    parent_featured: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Replay future-only 15m limit fills after closed parent signals."""

    side_cost = (
        parameters.normal_side_cost_bps
        if side_cost_bps is None
        else float(side_cost_bps)
    )
    parent = _parent_frame(fifteen_minute, parameters.parent.timeframe)
    featured = (
        _feature_frame(parent, parameters.parent)
        if parent_featured is None
        else parent_featured.copy()
    )
    entries = {
        pd.Timestamp(row["decision_at"]): dict(row)
        for _, row in featured.loc[featured["entry_signal"].astype(bool)].iterrows()
    }
    exits = {
        pd.Timestamp(row["decision_at"])
        for _, row in featured.loc[featured["exit_signal"].astype(bool)].iterrows()
    }
    rows = fifteen_minute.reset_index()
    timestamp_column = str(fifteen_minute.index.name or "timestamp")
    if timestamp_column not in rows:
        timestamp_column = rows.columns[0]
    trades: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    position: dict[str, Any] | None = None

    for index, bar in rows.iterrows():
        timestamp = pd.Timestamp(bar[timestamp_column])
        if position is not None and timestamp in exits:
            exit_price = float(bar["open"])
            trades.append(
                {
                    **position,
                    "exit_timestamp": timestamp.isoformat(),
                    "exit_price": exit_price,
                    "exit_reason": "PARENT_SIGNAL_NEXT_15M_OPEN",
                    "net_return": _trade_return(
                        float(position["entry_price"]),
                        exit_price,
                        side_cost,
                    ),
                }
            )
            position = None
            pending = None

        if pending is not None and position is None:
            if index > int(pending["expires_index"]):
                pending = None
            elif index >= int(pending["active_index"]):
                limit = float(pending["limit_price"])
                fill = (
                    min(float(bar["open"]), limit)
                    if float(bar["open"]) <= limit
                    else limit
                    if float(bar["low"]) <= limit <= float(bar["high"])
                    else None
                )
                if fill is not None:
                    stop = _overlay_stop(pending["signal"], fill, parameters)
                    risk = fill - stop
                    if risk > 0.0:
                        position = {
                            "entry_timestamp": timestamp.isoformat(),
                            "entry_price": fill,
                            "initial_stop": stop,
                            "target": fill + risk * parameters.parent.reward_risk,
                            "signal_timestamp": pending["signal_timestamp"],
                            "limit_price": limit,
                            "bars_waited": index - int(pending["active_index"]),
                        }
                    pending = None

        if position is not None and float(bar["low"]) <= float(position["initial_stop"]):
            stop = float(position["initial_stop"])
            exit_price = min(float(bar["open"]), stop)
            trades.append(
                {
                    **position,
                    "exit_timestamp": timestamp.isoformat(),
                    "exit_price": exit_price,
                    "exit_reason": "BOUNDED_STOP",
                    "net_return": _trade_return(
                        float(position["entry_price"]),
                        exit_price,
                        side_cost,
                    ),
                }
            )
            position = None

        signal = entries.get(timestamp)
        if signal is not None and position is None and pending is None:
            active_index = index + 1
            pending = {
                "signal": signal,
                "signal_timestamp": timestamp.isoformat(),
                "limit_price": float(signal["entry_level"]),
                "active_index": active_index,
                "expires_index": active_index + parameters.entry_window_15m_bars - 1,
            }

    if position is not None:
        bar = rows.iloc[-1]
        exit_price = float(bar["close"])
        trades.append(
            {
                **position,
                "exit_timestamp": pd.Timestamp(bar[timestamp_column]).isoformat(),
                "exit_price": exit_price,
                "exit_reason": "MARK_TO_END",
                "net_return": _trade_return(
                    float(position["entry_price"]),
                    exit_price,
                    side_cost,
                ),
            }
        )
    return trades


def _evaluate(
    settings: Settings,
    parameters: LimitOverlayParameters,
    markets: Sequence[str],
) -> dict[str, Any]:
    frames = {market: _load_frame(settings, market, "15m") for market in markets}
    normal = {
        market: simulate_limit_overlay_market(frame, parameters)
        for market, frame in frames.items()
    }
    stressed = {
        market: simulate_limit_overlay_market(
            frame,
            parameters,
            side_cost_bps=parameters.stressed_side_cost_bps,
        )
        for market, frame in frames.items()
    }
    starts = {market: frame.index.min() for market, frame in frames.items()}
    ends = {market: frame.index.max() for market, frame in frames.items()}
    normal_per_market = {market: _metrics(rows) for market, rows in normal.items()}
    stressed_per_market = {market: _metrics(rows) for market, rows in stressed.items()}
    development: dict[str, list[dict[str, Any]]] = {}
    oos: dict[str, list[dict[str, Any]]] = {}
    for market, rows in normal.items():
        split = starts[market] + (ends[market] - starts[market]) * 0.70
        development[market], oos[market] = _split_oos(rows, split)
    all_normal = [dict(row, market=market) for market, rows in normal.items() for row in rows]
    all_stressed = [dict(row, market=market) for market, rows in stressed.items() for row in rows]
    all_oos = [dict(row, market=market) for market, rows in oos.items() for row in rows]
    normal_metrics = _metrics(all_normal)
    normal_metrics["panel_net_return"] = _panel_net_return(normal_per_market)
    stressed_metrics = _metrics(all_stressed)
    stressed_metrics["panel_net_return"] = _panel_net_return(stressed_per_market)
    oos_metrics = _metrics(all_oos)
    oos_metrics["panel_net_return"] = _panel_net_return(
        {market: _metrics(rows) for market, rows in oos.items()}
    )
    stochastic = validate_strategy_return_paths(
        [float(row["net_return"]) for row in all_normal],
        [float(row["net_return"]) for row in all_stressed],
        policy=StochasticValidationPolicy(
            simulations=2_000,
            expected_block_length=10,
            maximum_drawdown=0.50,
            maximum_drawdown_breach_probability=0.20,
            maximum_terminal_loss_probability=0.20,
            minimum_p05_total_return=-0.20,
            dirichlet_blocks=8,
            minimum_observations=30,
            seed=20260731 + int(parameters.dna_hash[:8], 16),
            batch_size=128,
        ),
    )
    research_positive = bool(
        normal_metrics["panel_net_return"] > 0.0
        and normal_metrics["profit_factor"] > 1.0
        and normal_metrics["expectancy"] > 0.0
    )
    capital_warnings = []
    if stressed_metrics["profit_factor"] <= 1.0:
        capital_warnings.append("STRESSED_PROFIT_FACTOR_NOT_POSITIVE")
    if oos_metrics["panel_net_return"] <= 0.0:
        capital_warnings.append("OUT_OF_SAMPLE_NET_RETURN_NOT_POSITIVE")
    if not all(stochastic.get("checks", {}).values()):
        capital_warnings.append("MONTE_CARLO_OR_DIRICHLET_WARNING")
    return {
        "strategy_id": parameters.strategy_id,
        "strategy_dna_hash": parameters.dna_hash,
        "parent_strategy_id": parameters.parent.strategy_id,
        "parent_strategy_dna_hash": parameters.parent.dna_hash,
        "family": "CAUSAL_PARENT_ALPHA_15M_LIMIT_EXECUTION",
        "parent_timeframe": parameters.parent.timeframe,
        "execution_timeframe": "15m",
        "markets": list(markets),
        "parameters": {
            "parent": asdict(parameters.parent),
            "entry_window_15m_bars": parameters.entry_window_15m_bars,
            "normal_side_cost_bps": parameters.normal_side_cost_bps,
            "stressed_side_cost_bps": parameters.stressed_side_cost_bps,
        },
        "normal": normal_metrics,
        "normal_per_market": normal_per_market,
        "stressed": stressed_metrics,
        "out_of_sample": oos_metrics,
        "walk_forward": _walk_forward(normal, starts, ends),
        "stochastic_validation": stochastic,
        "research_positive": research_positive,
        "paper_eligible": research_positive,
        "live_authority_granted": False,
        "operator_live_dna_approval_required": True,
        "capital_warnings": capital_warnings,
        "integrity": {
            "real_market_data": True,
            "closed_parent_candle_only": True,
            "future_15m_bars_only": True,
            "no_lookahead": True,
            "no_repainting": True,
            "bounded_stop": True,
            "long_only_spot": True,
            "historical_orderflow_used": False,
        },
    }


def validate_mtf_limit_overlays(
    settings: Settings,
    *,
    markets: Sequence[str] = DEFAULT_MARKETS,
) -> dict[str, Any]:
    rows = [_evaluate(settings, parameters, markets) for parameters in overlay_parameter_set()]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "candidate_count": len(rows),
        "research_positive_count": sum(row["research_positive"] for row in rows),
        "paper_eligible_count": sum(row["paper_eligible"] for row in rows),
        "live_authority_granted_count": 0,
        "markets": list(markets),
        "parent_timeframes": ["1h", "2h"],
        "execution_timeframe": "15m",
        "limit_order_policy": "BOUNDED_NO_CHASE_NO_MARKET_FALLBACK",
        "stationary_bootstrap_monte_carlo": True,
        "dirichlet_time_concentration_stress": True,
        "candidates": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    reports = settings.paths.lab_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "mtf_15m_limit_overlay_v1.json"
    atomic_write_json(path, payload)
    pd.DataFrame(
        [
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "parent_timeframe": row["parent_timeframe"],
                "trade_count": row["normal"]["trade_count"],
                "net_return": row["normal"]["panel_net_return"],
                "profit_factor": row["normal"]["profit_factor"],
                "maximum_drawdown": row["normal"]["maximum_drawdown"],
                "stressed_profit_factor": row["stressed"]["profit_factor"],
                "oos_net_return": row["out_of_sample"]["panel_net_return"],
                "positive_folds": row["walk_forward"]["positive_folds"],
                "research_positive": row["research_positive"],
            }
            for row in rows
        ]
    ).to_csv(reports / "mtf_15m_limit_overlay_v1.csv", index=False)
    return {**payload, "artifact": str(path)}


def load_validated_mtf_limit_overlay_candidates(
    settings: Settings,
) -> list[dict[str, Any]]:
    """Load positive overlay DNA for paper; never grant live authority."""

    path = settings.paths.lab_dir / "reports" / "mtf_15m_limit_overlay_v1.json"
    if not path.is_file():
        return []
    report = dict(read_json(path))
    candidates: list[dict[str, Any]] = []
    for raw in report.get("candidates") or []:
        row = dict(raw)
        if row.get("research_positive") is not True:
            continue
        frozen_hash = stable_hash(
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "parent_strategy_dna_hash": row["parent_strategy_dna_hash"],
                "timeframe": "15m",
                "markets": list(row["markets"]),
                "parameters": dict(row["parameters"]),
                "paper_adapter": "MTF_15M_LIMIT_OVERLAY",
            },
            length=64,
        )
        candidates.append(
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "economic_hypothesis_family": row["family"],
                "timeframe": "15m",
                "markets": list(row["markets"]),
                "parameters": dict(row["parameters"]),
                "metrics": {
                    "net_return": row["normal"]["panel_net_return"],
                    "profit_factor": row["normal"]["profit_factor"],
                    "stressed_net_return": row["stressed"]["panel_net_return"],
                    "stressed_profit_factor": row["stressed"]["profit_factor"],
                    "maximum_drawdown": row["normal"]["maximum_drawdown"],
                    "trade_count": row["normal"]["trade_count"],
                    "net_expectancy_r": row["normal"]["expectancy"],
                    "out_of_sample_net_return": row["out_of_sample"][
                        "panel_net_return"
                    ],
                    "walk_forward_positive_folds": row["walk_forward"][
                        "positive_folds"
                    ],
                },
                "integrity": dict(row["integrity"]),
                "lifecycle": "BACKTEST_POSITIVE",
                "paper_adapter": "MTF_15M_LIMIT_OVERLAY",
                "frozen_candidate_hash": frozen_hash,
                "source_report": str(path),
                "auto_live_promotion": False,
                "operator_live_dna_approval_required": True,
            }
        )
    return candidates


__all__ = [
    "LimitOverlayParameters",
    "load_validated_mtf_limit_overlay_candidates",
    "overlay_parameter_set",
    "simulate_limit_overlay_market",
    "validate_mtf_limit_overlays",
]
