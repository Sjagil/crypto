"""Causal PD-array, sweep/SMT, displacement and FVG entry research."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.settings import Settings
from research.features import atr, market_structure_features
from research.portfolio_selection import _profit_factor
from research.stochastic_validation import (
    policy_from_research_settings,
    validate_strategy_return_paths,
)
from utils.common import atomic_write_json, sha256_file, stable_hash

PD_ARRAY_FVG_CAMPAIGN = "PD_ARRAY_SWEEP_DISPLACEMENT_FVG_V1"
PD_ARRAY_FVG_ENGINE_VERSION = "1.0.0"
PD_ARRAY_FVG_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "ADA-EUR",
    "AVAX-EUR",
    "BNB-EUR",
    "XRP-EUR",
)
PERIODS_PER_YEAR = {
    "15m": 365.25 * 24.0 * 4.0,
    "1h": 365.25 * 24.0,
    "4h": 365.25 * 6.0,
}


@dataclass(frozen=True, slots=True)
class PdArrayFvgParameters:
    timeframe: str
    signal_mode: str
    fvg_entry_depth: float
    pd_lookback: int = 20
    signal_to_displacement_bars: int = 4
    pending_entry_bars: int = 8
    maximum_holding_bars: int = 24
    minimum_reward_risk: float = 1.25
    position_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.timeframe not in PERIODS_PER_YEAR:
            raise ValueError("unsupported PD/FVG timeframe")
        if self.signal_mode not in {"SWEEP", "SMT"}:
            raise ValueError("signal mode must be SWEEP or SMT")
        if self.fvg_entry_depth not in {0.50, 0.79}:
            raise ValueError("entry depth must be 0.50 or 0.79")
        if self.position_fraction != 0.20:
            raise ValueError("v1 position fraction is fixed at 20 percent")

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "campaign": PD_ARRAY_FVG_CAMPAIGN,
                "engine_version": PD_ARRAY_FVG_ENGINE_VERSION,
                "parameters": asdict(self),
                "long_only_spot": True,
            },
            length=64,
        )

    @property
    def strategy_id(self) -> str:
        depth = int(round(self.fvg_entry_depth * 100))
        return f"PD_FVG_{self.timeframe}_{self.signal_mode}_D{depth}"


def pd_array_fvg_parameter_set() -> tuple[PdArrayFvgParameters, ...]:
    rows = tuple(
        PdArrayFvgParameters(timeframe, signal, depth)
        for timeframe in ("15m", "1h", "4h")
        for signal in ("SWEEP", "SMT")
        for depth in (0.50, 0.79)
    )
    if len(rows) != 12 or len({row.dna_hash for row in rows}) != 12:
        raise RuntimeError("PD/FVG strategy DNA cardinality drift")
    return rows


def _normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    if "timestamp" in selected:
        selected.index = pd.to_datetime(selected.pop("timestamp"), utc=True)
    else:
        selected.index = pd.to_datetime(selected.index, utc=True)
    selected = selected[~selected.index.duplicated(keep="last")].sort_index()
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in selected]
    if missing:
        raise ValueError(f"OHLCV is missing columns: {missing}")
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(subset=list(required))
    price_columns = ["open", "high", "low", "close"]
    if (
        selected.empty
        or bool((selected[price_columns] <= 0.0).any().any())
        or bool((selected["volume"] < 0.0).any())
    ):
        raise ValueError("OHLCV contains no usable positive rows")
    if bool((selected["high"] < selected["low"]).any()):
        raise ValueError("OHLCV high is below low")
    return selected


def _smt_divergence(
    data: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    lookback: int = 5,
) -> pd.Series:
    aligned = pd.concat(
        [
            data["low"].rename("asset_low"),
            benchmark["low"].rename("benchmark_low"),
        ],
        axis=1,
        join="inner",
    )
    asset_prior = aligned["asset_low"].rolling(lookback).min().shift(1)
    benchmark_prior = (
        aligned["benchmark_low"].rolling(lookback).min().shift(1)
    )
    signal = (
        (aligned["asset_low"] < asset_prior)
        & (aligned["benchmark_low"] >= benchmark_prior)
    )
    return signal.reindex(data.index, fill_value=False).astype(bool)


def _formation_table(
    data: pd.DataFrame,
    benchmark: pd.DataFrame,
    parameters: PdArrayFvgParameters,
    *,
    structure: pd.DataFrame | None = None,
    volatility: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if structure is None:
        structure = market_structure_features(data)
    if volatility is None:
        volatility = atr(data, 14)
    previous_pd = (
        structure["bullish_fvg"].astype(int)
        + structure["bullish_order_block_proxy"].astype(int)
    ).rolling(parameters.pd_lookback).max().shift(1).fillna(0).astype(bool)
    pd_array_signal = previous_pd & structure["discount_zone"].fillna(False)
    if parameters.signal_mode == "SWEEP":
        trigger = structure["bullish_liquidity_sweep"].fillna(False)
    else:
        trigger = _smt_divergence(data, benchmark)
    qualified_trigger = pd_array_signal & trigger
    recent_trigger = (
        qualified_trigger.astype(int)
        .shift(1)
        .rolling(parameters.signal_to_displacement_bars, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    displacement_before_gap = (
        structure["bullish_displacement"].astype(bool).shift(1, fill_value=False)
    )
    formation = (
        structure["bullish_fvg"].fillna(False).astype(bool)
        & displacement_before_gap
        & recent_trigger
    )
    gap_lower = pd.to_numeric(
        structure["bullish_fvg_lower"], errors="coerce"
    )
    gap_upper = pd.to_numeric(
        structure["bullish_fvg_upper"], errors="coerce"
    )
    entry = gap_upper - parameters.fvg_entry_depth * (
        gap_upper - gap_lower
    )
    sweep_low = data["low"].rolling(
        parameters.signal_to_displacement_bars + 3
    ).min()
    stop = sweep_low - 0.10 * volatility
    confirmed_high = pd.to_numeric(
        structure["confirmed_fractal_high_price"], errors="coerce"
    ).ffill()
    prior_high = data["high"].rolling(20).max().shift(1)
    target = pd.concat([confirmed_high, prior_high], axis=1).max(axis=1)
    risk = entry - stop
    reward_risk = (target - entry) / risk.replace(0.0, np.nan)
    valid = (
        formation
        & entry.notna()
        & stop.notna()
        & target.notna()
        & (stop < entry)
        & (entry < target)
        & (reward_risk >= parameters.minimum_reward_risk)
    )
    table = pd.DataFrame(
        {
            "formation": valid,
            "entry": entry,
            "stop": stop,
            "target": target,
            "reward_risk": reward_risk,
        },
        index=data.index,
    )
    diagnostics = {
        "prior_pd_array_bars": int(pd_array_signal.sum()),
        "raw_trigger_bars": int(trigger.sum()),
        "qualified_trigger_bars": int(qualified_trigger.sum()),
        "displacement_fvg_sequences": int(formation.sum()),
        "valid_fvg_entry_setups": int(valid.sum()),
    }
    return table, diagnostics


def _metrics(
    equity: pd.Series,
    trades: list[dict[str, Any]],
    *,
    periods_per_year: float,
    total_costs: float,
) -> dict[str, Any]:
    returns = equity.pct_change(fill_method=None).dropna()
    elapsed_days = max(
        1.0,
        (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0,
    )
    years = elapsed_days / 365.25
    standard = float(returns.std(ddof=0))
    downside = returns[returns < 0.0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))))
        if len(downside)
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    pnls = np.asarray([float(row["net_pnl"]) for row in trades], dtype=float)
    wins = float(pnls[pnls > 0.0].sum()) if pnls.size else 0.0
    losses = float(-pnls[pnls < 0.0].sum()) if pnls.size else 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "net_total_return": total_return,
        "cagr": float((1.0 + total_return) ** (1.0 / years) - 1.0),
        "sharpe": (
            float(returns.mean() / standard * math.sqrt(periods_per_year))
            if standard > 0.0
            else 0.0
        ),
        "sortino": (
            float(
                returns.mean()
                / downside_deviation
                * math.sqrt(periods_per_year)
            )
            if downside_deviation > 0.0
            else 0.0
        ),
        "maximum_drawdown": float(drawdown.min()),
        "portfolio_period_profit_factor": _profit_factor(returns),
        "closed_position_profit_factor": (
            wins / losses if losses > 0.0 else (math.inf if wins > 0.0 else 0.0)
        ),
        "net_expectancy_eur": float(pnls.mean()) if pnls.size else 0.0,
        "trade_count": int(len(trades)),
        "win_rate": float(np.mean(pnls > 0.0)) if pnls.size else 0.0,
        "average_holding_bars": (
            float(np.mean([row["holding_bars"] for row in trades]))
            if trades
            else 0.0
        ),
        "modeled_costs_eur": float(total_costs),
        "observations": int(len(returns)),
        "periods_per_year": periods_per_year,
    }


def backtest_pd_array_fvg(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    parameters: PdArrayFvgParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    initial_equity: float = 10_000.0,
    prepared_data: pd.DataFrame | None = None,
    prepared_benchmark: pd.DataFrame | None = None,
    prepared_formation: tuple[pd.DataFrame, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Backtest a long-only limit-retrace path with conservative collisions."""

    if min(fee_rate, slippage_bps, spread_bps) < 0.0:
        raise ValueError("cost assumptions cannot be negative")
    data = prepared_data if prepared_data is not None else _normalized_ohlcv(frame)
    benchmark = (
        prepared_benchmark
        if prepared_benchmark is not None
        else _normalized_ohlcv(benchmark_frame)
    )
    if prepared_formation is None:
        table, diagnostics = _formation_table(data, benchmark, parameters)
    else:
        table, diagnostics = prepared_formation
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0
    cash = float(initial_equity)
    units = 0.0
    pending: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_values: list[float] = []
    total_costs = 0.0
    previous_equity = float(initial_equity)
    returns: list[float] = []
    timestamps = data.index
    lows = data["low"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    formations = table["formation"].to_numpy(dtype=bool)
    entries = table["entry"].to_numpy(dtype=float)
    stops = table["stop"].to_numpy(dtype=float)
    targets = table["target"].to_numpy(dtype=float)
    reward_risks = table["reward_risk"].to_numpy(dtype=float)
    for bar_number in range(len(data)):
        timestamp = timestamps[bar_number]
        low = lows[bar_number]
        high = highs[bar_number]
        close = closes[bar_number]
        if position is not None:
            exit_price: float | None = None
            exit_reason: str | None = None
            if low <= float(position["stop"]):
                exit_price = float(position["stop"])
                exit_reason = "STOP_LOSS"
            elif high >= float(position["target"]):
                exit_price = float(position["target"])
                exit_reason = "SWING_HIGH_TARGET"
            elif bar_number - int(position["entry_bar"]) >= parameters.maximum_holding_bars:
                exit_price = close
                exit_reason = "TIME_EXIT"
            if exit_price is not None:
                exit_notional = units * exit_price
                exit_cost = exit_notional * one_way_cost
                total_costs += exit_cost
                cash += exit_notional - exit_cost
                net_pnl = cash - float(position["equity_before_entry"])
                trades.append(
                    {
                        "entry_at": position["entry_at"],
                        "exit_at": timestamp.isoformat(),
                        "entry_price": float(position["entry"]),
                        "exit_price": exit_price,
                        "stop_loss": float(position["stop"]),
                        "target": float(position["target"]),
                        "net_pnl": net_pnl,
                        "holding_bars": bar_number - int(position["entry_bar"]),
                        "exit_reason": exit_reason,
                    }
                )
                units = 0.0
                position = None
        if position is None and pending is not None:
            if bar_number > int(pending["expires_bar"]):
                pending = None
            elif low <= float(pending["entry"]) <= high:
                equity_before = cash
                allocation = equity_before * parameters.position_fraction
                entry_cost = allocation * one_way_cost
                total_costs += entry_cost
                cash -= allocation + entry_cost
                units = allocation / float(pending["entry"])
                position = {
                    **pending,
                    "entry_at": timestamp.isoformat(),
                    "entry_bar": bar_number,
                    "equity_before_entry": equity_before,
                }
                pending = None
                if low <= float(position["stop"]):
                    exit_notional = units * float(position["stop"])
                    exit_cost = exit_notional * one_way_cost
                    total_costs += exit_cost
                    cash += exit_notional - exit_cost
                    trades.append(
                        {
                            "entry_at": position["entry_at"],
                            "exit_at": timestamp.isoformat(),
                            "entry_price": float(position["entry"]),
                            "exit_price": float(position["stop"]),
                            "stop_loss": float(position["stop"]),
                            "target": float(position["target"]),
                            "net_pnl": cash - equity_before,
                            "holding_bars": 0,
                            "exit_reason": "SAME_BAR_STOP_CONSERVATIVE",
                        }
                    )
                    units = 0.0
                    position = None
        if position is None and formations[bar_number]:
            pending = {
                "formed_at": timestamp.isoformat(),
                "entry": entries[bar_number],
                "stop": stops[bar_number],
                "target": targets[bar_number],
                "reward_risk": reward_risks[bar_number],
                "expires_bar": bar_number + parameters.pending_entry_bars,
            }
        equity_now = cash + units * close
        equity_values.append(equity_now)
        returns.append(equity_now / previous_equity - 1.0)
        previous_equity = equity_now
    if position is not None:
        final_price = float(data["close"].iloc[-1])
        exit_notional = units * final_price
        exit_cost = exit_notional * one_way_cost
        total_costs += exit_cost
        cash += exit_notional - exit_cost
        trades.append(
            {
                "entry_at": position["entry_at"],
                "exit_at": data.index[-1].isoformat(),
                "entry_price": float(position["entry"]),
                "exit_price": final_price,
                "stop_loss": float(position["stop"]),
                "target": float(position["target"]),
                "net_pnl": cash - float(position["equity_before_entry"]),
                "holding_bars": len(data) - 1 - int(position["entry_bar"]),
                "exit_reason": "FINAL_CLOSE",
            }
        )
        equity_values[-1] = cash
        returns[-1] = cash / previous_equity - 1.0
    equity = pd.Series(equity_values, index=data.index, name="equity")
    metrics = _metrics(
        equity,
        trades,
        periods_per_year=PERIODS_PER_YEAR[parameters.timeframe],
        total_costs=total_costs,
    )
    return {
        "strategy_id": parameters.strategy_id,
        "strategy_dna_hash": parameters.dna_hash,
        "parameters": asdict(parameters),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "trades": trades,
        "equity": equity,
        "returns": np.asarray(returns[1:], dtype=float),
        "integrity": {
            "closed_candles_only": True,
            "sequence_is_causal": True,
            "fvg_known_before_limit_entry": True,
            "no_forward_fill": True,
            "same_bar_collision_policy": "STOP_FIRST",
            "long_only_spot": True,
            "costs_included": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    }


def pd_array_fvg_report_path(settings: Settings) -> Path:
    return settings.paths.lab_dir / "reports" / "pd_array_fvg_campaign_v1.json"


def plan_pd_array_fvg_campaign(settings: Settings) -> dict[str, Any]:
    candidates = pd_array_fvg_parameter_set()
    payload = {
        "schema_version": "pd_array_fvg_plan_v1",
        "status": "CAMPAIGN_PLAN",
        "campaign": PD_ARRAY_FVG_CAMPAIGN,
        "engine_version": PD_ARRAY_FVG_ENGINE_VERSION,
        "trial_count": len(candidates),
        "market_count": len(PD_ARRAY_FVG_MARKETS),
        "experiment_count": len(candidates) * len(PD_ARRAY_FVG_MARKETS),
        "markets": list(PD_ARRAY_FVG_MARKETS),
        "timeframes": list(PERIODS_PER_YEAR),
        "strategy_dna": [asdict(row) for row in candidates],
        "strategy_dna_hashes": [row.dna_hash for row in candidates],
        "search_space_hash": stable_hash(
            [row.dna_hash for row in candidates], length=64
        ),
        "mechanical_definition": {
            "pd_array": "RECENT_BULLISH_FVG_OR_ORDER_BLOCK_IN_DISCOUNT",
            "signal": "CONFIRMED_LIQUIDITY_SWEEP_OR_CAUSAL_SMT",
            "displacement": "BULLISH_BODY_ABOVE_1_25_ATR",
            "fvg": "THREE_CANDLE_LOW_T_ABOVE_HIGH_T_MINUS_2",
            "entry": "FUTURE_LIMIT_RETRACE_AT_50_OR_79_PERCENT_OF_FVG",
            "stop": "BELOW_RECENT_SWEEP_LOW_MINUS_0_10_ATR",
            "target": "MOST_RECENT_KNOWN_SWING_HIGH",
        },
        "selection_policy": "REPORT_ALL_NO_AUTOMATIC_PROMOTION",
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    path = settings.paths.lab_dir / "plans" / "pd_array_fvg_plan_v1.json"
    atomic_write_json(path, payload)
    return {**payload, "plan_path": str(path), "plan_sha256": sha256_file(path)}


def _benchmark_market(market: str) -> str:
    return "ETH-EUR" if market == "BTC-EUR" else "BTC-EUR"


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def run_pd_array_fvg_campaign(settings: Settings) -> dict[str, Any]:
    plan = plan_pd_array_fvg_campaign(settings)
    candidates = pd_array_fvg_parameter_set()
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
    data_hashes: dict[str, str] = {}
    for timeframe in PERIODS_PER_YEAR:
        needed = set(PD_ARRAY_FVG_MARKETS) | {"BTC-EUR", "ETH-EUR"}
        for market in needed:
            path = settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"
            if not path.is_file():
                continue
            frame_cache[(market, timeframe)] = pd.read_parquet(path)
            data_hashes[f"{market}:{timeframe}"] = sha256_file(path)
    normalized_cache = {
        key: _normalized_ohlcv(frame) for key, frame in frame_cache.items()
    }
    structure_cache = {
        key: market_structure_features(frame)
        for key, frame in normalized_cache.items()
    }
    volatility_cache = {
        key: atr(frame, 14) for key, frame in normalized_cache.items()
    }
    results: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}
    stressed_results: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for market in PD_ARRAY_FVG_MARKETS:
            key = (market, candidate.timeframe)
            benchmark_key = (_benchmark_market(market), candidate.timeframe)
            if key not in frame_cache or benchmark_key not in frame_cache:
                continue
            prepared_formation = _formation_table(
                normalized_cache[key],
                normalized_cache[benchmark_key],
                candidate,
                structure=structure_cache[key],
                volatility=volatility_cache[key],
            )
            normal = backtest_pd_array_fvg(
                frame_cache[key],
                frame_cache[benchmark_key],
                candidate,
                fee_rate=settings.costs.default_fee,
                slippage_bps=settings.costs.slippage_bps,
                spread_bps=settings.costs.spread_bps,
                prepared_data=normalized_cache[key],
                prepared_benchmark=normalized_cache[benchmark_key],
                prepared_formation=prepared_formation,
            )
            stressed = backtest_pd_array_fvg(
                frame_cache[key],
                frame_cache[benchmark_key],
                candidate,
                fee_rate=(
                    settings.costs.default_fee
                    * settings.costs.stressed_cost_multiplier
                ),
                slippage_bps=(
                    settings.costs.slippage_bps
                    * settings.costs.stressed_cost_multiplier
                ),
                spread_bps=(
                    settings.costs.spread_bps
                    * settings.costs.stressed_cost_multiplier
                ),
                prepared_data=normalized_cache[key],
                prepared_benchmark=normalized_cache[benchmark_key],
                prepared_formation=prepared_formation,
            )
            identity = f"{candidate.dna_hash}:{market}"
            raw_results[identity] = normal
            stressed_results[identity] = stressed
            results.append(
                {
                    "market": market,
                    "benchmark_market": benchmark_key[0],
                    "strategy_id": candidate.strategy_id,
                    "strategy_dna_hash": candidate.dna_hash,
                    "timeframe": candidate.timeframe,
                    "signal_mode": candidate.signal_mode,
                    "fvg_entry_depth": candidate.fvg_entry_depth,
                    "normal": normal["metrics"],
                    "stressed": stressed["metrics"],
                    "diagnostics": normal["diagnostics"],
                    "integrity": normal["integrity"],
                    "backtest_positive": bool(
                        normal["metrics"]["net_total_return"] > 0.0
                        and normal["metrics"]["closed_position_profit_factor"] > 1.0
                        and normal["metrics"]["net_expectancy_eur"] > 0.0
                    ),
                    "stressed_positive": bool(
                        stressed["metrics"]["net_total_return"] > 0.0
                        and stressed["metrics"]["closed_position_profit_factor"] > 1.0
                    ),
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            )
    results.sort(
        key=lambda row: (
            bool(row["backtest_positive"]),
            float(row["normal"]["sharpe"]),
            float(row["normal"]["net_total_return"]),
        ),
        reverse=True,
    )
    stochastic_policy = replace(
        policy_from_research_settings(
            settings.research,
            seed=settings.app.random_seed,
            expected_block_length=12,
        ),
        simulations=min(2_000, settings.research.monte_carlo_runs),
    )
    robustness: dict[str, Any] = {}
    for row in results[:5]:
        identity = f"{row['strategy_dna_hash']}:{row['market']}"
        normal = raw_results[identity]
        stressed = stressed_results[identity]
        robustness[identity] = validate_strategy_return_paths(
            normal["returns"],
            stressed["returns"],
            policy=stochastic_policy,
            seed_offset=len(robustness) * 10_000,
        )
    report_path = pd_array_fvg_report_path(settings)
    csv_path = report_path.with_suffix(".csv")
    payload = {
        "schema_version": "pd_array_fvg_report_v1",
        "status": "COMPLETED_RESEARCH_ONLY",
        "campaign": PD_ARRAY_FVG_CAMPAIGN,
        "plan": plan,
        "data_hashes": data_hashes,
        "experiment_count": len(results),
        "backtest_positive_count": sum(
            bool(row["backtest_positive"]) for row in results
        ),
        "stressed_positive_count": sum(
            bool(row["stressed_positive"]) for row in results
        ),
        "results": results,
        "top_five_robustness": robustness,
        "interpretation": (
            "Retrospective real-data test of a mechanical interpretation. "
            "Reddit profitability claims are not treated as evidence."
        ),
        "paper_candidates": 0,
        "live_authority": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(report_path, _json_ready(payload))
    pd.DataFrame(
        [
            {
                "rank": rank,
                "market": row["market"],
                "strategy_id": row["strategy_id"],
                "timeframe": row["timeframe"],
                "signal_mode": row["signal_mode"],
                "entry_depth": row["fvg_entry_depth"],
                "net_total_return": row["normal"]["net_total_return"],
                "cagr": row["normal"]["cagr"],
                "profit_factor": row["normal"]["closed_position_profit_factor"],
                "sharpe": row["normal"]["sharpe"],
                "maximum_drawdown": row["normal"]["maximum_drawdown"],
                "trade_count": row["normal"]["trade_count"],
                "stressed_return": row["stressed"]["net_total_return"],
                "stressed_profit_factor": row["stressed"]["closed_position_profit_factor"],
                "backtest_positive": row["backtest_positive"],
                "stressed_positive": row["stressed_positive"],
            }
            for rank, row in enumerate(results, start=1)
        ]
    ).to_csv(csv_path, index=False)
    return {
        "status": payload["status"],
        "campaign": PD_ARRAY_FVG_CAMPAIGN,
        "experiment_count": len(results),
        "backtest_positive_count": payload["backtest_positive_count"],
        "stressed_positive_count": payload["stressed_positive_count"],
        "top_results": results[:10],
        "report": str(report_path),
        "csv": str(csv_path),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "PD_ARRAY_FVG_CAMPAIGN",
    "PD_ARRAY_FVG_MARKETS",
    "PdArrayFvgParameters",
    "backtest_pd_array_fvg",
    "pd_array_fvg_parameter_set",
    "pd_array_fvg_report_path",
    "plan_pd_array_fvg_campaign",
    "run_pd_array_fvg_campaign",
]
