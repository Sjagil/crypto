"""Causal 15m relative-pair research for EUR spot markets.

Bitvavo does not expose BTC-quoted markets such as TAO-BTC or ETH-BTC.  This
module therefore treats those names as *research ratios* built from two EUR
spot legs.  Evaluation rotates long-only between the base asset, the BTC leg
and cash.  It never represents the result as a native market or a short.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS, Settings
from utils.common import atomic_write_json, atomic_write_text, stable_hash, utc_iso

PAIR_ENGINE_VERSION = "1.1.0"
PAIR_TIMEFRAME = "15m"
CONTEXT_TIMEFRAMES = ("1h", "4h")


@dataclass(frozen=True, slots=True)
class RelativePairSpec:
    synthetic_symbol: str
    base_market: str
    benchmark_market: str = "BTC-EUR"

    @property
    def identity(self) -> str:
        return stable_hash(
            {
                "engine_version": PAIR_ENGINE_VERSION,
                **asdict(self),
                "execution_timeframe": PAIR_TIMEFRAME,
                "context_timeframes": CONTEXT_TIMEFRAMES,
                "native_market": False,
                "long_only_spot_rotation": True,
            },
            length=64,
        )


@dataclass(frozen=True, slots=True)
class RelativeStrategySpec:
    strategy_id: str
    mechanism: str
    stop_atr: float
    target_atr: float
    maximum_holding_bars: int

    @property
    def dna_hash(self) -> str:
        return stable_hash(
            {
                "engine_version": PAIR_ENGINE_VERSION,
                **asdict(self),
                "execution_timeframe": PAIR_TIMEFRAME,
                "context_timeframes": CONTEXT_TIMEFRAMES,
                "next_open_execution": True,
                "maximum_exposure": 0.50,
                "long_only_spot_rotation": True,
            },
            length=64,
        )


PAIR_SPECS = (
    RelativePairSpec("TAO/BTC", "TAO-EUR"),
    RelativePairSpec("ETH/BTC", "ETH-EUR"),
)

STRATEGY_SPECS = (
    RelativeStrategySpec("PAIR15_RELATIVE_BREAKOUT", "relative_breakout", 2.0, 4.0, 192),
    RelativeStrategySpec("PAIR15_PULLBACK_RECLAIM", "pullback_reclaim", 1.6, 3.2, 128),
    RelativeStrategySpec("PAIR15_MOMENTUM_ACCELERATION", "momentum_acceleration", 1.8, 3.6, 160),
    RelativeStrategySpec("PAIR15_LIQUIDITY_SWEEP", "liquidity_sweep", 1.4, 2.8, 96),
    RelativeStrategySpec("PAIR15_ZSCORE_REVERSION", "zscore_reversion", 1.5, 3.0, 96),
    RelativeStrategySpec("PAIR15_MTF_ENSEMBLE", "mtf_ensemble", 1.8, 3.8, 160),
)


def _load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} has no timestamp column or DatetimeIndex")
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path.name} missing OHLCV columns: {missing}")
    result = frame.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    result = result.dropna().sort_index()
    result = result[~result.index.duplicated(keep="last")]
    if result.empty or (result.loc[:, required] <= 0).any().any():
        raise ValueError(f"{path.name} contains empty or non-positive OHLCV")
    return result


def build_synthetic_cross(
    base: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Build an inner-joined cross without forward filling either leg.

    Cross high/low are conservative OHLC bounds.  They are not represented as
    provider-native intrabar observations.
    """

    common = base.index.intersection(benchmark.index)
    if common.empty:
        raise ValueError(f"no synchronized closed candles for {symbol}")
    left = base.reindex(common)
    right = benchmark.reindex(common)
    cross = pd.DataFrame(index=common)
    cross["open"] = left["open"] / right["open"]
    cross["high"] = left["high"] / right["low"]
    cross["low"] = left["low"] / right["high"]
    cross["close"] = left["close"] / right["close"]
    cross["base_quote_volume_eur"] = left["volume"] * left["close"]
    cross["benchmark_quote_volume_eur"] = right["volume"] * right["close"]
    # FeaturePipeline requires an OHLCV schema.  This column remains explicitly
    # base-leg EUR quote volume and is never eligible for pair-volume signals.
    cross["volume"] = cross["base_quote_volume_eur"]
    cross.attrs.update(
        synthetic_symbol=symbol,
        native_market=False,
        no_forward_fill=True,
        high_low_semantics="CONSERVATIVE_CROSS_BOUNDS",
        volume_semantics="BASE_LEG_QUOTE_VOLUME_EUR_NOT_CROSS_VOLUME",
    )
    return cross.replace([np.inf, -np.inf], np.nan).dropna()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        (
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _context_state(cross: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    close = cross["close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    state = pd.DataFrame(index=cross.index)
    state[f"{timeframe}_bull"] = (close > ema50) & (ema20 > ema50) & (ema20.diff(3) > 0)
    state[f"{timeframe}_bear"] = (close < ema50) & (ema20 < ema50) & (ema20.diff(3) < 0)
    state[f"{timeframe}_source_timestamp"] = cross.index
    availability = cross.index + timedelta(seconds=int(TIMEFRAME_SECONDS[timeframe]))
    state.index = availability
    return state


def align_closed_context(
    execution_index: pd.DatetimeIndex,
    context_crosses: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Expose context only after the source candle's close timestamp."""

    decision_index = execution_index + timedelta(
        seconds=int(TIMEFRAME_SECONDS[PAIR_TIMEFRAME])
    )
    output = pd.DataFrame(index=execution_index)
    for timeframe, cross in sorted(context_crosses.items()):
        available = _context_state(cross, timeframe)
        aligned = available.reindex(decision_index, method="ffill")
        aligned.index = execution_index
        for column in aligned:
            output[column] = aligned[column]
        source = pd.to_datetime(output[f"{timeframe}_source_timestamp"], utc=True)
        source_close = source + timedelta(seconds=int(TIMEFRAME_SECONDS[timeframe]))
        if (source_close.dropna() > pd.Series(decision_index, index=execution_index).loc[source_close.dropna().index]).any():
            raise AssertionError("higher-timeframe lookahead detected")
    return output


def pair_features(
    execution_cross: pd.DataFrame,
    context_crosses: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output = execution_cross.copy()
    close = output["close"]
    output["ema20"] = _ema(close, 20)
    output["ema50"] = _ema(close, 50)
    output["atr14"] = _atr(output)
    output["rsi14"] = _rsi(close)
    output["roc12"] = close.pct_change(12)
    output["acceleration"] = output["roc12"].diff(3)
    output["prior_high20"] = output["high"].shift(1).rolling(20, min_periods=20).max()
    output["prior_low20"] = output["low"].shift(1).rolling(20, min_periods=20).min()
    mean = close.rolling(96, min_periods=48).mean()
    std = close.rolling(96, min_periods=48).std(ddof=0).replace(0.0, np.nan)
    output["zscore96"] = (close - mean) / std
    quote_volume = output["base_quote_volume_eur"]
    output["relative_volume20"] = quote_volume / quote_volume.rolling(20, min_periods=20).mean()
    return output.join(align_closed_context(output.index, context_crosses))


def _crossed_above(series: pd.Series, level: pd.Series | float) -> pd.Series:
    selected = level if isinstance(level, pd.Series) else pd.Series(level, index=series.index)
    return (series > selected) & (series.shift(1) <= selected.shift(1))


def _crossed_below(series: pd.Series, level: pd.Series | float) -> pd.Series:
    selected = level if isinstance(level, pd.Series) else pd.Series(level, index=series.index)
    return (series < selected) & (series.shift(1) >= selected.shift(1))


def target_states(features: pd.DataFrame, mechanism: str) -> pd.Series:
    """Return BASE/BENCHMARK/CASH targets; never a short position."""

    close = features["close"]
    bull = features["1h_bull"].eq(True) & features["4h_bull"].eq(True)
    bear = features["1h_bear"].eq(True) & features["4h_bear"].eq(True)
    volume = features["relative_volume20"].fillna(0.0)
    breakout_up = _crossed_above(close, features["prior_high20"]) & (volume >= 1.0)
    breakout_down = _crossed_below(close, features["prior_low20"]) & (volume >= 1.0)
    pullback_up = _crossed_above(close, features["ema20"]) & features["rsi14"].between(40, 68)
    pullback_down = _crossed_below(close, features["ema20"]) & features["rsi14"].between(32, 60)
    accel_up = (features["roc12"] > 0) & _crossed_above(features["acceleration"], 0.0)
    accel_down = (features["roc12"] < 0) & _crossed_below(features["acceleration"], 0.0)
    sweep_up = (features["low"] < features["prior_low20"]) & (close > features["prior_low20"])
    sweep_down = (features["high"] > features["prior_high20"]) & (close < features["prior_high20"])
    z_up = _crossed_above(features["zscore96"], -1.2) & (features["zscore96"].shift(1) < -1.5)
    z_down = _crossed_below(features["zscore96"], 1.2) & (features["zscore96"].shift(1) > 1.5)
    if mechanism == "relative_breakout":
        enter_base, enter_benchmark = bull & breakout_up, bear & breakout_down
    elif mechanism == "pullback_reclaim":
        enter_base, enter_benchmark = bull & pullback_up, bear & pullback_down
    elif mechanism == "momentum_acceleration":
        enter_base, enter_benchmark = bull & accel_up & (volume >= 0.9), bear & accel_down
    elif mechanism == "liquidity_sweep":
        enter_base, enter_benchmark = ~bear & sweep_up, ~bull & sweep_down
    elif mechanism == "zscore_reversion":
        enter_base, enter_benchmark = ~bear & z_up, ~bull & z_down
    elif mechanism == "mtf_ensemble":
        up_score = breakout_up.astype(int) + pullback_up.astype(int) + accel_up.astype(int) + sweep_up.astype(int)
        down_score = breakout_down.astype(int) + pullback_down.astype(int) + accel_down.astype(int) + sweep_down.astype(int)
        enter_base, enter_benchmark = bull & (up_score >= 2), bear & (down_score >= 2)
    else:
        raise ValueError(f"unsupported relative-pair mechanism: {mechanism}")
    # The 15m layer times entries; it must not churn the higher-timeframe idea.
    # Require three closed bars beyond the slower mean and impose a two-hour
    # post-exit cooldown.  These constants are invariant across both pairs and
    # all mechanisms, rather than tuned against their individual results.
    weak_base = (close < features["ema50"]).rolling(3, min_periods=3).sum() >= 3
    weak_benchmark = (close > features["ema50"]).rolling(3, min_periods=3).sum() >= 3
    exit_base = weak_base | bear
    exit_benchmark = weak_benchmark | bull
    state = "CASH"
    held_bars = 0
    cooldown_bars = 0
    rows: list[str] = []
    for timestamp in features.index:
        cooldown_bars = max(0, cooldown_bars - 1)
        if state == "CASH":
            held_bars = 0
            if cooldown_bars == 0 and bool(enter_base.loc[timestamp]):
                state = "BASE"
            elif cooldown_bars == 0 and bool(enter_benchmark.loc[timestamp]):
                state = "BENCHMARK"
        else:
            held_bars += 1
            should_exit = (
                state == "BASE" and bool(exit_base.loc[timestamp])
            ) or (
                state == "BENCHMARK" and bool(exit_benchmark.loc[timestamp])
            )
            if held_bars >= 4 and should_exit:
                state = "CASH"
                held_bars = 0
                cooldown_bars = 8
        rows.append(state)
    return pd.Series(rows, index=features.index, dtype="string")


def _drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _simulate(
    base: pd.DataFrame,
    benchmark: pd.DataFrame,
    targets: pd.Series,
    spec: RelativeStrategySpec,
    *,
    fee_fraction: float,
    spread_bps: float,
    slippage_bps: float,
    cost_multiplier: float,
    initial_cash: float = 10_000.0,
    periods_per_year: float = 365.25 * 96,
) -> dict[str, Any]:
    common = targets.index.intersection(base.index).intersection(benchmark.index)
    base, benchmark, targets = base.reindex(common), benchmark.reindex(common), targets.reindex(common)
    impact = (slippage_bps + spread_bps / 2.0) * cost_multiplier / 10_000.0
    fee = fee_fraction * cost_multiplier
    cash = initial_cash
    asset = "CASH"
    quantity = 0.0
    entry_outlay = 0.0
    stop = target = 0.0
    held = 0
    pending = "CASH"
    blocked_asset: str | None = None
    episodes: list[float] = []
    turnover = costs = 0.0
    rotations = 0
    rows: list[dict[str, Any]] = []
    atr_by_asset = {
        "BASE": _atr(base).reindex(common),
        "BENCHMARK": _atr(benchmark).reindex(common),
    }

    def frame_for(name: str) -> pd.DataFrame:
        return base if name == "BASE" else benchmark

    def close_position(timestamp: pd.Timestamp, raw_price: float) -> None:
        nonlocal cash, asset, quantity, entry_outlay, turnover, costs, held
        fill = raw_price * (1.0 - impact)
        gross = quantity * fill
        charge = gross * fee
        proceeds = gross - charge
        costs += charge + quantity * max(0.0, raw_price - fill)
        turnover += gross
        cash += proceeds
        episodes.append(proceeds - entry_outlay)
        asset, quantity, entry_outlay, held = "CASH", 0.0, 0.0, 0

    for position, timestamp in enumerate(common):
        desired = str(pending)
        if blocked_asset and desired != blocked_asset:
            blocked_asset = None
        if desired != asset and not (blocked_asset and desired == blocked_asset):
            if asset != "CASH":
                close_position(timestamp, float(frame_for(asset).loc[timestamp, "open"]))
            if desired in {"BASE", "BENCHMARK"}:
                selected = frame_for(desired)
                raw = float(selected.loc[timestamp, "open"])
                fill = raw * (1.0 + impact)
                outlay = min(cash, cash * 0.50)
                charge_fraction = 1.0 + fee
                quantity = outlay / (fill * charge_fraction)
                charge = quantity * fill * fee
                cash -= outlay
                costs += charge + quantity * max(0.0, fill - raw)
                turnover += quantity * fill
                asset, entry_outlay = desired, outlay
                atr = atr_by_asset[desired].iloc[position]
                distance = max(float(atr) if np.isfinite(atr) else raw * 0.01, raw * 0.0025)
                stop = fill - spec.stop_atr * distance
                target = fill + spec.target_atr * distance
                held = 0
                rotations += 1
        if asset != "CASH":
            selected_bar = frame_for(asset).loc[timestamp]
            held += 1
            risk_exit = None
            if float(selected_bar["low"]) <= stop:
                risk_exit = stop
            elif float(selected_bar["high"]) >= target:
                risk_exit = target
            elif held >= spec.maximum_holding_bars:
                risk_exit = float(selected_bar["close"])
            if risk_exit is not None:
                exited_asset = asset
                close_position(timestamp, risk_exit)
                blocked_asset = exited_asset
        equity = cash
        if asset != "CASH":
            equity += quantity * float(frame_for(asset).loc[timestamp, "close"])
        rows.append({"timestamp": timestamp, "equity": equity, "asset": asset})
        pending = str(targets.iloc[position])
    if asset != "CASH" and common.size:
        close_position(common[-1], float(frame_for(asset).iloc[-1]["close"]))
        rows[-1]["equity"] = cash
        rows[-1]["asset"] = "CASH"
    curve = pd.DataFrame(rows).set_index("timestamp")
    returns = curve["equity"].pct_change().fillna(0.0)
    years = max((common[-1] - common[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    total_return = cash / initial_cash - 1.0
    annualized_vol = float(returns.std(ddof=0) * math.sqrt(periods_per_year))
    cagr = float((cash / initial_cash) ** (1.0 / years) - 1.0) if cash > 0 else -1.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * math.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else 0.0
    )
    positive = sum(value for value in episodes if value > 0)
    negative = abs(sum(value for value in episodes if value < 0))
    pf = float(positive / negative) if negative > 0 else (float("inf") if positive > 0 else 0.0)
    return {
        "net_total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "annualized_volatility": annualized_vol,
        "maximum_drawdown": _drawdown(curve["equity"]),
        "closed_holding_episodes": len(episodes),
        "closed_position_profit_factor": pf,
        "net_expectancy_eur": float(np.mean(episodes)) if episodes else 0.0,
        "win_rate": float(np.mean(np.asarray(episodes) > 0)) if episodes else 0.0,
        "rotations": rotations,
        "turnover_eur": turnover,
        "modeled_costs_eur": costs,
        "average_exposure": float((curve["asset"] != "CASH").mean() * 0.50),
        "ending_equity_eur": cash,
        "episode_pnl": episodes,
        "equity_curve": curve,
    }


def _robustness(episodes: Iterable[float], *, simulations: int, seed: int) -> dict[str, Any]:
    values = np.asarray(tuple(episodes), dtype=float)
    if values.size < 2:
        return {"status": "INSUFFICIENT_SAMPLE", "sample": int(values.size)}
    rng = np.random.default_rng(seed)
    bootstrap_means = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(simulations)]
    )
    weights = rng.dirichlet(np.ones(values.size), size=simulations)
    dirichlet_means = weights @ values
    return {
        "status": "COMPUTED",
        "sample": int(values.size),
        "monte_carlo_bootstrap": {
            "simulations": simulations,
            "mean_p05_eur": float(np.quantile(bootstrap_means, 0.05)),
            "probability_positive": float(np.mean(bootstrap_means > 0)),
        },
        "dirichlet_bayesian_bootstrap": {
            "simulations": simulations,
            "mean_p05_eur": float(np.quantile(dirichlet_means, 0.05)),
            "probability_positive": float(np.mean(dirichlet_means > 0)),
        },
    }


def catalogue() -> dict[str, Any]:
    return {
        "schema_version": "relative_pair_15m_catalogue_v1",
        "engine_version": PAIR_ENGINE_VERSION,
        "pairs": [
            {
                **asdict(pair),
                "pair_identity": pair.identity,
                "native_bitvavo_market": False,
                "execution_mapping": [pair.base_market, pair.benchmark_market],
            }
            for pair in PAIR_SPECS
        ],
        "strategies": [
            {
                **asdict(spec),
                "strategy_dna": spec.dna_hash,
                "status": "RESEARCH_ONLY_PENDING_EXACT_VALIDATION",
                "live_authority": False,
            }
            for spec in STRATEGY_SPECS
        ],
        "integrity": {
            "closed_candles_only": True,
            "higher_timeframe_available_at_close": True,
            "next_open_execution": True,
            "no_forward_fill": True,
            "long_only_spot": True,
            "shorting": False,
        },
    }


def run_relative_pair_campaign(
    settings: Settings,
    *,
    pairs: Iterable[str] | None = None,
    maximum_rows: int = 0,
    simulations: int = 1_000,
) -> dict[str, Any]:
    if simulations < 100:
        raise ValueError("simulations must be at least 100")
    requested = {value.upper().replace("-", "/") for value in (pairs or ())}
    selected_pairs = tuple(pair for pair in PAIR_SPECS if not requested or pair.synthetic_symbol in requested)
    if not selected_pairs:
        raise ValueError(f"no registered relative pairs selected: {sorted(requested)}")
    normalized = settings.paths.processed_data_dir
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in selected_pairs:
        frames: dict[str, dict[str, pd.DataFrame]] = {}
        for timeframe in (PAIR_TIMEFRAME, *CONTEXT_TIMEFRAMES):
            paths = {
                "base": normalized / f"{pair.base_market}_{timeframe}.parquet",
                "benchmark": normalized / f"{pair.benchmark_market}_{timeframe}.parquet",
            }
            if any(not path.is_file() for path in paths.values()):
                missing.extend(str(path) for path in paths.values() if not path.is_file())
                frames = {}
                break
            frames[timeframe] = {name: _load_ohlcv(path) for name, path in paths.items()}
        if not frames:
            continue
        execution_base = frames[PAIR_TIMEFRAME]["base"]
        execution_benchmark = frames[PAIR_TIMEFRAME]["benchmark"]
        execution_cross = build_synthetic_cross(execution_base, execution_benchmark, symbol=pair.synthetic_symbol)
        if maximum_rows > 0:
            execution_cross = execution_cross.iloc[-maximum_rows:]
            execution_base = execution_base.reindex(execution_cross.index)
            execution_benchmark = execution_benchmark.reindex(execution_cross.index)
        context = {
            timeframe: build_synthetic_cross(
                frames[timeframe]["base"],
                frames[timeframe]["benchmark"],
                symbol=pair.synthetic_symbol,
            )
            for timeframe in CONTEXT_TIMEFRAMES
        }
        features = pair_features(execution_cross, context)
        for strategy in STRATEGY_SPECS:
            targets = target_states(features, strategy.mechanism)
            common_arguments = {
                "fee_fraction": settings.costs.default_fee,
                "spread_bps": settings.costs.spread_bps,
                "slippage_bps": settings.costs.slippage_bps,
            }
            normal = _simulate(
                execution_base,
                execution_benchmark,
                targets,
                strategy,
                cost_multiplier=1.0,
                **common_arguments,
            )
            stressed = _simulate(
                execution_base,
                execution_benchmark,
                targets,
                strategy,
                cost_multiplier=settings.costs.stressed_cost_multiplier,
                **common_arguments,
            )
            robustness = _robustness(
                normal.pop("episode_pnl"),
                simulations=simulations,
                seed=settings.app.random_seed,
            )
            normal.pop("equity_curve")
            stressed.pop("episode_pnl")
            stressed.pop("equity_curve")
            positive = (
                normal["net_total_return"] > 0
                and normal["closed_position_profit_factor"] > 1.0
                and normal["net_expectancy_eur"] > 0
            )
            results.append(
                {
                    "synthetic_pair": pair.synthetic_symbol,
                    "pair_identity": pair.identity,
                    "base_market": pair.base_market,
                    "benchmark_market": pair.benchmark_market,
                    "strategy_id": strategy.strategy_id,
                    "strategy_dna": strategy.dna_hash,
                    "mechanism": strategy.mechanism,
                    "timeframe": PAIR_TIMEFRAME,
                    "context_timeframes": list(CONTEXT_TIMEFRAMES),
                    "rows": len(features),
                    "start": features.index.min().isoformat(),
                    "end": features.index.max().isoformat(),
                    "normal": normal,
                    "stressed": stressed,
                    "robustness": robustness,
                    "backtest_positive": positive,
                    "recommended_phase": "PAPER_CANDIDATE" if positive else "RESEARCH_ONLY",
                    "live_authority": False,
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
    output_dir = settings.paths.output_dir / "research" / "relative_pairs_15m"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "relative_pair_15m_campaign_v1",
        "generated_at": utc_iso(),
        "status": "COMPLETE" if results else "DATA_BLOCKED",
        "pairs_requested": [pair.synthetic_symbol for pair in selected_pairs],
        "result_count": len(results),
        "positive_count": sum(bool(row["backtest_positive"]) for row in results),
        "missing_data": sorted(set(missing)),
        "results": results,
        "execution_contract": {
            "native_btc_quote_markets_available_on_bitvavo": False,
            "research_ratio_only": True,
            "paper_requires_two_leg_eur_rotation": True,
            "live_requires_separate_operator_approval_per_dna": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    }
    json_path = output_dir / "relative_pair_15m_campaign_v1.json"
    csv_path = output_dir / "relative_pair_15m_campaign_v1.csv"
    md_path = output_dir / "relative_pair_15m_campaign_v1.md"
    catalogue_path = output_dir / "relative_pair_15m_catalogue_v1.json"
    atomic_write_json(json_path, payload)
    atomic_write_json(catalogue_path, catalogue())
    columns = (
        "synthetic_pair",
        "strategy_id",
        "strategy_dna",
        "rows",
        "backtest_positive",
        "recommended_phase",
        "normal_net_total_return",
        "normal_cagr",
        "normal_sharpe",
        "normal_maximum_drawdown",
        "normal_profit_factor",
        "normal_episodes",
        "stressed_net_total_return",
        "stressed_profit_factor",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "synthetic_pair": row["synthetic_pair"],
                    "strategy_id": row["strategy_id"],
                    "strategy_dna": row["strategy_dna"],
                    "rows": row["rows"],
                    "backtest_positive": row["backtest_positive"],
                    "recommended_phase": row["recommended_phase"],
                    "normal_net_total_return": row["normal"]["net_total_return"],
                    "normal_cagr": row["normal"]["cagr"],
                    "normal_sharpe": row["normal"]["sharpe"],
                    "normal_maximum_drawdown": row["normal"]["maximum_drawdown"],
                    "normal_profit_factor": row["normal"]["closed_position_profit_factor"],
                    "normal_episodes": row["normal"]["closed_holding_episodes"],
                    "stressed_net_total_return": row["stressed"]["net_total_return"],
                    "stressed_profit_factor": row["stressed"]["closed_position_profit_factor"],
                }
            )
    lines = [
        "# Causal 15m relative-pair campaign",
        "",
        "TAO/BTC and ETH/BTC are synthetic research ratios. Execution remains EUR spot-only.",
        "",
        "| Pair | Strategy | Return | PF | Sharpe | Max DD | Episodes | Stress PF | Phase |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        normal, stressed = row["normal"], row["stressed"]
        lines.append(
            f"| {row['synthetic_pair']} | {row['strategy_id']} | "
            f"{normal['net_total_return']:.2%} | {normal['closed_position_profit_factor']:.3f} | "
            f"{normal['sharpe']:.2f} | {normal['maximum_drawdown']:.2%} | "
            f"{normal['closed_holding_episodes']} | {stressed['closed_position_profit_factor']:.3f} | "
            f"{row['recommended_phase']} |"
        )
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    payload["artifacts"] = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
        "catalogue": str(catalogue_path.resolve()),
    }
    atomic_write_json(json_path, payload)
    return payload


def current_pair_status(settings: Settings) -> dict[str, Any]:
    report = settings.paths.output_dir / "research" / "relative_pairs_15m" / "relative_pair_15m_campaign_v1.json"
    if report.is_file():
        import json

        payload = json.loads(report.read_text(encoding="utf-8"))
        return {
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "result_count": payload.get("result_count"),
            "positive_count": payload.get("positive_count"),
            "top_results": (payload.get("results") or [])[:5],
            "artifacts": payload.get("artifacts"),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    return {
        "status": "NOT_RUN",
        "catalogue": catalogue(),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def scan_relative_pairs(
    settings: Settings,
    *,
    pairs: Iterable[str] | None = None,
    maximum_rows: int = 5_000,
) -> dict[str, Any]:
    """Return current closed-candle pair states without creating orders."""

    requested = {value.upper().replace("-", "/") for value in (pairs or ())}
    selected_pairs = tuple(
        pair for pair in PAIR_SPECS if not requested or pair.synthetic_symbol in requested
    )
    if not selected_pairs:
        raise ValueError(f"no registered relative pairs selected: {sorted(requested)}")
    normalized = settings.paths.processed_data_dir
    rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        loaded: dict[str, dict[str, pd.DataFrame]] = {}
        for timeframe in (PAIR_TIMEFRAME, *CONTEXT_TIMEFRAMES):
            loaded[timeframe] = {
                "base": _load_ohlcv(
                    normalized / f"{pair.base_market}_{timeframe}.parquet"
                ),
                "benchmark": _load_ohlcv(
                    normalized / f"{pair.benchmark_market}_{timeframe}.parquet"
                ),
            }
        execution = build_synthetic_cross(
            loaded[PAIR_TIMEFRAME]["base"],
            loaded[PAIR_TIMEFRAME]["benchmark"],
            symbol=pair.synthetic_symbol,
        ).iloc[-maximum_rows:]
        context = {
            timeframe: build_synthetic_cross(
                loaded[timeframe]["base"],
                loaded[timeframe]["benchmark"],
                symbol=pair.synthetic_symbol,
            )
            for timeframe in CONTEXT_TIMEFRAMES
        }
        features = pair_features(execution, context)
        for strategy in STRATEGY_SPECS:
            targets = target_states(features, strategy.mechanism)
            current = str(targets.iloc[-1])
            previous = str(targets.iloc[-2]) if len(targets) > 1 else "CASH"
            rows.append(
                {
                    "synthetic_pair": pair.synthetic_symbol,
                    "strategy_id": strategy.strategy_id,
                    "strategy_dna": strategy.dna_hash,
                    "closed_candle_timestamp": features.index[-1].isoformat(),
                    "ratio_close": float(features["close"].iloc[-1]),
                    "target": current,
                    "previous_target": previous,
                    "signal_state": (
                        "FRESH_ROTATION" if current != previous else
                        "ACTIVE_RELATIVE_STATE" if current != "CASH" else "NO_SIGNAL"
                    ),
                    "one_hour_bull": bool(features["1h_bull"].eq(True).iloc[-1]),
                    "one_hour_bear": bool(features["1h_bear"].eq(True).iloc[-1]),
                    "four_hour_bull": bool(features["4h_bull"].eq(True).iloc[-1]),
                    "four_hour_bear": bool(features["4h_bear"].eq(True).iloc[-1]),
                    "live_authority": False,
                    "orders_generated": 0,
                }
            )
    payload = {
        "schema_version": "relative_pair_15m_scan_v1",
        "generated_at": utc_iso(),
        "status": "READY",
        "rows": rows,
        "fresh_rotations": sum(row["signal_state"] == "FRESH_ROTATION" for row in rows),
        "active_states": sum(row["target"] != "CASH" for row in rows),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    target = (
        settings.paths.output_dir
        / "research"
        / "relative_pairs_15m"
        / "current_pair_signals.json"
    )
    atomic_write_json(target, payload)
    payload["artifact"] = str(target.resolve())
    return payload


__all__ = [
    "CONTEXT_TIMEFRAMES",
    "PAIR_SPECS",
    "PAIR_TIMEFRAME",
    "STRATEGY_SPECS",
    "align_closed_context",
    "build_synthetic_cross",
    "catalogue",
    "current_pair_status",
    "pair_features",
    "run_relative_pair_campaign",
    "scan_relative_pairs",
    "target_states",
]
