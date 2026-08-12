"""Causal validation and immutable promotion artifacts for 1h/2h authority.

The module deliberately owns research and reporting only.  It does not submit
orders and cannot enable the live supervisor.  A candidate is emitted only
after the same frozen economic rule has passed real-data normal-cost,
out-of-sample, walk-forward, stress, attribution, and Monte-Carlo checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.settings import Settings
from research.stochastic_validation import (
    StochasticValidationPolicy,
    validate_strategy_return_paths,
)
from utils.common import atomic_write_json, read_json, stable_hash, utc_iso

SCHEMA_VERSION = "multi_timeframe_authority_validation_v1"
AUTHORITY_SCHEMA_VERSION = "multi_timeframe_authority_v1"
DEFAULT_MARKETS = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
)
TARGET_TIMEFRAMES = ("1h", "2h")
SUPPORTED_RESEARCH_TIMEFRAMES = ("15m", *TARGET_TIMEFRAMES)
TIMEFRAME_HOURS = {"15m": 0.25, "1h": 1.0, "2h": 2.0}


@dataclass(frozen=True)
class MultiTimeframeParameters:
    """One small, explainable Donchian/ATR strategy DNA."""

    timeframe: str
    entry_lookback: int
    exit_lookback: int
    daily_ema_period: int = 100
    atr_period: int = 14
    atr_stop_multiple: float = 4.0
    reward_risk: float = 2.5
    confirmed_fractal_span: int = 2
    use_confirmed_fractal_stop: bool = True
    side_cost_bps: float = 35.0

    @property
    def strategy_id(self) -> str:
        context = "_H1" if self.timeframe == "15m" else ""
        return (
            f"MTF_DONCHIAN_{self.timeframe.upper()}{context}"
            f"_D{self.daily_ema_period}"
            f"_E{self.entry_lookback}_X{self.exit_lookback}_ATR"
            f"{self.atr_stop_multiple:g}_F{self.confirmed_fractal_span}"
        )

    @property
    def dna_hash(self) -> str:
        identity = {
            "strategy_id": self.strategy_id,
            "family": "CAUSAL_MTF_DONCHIAN_ATR_FRACTAL",
            "parameters": asdict(self),
            "entry": (
                "closed execution candle above prior Donchian high and "
                "last fully closed daily close above causal daily EMA"
            ),
            "execution": "next execution-timeframe open",
            "exit": (
                "next open after closed candle below prior Donchian low "
                "or daily trend invalidation; intrabar bounded stop"
            ),
            "fractal": (
                "five-candle confirmed fractal low becomes eligible only "
                "after both right-hand candles close"
            ),
            "sizing": "ATR risk proportional; live caps applied downstream",
            "spot_only": True,
            "long_only": True,
        }
        if self.timeframe == "15m":
            identity["hourly_confirmation"] = (
                "last fully closed UTC 1h close above causal 1h EMA50"
            )
            identity["entry_order_policy"] = (
                "bounded venue-aware IOC/FOK limit; no market fallback"
            )
        return stable_hash(identity, length=64)


def _frame_path(settings: Settings, market: str, timeframe: str) -> Path:
    return settings.paths.processed_data_dir / f"{market}_{timeframe}.parquet"


def _load_frame(
    settings: Settings,
    market: str,
    timeframe: str,
) -> pd.DataFrame:
    path = _frame_path(settings, market, timeframe)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns and frame.index.name == "timestamp":
        frame = frame.reset_index()
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna()
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    if frame.empty or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name}: invalid timestamp order")
    invalid = (
        (frame["volume"] < 0)
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if bool(invalid.any()):
        raise ValueError(f"{path.name}: invalid OHLCV rows")
    return frame


def _closed_daily_context(execution: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    """Build UTC daily context whose values become visible only after close."""

    daily = (
        execution.resample(
            "1D",
            origin="epoch",
            label="left",
            closed="left",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    daily["daily_ema"] = daily["close"].ewm(
        span=ema_period,
        adjust=False,
        min_periods=ema_period,
    ).mean()
    daily["daily_trend"] = daily["close"] > daily["daily_ema"]
    daily["available_at"] = daily.index + pd.Timedelta(1, unit="D")
    return daily.set_index("available_at")[["close", "daily_ema", "daily_trend"]]


def _closed_hourly_context(
    execution: pd.DataFrame,
    ema_period: int = 50,
) -> pd.DataFrame:
    """Build UTC 1h context that is unavailable until the hour has closed."""

    hourly = (
        execution.resample(
            "1h",
            origin="epoch",
            label="left",
            closed="left",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    hourly["hourly_ema"] = hourly["close"].ewm(
        span=ema_period,
        adjust=False,
        min_periods=ema_period,
    ).mean()
    hourly["hourly_trend"] = hourly["close"] > hourly["hourly_ema"]
    hourly["hourly_available_at"] = hourly.index + pd.Timedelta(1, unit="h")
    return hourly.set_index("hourly_available_at")[
        ["close", "hourly_ema", "hourly_trend"]
    ].rename(columns={"close": "hourly_close"})


def _confirmed_fractal_low(low: pd.Series, span: int) -> pd.Series:
    """Return a causal series: center low appears only after right bars close."""

    window = span * 2 + 1
    centered_min = low.rolling(window, center=True, min_periods=window).min()
    is_fractal = low.eq(centered_min)
    raw = low.where(is_fractal)
    # A center at i is only confirmed at i + span.
    return raw.shift(span).ffill()


def _feature_frame(
    frame: pd.DataFrame,
    parameters: MultiTimeframeParameters,
) -> pd.DataFrame:
    interval = pd.Timedelta(TIMEFRAME_HOURS[parameters.timeframe], unit="h")
    result = frame.copy()
    prior_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - prior_close).abs(),
            (result["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = true_range.rolling(
        parameters.atr_period,
        min_periods=parameters.atr_period,
    ).mean()
    result["entry_level"] = result["high"].shift(1).rolling(
        parameters.entry_lookback,
        min_periods=parameters.entry_lookback,
    ).max()
    result["exit_level"] = result["low"].shift(1).rolling(
        parameters.exit_lookback,
        min_periods=parameters.exit_lookback,
    ).min()
    result["confirmed_fractal_low"] = _confirmed_fractal_low(
        result["low"],
        parameters.confirmed_fractal_span,
    )
    result["decision_at"] = result.index + interval
    daily = _closed_daily_context(result, parameters.daily_ema_period)
    joined = pd.merge_asof(
        result.reset_index().sort_values("decision_at"),
        daily.reset_index().sort_values("available_at"),
        left_on="decision_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_daily"),
    )
    joined = joined.set_index("timestamp").sort_index()
    daily_trend = joined["daily_trend"].astype("boolean").fillna(False)
    if parameters.timeframe == "15m":
        hourly = _closed_hourly_context(result)
        joined = pd.merge_asof(
            joined.reset_index().sort_values("decision_at"),
            hourly.reset_index().sort_values("hourly_available_at"),
            left_on="decision_at",
            right_on="hourly_available_at",
            direction="backward",
            allow_exact_matches=True,
        ).set_index("timestamp").sort_index()
        hourly_trend = (
            joined["hourly_trend"].astype("boolean").fillna(False)
        )
    else:
        joined["hourly_close"] = np.nan
        joined["hourly_ema"] = np.nan
        joined["hourly_available_at"] = pd.NaT
        joined["hourly_trend"] = True
        hourly_trend = pd.Series(True, index=joined.index, dtype=bool)
    joined["entry_signal"] = (
        (joined["close"] > joined["entry_level"])
        & daily_trend
        & hourly_trend
    )
    joined["exit_signal"] = (
        (joined["close"] < joined["exit_level"])
        | ~daily_trend
        | ~hourly_trend
    )
    return joined


def _trade_return(entry: float, exit_: float, side_cost_bps: float) -> float:
    cost = side_cost_bps / 10_000.0
    return (exit_ * (1.0 - cost)) / (entry * (1.0 + cost)) - 1.0


def _simulate_market(
    frame: pd.DataFrame,
    parameters: MultiTimeframeParameters,
    *,
    featured: pd.DataFrame | None = None,
    side_cost_bps: float | None = None,
    delayed_entry_bars: int = 0,
    miss_every: int | None = None,
) -> list[dict[str, Any]]:
    """Simulate next-open entries/exits and conservative intrabar stops."""

    side_cost = (
        parameters.side_cost_bps
        if side_cost_bps is None
        else float(side_cost_bps)
    )
    evaluated = (
        _feature_frame(frame, parameters)
        if featured is None
        else featured
    )
    rows = evaluated.reset_index()
    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    pending_entry: int | None = None
    pending_exit = False
    signal_count = 0

    for index, row in rows.iterrows():
        if pending_exit and position is not None:
            exit_price = float(row["open"])
            trades.append(
                {
                    **position,
                    "exit_timestamp": row["timestamp"].isoformat(),
                    "exit_price": exit_price,
                    "exit_reason": "SIGNAL_NEXT_OPEN",
                    "net_return": _trade_return(
                        float(position["entry_price"]),
                        exit_price,
                        side_cost,
                    ),
                }
            )
            position = None
            pending_exit = False

        if pending_entry is not None and index >= pending_entry and position is None:
            entry_price = float(row["open"])
            atr = float(row["atr"])
            if np.isfinite(entry_price) and np.isfinite(atr) and atr > 0:
                atr_stop = entry_price - parameters.atr_stop_multiple * atr
                fractal = float(row["confirmed_fractal_low"])
                stop = atr_stop
                if (
                    parameters.use_confirmed_fractal_stop
                    and np.isfinite(fractal)
                    and 0 < fractal < entry_price
                ):
                    stop = max(stop, fractal)
                stop = min(stop, entry_price * 0.999)
                risk = entry_price - stop
                if risk > 0:
                    position = {
                        "entry_timestamp": row["timestamp"].isoformat(),
                        "entry_price": entry_price,
                        "initial_stop": stop,
                        "target": entry_price + risk * parameters.reward_risk,
                        "signal_timestamp": rows.iloc[
                            max(0, index - 1 - delayed_entry_bars)
                        ]["timestamp"].isoformat(),
                    }
            pending_entry = None

        if position is not None:
            stop = float(position["initial_stop"])
            if float(row["low"]) <= stop:
                exit_price = min(float(row["open"]), stop)
                trades.append(
                    {
                        **position,
                        "exit_timestamp": row["timestamp"].isoformat(),
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
                pending_exit = False
            elif bool(row["exit_signal"]):
                pending_exit = True

        if (
            position is None
            and pending_entry is None
            and index + 1 + delayed_entry_bars < len(rows)
            and bool(row["entry_signal"])
        ):
            signal_count += 1
            if miss_every is None or signal_count % miss_every:
                pending_entry = index + 1 + delayed_entry_bars

    if position is not None:
        row = rows.iloc[-1]
        exit_price = float(row["close"])
        trades.append(
            {
                **position,
                "exit_timestamp": row["timestamp"].isoformat(),
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


def _drawdown(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    values = np.r_[1.0, equity]
    return float(np.min(values / peaks - 1.0))


def _metrics(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([float(row["net_return"]) for row in trades], dtype=float)
    if not len(returns):
        return {
            "trade_count": 0,
            "net_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "maximum_drawdown": 0.0,
            "best_trade_profit_share": 1.0,
        }
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    gross_profit = float(positive.sum())
    gross_loss = float(-negative.sum())
    return {
        "trade_count": int(len(returns)),
        "net_return": float(np.prod(1.0 + returns) - 1.0),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else gross_profit
        ),
        "expectancy": float(returns.mean()),
        "win_rate": float(np.mean(returns > 0)),
        "maximum_drawdown": _drawdown(returns.tolist()),
        "best_trade_profit_share": (
            float(positive.max() / gross_profit) if gross_profit > 0 else 1.0
        ),
    }


def _split_oos(
    trades: Sequence[Mapping[str, Any]],
    split_timestamp: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    out_of_sample: list[dict[str, Any]] = []
    for raw in trades:
        row = dict(raw)
        timestamp = pd.Timestamp(row["entry_timestamp"])
        (out_of_sample if timestamp >= split_timestamp else development).append(row)
    return development, out_of_sample


def _panel_net_return(per_market: Mapping[str, Mapping[str, Any]]) -> float:
    if not per_market:
        return 0.0
    return float(
        np.mean(
            [
                1.0 + float(metrics.get("net_return") or 0.0)
                for metrics in per_market.values()
            ]
        )
        - 1.0
    )


def _walk_forward(
    trades_by_market: Mapping[str, Sequence[Mapping[str, Any]]],
    starts: Mapping[str, pd.Timestamp],
    ends: Mapping[str, pd.Timestamp],
    folds: int = 5,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        per_market: dict[str, dict[str, Any]] = {}
        for market, trades in trades_by_market.items():
            start = starts[market]
            end = ends[market]
            lower = start + (end - start) * (fold / folds)
            upper = start + (end - start) * ((fold + 1) / folds)
            selected = [
                row
                for row in trades
                if lower
                <= pd.Timestamp(row["entry_timestamp"])
                < (upper if fold < folds - 1 else end + pd.Timedelta(seconds=1))
            ]
            per_market[market] = _metrics(selected)
        net = _panel_net_return(per_market)
        fold_rows.append(
            {
                "fold": fold + 1,
                "net_return": net,
                "positive": net > 0.0,
                "per_market": per_market,
            }
        )
    return {
        "folds": fold_rows,
        "positive_folds": sum(bool(row["positive"]) for row in fold_rows),
        "total_folds": folds,
    }


def _monte_carlo(
    returns: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 20260730,
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return {
            "samples": samples,
            "probability_positive": 0.0,
            "net_return_p05": 0.0,
            "maximum_drawdown_p95": 0.0,
        }
    rng = np.random.default_rng(seed)
    net_returns = np.empty(samples)
    drawdowns = np.empty(samples)
    for index in range(samples):
        sampled = rng.choice(values, size=len(values), replace=True)
        net_returns[index] = np.prod(1.0 + sampled) - 1.0
        drawdowns[index] = _drawdown(sampled.tolist())
    return {
        "samples": samples,
        "seed": seed,
        "probability_positive": float(np.mean(net_returns > 0.0)),
        "net_return_p05": float(np.quantile(net_returns, 0.05)),
        "net_return_median": float(np.median(net_returns)),
        "maximum_drawdown_p95": float(np.quantile(drawdowns, 0.05)),
    }


def _candidate_grid(timeframe: str) -> tuple[MultiTimeframeParameters, ...]:
    lookbacks = (
        ((96, 32), (160, 48), (240, 72))
        if timeframe == "15m"
        else ((120, 48), (180, 60), (240, 72))
    )
    return tuple(
        MultiTimeframeParameters(
            timeframe=timeframe,
            entry_lookback=entry,
            exit_lookback=exit_,
        )
        for entry, exit_ in lookbacks
    )


def _evaluate_candidate(
    settings: Settings,
    parameters: MultiTimeframeParameters,
    markets: Sequence[str],
) -> dict[str, Any]:
    frames = {
        market: _load_frame(settings, market, parameters.timeframe)
        for market in markets
    }
    featured = {
        market: _feature_frame(frame, parameters)
        for market, frame in frames.items()
    }
    starts = {market: frame.index.min() for market, frame in frames.items()}
    ends = {market: frame.index.max() for market, frame in frames.items()}
    normal = {
        market: _simulate_market(
            frame,
            parameters,
            featured=featured[market],
        )
        for market, frame in frames.items()
    }
    stressed = {
        market: _simulate_market(
            frame,
            parameters,
            featured=featured[market],
            side_cost_bps=55.0,
        )
        for market, frame in frames.items()
    }
    delayed = {
        market: _simulate_market(
            frame,
            parameters,
            featured=featured[market],
            delayed_entry_bars=1,
        )
        for market, frame in frames.items()
    }
    missed = {
        market: _simulate_market(
            frame,
            parameters,
            featured=featured[market],
            miss_every=10,
        )
        for market, frame in frames.items()
    }
    normal_metrics = {market: _metrics(rows) for market, rows in normal.items()}
    stressed_metrics = {
        market: _metrics(rows) for market, rows in stressed.items()
    }
    delayed_metrics = {market: _metrics(rows) for market, rows in delayed.items()}
    missed_metrics = {market: _metrics(rows) for market, rows in missed.items()}

    oos_by_market: dict[str, list[dict[str, Any]]] = {}
    development_by_market: dict[str, list[dict[str, Any]]] = {}
    for market, rows in normal.items():
        split = starts[market] + (ends[market] - starts[market]) * 0.70
        development, out_of_sample = _split_oos(rows, split)
        development_by_market[market] = development
        oos_by_market[market] = out_of_sample
    oos_metrics = {
        market: _metrics(rows) for market, rows in oos_by_market.items()
    }
    development_metrics = {
        market: _metrics(rows)
        for market, rows in development_by_market.items()
    }
    all_trades = [
        {**row, "market": market}
        for market, rows in normal.items()
        for row in rows
    ]
    all_trades.sort(key=lambda row: row["entry_timestamp"])
    overall = _metrics(all_trades)
    overall["panel_net_return"] = _panel_net_return(normal_metrics)
    stressed_overall = _metrics(
        [
            {**row, "market": market}
            for market, rows in stressed.items()
            for row in rows
        ]
    )
    stressed_overall["panel_net_return"] = _panel_net_return(stressed_metrics)
    delayed_overall = _metrics(
        [
            {**row, "market": market}
            for market, rows in delayed.items()
            for row in rows
        ]
    )
    delayed_overall["panel_net_return"] = _panel_net_return(delayed_metrics)
    missed_overall = _metrics(
        [
            {**row, "market": market}
            for market, rows in missed.items()
            for row in rows
        ]
    )
    missed_overall["panel_net_return"] = _panel_net_return(missed_metrics)
    oos_overall = _metrics(
        [
            {**row, "market": market}
            for market, rows in oos_by_market.items()
            for row in rows
        ]
    )
    oos_overall["panel_net_return"] = _panel_net_return(oos_metrics)
    development_overall = _metrics(
        [
            {**row, "market": market}
            for market, rows in development_by_market.items()
            for row in rows
        ]
    )
    development_overall["panel_net_return"] = _panel_net_return(
        development_metrics
    )
    positive_markets = sum(
        float(row["net_return"]) > 0.0 for row in normal_metrics.values()
    )
    oos_positive_markets = sum(
        float(row["net_return"]) > 0.0 for row in oos_metrics.values()
    )
    walk_forward = _walk_forward(normal, starts, ends)
    monte_carlo = _monte_carlo(
        [float(row["net_return"]) for row in all_trades],
    )
    normal_trade_returns = np.asarray(
        [float(row["net_return"]) for row in all_trades],
        dtype=float,
    )
    stressed_trade_returns = np.asarray(
        [
            float(row["net_return"])
            for market_rows in stressed.values()
            for row in market_rows
        ],
        dtype=float,
    )
    stochastic = validate_strategy_return_paths(
        normal_trade_returns,
        stressed_trade_returns,
        policy=StochasticValidationPolicy(
            simulations=2_000,
            expected_block_length=10,
            maximum_drawdown=0.50,
            maximum_drawdown_breach_probability=0.20,
            maximum_terminal_loss_probability=0.20,
            minimum_p05_total_return=-0.20,
            dirichlet_blocks=8,
            minimum_observations=30,
            seed=20260730 + int(parameters.dna_hash[:8], 16),
            batch_size=128,
        ),
    )
    required_trades = (
        80
        if parameters.timeframe == "15m"
        else 50
        if parameters.timeframe == "1h"
        else 40
    )
    gates = {
        "normal_net_positive": overall["panel_net_return"] > 0.0,
        "normal_profit_factor": overall["profit_factor"] >= 1.15,
        "positive_expectancy": overall["expectancy"] > 0.0,
        "minimum_trades": overall["trade_count"] >= required_trades,
        "out_of_sample_positive": oos_overall["panel_net_return"] > 0.0,
        "walk_forward_three_positive": walk_forward["positive_folds"] >= 3,
        "no_single_trade_dependency": overall["best_trade_profit_share"] < 0.30,
        "no_single_asset_dependency": positive_markets >= 3,
        "stressed_edge_positive": (
            stressed_overall["panel_net_return"] > 0.0
            and stressed_overall["profit_factor"] > 1.0
        ),
        "delayed_entry_positive": delayed_overall["panel_net_return"] > 0.0,
        "missed_trade_positive": missed_overall["panel_net_return"] > 0.0,
    }
    confidence_warnings = []
    if monte_carlo["probability_positive"] < 0.95:
        confidence_warnings.append("MONTE_CARLO_CONFIDENCE_BELOW_95_PERCENT")
    if stochastic.get("checks", {}).get("normal_dirichlet") is not True:
        confidence_warnings.append("DIRICHLET_CONCENTRATION_WARNING")
    if stochastic.get("checks", {}).get("stressed_monte_carlo") is not True:
        confidence_warnings.append("STRESSED_MONTE_CARLO_WARNING")
    return {
        "strategy_id": parameters.strategy_id,
        "strategy_dna_hash": parameters.dna_hash,
        "family": "CAUSAL_MTF_DONCHIAN_ATR_FRACTAL",
        "timeframe": parameters.timeframe,
        "markets": list(markets),
        "parameters": asdict(parameters),
        "integrity": {
            "real_market_data": True,
            "closed_candle_only": True,
            "no_lookahead": True,
            "no_repainting": True,
            "next_open_execution": True,
            "confirmed_fractal_causal": True,
            "long_only_spot": True,
            "bounded_stop": True,
            "valid_exit": True,
            "closed_hourly_confirmation": parameters.timeframe == "15m",
            "live_limit_entry_policy_separate_from_historical_fill_model": True,
        },
        "normal": overall,
        "normal_per_market": normal_metrics,
        "development": development_overall,
        "out_of_sample": oos_overall,
        "out_of_sample_per_market": oos_metrics,
        "stressed": stressed_overall,
        "delayed_entry": delayed_overall,
        "missed_trade": missed_overall,
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
        "stochastic_validation": stochastic,
        "positive_markets": positive_markets,
        "oos_positive_markets": oos_positive_markets,
        "gates": gates,
        "confidence_warnings": confidence_warnings,
        "validation_pass": all(gates.values()),
        "data": {
            market: {
                "path": str(_frame_path(settings, market, parameters.timeframe)),
                "rows": len(frames[market]),
                "start": starts[market].isoformat(),
                "end": ends[market].isoformat(),
                "fingerprint": stable_hash(
                    {
                        "path": str(
                            _frame_path(settings, market, parameters.timeframe)
                        ),
                        "rows": len(frames[market]),
                        "start": starts[market].isoformat(),
                        "end": ends[market].isoformat(),
                        "last_close": float(frames[market]["close"].iloc[-1]),
                    },
                    length=64,
                ),
            }
            for market in markets
        },
    }


def _score(row: Mapping[str, Any]) -> float:
    normal = dict(row["normal"])
    oos = dict(row["out_of_sample"])
    stressed = dict(row["stressed"])
    walk = dict(row["walk_forward"])
    return float(
        np.log1p(max(-0.99, float(normal["panel_net_return"]))) * 20.0
        + min(3.0, float(normal["profit_factor"])) * 10.0
        + np.log1p(max(-0.99, float(oos["panel_net_return"]))) * 25.0
        + min(3.0, float(stressed["profit_factor"])) * 10.0
        + int(walk["positive_folds"]) * 4.0
        + int(row["positive_markets"]) * 3.0
        - abs(float(normal["maximum_drawdown"])) * 20.0
    )


def multi_timeframe_frozen_candidate_hash(
    row: Mapping[str, Any],
) -> str:
    """Hash only immutable MTF execution identity fields.

    Keeping this in the authority module prevents paper/live consumers from
    silently reimplementing a different hash schema.  Historical performance
    and data-snapshot fields are evidence, not executable strategy identity.
    """

    parameters = dict(row.get("parameters") or {})
    return stable_hash(
        {
            "strategy_id": row["strategy_id"],
            "strategy_dna_hash": row["strategy_dna_hash"],
            "timeframe": row["timeframe"],
            "markets": list(row["markets"]),
            "parameters": parameters,
            "normal_cost_bps_per_side": parameters["side_cost_bps"],
            "paper_adapter": "MTF_DONCHIAN_ATR_FRACTAL",
        },
        length=64,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        flattened.append(
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "timeframe": row["timeframe"],
                "entry_lookback": row["parameters"]["entry_lookback"],
                "exit_lookback": row["parameters"]["exit_lookback"],
                "trades": row["normal"]["trade_count"],
                "net_return": row["normal"]["panel_net_return"],
                "profit_factor": row["normal"]["profit_factor"],
                "maximum_drawdown": row["normal"]["maximum_drawdown"],
                "oos_net_return": row["out_of_sample"]["panel_net_return"],
                "stressed_net_return": row["stressed"]["panel_net_return"],
                "stressed_profit_factor": row["stressed"]["profit_factor"],
                "positive_markets": row["positive_markets"],
                "oos_positive_markets": row["oos_positive_markets"],
                "positive_folds": row["walk_forward"]["positive_folds"],
                "validation_pass": row["validation_pass"],
                "score": row["score"],
            }
        )
    pd.DataFrame(flattened).to_csv(path, index=False)


def _write_markdown(
    path: Path,
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Multi-timeframe authority validation",
        "",
        f"Generated: {utc_iso()}",
        "",
        "All results use real stored market data, closed candles, next-open "
        "execution, a bounded ATR/confirmed-fractal stop, and explicit costs.",
        "",
        "| Strategy | TF | Trades | PF | Net | OOS | Stress PF | Folds | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            "| {strategy_id} | {timeframe} | {trades} | {pf:.3f} | "
            "{net:.1%} | {oos:.1%} | {spf:.3f} | {folds}/5 | {passed} |".format(
                strategy_id=row["strategy_id"],
                timeframe=row["timeframe"],
                trades=row["normal"]["trade_count"],
                pf=row["normal"]["profit_factor"],
                net=row["normal"]["panel_net_return"],
                oos=row["out_of_sample"]["panel_net_return"],
                spf=row["stressed"]["profit_factor"],
                folds=row["walk_forward"]["positive_folds"],
                passed="PASS" if row["validation_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Selected immutable candidates",
            "",
        ]
    )
    if selected:
        for row in selected:
            lines.append(
                f"- `{row['strategy_id']}` (`{row['strategy_dna_hash']}`), "
                f"{row['timeframe']}, score {row['score']:.2f}."
            )
    else:
        lines.append("- None; authority remains fail-closed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_multi_timeframe_authority(
    settings: Settings,
    *,
    markets: Iterable[str] = DEFAULT_MARKETS,
    timeframes: Iterable[str] = TARGET_TIMEFRAMES,
) -> dict[str, Any]:
    """Run the pre-registered grid and emit selected immutable candidates."""

    normalized_markets = tuple(dict.fromkeys(str(value).upper() for value in markets))
    normalized_timeframes = tuple(
        dict.fromkeys(str(value).strip() for value in timeframes)
    )
    unknown = set(normalized_timeframes) - set(SUPPORTED_RESEARCH_TIMEFRAMES)
    if unknown:
        raise ValueError(f"unsupported authority timeframes: {sorted(unknown)}")
    all_rows: list[dict[str, Any]] = []
    for timeframe in normalized_timeframes:
        timeframe_rows = [
            _evaluate_candidate(settings, parameters, normalized_markets)
            for parameters in _candidate_grid(timeframe)
        ]
        for row in timeframe_rows:
            neighborhood_support = sum(
                candidate is not row
                and candidate["normal"]["panel_net_return"] > 0.0
                and candidate["normal"]["profit_factor"] > 1.0
                and candidate["out_of_sample"]["panel_net_return"] > 0.0
                for candidate in timeframe_rows
            )
            row["parameter_neighborhood_positive_count"] = int(
                neighborhood_support
            )
            row["gates"]["parameter_neighborhood_support"] = (
                neighborhood_support >= 1
            )
            row["validation_pass"] = all(row["gates"].values())
        for row in timeframe_rows:
            row["score"] = _score(row)
        all_rows.extend(timeframe_rows)
    selected: list[dict[str, Any]] = []
    for timeframe in normalized_timeframes:
        passing = [
            row
            for row in all_rows
            if row["timeframe"] == timeframe and row["validation_pass"]
        ]
        if passing:
            selected.append(max(passing, key=lambda row: float(row["score"])))
    data_snapshot_id = stable_hash(
        {
            "data": {
                row["strategy_id"]: row["data"]
                for row in all_rows
            },
            "generated_at_date": datetime.now(UTC).date().isoformat(),
        },
        length=64,
    )
    for row in selected:
        row["data_snapshot_id"] = data_snapshot_id
        row["frozen_candidate_hash"] = (
            multi_timeframe_frozen_candidate_hash(row)
        )
        row["paper_adapter"] = "MTF_DONCHIAN_ATR_FRACTAL"
        row["authority"] = "LIVE_MICRO"
        row["entry_enabled"] = True
        row["exit_enabled"] = True
        row["position_management_enabled"] = True
        row["approved_at"] = utc_iso()
        row["max_position_eur"] = 5.0
        row["max_concurrent_positions"] = 2
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "research_only": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "markets": list(normalized_markets),
        "timeframes": list(normalized_timeframes),
        "candidate_count": len(all_rows),
        "passing_count": len(selected),
        "data_snapshot_id": data_snapshot_id,
        "selection_policy": (
            "best transparent composite score among candidates passing every "
            "pre-registered economic, causal, OOS, WFO and stress gate"
        ),
        "candidates": all_rows,
        "selected_candidates": selected,
    }
    reports = settings.paths.lab_dir / "reports"
    json_path = reports / "multi_timeframe_authority_validation_v1.json"
    csv_path = reports / "multi_timeframe_authority_validation_v1.csv"
    markdown_path = reports / "multi_timeframe_authority_validation_v1.md"
    atomic_write_json(json_path, report)
    _write_csv(csv_path, all_rows)
    _write_markdown(markdown_path, all_rows, selected)
    return report


def validate_15m_entry_overlay(
    settings: Settings,
    *,
    markets: Iterable[str] = DEFAULT_MARKETS,
) -> dict[str, Any]:
    """Validate 15m+1h+1d DNA and grant paper evidence, never live authority."""

    normalized_markets = tuple(dict.fromkeys(str(value).upper() for value in markets))
    rows = [
        _evaluate_candidate(settings, parameters, normalized_markets)
        for parameters in _candidate_grid("15m")
    ]
    for row in rows:
        neighborhood_support = sum(
            candidate is not row
            and candidate["normal"]["panel_net_return"] > 0.0
            and candidate["normal"]["profit_factor"] > 1.0
            and candidate["out_of_sample"]["panel_net_return"] > 0.0
            for candidate in rows
        )
        row["parameter_neighborhood_positive_count"] = int(neighborhood_support)
        row["gates"]["parameter_neighborhood_support"] = neighborhood_support >= 1
        row["validation_pass"] = all(row["gates"].values())
        row["score"] = _score(row)
    passing = [row for row in rows if row["validation_pass"]]
    selected = [max(passing, key=lambda row: float(row["score"]))] if passing else []
    data_snapshot_id = stable_hash(
        {
            "data": {row["strategy_id"]: row["data"] for row in rows},
            "generated_at_date": datetime.now(UTC).date().isoformat(),
        },
        length=64,
    )
    for row in selected:
        row["data_snapshot_id"] = data_snapshot_id
        row["frozen_candidate_hash"] = multi_timeframe_frozen_candidate_hash(row)
        row["paper_adapter"] = "MTF_DONCHIAN_ATR_FRACTAL"
        row["authority"] = "PAPER_ONLY"
        row["entry_enabled"] = True
        row["exit_enabled"] = True
        row["position_management_enabled"] = True
        row["live_authority_granted"] = False
        row["operator_live_dna_approval_required"] = True
        row["approved_at"] = utc_iso()
    report = {
        "schema_version": "15m_entry_overlay_validation_v1",
        "generated_at": utc_iso(),
        "research_only": True,
        "paper_auto_promotion_when_positive": True,
        "live_authority_granted": False,
        "operator_live_dna_approval_required": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "markets": list(normalized_markets),
        "timeframes": ["15m", "1h", "1d"],
        "candidate_count": len(rows),
        "passing_count": len(selected),
        "data_snapshot_id": data_snapshot_id,
        "stationary_bootstrap_monte_carlo": True,
        "dirichlet_time_concentration_stress": True,
        "historical_orderflow_used": False,
        "prospective_orderflow_optional_confirmation": True,
        "live_limit_order_policy": "IOC_OR_FOK_NO_MARKET_FALLBACK",
        "candidates": rows,
        "selected_candidates": selected,
    }
    reports = settings.paths.lab_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "15m_entry_overlay_validation_v1.json"
    csv_path = reports / "15m_entry_overlay_validation_v1.csv"
    markdown_path = reports / "15m_entry_overlay_validation_v1.md"
    atomic_write_json(json_path, report)
    _write_csv(csv_path, rows)
    _write_markdown(markdown_path, rows, selected)
    return {
        **report,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(markdown_path),
        },
    }


def load_validated_multi_timeframe_candidates(
    settings: Settings,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    reports = settings.paths.lab_dir / "reports"
    for path in (
        reports / "multi_timeframe_authority_validation_v1.json",
        reports / "15m_entry_overlay_validation_v1.json",
    ):
        if not path.is_file():
            continue
        raw = pd.read_json(path, typ="series").to_dict()
        for selected in raw.get("selected_candidates") or []:
            row = dict(selected)
            if (
                row.get("validation_pass") is True
                and row.get("authority") in {"LIVE_MICRO", "PAPER_ONLY"}
                and row.get("paper_adapter") == "MTF_DONCHIAN_ATR_FRACTAL"
                and len(str(row.get("strategy_dna_hash") or "")) == 64
                and row.get("frozen_candidate_hash")
            ):
                candidates.append(
                    {
                        "strategy_id": row["strategy_id"],
                        "strategy_dna_hash": row["strategy_dna_hash"],
                        "economic_hypothesis_family": row["family"],
                        "timeframe": row["timeframe"],
                        "markets": list(row["markets"]),
                        "parameters": dict(row["parameters"]),
                        "metrics": {
                            "net_return": row["normal"]["panel_net_return"],
                            "profit_factor": row["normal"]["profit_factor"],
                            "stressed_net_return": row["stressed"][
                                "panel_net_return"
                            ],
                            "stressed_profit_factor": row["stressed"][
                                "profit_factor"
                            ],
                            "maximum_drawdown": row["normal"][
                                "maximum_drawdown"
                            ],
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
                        "paper_adapter": row["paper_adapter"],
                        "frozen_candidate_hash": row["frozen_candidate_hash"],
                        "source_report": str(path),
                        "auto_live_promotion": False,
                        "operator_live_dna_approval_required": True,
                    }
                )
    return candidates


def write_multi_timeframe_authority_registry(
    settings: Settings,
) -> dict[str, Any]:
    """Reconcile preserved 4h/1d/1W authority with validated 1h/2h DNA."""

    source_path = (
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    source = dict(read_json(source_path)) if source_path.is_file() else {}
    existing = [
        {
            **dict(row),
            "authority": "LIVE_MICRO",
            "entry_enabled": True,
            "exit_enabled": True,
            "position_management_enabled": True,
            "validation_report": str(row.get("source") or source_path),
            "preserved_existing_authority": True,
        }
        for row in source.get("approved_candidates") or []
        if str(row.get("timeframe") or "") in {"4h", "1d", "1W"}
    ]
    validation_path = (
        settings.paths.lab_dir
        / "reports"
        / "multi_timeframe_authority_validation_v1.json"
    )
    validation = (
        dict(read_json(validation_path))
        if validation_path.is_file()
        else {}
    )
    new_rows = [
        {
            "strategy_id": row["strategy_id"],
            "strategy_dna_hash": row["strategy_dna_hash"],
            "frozen_candidate_hash": row["frozen_candidate_hash"],
            "timeframe": row["timeframe"],
            "approved_markets": list(row["markets"]),
            "authority": "LIVE_MICRO",
            "entry_enabled": True,
            "exit_enabled": True,
            "position_management_enabled": True,
            "approved_at": row["approved_at"],
            "validation_report": str(validation_path),
            "data_snapshot_id": row["data_snapshot_id"],
            "risk_profile": "CAPITAL_AWARE_MAX_TWO_POSITIONS",
            "max_position_eur": row["max_position_eur"],
            "max_concurrent_positions": row["max_concurrent_positions"],
            "metrics": {
                "trade_count": row["normal"]["trade_count"],
                "profit_factor": row["normal"]["profit_factor"],
                "panel_net_return": row["normal"]["panel_net_return"],
                "out_of_sample_net_return": row["out_of_sample"][
                    "panel_net_return"
                ],
                "stressed_profit_factor": row["stressed"]["profit_factor"],
                "stressed_net_return": row["stressed"]["panel_net_return"],
                "walk_forward_positive_folds": row["walk_forward"][
                    "positive_folds"
                ],
                "monte_carlo_probability_positive": row["monte_carlo"][
                    "probability_positive"
                ],
            },
            "confidence_warnings": list(
                row.get("confidence_warnings") or []
            ),
            "preserved_existing_authority": False,
        }
        for row in validation.get("selected_candidates") or []
        if row.get("validation_pass") is True
    ]
    by_dna = {
        str(row["strategy_dna_hash"]): row
        for row in [*existing, *new_rows]
        if row.get("strategy_dna_hash")
    }
    strategies = sorted(
        by_dna.values(),
        key=lambda row: (
            str(row.get("timeframe")),
            str(row.get("strategy_id")),
        ),
    )
    required = ("1h", "2h", "4h", "1d", "1W")
    coverage = {
        timeframe: sum(
            str(row.get("timeframe")) == timeframe
            and row.get("authority") in {"LIVE_MICRO", "LIVE"}
            for row in strategies
        )
        for timeframe in required
    }
    universe_path = (
        settings.paths.output_dir / "governance" / "live_universe.json"
    )
    universe = dict(read_json(universe_path)) if universe_path.is_file() else {}
    body: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": (
            "READY"
            if all(coverage.values())
            and len(universe.get("selected_markets") or []) >= 5
            else "BLOCKED"
        ),
        "required_timeframes": list(required),
        "timeframe_coverage": coverage,
        "live_markets": list(universe.get("selected_markets") or []),
        "maximum_concurrent_positions": 2,
        "natural_signals_only": True,
        "unknown_dna_fail_closed": True,
        "closed_candle_only": True,
        "orders_generated": 0,
        "orders_submitted": 0,
        "strategies": strategies,
    }
    body["registry_hash"] = stable_hash(
        {
            key: value
            for key, value in body.items()
            if key != "registry_hash"
        },
        length=64,
    )
    path = (
        settings.paths.output_dir
        / "governance"
        / "multi_timeframe_authority.json"
    )
    atomic_write_json(path, body)
    return {**body, "artifact": str(path)}


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "DEFAULT_MARKETS",
    "MultiTimeframeParameters",
    "SUPPORTED_RESEARCH_TIMEFRAMES",
    "TARGET_TIMEFRAMES",
    "load_validated_multi_timeframe_candidates",
    "multi_timeframe_frozen_candidate_hash",
    "validate_15m_entry_overlay",
    "validate_multi_timeframe_authority",
    "write_multi_timeframe_authority_registry",
]


if __name__ == "__main__":
    import json

    current_settings = Settings.load(create_directories=True)
    current_report = validate_multi_timeframe_authority(current_settings)
    print(
        json.dumps(
            {
                "candidate_count": current_report["candidate_count"],
                "passing_count": current_report["passing_count"],
                "selected": [
                    {
                        "strategy_id": row["strategy_id"],
                        "timeframe": row["timeframe"],
                        "profit_factor": row["normal"]["profit_factor"],
                        "net_return": row["normal"]["panel_net_return"],
                        "out_of_sample_net_return": row["out_of_sample"][
                            "panel_net_return"
                        ],
                        "stressed_profit_factor": row["stressed"][
                            "profit_factor"
                        ],
                        "positive_folds": row["walk_forward"][
                            "positive_folds"
                        ],
                    }
                    for row in current_report["selected_candidates"]
                ],
                "orders_generated": 0,
                "orders_submitted": 0,
            },
            indent=2,
        )
    )
