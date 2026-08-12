"""Reproducible signal-funnel and trade-count audit for real strategy evidence.

The audit explains where calendar bars and candidate entries disappear.  It
does not loosen strategy definitions, fabricate fills, or authorize orders.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.settings import TIMEFRAME_SECONDS, Settings, normalize_timeframe
from data.market_data import load_ohlcv
from research.combinatorial_lab import (
    BlockRole,
    CombinationGenerator,
    CombinatorialStrategy,
    LogicMode,
    fast_screen,
    signal_block_registry,
)
from research.features import FeaturePipeline, volume_features
from research.volume_strategy_campaign import (
    VolumeStrategyDNA,
    _available_paths,
    _money_flow_index,
    _signals,
    backtest_volume_strategy_batch,
    volume_strategy_dna,
)
from utils.common import atomic_write_json, stable_hash, utc_iso

AUDIT_SCHEMA_VERSION = "trade_count_signal_funnel_v1"
AUDIT_TIMEFRAMES = ("15m", "1h", "4h", "1d")
LOW_TRADE_THRESHOLD = 100


@dataclass(frozen=True)
class SignalDefinition:
    conditions: OrderedDict[str, pd.Series]
    exit_signal: pd.Series
    feature_values: Mapping[str, pd.Series]
    warmup_bars: int

    @property
    def entry_signal(self) -> pd.Series:
        if not self.conditions:
            raise ValueError("entry definition requires at least one condition")
        result = pd.Series(True, index=next(iter(self.conditions.values())).index)
        for condition in self.conditions.values():
            result &= condition.fillna(False).astype(bool)
        return result.fillna(False)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    selected.index = pd.to_datetime(selected.index, utc=True)
    return selected[
        ~selected.index.duplicated(keep="last")
    ].sort_index()


def _volume_signal_definition(
    frame: pd.DataFrame,
    row: VolumeStrategyDNA,
    *,
    prepared_volume_features: pd.DataFrame | None = None,
    prepared_mfi: pd.Series | None = None,
) -> SignalDefinition:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    features = (
        prepared_volume_features
        if prepared_volume_features is not None
        else volume_features(frame)
    )
    mfi = prepared_mfi if prepared_mfi is not None else _money_flow_index(frame)
    parameters = row.parameters
    values: dict[str, pd.Series] = {}
    conditions: OrderedDict[str, pd.Series] = OrderedDict()
    if row.archetype == "DONCHIAN_RVOL_BREAKOUT":
        entry_lookback = int(parameters["entry_lookback"])
        exit_lookback = int(parameters["exit_lookback"])
        prior_high = high.shift(1).rolling(
            entry_lookback,
            min_periods=entry_lookback,
        ).max()
        prior_low = low.shift(1).rolling(
            exit_lookback,
            min_periods=exit_lookback,
        ).min()
        conditions["close_above_prior_donchian_high"] = close > prior_high
        conditions["relative_volume_at_least_threshold"] = (
            features["relative_volume_20"]
            >= float(parameters["minimum_rvol"])
        )
        exit_signal = close < prior_low
        values = {
            "prior_donchian_high": prior_high,
            "prior_donchian_low": prior_low,
            "relative_volume_20": features["relative_volume_20"],
        }
        warmup = max(entry_lookback, exit_lookback, 20)
    elif row.archetype == "TREND_PULLBACK_DRYUP_RECOVERY":
        ema_period = int(parameters["ema_period"])
        trend = close.ewm(
            span=ema_period,
            adjust=False,
            min_periods=ema_period,
        ).mean()
        conditions["prior_close_near_or_above_trend"] = (
            close.shift(1) > trend.shift(1) * 0.98
        )
        conditions["prior_low_touched_trend_zone"] = (
            low.shift(1) <= trend.shift(1) * 1.02
        )
        conditions["prior_volume_dry_up"] = (
            features["relative_volume_20"].shift(1)
            <= float(parameters["maximum_pullback_rvol"])
        )
        conditions["close_above_prior_high"] = close > high.shift(1)
        conditions["close_above_trend"] = close > trend
        conditions["recovery_volume_confirmed"] = (
            features["relative_volume_20"]
            >= float(parameters["minimum_recovery_rvol"])
        )
        exit_signal = close < trend
        values = {
            "ema_trend": trend,
            "relative_volume_20": features["relative_volume_20"],
        }
        warmup = max(ema_period, 20)
    elif row.archetype == "VOLUME_CONTRACTION_BREAKOUT":
        channel = int(parameters["channel_lookback"])
        contraction = int(parameters["contraction_lookback"])
        prior_high = high.shift(1).rolling(
            channel,
            min_periods=channel,
        ).max()
        prior_rvol = features["relative_volume_20"].shift(1).rolling(
            contraction,
            min_periods=contraction,
        ).mean()
        exit_trend = close.ewm(
            span=max(10, channel),
            adjust=False,
            min_periods=max(10, channel),
        ).mean()
        conditions["close_above_prior_channel"] = close > prior_high
        conditions["prior_volume_contracted"] = (
            prior_rvol <= float(parameters["maximum_prior_rvol"])
        )
        conditions["breakout_volume_confirmed"] = (
            features["relative_volume_20"]
            >= float(parameters["minimum_breakout_rvol"])
        )
        exit_signal = close < exit_trend
        values = {
            "prior_channel_high": prior_high,
            "prior_relative_volume_mean": prior_rvol,
            "relative_volume_20": features["relative_volume_20"],
            "exit_ema": exit_trend,
        }
        warmup = max(channel, 20 + contraction, 10)
    elif row.archetype == "OBV_CMF_CONTINUATION":
        ema_period = int(parameters["ema_period"])
        obv_lookback = int(parameters["obv_lookback"])
        trend = close.ewm(
            span=ema_period,
            adjust=False,
            min_periods=ema_period,
        ).mean()
        prior_obv_high = features["obv"].shift(1).rolling(
            obv_lookback,
            min_periods=obv_lookback,
        ).max()
        conditions["close_above_trend"] = close > trend
        conditions["obv_above_prior_high"] = (
            features["obv"] > prior_obv_high
        )
        conditions["cmf_at_least_threshold"] = (
            features["chaikin_money_flow_20"]
            >= float(parameters["minimum_cmf"])
        )
        exit_signal = (
            (close < trend)
            | (features["chaikin_money_flow_20"] < 0.0)
        )
        values = {
            "ema_trend": trend,
            "obv": features["obv"],
            "prior_obv_high": prior_obv_high,
            "chaikin_money_flow_20": features[
                "chaikin_money_flow_20"
            ],
        }
        warmup = max(ema_period, obv_lookback, 20)
    elif row.archetype == "VWAP_MFI_RECLAIM":
        period = int(parameters["vwap_period"])
        typical = (high + low + close) / 3.0
        vwap = (
            (typical * volume).rolling(
                period,
                min_periods=period,
            ).sum()
            / volume.rolling(
                period,
                min_periods=period,
            ).sum().replace(0.0, np.nan)
        )
        threshold = float(parameters["mfi_reclaim"])
        conditions["prior_close_at_or_below_vwap"] = (
            close.shift(1) <= vwap.shift(1)
        )
        conditions["close_reclaimed_vwap"] = close > vwap
        conditions["prior_mfi_at_or_below_threshold"] = (
            mfi.shift(1) <= threshold
        )
        conditions["mfi_reclaimed_threshold"] = mfi > threshold
        conditions["relative_volume_confirmed"] = (
            features["relative_volume_20"]
            >= float(parameters["minimum_rvol"])
        )
        exit_signal = (close < vwap) | (mfi >= 80.0)
        values = {
            "rolling_vwap": vwap,
            "money_flow_index_14": mfi,
            "relative_volume_20": features["relative_volume_20"],
        }
        warmup = max(period, 20)
    else:
        raise ValueError(f"unsupported audit archetype: {row.archetype}")
    definition = SignalDefinition(
        conditions=conditions,
        exit_signal=exit_signal.fillna(False).astype(bool),
        feature_values=values,
        warmup_bars=warmup,
    )
    expected_entries, expected_exits = _signals(frame, (row,))
    if not definition.entry_signal.equals(expected_entries[row.strategy_id]):
        raise RuntimeError(f"entry audit drift: {row.strategy_id}")
    if not definition.exit_signal.equals(expected_exits[row.strategy_id]):
        raise RuntimeError(f"exit audit drift: {row.strategy_id}")
    return definition


def _multi_timeframe_definition(
    frame_4h: pd.DataFrame,
    frame_1d: pd.DataFrame,
    row: VolumeStrategyDNA,
) -> tuple[SignalDefinition, dict[str, Any]]:
    base = _volume_signal_definition(frame_4h, row)
    daily_close = frame_1d["close"].astype(float)
    daily_ema = daily_close.ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()
    daily_filter = (daily_close > daily_ema).astype(bool)
    known_at = pd.DatetimeIndex(
        pd.to_datetime(frame_1d.index, utc=True)
    ) + pd.to_timedelta(1, unit="D")
    filter_by_close = pd.Series(
        daily_filter.to_numpy(dtype=bool),
        index=known_at,
    )
    execution_close = pd.DatetimeIndex(
        pd.to_datetime(frame_4h.index, utc=True)
    ) + pd.to_timedelta(4, unit="h")
    aligned = (
        filter_by_close.reindex(
            execution_close,
            method="ffill",
        )
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    aligned.index = frame_4h.index
    conditions = OrderedDict(base.conditions)
    conditions["last_fully_closed_1d_above_ema200"] = aligned
    definition = SignalDefinition(
        conditions=conditions,
        exit_signal=base.exit_signal,
        feature_values={
            **base.feature_values,
            "aligned_closed_1d_trend_filter": aligned.astype(float),
        },
        warmup_bars=max(base.warmup_bars, 200 * 6),
    )
    return definition, {
        "strategy_id": "VOL_BTC_EUR_4h_DONCHIAN_N2_WITH_1d_EMA200",
        "execution_timeframe": "4h",
        "context_timeframe": "1d",
        "alignment": "LAST_FULLY_CLOSED_BACKWARD_ASOF",
        "execution_rows_before_join": len(frame_4h),
        "execution_rows_after_join": len(frame_4h),
        "rows_removed": 0,
        "first_context_candle_close": (
            str(filter_by_close.index.min())
            if len(filter_by_close)
            else None
        ),
        "utc_aware": bool(frame_4h.index.tz is not None),
        "causal": True,
        "incomplete_context_visible": False,
    }


def _simulate_round_trips(
    frame: pd.DataFrame,
    *,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
) -> dict[str, Any]:
    entry = entry_signal.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    exit_ = exit_signal.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    holding = False
    entry_execution_index: int | None = None
    blocked_existing = 0
    approved = 0
    filled_entries = 0
    filled_exits = 0
    trades: list[dict[str, Any]] = []
    for signal_index in range(len(frame)):
        if holding and entry[signal_index]:
            blocked_existing += 1
        if holding and exit_[signal_index]:
            if signal_index + 1 < len(frame):
                exit_execution_index = signal_index + 1
                trades.append(
                    {
                        "entry_index": int(entry_execution_index or 0),
                        "exit_index": exit_execution_index,
                        "exit_reason": "strategy_exit",
                    }
                )
                filled_exits += 1
                holding = False
                entry_execution_index = None
            continue
        if not holding and entry[signal_index] and not exit_[signal_index]:
            approved += 1
            if signal_index + 1 < len(frame):
                holding = True
                entry_execution_index = signal_index + 1
                filled_entries += 1
    terminal_liquidations = 0
    if holding and entry_execution_index is not None:
        trades.append(
            {
                "entry_index": entry_execution_index,
                "exit_index": len(frame) - 1,
                "exit_reason": "terminal_liquidation",
            }
        )
        filled_exits += 1
        terminal_liquidations = 1
        holding = False
    for trade in trades:
        bars = max(0, int(trade["exit_index"]) - int(trade["entry_index"]))
        trade["holding_bars"] = bars
        trade["entry_timestamp"] = str(frame.index[int(trade["entry_index"])])
        trade["exit_timestamp"] = str(frame.index[int(trade["exit_index"])])
    return {
        "blocked_existing_position": blocked_existing,
        "approved_entry_count": approved,
        "submitted_entry_count": approved,
        "filled_entry_count": filled_entries,
        "filled_exit_count": filled_exits,
        "completed_round_trip_count": len(trades),
        "open_at_end_count": 0,
        "terminal_liquidation_count": terminal_liquidations,
        "trades": trades,
    }


def _holding_metrics(
    trades: list[dict[str, Any]],
    *,
    timeframe: str,
) -> dict[str, Any]:
    bars = np.asarray(
        [float(item["holding_bars"]) for item in trades],
        dtype=float,
    )
    hours_per_bar = TIMEFRAME_SECONDS[normalize_timeframe(timeframe)] / 3_600.0
    if not len(bars):
        bars = np.asarray([0.0])
    return {
        "minimum_holding_bars": float(np.min(bars)),
        "average_holding_bars": float(np.mean(bars)),
        "median_holding_bars": float(np.median(bars)),
        "p75_holding_bars": float(np.percentile(bars, 75)),
        "p90_holding_bars": float(np.percentile(bars, 90)),
        "maximum_holding_bars": float(np.max(bars)),
        "average_holding_days": float(np.mean(bars) * hours_per_bar / 24.0),
        "median_holding_days": float(np.median(bars) * hours_per_bar / 24.0),
        "maximum_holding_days": float(np.max(bars) * hours_per_bar / 24.0),
    }


def _performance_metrics(
    frame: pd.DataFrame,
    *,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    settings: Settings,
    timeframe: str,
) -> dict[str, Any]:
    entry = entry_signal.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    exit_ = exit_signal.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    state = False
    targets = np.zeros(len(frame), dtype=float)
    for index in range(len(frame)):
        state = state and not exit_[index]
        state = state or (entry[index] and not exit_[index])
        targets[index] = 0.20 if state else 0.0
    executed = np.zeros(len(frame), dtype=float)
    executed[1:] = targets[:-1]
    opens = frame["open"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    if len(frame) < 2:
        return {
            "net_return": 0.0,
            "profit_factor": 0.0,
            "maximum_drawdown": 0.0,
            "turnover": 0.0,
        }
    underlying = np.empty(len(frame) - 1, dtype=float)
    if len(frame) > 2:
        underlying[:-1] = opens[2:] / opens[1:-1] - 1.0
    underlying[-1] = closes[-1] / opens[-1] - 1.0
    exposure = executed[1:]
    turnover = np.abs(np.diff(executed))
    turnover[-1] += executed[-1]
    one_way = (
        settings.costs.default_fee
        + settings.costs.slippage_bps / 10_000.0
        + settings.costs.spread_bps / 20_000.0
    )
    returns = (1.0 - turnover * one_way) * (
        1.0 + exposure * underlying
    ) - 1.0
    equity = pd.Series(1.0 + returns).cumprod()
    positive = float(returns[returns > 0.0].sum())
    negative = abs(float(returns[returns < 0.0].sum()))
    return {
        "net_return": float(equity.iloc[-1] - 1.0),
        "profit_factor": (
            positive / negative
            if negative > 0.0
            else (math.inf if positive > 0.0 else 0.0)
        ),
        "maximum_drawdown": float(
            (equity / equity.cummax() - 1.0).min()
        ),
        "turnover": float(turnover.sum()),
    }


def _classification(
    funnel: Mapping[str, Any],
    holding: Mapping[str, Any],
    *,
    timeframe: str,
    raw_calendar_days: float,
    combined_signals: int,
    all_individual_conditions_nonzero: bool,
) -> str:
    completed = int(funnel["completed_round_trip_count"])
    raw_entries = int(funnel["raw_entry_signal_count"])
    if timeframe == "15m" and raw_calendar_days < 365.25 * 7:
        return "DATA_COVERAGE_LIMITATION"
    if completed == 0 and combined_signals == 0:
        return (
            "IMPOSSIBLE_CONDITION"
            if all_individual_conditions_nonzero
            else "EXPECTED_SELECTIVE_STRATEGY"
        )
    if (
        float(holding["maximum_holding_days"]) > 90.0
        and int(funnel["blocked_existing_position"]) > completed
    ):
        return "LONG_HOLDING_PERIOD"
    if raw_entries and (
        int(funnel["blocked_existing_position"]) / raw_entries >= 0.50
    ):
        return "SINGLE_POSITION_SUPPRESSION"
    return "EXPECTED_SELECTIVE_STRATEGY"


def _audit_one(
    frame: pd.DataFrame,
    *,
    strategy_id: str,
    market: str,
    timeframe: str,
    definition: SignalDefinition,
    settings: Settings,
    exact_metrics: Mapping[str, Any] | None = None,
    multi_timeframe: bool = False,
) -> dict[str, Any]:
    raw_start = frame.index.min()
    raw_end = frame.index.max()
    raw_days = float(
        (raw_end - raw_start).total_seconds() / 86_400.0
    )
    finite = np.isfinite(
        frame.loc[:, ["open", "high", "low", "close", "volume"]]
        .to_numpy(dtype=float)
    ).all(axis=1)
    quality = pd.Series(finite, index=frame.index)
    warmup = pd.Series(
        np.arange(len(frame)) >= definition.warmup_bars,
        index=frame.index,
    )
    tradable = quality & warmup
    sequential = tradable.copy()
    attrition: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = {}
    for order, (name, condition) in enumerate(
        definition.conditions.items(),
        start=1,
    ):
        selected = condition.fillna(False).astype(bool)
        before = int(sequential.sum())
        condition_counts[name] = int(selected.sum())
        sequential &= selected
        after = int(sequential.sum())
        attrition.append(
            {
                "strategy_id": strategy_id,
                "market": market,
                "timeframe": timeframe,
                "condition_order": order,
                "condition": name,
                "condition_true_count": condition_counts[name],
                "events_before": before,
                "events_after": after,
                "events_removed": before - after,
                "retention_ratio": after / before if before else 0.0,
            }
        )
    entry = definition.entry_signal & tradable
    edge = entry & ~entry.shift(1, fill_value=False)
    lifecycle = _simulate_round_trips(
        frame,
        entry_signal=entry,
        exit_signal=definition.exit_signal,
    )
    holding = _holding_metrics(
        lifecycle["trades"],
        timeframe=timeframe,
    )
    time_in_market = (
        sum(float(item["holding_bars"]) for item in lifecycle["trades"])
        / len(frame)
        if len(frame)
        else 0.0
    )
    funnel = {
        "strategy_id": strategy_id,
        "market": market,
        "timeframe": timeframe,
        "raw_calendar_start": str(raw_start),
        "raw_calendar_end": str(raw_end),
        "raw_calendar_days": raw_days,
        "raw_bar_count": len(frame),
        "closed_bar_count": len(frame),
        "post_quality_bar_count": int(quality.sum()),
        "post_warmup_bar_count": int(warmup.sum()),
        "post_alignment_bar_count": len(frame),
        "tradable_bar_count": int(tradable.sum()),
        "raw_condition_true_count_json": json.dumps(
            condition_counts,
            sort_keys=True,
        ),
        "combined_entry_condition_count": int(entry.sum()),
        "raw_entry_signal_count": int(entry.sum()),
        "edge_trigger_count": int(edge.sum()),
        "deduplicated_signal_count": int(edge.sum()),
        "blocked_existing_position": lifecycle[
            "blocked_existing_position"
        ],
        "blocked_portfolio_position_limit": 0,
        "blocked_cooldown": 0,
        "blocked_regime": 0,
        "blocked_missing_feature": 0,
        "blocked_risk": 0,
        "blocked_liquidity": 0,
        "blocked_minimum_order": 0,
        "blocked_correlation": 0,
        "approved_entry_count": lifecycle["approved_entry_count"],
        "submitted_entry_count": lifecycle["submitted_entry_count"],
        "filled_entry_count": lifecycle["filled_entry_count"],
        "filled_exit_count": lifecycle["filled_exit_count"],
        "completed_round_trip_count": lifecycle[
            "completed_round_trip_count"
        ],
        "open_at_end_count": lifecycle["open_at_end_count"],
        "terminal_liquidation_count": lifecycle[
            "terminal_liquidation_count"
        ],
        "trade_definition": "COMPLETED_ROUND_TRIP",
        "multi_timeframe": multi_timeframe,
        "time_in_market": time_in_market,
    }
    selected_metrics = dict(
        exact_metrics
        or _performance_metrics(
            frame,
            entry_signal=entry,
            exit_signal=definition.exit_signal,
            settings=settings,
            timeframe=timeframe,
        )
    )
    funnel.update(
        {
            "net_return": selected_metrics.get("net_return"),
            "profit_factor": selected_metrics.get("profit_factor"),
            "maximum_drawdown": selected_metrics.get("maximum_drawdown"),
            "turnover": selected_metrics.get("turnover"),
        }
    )
    classification = _classification(
        funnel,
        holding,
        timeframe=timeframe,
        raw_calendar_days=raw_days,
        combined_signals=int(entry.sum()),
        all_individual_conditions_nonzero=all(
            value > 0 for value in condition_counts.values()
        ),
    )
    funnel["low_trade_count"] = (
        int(funnel["completed_round_trip_count"]) < LOW_TRADE_THRESHOLD
    )
    funnel["low_trade_count_classification"] = classification
    holding_row = {
        "strategy_id": strategy_id,
        "market": market,
        "timeframe": timeframe,
        **holding,
        "time_in_market": time_in_market,
    }
    exit_counts = Counter(
        item["exit_reason"] for item in lifecycle["trades"]
    )
    exit_rows = [
        {
            "strategy_id": strategy_id,
            "market": market,
            "timeframe": timeframe,
            "exit_reason": reason,
            "count": int(exit_counts.get(reason, 0)),
        }
        for reason in (
            "strategy_exit",
            "stop",
            "target",
            "trailing_stop",
            "time_exit",
            "regime_exit",
            "risk_exit",
            "terminal_liquidation",
            "data_end",
        )
    ]
    data_rows = []
    for feature_name, series in definition.feature_values.items():
        aligned = series.reindex(frame.index)
        missing = aligned.isna()
        first_valid = aligned.first_valid_index()
        data_rows.append(
            {
                "strategy_id": strategy_id,
                "market": market,
                "timeframe": timeframe,
                "feature": feature_name,
                "first_valid_timestamp": (
                    str(first_valid) if first_valid is not None else None
                ),
                "missing_count": int(missing.sum()),
                "missing_ratio": float(missing.mean()),
                "bars_removed": int(
                    missing.iloc[: definition.warmup_bars].sum()
                ),
                "required_by_strategy": True,
            }
        )
    ablation_rows = []
    cumulative = tradable.copy()
    previous_metrics: dict[str, Any] | None = None
    for order, (name, condition) in enumerate(
        definition.conditions.items(),
        start=1,
    ):
        cumulative &= condition.fillna(False).astype(bool)
        simulation = _simulate_round_trips(
            frame,
            entry_signal=cumulative,
            exit_signal=definition.exit_signal,
        )
        metrics = _performance_metrics(
            frame,
            entry_signal=cumulative,
            exit_signal=definition.exit_signal,
            settings=settings,
            timeframe=timeframe,
        )
        ablation_rows.append(
            {
                "strategy_id": strategy_id,
                "market": market,
                "timeframe": timeframe,
                "step": order,
                "added_condition": name,
                "signals": int(cumulative.sum()),
                "completed_trades": simulation[
                    "completed_round_trip_count"
                ],
                "net_return": metrics["net_return"],
                "profit_factor": metrics["profit_factor"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "average_holding_bars": _holding_metrics(
                    simulation["trades"],
                    timeframe=timeframe,
                )["average_holding_bars"],
                "turnover": metrics["turnover"],
                "net_return_delta": (
                    metrics["net_return"]
                    - float(previous_metrics["net_return"])
                    if previous_metrics is not None
                    else None
                ),
            }
        )
        previous_metrics = metrics
    return {
        "funnel": funnel,
        "filter_attrition": attrition,
        "holding": holding_row,
        "exit_reasons": exit_rows,
        "data_attrition": data_rows,
        "ablation": ablation_rows,
    }


def _load_positive_volume_ids(settings: Settings) -> set[str]:
    path = settings.paths.output_dir / "strategies" / "backtest_positive.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload if isinstance(payload, list) else []
    return {
        str(row.get("strategy_id"))
        for row in rows
        if isinstance(row, dict)
        and str(row.get("strategy_id") or "").startswith("VOL_")
        and str(row.get("timeframe") or "") in {"1h", "4h"}
    }


def _classical_positive_rows(settings: Settings) -> list[dict[str, Any]]:
    path = (
        settings.paths.output_dir
        / "strategies"
        / "classical_backtest_positive.json"
    )
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [
        row
        for row in payload.get("candidates") or []
        if isinstance(row, dict)
        and str(row.get("timeframe") or "") in {"1h", "4h"}
        and row.get("block_ids")
    ]


def _feature_frames_for_classical_candidate(
    settings: Settings,
    *,
    markets: tuple[str, ...],
    timeframe: str,
) -> dict[str, pd.DataFrame]:
    benchmark_path = (
        settings.paths.processed_data_dir
        / f"BTC-EUR_{timeframe}.parquet"
    )
    if not benchmark_path.is_file():
        return {}
    benchmark = load_ohlcv(
        benchmark_path,
        market="BTC-EUR",
        timeframe=timeframe,
        closed_candles_only=True,
    )
    benchmark.attrs.update(market="BTC-EUR", timeframe=timeframe)
    output: dict[str, pd.DataFrame] = {}
    for market in markets:
        path = (
            settings.paths.processed_data_dir
            / f"{market}_{timeframe}.parquet"
        )
        if not path.is_file():
            continue
        raw = load_ohlcv(
            path,
            market=market,
            timeframe=timeframe,
            closed_candles_only=True,
        )
        raw.attrs.update(market=market, timeframe=timeframe)
        higher_timeframes: dict[str, pd.DataFrame] = {}
        for higher_timeframe in ("4h", "1d", "1W"):
            if (
                TIMEFRAME_SECONDS[higher_timeframe]
                <= TIMEFRAME_SECONDS[timeframe]
            ):
                continue
            higher_path = (
                settings.paths.processed_data_dir
                / f"{market}_{higher_timeframe}.parquet"
            )
            if not higher_path.is_file():
                continue
            higher = load_ohlcv(
                higher_path,
                market=market,
                timeframe=higher_timeframe,
                closed_candles_only=True,
            )
            higher.attrs.update(
                market=market,
                timeframe=higher_timeframe,
            )
            higher_timeframes[higher_timeframe] = higher
        output[market] = FeaturePipeline().build(
            raw,
            market=market,
            benchmark=benchmark,
            higher_timeframes=higher_timeframes,
        )
    return output


def _audit_classical_block_candidates(
    settings: Settings,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        "funnels": [],
        "filter_attrition": [],
        "holding_periods": [],
        "exit_reasons": [],
        "data_attrition": [],
        "ablations": [],
        "skipped": [],
    }
    registry = signal_block_registry()
    generator = CombinationGenerator(registry)
    round_trip_cost = (
        2.0 * settings.costs.default_fee
        + 2.0 * settings.costs.slippage_bps / 10_000.0
        + settings.costs.spread_bps / 10_000.0
    )
    for row in _classical_positive_rows(settings):
        strategy_id = str(
            row.get("strategy_dna_hash")
            or row.get("combination_id")
        )
        timeframe = normalize_timeframe(str(row["timeframe"]))
        markets = tuple(str(value) for value in row.get("markets") or ())
        try:
            combination = generator.materialize_membership(
                tuple(str(value) for value in row["block_ids"]),
                logic_mode=LogicMode.LAYERED,
                timeframes=(timeframe,),
            )
            strategy = CombinatorialStrategy(combination, registry)
            frames = _feature_frames_for_classical_candidate(
                settings,
                markets=markets,
                timeframe=timeframe,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            output["skipped"].append(
                {
                    "strategy_id": strategy_id,
                    "timeframe": timeframe,
                    "reason": type(exc).__name__,
                }
            )
            continue
        for market, features in frames.items():
            screen = fast_screen(
                {market: features},
                strategy,
                round_trip_cost=round_trip_cost,
            )
            source = dict(screen["signal_funnel"]["per_market"][market])
            raw_condition_counts = dict(source.pop("condition_true_counts"))
            raw_entries = int(source["raw_entry_signal_count"])
            completed = int(source["completed_round_trip_count"])
            holding_days = (
                float(source["average_holding_bars"])
                * TIMEFRAME_SECONDS[timeframe]
                / 86_400.0
            )
            funnel = {
                "strategy_id": strategy_id,
                "market": market,
                "timeframe": timeframe,
                **source,
                "raw_condition_true_count_json": json.dumps(
                    raw_condition_counts,
                    sort_keys=True,
                ),
                "combined_entry_condition_count": raw_entries,
                "trade_definition": "COMPLETED_ROUND_TRIP",
                "multi_timeframe": any(
                    block_id.startswith("htf_")
                    for block_id in row["block_ids"]
                ),
                "net_return": float(screen["screening_return"]),
                "profit_factor": None,
                "maximum_drawdown": None,
                "turnover": None,
                "low_trade_count": completed < LOW_TRADE_THRESHOLD,
                "low_trade_count_classification": (
                    "LONG_HOLDING_PERIOD"
                    if holding_days > 90.0
                    else "SINGLE_POSITION_SUPPRESSION"
                    if raw_entries
                    and int(source["blocked_existing_position"])
                    / raw_entries
                    >= 0.50
                    else "EXPECTED_SELECTIVE_STRATEGY"
                ),
                "evidence_layer": "CANONICAL_FAST_SCREEN_AUDIT",
            }
            output["funnels"].append(funnel)
            output["holding_periods"].append(
                {
                    "strategy_id": strategy_id,
                    "market": market,
                    "timeframe": timeframe,
                    "minimum_holding_bars": None,
                    "average_holding_bars": source[
                        "average_holding_bars"
                    ],
                    "median_holding_bars": source[
                        "median_holding_bars"
                    ],
                    "p75_holding_bars": screen["signal_funnel"][
                        "p75_holding_bars"
                    ],
                    "p90_holding_bars": screen["signal_funnel"][
                        "p90_holding_bars"
                    ],
                    "maximum_holding_bars": source[
                        "maximum_holding_bars"
                    ],
                    "average_holding_days": holding_days,
                    "median_holding_days": (
                        float(source["median_holding_bars"])
                        * TIMEFRAME_SECONDS[timeframe]
                        / 86_400.0
                    ),
                    "maximum_holding_days": (
                        float(source["maximum_holding_bars"])
                        * TIMEFRAME_SECONDS[timeframe]
                        / 86_400.0
                    ),
                    "time_in_market": source["time_in_market"],
                }
            )
            for reason, count in screen["exit_reason_counts"].items():
                output["exit_reasons"].append(
                    {
                        "strategy_id": strategy_id,
                        "market": market,
                        "timeframe": timeframe,
                        "exit_reason": reason,
                        "count": int(count),
                    }
                )
            sequential = pd.Series(True, index=features.index)
            for order, block_id in enumerate(row["block_ids"], start=1):
                block = registry[str(block_id)]
                signal = block.calculate(features)
                if block.role is BlockRole.AVOIDANCE_FILTER:
                    signal = ~signal
                before = int(sequential.sum())
                sequential &= signal.fillna(False)
                after = int(sequential.sum())
                output["filter_attrition"].append(
                    {
                        "strategy_id": strategy_id,
                        "market": market,
                        "timeframe": timeframe,
                        "condition_order": order,
                        "condition": block_id,
                        "condition_true_count": int(signal.sum()),
                        "events_before": before,
                        "events_after": after,
                        "events_removed": before - after,
                        "retention_ratio": (
                            after / before if before else 0.0
                        ),
                    }
                )
                for feature_name in block.required_features:
                    values = features[feature_name]
                    first_valid = values.first_valid_index()
                    missing = values.isna()
                    output["data_attrition"].append(
                        {
                            "strategy_id": strategy_id,
                            "market": market,
                            "timeframe": timeframe,
                            "feature": feature_name,
                            "first_valid_timestamp": (
                                str(first_valid)
                                if first_valid is not None
                                else None
                            ),
                            "missing_count": int(missing.sum()),
                            "missing_ratio": float(missing.mean()),
                            "bars_removed": int(missing.sum()),
                            "required_by_strategy": True,
                        }
                    )
    return output


def _audit_strategy_rows(settings: Settings) -> tuple[VolumeStrategyDNA, ...]:
    paths = _available_paths(settings)
    all_rows = volume_strategy_dna(tuple(sorted(paths)))
    by_id = {row.strategy_id: row for row in all_rows}
    selected_ids = _load_positive_volume_ids(settings)
    for market, timeframe, archetype in (
        ("BTC-EUR", "1d", "DONCHIAN_RVOL_BREAKOUT"),
        ("BTC-EUR", "4h", "DONCHIAN_RVOL_BREAKOUT"),
        ("BTC-EUR", "1h", "DONCHIAN_RVOL_BREAKOUT"),
        ("BTC-EUR", "1d", "OBV_CMF_CONTINUATION"),
        ("ETH-EUR", "4h", "VOLUME_CONTRACTION_BREAKOUT"),
        ("ETH-EUR", "1d", "TREND_PULLBACK_DRYUP_RECOVERY"),
    ):
        selected_ids.add(
            VolumeStrategyDNA(
                market=market,
                timeframe=timeframe,
                archetype=archetype,
                coordinate=2,
                parameters=_parameter_for(archetype, 2),
            ).strategy_id
        )
    selected_ids.update(
        row.strategy_id
        for row in all_rows
        if row.timeframe == "15m"
        and row.coordinate == 2
    )
    return tuple(
        sorted(
            (
                by_id[strategy_id]
                for strategy_id in selected_ids
                if strategy_id in by_id
            ),
            key=lambda row: (
                TIMEFRAME_SECONDS[normalize_timeframe(row.timeframe)],
                row.market,
                row.archetype,
                row.coordinate,
            ),
        )
    )


def _parameter_for(
    archetype: str,
    coordinate: int,
) -> Mapping[str, float | int]:
    from research.volume_strategy_campaign import _parameter_path

    return _parameter_path(archetype, coordinate)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_trade_count_audit(settings: Settings) -> dict[str, Any]:
    """Run the real-data audit and persist all requested artifacts."""

    output_dir = (
        settings.paths.output_dir / "research" / "trade_count_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _audit_strategy_rows(settings)
    grouped: defaultdict[tuple[str, str], list[VolumeStrategyDNA]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(row.market, row.timeframe)].append(row)
    funnels: list[dict[str, Any]] = []
    filter_attrition: list[dict[str, Any]] = []
    holding_periods: list[dict[str, Any]] = []
    exit_reasons: list[dict[str, Any]] = []
    data_attrition: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for (market, timeframe), selected_rows in sorted(grouped.items()):
        path = (
            settings.paths.processed_data_dir
            / f"{market}_{timeframe}.parquet"
        )
        btc_path = (
            settings.paths.processed_data_dir
            / f"BTC-EUR_{timeframe}.parquet"
        )
        if not path.is_file() or not btc_path.is_file():
            skipped.append(
                {
                    "market": market,
                    "timeframe": timeframe,
                    "reason": "REAL_DATA_NOT_AVAILABLE",
                }
            )
            continue
        frame = _clean_frame(
            load_ohlcv(
                path,
                market=market,
                timeframe=timeframe,
                closed_candles_only=True,
            )
        )
        btc_frame = _clean_frame(
            load_ohlcv(
                btc_path,
                market="BTC-EUR",
                timeframe=timeframe,
                closed_candles_only=True,
            )
        )
        if len(frame) < 500 or len(btc_frame) < 500:
            skipped.append(
                {
                    "market": market,
                    "timeframe": timeframe,
                    "reason": "INSUFFICIENT_REAL_BARS",
                    "rows": len(frame),
                }
            )
            continue
        frame_cache[(market, timeframe)] = frame
        batch = backtest_volume_strategy_batch(
            frame,
            btc_frame,
            tuple(selected_rows),
            fee_rate=settings.costs.default_fee,
            slippage_bps=settings.costs.slippage_bps,
            spread_bps=settings.costs.spread_bps,
            stressed_cost_multiplier=settings.costs.stressed_cost_multiplier,
        )
        prepared_features = volume_features(frame)
        prepared_mfi = _money_flow_index(frame)
        for row in selected_rows:
            definition = _volume_signal_definition(
                frame,
                row,
                prepared_volume_features=prepared_features,
                prepared_mfi=prepared_mfi,
            )
            returns = batch.returns[row.strategy_id]
            equity = (1.0 + returns).cumprod()
            positive = float(returns[returns > 0.0].sum())
            negative = abs(float(returns[returns < 0.0].sum()))
            result = _audit_one(
                frame,
                strategy_id=row.strategy_id,
                market=market,
                timeframe=timeframe,
                definition=definition,
                settings=settings,
                exact_metrics={
                    "net_return": float(equity.iloc[-1] - 1.0),
                    "profit_factor": (
                        positive / negative
                        if negative > 0.0
                        else (
                            math.inf if positive > 0.0 else 0.0
                        )
                    ),
                    "maximum_drawdown": float(
                        (equity / equity.cummax() - 1.0).min()
                    ),
                    "turnover": float(
                        batch.turnover[row.strategy_id].sum()
                    ),
                },
            )
            funnels.append(result["funnel"])
            filter_attrition.extend(result["filter_attrition"])
            holding_periods.append(result["holding"])
            exit_reasons.extend(result["exit_reasons"])
            data_attrition.extend(result["data_attrition"])
            ablations.extend(result["ablation"])
    alignment_rows: list[dict[str, Any]] = []
    path_4h = settings.paths.processed_data_dir / "BTC-EUR_4h.parquet"
    path_1d = settings.paths.processed_data_dir / "BTC-EUR_1d.parquet"
    if path_4h.is_file() and path_1d.is_file():
        frame_4h = frame_cache.get(("BTC-EUR", "4h"))
        if frame_4h is None:
            frame_4h = _clean_frame(
                load_ohlcv(
                    path_4h,
                    market="BTC-EUR",
                    timeframe="4h",
                    closed_candles_only=True,
                )
            )
        frame_1d = _clean_frame(
            load_ohlcv(
                path_1d,
                market="BTC-EUR",
                timeframe="1d",
                closed_candles_only=True,
            )
        )
        mtf_row = VolumeStrategyDNA(
            market="BTC-EUR",
            timeframe="4h",
            archetype="DONCHIAN_RVOL_BREAKOUT",
            coordinate=2,
            parameters=_parameter_for("DONCHIAN_RVOL_BREAKOUT", 2),
        )
        definition, alignment = _multi_timeframe_definition(
            frame_4h,
            frame_1d,
            mtf_row,
        )
        result = _audit_one(
            frame_4h,
            strategy_id=alignment["strategy_id"],
            market="BTC-EUR",
            timeframe="4h",
            definition=definition,
            settings=settings,
            multi_timeframe=True,
        )
        funnels.append(result["funnel"])
        filter_attrition.extend(result["filter_attrition"])
        holding_periods.append(result["holding"])
        exit_reasons.extend(result["exit_reasons"])
        data_attrition.extend(result["data_attrition"])
        ablations.extend(result["ablation"])
        alignment_rows.append(alignment)
    classical = _audit_classical_block_candidates(settings)
    funnels.extend(classical["funnels"])
    filter_attrition.extend(classical["filter_attrition"])
    holding_periods.extend(classical["holding_periods"])
    exit_reasons.extend(classical["exit_reasons"])
    data_attrition.extend(classical["data_attrition"])
    ablations.extend(classical["ablations"])
    skipped.extend(classical["skipped"])
    paths = {
        "signal_funnels": output_dir / "signal_funnels.csv",
        "filter_attrition": output_dir / "filter_attrition.csv",
        "holding_periods": output_dir / "holding_periods.csv",
        "exit_reasons": output_dir / "exit_reasons.csv",
        "data_attrition": output_dir / "data_attrition.csv",
        "ablation_results": output_dir / "ablation_results.csv",
    }
    for key, values in (
        ("signal_funnels", funnels),
        ("filter_attrition", filter_attrition),
        ("holding_periods", holding_periods),
        ("exit_reasons", exit_reasons),
        ("data_attrition", data_attrition),
        ("ablation_results", ablations),
    ):
        _write_csv(paths[key], values)
    alignment_payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "joins": alignment_rows,
        "required_semantics": "LAST_FULLY_CLOSED_BACKWARD_ASOF",
        "all_joins_causal": all(
            bool(row["causal"]) for row in alignment_rows
        ),
        "ordinary_inner_join_used": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(
        output_dir / "alignment_audit.json",
        alignment_payload,
    )
    atomic_write_json(
        output_dir / "cooldown_audit.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "configured_cooldown": 0,
            "cooldown_unit": "BARS",
            "effective_cooldown_hours": 0.0,
            "effective_cooldown_days": 0.0,
            "signals_blocked_by_cooldown": 0,
            "status": "NO_COOLDOWN_SUPPRESSION",
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    atomic_write_json(
        output_dir / "position_lock_audit.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "lock_scope": "ONE_POSITION_PER_STRATEGY_ASSET_SLEEVE",
            "global_portfolio_lock": False,
            "cross_asset_suppression": False,
            "cross_strategy_suppression": False,
            "blocked_existing_position": sum(
                int(row["blocked_existing_position"]) for row in funnels
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    classifications = Counter(
        str(row["low_trade_count_classification"])
        for row in funnels
        if row["low_trade_count"]
    )
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "COMPLETE",
        "generated_at": utc_iso(),
        "strategy_asset_funnel_count": len(funnels),
        "unique_strategy_count": len(
            {
                str(row["strategy_id"])
                for row in funnels
            }
        ),
        # Backward-compatible alias retained for existing report consumers.
        "strategy_count": len(funnels),
        "positive_1h_4h_strategy_ids_requested": len(
            _load_positive_volume_ids(settings)
        )
        + len(_classical_positive_rows(settings)),
        "available_15m_baselines_requested": sum(
            row.timeframe == "15m" for row in rows
        ),
        "multi_timeframe_audits": len(alignment_rows),
        "low_trade_strategy_count": sum(
            bool(row["low_trade_count"]) for row in funnels
        ),
        "low_trade_classifications": dict(sorted(classifications.items())),
        "trade_definition": "COMPLETED_ROUND_TRIP",
        "profit_factor_unit": "PORTFOLIO_PERIOD_RETURN",
        "normal_costs_included": True,
        "fee_rate": settings.costs.default_fee,
        "slippage_bps": settings.costs.slippage_bps,
        "spread_bps": settings.costs.spread_bps,
        "skipped": skipped,
        "artifact_hash": stable_hash(
            {
                "funnels": funnels,
                "alignment": alignment_rows,
                "skipped": skipped,
            },
            length=64,
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    report_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['strategy_id']))}</td>"
        f"<td>{html.escape(str(row['market']))}</td>"
        f"<td>{html.escape(str(row['timeframe']))}</td>"
        f"<td>{int(row['raw_entry_signal_count'])}</td>"
        f"<td>{int(row['completed_round_trip_count'])}</td>"
        f"<td>{float(row['time_in_market']):.2%}</td>"
        f"<td>{html.escape(str(row['low_trade_count_classification']))}</td>"
        "</tr>"
        for row in funnels
    )
    report = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Trade-count audit</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:"
        "collapse;width:100%}th,td{border:1px solid #ccc;padding:.35rem;"
        "text-align:left}th{background:#eee}</style></head><body>"
        "<h1>Trade-count and signal-funnel audit</h1>"
        f"<p>Strategies: {len(funnels)}. Trade definition: completed round "
        "trip. Normal costs included. Orders generated: 0.</p>"
        "<table><thead><tr><th>Strategy</th><th>Market</th><th>TF</th>"
        "<th>Raw signals</th><th>Round trips</th><th>Time in market</th>"
        "<th>Classification</th></tr></thead><tbody>"
        f"{report_rows}</tbody></table></body></html>"
    )
    (output_dir / "report.html").write_text(report, encoding="utf-8")
    return {
        **summary,
        "artifacts": {
            **{key: str(value.resolve()) for key, value in paths.items()},
            "summary": str((output_dir / "summary.json").resolve()),
            "alignment_audit": str(
                (output_dir / "alignment_audit.json").resolve()
            ),
            "cooldown_audit": str(
                (output_dir / "cooldown_audit.json").resolve()
            ),
            "position_lock_audit": str(
                (output_dir / "position_lock_audit.json").resolve()
            ),
            "report": str((output_dir / "report.html").resolve()),
        },
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "SignalDefinition",
    "run_trade_count_audit",
]
