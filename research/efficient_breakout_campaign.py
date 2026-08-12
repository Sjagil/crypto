"""Bounded, causal and cost-aware 4h ATR breakout efficiency campaign.

The campaign separates cheap Stage-0 ranking from the exact portfolio engine.
Only development data may select DNA; the final chronological test is evaluated
once after a winner is frozen.  This module has no order or promotion authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.settings import Settings
from research.adaptive_crypto_campaign import PROMOTION_UNIVERSE, _load_frames
from research.portfolio_breakout import (
    EfficientAtrRiskBreakoutParameters,
    backtest_breakout_portfolio,
    efficient_atr_breakout_parameter_set,
)
from research.portfolio_selection import RotationPortfolioPolicy, _profit_factor
from utils.common import atomic_write_json, sha256_file, stable_hash, utc_iso

CAMPAIGN = "EFFICIENT_ATR_BREAKOUT_V2"
SCHEMA_VERSION = "efficient_atr_breakout_campaign_v1"
STAGE0_ENGINE_VERSION = "1.0.0"
PURGE_BARS = 180  # 30 days on 4h bars
EXACT_SURVIVOR_LIMIT = 3


def _panels(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = frames["BTC-EUR"].copy()
    benchmark.index = pd.to_datetime(benchmark.index, utc=True)
    index = benchmark.index
    columns = tuple(frames)
    opens = pd.DataFrame(index=index, columns=columns, dtype=float)
    highs = opens.copy()
    lows = opens.copy()
    closes = opens.copy()
    for market, raw in frames.items():
        frame = raw.copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        opens[market] = frame["open"].reindex(index)
        highs[market] = frame["high"].reindex(index)
        lows[market] = frame["low"].reindex(index)
        closes[market] = frame["close"].reindex(index)
    return opens, highs, lows, closes


def stage0_breakout_screen(
    frames: Mapping[str, pd.DataFrame],
    parameters: EfficientAtrRiskBreakoutParameters,
    *,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    minimum_history_observations: int = 1_200,
) -> dict[str, Any]:
    """Fast causal approximation used only to rank exact-backtest survivors."""

    if min(fee_rate, slippage_bps, spread_bps) < 0:
        raise ValueError("cost assumptions cannot be negative")
    opens, highs, lows, closes = _panels(frames)
    upper = closes.shift(1).rolling(parameters.entry_lookback).max()
    lower = closes.shift(1).rolling(parameters.exit_lookback).min()
    ema = closes.ewm(
        span=parameters.trend_ema_period,
        adjust=False,
        min_periods=parameters.trend_ema_period,
    ).mean()
    volatility = closes.pct_change(fill_method=None).rolling(
        parameters.volatility_lookback
    ).std(ddof=0)
    prior_close = closes.shift(1)
    true_range = pd.concat(
        [
            highs - lows,
            (highs - prior_close).abs(),
            (lows - prior_close).abs(),
        ],
        axis=1,
        keys=("range", "high_gap", "low_gap"),
    ).T.groupby(level=1).max().T
    atr_fraction = (
        true_range.rolling(parameters.atr_lookback).mean() / closes
    ).replace([np.inf, -np.inf], np.nan)
    strength = ((closes / upper - 1.0) / volatility).replace(
        [np.inf, -np.inf], np.nan
    )
    history = closes.notna().cumsum()
    valid = (
        (history >= minimum_history_observations)
        & closes.notna()
        & upper.notna()
        & lower.notna()
        & ema.notna()
        & atr_fraction.notna()
    )
    entries = valid & (closes > upper) & (closes > ema)
    exits = (~valid) | (closes < lower) | (closes < ema)

    open_values = opens.to_numpy(dtype=float)
    atr_values = atr_fraction.to_numpy(dtype=float)
    strength_values = strength.to_numpy(dtype=float)
    entry_values = entries.to_numpy(dtype=bool)
    exit_values = exits.to_numpy(dtype=bool)
    columns = tuple(closes.columns)
    warmup = max(
        parameters.entry_lookback,
        parameters.exit_lookback,
        parameters.trend_ema_period,
        parameters.atr_lookback,
        minimum_history_observations,
    )
    current = np.zeros(len(columns), dtype=float)
    equity = 1.0
    equity_rows: list[tuple[pd.Timestamp, float]] = [(closes.index[warmup], equity)]
    total_turnover = 0.0
    rebalance_count = 0
    one_way_cost = fee_rate + slippage_bps / 10_000.0 + spread_bps / 20_000.0

    for execution_index in range(warmup + 1, len(closes)):
        decision_index = execution_index - 1
        held = current > 1e-12
        urgent_exit = bool(np.any(held & exit_values[decision_index]))
        entry_now = bool(np.any(entry_values[decision_index]))
        scheduled = (
            (decision_index - warmup) % parameters.rebalance_days == 0
        )
        if scheduled or urgent_exit or entry_now:
            retained = [
                index
                for index in np.flatnonzero(held)
                if not exit_values[decision_index, index]
            ]
            candidates = [
                index
                for index in np.flatnonzero(entry_values[decision_index])
                if index not in retained
            ]
            candidates.sort(
                key=lambda index: (
                    -float(strength_values[decision_index, index]),
                    columns[index],
                )
            )
            selected = (retained + candidates)[: parameters.maximum_positions]
            target = np.zeros_like(current)
            for index in selected:
                atr = float(atr_values[decision_index, index])
                if math.isfinite(atr) and atr > 0:
                    target[index] = min(
                        0.20,
                        parameters.risk_fraction_per_position
                        / (atr * parameters.atr_stop_multiple),
                    )
            exposure = float(target.sum())
            if exposure > 0.40:
                target *= 0.40 / exposure
            executable = np.isfinite(open_values[execution_index])
            target = np.where(executable, target, current)
            same_selection = set(np.flatnonzero(target > 1e-12)) == set(
                np.flatnonzero(current > 1e-12)
            )
            if (
                same_selection
                and bool(np.any(target > 1e-12))
                and float(np.max(np.abs(target - current)))
                < parameters.rebalance_buffer
            ):
                target = current.copy()
            turnover = float(np.abs(target - current).sum())
            if turnover > 1e-15:
                equity *= 1.0 - turnover * one_way_cost
                total_turnover += turnover
                rebalance_count += 1
            current = target

        terminal = execution_index == len(closes) - 1
        if terminal:
            next_return = closes.iloc[execution_index].to_numpy(dtype=float) / open_values[
                execution_index
            ] - 1.0
        else:
            next_return = open_values[execution_index + 1] / open_values[execution_index] - 1.0
        next_return = np.where(np.isfinite(next_return), next_return, 0.0)
        equity *= 1.0 + float(np.dot(current, next_return))
        if terminal:
            turnover = float(current.sum())
            equity *= 1.0 - turnover * one_way_cost
            total_turnover += turnover
        timestamp = closes.index[execution_index] if terminal else opens.index[execution_index + 1]
        equity_rows.append((timestamp, equity))

    curve = pd.Series(
        [value for _, value in equity_rows],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in equity_rows]),
        dtype=float,
        name="stage0_equity",
    )
    weekly = curve.resample("7D").last().pct_change(fill_method=None).dropna()
    drawdown = curve / curve.cummax() - 1.0
    return {
        "strategy_dna_hash": parameters.dna_hash,
        "stage0_engine_version": STAGE0_ENGINE_VERSION,
        "result_type": "CAUSAL_APPROXIMATION_NOT_PROMOTION_EVIDENCE",
        "net_return": float(curve.iloc[-1] - 1.0),
        "portfolio_period_profit_factor": _profit_factor(weekly),
        "maximum_drawdown": float(drawdown.min()),
        "turnover": total_turnover,
        "rebalance_count": rebalance_count,
        "observations": int(len(curve)),
        "no_lookahead": True,
        "decision_at_close_execution_next_open": True,
        "orders_generated": 0,
    }


def _period_metrics(curve: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    period = curve.loc[(curve.index >= start) & (curve.index <= end)].dropna()
    if len(period) < 3:
        return {"evaluable": False, "observations": int(len(period))}
    normalized = period / float(period.iloc[0])
    weekly = normalized.resample("7D").last().pct_change(fill_method=None).dropna()
    drawdown = normalized / normalized.cummax() - 1.0
    return {
        "evaluable": True,
        "start": period.index[0].isoformat(),
        "end": period.index[-1].isoformat(),
        "observations": int(len(period)),
        "net_return": float(normalized.iloc[-1] - 1.0),
        "profit_factor": _profit_factor(weekly),
        "maximum_drawdown": float(drawdown.min()),
    }


def _exact_gate(normal: dict[str, Any], stressed: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for label, row in (("NORMAL", normal), ("STRESSED", stressed)):
        if not row.get("evaluable"):
            failures.append(f"{label}_NOT_EVALUABLE")
            continue
        if float(row["net_return"]) <= 0.0:
            failures.append(f"{label}_NET_NOT_POSITIVE")
        if float(row["profit_factor"]) <= 1.0:
            failures.append(f"{label}_PF_NOT_ABOVE_ONE")
        if float(row["maximum_drawdown"]) < -0.25:
            failures.append(f"{label}_DRAWDOWN_ABOVE_25_PERCENT")
    return not failures, failures


def run_efficient_breakout_campaign(settings: Settings) -> dict[str, Any]:
    """Run bounded Stage-0, exact development selection, then one untouched test."""

    frames, hashes = _load_frames(
        settings.paths.processed_data_dir,
        PROMOTION_UNIVERSE,
        "4h",
    )
    index = pd.DatetimeIndex(frames["BTC-EUR"].index)
    train_end_index = int(len(index) * 0.60)
    test_start_index = int(len(index) * 0.80)
    train_end = index[train_end_index]
    validation_start = index[train_end_index + PURGE_BARS]
    development_end = index[test_start_index - PURGE_BARS]
    test_start = index[test_start_index + PURGE_BARS]
    development_frames = {
        market: frame.loc[:development_end].copy() for market, frame in frames.items()
    }
    parameters = efficient_atr_breakout_parameter_set()
    normal_costs = {
        "fee_rate": settings.costs.default_fee,
        "slippage_bps": settings.costs.slippage_bps,
        "spread_bps": settings.costs.spread_bps,
    }
    multiplier = settings.costs.stressed_cost_multiplier
    stressed_costs = {
        "fee_rate": settings.costs.default_fee * multiplier,
        "slippage_bps": settings.costs.slippage_bps * multiplier,
        "spread_bps": settings.costs.spread_bps * multiplier,
    }
    policy = RotationPortfolioPolicy(
        allowed_markets=PROMOTION_UNIVERSE,
        maximum_total_exposure=0.40,
        maximum_position_exposure=0.20,
        minimum_cash=0.60,
        minimum_history_observations=1_200,
    )
    stage0_rows: list[dict[str, Any]] = []
    for row in parameters:
        normal = stage0_breakout_screen(development_frames, row, **normal_costs)
        stressed = stage0_breakout_screen(development_frames, row, **stressed_costs)
        passed = bool(
            normal["net_return"] > 0.0
            and normal["portfolio_period_profit_factor"] > 1.0
            and stressed["net_return"] > 0.0
            and stressed["portfolio_period_profit_factor"] > 1.0
        )
        stage0_rows.append(
            {
                "parameters": asdict(row),
                "strategy_dna_hash": row.dna_hash,
                "normal": normal,
                "stressed": stressed,
                "passed": passed,
                "ranking_score": min(
                    float(normal["net_return"]),
                    float(stressed["net_return"]),
                ),
            }
        )
    ranked = sorted(
        stage0_rows,
        key=lambda row: (not row["passed"], -float(row["ranking_score"]), row["strategy_dna_hash"]),
    )
    survivors = ranked[:EXACT_SURVIVOR_LIMIT]
    exact_rows: list[dict[str, Any]] = []
    by_hash = {row.dna_hash: row for row in parameters}
    for survivor in survivors:
        row = by_hash[survivor["strategy_dna_hash"]]
        normal_result = backtest_breakout_portfolio(
            development_frames, row, portfolio_policy=policy, **normal_costs
        )
        stressed_result = backtest_breakout_portfolio(
            development_frames, row, portfolio_policy=policy, **stressed_costs
        )
        train_normal = _period_metrics(normal_result.equity_curve, index[0], train_end)
        train_stressed = _period_metrics(stressed_result.equity_curve, index[0], train_end)
        validation_normal = _period_metrics(
            normal_result.equity_curve, validation_start, development_end
        )
        validation_stressed = _period_metrics(
            stressed_result.equity_curve, validation_start, development_end
        )
        train_passed, train_failures = _exact_gate(train_normal, train_stressed)
        validation_passed, validation_failures = _exact_gate(
            validation_normal, validation_stressed
        )
        exact_rows.append(
            {
                "strategy_dna_hash": row.dna_hash,
                "parameters": asdict(row),
                "train": {"normal": train_normal, "stressed": train_stressed},
                "validation": {
                    "normal": validation_normal,
                    "stressed": validation_stressed,
                },
                "development_exact": {
                    "normal": normal_result.summary(),
                    "stressed": stressed_result.summary(),
                },
                "passed": train_passed and validation_passed,
                "failure_reasons": train_failures + validation_failures,
                "ranking_score": min(
                    float(validation_normal.get("net_return", -1.0)),
                    float(validation_stressed.get("net_return", -1.0)),
                ),
            }
        )
    exact_ranked = sorted(
        exact_rows,
        key=lambda row: (not row["passed"], -float(row["ranking_score"]), row["strategy_dna_hash"]),
    )
    winner = next((row for row in exact_ranked if row["passed"]), None)
    untouched: dict[str, Any] = {
        "status": "NOT_RUN_NO_DEVELOPMENT_SURVIVOR",
        "passed": False,
    }
    if winner is not None:
        frozen = by_hash[winner["strategy_dna_hash"]]
        normal_result = backtest_breakout_portfolio(
            frames, frozen, portfolio_policy=policy, **normal_costs
        )
        stressed_result = backtest_breakout_portfolio(
            frames, frozen, portfolio_policy=policy, **stressed_costs
        )
        normal = _period_metrics(normal_result.equity_curve, test_start, index[-1])
        stressed = _period_metrics(stressed_result.equity_curve, test_start, index[-1])
        passed, failures = _exact_gate(normal, stressed)
        untouched = {
            "status": "EVALUATED_ONCE_AFTER_DNA_FREEZE",
            "strategy_dna_hash": frozen.dna_hash,
            "normal": normal,
            "stressed": stressed,
            "passed": passed,
            "failure_reasons": failures,
            "full_history_integrity": normal_result.integrity,
        }

    report_dir = settings.paths.lab_dir / "reports"
    report_path = report_dir / "efficient_atr_breakout_v2.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "generated_at": utc_iso(),
        "status": (
            "UNTOUCHED_TEST_PASSED_SHADOW_REVIEW_ONLY"
            if untouched["passed"]
            else "REJECTED_OR_DATA_PENDING"
        ),
        "selection_protocol": {
            "parameter_set_frozen_before_evaluation": True,
            "candidate_count": len(parameters),
            "exact_survivor_limit": EXACT_SURVIVOR_LIMIT,
            "train_end": train_end.isoformat(),
            "validation_start_after_purge": validation_start.isoformat(),
            "development_end_before_purge": development_end.isoformat(),
            "untouched_test_start_after_embargo": test_start.isoformat(),
            "purge_bars": PURGE_BARS,
            "test_used_for_selection": False,
        },
        "costs": {"normal": normal_costs, "stressed": stressed_costs},
        "data_hashes": hashes,
        "data_fingerprint": stable_hash(hashes, length=64),
        "stage0": ranked,
        "exact_development": exact_ranked,
        "frozen_winner_dna_hash": winner["strategy_dna_hash"] if winner else None,
        "untouched_test": untouched,
        "paper_candidate": False,
        "live_candidate": False,
        "promotion_implied": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(report_path, payload)
    return {
        "status": payload["status"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "stage0_passed": sum(bool(row["passed"]) for row in stage0_rows),
        "exact_passed": sum(bool(row["passed"]) for row in exact_rows),
        "untouched_test_passed": bool(untouched["passed"]),
        "orders_generated": 0,
    }


__all__ = [
    "CAMPAIGN",
    "EXACT_SURVIVOR_LIMIT",
    "PURGE_BARS",
    "run_efficient_breakout_campaign",
    "stage0_breakout_screen",
]
