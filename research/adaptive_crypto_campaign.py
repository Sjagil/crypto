"""Exact adaptive intraday challengers for residual reversal and Turtle breakout.

The campaign compares fixed, causal DNA across a promotion-compatible universe
and a broader discovery-only universe. It has no order or promotion authority.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import Settings
from research.global_trial_accounting import resolve_known_trial_count
from research.optimization import multiple_testing_bootstrap
from research.portfolio_breakout import (
    AtrRiskBreakoutParameters,
    backtest_breakout_portfolio,
)
from research.portfolio_selection import RotationPortfolioPolicy
from research.residual_reversal import (
    adaptive_residual_reversal_parameter_set,
    backtest_residual_reversal,
)
from research.stochastic_validation import (
    policy_from_research_settings,
    validate_strategy_return_paths,
)
from utils.common import atomic_write_json, sha256_file, stable_hash, utc_iso

CAMPAIGN = "ADAPTIVE_CRYPTO_INTRADAY_V1"
PROMOTION_UNIVERSE = (
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "LINK-EUR",
    "TAO-EUR",
)
CORE_UNIVERSE = PROMOTION_UNIVERSE[:4]
DATA_PENDING_MARKETS = ("NPC-EUR",)
DISCOVERY_UNIVERSE = PROMOTION_UNIVERSE + (
    "ADA-EUR",
    "BNB-EUR",
    "DOGE-EUR",
    "XRP-EUR",
    "TRX-EUR",
    "AVAX-EUR",
    "NEAR-EUR",
    "SUI-EUR",
)


def _load_frames(
    root: Path,
    markets: tuple[str, ...],
    timeframe: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for market in markets:
        path = root / f"{market}_{timeframe}.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            timestamps = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise")
            frame.index = pd.DatetimeIndex(timestamps)
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
        else:
            raise ValueError(
                f"{path} has neither a timestamp column nor a DatetimeIndex"
            )
        if frame.index.hasnans or frame.index.has_duplicates:
            raise ValueError(f"{path} contains invalid or duplicate timestamps")
        frame = frame.sort_index()
        frames[market] = frame
        hashes[market] = sha256_file(path)
    if missing:
        raise FileNotFoundError(
            f"adaptive campaign lacks {timeframe} datasets: {missing}"
        )
    benchmark = frames.get("BTC-EUR")
    if benchmark is None or benchmark.empty:
        raise ValueError(f"BTC-EUR benchmark is unavailable for {timeframe}")
    minimum = 5_000 if timeframe == "1h" else 1_200
    insufficient = {
        market: len(frame)
        for market, frame in frames.items()
        if len(frame) < minimum
    }
    if insufficient:
        raise ValueError(
            f"individual {timeframe} histories are below warmup: {insufficient}"
        )
    # The backtest panel uses the complete BTC calendar. Each altcoin is
    # reindexed only inside the causal portfolio engine and remains NaN before
    # its real listing/history. Its own minimum-history gate controls when it
    # becomes eligible. A young asset must never truncate older assets.
    hashes["__point_in_time_panel__"] = stable_hash(
        {
            "timeframe": timeframe,
            "benchmark": "BTC-EUR",
            "benchmark_start": benchmark.index[0].isoformat(),
            "benchmark_end": benchmark.index[-1].isoformat(),
            "benchmark_rows": len(benchmark),
            "members": {
                market: {
                    "start": frame.index[0].isoformat(),
                    "end": frame.index[-1].isoformat(),
                    "rows": len(frame),
                }
                for market, frame in sorted(frames.items())
            },
            "selection": "BTC_CALENDAR_POINT_IN_TIME_MEMBER_AVAILABILITY",
            "pre_listing_fill": False,
        },
        length=64,
    )
    return frames, hashes


def _daily_returns(equity: pd.Series) -> pd.Series:
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change(fill_method=None).dropna()


def _candidate_row(
    *,
    strategy_id: str,
    family: str,
    timeframe: str,
    universe_label: str,
    universe: tuple[str, ...],
    parameters: Any,
    normal: Any,
    stressed: Any,
    settings: Settings,
    seed_offset: int,
) -> tuple[dict[str, Any], pd.Series]:
    normal_returns = _daily_returns(normal.equity_curve)
    stressed_returns = _daily_returns(stressed.equity_curve)
    aligned = pd.concat(
        [normal_returns.rename("normal"), stressed_returns.rename("stressed")],
        axis=1,
    ).dropna()
    stochastic_policy = policy_from_research_settings(
        settings.research,
        seed=settings.app.random_seed,
        expected_block_length=7,
    )
    stochastic = validate_strategy_return_paths(
        aligned["normal"],
        aligned["stressed"],
        policy=stochastic_policy,
        seed_offset=seed_offset,
    )
    integrity = dict(normal.integrity)
    normal_positive = bool(
        normal.metrics["net_return"] > 0.0
        and normal.metrics["portfolio_period_profit_factor"] > 1.0
        and integrity.get("decision_at_close_execution_next_open") is True
        and integrity.get("long_only_spot") is True
        and (
            integrity.get("no_lookahead") is True
            or integrity.get("strictly_prior_beta_estimation") is True
        )
    )
    stressed_positive = bool(
        stressed.metrics["net_return"] > 0.0
        and stressed.metrics["portfolio_period_profit_factor"] > 1.0
    )
    paper_adapter = (
        "ATR_TURTLE_4H"
        if family == "MULTI_ASSET_4H_TURTLE_ATR_RISK"
        else None
    )
    paper_eligible = bool(
        normal_positive
        and stressed_positive
        and stochastic.get("passed") is True
        and universe_label == "PROMOTION_COMPATIBLE"
        and paper_adapter is not None
    )
    paper_gate_failures: list[str] = []
    if not normal_positive:
        paper_gate_failures.append("NORMAL_ECONOMICS_OR_INTEGRITY_FAILED")
    if not stressed_positive:
        paper_gate_failures.append("DOUBLE_COST_ECONOMICS_FAILED")
    if stochastic.get("passed") is not True:
        paper_gate_failures.append("STOCHASTIC_VALIDATION_FAILED")
    if universe_label != "PROMOTION_COMPATIBLE":
        paper_gate_failures.append("DISCOVERY_UNIVERSE_ONLY")
    if paper_adapter is None:
        paper_gate_failures.append("PAPER_ADAPTER_UNAVAILABLE")
    row = {
        "strategy_id": strategy_id,
        "strategy_family": family,
        "strategy_dna_hash": parameters.dna_hash,
        "timeframe": timeframe,
        "universe_label": universe_label,
        "universe": list(universe),
        "parameters": asdict(parameters),
        "normal": normal.summary(),
        "stressed": stressed.summary(),
        "stochastic_validation": stochastic,
        "economic_checks": {
            "net_return_positive": normal.metrics["net_return"] > 0.0,
            "portfolio_period_profit_factor_above_one": (
                normal.metrics["portfolio_period_profit_factor"] > 1.0
            ),
            "stressed_net_return_positive": stressed.metrics["net_return"] > 0.0,
            "stressed_profit_factor_above_one": (
                stressed.metrics["portfolio_period_profit_factor"] > 1.0
            ),
        },
        "backtest_positive": normal_positive,
        "lifecycle": (
            "PAPER_ACTIVE"
            if paper_eligible
            else "RESEARCH_POSITIVE"
            if normal_positive
            else "REJECT"
        ),
        "paper_eligible": paper_eligible,
        "paper_adapter": paper_adapter,
        "paper_gate_failures": paper_gate_failures,
        "stochastic_tests_are_paper_gate": True,
        "multiple_testing_is_capital_scaling_evidence_only": True,
        "capital_scaling_warnings": list(
            stochastic.get("reason_codes") or []
        ),
        "promotion_scope": (
            "PAPER_REVIEW_ELIGIBLE_IF_ALL_GATES_PASS"
            if universe_label == "PROMOTION_COMPATIBLE"
            else "DISCOVERY_ONLY_REQUIRES_SEPARATE_ELIGIBILITY_REVIEW"
        ),
        "orders_generated": 0,
    }
    return row, normal_returns.rename(strategy_id)


def run_adaptive_crypto_campaign(settings: Settings) -> dict[str, Any]:
    """Run fixed 4h challengers plus multiple-testing and stochastic evidence."""

    processed = settings.paths.processed_data_dir
    cost_multiplier = settings.costs.stressed_cost_multiplier
    candidates: list[dict[str, Any]] = []
    return_paths: list[pd.Series] = []
    data_hashes: dict[str, dict[str, str]] = {}

    adaptive_parameters = {
        row.timeframe: row for row in adaptive_residual_reversal_parameter_set()
    }
    definitions = (
        ("ADAPTIVE_RR_4H_CORE4", "adaptive_rr", "4h", "PROMOTION_COMPATIBLE", CORE_UNIVERSE),
        (
            "ADAPTIVE_RR_1H_CORE5",
            "adaptive_rr",
            "1h",
            "PROMOTION_COMPATIBLE",
            PROMOTION_UNIVERSE,
        ),
        ("ATR_TURTLE_4H_CORE5", "atr_turtle", "4h", "PROMOTION_COMPATIBLE", PROMOTION_UNIVERSE),
        ("ATR_TURTLE_4H_EXPANDED13", "atr_turtle", "4h", "DISCOVERY_ONLY", DISCOVERY_UNIVERSE),
    )
    for ordinal, (strategy_id, kind, timeframe, label, universe) in enumerate(
        definitions,
        start=1,
    ):
        frames, hashes = _load_frames(processed, universe, timeframe)
        data_hashes[f"{label}:{timeframe}"] = hashes
        minimum_history = 1_200 if timeframe == "4h" else 5_000
        policy = RotationPortfolioPolicy(
            allowed_markets=universe,
            maximum_total_exposure=0.40,
            maximum_position_exposure=0.20,
            minimum_cash=0.60,
            minimum_history_observations=minimum_history,
        )
        if kind == "adaptive_rr":
            parameters = adaptive_parameters[timeframe]
            normal = backtest_residual_reversal(
                frames,
                parameters,
                fee_rate=settings.costs.default_fee,
                slippage_bps=settings.costs.slippage_bps,
                spread_bps=settings.costs.spread_bps,
                portfolio_policy=policy,
            )
            stressed = backtest_residual_reversal(
                frames,
                parameters,
                fee_rate=settings.costs.default_fee * cost_multiplier,
                slippage_bps=settings.costs.slippage_bps * cost_multiplier,
                spread_bps=settings.costs.spread_bps * cost_multiplier,
                portfolio_policy=policy,
            )
            family = "BTC_REGIME_ADAPTIVE_PERCENTILE_RESIDUAL_MEAN_REVERSION"
        else:
            parameters = AtrRiskBreakoutParameters()
            normal = backtest_breakout_portfolio(
                frames,
                parameters,
                fee_rate=settings.costs.default_fee,
                slippage_bps=settings.costs.slippage_bps,
                spread_bps=settings.costs.spread_bps,
                portfolio_policy=policy,
            )
            stressed = backtest_breakout_portfolio(
                frames,
                parameters,
                fee_rate=settings.costs.default_fee * cost_multiplier,
                slippage_bps=settings.costs.slippage_bps * cost_multiplier,
                spread_bps=settings.costs.spread_bps * cost_multiplier,
                portfolio_policy=policy,
            )
            family = "MULTI_ASSET_4H_TURTLE_ATR_RISK"
        row, returns = _candidate_row(
            strategy_id=strategy_id,
            family=family,
            timeframe=timeframe,
            universe_label=label,
            universe=universe,
            parameters=parameters,
            normal=normal,
            stressed=stressed,
            settings=settings,
            seed_offset=ordinal * 10_000,
        )
        candidates.append(row)
        return_paths.append(returns)

    matrix = pd.concat(return_paths, axis=1).dropna(how="any")
    prior_known_trials = resolve_known_trial_count(
        settings.paths.lab_dir,
        local_known_trial_count=1,
    )
    multiple = multiple_testing_bootstrap(
        matrix,
        bootstrap_samples=settings.research.multiple_testing_bootstrap_samples,
        block_size=max(7, settings.research.multiple_testing_block_size),
        seed=settings.app.random_seed,
        known_trial_count=prior_known_trials + len(candidates),
    )
    for row in candidates:
        row["multiple_testing"] = {
            "deflated_sharpe_probability": (
                multiple.deflated_sharpe_probabilities.get(row["strategy_id"])
            ),
            "family_white_reality_check_pvalue": multiple.white_reality_check_pvalue,
            "family_hansen_spa_pvalue": multiple.hansen_spa_pvalue,
            "family_probability_of_backtest_overfitting": (
                multiple.probability_of_backtest_overfitting
            ),
        }

    report_dir = settings.paths.lab_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "adaptive_crypto_intraday_v1.json"
    csv_path = report_dir / "adaptive_crypto_intraday_v1.csv"
    md_path = report_dir / "adaptive_crypto_intraday_v1.md"
    payload = {
        "schema_version": "adaptive_crypto_intraday_v1",
        "campaign": CAMPAIGN,
        "status": "COMPLETED_RESEARCH_ONLY",
        "generated_at": utc_iso(),
        "hypotheses": [
            "Lower execution timeframes may increase decisions but not necessarily independent evidence.",
            "Strictly-prior residual percentiles may adapt thresholds without altering frozen RR DNA.",
            "ATR risk units target equal ex-ante EUR risk subject to position and portfolio caps.",
            "Expanded-universe results are discovery-only and cannot bypass execution eligibility.",
        ],
        "candidate_count": len(candidates),
        "return_matrix_observations": int(len(matrix)),
        "multiple_testing": asdict(multiple),
        "prior_known_trials": prior_known_trials,
        "total_known_trials": prior_known_trials + len(candidates),
        "candidates": candidates,
        "data_hashes": data_hashes,
        "data_pending_markets": {
            "NPC-EUR": (
                "Excluded because its individual real 4h history remains below "
                "the fixed causal warmup; prior standalone NPC screens remain."
            ),
        },
        "data_fingerprint": stable_hash(data_hashes, length=64),
        "frozen_controls_modified": False,
        "paper_candidates": sum(
            bool(row.get("paper_eligible")) for row in candidates
        ),
        "live_candidates": 0,
        "orders_generated": 0,
    }
    atomic_write_json(json_path, payload)
    pd.DataFrame(
        [
            {
                "strategy_id": row["strategy_id"],
                "family": row["strategy_family"],
                "timeframe": row["timeframe"],
                "universe_label": row["universe_label"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "net_return": row["normal"]["metrics"]["net_return"],
                "stressed_net_return": row["stressed"]["metrics"]["net_return"],
                "profit_factor": row["normal"]["metrics"]["portfolio_period_profit_factor"],
                "stressed_profit_factor": row["stressed"]["metrics"]["portfolio_period_profit_factor"],
                "sharpe": row["normal"]["metrics"]["sharpe"],
                "maximum_drawdown": row["normal"]["metrics"]["maximum_drawdown"],
                "decision_count": row["normal"]["metrics"].get(
                    "decision_count",
                    row["normal"]["metrics"].get("rebalance_count", 0),
                ),
                "effective_sample_size": row["normal"]["metrics"]["portfolio_period_effective_sample_size"],
                "average_exposure": row["normal"]["metrics"]["average_exposure"],
                "monte_carlo_dirichlet_pass": row["stochastic_validation"]["passed"],
                "dsr": row["multiple_testing"]["deflated_sharpe_probability"],
                "wrc_pvalue": row["multiple_testing"]["family_white_reality_check_pvalue"],
                "spa_pvalue": row["multiple_testing"]["family_hansen_spa_pvalue"],
                "pbo": row["multiple_testing"]["family_probability_of_backtest_overfitting"],
            }
            for row in candidates
        ]
    ).to_csv(csv_path, index=False)
    lines = [
        "# Adaptive crypto intraday v1",
        "",
        "Frozen controls remain byte-for-byte unchanged; every row below is a new DNA.",
        "",
        "| Strategy | TF | Universe | Net | Stress net | PF | Stress PF | Sharpe | Max DD | Decisions | ESS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        normal = row["normal"]["metrics"]
        stressed = row["stressed"]["metrics"]
        lines.append(
            f"| {row['strategy_id']} | {row['timeframe']} | {row['universe_label']} "
            f"| {normal['net_return']:.2%} | {stressed['net_return']:.2%} "
            f"| {normal['portfolio_period_profit_factor']:.3f} "
            f"| {stressed['portfolio_period_profit_factor']:.3f} "
            f"| {normal['sharpe']:.2f} | {normal['maximum_drawdown']:.2%} "
            f"| {normal.get('decision_count', normal.get('rebalance_count', 0))} "
            f"| {normal['portfolio_period_effective_sample_size']} |"
        )
    lines.extend(
        [
            "",
            f"- WRC p-value: {multiple.white_reality_check_pvalue:.4f}",
            f"- Hansen SPA p-value: {multiple.hansen_spa_pvalue:.4f}",
            f"- PBO: {multiple.probability_of_backtest_overfitting}",
            "- Orders generated: 0",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": payload["status"],
        "campaign": CAMPAIGN,
        "report": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "candidate_count": len(candidates),
        "orders_generated": 0,
    }


__all__ = [
    "CAMPAIGN",
    "DISCOVERY_UNIVERSE",
    "DATA_PENDING_MARKETS",
    "PROMOTION_UNIVERSE",
    "run_adaptive_crypto_campaign",
]
