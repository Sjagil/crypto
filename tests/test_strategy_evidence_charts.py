from __future__ import annotations

import numpy as np
import pandas as pd

from reporting.strategy_evidence_charts import (
    generate_campaign_stochastic_chart,
    generate_seven_year_evidence_tree,
    generate_seven_year_run_evidence,
    generate_strategy_evidence_bundle,
)
from research.backtest import BacktestResult
from utils.common import atomic_write_json, read_json


def _result(strategy_id: str, returns: np.ndarray) -> BacktestResult:
    index = pd.date_range("2025-01-01", periods=len(returns) + 1, freq="D", tz="UTC")
    equity = 10_000.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    return BacktestResult(
        strategy_id=strategy_id,
        initial_cash_eur=10_000.0,
        ending_equity_eur=float(equity[-1]),
        equity_curve=pd.DataFrame({"equity": equity}, index=index),
        trades=(),
        orders=(),
        metrics={},
        integrity={},
    )


def test_strategy_evidence_bundle_writes_chart_regimes_and_stochastic_evidence(
    tmp_path,
) -> None:
    returns = np.tile(np.array([0.01, -0.004, 0.006, -0.002]), 90)
    normal = _result("mtf-unit", returns)
    stressed = _result("mtf-unit", returns - 0.0002)
    benchmark_index = normal.equity_curve.index
    benchmark = pd.DataFrame(
        {
            "close": np.linspace(1_000.0, 1_300.0, len(benchmark_index)),
        },
        index=benchmark_index,
    )
    monte_carlo = {
        "passed": True,
        "p05_total_return": 0.05,
        "median_total_return": 0.20,
        "p95_total_return": 0.40,
    }
    dirichlet = {
        "passed": True,
        "profiles": [
            {"concentration_alpha": 0.5, "probability_positive": 0.90},
            {"concentration_alpha": 1.0, "probability_positive": 0.95},
        ],
    }
    stochastic = {
        "policy_hash": "a" * 64,
        "normal": {"monte_carlo": monte_carlo, "dirichlet": dirichlet},
        "stressed": {"monte_carlo": monte_carlo, "dirichlet": dirichlet},
    }

    evidence = generate_strategy_evidence_bundle(
        tmp_path,
        strategy_dna="b" * 64,
        timeframe="1h",
        normal_result=normal,
        stressed_result=stressed,
        stochastic=stochastic,
        benchmark=benchmark,
    )

    assert evidence["monte_carlo_passed"] is True
    assert evidence["dirichlet_passed"] is True
    assert evidence["regime_attribution"]["rows"]
    assert (tmp_path / ("b" * 24) / "strategy_robustness.png").is_file()
    assert (tmp_path / ("b" * 24) / "regime_attribution.csv").is_file()
    assert (tmp_path / ("b" * 24) / "strategy_evidence.json").is_file()


def test_campaign_stochastic_chart_uses_persisted_evidence(
    tmp_path,
) -> None:
    monte_carlo = {
        "passed": True,
        "simulations": 100,
        "p05_total_return": 0.05,
        "median_total_return": 0.20,
        "p95_total_return": 0.40,
        "median_maximum_drawdown": 0.04,
        "p95_maximum_drawdown": 0.08,
    }
    dirichlet = {
        "passed": True,
        "simulations_per_profile": 100,
        "profiles": [
            {"concentration_alpha": 0.5, "probability_positive": 0.90},
            {"concentration_alpha": 1.0, "probability_positive": 0.95},
        ],
    }
    periods = {
        name: {"net_return": value}
        for name, value in (
            ("development", 0.20),
            ("validation", 0.10),
            ("confirmation", 0.05),
        )
    }
    report_path = tmp_path / "campaign.json"
    atomic_write_json(
        report_path,
        {
            "primary_result": {
                "strategy_id": "RR_TEST",
                "strategy_dna_hash": "c" * 64,
                "periods": periods,
                "stressed_periods": periods,
                "gates": {
                    "stochastic_validation": {
                        "normal": {
                            "monte_carlo": monte_carlo,
                            "dirichlet": dirichlet,
                        },
                        "stressed": {
                            "monte_carlo": monte_carlo,
                            "dirichlet": dirichlet,
                        },
                    }
                },
            }
        },
    )
    evidence = generate_campaign_stochastic_chart(
        report_path,
        tmp_path / "evidence",
    )
    assert evidence["orders_generated"] == 0
    assert evidence["monte_carlo"]["simulations"] == 100
    assert (tmp_path / "evidence" / ("c" * 24) / "stochastic_robustness.png").is_file()


def test_seven_year_evidence_exports_reconciled_tables_and_chart(
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "seven-year"
    run_dir.mkdir(parents=True)
    result_path = run_dir / "seven_year_result.json"
    atomic_write_json(
        result_path,
        {
            "strategy_id": "seven-year-test",
            "strategy_dna_hash": "d" * 64,
            "market": "BTC-EUR",
            "timeframe": "4h",
            "status": "FAILED_STRESS",
            "annual_returns": [
                {"year": 2025, "net_return": 0.10},
                {"year": 2026, "net_return": -0.02},
            ],
            "regime_performance": [
                {"regime": "UPTREND", "compounded_return": 0.15},
                {"regime": "DOWNTREND", "compounded_return": -0.04},
            ],
            "walk_forward": {
                "anchored": {
                    "folds": [{"fold": 1, "net_return": 0.01}],
                },
                "rolling": {
                    "folds": [{"fold": 1, "net_return": -0.01}],
                },
            },
            "normal_costs": {
                "metrics": {
                    "net_return": 0.08,
                    "profit_factor": 1.20,
                    "maximum_drawdown": 0.05,
                    "trade_count": 100,
                }
            },
            "stressed_costs": {
                "metrics": {
                    "net_return": -0.01,
                    "profit_factor": 0.95,
                }
            },
            "capacity": [{"notional_eur": 1000, "net_return": 0.07}],
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    pd.DataFrame(
        {"equity": [10_000.0, 10_500.0, 10_300.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
    ).to_csv(run_dir / "normal_costs_equity.csv")

    evidence = generate_seven_year_run_evidence(result_path)
    tree = generate_seven_year_evidence_tree(tmp_path)

    assert evidence["orders_generated"] == 0
    assert (run_dir / "annual_returns.csv").is_file()
    assert (run_dir / "regime_performance.csv").is_file()
    assert (run_dir / "walk_forward.csv").is_file()
    assert (run_dir / "stress_results.csv").is_file()
    assert (run_dir / "capacity.csv").is_file()
    assert (run_dir / "rolling_12m_metrics.csv").is_file()
    assert (run_dir / "mandatory_statistics.json").is_file()
    assert (run_dir / "seven_year_evidence.png").is_file()
    mandatory = read_json(run_dir / "mandatory_statistics.json")
    assert mandatory["available"] is True
    assert mandatory["normal_metrics"]["annualized_volatility"] >= 0
    assert mandatory["orders_submitted"] == 0
    assert tree["run_count"] == 1
    assert (tmp_path / "evidence_index.json").is_file()
