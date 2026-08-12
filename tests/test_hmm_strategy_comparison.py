from __future__ import annotations

import pandas as pd

from core.cli import build_parser
from research.hmm_strategy_comparison import (
    HMMConditionedStrategy,
    _flat_rows,
    _write_chart,
    _write_human_reports,
    _write_top50_mtf_pipeline_reports,
)
from research.strategies import get_strategy
from utils.common import stable_hash


def _features() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=250, freq="1D", tz="UTC")
    close = pd.Series(range(100, 350), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "close": close,
            "ema_20": close - 1.0,
            "ema_50": close - 2.0,
            "ema_200": close - 3.0,
            "ema_50_slope": 0.01,
            "rsi_14": 55.0,
            "atr_14": 2.0,
            "bull_regime": True,
            "bear_regime": False,
            "_hmm_entry_size_multiplier": 0.55,
        },
        index=index,
    )
    return frame


def test_hmm_wrapper_changes_only_size() -> None:
    base = get_strategy("ema_trend_pullback")
    wrapped = HMMConditionedStrategy(
        base,
        dna_hash=stable_hash("test-hmm-wrapper"),
    )
    features = _features()
    original = base.generate(features)
    conditioned = wrapped.generate(features)
    assert conditioned.entry.equals(original.entry)
    assert conditioned.exit.equals(original.exit)
    assert conditioned.stop_distance.equals(original.stop_distance)
    assert conditioned.target_distance.equals(original.target_distance)
    assert conditioned.trailing_distance.equals(original.trailing_distance)
    assert conditioned.size_multiplier.equals(
        original.size_multiplier * 0.55
    )


def test_hmm_compare_all_cli_is_registered() -> None:
    args = build_parser().parse_args(["hmm", "compare-all"])
    assert args.command == "hmm"
    assert args.hmm_command == "compare-all"


def test_hmm_top50_mtf_cli_is_registered() -> None:
    args = build_parser().parse_args(["hmm", "top50-mtf"])
    assert args.command == "hmm"
    assert args.hmm_command == "top50-mtf"


def test_hmm_comparison_reports_finalize(tmp_path) -> None:
    metrics = {
        "net_return": 0.10,
        "cagr": 0.05,
        "profit_factor": 1.20,
        "net_expectancy_r": 0.10,
        "net_expectancy_eur": 1.0,
        "trade_count": 40,
        "effective_sample_size": 35.0,
        "maximum_drawdown": 0.08,
        "sharpe": 0.8,
        "sortino": 1.0,
        "calmar": 0.625,
        "average_exposure": 0.10,
        "turnover": 2.0,
        "transaction_costs_eur": 4.0,
        "monte_carlo_p95_drawdown": 0.12,
        "probability_of_loss": 0.1,
    }
    row = {
        "comparison_id": "comparison",
        "base_strategy_id": "base",
        "hmm_strategy_id": "base__hmm",
        "hmm_strategy_dna": "a" * 64,
        "family": "test",
        "timeframe": "1d",
        "comparison_score": 50.0,
        "base_normal": metrics,
        "hmm_normal": metrics,
        "delta_normal": {key: 0.0 for key in metrics},
        "base_stressed": metrics,
        "hmm_stressed": metrics,
        "drawdown_improvement": 0.0,
        "hmm_positive_after_costs": True,
        "stochastic_validation": {"passed": True},
    }
    flat = _flat_rows([row])
    index = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
    equity = pd.Series(range(100, 110), index=index, dtype=float)
    _write_chart(
        tmp_path / "hmm_all_strategies_comparison_v1.png",
        [row],
        {"comparison": (equity, equity)},
    )
    payload = {
        "campaign_id": "test",
        "summary": {
            "comparison_count": 1,
            "hmm_cagr_improvement_count": 0,
            "hmm_drawdown_improvement_count": 0,
            "hmm_positive_count": 1,
        },
    }
    _write_human_reports(tmp_path, payload, flat)
    artifacts = _write_top50_mtf_pipeline_reports(
        tmp_path,
        {
            **payload,
            "generated_at": "2026-08-07T00:00:00Z",
            "universe": ["BTC-EUR", "ETH-EUR"],
        },
        flat,
    )
    assert (tmp_path / "hmm_all_strategies_comparison_v1.png").is_file()
    assert (tmp_path / "hmm_all_strategies_comparison_v1.md").is_file()
    assert (tmp_path / "hmm_all_strategies_comparison_v1.html").is_file()
    assert (tmp_path / "top_50_mtf_strategy_pipeline_v1.json").is_file()
    assert (tmp_path / "top_50_mtf_strategy_pipeline_v1.csv").is_file()
    assert (tmp_path / "top_50_mtf_strategy_pipeline_v1.md").is_file()
    assert artifacts["json"].endswith("top_50_mtf_strategy_pipeline_v1.json")
