from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from data.downloader import CanonicalDownloader
from data.market_data import save_ohlcv
from research.seven_year import (
    _analytic_regime_returns,
    _rolling_twelve_month_metrics,
    _walk_forward_oos_summary,
    audit_dataset,
    audit_repository,
    build_legacy_comparison,
    build_positive_timeframe_rerun_queue,
    build_seven_year_rankings,
    common_window,
    exact_calendar_start,
    has_exact_calendar_years,
    minimum_trades_for_timeframe,
    record_seven_year_history_exclusion,
    strategy_history_status,
)
from utils.common import atomic_write_json

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)


def _frame(start: str, end: str, *, frequency: str = "1D") -> pd.DataFrame:
    index = pd.date_range(start, end, freq=frequency, tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        },
        index=index,
    )


def _write(frame: pd.DataFrame, tmp_path, name: str = "BTC-EUR_1d.parquet"):
    path = tmp_path / name
    frame.to_parquet(path)
    return path


def test_exact_seven_calendar_years_includes_leap_day() -> None:
    end = datetime(2026, 7, 28, tzinfo=UTC)
    start = exact_calendar_start(end)
    assert start == datetime(2019, 7, 28, tzinfo=UTC)
    assert has_exact_calendar_years(start, end)


def test_one_day_short_is_not_seven_years() -> None:
    end = datetime(2026, 7, 28, tzinfo=UTC)
    assert not has_exact_calendar_years(
        datetime(2019, 7, 29, tzinfo=UTC),
        end,
    )


def test_dataset_is_seven_year_eligible_after_calendar_audit(tmp_path) -> None:
    path = _write(_frame("2019-07-28", "2026-07-28"), tmp_path)
    manifest = audit_dataset(path, now=NOW)
    assert manifest.seven_year_eligible is True
    assert manifest.rejection_reason is None
    assert manifest.usable_calendar_days >= 7 * 365
    assert manifest.dataset_hash


def test_non_ohlcv_context_dataset_is_hashed_not_reported_as_error(
    tmp_path,
    monkeypatch,
) -> None:
    import reporting.top_existing_strategies as top_existing

    monkeypatch.setattr(top_existing, "collect_longlist", lambda _root: [])
    monkeypatch.setattr(top_existing, "score_candidates", lambda rows: rows)
    monkeypatch.setattr(
        top_existing,
        "select_top_strategies",
        lambda rows, *, limit: rows[:limit],
    )
    normalized = tmp_path / "data_store" / "normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC"),
    ).to_parquet(normalized / "macro_context.parquet")

    audit = audit_repository(tmp_path, now=NOW)

    assert audit["dataset_manifest_errors"] == []
    assert audit["summary"]["non_ohlcv_datasets_inventoried"] == 1
    context = audit["non_ohlcv_dataset_manifests"][0]
    assert context["dataset_hash"]
    assert context["classification"] == (
        "NON_OHLCV_CONTEXT_DATASET_NOT_BAR_RANKABLE"
    )


def test_indicator_warmup_reduces_usable_history(tmp_path) -> None:
    path = _write(_frame("2019-07-28", "2026-07-28"), tmp_path)
    manifest = audit_dataset(path, now=NOW, warmup_bars=200)
    assert manifest.seven_year_eligible is False
    assert manifest.rejection_reason == "INSUFFICIENT_MARKET_HISTORY"
    assert manifest.warmup_bars == 200


def test_verified_source_transition_is_preserved(tmp_path) -> None:
    path = _write(_frame("2018-01-01", "2026-07-28"), tmp_path)
    atomic_write_json(
        path.with_suffix(f"{path.suffix}.manifest.json"),
        {
            "provider": "native_then_archive",
            "exchange": "BITVAVO",
            "source_transition_validated": True,
            "source_segments": [
                {
                    "provider": "archive",
                    "exchange": "BITVAVO",
                    "market_identity": "BTC-EUR",
                    "start": "2018-01-01T00:00:00Z",
                    "end": "2020-01-01T00:00:00Z",
                },
                {
                    "provider": "native",
                    "exchange": "BITVAVO",
                    "market_identity": "BTC-EUR",
                    "start": "2019-12-01T00:00:00Z",
                    "end": "2026-07-28T00:00:00Z",
                    "overlap_check": "PASSED",
                },
            ],
        },
    )
    manifest = audit_dataset(path, now=NOW)
    assert manifest.seven_year_eligible
    assert len(manifest.source_segments) == 2
    assert "SOURCE_TRANSITION_UNVERIFIED" not in manifest.quality_reasons


def test_unverified_composite_source_fails_closed(tmp_path) -> None:
    path = _write(_frame("2018-01-01", "2026-07-28"), tmp_path)
    atomic_write_json(
        path.with_suffix(f"{path.suffix}.manifest.json"),
        {
            "provider": "mixed",
            "exchange": "MIXED",
            "source_segments": [
                {
                    "provider": "source_a",
                    "market_identity": "BTC-EUR",
                },
                {
                    "provider": "source_b",
                    "market_identity": "BTC-EUR",
                },
            ],
        },
    )
    manifest = audit_dataset(path, now=NOW)
    assert not manifest.seven_year_eligible
    assert manifest.rejection_reason == "DATA_QUALITY_FAILED"
    assert "SOURCE_TRANSITION_UNVERIFIED" in manifest.quality_reasons


def test_source_market_identity_mismatch_fails_closed(tmp_path) -> None:
    path = _write(_frame("2018-01-01", "2026-07-28"), tmp_path)
    atomic_write_json(
        path.with_suffix(f"{path.suffix}.manifest.json"),
        {
            "provider": "archive",
            "exchange": "ARCHIVE",
            "source_segments": [
                {
                    "provider": "archive",
                    "market_identity": "BTC-USDT",
                }
            ],
        },
    )
    manifest = audit_dataset(path, now=NOW)
    assert manifest.rejection_reason == "DATA_QUALITY_FAILED"
    assert "SOURCE_MARKET_IDENTITY_MISMATCH" in manifest.quality_reasons


def test_gap_quality_failure_is_explicit(tmp_path) -> None:
    frame = _frame("2019-07-28", "2026-07-28")
    frame = frame.drop(frame.index[100:500])
    manifest = audit_dataset(_write(frame, tmp_path), now=NOW)
    assert manifest.missing_bar_count >= 400
    assert manifest.largest_gap_bars >= 400
    assert manifest.rejection_reason == "DATA_QUALITY_FAILED"
    assert "EXCESSIVE_GAPS" in manifest.quality_reasons


def test_duplicate_candles_are_not_silently_deduplicated(tmp_path) -> None:
    frame = _frame("2019-07-28", "2026-07-28")
    duplicated = pd.concat([frame, frame.iloc[[10]]]).sort_index()
    manifest = audit_dataset(_write(duplicated, tmp_path), now=NOW)
    assert manifest.duplicate_count == 2
    assert manifest.rejection_reason == "DATA_QUALITY_FAILED"


def test_invalid_ohlc_fails_quality(tmp_path) -> None:
    frame = _frame("2019-07-28", "2026-07-28")
    frame.iloc[20, frame.columns.get_loc("high")] = 50.0
    manifest = audit_dataset(_write(frame, tmp_path), now=NOW)
    assert manifest.invalid_bar_count == 1
    assert manifest.rejection_reason == "DATA_QUALITY_FAILED"


def test_open_candle_is_excluded(tmp_path) -> None:
    frame = _frame("2019-07-28", "2026-07-29")
    manifest = audit_dataset(_write(frame, tmp_path), now=NOW)
    assert manifest.stale_bar_count == 1
    assert manifest.actual_last_timestamp == datetime(2026, 7, 29, tzinfo=UTC)
    assert manifest.evaluation_end == datetime(2026, 7, 28, tzinfo=UTC)


def test_common_window_uses_shortest_overlap(tmp_path) -> None:
    btc = audit_dataset(
        _write(_frame("2018-01-01", "2026-07-28"), tmp_path, "BTC-EUR_1d.parquet"),
        now=NOW,
    )
    eth = audit_dataset(
        _write(_frame("2019-07-28", "2026-07-28"), tmp_path, "ETH-EUR_1d.parquet"),
        now=NOW,
    )
    window = common_window([btc, eth])
    assert window.start == datetime(2019, 7, 28, tzinfo=UTC)
    assert window.end == datetime(2026, 7, 28, tzinfo=UTC)
    assert window.seven_year_eligible is True


def test_multi_timeframe_short_overlap_fails(tmp_path) -> None:
    daily = audit_dataset(
        _write(_frame("2018-01-01", "2026-07-28"), tmp_path, "BTC-EUR_1d.parquet"),
        now=NOW,
    )
    hourly = audit_dataset(
        _write(
            _frame("2020-01-01", "2026-07-28 11:00", frequency="1h"),
            tmp_path,
            "BTC-EUR_1h.parquet",
        ),
        now=NOW,
    )
    window = common_window([daily, hourly])
    assert window.seven_year_eligible is False
    assert "COMMON_WINDOW_SHORTER_THAN_REQUIRED" in window.rejection_reasons


def test_timeframe_specific_trade_minimums() -> None:
    assert minimum_trades_for_timeframe("1d") == 35
    assert minimum_trades_for_timeframe("4h") == 70
    assert minimum_trades_for_timeframe("1h") == 120
    assert minimum_trades_for_timeframe("15m") == 250
    assert minimum_trades_for_timeframe("5m") == 400


def test_short_history_never_promotes(tmp_path) -> None:
    manifest = audit_dataset(
        _write(_frame("2021-01-01", "2026-07-28"), tmp_path),
        now=NOW,
    )
    status, reasons = strategy_history_status(
        manifests=[manifest],
        timeframe="1d",
        trade_count=100,
        rerun_complete=True,
        causality_passed=True,
        stress_passed=True,
        walk_forward_passed=True,
        stability_passed=True,
    )
    assert status == "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY"
    assert reasons == ("INSUFFICIENT_MARKET_HISTORY",)


def test_complete_candidate_requires_all_deep_validation(tmp_path) -> None:
    manifest = audit_dataset(
        _write(_frame("2018-01-01", "2026-07-28"), tmp_path),
        now=NOW,
    )
    status, reasons = strategy_history_status(
        manifests=[manifest],
        timeframe="1d",
        trade_count=35,
        rerun_complete=True,
        normal_economics_passed=True,
        causality_passed=True,
        stress_passed=True,
        walk_forward_passed=True,
        stability_passed=True,
    )
    assert status == "SEVEN_YEAR_RESEARCH_CANDIDATE"
    assert reasons == ()


def test_negative_normal_economics_are_rejected(tmp_path) -> None:
    manifest = audit_dataset(
        _write(_frame("2018-01-01", "2026-07-28"), tmp_path),
        now=NOW,
    )
    status, reasons = strategy_history_status(
        manifests=[manifest],
        timeframe="1d",
        trade_count=35,
        rerun_complete=True,
        normal_economics_passed=False,
        causality_passed=True,
        stress_passed=True,
        walk_forward_passed=True,
        stability_passed=True,
    )
    assert status == "RESEARCH_REJECTED"
    assert reasons == ("NON_POSITIVE_NORMAL_COST_ECONOMICS",)


def test_history_exclusion_is_persisted_with_dataset_hashes(
    tmp_path,
    isolated_settings,
) -> None:
    processed = tmp_path / "normalized"
    output = tmp_path / "output"
    processed.mkdir()
    for market in ("BTC-EUR", "ETH-EUR"):
        _write(
            _frame("2022-01-01", "2026-07-28"),
            processed,
            f"{market}_1d.parquet",
        )
    paths = isolated_settings.paths.model_copy(
        update={
            "processed_data_dir": processed,
            "output_dir": output,
        }
    )
    settings = isolated_settings.model_copy(update={"paths": paths})

    payload = record_seven_year_history_exclusion(
        settings,
        strategy_id="PORTFOLIO_TEST",
        strategy_dna_hash="a" * 64,
        markets=("BTC-EUR", "ETH-EUR"),
        timeframe="1d",
        warmup_bars=200,
        material_difference_reason="POST_WARMUP_HISTORY_EXCLUSION",
    )

    assert payload["status"] == "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY"
    assert payload["exclusion_proven"] is True
    assert set(payload["dataset_manifests"]) == {"BTC-EUR", "ETH-EUR"}
    assert all(
        item["dataset_hash"] for item in payload["dataset_manifests"].values()
    )
    assert (
        output
        / "research"
        / "seven_year"
        / "runs"
        / "PORTFOLIO_TEST__BTC-EUR-ETH-EUR__1d"
        / "seven_year_result.json"
    ).is_file()


def test_common_window_ranking_keeps_failed_completed_runs(tmp_path) -> None:
    directory = tmp_path / "output" / "research" / "seven_year"
    run_dir = directory / "runs" / "common-failure"
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "seven_year_result.json",
        {
            "strategy_id": "COMMON_FAILURE",
            "strategy_dna_hash": "b" * 64,
            "market": "BTC-EUR",
            "timeframe": "4h",
            "window_kind": "COMMON_WINDOW",
            "status": "FAILED_STRESS",
            "status_reasons": ["STRESSED_COST_ECONOMICS_FAILED"],
            "normal_costs": {
                "metrics": {
                    "net_return": 0.10,
                    "cagr": 0.014,
                    "profit_factor": 1.20,
                    "net_expectancy_r": 0.02,
                    "maximum_drawdown": 0.08,
                    "trade_count": 100,
                }
            },
            "stressed_costs": {
                "metrics": {
                    "net_return": -0.01,
                    "profit_factor": 0.95,
                }
            },
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )

    payload = build_seven_year_rankings(
        tmp_path,
        output_directory=directory,
    )

    common = payload["rankings"]["COMMON_WINDOW"]
    assert len(common) == 1
    assert common[0]["status"] == "FAILED_STRESS"
    assert payload["ranking_method"][
        "common_window_is_not_a_promotion_shortlist"
    ]
    assert (directory / "rankings.json").is_file()
    assert (directory / "rankings.csv").is_file()
    assert (directory / "rankings.html").is_file()


def test_legacy_comparison_writes_json_csv_and_html(tmp_path) -> None:
    directory = tmp_path / "seven_year"
    directory.mkdir()
    atomic_write_json(
        directory / "legacy_top30_gap.json",
        {
            "strategies": [
                {
                    "legacy_rank": 1,
                    "strategy_name": "LEGACY",
                    "strategy_dna_hash": "a" * 64,
                    "strategy_family": "TEST",
                    "assets_universe": ["BTC-EUR"],
                    "timeframe": "1h",
                    "legacy_net_return": 0.10,
                    "legacy_profit_factor": 1.20,
                    "legacy_max_drawdown": 0.05,
                    "new_seven_year_status": "RESEARCH_REJECTED",
                    "new_status_reasons": [
                        "NON_POSITIVE_NORMAL_COST_ECONOMICS"
                    ],
                    "new_net_return": -0.01,
                    "new_profit_factor": 0.95,
                }
            ]
        },
    )

    payload = build_legacy_comparison(directory)

    assert payload["row_count"] == 1
    assert payload["status_counts"] == {"RESEARCH_REJECTED": 1}
    assert (directory / "legacy_vs_seven_year.json").is_file()
    assert (directory / "legacy_vs_seven_year.csv").is_file()
    assert (directory / "legacy_vs_seven_year.html").is_file()
    assert payload["orders_submitted"] == 0


def test_positive_intraday_rerun_queue_is_deduplicated_and_resumable() -> None:
    rows = [
        {
            "strategy_name": "VOL_BTC_EUR_1h_OBV_CMF_CONTINUATION_N1",
            "strategy_dna_hash": "c" * 64,
            "strategy_family": "VOLUME",
            "assets_universe": ["BTC-EUR"],
            "timeframe": "1h",
            "legacy_net_return": 0.10,
            "legacy_profit_factor": 1.2,
            "new_seven_year_status": "QUEUED",
            "new_status_reasons": [],
            "matching_dataset_manifests": [
                {"dataset_hash": "d" * 64},
            ],
        },
        {
            "strategy_name": "VOL_SOL_EUR_4h_DONCHIAN_RVOL_BREAKOUT_N1",
            "strategy_dna_hash": "e" * 64,
            "strategy_family": "VOLUME",
            "assets_universe": ["SOL-EUR"],
            "timeframe": "4h",
            "legacy_net_return": 0.20,
            "legacy_profit_factor": 1.3,
            "new_seven_year_status": (
                "DEGRADED_SHORT_HISTORY_RESEARCH_ONLY"
            ),
            "new_status_reasons": ["INSUFFICIENT_MARKET_HISTORY"],
            "matching_dataset_manifests": [
                {"dataset_hash": "f" * 64},
            ],
        },
        {
            "strategy_name": "NEGATIVE",
            "strategy_dna_hash": "0" * 64,
            "assets_universe": ["ETH-EUR"],
            "timeframe": "1h",
            "legacy_net_return": -0.01,
            "new_seven_year_status": "QUEUED",
        },
    ]

    payload = build_positive_timeframe_rerun_queue(rows)

    assert payload["job_count"] == 2
    assert payload["queued_count"] == 1
    assert payload["final_status_count"] == 1
    assert payload["timeframe_counts"] == {"1h": 1, "4h": 1}
    assert payload["deduplicated"] is True
    assert payload["resumable"] is True
    assert len({row["job_id"] for row in payload["jobs"]}) == 2
    assert payload["orders_submitted"] == 0


def test_rolling_and_oos_metrics_are_explicit() -> None:
    index = pd.date_range("2024-01-01", periods=800, freq="1D", tz="UTC")
    returns = pd.Series([0.002, -0.0005] * 400, index=index)
    equity = pd.Series(
        10_000.0 * (1.0 + returns).cumprod(),
        index=index,
    )
    frame, summary = _rolling_twelve_month_metrics(equity)
    folds = (
        SimpleNamespace(
            trade_count=10,
            net_expectancy_r=0.10,
            profit_factor=1.2,
            net_pnl_eur=20.0,
        ),
        SimpleNamespace(
            trade_count=20,
            net_expectancy_r=0.20,
            profit_factor=1.4,
            net_pnl_eur=40.0,
        ),
    )
    result = SimpleNamespace(
        mode="anchored",
        folds=folds,
        positive_folds=2,
        valid=True,
    )
    oos = _walk_forward_oos_summary(
        result,
        full_sample_expectancy_r=0.25,
    )

    assert frame["rolling_12m_return"].dropna().iloc[-1] > 0
    assert summary["return"]["latest"] > 0
    assert summary["period_profit_factor"]["latest"] > 1
    assert oos["oos_trade_count"] == 30
    assert oos["oos_weighted_expectancy_r"] == pytest.approx(1 / 6)
    assert oos["walk_forward_efficiency"] == pytest.approx(2 / 3)


def test_regime_breakdown_contains_all_required_analytic_labels() -> None:
    index = pd.date_range("2025-01-01", periods=180, freq="1D", tz="UTC")
    close = pd.Series(
        100.0 + pd.Series(range(len(index)), dtype=float).to_numpy() * 0.2,
        index=index,
    )
    features = pd.DataFrame(
        {
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "ema_50": close.ewm(span=50, adjust=False).mean(),
            "bull_regime": True,
            "bear_regime": False,
        },
        index=index,
    )
    equity = pd.Series(
        10_000.0 * (1.0002 ** pd.Series(range(len(index))).to_numpy()),
        index=index,
    )
    rows = _analytic_regime_returns(features, equity)

    assert {row["regime"] for row in rows} == {
        "BULL_MARKET",
        "BEAR_MARKET",
        "SIDEWAYS_MARKET",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
        "LIQUIDITY_STRESS",
        "CRASH_PERIOD",
        "RECOVERY_PERIOD",
    }
    assert all(
        row["label_usage"]
        == "RETROSPECTIVE_ANALYTIC_ONLY_NOT_TRADABLE_INPUT"
        for row in rows
    )


@pytest.mark.asyncio
async def test_canonical_downloader_resume_backfills_prefix(
    tmp_path,
    isolated_settings,
) -> None:
    class Provider:
        name = "bitvavo"

        def __init__(self) -> None:
            self.calls: list[tuple[datetime, datetime]] = []

        async def fetch_candles(
            self,
            market: str,
            timeframe: str,
            *,
            start: datetime,
            end: datetime,
        ) -> pd.DataFrame:
            del market, timeframe
            self.calls.append((start, end))
            source = _frame("2019-01-01", "2019-01-05")
            return source.loc[
                (source.index >= pd.Timestamp(start))
                & (source.index <= pd.Timestamp(end))
            ]

    paths = isolated_settings.paths.model_copy(
        update={"processed_data_dir": tmp_path}
    )
    settings = isolated_settings.model_copy(update={"paths": paths})
    existing = _frame("2019-01-03", "2019-01-05")
    save_ohlcv(
        existing,
        tmp_path / "BTC-EUR_1d.parquet",
        market="BTC-EUR",
        timeframe="1d",
        now=datetime(2019, 1, 6, tzinfo=UTC),
    )
    provider = Provider()
    result = await CanonicalDownloader(settings).download_one(
        market="BTC-EUR",
        timeframe="1d",
        provider=provider,
        start=datetime(2019, 1, 1, tzinfo=UTC),
        end=datetime(2019, 1, 6, tzinfo=UTC),
        resume=True,
    )
    assert provider.calls == [
        (
            datetime(2019, 1, 1, tzinfo=UTC),
            datetime(2019, 1, 3, tzinfo=UTC),
        )
    ]
    assert result.start == datetime(2019, 1, 1, tzinfo=UTC)
    assert result.end == datetime(2019, 1, 5, tzinfo=UTC)
