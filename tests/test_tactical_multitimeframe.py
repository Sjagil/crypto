from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import core.active_trading as active_trading
from config.settings import PathSettings, Settings
from core.active_trading import (
    _family_canary_authority,
    _freshness,
    _latest_intelligence_context,
    _load_scan_frames,
    _public_close_return,
    _public_latest,
    _regime_policy,
    _scan_tactical_opportunities,
    _stablecoin_liquidity_context,
    _weighted_timeframe_assessment,
    build_lower_timeframe_candidate_queue,
    build_proactive_allocation_plan,
    build_rotation_ranking,
)
from research.tactical_multitimeframe import (
    TacticalMultiTimeframeStrategy,
    tactical_strategy_specs,
)


def test_scan_frames_replace_stale_2h_cache_with_causal_1h_resample(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 8, 7, 17, tzinfo=UTC)

    fresh_index = pd.date_range(
        "2026-08-03T13:00:00Z",
        "2026-08-07T16:00:00Z",
        freq="1h",
    )
    fresh = pd.DataFrame(
        {
            "timestamp": fresh_index,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    )
    fresh.to_parquet(
        settings.paths.processed_data_dir / "BTC-EUR_1h.parquet"
    )
    stale = fresh.iloc[:10].copy()
    stale["timestamp"] = pd.date_range(
        "2026-08-01T00:00:00Z", periods=10, freq="2h"
    )
    stale.to_parquet(
        settings.paths.processed_data_dir / "BTC-EUR_2h.parquet"
    )

    frames, failures = _load_scan_frames(
        settings,
        ("BTC-EUR",),
        now=now,
        maximum_rows=1_500,
    )

    two_hour = frames[("BTC-EUR", "2h")]
    assert two_hour.index[-1] == pd.Timestamp("2026-08-07T14:00:00Z")
    assert two_hour.attrs["data_provenance"]["source_type"] == (
        "CAUSAL_CLOSED_CANDLE_RESAMPLE"
    )
    assert ("BTC-EUR", "2h") not in failures


def test_scan_status_never_implies_live_authority_without_delegation() -> None:
    status, reason = active_trading._scan_runtime_status(
        orders_submitted=0,
        actionable_count=3,
        early_move_count=0,
        execution_delegated=False,
    )

    assert status == "ACTIONABLE_SIGNAL_OBSERVED_NO_EXECUTION"
    assert reason == "TACTICAL_SIGNAL_HAS_NO_EXECUTION_DELEGATION"
    assert "LIVE_ACTIVE" not in status


def test_scan_status_reports_only_canonical_submitted_order_as_submitted() -> None:
    status, reason = active_trading._scan_runtime_status(
        orders_submitted=1,
        actionable_count=3,
        early_move_count=0,
        execution_delegated=True,
    )

    assert status == "CANONICAL_ORDER_SUBMITTED"
    assert reason == "NATURAL_APPROVED_ENTRY_SUBMITTED"


def test_legacy_live_active_artifact_is_normalized_fail_closed_on_read(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    path = tmp_path / "output" / "active_trading" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "LIVE_ACTIVE_ENTRY_AVAILABLE",
                "top_5_actionable": [{"market": "LINK-EUR"}],
                "execution_delegated_to_canonical_live_engine": False,
                "orders_submitted": 0,
            }
        ),
        encoding="utf-8",
    )

    status = active_trading.active_trading_status(settings)

    assert status["status"] == "ACTIONABLE_SIGNAL_OBSERVED_NO_EXECUTION"
    assert status["raw_writer_status"] == "LIVE_ACTIVE_ENTRY_AVAILABLE"
    assert status["status_semantics_normalized_on_read"] is True


@pytest.mark.asyncio
async def test_fast_frame_recovery_overlays_only_closed_real_candles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    settings.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 8, 8, 12, 7, tzinfo=UTC)
    stale = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-08-07T00:00:00Z", periods=8, freq="15min"
            ),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    )
    stale.to_parquet(
        settings.paths.processed_data_dir / "BTC-EUR_15m.parquet"
    )

    class _Loader:
        async def download_ohlcv(self, **kwargs: object) -> list[object]:
            if kwargs.get("timeframe") == "1h":
                raise RuntimeError("TEST_1H_UNAVAILABLE")
            rows: list[object] = []
            for timestamp, closed in (
                (datetime(2026, 8, 8, 11, 30, tzinfo=UTC), True),
                (datetime(2026, 8, 8, 11, 45, tzinfo=UTC), True),
                (datetime(2026, 8, 8, 12, 0, tzinfo=UTC), False),
            ):
                rows.append(
                    SimpleNamespace(
                        timestamp=timestamp,
                        closed=closed,
                        values={
                            "open": 101.0,
                            "high": 102.0,
                            "low": 100.0,
                            "close": 101.5,
                            "volume": 11.0,
                        },
                    )
                )
            return rows

    monkeypatch.setattr(active_trading, "DataLoader", lambda _settings: _Loader())
    recovered, report = await active_trading._recover_recent_fast_frames(
        settings,
        ("BTC-EUR",),
        frames={},
        now=now,
        maximum_rows=1_500,
    )

    frame = recovered[("BTC-EUR", "15m")]
    assert frame.index[-1] == pd.Timestamp("2026-08-08T11:45:00Z")
    assert pd.Timestamp("2026-08-08T12:00:00Z") not in frame.index
    assert report["status"] == "PARTIAL"  # 1h was intentionally unavailable.
    assert report["synthetic_data_used"] is False
    assert report["canonical_files_mutated"] is False


@pytest.mark.asyncio
async def test_fast_frame_recovery_matches_hard_freshness_slo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    now = datetime(2026, 8, 8, 12, 11, tzinfo=UTC)

    def frame(index: str, *, frequency: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            },
            index=pd.date_range(index, periods=1, freq=frequency),
        )

    stale_15m = frame("2026-08-08T11:30:00Z", frequency="15min")
    fresh_1h = frame("2026-08-08T11:00:00Z", frequency="1h")
    realtime_path = (
        settings.paths.output_dir / "operations" / "realtime_candles.json"
    )
    realtime_path.parent.mkdir(parents=True, exist_ok=True)
    realtime_path.write_text(
        json.dumps(
            {
                "closed_candles": {
                    "BTC-EUR:15m": [
                        {
                            "timestamp": "2026-08-08T11:45:00Z",
                            "open": 101.0,
                            "high": 102.0,
                            "low": 100.0,
                            "close": 101.5,
                            "volume": 11.0,
                            "closed": True,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    class _NoRestLoader:
        async def download_ohlcv(self, **_kwargs: object) -> list[object]:
            raise AssertionError("fresh websocket recovery must avoid REST")

    monkeypatch.setattr(
        active_trading,
        "DataLoader",
        lambda _settings: _NoRestLoader(),
    )
    recovered, report = await active_trading._recover_recent_fast_frames(
        settings,
        ("BTC-EUR",),
        frames={
            ("BTC-EUR", "15m"): stale_15m,
            ("BTC-EUR", "1h"): fresh_1h,
        },
        now=now,
        maximum_rows=1_500,
    )

    assert report["requested"] == 1
    assert report["status"] == "READY"
    assert report["rows"]["BTC-EUR:15m"]["status"] == (
        "RECOVERED_FROM_WEBSOCKET"
    )
    assert recovered[("BTC-EUR", "15m")].index[-1] == pd.Timestamp(
        "2026-08-08T11:45:00Z"
    )


@pytest.mark.asyncio
async def test_macro_refresh_appends_btc_and_eth_gex_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})

    class _Loader:
        def __init__(self) -> None:
            self.gex_calls: list[tuple[str, bool]] = []

        async def download_macro_series(self, **_kwargs: object) -> list[object]:
            return []

        async def download_cmc_rankings(self, **_kwargs: object) -> list[object]:
            return []

        async def download_derivatives_context(
            self,
            **_kwargs: object,
        ) -> list[object]:
            return []

        async def download_gex_context(
            self,
            *,
            underlying: str,
            persist: bool,
        ) -> dict[str, object]:
            self.gex_calls.append((underlying, persist))
            return {
                "provider": "deribit",
                "available_at": "2026-08-01T12:00:00+00:00",
            }

    loader = _Loader()
    monkeypatch.setattr(active_trading, "DataLoader", lambda _settings: loader)

    async def no_intelligence(
        _settings: Settings,
        *,
        now: datetime,
    ) -> dict[str, object]:
        return {"observed_at": now.isoformat(), "status": "EMPTY"}

    monkeypatch.setattr(
        active_trading,
        "_refresh_scraper_intelligence",
        no_intelligence,
    )

    result = await active_trading.refresh_public_macro_context(settings)

    assert sorted(loader.gex_calls) == [("BTC", True), ("ETH", True)]
    assert result["sources"]["deribit_btc_gex"]["status"] == "READY"
    assert result["sources"]["deribit_eth_gex"][
        "point_in_time_history_appended"
    ] is True


def test_proactive_allocation_is_bounded_and_never_grants_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    monkeypatch.setattr(
        active_trading,
        "build_capital_utilization",
        lambda _settings: {
            "account_equity_eur": 500.0,
            "eur_cash": 75.0,
            "material_inventory": [
                {"market": "TAO-EUR", "estimated_value_eur": 425.0}
            ],
        },
    )

    payload = build_proactive_allocation_plan(
        settings,
        rotation=[
            {
                "market": "ETH-EUR",
                "decision": "FAVOUR",
                "rotation_score": 90.0,
            }
        ],
        opportunities=[
            {
                "market": "ETH-EUR",
                "status": "NEAR_ENTRY",
                "score": 75.0,
                "strategy": "TACTICAL_15M_LIQUIDITY_SWEEP",
                "timeframe": "15m",
                "live_authority_granted": False,
            }
        ],
        regime="MACRO_RISK_OFF",
        live_markets={"ETH-EUR", "TAO-EUR"},
    )

    rows = {row["asset"]: row for row in payload["rows"]}
    assert rows["ETH-EUR"]["target_weight"] <= 0.20
    assert rows["ETH-EUR"]["action"] == (
        "WAIT_FOR_DNA_APPROVAL_AND_NATURAL_SIGNAL"
    )
    assert rows["TAO-EUR"]["action"] == (
        "REDUCE_EXTERNAL_INVENTORY_REVIEW"
    )
    assert rows["TAO-EUR"]["target_weight"] == pytest.approx(0.20)
    assert rows["ETH-EUR"]["target_weight"] == pytest.approx(0.20)
    assert rows["EUR"]["target_weight"] == pytest.approx(0.60)
    assert (
        payload["external_inventory_retention_is_strategy_authority"]
        is False
    )
    assert payload["does_not_expand_execution_authority"] is True
    assert payload["external_inventory_not_claimed"] is True
    assert payload["orders_generated"] == 0
    assert payload["orders_submitted"] == 0


def test_family_canary_route_does_not_claim_exact_dna_authority(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "live_playbook_authority.json").write_text(
        json.dumps(
            {
                "active": True,
                "maximum_order_eur": "25",
                "approved_playbooks": [
                    {
                        "active": True,
                        "playbook_id": "TREND_PULLBACK_V1",
                        "markets": ["ETH-EUR"],
                        "evidence_multiplier": "0.40",
                        "strategy_role": "EXPERIMENTAL_CANARY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    routed = _family_canary_authority(
        settings,
        family="TREND_PULLBACK",
        market="ETH-EUR",
        entry_timeframe="1h",
    )
    excluded_market = _family_canary_authority(
        settings,
        family="TREND_PULLBACK",
        market="SOL-EUR",
        entry_timeframe="1h",
    )
    context_only = _family_canary_authority(
        settings,
        family="TREND_PULLBACK",
        market="ETH-EUR",
        entry_timeframe="2h",
    )

    assert routed["available"] is True
    assert routed["playbook_ids"] == ["TREND_PULLBACK_V1"]
    assert routed["maximum_effective_order_eur"] == pytest.approx(10.0)
    assert routed["exact_dna_authority_granted"] is False
    assert excluded_market["available"] is False
    assert context_only["available"] is False
    assert context_only["entry_timeframe_route_eligible"] is False


def test_nonfinite_tactical_trigger_is_fail_closed_without_breaking_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    features: pd.DataFrame,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    selected = next(
        row
        for row in tactical_strategy_specs()
        if row.strategy_id == "TACTICAL_1H_TREND_PULLBACK"
    )
    monkeypatch.setattr(
        active_trading,
        "tactical_strategy_specs",
        lambda: [selected],
    )
    monkeypatch.setattr(
        active_trading,
        "_trigger",
        lambda *_args, **_kwargs: (float("nan"), "TEST_NONFINITE"),
    )
    frames = {
        ("BTC-EUR", timeframe): features
        for timeframe in ("15m", "1h", "2h", "4h", "1d", "1W")
    }

    rows, _evaluations = _scan_tactical_opportunities(
        settings,
        frames,
        frames,
        ["BTC-EUR"],
        macro={"regime": "RECOVERY", "confidence": 0.7},
        rotation=[],
    )

    assert len(rows) == 1
    assert rows[0]["trigger"] is None
    assert rows[0]["trigger_data_valid"] is False
    assert rows[0]["live_authority_granted"] is False
    assert rows[0]["reason_not_yet_entered"] == (
        "INVALID_OR_MISSING_CAUSAL_TRIGGER"
    )


def test_catalogue_has_independent_15m_1h_and_2h_families() -> None:
    specs = tactical_strategy_specs()
    fifteen_minute = [spec for spec in specs if spec.timeframe == "15m"]
    one_hour = [spec for spec in specs if spec.timeframe == "1h"]
    two_hour = [spec for spec in specs if spec.timeframe == "2h"]

    assert len(fifteen_minute) == 10
    assert len(one_hour) == 14
    assert len(two_hour) == 10
    assert len({spec.family for spec in one_hour}) == 14
    assert len({spec.family for spec in two_hour}) == 10
    assert len({spec.dna_hash for spec in specs}) == len(specs)
    assert all(
        spec.confirmation_timeframe == ("1h" if spec.timeframe == "15m" else "4h")
        for spec in specs
    )
    assert all(
        spec.regime_timeframe == ("4h" if spec.timeframe == "15m" else "1d")
        for spec in specs
    )


def test_every_tactical_strategy_is_bounded_and_has_no_live_authority(
    features: pd.DataFrame,
) -> None:
    selected = features.copy()
    selected["htf_4h_trend_bullish"] = True
    selected["htf_1d_trend_bullish"] = True

    for spec in tactical_strategy_specs():
        output = TacticalMultiTimeframeStrategy(spec).generate(selected)
        assert output.entry.dtype == bool
        assert output.exit.dtype == bool
        assert output.avoid.dtype == bool
        assert output.size_multiplier.between(0.0, 1.0).all()
        assert output.metadata["strategy_dna_hash"] == spec.dna_hash
        assert output.metadata["next_open_execution"] is True
        assert output.metadata.get("live_authority_granted") is None


def test_bearish_daily_context_is_soft_for_valid_intraday_trigger(
    features: pd.DataFrame,
) -> None:
    selected = features.copy()
    selected["htf_4h_trend_bullish"] = False
    selected["htf_1d_trend_bullish"] = False
    selected["ema_200"] = 90.0
    selected["ema_50"] = 95.0
    selected["ema_20"] = 100.0
    selected["rsi_14"] = 55.0
    selected.loc[selected.index[-2], "close"] = 99.0
    selected.loc[selected.index[-1], "close"] = 101.0
    spec = next(
        row
        for row in tactical_strategy_specs()
        if row.strategy_id == "TACTICAL_1H_TREND_PULLBACK"
    )

    output = TacticalMultiTimeframeStrategy(spec).generate(selected)

    assert output.entry.iloc[-1]
    assert not output.avoid.iloc[-1]
    assert output.metadata["higher_timeframe_context_is_soft"] is True
    assert output.metadata["standalone_1d_1w_veto"] is False


def test_weighted_timeframe_score_routes_bearish_daily_rally_as_countertrend() -> None:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for timeframe in ("15m", "1h", "2h", "4h", "1d", "1W"):
        periods = 90
        rising = timeframe in {"15m", "1h", "2h", "4h"}
        close = pd.Series(
            [100.0 + (index if rising else -index) for index in range(periods)],
            dtype=float,
        )
        frames[("BTC-EUR", timeframe)] = pd.DataFrame(
            {
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": pd.Series([100.0 + index for index in range(periods)]),
            }
        )

    result = _weighted_timeframe_assessment(frames, "BTC-EUR")

    assert result["fast_score"] > 0.35
    assert result["slow_score"] < -0.20
    assert result["trade_type"] == "COUNTERTREND_LONG"
    assert result["score"] >= result["entry_threshold"]
    assert result["hard_blocked_by_1d_or_1w"] is False
    assert result["weights"]["1d"] + result["weights"]["1W"] == pytest.approx(0.10)


def test_strategy_directional_gate_uses_fast_structure_not_daily_veto() -> None:
    gate = active_trading._strategy_directional_gate(
        "TREND_PULLBACK",
        {
            "score": 0.18,
            "fast_score": 0.31,
            "entry_threshold": 0.40,
        },
    )

    assert gate["approved"] is True
    assert gate["effective_threshold"] == pytest.approx(0.15)
    assert gate["higher_timeframes_are_soft"] is True


def test_reversal_trigger_can_route_near_neutral_but_not_deep_bear() -> None:
    recovery = active_trading._strategy_directional_gate(
        "FAILED_BREAKOUT_REVERSAL",
        {
            "score": -0.14,
            "fast_score": -0.12,
            "entry_threshold": 0.40,
        },
    )
    falling_knife = active_trading._strategy_directional_gate(
        "FAILED_BREAKOUT_REVERSAL",
        {
            "score": -0.64,
            "fast_score": -0.61,
            "entry_threshold": 0.40,
        },
    )

    assert recovery["approved"] is True
    assert falling_knife["approved"] is False


def test_future_feature_change_does_not_change_prior_tactical_signal(
    features: pd.DataFrame,
) -> None:
    selected = features.copy()
    selected["htf_4h_trend_bullish"] = True
    selected["htf_1d_trend_bullish"] = True
    revised = selected.copy()
    revised.loc[revised.index[-1], "close"] *= 1.50
    revised.loc[revised.index[-1], "volume"] *= 20.0

    for spec in tactical_strategy_specs():
        strategy = TacticalMultiTimeframeStrategy(spec)
        baseline = strategy.generate(selected)
        changed = strategy.generate(revised)
        pd.testing.assert_series_equal(
            baseline.entry.iloc[:-1],
            changed.entry.iloc[:-1],
        )
        pd.testing.assert_series_equal(
            baseline.exit.iloc[:-1],
            changed.exit.iloc[:-1],
        )


def test_macro_risk_off_reduces_recovery_but_shadows_alt_breakout() -> None:
    recovery = _regime_policy(
        "MACRO_RISK_OFF",
        "LIQUIDITY_SWEEP_RECOVERY",
        "ETH-EUR",
    )
    alt_breakout = _regime_policy(
        "MACRO_RISK_OFF",
        "DONCHIAN_BREAKOUT",
        "TAO-EUR",
    )

    assert recovery[0] == "REDUCE"
    assert 0.0 < recovery[1] < 1.0
    assert alt_breakout[0] == "REDUCE"
    assert alt_breakout[1] == 0.65


def test_public_macro_latest_uses_timestamp_not_provider_order() -> None:
    refreshed = {
        "sources": {
            "fear_and_greed": {
                "records": [
                    {
                        "available_at": "2026-07-30T00:00:00Z",
                        "values": {"fear_greed": 30},
                    },
                    {
                        "available_at": "2018-02-01T00:00:00Z",
                        "values": {"fear_greed": 10},
                    },
                ]
            }
        }
    }

    latest = _public_latest(refreshed, "fear_and_greed")

    assert latest["values"]["fear_greed"] == 30


def test_public_macro_latest_breaks_availability_ties_by_observation_time() -> None:
    refreshed = {
        "sources": {
            "eodhd_ndx": {
                "records": [
                    {
                        "available_at": "2026-07-31T10:00:00Z",
                        "timestamp": "2026-07-29T00:00:00Z",
                        "values": {"close": 100.0},
                    },
                    {
                        "available_at": "2026-07-31T10:00:00Z",
                        "timestamp": "2026-07-30T00:00:00Z",
                        "values": {"close": 105.0},
                    },
                ]
            }
        }
    }

    latest = _public_latest(refreshed, "eodhd_ndx")

    assert latest["values"]["close"] == 105.0


def test_public_equity_return_uses_timestamp_order() -> None:
    refreshed = {
        "sources": {
            "eodhd_spx": {
                "records": [
                    {
                        "timestamp": f"2026-07-{day:02d}T00:00:00Z",
                        "values": {"close": float(100 + day)},
                    }
                    for day in range(1, 8)
                ]
            }
        }
    }

    result = _public_close_return(refreshed, "eodhd_spx", periods=5)

    assert result == (107.0 / 102.0) - 1.0


def test_future_macro_observation_is_not_fresh() -> None:
    now = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)

    result = _freshness(
        "2026-07-31T10:00:01Z",
        maximum_age_hours=1.0,
        now=now,
    )

    assert result["fresh"] is False
    assert result["from_future"] is True
    assert result["freshness"] == "FUTURE"


def test_scraper_context_excludes_self_test_and_summarizes_real_events(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    target = settings.paths.intelligence_dir / "crypto_intelligence.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)
    pd.DataFrame(
        [
            {
                "source": "Self Test",
                "url": "https://example.test/btc",
                "usable_at": "2026-07-31T09:00:00Z",
                "categories": json.dumps(["exchange_risk"]),
                "relevance_score": 1.0,
                "impact_score": 1.0,
                "sentiment_score": -1.0,
            },
            {
                "source": "Kraken",
                "url": "https://example.org/real",
                "usable_at": "2026-07-31T09:30:00Z",
                "categories": json.dumps(["exchange_risk"]),
                "relevance_score": 0.8,
                "impact_score": 0.75,
                "sentiment_score": -0.5,
            },
        ]
    ).to_parquet(target, index=False)

    result = _latest_intelligence_context(settings, now=now)

    assert result["provider"] == "multi_source_scraper"
    assert result["values"]["event_count"] == 1
    assert result["values"]["source_count"] == 1
    assert abs(result["values"]["event_risk"] - 0.6) < 1e-12


def test_stablecoin_liquidity_tracks_usdt_usdc_and_ignores_future_rows(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    context = settings.paths.context_data_dir
    context.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    ranking_rows = []
    for symbol, earlier, current in (
        ("USDT", 100.0, 90.0),
        ("USDC", 50.0, 45.0),
    ):
        ranking_rows.extend(
            [
                {
                    "provider": "coinmarketcap",
                    "available_at": "2026-07-31T06:00:00Z",
                    "observed_at": "2026-07-31T06:00:00Z",
                    "symbol": symbol,
                    "cmc_rank": 3,
                    "market_cap": earlier,
                    "circulating_supply": earlier,
                    "volume_24h": 10.0,
                },
                {
                    "provider": "coinmarketcap",
                    "available_at": "2026-07-31T12:00:00Z",
                    "observed_at": "2026-07-31T12:00:00Z",
                    "symbol": symbol,
                    "cmc_rank": 3,
                    "market_cap": current,
                    "circulating_supply": current,
                    "volume_24h": 10.0,
                },
                {
                    "provider": "coinmarketcap",
                    "available_at": "2026-07-31T13:00:00Z",
                    "observed_at": "2026-07-31T13:00:00Z",
                    "symbol": symbol,
                    "cmc_rank": 3,
                    "market_cap": 999.0,
                    "circulating_supply": 999.0,
                    "volume_24h": 10.0,
                },
            ]
        )
    pd.DataFrame(ranking_rows).to_parquet(
        context / "coinmarketcap_rankings.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "available_at": "2026-07-31T11:59:00Z",
                "observation_time": "2026-07-30T00:00:00Z",
                "stablecoin_market_cap": 300.0,
            },
            {
                "available_at": "2026-07-31T11:59:00Z",
                "observation_time": "2026-07-31T00:00:00Z",
                "stablecoin_market_cap": 290.0,
            },
        ]
    ).to_parquet(context / "defillama_stablecoins.parquet", index=False)

    result = _stablecoin_liquidity_context(
        settings,
        refreshed=None,
        now=now,
    )

    assert result["state"] == "DRAINING"
    assert result["risk_multiplier"] == 0.75
    assert result["usdt"]["market_cap_eur"] == 90.0
    assert result["usdc"]["market_cap_eur"] == 45.0
    assert result["combined_market_cap_change_6h"] == pytest.approx(-0.10)
    assert result["aggregate"]["change_1d"] == pytest.approx(290 / 300 - 1)
    assert result["aggregate"]["backtest_safe"] is False
    assert result["is_entry_signal"] is False


def test_rotation_ranking_is_cross_sectional_and_order_free(
    ohlcv: pd.DataFrame,
) -> None:
    btc = ohlcv.copy()
    btc.attrs["timeframe"] = "1h"
    eth = ohlcv.copy()
    eth["close"] *= pd.Series(
        range(1, len(eth) + 1),
        index=eth.index,
    ).pow(0.0002)
    eth.attrs["timeframe"] = "1h"
    frames = {
        ("BTC-EUR", "1h"): btc,
        ("ETH-EUR", "1h"): eth,
    }

    rows = build_rotation_ranking(
        frames,
        ("BTC-EUR", "ETH-EUR"),
        regime="MODERATE_RISK_ON",
    )

    assert [row["rank"] for row in rows] == [1, 2]
    assert {row["market"] for row in rows} == {"BTC-EUR", "ETH-EUR"}
    assert all(0.0 <= row["rotation_score"] <= 100.0 for row in rows)


def test_public_latest_handles_missing_source() -> None:
    assert _public_latest({}, "missing") == {}
    assert _public_latest(
        {
            "sources": {
                "empty": {
                    "records": [],
                    "refreshed_at": datetime.now(UTC).isoformat(),
                }
            }
        },
        "empty",
    ) == {}


def test_lower_timeframe_queue_reconciles_paper_and_explicit_live_authority(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    strategies = tmp_path / "output" / "strategies"
    paper = tmp_path / "output" / "paper"
    governance = tmp_path / "output" / "governance"
    for directory in (strategies, paper, governance):
        directory.mkdir(parents=True, exist_ok=True)
    authorized_dna = "a" * 64
    pending_dna = "b" * 64

    def candidate(dna: str, timeframe: str, trades: int) -> dict:
        return {
            "strategy_id": f"SIMPLE_{dna[:4]}",
            "strategy_dna_hash": dna,
            "frozen_candidate_hash": "f" * 64,
            "timeframe": timeframe,
            "economic_hypothesis_family": "TREND+VOLUME_FLOW",
            "markets": ["BTC-EUR"],
            "source": "CONTINUOUS_SIMPLE_LAB_EXACT",
            "metrics": {
                "profit_factor": 1.40 if trades >= 30 else 8.0,
                "stressed_profit_factor": 1.15 if trades >= 30 else None,
                "net_return": 0.20,
                "net_expectancy_r": 0.10,
                "trade_count": trades,
                "maximum_drawdown": 0.10,
                "monte_carlo_p95_drawdown": 0.20,
            },
        }

    (strategies / "frozen_classical_paper_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    candidate(authorized_dna, "1h", 100),
                    candidate(pending_dna, "2h", 8),
                ]
            }
        ),
        encoding="utf-8",
    )
    (paper / "generated_strategy_state.json").write_text(
        json.dumps(
            {
                "evaluations": {
                    authorized_dna: {"status": "EVALUATED"},
                    pending_dna: {"status": "EVALUATED"},
                }
            }
        ),
        encoding="utf-8",
    )
    (governance / "positive_strategy_live_authority.json").write_text(
        json.dumps(
            {
                "approved_candidates": [
                    {"strategy_dna_hash": authorized_dna}
                ]
            }
        ),
        encoding="utf-8",
    )

    queue = build_lower_timeframe_candidate_queue(settings)

    assert queue["candidate_count"] == 2
    assert queue["paper_evaluated_count"] == 2
    assert queue["live_authorized_count"] == 1
    assert queue["pending_operator_dna_approval_count"] == 1
    authorized = next(
        row
        for row in queue["candidates"]
        if row["strategy_dna_hash"] == authorized_dna
    )
    pending = next(
        row
        for row in queue["candidates"]
        if row["strategy_dna_hash"] == pending_dna
    )
    assert authorized["status"] == "LIVE_MICRO_AUTHORIZED"
    assert pending["status"] == (
        "LIVE_MICRO_ELIGIBLE_REQUIRES_OPERATOR_DNA_APPROVAL"
    )
    assert pending["adjusted_profit_factor"] < pending["profit_factor"]
    assert pending["approval_priority"] == "DEFER_WEAK_EVIDENCE"
    assert authorized["approval_readiness_score"] > pending[
        "approval_readiness_score"
    ]
    assert "approve-positive-dna" in pending["approval_command"]
    assert "SMALL_SAMPLE_BELOW_30_TRADES" in pending[
        "capital_scaling_warnings"
    ]
    assert queue["deferred_candidate_count"] == 1
    assert queue["orders_generated"] == 0
    assert queue["orders_submitted"] == 0


def test_lower_timeframe_queue_does_not_treat_missing_monte_carlo_as_zero_risk(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    strategies = tmp_path / "output" / "strategies"
    paper = tmp_path / "output" / "paper"
    governance = tmp_path / "output" / "governance"
    for directory in (strategies, paper, governance):
        directory.mkdir(parents=True, exist_ok=True)
    dna = "c" * 64
    candidate = {
        "strategy_id": "MISSING_MC",
        "strategy_dna_hash": dna,
        "frozen_candidate_hash": "f" * 64,
        "timeframe": "15m",
        "economic_hypothesis_family": "TREND",
        "markets": ["BTC-EUR"],
        "source": "CONTINUOUS_SIMPLE_LAB_EXACT",
        "metrics": {
            "profit_factor": 1.40,
            "stressed_profit_factor": 1.20,
            "net_return": 0.20,
            "net_expectancy_r": 0.10,
            "trade_count": 100,
            "maximum_drawdown": 0.10,
        },
    }
    (strategies / "frozen_classical_paper_candidates.json").write_text(
        json.dumps({"candidates": [candidate]}),
        encoding="utf-8",
    )
    (paper / "generated_strategy_state.json").write_text(
        json.dumps({"evaluations": {dna: {"status": "EVALUATED"}}}),
        encoding="utf-8",
    )
    (governance / "positive_strategy_live_authority.json").write_text(
        json.dumps({"approved_candidates": []}),
        encoding="utf-8",
    )

    row = build_lower_timeframe_candidate_queue(settings)["candidates"][0]

    assert row["micro_live_eligible"] is True
    assert row["monte_carlo_p95_drawdown"] is None
    assert row["approval_priority"] == "DEFER_WEAK_EVIDENCE"
    assert "MONTE_CARLO_EVIDENCE_MISSING" in row[
        "capital_scaling_warnings"
    ]
    assert row["approval_readiness_score"] < 80.0


def test_positive_candidate_waiting_for_next_paper_cycle_is_not_hard_blocked(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    strategies = tmp_path / "output" / "strategies"
    paper = tmp_path / "output" / "paper"
    governance = tmp_path / "output" / "governance"
    for directory in (strategies, paper, governance):
        directory.mkdir(parents=True, exist_ok=True)
    dna = "d" * 64
    (strategies / "frozen_classical_paper_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "strategy_id": "NEW_POSITIVE_1H",
                        "strategy_dna_hash": dna,
                        "frozen_candidate_hash": "f" * 64,
                        "timeframe": "1h",
                        "economic_hypothesis_family": "TREND+VOLUME_FLOW",
                        "markets": ["BTC-EUR"],
                        "metrics": {
                            "profit_factor": 1.25,
                            "net_return": 0.08,
                            "net_expectancy_r": 0.02,
                            "trade_count": 50,
                            "maximum_drawdown": 0.10,
                            "monte_carlo_p95_drawdown": 0.18,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (paper / "generated_strategy_state.json").write_text(
        json.dumps({"evaluations": {}}),
        encoding="utf-8",
    )
    (governance / "positive_strategy_live_authority.json").write_text(
        json.dumps({"approved_candidates": []}),
        encoding="utf-8",
    )

    row = build_lower_timeframe_candidate_queue(settings)["candidates"][0]

    assert row["status"] == "PAPER_PENDING_EVALUATION"
    assert row["paper_eligible"] is True
    assert row["paper_evaluated"] is False
    assert row["micro_live_eligible"] is False
    assert row["hard_blockers"] == []
    assert "PAPER_EVALUATION_PENDING" in row["capital_scaling_warnings"]


@pytest.mark.asyncio
async def test_execution_scan_persists_current_macro_before_delegation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"regime": 0, "opportunity": 0}
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    ).model_copy(update={"paths": PathSettings(project_root=tmp_path)})
    current_macro = {
        "schema_version": "crypto_macro_context_v1",
        "observed_at": "2026-07-31T12:00:00+00:00",
        "available_at": "2026-07-31T12:00:00+00:00",
        "regime": "RECOVERY",
        "confidence": 0.9,
        "features": {"btc_4h_trend_up": True},
        "sources": {},
    }

    monkeypatch.setattr(
        active_trading,
        "active_trading_status",
        lambda _settings: {},
    )
    monkeypatch.setattr(
        active_trading,
        "live_universe_status",
        lambda _settings: {"selected_markets": ["BTC-EUR"]},
    )
    monkeypatch.setattr(
        active_trading,
        "build_tiered_trading_universe",
        lambda *_args, **_kwargs: {
            "shadow_markets": ["BTC-EUR"],
            "live_executable_markets": ["BTC-EUR"],
            "counts": {"shadow": 1, "live_executable": 1},
            "rows": [
                {"market": "BTC-EUR", "highest_tier": "LIVE_EXECUTABLE"}
            ],
        },
    )

    async def fake_refresh(_settings: Settings) -> dict[str, object]:
        return {"status": "REFRESHED"}

    monkeypatch.setattr(
        active_trading,
        "refresh_public_macro_context",
        fake_refresh,
    )
    monkeypatch.setattr(
        active_trading,
        "_load_scan_frames",
        lambda *_args, **_kwargs: ({}, {}),
    )

    async def fake_fast_recovery(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[object, object], dict[str, object]]:
        return {}, {"status": "NOT_REQUIRED"}

    monkeypatch.setattr(
        active_trading,
        "_recover_recent_fast_frames",
        fake_fast_recovery,
    )
    monkeypatch.setattr(
        active_trading,
        "_feature_frames",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        active_trading,
        "build_crypto_macro_snapshot",
        lambda *_args, **_kwargs: current_macro,
    )
    monkeypatch.setattr(
        active_trading,
        "build_rotation_ranking",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        active_trading,
        "_scan_tactical_opportunities",
        lambda *_args, **_kwargs: ([], {"1h": 0, "2h": 0}),
    )
    monkeypatch.setattr(
        active_trading,
        "build_capital_utilization",
        lambda _settings: {},
    )
    monkeypatch.setattr(
        active_trading,
        "build_tao_inventory_policy",
        lambda _settings: {},
    )
    monkeypatch.setattr(
        active_trading,
        "_timeframe_status",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        active_trading,
        "_notify_regime_change",
        lambda *_args, **_kwargs: called.__setitem__(
            "regime", called["regime"] + 1
        ),
    )
    monkeypatch.setattr(
        active_trading,
        "_notify_opportunity_update",
        lambda *_args, **_kwargs: called.__setitem__(
            "opportunity", called["opportunity"] + 1
        ),
    )

    async def fake_execute(
        _settings: Settings,
        **_kwargs: object,
    ) -> dict[str, object]:
        persisted = json.loads(
            (
                tmp_path
                / "output"
                / "active_trading"
                / "macro_crypto.json"
            ).read_text(encoding="utf-8")
        )
        assert persisted == current_macro
        return {
            "orders_generated_this_cycle": 0,
            "orders_submitted_this_cycle": 0,
        }

    monkeypatch.setattr(
        active_trading,
        "execute_generated_strategy_live_once",
        fake_execute,
    )

    result = await active_trading.scan_all(
        settings,
        execute=True,
        notify=False,
    )

    assert result["macro"] == current_macro
    assert result["scan_interval_minutes"] == 5
    assert result["execution_delegated_to_canonical_live_engine"] is True
    assert result["orders_submitted"] == 0
    assert called == {"regime": 0, "opportunity": 0}
    assert result["notifications_enabled"] is False
    assert result["telegram_regime_update"]["delivery_status"] == (
        "SKIPPED_AUDIT_NO_NOTIFY"
    )
    assert result["telegram_opportunity_update"]["delivery_status"] == (
        "SKIPPED_AUDIT_NO_NOTIFY"
    )
    assert set(result["data_health"]["timeframes"]) == {
        "1m",
        "5m",
        "15m",
        "1h",
        "2h",
        "4h",
        "1d",
        "1W",
    }
