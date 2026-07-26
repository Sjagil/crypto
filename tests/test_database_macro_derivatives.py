from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from config.settings import Settings
from data.data_loader import DataLoader
from data.database import Database
from data.derivatives_context import (
    CryptoGEXAnalyzer,
    FundingRateCollector,
    OptionsContract,
    annualize_funding,
)
from research.macro_context import (
    GROUPS,
    MacroContextEngine,
    MacroSourceSpec,
    build_persisted_macro_context,
    cadence_change,
)


def test_database_idempotency_and_rollback(tmp_path) -> None:
    database = Database(sqlite_path=tmp_path / "test.db")
    database.migrate()
    item = {
        "run_id": "same-run",
        "status": "RUNNING",
        "timestamp": datetime.now(UTC),
    }
    database.upsert_records("test_runs", [item])
    database.upsert_records("test_runs", [{**item, "status": "PASSED"}])
    health = database.health()
    assert health["table_counts"]["test_runs"] == 1
    assert health["read_latency_ms"] >= 0
    assert health["write_latency_ms"] >= 0

    database.upsert_records(
        "strategy_signals",
        [
            {
                "external_id": "stable-candle-signal",
                "market": "BTC-EUR",
                "timeframe": "5m",
                "evaluated_at": "2026-07-24T10:00:01Z",
            }
        ],
    )
    database.upsert_records(
        "strategy_signals",
        [
            {
                "external_id": "stable-candle-signal",
                "market": "BTC-EUR",
                "timeframe": "5m",
                "evaluated_at": "2026-07-24T10:00:20Z",
            }
        ],
    )
    signals = database.fetch_records("strategy_signals")
    assert len(signals) == 1
    assert signals[0]["payload"]["first_evaluated_at"] == "2026-07-24T10:00:01Z"
    assert signals[0]["payload"]["evaluation_count"] == 2
    assert database.fetch_records("test_runs")[0]["payload"]["status"] == "PASSED"
    with pytest.raises(RuntimeError):
        with database.transaction() as connection:
            database.upsert_records(
                "orders",
                [{"order_id": "rollback", "status": "OPEN"}],
                connection=connection,
            )
            raise RuntimeError("rollback")
    assert database.health()["table_counts"]["orders"] == 0
    export = database.export("test_runs", tmp_path / "runs.csv")
    assert export.is_file()
    database.close()


@pytest.mark.asyncio
async def test_fred_revisions_preserve_availability_time(
    isolated_settings: Settings,
) -> None:
    providers = isolated_settings.providers.model_copy(
        update={"fred_api_key": SecretStr("unit")}
    )
    settings = isolated_settings.model_copy(update={"providers": providers})

    async def request(method, url, params, headers):
        del method, url, params, headers
        return {
            "observations": [
                {
                    "date": "2024-01-01",
                    "realtime_start": "2024-01-02",
                    "realtime_end": "2024-02-01",
                    "value": "1.0",
                },
                {
                    "date": "2024-01-01",
                    "realtime_start": "2024-02-02",
                    "realtime_end": "9999-12-31",
                    "value": "1.1",
                },
            ]
        }

    records = await DataLoader(settings, requester=request).download_macro_series(
        provider="fred", series="UNIT"
    )
    assert records[0].timestamp == records[1].timestamp
    assert records[0].available_at != records[1].available_at
    assert records[1].available_at == datetime(2024, 2, 2, tzinfo=UTC)


@pytest.mark.asyncio
async def test_sec_uses_acceptance_as_point_in_time_availability(
    isolated_settings: Settings,
) -> None:
    providers = isolated_settings.providers.model_copy(
        update={"sec_user_agent": "unit-test contact@example.test"}
    )
    settings = isolated_settings.model_copy(update={"providers": providers})

    async def request(method, url, params, headers):
        del method, url, params
        assert "User-Agent" in headers
        return {
            "filings": {
                "recent": {
                    "accessionNumber": ["0001-25-000001"],
                    "primaryDocument": ["filing.htm"],
                    "primaryDocDescription": ["Bitcoin custody disclosure"],
                    "acceptanceDateTime": ["2025-01-02T12:30:00Z"],
                    "form": ["8-K"],
                }
            }
        }

    records = await DataLoader(settings, requester=request).download_macro_series(
        provider="sec", series="320193"
    )
    assert records[0].available_at == datetime(2025, 1, 2, 12, 30, tzinfo=UTC)
    assert records[0].values["point_in_time_status"] == "SOURCE_ACCEPTED_TIME"
    assert records[0].values["source_url"].startswith("https://www.sec.gov/")


@pytest.mark.asyncio
async def test_actual_funding_interval_annualization() -> None:
    async def request(method, url, params, headers):
        del method, params, headers
        if "funding_rate" in url:
            return {"data": {"fundingRate": "0.0001", "collectCycle": 4}}
        if "open_interest" in url:
            return {"data": {"holdVol": "1000"}}
        return {
            "data": {
                "fairPrice": "20100",
                "indexPrice": "20000",
                "volume24": "1234",
                "amount24": "24680000",
                "timestamp": 1_735_689_600_000,
            }
        }

    result = await FundingRateCollector(requester=request).collect()
    values = result[0].values
    assert values["funding_interval_seconds"] == 14_400
    assert values["annualized_funding"] == pytest.approx(
        annualize_funding(0.0001, 14_400)
    )
    assert values["funding_periods_per_year"] != 1095
    assert values["perpetual_base_volume_24h"] == 1234
    assert values["perpetual_quote_volume_24h"] == 24_680_000


@pytest.mark.asyncio
async def test_derivatives_context_persists_typed_availability(
    isolated_settings: Settings,
) -> None:
    async def request(method, url, params, headers):
        del method, params, headers
        if "funding_rate" in url:
            return {"data": {"fundingRate": "0.0001", "collectCycle": 8}}
        return {
            "data": {
                "symbol": "BTC_USDT",
                "fairPrice": "20100",
                "indexPrice": "20000",
                "holdVol": "1000",
                "volume24": "1234",
                "amount24": "24680000",
                "timestamp": 1_735_689_600_000,
            }
        }

    loader = DataLoader(isolated_settings, requester=request)
    await loader.download_derivatives_context(
        provider="mexc",
        market="BTC-USDT",
        persist=True,
    )
    await loader.download_derivatives_context(
        provider="mexc",
        market="BTC-USDT",
        persist=True,
    )
    target = (
        isolated_settings.paths.context_data_dir
        / "derivatives_mexc_BTC.parquet"
    )
    frame = pd.read_parquet(target)
    assert str(frame["available_at"].dtype) == "datetime64[ns, UTC]"
    assert frame["source_available_at"].notna().sum() >= 2


def test_gex_formulas_and_convention_metadata() -> None:
    observed = datetime.now(UTC)
    contracts = [
        OptionsContract(
            provider="unit",
            underlying="BTC",
            expiry=observed + timedelta(days=7),
            strike=20_000,
            option_type=option_type,
            spot_or_index_price=20_000,
            open_interest=100,
            gamma=0.0001,
            contract_multiplier=1,
            observed_at=observed,
            available_at=observed,
        )
        for option_type in ("call", "put")
    ]
    result = CryptoGEXAnalyzer().calculate(contracts, now=observed)
    assert result["call_gex_proxy"] == pytest.approx(40_000)
    assert result["put_gex_proxy"] == pytest.approx(40_000)
    assert result["gross_gex_proxy"] == pytest.approx(80_000)
    assert result["net_gex_proxy"] == pytest.approx(0)
    assert not result["assumptions"]["dealer_positioning_known"]
    assert "heuristic" in result["assumptions"]["warning"]


def source_spec(
    provider: str,
    frequency: str,
    cadence: timedelta,
    maximum_age: timedelta,
    units: dict[str, str],
) -> MacroSourceSpec:
    return MacroSourceSpec(
        provider=provider,
        source_frequency=frequency,
        expected_cadence=cadence,
        maximum_age=maximum_age,
        units=units,
    )


def test_persisted_macro_build_uses_real_available_at_and_breadth(tmp_path) -> None:
    context = tmp_path / "context"
    processed = tmp_path / "normalized"
    context.mkdir()
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    pd.DataFrame(
        {
            "provider": "alternative_me",
            "available_at": index,
            "observed_at": index,
            "fear_greed": np.linspace(20, 80, len(index)),
        }
    ).to_parquet(context / "alternative_me_fear_and_greed.parquet", index=False)
    hourly = pd.date_range("2025-01-01", periods=24 * 40, freq="h", tz="UTC")
    for market, multiplier in (("BTC-EUR", 1.0), ("ETH-EUR", 0.08)):
        target = processed / "bitvavo" / market
        target.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp": hourly,
                "canonical_market": market,
                "close": np.linspace(100, 140, len(hourly)) * multiplier,
            }
        ).to_parquet(target / "1h.parquet", index=False)
    report = build_persisted_macro_context(
        context_dir=context,
        processed_dir=processed,
        timeframes=["1h"],
    )
    assert report["status"] == "READY"
    output = context / "macro_context_1h.parquet"
    result = pd.read_parquet(output)
    assert not result.empty
    assert "breadth_fraction_above_ema20" in result
    assert "breadth_positive_return_7d" in result
    assert result.index.max() <= hourly.max()
    coverage = pd.read_csv(context / "macro_context_coverage.csv")
    assert {"sentiment", "breadth", "relative_strength"} <= set(
        coverage["feature_group"]
    )


def test_macro_cadence_causality_all_groups_and_weighted_completeness() -> None:
    base = pd.date_range("2025-01-01", periods=300, freq="h", tz="UTC")
    daily = pd.date_range("2024-10-01", periods=110, freq="D", tz="UTC")
    hourly_values = np.linspace(100, 150, len(base))
    fear = pd.DataFrame({"fear_greed": np.linspace(20, 80, len(daily))}, index=daily)
    dominance = pd.DataFrame(
        {
            "btc_dominance": np.linspace(0.5, 0.55, len(daily)),
            "stablecoin_dominance": np.linspace(0.1, 0.08, len(daily)),
            "total_market_cap": np.linspace(1e12, 1.2e12, len(daily)),
        },
        index=daily,
    )
    relative = pd.DataFrame(
        {
            "btc": hourly_values,
            "eth": hourly_values * 0.08,
            "sol": hourly_values * 0.004,
        },
        index=base,
    )
    breadth = pd.DataFrame(
        {f"asset_{number}": hourly_values * (1 + number / 20) for number in range(5)},
        index=base,
    )
    derivatives = pd.DataFrame(
        {
            "funding_rate": np.full(len(base), 0.0001),
            "funding_interval_seconds": np.full(len(base), 28_800),
            "open_interest": np.linspace(1e8, 1.2e8, len(base)),
            "basis": np.linspace(0, 100, len(base)),
        },
        index=base,
    )
    flows = pd.DataFrame(
        {"btc_etf_flow": np.ones(len(daily)), "eth_etf_flow": np.ones(len(daily))},
        index=daily,
    )
    onchain = pd.DataFrame(
        {
            "mvrv": np.linspace(1, 2, len(daily)),
            "sopr": np.linspace(0.9, 1.1, len(daily)),
            "active_addresses": np.linspace(1e5, 2e5, len(daily)),
        },
        index=daily,
    )
    global_macro = pd.DataFrame(
        {
            "dxy": np.linspace(105, 100, len(daily)),
            "nasdaq": np.linspace(15_000, 17_000, len(daily)),
            "vix": np.linspace(25, 18, len(daily)),
        },
        index=daily,
    )
    event_available = pd.DatetimeIndex([base[30], base[200]])
    events = pd.DataFrame(
        {
            "event_at": [base[40], base[220]],
            "impact": ["high", "medium"],
            "event_type": ["macro", "token_unlock"],
            "unlock_fraction": [0.0, 0.02],
        },
        index=event_available,
    )
    gex = pd.DataFrame(
        {
            "gross_gex_proxy": [1e9],
            "net_gex_proxy": [2e8],
            "gamma_concentration": [0.5],
            "dominant_gamma_strike": [100],
            "spot_distance_from_dominant_gamma": [0.01],
            "stale": [False],
        },
        index=pd.DatetimeIndex([base[50]]),
    )
    specs = {
        "sentiment": source_spec("unit", "1d", timedelta(days=1), timedelta(days=2), {"fear_greed": "index"}),
        "dominance": source_spec("unit", "1d", timedelta(days=1), timedelta(days=2), {"btc_dominance": "fraction", "stablecoin_dominance": "fraction", "total_market_cap": "currency"}),
        "relative_strength": source_spec("unit", "1h", timedelta(hours=1), timedelta(hours=2), {"btc": "price", "eth": "price", "sol": "price"}),
        "breadth": source_spec("unit", "1h", timedelta(hours=1), timedelta(hours=2), {column: "price" for column in breadth}),
        "derivatives": source_spec("unit", "1h", timedelta(hours=1), timedelta(hours=2), {"funding_rate": "fraction", "funding_interval_seconds": "count", "open_interest": "currency", "basis": "currency"}),
        "flows": source_spec("unit", "1d", timedelta(days=1), timedelta(days=4), {"btc_etf_flow": "currency", "eth_etf_flow": "currency"}),
        "onchain": source_spec("unit", "1d", timedelta(days=1), timedelta(days=3), {"mvrv": "index", "sopr": "index", "active_addresses": "count"}),
        "global_macro": source_spec("unit", "1d", timedelta(days=1), timedelta(days=2), {"dxy": "index", "nasdaq": "index", "vix": "index"}),
        "events": source_spec("unit", "event", timedelta(hours=1), timedelta(days=30), {"unlock_fraction": "fraction"}),
        "gex": source_spec("unit", "snapshot", timedelta(hours=1), timedelta(days=2), {"gross_gex_proxy": "currency", "net_gex_proxy": "currency", "gamma_concentration": "fraction", "dominant_gamma_strike": "price", "spot_distance_from_dominant_gamma": "fraction"}),
    }
    result = MacroContextEngine().build(
        base,
        fear_greed=fear,
        dominance=dominance,
        relative_prices=relative,
        breadth_prices=breadth,
        derivatives=derivatives,
        etf_flows=flows,
        onchain=onchain,
        global_macro=global_macro,
        events=events,
        gex=gex,
        source_specs=specs,
    )
    assert all(f"{group}_completeness" in result for group in GROUPS)
    assert result["weighted_total_completeness"].between(0, 1).all()
    assert result.loc[base[37], "events_event_count"] == 0
    assert result.loc[base[38], "events_high_impact_event_risk"]
    assert result["derivatives_annualized_funding"].dropna().iloc[-1] == pytest.approx(
        annualize_funding(0.0001, 28_800)
    )
    hourly = pd.Series(np.arange(300) + 100, index=base)
    seven_days = cadence_change(
        hourly,
        window=7,
        unit="days",
        expected_cadence=timedelta(hours=1),
    )
    assert seven_days.iloc[168] == pytest.approx(hourly.iloc[168] / hourly.iloc[0] - 1)
    assert result.attrs["source_specs"]["sentiment"]["source_frequency"] == "1d"
