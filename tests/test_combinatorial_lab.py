from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest
from pydantic import SecretStr

from config.settings import Settings
from core.cli import build_parser
from core.contracts import DataValidationError
from data.data_loader import DataLoader
from data.database import TABLE_NAMES, Database
from data.market_data import save_ohlcv
from research.backtest import BacktestConfig, BacktestEngine
from research.combinatorial_lab import (
    ECONOMIC_HYPOTHESIS_TEMPLATES,
    HYPOTHESIS_BLOCKS,
    BlockDirection,
    BlockRole,
    CombinationGenerator,
    CombinationState,
    CombinatorialStrategy,
    GenerationMode,
    LabControl,
    LabRunner,
    LabStore,
    LogicMode,
    ParameterKind,
    ParameterSpec,
    UniverseManager,
    UniverseType,
    _matches_research_slice,
    canonical_parameters,
    fast_screen,
    parameter_hash,
    screening_survivor_score,
    signal_block_registry,
    validate_blocks,
)
from research.features import (
    FeaturePipeline,
    ema,
    parameterized_feature_series,
    rsi,
    sma,
)
from research.strategies import StrategyOutput
from utils.common import atomic_write_json


def lab_settings(settings: Settings, tmp_path) -> Settings:
    paths = settings.paths.model_copy(
        update={
            "lab_dir": (tmp_path / "lab").resolve(),
            "database_path": (tmp_path / "lab.db").resolve(),
            "checkpoints_dir": (tmp_path / "checkpoints").resolve(),
            "processed_data_dir": (tmp_path / "normalized").resolve(),
        }
    )
    return settings.model_copy(update={"paths": paths})


def test_exact_half_steps_hashes_and_generalized_smoothing(features) -> None:
    specification = ParameterSpec(
        name="value",
        kind=ParameterKind.HALF_STEP,
        minimum=Decimal("13.0"),
        maximum=Decimal("15.0"),
        step=Decimal("0.5"),
        default=Decimal("14.0"),
    )
    values = specification.values()
    assert values == (
        Decimal("13.0"),
        Decimal("13.5"),
        Decimal("14.0"),
        Decimal("14.5"),
        Decimal("15.0"),
    )
    hashes = {
        parameter_hash({"rsi_threshold": {"value": value}})
        for value in values
    }
    assert len(hashes) == 5
    assert canonical_parameters({"value": Decimal("14.5")}) == {"value": "14.5"}
    assert rsi(features["close"], Decimal("13.5")).notna().any()
    assert ema(features["close"], Decimal("20.5")).notna().any()
    with pytest.raises(TypeError, match="integer-only"):
        sma(features["close"], Decimal("20.5"))  # type: ignore[arg-type]


def test_generalized_parameter_families_have_exact_hashes_outputs_and_cache(
    ohlcv,
    features,
    tmp_path,
) -> None:
    families = {
        "rsi": (
            (Decimal(value) for value in ("13.0", "13.5", "14.0", "14.5", "15.0")),
            lambda value: parameterized_feature_series(
                ohlcv,
                "rsi",
                {"period": value},
                provider_context_hash="acceptance-rsi",
                cache_dir=tmp_path,
            ),
        ),
        "ema": (
            (Decimal(value) for value in ("19.0", "19.5", "20.0", "20.5", "21.0")),
            lambda value: parameterized_feature_series(
                ohlcv,
                "ema",
                {"period": value},
                provider_context_hash="acceptance-ema",
                cache_dir=tmp_path,
            ),
        ),
        "atr": (
            (Decimal(value) for value in ("13.0", "13.5", "14.0", "14.5", "15.0")),
            lambda value: parameterized_feature_series(
                ohlcv,
                "atr",
                {"period": value},
                provider_context_hash="acceptance-atr",
                cache_dir=tmp_path,
            ),
        ),
        "bollinger": (
            (Decimal(value) for value in ("1.5", "2.0", "2.5", "3.0")),
            lambda value: parameterized_feature_series(
                ohlcv,
                "bollinger_lower",
                {"period": 20, "multiplier": value},
                provider_context_hash="acceptance-bollinger",
                cache_dir=tmp_path,
            ),
        ),
    }
    all_hashes: set[str] = set()
    for family, (raw_values, calculate) in families.items():
        values = tuple(raw_values)
        outputs = [calculate(value) for value in values]
        keys = [output.attrs["feature_cache_key"] for output in outputs]
        assert len(set(keys)) == len(values)
        assert len({str(value) for value in values}) == len(values)
        assert len({round(float(output.dropna().iloc[-1]), 12) for output in outputs}) == len(values)
        all_hashes.update(
            parameter_hash({family: {"value": value}}) for value in values
        )
        repeated = calculate(values[0])
        assert repeated.attrs["feature_cache"] == "MEMORY_HIT"
        assert repeated.attrs["feature_cache_key"] == keys[0]

    registry = signal_block_registry()
    adx_counts = [
        int(registry["adx_trend_strength"].calculate(features, {"value": value}).sum())
        for value in map(Decimal, ("20.0", "20.5", "21.0", "21.5", "22.0"))
    ]
    volume_counts = [
        int(registry["relative_volume"].calculate(features, {"value": value}).sum())
        for value in map(Decimal, ("1.0", "1.5", "2.0", "2.5"))
    ]
    assert adx_counts[0] != adx_counts[-1]
    assert volume_counts[0] != volume_counts[-1]
    assert len(all_hashes) == 19


def test_signal_block_registry_is_safe_complete_and_constrained() -> None:
    registry = signal_block_registry()
    validation = validate_blocks(registry)
    assert validation["status"] == "PASSED"
    assert validation["registered"] >= 100
    assert "rsi_threshold" in registry
    assert "ema_trend" in registry
    assert "relative_volume" in registry
    assert not any(block_id.startswith("raw_fractal") for block_id in registry)
    assert not any(
        block.direction is BlockDirection.BEARISH
        and block.role is BlockRole.ENTRY_TRIGGER
        for block in registry.values()
    )
    with pytest.raises(ValueError, match="below"):
        registry["ema_trend"].parameters(
            {"fast": Decimal("50.0"), "slow": Decimal("20.0")}
        )


def test_economic_hypothesis_templates_are_exact_and_statically_valid() -> None:
    registry = signal_block_registry()
    assert set(ECONOMIC_HYPOTHESIS_TEMPLATES) == {
        "TREND_BREAKOUT",
        "PULLBACK_IN_UPTREND",
        "RANGE_MEAN_REVERSION",
        "VOLATILITY_EXPANSION",
        "BTC_RELATIVE_STRENGTH",
    }
    for membership in ECONOMIC_HYPOTHESIS_TEMPLATES.values():
        generated = CombinationGenerator(registry).generate(
            sizes=(len(membership),),
            logic_modes=(LogicMode.LAYERED,),
            mode=GenerationMode.FAMILY_AWARE,
            block_ids=membership,
            timeframes=("1h",),
        )
        exact = [
            item
            for item in generated
            if item.block_ids == tuple(sorted(membership))
        ]
        assert len(exact) == 1
        assert exact[0].eligibility_status is CombinationState.GENERATED


def test_combinations_one_through_five_are_canonical_and_accounted() -> None:
    complete = signal_block_registry()
    selected_ids = (
        "rsi_threshold",
        "donchian20_breakout",
        "bullish_bos",
        "bullish_liquidity_sweep",
        "positive_return_20",
    )
    selected = {block_id: complete[block_id] for block_id in selected_ids}
    generator = CombinationGenerator(selected)
    combinations = generator.generate(
        sizes=(1, 2, 3, 4, 5),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )
    assert len(combinations) == 31
    assert all(item.block_ids == tuple(sorted(item.block_ids)) for item in combinations)
    assert len({item.strategy_dna_hash for item in combinations}) == 31
    assert all(item.eligibility_status is CombinationState.GENERATED for item in combinations)
    reversed_generator = CombinationGenerator(
        {block_id: selected[block_id] for block_id in reversed(selected_ids)}
    )
    reversed_combinations = reversed_generator.generate(
        sizes=(5,),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )
    assert reversed_combinations[0].strategy_dna_hash == combinations[-1].strategy_dna_hash


def test_combination_cursor_resume_and_full_timeframe_intersection() -> None:
    registry = signal_block_registry()
    selected_ids = ("bullish_bos", "bullish_choch", "rsi_threshold")
    generator = CombinationGenerator(
        {block_id: registry[block_id] for block_id in selected_ids}
    )
    full = generator.generate(
        sizes=(1, 2),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("5m", "1h"),
    )
    first = generator.generate(
        sizes=(1, 2),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("5m", "1h"),
        maximum_rows=2,
    )
    assert generator.last_generation_status["status"] == "PARTIAL_GENERATION"
    second = generator.generate(
        sizes=(1, 2),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("5m", "1h"),
        maximum_rows=2,
        continuation_cursor=first[-1].strategy_dna_hash,
    )
    third = generator.generate(
        sizes=(1, 2),
        logic_modes=(LogicMode.LAYERED,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("5m", "1h"),
        continuation_cursor=second[-1].strategy_dna_hash,
    )
    assert [item.strategy_dna_hash for item in [*first, *second, *third]] == [
        item.strategy_dna_hash for item in full
    ]
    assert all(
        item.common_supported_timeframes == ("1h", "5m")
        for item in full
    )
    assert all(item.excluded_timeframes == () for item in full)


def test_invalid_combinations_store_reasons_and_candles_require_context() -> None:
    registry = signal_block_registry()
    generator = CombinationGenerator(
        {
            key: registry[key]
            for key in ("bearish_bos", "doji", "bearish_engulfing")
        }
    )
    combinations = generator.generate(
        sizes=(1, 2),
        mode=GenerationMode.FAMILY_AWARE,
        timeframes=("1h",),
    )
    assert all(
        item.eligibility_status is CombinationState.INVALID_STATIC_RULES
        for item in combinations
    )
    assert {item.exclusion_reason for item in combinations} <= {
        "NO_ENTRY_CAPABLE_BLOCK",
        "CANDLE_CONTEXT_REQUIRED",
        "REDUNDANT_INFORMATION_FAMILY",
    }


def ranking(
    rank: int,
    symbol: str,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "cmc_rank": rank,
        "cmc_id": rank,
        "symbol": symbol,
        "name": name or symbol,
        "slug": (name or symbol).casefold().replace(" ", "-"),
        "market_cap": 1_000_000_000 / rank,
        "circulating_supply": 1_000_000,
        "total_supply": 2_000_000,
        "maximum_supply": 3_000_000,
        "volume_24h": 100_000_000,
        "provider_timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        "tags": tags or [],
    }


def test_universe_scans_past_exclusions_and_persists_point_in_time(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    database = Database(sqlite_path=settings.paths.database_path)
    manager = UniverseManager(settings, database=database)
    snapshot = manager.build_snapshot(
        [
            ranking(1, "BTC", name="Bitcoin"),
            ranking(2, "USDT", name="Tether", tags=["stablecoin"]),
            ranking(3, "WBTC", name="Wrapped Bitcoin", tags=["wrapped"]),
            ranking(4, "ETH", name="Ethereum"),
            ranking(5, "SOL", name="Solana"),
        ],
        provider_markets={
            "bitvavo": {"BTC-EUR", "ETH-EUR", "SOL-EUR"},
            "kraken": {"BTC-EUR", "ETH-EUR", "SOL-EUR"},
            "mexc": {"BTC-USDT", "ETH-USDT", "SOL-USDT"},
        },
        target_size=3,
        scan_limit=5,
        historical_rows={"BTC": 1_000, "ETH": 1_000, "SOL": 1_000},
        observed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert [member.symbol for member in snapshot.research_eligible] == [
        "BTC",
        "ETH",
        "SOL",
    ]
    assert [
        member.symbol
        for member in snapshot.members
        if UniverseType.RAW_CMC_TOP_N in member.universe_types
    ] == ["BTC", "USDT", "WBTC"]
    excluded = {member.symbol: member.exclusion_reasons for member in snapshot.members}
    assert "STABLECOIN" in excluded["USDT"]
    assert "WRAPPED_REPRESENTATION" in excluded["WBTC"]
    assert snapshot.bias_label == "CURRENT_UNIVERSE_RETROSPECTIVE"
    assert all(
        UniverseType.EXECUTION_ELIGIBLE in member.universe_types
        for member in snapshot.execution_eligible
    )
    assert database.health()["table_counts"]["universe_snapshots"] == 1
    assert database.health()["table_counts"]["universe_members"] == 5


def test_discovery_universe_reaches_25_independent_of_allowlist(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    manager = UniverseManager(settings, database=Database(sqlite_path=settings.paths.database_path))
    symbols = ["BTC", "ETH", "SOL", "LINK", *[f"A{index:02d}" for index in range(1, 28)]]
    rankings = [
        ranking(index, symbol, name=f"Asset {symbol}")
        for index, symbol in enumerate(symbols, start=1)
    ]
    eur_markets = {f"{symbol}-EUR" for symbol in symbols}
    snapshot = manager.build_snapshot(
        rankings,
        provider_markets={
            "bitvavo": eur_markets,
            "kraken": eur_markets,
            "mexc": {f"{symbol}-USDT" for symbol in symbols},
        },
        target_size=25,
        scan_limit=len(symbols),
        historical_rows={symbol: 3_000 for symbol in symbols},
        observed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    discovery = [
        member
        for member in snapshot.members
        if UniverseType.DISCOVERY_UNIVERSE in member.universe_types
    ]
    allowed = [
        member
        for member in snapshot.members
        if UniverseType.ALLOWED_RESEARCH in member.universe_types
    ]
    review_only = [
        member
        for member in snapshot.members
        if UniverseType.REVIEW_RESEARCH_ONLY in member.universe_types
    ]
    assert len(discovery) == 25
    assert {member.symbol for member in allowed} == {"BTC", "ETH", "SOL", "LINK"}
    assert len(review_only) == 21
    assert {member.symbol for member in snapshot.execution_eligible} == {
        "BTC",
        "ETH",
        "SOL",
        "LINK",
    }


@pytest.mark.asyncio
async def test_cmc_rank_ingestion_preserves_required_fields(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    providers = settings.providers.model_copy(
        update={"coinmarketcap_api_key": SecretStr("test-key")}
    )
    settings = settings.model_copy(update={"providers": providers})

    async def request(method, url, params, headers):
        assert method == "GET"
        assert url.endswith("/cryptocurrency/listings/latest")
        assert params["limit"] == 25
        assert headers["X-CMC_PRO_API_KEY"] == "test-key"
        return {
            "status": {"timestamp": "2026-01-01T00:00:00Z"},
            "data": [
                {
                    "id": 1,
                    "cmc_rank": 1,
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "slug": "bitcoin",
                    "circulating_supply": 19,
                    "total_supply": 21,
                    "max_supply": 21,
                    "last_updated": "2026-01-01T00:00:00Z",
                    "tags": ["mineable"],
                    "quote": {
                        "EUR": {
                            "market_cap": 1_000,
                            "volume_24h": 100,
                        }
                    },
                }
            ],
        }

    records = await DataLoader(settings, requester=request).download_cmc_rankings(
        limit=25
    )
    assert len(records) == 1
    values = records[0].values
    assert values["cmc_rank"] == 1
    assert values["cmc_id"] == 1
    assert values["market_cap"] == 1_000
    assert values["circulating_supply"] == 19
    assert values["maximum_supply"] == 21
    assert records[0].data_kind == "universe_ranking"


def test_combinatorial_strategy_uses_canonical_backtester_and_no_short(features) -> None:
    registry = signal_block_registry()
    combination = CombinationGenerator(
        {
            key: registry[key]
            for key in ("rsi_threshold", "bearish_bos")
        }
    ).generate(
        sizes=(2,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )[0]
    strategy = CombinatorialStrategy(combination, registry)
    output = strategy.generate(features)
    assert output.metadata["long_only"]
    result = BacktestEngine(
        BacktestConfig(bootstrap_samples=100, monte_carlo_runs=100)
    ).run({"BTC-EUR": features}, strategy)
    assert result.integrity["next_open_execution"]
    assert result.integrity["long_only_spot"]
    assert all(order.side != "SELL_SHORT" for order in result.orders)
    screen = fast_screen(
        {"BTC-EUR": features},
        strategy,
        round_trip_cost=0.005,
    )
    assert screen["source"] == "SCREENING_ONLY"
    assert screen["one_bar_signal_shift"]
    assert screen["canonical_exit_family_approximated"]
    assert not screen["paper_candidate_permitted"]


def test_fast_screen_uses_maximum_holding_exit_instead_of_holding_forever(
    ohlcv,
) -> None:
    selected = ohlcv.iloc[:12].copy()
    selected[["open", "high", "low", "close"]] = 100.0

    class AlwaysEnter:
        def generate(self, frame):
            index = frame.index
            return StrategyOutput(
                entry=pd.Series(True, index=index),
                exit=pd.Series(False, index=index),
                avoid=pd.Series(False, index=index),
                reduce=pd.Series(False, index=index),
                stop_distance=pd.Series(50.0, index=index),
                target_distance=pd.Series(50.0, index=index),
                trailing_distance=pd.Series(0.0, index=index),
                size_multiplier=pd.Series(1.0, index=index),
                maximum_holding_bars=3,
                entry_reason="TEST",
                exit_reason="TEST",
            )

    screen = fast_screen(
        {"BTC-EUR": selected},
        AlwaysEnter(),
        round_trip_cost=0.01,
    )
    assert screen["trades"] == 4
    assert screen["screening_return"] == pytest.approx(0.99**4 - 1.0)
    assert screen["canonical_exit_family_approximated"]
    assert screen["conservative_same_bar_stop_priority"]


def test_fast_screen_survivor_score_requires_minimum_trades_and_finite_score() -> None:
    assert (
        screening_survivor_score(
            {"trades": 0, "screening_score": 0.0},
            minimum_trades=30,
        )
        is None
    )
    assert (
        screening_survivor_score(
            {"trades": 29, "screening_score": 100.0},
            minimum_trades=30,
        )
        is None
    )
    assert (
        screening_survivor_score(
            {"trades": 30, "screening_score": float("nan")},
            minimum_trades=30,
        )
        is None
    )
    assert screening_survivor_score(
        {"trades": 30, "screening_score": 1.25},
        minimum_trades=30,
    ) == pytest.approx(1.25)


def test_weighted_vote_uses_weights_and_overlay_reduces_once(features) -> None:
    registry = signal_block_registry()
    controlled = features.copy()
    controlled["bullish_bos"] = True
    controlled["bullish_choch"] = False
    weighted_combination = CombinationGenerator(
        {
            key: registry[key]
            for key in ("bullish_bos", "bullish_choch")
        }
    ).generate(
        sizes=(2,),
        logic_modes=(LogicMode.WEIGHTED_VOTE,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )[0]
    weighted = CombinatorialStrategy(weighted_combination, registry)
    favors_true = weighted.generate(
        controlled,
        {
            "logic__vote_threshold": Decimal("0.7"),
            "logic__weight__bullish_bos": Decimal("1.5"),
            "logic__weight__bullish_choch": Decimal("0.5"),
        },
    )
    favors_false = weighted.generate(
        controlled,
        {
            "logic__vote_threshold": Decimal("0.7"),
            "logic__weight__bullish_bos": Decimal("0.5"),
            "logic__weight__bullish_choch": Decimal("1.5"),
        },
    )
    assert favors_true.entry.all()
    assert not favors_false.entry.any()

    controlled["gex_gamma_concentration"] = 0.0
    controlled.iloc[10:13, controlled.columns.get_loc("gex_gamma_concentration")] = 1.0
    controlled.iloc[20:22, controlled.columns.get_loc("gex_gamma_concentration")] = 1.0
    overlay_combination = CombinationGenerator(
        {
            key: registry[key]
            for key in ("bullish_bos", "gamma_concentration")
        }
    ).generate(
        sizes=(2,),
        mode=GenerationMode.EXHAUSTIVE,
        timeframes=("1h",),
    )[0]
    overlay = CombinatorialStrategy(overlay_combination, registry).generate(controlled)
    assert list(overlay.reduce[overlay.reduce].index) == [
        controlled.index[10],
        controlled.index[20],
    ]


def test_real_frames_require_real_provenance_and_never_fallback(
    isolated_settings,
    tmp_path,
    ohlcv,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    runner = LabRunner(settings)
    with pytest.raises(DataValidationError, match="BLOCKED_DATA_UNAVAILABLE"):
        runner._frames(
            markets=["BTC-EUR"],
            timeframe="1h",
            rows=200,
            data_mode="real",
        )
    path, manifest = save_ohlcv(
        ohlcv,
        settings.paths.processed_data_dir / "BTC-EUR_1h.parquet",
        market="BTC-EUR",
        timeframe="1h",
    )
    provenance = {
        "source_type": "REAL_PROVIDER_DATA",
        "providers_used": ["bitvavo", "kraken"],
        "closed_candles_only": True,
        "data_sha256": manifest.sha256,
    }
    atomic_write_json(path.with_suffix(f"{path.suffix}.provenance.json"), provenance)
    frames, _, loaded_provenance = runner._frames(
        markets=["BTC-EUR"],
        timeframe="1h",
        rows=200,
        data_mode="real",
    )
    assert len(frames["BTC-EUR"]) == 200
    assert loaded_provenance["BTC-EUR"]["source_type"] == "REAL_PROVIDER_DATA"
    assert (
        frames["BTC-EUR"].attrs["data_provenance"]["source_type"]
        == "REAL_PROVIDER_DATA"
    )
    assert loaded_provenance["BTC-EUR"]["research_slice"]["rows"] == 200
    assert len(loaded_provenance["BTC-EUR"]["feature_definition_hash"]) == 64
    assert len(loaded_provenance["BTC-EUR"]["feature_output_hash"]) == 64
    assert len(loaded_provenance["BTC-EUR"]["feature_hash"]) == 64
    assert len(loaded_provenance["BTC-EUR"]["context_hash"]) == 64
    assert (
        frames["BTC-EUR"].attrs["data_provenance"]["feature_hash"]
        == loaded_provenance["BTC-EUR"]["feature_hash"]
    )

    full_frames, full_hash, _ = runner._frames(
        markets=["BTC-EUR"],
        timeframe="1h",
        rows=None,
        data_mode="real",
    )
    assert len(full_frames["BTC-EUR"]) == len(ohlcv)
    assert len(full_frames["BTC-EUR"]) > 500
    sliced_frames, sliced_hash, sliced_provenance = runner._frames(
        markets=["BTC-EUR"],
        timeframe="1h",
        rows=None,
        data_mode="real",
        start_at=ohlcv.index[100],
        end_at=ohlcv.index[500],
    )
    assert sliced_frames["BTC-EUR"].index[0] == ohlcv.index[100]
    assert sliced_frames["BTC-EUR"].index[-1] == ohlcv.index[500]
    assert len(sliced_frames["BTC-EUR"]) == 401
    assert sliced_hash != full_hash
    assert (
        sliced_provenance["BTC-EUR"]["feature_hash"]
        != loaded_provenance["BTC-EUR"]["feature_hash"]
    )
    assert sliced_provenance["BTC-EUR"]["research_slice"][
        "common_period_requested"
    ]


def test_durable_job_dedup_resume_and_half_step_trial_rows(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    store = LabStore(settings)
    registry = signal_block_registry()
    combination = CombinationGenerator({"rsi_threshold": registry["rsi_threshold"]}).generate(
        sizes=(1,),
        timeframes=("1h",),
    )[0]
    store.persist_combinations([combination])
    hashes = set()
    for value in (
        Decimal("13.0"),
        Decimal("13.5"),
        Decimal("14.0"),
        Decimal("14.5"),
        Decimal("15.0"),
    ):
        parameters = {
            "rsi_threshold": {
                "value": value,
                "period": Decimal("14.0"),
            }
        }
        job = store.queue_job(
            run_id="unit",
            combination=combination,
            snapshot_id="snapshot",
            markets=["BTC-EUR"],
            timeframe="1h",
            parameters=parameters,
            data_hash="data",
        )
        hashes.add(job["parameter_hash"])
        store.save_result(
            "experiment_trials",
            job=job,
            result={
                "trial_id": f"trial-{value}",
                "source": "SCREENING_ONLY",
                "parameter_hash": job["parameter_hash"],
            },
            status="SCREENING_ONLY",
        )
        completed = store.update_job(
            job,
            status=CombinationState.BASELINE_COMPLETED,
            reason_code="UNIT_COMPLETE",
        )
        duplicate = store.queue_job(
            run_id="resume",
            combination=combination,
            snapshot_id="snapshot",
            markets=["BTC-EUR"],
            timeframe="1h",
            parameters=parameters,
            data_hash="data",
        )
        assert completed["job_id"] == duplicate["job_id"]
        assert duplicate["deduplicated"]
    assert len(hashes) == 5
    assert store.database.health()["table_counts"]["experiment_trials"] == 5
    assert store.database.health()["table_counts"]["experiment_jobs"] == 5


def test_queue_status_is_run_scoped_and_old_incomplete_jobs_are_superseded(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    store = LabStore(settings)
    registry = signal_block_registry()
    combination = CombinationGenerator(
        {"rsi_threshold": registry["rsi_threshold"]}
    ).generate(sizes=(1,), timeframes=("1h",))[0]
    old = store.queue_job(
        run_id="old-run",
        combination=combination,
        snapshot_id="snapshot",
        markets=["BTC-EUR"],
        timeframe="1h",
        parameters={"rsi_threshold": {"value": 30, "period": 14}},
        data_hash="old-data",
    )
    current = store.queue_job(
        run_id="current-run",
        combination=combination,
        snapshot_id="snapshot",
        markets=["BTC-EUR"],
        timeframe="1h",
        parameters={"rsi_threshold": {"value": 30, "period": 14}},
        data_hash="current-data",
    )
    assert store.queue_status(run_id="current-run")["total"] == 1
    assert store.supersede_incomplete_jobs(active_run_id="current-run") == 1
    assert store.job(old["job_id"])["status"] == CombinationState.SUPERSEDED.value
    assert (
        store.job(current["job_id"])["status"]
        == CombinationState.QUEUED_BASELINE.value
    )
    assert store.queue_status()["remaining_work"] == 1


def test_persisted_results_must_match_the_exact_active_data_slice() -> None:
    base = {
        "data_hash": "current-5m-hash",
        "universe_snapshot_id": "snapshot",
        "source": "BASELINE_REAL",
        "assets_tested": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
        "timeframes_tested": ["5m"],
    }
    arguments = {
        "data_hashes_by_timeframe": {
            "5m": "current-5m-hash",
            "15m": "current-15m-hash",
        },
        "feature_hashes_by_timeframe": {
            "5m": "current-5m-features",
            "15m": "current-15m-features",
        },
        "screening_engine_version": "2.0.0",
        "markets": ["BTC-EUR", "ETH-EUR", "SOL-EUR", "LINK-EUR"],
        "snapshot_id": "snapshot",
        "sources": {"BASELINE_REAL"},
    }
    base["feature_hash"] = "current-5m-features"
    base["screening_engine_version"] = "2.0.0"
    assert _matches_research_slice(base, **arguments)
    assert not _matches_research_slice(
        {**base, "data_hash": "old-tail-hash"},
        **arguments,
    )
    assert not _matches_research_slice(
        {**base, "timeframes_tested": ["5m", "15m"]},
        **arguments,
    )
    assert not _matches_research_slice(
        {**base, "feature_hash": "old-feature-build"},
        **arguments,
    )
    assert not _matches_research_slice(
        {**base, "screening_engine_version": "1.0.0"},
        **arguments,
    )
    assert not _matches_research_slice(
        {**base, "assets_tested": ["BTC-EUR"]},
        **arguments,
    )


@pytest.mark.asyncio
async def test_guarded_run_persists_terminal_failure_status(
    isolated_settings,
    tmp_path,
    monkeypatch,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    runner = LabRunner(settings)
    atomic_write_json(
        runner.current_status_path,
        {
            "run_id": "failed-run",
            "status": "RUNNING",
            "started_at": "2026-01-01T00:00:00Z",
        },
    )

    async def fail(**_arguments):
        raise MemoryError("bounded estimate exceeded")

    monkeypatch.setattr(runner, "run_once", fail)
    with pytest.raises(MemoryError, match="bounded estimate exceeded"):
        await runner.run_once_guarded(profile="hypotheses")
    status = runner.status()
    assert status["status"] == "FAILED"
    assert status["reason_code"] == "MemoryError"
    assert status["live_orders"] == 0
    assert status["queue"]["run_id"] == "failed-run"


@pytest.mark.asyncio
async def test_baseline_stage_is_fast_screen_before_exact_backtest(
    isolated_settings,
    tmp_path,
    features,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    store = LabStore(settings)
    runner = LabRunner(settings, store=store)
    registry = signal_block_registry()
    combination = CombinationGenerator(
        {"positive_return_20": registry["positive_return_20"]}
    ).generate(
        sizes=(1,),
        timeframes=("5m",),
    )[0]
    provenance = {
        "BTC-EUR": {
            "source_type": "REAL_PROVIDER_DATA",
            "feature_hash": "feature-hash",
        }
    }
    job = store.queue_job(
        run_id="screen-run",
        combination=combination,
        snapshot_id="snapshot",
        markets=["BTC-EUR"],
        timeframe="5m",
        parameters=combination.default_parameters,
        data_hash="data-hash",
        feature_hash="feature-hash",
        data_provenance=provenance,
    )
    selected = features.iloc[:300].copy()
    selected.attrs.update(features.attrs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        completed, payload = await runner._baseline_job(
            job=job,
            combination=combination,
            frames={"BTC-EUR": selected},
            bias_label="CURRENT_UNIVERSE_RETROSPECTIVE",
            semaphore=asyncio.Semaphore(1),
            executor=executor,
        )
    assert completed["status"] == CombinationState.SCREENING_COMPLETED.value
    assert payload is not None
    assert payload["source"] == "FAST_SCREEN_REAL"
    assert payload["integrity"]["exact_event_driven"] is False
    assert payload["paper_candidate_permitted"] is False
    assert len(store.database.fetch_records("experiment_trials")) == 1
    assert store.database.fetch_records("baseline_results") == []
    assert store.queue_status(run_id="screen-run")["remaining_work"] == 0


def test_nested_real_provenance_and_default_leaderboard_are_fail_closed(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    store = LabStore(settings)
    registry = signal_block_registry()
    combination = CombinationGenerator(
        {"rsi_threshold": registry["rsi_threshold"]}
    ).generate(sizes=(1,), timeframes=("1h",))[0]
    job = store.queue_job(
        run_id="real",
        combination=combination,
        snapshot_id="snapshot",
        markets=["BTC-EUR"],
        timeframe="1h",
        parameters=combination.default_parameters,
        data_hash="data",
        data_provenance={
            "BTC-EUR": {
                "source_type": "REAL_PROVIDER_DATA",
                "providers_used": ["bitvavo", "kraken"],
            }
        },
    )
    assert job["source_type"] == "REAL_PROVIDER_DATA"
    for entry_id, source_type, score in (
        ("real", "REAL_PROVIDER_DATA", 0.0),
        ("synthetic", "SYNTHETIC_SMOKE", 10.0),
        ("unknown", None, 20.0),
    ):
        store.save_leaderboard_entry(
            {
                "entry_id": entry_id,
                "lifecycle_status": "DEGRADED",
                "source_type": source_type,
                "robust_score": score,
                "strategy_dna_hash": entry_id,
            }
        )
    assert [row["entry_id"] for row in store.leaderboard()] == ["real"]
    assert {row["entry_id"] for row in store.leaderboard(include_synthetic=True)} == {
        "real",
        "synthetic",
        "unknown",
    }


def test_lab_lifecycle_lock_controls_and_schema(
    isolated_settings,
    tmp_path,
) -> None:
    settings = lab_settings(isolated_settings, tmp_path)
    runner = LabRunner(settings)
    descriptor = runner._acquire_lock()
    try:
        with pytest.raises(RuntimeError, match="LAB_ALREADY_RUNNING"):
            runner._acquire_lock()
    finally:
        runner._release_lock(descriptor)
    assert runner.control(LabControl.PAUSE)["action"] == "PAUSE"
    assert runner.control(LabControl.RESUME)["action"] == "RESUME"
    assert runner.control(LabControl.DRAIN)["action"] == "DRAIN"
    assert runner.control(LabControl.STOP)["action"] == "STOP"
    required_tables = {
        "universe_snapshots",
        "universe_members",
        "signal_blocks",
        "strategy_combinations",
        "experiment_jobs",
        "experiment_trials",
        "baseline_results",
        "exact_backtest_results",
        "walk_forward_results",
        "leaderboard_entries",
        "lab_heartbeats",
        "lab_events",
    }
    assert required_tables <= set(TABLE_NAMES)


def test_lab_cli_command_surface_parses() -> None:
    parser = build_parser()
    commands = [
        ["lab", "universe", "refresh"],
        ["lab", "universe", "show"],
        ["lab", "blocks", "list"],
        ["lab", "blocks", "describe", "--block", "rsi_threshold"],
        ["lab", "combinations", "estimate"],
        ["lab", "combinations", "generate", "--yes"],
        ["lab", "run", "--once"],
        ["lab", "campaign", "estimate"],
        ["lab", "campaign", "run", "--workers", "6", "--yes"],
        ["lab", "campaign", "status"],
        ["lab", "campaign", "report"],
        [
            "lab",
            "run",
            "--profile",
            "hypotheses",
            "--history-mode",
            "common_full_history",
            "--timeframes",
            "5m,15m",
            "--allowed-universe",
            "--once",
        ],
        [
            "lab",
            "data",
            "prepare",
            "--markets",
            "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
            "--allowed-universe",
            "--timeframes",
            "5m,15m",
        ],
        ["lab", "pause"],
        ["lab", "resume"],
        ["lab", "drain"],
        ["lab", "stop"],
        ["lab", "status"],
        ["lab", "queue"],
        ["lab", "workers"],
        ["lab", "failures"],
        ["lab", "retry"],
        ["lab", "leaderboard"],
        ["lab", "leaderboard", "export"],
        ["lab", "leaderboard", "inspect", "--id", "example"],
        ["lab", "leaderboard", "history"],
        ["lab", "retest"],
        ["lab", "validate"],
        ["lab", "report"],
    ]
    for argv in commands:
        parsed = parser.parse_args(argv)
        assert parsed.command == "lab"
    parsed_run = parser.parse_args(["lab", "run", "--once"])
    assert parsed_run.data_mode == "real"
    assert parser.parse_args(
        ["lab", "run", "--once", "--data-mode", "synthetic"]
    ).data_mode == "synthetic"


def test_hypothesis_profile_contains_entries_filters_exits_and_avoidance() -> None:
    registry = signal_block_registry()
    roles = {registry[block_id].role for block_id in HYPOTHESIS_BLOCKS}
    assert {
        BlockRole.ENTRY_TRIGGER,
        BlockRole.TREND_FILTER,
        BlockRole.REGIME_FILTER,
        BlockRole.CONFIRMATION,
        BlockRole.EXIT_TRIGGER,
        BlockRole.AVOIDANCE_FILTER,
    } <= roles


def test_every_5m_15m_campaign_block_explicitly_supports_both_timeframes() -> None:
    registry = signal_block_registry()
    unsupported = {
        block_id: sorted({"5m", "15m"} - set(registry[block_id].supported_timeframes))
        for block_id in HYPOTHESIS_BLOCKS
        if not {"5m", "15m"}.issubset(
            set(registry[block_id].supported_timeframes)
        )
    }
    assert unsupported == {}
    assert {
        "donchian20_breakout",
        "bollinger_lower_reversion",
        "bollinger_keltner_squeeze",
        "btc_relative_momentum",
        "rsi_overbought_exit",
        "negative_return_exit",
    } <= set(HYPOTHESIS_BLOCKS)


def test_fractal_features_include_causal_structure_metrics(ohlcv) -> None:
    features = FeaturePipeline(fractal_left=2, fractal_right=2).build(ohlcv)
    required = {
        "higher_high",
        "higher_low",
        "lower_high",
        "lower_low",
        "fractal_density_50",
        "fractal_amplitude_atr",
        "distance_to_last_fractal_high_atr",
        "distance_to_last_fractal_low_atr",
        "bars_since_fractal_high",
        "bars_since_fractal_low",
    }
    assert required <= set(features)
    assert features.attrs["feature_knowability"]["confirmed_fractal_high"][
        "confirmation_lag_bars"
    ] == 2
    assert not any(column.startswith("raw_fractal") for column in features)
