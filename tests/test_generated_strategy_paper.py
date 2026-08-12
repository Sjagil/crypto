from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from config.settings import PathSettings, Settings
from core import generated_strategy_paper as generated_paper
from research.mtf_limit_overlay import LimitOverlayParameters
from research.multi_timeframe_authority import MultiTimeframeParameters
from research.portfolio_breakout import AtrRiskBreakoutParameters
from research.strategies import StrategyOutput
from utils.common import atomic_write_json, read_json, utc_now


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    )
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _candidate() -> dict:
    return {
        "strategy_dna_hash": "a" * 64,
        "frozen_candidate_hash": "b" * 64,
        "combination_id": "cmb-test",
        "economic_hypothesis_family": "TEST_BREAKOUT",
        "block_ids": ["test_entry"],
        "logic_mode": "LAYERED",
        "parameters": {},
        "parameter_hash": "c" * 64,
        "timeframe": "4h",
        "markets": ["BTC-EUR"],
        "data_period": {
            "start": "2019-01-01T00:00:00Z",
            "end": "2026-01-02T00:00:00Z",
        },
        "data_hash": "d" * 64,
        "feature_hash": "e" * 64,
        "metrics": {
            "net_return": 0.05,
            "profit_factor": 1.4,
            "net_expectancy_r": 0.1,
            "trade_count": 120,
            "stressed_net_return": 0.03,
            "holdout_net_return": 0.02,
        },
        "integrity": {
            "no_lookahead": True,
            "no_repainting": True,
            "next_open_execution": True,
            "long_only_spot": True,
        },
        "lifecycle": "BACKTEST_POSITIVE",
    }


def test_prospective_entry_cohort_rejects_negative_holdout() -> None:
    robust = _candidate()
    rejected = _candidate()
    rejected["strategy_dna_hash"] = "b" * 64
    rejected["metrics"]["holdout_net_return"] = -0.01

    cohort = generated_paper._prospective_entry_cohort([rejected, robust])

    assert [row["strategy_dna_hash"] for row in cohort] == ["a" * 64]
    assert cohort[0]["prospective_selection"]["live_authority"] is False


def test_paper_candidate_universe_can_exceed_live_launch_markets(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    assert "DOGE-EUR" not in settings.operational.markets
    assert generated_paper._paper_market_allowed(settings, "DOGE-EUR") is True
    assert generated_paper._paper_market_allowed(settings, "DOGE/USDT") is False
    assert generated_paper._paper_market_allowed(settings, "doge-eur") is False


def test_paper_fill_notification_is_never_labelled_live(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    calls: list[tuple[str, dict]] = []

    class CapturingNotifier:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def notify_order_event(self, event_type, payload):
            calls.append((event_type, dict(payload)))
            return {"delivery_status": "PENDING"}

    monkeypatch.setattr(generated_paper, "TelegramNotifier", CapturingNotifier)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            market="BTC-EUR",
            side=SimpleNamespace(value="BUY"),
            order_type=SimpleNamespace(value="MARKET"),
        ),
        average_fill_price=Decimal("50000"),
        filled_quantity=Decimal("0.001"),
        order_id="paper-order-private-id",
        status=SimpleNamespace(value="FILLED"),
    )

    generated_paper._notify_fill(
        settings,
        order=order,
        candidate={"economic_hypothesis_family": "TEST_PAPER"},
    )

    assert len(calls) == 1
    event_type, payload = calls[0]
    assert event_type == "PAPER_ORDER_FILLED"
    assert payload["execution_mode"] == "PAPER_ONLY"
    assert payload["paper_only"] is True
    assert payload["real_exchange_request"] is False


def _exact_record(
    *,
    source: str = "NORMAL",
    net_return: float = 0.05,
    profit_factor: float = 1.4,
    expectancy: float = 0.1,
    trades: int = 12,
    parameter_value: str = "20",
) -> dict:
    return {
        "payload": {
            "source": source,
            "source_type": "REAL_PROVIDER_DATA",
            "result_type": "EXACT_BACKTEST",
            "status": "COMPLETED",
            "bias_label": "CURRENT_UNIVERSE_RETROSPECTIVE",
            "experiment_hash": "experiment-1",
            "strategy_dna_hash": "block-dna",
            "combination_id": "cmb-exact",
            "block_ids": ["donchian20_breakout"],
            "families": ["BREAKOUT"],
            "logic_mode": "LAYERED",
            "parameters": {
                "donchian20_breakout": {"lookback": parameter_value}
            },
            "parameter_hash": f"parameter-{parameter_value}",
            "exit_model_version": "atr-exit-v1",
            "timeframes_tested": ["4h"],
            "assets_tested": ["BTC-EUR"],
            "data_hash": "data-hash",
            "feature_hash": "feature-hash",
            "data_period": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-01-02T00:00:00Z",
            },
            "metrics": {
                "net_return": net_return,
                "profit_factor": profit_factor,
                "net_expectancy_r": expectancy,
                "trade_count": trades,
                "maximum_drawdown": 0.05,
            },
            "integrity": {
                "no_lookahead": True,
                "no_repainting": True,
                "next_open_execution": True,
                "long_only_spot": True,
            },
        }
    }


def test_generated_candidate_requires_sample_and_seven_year_history(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    too_few_trades = _candidate()
    too_few_trades["metrics"] = {
        **too_few_trades["metrics"],
        "trade_count": 99,
    }
    too_short = _candidate()
    too_short["data_period"] = {
        "start": "2021-01-01T00:00:00Z",
        "end": "2026-01-01T00:00:00Z",
    }
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [too_few_trades, too_short],
        },
    )

    assert generated_paper._load_candidates(settings) == []


def test_simple_lab_exact_positive_is_bridged_to_paper_only_registry(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    normal = _exact_record()
    stressed = _exact_record(
        source="STRESSED",
        net_return=-0.01,
        profit_factor=0.9,
    )

    result = generated_paper.refresh_simple_lab_positive_candidates(
        settings,
        records=[normal, stressed],
    )

    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["lifecycle"] == "BACKTEST_POSITIVE"
    assert candidate["paper_eligibility"] == "PAPER_ELIGIBLE_EXACT_POSITIVE"
    assert candidate["paper_risk_multiplier"] == 0.25
    assert candidate["auto_live_promotion"] is False
    assert candidate["block_strategy_dna_hash"] == "block-dna"
    assert candidate["strategy_dna_hash"] != "block-dna"
    assert "STRESSED_COST_EDGE_NOT_POSITIVE" in candidate[
        "capital_scaling_warnings"
    ]
    assert generated_paper._paths(settings)["simple_registry"].is_file()


def test_simple_lab_bridge_rejects_nonpositive_or_integrity_failed_results(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    negative = _exact_record(net_return=-0.01)
    lookahead = _exact_record(parameter_value="55")
    lookahead["payload"]["experiment_hash"] = "experiment-2"
    lookahead["payload"]["integrity"]["no_lookahead"] = False

    result = generated_paper.refresh_simple_lab_positive_candidates(
        settings,
        records=[negative, lookahead],
    )

    assert result["candidate_count"] == 0
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0


def test_simple_lab_parameter_variants_receive_distinct_frozen_dna(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    first = _exact_record(parameter_value="20")
    second = _exact_record(parameter_value="55")
    second["payload"]["experiment_hash"] = "experiment-2"

    result = generated_paper.refresh_simple_lab_positive_candidates(
        settings,
        records=[first, second],
    )

    assert result["candidate_count"] == 2
    assert len(
        {
            candidate["strategy_dna_hash"]
            for candidate in result["candidates"]
        }
    ) == 2


def test_volume_catalog_bridge_uses_new_paper_forward_dna(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    strategy_id = "VOL_ETH_EUR_4h_VOLUME_CONTRACTION_BREAKOUT_N3"
    adapter = generated_paper.volume_strategy_adapter(strategy_id)
    source = generated_paper._paths(settings)["volume_catalog"]
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "strategy_id": strategy_id,
                "strategy_dna_hash": adapter.legacy_strategy_dna_hash,
                "market": "ETH-EUR",
                "timeframe": "4h",
                "full_net_return": 0.35,
                "full_profit_factor": 1.18,
                "full_trade_entries": 56,
                "full_maximum_drawdown": -0.067,
                "stressed_full_net_return": 0.24,
                "validation_net_return": 0.06,
                "confirmation_net_return": 0.01,
            }
        ]
    ).to_csv(source, index=False)

    candidates = generated_paper._load_volume_catalog_paper_candidates(
        settings,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["source_strategy_dna_hash"] == (
        adapter.legacy_strategy_dna_hash
    )
    assert candidate["strategy_dna_hash"] == (
        adapter.canonical_adapter_dna_hash
    )
    assert candidate["strategy_dna_hash"] != candidate[
        "source_strategy_dna_hash"
    ]
    assert candidate["paper_adapter"] == "VOLUME_CATALOG_BOUNDED_RISK"
    assert candidate["adapter_validation_mode"] == "PAPER_FORWARD_ONLY"
    assert candidate["auto_live_promotion"] is False
    assert candidate["parameters"]["stop_fraction"] == 0.10
    assert candidate["parameters"]["target_fraction"] == 0.30


def test_volume_catalog_adapter_produces_bounded_paper_levels() -> None:
    strategy_id = "VOL_ETH_EUR_4h_VOLUME_CONTRACTION_BREAKOUT_N3"
    adapter = generated_paper.volume_strategy_adapter(strategy_id)
    index = pd.date_range(
        end=pd.Timestamp(utc_now()).floor("4h") - pd.Timedelta(4, unit="h"),
        periods=300,
        freq="4h",
    )
    close = pd.Series(
        [100.0 + number * 0.01 for number in range(len(index))],
        index=index,
    )
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    candidate = {
        "strategy_id": strategy_id,
        "strategy_dna_hash": adapter.canonical_adapter_dna_hash,
        "source_strategy_dna_hash": adapter.legacy_strategy_dna_hash,
        "timeframe": "4h",
        "markets": ["ETH-EUR"],
        "parameters": dict(adapter.defaults),
        "paper_adapter": "VOLUME_CATALOG_BOUNDED_RISK",
    }

    evaluations = generated_paper._strategy_market_evaluations(
        adapter,
        candidate,
        {("ETH-EUR", "4h"): frame},
        parameters=candidate["parameters"],
    )

    assert len(evaluations) == 1
    assert evaluations[0]["risk_levels_valid"] is True
    assert evaluations[0]["stop_distance"] == close.iloc[-1] * 0.10
    assert evaluations[0]["target_distance"] == close.iloc[-1] * 0.30


def test_generated_exact_positive_executes_paper_once_and_never_live(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    index = pd.date_range(
        utc_now() - timedelta(hours=12),
        periods=3,
        freq="4h",
    )
    frame = pd.DataFrame({"close": [98.0, 99.0, 100.0]}, index=index)

    async def frames(_settings, _candidates):
        return {("BTC-EUR", "4h"): frame}

    async def price(_settings, _market):
        return Decimal("100")

    class FakeStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, selected):
            false = pd.Series(False, index=selected.index)
            entry = pd.Series([False, False, True], index=selected.index)
            distance = pd.Series(5.0, index=selected.index)
            size = pd.Series(1.0, index=selected.index)
            return StrategyOutput(
                entry=entry,
                exit=false,
                avoid=false,
                reduce=false,
                stop_distance=distance,
                target_distance=distance,
                trailing_distance=pd.Series(0.0, index=selected.index),
                size_multiplier=size,
                maximum_holding_bars=120,
                entry_reason="TEST",
                exit_reason="TEST_EXIT",
            )

    monkeypatch.setattr(
        generated_paper,
        "_combination",
        lambda _candidate: (object(), {}),
    )
    monkeypatch.setattr(generated_paper, "CombinatorialStrategy", FakeStrategy)

    first = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )
    second = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )

    assert first["orders_generated_this_cycle"] == 1
    assert first["open_positions"] == 1
    assert first["candidate_dispositions"]["a" * 64]["status"] == (
        "PAPER_ORDER_FILLED"
    )
    assert first["candidate_disposition_status_counts"] == {
        "PAPER_ORDER_FILLED": 1
    }
    assert read_json(paths["dispositions"])["snapshot_sha256"] == second[
        "candidate_disposition_snapshot_sha256"
    ]
    assert second["orders_generated_this_cycle"] == 0
    assert second["paper_orders_placed"] == 1
    assert second["candidate_dispositions"]["a" * 64]["status"] == (
        "POSITION_MANAGED"
    )
    assert second["real_orders_placed"] == 0
    assert second["real_exchange_requests"] == 0
    assert second["auto_live_promotion"] is False


def test_generated_paper_enforces_backtest_maximum_holding_bars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    index = pd.date_range(
        utc_now() - timedelta(hours=12),
        periods=4,
        freq="4h",
    )
    calls = 0

    async def frames(_settings, _candidates):
        nonlocal calls
        calls += 1
        selected = index[:3] if calls == 1 else index
        return {
            ("BTC-EUR", "4h"): pd.DataFrame(
                {"close": [98.0, 99.0, 100.0, 101.0][: len(selected)]},
                index=selected,
            )
        }

    async def price(_settings, _market):
        return Decimal("100")

    class TimeExitStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, selected):
            false = pd.Series(False, index=selected.index)
            entry = false.copy()
            entry.iloc[2] = True
            distance = pd.Series(5.0, index=selected.index)
            return StrategyOutput(
                entry=entry,
                exit=false,
                avoid=false,
                reduce=false,
                stop_distance=distance,
                target_distance=pd.Series(
                    100_000_000.0,
                    index=selected.index,
                ),
                trailing_distance=pd.Series(0.0, index=selected.index),
                size_multiplier=pd.Series(1.0, index=selected.index),
                maximum_holding_bars=1,
                entry_reason="TEST",
                exit_reason="TIME_EXIT",
                metadata={"exit_profile": "TIME_ONLY"},
            )

    monkeypatch.setattr(
        generated_paper,
        "_combination",
        lambda _candidate: (object(), {}),
    )
    monkeypatch.setattr(
        generated_paper,
        "CombinatorialStrategy",
        TimeExitStrategy,
    )

    opened = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )
    closed = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )

    assert opened["positions"]["a" * 64]["maximum_holding_bars"] == 1
    assert closed["positions"] == {}
    assert closed["orders_generated_this_cycle"] == 1
    assert closed["last_closed_position"]["exit_reason"] == (
        "PAPER_MAXIMUM_HOLDING"
    )
    assert closed["evaluations"]["a" * 64]["closed_holding_bars"] == 1


def test_mtf_limit_overlay_uses_latest_causal_touch(
    monkeypatch,
) -> None:
    parent = MultiTimeframeParameters(
        timeframe="2h",
        entry_lookback=240,
        exit_lookback=72,
    )
    parameters = LimitOverlayParameters(
        parent=parent,
        entry_window_15m_bars=8,
    )
    latest = pd.Timestamp(utc_now()).floor("15min") - pd.Timedelta(15, unit="m")
    decision_at = latest - pd.Timedelta(15, unit="m")
    fifteen = pd.DataFrame(
        {
            "open": [101.0],
            "high": [102.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex([latest]),
    )
    parent_frame = pd.DataFrame(
        {
            "open": [100.0],
            "high": [106.0],
            "low": [98.0],
            "close": [105.0],
            "volume": [100.0],
        },
        index=pd.DatetimeIndex([latest - pd.Timedelta(2, unit="h")]),
    )
    featured = parent_frame.assign(
        decision_at=decision_at,
        entry_level=100.0,
        exit_level=90.0,
        atr=2.0,
        confirmed_fractal_low=97.0,
        entry_signal=True,
        exit_signal=False,
    )
    monkeypatch.setattr(
        generated_paper,
        "_feature_frame",
        lambda _frame, _parameters: featured,
    )
    candidate = {
        "strategy_id": parameters.strategy_id,
        "strategy_dna_hash": parameters.dna_hash,
        "timeframe": "15m",
        "markets": ["BTC-EUR"],
        "parameters": asdict(parameters),
        "paper_adapter": "MTF_15M_LIMIT_OVERLAY",
    }

    evaluations = generated_paper._mtf_limit_overlay_market_evaluations(
        candidate,
        {
            ("BTC-EUR", "15m"): fifteen,
            ("BTC-EUR", "2h"): parent_frame,
        },
    )

    assert len(evaluations) == 1
    assert evaluations[0]["entry"] is True
    assert evaluations[0]["paper_fill_price"] == 100.0
    assert evaluations[0]["limit_price"] == 100.0
    assert evaluations[0]["stop_distance"] == 3.0
    assert evaluations[0]["target_distance"] == 7.5
    assert evaluations[0]["order_policy"] == "LIMIT_NO_CHASE_NO_MARKET_FALLBACK"

    no_touch = fifteen.copy()
    no_touch.loc[latest, ["open", "high", "low", "close"]] = [
        101.5,
        102.0,
        101.0,
        101.5,
    ]
    missed = generated_paper._mtf_limit_overlay_market_evaluations(
        candidate,
        {
            ("BTC-EUR", "15m"): no_touch,
            ("BTC-EUR", "2h"): parent_frame,
        },
    )[0]
    assert missed["entry"] is False
    assert missed["stop_distance"] is None
    assert missed["target_distance"] is None
    json.dumps(missed, allow_nan=False)


def test_mtf_limit_overlay_paper_fill_is_limit_and_never_live(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    candidate = {
        "strategy_id": "LIMIT15M_TEST",
        "strategy_dna_hash": "1" * 64,
        "frozen_candidate_hash": "2" * 64,
        "economic_hypothesis_family": "CAUSAL_MTF_LIMIT_ENTRY_OVERLAY",
        "timeframe": "15m",
        "markets": ["BTC-EUR"],
        "parameters": {},
        "metrics": {
            "net_return": 0.05,
            "profit_factor": 1.4,
            "trade_count": 40,
            "stressed_net_return": 0.03,
            "holdout_net_return": 0.02,
        },
        "integrity": {
            "no_lookahead": True,
            "no_repainting": True,
            "next_open_execution": True,
            "long_only_spot": True,
        },
        "paper_adapter": "MTF_15M_LIMIT_OVERLAY",
        "lifecycle": "BACKTEST_POSITIVE",
        "auto_live_promotion": False,
    }
    monkeypatch.setattr(
        generated_paper,
        "_load_candidates",
        lambda _settings: [candidate],
    )
    monkeypatch.setattr(
        generated_paper,
        "_mtf_limit_overlay_market_evaluations",
        lambda *_args: [
            {
                "market": "BTC-EUR",
                "signal_timestamp": "2026-08-01T20:00:00+00:00",
                "entry": True,
                "exit": False,
                "stale": False,
                "paper_fill_price": 100.0,
                "limit_price": 100.0,
                "stop_distance": 3.0,
                "target_distance": 7.5,
                "risk_levels_valid": True,
                "risk_level_block_reason": None,
                "size_multiplier": 1.0,
            }
        ],
    )
    monkeypatch.setattr(
        generated_paper,
        "_notify_paper_promotions",
        lambda *_args: {"delivery_status": "SKIPPED_DUPLICATE"},
    )
    monkeypatch.setattr(
        generated_paper,
        "_notify_fill",
        lambda *_args, **_kwargs: None,
    )

    async def frames(_settings, _candidates):
        return {}

    async def forbidden_ticker(_settings, _market):
        raise AssertionError("a touched paper limit must not chase the ticker")

    result = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=forbidden_ticker,
        )
    )

    position = result["positions"]["1" * 64]
    ledger_rows = [
        json.loads(line)
        for line in generated_paper._paths(settings)["ledger"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    intent = next(row for row in ledger_rows if row["event_type"] == "ORDER_INTENT")
    assert Decimal(position["entry_price"]) == Decimal("100")
    assert intent["payload"]["order_type"] == "LIMIT"
    assert intent["payload"]["limit_price"] == "100.0"
    assert result["real_orders_placed"] == 0
    assert result["real_exchange_requests"] == 0
    assert result["auto_live_promotion"] is False


def test_indicator_exit_paper_position_does_not_require_hidden_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    index = pd.date_range(
        utc_now() - timedelta(hours=12),
        periods=3,
        freq="4h",
    )
    frame = pd.DataFrame({"close": [98.0, 99.0, 100.0]}, index=index)

    async def frames(_settings, _candidates):
        return {("BTC-EUR", "4h"): frame}

    async def price(_settings, _market):
        return Decimal("100")

    class IndicatorExitStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, selected):
            false = pd.Series(False, index=selected.index)
            return StrategyOutput(
                entry=pd.Series([False, False, True], index=selected.index),
                exit=false,
                avoid=false,
                reduce=false,
                stop_distance=pd.Series(5.0, index=selected.index),
                target_distance=pd.Series(100_000_000.0, index=selected.index),
                trailing_distance=pd.Series(0.0, index=selected.index),
                size_multiplier=pd.Series(1.0, index=selected.index),
                maximum_holding_bars=None,
                entry_reason="TEST",
                exit_reason="SUPERTREND_EXIT",
                metadata={"exit_profile": "SUPERTREND_EXIT"},
            )

    monkeypatch.setattr(
        generated_paper,
        "_combination",
        lambda _candidate: (object(), {}),
    )
    monkeypatch.setattr(
        generated_paper,
        "CombinatorialStrategy",
        IndicatorExitStrategy,
    )

    first = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )
    second = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )

    position = first["positions"]["a" * 64]
    evaluation = first["evaluations"]["a" * 64]["markets"][0]
    assert evaluation["risk_levels_valid"] is True
    assert evaluation["target_required"] is False
    assert position["take_profit_1"] is None
    assert position["take_profit_2"] is None
    assert second["open_positions"] == 1
    assert second["orders_generated_this_cycle"] == 0
    assert second["real_orders_placed"] == 0


def test_missing_paper_ticker_is_isolated_per_market(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    index = pd.date_range(
        utc_now() - timedelta(hours=12),
        periods=3,
        freq="4h",
    )
    frame = pd.DataFrame({"close": [98.0, 99.0, 100.0]}, index=index)

    async def frames(_settings, _candidates):
        return {("BTC-EUR", "4h"): frame}

    async def missing_price(_settings, market):
        raise ValueError(f"TICKER_PRICE_MISSING:{market}")

    class FakeStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, selected):
            false = pd.Series(False, index=selected.index)
            distance = pd.Series(5.0, index=selected.index)
            return StrategyOutput(
                entry=pd.Series([False, False, True], index=selected.index),
                exit=false,
                avoid=false,
                reduce=false,
                stop_distance=distance,
                target_distance=distance,
                trailing_distance=pd.Series(0.0, index=selected.index),
                size_multiplier=pd.Series(1.0, index=selected.index),
                maximum_holding_bars=120,
                entry_reason="TEST",
                exit_reason="TEST_EXIT",
            )

    monkeypatch.setattr(
        generated_paper,
        "_combination",
        lambda _candidate: (object(), {}),
    )
    monkeypatch.setattr(generated_paper, "CombinatorialStrategy", FakeStrategy)

    result = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=missing_price,
        )
    )

    assert result["status"] == "READY"
    assert result["orders_generated_this_cycle"] == 0
    evaluation = result["evaluations"]["a" * 64]
    assert evaluation["price_status"] == "UNAVAILABLE"
    assert evaluation["unavailable_markets"] == ["BTC-EUR"]
    disposition = result["candidate_dispositions"]["a" * 64]
    assert disposition["status"] == "PRICE_UNAVAILABLE"
    assert disposition["natural_entry_signal_count"] == 1


def test_generated_paper_explains_market_overlap_without_new_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    atomic_write_json(
        paths["state"],
        {
            "schema_version": "generated_strategy_paper_v1",
            "status": "ACTIVE",
            "positions": {
                "legacy-management-dna": {
                    "market": "BTC-EUR",
                    "paper_only": True,
                }
            },
            "evaluations": {},
            "promoted_dna": [],
        },
    )
    index = pd.date_range(
        utc_now() - timedelta(hours=12),
        periods=3,
        freq="4h",
    )
    frame = pd.DataFrame({"close": [98.0, 99.0, 100.0]}, index=index)

    async def frames(_settings, _candidates):
        return {("BTC-EUR", "4h"): frame}

    async def price(_settings, _market):
        raise AssertionError("occupied entry must not request a ticker")

    class EntryStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, selected):
            false = pd.Series(False, index=selected.index)
            return StrategyOutput(
                entry=pd.Series([False, False, True], index=selected.index),
                exit=false,
                avoid=false,
                reduce=false,
                stop_distance=pd.Series(5.0, index=selected.index),
                target_distance=pd.Series(10.0, index=selected.index),
                trailing_distance=pd.Series(0.0, index=selected.index),
                size_multiplier=pd.Series(1.0, index=selected.index),
                maximum_holding_bars=120,
                entry_reason="TEST",
                exit_reason="TEST_EXIT",
            )

    monkeypatch.setattr(
        generated_paper,
        "_combination",
        lambda _candidate: (object(), {}),
    )
    monkeypatch.setattr(
        generated_paper,
        "CombinatorialStrategy",
        EntryStrategy,
    )

    result = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
            price_loader=price,
        )
    )

    disposition = result["candidate_dispositions"]["a" * 64]
    assert disposition["status"] == "MARKET_OCCUPIED_BY_OTHER_DNA"
    assert disposition["natural_entry_markets"] == ["BTC-EUR"]
    assert disposition["occupied_entry_markets"] == ["BTC-EUR"]
    assert result["orders_generated_this_cycle"] == 0
    assert result["real_orders_placed"] == 0
    assert result["real_exchange_requests"] == 0


def test_current_price_accepts_normalized_bitvavo_last_price(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)

    class Record:
        values = {
            "last_price": "54319.25",
            "best_bid": "54319.00",
            "best_ask": "54319.50",
        }

    class Loader:
        def __init__(self, _settings):
            pass

        async def download_ticker(self, **kwargs):
            assert kwargs["provider"] == "bitvavo"
            assert kwargs["market"] == "BTC-EUR"
            assert kwargs["mode"] == "paper"
            return Record()

    monkeypatch.setattr(generated_paper, "DataLoader", Loader)

    price = asyncio.run(generated_paper._current_price(settings, "BTC-EUR"))

    assert price == Decimal("54319.25")


def test_paper_promotions_send_one_summary_and_are_restart_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    paths = generated_paper._paths(settings)
    atomic_write_json(
        paths["registry"],
        {
            "schema_version": "classical_backtest_positive_v1",
            "candidates": [_candidate()],
        },
    )
    calls: list[list[str]] = []

    def notify(_settings, candidates):
        calls.append(
            [str(candidate["strategy_dna_hash"]) for candidate in candidates]
        )
        return {"delivery_status": "PENDING"}

    async def frames(_settings, _candidates):
        raise RuntimeError("expected data block")

    monkeypatch.setattr(generated_paper, "_notify_paper_promotions", notify)

    first = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
        )
    )
    second = asyncio.run(
        generated_paper.run_generated_paper_once(
            settings,
            frame_loader=frames,
        )
    )

    assert calls == [["a" * 64], []]
    assert first["notified_promotion_dna"] == ["a" * 64]
    assert second["notified_promotion_dna"] == ["a" * 64]
    assert first["real_orders_placed"] == 0
    assert second["real_orders_placed"] == 0


def test_paper_promotion_notification_failure_isolated_from_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)

    class FailingNotifier:
        def __init__(self, *_args, **_kwargs):
            raise TimeoutError

    monkeypatch.setattr(generated_paper, "TelegramNotifier", FailingNotifier)

    result = generated_paper._notify_paper_promotions(
        settings,
        [_candidate()],
    )

    assert result == {
        "delivery_status": "FAILED_ISOLATED",
        "reason_code": "PAPER_PROMOTION_NOTIFICATION_TIMEOUTERROR",
    }


def test_frozen_generated_dna_drift_is_hard_blocked(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = generated_paper.freeze_generated_candidates(settings, [_candidate()])
    changed = {**_candidate(), "frozen_candidate_hash": "f" * 64}
    second = generated_paper.freeze_generated_candidates(settings, [changed])

    assert first["added_dna"] == ["a" * 64]
    assert second["added_dna"] == []
    assert second["identity_drift_blockers"] == [
        {
            "strategy_dna_hash": "a" * 64,
            "reason": "FROZEN_IDENTITY_DRIFT",
        }
    ]


def test_semantically_identical_legacy_mtf_hash_is_migrated(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    candidate = {
        "strategy_id": "MTF_DONCHIAN_1H_D100_E180_X60_ATR4_F2",
        "strategy_dna_hash": "d" * 64,
        "timeframe": "1h",
        "markets": ["BTC-EUR", "ETH-EUR"],
        "parameters": {
            "atr_period": 14,
            "atr_stop_multiple": 4.0,
            "confirmed_fractal_span": 2,
            "daily_ema_period": 100,
            "entry_lookback": 180,
            "exit_lookback": 60,
            "reward_risk": 2.5,
            "side_cost_bps": 35.0,
            "timeframe": "1h",
            "use_confirmed_fractal_stop": True,
        },
        "paper_adapter": "MTF_DONCHIAN_ATR_FRACTAL",
        "lifecycle": "BACKTEST_POSITIVE",
    }
    current_hash = generated_paper.multi_timeframe_frozen_candidate_hash(
        candidate
    )
    candidate["frozen_candidate_hash"] = current_hash
    legacy = {
        **candidate,
        "frozen_candidate_hash": "e" * 64,
        "paper_frozen_at": "2026-07-30T20:46:08Z",
    }
    frozen_path = generated_paper._paths(settings)["frozen"]
    atomic_write_json(
        frozen_path,
        {
            "schema_version": "frozen_classical_paper_candidates_v1",
            "candidates": [legacy],
        },
    )

    result = generated_paper.freeze_generated_candidates(
        settings,
        [candidate],
    )
    stored = read_json(frozen_path)["candidates"][0]

    assert result["identity_drift_blockers"] == []
    assert result["identity_hash_migrations"] == [
        {
            "strategy_dna_hash": "d" * 64,
            "previous_frozen_candidate_hash": "e" * 64,
            "current_frozen_candidate_hash": current_hash,
            "reason": "SEMANTICALLY_IDENTICAL_MTF_HASH_SCHEMA_MIGRATION",
        }
    ]
    assert stored["frozen_candidate_hash"] == current_hash
    assert stored["previous_frozen_candidate_hash"] == "e" * 64
    assert stored["paper_frozen_at"] == "2026-07-30T20:46:08Z"


def test_simple_exact_evidence_refresh_preserves_frozen_execution_identity(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    current = generated_paper.refresh_simple_lab_positive_candidates(
        settings,
        records=[_exact_record()],
    )["candidates"][0]
    legacy = {
        **current,
        "data_hash": "older-data-snapshot",
        "feature_hash": "older-feature-snapshot",
        "frozen_candidate_hash": "e" * 64,
        "paper_frozen_at": "2026-07-30T20:46:08Z",
    }
    legacy.pop("frozen_identity_schema", None)
    frozen_path = generated_paper._paths(settings)["frozen"]
    atomic_write_json(
        frozen_path,
        {
            "schema_version": "frozen_classical_paper_candidates_v1",
            "candidates": [legacy],
        },
    )

    result = generated_paper.freeze_generated_candidates(
        settings,
        [current],
    )
    stored = read_json(frozen_path)["candidates"][0]

    assert result["identity_drift_blockers"] == []
    assert result["identity_hash_migrations"][0]["reason"] == (
        "SEMANTICALLY_IDENTICAL_EXECUTION_IDENTITY_MIGRATION"
    )
    assert stored["frozen_candidate_hash"] == (
        generated_paper.generated_paper_candidate_semantic_hash(current)
    )
    assert stored["paper_frozen_at"] == "2026-07-30T20:46:08Z"


def test_simple_exact_timeframe_drift_remains_hard_blocked(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    current = generated_paper.refresh_simple_lab_positive_candidates(
        settings,
        records=[_exact_record()],
    )["candidates"][0]
    changed_identity = {
        **current,
        "timeframe": "1h",
        "frozen_candidate_hash": "e" * 64,
    }
    changed_identity.pop("frozen_identity_schema", None)
    frozen_path = generated_paper._paths(settings)["frozen"]
    atomic_write_json(
        frozen_path,
        {
            "schema_version": "frozen_classical_paper_candidates_v1",
            "candidates": [changed_identity],
        },
    )

    result = generated_paper.freeze_generated_candidates(
        settings,
        [current],
    )

    assert result["identity_drift_blockers"] == [
        {
            "strategy_dna_hash": current["strategy_dna_hash"],
            "reason": "FROZEN_IDENTITY_DRIFT",
        }
    ]


def test_simple_lab_auto_block_reconstructs_for_paper() -> None:
    blocks = (
        "adx_trend_strength",
        "auto__market_structure_equal_highs__equal_highs__true__entry_trigger",
        "htf_1d_regime_bullish",
    )
    registry = generated_paper.registry_driven_signal_blocks()
    generated = generated_paper.CombinationGenerator(registry).generate(
        sizes=(len(blocks),),
        logic_modes=(generated_paper.LogicMode.LAYERED,),
        block_ids=blocks,
        timeframes=("1h",),
    )
    expected = next(row for row in generated if row.block_ids == tuple(sorted(blocks)))
    candidate = {
        "strategy_dna_hash": "variant-dna",
        "block_strategy_dna_hash": expected.strategy_dna_hash,
        "block_ids": list(blocks),
        "logic_mode": "LAYERED",
        "timeframe": "1h",
    }

    reconstructed, returned_registry = generated_paper._combination(candidate)

    assert reconstructed.strategy_dna_hash == expected.strategy_dna_hash
    assert blocks[1] in returned_registry


def test_atr_turtle_adapter_uses_prior_channel_and_closed_latest_bar() -> None:
    parameters = AtrRiskBreakoutParameters()
    index = pd.date_range(
        end=pd.Timestamp(utc_now()).floor("4h") - pd.Timedelta(4, unit="h"),
        periods=700,
        freq="4h",
    )
    close = pd.Series(
        [100.0 + index_value * 0.01 for index_value in range(700)],
        index=index,
    )
    close.iloc[-1] = float(close.iloc[-121:-1].max() + 5.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    candidate = {
        "strategy_dna_hash": parameters.dna_hash,
        "timeframe": "4h",
        "markets": ["BTC-EUR"],
        "parameters": asdict(parameters),
    }

    evaluations = generated_paper._atr_turtle_market_evaluations(
        candidate,
        {("BTC-EUR", "4h"): frame},
    )

    assert len(evaluations) == 1
    assert evaluations[0]["entry"] is True
    assert evaluations[0]["stop_distance"] > 0
