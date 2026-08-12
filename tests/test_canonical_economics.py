from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from config.settings import Settings
from reporting.canonical_economics import (
    aggregate_dimension,
    apply_mfe_mae,
    build_canonical_strategy_economics,
    build_validation_backlog,
    canonical_family,
    classify_entry_type,
    classify_exit_type,
    economic_metrics,
    promotion_table,
    reconstruct_canonical_episodes,
    signal_to_trade_funnel,
    tp_evidence,
)


def event(event_type: str, recorded_at: str, **payload: object) -> dict[str, object]:
    return {
        "event_type": event_type,
        "recorded_at": recorded_at,
        "payload": payload,
    }


def intent(
    intent_id: str,
    *,
    side: str,
    quantity: str,
    recorded_at: str,
    strategy_id: str | None = "MOMENTUM_BREAKOUT_V1",
    signal_id: str | None = "signal-1",
    reason_codes: tuple[str, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "intent_id": intent_id,
        "idempotency_key": f"key:{intent_id}",
        "market": "BTC-EUR",
        "side": side,
        "quantity": quantity,
        "order_type": "MARKET",
        "reason_codes": [*reason_codes, "PAPER_ONLY"],
    }
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id
        payload["strategy_dna_hash"] = "dna-1"
    if signal_id is not None:
        payload["signal_id"] = signal_id
    return event("ORDER_INTENT", recorded_at, **payload)


def fill(
    fill_id: str,
    intent_id: str,
    *,
    side: str,
    quantity: str,
    price: str,
    fee: str,
    recorded_at: str,
    strategy_id: str | None = "MOMENTUM_BREAKOUT_V1",
    signal_id: str | None = "signal-1",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "fill_id": fill_id,
        "intent_id": intent_id,
        "order_id": f"order:{intent_id}",
        "market": "BTC-EUR",
        "side": side,
        "quantity": quantity,
        "price": price,
        "fee_eur": fee,
        "fee_known": True,
        "filled_at": recorded_at,
    }
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id
        payload["strategy_dna_hash"] = "dna-1"
    if signal_id is not None:
        payload["signal_id"] = signal_id
    return event("FILL", recorded_at, **payload)


def completed_events(*, duplicate_entry: bool = False) -> list[dict[str, object]]:
    entry_intent = intent(
        "buy-1",
        side="BUY",
        quantity="2",
        recorded_at="2026-01-01T00:00:00Z",
    )
    entry_fill = fill(
        "fill-buy",
        "buy-1",
        side="BUY",
        quantity="2",
        price="100",
        fee="1",
        recorded_at="2026-01-01T00:00:01Z",
    )
    rows = [entry_intent, entry_fill]
    if duplicate_entry:
        rows.append(deepcopy(entry_fill))
    rows.extend(
        [
            intent(
                "sell-1",
                side="SELL",
                quantity="1",
                recorded_at="2026-01-01T00:05:00Z",
                reason_codes=("TAKE_PROFIT_1",),
            ),
            fill(
                "fill-sell-1",
                "sell-1",
                side="SELL",
                quantity="1",
                price="110",
                fee="0.5",
                recorded_at="2026-01-01T00:05:01Z",
            ),
            intent(
                "sell-2",
                side="SELL",
                quantity="1",
                recorded_at="2026-01-01T00:10:00Z",
                reason_codes=("TAKE_PROFIT_2",),
            ),
            fill(
                "fill-sell-2",
                "sell-2",
                side="SELL",
                quantity="1",
                price="120",
                fee="0.5",
                recorded_at="2026-01-01T00:10:01Z",
            ),
        ]
    )
    return rows


def test_roundtrip_partial_exit_fees_ownership_and_causal_dimensions() -> None:
    snapshots = {
        "signal-1": {
            "decision_timestamp": "2026-01-01T00:00:00Z",
            "values": {
                "context_timeframe": "1h",
                "macro_regime": "RISK_ON",
                "entry_price": 100,
            },
        }
    }
    state, episodes, unknowns = reconstruct_canonical_episodes(
        completed_events(duplicate_entry=True),
        causal_snapshots=snapshots,
    )

    assert not unknowns
    assert len(state.fills) == 3
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.reconstruction_status == "CANONICALLY_RECONSTRUCTABLE"
    assert episode.strategy_family == "MOMENTUM"
    assert episode.strategy_id == "MOMENTUM_BREAKOUT_V1"
    assert episode.market == "BTC-EUR"
    assert episode.timeframe == "1h"
    assert episode.regime == "RISK_ON"
    assert episode.entry_quantity == episode.exit_quantity == Decimal("2")
    assert episode.gross_pnl_before_costs_eur == Decimal("30")
    assert episode.fees_eur == Decimal("2.0")
    assert episode.net_pnl_eur == Decimal("28.0")
    assert episode.exit_type == "MIXED"
    assert episode.exit_reason_codes == ("TAKE_PROFIT_1", "TAKE_PROFIT_2")
    assert economic_metrics(episodes)["cost_classification"] == (
        "GROSS_POSITIVE_NET_POSITIVE"
    )
    assert aggregate_dimension(episodes, "market")[0]["dimension_value"] == (
        "BTC-EUR"
    )


def test_taxonomy_entry_exit_promotion_and_unknown_owner() -> None:
    assert canonical_family("VWAP_RECLAIM_V1") == (
        "MEAN_REVERSION",
        ("ORDERFLOW",),
    )
    assert canonical_family("FAILED_BREAKDOWN_REVERSAL_V1") == (
        "FAILED_BREAKDOWN_REVERSAL",
        ("MEAN_REVERSION",),
    )
    assert canonical_family("FAILED_BREAKOUT_REVERSAL_V1") == (
        "FAILED_BREAKOUT_REVERSAL",
        ("MEAN_REVERSION",),
    )
    assert classify_entry_type("FAILED_BREAKDOWN_REVERSAL_V1") == (
        "FAILED_BREAKDOWN_REVERSAL"
    )
    assert classify_entry_type("FAILED_BREAKOUT_REVERSAL_V1") == (
        "FAILED_BREAKOUT_REVERSAL"
    )
    assert classify_exit_type(["TIME_STOP"]) == "TIME_EXIT"

    rows = [
        {
            "dimension_value": "NEGATIVE",
            "closed_episode_count": 30,
            "net_pnl_eur": "-1",
            "net_expectancy_eur": "-0.03",
            "profit_factor": 0.8,
        },
        {
            "dimension_value": "PROMISING",
            "closed_episode_count": 30,
            "net_pnl_eur": "1",
            "net_expectancy_eur": "0.03",
            "profit_factor": 1.2,
        },
        {
            "dimension_value": "TINY",
            "closed_episode_count": 2,
            "net_pnl_eur": "1",
            "net_expectancy_eur": "0.5",
            "profit_factor": 2.0,
        },
    ]
    statuses = {
        row["strategy_family"]: row["promotion_status"]
        for row in promotion_table(rows)
    }
    assert statuses == {
        "NEGATIVE": "BLOCKED_NEGATIVE_EXPECTANCY",
        "PROMISING": "PAPER_POSITIVE",
        "TINY": "INSUFFICIENT_SAMPLE",
    }

    unknown_events = [
        intent(
            "unknown-buy",
            side="BUY",
            quantity="1",
            recorded_at="2026-01-01T00:00:00Z",
            strategy_id=None,
            signal_id=None,
        ),
        fill(
            "unknown-entry",
            "unknown-buy",
            side="BUY",
            quantity="1",
            price="100",
            fee="0",
            recorded_at="2026-01-01T00:00:01Z",
            strategy_id=None,
            signal_id=None,
        ),
        intent(
            "unknown-sell",
            side="SELL",
            quantity="1",
            recorded_at="2026-01-01T00:05:00Z",
            strategy_id=None,
            signal_id=None,
        ),
        fill(
            "unknown-exit",
            "unknown-sell",
            side="SELL",
            quantity="1",
            price="101",
            fee="0",
            recorded_at="2026-01-01T00:05:01Z",
            strategy_id=None,
            signal_id=None,
        ),
    ]
    _, episodes, _ = reconstruct_canonical_episodes(unknown_events)
    assert episodes[0].strategy_id == "UNKNOWN_OWNER"
    assert episodes[0].strategy_family == "UNKNOWN"
    assert episodes[0].reconstruction_status == "PARTIALLY_RECONSTRUCTABLE"


def test_signal_trade_funnel_and_tp_semantics_are_separate(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle.jsonl"
    lifecycle.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"opportunity_id": "signal-1", "to_state": "WATCHING"},
                {"opportunity_id": "signal-1", "to_state": "ENTRY_READY"},
                {"opportunity_id": "signal-other", "to_state": "WATCHING"},
            )
        ),
        encoding="utf-8",
    )
    events = completed_events()
    state, episodes, _ = reconstruct_canonical_episodes(events)
    funnel = signal_to_trade_funnel(
        lifecycle,
        events,
        episodes,
        canonical_state=state,
    )
    conversions = [
        row["conversion_from_previous"]
        for row in funnel["stages"]
        if row["conversion_from_previous"] is not None
    ]
    assert all(0 <= value <= 1 for value in conversions)

    telegram = tmp_path / "telegram.jsonl"
    telegram.write_text('{"message":"proposed TP2 120"}\n', encoding="utf-8")
    result = tp_evidence(
        episodes,
        telegram_ledger_path=telegram,
        causal_snapshot_count=1,
        mfe_mae={"covered_episode_count": 0},
    )
    assert result["telegram_claim_status"] == (
        "NOT_EVALUABLE_FROM_TELEGRAM_LEDGER"
    )
    assert result["actual_executed_trade_outcomes"][
        "tp2_observed_episode_count"
    ] == 1
    assert result["signal_outcome_evaluator"][
        "signal_outcome_is_executed_trade_outcome"
    ] is False


def test_mfe_mae_uses_only_closed_candles_inside_holding_window(
    tmp_path: Path,
) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    _, episodes, _ = reconstruct_canonical_episodes(completed_events())
    target = tmp_path / "BTC-EUR"
    target.mkdir(parents=True)
    table = pyarrow.Table.from_pylist(
        [
            {
                "timestamp": "2026-01-01T00:05:00Z",
                "closed": True,
                "values": {"high": 115.0, "low": 95.0},
            },
            {
                "timestamp": "2026-01-01T00:11:00Z",
                "closed": True,
                "values": {"high": 999.0, "low": 1.0},
            },
        ]
    )
    parquet.write_table(table, target / "1m.parquet")

    result = apply_mfe_mae(episodes, tmp_path)

    assert result["covered_episode_count"] == 1
    assert episodes[0].mfe_pct == Decimal("15.00")
    assert episodes[0].mae_pct == Decimal("-5.00")
    assert episodes[0].market_path_evidence == "CAUSAL_CLOSED_1M_CANDLES"


def test_immutable_artifact_rerun_has_same_identity(
    tmp_path: Path,
    isolated_settings: Settings,
) -> None:
    output = tmp_path / "output"
    paper = output / "paper" / "ledger.jsonl"
    paper.parent.mkdir(parents=True)
    events = completed_events()
    paper.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )
    lifecycle = output / "live" / "events" / "lifecycle.jsonl"
    lifecycle.parent.mkdir(parents=True)
    lifecycle.write_text("", encoding="utf-8")
    paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": output,
            "processed_data_dir": tmp_path / "normalized",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": paths})

    first = build_canonical_strategy_economics(
        settings,
        paper_ledger_path=paper,
        lifecycle_path=lifecycle,
        include_mfe_mae=False,
    )
    second = build_canonical_strategy_economics(
        settings,
        paper_ledger_path=paper,
        lifecycle_path=lifecycle,
        include_mfe_mae=False,
    )

    assert first["run_id"] == second["run_id"]
    assert first["artifact_hash"] == second["artifact_hash"]
    artifact = json.loads(Path(first["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["replay_deterministic"] is True
    assert artifact["closed_episode_count"] == 1
    assert artifact["evidence_hashes"]["paper_execution_ledger_sha256"] == (
        sha256(paper.read_bytes()).hexdigest()
    )
    assert artifact["evidence_hashes"]["paper_execution_ledger_byte_count"] == (
        paper.stat().st_size
    )
    assert artifact["evidence_layers"]["PAPER"]["net_pnl_eur"] == "28.0"
    assert artifact["evidence_layers"]["LIVE"]["live_validated_family_count"] == 0
    assert artifact["safety"]["private_bitvavo_mutations"] == 0
    latest = json.loads((output / "economics" / "latest.json").read_text())
    assert latest["paper_30d_net_pnl_eur"] == "28.0"
    assert latest["live_canary_net_pnl_eur"] is None
    assert latest["best_validated_family"] is None


def test_default_economics_consolidates_canonical_paper_ledgers(
    tmp_path: Path,
    isolated_settings: Settings,
) -> None:
    output = tmp_path / "output"
    paper_directory = output / "paper"
    paper_directory.mkdir(parents=True)
    first = completed_events()
    second = json.loads(
        json.dumps(completed_events())
        .replace("2026-01-01", "2026-01-02")
        .replace("buy-1", "buy-2")
        .replace("sell-1", "sell-3")
        .replace("sell-2", "sell-4")
        .replace("fill-buy", "fill-buy-2")
        .replace("fill-sell-1", "fill-sell-3")
        .replace("fill-sell-2", "fill-sell-4")
    )
    for path, rows in (
        (paper_directory / "event_driven_playbook_execution.jsonl", first),
        (paper_directory / "generated_strategy_execution.jsonl", second),
    ):
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    paths = isolated_settings.paths.model_copy(
        update={
            "output_dir": output,
            "processed_data_dir": tmp_path / "normalized",
        }
    )
    settings = isolated_settings.model_copy(update={"paths": paths})

    result = build_canonical_strategy_economics(settings, include_mfe_mae=False)
    artifact = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))

    assert artifact["closed_episode_count"] == 2
    assert artifact["source_event_count"] == 12
    assert len(artifact["evidence_hashes"]["paper_execution_ledgers"]) == 2
    assert artifact["evidence_layers"]["PAPER"]["closed_episode_count"] == 2


def test_validation_backlog_prefers_exact_strategy_dna_evidence() -> None:
    dna = "a" * 64
    registry = {
        "economic_evidence": [
            {
                "strategy_id": "FULL_RESEARCH_STRATEGY_NAME",
                "strategy_dna": dna,
                "backtest_positive": True,
                "sample_count": 100,
                "normal_profit_factor": 1.4,
            }
        ]
    }
    dna_economics = [
        {
            "dimension_value": dna,
            "closed_episode_count": 12,
            "net_pnl_eur": "3.0",
        }
    ]

    backlog = build_validation_backlog(registry, [], dna_economics)

    assert backlog[0]["canonical_paper_episode_count"] == 12
    assert backlog[0]["rank_inputs"]["paper_evidence_available"] is True
