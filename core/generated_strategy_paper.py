"""Restart-safe paper execution for exact-positive generated strategy DNA.

This module deliberately has no live broker path.  Generated DNA first proves
its complete entry/position/exit lifecycle in a separate paper ledger.  Live
authority remains a different, per-DNA operator-controlled concern.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import pandas as pd

from config.settings import Settings
from core.contracts import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
)
from core.swing_trading import execution_timeframe_allowed
from data.data_loader import TIMEFRAME_SECONDS, DataLoader, NormalizedDataRecord
from data.database import Database
from execution.execution import ExecutionMarketRules, PaperBroker
from notifications.telegram import TelegramNotifier
from research.combinatorial_lab import (
    CombinationGenerator,
    CombinatorialStrategy,
    LogicMode,
)
from research.features import FeaturePipeline
from research.mtf_limit_overlay import (
    LimitOverlayParameters,
    _overlay_stop,
    load_validated_mtf_limit_overlay_candidates,
)
from research.multi_timeframe_authority import (
    MultiTimeframeParameters,
    _feature_frame,
    load_validated_multi_timeframe_candidates,
    multi_timeframe_frozen_candidate_hash,
)
from research.simple_strategy_lab import registry_driven_signal_blocks
from research.volume_strategy_campaign import volume_strategy_adapter
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    sha256_file,
    stable_hash,
    utc_iso,
    utc_now,
)

FrameLoader = Callable[
    [Settings, Sequence[Mapping[str, Any]]],
    Awaitable[dict[tuple[str, str], pd.DataFrame]],
]
PriceLoader = Callable[[Settings, str], Awaitable[Decimal]]


def _paper_market_allowed(settings: Settings, market: object) -> bool:
    """Allow causal paper research beyond the narrower live launch universe."""

    normalized = str(market or "").strip().upper().replace("/", "-")
    if normalized != str(market or "") or not normalized.endswith("-EUR"):
        return False
    return settings.shariah.eligibility(normalized).status.value == "ALLOWED"


def _paths(settings: Settings) -> dict[str, Path]:
    paper = settings.paths.output_dir / "paper"
    strategies = settings.paths.output_dir / "strategies"
    autopilot = settings.paths.output_dir / "autopilot"
    paper.mkdir(parents=True, exist_ok=True)
    strategies.mkdir(parents=True, exist_ok=True)
    autopilot.mkdir(parents=True, exist_ok=True)
    ledger = paper / "generated_strategy_execution.jsonl"
    ledger.touch(exist_ok=True)
    return {
        "registry": strategies / "classical_backtest_positive.json",
        "simple_registry": strategies / "simple_lab_backtest_positive.json",
        "frozen": strategies / "frozen_classical_paper_candidates.json",
        "state": paper / "generated_strategy_state.json",
        "dispositions": paper / "generated_strategy_dispositions_latest.json",
        "ledger": ledger,
        "promotions": autopilot / "promotions.jsonl",
        "volume_catalog": (
            settings.paths.lab_dir
            / "reports"
            / "volume_strategy_catalog_campaign_v1.csv"
        ),
    }


def _load_volume_catalog_paper_candidates(
    settings: Settings,
) -> list[dict[str, Any]]:
    """Bridge robust legacy volume rows to a separately hashed paper adapter.

    The historical campaign used an unbounded 20%-exposure sleeve.  Its
    results therefore remain evidence for the legacy signal only.  Paper uses
    a new canonical DNA with an explicit stop and disaster target, and must
    collect fresh forward evidence before it can ever be considered for live.
    """

    source_path = _paths(settings)["volume_catalog"]
    if not source_path.is_file():
        return []
    source_hash = sha256_file(source_path)
    try:
        rows = pd.read_csv(source_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return []
    required = {
        "strategy_id",
        "strategy_dna_hash",
        "market",
        "timeframe",
        "full_net_return",
        "full_profit_factor",
        "full_trade_entries",
        "full_maximum_drawdown",
        "stressed_full_net_return",
        "validation_net_return",
        "confirmation_net_return",
    }
    if not required.issubset(rows.columns):
        return []
    candidates: list[dict[str, Any]] = []
    for raw in rows.to_dict(orient="records"):
        market = str(raw.get("market") or "")
        timeframe = str(raw.get("timeframe") or "")
        if not _paper_market_allowed(settings, market) or timeframe not in {
            "1h",
            "4h",
        }:
            continue
        try:
            net_return = float(raw["full_net_return"])
            profit_factor = float(raw["full_profit_factor"])
            trade_count = int(raw["full_trade_entries"])
            stressed_return = float(raw["stressed_full_net_return"])
            validation_return = float(raw["validation_net_return"])
            confirmation_return = float(raw["confirmation_net_return"])
            maximum_drawdown = abs(float(raw["full_maximum_drawdown"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not (
            net_return > 0.0
            and profit_factor > 1.0
            and trade_count >= 8
            and stressed_return > 0.0
            and validation_return > 0.0
            and confirmation_return > 0.0
        ):
            continue
        strategy_id = str(raw["strategy_id"])
        try:
            adapter = volume_strategy_adapter(strategy_id)
        except (KeyError, TypeError, ValueError):
            continue
        legacy_dna = str(raw["strategy_dna_hash"])
        if (
            adapter.legacy_strategy_dna_hash != legacy_dna
            or adapter.row.market != market
            or adapter.row.timeframe != timeframe
        ):
            continue
        parameters = dict(adapter.defaults)
        canonical_dna = adapter.canonical_adapter_dna_hash
        frozen_hash = stable_hash(
            {
                "strategy_id": strategy_id,
                "legacy_strategy_dna_hash": legacy_dna,
                "canonical_adapter_dna_hash": canonical_dna,
                "market": market,
                "timeframe": timeframe,
                "parameters": parameters,
                "source_csv_sha256": source_hash,
                "paper_adapter": "VOLUME_CATALOG_BOUNDED_RISK",
            },
            length=64,
        )
        candidates.append(
            {
                "strategy_id": strategy_id,
                "strategy_dna_hash": canonical_dna,
                "source_strategy_dna_hash": legacy_dna,
                "frozen_candidate_hash": frozen_hash,
                "economic_hypothesis_family": (
                    f"VOLUME_{adapter.row.archetype}"
                ),
                "timeframe": timeframe,
                "markets": [market],
                "parameters": parameters,
                "metrics": {
                    "net_return": net_return,
                    "profit_factor": profit_factor,
                    "net_expectancy_r": net_return / trade_count,
                    "maximum_drawdown": maximum_drawdown,
                    "trade_count": trade_count,
                    "stressed_net_return": stressed_return,
                    "validation_net_return": validation_return,
                    "confirmation_net_return": confirmation_return,
                },
                "integrity": {
                    "no_lookahead": True,
                    "no_repainting": True,
                    "next_open_execution": True,
                    "long_only_spot": True,
                },
                "lifecycle": "BACKTEST_POSITIVE_SIGNAL_LEGACY",
                "paper_eligibility": "PAPER_FORWARD_ADAPTER_VALIDATION",
                "paper_adapter": "VOLUME_CATALOG_BOUNDED_RISK",
                "adapter_validation_mode": "PAPER_FORWARD_ONLY",
                "material_difference_reason": (
                    adapter.material_difference_reason
                ),
                "source_report": str(source_path),
                "source_csv_sha256": source_hash,
                "paper_risk_multiplier": 0.25,
                "capital_scaling_warnings": [
                    "CANONICAL_BOUNDED_RISK_ADAPTER_NOT_HISTORICALLY_RETESTED",
                    "LEGACY_SIGNAL_EVIDENCE_ONLY",
                    "PROSPECTIVE_SAMPLE_INSUFFICIENT",
                ],
                "academic_tests": "CAPITAL_SCALING_WARNINGS",
                "auto_live_promotion": False,
            }
        )
    candidates.sort(
        key=lambda row: (
            float((row.get("metrics") or {}).get("profit_factor") or 0.0),
            float((row.get("metrics") or {}).get("net_return") or 0.0),
        ),
        reverse=True,
    )
    return candidates


def generated_paper_candidate_semantic_hash(
    candidate: Mapping[str, Any],
) -> str:
    """Hash immutable execution semantics, never rolling research evidence.

    Market-data and feature hashes change when a causal backtest receives new
    closed candles.  Those hashes belong to the evidence snapshot, not to the
    frozen strategy identity.  Mixing both concepts previously suspended an
    otherwise unchanged paper strategy after every data refresh.
    """

    return stable_hash(
        {
            "strategy_id": candidate.get("strategy_id"),
            "strategy_dna_hash": candidate.get("strategy_dna_hash"),
            "block_strategy_dna_hash": candidate.get(
                "block_strategy_dna_hash"
            ),
            "combination_id": candidate.get("combination_id"),
            "block_ids": sorted(
                str(value) for value in candidate.get("block_ids") or []
            ),
            "logic_mode": candidate.get("logic_mode"),
            "parameters": candidate.get("parameters") or {},
            "parameter_hash": candidate.get("parameter_hash"),
            "exit_model_version": candidate.get("exit_model_version"),
            "timeframe": candidate.get("timeframe"),
            "markets": sorted(
                str(value) for value in candidate.get("markets") or []
            ),
            "paper_adapter": candidate.get("paper_adapter"),
        },
        length=64,
    )


def _candidate_from_exact_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build one immutable paper candidate from a normal exact result family."""

    payloads = [
        dict(record.get("payload") or record)
        for record in records
        if isinstance(record, Mapping)
    ]
    by_source = {
        str(payload.get("source") or "").upper(): payload
        for payload in payloads
    }
    normal = by_source.get("NORMAL") or by_source.get("EXACT_REAL")
    if normal is None:
        return None
    metrics = dict(normal.get("metrics") or {})
    integrity = dict(normal.get("integrity") or {})
    data_period = dict(normal.get("data_period") or {})
    try:
        history_days = int(
            (
                pd.Timestamp(data_period["end"])
                - pd.Timestamp(data_period["start"])
            ).total_seconds()
            // 86_400
        )
    except (KeyError, TypeError, ValueError):
        history_days = 0
    trade_count = int(metrics.get("trade_count") or 0)
    source_type = str(normal.get("source_type") or "").upper()
    if not (
        str(normal.get("result_type") or "").upper() == "EXACT_BACKTEST"
        and source_type == "REAL_PROVIDER_DATA"
        and float(metrics.get("net_return") or 0.0) > 0.0
        and float(metrics.get("profit_factor") or 0.0) > 1.0
        and float(metrics.get("net_expectancy_r") or 0.0) > 0.0
        and trade_count >= 8
        and history_days >= 365
        and integrity.get("no_lookahead") is True
        and integrity.get("no_repainting") is True
        and integrity.get("next_open_execution") is True
        and integrity.get("long_only_spot") is True
    ):
        return None
    block_dna = str(normal.get("strategy_dna_hash") or "")
    blocks = [str(value) for value in normal.get("block_ids") or []]
    parameters = dict(normal.get("parameters") or {})
    if not block_dna or not blocks:
        return None
    variant_dna = stable_hash(
        {
            "block_strategy_dna_hash": block_dna,
            "parameters": parameters,
            "exit_model_version": normal.get("exit_model_version"),
        },
        length=64,
    )
    stressed = dict((by_source.get("STRESSED") or {}).get("metrics") or {})
    double_cost = dict(
        (by_source.get("DOUBLE_COST") or {}).get("metrics") or {}
    )
    holdout = dict(
        (by_source.get("FINAL_HOLDOUT") or {}).get("metrics") or {}
    )
    warnings = [
        "CONTINUOUS_SIMPLE_LAB_DISCOVERY",
        "UNTOUCHED_HOLDOUT_MISSING_OR_NOT_POSITIVE",
        "PROSPECTIVE_SAMPLE_INSUFFICIENT",
    ]
    if str(normal.get("bias_label") or "").upper() == (
        "CURRENT_UNIVERSE_RETROSPECTIVE"
    ):
        warnings.append("CURRENT_UNIVERSE_RETROSPECTIVE")
    if trade_count < 100:
        warnings.append("SMALL_EXACT_SAMPLE")
    if (
        stressed
        and (
            float(stressed.get("net_return") or 0.0) <= 0.0
            or float(stressed.get("profit_factor") or 0.0) <= 1.0
        )
    ):
        warnings.append("STRESSED_COST_EDGE_NOT_POSITIVE")
    identity = {
        "strategy_id": f"SIMPLE_EXACT_{variant_dna[:16]}",
        "strategy_dna_hash": variant_dna,
        "block_strategy_dna_hash": block_dna,
        "combination_id": normal.get("combination_id"),
        "block_ids": blocks,
        "logic_mode": normal.get("logic_mode") or "LAYERED",
        "parameters": parameters,
        "parameter_hash": normal.get("parameter_hash"),
        "exit_model_version": normal.get("exit_model_version"),
        "timeframe": str((normal.get("timeframes_tested") or [""])[0]),
        "markets": list(normal.get("assets_tested") or []),
        "paper_adapter": None,
    }
    frozen_hash = generated_paper_candidate_semantic_hash(identity)
    families = sorted(
        {str(value) for value in normal.get("families") or []}
    )
    return {
        "strategy_id": identity["strategy_id"],
        "strategy_dna_hash": variant_dna,
        "block_strategy_dna_hash": block_dna,
        "frozen_candidate_hash": frozen_hash,
        "frozen_identity_schema": "EXECUTION_SEMANTICS_V2",
        "experiment_hash": normal.get("experiment_hash"),
        "combination_id": normal.get("combination_id"),
        "economic_hypothesis_family": (
            "+".join(families) if families else "GENERATED_CLASSICAL"
        ),
        "block_ids": blocks,
        "logic_mode": normal.get("logic_mode") or "LAYERED",
        "parameters": parameters,
        "parameter_hash": normal.get("parameter_hash"),
        "exit_model_version": normal.get("exit_model_version"),
        "timeframe": str(
            (normal.get("timeframes_tested") or [""])[0]
        ),
        "markets": list(normal.get("assets_tested") or []),
        "data_hash": normal.get("data_hash"),
        "feature_hash": normal.get("feature_hash"),
        "data_period": data_period,
        "history_days": history_days,
        "source_type": source_type,
        "source": "CONTINUOUS_SIMPLE_LAB_EXACT",
        "metrics": {
            **{
                key: metrics.get(key)
                for key in (
                    "net_return",
                    "cagr",
                    "profit_factor",
                    "net_expectancy_r",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "maximum_drawdown",
                    "trade_count",
                    "effective_sample_size",
                    "monte_carlo_p95_drawdown",
                )
            },
            "stressed_net_return": stressed.get("net_return"),
            "stressed_profit_factor": stressed.get("profit_factor"),
            "double_cost_net_return": double_cost.get("net_return"),
            "double_cost_profit_factor": double_cost.get("profit_factor"),
            "holdout_net_return": holdout.get("net_return"),
            "holdout_profit_factor": holdout.get("profit_factor"),
        },
        "integrity": integrity,
        "lifecycle": "BACKTEST_POSITIVE",
        "paper_eligibility": "PAPER_ELIGIBLE_EXACT_POSITIVE",
        "paper_risk_multiplier": 0.25,
        "capital_scaling_warnings": sorted(set(warnings)),
        "academic_tests": "CAPITAL_SCALING_WARNINGS",
        "auto_live_promotion": False,
    }


def refresh_simple_lab_positive_candidates(
    settings: Settings,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh the paper-only bridge from canonical exact real evidence."""

    if records is None:
        database = Database(sqlite_path=settings.paths.database_path)
        try:
            selected_records = database.fetch_recent_records(
                "exact_backtest_results",
                limit=50_000,
            )
        finally:
            database.close()
    else:
        selected_records = [dict(record) for record in records]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in selected_records:
        payload = dict(record.get("payload") or record)
        experiment_hash = str(payload.get("experiment_hash") or "")
        if not experiment_hash:
            continue
        grouped.setdefault(experiment_hash, []).append(record)
    candidates = [
        candidate
        for experiment_hash in sorted(grouped)
        if (
            candidate := _candidate_from_exact_records(
                grouped[experiment_hash]
            )
        )
        is not None
    ]
    candidates.sort(
        key=lambda candidate: (
            float(
                (candidate.get("metrics") or {}).get("profit_factor")
                or 0.0
            ),
            float(
                (candidate.get("metrics") or {}).get("net_return")
                or 0.0
            ),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "simple_lab_backtest_positive_v1",
        "updated_at": utc_iso(),
        "source": "canonical_exact_backtest_results",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "minimum_exact_trades": 8,
        "minimum_history_days": 365,
        "paper_only": True,
        "auto_live_promotion": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(_paths(settings)["simple_registry"], payload)
    return payload


def _market_rules(markets: Sequence[str]) -> dict[str, ExecutionMarketRules]:
    return {
        market: ExecutionMarketRules(
            minimum_order_value_eur=Decimal("5"),
            quantity_decimals=8,
        )
        for market in markets
        if market.endswith("-EUR")
    }


def _broker(settings: Settings, markets: Sequence[str]) -> PaperBroker:
    return PaperBroker(
        initial_balances={
            "EUR": Decimal(str(settings.paper_automation.initial_capital_eur))
        },
        market_rules=_market_rules(markets),
        fee_fraction=Decimal(str(settings.costs.default_fee)),
        slippage_bps=Decimal(str(settings.costs.slippage_bps)),
        spread_bps=Decimal(str(settings.costs.spread_bps)),
        ledger_path=_paths(settings)["ledger"],
    )


def _state(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["state"]
    if path.is_file():
        return dict(read_json(path))
    return {
        "schema_version": "generated_strategy_paper_v1",
        "status": "READY",
        "positions": {},
        "evaluations": {},
        "promoted_dna": [],
        "paper_orders_placed": 0,
        "real_orders_placed": 0,
        "real_exchange_requests": 0,
        "last_cycle_at": None,
    }


def _load_candidates(settings: Settings) -> list[dict[str, Any]]:
    path = _paths(settings)["registry"]
    registry = dict(read_json(path)) if path.is_file() else {}
    candidates = []
    for raw in registry.get("candidates") or []:
        candidate = dict(raw)
        integrity = dict(candidate.get("integrity") or {})
        metrics = dict(candidate.get("metrics") or {})
        data_period = dict(candidate.get("data_period") or {})
        try:
            history_days = int(
                (
                    pd.Timestamp(data_period["end"])
                    - pd.Timestamp(data_period["start"])
                ).total_seconds()
                // 86_400
            )
        except (KeyError, TypeError, ValueError):
            history_days = 0
        if (
            candidate.get("lifecycle") == "BACKTEST_POSITIVE"
            and float(metrics.get("net_return") or 0.0) > 0.0
            and float(metrics.get("profit_factor") or 0.0) > 1.0
            and float(metrics.get("net_expectancy_r") or 0.0) > 0.0
            and int(metrics.get("trade_count") or 0)
            >= settings.research.minimum_trades
            and history_days >= settings.research.minimum_history_days
            and integrity.get("no_lookahead") is True
            and integrity.get("no_repainting") is True
            and integrity.get("next_open_execution") is True
            and integrity.get("long_only_spot") is True
        ):
            candidates.append(candidate)
    adaptive_path = (
        settings.paths.lab_dir
        / "reports"
        / "adaptive_crypto_intraday_v1.json"
    )
    if adaptive_path.is_file():
        adaptive = dict(read_json(adaptive_path))
        for raw in adaptive.get("candidates") or []:
            row = dict(raw)
            if (
                row.get("strategy_id") != "ATR_TURTLE_4H_CORE5"
                or row.get("universe_label") != "PROMOTION_COMPATIBLE"
                or row.get("paper_eligible") is not True
            ):
                continue
            normal = dict(row.get("normal") or {})
            stressed = dict(row.get("stressed") or {})
            metrics = dict(normal.get("metrics") or {})
            stressed_metrics = dict(stressed.get("metrics") or {})
            integrity = dict(normal.get("integrity") or {})
            profit_factor = float(
                metrics.get("portfolio_period_profit_factor") or 0.0
            )
            net_return = float(metrics.get("net_return") or 0.0)
            stressed_net_return = float(
                stressed_metrics.get("net_return") or 0.0
            )
            stressed_profit_factor = float(
                stressed_metrics.get("portfolio_period_profit_factor") or 0.0
            )
            if not (
                net_return > 0.0
                and profit_factor > 1.0
                and stressed_net_return > 0.0
                and stressed_profit_factor > 1.0
                and integrity.get("no_lookahead") is True
                and integrity.get("decision_at_close_execution_next_open")
                is True
                and integrity.get("long_only_spot") is True
            ):
                continue
            parameters = dict(row.get("parameters") or {})
            dna = str(row.get("strategy_dna_hash") or "")
            frozen_hash = generated_paper_candidate_semantic_hash(
                {
                    "strategy_id": row.get("strategy_id"),
                    "strategy_dna_hash": dna,
                    "timeframe": row.get("timeframe"),
                    "markets": row.get("universe"),
                    "parameters": parameters,
                    "paper_adapter": "ATR_TURTLE_4H",
                }
            )
            candidates.append(
                {
                    "strategy_id": row.get("strategy_id"),
                    "strategy_dna_hash": dna,
                    "economic_hypothesis_family": row.get(
                        "strategy_family",
                    ),
                    "timeframe": row.get("timeframe"),
                    "markets": list(row.get("universe") or []),
                    "parameters": parameters,
                    "metrics": {
                        "net_return": net_return,
                        "profit_factor": profit_factor,
                        "stressed_net_return": stressed_metrics.get(
                            "net_return",
                        ),
                        "stressed_profit_factor": stressed_metrics.get(
                            "portfolio_period_profit_factor",
                        ),
                        "maximum_drawdown": metrics.get(
                            "maximum_drawdown",
                        ),
                        "trade_count": metrics.get(
                            "closed_position_episodes",
                        ),
                        "net_expectancy_r": (
                            net_return
                            / max(
                                1,
                                int(
                                    metrics.get(
                                        "closed_position_episodes",
                                    )
                                    or 0
                                ),
                            )
                        ),
                    },
                    "integrity": {
                        "no_lookahead": True,
                        "no_repainting": True,
                        "next_open_execution": True,
                        "long_only_spot": True,
                    },
                    "lifecycle": "BACKTEST_POSITIVE",
                    "paper_adapter": "ATR_TURTLE_4H",
                    "frozen_candidate_hash": frozen_hash,
                    "frozen_identity_schema": "EXECUTION_SEMANTICS_V2",
                    "source_report": str(adaptive_path),
                    "auto_live_promotion": False,
                }
            )
    try:
        simple_registry = refresh_simple_lab_positive_candidates(settings)
    except Exception:
        simple_path = _paths(settings)["simple_registry"]
        simple_registry = (
            dict(read_json(simple_path))
            if simple_path.is_file()
            else {"candidates": []}
        )
    candidates.extend(
        dict(candidate)
        for candidate in simple_registry.get("candidates") or []
    )
    candidates.extend(load_validated_multi_timeframe_candidates(settings))
    candidates.extend(load_validated_mtf_limit_overlay_candidates(settings))
    candidates.extend(_load_volume_catalog_paper_candidates(settings))
    deduplicated = {
        str(candidate["strategy_dna_hash"]): candidate
        for candidate in candidates
        if candidate.get("strategy_dna_hash")
    }
    return [
        deduplicated[dna]
        for dna in sorted(deduplicated)
    ]


def freeze_generated_candidates(
    settings: Settings,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Append new exact-positive DNA and fail closed on identity drift."""

    path = _paths(settings)["frozen"]
    stored = (
        dict(read_json(path))
        if path.is_file()
        else {
            "schema_version": "frozen_classical_paper_candidates_v1",
            "candidates": [],
        }
    )
    frozen_by_dna = {
        str(row["strategy_dna_hash"]): dict(row)
        for row in stored.get("candidates") or []
    }
    added: list[str] = []
    blocked: list[dict[str, str]] = []
    identity_hash_migrations: list[dict[str, str]] = []
    for raw in candidates:
        candidate = dict(raw)
        dna = str(candidate.get("strategy_dna_hash") or "")
        frozen_hash = str(candidate.get("frozen_candidate_hash") or "")
        if not dna or not frozen_hash:
            blocked.append({"strategy_dna_hash": dna, "reason": "INCOMPLETE_FROZEN_IDENTITY"})
            continue
        existing = frozen_by_dna.get(dna)
        if existing is not None:
            if str(existing.get("frozen_candidate_hash")) != frozen_hash:
                is_mtf = (
                    candidate.get("paper_adapter")
                    == "MTF_DONCHIAN_ATR_FRACTAL"
                    and existing.get("paper_adapter")
                    == "MTF_DONCHIAN_ATR_FRACTAL"
                )
                is_execution_semantic_v2 = (
                    candidate.get("frozen_identity_schema")
                    == "EXECUTION_SEMANTICS_V2"
                    and (
                        candidate.get("source")
                        == "CONTINUOUS_SIMPLE_LAB_EXACT"
                        or candidate.get("paper_adapter")
                        == "ATR_TURTLE_4H"
                    )
                )
                expected_hash = (
                    multi_timeframe_frozen_candidate_hash(candidate)
                    if is_mtf
                    else generated_paper_candidate_semantic_hash(candidate)
                    if is_execution_semantic_v2
                    else None
                )
                existing_semantic_hash = (
                    multi_timeframe_frozen_candidate_hash(existing)
                    if is_mtf
                    else generated_paper_candidate_semantic_hash(existing)
                    if is_execution_semantic_v2
                    else None
                )
                if (
                    (is_mtf or is_execution_semantic_v2)
                    and frozen_hash == expected_hash
                    and existing_semantic_hash == expected_hash
                ):
                    previous_hash = str(
                        existing.get("frozen_candidate_hash") or ""
                    )
                    migrated = {
                        **candidate,
                        "paper_frozen_at": existing.get("paper_frozen_at"),
                        "previous_frozen_candidate_hash": previous_hash,
                        "frozen_identity_hash_migrated_at": utc_iso(),
                        "frozen_identity_hash_migration_reason": (
                            "DETERMINISTIC_MTF_IDENTITY_HASH_SCHEMA_V2"
                            if is_mtf
                            else "EXECUTION_SEMANTICS_IDENTITY_SCHEMA_V2"
                        ),
                        "auto_live_promotion": False,
                    }
                    frozen_by_dna[dna] = migrated
                    identity_hash_migrations.append(
                        {
                            "strategy_dna_hash": dna,
                            "previous_frozen_candidate_hash": previous_hash,
                            "current_frozen_candidate_hash": frozen_hash,
                            "reason": (
                                "SEMANTICALLY_IDENTICAL_MTF_HASH_SCHEMA_MIGRATION"
                                if is_mtf
                                else "SEMANTICALLY_IDENTICAL_EXECUTION_IDENTITY_MIGRATION"
                            ),
                        }
                    )
                else:
                    blocked.append(
                        {
                            "strategy_dna_hash": dna,
                            "reason": "FROZEN_IDENTITY_DRIFT",
                        }
                    )
            continue
        if (
            candidate.get("paper_adapter")
            == "MTF_DONCHIAN_ATR_FRACTAL"
            and frozen_hash
            != multi_timeframe_frozen_candidate_hash(candidate)
        ):
            blocked.append(
                {
                    "strategy_dna_hash": dna,
                    "reason": "INVALID_MTF_FROZEN_IDENTITY_HASH",
                }
            )
            continue
        candidate["paper_frozen_at"] = utc_iso()
        candidate["auto_live_promotion"] = False
        frozen_by_dna[dna] = candidate
        added.append(dna)
    payload = {
        "schema_version": "frozen_classical_paper_candidates_v1",
        "updated_at": utc_iso(),
        "candidate_count": len(frozen_by_dna),
        "candidates": [frozen_by_dna[key] for key in sorted(frozen_by_dna)],
        "identity_drift_blockers": blocked,
        "identity_hash_migrations": identity_hash_migrations,
        "auto_live_promotion": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(path, payload)
    return {**payload, "added_dna": added}


def _records_frame(
    records: Sequence[NormalizedDataRecord],
    *,
    market: str,
    timeframe: str,
    maximum_rows: int = 3_000,
) -> pd.DataFrame:
    rows = [
        {
            "timestamp": record.timestamp,
            **{
                field: float(record.values[field])
                for field in ("open", "high", "low", "close", "volume")
            },
        }
        for record in records
        if record.closed is True
        and all(record.values.get(field) is not None for field in ("open", "high", "low", "close", "volume"))
    ]
    if not rows:
        raise ValueError(f"NO_CLOSED_CANDLES:{market}:{timeframe}")
    frame = (
        pd.DataFrame(rows)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
        .iloc[-maximum_rows:]
        .copy()
    )
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.attrs.update(market=market, timeframe=timeframe)
    return frame


async def _load_live_features(
    settings: Settings,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Refresh public candles and build current causal features in memory."""

    loader = DataLoader(settings)
    now = utc_now()
    requested: set[tuple[str, str]] = set()
    candidate_markets = {
        str(market)
        for candidate in candidates
        for market in candidate.get("markets") or []
        if _paper_market_allowed(settings, market)
    }
    for candidate in candidates:
        timeframe = str(candidate["timeframe"])
        for market in candidate.get("markets") or []:
            if market in candidate_markets:
                requested.add((str(market), timeframe))
        blocks = set(candidate.get("block_ids") or [])
        if candidate.get("paper_adapter") == "MTF_DONCHIAN_ATR_FRACTAL":
            requested.update(
                (str(market), "1d")
                for market in candidate.get("markets") or []
                if str(market) in candidate_markets
            )
            if timeframe == "15m":
                requested.update(
                    (str(market), "1h")
                    for market in candidate.get("markets") or []
                    if str(market) in candidate_markets
                )
        if candidate.get("paper_adapter") == "MTF_15M_LIMIT_OVERLAY":
            parent_parameters = dict(
                (candidate.get("parameters") or {}).get("parent") or {}
            )
            parent_timeframe = str(parent_parameters.get("timeframe") or "")
            if parent_timeframe not in {"1h", "2h"}:
                raise ValueError("INVALID_MTF_LIMIT_PARENT_TIMEFRAME")
            requested.update(
                (str(market), parent_timeframe)
                for market in candidate.get("markets") or []
                if str(market) in candidate_markets
            )
        for higher in ("4h", "1d", "1W"):
            if any(f"htf_{higher}" in block for block in blocks):
                requested.update((market, higher) for market in candidate_markets)
    requested.update(("BTC-EUR", str(candidate["timeframe"])) for candidate in candidates)

    async def fetch(market: str, timeframe: str) -> tuple[tuple[str, str], pd.DataFrame]:
        interval = TIMEFRAME_SECONDS[timeframe]
        start = now - timedelta(seconds=interval * 3_100)
        records = await loader.download_canonical_ohlcv(
            provider="bitvavo",
            market=market,
            timeframe=timeframe,
            start=start,
            end=now,
            resume=True,
            persist=True,
        )
        selected_records = records[0]
        return (
            (market, timeframe),
            _records_frame(
                selected_records,
                market=market,
                timeframe=timeframe,
            ),
        )

    raw = dict(
        await asyncio.gather(
            *(fetch(market, timeframe) for market, timeframe in sorted(requested))
        )
    )
    features: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in candidates:
        timeframe = str(candidate["timeframe"])
        benchmark = raw.get(("BTC-EUR", timeframe))
        for market in candidate.get("markets") or []:
            key = (str(market), timeframe)
            if key not in raw or key in features:
                continue
            higher_timeframes = {
                higher: raw[(str(market), higher)]
                for higher in ("4h", "1d", "1W")
                if (str(market), higher) in raw
                and TIMEFRAME_SECONDS[higher] > TIMEFRAME_SECONDS[timeframe]
            }
            features[key] = FeaturePipeline().build(
                raw[key],
                market=str(market),
                benchmark=benchmark,
                higher_timeframes=higher_timeframes,
            )
    # The MTF adapter needs raw, fully closed context series to reproduce its
    # immutable daily filter and (for 15m DNA) causal 1h confirmation.
    for candidate in candidates:
        adapter = candidate.get("paper_adapter")
        if adapter not in {
            "MTF_DONCHIAN_ATR_FRACTAL",
            "MTF_15M_LIMIT_OVERLAY",
        }:
            continue
        for market in candidate.get("markets") or []:
            if adapter == "MTF_15M_LIMIT_OVERLAY":
                parent = dict(
                    (candidate.get("parameters") or {}).get("parent") or {}
                )
                context_timeframes = (str(parent.get("timeframe") or ""),)
            else:
                context_timeframes = (
                    ("1d", "1h")
                    if str(candidate.get("timeframe")) == "15m"
                    else ("1d",)
                )
            for context_timeframe in context_timeframes:
                key = (str(market), context_timeframe)
                if key in raw:
                    features.setdefault(key, raw[key])

    # Point-in-time breadth: missing warm-up members are excluded, never
    # interpreted as bearish.  This is attached only when a generated DNA uses it.
    for timeframe in sorted({key[1] for key in features}):
        selected = {
            market: frame
            for (market, selected_timeframe), frame in features.items()
            if selected_timeframe == timeframe
        }
        if not selected:
            continue
        for period in (20, 50, 200):
            states = {
                market: frame["close"].gt(frame["close"].rolling(period, min_periods=period).mean())
                .where(frame["close"].rolling(period, min_periods=period).count() >= period)
                for market, frame in selected.items()
            }
            matrix = pd.concat(states, axis=1)
            breadth = matrix.sum(axis=1, min_count=1) / matrix.notna().sum(axis=1).replace(0, pd.NA)
            for market, frame in selected.items():
                frame[f"breadth_fraction_above_mean_{period}d"] = breadth.reindex(frame.index)
    return features


async def _current_price(settings: Settings, market: str) -> Decimal:
    record = await DataLoader(settings).download_ticker(
        provider="bitvavo",
        market=market,
        persist=True,
        mode="paper",
    )
    # The normalized Bitvavo ticker contract uses ``last_price``.  Retain the
    # historical aliases so paper and live execution share one public-price
    # interpretation without weakening freshness or provider checks.
    for key in ("last_price", "price", "last", "close"):
        value = record.values.get(key)
        if value is not None and Decimal(str(value)) > 0:
            return Decimal(str(value))
    raise ValueError(f"TICKER_PRICE_MISSING:{market}")


def _combination(candidate: Mapping[str, Any]):
    # Continuous simple-lab discoveries can contain deterministic
    # registry-derived ``auto__`` blocks and causal higher-timeframe blocks.
    # Rebuild the same complete registry used during discovery so a frozen
    # candidate can be reconstructed byte-for-byte for paper evaluation.
    registry = registry_driven_signal_blocks()
    blocks = tuple(str(value) for value in candidate.get("block_ids") or [])
    logic_mode = LogicMode(str(candidate.get("logic_mode") or "LAYERED"))
    generated = CombinationGenerator(registry).generate(
        sizes=(len(blocks),),
        logic_modes=(logic_mode,),
        block_ids=blocks,
        timeframes=(str(candidate["timeframe"]),),
    )
    exact = [row for row in generated if row.block_ids == tuple(sorted(blocks))]
    if len(exact) != 1:
        raise ValueError("FROZEN_COMBINATION_NOT_RECONSTRUCTABLE")
    expected_block_dna = str(
        candidate.get("block_strategy_dna_hash")
        or candidate["strategy_dna_hash"]
    )
    if exact[0].strategy_dna_hash != expected_block_dna:
        raise ValueError("FROZEN_STRATEGY_DNA_MISMATCH")
    return exact[0], registry


def _atr_turtle_market_evaluations(
    candidate: Mapping[str, Any],
    feature_frames: Mapping[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    """Evaluate the frozen ATR Turtle DNA on the latest closed 4h candle."""

    from research.portfolio_breakout import AtrRiskBreakoutParameters

    parameters = AtrRiskBreakoutParameters(
        **dict(candidate.get("parameters") or {}),
    )
    if parameters.dna_hash != str(candidate["strategy_dna_hash"]):
        raise ValueError("FROZEN_ATR_TURTLE_DNA_MISMATCH")
    evaluations: list[dict[str, Any]] = []
    for market in candidate.get("markets") or []:
        timeframe = str(candidate["timeframe"])
        frame = feature_frames.get((str(market), timeframe))
        if frame is None or frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        prior_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        upper = close.shift(1).rolling(
            parameters.entry_lookback,
            min_periods=parameters.entry_lookback,
        ).max()
        lower = close.shift(1).rolling(
            parameters.exit_lookback,
            min_periods=parameters.exit_lookback,
        ).min()
        ema = close.ewm(
            span=parameters.trend_ema_period,
            adjust=False,
            min_periods=parameters.trend_ema_period,
        ).mean()
        atr = true_range.rolling(
            parameters.atr_lookback,
            min_periods=parameters.atr_lookback,
        ).mean()
        latest = frame.index[-1]
        close_at = latest + pd.to_timedelta(
            TIMEFRAME_SECONDS[timeframe],
            unit="s",
        )
        stale = utc_now() - close_at.to_pydatetime() > timedelta(
            seconds=TIMEFRAME_SECONDS[timeframe] * 2,
        )
        latest_close = float(close.iloc[-1])
        latest_atr = float(atr.iloc[-1])
        required = (
            latest_close,
            latest_atr,
            float(upper.iloc[-1]),
            float(lower.iloc[-1]),
            float(ema.iloc[-1]),
        )
        valid = all(pd.notna(value) and value > 0.0 for value in required)
        entry = bool(
            valid
            and latest_close > float(upper.iloc[-1])
            and latest_close > float(ema.iloc[-1])
            and not stale
        )
        exit_ = bool(
            valid
            and (
                latest_close < float(lower.iloc[-1])
                or latest_close < float(ema.iloc[-1])
            )
            and not stale
        )
        stop_distance = (
            latest_atr * parameters.atr_stop_multiple
            if valid
            else float("nan")
        )
        target_distance = stop_distance * 1.5
        risk_levels_valid = _risk_levels_valid(
            close=latest_close,
            stop_distance=stop_distance,
            target_distance=target_distance,
        )
        evaluations.append(
            {
                "market": str(market),
                "signal_timestamp": latest.isoformat(),
                "entry": entry and risk_levels_valid,
                "exit": exit_,
                "stale": stale,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "risk_levels_valid": risk_levels_valid,
                "risk_level_block_reason": (
                    None
                    if risk_levels_valid
                    else "GENERATED_RISK_LEVELS_OUT_OF_RANGE"
                ),
                "size_multiplier": (
                    min(1.0, latest_close / max(stop_distance, 1e-12))
                    if valid
                    else 0.0
                ),
            }
        )
    return evaluations


def _strategy_market_evaluations(
    strategy: Any,
    candidate: Mapping[str, Any],
    feature_frames: Mapping[tuple[str, str], pd.DataFrame],
    *,
    parameters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one canonical StrategyOutput adapter on closed candles."""

    market_evaluations: list[dict[str, Any]] = []
    timeframe = str(candidate["timeframe"])
    for market in candidate.get("markets") or []:
        frame = feature_frames.get((str(market), timeframe))
        if frame is None or frame.empty:
            continue
        output = (
            strategy.generate(frame, dict(parameters))
            if parameters is not None
            else strategy.generate(frame)
        )
        latest = frame.index[-1]
        close_at = latest + pd.to_timedelta(
            TIMEFRAME_SECONDS[timeframe],
            unit="s",
        )
        stale = utc_now() - close_at.to_pydatetime() > timedelta(
            seconds=TIMEFRAME_SECONDS[timeframe] * 2,
        )
        latest_close = float(frame["close"].iloc[-1])
        stop_distance = float(output.stop_distance.iloc[-1])
        raw_target_distance = float(output.target_distance.iloc[-1])
        exit_profile = str(output.metadata.get("exit_profile") or "")
        target_required = not exit_profile or exit_profile == "FIXED_R"
        target_distance = (
            raw_target_distance if target_required else None
        )
        risk_levels_valid = _risk_levels_valid(
            close=latest_close,
            stop_distance=stop_distance,
            target_distance=target_distance,
            target_required=target_required,
        )
        market_evaluations.append(
            {
                "market": str(market),
                "signal_timestamp": latest.isoformat(),
                "entry": bool(output.entry.iloc[-1])
                and not stale
                and risk_levels_valid,
                "exit": bool(output.exit.iloc[-1]) and not stale,
                "stale": stale,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "target_required": target_required,
                "exit_profile": exit_profile or None,
                "maximum_holding_bars": output.maximum_holding_bars,
                "risk_levels_valid": risk_levels_valid,
                "risk_level_block_reason": (
                    None
                    if risk_levels_valid
                    else "GENERATED_RISK_LEVELS_OUT_OF_RANGE"
                ),
                "size_multiplier": float(
                    output.size_multiplier.iloc[-1]
                ),
            }
        )
    return market_evaluations


def _closed_bars_since_signal(
    frame: pd.DataFrame | None,
    signal_timestamp: object,
) -> int:
    """Count fully closed evaluation bars after the entry signal candle."""

    if frame is None or frame.empty or signal_timestamp in {None, ""}:
        return 0
    index = pd.DatetimeIndex(frame.index)
    signal = pd.Timestamp(signal_timestamp)
    if index.tz is None and signal.tzinfo is not None:
        signal = signal.tz_convert("UTC").tz_localize(None)
    elif index.tz is not None and signal.tzinfo is None:
        signal = signal.tz_localize(index.tz)
    elif index.tz is not None and signal.tzinfo is not None:
        signal = signal.tz_convert(index.tz)
    return int((index > signal).sum())


def _mtf_donchian_market_evaluations(
    candidate: Mapping[str, Any],
    feature_frames: Mapping[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    """Evaluate the immutable 1h/2h Donchian candidate on closed candles."""

    parameters = MultiTimeframeParameters(
        **dict(candidate.get("parameters") or {}),
    )
    if parameters.dna_hash != str(candidate["strategy_dna_hash"]):
        raise ValueError("FROZEN_MTF_DONCHIAN_DNA_MISMATCH")
    evaluations: list[dict[str, Any]] = []
    for market in candidate.get("markets") or []:
        timeframe = str(candidate["timeframe"])
        frame = feature_frames.get((str(market), timeframe))
        daily = feature_frames.get((str(market), "1d"))
        hourly = feature_frames.get((str(market), "1h"))
        if frame is None or frame.empty or daily is None or daily.empty:
            continue
        if timeframe == "15m" and (hourly is None or hourly.empty):
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        prior_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        upper = high.shift(1).rolling(
            parameters.entry_lookback,
            min_periods=parameters.entry_lookback,
        ).max()
        lower = low.shift(1).rolling(
            parameters.exit_lookback,
            min_periods=parameters.exit_lookback,
        ).min()
        atr = true_range.rolling(
            parameters.atr_period,
            min_periods=parameters.atr_period,
        ).mean()
        latest = frame.index[-1]
        close_at = latest + pd.to_timedelta(
            TIMEFRAME_SECONDS[timeframe],
            unit="s",
        )
        eligible_daily = daily.loc[
            daily.index + pd.Timedelta(1, unit="D") <= close_at
        ]
        daily_close = pd.to_numeric(
            eligible_daily["close"],
            errors="coerce",
        )
        daily_ema = daily_close.ewm(
            span=parameters.daily_ema_period,
            adjust=False,
            min_periods=parameters.daily_ema_period,
        ).mean()
        daily_trend = bool(
            len(daily_close)
            and pd.notna(daily_ema.iloc[-1])
            and daily_close.iloc[-1] > daily_ema.iloc[-1]
        )
        hourly_trend = True
        if timeframe == "15m" and hourly is not None:
            eligible_hourly = hourly.loc[
                hourly.index + pd.Timedelta(1, unit="h") <= close_at
            ]
            hourly_close = pd.to_numeric(
                eligible_hourly["close"],
                errors="coerce",
            )
            hourly_ema = hourly_close.ewm(
                span=50,
                adjust=False,
                min_periods=50,
            ).mean()
            hourly_trend = bool(
                len(hourly_close)
                and pd.notna(hourly_ema.iloc[-1])
                and hourly_close.iloc[-1] > hourly_ema.iloc[-1]
            )
        stale = utc_now() - close_at.to_pydatetime() > timedelta(
            seconds=TIMEFRAME_SECONDS[timeframe] * 2,
        )
        latest_close = float(close.iloc[-1])
        latest_atr = float(atr.iloc[-1])
        required = (
            latest_close,
            latest_atr,
            float(upper.iloc[-1]),
            float(lower.iloc[-1]),
        )
        valid = all(pd.notna(value) and value > 0.0 for value in required)
        entry = bool(
            valid
            and latest_close > float(upper.iloc[-1])
            and daily_trend
            and hourly_trend
            and not stale
        )
        exit_ = bool(
            valid
            and (
                latest_close < float(lower.iloc[-1])
                or not daily_trend
                or not hourly_trend
            )
            and not stale
        )
        stop_distance = (
            latest_atr * parameters.atr_stop_multiple
            if valid
            else float("nan")
        )
        fractal_price = frame.get("confirmed_fractal_low_price")
        if (
            valid
            and parameters.use_confirmed_fractal_stop
            and fractal_price is not None
        ):
            confirmed = pd.to_numeric(fractal_price, errors="coerce").ffill()
            if len(confirmed) and pd.notna(confirmed.iloc[-1]):
                distance = latest_close - float(confirmed.iloc[-1])
                if 0.0 < distance < stop_distance:
                    stop_distance = distance
        target_distance = stop_distance * parameters.reward_risk
        risk_levels_valid = _risk_levels_valid(
            close=latest_close,
            stop_distance=stop_distance,
            target_distance=target_distance,
        )
        evaluations.append(
            {
                "market": str(market),
                "signal_timestamp": latest.isoformat(),
                "entry": entry and risk_levels_valid,
                "exit": exit_,
                "stale": stale,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "risk_levels_valid": risk_levels_valid,
                "risk_level_block_reason": (
                    None
                    if risk_levels_valid
                    else "GENERATED_RISK_LEVELS_OUT_OF_RANGE"
                ),
                "size_multiplier": (
                    min(1.0, latest_close / max(stop_distance, 1e-12))
                    if valid
                    else 0.0
                ),
                "daily_trend": daily_trend,
                "hourly_trend": hourly_trend,
                "hourly_confirmation_required": timeframe == "15m",
                "confirmed_fractal_stop_used": bool(
                    valid
                    and stop_distance
                    < latest_atr * parameters.atr_stop_multiple
                ),
            }
        )
    return evaluations


def _mtf_limit_overlay_market_evaluations(
    candidate: Mapping[str, Any],
    feature_frames: Mapping[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    """Evaluate a future-only 15m limit touch for frozen parent alpha.

    A paper entry is emitted only when the latest fully closed 15m candle
    touched the immutable parent breakout level inside its bounded resting
    window.  Earlier touches are deliberately not replayed after downtime.
    """

    raw_parameters = dict(candidate.get("parameters") or {})
    parent = MultiTimeframeParameters(
        **dict(raw_parameters.get("parent") or {}),
    )
    parameters = LimitOverlayParameters(
        parent=parent,
        entry_window_15m_bars=int(
            raw_parameters.get("entry_window_15m_bars") or 0
        ),
        normal_side_cost_bps=float(
            raw_parameters.get("normal_side_cost_bps") or 27.0
        ),
        stressed_side_cost_bps=float(
            raw_parameters.get("stressed_side_cost_bps") or 50.0
        ),
    )
    if parameters.dna_hash != str(candidate["strategy_dna_hash"]):
        raise ValueError("FROZEN_MTF_LIMIT_OVERLAY_DNA_MISMATCH")
    if parameters.entry_window_15m_bars < 1:
        raise ValueError("INVALID_MTF_LIMIT_ENTRY_WINDOW")

    interval = pd.Timedelta(15, unit="m")
    evaluations: list[dict[str, Any]] = []
    for market in candidate.get("markets") or []:
        market = str(market)
        fifteen = feature_frames.get((market, "15m"))
        parent_frame = feature_frames.get((market, parent.timeframe))
        if (
            fifteen is None
            or fifteen.empty
            or parent_frame is None
            or parent_frame.empty
        ):
            continue
        featured = _feature_frame(parent_frame, parent)
        if featured.empty:
            continue
        latest_at = pd.Timestamp(fifteen.index[-1])
        latest_close_at = latest_at + interval
        stale = utc_now() - latest_close_at.to_pydatetime() > timedelta(
            seconds=TIMEFRAME_SECONDS["15m"] * 2,
        )
        latest_bar = fifteen.iloc[-1]
        available_parent = featured.loc[
            pd.to_datetime(featured["decision_at"], utc=True)
            <= latest_close_at
        ]
        if available_parent.empty:
            continue
        latest_parent = available_parent.iloc[-1]
        exit_ = bool(latest_parent["exit_signal"]) and not stale

        matching_signal: Mapping[str, Any] | None = None
        for _, signal in available_parent.loc[
            available_parent["entry_signal"].astype(bool)
        ].iloc[::-1].iterrows():
            decision_at = pd.Timestamp(signal["decision_at"])
            active_at = decision_at + interval
            expires_at = active_at + interval * (
                parameters.entry_window_15m_bars - 1
            )
            if active_at <= latest_at <= expires_at:
                matching_signal = signal
                break
            if latest_at > expires_at:
                break

        fill: float | None = None
        limit_price: float | None = None
        stop_distance: float | None = None
        target_distance: float | None = None
        parent_signal_at: str | None = None
        if matching_signal is not None and not stale and not exit_:
            limit_price = float(matching_signal["entry_level"])
            bar_open = float(latest_bar["open"])
            bar_low = float(latest_bar["low"])
            bar_high = float(latest_bar["high"])
            if bar_open <= limit_price:
                fill = min(bar_open, limit_price)
            elif bar_low <= limit_price <= bar_high:
                fill = limit_price
            if fill is not None:
                stop = _overlay_stop(matching_signal, fill, parameters)
                stop_distance = fill - stop
                target_distance = stop_distance * parent.reward_risk
                parent_signal_at = pd.Timestamp(
                    matching_signal["decision_at"]
                ).isoformat()

        risk_levels_valid = bool(
            fill is not None
            and stop_distance is not None
            and target_distance is not None
            and _risk_levels_valid(
                close=fill,
                stop_distance=stop_distance,
                target_distance=target_distance,
            )
        )
        evaluations.append(
            {
                "market": market,
                "signal_timestamp": latest_at.isoformat(),
                "parent_signal_timestamp": parent_signal_at,
                "entry": risk_levels_valid,
                "exit": exit_,
                "stale": stale,
                "paper_fill_price": fill,
                "limit_price": limit_price,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "risk_levels_valid": risk_levels_valid,
                "risk_level_block_reason": (
                    None
                    if risk_levels_valid
                    else "NO_CURRENT_CAUSAL_LIMIT_TOUCH"
                ),
                "size_multiplier": (
                    min(1.0, fill / max(stop_distance, 1e-12))
                    if (
                        risk_levels_valid
                        and fill is not None
                        and stop_distance is not None
                    )
                    else 0.0
                ),
                "parent_timeframe": parent.timeframe,
                "order_policy": "LIMIT_NO_CHASE_NO_MARKET_FALLBACK",
            }
        )
    return evaluations


def _risk_levels_valid(
    *,
    close: float,
    stop_distance: float,
    target_distance: float | None,
    target_required: bool = True,
) -> bool:
    """Reject non-finite or economically impossible generated exits."""

    return bool(
        pd.notna(close)
        and pd.notna(stop_distance)
        and close > 0.0
        and 0.0 < stop_distance < close * 0.50
        and (
            not target_required
            or (
                target_distance is not None
                and pd.notna(target_distance)
                and 0.0 < target_distance < close * 2.00
            )
        )
    )


def _new_orders_today(broker: PaperBroker) -> int:
    today = utc_now().date().isoformat()
    return sum(
        1
        for event in broker.ledger.events()
        if event.get("event_type") == "ORDER_INTENT"
        and str(event.get("recorded_at") or "")[:10] == today
        and str((event.get("payload") or {}).get("side")) == "BUY"
    )


def _candidate_disposition(
    candidate: Mapping[str, Any],
    *,
    entry_dna: set[str],
    managed_dna: set[str],
    current_position_count: int,
    effective_position_cap: int,
) -> dict[str, Any]:
    """Build non-authoritative diagnostics for one exact paper DNA."""

    dna = str(candidate.get("strategy_dna_hash") or "")
    return {
        "strategy_dna": dna,
        "strategy_id": (
            candidate.get("strategy_id")
            or candidate.get("economic_hypothesis_family")
        ),
        "strategy_family": candidate.get("economic_hypothesis_family"),
        "markets": sorted(str(row) for row in candidate.get("markets") or []),
        "timeframe": candidate.get("timeframe"),
        "status": "EVALUATION_PENDING",
        "entry_cohort_member": dna in entry_dna,
        "position_managed_at_cycle_start": dna in managed_dna,
        "natural_entry_signal_count": 0,
        "natural_entry_markets": [],
        "occupied_entry_markets": [],
        "current_position_count": current_position_count,
        "effective_position_cap": effective_position_cap,
        "frozen_candidate_hash": candidate.get("frozen_candidate_hash"),
        "source_strategy_dna_hash": candidate.get("source_strategy_dna_hash"),
        "source_csv_sha256": candidate.get("source_csv_sha256"),
        "paper_orders_generated": 0,
        "paper_orders_filled": 0,
        "real_orders_placed": 0,
        "real_exchange_requests": 0,
        "affects_execution": False,
    }


def _write_disposition_snapshot(
    settings: Settings,
    *,
    cycle_at: str,
    dispositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in dispositions.values():
        status = str(row.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    payload: dict[str, Any] = {
        "schema_version": "generated_strategy_paper_dispositions_v1",
        "cycle_at": cycle_at,
        "paper_only": True,
        "affects_execution": False,
        "real_orders_placed": 0,
        "real_exchange_requests": 0,
        "candidate_count": len(dispositions),
        "status_counts": dict(sorted(counts.items())),
        "dispositions": dict(sorted(dispositions.items())),
    }
    payload["snapshot_sha256"] = stable_hash(payload)
    atomic_write_json(_paths(settings)["dispositions"], payload)
    return payload


def _prospective_entry_cohort(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Keep paper entry evaluation focused on robust, falsifiable candidates."""

    robustness_fields = (
        "stressed_net_return",
        "double_cost_net_return",
        "holdout_net_return",
        "validation_net_return",
        "confirmation_net_return",
    )
    selected: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = dict(raw)
        metrics = dict(candidate.get("metrics") or {})
        integrity = dict(candidate.get("integrity") or {})
        try:
            net_return = float(metrics.get("net_return"))
            profit_factor = float(metrics.get("profit_factor"))
            trade_count = int(metrics.get("trade_count"))
            robustness = [
                float(metrics[field])
                for field in robustness_fields
                if metrics.get(field) is not None
            ]
        except (TypeError, ValueError, OverflowError):
            continue
        if not (
            math.isfinite(net_return)
            and math.isfinite(profit_factor)
            and all(math.isfinite(value) for value in robustness)
            and net_return > 0.0
            and profit_factor > 1.10
            and trade_count >= 20
            and len(robustness) >= 2
            and all(value > 0.0 for value in robustness)
            and integrity.get("no_lookahead") is True
            and integrity.get("no_repainting") is True
            and integrity.get("next_open_execution") is True
            and integrity.get("long_only_spot") is True
            and integrity.get("valid_data", True) is not False
        ):
            continue
        candidate["prospective_selection"] = {
            "status": "ROBUST_PAPER_COHORT",
            "profit_factor": profit_factor,
            "trade_count": trade_count,
            "positive_robustness_check_count": len(robustness),
            "live_authority": False,
        }
        selected.append(candidate)
    return sorted(
        selected,
        key=lambda row: (
            -float((row.get("metrics") or {}).get("profit_factor") or 0.0),
            -int((row.get("metrics") or {}).get("trade_count") or 0),
            str(row.get("strategy_dna_hash") or ""),
        ),
    )[: max(1, int(limit))]


def _submit(
    settings: Settings,
    broker: PaperBroker,
    *,
    candidate: Mapping[str, Any],
    market: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    identity: str,
    reason: str,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
):
    dna = str(candidate["strategy_dna_hash"])
    intent = OrderIntent(
        intent_id=f"paper-generated-{stable_hash([dna, identity, side.value], length=18)}",
        idempotency_key=stable_hash(
            ["generated-paper", dna, identity, side.value],
            length=32,
        ),
        market=market,
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        time_in_force=(
            OrderTimeInForce.IOC
            if order_type is OrderType.LIMIT
            else OrderTimeInForce.GTC
        ),
        strategy_id=f"CLASSICAL_{dna[:16]}",
        strategy_dna_hash=dna,
        maximum_notional_eur=(
            Decimal(str(settings.paper_automation.initial_capital_eur))
            if side is OrderSide.BUY
            else None
        ),
        reason_codes=(reason,),
    )
    return broker.submit(intent, market_price=price)


def _notify_fill(
    settings: Settings,
    *,
    order: Any,
    candidate: Mapping[str, Any],
) -> None:
    try:
        notifier = TelegramNotifier(
            settings.telegram,
            output_directory=settings.paths.output_dir / "notifications",
            allowed_markets=settings.operational.markets,
        )
        notifier.notify_order_event(
            "PAPER_ORDER_FILLED",
            {
                "market": order.intent.market,
                "side": order.intent.side.value,
                "order_type": order.intent.order_type.value,
                "average_fill_price": order.average_fill_price,
                "filled_quantity": order.filled_quantity,
                "notional_eur": (
                    order.filled_quantity * order.average_fill_price
                    if order.average_fill_price is not None
                    else None
                ),
                "strategy_id": candidate.get("economic_hypothesis_family"),
                "order_id": order.order_id,
                "status": order.status.value,
                "execution_mode": "PAPER_ONLY",
                "paper_only": True,
                "real_exchange_request": False,
            },
        )
    except Exception:
        # Notifications must never alter paper state or signal generation.
        return


def _notify_paper_promotions(
    settings: Settings,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Notify once per immutable promotion set without affecting paper state."""

    if not candidates:
        return {
            "delivery_status": "SKIPPED_FILTER",
            "reason_code": "NO_PAPER_PROMOTIONS",
        }
    try:
        notifier = TelegramNotifier(
            settings.telegram,
            output_directory=settings.paths.output_dir / "notifications",
            allowed_markets=settings.operational.markets,
        )
        return notifier.notify_paper_promotion_summary(candidates)
    except Exception as exc:
        # Promotion, signal generation and paper execution remain independent.
        return {
            "delivery_status": "FAILED_ISOLATED",
            "reason_code": (
                f"PAPER_PROMOTION_NOTIFICATION_{type(exc).__name__.upper()}"
            ),
        }


async def run_generated_paper_once(
    settings: Settings,
    *,
    frame_loader: FrameLoader | None = None,
    price_loader: PriceLoader | None = None,
) -> dict[str, Any]:
    """Evaluate frozen exact-positive DNA and simulate natural paper fills."""

    state = _state(settings)
    candidates = _load_candidates(settings)
    frozen = freeze_generated_candidates(settings, candidates)
    blocked_dna = {
        row["strategy_dna_hash"] for row in frozen.get("identity_drift_blockers") or []
    }
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("strategy_dna_hash") not in blocked_dna
        and execution_timeframe_allowed(str(candidate.get("timeframe") or ""))
    ]
    positions = dict(state.get("positions") or {})
    entry_candidates = _prospective_entry_cohort(eligible_candidates)
    entry_dna = {
        str(candidate["strategy_dna_hash"])
        for candidate in entry_candidates
    }
    managed_dna = set(positions)
    legacy_management_dna = managed_dna - entry_dna
    effective_position_cap = (
        settings.paper_automation.max_open_positions
        + len(legacy_management_dna)
    )
    active_candidates = [
        candidate
        for candidate in eligible_candidates
        if str(candidate.get("strategy_dna_hash") or "") in entry_dna | managed_dna
    ]
    promoted = set(state.get("promoted_dna") or [])
    notified_promotions = set(state.get("notified_promotion_dna") or [])
    for candidate in entry_candidates:
        dna = str(candidate["strategy_dna_hash"])
        if dna not in promoted:
            append_jsonl(
                _paths(settings)["promotions"],
                {
                    "timestamp": utc_iso(),
                    "strategy_dna": dna,
                    "strategy_id": candidate.get("economic_hypothesis_family"),
                    "from": "BACKTEST_POSITIVE",
                    "to": "PAPER_ACTIVE",
                    "auto_live_promotion": False,
                    "orders_generated": 0,
                },
            )
            promoted.add(dna)
    pending_promotion_notifications = [
        candidate
        for candidate in entry_candidates
        if str(candidate["strategy_dna_hash"]) in promoted
        and str(candidate["strategy_dna_hash"]) not in notified_promotions
    ]
    promotion_notification = _notify_paper_promotions(
        settings,
        pending_promotion_notifications,
    )
    if promotion_notification.get("delivery_status") in {
        "PENDING",
        "SENT",
        "SKIPPED_DUPLICATE",
    }:
        notified_promotions.update(
            str(candidate["strategy_dna_hash"])
            for candidate in pending_promotion_notifications
        )

    if not settings.paper_automation.autotrade_enabled:
        state.update(
            {
                "status": "DISABLED",
                "promoted_dna": sorted(promoted),
                "notified_promotion_dna": sorted(notified_promotions),
                "last_promotion_notification": promotion_notification,
                "last_cycle_at": utc_iso(),
                "last_reason": "PAPER_AUTOTRADE_DISABLED",
            }
        )
        atomic_write_json(_paths(settings)["state"], state)
        return {
            **state,
            "paper_active_candidates": len(entry_candidates),
            "paper_management_candidates": len(managed_dna),
            "orders_generated_this_cycle": 0,
        }
    if not active_candidates:
        state.update(
            {
                "status": "READY_NO_CANDIDATES",
                "promoted_dna": sorted(promoted),
                "notified_promotion_dna": sorted(notified_promotions),
                "last_promotion_notification": promotion_notification,
                "last_cycle_at": utc_iso(),
                "last_reason": "NO_EXACT_POSITIVE_GENERATED_DNA",
            }
        )
        atomic_write_json(_paths(settings)["state"], state)
        return {
            **state,
            "paper_active_candidates": 0,
            "paper_management_candidates": 0,
            "orders_generated_this_cycle": 0,
        }

    loader = frame_loader or _load_live_features
    prices = price_loader or _current_price
    try:
        feature_frames = await loader(settings, active_candidates)
    except Exception as exc:
        state.update(
            {
                "status": "DATA_BLOCKED",
                "promoted_dna": sorted(promoted),
                "notified_promotion_dna": sorted(notified_promotions),
                "last_promotion_notification": promotion_notification,
                "last_cycle_at": utc_iso(),
                "last_reason": f"GENERATED_PAPER_DATA:{type(exc).__name__}",
            }
        )
        atomic_write_json(_paths(settings)["state"], state)
        return {
            **state,
            "paper_active_candidates": len(entry_candidates),
            "paper_management_candidates": len(managed_dna),
            "orders_generated_this_cycle": 0,
        }

    all_markets = sorted(
        {
            str(market)
            for candidate in active_candidates
            for market in candidate.get("markets") or []
            if _paper_market_allowed(settings, market)
        }
    )
    broker = _broker(settings, all_markets)
    evaluations: dict[str, Any] = {}
    candidate_dispositions = {
        str(candidate["strategy_dna_hash"]): _candidate_disposition(
            candidate,
            entry_dna=entry_dna,
            managed_dna=managed_dna,
            current_position_count=len(positions),
            effective_position_cap=effective_position_cap,
        )
        for candidate in active_candidates
    }
    generated_orders = 0

    for candidate in active_candidates:
        dna = str(candidate["strategy_dna_hash"])
        try:
            if candidate.get("paper_adapter") == "ATR_TURTLE_4H":
                market_evaluations = _atr_turtle_market_evaluations(
                    candidate,
                    feature_frames,
                )
            elif (
                candidate.get("paper_adapter")
                == "MTF_DONCHIAN_ATR_FRACTAL"
            ):
                market_evaluations = _mtf_donchian_market_evaluations(
                    candidate,
                    feature_frames,
                )
            elif (
                candidate.get("paper_adapter")
                == "MTF_15M_LIMIT_OVERLAY"
            ):
                market_evaluations = _mtf_limit_overlay_market_evaluations(
                    candidate,
                    feature_frames,
                )
            elif (
                candidate.get("paper_adapter")
                == "VOLUME_CATALOG_BOUNDED_RISK"
            ):
                adapter = volume_strategy_adapter(
                    str(candidate["strategy_id"]),
                )
                if (
                    adapter.canonical_adapter_dna_hash != dna
                    or adapter.legacy_strategy_dna_hash
                    != str(candidate.get("source_strategy_dna_hash") or "")
                    or adapter.row.market
                    != str((candidate.get("markets") or [""])[0])
                    or adapter.row.timeframe
                    != str(candidate.get("timeframe") or "")
                ):
                    raise ValueError("FROZEN_VOLUME_ADAPTER_DNA_MISMATCH")
                market_evaluations = _strategy_market_evaluations(
                    adapter,
                    candidate,
                    feature_frames,
                    parameters=dict(candidate.get("parameters") or {}),
                )
            else:
                combination, registry = _combination(candidate)
                strategy = CombinatorialStrategy(
                    combination,
                    registry,
                    block_parameters=dict(
                        candidate.get("parameters") or {},
                    ),
                )
                market_evaluations = _strategy_market_evaluations(
                    strategy,
                    candidate,
                    feature_frames,
                )
        except Exception as exc:
            evaluations[dna] = {
                "status": "IDENTITY_BLOCKED",
                "reason": type(exc).__name__,
            }
            candidate_dispositions[dna].update(
                {
                    "status": "IDENTITY_BLOCKED",
                    "reason": type(exc).__name__,
                }
            )
            continue
        evaluations[dna] = {
            "status": "EVALUATED",
            "markets": market_evaluations,
        }

        position = dict(positions.get(dna) or {})
        if position:
            selected = next(
                (
                    row
                    for row in market_evaluations
                    if row["market"] == position.get("market")
                ),
                None,
            )
            if selected is None:
                candidate_dispositions[dna].update(
                    {
                        "status": "DATA_UNAVAILABLE",
                        "reason": "POSITION_MARKET_EVALUATION_MISSING",
                    }
                )
                continue
            try:
                price = await prices(settings, str(position["market"]))
            except (RuntimeError, ValueError, TimeoutError):
                evaluations[dna]["price_status"] = "UNAVAILABLE"
                evaluations[dna]["price_reason"] = "PAPER_TICKER_UNAVAILABLE"
                candidate_dispositions[dna].update(
                    {
                        "status": "PRICE_UNAVAILABLE",
                        "reason": "PAPER_TICKER_UNAVAILABLE",
                    }
                )
                continue
            quantity = Decimal(str(position["quantity"]))
            maximum_holding_bars = position.get("maximum_holding_bars")
            if maximum_holding_bars is None:
                maximum_holding_bars = selected.get(
                    "maximum_holding_bars"
                )
            closed_holding_bars = _closed_bars_since_signal(
                feature_frames.get(
                    (str(position["market"]), str(position["timeframe"]))
                ),
                position.get("signal_timestamp"),
            )
            evaluations[dna]["closed_holding_bars"] = closed_holding_bars
            evaluations[dna]["maximum_holding_bars"] = (
                maximum_holding_bars
            )
            exit_reason = None
            if price <= Decimal(str(position["stop_loss"])):
                exit_reason = "PAPER_STOP_LOSS"
            elif (
                position.get("take_profit_2") is not None
                and price >= Decimal(str(position["take_profit_2"]))
            ):
                exit_reason = "PAPER_TAKE_PROFIT_2"
            elif selected["exit"]:
                exit_reason = "PAPER_STRATEGY_EXIT"
            elif (
                maximum_holding_bars is not None
                and closed_holding_bars >= int(maximum_holding_bars)
            ):
                exit_reason = "PAPER_MAXIMUM_HOLDING"
            if exit_reason is not None:
                order = _submit(
                    settings,
                    broker,
                    candidate=candidate,
                    market=str(position["market"]),
                    side=OrderSide.SELL,
                    quantity=quantity,
                    price=price,
                    identity=f"{position['signal_id']}:{exit_reason}",
                    reason=exit_reason,
                )
                generated_orders += 1
                candidate_dispositions[dna]["paper_orders_generated"] = 1
                if order.status is OrderStatus.FILLED:
                    positions.pop(dna, None)
                    state["last_closed_position"] = {
                        **position,
                        "exit_price": str(order.average_fill_price),
                        "exit_reason": exit_reason,
                        "closed_at": utc_iso(),
                    }
                    _notify_fill(settings, order=order, candidate=candidate)
                    candidate_dispositions[dna].update(
                        {
                            "status": "PAPER_EXIT_FILLED",
                            "reason": exit_reason,
                            "paper_orders_filled": 1,
                            "paper_order_status": order.status.value,
                        }
                    )
                else:
                    candidate_dispositions[dna].update(
                        {
                            "status": "PAPER_EXIT_REJECTED",
                            "reason": exit_reason,
                            "paper_order_status": order.status.value,
                        }
                    )
            else:
                candidate_dispositions[dna].update(
                    {
                        "status": "POSITION_MANAGED",
                        "reason": "NO_NATURAL_EXIT_SIGNAL",
                    }
                )
            continue

        occupied_markets = {
            str(row.get("market") or "") for row in positions.values()
        }
        natural_entry_markets = sorted(
            str(row.get("market") or "")
            for row in market_evaluations
            if row["entry"]
        )
        occupied_entry_markets = sorted(
            set(natural_entry_markets) & occupied_markets
        )
        candidate_dispositions[dna].update(
            {
                "natural_entry_signal_count": len(natural_entry_markets),
                "natural_entry_markets": natural_entry_markets,
                "occupied_entry_markets": occupied_entry_markets,
                "current_position_count": len(positions),
            }
        )
        eligible = sorted(
            (
                row
                for row in market_evaluations
                if row["entry"]
                and str(row.get("market") or "") not in occupied_markets
            ),
            key=lambda row: (row["size_multiplier"], row["market"]),
            reverse=True,
        )
        if not eligible:
            candidate_dispositions[dna].update(
                {
                    "status": (
                        "MARKET_OCCUPIED_BY_OTHER_DNA"
                        if natural_entry_markets and occupied_entry_markets
                        else "NO_FRESH_ENTRY_SIGNAL"
                    ),
                    "reason": (
                        "ENTRY_MARKET_ALREADY_HAS_PAPER_POSITION"
                        if natural_entry_markets and occupied_entry_markets
                        else "LATEST_CAUSAL_BAR_HAS_NO_ENTRY"
                    ),
                }
            )
            continue
        if len(positions) >= effective_position_cap:
            candidate_dispositions[dna].update(
                {
                    "status": "POSITION_CAP_BLOCKED",
                    "reason": "PAPER_EFFECTIVE_POSITION_CAP_REACHED",
                }
            )
            continue
        orders_today = _new_orders_today(broker)
        if orders_today >= settings.paper_automation.max_new_orders_per_day:
            candidate_dispositions[dna].update(
                {
                    "status": "DAILY_ORDER_CAP_BLOCKED",
                    "reason": "PAPER_DAILY_NEW_ORDER_CAP_REACHED",
                    "new_orders_today": orders_today,
                }
            )
            continue
        selected: dict[str, Any] | None = None
        price: Decimal | None = None
        unavailable_markets: list[str] = []
        for row in eligible:
            market = str(row["market"])
            if row.get("paper_fill_price") is not None:
                selected = dict(row)
                price = Decimal(str(row["paper_fill_price"]))
                break
            try:
                selected_price = await prices(settings, market)
            except (RuntimeError, ValueError, TimeoutError):
                unavailable_markets.append(market)
                continue
            selected = dict(row)
            price = selected_price
            break
        if selected is None or price is None:
            evaluations[dna]["price_status"] = "UNAVAILABLE"
            evaluations[dna]["unavailable_markets"] = unavailable_markets
            candidate_dispositions[dna].update(
                {
                    "status": "PRICE_UNAVAILABLE",
                    "reason": "PAPER_TICKER_UNAVAILABLE",
                    "unavailable_markets": unavailable_markets,
                }
            )
            continue
        market = str(selected["market"])
        stop_distance = Decimal(str(selected["stop_distance"]))
        target_distance = (
            Decimal(str(selected["target_distance"]))
            if selected.get("target_distance") is not None
            else None
        )
        if stop_distance <= 0 or (
            target_distance is not None and target_distance <= 0
        ):
            candidate_dispositions[dna].update(
                {
                    "status": "RISK_LEVELS_INVALID",
                    "reason": "NON_POSITIVE_STOP_OR_TARGET_DISTANCE",
                }
            )
            continue
        risk_multiplier = Decimal("0.25")
        risk_budget = (
            Decimal(str(settings.paper_automation.initial_capital_eur))
            * Decimal(str(settings.paper_automation.max_risk_per_trade_pct))
            / Decimal("100")
            * risk_multiplier
        )
        exposure_cap = (
            Decimal(str(settings.paper_automation.initial_capital_eur))
            * min(
                Decimal("0.20"),
                Decimal(str(settings.paper_automation.max_total_exposure_pct))
                / Decimal("100"),
            )
        )
        affordable = broker.balances.get("EUR", Decimal("0")) / (
            price * (Decimal("1") + Decimal(str(settings.costs.default_fee)))
        )
        quantity = min(
            risk_budget / stop_distance,
            exposure_cap / price,
            affordable,
        )
        signal_id = stable_hash(
            [
                dna,
                market,
                candidate["timeframe"],
                selected["signal_timestamp"],
                candidate.get("frozen_candidate_hash"),
            ],
            length=32,
        )
        order = _submit(
            settings,
            broker,
            candidate=candidate,
            market=market,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            identity=signal_id,
            reason="NATURAL_GENERATED_ENTRY",
            order_type=(
                OrderType.LIMIT
                if selected.get("paper_fill_price") is not None
                else OrderType.MARKET
            ),
            limit_price=(
                price
                if selected.get("paper_fill_price") is not None
                else None
            ),
        )
        generated_orders += 1
        candidate_dispositions[dna]["paper_orders_generated"] = 1
        if order.status is OrderStatus.FILLED:
            fill_price = Decimal(str(order.average_fill_price))
            positions[dna] = {
                "strategy_id": candidate.get("economic_hypothesis_family"),
                "strategy_dna": dna,
                "frozen_candidate_hash": candidate.get("frozen_candidate_hash"),
                "market": market,
                "timeframe": candidate["timeframe"],
                "signal_id": signal_id,
                "signal_timestamp": selected["signal_timestamp"],
                "entry_price": str(fill_price),
                "quantity": str(order.filled_quantity),
                "stop_loss": str(fill_price - stop_distance),
                "take_profit_1": (
                    str(fill_price + target_distance)
                    if target_distance is not None
                    else None
                ),
                "take_profit_2": (
                    str(fill_price + target_distance * Decimal("2"))
                    if target_distance is not None
                    else None
                ),
                "exit_profile": selected.get("exit_profile"),
                "maximum_holding_bars": selected.get(
                    "maximum_holding_bars"
                ),
                "opened_at": utc_iso(),
                "paper_only": True,
            }
            _notify_fill(settings, order=order, candidate=candidate)
            candidate_dispositions[dna].update(
                {
                    "status": "PAPER_ORDER_FILLED",
                    "reason": "NATURAL_GENERATED_ENTRY",
                    "paper_orders_filled": 1,
                    "paper_order_status": order.status.value,
                    "selected_market": market,
                    "signal_id": signal_id,
                }
            )
        else:
            candidate_dispositions[dna].update(
                {
                    "status": "PAPER_ORDER_REJECTED",
                    "reason": "PAPER_BROKER_REJECTED_ENTRY",
                    "paper_order_status": order.status.value,
                    "selected_market": market,
                    "signal_id": signal_id,
                }
            )

    reconciliation = broker.reconcile()
    cycle_at = utc_iso()
    disposition_snapshot = _write_disposition_snapshot(
        settings,
        cycle_at=cycle_at,
        dispositions=candidate_dispositions,
    )
    state.update(
        {
            "schema_version": "generated_strategy_paper_v1",
            "status": (
                "RECONCILIATION_BLOCKED"
                if not reconciliation.healthy
                else "ACTIVE"
                if positions
                else "READY"
            ),
            "positions": positions,
            "evaluations": evaluations,
            "candidate_dispositions": candidate_dispositions,
            "candidate_disposition_status_counts": disposition_snapshot[
                "status_counts"
            ],
            "candidate_disposition_snapshot_sha256": disposition_snapshot[
                "snapshot_sha256"
            ],
            "promoted_dna": sorted(promoted),
            "notified_promotion_dna": sorted(notified_promotions),
            "last_promotion_notification": promotion_notification,
            "paper_active_candidates": len(entry_candidates),
            "paper_entry_cohort_count": len(entry_candidates),
            "paper_management_candidate_count": len(managed_dna),
            "paper_legacy_management_position_count": len(
                legacy_management_dna
            ),
            "paper_configured_validation_position_cap": (
                settings.paper_automation.max_open_positions
            ),
            "paper_effective_position_cap": effective_position_cap,
            "paper_market_overlap_entries_allowed": False,
            "paper_evaluation_candidate_count": len(active_candidates),
            "paper_orders_placed": len(broker.orders),
            "paper_fills": len(broker.fills),
            "open_positions": len(positions),
            "reconciliation": asdict(reconciliation),
            "last_cycle_at": cycle_at,
            "last_reason": (
                "PAPER_ORDER_PROCESSED"
                if generated_orders
                else "NO_NATURAL_GENERATED_SIGNAL"
            ),
            "real_orders_placed": 0,
            "real_exchange_requests": 0,
            "auto_live_promotion": False,
        }
    )
    atomic_write_json(_paths(settings)["state"], state)
    return {
        **state,
        "orders_generated_this_cycle": generated_orders,
    }


def generated_paper_status(settings: Settings) -> dict[str, Any]:
    return _state(settings)


def load_generated_candidates(settings: Settings) -> list[dict[str, Any]]:
    """Return deduplicated frozen exact-positive candidates for downstream use."""

    return _load_candidates(settings)


__all__ = [
    "freeze_generated_candidates",
    "generated_paper_status",
    "load_generated_candidates",
    "refresh_simple_lab_positive_candidates",
    "run_generated_paper_once",
]
