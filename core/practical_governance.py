"""Execution-first governance, universe and portfolio control artifacts.

The module deliberately separates historical economic eligibility from live
authority.  Academic multiple-testing evidence changes confidence and scaling,
while integrity, economic validity and operational safety remain fail-closed.
It never submits an exchange order.
"""

from __future__ import annotations

import csv
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config.settings import SUPPORTED_TIMEFRAMES, Settings
from reporting.top_existing_strategies import collect_longlist, score_candidates
from research.strategies import describe_strategies
from utils.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    read_json,
    stable_hash,
    utc_iso,
)

PRIMARY_STRATEGY_ID = "RR_B60_H5_Z20"
PRIMARY_STRATEGY_DNA = "4571ae8e81aeb4299367643922061e2eabb6523c892ec9a63f08d33f32a939d0"
PRIMARY_MARKET = "ETH-EUR"
PRIMARY_TIMEFRAME = "1d"

REQUESTED_TIMEFRAMES = tuple(SUPPORTED_TIMEFRAMES)

ACADEMIC_WARNING_FIELDS = {
    "deflated_sharpe": "DSR_FAILED",
    "ordinary_pbo": "PBO_FAILED",
    "selection_pbo": "SELECTION_PBO_FAILED",
    "white_reality_check": "WHITE_REALITY_CHECK_FAILED",
    "hansen_spa": "HANSEN_SPA_FAILED",
}

HARD_BLOCKERS = {
    "LOOKAHEAD",
    "REPAINTING",
    "FUTURE_DATA",
    "CORRUPT_MARKET_DATA",
    "STRATEGY_DNA_MISMATCH",
    "UNRELIABLE_TIMESTAMPS",
    "SYNTHETIC_PERFORMANCE_EVIDENCE",
    "NET_EXPECTANCY_NOT_POSITIVE",
    "PROFIT_FACTOR_NOT_ABOVE_ONE",
    "NET_RETURN_NOT_POSITIVE",
    "COSTS_NOT_INCLUDED",
    "ENTRY_MISSING",
    "EXIT_MISSING",
    "RISK_UNBOUNDED",
    "MARKET_NOT_EXECUTABLE",
    "STALE_DATA",
    "EXCHANGE_OFFLINE",
    "RECONCILIATION_MISMATCH",
    "UNKNOWN_EXCHANGE_ORDER",
    "UNKNOWN_POSITION",
    "KILL_SWITCH_ACTIVE",
    "UNSAFE_API_SCOPE",
    "INVALID_QUANTITY_OR_PRECISION",
    "ORDER_CAP_EXCEEDED",
    "DAILY_LOSS_LIMIT",
    "MAXIMUM_DRAWDOWN_LIMIT",
}


class PracticalLifecycle(StrEnum):
    REJECT = "REJECT"
    BACKTEST_POSITIVE = "BACKTEST_POSITIVE"
    RESEARCH_POSITIVE = "RESEARCH_POSITIVE"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    LIVE_CANARY_ELIGIBLE = "LIVE_CANARY_ELIGIBLE"
    LIVE_CANARY_ACTIVE = "LIVE_CANARY_ACTIVE"
    LIVE_VALIDATED = "LIVE_VALIDATED"
    CAPITAL_SCALE_ELIGIBLE = "CAPITAL_SCALE_ELIGIBLE"
    PORTFOLIO_ACTIVE = "PORTFOLIO_ACTIVE"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"


class AllocationMode(StrEnum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    MICRO_ONLY = "MICRO_ONLY"
    PAPER_ONLY = "PAPER_ONLY"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class GovernancePaths:
    root: Path

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def governance(self) -> Path:
        return self.output / "governance"

    @property
    def strategies(self) -> Path:
        return self.output / "strategies"

    @property
    def universe(self) -> Path:
        return self.output / "universe"

    @property
    def portfolio(self) -> Path:
        return self.output / "portfolio"

    @property
    def autopilot(self) -> Path:
        return self.output / "autopilot"

    @property
    def paper(self) -> Path:
        return self.output / "paper"

    def ensure(self) -> None:
        for path in (
            self.governance,
            self.strategies,
            self.universe,
            self.portfolio,
            self.autopilot,
            self.paper,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _candidate_hard_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if candidate.get("lookahead_status") != "PASSED":
        blockers.append("LOOKAHEAD")
    if candidate.get("repainting_status") != "PASSED":
        blockers.append("REPAINTING")
    if not candidate.get("costs_included"):
        blockers.append("COSTS_NOT_INCLUDED")
    if float(candidate.get("normal_profit_factor") or 0.0) <= 1.0:
        blockers.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
    if float(candidate.get("net_total_return") or 0.0) <= 0.0:
        blockers.append("NET_RETURN_NOT_POSITIVE")
    if not str(candidate.get("entry_logic") or "").strip():
        blockers.append("ENTRY_MISSING")
    if not str(candidate.get("exit_logic") or "").strip():
        blockers.append("EXIT_MISSING")
    integrity = candidate.get("integrity") or {}
    if integrity.get("synthetic_data_used"):
        blockers.append("SYNTHETIC_PERFORMANCE_EVIDENCE")
    return sorted(set(blockers))


def _academic_warnings(candidate: Mapping[str, Any]) -> list[str]:
    evidence = candidate.get("statistical_evidence") or {}
    passes = evidence.get("test_passes") or {}
    warnings = [
        warning
        for field, warning in ACADEMIC_WARNING_FIELDS.items()
        if passes.get(field) is False
    ]
    holdout = str(candidate.get("holdout_status") or "")
    if not holdout or "NO_GLOBALLY_UNTOUCHED" in holdout or "MISSING" in holdout:
        warnings.append("UNTOUCHED_HOLDOUT_MISSING")
    if int(candidate.get("forward_observations") or 0) <= 0:
        warnings.append("PROSPECTIVE_SAMPLE_INSUFFICIENT")
    return sorted(set(warnings))


def _confidence_multiplier(candidate: Mapping[str, Any]) -> float:
    score = float((candidate.get("scores") or {}).get("composite") or 0.0)
    stressed_pf = candidate.get("stressed_profit_factor")
    if score >= 75.0 and stressed_pf is not None and float(stressed_pf) > 1.0:
        return 1.0
    if score >= 50.0:
        return 0.5
    return 0.25


def _strategy_record(
    candidate: Mapping[str, Any],
    *,
    approved_live_dna: set[str],
    settings: Settings,
) -> dict[str, Any]:
    blockers = _candidate_hard_blockers(candidate)
    warnings = _academic_warnings(candidate)
    strategy_id = str(candidate["strategy_name"])
    dna = str(candidate["strategy_dna_hash"])
    pf = float(candidate.get("normal_profit_factor") or 0.0)
    compatible = bool(candidate.get("bitvavo_spot_long_only_compatible"))
    research_positive = not blockers
    paper_adapter_available = strategy_id == PRIMARY_STRATEGY_ID
    paper_active = bool(
        research_positive
        and compatible
        and paper_adapter_available
        and settings.paper_automation.autotrade_enabled
        and settings.paper_automation.auto_promotion_from_research_positive
    )
    canary_eligible = bool(
        paper_active
        and pf >= settings.governance.minimum_canary_profit_factor
        and candidate.get("entry_logic")
        and candidate.get("exit_logic")
    )
    canary_active = bool(canary_eligible and dna in approved_live_dna)
    state = (
        PracticalLifecycle.LIVE_CANARY_ACTIVE
        if canary_active
        else PracticalLifecycle.LIVE_CANARY_ELIGIBLE
        if canary_eligible
        else PracticalLifecycle.PAPER_ACTIVE
        if paper_active
        else PracticalLifecycle.RESEARCH_POSITIVE
        if research_positive
        else PracticalLifecycle.REJECT
    )
    markets = list(candidate.get("assets_universe") or [])
    if strategy_id == PRIMARY_STRATEGY_ID:
        if dna != PRIMARY_STRATEGY_DNA:
            blockers.append("STRATEGY_DNA_MISMATCH")
            state = PracticalLifecycle.REJECT
            research_positive = paper_active = canary_eligible = canary_active = False
        markets = [PRIMARY_MARKET]
    return {
        "strategy_id": strategy_id,
        "strategy_dna": dna,
        "strategy_family": candidate.get("strategy_family"),
        "family_cluster": candidate.get("family_cluster"),
        "market": markets[0] if len(markets) == 1 else None,
        "markets": markets,
        "timeframe": candidate.get("timeframe"),
        "lifecycle_state": state.value,
        "backtest_positive": research_positive,
        "research_positive": research_positive,
        "paper_active": paper_active,
        "paper_adapter_available": paper_adapter_available,
        "paper_activation_pending": bool(
            research_positive and compatible and not paper_adapter_available
        ),
        "paper_activation_pending_reason": (
            "FROZEN_EXECUTION_ADAPTER_REQUIRED"
            if research_positive and compatible and not paper_adapter_available
            else None
        ),
        "live_canary_eligible": canary_eligible,
        "live_canary_active": canary_active,
        "live_validated": False,
        "capital_scale_eligible": False,
        "portfolio_active": False,
        "allocation_mode": (
            AllocationMode.MICRO_ONLY.value if canary_active else AllocationMode.PAPER_ONLY.value
        ),
        "normal_profit_factor": candidate.get("normal_profit_factor"),
        "stressed_profit_factor": candidate.get("stressed_profit_factor"),
        "double_cost_profit_factor": candidate.get("double_cost_profit_factor"),
        "net_total_return": candidate.get("net_total_return"),
        "net_cagr": candidate.get("net_cagr"),
        "maximum_drawdown": candidate.get("maximum_drawdown"),
        "expectancy": candidate.get("expectancy"),
        "sample_count": candidate.get("sample_count"),
        "sample_unit": candidate.get("sample_unit"),
        "historical_score": (candidate.get("scores") or {}).get("historical_performance"),
        "composite_score": (candidate.get("scores") or {}).get("composite"),
        "paper_risk_multiplier": _confidence_multiplier(candidate),
        "costs_included": candidate.get("costs_included"),
        "lookahead_status": candidate.get("lookahead_status"),
        "repainting_status": candidate.get("repainting_status"),
        "entry_logic": candidate.get("entry_logic"),
        "exit_logic": candidate.get("exit_logic"),
        "stop_logic": (
            "Risk-bounded strategy stop/exposure exit; concrete execution stop is frozen "
            "with the strategy adapter."
        ),
        "hard_blockers": sorted(set(blockers)),
        "capital_scaling_warnings": warnings,
        "academic_tests_are_execution_blockers": False,
        "operator_approval_required_for_live": True,
        "operator_approved_for_live": canary_active,
        "source_evidence": candidate.get("evidence"),
        "frozen": True,
        "parameters": candidate.get("parameters"),
    }


def _load_live_approved_dna(root: Path) -> set[str]:
    path = root / "config" / "live_strategy_approvals.yaml"
    if not path.is_file():
        return set()
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    strategies = raw.get("strategies") or {}
    return {
        str(row.get("strategy_dna_hash"))
        for row in strategies.values()
        if isinstance(row, dict) and row.get("approved_for_live") is True
    }


def activate_live_canary_authority(
    root: Path,
    settings: Settings,
    *,
    strategy_id: str,
    approval_phrase: str,
) -> dict[str, Any]:
    """Persist scoped Level-1 authority without storing the approval phrase."""

    if not secrets.compare_digest(
        approval_phrase.strip(),
        settings.execution.required_manual_approval_phrase,
    ):
        raise PermissionError("live canary approval phrase does not match")
    governance = read_json(
        root.resolve() / "output" / "governance" / "reclassified_strategies.json"
    )
    record = next(
        (
            row
            for row in governance.get("records", [])
            if row.get("strategy_id") == strategy_id
        ),
        None,
    )
    if record is None:
        raise KeyError(f"unknown practical-governance strategy: {strategy_id}")
    if strategy_id != PRIMARY_STRATEGY_ID or record.get("strategy_dna") != PRIMARY_STRATEGY_DNA:
        raise PermissionError("this operator approval is scoped only to the frozen RR DNA")
    if not record.get("live_canary_eligible"):
        raise PermissionError("strategy is not live-canary eligible")
    if PRIMARY_MARKET not in set(record.get("markets") or []):
        raise PermissionError("approved market is absent from strategy evidence")
    registry_dna = _load_live_approved_dna(root.resolve())
    if PRIMARY_STRATEGY_DNA not in registry_dna:
        raise PermissionError("immutable live approval registry is not approved")
    payload = {
        "schema_version": "live_canary_authority_v1",
        "active": True,
        "activated_at": utc_iso(),
        "strategy_id": PRIMARY_STRATEGY_ID,
        "strategy_dna": PRIMARY_STRATEGY_DNA,
        "market": PRIMARY_MARKET,
        "timeframe": PRIMARY_TIMEFRAME,
        "maximum_order_eur": 10.0,
        "maximum_total_exposure_eur": 10.0,
        "maximum_open_positions": 1,
        "maximum_new_orders_per_day": 1,
        "maximum_risk_per_trade_eur": 2.0,
        "maximum_daily_loss_eur": 5.0,
        "maximum_drawdown_eur": 10.0,
        "spot_only": True,
        "autoscale": False,
        "operator_approval_reference": "explicit_rr_canary_authority_20260727",
        "approval_phrase_stored": False,
    }
    path = root.resolve() / "output" / "governance" / "live_canary_authority.json"
    atomic_write_json(path, payload)
    append_jsonl(
        root.resolve() / "output" / "governance" / "live_canary_authority_audit.jsonl",
        {
            "event": "LIVE_CANARY_ACTIVATED",
            "recorded_at": payload["activated_at"],
            "strategy_id": PRIMARY_STRATEGY_ID,
            "strategy_dna": PRIMARY_STRATEGY_DNA,
            "market": PRIMARY_MARKET,
            "caps_hash": stable_hash(
                {
                    key: payload[key]
                    for key in (
                        "maximum_order_eur",
                        "maximum_total_exposure_eur",
                        "maximum_open_positions",
                        "maximum_new_orders_per_day",
                        "spot_only",
                        "autoscale",
                    )
                }
            ),
            "approval_phrase_stored": False,
        },
    )
    return payload


def deactivate_live_canary_authority(root: Path, *, reason: str) -> dict[str, Any]:
    path = root.resolve() / "output" / "governance" / "live_canary_authority.json"
    current = dict(read_json(path)) if path.is_file() else {}
    payload = {
        **current,
        "schema_version": "live_canary_authority_v1",
        "active": False,
        "deactivated_at": utc_iso(),
        "deactivation_reason": str(reason),
        "approval_phrase_stored": False,
    }
    atomic_write_json(path, payload)
    append_jsonl(
        root.resolve() / "output" / "governance" / "live_canary_authority_audit.jsonl",
        {
            "event": "LIVE_CANARY_DEACTIVATED",
            "recorded_at": payload["deactivated_at"],
            "strategy_id": current.get("strategy_id"),
            "strategy_dna": current.get("strategy_dna"),
            "reason": str(reason),
        },
    )
    return payload


def live_canary_authority(
    root: Path,
    *,
    strategy_id: str = PRIMARY_STRATEGY_ID,
    strategy_dna: str = PRIMARY_STRATEGY_DNA,
    market: str = PRIMARY_MARKET,
) -> tuple[bool, dict[str, Any], list[str]]:
    path = root.resolve() / "output" / "governance" / "live_canary_authority.json"
    if not path.is_file():
        return False, {}, ["LIVE_CANARY_AUTHORITY_MISSING"]
    payload = dict(read_json(path))
    failures: list[str] = []
    if payload.get("active") is not True:
        failures.append("LIVE_CANARY_AUTHORITY_INACTIVE")
    if payload.get("strategy_id") != strategy_id:
        failures.append("LIVE_CANARY_STRATEGY_ID_MISMATCH")
    if payload.get("strategy_dna") != strategy_dna:
        failures.append("LIVE_CANARY_STRATEGY_DNA_MISMATCH")
    if payload.get("market") != market:
        failures.append("LIVE_CANARY_MARKET_MISMATCH")
    expected = {
        "maximum_order_eur": 10.0,
        "maximum_total_exposure_eur": 10.0,
        "maximum_open_positions": 1,
        "maximum_new_orders_per_day": 1,
        "spot_only": True,
        "autoscale": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(f"LIVE_CANARY_CAP_MISMATCH:{key}")
    return not failures, payload, failures


def reclassify_existing_strategies(
    root: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Reclassify repository evidence without running a new parameter search."""

    paths = GovernancePaths(root.resolve())
    paths.ensure()
    candidates = score_candidates(collect_longlist(paths.root))
    approved_dna = _load_live_approved_dna(paths.root)
    records = [
        _strategy_record(row, approved_live_dna=approved_dna, settings=settings)
        for row in candidates
    ]
    classical_paths = (
        paths.strategies / "classical_backtest_positive.json",
        paths.strategies / "simple_lab_backtest_positive.json",
    )
    classical_candidates: list[tuple[Path, dict[str, Any]]] = []
    for registry_path in classical_paths:
        if not registry_path.is_file():
            continue
        registry = dict(read_json(registry_path))
        classical_candidates.extend(
            (registry_path, dict(candidate))
            for candidate in registry.get("candidates") or []
        )
    if classical_candidates:
        existing_dna = {str(row["strategy_dna"]) for row in records}
        for classical_path, candidate in classical_candidates:
            dna = str(candidate.get("strategy_dna_hash") or "")
            metrics = dict(candidate.get("metrics") or {})
            integrity = dict(candidate.get("integrity") or {})
            if not dna or dna in existing_dna:
                continue
            blockers = []
            if integrity.get("no_lookahead") is not True:
                blockers.append("LOOKAHEAD")
            if integrity.get("no_repainting") is not True:
                blockers.append("REPAINTING")
            if float(metrics.get("profit_factor") or 0.0) <= 1.0:
                blockers.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
            if float(metrics.get("net_return") or 0.0) <= 0.0:
                blockers.append("NET_RETURN_NOT_POSITIVE")
            research_positive = not blockers
            paper_active = bool(
                research_positive
                and settings.paper_automation.autotrade_enabled
                and settings.paper_automation.auto_promotion_from_research_positive
            )
            records.append(
                {
                    "strategy_id": (
                        f"CLASSICAL_{candidate.get('economic_hypothesis_family')}_"
                        f"{dna[:8]}"
                    ),
                    "strategy_dna": dna,
                    "strategy_family": candidate.get("economic_hypothesis_family"),
                    "family_cluster": "CLASSICAL_GENERATED",
                    "market": None,
                    "markets": list(candidate.get("markets") or []),
                    "timeframe": candidate.get("timeframe"),
                    "lifecycle_state": (
                        PracticalLifecycle.PAPER_ACTIVE.value
                        if paper_active
                        else PracticalLifecycle.RESEARCH_POSITIVE.value
                        if research_positive
                        else PracticalLifecycle.REJECT.value
                    ),
                    "backtest_positive": research_positive,
                    "research_positive": research_positive,
                    "paper_active": paper_active,
                    "paper_adapter_available": True,
                    "paper_activation_pending": False,
                    "paper_activation_pending_reason": None,
                    "live_canary_eligible": False,
                    "live_canary_active": False,
                    "live_validated": False,
                    "capital_scale_eligible": False,
                    "portfolio_active": False,
                    "allocation_mode": AllocationMode.PAPER_ONLY.value,
                    "normal_profit_factor": metrics.get("profit_factor"),
                    # Never relabel the normal-cost result as stressed
                    # evidence. Generated registries expose an explicit
                    # stressed metric only after that run actually exists.
                    "stressed_profit_factor": metrics.get(
                        "stressed_profit_factor"
                    ),
                    "double_cost_profit_factor": None,
                    "net_total_return": metrics.get("net_return"),
                    "net_cagr": metrics.get("cagr"),
                    "maximum_drawdown": metrics.get("maximum_drawdown"),
                    "expectancy": metrics.get("net_expectancy_r"),
                    "sample_count": metrics.get("trade_count"),
                    "sample_unit": "closed_round_trips",
                    "historical_score": None,
                    "composite_score": None,
                    "paper_risk_multiplier": 0.25,
                    "costs_included": True,
                    "lookahead_status": (
                        "PASSED" if integrity.get("no_lookahead") is True else "FAILED"
                    ),
                    "repainting_status": (
                        "PASSED" if integrity.get("no_repainting") is True else "FAILED"
                    ),
                    "entry_logic": " + ".join(candidate.get("block_ids") or []),
                    "exit_logic": (
                        "Frozen canonical exit blocks plus ATR stop/target and "
                        "maximum-holding exit."
                    ),
                    "stop_logic": "Frozen canonical ATR risk distance.",
                    "hard_blockers": blockers,
                    "capital_scaling_warnings": [
                        "SMALL_EXACT_SAMPLE",
                        "UNTOUCHED_HOLDOUT_MISSING",
                        "PROSPECTIVE_SAMPLE_INSUFFICIENT",
                    ],
                    "academic_tests_are_execution_blockers": False,
                    "operator_approval_required_for_live": True,
                    "operator_approved_for_live": False,
                    "source_evidence": [
                        str(classical_path),
                        str(
                            (
                                paths.root
                                / "output"
                                / "research"
                                / "simple_strategy_lab"
                                / "canonical_result_evidence.json"
                            )
                            if classical_path.name
                            == "simple_lab_backtest_positive.json"
                            else (
                                paths.root
                                / "output"
                                / "lab"
                                / "reports"
                                / "classical_strategy_factory_v1_report.json"
                            )
                        ),
                    ],
                    "frozen": True,
                    "frozen_candidate_hash": candidate.get("frozen_candidate_hash"),
                    "parameters": candidate.get("parameters"),
                }
            )
            existing_dna.add(dna)
    adaptive_path = (
        paths.root
        / "output"
        / "lab"
        / "reports"
        / "adaptive_crypto_intraday_v1.json"
    )
    if adaptive_path.is_file():
        adaptive_report = dict(read_json(adaptive_path))
        existing_dna = {str(row["strategy_dna"]) for row in records}
        for candidate in adaptive_report.get("candidates") or []:
            dna = str(candidate.get("strategy_dna_hash") or "")
            if not dna or dna in existing_dna:
                continue
            normal = dict(candidate.get("normal") or {})
            stressed = dict(candidate.get("stressed") or {})
            metrics = dict(normal.get("metrics") or {})
            stressed_metrics = dict(stressed.get("metrics") or {})
            integrity = dict(normal.get("integrity") or {})
            profit_factor = float(
                metrics.get("portfolio_period_profit_factor") or 0.0
            )
            net_return = float(metrics.get("net_return") or 0.0)
            hard_blockers: list[str] = []
            if net_return <= 0.0:
                hard_blockers.append("NET_RETURN_NOT_POSITIVE")
            if profit_factor <= 1.0:
                hard_blockers.append("PROFIT_FACTOR_NOT_ABOVE_ONE")
            causal = bool(
                integrity.get("no_lookahead") is True
                or integrity.get("strictly_prior_beta_estimation") is True
            )
            if not causal:
                hard_blockers.append("LOOKAHEAD")
            research_positive = not hard_blockers
            compatible = (
                candidate.get("universe_label")
                == "PROMOTION_COMPATIBLE"
            )
            paper_adapter = candidate.get("paper_adapter")
            paper_active = bool(
                research_positive
                and compatible
                and paper_adapter
                and candidate.get("paper_eligible") is True
                and settings.paper_automation.autotrade_enabled
                and settings.paper_automation.auto_promotion_from_research_positive
            )
            records.append(
                {
                    "strategy_id": candidate.get("strategy_id"),
                    "strategy_dna": dna,
                    "strategy_family": candidate.get("strategy_family"),
                    "family_cluster": "ADAPTIVE_INTRADAY",
                    "market": None,
                    "markets": list(candidate.get("universe") or []),
                    "timeframe": candidate.get("timeframe"),
                    "lifecycle_state": (
                        PracticalLifecycle.PAPER_ACTIVE.value
                        if paper_active
                        else PracticalLifecycle.RESEARCH_POSITIVE.value
                        if research_positive
                        else PracticalLifecycle.REJECT.value
                    ),
                    "backtest_positive": research_positive,
                    "research_positive": research_positive,
                    "paper_active": paper_active,
                    "paper_adapter_available": bool(paper_adapter),
                    "paper_activation_pending": bool(
                        research_positive and not paper_active
                    ),
                    "paper_activation_pending_reason": (
                        "DISCOVERY_UNIVERSE_NOT_EXECUTION_APPROVED"
                        if research_positive and not compatible
                        else "PAPER_EVIDENCE_GATE_FAILED"
                        if research_positive
                        and candidate.get("paper_eligible") is not True
                        else "PAPER_ADAPTER_PENDING"
                        if research_positive and not paper_adapter
                        else None
                    ),
                    "live_canary_eligible": False,
                    "live_canary_active": False,
                    "live_validated": False,
                    "capital_scale_eligible": False,
                    "portfolio_active": False,
                    "allocation_mode": AllocationMode.PAPER_ONLY.value,
                    "normal_profit_factor": profit_factor,
                    "stressed_profit_factor": stressed_metrics.get(
                        "portfolio_period_profit_factor",
                    ),
                    "double_cost_profit_factor": None,
                    "net_total_return": net_return,
                    "net_cagr": metrics.get("annualized_return"),
                    "maximum_drawdown": metrics.get("maximum_drawdown"),
                    "expectancy": (
                        net_return
                        / max(
                            1,
                            int(
                                metrics.get(
                                    "closed_position_episodes",
                                )
                                or metrics.get(
                                    "raw_portfolio_period_observations",
                                )
                                or 0
                            ),
                        )
                    ),
                    "sample_count": (
                        metrics.get("closed_position_episodes")
                        or metrics.get(
                            "portfolio_period_effective_sample_size",
                        )
                    ),
                    "sample_unit": "closed_round_trips_or_effective_periods",
                    "historical_score": None,
                    "composite_score": None,
                    "paper_risk_multiplier": 0.25,
                    "costs_included": True,
                    "lookahead_status": (
                        "PASSED" if causal else "FAILED"
                    ),
                    "repainting_status": "PASSED",
                    "entry_logic": (
                        "Frozen adaptive percentile residual entry."
                        if "ADAPTIVE_PERCENTILE" in str(
                            candidate.get("strategy_family"),
                        )
                        else "Prior 120-bar Donchian breakout above EMA600."
                    ),
                    "exit_logic": (
                        "Prior 60-bar channel/EMA exit at next real open."
                    ),
                    "stop_logic": "ATR(14) x2 EUR-risk distance.",
                    "hard_blockers": hard_blockers,
                    "capital_scaling_warnings": list(
                        (
                            candidate.get("stochastic_validation")
                            or {}
                        ).get("reason_codes")
                        or []
                    )
                    + [
                        "UNTOUCHED_HOLDOUT_MISSING",
                        "PROSPECTIVE_SAMPLE_INSUFFICIENT",
                    ],
                    "academic_tests_are_execution_blockers": False,
                    "operator_approval_required_for_live": True,
                    "operator_approved_for_live": False,
                    "source_evidence": [str(adaptive_path)],
                    "frozen": True,
                    "parameters": candidate.get("parameters"),
                    "paper_adapter": paper_adapter,
                }
            )
            existing_dna.add(dna)
    records.sort(
        key=lambda row: (
            bool(row["research_positive"]),
            float(row["composite_score"] or 0.0),
        ),
        reverse=True,
    )

    evidence_dna = {row["strategy_dna"] for row in records}
    registered = []
    implementation_descriptions = list(describe_strategies())

    # Tactical and market-mechanics strategies deliberately live outside the
    # legacy single-frame Strategy registry.  They still need a durable DNA in
    # governance so the research autopilot can keep retesting them.  Merely
    # registering these implementations does not grant paper or live authority.
    from research.gex_orderflow_strategies import (
        market_mechanics_strategy_specs,
    )
    from research.tactical_multitimeframe import tactical_strategy_specs

    implementation_descriptions.extend(
        {
            "strategy_id": spec.strategy_id,
            "family": spec.family,
            "timeframe": spec.timeframe,
            "strategy_dna": spec.dna_hash,
            "implementation_class": "TACTICAL_MULTI_TIMEFRAME",
            "confirmation_timeframe": spec.confirmation_timeframe,
            "regime_timeframe": spec.regime_timeframe,
            "mechanism": spec.mechanism,
            "closed_candle_only": True,
            "next_open_execution": True,
            "live_authority_granted": False,
        }
        for spec in tactical_strategy_specs()
    )
    implementation_descriptions.extend(
        {
            "strategy_id": spec.strategy_id,
            "family": spec.family,
            "timeframe": spec.entry_timeframe,
            "strategy_dna": spec.dna_hash,
            "implementation_class": "GEX_ORDERFLOW_MULTI_TIMEFRAME",
            "confirmation_timeframe": "2h",
            "regime_timeframe": "4h",
            "mechanism": spec.mechanism,
            "gex_policy": spec.gex_policy,
            "prospective_orderflow_only": True,
            "closed_candle_only": True,
            "next_open_execution": True,
            "live_authority_granted": False,
        }
        for spec in market_mechanics_strategy_specs()
    )
    for description in implementation_descriptions:
        dna = str(description.get("strategy_dna") or stable_hash(description))
        if dna in evidence_dna:
            continue
        registered.append(
            {
                "strategy_id": description["strategy_id"],
                "strategy_dna": dna,
                "strategy_family": description.get("family"),
                "timeframe": description.get("timeframe"),
                "lifecycle_state": "DATA_PENDING",
                "backtest_positive": False,
                "paper_active": False,
                "live_canary_eligible": False,
                "hard_blockers": [],
                "capital_scaling_warnings": [
                    (
                        "PROSPECTIVE_GEX_ORDERFLOW_EVIDENCE_ACCUMULATING"
                        if description.get("implementation_class")
                        == "GEX_ORDERFLOW_MULTI_TIMEFRAME"
                        else "ECONOMIC_EVIDENCE_NOT_MAPPED"
                    )
                ],
                "metadata": description,
            }
        )

    positives = [row for row in records if row["research_positive"]]
    paper = [row for row in records if row["paper_active"]]
    canary = [row for row in records if row["live_canary_eligible"]]
    live_validated = [row for row in records if row["live_validated"]]
    degraded = [row for row in records if row["lifecycle_state"] == "DEGRADED"]
    generated_at = utc_iso()
    summary = {
        "schema_version": "practical_governance_v1",
        "generated_at": generated_at,
        "practical_governance_enabled": settings.governance.practical_enabled,
        "academic_tests_are_warnings_for_canary": (
            settings.governance.academic_tests_are_warnings_for_canary
        ),
        "unique_economic_candidates": len(records),
        "registered_strategy_implementations": len(registered),
        "research_positive": len(positives),
        "paper_active": len(paper),
        "paper_adapter_pending": sum(
            row["paper_activation_pending"] for row in records
        ),
        "live_canary_eligible": len(canary),
        "live_canary_active": sum(row["live_canary_active"] for row in records),
        "live_validated": len(live_validated),
        "hard_rejects": sum(bool(row["hard_blockers"]) for row in records),
        "formerly_fully_rejected_now_research_positive": sum(
            row["research_positive"]
            and bool(row["capital_scaling_warnings"])
            for row in records
        ),
        "hard_blocker_catalog": sorted(HARD_BLOCKERS),
        "records": records,
    }
    atomic_write_json(paths.governance / "reclassified_strategies.json", summary)
    _write_strategy_csv(paths.governance / "reclassified_strategies.csv", records)
    atomic_write_text(
        paths.governance / "reclassification_report.md",
        _reclassification_markdown(summary),
    )
    atomic_write_json(
        paths.strategies / "all_strategy_dna.json",
        {
            "generated_at": generated_at,
            "economic_evidence": records,
            "registered_pending": registered,
        },
    )
    atomic_write_json(paths.strategies / "backtest_positive.json", positives)
    atomic_write_json(paths.strategies / "paper_active.json", paper)
    atomic_write_json(paths.strategies / "live_canary_queue.json", canary)
    atomic_write_json(paths.strategies / "live_validated.json", live_validated)
    atomic_write_json(paths.strategies / "degraded.json", degraded)
    _append_new_promotions(paths, paper)
    return summary


def _write_strategy_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "strategy_id",
        "strategy_dna",
        "strategy_family",
        "market",
        "timeframe",
        "lifecycle_state",
        "normal_profit_factor",
        "stressed_profit_factor",
        "net_total_return",
        "net_cagr",
        "maximum_drawdown",
        "sample_count",
        "paper_risk_multiplier",
        "paper_active",
        "live_canary_eligible",
        "live_canary_active",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _reclassification_markdown(summary: Mapping[str, Any]) -> str:
    return (
        "# Practical governance reclassification\n\n"
        f"Generated: {summary['generated_at']}\n\n"
        f"- Unique economic candidates: {summary['unique_economic_candidates']}\n"
        f"- Research positive: {summary['research_positive']}\n"
        f"- Paper active: {summary['paper_active']}\n"
        f"- Live canary eligible: {summary['live_canary_eligible']}\n"
        f"- Live canary active: {summary['live_canary_active']}\n"
        f"- Live validated: {summary['live_validated']}\n"
        f"- Academic-warning candidates restored to research/paper: "
        f"{summary['formerly_fully_rejected_now_research_positive']}\n\n"
        "Academic tests remain visible capital-scaling warnings. Integrity, positive "
        "net edge, costs, entry/exit identity and operational safety remain fail-closed.\n"
    )


def _append_new_promotions(paths: GovernancePaths, paper: Iterable[Mapping[str, Any]]) -> None:
    ledger_path = paths.autopilot / "promotions.jsonl"
    existing: set[str] = set()
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                row = __import__("json").loads(line)
            except ValueError:
                continue
            existing.add(str(row.get("promotion_id")))
    for row in paper:
        promotion_id = stable_hash(
            {
                "strategy_dna": row["strategy_dna"],
                "target": PracticalLifecycle.PAPER_ACTIVE.value,
            }
        )
        if promotion_id in existing:
            continue
        append_jsonl(
            ledger_path,
            {
                "promotion_id": promotion_id,
                "recorded_at": utc_iso(),
                "strategy_id": row["strategy_id"],
                "strategy_dna": row["strategy_dna"],
                "from": PracticalLifecycle.RESEARCH_POSITIVE.value,
                "to": PracticalLifecycle.PAPER_ACTIVE.value,
                "automatic": True,
                "live_order_authorized": False,
            },
        )


def _snapshot_source(root: Path) -> dict[str, Any]:
    checkpoint = read_json(root / "output" / "checkpoints" / "prospective_context_hourly.json")
    if checkpoint.get("status") != "PASSED":
        raise RuntimeError("point-in-time universe checkpoint is not PASSED")
    snapshot_path = Path(str(checkpoint["snapshot_path"]))
    snapshot = read_json(snapshot_path)
    if snapshot.get("synthetic_data_used") is True:
        raise RuntimeError("synthetic universe snapshot cannot be promoted")
    return snapshot


def build_top50_universe(
    root: Path,
    settings: Settings,
    *,
    venue_markets: set[str] | None = None,
) -> dict[str, Any]:
    """Build an immutable point-in-time top-50 eligibility snapshot."""

    paths = GovernancePaths(root.resolve())
    paths.ensure()
    snapshot = _snapshot_source(paths.root)
    source_rows = list(snapshot.get("coinmarketcap_top50") or [])
    if len(source_rows) != 50:
        raise RuntimeError(f"top-50 source contains {len(source_rows)} rows")
    known_venue = {
        str(value).upper().replace("/", "-")
        for value in (venue_markets or set(settings.market_data.symbols))
    }
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        values = source.get("values") or {}
        tags = {str(tag).casefold() for tag in values.get("tags") or []}
        symbol = str(values.get("symbol") or "").upper()
        market = f"{symbol}-EUR"
        stablecoin = "stablecoin" in tags or symbol in {"USDT", "USDC", "DAI", "USDD", "RLUSD"}
        wrapped = "wrapped-tokens" in tags or symbol in {
            "WBTC",
            "WETH",
            "WBNB",
            "WAVAX",
            "WSOL",
        }
        leveraged = any(token in symbol for token in ("3L", "3S", "BULL", "BEAR", "UP", "DOWN"))
        staking_derivative = bool(
            {"liquid-staking-derivatives", "liquid-staking-tokens"} & tags
            or symbol in {"STETH", "WSTETH", "RETH", "WBETH"}
        )
        venue_available = market in known_venue
        shariah = settings.shariah.eligibility(market)
        shariah_status = shariah.status.value
        volume = float(values.get("volume_24h") or 0.0)
        context_only = stablecoin or wrapped or leveraged or staking_derivative
        execution_eligible = bool(
            venue_available
            and not context_only
            and shariah_status == "ALLOWED"
            and volume >= settings.lab.minimum_volume_24h_eur
        )
        rows.append(
            {
                "rank": int(values["cmc_rank"]),
                "symbol": symbol,
                "name": values.get("name"),
                "market_cap": values.get("market_cap"),
                "circulating_supply": values.get("circulating_supply"),
                "volume_24h": values.get("volume_24h"),
                "venue_availability": venue_available,
                "eur_spot_market": market if venue_available else None,
                "liquidity": "ELIGIBLE" if volume >= settings.lab.minimum_volume_24h_eur else "LOW",
                "spread_bps": None,
                "minimum_order_eur": None,
                "amount_precision": None,
                "price_precision": None,
                "stablecoin": stablecoin,
                "wrapped": wrapped,
                "leveraged_token": leveraged,
                "staking_derivative": staking_derivative,
                "shariah_status": shariah_status,
                "research_eligibility": "CONTEXT_ONLY" if context_only else "RESEARCH_ELIGIBLE",
                "execution_eligibility": (
                    "LIVE_ELIGIBLE" if execution_eligible else "NOT_EXECUTION_ELIGIBLE"
                ),
                "execution_reason": (
                    "PASSED"
                    if execution_eligible
                    else "CONTEXT_ONLY_ASSET"
                    if context_only
                    else "VENUE_MARKET_MISSING"
                    if not venue_available
                    else "SHARIAH_REVIEW_REQUIRED"
                    if shariah_status != "ALLOWED"
                    else "LIQUIDITY_BELOW_THRESHOLD"
                ),
                "snapshot_timestamp": source.get("timestamp"),
                "available_at": source.get("available_at"),
                "raw_hash": source.get("raw_hash"),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    payload = {
        "schema_version": "top50_point_in_time_v1",
        "generated_at": utc_iso(),
        "source_snapshot_hash": snapshot.get("snapshot_hash"),
        "source_collected_at": snapshot.get("collected_at"),
        "count": len(rows),
        "available_markets": sum(row["venue_availability"] for row in rows),
        "execution_eligible": sum(
            row["execution_eligibility"] == "LIVE_ELIGIBLE" for row in rows
        ),
        "synthetic_data_used": False,
        "rows": rows,
    }
    atomic_write_json(paths.universe / "top50_current.json", payload)
    atomic_write_json(
        paths.universe / "top50_eligibility.json",
        {
            "generated_at": payload["generated_at"],
            "source_snapshot_hash": payload["source_snapshot_hash"],
            "rows": [
                {
                    key: row[key]
                    for key in (
                        "rank",
                        "symbol",
                        "eur_spot_market",
                        "stablecoin",
                        "wrapped",
                        "leveraged_token",
                        "staking_derivative",
                        "shariah_status",
                        "research_eligibility",
                        "execution_eligibility",
                        "execution_reason",
                        "available_at",
                    )
                }
                for row in rows
            ],
        },
    )
    _append_top50_history(paths.universe / "top50_history.parquet", rows)
    return payload


def _append_top50_history(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if path.is_file():
        previous = pd.read_parquet(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=["rank", "symbol", "available_at"], keep="last")
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_portfolio_artifacts(
    root: Path,
    settings: Settings,
    governance: Mapping[str, Any],
) -> dict[str, Any]:
    paths = GovernancePaths(root.resolve())
    paths.ensure()
    now = utc_iso()
    paper = [row for row in governance["records"] if row["paper_active"]]
    total_score = sum(max(float(row["composite_score"] or 0.0), 0.0) for row in paper)
    risk_budgets = [
        {
            "strategy_id": row["strategy_id"],
            "strategy_dna": row["strategy_dna"],
            "timeframe": row["timeframe"],
            "paper_risk_multiplier": row["paper_risk_multiplier"],
            "relative_budget": (
                max(float(row["composite_score"] or 0.0), 0.0) / total_score
                if total_score
                else 0.0
            ),
            "live_budget_eur": 5.0 if row["live_canary_active"] else 0.0,
        }
        for row in paper
    ]
    def load_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        payload = read_json(path)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if pd.notna(result) else None

    def position_rows(
        state: Mapping[str, Any],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dna, raw in dict(state.get("positions") or {}).items():
            position = dict(raw)
            entry = number(position.get("entry_price"))
            quantity = number(position.get("quantity"))
            rows.append(
                {
                    **position,
                    "strategy_dna": str(
                        position.get("strategy_dna") or dna
                    ),
                    "source": source,
                    "entry_notional_eur": (
                        entry * quantity
                        if entry is not None and quantity is not None
                        else None
                    ),
                }
            )
        return rows

    primary_paper = load_mapping(paths.output / "paper" / "paper_state.json")
    generated_paper = load_mapping(
        paths.output / "paper" / "generated_strategy_state.json"
    )
    generated_live = load_mapping(
        paths.output / "live" / "generated_strategy_live_state.json"
    )
    current_canary = load_mapping(
        paths.output / "governance" / "current_position.json"
    )
    account_health = load_mapping(
        paths.output / "operations" / "live_account_health.json"
    )
    paper_positions = position_rows(
        primary_paper,
        source="PRIMARY_PAPER_BROKER",
    )
    if bool((generated_paper.get("reconciliation") or {}).get("healthy")):
        paper_positions.extend(
            position_rows(
                generated_paper,
                source="GENERATED_PAPER_BROKER_RECONCILED",
            )
        )
    live_reconciled = bool(
        (account_health.get("reconciliation") or {}).get("healthy")
    )
    live_positions = (
        position_rows(
            generated_live,
            source="GENERATED_LIVE_RECONCILED",
        )
        if live_reconciled
        else []
    )
    if live_reconciled and current_canary.get("position"):
        live_positions.extend(
            position_rows(
                {"positions": {PRIMARY_STRATEGY_DNA: current_canary["position"]}},
                source="PRIMARY_LIVE_CANARY_RECONCILED",
            )
        )

    account = dict(account_health.get("account") or {})
    valuation = dict(account.get("portfolio_valuation") or {})
    wallet_holdings = (
        [
            {
                **dict(row),
                "source": "BITVAVO_WALLET_MARK_TO_MARKET",
                "managed_strategy_position": any(
                    str(position.get("market")) == str(row.get("market"))
                    for position in live_positions
                ),
            }
            for row in valuation.get("holdings") or []
        ]
        if valuation.get("status") == "COMPLETE_MARK_TO_MARKET"
        else []
    )
    cash_eur = number(account.get("eur_available"))
    total_equity_eur = number(valuation.get("estimated_total_equity_eur"))
    cash_fraction = (
        cash_eur / total_equity_eur
        if (
            cash_eur is not None
            and total_equity_eur is not None
            and total_equity_eur > 0.0
        )
        else None
    )
    total_wallet_asset_exposure = sum(
        number(row.get("estimated_value_eur")) or 0.0
        for row in wallet_holdings
    )
    total_managed_live_exposure = sum(
        (number(row.get("entry_notional_eur")) or 0.0)
        for row in live_positions
    )
    current = {
        "generated_at": now,
        "cash_eur": cash_eur,
        "cash_fraction": cash_fraction,
        "estimated_total_equity_eur": total_equity_eur,
        "paper_positions": paper_positions,
        "live_positions": live_positions,
        "wallet_holdings": wallet_holdings,
        "total_live_exposure_eur": total_managed_live_exposure,
        "total_wallet_asset_exposure_eur": total_wallet_asset_exposure,
        "valuation_status": valuation.get("status") or "VALUATION_PENDING",
        "live_reconciliation_healthy": live_reconciled,
        "allocation_note": (
            "Wallet holdings are reported separately from strategy-managed "
            "positions; no strategy ownership is inferred without evidence."
        ),
    }
    buckets = {
        "intraday": settings.portfolio_allocation.intraday_bucket_pct,
        "swing_1h_4h": settings.portfolio_allocation.swing_bucket_pct,
        "daily": settings.portfolio_allocation.daily_bucket_pct,
        "weekly": settings.portfolio_allocation.weekly_bucket_pct,
        "monthly_rotation": settings.portfolio_allocation.monthly_bucket_pct,
    }
    atomic_write_json(paths.portfolio / "current_allocation.json", current)
    atomic_write_json(
        paths.portfolio / "strategy_risk_budgets.json",
        {"generated_at": now, "budgets": risk_budgets},
    )
    atomic_write_json(
        paths.portfolio / "timeframe_risk_budgets.json",
        {"generated_at": now, "buckets_pct": buckets, "dynamic": True},
    )
    atomic_write_json(
        paths.portfolio / "asset_exposures.json",
        {
            "generated_at": now,
            "exposures": wallet_holdings,
            "total_wallet_asset_exposure_eur": total_wallet_asset_exposure,
            "total_managed_live_exposure_eur": total_managed_live_exposure,
            "max_single_coin_pct": settings.portfolio_allocation.max_single_coin_exposure_pct,
        },
    )
    atomic_write_json(
        paths.portfolio / "correlation_clusters.json",
        {
            "generated_at": now,
            "clusters": [],
            "status": (
                "RECONCILED_POSITIONS_AVAILABLE"
                if live_positions
                else "NO_RECONCILED_MANAGED_OPEN_POSITIONS"
            ),
            "max_cluster_pct": (
                settings.portfolio_allocation.max_correlated_cluster_exposure_pct
            ),
        },
    )
    return {
        "current_allocation": current,
        "risk_budget_strategy_count": len(risk_budgets),
        "timeframe_buckets_pct": buckets,
    }


def governance_status(root: Path) -> dict[str, Any]:
    paths = GovernancePaths(root.resolve())
    report = read_json(paths.governance / "reclassified_strategies.json")
    rr = next(
        (row for row in report.get("records", []) if row["strategy_id"] == PRIMARY_STRATEGY_ID),
        None,
    )
    return {
        "practical_governance_enabled": report.get("practical_governance_enabled"),
        "academic_tests_are_warnings_for_canary": report.get(
            "academic_tests_are_warnings_for_canary"
        ),
        "counts": {
            key: report.get(key)
            for key in (
                "unique_economic_candidates",
                "research_positive",
                "paper_active",
                "live_canary_eligible",
                "live_canary_active",
                "live_validated",
            )
        },
        "rr": rr,
        "hard_blocker_catalog": report.get("hard_blocker_catalog"),
    }


def capital_level(
    *,
    flawless_round_trips: int,
    net_live_expectancy: float | None,
    operator_approved_level: int = 1,
) -> dict[str, Any]:
    eligible = 1
    if flawless_round_trips >= 3:
        eligible = 2
    if flawless_round_trips >= 10 and (net_live_expectancy or 0.0) >= 0.0:
        eligible = 3
    if flawless_round_trips >= 20 and (net_live_expectancy or 0.0) > 0.0:
        eligible = 4
    active = min(eligible, max(1, operator_approved_level))
    caps = {
        1: {"max_order_eur": 10.0, "max_exposure_eur": 10.0, "max_positions": 1},
        2: {"max_order_eur": 25.0, "max_exposure_eur": 75.0, "max_positions": 2},
        3: {"max_order_eur": 100.0, "max_exposure_eur": 300.0, "max_positions": 3},
        4: {
            "max_order_eur": None,
            "max_exposure_pct": 5.0,
            "max_positions": 3,
            "max_risk_per_trade_pct": 0.25,
        },
    }
    return {
        "active_level": active,
        "eligible_level": eligible,
        "operator_approval_required_to_raise": eligible > active,
        "caps": caps[active],
        "autoscale": False,
    }


def live_capital_evidence(
    root: Path,
    *,
    strategy_id: str = PRIMARY_STRATEGY_ID,
) -> dict[str, Any]:
    """Reconstruct strategy-scoped round trips from the canonical live ledger.

    The canonical ledger also contains fills from approved event playbooks,
    generated exact DNA and explicit inventory reallocations.  Those fills are
    valid live activity, but they are not evidence for scaling ``strategy_id``.
    Modern records are therefore filtered by immutable strategy identity.  A
    legacy record without a strategy identity retains the original market-only
    fallback so historical RR evidence remains readable.
    """

    ledger_path = root.resolve() / "output" / "checkpoints" / "live_execution.jsonl"
    if not ledger_path.is_file():
        return {
            "ledger": str(ledger_path),
            "flawless_round_trips": 0,
            "net_live_expectancy": None,
            "critical_incidents": [],
            "open_quantity": 0.0,
            "evidence_scope_strategy_id": strategy_id,
            "out_of_scope_fill_count": 0,
        }
    critical_event_types = {
        "AMBIGUOUS_ORDER",
        "DUPLICATE_ORDER",
        "RECONCILIATION_MISMATCH",
        "UNKNOWN_ORDER",
        "UNKNOWN_POSITION",
    }
    critical_incidents: list[str] = []
    quantity = 0.0
    cost_basis = 0.0
    episode_realized_pnl = 0.0
    round_trip_pnl: list[float] = []
    out_of_scope_fill_count = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            critical_incidents.append("INVALID_LEDGER_JSON")
            continue
        event_type = str(event.get("event_type") or "")
        if event_type in critical_event_types:
            critical_incidents.append(event_type)
        if event_type != "FILL":
            continue
        payload = event.get("payload") or {}
        fill_strategy_id = str(payload.get("strategy_id") or "").strip()
        if fill_strategy_id and fill_strategy_id != strategy_id:
            out_of_scope_fill_count += 1
            continue
        if str(payload.get("market") or "").upper() != PRIMARY_MARKET:
            critical_incidents.append("UNEXPECTED_LIVE_MARKET_FILL")
            continue
        if payload.get("fee_known") is not True:
            critical_incidents.append("UNKNOWN_LIVE_FILL_FEE")
        try:
            fill_quantity = float(payload["quantity"])
            fill_price = float(payload["price"])
            fee_eur = float(payload["fee_eur"])
        except (KeyError, TypeError, ValueError):
            critical_incidents.append("INVALID_FILL_RECORD")
            continue
        if fill_quantity <= 0.0 or fill_price <= 0.0 or fee_eur < 0.0:
            critical_incidents.append("INVALID_FILL_VALUES")
            continue
        side = str(payload.get("side") or "").upper()
        if side == "BUY":
            quantity += fill_quantity
            cost_basis += fill_quantity * fill_price + fee_eur
            continue
        if side != "SELL" or quantity <= 0.0 or fill_quantity > quantity + 1e-12:
            critical_incidents.append("UNMATCHED_SELL_FILL")
            continue
        quantity_before = quantity
        allocated_basis = cost_basis * min(fill_quantity / quantity_before, 1.0)
        proceeds = fill_quantity * fill_price - fee_eur
        quantity = max(0.0, quantity - fill_quantity)
        cost_basis = max(0.0, cost_basis - allocated_basis)
        episode_realized_pnl += proceeds - allocated_basis
        if quantity <= 1e-12:
            round_trip_pnl.append(episode_realized_pnl)
            quantity = 0.0
            cost_basis = 0.0
            episode_realized_pnl = 0.0
    flawless_round_trips = 0 if critical_incidents else len(round_trip_pnl)
    return {
        "ledger": str(ledger_path),
        "flawless_round_trips": flawless_round_trips,
        "net_live_expectancy": (
            sum(round_trip_pnl) / len(round_trip_pnl) if round_trip_pnl else None
        ),
        "critical_incidents": sorted(set(critical_incidents)),
        "open_quantity": quantity,
        "evidence_scope_strategy_id": strategy_id,
        "out_of_scope_fill_count": out_of_scope_fill_count,
    }


def capital_scaling_status(
    root: Path,
    *,
    strategy_id: str,
    flawless_round_trips: int,
    net_live_expectancy: float | None,
) -> dict[str, Any]:
    """Return evidence-based scaling status; missing authority always means Level 1."""

    authority_path = (
        root.resolve() / "output" / "governance" / "capital_level_authority.json"
    )
    authority: dict[str, Any] = {}
    if authority_path.is_file():
        candidate = dict(read_json(authority_path))
        if (
            candidate.get("strategy_id") == strategy_id
            and candidate.get("active") is True
        ):
            authority = candidate
    approved_level = int(authority.get("approved_level") or 1)
    status = capital_level(
        flawless_round_trips=flawless_round_trips,
        net_live_expectancy=net_live_expectancy,
        operator_approved_level=approved_level,
    )
    return {
        **status,
        "strategy_id": strategy_id,
        "flawless_round_trips": flawless_round_trips,
        "net_live_expectancy": net_live_expectancy,
        "authority_present": bool(authority),
        "approval_phrase_stored": False,
        "required_approval_phrase": (
            f"I APPROVE CAPITAL LEVEL <LEVEL> FOR {strategy_id}"
        ),
    }


def capital_scaling_status_from_ledger(
    root: Path,
    *,
    strategy_id: str = PRIMARY_STRATEGY_ID,
) -> dict[str, Any]:
    evidence = live_capital_evidence(root, strategy_id=strategy_id)
    return {
        **capital_scaling_status(
            root,
            strategy_id=strategy_id,
            flawless_round_trips=int(evidence["flawless_round_trips"]),
            net_live_expectancy=evidence["net_live_expectancy"],
        ),
        "evidence": evidence,
    }


def approve_capital_level(
    root: Path,
    *,
    strategy_id: str,
    requested_level: int,
    approval_phrase: str,
    flawless_round_trips: int,
    net_live_expectancy: float | None,
) -> dict[str, Any]:
    """Approve an evidence-eligible capital level without enabling autoscaling."""

    if requested_level not in {2, 3, 4}:
        raise ValueError("capital level must be 2, 3 or 4")
    if strategy_id != PRIMARY_STRATEGY_ID:
        raise PermissionError("capital scaling is not implemented for this strategy adapter")
    expected = f"I APPROVE CAPITAL LEVEL {requested_level} FOR {strategy_id}"
    if not secrets.compare_digest(approval_phrase.strip(), expected):
        raise PermissionError("capital level approval phrase does not match")
    eligible = capital_level(
        flawless_round_trips=flawless_round_trips,
        net_live_expectancy=net_live_expectancy,
        operator_approved_level=1,
    )["eligible_level"]
    if requested_level > eligible:
        raise PermissionError(
            f"capital level {requested_level} is not evidence-eligible; "
            f"eligible level is {eligible}"
        )
    payload = {
        "schema_version": "capital_level_authority_v1",
        "active": True,
        "approved_at": utc_iso(),
        "strategy_id": strategy_id,
        "approved_level": requested_level,
        "eligible_level_at_approval": eligible,
        "flawless_round_trips_at_approval": flawless_round_trips,
        "net_live_expectancy_at_approval": net_live_expectancy,
        "autoscale": False,
        "approval_phrase_stored": False,
    }
    governance_dir = root.resolve() / "output" / "governance"
    atomic_write_json(governance_dir / "capital_level_authority.json", payload)
    append_jsonl(
        governance_dir / "capital_level_authority_audit.jsonl",
        {
            "event": "CAPITAL_LEVEL_APPROVED",
            "recorded_at": payload["approved_at"],
            "strategy_id": strategy_id,
            "approved_level": requested_level,
            "evidence_hash": stable_hash(
                {
                    "flawless_round_trips": flawless_round_trips,
                    "net_live_expectancy": net_live_expectancy,
                    "eligible_level": eligible,
                }
            ),
            "approval_phrase_stored": False,
        },
    )
    return capital_scaling_status(
        root,
        strategy_id=strategy_id,
        flawless_round_trips=flawless_round_trips,
        net_live_expectancy=net_live_expectancy,
    )


__all__ = [
    "AllocationMode",
    "GovernancePaths",
    "HARD_BLOCKERS",
    "PRIMARY_MARKET",
    "PRIMARY_STRATEGY_DNA",
    "PRIMARY_STRATEGY_ID",
    "PRIMARY_TIMEFRAME",
    "PracticalLifecycle",
    "REQUESTED_TIMEFRAMES",
    "activate_live_canary_authority",
    "approve_capital_level",
    "build_portfolio_artifacts",
    "build_top50_universe",
    "capital_level",
    "capital_scaling_status",
    "capital_scaling_status_from_ledger",
    "deactivate_live_canary_authority",
    "governance_status",
    "live_canary_authority",
    "live_capital_evidence",
    "reclassify_existing_strategies",
]
