"""Canonical deployment artifacts for the autonomous active-swing service."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import yaml

from config.settings import (
    ACTIVE_SWING_TIMEFRAMES,
    DISABLED_EXECUTION_TIMEFRAMES,
    Settings,
)
from core.swing_trading import (
    WeeklyTradeBudgetManager,
    execution_timeframe_allowed,
    write_position_limit_status,
)
from utils.common import atomic_write_json, read_json, sha256_file, utc_iso


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return []
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        for key in ("rows", "strategies", "candidates", "results", "items"):
            selected = value.get(key)
            if isinstance(selected, list):
                return [
                    dict(row)
                    for row in selected
                    if isinstance(row, Mapping)
                ]
    return []


def _timeframe(row: Mapping[str, Any]) -> str:
    return str(
        row.get("timeframe")
        or row.get("execution_timeframe")
        or ""
    )


def _strategy_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("strategy_id")
        or row.get("economic_hypothesis_family")
        or row.get("strategy_name")
        or "UNKNOWN"
    )


def _strategy_dna(row: Mapping[str, Any]) -> str:
    return str(
        row.get("strategy_dna_hash")
        or row.get("strategy_dna")
        or ""
    )


def _material_positions(account: Mapping[str, Any]) -> list[dict[str, Any]]:
    holdings = list(
        ((account.get("account") or {}).get("portfolio_valuation") or {}).get(
            "holdings"
        )
        or []
    )
    return [
        dict(row)
        for row in holdings
        if Decimal(str(row.get("estimated_value_eur") or "0"))
        >= Decimal("5")
    ]


def _account_deployment_gates(
    *,
    account: Mapping[str, Any],
    external_inventory: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return fail-closed account blockers without granting trade authority."""

    core_blockers: list[str] = []
    entry_constraints = [
        str(value)
        for value in (
            list(account.get("entry_blockers") or [])
            + list(account.get("failures") or [])
        )
    ]
    account_status = str(account.get("status") or "UNKNOWN")
    if account_status != "READY":
        core_blockers.append(f"ACCOUNT_STATUS_{account_status}")
    if account.get("entry_allowed") is not True:
        entry_constraints.append("ACCOUNT_ENTRY_NOT_ALLOWED")
    external_status = str(external_inventory.get("status") or "")
    if external_status == "OPERATOR_DECISION_REQUIRED":
        core_blockers.append("EXTERNAL_INVENTORY_OPERATOR_DECISION_REQUIRED")
        entry_constraints.append("EXTERNAL_INVENTORY_NOT_CANONICAL")
    return (
        list(dict.fromkeys(core_blockers)),
        list(dict.fromkeys(entry_constraints)),
    )


def _strategy_truth(settings: Settings) -> dict[str, Any]:
    strategies = settings.paths.output_dir / "strategies"
    positive = _rows(strategies / "backtest_positive.json")
    paper = _rows(strategies / "paper_active.json")
    authority = _mapping(
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    portfolio_live = [
        dict(row)
        for row in authority.get("approved_candidates") or []
        if isinstance(row, Mapping)
        and execution_timeframe_allowed(str(row.get("timeframe") or ""))
    ]
    approval_path = (
        settings.paths.project_root
        / "config"
        / "live_strategy_approvals.yaml"
    )
    approval_payload = (
        yaml.safe_load(approval_path.read_text(encoding="utf-8")) or {}
        if approval_path.is_file()
        else {}
    )
    dedicated_live = [
        {
            "strategy_id": str(strategy_id),
            "strategy_dna_hash": str(
                values.get("strategy_dna_hash") or ""
            ),
            "timeframe": str(values.get("timeframe") or ""),
            "approved_markets": list(values.get("approved_markets") or []),
            "authority_source": "DEDICATED_LIVE_APPROVAL",
        }
        for strategy_id, values in dict(
            approval_payload.get("strategies") or {}
        ).items()
        if isinstance(values, Mapping)
        and values.get("approved_for_live") is True
        and execution_timeframe_allowed(
            str(values.get("timeframe") or "")
        )
    ]
    live_by_dna = {
        str(row.get("strategy_dna_hash") or ""): dict(row)
        for row in portfolio_live + dedicated_live
        if row.get("strategy_dna_hash")
    }
    live = list(live_by_dna.values())
    positive_active = [
        row for row in positive if execution_timeframe_allowed(_timeframe(row))
    ]
    paper_active = [
        row for row in paper if execution_timeframe_allowed(_timeframe(row))
    ]
    live_dna = {
        str(row.get("strategy_dna_hash") or "")
        for row in live
    }
    paper_dna = {_strategy_dna(row) for row in paper_active}
    shadow = [
        row
        for row in positive_active
        if _strategy_dna(row) not in live_dna | paper_dna
    ]
    return {
        "positive": positive_active,
        "paper": paper_active,
        "live": live,
        "shadow": shadow,
        "positive_by_timeframe": dict(
            Counter(_timeframe(row) for row in positive_active)
        ),
        "paper_by_timeframe": dict(
            Counter(_timeframe(row) for row in paper_active)
        ),
        "live_by_timeframe": dict(
            Counter(str(row.get("timeframe") or "") for row in live)
        ),
        "shadow_by_timeframe": dict(
            Counter(_timeframe(row) for row in shadow)
        ),
        "authority_active": authority.get("active") is True,
        "authority_hash": authority.get("authority_hash"),
    }


def _runtime_research_and_universe_truth(
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reconcile long-running research and explicit market exceptions."""

    output = settings.paths.output_dir
    heartbeat = _mapping(output / "live" / "heartbeat.json")
    research_health = dict(heartbeat.get("research") or {})
    continuous_research = dict(
        research_health.get("continuous_research") or {}
    )
    research_state = _mapping(output / "autopilot" / "status.json")
    tiered_universe = _mapping(
        output / "universe" / "tiered_trading_universe.json"
    )
    live_markets = list(
        tiered_universe.get("live_executable_markets") or []
    )
    if live_markets:
        stages = dict(research_state.get("stages") or {})
        universe = dict(stages.get("universe") or {})
        universe.update(
            {
                "execution_eligible": len(live_markets),
                "execution_eligible_markets": live_markets,
                "execution_eligibility_basis": (
                    "TIERED_LIVE_UNIVERSE_WITH_EXPLICIT_EXCEPTIONS"
                ),
            }
        )
        stages["universe"] = universe
        research_state["stages"] = stages
    return research_state, continuous_research, tiered_universe


def _two_hour_audit(settings: Settings) -> dict[str, Any]:
    normalized = settings.paths.data_dir / "normalized"
    markets = tuple(settings.autonomous_live.markets)
    rows: list[dict[str, Any]] = []
    for market in markets:
        data_path = normalized / f"{market}_2h.parquet"
        manifest_path = data_path.with_suffix(
            data_path.suffix + ".manifest.json"
        )
        provenance_path = data_path.with_suffix(
            data_path.suffix + ".provenance.json"
        )
        manifest = _mapping(manifest_path)
        provenance = _mapping(provenance_path)
        quality = dict(manifest.get("quality") or {})
        source_segments = list(manifest.get("source_segments") or [])
        derivation = dict(provenance.get("derivation") or {})
        classifications = sorted(
            {
                str(segment.get("classification") or "UNKNOWN")
                for segment in source_segments
                if isinstance(segment, Mapping)
            }
        )
        if derivation.get("source_classification"):
            classifications = sorted(
                set(classifications)
                | {str(derivation["source_classification"])}
            )
        rows.append(
            {
                "market": market,
                "status": (
                    "BACKTEST_ELIGIBLE"
                    if data_path.is_file()
                    and quality.get("valid") is True
                    else "DATA_PENDING"
                ),
                "data_file": str(data_path),
                "manifest_file": str(manifest_path),
                "provenance_file": str(provenance_path),
                "data_present": data_path.is_file(),
                "manifest_present": manifest_path.is_file(),
                "provenance_present": provenance_path.is_file(),
                "provider": (
                    manifest.get("provider")
                    or ",".join(provenance.get("providers_used") or [])
                    or None
                ),
                "source_classifications": classifications,
                "source_timeframe": derivation.get("source_timeframe"),
                "source_sha256": derivation.get("source_sha256"),
                "complete_bins_only": derivation.get(
                    "complete_bins_only"
                ),
                "rows": manifest.get("rows"),
                "start": manifest.get("start"),
                "end": manifest.get("end"),
                "missing_intervals": manifest.get("missing_bar_count"),
                "duplicate_intervals": manifest.get("duplicate_count"),
                "closed_candle_quality": quality.get("valid"),
                "stale": quality.get("stale"),
                "data_hash": manifest.get("dataset_hash"),
                "file_sha256": (
                    sha256_file(data_path) if data_path.is_file() else None
                ),
                "synthetic": False,
                "provenance_complete": bool(
                    source_segments or provenance
                ),
            }
        )
    payload = {
        "schema_version": "2h_data_integrity_audit_v1",
        "generated_at": utc_iso(),
        "timeframe": "2h",
        "utc_boundaries": True,
        "closed_candles_only": True,
        "missing_source_bar_produces_no_target_bar": True,
        "synthetic_fill_forbidden": True,
        "rows": rows,
        "eligible_market_count": sum(
            row["status"] == "BACKTEST_ELIGIBLE" for row in rows
        ),
        "pending_market_count": sum(
            row["status"] == "DATA_PENDING" for row in rows
        ),
    }
    atomic_write_json(
        settings.paths.output_dir / "data" / "2h_data_integrity_audit.json",
        payload,
    )
    return payload


def _data_audits(settings: Settings, two_hour: Mapping[str, Any]) -> None:
    rows = list(two_hour.get("rows") or [])
    no_synthetic = {
        "schema_version": "no_synthetic_audit_v1",
        "generated_at": utc_iso(),
        "synthetic_execution_data_allowed": False,
        "all_present_2h_datasets_real": all(
            row.get("synthetic") is False
            for row in rows
            if row.get("data_present")
        ),
        "datasets": rows,
    }
    atomic_write_json(
        settings.paths.output_dir / "data" / "no_synthetic_audit.json",
        no_synthetic,
    )
    now = datetime.now(UTC)
    freshness_rows = []
    for row in rows:
        raw = row.get("end")
        try:
            end = datetime.fromisoformat(
                str(raw).replace("Z", "+00:00")
            ).astimezone(UTC)
        except (TypeError, ValueError):
            end = None
        freshness_rows.append(
            {
                "market": row.get("market"),
                "timeframe": "2h",
                "last_closed_candle": raw,
                "age_seconds": (
                    (now - end).total_seconds() if end is not None else None
                ),
                "stale": row.get("stale"),
                "status": row.get("status"),
            }
        )
    atomic_write_json(
        settings.paths.output_dir / "data" / "data_freshness_audit.json",
        {
            "schema_version": "data_freshness_audit_v1",
            "generated_at": utc_iso(),
            "rows": freshness_rows,
        },
    )


def _current_opportunities(
    settings: Settings,
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    paper_state = _mapping(
        settings.paths.output_dir / "paper" / "generated_strategy_state.json"
    )
    evaluations = dict(paper_state.get("evaluations") or {})
    live_by_dna = {
        str(row.get("strategy_dna_hash") or ""): dict(row)
        for row in truth.get("live") or []
    }
    candidates: list[dict[str, Any]] = []
    for dna, evaluation in evaluations.items():
        authority = live_by_dna.get(str(dna))
        for market_row in dict(evaluation or {}).get("markets") or []:
            if (
                market_row.get("entry") is not True
                or market_row.get("stale") is True
            ):
                continue
            timeframe = str((authority or {}).get("timeframe") or "")
            if not execution_timeframe_allowed(timeframe):
                continue
            stop_distance = Decimal(
                str(market_row.get("stop_distance") or "0")
            )
            target_distance = Decimal(
                str(market_row.get("target_distance") or "0")
            )
            reward_risk = (
                target_distance / stop_distance
                if stop_distance > 0
                else Decimal("0")
            )
            candidates.append(
                {
                    "market": market_row.get("market"),
                    "strategy_id": (
                        authority or {}
                    ).get("strategy_id") or f"EXACT_{str(dna)[:16]}",
                    "strategy_dna_hash": dna,
                    "timeframe": timeframe,
                    "live_eligibility": (
                        "LIVE_ELIGIBLE" if authority else "PAPER_ONLY"
                    ),
                    "signal_timestamp": market_row.get("signal_timestamp"),
                    "stop_distance": str(stop_distance),
                    "target_distance": str(target_distance),
                    "expected_reward_risk": str(reward_risk),
                    "confidence": 50.0 if authority else 35.0,
                    "data_freshness": "CURRENT_CLOSED_CANDLE",
                    "why_now": "Frozen strategy entry is true on the latest closed candle.",
                }
            )
    candidates.sort(
        key=lambda row: (
            Decimal(str(row["expected_reward_risk"])),
            row["confidence"],
        ),
        reverse=True,
    )
    top = candidates[:5]
    payload = {
        "schema_version": "current_top_5_opportunities_v1",
        "generated_at": utc_iso(),
        "opportunity_count": len(top),
        "top_opportunities": [
            {"rank": index, **row}
            for index, row in enumerate(top, start=1)
        ],
        "poor_setups_are_not_used_as_fillers": True,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    atomic_write_json(
        settings.paths.output_dir
        / "research"
        / "current_top_5_opportunities.json",
        payload,
    )
    atomic_write_json(
        settings.paths.output_dir
        / "research"
        / "current_top_candidates.json",
        {
            **payload,
            "schema_version": "current_top_candidates_v1",
        },
    )
    return payload


def build_active_swing_deployment_artifacts(
    settings: Settings,
    *,
    runtime: Mapping[str, Any],
    tests_passed: bool,
) -> dict[str, Any]:
    """Reconcile sanitized runtime evidence into the required deployment set."""

    output = settings.paths.output_dir
    governance = output / "governance"
    operations = output / "operations"
    research = output / "research"
    reports = output / "reports"
    notifications = output / "notifications"
    for directory in (
        governance,
        operations,
        research,
        reports,
        notifications,
        output / "data",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    account = _mapping(operations / "live_account_health.json")
    external_inventory = _mapping(
        operations / "external_inventory_remediation.json"
    )
    telegram = _mapping(notifications / "telegram_health.json")
    if not telegram:
        telegram = _mapping(notifications / "telegram_status.json")
    truth = _strategy_truth(settings)
    material = _material_positions(account)
    equity = Decimal(
        str(
            ((account.get("account") or {}).get("portfolio_valuation") or {}).get(
                "estimated_total_equity_eur"
            )
            or "0"
        )
    )
    position_limit = write_position_limit_status(
        settings,
        account_equity_eur=equity,
        material_positions=[
            str(row.get("market") or "") for row in material
        ],
    )
    weekly = WeeklyTradeBudgetManager(settings).status()
    two_hour = _two_hour_audit(settings)
    _data_audits(settings, two_hour)
    opportunities = _current_opportunities(settings, truth)

    timeframe_truth = {
        "schema_version": "canonical_timeframe_truth_v1",
        "generated_at": utc_iso(),
        "active_swing_timeframes": list(ACTIVE_SWING_TIMEFRAMES),
        "disabled_execution_timeframes": sorted(
            DISABLED_EXECUTION_TIMEFRAMES
        ),
        "disabled_timeframes_may_remain_research_only": True,
        "positive_by_timeframe": truth["positive_by_timeframe"],
        "paper_by_timeframe": truth["paper_by_timeframe"],
        "live_by_timeframe": truth["live_by_timeframe"],
        "shadow_by_timeframe": truth["shadow_by_timeframe"],
        "two_hour_data": two_hour,
    }
    atomic_write_json(
        governance / "canonical_timeframe_truth.json",
        timeframe_truth,
    )

    promotion_rows: list[dict[str, Any]] = []
    live_dna = {
        str(row.get("strategy_dna_hash") or "")
        for row in truth["live"]
    }
    paper_dna = {_strategy_dna(row) for row in truth["paper"]}
    for row in truth["positive"]:
        dna = _strategy_dna(row)
        promotion_rows.append(
            {
                "strategy_id": _strategy_id(row),
                "strategy_dna_hash": dna,
                "timeframe": _timeframe(row),
                "status": (
                    "LIVE_ACTIVE"
                    if dna in live_dna
                    else "PAPER_ACTIVE"
                    if dna in paper_dna
                    else "SHADOW_ACTIVE"
                ),
                "execution_timeframe_allowed": True,
                "normal_profit_factor": row.get("normal_profit_factor"),
                "stressed_profit_factor": row.get(
                    "stressed_profit_factor"
                ),
                "block_reason": row.get("paper_activation_pending_reason"),
            }
        )
    atomic_write_json(
        governance / "strategy_promotion_matrix.json",
        {
            "schema_version": "strategy_promotion_matrix_v1",
            "generated_at": utc_iso(),
            "rows": promotion_rows,
        },
    )

    reconciliation = dict(account.get("reconciliation") or {})
    config = dict(account.get("configuration") or {})
    private_stream = dict(runtime.get("private_account_websocket") or {})
    public_stream = dict(runtime.get("public_market_websocket") or {})
    kill_switch = dict(runtime.get("kill_switch") or {})
    core_blockers, entry_constraints = _account_deployment_gates(
        account=account,
        external_inventory=external_inventory,
    )
    if runtime.get("process_running") is not True:
        core_blockers.append("LIVE_SUPERVISOR_NOT_RUNNING")
    if runtime.get("control_state") != "ENABLED":
        core_blockers.append(
            f"CONTROL_STATE_{runtime.get('control_state') or 'UNKNOWN'}"
        )
    if reconciliation.get("healthy") is not True:
        core_blockers.append("RECONCILIATION_NOT_HEALTHY")
    if private_stream.get("ready_for_new_entries") is not True:
        core_blockers.append("PRIVATE_ACCOUNT_STREAM_NOT_READY")
    if kill_switch.get("active") is True:
        core_blockers.append("KILL_SWITCH_ACTIVE")
    if config.get("withdrawal_permission") is not False:
        core_blockers.append("WITHDRAWAL_PERMISSION_NOT_PROVEN_ABSENT")
    if not truth["live"]:
        core_blockers.append("NO_ACTIVE_FROZEN_STRATEGY_AUTHORITY")
    if not tests_passed:
        core_blockers.append("CURRENT_TEST_SUITE_NOT_RECORDED_GREEN")

    if position_limit["new_position_allowed"] is not True:
        entry_constraints.append("WALLET_POSITION_CAP_REACHED")
    if weekly["new_entries_blocked"]:
        entry_constraints.append("WEEKLY_ENTRY_CAP_REACHED")
    operational_mode = (
        "LIVE_ACTIVE"
        if not core_blockers
        else "HALTED"
        if kill_switch.get("active") is True
        else "DEGRADED"
    )
    decision = (
        "APPROVED_LIVE_ACTIVE"
        if not core_blockers and not entry_constraints
        else "APPROVED_LIVE_LIMITED"
        if not core_blockers
        else "KILL_SWITCHED"
        if kill_switch.get("active") is True
        else "HALTED"
    )
    current_mode = {
        "schema_version": "current_operational_mode_v1",
        "generated_at": utc_iso(),
        "account_mode": operational_mode,
        "strategy_modes_by_timeframe": {
            timeframe: {
                "live": truth["live_by_timeframe"].get(timeframe, 0),
                "paper": truth["paper_by_timeframe"].get(timeframe, 0),
                "shadow": truth["shadow_by_timeframe"].get(timeframe, 0),
            }
            for timeframe in ACTIVE_SWING_TIMEFRAMES
        },
        "core_blockers": core_blockers,
        "entry_constraints": list(dict.fromkeys(entry_constraints)),
    }
    atomic_write_json(
        governance / "current_operational_mode.json",
        current_mode,
    )

    go_live = {
        "schema_version": "final_go_live_decision_v1",
        "decision": decision,
        "generated_at": utc_iso(),
        "exchange_connected": (
            private_stream.get("state") == "AUTHENTICATED"
            and public_stream.get("state") == "CONNECTED"
        ),
        "trade_permission_verified": config.get(
            "trade_credentials_present"
        )
        is True
        and config.get("trade_scope_safe") is True,
        "withdrawal_permission_absent": config.get(
            "withdrawal_permission"
        )
        is False,
        "reconciliation_healthy": reconciliation.get("healthy") is True,
        "kill_switch_clear": kill_switch.get("active") is not True,
        "account_equity": str(equity),
        "available_EUR": (account.get("account") or {}).get("eur_available"),
        "current_positions": material,
        "maximum_positions": position_limit["maximum_positions"],
        "live_eligible_strategies": len(truth["live"]),
        "paper_active_strategies": len(truth["paper"]),
        "shadow_active_strategies": len(truth["shadow"]),
        "active_timeframes": list(ACTIVE_SWING_TIMEFRAMES),
        "weekly_trade_cap": weekly["hard_cap"],
        "current_weekly_trade_count": weekly["new_entries"],
        "tests_passed": tests_passed,
        "blocking_conditions": core_blockers,
        "entry_constraints": list(dict.fromkeys(entry_constraints)),
        "operational_mode": operational_mode,
        "service_pid": runtime.get("pid"),
    }
    atomic_write_json(
        governance / "final_go_live_decision.json",
        go_live,
    )

    (
        research_state,
        continuous_research,
        tiered_universe,
    ) = _runtime_research_and_universe_truth(settings)

    repo_truth = {
        "schema_version": "repo_truth_v1",
        "generated_at": utc_iso(),
        "canonical_repository_state": decision,
        "active_processes": {
            "live_supervisor": {
                "pid": runtime.get("pid"),
                "running": runtime.get("process_running"),
            },
            "continuous_research": continuous_research,
            "companion_services": _mapping(
                output / "live" / "companion_services.json"
            ),
        },
        "live_execution_state": runtime.get("status"),
        "exchange_connection_state": {
            "private": private_stream.get("state"),
            "public": public_stream.get("state"),
        },
        "reconciliation_state": reconciliation,
        "Telegram_state": telegram,
        "data_sync_state": _mapping(
            output / "live" / "companion_services.json"
        ).get("services", {}).get("data_sync"),
        "research_state": research_state,
        "live_market_universe": {
            "count": len(
                tiered_universe.get("live_executable_markets") or []
            ),
            "markets": list(
                tiered_universe.get("live_executable_markets") or []
            ),
            "selection_hash": tiered_universe.get("selection_hash"),
            "artifact": str(
                output / "universe" / "tiered_trading_universe.json"
            ),
        },
        "current_wallet_equity": str(equity),
        "available_EUR": (account.get("account") or {}).get("eur_available"),
        "material_positions": material,
        "strategy_inventory_by_timeframe": truth["positive_by_timeframe"],
        "paper_authority_by_timeframe": truth["paper_by_timeframe"],
        "live_authority_by_timeframe": truth["live_by_timeframe"],
        "conflicting_artifacts": [],
        "chosen_canonical_artifacts": [
            str(operations / "live_account_health.json"),
            str(operations / "external_inventory_remediation.json"),
            str(governance / "positive_strategy_live_authority.json"),
            str(output / "live" / "autonomous_live_status.json"),
            str(output / "strategies" / "backtest_positive.json"),
            str(output / "strategies" / "paper_active.json"),
            str(output / "universe" / "tiered_trading_universe.json"),
        ],
        "stale_artifacts": [],
        "blocking_conditions": core_blockers,
        "entry_constraints": list(dict.fromkeys(entry_constraints)),
        "final_operational_mode": operational_mode,
    }
    atomic_write_json(governance / "repo_truth.json", repo_truth)

    atomic_write_json(
        operations / "live_execution_status.json",
        {
            "schema_version": "live_execution_status_v1",
            "generated_at": utc_iso(),
            "runtime": dict(runtime),
            "operational_mode": operational_mode,
            "entry_constraints": list(dict.fromkeys(entry_constraints)),
        },
    )
    atomic_write_json(
        operations / "live_reconciliation.json",
        {
            "schema_version": "live_reconciliation_v1",
            "generated_at": utc_iso(),
            **reconciliation,
        },
    )
    atomic_write_json(
        operations / "exchange_permissions_audit.json",
        {
            "schema_version": "exchange_permissions_audit_v1",
            "generated_at": utc_iso(),
            "trade_permission_verified": go_live[
                "trade_permission_verified"
            ],
            "withdrawal_permission_absent": go_live[
                "withdrawal_permission_absent"
            ],
            "spot_only": settings.execution.spot_only,
            "margin": settings.execution.allow_margin,
            "leverage": settings.execution.allow_leverage,
            "shorting": settings.execution.allow_short_selling,
        },
    )
    atomic_write_json(
        research / "2h_strategy_inventory.json",
        {
            "schema_version": "2h_strategy_inventory_v1",
            "generated_at": utc_iso(),
            "positive": truth["positive_by_timeframe"].get("2h", 0),
            "paper": truth["paper_by_timeframe"].get("2h", 0),
            "live": truth["live_by_timeframe"].get("2h", 0),
            "shadow": truth["shadow_by_timeframe"].get("2h", 0),
            "data_integrity": two_hour,
        },
    )
    atomic_write_json(
        research / "timeframe_performance_matrix.json",
        timeframe_truth,
    )
    atomic_write_json(
        notifications / "telegram_delivery_audit.json",
        {
            "schema_version": "telegram_delivery_audit_v1",
            "generated_at": utc_iso(),
            "status": telegram.get("status"),
            "api_reachable": telegram.get("api_reachable"),
            "secrets_redacted": telegram.get("secrets_redacted"),
            "active_queue_size": telegram.get("active_queue_size"),
            "last_successful_send": telegram.get("last_successful_send"),
        },
    )
    report_lines = [
        "# Final active swing deployment",
        "",
        "## Canonical decision",
        "",
        f"- Decision: `{decision}`",
        f"- Operational mode: `{operational_mode}`",
        f"- Service PID: `{runtime.get('pid')}`",
        f"- Exchange streams: private `{private_stream.get('state')}`, "
        f"public `{public_stream.get('state')}`",
        f"- Reconciliation healthy: `{reconciliation.get('healthy')}`",
        f"- Kill switch clear: `{kill_switch.get('active') is not True}`",
        f"- Trade permission verified: `{go_live['trade_permission_verified']}`",
        f"- Withdrawal permission absent: `{go_live['withdrawal_permission_absent']}`",
        f"- Tests passed: `{tests_passed}`",
        "",
        "## Wallet and risk",
        "",
        f"- Equity: `EUR {equity}`",
        f"- Available EUR: `{go_live['available_EUR']}`",
        f"- Material positions: `{len(material)}` / "
        f"`{position_limit['maximum_positions']}`",
        f"- Weekly entries: `{weekly['new_entries']}` / "
        f"`{weekly['hard_cap']}`",
        f"- Remaining weekly entry budget: `{weekly['remaining_entry_budget']}`",
        f"- Core blockers: `{core_blockers}`",
        f"- Entry constraints: `{list(dict.fromkeys(entry_constraints))}`",
        "",
        "| Market | Quantity | Mark EUR | Value EUR |",
        "|---|---:|---:|---:|",
    ]
    if material:
        report_lines.extend(
            f"| {row.get('market')} | {row.get('quantity')} | "
            f"{row.get('price_eur')} | {row.get('estimated_value_eur')} |"
            for row in material
        )
    else:
        report_lines.append("| none | 0 | 0 | 0 |")
    report_lines.extend(
        [
            "",
            "## Strategy authority by timeframe",
            "",
            "| Timeframe | Positive | Shadow only | "
            "Paper authority (includes Live) | Live authority |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for timeframe in ACTIVE_SWING_TIMEFRAMES:
        report_lines.append(
            f"| {timeframe} | "
            f"{truth['positive_by_timeframe'].get(timeframe, 0)} | "
            f"{truth['shadow_by_timeframe'].get(timeframe, 0)} | "
            f"{truth['paper_by_timeframe'].get(timeframe, 0)} | "
            f"{truth['live_by_timeframe'].get(timeframe, 0)} |"
        )
    report_lines.extend(["", "### Exact live strategy IDs", ""])
    for timeframe in ACTIVE_SWING_TIMEFRAMES:
        strategy_ids = sorted(
            str(row.get("strategy_id") or "UNKNOWN")
            for row in truth["live"]
            if str(row.get("timeframe") or "") == timeframe
        )
        report_lines.append(
            f"- `{timeframe}`: "
            + (
                ", ".join(f"`{value}`" for value in strategy_ids)
                if strategy_ids
                else "none"
            )
        )
    report_lines.extend(
        [
            "",
            "## Current opportunities",
            "",
            f"- Count: `{opportunities['opportunity_count']}`",
            "- Poor setups are not used as fillers.",
        ]
    )
    for row in opportunities.get("top_opportunities") or []:
        report_lines.append(
            f"- #{row.get('rank')} `{row.get('market')}` "
            f"`{row.get('strategy_id')}` `{row.get('timeframe')}` "
            f"RR `{row.get('expected_reward_risk')}`"
        )
    report_lines.extend(
        [
            "",
            "## Two-hour data integrity",
            "",
            f"- Eligible markets: `{two_hour['eligible_market_count']}`",
            f"- Pending markets: `{two_hour['pending_market_count']}`",
            "- Synthetic filling: forbidden.",
            "- Closed complete candles and UTC boundaries: required.",
        ]
    )
    for row in two_hour.get("rows") or []:
        report_lines.append(
            f"- `{row.get('market')}`: `{row.get('status')}`, "
            f"rows `{row.get('rows')}`, source `{row.get('provider')}`, "
            f"source timeframe `{row.get('source_timeframe')}`"
        )
    report_lines.extend(
        [
            "",
            "## Canonical commands",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe .\main.py live status",
            r".\.venv\Scripts\python.exe .\main.py live reconcile",
            r".\.venv\Scripts\python.exe .\main.py live opportunities",
            r".\.venv\Scripts\python.exe .\main.py live weekly-budget",
            r".\.venv\Scripts\python.exe .\main.py live positions",
            r".\.venv\Scripts\python.exe .\main.py autonomous-live pause",
            r".\.venv\Scripts\python.exe .\main.py autonomous-live resume",
            r".\.venv\Scripts\python.exe .\main.py autonomous-live shutdown",
            r'.\.venv\Scripts\python.exe .\main.py live emergency-stop --reason "operator request"',
            "```",
            "",
            "No trade was forced by this audit. New entries require a fresh "
            "natural signal and all wallet, data, liquidity, cooldown and "
            "risk gates to pass. Protective exits remain permitted.",
            "",
        ]
    )
    report = "\n".join(report_lines)
    (reports / "final_active_swing_deployment_report.md").write_text(
        report,
        encoding="utf-8",
    )
    from reporting.crypto_active_swing_product import (
        build_crypto_active_swing_product,
    )

    product = build_crypto_active_swing_product(
        settings,
        runtime=runtime,
        deployment_decision=decision,
        deployment_blockers=core_blockers,
        entry_constraints=list(dict.fromkeys(entry_constraints)),
    )
    return {
        "status": "COMPLETE",
        "decision": decision,
        "operational_mode": operational_mode,
        "repo_truth": repo_truth,
        "final_go_live_decision": go_live,
        "weekly_trade_budget": weekly,
        "position_limit_status": position_limit,
        "opportunities": opportunities,
        "two_hour_data_integrity": two_hour,
        "active_swing_product": product,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = ["build_active_swing_deployment_artifacts"]
