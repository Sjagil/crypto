"""Bounded live portfolio canaries for frozen exact-positive strategy DNA.

The generated-strategy paper engine remains the signal source.  This module
only adds portfolio authority, restart-safe position ownership, wallet-wide
position limits and canonical Bitvavo execution.  It never invents a signal
and never changes strategy parameters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.settings import Settings
from core.contracts import (
    ExecutionBlocked,
    OrderIntent,
    OrderSide,
    OrderTimeInForce,
    OrderType,
    ReconciliationRequired,
    ResearchStatus,
)
from core.economics import CanonicalCostModel
from core.event_driven_paper import load_canonical_entry_economics_gate
from core.generated_strategy_paper import load_generated_candidates
from core.live_capital import (
    APPROVAL_PHRASE as CAPITAL_LEVEL_2_APPROVAL_PHRASE,
)
from core.live_capital import (
    CAPITAL_LEVEL,
    MAXIMUM_MANAGED_POSITIONS,
    MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR,
    capital_level_2_capacity,
    managed_live_portfolio,
    submit_level_2_buy_atomically,
)
from core.live_capital import (
    MAXIMUM_NEW_ORDERS_PER_DAY as LEVEL_2_MAXIMUM_NEW_ORDERS_PER_DAY,
)
from core.live_capital import (
    MAXIMUM_ORDER_EUR as LEVEL_2_MAXIMUM_ORDER_EUR,
)
from core.regime_policy import regime_policy
from core.strategy_degradation import degradation_by_dna
from core.swing_trading import (
    SwingCooldownManager,
    WeeklyTradeBudgetManager,
    execution_timeframe_allowed,
    write_position_limit_status,
)
from data.data_loader import TIMEFRAME_SECONDS
from execution.execution import (
    EntryOrderPlan,
    LivePreflight,
    build_live_client,
    plan_bounded_entry_order,
    quantity_is_protectable_at_stop,
)
from portfolio.buy_chain import (
    canonicalize_approved_buy_order,
    planned_target_net_edge,
)
from risk.canary_guard import CanaryPolicy, InstitutionalCanaryGuard
from risk.risk_manager import KillSwitch, PortfolioSnapshot, RiskManager
from utils.common import (
    atomic_write_json,
    read_json,
    stable_hash,
    utc_iso,
    utc_now,
)

SCHEMA_VERSION = "positive_strategy_live_authority_v1"
STATE_SCHEMA_VERSION = "generated_strategy_live_state_v1"
BASELINE_SCHEMA_VERSION = "generated_strategy_inventory_baseline_v1"
APPROVAL_PHRASE = "LIVE POSITIVE STRATEGY PORTFOLIO CONFIRMED"
DNA_APPROVAL_PREFIX = "LIVE POSITIVE DNA"

MAXIMUM_ORDER_EUR = LEVEL_2_MAXIMUM_ORDER_EUR
MAXIMUM_TOTAL_EXPOSURE_EUR = MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR
MAXIMUM_OPEN_POSITIONS = MAXIMUM_MANAGED_POSITIONS
MAXIMUM_NEW_ORDERS_PER_DAY = LEVEL_2_MAXIMUM_NEW_ORDERS_PER_DAY
MINIMUM_MATERIAL_POSITION_EUR = Decimal("5")
ENTRY_WINDOW_MINUTES = 15
MACRO_OVERLAY_MAXIMUM_AGE = timedelta(hours=3)
ZERO = Decimal("0")
LIVE_TARGET_R_MULTIPLE = Decimal("1.5")


def _paths(settings: Settings) -> dict[str, Path]:
    governance = settings.paths.output_dir / "governance"
    live = settings.paths.output_dir / "live"
    governance.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    return {
        "authority": governance / "positive_strategy_live_authority.json",
        "baseline": governance / "positive_strategy_inventory_baseline.json",
        "state": live / "generated_strategy_live_state.json",
        "status": live / "generated_strategy_live_status.json",
        "ledger": settings.paths.checkpoints_dir / "live_execution.jsonl",
    }


def _load_live_macro_overlay(
    settings: Settings,
    *,
    observed_at: datetime,
) -> dict[str, Any]:
    """Load a causal, freshness-checked macro selector for live entries."""

    path = settings.paths.output_dir / "active_trading" / "macro_crypto.json"
    blocked = {
        "status": "DATA_BLOCKED",
        "regime": "DATA_BLOCKED",
        "confidence": 0.0,
        "available_at": None,
        "reason_code": "MACRO_CONTEXT_MISSING",
        "macro_is_entry_signal": False,
        "source_path": str(path),
    }
    if not path.is_file():
        return blocked
    try:
        payload = dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {**blocked, "reason_code": "MACRO_CONTEXT_INVALID"}
    try:
        available_at = datetime.fromisoformat(
            str(payload.get("available_at") or "").replace("Z", "+00:00")
        ).astimezone(UTC)
    except ValueError:
        return {**blocked, "reason_code": "MACRO_TIMESTAMP_INVALID"}
    now = observed_at.astimezone(UTC)
    if available_at > now + timedelta(minutes=5):
        return {
            **blocked,
            "available_at": available_at.isoformat(),
            "reason_code": "MACRO_CONTEXT_FROM_FUTURE",
        }
    age = now - available_at
    if age > MACRO_OVERLAY_MAXIMUM_AGE:
        return {
            **blocked,
            "available_at": available_at.isoformat(),
            "age_seconds": age.total_seconds(),
            "reason_code": "MACRO_CONTEXT_STALE",
        }
    confidence = max(
        0.0,
        min(1.0, float(payload.get("confidence") or 0.0)),
    )
    regime = str(payload.get("regime") or "DATA_BLOCKED").upper()
    if confidence < 0.50 or regime == "DATA_BLOCKED":
        return {
            **blocked,
            "available_at": available_at.isoformat(),
            "age_seconds": age.total_seconds(),
            "confidence": confidence,
            "reason_code": "MACRO_CONTEXT_INSUFFICIENT_CONFIDENCE",
        }
    sources = {
        str(name): {
            "provider": row.get("provider"),
            "available_at": row.get("available_at"),
            "freshness": row.get("freshness"),
            "fresh": row.get("fresh") is True,
            "confidence": row.get("confidence"),
        }
        for name, value in dict(payload.get("sources") or {}).items()
        if isinstance(value, Mapping)
        for row in [dict(value)]
    }
    return {
        "status": "FRESH",
        "regime": regime,
        "confidence": confidence,
        "observed_at": payload.get("observed_at"),
        "available_at": available_at.isoformat(),
        "age_seconds": age.total_seconds(),
        "reason_code": "MACRO_CONTEXT_FRESH",
        "features": dict(payload.get("features") or {}),
        "sources": sources,
        "macro_is_entry_signal": False,
        "source_path": str(path),
    }


def _decimal(
    value: Any,
    *,
    default: Decimal = ZERO,
) -> Decimal:
    raw = default if value in (None, "") else value
    try:
        parsed = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return default
    return parsed if parsed.is_finite() else default


def _resolve_live_risk_distances(
    selected: Mapping[str, Any],
) -> tuple[Decimal, Decimal, str]:
    """Return bounded protective distances for an already-valid signal.

    Some canonical strategies use a trailing or indicator exit and therefore
    intentionally emit no fixed target.  Live execution still needs an
    auditable protective take-profit path.  A missing target may fall back to
    1.5R, but a missing/invalid stop never receives a cosmetic fallback.
    """

    stop_distance = _decimal(selected.get("stop_distance"))
    target_distance = _decimal(selected.get("target_distance"))
    if stop_distance <= ZERO:
        return ZERO, ZERO, "INVALID_STOP_DISTANCE"
    if target_distance > ZERO:
        return stop_distance, target_distance, "STRATEGY_TARGET_DISTANCE"
    return (
        stop_distance,
        stop_distance * LIVE_TARGET_R_MULTIPLE,
        "PROTECTIVE_1_5R_TARGET_FALLBACK",
    )


def _candidate_identity(candidate: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    service_authority_path = (
        settings.paths.output_dir
        / "live"
        / "autonomous_live_authority.json"
    )
    service_authority = (
        dict(read_json(service_authority_path))
        if service_authority_path.is_file()
        else {}
    )
    service_markets = {
        str(market).upper()
        for market in (
            service_authority.get("markets")
            if service_authority.get("active") is True
            else settings.operational.markets
        )
        or []
    }
    markets = sorted(
        {
            str(market).upper()
            for market in candidate.get("markets") or []
            if str(market).upper() in settings.operational.markets
            and str(market).upper() in service_markets
            and settings.shariah.eligibility(str(market)).status.value == "ALLOWED"
        }
    )
    identity = {
        "strategy_id": str(
            candidate.get("strategy_id")
            or candidate.get("economic_hypothesis_family")
            or f"EXACT_POSITIVE_{str(candidate.get('strategy_dna_hash') or '')[:16]}"
        ),
        "strategy_dna_hash": str(candidate.get("strategy_dna_hash") or ""),
        "frozen_candidate_hash": str(candidate.get("frozen_candidate_hash") or ""),
        "timeframe": str(candidate.get("timeframe") or ""),
        "approved_markets": markets,
        "source": str(candidate.get("source") or candidate.get("source_report") or ""),
    }
    if candidate.get("frozen_identity_schema"):
        identity["frozen_identity_schema"] = candidate[
            "frozen_identity_schema"
        ]
    return identity


def _candidate_snapshot(
    settings: Settings,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        _candidate_identity(candidate, settings)
        for candidate in candidates
    ]
    return sorted(
        (
            row
            for row in rows
            if len(row["strategy_dna_hash"]) == 64
            and row["frozen_candidate_hash"]
            and row["timeframe"] in TIMEFRAME_SECONDS
            and execution_timeframe_allowed(row["timeframe"])
            and row["approved_markets"]
        ),
        key=lambda row: row["strategy_dna_hash"],
    )


def activate_positive_strategy_live_authority(
    settings: Settings,
    *,
    approval_phrase: str | None = None,
    approval_reference: str | None = None,
    explicit_goal_authority: bool = False,
) -> dict[str, Any]:
    """Freeze the current exact-positive set under one small portfolio sleeve."""

    if not explicit_goal_authority and approval_phrase != APPROVAL_PHRASE:
        raise PermissionError("positive strategy portfolio approval phrase mismatch")
    if explicit_goal_authority and not approval_reference:
        raise PermissionError("explicit goal authority requires an approval reference")
    candidates = load_generated_candidates(settings)
    approved = _candidate_snapshot(settings, candidates)
    if not approved:
        raise ValueError("no exact-positive executable strategy DNA available")
    existing_state = _state(settings)
    if existing_state.get("positions"):
        raise PermissionError(
            "cannot replace positive strategy authority while positions exist"
        )
    existing_baseline_path = _paths(settings)["baseline"]
    existing_baseline = (
        dict(read_json(existing_baseline_path))
        if existing_baseline_path.is_file()
        else {}
    )
    if existing_baseline and (
        existing_baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
        or existing_baseline.get("baseline_hash")
        != _baseline_hash(existing_baseline)
    ):
        raise PermissionError(
            "existing positive strategy inventory baseline is invalid"
        )
    activated_at = utc_iso()
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "active": True,
        "activated_at": activated_at,
        "operator_approval_reference": (
            approval_reference or "local_exact_phrase_positive_portfolio"
        ),
        "approval_phrase_stored": False,
        "approval_scope": "EXACT_POSITIVE_FROZEN_DNA",
        # A continuously changing research shortlist must never grant live
        # authority to a previously unknown DNA.  New DNA needs a fresh,
        # explicit operator approval; frozen DNA that temporarily disappears
        # from the shortlist remains frozen but is simply not evaluated.
        "auto_enroll_future_exact_positive_dna": False,
        "approved_candidates": approved,
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
        "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
        "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        "maximum_one_position_per_market": True,
        "maximum_one_position_per_strategy_dna": True,
        "spot_only": True,
        "long_only": True,
        "autoscale": False,
        "withdrawals_available": False,
        "natural_signals_only": True,
        "future_unknown_dna_fail_closed": True,
        "inventory_baseline_status": (
            "ACTIVE"
            if existing_baseline
            else "PENDING_PRIVATE_BALANCE_SNAPSHOT"
        ),
    }
    body["candidate_snapshot_hash"] = stable_hash(approved, length=64)
    body["authority_hash"] = stable_hash(
        {key: value for key, value in body.items() if key != "authority_hash"},
        length=64,
    )
    path = _paths(settings)["authority"]
    atomic_write_json(path, body)
    if existing_baseline:
        existing_baseline["authority_hash"] = body["authority_hash"]
        existing_baseline["baseline_hash"] = _baseline_hash(
            existing_baseline
        )
        atomic_write_json(existing_baseline_path, existing_baseline)
    return {
        **body,
        "artifact": str(path),
        "approved_candidate_count": len(approved),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def migrate_positive_strategy_live_order_cap(
    settings: Settings,
    *,
    approval_phrase: str,
    maximum_order_eur: Decimal,
    preserve_strategy_dna: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Change only the cap of the already frozen live DNA snapshot.

    Unlike activation, this migration deliberately does not rescan the current
    research inventory.  It therefore cannot grant live authority to newly
    generated strategy DNA as a side effect of a capital-limit change.
    """

    if approval_phrase != APPROVAL_PHRASE:
        raise PermissionError("positive strategy portfolio approval phrase mismatch")
    requested_cap = _decimal(maximum_order_eur)
    if requested_cap != MAXIMUM_ORDER_EUR:
        raise ValueError(
            f"maximum order must equal the configured canary cap {MAXIMUM_ORDER_EUR}"
        )
    state = _state(settings)
    if state.get("positions"):
        raise PermissionError(
            "cannot migrate positive strategy authority while positions exist"
        )
    path = _paths(settings)["authority"]
    if not path.is_file():
        raise FileNotFoundError("positive strategy authority is missing")
    authority = dict(read_json(path))
    if (
        authority.get("schema_version") != SCHEMA_VERSION
        or not _validate_authority_hash(authority)
        or authority.get("active") is not True
    ):
        raise PermissionError("existing positive strategy authority is invalid")
    approved_before = [
        dict(row) for row in authority.get("approved_candidates") or []
    ]
    if not approved_before:
        raise PermissionError("existing positive strategy DNA snapshot is empty")
    before_hash = str(authority.get("candidate_snapshot_hash") or "")
    if before_hash != stable_hash(approved_before, length=64):
        raise PermissionError("existing positive strategy DNA snapshot hash mismatch")
    approved = approved_before
    if preserve_strategy_dna is not None:
        requested = {str(value).strip() for value in preserve_strategy_dna}
        if not requested or "" in requested:
            raise ValueError("preserved strategy DNA set must not be empty")
        existing_by_dna = {
            str(row.get("strategy_dna_hash") or ""): row
            for row in approved_before
        }
        unknown = sorted(requested - set(existing_by_dna))
        if unknown:
            raise PermissionError(
                "cannot add unknown strategy DNA during cap migration: "
                + ",".join(unknown)
            )
        approved = sorted(
            (dict(existing_by_dna[dna]) for dna in requested),
            key=lambda row: row["strategy_dna_hash"],
        )
        authority["approved_candidates"] = approved
        authority["candidate_snapshot_hash"] = stable_hash(
            approved,
            length=64,
        )
    previous_cap = str(authority.get("maximum_order_eur") or "")
    authority["maximum_order_eur"] = str(requested_cap)
    authority["last_order_cap_migration_at"] = utc_iso()
    authority["last_order_cap_previous_eur"] = previous_cap
    authority["approval_phrase_stored"] = False
    authority["authority_hash"] = stable_hash(
        {
            key: value
            for key, value in authority.items()
            if key != "authority_hash"
        },
        length=64,
    )
    atomic_write_json(path, authority)
    baseline_path = _paths(settings)["baseline"]
    if baseline_path.is_file():
        baseline = dict(read_json(baseline_path))
        if (
            baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
            or baseline.get("baseline_hash") != _baseline_hash(baseline)
        ):
            raise PermissionError("existing inventory baseline is invalid")
        baseline["authority_hash"] = authority["authority_hash"]
        baseline["baseline_hash"] = _baseline_hash(baseline)
        atomic_write_json(baseline_path, baseline)
    return {
        "status": "MIGRATED",
        "approved_candidate_count": len(approved),
        "candidate_snapshot_reduced_only": len(approved) <= len(approved_before),
        "previous_approved_candidate_count": len(approved_before),
        "maximum_order_eur": authority["maximum_order_eur"],
        "maximum_total_exposure_eur": authority.get(
            "maximum_total_exposure_eur"
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def positive_strategy_dna_approval_phrase(strategy_id: str) -> str:
    """Return the exact local phrase for one immutable strategy identity."""

    return f"{DNA_APPROVAL_PREFIX} {str(strategy_id).strip()} CONFIRMED"


def migrate_positive_strategy_live_capital_level_2(
    settings: Settings,
    *,
    approval_phrase: str,
) -> dict[str, Any]:
    """Raise only capital limits for the existing frozen DNA snapshot."""

    if approval_phrase.strip() != CAPITAL_LEVEL_2_APPROVAL_PHRASE:
        raise PermissionError("capital Level-2 approval phrase mismatch")
    path = _paths(settings)["authority"]
    if not path.is_file():
        raise FileNotFoundError("positive strategy authority is missing")
    authority = dict(read_json(path))
    safely_recoverable_cap_mismatch = (
        authority.get("active") is False
        and authority.get("deactivation_reason")
        == "POSITIVE_STRATEGY_AUTHORITY_BLOCKED"
        and _validate_authority_hash(authority)
    )
    if (
        authority.get("schema_version") != SCHEMA_VERSION
        or (
            authority.get("active") is not True
            and not safely_recoverable_cap_mismatch
        )
        or not _validate_authority_hash(authority)
        or authority.get("autoscale") is not False
        or authority.get("spot_only") is not True
    ):
        raise PermissionError("existing positive strategy authority is invalid")
    portfolio = managed_live_portfolio(settings)
    if (
        int(portfolio["managed_position_count"]) > MAXIMUM_OPEN_POSITIONS
        or _decimal(portfolio["managed_exposure_eur"])
        > MAXIMUM_TOTAL_EXPOSURE_EUR
    ):
        raise PermissionError("current managed portfolio exceeds Level-2 caps")
    previous = {
        "maximum_order_eur": authority.get("maximum_order_eur"),
        "maximum_total_exposure_eur": authority.get(
            "maximum_total_exposure_eur"
        ),
        "maximum_open_positions": authority.get("maximum_open_positions"),
        "maximum_new_orders_per_day": authority.get(
            "maximum_new_orders_per_day"
        ),
    }
    authority.update(
        {
            "active": True,
            "capital_level": CAPITAL_LEVEL,
            "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
            "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
            "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
            "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
            "maximum_risk_per_trade_eur": "2",
            "autoscale": False,
            "last_capital_level_migration_at": utc_iso(),
            "last_capital_level_previous": previous,
            "deactivated_at": None,
            "deactivation_reason": None,
            "approval_phrase_stored": False,
        }
    )
    authority["authority_hash"] = stable_hash(
        {
            key: value
            for key, value in authority.items()
            if key != "authority_hash"
        },
        length=64,
    )
    atomic_write_json(path, authority)
    baseline_path = _paths(settings)["baseline"]
    if baseline_path.is_file():
        baseline = dict(read_json(baseline_path))
        if baseline.get("baseline_hash") != _baseline_hash(baseline):
            raise PermissionError("existing inventory baseline is invalid")
        baseline["authority_hash"] = authority["authority_hash"]
        baseline["baseline_hash"] = _baseline_hash(baseline)
        atomic_write_json(baseline_path, baseline)
    return {
        "status": "CAPITAL_LEVEL_2_ACTIVE",
        "capital_level": CAPITAL_LEVEL,
        "maximum_order_eur": str(MAXIMUM_ORDER_EUR),
        "maximum_total_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
        "maximum_open_positions": MAXIMUM_OPEN_POSITIONS,
        "maximum_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
        "maximum_risk_per_trade_eur": "2",
        "autoscale": False,
        "frozen_dna_unchanged": True,
        "managed_portfolio": portfolio,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def approve_positive_strategy_dna(
    settings: Settings,
    *,
    strategy_id: str,
    approval_phrase: str,
) -> dict[str, Any]:
    """Append exactly one current frozen DNA to the existing live authority."""

    selected_id = str(strategy_id or "").strip()
    if not selected_id:
        raise ValueError("strategy id is required")
    expected_phrase = positive_strategy_dna_approval_phrase(selected_id)
    if approval_phrase != expected_phrase:
        raise PermissionError("positive strategy DNA approval phrase mismatch")
    active, authority, failures = (
        synchronize_positive_strategy_live_authority(settings)
    )
    if not active:
        raise PermissionError(
            "positive strategy authority is not valid: " + ",".join(failures)
        )
    current = _candidate_snapshot(
        settings,
        load_generated_candidates(settings),
    )
    matches = [row for row in current if row["strategy_id"] == selected_id]
    if len(matches) != 1:
        raise ValueError(
            "strategy id must identify exactly one current frozen candidate"
        )
    selected = matches[0]
    approved = [
        dict(row) for row in authority.get("approved_candidates") or []
    ]
    approved_by_dna = {
        str(row.get("strategy_dna_hash") or ""): row for row in approved
    }
    dna = selected["strategy_dna_hash"]
    if dna in approved_by_dna:
        if approved_by_dna[dna] != selected:
            raise PermissionError("approved strategy DNA identity drift")
        return {
            "status": "ALREADY_APPROVED",
            "strategy_id": selected_id,
            "strategy_dna_hash": dna,
            "approval_phrase_stored": False,
            "approved_candidate_count": len(approved),
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    approved.append(selected)
    approved.sort(key=lambda row: row["strategy_dna_hash"])
    authority["approved_candidates"] = approved
    authority["candidate_snapshot_hash"] = stable_hash(approved, length=64)
    authority["last_operator_approved_at"] = utc_iso()
    authority["last_operator_approved_strategy_id"] = selected_id
    authority["last_operator_approved_dna"] = dna
    authority["approval_phrase_stored"] = False
    authority["authority_hash"] = stable_hash(
        {
            key: value
            for key, value in authority.items()
            if key != "authority_hash"
        },
        length=64,
    )
    authority_path = _paths(settings)["authority"]
    atomic_write_json(authority_path, authority)
    baseline_path = _paths(settings)["baseline"]
    if baseline_path.is_file():
        baseline = dict(read_json(baseline_path))
        if (
            baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
            or baseline.get("baseline_hash") != _baseline_hash(baseline)
        ):
            raise PermissionError("existing inventory baseline is invalid")
        baseline["authority_hash"] = authority["authority_hash"]
        baseline["baseline_hash"] = _baseline_hash(baseline)
        atomic_write_json(baseline_path, baseline)
    return {
        "status": "APPROVED",
        "strategy_id": selected_id,
        "strategy_dna_hash": dna,
        "frozen_candidate_hash": selected["frozen_candidate_hash"],
        "timeframe": selected["timeframe"],
        "approved_markets": selected["approved_markets"],
        "approval_phrase_stored": False,
        "approved_candidate_count": len(approved),
        "authority_artifact": str(authority_path),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def deactivate_positive_strategy_live_authority(
    settings: Settings,
    *,
    reason: str,
) -> dict[str, Any]:
    path = _paths(settings)["authority"]
    payload = dict(read_json(path)) if path.is_file() else {}
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "active": False,
            "deactivated_at": utc_iso(),
            "deactivation_reason": str(reason),
        }
    )
    if payload:
        payload["authority_hash"] = stable_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "authority_hash"
            },
            length=64,
        )
        atomic_write_json(path, payload)
    return {
        "status": "DEACTIVATED",
        "reason": str(reason),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _validate_authority_hash(authority: Mapping[str, Any]) -> bool:
    expected = stable_hash(
        {
            key: value
            for key, value in authority.items()
            if key != "authority_hash"
        },
        length=64,
    )
    return str(authority.get("authority_hash") or "") == expected


def synchronize_positive_strategy_live_authority(
    settings: Settings,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Enroll new exact-positive frozen DNA without accepting identity drift."""

    path = _paths(settings)["authority"]
    if not path.is_file():
        return False, {}, ["POSITIVE_STRATEGY_AUTHORITY_MISSING"]
    authority = dict(read_json(path))
    failures: list[str] = []
    if authority.get("schema_version") != SCHEMA_VERSION:
        failures.append("POSITIVE_STRATEGY_AUTHORITY_SCHEMA_MISMATCH")
    if authority.get("active") is not True:
        failures.append("POSITIVE_STRATEGY_AUTHORITY_INACTIVE")
    if not _validate_authority_hash(authority):
        failures.append("POSITIVE_STRATEGY_AUTHORITY_HASH_MISMATCH")
    if (
        _decimal(authority.get("maximum_order_eur")) != MAXIMUM_ORDER_EUR
        or _decimal(authority.get("maximum_total_exposure_eur"))
        != MAXIMUM_TOTAL_EXPOSURE_EUR
        or int(authority.get("maximum_open_positions") or 0)
        != MAXIMUM_OPEN_POSITIONS
        or int(authority.get("maximum_new_orders_per_day") or 0)
        != MAXIMUM_NEW_ORDERS_PER_DAY
        or authority.get("spot_only") is not True
        or authority.get("long_only") is not True
        or authority.get("autoscale") is not False
    ):
        failures.append("POSITIVE_STRATEGY_AUTHORITY_CAP_MISMATCH")
    if failures:
        return False, authority, failures

    current = _candidate_snapshot(settings, load_generated_candidates(settings))
    current_by_dna = {row["strategy_dna_hash"]: row for row in current}
    approved = [
        dict(row) for row in authority.get("approved_candidates") or []
    ]
    approved_by_dna = {row.get("strategy_dna_hash"): row for row in approved}
    migrated_dna: list[str] = []
    for dna, stored in approved_by_dna.items():
        candidate = current_by_dna.get(str(dna))
        if candidate is None or candidate == stored:
            continue
        comparable_candidate = {
            key: value
            for key, value in candidate.items()
            if key not in {"frozen_candidate_hash", "frozen_identity_schema"}
        }
        comparable_stored = {
            key: value
            for key, value in stored.items()
            if key not in {"frozen_candidate_hash", "frozen_identity_schema"}
        }
        if (
            candidate.get("frozen_identity_schema")
            == "EXECUTION_SEMANTICS_V2"
            and comparable_candidate == comparable_stored
        ):
            stored.update(candidate)
            migrated_dna.append(str(dna))
            continue
        failures.append(f"APPROVED_DNA_IDENTITY_DRIFT:{dna}")
    if failures:
        return False, authority, failures

    if migrated_dna:
        approved.sort(key=lambda row: row["strategy_dna_hash"])
        authority["approved_candidates"] = approved
        authority["candidate_snapshot_hash"] = stable_hash(
            approved,
            length=64,
        )
        authority["last_identity_schema_migration_at"] = utc_iso()
        authority["last_identity_schema_migrated_dna"] = sorted(
            migrated_dna
        )
        authority["authority_hash"] = stable_hash(
            {
                key: value
                for key, value in authority.items()
                if key != "authority_hash"
            },
            length=64,
        )
        atomic_write_json(path, authority)
        baseline_path = _paths(settings)["baseline"]
        if baseline_path.is_file():
            baseline = dict(read_json(baseline_path))
            baseline["authority_hash"] = authority["authority_hash"]
            baseline["baseline_hash"] = _baseline_hash(baseline)
            atomic_write_json(baseline_path, baseline)

    added = [
        row
        for dna, row in current_by_dna.items()
        if dna not in approved_by_dna
    ]
    if added and authority.get("auto_enroll_future_exact_positive_dna") is True:
        approved.extend(added)
        approved.sort(key=lambda row: row["strategy_dna_hash"])
        authority["approved_candidates"] = approved
        authority["candidate_snapshot_hash"] = stable_hash(approved, length=64)
        authority["last_auto_enrollment_at"] = utc_iso()
        authority["last_auto_enrolled_dna"] = [
            row["strategy_dna_hash"] for row in added
        ]
        authority["authority_hash"] = stable_hash(
            {
                key: value
                for key, value in authority.items()
                if key != "authority_hash"
            },
            length=64,
        )
        atomic_write_json(path, authority)
        baseline_path = _paths(settings)["baseline"]
        if baseline_path.is_file():
            baseline = dict(read_json(baseline_path))
            baseline["authority_hash"] = authority["authority_hash"]
            baseline["baseline_hash"] = _baseline_hash(baseline)
            atomic_write_json(baseline_path, baseline)
    return True, authority, []


def positive_strategy_live_authority_status(
    settings: Settings,
) -> dict[str, Any]:
    active, authority, failures = synchronize_positive_strategy_live_authority(
        settings
    )
    state_path = _paths(settings)["state"]
    state = dict(read_json(state_path)) if state_path.is_file() else {}
    return {
        "status": "ACTIVE" if active else "BLOCKED",
        "authority_active": active,
        "authority_failures": failures,
        "approved_candidate_count": len(
            authority.get("approved_candidates") or []
        ),
        "maximum_order_eur": authority.get("maximum_order_eur"),
        "maximum_total_exposure_eur": authority.get(
            "maximum_total_exposure_eur"
        ),
        "maximum_open_positions": authority.get("maximum_open_positions"),
        "maximum_new_orders_per_day": authority.get(
            "maximum_new_orders_per_day"
        ),
        "open_generated_positions": len(state.get("positions") or {}),
        "positions": state.get("positions") or {},
        "last_cycle_at": state.get("last_cycle_at"),
        "last_reason": state.get("last_reason"),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _baseline_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "schema_version": payload.get("schema_version"),
            "authority_hash": payload.get("authority_hash"),
            "quantities": payload.get("quantities"),
        },
        length=64,
    )


def _write_baseline(
    settings: Settings,
    *,
    authority: Mapping[str, Any],
    balances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    quantities = {
        str(row.get("symbol") or "").upper(): str(
            _decimal(row.get("available"))
            + _decimal(row.get("inOrder"))
        )
        for row in balances
        if str(row.get("symbol") or "").upper() not in {"", "EUR"}
        and _decimal(row.get("available")) + _decimal(row.get("inOrder")) > 0
    }
    payload: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "created_at": utc_iso(),
        "source": "BITVAVO_PRIVATE_BALANCE_READ",
        "authority_hash": authority["authority_hash"],
        "quantities": dict(sorted(quantities.items())),
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    payload["baseline_hash"] = _baseline_hash(payload)
    atomic_write_json(_paths(settings)["baseline"], payload)
    return payload


def _load_baseline(
    settings: Settings,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Decimal], list[str]]:
    path = _paths(settings)["baseline"]
    if not path.is_file():
        return {}, ["POSITIVE_STRATEGY_INVENTORY_BASELINE_MISSING"]
    payload = dict(read_json(path))
    failures: list[str] = []
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        failures.append("POSITIVE_STRATEGY_BASELINE_SCHEMA_MISMATCH")
    # A deactivation changes the authority hash by design.  The immutable,
    # internally hashed balance baseline remains valid for exit-only
    # reconciliation; requiring the new inactive hash here would prevent the
    # very protective-stop recovery that must continue without entry authority.
    if (
        payload.get("authority_hash") != authority.get("authority_hash")
        and authority.get("active") is True
    ):
        failures.append("POSITIVE_STRATEGY_BASELINE_AUTHORITY_MISMATCH")
    if payload.get("baseline_hash") != _baseline_hash(payload):
        failures.append("POSITIVE_STRATEGY_BASELINE_HASH_MISMATCH")
    quantities = {
        str(symbol).upper(): _decimal(quantity)
        for symbol, quantity in dict(payload.get("quantities") or {}).items()
    }
    return quantities, failures


def _state(settings: Settings) -> dict[str, Any]:
    path = _paths(settings)["state"]
    if path.is_file():
        state = dict(read_json(path))
        last_closed = state.get("last_closed_position")
        if isinstance(last_closed, Mapping) and last_closed.get("closed_at"):
            normalized = {**last_closed, "status": "CLOSED"}
            if (
                normalized.get("exit_reason")
                == "NATIVE_PROTECTIVE_STOP_FILLED"
            ):
                normalized["protective_stop_status"] = "filled"
                normalized["native_protective_stop_active"] = False
            state["last_closed_position"] = normalized
        return state
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "READY",
        "positions": {},
        "last_closed_position": None,
        "last_cycle_at": None,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def signal_execution_window(
    *,
    signal_timestamp: str,
    timeframe: str,
    observed_at: datetime | None = None,
) -> tuple[bool, str, str]:
    """Return whether a closed-candle signal is inside its next-open window."""

    if timeframe not in TIMEFRAME_SECONDS:
        return False, "", "UNSUPPORTED_TIMEFRAME"
    try:
        opened_at = datetime.fromisoformat(
            signal_timestamp.replace("Z", "+00:00")
        ).astimezone(UTC)
    except (TypeError, ValueError):
        return False, "", "INVALID_SIGNAL_TIMESTAMP"
    closes_at = opened_at + timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    now = (observed_at or utc_now()).astimezone(UTC)
    window_ends = closes_at + timedelta(minutes=ENTRY_WINDOW_MINUTES)
    if now < closes_at:
        return False, closes_at.isoformat(), "SIGNAL_CANDLE_NOT_CLOSED"
    if now > window_ends:
        return False, closes_at.isoformat(), "NEXT_OPEN_WINDOW_EXPIRED"
    return True, closes_at.isoformat(), "NEXT_OPEN_WINDOW_ACTIVE"


def rank_natural_entries(
    *,
    candidates: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Any],
    authority: Mapping[str, Any],
    observed_at: datetime | None = None,
    occupied_markets: Sequence[str] = (),
    occupied_dna: Sequence[str] = (),
    degradation: Mapping[str, Mapping[str, Any]] | None = None,
    macro_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    approved = {
        str(row["strategy_dna_hash"]): dict(row)
        for row in authority.get("approved_candidates") or []
    }
    occupied_market_set = {str(value) for value in occupied_markets}
    occupied_dna_set = {str(value) for value in occupied_dna}
    degradation = degradation or {}
    macro_overlay_enabled = macro_context is not None
    selected_macro = dict(macro_context or {})
    macro_regime = str(
        selected_macro.get("regime") or "DATA_BLOCKED"
    ).upper()
    macro_confidence = max(
        ZERO,
        min(
            Decimal("1"),
            _decimal(selected_macro.get("confidence")),
        ),
    )
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        dna = str(candidate.get("strategy_dna_hash") or "")
        if dna not in approved or dna in occupied_dna_set:
            continue
        stored = approved[dna]
        degradation_row = dict(degradation.get(dna) or {})
        if degradation_row.get("entry_allowed") is False:
            continue
        if (
            dna != stored.get("strategy_dna_hash")
            or str(candidate.get("frozen_candidate_hash") or "")
            != stored.get("frozen_candidate_hash")
            or str(candidate.get("timeframe") or "")
            != stored.get("timeframe")
        ):
            continue
        evaluation = dict(evaluations.get(dna) or {})
        if evaluation.get("status") != "EVALUATED":
            continue
        metrics = dict(candidate.get("metrics") or {})
        family = str(
            candidate.get("economic_hypothesis_family")
            or candidate.get("family")
            or candidate.get("strategy_id")
            or "UNKNOWN"
        ).upper()
        for market_eval in evaluation.get("markets") or []:
            market = str(market_eval.get("market") or "")
            if (
                market not in set(stored.get("approved_markets") or [])
                or market in occupied_market_set
                or market_eval.get("entry") is not True
                or market_eval.get("stale") is True
            ):
                continue
            macro_policy = "NOT_APPLIED"
            macro_risk_multiplier = Decimal("1")
            macro_reason = "MACRO_OVERLAY_NOT_PROVIDED"
            if macro_overlay_enabled:
                (
                    macro_policy,
                    raw_macro_multiplier,
                    macro_reason,
                ) = regime_policy(macro_regime, family, market)
                macro_risk_multiplier = max(
                    ZERO,
                    min(Decimal("1"), Decimal(str(raw_macro_multiplier))),
                )
                if macro_policy not in {"ENABLE", "REDUCE"}:
                    continue
            active, execute_at, window_reason = signal_execution_window(
                signal_timestamp=str(
                    market_eval.get("signal_timestamp") or ""
                ),
                timeframe=str(candidate.get("timeframe") or ""),
                observed_at=observed_at,
            )
            if not active:
                continue
            observed_profit_factor = max(
                Decimal("0"),
                min(Decimal("5"), _decimal(metrics.get("profit_factor"))),
            )
            sample_prior = Decimal("1")
            sample_k = Decimal("50")
            trade_count = max(0, int(metrics.get("trade_count") or 0))
            sample_weight = (
                Decimal(trade_count) / (Decimal(trade_count) + sample_k)
                if trade_count > 0
                else ZERO
            )
            adjusted_profit_factor = (
                sample_weight * observed_profit_factor
                + (Decimal("1") - sample_weight) * sample_prior
            )
            available_robust_pfs = [
                _decimal(metrics.get(field), default=Decimal("NaN"))
                for field in (
                    "profit_factor",
                    "stressed_profit_factor",
                    "double_cost_profit_factor",
                    "holdout_profit_factor",
                    "walk_forward_profit_factor",
                    "cross_asset_profit_factor",
                )
            ]
            available_robust_pfs = [
                value
                for value in available_robust_pfs
                if value.is_finite() and value > ZERO
            ]
            observed_robust_profit_factor = (
                min(available_robust_pfs)
                if available_robust_pfs
                else observed_profit_factor
            )
            robust_profit_factor = (
                sample_weight * observed_robust_profit_factor
                + (Decimal("1") - sample_weight) * sample_prior
            )
            confidence_completeness = Decimal(
                min(4, len(available_robust_pfs))
            ) / Decimal("4")
            sample_confidence_multiplier = (
                Decimal("0.5") + Decimal("0.5") * sample_weight
            )
            net_return = max(
                Decimal("-1"),
                min(Decimal("5"), _decimal(metrics.get("net_return"))),
            )
            degradation_multiplier = max(
                ZERO,
                min(
                    Decimal("1"),
                    _decimal(
                        degradation_row.get("risk_multiplier"),
                        default=Decimal("1"),
                    ),
                ),
            )
            macro_confidence_multiplier = (
                Decimal("0.5") + Decimal("0.5") * macro_confidence
                if macro_overlay_enabled
                else Decimal("1")
            )
            combined_risk_multiplier = (
                degradation_multiplier * macro_risk_multiplier
            )
            score = (
                adjusted_profit_factor * Decimal("30")
                + robust_profit_factor * Decimal("20")
                + max(Decimal("0"), net_return) * Decimal("15")
                + min(Decimal("15"), Decimal(trade_count).sqrt())
                + confidence_completeness * Decimal("10")
            ) * (
                sample_confidence_multiplier
                * combined_risk_multiplier
                * macro_confidence_multiplier
            )
            signal_size_multiplier = max(
                ZERO,
                min(
                    Decimal("1"),
                    _decimal(
                        market_eval.get("size_multiplier"),
                        default=Decimal("1"),
                    ),
                ),
            )
            signal_id = stable_hash(
                [
                    dna,
                    market,
                    candidate.get("timeframe"),
                    market_eval.get("signal_timestamp"),
                    candidate.get("frozen_candidate_hash"),
                ],
                length=40,
            )
            ranked.append(
                {
                    "strategy_id": stored["strategy_id"],
                    "strategy_dna_hash": dna,
                    "frozen_candidate_hash": stored[
                        "frozen_candidate_hash"
                    ],
                    "market": market,
                    "timeframe": candidate["timeframe"],
                    "signal_id": signal_id,
                    "signal_timestamp": market_eval["signal_timestamp"],
                    "execute_at": execute_at,
                    "window_reason": window_reason,
                    "stop_distance": market_eval.get("stop_distance"),
                    "target_distance": market_eval.get("target_distance"),
                    "size_multiplier": float(signal_size_multiplier),
                    "effective_size_multiplier": float(
                        signal_size_multiplier * combined_risk_multiplier
                    ),
                    "score": float(score),
                    "adjusted_profit_factor": float(
                        adjusted_profit_factor
                    ),
                    "robust_profit_factor": float(robust_profit_factor),
                    "sample_weight": float(sample_weight),
                    "selection_confidence_completeness": float(
                        confidence_completeness
                    ),
                    "sample_confidence_multiplier": float(
                        sample_confidence_multiplier
                    ),
                    "degradation_state": degradation_row.get(
                        "degradation_state",
                        "VALIDATING",
                    ),
                    "degradation_risk_multiplier": float(
                        degradation_multiplier
                    ),
                    "macro_overlay_status": selected_macro.get(
                        "status",
                        "NOT_PROVIDED",
                    ),
                    "macro_regime": (
                        macro_regime
                        if macro_overlay_enabled
                        else "NOT_APPLIED"
                    ),
                    "macro_policy": macro_policy,
                    "macro_policy_reason": macro_reason,
                    "macro_confidence": float(macro_confidence),
                    "macro_risk_multiplier": float(
                        macro_risk_multiplier
                    ),
                    "macro_confidence_multiplier": float(
                        macro_confidence_multiplier
                    ),
                    "risk_multiplier": float(combined_risk_multiplier),
                    "profit_factor": float(
                        _decimal(metrics.get("profit_factor"))
                    ),
                    "net_return": float(
                        _decimal(metrics.get("net_return"))
                    ),
                    "trade_count": trade_count,
                }
            )
    return sorted(
        ranked,
        key=lambda row: (
            row["score"],
            row["trade_count"],
            row["strategy_dna_hash"],
            row["market"],
        ),
        reverse=True,
    )


def _health(settings: Settings) -> tuple[dict[str, Any], list[str]]:
    path = settings.paths.output_dir / "operations" / "live_account_health.json"
    if not path.is_file():
        return {}, ["LIVE_ACCOUNT_HEALTH_MISSING"]
    payload = dict(read_json(path))
    failures: list[str] = []
    try:
        checked_at = datetime.fromisoformat(
            str(payload.get("checked_at") or "").replace("Z", "+00:00")
        ).astimezone(UTC)
    except ValueError:
        failures.append("LIVE_ACCOUNT_HEALTH_TIMESTAMP_INVALID")
    else:
        if utc_now() - checked_at > timedelta(minutes=10):
            failures.append("LIVE_ACCOUNT_HEALTH_STALE")
    if payload.get("status") != "READY":
        failures.append("LIVE_ACCOUNT_HEALTH_NOT_READY")
    valuation = dict(
        (payload.get("account") or {}).get("portfolio_valuation") or {}
    )
    if valuation.get("status") != "COMPLETE_MARK_TO_MARKET":
        failures.append("WALLET_VALUATION_INCOMPLETE")
    return payload, failures


def _material_wallet_positions(health: Mapping[str, Any]) -> tuple[int, list[str]]:
    holdings = list(
        ((health.get("account") or {}).get("portfolio_valuation") or {}).get(
            "holdings"
        )
        or []
    )
    markets = sorted(
        {
            str(row.get("market") or "")
            for row in holdings
            if _decimal(row.get("estimated_value_eur"))
            >= MINIMUM_MATERIAL_POSITION_EUR
        }
    )
    return len(markets), markets


def _managed_excess(
    balances: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    current = {
        str(row.get("symbol") or "").upper(): (
            _decimal(row.get("available"))
            + _decimal(row.get("inOrder"))
        )
        for row in balances
        if str(row.get("symbol") or "").upper() not in {"", "EUR"}
    }
    return {
        symbol: quantity - baseline.get(symbol, Decimal("0"))
        for symbol, quantity in current.items()
        if quantity - baseline.get(symbol, Decimal("0")) > 0
    }


def _replacement_entry_notional(
    position: Mapping[str, Any],
) -> Decimal:
    """Preserve, never increase, the original pending entry allocation."""

    return min(
        MAXIMUM_ORDER_EUR,
        _decimal(
            position.get("requested_quantity")
            or position.get("quantity")
        )
        * _decimal(
            position.get("limit_price")
            or position.get("entry_price")
        ),
    )


def _position_quantity_from_order(
    order: Mapping[str, Any],
    *,
    fallback: Decimal,
) -> Decimal:
    if order.get("filledAmount") is not None:
        return _decimal(order.get("filledAmount"))
    return fallback


def _position_price_from_order(
    order: Mapping[str, Any],
    *,
    quantity: Decimal,
    fallback: Decimal,
) -> Decimal:
    quote = _decimal(order.get("filledAmountQuote"))
    if quote > 0 and quantity > 0:
        return quote / quantity
    return _decimal(order.get("price")) or fallback


def _normalized_order_status(order: Mapping[str, Any]) -> str:
    return (
        str(order.get("status") or "")
        .replace("_", "")
        .replace("-", "")
        .casefold()
    )


def _live_order_notification_payload(
    order: Mapping[str, Any],
    *,
    market: str,
    side: str,
    order_type: str,
    requested_quantity: Decimal,
    fallback_price: Decimal,
    strategy_id: str,
    timeframe: str,
) -> dict[str, Any]:
    """Build a complete, REST-verified and secret-safe live order message."""

    status = _normalized_order_status(order)
    filled = _position_quantity_from_order(
        order,
        fallback=requested_quantity if status == "filled" else Decimal("0"),
    )
    average_price = _position_price_from_order(
        order,
        quantity=filled,
        fallback=fallback_price,
    )
    remaining = _decimal(
        order.get("amountRemaining")
        or order.get("remainingAmount")
    )
    if remaining <= 0 and status not in {"filled", "canceled", "cancelled"}:
        remaining = max(Decimal("0"), requested_quantity - filled)
    invested = _decimal(order.get("filledAmountQuote"))
    if invested <= 0:
        invested = filled * average_price
    return {
        "order_id": order.get("orderId"),
        "market": market,
        "side": side,
        "order_type": order_type,
        "requested_quantity": str(requested_quantity),
        "filled_quantity": str(filled),
        "remaining_quantity": str(remaining),
        "average_fill_price": str(average_price),
        "invested_eur": str(invested),
        "fee": order.get("feePaid") or order.get("fee"),
        "strategy_id": strategy_id,
        "timeframe": timeframe,
        "venue_timestamp": order.get("updated") or order.get("created"),
        "status": order.get("status"),
        "verification_source": "BITVAVO_REST_ORDER_RESPONSE",
    }


def _gtc_entry_expired(
    position: Mapping[str, Any],
    *,
    observed_at: datetime,
    validity_seconds: int,
) -> bool:
    if str(position.get("time_in_force") or "") != "GTC":
        return False
    raw = position.get("entry_order_submitted_at") or position.get("opened_at")
    try:
        submitted_at = datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        ).astimezone(UTC)
    except (TypeError, ValueError):
        return True
    return (observed_at - submitted_at).total_seconds() >= validity_seconds


def _dynamic_entry_notional(
    *,
    account_equity_eur: Decimal,
    available_eur: Decimal,
    selected_entry: Mapping[str, Any],
    authority_cap_eur: Decimal,
    venue_minimum_eur: Decimal = Decimal("5"),
    live_minimum_eur: Decimal = Decimal("10"),
) -> tuple[Decimal, dict[str, Any]]:
    """Size by equity and setup strength while preserving hard authority caps."""

    score = _decimal(selected_entry.get("score"))
    if score >= Decimal("60"):
        setup_tier = "EXCEPTIONALLY_STRONG"
        base_fraction = Decimal("0.015")
    elif score >= Decimal("45"):
        setup_tier = "STRONG"
        base_fraction = Decimal("0.010")
    else:
        setup_tier = "NORMAL"
        base_fraction = Decimal("0.005")
    signal_multiplier = max(
        Decimal("0.05"),
        min(
            Decimal("1"),
            _decimal(
                selected_entry.get("effective_size_multiplier"),
                default=Decimal("1"),
            ),
        ),
    )
    raw_target = account_equity_eur * base_fraction * signal_multiplier
    ceiling = min(authority_cap_eur, max(Decimal("0"), available_eur))
    minimum_required = max(venue_minimum_eur, live_minimum_eur)
    if ceiling < minimum_required:
        return Decimal("0"), {
            "setup_tier": setup_tier,
            "reason": "AVAILABLE_OR_AUTHORISED_CAP_BELOW_LIVE_MINIMUM",
            "raw_target_eur": str(raw_target),
            "approved_notional_eur": "0",
        }
    approved = min(ceiling, max(minimum_required, raw_target))
    return approved, {
        "setup_tier": setup_tier,
        "base_equity_fraction": str(base_fraction),
        "signal_and_regime_multiplier": str(signal_multiplier),
        "raw_target_eur": str(raw_target),
        "approved_notional_eur": str(approved),
        "authority_cap_eur": str(authority_cap_eur),
        "venue_minimum_eur": str(venue_minimum_eur),
        "live_minimum_eur": str(live_minimum_eur),
        "live_minimum_dominated": raw_target < minimum_required,
    }


async def _plan_live_entry_order(
    settings: Settings,
    *,
    client: Any,
    market: str,
    requested_notional_eur: Decimal,
    public_price: Decimal,
    liquidity: Mapping[str, Any],
) -> EntryOrderPlan:
    """Prefer a bounded GTC/IOC/FOK limit without weakening the EUR cap."""

    limit_enabled = settings.execution.live_limit_entries_enabled
    fallback_enabled = (
        settings.execution.live_limit_market_fallback_enabled
    )
    try:
        rules = await client.execution_market_rules(market)
    except ExecutionBlocked:
        if limit_enabled and not fallback_enabled:
            raise
        return EntryOrderPlan(
            order_type=OrderType.MARKET,
            quantity=requested_notional_eur / public_price,
            limit_price=None,
            time_in_force=OrderTimeInForce.GTC,
            planned_notional_eur=requested_notional_eur,
            execution_policy="QUOTE_MARKET_LIQUIDITY_PREFLIGHT",
            fallback_reason="VENUE_EXECUTION_RULES_UNAVAILABLE",
        )
    limits = dict(liquidity.get("limits") or {})
    plan = plan_bounded_entry_order(
        requested_notional_eur=requested_notional_eur,
        best_ask=_decimal(liquidity.get("best_ask")),
        estimated_average_price=_decimal(
            liquidity.get("estimated_average_price")
        ),
        maximum_slippage_bps=_decimal(
            limits.get("maximum_slippage_bps")
        ),
        rules=rules,
        limit_enabled=limit_enabled,
        time_in_force=OrderTimeInForce(
            settings.execution.live_limit_entry_time_in_force
        ),
        price_buffer_bps=Decimal(
            str(settings.execution.live_limit_price_buffer_bps)
        ),
        market_fallback_enabled=fallback_enabled,
    )
    if plan.order_type is OrderType.MARKET:
        return EntryOrderPlan(
            order_type=plan.order_type,
            quantity=requested_notional_eur / public_price,
            limit_price=None,
            time_in_force=OrderTimeInForce.GTC,
            planned_notional_eur=requested_notional_eur,
            execution_policy=plan.execution_policy,
            fallback_reason=plan.fallback_reason,
        )
    return plan


def _supports_order_type(
    supported_order_types: Sequence[str],
    expected: str,
) -> bool:
    normalized = {
        str(value).replace("_", "").replace("-", "").casefold()
        for value in supported_order_types
    }
    return expected.replace("_", "").replace("-", "").casefold() in normalized


def _generated_protective_stop_intent(
    position: Mapping[str, Any],
    *,
    quantity: Decimal,
    trigger_price: Decimal,
) -> OrderIntent:
    identity = stable_hash(
        [
            "GENERATED_NATIVE_PROTECTIVE_STOP",
            position.get("strategy_dna_hash"),
            position.get("signal_id"),
            position.get("market"),
            str(quantity),
            str(trigger_price),
        ],
        length=40,
    )
    return OrderIntent(
        intent_id=identity[:32],
        idempotency_key=f"generated-live-protective-stop:{identity}",
        market=str(position["market"]),
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=quantity,
        trigger_price=trigger_price,
        trigger_reference="bestBid",
        strategy_id=str(position.get("strategy_id") or ""),
        strategy_dna_hash=str(position.get("strategy_dna_hash") or ""),
        signal_id=str(position.get("signal_id") or ""),
        portfolio_decision_id=identity,
        reason_codes=(
            "NATIVE_PROTECTIVE_STOP",
            "LOCAL_HARD_STOP_REMAINS_ACTIVE",
        ),
    )


async def _place_generated_native_stop(
    client: Any,
    *,
    capability: Any,
    position: Mapping[str, Any],
    quantity: Decimal,
    estimated_price: Decimal,
) -> dict[str, Any]:
    market = str(position["market"])
    rules = await client.execution_market_rules(market)
    if not _supports_order_type(rules.supported_order_types, "stopLoss"):
        raise ExecutionBlocked("venue does not support native stop-loss orders")
    trigger = rules.price(_decimal(position.get("stop_loss")))
    protected_quantity = rules.amount(quantity)
    if not quantity_is_protectable_at_stop(
        quantity=protected_quantity,
        stop_price=trigger,
        rules=rules,
    ):
        raise ExecutionBlocked("filled quantity is below native stop minimum")
    intent = _generated_protective_stop_intent(
        position,
        quantity=protected_quantity,
        trigger_price=trigger,
    )
    order = await client.submit_order(
        intent,
        capability=capability,
        estimated_price=estimated_price,
        reconciled_owned_quantity=quantity,
        exchange_minimum_order_eur=rules.minimum_order_value_eur,
    )
    status = _normalized_order_status(order)
    if status not in {"new", "awaitingtrigger"}:
        raise ReconciliationRequired(
            "native generated-strategy protective stop is not active"
        )
    return {
        "protective_stop_order_id": str(order["orderId"]),
        "protective_stop_client_order_id": client.client_order_id_for(
            intent.idempotency_key
        ),
        "protective_stop_status": str(order.get("status") or ""),
        "protective_stop_trigger": str(trigger),
        "local_hard_stop_active": True,
        "native_protective_stop_active": True,
        "native_protective_stop_order": order,
    }


def _write_state(settings: Settings, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    atomic_write_json(_paths(settings)["state"], payload)
    atomic_write_json(_paths(settings)["status"], payload)


async def execute_generated_strategy_live_once(
    settings: Settings,
    *,
    submit: bool,
    allow_new_entry: bool = True,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Manage exits first, then submit at most one natural entry per cycle."""

    now = (observed_at or utc_now()).astimezone(UTC)
    weekly_budget = WeeklyTradeBudgetManager(settings)
    cooldowns = SwingCooldownManager(settings)
    weekly_status = weekly_budget.status(observed_at=now)
    degradation_path = (
        settings.paths.output_dir
        / "operations"
        / "strategy_degradation.json"
    )
    degradation_payload = (
        dict(read_json(degradation_path))
        if degradation_path.is_file()
        else {}
    )
    degradation = degradation_by_dna(degradation_payload)
    active, authority, authority_failures = (
        synchronize_positive_strategy_live_authority(settings)
    )
    economics_gate = load_canonical_entry_economics_gate(settings)
    economics_live_dna = set(
        economics_gate["live_entry_strategy_dna_hashes"]
    )
    state = _state(settings)
    state["schema_version"] = STATE_SCHEMA_VERSION
    state["last_cycle_at"] = now.isoformat()
    state["orders_generated_this_cycle"] = 0
    state["orders_submitted_this_cycle"] = 0
    state["orders_cancelled_this_cycle"] = 0
    state["orders_repriced_this_cycle"] = 0
    state["fills_verified_this_cycle"] = 0
    state["private_exchange_requests"] = 0
    state["health_failures"] = []
    state["selected_entry"] = None
    state["canonical_economics_entry_gate"] = {
        "status": economics_gate["status"],
        "artifact_hash": economics_gate["artifact_hash"],
        "allowed_strategy_dna_hashes": sorted(economics_live_dna),
        "new_entries_allowed": bool(economics_live_dna),
        "position_management_affected": False,
    }
    # Cycle-local diagnostics must not survive after the underlying balance
    # discrepancy has reconciled; stale values otherwise make live status look
    # blocked even when the current account snapshot is clean.
    for diagnostic in (
        "unknown_excess",
        "baseline_failures",
        "preflight_failures",
        "reconciliation",
        "native_protective_stop",
        "entry_liquidity",
        "entry_order_plan",
        "position_sizing",
        "position_limit_status",
        "material_wallet_position_count",
        "material_wallet_markets",
    ):
        state.pop(diagnostic, None)
    macro_overlay = _load_live_macro_overlay(
        settings,
        observed_at=now,
    )
    state["macro_overlay"] = macro_overlay
    # Entry authority is deliberately separate from exit and reconciliation
    # authority.  Existing positions must still ingest native stop fills and
    # retain protection after entry authority is deactivated.
    state["authority_failures"] = [] if active else list(authority_failures)
    candidates = load_generated_candidates(settings)
    candidate_by_dna = {
        str(candidate["strategy_dna_hash"]): candidate
        for candidate in candidates
    }
    paper_path = settings.paths.output_dir / "paper" / "generated_strategy_state.json"
    paper = dict(read_json(paper_path)) if paper_path.is_file() else {}
    evaluations = dict(paper.get("evaluations") or {})
    health, health_failures = _health(settings)
    state["health_failures"] = list(health_failures)
    state["authority"] = {
        "approved_candidate_count": len(
            authority.get("approved_candidates") or []
        ),
        "maximum_order_eur": authority["maximum_order_eur"],
        "maximum_total_exposure_eur": authority[
            "maximum_total_exposure_eur"
        ],
        "maximum_open_positions": authority["maximum_open_positions"],
    }
    kill_switch = KillSwitch(
        settings.paths.checkpoints_dir / "kill_switch.json"
    )
    if not submit:
        if not active:
            state.update(
                {
                    "status": "AUTHORITY_BLOCKED",
                    "last_reason": "POSITIVE_STRATEGY_AUTHORITY_BLOCKED",
                    "authority_failures": authority_failures,
                }
            )
            _write_state(settings, state)
            return state
        ranking_inputs = {
            "candidates": candidates,
            "evaluations": evaluations,
            "authority": authority,
            "observed_at": now,
            "occupied_markets": [
                str(row.get("market") or "")
                for row in (state.get("positions") or {}).values()
            ],
            "occupied_dna": list((state.get("positions") or {}).keys()),
            "degradation": degradation,
        }
        entries_without_macro = rank_natural_entries(
            **ranking_inputs,
        )
        entries = rank_natural_entries(
            **ranking_inputs,
            macro_context=macro_overlay,
        )
        entries = [
            row
            for row in entries
            if str(row.get("strategy_dna_hash") or "").lower()
            in economics_live_dna
        ]
        macro_blocked_entry_count = max(
            0,
            len(entries_without_macro) - len(entries),
        )
        state.update(
            {
                "macro_blocked_entry_count": macro_blocked_entry_count,
                "macro_overlay": macro_overlay,
            }
        )
        state.update(
            {
                "status": "READY_NOT_SUBMITTED",
                "last_reason": (
                    "NATURAL_ENTRY_AVAILABLE"
                    if entries
                    else "MACRO_REGIME_BLOCKED_NATURAL_ENTRY"
                    if macro_blocked_entry_count
                    else "NO_FRESH_NATURAL_GENERATED_ENTRY"
                ),
                "ranked_natural_entries": entries,
                "health_failures": health_failures,
                "kill_switch_active": kill_switch.active,
                "weekly_trade_budget": weekly_status,
            }
        )
        _write_state(settings, state)
        return state

    import aiohttp

    from core.autonomous_trading import (
        bitvavo_entry_liquidity,
        bitvavo_public_price,
        notify_autonomous_event_safely,
    )

    markets = tuple(
        sorted(
            {
                market
                for row in authority.get("approved_candidates") or []
                for market in row.get("approved_markets") or []
            }
        )
    )
    positions = {
        str(dna): dict(position)
        for dna, position in dict(state.get("positions") or {}).items()
    }
    async with aiohttp.ClientSession() as session:
        client = build_live_client(
            settings,
            session=session,
            ledger_path=_paths(settings)["ledger"],
        )
        balances = await client.balances()
        baseline, baseline_failures = _load_baseline(settings, authority)
        if baseline_failures == [
            "POSITIVE_STRATEGY_INVENTORY_BASELINE_MISSING"
        ]:
            authority["inventory_baseline_status"] = "ACTIVE"
            authority["authority_hash"] = stable_hash(
                {
                    key: value
                    for key, value in authority.items()
                    if key != "authority_hash"
                },
                length=64,
            )
            atomic_write_json(_paths(settings)["authority"], authority)
            _write_baseline(
                settings,
                authority=authority,
                balances=balances,
            )
            state.update(
                {
                    "status": "BASELINE_CREATED",
                    "last_reason": "PREEXISTING_INVENTORY_ADOPTED_NO_TRADE",
                    "positions": positions,
                    "private_exchange_requests": 1,
                }
            )
            _write_state(settings, state)
            return state
        if baseline_failures:
            state.update(
                {
                    "status": "RECONCILIATION_BLOCKED",
                    "last_reason": "INVENTORY_BASELINE_INVALID",
                    "baseline_failures": baseline_failures,
                    "positions": positions,
                    "private_exchange_requests": 1,
                }
            )
            _write_state(settings, state)
            return state

        reconciliation = await client.reconcile(markets=markets)
        if not reconciliation.healthy:
            state.update(
                {
                    "status": "RECONCILIATION_BLOCKED",
                    "last_reason": "LIVE_RECONCILIATION_FAILED",
                    "reconciliation": {
                        "healthy": reconciliation.healthy,
                        "reason_codes": list(reconciliation.reason_codes),
                    },
                    "positions": positions,
                    "private_exchange_requests": 2 + len(markets),
                }
            )
            _write_state(settings, state)
            return state

        excess = _managed_excess(balances, baseline)
        for dna, position in list(positions.items()):
            status = str(position.get("status") or "OPEN")
            market = str(position.get("market") or "")
            base = market.split("-")[0]
            client_order_id = str(position.get("client_order_id") or "")
            if status in {
                "ENTRY_PENDING_RECONCILIATION",
                "EXIT_PENDING_RECONCILIATION",
            } and client_order_id:
                try:
                    order = await client.get_order(
                        market=market,
                        client_order_id=client_order_id,
                    )
                except (ExecutionBlocked, ReconciliationRequired):
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "last_reason": "PENDING_ORDER_RECONCILIATION_FAILED",
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
                order_status = _normalized_order_status(order)
                if status == "ENTRY_PENDING_RECONCILIATION":
                    quantity = _position_quantity_from_order(
                        order,
                        fallback=_decimal(position.get("quantity")),
                    )
                    open_partial_requires_protection = (
                        order_status == "partiallyfilled" and quantity > 0
                    )
                    if (
                        order_status
                        not in {"filled", "canceled", "cancelled", "rejected"}
                        and (
                            open_partial_requires_protection
                            or _gtc_entry_expired(
                                position,
                                observed_at=now,
                                validity_seconds=(
                                    settings.execution.live_limit_entry_validity_seconds
                                ),
                            )
                        )
                    ):
                        cancellation_preflight = LivePreflight.evaluate(
                            settings,
                            markets=markets,
                            strategy_status=ResearchStatus.PAPER_CANDIDATE,
                            data_healthy=not health_failures,
                            risk_manager_healthy=True,
                            exchange_healthy=True,
                            reconciliation_healthy=True,
                            kill_switch_active=kill_switch.active,
                            canary_exception_approved=True,
                            operator_canary_authorized=True,
                            portfolio_canary=True,
                            cap_limits={
                                "capital_level": CAPITAL_LEVEL,
                                "max_order_eur": str(MAXIMUM_ORDER_EUR),
                                "max_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
                                "max_positions": int(
                                    authority.get("maximum_open_positions")
                                    or MAXIMUM_OPEN_POSITIONS
                                ),
                                "max_new_orders_per_day": (
                                    MAXIMUM_NEW_ORDERS_PER_DAY
                                ),
                            },
                        )
                        if (
                            not cancellation_preflight.passed
                            or cancellation_preflight.capability is None
                            or not order.get("orderId")
                        ):
                            state.update(
                                {
                                    "status": "PREFLIGHT_BLOCKED",
                                    "last_reason": (
                                        "PARTIAL_FILL_PROTECTION_PREFLIGHT_BLOCKED"
                                        if open_partial_requires_protection
                                        else "GTC_CANCELLATION_PREFLIGHT_BLOCKED"
                                    ),
                                    "positions": positions,
                                }
                            )
                            _write_state(settings, state)
                            return state
                        try:
                            await client.cancel_order(
                                market=market,
                                order_id=str(order["orderId"]),
                                capability=cancellation_preflight.capability,
                            )
                            order = await client.get_order(
                                market=market,
                                client_order_id=client_order_id,
                            )
                        except (ExecutionBlocked, ReconciliationRequired):
                            state.update(
                                {
                                    "status": "RECONCILIATION_BLOCKED",
                                    "last_reason": (
                                        "GTC_CANCELLATION_STATE_AMBIGUOUS"
                                    ),
                                    "positions": positions,
                                }
                            )
                            _write_state(settings, state)
                            return state
                        quantity = _position_quantity_from_order(
                            order,
                            fallback=Decimal("0"),
                        )
                        if quantity > 0:
                            fill_price = _position_price_from_order(
                                order,
                                quantity=quantity,
                                fallback=_decimal(position.get("entry_price")),
                            )
                            client.record_final_fill(
                                order,
                                fallback_market=market,
                                fallback_side=OrderSide.BUY,
                                fallback_quantity=quantity,
                                fallback_price=fill_price,
                                allow_terminal_partial=True,
                            )
                            position.update(
                                {
                                    "status": "OPEN",
                                    "quantity": str(quantity),
                                    "entry_price": str(fill_price),
                                    "order_status": order.get("status"),
                                    "partial_fill_final": True,
                                    "last_reconciled_at": utc_iso(),
                                }
                            )
                            state["fills_verified_this_cycle"] = int(
                                state.get("fills_verified_this_cycle") or 0
                            ) + 1
                            state["orders_cancelled_this_cycle"] = int(
                                state.get("orders_cancelled_this_cycle") or 0
                            ) + 1
                            notify_autonomous_event_safely(
                                settings,
                                "ORDER_PARTIALLY_FILLED",
                                {
                                    **_live_order_notification_payload(
                                        order,
                                        market=market,
                                        side="BUY",
                                        order_type="LIMIT",
                                        requested_quantity=_decimal(
                                            position.get("requested_quantity")
                                            or position.get("quantity")
                                        ),
                                        fallback_price=fill_price,
                                        strategy_id=str(
                                            position.get("strategy_id") or ""
                                        ),
                                        timeframe=str(
                                            position.get("timeframe") or ""
                                        ),
                                    ),
                                    "verification_source": (
                                        "BITVAVO_REST_RECONCILIATION"
                                    ),
                                },
                            )
                            positions[dna] = position
                            continue

                        reprice_count = int(
                            position.get("entry_reprice_count") or 0
                        )
                        window_active, _, _ = signal_execution_window(
                            signal_timestamp=str(
                                position.get("signal_timestamp") or ""
                            ),
                            timeframe=str(position.get("timeframe") or ""),
                            observed_at=now,
                        )
                        if (
                            not window_active
                            or reprice_count
                            >= settings.execution.live_limit_entry_max_reprices
                        ):
                            positions.pop(dna, None)
                            state["orders_cancelled_this_cycle"] = int(
                                state.get("orders_cancelled_this_cycle") or 0
                            ) + 1
                            notify_autonomous_event_safely(
                                settings,
                                "ORDER_CANCELLED",
                                {
                                    **_live_order_notification_payload(
                                        order,
                                        market=market,
                                        side="BUY",
                                        order_type="LIMIT",
                                        requested_quantity=_decimal(
                                            position.get("requested_quantity")
                                            or position.get("quantity")
                                        ),
                                        fallback_price=_decimal(
                                            position.get("entry_price")
                                        ),
                                        strategy_id=str(
                                            position.get("strategy_id") or ""
                                        ),
                                        timeframe=str(
                                            position.get("timeframe") or ""
                                        ),
                                    ),
                                    "verification_source": (
                                        "BITVAVO_REST_RECONCILIATION"
                                    ),
                                },
                            )
                            continue

                        public_price = await bitvavo_public_price(session, market)
                        original_entry_notional = (
                            _replacement_entry_notional(position)
                        )
                        if original_entry_notional <= 0:
                            state.update(
                                {
                                    "status": "RECONCILIATION_BLOCKED",
                                    "last_reason": (
                                        "GTC_REPRICE_ORIGINAL_NOTIONAL_INVALID"
                                    ),
                                    "positions": positions,
                                }
                            )
                            _write_state(settings, state)
                            return state
                        liquidity = await bitvavo_entry_liquidity(
                            session,
                            market=market,
                            requested_notional_eur=original_entry_notional,
                            settings=settings,
                        )
                        if liquidity.get("status") != "PASSED":
                            positions.pop(dna, None)
                            continue
                        try:
                            replacement_plan = await _plan_live_entry_order(
                                settings,
                                client=client,
                                market=market,
                                requested_notional_eur=original_entry_notional,
                                public_price=public_price,
                                liquidity=liquidity,
                            )
                        except ExecutionBlocked:
                            positions.pop(dna, None)
                            continue
                        replacement_identity = stable_hash(
                            [
                                "GENERATED_LIVE_ENTRY_REPRICE",
                                dna,
                                position.get("signal_id"),
                                market,
                                reprice_count + 1,
                            ],
                            length=40,
                        )
                        replacement_intent = OrderIntent(
                            intent_id=replacement_identity[:32],
                            idempotency_key=(
                                f"generated-live-entry-reprice:{replacement_identity}"
                            ),
                            market=market,
                            side=OrderSide.BUY,
                            order_type=replacement_plan.order_type,
                            quantity=replacement_plan.quantity,
                            limit_price=replacement_plan.limit_price,
                            time_in_force=replacement_plan.time_in_force,
                            strategy_id=str(position.get("strategy_id") or ""),
                            strategy_dna_hash=dna,
                            signal_id=str(position.get("signal_id") or ""),
                            portfolio_decision_id=replacement_identity,
                            maximum_notional_eur=original_entry_notional,
                            reason_codes=(
                                "EXACT_POSITIVE_FROZEN_DNA",
                                "GTC_LIMIT_REPRICE",
                            ),
                        )
                        replacement_candidate = dict(
                            candidate_by_dna.get(dna) or {}
                        )
                        replacement_metrics = dict(
                            replacement_candidate.get("metrics")
                            or position.get("candidate_metrics")
                            or {}
                        )
                        replacement_trade_count = max(
                            0,
                            int(replacement_metrics.get("trade_count") or 0),
                        )
                        replacement_edge = (
                            _decimal(replacement_metrics.get("net_return"))
                            / Decimal(max(1, replacement_trade_count))
                        )
                        replacement_price = max(
                            public_price,
                            replacement_plan.limit_price or public_price,
                        )
                        prior_entry_price = _decimal(
                            position.get("entry_price")
                        )
                        replacement_target_distance = max(
                            Decimal("0"),
                            _decimal(position.get("take_profit_1"))
                            - prior_entry_price,
                        )
                        if replacement_edge <= 0:
                            replacement_edge = planned_target_net_edge(
                                entry_price=replacement_price,
                                target_price=(
                                    replacement_price
                                    + replacement_target_distance
                                ),
                                costs=CanonicalCostModel.from_settings(
                                    settings
                                ),
                            )
                        replacement_risk = _decimal(
                            position.get("planned_risk_eur")
                        )
                        if replacement_risk <= 0:
                            replacement_risk = (
                                replacement_plan.quantity
                                * max(
                                    Decimal("0"),
                                    prior_entry_price
                                    - _decimal(position.get("stop_loss")),
                                )
                            )
                        replacement_equity = _decimal(
                            (
                                (health.get("account") or {}).get(
                                    "portfolio_valuation"
                                )
                                or {}
                            ).get("estimated_total_equity_eur")
                        )
                        try:
                            replacement_canonical = (
                                canonicalize_approved_buy_order(
                                    settings,
                                    replacement_intent,
                                    mark_price=replacement_price,
                                    current_quantity=excess.get(
                                        market.split("-")[0],
                                        Decimal("0"),
                                    ),
                                    equity_eur=replacement_equity,
                                    approved_risk_eur=replacement_risk,
                                    expected_net_edge=replacement_edge,
                                    confidence=max(
                                        Decimal("0.01"),
                                        min(
                                            Decimal("1"),
                                            Decimal(replacement_trade_count)
                                            / Decimal(
                                                replacement_trade_count + 50
                                            )
                                            if replacement_trade_count > 0
                                            else Decimal("0.5"),
                                        ),
                                    ),
                                    family=str(
                                        replacement_candidate.get(
                                            "economic_hypothesis_family"
                                        )
                                        or replacement_candidate.get(
                                            "family"
                                        )
                                        or position.get("strategy_id")
                                        or "GENERATED_STRATEGY"
                                    ),
                                    evidence_id=str(
                                        position.get("frozen_candidate_hash")
                                        or position.get("signal_id")
                                    ),
                                    policy_version=str(
                                        authority.get("authority_hash")
                                        or "positive_strategy_live_authority_v1"
                                    ),
                                    account_state={
                                        "health_status": health.get("status"),
                                        "equity_eur": str(
                                            replacement_equity
                                        ),
                                        "entry_allowed": health.get(
                                            "entry_allowed"
                                        ),
                                    },
                                    portfolio_state={
                                        "generated_positions": positions,
                                        "replacing_dna": dna,
                                        "reprice_count": reprice_count + 1,
                                    },
                                    horizon_seconds=max(
                                        60,
                                        int(
                                            TIMEFRAME_SECONDS.get(
                                                str(
                                                    position.get("timeframe")
                                                    or ""
                                                ),
                                                60,
                                            )
                                        ),
                                    ),
                                )
                            )
                        except ExecutionBlocked:
                            positions.pop(dna, None)
                            continue
                        replacement_intent = replacement_canonical.order
                        async def submit_reserved_replacement(
                            fresh_portfolio: Mapping[str, Any],
                        ) -> dict[str, Any]:
                            return await client.submit_order(
                                replacement_intent,
                                capability=cancellation_preflight.capability,
                                estimated_price=public_price,
                                reconciled_owned_quantity=excess.get(
                                    market.split("-")[0], Decimal("0")
                                ),
                                reconciled_total_exposure_eur=_decimal(
                                    fresh_portfolio[
                                        "capacity_managed_exposure_eur"
                                    ]
                                ),
                                reconciled_open_positions=int(
                                    fresh_portfolio[
                                        "capacity_managed_position_count"
                                    ]
                                ),
                                exchange_minimum_order_eur=Decimal("5"),
                                canonical_chain=(
                                    replacement_canonical.chain
                                ),
                            )

                        try:
                            (
                                replacement_approved,
                                replacement_reason,
                                replacement_portfolio,
                                replacement,
                            ) = await submit_level_2_buy_atomically(
                                settings,
                                requested_notional_eur=(
                                    replacement_plan.quantity
                                    * max(
                                        public_price,
                                        replacement_plan.limit_price
                                        or public_price,
                                    )
                                ),
                                submit_order=submit_reserved_replacement,
                                replacing_source="GENERATED_DNA",
                                replacing_identity=dna,
                            )
                            if (
                                not replacement_approved
                                or replacement is None
                            ):
                                state.update(
                                    {
                                        "status": "PORTFOLIO_CAP_BLOCKED",
                                        "last_reason": replacement_reason,
                                        "managed_portfolio": (
                                            replacement_portfolio
                                        ),
                                        "positions": positions,
                                    }
                                )
                                _write_state(settings, state)
                                return state
                        except ReconciliationRequired:
                            state.update(
                                {
                                    "status": "RECONCILIATION_BLOCKED",
                                    "last_reason": (
                                        "GTC_REPRICE_SUBMISSION_AMBIGUOUS"
                                    ),
                                    "positions": positions,
                                }
                            )
                            _write_state(settings, state)
                            return state
                        except ExecutionBlocked:
                            positions.pop(dna, None)
                            continue
                        prior_entry = _decimal(position.get("entry_price"))
                        stop_distance = max(
                            Decimal("0"),
                            prior_entry - _decimal(position.get("stop_loss")),
                        )
                        target_1_distance = max(
                            Decimal("0"),
                            _decimal(position.get("take_profit_1")) - prior_entry,
                        )
                        target_2_distance = max(
                            Decimal("0"),
                            _decimal(position.get("take_profit_2")) - prior_entry,
                        )
                        position.update(
                            {
                                "client_order_id": str(
                                    replacement.get("clientOrderId")
                                    or client.client_order_id_for(
                                        replacement_intent.idempotency_key
                                    )
                                ),
                                "order_status": replacement.get("status"),
                                "entry_order_submitted_at": now.isoformat(),
                                "entry_reprice_count": reprice_count + 1,
                                "requested_quantity": str(
                                    replacement_plan.quantity
                                ),
                                "quantity": str(replacement_plan.quantity),
                                "entry_price": str(public_price),
                                "stop_loss": str(public_price - stop_distance),
                                "take_profit_1": str(
                                    public_price + target_1_distance
                                ),
                                "take_profit_2": str(
                                    public_price + target_2_distance
                                ),
                                "limit_price": (
                                    str(replacement_plan.limit_price)
                                    if replacement_plan.limit_price is not None
                                    else None
                                ),
                            }
                        )
                        positions[dna] = position
                        notify_autonomous_event_safely(
                            settings,
                            "ORDER_SUBMITTING",
                            _live_order_notification_payload(
                                replacement,
                                market=market,
                                side="BUY",
                                order_type=replacement_plan.order_type.value,
                                requested_quantity=replacement_plan.quantity,
                                fallback_price=public_price,
                                strategy_id=str(
                                    position.get("strategy_id") or ""
                                ),
                                timeframe=str(position.get("timeframe") or ""),
                            ),
                        )
                        state.update(
                            {
                                "status": "ENTRY_REPRICED",
                                "last_reason": "GTC_LIMIT_REPRICED_ON_VALID_SETUP",
                                "positions": positions,
                                "orders_generated_this_cycle": 1,
                                "orders_submitted_this_cycle": 1,
                                "orders_cancelled_this_cycle": int(
                                    state.get("orders_cancelled_this_cycle")
                                    or 0
                                )
                                + 1,
                                "orders_repriced_this_cycle": 1,
                                "orders_generated": int(
                                    state.get("orders_generated") or 0
                                )
                                + 1,
                                "orders_submitted": int(
                                    state.get("orders_submitted") or 0
                                )
                                + 1,
                            }
                        )
                        _write_state(settings, state)
                        return state
                    terminal_partial = (
                        str(position.get("time_in_force") or "")
                        in {"IOC", "FOK"}
                        and order_status
                        in {"partiallyfilled", "canceled", "cancelled"}
                        and quantity > 0
                    )
                    if order_status == "filled" or terminal_partial:
                        fill_price = _position_price_from_order(
                            order,
                            quantity=quantity,
                            fallback=_decimal(position.get("entry_price")),
                        )
                        client.record_final_fill(
                            order,
                            fallback_market=market,
                            fallback_side=OrderSide.BUY,
                            fallback_quantity=quantity,
                            fallback_price=fill_price,
                            allow_terminal_partial=terminal_partial,
                        )
                        position.update(
                            {
                                "status": "OPEN",
                                "quantity": str(quantity),
                                "entry_price": str(fill_price),
                                "order_status": order.get("status"),
                                "partial_fill_final": terminal_partial,
                                "last_reconciled_at": utc_iso(),
                            }
                        )
                        state["fills_verified_this_cycle"] = int(
                            state.get("fills_verified_this_cycle") or 0
                        ) + 1
                        notify_autonomous_event_safely(
                            settings,
                            "ORDER_FILLED"
                            if order_status == "filled"
                            else "ORDER_PARTIALLY_FILLED",
                            {
                                **_live_order_notification_payload(
                                    order,
                                    market=market,
                                    side="BUY",
                                    order_type=str(
                                        position.get("order_type") or "LIMIT"
                                    ),
                                    requested_quantity=_decimal(
                                        position.get("quantity")
                                    ),
                                    fallback_price=fill_price,
                                    strategy_id=str(
                                        position.get("strategy_id") or ""
                                    ),
                                    timeframe=str(
                                        position.get("timeframe") or ""
                                    ),
                                ),
                                "verification_source": (
                                    "BITVAVO_REST_RECONCILIATION"
                                ),
                            },
                        )
                    elif order_status in {"canceled", "cancelled", "rejected"}:
                        positions.pop(dna, None)
                        continue
                elif status == "EXIT_PENDING_RECONCILIATION":
                    if order_status == "filled" and excess.get(
                        base, Decimal("0")
                    ) < Decimal("0.00000001"):
                        cooldowns.record_exit(
                            strategy_id=str(
                                position.get("strategy_id") or ""
                            ),
                            strategy_dna_hash=dna,
                            market=market,
                            timeframe=str(position.get("timeframe") or ""),
                            reason=str(
                                position.get("exit_reason")
                                or "STRATEGY_EXIT"
                            ),
                            observed_at=now,
                        )
                        state["last_closed_position"] = {
                            **position,
                            "status": "CLOSED",
                            "closed_at": utc_iso(),
                            "exit_order": {
                                "status": order.get("status"),
                                "order_id_masked": stable_hash(
                                    str(order.get("orderId") or ""),
                                    length=12,
                                ),
                            },
                        }
                        state["fills_verified_this_cycle"] = int(
                            state.get("fills_verified_this_cycle") or 0
                        ) + 1
                        positions.pop(dna, None)
                        continue
                positions[dna] = position

        # Unknown strategy-created inventory is always fail-closed.  Dust below
        # one euro is ignored, but never sold or adopted.
        managed_bases = {
            str(position.get("market") or "").split("-")[0]
            for position in positions.values()
        }
        unknown_excess: dict[str, str] = {}
        for symbol, quantity in excess.items():
            if symbol in managed_bases:
                continue
            market = f"{symbol}-EUR"
            if market not in markets:
                continue
            try:
                value = quantity * await bitvavo_public_price(session, market)
            except RuntimeError:
                value = MINIMUM_MATERIAL_POSITION_EUR
            if value >= Decimal("1"):
                unknown_excess[symbol] = str(quantity)
        if unknown_excess:
            # Unknown/manual inventory blocks every new BUY, but it must not
            # pre-empt reconciliation, native-stop repair or a risk-reducing
            # exit for a separately identified canonical managed position.
            state["unknown_excess"] = unknown_excess

        # Every confirmed generated-strategy fill must have a durable
        # venue-native stop.  This recovery path also protects positions that
        # were opened by an older runtime before native stops were supported.
        for dna, position in sorted(positions.items()):
            if str(position.get("status") or "OPEN") != "OPEN":
                continue
            market = str(position.get("market") or "")
            owned = excess.get(market.split("-")[0], Decimal("0"))
            expected = _decimal(position.get("quantity"))
            protective_client_id = str(
                position.get("protective_stop_client_order_id") or ""
            )
            if protective_client_id:
                # Query the venue even when wallet inventory is now zero.  A
                # native stop that filled between private-stream events removes
                # the asset before local state is closed; skipping here leaves
                # a ghost position and misclassifies the sale proceeds as an
                # unexplained EUR balance change.
                if expected <= 0:
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "last_reason": "PROTECTIVE_STOP_INVALID_EXPECTED_QUANTITY",
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
                try:
                    protective = await client.get_order(
                        market=market,
                        client_order_id=protective_client_id,
                    )
                except (ExecutionBlocked, ReconciliationRequired):
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "last_reason": "PROTECTIVE_STOP_RECONCILIATION_FAILED",
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
                protective_status = _normalized_order_status(protective)
                if protective_status == "filled":
                    client.record_final_fill(
                        protective,
                        fallback_market=market,
                        fallback_side=OrderSide.SELL,
                        fallback_quantity=expected,
                        fallback_price=_decimal(position.get("stop_loss")),
                    )
                    cooldowns.record_exit(
                        strategy_id=str(position.get("strategy_id") or ""),
                        strategy_dna_hash=dna,
                        market=market,
                        timeframe=str(position.get("timeframe") or ""),
                        reason="NATIVE_PROTECTIVE_STOP_FILLED",
                        observed_at=now,
                    )
                    state["last_closed_position"] = {
                        **position,
                        "status": "CLOSED",
                        "closed_at": utc_iso(),
                        "exit_reason": "NATIVE_PROTECTIVE_STOP_FILLED",
                        "protective_stop_status": str(
                            protective.get("status") or "filled"
                        ),
                        "native_protective_stop_active": False,
                        "exit_order": {
                            "status": protective.get("status"),
                            "order_id_masked": stable_hash(
                                str(protective.get("orderId") or ""),
                                length=12,
                            ),
                        },
                    }
                    positions.pop(dna, None)
                    state["fills_verified_this_cycle"] = int(
                        state.get("fills_verified_this_cycle") or 0
                    ) + 1
                    state.update(
                        {
                            "status": "POSITION_CLOSED",
                            "last_reason": "NATIVE_PROTECTIVE_STOP_FILLED",
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
                if protective_status in {"new", "awaitingtrigger"}:
                    if owned <= 0:
                        state.update(
                            {
                                "status": "RECONCILIATION_BLOCKED",
                                "last_reason": (
                                    "PROTECTIVE_STOP_OPEN_BUT_MANAGED_INVENTORY_MISSING"
                                ),
                                "positions": positions,
                            }
                        )
                        _write_state(settings, state)
                        return state
                    position["protective_stop_status"] = str(
                        protective.get("status") or ""
                    )
                    positions[dna] = position
                    continue
                if protective_status == "partiallyfilled":
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "last_reason": "PROTECTIVE_STOP_PARTIAL_FILL",
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
                for key in (
                    "protective_stop_order_id",
                    "protective_stop_client_order_id",
                    "protective_stop_status",
                    "protective_stop_trigger",
                    "native_protective_stop_active",
                ):
                    position.pop(key, None)

            if owned <= 0 or expected <= 0:
                continue

            preflight = LivePreflight.evaluate(
                settings,
                markets=markets,
                strategy_status=ResearchStatus.PAPER_CANDIDATE,
                data_healthy=True,
                risk_manager_healthy=True,
                exchange_healthy=True,
                reconciliation_healthy=True,
                kill_switch_active=False,
                canary_exception_approved=True,
                operator_canary_authorized=True,
                portfolio_canary=True,
                cap_limits={
                    "capital_level": CAPITAL_LEVEL,
                    "max_order_eur": str(MAXIMUM_ORDER_EUR),
                    "max_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
                    "max_positions": MAXIMUM_OPEN_POSITIONS,
                    "max_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
                },
            )
            if not preflight.passed or preflight.capability is None:
                state.update(
                    {
                        "status": "PREFLIGHT_BLOCKED",
                        "last_reason": "PROTECTIVE_STOP_PREFLIGHT_BLOCKED",
                        "preflight_failures": list(preflight.failures),
                        "positions": positions,
                    }
                )
                _write_state(settings, state)
                return state
            public_price = await bitvavo_public_price(session, market)
            try:
                protective_result = await _place_generated_native_stop(
                    client,
                    capability=preflight.capability,
                    position=position,
                    quantity=min(expected, owned),
                    estimated_price=public_price,
                )
            except (ExecutionBlocked, ReconciliationRequired):
                state.update(
                    {
                        "status": "RECONCILIATION_BLOCKED",
                        "last_reason": "NATIVE_PROTECTIVE_STOP_NOT_CONFIRMED",
                        "positions": positions,
                        "local_hard_stop_active": True,
                    }
                )
                _write_state(settings, state)
                return state
            protective_order = dict(
                protective_result.pop("native_protective_stop_order")
            )
            position.update(protective_result)
            positions[dna] = position
            notify_autonomous_event_safely(
                settings,
                "ORDER_SUBMITTING",
                {
                    **_live_order_notification_payload(
                        protective_order,
                        market=market,
                        side="SELL",
                        order_type="STOP_LOSS",
                        requested_quantity=min(expected, owned),
                        fallback_price=_decimal(position.get("stop_loss")),
                        strategy_id=str(position.get("strategy_id") or ""),
                        timeframe=str(position.get("timeframe") or ""),
                    ),
                    "reason_code": "NATIVE_PROTECTIVE_STOP_ACCEPTED",
                    "trigger_price": position.get("protective_stop_trigger"),
                },
            )
            state.update(
                {
                    "status": "POSITION_PROTECTED",
                    "last_reason": "NATIVE_PROTECTIVE_STOP_ACCEPTED",
                    "positions": positions,
                    "orders_generated": int(state.get("orders_generated") or 0)
                    + 1,
                    "orders_submitted": int(state.get("orders_submitted") or 0)
                    + 1,
                    "orders_generated_this_cycle": 1,
                    "orders_submitted_this_cycle": 1,
                }
            )
            _write_state(settings, state)
            return state

        # Exit management has priority and remains available while entries are
        # paused.  A kill switch may force risk reduction but never a buy.
        for dna, position in sorted(positions.items()):
            if str(position.get("status") or "OPEN") != "OPEN":
                continue
            market = str(position["market"])
            public_price = await bitvavo_public_price(session, market)
            candidate_eval = next(
                (
                    row
                    for row in dict(evaluations.get(dna) or {}).get(
                        "markets"
                    )
                    or []
                    if row.get("market") == market
                ),
                {},
            )
            stop = _decimal(position.get("stop_loss"))
            target_1 = _decimal(position.get("take_profit_1"))
            target_2 = _decimal(position.get("take_profit_2"))
            exit_reason: str | None = None
            if kill_switch.active:
                exit_reason = "KILL_SWITCH_RISK_EXIT"
            elif stop > 0 and public_price <= stop:
                exit_reason = "STOP_LOSS_REACHED"
            elif target_2 > 0 and public_price >= target_2:
                exit_reason = "TAKE_PROFIT_2_REACHED"
            elif (
                candidate_eval.get("exit") is True
                and candidate_eval.get("stale") is not True
            ):
                exit_reason = "STRATEGY_EXIT"
            elif (
                target_1 > 0
                and public_price >= target_1
                and not position.get("tp1_reached")
            ):
                position["tp1_reached"] = True
                position["tp1_reached_at"] = utc_iso()
                positions[dna] = position
                notify_autonomous_event_safely(
                    settings,
                    "TP1_REACHED",
                    {
                        "market": market,
                        "strategy_id": position.get("strategy_id"),
                        "price": float(public_price),
                        "status": "POSITION_HELD_BECAUSE_PARTIAL_IS_BELOW_MINIMUM",
                    },
                )
                continue
            if exit_reason is None:
                continue
            owned = excess.get(market.split("-")[0], Decimal("0"))
            expected = _decimal(position.get("quantity"))
            if owned <= 0 or (
                expected > 0 and owned < expected * Decimal("0.995")
            ):
                state.update(
                    {
                        "status": "RECONCILIATION_BLOCKED",
                        "last_reason": "MANAGED_POSITION_QUANTITY_MISMATCH",
                        "positions": positions,
                    }
                )
                _write_state(settings, state)
                return state
            preflight = LivePreflight.evaluate(
                settings,
                markets=markets,
                strategy_status=ResearchStatus.PAPER_CANDIDATE,
                data_healthy=True,
                risk_manager_healthy=True,
                exchange_healthy=True,
                reconciliation_healthy=True,
                kill_switch_active=False,
                canary_exception_approved=True,
                operator_canary_authorized=True,
                portfolio_canary=True,
                cap_limits={
                    "capital_level": CAPITAL_LEVEL,
                    "max_order_eur": str(MAXIMUM_ORDER_EUR),
                    "max_exposure_eur": str(
                        MAXIMUM_TOTAL_EXPOSURE_EUR
                    ),
                    "max_positions": MAXIMUM_OPEN_POSITIONS,
                    "max_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
                },
            )
            if not preflight.passed or preflight.capability is None:
                state.update(
                    {
                        "status": "PREFLIGHT_BLOCKED",
                        "last_reason": "EXIT_PREFLIGHT_BLOCKED",
                        "preflight_failures": list(preflight.failures),
                        "positions": positions,
                    }
                )
                _write_state(settings, state)
                return state
            protective_client_id = str(
                position.get("protective_stop_client_order_id") or ""
            )
            if protective_client_id:
                try:
                    protective = await client.get_order(
                        market=market,
                        client_order_id=protective_client_id,
                    )
                    protective_status = _normalized_order_status(protective)
                    if protective_status == "filled":
                        state.update(
                            {
                                "status": "RECONCILIATION_BLOCKED",
                                "last_reason": (
                                    "PROTECTIVE_STOP_FILLED_DURING_EXIT"
                                ),
                                "positions": positions,
                            }
                        )
                        _write_state(settings, state)
                        return state
                    if protective_status in {
                        "new",
                        "awaitingtrigger",
                        "partiallyfilled",
                    }:
                        await client.cancel_order(
                            market=market,
                            order_id=str(protective["orderId"]),
                            capability=preflight.capability,
                        )
                    position["protective_stop_status"] = "CANCELLED_FOR_EXIT"
                    position["native_protective_stop_active"] = False
                except (ExecutionBlocked, ReconciliationRequired):
                    state.update(
                        {
                            "status": "RECONCILIATION_BLOCKED",
                            "last_reason": (
                                "PROTECTIVE_STOP_CANCELLATION_AMBIGUOUS"
                            ),
                            "positions": positions,
                        }
                    )
                    _write_state(settings, state)
                    return state
            identity = stable_hash(
                [
                    "GENERATED_LIVE_EXIT",
                    dna,
                    position.get("signal_id"),
                    exit_reason,
                ],
                length=40,
            )
            intent = OrderIntent(
                intent_id=identity[:32],
                idempotency_key=f"generated-live-exit:{identity}",
                market=market,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=owned,
                strategy_id=str(position.get("strategy_id") or ""),
                strategy_dna_hash=dna,
                signal_id=str(position.get("signal_id") or ""),
                portfolio_decision_id=identity,
                reason_codes=(exit_reason,),
            )
            try:
                order = await client.submit_order(
                    intent,
                    capability=preflight.capability,
                    estimated_price=public_price,
                    reconciled_owned_quantity=owned,
                    reconciled_total_exposure_eur=None,
                    reconciled_open_positions=len(positions),
                    exchange_minimum_order_eur=Decimal("5"),
                )
            except ReconciliationRequired:
                order = {
                    "clientOrderId": client.client_order_id_for(
                        intent.idempotency_key
                    ),
                    "status": "ambiguous",
                }
            except ExecutionBlocked:
                state.update(
                    {
                        "status": "ORDER_REJECTED",
                        "last_reason": "GENERATED_LIVE_EXIT_REJECTED",
                        "positions": positions,
                        "orders_generated_this_cycle": 1,
                    }
                )
                _write_state(settings, state)
                return state
            position.update(
                {
                    "status": "EXIT_PENDING_RECONCILIATION",
                    "exit_reason": exit_reason,
                    "exit_submitted_at": utc_iso(),
                    "client_order_id": str(
                        order.get("clientOrderId")
                        or client.client_order_id_for(
                            intent.idempotency_key
                        )
                    ),
                    "exit_price_reference": str(public_price),
                }
            )
            positions[dna] = position
            notify_autonomous_event_safely(
                settings,
                "ORDER_FILLED"
                if _normalized_order_status(order) == "filled"
                else "ORDER_SUBMITTING",
                {
                    **_live_order_notification_payload(
                        order,
                        market=market,
                        side="SELL",
                        order_type="MARKET",
                        requested_quantity=owned,
                        fallback_price=public_price,
                        strategy_id=str(position.get("strategy_id") or ""),
                        timeframe=str(position.get("timeframe") or ""),
                    ),
                    "reason_code": exit_reason,
                },
            )
            state.update(
                {
                    "status": "EXIT_SUBMITTED",
                    "last_reason": exit_reason,
                    "positions": positions,
                    "orders_generated": int(
                        state.get("orders_generated") or 0
                    )
                    + 1,
                    "orders_submitted": int(
                        state.get("orders_submitted") or 0
                    )
                    + 1,
                    "orders_generated_this_cycle": 1,
                    "orders_submitted_this_cycle": 1,
                }
            )
            state["weekly_trade_budget"] = weekly_budget.record_exit(
                exit_identity=identity,
                market=market,
                strategy_id=str(position.get("strategy_id") or ""),
                reason=exit_reason,
                observed_at=now,
            )
            _write_state(settings, state)
            return state

        if (
            not allow_new_entry
            or not active
            or not economics_live_dna
            or kill_switch.active
            or health_failures
            or unknown_excess
        ):
            current_entry_blockers = (
                list(health.get("entry_blockers") or [])
                if health_failures
                else []
            )
            if not active:
                current_entry_blockers.extend(authority_failures)
            if not economics_live_dna:
                current_entry_blockers.append(
                    "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
                )
            if unknown_excess:
                current_entry_blockers.append(
                    "UNKNOWN_GENERATED_STRATEGY_INVENTORY"
                )
            state.update(
                {
                    "status": (
                        "AUTHORITY_BLOCKED" if not active else "ENTRY_BLOCKED"
                    ),
                    "last_reason": (
                        "POSITIVE_STRATEGY_AUTHORITY_BLOCKED"
                        if not active
                        else "CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"
                        if not economics_live_dna
                        else "KILL_SWITCH_ACTIVE"
                        if kill_switch.active
                        else "UNKNOWN_GENERATED_STRATEGY_INVENTORY"
                        if unknown_excess
                        else "LIVE_ACCOUNT_HEALTH_BLOCKED"
                        if health_failures
                        else "NEW_ENTRIES_PAUSED"
                    ),
                    "health_failures": health_failures,
                    "entry_blockers": current_entry_blockers,
                    "unknown_excess": unknown_excess,
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state

        material_count, material_markets = _material_wallet_positions(health)
        shared_managed = managed_live_portfolio(settings)
        managed_count = int(shared_managed["managed_position_count"])
        current_equity = _decimal(
            ((health.get("account") or {}).get("portfolio_valuation") or {}).get(
                "estimated_total_equity_eur"
            )
        )
        effective_maximum_positions = min(
            int(authority.get("maximum_open_positions") or MAXIMUM_OPEN_POSITIONS),
            MAXIMUM_OPEN_POSITIONS,
        )
        position_limit_status = write_position_limit_status(
            settings,
            account_equity_eur=current_equity,
            material_positions=material_markets,
            managed_positions=[
                str(position.get("market") or "")
                for position in shared_managed.get("positions") or []
            ],
            maximum_managed_positions=effective_maximum_positions,
        )
        occupied_markets = {
            str(position.get("market") or "")
            for position in positions.values()
        } | {
            str(position.get("market") or "")
            for position in shared_managed.get("positions") or []
        }
        occupied_dna = set(positions)
        ranking_inputs = {
            "candidates": candidates,
            "evaluations": evaluations,
            "authority": authority,
            "observed_at": now,
            "occupied_markets": sorted(occupied_markets),
            "occupied_dna": sorted(occupied_dna),
            "degradation": degradation,
        }
        entries_without_macro = rank_natural_entries(**ranking_inputs)
        entries = rank_natural_entries(
            **ranking_inputs,
            macro_context=macro_overlay,
        )
        entries = [
            row
            for row in entries
            if str(row.get("strategy_dna_hash") or "").lower()
            in economics_live_dna
        ]
        macro_blocked_entry_count = max(
            0,
            len(entries_without_macro) - len(entries),
        )
        if not entries:
            state.update(
                {
                    "status": "READY",
                    "last_reason": (
                        "MACRO_REGIME_BLOCKED_NATURAL_ENTRY"
                        if macro_blocked_entry_count
                        else "NO_FRESH_NATURAL_GENERATED_ENTRY"
                    ),
                    "positions": positions,
                    "material_wallet_position_count": material_count,
                    "material_wallet_markets": material_markets,
                    "ranked_natural_entries": [],
                    "entry_blockers": [],
                    "macro_blocked_entry_count": (
                        macro_blocked_entry_count
                    ),
                    "macro_overlay": macro_overlay,
                    "weekly_trade_budget": weekly_status,
                    "position_limit_status": position_limit_status,
                }
            )
            _write_state(settings, state)
            return state
        if managed_count >= effective_maximum_positions:
            state.update(
                {
                    "status": "PORTFOLIO_CAP_BLOCKED",
                    "last_reason": "MANAGED_POSITION_LIMIT_REACHED",
                    "positions": positions,
                    "managed_position_count": managed_count,
                    "managed_portfolio": shared_managed,
                    "material_wallet_position_count": material_count,
                    "material_wallet_markets": material_markets,
                    "ranked_natural_entries": entries,
                    "weekly_trade_budget": weekly_status,
                    "position_limit_status": position_limit_status,
                }
            )
            _write_state(settings, state)
            return state

        selected = entries[0]
        cooldown_gate = cooldowns.assess_entry(
            strategy_id=str(selected["strategy_id"]),
            strategy_dna_hash=str(selected["strategy_dna_hash"]),
            market=str(selected["market"]),
            timeframe=str(selected["timeframe"]),
            signal_candle_at=str(selected["signal_timestamp"]),
            observed_at=now,
        )
        if cooldown_gate.get("approved") is not True:
            state.update(
                {
                    "status": "COOLDOWN_BLOCKED",
                    "last_reason": cooldown_gate.get("reason_code"),
                    "cooldown": cooldown_gate,
                    "positions": positions,
                    "ranked_natural_entries": entries,
                    "weekly_trade_budget": weekly_budget.record_rejection(
                        strategy_id=str(selected["strategy_id"]),
                        market=str(selected["market"]),
                        timeframe=str(selected["timeframe"]),
                        reason_code=str(
                            cooldown_gate.get("reason_code")
                            or "COOLDOWN_BLOCKED"
                        ),
                        signal_id=str(selected.get("signal_id") or ""),
                        observed_at=now,
                    ),
                    "position_limit_status": position_limit_status,
                }
            )
            _write_state(settings, state)
            return state
        entry_blockers = list(health.get("entry_blockers") or [])
        if health.get("entry_allowed") is not True or entry_blockers:
            state.update(
                {
                    "status": "ENTRY_CAPACITY_BLOCKED",
                    "last_reason": (
                        entry_blockers[0]
                        if entry_blockers
                        else "LIVE_ACCOUNT_ENTRY_NOT_READY"
                    ),
                    "entry_blockers": entry_blockers,
                    "positions": positions,
                    "ranked_natural_entries": entries,
                    "weekly_trade_budget": weekly_status,
                    "position_limit_status": position_limit_status,
                }
            )
            _write_state(settings, state)
            return state
        weekly_gate = weekly_budget.assess_entry(
            score=selected.get("score") or 0,
            observed_at=now,
        )
        if weekly_gate.get("approved") is not True:
            state.update(
                {
                    "status": "WEEKLY_BUDGET_BLOCKED",
                    "last_reason": weekly_gate.get("reason_code"),
                    "positions": positions,
                    "ranked_natural_entries": entries,
                    "weekly_trade_budget": weekly_gate.get("status"),
                    "position_limit_status": position_limit_status,
                }
            )
            _write_state(settings, state)
            return state
        candidate = candidate_by_dna[selected["strategy_dna_hash"]]
        market = str(selected["market"])
        public_price = await bitvavo_public_price(session, market)
        (
            stop_distance,
            target_distance,
            risk_level_source,
        ) = _resolve_live_risk_distances(selected)
        state["selected_risk_level_source"] = risk_level_source
        if stop_distance <= 0 or target_distance <= 0:
            state.update(
                {
                    "status": "SIGNAL_BLOCKED",
                    "last_reason": "GENERATED_SIGNAL_RISK_LEVELS_INVALID",
                    "positions": positions,
                    "ranked_natural_entries": entries,
                }
            )
            _write_state(settings, state)
            return state
        available_eur = next(
            (
                _decimal(row.get("available"))
                for row in balances
                if str(row.get("symbol") or "").upper() == "EUR"
            ),
            Decimal("0"),
        )
        requested_notional, sizing = _dynamic_entry_notional(
            account_equity_eur=current_equity,
            available_eur=available_eur,
            selected_entry=selected,
            authority_cap_eur=MAXIMUM_ORDER_EUR,
        )
        state["position_sizing"] = sizing
        if requested_notional <= 0:
            state.update(
                {
                    "status": "RISK_BLOCKED",
                    "last_reason": "DYNAMIC_POSITION_SIZE_BELOW_VENUE_MINIMUM",
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        liquidity = await bitvavo_entry_liquidity(
            session,
            market=market,
            requested_notional_eur=requested_notional,
            settings=settings,
        )
        if liquidity.get("status") != "PASSED":
            state.update(
                {
                    "status": "LIQUIDITY_BLOCKED",
                    "last_reason": "CURRENT_LIQUIDITY_NOT_EXECUTABLE",
                    "entry_liquidity": liquidity,
                    "positions": positions,
                    "ranked_natural_entries": entries,
                }
            )
            _write_state(settings, state)
            return state

        day_start_equity = _decimal(
            (health.get("daily_profit_target") or {}).get(
                "risk_adjusted_day_start_equity_eur"
            )
            or (health.get("daily_profit_target") or {}).get(
                "day_start_equity_eur"
            )
        ) or current_equity
        risk_state_path = (
            settings.paths.output_dir
            / "governance"
            / "live_canary_risk_state.json"
        )
        risk_state = (
            dict(read_json(risk_state_path))
            if risk_state_path.is_file()
            else {}
        )
        peak_equity = max(
            current_equity,
            _decimal(risk_state.get("peak_equity_eur")),
        )
        snapshot = PortfolioSnapshot(
            equity_eur=float(current_equity),
            cash_eur=float(available_eur),
            day_start_equity_eur=float(day_start_equity),
            peak_equity_eur=float(peak_equity),
            trades_today=0,
            reconciled=True,
        )
        stop_price = public_price - stop_distance
        risk = RiskManager.from_settings(
            settings,
            kill_switch_path=(
                settings.paths.checkpoints_dir / "kill_switch.json"
            ),
        ).assess_entry(
            market=market,
            entry_price=float(public_price),
            stop_price=float(stop_price),
            snapshot=snapshot,
            live_mode=True,
        )
        if not risk.approved:
            state.update(
                {
                    "status": "RISK_BLOCKED",
                    "last_reason": "CENTRAL_RISK_MANAGER_REJECTED",
                    "risk_reason_codes": [
                        getattr(value, "value", str(value))
                        for value in risk.reason_codes
                    ],
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        quantity = min(
            requested_notional / public_price,
            Decimal(str(risk.approved_quantity)),
        )
        requested_notional = quantity * public_price
        capacity_ok, capacity_reason, shared_managed = (
            capital_level_2_capacity(
                settings,
                requested_notional_eur=requested_notional,
            )
        )
        if not capacity_ok:
            state.update(
                {
                    "status": "PORTFOLIO_CAP_BLOCKED",
                    "last_reason": capacity_reason,
                    "managed_portfolio": shared_managed,
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        managed_exposure = _decimal(shared_managed["managed_exposure_eur"])
        policy = CanaryPolicy.from_cap_limits(
            settings,
            maximum_order_eur=MAXIMUM_ORDER_EUR,
            maximum_total_eur=MAXIMUM_TOTAL_EXPOSURE_EUR,
            maximum_open_positions=effective_maximum_positions,
            capital_level=CAPITAL_LEVEL,
            enabled=True,
        )
        guard = InstitutionalCanaryGuard(policy).assess_buy(
            requested_notional_eur=requested_notional,
            current_total_exposure_eur=managed_exposure,
            current_open_positions=managed_count,
            exchange_minimum_order_eur=Decimal("5"),
        )
        if not guard.approved:
            state.update(
                {
                    "status": "PORTFOLIO_CAP_BLOCKED",
                    "last_reason": guard.reason_code,
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        quantity = guard.approved_notional_eur / public_price
        planned_risk = quantity * stop_distance
        if planned_risk > Decimal(
            str(settings.execution.maximum_live_risk_per_trade_eur)
        ):
            state.update(
                {
                    "status": "RISK_BLOCKED",
                    "last_reason": "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
                    "planned_risk_eur": str(planned_risk),
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        preflight = LivePreflight.evaluate(
            settings,
            markets=markets,
            strategy_status=ResearchStatus.PAPER_CANDIDATE,
            data_healthy=True,
            risk_manager_healthy=True,
            exchange_healthy=True,
            reconciliation_healthy=True,
            kill_switch_active=False,
            canary_exception_approved=True,
            operator_canary_authorized=True,
            portfolio_canary=True,
            cap_limits={
                "capital_level": CAPITAL_LEVEL,
                "max_order_eur": str(MAXIMUM_ORDER_EUR),
                "max_exposure_eur": str(MAXIMUM_TOTAL_EXPOSURE_EUR),
                "max_positions": effective_maximum_positions,
                "max_new_orders_per_day": MAXIMUM_NEW_ORDERS_PER_DAY,
            },
        )
        if not preflight.passed or preflight.capability is None:
            state.update(
                {
                    "status": "PREFLIGHT_BLOCKED",
                    "last_reason": "GENERATED_ENTRY_PREFLIGHT_BLOCKED",
                    "preflight_failures": list(preflight.failures),
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        try:
            entry_plan = await _plan_live_entry_order(
                settings,
                client=client,
                market=market,
                requested_notional_eur=guard.approved_notional_eur,
                public_price=public_price,
                liquidity=liquidity,
            )
        except ExecutionBlocked:
            state.update(
                {
                    "status": "EXECUTION_POLICY_BLOCKED",
                    "last_reason": "BOUNDED_ENTRY_ORDER_NOT_FEASIBLE",
                    "positions": positions,
                    "entry_liquidity": liquidity,
                }
            )
            _write_state(settings, state)
            return state
        quantity = entry_plan.quantity
        planned_risk = quantity * stop_distance
        if planned_risk > Decimal(
            str(settings.execution.maximum_live_risk_per_trade_eur)
        ):
            state.update(
                {
                    "status": "RISK_BLOCKED",
                    "last_reason": "PLANNED_ORDER_RISK_EXCEEDS_CAP",
                    "planned_risk_eur": str(planned_risk),
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        identity = stable_hash(
            [
                "GENERATED_LIVE_ENTRY",
                selected["strategy_dna_hash"],
                selected["signal_id"],
                market,
            ],
            length=40,
        )
        intent = OrderIntent(
            intent_id=identity[:32],
            idempotency_key=f"generated-live-entry:{identity}",
            market=market,
            side=OrderSide.BUY,
            order_type=entry_plan.order_type,
            quantity=quantity,
            limit_price=entry_plan.limit_price,
            time_in_force=entry_plan.time_in_force,
            strategy_id=str(selected["strategy_id"]),
            strategy_dna_hash=str(selected["strategy_dna_hash"]),
            signal_id=str(selected["signal_id"]),
            portfolio_decision_id=identity,
            maximum_notional_eur=MAXIMUM_ORDER_EUR,
            reason_codes=(
                "EXACT_POSITIVE_FROZEN_DNA",
                "NATURAL_NEXT_OPEN_SIGNAL",
                "OPERATOR_APPROVED_POSITIVE_PORTFOLIO",
                entry_plan.execution_policy,
            ),
        )
        expected_net_edge = (
            _decimal(selected.get("net_return"))
            / Decimal(max(1, int(selected.get("trade_count") or 0)))
        )
        if expected_net_edge <= 0:
            expected_net_edge = planned_target_net_edge(
                entry_price=max(
                    public_price,
                    entry_plan.limit_price or public_price,
                ),
                target_price=public_price + target_distance,
                costs=CanonicalCostModel.from_settings(settings),
            )
        confidence = max(
            Decimal("0.01"),
            min(
                Decimal("1"),
                _decimal(selected.get("selection_confidence_completeness"))
                * _decimal(selected.get("sample_confidence_multiplier")),
            ),
        )
        try:
            canonical_plan = canonicalize_approved_buy_order(
                settings,
                intent,
                mark_price=max(
                    public_price,
                    entry_plan.limit_price or public_price,
                ),
                current_quantity=excess.get(
                    market.split("-")[0], Decimal("0")
                ),
                equity_eur=current_equity,
                approved_risk_eur=planned_risk,
                expected_net_edge=expected_net_edge,
                confidence=confidence,
                family=str(
                    candidate.get("economic_hypothesis_family")
                    or candidate.get("family")
                    or selected["strategy_id"]
                ),
                evidence_id=str(selected["frozen_candidate_hash"]),
                policy_version=str(
                    authority.get("authority_hash")
                    or "positive_strategy_live_authority_v1"
                ),
                account_state={
                    "health_status": health.get("status"),
                    "equity_eur": str(current_equity),
                    "available_eur": str(available_eur),
                    "entry_allowed": health.get("entry_allowed"),
                },
                portfolio_state={
                    "managed_portfolio": shared_managed,
                    "generated_positions": positions,
                },
                horizon_seconds=max(
                    60,
                    int(TIMEFRAME_SECONDS[str(selected["timeframe"])]),
                ),
            )
        except ExecutionBlocked:
            state.update(
                {
                    "status": "RISK_BLOCKED",
                    "last_reason": "CANONICAL_BUY_CHAIN_REJECTED",
                    "positions": positions,
                }
            )
            _write_state(settings, state)
            return state
        intent = canonical_plan.order
        async def submit_reserved_generated_entry(
            fresh_portfolio: Mapping[str, Any],
        ) -> dict[str, Any]:
            return await client.submit_order(
                intent,
                capability=preflight.capability,
                estimated_price=public_price,
                reconciled_owned_quantity=excess.get(
                    market.split("-")[0],
                    Decimal("0"),
                ),
                reconciled_total_exposure_eur=_decimal(
                    fresh_portfolio["capacity_managed_exposure_eur"]
                ),
                reconciled_open_positions=int(
                    fresh_portfolio["capacity_managed_position_count"]
                ),
                exchange_minimum_order_eur=Decimal("5"),
                canonical_chain=canonical_plan.chain,
            )

        try:
            (
                reservation_approved,
                reservation_reason,
                reservation_portfolio,
                order,
            ) = await submit_level_2_buy_atomically(
                settings,
                requested_notional_eur=(
                    quantity
                    * max(
                        public_price,
                        entry_plan.limit_price or public_price,
                    )
                ),
                submit_order=submit_reserved_generated_entry,
            )
            if not reservation_approved or order is None:
                state.update(
                    {
                        "status": "PORTFOLIO_CAP_BLOCKED",
                        "last_reason": reservation_reason,
                        "managed_portfolio": reservation_portfolio,
                        "positions": positions,
                    }
                )
                _write_state(settings, state)
                return state
        except ReconciliationRequired:
            order = {
                "clientOrderId": client.client_order_id_for(
                    intent.idempotency_key
                ),
                "status": "ambiguous",
            }
        except ExecutionBlocked:
            state.update(
                {
                    "status": "ORDER_REJECTED",
                    "last_reason": "GENERATED_LIVE_ENTRY_REJECTED",
                    "positions": positions,
                    "orders_generated_this_cycle": 1,
                }
            )
            _write_state(settings, state)
            return state
        filled_quantity = _position_quantity_from_order(
            order,
            fallback=quantity,
        )
        fill_price = _position_price_from_order(
            order,
            quantity=filled_quantity,
            fallback=public_price,
        )
        new_position = {
            "status": "ENTRY_PENDING_RECONCILIATION",
            "strategy_id": selected["strategy_id"],
            "strategy_dna_hash": selected["strategy_dna_hash"],
            "frozen_candidate_hash": selected["frozen_candidate_hash"],
            "market": market,
            "timeframe": selected["timeframe"],
            "signal_id": selected["signal_id"],
            "signal_timestamp": selected["signal_timestamp"],
            "entry_price": str(fill_price),
            "quantity": str(filled_quantity),
            "stop_loss": str(fill_price - stop_distance),
            "take_profit_1": str(fill_price + target_distance),
            "take_profit_2": str(
                fill_price + target_distance * Decimal("2")
            ),
            "tp1_reached": False,
            "opened_at": utc_iso(),
            "entry_order_submitted_at": now.isoformat(),
            "entry_reprice_count": 0,
            "requested_quantity": str(quantity),
            "client_order_id": str(
                order.get("clientOrderId")
                or client.client_order_id_for(intent.idempotency_key)
            ),
            "order_status": order.get("status"),
            "order_type": entry_plan.order_type.value,
            "time_in_force": entry_plan.time_in_force.value,
            "limit_price": (
                str(entry_plan.limit_price)
                if entry_plan.limit_price is not None
                else None
            ),
            "execution_policy": entry_plan.execution_policy,
            "execution_fallback_reason": entry_plan.fallback_reason,
            "order_id_masked": stable_hash(
                str(order.get("orderId") or ""),
                length=12,
            ),
            "planned_risk_eur": str(planned_risk),
            "candidate_metrics": {
                key: (candidate.get("metrics") or {}).get(key)
                for key in (
                    "net_return",
                    "profit_factor",
                    "stressed_profit_factor",
                    "trade_count",
                )
            },
        }
        positions[selected["strategy_dna_hash"]] = new_position
        if _normalized_order_status(order) == "filled":
            try:
                protective_result = await _place_generated_native_stop(
                    client,
                    capability=preflight.capability,
                    position=new_position,
                    quantity=filled_quantity,
                    estimated_price=fill_price,
                )
            except (ExecutionBlocked, ReconciliationRequired):
                state.update(
                    {
                        "status": "RECONCILIATION_BLOCKED",
                        "last_reason": "NATIVE_PROTECTIVE_STOP_NOT_CONFIRMED",
                        "positions": positions,
                        "local_hard_stop_active": True,
                        "orders_generated": int(
                            state.get("orders_generated") or 0
                        )
                        + 1,
                        "orders_submitted": int(
                            state.get("orders_submitted") or 0
                        )
                        + 1,
                        "orders_generated_this_cycle": 1,
                        "orders_submitted_this_cycle": 1,
                    }
                )
                _write_state(settings, state)
                return state
            protective_order = dict(
                protective_result.pop("native_protective_stop_order")
            )
            new_position.update(protective_result)
            positions[selected["strategy_dna_hash"]] = new_position
            state["native_protective_stop"] = {
                "status": protective_order.get("status"),
                "trigger_price": new_position.get(
                    "protective_stop_trigger"
                ),
                "order_id_masked": stable_hash(
                    str(protective_order.get("orderId") or ""),
                    length=12,
                ),
            }
        weekly_record = weekly_budget.record_entry(
            entry_identity=identity,
            strategy_id=str(selected["strategy_id"]),
            strategy_dna_hash=str(selected["strategy_dna_hash"]),
            market=market,
            timeframe=str(selected["timeframe"]),
            regime=None,
            order_status=str(order.get("status") or ""),
            observed_at=now,
        )
        cooldowns.record_entry(
            strategy_id=str(selected["strategy_id"]),
            strategy_dna_hash=str(selected["strategy_dna_hash"]),
            market=market,
            timeframe=str(selected["timeframe"]),
            signal_candle_at=str(selected["signal_timestamp"]),
            observed_at=now,
        )
        notify_autonomous_event_safely(
            settings,
            "ORDER_FILLED"
            if _normalized_order_status(order) == "filled"
            else "ORDER_PARTIALLY_FILLED"
            if _position_quantity_from_order(
                order,
                fallback=Decimal("0"),
            )
            > 0
            else "ORDER_SUBMITTING",
            {
                **_live_order_notification_payload(
                    order,
                    market=market,
                    side="BUY",
                    order_type=entry_plan.order_type.value,
                    requested_quantity=quantity,
                    fallback_price=fill_price,
                    strategy_id=str(selected["strategy_id"]),
                    timeframe=str(selected["timeframe"]),
                ),
                "stop_loss": float(fill_price - stop_distance),
                "take_profit_1": float(
                    fill_price + target_distance
                ),
                "take_profit_2": float(
                    fill_price + target_distance * Decimal("2")
                ),
                "execution": "LIVE_POSITIVE_PORTFOLIO_CANARY",
                "time_in_force": entry_plan.time_in_force.value,
                "limit_price": (
                    float(entry_plan.limit_price)
                    if entry_plan.limit_price is not None
                    else None
                ),
                "execution_fallback_reason": entry_plan.fallback_reason,
            },
        )
        state.update(
            {
                "status": "ENTRY_SUBMITTED",
                "last_reason": "NATURAL_GENERATED_ENTRY_SUBMITTED",
                "positions": positions,
                "selected_entry": selected,
                "entry_liquidity": liquidity,
                "entry_order_plan": {
                    "order_type": entry_plan.order_type.value,
                    "time_in_force": entry_plan.time_in_force.value,
                    "limit_price": (
                        str(entry_plan.limit_price)
                        if entry_plan.limit_price is not None
                        else None
                    ),
                    "planned_notional_eur": str(
                        entry_plan.planned_notional_eur
                    ),
                    "execution_policy": entry_plan.execution_policy,
                    "fallback_reason": entry_plan.fallback_reason,
                },
                "material_wallet_position_count": material_count,
                "material_wallet_markets": material_markets,
                "weekly_trade_budget": weekly_record.get("status"),
                "position_limit_status": position_limit_status,
                "orders_generated": int(
                    state.get("orders_generated") or 0
                )
                + (2 if _normalized_order_status(order) == "filled" else 1),
                "orders_submitted": int(
                    state.get("orders_submitted") or 0
                )
                + (2 if _normalized_order_status(order) == "filled" else 1),
                "orders_generated_this_cycle": (
                    2 if _normalized_order_status(order) == "filled" else 1
                ),
                "orders_submitted_this_cycle": (
                    2 if _normalized_order_status(order) == "filled" else 1
                ),
            }
        )
        _write_state(settings, state)
        return state


__all__ = [
    "APPROVAL_PHRASE",
    "DNA_APPROVAL_PREFIX",
    "MAXIMUM_NEW_ORDERS_PER_DAY",
    "MAXIMUM_OPEN_POSITIONS",
    "MAXIMUM_ORDER_EUR",
    "MAXIMUM_TOTAL_EXPOSURE_EUR",
    "activate_positive_strategy_live_authority",
    "approve_positive_strategy_dna",
    "deactivate_positive_strategy_live_authority",
    "execute_generated_strategy_live_once",
    "migrate_positive_strategy_live_order_cap",
    "migrate_positive_strategy_live_capital_level_2",
    "positive_strategy_live_authority_status",
    "positive_strategy_dna_approval_phrase",
    "rank_natural_entries",
    "signal_execution_window",
    "synchronize_positive_strategy_live_authority",
]
