"""Read-only repository and live-capability audit.

The audit deliberately consumes durable repository evidence and sanitized
runtime artifacts.  It never imports credentials, calls a broker, mutates
strategy state, or creates an order.
"""

from __future__ import annotations

import ast
import ctypes
import json
import os
import warnings
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from config.settings import Settings
from execution.canonical_state import (
    assert_replay_deterministic,
    replay_execution_events,
)
from reporting.reference_integration_health import (
    build_reference_integration_health,
)
from utils.common import atomic_write_json, read_json, sha256_file, stable_hash, utc_iso

SCHEMA_VERSION = "system_audit_v1"
AUDIT_FILENAMES = (
    "repository_inventory.json",
    "strategy_inventory.json",
    "strategy_family_map.json",
    "duplicate_strategy_report.json",
    "execution_capability_report.json",
    "live_blocker_report.json",
    "risk_control_report.json",
    "data_dependency_report.json",
    "reference_integration_report.json",
    "architecture_gap_report.md",
)
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data_store",
    "output",
}
REFERENCE_ROOT_PARTS = {"crypto-references", "reference_repos"}


def _json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _python_files(
    root: Path,
    *,
    excluded_parts: set[str] | None = None,
) -> list[Path]:
    excluded = EXCLUDED_PARTS if excluded_parts is None else excluded_parts
    paths: list[Path] = []
    for directory, child_directories, filenames in os.walk(root):
        child_directories[:] = sorted(
            name for name in child_directories if name not in excluded
        )
        base = Path(directory)
        paths.extend(
            base / filename
            for filename in sorted(filenames)
            if filename.endswith(".py")
        )
    return sorted(paths)


def _module_inventory(path: Path, root: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parse_status = "PASSED"
    except (OSError, SyntaxError, UnicodeDecodeError):
        source = ""
        tree = ast.Module(body=[], type_ignores=[])
        parse_status = "FAILED"
    imports: set[str] = set()
    classes: list[str] = []
    functions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "line_count": len(source.splitlines()),
        "parse_status": parse_status,
        "imports": sorted(imports),
        "classes": sorted(set(classes)),
        "functions": sorted(set(functions)),
    }


def _reference_module_inventory(path: Path, root: Path) -> dict[str, Any]:
    """Index vendored references without expensive symbol extraction.

    Reference repositories are evidence sources, not production modules.  A
    path/hash/parse-status index preserves integrity and syntax visibility
    without walking every foreign AST or serializing its complete symbol set.
    """

    try:
        content = path.read_bytes()
        read_succeeded = True
    except OSError:
        content = b""
        read_succeeded = False
    try:
        source = content.decode("utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(source, filename=str(path))
        parse_status = "PASSED" if read_succeeded else "FAILED"
    except (SyntaxError, UnicodeDecodeError):
        source = ""
        parse_status = "FAILED"
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "sha256": sha256(content).hexdigest() if content else None,
        "bytes": len(content),
        "line_count": len(source.splitlines()),
        "parse_status": parse_status,
    }


def _registered_cli_commands(cli_path: Path) -> list[str]:
    if not cli_path.is_file():
        return []
    try:
        tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    commands: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            commands.add(node.args[0].value)
    return sorted(commands)


def _canonical_family(value: str) -> str:
    normalized = value.upper()
    if "LIQUIDITY_SWEEP" in normalized:
        return "liquidity_sweep_recovery"
    if "CROSS_SECTION" in normalized or "RELATIVE" in normalized:
        return "cross_sectional_momentum"
    if "ROTATION" in normalized or "CASH" in normalized:
        return "defensive_cash_rotation"
    if "PULLBACK" in normalized:
        return "trend_pullback"
    if "BREAKOUT" in normalized or "TURTLE" in normalized or "DONCHIAN" in normalized:
        return "volatility_compression_breakout"
    if "STRUCTURE" in normalized or "FRACTAL" in normalized:
        return "fractal_market_structure"
    if "VOLUME" in normalized or "ORDERFLOW" in normalized or "OBV" in normalized:
        return "volume_orderflow_confirmation"
    if "REGIME" in normalized and ("REVERS" in normalized or "MEAN" in normalized):
        return "regime_conditioned_reversal"
    if "REVERS" in normalized or "MEAN_REVERSION" in normalized:
        return "range_mean_reversion"
    if "MOMENTUM" in normalized or "TREND" in normalized:
        return "time_series_momentum"
    return "unclassified_research"


def _strategy_evidence_tier(row: Mapping[str, Any]) -> str:
    """Describe attained evidence without upgrading weaker lifecycle states."""

    lifecycle = str(row.get("lifecycle_state") or "").upper()
    if lifecycle in {"LIVE_VALIDATED", "LIVE_PRODUCTION_VALIDATED"} or row.get(
        "live_validated"
    ) is True:
        return "LIVE_VALIDATED"
    if row.get("live_canary_active") is True:
        return "LIVE_CANARY_ACTIVE"
    if row.get("live_canary_eligible") is True:
        return "LIVE_CANARY_ELIGIBLE"
    if row.get("paper_active") is True:
        return "PAPER_ACTIVE"
    if row.get("paper_activation_pending") is True:
        return "PAPER_ADAPTER_PENDING"
    if row.get("research_positive") is True or row.get("backtest_positive") is True:
        return "BACKTEST_POSITIVE_ONLY"
    return "REJECTED_OR_UNPROVEN"


def _strategy_rows(settings: Settings) -> list[dict[str, Any]]:
    registry = _json_mapping(
        settings.paths.output_dir / "strategies" / "all_strategy_dna.json"
    )
    rows: list[dict[str, Any]] = []
    for source_key in ("economic_evidence", "registered_pending"):
        for raw in registry.get(source_key) or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            strategy_id = str(
                row.get("strategy_id")
                or row.get("strategy_name")
                or row.get("strategy_dna")
                or "UNIDENTIFIED"
            )
            family = str(
                row.get("strategy_family")
                or row.get("family_cluster")
                or (row.get("metadata") or {}).get("family")
                or "UNCLASSIFIED"
            )
            markets = sorted(
                {
                    str(value)
                    for value in (
                        row.get("markets")
                        or ([row.get("market")] if row.get("market") else [])
                    )
                    if value
                }
            )
            strategy_row = {
                    "strategy_id": strategy_id,
                    "strategy_dna": str(
                        row.get("strategy_dna")
                        or row.get("strategy_dna_hash")
                        or ""
                    ),
                    "source": source_key,
                    "strategy_family": family,
                    "canonical_family": _canonical_family(family),
                    "timeframe": row.get("timeframe"),
                    "markets": markets,
                    "entry_logic": row.get("entry_logic"),
                    "exit_logic": row.get("exit_logic"),
                    "stop_logic": row.get("stop_logic"),
                    "lifecycle_state": row.get("lifecycle_state"),
                    "backtest_positive": bool(row.get("backtest_positive")),
                    "research_positive": bool(row.get("research_positive")),
                    "paper_active": bool(row.get("paper_active")),
                    "paper_adapter_available": bool(
                        row.get("paper_adapter_available")
                    ),
                    "paper_activation_pending": bool(
                        row.get("paper_activation_pending")
                    ),
                    "live_canary_eligible": bool(row.get("live_canary_eligible")),
                    "live_canary_active": bool(row.get("live_canary_active")),
                    "live_validated": bool(row.get("live_validated")),
                    "capital_scale_eligible": bool(
                        row.get("capital_scale_eligible")
                    ),
                    "hard_blocker_count": len(row.get("hard_blockers") or []),
                    "capital_scaling_warning_count": len(
                        row.get("capital_scaling_warnings") or []
                    ),
                    "frozen": bool(row.get("frozen")),
                    "costs_included": row.get("costs_included"),
                    "lookahead_status": row.get("lookahead_status"),
                    "repainting_status": row.get("repainting_status"),
                    "normal_profit_factor": row.get("normal_profit_factor"),
                    "net_total_return": row.get("net_total_return"),
                    "maximum_drawdown": row.get("maximum_drawdown"),
                }
            strategy_row["evidence_tier"] = _strategy_evidence_tier(
                strategy_row
            )
            rows.append(strategy_row)
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row["strategy_dna"] or stable_hash(
            [
                row["strategy_id"],
                row["strategy_family"],
                row["timeframe"],
                row["markets"],
                row["source"],
            ],
            length=64,
        )
        existing = deduplicated.get(identity)
        if existing is None or (
            existing["source"] == "registered_pending"
            and row["source"] == "economic_evidence"
        ):
            deduplicated[identity] = row
    return [deduplicated[key] for key in sorted(deduplicated)]


def _semantic_signature(row: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "family": row.get("canonical_family"),
            "timeframe": row.get("timeframe"),
            "markets": row.get("markets"),
            "entry_logic": row.get("entry_logic"),
            "exit_logic": row.get("exit_logic"),
            "stop_logic": row.get("stop_logic"),
        },
        length=32,
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _safe_model_fields(model: Any, names: Iterable[str]) -> dict[str, Any]:
    return {
        name: getattr(model, name)
        for name in names
        if hasattr(model, name)
    }


def _repository_inventory(settings: Settings) -> dict[str, Any]:
    root = settings.paths.project_root
    production_modules = [
        _module_inventory(path, root)
        for path in _python_files(
            root,
            excluded_parts=EXCLUDED_PARTS | REFERENCE_ROOT_PARTS,
        )
    ]
    reference_modules = [
        _reference_module_inventory(path, root)
        for reference_name in sorted(REFERENCE_ROOT_PARTS)
        if (root / reference_name).is_dir()
        for path in _python_files(root / reference_name)
    ]
    cli_commands = _registered_cli_commands(root / "core" / "cli.py")
    production_parse_failures = sum(
        row["parse_status"] != "PASSED" for row in production_modules
    )
    reference_parse_failures = sum(
        row["parse_status"] != "PASSED" for row in reference_modules
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "project_root": str(root),
        "audit_scope": "PRODUCTION_REPOSITORY_ONLY",
        "python_file_count": len(production_modules),
        "test_file_count": sum(
            row["path"].startswith("tests/") for row in production_modules
        ),
        "total_python_lines": sum(
            int(row["line_count"]) for row in production_modules
        ),
        "parse_failure_count": production_parse_failures,
        "production_scope": {
            "python_file_count": len(production_modules),
            "parse_failure_count": production_parse_failures,
            "affects_production_health": True,
        },
        "reference_scope": {
            "roots": sorted(REFERENCE_ROOT_PARTS),
            "inventory_mode": "PATH_HASH_PARSE_STATUS_ONLY",
            "symbol_inventory_collected": False,
            "python_file_count": len(reference_modules),
            "parse_failure_count": reference_parse_failures,
            "informational_only": True,
            "affects_production_health": False,
            "modules": reference_modules,
        },
        "registered_cli_tokens": cli_commands,
        "registered_cli_token_count": len(cli_commands),
        "modules": production_modules,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _strategy_inventory(settings: Settings) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _strategy_rows(settings)
    registry = _json_mapping(
        settings.paths.output_dir / "strategies" / "all_strategy_dna.json"
    )
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[str(row["canonical_family"])].append(row)
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "registered_implementation_count": len(
            registry.get("registered_pending") or []
        ),
        "economic_candidate_count": len(
            registry.get("economic_evidence") or []
        ),
        "deduplicated_research_variant_count": len(rows),
        "live_eligible_family_count": len(
            {
                row["canonical_family"]
                for row in rows
                if row["live_canary_eligible"]
            }
        ),
        "live_validated_family_count": len(
            {
                row["canonical_family"]
                for row in rows
                if str(row.get("lifecycle_state") or "").upper()
                in {"LIVE_VALIDATED", "LIVE_PRODUCTION_VALIDATED"}
            }
        ),
        "count_semantics": {
            "registered_implementation_count": "registered_pending implementation records",
            "economic_candidate_count": "raw economic_evidence candidate records",
            "deduplicated_research_variant_count": "unique DNA or stable research identity",
            "live_eligible_family_count": "families containing a live-canary-eligible variant",
            "live_validated_family_count": "families explicitly in a LIVE_VALIDATED lifecycle state",
        },
        "backtest_positive_count": sum(row["backtest_positive"] for row in rows),
        "backtest_positive_count_semantics": (
            "raw backtest-positive variants; not proof of robustness, paper "
            "permission, live eligibility, or profitability"
        ),
        "research_positive_count": sum(row["research_positive"] for row in rows),
        "paper_active_count": sum(row["paper_active"] for row in rows),
        "paper_adapter_pending_count": sum(
            row["paper_activation_pending"] for row in rows
        ),
        "paper_adapter_available_count": sum(
            row["paper_adapter_available"] for row in rows
        ),
        "capital_scale_eligible_count": sum(
            row["capital_scale_eligible"] for row in rows
        ),
        "live_canary_active_count": sum(row["live_canary_active"] for row in rows),
        "timeframe_counts": dict(
            sorted(Counter(str(row.get("timeframe")) for row in rows).items())
        ),
        "strategies": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    family_map = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "family_count": len(family_rows),
        "families": [
            {
                "family_id": family,
                "deduplicated_variant_count": len(members),
                "backtest_positive_count": sum(
                    row["backtest_positive"] for row in members
                ),
                # Deprecated compatibility alias. The explicit name above and
                # the semantics block prevent raw positivity being mistaken
                # for forward or live validation.
                "positive_count": sum(row["backtest_positive"] for row in members),
                "research_positive_count": sum(
                    row["research_positive"] for row in members
                ),
                "paper_active_count": sum(row["paper_active"] for row in members),
                "paper_adapter_pending_count": sum(
                    row["paper_activation_pending"] for row in members
                ),
                "live_canary_eligible_count": sum(
                    row["live_canary_eligible"] for row in members
                ),
                "live_canary_active_count": sum(
                    row["live_canary_active"] for row in members
                ),
                "live_validated_count": sum(
                    row["evidence_tier"] == "LIVE_VALIDATED" for row in members
                ),
                "capital_scale_eligible_count": sum(
                    row["capital_scale_eligible"] for row in members
                ),
                "evidence_tier_counts": dict(
                    sorted(Counter(row["evidence_tier"] for row in members).items())
                ),
                "strategy_ids": sorted({str(row["strategy_id"]) for row in members}),
                "timeframes": sorted(
                    {
                        str(row["timeframe"])
                        for row in members
                        if row.get("timeframe")
                    }
                ),
            }
            for family, members in sorted(family_rows.items())
        ],
        "count_semantics": {
            "positive_count": (
                "deprecated alias of backtest_positive_count; never interpret "
                "as robust, paper-permitted, live-validated, or profitable"
            ),
            "backtest_positive_count": "raw positive historical backtest result",
            "paper_active_count": "variants currently collecting paper evidence",
            "paper_adapter_pending_count": (
                "research-positive variants missing an active paper lifecycle; "
                "not paper permission"
            ),
            "live_validated_count": (
                "variants explicitly in a LIVE_VALIDATED lifecycle state"
            ),
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    by_dna: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["strategy_dna"]:
            by_dna[str(row["strategy_dna"])].append(row)
        by_semantic[_semantic_signature(row)].append(row)
    exact_groups = [
        {
            "strategy_dna": dna,
            "member_count": len(members),
            "strategy_ids": sorted({str(row["strategy_id"]) for row in members}),
            "recommended_action": "KEEP_CANONICAL_IDENTITY",
        }
        for dna, members in sorted(by_dna.items())
        if len(members) > 1
    ]
    semantic_groups = [
        {
            "semantic_signature": signature,
            "member_count": len(members),
            "canonical_strategy": sorted(
                members,
                key=lambda row: (
                    not row["backtest_positive"],
                    not row["frozen"],
                    str(row["strategy_id"]),
                ),
            )[0]["strategy_id"],
            "duplicate_candidates": sorted(
                {str(row["strategy_id"]) for row in members}
            ),
            "family_id": members[0]["canonical_family"],
            "timeframe": members[0].get("timeframe"),
            "markets": members[0].get("markets"),
            "recommended_action": "REVIEW_MERGE_AS_VARIANT",
        }
        for signature, members in sorted(by_semantic.items())
        if len({str(row["strategy_id"]) for row in members}) > 1
    ]
    duplicates = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "identity_scope": "CURRENT_DEDUPLICATED_STRATEGY_REGISTRY",
        "exact_dna_duplicate_cluster_count": len(exact_groups),
        "semantic_review_cluster_count": len(semantic_groups),
        "exact_dna_clusters": exact_groups,
        "semantic_review_clusters": semantic_groups,
        "history_deleted": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return inventory, family_map, duplicates


def _canonical_execution_evidence(settings: Settings) -> dict[str, Any]:
    """Read/replay canonical execution evidence without contacting the venue."""

    output = settings.paths.output_dir / "live"
    persisted = _json_mapping(output / "canonical_execution_state.json")
    replay_report = _json_mapping(output / "canonical_execution_replay.json")
    ledger_path = settings.paths.checkpoints_dir / "live_execution.jsonl"
    if persisted:
        return {
            "status": "READY",
            "state_source": "PERSISTED_REBUILDABLE_CANONICAL_READ_MODEL",
            "ledger_source": str(ledger_path),
            "state_hash": persisted.get("state_hash"),
            "position_count": sum(
                float((row or {}).get("quantity") or 0) > 0
                for row in dict(persisted.get("positions") or {}).values()
            ),
            "evidence_gap_count": len(persisted.get("evidence_gaps") or []),
            "deterministic_replay_verified": bool(
                replay_report.get("deterministic")
            ),
            "replay_report": replay_report,
            "private_exchange_requests": 0,
        }
    if not ledger_path.is_file():
        return {
            "status": "NOT_AVAILABLE",
            "state_source": "CANONICAL_EXECUTION_LEDGER_REPLAY",
            "ledger_source": str(ledger_path),
            "deterministic_replay_verified": False,
            "private_exchange_requests": 0,
        }
    try:
        events = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        state = replay_execution_events(events)
        state_hash = assert_replay_deterministic(events)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "status": "RECONCILIATION_REQUIRED",
            "state_source": "CANONICAL_EXECUTION_LEDGER_REPLAY",
            "ledger_source": str(ledger_path),
            "deterministic_replay_verified": False,
            "private_exchange_requests": 0,
        }
    return {
        "status": "READY",
        "state_source": "CANONICAL_EXECUTION_LEDGER_REPLAY",
        "ledger_source": str(ledger_path),
        "state_hash": state_hash,
        "raw_event_count": len(events),
        "unique_event_count": len(state.processed_event_ids),
        "position_count": sum(
            position.quantity > 0 for position in state.positions.values()
        ),
        "evidence_gap_count": len(state.evidence_gaps),
        "deterministic_replay_verified": True,
        "private_exchange_requests": 0,
    }


def _runtime_evidence(settings: Settings) -> dict[str, Any]:
    output = settings.paths.output_dir
    heartbeat = _json_mapping(output / "live" / "heartbeat.json")
    live_lock = _json_mapping(output / "live" / "autonomous_live.lock")
    account = _json_mapping(output / "operations" / "live_account_health.json")
    authority = _json_mapping(
        output / "governance" / "positive_strategy_live_authority.json"
    )
    playbook_authority = _json_mapping(
        settings.paths.project_root / "config" / "live_playbook_authority.json"
    )
    telegram = _json_mapping(output / "notifications" / "telegram_status.json")
    attribution_path = (
        output / "operations" / "decision_execution_attribution.json"
    )
    attribution = _json_mapping(attribution_path)
    intelligence_path = output / "intelligence" / "model_status.json"
    intelligence = _json_mapping(intelligence_path)
    drift = dict(intelligence.get("drift_monitor") or {})
    pid = int(live_lock.get("pid") or heartbeat.get("pid") or 0)
    reconciliation = dict(account.get("reconciliation") or {})
    return {
        "pid": pid or None,
        "process_running": _pid_alive(pid),
        "control_state": heartbeat.get("control_state"),
        "private_account_websocket": {
            "state": (
                heartbeat.get("private_account_websocket") or {}
            ).get("state"),
            "ready_for_new_entries": bool(
                (heartbeat.get("private_account_websocket") or {}).get(
                    "ready_for_new_entries"
                )
            ),
            "secrets_serialized": bool(
                (heartbeat.get("private_account_websocket") or {}).get(
                    "secrets_serialized"
                )
            ),
        },
        "account_status": account.get("status"),
        "entry_allowed": account.get("entry_allowed") is True,
        "entry_blockers": list(account.get("entry_blockers") or []),
        "reconciliation": {
            "healthy": bool(reconciliation.get("healthy")),
            "local_open_orders": int(reconciliation.get("local_open_orders") or 0),
            "remote_open_orders": int(reconciliation.get("remote_open_orders") or 0),
            "reason_codes": list(reconciliation.get("reason_codes") or []),
            "checked_at": reconciliation.get("checked_at"),
        },
        "authority_active": (
            authority.get("active") is True
            or playbook_authority.get("active") is True
        ),
        "approved_strategy_count": (
            len(authority.get("approved_candidates") or [])
            + len(playbook_authority.get("approved_playbooks") or [])
        ),
        "maximum_order_eur": authority.get("maximum_order_eur"),
        "maximum_total_exposure_eur": authority.get("maximum_total_exposure_eur"),
        "maximum_open_positions": authority.get("maximum_open_positions"),
        "spot_only": authority.get("spot_only") is True,
        "long_only": authority.get("long_only") is True,
        "autoscale": authority.get("autoscale"),
        "withdrawals_available": authority.get("withdrawals_available"),
        "telegram": {
            "status": telegram.get("status"),
            "status_scope": "CURRENT_STATUS",
            "source": _relative(
                output / "notifications" / "telegram_status.json",
                settings.paths.project_root,
            ),
            "observed_at": telegram.get("updated_at")
            or telegram.get("checked_at")
            or telegram.get("generated_at"),
            "enabled": bool(telegram.get("enabled")),
            "queue_size": int(telegram.get("active_queue_size") or 0),
            "secrets_redacted": bool(telegram.get("secrets_redacted")),
            "historical_findings_are_current_status": False,
        },
        "decision_execution_attribution": {
            "status": attribution.get("status") or "NOT_AVAILABLE",
            "trade_count": int(attribution.get("trade_count") or 0),
            "closed_round_trips": int(
                attribution.get("closed_round_trips") or 0
            ),
            "open_positions": int(attribution.get("open_positions") or 0),
            "decision_price_mapped_count": int(
                attribution.get("decision_price_mapped_count") or 0
            ),
            "privacy_safe": all(
                value is False
                for value in dict(attribution.get("privacy") or {}).values()
            ),
            "source": _relative(attribution_path, settings.paths.project_root),
        },
        "ml_drift_monitor": {
            "status": drift.get("status") or "NOT_AVAILABLE",
            "authority": drift.get("authority") or intelligence.get("authority"),
            "live_decision_influence": bool(
                drift.get("live_decision_influence")
            ),
            "critical_feature_count": int(
                drift.get("critical_feature_count") or 0
            ),
            "warning_feature_count": int(
                drift.get("warning_feature_count") or 0
            ),
            "model_status": intelligence.get("status") or "NOT_AVAILABLE",
            "model_row_count": int(intelligence.get("row_count") or 0),
            "source": _relative(intelligence_path, settings.paths.project_root),
        },
    }


def _execution_and_blockers(
    settings: Settings,
    *,
    strategy_inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = settings.paths.project_root
    execution_source = (
        (root / "execution" / "execution.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if (root / "execution" / "execution.py").is_file()
        else ""
    )
    live_capital_source = (
        (root / "core" / "live_capital.py").read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if (root / "core" / "live_capital.py").is_file()
        else ""
    )
    autonomous_buy_sources = {
        name: (
            (root / "core" / name).read_text(
                encoding="utf-8",
                errors="ignore",
            )
            if (root / "core" / name).is_file()
            else ""
        )
        for name in (
            "autonomous_trading.py",
            "generated_strategy_live.py",
            "event_driven_live.py",
            "cli.py",
        )
    }
    runtime = _runtime_evidence(settings)
    canonical_execution = _canonical_execution_evidence(settings)
    required_tokens = {
        "live_bitvavo_client": "BitvavoSpotClient",
        "idempotency": "idempotency",
        "reconciliation": "reconcil",
        "limit_orders": "LIMIT",
        "market_orders": "MARKET",
        "cancel_orders": "cancel",
        "balances": "balance",
        "open_orders": "open_orders",
        "fills": "fill",
        "decimal_prices_quantities": "Decimal",
    }
    capabilities = {
        key: token.casefold() in execution_source.casefold()
        for key, token in required_tokens.items()
    }
    capabilities["unknown_order_state_recovery"] = all(
        token.casefold() in execution_source.casefold()
        for token in (
            "ORDER_STATE_UNKNOWN",
            "recent_orders",
            "client_order_id",
            "UNKNOWN_ORDER_LOOKUP_FAILED",
        )
    )
    capabilities["unknown_cancellation_state_recovery"] = all(
        token.casefold() in execution_source.casefold()
        for token in (
            "CANCEL_REQUESTED",
            "CANCEL_STATE_UNKNOWN",
            "CANCEL_RESOLVED",
            "UNKNOWN_CANCELLATION_STATE",
            "CANCELLATION_PARTIAL_FILL_STILL_OPEN",
        )
    )
    capabilities["incremental_partial_fill_accounting"] = all(
        token.casefold() in execution_source.casefold()
        for token in (
            "record_order_fill_progress",
            "PARTIALLY_FILLED_PROGRESS",
            "cumulative_quantity",
            "ORDER_STATUS_OBSERVED",
            "venue cumulative fill regressed",
        )
    )
    capabilities["pending_buy_exposure_reserved_across_restart"] = all(
        token.casefold() in live_capital_source.casefold()
        for token in (
            "_ledger_pending_buy_reservations",
            "CANONICAL_LEDGER",
            "PENDING_ORDER_EXPOSURE_UNRECONCILED",
            "ledger_recovered_pending_exposure_eur",
            "private_order_identifiers_serialized",
        )
    )
    capabilities["atomic_cross_engine_buy_reservation"] = all(
        token.casefold() in live_capital_source.casefold()
        for token in (
            "LiveEntryReservation",
            "submit_level_2_buy_atomically",
            "LIVE_ENTRY_RESERVATION_BUSY",
            "capital_level_2_capacity",
            "LK_NBLCK",
        )
    )
    capabilities["all_autonomous_buy_routes_atomic"] = (
        "RR_PRIMARY" in live_capital_source
        and "replacing_source" in live_capital_source
        and all(
            "submit_level_2_buy_atomically" in source
            for source in autonomous_buy_sources.values()
        )
        and "replacing_source=\"GENERATED_DNA\""
        in autonomous_buy_sources["generated_strategy_live.py"]
    )
    attribution = runtime["decision_execution_attribution"]
    capabilities["decision_to_fill_attribution"] = (
        attribution["status"]
        in {"READY", "NO_CANONICAL_STRATEGY_FILLS"}
        and attribution["privacy_safe"]
    )
    drift = runtime["ml_drift_monitor"]
    capabilities["ml_drift_monitor_shadow_only"] = (
        drift["status"] != "NOT_AVAILABLE"
        and drift["authority"] == "SHADOW_ONLY"
        and drift["live_decision_influence"] is False
    )
    capabilities["deterministic_canonical_execution_replay"] = bool(
        canonical_execution.get("deterministic_replay_verified")
    )
    execution = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "venue": "bitvavo",
        "quote_currency": "EUR",
        "capabilities": capabilities,
        "runtime": runtime,
        "canonical_execution_state": canonical_execution,
        "secrets_serialized": False,
        "orders_generated_by_audit": 0,
        "orders_submitted_by_audit": 0,
    }
    blocker_categories: dict[str, list[dict[str, Any]]] = {
        "technical_runtime_blockers": [],
        "execution_safety_blockers": [],
        "strategy_evidence_blockers": [],
        "capital_authority_blockers": [],
    }

    def add_blocker(category: str, code: str, evidence: Any) -> None:
        blocker_categories[category].append(
            {"category": category, "code": code, "evidence": evidence}
        )

    if not runtime["process_running"]:
        add_blocker(
            "technical_runtime_blockers",
            "LIVE_PROCESS_NOT_RUNNING",
            runtime["pid"],
        )
    if runtime["control_state"] != "ENABLED":
        add_blocker(
            "capital_authority_blockers",
            "LIVE_CONTROL_NOT_ENABLED",
            runtime["control_state"],
        )
    if not runtime["private_account_websocket"]["ready_for_new_entries"]:
        add_blocker(
            "technical_runtime_blockers",
            "PRIVATE_ACCOUNT_STREAM_NOT_READY",
            runtime["private_account_websocket"]["state"],
        )
    if runtime["account_status"] != "READY":
        add_blocker(
            "technical_runtime_blockers",
            "ACCOUNT_NOT_READY",
            runtime["account_status"],
        )
    if not runtime["entry_allowed"]:
        for reason in runtime["entry_blockers"]:
            add_blocker(
                "execution_safety_blockers",
                str(reason),
                True,
            )
    if not runtime["reconciliation"]["healthy"]:
        add_blocker(
            "technical_runtime_blockers",
            "RECONCILIATION_NOT_HEALTHY",
            runtime["reconciliation"],
        )
    if not runtime["authority_active"]:
        add_blocker(
            "capital_authority_blockers",
            "LIVE_STRATEGY_AUTHORITY_INACTIVE",
            False,
        )
    if runtime["approved_strategy_count"] <= 0:
        add_blocker(
            "strategy_evidence_blockers",
            "NO_APPROVED_STRATEGY_EVIDENCE",
            runtime["approved_strategy_count"],
        )
    if int(strategy_inventory.get("live_validated_family_count") or 0) <= 0:
        add_blocker(
            "strategy_evidence_blockers",
            "NO_LIVE_VALIDATED_STRATEGY_FAMILY",
            strategy_inventory.get("live_validated_family_count"),
        )
    if settings.execution.withdrawals_enabled:
        add_blocker(
            "execution_safety_blockers",
            "WITHDRAWALS_ENABLED",
            True,
        )
    if (
        settings.execution.allow_margin
        or settings.execution.allow_leverage
        or settings.execution.allow_short_selling
        or settings.execution.allow_derivatives
    ):
        add_blocker(
            "execution_safety_blockers",
            "NON_SPOT_EXECUTION_ENABLED",
            True,
        )
    blockers = [
        row
        for category in blocker_categories.values()
        for row in category
    ]
    live_status = "LIVE_ACTIVE" if not blockers else "LIVE_BLOCKED"
    category_counts = {
        category: len(rows) for category, rows in blocker_categories.items()
    }
    blocker_report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "status": live_status,
        "overall_readiness_status": live_status,
        "technical_runtime_status": (
            "READY"
            if not blocker_categories["technical_runtime_blockers"]
            else "BLOCKED"
        ),
        "overall_blocker_count": len(blockers),
        "blocker_count": len(blockers),
        "blocker_count_semantics": "deprecated alias of overall_blocker_count",
        "category_counts": category_counts,
        **blocker_categories,
        "blockers": blockers,
        "non_blocking_runtime_state": {
            "natural_signal_required_for_order": True,
            "no_signal_is_not_a_live_blocker": True,
        },
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    return execution, blocker_report


def _risk_report(settings: Settings) -> dict[str, Any]:
    authority = _json_mapping(
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    root = settings.paths.project_root

    def source(relative: str) -> str:
        path = root / relative
        return (
            path.read_text(encoding="utf-8", errors="ignore").casefold()
            if path.is_file()
            else ""
        )

    sources = {
        "risk_manager": source("risk/risk_manager.py"),
        "canary_guard": source("risk/canary_guard.py"),
        "live_capital": source("core/live_capital.py"),
        "autonomous_trading": source("core/autonomous_trading.py"),
        "generated_live": source("core/generated_strategy_live.py"),
        "event_live": source("core/event_driven_live.py"),
        "manual_live": source("core/cli.py"),
        "execution": source("execution/execution.py"),
    }

    def has_all(source_name: str, *tokens: str) -> bool:
        body = sources[source_name]
        return bool(body) and all(token.casefold() in body for token in tokens)

    controls = {
        "kill_switch": has_all(
            "risk_manager", "kill_switch.active", "KILL_SWITCH_ACTIVE"
        ),
        "maximum_daily_loss": has_all(
            "risk_manager", "maximum_daily_loss", "DAILY_LOSS_LIMIT"
        ),
        "maximum_drawdown": has_all(
            "risk_manager", "maximum_portfolio_drawdown", "DRAWDOWN_LIMIT"
        ),
        "maximum_open_positions": has_all(
            "live_capital",
            "MAXIMUM_MANAGED_POSITIONS",
            "MANAGED_POSITION_LIMIT_REACHED",
        ),
        "maximum_total_exposure": has_all(
            "live_capital",
            "MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR",
            "MANAGED_EXPOSURE_LIMIT_REACHED",
        ),
        "maximum_risk_per_trade": (
            has_all(
                "autonomous_trading",
                "maximum_live_risk_per_trade_eur",
                "LIVE_MAX_RISK_PER_TRADE_EUR_EXCEEDED",
            )
            and has_all(
                "generated_live",
                "maximum_live_risk_per_trade_eur",
                "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
            )
            and has_all(
                "event_live",
                "MAXIMUM_RISK_PER_TRADE_EUR",
                "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
            )
            and has_all("manual_live", "assess_entry", "live_mode=True")
        ),
        "duplicate_order": has_all(
            "execution", "idempotency", "duplicate live order intent"
        ),
        "stale_data": has_all(
            "risk_manager", "snapshot.data_healthy", "DATA_UNHEALTHY"
        ) and has_all(
            "generated_live", "LIVE_ACCOUNT_HEALTH_STALE", 'get("stale")'
        ),
        "reconciliation": has_all(
            "risk_manager", "snapshot.reconciled", "RECONCILIATION_REQUIRED"
        ),
    }
    controls.update(
        {
            "spot_only": settings.execution.spot_only,
            "margin_disabled": not settings.execution.allow_margin,
            "leverage_disabled": not settings.execution.allow_leverage,
            "shorting_disabled": not settings.execution.allow_short_selling,
            "derivatives_disabled": not settings.execution.allow_derivatives,
            "withdrawals_disabled": not settings.execution.withdrawals_enabled,
            "autoscale_disabled": authority.get("autoscale") is False,
        }
    )
    mandatory_controls = (
        "kill_switch",
        "maximum_daily_loss",
        "maximum_drawdown",
        "maximum_open_positions",
        "maximum_total_exposure",
        "maximum_risk_per_trade",
        "duplicate_order",
        "stale_data",
        "reconciliation",
        "spot_only",
        "margin_disabled",
        "leverage_disabled",
        "shorting_disabled",
        "derivatives_disabled",
        "withdrawals_disabled",
        "autoscale_disabled",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "configured_fractional_limits": _safe_model_fields(
            settings.risk,
            (
                "risk_per_trade",
                "maximum_risk_per_trade",
                "maximum_live_risk_per_trade",
                "maximum_total_open_risk",
                "maximum_position_fraction",
                "maximum_portfolio_exposure",
                "maximum_daily_loss",
                "maximum_portfolio_drawdown",
                "maximum_trades_per_day",
                "reserve_cash_fraction",
            ),
        ),
        "active_live_authority_limits": {
            key: authority.get(key)
            for key in (
                "maximum_order_eur",
                "maximum_total_exposure_eur",
                "maximum_open_positions",
                "maximum_new_orders_per_day",
                "maximum_one_position_per_market",
                "maximum_one_position_per_strategy_dna",
            )
        },
        "controls": controls,
        "mandatory_controls": list(mandatory_controls),
        "all_mandatory_safety_modes_enforced": all(
            controls[key] for key in mandatory_controls
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _data_report(settings: Settings) -> dict[str, Any]:
    processed = settings.paths.processed_data_dir
    parquet_files = sorted(processed.glob("*.parquet")) if processed.is_dir() else []
    timeframes: Counter[str] = Counter()
    markets: set[str] = set()
    for path in parquet_files:
        stem = path.stem
        if "_" not in stem:
            continue
        market, timeframe = stem.rsplit("_", 1)
        markets.add(market)
        timeframes[timeframe] += 1
    sync = _json_mapping(
        settings.paths.output_dir / "research" / "data_sync_progress.json"
    )
    service_lock = _json_mapping(
        settings.paths.checkpoints_dir / "data_service.lock"
    )
    service_heartbeat = _json_mapping(
        settings.paths.checkpoints_dir
        / "continuous-data-service_heartbeat.json"
    )
    service_owner = dict(service_lock.get("owner") or {})
    service_pid = int(
        service_owner.get("pid")
        or service_heartbeat.get("pid")
        or 0
    )
    top50 = _json_mapping(
        settings.paths.output_dir / "universe" / "top50_current.json"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "processed_parquet_file_count": len(parquet_files),
        "processed_data_bytes": sum(path.stat().st_size for path in parquet_files),
        "market_count": len(markets),
        "timeframe_file_counts": dict(sorted(timeframes.items())),
        "configured_providers": list(settings.market_data.providers),
        "configured_timeframes": list(settings.market_data.timeframes),
        "continuous_sync": {
            "status": sync.get("status"),
            "phase": sync.get("phase"),
            "total_operations": sync.get("total_operations"),
            "completed_operations": sync.get("completed_operations"),
            "failure_count": sync.get("failure_count"),
            "synthetic_fallback": sync.get("synthetic_fallback"),
            "updated_at": sync.get("updated_at"),
        },
        "continuous_service": {
            "pid": service_pid or None,
            "process_running": _pid_alive(service_pid),
            "heartbeat_at": service_heartbeat.get("heartbeat_at"),
            "heartbeat_status": service_heartbeat.get("status")
            or service_heartbeat.get("state"),
            "lock_mode": service_owner.get("mode"),
            "service_id": service_owner.get("service_id"),
        },
        "top50_snapshot": {
            "present": bool(top50),
            "count": len(top50.get("rows") or top50.get("assets") or []),
            "snapshot_timestamp": top50.get("snapshot_timestamp")
            or top50.get("generated_at"),
        },
        "database_present": settings.paths.database_path.is_file(),
        "database_bytes": (
            settings.paths.database_path.stat().st_size
            if settings.paths.database_path.is_file()
            else 0
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _gap_rows(
    *,
    repository: Mapping[str, Any],
    strategy: Mapping[str, Any],
    execution: Mapping[str, Any],
    blockers: Mapping[str, Any],
    risk: Mapping[str, Any],
    data: Mapping[str, Any],
    reference_integration: Mapping[str, Any],
    root: Path,
) -> list[dict[str, str]]:
    cli = set(repository.get("registered_cli_tokens") or [])
    docs = {
        path.name
        for path in (root / "docs").glob("*.md")
    } if (root / "docs").is_dir() else set()
    required_docs = {
        "ARCHITECTURE.md",
        "STRATEGY_FAMILIES.md",
        "REGIME_ENGINE.md",
        "COIN_RANKING.md",
        "TOKEN_FUNDAMENTALS.md",
        "PORTFOLIO_ALLOCATOR.md",
        "RISK_MANAGEMENT.md",
        "BITVAVO_EXECUTION.md",
        "LIVE_RUNBOOK.md",
        "RECONCILIATION.md",
        "CAPITAL_SCALING.md",
        "INCIDENT_RESPONSE.md",
    }
    execution_capabilities = dict(execution.get("capabilities") or {})
    live_execution_implemented = bool(execution_capabilities) and all(
        value is True for value in execution_capabilities.values()
    )
    reference_implementation_checks = (
        "all_nine_native_integrations_complete",
        "all_reference_call_sites_and_tests_present",
        "all_reference_usage_counts_positive",
        "architecture_ownership_explicit",
        "canonical_cost_model",
        "canonical_execution_replay_deterministic",
        "canonical_pit_dataset_and_purged_model_pipeline",
        "dashboard_health_read_model",
        "exact_native_validation_authoritative",
        "labels_strictly_separated_from_features",
        "lookahead_tests_exist",
        "ml_authority_evidence_gated",
        "ml_dataset_contract_immutable_content_addressed",
        "model_registry_exists",
        "no_authority_or_risk_increase",
        "no_duplicate_financial_truth_added",
        "no_fill_invented",
        "no_trade_forced",
        "one_primary_responsibility_each",
        "portfolio_target_layer_functions",
        "reference_directories_unchanged",
        "reference_failures_isolated",
        "risk_engine_highest_authority",
        "nine_references_exact_commit_tree_and_license",
        "stage0_vectorized",
        "startup_warmup_tests_exist",
        "strategy_to_buy_order_bypass_impossible",
        "walk_forward_chronological_purged",
    )
    reference_acceptance = dict(reference_integration.get("acceptance") or {})
    reference_health = list(reference_integration.get("reference_health") or [])
    reference_integration_implemented = bool(
        len(reference_health) == 9
        and all(row.get("healthy") is True for row in reference_health)
        and all(
            reference_acceptance.get(check) is True
            for check in reference_implementation_checks
        )
    )

    def row(component: str, complete: bool, evidence: str, next_action: str) -> dict[str, str]:
        return {
            "component": component,
            "status": "COMPLETE" if complete else "PARTIAL_OR_MISSING",
            "evidence": evidence,
            "next_action": "" if complete else next_action,
        }

    return [
        row(
            "repository_audit",
            repository.get("parse_failure_count") == 0,
            f"{repository.get('python_file_count')} Python files parsed",
            "Repair parse failures and rerun system audit.",
        ),
        row(
            "strategy_classification",
            int(strategy.get("deduplicated_research_variant_count") or 0) > 0,
            (
                f"{strategy.get('deduplicated_research_variant_count')} "
                "deduplicated research variants classified"
            ),
            "Map remaining unclassified research strategies to economic families.",
        ),
        row(
            "live_execution",
            live_execution_implemented,
            (
                f"{sum(value is True for value in execution_capabilities.values())}/"
                f"{len(execution_capabilities)} capabilities; "
                f"operational_status={blockers.get('status')}"
            ),
            "Implement every missing execution capability invariant.",
        ),
        row(
            "central_risk",
            bool(risk.get("all_mandatory_safety_modes_enforced")),
            "spot/no-margin/no-leverage/no-short/no-withdrawal controls",
            "Restore all mandatory fail-closed safety modes.",
        ),
        row(
            "data_pipeline",
            (
                data.get("continuous_service", {}).get("process_running") is True
                and data.get("continuous_sync", {}).get("synthetic_fallback") is not True
            ),
            (
                f"sync={data.get('continuous_sync', {}).get('status')}, "
                f"service_running={data.get('continuous_service', {}).get('process_running')}"
            ),
            "Start or repair the continuous real-data synchronization service.",
        ),
        row(
            "formal_regime_cli",
            "regime" in cli and {"status", "explain"} <= cli,
            "main.py regime status/explain",
            "Add causal regime build and history commands.",
        ),
        row(
            "formal_coin_ranking_cli",
            "ranking" in cli,
            "main.py ranking command" if "ranking" in cli else "top50 universe exists",
            "Add main.py ranking build/inspect with transparent subscores.",
        ),
        row(
            "token_fundamentals",
            "tokenomics" in cli,
            "main.py tokenomics command" if "tokenomics" in cli else "not registered",
            "Implement point-in-time tokenomics refresh and inspect commands.",
        ),
        row(
            "system_audit_cli",
            "system" in cli,
            "main.py system audit/architecture",
            "Register the canonical system command.",
        ),
        row(
            "documentation_set",
            required_docs <= docs,
            f"{len(required_docs & docs)}/{len(required_docs)} required documents present",
            "Create the missing operations and architecture documents.",
        ),
        row(
            "restart_recovery",
            execution.get("runtime", {}).get("process_running") is True,
            "active autonomous-live process and persistent state",
            "Verify startup/task installation and restart reconciliation.",
        ),
        row(
            "reference_integration",
            reference_integration_implemented,
            (
                f"implementation={'COMPLETE' if reference_integration_implemented else 'INCOMPLETE'}, "
                f"evidence_status={reference_integration.get('status')}, "
                f"live_readiness={reference_integration.get('live_readiness')}"
            ),
            "Complete missing reference contracts or repository health checks.",
        ),
    ]


def _architecture_markdown(gaps: list[dict[str, str]], generated_at: str) -> str:
    complete = sum(row["status"] == "COMPLETE" for row in gaps)
    lines = [
        "# Architecture Gap Report",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Completed components: **{complete}/{len(gaps)}**.",
        "",
        "| Component | Status | Evidence | Next action |",
        "|---|---|---|---|",
    ]
    for row in gaps:
        lines.append(
            "| {component} | {status} | {evidence} | {next_action} |".format(
                **{
                    key: str(value).replace("|", "\\|")
                    for key, value in row.items()
                }
            )
        )
    lines.extend(
        [
            "",
            "Architecture implementation completeness is not trading readiness. "
            "Operational and evidence blockers remain authoritative in "
            "live_blocker_report.json and reference_integration_report.json.",
            "",
            "The audit is read-only. It generated and submitted zero orders.",
            "",
        ]
    )
    return "\n".join(lines)


def run_system_audit(settings: Settings) -> dict[str, Any]:
    """Generate the complete sanitized system-audit artifact set."""

    output = settings.paths.reports_dir / "system_audit"
    output.mkdir(parents=True, exist_ok=True)
    repository = _repository_inventory(settings)
    strategies, family_map, duplicates = _strategy_inventory(settings)
    execution, blockers = _execution_and_blockers(
        settings,
        strategy_inventory=strategies,
    )
    risk = _risk_report(settings)
    data = _data_report(settings)
    reference_integration = build_reference_integration_health(
        settings.paths.project_root
    )["payload"]
    generated_at = utc_iso()
    gaps = _gap_rows(
        repository=repository,
        strategy=strategies,
        execution=execution,
        blockers=blockers,
        risk=risk,
        data=data,
        reference_integration=reference_integration,
        root=settings.paths.project_root,
    )
    artifacts: dict[str, Any] = {
        "repository_inventory.json": repository,
        "strategy_inventory.json": strategies,
        "strategy_family_map.json": family_map,
        "duplicate_strategy_report.json": duplicates,
        "execution_capability_report.json": execution,
        "live_blocker_report.json": blockers,
        "risk_control_report.json": risk,
        "data_dependency_report.json": data,
        "reference_integration_report.json": reference_integration,
    }
    for filename, payload in artifacts.items():
        atomic_write_json(output / filename, payload)
    (output / "architecture_gap_report.md").write_text(
        _architecture_markdown(gaps, generated_at),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "COMPLETE",
        "live_status": blockers["status"],
        "artifact_count": len(AUDIT_FILENAMES),
        "artifacts": {
            filename: {
                "path": str((output / filename).resolve()),
                "sha256": sha256_file(output / filename),
            }
            for filename in AUDIT_FILENAMES
        },
        "architecture_components_complete": sum(
            row["status"] == "COMPLETE" for row in gaps
        ),
        "architecture_component_count": len(gaps),
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_requests": 0,
        "secrets_serialized": False,
    }
    manifest["manifest_hash"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"},
        length=64,
    )
    atomic_write_json(output / "audit_manifest.json", manifest)
    return {
        **manifest,
        "output_directory": str(output.resolve()),
        "strategy_counts": {
            key: strategies[key]
            for key in (
                "registered_implementation_count",
                "economic_candidate_count",
                "deduplicated_research_variant_count",
                "live_eligible_family_count",
                "live_validated_family_count",
            )
        },
        "strategy_family_count": family_map["family_count"],
        "exact_duplicate_cluster_count": duplicates[
            "exact_dna_duplicate_cluster_count"
        ],
        "semantic_review_cluster_count": duplicates[
            "semantic_review_cluster_count"
        ],
        "live_blocker_counts": blockers["category_counts"],
        "overall_live_blocker_count": blockers["overall_blocker_count"],
    }


def system_architecture_status(settings: Settings) -> dict[str, Any]:
    """Return the latest architecture audit, generating it when absent."""

    directory = settings.paths.reports_dir / "system_audit"
    manifest_path = directory / "audit_manifest.json"
    if not manifest_path.is_file():
        return run_system_audit(settings)
    manifest = _json_mapping(manifest_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": manifest.get("status"),
        "live_status": manifest.get("live_status"),
        "generated_at": manifest.get("generated_at"),
        "architecture_components_complete": manifest.get(
            "architecture_components_complete"
        ),
        "architecture_component_count": manifest.get(
            "architecture_component_count"
        ),
        "report": str((directory / "architecture_gap_report.md").resolve()),
        "manifest": str(manifest_path.resolve()),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "AUDIT_FILENAMES",
    "run_system_audit",
    "system_architecture_status",
]
