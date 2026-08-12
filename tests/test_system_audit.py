from __future__ import annotations

import json
import os
from pathlib import Path

from config.settings import PathSettings, Settings
from reporting.system_audit import AUDIT_FILENAMES, run_system_audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_system_audit_is_complete_sanitized_and_orderless(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    paths = PathSettings(project_root=tmp_path)
    settings = isolated_settings.model_copy(update={"paths": paths})
    (tmp_path / "core").mkdir(parents=True)
    (tmp_path / "execution").mkdir(parents=True)
    (tmp_path / "risk").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "core" / "cli.py").write_text(
        "\n".join(
            (
                'commands.add_parser("system")',
                'commands.add_parser("audit")',
                'commands.add_parser("architecture")',
                'commands.add_parser("regime")',
                'commands.add_parser("status")',
                'commands.add_parser("explain")',
                "submit_level_2_buy_atomically",
                "manager.assess_entry(live_mode=True)",
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "execution" / "execution.py").write_text(
        """
from decimal import Decimal
class BitvavoSpotClient:
    def idempotency(self): ...
    def reconcile(self): ...
    def open_orders(self): ...
    def recent_orders(self): ...
    def cancel(self): ...
    def balance(self): ...
    def fill(self): ...
    client_order_id = "client"
ORDER_STATE_UNKNOWN = "ORDER_STATE_UNKNOWN"
UNKNOWN_ORDER_LOOKUP_FAILED = "UNKNOWN_ORDER_LOOKUP_FAILED"
CANCEL_REQUESTED = "CANCEL_REQUESTED"
CANCEL_STATE_UNKNOWN = "CANCEL_STATE_UNKNOWN"
CANCEL_RESOLVED = "CANCEL_RESOLVED"
UNKNOWN_CANCELLATION_STATE = "UNKNOWN_CANCELLATION_STATE"
CANCELLATION_PARTIAL_FILL_STILL_OPEN = "CANCELLATION_PARTIAL_FILL_STILL_OPEN"
def record_order_fill_progress(): ...
def idempotency_guard(): raise RuntimeError("duplicate live order intent")
PARTIALLY_FILLED_PROGRESS = "PARTIALLY_FILLED_PROGRESS"
cumulative_quantity = "cumulative_quantity"
ORDER_STATUS_OBSERVED = "ORDER_STATUS_OBSERVED"
venue_cumulative_guard = "venue cumulative fill regressed"
LIMIT = "LIMIT"
MARKET = "MARKET"
""",
        encoding="utf-8",
    )
    (tmp_path / "core" / "live_capital.py").write_text(
        "_ledger_pending_buy_reservations CANONICAL_LEDGER "
        "PENDING_ORDER_EXPOSURE_UNRECONCILED "
        "ledger_recovered_pending_exposure_eur "
        "private_order_identifiers_serialized "
        "LiveEntryReservation submit_level_2_buy_atomically "
        "LIVE_ENTRY_RESERVATION_BUSY capital_level_2_capacity LK_NBLCK "
        "RR_PRIMARY replacing_source MAXIMUM_MANAGED_POSITIONS "
        "MANAGED_POSITION_LIMIT_REACHED "
        "MAXIMUM_TOTAL_MANAGED_EXPOSURE_EUR "
        "MANAGED_EXPOSURE_LIMIT_REACHED",
        encoding="utf-8",
    )
    (tmp_path / "core" / "autonomous_trading.py").write_text(
        "submit_level_2_buy_atomically maximum_live_risk_per_trade_eur "
        "LIVE_MAX_RISK_PER_TRADE_EUR_EXCEEDED",
        encoding="utf-8",
    )
    (tmp_path / "core" / "generated_strategy_live.py").write_text(
        'submit_level_2_buy_atomically replacing_source="GENERATED_DNA" '
        "maximum_live_risk_per_trade_eur MAXIMUM_RISK_PER_TRADE_EXCEEDED "
        'LIVE_ACCOUNT_HEALTH_STALE candidate.get("stale")',
        encoding="utf-8",
    )
    (tmp_path / "core" / "event_driven_live.py").write_text(
        "submit_level_2_buy_atomically MAXIMUM_RISK_PER_TRADE_EUR "
        "MAXIMUM_RISK_PER_TRADE_EXCEEDED",
        encoding="utf-8",
    )
    (tmp_path / "risk" / "risk_manager.py").write_text(
        "self.kill_switch.active KILL_SWITCH_ACTIVE maximum_daily_loss "
        "DAILY_LOSS_LIMIT maximum_portfolio_drawdown DRAWDOWN_LIMIT "
        "snapshot.data_healthy DATA_UNHEALTHY snapshot.reconciled "
        "RECONCILIATION_REQUIRED",
        encoding="utf-8",
    )
    (tmp_path / "risk" / "canary_guard.py").write_text(
        "spot_only",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_smoke.py").write_text(
        "def test_smoke(): assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "crypto-references" / "vendor").mkdir(parents=True)
    (tmp_path / "crypto-references" / "vendor" / "foreign_fixture.py").write_text(
        "def foreign_syntax(\n",
        encoding="utf-8",
    )
    _write_json(
        paths.output_dir / "strategies" / "all_strategy_dna.json",
        {
            "economic_evidence": [
                {
                    "strategy_id": "TREND_EXACT",
                    "strategy_dna": "a" * 64,
                    "strategy_family": "TREND",
                    "timeframe": "4h",
                    "markets": ["BTC-EUR"],
                    "backtest_positive": True,
                    "paper_active": True,
                    "lifecycle_state": "LIVE_VALIDATED",
                    "frozen": True,
                }
            ],
            "registered_pending": [],
        },
    )
    _write_json(
        paths.output_dir / "live" / "autonomous_live.lock",
        {"pid": os.getpid()},
    )
    _write_json(
        paths.output_dir / "live" / "heartbeat.json",
        {
            "pid": os.getpid(),
            "control_state": "ENABLED",
            "private_account_websocket": {
                "state": "AUTHENTICATED",
                "ready_for_new_entries": True,
                "secrets_serialized": False,
            },
        },
    )
    _write_json(
        paths.output_dir / "operations" / "live_account_health.json",
        {
            "status": "READY",
            "reconciliation": {
                "healthy": True,
                "local_open_orders": 0,
                "remote_open_orders": 0,
                "reason_codes": ["RECONCILED"],
            },
        },
    )
    _write_json(
        paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json",
        {
            "active": True,
            "approved_candidates": [{"strategy_dna": "a" * 64}],
            "maximum_order_eur": "5",
            "maximum_total_exposure_eur": "15",
            "maximum_open_positions": 3,
            "maximum_new_orders_per_day": 3,
            "maximum_one_position_per_market": True,
            "maximum_one_position_per_strategy_dna": True,
            "spot_only": True,
            "long_only": True,
            "autoscale": False,
            "withdrawals_available": False,
        },
    )
    _write_json(
        paths.output_dir / "notifications" / "telegram_status.json",
        {
            "status": "ENABLED",
            "enabled": True,
            "active_queue_size": 0,
            "secrets_redacted": True,
        },
    )
    _write_json(
        paths.output_dir / "research" / "data_sync_progress.json",
        {
            "status": "RUNNING",
            "phase": "FETCHING_PROVIDER_HISTORY",
            "total_operations": 10,
            "completed_operations": 5,
            "failure_count": 0,
            "synthetic_fallback": False,
        },
    )
    _write_json(
        paths.checkpoints_dir / "data_service.lock",
        {
            "owner": {
                "pid": os.getpid(),
                "mode": "research",
                "service_id": "continuous-data-service",
            }
        },
    )
    _write_json(
        paths.checkpoints_dir / "continuous-data-service_heartbeat.json",
        {
            "pid": os.getpid(),
            "status": "RUNNING",
        },
    )
    _write_json(
        paths.output_dir / "universe" / "top50_current.json",
        {"rows": [{"symbol": f"C{index}"} for index in range(50)]},
    )
    paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    (paths.processed_data_dir / "BTC-EUR_4h.parquet").write_bytes(b"fixture")
    paths.database_path.parent.mkdir(parents=True, exist_ok=True)
    paths.database_path.write_bytes(b"sqlite")

    result = run_system_audit(settings)

    assert result["status"] == "COMPLETE"
    assert result["live_status"] == "LIVE_ACTIVE"
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0
    assert result["secrets_serialized"] is False
    assert result["artifact_count"] == len(AUDIT_FILENAMES)
    assert result["strategy_counts"] == {
        "registered_implementation_count": 0,
        "economic_candidate_count": 1,
        "deduplicated_research_variant_count": 1,
        "live_eligible_family_count": 0,
        "live_validated_family_count": 1,
    }
    directory = paths.reports_dir / "system_audit"
    assert all((directory / filename).is_file() for filename in AUDIT_FILENAMES)
    manifest = json.loads(
        (directory / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["manifest_hash"]) == 64
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"].values())
    blocker_report = json.loads(
        (directory / "live_blocker_report.json").read_text(encoding="utf-8")
    )
    assert blocker_report["blockers"] == []
    assert blocker_report["overall_blocker_count"] == 0
    assert blocker_report["technical_runtime_status"] == "READY"
    assert all(count == 0 for count in blocker_report["category_counts"].values())
    repository_report = json.loads(
        (directory / "repository_inventory.json").read_text(encoding="utf-8")
    )
    assert repository_report["parse_failure_count"] == repository_report[
        "production_scope"
    ]["parse_failure_count"]
    assert repository_report["reference_scope"]["parse_failure_count"] == 1
    assert repository_report["reference_scope"]["affects_production_health"] is False
    assert repository_report["reference_scope"]["inventory_mode"] == (
        "PATH_HASH_PARSE_STATUS_ONLY"
    )
    assert repository_report["reference_scope"]["symbol_inventory_collected"] is False
    assert "functions" not in repository_report["reference_scope"]["modules"][0]
    assert all(
        not row["path"].startswith("crypto-references/")
        for row in repository_report["modules"]
    )
    family_report = json.loads(
        (directory / "strategy_family_map.json").read_text(encoding="utf-8")
    )
    trend_family = next(
        row
        for row in family_report["families"]
        if row["family_id"] == "time_series_momentum"
    )
    assert trend_family["positive_count"] == 1
    assert trend_family["backtest_positive_count"] == 1
    assert trend_family["paper_active_count"] == 1
    assert trend_family["live_validated_count"] == 1
    assert trend_family["evidence_tier_counts"] == {"LIVE_VALIDATED": 1}
    assert "deprecated alias" in family_report["count_semantics"][
        "positive_count"
    ]
    capability_report = json.loads(
        (directory / "execution_capability_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert capability_report["capabilities"][
        "unknown_order_state_recovery"
    ] is True
    assert capability_report["capabilities"][
        "unknown_cancellation_state_recovery"
    ] is True
    assert capability_report["capabilities"][
        "incremental_partial_fill_accounting"
    ] is True
    assert capability_report["capabilities"][
        "pending_buy_exposure_reserved_across_restart"
    ] is True
    assert capability_report["capabilities"][
        "atomic_cross_engine_buy_reservation"
    ] is True
    assert capability_report["capabilities"][
        "all_autonomous_buy_routes_atomic"
    ] is True
    assert capability_report["runtime"]["telegram"]["status_scope"] == (
        "CURRENT_STATUS"
    )
    assert capability_report["runtime"]["telegram"][
        "historical_findings_are_current_status"
    ] is False
    risk_report = json.loads(
        (directory / "risk_control_report.json").read_text(encoding="utf-8")
    )
    assert all(risk_report["controls"].values())
    assert risk_report["all_mandatory_safety_modes_enforced"] is True

    registry_path = paths.output_dir / "strategies" / "all_strategy_dna.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["economic_evidence"][0]["lifecycle_state"] = "PAPER_ACTIVE"
    _write_json(registry_path, registry)
    blocked = run_system_audit(settings)
    blocker_report = json.loads(
        (directory / "live_blocker_report.json").read_text(encoding="utf-8")
    )
    assert blocked["live_status"] == "LIVE_BLOCKED"
    assert blocker_report["technical_runtime_status"] == "READY"
    assert blocker_report["category_counts"][
        "technical_runtime_blockers"
    ] == 0
    assert blocker_report["category_counts"][
        "strategy_evidence_blockers"
    ] == 1
    assert blocker_report["overall_blocker_count"] == 1
    architecture = (directory / "architecture_gap_report.md").read_text(
        encoding="utf-8"
    )
    assert "operational_status=LIVE_BLOCKED" in architecture
    assert "Architecture implementation completeness is not trading readiness" in (
        architecture
    )
