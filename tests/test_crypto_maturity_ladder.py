from __future__ import annotations

import json
from pathlib import Path

from core.cli import build_parser
from reporting.crypto_beginner_foundation import build_beginner_foundation
from reporting.crypto_level_certification import build_level_certification
from reporting.crypto_maturity_ladder import (
    LEVEL_RANGES,
    PROJECT_NAMES,
    build_maturity_ladder,
)
from ui.server import HTML_DOCUMENT


def _write(path: Path, payload: object | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_ladder_has_33_execution_early_projects_and_never_skips_levels(tmp_path: Path) -> None:
    ladder = build_maturity_ladder(tmp_path)

    assert len(PROJECT_NAMES) == 33
    assert [item["project_id"] for item in ladder["projects"]] == list(range(1, 34))
    assert ladder["projects"][8]["name"].startswith("Execution foundation")
    assert ladder["projects"][12]["name"].startswith("Tiny live canary")
    assert ladder["levels"][0]["status"] == "ACTIVE"
    assert all(
        level["status"] == "WAITING_FOR_PREVIOUS_LEVEL"
        for level in ladder["levels"][1:]
    )
    assert ladder["live_canary_enabled"] is False
    assert ladder["certification_scope"] == "IMPLEMENTATION_AND_BOUNDED_TESTS_ONLY"
    assert ladder["implementation_certification_grants_live_authority"] is False
    assert ladder["operational_readiness"]["autonomous_live_status"] == "BLOCKED"


def test_maturity_refresh_cli_is_registered() -> None:
    args = build_parser().parse_args(["system", "maturity"])

    assert args.system_command == "maturity"


def test_expert_implementation_never_overrides_current_live_blockers(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "output" / "roadmap"
    for level in ("BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"):
        filename = (
            "beginner_foundation.json"
            if level == "BEGINNER"
            else f"{level.lower()}_certification.json"
        )
        _write(
            roadmap / filename,
            {
                "status": f"{level}_CERTIFIED",
                "project_gates": [
                    {"project_id": project_id, "passed": True}
                    for project_id in range(1, 34)
                    if project_id in LEVEL_RANGES[level]
                ],
            },
        )
    _write(
        tmp_path / "output/operations/live_account_health.json",
        {
            "status": "BLOCKED",
            "entry_allowed": False,
            "execution_authority": {"control_state": "PAUSED"},
        },
    )
    _write(
        tmp_path / "output/operations/external_inventory_remediation.json",
        {"status": "OPERATOR_DECISION_REQUIRED"},
    )
    _write(
        tmp_path / "output/economics/latest.json",
        {"best_validated_family": None},
    )

    ladder = build_maturity_ladder(tmp_path)

    assert ladder["implementation_complete"] is True
    assert ladder["current_level"] == "COMPLETE"
    assert ladder["current_operational_stage"] == "LIVE_VALIDATION_BLOCKED"
    assert ladder["operational_readiness"]["autonomous_live_blockers"] == [
        "LIVE_ACCOUNT_ENTRY_NOT_READY",
        "AUTONOMOUS_CONTROL_NOT_RUNNING",
        "EXTERNAL_INVENTORY_DECISION_REQUIRED",
        "NO_FINANCIALLY_VALIDATED_STRATEGY_FAMILY",
    ]


def test_beginner_builder_certifies_only_projects_one_to_four(tmp_path: Path) -> None:
    _write(
        tmp_path / "output/multi_source/status.json",
        {
            "observed_at": "2026-08-11T00:00:00Z",
            "pid": 42,
            "platform_schema_version": "multi_source_platform_v1",
            "book_coverage": {"BTC": {}, "ETH": {}, "SOL": {}},
            "readiness": {},
            "stream_health": {},
            "source_status": {},
            "execution": {"orders_generated": 0},
        },
    )
    _write(
        tmp_path / "output/reports/system_audit/data_dependency_report.json",
        {
            "database_present": True,
            "market_count": 3,
            "configured_timeframes": ["1h", "4h", "1d"],
        },
    )
    _write(tmp_path / "data/market_data.py", "def validate_ohlcv(): pass\n")
    _write(tmp_path / "tests/test_market_data.py", "def test_quality(): pass\n")
    _write(tmp_path / "tests/test_crypto_performance.py", "def test_analyzer(): pass\n")
    _write(tmp_path / "research/crypto_performance.py", "# analyzer\n")
    _write(
        tmp_path / "ui/server.py",
        'paths = ["/api/snapshot", "/api/candles", "/health"]\nid="maturity"\n',
    )

    artifact, ladder = build_beginner_foundation(tmp_path)

    assert artifact["status"] == "BEGINNER_CERTIFIED"
    assert all(item["passed"] for item in artifact["project_gates"])
    assert all(item["status"] == "CERTIFIED" for item in ladder["projects"][:4])
    assert ladder["levels"][0]["status"] == "CERTIFIED"
    assert ladder["levels"][1]["status"] == "ACTIVE"
    assert all(item["status"] != "CERTIFIED" for item in ladder["projects"][4:])
    assert artifact["side_effects"]["orders_submitted"] == 0
    assert (tmp_path / "output/roadmap/beginner_foundation.json").exists()


def test_dashboard_contains_read_only_maturity_panel() -> None:
    assert 'id="maturity"' in HTML_DOCUMENT
    assert "snapshot.crypto_maturity" in HTML_DOCUMENT
    assert "/api/order" not in HTML_DOCUMENT


def test_higher_level_certification_is_blocked_without_previous_level(tmp_path: Path) -> None:
    validation = {"passed": True, "return_code": 0, "test_files": []}

    artifact, ladder = build_level_certification(
        tmp_path,
        "ADVANCED",
        validation=validation,
    )

    assert artifact["status"] == "ADVANCED_BLOCKED"
    assert artifact["previous_level_certified"] is False
    assert ladder["levels"][0]["status"] == "ACTIVE"
    assert ladder["levels"][2]["status"] == "WAITING_FOR_PREVIOUS_LEVEL"
