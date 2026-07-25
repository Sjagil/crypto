from __future__ import annotations

import json

import main
from core.contracts import ResearchStatus
from utils.common import atomic_write_json, sha256_file


def output_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_version_and_doctor_commands(capsys) -> None:
    assert main.main(["version"]) == 0
    assert output_json(capsys)["version"] == "1.0.0"
    assert main.main(["doctor"]) == 0
    doctor = output_json(capsys)
    assert doctor["research_ready"]
    assert not doctor["safety"]["withdrawals_enabled"]


def test_unknown_eligibility_and_live_status_fail_closed(capsys) -> None:
    assert main.main(["eligibility", "check", "UNKNOWN-EUR"]) == 3
    assert output_json(capsys)["status"] == "REVIEW_REQUIRED"
    assert main.main(["live", "status"]) == 0
    status = output_json(capsys)
    assert not status["live_ready"]
    assert status["failures"]


def test_required_command_families_are_registered() -> None:
    parser = main.build_parser()
    for arguments in (
        ["config", "validate"],
        ["eligibility", "check", "--market", "BTC-EUR"],
        ["data", "providers"],
        ["scrape", "status"],
        ["features", "build"],
        ["strategies", "list"],
        ["backtest"],
        ["optimize"],
        ["walk-forward"],
        ["monte-carlo"],
        ["research"],
        ["paper", "status"],
        ["live", "preflight"],
        ["operate", "preflight", "--mode", "shadow"],
        ["operate", "start", "--mode", "shadow", "--soak-minutes", "15"],
        ["operate", "drain", "--mode", "shadow", "--wait-seconds", "30"],
        ["operate", "stop", "--mode", "shadow", "--wait-seconds", "30"],
        ["operate", "candidates"],
        ["operate", "task-install", "--dry-run"],
        ["lab", "campaign", "plan", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "plan", "--name", "institutional-rotation-v2"],
        ["lab", "campaign", "plan", "--name", "capital-utilization-v1"],
        ["lab", "campaign", "plan", "--name", "diversified-rotation-v1"],
        ["lab", "campaign", "plan", "--name", "portfolio-breakout-v1"],
        ["lab", "campaign", "plan", "--name", "absolute-momentum-v1"],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "portfolio-storm-v1",
            "--storm-trials",
            "5000",
        ],
        ["lab", "campaign", "forward", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "external", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "audit", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "observe", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "package", "--name", "cross-sectional-ensemble"],
        ["lab", "campaign", "autopilot"],
        ["lab", "campaign", "autopilot", "--mode", "status"],
        ["lab", "campaign", "autopilot", "--skip-feature-store"],
        [
            "lab",
            "campaign",
            "autopilot",
            "--mode",
            "task-install",
            "--dry-run",
        ],
    ):
        assert parser.parse_args(arguments).command

    full = parser.parse_args(
        [
            "research",
            "--providers",
            "bitvavo,kraken,coinmarketcap",
            "--scrapers",
            "all",
            "--markets",
            "BTC-EUR,ETH-EUR,SOL-EUR,LINK-EUR",
            "--timeframes",
            "1h,4h,1d",
            "--strategies",
            "all",
            "--profile",
            "standard",
            "--capital",
            "2000",
            "--risk-per-trade",
            "0.005",
            "--fee",
            "0.0025",
            "--slippage-bps",
            "8",
            "--walk-forward-folds",
            "6",
            "--bootstrap-samples",
            "100",
            "--monte-carlo-runs",
            "100",
            "--output-dir",
            "output/research/test",
        ]
    )
    assert full.command == "research"


def test_research_report_requires_matching_manifest(tmp_path) -> None:
    report = tmp_path / "research_summary.json"
    atomic_write_json(
        report,
        {
            "status": "PAPER_CANDIDATE",
            "passed": True,
            "lookahead_safe": True,
            "repainting_safe": True,
        },
    )
    manifest = tmp_path / "research_manifest.json"
    atomic_write_json(
        manifest,
        {
            "run_kind": "research",
            "artifacts": [{"path": str(report.resolve()), "sha256": sha256_file(report)}],
        },
    )
    assert main.research_status_from_report(str(report)) == (
        ResearchStatus.PAPER_CANDIDATE,
        True,
    )
    report.write_text("{}", encoding="utf-8")
    assert main.research_status_from_report(str(report)) == (
        ResearchStatus.LIVE_BLOCKED,
        False,
    )
