from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import pytest

import main
from config.settings import PathSettings, Settings
from core.cli import command_practical_live, emit
from core.contracts import ResearchStatus
from utils.common import atomic_write_json, sha256_file


def output_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_emit_ascii_fallback_remains_valid_json(monkeypatch) -> None:
    rendered: list[str] = []

    def ascii_only_print(value: Any) -> None:
        text = str(value)
        text.encode("ascii")
        rendered.append(text)

    monkeypatch.setattr("builtins.print", ascii_only_print)
    payload = {"status": "🟡 ACTIEF · geen entry", "currency": "€"}

    emit(payload)

    assert json.loads(rendered[0]) == payload
    assert "\\U" not in rendered[0]
    assert "\\x" not in rendered[0]


def test_version_and_doctor_commands(capsys) -> None:
    assert main.main(["version"]) == 0
    assert output_json(capsys)["version"] == "1.0.0"
    assert main.main(["doctor"]) == 0
    doctor = output_json(capsys)
    assert doctor["research_ready"]
    assert not doctor["safety"]["withdrawals_enabled"]


def test_unknown_eligibility_fails_closed_and_live_status_is_reconciled(
    capsys,
    restrictive_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.cli.Settings.load",
        lambda *_args, **_kwargs: restrictive_settings,
    )
    assert main.main(["eligibility", "check", "UNKNOWN-EUR"]) == 3
    assert output_json(capsys)["status"] == "REVIEW_REQUIRED"
    monkeypatch.undo()
    assert main.main(["live", "status"]) == 0
    status = output_json(capsys)
    assert status["status"] in {
        "LIVE_RUNNING",
        "LIVE_DEGRADED",
        "LIVE_BLOCKED",
        "LIVE_STOPPED",
    }
    if status["live_ready"]:
        assert status["status"] == "LIVE_RUNNING"
        assert status["failures"] == []
    else:
        assert status["failures"]


def test_required_command_families_are_registered() -> None:
    parser = main.build_parser()
    for arguments in (
        ["system", "audit"],
        ["system", "architecture"],
        ["ranking", "build"],
        ["ranking", "inspect", "--asset", "BTC"],
        ["tokenomics", "refresh"],
        ["tokenomics", "inspect", "--asset", "BTC"],
        ["telegram", "health"],
        ["telegram", "test"],
        ["telegram", "status"],
        ["telegram", "announce-autopilot"],
        ["telegram", "clarify-paper-fills"],
        ["telegram", "send-latest-signals"],
        ["signals", "scan"],
        ["daily", "--notifications-only"],
        ["run"],
        ["regime", "status"],
        ["regime", "explain"],
        ["router", "status"],
        ["opportunities", "scan"],
        ["opportunities", "top"],
        ["opportunities", "explain", "--id", "example"],
        ["trading", "status"],
        ["trading", "preflight"],
        ["trading", "run-once"],
        ["trading", "position"],
        ["trading", "close"],
        ["trading", "smoke-canary"],
        ["autopilot", "status"],
        ["autopilot", "run-once"],
        ["autopilot", "start"],
        ["autopilot", "stop"],
        ["autopilot", "task-install"],
        ["autopilot", "task-status"],
        ["autopilot", "task-remove"],
        ["multi-timeframe", "validate-15m"],
        ["multi-timeframe", "validate-limit-overlay"],
        ["live", "start", "--exchange", "bitvavo"],
        ["live", "stop"],
        ["live", "reconcile"],
        ["live", "positions"],
        ["live", "orders", "--limit", "25"],
        [
            "live",
            "inventory-reallocate",
            "--market",
            "TAO-EUR",
            "--approval-reference",
            "test-reference",
            "--target-weight",
            "0.20",
        ],
        ["live", "emergency-stop", "--reason", "operator-test"],
        ["strategies", "top", "--limit", "20"],
        ["capital", "status"],
        [
            "capital",
            "approve-level",
            "--strategy-id",
            "RR_B60_H5_Z20",
            "--level",
            "2",
            "--approval",
            "example",
        ],
        ["config", "validate"],
        ["history", "audit", "--min-years", "7"],
        ["history", "download", "--min-years", "7", "--resume"],
        ["history", "status", "--min-years", "7"],
        ["eligibility", "check", "--market", "BTC-EUR"],
        ["data", "providers"],
        ["data", "status", "--compact"],
        ["microstructure", "plan"],
        ["microstructure", "status"],
        ["microstructure", "data-status"],
        ["microstructure", "observe"],
        ["microstructure", "observer-audit"],
        ["microstructure", "audit"],
        ["microstructure", "readiness-report"],
        [
            "microstructure",
            "gate-check",
            "--stage",
            "technical_feature_validation",
        ],
        ["scrape", "status"],
        ["features", "build"],
        ["strategies", "list"],
        ["backtest"],
        ["optimize"],
        ["walk-forward"],
        ["monte-carlo"],
        ["research"],
        ["research", "backtest-all", "--min-years", "7", "--resume"],
        ["research", "backtest-top30", "--min-years", "7", "--resume"],
        [
            "research",
            "backtest-timeframe",
            "--timeframe",
            "1h",
            "--min-years",
            "7",
            "--resume",
        ],
        ["research", "validate-survivors", "--min-years", "7", "--resume"],
        ["leaderboard", "build", "--window", "seven-year"],
        ["leaderboard", "compare-legacy"],
        ["report", "build", "--scope", "seven-year"],
        ["paper", "status"],
        ["live", "canary-policy"],
        ["live", "preflight"],
        ["operate", "preflight", "--mode", "shadow"],
        ["operate", "start", "--mode", "shadow", "--soak-minutes", "15"],
        ["operate", "drain", "--mode", "shadow", "--wait-seconds", "30"],
        ["operate", "stop", "--mode", "shadow", "--wait-seconds", "30"],
        ["operate", "candidates"],
        ["operate", "task-install", "--dry-run"],
        ["operate", "startup-install", "--mode", "shadow"],
        ["operate", "startup-status", "--mode", "shadow"],
        ["operate", "supervisor-status"],
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
            "volume-strategy-catalog-v1",
        ],
        ["lab", "campaign", "observe", "--name", "absolute-momentum-v1"],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "absolute-momentum-plateau-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "absolute-momentum-plateau-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "lower-timeframe-mtf-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "owned-asset-high-sample-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "volatility-contraction-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "volatility-contraction-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "multi-alpha-ensemble-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "multi-alpha-ensemble-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "trend-pullback-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "trend-pullback-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "range-expansion-4h-v1-1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "range-expansion-4h-v1-1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "sentiment-recovery-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "sentiment-recovery-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "residual-momentum-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "residual-momentum-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "dual-asset-trend-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "dual-asset-trend-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "liquidity-sweep-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "liquidity-sweep-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "residual-reversal-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "residual-reversal-v1",
        ],
        [
            "lab",
            "campaign",
            "plan",
            "--name",
            "macro-liquidity-v1",
        ],
        [
            "lab",
            "campaign",
            "observe",
            "--name",
            "macro-liquidity-v1",
        ],
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
        [
            "lab",
            "run",
            "--once",
            "--data-mode",
            "real",
            "--markets",
            "TAO-EUR,NPC-EUR",
        ],
        ["lab", "trials", "audit"],
        ["live", "approval-candidates", "--timeframe", "1h"],
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


def test_startup_launcher_is_hidden_and_uses_exact_environment(
    isolated_settings,
    tmp_path,
    monkeypatch,
) -> None:
    from core.cli import (
        _startup_launcher,
        _startup_launcher_path,
    )

    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = _startup_launcher_path(isolated_settings)
    launcher = _startup_launcher(
        isolated_settings,
        mode="shadow",
        profile="practical_spot_v1",
    )
    assert path.name == "CryptoPracticalSpotShadow.vbs"
    assert "operate supervise --mode shadow" in launcher
    assert ", 0, False" in launcher


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


@pytest.mark.asyncio
async def test_live_emergency_stop_persists_and_submits_zero_orders(
    isolated_settings: Settings,
    tmp_path,
    capsys,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    result = await command_practical_live(
        Namespace(
            live_command="emergency-stop",
            reason="UNIT_TEST_EMERGENCY",
        ),
        settings,
    )
    assert result == 0
    payload = output_json(capsys)
    assert payload["status"] == "EMERGENCY_STOP_ACTIVE"
    assert payload["new_entries_allowed"] is False
    assert payload["position_monitoring_remains_active"] is True
    assert payload["orders_submitted_by_emergency_stop"] == 0
    state = json.loads(
        (
            settings.paths.checkpoints_dir / "kill_switch.json"
        ).read_text(encoding="utf-8")
    )
    assert state["active"] is True
    assert state["reason"] == "UNIT_TEST_EMERGENCY"


@pytest.mark.asyncio
async def test_live_approval_candidates_is_orderless(
    isolated_settings: Settings,
    tmp_path,
    capsys,
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    result = await command_practical_live(
        Namespace(
            live_command="approval-candidates",
            timeframe="1h",
            limit=10,
        ),
        settings,
    )
    assert result == 0
    payload = output_json(capsys)
    assert payload["auto_approval"] is False
    assert payload["separate_operator_phrase_required_per_dna"] is True
    assert payload["orders_generated"] == 0
    assert payload["orders_submitted"] == 0
