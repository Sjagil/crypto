from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from pydantic import SecretStr

import main
from config.settings import PathSettings, Settings
from core.cli import (
    _parse_utc_datetime,
    _safe_exception_message,
    _task_xml,
    command_test,
)
from reporting.reports import write_operational_reports
from reporting.visualizations import VisualizationReporter
from research.features import FeaturePipeline
from research.macro_context import MacroSourceSpec
from utils.common import AlertThrottle, configure_logging


def test_jsonl_logs_and_secret_redaction(tmp_path) -> None:
    secret = "configured-unit-secret"
    text_path = tmp_path / "application.log"
    jsonl_path = tmp_path / "application.jsonl"
    logger = configure_logging(
        level=logging.INFO,
        log_file=text_path,
        jsonl_file=jsonl_path,
        secrets=(secret,),
    )
    logger.info(
        "request used %s",
        secret,
        extra={
            "run_id": "run",
            "component": "provider",
            "provider": "unit",
            "market": "BTC-EUR",
            "timeframe": "1h",
            "operation": "download",
            "duration": 0.1,
            "status": "PASSED",
            "reason_code": "OK",
            "exception_type": None,
            "retry_number": 0,
            "correlation_id": "correlation",
        },
    )
    payload = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert secret not in jsonl_path.read_text(encoding="utf-8")
    assert secret not in text_path.read_text(encoding="utf-8")
    assert payload["run_id"] == "run"
    assert payload["provider"] == "unit"
    assert payload["message"] == "request used ***REDACTED***"


def test_provider_exception_url_is_redacted(isolated_settings: Settings) -> None:
    providers = isolated_settings.providers.model_copy(
        update={"fred_api_key": SecretStr("credential-in-query")}
    )
    settings = isolated_settings.model_copy(update={"providers": providers})
    rendered = _safe_exception_message(
        RuntimeError("https://example.test?api_key=credential-in-query"),
        settings,
    )
    assert "credential-in-query" not in rendered
    assert "***REDACTED***" in rendered


def test_parse_utc_datetime_normalizes_naive_and_offset_values() -> None:
    naive = _parse_utc_datetime("2026-07-24T22:00:00")
    offset = _parse_utc_datetime("2026-07-25T00:00:00+02:00")
    assert naive == datetime(2026, 7, 24, 22, tzinfo=UTC)
    assert offset == naive


def test_operational_reports_alert_throttling_and_windows_task(
    tmp_path,
    isolated_settings: Settings,
) -> None:
    paths = write_operational_reports(
        tmp_path / "operations",
        status={
            "service_state": "IDLE_NO_APPROVED_CANDIDATE",
            "mode": "shadow",
            "active_candidate": None,
            "provider_health": [],
            "latest_signals": [],
        },
    )
    assert len(paths) == 6
    assert all(path.is_file() for path in paths.values())
    failures: list[str] = []

    def broken_delivery(event_type, payload):
        del event_type, payload
        failures.append("called")
        raise RuntimeError("optional delivery unavailable")

    alerts = AlertThrottle(
        state_path=tmp_path / "alerts.json",
        audit_path=tmp_path / "alerts.jsonl",
        cooldown_seconds=60,
        secrets=("unit-secret",),
        delivery=broken_delivery,
    )
    now = datetime.now(UTC)
    assert alerts.send("KILL_SWITCH", {"token": "unit-secret"}, now=now)
    assert not alerts.send("KILL_SWITCH", {"token": "unit-secret"}, now=now)
    assert failures == ["called"]
    assert "unit-secret" not in (tmp_path / "alerts.jsonl").read_text(
        encoding="utf-8"
    )
    xml = _task_xml(
        isolated_settings,
        mode="shadow",
        profile="practical_spot_v1",
    )
    assert ".venv" in xml
    assert "<WorkingDirectory>" in xml
    assert "--mode shadow" in xml
    assert "--mode live" not in xml


def test_chart_files_and_index_generation(tmp_path) -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    reporter = VisualizationReporter(tmp_path)
    result = reporter.generate(
        {
            "research": pd.DataFrame(
                {
                    "equity": range(20),
                    "drawdown": [-value / 100 for value in range(20)],
                    "rolling_return": range(20),
                    "rolling_volatility": range(20),
                    "rolling_sharpe": range(20),
                },
                index=index,
            )
        }
    )
    assert result["passed"] >= 6
    assert result["failed"] == 0
    assert (tmp_path / "equity_curve.png").is_file()
    assert (tmp_path / "index.json").is_file()
    parsed = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert any(item["name"] == "equity_curve" for item in parsed["plots"])


def test_feature_pipeline_optional_macro_interface(ohlcv: pd.DataFrame) -> None:
    index = ohlcv.index
    macro = pd.DataFrame({"value": range(len(index))}, index=index)
    specs = {
        "sentiment": MacroSourceSpec(
            provider="unit",
            source_frequency="1h",
            expected_cadence=timedelta(hours=1),
            maximum_age=timedelta(hours=2),
            units={"value": "index"},
        )
    }
    features = FeaturePipeline().build(
        ohlcv,
        market="BTC-EUR",
        macro_context={
            "fear_greed": macro,
            "source_specs": specs,
        },
    )
    assert "sentiment_fear_greed" in features
    assert features.attrs["feature_knowability"]["sentiment_fear_greed"][
        "lookahead_safe"
    ]


def test_complete_test_run_artifact_tree_and_statuses(
    isolated_settings: Settings, tmp_path, capsys
) -> None:
    settings = isolated_settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )
    args = argparse.Namespace(test_mode="reporting", duration=0)
    assert asyncio.run(command_test(args, settings)) == 0
    output = json.loads(capsys.readouterr().out)
    root = settings.paths.test_runs_dir / output["run_id"]
    expected = {
        "summary.json",
        "provider_status.json",
        "data_quality.json",
        "websocket_health.json",
        "orderbook_health.json",
        "database_health.json",
        "scraper_status.json",
        "macro_context_status.json",
        "gex_status.json",
        "risk_status.json",
        "paper_execution_status.json",
        "test_results.json",
        "statistics.json",
        "secret_audit.json",
    }
    assert expected.issubset({path.name for path in root.iterdir()})
    assert all(
        (root / name).is_dir()
        for name in ("logs", "charts", "csv", "html", "manifests")
    )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] in {"PASSED", "PARTIAL"}
    provider = json.loads(
        (root / "provider_status.json").read_text(encoding="utf-8")
    )
    assert provider["status"] == "SKIPPED"


def test_extended_cli_parsing() -> None:
    parser = main.build_parser()
    commands = (
        ["providers", "list"],
        ["providers", "test"],
        ["data", "historical"],
        ["data", "live"],
        ["data", "reconcile"],
        ["data", "database-health"],
        ["websocket", "run"],
        ["websocket", "status"],
        ["websocket", "soak"],
        ["orderbook", "snapshot"],
        ["orderbook", "stream"],
        ["orderbook", "inspect"],
        ["macro", "build"],
        ["macro", "inspect"],
        ["gex", "collect"],
        ["gex", "inspect"],
        ["positions", "status"],
        ["positions", "reconcile"],
        ["positions", "pnl"],
        ["risk", "correlation"],
        ["risk", "drawdown"],
        ["risk", "kill-switch-status"],
        ["report", "statistics"],
        ["report", "charts"],
        ["report", "full"],
        ["test", "offline"],
        ["test", "network"],
        ["test", "full"],
        ["test", "soak"],
    )
    assert all(parser.parse_args(command).command for command in commands)
