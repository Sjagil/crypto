"""Build immutable evidence that projects 1-4 satisfy the Beginner level."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reporting.crypto_maturity_ladder import build_maturity_ladder
from research.crypto_performance import analyze_crypto_performance


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_beginner_foundation(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = workspace.resolve()
    output_root = workspace / "output"
    status_path = output_root / "multi_source" / "status.json"
    dependency_path = output_root / "reports" / "system_audit" / "data_dependency_report.json"
    status = _read_json(status_path)
    dependency = _read_json(dependency_path)

    index = pd.date_range("2025-01-01", periods=24 * 21, freq="1h", tz="UTC")
    deterministic_returns = 0.00015 + 0.002 * np.sin(np.arange(len(index)) / 11.0)
    deterministic_returns[150] = -0.06
    equity = pd.Series(100.0 * np.cumprod(1.0 + deterministic_returns), index=index)
    analyzer_report = analyze_crypto_performance(equity)

    platform_checks = {
        "runtime_status_present": bool(status.get("observed_at") and status.get("pid")),
        "primary_assets_present": all(
            asset in json.dumps(status.get("book_coverage", {}))
            for asset in ("BTC", "ETH", "SOL")
        ),
        "multi_source_schema_present": bool(status.get("platform_schema_version")),
        "historical_database_present": bool(dependency.get("database_present")),
        "market_count_at_least_three": int(dependency.get("market_count", 0)) >= 3,
        "configured_timeframes_present": len(dependency.get("configured_timeframes", [])) >= 3,
        "collector_generated_no_orders": int(status.get("execution", {}).get("orders_generated", 0)) == 0,
    }
    quality_checks = {
        "causal_validator_present": (workspace / "data" / "market_data.py").exists(),
        "market_data_tests_present": (workspace / "tests" / "test_market_data.py").exists(),
        "readiness_contract_present": isinstance(status.get("readiness"), dict),
        "stream_health_present": isinstance(status.get("stream_health"), dict),
        "source_status_present": isinstance(status.get("source_status"), dict),
    }
    analyzer_checks = {
        "analysis_only": analyzer_report.get("analysis_only") is True,
        "performance_metrics_complete": all(
            key in analyzer_report["performance"]
            for key in (
                "cagr",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "upside_capture",
                "downside_capture",
            )
        ),
        "risk_metrics_complete": all(
            key in analyzer_report["risk"]
            for key in (
                "max_drawdown",
                "value_at_risk_95",
                "expected_shortfall_95",
                "skewness",
                "excess_kurtosis",
                "ulcer_index",
                "longest_recovery_days",
            )
        ),
        "crypto_metrics_complete": all(
            key in analyzer_report["crypto_specific"]
            for key in (
                "weekend_volatility",
                "max_decline_1h",
                "max_decline_4h",
                "max_decline_24h",
                "crash_frequency_24h",
            )
        ),
        "self_test_orders_zero": analyzer_report["side_effects"]["orders_submitted"] == 0,
    }
    ui_source = (workspace / "ui" / "server.py").read_text(encoding="utf-8")
    dashboard_checks = {
        "snapshot_endpoint": '"/api/snapshot"' in ui_source,
        "candles_endpoint": '"/api/candles"' in ui_source,
        "health_endpoint": '"/health"' in ui_source,
        "roadmap_panel": 'id="maturity"' in ui_source,
        "direct_order_endpoint_absent": "/api/order" not in ui_source,
    }
    gates = [
        (1, "Multi-source crypto data platform", platform_checks),
        (2, "Data quality and causal candle controls", quality_checks),
        (3, "Crypto performance and risk analyzer", analyzer_checks),
        (4, "Read-only operations dashboard", dashboard_checks),
    ]
    project_gates = [
        {
            "project_id": project_id,
            "name": name,
            "passed": all(checks.values()),
            "checks": checks,
        }
        for project_id, name, checks in gates
    ]
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    source_paths = [
        status_path,
        dependency_path,
        workspace / "data" / "market_data.py",
        workspace / "research" / "crypto_performance.py",
        workspace / "ui" / "server.py",
    ]
    artifact: dict[str, Any] = {
        "schema_version": "crypto_beginner_foundation_v1",
        "generated_at": generated_at,
        "status": (
            "BEGINNER_CERTIFIED"
            if all(item["passed"] for item in project_gates)
            else "BEGINNER_BLOCKED"
        ),
        "project_gates": project_gates,
        "analyzer_self_test": analyzer_report,
        "source_hashes": {
            str(path.relative_to(workspace)): _hash_file(path) for path in source_paths
        },
        "claims": {
            "live_profitability_proven": False,
            "telegram_signal_accuracy_validated": False,
            "higher_levels_certified": False,
        },
        "side_effects": {
            "orders_submitted": 0,
            "exchange_mutations": 0,
            "trading_authority_changed": False,
        },
    }
    digest = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact["artifact_id"] = f"BEGINNER_{digest[:16]}"
    artifact["content_hash"] = digest

    ladder = build_maturity_ladder(workspace, beginner_artifact=artifact)
    run_root = output_root / "roadmap" / "runs" / artifact["artifact_id"]
    _write_json(run_root / "beginner_foundation.json", artifact)
    _write_json(run_root / "crypto_maturity_ladder.json", ladder)
    _write_json(output_root / "roadmap" / "beginner_foundation.json", artifact)
    _write_json(output_root / "roadmap" / "crypto_maturity_ladder.json", ladder)
    return artifact, ladder
