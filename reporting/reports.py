"""Auditable console, JSON, CSV and HTML research reporting."""

from __future__ import annotations

import html
import math
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from config.settings import Settings
from research.backtest import BacktestResult
from research.optimization import ResearchOutcome
from utils.common import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)


def _safe(value: Any) -> Any:
    """Convert values to strict JSON-safe structures without retaining frames."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.DataFrame):
        return {"rows": len(value), "columns": list(value.columns)}
    if isinstance(value, pd.Series):
        return {"rows": len(value), "name": value.name}
    if isinstance(value, BaseModel):
        return _safe(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_write_text(path, frame.to_csv(index=True, lineterminator="\n"))


def _result_summary(result: BacktestResult) -> dict[str, Any]:
    return {
        "strategy_id": result.strategy_id,
        "initial_cash_eur": result.initial_cash_eur,
        "ending_equity_eur": result.ending_equity_eur,
        "metrics": _safe(result.metrics),
        "integrity": _safe(result.integrity),
        "trade_count": len(result.trades),
        "order_count": len(result.orders),
    }


def console_backtest_summary(result: BacktestResult) -> str:
    metrics = result.metrics
    return (
        f"{result.strategy_id}: trades={int(metrics.get('trade_count') or 0)}, "
        f"net_pnl_eur={float(metrics.get('net_pnl_eur') or 0):.2f}, "
        f"expectancy_r={float(metrics.get('net_expectancy_r') or 0):.4f}, "
        f"max_drawdown={float(metrics.get('maximum_drawdown') or 0):.2%}"
    )


def write_backtest_report(
    result: BacktestResult,
    output_directory: Path | str,
    *,
    label: str = "backtest",
) -> dict[str, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = _result_summary(result)
    trades = pd.DataFrame([trade.model_dump(mode="json") for trade in result.trades])
    orders = pd.DataFrame([order.to_dict() for order in result.orders])
    paths = {
        "summary": atomic_write_json(directory / f"{label}_summary.json", summary),
        "equity": _write_frame(directory / f"{label}_equity.csv", result.equity_curve),
        "trades": _write_frame(directory / f"{label}_trades.csv", trades),
        "orders": _write_frame(directory / f"{label}_orders.csv", orders),
    }
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["metrics"].items()
    )
    document = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        f"<title>{html.escape(label)} report</title>"
        "<style>body{font:15px system-ui;max-width:900px;margin:40px auto;"
        "color:#17202a}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd1d1;padding:7px;text-align:left}"
        "th{background:#f4f6f7}</style>"
        f"<h1>{html.escape(result.strategy_id)} backtest</h1>"
        "<p>Research output only. It is not a profitability claim or live approval.</p>"
        f"<table>{rows}</table></html>"
    )
    paths["html"] = atomic_write_text(directory / f"{label}.html", document)
    return paths


def write_research_report(
    outcome: ResearchOutcome,
    settings: Settings,
    output_directory: Path | str,
    *,
    label: str = "research",
) -> dict[str, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy_id": outcome.strategy_id,
        "parameters": _safe(outcome.parameters),
        "status": outcome.gate.status.value,
        "passed": outcome.gate.passed,
        "gate_reasons": list(outcome.gate.reasons),
        "gate_metrics": _safe(outcome.gate.metrics),
        "lookahead_safe": outcome.lookahead_safe,
        "repainting_safe": outcome.repainting_safe,
        "normal": _result_summary(outcome.normal_result),
        "stressed": _result_summary(outcome.stressed_result),
        "holdout": _result_summary(outcome.holdout_result),
        "walk_forward": _safe(outcome.walk_forward),
        "stability": _safe(outcome.stability),
        "optimization": {
            "method": outcome.optimization.method,
            "best_parameters": _safe(outcome.optimization.best_parameters),
            "best_score": _safe(outcome.optimization.best_score),
            "trial_count": len(outcome.optimization.trials),
            "resumed_trials": outcome.optimization.resumed_trials,
        },
    }
    paths: dict[str, Path] = {
        "summary": atomic_write_json(directory / f"{label}_summary.json", summary),
        "config": atomic_write_json(
            directory / f"{label}_config_redacted.json",
            settings.redacted_dict(),
        ),
    }
    trials = pd.DataFrame([_safe(asdict(trial)) for trial in outcome.optimization.trials])
    folds = pd.DataFrame([_safe(asdict(fold)) for fold in outcome.walk_forward.folds])
    paths["trials"] = _write_frame(directory / f"{label}_trials.csv", trials)
    paths["folds"] = _write_frame(directory / f"{label}_walk_forward.csv", folds)
    paths.update(
        {
            f"normal_{key}": value
            for key, value in write_backtest_report(
                outcome.normal_result,
                directory,
                label=f"{label}_normal",
            ).items()
        }
    )
    reason_items = "".join(
        f"<li>{html.escape(reason)}</li>" for reason in outcome.gate.reasons
    ) or "<li>None</li>"
    document = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        f"<title>{html.escape(label)} research report</title>"
        "<style>body{font:15px system-ui;max-width:960px;margin:40px auto;"
        "color:#17202a}.pass{color:#18794e}.fail{color:#b42318}"
        "code{background:#f2f4f7;padding:2px 4px}</style>"
        f"<h1>{html.escape(outcome.strategy_id)}</h1>"
        f"<h2 class=\"{'pass' if outcome.gate.passed else 'fail'}\">"
        f"{html.escape(outcome.gate.status.value)}</h2>"
        "<p>Research output only. Passing research does not authorize live trading.</p>"
        f"<h3>Gate reasons</h3><ul>{reason_items}</ul>"
        f"<p>Lookahead safe: <code>{outcome.lookahead_safe}</code>; "
        f"repainting safe: <code>{outcome.repainting_safe}</code></p></html>"
    )
    paths["html"] = atomic_write_text(directory / f"{label}.html", document)
    paths["manifest"] = write_run_manifest(
        directory / f"{label}_manifest.json",
        artifacts=tuple(paths.values()),
        settings=settings,
        run_kind="research",
        run_id=stable_hash(summary, length=20),
    )
    return paths


def write_run_manifest(
    path: Path | str,
    *,
    artifacts: tuple[Path, ...],
    settings: Settings,
    run_kind: str,
    run_id: str,
) -> Path:
    target = Path(path)
    files = [
        {
            "path": str(artifact.resolve()),
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }
        for artifact in artifacts
        if artifact.is_file() and artifact.resolve() != target.resolve()
    ]
    return atomic_write_json(
        target,
        {
            "run_id": run_id,
            "run_kind": run_kind,
            "created_at": datetime.now(UTC),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "app_version": settings.app.version,
            "artifacts": files,
        },
    )


def write_operational_reports(
    output_directory: Path | str,
    *,
    status: dict[str, Any],
    daily_summary: dict[str, Any] | None = None,
    candidate_health: dict[str, Any] | None = None,
    provider_health: list[dict[str, Any]] | dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the practical JSON/HTML status surface without running a server."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    current = {**_safe(status), "generated_at": now}
    daily = {
        "generated_at": now,
        "mode": current.get("mode"),
        "service_state": current.get("service_state"),
        "realized_pnl": current.get("realized_pnl", 0.0),
        "unrealized_pnl": current.get("unrealized_pnl", 0.0),
        "daily_pnl": current.get("daily_pnl", 0.0),
        "drawdown": current.get("drawdown", 0.0),
        "risk_state": current.get("risk_state", "NORMAL"),
        "signals": current.get("latest_signals", []),
        **_safe(daily_summary or {}),
    }
    candidate = {
        "generated_at": now,
        "active_candidate": current.get("active_candidate"),
        "shadow_challengers": current.get("shadow_challengers", []),
        "status": (
            "IDLE_NO_APPROVED_CANDIDATE"
            if not current.get("active_candidate")
            else "ACTIVE"
        ),
        **_safe(candidate_health or {}),
    }
    providers = {
        "generated_at": now,
        "providers": _safe(provider_health or []),
    }
    paths = {
        "current_status_json": atomic_write_json(
            directory / "current_status.json", current
        ),
        "daily_summary_json": atomic_write_json(
            directory / "daily_summary.json", daily
        ),
        "candidate_health_json": atomic_write_json(
            directory / "candidate_health.json", candidate
        ),
        "provider_health_json": atomic_write_json(
            directory / "provider_health.json", providers
        ),
    }

    def document(title: str, payload: dict[str, Any]) -> str:
        rows = "".join(
            "<tr><th>"
            + html.escape(str(key))
            + "</th><td><pre>"
            + html.escape(str(_safe(value)))
            + "</pre></td></tr>"
            for key, value in payload.items()
        )
        return (
            "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font:14px system-ui;max-width:1100px;margin:32px auto;"
            "color:#17202a}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #d0d5dd;padding:7px;text-align:left;"
            "vertical-align:top}th{width:240px;background:#f2f4f7}"
            "pre{white-space:pre-wrap;margin:0}</style>"
            f"<h1>{html.escape(title)}</h1>"
            "<p>Shadow/paper operational telemetry; not a live-readiness or "
            "profitability claim.</p>"
            f"<table>{rows}</table></html>"
        )

    paths["current_status_html"] = atomic_write_text(
        directory / "current_status.html",
        document("Operational status", current),
    )
    paths["daily_summary_html"] = atomic_write_text(
        directory / "daily_summary.html",
        document("Operational daily summary", daily),
    )
    return paths


__all__ = [
    "console_backtest_summary",
    "write_backtest_report",
    "write_research_report",
    "write_run_manifest",
    "write_operational_reports",
]
