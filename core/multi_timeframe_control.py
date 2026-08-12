"""Operational orchestration for multi-timeframe authority and local UI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

from config.settings import Settings
from core.generated_strategy_live import (
    synchronize_positive_strategy_live_authority,
)
from core.live_universe import (
    candle_health,
    live_universe_status,
    refresh_live_universe,
)
from research.multi_timeframe_authority import (
    load_validated_multi_timeframe_candidates,
    validate_multi_timeframe_authority,
    write_multi_timeframe_authority_registry,
)
from ui.server import start_ui
from utils.common import read_json, stable_hash, utc_iso


def _safe_read(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def authority_status(settings: Settings) -> dict[str, Any]:
    path = (
        settings.paths.output_dir
        / "governance"
        / "multi_timeframe_authority.json"
    )
    if not path.is_file():
        return {
            "status": "MISSING",
            "timeframe_coverage": {},
            "strategies": [],
            "orders_generated": 0,
            "orders_submitted": 0,
        }
    return dict(read_json(path))


def audit_authority(settings: Settings) -> dict[str, Any]:
    registry = authority_status(settings)
    failures: list[str] = []
    expected_hash = stable_hash(
        {
            key: value
            for key, value in registry.items()
            if key not in {"registry_hash", "artifact"}
        },
        length=64,
    )
    if registry.get("registry_hash") != expected_hash:
        failures.append("REGISTRY_HASH_MISMATCH")
    required = ("1h", "2h", "4h", "1d", "1W")
    coverage = dict(registry.get("timeframe_coverage") or {})
    for timeframe in required:
        if int(coverage.get(timeframe) or 0) < 1:
            failures.append(f"TIMEFRAME_AUTHORITY_MISSING:{timeframe}")
    universe = live_universe_status(settings)
    if universe.get("status") != "READY":
        failures.append("LIVE_UNIVERSE_NOT_READY")
    if len(universe.get("selected_markets") or []) < 5:
        failures.append("FEWER_THAN_FIVE_LIVE_MARKETS")
    validation = _safe_read(
        settings.paths.lab_dir
        / "reports"
        / "multi_timeframe_authority_validation_v1.json"
    )
    validated_by_dna = {
        str(row.get("strategy_dna_hash")): row
        for row in validation.get("selected_candidates") or []
    }
    for row in registry.get("strategies") or []:
        timeframe = str(row.get("timeframe") or "")
        if timeframe not in {"1h", "2h"}:
            continue
        dna = str(row.get("strategy_dna_hash") or "")
        selected = validated_by_dna.get(dna)
        if selected is None:
            failures.append(f"VALIDATION_ARTIFACT_MISSING:{dna}")
        elif selected.get("validation_pass") is not True:
            failures.append(f"VALIDATION_NOT_PASSED:{dna}")
        elif row.get("frozen_candidate_hash") != selected.get(
            "frozen_candidate_hash"
        ):
            failures.append(f"FROZEN_IDENTITY_MISMATCH:{dna}")
    return {
        "schema_version": "multi_timeframe_authority_audit_v1",
        "audited_at": utc_iso(),
        "status": "PASSED" if not failures else "FAILED",
        "failures": failures,
        "timeframe_coverage": coverage,
        "live_markets": universe.get("selected_markets") or [],
        "registered_multi_timeframe_strategy_count": len(
            registry.get("strategies") or []
        ),
        "unknown_dna_fail_closed": registry.get("unknown_dna_fail_closed"),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def _process_state(settings: Settings) -> dict[str, Any]:
    locks = {
        "live_supervisor": settings.paths.output_dir
        / "live"
        / "autonomous_live.lock",
        "data_sync": settings.paths.checkpoints_dir / "data_service.lock",
        "research_lab": settings.paths.output_dir
        / "research"
        / "simple_strategy_lab"
        / "service.lock",
    }
    result: dict[str, Any] = {}
    for name, path in locks.items():
        raw = _safe_read(path)
        result[name] = {
            "status": "RUNNING" if raw else "NOT_RUNNING",
            "pid": raw.get("pid"),
            "mode": raw.get("mode"),
            "started_at": raw.get("started_at") or raw.get("acquired_at"),
        }
    return result


def write_expansion_report(
    settings: Settings,
    *,
    candles: Mapping[str, Any],
    universe: Mapping[str, Any],
    validation: Mapping[str, Any],
    registry: Mapping[str, Any],
    authority_audit: Mapping[str, Any],
    ui: Mapping[str, Any],
) -> Path:
    path = (
        settings.paths.output_dir
        / "reports"
        / "multi_timeframe_live_expansion_report.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multi-timeframe live expansion",
        "",
        f"Generated: {utc_iso()}",
        "",
        "## Outcome",
        "",
        f"- Authority audit: **{authority_audit.get('status')}**",
        f"- Live universe: **{universe.get('status')}**",
        f"- Selected markets: {', '.join(universe.get('selected_markets') or [])}",
        f"- Candle series healthy: {candles.get('healthy_series')}/{candles.get('total_series')}",
        f"- UI: {ui.get('status')} at {ui.get('url')}",
        f"- Maximum concurrent positions: {registry.get('maximum_concurrent_positions')}",
        "- Natural signals only; this expansion generated and submitted zero orders.",
        "",
        "## Authority coverage",
        "",
        "| Timeframe | Strategies |",
        "|---|---:|",
    ]
    for timeframe in ("1h", "2h", "4h", "1d", "1W"):
        lines.append(
            f"| {timeframe} | "
            f"{int((registry.get('timeframe_coverage') or {}).get(timeframe) or 0)} |"
        )
    lines.extend(
        [
            "",
            "## New 1h/2h validation",
            "",
            "| Strategy | TF | Trades | PF | Net | OOS | Stress PF | Folds | MC P(+)|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation.get("selected_candidates") or []:
        lines.append(
            "| {id} | {tf} | {trades} | {pf:.3f} | {net:.1%} | "
            "{oos:.1%} | {spf:.3f} | {folds}/5 | {mc:.1%} |".format(
                id=row["strategy_id"],
                tf=row["timeframe"],
                trades=row["normal"]["trade_count"],
                pf=row["normal"]["profit_factor"],
                net=row["normal"]["panel_net_return"],
                oos=row["out_of_sample"]["panel_net_return"],
                spf=row["stressed"]["profit_factor"],
                folds=row["walk_forward"]["positive_folds"],
                mc=row["monte_carlo"]["probability_positive"],
            )
        )
    lines.extend(
        [
            "",
            "## Capital warnings",
            "",
            "- Monte Carlo is confidence evidence, not a promotion guarantee.",
            "- The selected 1h/2h paths still show severe compounded drawdown "
            "tails; live use therefore remains micro-canary only.",
            "- Maximum order is EUR 10, total generated-strategy exposure is "
            "EUR 15, autoscaling is disabled, and new DNA fails closed.",
            "",
            "## Runtime processes",
            "",
            "```json",
            str(_process_state(settings)),
            "```",
            "",
            "## Daily commands",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe .\main.py authority status",
            r".\.venv\Scripts\python.exe .\main.py universe status",
            r".\.venv\Scripts\python.exe .\main.py candles status",
            r".\.venv\Scripts\python.exe .\main.py ui status",
            r".\.venv\Scripts\python.exe .\main.py autonomous-live status",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def expand_live(settings: Settings) -> dict[str, Any]:
    """Run the controlled expansion without starting a second live process."""

    processes_before = _process_state(settings)
    candles = candle_health(settings)
    universe = await refresh_live_universe(settings)
    validation = await asyncio.to_thread(
        validate_multi_timeframe_authority,
        settings,
    )
    registry = write_multi_timeframe_authority_registry(settings)
    authority_active, _, authority_failures = (
        synchronize_positive_strategy_live_authority(settings)
    )
    audit = audit_authority(settings)
    ui = await asyncio.to_thread(start_ui, settings)
    report = write_expansion_report(
        settings,
        candles=candles,
        universe=universe,
        validation=validation,
        registry=registry,
        authority_audit=audit,
        ui=ui,
    )
    service_authority = _safe_read(
        settings.paths.output_dir
        / "live"
        / "autonomous_live_authority.json"
    )
    service_markets = list(service_authority.get("markets") or [])
    requires_restart = set(service_markets) != set(
        universe.get("selected_markets") or []
    )
    return {
        "schema_version": "multi_timeframe_expand_live_v1",
        "status": (
            "READY_RESTART_REQUIRED"
            if requires_restart
            else "READY"
            if audit.get("status") == "PASSED"
            and authority_active
            and ui.get("status") == "RUNNING"
            else "BLOCKED"
        ),
        "generated_at": utc_iso(),
        "processes_before": processes_before,
        "processes_after": _process_state(settings),
        "candle_health": {
            "healthy": candles.get("healthy_series"),
            "total": candles.get("total_series"),
        },
        "live_markets": universe.get("selected_markets") or [],
        "timeframe_coverage": registry.get("timeframe_coverage"),
        "validated_candidates": [
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "timeframe": row["timeframe"],
            }
            for row in validation.get("selected_candidates") or []
        ],
        "positive_strategy_authority_active": authority_active,
        "positive_strategy_authority_failures": authority_failures,
        "service_markets": service_markets,
        "live_supervisor_restart_required": requires_restart,
        "ui": ui,
        "report": str(report),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def research_candidates_status(settings: Settings) -> dict[str, Any]:
    path = (
        settings.paths.lab_dir
        / "reports"
        / "multi_timeframe_authority_validation_v1.json"
    )
    report = _safe_read(path)
    return {
        "status": "READY" if report else "NOT_RUN",
        "candidate_count": report.get("candidate_count", 0),
        "passing_count": report.get("passing_count", 0),
        "selected_candidates": [
            {
                "strategy_id": row.get("strategy_id"),
                "strategy_dna_hash": row.get("strategy_dna_hash"),
                "timeframe": row.get("timeframe"),
                "metrics": {
                    "trades": (row.get("normal") or {}).get("trade_count"),
                    "profit_factor": (row.get("normal") or {}).get(
                        "profit_factor"
                    ),
                    "net_return": (row.get("normal") or {}).get(
                        "panel_net_return"
                    ),
                    "out_of_sample_net_return": (
                        row.get("out_of_sample") or {}
                    ).get("panel_net_return"),
                },
            }
            for row in report.get("selected_candidates") or []
        ],
        "loaded_executable_candidates": len(
            load_validated_multi_timeframe_candidates(settings)
        ),
        "artifact": str(path),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


__all__ = [
    "audit_authority",
    "authority_status",
    "expand_live",
    "research_candidates_status",
    "write_expansion_report",
]
