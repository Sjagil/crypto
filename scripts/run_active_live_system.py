#!/usr/bin/env python3
"""Round-35 sidecar for the canonical autonomous live supervisor.

The sidecar refreshes a private-account-derived portfolio target plan.  It does
not submit broker orders.  Actual BUY/SELL/stop-loss/take-profit execution stays
inside `main.py autonomous-live`, which owns reconciliation, strategy authority
and the Bitvavo execution client.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from utils.common import atomic_write_json, read_json, utc_iso


def _run(
    args: Sequence[str],
    *,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )


def _marker_path(settings: Settings) -> Path:
    return settings.paths.output_dir / "portfolio" / "active_live_orchestrator.json"


def _write_marker(settings: Settings, *, enabled: bool, mode: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "active_live_orchestrator_v1",
        "enabled": enabled,
        "mode": mode,
        "updated_at": utc_iso(),
        "exchange_authority": "CANONICAL_AUTONOMOUS_LIVE_ONLY",
        "direct_order_submission": False,
    }
    path = _marker_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return payload


def _existing_marker(settings: Settings) -> dict[str, object]:
    path = _marker_path(settings)
    if not path.is_file():
        return {"enabled": False, "status": "NOT_CONFIGURED"}
    try:
        return dict(read_json(path))
    except (OSError, TypeError, ValueError):
        return {"enabled": False, "status": "UNREADABLE"}


def refresh_live_plan() -> int:
    process = _run(
        [
            str(ROOT / "scripts" / "run_active_portfolio_cycle.py"),
            "--mode",
            "live-plan",
            "--private-read",
            "--sync-data",
            "--refresh-whole-stack",
        ]
    )
    return int(process.returncode)


def supervisor_status(*, quiet: bool = False) -> int:
    return int(_run(["main.py", "autonomous-live", "status"], quiet=quiet).returncode)


def ensure_supervisor_started() -> int:
    if supervisor_status(quiet=True) == 0:
        return 0
    return int(_run(["main.py", "autonomous-live", "start"]).returncode)


def status_payload(settings: Settings) -> dict[str, object]:
    plan = settings.paths.output_dir / "portfolio" / "active_live_plan.json"
    return {
        "schema_version": "round35_active_live_status_v1",
        "orchestrator": _existing_marker(settings),
        "active_live_plan_present": plan.is_file(),
        "active_live_plan": str(plan),
        "autonomous_live_status_rc": supervisor_status(quiet=True),
        "actual_execution_owner": "main.py autonomous-live",
        "sidecar_submits_orders": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "start", "run", "shutdown"))
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--cycles", type=int, default=0)
    args = parser.parse_args()

    settings = Settings.load(create_directories=True)

    if args.command == "status":
        print(json.dumps(status_payload(settings), indent=2, sort_keys=True))
        return 0

    if args.command == "shutdown":
        _write_marker(settings, enabled=False, mode="shutdown")
        return int(_run(["main.py", "autonomous-live", "shutdown"]).returncode)

    if args.interval_seconds < 60:
        raise SystemExit("minimum interval is 60 seconds")

    _write_marker(settings, enabled=True, mode=args.command)
    plan_rc = refresh_live_plan()
    if plan_rc != 0:
        print(
            "[round35] private live plan is not ready; new portfolio-driven "
            "entries remain fail-closed while reconciliation/protective exits "
            "can continue",
            flush=True,
        )

    start_rc = ensure_supervisor_started()
    if start_rc != 0:
        print("[round35] autonomous-live could not be started", flush=True)
        return start_rc

    if args.command == "start":
        print(json.dumps(status_payload(settings), indent=2, sort_keys=True))
        return 0

    cycle = 0
    while True:
        cycle += 1
        print(f"[round35] plan cycle={cycle} START", flush=True)
        plan_rc = refresh_live_plan()
        status_rc = supervisor_status(quiet=False)
        print(
            f"[round35] plan cycle={cycle} END "
            f"plan_rc={plan_rc} autonomous_live_status_rc={status_rc}",
            flush=True,
        )
        if args.cycles and cycle >= args.cycles:
            return 0 if status_rc == 0 else status_rc
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
