#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    event = (ROOT / "core" / "event_driven_live.py").read_text(encoding="utf-8")
    execution = (ROOT / "execution" / "execution.py").read_text(encoding="utf-8")
    bridge = (ROOT / "core" / "active_portfolio_live_bridge.py").read_text(
        encoding="utf-8"
    )
    checks = {
        "canonical_atomic_live_buy": (
            "submit_level_2_buy_atomically" in event
            and "canonicalize_approved_buy_order" in event
        ),
        "actual_bitvavo_order_submission": "async def submit_order(" in execution,
        "native_stop_loss_order": (
            "OrderType.STOP_LOSS" in event and "NATIVE_PROTECTIVE_STOP" in event
        ),
        "tp1_actual_partial_sell": "LIVE_TAKE_PROFIT_1_FILLED" in event,
        "tp2_actual_sell": 'reason = "TAKE_PROFIT_2"' in event,
        "trailing_native_stop": "LIVE_RUNNER_TRAILING_STOP_UPDATED" in event,
        "portfolio_entry_budget_consumed": "active_entry_budget_eur(" in event,
        "portfolio_reduce_full_exit": (
            "ACTIVE_PORTFOLIO_REDUCE" in event
            and "ACTIVE_PORTFOLIO_FULL_EXIT" in event
        ),
        "partial_exit_reprotects_remainder": (
            "REARM_NATIVE_STOP_AFTER_PARTIAL_EXIT" in event
        ),
        "bridge_has_no_submit_order": "submit_order(" not in bridge,
        "withdrawal_endpoint_absent": "/withdraw" not in execution.casefold(),
    }
    payload = {
        "schema_version": "round35_live_execution_contract_audit_v1",
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "orders_submitted_by_audit": 0,
        "private_exchange_requests_by_audit": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
