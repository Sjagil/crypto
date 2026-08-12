"""Strict, shared loader for explicitly approved execution markets.

Market exceptions broaden technical market eligibility only.  They never
grant strategy-DNA authority, create an order intent, or waive a natural
signal, liquidity, reconciliation, Shariah, or risk check.
"""

from __future__ import annotations

from typing import Any

import yaml

from config.settings import Settings


def load_execution_market_exceptions(
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Load the fail-closed operator exception registry.

    Invalid or widened limits are rejected at load time so all consumers use
    exactly the same safety contract.
    """

    path = (
        settings.paths.project_root
        / "config"
        / "execution_market_exceptions.yaml"
    )
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("default_policy") != "FAIL_CLOSED":
        raise ValueError("execution market exception registry is not fail-closed")
    markets = raw.get("markets")
    if not isinstance(markets, dict):
        raise ValueError("execution market exception registry lacks markets")
    result: dict[str, dict[str, Any]] = {}
    for market, values in markets.items():
        normalized = str(market).strip().upper().replace("/", "-")
        if not isinstance(values, dict) or not values.get("approved"):
            continue
        if not values.get("spot_only"):
            raise ValueError("execution market exceptions must remain spot-only")
        if float(values.get("maximum_order_eur") or 0.0) > 10.0:
            raise ValueError("execution market exception exceeds EUR 10 order cap")
        if float(values.get("maximum_total_exposure_eur") or 0.0) > 10.0:
            raise ValueError("execution market exception exceeds EUR 10 exposure cap")
        if not values.get("requires_approved_strategy_dna"):
            raise ValueError("execution market exception must require approved DNA")
        if not values.get("requires_natural_signal"):
            raise ValueError("execution market exception must require a natural signal")
        result[normalized] = dict(values)
    return result


__all__ = ["load_execution_market_exceptions"]
