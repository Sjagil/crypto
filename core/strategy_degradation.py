"""Sample-aware live strategy degradation without blocking protective exits."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from utils.common import utc_iso

ZERO = Decimal("0")


def _decimal(value: Any, *, default: Decimal = ZERO) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default
    return result if result.is_finite() else default


def _window_metrics(values: Sequence[Any], window: int) -> dict[str, Any]:
    pnls = [_decimal(value) for value in values][-window:]
    gains = sum((value for value in pnls if value > ZERO), ZERO)
    losses = abs(sum((value for value in pnls if value < ZERO), ZERO))
    return {
        "requested_window": window,
        "observations": len(pnls),
        "expectancy_eur": (
            str(sum(pnls, ZERO) / Decimal(len(pnls))) if pnls else None
        ),
        "profit_factor": (
            float(gains / losses)
            if losses > ZERO
            else None
        ),
        "net_pnl_eur": str(sum(pnls, ZERO)),
    }


def evaluate_strategy_degradation(
    accounts: Mapping[str, Mapping[str, Any]],
    *,
    integrity_failures: Sequence[str] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return conservative entry authority based on independent live round trips.

    Ten trades only permit a reduced state.  A paper demotion needs at least
    twenty trades and a materially negative 20-trade window.  Shadow demotion
    needs at least thirty trades and stronger negative evidence.  Protective
    exits remain permitted in every state.
    """

    failures = sorted({str(value) for value in integrity_failures if value})
    rows: list[dict[str, Any]] = []
    for strategy_id, account in sorted(accounts.items()):
        values = list(account.get("realized_round_trips") or [])
        windows = {
            str(window): _window_metrics(values, window)
            for window in (10, 20, 30)
        }
        observations = len(values)
        state = "VALIDATING"
        reasons: list[str] = []
        risk_multiplier = Decimal("1")
        entry_allowed = True

        w10_expectancy = _decimal(
            windows["10"].get("expectancy_eur"),
            default=Decimal("NaN"),
        )
        w10_pf = windows["10"].get("profit_factor")
        w20_expectancy = _decimal(
            windows["20"].get("expectancy_eur"),
            default=Decimal("NaN"),
        )
        w20_pf = windows["20"].get("profit_factor")
        w30_expectancy = _decimal(
            windows["30"].get("expectancy_eur"),
            default=Decimal("NaN"),
        )
        w30_pf = windows["30"].get("profit_factor")

        if failures:
            state = "DISABLED"
            reasons.extend(failures)
            risk_multiplier = ZERO
            entry_allowed = False
        elif (
            observations >= 30
            and w30_expectancy.is_finite()
            and w30_expectancy < ZERO
            and w30_pf is not None
            and float(w30_pf) < 0.80
        ):
            state = "SHADOW_ACTIVE"
            reasons.append("ROLLING_30_MATERIALLY_NEGATIVE")
            risk_multiplier = ZERO
            entry_allowed = False
        elif (
            observations >= 20
            and w20_expectancy.is_finite()
            and w20_expectancy < ZERO
            and w20_pf is not None
            and float(w20_pf) < 0.90
        ):
            state = "PAPER_ACTIVE"
            reasons.append("ROLLING_20_NEGATIVE")
            risk_multiplier = ZERO
            entry_allowed = False
        elif (
            observations >= 10
            and w10_expectancy.is_finite()
            and w10_expectancy < ZERO
            and w10_pf is not None
            and float(w10_pf) < 0.75
        ):
            state = "LIVE_REDUCED"
            reasons.append("ROLLING_10_EARLY_WARNING")
            risk_multiplier = Decimal("0.5")
        elif observations >= 10:
            state = "LIVE_ACTIVE"

        rows.append(
            {
                "strategy_id": str(strategy_id),
                "strategy_dna_hash": str(
                    account.get("strategy_dna") or ""
                ),
                "closed_round_trips": observations,
                "degradation_state": state,
                "reason_codes": reasons,
                "entry_allowed": entry_allowed,
                "protective_exits_allowed": True,
                "risk_multiplier": str(risk_multiplier),
                "rolling_windows": windows,
                "automatic_cap_increase_permitted": False,
            }
        )
    return {
        "schema_version": "strategy_degradation_v1",
        "generated_at": generated_at or utc_iso(),
        "sample_policy": {
            "early_warning_minimum_round_trips": 10,
            "paper_demotion_minimum_round_trips": 20,
            "shadow_demotion_minimum_round_trips": 30,
            "single_loss_streak_never_disables_strategy": True,
        },
        "strategies": rows,
        "degraded_strategy_count": sum(
            row["degradation_state"]
            in {"LIVE_REDUCED", "PAPER_ACTIVE", "SHADOW_ACTIVE", "DISABLED"}
            for row in rows
        ),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def degradation_by_dna(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("strategy_dna_hash") or ""): dict(row)
        for row in payload.get("strategies") or []
        if row.get("strategy_dna_hash")
    }


__all__ = ["degradation_by_dna", "evaluate_strategy_degradation"]
