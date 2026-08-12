from __future__ import annotations

from core.strategy_degradation import evaluate_strategy_degradation


def _account(values: list[float], *, dna: str = "a" * 64) -> dict:
    return {
        "strategy_dna": dna,
        "realized_round_trips": values,
    }


def test_degradation_is_sample_aware_and_never_blocks_exits() -> None:
    payload = evaluate_strategy_degradation(
        {
            "VALIDATING": _account([-1.0] * 9),
            "REDUCED": _account([-1.0] * 10, dna="b" * 64),
            "PAPER": _account([-1.0] * 20, dna="c" * 64),
            "SHADOW": _account([-1.0] * 30, dna="d" * 64),
        },
        generated_at="2026-07-30T00:00:00Z",
    )
    rows = {row["strategy_id"]: row for row in payload["strategies"]}

    assert rows["VALIDATING"]["degradation_state"] == "VALIDATING"
    assert rows["VALIDATING"]["entry_allowed"] is True
    assert rows["REDUCED"]["degradation_state"] == "LIVE_REDUCED"
    assert rows["REDUCED"]["risk_multiplier"] == "0.5"
    assert rows["PAPER"]["degradation_state"] == "PAPER_ACTIVE"
    assert rows["PAPER"]["entry_allowed"] is False
    assert rows["SHADOW"]["degradation_state"] == "SHADOW_ACTIVE"
    assert all(row["protective_exits_allowed"] for row in rows.values())


def test_integrity_failure_disables_entries_not_protective_exits() -> None:
    payload = evaluate_strategy_degradation(
        {"TEST": _account([1.0] * 40)},
        integrity_failures=["STRATEGY_DNA_MISMATCH"],
    )
    row = payload["strategies"][0]

    assert row["degradation_state"] == "DISABLED"
    assert row["entry_allowed"] is False
    assert row["protective_exits_allowed"] is True
    assert row["reason_codes"] == ["STRATEGY_DNA_MISMATCH"]
