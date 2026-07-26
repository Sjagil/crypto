from __future__ import annotations

import pytest

from research.microstructure_preregistration import (
    crowding_avoidance_plan,
    write_crowding_avoidance_plan,
)


def test_crowding_family_is_small_fixed_and_economic() -> None:
    plan = crowding_avoidance_plan()
    assert plan["family_id"] == "CROWDING_AVOIDANCE_V1"
    assert len(plan["primary_dna"]) == 4
    assert len(
        {row["id"] for row in plan["primary_dna"]}
    ) == 4
    assert not plan["adaptive_expansion_permitted"]
    assert not plan[
        "threshold_changes_after_results_permitted"
    ]
    assert plan["action"] == "BLOCK_NEW_LONG_ONLY"
    assert plan["data_readiness"] == (
        "INSUFFICIENT_NOT_BACKTESTED"
    )
    assert plan["orders_generated"] == 0
    assert not plan["live_ready"]


def test_microstructure_plan_is_immutable(tmp_path) -> None:
    path = tmp_path / "plan.json"
    first = write_crowding_avoidance_plan(path)
    second = write_crowding_avoidance_plan(path)
    assert first["plan_hash"] == second["plan_hash"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_REVISION"):
        write_crowding_avoidance_plan(path)
