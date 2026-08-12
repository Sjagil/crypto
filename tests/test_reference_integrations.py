from __future__ import annotations

from pathlib import Path

from research.reference_integrations import (
    REFERENCE_SPECS,
    finrl_robust_zscores,
    freqtrade_timeframe_seconds,
    pybroker_returns,
    qlib_sum_by_index,
    rdagent_feedback_acceptability,
    tradingagents_research_plan,
    vectorbt_book_invariants,
    verify_reference_repositories,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def test_all_nine_reference_repositories_are_pinned() -> None:
    rows = verify_reference_repositories(WORKSPACE)
    assert len(rows) == len(REFERENCE_SPECS) == 9
    assert all(row["commit_verified"] for row in rows)
    assert all(row["license_file_sha256"] for row in rows)


def test_python_reference_functions_execute_from_upstream_sources() -> None:
    assert finrl_robust_zscores(WORKSPACE, [1.0, 1.0, 2.0, 3.0], window=3) == [
        None,
        None,
        10.0,
        1.0,
    ]
    assert freqtrade_timeframe_seconds(WORKSPACE, "5m") == 300
    assert pybroker_returns(WORKSPACE, [100.0, 101.0, 99.0]) == [
        None,
        0.01,
        (99.0 - 101.0) / 101.0,
    ]
    assert qlib_sum_by_index(
        WORKSPACE,
        [{1: 2.0, 3: 4.0}, {1: 1.0, 2: 5.0}],
        [1, 2, 3],
    ) == {"1": 3.0, "2": 5.0, "3": 4.0}
    assert vectorbt_book_invariants(WORKSPACE, 100.0, 101.0) == {
        "strictly_uncrossed": True,
        "locked_within_tolerance": False,
    }
    assert rdagent_feedback_acceptability(WORKSPACE) == {
        "acceptable": True,
        "finished": True,
    }
    assert tradingagents_research_plan(WORKSPACE)["recommendation"] == "Hold"
