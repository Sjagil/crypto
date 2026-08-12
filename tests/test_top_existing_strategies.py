from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from reporting.top_existing_strategies import (
    SCORE_WEIGHTS,
    build_report,
    normalize_fraction,
    select_top_strategies,
    select_top_ten,
    verify_reports,
    write_reports,
)


def test_metric_normalization_handles_percent_and_drawdown_signs() -> None:
    assert normalize_fraction(-0.15, kind="drawdown") == pytest.approx(0.15)
    assert normalize_fraction(0.15, kind="drawdown") == pytest.approx(0.15)
    assert normalize_fraction(15, kind="drawdown") == pytest.approx(0.15)
    assert normalize_fraction(14.5, kind="cagr") == pytest.approx(0.145)
    assert normalize_fraction(1.5, kind="return") == pytest.approx(1.5)


def test_score_weights_are_exactly_the_requested_weights() -> None:
    assert SCORE_WEIGHTS == {
        "historical_performance": 0.30,
        "robustness": 0.30,
        "drawdown_capital_protection": 0.20,
        "sample_quality": 0.10,
        "practical_deployability": 0.10,
    }
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_family_cap_prevents_duplicate_dominance() -> None:
    candidates = [
        {
            "strategy_name": f"A{index}",
            "family_cluster": "A",
            "scores": {"composite": 100 - index},
        }
        for index in range(5)
    ] + [
        {
            "strategy_name": f"B{index}",
            "family_cluster": f"B{index}",
            "scores": {"composite": 90 - index},
        }
        for index in range(8)
    ]
    selected = select_top_ten(candidates)
    assert len(selected) == 10
    assert max(Counter(row["family_cluster"] for row in selected).values()) <= 2


def test_top_twenty_uses_same_family_cap_and_real_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    report, evidence = build_report(root, limit=20)
    selected = report["top_20"]
    assert len(selected) == 20
    assert len({row["strategy_name"] for row in selected}) == 20
    assert max(Counter(row["family_cluster"] for row in selected).values()) <= 2
    assert report["new_backtests_run"] == 0
    assert report["orders_generated"] == 0
    assert evidence["report_invariants"]["ranked_strategy_count"] == 20
    assert select_top_strategies(selected, limit=10)


def test_real_evidence_report_has_reconciled_top_ten() -> None:
    root = Path(__file__).resolve().parents[1]
    report, evidence = build_report(root)

    names = [row["strategy_name"] for row in report["top_10"]]
    assert len(names) == len(set(names)) == 10
    assert report["new_backtests_run"] == 0
    assert report["strategy_parameters_changed"] == 0
    assert report["orders_generated"] == 0
    assert report["executive_summary"]["proven_profitable_strategy_exists"] is False
    assert report["executive_summary"]["unique_valid_strategies_found"] >= 350
    assert evidence["database_identity"]["row_counts"]["orders"] == 0
    assert evidence["database_identity"]["row_counts"]["fills"] == 0
    assert evidence["database_identity"]["row_counts"]["positions"] == 0
    assert report["audit_inventory"]["inventory_scope"] == (
        "BOUNDED_IMMUTABLE_RESEARCH_EVIDENCE"
    )
    assert "RR_B60_H5_Z20" in names
    assert "ROTATION_FROZEN_CONTROL" in names
    assert len(report["canary_selection"]["frozen_shadow"]) == 2


def test_written_reports_reconcile(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    # Report generation is intentionally rooted in the real repository evidence.
    paths = write_reports(root)
    verification = verify_reports(root, paths)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8", newline="")))

    assert verification["status"] == "PASSED"
    assert [row["strategy_name"] for row in report["top_10"]] == [
        row["strategy_name"] for row in csv_rows
    ]
    assert all(path.is_file() for path in paths.values())
    assert tmp_path.is_dir()
