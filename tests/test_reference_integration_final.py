from __future__ import annotations

import json
from pathlib import Path

from reporting.reference_integration_final import build_reference_integration_final


def test_final_report_contains_exact_a_through_ad_sections(tmp_path: Path) -> None:
    phase_a = tmp_path / "output" / "reference_integration" / "phase_a"
    run = phase_a / "runs" / "baseline"
    run.mkdir(parents=True)
    baseline = {
        "generated_at": "2026-08-01T00:00:00Z",
        "production_baseline": {"portfolio_target_owner": None},
        "repositories": [],
    }
    (run / "reference_integration_phase_a.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    (phase_a / "latest.json").write_text("{}", encoding="utf-8")
    verification = {
        "targeted_passed": True,
        "integration_passed": True,
        "full_suite_passed": True,
        "lint_passed": True,
        "compile_passed": True,
        "diff_check_passed": True,
    }

    result = build_reference_integration_final(tmp_path, verification=verification)
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    expected = [chr(value) for value in range(ord("A"), ord("Z") + 1)] + [
        "AA",
        "AB",
        "AC",
        "AD",
    ]
    assert set(payload["sections"]) == set(expected)
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert markdown.index("## A. ") < markdown.index("## B. ")
    assert markdown.index("## Z. ") < markdown.index("## AA. ")
    assert markdown.index("## AC. ") < markdown.index("## AD. ")
    assert Path(result["json_path"]).parent.name == result["artifact_hash"]
    assert payload["side_effects"]["orders_submitted"] == 0
    assert build_reference_integration_final(
        tmp_path, verification=verification
    )["artifact_hash"] == result["artifact_hash"]
