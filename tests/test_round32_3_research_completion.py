from __future__ import annotations

from pathlib import Path

from research.combinatorial_lab import (
    CombinationGenerator,
    LabRunner,
    signal_block_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def test_dependent_ema_sensitivity_is_pruned(isolated_settings) -> None:
    registry = signal_block_registry()
    combination = CombinationGenerator(
        {"ema_trend": registry["ema_trend"]}
    ).generate(sizes=(1,), timeframes=("1h",))[0]
    runner = LabRunner(isolated_settings, registry=registry)
    rows = runner._parameter_sets(combination, None)
    assert rows
    for _, parameters in rows:
        selected = parameters["ema_trend"]
        assert selected["fast"] < selected["slow"]


def test_research_cli_supports_local_resume_without_redownload() -> None:
    source = (ROOT / "core" / "cli.py").read_text(encoding="utf-8")
    assert 'research.add_argument("--skip-download", action="store_true")' in source
    assert 'not getattr(args, "skip_download", False)' in source


def test_top_reporting_marks_missing_accounting_as_missing_evidence() -> None:
    source = (
        ROOT / "reporting" / "top_existing_strategies.py"
    ).read_text(encoding="utf-8")
    assert '"missing_evidence_artifacts"' in source
    assert '"evidence_artifacts_available"' in source
