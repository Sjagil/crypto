from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_row_budget_exists() -> None:
    source = (ROOT / "core" / "cli.py").read_text(encoding="utf-8")
    assert 'research.add_argument("--max-rows", type=int)' in source
    assert 'maximum_rows = getattr(args, "max_rows", None)' in source
    assert "frame.tail(maximum_rows)" in source


def test_random_search_has_validated_baseline_and_invalid_draw_isolation() -> None:
    source = (ROOT / "research" / "optimization.py").read_text(encoding="utf-8")
    assert "baseline = strategy.parameters()" in source
    assert "raw_trial_id = stable_hash(" in source
    assert "except ValueError as exc:" in source
    assert 'status="FAILED"' in source
