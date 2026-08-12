from __future__ import annotations

import json
from pathlib import Path

from reporting.ml_legacy_assessment import assess_legacy_shadow_ml
from reporting.reference_integration_health import build_reference_integration_health


def test_legacy_ml_assessment_never_promotes_incomplete_provenance(tmp_path: Path) -> None:
    intelligence = tmp_path / "output" / "intelligence"
    intelligence.mkdir(parents=True)
    (intelligence / "opportunity_training_rows.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "decision_timestamp": "2026-08-01T00:00:00+00:00",
                        "label_uses_future_features": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (intelligence / "model_bundle.joblib").write_bytes(b"legacy-shadow")
    (intelligence / "model_status.json").write_text(
        json.dumps(
            {
                "status": "SHADOW_ONLY",
                "live_decision_influence": False,
                "validation_folds": [{"fold": 1, "train_rows": 10, "test_rows": 5}],
            }
        ),
        encoding="utf-8",
    )

    result = assess_legacy_shadow_ml(tmp_path)

    assert result["status"] == "BLOCKED_LEGACY_PROVENANCE"
    assert result["dataset"]["canonical_registry_status"] == "NOT_REGISTERED"
    assert result["model"]["canonical_registry_status"] == "NOT_REGISTERED"
    assert result["promotion_evaluation"]["permitted"] is False
    assert result["live_decision_influence"] is False


def test_health_artifact_is_hash_addressed_and_side_effect_free(tmp_path: Path) -> None:
    for relative, content in {
        "portfolio/contracts.py": "class PortfolioTarget: pass\n",
        "portfolio/targets.py": "def construct_portfolio_target(): pass\nNO_TRADE = True\n",
        "execution/canonical_state.py": "replay_execution_events assert_replay_deterministic PORTFOLIO_TARGET RISK_APPROVAL EXECUTION_INTENT\n",
        "execution/execution.py": "canonical_chain validate_execution_chain CANONICAL_EXECUTION_CHAIN_REQUIRED\n",
        "research/research_factory.py": "simulate_stage0 run_exact_rejection_review build_walk_forward_manifest purge_bars embargo_bars\n",
        "ml/contracts.py": "manifest\n",
        "ml/labels.py": "labels\n",
        "ml/registry.py": "registry\n",
        "ml/lifecycle.py": "audit_point_in_time_features evaluate_model_freshness\n",
        "tests/test_ml_lifecycle.py": "test\n",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    result = build_reference_integration_health(tmp_path)
    artifact = Path(result["artifact_path"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert artifact.is_file()
    assert artifact.parent.name == result["artifact_hash"]
    assert payload["side_effects"]["orders_submitted"] == 0
    assert payload["live_readiness"] == "NO_GO"
    assert payload["phases"]["I"]["status"] == "NOT_EVALUABLE_SHADOW_ONLY"
    assert build_reference_integration_health(tmp_path)["artifact_hash"] == result[
        "artifact_hash"
    ]
