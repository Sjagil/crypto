from pathlib import Path

import pytest

from research.evidence_accounting import (
    audit_forward_evidence_accounting,
)
from research.global_trial_accounting import (
    GlobalTrialAccountingError,
    audit_global_trial_accounting,
    deduplicate_evaluation_records,
    resolve_known_trial_count,
)
from research.strategy_registry import ContentAddressedTrialRegistry
from utils.common import atomic_write_json, read_json


def _write_campaign_report(
    path: Path,
    *,
    campaign: str,
    **values: object,
) -> None:
    atomic_write_json(path, {"campaign": campaign, **values})


def _write_storm(
    lab_dir: Path,
    *,
    directory: str,
    campaign: str,
    trial_count: int,
    fingerprint: str,
    search_space_hash: str,
) -> None:
    report = lab_dir / directory / "epoch_1" / "report.json"
    _write_campaign_report(
        report,
        campaign=campaign,
        trial_count=trial_count,
        search_space_hash=search_space_hash,
        total_known_trials=trial_count,
    )
    atomic_write_json(
        lab_dir / directory / "index.json",
        {
            "campaign": campaign,
            "epochs": [
                {
                    "epoch_id": "epoch_1",
                    "data_fingerprint": fingerprint,
                    "strategy_search_space_hash": search_space_hash,
                    "evaluated_strategy_count": trial_count,
                    "report_total_known_trials_at_birth": trial_count,
                    "report": str(report),
                }
            ],
        },
    )


def _synthetic_lab(tmp_path: Path) -> Path:
    lab_dir = tmp_path / "lab"
    reports = lab_dir / "reports"
    _write_campaign_report(
        reports / "diversified_rotation_campaign_v1.json",
        campaign="DIVERSIFIED_ROTATION_V1",
        total_known_trials=1_304,
    )
    _write_campaign_report(
        reports / "portfolio_breakout_campaign_v1.json",
        campaign="PORTFOLIO_BREAKOUT_V1",
        prior_trials_accounted=1_304,
        parameters_tested=8,
        total_known_trials=1_312,
    )
    _write_campaign_report(
        reports / "absolute_momentum_campaign_v1.json",
        campaign="ABSOLUTE_MOMENTUM_V1",
        total_known_trials=16_715,
        exploration_ledger={
            "prior_formal_and_storm_trials": 16_312,
            "absolute_momentum_development_grid": 288,
            "component_ablation_paths": 6,
            "mean_reversion_development_grid": 108,
            "midpoint_risk_budget_path": 1,
        },
    )
    _write_storm(
        lab_dir,
        directory="storm_epochs",
        campaign="PORTFOLIO_STORM_V1",
        trial_count=2,
        fingerprint="portfolio-fingerprint",
        search_space_hash="portfolio-space",
    )
    _write_storm(
        lab_dir,
        directory="signal_storm_epochs",
        campaign="SIGNAL_SYNTHESIS_STORM_V1",
        trial_count=3,
        fingerprint="signal-fingerprint",
        search_space_hash="signal-space",
    )
    registry = ContentAddressedTrialRegistry(
        lab_dir / "strategy_registry" / "example_v1",
        campaign_id="EXAMPLE_V1",
    )
    for fingerprint, dna in (
        ("data-a", "dna-a"),
        ("data-b", "dna-a"),
        ("data-b", "dna-b"),
    ):
        registry.register(
            data_fingerprint=fingerprint,
            strategy_family="EXAMPLE",
            strategy_dna_hash=dna,
            parameters={"dna": dna},
            metrics_at_birth={"net_return": 0.0},
            return_path_hash=f"return-{fingerprint}-{dna}",
            selection_metadata={"selected": False},
        )
    return lab_dir


def test_same_dna_and_data_is_deduplicated_but_new_epoch_counts() -> None:
    records = [
        {
            "campaign_id": "A",
            "strategy_dna_hash": "DNA",
            "data_fingerprint": "DATA-1",
            "content_hash": "hash-1",
        },
        {
            "campaign_id": "A",
            "strategy_dna_hash": "DNA",
            "data_fingerprint": "DATA-1",
            "content_hash": "hash-1",
        },
        {
            "campaign_id": "A",
            "strategy_dna_hash": "DNA",
            "data_fingerprint": "DATA-2",
            "content_hash": "hash-2",
        },
    ]

    result = deduplicate_evaluation_records(records)

    assert result["evaluation_trial_count"] == 2
    assert result["unique_strategy_dna_count"] == 1
    assert result["duplicate_retry_count"] == 1


def test_conflicting_content_for_one_evaluation_fails_closed() -> None:
    base = {
        "campaign_id": "A",
        "strategy_dna_hash": "DNA",
        "data_fingerprint": "DATA",
    }

    with pytest.raises(
        GlobalTrialAccountingError,
        match="GLOBAL_TRIAL_IDENTITY_CONTENT_CONFLICT",
    ):
        deduplicate_evaluation_records(
            [
                {**base, "content_hash": "hash-1"},
                {**base, "content_hash": "hash-2"},
            ]
        )


def test_global_accounting_reconciles_and_snapshots_once(
    tmp_path: Path,
) -> None:
    lab_dir = _synthetic_lab(tmp_path)

    first = audit_global_trial_accounting(
        lab_dir,
        persist=True,
    )
    second = audit_global_trial_accounting(
        lab_dir,
        persist=True,
    )

    assert first["status"] == "PASSED"
    assert first["global_multiple_testing_denominator"] == 1_723
    assert first["unique_strategy_dna_equivalent_count"] == 1_722
    assert first["accounting_root_hash"] == second[
        "accounting_root_hash"
    ]
    index = read_json(Path(first["index"]))
    assert len(index["entries"]) == 1
    assert first["orders_generated"] == 0
    assert first["ai_governance_status"] == (
        "AI_DEVELOPMENT_EMBARGOED"
    )
    assert resolve_known_trial_count(
        lab_dir,
        local_known_trial_count=1_500,
    ) == 1_723


def test_known_trial_count_allows_ordered_clean_bootstrap(
    tmp_path: Path,
) -> None:
    assert resolve_known_trial_count(
        tmp_path / "empty-lab",
        local_known_trial_count=17,
    ) == 17


def test_global_accounting_index_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    lab_dir = _synthetic_lab(tmp_path)
    result = audit_global_trial_accounting(
        lab_dir,
        persist=True,
    )
    index_path = Path(result["index"])
    index = read_json(index_path)
    index["entries"][0]["record_hash"] = "0" * 64
    atomic_write_json(index_path, index)

    with pytest.raises(
        GlobalTrialAccountingError,
        match="GLOBAL_TRIAL_INDEX_RECORD_CORRUPT",
    ):
        audit_global_trial_accounting(
            lab_dir,
            persist=True,
        )


def test_forward_observations_are_reported_but_not_counted_as_trials(
    tmp_path: Path,
) -> None:
    lab_dir = _synthetic_lab(tmp_path)
    observer = lab_dir / "observers" / "family" / "candidate.json"
    atomic_write_json(
        observer,
        {
            "family": "FAMILY",
            "strategy_dna_hash": "dna",
            "status": "FROZEN_FORWARD_RESEARCH",
            "forward_observations": [
                {"timestamp": "2026-07-27T00:00:00Z"},
                {"timestamp": "2026-07-28T00:00:00Z"},
            ],
            "forward_decisions": [{"action": "NO_ENTRY"}],
            "orders_generated": 0,
        },
    )

    result = audit_forward_evidence_accounting(
        lab_dir,
        persist=True,
    )

    assert result["status"] == "PASSED"
    assert result["historical_evaluation_trials"] == 1_723
    assert result["global_multiple_testing_denominator"] == 1_723
    assert result["forward_observation_count"] == 2
    assert result["forward_decision_count"] == 1
    assert result["forward_reselection_event_count"] == 0
    assert not result[
        "forward_observations_counted_in_multiple_testing"
    ]


def test_forward_reselection_is_an_integrity_failure(
    tmp_path: Path,
) -> None:
    lab_dir = _synthetic_lab(tmp_path)
    atomic_write_json(
        lab_dir / "observers" / "family" / "candidate.json",
        {
            "strategy_dna_hash": "dna",
            "forward_reselection_events": [
                {"timestamp": "2026-07-27T00:00:00Z"}
            ],
        },
    )

    result = audit_forward_evidence_accounting(
        lab_dir,
        persist=False,
    )

    assert result["status"] == "FAILED"
    assert result["forward_reselection_event_count"] == 1
