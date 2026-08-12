from __future__ import annotations

from core.strategy_evidence_watch import build_strategy_evidence_watch
from utils.common import atomic_write_json


def _fixtures(tmp_path) -> None:
    atomic_write_json(
        tmp_path / "config" / "live_playbook_authority.json",
        {
            "approved_playbooks": [
                {
                    "playbook_id": "NEG",
                    "playbook_dna": "neg-dna",
                    "family": "MOMENTUM",
                    "active": True,
                },
                {
                    "playbook_id": "POS",
                    "playbook_dna": "pos-dna",
                    "family": "REVERSAL",
                    "active": True,
                },
                {
                    "playbook_id": "NEW",
                    "playbook_dna": "new-dna",
                    "family": "RANGE",
                    "active": True,
                },
            ]
        },
    )
    atomic_write_json(
        tmp_path / "output" / "live" / "strategy_performance.json",
        {
            "integrity_status": "PASSED",
            "strategies": [
                {
                    "strategy_id": "NEG",
                    "closed_trade_count": 0,
                    "open_trade_count": 0,
                },
                {
                    "strategy_id": "EXACT",
                    "strategy_family": "EXACT_DNA",
                    "closed_trade_count": 0,
                    "open_trade_count": 1,
                },
            ],
        },
    )


def test_strategy_evidence_watch_is_observational_and_sample_aware(tmp_path) -> None:
    _fixtures(tmp_path)
    report = build_strategy_evidence_watch(
        tmp_path,
        execution_evidence={
            "simulated_execution_pnl": {
                "by_playbook": {
                    "NEG": {
                        "closed_round_trips": 25,
                        "paper_net_expectancy_eur": -0.2,
                    },
                    "POS": {
                        "closed_round_trips": 8,
                        "paper_net_expectancy_eur": 0.5,
                    },
                },
                "by_playbook_dna": {
                    "neg-dna": {
                        "playbook_id": "NEG",
                        "closed_round_trips": 25,
                        "paper_net_expectancy_eur": -0.2,
                    },
                    "pos-dna": {
                        "playbook_id": "POS",
                        "closed_round_trips": 8,
                        "paper_net_expectancy_eur": 0.5,
                    },
                },
            }
        },
    )

    rows = {row["strategy_id"]: row for row in report["strategies"]}
    assert rows["NEG"]["recommendation"] == "PAPER_NEGATIVE_REVIEW_REQUIRED"
    assert rows["POS"]["recommendation"] == "PRIORITIZE_MORE_EVIDENCE"
    assert rows["NEW"]["recommendation"] == "COLLECT_PAPER_AND_SHADOW"
    assert rows["EXACT"]["live"]["open_positions"] == 1
    assert report["policy"]["automatic_authority_changes"] is False
    assert all(row["automatic_authority_change"] is False for row in rows.values())
    assert (
        tmp_path
        / "output"
        / "operations"
        / "strategy_evidence_watch.json"
    ).is_file()


def test_family_history_does_not_demote_a_migrated_current_dna(tmp_path) -> None:
    _fixtures(tmp_path)
    report = build_strategy_evidence_watch(
        tmp_path,
        execution_evidence={
            "simulated_execution_pnl": {
                "by_playbook": {
                    "NEG": {
                        "closed_round_trips": 40,
                        "paper_net_expectancy_eur": -1.0,
                    }
                },
                "by_playbook_dna": {
                    "old-neg-dna": {
                        "playbook_id": "NEG",
                        "closed_round_trips": 40,
                        "paper_net_expectancy_eur": -1.0,
                    }
                },
            }
        },
    )

    rows = {row["strategy_id"]: row for row in report["strategies"]}
    assert rows["NEG"]["recommendation"] == "COLLECT_PAPER_AND_SHADOW"
    assert rows["NEG"]["paper"]["closed_round_trips"] == 0
    assert rows["NEG"]["paper_family_history"]["closed_round_trips"] == 40
    assert rows["NEG"]["paper_family_history"][
        "may_not_demote_current_dna"
    ] is True


def test_strategy_evidence_watch_fails_closed_on_accounting_integrity(tmp_path) -> None:
    _fixtures(tmp_path)
    atomic_write_json(
        tmp_path / "output" / "live" / "strategy_performance.json",
        {"integrity_status": "FAILED", "strategies": []},
    )

    report = build_strategy_evidence_watch(tmp_path, execution_evidence={})

    assert report["live_accounting_integrity"] == "NOT_PASSED"
    assert {
        row["recommendation"] for row in report["strategies"]
    } == {"DISABLE_NEW_ENTRIES_RECOMMENDED"}
