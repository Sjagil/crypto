from __future__ import annotations

from decimal import Decimal

from core.execution_evidence import build_execution_evidence_layers
from utils.common import atomic_write_json


def test_execution_evidence_keeps_theoretical_paper_and_live_separate(
    tmp_path,
) -> None:
    operations = tmp_path / "output" / "operations"
    live = tmp_path / "output" / "live"
    atomic_write_json(
        operations / "daily_opportunity_audit.json",
        {
            "resolved_counterfactual_count": 12,
            "false_breakout_rate": 0.25,
            "mfe_distribution_pct": {"count": 12, "p50": 2.0},
            "mae_distribution_pct": {"count": 12, "p50": -1.0},
            "paper_execution_evidence": {
                "closed_round_trips": 3,
                "paper_gross_expectancy_eur": 1.0,
                "paper_net_expectancy_eur": 0.5,
                "source": "paper.jsonl",
                "by_playbook": {
                    "A": {
                        "paper_net_pnl_eur": "2.0",
                        "paper_fees_eur": "0.4",
                    },
                    "B": {
                        "paper_net_pnl_eur": "-0.5",
                        "paper_fees_eur": "0.2",
                    },
                },
                "by_playbook_dna": {
                    "dna-a": {
                        "paper_net_pnl_eur": "2.0",
                        "paper_fees_eur": "0.4",
                    }
                },
            },
        },
    )
    atomic_write_json(
        live / "strategy_performance.json",
        {
            "integrity_status": "PASSED",
            "strategies": [
                {
                    "strategy_id": "LIVE_A",
                    "closed_trade_count": 1,
                    "open_trade_count": 1,
                    "realised_pnl_eur": "1.25",
                    "unrealised_pnl_eur": "0.50",
                    "net_pnl_eur": "1.75",
                    "fees_paid_eur": "0.10",
                },
                {
                    "strategy_id": "LIVE_B",
                    "closed_trade_count": 2,
                    "open_trade_count": 0,
                    "realised_pnl_eur": "-0.25",
                    "unrealised_pnl_eur": "0",
                    "net_pnl_eur": "-0.25",
                    "fees_paid_eur": "0.20",
                },
            ],
        },
    )

    report = build_execution_evidence_layers(tmp_path)

    assert report["theoretical_signal_pnl"]["gross_or_net_pnl_eur"] is None
    assert report["simulated_execution_pnl"]["net_pnl_eur"] == "1.5"
    assert report["simulated_execution_pnl"]["fees_eur"] == "0.6"
    assert report["actual_live_pnl"]["closed_round_trips"] == 3
    assert Decimal(report["actual_live_pnl"]["net_pnl_eur"]) == Decimal("1.50")
    assert report["actual_live_pnl"]["active_strategy_count"] == 2
    assert report["comparison_policy"]["paper_is_not_reported_as_live"] is True
    assert "dna-a" in report["simulated_execution_pnl"]["by_playbook_dna"]
    assert (
        tmp_path
        / "output"
        / "operations"
        / "execution_evidence_layers.json"
    ).is_file()
