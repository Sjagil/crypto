"""Build the immutable P1.2.3 Bitvavo L2 forensic/replay report."""

from __future__ import annotations

from pathlib import Path

from reporting.bitvavo_l2_recovery import build_recovery_report
from utils.common import atomic_write_json


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    report = build_recovery_report(
        workspace=workspace,
        v1_snapshot_root=workspace / "data_store" / "context" / "microstructure_hourly",
        raw_root=workspace / "data_store" / "raw" / "bitvavo" / "prospective_pit",
        # Prospective V2 snapshots are now part of the canonical source-neutral
        # ledger.  The earlier auxiliary capture remains immutable evidence but
        # is deliberately not needed to reconstruct the deployed stream.
        auxiliary_snapshot_root=None,
    )
    root = workspace / "output" / "multi_source" / "p1_2_3"
    atomic_write_json(root / "bitvavo_l2_recovery_latest.json", report)
    print(root / "bitvavo_l2_recovery_latest.json")
    print(report["report_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
