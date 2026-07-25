"""Cryptographic preflight audit for append-only forward research ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research.forward_observer import (
    ForwardHistoryRevisionError,
    build_forward_hash_chain,
)
from utils.common import read_json, stable_hash


class ForwardLedgerIntegrityError(RuntimeError):
    """Raised when a persisted forward ledger is incomplete or corrupted."""


@dataclass(frozen=True, slots=True)
class ForwardLedgerAudit:
    path: str
    strategy_dna_hash: str
    observation_count: int
    chain_root_hash: str
    passed: bool = True


def audit_forward_ledger(path: Path | str) -> ForwardLedgerAudit:
    """Verify identity, orderlessness, observations and the full hash chain."""

    selected = Path(path)
    if not selected.is_file():
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_MISSING:{selected}"
        )
    payload = read_json(selected)
    required_identity = (
        "source_candidate_identity",
        "strategy_dna_hash",
        "execution_identity",
        "forward_start",
    )
    missing = [
        field for field in required_identity if not payload.get(field)
    ]
    if missing:
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_IDENTITY_MISSING:{selected}:{missing}"
        )
    forbidden = {
        "orders_generated": payload.get("orders_generated", 0),
        "orders_submitted": payload.get("orders_submitted", 0),
        "paper_candidate_permitted": payload.get(
            "paper_candidate_permitted",
            False,
        ),
        "live_ready": payload.get("live_ready", False),
    }
    if any(bool(value) for value in forbidden.values()):
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_ORDERLESS_INVARIANT:{selected}"
        )
    observations = list(payload.get("forward_observations") or [])
    stored_chain = payload.get("forward_hash_chain")
    if stored_chain is None:
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_HASH_CHAIN_MISSING:{selected}"
        )
    try:
        expected_chain = build_forward_hash_chain(observations)
    except ForwardHistoryRevisionError as exc:
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_OBSERVATION_CORRUPT:{selected}:{exc}"
        ) from exc
    if dict(stored_chain) != expected_chain:
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_HASH_CHAIN_MISMATCH:{selected}"
        )
    if int(stored_chain.get("record_count") or 0) != len(observations):
        raise ForwardLedgerIntegrityError(
            f"FORWARD_LEDGER_COUNT_MISMATCH:{selected}"
        )
    return ForwardLedgerAudit(
        path=str(selected),
        strategy_dna_hash=str(payload["strategy_dna_hash"]),
        observation_count=len(observations),
        chain_root_hash=str(expected_chain["root_hash"]),
    )


def audit_forward_ledgers(
    paths: Iterable[Path | str],
) -> dict[str, Any]:
    """Audit every declared ledger and return deterministic preflight evidence."""

    selected = tuple(sorted({str(Path(path)) for path in paths}))
    if not selected:
        raise ForwardLedgerIntegrityError(
            "FORWARD_LEDGER_SET_EMPTY"
        )
    audits = [audit_forward_ledger(path) for path in selected]
    return {
        "status": "PASSED",
        "audit": "FORWARD_LEDGER_CRYPTOGRAPHIC_PREFLIGHT_V1",
        "ledger_count": len(audits),
        "observation_count": sum(
            row.observation_count for row in audits
        ),
        "ledger_set_hash": stable_hash(
            [
                {
                    "path": row.path,
                    "strategy_dna_hash": row.strategy_dna_hash,
                    "observation_count": row.observation_count,
                    "chain_root_hash": row.chain_root_hash,
                }
                for row in audits
            ],
            length=64,
        ),
        "ledgers": [
            {
                "path": row.path,
                "strategy_dna_hash": row.strategy_dna_hash,
                "observation_count": row.observation_count,
                "chain_root_hash": row.chain_root_hash,
                "passed": row.passed,
            }
            for row in audits
        ],
        "orders_generated": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


__all__ = [
    "ForwardLedgerAudit",
    "ForwardLedgerIntegrityError",
    "audit_forward_ledger",
    "audit_forward_ledgers",
]
