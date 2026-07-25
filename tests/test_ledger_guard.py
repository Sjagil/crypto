from __future__ import annotations

from copy import deepcopy

import pytest

from research.forward_observer import build_forward_hash_chain
from research.ledger_guard import (
    ForwardLedgerIntegrityError,
    audit_forward_ledger,
    audit_forward_ledgers,
)
from utils.common import atomic_write_json, stable_hash


def _observation() -> dict:
    row = {
        "schema_version": "portfolio_forward_observer_v1",
        "observation_id": "observation-1",
        "execution_at": "2026-07-25T00:00:00+00:00",
        "realization_at": "2026-07-26T00:00:00+00:00",
        "net_return": 0.01,
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }
    row["observation_hash"] = stable_hash(row, length=64)
    return row


def _ledger() -> dict:
    observations = [_observation()]
    return {
        "source_candidate_identity": "candidate",
        "strategy_dna_hash": "dna",
        "execution_identity": "execution",
        "forward_start": "2026-07-25T00:00:00+00:00",
        "forward_observations": observations,
        "forward_hash_chain": build_forward_hash_chain(observations),
        "orders_generated": 0,
        "orders_submitted": 0,
        "paper_candidate_permitted": False,
        "live_ready": False,
    }


def test_ledger_guard_accepts_valid_chain_and_set(tmp_path) -> None:
    path = tmp_path / "observer.json"
    atomic_write_json(path, _ledger())

    single = audit_forward_ledger(path)
    aggregate = audit_forward_ledgers([path])

    assert single.observation_count == 1
    assert aggregate["status"] == "PASSED"
    assert aggregate["ledger_count"] == 1
    assert aggregate["observation_count"] == 1
    assert aggregate["orders_generated"] == 0
    assert not aggregate["live_ready"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("observation", "OBSERVATION_CORRUPT"),
        ("chain", "HASH_CHAIN_MISMATCH"),
        ("orders", "ORDERLESS_INVARIANT"),
        ("identity", "IDENTITY_MISSING"),
    ],
)
def test_ledger_guard_fails_closed_on_corruption(
    tmp_path,
    mutation,
    match,
) -> None:
    payload = deepcopy(_ledger())
    if mutation == "observation":
        payload["forward_observations"][0]["net_return"] = -0.20
    elif mutation == "chain":
        payload["forward_hash_chain"]["root_hash"] = "f" * 64
    elif mutation == "orders":
        payload["orders_generated"] = 1
    else:
        payload.pop("execution_identity")
    path = tmp_path / "observer.json"
    atomic_write_json(path, payload)

    with pytest.raises(ForwardLedgerIntegrityError, match=match):
        audit_forward_ledger(path)


def test_ledger_guard_rejects_empty_set() -> None:
    with pytest.raises(
        ForwardLedgerIntegrityError,
        match="FORWARD_LEDGER_SET_EMPTY",
    ):
        audit_forward_ledgers([])
