from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from core.account_inventory import (
    classify_account_inventory,
    expected_inventory_after_canonical_fills,
    grandfathered_inventory_from_expected,
    load_inventory_baseline,
    reconcile_inventory,
    write_inventory_baseline,
)
from utils.common import atomic_write_json, read_json


def _settings(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(
            output_dir=tmp_path / "output",
            checkpoints_dir=tmp_path / "output" / "checkpoints",
        )
    )


def _authority() -> dict[str, object]:
    return {
        "strategy_id": "RR_B60_H5_Z20",
        "strategy_dna": "dna",
        "market": "ETH-EUR",
        "operator_approval_reference": "approval",
    }


def test_inventory_baseline_allows_only_preexisting_quantities(
    tmp_path,
) -> None:
    settings = _settings(tmp_path)
    written = write_inventory_baseline(
        settings,
        holdings=[
            {"symbol": "TAO", "total": "3.0"},
            {"symbol": "NPC", "total": "100"},
        ],
        authority=_authority(),
    )
    baseline, failures = load_inventory_baseline(
        settings,
        authority=_authority(),
    )
    assert failures == ()
    assert baseline == {
        "NPC": Decimal("100"),
        "TAO": Decimal("3.0"),
    }
    reconciled = reconcile_inventory(
        [
            {"symbol": "TAO", "available": "2.5", "inOrder": "0"},
            {"symbol": "NPC", "available": "100", "inOrder": "0"},
            {"symbol": "ETH", "available": "0.01", "inOrder": "0"},
        ],
        baseline,
    )
    assert reconciled["excess"] == {"ETH": "0.01"}
    assert reconciled["missing_or_reduced"] == {"TAO": "0.5"}
    assert reconciled["reconciled"] is False
    assert written["orders_submitted"] == 0


def test_sub_euro_inventory_dust_does_not_block_account_reconciliation() -> None:
    reconciled = reconcile_inventory(
        [
            {"symbol": "ETH", "available": "0.00000066", "inOrder": "0"},
            {"symbol": "TAO", "available": "0.5", "inOrder": "0"},
        ],
        {"ETH": Decimal("0.0000005"), "TAO": Decimal("3")},
        prices_eur={"ETH": Decimal("1625"), "TAO": Decimal("172")},
        minimum_material_excess_eur=Decimal("1"),
    )

    assert reconciled["excess"] == {}
    assert reconciled["ignored_dust_excess"] == {"ETH": "1.6E-7"}
    assert reconciled["missing_or_reduced"] == {"TAO": "2.5"}
    assert reconciled["reconciled"] is True


def test_inventory_baseline_hash_tampering_fails_closed(tmp_path) -> None:
    settings = _settings(tmp_path)
    written = write_inventory_baseline(
        settings,
        holdings=[{"symbol": "TAO", "total": "3"}],
        authority=_authority(),
    )
    path = written["artifact"]
    payload = dict(read_json(path))
    payload["quantities"]["TAO"] = "4"
    atomic_write_json(path, payload)
    _, failures = load_inventory_baseline(
        settings,
        authority=_authority(),
    )
    assert "PREEXISTING_INVENTORY_HASH_MISMATCH" in failures


def test_canonical_buy_fill_becomes_expected_inventory(tmp_path) -> None:
    settings = _settings(tmp_path)
    write_inventory_baseline(
        settings,
        holdings=[{"symbol": "TAO", "total": "3"}],
        authority=_authority(),
    )
    settings.paths.checkpoints_dir.mkdir(parents=True)
    (settings.paths.checkpoints_dir / "live_execution.jsonl").write_text(
        '{"event_type":"FILL","recorded_at":"2099-01-01T00:00:00Z",'
        '"payload":{"market":"BTC-EUR","side":"BUY",'
        '"quantity":"0.00018"}}\n',
        encoding="utf-8",
    )
    baseline, failures = load_inventory_baseline(
        settings,
        authority=_authority(),
    )

    expected = expected_inventory_after_canonical_fills(settings, baseline)

    assert failures == ()
    assert expected["TAO"] == Decimal("3")
    assert expected["BTC"] == Decimal("0.00018")


def test_managed_fill_is_not_also_classified_as_grandfathered() -> None:
    expected_total = {
        "TAO": Decimal("3"),
        "LINK": Decimal("1.39429176"),
    }
    managed = {"LINK": Decimal("1.39429176")}

    grandfathered = grandfathered_inventory_from_expected(
        expected_total,
        managed,
    )
    rows = classify_account_inventory(
        [
            {
                "symbol": "LINK",
                "available": "22.38980567",
                "inOrder": "1.39429176",
            }
        ],
        baseline=grandfathered,
        managed_quantities=managed,
    )

    assert grandfathered == {"TAO": Decimal("3")}
    assert rows[0]["managed_quantity"] == "1.39429176"
    assert rows[0]["grandfathered_quantity"] == "0"
    assert rows[0]["external_quantity"] == "22.38980567"
    assert rows[0]["autonomous_exit_authority_quantity"] == "1.39429176"


def test_real_holdings_are_never_invisible_to_portfolio_heat() -> None:
    rows = classify_account_inventory(
        [
            {"symbol": "TAO", "available": "0.56", "inOrder": "0"},
            {"symbol": "ETH", "available": "0.01", "inOrder": "0"},
            {"symbol": "SOL", "available": "0.2", "inOrder": "0"},
        ],
        baseline={"TAO": Decimal("0.68")},
        managed_quantities={"ETH": Decimal("0.01")},
    )
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["TAO"]["classification"] == "GRANDFATHERED_INVENTORY"
    assert by_symbol["ETH"]["classification"] == "MANAGED_POSITION"
    assert by_symbol["ETH"]["autonomous_exit_authority_quantity"] == "0.01"
    assert by_symbol["ETH"]["external_quantity_remains_unmanaged"] == "true"
    assert by_symbol["SOL"]["classification"] == "MANUAL_EXTERNAL_POSITION"
    assert all(row["counts_toward_portfolio_heat"] == "true" for row in rows)
