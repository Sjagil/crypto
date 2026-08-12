from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml

from core.inventory_reallocation import (
    evaluate_sell_book,
    target_reallocation_quantity,
    validate_reallocation_authority,
)


def test_sell_book_estimates_full_fill_and_costs() -> None:
    result = evaluate_sell_book(
        market="NPC-EUR",
        quantity=Decimal("1000"),
        bids=(
            ("0.0050", "800"),
            ("0.0049", "500"),
        ),
        asks=(("0.0051", "1000"),),
        quote_volume_24h_eur=Decimal("20000"),
        limits={
            "maximum_spread_bps": 250,
            "maximum_slippage_bps": 100,
            "minimum_visible_ask_depth_eur": 1,
            "minimum_24h_quote_volume_eur": 10000,
            "maximum_visible_liquidity_participation_pct": 100,
        },
    )

    assert result["status"] == "PASSED"
    assert Decimal(result["estimated_gross_eur"]) == Decimal("4.98")
    assert Decimal(result["unfilled_quantity"]) == 0
    assert Decimal(result["estimated_sell_slippage_bps"]) == Decimal("40")
    assert Decimal(result["marketable_limit_price"]) == Decimal("0.0049")


def test_sell_book_fails_closed_on_insufficient_depth() -> None:
    result = evaluate_sell_book(
        market="NPC-EUR",
        quantity=Decimal("2000"),
        bids=(("0.0050", "100"),),
        asks=(("0.0051", "100"),),
        quote_volume_24h_eur=Decimal("20000"),
        limits={
            "maximum_spread_bps": 250,
            "maximum_slippage_bps": 250,
            "minimum_visible_ask_depth_eur": 1,
            "minimum_24h_quote_volume_eur": 10000,
            "maximum_visible_liquidity_participation_pct": 100,
        },
    )

    assert result["status"] == "BLOCKED"
    assert (
        "INSUFFICIENT_VISIBLE_BID_LIQUIDITY"
        in result["blocking_reasons"]
    )


def test_npc_precision_rounds_down_without_overselling() -> None:
    available = Decimal("92961.75322202")
    tradable = available.quantize(
        Decimal("0.001"),
        rounding="ROUND_DOWN",
    )

    assert tradable == Decimal("92961.753")
    assert tradable <= available
    assert available - tradable == Decimal("0.00022202")


def test_target_weight_reallocation_keeps_bounded_inventory() -> None:
    available = Decimal("2.91151182")
    price = Decimal("165.55")
    equity = Decimal("564.287343")

    quantity = target_reallocation_quantity(
        available_quantity=available,
        mark_price_eur=price,
        account_equity_eur=equity,
        target_weight=Decimal("0.20"),
    )
    remaining = available - quantity

    assert quantity > Decimal("2")
    assert float(remaining * price / equity) == pytest.approx(0.20)
    assert target_reallocation_quantity(
        available_quantity=available,
        mark_price_eur=price,
        account_equity_eur=equity,
        target_weight=None,
    ) == available


def test_target_weight_reallocation_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="target weight"):
        target_reallocation_quantity(
            available_quantity=Decimal("1"),
            mark_price_eur=Decimal("100"),
            account_equity_eur=Decimal("500"),
            target_weight=Decimal("1"),
        )


def test_reallocation_authority_requires_exact_persisted_reference(
    tmp_path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "execution_market_exceptions.yaml").write_text(
        yaml.safe_dump(
            {
                "default_policy": "FAIL_CLOSED",
                "markets": {
                    "NPC-EUR": {
                        "approved": True,
                        "spot_only": True,
                        "maximum_order_eur": 5,
                        "maximum_total_exposure_eur": 10,
                        "requires_approved_strategy_dna": True,
                        "requires_natural_signal": True,
                        "approval_reference": (
                            "codex_operator_explicit_tao_npc_exception_20260728"
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        paths=SimpleNamespace(project_root=tmp_path)
    )
    approved = validate_reallocation_authority(
        settings,
        market="NPC-EUR",
        approval_reference=(
            "codex_operator_explicit_tao_npc_exception_20260728"
        ),
    )
    rejected = validate_reallocation_authority(
        settings,
        market="NPC-EUR",
        approval_reference="wrong-reference",
    )

    assert approved["approved"] is True
    assert approved["strategy_performance_attribution"] is False
    assert approved["approval_phrase_stored"] is False
    assert rejected["approved"] is False
    assert "APPROVAL_REFERENCE_MISMATCH" in rejected["failures"]
