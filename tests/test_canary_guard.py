from __future__ import annotations

from decimal import Decimal

import pytest

from config.settings import ExecutionSettings
from risk.canary_guard import (
    CanaryPolicy,
    InstitutionalCanaryGuard,
    write_canary_policy_manifest,
)


def _enabled_policy() -> CanaryPolicy:
    return CanaryPolicy(
        policy_id="BITVAVO_SPOT_CANARY_V1",
        enabled=True,
        maximum_order_eur=Decimal("5"),
        maximum_total_eur=Decimal("5"),
        maximum_open_positions=1,
        leverage=Decimal("1"),
        shorts_allowed=False,
        autoscale=False,
        spot_only=True,
    )


def test_default_canary_is_registered_but_disabled(
    isolated_settings,
) -> None:
    policy = CanaryPolicy.from_settings(isolated_settings)
    assert policy.maximum_order_eur == Decimal("5")
    assert policy.maximum_total_eur == Decimal("5")
    assert policy.maximum_open_positions == 1
    assert not policy.enabled
    decision = InstitutionalCanaryGuard(policy).assess_buy(
        requested_notional_eur=Decimal("5"),
        current_total_exposure_eur=Decimal("0"),
        current_open_positions=0,
        exchange_minimum_order_eur=Decimal("5"),
    )
    assert not decision.approved
    assert decision.reason_code == "CANARY_DISABLED"


def test_canary_caps_order_and_total_exposure() -> None:
    guard = InstitutionalCanaryGuard(_enabled_policy())
    capped = guard.assess_buy(
        requested_notional_eur=Decimal("50"),
        current_total_exposure_eur=Decimal("0"),
        current_open_positions=0,
        exchange_minimum_order_eur=Decimal("5"),
    )
    assert capped.approved
    assert capped.approved_notional_eur == Decimal("5")
    full = guard.assess_buy(
        requested_notional_eur=Decimal("5"),
        current_total_exposure_eur=Decimal("5"),
        current_open_positions=0,
        exchange_minimum_order_eur=Decimal("5"),
    )
    assert not full.approved
    assert full.approved_notional_eur == Decimal("0")


def test_canary_never_autoscales_to_exchange_minimum() -> None:
    decision = InstitutionalCanaryGuard(_enabled_policy()).assess_buy(
        requested_notional_eur=Decimal("4.99"),
        current_total_exposure_eur=Decimal("0"),
        current_open_positions=0,
        exchange_minimum_order_eur=Decimal("5"),
    )
    assert not decision.approved
    assert decision.reason_code == "CANARY_BELOW_EXCHANGE_MINIMUM"


def test_canary_fails_closed_for_unknown_exposure_or_existing_position() -> None:
    guard = InstitutionalCanaryGuard(_enabled_policy())
    unknown = guard.assess_buy(
        requested_notional_eur=Decimal("5"),
        current_total_exposure_eur=None,
        current_open_positions=0,
        exchange_minimum_order_eur=Decimal("5"),
    )
    existing = guard.assess_buy(
        requested_notional_eur=Decimal("5"),
        current_total_exposure_eur=Decimal("0"),
        current_open_positions=1,
        exchange_minimum_order_eur=Decimal("5"),
    )
    assert unknown.reason_code == "CANARY_EXPOSURE_NOT_RECONCILED"
    assert existing.reason_code == "CANARY_POSITION_LIMIT"


def test_execution_settings_reject_larger_or_autoscaled_canary() -> None:
    with pytest.raises(ValueError):
        ExecutionSettings(maximum_live_order_eur=5, maximum_live_total_eur=4)
    with pytest.raises(ValueError):
        ExecutionSettings(live_canary_autoscale=True)
    with pytest.raises(ValueError):
        ExecutionSettings(maximum_live_order_eur=5.01)


def test_canary_manifest_is_content_stable_and_revision_protected(
    isolated_settings,
    tmp_path,
) -> None:
    path = tmp_path / "canary.json"
    first = write_canary_policy_manifest(isolated_settings, path)
    second = write_canary_policy_manifest(isolated_settings, path)
    assert first["policy_hash"] == second["policy_hash"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HISTORY_REVISION"):
        write_canary_policy_manifest(isolated_settings, path)
