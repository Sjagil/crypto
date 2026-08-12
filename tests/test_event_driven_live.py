from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import PathSettings, Settings
from core.contracts import ExecutionBlocked, OrderType
from core.event_driven_live import (
    _candidate_selection_score,
    _microstructure_exit_reason,
    _risk_limited_entry_notional,
    _wallet_exposure,
    approval_phrase,
    approve_playbook_live,
    execute_event_driven_live_once,
    execution_block_reason_code,
    execution_block_requires_authority_deactivation,
    is_playbook_opportunity_authorized,
    migrate_playbook_live_capital_level_2,
    playbook_authority_status,
    playbook_catalog,
)
from core.inventory_risk_override import evaluate_inventory_risk_override
from execution.execution import ExecutionMarketRules


def _settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _opportunity() -> dict[str, Any]:
    playbook = playbook_catalog()[0]
    return {
        "opportunity_id": "opportunity-1",
        "market": "BTC-EUR",
        "playbook_id": playbook["playbook_id"],
        "family": playbook["family"],
        "playbook_dna": playbook["playbook_dna"],
        "parameter_band_hash": playbook["parameter_band_hash"],
        "parameter_band_status": "VALIDATED",
        "validated_parameters": {"score": 75.0, "confirmations": 4.0},
        "state": "ENTRY_READY",
        "score": 75.0,
        "tier": "A",
        "hard_blockers": [],
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit_1": 103.0,
        "take_profit_2": 106.0,
        "time_stop_minutes": 30,
        "realtime_inputs": {
            "taker_buy_ratio_1m": 0.62,
            "ofi_1m": 0.2,
        },
    }


def _realtime() -> dict[str, Any]:
    return {
        "markets": [
            {
                "market": "BTC-EUR",
                "price": 100.0,
                "fresh": True,
                "sequence_valid": True,
                "estimated_buy_slippage_bps": 2.0,
                "book": {"best_bid": 100.0, "best_ask": 100.01},
            }
        ]
    }


def test_entry_notional_is_capped_by_structural_stop_risk() -> None:
    assert _risk_limited_entry_notional(
        desired_notional_eur=Decimal("25"),
        entry_price=Decimal("100"),
        stop_price=Decimal("50"),
        maximum_risk_eur=Decimal("2"),
    ) == Decimal("4")
    assert _risk_limited_entry_notional(
        desired_notional_eur=Decimal("25"),
        entry_price=Decimal("100"),
        stop_price=Decimal("99"),
        maximum_risk_eur=Decimal("2"),
    ) == Decimal("25")


def test_preexisting_inventory_counts_toward_total_wallet_heat(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    result = _wallet_exposure(
        settings,
        [
            {"symbol": "EUR", "available": "20.25", "inOrder": "0"},
            {"symbol": "TAO", "available": "0.68", "inOrder": "0"},
        ],
        {"TAO-EUR": {"price": "168.00"}},
    )
    assert Decimal(result["total_wallet_asset_exposure_eur"]) == Decimal(
        "114.24"
    )
    assert result["largest_asset"] == "TAO"
    assert Decimal(result["wallet_concentration_fraction"]) > Decimal("0.80")
    assert result["status"] == "READY"


def test_tao_override_grandfathers_only_the_approved_existing_quantity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    config = settings.paths.project_root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "inventory_risk_override.json").write_text(
        json.dumps(
            {
                "active": True,
                "asset": "TAO",
                "maximum_quantity": "0.68008471",
                "maximum_additional_managed_exposure_eur": "10",
                "allow_inventory_increase": False,
            }
        ),
        encoding="utf-8",
    )

    accepted = evaluate_inventory_risk_override(
        settings,
        [{"symbol": "TAO", "available": "0.68", "inOrder": "0"}],
    )
    increased = evaluate_inventory_risk_override(
        settings,
        [{"symbol": "TAO", "available": "0.69", "inOrder": "0"}],
    )

    assert accepted["active"] is True
    assert accepted["allow_inventory_increase"] is False
    assert increased["active"] is False


@pytest.mark.parametrize(
    ("realtime_patch", "expected"),
    [
        (
            {"windows": {"1m": {"taker_buy_ratio": 0.44}}, "ofi_1m": -0.1},
            "ORDERFLOW_EXHAUSTION",
        ),
        (
            {"windows": {"1m": {"taker_buy_ratio": 0.47, "cvd_quote_eur": -5}}},
            "NEGATIVE_CVD_REVERSAL",
        ),
        (
            {"book": {"mlobi_top_10": -0.2, "bid_depth_eur_top_10": 50}},
            "BID_SUPPORT_WITHDRAWAL",
        ),
        ({"book": {"spread_bps": 30}}, "SPREAD_EXPANSION"),
        (
            {"windows": {"1m": {"return": -0.004}, "5m": {"return": -0.007}}},
            "MOMENTUM_DECAY",
        ),
    ],
)
def test_microstructure_position_exits_are_causal(
    realtime_patch: dict[str, Any], expected: str
) -> None:
    position = {
        "entry_bid_depth_eur_top_10": "100",
        "entry_spread_bps": "5",
    }
    realtime = {
        "fresh": True,
        "sequence_valid": True,
        **realtime_patch,
    }
    assert _microstructure_exit_reason(position, realtime) == expected


def test_stale_microstructure_cannot_trigger_soft_exit() -> None:
    position = {
        "entry_bid_depth_eur_top_10": "100",
        "entry_spread_bps": "5",
    }
    stale = {
        "fresh": False,
        "sequence_valid": True,
        "windows": {"1m": {"taker_buy_ratio": 0.1}},
        "ofi_1m": -1.0,
    }

    assert _microstructure_exit_reason(position, stale) is None


def test_only_credential_scope_blocks_revoke_playbook_authority() -> None:
    routine = ExecutionBlocked("live canary daily new-order limit reached")
    unsafe = ExecutionBlocked("LIVE_BLOCKED_UNSAFE_CREDENTIAL_SCOPE")
    excessive_risk = ExecutionBlocked(
        "event entry exceeds maximum risk per trade"
    )

    assert execution_block_reason_code(routine) == (
        "DAILY_NEW_ORDER_LIMIT_REACHED"
    )
    assert execution_block_requires_authority_deactivation(routine) is False
    assert execution_block_reason_code(unsafe) == "UNSAFE_CREDENTIAL_SCOPE"
    assert execution_block_reason_code(excessive_risk) == (
        "MAXIMUM_RISK_PER_TRADE_EXCEEDED"
    )
    assert execution_block_requires_authority_deactivation(unsafe) is True


def _write_authority(settings: Settings, *, active: bool) -> None:
    path = settings.paths.project_root / "config" / "live_playbook_authority.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    playbook = playbook_catalog()[0]
    path.write_text(
        json.dumps(
            {
                "active": active,
                "approved_playbooks": [
                    {
                        "active": True,
                        "playbook_id": playbook["playbook_id"],
                        "family": playbook["family"],
                        "playbook_dna": playbook["playbook_dna"],
                        "parameter_band": playbook["parameter_band"],
                        "parameter_band_hash": playbook[
                            "parameter_band_hash"
                        ],
                        "markets": ["BTC-EUR"],
                        "maximum_order_eur": 10,
                        "autoscale": False,
                        "authority_level": "VALIDATED_PLAYBOOK",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_validated_playbook_authority_accepts_bound_variant_only() -> None:
    playbook = playbook_catalog()[0]
    parameters = {
        key: (float(bounds[0]) + float(bounds[1])) / 2.0
        for key, bounds in playbook["parameter_band"].items()
    }
    authority = {
        "approved_playbooks": [
            {
                "active": True,
                "authority_level": "VALIDATED_PLAYBOOK",
                "family": playbook["family"],
                "playbook_dna": playbook["playbook_dna"],
                "parameter_band": playbook["parameter_band"],
                "parameter_band_hash": playbook["parameter_band_hash"],
                "markets": ["BTC-EUR"],
                "maximum_order_eur": 10,
                "autoscale": False,
            }
        ]
    }
    variant = {
        **_opportunity(),
        "base_playbook_dna": playbook["playbook_dna"],
        "playbook_parameters": parameters,
    }
    variant["playbook_dna"] = __import__(
        "utils.common", fromlist=["stable_hash"]
    ).stable_hash(
        {
            "base_playbook_dna": playbook["playbook_dna"],
            "family": playbook["family"],
            "playbook_id": playbook["playbook_id"],
            "playbook_parameters": parameters,
        },
        length=64,
    )

    assert is_playbook_opportunity_authorized(authority, variant) is True
    variant["playbook_parameters"] = {**parameters, "score": 101.0}
    assert is_playbook_opportunity_authorized(authority, variant) is False


def test_playbook_approval_is_bounded_and_does_not_store_phrase(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    service = settings.paths.output_dir / "live" / "autonomous_live_authority.json"
    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_text(json.dumps({"markets": ["BTC-EUR"]}), encoding="utf-8")
    selected = playbook_catalog()[0]

    with pytest.raises(PermissionError, match="does not match"):
        approve_playbook_live(
            settings,
            playbook_id=selected["playbook_id"],
            markets=("BTC-EUR",),
            approval="wrong",
        )
    payload = approve_playbook_live(
        settings,
        playbook_id=selected["playbook_id"],
        markets=("BTC-EUR",),
        approval=approval_phrase(selected["playbook_id"]),
    )
    stored = (
        settings.paths.project_root / "config" / "live_playbook_authority.json"
    ).read_text(encoding="utf-8")

    assert payload["status"] == "APPROVED"
    assert payload["maximum_order_eur"] == "25"
    assert payload["evidence_multiplier"] == "0.40"
    assert payload["maximum_effective_order_eur"] == "10.00"
    authority = json.loads(stored)
    approved = authority["approved_playbooks"][0]
    assert approved["strategy_role"] == "EXPERIMENTAL_CANARY"
    assert approved["maximum_family_positions"] == 1
    assert approval_phrase(selected["playbook_id"]) not in stored
    assert playbook_authority_status(settings)["active"] is True


def test_evidence_weight_changes_selection_not_entry_validity() -> None:
    opportunity = {
        "score": 80.0,
        "weighted_timeframe_score": 0.75,
        "microstructure_state": "SUPPORTIVE",
        "execution_economics": {
            "expected_net_value_bps": 45.0,
            "cost_to_target_2_ratio": 0.25,
        },
        "execution_scorecard": {"friction_liquidity": 4.0},
    }
    canary = _candidate_selection_score(
        opportunity, {"evidence_multiplier": "0.40"}
    )
    established = _candidate_selection_score(
        opportunity, {"evidence_multiplier": "1.00"}
    )

    assert established > canary
    assert 0.0 <= canary <= 1.0


def test_level_2_migration_repairs_metadata_only_playbook_identity(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    authority_path = (
        settings.paths.project_root / "config" / "live_playbook_authority.json"
    )
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    selected = playbook_catalog()[0]
    authority_path.write_text(
        json.dumps(
            {
                "active": True,
                "autoscale": False,
                "spot_only": True,
                "approved_playbooks": [
                    {
                        "active": True,
                        "playbook_id": selected["playbook_id"],
                        "family": selected["family"],
                        "playbook_dna": "old-metadata-only-dna",
                        "parameter_band": selected["parameter_band"],
                        "parameter_band_hash": selected[
                            "parameter_band_hash"
                        ],
                        "markets": ["BTC-EUR"],
                        "autoscale": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = migrate_playbook_live_capital_level_2(
        settings,
        approval_phrase="I APPROVE LIVE CAPITAL LEVEL 2",
    )
    stored = json.loads(authority_path.read_text(encoding="utf-8"))

    assert result["identity_migration_count"] == 1
    assert result["approved_playbook_count"] == 1
    assert stored["approved_playbooks"][0]["playbook_dna"] == selected[
        "playbook_dna"
    ]
    assert stored["approved_playbooks"][0]["previous_playbook_dna"] == (
        "old-metadata-only-dna"
    )
    assert stored["maximum_order_eur"] == "25"
    assert stored["maximum_open_positions"] == 3


def test_level_2_migration_does_not_reapprove_changed_parameter_band(
    isolated_settings: Settings,
    tmp_path: Path,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    authority_path = (
        settings.paths.project_root / "config" / "live_playbook_authority.json"
    )
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    selected = playbook_catalog()[0]
    authority_path.write_text(
        json.dumps(
            {
                "active": True,
                "autoscale": False,
                "spot_only": True,
                "approved_playbooks": [
                    {
                        "active": True,
                        "playbook_id": selected["playbook_id"],
                        "family": selected["family"],
                        "playbook_dna": "old-semantic-dna",
                        "parameter_band": {"score": [0, 1]},
                        "parameter_band_hash": "changed-band-hash",
                        "markets": ["BTC-EUR"],
                        "autoscale": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = migrate_playbook_live_capital_level_2(
        settings,
        approval_phrase="I APPROVE LIVE CAPITAL LEVEL 2",
    )
    stored = json.loads(authority_path.read_text(encoding="utf-8"))

    assert result["identity_migration_count"] == 0
    assert stored["approved_playbooks"][0]["playbook_dna"] == (
        "old-semantic-dna"
    )


@pytest.mark.asyncio
async def test_disabled_authority_makes_zero_private_requests(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=False)

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("private client must not be built")

    monkeypatch.setattr("core.event_driven_live.build_live_client", forbidden)
    result = await execute_event_driven_live_once(
        settings,
        opportunities=[_opportunity()],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
    )

    assert result["status"] == "AUTHORITY_DISABLED"
    assert result["orders_submitted_this_cycle"] == 0


@pytest.mark.asyncio
async def test_no_order_state_explains_unauthorized_entry_ready_candidate(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)
    opportunity = _opportunity()
    opportunity["market"] = "ETH-EUR"

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("unauthorized candidate must not build private client")

    monkeypatch.setattr("core.event_driven_live.build_live_client", forbidden)
    result = await execute_event_driven_live_once(
        settings,
        opportunities=[opportunity],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
    )

    assert result["status"] == "READY"
    assert result["reason_code"] == "NO_APPROVED_EVENT_ENTRY_READY"
    assert result["private_exchange_requests"] == 0
    assert result["entry_ready_rejections"] == [
        {
            "opportunity_id": "opportunity-1",
            "market": "ETH-EUR",
            "playbook_id": "MOMENTUM_BREAKOUT_V1",
            "tier": "A",
            "score": 75.0,
            "reasons": ["PLAYBOOK_OR_MARKET_NOT_AUTHORIZED"],
            "hard_blockers": [],
        }
    ]


@pytest.mark.asyncio
async def test_maker_partial_is_cancelled_recorded_and_not_chased(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.submitted: list[Any] = []
            self.cancelled = 0
            self.gets = 0
            self.recorded = 0

        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "100"},
                {"symbol": "BTC", "available": "0"},
            ]

        async def arm_cancel_on_disconnect(self, **_: object) -> dict[str, str]:
            return {"status": "armed"}

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(self, market: str) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                # Keep this test above two buffered venue minima so it
                # exercises the larger-order maker-partial lifecycle. Tiny
                # canaries use atomic FOK instead.
                minimum_order_value_eur=Decimal("2"),
                quantity_decimals=8,
                notional_decimals=2,
                tick_size=Decimal("0.01"),
            )

        async def submit_order(self, intent: Any, **_: object) -> dict[str, str]:
            self.submitted.append(intent)
            if intent.order_type is OrderType.STOP_LOSS:
                return {
                    "orderId": "stop-1",
                    "market": "BTC-EUR",
                    "side": "sell",
                    "status": "awaitingTrigger",
                    "filledAmount": "0",
                }
            return {
                "orderId": "maker-1",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "open",
                "filledAmount": "0",
            }

        def client_order_id_for(self, _: str) -> str:
            return "client-maker-1"

        async def get_order(self, **_: object) -> dict[str, str]:
            self.gets += 1
            return {
                "orderId": "maker-1",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "partiallyFilled" if self.gets == 1 else "canceled",
                "filledAmount": "0.03",
                "filledAmountQuote": "3.00",
                "price": "100",
            }

        async def cancel_order(self, **_: object) -> dict[str, str]:
            self.cancelled += 1
            return {"status": "canceled"}

        def record_final_fill(self, *_: object, **__: object) -> bool:
            self.recorded += 1
            return True

    client = Client()
    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: client,
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )
    monkeypatch.setattr("core.event_driven_live.MAKER_WAIT_SECONDS", 0)

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[_opportunity()],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )

    assert len(client.submitted) == 2
    assert client.submitted[0].post_only is True
    assert client.submitted[1].order_type is OrderType.STOP_LOSS
    assert client.cancelled == 1
    assert client.recorded == 1
    assert result["status"] == "POSITION_OPENED"
    assert result["fills_verified_this_cycle"] == 1
    assert result["positions"]["opportunity-1"]["quantity"] == "0.03"
    assert result["events"][0]["event"] == "LIVE_ORDER_INTENT_CREATED"
    assert any(
        event["event"] == "LIVE_ORDER_PARTIALLY_FILLED"
        for event in result["events"]
    )


@pytest.mark.asyncio
async def test_restart_recovers_acknowledged_late_fill_without_new_order(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)
    live = settings.paths.output_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    opportunity = _opportunity()
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps({"opportunities": {"opportunity-1": opportunity}}),
        encoding="utf-8",
    )
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event_type": "ORDER_INTENT",
                    "recorded_at": "2026-08-03T12:00:00+00:00",
                    "payload": {
                        "intent_id": "intent-1",
                        "client_order_id": "client-1",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "quantity": "0.1",
                        "strategy_id": opportunity["playbook_id"],
                        "strategy_dna_hash": opportunity["playbook_dna"],
                        "signal_id": "opportunity-1",
                    },
                },
                {
                    "event_type": "ORDER_ACKNOWLEDGED",
                    "recorded_at": "2026-08-03T12:00:01+00:00",
                    "payload": {
                        "intent_id": "intent-1",
                        "client_order_id": "client-1",
                        "order_id": "order-1",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "strategy_id": opportunity["playbook_id"],
                        "strategy_dna_hash": opportunity["playbook_dna"],
                        "signal_id": "opportunity-1",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.submitted = 0
            self.recorded = 0

        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "40"},
                {"symbol": "BTC", "available": "0.1"},
            ]

        async def arm_cancel_on_disconnect(self, **_: object) -> dict[str, str]:
            return {"status": "armed"}

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def get_order(self, **kwargs: object) -> dict[str, str]:
            if kwargs.get("client_order_id") != "client-1":
                return {
                    "orderId": "stop-1",
                    "market": "BTC-EUR",
                    "side": "sell",
                    "status": "awaitingTrigger",
                    "filledAmount": "0",
                }
            return {
                "orderId": "order-1",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "filled",
                "filledAmount": "0.1",
                "filledAmountQuote": "10",
                "price": "100",
                "updated": str(int(datetime(2026, 8, 3, 12, 0, tzinfo=UTC).timestamp() * 1000)),
            }

        def record_final_fill(self, *_: object, **__: object) -> bool:
            self.recorded += 1
            return True

        async def execution_market_rules(
            self,
            market: str,
        ) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
                notional_decimals=2,
                tick_size=Decimal("0.01"),
            )

        def client_order_id_for(self, _: str) -> str:
            return "client-stop-1"

        async def submit_order(self, *_: object, **__: object) -> dict[str, str]:
            self.submitted += 1
            return {
                "orderId": "stop-1",
                "market": "BTC-EUR",
                "side": "sell",
                "status": "awaitingTrigger",
                "filledAmount": "0",
            }

    client = Client()
    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: client,
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=False,
        observed_at=datetime(2026, 8, 3, 12, 5, tzinfo=UTC),
    )

    assert client.submitted == 1
    assert client.recorded == 1
    assert result["status"] == "ENTRIES_DISABLED"
    assert result["positions"]["opportunity-1"]["quantity"] == "0.1"
    assert result["positions"]["opportunity-1"]["recovered_after_restart"] is True
    assert any(
        event["event"] == "LIVE_POSITION_RECOVERED_AFTER_RESTART"
        for event in result["events"]
    )


@pytest.mark.asyncio
async def test_restart_cancels_open_partial_and_protects_real_inventory(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)
    live = settings.paths.output_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    opportunity = _opportunity()
    (live / "opportunity_lifecycle_state.json").write_text(
        json.dumps({"opportunities": {"opportunity-1": opportunity}}),
        encoding="utf-8",
    )
    ledger = settings.paths.checkpoints_dir / "live_execution.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event_type": "ORDER_INTENT",
                    "recorded_at": "2026-08-03T12:00:00+00:00",
                    "payload": {
                        "intent_id": "intent-partial",
                        "client_order_id": "client-partial",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "quantity": "0.1",
                        "strategy_id": opportunity["playbook_id"],
                        "strategy_dna_hash": opportunity["playbook_dna"],
                        "signal_id": "opportunity-1",
                    },
                },
                {
                    "event_type": "ORDER_ACKNOWLEDGED",
                    "recorded_at": "2026-08-03T12:00:01+00:00",
                    "payload": {
                        "intent_id": "intent-partial",
                        "client_order_id": "client-partial",
                        "order_id": "order-partial",
                        "market": "BTC-EUR",
                        "side": "BUY",
                        "strategy_id": opportunity["playbook_id"],
                        "strategy_dna_hash": opportunity["playbook_dna"],
                        "signal_id": "opportunity-1",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.cancelled = 0
            self.stop_submitted = 0
            self.record_kwargs: dict[str, object] = {}

        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "40"},
                {"symbol": "BTC", "available": "0.04"},
            ]

        async def arm_cancel_on_disconnect(self, **_: object) -> dict[str, str]:
            return {"status": "armed"}

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def get_order(self, **kwargs: object) -> dict[str, str]:
            if kwargs.get("client_order_id") != "client-partial":
                return {
                    "orderId": "stop-partial",
                    "market": "BTC-EUR",
                    "side": "sell",
                    "status": "awaitingTrigger",
                    "filledAmount": "0",
                }
            return {
                "orderId": "order-partial",
                "clientOrderId": "client-partial",
                "market": "BTC-EUR",
                "side": "buy",
                "status": "canceled" if self.cancelled else "partiallyFilled",
                "filledAmount": "0.04",
                "filledAmountQuote": "4",
                "price": "100",
            }

        async def cancel_order(self, **_: object) -> dict[str, str]:
            self.cancelled += 1
            return {"status": "canceled"}

        def record_final_fill(self, *_: object, **kwargs: object) -> bool:
            self.record_kwargs = kwargs
            return True

        async def execution_market_rules(
            self,
            market: str,
        ) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("2"),
                quantity_decimals=8,
                notional_decimals=2,
                tick_size=Decimal("0.01"),
            )

        def client_order_id_for(self, _: str) -> str:
            return "client-stop-partial"

        async def submit_order(self, *_: object, **__: object) -> dict[str, str]:
            self.stop_submitted += 1
            return {
                "orderId": "stop-partial",
                "market": "BTC-EUR",
                "side": "sell",
                "status": "awaitingTrigger",
                "filledAmount": "0",
            }

    client = Client()
    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: client,
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=False,
        observed_at=datetime(2026, 8, 3, 12, 5, tzinfo=UTC),
    )

    assert client.cancelled == 1
    assert client.stop_submitted == 1
    assert client.record_kwargs["allow_terminal_partial"] is True
    assert result["positions"]["opportunity-1"]["quantity"] == "0.04"
    assert result["positions"]["opportunity-1"][
        "recovered_after_restart"
    ] is True


@pytest.mark.asyncio
async def test_venue_minimum_cannot_override_two_euro_risk_cap(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "100"},
                {"symbol": "BTC", "available": "0"},
            ]

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(
            self,
            market: str,
        ) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
                notional_decimals=2,
                tick_size=Decimal("0.01"),
            )

        async def submit_order(self, *_: object, **__: object) -> dict[str, str]:
            raise AssertionError("risk-blocked entry must not reach Bitvavo")

    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: Client(),
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )
    opportunity = _opportunity()
    opportunity["stop_loss"] = 50.0

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[opportunity],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )

    assert result["status"] == "ENTRY_BLOCKED"
    assert result["reason_code"] == "MAXIMUM_RISK_PER_TRADE_EXCEEDED"
    assert Decimal(result["planned_risk_eur"]) > Decimal("2")
    assert result["maximum_risk_per_trade_eur"] == "2"
    assert result["orders_submitted_this_cycle"] == 0


@pytest.mark.asyncio
async def test_exact_ten_euro_canary_never_uses_unbounded_market_fallback(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.submitted: list[Any] = []
            self.kwargs: list[dict[str, Any]] = []

        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "50"},
                {"symbol": "BTC", "available": "0"},
            ]

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(self, market: str) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=0,
                tick_size=Decimal("0.01"),
            )

        async def submit_order(self, intent: Any, **kwargs: Any) -> dict[str, str]:
            self.submitted.append(intent)
            self.kwargs.append(kwargs)
            raise AssertionError("unbounded market fallback must not be submitted")

    client = Client()
    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: client,
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )
    realtime = _realtime()
    realtime["markets"][0].update(
        {
            "price": 30.0,
            "estimated_buy_slippage_bps": 2.0,
            "book": {"best_bid": 29.99, "best_ask": 30.0},
        }
    )

    opportunity = _opportunity()
    opportunity["stop_loss"] = 28.0
    result = await execute_event_driven_live_once(
        settings,
        opportunities=[opportunity],
        realtime_snapshot=realtime,
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )

    assert client.submitted == []
    assert client.kwargs == []
    assert result["status"] == "ENTRY_BLOCKED"
    assert result["reason_code"] == "SAFE_VENUE_MINIMUM_EXCEEDS_LIVE_CAP"
    assert result["protectable_minimum_order_eur"] == "30.00"
    assert result["orders_submitted_this_cycle"] == 0
    assert result["fills_verified_this_cycle"] == 0


@pytest.mark.asyncio
async def test_temporary_market_rules_failure_preserves_live_authority(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "50"},
                {"symbol": "BTC", "available": "0"},
            ]

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(self, _: str) -> ExecutionMarketRules:
            raise ExecutionBlocked("temporary public rules read failure")

    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: Client(),
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[_opportunity()],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
    )

    assert result["status"] == "EXECUTION_RULES_BLOCKED"
    assert result["orders_submitted_this_cycle"] == 0
    authority = json.loads(
        (
            settings.paths.project_root
            / "config"
            / "live_playbook_authority.json"
        ).read_text(encoding="utf-8")
    )
    assert authority["active"] is True


@pytest.mark.asyncio
async def test_routine_entry_policy_block_preserves_live_authority(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "50"},
                {"symbol": "BTC", "available": "0"},
            ]

        async def arm_cancel_on_disconnect(self, **_: object) -> dict[str, str]:
            return {"status": "armed"}

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            assert markets == ("BTC-EUR",)
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(
            self, market: str
        ) -> ExecutionMarketRules:
            assert market == "BTC-EUR"
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
                tick_size=Decimal("0.01"),
            )

        async def submit_order(self, *_: object, **__: object) -> dict[str, str]:
            raise ExecutionBlocked("live canary daily new-order limit reached")

    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: Client(),
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[_opportunity()],
        realtime_snapshot=_realtime(),
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
    )

    assert result["status"] == "ENTRY_BLOCKED"
    assert result["reason_code"] == "DAILY_NEW_ORDER_LIMIT_REACHED"
    assert result["orders_submitted_this_cycle"] == 0
    authority = json.loads(
        (
            settings.paths.project_root
            / "config"
            / "live_playbook_authority.json"
        ).read_text(encoding="utf-8")
    )
    assert authority["active"] is True


@pytest.mark.asyncio
async def test_partial_exit_keeps_residual_position_managed(
    isolated_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(isolated_settings, tmp_path)
    _write_authority(settings, active=True)
    playbook = playbook_catalog()[0]
    live = settings.paths.output_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    (live / "event_driven_execution_state.json").write_text(
        json.dumps(
            {
                "status": "MANAGING",
                "positions": {
                    "opportunity-1": {
                        "opportunity_id": "opportunity-1",
                        "market": "BTC-EUR",
                        "playbook_id": playbook["playbook_id"],
                        "playbook_dna": playbook["playbook_dna"],
                        "quantity": "0.06",
                        "entry_price": "100",
                        "stop_loss": "99",
                        "take_profit_1": "103",
                        "take_profit_2": "106",
                        "opened_at": "2026-08-03T12:00:00+00:00",
                        "time_stop_minutes": 30,
                    }
                },
                "orders_generated": 0,
                "orders_submitted": 0,
                "fills_verified": 0,
            }
        ),
        encoding="utf-8",
    )

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class Client:
        async def balances(self) -> list[dict[str, str]]:
            return [
                {"symbol": "EUR", "available": "100"},
                {"symbol": "BTC", "available": "0.06"},
            ]

        async def reconcile(self, *, markets: tuple[str, ...]) -> Any:
            return SimpleNamespace(healthy=True, reason_codes=("RECONCILED",))

        async def execution_market_rules(self, _: str) -> ExecutionMarketRules:
            return ExecutionMarketRules(
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
                tick_size=Decimal("0.01"),
            )

        async def submit_order(self, intent: Any, **_: object) -> dict[str, str]:
            assert intent.side.value == "SELL"
            return {
                "orderId": "exit-1",
                "status": "canceled",
                "filledAmount": "0.02",
                "filledAmountQuote": "1.96",
            }

    monkeypatch.setattr("core.event_driven_live.aiohttp.ClientSession", Session)
    monkeypatch.setattr(
        "core.event_driven_live.build_live_client",
        lambda *_, **__: Client(),
    )
    monkeypatch.setattr(
        "core.event_driven_live._live_capability",
        lambda *_, **__: SimpleNamespace(
            passed=True,
            capability=SimpleNamespace(token="x" * 32),
            failures=(),
        ),
    )
    realtime = _realtime()
    realtime["markets"][0]["price"] = 98.0
    realtime["markets"][0]["book"] = {
        "best_bid": 97.99,
        "best_ask": 98.01,
    }

    result = await execute_event_driven_live_once(
        settings,
        opportunities=[],
        realtime_snapshot=realtime,
        submit=True,
        allow_new_entry=True,
        allowed_economics_entry_families={"MOMENTUM"},
        observed_at=datetime(2026, 8, 3, 12, 1, tzinfo=UTC),
    )

    assert result["status"] == "POSITION_REDUCED"
    assert result["positions"]["opportunity-1"]["quantity"] == "0.04"
    assert result["fills_verified"] == 1
    assert any(
        event["event"] == "LIVE_POSITION_REDUCED"
        for event in result["events"]
    )
