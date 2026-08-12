from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from config.settings import PathSettings, Settings
from core import generated_strategy_live as generated_live
from core.contracts import ExecutionBlocked, OrderSide, OrderType, ResearchStatus
from execution.execution import ExecutionMarketRules, LivePreflight


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(
        env_file=tmp_path / "does-not-exist.env",
        create_directories=False,
    )
    return settings.model_copy(
        update={"paths": PathSettings(project_root=tmp_path)}
    )


def _candidate(
    *,
    dna: str = "a" * 64,
    frozen: str = "b" * 64,
    timeframe: str = "4h",
) -> dict:
    return {
        "strategy_id": f"TEST_{dna[:8]}",
        "strategy_dna_hash": dna,
        "frozen_candidate_hash": frozen,
        "timeframe": timeframe,
        "markets": ["BTC-EUR", "ETH-EUR"],
        "source": "CONTINUOUS_SIMPLE_LAB_EXACT",
        "metrics": {
            "net_return": 0.15,
            "profit_factor": 1.4,
            "net_expectancy_r": 0.05,
            "trade_count": 50,
        },
    }


def test_dynamic_notional_uses_equity_strength_and_hard_cap() -> None:
    normal, normal_meta = generated_live._dynamic_entry_notional(
        account_equity_eur=Decimal("1000"),
        available_eur=Decimal("500"),
        selected_entry={
            "score": 40,
            "effective_size_multiplier": 1,
        },
        authority_cap_eur=Decimal("25"),
    )
    exceptional, exceptional_meta = generated_live._dynamic_entry_notional(
        account_equity_eur=Decimal("10000"),
        available_eur=Decimal("500"),
        selected_entry={
            "score": 70,
            "effective_size_multiplier": 1,
        },
        authority_cap_eur=Decimal("100"),
    )

    assert normal == Decimal("10")
    assert normal_meta["setup_tier"] == "NORMAL"
    assert exceptional == Decimal("100")
    assert exceptional_meta["setup_tier"] == "EXCEPTIONALLY_STRONG"


def test_live_risk_distance_uses_protective_target_for_trailing_exit() -> None:
    stop, target, source = generated_live._resolve_live_risk_distances(
        {"stop_distance": "2.0", "target_distance": None},
    )

    assert stop == Decimal("2.0")
    assert target == Decimal("3.00")
    assert source == "PROTECTIVE_1_5R_TARGET_FALLBACK"


def test_live_risk_distance_never_invents_missing_stop() -> None:
    stop, target, source = generated_live._resolve_live_risk_distances(
        {"stop_distance": None, "target_distance": "3.0"},
    )

    assert stop == Decimal("0")
    assert target == Decimal("0")
    assert source == "INVALID_STOP_DISTANCE"


def test_gtc_entry_expiration_is_bounded() -> None:
    observed = datetime(2026, 8, 2, 12, 2, tzinfo=UTC)
    position = {
        "time_in_force": "GTC",
        "entry_order_submitted_at": "2026-08-02T12:00:00+00:00",
    }

    assert generated_live._gtc_entry_expired(
        position,
        observed_at=observed,
        validity_seconds=90,
    )
    assert not generated_live._gtc_entry_expired(
        position,
        observed_at=observed,
        validity_seconds=180,
    )


def test_gtc_reprice_preserves_original_entry_notional() -> None:
    assert generated_live._replacement_entry_notional(
        {
            "requested_quantity": "2",
            "limit_price": "5",
            "quantity": "4",
            "entry_price": "5",
        }
    ) == Decimal("10")
    assert generated_live._replacement_entry_notional(
        {
            "requested_quantity": "10",
            "limit_price": "5",
        }
    ) == generated_live.MAXIMUM_ORDER_EUR


def test_generated_protective_stop_is_deterministic_and_native() -> None:
    position = {
        "market": "ETH-EUR",
        "strategy_id": "TEST_STRATEGY",
        "strategy_dna_hash": "a" * 64,
        "signal_id": "signal-1",
    }
    first = generated_live._generated_protective_stop_intent(
        position,
        quantity=Decimal("0.01"),
        trigger_price=Decimal("1600"),
    )
    second = generated_live._generated_protective_stop_intent(
        position,
        quantity=Decimal("0.01"),
        trigger_price=Decimal("1600"),
    )

    assert first.idempotency_key == second.idempotency_key
    assert first.side is OrderSide.SELL
    assert first.order_type is OrderType.STOP_LOSS
    assert first.trigger_price == Decimal("1600")
    assert first.trigger_reference == "bestBid"


def test_native_stop_reconciliation_precedes_zero_inventory_skip() -> None:
    source = Path(generated_live.__file__).read_text(encoding="utf-8")
    loop_start = source.index("# Every confirmed generated-strategy fill")
    loop_end = source.index("preflight = LivePreflight.evaluate", loop_start)
    reconciliation_loop = source[loop_start:loop_end]

    query = reconciliation_loop.index("protective = await client.get_order")
    zero_inventory_skip = reconciliation_loop.rindex(
        "if owned <= 0 or expected <= 0"
    )
    assert query < zero_inventory_skip
    assert "PROTECTIVE_STOP_OPEN_BUT_MANAGED_INVENTORY_MISSING" in reconciliation_loop


def test_inactive_entry_authority_does_not_precede_exit_reconciliation() -> None:
    source = Path(generated_live.__file__).read_text(encoding="utf-8")
    function_start = source.index("async def execute_generated_strategy_live_once")
    function_source = source[function_start:]

    reconciliation = function_source.index("reconciliation = await client.reconcile")
    entry_authority_block = function_source.index(
        '"POSITIVE_STRATEGY_AUTHORITY_BLOCKED"',
        reconciliation,
    )
    economics_entry_block = function_source.index(
        '"CANONICAL_ECONOMICS_LIVE_VALIDATION_MISSING"',
        reconciliation,
    )
    assert reconciliation < entry_authority_block
    assert reconciliation < economics_entry_block


def test_unknown_external_inventory_does_not_precede_managed_protection() -> None:
    source = Path(generated_live.__file__).read_text(encoding="utf-8")
    function_start = source.index("async def execute_generated_strategy_live_once")
    function_source = source[function_start:]

    protection = function_source.index(
        "# Every confirmed generated-strategy fill"
    )
    exit_management = function_source.index(
        "# Exit management has priority", protection
    )
    entry_block = function_source.index(
        'else "UNKNOWN_GENERATED_STRATEGY_INVENTORY"',
        exit_management,
    )
    assert protection < exit_management < entry_block
    unknown_detection = function_source.index("if unknown_excess:")
    unknown_section = function_source[unknown_detection:protection]
    assert "return state" not in unknown_section


def test_inactive_authority_keeps_hashed_baseline_available_for_exits(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    baseline = {
        "schema_version": generated_live.BASELINE_SCHEMA_VERSION,
        "created_at": "2026-08-11T00:00:00Z",
        "source": "TEST",
        "authority_hash": "original-active-authority",
        "quantities": {"BTC": "0", "LINK": "0"},
        "orders_generated": 0,
        "orders_submitted": 0,
    }
    baseline["baseline_hash"] = generated_live._baseline_hash(baseline)
    path = generated_live._paths(settings)["baseline"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline), encoding="utf-8")

    quantities, inactive_failures = generated_live._load_baseline(
        settings,
        {"active": False, "authority_hash": "inactive-authority"},
    )
    _, active_failures = generated_live._load_baseline(
        settings,
        {"active": True, "authority_hash": "different-active-authority"},
    )

    assert quantities["BTC"] == Decimal("0")
    assert inactive_failures == []
    assert active_failures == ["POSITIVE_STRATEGY_BASELINE_AUTHORITY_MISMATCH"]


def test_positive_portfolio_authority_requires_operator_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: [_candidate()],
    )

    with pytest.raises(PermissionError):
        generated_live.activate_positive_strategy_live_authority(
            settings,
            approval_phrase="wrong",
        )

    result = generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )

    assert result["active"] is True
    assert result["approved_candidate_count"] == 1
    assert result["maximum_order_eur"] == "25"
    assert result["maximum_total_exposure_eur"] == "75"
    assert result["maximum_open_positions"] == 3
    serialized = generated_live._paths(settings)["authority"].read_text(
        encoding="utf-8"
    )
    assert generated_live.APPROVAL_PHRASE not in serialized


def test_positive_portfolio_keeps_unknown_future_dna_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    current = [_candidate()]
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: list(current),
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    current.append(_candidate(dna="c" * 64, frozen="d" * 64))

    active, authority, failures = (
        generated_live.synchronize_positive_strategy_live_authority(settings)
    )

    assert active is True
    assert failures == []
    assert len(authority["approved_candidates"]) == 1
    assert authority["auto_enroll_future_exact_positive_dna"] is False

    current[0] = _candidate(frozen="e" * 64)
    active, _, failures = (
        generated_live.synchronize_positive_strategy_live_authority(settings)
    )
    assert active is False
    assert any("IDENTITY_DRIFT" in reason for reason in failures)


def test_order_cap_migration_can_only_preserve_or_reduce_frozen_dna(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    first = _candidate()
    second = _candidate(dna="c" * 64, frozen="d" * 64)
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: [first, second],
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )

    result = generated_live.migrate_positive_strategy_live_order_cap(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
        maximum_order_eur=Decimal("25"),
        preserve_strategy_dna=[first["strategy_dna_hash"]],
    )

    assert result["status"] == "MIGRATED"
    assert result["approved_candidate_count"] == 1
    assert result["previous_approved_candidate_count"] == 2
    authority = json.loads(
        generated_live._paths(settings)["authority"].read_text(
            encoding="utf-8"
        )
    )
    assert [
        row["strategy_dna_hash"]
        for row in authority["approved_candidates"]
    ] == [first["strategy_dna_hash"]]
    assert authority["maximum_order_eur"] == "25"

    with pytest.raises(PermissionError, match="cannot add unknown"):
        generated_live.migrate_positive_strategy_live_order_cap(
            settings,
            approval_phrase=generated_live.APPROVAL_PHRASE,
            maximum_order_eur=Decimal("25"),
            preserve_strategy_dna=["e" * 64],
        )


def test_level_2_migration_recovers_only_cap_mismatch_deactivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: [_candidate()],
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    path = generated_live._paths(settings)["authority"]
    authority = json.loads(path.read_text(encoding="utf-8"))
    authority.update(
        {
            "active": False,
            "deactivated_at": "2026-08-08T00:00:00Z",
            "deactivation_reason": "POSITIVE_STRATEGY_AUTHORITY_BLOCKED",
        }
    )
    authority["authority_hash"] = generated_live.stable_hash(
        {key: value for key, value in authority.items() if key != "authority_hash"},
        length=64,
    )
    path.write_text(json.dumps(authority), encoding="utf-8")

    result = generated_live.migrate_positive_strategy_live_capital_level_2(
        settings,
        approval_phrase="I APPROVE LIVE CAPITAL LEVEL 2",
    )
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert result["status"] == "CAPITAL_LEVEL_2_ACTIVE"
    assert stored["active"] is True
    assert stored["deactivation_reason"] is None
    assert stored["maximum_order_eur"] == "25"


def test_live_authority_migrates_only_semantic_v2_evidence_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    current = [_candidate()]
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: list(current),
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    current[0] = {
        **current[0],
        "frozen_candidate_hash": "e" * 64,
        "frozen_identity_schema": "EXECUTION_SEMANTICS_V2",
    }

    active, authority, failures = (
        generated_live.synchronize_positive_strategy_live_authority(settings)
    )

    assert active is True
    assert failures == []
    assert authority["approved_candidates"][0][
        "frozen_candidate_hash"
    ] == "e" * 64
    assert authority["last_identity_schema_migrated_dna"] == ["a" * 64]


def test_live_authority_semantic_v2_does_not_hide_market_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    current = [_candidate()]
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: list(current),
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    current[0] = {
        **current[0],
        "frozen_candidate_hash": "e" * 64,
        "frozen_identity_schema": "EXECUTION_SEMANTICS_V2",
        "markets": ["ETH-EUR"],
    }

    active, _, failures = (
        generated_live.synchronize_positive_strategy_live_authority(settings)
    )

    assert active is False
    assert failures == [f"APPROVED_DNA_IDENTITY_DRIFT:{'a' * 64}"]


def test_operator_can_approve_exactly_one_new_frozen_dna(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    first = _candidate()
    second = _candidate(dna="c" * 64, frozen="d" * 64, timeframe="1h")
    third = _candidate(dna="e" * 64, frozen="f" * 64, timeframe="2h")
    current = [first]
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: list(current),
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    current.extend([second, third])

    with pytest.raises(PermissionError, match="phrase mismatch"):
        generated_live.approve_positive_strategy_dna(
            settings,
            strategy_id=second["strategy_id"],
            approval_phrase="wrong",
        )

    result = generated_live.approve_positive_strategy_dna(
        settings,
        strategy_id=second["strategy_id"],
        approval_phrase=(
            generated_live.positive_strategy_dna_approval_phrase(
                second["strategy_id"]
            )
        ),
    )

    assert result["status"] == "APPROVED"
    assert result["approved_candidate_count"] == 2
    authority = json.loads(
        generated_live._paths(settings)["authority"].read_text(
            encoding="utf-8"
        )
    )
    approved = {
        row["strategy_dna_hash"]
        for row in authority["approved_candidates"]
    }
    assert approved == {
        first["strategy_dna_hash"],
        second["strategy_dna_hash"],
    }
    assert third["strategy_dna_hash"] not in approved
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0
    serialized = json.dumps(authority)
    assert "LIVE POSITIVE DNA" not in serialized


def test_positive_portfolio_reactivation_rejects_tampered_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        generated_live,
        "load_generated_candidates",
        lambda _settings: [_candidate()],
    )
    generated_live.activate_positive_strategy_live_authority(
        settings,
        approval_phrase=generated_live.APPROVAL_PHRASE,
    )
    baseline = generated_live._paths(settings)["baseline"]
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        '{"schema_version":"generated_strategy_inventory_baseline_v1",'
        '"quantities":{"BTC":"99"},"baseline_hash":"tampered"}',
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="baseline is invalid"):
        generated_live.activate_positive_strategy_live_authority(
            settings,
            approval_phrase=generated_live.APPROVAL_PHRASE,
        )


def test_next_open_window_and_ranked_natural_entry() -> None:
    candidate = _candidate()
    authority = {
        "approved_candidates": [
            {
                "strategy_id": candidate["strategy_id"],
                "strategy_dna_hash": candidate["strategy_dna_hash"],
                "frozen_candidate_hash": candidate[
                    "frozen_candidate_hash"
                ],
                "timeframe": "4h",
                "approved_markets": ["BTC-EUR", "ETH-EUR"],
            }
        ]
    }
    observed_at = datetime(2026, 7, 30, 8, 5, tzinfo=UTC)
    active, execute_at, reason = generated_live.signal_execution_window(
        signal_timestamp="2026-07-30T04:00:00+00:00",
        timeframe="4h",
        observed_at=observed_at,
    )
    assert active is True
    assert execute_at == "2026-07-30T08:00:00+00:00"
    assert reason == "NEXT_OPEN_WINDOW_ACTIVE"

    ranked = generated_live.rank_natural_entries(
        candidates=[candidate],
        evaluations={
            candidate["strategy_dna_hash"]: {
                "status": "EVALUATED",
                "markets": [
                    {
                        "market": "BTC-EUR",
                        "signal_timestamp": (
                            "2026-07-30T04:00:00+00:00"
                        ),
                        "entry": True,
                        "exit": False,
                        "stale": False,
                        "stop_distance": 1000,
                        "target_distance": 1500,
                        "size_multiplier": 1,
                    }
                ],
            }
        },
        authority=authority,
        observed_at=observed_at,
    )
    assert len(ranked) == 1
    assert ranked[0]["market"] == "BTC-EUR"
    assert ranked[0]["signal_id"]

    expired = generated_live.rank_natural_entries(
        candidates=[candidate],
        evaluations={
            candidate["strategy_dna_hash"]: {
                "status": "EVALUATED",
                "markets": [
                    {
                        "market": "BTC-EUR",
                        "signal_timestamp": (
                            "2026-07-30T04:00:00+00:00"
                        ),
                        "entry": True,
                        "stale": False,
                    }
                ],
            }
        },
        authority=authority,
        observed_at=datetime(2026, 7, 30, 8, 16, tzinfo=UTC),
    )
    assert expired == []


def test_rank_shrinks_small_sample_pf_and_respects_degradation() -> None:
    small = _candidate(dna="a" * 64)
    small["metrics"].update(
        {"profit_factor": 5.0, "trade_count": 10, "net_return": 0.15}
    )
    robust = _candidate(dna="c" * 64, frozen="d" * 64)
    robust["metrics"].update(
        {
            "profit_factor": 1.5,
            "stressed_profit_factor": 1.3,
            "trade_count": 200,
            "net_return": 0.15,
        }
    )
    authority = {
        "approved_candidates": [
            {
                "strategy_id": row["strategy_id"],
                "strategy_dna_hash": row["strategy_dna_hash"],
                "frozen_candidate_hash": row["frozen_candidate_hash"],
                "timeframe": "4h",
                "approved_markets": ["BTC-EUR"],
            }
            for row in (small, robust)
        ]
    }
    evaluations = {
        row["strategy_dna_hash"]: {
            "status": "EVALUATED",
            "markets": [
                {
                    "market": "BTC-EUR",
                    "signal_timestamp": "2026-07-30T04:00:00+00:00",
                    "entry": True,
                    "stale": False,
                }
            ],
        }
        for row in (small, robust)
    }

    ranked = generated_live.rank_natural_entries(
        candidates=[small, robust],
        evaluations=evaluations,
        authority=authority,
        observed_at=datetime(2026, 7, 30, 8, 5, tzinfo=UTC),
    )

    assert ranked[0]["strategy_dna_hash"] == robust["strategy_dna_hash"]
    assert ranked[1]["sample_weight"] < ranked[0]["sample_weight"]
    assert ranked[1]["adjusted_profit_factor"] < small["metrics"]["profit_factor"]

    degraded = generated_live.rank_natural_entries(
        candidates=[small, robust],
        evaluations=evaluations,
        authority=authority,
        degradation={
            robust["strategy_dna_hash"]: {
                "entry_allowed": False,
                "degradation_state": "PAPER_ACTIVE",
                "risk_multiplier": "0",
            }
        },
        observed_at=datetime(2026, 7, 30, 8, 5, tzinfo=UTC),
    )
    assert [row["strategy_dna_hash"] for row in degraded] == [
        small["strategy_dna_hash"]
    ]


def test_macro_overlay_selects_but_never_creates_live_entry() -> None:
    trend = _candidate(timeframe="1h")
    trend["economic_hypothesis_family"] = (
        "CAUSAL_MTF_DONCHIAN_ATR_FRACTAL"
    )
    authority = {
        "approved_candidates": [
            {
                "strategy_id": trend["strategy_id"],
                "strategy_dna_hash": trend["strategy_dna_hash"],
                "frozen_candidate_hash": trend["frozen_candidate_hash"],
                "timeframe": "1h",
                "approved_markets": ["BTC-EUR"],
            }
        ]
    }
    evaluations = {
        trend["strategy_dna_hash"]: {
            "status": "EVALUATED",
            "markets": [
                {
                    "market": "BTC-EUR",
                    "signal_timestamp": "2026-07-30T07:00:00+00:00",
                    "entry": True,
                    "stale": False,
                }
            ],
        }
    }
    observed_at = datetime(2026, 7, 30, 8, 5, tzinfo=UTC)

    supportive = generated_live.rank_natural_entries(
        candidates=[trend],
        evaluations=evaluations,
        authority=authority,
        macro_context={
            "status": "FRESH",
            "regime": "MODERATE_RISK_ON",
            "confidence": 1.0,
        },
        observed_at=observed_at,
    )
    risk_off = generated_live.rank_natural_entries(
        candidates=[trend],
        evaluations=evaluations,
        authority=authority,
        macro_context={
            "status": "FRESH",
            "regime": "MACRO_RISK_OFF",
            "confidence": 1.0,
        },
        observed_at=observed_at,
    )
    no_signal = generated_live.rank_natural_entries(
        candidates=[trend],
        evaluations={
            trend["strategy_dna_hash"]: {
                "status": "EVALUATED",
                "markets": [
                    {
                        "market": "BTC-EUR",
                        "signal_timestamp": (
                            "2026-07-30T07:00:00+00:00"
                        ),
                        "entry": False,
                        "stale": False,
                    }
                ],
            }
        },
        authority=authority,
        macro_context={
            "status": "FRESH",
            "regime": "STRONG_RISK_ON",
            "confidence": 1.0,
        },
        observed_at=observed_at,
    )

    assert len(supportive) == 1
    assert supportive[0]["macro_policy"] == "ENABLE"
    assert supportive[0]["macro_risk_multiplier"] == 1.0
    assert len(risk_off) == 1
    assert risk_off[0]["macro_policy"] == "REDUCE"
    assert risk_off[0]["macro_risk_multiplier"] == 0.65
    assert no_signal == []


def test_macro_risk_off_keeps_confirmed_eth_recovery_reduced() -> None:
    recovery = _candidate(timeframe="1h")
    recovery["economic_hypothesis_family"] = "LIQUIDITY_SWEEP_RECOVERY"
    authority = {
        "approved_candidates": [
            {
                "strategy_id": recovery["strategy_id"],
                "strategy_dna_hash": recovery["strategy_dna_hash"],
                "frozen_candidate_hash": recovery["frozen_candidate_hash"],
                "timeframe": "1h",
                "approved_markets": ["ETH-EUR"],
            }
        ]
    }
    ranked = generated_live.rank_natural_entries(
        candidates=[recovery],
        evaluations={
            recovery["strategy_dna_hash"]: {
                "status": "EVALUATED",
                "markets": [
                    {
                        "market": "ETH-EUR",
                        "signal_timestamp": (
                            "2026-07-30T07:00:00+00:00"
                        ),
                        "entry": True,
                        "stale": False,
                        "size_multiplier": 0.8,
                    }
                ],
            }
        },
        authority=authority,
        macro_context={
            "status": "FRESH",
            "regime": "MACRO_RISK_OFF",
            "confidence": 1.0,
        },
        observed_at=datetime(2026, 7, 30, 8, 5, tzinfo=UTC),
    )

    assert len(ranked) == 1
    assert ranked[0]["macro_policy"] == "REDUCE"
    assert ranked[0]["macro_risk_multiplier"] == 0.65
    assert ranked[0]["effective_size_multiplier"] == pytest.approx(0.52)


def test_live_macro_overlay_is_fail_closed_when_missing_or_stale(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    observed_at = datetime(2026, 7, 30, 8, 0, tzinfo=UTC)

    missing = generated_live._load_live_macro_overlay(
        settings,
        observed_at=observed_at,
    )
    assert missing["status"] == "DATA_BLOCKED"
    assert missing["reason_code"] == "MACRO_CONTEXT_MISSING"

    path = tmp_path / "output" / "active_trading" / "macro_crypto.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "available_at": "2026-07-30T04:00:00+00:00",
                "regime": "MODERATE_RISK_ON",
                "confidence": 1.0,
            }
        ),
        encoding="utf-8",
    )
    stale = generated_live._load_live_macro_overlay(
        settings,
        observed_at=observed_at,
    )
    assert stale["status"] == "DATA_BLOCKED"
    assert stale["reason_code"] == "MACRO_CONTEXT_STALE"

    path.write_text(
        json.dumps(
            {
                "observed_at": "2026-07-30T07:55:00+00:00",
                "available_at": "2026-07-30T07:55:00+00:00",
                "regime": "MODERATE_RISK_ON",
                "confidence": 0.9,
                "features": {"btc_1d_trend_up": True},
                "sources": {
                    "coinmarketcap_global": {
                        "provider": "coinmarketcap",
                        "fresh": True,
                        "freshness": "FRESH",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    fresh = generated_live._load_live_macro_overlay(
        settings,
        observed_at=observed_at,
    )
    assert fresh["status"] == "FRESH"
    assert fresh["regime"] == "MODERATE_RISK_ON"
    assert fresh["macro_is_entry_signal"] is False


def test_wallet_wide_position_count_ignores_dust() -> None:
    count, markets = generated_live._material_wallet_positions(
        {
            "account": {
                "portfolio_valuation": {
                    "holdings": [
                        {
                            "market": "NPC-EUR",
                            "estimated_value_eur": "500",
                        },
                        {
                            "market": "TAO-EUR",
                            "estimated_value_eur": "505",
                        },
                        {
                            "market": "ETH-EUR",
                            "estimated_value_eur": "0.001",
                        },
                    ]
                }
            }
        }
    )
    assert count == 2
    assert markets == ["NPC-EUR", "TAO-EUR"]


def test_grandfathered_inventory_does_not_consume_managed_level_two_slot(
    tmp_path: Path,
) -> None:
    status = generated_live.write_position_limit_status(
        _settings(tmp_path),
        account_equity_eur=Decimal("369"),
        material_positions=["ETH-EUR", "TAO-EUR"],
        managed_positions=["ETH-EUR"],
        maximum_managed_positions=3,
    )

    assert status["wallet_material_position_count"] == 2
    assert status["managed_position_count"] == 1
    assert status["maximum_managed_positions"] == 3
    assert status["remaining_slots"] == 2
    assert status["new_position_allowed"] is True
    assert status["grandfathered_inventory_consumes_managed_slot"] is False


def test_level_one_portfolio_canary_allows_three_tiny_positions() -> None:
    settings = Settings.load()
    passed = LivePreflight.evaluate(
        settings,
        markets=("BTC-EUR", "ETH-EUR"),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
        portfolio_canary=True,
        cap_limits={
            "capital_level": 1,
            "max_order_eur": 10,
            "max_exposure_eur": 15,
            "max_positions": 3,
            "max_new_orders_per_day": 3,
        },
    )
    assert passed.passed is True
    assert passed.capability is not None
    assert passed.capability.maximum_order_eur == Decimal("10")
    assert passed.capability.maximum_total_eur == Decimal("15")
    assert passed.capability.maximum_open_positions == 3

    blocked = LivePreflight.evaluate(
        settings,
        markets=("BTC-EUR",),
        strategy_status=ResearchStatus.LIVE_BLOCKED,
        data_healthy=True,
        risk_manager_healthy=True,
        exchange_healthy=True,
        reconciliation_healthy=True,
        kill_switch_active=False,
        canary_exception_approved=True,
        operator_canary_authorized=True,
        portfolio_canary=True,
        cap_limits={
            "capital_level": 1,
            "max_order_eur": 11,
            "max_exposure_eur": 15,
            "max_positions": 3,
            "max_new_orders_per_day": 3,
        },
    )
    assert blocked.passed is False
    assert "LIVE_BLOCKED_INVALID_CAPITAL_AUTHORITY" in blocked.failures


@pytest.mark.asyncio
async def test_generated_live_entry_planner_blocks_when_bounded_limit_is_infeasible(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    class Client:
        async def execution_market_rules(self, market: str):
            assert market == "ETH-EUR"
            return ExecutionMarketRules(
                minimum_order_amount=Decimal("0.00304033"),
                minimum_order_value_eur=Decimal("5"),
                quantity_decimals=8,
                notional_decimals=2,
                tick_size=Decimal("0.01"),
            )

    with pytest.raises(
        ExecutionBlocked,
        match="venue rounding makes bounded limit infeasible",
    ):
        await generated_live._plan_live_entry_order(
            settings,
            client=Client(),
            market="ETH-EUR",
            requested_notional_eur=Decimal("5"),
            public_price=Decimal("1667.50"),
            liquidity={
                "best_ask": "1667.60",
                "estimated_average_price": "1667.60",
                "limits": {"maximum_slippage_bps": "25"},
            },
        )
