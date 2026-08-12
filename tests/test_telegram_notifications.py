from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from config.settings import PathSettings, TelegramSettings
from core import cli
from notifications.telegram import (
    TelegramHttpResponse,
    TelegramNotifier,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
TOKEN = "unit-test-telegram-token"
CHAT = "unit-test-chat-id"


class FakeTransport:
    def __init__(self, *responses: TelegramHttpResponse | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        form: dict[str, str] | None,
        timeout: float,
    ) -> TelegramHttpResponse:
        self.calls.append(
            {
                "method": method,
                "form": form,
                "timeout": timeout,
            }
        )
        response = (
            self.responses.pop(0)
            if self.responses
            else TelegramHttpResponse(200, {"ok": True})
        )
        if isinstance(response, BaseException):
            raise response
        return response


def settings(**overrides: Any) -> TelegramSettings:
    values: dict[str, Any] = {
        "notifications_enabled": True,
        "bot_token": TOKEN,
        "chat_id": CHAT,
        "max_retries": 2,
        "request_timeout_seconds": 1,
    }
    values.update(overrides)
    return TelegramSettings(**values)


def notifier(
    tmp_path: Path,
    *,
    transport: FakeTransport | None = None,
    telegram_settings: TelegramSettings | None = None,
    sleeper=lambda _: None,
) -> TelegramNotifier:
    return TelegramNotifier(
        telegram_settings or settings(),
        output_directory=tmp_path / "notifications",
        allowed_markets=("BTC-EUR", "ETH-EUR"),
        transport=transport or FakeTransport(),
        sleeper=sleeper,
        clock=lambda: NOW,
    )


def buy_signal(**overrides: Any) -> dict[str, Any]:
    payload = {
        "signal_id": "signal-1",
        "status": "MANUAL_ACTIONABLE",
        "signal_authority": "MANUAL_ACTIONABLE",
        "action": "BUY",
        "market": "BTC-EUR",
        "strategy_dna_hash": "frozen-dna",
        "strategy_frozen": True,
        "entry_low": 58_200,
        "entry_high": 58_500,
        "preferred_entry": 58_350,
        "stop_loss": 56_900,
        "take_profit_1": 61_000,
        "take_profit_2": 62_400,
        "maximum_planned_loss_eur": 2,
        "suggested_order_value_eur": 25,
        "strategy_name": "Trend Breakout",
        "timeframe": "4h",
        "confidence": 74,
        "expires_at": NOW + timedelta(hours=6),
        "reason": "Trend omhoog en volume bovengemiddeld.",
        "data_stale": False,
    }
    payload.update(overrides)
    return payload


def near_entry(**overrides: Any) -> dict[str, Any]:
    payload = {
        "status": "NEAR_ENTRY",
        "market": "ETH-EUR",
        "strategy": "TACTICAL_1H_LIQUIDITY_SWEEP",
        "timeframe": "1h",
        "trigger": 1_638.33,
        "stop": 1_627.49,
        "target_1": 1_654.59,
        "target_2": 1_663.63,
        "confidence": 70,
        "reason_not_yet_entered": "ENTRY_TRIGGER_NOT_CONFIRMED",
        "live_authority_granted": False,
    }
    payload.update(overrides)
    return payload


def early_move(**overrides: Any) -> dict[str, Any]:
    payload = {
        "status": "EARLY_MOMENTUM_ALERT",
        "market": "ETH-EUR",
        "strategy": "EARLY_MOVE_VOLUME_FLOW_15M",
        "timeframe": "15m",
        "trigger": 1_700.0,
        "stop": 1_675.0,
        "target_1": 1_737.5,
        "target_2": 1_762.5,
        "confidence": 78,
        "reason_not_yet_entered": (
            "EARLY_ALERT_REQUIRES_CLOSED_CANDLE_ENTRY_CONFIRMATION"
        ),
        "live_authority_granted": False,
        "formula": {
            "return_15m": 0.012,
            "return_1h": 0.025,
            "relative_volume_20": 2.4,
            "volume_robust_zscore": 3.1,
            "extension_atr": 1.2,
        },
    }
    payload.update(overrides)
    return payload


def test_near_entry_update_is_compact_deduplicated_and_orderless(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)

    first = selected.notify_opportunity_update([near_entry()])
    duplicate = selected.notify_opportunity_update(
        [near_entry(trigger=1_639.00, confidence=73)]
    )
    changed = selected.notify_opportunity_update(
        [near_entry(stop=1_620.00)]
    )
    actionable = selected.notify_opportunity_update(
        [near_entry(status="ACTIONABLE", stop=1_620.00)]
    )

    assert first["orders_submitted"] == 0
    assert duplicate["delivery_status"] == "SKIPPED_DUPLICATE"
    assert duplicate["reason_code"] == "OPPORTUNITY_NOT_MATERIALLY_CHANGED"
    assert changed["orders_generated"] == 0
    assert changed["reason_code"] == "TACTICAL_UPDATE_COOLDOWN_ACTIVE"
    assert actionable["delivery_status"] == "PENDING"
    assert len(transport.calls) == 2
    text = transport.calls[0]["form"]["text"]
    assert "NOG GEEN ORDER" in text
    assert "aparte DNA-goedkeuring vereist" in text
    assert TOKEN not in text
    assert CHAT not in text
    evidence = [
        json.loads(line)
        for line in selected.opportunity_evidence_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(evidence) == 2
    assert evidence[0]["previous_hash"] == "GENESIS"
    assert evidence[1]["previous_hash"] == evidence[0]["record_hash"]
    assert evidence[0]["rows"][0]["trigger"] == 1_638.33
    assert evidence[0]["rows"][0]["entry_condition"] == (
        "HIGH_AT_OR_ABOVE_TRIGGER"
    )
    assert evidence[0]["delivery_status_at_capture"] == "PENDING"


def test_early_move_alert_exposes_volume_formula_but_stays_orderless(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)

    result = selected.notify_opportunity_update([early_move()])

    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0
    assert len(transport.calls) == 1
    message = transport.calls[0]["form"]["text"]
    assert "VROEGE BEWEGING GEDETECTEERD" in message
    assert "15m / 1h: 1,20% / 2,50%" in message
    assert "RVOL20 / robuuste z: 2,40x / 3,10" in message
    assert "aparte DNA-goedkeuring vereist" in message
    assert "plaatst nul orders" in message


@pytest.mark.parametrize(
    ("token", "chat_id", "expected_missing"),
    [
        (None, CHAT, "TELEGRAM_BOT_TOKEN"),
        (TOKEN, None, "TELEGRAM_CHAT_ID"),
    ],
)
def test_missing_configuration_is_disabled_without_network(
    tmp_path: Path,
    token: str | None,
    chat_id: str | None,
    expected_missing: str,
) -> None:
    transport = FakeTransport()
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=TelegramSettings(
            notifications_enabled=True,
            bot_token=token,
            chat_id=chat_id,
        ),
    )
    health = selected.health()
    assert health["status"] == "DISABLED_MISSING_CONFIG"
    assert expected_missing in health["missing_configuration"]
    assert not transport.calls


def test_disabled_notifications_do_not_probe_or_send(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=settings(notifications_enabled=False),
    )
    assert selected.health()["status"] == "DISABLED"
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["sent"] == 0
    assert not transport.calls


def test_dry_run_persists_sent_without_http(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=settings(dry_run=True),
    )
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["sent"] == 1
    assert not transport.calls
    assert selected.status()["sent_count"] == 1


def test_successful_test_message_is_compact_and_orderless(tmp_path: Path) -> None:
    transport = FakeTransport(TelegramHttpResponse(200, {"ok": True}))
    selected = notifier(tmp_path, transport=transport)
    result = selected.send_test_message()
    assert result["status"] == "SENT"
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0
    text = transport.calls[0]["form"]["text"]
    assert "SIGNALS_ONLY" in text
    assert "Orders geplaatst: 0" in text


def test_health_and_artifacts_are_secret_safe(tmp_path: Path) -> None:
    selected = notifier(
        tmp_path,
        transport=FakeTransport(TelegramHttpResponse(200, {"ok": True})),
    )
    assert selected.health()["status"] == "HEALTHY"
    selected.notify_system_event("SERVICE_START", {"reason": TOKEN, "market": "BTC-EUR"})
    rendered = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "notifications").iterdir()
        if path.is_file()
    )
    assert TOKEN not in rendered
    assert CHAT not in rendered
    assert "***REDACTED***" in rendered


def test_duplicate_signal_is_sent_once(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    first = selected.process_signals([buy_signal()])
    second = selected.process_signals([buy_signal()])
    assert first["delivery"]["sent"] == 1
    assert second["duplicates"] == 1
    assert len(transport.calls) == 1
    assert selected.status()["duplicates_skipped"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"stop_loss": 57_100},
        {"take_profit_1": 61_500},
    ],
)
def test_changed_stop_or_target_creates_new_notification(
    tmp_path: Path,
    change: dict[str, Any],
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.process_signals([buy_signal()])
    selected.process_signals([buy_signal(**change)])
    assert len(transport.calls) == 2
    assert selected.status()["sent_count"] == 2


def test_changed_expiration_creates_new_notification(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.process_signals([buy_signal()])
    selected.process_signals(
        [buy_signal(expires_at=NOW + timedelta(hours=12))]
    )
    assert len(transport.calls) == 2
    assert selected.status()["sent_count"] == 2


def test_expired_signal_is_filtered(tmp_path: Path) -> None:
    selected = notifier(tmp_path)
    result = selected.process_signals(
        [buy_signal(expires_at=NOW - timedelta(seconds=1))]
    )
    assert result["filtered"] == 1
    assert result["delivery"]["sent"] == 0


def test_expired_lifecycle_event_is_sent_after_expiration(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    result = selected.process_signals(
        [
            buy_signal(
                status="EXPIRED",
                lifecycle_status="EXPIRED",
                expires_at=NOW - timedelta(seconds=1),
            )
        ]
    )
    assert result["delivery"]["sent"] == 1
    assert "🔴 EXIT" in transport.calls[0]["form"]["text"]


def test_watchlist_toggle_and_confidence_filtering(tmp_path: Path) -> None:
    disabled = notifier(
        tmp_path / "disabled",
        telegram_settings=settings(send_watchlist=False),
    )
    watchlist = buy_signal(
        signal_id="watch",
        action="WATCHLIST",
        confidence=61,
        strategy_frozen=False,
    )
    assert disabled.process_signals([watchlist])["filtered"] == 1
    low = notifier(tmp_path / "low")
    assert low.process_signals([{**watchlist, "confidence": 59}])["filtered"] == 1
    passing = notifier(tmp_path / "passing")
    assert passing.process_signals([watchlist])["delivery"]["sent"] == 1


def test_reward_risk_filtering(tmp_path: Path) -> None:
    selected = notifier(tmp_path)
    result = selected.process_signals(
        [
            buy_signal(
                preferred_entry=100,
                stop_loss=99,
                take_profit_1=101,
                take_profit_2=102,
            )
        ]
    )
    assert result["filtered"] == 1


@pytest.mark.parametrize(
    ("lifecycle", "expected"),
    [
        ("INVALIDATED", "🔴 EXIT"),
        ("STOPPED_OUT", "🔴 EXIT"),
        ("TP1_REACHED", "✅ TP1 BEREIKT"),
        ("TP2_REACHED", "✅ TP2 BEREIKT"),
        ("EXPIRED", "🔴 EXIT"),
        ("CLOSED", "🔴 EXIT"),
    ],
)
def test_signal_lifecycle_messages(
    tmp_path: Path,
    lifecycle: str,
    expected: str,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.process_signals(
        [buy_signal(status=lifecycle, lifecycle_status=lifecycle)]
    )
    assert expected in transport.calls[0]["form"]["text"]


def test_exit_alert_contains_signal_identity(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.process_signals(
        [
            buy_signal(
                action="EXIT",
                status="MANUAL_ACTIONABLE",
                current_price=60_000,
                original_entry=58_350,
                result_pct=0.028,
            )
        ]
    )
    text = transport.calls[0]["form"]["text"]
    assert "Signal ID: signal-1" in text
    assert "Oorspronkelijke entry" in text


def test_kill_switch_alert_is_critical(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.notify_system_event(
        "KILL_SWITCH_ACTIVATED",
        {"status": "ACTIVE", "reason_code": "MAX_DRAWDOWN"},
    )
    assert transport.calls[0]["form"]["text"].startswith("🚨")


def test_stop_loss_alert_is_warning(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.notify_system_event(
        "STOP_LOSS_REACHED",
        {"market": "BTC-EUR", "status": "CLOSED"},
    )
    assert transport.calls[0]["form"]["text"].startswith("⚠️")


def test_paper_promotion_summary_is_compact_deduplicated_and_orderless(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    candidates = [
        {
            "strategy_dna_hash": "a" * 64,
            "economic_hypothesis_family": "BREAKOUT+VOLUME",
            "timeframe": "4h",
            "metrics": {"profit_factor": 1.41},
        },
        {
            "strategy_dna_hash": "b" * 64,
            "economic_hypothesis_family": "TREND",
            "timeframe": "1h",
            "metrics": {"profit_factor": 1.2},
        },
    ]

    first = selected.notify_paper_promotion_summary(candidates)
    second = selected.notify_paper_promotion_summary(candidates)

    assert first["delivery_status"] == "PENDING"
    assert second["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1
    text = transport.calls[0]["form"]["text"]
    assert "NIEUWE PAPERSTRATEGIEËN" in text
    assert "4h: 1" in text
    assert "1h: 1" in text
    assert "Automatische livepromoties: 0" in text
    assert "Echte orders geplaatst: 0" in text


def test_strategy_performance_notification_is_deduplicated(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    payload = {
        "strategy_id": "RR_B60_H5_Z20",
        "strategy_dna": "frozen-dna",
        "authority_level": "LIVE_CANARY",
        "closed_trade_count": 1,
        "strategy_equity_eur": "10.80",
        "maximum_drawdown_eur": "0",
        "last_closed_trade": {
            "market": "ETH-EUR",
            "closed_at": "2026-07-30T12:00:00Z",
            "net_pnl_eur": "0.80",
            "fees_eur": "0.02",
            "average_slippage_bps": "1.5",
            "holding_seconds": 3600,
        },
    }

    first = selected.notify_strategy_performance(payload)
    second = selected.notify_strategy_performance(payload)

    assert first["delivery_status"] == "PENDING"
    assert second["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1
    text = transport.calls[0]["form"]["text"]
    assert "STRATEGY TRADE CLOSED — ETH-EUR" in text
    assert "Netto P&L: €0,80" in text
    assert "Holding time: 1u 0m" in text
    assert "Authority: LIVE_CANARY" in text


def test_daily_performance_notification_is_once_per_date(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    payload = {
        "date_utc": "2026-07-30",
        "account_identity_hash": "account-hash",
        "wallet_value_eur": "1045.84",
        "daily_pnl_eur": "22.88",
        "realised_pnl_eur": "0",
        "unrealised_pnl_eur": "0",
        "best_strategy": "RR_B60_H5_Z20",
        "worst_strategy": "RR_B60_H5_Z20",
        "fees_eur": "0",
        "maximum_drawdown_eur": "0",
        "active_capital_eur": "0",
        "cash_reserve_eur": "30.04",
        "authority_status": "ENABLED/LIVE_CANARY",
        "open_positions": 0,
        "live_orders_today": 0,
    }

    first = selected.notify_daily_performance(payload)
    second = selected.notify_daily_performance(payload)

    assert first["delivery_status"] == "PENDING"
    assert second["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1
    text = transport.calls[0]["form"]["text"]
    assert "DAGELIJKS LIVE-OVERZICHT — 2026-07-30" in text
    assert "Walletwaarde: €1.045,84" in text
    assert "Echte orders vandaag: 0" in text


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("ORDER_SUBMITTING", "📤 LIVE ORDER SUBMITTED"),
        ("ORDER_PARTIALLY_FILLED", "⚠️ LIVE ORDER PARTIALLY FILLED"),
        ("ORDER_FILLED", "✅ LIVE ORDER FILLED"),
        ("ORDER_REJECTED", "🚨 LIVE ORDER REJECTED"),
    ],
)
def test_order_lifecycle_messages(
    tmp_path: Path,
    event: str,
    expected: str,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    selected.notify_order_event(
        event,
        {
            "order_id": "private-order-id",
            "market": "BTC-EUR",
            "side": "BUY",
            "order_type": "LIMIT",
            "price": 58_300,
            "quantity": 0.001,
            "notional_eur": 58.3,
            "strategy_id": "strategy",
            "reason_code": "VENUE_REJECTED",
            "verification_source": "BITVAVO_REST_ORDER_RESPONSE",
        },
    )
    text = transport.calls[0]["form"]["text"]
    assert expected in text
    assert "private-order-id" not in text
    if event.startswith("PAPER_"):
        assert "Execution: PAPER_ONLY — geen echte Bitvavo-order" in text
        assert "✅ ORDER FILLED" not in text
    else:
        assert "Execution: LIVE/EXCHANGE — exchangebevestiging vereist" in text
    if event.endswith("ORDER_REJECTED"):
        assert "Geen automatische retry uitgevoerd" in text


def test_paper_evidence_relabels_legacy_generic_fill(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)

    result = selected.notify_order_event(
        "ORDER_FILLED",
        {
            "order_id": "paper-order-id",
            "market": "ETH-EUR",
            "side": "BUY",
            "order_type": "MARKET",
            "price": 1_600,
            "quantity": 0.01,
            "paper_only": True,
            "execution_mode": "PAPER_ONLY",
        },
    )

    assert result["delivery_status"] == "SKIPPED_FILTER"
    assert result["reason_code"] == "INDIVIDUAL_PAPER_LIFECYCLE_SUMMARY_ONLY"
    assert transport.calls == []


def test_private_order_and_fill_replay_share_one_final_notification(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    common = {
        "order_id": "same-private-order-id",
        "market": "BTC-EUR",
        "side": "BUY",
        "verification_source": "BITVAVO_PRIVATE_ACCOUNT_STREAM",
    }

    first = selected.notify_order_event(
        "LIVE_ORDER_FILLED",
        {**common, "status": "filled", "filled_quantity": "0.00018"},
    )
    replay = selected.notify_order_event(
        "LIVE_ORDER_FILLED",
        {**common, "status": None, "filled_quantity": None},
    )

    assert first["delivery_status"] == "PENDING"
    assert replay["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1


def test_position_closed_message_is_exact_and_deduplicated(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    payload = {
        "signal_id": "signal-1",
        "market": "BTC-EUR",
        "entry_price": "55541",
        "exit_price": "55438",
        "sell_quantity": "0.00018004",
        "fees_eur": "0.05145588",
        "net_pnl_eur": "-0.07000000",
        "reason": "ORDERFLOW_EXHAUSTION",
        "strategy_id": "LIQUIDITY_SWEEP_RECLAIM_V1",
    }

    first = selected.notify_position_closed(payload)
    duplicate = selected.notify_position_closed(payload)

    assert first["delivery_status"] == "PENDING"
    assert duplicate["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1
    message = transport.calls[0]["form"]["text"]
    assert "POSITION CLOSED — BTC-EUR" in message
    assert "Netto PnL: −€0,07" in message
    assert "ORDERFLOW_EXHAUSTION" in message


def test_scheduled_macro_message_is_compact_and_deduplicated(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)
    payload = {
        "observed_at": "2026-08-05T05:59:00+00:00",
        "macro_regime": "RECOVERY",
        "structural_regime": "RISK_OFF",
        "btc_1d_trend": "BEARISH",
        "btc_4h_trend": "BULLISH",
        "btc_return_24h_pct": 2.1,
        "altcoin_breadth_pct": 58.0,
        "btc_dominance_pct": 54.4,
        "fear_greed": 47,
        "risk_multiplier": 0.4,
        "active_playbooks": "liquidity-sweep, failed-breakdown",
        "blocked_playbooks": "ongefilterde trendbreakouts",
        "entry_candidates": 2,
        "open_positions": 1,
        "open_orders": 1,
        "eur_available": 20.25,
        "live_status": "RUNNING",
    }

    first = selected.notify_macro_summary(payload, slot="2026-08-05 08:00")
    duplicate = selected.notify_macro_summary(
        payload,
        slot="2026-08-05 08:00",
    )

    assert first["delivery_status"] == "PENDING"
    assert duplicate["delivery_status"] == "SKIPPED_DUPLICATE"
    assert len(transport.calls) == 1
    message = transport.calls[0]["form"]["text"]
    assert "MACRO & MARKT — 2026-08-05 08:00 NL" in message
    assert "BTC 1D / 4H: BEARISH / BULLISH" in message
    assert "Verse entrykandidaten: 2" in message


def test_unverified_live_fill_is_never_claimed_as_filled(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(tmp_path, transport=transport)

    result = selected.notify_system_event(
        "ORDER_FILLED",
        {
            "market": "BTC-EUR",
            "side": "BUY",
            "order_type": "LIMIT",
            "order_id": "local-only-id",
            "filled_quantity": "0.001",
            "status": "filled",
        },
    )

    text = transport.calls[0]["form"]["text"]
    assert result["delivery_status"] == "PENDING"
    assert "LIVE ORDER STATUS UNVERIFIED" in text
    assert "LIVE ORDER FILLED" not in text
    assert "Geen fillclaim" in text


def test_timeout_retries_then_succeeds(tmp_path: Path) -> None:
    delays: list[float] = []
    transport = FakeTransport(
        TimeoutError(),
        TelegramHttpResponse(200, {"ok": True}),
    )
    selected = notifier(
        tmp_path,
        transport=transport,
        sleeper=delays.append,
    )
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["sent"] == 1
    assert len(transport.calls) == 2
    assert delays == [1]
    assert selected.status()["retry_events"] == 1


def test_http_429_respects_retry_after(tmp_path: Path) -> None:
    delays: list[float] = []
    transport = FakeTransport(
        TelegramHttpResponse(
            429,
            {"ok": False, "parameters": {"retry_after": 0}},
        ),
        TelegramHttpResponse(200, {"ok": True}),
    )
    selected = notifier(
        tmp_path,
        transport=transport,
        sleeper=delays.append,
    )
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["sent"] == 1
    assert delays == [0]


def test_long_http_429_is_persisted_without_blocking(tmp_path: Path) -> None:
    transport = FakeTransport(
        TelegramHttpResponse(
            429,
            {"ok": False, "parameters": {"retry_after": 30}},
        )
    )
    selected = notifier(tmp_path, transport=transport)
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["deferred"] == 1
    assert selected.status()["active_queue_size"] == 1


def test_restart_recovers_pending_message(tmp_path: Path) -> None:
    first = notifier(tmp_path, transport=FakeTransport())
    queued = first.enqueue_signal(buy_signal())
    assert queued["delivery_status"] == "PENDING"
    transport = FakeTransport(TelegramHttpResponse(200, {"ok": True}))
    restarted = notifier(tmp_path, transport=transport)
    assert restarted.flush()["sent"] == 1
    assert len(transport.calls) == 1


def test_concurrent_flush_claims_notification_exactly_once(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def blocking_transport(
        method: str,
        url: str,
        form: dict[str, str] | None,
        timeout: float,
    ) -> TelegramHttpResponse:
        del method, url, form, timeout
        calls.append("send")
        entered.set()
        assert release.wait(timeout=2.0)
        return TelegramHttpResponse(200, {"ok": True})

    first = TelegramNotifier(
        settings(),
        output_directory=tmp_path / "notifications",
        allowed_markets=("BTC-EUR",),
        transport=blocking_transport,
        clock=lambda: NOW,
    )
    second = TelegramNotifier(
        settings(),
        output_directory=tmp_path / "notifications",
        allowed_markets=("BTC-EUR",),
        transport=blocking_transport,
        clock=lambda: NOW,
    )
    first.enqueue_signal(buy_signal())
    results: list[dict[str, Any]] = []
    workers = [
        threading.Thread(target=lambda item=item: results.append(item.flush()))
        for item in (first, second)
    ]
    workers[0].start()
    assert entered.wait(timeout=1.0)
    workers[1].start()
    time.sleep(0.1)
    release.set()
    for worker in workers:
        worker.join(timeout=3.0)
        assert not worker.is_alive()

    assert calls == ["send"]
    assert sum(int(result["sent"]) for result in results) == 1
    assert first.status()["active_queue_size"] == 0


def test_restart_does_not_resend_ambiguous_sending_message(tmp_path: Path) -> None:
    first = notifier(tmp_path, transport=FakeTransport())
    queued = first.enqueue_signal(buy_signal())
    first._record(
        notification_id=queued["notification_id"],
        signal_id=queued["signal_id"],
        message_type=queued["message_type"],
        delivery_status="SENDING",
        message_hash=queued["message_hash"],
    )
    transport = FakeTransport()
    restarted = notifier(tmp_path, transport=transport)
    result = restarted.flush()
    assert result["failed_final"] == 1
    assert result["active_queue_size"] == 0
    assert not transport.calls


def test_failed_final_status_and_failure_ledger(tmp_path: Path) -> None:
    transport = FakeTransport(
        TelegramHttpResponse(500, {"ok": False}),
        TelegramHttpResponse(500, {"ok": False}),
    )
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=settings(max_retries=1),
    )
    result = selected.process_signals([buy_signal()])
    assert result["delivery"]["failed_final"] == 1
    assert selected.status()["failed_final_count"] == 1
    assert selected.failures_path.read_text(encoding="utf-8").strip()
    second = selected.process_signals([buy_signal()])
    assert second["duplicates"] == 1
    assert len(transport.calls) == 2


def test_telegram_failure_does_not_stop_signal_generation_or_create_order(
    tmp_path: Path,
) -> None:
    selected = notifier(
        tmp_path,
        transport=FakeTransport(
            OSError(),
            OSError(),
            OSError(),
        ),
    )
    result = selected.process_signals([buy_signal()])
    assert result["signal_generation_continues"] is True
    assert result["orders_generated"] == 0
    assert result["orders_submitted"] == 0


def test_rate_limit_leaves_excess_message_queued(tmp_path: Path) -> None:
    transport = FakeTransport()
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=settings(max_messages_per_minute=1),
    )
    selected.enqueue_signal(buy_signal(signal_id="one"))
    selected.enqueue_signal(buy_signal(signal_id="two", stop_loss=57_000))
    result = selected.flush()
    assert result["sent"] == 1
    assert result["active_queue_size"] == 1


def test_signal_scan_is_read_only_and_orderless(tmp_path: Path) -> None:
    selected = notifier(tmp_path)
    result = selected.scan_signals([buy_signal(), {"action": "NO_SIGNAL"}])
    assert result["signals_scanned"] == 2
    assert result["actionable"] == 1
    assert result["orders_generated"] == 0
    assert selected.status()["sent_count"] == 0


def test_cli_telegram_test_reports_zero_order_delta(
    tmp_path: Path,
    isolated_settings,
    monkeypatch,
    capsys,
) -> None:
    selected = notifier(tmp_path)
    monkeypatch.setattr(cli, "_telegram_notifier", lambda _: selected)
    monkeypatch.setattr(
        cli,
        "_execution_table_counts",
        lambda _: {"orders": 0, "fills": 0, "positions": 0},
    )
    code = cli.command_telegram(
        argparse.Namespace(telegram_command="test"),
        isolated_settings,
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["orders_generated"] == 0
    assert payload["telegram_command_changed_execution_state"] is False


def test_cli_telegram_evidence_is_orderless(
    tmp_path: Path,
    isolated_settings,
    monkeypatch,
    capsys,
) -> None:
    from reporting import telegram_signal_evidence

    selected = notifier(tmp_path)
    monkeypatch.setattr(cli, "_telegram_notifier", lambda _: selected)
    monkeypatch.setattr(
        cli,
        "_execution_table_counts",
        lambda _: {"orders": 0, "fills": 0, "positions": 0},
    )
    monkeypatch.setattr(
        telegram_signal_evidence,
        "build_telegram_signal_evidence",
        lambda _settings, force: {
            "artifact": str(tmp_path / "telegram_signal_evidence.json"),
            "evidence_hash": "test-evidence-hash",
            "claim_under_test": {"status": "NOT_CONFIRMED"},
            "prospective_exact_evidence": {
                "hash_chain_status": "VALID",
                "integrity_errors": [],
                "event_count": 0,
                "summary": {"alert_count": 0},
            },
            "legacy_preview_diagnostic": {
                "status": "INDICATIVE_ONLY_ROUNDED_LEVELS",
                "excluded_from_all_promotion_and_authority_decisions": True,
                "summary": {"alert_count": 0},
            },
            "paper_shadow_gate": {
                "status": "COLLECT_EXACT_PROSPECTIVE_EVIDENCE",
                "automatic_live_authority_changes": False,
            },
        },
    )

    code = cli.command_telegram(
        argparse.Namespace(telegram_command="evidence"),
        isolated_settings,
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["telegram_command_changed_execution_state"] is False
    assert payload["orders_generated"] == 0
    assert payload["orders_submitted"] == 0


def test_cli_parser_exposes_telegram_evidence() -> None:
    args = cli.build_parser().parse_args(["telegram", "evidence"])
    assert args.command == "telegram"
    assert args.telegram_command == "evidence"


def test_telegram_failure_does_not_duplicate_paper_order(
    tmp_path: Path,
    isolated_settings,
    monkeypatch,
    capsys,
) -> None:
    from execution.execution import DurableLedger

    transport = FakeTransport(*(OSError() for _ in range(6)))
    selected = notifier(
        tmp_path,
        transport=transport,
        telegram_settings=settings(max_retries=2),
    )
    monkeypatch.setattr(cli, "_telegram_notifier", lambda _: selected)
    configured = isolated_settings.model_copy(
        update={
            "paths": PathSettings(project_root=tmp_path),
            "telegram": settings(max_retries=2),
        }
    )
    code = cli.command_paper(
        argparse.Namespace(
            paper_command="run",
            market="BTC-EUR",
            markets_csv=None,
            candidates=None,
            strategy="telegram-isolation-test",
            capital=2_000.0,
            price=20_000.0,
            quantity=0.001,
            stop_fraction=0.05,
            idempotency_key="telegram-failure-order-idempotency",
        ),
        configured,
    )
    capsys.readouterr()
    events = DurableLedger(cli.paper_ledger(configured)).events()
    assert code == 0
    assert sum(row["event_type"] == "ORDER_INTENT" for row in events) == 1
    assert sum(row["event_type"] == "FILL" for row in events) == 1
