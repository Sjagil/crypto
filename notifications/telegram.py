"""Failure-isolated Telegram delivery for canonical signals and operations.

Telegram is deliberately downstream from signal generation and execution.  This
module cannot create signals, submit orders, or change lifecycle state.  It only
formats existing records, persists an append-only outbound ledger, and performs
bounded HTTPS calls to the Telegram Bot API.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from config.settings import TelegramSettings
from utils.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    clean_text,
    read_json,
    redact,
    sha256_text,
    stable_hash,
    utc_iso,
    utc_now,
)

LOGGER = logging.getLogger(__name__)

QUEUE_STATUSES = frozenset({"PENDING", "SENDING", "RETRY_PENDING"})
TERMINAL_STATUSES = frozenset({"SENT", "FAILED_FINAL"})
DELIVERY_STATUSES = QUEUE_STATUSES | TERMINAL_STATUSES
ALL_STATUSES = DELIVERY_STATUSES | frozenset(
    {"SKIPPED_DUPLICATE", "SKIPPED_FILTER"}
)
SIGNAL_ACTIONS = frozenset(
    {"STRONG_BUY", "BUY", "WATCHLIST", "HOLD", "REDUCE", "EXIT", "AVOID"}
)
ENTRY_ACTIONS = frozenset({"STRONG_BUY", "BUY"})
EXIT_LIFECYCLES = frozenset(
    {"INVALIDATED", "STOPPED_OUT", "EXPIRED", "CLOSED"}
)
MILESTONE_LIFECYCLES = frozenset({"TP1_REACHED", "TP2_REACHED"})
ACTIONABLE_LIFECYCLES = frozenset(
    {
        "MANUAL_ACTIONABLE",
        "TRIGGERED",
        *EXIT_LIFECYCLES,
        *MILESTONE_LIFECYCLES,
    }
)
CRITICAL_EVENTS = frozenset(
    {
        "KILL_SWITCH",
        "KILL_SWITCH_ACTIVATED",
        "DAILY_LOSS_LIMIT",
        "DAILY_LOSS_LIMIT_REACHED",
        "MAXIMUM_DRAWDOWN",
        "MAXIMUM_DRAWDOWN_REACHED",
        "RECONCILIATION_MISMATCH",
        "UNKNOWN_EXCHANGE_ORDER",
        "ORDER_REJECTED",
        "AUTOPILOT_FAILURE_BUDGET",
        "DATABASE_CORRUPTION",
        "UNEXPECTED_BROKER_CALL",
    }
)
WARNING_EVENTS = frozenset(
    {
        "STALE_DATA",
        "PROVIDER_OFFLINE",
        "PROVIDER_OUTAGE",
        "ORDER_PARTIALLY_FILLED",
        "STOP_LOSS_REACHED",
        "OPERATIONAL_DEGRADATION",
        "AUTOPILOT_HEARTBEAT_STALE",
        "AUTOPILOT_DISK_BUDGET",
    }
)
RECOVERY_EVENTS = frozenset(
    {
        "TELEGRAM_CONNECTION_RESTORED",
        "PROVIDER_RECOVERED",
        "ORDER_FILLED",
        "TP1_REACHED",
        "TP2_REACHED",
        "SERVICE_START",
        "SERVICE_STOP",
        "OPERATIONAL_RECOVERY",
    }
)
ORDER_NOTIFICATION_EVENTS = frozenset(
    {
        "ORDER_SUBMITTING",
        "ORDER_PARTIALLY_FILLED",
        "ORDER_FILLED",
        "ORDER_CANCELLED",
        "ORDER_REJECTED",
        "LIVE_ORDER_SUBMITTED",
        "LIVE_ORDER_PARTIALLY_FILLED",
        "LIVE_ORDER_FILLED",
        "LIVE_ORDER_CANCELLED",
        "LIVE_ORDER_REJECTED",
        "PAPER_SIGNAL",
        "PAPER_ORDER",
        "PAPER_FILL",
        "PAPER_ORDER_SUBMITTING",
        "PAPER_ORDER_PARTIALLY_FILLED",
        "PAPER_ORDER_FILLED",
        "PAPER_ORDER_REJECTED",
    }
)
VERIFIED_LIVE_FILL_SOURCES = frozenset(
    {
        "BITVAVO_PRIVATE_ACCOUNT_STREAM",
        "BITVAVO_REST_ORDER_RESPONSE",
        "BITVAVO_REST_RECONCILIATION",
    }
)
TELEGRAM_EVIDENCE_ROUNDTRIP_COST_BPS = 35.0


@dataclass(frozen=True)
class TelegramHttpResponse:
    """Small transport-neutral HTTP result used by real and fake clients."""

    status: int
    payload: dict[str, Any]


TelegramTransport = Callable[
    [str, str, dict[str, str] | None, float],
    TelegramHttpResponse,
]


@lru_cache(maxsize=1)
def _trusted_ssl_context() -> ssl.SSLContext:
    """Use Python and Windows trust anchors without weakening verification."""

    context = ssl.create_default_context()
    certificate_reader = getattr(ssl, "enum_certificates", None)
    if certificate_reader is None:
        return context
    for store in ("ROOT", "CA"):
        try:
            certificates = certificate_reader(store)
        except OSError:
            continue
        for certificate, encoding, _trust in certificates:
            if encoding != "x509_asn":
                continue
            try:
                context.load_verify_locations(
                    cadata=ssl.DER_cert_to_PEM_cert(certificate)
                )
            except ssl.SSLError:
                continue
    return context


def _default_transport(
    method: str,
    url: str,
    form: dict[str, str] | None,
    timeout: float,
) -> TelegramHttpResponse:
    data = urllib.parse.urlencode(form).encode("utf-8") if form is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(  # noqa: S310
            request,
            timeout=timeout,
            context=_trusted_ssl_context(),
        ) as response:
            raw = response.read(1_000_000)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            return TelegramHttpResponse(int(response.status), dict(payload))
    except urllib.error.HTTPError as exc:
        raw = exc.read(1_000_000)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        return TelegramHttpResponse(int(exc.code), dict(payload))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = payload.get(name)
        if value is not None and value != "":
            return value
    return None


def _price(value: Any) -> str:
    selected = _finite(value)
    if selected is None:
        return "n.b."
    decimals = 4 if abs(selected) < 1 else 2
    rendered = f"{selected:,.{decimals}f}"
    return "€" + rendered.replace(",", "_").replace(".", ",").replace("_", ".")


def _number(value: Any, decimals: int = 2) -> str:
    selected = _finite(value)
    if selected is None:
        return "n.b."
    return f"{selected:.{decimals}f}".replace(".", ",")


def _percentage(value: Any) -> str:
    selected = _finite(value)
    if selected is None:
        return "n.b."
    if abs(selected) <= 1:
        selected *= 100
    return f"{selected:.2f}%".replace(".", ",")


def _visible_datetime(value: Any) -> str:
    selected = _utc(value)
    return selected.strftime("%d-%m-%Y %H:%M UTC") if selected else "n.b."


def _duration(value: Any) -> str:
    selected = _finite(value)
    if selected is None or selected < 0:
        return "n.b."
    seconds = int(selected)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, _ = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}u")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _safe_reason(value: Any) -> str:
    text = clean_text(str(value or "Geen aanvullende reden opgegeven."), maximum_length=500)
    # Raw query strings can contain credentials even when callers made a mistake.
    return text.split("?", 1)[0]


def _entry_zone(signal: Mapping[str, Any]) -> tuple[Any, Any]:
    low = _first(signal, "entry_low", "entry_min", "entry_zone_low")
    high = _first(signal, "entry_high", "entry_max", "entry_zone_high")
    zone = signal.get("entry_zone")
    if isinstance(zone, (list, tuple)) and len(zone) >= 2:
        low, high = zone[0], zone[1]
    elif isinstance(zone, Mapping):
        low = low if low is not None else _first(zone, "low", "min", "from")
        high = high if high is not None else _first(zone, "high", "max", "to")
    elif zone is not None and low is None and high is None:
        low = high = zone
    preferred = _first(signal, "preferred_entry", "entry_price", "price")
    if low is None:
        low = preferred
    if high is None:
        high = preferred
    return low, high


def _reward_risk(signal: Mapping[str, Any], target_name: str) -> float | None:
    names = [
        f"reward_risk_{target_name}",
        f"risk_reward_{target_name}",
    ]
    if target_name == "tp1":
        names.extend(("reward_risk", "risk_reward"))
    direct = _first(signal, *names)
    if (selected := _finite(direct)) is not None:
        return selected
    entry = _finite(_first(signal, "preferred_entry", "entry_price", "price"))
    stop = _finite(_first(signal, "stop_loss", "stop_price", "stop"))
    target = _finite(
        _first(
            signal,
            target_name,
            "take_profit_1" if target_name == "tp1" else "take_profit_2",
        )
    )
    if entry is None or stop is None or target is None or entry == stop:
        return None
    return abs(target - entry) / abs(entry - stop)


class TelegramNotifier:
    """Persistent, deduplicated and secret-safe Telegram notification adapter."""

    def __init__(
        self,
        settings: TelegramSettings,
        *,
        output_directory: Path,
        allowed_markets: Iterable[str],
        transport: TelegramTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.settings = settings
        self.output_directory = output_directory.resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.allowed_markets = frozenset(
            str(market).strip().upper().replace("/", "-")
            for market in allowed_markets
        )
        self.transport = transport or _default_transport
        self.sleeper = sleeper
        self.clock = clock
        self.ledger_path = self.output_directory / "telegram_notifications.jsonl"
        self.failures_path = self.output_directory / "telegram_failures.jsonl"
        self.health_path = self.output_directory / "telegram_health.json"
        self.status_path = self.output_directory / "telegram_status.json"
        self.preview_path = self.output_directory / "latest_telegram_preview.html"
        self.preview_cache_path = self.output_directory / "telegram_preview.json"
        self.opportunity_evidence_path = (
            self.output_directory / "telegram_opportunity_evidence.jsonl"
        )
        self.queue_lock_path = self.output_directory / "telegram_queue.lock"
        self._secrets = tuple(
            value
            for secret in (settings.bot_token, settings.chat_id)
            if secret is not None
            and (value := secret.get_secret_value())
        )
        self.ledger_path.touch(exist_ok=True)
        self.failures_path.touch(exist_ok=True)
        self.opportunity_evidence_path.touch(exist_ok=True)
        if not self.preview_cache_path.is_file():
            atomic_write_json(self.preview_cache_path, [])
        if not self.preview_path.is_file():
            atomic_write_text(
                self.preview_path,
                (
                    "<!doctype html><html lang=\"nl\"><meta charset=\"utf-8\">"
                    "<title>Telegram preview</title><h1>Telegram preview</h1>"
                    "<p>Nog geen berichten.</p></html>"
                ),
            )
        if self.enabled_status in {"DISABLED", "DISABLED_MISSING_CONFIG"}:
            self.health(probe=False)

    @property
    def enabled_status(self) -> str:
        if not self.settings.notifications_enabled:
            return "DISABLED"
        if not self.settings.configured:
            return "DISABLED_MISSING_CONFIG"
        if self.settings.dry_run:
            return "DRY_RUN"
        return "ENABLED"

    @property
    def chat_identity_hash(self) -> str | None:
        if not self.settings.chat_id:
            return None
        value = self.settings.chat_id.get_secret_value()
        return stable_hash(["telegram-chat", value], length=16) if value else None

    def _safe(self, value: Any) -> Any:
        return redact(value, self._secrets)

    @contextmanager
    def _queue_lock(self, *, wait_seconds: float = 2.0) -> Iterable[bool]:
        """Serialize queue claims across scheduler, CLI and supervisor processes.

        Telegram is downstream of trading, so contention is bounded: a caller that
        cannot claim the queue leaves it for the next cycle instead of blocking the
        execution loop.  The exclusive-create lock also closes the read-then-append
        race that could previously deliver one notification twice.
        """

        deadline = time.monotonic() + max(0.0, wait_seconds)
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    self.queue_lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(
                    descriptor,
                    json.dumps(
                        {"pid": os.getpid(), "claimed_at": utc_iso(self.clock())}
                    ).encode("utf-8"),
                )
            except FileExistsError:
                try:
                    age_seconds = time.time() - self.queue_lock_path.stat().st_mtime
                except OSError:
                    continue
                stale_after = max(
                    300.0,
                    float(self.settings.request_timeout_seconds)
                    * float(self.settings.max_retries + 2),
                )
                if age_seconds > stale_after:
                    try:
                        self.queue_lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.025)
        try:
            yield True
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                self.queue_lock_path.unlink()
            except OSError:
                pass

    def _url(self, method: str) -> str:
        if not self.settings.bot_token:
            raise RuntimeError("TELEGRAM_CONFIGURATION_MISSING")
        token = self.settings.bot_token.get_secret_value()
        if not token:
            raise RuntimeError("TELEGRAM_CONFIGURATION_MISSING")
        return f"https://api.telegram.org/bot{token}/{method}"

    def _history(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.ledger_path)

    def _delivery_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for record in self._history():
            notification_id = str(record.get("notification_id") or "")
            if notification_id and record.get("delivery_status") in ALL_STATUSES:
                state[notification_id] = record
        return state

    def _record(
        self,
        *,
        notification_id: str,
        signal_id: str | None,
        message_type: str,
        delivery_status: str,
        message_hash: str,
        retry_count: int = 0,
        telegram_http_status: int | None = None,
        reason_code: str | None = None,
        next_attempt_at: datetime | None = None,
        delivery_mode: str = "TELEGRAM",
    ) -> dict[str, Any]:
        if delivery_status not in ALL_STATUSES:
            raise ValueError(f"unsupported Telegram delivery status: {delivery_status}")
        record = {
            "notification_id": notification_id,
            "signal_id": signal_id,
            "timestamp": utc_iso(self.clock()),
            "message_type": clean_text(message_type, maximum_length=80),
            "delivery_status": delivery_status,
            "telegram_http_status": telegram_http_status,
            "retry_count": int(retry_count),
            "message_hash": message_hash,
            "chat_identity_hash": self.chat_identity_hash,
            "reason_code": clean_text(reason_code, maximum_length=120)
            if reason_code
            else None,
            "next_attempt_at": utc_iso(next_attempt_at) if next_attempt_at else None,
            "delivery_mode": delivery_mode,
        }
        append_jsonl(self.ledger_path, self._safe(record))
        return record

    def _append_opportunity_evidence(
        self,
        *,
        notification_id: str,
        rows: Sequence[Mapping[str, Any]],
        recorded_at: datetime,
    ) -> dict[str, Any]:
        """Durably bind exact tactical rows to the notification cohort.

        This ledger is intentionally separate from Telegram delivery history:
        delivery remains downstream-only, while this hash chain preserves the
        exact, prospectively observable levels needed for independent outcome
        measurement.  Callers must hold ``_queue_lock`` so notification and
        evidence deduplication cannot race across processes.
        """

        history = _read_jsonl(self.opportunity_evidence_path)
        previous_hash = (
            str(history[-1].get("record_hash") or "")
            if history
            else "GENESIS"
        )
        body = {
            "schema_version": "telegram_opportunity_evidence_event_v1",
            "recorded_at": utc_iso(recorded_at),
            "notification_id": notification_id,
            "delivery_status_at_capture": "PENDING",
            "rows": [dict(row) for row in rows],
            "previous_hash": previous_hash,
        }
        record = {**body, "record_hash": stable_hash(body, length=64)}
        append_jsonl(self.opportunity_evidence_path, self._safe(record))
        return record

    def _failure(
        self,
        *,
        notification_id: str,
        message_type: str,
        error_code: str,
        retry_count: int,
        http_status: int | None,
    ) -> None:
        append_jsonl(
            self.failures_path,
            {
                "notification_id": notification_id,
                "message_type": clean_text(message_type, maximum_length=80),
                "recorded_at": utc_iso(self.clock()),
                "error_code": clean_text(error_code, maximum_length=120),
                "retry_count": retry_count,
                "telegram_http_status": http_status,
                "chat_identity_hash": self.chat_identity_hash,
            },
        )

    def _preview(self, notification_id: str, message_type: str, message: str) -> None:
        safe_message = str(self._safe(message))
        try:
            previews = (
                list(read_json(self.preview_cache_path))
                if self.preview_cache_path.is_file()
                else []
            )
        except (OSError, ValueError, TypeError):
            previews = []
        previews.append(
            {
                "notification_id": notification_id,
                "message_type": message_type,
                "created_at": utc_iso(self.clock()),
                "message": safe_message,
            }
        )
        previews = previews[-50:]
        atomic_write_json(self.preview_cache_path, previews)
        sections = "".join(
            "<article><h2>"
            + html.escape(str(item["message_type"]))
            + "</h2><time>"
            + html.escape(str(item["created_at"]))
            + "</time><pre>"
            + html.escape(str(item["message"]))
            + "</pre></article>"
            for item in reversed(previews)
        )
        atomic_write_text(
            self.preview_path,
            (
                "<!doctype html><html lang=\"nl\"><meta charset=\"utf-8\">"
                "<title>Telegram preview</title><style>"
                "body{font:14px system-ui;max-width:900px;margin:2rem auto}"
                "article{border:1px solid #ddd;padding:1rem;margin:1rem 0}"
                "pre{white-space:pre-wrap}</style><h1>Telegram preview</h1>"
                + sections
                + "</html>"
            ),
        )

    def _identity(self, signal: Mapping[str, Any]) -> str:
        low, high = _entry_zone(signal)
        payload = {
            "signal_id": _first(signal, "signal_id", "external_id", "id"),
            "signal_status": _first(
                signal,
                "lifecycle_status",
                "signal_status",
                "status",
            ),
            "action": self._action(signal),
            "market": _first(signal, "market", "canonical_market"),
            "strategy_dna": _first(
                signal,
                "strategy_dna_hash",
                "strategy_dna",
                "candidate_manifest_hash",
            ),
            "entry_zone": [low, high],
            "stop_loss": _first(signal, "stop_loss", "stop_price", "stop"),
            "take_profit_1": _first(signal, "take_profit_1", "tp1"),
            "take_profit_2": _first(signal, "take_profit_2", "tp2"),
            "confidence": _first(signal, "confidence", "confidence_pct"),
            "expiration": _first(
                signal,
                "expires_at",
                "valid_until",
                "expiration",
            ),
        }
        return stable_hash(payload, length=40)

    @staticmethod
    def _action(signal: Mapping[str, Any]) -> str:
        lifecycle = str(
            _first(signal, "lifecycle_status", "signal_status", "status") or ""
        ).upper()
        if lifecycle in EXIT_LIFECYCLES:
            return "EXIT"
        raw = str(_first(signal, "action", "signal", "recommendation") or "").upper()
        aliases = {
            "ENTER": "BUY",
            "LONG": "BUY",
            "NO_ENTRY": "NO_SIGNAL",
            "NO_SIGNAL": "NO_SIGNAL",
        }
        return aliases.get(raw, raw)

    def _filter(self, signal: Mapping[str, Any]) -> str | None:
        action = self._action(signal)
        lifecycle = str(
            _first(signal, "lifecycle_status", "signal_status", "status") or ""
        ).upper()
        market = str(_first(signal, "market", "canonical_market") or "").upper()
        if action in {"", "NO_SIGNAL"}:
            return "NO_SIGNAL"
        if action not in SIGNAL_ACTIONS and lifecycle not in MILESTONE_LIFECYCLES:
            return "UNSUPPORTED_ACTION"
        if action == "WATCHLIST" and not self.settings.send_watchlist:
            return "WATCHLIST_DISABLED"
        if action in {"EXIT", "REDUCE"} or lifecycle in (
            EXIT_LIFECYCLES | MILESTONE_LIFECYCLES
        ):
            if not self.settings.send_exits:
                return "EXITS_DISABLED"
        elif not self.settings.send_signals:
            return "SIGNALS_DISABLED"
        authority = str(
            _first(signal, "signal_authority", "authority", "lifecycle_status") or ""
        ).upper()
        if authority != "MANUAL_ACTIONABLE" and lifecycle not in ACTIONABLE_LIFECYCLES:
            return "INSUFFICIENT_SIGNAL_AUTHORITY"
        confidence = _finite(_first(signal, "confidence", "confidence_pct"))
        if (
            action in ENTRY_ACTIONS | {"WATCHLIST"}
            and (confidence is None or confidence < self.settings.min_confidence)
        ):
            return "CONFIDENCE_BELOW_MINIMUM"
        if market not in self.allowed_markets:
            return "MARKET_NOT_ALLOWED"
        if bool(_first(signal, "data_stale", "stale_data", "stale")):
            return "STALE_DATA"
        expiration = _utc(
            _first(signal, "expires_at", "valid_until", "expiration")
        )
        if (
            expiration is not None
            and expiration <= self.clock()
            and lifecycle not in (EXIT_LIFECYCLES | MILESTONE_LIFECYCLES)
        ):
            return "SIGNAL_EXPIRED"
        if action in ENTRY_ACTIONS:
            frozen = bool(
                _first(signal, "strategy_frozen", "parameters_frozen")
            ) or str(
                _first(signal, "candidate_state", "strategy_state") or ""
            ).upper() in {
                "FROZEN",
                "FROZEN_SHADOW",
                "SHADOW_CANDIDATE",
                "SHADOW_ACTIVE",
                "PAPER_CANDIDATE",
                "PAPER_ACTIVE",
            }
            if not frozen:
                return "STRATEGY_NOT_FROZEN"
            stop = _first(signal, "stop_loss", "stop_price", "stop")
            target = _first(signal, "take_profit_1", "tp1", "exit_price")
            if stop is None:
                return "STOP_MISSING"
            if target is None:
                return "EXIT_MISSING"
            reward_risk = _reward_risk(signal, "tp1")
            if (
                reward_risk is None
                or reward_risk < self.settings.min_reward_risk
            ):
                return "REWARD_RISK_BELOW_MINIMUM"
        return None

    def format_signal(self, signal: Mapping[str, Any]) -> str:
        action = self._action(signal)
        lifecycle = str(
            _first(signal, "lifecycle_status", "signal_status", "status") or ""
        ).upper()
        market = str(_first(signal, "market", "canonical_market") or "ONBEKEND").upper()
        strategy = str(
            _first(signal, "strategy_name", "strategy_id", "candidate_id")
            or "Onbekende strategie"
        )
        timeframe = str(_first(signal, "timeframe") or "n.b.")
        signal_id = str(_first(signal, "signal_id", "external_id", "id") or "n.b.")
        current = _first(signal, "current_price", "price", "entry_price")
        if lifecycle in MILESTONE_LIFECYCLES:
            target = "TP1" if lifecycle == "TP1_REACHED" else "TP2"
            return (
                f"✅ {target} BEREIKT — {market}\n\n"
                f"Huidige prijs: {_price(current)}\n"
                f"Strategie: {strategy}\n"
                f"Timeframe: {timeframe}\n"
                f"Signal ID: {signal_id}"
            )
        if action in {"EXIT", "REDUCE"}:
            return (
                f"🔴 {action} — {market}\n\n"
                f"Actie: {'positie sluiten' if action == 'EXIT' else 'positie verkleinen'}\n"
                f"Huidige prijs: {_price(current)}\n"
                f"Reden: {_safe_reason(_first(signal, 'reason', 'reason_code', 'explanation'))}\n"
                f"Oorspronkelijke entry: {_price(_first(signal, 'original_entry', 'entry_price'))}\n"
                f"Stop-loss: {_price(_first(signal, 'stop_loss', 'stop_price', 'stop'))}\n"
                f"Resultaat sinds signaal: {_percentage(_first(signal, 'return_since_signal', 'result_pct'))}\n\n"
                f"Strategie: {strategy}\nTimeframe: {timeframe}\nSignal ID: {signal_id}"
            )
        low, high = _entry_zone(signal)
        confidence = _first(signal, "confidence", "confidence_pct")
        expiration = _first(signal, "expires_at", "valid_until", "expiration")
        if action == "WATCHLIST":
            return (
                f"🟡 WATCHLIST — {market}\n\n"
                f"Huidige prijs: {_price(current)}\n"
                f"Interessante entryzone: {_price(low)} – {_price(high)}\n"
                f"Trigger: {_safe_reason(_first(signal, 'trigger', 'entry_trigger'))}\n"
                f"Invalidatie: {_price(_first(signal, 'invalidation', 'stop_loss', 'stop'))}\n"
                f"Timeframe: {timeframe}\nConfidence: {_number(confidence, 0)}%\n"
                f"Geldig tot: {_visible_datetime(expiration)}\n\nNog geen entry."
            )
        emoji = "🟢" if action in ENTRY_ACTIONS else "🟡"
        return (
            f"{emoji} {action} — {market}\n\n"
            f"Entry: {_price(low)} – {_price(high)}\n"
            f"Voorkeursprijs: {_price(_first(signal, 'preferred_entry', 'entry_price'))}\n"
            f"Stop-loss: {_price(_first(signal, 'stop_loss', 'stop_price', 'stop'))}\n"
            f"Take-profit 1: {_price(_first(signal, 'take_profit_1', 'tp1'))}\n"
            f"Take-profit 2: {_price(_first(signal, 'take_profit_2', 'tp2'))}\n\n"
            f"Risk/reward TP1: {_number(_reward_risk(signal, 'tp1'), 1)}\n"
            f"Risk/reward TP2: {_number(_reward_risk(signal, 'tp2'), 1)}\n"
            f"Maximaal gepland verlies: {_price(_first(signal, 'maximum_planned_loss_eur', 'max_loss_eur'))}\n"
            f"Voorgestelde orderwaarde: {_price(_first(signal, 'suggested_order_value_eur', 'order_value_eur'))}\n\n"
            f"Strategie: {strategy}\nTimeframe: {timeframe}\n"
            f"Confidence: {_number(confidence, 0)}%\n"
            f"Geldig tot: {_visible_datetime(expiration)}\n\n"
            "Reden:\n"
            f"{_safe_reason(_first(signal, 'reason', 'explanation', 'reason_code'))}\n\n"
            "Dit is een modelsignaal, geen gegarandeerde uitkomst."
        )

    def enqueue(
        self,
        *,
        notification_id: str,
        signal_id: str | None,
        message_type: str,
        message: str,
    ) -> dict[str, Any]:
        with self._queue_lock() as acquired:
            if not acquired:
                return {
                    "notification_id": notification_id,
                    "delivery_status": "RETRY_PENDING",
                    "reason_code": "TELEGRAM_QUEUE_BUSY",
                }
            return self._enqueue_unlocked(
                notification_id=notification_id,
                signal_id=signal_id,
                message_type=message_type,
                message=message,
            )

    def _enqueue_unlocked(
        self,
        *,
        notification_id: str,
        signal_id: str | None,
        message_type: str,
        message: str,
    ) -> dict[str, Any]:
        safe_message = str(self._safe(message))
        message_hash = stable_hash(safe_message, length=64)
        history = self._history()
        if self.enabled_status in {"DISABLED", "DISABLED_MISSING_CONFIG"}:
            return self._record(
                notification_id=notification_id,
                signal_id=signal_id,
                message_type=message_type,
                delivery_status="SKIPPED_FILTER",
                message_hash=message_hash,
                reason_code=self.enabled_status,
            )
        delivered = any(
            row.get("notification_id") == notification_id
            and row.get("delivery_status") == "SENT"
            for row in history
        )
        failed_final = any(
            row.get("notification_id") == notification_id
            and row.get("delivery_status") == "FAILED_FINAL"
            for row in history
        )
        state = self._delivery_state().get(notification_id)
        already_skipped = any(
            row.get("notification_id") == notification_id
            and row.get("delivery_status") == "SKIPPED_DUPLICATE"
            for row in history
        )
        if delivered or failed_final or (
            state is not None and state.get("delivery_status") in QUEUE_STATUSES
        ):
            if not already_skipped:
                return self._record(
                    notification_id=notification_id,
                    signal_id=signal_id,
                    message_type=message_type,
                    delivery_status="SKIPPED_DUPLICATE",
                    message_hash=message_hash,
                    reason_code=(
                        "IDENTICAL_NOTIFICATION_FAILED_FINAL"
                        if failed_final
                        else "IDENTICAL_NOTIFICATION_ALREADY_KNOWN"
                    ),
                )
            return {
                "notification_id": notification_id,
                "delivery_status": "SKIPPED_DUPLICATE",
                "message_hash": message_hash,
            }
        record = self._record(
            notification_id=notification_id,
            signal_id=signal_id,
            message_type=message_type,
            delivery_status="PENDING",
            message_hash=message_hash,
        )
        self._preview(notification_id, message_type, safe_message)
        return {**record, "message": safe_message}

    def enqueue_signal(self, signal: Mapping[str, Any]) -> dict[str, Any]:
        notification_id = self._identity(signal)
        signal_id = str(_first(signal, "signal_id", "external_id", "id") or "")
        action = self._action(signal)
        rejection = self._filter(signal)
        message_hash = stable_hash(
            {"notification_id": notification_id, "reason": rejection},
            length=64,
        )
        if rejection is not None:
            if rejection == "NO_SIGNAL":
                return {
                    "notification_id": notification_id,
                    "delivery_status": "SKIPPED_FILTER",
                    "reason_code": rejection,
                }
            history = self._history()
            if not any(
                row.get("notification_id") == notification_id
                and row.get("delivery_status") == "SKIPPED_FILTER"
                for row in history
            ):
                return self._record(
                    notification_id=notification_id,
                    signal_id=signal_id or None,
                    message_type=action or "NO_SIGNAL",
                    delivery_status="SKIPPED_FILTER",
                    message_hash=message_hash,
                    reason_code=rejection,
                )
            return {
                "notification_id": notification_id,
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": rejection,
            }
        return self.enqueue(
            notification_id=notification_id,
            signal_id=signal_id or None,
            message_type=action,
            message=self.format_signal(signal),
        )

    def _rate_limit(self) -> dict[str, Any]:
        now = self.clock()
        cutoff = now - timedelta(minutes=1)
        timestamps = [
            selected
            for row in self._history()
            if row.get("delivery_status") == "SENT"
            and (selected := _utc(row.get("timestamp"))) is not None
            and selected >= cutoff
        ]
        used = len(timestamps)
        return {
            "maximum_per_minute": self.settings.max_messages_per_minute,
            "used_last_minute": used,
            "remaining": max(0, self.settings.max_messages_per_minute - used),
            "limited": used >= self.settings.max_messages_per_minute,
        }

    def _message_for(self, notification_id: str) -> str | None:
        if not self.preview_cache_path.is_file():
            return None
        try:
            previews = read_json(self.preview_cache_path)
        except (OSError, ValueError, TypeError):
            return None
        for item in reversed(previews):
            if item.get("notification_id") == notification_id:
                return str(item.get("message") or "")
        return None

    def _retry_after(self, response: TelegramHttpResponse) -> float:
        parameters = response.payload.get("parameters")
        if not isinstance(parameters, Mapping):
            return 0.0
        return max(0.0, float(parameters.get("retry_after") or 0.0))

    def _deliver(
        self,
        record: Mapping[str, Any],
        message: str,
    ) -> dict[str, Any]:
        notification_id = str(record["notification_id"])
        signal_id = (
            str(record["signal_id"]) if record.get("signal_id") is not None else None
        )
        message_type = str(record["message_type"])
        message_hash = str(record["message_hash"])
        if self.enabled_status in {"DISABLED", "DISABLED_MISSING_CONFIG"}:
            return {
                "notification_id": notification_id,
                "delivery_status": self.enabled_status,
            }
        first_retry = min(
            int(record.get("retry_count") or 0),
            self.settings.max_retries,
        )
        for retry_count in range(first_retry, self.settings.max_retries + 1):
            self._record(
                notification_id=notification_id,
                signal_id=signal_id,
                message_type=message_type,
                delivery_status="SENDING",
                message_hash=message_hash,
                retry_count=retry_count,
                delivery_mode="DRY_RUN" if self.settings.dry_run else "TELEGRAM",
            )
            if self.settings.dry_run:
                return self._record(
                    notification_id=notification_id,
                    signal_id=signal_id,
                    message_type=message_type,
                    delivery_status="SENT",
                    message_hash=message_hash,
                    retry_count=retry_count,
                    delivery_mode="DRY_RUN",
                )
            try:
                response = self.transport(
                    "POST",
                    self._url("sendMessage"),
                    {
                        "chat_id": self.settings.chat_id.get_secret_value()
                        if self.settings.chat_id
                        else "",
                        "text": message,
                        "disable_web_page_preview": "true",
                    },
                    self.settings.request_timeout_seconds,
                )
                if response.status == 200 and response.payload.get("ok", True):
                    return self._record(
                        notification_id=notification_id,
                        signal_id=signal_id,
                        message_type=message_type,
                        delivery_status="SENT",
                        message_hash=message_hash,
                        retry_count=retry_count,
                        telegram_http_status=response.status,
                    )
                transient = response.status == 429 or response.status >= 500
                error_code = (
                    "TELEGRAM_RATE_LIMITED"
                    if response.status == 429
                    else f"TELEGRAM_HTTP_{response.status}"
                )
                delay = (
                    self._retry_after(response)
                    if response.status == 429
                    else min(2**retry_count, self.settings.request_timeout_seconds)
                )
                http_status = response.status
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                transient = True
                error_code = f"TELEGRAM_{type(exc).__name__.upper()}"
                delay = min(2**retry_count, self.settings.request_timeout_seconds)
                http_status = None
            except Exception as exc:  # defensive isolation from custom transports
                transient = False
                error_code = f"TELEGRAM_{type(exc).__name__.upper()}"
                delay = 0.0
                http_status = None
            if transient and retry_count < self.settings.max_retries:
                next_attempt = self.clock() + timedelta(seconds=delay)
                pending = self._record(
                    notification_id=notification_id,
                    signal_id=signal_id,
                    message_type=message_type,
                    delivery_status="RETRY_PENDING",
                    message_hash=message_hash,
                    retry_count=retry_count + 1,
                    telegram_http_status=http_status,
                    reason_code=error_code,
                    next_attempt_at=next_attempt,
                )
                # Long Telegram retry_after values stay persistent and are recovered
                # by the next scheduler cycle instead of blocking the trading loop.
                if delay > self.settings.request_timeout_seconds:
                    return pending
                self.sleeper(delay)
                continue
            failed = self._record(
                notification_id=notification_id,
                signal_id=signal_id,
                message_type=message_type,
                delivery_status="FAILED_FINAL",
                message_hash=message_hash,
                retry_count=retry_count,
                telegram_http_status=http_status,
                reason_code=error_code,
            )
            self._failure(
                notification_id=notification_id,
                message_type=message_type,
                error_code=error_code,
                retry_count=retry_count,
                http_status=http_status,
            )
            return failed
        raise AssertionError("Telegram retry loop exhausted")

    def flush(self) -> dict[str, Any]:
        with self._queue_lock() as acquired:
            if not acquired:
                status = self.status(write=True)
                return {
                    "status": "DEFERRED_QUEUE_BUSY",
                    "sent": 0,
                    "failed_final": 0,
                    "deferred": status["active_queue_size"],
                    "active_queue_size": status["active_queue_size"],
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }
            return self._flush_unlocked()

    def _flush_unlocked(self) -> dict[str, Any]:
        state = self._delivery_state()
        sent = 0
        failed = 0
        deferred = 0
        for notification_id, record in state.items():
            if record.get("delivery_status") not in QUEUE_STATUSES:
                continue
            if record.get("delivery_status") == "SENDING":
                failed += 1
                retry_count = int(record.get("retry_count") or 0)
                self._record(
                    notification_id=notification_id,
                    signal_id=record.get("signal_id"),
                    message_type=str(record.get("message_type") or "UNKNOWN"),
                    delivery_status="FAILED_FINAL",
                    message_hash=str(record.get("message_hash") or ""),
                    retry_count=retry_count,
                    reason_code="AMBIGUOUS_PRIOR_SEND_NOT_RETRIED",
                )
                self._failure(
                    notification_id=notification_id,
                    message_type=str(record.get("message_type") or "UNKNOWN"),
                    error_code="AMBIGUOUS_PRIOR_SEND_NOT_RETRIED",
                    retry_count=retry_count,
                    http_status=None,
                )
                continue
            next_attempt = _utc(record.get("next_attempt_at"))
            if next_attempt is not None and next_attempt > self.clock():
                deferred += 1
                continue
            if self._rate_limit()["limited"]:
                deferred += 1
                break
            message = self._message_for(notification_id)
            if message is None:
                failed += 1
                self._record(
                    notification_id=notification_id,
                    signal_id=record.get("signal_id"),
                    message_type=str(record.get("message_type") or "UNKNOWN"),
                    delivery_status="FAILED_FINAL",
                    message_hash=str(record.get("message_hash") or ""),
                    retry_count=int(record.get("retry_count") or 0),
                    reason_code="MESSAGE_PREVIEW_MISSING",
                )
                continue
            result = self._deliver(record, message)
            sent += int(result.get("delivery_status") == "SENT")
            failed += int(result.get("delivery_status") == "FAILED_FINAL")
            deferred += int(result.get("delivery_status") == "RETRY_PENDING")
        status = self.status(write=True)
        return {
            "status": "PASSED" if failed == 0 else "DEGRADED",
            "sent": sent,
            "failed_final": failed,
            "deferred": deferred,
            "active_queue_size": status["active_queue_size"],
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def process_signals(self, signals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        queued = 0
        filtered = 0
        duplicates = 0
        malformed = 0
        for signal in signals:
            try:
                result = self.enqueue_signal(signal)
            except Exception as exc:  # signal production must remain independent
                malformed += 1
                self._failure(
                    notification_id=stable_hash(
                        ["malformed-signal", type(exc).__name__, self.clock()],
                        length=40,
                    ),
                    message_type="SIGNAL",
                    error_code=f"SIGNAL_NOTIFICATION_{type(exc).__name__.upper()}",
                    retry_count=0,
                    http_status=None,
                )
                continue
            status = result.get("delivery_status")
            queued += int(status == "PENDING")
            filtered += int(status == "SKIPPED_FILTER")
            duplicates += int(status == "SKIPPED_DUPLICATE")
        delivery = self.flush()
        return {
            "status": "PASSED" if malformed == 0 else "DEGRADED",
            "signals_considered": queued + filtered + duplicates + malformed,
            "queued": queued,
            "filtered": filtered,
            "duplicates": duplicates,
            "malformed": malformed,
            "delivery": delivery,
            "signal_generation_continues": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def scan_signals(self, signals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Classify canonical signals without enqueueing or sending anything."""

        details: list[dict[str, Any]] = []
        for signal in signals:
            notification_id = self._identity(signal)
            reason = self._filter(signal)
            details.append(
                {
                    "notification_id": notification_id,
                    "signal_id": str(
                        _first(signal, "signal_id", "external_id", "id") or ""
                    ),
                    "market": str(
                        _first(signal, "market", "canonical_market") or ""
                    ).upper(),
                    "action": self._action(signal),
                    "eligible_for_telegram": reason is None,
                    "filter_reason": reason,
                }
            )
        return {
            "status": "PASSED",
            "signals_scanned": len(details),
            "actionable": sum(row["eligible_for_telegram"] for row in details),
            "filtered": sum(not row["eligible_for_telegram"] for row in details),
            "signals": details,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def notify_system_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = str(event_type).upper()
        if selected == "POSITION_CLOSED":
            return self.notify_position_closed(payload)
        if selected in ORDER_NOTIFICATION_EVENTS:
            return self.notify_order_event(selected, payload)
        is_risk = selected in CRITICAL_EVENTS or selected in WARNING_EVENTS
        if is_risk and not self.settings.send_risk_alerts:
            return {"delivery_status": "SKIPPED_FILTER", "reason_code": "RISK_ALERTS_DISABLED"}
        if not is_risk and not self.settings.send_system_alerts:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "SYSTEM_ALERTS_DISABLED",
            }
        icon = (
            "🚨"
            if selected in CRITICAL_EVENTS
            else "⚠️"
            if selected in WARNING_EVENTS
            else "✅"
            if selected in RECOVERY_EVENTS
            else "ℹ️"
        )
        safe_payload = self._safe(dict(payload))
        details = [
            f"{str(key).replace('_', ' ').title()}: {_safe_reason(value)}"
            for key, value in safe_payload.items()
            if key
            in {
                "market",
                "status",
                "reason",
                "reason_code",
                "mode",
                "candidate_id",
                "provider",
            }
            and value is not None
            and value != ""
        ]
        message = f"{icon} {selected.replace('_', ' ')}"
        if details:
            message += "\n\n" + "\n".join(details)
        identity = stable_hash(
            {
                "event_type": selected,
                "payload": {
                    key: safe_payload.get(key)
                    for key in (
                        "market",
                        "status",
                        "reason",
                        "reason_code",
                        "mode",
                        "candidate_id",
                        "provider",
                    )
                },
            },
            length=40,
        )
        result = self.enqueue(
            notification_id=identity,
            signal_id=None,
            message_type=selected,
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_position_closed(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        market = str(_first(payload, "market") or "ONBEKEND").upper()
        signal_id = str(_first(payload, "signal_id", "opportunity_id") or "")
        net_pnl = _finite(_first(payload, "net_pnl_eur"))
        net_pnl_label = (
            "n.b."
            if net_pnl is None
            else ("+" if net_pnl > 0 else "−" if net_pnl < 0 else "")
            + _price(abs(net_pnl))
        )
        lines = [
            f"✅ POSITION CLOSED — {market}",
            "",
            f"Entry: {_price(_first(payload, 'entry_price'))}",
            f"Exit: {_price(_first(payload, 'exit_price', 'price'))}",
            f"Quantity: {_number(_first(payload, 'sell_quantity', 'quantity'), 8)}",
            f"Fees: {_price(_first(payload, 'fees_eur', 'fee_eur'))}",
            f"Netto PnL: {net_pnl_label}",
            f"Reden: {_safe_reason(_first(payload, 'reason', 'reason_code'))}",
            f"Strategie: {_first(payload, 'strategy_id', 'playbook_id') or 'n.b.'}",
            "Execution: LIVE/EXCHANGE — round trip gereconcilieerd",
        ]
        identity = stable_hash(
            ["POSITION_CLOSED", signal_id, market],
            length=40,
        )
        result = self.enqueue(
            notification_id=identity,
            signal_id=signal_id or None,
            message_type="POSITION_CLOSED",
            message="\n".join(lines),
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_opportunity_update(
        self,
        opportunities: Iterable[Mapping[str, Any]],
        *,
        maximum_rows: int = 3,
    ) -> dict[str, Any]:
        """Send one compact, material-change-only tactical opportunity update.

        Tactical opportunities are observations, not execution authority.  This
        method deliberately bypasses the actionable-signal filter while keeping
        them labelled as unconfirmed and non-executable.  It never creates an
        order and persists a small comparison state so tiny hourly price changes
        do not produce repetitive watchlist messages.
        """

        if not self.settings.send_watchlist:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "WATCHLIST_DISABLED",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        selected: list[dict[str, Any]] = []
        observed_at = self.clock().astimezone(UTC)
        for raw in opportunities:
            market = str(_first(raw, "market", "canonical_market") or "").upper()
            status = str(_first(raw, "status") or "NEAR_ENTRY").upper()
            if market not in self.allowed_markets or status not in {
                "NEAR_ENTRY",
                "ACTIONABLE",
                "EARLY_MOMENTUM_ALERT",
                "PULLBACK_PENDING",
            }:
                continue
            formula = dict(raw.get("formula") or {})
            strategy = str(
                _first(raw, "strategy", "strategy_id", "strategy_name")
                or "ONBEKENDE_STRATEGIE"
            )
            timeframe = str(
                _first(raw, "timeframe", "entry_timeframe") or "n.b."
            )
            signal_time = _utc(
                _first(
                    raw,
                    "signal_timestamp",
                    "detected_at",
                    "observed_at",
                    "created_at",
                    "timestamp",
                )
            ) or observed_at
            trigger = _finite(_first(raw, "trigger", "entry_price"))
            current_price = _finite(_first(raw, "current_price", "price", "close"))
            semantic_pullback = status == "PULLBACK_PENDING" or any(
                token in strategy.upper()
                for token in ("PULLBACK", "RETEST", "DIP")
            )
            if trigger is not None and current_price is not None:
                entry_condition = (
                    "HIGH_AT_OR_ABOVE_TRIGGER"
                    if trigger >= current_price
                    else "LOW_AT_OR_BELOW_TRIGGER"
                )
                entry_condition_source = "ALERT_CURRENT_PRICE"
            elif semantic_pullback:
                entry_condition = "LOW_AT_OR_BELOW_TRIGGER"
                entry_condition_source = "STRATEGY_SEMANTIC_FALLBACK"
            else:
                entry_condition = "HIGH_AT_OR_ABOVE_TRIGGER"
                entry_condition_source = "LONG_BREAKOUT_FALLBACK"
            opportunity_id = str(
                _first(raw, "opportunity_id", "signal_id", "external_id", "id")
                or stable_hash(
                    [
                        "telegram-opportunity-fallback",
                        market,
                        strategy,
                        timeframe,
                        utc_iso(signal_time),
                    ],
                    length=32,
                )
            )
            selected.append(
                {
                    "opportunity_id": opportunity_id,
                    "signal_timestamp": utc_iso(signal_time),
                    "market": market,
                    "status": status,
                    "strategy": strategy,
                    "strategy_dna_hash": str(
                        _first(raw, "strategy_dna_hash", "dna_hash") or ""
                    ),
                    "timeframe": timeframe,
                    "current_price": current_price,
                    "trigger": trigger,
                    "stop": _finite(_first(raw, "stop", "stop_loss")),
                    "target_1": _finite(
                        _first(raw, "target_1", "take_profit_1", "tp1")
                    ),
                    "target_2": _finite(
                        _first(raw, "target_2", "take_profit_2", "tp2")
                    ),
                    "confidence": _finite(
                        _first(raw, "confidence", "confidence_pct")
                    ),
                    "reason": str(
                        _first(
                            raw,
                            "reason_not_yet_entered",
                            "trigger_reason",
                            "reason",
                        )
                        or "BEVESTIGING_NOG_NODIG"
                    ),
                    "live_authority_granted": bool(
                        _first(raw, "live_authority_granted")
                    ),
                    "entry_condition": entry_condition,
                    "entry_condition_source": entry_condition_source,
                    "expected_holding_period": str(
                        _first(raw, "expected_holding_period") or "24h"
                    ),
                    "evaluation_roundtrip_cost_bps": (
                        TELEGRAM_EVIDENCE_ROUNDTRIP_COST_BPS
                    ),
                    "return_15m": _finite(formula.get("return_15m")),
                    "return_1h": _finite(formula.get("return_1h")),
                    "relative_volume_20": _finite(
                        formula.get("relative_volume_20")
                    ),
                    "volume_robust_zscore": _finite(
                        formula.get("volume_robust_zscore")
                    ),
                    "extension_atr": _finite(formula.get("extension_atr")),
                }
            )
            if len(selected) >= max(1, int(maximum_rows)):
                break
        if not selected:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "NO_NEAR_ENTRY_OR_ACTIONABLE_OPPORTUNITY",
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        state_path = self.output_directory / "telegram_opportunity_state.json"
        previous = read_json(state_path) if state_path.is_file() else {}
        previous_rows = list(previous.get("rows") or [])

        def materially_changed() -> bool:
            if len(previous_rows) != len(selected):
                return True
            for old, new in zip(previous_rows, selected, strict=True):
                for key in (
                    "opportunity_id",
                    "market",
                    "status",
                    "strategy",
                    "strategy_dna_hash",
                    "timeframe",
                ):
                    if old.get(key) != new.get(key):
                        return True
                old_confidence = _finite(old.get("confidence"))
                new_confidence = _finite(new.get("confidence"))
                if (
                    old_confidence is None
                    or new_confidence is None
                    or abs(old_confidence - new_confidence) >= 5.0
                ):
                    return True
                for key in (
                    "trigger",
                    "stop",
                    "target_1",
                    "target_2",
                    "return_15m",
                    "return_1h",
                    "relative_volume_20",
                    "extension_atr",
                ):
                    old_value = _finite(old.get(key))
                    new_value = _finite(new.get(key))
                    if old_value is None or new_value is None:
                        if old_value != new_value:
                            return True
                        continue
                    denominator = max(abs(old_value), 1e-12)
                    if abs(new_value - old_value) / denominator >= 0.0025:
                        return True
            return False

        if not materially_changed():
            return {
                "delivery_status": "SKIPPED_DUPLICATE",
                "reason_code": "OPPORTUNITY_NOT_MATERIALLY_CHANGED",
                "orders_generated": 0,
                "orders_submitted": 0,
            }

        previous_by_market = {
            str(row.get("market") or ""): str(row.get("status") or "")
            for row in previous_rows
        }
        fresh_actionable_transition = any(
            row["status"] == "ACTIONABLE"
            and previous_by_market.get(row["market"]) != "ACTIONABLE"
            for row in selected
        )
        last_sent_at = previous.get("updated_at")
        if last_sent_at and not fresh_actionable_transition:
            try:
                last_sent = datetime.fromisoformat(
                    str(last_sent_at).replace("Z", "+00:00")
                ).astimezone(UTC)
            except ValueError:
                last_sent = None
            if (
                last_sent is not None
                and self.clock().astimezone(UTC) - last_sent
                < timedelta(hours=2)
            ):
                return {
                    "delivery_status": "SKIPPED_DUPLICATE",
                    "reason_code": "TACTICAL_UPDATE_COOLDOWN_ACTIVE",
                    "orders_generated": 0,
                    "orders_submitted": 0,
                }

        early_move_present = any(
            row["status"] in {"EARLY_MOMENTUM_ALERT", "PULLBACK_PENDING"}
            for row in selected
        )
        lines = [
            (
                "⚡ VROEGE BEWEGING GEDETECTEERD — NOG GEEN ORDER"
                if early_move_present
                else "🟡 NIEUWE TACTISCHE SETUP — NOG GEEN ORDER"
            ),
            "",
        ]
        for index, row in enumerate(selected, start=1):
            authority = (
                "live goedgekeurd"
                if row["live_authority_granted"]
                else "aparte DNA-goedkeuring vereist"
            )
            lines.extend(
                [
                    f"{index}. {row['market']} · {row['timeframe']} · {row['strategy']}",
                    f"Trigger: {_price(row['trigger'])}",
                    f"Stop: {_price(row['stop'])}",
                    f"TP1 / TP2: {_price(row['target_1'])} / {_price(row['target_2'])}",
                    f"Confidence: {_number(row['confidence'], 0)}%",
                    *(
                        [
                            "15m / 1h: "
                            f"{_number(100.0 * (row['return_15m'] or 0.0), 2)}% / "
                            f"{_number(100.0 * (row['return_1h'] or 0.0), 2)}%",
                            "RVOL20 / robuuste z: "
                            f"{_number(row['relative_volume_20'], 2)}x / "
                            f"{_number(row['volume_robust_zscore'], 2)}",
                            "Afstand boven EMA20: "
                            f"{_number(row['extension_atr'], 2)} ATR",
                        ]
                        if row["status"]
                        in {"EARLY_MOMENTUM_ALERT", "PULLBACK_PENDING"}
                        else []
                    ),
                    f"Status: {row['status']} — {authority}",
                    f"Wacht op: {_safe_reason(row['reason'])}",
                    "",
                ]
            )
        lines.extend(
            [
                "Alle niveaus komen uit gesloten candles.",
                "Dit bericht verleent geen execution authority en plaatst nul orders.",
            ]
        )
        message = "\n".join(lines)
        identity = stable_hash(
            {"event_type": "TACTICAL_OPPORTUNITY_UPDATE", "rows": selected},
            length=40,
        )
        with self._queue_lock() as acquired:
            if not acquired:
                result = {
                    "notification_id": identity,
                    "delivery_status": "RETRY_PENDING",
                    "reason_code": "TELEGRAM_QUEUE_BUSY",
                }
            else:
                result = self._enqueue_unlocked(
                    notification_id=identity,
                    signal_id=None,
                    message_type="TACTICAL_OPPORTUNITY_UPDATE",
                    message=message,
                )
                if result.get("delivery_status") == "PENDING":
                    evidence_record = self._append_opportunity_evidence(
                        notification_id=identity,
                        rows=selected,
                        recorded_at=observed_at,
                    )
                    result = {
                        **result,
                        "opportunity_evidence_record_hash": evidence_record[
                            "record_hash"
                        ],
                    }
        if result.get("delivery_status") == "PENDING":
            self.flush()
        atomic_write_json(
            state_path,
            {
                "schema_version": "telegram_opportunity_state_v1",
                "updated_at": utc_iso(self.clock()),
                "rows": selected,
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        return {
            **result,
            "opportunities_considered": len(selected),
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def notify_playbook_event(
        self,
        opportunity: Mapping[str, Any],
        *,
        live_authorized: bool = False,
    ) -> dict[str, Any]:
        """Send one material event-driven mover/lifecycle transition."""

        market = str(opportunity.get("market") or "").upper()
        state = str(opportunity.get("state") or "").upper()
        if market not in self.allowed_markets or state not in {
            "ORDER_SUBMITTED",
            "PARTIALLY_FILLED",
            "FILLED",
            "MANAGING",
            "EXITING",
            "CLOSED",
        }:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "PLAYBOOK_STATE_NOT_NOTIFIABLE",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        icons = {
            "ORDER_SUBMITTED": "📤",
            "PARTIALLY_FILLED": "⚠️",
            "FILLED": "✅",
            "MANAGING": "🟢",
            "EXITING": "🔴",
            "CLOSED": "✅",
        }
        title = {
            "ORDER_SUBMITTED": "ORDER SUBMITTED",
            "PARTIALLY_FILLED": "PARTIALLY FILLED",
            "FILLED": "POSITION FILLED",
            "MANAGING": "POSITION MANAGING",
            "EXITING": "POSITION EXITING",
            "CLOSED": "POSITION CLOSED",
        }[state]
        components = dict(opportunity.get("score_components") or {})
        confirmations = [
            key
            for key, value in (opportunity.get("confirmations") or {}).items()
            if value
        ]
        blockers = list(opportunity.get("hard_blockers") or [])
        authority = (
            "LIVE_CANARY"
            if live_authorized
            else "PAPER/SHADOW — live playbook-authority ontbreekt"
        )
        lines = [
            f"{icons[state]} {title} — {market}",
            "",
            f"Playbook: {opportunity.get('playbook_id') or 'n.b.'}",
            f"Score: {_number(opportunity.get('score'), 1)} · Tier {opportunity.get('tier') or 'n.b.'}",
            f"Entry: {_price(opportunity.get('entry_price'))}",
            f"Stop: {_price(opportunity.get('stop_loss'))}",
            "TP1 / TP2: "
            f"{_price(opportunity.get('take_profit_1'))} / "
            f"{_price(opportunity.get('take_profit_2'))}",
            f"Confirmaties ({len(confirmations)}): {', '.join(confirmations) or 'geen'}",
            (
                "Score T/V/F/OB/RS/E/M: "
                f"{_number(components.get('technical'), 1)}/"
                f"{_number(components.get('volume_acceleration'), 1)}/"
                f"{_number(components.get('executed_flow_cvd'), 1)}/"
                f"{_number(components.get('orderbook_liquidity'), 1)}/"
                f"{_number(components.get('relative_strength'), 1)}/"
                f"{_number(components.get('strategy_evidence'), 1)}/"
                f"{_number(components.get('macro'), 1)}"
            ),
            f"Macro sizing: {_number(opportunity.get('macro_risk_multiplier'), 2)}x",
            f"Authority: {authority}",
        ]
        if blockers:
            lines.append(f"Blockers: {', '.join(map(str, blockers))}")
        identity = stable_hash(
            {
                "event_type": "EVENT_DRIVEN_PLAYBOOK",
                "opportunity_id": opportunity.get("opportunity_id"),
                "state": state,
                "tier": opportunity.get("tier"),
                "blockers": blockers,
            },
            length=40,
        )
        result = self.enqueue(
            notification_id=identity,
            signal_id=str(opportunity.get("opportunity_id") or "") or None,
            message_type=f"PLAYBOOK_{state}",
            message="\n".join(lines),
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return {
            **result,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def notify_order_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = str(event_type).upper()
        execution_mode = str(
            _first(payload, "execution_mode", "execution", "mode") or ""
        ).upper()
        paper_only = bool(_first(payload, "paper_only")) or (
            "PAPER" in execution_mode
        )
        # Fail visibly toward PAPER whenever a caller supplies paper evidence,
        # even if it accidentally retained the legacy generic event name.
        # This prevents a simulated fill from looking like a Bitvavo fill.
        if paper_only and selected in {
            "ORDER_SUBMITTING",
            "ORDER_PARTIALLY_FILLED",
            "ORDER_FILLED",
            "ORDER_REJECTED",
        }:
            selected = f"PAPER_{selected}"
        if selected.startswith("PAPER_"):
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "INDIVIDUAL_PAPER_LIFECYCLE_SUMMARY_ONLY",
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        live_aliases = {
            "ORDER_SUBMITTING": "LIVE_ORDER_SUBMITTED",
            "ORDER_PARTIALLY_FILLED": "LIVE_ORDER_PARTIALLY_FILLED",
            "ORDER_FILLED": "LIVE_ORDER_FILLED",
            "ORDER_CANCELLED": "LIVE_ORDER_CANCELLED",
            "ORDER_REJECTED": "LIVE_ORDER_REJECTED",
        }
        selected = live_aliases.get(selected, selected)
        verification_source = str(
            _first(payload, "verification_source") or ""
        ).upper()
        venue_order_id = _first(
            payload,
            "order_public_id",
            "order_id",
        )
        if selected in {
            "LIVE_ORDER_PARTIALLY_FILLED",
            "LIVE_ORDER_FILLED",
        } and (
            verification_source not in VERIFIED_LIVE_FILL_SOURCES
            or not venue_order_id
        ):
            # Never let a local lifecycle guess, paper state or ambiguous
            # response claim that real money was filled at Bitvavo.
            selected = "LIVE_ORDER_STATUS_UNVERIFIED"
        market = str(_first(payload, "market") or "ONBEKEND").upper()
        labels = {
            "LIVE_ORDER_SUBMITTED": "📤 LIVE ORDER SUBMITTED",
            "LIVE_ORDER_PARTIALLY_FILLED": "⚠️ LIVE ORDER PARTIALLY FILLED",
            "LIVE_ORDER_FILLED": "✅ LIVE ORDER FILLED",
            "LIVE_ORDER_CANCELLED": "⚠️ LIVE ORDER CANCELLED",
            "LIVE_ORDER_REJECTED": "🚨 LIVE ORDER REJECTED",
            "LIVE_ORDER_STATUS_UNVERIFIED": "⚠️ LIVE ORDER STATUS UNVERIFIED",
            "PAPER_SIGNAL": "🧪 PAPER SIGNAL",
            "PAPER_ORDER": "🧪 PAPER ORDER",
            "PAPER_FILL": "🧪 PAPER FILL",
            "PAPER_ORDER_SUBMITTING": "🧪 PAPER ORDER SUBMITTING",
            "PAPER_ORDER_PARTIALLY_FILLED": "🧪 PAPER ORDER PARTIALLY FILLED",
            "PAPER_ORDER_FILLED": "🧪 PAPER ORDER FILLED",
            "PAPER_ORDER_REJECTED": "🧪 PAPER ORDER REJECTED",
        }
        label = labels.get(selected, f"⚠️ {selected.replace('_', ' ')}")
        order_id = venue_order_id or _first(
            payload,
            "intent_id",
            "idempotency_key",
        )
        public_order_id = "n.b."
        if order_id:
            order_text = str(order_id)
            public_order_id = (
                order_text
                if order_text.startswith(("ord_", "client_"))
                else f"ord_{sha256_text(order_text)[:20]}"
            )
        requested_quantity = _first(
            payload,
            "requested_quantity",
            "quantity",
        )
        filled_quantity = _first(
            payload,
            "filled_quantity",
            "filled_amount",
        )
        remaining_quantity = _first(
            payload,
            "remaining_quantity",
            "amount_remaining",
        )
        lines = [
            f"{label} — {market}",
            "",
            f"Side: {_first(payload, 'side') or 'n.b.'}",
            f"Type: {_first(payload, 'order_type', 'type') or 'n.b.'}",
            f"Aangevraagd: {_number(requested_quantity, 8)}",
            f"Gevuld: {_number(filled_quantity, 8)}",
            f"Resterend: {_number(remaining_quantity, 8)}",
            f"Gemiddelde fill: {_price(_first(payload, 'average_fill_price', 'fill_price', 'price'))}",
            f"Geïnvesteerd: {_price(_first(payload, 'invested_eur', 'notional_eur', 'notional'))}",
            f"Fee: {_price(_first(payload, 'fee_eur', 'fee'))}",
            f"Strategy: {_first(payload, 'strategy_id', 'strategy') or 'n.b.'}",
            f"Timeframe: {_first(payload, 'timeframe') or 'n.b.'}",
            f"Timestamp: {_first(payload, 'venue_timestamp', 'timestamp', 'observed_at') or 'n.b.'}",
            f"Status: {_first(payload, 'status') or 'n.b.'}",
            (
                "Execution: PAPER_ONLY — geen echte Bitvavo-order"
                if selected.startswith("PAPER_")
                else "Execution: LIVE/EXCHANGE — exchangebevestiging vereist"
            ),
            f"Order ID: {public_order_id}",
        ]
        if _first(payload, "stop_loss", "stop") is not None:
            lines.extend(
                [
                    f"Stop: {_price(_first(payload, 'stop_loss', 'stop'))}",
                    "TP1 / TP2: "
                    f"{_price(_first(payload, 'take_profit_1', 'tp1'))} / "
                    f"{_price(_first(payload, 'take_profit_2', 'tp2'))}",
                    f"Regime: {_first(payload, 'regime', 'macro_regime') or 'n.b.'}",
                    f"Orderflow: {_first(payload, 'orderflow_status') or 'n.b.'}",
                    f"Slippage: {_number(_first(payload, 'slippage_bps', 'estimated_slippage_bps'), 2)} bps",
                    f"Resterend risico: {_price(payload.get('remaining_risk_eur'))}",
                ]
            )
        if selected == "LIVE_ORDER_STATUS_UNVERIFIED":
            lines.append(
                "Geen fillclaim: private stream of geverifieerde REST-bevestiging ontbreekt."
            )
        if selected in {"LIVE_ORDER_REJECTED", "PAPER_ORDER_REJECTED"}:
            lines.extend(
                [
                    f"Reden: {_safe_reason(_first(payload, 'reason', 'reason_code'))}",
                    "Geen automatische retry uitgevoerd.",
                ]
            )
        identity = stable_hash(
            # The private account stream can report one venue fill first as
            # an ORDER(status=filled) and then as a FILL, and can replay both
            # after reconnect.  The venue order identity plus lifecycle phase
            # is the durable notification identity.  A partial and final fill
            # remain distinct because ``selected`` differs.
            [selected, market, public_order_id],
            length=40,
        )
        result = self.enqueue(
            notification_id=identity,
            signal_id=None,
            message_type=selected,
            message="\n".join(lines),
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_autopilot_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.settings.send_system_alerts:
            return {"delivery_status": "SKIPPED_FILTER"}
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
        best = summary.get("best_candidate") if isinstance(summary.get("best_candidate"), Mapping) else {}
        message = (
            "🤖 Research-autopilot afgerond\n\n"
            f"Nieuwe strategieën: {int(summary.get('new_strategies') or 0)}\n"
            f"Backtests: {int(summary.get('backtests') or summary.get('trials') or 0)}\n"
            f"Harde rejects: {int(summary.get('hard_rejects') or 0)}\n"
            f"Research candidates: {int(summary.get('research_candidates') or 0)}\n"
            f"Frozen shadow: {int(summary.get('frozen_shadow') or 0)}\n"
            f"Nieuwe manual-signal candidates: {int(summary.get('manual_signal_candidates') or 0)}\n\n"
            "Beste nieuwe kandidaat:\n"
            f"Naam: {best.get('name') or 'geen'}\n"
            f"PF: {_number(best.get('profit_factor'), 3)}\n"
            f"CAGR: {_percentage(best.get('cagr'))}\n"
            f"Max drawdown: {_percentage(best.get('maximum_drawdown'))}\n\n"
            "Automatische livepromoties: 0\nOrders geplaatst: 0"
        )
        cycle_id = _first(payload, "cycle_id", "run_id", "completed_at") or stable_hash(
            summary,
            length=24,
        )
        result = self.enqueue(
            notification_id=stable_hash(["autopilot-summary", cycle_id], length=40),
            signal_id=None,
            message_type="AUTOPILOT_SUMMARY",
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_paper_promotion_summary(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Send one deduplicated summary for newly paper-promoted DNA."""

        if not self.settings.send_system_alerts:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "SYSTEM_ALERTS_DISABLED",
            }
        safe_candidates = [
            self._safe(dict(candidate))
            for candidate in candidates
            if candidate.get("strategy_dna_hash")
        ]
        if not safe_candidates:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "NO_PAPER_PROMOTIONS",
            }
        timeframes: dict[str, int] = {}
        for candidate in safe_candidates:
            timeframe = clean_text(
                str(candidate.get("timeframe") or "onbekend"),
                maximum_length=16,
            )
            timeframes[timeframe] = timeframes.get(timeframe, 0) + 1
        best = max(
            safe_candidates,
            key=lambda candidate: (
                _finite((candidate.get("metrics") or {}).get("profit_factor"))
                or 0.0
            ),
        )
        best_metrics = (
            best.get("metrics")
            if isinstance(best.get("metrics"), Mapping)
            else {}
        )
        best_dna = clean_text(
            str(best.get("strategy_dna_hash") or ""),
            maximum_length=64,
        )
        timeframe_text = ", ".join(
            f"{timeframe}: {count}"
            for timeframe, count in sorted(timeframes.items())
        )
        message = (
            "🧪 NIEUWE PAPERSTRATEGIEËN\n\n"
            f"Aantal: {len(safe_candidates)}\n"
            f"Timeframes: {timeframe_text}\n"
            "Beste familie: "
            f"{clean_text(str(best.get('economic_hypothesis_family') or 'n.b.'), maximum_length=120)}\n"
            f"Beste PF: {_number(best_metrics.get('profit_factor'), 3)}\n"
            f"DNA: {best_dna[:16] or 'n.b.'}\n\n"
            "Modus: PAPER_ONLY\n"
            "Automatische livepromoties: 0\n"
            "Echte orders geplaatst: 0"
        )
        dna_values = sorted(
            clean_text(
                str(candidate.get("strategy_dna_hash")),
                maximum_length=64,
            )
            for candidate in safe_candidates
        )
        result = self.enqueue(
            notification_id=stable_hash(
                ["paper-promotion-summary", dna_values],
                length=40,
            ),
            signal_id=None,
            message_type="PAPER_PROMOTION_SUMMARY",
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_strategy_performance(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Report cumulative strategy state after a newly closed live trade."""

        if not self.settings.send_system_alerts:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "SYSTEM_ALERTS_DISABLED",
            }
        selected = self._safe(dict(payload))
        trade = (
            selected.get("last_closed_trade")
            if isinstance(selected.get("last_closed_trade"), Mapping)
            else {}
        )
        strategy_id = clean_text(
            str(selected.get("strategy_id") or "ONBEKENDE_STRATEGIE"),
            maximum_length=120,
        )
        market = clean_text(
            str(trade.get("market") or selected.get("market") or "ONBEKEND"),
            maximum_length=40,
        ).upper()
        message = (
            f"📊 STRATEGY TRADE CLOSED — {market}\n\n"
            f"Strategie: {strategy_id}\n"
            f"Netto P&L: {_price(trade.get('net_pnl_eur'))}\n"
            f"Fees: {_price(trade.get('fees_eur'))}\n"
            f"Slippage: {_number(trade.get('average_slippage_bps'), 2)} bps\n"
            f"Holding time: {_duration(trade.get('holding_seconds'))}\n"
            f"Strategy equity: {_price(selected.get('strategy_equity_eur'))}\n"
            f"Strategy drawdown: {_price(selected.get('maximum_drawdown_eur'))}\n"
            f"Gesloten trades: {int(selected.get('closed_trade_count') or 0)}\n"
            f"Authority: {clean_text(str(selected.get('authority_level') or 'n.b.'), maximum_length=60)}"
        )
        identity = stable_hash(
            [
                "strategy-performance",
                selected.get("strategy_dna"),
                selected.get("closed_trade_count"),
                trade.get("closed_at"),
                trade.get("net_pnl_eur"),
            ],
            length=40,
        )
        result = self.enqueue(
            notification_id=identity,
            signal_id=None,
            message_type="STRATEGY_PERFORMANCE",
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_daily_performance(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Send one deduplicated daily wallet and strategy summary."""

        if not self.settings.send_system_alerts:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "SYSTEM_ALERTS_DISABLED",
            }
        selected = self._safe(dict(payload))
        report_date = clean_text(
            str(selected.get("date_utc") or "onbekend"),
            maximum_length=20,
        )
        message = (
            f"📈 DAGELIJKS LIVE-OVERZICHT — {report_date}\n\n"
            f"Walletwaarde: {_price(selected.get('wallet_value_eur'))}\n"
            f"Dagresultaat: {_price(selected.get('daily_pnl_eur'))}\n"
            f"Gerealiseerd: {_price(selected.get('realised_pnl_eur'))}\n"
            f"Ongerealiseerd: {_price(selected.get('unrealised_pnl_eur'))}\n"
            f"Beste strategie: {clean_text(str(selected.get('best_strategy') or 'geen'), maximum_length=120)}\n"
            f"Slechtste strategie: {clean_text(str(selected.get('worst_strategy') or 'geen'), maximum_length=120)}\n"
            f"Fees: {_price(selected.get('fees_eur'))}\n"
            f"Strategy drawdown: {_price(selected.get('maximum_drawdown_eur'))}\n"
            f"Actief kapitaal: {_price(selected.get('active_capital_eur'))}\n"
            f"Cashreserve: {_price(selected.get('cash_reserve_eur'))}\n"
            f"Authority: {clean_text(str(selected.get('authority_status') or 'n.b.'), maximum_length=60)}\n"
            f"Open posities: {int(selected.get('open_positions') or 0)}\n"
            f"Echte orders vandaag: {int(selected.get('live_orders_today') or 0)}"
        )
        result = self.enqueue(
            notification_id=stable_hash(
                [
                    "daily-performance",
                    report_date,
                    selected.get("account_identity_hash"),
                ],
                length=40,
            ),
            signal_id=None,
            message_type="DAILY_PERFORMANCE",
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def notify_macro_summary(
        self,
        payload: Mapping[str, Any],
        *,
        slot: str,
    ) -> dict[str, Any]:
        """Send one concise, deduplicated 08:00/23:00 Amsterdam macro note."""

        if not self.settings.send_system_alerts:
            return {
                "delivery_status": "SKIPPED_FILTER",
                "reason_code": "SYSTEM_ALERTS_DISABLED",
            }
        selected = self._safe(dict(payload))
        message = (
            f"🌍 MACRO & MARKT — {slot} NL\n\n"
            f"Regime: {clean_text(str(selected.get('macro_regime') or 'UNKNOWN'), maximum_length=40)}\n"
            f"Structureel: {clean_text(str(selected.get('structural_regime') or 'UNKNOWN'), maximum_length=60)}\n"
            f"BTC 1D / 4H: {selected.get('btc_1d_trend') or 'n.b.'} / {selected.get('btc_4h_trend') or 'n.b.'}\n"
            f"BTC 24u: {_number(selected.get('btc_return_24h_pct'), 2)}%\n"
            f"Altcoinbreadth: {_number(selected.get('altcoin_breadth_pct'), 1)}%\n"
            f"BTC-dominance: {_number(selected.get('btc_dominance_pct'), 1)}%\n"
            f"Fear & Greed: {_number(selected.get('fear_greed'), 0)}\n"
            f"Risicomultiplier: {_number(selected.get('risk_multiplier'), 2)}x\n\n"
            f"Actief: {clean_text(str(selected.get('active_playbooks') or 'geen'), maximum_length=200)}\n"
            f"Geblokkeerd: {clean_text(str(selected.get('blocked_playbooks') or 'geen'), maximum_length=200)}\n"
            f"Verse entrykandidaten: {int(selected.get('entry_candidates') or 0)}\n"
            f"Open posities / orders: {int(selected.get('open_positions') or 0)} / {int(selected.get('open_orders') or 0)}\n"
            f"Beschikbare EUR: {_price(selected.get('eur_available'))}\n"
            f"Live: {clean_text(str(selected.get('live_status') or 'n.b.'), maximum_length=40)}"
        )
        result = self.enqueue(
            notification_id=stable_hash(
                ["scheduled-macro", slot, selected.get("observed_at")],
                length=40,
            ),
            signal_id=None,
            message_type="SCHEDULED_MACRO_SUMMARY",
            message=message,
        )
        if result.get("delivery_status") == "PENDING":
            self.flush()
        return result

    def send_test_message(self) -> dict[str, Any]:
        message = (
            "✅ Crypto bot Telegramkoppeling werkt.\n"
            "Modus: SIGNALS_ONLY\n"
            "Orders geplaatst: 0"
        )
        # A timestamp makes explicit operator tests independently deliverable.
        identity = stable_hash(
            ["telegram-test", self.clock().isoformat()],
            length=40,
        )
        queued = self.enqueue(
            notification_id=identity,
            signal_id=None,
            message_type="SYSTEM_TEST",
            message=message,
        )
        delivery = self.flush() if queued.get("delivery_status") == "PENDING" else {}
        state = self._delivery_state().get(identity, queued)
        return {
            "status": state.get("delivery_status"),
            "notification_id": identity,
            "delivery": delivery,
            "orders_generated": 0,
            "orders_submitted": 0,
        }

    def status(self, *, write: bool = True) -> dict[str, Any]:
        history = self._history()
        state = self._delivery_state()
        queue_size = sum(
            row.get("delivery_status") in QUEUE_STATUSES for row in state.values()
        )
        sent = [row for row in history if row.get("delivery_status") == "SENT"]
        failures = [
            row for row in history if row.get("delivery_status") == "FAILED_FINAL"
        ]
        payload = {
            "status": self.enabled_status,
            "enabled": self.settings.notifications_enabled,
            "configured": self.settings.configured,
            "dry_run": self.settings.dry_run,
            "signals_enabled": self.settings.send_signals,
            "watchlist_enabled": self.settings.send_watchlist,
            "exits_enabled": self.settings.send_exits,
            "risk_alerts_enabled": self.settings.send_risk_alerts,
            "system_alerts_enabled": self.settings.send_system_alerts,
            "chat_identity_hash": self.chat_identity_hash,
            "active_queue_size": queue_size,
            "sent_count": len(sent),
            "duplicates_skipped": sum(
                row.get("delivery_status") == "SKIPPED_DUPLICATE"
                for row in history
            ),
            "signals_filtered": sum(
                row.get("delivery_status") == "SKIPPED_FILTER"
                for row in history
            ),
            "retry_events": sum(
                row.get("delivery_status") == "RETRY_PENDING"
                for row in history
            ),
            "failed_final_count": len(failures),
            "last_successful_send": sent[-1].get("timestamp") if sent else None,
            "last_error": failures[-1].get("reason_code") if failures else None,
            "rate_limit": self._rate_limit(),
            "ledger_path": str(self.ledger_path),
            "failure_ledger_path": str(self.failures_path),
            "opportunity_evidence_path": str(self.opportunity_evidence_path),
            "opportunity_evidence_event_count": len(
                _read_jsonl(self.opportunity_evidence_path)
            ),
            "preview_path": str(self.preview_path),
            "orders_generated": 0,
            "orders_submitted": 0,
            "secrets_redacted": True,
        }
        if write:
            atomic_write_json(self.status_path, payload)
        return payload

    def health(self, *, probe: bool = True) -> dict[str, Any]:
        previous = read_json(self.health_path) if self.health_path.is_file() else {}
        reachable: bool | None = None
        api_status: int | None = None
        error_code: str | None = None
        status = self.enabled_status
        if status == "ENABLED" and probe:
            try:
                response = self.transport(
                    "GET",
                    self._url("getMe"),
                    None,
                    self.settings.request_timeout_seconds,
                )
                api_status = response.status
                reachable = response.status == 200 and response.payload.get("ok", True)
                status = "HEALTHY" if reachable else "UNREACHABLE"
                if not reachable:
                    error_code = f"TELEGRAM_HTTP_{response.status}"
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                reachable = False
                status = "UNREACHABLE"
                error_code = f"TELEGRAM_{type(exc).__name__.upper()}"
            except Exception as exc:
                reachable = False
                status = "UNREACHABLE"
                error_code = f"TELEGRAM_{type(exc).__name__.upper()}"
        elif status == "DRY_RUN":
            reachable = None
        current = self.status(write=True)
        payload = {
            "status": status,
            "checked_at": utc_iso(self.clock()),
            "enabled": self.settings.notifications_enabled,
            "token_present": bool(self.settings.bot_token),
            "chat_id_present": bool(self.settings.chat_id),
            "missing_configuration": self.settings.missing_configuration,
            "api_reachable": reachable,
            "telegram_http_status": api_status,
            "last_successful_send": current["last_successful_send"],
            "last_error": error_code or current["last_error"],
            "active_queue_size": current["active_queue_size"],
            "rate_limit": current["rate_limit"],
            "chat_identity_hash": self.chat_identity_hash,
            "secrets_redacted": True,
            "orders_generated": 0,
            "orders_submitted": 0,
        }
        atomic_write_json(self.health_path, payload)
        if (
            previous.get("status") == "UNREACHABLE"
            and status == "HEALTHY"
            and self.settings.send_system_alerts
        ):
            self.notify_system_event(
                "TELEGRAM_CONNECTION_RESTORED",
                {"status": "HEALTHY"},
            )
        return payload
