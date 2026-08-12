"""Central active-swing execution policy and durable weekly trade budget."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping

from config.settings import (
    ACTIVE_SWING_TIMEFRAMES,
    DISABLED_EXECUTION_TIMEFRAMES,
    TIMEFRAME_SECONDS,
    Settings,
    normalize_timeframe,
)
from utils.common import append_jsonl, atomic_write_json, utc_iso

WEEKLY_ENTRY_CAP = 20
WEEKLY_WARNING_START = 15
WEEKLY_EXCEPTIONAL_START = 18
NORMAL_EXIT_COOLDOWN_BARS = {
    "15m": 12,
    "1h": 6,
    "2h": 4,
    "4h": 2,
    "1d": 1,
    "1W": 1,
}
STOP_EXIT_COOLDOWN_BARS = {
    "15m": 24,
    "1h": 12,
    "2h": 8,
    "4h": 4,
    "1d": 3,
    "1W": 2,
}


def _parse_utc(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        selected = value
    elif value:
        try:
            selected = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
        except ValueError:
            return None
    else:
        return None
    if selected.tzinfo is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def execution_timeframe_allowed(timeframe: str) -> bool:
    """Return whether a strategy timeframe may consume execution authority."""

    try:
        selected = normalize_timeframe(timeframe)
    except ValueError:
        return False
    return (
        selected in ACTIVE_SWING_TIMEFRAMES
        and selected not in DISABLED_EXECUTION_TIMEFRAMES
    )


def material_position_limit(account_equity_eur: Decimal | float | str) -> int:
    """Capital-tier position cap, inclusive of pre-existing wallet inventory."""

    equity = Decimal(str(account_equity_eur or "0"))
    if equity < Decimal("1000"):
        return 2
    if equity < Decimal("10000"):
        return 3
    if equity < Decimal("50000"):
        return 5
    if equity < Decimal("100000"):
        return 7
    if equity < Decimal("250000"):
        return 10
    return 12


def weekly_gate_multiplier(entry_count: int) -> Decimal | None:
    if entry_count >= WEEKLY_ENTRY_CAP:
        return None
    if entry_count >= WEEKLY_EXCEPTIONAL_START:
        return Decimal("1.35")
    if entry_count >= WEEKLY_WARNING_START:
        return Decimal("1.20")
    if entry_count >= 12:
        return Decimal("1.10")
    return Decimal("1.00")


class WeeklyTradeBudgetManager:
    """Append-only, duplicate-safe ISO-week entry accounting."""

    def __init__(self, settings: Settings) -> None:
        directory = settings.paths.output_dir / "operations"
        directory.mkdir(parents=True, exist_ok=True)
        self.ledger_path = directory / "weekly_trade_budget.jsonl"
        self.status_path = directory / "weekly_trade_budget.json"
        self.ledger_path.touch(exist_ok=True)

    @staticmethod
    def _week_key(moment: datetime) -> str:
        selected = moment.astimezone(UTC)
        iso = selected.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not self.ledger_path.is_file():
            return events
        for raw in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(value, Mapping):
                events.append(dict(value))
        return events

    def status(self, *, observed_at: datetime | None = None) -> dict[str, Any]:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        week = self._week_key(now)
        events = [row for row in self._events() if row.get("iso_week") == week]
        entries = [row for row in events if row.get("event") == "ENTRY_ACCEPTED"]
        exits = [row for row in events if row.get("event") == "EXIT"]
        rejections = [
            row for row in events if row.get("event") == "ENTRY_REJECTED"
        ]
        entry_ids = {
            str(row.get("entry_identity") or "")
            for row in entries
            if row.get("entry_identity")
        }
        unique_entries = [
            row
            for row in entries
            if str(row.get("entry_identity") or "") in entry_ids
        ]
        deduplicated: dict[str, dict[str, Any]] = {}
        for row in unique_entries:
            deduplicated.setdefault(str(row["entry_identity"]), row)
        unique_entries = list(deduplicated.values())
        count = len(unique_entries)
        multiplier = weekly_gate_multiplier(count)
        payload = {
            "schema_version": "weekly_trade_budget_v1",
            "generated_at": utc_iso(),
            "iso_week": week,
            "hard_cap": WEEKLY_ENTRY_CAP,
            "soft_warning_start": WEEKLY_WARNING_START,
            "exceptional_only_start": WEEKLY_EXCEPTIONAL_START,
            "new_entries": count,
            "remaining_entry_budget": max(0, WEEKLY_ENTRY_CAP - count),
            "new_entries_blocked": multiplier is None,
            "weekly_gate_multiplier": (
                str(multiplier) if multiplier is not None else None
            ),
            "warning": count >= WEEKLY_WARNING_START,
            "entries_by_strategy": dict(
                Counter(str(row.get("strategy_id") or "UNKNOWN") for row in unique_entries)
            ),
            "entries_by_timeframe": dict(
                Counter(str(row.get("timeframe") or "UNKNOWN") for row in unique_entries)
            ),
            "entries_by_asset": dict(
                Counter(str(row.get("market") or "UNKNOWN") for row in unique_entries)
            ),
            "entries_by_regime": dict(
                Counter(str(row.get("regime") or "UNKNOWN") for row in unique_entries)
            ),
            "exits": len(exits),
            "stop_outs": sum(
                str(row.get("reason") or "").upper().startswith("STOP")
                for row in exits
            ),
            "duplicate_entries_suppressed": max(0, len(entries) - count),
            "rejected_signals": len(rejections),
            "duplicate_signals": sum(
                row.get("reason_code") == "DUPLICATE_SETUP_SUPPRESSED"
                for row in rejections
            ),
            "cooldown_blocks": sum(
                str(row.get("reason_code") or "").startswith("COOLDOWN_")
                for row in rejections
            ),
            "protective_exits_always_allowed": True,
            "ledger": str(self.ledger_path),
        }
        atomic_write_json(self.status_path, payload)
        return payload

    def assess_entry(
        self,
        *,
        score: Decimal | float | str,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        status = self.status(observed_at=observed_at)
        multiplier_raw = status["weekly_gate_multiplier"]
        if multiplier_raw is None:
            return {
                "approved": False,
                "reason_code": "WEEKLY_ENTRY_CAP_REACHED",
                "required_score": None,
                "status": status,
            }
        normalized_score = Decimal(str(score or "0"))
        if normalized_score <= 1:
            normalized_score *= Decimal("100")
        required = Decimal("50") * Decimal(multiplier_raw)
        return {
            "approved": normalized_score >= required,
            "reason_code": (
                "WEEKLY_BUDGET_AVAILABLE"
                if normalized_score >= required
                else "WEEKLY_BUDGET_SCORE_TOO_LOW"
            ),
            "required_score": str(required),
            "observed_score": str(normalized_score),
            "status": status,
        }

    def record_entry(
        self,
        *,
        entry_identity: str,
        strategy_id: str,
        strategy_dna_hash: str,
        market: str,
        timeframe: str,
        regime: str | None,
        order_status: str | None,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        if any(
            row.get("event") == "ENTRY_ACCEPTED"
            and row.get("entry_identity") == entry_identity
            for row in self._events()
        ):
            status = self.status(observed_at=now)
            return {
                "recorded": False,
                "reason_code": "DUPLICATE_ENTRY_IDENTITY",
                "status": status,
            }
        append_jsonl(
            self.ledger_path,
            {
                "event": "ENTRY_ACCEPTED",
                "recorded_at": now.isoformat(),
                "iso_week": self._week_key(now),
                "entry_identity": entry_identity,
                "strategy_id": strategy_id,
                "strategy_dna_hash": strategy_dna_hash,
                "market": market,
                "timeframe": normalize_timeframe(timeframe),
                "regime": regime or "UNKNOWN",
                "order_status": order_status,
            },
        )
        return {
            "recorded": True,
            "reason_code": "ENTRY_COUNTED_ONCE",
            "status": self.status(observed_at=now),
        }

    def record_rejection(
        self,
        *,
        strategy_id: str,
        market: str,
        timeframe: str,
        reason_code: str,
        signal_id: str | None = None,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        if signal_id and any(
            row.get("event") == "ENTRY_REJECTED"
            and row.get("signal_id") == signal_id
            and row.get("reason_code") == reason_code
            for row in self._events()
        ):
            return self.status(observed_at=now)
        append_jsonl(
            self.ledger_path,
            {
                "event": "ENTRY_REJECTED",
                "recorded_at": now.isoformat(),
                "iso_week": self._week_key(now),
                "strategy_id": strategy_id,
                "market": market,
                "timeframe": normalize_timeframe(timeframe),
                "reason_code": reason_code,
                "signal_id": signal_id,
            },
        )
        return self.status(observed_at=now)

    def record_exit(
        self,
        *,
        exit_identity: str,
        market: str,
        strategy_id: str,
        reason: str,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        if not any(
            row.get("event") == "EXIT"
            and row.get("exit_identity") == exit_identity
            for row in self._events()
        ):
            append_jsonl(
                self.ledger_path,
                {
                    "event": "EXIT",
                    "recorded_at": now.isoformat(),
                    "iso_week": self._week_key(now),
                    "exit_identity": exit_identity,
                    "strategy_id": strategy_id,
                    "market": market,
                    "reason": reason,
                },
            )
        return self.status(observed_at=now)


class SwingCooldownManager:
    """Restart-safe, closed-candle cooldown and same-setup suppression."""

    def __init__(self, settings: Settings) -> None:
        directory = settings.paths.output_dir / "operations"
        directory.mkdir(parents=True, exist_ok=True)
        self.ledger_path = directory / "swing_cooldowns.jsonl"
        self.status_path = directory / "swing_cooldowns.json"
        self.ledger_path.touch(exist_ok=True)

    def _events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for raw in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, Mapping):
                events.append(dict(payload))
        return events

    @staticmethod
    def _matches(
        row: Mapping[str, Any],
        *,
        strategy_dna_hash: str,
        market: str,
        timeframe: str,
    ) -> bool:
        return (
            row.get("strategy_dna_hash") == strategy_dna_hash
            and row.get("market") == market
            and row.get("timeframe") == timeframe
        )

    def assess_entry(
        self,
        *,
        strategy_id: str,
        strategy_dna_hash: str,
        market: str,
        timeframe: str,
        signal_candle_at: datetime | str,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        selected_timeframe = normalize_timeframe(timeframe)
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        candle_at = _parse_utc(signal_candle_at)
        if candle_at is None:
            return {
                "approved": False,
                "reason_code": "COOLDOWN_SIGNAL_CANDLE_INVALID",
            }
        matching = [
            row
            for row in self._events()
            if self._matches(
                row,
                strategy_dna_hash=strategy_dna_hash,
                market=market,
                timeframe=selected_timeframe,
            )
        ]
        if any(
            row.get("event") == "ENTRY"
            and row.get("signal_candle_at") == candle_at.isoformat()
            for row in matching
        ):
            return {
                "approved": False,
                "reason_code": "DUPLICATE_SETUP_SUPPRESSED",
            }
        exits = [row for row in matching if row.get("event") == "EXIT"]
        latest_exit = max(
            exits,
            key=lambda row: _parse_utc(row.get("recorded_at"))
            or datetime.min.replace(tzinfo=UTC),
            default=None,
        )
        if latest_exit is None:
            return {
                "approved": True,
                "reason_code": "COOLDOWN_CLEAR",
                "strategy_id": strategy_id,
            }
        exited_at = _parse_utc(latest_exit.get("recorded_at"))
        if exited_at is None:
            return {
                "approved": False,
                "reason_code": "COOLDOWN_LEDGER_TIMESTAMP_INVALID",
            }
        reason = str(latest_exit.get("reason") or "").upper()
        stopped = "STOP" in reason or "REGIME" in reason
        bars = (
            STOP_EXIT_COOLDOWN_BARS
            if stopped
            else NORMAL_EXIT_COOLDOWN_BARS
        )[selected_timeframe]
        eligible_at = exited_at.timestamp() + (
            bars * TIMEFRAME_SECONDS[selected_timeframe]
        )
        approved = now.timestamp() >= eligible_at
        payload = {
            "approved": approved,
            "reason_code": (
                "COOLDOWN_CLEAR"
                if approved
                else "COOLDOWN_ACTIVE_AFTER_STOP"
                if stopped
                else "COOLDOWN_ACTIVE_AFTER_EXIT"
            ),
            "required_closed_bars": bars,
            "eligible_at": datetime.fromtimestamp(
                eligible_at,
                tz=UTC,
            ).isoformat(),
        }
        atomic_write_json(
            self.status_path,
            {
                "schema_version": "swing_cooldowns_v1",
                "generated_at": utc_iso(),
                "strategy_id": strategy_id,
                "strategy_dna_hash": strategy_dna_hash,
                "market": market,
                "timeframe": selected_timeframe,
                **payload,
            },
        )
        return payload

    def record_entry(
        self,
        *,
        strategy_id: str,
        strategy_dna_hash: str,
        market: str,
        timeframe: str,
        signal_candle_at: datetime | str,
        observed_at: datetime | None = None,
    ) -> None:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        candle_at = _parse_utc(signal_candle_at)
        if candle_at is None:
            raise ValueError("signal_candle_at must be a valid timestamp")
        append_jsonl(
            self.ledger_path,
            {
                "event": "ENTRY",
                "recorded_at": now.isoformat(),
                "strategy_id": strategy_id,
                "strategy_dna_hash": strategy_dna_hash,
                "market": market,
                "timeframe": normalize_timeframe(timeframe),
                "signal_candle_at": candle_at.isoformat(),
            },
        )

    def record_exit(
        self,
        *,
        strategy_id: str,
        strategy_dna_hash: str,
        market: str,
        timeframe: str,
        reason: str,
        observed_at: datetime | None = None,
    ) -> None:
        now = (observed_at or datetime.now(UTC)).astimezone(UTC)
        append_jsonl(
            self.ledger_path,
            {
                "event": "EXIT",
                "recorded_at": now.isoformat(),
                "strategy_id": strategy_id,
                "strategy_dna_hash": strategy_dna_hash,
                "market": market,
                "timeframe": normalize_timeframe(timeframe),
                "reason": reason,
            },
        )


def write_position_limit_status(
    settings: Settings,
    *,
    account_equity_eur: Decimal | float | str,
    material_positions: list[str] | tuple[str, ...],
    managed_positions: list[str] | tuple[str, ...] | None = None,
    maximum_managed_positions: int | None = None,
) -> dict[str, Any]:
    wallet_limit = material_position_limit(account_equity_eur)
    positions = sorted(set(str(value) for value in material_positions if value))
    managed = sorted(
        set(
            str(value)
            for value in (
                material_positions
                if managed_positions is None
                else managed_positions
            )
            if value
        )
    )
    managed_limit = (
        wallet_limit
        if maximum_managed_positions is None
        else max(0, int(maximum_managed_positions))
    )
    payload = {
        "schema_version": "position_limit_status_v2",
        "generated_at": utc_iso(),
        "account_equity_eur": str(account_equity_eur),
        # Keep the historical fields for consumers, but make their wallet-wide
        # meaning explicit.  Grandfathered inventory informs portfolio heat;
        # it does not silently consume an operator-approved managed slot.
        "material_positions": positions,
        "material_position_count": len(positions),
        "wallet_material_positions": positions,
        "wallet_material_position_count": len(positions),
        "wallet_advisory_position_limit": wallet_limit,
        "managed_positions": managed,
        "managed_position_count": len(managed),
        "maximum_positions": managed_limit,
        "maximum_managed_positions": managed_limit,
        "remaining_slots": max(0, managed_limit - len(managed)),
        "new_position_allowed": len(managed) < managed_limit,
        "wallet_wide": True,
        "grandfathered_inventory_counts_toward_portfolio_heat": True,
        "grandfathered_inventory_consumes_managed_slot": False,
    }
    path = settings.paths.output_dir / "operations" / "position_limit_status.json"
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "ACTIVE_SWING_TIMEFRAMES",
    "DISABLED_EXECUTION_TIMEFRAMES",
    "WEEKLY_ENTRY_CAP",
    "SwingCooldownManager",
    "WeeklyTradeBudgetManager",
    "execution_timeframe_allowed",
    "material_position_limit",
    "weekly_gate_multiplier",
    "write_position_limit_status",
]
