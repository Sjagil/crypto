"""Persistent drawdown state machine; RiskManager remains final authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common import append_jsonl, atomic_write_json, read_json, utc_now


class DrawdownState(StrEnum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    REDUCE_RISK = "REDUCE_RISK"
    BLOCK_NEW_ENTRIES = "BLOCK_NEW_ENTRIES"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass(frozen=True)
class DrawdownThresholds:
    warning: float = 0.04
    reduce_risk: float = 0.07
    block_new_entries: float = 0.10
    kill_switch: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.warning,
            self.reduce_risk,
            self.block_new_entries,
            self.kill_switch,
        )
        if not (0 < values[0] < values[1] < values[2] < values[3] < 1):
            raise ValueError("drawdown thresholds must be strictly increasing")


class DrawdownProtection:
    severe_states = {DrawdownState.BLOCK_NEW_ENTRIES, DrawdownState.KILL_SWITCH}

    def __init__(
        self,
        *,
        state_path: Path | str | None = None,
        audit_path: Path | str | None = None,
        thresholds: DrawdownThresholds | None = None,
        cooldown: timedelta = timedelta(hours=24),
    ) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.audit_path = Path(audit_path) if audit_path else None
        self.thresholds = thresholds or DrawdownThresholds()
        self.cooldown = cooldown
        self.state = DrawdownState.NORMAL
        self.reason_codes: tuple[str, ...] = ("WITHIN_LIMITS",)
        self.activated_at: datetime | None = None
        self.cooldown_until: datetime | None = None
        self.manual_reset_required = False
        self.consecutive_losses = 0
        if self.state_path and self.state_path.is_file():
            self._load()

    @staticmethod
    def drawdown(equity: pd.Series) -> pd.Series:
        selected = equity.astype(float)
        peak = selected.cummax()
        return (selected / peak - 1).fillna(0.0)

    @classmethod
    def window_drawdown(cls, equity: pd.Series, window: str) -> float:
        if equity.empty:
            return 0.0
        selected = equity.sort_index()
        latest = pd.Timestamp(selected.index[-1])
        durations = {
            "1D": timedelta(days=1),
            "24h": timedelta(hours=24),
            "7D": timedelta(days=7),
        }
        if window not in durations:
            raise ValueError("unsupported drawdown window")
        recent = selected.loc[selected.index >= latest - durations[window]]
        if recent.empty or recent.max() <= 0:
            return 0.0
        return max(0.0, 1.0 - float(recent.iloc[-1]) / float(recent.max()))

    @staticmethod
    def time_under_water(equity: pd.Series) -> timedelta:
        if equity.empty:
            return timedelta(0)
        drawdown = DrawdownProtection.drawdown(equity)
        underwater = drawdown < 0
        if not bool(underwater.iloc[-1]):
            return timedelta(0)
        last_peak_positions = [index for index, value in enumerate(underwater) if not value]
        start_position = last_peak_positions[-1] if last_peak_positions else 0
        return equity.index[-1].to_pydatetime() - equity.index[start_position].to_pydatetime()

    @staticmethod
    def consecutive_loss_count(pnl: pd.Series) -> int:
        count = 0
        for value in reversed(pnl.dropna().tolist()):
            if value < 0:
                count += 1
            else:
                break
        return count

    def evaluate(
        self,
        *,
        portfolio_equity: pd.Series,
        trade_pnl: pd.Series | None = None,
        strategy_equity: dict[str, pd.Series] | None = None,
        symbol_equity: dict[str, pd.Series] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        selected_now = now or utc_now()
        equity = portfolio_equity.sort_index()
        if equity.empty:
            return self._transition(
                DrawdownState.BLOCK_NEW_ENTRIES,
                ("MISSING_EQUITY_HISTORY",),
                selected_now,
            )
        metrics = {
            "intraday_drawdown": self.window_drawdown(equity, "1D"),
            "rolling_24h_drawdown": self.window_drawdown(equity, "24h"),
            "rolling_7d_drawdown": self.window_drawdown(equity, "7D"),
            "peak_to_trough_drawdown": abs(float(self.drawdown(equity).min())),
            "strategy_drawdown": {
                key: abs(float(self.drawdown(value).min()))
                for key, value in (strategy_equity or {}).items()
                if not value.empty
            },
            "symbol_drawdown": {
                key: abs(float(self.drawdown(value).min()))
                for key, value in (symbol_equity or {}).items()
                if not value.empty
            },
            "time_under_water_seconds": self.time_under_water(equity).total_seconds(),
        }
        self.consecutive_losses = self.consecutive_loss_count(
            trade_pnl if trade_pnl is not None else pd.Series(dtype=float)
        )
        metrics["consecutive_losses"] = self.consecutive_losses
        maximum = max(
            metrics["intraday_drawdown"],
            metrics["rolling_24h_drawdown"],
            metrics["rolling_7d_drawdown"],
            metrics["peak_to_trough_drawdown"],
            max(metrics["strategy_drawdown"].values(), default=0.0),
            max(metrics["symbol_drawdown"].values(), default=0.0),
        )
        desired, reason = self._desired_state(maximum)
        if self.manual_reset_required and self.state in self.severe_states:
            desired = self.state
            reason = "MANUAL_RESET_REQUIRED"
        elif (
            self._rank(desired) < self._rank(self.state)
            and self.cooldown_until
            and selected_now < self.cooldown_until
        ):
            desired = self.state
            reason = "COOLDOWN_ACTIVE"
        status = self._transition(desired, (reason,), selected_now)
        status["metrics"] = metrics
        status["recovery_status"] = (
            "RECOVERED"
            if maximum < self.thresholds.warning
            else "UNDER_WATER"
        )
        return status

    def _desired_state(self, maximum: float) -> tuple[DrawdownState, str]:
        if maximum >= self.thresholds.kill_switch:
            return DrawdownState.KILL_SWITCH, "DRAWDOWN_KILL_SWITCH"
        if maximum >= self.thresholds.block_new_entries:
            return DrawdownState.BLOCK_NEW_ENTRIES, "DRAWDOWN_BLOCK_ENTRIES"
        if maximum >= self.thresholds.reduce_risk:
            return DrawdownState.REDUCE_RISK, "DRAWDOWN_REDUCE_RISK"
        if maximum >= self.thresholds.warning:
            return DrawdownState.WARNING, "DRAWDOWN_WARNING"
        return DrawdownState.NORMAL, "WITHIN_LIMITS"

    @staticmethod
    def _rank(state: DrawdownState) -> int:
        return list(DrawdownState).index(state)

    def _transition(
        self,
        desired: DrawdownState,
        reasons: tuple[str, ...],
        now: datetime,
    ) -> dict[str, Any]:
        previous = self.state
        if desired != previous:
            self.state = desired
            self.reason_codes = reasons
            self.activated_at = now
            if self._rank(desired) > self._rank(DrawdownState.NORMAL):
                self.cooldown_until = now + self.cooldown
            if desired in self.severe_states:
                self.manual_reset_required = True
            self._audit(
                {
                    "event": "STATE_TRANSITION",
                    "previous_state": previous.value,
                    "state": desired.value,
                    "reason_codes": reasons,
                    "timestamp": now.isoformat(),
                }
            )
            self._persist()
        return self.status(now=now)

    def manual_reset(self, *, reason: str, now: datetime | None = None) -> None:
        selected_reason = reason.strip()
        if not selected_reason:
            raise ValueError("manual reset requires an explicit reason")
        selected_now = now or utc_now()
        previous = self.state
        self.state = DrawdownState.NORMAL
        self.reason_codes = ("MANUAL_RESET",)
        self.manual_reset_required = False
        self.cooldown_until = None
        self.activated_at = selected_now
        self._audit(
            {
                "event": "MANUAL_RESET",
                "previous_state": previous.value,
                "state": self.state.value,
                "reason": selected_reason,
                "timestamp": selected_now.isoformat(),
            }
        )
        self._persist()

    def status(self, *, now: datetime | None = None) -> dict[str, Any]:
        selected_now = now or utc_now()
        return {
            "state": self.state.value,
            "reason_codes": list(self.reason_codes),
            "new_entries_allowed": self.state
            not in {DrawdownState.BLOCK_NEW_ENTRIES, DrawdownState.KILL_SWITCH},
            "risk_multiplier": {
                DrawdownState.NORMAL: 1.0,
                DrawdownState.WARNING: 0.75,
                DrawdownState.REDUCE_RISK: 0.5,
                DrawdownState.BLOCK_NEW_ENTRIES: 0.0,
                DrawdownState.KILL_SWITCH: 0.0,
            }[self.state],
            "manual_reset_required": self.manual_reset_required,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "checked_at": selected_now.isoformat(),
        }

    def _persist(self) -> None:
        if self.state_path:
            atomic_write_json(self.state_path, self.status())

    def _audit(self, event: dict[str, Any]) -> None:
        if self.audit_path:
            append_jsonl(self.audit_path, event)

    def _load(self) -> None:
        try:
            payload = read_json(self.state_path)
            self.state = DrawdownState(payload["state"])
            self.reason_codes = tuple(payload.get("reason_codes", ["PERSISTED_STATE"]))
            self.manual_reset_required = bool(payload.get("manual_reset_required", False))
            if payload.get("activated_at"):
                self.activated_at = datetime.fromisoformat(payload["activated_at"])
            if payload.get("cooldown_until"):
                self.cooldown_until = datetime.fromisoformat(payload["cooldown_until"])
        except (OSError, ValueError, KeyError, TypeError):
            self.state = DrawdownState.KILL_SWITCH
            self.reason_codes = ("UNREADABLE_DRAWDOWN_STATE",)
            self.manual_reset_required = True


__all__ = ["DrawdownProtection", "DrawdownState", "DrawdownThresholds"]
