"""Local dynamic UI backed by atomic production snapshots.

The HTTP service is deliberately isolated from exchange execution.  It can
read snapshots and invoke only existing pause/resume/reconcile/kill-switch
services.  It never constructs an order intent.
"""

from __future__ import annotations

import asyncio
import ctypes
import html
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from config.settings import Settings
from research.features import confirmed_fractals
from risk.risk_manager import KillSwitch
from utils.common import (
    append_jsonl,
    atomic_write_json,
    read_json,
    stable_hash,
    utc_iso,
)

HOST = "127.0.0.1"
PORT = 8765
SCHEMA_VERSION = "local_trading_ui_v1"


def _paths(settings: Settings) -> dict[str, Path]:
    root = settings.paths.output_dir / "ui"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "lock": root / "ui.lock",
        "health": root / "ui_health.json",
        "audit": root / "ui_controls.jsonl",
        "log": settings.paths.logs_dir / "local_trading_ui.log",
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            0x1000,
            False,
            pid,
        )
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _safe_read(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _tail_jsonl(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _process_projection(path: Path) -> dict[str, Any]:
    raw = _safe_read(path)
    pid = int(raw.get("pid") or 0)
    return {
        "pid": pid or None,
        "running": _pid_alive(pid),
        "acquired_at": raw.get("acquired_at"),
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_daily_pnl_calendar(
    settings: Settings,
    *,
    limit: int = 366,
) -> dict[str, Any]:
    """Project account-wide daily mark-to-market snapshots into a calendar."""

    events = _tail_jsonl(
        settings.paths.output_dir / "live" / "events" / "pnl.jsonl",
        limit=50_000,
    )
    fill_events = _tail_jsonl(
        settings.paths.output_dir / "live" / "events" / "fills.jsonl",
        limit=50_000,
    )
    fill_times = sorted(
        timestamp
        for event in fill_events
        if (timestamp := pd.to_datetime(event.get("recorded_at"), utc=True, errors="coerce"))
        is not pd.NaT
    )
    account_fees_by_date: dict[str, float] = {}
    account_fills_by_date: dict[str, int] = {}
    canonical_fees_by_date: dict[str, float] = {}
    canonical_fills_by_date: dict[str, int] = {}
    for event in fill_events:
        payload = dict(event.get("payload") or {})
        timestamp = pd.to_datetime(
            event.get("recorded_at") or payload.get("received_at"),
            utc=True,
            errors="coerce",
        )
        if timestamp is pd.NaT:
            continue
        date_utc = timestamp.strftime("%Y-%m-%d")
        fee = _safe_float(payload.get("fee_eur") or payload.get("fee"))
        if str(event.get("event") or "") == "BITVAVO_ACCOUNT_FILL":
            if fee is not None:
                account_fees_by_date[date_utc] = (
                    account_fees_by_date.get(date_utc, 0.0) + fee
                )
            account_fills_by_date[date_utc] = (
                account_fills_by_date.get(date_utc, 0) + 1
            )
        elif str(event.get("event") or "") == "CANONICAL_FILL":
            if fee is not None:
                canonical_fees_by_date[date_utc] = (
                    canonical_fees_by_date.get(date_utc, 0.0) + fee
                )
            canonical_fills_by_date[date_utc] = (
                canonical_fills_by_date.get(date_utc, 0) + 1
            )
    # Canonical fill records summarize the same venue fills after reconciliation.
    # Prefer granular account-stream fills and use canonical records only as a
    # fallback so fees and fill counts are never double counted.
    fees_by_date = {
        date: account_fees_by_date.get(date, canonical_fees_by_date.get(date, 0.0))
        for date in set(account_fees_by_date) | set(canonical_fees_by_date)
    }
    fills_by_date = {
        date: account_fills_by_date.get(date, canonical_fills_by_date.get(date, 0))
        for date in set(account_fills_by_date) | set(canonical_fills_by_date)
    }
    capital_flows = _tail_jsonl(
        settings.paths.output_dir / "portfolio" / "external_capital_flows.jsonl",
        limit=50_000,
    )
    flows_by_date: dict[str, float] = {}
    flow_times: list[pd.Timestamp] = []
    for flow in capital_flows:
        if not flow.get("operator_confirmed"):
            continue
        flow_date = str(flow.get("date_utc") or "")
        amount = _safe_float(flow.get("amount_eur"))
        timestamp = pd.to_datetime(
            flow.get("effective_at"), utc=True, errors="coerce"
        )
        if len(flow_date) == 10 and amount is not None:
            flows_by_date[flow_date] = flows_by_date.get(flow_date, 0.0) + amount
        if timestamp is not pd.NaT:
            flow_times.append(timestamp)
    flow_times.sort()
    by_date: dict[str, dict[str, Any]] = {}
    previous_snapshot: dict[str, dict[str, Any]] = {}
    unexplained_steps: dict[str, float] = {}
    for event in events:
        if str(event.get("event") or "") != "DAILY_PNL_TARGET_SNAPSHOT":
            continue
        state = dict(event.get("state") or {})
        date_utc = str(state.get("date_utc") or "")
        if len(date_utc) != 10:
            continue
        valuation_status = str(state.get("valuation_status") or "")
        # Keep incomplete account snapshots in the append-only source ledger,
        # but never let a temporary missing public price become trading P&L.
        # A later complete snapshot on the same day remains eligible.
        if valuation_status in {
            "PUBLIC_PRICE_VALUATION_UNAVAILABLE",
            "INCOMPLETE_MARK_TO_MARKET",
            "VALUATION_UNAVAILABLE",
        }:
            continue
        recorded_at = str(event.get("recorded_at") or "")
        previous = by_date.get(date_utc)
        if previous and str(previous.get("recorded_at") or "") >= recorded_at:
            continue
        pnl = _safe_float(state.get("mark_to_market_pnl_eur"))
        start = _safe_float(state.get("day_start_equity_eur"))
        end = _safe_float(state.get("current_estimated_equity_eur"))
        recorded_timestamp = pd.to_datetime(recorded_at, utc=True, errors="coerce")
        prior = previous_snapshot.get(date_utc)
        if prior and end is not None and recorded_timestamp is not pd.NaT:
            prior_end = _safe_float(prior.get("equity"))
            prior_timestamp = pd.to_datetime(
                prior.get("recorded_at"),
                utc=True,
                errors="coerce",
            )
            if prior_end is not None and prior_timestamp is not pd.NaT:
                step = end - prior_end
                material_threshold = max(20.0, abs(prior_end) * 0.03)
                explained_event_observed = any(
                    prior_timestamp < fill_time <= recorded_timestamp
                    for fill_time in fill_times
                ) or any(
                    prior_timestamp < flow_time <= recorded_timestamp
                    for flow_time in flow_times
                )
                if abs(step) >= material_threshold and not explained_event_observed:
                    current_largest = unexplained_steps.get(date_utc, 0.0)
                    if abs(step) > abs(current_largest):
                        unexplained_steps[date_utc] = step
        previous_snapshot[date_utc] = {
            "recorded_at": recorded_at,
            "equity": end,
        }
        unexplained_step = unexplained_steps.get(date_utc)
        external_flow = flows_by_date.get(date_utc, 0.0)
        raw_adjusted_pnl = pnl - external_flow if pnl is not None else None
        # A material, unexplained equity discontinuity usually means that an
        # inventory valuation disappeared between snapshots (for example while
        # a provider market was temporarily unavailable).  Preserve the raw
        # observation for audit, but never present it as verified trading P&L.
        adjusted_pnl = (
            None if unexplained_step is not None else raw_adjusted_pnl
        )
        by_date[date_utc] = {
            "date": date_utc,
            "recorded_at": recorded_at,
            "day_start_equity_eur": start,
            "day_end_equity_eur": end,
            "account_wide_mtm_pnl_eur": pnl,
            "external_capital_flow_eur": external_flow,
            "raw_cash_flow_adjusted_pnl_eur": raw_adjusted_pnl,
            "cash_flow_adjusted_pnl_eur": adjusted_pnl,
            "return_fraction": (
                adjusted_pnl / start
                if adjusted_pnl is not None and start not in {None, 0.0}
                else None
            ),
            "status": "UNVERIFIED"
            if unexplained_step is not None
            else "PROFIT"
            if adjusted_pnl is not None and adjusted_pnl > 0.005
            else "LOSS"
            if adjusted_pnl is not None and adjusted_pnl < -0.005
            else "FLAT",
            "target_eur": _safe_float(state.get("scaled_daily_target_eur")),
            "target_non_binding": state.get("non_binding") is True,
            "valuation_status": state.get("valuation_status"),
            "fees_eur": fees_by_date.get(date_utc, 0.0),
            "fill_events": fills_by_date.get(date_utc, 0),
            "pnl_quality": (
                "UNEXPLAINED_CAPITAL_FLOW_OR_VALUATION_JUMP"
                if unexplained_step is not None
                else "OPERATOR_CONFIRMED_CASH_FLOW_ADJUSTED"
                if external_flow != 0.0
                else "RAW_ACCOUNT_WIDE_MARK_TO_MARKET"
            ),
            "unexplained_equity_step_eur": unexplained_step,
            "external_cash_flow_adjusted": external_flow != 0.0,
            "scope": "ACCOUNT_WIDE_INCLUDING_EXTERNAL_INVENTORY",
            "orders_generated": int(state.get("orders_generated") or 0),
            "orders_submitted": int(state.get("orders_submitted") or 0),
        }
    rows = [by_date[key] for key in sorted(by_date)][-max(1, limit) :]
    months: dict[str, dict[str, Any]] = {}
    for row in rows:
        month = row["date"][:7]
        summary = months.setdefault(
            month,
            {
                "month": month,
                "pnl_eur": 0.0,
                "positive_days": 0,
                "negative_days": 0,
                "flat_days": 0,
                "unverified_days": 0,
                "validated_days": 0,
                "observed_days": 0,
            },
        )
        pnl = row.get("cash_flow_adjusted_pnl_eur")
        if pnl is not None:
            summary["pnl_eur"] += float(pnl)
        summary["observed_days"] += 1
        if row["status"] == "UNVERIFIED":
            summary["unverified_days"] += 1
        else:
            summary["validated_days"] += 1
            summary[
                "positive_days"
                if row["status"] == "PROFIT"
                else "negative_days"
                if row["status"] == "LOSS"
                else "flat_days"
            ] += 1
    return {
        "schema_version": "daily_pnl_calendar_v1",
        "generated_at": utc_iso(),
        "timezone": "UTC",
        "rows": rows,
        "months": list(months.values()),
        "observed_days": len(rows),
        "positive_days": sum(row["status"] == "PROFIT" for row in rows),
        "negative_days": sum(row["status"] == "LOSS" for row in rows),
        "flat_days": sum(row["status"] == "FLAT" for row in rows),
        "unverified_days": sum(
            row["status"] == "UNVERIFIED" for row in rows
        ),
        "validated_days": sum(
            row["status"] != "UNVERIFIED" for row in rows
        ),
        "days_with_unexplained_discontinuity": sum(
            row.get("pnl_quality")
            == "UNEXPLAINED_CAPITAL_FLOW_OR_VALUATION_JUMP"
            for row in rows
        ),
        "scope": "ACCOUNT_WIDE_INCLUDING_EXTERNAL_INVENTORY",
        "strategy_only_pnl_available": False,
        "external_cash_flow_adjustment_available": True,
        "net_external_capital_flow_eur": sum(flows_by_date.values()),
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_trending_read_model(
    active_trading: dict[str, Any],
    opportunities: dict[str, Any],
    macro: dict[str, Any],
) -> dict[str, Any]:
    """Rank hot markets without converting momentum into an entry signal."""

    all_opportunities = [
        dict(row) for row in opportunities.get("all") or [] if isinstance(row, dict)
    ]
    status_priority = {
        "ACTIONABLE": 5,
        "EARLY_MOMENTUM_ALERT": 4,
        "NEAR_ENTRY": 3,
        "PULLBACK_PENDING": 2,
        "APPROACHING": 1,
    }
    best_by_market: dict[str, dict[str, Any]] = {}
    for row in all_opportunities:
        market = str(row.get("market") or "")
        if not market:
            continue
        previous = best_by_market.get(market)
        rank = (
            status_priority.get(str(row.get("status") or ""), 0),
            _safe_float(row.get("score")) or 0.0,
        )
        previous_rank = (
            status_priority.get(str((previous or {}).get("status") or ""), 0),
            _safe_float((previous or {}).get("score")) or 0.0,
        )
        if previous is None or rank > previous_rank:
            best_by_market[market] = row
    rotation = [
        dict(row)
        for row in active_trading.get("top_5_rotation") or []
        if isinstance(row, dict)
    ]
    if not rotation:
        rotation = [
            {
                "market": market,
                "rank": index + 1,
                "rotation_score": opportunity.get("rotation_score") or 0.0,
                "decision": "WATCH",
                "returns": {},
                "live_market_executable": opportunity.get(
                    "live_market_executable"
                ),
            }
            for index, (market, opportunity) in enumerate(
                sorted(
                    best_by_market.items(),
                    key=lambda item: _safe_float(item[1].get("score")) or 0.0,
                    reverse=True,
                )[:10]
            )
        ]
    rotation_markets = {str(row.get("market") or "") for row in rotation}
    for market, opportunity in best_by_market.items():
        status = str(opportunity.get("status") or "")
        if market in rotation_markets or status not in {
            "EARLY_MOMENTUM_ALERT",
            "PULLBACK_PENDING",
        }:
            continue
        formula = dict(opportunity.get("formula") or {})
        rotation.append(
            {
                "market": market,
                "rank": len(rotation) + 1,
                "rotation_score": opportunity.get("rotation_score") or 0.0,
                "decision": "WATCH",
                "returns": {
                    "return_1h": formula.get("return_1h"),
                    "return_4h": formula.get("return_4h"),
                },
                "live_market_executable": opportunity.get(
                    "live_market_executable"
                ),
            }
        )
        rotation_markets.add(market)
    stablecoin = dict(macro.get("stablecoin_liquidity") or {})
    stablecoin_state = str(stablecoin.get("state") or "DATA_PENDING")
    rows: list[dict[str, Any]] = []
    for rotation_row in rotation:
        market = str(rotation_row.get("market") or "")
        opportunity = best_by_market.get(market, {})
        status = str(opportunity.get("status") or "HOT_NO_ENTRY")
        live_authority = opportunity.get("live_authority_granted") is True
        if status == "ACTIONABLE" and live_authority:
            action = "BUY_READY"
            advice = "Bevestigde entry; uitsluitend via de canonieke live-engine."
        elif status == "ACTIONABLE":
            action = "RESEARCH_SIGNAL"
            advice = "Entry technisch actief, maar DNA heeft geen live-authority."
        elif status == "NEAR_ENTRY":
            action = "WATCH_NEAR_ENTRY"
            advice = "Wacht op bevestiging van de laatste gesloten candle."
        elif status == "EARLY_MOMENTUM_ALERT":
            action = "EARLY_MOVE"
            advice = (
                "Vroege prijs- en volumeversnelling; wacht op een geldige "
                "entrybevestiging en jaag de candle niet na."
            )
        elif status == "PULLBACK_PENDING":
            action = "PULLBACK_PENDING"
            advice = (
                "Sterke beweging, maar al uitgerekt ten opzichte van ATR/EMA; "
                "wacht op pullback en een nieuwe gesloten candle."
            )
        elif str(rotation_row.get("decision") or "") == "FAVOUR":
            action = "HOT_NO_ENTRY"
            advice = "Sterke relatieve beweging; nog geen geldige entry."
        else:
            action = "WATCH"
            advice = "Volgen; huidige setup rechtvaardigt nog geen order."
        if stablecoin_state == "DRAINING":
            advice += " Stablecoinliquiditeit daalt: sizing verlagen."
        elif stablecoin_state == "EXPANDING":
            advice += " Stablecoinliquiditeit groeit: context is ondersteunend."
        rotation_score = _safe_float(rotation_row.get("rotation_score")) or 0.0
        opportunity_score = _safe_float(opportunity.get("score")) or 0.0
        hot_score = max(0.0, min(100.0, 0.70 * rotation_score + 0.30 * opportunity_score))
        rows.append(
            {
                "rank": int(rotation_row.get("rank") or len(rows) + 1),
                "market": market,
                "action": action,
                "advice": advice,
                "hot_score": hot_score,
                "rotation_score": rotation_score,
                "returns": dict(rotation_row.get("returns") or {}),
                "strategy": opportunity.get("strategy"),
                "family": opportunity.get("family"),
                "timeframe": opportunity.get("timeframe"),
                "status": status,
                "current_price": opportunity.get("current_price"),
                "entry_zone": opportunity.get("entry_zone"),
                "trigger": opportunity.get("trigger"),
                "stop_loss": opportunity.get("stop"),
                "take_profit_1": opportunity.get("target_1"),
                "take_profit_2": opportunity.get("target_2"),
                "distance_to_trigger": opportunity.get("distance_to_trigger"),
                "confidence": opportunity.get("confidence"),
                "timeframe_conflicts": opportunity.get("timeframe_conflicts") or [],
                "macro_regime": macro.get("regime"),
                "stablecoin_liquidity_state": stablecoin_state,
                "live_market_executable": rotation_row.get(
                    "live_market_executable"
                ),
                "live_authority_granted": live_authority,
                "reason_not_entered": opportunity.get("reason_not_yet_entered")
                or "NO_CONFIRMED_ENTRY",
                "early_move_formula": dict(opportunity.get("formula") or {}),
            }
        )
    rows.sort(key=lambda row: (-row["hot_score"], row["rank"]))
    return {
        "schema_version": "trending_market_read_model_v2",
        "generated_at": utc_iso(),
        "rows": rows[:10],
        "macro_regime": macro.get("regime"),
        "stablecoin_liquidity": stablecoin,
        "advice_is_entry_signal": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_signal_cards(
    opportunities: dict[str, Any],
    generated_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create one explicit UI contract for entries and managed positions."""

    cards: list[dict[str, Any]] = []
    for opportunity in [
        *(opportunities.get("top_5_actionable") or []),
        *(opportunities.get("top_5_near_entry") or []),
    ]:
        row = dict(opportunity)
        status = str(row.get("status") or "WATCH")
        live_authority = row.get("live_authority_granted") is True
        cards.append(
            {
                "action": "BUY_READY"
                if status == "ACTIONABLE" and live_authority
                else "RESEARCH_SIGNAL"
                if status == "ACTIONABLE"
                else "WATCH",
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "timeframe": row.get("timeframe"),
                "status": status,
                "entry_zone": row.get("entry_zone"),
                "trigger": row.get("trigger"),
                "current_price": row.get("current_price"),
                "stop_loss": row.get("stop"),
                "take_profit_1": row.get("target_1"),
                "take_profit_2": row.get("target_2"),
                "confidence": row.get("confidence"),
                "distance_to_trigger": row.get("distance_to_trigger"),
                "reason": row.get("reason_not_yet_entered"),
                "live_authority_granted": live_authority,
                "execution": "CANONICAL_LIVE_ENGINE_ONLY"
                if live_authority
                else "NO_LIVE_EXECUTION_AUTHORITY",
            }
        )
    for position in dict(generated_state.get("positions") or {}).values():
        row = dict(position)
        cards.insert(
            0,
            {
                "action": "HOLD_POSITION"
                if str(row.get("status") or "OPEN") == "OPEN"
                else "ORDER_PENDING",
                "market": row.get("market"),
                "strategy": row.get("strategy_id"),
                "timeframe": row.get("timeframe"),
                "status": row.get("status"),
                "entry_zone": [row.get("entry_price"), row.get("entry_price")],
                "trigger": row.get("entry_price"),
                "current_price": None,
                "stop_loss": row.get("stop_loss"),
                "take_profit_1": row.get("take_profit_1"),
                "take_profit_2": row.get("take_profit_2"),
                "quantity": row.get("quantity"),
                "confidence": None,
                "reason": "MANAGED_LIVE_POSITION",
                "live_authority_granted": True,
                "execution": "LIVE_POSITION_MANAGEMENT",
            },
        )
    return cards[:20]


def _trend_label(value: Any) -> str:
    selected = _safe_float(value)
    if selected is None:
        return "DATA_PENDING"
    if selected >= 0.005:
        return "BULLISH"
    if selected <= -0.005:
        return "BEARISH"
    return "NEUTRAL"


def build_multi_timeframe_matrix(
    settings: Settings,
    rotation: dict[str, Any],
    opportunities: dict[str, Any],
) -> dict[str, Any]:
    """Build a closed-candle market matrix without creating signals."""

    all_opportunities = [
        dict(row)
        for row in opportunities.get("all") or []
        if isinstance(row, dict)
    ]
    by_market: dict[str, list[dict[str, Any]]] = {}
    for opportunity in all_opportunities:
        market = str(opportunity.get("market") or "")
        if market:
            by_market.setdefault(market, []).append(opportunity)
    rows: list[dict[str, Any]] = []
    for rotation_row in rotation.get("rows") or []:
        selected = dict(rotation_row)
        market = str(selected.get("market") or "")
        if not market:
            continue
        returns = dict(selected.get("returns") or {})
        weekly_return: float | None = None
        weekly_path = (
            settings.paths.processed_data_dir / f"{market}_1W.parquet"
        )
        if weekly_path.is_file():
            try:
                weekly = pd.read_parquet(weekly_path, columns=["close"])
            except (OSError, ValueError):
                weekly = pd.DataFrame()
            closes = pd.to_numeric(
                weekly.get("close", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            if len(closes) >= 2 and float(closes.iloc[-2]) > 0:
                weekly_return = float(
                    closes.iloc[-1] / closes.iloc[-2] - 1.0
                )
        market_opportunities = by_market.get(market, [])
        active = [
            row
            for row in market_opportunities
            if str(row.get("status") or "")
            in {"ACTIONABLE", "NEAR_ENTRY", "APPROACHING"}
        ]
        trend_values = {
            "15m": returns.get("return_15m"),
            "1h": returns.get("return_1h"),
            "2h": returns.get("return_2h"),
            "4h": returns.get("return_4h"),
            "1d": returns.get("return_1d"),
            "1W": weekly_return,
        }
        labels = {
            timeframe: _trend_label(value)
            for timeframe, value in trend_values.items()
        }
        valid = [
            label for label in labels.values() if label != "DATA_PENDING"
        ]
        bullish = sum(label == "BULLISH" for label in valid)
        bearish = sum(label == "BEARISH" for label in valid)
        alignment = (
            100.0 * max(bullish, bearish) / len(valid) if valid else 0.0
        )
        rows.append(
            {
                "rank": selected.get("rank"),
                "market": market,
                "market_tier": selected.get("market_tier"),
                "live_market_executable": selected.get(
                    "live_market_executable"
                ),
                "trends": labels,
                "returns": {**trend_values},
                "alignment_score": alignment,
                "active_strategy_count": len(
                    {
                        str(row.get("strategy") or "") for row in active
                    }
                ),
                "active_families": sorted(
                    {
                        str(row.get("family") or "")
                        for row in active
                        if row.get("family")
                    }
                ),
                "best_opportunity_status": (
                    str(active[0].get("status")) if active else "NONE"
                ),
                "best_opportunity": (
                    str(active[0].get("strategy")) if active else None
                ),
                "rotation_score": selected.get("rotation_score"),
            }
        )
    return {
        "schema_version": "multi_timeframe_ui_matrix_v1",
        "generated_at": utc_iso(),
        "closed_candle_only": True,
        "rows": rows,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def build_paper_read_model(
    paper: dict[str, Any],
    universe: dict[str, Any],
) -> dict[str, Any]:
    """Add current public marks to paper positions without mixing live state."""

    midpoints = {
        str(row.get("market")): _safe_float(row.get("midpoint"))
        for row in universe.get("rows") or []
        if isinstance(row, dict) and row.get("market")
    }
    positions: dict[str, dict[str, Any]] = {}
    total_gross_unrealized = 0.0
    marked_positions = 0
    for dna, raw in (paper.get("positions") or {}).items():
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        market = str(row.get("market") or "")
        entry = _safe_float(row.get("entry_price"))
        quantity = _safe_float(row.get("quantity"))
        stop = _safe_float(row.get("stop_loss"))
        mark = midpoints.get(market)
        if entry is not None and quantity is not None and mark is not None:
            gross_pnl = (mark - entry) * quantity
            row.update(
                {
                    "current_price": mark,
                    "entry_notional_eur": entry * quantity,
                    "current_notional_eur": mark * quantity,
                    "gross_unrealized_pnl_eur": gross_pnl,
                    "gross_return_fraction": (
                        (mark / entry) - 1.0 if entry > 0 else None
                    ),
                    "distance_to_stop_fraction": (
                        (mark - stop) / mark
                        if stop is not None and mark > 0
                        else None
                    ),
                    "valuation_status": "PUBLIC_MIDPOINT_MARKED",
                }
            )
            total_gross_unrealized += gross_pnl
            marked_positions += 1
        else:
            row["valuation_status"] = "MARK_UNAVAILABLE"
        positions[str(dna)] = row
    return {
        **paper,
        "positions": positions,
        "marked_positions": marked_positions,
        "gross_unrealized_pnl_eur": total_gross_unrealized,
        "valuation_scope": "PAPER_ONLY_PUBLIC_MIDPOINT_GROSS_OF_EXIT_COSTS",
    }


def build_ui_snapshot(settings: Settings) -> dict[str, Any]:
    """Build one sanitized, non-blocking UI read model."""

    output = settings.paths.output_dir
    crypto_maturity = _safe_read(
        output / "roadmap" / "crypto_maturity_ladder.json"
    )
    live = _safe_read(output / "live" / "autonomous_live_status.json")
    state = _safe_read(output / "live" / "autonomous_live_state.json")
    account = _safe_read(output / "operations" / "live_account_health.json")
    universe = _safe_read(output / "governance" / "live_universe.json")
    candles = _safe_read(output / "data" / "candle_health.json")
    authority = _safe_read(
        output / "governance" / "positive_strategy_live_authority.json"
    )
    mtf_authority = _safe_read(
        output / "governance" / "multi_timeframe_authority.json"
    )
    generated_live = _safe_read(
        output / "live" / "generated_strategy_live_status.json"
    )
    generated_live_state = _safe_read(
        output / "live" / "generated_strategy_live_state.json"
    )
    generated_paper = _safe_read(
        output / "paper" / "generated_strategy_state.json"
    )
    paper_read_model = build_paper_read_model(generated_paper, universe)
    active_trading = _safe_read(
        output / "active_trading" / "status.json"
    )
    active_opportunities = _safe_read(
        output / "active_trading" / "opportunities.json"
    )
    active_rotation = _safe_read(
        output / "active_trading" / "rotation.json"
    )
    active_timeframes = _safe_read(
        output / "active_trading" / "timeframes.json"
    )
    active_macro = _safe_read(
        output / "active_trading" / "macro_crypto.json"
    )
    market_mechanics = _safe_read(
        output / "active_trading" / "market_mechanics.json"
    )
    reference_integration = _safe_read(
        output / "reference_integration" / "system_health.json"
    )
    capital_utilization = _safe_read(
        output / "portfolio" / "capital_utilization.json"
    )
    proactive_allocation = _safe_read(
        output / "portfolio" / "proactive_allocation.json"
    )
    inventory_reallocation = _safe_read(
        output / "operations" / "inventory_reallocation.json"
    )
    research = _safe_read(
        output
        / "lab"
        / "reports"
        / "multi_timeframe_authority_validation_v1.json"
    )
    telegram = _safe_read(output / "notifications" / "telegram_status.json")
    kill_switch = KillSwitch(settings.paths.checkpoints_dir / "kill_switch.json")
    events = {
        stream: _tail_jsonl(output / "live" / "events" / f"{stream}.jsonl")
        for stream in (
            "signals",
            "orders",
            "fills",
            "positions",
            "pnl",
            "risk",
            "errors",
        )
    }
    mtf_selected = list(mtf_authority.get("strategies") or [])
    approved = (
        mtf_selected
        if mtf_authority.get("status") == "READY" and mtf_selected
        else list(authority.get("approved_candidates") or [])
    )
    account_projection = dict(account.get("account") or {})
    portfolio_valuation = dict(
        account_projection.get("portfolio_valuation") or {}
    )
    pnl_calendar = build_daily_pnl_calendar(settings)
    trending = build_trending_read_model(
        active_trading,
        active_opportunities,
        active_macro,
    )
    signal_cards = build_signal_cards(
        active_opportunities,
        generated_live_state,
    )
    multi_timeframe_matrix = build_multi_timeframe_matrix(
        settings,
        active_rotation,
        active_opportunities,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_iso(),
        "crypto_maturity": crypto_maturity,
        "system": {
            "state": state.get("state") or live.get("state") or "UNKNOWN",
            "status": live.get("status") or "UNKNOWN",
            "last_heartbeat": live.get("last_heartbeat")
            or live.get("updated_at"),
            "reconciliation": account.get("status") or "UNKNOWN",
            "kill_switch_active": kill_switch.active,
            "kill_switch_reason": kill_switch.reason,
            "processes": {
                "live_supervisor": _process_projection(
                    output / "live" / "autonomous_live.lock"
                ),
                "data_sync": _process_projection(
                    settings.paths.checkpoints_dir / "data_service.lock"
                ),
                "research_lab": _process_projection(
                    output
                    / "research"
                    / "simple_strategy_lab"
                    / "service.lock"
                ),
            },
        },
        "account": {
            "estimated_equity_eur": account_projection.get(
                "estimated_equity_eur"
            )
            or portfolio_valuation.get("estimated_total_equity_eur")
            or account.get("estimated_equity_eur"),
            "eur_available": account_projection.get("eur_available")
            or account.get("eur_available"),
            "realized_pnl": generated_live.get("realized_pnl_eur"),
            "unrealized_pnl": generated_live.get("unrealized_pnl_eur"),
            "daily_pnl": generated_live.get("daily_pnl_eur"),
            "positions": account_projection.get("non_eur_holdings")
            or account.get("non_eur_holdings")
            or [],
            "entry_allowed": account.get("entry_allowed"),
            "entry_blockers": list(account.get("entry_blockers") or []),
            "eur_cash_continuity": dict(
                account.get("eur_cash_continuity") or {}
            ),
        },
        "universe": universe,
        "candle_health": candles,
        "authority": {
            "active": authority.get("active"),
            "approved_candidates": approved,
            "approved_candidate_count": len(approved),
            "timeframes": sorted(
                {
                    str(row.get("timeframe"))
                    for row in approved
                    if row.get("timeframe")
                }
            ),
            "multi_timeframe": mtf_selected,
        },
        "strategies": approved,
        "opportunities": active_opportunities.get("all")
        or generated_live.get("opportunities")
        or generated_live.get("evaluations")
        or [],
        "active_trading": {
            "generated_at": active_trading.get("generated_at"),
            "scan_interval_minutes": (
                active_trading.get("scan_interval_minutes")
                or settings.autonomous_live.active_trading_scan_minutes
            ),
            "scan_poll_seconds": (
                active_trading.get("scan_poll_seconds")
                or settings.autonomous_live.active_trading_poll_seconds
            ),
            "scan_maximum_rows": (
                active_trading.get("scan_maximum_rows")
                or settings.autonomous_live.active_trading_maximum_rows
            ),
            "status": active_trading.get("status") or "NOT_SCANNED",
            "reason": active_trading.get("reason"),
            "markets_scanned": active_trading.get("markets_scanned") or [],
            "orders_generated": active_trading.get("orders_generated") or 0,
            "orders_submitted": active_trading.get("orders_submitted") or 0,
            "top_5_actionable": (
                active_opportunities.get("top_5_actionable") or []
            ),
            "top_5_near_entry": (
                active_opportunities.get("top_5_near_entry") or []
            ),
            "top_5_rotation": (
                active_opportunities.get("top_5_rotation") or []
            ),
        },
        "timeframe_status": active_timeframes,
        "macro_crypto": active_macro,
        "market_mechanics": market_mechanics,
        "reference_integration": {
            "status": reference_integration.get("status") or "NOT_BUILT",
            "live_readiness": reference_integration.get("live_readiness")
            or "NO_GO",
            "phases": reference_integration.get("phases") or {},
            "reference_health": reference_integration.get("reference_health")
            or [],
            "research_state": reference_integration.get("research_state") or {},
            "model_state": reference_integration.get("model_state") or {},
            "canonical_model_state": reference_integration.get(
                "canonical_model_state"
            )
            or {},
            "execution_state": reference_integration.get("execution_state") or {},
        },
        "stablecoin_liquidity": active_macro.get("stablecoin_liquidity") or {},
        "pnl_calendar": pnl_calendar,
        "trending": trending,
        "signal_cards": signal_cards,
        "multi_timeframe_matrix": multi_timeframe_matrix,
        "execution_policy": {
            "limit_entries_enabled": (
                settings.execution.live_limit_entries_enabled
            ),
            "limit_entry_time_in_force": (
                settings.execution.live_limit_entry_time_in_force
            ),
            "limit_price_buffer_bps": (
                settings.execution.live_limit_price_buffer_bps
            ),
            "market_fallback_enabled": (
                settings.execution.live_limit_market_fallback_enabled
            ),
            "risk_exits": "MARKET",
            "latest_entry_order_plan": generated_live.get(
                "entry_order_plan"
            ),
            "orders_generated": 0,
            "orders_submitted": 0,
        },
        "capital_utilization": capital_utilization,
        "proactive_allocation": proactive_allocation,
        "inventory_reallocation": inventory_reallocation,
        "paper": paper_read_model,
        "live": generated_live,
        "research": {
            "candidate_count": research.get("candidate_count"),
            "passing_count": research.get("passing_count"),
            "candidates": research.get("candidates") or [],
            "selected_candidates": research.get("selected_candidates") or [],
        },
        "events": events,
        "telegram": {
            "status": telegram.get("status") or "UNKNOWN",
            "last_successful_send": telegram.get("last_successful_send"),
            "queue_size": telegram.get("queue_size"),
        },
        "orders_generated_by_ui": 0,
        "orders_submitted_by_ui": 0,
    }


def candle_payload(
    settings: Settings,
    market: str,
    timeframe: str,
    *,
    limit: int = 240,
) -> dict[str, Any]:
    normalized_market = market.strip().upper().replace("/", "-")
    normalized_timeframe = timeframe.strip()
    universe = _safe_read(
        settings.paths.output_dir / "governance" / "live_universe.json"
    )
    if normalized_market not in universe.get("selected_markets", []):
        raise ValueError("MARKET_NOT_IN_LIVE_UNIVERSE")
    if normalized_timeframe not in {"15m", "1h", "2h", "4h", "1d", "1W"}:
        raise ValueError("UNSUPPORTED_TIMEFRAME")
    path = (
        settings.paths.processed_data_dir
        / f"{normalized_market}_{normalized_timeframe}.parquet"
    )
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = (
        frame.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    context = frame.iloc[-max(limit + 10, 300) :].copy()
    fractals = confirmed_fractals(context, left=2, right=2)
    selected = context.iloc[-limit:]
    fractals = fractals.reindex(selected.index)
    candles = [
        {
            "time": int(timestamp.timestamp()),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for timestamp, row in selected.iterrows()
    ]
    markers: list[dict[str, Any]] = []
    for timestamp, value in fractals["confirmed_fractal_high_price"].dropna().items():
        markers.append(
            {
                "time": int(timestamp.timestamp()),
                "type": "FRACTAL_HIGH",
                "price": float(value),
            }
        )
    for timestamp, value in fractals["confirmed_fractal_low_price"].dropna().items():
        markers.append(
            {
                "time": int(timestamp.timestamp()),
                "type": "FRACTAL_LOW",
                "price": float(value),
            }
        )
    for event in _tail_jsonl(
        settings.paths.output_dir / "live" / "events" / "signals.jsonl",
        limit=2_000,
    ):
        if str(event.get("market") or "") != normalized_market:
            continue
        if str(event.get("timeframe") or "") != normalized_timeframe:
            continue
        signal = str(event.get("signal") or "").upper()
        if signal not in {"BUY", "FRESH_BUY", "EXIT", "SELL_SIGNAL"}:
            continue
        timestamp_value = event.get("timestamp_utc") or event.get(
            "recorded_at"
        )
        price = event.get("entry_price_reference") or event.get(
            "expected_exit_price"
        )
        try:
            marker_time = int(pd.Timestamp(timestamp_value).timestamp())
            marker_price = float(price)
        except (TypeError, ValueError):
            continue
        markers.append(
            {
                "time": marker_time,
                "type": "ENTRY" if signal in {"BUY", "FRESH_BUY"} else "EXIT",
                "price": marker_price,
                "strategy_id": event.get("strategy_id"),
            }
        )
    generated_state = _safe_read(
        settings.paths.output_dir
        / "live"
        / "generated_strategy_live_state.json"
    )
    levels: list[dict[str, Any]] = []
    for position in (generated_state.get("positions") or {}).values():
        if str(position.get("market") or "") != normalized_market:
            continue
        for kind, field in (
            ("STOP", "stop_loss"),
            ("TAKE_PROFIT", "take_profit_1"),
            ("TRAILING_STOP", "trailing_stop"),
        ):
            try:
                price = float(position[field])
            except (KeyError, TypeError, ValueError):
                continue
            levels.append(
                {
                    "type": kind,
                    "price": price,
                    "strategy_id": position.get("strategy_id"),
                }
            )
    return {
        "market": normalized_market,
        "timeframe": normalized_timeframe,
        "candles": candles,
        "markers": sorted(markers, key=lambda row: row["time"]),
        "levels": levels,
        "closed_candle_only": True,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


HTML_DOCUMENT = r"""<!doctype html>
<html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="csrf-token" content="__CSRF__">
<title>Crypto Control Room</title>
<style>
:root{--bg:#071014;--panel:#0d1b20;--panel2:#11252b;--line:#214047;--txt:#e8f4f1;--muted:#8eaaa7;--green:#37e6a1;--red:#ff657a;--yellow:#ffc857;--blue:#57b8ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% 0,#12333a 0,#071014 44%);color:var(--txt);font:14px Inter,Segoe UI,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;padding:18px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#071014e8;backdrop-filter:blur(12px);z-index:3}
h1{font-size:19px;margin:0;letter-spacing:.04em}.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green);margin-right:9px}
.toolbar{display:flex;gap:8px}.btn{border:1px solid var(--line);background:var(--panel2);color:var(--txt);padding:8px 12px;border-radius:8px;cursor:pointer}.btn.danger{border-color:#743844;color:#ffb2bd}
nav{display:flex;gap:4px;padding:10px 24px;overflow:auto}.tab{border:0;background:transparent;color:var(--muted);padding:9px 13px;cursor:pointer;border-radius:8px}.tab.active{background:var(--panel2);color:var(--txt)}
main{padding:0 24px 36px}.page{display:none}.page.active{display:block}.grid{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px}.card{background:linear-gradient(150deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:13px;padding:14px;min-height:88px;min-width:0}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.value{font-size:clamp(16px,1.6vw,22px);margin-top:8px;font-weight:650;overflow-wrap:anywhere}.ok{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--yellow)}
.section{margin-top:16px}.section h2{font-size:15px;font-weight:600;margin:0 0 10px}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #183138;font-size:12px}th{color:var(--muted);font-weight:500}tr:last-child td{border:0}.badge{padding:3px 7px;border-radius:99px;background:#18383b;color:var(--green);font-size:11px}.badge.blocked{background:#3b2028;color:#ff9cab}
.chartbar{display:flex;gap:8px;margin-bottom:10px}select{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:7px;padding:7px}canvas{width:100%;height:440px;background:#09171b;border:1px solid var(--line);border-radius:12px}
pre{white-space:pre-wrap;background:#09171b;border:1px solid var(--line);padding:12px;border-radius:10px;color:#b9d5d1;max-height:420px;overflow:auto}.empty{color:var(--muted);padding:20px}.foot{color:var(--muted);margin-top:14px;font-size:11px}
.signal-grid,.trend-grid,.calendar-months{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}.signal-card,.trend-card,.calendar-month{background:linear-gradient(150deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:13px;padding:14px}.signal-head,.trend-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.signal-action{font-weight:750;letter-spacing:.05em}.signal-action.buy{color:var(--green)}.signal-action.sell{color:var(--red)}.signal-action.watch{color:var(--yellow)}.levels{display:grid;grid-template-columns:repeat(2,1fr);gap:7px;margin-top:12px}.level{background:#09171b;border:1px solid #183138;border-radius:8px;padding:8px}.level span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.level strong{display:block;margin-top:3px}.advice{color:#c4d9d6;line-height:1.45;margin:10px 0 0}.calendar-month h3{margin:0 0 10px;font-size:14px}.calendar-week,.calendar-days{display:grid;grid-template-columns:repeat(7,1fr);gap:5px}.calendar-week span{color:var(--muted);font-size:10px;text-align:center}.day{min-height:68px;background:#09171b;border:1px solid #183138;border-radius:8px;padding:6px;overflow:hidden}.day.blank{background:transparent;border-color:transparent}.day.profit{border-color:#227a59;background:#0c2820}.day.loss{border-color:#71323d;background:#2a1217}.day.flat{border-color:#645a2e}.day.unverified{border-color:#856f2a;background:#2a2412}.day .n{font-size:10px;color:var(--muted)}.day .pnl{display:block;font-size:12px;font-weight:700;margin-top:8px}.day .ret{font-size:9px;color:var(--muted)}.scope-note{border-left:3px solid var(--yellow);padding:9px 12px;background:#1b1a10;color:#d8cfa6;margin-bottom:12px;border-radius:4px}.meter{height:6px;background:#09171b;border-radius:9px;overflow:hidden;margin-top:9px}.meter span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--green))}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.grid{grid-template-columns:1fr}header,main,nav{padding-left:12px;padding-right:12px}}
</style></head><body>
<header><h1><span class="pulse"></span>Crypto Control Room</h1><div class="toolbar">
<button class="btn" data-action="pause">Pause entries</button><button class="btn" data-action="resume">Resume</button><button class="btn" data-action="reconcile">Reconcile</button><button class="btn danger" data-action="emergency-stop">Emergency stop</button></div></header>
<nav id="tabs"></nav><main>
<section class="page active" data-page="Dashboard"><div class="grid" id="summary"></div><div class="section"><h2>Processen</h2><div id="processes"></div></div><div class="section"><h2>Crypto maturity ladder — beginner naar expert</h2><div class="scope-note">Levels worden uitsluitend opeenvolgend gecertificeerd. Bestaande implementatie telt als bewijs, maar activeert nooit automatisch live trading.</div><div id="maturity"></div></div></section>
<section class="page" data-page="Markets"><div id="markets"></div></section>
<section class="page" data-page="Candles"><div class="chartbar"><select id="marketSelect"></select><select id="tfSelect"><option>15m</option><option>1h</option><option>2h</option><option>4h</option><option>1d</option><option>1W</option></select></div><canvas id="chart" width="1500" height="440"></canvas><div class="foot">Gesloten candles; blauwe/oranje markeringen zijn causaal bevestigde fractals.</div></section>
<section class="page" data-page="Strategies"><div class="section"><h2>Goedgekeurde live-strategieën</h2><div id="strategies"></div></div><div class="section"><h2>Micro-live eligible — operatorgoedkeuring per DNA vereist</h2><div class="scope-note">Deze kandidaten mogen niet automatisch live gaan. Controleer de evidence en voer alleen bewust het getoonde main.py-commando uit.</div><div id="pendingApprovals"></div></div></section>
<section class="page" data-page="Opportunities"><div id="opportunities"></div></section>
<section class="page" data-page="Signals"><div class="scope-note">Alle BUY/SELL-kaarten komen uit de bestaande signal- en position-lifecycle. De UI plaatst zelf nooit orders.</div><div class="signal-grid" id="signalCards"></div></section>
<section class="page" data-page="Trending"><div class="grid" id="stablecoinCards"></div><div class="section"><h2>Hot & trending — context, geen losse entrytrigger</h2><div class="trend-grid" id="trending"></div></div></section>
<section class="page" data-page="Flow & GEX"><div class="scope-note">Deribit-GEX is een regime-overlay; open interest bewijst geen dealerpositionering. Orderflow is uitsluitend prospectief verzameld en wordt nooit historisch verzonnen.</div><div class="grid" id="gexCards"></div><div class="section"><h2>Top-20 spot-orderflow</h2><div id="orderflow"></div></div></section>
<section class="page" data-page="P&L Calendar"><div class="scope-note">Dagresultaten zijn account-wide mark-to-market inclusief externe TAO-inventory. Bevestigde stortingen en opnames worden uit de P&L verwijderd; onverklaarde kapitaalsprongen blijven zichtbaar als waarschuwing.</div><div class="grid" id="pnlSummary"></div><div class="section"><div class="calendar-months" id="pnlCalendar"></div></div><div class="section"><h2>Dagdetails</h2><div id="pnlDays"></div></div><div class="section"><h2>Maandsamenvatting</h2><div id="pnlMonths"></div></div></section>
<section class="page" data-page="Timeframes"><div class="section"><h2>Strategiedekking per timeframe</h2><div id="timeframes"></div></div><div class="section"><h2>Per-markt multi-timeframe matrix</h2><div class="scope-note">Alle trends gebruiken uitsluitend gesloten candles; DATA_PENDING wordt niet als neutraal ingevuld.</div><div id="mtfMatrix"></div></div></section>
<section class="page" data-page="Macro & Capital"><div class="grid" id="capital"></div><div class="section"><h2>Actueel versus regime-doel</h2><div class="scope-note">Doelgewichten zijn beslisondersteuning. Alleen een goedgekeurd frozen DNA met een natuurlijk signaal mag een order starten; externe inventory wordt niet automatisch overgenomen.</div><div id="allocation"></div></div><div class="section"><h2>Operator-only inventoryreallocatie</h2><div class="scope-note">Dit is uitsluitend een read-only, sell-only plan voor bestaande externe inventory. De UI kan het plan niet insturen en het resultaat telt niet als strategieperformance.</div><div id="inventoryReallocation"></div></div><div class="section"><h2>Crypto-macro</h2><div id="macro"></div></div><div class="section"><h2>Rotatie</h2><div id="rotation"></div></div></section>
<section class="page" data-page="Positions & Orders"><div class="scope-note">Paperposities zijn simulaties en staan bewust apart van de echte Bitvavo-accountinventory. Alleen order/fill-events met een live/exchange-identiteit zijn echte uitvoering.</div><div class="section"><h2>Automatische paperposities</h2><div id="paperPositions"></div></div><div class="section"><h2>Bitvavo accountinventory</h2><div id="positions"></div></div><div class="section"><h2>Order/fill events</h2><div id="orders"></div></div></section>
<section class="page" data-page="Research"><div id="research"></div></section>
<section class="page" data-page="System"><pre id="system"></pre></section>
<div class="foot" id="updated"></div></main>
<script>
const pages=["Dashboard","Markets","Candles","Strategies","Opportunities","Signals","Trending","Flow & GEX","P&L Calendar","Timeframes","Macro & Capital","Positions & Orders","Research","System"];const tabs=document.querySelector("#tabs");
pages.forEach((p,i)=>{const b=document.createElement("button");b.className="tab"+(i?"":" active");b.textContent=p;b.onclick=()=>{document.querySelectorAll(".tab,.page").forEach(x=>x.classList.remove("active"));b.classList.add("active");document.querySelector(`[data-page="${p}"]`).classList.add("active");if(p==="Candles")loadCandles()};tabs.appendChild(b)});
const esc=v=>String(v??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));const pct=v=>v==null?"—":(Number(v)*100).toFixed(1)+"%";const eur=v=>v==null?"—":"€"+Number(v).toLocaleString("nl-NL",{maximumFractionDigits:2});
function table(rows,cols){if(!rows?.length)return '<div class="empty">Geen actuele records.</div>';return `<table><thead><tr>${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${esc(c[1](r))}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
const price=v=>v==null?"—":"€"+Number(v).toLocaleString("nl-NL",{maximumFractionDigits:8});const compact=v=>v==null?"—":Number(v).toLocaleString("nl-NL",{notation:"compact",maximumFractionDigits:2});const zone=v=>Array.isArray(v)?v.map(price).join(" – "):price(v);const actionClass=a=>String(a||"").includes("BUY")?"buy":String(a||"").includes("SELL")?"sell":"watch";
function renderSignalCards(rows){if(!rows?.length)return '<div class="empty">Geen verse entry, exit of beheerde positie.</div>';return rows.map(r=>`<article class="signal-card"><div class="signal-head"><div><div class="signal-action ${actionClass(r.action)}">${esc(r.action)}</div><strong>${esc(r.market)}</strong></div><span class="badge ${r.live_authority_granted?'':'blocked'}">${r.live_authority_granted?'LIVE AUTH':'GEEN LIVE AUTH'}</span></div><div class="advice">${esc(r.strategy||'—')} · ${esc(r.timeframe||'—')} · ${esc(r.status||'—')}</div><div class="levels"><div class="level"><span>Entryzone</span><strong>${esc(zone(r.entry_zone))}</strong></div><div class="level"><span>Trigger</span><strong>${esc(price(r.trigger))}</strong></div><div class="level"><span>Stop-loss</span><strong>${esc(price(r.stop_loss))}</strong></div><div class="level"><span>Take-profit 1</span><strong>${esc(price(r.take_profit_1))}</strong></div><div class="level"><span>Take-profit 2</span><strong>${esc(price(r.take_profit_2))}</strong></div><div class="level"><span>Confidence</span><strong>${esc(r.confidence==null?'—':Number(r.confidence).toFixed(1)+'%')}</strong></div></div><p class="advice">${esc(r.reason||'—')}</p></article>`).join("")}
function renderTrending(rows){if(!rows?.length)return '<div class="empty">Geen trending records.</div>';return rows.map(r=>{const rets=r.returns||{},f=r.early_move_formula||{},authority=r.live_authority_granted===true?'LIVE AUTHORITY':'GEEN LIVE AUTHORITY',early=Object.keys(f).length>0;return `<article class="trend-card"><div class="trend-head"><div><span class="signal-action ${actionClass(r.action)}">${esc(r.action)}</span><h3>${esc(r.market)}</h3></div><strong>${Number(r.hot_score||0).toFixed(1)}</strong></div><div class="meter"><span style="width:${Math.max(0,Math.min(100,Number(r.hot_score||0)))}%"></span></div><div class="levels"><div class="level"><span>Actuele prijs</span><strong>${esc(price(r.current_price))}</strong></div><div class="level"><span>Entryzone</span><strong>${esc(Array.isArray(r.entry_zone)?r.entry_zone.map(price).join(' – '):price(r.entry_zone))}</strong></div><div class="level"><span>Trigger</span><strong>${esc(price(r.trigger))}</strong></div><div class="level"><span>Stop-loss</span><strong>${esc(price(r.stop_loss))}</strong></div><div class="level"><span>TP1 / TP2</span><strong>${esc(price(r.take_profit_1))} / ${esc(price(r.take_profit_2))}</strong></div><div class="level"><span>Confidence</span><strong>${r.confidence==null?'—':esc(Number(r.confidence).toFixed(1)+'%')}</strong></div><div class="level"><span>1h / 2h</span><strong>${esc(pct(rets.return_1h))} / ${esc(pct(rets.return_2h))}</strong></div><div class="level"><span>4h / 1d</span><strong>${esc(pct(rets.return_4h))} / ${esc(pct(rets.return_1d))}</strong></div>${early?`<div class="level"><span>15m versnelling</span><strong>${esc(pct(f.return_15m))}</strong></div><div class="level"><span>RVOL20 / volume-z</span><strong>${esc(Number(f.relative_volume_20||0).toFixed(2))}x / ${esc(Number(f.volume_robust_zscore||0).toFixed(2))}</strong></div><div class="level"><span>Afstand boven EMA20</span><strong>${esc(Number(f.extension_atr||0).toFixed(2))} ATR</strong></div>`:''}</div><p class="advice">${esc(r.advice)}</p><div class="foot">${esc(r.strategy||'Geen entrystrategie')} · ${esc(r.timeframe||'—')} · ${esc(r.status||'—')} · ${esc(authority)} · ${esc(r.reason_not_entered)}</div></article>`}).join("")}
function renderCalendars(data){const rows=data?.rows||[];if(!rows.length)return '<div class="empty">Nog geen dagelijkse P&L-snapshots.</div>';const byDate=Object.fromEntries(rows.map(r=>[r.date,r]));const months=[...new Set(rows.map(r=>r.date.slice(0,7)))].slice(-6).reverse();return months.map(month=>{const [y,m]=month.split('-').map(Number),days=new Date(Date.UTC(y,m,0)).getUTCDate(),offset=(new Date(Date.UTC(y,m-1,1)).getUTCDay()+6)%7,cells=[];for(let i=0;i<offset;i++)cells.push('<div class="day blank"></div>');for(let day=1;day<=days;day++){const date=`${month}-${String(day).padStart(2,'0')}`,r=byDate[date];cells.push(r?`<div class="day ${String(r.status||'flat').toLowerCase()}" title="${esc(r.pnl_quality||r.scope)}"><span class="n">${day}</span><span class="pnl">${esc(eur(r.cash_flow_adjusted_pnl_eur))}</span><span class="ret">${esc(pct(r.return_fraction))}</span></div>`:`<div class="day"><span class="n">${day}</span></div>`)}return `<section class="calendar-month"><h3>${new Date(Date.UTC(y,m-1,1)).toLocaleDateString('nl-NL',{month:'long',year:'numeric',timeZone:'UTC'})}</h3><div class="calendar-week"><span>ma</span><span>di</span><span>wo</span><span>do</span><span>vr</span><span>za</span><span>zo</span></div><div class="calendar-days">${cells.join('')}</div></section>`}).join('')}
let snapshot={};async function refresh(){try{snapshot=await(await fetch("/api/snapshot",{cache:"no-store"})).json();render()}catch(e){document.querySelector("#updated").textContent="UI snapshot tijdelijk niet beschikbaar"}}
function render(){const s=snapshot.system||{},u=snapshot.universe||{},a=snapshot.account||{},auth=snapshot.authority||{},at=snapshot.active_trading||{},mc=snapshot.macro_crypto||{},mm=snapshot.market_mechanics||{},cu=snapshot.capital_utilization||{},pa=snapshot.proactive_allocation||{},ir=snapshot.inventory_reallocation||{},cal=snapshot.pnl_calendar||{},tr=snapshot.trending||{},sl=snapshot.stablecoin_liquidity||{},ep=snapshot.execution_policy||{},paper=snapshot.paper||{},maturity=snapshot.crypto_maturity||{},cashGuard=a.eur_cash_continuity||{};const calRows=cal.rows||[],today=calRows.length?calRows[calRows.length-1]:{};
const marketFallback=ep.market_fallback_enabled??ep.limit_market_fallback_enabled??false;const scanDisplay=at.status==="LIVE_ACTIVE_NO_CURRENT_ENTRY"?"ACTIEF — GEEN GELDIGE ENTRY":at.status==="LIVE_ACTIVE_ENTRY_EXECUTED"?"ACTIEF — ENTRY UITGEVOERD":at.status;const paperPnl=Number(paper.gross_unrealized_pnl_eur||0);const cashDelta=cashGuard.observed_delta_eur==null?"—":eur(cashGuard.observed_delta_eur);const cards=[["Runtime",s.state,s.state==="ENABLED"?"ok":"warn"],["Nieuwe entries",a.entry_allowed===true?"TOEGESTAAN":"GEBLOKKEERD",a.entry_allowed===true?"ok":"bad"],["EUR-cashcontinuïteit",cashGuard.status||"DATA_PENDING",cashGuard.new_entries_blocked?"bad":"ok"],["Laatste EUR-mutatie",cashDelta,cashGuard.new_entries_blocked?"bad":""],["Entryblockers",(a.entry_blockers||[]).join(", ")||"Geen",(a.entry_blockers||[]).length?"bad":"ok"],["Actieve scan",scanDisplay,at.status==="LIVE_ACTIVE_ENTRY_EXECUTED"?"ok":at.status==="LIVE_ACTIVE_NO_CURRENT_ENTRY"?"warn":""],["Scaninterval",`${at.scan_interval_minutes||15} min`,"ok"],["Nieuwe-candlepoll",`${at.scan_poll_seconds||30} sec`,"ok"],["Live featurevenster",`${at.scan_maximum_rows||1500} candles`,"ok"],["Laatste volledige scan",at.generated_at||"—",""],["Regime",mc.regime||"—",""],["Stablecoinliquiditeit",sl.state||"DATA_PENDING",sl.state==="EXPANDING"?"ok":sl.state==="DRAINING"?"warn":""],["Entry-orderpolicy",ep.limit_entries_enabled?`LIMIT ${ep.limit_entry_time_in_force} / ${marketFallback?"market fallback":"geen market fallback"}`:"MARKET",ep.limit_entries_enabled?"ok":"warn"],["Equity",eur(a.estimated_equity_eur),""],["EUR beschikbaar",eur(a.eur_available),""],["Dag-P&L gecorrigeerd",eur(today.cash_flow_adjusted_pnl_eur),today.status==="PROFIT"?"ok":today.status==="LOSS"?"bad":""],["Kapitaalgebruik",pct(cu.capital_utilization),""],["Paperposities",paper.open_positions??Object.keys(paper.positions||{}).length,""],["Paper-P&L bruto",eur(paperPnl),paperPnl>0?"ok":paperPnl<0?"bad":""],["Paperfills",paper.paper_fills??0,""],["Live markten",u.live_eligible_count??0,(u.live_eligible_count??0)>=5?"ok":"bad"],["Authority TF",auth.timeframes?.join(", ")||"—",""],["Strategieën",auth.approved_candidate_count??0,""],["Reconciliatie",s.reconciliation,s.reconciliation==="READY"?"ok":"warn"],["Kill switch",s.kill_switch_active?"ACTIEF":"Uit",s.kill_switch_active?"bad":"ok"],["Orders scan",at.orders_submitted??0,(at.orders_submitted??0)>0?"ok":""]];
document.querySelector("#summary").innerHTML=cards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div></div>`).join("");
const procs=Object.entries(s.processes||{}).map(([name,v])=>({name,...v}));document.querySelector("#processes").innerHTML=table(procs,[["Proces",r=>r.name],["PID",r=>r.pid],["Status",r=>r.running?"RUNNING":"STOPPED"]]);
document.querySelector("#maturity").innerHTML=table(maturity.projects||[],[["#",r=>r.project_id],["Level",r=>r.level],["Project",r=>r.name],["Status",r=>r.status]]);
const markets=u.rows||[];document.querySelector("#markets").innerHTML=table(markets,[["Markt",r=>r.market],["Status",r=>r.status],["Spread bps",r=>Number(r.spread_bps||0).toFixed(2)],["24h volume",r=>eur(r.quote_volume_24h_eur)],["Ask depth",r=>eur(r.visible_ask_depth_eur)],["Candles",r=>`${r.candle_timeframes_healthy||0}/${r.required_candle_timeframes||5}`],["Reden",r=>(r.reason_codes||[]).join(", ")]]);
const sel=document.querySelector("#marketSelect"),chosen=sel.value;sel.innerHTML=(u.selected_markets||[]).map(m=>`<option>${m}</option>`).join("");if(chosen&&[...sel.options].some(o=>o.value===chosen))sel.value=chosen;
document.querySelector("#strategies").innerHTML=table(snapshot.strategies,[["Strategie",r=>r.strategy_id],["TF",r=>r.timeframe],["Markten",r=>(r.approved_markets||[]).join(", ")],["DNA",r=>(r.strategy_dna_hash||"").slice(0,16)],["Bron",r=>r.source]]);
const pendingById={};Object.values(snapshot.timeframe_status?.timeframes||{}).forEach(tf=>{(tf.pending_live_dna_approval||[]).forEach(r=>{pendingById[r.strategy_id]=r})});const priority={PRIORITY_MICRO:4,SECONDARY_MICRO:3,DEFER_NEGATIVE_HOLDOUT:2,DEFER_WEAK_EVIDENCE:1};const pending=Object.values(pendingById).sort((a,b)=>(priority[b.approval_priority]||0)-(priority[a.approval_priority]||0)||Number(b.approval_readiness_score||0)-Number(a.approval_readiness_score||0));document.querySelector("#pendingApprovals").innerHTML=table(pending,[["Strategie",r=>r.strategy_id],["TF",r=>r.timeframe],["Prioriteit",r=>r.approval_priority],["Readiness",r=>Number(r.approval_readiness_score||0).toFixed(1)],["Familie",r=>r.family],["Markten",r=>(r.markets||[]).join(", ")],["PF",r=>Number(r.profit_factor||0).toFixed(3)],["Stress-PF",r=>r.stressed_profit_factor==null?"ontbreekt":Number(r.stressed_profit_factor).toFixed(3)],["Holdout-PF",r=>r.holdout_profit_factor==null?"ontbreekt":Number(r.holdout_profit_factor).toFixed(3)],["Trades",r=>r.trade_count],["Max DD",r=>pct(r.maximum_drawdown)],["MC P95 DD",r=>pct(r.monte_carlo_p95_drawdown)],["Waarschuwingen",r=>(r.capital_scaling_warnings||[]).join(", ")],["Goedkeuringscommando",r=>r.approval_command]]);
const opp=Array.isArray(snapshot.opportunities)?snapshot.opportunities:Object.values(snapshot.opportunities||{});document.querySelector("#opportunities").innerHTML=table(opp.slice(0,50),[["Rang",r=>r.rank],["Markt",r=>r.market],["Strategie",r=>r.strategy||r.strategy_id||r.strategy_dna_hash],["TF",r=>r.timeframe],["Status",r=>r.status||r.reason||r.risk_level_block_reason],["Score",r=>Number(r.score||0).toFixed(1)],["Trigger",r=>r.trigger],["Afstand",r=>pct(r.distance_to_trigger)],["Stop",r=>r.stop],["TP1",r=>r.target_1],["Regime",r=>r.regime],["Conflict",r=>(r.timeframe_conflicts||[]).join(", ")]]);
document.querySelector("#signalCards").innerHTML=renderSignalCards(snapshot.signal_cards||[]);
document.querySelector("#trending").innerHTML=renderTrending(tr.rows||[]);
const bg=mm.btc_gex||{},eg=mm.eth_gex||{},of=mm.orderflow||{};const flowReady=Number(of.ready_market_count||0),flowRequested=Number(of.requested_market_count||0);const gexCards=[["BTC GEX",bg.regime||bg.status||"DATA_PENDING",bg.fresh?"ok":"warn"],["BTC gamma flip",price(bg.gamma_flip),""],["BTC call / put wall",`${price(bg.call_wall)} / ${price(bg.put_wall)}`,""],["ETH GEX",eg.regime||eg.status||"DATA_PENDING",eg.fresh?"ok":"warn"],["ETH gamma flip",price(eg.gamma_flip),""],["ETH call / put wall",`${price(eg.call_wall)} / ${price(eg.put_wall)}`,""],["Spot-flow dekking",`${flowReady}/${flowRequested} READY`,flowRequested>0&&flowReady===flowRequested?"ok":"warn"],["Flow data-gaps",Number(of.data_gap_market_count||0),Number(of.data_gap_market_count||0)===0?"ok":"warn"]];document.querySelector("#gexCards").innerHTML=gexCards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div></div>`).join("");const flowRows=Object.entries(mm.markets||{}).map(([market,v])=>({market,...(v.orderflow||{}),gex_scope:v.gex_scope,gex_regime:v.gex?.regime}));document.querySelector("#orderflow").innerHTML=table(flowRows,[["Markt",r=>r.market],["Status",r=>r.status],["GEX-regime",r=>r.gex_regime],["Scope",r=>r.gex_scope],["CVD z",r=>r.spot_cvd_robust_zscore==null?"—":Number(r.spot_cvd_robust_zscore).toFixed(2)],["OFI 1h",r=>r.horizons?.["1h"]?.ofi_normalized_mean==null?"—":Number(r.horizons["1h"].ofi_normalized_mean).toFixed(3)],["OBI top10",r=>r.orderbook_imbalance_top_10==null?"—":Number(r.orderbook_imbalance_top_10).toFixed(3)],["Spread bps",r=>r.spread_bps==null?"—":Number(r.spread_bps).toFixed(2)],["Absorptie",r=>r.bullish_absorption_score==null?"—":Number(r.bullish_absorption_score).toFixed(2)]]);
const usdt=sl.usdt||{},usdc=sl.usdc||{},agg=sl.aggregate||{};const stableCards=[["Liquiditeitsregime",sl.state||"DATA_PENDING",sl.state==="EXPANDING"?"ok":sl.state==="DRAINING"?"bad":"warn"],["USDT market cap",eur(usdt.market_cap_eur),""],["USDT 1h / 24h",`${pct(usdt.market_cap_change_1h)} / ${pct(usdt.market_cap_change_24h)}`,""],["USDC market cap",eur(usdc.market_cap_eur),""],["USDC 1h / 24h",`${pct(usdc.market_cap_change_1h)} / ${pct(usdc.market_cap_change_24h)}`,""],["Stablecoins totaal",agg.total_market_cap_usd==null?"—":"$"+compact(agg.total_market_cap_usd),""],["Totaal 1d / 7d",`${pct(agg.change_1d)} / ${pct(agg.change_7d)}`,""],["Risk multiplier",sl.risk_multiplier==null?"—":Number(sl.risk_multiplier).toFixed(2)+"×",""]];document.querySelector("#stablecoinCards").innerHTML=stableCards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div></div>`).join("");
const targetProgress=today.target_eur>0&&today.cash_flow_adjusted_pnl_eur!=null?Number(today.cash_flow_adjusted_pnl_eur)/Number(today.target_eur):null;const pnlCards=[["Geobserveerde dagen",cal.observed_days??0,""],["Gevalideerde dagen",cal.validated_days??0,"ok"],["Winstdagen",cal.positive_days??0,"ok"],["Verliesdagen",cal.negative_days??0,(cal.negative_days??0)>0?"bad":""],["Ongeverifieerde dagen",cal.unverified_days??0,(cal.unverified_days??0)>0?"warn":"ok"],["Laatste dag",today.date||"—",""],["Gevalideerde P&L",eur(today.cash_flow_adjusted_pnl_eur),today.status==="PROFIT"?"ok":today.status==="LOSS"?"bad":""],["Betaalde fees",eur(today.fees_eur),Number(today.fees_eur||0)>0?"warn":""],["Fill-events",today.fill_events??0,""],["Ruwe snapshotmutatie",eur(today.raw_cash_flow_adjusted_pnl_eur),today.status==="UNVERIFIED"?"warn":""],["Zacht dagdoel (geschaald)",eur(today.target_eur),""],["Voortgang zacht doel",targetProgress==null?"—":Number(targetProgress*100).toFixed(1)+"%",targetProgress>=1?"ok":targetProgress<0?"bad":""],["Externe kasstroom",eur(today.external_capital_flow_eur),today.external_capital_flow_eur?"warn":""],["P&L-kwaliteit",today.pnl_quality||"—",today.pnl_quality==="UNEXPLAINED_CAPITAL_FLOW_OR_VALUATION_JUMP"?"warn":"ok"]];document.querySelector("#pnlSummary").innerHTML=pnlCards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div></div>`).join("");document.querySelector("#pnlCalendar").innerHTML=renderCalendars(cal);document.querySelector("#pnlDays").innerHTML=table([...(cal.rows||[])].reverse(),[["Datum",r=>r.date],["Start equity",r=>eur(r.day_start_equity_eur)],["Eind equity",r=>eur(r.day_end_equity_eur)],["Ruwe account-P&L",r=>eur(r.account_wide_mtm_pnl_eur)],["Fees",r=>eur(r.fees_eur)],["Fills",r=>r.fill_events],["Externe kasstroom",r=>eur(r.external_capital_flow_eur)],["Ruwe gecorrigeerde mutatie",r=>eur(r.raw_cash_flow_adjusted_pnl_eur)],["Gevalideerde P&L",r=>eur(r.cash_flow_adjusted_pnl_eur)],["Zacht dagdoel",r=>eur(r.target_eur)],["Rendement",r=>pct(r.return_fraction)],["Kwaliteit",r=>r.pnl_quality],["Orders",r=>r.orders_submitted]]);document.querySelector("#pnlMonths").innerHTML=table([...(cal.months||[])].reverse(),[["Maand",r=>r.month],["Som gevalideerde dag-P&L",r=>eur(r.pnl_eur)],["Winstdagen",r=>r.positive_days],["Verliesdagen",r=>r.negative_days],["Vlak",r=>r.flat_days],["Ongeverifieerd",r=>r.unverified_days],["Gevalideerde dagen",r=>r.validated_days],["Alle snapshots",r=>r.observed_days]]);
const tf=Object.entries(snapshot.timeframe_status?.timeframes||{}).map(([timeframe,v])=>({timeframe,...v}));document.querySelector("#timeframes").innerHTML=table(tf,[["TF",r=>r.timeframe],["Totaal",r=>r.total_strategies],["Families",r=>r.independent_families],["Historisch positief",r=>r.positive_historical_candidates],["Shadow",r=>r.shadow],["Paper",r=>r.paper],["Micro live",r=>r.micro_live],["Economisch geblokkeerd",r=>r.economic_blocked],["Evaluaties",r=>r.strategies_evaluated],["Near",r=>r.near_entries],["Actionable",r=>r.actionable_entries]]);
document.querySelector("#mtfMatrix").innerHTML=table(snapshot.multi_timeframe_matrix?.rows||[],[["Rang",r=>r.rank],["Markt",r=>r.market],["Tier",r=>r.market_tier],["15m",r=>r.trends?.["15m"]],["1h",r=>r.trends?.["1h"]],["2h",r=>r.trends?.["2h"]],["4h",r=>r.trends?.["4h"]],["1d",r=>r.trends?.["1d"]],["1W",r=>r.trends?.["1W"]],["Alignment",r=>Number(r.alignment_score||0).toFixed(0)+"%"],["Actieve families",r=>(r.active_families||[]).join(", ")||"—"],["Beste setup",r=>r.best_opportunity||r.best_opportunity_status]]);
const cap=[["EUR cash",eur(cu.eur_cash),""],["Inventory",eur(cu.inventory_exposure_eur),""],["Strategie-exposure",eur(cu.strategy_exposure_eur),""],["Ongebruikt budget",eur(cu.unused_strategy_risk_budget_eur),""],["Stage",cu.current_stage||"—",""],["Autoscale",cu.autoscale?"Aan":"Uit",cu.autoscale?"warn":"ok"]];document.querySelector("#capital").innerHTML=cap.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value ${c[2]}">${c[1]}</div></div>`).join("");
document.querySelector("#allocation").innerHTML=table(pa.rows||[],[["Asset",r=>r.asset],["Actueel",r=>eur(r.current_value_eur)],["Actueel %",r=>pct(r.current_weight)],["Regime-doel",r=>pct(r.target_weight)],["Doelwaarde",r=>eur(r.target_value_eur)],["Verschil",r=>eur(r.delta_value_eur)],["Actie",r=>r.action],["Signaal",r=>r.signal_status||"—"],["Strategie",r=>r.strategy_id||"—"],["TF",r=>r.timeframe||"—"]]);
const reallocationRows=ir.market?[ir]:[];document.querySelector("#inventoryReallocation").innerHTML=table(reallocationRows,[["Markt",r=>r.market],["Status",r=>r.status],["Actie",r=>`${r.side||'—'} ${r.order_type||'—'} ${r.time_in_force||''}`],["Quantity",r=>r.quantity],["Limiet",r=>price(r.limit_price)],["Bruto schatting",r=>eur(r.estimated_gross_eur)],["Slippage bps",r=>r.liquidity?.estimated_sell_slippage_bps==null?'—':Number(r.liquidity.estimated_sell_slippage_bps).toFixed(2)],["Risk",r=>r.risk?.approved?'GOEDGEKEURD':'GEBLOKKEERD'],["Orders",r=>`${r.orders_submitted||0} verstuurd`],["Controle",r=>r.mode]]);
const mf=Object.entries(mc.features||{}).map(([feature,value])=>({feature,value}));document.querySelector("#macro").innerHTML=table(mf,[["Feature",r=>r.feature],["Waarde",r=>r.value]]);
document.querySelector("#rotation").innerHTML=table(at.top_5_rotation||[],[["Rang",r=>r.rank],["Markt",r=>r.market],["Score",r=>Number(r.rotation_score||0).toFixed(1)],["Besluit",r=>r.decision],["Regime",r=>r.regime]]);
const paperPositions=Object.entries(paper.positions||{}).map(([dna,value])=>({dna,...value}));document.querySelector("#paperPositions").innerHTML=table(paperPositions,[["Markt",r=>r.market],["Strategie",r=>r.strategy_id],["TF",r=>r.timeframe],["DNA",r=>(r.dna||r.strategy_dna||"").slice(0,16)],["Quantity",r=>r.quantity],["Entry",r=>price(r.entry_price)],["Mark",r=>price(r.current_price)],["P&L bruto",r=>eur(r.gross_unrealized_pnl_eur)],["Rendement",r=>pct(r.gross_return_fraction)],["Stop",r=>price(r.stop_loss)],["TP1",r=>price(r.take_profit_1)],["TP2",r=>price(r.take_profit_2)],["Exitprofiel",r=>r.exit_profile],["Geopend",r=>r.opened_at]]);
document.querySelector("#positions").innerHTML=table(a.positions||[],[["Asset",r=>r.symbol],["Beschikbaar",r=>r.available],["In order",r=>r.in_order],["Totaal",r=>r.total]]);
document.querySelector("#orders").innerHTML=table([...(snapshot.events?.orders||[]),...(snapshot.events?.fills||[])].slice(-30),[["Event",r=>r.event||r.event_type],["Markt",r=>r.market||r.payload?.market],["Status",r=>r.status||r.payload?.status],["Tijd",r=>r.recorded_at||r.timestamp]]);
document.querySelector("#research").innerHTML=table(snapshot.research?.candidates||[],[["Strategie",r=>r.strategy_id],["TF",r=>r.timeframe],["Trades",r=>r.normal?.trade_count],["PF",r=>Number(r.normal?.profit_factor||0).toFixed(3)],["Net",r=>pct(r.normal?.panel_net_return)],["OOS",r=>pct(r.out_of_sample?.panel_net_return)],["Folds",r=>`${r.walk_forward?.positive_folds||0}/5`],["Status",r=>r.validation_pass?"PASS":(Object.entries(r.gates||{}).filter(x=>!x[1]).map(x=>x[0]).join(", "))]]);
document.querySelector("#system").textContent=JSON.stringify({system:s,crypto_maturity:maturity,active_trading:at,telegram:snapshot.telegram,candle_health:snapshot.candle_health,events:snapshot.events?.errors},null,2);document.querySelector("#updated").textContent="Bijgewerkt "+snapshot.generated_at}
async function loadCandles(){const m=document.querySelector("#marketSelect").value,tf=document.querySelector("#tfSelect").value;if(!m)return;const d=await(await fetch(`/api/candles?market=${encodeURIComponent(m)}&timeframe=${encodeURIComponent(tf)}`)).json();draw(d)}
function draw(d){const c=document.querySelector("#chart"),x=c.getContext("2d"),w=c.width,h=c.height;x.clearRect(0,0,w,h);const rows=d.candles||[];if(!rows.length)return;const hi=Math.max(...rows.map(r=>r.high)),lo=Math.min(...rows.map(r=>r.low)),pad=25,sy=v=>pad+(hi-v)/(hi-lo)*(h-pad*2),dx=(w-pad*2)/rows.length;x.strokeStyle="#183138";for(let i=0;i<6;i++){const y=pad+i*(h-pad*2)/5;x.beginPath();x.moveTo(pad,y);x.lineTo(w-pad,y);x.stroke()}rows.forEach((r,i)=>{const cx=pad+(i+.5)*dx,up=r.close>=r.open,col=up?"#37e6a1":"#ff657a";x.strokeStyle=col;x.fillStyle=col;x.beginPath();x.moveTo(cx,sy(r.high));x.lineTo(cx,sy(r.low));x.stroke();const y=Math.min(sy(r.open),sy(r.close)),bh=Math.max(1,Math.abs(sy(r.open)-sy(r.close)));x.fillRect(cx-dx*.3,y,Math.max(1,dx*.6),bh)});(d.levels||[]).forEach(l=>{const y=sy(l.price);x.strokeStyle=l.type==="STOP"?"#ff657a":l.type==="TAKE_PROFIT"?"#37e6a1":"#ffc857";x.setLineDash([7,5]);x.beginPath();x.moveTo(pad,y);x.lineTo(w-pad,y);x.stroke();x.setLineDash([])});const byTime=Object.fromEntries(rows.map((r,i)=>[r.time,i]));(d.markers||[]).forEach(m=>{const i=byTime[m.time];if(i==null)return;const cx=pad+(i+.5)*dx,y=sy(m.price),colors={FRACTAL_HIGH:"#57b8ff",FRACTAL_LOW:"#ffc857",ENTRY:"#37e6a1",EXIT:"#ff657a"};x.fillStyle=colors[m.type]||"#fff";x.beginPath();x.arc(cx,y,m.type==="ENTRY"||m.type==="EXIT"?6:3,0,Math.PI*2);x.fill()})}
document.querySelector("#marketSelect").onchange=loadCandles;document.querySelector("#tfSelect").onchange=loadCandles;
document.querySelectorAll("[data-action]").forEach(b=>b.onclick=async()=>{const action=b.dataset.action,payload={};if(action==="emergency-stop"){payload.reason=prompt("Reden voor emergency stop:")||"";payload.confirm=prompt('Typ exact: EMERGENCY STOP')||"";if(payload.confirm!=="EMERGENCY STOP")return}const token=document.querySelector('meta[name="csrf-token"]').content;const r=await fetch(`/api/control/${action}`,{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":token},body:JSON.stringify(payload)});alert(JSON.stringify(await r.json(),null,2));refresh()});
refresh();setInterval(refresh,5000);
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    settings: Settings
    csrf_token: str

    def log_message(self, format_: str, *args: Any) -> None:
        # Avoid request query strings and keep the production log sanitized.
        return

    def _json(
        self,
        payload: Any,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_DOCUMENT.replace(
                "__CSRF__",
                html.escape(self.csrf_token, quote=True),
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/snapshot":
            self._json(build_ui_snapshot(self.settings))
            return
        if parsed.path == "/api/candles":
            query = parse_qs(parsed.query)
            try:
                payload = candle_payload(
                    self.settings,
                    str((query.get("market") or [""])[0]),
                    str((query.get("timeframe") or [""])[0]),
                )
            except (OSError, ValueError) as exc:
                self._json(
                    {"status": "ERROR", "reason": type(exc).__name__},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._json(payload)
            return
        if parsed.path == "/health":
            self._json(ui_status(self.settings))
            return
        self._json(
            {"status": "NOT_FOUND"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/control/"):
            self._json(
                {"status": "NOT_FOUND"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            self._json(
                {"status": "FORBIDDEN"},
                status=HTTPStatus.FORBIDDEN,
            )
            return
        supplied = self.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(supplied, self.csrf_token):
            self._json(
                {"status": "CSRF_REJECTED"},
                status=HTTPStatus.FORBIDDEN,
            )
            return
        length = min(int(self.headers.get("Content-Length") or 0), 8_192)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(
                {"status": "INVALID_JSON"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        action = parsed.path.rsplit("/", maxsplit=1)[-1]
        try:
            payload = self._control(action, dict(body))
            status = HTTPStatus.OK
        except (PermissionError, ValueError) as exc:
            payload = {
                "status": "REJECTED",
                "reason": type(exc).__name__,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
            status = HTTPStatus.BAD_REQUEST
        append_jsonl(
            _paths(self.settings)["audit"],
            {
                "timestamp": utc_iso(),
                "action": action,
                "status": payload.get("status"),
                "reason_hash": stable_hash(
                    str(body.get("reason") or ""),
                    length=16,
                ),
                "remote": "LOOPBACK",
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )
        self._json(payload, status=status)

    def _control(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        from core.autonomous_live import AutonomousLiveSupervisor

        supervisor = AutonomousLiveSupervisor(self.settings)
        if action == "pause":
            return supervisor.pause()
        if action == "resume":
            return supervisor.resume()
        if action == "reconcile":
            return asyncio.run(supervisor.reconcile())
        if action == "emergency-stop":
            reason = str(body.get("reason") or "").strip()
            if body.get("confirm") != "EMERGENCY STOP":
                raise PermissionError("double confirmation required")
            if len(reason) < 5:
                raise ValueError("emergency-stop reason is required")
            KillSwitch(
                self.settings.paths.checkpoints_dir / "kill_switch.json"
            ).activate(f"UI_EMERGENCY_STOP:{reason[:120]}")
            paused = supervisor.pause()
            return {
                **paused,
                "status": "EMERGENCY_STOPPED",
                "kill_switch_active": True,
                "orders_generated": 0,
                "orders_submitted": 0,
            }
        raise ValueError("unknown UI control")


def ui_status(settings: Settings) -> dict[str, Any]:
    paths = _paths(settings)
    lock = _safe_read(paths["lock"])
    pid = int(lock.get("pid") or 0)
    health = _safe_read(paths["health"])
    running = _pid_alive(pid)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING" if running else "STOPPED",
        "host": HOST,
        "port": PORT,
        "url": f"http://{HOST}:{PORT}",
        "pid": pid if running else None,
        "started_at": health.get("started_at"),
        "last_health_at": health.get("last_health_at"),
        "local_only": True,
        "csrf_enabled": True,
        "control_actions": [
            "pause",
            "resume",
            "reconcile",
            "emergency-stop",
        ],
        "direct_order_construction": False,
        "orders_generated": 0,
        "orders_submitted": 0,
    }


def start_ui(settings: Settings) -> dict[str, Any]:
    status = ui_status(settings)
    if status["status"] == "RUNNING":
        return {**status, "start_status": "ALREADY_RUNNING"}
    paths = _paths(settings)
    stream = paths["log"].open("a", encoding="utf-8")
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    process = subprocess.Popen(
        [
            sys.executable,
            str(settings.paths.project_root / "main.py"),
            "ui",
            "run",
        ],
        cwd=settings.paths.project_root,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=stream,
        creationflags=creation_flags,
        close_fds=True,
    )
    stream.close()
    for _ in range(50):
        time.sleep(0.1)
        status = ui_status(settings)
        if status["status"] == "RUNNING":
            return {**status, "start_status": "STARTED"}
        if process.poll() is not None:
            break
    return {
        **ui_status(settings),
        "start_status": "FAILED",
        "spawned_pid": process.pid,
    }


def stop_ui(settings: Settings) -> dict[str, Any]:
    status = ui_status(settings)
    pid = int(status.get("pid") or 0)
    if not pid:
        return {**status, "stop_status": "ALREADY_STOPPED"}
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        time.sleep(0.1)
        if not _pid_alive(pid):
            return {**ui_status(settings), "stop_status": "STOPPED"}
    return {**ui_status(settings), "stop_status": "STOP_PENDING"}


def serve_ui(settings: Settings) -> None:
    paths = _paths(settings)
    if paths["lock"].is_file():
        existing = _safe_read(paths["lock"])
        if _pid_alive(int(existing.get("pid") or 0)):
            raise RuntimeError("local trading UI already running")
        paths["lock"].unlink(missing_ok=True)
    token = secrets.token_urlsafe(32)
    token_hash = stable_hash(token, length=64)
    started = utc_iso()
    atomic_write_json(
        paths["lock"],
        {
            "pid": os.getpid(),
            "started_at": started,
            "host": HOST,
            "port": PORT,
        },
    )
    atomic_write_json(
        paths["health"],
        {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "pid": os.getpid(),
            "host": HOST,
            "port": PORT,
            "url": f"http://{HOST}:{PORT}",
            "started_at": started,
            "last_health_at": started,
            "local_only": True,
            "csrf_enabled": True,
            "csrf_token_hash": token_hash,
            "csrf_token_stored": False,
            "direct_order_construction": False,
            "orders_generated": 0,
            "orders_submitted": 0,
        },
    )
    handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {"settings": settings, "csrf_token": token},
    )
    server = ThreadingHTTPServer((HOST, PORT), handler)
    stop_event = threading.Event()

    def stop_handler(_signum: int, _frame: Any) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop_handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        paths["lock"].unlink(missing_ok=True)
        atomic_write_json(
            paths["health"],
            {
                **_safe_read(paths["health"]),
                "status": "STOPPED",
                "stopped_at": utc_iso(),
                "last_health_at": utc_iso(),
                "orders_generated": 0,
                "orders_submitted": 0,
            },
        )


__all__ = [
    "HOST",
    "PORT",
    "build_ui_snapshot",
    "candle_payload",
    "serve_ui",
    "start_ui",
    "stop_ui",
    "ui_status",
]
