"""Daily causal audit of detected, converted and missed market moves."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from config.settings import Settings
from utils.common import append_jsonl, atomic_write_json, stable_hash, utc_now

SIGNIFICANT_MOVE_PCT = 3.0
COUNTERFACTUAL_NOTIONAL_EUR = Decimal("5")
STATE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "DISCOVERED",
            "WATCHING",
            "ARMED",
            "ENTRY_READY",
            "ORDER_INTENT_CREATED",
            "ORDER_SUBMITTED",
            "PARTIALLY_FILLED",
            "FILLED",
            "MANAGING",
            "EXITING",
            "CLOSED",
            "INVALIDATED",
            "EXPIRED",
        )
    )
}
TERMINAL_NON_EXECUTION_STATES = {"INVALIDATED", "EXPIRED"}


def _furthest_progress_state(
    transitions: Iterable[Mapping[str, Any]],
) -> str:
    """Return the furthest reached stage without erasing prior readiness.

    INVALIDATED and EXPIRED are terminal outcomes, not higher execution
    achievements.  A later rejected sibling opportunity must therefore never
    hide that another opportunity for the same market already reached
    ENTRY_READY or an exchange lifecycle state.
    """

    states = [
        str(row.get("to_state") or "DISCOVERED") for row in transitions
    ]
    progress = [
        state for state in states if state not in TERMINAL_NON_EXECUTION_STATES
    ]
    selected = progress or states
    return max(
        selected,
        default="NONE",
        key=lambda state: STATE_ORDER.get(state, -1),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        selected = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if selected.tzinfo is None:
        selected = selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def _decimal(value: object) -> Decimal | None:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _is_live_strategy_entry_event(row: Mapping[str, Any]) -> bool:
    """Return true only for canonical BUY events tied to a strategy signal.

    Operator inventory maintenance and exit orders are real exchange activity,
    but they are not evidence that the opportunity-to-entry funnel is
    converting.  Keeping them out of the cadence metric prevents a manual TAO
    rebalance or a protective sell from being reported as a new strategy trade.
    """

    payload = row.get("payload") or {}
    strategy_id = str(payload.get("strategy_id") or "")
    return bool(
        payload.get("signal_id")
        and strategy_id
        and not strategy_id.startswith("OPERATOR_")
        and str(payload.get("side") or "").upper() == "BUY"
    )


def _paper_fill_evidence(
    settings: Settings,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reconcile paper fills by canonical signal identity.

    The execution ledger stores fills and the originating signal in separate
    records.  Joining them through ``intent_id`` keeps paper evidence distinct
    from live conversion and avoids inferring PnL from lifecycle labels.
    """

    path = (
        settings.paths.output_dir
        / "paper"
        / "event_driven_playbook_execution.jsonl"
    )
    rows = _read_jsonl(path)
    intent_to_signal: dict[str, dict[str, str]] = {}
    fills: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        if row.get("event_type") == "ORDER_RESULT":
            record = payload.get("record") or {}
            intent = record.get("intent") or {}
            intent_id = str(intent.get("intent_id") or "")
            signal_id = str(intent.get("signal_id") or "")
            if intent_id and signal_id:
                intent_to_signal[intent_id] = {
                    "signal_id": signal_id,
                    "playbook_id": str(intent.get("strategy_id") or "UNKNOWN"),
                    "playbook_dna": str(intent.get("strategy_dna_hash") or ""),
                    "reason": ",".join(str(value) for value in intent.get("reason_codes") or []),
                }
        elif row.get("event_type") == "FILL":
            fills.append(dict(payload))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped = 0
    for fill in fills:
        intent_id = str(fill.get("intent_id") or "")
        identity = intent_to_signal.get(intent_id)
        if not identity:
            unmapped += 1
            continue
        grouped[identity["signal_id"]].append({**fill, **identity})

    outcomes: dict[str, dict[str, Any]] = {}
    for signal_id, signal_fills in grouped.items():
        buy_qty = Decimal("0")
        sell_qty = Decimal("0")
        gross_buys = Decimal("0")
        gross_sells = Decimal("0")
        fees = Decimal("0")
        for fill in signal_fills:
            price = _decimal(fill.get("price")) or Decimal("0")
            quantity = _decimal(fill.get("quantity")) or Decimal("0")
            fee = _decimal(fill.get("fee_eur")) or Decimal("0")
            fees += fee
            if str(fill.get("side") or "").upper() == "BUY":
                buy_qty += quantity
                gross_buys += price * quantity
            elif str(fill.get("side") or "").upper() == "SELL":
                sell_qty += quantity
                gross_sells += price * quantity
        closed = buy_qty > 0 and sell_qty >= buy_qty * Decimal("0.999999")
        gross_pnl = gross_sells - gross_buys if closed else None
        net_pnl = gross_pnl - fees if gross_pnl is not None else None
        outcomes[signal_id] = {
            "evidence": "CANONICAL_PAPER_FILL_LEDGER",
            "paper_fill_count": len(signal_fills),
            "paper_buy_quantity": str(buy_qty),
            "paper_sell_quantity": str(sell_qty),
            "paper_closed": closed,
            "paper_fees_eur": float(fees),
            "paper_gross_pnl_eur": float(gross_pnl) if gross_pnl is not None else None,
            "paper_net_pnl_eur": float(net_pnl) if net_pnl is not None else None,
            "paper_average_buy_fill_price": (
                float(gross_buys / buy_qty) if buy_qty > 0 else None
            ),
            "paper_average_sell_fill_price": (
                float(gross_sells / sell_qty) if sell_qty > 0 else None
            ),
            "playbook_id": signal_fills[0]["playbook_id"],
            "playbook_dna": signal_fills[0].get("playbook_dna") or "",
            "exit_reason": next(
                (
                    str(fill.get("reason"))
                    for fill in reversed(signal_fills)
                    if str(fill.get("side") or "").upper() == "SELL"
                ),
                None,
            ),
        }
    closed = [row for row in outcomes.values() if row["paper_closed"]]
    gross_values = [float(row["paper_gross_pnl_eur"]) for row in closed]
    net_values = [float(row["paper_net_pnl_eur"]) for row in closed]
    by_playbook: dict[str, dict[str, Any]] = {}
    for playbook_id in sorted(
        {str(row.get("playbook_id") or "UNKNOWN") for row in closed}
    ):
        selected = [
            row
            for row in closed
            if str(row.get("playbook_id") or "UNKNOWN") == playbook_id
        ]
        selected_net = [float(row["paper_net_pnl_eur"]) for row in selected]
        gross_wins = sum(value for value in selected_net if value > 0)
        gross_losses = abs(sum(value for value in selected_net if value < 0))
        by_playbook[playbook_id] = {
            "closed_round_trips": len(selected),
            "winning_round_trips": sum(value > 0 for value in selected_net),
            "losing_round_trips": sum(value < 0 for value in selected_net),
            "win_rate": (
                sum(value > 0 for value in selected_net) / len(selected_net)
                if selected_net
                else None
            ),
            "paper_fees_eur": sum(
                float(row.get("paper_fees_eur") or 0.0) for row in selected
            ),
            "paper_net_pnl_eur": sum(selected_net),
            "paper_net_expectancy_eur": (
                sum(selected_net) / len(selected_net)
                if selected_net
                else None
            ),
            "closed_position_profit_factor": (
                gross_wins / gross_losses
                if gross_losses > 0
                else None
            ),
        }
    by_playbook_dna: dict[str, dict[str, Any]] = {}
    for playbook_dna in sorted(
        {
            str(row.get("playbook_dna") or "")
            for row in closed
            if row.get("playbook_dna")
        }
    ):
        selected = [
            row
            for row in closed
            if str(row.get("playbook_dna") or "") == playbook_dna
        ]
        selected_net = [float(row["paper_net_pnl_eur"]) for row in selected]
        gross_wins = sum(value for value in selected_net if value > 0)
        gross_losses = abs(sum(value for value in selected_net if value < 0))
        by_playbook_dna[playbook_dna] = {
            "playbook_id": str(selected[0].get("playbook_id") or "UNKNOWN"),
            "playbook_dna": playbook_dna,
            "closed_round_trips": len(selected),
            "winning_round_trips": sum(value > 0 for value in selected_net),
            "losing_round_trips": sum(value < 0 for value in selected_net),
            "win_rate": (
                sum(value > 0 for value in selected_net) / len(selected_net)
                if selected_net
                else None
            ),
            "paper_fees_eur": sum(
                float(row.get("paper_fees_eur") or 0.0) for row in selected
            ),
            "paper_net_pnl_eur": sum(selected_net),
            "paper_net_expectancy_eur": (
                sum(selected_net) / len(selected_net)
                if selected_net
                else None
            ),
            "closed_position_profit_factor": (
                gross_wins / gross_losses if gross_losses > 0 else None
            ),
        }
    summary = {
        "source": str(path),
        "mapped_fill_count": sum(int(row["paper_fill_count"]) for row in outcomes.values()),
        "unmapped_fill_count": unmapped,
        "closed_round_trips": len(closed),
        "paper_gross_expectancy_eur": (
            sum(gross_values) / len(gross_values) if gross_values else None
        ),
        "paper_net_expectancy_eur": (
            sum(net_values) / len(net_values) if net_values else None
        ),
        "by_playbook": by_playbook,
        "by_playbook_dna": by_playbook_dna,
        "average_slippage_bps": None,
    }
    return outcomes, summary


def _counterfactual_outcome(
    frame: pd.DataFrame,
    *,
    detected_at: datetime | None,
    entry_price: object,
    stop_loss: object,
    take_profit_1: object,
) -> dict[str, Any]:
    """Measure observable post-detection price paths without claiming a fill."""

    if detected_at is None or frame.empty:
        return {
            "status": "UNAVAILABLE_NO_PROSPECTIVE_DETECTION",
            "theoretical_entry_price": None,
            "maximum_favorable_move_after_detection_pct": None,
            "maximum_adverse_move_after_detection_pct": None,
            "counterfactual_max_gross_pnl_eur_at_5": None,
            "tp1_or_stop_outcome": "UNRESOLVED",
            "false_positive": None,
        }
    selected = frame.loc[frame["timestamp"] >= detected_at].copy()
    if selected.empty:
        return {
            "status": "UNAVAILABLE_NO_POST_DETECTION_CANDLE",
            "theoretical_entry_price": None,
            "maximum_favorable_move_after_detection_pct": None,
            "maximum_adverse_move_after_detection_pct": None,
            "counterfactual_max_gross_pnl_eur_at_5": None,
            "tp1_or_stop_outcome": "UNRESOLVED",
            "false_positive": None,
        }
    entry = _decimal(entry_price) or _decimal(selected.iloc[0]["close"])
    if entry is None or entry <= 0:
        return {"status": "UNAVAILABLE_INVALID_ENTRY", "false_positive": None}
    highs = selected["high"].astype(float)
    lows = selected["low"].astype(float)
    favorable = (float(highs.max()) / float(entry) - 1.0) * 100.0
    adverse = (float(lows.min()) / float(entry) - 1.0) * 100.0
    stop = _decimal(stop_loss)
    target = _decimal(take_profit_1)
    outcome = "UNRESOLVED"
    first_hit_at: str | None = None
    for row in selected.itertuples(index=False):
        stop_hit = stop is not None and Decimal(str(row.low)) <= stop
        target_hit = target is not None and Decimal(str(row.high)) >= target
        if stop_hit and target_hit:
            outcome = "AMBIGUOUS_SAME_CANDLE"
            first_hit_at = row.timestamp.isoformat()
            break
        if stop_hit:
            outcome = "STOP_BEFORE_TP1"
            first_hit_at = row.timestamp.isoformat()
            break
        if target_hit:
            outcome = "TP1_BEFORE_STOP"
            first_hit_at = row.timestamp.isoformat()
            break
    false_positive = True if outcome == "STOP_BEFORE_TP1" else (
        False if outcome == "TP1_BEFORE_STOP" else None
    )
    return {
        "status": "COUNTERFACTUAL_NOT_A_FILL",
        "theoretical_entry_price": float(entry),
        "maximum_favorable_move_after_detection_pct": round(favorable, 6),
        "maximum_adverse_move_after_detection_pct": round(adverse, 6),
        "counterfactual_max_gross_pnl_eur_at_5": round(
            float(COUNTERFACTUAL_NOTIONAL_EUR) * max(0.0, favorable) / 100.0,
            8,
        ),
        "tp1_or_stop_outcome": outcome,
        "first_boundary_hit_at": first_hit_at,
        "false_positive": false_positive,
    }


def _load_day_frame(path: Path, *, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    except (OSError, ValueError, KeyError):
        return pd.DataFrame()
    if "timestamp" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index(names="timestamp")
        else:
            return pd.DataFrame()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[(timestamps >= start) & (timestamps <= end)].copy()
    frame["timestamp"] = timestamps.loc[frame.index]
    return frame.sort_values("timestamp")


def _market_moves(
    settings: Settings,
    *,
    start: datetime,
    end: datetime,
    markets: Iterable[str] | None,
) -> list[dict[str, Any]]:
    selected = (
        tuple(dict.fromkeys(markets))
        if markets is not None
        else tuple(
            path.name.removesuffix("_15m.parquet")
            for path in settings.paths.processed_data_dir.glob("*-EUR_15m.parquet")
        )
    )
    rows: list[dict[str, Any]] = []
    for market in selected:
        path = settings.paths.processed_data_dir / f"{market}_15m.parquet"
        frame = _load_day_frame(path, start=start, end=end)
        if frame.empty:
            continue
        origin = float(frame.iloc[0]["open"])
        if origin <= 0:
            continue
        up = (frame["high"].astype(float) / origin - 1.0) * 100.0
        down = (frame["low"].astype(float) / origin - 1.0) * 100.0
        maximum_up = float(up.max())
        maximum_down = float(down.min())
        direction = "UP" if maximum_up >= abs(maximum_down) else "DOWN"
        maximum_move = maximum_up if direction == "UP" else maximum_down
        threshold = up >= SIGNIFICANT_MOVE_PCT if direction == "UP" else down <= -SIGNIFICANT_MOVE_PCT
        detected_rows = frame.loc[threshold]
        detectable_at = (
            detected_rows.iloc[0]["timestamp"].to_pydatetime()
            if not detected_rows.empty
            else None
        )
        rows.append(
            {
                "market": market,
                "direction": direction,
                "day_open": origin,
                "latest_close": float(frame.iloc[-1]["close"]),
                "maximum_move_pct": round(maximum_move, 6),
                "significant": abs(maximum_move) >= SIGNIFICANT_MOVE_PCT,
                "detectable_at": (
                    detectable_at.isoformat() if detectable_at else None
                ),
                "data_start": frame.iloc[0]["timestamp"].isoformat(),
                "data_end": frame.iloc[-1]["timestamp"].isoformat(),
                "source": str(path),
            }
        )
    return rows


def _miss_reason(
    transitions: list[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
    *,
    furthest_state: str = "NONE",
    operational_blockers: Iterable[str] = (),
) -> str:
    def _is_data_or_sequence_blocker(value: str) -> bool:
        normalized = str(value).strip().upper()
        return normalized.startswith("STALE_") or normalized in {
            "DATA_SEQUENCE_INVALID",
            "ORDERBOOK_SEQUENCE_INVALID",
            "PRIVATE_ACCOUNT_STREAM_NOT_READY",
            "PUBLIC_MARKET_STREAM_NOT_READY",
        }

    blockers = [
        str(blocker)
        for row in transitions
        for blocker in row.get("hard_blockers") or []
    ]
    blockers.extend(str(value) for value in (current or {}).get("hard_blockers") or [])
    reasons = [
        str(reason)
        for row in transitions
        for reason in row.get("reason_codes") or []
    ]
    next_required = str((current or {}).get("next_required_condition") or "")
    if next_required:
        reasons.append(next_required)
    if (current or {}).get("live_authority_granted") is False:
        reasons.append("STRATEGY_AUTHORITY_NOT_GRANTED")
    combined = blockers + reasons + [str(value) for value in operational_blockers]
    if furthest_state == "ENTRY_READY":
        if any("UNEXPLAINED_EUR_BALANCE_CHANGE" in value for value in combined):
            return "UNEXPLAINED_EUR_BALANCE_CHANGE"
        if any("AUTHORITY" in value for value in combined):
            return "PLAYBOOK_AUTHORITY_MISSING"
        if any(_is_data_or_sequence_blocker(value) for value in combined):
            return "DATA_OR_SEQUENCE_BLOCKER"
        return "ENTRY_READY_NOT_CONVERTED"
    if any(_is_data_or_sequence_blocker(value) for value in combined):
        return "DATA_OR_SEQUENCE_BLOCKER"
    if any("SPREAD" in value or "SLIPPAGE" in value or "DEPTH" in value for value in combined):
        return "LIQUIDITY_BLOCKER"
    if any("CONFIRM" in value for value in blockers + reasons):
        return "INSUFFICIENT_REALTIME_CONFIRMATIONS"
    if transitions:
        return "SETUP_NEVER_REACHED_ENTRY_READY"
    return "NO_CAUSAL_SETUP_DETECTED"


def _gate_category(reason: object) -> str:
    """Map one exact blocker to a stable, reviewable gate family."""

    value = str(reason or "").strip().upper()
    rules = (
        (
            "DATA_QUALITY",
            ("STALE", "SEQUENCE", "STREAM_NOT_READY", "DATA_QUALITY"),
        ),
        (
            "RECONCILIATION",
            ("RECONCIL", "UNKNOWN_ORDER", "BALANCE_CHANGE"),
        ),
        (
            "AUTHORITY",
            ("AUTHORITY", "APPROVAL", "STRATEGY_DNA"),
        ),
        (
            "PORTFOLIO_RISK",
            (
                "PORTFOLIO",
                "EXPOSURE",
                "DAILY_LOSS",
                "DRAWDOWN",
                "KILL_SWITCH",
                "RISK_LIMIT",
            ),
        ),
        ("SPREAD", ("SPREAD",)),
        ("LIQUIDITY", ("LIQUID", "DEPTH", "SLIPPAGE")),
        (
            "ECR_COST",
            ("COST", "ECR", "POSITIVE_EXIT", "EXPECTED_VALUE"),
        ),
        (
            "TRIGGER",
            (
                "ENTRY_TRIGGER_NOT_CONFIRMED",
                "INSUFFICIENT_REALTIME_CONFIRMATIONS",
                "ENTRY_PERSISTENCE",
            ),
        ),
        (
            "RR_TARGET",
            ("NET_TARGET", "NET_RR", "SWING_UPSIDE", "TARGET_SPACE"),
        ),
        ("VOLATILITY", ("VOLAT", "ENTRY_CHASE", "ATR_")),
        (
            "HTF_CONTEXT",
            ("MACRO_", "HTF", "TIMEFRAME", "1H_4H_PARENT"),
        ),
        (
            "ORDERFLOW",
            (
                "CONFIRM",
                "ORDERFLOW",
                "OFI",
                "CVD",
                "MICROPRICE",
                "MLOBI",
                "TAKER",
                "ABSORPT",
            ),
        ),
        ("SCORE", ("SCORE", "PARAMETER_BAND")),
    )
    for category, needles in rules:
        if any(needle in value for needle in needles):
            return category
    return "OTHER"


def _counterfactual_net_result(
    counterfactual: Mapping[str, Any],
    economics: Mapping[str, Any],
) -> dict[str, float] | None:
    """Return a conservative boundary result after estimated roundtrip costs.

    Only an unambiguous first TP1/stop boundary is scored.  The calculation is
    explicitly a shadow estimate, never execution or fill evidence.
    """

    outcome = str(counterfactual.get("tp1_or_stop_outcome") or "")
    stop_bps = _decimal(economics.get("stop_bps"))
    costs_bps = _decimal(economics.get("roundtrip_cost_bps"))
    costs_bps = costs_bps if costs_bps is not None else Decimal("0")
    if outcome == "TP1_BEFORE_STOP":
        net_bps = _decimal(economics.get("net_target_1_bps"))
        if net_bps is None:
            entry = _decimal(counterfactual.get("theoretical_entry_price"))
            target = _decimal(economics.get("take_profit_1"))
            if entry is None or entry <= 0 or target is None:
                return None
            net_bps = (target / entry - Decimal("1")) * Decimal("10000")
            net_bps -= costs_bps
        risk_bps = (
            stop_bps + costs_bps
            if stop_bps is not None and stop_bps > 0
            else None
        )
        return {
            "net_return_pct": float(net_bps / Decimal("100")),
            "net_r": float(net_bps / risk_bps) if risk_bps else 0.0,
        }
    if outcome == "STOP_BEFORE_TP1" and stop_bps is not None:
        loss_bps = stop_bps + costs_bps
        return {
            "net_return_pct": -float(loss_bps / Decimal("100")),
            "net_r": -1.0,
        }
    return None


def build_daily_opportunity_audit(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
    markets: Iterable[str] | None = None,
) -> dict[str, Any]:
    now = (observed_at or utc_now()).astimezone(UTC)
    start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    lifecycle_path = (
        settings.paths.output_dir / "live" / "events" / "opportunity_lifecycle.jsonl"
    )
    projection_path = (
        settings.paths.output_dir / "live" / "opportunity_lifecycle_state.json"
    )
    all_lifecycle = _read_jsonl(lifecycle_path)
    lifecycle_coverage_started_at = min(
        (_timestamp(row.get("recorded_at")) for row in all_lifecycle),
        default=None,
        key=lambda value: value or now + timedelta(days=1),
    )
    lifecycle = [
        row
        for row in all_lifecycle
        if (_timestamp(row.get("recorded_at")) or start - timedelta(days=1)) >= start
        and (_timestamp(row.get("recorded_at")) or now + timedelta(days=1)) <= now
    ]
    projection = (
        json.loads(projection_path.read_text(encoding="utf-8"))
        if projection_path.is_file()
        else {}
    )
    current = {
        str(key): dict(value)
        for key, value in (projection.get("opportunities") or {}).items()
    }
    live_execution_path = (
        settings.paths.checkpoints_dir / "live_execution.jsonl"
    )
    all_canonical_live_events = _read_jsonl(live_execution_path)
    canonical_live_events = [
        row
        for row in all_canonical_live_events
        if (_timestamp(row.get("recorded_at")) or start - timedelta(days=1))
        >= start
        and (_timestamp(row.get("recorded_at")) or now + timedelta(days=1))
        <= now
    ]
    paper_outcomes, paper_summary = _paper_fill_evidence(settings)
    playbook_authority_path = (
        settings.paths.project_root / "config" / "live_playbook_authority.json"
    )
    playbook_authority = (
        json.loads(playbook_authority_path.read_text(encoding="utf-8"))
        if playbook_authority_path.is_file()
        else {}
    )
    account_health_path = (
        settings.paths.output_dir / "operations" / "live_account_health.json"
    )
    account_health = (
        json.loads(account_health_path.read_text(encoding="utf-8"))
        if account_health_path.is_file()
        else {}
    )
    live_state_path = (
        settings.paths.output_dir / "live" / "event_driven_execution_state.json"
    )
    live_state = (
        json.loads(live_state_path.read_text(encoding="utf-8"))
        if live_state_path.is_file()
        else {}
    )
    operational_blockers = list(account_health.get("entry_blockers") or [])
    if playbook_authority.get("active") is not True:
        operational_blockers.append("PLAYBOOK_LIVE_AUTHORITY_REQUIRED")
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lifecycle:
        market = str(row.get("market") or "")
        identity = str(row.get("opportunity_id") or "")
        if market:
            by_market[market].append(row)
        if identity:
            by_identity[identity].append(row)
    moves = _market_moves(settings, start=start, end=now, markets=markets)
    significant: list[dict[str, Any]] = []
    for move in moves:
        if not move["significant"]:
            continue
        transitions = by_market.get(str(move["market"]), [])
        first_transition = min(
            transitions,
            default=None,
            key=lambda row: _timestamp(row.get("recorded_at"))
            or now + timedelta(days=1),
        )
        first = (
            _timestamp(first_transition.get("recorded_at"))
            if first_transition
            else None
        )
        first_decision = (
            _timestamp(
                dict(first_transition.get("feature_snapshot") or {}).get(
                    "decision_timestamp"
                )
            )
            if first_transition
            else None
        )
        detectable = _timestamp(move.get("detectable_at"))
        prospectively_observable = not (
            detectable is not None
            and lifecycle_coverage_started_at is not None
            and detectable < lifecycle_coverage_started_at
        )
        identities = {
            str(row.get("opportunity_id"))
            for row in transitions
            if row.get("opportunity_id")
        }
        furthest = _furthest_progress_state(transitions)
        related_current = next(
            (
                row
                for identity, row in current.items()
                if identity in identities
            ),
            None,
        )
        source_frame = _load_day_frame(
            Path(str(move["source"])),
            start=start,
            end=now,
        )
        opportunity_outcomes: list[dict[str, Any]] = []
        for identity in sorted(identities):
            opportunity = current.get(identity) or {}
            identity_transitions = by_identity.get(identity, [])
            identity_first = min(
                (_timestamp(row.get("recorded_at")) for row in identity_transitions),
                default=None,
                key=lambda value: value or now + timedelta(days=1),
            )
            counterfactual = _counterfactual_outcome(
                source_frame,
                detected_at=identity_first,
                entry_price=opportunity.get("entry_price"),
                stop_loss=opportunity.get("stop_loss"),
                take_profit_1=opportunity.get("take_profit_1"),
            )
            paper_execution = paper_outcomes.get(
                identity,
                {
                    "evidence": "NO_CANONICAL_PAPER_FILL",
                    "paper_closed": False,
                    "paper_net_pnl_eur": None,
                },
            )
            reference_entry = _decimal(counterfactual.get("theoretical_entry_price"))
            paper_buy = _decimal(paper_execution.get("paper_average_buy_fill_price"))
            paper_slippage_bps = (
                float((paper_buy / reference_entry - Decimal("1")) * Decimal("10000"))
                if paper_buy is not None and reference_entry is not None and reference_entry > 0
                else None
            )
            opportunity_outcomes.append(
                {
                    "opportunity_id": identity,
                    "playbook_id": opportunity.get("playbook_id"),
                    "playbook_dna": opportunity.get("playbook_dna"),
                    "regime": opportunity.get("macro_regime"),
                    "timeframe": opportunity.get("context_timeframe"),
                    "detected_at": identity_first.isoformat() if identity_first else None,
                    "counterfactual": counterfactual,
                    "paper_execution": paper_execution,
                    "paper_entry_slippage_bps_vs_theoretical": paper_slippage_bps,
                    "live_execution": {
                        "evidence": "NO_CANONICAL_LIVE_FILL",
                        "live_closed": False,
                        "live_net_pnl_eur": None,
                    },
                }
            )
        live_transitions = [
            row
            for row in transitions
            if "PAPER_ONLY" not in set(row.get("reason_codes") or [])
            and not (row.get("details") or {}).get("paper_event")
        ]
        live_furthest = _furthest_progress_state(live_transitions)
        ready_identities = {
            str(row.get("opportunity_id") or "")
            for row in live_transitions
            if row.get("to_state") == "ENTRY_READY"
        }
        ready_transitions = [
            row
            for row in live_transitions
            if str(row.get("opportunity_id") or "") in ready_identities
        ]
        ready_current = next(
            (
                row
                for identity, row in current.items()
                if identity in ready_identities
            ),
            None,
        )
        paper_converted = any(
            "PAPER_ONLY" in set(row.get("reason_codes") or [])
            or bool((row.get("details") or {}).get("paper_event"))
            for row in transitions
        )
        converted = live_furthest in {
            "ORDER_SUBMITTED",
            "PARTIALLY_FILLED",
            "FILLED",
            "MANAGING",
            "EXITING",
            "CLOSED",
        }
        long_entry_opportunity = move["direction"] == "UP"
        missed = bool(long_entry_opportunity and not converted)
        if not long_entry_opportunity:
            miss_reason = "NOT_A_LONG_SPOT_ENTRY_OPPORTUNITY"
        elif not prospectively_observable and not transitions:
            miss_reason = "RUNTIME_NOT_OBSERVING_YET"
        elif converted:
            miss_reason = None
        else:
            miss_reason = _miss_reason(
                ready_transitions or transitions,
                ready_current or related_current,
                furthest_state=live_furthest,
                operational_blockers=operational_blockers,
            )
        significant.append(
            {
                **move,
                "long_entry_opportunity": long_entry_opportunity,
                "prospectively_observable": prospectively_observable,
                "runtime_coverage_started_at": (
                    lifecycle_coverage_started_at.isoformat()
                    if lifecycle_coverage_started_at
                    else None
                ),
                "opportunity_created": bool(transitions),
                "first_detected_at": first.isoformat() if first else None,
                "detection_latency_seconds": (
                    max(0.0, (first - first_decision).total_seconds())
                    if first and first_decision
                    else None
                ),
                "move_threshold_crossed_at": (
                    detectable.isoformat() if detectable else None
                ),
                "move_to_first_opportunity_seconds": (
                    max(0.0, (first - detectable).total_seconds())
                    if first and detectable and prospectively_observable
                    else None
                ),
                "detection_latency_from_runtime_start_seconds": (
                    max(
                        0.0,
                        (first - lifecycle_coverage_started_at).total_seconds(),
                    )
                    if first and lifecycle_coverage_started_at
                    else None
                ),
                "opportunity_count": len(identities),
                "furthest_lifecycle_state": furthest,
                "furthest_live_lifecycle_state": live_furthest,
                "paper_converted": paper_converted,
                "converted_to_order": converted,
                "operational_entry_blockers": (
                    list(dict.fromkeys(operational_blockers))
                    if live_furthest == "ENTRY_READY" and not converted
                    else []
                ),
                "missed": missed,
                "miss_reason": miss_reason,
                "realized_pnl_eur": None,
                "pnl_evidence": "UNAVAILABLE_UNLESS_FILLED_AND_CLOSED",
                "opportunity_outcomes": opportunity_outcomes,
            }
        )
    state_counts = Counter(
        str(row.get("to_state") or "UNKNOWN") for row in lifecycle
    )
    discovered_ids = set(by_identity)
    live_rows_by_identity = {
        identity: [
            row
            for row in rows
            if "PAPER_ONLY" not in set(row.get("reason_codes") or [])
            and not (row.get("details") or {}).get("paper_event")
        ]
        for identity, rows in by_identity.items()
    }
    paper_ids = {
        identity
        for identity, rows in by_identity.items()
        if any(
            "PAPER_ONLY" in set(row.get("reason_codes") or [])
            or bool((row.get("details") or {}).get("paper_event"))
            for row in rows
        )
    }
    order_ids = {
        identity
        for identity, rows in live_rows_by_identity.items()
        if any(row.get("to_state") == "ORDER_SUBMITTED" for row in rows)
    }
    fill_ids = {
        identity
        for identity, rows in live_rows_by_identity.items()
        if any(
            row.get("to_state") in {"PARTIALLY_FILLED", "FILLED", "MANAGING", "CLOSED"}
            for row in rows
        )
    }
    # The exchange execution ledger is the canonical order/fill source.  A
    # recovered fill may legitimately exist without an intermediate lifecycle
    # transition, so reconcile by stable signal identity instead of reporting
    # an impossible zero-order/one-fill state.
    canonical_order_ids = {
        str((row.get("payload") or {}).get("signal_id"))
        for row in canonical_live_events
        if str(row.get("event_type") or "").upper()
        in {"ORDER_INTENT", "ORDER_SUBMITTED", "ORDER_ACKNOWLEDGED"}
        and (row.get("payload") or {}).get("signal_id")
    }
    canonical_fill_ids = {
        str((row.get("payload") or {}).get("signal_id"))
        for row in canonical_live_events
        if row.get("event_type") == "FILL"
        and (row.get("payload") or {}).get("signal_id")
    }
    order_ids.update(canonical_order_ids)
    fill_ids.update(canonical_fill_ids)
    economically_valid_ids = {
        identity
        for identity, rows in live_rows_by_identity.items()
        if any(
            row.get("to_state") == "ENTRY_READY"
            and not row.get("hard_blockers")
            and dict(row.get("execution_economics") or {}).get(
                "positive_after_costs", True
            )
            is not False
            for row in rows
        )
    }
    executed_economically_valid_ids = economically_valid_ids & order_ids
    missed_reasons = Counter(
        str(row["miss_reason"])
        for row in significant
        if row.get("missed") and row.get("miss_reason")
    )
    resolved_counterfactuals = [
        outcome["counterfactual"]
        for move in significant
        for outcome in move.get("opportunity_outcomes") or []
        if outcome.get("counterfactual", {}).get("false_positive") is not None
    ]
    false_positives = sum(
        bool(row.get("false_positive")) for row in resolved_counterfactuals
    )
    latest_lifecycle = {
        identity: rows[-1]
        for identity, rows in by_identity.items()
        if rows
    }
    decision_traces = [
        dict(row.get("decision_trace") or {})
        for row in latest_lifecycle.values()
    ]
    duplicates_removed = sum(
        max(0, int(row.get("cluster_size") or 1) - 1)
        for row in latest_lifecycle.values()
    )
    entry_ready_before_cost_gate = sum(
        trace.get("original_rule_decision") == "ENTRY_READY"
        for trace in decision_traces
    )
    blocked_by_cost_gate = sum(
        trace.get("filter_that_blocked_trade")
        == "NO_POSITIVE_EXIT_PATH_AFTER_ALL_IN_COSTS"
        for trace in decision_traces
    )
    blocked_by_correlation_gate = sum(
        trace.get("filter_that_blocked_trade")
        == "INSUFFICIENT_REALTIME_CONFIRMATIONS"
        and trace.get("original_rule_decision") == "ENTRY_READY"
        for trace in decision_traces
    )
    counterfactual_mfe = [
        float(row["maximum_favorable_move_after_detection_pct"])
        for row in resolved_counterfactuals
        if row.get("maximum_favorable_move_after_detection_pct") is not None
    ]
    counterfactual_mae = [
        float(row["maximum_adverse_move_after_detection_pct"])
        for row in resolved_counterfactuals
        if row.get("maximum_adverse_move_after_detection_pct") is not None
    ]

    def distribution(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "p25": None, "p50": None, "p75": None}
        series = pd.Series(values, dtype=float)
        return {
            "count": len(values),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
        }

    gate_samples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for move in significant:
        for outcome in move.get("opportunity_outcomes") or []:
            identity = str(outcome.get("opportunity_id") or "")
            if not identity or identity in order_ids:
                continue
            transition_rows = by_identity.get(identity, [])
            opportunity = current.get(identity) or {}
            reasons = list(opportunity.get("hard_blockers") or [])
            current_trace = opportunity.get("decision_trace") or {}
            if current_trace.get("filter_that_blocked_trade"):
                reasons.append(current_trace["filter_that_blocked_trade"])
            for transition in transition_rows:
                reasons.extend(transition.get("hard_blockers") or [])
                trace = transition.get("decision_trace") or {}
                if trace.get("filter_that_blocked_trade"):
                    reasons.append(trace["filter_that_blocked_trade"])
            reasons = list(dict.fromkeys(str(value) for value in reasons if value))
            if not reasons:
                continue
            economics = dict(opportunity.get("execution_economics") or {})
            if not economics:
                economics = next(
                    (
                        dict(row.get("execution_economics") or {})
                        for row in reversed(transition_rows)
                        if row.get("execution_economics")
                    ),
                    {},
                )
            counterfactual = dict(outcome.get("counterfactual") or {})
            net_result = _counterfactual_net_result(
                counterfactual,
                economics,
            )
            sample = {
                "opportunity_id": identity,
                "market": move.get("market"),
                "playbook_id": outcome.get("playbook_id"),
                "timeframe": outcome.get("timeframe"),
                "reasons": reasons,
                "boundary_outcome": counterfactual.get(
                    "tp1_or_stop_outcome"
                ),
                "net_result": net_result,
                "mfe_pct": counterfactual.get(
                    "maximum_favorable_move_after_detection_pct"
                ),
                "mae_pct": counterfactual.get(
                    "maximum_adverse_move_after_detection_pct"
                ),
            }
            for category in sorted({_gate_category(reason) for reason in reasons}):
                gate_samples[category].append(sample)

    gate_counterfactual_attribution: dict[str, dict[str, Any]] = {}
    for category, samples in sorted(gate_samples.items()):
        resolved = [sample for sample in samples if sample["net_result"] is not None]
        net_returns = [
            float(sample["net_result"]["net_return_pct"])
            for sample in resolved
        ]
        net_r = [float(sample["net_result"]["net_r"]) for sample in resolved]
        mean_net_r = sum(net_r) / len(net_r) if net_r else None
        if len(resolved) < 5:
            assessment = "INSUFFICIENT_RESOLVED_COUNTERFACTUALS"
        elif mean_net_r is not None and mean_net_r > 0.10:
            assessment = "REVIEW_POSSIBLY_TOO_STRICT"
        elif mean_net_r is not None and mean_net_r < -0.10:
            assessment = "OBSERVED_VALUE_ADD"
        else:
            assessment = "OBSERVED_NEUTRAL_OR_INCONCLUSIVE"
        gate_counterfactual_attribution[category] = {
            "rejected_opportunity_count": len(samples),
            "resolved_boundary_count": len(resolved),
            "tp1_before_stop_count": sum(
                sample["boundary_outcome"] == "TP1_BEFORE_STOP"
                for sample in samples
            ),
            "stop_before_tp1_count": sum(
                sample["boundary_outcome"] == "STOP_BEFORE_TP1"
                for sample in samples
            ),
            "counterfactual_mean_net_return_pct": (
                sum(net_returns) / len(net_returns) if net_returns else None
            ),
            "counterfactual_mean_net_r": mean_net_r,
            "mfe_distribution_pct": distribution(
                [
                    float(sample["mfe_pct"])
                    for sample in samples
                    if sample["mfe_pct"] is not None
                ]
            ),
            "mae_distribution_pct": distribution(
                [
                    float(sample["mae_pct"])
                    for sample in samples
                    if sample["mae_pct"] is not None
                ]
            ),
            "assessment": assessment,
        }
    from core.event_driven_live import is_playbook_opportunity_authorized

    current_economically_valid = [
        row
        for row in current.values()
        if row.get("state") == "ENTRY_READY"
        and not row.get("hard_blockers")
        and dict(row.get("execution_economics") or {}).get(
            "positive_after_costs", True
        )
        is not False
    ]
    current_authority_only_blocked = [
        row
        for row in current_economically_valid
        if not is_playbook_opportunity_authorized(playbook_authority, row)
    ]
    daily_economically_valid: list[dict[str, Any]] = []
    for identity in sorted(economically_valid_ids):
        # Prefer the durable projection because a paper opportunity may have
        # progressed beyond ENTRY_READY to FILLED/CLOSED while still never
        # receiving live authority.  Fall back to the exact causal readiness
        # transition when no projection is available.
        selected = dict(current.get(identity) or {})
        if not selected:
            selected = next(
                (
                    dict(row)
                    for row in reversed(live_rows_by_identity.get(identity, []))
                    if row.get("to_state") == "ENTRY_READY"
                ),
                {},
            )
        if selected:
            selected.setdefault("opportunity_id", identity)
            daily_economically_valid.append(selected)
    daily_authority_only_blocked = [
        row
        for row in daily_economically_valid
        if str(row.get("opportunity_id") or "") not in order_ids
        and not is_playbook_opportunity_authorized(playbook_authority, row)
    ]
    trigger_counterfactual = gate_counterfactual_attribution.get(
        "TRIGGER", {}
    )
    paper_slippages = [
        float(outcome["paper_entry_slippage_bps_vs_theoretical"])
        for move in significant
        for outcome in move.get("opportunity_outcomes") or []
        if outcome.get("paper_entry_slippage_bps_vs_theoretical") is not None
    ]
    pnl_by_playbook: dict[str, float] = defaultdict(float)
    pnl_by_regime: dict[str, float] = defaultdict(float)
    pnl_by_timeframe: dict[str, float] = defaultdict(float)
    verified_closed_paper = 0
    for move in significant:
        for outcome in move.get("opportunity_outcomes") or []:
            execution = outcome.get("paper_execution") or {}
            pnl = execution.get("paper_net_pnl_eur")
            if execution.get("paper_closed") is not True or pnl is None:
                continue
            verified_closed_paper += 1
            pnl_by_playbook[str(outcome.get("playbook_id") or "UNKNOWN")] += float(pnl)
            pnl_by_regime[str(outcome.get("regime") or "UNKNOWN")] += float(pnl)
            pnl_by_timeframe[str(outcome.get("timeframe") or "UNKNOWN")] += float(pnl)
    cadence_windows: dict[str, Any] = {}
    for label, duration in (
        ("1h", timedelta(hours=1)),
        ("6h", timedelta(hours=6)),
        ("24h", timedelta(hours=24)),
        ("72h", timedelta(hours=72)),
        ("7d", timedelta(days=7)),
    ):
        cutoff = now - duration
        window_lifecycle = [
            row
            for row in all_lifecycle
            if (_timestamp(row.get("recorded_at")) or cutoff - timedelta(1))
            >= cutoff
            and (_timestamp(row.get("recorded_at")) or now + timedelta(1))
            <= now
        ]
        window_live_events = [
            row
            for row in all_canonical_live_events
            if (_timestamp(row.get("recorded_at")) or cutoff - timedelta(1))
            >= cutoff
            and (_timestamp(row.get("recorded_at")) or now + timedelta(1))
            <= now
        ]
        raw_ids = {
            str(row.get("opportunity_id"))
            for row in window_lifecycle
            if row.get("opportunity_id")
        }
        ready_ids = {
            str(row.get("opportunity_id"))
            for row in window_lifecycle
            if row.get("opportunity_id")
            and str(row.get("to_state")) == "ENTRY_READY"
        }
        near_entry_ids = {
            str(row.get("opportunity_id"))
            for row in window_lifecycle
            if row.get("opportunity_id")
            and str(row.get("to_state")) in {"ARMED", "ENTRY_READY"}
            and float(row.get("score") or 0.0) >= 65.0
        }
        blockers = Counter(
            str(reason)
            for row in window_lifecycle
            for reason in (row.get("hard_blockers") or [])
        )
        strategy_entry_events = [
            row for row in window_live_events if _is_live_strategy_entry_event(row)
        ]
        submitted_ids = {
            str(
                (row.get("payload") or {}).get("intent_id")
                or (row.get("payload") or {}).get("signal_id")
            )
            for row in strategy_entry_events
            if str(row.get("event_type") or "").upper()
            in {"ORDER_SUBMITTED", "ORDER_ACKNOWLEDGED"}
        }
        fill_ids_in_window = {
            str(
                (row.get("payload") or {}).get("intent_id")
                or (row.get("payload") or {}).get("fill_id")
                or (row.get("payload") or {}).get("signal_id")
            )
            for row in strategy_entry_events
            if str(row.get("event_type") or "").upper() == "FILL"
        }
        submitted = len(submitted_ids)
        fills = len(fill_ids_in_window)
        if not raw_ids:
            diagnosis = "INSUFFICIENT_OPPORTUNITY_SURFACE"
        elif not ready_ids:
            dominant = blockers.most_common(1)
            diagnosis = (
                f"ENTRY_GATE_BOTTLENECK:{dominant[0][0]}"
                if dominant
                else "SETUP_QUALITY_BOTTLENECK"
            )
        elif not submitted:
            diagnosis = "AUTHORITY_OR_EXECUTION_CONVERSION_BOTTLENECK"
        elif not fills:
            diagnosis = "VENUE_FILL_BOTTLENECK"
        else:
            diagnosis = "FUNNEL_CONVERTING"
        cadence_windows[label] = {
            "raw_opportunities": len(raw_ids),
            "strategy_triggers": len(raw_ids),
            "near_entry": len(near_entry_ids),
            "entry_ready": len(ready_ids),
            "orders_submitted": submitted,
            "fills": fills,
            "setup_to_entry_ready_rate": (
                len(ready_ids) / len(raw_ids) if raw_ids else 0.0
            ),
            "setup_to_near_entry_rate": (
                len(near_entry_ids) / len(raw_ids) if raw_ids else 0.0
            ),
            "entry_ready_per_near_entry_observation_ratio": (
                len(ready_ids) / len(near_entry_ids)
                if near_entry_ids
                else 0.0
            ),
            "conversion_metric_semantics": {
                "populations_nested": False,
                "operational_decision_eligible": False,
                "reason": (
                    "entry_ready and near_entry are independent lifecycle "
                    "observations inside the cadence window; this ratio may "
                    "exceed 1 and is not a conversion probability"
                ),
            },
            "top_hard_blockers": dict(blockers.most_common(10)),
            "diagnosis": diagnosis,
            "trade_quota_enforced": False,
            "execution_count_scope": "LIVE_STRATEGY_BUY_ENTRIES_ONLY",
        }

    payload: dict[str, Any] = {
        "schema_version": "daily_opportunity_audit_v1",
        "date_utc": start.date().isoformat(),
        "generated_at": now.isoformat(),
        "runtime_coverage_started_at": (
            lifecycle_coverage_started_at.isoformat()
            if lifecycle_coverage_started_at
            else None
        ),
        "significant_move_threshold_pct": SIGNIFICANT_MOVE_PCT,
        "market_count": len(moves),
        "significant_move_count": len(significant),
        "opportunities_detected": len(discovered_ids),
        "unique_opportunity_clusters": len(
            {
                str(row.get("cluster_id") or row.get("opportunity_id"))
                for row in lifecycle
                if row.get("opportunity_id")
            }
        ),
        "duplicates_removed": duplicates_removed,
        "entry_ready_before_cost_gate": entry_ready_before_cost_gate,
        "blocked_by_cost_gate": blocked_by_cost_gate,
        "blocked_by_correlation_gate": blocked_by_correlation_gate,
        "blocked_by_portfolio_heat": int(
            live_state.get("reason_code")
            in {
                "TOTAL_WALLET_PORTFOLIO_HEAT_LIMIT",
                "TOTAL_WALLET_EXPOSURE_VALUATION_INCOMPLETE",
            }
            or any("TOTAL_WALLET" in reason for reason in operational_blockers)
        ),
        "blocked_by_data_quality": sum(
            any(
                reason
                in {
                    "STALE_REALTIME_DATA",
                    "ORDERBOOK_SEQUENCE_INVALID",
                }
                for reason in (row.get("hard_blockers") or [])
            )
            for row in latest_lifecycle.values()
        ),
        "entry_ready": state_counts.get("ENTRY_READY", 0),
        "orders_submitted": len(order_ids),
        "opportunities_filled": len(fill_ids),
        "paper_opportunities_filled": len(paper_ids),
        "opportunity_conversion_rate": (
            len(order_ids) / len(discovered_ids) if discovered_ids else 0.0
        ),
        "fill_rate": len(fill_ids) / len(order_ids) if order_ids else 0.0,
        "opportunity_utilization": {
            "scope": "DAILY_CAUSAL_ENTRY_READY_AFTER_COSTS",
            "economically_valid_opportunities": len(
                economically_valid_ids
            ),
            "executed_valid_opportunities": len(
                executed_economically_valid_ids
            ),
            "ratio": (
                len(executed_economically_valid_ids)
                / len(economically_valid_ids)
                if economically_valid_ids
                else None
            ),
            "trade_quota_enforced": False,
        },
        "authority_leakage": {
            "scope": "DAILY_CAUSAL_ENTRY_READY_WITH_CURRENT_AUTHORITY_PROJECTION",
            "economically_valid_candidates": len(daily_economically_valid),
            "blocked_only_by_missing_authority": len(
                daily_authority_only_blocked
            ),
            "ratio": (
                len(daily_authority_only_blocked)
                / len(daily_economically_valid)
                if daily_economically_valid
                else None
            ),
            "candidates": [
                {
                    "opportunity_id": str(row.get("opportunity_id") or ""),
                    "market": str(row.get("market") or ""),
                    "playbook_id": str(row.get("playbook_id") or ""),
                    "family": str(row.get("family") or ""),
                    "projection_state": str(row.get("state") or ""),
                    "paper_lifecycle_is_not_live_execution": True,
                }
                for row in daily_authority_only_blocked[:50]
            ],
            "current_projection": {
                "scope": "CURRENT_ENTRY_READY_PROJECTION_ONLY",
                "economically_valid_candidates": len(
                    current_economically_valid
                ),
                "blocked_only_by_missing_authority": len(
                    current_authority_only_blocked
                ),
                "ratio": (
                    len(current_authority_only_blocked)
                    / len(current_economically_valid)
                    if current_economically_valid
                    else None
                ),
                "opportunity_ids": [
                    str(row.get("opportunity_id") or "")
                    for row in current_authority_only_blocked[:50]
                ],
            },
            # Retain the old field for report consumers while making its
            # narrower scope explicit above.
            "economically_valid_current_candidates": len(
                current_economically_valid
            ),
            "opportunity_ids": [
                str(row.get("opportunity_id") or "")
                for row in daily_authority_only_blocked[:50]
            ],
            "automatic_authority_changes": False,
        },
        "trigger_leakage": {
            "scope": "DAILY_REJECTED_TRIGGER_COUNTERFACTUALS",
            "rejected_opportunity_count": int(
                trigger_counterfactual.get("rejected_opportunity_count") or 0
            ),
            "resolved_boundary_count": int(
                trigger_counterfactual.get("resolved_boundary_count") or 0
            ),
            "counterfactual_mean_net_r": trigger_counterfactual.get(
                "counterfactual_mean_net_r"
            ),
            "assessment": trigger_counterfactual.get(
                "assessment", "NO_RESOLVED_TRIGGER_COUNTERFACTUALS"
            ),
            "automatic_threshold_changes": False,
        },
        "missed_move_rate": (
            sum(bool(row["missed"]) for row in significant) / len(significant)
            if significant
            else 0.0
        ),
        "average_detection_latency_seconds": (
            sum(
                float(row["detection_latency_seconds"])
                for row in significant
                if row.get("detection_latency_seconds") is not None
            )
            / sum(
                row.get("detection_latency_seconds") is not None
                for row in significant
            )
            if any(row.get("detection_latency_seconds") is not None for row in significant)
            else None
        ),
        "average_move_to_first_opportunity_seconds": (
            sum(
                float(row["move_to_first_opportunity_seconds"])
                for row in significant
                if row.get("move_to_first_opportunity_seconds") is not None
            )
            / sum(
                row.get("move_to_first_opportunity_seconds") is not None
                for row in significant
            )
            if any(
                row.get("move_to_first_opportunity_seconds") is not None
                for row in significant
            )
            else None
        ),
        "average_slippage_bps": (
            sum(paper_slippages) / len(paper_slippages)
            if paper_slippages
            else None
        ),
        "paper_gross_expectancy_eur": paper_summary["paper_gross_expectancy_eur"],
        "paper_net_expectancy_eur": paper_summary["paper_net_expectancy_eur"],
        "false_breakout_rate": (
            false_positives / len(resolved_counterfactuals)
            if resolved_counterfactuals
            else None
        ),
        "mfe_distribution_pct": distribution(counterfactual_mfe),
        "mae_distribution_pct": distribution(counterfactual_mae),
        "soft_exit_count": sum(
            str((row.get("payload") or {}).get("reason") or "").upper()
            in {
                "ORDERFLOW_EXHAUSTION",
                "NEGATIVE_CVD_REVERSAL",
                "BID_SUPPORT_WITHDRAWAL",
                "SPREAD_EXPANSION",
                "MOMENTUM_DECAY",
                "RELATIVE_STRENGTH_DETERIORATION",
            }
            for row in canonical_live_events
            if row.get("event_type") == "ORDER_INTENT"
        ),
        "soft_exit_saved_loss_eur": None,
        "soft_exit_missed_profit_eur": None,
        "soft_exit_counterfactual_status": "PENDING_CLEAN_POST_EXIT_LABELS",
        "resolved_counterfactual_count": len(resolved_counterfactuals),
        "gate_counterfactual_attribution": gate_counterfactual_attribution,
        "gate_counterfactual_method": {
            "scope": "REJECTED_OPPORTUNITIES_MULTI_LABEL",
            "result": "FIRST_UNAMBIGUOUS_TP1_OR_STOP_BOUNDARY",
            "costs": "RECORDED_ESTIMATED_ROUNDTRIP_COST_BPS",
            "execution_claim": False,
            "automatic_gate_changes": False,
        },
        "trade_cadence": {
            "windows": cadence_windows,
            "rolling_7d_design_targets": {
                "raw_strategy_triggers": [70, 200],
                "qualified_candidates": [20, 60],
                "entry_ready": [5, 15],
                "expected_fills": [3, 10],
            },
            "minimum_daily_trade_quota": None,
            "frequency_is_not_an_execution_override": True,
        },
        "state_transition_counts": dict(sorted(state_counts.items())),
        "missed_reason_counts": dict(sorted(missed_reasons.items())),
        "significant_moves": significant,
        "pnl_by_playbook": dict(sorted(pnl_by_playbook.items())),
        "pnl_by_regime": dict(sorted(pnl_by_regime.items())),
        "pnl_by_timeframe": dict(sorted(pnl_by_timeframe.items())),
        "pnl_status": (
            "VERIFIED_PAPER_CLOSED_FILLS_LIVE_UNAVAILABLE"
            if verified_closed_paper
            else "PENDING_VERIFIED_CLOSED_FILLS"
        ),
        "paper_execution_evidence": paper_summary,
        "canonical_live_execution": {
            "source": str(live_execution_path),
            "order_signal_count": len(canonical_order_ids),
            "fill_signal_count": len(canonical_fill_ids),
            "reconciled_with_lifecycle": True,
        },
        "source_files": [
            str(lifecycle_path),
            str(projection_path),
            str(paper_summary["source"]),
            str(live_execution_path),
            str(live_state_path),
        ],
    }
    payload["audit_hash"] = stable_hash(payload, length=64)
    operations = settings.paths.output_dir / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    current_path = operations / "daily_opportunity_audit.json"
    history_path = operations / "opportunity_audit_history.jsonl"
    prior_hash = None
    if current_path.is_file():
        try:
            prior_hash = json.loads(current_path.read_text(encoding="utf-8")).get(
                "audit_hash"
            )
        except (OSError, TypeError, ValueError):
            prior_hash = None
    atomic_write_json(current_path, payload)
    if payload["audit_hash"] != prior_hash:
        append_jsonl(history_path, payload)
    return payload


__all__ = ["build_daily_opportunity_audit"]
