"""Canonical strategy-family economics from immutable execution evidence.

This module is deliberately orderless.  It reads paper/live ledgers, reduces
them through the canonical execution aggregate, and writes immutable economic
evidence.  It never calls a venue or changes strategy authority.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config.settings import Settings
from execution.canonical_state import (
    CanonicalExecutionState,
    CanonicalFill,
    assert_replay_deterministic,
    replay_execution_events,
)
from utils.common import atomic_write_json, stable_hash, utc_iso

ECONOMIC_SCHEMA_VERSION = "canonical_strategy_family_economics_v5"
ZERO = Decimal("0")
MINIMUM_FAMILY_SAMPLE = 30
MINIMUM_RATIO_SAMPLE = 10

STANDARD_FAMILIES = {
    "TREND",
    "MOMENTUM",
    "BREAKOUT",
    "MEAN_REVERSION",
    "VOLATILITY",
    "VOLUME",
    "ORDERFLOW",
    "LIQUIDITY_SWEEP",
    "FAILED_BREAKDOWN_REVERSAL",
    "FAILED_BREAKOUT_REVERSAL",
    "RELATIVE_STRENGTH",
    "CROSS_SECTIONAL_MOMENTUM",
    "REGIME_CONDITIONED",
    "MULTI_TIMEFRAME",
    "MARKET_BREADTH",
    "DERIVATIVES_CONTEXT",
    "NEWS_SENTIMENT",
    "HYBRID",
    "UNKNOWN",
}

PLAYBOOK_FAMILY_MAP = {
    "MOMENTUM_BREAKOUT_V1": ("MOMENTUM", ("BREAKOUT",)),
    "VOLATILITY_EXPANSION_V1": ("VOLATILITY", ("VOLUME",)),
    "LIQUIDITY_SWEEP_RECLAIM_V1": ("LIQUIDITY_SWEEP", ("MEAN_REVERSION",)),
    "FAILED_BREAKDOWN_REVERSAL_V1": (
        "FAILED_BREAKDOWN_REVERSAL",
        ("MEAN_REVERSION",),
    ),
    "FAILED_BREAKOUT_REVERSAL_V1": (
        "FAILED_BREAKOUT_REVERSAL",
        ("MEAN_REVERSION",),
    ),
    "RELATIVE_STRENGTH_ROTATION_V1": (
        "RELATIVE_STRENGTH",
        ("CROSS_SECTIONAL_MOMENTUM",),
    ),
    "ORDERFLOW_CONTINUATION_V1": ("ORDERFLOW", ("TREND",)),
    "RANGE_EXPANSION_VOLUME_V1": ("VOLUME", ("VOLATILITY", "BREAKOUT")),
    "VWAP_RECLAIM_V1": ("MEAN_REVERSION", ("ORDERFLOW",)),
}


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return selected if selected.is_finite() else None


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json_bytes(path.read_bytes())


def _read_json_bytes(content: bytes) -> dict[str, Any]:
    if not content:
        return {}
    try:
        value = json.loads(content)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _read_jsonl_bytes(path.read_bytes())


def _read_jsonl_bytes(content: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def canonical_family(
    strategy_id: str | None,
    declared_family: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Return a conservative primary family and optional components."""

    strategy = str(strategy_id or "").upper()
    if strategy in PLAYBOOK_FAMILY_MAP:
        return PLAYBOOK_FAMILY_MAP[strategy]
    value = str(declared_family or strategy_id or "").upper()
    rules = (
        ("FAILED_BREAKDOWN", "FAILED_BREAKDOWN_REVERSAL"),
        ("FAILED_BREAKOUT", "FAILED_BREAKOUT_REVERSAL"),
        ("LIQUIDITY_SWEEP", "LIQUIDITY_SWEEP"),
        ("CROSS_SECTION", "CROSS_SECTIONAL_MOMENTUM"),
        ("RELATIVE_STRENGTH", "RELATIVE_STRENGTH"),
        ("ORDERFLOW", "ORDERFLOW"),
        ("MARKET_BREADTH", "MARKET_BREADTH"),
        ("DERIVATIVE", "DERIVATIVES_CONTEXT"),
        ("SENTIMENT", "NEWS_SENTIMENT"),
        ("REGIME", "REGIME_CONDITIONED"),
        ("MULTI_TIMEFRAME", "MULTI_TIMEFRAME"),
        ("MEAN_REVERSION", "MEAN_REVERSION"),
        ("RECLAIM", "MEAN_REVERSION"),
        ("VOLATILITY", "VOLATILITY"),
        ("VOLUME", "VOLUME"),
        ("BREAKOUT", "BREAKOUT"),
        ("MOMENTUM", "MOMENTUM"),
        ("TREND", "TREND"),
        ("HYBRID", "HYBRID"),
    )
    for token, family in rules:
        if token in value:
            return family, ()
    return "UNKNOWN", ()


def classify_entry_type(strategy_id: str | None) -> str:
    value = str(strategy_id or "").upper()
    if "LIQUIDITY_SWEEP" in value:
        return "LIQUIDITY_SWEEP"
    if "FAILED_BREAKDOWN" in value:
        return "FAILED_BREAKDOWN_REVERSAL"
    if "FAILED_BREAKOUT" in value:
        return "FAILED_BREAKOUT_REVERSAL"
    if "VWAP_RECLAIM" in value:
        return "VWAP_RECLAIM"
    if "RANGE_EXPANSION" in value or "VOLATILITY_EXPANSION" in value:
        return "VOLUME_EXPANSION"
    if "PULLBACK" in value:
        return "PULLBACK"
    if "RETEST" in value:
        return "RETEST"
    if "BREAKOUT" in value:
        return "BREAKOUT"
    if "MEAN_REVERSION" in value:
        return "MEAN_REVERSION"
    if "CONTINUATION" in value or "RELATIVE_STRENGTH" in value:
        return "TREND_CONTINUATION"
    return "OTHER"


def classify_exit_type(reason_codes: Sequence[str]) -> str:
    reasons = {str(value).upper() for value in reason_codes}
    mapping = (
        ("TAKE_PROFIT_2", "TP2"),
        ("TAKE_PROFIT_1", "TP1"),
        ("PROTECTIVE_STOP", "PROTECTIVE_STOP"),
        ("HARD_STOP", "HARD_STOP"),
        ("TRAILING_STOP", "TRAILING_STOP"),
        ("TIME_STOP", "TIME_EXIT"),
        ("ORDERFLOW_EXHAUSTION", "SIGNAL_DETERIORATION"),
        ("REGIME_EXIT", "REGIME_EXIT"),
        ("PORTFOLIO_EXIT", "PORTFOLIO_EXIT"),
        ("MANUAL", "MANUAL"),
        ("RECONCILIATION", "RECONCILIATION"),
    )
    for raw, classified in mapping:
        if raw in reasons:
            return classified
    return "OTHER"


def _intent_metadata(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = dict(event.get("payload") or {})
        candidates: list[Mapping[str, Any]] = [payload]
        record = payload.get("record")
        if isinstance(record, Mapping):
            intent = record.get("intent")
            if isinstance(intent, Mapping):
                candidates.append(intent)
        embedded = payload.get("intent")
        if isinstance(embedded, Mapping):
            candidates.append(embedded)
        for candidate in candidates:
            intent_id = str(candidate.get("intent_id") or "")
            if not intent_id:
                continue
            selected = metadata.setdefault(intent_id, {})
            for key, value in candidate.items():
                if value not in (None, "", [], {}) or key not in selected:
                    selected[key] = value
            if payload.get("order_id"):
                selected["order_id"] = payload["order_id"]
            if isinstance(record, Mapping) and record.get("order_id"):
                selected["order_id"] = record["order_id"]
    return metadata


def load_causal_snapshots(path: Path) -> dict[str, dict[str, Any]]:
    """Load the first immutable decision-time feature snapshot per signal."""

    if not path.is_file():
        return {}
    return _load_causal_snapshots_bytes(path.read_bytes())


def _load_causal_snapshots_bytes(content: bytes) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for line in content.splitlines():
        if b"feature_snapshot" not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        identity = str(event.get("opportunity_id") or "")
        snapshot = event.get("feature_snapshot")
        if identity and isinstance(snapshot, Mapping):
            snapshots.setdefault(identity, dict(snapshot))
    return snapshots


@dataclass
class CanonicalEconomicEpisode:
    episode_id: str
    evidence_layer: str
    reconstruction_status: str
    market: str
    strategy_family: str
    secondary_components: tuple[str, ...]
    strategy_id: str
    strategy_version: str | None
    strategy_dna_hash: str | None
    setup_id: str | None
    signal_id: str | None
    ownership_state: str
    entry_type: str
    exit_type: str
    exit_reason_codes: tuple[str, ...]
    timeframe: str
    regime: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    holding_seconds: float
    entry_quantity: Decimal
    exit_quantity: Decimal
    average_entry_price: Decimal
    average_exit_price: Decimal
    entry_notional_eur: Decimal
    exit_notional_eur: Decimal
    gross_pnl_before_costs_eur: Decimal
    fees_eur: Decimal
    observed_slippage_eur: Decimal | None
    observed_slippage_bps: Decimal | None
    net_pnl_eur: Decimal | None
    canonical_realized_pnl_eur: Decimal | None
    canonical_pnl_complete: bool
    entry_fill_ids: tuple[str, ...]
    exit_fill_ids: tuple[str, ...]
    all_fill_ids: tuple[str, ...]
    mfe_pct: Decimal | None = None
    mae_pct: Decimal | None = None
    market_path_evidence: str = "NOT_EVALUABLE"

    def to_dict(self) -> dict[str, Any]:
        def ready(value: Any) -> Any:
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, datetime):
                return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if isinstance(value, tuple):
                return [ready(item) for item in value]
            return value

        return {key: ready(value) for key, value in asdict(self).items()}


@dataclass
class _OpenEpisode:
    market: str
    quantity: Decimal = ZERO
    entries: list[CanonicalFill] = field(default_factory=list)
    exits: list[CanonicalFill] = field(default_factory=list)


def reconstruct_canonical_episodes(
    events: Sequence[Mapping[str, Any]],
    *,
    evidence_layer: str = "PAPER",
    causal_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    CanonicalExecutionState,
    list[CanonicalEconomicEpisode],
    list[dict[str, Any]],
]:
    """Build position-flat-to-position-flat episodes from canonical fills."""

    snapshots = dict(causal_snapshots or {})
    state = replay_execution_events(events)
    intents = _intent_metadata(events)
    fills = sorted(
        state.fills.values(),
        key=lambda row: (row.filled_at, row.fill_id),
    )
    open_by_market: dict[str, _OpenEpisode] = {}
    episodes: list[CanonicalEconomicEpisode] = []
    unknowns: list[dict[str, Any]] = []

    for fill in fills:
        current = open_by_market.get(fill.market)
        if fill.side == "BUY":
            if current is None or current.quantity <= ZERO:
                current = _OpenEpisode(market=fill.market)
                open_by_market[fill.market] = current
            current.entries.append(fill)
            current.quantity += fill.quantity
            continue
        if current is None or current.quantity <= ZERO:
            unknowns.append(
                {
                    "classification": "NOT_RECONSTRUCTABLE",
                    "code": "SELL_WITHOUT_OPEN_CANONICAL_EPISODE",
                    "fill_id": fill.fill_id,
                    "market": fill.market,
                    "quantity": str(fill.quantity),
                }
            )
            continue
        current.exits.append(fill)
        matched_quantity = min(current.quantity, fill.quantity)
        current.quantity -= matched_quantity
        if fill.quantity > matched_quantity:
            unknowns.append(
                {
                    "classification": "PARTIALLY_RECONSTRUCTABLE",
                    "code": "EXIT_EXCEEDS_OPEN_EPISODE_QUANTITY",
                    "fill_id": fill.fill_id,
                    "market": fill.market,
                    "unmatched_quantity": str(fill.quantity - matched_quantity),
                }
            )
        if current.quantity > ZERO:
            continue
        episodes.append(
            _finalize_episode(
                current,
                state=state,
                intents=intents,
                snapshots=snapshots,
                evidence_layer=evidence_layer,
            )
        )
        del open_by_market[fill.market]

    for current in open_by_market.values():
        unknowns.append(
            {
                "classification": "PARTIALLY_RECONSTRUCTABLE",
                "code": "OPEN_EPISODE_AT_DATA_CUTOFF",
                "market": current.market,
                "quantity": str(current.quantity),
                "entry_fill_ids": [fill.fill_id for fill in current.entries],
            }
        )
    return state, episodes, unknowns


def _finalize_episode(
    current: _OpenEpisode,
    *,
    state: CanonicalExecutionState,
    intents: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    evidence_layer: str,
) -> CanonicalEconomicEpisode:
    entries = current.entries
    exits = current.exits
    entry_metadata = [dict(intents.get(fill.intent_id) or {}) for fill in entries]
    exit_metadata = [dict(intents.get(fill.intent_id) or {}) for fill in exits]
    owners = {
        str(fill.strategy_id or meta.get("strategy_id") or "UNKNOWN_OWNER")
        for fill, meta in zip(entries, entry_metadata, strict=True)
    }
    ownership_state = (
        "UNKNOWN"
        if "UNKNOWN_OWNER" in owners
        else "KNOWN"
        if len(owners) == 1
        else "MIXED"
    )
    strategy_id = next(iter(owners)) if len(owners) == 1 else "MIXED_OWNER"
    if ownership_state == "UNKNOWN":
        strategy_id = "UNKNOWN_OWNER"
    family, secondary = canonical_family(strategy_id)
    dna_values = {
        str(fill.strategy_dna_hash or meta.get("strategy_dna_hash") or "")
        for fill, meta in zip(entries, entry_metadata, strict=True)
        if fill.strategy_dna_hash or meta.get("strategy_dna_hash")
    }
    signal_values = {
        str(fill.signal_id or meta.get("signal_id") or "")
        for fill, meta in zip(entries, entry_metadata, strict=True)
        if fill.signal_id or meta.get("signal_id")
    }
    setup_values = {
        str(fill.setup_id or meta.get("setup_id") or "")
        for fill, meta in zip(entries, entry_metadata, strict=True)
        if fill.setup_id or meta.get("setup_id")
    }
    signal_id = next(iter(signal_values)) if len(signal_values) == 1 else None
    snapshot = dict(snapshots.get(signal_id or "") or {})
    features = dict(snapshot.get("values") or {})
    timeframe = str(
        features.get("entry_timeframe")
        or features.get("timeframe")
        or features.get("context_timeframe")
        or "UNKNOWN_TIMEFRAME"
    )
    regime = str(
        features.get("macro_regime")
        or features.get("regime")
        or "UNKNOWN_REGIME"
    )
    entry_quantity = sum((fill.quantity for fill in entries), ZERO)
    exit_quantity = sum((fill.quantity for fill in exits), ZERO)
    entry_notional = sum((fill.quantity * fill.price for fill in entries), ZERO)
    exit_notional = sum((fill.quantity * fill.price for fill in exits), ZERO)
    fees = sum((fill.fee_eur for fill in (*entries, *exits)), ZERO)
    gross = exit_notional - entry_notional
    net = gross - fees
    realized_rows = [state.realized_pnl_events.get(fill.fill_id) for fill in exits]
    realized_complete = all(row is not None and row.complete for row in realized_rows)
    canonical_realized = (
        sum(
            (
                row.realized_pnl_eur
                for row in realized_rows
                if row is not None and row.realized_pnl_eur is not None
            ),
            ZERO,
        )
        if realized_complete
        else None
    )
    if canonical_realized is not None and abs(canonical_realized - net) > Decimal(
        "0.00000001"
    ):
        realized_complete = False
        canonical_realized = None
    reason_codes = tuple(
        sorted(
            {
                str(reason).upper()
                for meta in exit_metadata
                for reason in meta.get("reason_codes") or []
                if str(reason).upper() != "PAPER_ONLY"
            }
        )
    )
    exit_types = {
        classify_exit_type(
            [
                str(reason)
                for reason in meta.get("reason_codes") or []
                if str(reason).upper() != "PAPER_ONLY"
            ]
        )
        for meta in exit_metadata
    }
    exit_type = next(iter(exit_types)) if len(exit_types) == 1 else "MIXED"
    entry_at = min(fill.filled_at for fill in entries)
    exit_at = max(fill.filled_at for fill in exits)
    expected_entry = _decimal(features.get("entry_price"))
    average_entry = entry_notional / entry_quantity if entry_quantity > ZERO else ZERO
    average_exit = exit_notional / exit_quantity if exit_quantity > ZERO else ZERO
    slippage_bps = (
        (average_entry - expected_entry) / expected_entry * Decimal("10000")
        if expected_entry is not None and expected_entry > ZERO
        else None
    )
    slippage_eur = (
        (average_entry - expected_entry) * entry_quantity
        if expected_entry is not None
        else None
    )
    reconstruction = (
        "CANONICALLY_RECONSTRUCTABLE"
        if realized_complete and ownership_state == "KNOWN"
        else "PARTIALLY_RECONSTRUCTABLE"
    )
    return CanonicalEconomicEpisode(
        episode_id=stable_hash(
            [
                ECONOMIC_SCHEMA_VERSION,
                [fill.fill_id for fill in entries],
                [fill.fill_id for fill in exits],
            ],
            length=48,
        ),
        evidence_layer=evidence_layer,
        reconstruction_status=reconstruction,
        market=current.market,
        strategy_family=family,
        secondary_components=secondary,
        strategy_id=strategy_id,
        strategy_version=strategy_id if strategy_id != "UNKNOWN_OWNER" else None,
        strategy_dna_hash=next(iter(dna_values)) if len(dna_values) == 1 else None,
        setup_id=next(iter(setup_values)) if len(setup_values) == 1 else None,
        signal_id=signal_id,
        ownership_state=ownership_state,
        entry_type=classify_entry_type(strategy_id),
        exit_type=exit_type,
        exit_reason_codes=reason_codes,
        timeframe=timeframe,
        regime=regime,
        entry_timestamp=entry_at,
        exit_timestamp=exit_at,
        holding_seconds=(exit_at - entry_at).total_seconds(),
        entry_quantity=entry_quantity,
        exit_quantity=exit_quantity,
        average_entry_price=average_entry,
        average_exit_price=average_exit,
        entry_notional_eur=entry_notional,
        exit_notional_eur=exit_notional,
        gross_pnl_before_costs_eur=gross,
        fees_eur=fees,
        observed_slippage_eur=slippage_eur,
        observed_slippage_bps=slippage_bps,
        net_pnl_eur=canonical_realized,
        canonical_realized_pnl_eur=canonical_realized,
        canonical_pnl_complete=realized_complete,
        entry_fill_ids=tuple(fill.fill_id for fill in entries),
        exit_fill_ids=tuple(fill.fill_id for fill in exits),
        all_fill_ids=tuple(fill.fill_id for fill in (*entries, *exits)),
    )


def economic_metrics(
    episodes: Sequence[CanonicalEconomicEpisode],
) -> dict[str, Any]:
    complete = [row for row in episodes if row.net_pnl_eur is not None]
    pnl = [row.net_pnl_eur for row in complete if row.net_pnl_eur is not None]
    gross_profit = sum((value for value in pnl if value > ZERO), ZERO)
    gross_loss = abs(sum((value for value in pnl if value < ZERO), ZERO))
    wins = [value for value in pnl if value > ZERO]
    losses = [value for value in pnl if value < ZERO]
    fees = sum((row.fees_eur for row in complete), ZERO)
    pre_cost = sum((row.gross_pnl_before_costs_eur for row in complete), ZERO)
    net = sum(pnl, ZERO)
    cumulative = ZERO
    peak = ZERO
    maximum_drawdown = ZERO
    maximum_drawdown_duration = 0
    underwater_since: int | None = None
    for index, value in enumerate(pnl):
        cumulative += value
        if cumulative >= peak:
            peak = cumulative
            underwater_since = None
        else:
            if underwater_since is None:
                underwater_since = index
            maximum_drawdown = max(maximum_drawdown, peak - cumulative)
            maximum_drawdown_duration = max(
                maximum_drawdown_duration,
                index - underwater_since + 1,
            )
    float_pnl = [float(value) for value in pnl]
    mean = statistics.fmean(float_pnl) if float_pnl else None
    standard_deviation = (
        statistics.stdev(float_pnl) if len(float_pnl) >= 2 else None
    )
    downside = [min(0.0, value) for value in float_pnl]
    downside_deviation = (
        math.sqrt(statistics.fmean([value * value for value in downside]))
        if downside and any(value < 0 for value in downside)
        else None
    )
    entry_capital = sum((row.entry_notional_eur for row in complete), ZERO)
    turnover = sum(
        (row.entry_notional_eur + row.exit_notional_eur for row in complete),
        ZERO,
    )
    median = statistics.median(pnl) if pnl else None
    observed_slippage = [
        row.observed_slippage_eur
        for row in complete
        if row.observed_slippage_eur is not None
    ]
    metrics = {
        "closed_episode_count": len(complete),
        "gross_profit_eur": str(gross_profit),
        "gross_loss_eur": str(gross_loss),
        "gross_pnl_before_costs_eur": str(pre_cost),
        "fees_eur": str(fees),
        "observed_slippage_eur": (
            str(sum(observed_slippage, ZERO)) if observed_slippage else None
        ),
        "slippage_observation_count": len(observed_slippage),
        "net_pnl_eur": str(net),
        "net_expectancy_eur": str(net / len(complete)) if complete else None,
        "profit_factor": (
            float(gross_profit / gross_loss)
            if gross_loss > ZERO
            else None
        ),
        "win_rate": len(wins) / len(complete) if complete else None,
        "average_win_eur": str(sum(wins, ZERO) / len(wins)) if wins else None,
        "average_loss_eur": (
            str(sum(losses, ZERO) / len(losses)) if losses else None
        ),
        "payoff_ratio": (
            float(
                (sum(wins, ZERO) / len(wins))
                / abs(sum(losses, ZERO) / len(losses))
            )
            if wins and losses
            else None
        ),
        "median_trade_eur": str(median) if median is not None else None,
        "trade_sharpe": (
            mean / standard_deviation
            if len(float_pnl) >= MINIMUM_FAMILY_SAMPLE
            and standard_deviation not in (None, 0.0)
            else None
        ),
        "trade_sortino": (
            mean / downside_deviation
            if len(float_pnl) >= MINIMUM_FAMILY_SAMPLE
            and downside_deviation not in (None, 0.0)
            else None
        ),
        "maximum_drawdown_eur": str(maximum_drawdown),
        "maximum_drawdown_duration_trades": maximum_drawdown_duration,
        "turnover_eur": str(turnover),
        "capital_employed_eur": str(entry_capital),
        "return_on_capital_employed": (
            float(net / entry_capital) if entry_capital > ZERO else None
        ),
        "average_holding_seconds": (
            statistics.fmean(row.holding_seconds for row in complete)
            if complete
            else None
        ),
        "cost_drag_fraction_of_positive_gross_edge": (
            float(fees / pre_cost) if pre_cost > ZERO else None
        ),
        "cost_classification": (
            "GROSS_POSITIVE_NET_POSITIVE"
            if pre_cost > ZERO and net > ZERO
            else "GROSS_POSITIVE_NET_NEGATIVE"
            if pre_cost > ZERO and net <= ZERO
            else "GROSS_NEGATIVE"
        ),
        "sample_warning": (
            None
            if len(complete) >= MINIMUM_FAMILY_SAMPLE
            else "INSUFFICIENT_SAMPLE_FOR_STABLE_PROMOTION_INFERENCE"
        ),
    }
    return metrics


def aggregate_dimension(
    episodes: Sequence[CanonicalEconomicEpisode],
    attribute: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[CanonicalEconomicEpisode]] = defaultdict(list)
    for episode in episodes:
        groups[str(getattr(episode, attribute))].append(episode)
    rows = [
        {"dimension_value": value, **economic_metrics(group)}
        for value, group in groups.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            -float(_decimal(row.get("net_expectancy_eur")) or ZERO),
            -int(row["closed_episode_count"]),
            str(row["dimension_value"]),
        ),
    )


def aggregate_matrix(
    episodes: Sequence[CanonicalEconomicEpisode],
    row_attribute: str,
    column_attribute: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[CanonicalEconomicEpisode]] = defaultdict(
        list
    )
    for episode in episodes:
        groups[
            (
                str(getattr(episode, row_attribute)),
                str(getattr(episode, column_attribute)),
            )
        ].append(episode)
    return [
        {
            "row": row,
            "column": column,
            **economic_metrics(group),
        }
        for (row, column), group in sorted(groups.items())
    ]


def apply_mfe_mae(
    episodes: Sequence[CanonicalEconomicEpisode],
    normalized_root: Path,
) -> dict[str, Any]:
    """Attach causal 1m MFE/MAE only where closed candle data covers the hold."""

    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return {
            "status": "NOT_EVALUABLE_MISSING_PYARROW",
            "episode_count": len(episodes),
            "covered_episode_count": 0,
        }
    by_market: dict[str, list[CanonicalEconomicEpisode]] = defaultdict(list)
    for episode in episodes:
        by_market[episode.market].append(episode)
    covered = 0
    missing_files = 0
    missing_windows = 0
    for market, market_episodes in by_market.items():
        path = normalized_root / market / "1m.parquet"
        if not path.is_file():
            missing_files += len(market_episodes)
            continue
        start = min(row.entry_timestamp for row in market_episodes)
        end = max(row.exit_timestamp for row in market_episodes)
        start_text = start.isoformat().replace("+00:00", "Z")
        end_text = end.isoformat().replace("+00:00", "Z")
        try:
            table = parquet.read_table(
                path,
                columns=["timestamp", "closed", "values"],
                filters=[
                    ("timestamp", ">=", start_text),
                    ("timestamp", "<=", end_text),
                ],
            )
        except (OSError, TypeError, ValueError):
            missing_files += len(market_episodes)
            continue
        candles: list[tuple[datetime, Decimal, Decimal]] = []
        for row in table.to_pylist():
            timestamp = _timestamp(row.get("timestamp"))
            values = row.get("values")
            high = _decimal(values.get("high")) if isinstance(values, Mapping) else None
            low = _decimal(values.get("low")) if isinstance(values, Mapping) else None
            if timestamp is None or high is None or low is None:
                continue
            if row.get("closed") is False:
                continue
            candles.append((timestamp, high, low))
        for episode in market_episodes:
            selected = [
                candle
                for candle in candles
                if episode.entry_timestamp <= candle[0] <= episode.exit_timestamp
            ]
            if not selected or episode.average_entry_price <= ZERO:
                missing_windows += 1
                episode.market_path_evidence = (
                    "NO_POINT_IN_TIME_MARKET_DATA_FOR_HOLDING_WINDOW"
                )
                continue
            maximum = max(row[1] for row in selected)
            minimum = min(row[2] for row in selected)
            episode.mfe_pct = (
                (maximum - episode.average_entry_price)
                / episode.average_entry_price
                * Decimal("100")
            )
            episode.mae_pct = (
                (minimum - episode.average_entry_price)
                / episode.average_entry_price
                * Decimal("100")
            )
            episode.market_path_evidence = "CAUSAL_CLOSED_1M_CANDLES"
            covered += 1
    return {
        "status": "READY" if covered else "NOT_EVALUABLE_CURRENT_DATA_COVERAGE",
        "episode_count": len(episodes),
        "covered_episode_count": covered,
        "missing_file_episode_count": missing_files,
        "missing_window_episode_count": missing_windows,
        "future_data_used": False,
        "policy_changes": 0,
    }


def _promotion(metrics: Mapping[str, Any]) -> tuple[str, str]:
    count = int(metrics.get("closed_episode_count") or 0)
    net = _decimal(metrics.get("net_pnl_eur")) or ZERO
    profit_factor = metrics.get("profit_factor")
    if count >= MINIMUM_FAMILY_SAMPLE and net < ZERO:
        return "BLOCKED_NEGATIVE_EXPECTANCY", "PAUSE_PAPER_GENERATION"
    if count >= MINIMUM_FAMILY_SAMPLE and net > ZERO and profit_factor is not None and float(profit_factor) > 1.0:
        return "PAPER_POSITIVE", "CANDIDATE_FOR_PROMOTION"
    if count < MINIMUM_FAMILY_SAMPLE and net > ZERO:
        return "INSUFFICIENT_SAMPLE", "REQUIRES_MORE_SAMPLE"
    if count < MINIMUM_FAMILY_SAMPLE:
        return "INSUFFICIENT_SAMPLE", "REVALIDATE"
    return "RESEARCH_ONLY", "KEEP_RESEARCHING"


def promotion_table(family_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in family_rows:
        status, recommendation = _promotion(family)
        rows.append(
            {
                "strategy_family": family["dimension_value"],
                "promotion_status": status,
                "recommendation": recommendation,
                "closed_episode_count": family["closed_episode_count"],
                "net_pnl_eur": family["net_pnl_eur"],
                "net_expectancy_eur": family["net_expectancy_eur"],
                "profit_factor": family["profit_factor"],
                "live_validated": False,
                "automatic_authority_change": False,
            }
        )
    return rows


def family_overlap_and_incremental_value(
    episodes: Sequence[CanonicalEconomicEpisode],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[CanonicalEconomicEpisode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.strategy_family].append(episode)
    families = sorted(groups)
    pnl_bins: dict[str, dict[datetime, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for episode in episodes:
        if episode.net_pnl_eur is None:
            continue
        stamp = episode.exit_timestamp.replace(
            minute=(episode.exit_timestamp.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        pnl_bins[episode.strategy_family][stamp] += float(episode.net_pnl_eur)
    correlations: list[dict[str, Any]] = []
    for index, first in enumerate(families):
        for second in families[index + 1 :]:
            first_rows = groups[first]
            second_rows = groups[second]
            union_times = sorted(set(pnl_bins[first]) | set(pnl_bins[second]))
            first_series = [pnl_bins[first].get(stamp, 0.0) for stamp in union_times]
            second_series = [pnl_bins[second].get(stamp, 0.0) for stamp in union_times]
            correlation: float | None = None
            if len(union_times) >= MINIMUM_RATIO_SAMPLE:
                first_std = statistics.pstdev(first_series)
                second_std = statistics.pstdev(second_series)
                if first_std > 0 and second_std > 0:
                    first_mean = statistics.fmean(first_series)
                    second_mean = statistics.fmean(second_series)
                    covariance = statistics.fmean(
                        (left - first_mean) * (right - second_mean)
                        for left, right in zip(first_series, second_series, strict=True)
                    )
                    correlation = covariance / (first_std * second_std)
            first_overlapping = sum(
                any(
                    left.entry_timestamp <= right.exit_timestamp
                    and right.entry_timestamp <= left.exit_timestamp
                    for right in second_rows
                )
                for left in first_rows
            )
            second_overlapping = sum(
                any(
                    left.entry_timestamp <= right.exit_timestamp
                    and right.entry_timestamp <= left.exit_timestamp
                    for left in first_rows
                )
                for right in second_rows
            )
            overlap = statistics.fmean(
                (
                    first_overlapping / len(first_rows) if first_rows else 0.0,
                    second_overlapping / len(second_rows) if second_rows else 0.0,
                )
            )
            first_assets = {row.market for row in first_rows}
            second_assets = {row.market for row in second_rows}
            asset_union = first_assets | second_assets
            asset_jaccard = (
                len(first_assets & second_assets) / len(asset_union)
                if asset_union
                else 0.0
            )
            correlations.append(
                {
                    "family_a": first,
                    "family_b": second,
                    "pnl_correlation_15m": correlation,
                    "trade_interval_overlap_fraction": overlap,
                    "asset_jaccard": asset_jaccard,
                    "false_diversification_warning": bool(
                        correlation is not None
                        and correlation >= 0.8
                        and overlap >= 0.5
                        and asset_jaccard >= 0.5
                    ),
                    "regime_overlap": "NOT_EVALUABLE_INSUFFICIENT_CAUSAL_LABELS",
                    "timeframe_overlap": "NOT_EVALUABLE_INSUFFICIENT_CAUSAL_LABELS",
                }
            )
    total_metrics = economic_metrics(episodes)
    total_drawdown = _decimal(total_metrics["maximum_drawdown_eur"]) or ZERO
    incremental: list[dict[str, Any]] = []
    for family in families:
        without = [row for row in episodes if row.strategy_family != family]
        without_metrics = economic_metrics(without)
        without_drawdown = _decimal(without_metrics["maximum_drawdown_eur"]) or ZERO
        family_net = _decimal(economic_metrics(groups[family])["net_pnl_eur"]) or ZERO
        incremental.append(
            {
                "strategy_family": family,
                "standalone_net_pnl_eur": str(family_net),
                "portfolio_net_pnl_without_family_eur": without_metrics["net_pnl_eur"],
                "incremental_net_pnl_eur": str(family_net),
                "maximum_drawdown_change_when_removed_eur": str(
                    without_drawdown - total_drawdown
                ),
                "promotion_authority": False,
            }
        )
    return correlations, incremental


def signal_to_trade_funnel(
    lifecycle_path: Path,
    events: Sequence[Mapping[str, Any]],
    episodes: Sequence[CanonicalEconomicEpisode],
    *,
    canonical_state: CanonicalExecutionState | None = None,
    lifecycle_states: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    states: dict[str, set[str]] = defaultdict(set)
    if lifecycle_states is not None:
        for identity, observed in lifecycle_states.items():
            states[str(identity)].update(str(value).upper() for value in observed)
    elif lifecycle_path.is_file():
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            identity = str(event.get("opportunity_id") or "")
            if identity and event.get("to_state"):
                states[identity].add(str(event["to_state"]).upper())
    signal_created = set(states)
    setup_valid = {
        identity
        for identity, observed in states.items()
        if observed
        & {
            "WATCHING",
            "ARMED",
            "ENTRY_READY",
            "FILLED",
            "CLOSED",
        }
    }
    entry_ready = {
        identity
        for identity, observed in states.items()
        if "ENTRY_READY" in observed
    }
    intents = _intent_metadata(events)
    buy_intents = {
        str(row.get("signal_id"))
        for row in intents.values()
        if str(row.get("side") or "").upper() == "BUY" and row.get("signal_id")
    }
    state = canonical_state or replay_execution_events(events)
    full_fills = {
        str(fill.signal_id)
        for fill in state.fills.values()
        if fill.side == "BUY" and fill.signal_id
    }
    closed = {str(row.signal_id) for row in episodes if row.signal_id}
    nested = {
        "SIGNAL_CREATED": signal_created,
        "SETUP_VALID": setup_valid & signal_created,
        "ENTRY_READY": entry_ready & setup_valid,
        "ORDER_INTENT": buy_intents & entry_ready,
        "ORDER_SUBMITTED": buy_intents & entry_ready,
        "FULL_FILL": full_fills & buy_intents & entry_ready,
        "EXIT": closed & full_fills & buy_intents & entry_ready,
        "ROUNDTRIP_CLOSED": closed & full_fills & buy_intents & entry_ready,
    }
    ordered = list(nested)
    rows: list[dict[str, Any]] = []
    previous: set[str] | None = None
    for stage in ordered:
        selected = nested[stage]
        conversion = len(selected) / len(previous) if previous else None
        if conversion is not None and not 0.0 <= conversion <= 1.0:
            raise ValueError(f"non-nested funnel conversion escaped: {stage}")
        rows.append(
            {
                "stage": stage,
                "nested_population_count": len(selected),
                "conversion_from_previous": conversion,
            }
        )
        previous = selected
    return {
        "schema_version": "canonical_signal_trade_funnel_v1",
        "stages": rows,
        "not_recorded_stages": [
            "PORTFOLIO_APPROVED",
            "RISK_APPROVED",
            "PARTIAL_FILL",
            "POSITION_PROTECTED",
        ],
        "raw_observed_counts": {
            "signal_created": len(signal_created),
            "setup_valid": len(setup_valid),
            "entry_ready": len(entry_ready),
            "buy_order_intent": len(buy_intents),
            "buy_full_fill": len(full_fills),
            "roundtrip_closed": len(closed),
        },
        "population_policy": "EACH_CONVERSION_USES_INTERSECTION_WITH_PREVIOUS_STAGE",
        "operational_authority": False,
    }


def _lifecycle_states_from_bytes(content: bytes) -> dict[str, set[str]]:
    states: dict[str, set[str]] = defaultdict(set)
    for line in content.splitlines():
        if b'"to_state"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        identity = str(event.get("opportunity_id") or "")
        if identity and event.get("to_state"):
            states[identity].add(str(event["to_state"]).upper())
    return states


def tp_evidence(
    episodes: Sequence[CanonicalEconomicEpisode],
    *,
    telegram_ledger_path: Path,
    causal_snapshot_count: int,
    mfe_mae: Mapping[str, Any],
) -> dict[str, Any]:
    tp1 = sum("TAKE_PROFIT_1" in row.exit_reason_codes for row in episodes)
    tp2 = sum("TAKE_PROFIT_2" in row.exit_reason_codes for row in episodes)
    stopped = sum(
        bool({"HARD_STOP", "PROTECTIVE_STOP"} & set(row.exit_reason_codes))
        and not bool({"TAKE_PROFIT_1", "TAKE_PROFIT_2"} & set(row.exit_reason_codes))
        for row in episodes
    )
    notification_count = 0
    outcome_record_count = 0
    if telegram_ledger_path.is_file():
        for line in telegram_ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            notification_count += 1
            lowered = line.casefold()
            if any(token in lowered for token in ("tp2_hit", "tp2 bereikt", "target_2_hit")):
                outcome_record_count += 1
    closed = len(episodes)
    return {
        "telegram_claim_status": "NOT_EVALUABLE_FROM_TELEGRAM_LEDGER",
        "telegram_notification_record_count": notification_count,
        "telegram_outcome_record_count": outcome_record_count,
        "reason": (
            "notification messages contain proposed levels but no immutable, "
            "deduplicated outcome population linking every alert to TP/stop order"
        ),
        "actual_executed_trade_outcomes": {
            "closed_episode_count": closed,
            "tp1_observed_episode_count": tp1,
            "tp2_observed_episode_count": tp2,
            "tp1_observed_fraction": tp1 / closed if closed else None,
            "tp2_observed_fraction": tp2 / closed if closed else None,
            "stop_without_prior_recorded_target_count": stopped,
            "population": "CANONICAL_EXECUTED_PAPER_EPISODES",
        },
        "signal_outcome_evaluator": {
            "status": (
                "PARTIALLY_EVALUABLE"
                if int(mfe_mae.get("covered_episode_count") or 0) > 0
                else "NOT_EVALUABLE_CURRENT_MARKET_PATH_COVERAGE"
            ),
            "immutable_causal_snapshot_count": causal_snapshot_count,
            "market_path_covered_executed_episode_count": int(
                mfe_mae.get("covered_episode_count") or 0
            ),
            "signal_outcome_is_executed_trade_outcome": False,
            "future_information_used_at_signal_time": False,
        },
    }


def build_validation_backlog(
    registry: Mapping[str, Any],
    strategy_rows: Sequence[Mapping[str, Any]],
    strategy_dna_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    paper_by_strategy = {
        str(row["dimension_value"]): row for row in strategy_rows
    }
    paper_by_dna = {
        str(row["dimension_value"]): row
        for row in strategy_dna_rows
        if str(row.get("dimension_value") or "") not in {"", "None"}
    }
    selected: dict[str, dict[str, Any]] = {}
    for source in ("registered_pending", "economic_evidence"):
        for raw in registry.get(source) or []:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            identity = str(row.get("strategy_dna") or stable_hash(row, length=64))
            row["registry_source"] = source
            if identity not in selected or source == "economic_evidence":
                selected[identity] = row
    backlog: list[dict[str, Any]] = []
    for dna, row in selected.items():
        metadata = dict(row.get("metadata") or {})
        parameter_space = dict(metadata.get("parameter_space") or {})
        combinations = math.prod(
            len(values) if isinstance(values, list) and values else 1
            for values in parameter_space.values()
        ) if parameter_space else 1
        completed = 1 if row.get("registry_source") == "economic_evidence" else 0
        required = max(1, combinations)
        strategy_id = str(row.get("strategy_id") or "UNKNOWN_STRATEGY")
        paper = paper_by_dna.get(dna) or paper_by_strategy.get(strategy_id)
        family, _ = canonical_family(
            strategy_id,
            str(row.get("strategy_family") or metadata.get("family") or ""),
        )
        sample = int(row.get("sample_count") or 0)
        normal_pf = float(row.get("normal_profit_factor") or 0.0)
        promise_score = (
            50.0 * int(bool(row.get("backtest_positive")))
            + min(25.0, math.log1p(sample) * 4.0)
            + min(20.0, max(0.0, normal_pf - 1.0) * 10.0)
            + (20.0 if paper and (_decimal(paper.get("net_pnl_eur")) or ZERO) > ZERO else 0.0)
            - min(20.0, max(0, required - completed) / 10.0)
        )
        backlog.append(
            {
                "strategy_id": strategy_id,
                "strategy_dna_hash": dna,
                "strategy_family": family,
                "asset_set": row.get("markets")
                or ([row.get("market")] if row.get("market") else []),
                "timeframe": row.get("timeframe"),
                "parameter_combination_count": combinations,
                "required_backtests": required,
                "completed_backtests": completed,
                "pending_backtests": max(0, required - completed),
                "estimated_validation_cost": "NOT_MEASURABLE_FROM_CURRENT_ARTIFACTS",
                "promotion_status": row.get("lifecycle_state") or "UNKNOWN",
                "canonical_paper_episode_count": int(
                    (paper or {}).get("closed_episode_count") or 0
                ),
                "economic_promise_score": promise_score,
                "rank_inputs": {
                    "backtest_positive": bool(row.get("backtest_positive")),
                    "historical_sample_count": sample,
                    "normal_profit_factor": row.get("normal_profit_factor"),
                    "paper_evidence_available": paper is not None,
                    "incremental_diversification_value": "NOT_EVALUABLE",
                },
            }
        )
    return sorted(
        backlog,
        key=lambda row: (
            -float(row["economic_promise_score"]),
            int(row["pending_backtests"]),
            str(row["strategy_id"]),
        ),
    )


def _comparison(
    canonical: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "closed_episode_count": ("closed_round_trips", Decimal("0")),
        "net_pnl_eur": ("net_pnl_eur", Decimal("0.00000001")),
        "fees_eur": ("fees_eur", Decimal("0.00000001")),
    }
    rows: list[dict[str, Any]] = []
    for name, (legacy_name, tolerance) in fields.items():
        current = _decimal(canonical.get(name))
        legacy = _decimal(previous.get(legacy_name))
        classification = (
            "EXACT_MATCH"
            if current is not None
            and legacy is not None
            and abs(current - legacy) <= tolerance
            else "MISSING_HISTORICAL_EVIDENCE"
            if legacy is None
            else "LEGACY_ATTRIBUTION_DEFECT"
        )
        rows.append(
            {
                "metric": name,
                "previous_value": str(legacy) if legacy is not None else None,
                "canonical_value": str(current) if current is not None else None,
                "classification": classification,
            }
        )
    previous_gross_expectancy = _decimal(previous.get("gross_expectancy_eur"))
    previous_count = _decimal(previous.get("closed_round_trips"))
    previous_gross = (
        previous_gross_expectancy * previous_count
        if previous_gross_expectancy is not None and previous_count is not None
        else None
    )
    current_gross = _decimal(canonical.get("gross_pnl_before_costs_eur"))
    rows.append(
        {
            "metric": "gross_pnl_before_costs_eur",
            "previous_value": str(previous_gross) if previous_gross is not None else None,
            "canonical_value": str(current_gross) if current_gross is not None else None,
            "classification": (
                "EXACT_MATCH"
                if previous_gross is not None
                and current_gross is not None
                and abs(previous_gross - current_gross) <= Decimal("0.00000001")
                else "MISSING_HISTORICAL_EVIDENCE"
            ),
        }
    )
    return {
        "status": (
            "EXACT_MATCH"
            if all(row["classification"] == "EXACT_MATCH" for row in rows)
            else "DIVERGENCE_REQUIRES_CLASSIFICATION"
        ),
        "rows": rows,
        "allowed_classifications": [
            "EXACT_MATCH",
            "EXPECTED_ACCOUNTING_DIFFERENCE",
            "LEGACY_ATTRIBUTION_DEFECT",
            "MISSING_HISTORICAL_EVIDENCE",
            "CANONICAL_RECONSTRUCTION_DEFECT",
        ],
    }


def _signal_failure_attribution(
    episodes: Sequence[CanonicalEconomicEpisode],
) -> dict[str, Any]:
    rows = Counter()
    for episode in episodes:
        if episode.net_pnl_eur is None or episode.net_pnl_eur >= ZERO:
            continue
        if episode.gross_pnl_before_costs_eur > ZERO:
            rows["EXECUTION_FAILURE_COST_DRAG"] += 1
        elif episode.exit_type in {"HARD_STOP", "PROTECTIVE_STOP"}:
            rows["ALPHA_FAILURE_WRONG_DIRECTION_OR_STOP"] += 1
        elif episode.exit_type == "SIGNAL_DETERIORATION":
            rows["ALPHA_OR_EXIT_LOGIC_FAILURE"] += 1
        elif episode.exit_type == "TIME_EXIT":
            rows["ALPHA_STAGNATION_OR_TIME_EXIT"] += 1
        else:
            rows["UNKNOWN"] += 1
    return {
        "method": "EVIDENCE_BOUND_PRIMARY_CAUSE_CLASSIFICATION",
        "counts": dict(sorted(rows.items())),
        "latency_attribution": "NOT_EVALUABLE_NO_DECISION_TO_FILL_TIMESTAMPS_FOR_PAPER",
        "spread_attribution": "PARTIAL_ONLY_CAUSAL_SNAPSHOTS",
        "automatic_policy_changes": False,
    }


def build_canonical_strategy_economics(
    settings: Settings,
    *,
    paper_ledger_path: Path | None = None,
    lifecycle_path: Path | None = None,
    include_mfe_mae: bool = True,
) -> dict[str, Any]:
    """Build and persist one immutable P0.5 evidence run."""

    paper_paths = (
        [paper_ledger_path]
        if paper_ledger_path is not None
        else [
            settings.paths.output_dir
            / "paper"
            / "event_driven_playbook_execution.jsonl",
            settings.paths.output_dir
            / "paper"
            / "generated_strategy_execution.jsonl",
        ]
    )
    paper_sources = [
        (path, path.read_bytes())
        for path in paper_paths
        if path is not None and path.is_file()
    ]
    lifecycle = lifecycle_path or (
        settings.paths.output_dir
        / "live"
        / "events"
        / "opportunity_lifecycle.jsonl"
    )
    if not paper_sources:
        raise FileNotFoundError(
            "no canonical paper execution ledger found: "
            + ", ".join(str(path) for path in paper_paths if path is not None)
        )
    paper_bytes = b"".join(content for _, content in paper_sources)
    lifecycle_bytes = lifecycle.read_bytes() if lifecycle.is_file() else b""
    source_events = [
        (path, _read_jsonl_bytes(content))
        for path, content in paper_sources
    ]
    events = [event for _, rows in source_events for event in rows]
    snapshots = _load_causal_snapshots_bytes(lifecycle_bytes)
    lifecycle_states = _lifecycle_states_from_bytes(lifecycle_bytes)
    source_states: list[tuple[Path, CanonicalExecutionState]] = []
    episodes: list[CanonicalEconomicEpisode] = []
    unknowns: list[dict[str, Any]] = []
    for path, rows in source_events:
        source_state, source_episodes, source_unknowns = reconstruct_canonical_episodes(
            rows,
            evidence_layer="PAPER",
            causal_snapshots=snapshots,
        )
        source_replay_hash = assert_replay_deterministic(rows)
        if source_replay_hash != source_state.state_hash:
            raise AssertionError(
                f"economic replay hash differs from canonical state: {path}"
            )
        source_states.append((path, source_state))
        episodes.extend(source_episodes)
        unknowns.extend(
            {**row, "paper_execution_source": str(path.resolve())}
            for row in source_unknowns
        )
    replay_hash = stable_hash(
        [
            {"path": str(path.resolve()), "state_hash": state.state_hash}
            for path, state in source_states
        ],
        length=64,
    )
    funnel_state = CanonicalExecutionState()
    for path, source_state in source_states:
        overlap = set(funnel_state.fills) & set(source_state.fills)
        if overlap:
            raise ValueError(
                f"duplicate fill identity across paper ledgers: {path}: {sorted(overlap)}"
            )
        funnel_state.fills.update(source_state.fills)
    mfe_mae = (
        apply_mfe_mae(
            episodes,
            settings.paths.processed_data_dir / "bitvavo",
        )
        if include_mfe_mae
        else {
            "status": "NOT_REQUESTED",
            "episode_count": len(episodes),
            "covered_episode_count": 0,
        }
    )
    aggregate = economic_metrics(episodes)
    family_results = aggregate_dimension(episodes, "strategy_family")
    strategy_results = aggregate_dimension(episodes, "strategy_id")
    strategy_dna_results = aggregate_dimension(episodes, "strategy_dna_hash")
    asset_results = aggregate_dimension(episodes, "market")
    timeframe_results = aggregate_dimension(episodes, "timeframe")
    regime_results = aggregate_dimension(episodes, "regime")
    entry_results = aggregate_dimension(episodes, "entry_type")
    exit_results = aggregate_dimension(episodes, "exit_type")
    family_asset = aggregate_matrix(episodes, "strategy_family", "market")
    family_timeframe = aggregate_matrix(
        episodes,
        "strategy_family",
        "timeframe",
    )
    overlap, incremental = family_overlap_and_incremental_value(episodes)
    promotions = promotion_table(family_results)
    previous_layers_path = (
        settings.paths.output_dir / "operations" / "execution_evidence_layers.json"
    )
    registry_path = (
        settings.paths.output_dir / "strategies" / "all_strategy_dna.json"
    )
    training_path = (
        settings.paths.output_dir
        / "intelligence"
        / "opportunity_training_rows.json"
    )
    model_status_path = (
        settings.paths.output_dir / "intelligence" / "model_status.json"
    )
    authority_path = (
        settings.paths.output_dir
        / "governance"
        / "positive_strategy_live_authority.json"
    )
    frozen_json_sources = {
        "execution_evidence_layers": (
            previous_layers_path.read_bytes()
            if previous_layers_path.is_file()
            else b""
        ),
        "strategy_registry": (
            registry_path.read_bytes() if registry_path.is_file() else b""
        ),
        "training_rows": (
            training_path.read_bytes() if training_path.is_file() else b""
        ),
        "model_status": (
            model_status_path.read_bytes() if model_status_path.is_file() else b""
        ),
        "live_authority": (
            authority_path.read_bytes() if authority_path.is_file() else b""
        ),
    }
    previous_layers = _read_json_bytes(
        frozen_json_sources["execution_evidence_layers"]
    )
    previous = dict(previous_layers.get("simulated_execution_pnl") or {})
    comparison = _comparison(aggregate, previous)
    funnel = signal_to_trade_funnel(
        lifecycle,
        events,
        episodes,
        canonical_state=funnel_state,
        lifecycle_states=lifecycle_states,
    )
    tp = tp_evidence(
        episodes,
        telegram_ledger_path=(
            settings.paths.output_dir
            / "notifications"
            / "telegram_notifications.jsonl"
        ),
        causal_snapshot_count=len(snapshots),
        mfe_mae=mfe_mae,
    )
    registry = _read_json_bytes(frozen_json_sources["strategy_registry"])
    backlog = build_validation_backlog(
        registry,
        strategy_results,
        strategy_dna_results,
    )
    training = _read_json_bytes(frozen_json_sources["training_rows"])
    model_status = _read_json_bytes(frozen_json_sources["model_status"])
    authority = _read_json_bytes(frozen_json_sources["live_authority"])
    evidence_hashes = {
        "paper_execution_ledger_sha256": sha256(paper_bytes).hexdigest(),
        "paper_execution_ledger_byte_count": len(paper_bytes),
        "paper_execution_ledgers": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(content).hexdigest(),
                "byte_count": len(content),
                "event_count": len(_read_jsonl_bytes(content)),
            }
            for path, content in paper_sources
        ],
        "lifecycle_ledger_sha256": sha256(lifecycle_bytes).hexdigest()
        if lifecycle_bytes
        else None,
        "lifecycle_ledger_byte_count": len(lifecycle_bytes),
        **{
            f"{name}_sha256": sha256(content).hexdigest() if content else None
            for name, content in frozen_json_sources.items()
        },
        **{
            f"{name}_byte_count": len(content)
            for name, content in frozen_json_sources.items()
        },
    }
    run_id = stable_hash(
        [ECONOMIC_SCHEMA_VERSION, evidence_hashes],
        length=32,
    )
    data_cutoff = max(
        (episode.exit_timestamp for episode in episodes),
        default=None,
    )
    canonical_counts = Counter(row.reconstruction_status for row in episodes)
    payload: dict[str, Any] = {
        "schema_version": ECONOMIC_SCHEMA_VERSION,
        "economic_schema_version": ECONOMIC_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_iso(),
        "data_cutoff": (
            data_cutoff.isoformat().replace("+00:00", "Z")
            if data_cutoff
            else None
        ),
        "canonical_state_version": CanonicalExecutionState().schema_version,
        "canonical_state_hash": replay_hash,
        "replay_hash": replay_hash,
        "replay_deterministic": True,
        "source_event_count": len(events),
        "canonical_fill_count": sum(
            len(state.fills) for _, state in source_states
        ),
        "canonical_sell_fill_count": sum(
            len(state.realized_pnl_events) for _, state in source_states
        ),
        "canonical_open_position_count": sum(
            position.quantity > ZERO
            for _, state in source_states
            for position in state.positions.values()
        ),
        "open_positions_are_not_realized_economic_labels": True,
        "closed_episode_count": len(episodes),
        "reconstructable_episode_count": canonical_counts[
            "CANONICALLY_RECONSTRUCTABLE"
        ],
        "partially_reconstructable_episode_count": canonical_counts[
            "PARTIALLY_RECONSTRUCTABLE"
        ],
        "not_reconstructable_evidence_count": sum(
            row.get("classification") == "NOT_RECONSTRUCTABLE"
            for row in unknowns
        ),
        "aggregate": aggregate,
        "previous_vs_canonical": comparison,
        "family_results": family_results,
        "strategy_results": strategy_results,
        "strategy_dna_results": strategy_dna_results,
        "asset_results": asset_results,
        "timeframe_results": timeframe_results,
        "regime_results": regime_results,
        "entry_results": entry_results,
        "exit_results": exit_results,
        "family_asset_matrix": family_asset,
        "family_timeframe_matrix": family_timeframe,
        "mfe_mae": mfe_mae,
        "tp1_tp2_evidence": tp,
        "signal_vs_execution_failure": _signal_failure_attribution(episodes),
        "signal_to_trade_funnel": funnel,
        "family_overlap_correlation": overlap,
        "incremental_portfolio_value": incremental,
        "promotion_recommendations": promotions,
        "validation_backlog": backlog,
        "validation_backlog_count": len(backlog),
        "ml_status": {
            "authority": "SHADOW_ONLY",
            "complete_labeled_episode_count": int(
                training.get("complete_labeled_episodes") or 0
            ),
            "genuine_fill_labeled_episode_count": int(
                training.get("fill_labeled_episodes") or 0
            ),
            "shadow_labeled_episode_count": int(
                training.get("shadow_labeled_episodes") or 0
            ),
            "model_status": model_status.get("status") or "NOT_AVAILABLE",
            "neural_network": "NOT_EVALUABLE",
            "transformer": "NOT_EVALUABLE",
            "mixture_of_experts": "NOT_EVALUABLE",
            "reinforcement_learning": "NOT_EVALUABLE",
            "authority_changes": 0,
        },
        "prospective_evidence_policy": {
            "retrospective_backtest_pooled_with_paper": False,
            "paper_pooled_with_live": False,
            "shadow_pooled_with_economic_labels": False,
            "live_validated_family_count": 0,
        },
        "evidence_layers": {
            "RETROSPECTIVE_BACKTEST": {
                "status": "SEPARATE_NOT_POOLED",
                "economic_candidate_count": len(
                    registry.get("economic_evidence") or []
                ),
            },
            "PAPER": {
                "status": "CANONICAL_EXECUTION_ECONOMICS",
                **aggregate,
            },
            "SHADOW": {
                "status": "SEPARATE_NOT_ECONOMIC_AUTHORITY",
                "labeled_episode_count": int(
                    training.get("shadow_labeled_episodes") or 0
                ),
            },
            "LIVE_CANARY": dict(
                previous_layers.get("actual_live_pnl") or {}
            ),
            "LIVE": {
                "status": "NOT_EVALUABLE_NO_LIVE_VALIDATED_FAMILY",
                "live_validated_family_count": 0,
            },
        },
        "vectorbt_stage0_contract": {
            "status": "INTERFACE_PREPARED_NOT_EXECUTED",
            "ohlcv_schema": "canonical_market,timestamp,open,high,low,close,volume,closed",
            "signal_schema": "signal_id,decision_timestamp,market,timeframe,strategy_family,parameters",
            "cost_schema": "fee_bps,spread_bps,slippage_bps,market_capacity_eur",
            "parameter_schema": "strategy_dna_hash,parameter_name,value",
            "dimensions": ["strategy_family", "market", "timeframe"],
            "candidate_output_schema": "candidate_id,rejection_reason,screen_metrics,evidence_hash",
            "production_authority": False,
        },
        "unknowns": unknowns,
        "episodes": [row.to_dict() for row in episodes],
        "evidence_hashes": evidence_hashes,
        "authority_snapshot": {
            "active": authority.get("active") is True,
            "approved_candidate_count": len(authority.get("approved_candidates") or []),
            "modified_by_analysis": False,
        },
        "safety": {
            "orders_generated": 0,
            "orders_submitted": 0,
            "orders_cancelled": 0,
            "protective_orders_modified": 0,
            "private_bitvavo_mutations": 0,
            "live_authority_increases": 0,
            "risk_limit_increases": 0,
            "shariah_policy_changes": 0,
        },
    }
    payload["artifact_hash"] = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"artifact_hash", "created_at"}
        },
        length=64,
    )
    root = settings.paths.output_dir / "economics"
    run_dir = root / "runs" / run_id
    artifact_path = run_dir / "canonical_strategy_family_economics.json"
    if artifact_path.is_file():
        existing = _read_json(artifact_path)
        if existing.get("artifact_hash") != payload["artifact_hash"]:
            # created_at is intentionally excluded when comparing a rerun of
            # the same immutable inputs.
            comparable_existing = {
                key: value
                for key, value in existing.items()
                if key not in {"created_at", "artifact_hash"}
            }
            comparable_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"created_at", "artifact_hash"}
            }
            if stable_hash(comparable_existing) != stable_hash(comparable_payload):
                raise FileExistsError(
                    f"immutable economic artifact collision: {artifact_path}"
                )
        payload = existing
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact_path, payload)
    family_negative = [
        row
        for row in family_results
        if (_decimal(row.get("net_pnl_eur")) or ZERO) < ZERO
    ]
    family_positive = [
        row
        for row in family_results
        if (_decimal(row.get("net_pnl_eur")) or ZERO) > ZERO
    ]
    paper_30d = economic_metrics(
        [
            row
            for row in episodes
            if data_cutoff is not None
            and row.exit_timestamp >= data_cutoff - timedelta(days=30)
        ]
    )
    actual_live = dict(previous_layers.get("actual_live_pnl") or {})
    operator_summary = {
        "schema_version": "canonical_economics_operator_summary_v1",
        "run_id": run_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": payload["artifact_hash"],
        "canonical_closed_trades": len(episodes),
        "canonical_open_positions": sum(
            position.quantity > ZERO
            for _, state in source_states
            for position in state.positions.values()
        ),
        "paper_execution_source_count": len(paper_sources),
        "paper_net_pnl_eur": aggregate["net_pnl_eur"],
        "paper_30d_net_pnl_eur": paper_30d["net_pnl_eur"],
        "paper_profit_factor": aggregate["profit_factor"],
        "live_canary_net_pnl_eur": actual_live.get("net_pnl_eur"),
        "live_canary_realised_pnl_eur": actual_live.get("realised_pnl_eur"),
        "best_validated_family": None,
        "best_evidenced_family": (
            family_positive[0]["dimension_value"] if family_positive else None
        ),
        "worst_family": (
            min(
                family_negative,
                key=lambda row: _decimal(row.get("net_pnl_eur")) or ZERO,
            )["dimension_value"]
            if family_negative
            else None
        ),
        "family_sample_counts": {
            str(row["dimension_value"]): int(row["closed_episode_count"])
            for row in family_results
        },
        "live_validated_family_count": 0,
        "current_live_authority_active": authority.get("active") is True,
        "ml_authority": "SHADOW_ONLY",
        "automatic_authority_changes": False,
        "telegram_research_notifications_sent": 0,
    }
    atomic_write_json(root / "latest.json", operator_summary)
    return {
        "status": "COMPLETE",
        "run_id": run_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_hash": payload["artifact_hash"],
        "canonical_state_hash": replay_hash,
        "closed_episode_count": len(episodes),
        "reconstructable_episode_count": canonical_counts[
            "CANONICALLY_RECONSTRUCTABLE"
        ],
        "aggregate": aggregate,
        "family_result_count": len(family_results),
        "strategy_result_count": len(strategy_results),
        "asset_result_count": len(asset_results),
        "orders_generated": 0,
        "orders_submitted": 0,
        "private_exchange_mutations": 0,
    }


__all__ = [
    "CanonicalEconomicEpisode",
    "ECONOMIC_SCHEMA_VERSION",
    "aggregate_dimension",
    "aggregate_matrix",
    "apply_mfe_mae",
    "build_canonical_strategy_economics",
    "build_validation_backlog",
    "canonical_family",
    "classify_entry_type",
    "classify_exit_type",
    "economic_metrics",
    "family_overlap_and_incremental_value",
    "load_causal_snapshots",
    "promotion_table",
    "reconstruct_canonical_episodes",
    "signal_to_trade_funnel",
    "tp_evidence",
]
