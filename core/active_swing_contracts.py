"""Canonical contracts for the retail crypto active-swing product.

These schemas normalize evidence and lifecycle state.  They never size orders,
grant authority, mutate portfolio state or communicate with Bitvavo.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from core.contracts import FrozenModel, normalize_market, require_utc
from utils.common import stable_hash

ZERO = Decimal("0")
ONE = Decimal("1")
CORE_DECISION_TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d", "1W")
TACTICAL_EXECUTION_TIMEFRAMES = ("tick", "1m")


class SwingLifecycle(StrEnum):
    DISCOVERED = "DISCOVERED"
    WATCHING = "WATCHING"
    NEAR_SETUP = "NEAR_SETUP"
    SETUP_VALID = "SETUP_VALID"
    ENTRY_TRIGGER_PENDING = "ENTRY_TRIGGER_PENDING"
    ENTRY_READY = "ENTRY_READY"
    ENTERED = "ENTERED"
    POSITION_ACTIVE = "POSITION_ACTIVE"
    PROFIT_PROTECTION = "PROFIT_PROTECTION"
    REDUCE_CANDIDATE = "REDUCE_CANDIDATE"
    ROTATION_CANDIDATE = "ROTATION_CANDIDATE"
    EXIT_READY = "EXIT_READY"
    CLOSED = "CLOSED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class ThesisState(StrEnum):
    THESIS_INTACT = "THESIS_INTACT"
    THESIS_WEAKENING = "THESIS_WEAKENING"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"


class TimeframeContract(FrozenModel):
    entry_timeframe: str
    setup_timeframe: str
    context_timeframes: tuple[str, ...]
    structural_timeframe: str
    management_timeframe: str
    exit_timeframe: str
    required_timeframes: tuple[str, ...]
    optional_timeframes: tuple[str, ...] = ()
    maximum_signal_age_seconds: int = Field(gt=0)
    maximum_setup_age_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_roles(self) -> "TimeframeContract":
        required = set(self.required_timeframes)
        if self.entry_timeframe not in CORE_DECISION_TIMEFRAMES:
            raise ValueError("entry timeframe must be a core active-swing timeframe")
        if not {
            self.entry_timeframe,
            self.setup_timeframe,
            self.structural_timeframe,
            self.management_timeframe,
            self.exit_timeframe,
        }.issubset(required | set(self.optional_timeframes)):
            raise ValueError("every timeframe role must be declared required or optional")
        if required & set(self.optional_timeframes):
            raise ValueError("required and optional timeframes must be disjoint")
        return self


class TimeframeObservation(FrozenModel):
    market: str
    timeframe: str
    bar_close_time: datetime
    available_at: datetime
    source: str
    quality: str
    features: dict[str, Any]

    _market = field_validator("market")(normalize_market)
    _close = field_validator("bar_close_time")(require_utc)
    _available = field_validator("available_at")(require_utc)

    @model_validator(mode="after")
    def availability_follows_close(self) -> "TimeframeObservation":
        if self.available_at < self.bar_close_time:
            raise ValueError("timeframe observation cannot be available before bar close")
        return self


class PositionThesis(FrozenModel):
    thesis_id: str
    market: str
    strategy_id: str
    strategy_dna_hash: str
    timeframe_contract: TimeframeContract
    entered_at: datetime
    entry_price: Decimal = Field(gt=ZERO)
    expected_net_r_at_entry: Decimal | None = None
    expected_holding_seconds: int = Field(gt=0)
    stop: Decimal = Field(gt=ZERO)
    targets: tuple[Decimal, ...]
    entry_reason_codes: tuple[str, ...]
    entry_feature_snapshot_id: str
    model_versions: tuple[str, ...] = ()
    original_thesis: str
    current_thesis_state: ThesisState = ThesisState.THESIS_INTACT
    current_thesis_reason_codes: tuple[str, ...] = ()

    _market = field_validator("market")(normalize_market)
    _entered = field_validator("entered_at")(require_utc)

    @model_validator(mode="after")
    def valid_spot_long_levels(self) -> "PositionThesis":
        if self.stop >= self.entry_price:
            raise ValueError("spot-long stop must be below entry")
        if not self.targets or any(target <= self.entry_price for target in self.targets):
            raise ValueError("spot-long targets must be above entry")
        return self


class ActiveSwingOpportunity(FrozenModel):
    opportunity_contract_id: str
    market: str
    base_asset: str
    quote_asset: Literal["EUR"]
    strategy_id: str
    strategy_family: str
    strategy_dna_hash: str
    timeframe_contract: TimeframeContract
    lifecycle: SwingLifecycle
    decision_time: datetime
    signal_timestamp: datetime
    signal_age_seconds: int = Field(ge=0)
    setup_origin_timestamp: datetime
    setup_age_seconds: int = Field(ge=0)
    setup_hash: str
    valid_until: datetime
    entry_price_reference: Decimal = Field(gt=ZERO)
    stop: Decimal = Field(gt=ZERO)
    target_1: Decimal = Field(gt=ZERO)
    target_2: Decimal | None = Field(default=None, gt=ZERO)
    risk_per_unit: Decimal = Field(gt=ZERO)
    expected_holding_seconds: int = Field(gt=0)
    expected_net_return: Decimal | None = None
    expected_net_r: Decimal | None = None
    expected_mae: Decimal | None = None
    expected_mfe: Decimal | None = None
    current_regime: str
    btc_context: dict[str, Any]
    market_breadth_context: dict[str, Any]
    multi_timeframe_alignment: Decimal | None = None
    multi_timeframe_conflict: bool
    liquidity: dict[str, Any]
    spread_bps: Decimal | None = Field(default=None, ge=ZERO)
    depth_eur: Decimal | None = Field(default=None, ge=ZERO)
    slippage_estimate_bps: Decimal | None = Field(default=None, ge=ZERO)
    fee_estimate_bps: Decimal | None = Field(default=None, ge=ZERO)
    theoretical_target_quantity: Decimal | None = Field(default=None, ge=ZERO)
    bitvavo_realizable_quantity: Decimal | None = Field(default=None, ge=ZERO)
    minimum_order_feasible: bool | None
    quantity_precision_valid: bool | None
    portfolio_fit: str
    correlation_cluster: str | None
    shariah_status: str
    authority_status: str
    data_health: str
    execution_health: str
    retail_realizable: bool
    evidence_ids: tuple[str, ...]
    spot_only: Literal[True] = True
    leverage: Literal[False] = False
    shorting: Literal[False] = False
    execution_authority: Literal[False] = False

    _market = field_validator("market")(normalize_market)
    _decision = field_validator("decision_time")(require_utc)
    _signal = field_validator("signal_timestamp")(require_utc)
    _setup = field_validator("setup_origin_timestamp")(require_utc)
    _valid = field_validator("valid_until")(require_utc)

    @model_validator(mode="after")
    def validate_contract(self) -> "ActiveSwingOpportunity":
        base, quote = self.market.split("-")
        if self.base_asset != base or self.quote_asset != quote:
            raise ValueError("base/quote assets must match market")
        if self.signal_timestamp > self.decision_time:
            raise ValueError("signal cannot occur after decision time")
        if self.setup_origin_timestamp > self.decision_time:
            raise ValueError("setup cannot occur after decision time")
        if self.valid_until <= self.decision_time:
            raise ValueError("opportunity must remain valid after decision time")
        if self.stop >= self.entry_price_reference:
            raise ValueError("spot-long stop must be below entry")
        if self.target_1 <= self.entry_price_reference:
            raise ValueError("spot-long target must be above entry")
        if self.target_2 is not None and self.target_2 <= self.target_1:
            raise ValueError("target 2 must exceed target 1")
        if self.signal_age_seconds > self.timeframe_contract.maximum_signal_age_seconds:
            raise ValueError("signal exceeds timeframe contract age")
        if self.setup_age_seconds > self.timeframe_contract.maximum_setup_age_seconds:
            raise ValueError("setup exceeds timeframe contract age")
        if self.retail_realizable:
            if not all(
                value is True
                for value in (self.minimum_order_feasible, self.quantity_precision_valid)
            ):
                raise ValueError("retail-realizable opportunity must pass venue constraints")
            if self.bitvavo_realizable_quantity in (None, ZERO):
                raise ValueError("retail-realizable opportunity requires positive quantity")
            if self.expected_net_return is None or self.expected_net_return <= ZERO:
                raise ValueError("retail-realizable opportunity requires positive net edge")
        return self


def causal_asof_join(
    *,
    decision_time: datetime,
    observations: Iterable[TimeframeObservation],
    required_timeframes: Iterable[str],
) -> dict[str, TimeframeObservation]:
    """Select the latest fully available closed observation for each timeframe."""

    decision_time = require_utc(decision_time)
    selected: dict[str, TimeframeObservation] = {}
    for row in observations:
        if row.bar_close_time > decision_time or row.available_at > decision_time:
            continue
        current = selected.get(row.timeframe)
        if current is None or (row.bar_close_time, row.available_at) > (
            current.bar_close_time,
            current.available_at,
        ):
            selected[row.timeframe] = row
    missing = sorted(set(required_timeframes) - set(selected))
    if missing:
        raise ValueError(f"missing required causal timeframes: {','.join(missing)}")
    return selected


def normalize_live_opportunity(
    row: Mapping[str, Any],
    *,
    decision_time: datetime | str,
) -> ActiveSwingOpportunity:
    """Map one existing event-driven opportunity into the canonical product contract."""

    decision_time = _utc_from(decision_time)
    market = normalize_market(str(row.get("market") or ""))
    base, _quote = market.split("-")
    parent = dict(row.get("higher_timeframe_parent") or {})
    economics = dict(row.get("execution_economics") or {})
    realtime = dict(row.get("realtime_inputs") or {})
    entry_timeframe = str(
        parent.get("entry_timeframe")
        or row.get("context_timeframe")
        or "15m"
    )
    setup_timeframe = str(parent.get("confirmation_timeframe") or entry_timeframe)
    structural = str(parent.get("regime_timeframe") or setup_timeframe)
    required = tuple(dict.fromkeys((entry_timeframe, setup_timeframe, structural)))
    detected = _utc_from(row.get("detected_at") or decision_time)
    setup_origin = _utc_from(row.get("setup_detected_ts") or detected)
    valid_until = _utc_from(row.get("valid_until"))
    maximum_signal_age = max(1, int((valid_until - detected).total_seconds()))
    maximum_setup_age = max(maximum_signal_age, int((valid_until - setup_origin).total_seconds()))
    timeframe_contract = TimeframeContract(
        entry_timeframe=entry_timeframe,
        setup_timeframe=setup_timeframe,
        context_timeframes=tuple(dict.fromkeys((setup_timeframe, structural))),
        structural_timeframe=structural,
        management_timeframe=setup_timeframe,
        exit_timeframe=setup_timeframe,
        required_timeframes=required,
        maximum_signal_age_seconds=maximum_signal_age,
        maximum_setup_age_seconds=maximum_setup_age,
    )
    entry = Decimal(str(row.get("entry_price") or "0"))
    stop = Decimal(str(row.get("stop_loss") or "0"))
    target_1 = Decimal(str(economics.get("target_1") or row.get("take_profit_1") or "0"))
    target_2_raw = economics.get("target_2") or row.get("take_profit_2")
    target_2 = Decimal(str(target_2_raw)) if target_2_raw is not None else None
    state_map = {
        "DISCOVERED": SwingLifecycle.DISCOVERED,
        "WATCHING": SwingLifecycle.WATCHING,
        "ARMED": SwingLifecycle.ENTRY_TRIGGER_PENDING,
        "ENTRY_READY": SwingLifecycle.ENTRY_READY,
        "ORDER_INTENT_CREATED": SwingLifecycle.ENTERED,
        "ORDER_SUBMITTED": SwingLifecycle.ENTERED,
        "PARTIALLY_FILLED": SwingLifecycle.ENTERED,
        "FILLED": SwingLifecycle.POSITION_ACTIVE,
        "MANAGING": SwingLifecycle.POSITION_ACTIVE,
        "EXITING": SwingLifecycle.EXIT_READY,
        "CLOSED": SwingLifecycle.CLOSED,
        "INVALIDATED": SwingLifecycle.INVALIDATED,
        "EXPIRED": SwingLifecycle.EXPIRED,
    }
    expected_net_bps = economics.get("expected_net_value_bps")
    fee_bps = economics.get("roundtrip_fee_bps") or economics.get("fee_bps")
    identity = {
        "source_opportunity": row.get("opportunity_id"),
        "market": market,
        "strategy": row.get("playbook_id"),
        "setup_origin": setup_origin.isoformat(),
        "timeframes": timeframe_contract.model_dump(mode="json"),
    }
    return ActiveSwingOpportunity(
        opportunity_contract_id=f"active_swing_{stable_hash(identity, length=40)}",
        market=market,
        base_asset=base,
        quote_asset="EUR",
        strategy_id=str(row.get("playbook_id") or row.get("context_strategy") or "UNKNOWN"),
        strategy_family=str(row.get("family") or "UNKNOWN"),
        strategy_dna_hash=str(row.get("playbook_dna") or "UNKNOWN"),
        timeframe_contract=timeframe_contract,
        lifecycle=state_map.get(str(row.get("state") or ""), SwingLifecycle.DISCOVERED),
        decision_time=decision_time,
        signal_timestamp=detected,
        signal_age_seconds=max(0, int((decision_time - detected).total_seconds())),
        setup_origin_timestamp=setup_origin,
        setup_age_seconds=max(0, int((decision_time - setup_origin).total_seconds())),
        setup_hash=stable_hash(identity, length=64),
        valid_until=valid_until,
        entry_price_reference=entry,
        stop=stop,
        target_1=target_1,
        target_2=target_2,
        risk_per_unit=entry - stop,
        expected_holding_seconds=int(row.get("time_stop_minutes") or 1_440) * 60,
        expected_net_return=(
            Decimal(str(expected_net_bps)) / Decimal("10000")
            if expected_net_bps is not None
            else None
        ),
        expected_net_r=_decimal_or_none(economics.get("net_rr_target_2")),
        expected_mae=None,
        expected_mfe=None,
        current_regime=str(row.get("macro_regime") or "UNKNOWN"),
        btc_context={"parent": parent},
        market_breadth_context={"status": "NOT_AVAILABLE_AT_THIS_BOUNDARY"},
        multi_timeframe_alignment=_decimal_or_none(parent.get("timeframe_alignment_score")),
        multi_timeframe_conflict=bool(row.get("timeframe_disagreement")),
        liquidity={"microstructure_state": row.get("microstructure_state")},
        spread_bps=_decimal_or_none(realtime.get("spread_bps")),
        depth_eur=_decimal_or_none(realtime.get("ask_depth_eur_top_10")),
        slippage_estimate_bps=_decimal_or_none(realtime.get("estimated_buy_slippage_bps")),
        fee_estimate_bps=_decimal_or_none(fee_bps),
        theoretical_target_quantity=None,
        bitvavo_realizable_quantity=None,
        minimum_order_feasible=None,
        quantity_precision_valid=None,
        portfolio_fit="PENDING_CANONICAL_PORTFOLIO_TARGET",
        correlation_cluster=None,
        shariah_status="PENDING_RUNTIME_GATE",
        authority_status="PENDING_RUNTIME_GATE",
        data_health="PIT_FEATURE_SNAPSHOT_PRESENT" if row.get("feature_snapshot") else "PENDING",
        execution_health="PENDING_FRESH_QUOTE_AND_RECONCILIATION",
        retail_realizable=False,
        evidence_ids=tuple(
            str(value)
            for value in (row.get("opportunity_id"), row.get("episode_id"))
            if value
        ),
    )


def _utc_from(value: Any) -> datetime:
    if isinstance(value, datetime):
        return require_utc(value)
    return require_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _decimal_or_none(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


__all__ = [
    "ActiveSwingOpportunity",
    "CORE_DECISION_TIMEFRAMES",
    "PositionThesis",
    "SwingLifecycle",
    "TACTICAL_EXECUTION_TIMEFRAMES",
    "ThesisState",
    "TimeframeContract",
    "TimeframeObservation",
    "causal_asof_join",
    "normalize_live_opportunity",
]
