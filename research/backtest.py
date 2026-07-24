"""Deterministic multi-market, next-open, long-only crypto spot backtester."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings
from core.contracts import EligibilityStatus, Trade
from data.market_data import OHLCV_COLUMNS, timeframe_delta, validate_ohlcv
from research.strategies import Strategy
from research.trading_math import (
    bootstrap_expectancy,
    breakeven_win_rate,
    empirical_risk_of_ruin,
    maximum_drawdown,
    trade_statistics,
)
from utils.common import stable_hash


@dataclass(frozen=True)
class CostModel:
    fee_fraction: float = 0.0025
    slippage_bps: float = 8.0
    spread_bps: float = 5.0
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.fee_fraction <= 0.05:
            raise ValueError("fee_fraction must be between zero and 5%")
        if self.slippage_bps < 0 or self.spread_bps < 0:
            raise ValueError("slippage and spread cannot be negative")
        if self.multiplier < 1:
            raise ValueError("cost multiplier cannot be below one")

    @property
    def effective_fee(self) -> float:
        return self.fee_fraction * self.multiplier

    @property
    def buy_impact(self) -> float:
        return (
            self.slippage_bps * self.multiplier + self.spread_bps * self.multiplier / 2.0
        ) / 10_000.0

    @property
    def sell_impact(self) -> float:
        return self.buy_impact

    def buy_price(self, raw_price: float) -> float:
        return raw_price * (1.0 + self.buy_impact)

    def sell_price(self, raw_price: float) -> float:
        return raw_price * (1.0 - self.sell_impact)


@dataclass(frozen=True)
class MarketRules:
    minimum_order_amount: float = 0.0
    minimum_order_value_eur: float = 5.0
    amount_decimals: int = 8
    price_decimals: int = 8

    def floor_amount(self, amount: float) -> float:
        factor = 10**self.amount_decimals
        return math.floor(max(0.0, amount) * factor) / factor


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash_eur: float = 2_000.0
    risk_per_trade: float = 0.005
    maximum_total_open_risk: float = 0.02
    maximum_position_fraction: float = 0.25
    maximum_portfolio_exposure: float = 0.75
    reserve_cash_fraction: float = 0.10
    maximum_open_positions: int = 4
    maximum_trades_per_day: int = 6
    costs: CostModel = field(default_factory=CostModel)
    default_market_rules: MarketRules = field(default_factory=MarketRules)
    market_rules: dict[str, MarketRules] = field(default_factory=dict)
    liquidate_at_end: bool = True
    bootstrap_samples: int = 500
    monte_carlo_runs: int = 500
    random_seed: int = 42
    allow_review_required_research_only: bool = False

    def __post_init__(self) -> None:
        if self.initial_cash_eur <= 0:
            raise ValueError("initial cash must be positive")
        if not 0 < self.risk_per_trade <= 0.02:
            raise ValueError("risk_per_trade must be in (0, 2%]")
        if self.maximum_total_open_risk < self.risk_per_trade:
            raise ValueError("open risk cap cannot be below per-trade risk")
        if not 0 < self.maximum_position_fraction <= 1:
            raise ValueError("maximum_position_fraction must be in (0, 1]")
        if not 0 < self.maximum_portfolio_exposure <= 1:
            raise ValueError("maximum_portfolio_exposure must be in (0, 1]")
        if self.maximum_portfolio_exposure + self.reserve_cash_fraction > 1:
            raise ValueError("exposure plus reserve cash cannot exceed equity")
        if self.maximum_open_positions < 1 or self.maximum_trades_per_day < 1:
            raise ValueError("position and daily trade limits must be positive")
        if self.bootstrap_samples < 100 or self.monte_carlo_runs < 100:
            raise ValueError("bootstrap and Monte Carlo counts must be at least 100")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        initial_cash_eur: float = 2_000.0,
        stressed: bool = False,
        allow_review_required_research_only: bool = False,
    ) -> "BacktestConfig":
        return cls(
            initial_cash_eur=initial_cash_eur,
            risk_per_trade=settings.risk.risk_per_trade,
            maximum_total_open_risk=settings.risk.maximum_total_open_risk,
            maximum_position_fraction=settings.risk.maximum_position_fraction,
            maximum_portfolio_exposure=settings.risk.maximum_portfolio_exposure,
            reserve_cash_fraction=settings.risk.reserve_cash_fraction,
            maximum_trades_per_day=settings.risk.maximum_trades_per_day,
            costs=CostModel(
                fee_fraction=settings.costs.default_fee,
                slippage_bps=settings.costs.slippage_bps,
                spread_bps=settings.costs.spread_bps,
                multiplier=(
                    settings.costs.stressed_cost_multiplier if stressed else 1.0
                ),
            ),
            bootstrap_samples=settings.research.bootstrap_samples,
            monte_carlo_runs=settings.research.monte_carlo_runs,
            random_seed=settings.app.random_seed,
            allow_review_required_research_only=allow_review_required_research_only,
        )


@dataclass
class Position:
    market: str
    strategy_id: str
    quantity: float
    entry_price: float
    raw_entry_price: float
    entry_fee_remaining: float
    entry_at: pd.Timestamp
    signal_at: pd.Timestamp
    stop_price: float
    target_price: float
    trailing_distance: float
    trailing_stop: float | None
    initial_risk_per_unit: float
    initial_risk_remaining: float
    maximum_holding_bars: int | None
    bars_held: int = 0
    minimum_price_seen: float = math.inf
    maximum_price_seen: float = -math.inf


@dataclass(frozen=True)
class PendingAction:
    action: str
    signal_at: pd.Timestamp
    reason: str
    stop_distance: float | None = None
    target_distance: float | None = None
    trailing_distance: float = 0.0
    size_multiplier: float = 1.0
    maximum_holding_bars: int | None = None


@dataclass(frozen=True)
class BacktestOrder:
    order_id: str
    market: str
    side: str
    action: str
    signal_at: datetime | None
    executed_at: datetime
    raw_price: float
    fill_price: float
    quantity: float
    notional_eur: float
    fee_eur: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    initial_cash_eur: float
    ending_equity_eur: float
    equity_curve: pd.DataFrame
    trades: tuple[Trade, ...]
    orders: tuple[BacktestOrder, ...]
    metrics: dict[str, float | int | bool | None]
    integrity: dict[str, Any]


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.cash = config.initial_cash_eur
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, PendingAction] = {}
        self.orders: list[BacktestOrder] = []
        self.trades: list[Trade] = []
        self.trade_counts: Counter[Any] = Counter()
        self.last_prices: dict[str, float] = {}
        self.turnover_eur = 0.0
        self.total_costs_eur = 0.0

    def _reset(self) -> None:
        self.cash = self.config.initial_cash_eur
        self.positions.clear()
        self.pending.clear()
        self.orders.clear()
        self.trades.clear()
        self.trade_counts.clear()
        self.last_prices.clear()
        self.turnover_eur = 0.0
        self.total_costs_eur = 0.0

    def _equity(self, prices: dict[str, float] | None = None) -> float:
        selected = prices or self.last_prices
        return self.cash + sum(
            position.quantity * selected.get(market, position.entry_price)
            for market, position in self.positions.items()
        )

    def _exposure(self, prices: dict[str, float] | None = None) -> float:
        selected = prices or self.last_prices
        return sum(
            position.quantity * selected.get(market, position.entry_price)
            for market, position in self.positions.items()
        )

    def _open_risk(self) -> float:
        return sum(position.initial_risk_remaining for position in self.positions.values())

    def _rules(self, market: str) -> MarketRules:
        return self.config.market_rules.get(market, self.config.default_market_rules)

    def _record_rejection(
        self,
        *,
        market: str,
        action: PendingAction,
        executed_at: pd.Timestamp,
        raw_price: float,
        reason: str,
    ) -> None:
        self.orders.append(
            BacktestOrder(
                order_id=stable_hash(
                    {
                        "market": market,
                        "signal": action.signal_at,
                        "executed": executed_at,
                        "action": action.action,
                        "reason": reason,
                    },
                    length=24,
                ),
                market=market,
                side="BUY" if action.action == "ENTER" else "SELL",
                action=action.action,
                signal_at=action.signal_at.to_pydatetime(),
                executed_at=executed_at.to_pydatetime(),
                raw_price=raw_price,
                fill_price=raw_price,
                quantity=0.0,
                notional_eur=0.0,
                fee_eur=0.0,
                status="REJECTED",
                reason=reason,
            )
        )

    def _enter(
        self,
        market: str,
        action: PendingAction,
        timestamp: pd.Timestamp,
        raw_price: float,
        strategy_id: str,
    ) -> None:
        if market in self.positions:
            return
        if len(self.positions) >= self.config.maximum_open_positions:
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="MAXIMUM_OPEN_POSITIONS",
            )
            return
        day = timestamp.date()
        if self.trade_counts[day] >= self.config.maximum_trades_per_day:
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="MAXIMUM_TRADES_PER_DAY",
            )
            return
        if (
            action.stop_distance is None
            or action.target_distance is None
            or action.stop_distance <= 0
            or action.target_distance <= 0
        ):
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="INVALID_STOP_OR_TARGET",
            )
            return
        fill_price = self.config.costs.buy_price(raw_price)
        stop_price = raw_price - action.stop_distance
        target_price = raw_price + action.target_distance
        if stop_price <= 0:
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="INVALID_STOP_PRICE",
            )
            return
        equity = self._equity()
        fee = self.config.costs.effective_fee
        estimated_exit_price = self.config.costs.sell_price(stop_price)
        risk_per_unit = (fill_price - estimated_exit_price) + fee * (
            fill_price + estimated_exit_price
        )
        risk_budget = equity * self.config.risk_per_trade * action.size_multiplier
        units_by_risk = risk_budget / max(risk_per_unit, 1e-12)
        units_by_position = equity * self.config.maximum_position_fraction / fill_price
        remaining_exposure = max(
            0.0,
            equity * self.config.maximum_portfolio_exposure - self._exposure(),
        )
        units_by_exposure = remaining_exposure / fill_price
        reserve = equity * self.config.reserve_cash_fraction
        spendable_cash = max(0.0, self.cash - reserve)
        units_by_cash = spendable_cash / (fill_price * (1.0 + fee))
        remaining_risk = max(
            0.0,
            equity * self.config.maximum_total_open_risk - self._open_risk(),
        )
        units_by_open_risk = remaining_risk / max(risk_per_unit, 1e-12)
        rules = self._rules(market)
        units = rules.floor_amount(
            min(
                units_by_risk,
                units_by_position,
                units_by_exposure,
                units_by_cash,
                units_by_open_risk,
            )
        )
        notional = units * fill_price
        entry_fee = notional * fee
        if (
            units <= 0
            or units < rules.minimum_order_amount
            or notional < rules.minimum_order_value_eur
        ):
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="MINIMUM_ORDER_OR_RISK_CAP",
            )
            return
        total = notional + entry_fee
        if total > self.cash + 1e-9:
            self._record_rejection(
                market=market,
                action=action,
                executed_at=timestamp,
                raw_price=raw_price,
                reason="INSUFFICIENT_CASH",
            )
            return
        self.cash -= total
        initial_risk = units * risk_per_unit
        trailing_stop = (
            raw_price - action.trailing_distance
            if action.trailing_distance > 0
            else None
        )
        self.positions[market] = Position(
            market=market,
            strategy_id=strategy_id,
            quantity=units,
            entry_price=fill_price,
            raw_entry_price=raw_price,
            entry_fee_remaining=entry_fee,
            entry_at=timestamp,
            signal_at=action.signal_at,
            stop_price=stop_price,
            target_price=target_price,
            trailing_distance=action.trailing_distance,
            trailing_stop=trailing_stop,
            initial_risk_per_unit=risk_per_unit,
            initial_risk_remaining=initial_risk,
            maximum_holding_bars=action.maximum_holding_bars,
            minimum_price_seen=raw_price,
            maximum_price_seen=raw_price,
        )
        self.trade_counts[day] += 1
        self.turnover_eur += notional
        self.total_costs_eur += entry_fee + units * (fill_price - raw_price)
        self.orders.append(
            BacktestOrder(
                order_id=stable_hash(
                    {"market": market, "signal": action.signal_at, "entry": timestamp},
                    length=24,
                ),
                market=market,
                side="BUY",
                action="ENTER",
                signal_at=action.signal_at.to_pydatetime(),
                executed_at=timestamp.to_pydatetime(),
                raw_price=raw_price,
                fill_price=fill_price,
                quantity=units,
                notional_eur=notional,
                fee_eur=entry_fee,
                status="FILLED",
                reason=action.reason,
            )
        )

    def _exit(
        self,
        market: str,
        *,
        timestamp: pd.Timestamp,
        raw_price: float,
        reason: str,
        signal_at: pd.Timestamp | None,
        fraction: float = 1.0,
    ) -> None:
        position = self.positions.get(market)
        if position is None:
            return
        rules = self._rules(market)
        requested = position.quantity * min(1.0, max(0.0, fraction))
        quantity = (
            position.quantity
            if fraction >= 1.0
            else rules.floor_amount(requested)
        )
        if quantity <= 0:
            return
        if fraction < 1.0 and (
            quantity < rules.minimum_order_amount
            or quantity * raw_price < rules.minimum_order_value_eur
        ):
            return
        fill_price = self.config.costs.sell_price(raw_price)
        notional = quantity * fill_price
        fee = notional * self.config.costs.effective_fee
        proceeds = notional - fee
        proportion = quantity / position.quantity
        allocated_entry_fee = position.entry_fee_remaining * proportion
        entry_cost = quantity * position.entry_price + allocated_entry_fee
        pnl = proceeds - entry_cost
        risk = quantity * position.initial_risk_per_unit
        r_multiple = pnl / risk if risk > 0 else 0.0
        mae = (
            (position.minimum_price_seen - position.raw_entry_price)
            / max(position.raw_entry_price - position.stop_price, 1e-12)
        )
        mfe = (
            (position.maximum_price_seen - position.raw_entry_price)
            / max(position.raw_entry_price - position.stop_price, 1e-12)
        )
        trade_id = stable_hash(
            {
                "market": market,
                "entry": position.entry_at,
                "exit": timestamp,
                "quantity": quantity,
                "count": len(self.trades),
            },
            length=24,
        )
        self.trades.append(
            Trade(
                trade_id=trade_id,
                market=market,
                strategy_id=position.strategy_id,
                entry_at=position.entry_at.to_pydatetime(),
                exit_at=timestamp.to_pydatetime(),
                quantity=Decimal(str(quantity)),
                entry_price=Decimal(str(position.entry_price)),
                exit_price=Decimal(str(fill_price)),
                entry_fee_eur=Decimal(str(allocated_entry_fee)),
                exit_fee_eur=Decimal(str(fee)),
                net_pnl_eur=Decimal(str(pnl)),
                r_multiple=r_multiple,
                exit_reason=reason,
                mae_r=mae,
                mfe_r=mfe,
            )
        )
        self.cash += proceeds
        self.turnover_eur += notional
        self.total_costs_eur += fee + quantity * max(0.0, raw_price - fill_price)
        self.orders.append(
            BacktestOrder(
                order_id=stable_hash(
                    {
                        "market": market,
                        "signal": signal_at,
                        "exit": timestamp,
                        "reason": reason,
                        "count": len(self.orders),
                    },
                    length=24,
                ),
                market=market,
                side="SELL",
                action="REDUCE" if fraction < 1.0 else "EXIT",
                signal_at=signal_at.to_pydatetime() if signal_at is not None else None,
                executed_at=timestamp.to_pydatetime(),
                raw_price=raw_price,
                fill_price=fill_price,
                quantity=quantity,
                notional_eur=notional,
                fee_eur=fee,
                status="FILLED",
                reason=reason,
            )
        )
        if quantity >= position.quantity - 10 ** (-rules.amount_decimals):
            del self.positions[market]
        else:
            position.quantity -= quantity
            position.entry_fee_remaining -= allocated_entry_fee
            position.initial_risk_remaining -= risk

    def _manage_position(
        self,
        market: str,
        timestamp: pd.Timestamp,
        bar: pd.Series,
    ) -> None:
        position = self.positions.get(market)
        if position is None:
            return
        position.bars_held += 1
        position.minimum_price_seen = min(position.minimum_price_seen, float(bar["low"]))
        position.maximum_price_seen = max(position.maximum_price_seen, float(bar["high"]))
        active_stop = max(
            position.stop_price,
            position.trailing_stop if position.trailing_stop is not None else -math.inf,
        )
        raw_exit: float | None = None
        reason: str | None = None
        if float(bar["open"]) <= active_stop:
            raw_exit = float(bar["open"])
            reason = "GAP_STOP"
        elif float(bar["open"]) >= position.target_price:
            raw_exit = position.target_price
            reason = "GAP_TARGET_CONSERVATIVE"
        else:
            stop_hit = float(bar["low"]) <= active_stop
            target_hit = float(bar["high"]) >= position.target_price
            if stop_hit:
                raw_exit = active_stop
                reason = "STOP_LOSS" if not target_hit else "STOP_FIRST_SAME_BAR"
            elif target_hit:
                raw_exit = position.target_price
                reason = "TAKE_PROFIT"
        if raw_exit is not None:
            self._exit(
                market,
                timestamp=timestamp,
                raw_price=raw_exit,
                reason=reason or "RISK_EXIT",
                signal_at=None,
            )
            return
        if (
            position.maximum_holding_bars is not None
            and position.bars_held >= position.maximum_holding_bars
        ):
            self.pending[market] = PendingAction(
                action="EXIT",
                signal_at=timestamp,
                reason="TIME_STOP",
            )
            return
        if position.trailing_distance > 0:
            candidate = float(bar["close"]) - position.trailing_distance
            position.trailing_stop = max(
                position.trailing_stop or position.stop_price,
                candidate,
            )

    def run(
        self,
        data_by_market: dict[str, pd.DataFrame],
        strategy: Strategy,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> BacktestResult:
        self._reset()
        if not data_by_market:
            raise ValueError("at least one market dataset is required")
        frames: dict[str, pd.DataFrame] = {}
        outputs = {}
        for market, source in sorted(data_by_market.items()):
            normalized_market = market.upper().replace("/", "-").replace("_", "-")
            if self.settings is not None:
                eligibility = self.settings.shariah.eligibility(normalized_market)
                research_only_review = (
                    self.config.allow_review_required_research_only
                    and eligibility.status is EligibilityStatus.REVIEW_REQUIRED
                )
                if (
                    eligibility.status is not EligibilityStatus.ALLOWED
                    and not research_only_review
                ):
                    raise PermissionError(
                        f"backtest market is not ALLOWED: {normalized_market}"
                    )
            frame = validate_ohlcv(
                source.loc[:, list(OHLCV_COLUMNS)],
                closed_candles_only=False,
            )
            feature_knowability = source.attrs.get("feature_knowability")
            if not isinstance(feature_knowability, dict):
                raise ValueError(
                    f"feature knowability metadata missing for {normalized_market}"
                )
            prepared = source.copy()
            timeframe = source.attrs.get("timeframe")
            gap_flags = pd.Series(False, index=prepared.index)
            gap_integrity = {
                "timeframe": timeframe,
                "gap_events": 0,
                "missing_bars": 0,
                "largest_gap_bars": 0,
            }
            if timeframe:
                interval = pd.Timedelta(timeframe_delta(str(timeframe)))
                elapsed = prepared.index.to_series().diff()
                missing = (
                    (elapsed / interval) - 1.0
                ).fillna(0.0).clip(lower=0.0)
                gap_flags = missing > 0.0
                gap_integrity = {
                    "timeframe": timeframe,
                    "gap_events": int(gap_flags.sum()),
                    "missing_bars": int(missing.sum()),
                    "largest_gap_bars": int(missing.max()),
                }
            prepared["_data_gap_flag"] = gap_flags.astype(bool)
            prepared.attrs.update(source.attrs)
            prepared.attrs["gap_integrity"] = gap_integrity
            frames[normalized_market] = prepared
            outputs[normalized_market] = strategy.generate(source, parameters)

        timeline = sorted(set().union(*(frame.index for frame in frames.values())))
        equity_rows: list[dict[str, Any]] = []
        for timestamp in timeline:
            ts = pd.Timestamp(timestamp)
            for market in sorted(frames):
                frame = frames[market]
                if ts not in frame.index:
                    continue
                bar = frame.loc[ts]
                self.last_prices[market] = float(bar["open"])
                pending = self.pending.pop(market, None)
                if (
                    pending is not None
                    and pending.action == "ENTER"
                    and bool(bar.get("_data_gap_flag", False))
                ):
                    pending = None
                if pending is not None:
                    if pending.signal_at >= ts:
                        raise AssertionError("next-open execution invariant violated")
                    if pending.action == "ENTER":
                        self._enter(
                            market,
                            pending,
                            ts,
                            float(bar["open"]),
                            strategy.strategy_id,
                        )
                    elif pending.action == "EXIT":
                        self._exit(
                            market,
                            timestamp=ts,
                            raw_price=float(bar["open"]),
                            reason=pending.reason,
                            signal_at=pending.signal_at,
                        )
                    elif pending.action == "REDUCE":
                        self._exit(
                            market,
                            timestamp=ts,
                            raw_price=float(bar["open"]),
                            reason=pending.reason,
                            signal_at=pending.signal_at,
                            fraction=0.5,
                        )
                self._manage_position(market, ts, bar)
                self.last_prices[market] = float(bar["close"])
                output = outputs[market]
                if market in self.positions:
                    if bool(output.exit.loc[ts]):
                        self.pending[market] = PendingAction(
                            action="EXIT",
                            signal_at=ts,
                            reason=output.exit_reason,
                        )
                    elif bool(output.reduce.loc[ts]):
                        self.pending[market] = PendingAction(
                            action="REDUCE",
                            signal_at=ts,
                            reason="STRATEGY_REDUCE",
                        )
                elif (
                    bool(output.entry.loc[ts])
                    and not bool(output.avoid.loc[ts])
                    and not bool(bar.get("_data_gap_flag", False))
                ):
                    stop_distance = float(output.stop_distance.loc[ts])
                    target_distance = float(output.target_distance.loc[ts])
                    trailing_distance = float(output.trailing_distance.loc[ts])
                    size_multiplier = float(output.size_multiplier.loc[ts])
                    if all(
                        np.isfinite(value)
                        for value in (
                            stop_distance,
                            target_distance,
                            trailing_distance,
                            size_multiplier,
                        )
                    ):
                        self.pending[market] = PendingAction(
                            action="ENTER",
                            signal_at=ts,
                            reason=output.entry_reason,
                            stop_distance=stop_distance,
                            target_distance=target_distance,
                            trailing_distance=trailing_distance,
                            size_multiplier=size_multiplier,
                            maximum_holding_bars=output.maximum_holding_bars,
                        )
            equity = self._equity()
            exposure = self._exposure()
            equity_rows.append(
                {
                    "timestamp": ts,
                    "equity": equity,
                    "cash": self.cash,
                    "exposure_eur": exposure,
                    "exposure_fraction": exposure / equity if equity > 0 else 0.0,
                    "open_positions": len(self.positions),
                    "open_risk_eur": self._open_risk(),
                }
            )

        if self.config.liquidate_at_end:
            for market in sorted(tuple(self.positions)):
                frame = frames[market]
                timestamp = frame.index[-1]
                self._exit(
                    market,
                    timestamp=timestamp,
                    raw_price=float(frame["close"].iloc[-1]),
                    reason="END_OF_DATA",
                    signal_at=None,
                )
            if equity_rows:
                equity_rows[-1].update(
                    {
                        "equity": self.cash,
                        "cash": self.cash,
                        "exposure_eur": 0.0,
                        "exposure_fraction": 0.0,
                        "open_positions": 0,
                        "open_risk_eur": 0.0,
                    }
                )
        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        metrics = calculate_metrics(
            equity_curve,
            self.trades,
            initial_cash=self.config.initial_cash_eur,
            turnover_eur=self.turnover_eur,
            transaction_costs_eur=self.total_costs_eur,
            bootstrap_samples=self.config.bootstrap_samples,
            monte_carlo_runs=self.config.monte_carlo_runs,
            risk_fraction=self.config.risk_per_trade,
            seed=self.config.random_seed,
        )
        integrity = self._integrity(frames)
        return BacktestResult(
            strategy_id=strategy.strategy_id,
            initial_cash_eur=self.config.initial_cash_eur,
            ending_equity_eur=float(equity_curve["equity"].iloc[-1]),
            equity_curve=equity_curve,
            trades=tuple(self.trades),
            orders=tuple(self.orders),
            metrics=metrics,
            integrity=integrity,
        )

    def _integrity(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        negative_units = any(position.quantity < -1e-12 for position in self.positions.values())
        negative_cash = self.cash < -1e-8
        invalid_order_timing = sum(
            order.signal_at is not None and order.executed_at <= order.signal_at
            for order in self.orders
        )
        data_errors: dict[str, str] = {}
        closed_candle_integrity = True
        provider_integrity = True
        provider_sources: dict[str, Any] = {}
        for market, frame in frames.items():
            try:
                validate_ohlcv(
                    frame,
                    timeframe=frame.attrs.get("timeframe"),
                    closed_candles_only=True,
                )
            except Exception as exc:
                data_errors[market] = type(exc).__name__
                closed_candle_integrity = False
            provenance = frame.attrs.get("data_provenance")
            if provenance:
                provider_sources[market] = provenance
                provider_integrity &= provenance.get("source_type") in {
                    "REAL_PROVIDER_DATA",
                    "SYNTHETIC_SMOKE",
                }
            else:
                provider_integrity = False
        intelligence_states = {
            market: frame.attrs.get("intelligence_timing_integrity", "NOT_USED")
            for market, frame in frames.items()
        }
        benchmark_states = {
            market: frame.attrs.get("benchmark_staleness_integrity", "NOT_USED")
            for market, frame in frames.items()
        }
        higher_timeframe_states = {
            market: frame.attrs.get("higher_timeframe_integrity", "NOT_USED")
            for market, frame in frames.items()
        }
        gap_integrity = {
            market: dict(frame.attrs.get("gap_integrity") or {})
            for market, frame in frames.items()
        }
        return {
            "valid_data": not data_errors,
            "data_validation_errors": data_errors,
            "valid_intelligence_timing": all(
                state in {"PASSED", "NOT_USED"} for state in intelligence_states.values()
            ),
            "intelligence_timing_integrity": intelligence_states,
            "no_lookahead": invalid_order_timing == 0,
            "no_repainting": all(
                all(
                    details.get("lookahead_safe") and not details.get("repaint")
                    for details in frame.attrs["feature_knowability"].values()
                )
                for frame in frames.values()
            ),
            "next_open_execution": invalid_order_timing == 0,
            "provider_data_integrity": provider_integrity,
            "provider_provenance": provider_sources,
            "closed_candle_integrity": closed_candle_integrity,
            "benchmark_staleness_integrity": benchmark_states,
            "higher_timeframe_integrity": higher_timeframe_states,
            "gap_integrity": gap_integrity,
            "gap_entry_blocking": True,
            "long_only_spot": True,
            "negative_asset_balance": negative_units,
            "negative_cash_balance": negative_cash,
            "invalid_order_timing_count": invalid_order_timing,
        }


def _effective_sample_size(values: np.ndarray) -> float:
    count = len(values)
    if count < 3 or np.std(values) == 0:
        return float(count)
    maximum_lag = min(count // 4, int(math.sqrt(count)))
    correlations = []
    for lag in range(1, maximum_lag + 1):
        correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
        if not np.isfinite(correlation) or correlation <= 0:
            break
        correlations.append(correlation)
    return max(1.0, count / (1.0 + 2.0 * sum(correlations)))


def _drawdown_duration(equity: pd.Series) -> tuple[int, float]:
    underwater = equity < equity.cummax()
    maximum = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum, float(underwater.mean())


def _monte_carlo_drawdown_probability(
    r_multiples: np.ndarray,
    *,
    risk_fraction: float,
    simulations: int,
    threshold: float,
    seed: int,
) -> float:
    if len(r_multiples) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    block_size = min(max(1, int(round(len(r_multiples) ** (1 / 3)))), len(r_multiples))
    starts = np.arange(0, len(r_multiples) - block_size + 1)
    for _ in range(simulations):
        blocks = math.ceil(len(r_multiples) / block_size)
        chosen = rng.choice(starts, size=blocks, replace=True)
        sample = np.concatenate(
            [r_multiples[start : start + block_size] for start in chosen]
        )[: len(r_multiples)]
        curve = np.cumprod(np.r_[1.0, np.maximum(0.0, 1.0 + risk_fraction * sample)])
        exceed += maximum_drawdown(curve) >= threshold
    return exceed / simulations


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    *,
    initial_cash: float,
    turnover_eur: float,
    transaction_costs_eur: float,
    bootstrap_samples: int,
    monte_carlo_runs: int,
    risk_fraction: float,
    seed: int,
) -> dict[str, float | int | bool | None]:
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elapsed_seconds = max(
        1.0,
        (equity.index[-1] - equity.index[0]).total_seconds(),
    )
    years = elapsed_seconds / (365.25 * 86_400)
    ending = float(equity.iloc[-1])
    net_return = ending / initial_cash - 1.0
    cagr = (
        (ending / initial_cash) ** (1.0 / years) - 1.0
        if ending > 0 and years > 0
        else -1.0
    )
    median_seconds = (
        equity.index.to_series().diff().dropna().dt.total_seconds().median()
        if len(equity) > 1
        else 86_400.0
    )
    periods_per_year = 365.25 * 86_400 / max(1.0, float(median_seconds))
    standard = float(returns.std(ddof=1))
    downside = float(returns[returns < 0].std(ddof=1))
    sharpe = float(returns.mean() / standard * math.sqrt(periods_per_year)) if standard > 0 else 0.0
    sortino = float(returns.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0
    max_drawdown = maximum_drawdown(equity.to_numpy())
    drawdown_duration, time_under_water = _drawdown_duration(equity)
    calmar = cagr / max_drawdown if max_drawdown > 0 else 0.0
    losses = -returns[returns < 0]
    gains = returns[returns > 0]
    omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else (float("inf") if gains.sum() > 0 else 0.0)
    drawdowns = (equity / equity.cummax() - 1.0).clip(upper=0)
    ulcer = float(np.sqrt(np.mean(np.square(drawdowns))))

    pnl = np.array([float(trade.net_pnl_eur) for trade in trades], dtype=float)
    r_values = np.array([trade.r_multiple for trade in trades], dtype=float)
    if len(trades):
        statistics = trade_statistics(r_values)
        average_net_win = float(pnl[pnl > 0].mean()) if np.any(pnl > 0) else 0.0
        average_net_loss = abs(float(pnl[pnl < 0].mean())) if np.any(pnl < 0) else 0.0
        win_rate = float(np.mean(pnl > 0))
        expectancy_eur = float(np.mean(pnl))
        expectancy_fraction = expectancy_eur / initial_cash
        payoff = (
            average_net_win / average_net_loss
            if average_net_loss > 0
            else (float("inf") if average_net_win > 0 else 0.0)
        )
        break_even = (
            breakeven_win_rate(average_net_win, average_net_loss)
            if average_net_loss > 0
            else 0.0
        )
        bootstrap = bootstrap_expectancy(
            r_values,
            bootstrap_samples=max(100, bootstrap_samples),
            block_size=min(max(1, int(round(len(r_values) ** (1 / 3)))), len(r_values)),
            seed=seed,
        )
        simulation = empirical_risk_of_ruin(
            r_values,
            risk_fraction=risk_fraction,
            initial_equity=initial_cash,
            trades_per_simulation=max(len(r_values), 100),
            simulations=max(100, monte_carlo_runs),
            ruin_drawdown=0.50,
            block_size=min(max(1, int(round(len(r_values) ** (1 / 3)))), len(r_values)),
            seed=seed,
        )
        drawdown_probabilities = {
            threshold: _monte_carlo_drawdown_probability(
                r_values,
                risk_fraction=risk_fraction,
                simulations=max(100, monte_carlo_runs),
                threshold=threshold,
                seed=seed + int(threshold * 100),
            )
            for threshold in (0.10, 0.20, 0.30, 0.50)
        }
        symbol_profit = defaultdict(float)
        for trade in trades:
            symbol_profit[trade.market] += float(trade.net_pnl_eur)
        positive_total = sum(max(0.0, value) for value in symbol_profit.values())
        concentration = (
            max((max(0.0, value) for value in symbol_profit.values()), default=0.0)
            / positive_total
            if positive_total > 0
            else 1.0
        )
        mae = float(np.mean([trade.mae_r or 0.0 for trade in trades]))
        mfe = float(np.mean([trade.mfe_r or 0.0 for trade in trades]))
    else:
        statistics = None
        win_rate = expectancy_eur = expectancy_fraction = average_net_win = average_net_loss = 0.0
        payoff = break_even = 0.0
        bootstrap = simulation = None
        drawdown_probabilities = {
            0.10: 1.0,
            0.20: 1.0,
            0.30: 1.0,
            0.50: 1.0,
        }
        concentration = 1.0
        mae = mfe = 0.0

    return {
        "net_return": net_return,
        "cagr": cagr,
        "maximum_drawdown": max_drawdown,
        "drawdown_duration_bars": drawdown_duration,
        "time_under_water": time_under_water,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
        "ulcer_index": ulcer,
        "trade_count": len(trades),
        "win_rate": win_rate,
        "profit_factor": statistics.profit_factor if statistics else 0.0,
        "net_expectancy_r": statistics.expectancy_r if statistics else 0.0,
        "net_expectancy_eur": expectancy_eur,
        "net_expectancy_equity_fraction": expectancy_fraction,
        "average_net_win_eur": average_net_win,
        "average_net_loss_eur": average_net_loss,
        "payoff_ratio": payoff,
        "break_even_win_rate": break_even,
        "effective_sample_size": _effective_sample_size(r_values),
        "average_exposure": float(equity_curve["exposure_fraction"].mean()),
        "turnover": turnover_eur / initial_cash,
        "average_mae_r": mae,
        "average_mfe_r": mfe,
        "transaction_costs_eur": transaction_costs_eur,
        "maximum_losing_streak": statistics.max_consecutive_losses if statistics else 0,
        "probability_of_loss": (
            1.0 - bootstrap.probability_expectancy_positive if bootstrap else 1.0
        ),
        "risk_of_ruin": simulation.risk_of_ruin if simulation else 1.0,
        "monte_carlo_p95_drawdown": simulation.p95_max_drawdown if simulation else 1.0,
        "probability_of_10pct_drawdown": drawdown_probabilities[0.10],
        "probability_of_20pct_drawdown": drawdown_probabilities[0.20],
        "probability_of_30pct_drawdown": drawdown_probabilities[0.30],
        "probability_of_50pct_drawdown": drawdown_probabilities[0.50],
        "symbol_profit_concentration": concentration,
    }


__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestOrder",
    "BacktestResult",
    "CostModel",
    "MarketRules",
    "calculate_metrics",
]
